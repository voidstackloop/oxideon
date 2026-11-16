//! RWKV-6 ("Finch") layers, computed the way llama.cpp's `build_rwkv6_time_mix` / `build_rwkv6_channel_mix` do.
//!
//! A layer is `x + time_mix(ln1(x))`, then `+ channel_mix(ln2(.))`. The time mix interpolates every token with its
//! predecessor (a data-dependent "token shift"), projects `r k v g` and a per-channel decay `w = exp(-exp(.))`, and
//! runs the WKV recurrence per head on a `[head, head]` state:
//! `out = r (first * k v^T + S)`, `S = diag(w) S + k v^T`. The recurrence is sequential and runs on the host in `f32`;
//! the projections stay tensor matmuls. The decoding cache of a layer is `(state [b, heads, hs, hs], shift [b, 2, d])`
//! where `shift` holds the last normed token of the time mix and of the channel mix.
//!
//! RWKV-7 ("Goose", [`Rwkv7Layer`]) keeps the same layout but replaces the recurrence by a generalized delta rule
//! `S = S diag(w) + (S a) b^T + v k^T` with low-rank in-context learning rates and a value residual from layer 0.

use super::gguf_model::{Norm, lin_last};
use crate::error::{Error, Result};
use crate::tensor::Tensor;

pub struct Rwkv6Layer {
    /// LayerNorms (with bias) before the time mix and the channel mix.
    pub attn_norm: Norm,
    pub ffn_norm: Norm,
    /// `[5 * extra, d]` and `[5, d, extra]`: the low-rank data-dependent interpolation factors for `w k v r g`.
    pub mix_w1: Tensor,
    pub mix_w2: Tensor,
    pub lerp_x: Tensor,
    /// `[5, d]`: static interpolation offsets in the order `w k v r g`.
    pub lerp: Tensor,
    /// `[heads, head_size]`: the bonus applied to the current token.
    pub first: Tensor,
    pub decay: Tensor,
    pub decay_w1: Tensor,
    pub decay_w2: Tensor,
    pub key: Tensor,
    pub value: Tensor,
    pub receptance: Tensor,
    pub gate: Tensor,
    /// Per-head group norm: weight and bias `[d]`, epsilon `64e-5`.
    pub ln_weight: Tensor,
    pub ln_bias: Tensor,
    pub output: Tensor,
    pub cm_lerp_k: Tensor,
    pub cm_lerp_r: Tensor,
    pub cm_key: Tensor,
    pub cm_value: Tensor,
    pub cm_receptance: Tensor,
    pub head_size: usize,
    /// Halve the output of this layer (`rescale_every_n_layers`).
    pub rescale: bool,
}

/// `x_prev[t] = x[t - 1]` with `first` standing in for `x[-1]`: `[b, s, d]` from `[b, 1, d]` and `[b, s, d]`.
fn shifted(first: &Tensor, x: &Tensor) -> Result<Tensor> {
    let s = x.shape()[1];
    if s == 1 {
        return Ok(first.clone());
    }
    Tensor::cat(&[first.clone(), x.narrow(1, 0, s - 1)?.contiguous()?], 1)
}

impl Rwkv6Layer {
    pub fn forward(&self, x: &Tensor, cache: &mut Option<(Tensor, Tensor)>) -> Result<Tensor> {
        let (b, s, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let hs = self.head_size;
        if hs == 0 || d % hs != 0 {
            return Err(Error::Msg(format!(
                "RWKV head size {hs} does not divide the model width {d}"
            )));
        }
        let heads = d / hs;
        let (state, shift) = match cache.take() {
            Some(c) => c,
            None => (
                Tensor::zeros_f32(&[b, heads, hs, hs])?,
                Tensor::zeros_f32(&[b, 2, d])?,
            ),
        };
        let shift_att = shift.narrow(1, 0, 1)?.contiguous()?;
        let shift_ffn = shift.narrow(1, 1, 1)?.contiguous()?;

        // --- time mix
        let att = self.attn_norm.forward(x)?;
        let sx = shifted(&shift_att, &att)?.sub(&att)?;
        let xxx = sx.mul(&self.lerp_x)?.add(&att)?;
        let extra = self.mix_w1.shape()[0] / 5;
        let low = lin_last(&xxx, &self.mix_w1)?.tanh()?.reshape(&[b, s, 5, extra])?;
        let mixed = |j: usize| -> Result<Tensor> {
            let w2 = self.mix_w2.narrow(0, j, 1)?.contiguous()?.reshape(&[d, extra])?;
            let lerp = self.lerp.narrow(0, j, 1)?.contiguous()?.reshape(&[d])?;
            let m = lin_last(&low.narrow(2, j, 1)?.contiguous()?.reshape(&[b, s, extra])?, &w2)?;
            m.add(&lerp)?.mul(&sx)?.add(&att)
        };
        let (xw, xk, xv, xr, xg) = (mixed(0)?, mixed(1)?, mixed(2)?, mixed(3)?, mixed(4)?);
        let r = lin_last(&xr, &self.receptance)?;
        let k = lin_last(&xk, &self.key)?;
        let v = lin_last(&xv, &self.value)?;
        let g = lin_last(&xg, &self.gate)?.silu()?;
        let w = lin_last(&lin_last(&xw, &self.decay_w1)?.tanh()?, &self.decay_w2)?
            .add(&self.decay)?;
        let w = w.exp()?.neg()?.exp()?;

        // WKV6 recurrence on the host.
        let (r, k, v, w) = (
            r.to_vec::<f32>()?,
            k.to_vec::<f32>()?,
            v.to_vec::<f32>()?,
            w.to_vec::<f32>()?,
        );
        let first = self.first.to_vec::<f32>()?;
        let mut st = state.to_vec::<f32>()?; // [b, heads, i, j]
        let mut out = vec![0f32; b * s * d];
        for bi in 0..b {
            for t in 0..s {
                let row = (bi * s + t) * d;
                for h in 0..heads {
                    let sbase = (bi * heads + h) * hs * hs;
                    for i in 0..hs {
                        let (ki, ri, wi, fi) = (
                            k[row + h * hs + i],
                            r[row + h * hs + i],
                            w[row + h * hs + i],
                            first[h * hs + i],
                        );
                        for j in 0..hs {
                            let kv = ki * v[row + h * hs + j];
                            let s_ij = &mut st[sbase + i * hs + j];
                            out[row + h * hs + j] += ri * (fi * kv + *s_ij);
                            *s_ij = *s_ij * wi + kv;
                        }
                    }
                }
            }
        }
        let new_state = Tensor::from_slice_on(&st, &[b, heads, hs, hs], &x.device())?;
        let out = Tensor::from_slice_on(&out, &[b, s, heads, hs], &x.device())?;
        // Group norm per head (no affine inside), then the per-channel affine and the gate.
        let mean = out.mean_axis(-1, true)?;
        let centered = out.sub(&mean)?;
        let var = centered.square()?.mean_axis(-1, true)?;
        let normed = centered
            .mul(&var.add_scalar(64e-5)?.rsqrt()?)?
            .reshape(&[b, s, d])?;
        let cur = normed.mul(&self.ln_weight)?.add(&self.ln_bias)?.mul(&g)?;
        let x = x.add(&lin_last(&cur, &self.output)?)?;

        // --- channel mix
        let ffn = self.ffn_norm.forward(&x)?;
        let sx = shifted(&shift_ffn, &ffn)?.sub(&ffn)?;
        let xk = sx.mul(&self.cm_lerp_k)?.add(&ffn)?;
        let xr = sx.mul(&self.cm_lerp_r)?.add(&ffn)?;
        let r = lin_last(&xr, &self.cm_receptance)?.sigmoid()?;
        let k = lin_last(&xk, &self.cm_key)?.relu()?.square()?;
        let cur = r.mul(&lin_last(&k, &self.cm_value)?)?;
        let mut y = x.add(&cur)?;
        if self.rescale {
            y = y.mul_scalar(0.5)?;
        }

        let last = |t: &Tensor| -> Result<Tensor> { t.narrow(1, s - 1, 1)?.contiguous() };
        let new_shift = Tensor::cat(&[last(&att)?, last(&ffn)?], 1)?;
        *cache = Some((new_state, new_shift));
        Ok(y)
    }
}

/// Per-head group norm without affine part (epsilon `64e-5`) of `[b, s, heads, hs]`, flattened to `[b, s, d]`.
fn head_norm(x: &Tensor, b: usize, s: usize, d: usize) -> Result<Tensor> {
    let mean = x.mean_axis(-1, true)?;
    let centered = x.sub(&mean)?;
    let var = centered.square()?.mean_axis(-1, true)?;
    centered.mul(&var.add_scalar(64e-5)?.rsqrt()?)?.reshape(&[b, s, d])
}

/// The feed-forward half of an RWKV-7 layer.
pub enum RwkvFfn {
    /// RWKV-7's channel mix: ReLU² of an interpolated key, no receptance. Needs a second token-shift slot.
    ChannelMix {
        norm: Norm,
        lerp_k: Tensor,
        key: Tensor,
        value: Tensor,
    },
    /// ARWKV: a SwiGLU FFN behind an RMS norm, no token shift.
    SwiGlu {
        norm: Norm,
        gate: Tensor,
        up: Tensor,
        down: Tensor,
    },
}

/// One RWKV-7 layer. The time mix has no receptance gate on the channel mix and no rescaling; layer 0 stores its value
/// vectors and every later layer blends them back in through a low-rank sigmoid gate.
pub struct Rwkv7Layer {
    pub attn_norm: Norm,
    pub ffn: RwkvFfn,
    /// `[6, d]`: static interpolation offsets in the order `r w k v a g`.
    pub lerp: Tensor,
    pub w0: Tensor,
    pub w1: Tensor,
    pub w2: Tensor,
    pub a0: Tensor,
    pub a1: Tensor,
    pub a2: Tensor,
    pub v0: Tensor,
    pub v1: Tensor,
    pub v2: Tensor,
    /// Output gate `g2 sigmoid(g1 x)`, absent in some checkpoints.
    pub gate: Option<(Tensor, Tensor)>,
    pub k_k: Tensor,
    pub k_a: Tensor,
    pub r_k: Tensor,
    pub key: Tensor,
    pub value: Tensor,
    pub receptance: Tensor,
    /// Group-norm affine part; without it the normed output is used as is.
    pub ln: Option<(Tensor, Tensor)>,
    pub output: Tensor,
    pub head_size: usize,
    /// Layer 0 produces `v_first`; the others blend it in.
    pub first: bool,
}

impl Rwkv7Layer {
    /// `v_first` carries layer 0's value vectors to the later layers of the same forward pass.
    pub fn forward(
        &self,
        x: &Tensor,
        cache: &mut Option<(Tensor, Tensor)>,
        v_first: &mut Option<Tensor>,
    ) -> Result<Tensor> {
        let (b, s, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let hs = self.head_size;
        if hs == 0 || d % hs != 0 {
            return Err(Error::Msg(format!(
                "RWKV head size {hs} does not divide the model width {d}"
            )));
        }
        let heads = d / hs;
        let slots = match self.ffn {
            RwkvFfn::ChannelMix { .. } => 2,
            RwkvFfn::SwiGlu { .. } => 1,
        };
        let (state, shift) = match cache.take() {
            Some(c) => c,
            None => (
                Tensor::zeros_f32(&[b, heads, hs, hs])?,
                Tensor::zeros_f32(&[b, slots, d])?,
            ),
        };
        let shift_att = shift.narrow(1, 0, 1)?.contiguous()?;

        // --- time mix
        let att = self.attn_norm.forward(x)?;
        let sx = shifted(&shift_att, &att)?.sub(&att)?;
        let mixed = |j: usize| -> Result<Tensor> {
            let lerp = self.lerp.narrow(0, j, 1)?.contiguous()?.reshape(&[d])?;
            sx.mul(&lerp)?.add(&att)
        };
        let (xr, xw, xk, xv, xa) = (mixed(0)?, mixed(1)?, mixed(2)?, mixed(3)?, mixed(4)?);
        let r = lin_last(&xr, &self.receptance)?;
        let w = lin_last(&lin_last(&xw, &self.w1)?.tanh()?, &self.w2)?.add(&self.w0)?;
        let w = w.sigmoid()?.mul_scalar(-0.606_531)?.exp()?;
        let k = lin_last(&xk, &self.key)?;
        let mut v = lin_last(&xv, &self.value)?;
        match v_first.as_ref() {
            Some(first) if !self.first => {
                let gate = lin_last(&lin_last(&xv, &self.v1)?, &self.v2)?
                    .add(&self.v0)?
                    .sigmoid()?;
                v = v.add(&first.sub(&v)?.mul(&gate)?)?;
            }
            _ => *v_first = Some(v.clone()),
        }
        let g = match &self.gate {
            Some((g1, g2)) => Some(lin_last(&lin_last(&mixed(5)?, g1)?.sigmoid()?, g2)?),
            None => None,
        };
        let a = lin_last(&lin_last(&xa, &self.a1)?, &self.a2)?
            .add(&self.a0)?
            .sigmoid()?;

        // kk = l2norm(k * k_k) per head; k += a * (k * k_a) - k * k_a
        let kk = k.mul(&self.k_k)?.reshape(&[b, s, heads, hs])?;
        let norm = kk.square()?.sum_axis(-1, true)?.sqrt()?.clamp(1e-12, f64::MAX)?;
        let kk = kk.div(&norm)?.reshape(&[b, s, d])?;
        let ka = k.mul(&self.k_a)?;
        let k = k.add(&a.mul(&ka)?.sub(&ka)?)?;

        // WKV7 on the host: for each value row i, sa = <-kk, S_i>, S_i = S_i * w + v_i k + sa * (kk * a), out_i = <S_i, r>.
        let to_vec = |t: &Tensor| t.to_vec::<f32>();
        let (rv, wv, kv, vv) = (to_vec(&r)?, to_vec(&w)?, to_vec(&k)?, to_vec(&v)?);
        let (av, bv) = (to_vec(&kk.neg()?)?, to_vec(&kk.mul(&a)?)?);
        let mut st = state.to_vec::<f32>()?; // [b, heads, i, j]
        let mut out = vec![0f32; b * s * d];
        for bi in 0..b {
            for t in 0..s {
                let row = (bi * s + t) * d;
                for h in 0..heads {
                    let sbase = (bi * heads + h) * hs * hs;
                    let head = row + h * hs;
                    for i in 0..hs {
                        let si = &mut st[sbase + i * hs..sbase + (i + 1) * hs];
                        let sa: f32 = (0..hs).map(|j| av[head + j] * si[j]).sum();
                        let mut acc = 0f32;
                        for j in 0..hs {
                            si[j] = si[j] * wv[head + j] + vv[head + i] * kv[head + j] + sa * bv[head + j];
                            acc += si[j] * rv[head + j];
                        }
                        out[head + i] = acc;
                    }
                }
            }
        }
        let new_state = Tensor::from_slice_on(&st, &[b, heads, hs, hs], &x.device())?;
        let out = Tensor::from_slice_on(&out, &[b, s, heads, hs], &x.device())?;
        let mut cur = match &self.ln {
            Some((lw, lb)) => head_norm(&out, b, s, d)?.mul(lw)?.add(lb)?,
            None => out.reshape(&[b, s, d])?,
        };
        // Bonus for the current token: v * sum_j (k_j r_j r_k_j) per head.
        let rk = k
            .mul(&r)?
            .mul(&self.r_k)?
            .reshape(&[b, s, heads, hs])?
            .sum_axis(-1, true)?;
        cur = cur.add(&v.reshape(&[b, s, heads, hs])?.mul(&rk)?.reshape(&[b, s, d])?)?;
        if let Some(g) = g {
            cur = cur.mul(&g)?;
        }
        let x = x.add(&lin_last(&cur, &self.output)?)?;

        let last = |t: &Tensor| -> Result<Tensor> { t.narrow(1, s - 1, 1)?.contiguous() };
        let (y, new_shift) = match &self.ffn {
            RwkvFfn::ChannelMix {
                norm,
                lerp_k,
                key,
                value,
            } => {
                // channel mix (no receptance)
                let shift_ffn = shift.narrow(1, 1, 1)?.contiguous()?;
                let ffn = norm.forward(&x)?;
                let sx = shifted(&shift_ffn, &ffn)?.sub(&ffn)?;
                let xk = sx.mul(lerp_k)?.add(&ffn)?;
                let k = lin_last(&xk, key)?.relu()?.square()?;
                (
                    x.add(&lin_last(&k, value)?)?,
                    Tensor::cat(&[last(&att)?, last(&ffn)?], 1)?,
                )
            }
            RwkvFfn::SwiGlu {
                norm,
                gate,
                up,
                down,
            } => {
                let ffn = norm.forward(&x)?;
                let inner = lin_last(&ffn, gate)?.silu()?.mul(&lin_last(&ffn, up)?)?;
                (x.add(&lin_last(&inner, down)?)?, last(&att)?)
            }
        };
        *cache = Some((new_state, new_shift));
        Ok(y)
    }
}

/// A projection with an optional bias.
pub struct BiasedProj {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl BiasedProj {
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let y = lin_last(x, &self.weight)?;
        match &self.bias {
            Some(b) => y.add(b),
            None => Ok(y),
        }
    }
}

/// One QRWKV6 layer (`rwkv6qwen2`): RWKV-6's data-dependent interpolation and decay feeding a gated linear attention
/// recurrence (`S = diag(w) S + k v^T`, `out = scale * r S`, with `k` scaled by `1 - w`), grouped key / value heads, a
/// sigmoid output gate and no group norm; RMS norms and a SwiGLU feed-forward. The cache is
/// `(state [b, heads, hs, hs], shift [b, 1, d])`.
pub struct Qrwkv6Layer {
    pub attn_norm: Norm,
    pub ffn_norm: Norm,
    pub mix_w1: Tensor,
    pub mix_w2: Tensor,
    pub lerp_x: Tensor,
    /// `[5, d]` in the order `w k v r g`.
    pub lerp: Tensor,
    pub decay: Tensor,
    pub decay_w1: Tensor,
    pub decay_w2: Tensor,
    pub key: BiasedProj,
    pub value: BiasedProj,
    pub receptance: BiasedProj,
    pub gate: Tensor,
    pub output: Tensor,
    pub ffn_gate: Tensor,
    pub ffn_up: Tensor,
    pub ffn_down: Tensor,
    pub head_size: usize,
}

impl Qrwkv6Layer {
    pub fn forward(&self, x: &Tensor, cache: &mut Option<(Tensor, Tensor)>) -> Result<Tensor> {
        let (b, s, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let hs = self.head_size;
        if hs == 0 || d % hs != 0 {
            return Err(Error::Msg(format!(
                "RWKV head size {hs} does not divide the model width {d}"
            )));
        }
        let heads = d / hs;
        let kv_dim = self.key.weight.shape()[0];
        let kv_heads = kv_dim / hs;
        if kv_dim % hs != 0 || kv_heads == 0 || heads % kv_heads != 0 {
            return Err(Error::Msg(format!(
                "{kv_heads} key / value heads do not divide the {heads} attention heads"
            )));
        }
        let (state, shift) = match cache.take() {
            Some(c) => c,
            None => (
                Tensor::zeros_f32(&[b, heads, hs, hs])?,
                Tensor::zeros_f32(&[b, 1, d])?,
            ),
        };

        let att = self.attn_norm.forward(x)?;
        let sx = shifted(&shift, &att)?.sub(&att)?;
        let xxx = sx.mul(&self.lerp_x)?.add(&att)?;
        let extra = self.mix_w1.shape()[0] / 5;
        let low = lin_last(&xxx, &self.mix_w1)?.tanh()?.reshape(&[b, s, 5, extra])?;
        let mixed = |j: usize| -> Result<Tensor> {
            let w2 = self.mix_w2.narrow(0, j, 1)?.contiguous()?.reshape(&[d, extra])?;
            let lerp = self.lerp.narrow(0, j, 1)?.contiguous()?.reshape(&[d])?;
            let m = lin_last(&low.narrow(2, j, 1)?.contiguous()?.reshape(&[b, s, extra])?, &w2)?;
            m.add(&lerp)?.mul(&sx)?.add(&att)
        };
        let (xw, xk, xv, xr, xg) = (mixed(0)?, mixed(1)?, mixed(2)?, mixed(3)?, mixed(4)?);
        let r = self.receptance.apply(&xr)?;
        let mut k = self.key.apply(&xk)?;
        let mut v = self.value.apply(&xv)?;
        let g = lin_last(&xg, &self.gate)?.sigmoid()?;
        if kv_heads != heads {
            // Grouped heads: consecutive query heads share a key / value head.
            let rep = heads / kv_heads;
            let widen = |t: &Tensor| -> Result<Tensor> {
                t.reshape(&[b, s, kv_heads, 1, hs])?
                    .expand(&[b, s, kv_heads, rep, hs])?
                    .contiguous()?
                    .reshape(&[b, s, d])
            };
            k = widen(&k)?;
            v = widen(&v)?;
        }
        let w = lin_last(&lin_last(&xw, &self.decay_w1)?.tanh()?, &self.decay_w2)?
            .add(&self.decay)?;
        let w = w.exp()?.neg()?.exp()?;
        let k = k.sub(&k.mul(&w)?)?;

        let (rv, kv, vv, wv) = (
            r.to_vec::<f32>()?,
            k.to_vec::<f32>()?,
            v.to_vec::<f32>()?,
            w.to_vec::<f32>()?,
        );
        let scale = (hs as f32).powf(-0.5);
        let mut st = state.to_vec::<f32>()?; // [b, heads, i, j]
        let mut out = vec![0f32; b * s * d];
        for bi in 0..b {
            for t in 0..s {
                let row = (bi * s + t) * d;
                for h in 0..heads {
                    let sbase = (bi * heads + h) * hs * hs;
                    for i in 0..hs {
                        let (ki, qi, wi) = (
                            kv[row + h * hs + i],
                            rv[row + h * hs + i] * scale,
                            wv[row + h * hs + i],
                        );
                        for j in 0..hs {
                            let s_ij = &mut st[sbase + i * hs + j];
                            *s_ij = *s_ij * wi + ki * vv[row + h * hs + j];
                            out[row + h * hs + j] += *s_ij * qi;
                        }
                    }
                }
            }
        }
        let new_state = Tensor::from_slice_on(&st, &[b, heads, hs, hs], &x.device())?;
        let out = Tensor::from_slice_on(&out, &[b, s, d], &x.device())?;
        let x = x.add(&lin_last(&out.mul(&g)?, &self.output)?)?;

        let ffn = self.ffn_norm.forward(&x)?;
        let inner = lin_last(&ffn, &self.ffn_gate)?
            .silu()?
            .mul(&lin_last(&ffn, &self.ffn_up)?)?;
        let y = x.add(&lin_last(&inner, &self.ffn_down)?)?;

        *cache = Some((new_state, att.narrow(1, s - 1, 1)?.contiguous()?));
        Ok(y)
    }
}
