//! Mamba and Mamba-2 selective state-space blocks, computed the way llama.cpp's `ggml_ssm_conv` +
//! `ggml_ssm_scan` do: a causal depthwise convolution with a rolling state, then a per-channel (Mamba) or
//! per-head (Mamba-2) linear recurrence, gated by `silu(z)`.
//!
//! The recurrence is inherently sequential, so it runs on the host in `f32`; the dense projections stay
//! tensor matmuls. Per layer the decoding cache holds `(conv_state [b, d_conv - 1, channels],
//! ssm_state [b, d_inner * d_state])`.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SsmKind {
    /// `mamba`: one `dt` per channel, `A` is `[d_inner, d_state]`.
    Mamba,
    /// `mamba2`: `dt` per head, `A` / `D` per head, `B` / `C` shared by groups of heads.
    Mamba2,
    /// PLaMo-2: heads as in Mamba-2 but `B` / `C` / `dt` come from a projection of the convolved `x` (`dt` through a
    /// low-rank `ssm_dt`), each RMS-normed; `[z | x]` is interleaved per head, only `x` is convolved, and the
    /// output is `silu(z) * y` with no group norm.
    Plamo2,
    /// LFM2's gated short convolution: `out_proj(c * conv(b * x))`, no bias, no activation, no scan.
    ShortConv,
    /// Gated DeltaNet linear attention (Qwen3.5): a causal convolution over `[q | k | v]`, then the delta rule
    /// `S = exp(g) S; S += k (beta (v - S k))^T` per value head, read out with `q`, RMS-normed and gated by `silu(z)`.
    GatedDelta,
    /// Kimi Delta Attention: like `GatedDelta` with one head per value head, but the decay is a per-key-channel
    /// vector `exp(-exp(A) softplus(f_b(f_a x) + dt_bias))`, `q` / `k` / `v` each have their own convolution, and the
    /// output is `rms_norm(o) * sigmoid(g_b(g_a x))`.
    Kda,
    /// MiniMax-01's lightning attention: `silu(W_qkv x)` split per head into `q | k | v`, a per-head exponential decay
    /// `S = exp(-lambda) S + k v^T`, readout `q S`, then `rms_norm(o) * sigmoid(W_g x)` projected out. `in_proj` is `W_qkv`,
    /// `x_proj` holds `W_g`, `a` the per-head `lambda`, `norm_w` the norm over all heads.
    Lightning,
}

/// The extra projections of a Kimi delta-attention layer.
pub struct KdaParts {
    /// `[head_dim, d]` and `[d_inner, head_dim]`: the low-rank decay projection. Without `f_b`, `f_a` is one
    /// full-width `[d_inner, d]` projection.
    pub f_a: Tensor,
    pub f_b: Option<Tensor>,
    /// `[heads, d]`
    pub beta_proj: Tensor,
    /// `[head_dim, d]` and `[d_inner, head_dim]`: the low-rank output gate (`g_a` alone is `[d_inner, d]`).
    pub g_a: Tensor,
    pub g_b: Option<Tensor>,
    /// BailingMoE3's "safe gate": the log-decay is `bound * sigmoid((f_a x + dt_bias) * A)` instead of
    /// `softplus(f_b(f_a x) + dt_bias) * A`.
    pub lower_bound: Option<f32>,
    /// With `lower_bound`: the log-decay is `bound * sigmoid(-(raw * A))` (Kimi-K3) instead of `sigmoid(raw * A)`.
    pub bound_negated: bool,
    /// L2-norm epsilon for `q` / `k` when it differs from the RMS epsilon.
    pub l2_eps: Option<f32>,
}

/// The extra projections of a gated delta-net layer.
pub struct GdnParts {
    /// `[value_dim, d]`
    pub z_proj: Tensor,
    /// `[heads, d]`
    pub beta_proj: Tensor,
    pub alpha_proj: Tensor,
    pub z_scale: Option<f32>,
    pub beta_scale: Option<f32>,
    pub alpha_scale: Option<f32>,
    /// Qwen4-Exp gates the normed output with `sigmoid(z)` instead of `silu(z)`.
    pub sigmoid_gate: bool,
}

/// The `{arch}.ssm.*` metadata.
#[derive(Debug, Clone)]
pub struct SsmConfig {
    pub kind: SsmKind,
    pub d_conv: usize,
    pub d_inner: usize,
    pub d_state: usize,
    /// Mamba: rank of the `dt` projection. Mamba-2: number of heads.
    pub dt_rank: usize,
    pub n_group: usize,
    /// Falcon-Mamba style RMS norms on `dt`, `B` and `C`.
    pub dt_b_c_rms: bool,
}

pub struct SsmBlock {
    pub cfg: SsmConfig,
    /// `[in_dim, d]`: Mamba `[x | z]`, Mamba-2 `[z | x B C | dt]`.
    pub in_proj: Tensor,
    /// `[channels * d_conv]`, channel-major, oldest tap first.
    pub conv_w: Vec<f32>,
    pub conv_b: Vec<f32>,
    /// Mamba only: `[dt_rank + 2 * d_state, d_inner]` and `[d_inner, dt_rank]`.
    pub x_proj: Option<Tensor>,
    pub dt_proj: Option<Tensor>,
    pub dt_bias: Vec<f32>,
    /// Mamba `[d_inner * d_state]` (already `-exp(A_log)`), Mamba-2 `[n_head]`.
    pub a: Vec<f32>,
    /// Mamba `[d_inner]`, Mamba-2 `[n_head]`.
    pub d: Vec<f32>,
    /// Mamba-2 gated group RMS norm weight `[d_inner]`.
    pub norm_w: Option<Vec<f32>>,
    /// Falcon-Mamba norm weights for `dt`, `B`, `C`.
    pub dt_b_c_norm: Option<[Vec<f32>; 3]>,
    /// `[d, d_inner]`.
    pub out_proj: Tensor,
    pub eps: f32,
    /// Per-tensor output scales of the input / output projections (`ssm_in.scale`, `ssm_out.scale`).
    pub in_scale: Option<f32>,
    pub out_scale: Option<f32>,
    pub gdn: Option<GdnParts>,
    pub kda: Option<KdaParts>,
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// ggml's softplus: linear above 20 to avoid overflow.
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

fn rms_in_place(v: &mut [f32], w: &[f32], eps: f32) {
    let ms = v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32;
    let scale = 1.0 / (ms + eps).sqrt();
    for (x, w) in v.iter_mut().zip(w) {
        *x *= scale * w;
    }
}

impl SsmBlock {
    fn channels(&self) -> Result<usize> {
        // Dims are pub: the Mamba-2/Kda products overflow for absurd values.
        match self.cfg.kind {
            SsmKind::Mamba => Ok(self.cfg.d_inner),
            SsmKind::Mamba2 | SsmKind::GatedDelta => {
                let bc = 2usize
                    .checked_mul(self.cfg.n_group)
                    .and_then(|v| v.checked_mul(self.cfg.d_state))
                    .ok_or_else(|| Error::Msg("SsmBlock B/C width overflows".into()))?;
                self.cfg
                    .d_inner
                    .checked_add(bc)
                    .ok_or_else(|| Error::Msg("SsmBlock channel count overflows".into()))
            }
            SsmKind::ShortConv | SsmKind::Plamo2 => Ok(self.cfg.d_inner),
            SsmKind::Kda => 3usize
                .checked_mul(self.cfg.d_inner)
                .ok_or_else(|| Error::Msg("SsmBlock Kda channel count overflows".into())),
            SsmKind::Lightning => Ok(0),
        }
    }

    /// Width checks for the pub projection tensors / parameter vectors: the
    /// sub-forwards index projection outputs with config-derived widths, so a
    /// swapped or truncated projection must Err here instead of panicking (or
    /// silently truncating in a zip) mid-scan.
    fn check_proj(name: &str, w: &Tensor, rows: usize, cols: usize) -> Result<()> {
        let s = w.shape();
        if s.len() == 2 && s[0] == rows && s[1] == cols {
            Ok(())
        } else {
            Err(Error::Msg(format!(
                "SsmBlock {name} has shape {s:?}, expected [{rows}, {cols}]"
            )))
        }
    }

    fn check_len(name: &str, got: usize, want: usize) -> Result<()> {
        if got == want {
            Ok(())
        } else {
            Err(Error::Msg(format!(
                "SsmBlock {name} has length {got}, expected {want}"
            )))
        }
    }

    fn forward_lightning(
        &self,
        x: &Tensor,
        cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let c = &self.cfg;
        let (b, s, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let (nh, hd, di) = (c.dt_rank, c.d_state, c.d_inner);
        let three_di = 3usize
            .checked_mul(di)
            .ok_or_else(|| Error::Msg("lightning attention qkv width overflows".into()))?;
        Self::check_proj("lightning in_proj", &self.in_proj, three_di, d)?;
        Self::check_proj("lightning out_proj", &self.out_proj, d, di)?;
        Self::check_len("lightning a", self.a.len(), nh)?;
        let x2 = x.reshape(&[b * s, d])?;
        let qkv = x2.matmul(&self.in_proj.t()?)?.to_vec::<f32>()?;
        let gate_w = self
            .x_proj
            .as_ref()
            .ok_or_else(|| Error::Msg("lightning attention needs its gate".into()))?;
        Self::check_proj("lightning gate", gate_w, di, d)?;
        let gate = x2.matmul(&gate_w.t()?)?.to_vec::<f32>()?;
        let w = self
            .norm_w
            .as_deref()
            .ok_or_else(|| Error::Msg("lightning attention needs its output norm".into()))?;
        Self::check_len("lightning norm", w.len(), di)?;
        let state_len = nh
            .checked_mul(hd)
            .and_then(|v| v.checked_mul(hd))
            .ok_or_else(|| Error::Msg("lightning attention state length overflows".into()))?;
        let cache_len = b
            .checked_mul(state_len)
            .ok_or_else(|| Error::Msg("lightning attention cache size overflows".into()))?;
        let mut state = match cache.take() {
            Some((_, st)) => st.to_vec::<f32>()?,
            // `try_` (not `vec!`): a fitting-but-huge checked length still
            // panics ("capacity overflow") or aborts on allocation.
            None => crate::layout::try_filled(cache_len, 0.0)?,
        };
        if state.len() != cache_len {
            return Err(Error::Msg(
                "lightning attention cache does not match the batch size".into(),
            ));
        }
        let y_len = b
            .checked_mul(s)
            .and_then(|v| v.checked_mul(di))
            .ok_or_else(|| Error::Msg("lightning attention output size overflows".into()))?;
        let mut y = crate::layout::try_filled(y_len, 0f32)?;
        for bi in 0..b {
            let st = &mut state[bi * state_len..(bi + 1) * state_len];
            for t in 0..s {
                let r = bi * s + t;
                let row = &qkv[r * 3 * di..(r + 1) * 3 * di];
                let mut o_all = vec![0f32; di];
                for h in 0..nh {
                    let blk = &row[h * 3 * hd..(h + 1) * 3 * hd];
                    let (q, k, v) = (&blk[..hd], &blk[hd..2 * hd], &blk[2 * hd..]);
                    let decay = (-self.a[h]).exp();
                    let sh = &mut st[h * hd * hd..(h + 1) * hd * hd];
                    for j in 0..hd {
                        let mut o = 0f32;
                        for i in 0..hd {
                            // sh[i][j]: key channel i, value channel j
                            let cell = &mut sh[i * hd + j];
                            *cell = *cell * decay + silu(k[i]) * silu(v[j]);
                            o += silu(q[i]) * *cell;
                        }
                        o_all[h * hd + j] = o;
                    }
                }
                rms_in_place(&mut o_all, w, self.eps);
                for j in 0..di {
                    y[r * di + j] = o_all[j] / (1.0 + (-gate[r * di + j]).exp());
                }
            }
        }
        *cache = Some((
            Tensor::from_slice(&vec![0f32; b], &[b, 1])?,
            Tensor::from_slice(&state, &[b, state_len])?,
        ));
        Tensor::from_slice(&y, &[b * s, di])?
            .matmul(&self.out_proj.t()?)?
            .reshape(&[b, s, d])
    }

    /// `d_inner = heads * head_dim` (`c.dt_rank` heads of `c.d_state` channels).
    fn forward_kda(&self, x: &Tensor, cache: &mut Option<(Tensor, Tensor)>) -> Result<Tensor> {
        let kp = self
            .kda
            .as_ref()
            .ok_or_else(|| Error::Msg("kimi delta attention needs its projections".into()))?;
        let c = &self.cfg;
        let (b, s, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let (nh, hd, di) = (c.dt_rank, c.d_state, c.d_inner);
        let ch = 3usize
            .checked_mul(di)
            .ok_or_else(|| Error::Msg("kimi delta attention channel count overflows".into()))?;
        let nconv = c.d_conv - 1;
        Self::check_proj("kda in_proj", &self.in_proj, ch, d)?;
        Self::check_proj("kda out_proj", &self.out_proj, d, di)?;
        match &kp.f_b {
            Some(w) => {
                Self::check_proj("kda f_a", &kp.f_a, hd, d)?;
                Self::check_proj("kda f_b", w, di, hd)?;
            }
            None => Self::check_proj("kda f_a", &kp.f_a, di, d)?,
        }
        match &kp.g_b {
            Some(w) => {
                Self::check_proj("kda g_a", &kp.g_a, hd, d)?;
                Self::check_proj("kda g_b", w, di, hd)?;
            }
            None => Self::check_proj("kda g_a", &kp.g_a, di, d)?,
        }
        Self::check_proj("kda beta_proj", &kp.beta_proj, nh, d)?;
        Self::check_len("kda dt_bias", self.dt_bias.len(), di)?;
        Self::check_len("kda a", self.a.len(), nh)?;
        let conv_want = ch
            .checked_mul(c.d_conv)
            .ok_or_else(|| Error::Msg("kimi delta attention conv width overflows".into()))?;
        Self::check_len("kda conv_w", self.conv_w.len(), conv_want)?;
        let x2 = x.reshape(&[b * s, d])?;
        let mm =
            |a: &Tensor, w: &Tensor| -> Result<Vec<f32>> { a.matmul(&w.t()?)?.to_vec::<f32>() };
        let qkv = mm(&x2, &self.in_proj)?;
        let low_rank = |a: &Tensor, b_proj: &Option<Tensor>| -> Result<Vec<f32>> {
            match b_proj {
                Some(w) => mm(&Tensor::from_slice(&mm(&x2, a)?, &[b * s, hd])?, w),
                None => mm(&x2, a),
            }
        };
        let g1 = low_rank(&kp.f_a, &kp.f_b)?;
        let beta = mm(&x2, &kp.beta_proj)?;
        let g2 = low_rank(&kp.g_a, &kp.g_b)?;

        let state_len = nh
            .checked_mul(hd)
            .and_then(|v| v.checked_mul(hd))
            .ok_or_else(|| Error::Msg("kimi delta attention state length overflows".into()))?;
        let conv_len = b
            .checked_mul(nconv)
            .and_then(|v| v.checked_mul(ch))
            .ok_or_else(|| Error::Msg("kimi delta attention conv cache size overflows".into()))?;
        let cache_len = b
            .checked_mul(state_len)
            .ok_or_else(|| Error::Msg("kimi delta attention cache size overflows".into()))?;
        let (mut conv_state, mut kda_state) = match cache.take() {
            Some((cs, st)) => (cs.to_vec::<f32>()?, st.to_vec::<f32>()?),
            None => (
                crate::layout::try_filled(conv_len, 0.0)?,
                crate::layout::try_filled(cache_len, 0.0)?,
            ),
        };
        if conv_state.len() != conv_len || kda_state.len() != cache_len {
            return Err(Error::Msg(
                "kimi delta attention cache does not match the batch size".into(),
            ));
        }
        let l2 = |v: &mut [f32]| {
            let n = v.iter().map(|x| x * x).sum::<f32>() + kp.l2_eps.unwrap_or(self.eps);
            let inv = 1.0 / n.sqrt();
            v.iter_mut().for_each(|x| *x *= inv);
        };
        let q_scale = 1.0 / (hd as f32).sqrt();
        let w_o = self
            .norm_w
            .as_deref()
            .ok_or_else(|| Error::Msg("kimi delta attention needs its output norm".into()))?;
        // Per-head norm (head_dim wide), as the GGUF loader stores it.
        Self::check_len("kda norm", w_o.len(), hd)?;
        let y_len = b
            .checked_mul(s)
            .and_then(|v| v.checked_mul(di))
            .ok_or_else(|| Error::Msg("kimi delta attention output size overflows".into()))?;
        let mut y = crate::layout::try_filled(y_len, 0f32)?;
        for bi in 0..b {
            let cs = &mut conv_state[bi * nconv * ch..(bi + 1) * nconv * ch];
            let st = &mut kda_state[bi * state_len..(bi + 1) * state_len];
            for t in 0..s {
                let r = bi * s + t;
                let xin = &qkv[r * ch..(r + 1) * ch];
                let mut mixed = vec![0f32; ch];
                for ci in 0..ch {
                    let mut acc = 0f32;
                    for k in 0..c.d_conv {
                        let v = if k < nconv { cs[k * ch + ci] } else { xin[ci] };
                        acc += self.conv_w[ci * c.d_conv + k] * v;
                    }
                    mixed[ci] = silu(acc);
                }
                if nconv > 0 {
                    cs.copy_within(ch.., 0);
                    cs[(nconv - 1) * ch..].copy_from_slice(xin);
                }
                let (qs, rest) = mixed.split_at_mut(di);
                let (ks, vs) = rest.split_at_mut(di);
                for h in 0..nh {
                    let (qv, kv, vh) = (
                        &mut qs[h * hd..(h + 1) * hd],
                        &mut ks[h * hd..(h + 1) * hd],
                        &vs[h * hd..(h + 1) * hd],
                    );
                    l2(qv);
                    l2(kv);
                    let b_h = 1.0 / (1.0 + (-beta[r * nh + h]).exp());
                    let decay: Vec<f32> = (0..hd)
                        .map(|i| {
                            let raw = g1[r * di + h * hd + i] + self.dt_bias[h * hd + i];
                            match kp.lower_bound {
                                Some(bound) => {
                                    let z = if kp.bound_negated {
                                        -(raw * self.a[h])
                                    } else {
                                        raw * self.a[h]
                                    };
                                    (bound / (1.0 + (-z).exp())).exp()
                                }
                                None => (softplus(raw) * self.a[h]).exp(),
                            }
                        })
                        .collect();
                    let state = &mut st[h * hd * hd..(h + 1) * hd * hd];
                    let mut out = vec![0f32; hd];
                    for j in 0..hd {
                        let row = &mut state[j * hd..(j + 1) * hd];
                        let mut skj = 0f32;
                        for i in 0..hd {
                            row[i] *= decay[i];
                            skj += row[i] * kv[i];
                        }
                        let dj = (vh[j] - skj) * b_h;
                        let mut o = 0f32;
                        for i in 0..hd {
                            row[i] += kv[i] * dj;
                            o += row[i] * qv[i] * q_scale;
                        }
                        out[j] = o;
                    }
                    rms_in_place(&mut out, w_o, self.eps);
                    for j in 0..hd {
                        let gate = 1.0 / (1.0 + (-g2[r * di + h * hd + j]).exp());
                        y[r * di + h * hd + j] = out[j] * gate;
                    }
                }
            }
        }
        *cache = Some((
            Tensor::from_slice(&conv_state, &[b, nconv, ch])?,
            Tensor::from_slice(&kda_state, &[b, state_len])?,
        ));
        Tensor::from_slice(&y, &[b * s, di])?
            .matmul(&self.out_proj.t()?)?
            .reshape(&[b, s, d])
    }

    fn forward_gdn(&self, x: &Tensor, cache: &mut Option<(Tensor, Tensor)>) -> Result<Tensor> {
        let g = self
            .gdn
            .as_ref()
            .ok_or_else(|| Error::Msg("gated delta net needs its projections".into()))?;
        let c = &self.cfg;
        let (b, s, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let (hv, hk, sk) = (c.dt_rank, c.n_group, c.d_state);
        let sv = c.d_inner / hv;
        if sv != sk {
            return Err(Error::Msg(format!(
                "gated delta net needs equal key and value head sizes, got {sk} and {sv}"
            )));
        }
        let key_dim = hk
            .checked_mul(sk)
            .ok_or_else(|| Error::Msg("gated delta net key width overflows".into()))?;
        let (ch, nconv) = (self.channels()?, c.d_conv - 1);
        Self::check_proj("gdn in_proj", &self.in_proj, ch, d)?;
        Self::check_proj("gdn z_proj", &g.z_proj, c.d_inner, d)?;
        Self::check_proj("gdn beta_proj", &g.beta_proj, hv, d)?;
        Self::check_proj("gdn alpha_proj", &g.alpha_proj, hv, d)?;
        Self::check_proj("gdn out_proj", &self.out_proj, d, c.d_inner)?;
        Self::check_len("gdn dt_bias", self.dt_bias.len(), hv)?;
        Self::check_len("gdn a", self.a.len(), hv)?;
        let conv_want = ch
            .checked_mul(c.d_conv)
            .ok_or_else(|| Error::Msg("gated delta net conv width overflows".into()))?;
        Self::check_len("gdn conv_w", self.conv_w.len(), conv_want)?;
        let mm = |w: &Tensor, k: Option<f32>| -> Result<Vec<f32>> {
            let mut v = x.reshape(&[b * s, d])?.matmul(&w.t()?)?.to_vec::<f32>()?;
            if let Some(k) = k {
                v.iter_mut().for_each(|e| *e *= k);
            }
            Ok(v)
        };
        let qkv = mm(&self.in_proj, self.in_scale)?;
        let z = mm(&g.z_proj, g.z_scale)?;
        let beta = mm(&g.beta_proj, g.beta_scale)?;
        let alpha = mm(&g.alpha_proj, g.alpha_scale)?;

        let state_len = hv
            .checked_mul(sv)
            .and_then(|v| v.checked_mul(sk))
            .ok_or_else(|| Error::Msg("gated delta net state length overflows".into()))?;
        let conv_len = b
            .checked_mul(nconv)
            .and_then(|v| v.checked_mul(ch))
            .ok_or_else(|| Error::Msg("gated delta net conv cache size overflows".into()))?;
        let cache_len = b
            .checked_mul(state_len)
            .ok_or_else(|| Error::Msg("gated delta net cache size overflows".into()))?;
        let (mut conv_state, mut gdn_state) = match cache.take() {
            Some((cs, st)) => (cs.to_vec::<f32>()?, st.to_vec::<f32>()?),
            None => (
                crate::layout::try_filled(conv_len, 0.0)?,
                crate::layout::try_filled(cache_len, 0.0)?,
            ),
        };
        if conv_state.len() != conv_len || gdn_state.len() != cache_len {
            return Err(Error::Msg(
                "gated delta net cache does not match the batch size".into(),
            ));
        }
        let l2 = |v: &mut [f32]| {
            let n = v.iter().map(|x| x * x).sum::<f32>() + self.eps;
            let inv = 1.0 / n.sqrt();
            v.iter_mut().for_each(|x| *x *= inv);
        };
        let q_scale = 1.0 / (sk as f32).sqrt();
        let y_len = b
            .checked_mul(s)
            .and_then(|v| v.checked_mul(c.d_inner))
            .ok_or_else(|| Error::Msg("gated delta net output size overflows".into()))?;
        let mut y = crate::layout::try_filled(y_len, 0f32)?;
        for bi in 0..b {
            let cs = &mut conv_state[bi * nconv * ch..(bi + 1) * nconv * ch];
            let st = &mut gdn_state[bi * state_len..(bi + 1) * state_len];
            for t in 0..s {
                let r = bi * s + t;
                let xin = &qkv[r * ch..(r + 1) * ch];
                let mut mixed = vec![0f32; ch];
                for ci in 0..ch {
                    let mut acc = 0f32;
                    for k in 0..c.d_conv {
                        let v = if k < nconv { cs[k * ch + ci] } else { xin[ci] };
                        acc += self.conv_w[ci * c.d_conv + k] * v;
                    }
                    mixed[ci] = silu(acc);
                }
                if nconv > 0 {
                    cs.copy_within(ch.., 0);
                    cs[(nconv - 1) * ch..].copy_from_slice(xin);
                }
                let (qs, rest) = mixed.split_at_mut(key_dim);
                let (ks, vs) = rest.split_at_mut(key_dim);
                for h in 0..hk {
                    l2(&mut qs[h * sk..(h + 1) * sk]);
                    l2(&mut ks[h * sk..(h + 1) * sk]);
                }
                for h in 0..hv {
                    let kh = h % hk; // llama.cpp tiles the key heads over the value heads
                    let (qv, kv) = (&qs[kh * sk..(kh + 1) * sk], &ks[kh * sk..(kh + 1) * sk]);
                    let b_h = 1.0 / (1.0 + (-beta[r * hv + h]).exp());
                    let decay = (softplus(alpha[r * hv + h] + self.dt_bias[h]) * self.a[h]).exp();
                    let state = &mut st[h * sv * sk..(h + 1) * sv * sk];
                    let vh = &vs[h * sv..(h + 1) * sv];
                    let mut out = vec![0f32; sv];
                    for j in 0..sv {
                        let row = &mut state[j * sk..(j + 1) * sk];
                        let mut skj = 0f32;
                        for i in 0..sk {
                            row[i] *= decay;
                            skj += row[i] * kv[i];
                        }
                        let dj = (vh[j] - skj) * b_h;
                        let mut o = 0f32;
                        for i in 0..sk {
                            row[i] += kv[i] * dj;
                            o += row[i] * qv[i] * q_scale;
                        }
                        out[j] = o;
                    }
                    let w = self.norm_w.as_deref().unwrap_or(&[]);
                    if w.len() == sv {
                        rms_in_place(&mut out, w, self.eps);
                    }
                    let zh = &z[r * c.d_inner + h * sv..r * c.d_inner + (h + 1) * sv];
                    for j in 0..sv {
                        y[r * c.d_inner + h * sv + j] = out[j]
                            * if g.sigmoid_gate {
                                1.0 / (1.0 + (-zh[j]).exp())
                            } else {
                                silu(zh[j])
                            };
                    }
                }
            }
        }
        *cache = Some((
            Tensor::from_slice(&conv_state, &[b, nconv, ch])?,
            Tensor::from_slice(&gdn_state, &[b, state_len])?,
        ));
        let out = Tensor::from_slice(&y, &[b * s, c.d_inner])?.matmul(&self.out_proj.t()?)?;
        let out = match self.out_scale {
            Some(k) => out.mul_scalar(f64::from(k))?,
            None => out,
        };
        out.reshape(&[b, s, d])
    }

    fn forward_shortconv(
        &self,
        x: &Tensor,
        cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b, s, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let (l, nconv) = (self.cfg.d_conv, self.cfg.d_conv - 1);
        let three_d = 3usize
            .checked_mul(d)
            .ok_or_else(|| Error::Msg("short-conv width overflows".into()))?;
        Self::check_proj("short-conv in_proj", &self.in_proj, three_d, d)?;
        Self::check_proj("short-conv out_proj", &self.out_proj, d, d)?;
        let conv_want = d
            .checked_mul(l)
            .ok_or_else(|| Error::Msg("short-conv conv width overflows".into()))?;
        Self::check_len("short-conv conv_w", self.conv_w.len(), conv_want)?;
        let proj = x
            .reshape(&[b * s, d])?
            .matmul(&self.in_proj.t()?)?
            .to_vec::<f32>()?;
        let conv_len = b
            .checked_mul(nconv)
            .and_then(|v| v.checked_mul(d))
            .ok_or_else(|| Error::Msg("short-conv cache size overflows".into()))?;
        let mut conv_state = match cache.take() {
            Some((cs, _)) => cs.to_vec::<f32>()?,
            None => crate::layout::try_filled(conv_len, 0.0)?,
        };
        if conv_state.len() != conv_len {
            return Err(Error::Msg(
                "short-conv cache does not match the batch size".into(),
            ));
        }
        let y_len = b
            .checked_mul(s)
            .and_then(|v| v.checked_mul(d))
            .ok_or_else(|| Error::Msg("short-conv output size overflows".into()))?;
        let mut y = crate::layout::try_filled(y_len, 0f32)?;
        for bi in 0..b {
            let cs = &mut conv_state[bi * nconv * d..(bi + 1) * nconv * d];
            for t in 0..s {
                let r = bi * s + t;
                let row = &proj[r * 3 * d..(r + 1) * 3 * d];
                let (bg, cg, xv) = (&row[..d], &row[d..2 * d], &row[2 * d..]);
                let bx: Vec<f32> = bg.iter().zip(xv).map(|(a, b)| a * b).collect();
                for ci in 0..d {
                    let mut acc = 0f32;
                    for k in 0..l {
                        let v = if k < nconv { cs[k * d + ci] } else { bx[ci] };
                        acc += self.conv_w[ci * l + k] * v;
                    }
                    y[r * d + ci] = cg[ci] * acc;
                }
                if nconv > 0 {
                    cs.copy_within(d.., 0);
                    cs[(nconv - 1) * d..].copy_from_slice(&bx);
                }
            }
        }
        *cache = Some((
            Tensor::from_slice(&conv_state, &[b, nconv, d])?,
            Tensor::from_slice(&vec![0f32; b], &[b, 1])?,
        ));
        Tensor::from_slice(&y, &[b * s, d])?
            .matmul(&self.out_proj.t()?)?
            .reshape(&[b, s, d])
    }

    /// `x`: the normalized layer input `[b, s, d]`. Returns the block output `[b, s, d]` (before the residual).
    pub fn forward(&self, x: &Tensor, cache: &mut Option<(Tensor, Tensor)>) -> Result<Tensor> {
        if x.ndim() != 3 {
            return Err(Error::Msg(format!(
                "SsmBlock expects 3D x [B, S, D], got {:?}",
                x.shape()
            )));
        }
        // All dims are pub on the config; every sub-forward is private and
        // dispatched here, so one entry check covers them all: zero dims
        // underflow `d_conv - 1` and trap the head/group divisions, and
        // Mamba-2 needs at least as many heads as groups (`nh / g == 0`
        // traps at the group lookup).
        let c0 = &self.cfg;
        // ShortConv reads only d_conv (its width comes from x); the GGUF
        // loader legitimately builds it with d_state/dt_rank == 0.
        if c0.d_conv == 0
            || c0.d_inner == 0
            || c0.n_group == 0
            || (c0.kind != SsmKind::ShortConv && (c0.d_state == 0 || c0.dt_rank == 0))
        {
            return Err(Error::Msg(format!(
                "SsmBlock needs positive d_conv/d_inner/d_state/dt_rank/n_group, got {}/{}/{}/{}/{}",
                c0.d_conv, c0.d_inner, c0.d_state, c0.dt_rank, c0.n_group
            )));
        }
        if c0.kind == SsmKind::Mamba2 && c0.dt_rank < c0.n_group {
            return Err(Error::Msg(format!(
                "SsmBlock Mamba-2 needs dt_rank (heads) >= n_group, got {} < {}",
                c0.dt_rank, c0.n_group
            )));
        }
        // Head geometry the scans assume (the GGUF config loader enforces the
        // same equalities): a truncated head_dim drops channels silently, and
        // Mamba-2 group lookup walks past the B/C block.
        match c0.kind {
            SsmKind::Mamba2 | SsmKind::Plamo2 | SsmKind::GatedDelta => {
                if c0.d_inner % c0.dt_rank != 0 {
                    return Err(Error::Msg(format!(
                        "SsmBlock {:?} needs d_inner divisible by dt_rank (heads), got {} % {}",
                        c0.kind, c0.d_inner, c0.dt_rank
                    )));
                }
            }
            SsmKind::Kda | SsmKind::Lightning => {
                let heads_width = c0
                    .dt_rank
                    .checked_mul(c0.d_state)
                    .ok_or_else(|| Error::Msg("SsmBlock head width overflows".into()))?;
                if heads_width != c0.d_inner {
                    return Err(Error::Msg(format!(
                        "SsmBlock {:?} needs dt_rank * d_state == d_inner, got {} * {} != {}",
                        c0.kind, c0.dt_rank, c0.d_state, c0.d_inner
                    )));
                }
            }
            SsmKind::Mamba | SsmKind::ShortConv => {}
        }
        if c0.kind == SsmKind::Mamba2 && c0.dt_rank % c0.n_group != 0 {
            return Err(Error::Msg(format!(
                "SsmBlock Mamba-2 needs dt_rank (heads) divisible by n_group, got {} % {}",
                c0.dt_rank, c0.n_group
            )));
        }
        if self.cfg.kind == SsmKind::ShortConv {
            return self.forward_shortconv(x, cache);
        }
        if self.cfg.kind == SsmKind::GatedDelta {
            return self.forward_gdn(x, cache);
        }
        if self.cfg.kind == SsmKind::Kda {
            return self.forward_kda(x, cache);
        }
        if self.cfg.kind == SsmKind::Lightning {
            return self.forward_lightning(x, cache);
        }
        let c = &self.cfg;
        let (b, s, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        if b == 0 || s == 0 {
            // No tokens: the width division below would divide by zero. The cache is
            // untouched (nothing was consumed) and the empty output keeps its shape.
            return Tensor::from_slice(&[] as &[f32], &[b, s, d]);
        }
        let (ch, nconv, di, ds) = (self.channels()?, c.d_conv - 1, c.d_inner, c.d_state);
        let state_len = di
            .checked_mul(ds)
            .ok_or_else(|| Error::Msg("SSM state length overflows".into()))?;
        // The scan below slices the projected rows and the parameter vectors
        // with config widths; validate them all before any indexing.
        let want_in = match c.kind {
            SsmKind::Mamba | SsmKind::Plamo2 => di
                .checked_mul(2)
                .ok_or_else(|| Error::Msg("SSM input width overflows".into()))?,
            SsmKind::Mamba2 => di
                .checked_add(ch)
                .and_then(|v| v.checked_add(c.dt_rank))
                .ok_or_else(|| Error::Msg("SSM input width overflows".into()))?,
            SsmKind::ShortConv | SsmKind::GatedDelta | SsmKind::Kda | SsmKind::Lightning => {
                unreachable!("handled by their own forward")
            }
        };
        Self::check_proj("in_proj", &self.in_proj, want_in, d)?;
        Self::check_proj("out_proj", &self.out_proj, d, di)?;
        let conv_want = ch
            .checked_mul(c.d_conv)
            .ok_or_else(|| Error::Msg("SSM conv width overflows".into()))?;
        Self::check_len("conv_w", self.conv_w.len(), conv_want)?;
        Self::check_len("conv_b", self.conv_b.len(), ch)?;
        match c.kind {
            SsmKind::Mamba => {
                let wd = c
                    .dt_rank
                    .checked_add(
                        ds.checked_mul(2)
                            .ok_or_else(|| Error::Msg("mamba xdb width overflows".into()))?,
                    )
                    .ok_or_else(|| Error::Msg("mamba xdb width overflows".into()))?;
                let xp = self
                    .x_proj
                    .as_ref()
                    .ok_or_else(|| Error::Msg("mamba needs ssm_x".into()))?;
                Self::check_proj("mamba x_proj", xp, wd, di)?;
                let dp = self
                    .dt_proj
                    .as_ref()
                    .ok_or_else(|| Error::Msg("mamba needs ssm_dt".into()))?;
                Self::check_proj("mamba dt_proj", dp, di, c.dt_rank)?;
                Self::check_len("mamba dt_bias", self.dt_bias.len(), di)?;
                let a_want = di
                    .checked_mul(ds)
                    .ok_or_else(|| Error::Msg("mamba A length overflows".into()))?;
                Self::check_len("mamba a", self.a.len(), a_want)?;
                Self::check_len("mamba d", self.d.len(), di)?;
                if let Some([nd, nb, nc]) = &self.dt_b_c_norm {
                    Self::check_len("mamba dt norm", nd.len(), c.dt_rank)?;
                    Self::check_len("mamba B norm", nb.len(), ds)?;
                    Self::check_len("mamba C norm", nc.len(), ds)?;
                }
            }
            SsmKind::Mamba2 => {
                Self::check_len("mamba2 dt_bias", self.dt_bias.len(), c.dt_rank)?;
                Self::check_len("mamba2 a", self.a.len(), c.dt_rank)?;
                Self::check_len("mamba2 d", self.d.len(), c.dt_rank)?;
                if let Some(w) = &self.norm_w {
                    Self::check_len("mamba2 norm", w.len(), di)?;
                }
            }
            SsmKind::Plamo2 => {
                let xp = self
                    .x_proj
                    .as_ref()
                    .ok_or_else(|| Error::Msg("plamo2 needs ssm_x".into()))?;
                let xs_shape = xp.shape();
                let two_ds = ds
                    .checked_mul(2)
                    .ok_or_else(|| Error::Msg("plamo2 B/C width overflows".into()))?;
                if xs_shape.len() != 2 || xs_shape[1] != di || xs_shape[0] <= two_ds {
                    return Err(Error::Msg(format!(
                        "SsmBlock plamo2 x_proj has shape {xs_shape:?}, expected [dt_dim + {two_ds}, {di}] with dt_dim >= 1"
                    )));
                }
                let dt_dim = xs_shape[0] - two_ds;
                let dp = self
                    .dt_proj
                    .as_ref()
                    .ok_or_else(|| Error::Msg("plamo2 needs ssm_dt".into()))?;
                Self::check_proj("plamo2 dt_proj", dp, c.dt_rank, dt_dim)?;
                Self::check_len("plamo2 dt_bias", self.dt_bias.len(), c.dt_rank)?;
                Self::check_len("plamo2 a", self.a.len(), c.dt_rank)?;
                Self::check_len("plamo2 d", self.d.len(), c.dt_rank)?;
                if let Some([nd, nb, nc]) = &self.dt_b_c_norm {
                    Self::check_len("plamo2 dt norm", nd.len(), dt_dim)?;
                    Self::check_len("plamo2 B norm", nb.len(), ds)?;
                    Self::check_len("plamo2 C norm", nc.len(), ds)?;
                }
            }
            SsmKind::ShortConv | SsmKind::GatedDelta | SsmKind::Kda | SsmKind::Lightning => {
                unreachable!("handled by their own forward")
            }
        }
        let mut proj = x
            .reshape(&[b * s, d])?
            .matmul(&self.in_proj.t()?)?
            .to_vec::<f32>()?;
        if let Some(k) = self.in_scale {
            proj.iter_mut().for_each(|v| *v *= k);
        }
        let in_dim = proj.len() / (b * s);

        let conv_len = b
            .checked_mul(nconv)
            .and_then(|v| v.checked_mul(ch))
            .ok_or_else(|| Error::Msg("SSM conv cache size overflows".into()))?;
        let cache_len = b
            .checked_mul(state_len)
            .ok_or_else(|| Error::Msg("SSM cache size overflows".into()))?;
        let (mut conv_state, mut ssm_state) = match cache.take() {
            Some((cs, hs)) => (cs.to_vec::<f32>()?, hs.to_vec::<f32>()?),
            None => (
                crate::layout::try_filled(conv_len, 0.0)?,
                crate::layout::try_filled(cache_len, 0.0)?,
            ),
        };
        if conv_state.len() != conv_len || ssm_state.len() != cache_len {
            return Err(Error::Msg("SSM cache does not match the batch size".into()));
        }

        // 1. causal depthwise convolution (+ bias, silu), rolling the state forward
        let conv_in = |row: &[f32]| -> Vec<f32> {
            match c.kind {
                SsmKind::Mamba => row[..di].to_vec(),
                SsmKind::Mamba2 => row[di..di + ch].to_vec(),
                SsmKind::Plamo2 => {
                    let hd = di / c.dt_rank;
                    (0..di)
                        .map(|j| row[(j / hd) * 2 * hd + hd + j % hd])
                        .collect()
                }
                SsmKind::ShortConv | SsmKind::GatedDelta | SsmKind::Kda | SsmKind::Lightning => {
                    unreachable!("handled by their own forward")
                }
            }
        };
        let conv_out_len = b
            .checked_mul(s)
            .and_then(|v| v.checked_mul(ch))
            .ok_or_else(|| Error::Msg("SSM conv output size overflows".into()))?;
        let mut conv_out = crate::layout::try_filled(conv_out_len, 0f32)?;
        for bi in 0..b {
            let cs = &mut conv_state[bi * nconv * ch..(bi + 1) * nconv * ch];
            for t in 0..s {
                let row = &proj[(bi * s + t) * in_dim..(bi * s + t + 1) * in_dim];
                let xin = conv_in(row);
                let out = &mut conv_out[(bi * s + t) * ch..(bi * s + t + 1) * ch];
                for ci in 0..ch {
                    let mut acc = self.conv_b[ci];
                    for k in 0..c.d_conv {
                        let v = if k < nconv { cs[k * ch + ci] } else { xin[ci] };
                        acc += self.conv_w[ci * c.d_conv + k] * v;
                    }
                    out[ci] = silu(acc);
                }
                if nconv > 0 {
                    cs.copy_within(ch.., 0);
                    cs[(nconv - 1) * ch..].copy_from_slice(&xin);
                }
            }
        }

        // 2. selective scan
        let y_len = b
            .checked_mul(s)
            .and_then(|v| v.checked_mul(di))
            .ok_or_else(|| Error::Msg("SSM output size overflows".into()))?;
        let mut y = crate::layout::try_filled(y_len, 0f32)?;
        match c.kind {
            SsmKind::Mamba => {
                let xs = Tensor::from_slice(&conv_out, &[b * s, di])?;
                let x_proj_t = self
                    .x_proj
                    .as_ref()
                    .ok_or_else(|| Error::Msg("mamba needs ssm_x".into()))?
                    .t()?;
                let x_db = xs.matmul(&x_proj_t)?.to_vec::<f32>()?;
                let wd = c
                    .dt_rank
                    .checked_add(
                        2usize
                            .checked_mul(ds)
                            .ok_or_else(|| Error::Msg("mamba xdb width overflows".into()))?,
                    )
                    .ok_or_else(|| Error::Msg("mamba xdb width overflows".into()))?;
                let dt_len = b
                    .checked_mul(s)
                    .and_then(|v| v.checked_mul(c.dt_rank))
                    .ok_or_else(|| Error::Msg("mamba dt buffer size overflows".into()))?;
                let bc_len = b
                    .checked_mul(s)
                    .and_then(|v| v.checked_mul(ds))
                    .ok_or_else(|| Error::Msg("mamba B/C buffer size overflows".into()))?;
                let mut dt_in = crate::layout::try_filled(dt_len, 0f32)?;
                let (mut bm, mut cm) = (
                    crate::layout::try_filled(bc_len, 0f32)?,
                    crate::layout::try_filled(bc_len, 0f32)?,
                );
                for r in 0..b * s {
                    dt_in[r * c.dt_rank..(r + 1) * c.dt_rank]
                        .copy_from_slice(&x_db[r * wd..r * wd + c.dt_rank]);
                    bm[r * ds..(r + 1) * ds]
                        .copy_from_slice(&x_db[r * wd + c.dt_rank..r * wd + c.dt_rank + ds]);
                    cm[r * ds..(r + 1) * ds]
                        .copy_from_slice(&x_db[r * wd + c.dt_rank + ds..(r + 1) * wd]);
                    if let Some([nd, nb, nc]) = &self.dt_b_c_norm {
                        rms_in_place(&mut dt_in[r * c.dt_rank..(r + 1) * c.dt_rank], nd, self.eps);
                        rms_in_place(&mut bm[r * ds..(r + 1) * ds], nb, self.eps);
                        rms_in_place(&mut cm[r * ds..(r + 1) * ds], nc, self.eps);
                    }
                }
                let dt_proj_t = self
                    .dt_proj
                    .as_ref()
                    .ok_or_else(|| Error::Msg("mamba needs ssm_dt".into()))?
                    .t()?;
                let dt = Tensor::from_slice(&dt_in, &[b * s, c.dt_rank])?
                    .matmul(&dt_proj_t)?
                    .to_vec::<f32>()?;
                for bi in 0..b {
                    let hs = &mut ssm_state[bi * state_len..(bi + 1) * state_len];
                    for t in 0..s {
                        let r = bi * s + t;
                        for i in 0..di {
                            let dtp = softplus(dt[r * di + i] + self.dt_bias[i]);
                            let xv = conv_out[r * di + i];
                            let xdt = xv * dtp;
                            let mut acc = 0f32;
                            for k in 0..ds {
                                let h = &mut hs[i * ds + k];
                                *h = *h * (dtp * self.a[i * ds + k]).exp() + bm[r * ds + k] * xdt;
                                acc += *h * cm[r * ds + k];
                            }
                            y[r * di + i] = acc + xv * self.d[i];
                        }
                    }
                }
            }
            SsmKind::ShortConv | SsmKind::GatedDelta | SsmKind::Kda | SsmKind::Lightning => {
                unreachable!("handled by their own forward")
            }
            SsmKind::Plamo2 => {
                let (nh, hd) = (c.dt_rank, di / c.dt_rank);
                let xs = Tensor::from_slice(&conv_out, &[b * s, di])?;
                let x_proj_t = self
                    .x_proj
                    .as_ref()
                    .ok_or_else(|| Error::Msg("plamo2 needs ssm_x".into()))?
                    .t()?;
                let x_bcdt = xs.matmul(&x_proj_t)?.to_vec::<f32>()?;
                let wd = x_bcdt.len() / (b * s);
                let bc = 2usize
                    .checked_mul(ds)
                    .ok_or_else(|| Error::Msg("plamo2 B/C width overflows".into()))?;
                let dt_dim = wd
                    .checked_sub(bc)
                    .ok_or_else(|| Error::Msg("plamo2 dt width underflows".into()))?;
                let [nd, nb, nc] = self
                    .dt_b_c_norm
                    .as_ref()
                    .ok_or_else(|| Error::Msg("plamo2 needs its dt / B / C norms".into()))?;
                let bc_len = b
                    .checked_mul(s)
                    .and_then(|v| v.checked_mul(ds))
                    .ok_or_else(|| Error::Msg("plamo2 B/C buffer size overflows".into()))?;
                let dt_len = b
                    .checked_mul(s)
                    .and_then(|v| v.checked_mul(dt_dim))
                    .ok_or_else(|| Error::Msg("plamo2 dt buffer size overflows".into()))?;
                let (mut bm, mut cm) = (
                    crate::layout::try_filled(bc_len, 0f32)?,
                    crate::layout::try_filled(bc_len, 0f32)?,
                );
                let mut dt_in = crate::layout::try_filled(dt_len, 0f32)?;
                for r in 0..b * s {
                    bm[r * ds..(r + 1) * ds].copy_from_slice(&x_bcdt[r * wd..r * wd + ds]);
                    cm[r * ds..(r + 1) * ds].copy_from_slice(&x_bcdt[r * wd + ds..r * wd + 2 * ds]);
                    dt_in[r * dt_dim..(r + 1) * dt_dim]
                        .copy_from_slice(&x_bcdt[r * wd + 2 * ds..(r + 1) * wd]);
                    rms_in_place(&mut bm[r * ds..(r + 1) * ds], nb, self.eps);
                    rms_in_place(&mut cm[r * ds..(r + 1) * ds], nc, self.eps);
                    rms_in_place(&mut dt_in[r * dt_dim..(r + 1) * dt_dim], nd, self.eps);
                }
                let dt_proj_t = self
                    .dt_proj
                    .as_ref()
                    .ok_or_else(|| Error::Msg("plamo2 needs ssm_dt".into()))?
                    .t()?;
                let dt = Tensor::from_slice(&dt_in, &[b * s, dt_dim])?
                    .matmul(&dt_proj_t)?
                    .to_vec::<f32>()?;
                for bi in 0..b {
                    let hs = &mut ssm_state[bi * state_len..(bi + 1) * state_len];
                    for t in 0..s {
                        let r = bi * s + t;
                        for h in 0..nh {
                            let dtp = softplus(dt[r * nh + h] + self.dt_bias[h]);
                            let da = (dtp * self.a[h]).exp();
                            for e in 0..hd {
                                let xv = conv_out[r * di + h * hd + e];
                                let xdt = xv * dtp;
                                let mut acc = 0f32;
                                for k in 0..ds {
                                    let st = &mut hs[(h * hd + e) * ds + k];
                                    *st = *st * da + bm[r * ds + k] * xdt;
                                    acc += *st * cm[r * ds + k];
                                }
                                y[r * di + h * hd + e] = acc + xv * self.d[h];
                            }
                        }
                    }
                }
            }
            SsmKind::Mamba2 => {
                let (nh, g) = (c.dt_rank, c.n_group);
                let hd = di / nh;
                for bi in 0..b {
                    let hs = &mut ssm_state[bi * state_len..(bi + 1) * state_len];
                    for t in 0..s {
                        let r = bi * s + t;
                        let row = &proj[r * in_dim..(r + 1) * in_dim];
                        let dt_row = &row[di + ch..di + ch + nh];
                        let xbc = &conv_out[r * ch..(r + 1) * ch];
                        for h in 0..nh {
                            let dtp = softplus(dt_row[h] + self.dt_bias[h]);
                            let da = (dtp * self.a[h]).exp();
                            let grp = h / (nh / g);
                            let bvec = &xbc[di + grp * ds..di + (grp + 1) * ds];
                            let cvec = &xbc[di + g * ds + grp * ds..di + g * ds + (grp + 1) * ds];
                            for e in 0..hd {
                                let xv = xbc[h * hd + e];
                                let xdt = xv * dtp;
                                let mut acc = 0f32;
                                for k in 0..ds {
                                    let st = &mut hs[(h * hd + e) * ds + k];
                                    *st = *st * da + bvec[k] * xdt;
                                    acc += *st * cvec[k];
                                }
                                y[r * di + h * hd + e] = acc + xv * self.d[h];
                            }
                        }
                    }
                }
            }
        }

        // 3. gate with silu(z), group-norm (Mamba-2), project out
        for r in 0..b * s {
            let row = &proj[r * in_dim..(r + 1) * in_dim];
            let plamo_z: Vec<f32>;
            let two_di = di
                .checked_mul(2)
                .ok_or_else(|| Error::Msg("SSM gate width overflows".into()))?;
            let z = match c.kind {
                SsmKind::Mamba => &row[di..two_di],
                SsmKind::Mamba2 => &row[..di],
                SsmKind::Plamo2 => {
                    let hd = di / c.dt_rank;
                    plamo_z = (0..di).map(|j| row[(j / hd) * 2 * hd + j % hd]).collect();
                    &plamo_z[..]
                }
                SsmKind::ShortConv | SsmKind::GatedDelta | SsmKind::Kda | SsmKind::Lightning => {
                    unreachable!("handled by their own forward")
                }
            };
            let yr = &mut y[r * di..(r + 1) * di];
            for (v, zv) in yr.iter_mut().zip(z) {
                *v *= silu(*zv);
            }
            if let Some(w) = &self.norm_w {
                let per = di / c.n_group;
                for gi in 0..c.n_group {
                    rms_in_place(
                        &mut yr[gi * per..(gi + 1) * per],
                        &w[gi * per..(gi + 1) * per],
                        self.eps,
                    );
                }
            }
        }
        *cache = Some((
            Tensor::from_slice(&conv_state, &[b, nconv, ch])?,
            Tensor::from_slice(&ssm_state, &[b, state_len])?,
        ));
        let out = Tensor::from_slice(&y, &[b * s, di])?.matmul(&self.out_proj.t()?)?;
        let out = match self.out_scale {
            Some(k) => out.mul_scalar(f64::from(k))?,
            None => out,
        };
        out.reshape(&[b, s, d])
    }
}
