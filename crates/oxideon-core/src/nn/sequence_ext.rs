//! Sequence-model pieces: variable-length recurrent layers ([`lstm_masked`], [`gru_masked`]; the equivalent of
//! `pack_padded_sequence`), T5-style [`RelativePositionBias`], and the gated / learnable activations [`GeGlu`],
//! [`ReGlu`], [`Maxout`], [`Snake`], [`Swish`] and [`SquaredRelu`].

use super::{GRU, LSTM, Module};
use crate::error::{Error, Result};
use crate::tensor::Tensor;

fn check_lengths(input: &Tensor, lengths: &[usize]) -> Result<()> {
    if input.ndim() != 3 || lengths.len() != input.shape()[0] {
        return Err(Error::Msg("masked RNN: expects [B, T, F] input and one length per sequence".into()));
    }
    if lengths.iter().any(|&l| l == 0 || l > input.shape()[1]) {
        return Err(Error::Msg("masked RNN: every length must lie in 1..=T".into()));
    }
    Ok(())
}

/// `[B, 1]` mask of the sequences still running at step `t`.
fn step_mask(lengths: &[usize], t: usize, like: &Tensor) -> Result<Tensor> {
    let m: Vec<f64> = lengths.iter().map(|&l| if t < l { 1.0 } else { 0.0 }).collect();
    Tensor::from_slice_on(&m, &[lengths.len(), 1], &like.device())?.to_dtype(like.dtype())
}

/// Runs a multi-layer LSTM over padded sequences `[B, T, F]` whose true lengths are `lengths`. State does not advance
/// past a sequence's end and padded output positions are zero, so the result equals running every sequence alone.
/// Returns the padded outputs `[B, T, H]` and the final hidden state of the top layer at each sequence's last step.
pub fn lstm_masked(lstm: &LSTM, input: &Tensor, lengths: &[usize]) -> Result<(Tensor, Tensor)> {
    check_lengths(input, lengths)?;
    let t_max = input.shape()[1];
    let mut seq = input.clone();
    let mut last_h = None;
    for cell in &lstm.cells {
        let mut state: Option<(Tensor, Tensor)> = None;
        let mut outs = Vec::with_capacity(t_max);
        for t in 0..t_max {
            let xt = seq.select(1, t as isize)?;
            let m = step_mask(lengths, t, &xt)?;
            let (h_new, c_new) = cell.step(&xt, state.as_ref().map(|(h, c)| (h, c)))?;
            let (h, c) = match &state {
                Some((h_old, c_old)) => (
                    h_new.mul(&m)?.add(&h_old.mul(&m.neg()?.add_scalar(1.0)?)?)?,
                    c_new.mul(&m)?.add(&c_old.mul(&m.neg()?.add_scalar(1.0)?)?)?,
                ),
                None => (h_new.clone(), c_new.clone()),
            };
            outs.push(h_new.mul(&m)?);
            state = Some((h, c));
        }
        last_h = state.map(|(h, _)| h);
        seq = Tensor::stack(&outs, 1)?;
    }
    Ok((seq, last_h.ok_or_else(|| Error::Msg("lstm_masked: no layers".into()))?))
}

/// The GRU counterpart of [`lstm_masked`].
pub fn gru_masked(gru: &GRU, input: &Tensor, lengths: &[usize]) -> Result<(Tensor, Tensor)> {
    check_lengths(input, lengths)?;
    let t_max = input.shape()[1];
    let mut seq = input.clone();
    let mut last_h = None;
    for cell in &gru.cells {
        let mut state: Option<Tensor> = None;
        let mut outs = Vec::with_capacity(t_max);
        for t in 0..t_max {
            let xt = seq.select(1, t as isize)?;
            let m = step_mask(lengths, t, &xt)?;
            let h_new = cell.step(&xt, state.as_ref())?;
            let h = match &state {
                Some(h_old) => h_new.mul(&m)?.add(&h_old.mul(&m.neg()?.add_scalar(1.0)?)?)?,
                None => h_new.clone(),
            };
            outs.push(h_new.mul(&m)?);
            state = Some(h);
        }
        last_h = state;
        seq = Tensor::stack(&outs, 1)?;
    }
    Ok((seq, last_h.ok_or_else(|| Error::Msg("gru_masked: no layers".into()))?))
}

/// T5 relative position bias: learned per-head scalars indexed by a logarithmically bucketed relative distance.
pub struct RelativePositionBias {
    pub embedding: Tensor,
    pub num_buckets: usize,
    pub max_distance: usize,
    pub bidirectional: bool,
}

impl RelativePositionBias {
    pub fn new(num_buckets: usize, max_distance: usize, heads: usize, bidirectional: bool) -> Result<Self> {
        Ok(Self { embedding: Tensor::randn(&[num_buckets, heads])?.mul_scalar(0.02)?.requires_grad_(true), num_buckets, max_distance, bidirectional })
    }

    /// T5's bucketing of `relative = key_pos - query_pos`.
    pub fn bucket(&self, relative: i64) -> usize {
        let (mut buckets, mut n, mut out) = (self.num_buckets as i64, -relative, 0i64);
        if self.bidirectional {
            buckets /= 2;
            if relative > 0 {
                out += buckets;
            }
            n = relative.abs();
        } else {
            n = n.max(0);
        }
        let max_exact = buckets / 2;
        let large = if n < max_exact {
            n
        } else {
            let v = max_exact as f64 + ((n as f64 / max_exact as f64).ln() / (self.max_distance as f64 / max_exact as f64).ln() * (buckets - max_exact) as f64).floor();
            (v as i64).min(buckets - 1)
        };
        (out + if n < max_exact { n } else { large }) as usize
    }

    /// The bias `[1, heads, q_len, k_len]` to add to attention scores.
    pub fn forward(&self, q_len: usize, k_len: usize) -> Result<Tensor> {
        let mut idx = Vec::with_capacity(q_len * k_len);
        for q in 0..q_len {
            for k in 0..k_len {
                idx.push(self.bucket(k as i64 - q as i64) as i64);
            }
        }
        let ids = Tensor::from_slice_on(&idx, &[idx.len()], &self.embedding.device())?;
        let heads = self.embedding.shape()[1];
        self.embedding.index_select(0, &ids)?.reshape(&[q_len, k_len, heads])?.permute(&[2, 0, 1])?.unsqueeze(0)
    }
}

impl Module for RelativePositionBias {
    /// Input is ignored except for its length: a `[q_len, k_len]` tensor yields the bias for that size.
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        RelativePositionBias::forward(self, input.shape()[0], input.shape()[1])
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![("embedding".to_string(), self.embedding.clone())]
    }
}

fn halves(x: &Tensor) -> Result<(Tensor, Tensor)> {
    let d = *x.shape().last().ok_or_else(|| Error::Msg("gated activation needs at least one dimension".into()))?;
    if d % 2 != 0 {
        return Err(Error::Msg("gated activation: the last dimension must be even".into()));
    }
    let ax = x.ndim() as isize - 1;
    Ok((x.narrow(ax, 0, d / 2)?, x.narrow(ax, d / 2, d / 2)?))
}

/// `a * gelu(b)` over the two halves of the last dimension.
pub struct GeGlu;

impl Module for GeGlu {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (a, b) = halves(x)?;
        a.mul(&b.gelu()?)
    }
}

/// `a * relu(b)` over the two halves of the last dimension.
pub struct ReGlu;

impl Module for ReGlu {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (a, b) = halves(x)?;
        a.mul(&b.relu()?)
    }
}

/// `relu(x)^2`.
pub struct SquaredRelu;

impl Module for SquaredRelu {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.relu()?.square()
    }
}

/// Maxout: the maximum over groups of `k` consecutive features.
pub struct Maxout(pub usize);

impl Module for Maxout {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let d = *x.shape().last().ok_or_else(|| Error::Msg("Maxout needs at least one dimension".into()))?;
        if self.0 == 0 || d % self.0 != 0 {
            return Err(Error::Msg(format!("Maxout: {d} features do not split into groups of {}", self.0)));
        }
        let mut shape = x.shape()[..x.ndim() - 1].to_vec();
        shape.extend([d / self.0, self.0]);
        x.reshape(&shape)?.max_axis(-1, false)
    }
}

/// Snake activation `x + sin(alpha x)^2 / alpha` with a learnable per-feature `alpha` (last dimension).
pub struct Snake {
    pub alpha: Tensor,
}

impl Snake {
    pub fn new(features: usize, alpha: f64) -> Result<Self> {
        Ok(Self { alpha: Tensor::full(&[features], alpha, crate::dtype::DType::F32, &crate::backend::Device::Cpu)?.requires_grad_(true) })
    }
}

impl Module for Snake {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let a = self.alpha.to_dtype(x.dtype())?;
        x.add(&x.mul(&a)?.sin()?.square()?.div(&a)?)
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![("alpha".to_string(), self.alpha.clone())]
    }
}

/// Swish `x * sigmoid(beta x)` with a learnable scalar `beta` (`beta = 1` is SiLU).
pub struct Swish {
    pub beta: Tensor,
}

impl Swish {
    pub fn new(beta: f64) -> Result<Self> {
        Ok(Self { beta: Tensor::full(&[1], beta, crate::dtype::DType::F32, &crate::backend::Device::Cpu)?.requires_grad_(true) })
    }
}

impl Module for Swish {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.mul(&x.mul(&self.beta.to_dtype(x.dtype())?)?.sigmoid()?)
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![("beta".to_string(), self.beta.clone())]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &Tensor, b: &Tensor, tol: f32) {
        let (x, y) = (a.to_vec::<f32>().unwrap(), b.to_vec::<f32>().unwrap());
        assert_eq!(x.len(), y.len());
        assert!(x.iter().zip(&y).all(|(p, q)| (p - q).abs() < tol), "{x:?} vs {y:?}");
    }

    #[test]
    fn masked_rnns_equal_running_each_sequence_alone() {
        let lstm = LSTM::new(3, 4, 2).unwrap();
        let gru = GRU::new(3, 4, 1).unwrap();
        let x = Tensor::randn(&[3, 6, 3]).unwrap();
        let lengths = [6usize, 2, 4];
        let (out, last) = lstm_masked(&lstm, &x, &lengths).unwrap();
        let (gout, glast) = gru_masked(&gru, &x, &lengths).unwrap();
        for (b, &len) in lengths.iter().enumerate() {
            let alone = x.narrow(0, b, 1).unwrap().narrow(1, 0, len).unwrap().contiguous().unwrap();
            let (want, _) = lstm.forward_seq(&alone, None).unwrap();
            close(&out.narrow(0, b, 1).unwrap().narrow(1, 0, len).unwrap().contiguous().unwrap(), &want, 1e-5);
            close(&last.narrow(0, b, 1).unwrap(), &want.narrow(1, len - 1, 1).unwrap().reshape(&[1, 4]).unwrap(), 1e-5);
            // padded steps are exactly zero
            if len < 6 {
                let tail = out.narrow(0, b, 1).unwrap().narrow(1, len, 6 - len).unwrap().contiguous().unwrap();
                assert!(tail.to_vec::<f32>().unwrap().iter().all(|&v| v == 0.0));
            }
            let (gwant, _) = gru.forward_seq(&alone, None).unwrap();
            close(&gout.narrow(0, b, 1).unwrap().narrow(1, 0, len).unwrap().contiguous().unwrap(), &gwant, 1e-5);
            close(&glast.narrow(0, b, 1).unwrap(), &gwant.narrow(1, len - 1, 1).unwrap().reshape(&[1, 4]).unwrap(), 1e-5);
        }
        assert!(lstm_masked(&lstm, &x, &[6, 7, 1]).is_err() && lstm_masked(&lstm, &x, &[6, 0, 1]).is_err() && lstm_masked(&lstm, &x, &[6]).is_err());
    }

    #[test]
    fn t5_buckets_follow_the_reference_algorithm() {
        let b = RelativePositionBias::new(32, 128, 4, true).unwrap();
        // bidirectional, 16 buckets per direction, 8 exact: distances 0..7 are exact
        for d in 0..8i64 {
            assert_eq!(b.bucket(-d), d as usize);
            if d > 0 {
                assert_eq!(b.bucket(d), 16 + d as usize);
            }
        }
        // far distances saturate at the last bucket of each half
        assert_eq!(b.bucket(-10_000), 15);
        assert_eq!(b.bucket(10_000), 31);
        // buckets never decrease with distance
        let mut prev = 0;
        for d in 0..300i64 {
            let k = b.bucket(-d);
            assert!(k >= prev && k < 16);
            prev = k;
        }
        // unidirectional: future positions fall in bucket 0
        let u = RelativePositionBias::new(32, 128, 4, false).unwrap();
        assert_eq!(u.bucket(5), 0);
        assert_eq!(u.bucket(-3), 3);
        assert_eq!(u.bucket(-100_000), 31);
        let bias = b.forward(3, 5).unwrap();
        assert_eq!(bias.shape(), [1, 4, 3, 5]);
        // the bias depends only on key - query
        let v = bias.to_vec::<f32>().unwrap();
        assert_eq!(v[5 + 2], v[1]);
        bias.sum().unwrap().backward().unwrap();
        assert!(b.embedding.grad().is_some());
    }

    #[test]
    fn gated_and_learnable_activations() {
        let x = Tensor::from_slice(&[2.0f32, -1.0, 0.5, 3.0], &[1, 4]).unwrap();
        close(&ReGlu.forward(&x).unwrap(), &Tensor::from_slice(&[2.0f32 * 0.5, -3.0], &[1, 2]).unwrap(), 1e-6);
        let ge = GeGlu.forward(&x).unwrap().to_vec::<f32>().unwrap();
        let gelu = |v: f32| 0.5 * v * (1.0 + libm::erff(v / std::f32::consts::SQRT_2));
        assert!((ge[0] - 2.0 * gelu(0.5)).abs() < 1e-5 && (ge[1] + gelu(3.0)).abs() < 1e-5);
        assert!(GeGlu.forward(&Tensor::zeros_f32(&[1, 3]).unwrap()).is_err());
        close(&SquaredRelu.forward(&x).unwrap(), &Tensor::from_slice(&[4.0f32, 0.0, 0.25, 9.0], &[1, 4]).unwrap(), 1e-6);
        close(&Maxout(2).forward(&x).unwrap(), &Tensor::from_slice(&[2.0f32, 3.0], &[1, 2]).unwrap(), 1e-6);
        assert!(Maxout(3).forward(&x).is_err());
        let snake = Snake::new(4, 1.0).unwrap();
        let y = snake.forward(&x).unwrap().to_vec::<f32>().unwrap();
        assert!((y[0] - (2.0 + 2.0f32.sin().powi(2))).abs() < 1e-5);
        // Swish with beta = 1 is SiLU
        close(&Swish::new(1.0).unwrap().forward(&x).unwrap(), &x.silu().unwrap(), 1e-6);
        // beta and alpha are trained
        Swish::new(2.0).unwrap().forward(&x).unwrap().sum().unwrap().backward().unwrap();
        snake.forward(&x).unwrap().sum().unwrap().backward().unwrap();
        assert!(snake.alpha.grad().is_some());
    }
}
