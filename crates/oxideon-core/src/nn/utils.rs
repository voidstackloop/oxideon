//! `torch.nn.utils`-style helpers: [`parameters_to_vector`] / [`vector_to_parameters`], gradient clipping with any
//! norm type ([`clip_grad_norm_p`], [`get_total_norm`]), exponential moving averages ([`Ema`]) and stochastic weight
//! averaging ([`Swa`]), [`pad_sequence`], and magnitude / random pruning masks ([`prune_l1`], [`prune_random`],
//! [`prune_global_l1`]).

use crate::error::{Error, Result};
use crate::rng;
use crate::tensor::{Tensor, no_grad};

/// Concatenates all parameters into one flat vector.
pub fn parameters_to_vector(params: &[Tensor]) -> Result<Tensor> {
    if params.is_empty() {
        return Err(Error::Msg("parameters_to_vector: no parameters".into()));
    }
    let flat: Vec<Tensor> = params
        .iter()
        .map(|p| p.reshape(&[p.numel()]))
        .collect::<Result<_>>()?;
    Tensor::cat(&flat, 0)
}

/// Copies slices of `vec` back into the parameters (the inverse of [`parameters_to_vector`]).
pub fn vector_to_parameters(vec: &Tensor, params: &[Tensor]) -> Result<()> {
    let total: usize = params.iter().map(Tensor::numel).sum();
    if vec.ndim() != 1 || vec.numel() != total {
        return Err(Error::Msg(format!("vector_to_parameters: expected a vector of {total} values, got shape {:?}", vec.shape())));
    }
    no_grad(|| {
        let mut at = 0;
        for p in params {
            let n = p.numel();
            let chunk = vec.narrow(0, at, n)?.reshape(p.shape())?.to_dtype(p.dtype())?;
            p.assign(&chunk)?;
            at += n;
        }
        Ok(())
    })
}

/// The `norm_type`-norm of all gradients taken together (`inf` for the maximum absolute value).
pub fn get_total_norm(grads: &[Tensor], norm_type: f64) -> Result<f64> {
    if grads.is_empty() {
        return Ok(0.0);
    }
    no_grad(|| {
        let mut per = Vec::with_capacity(grads.len());
        for g in grads {
            let n = if norm_type.is_infinite() {
                g.abs()?.amax(&[], false)?.item::<f64>()?
            } else {
                g.vector_norm(norm_type, &[], false)?.item::<f64>()?
            };
            per.push(n);
        }
        Ok(if norm_type.is_infinite() {
            per.into_iter().fold(0.0, f64::max)
        } else {
            per.iter().map(|n| n.powf(norm_type)).sum::<f64>().powf(1.0 / norm_type)
        })
    })
}

/// Scales gradients so their total `norm_type`-norm is at most `max_norm`; returns the norm before clipping
/// (`torch.nn.utils.clip_grad_norm_`). With `error_if_nonfinite` a NaN / infinite norm is an error.
pub fn clip_grad_norm_p(params: &[Tensor], max_norm: f64, norm_type: f64, error_if_nonfinite: bool) -> Result<f64> {
    if max_norm.is_nan() || max_norm < 0.0 || norm_type <= 0.0 || norm_type.is_nan() {
        return Err(Error::Msg("clip_grad_norm_p: max_norm must be >= 0 and norm_type > 0".into()));
    }
    let grads: Vec<Tensor> = params.iter().filter_map(Tensor::grad).collect();
    let total = get_total_norm(&grads, norm_type)?;
    if error_if_nonfinite && !total.is_finite() {
        return Err(Error::Msg(format!("clip_grad_norm_p: the total gradient norm is {total}")));
    }
    let coef = (max_norm / (total + 1e-6)).min(1.0);
    if coef < 1.0 {
        no_grad(|| -> Result<()> {
            for p in params {
                if let Some(g) = p.grad() {
                    p.set_grad(Some(g.mul_scalar(coef)?));
                }
            }
            Ok(())
        })?;
    }
    Ok(total)
}

/// Exponential moving average of parameters. [`update`](Self::update) after each optimizer step;
/// [`apply`](Self::apply) swaps the averages into the model for evaluation and [`restore`](Self::restore) swaps the
/// training values back.
pub struct Ema {
    params: Vec<Tensor>,
    shadow: Vec<Tensor>,
    backup: Option<Vec<Tensor>>,
    pub decay: f64,
    /// Ramp the decay up from 0 as `min(decay, (1 + n) / (10 + n))` during the first updates.
    pub warmup: bool,
    updates: usize,
}

impl Ema {
    pub fn new(params: Vec<Tensor>, decay: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&decay) {
            return Err(Error::Msg(format!("Ema: decay must lie in [0, 1], got {decay}")));
        }
        let shadow = params.iter().map(|p| p.detach().contiguous().and_then(|c| c.mul_scalar(1.0))).collect::<Result<_>>()?;
        Ok(Self { params, shadow, backup: None, decay, warmup: false, updates: 0 })
    }
    pub fn update(&mut self) -> Result<()> {
        self.updates += 1;
        let d = if self.warmup {
            self.decay.min((1.0 + self.updates as f64) / (10.0 + self.updates as f64))
        } else {
            self.decay
        };
        no_grad(|| {
            for (s, p) in self.shadow.iter_mut().zip(&self.params) {
                *s = s.mul_scalar(d)?.add(&p.detach().mul_scalar(1.0 - d)?)?;
            }
            Ok(())
        })
    }
    /// Loads the averaged values into the parameters, remembering the current ones.
    pub fn apply(&mut self) -> Result<()> {
        if self.backup.is_some() {
            return Err(Error::Msg("Ema::apply: averages are already applied (call restore first)".into()));
        }
        let backup = self.params.iter().map(|p| p.detach().mul_scalar(1.0)).collect::<Result<Vec<_>>>()?;
        no_grad(|| {
            for (p, s) in self.params.iter().zip(&self.shadow) {
                p.assign(s)?;
            }
            Ok::<(), Error>(())
        })?;
        self.backup = Some(backup);
        Ok(())
    }
    pub fn restore(&mut self) -> Result<()> {
        let backup = self.backup.take().ok_or_else(|| Error::Msg("Ema::restore: nothing to restore".into()))?;
        no_grad(|| {
            for (p, b) in self.params.iter().zip(&backup) {
                p.assign(b)?;
            }
            Ok(())
        })
    }
    pub fn shadow(&self) -> &[Tensor] {
        &self.shadow
    }
}

/// Stochastic weight averaging: the running arithmetic mean of the parameters over the calls to
/// [`update`](Self::update).
pub struct Swa {
    params: Vec<Tensor>,
    average: Vec<Tensor>,
    count: usize,
}

impl Swa {
    pub fn new(params: Vec<Tensor>) -> Result<Self> {
        let average = params.iter().map(|p| p.zeros_like()).collect::<Result<_>>()?;
        Ok(Self { params, average, count: 0 })
    }
    pub fn update(&mut self) -> Result<()> {
        no_grad(|| {
            for (a, p) in self.average.iter_mut().zip(&self.params) {
                *a = a.add(&p.detach().sub(a)?.div_scalar((self.count + 1) as f64)?)?;
            }
            Ok::<(), Error>(())
        })?;
        self.count += 1;
        Ok(())
    }
    pub fn average(&self) -> &[Tensor] {
        &self.average
    }
    pub fn num_averaged(&self) -> usize {
        self.count
    }
    /// Writes the averaged weights into the parameters.
    pub fn apply(&self) -> Result<()> {
        no_grad(|| {
            for (p, a) in self.params.iter().zip(&self.average) {
                p.assign(a)?;
            }
            Ok(())
        })
    }
}

/// Stacks variable-length sequences `[len_i, ...]` into one tensor, padding with `padding_value`
/// (`torch.nn.utils.rnn.pad_sequence`). Output is `[batch, max_len, ...]` with `batch_first`, else
/// `[max_len, batch, ...]`.
pub fn pad_sequence(seqs: &[Tensor], batch_first: bool, padding_value: f64) -> Result<Tensor> {
    if seqs.is_empty() {
        return Err(Error::Msg("pad_sequence: no sequences".into()));
    }
    let max_len = seqs.iter().map(|s| s.shape()[0]).max().unwrap_or(0);
    let padded: Vec<Tensor> = seqs
        .iter()
        .map(|s| {
            let mut pads = vec![(0, 0); s.ndim()];
            pads[0] = (0, max_len - s.shape()[0]);
            s.pad(&pads, padding_value)
        })
        .collect::<Result<_>>()?;
    Tensor::stack(&padded, if batch_first { 0 } else { 1 })
}

/// Zeroes the `amount` fraction of smallest-magnitude entries of `param` in place and returns the 0/1 mask.
pub fn prune_l1(param: &Tensor, amount: f64) -> Result<Tensor> {
    prune_global_l1(std::slice::from_ref(param), amount).map(|mut m| m.remove(0))
}

/// Zeroes the `amount` fraction of smallest-magnitude entries over **all** `params` together; returns one mask each.
pub fn prune_global_l1(params: &[Tensor], amount: f64) -> Result<Vec<Tensor>> {
    if !(0.0..=1.0).contains(&amount) {
        return Err(Error::Msg(format!("prune: amount must lie in [0, 1], got {amount}")));
    }
    let mags: Vec<Vec<f64>> = params.iter().map(|p| p.to_vec::<f64>().map(|v| v.into_iter().map(f64::abs).collect())).collect::<Result<_>>()?;
    let mut all: Vec<f64> = mags.iter().flatten().copied().collect();
    let k = (amount * all.len() as f64).round() as usize;
    all.sort_by(f64::total_cmp);
    let cutoff = if k == 0 { f64::NEG_INFINITY } else { all[k - 1] };
    // ties at the cutoff are pruned only until exactly k entries are removed
    let mut budget = k;
    let mut out = Vec::with_capacity(params.len());
    for (p, m) in params.iter().zip(&mags) {
        let mask: Vec<f64> = m
            .iter()
            .map(|&v| {
                if budget > 0 && v <= cutoff {
                    budget -= 1;
                    0.0
                } else {
                    1.0
                }
            })
            .collect();
        let mask = Tensor::from_slice_on(&mask, p.shape(), &p.device())?.to_dtype(p.dtype())?;
        no_grad(|| p.assign(&p.mul(&mask)?))?;
        out.push(mask);
    }
    Ok(out)
}

/// Zeroes a random `amount` fraction of the entries of `param` in place and returns the mask.
pub fn prune_random(param: &Tensor, amount: f64) -> Result<Tensor> {
    if !(0.0..=1.0).contains(&amount) {
        return Err(Error::Msg(format!("prune: amount must lie in [0, 1], got {amount}")));
    }
    let n = param.numel();
    let k = (amount * n as f64).round() as usize;
    let mut idx: Vec<usize> = (0..n).collect();
    rng::with_rng(|r| r.shuffle(&mut idx));
    let mut mask = vec![1.0f64; n];
    for &i in &idx[..k] {
        mask[i] = 0.0;
    }
    let mask = Tensor::from_slice_on(&mask, param.shape(), &param.device())?.to_dtype(param.dtype())?;
    no_grad(|| param.assign(&param.mul(&mask)?))?;
    Ok(mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(v: &[f64], shape: &[usize]) -> Tensor {
        Tensor::from_slice(v, shape).unwrap().requires_grad_(true)
    }
    fn vals(t: &Tensor) -> Vec<f64> {
        t.to_vec::<f64>().unwrap()
    }

    #[test]
    fn vector_round_trip() {
        let (a, b) = (p(&[1.0, 2.0], &[2]), p(&[3.0, 4.0, 5.0, 6.0], &[2, 2]));
        let v = parameters_to_vector(&[a.clone(), b.clone()]).unwrap();
        assert_eq!(vals(&v), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        vector_to_parameters(&v.mul_scalar(10.0).unwrap(), &[a.clone(), b.clone()]).unwrap();
        assert_eq!(vals(&a), vec![10.0, 20.0]);
        assert_eq!(vals(&b), vec![30.0, 40.0, 50.0, 60.0]);
        assert!(vector_to_parameters(&v.narrow(0, 0, 3).unwrap(), &[a, b]).is_err());
    }

    #[test]
    fn gradient_clipping_norm_types() {
        let a = p(&[0.0, 0.0], &[2]);
        let b = p(&[0.0], &[1]);
        a.set_grad(Some(Tensor::from_slice(&[3.0, 4.0], &[2]).unwrap()));
        b.set_grad(Some(Tensor::from_slice(&[12.0], &[1]).unwrap()));
        assert!((get_total_norm(&[a.grad().unwrap(), b.grad().unwrap()], 2.0).unwrap() - 13.0).abs() < 1e-9);
        assert_eq!(get_total_norm(&[a.grad().unwrap(), b.grad().unwrap()], f64::INFINITY).unwrap(), 12.0);
        let before = clip_grad_norm_p(&[a.clone(), b.clone()], 6.5, 2.0, true).unwrap();
        assert!((before - 13.0).abs() < 1e-9);
        let g = vals(&b.grad().unwrap())[0];
        assert!((g - 6.0).abs() < 1e-4, "{g}");
        // inf-norm clipping caps the largest entry
        clip_grad_norm_p(std::slice::from_ref(&a), 1.0, f64::INFINITY, true).unwrap();
        assert!(vals(&a.grad().unwrap()).iter().all(|v| v.abs() <= 1.0 + 1e-5));
        // a NaN gradient is reported when asked
        a.set_grad(Some(Tensor::from_slice(&[f64::NAN, 1.0], &[2]).unwrap()));
        assert!(clip_grad_norm_p(&[a], 1.0, 2.0, true).is_err());
    }

    #[test]
    fn ema_and_swa() {
        let w = p(&[1.0], &[1]);
        let mut ema = Ema::new(vec![w.clone()], 0.5).unwrap();
        w.assign(&Tensor::from_slice(&[3.0], &[1]).unwrap()).unwrap();
        ema.update().unwrap();
        assert_eq!(vals(&ema.shadow()[0]), vec![2.0]);
        ema.apply().unwrap();
        assert_eq!(vals(&w), vec![2.0]);
        assert!(ema.apply().is_err());
        ema.restore().unwrap();
        assert_eq!(vals(&w), vec![3.0]);
        assert!(ema.restore().is_err());
        let mut swa = Swa::new(vec![w.clone()]).unwrap();
        for v in [3.0, 5.0, 10.0] {
            w.assign(&Tensor::from_slice(&[v], &[1]).unwrap()).unwrap();
            swa.update().unwrap();
        }
        assert_eq!(swa.num_averaged(), 3);
        assert!((vals(&swa.average()[0])[0] - 6.0).abs() < 1e-9);
        swa.apply().unwrap();
        assert!((vals(&w)[0] - 6.0).abs() < 1e-9);
    }

    #[test]
    fn padding_and_pruning() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let b = Tensor::from_slice(&[4.0], &[1]).unwrap();
        assert_eq!(vals(&pad_sequence(&[a.clone(), b.clone()], true, 0.0).unwrap()), vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0]);
        let tf = pad_sequence(&[a, b], false, -1.0).unwrap();
        assert_eq!(tf.shape(), [3, 2]);
        assert_eq!(vals(&tf), vec![1.0, 4.0, 2.0, -1.0, 3.0, -1.0]);
        let w = p(&[0.1, -5.0, 0.3, 2.0], &[4]);
        let mask = prune_l1(&w, 0.5).unwrap();
        assert_eq!(vals(&mask), vec![0.0, 1.0, 0.0, 1.0]);
        assert_eq!(vals(&w), vec![0.0, -5.0, 0.0, 2.0]);
        let (u, v) = (p(&[1.0, 2.0], &[2]), p(&[0.1, 3.0], &[2]));
        let masks = prune_global_l1(&[u.clone(), v.clone()], 0.5).unwrap();
        assert_eq!(vals(&masks[0]), [1.0, 1.0].iter().enumerate().map(|(i, _)| if i == 0 { 0.0 } else { 1.0 }).collect::<Vec<_>>());
        assert_eq!(vals(&v), vec![0.0, 3.0]);
        rng::manual_seed(4);
        let r = p(&[1.0; 10], &[10]);
        prune_random(&r, 0.3).unwrap();
        assert_eq!(vals(&r).iter().filter(|&&x| x == 0.0).count(), 3);
        assert!(prune_l1(&r, 1.5).is_err());
    }
}

#[cfg(test)]
mod buffer_tests {
    use super::super::{BatchNorm1d, BatchNorm2d, Linear, Module, Sequential};
    use super::super::conv3d::BatchNorm3d;
    use crate::tensor::Tensor;

    #[test]
    fn batch_norm_statistics_survive_a_state_dict_round_trip() {
        let build = || Sequential::new().add(Linear::new(3, 4).unwrap()).add(BatchNorm1d::new(4).unwrap());
        let a = build();
        a.train(true);
        for _ in 0..5 {
            a.forward(&Tensor::randn(&[8, 3]).unwrap().mul_scalar(3.0).unwrap().add_scalar(1.0).unwrap()).unwrap();
        }
        let state = a.state_dict();
        assert!(state.contains_key("1.running_mean") && state.contains_key("1.running_var"), "{:?}", state.keys());
        let b = build();
        b.load_state_dict(&state).unwrap();
        a.eval();
        b.eval();
        let x = Tensor::randn(&[4, 3]).unwrap();
        let (ya, yb) = (a.forward(&x).unwrap().to_vec::<f32>().unwrap(), b.forward(&x).unwrap().to_vec::<f32>().unwrap());
        assert_eq!(ya, yb);
        // also through a file
        let path = std::env::temp_dir().join("oxideon_bn_roundtrip.safetensors");
        a.save(&path).unwrap();
        let c = build();
        c.load(&path).unwrap();
        c.eval();
        assert_eq!(c.forward(&x).unwrap().to_vec::<f32>().unwrap(), ya);
        let _ = std::fs::remove_file(&path);
        // a fresh model keeps its own statistics unless the buffers are loaded
        let fresh = build();
        fresh.eval();
        assert_ne!(fresh.forward(&x).unwrap().to_vec::<f32>().unwrap(), ya);
    }

    #[test]
    fn buffers_are_listed_for_every_batch_norm() {
        assert_eq!(BatchNorm2d::new(2).unwrap().named_buffers().len(), 2);
        assert_eq!(BatchNorm3d::new(2, 1e-5, 0.1, crate::dtype::DType::F32, crate::backend::Device::Cpu).unwrap().named_buffers().len(), 2);
        assert!(Linear::new(2, 2).unwrap().named_buffers().is_empty());
        assert_eq!(BatchNorm1d::new(2).unwrap().buffers().len(), 2);
    }
}
