//! Element-wise update kernels for the optimizers whose Python classes keep their state as tensors
//! (`oxideon.optim.RMSprop`, `Adagrad`, `Adadelta`, `Adamax`, `NAdam`, `RAdam`, `ASGD`, `Lion`, `Lamb`, `Adafactor`,
//! `Rprop`). Python keeps the per-parameter scalars (step counts, schedules) and calls [`optim_step`] once per
//! parameter; all per-element arithmetic happens here, in `f64`, with the results returned in the parameter's dtype.
//! The formulas follow `torch.optim` where PyTorch has the optimizer.

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::tensor::Tensor;

fn get(h: &HashMap<String, f64>, k: &str) -> Result<f64> {
    h.get(k).copied().ok_or_else(|| Error::Msg(format!("optim_step: missing hyper-parameter '{k}'")))
}

fn opt(h: &HashMap<String, f64>, k: &str, default: f64) -> f64 {
    h.get(k).copied().unwrap_or(default)
}

fn l2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// One optimizer step for one parameter. `states` are the algorithm's state tensors (see each arm), `h` its
/// hyper-parameters and the scalars Python tracks. Returns the new parameter and the new states, in order.
pub fn optim_step(
    kind: &str,
    param: &Tensor,
    grad: &Tensor,
    states: &[Tensor],
    h: &HashMap<String, f64>,
) -> Result<(Tensor, Vec<Tensor>)> {
    let shape = param.shape().to_vec();
    if grad.shape() != shape.as_slice() {
        return Err(Error::Msg(format!("optim_step: grad shape {:?} != param shape {shape:?}", grad.shape())));
    }
    let mut p = param.to_vec::<f64>()?;
    let mut g = grad.to_vec::<f64>()?;
    let mut s: Vec<Vec<f64>> = states.iter().map(|t| t.to_vec::<f64>()).collect::<Result<_>>()?;
    let n = p.len();
    let wd = opt(h, "weight_decay", 0.0);
    let need = |count: usize| -> Result<()> {
        if states.len() < count {
            return Err(Error::Msg(format!("optim_step({kind}): expected {count} state tensors, got {}", states.len())));
        }
        Ok(())
    };
    // L2-style weight decay (added to the gradient), used by the torch algorithms below
    let couple_wd = |g: &mut [f64], p: &[f64]| {
        if wd != 0.0 {
            for (gi, pi) in g.iter_mut().zip(p) {
                *gi += wd * pi;
            }
        }
    };
    match kind {
        // states: square_avg, momentum_buffer, grad_avg
        "rmsprop" => {
            need(3)?;
            couple_wd(&mut g, &p);
            let (lr, alpha, eps, momentum) = (get(h, "lr")?, get(h, "alpha")?, get(h, "eps")?, opt(h, "momentum", 0.0));
            let centered = opt(h, "centered", 0.0) != 0.0;
            let (sq, rest) = s.split_at_mut(1);
            let (buf, ga) = rest.split_at_mut(1);
            let (sq, buf, ga) = (&mut sq[0], &mut buf[0], &mut ga[0]);
            for i in 0..n {
                sq[i] = alpha * sq[i] + (1.0 - alpha) * g[i] * g[i];
                let avg = if centered {
                    ga[i] = alpha * ga[i] + (1.0 - alpha) * g[i];
                    (sq[i] - ga[i] * ga[i]).sqrt() + eps
                } else {
                    sq[i].sqrt() + eps
                };
                if momentum > 0.0 {
                    buf[i] = momentum * buf[i] + g[i] / avg;
                    p[i] -= lr * buf[i];
                } else {
                    p[i] -= lr * g[i] / avg;
                }
            }
        }
        // states: sum; h: clr (the decayed learning rate)
        "adagrad" => {
            need(1)?;
            couple_wd(&mut g, &p);
            let (clr, eps) = (get(h, "clr")?, get(h, "eps")?);
            let sum = &mut s[0];
            for i in 0..n {
                sum[i] += g[i] * g[i];
                p[i] -= clr * g[i] / (sum[i].sqrt() + eps);
            }
        }
        // states: square_avg, acc_delta
        "adadelta" => {
            need(2)?;
            couple_wd(&mut g, &p);
            let (lr, rho, eps) = (get(h, "lr")?, get(h, "rho")?, get(h, "eps")?);
            let (a, b) = s.split_at_mut(1);
            let (sq, acc) = (&mut a[0], &mut b[0]);
            for i in 0..n {
                sq[i] = rho * sq[i] + (1.0 - rho) * g[i] * g[i];
                let delta = (acc[i] + eps).sqrt() / (sq[i] + eps).sqrt() * g[i];
                acc[i] = rho * acc[i] + (1.0 - rho) * delta * delta;
                p[i] -= lr * delta;
            }
        }
        // states: exp_avg, exp_inf; h: step
        "adamax" => {
            need(2)?;
            couple_wd(&mut g, &p);
            let (lr, b1, b2, eps, step) = (get(h, "lr")?, get(h, "beta1")?, get(h, "beta2")?, get(h, "eps")?, get(h, "step")?);
            let clr = lr / (1.0 - b1.powf(step));
            let (a, b) = s.split_at_mut(1);
            let (m, u) = (&mut a[0], &mut b[0]);
            for i in 0..n {
                m[i] = b1 * m[i] + (1.0 - b1) * g[i];
                u[i] = (b2 * u[i]).max(g[i].abs() + eps);
                p[i] -= clr * m[i] / u[i];
            }
        }
        // states: exp_avg, exp_avg_sq; h: step, mu, mu_next, mu_product (already including mu)
        "nadam" => {
            need(2)?;
            couple_wd(&mut g, &p);
            let (lr, b1, b2, eps, step) = (get(h, "lr")?, get(h, "beta1")?, get(h, "beta2")?, get(h, "eps")?, get(h, "step")?);
            let (mu, mu_next, prod) = (get(h, "mu")?, get(h, "mu_next")?, get(h, "mu_product")?);
            let bc2 = 1.0 - b2.powf(step);
            let (a, b) = s.split_at_mut(1);
            let (m, v) = (&mut a[0], &mut b[0]);
            for i in 0..n {
                m[i] = b1 * m[i] + (1.0 - b1) * g[i];
                v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
                let denom = (v[i] / bc2).sqrt() + eps;
                p[i] -= lr * (1.0 - mu) / (1.0 - prod) * g[i] / denom;
                p[i] -= lr * mu_next / (1.0 - prod * mu_next) * m[i] / denom;
            }
        }
        // states: exp_avg, exp_avg_sq; h: step
        "radam" => {
            need(2)?;
            couple_wd(&mut g, &p);
            let (lr, b1, b2, eps, step) = (get(h, "lr")?, get(h, "beta1")?, get(h, "beta2")?, get(h, "eps")?, get(h, "step")?);
            let (bc1, bc2) = (1.0 - b1.powf(step), 1.0 - b2.powf(step));
            let rho_inf = 2.0 / (1.0 - b2) - 1.0;
            let rho_t = rho_inf - 2.0 * step * b2.powf(step) / bc2;
            let rect = if rho_t > 5.0 {
                Some(((rho_t - 4.0) * (rho_t - 2.0) * rho_inf / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)).sqrt())
            } else {
                None
            };
            let (a, b) = s.split_at_mut(1);
            let (m, v) = (&mut a[0], &mut b[0]);
            for i in 0..n {
                m[i] = b1 * m[i] + (1.0 - b1) * g[i];
                v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
                let m_hat = m[i] / bc1;
                p[i] -= match rect {
                    Some(r) => m_hat * lr * (bc2.sqrt() / (v[i].sqrt() + eps)) * r,
                    None => m_hat * lr,
                };
            }
        }
        // states: ax; h: eta, mu, lambd (the step's eta / mu, before their update)
        "asgd" => {
            need(1)?;
            couple_wd(&mut g, &p);
            let (eta, mu, lambd) = (get(h, "eta")?, get(h, "mu")?, get(h, "lambd")?);
            let ax = &mut s[0];
            for i in 0..n {
                p[i] *= 1.0 - lambd * eta;
                p[i] -= eta * g[i];
                if mu != 1.0 {
                    ax[i] += (p[i] - ax[i]) * mu;
                } else {
                    ax[i] = p[i];
                }
            }
        }
        // states: exp_avg (decoupled weight decay, sign update)
        "lion" => {
            need(1)?;
            let (lr, b1, b2) = (get(h, "lr")?, get(h, "beta1")?, get(h, "beta2")?);
            let m = &mut s[0];
            for i in 0..n {
                if wd != 0.0 {
                    p[i] -= lr * wd * p[i];
                }
                let c = b1 * m[i] + (1.0 - b1) * g[i];
                let sgn = if c > 0.0 { 1.0 } else if c < 0.0 { -1.0 } else { 0.0 };
                p[i] -= lr * sgn;
                m[i] = b2 * m[i] + (1.0 - b2) * g[i];
            }
        }
        // states: exp_avg, exp_avg_sq; h: step (layer-wise trust ratio)
        "lamb" => {
            need(2)?;
            let (lr, b1, b2, eps, step) = (get(h, "lr")?, get(h, "beta1")?, get(h, "beta2")?, get(h, "eps")?, get(h, "step")?);
            let (bc1, bc2) = (1.0 - b1.powf(step), 1.0 - b2.powf(step));
            let (a, b) = s.split_at_mut(1);
            let (m, v) = (&mut a[0], &mut b[0]);
            let mut upd = vec![0.0; n];
            for i in 0..n {
                m[i] = b1 * m[i] + (1.0 - b1) * g[i];
                v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
                upd[i] = (m[i] / bc1) / ((v[i] / bc2).sqrt() + eps);
                if wd > 0.0 {
                    upd[i] += wd * p[i];
                }
            }
            let (wn, rn) = (l2(&p), l2(&upd));
            let trust = if wn > 0.0 && rn > 0.0 { wn / rn } else { 1.0 };
            for i in 0..n {
                p[i] -= lr * trust * upd[i];
            }
        }
        // states: (exp_avg_sq_row, exp_avg_sq_col) for >= 2-D parameters or (exp_avg_sq) for vectors, then exp_avg
        // when beta1 >= 0; h: beta2 (this step's), beta1 (< 0 for none), eps1, eps2, clip_threshold
        "adafactor" => {
            let (lr, beta2, eps1, eps2, clip) = (get(h, "lr")?, get(h, "beta2")?, get(h, "eps1")?, get(h, "eps2")?, get(h, "clip_threshold")?);
            let beta1 = opt(h, "beta1", -1.0);
            let factored = shape.len() > 1;
            let base = if factored { 2 } else { 1 };
            need(base + usize::from(beta1 >= 0.0))?;
            let gsq: Vec<f64> = g.iter().map(|x| x * x + eps1).collect();
            let mut v = vec![0.0; n];
            if factored {
                let c = shape[shape.len() - 1];
                let r = shape[shape.len() - 2];
                let batches = n / (r * c).max(1);
                let (rows, cols) = s.split_at_mut(1);
                let (row, col) = (&mut rows[0], &mut cols[0]);
                for bi in 0..batches {
                    let off = bi * r * c;
                    for ri in 0..r {
                        let mean: f64 = gsq[off + ri * c..off + (ri + 1) * c].iter().sum::<f64>() / c as f64;
                        row[bi * r + ri] = beta2 * row[bi * r + ri] + (1.0 - beta2) * mean;
                    }
                    for ci in 0..c {
                        let mean: f64 = (0..r).map(|ri| gsq[off + ri * c + ci]).sum::<f64>() / r as f64;
                        col[bi * c + ci] = beta2 * col[bi * c + ci] + (1.0 - beta2) * mean;
                    }
                    let row_mean = (row[bi * r..(bi + 1) * r].iter().sum::<f64>() / r as f64).max(1e-12);
                    for ri in 0..r {
                        for ci in 0..c {
                            v[off + ri * c + ci] = row[bi * r + ri] / row_mean * col[bi * c + ci];
                        }
                    }
                }
            } else {
                let sq = &mut s[0];
                for i in 0..n {
                    sq[i] = beta2 * sq[i] + (1.0 - beta2) * gsq[i];
                    v[i] = sq[i];
                }
            }
            let mut u: Vec<f64> = (0..n).map(|i| g[i] / (v[i].sqrt() + eps2)).collect();
            let un = l2(&u);
            if clip > 0.0 && un > clip {
                u.iter_mut().for_each(|x| *x *= clip / un);
            }
            if beta1 >= 0.0 {
                let m = &mut s[base];
                for i in 0..n {
                    m[i] = beta1 * m[i] + (1.0 - beta1) * u[i];
                    u[i] = m[i];
                }
            }
            for i in 0..n {
                if wd > 0.0 {
                    p[i] -= lr * wd * p[i];
                }
                p[i] -= lr * u[i];
            }
        }
        // states: prev, step_size
        "rprop" => {
            need(2)?;
            let (eta_minus, eta_plus) = (get(h, "eta_minus")?, get(h, "eta_plus")?);
            let (smin, smax) = (get(h, "step_size_min")?, get(h, "step_size_max")?);
            let (a, b) = s.split_at_mut(1);
            let (prev, size) = (&mut a[0], &mut b[0]);
            for i in 0..n {
                let prod = g[i] * prev[i];
                let factor = if prod > 0.0 { eta_plus } else if prod < 0.0 { eta_minus } else { 1.0 };
                size[i] = (size[i] * factor).clamp(smin, smax);
                let gi = if prod < 0.0 { 0.0 } else { g[i] };
                let sgn = if gi > 0.0 { 1.0 } else if gi < 0.0 { -1.0 } else { 0.0 };
                p[i] -= sgn * size[i];
                prev[i] = gi;
            }
        }
        other => return Err(Error::Msg(format!("optim_step: unknown optimizer '{other}'"))),
    }
    let dev = param.device();
    let dt = param.dtype();
    let new_p = Tensor::from_slice_on(&p, &shape, &dev)?.to_dtype(dt)?;
    let new_states = states
        .iter()
        .zip(&s)
        .map(|(t, v)| Tensor::from_slice_on(v, t.shape(), &dev)?.to_dtype(t.dtype()))
        .collect::<Result<_>>()?;
    Ok((new_p, new_states))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn rmsprop_matches_the_formula_and_errors_are_reported() {
        let p = Tensor::from_slice(&[1.0f64, -2.0], &[2]).unwrap();
        let g = Tensor::from_slice(&[0.5f64, 0.25], &[2]).unwrap();
        let z = p.zeros_like().unwrap();
        let (np, st) = optim_step("rmsprop", &p, &g, &[z.clone(), z.clone(), z.clone()], &h(&[("lr", 0.1), ("alpha", 0.9), ("eps", 1e-8)])).unwrap();
        let sq: Vec<f64> = st[0].to_vec().unwrap();
        assert!((sq[0] - 0.025).abs() < 1e-15);
        let want = 1.0 - 0.1 * 0.5 / (0.025f64.sqrt() + 1e-8);
        assert!((np.to_vec::<f64>().unwrap()[0] - want).abs() < 1e-12);
        assert!(optim_step("rmsprop", &p, &g, std::slice::from_ref(&z), &h(&[("lr", 0.1)])).is_err());
        assert!(optim_step("nope", &p, &g, &[], &h(&[])).is_err());
    }

    #[test]
    fn rprop_clamps_and_zeroes_sign_flips() {
        let p = Tensor::from_slice(&[0.0f64, 0.0], &[2]).unwrap();
        let prev = Tensor::from_slice(&[1.0f64, 1.0], &[2]).unwrap();
        let size = Tensor::from_slice(&[0.1f64, 0.1], &[2]).unwrap();
        let g = Tensor::from_slice(&[2.0f64, -3.0], &[2]).unwrap();
        let hp = h(&[("eta_minus", 0.5), ("eta_plus", 1.2), ("step_size_min", 1e-6), ("step_size_max", 50.0)]);
        let (np, st) = optim_step("rprop", &p, &g, &[prev, size], &hp).unwrap();
        let v = np.to_vec::<f64>().unwrap();
        assert!((v[0] + 0.12).abs() < 1e-12 && v[1] == 0.0);
        assert_eq!(st[0].to_vec::<f64>().unwrap(), vec![2.0, 0.0]);
    }
}
