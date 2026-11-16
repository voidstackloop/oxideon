//! Autograd utilities: [`gradcheck`] / [`gradgradcheck`] (`torch.autograd.gradcheck`), [`inference_mode`], and an
//! anomaly guard ([`check_finite_grads`]) that reports the first parameter whose gradient contains NaN or infinity.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::{Tensor, grad, no_grad};

/// Runs `f` without recording gradients (`torch.inference_mode`; an alias of [`no_grad`] since this library has no
/// separate inference tensors).
pub fn inference_mode<R>(f: impl FnOnce() -> R) -> R {
    no_grad(f)
}

/// Returns an error naming the first tensor in `named` whose gradient holds a non-finite value (`torch.autograd.
/// detect_anomaly`, applied after a backward pass).
pub fn check_finite_grads(named: &[(String, Tensor)]) -> Result<()> {
    for (name, t) in named {
        if let Some(g) = t.grad() {
            if g.to_vec::<f64>()?.iter().any(|v| !v.is_finite()) {
                return Err(Error::Msg(format!("anomaly: the gradient of '{name}' contains NaN or infinity")));
            }
        }
    }
    Ok(())
}

fn to_f64_leaf(t: &Tensor) -> Result<Tensor> {
    Ok(t.detach().to_dtype(DType::F64)?.contiguous()?.requires_grad_(true))
}

/// Compares the analytic gradient of `f` (a function returning a tensor) against central finite differences of the
/// weighted sum `sum(f(x) * w)` for a fixed pseudo-random weight `w`. Inputs are promoted to `f64`.
pub fn gradcheck(f: impl Fn(&[Tensor]) -> Result<Tensor>, inputs: &[Tensor], eps: f64, atol: f64, rtol: f64) -> Result<()> {
    let leaves: Vec<Tensor> = inputs.iter().map(to_f64_leaf).collect::<Result<_>>()?;
    let weights = |out: &Tensor| -> Result<Tensor> {
        let w: Vec<f64> = (0..out.numel()).map(|i| ((i as f64 + 1.0) * 0.7).sin() + 1.1).collect();
        Tensor::from_slice_on(&w, out.shape(), &out.device())?.to_dtype(out.dtype())
    };
    let out = f(&leaves)?;
    let loss = out.mul(&weights(&out)?)?.sum()?;
    let grads = grad(&[loss], &leaves, None, false)?;
    for (k, leaf) in leaves.iter().enumerate() {
        let analytic = match &grads[k] {
            Some(g) => g.to_vec::<f64>()?,
            None => vec![0.0; leaf.numel()],
        };
        let base = leaf.to_vec::<f64>()?;
        for i in 0..base.len() {
            let eval = |delta: f64| -> Result<f64> {
                let mut args: Vec<Tensor> = leaves.iter().map(to_f64_leaf).collect::<Result<_>>()?;
                let mut v = base.clone();
                v[i] += delta;
                args[k] = Tensor::from_slice_on(&v, leaf.shape(), &leaf.device())?.requires_grad_(true);
                let o = f(&args)?;
                o.mul(&weights(&o)?)?.sum()?.item::<f64>()
            };
            let numeric = (eval(eps)? - eval(-eps)?) / (2.0 * eps);
            if (analytic[i] - numeric).abs() > atol + rtol * numeric.abs() {
                return Err(Error::Msg(format!(
                    "gradcheck failed for input {k}, element {i}: analytic {} vs numeric {numeric}",
                    analytic[i]
                )));
            }
        }
    }
    Ok(())
}

/// Second-order check: verifies that the gradient of `f` is itself correctly differentiable by running
/// [`gradcheck`] on the function `inputs -> d(sum(f * w)) / d(inputs)`.
pub fn gradgradcheck(f: impl Fn(&[Tensor]) -> Result<Tensor> + Copy, inputs: &[Tensor], eps: f64, atol: f64, rtol: f64) -> Result<()> {
    let first_order = move |xs: &[Tensor]| -> Result<Tensor> {
        let out = f(xs)?;
        let w: Vec<f64> = (0..out.numel()).map(|i| ((i as f64 + 1.0) * 0.7).sin() + 1.1).collect();
        let w = Tensor::from_slice_on(&w, out.shape(), &out.device())?.to_dtype(out.dtype())?;
        let loss = out.mul(&w)?.sum()?;
        let grads = grad(&[loss], xs, None, true)?;
        let flat: Vec<Tensor> = grads
            .into_iter()
            .zip(xs)
            .map(|(g, x)| match g {
                Some(g) => g.reshape(&[x.numel()]),
                None => Tensor::zeros(&[x.numel()], x.dtype(), &x.device()),
            })
            .collect::<Result<_>>()?;
        Tensor::cat(&flat, 0)
    };
    gradcheck(first_order, inputs, eps, atol, rtol)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(v: &[f64]) -> Tensor {
        Tensor::from_slice(v, &[v.len()]).unwrap()
    }

    #[test]
    fn accepts_correct_gradients_and_rejects_wrong_ones() {
        let xs = [t(&[0.3, -1.2, 2.0]), t(&[1.5, 0.4, -0.7])];
        gradcheck(|a| a[0].mul(&a[1])?.tanh(), &xs, 1e-6, 1e-6, 1e-5).unwrap();
        gradcheck(|a| a[0].lgamma(), &[t(&[0.7, 2.2])], 1e-6, 1e-6, 1e-5).unwrap();
        // an op with a deliberately wrong backward is caught: stop-gradient through detach
        let bad = gradcheck(|a| a[0].detach().mul(&a[0])?.add(&a[0].detach().square()?), &[t(&[0.5, 1.5])], 1e-6, 1e-6, 1e-5);
        assert!(bad.is_err(), "the detached branch makes the analytic gradient wrong");
    }

    #[test]
    fn second_order() {
        gradgradcheck(|a| a[0].sin()?.mul(&a[0].exp()?), &[t(&[0.4, -0.9])], 1e-5, 1e-4, 1e-4).unwrap();
        gradgradcheck(|a| a[0].polygamma(1), &[t(&[1.3, 2.4])], 1e-5, 1e-4, 1e-4).unwrap();
    }

    #[test]
    fn anomaly_guard_and_inference_mode() {
        let p = t(&[1.0]).requires_grad_(true);
        p.set_grad(Some(t(&[f64::NAN])));
        let err = check_finite_grads(&[("w".to_string(), p.clone())]).unwrap_err();
        assert!(err.to_string().contains("'w'"));
        p.set_grad(Some(t(&[1.0])));
        check_finite_grads(&[("w".to_string(), p.clone())]).unwrap();
        let y = inference_mode(|| p.mul_scalar(2.0).unwrap());
        assert!(!y.requires_grad());
    }
}
