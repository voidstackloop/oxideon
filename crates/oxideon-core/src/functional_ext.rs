//! More functional transforms: `jacfwd` (forward-mode Jacobian from JVPs), `jacrev`, `grad_and_value`, `vmap`
//! (loop-based batching), and `hessian_diag`.

use crate::error::{Error, Result};
use crate::functional::{hessian, jacobian, jvp};
use crate::tensor::{grad, Tensor};

/// Jacobian `[numel(y), numel(x)]` built column by column from forward-mode JVPs with one-hot tangents.
pub fn jacfwd<F>(f: F, x: &Tensor) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    let n = x.numel();
    let eye = Tensor::eye(n, x.dtype(), &x.device())?;
    let mut cols = Vec::with_capacity(n);
    for j in 0..n {
        let v = eye.select(0, j as isize)?.reshape(x.shape())?;
        let (y, t) = jvp(&f, x, &v)?;
        cols.push(t.reshape(&[y.numel()])?);
    }
    Tensor::stack(&cols, 1)
}

/// Reverse-mode Jacobian (alias of [`jacobian`]).
pub fn jacrev<F>(f: F, x: &Tensor) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    jacobian(f, x)
}

/// Gradient of the scalar function `f` at `x` together with its value.
pub fn grad_and_value<F>(f: F, x: &Tensor) -> Result<(Tensor, Tensor)>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    let xr = x.detach().requires_grad_(true);
    let y = f(&xr)?;
    if y.numel() != 1 {
        return Err(Error::Msg(format!("grad_and_value: f must return a scalar, got shape {:?}", y.shape())));
    }
    let g = grad(std::slice::from_ref(&y), std::slice::from_ref(&xr), None, false)?
        .pop()
        .flatten()
        .unwrap_or(x.zeros_like()?);
    Ok((g, y.detach()))
}

/// Applies `f` to every slice along `axis` 0 of `x` and stacks the results (`torch.vmap` by looping).
pub fn vmap<F>(f: F, x: &Tensor) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    let outs = (0..x.shape()[0]).map(|i| f(&x.select(0, i as isize)?)).collect::<Result<Vec<_>>>()?;
    Tensor::stack(&outs, 0)
}

/// Diagonal of the Hessian of a scalar function (via the full Hessian; fine for small inputs).
pub fn hessian_diag<F>(f: F, x: &Tensor) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    let n = x.numel();
    let h = hessian(f, x)?.reshape(&[n, n])?;
    let mask = Tensor::eye(n, h.dtype(), &h.device())?;
    h.mul(&mask)?.sum_axis(1, false)?.reshape(x.shape())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(t: &Tensor) -> Vec<f64> {
        t.to_vec::<f64>().unwrap()
    }

    #[test]
    fn forward_and_reverse_jacobians_agree() {
        let x = Tensor::from_slice(&[0.5f64, -1.0, 2.0], &[3]).unwrap();
        let f = |x: &Tensor| Tensor::stack(&[x.square()?.sum()?, x.sin()?.sum()?], 0);
        let (a, b) = (jacfwd(f, &x).unwrap(), jacrev(f, &x).unwrap());
        assert_eq!(a.shape(), [2, 3]);
        assert!(v(&a).iter().zip(v(&b)).all(|(p, q)| (p - q).abs() < 1e-9));
        assert!((v(&a)[0] - 1.0).abs() < 1e-9 && (v(&a)[3] - 0.5f64.cos()).abs() < 1e-9);
    }

    #[test]
    fn grad_value_vmap_hessian_diag() {
        let x = Tensor::from_slice(&[1.0f64, 2.0, 3.0], &[3]).unwrap();
        let (g, y) = grad_and_value(|x| x.powf(3.0)?.sum(), &x).unwrap();
        assert_eq!(v(&y), vec![36.0]);
        assert_eq!(v(&g), vec![3.0, 12.0, 27.0]);
        assert!(grad_and_value(|x| x.mul_scalar(2.0), &x).is_err());
        let m = Tensor::from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert_eq!(v(&vmap(|r| r.sum(), &m).unwrap()), vec![3.0, 7.0]);
        assert_eq!(v(&hessian_diag(|x| x.powf(3.0)?.sum(), &x).unwrap()), vec![6.0, 12.0, 18.0]);
    }
}
