//! Function transforms over several inputs (the `torch.func` surface beyond the single-input helpers of [`crate::functional`]):
//! [`value_and_grad`], [`grad_argnums`], [`jacrev`] (reverse mode, any number of inputs), [`jacfwd`] (forward mode),
//! [`per_sample_grads`] and [`functional_call`].

use crate::error::{Error, Result};
use crate::functional::jvp;
use crate::nn::Module;
use crate::tensor::{Tensor, enable_grad, grad, no_grad};
use std::collections::HashMap;

fn leaves(inputs: &[Tensor], argnums: &[usize]) -> Result<Vec<Tensor>> {
    if let Some(&bad) = argnums.iter().find(|&&a| a >= inputs.len()) {
        return Err(Error::Msg(format!("transforms: argnum {bad} is out of range for {} inputs", inputs.len())));
    }
    Ok(inputs.iter().enumerate().map(|(i, t)| if argnums.contains(&i) { t.detach().requires_grad_(true) } else { t.detach() }).collect())
}

/// `f(inputs)` (a scalar) and its gradients with respect to the inputs listed in `argnums`, in that order.
pub fn value_and_grad<F>(f: F, inputs: &[Tensor], argnums: &[usize]) -> Result<(Tensor, Vec<Tensor>)>
where
    F: Fn(&[Tensor]) -> Result<Tensor>,
{
    let xs = leaves(inputs, argnums)?;
    let y = enable_grad(|| f(&xs))?;
    if y.numel() != 1 {
        return Err(Error::Msg(format!("value_and_grad: the function must return a scalar, got shape {:?}", y.shape())));
    }
    let wrt: Vec<Tensor> = argnums.iter().map(|&a| xs[a].clone()).collect();
    let gs = grad(std::slice::from_ref(&y), &wrt, None, false)?;
    let grads = gs.into_iter().zip(&wrt).map(|(g, w)| match g {
        Some(g) => Ok(g),
        None => w.zeros_like(),
    }).collect::<Result<Vec<_>>>()?;
    Ok((y.detach(), grads))
}

/// Gradients of the scalar `f(inputs)` with respect to the inputs listed in `argnums`.
pub fn grad_argnums<F>(f: F, inputs: &[Tensor], argnums: &[usize]) -> Result<Vec<Tensor>>
where
    F: Fn(&[Tensor]) -> Result<Tensor>,
{
    value_and_grad(f, inputs, argnums).map(|(_, g)| g)
}

/// Reverse-mode Jacobians: for every input in `argnums`, a tensor of shape `[out_shape..., in_shape...]`.
pub fn jacrev<F>(f: F, inputs: &[Tensor], argnums: &[usize]) -> Result<Vec<Tensor>>
where
    F: Fn(&[Tensor]) -> Result<Tensor>,
{
    let xs = leaves(inputs, argnums)?;
    let y = enable_grad(|| f(&xs))?;
    let wrt: Vec<Tensor> = argnums.iter().map(|&a| xs[a].clone()).collect();
    let m = y.numel();
    let mut rows: Vec<Vec<Tensor>> = vec![Vec::with_capacity(m); wrt.len()];
    for i in 0..m {
        let mut onehot = vec![0.0f64; m];
        onehot[i] = 1.0;
        let seed = Tensor::from_slice(&onehot, &[m])?.to_dtype(y.dtype())?.reshape(y.shape())?;
        let gs = grad(std::slice::from_ref(&y), &wrt, Some(std::slice::from_ref(&seed)), true)?;
        for (k, (g, w)) in gs.into_iter().zip(&wrt).enumerate() {
            rows[k].push(match g {
                Some(g) => g.detach(),
                None => w.zeros_like()?,
            });
        }
    }
    rows.into_iter()
        .zip(&wrt)
        .map(|(r, w)| {
            let mut shape = y.shape().to_vec();
            shape.extend_from_slice(w.shape());
            Tensor::stack(&r, 0)?.reshape(&shape)
        })
        .collect()
}

/// Forward-mode Jacobian of a single-input function: one Jacobian-vector product per input element; shape
/// `[out_shape..., in_shape...]`.
pub fn jacfwd<F>(f: F, x: &Tensor) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    let n = x.numel();
    let mut cols = Vec::with_capacity(n);
    let mut out_shape = Vec::new();
    for j in 0..n {
        let mut basis = vec![0.0f64; n];
        basis[j] = 1.0;
        let v = Tensor::from_slice(&basis, &[n])?.to_dtype(x.dtype())?.reshape(x.shape())?;
        let (y, t) = jvp(&f, x, &v)?;
        out_shape = y.shape().to_vec();
        cols.push(t.detach());
    }
    let mut shape = out_shape;
    shape.extend_from_slice(x.shape());
    // columns are indexed by the input element: stack along a trailing axis, then restore the input shape
    let jac = Tensor::stack(&cols, -1)?;
    let lead: Vec<usize> = jac.shape()[..jac.ndim() - 1].to_vec();
    let mut full = lead;
    full.extend_from_slice(x.shape());
    let _ = shape;
    jac.reshape(&full)
}

/// Per-example gradients: `loss_of(i)` builds the scalar loss of example `i`; the result holds, for every parameter, a tensor
/// `[batch, ...param shape]` of the gradients of the individual losses (what `vmap(grad(...))` returns in `torch.func`).
pub fn per_sample_grads<F>(params: &[Tensor], batch: usize, loss_of: F) -> Result<Vec<Tensor>>
where
    F: Fn(usize) -> Result<Tensor>,
{
    let mut per_param: Vec<Vec<Tensor>> = vec![Vec::with_capacity(batch); params.len()];
    for i in 0..batch {
        for p in params {
            p.zero_grad();
        }
        let loss = enable_grad(|| loss_of(i))?;
        loss.backward()?;
        for (k, p) in params.iter().enumerate() {
            per_param[k].push(match p.grad() {
                Some(g) => g.detach().contiguous()?,
                None => p.zeros_like()?,
            });
        }
    }
    for p in params {
        p.zero_grad();
    }
    per_param.into_iter().map(|g| Tensor::stack(&g, 0)).collect()
}

/// Runs `module` with the parameters named in `params` temporarily replaced by the given values (`torch.func.functional_call`
/// for evaluation: ensembles, model averaging, what-if weights). Unlisted parameters keep their values; the originals are
/// restored afterwards, also when the forward pass fails. Gradients of the result flow to the module's own parameter
/// tensors, not to the supplied values.
pub fn functional_call(module: &dyn Module, params: &HashMap<String, Tensor>, input: &Tensor) -> Result<Tensor> {
    let named = module.named_parameters();
    for k in params.keys() {
        if !named.iter().any(|(n, _)| n == k) {
            return Err(Error::Msg(format!("functional_call: the module has no parameter named {k:?}")));
        }
    }
    let saved: Vec<(Tensor, Tensor)> = named.iter().filter(|(n, _)| params.contains_key(n)).map(|(_, t)| (t.clone(), t.to_dtype(crate::dtype::DType::F64).and_then(|c| c.contiguous()).and_then(|c| c.to_vec::<f64>()).and_then(|v| Tensor::from_slice(&v, t.shape())).and_then(|c| c.to_dtype(t.dtype())))).map(|(t, c)| c.map(|c| (t, c))).collect::<Result<_>>()?;
    let swap = |pairs: Vec<(&Tensor, &Tensor)>| -> Result<()> {
        no_grad(|| pairs.into_iter().try_for_each(|(dst, src)| dst.assign(src)))
    };
    let target: Vec<(&Tensor, &Tensor)> = named.iter().filter_map(|(n, t)| params.get(n).map(|v| (t, v))).collect();
    swap(target)?;
    let out = module.forward(input);
    swap(saved.iter().map(|(t, c)| (t, c)).collect())?;
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Linear;

    fn v(t: &Tensor) -> Vec<f64> {
        t.to_vec::<f64>().unwrap()
    }
    fn t64(x: &[f64], shape: &[usize]) -> Tensor {
        Tensor::from_slice(x, shape).unwrap()
    }

    #[test]
    fn gradients_and_jacobians_over_several_inputs() {
        let (x, y) = (t64(&[0.3, -0.7, 1.1], &[3]), t64(&[2.0, 0.5, -1.0], &[3]));
        // f = sum(sin(x) * y) + mean(x^2)
        let f = |a: &[Tensor]| a[0].sin()?.mul(&a[1])?.sum()?.add(&a[0].square()?.mean()?);
        let (val, g) = value_and_grad(f, &[x.clone(), y.clone()], &[0, 1]).unwrap();
        let want_val: f64 = (0..3).map(|i| [0.3f64, -0.7, 1.1][i].sin() * [2.0, 0.5, -1.0][i]).sum::<f64>() + [0.09f64, 0.49, 1.21].iter().sum::<f64>() / 3.0;
        assert!((val.item::<f64>().unwrap() - want_val).abs() < 1e-12);
        let gx = v(&g[0]);
        for i in 0..3 {
            let (xi, yi) = ([0.3f64, -0.7, 1.1][i], [2.0f64, 0.5, -1.0][i]);
            assert!((gx[i] - (xi.cos() * yi + 2.0 * xi / 3.0)).abs() < 1e-12);
            assert!((v(&g[1])[i] - xi.sin()).abs() < 1e-12);
        }
        // reverse and forward Jacobians of a vector function agree with each other and with the analytic one
        let w = t64(&[0.5, -1.0, 0.25, 2.0, 1.0, -0.5], &[2, 3]);
        let h = |a: &[Tensor]| a[0].matmul(&w.t().unwrap())?.tanh();
        let xm = t64(&[0.1, 0.2, -0.3, 0.4, 0.0, 0.9], &[2, 3]);
        let jr = jacrev(h, std::slice::from_ref(&xm), &[0]).unwrap().remove(0);
        assert_eq!(jr.shape(), [2, 2, 2, 3]);
        let jf = jacfwd(|a| a.matmul(&w.t().unwrap())?.tanh(), &xm).unwrap();
        assert_eq!(jf.shape(), [2, 2, 2, 3]);
        assert!(jr.allclose(&jf, 1e-12, 1e-12).unwrap());
        // d out[b, o] / d x[b, k] = (1 - out^2) * w[o, k]
        let out = v(&h(std::slice::from_ref(&xm)).unwrap());
        let jv = v(&jr);
        for b in 0..2 {
            for o in 0..2 {
                for k in 0..3 {
                    let want = (1.0 - out[b * 2 + o].powi(2)) * v(&w)[o * 3 + k];
                    assert!((jv[((b * 2 + o) * 2 + b) * 3 + k] - want).abs() < 1e-12);
                }
            }
        }
        assert!(value_and_grad(|a| Ok(a[0].clone()), std::slice::from_ref(&xm), &[0]).is_err());
        assert!(grad_argnums(f, &[x, y], &[2]).is_err());
    }

    #[test]
    fn per_sample_gradients_match_one_example_at_a_time_and_sum_to_the_batch_gradient() {
        crate::rng::manual_seed(6);
        let l = Linear::new(3, 2).unwrap();
        let x = Tensor::randn(&[5, 3]).unwrap();
        let y = Tensor::randn(&[5, 2]).unwrap();
        let params = vec![l.weight.clone(), l.bias.clone().unwrap()];
        let loss_i = |i: usize| l.forward(&x.narrow(0, i, 1)?)?.sub(&y.narrow(0, i, 1)?)?.square()?.sum();
        let g = per_sample_grads(&params, 5, loss_i).unwrap();
        assert_eq!(g[0].shape(), [5, 2, 3]);
        assert_eq!(g[1].shape(), [5, 2]);
        // the sum of per-sample gradients is the gradient of the summed loss
        for p in &params {
            p.zero_grad();
        }
        l.forward(&x).unwrap().sub(&y).unwrap().square().unwrap().sum().unwrap().backward().unwrap();
        for (k, p) in params.iter().enumerate() {
            let total = g[k].sum_axes(&[0], false).unwrap();
            assert!(total.allclose(&p.grad().unwrap(), 1e-4, 1e-4).unwrap());
        }
    }

    #[test]
    fn functional_call_swaps_parameters_and_restores_them() {
        let l = Linear::new(2, 2).unwrap();
        let x = Tensor::from_slice(&[1.0f32, 2.0], &[1, 2]).unwrap();
        let before = l.forward(&x).unwrap().to_vec::<f32>().unwrap();
        let mut p = HashMap::new();
        p.insert("weight".to_string(), Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2]).unwrap());
        p.insert("bias".to_string(), Tensor::from_slice(&[10.0f32, 20.0], &[2]).unwrap());
        assert_eq!(functional_call(&l, &p, &x).unwrap().to_vec::<f32>().unwrap(), vec![11.0, 22.0]);
        assert_eq!(l.forward(&x).unwrap().to_vec::<f32>().unwrap(), before);
        p.insert("nope".to_string(), x.clone());
        assert!(functional_call(&l, &p, &x).is_err());
    }
}
