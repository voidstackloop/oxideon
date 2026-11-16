//! Functional higher-order automatic differentiation (`jvp`, `vjp`, `jacobian`, `hessian`, `hvp`, `ntk_matrix`).
//!
//! Built on Oxideon's higher-order autograd engine (`create_graph = true`), allowing arbitrary
//! composition of forward and reverse-mode derivatives.

use crate::error::{Error, Result};
use crate::nn::Module;
use crate::tensor::{Tensor, grad};

/// Computes the output $y = f(x)$ and its Vector-Jacobian Product closure $v \mapsto J_f(x)^T v$.
pub fn vjp<F>(f: F, x: &Tensor) -> Result<(Tensor, impl Fn(&Tensor) -> Result<Tensor>)>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    let x_req = x.detach().requires_grad_(true);
    let y = f(&x_req)?;
    let y_clone = y.clone();
    let vjp_fn = move |v: &Tensor| -> Result<Tensor> {
        if v.shape() != y_clone.shape() {
            return Err(Error::ShapeMismatch {
                op: "vjp",
                expected: y_clone.shape().to_vec(),
                got: v.shape().to_vec(),
            });
        }
        let mut grads = grad(
            std::slice::from_ref(&y_clone),
            std::slice::from_ref(&x_req),
            Some(std::slice::from_ref(v)),
            true,
        )?;
        grads
            .pop()
            .flatten()
            .ok_or_else(|| Error::Msg("vjp: input tensor did not receive a gradient".into()))
    };
    Ok((y, vjp_fn))
}

/// Computes the output $y = f(x)$ and the Jacobian-Vector Product $J_f(x) v$ using the
/// forward-over-reverse (`create_graph = true`) trick:
/// Given a dummy cotangent $u$ with `requires_grad = true`, $g(u) = J_f(x)^T u$ is linear in $u$
/// with Jacobian $\frac{\partial g}{\partial u} = J_f(x)$, so $\nabla_u (g(u) \cdot v) = J_f(x) v$!
pub fn jvp<F>(f: F, x: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    if x.shape() != v.shape() {
        return Err(Error::ShapeMismatch {
            op: "jvp",
            expected: x.shape().to_vec(),
            got: v.shape().to_vec(),
        });
    }
    let x_req = x.detach().requires_grad_(true);
    let y = f(&x_req)?;
    let u = Tensor::zeros(y.shape(), y.dtype(), &y.device())?.requires_grad_(true);

    let vjp_u = grad(
        std::slice::from_ref(&y),
        std::slice::from_ref(&x_req),
        Some(std::slice::from_ref(&u)),
        true,
    )?
    .pop()
    .flatten()
    .ok_or_else(|| Error::Msg("jvp: input tensor is independent of output".into()))?;

    let jvp_out = grad(
        std::slice::from_ref(&vjp_u),
        std::slice::from_ref(&u),
        Some(std::slice::from_ref(v)),
        true,
    )?
    .pop()
    .flatten()
    .ok_or_else(|| Error::Msg("jvp: second-order derivative w.r.t. cotangent u failed".into()))?;

    Ok((y, jvp_out))
}

/// Computes the exact analytical Jacobian matrix $J \in \mathbb{R}^{m \times n}$ of $y = f(x)$,
/// where $m = \operatorname{numel}(y)$ and $n = \operatorname{numel}(x)$.
pub fn jacobian<F>(f: F, x: &Tensor) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    let x_req = x.detach().requires_grad_(true);
    let y = f(&x_req)?;
    let y_flat = y.reshape(&[y.numel()])?;
    let m = y_flat.numel();
    let n = x_req.numel();

    let eye = Tensor::eye(m, y.dtype(), &y.device())?;
    let mut rows = Vec::with_capacity(m);
    for i in 0..m {
        let e_i = eye.select(0, i as isize)?;
        let g_i = match grad(
            std::slice::from_ref(&y_flat),
            std::slice::from_ref(&x_req),
            Some(std::slice::from_ref(&e_i)),
            true,
        )?
        .pop()
        .flatten()
        {
            Some(g) => g,
            None => Tensor::zeros(x.shape(), x.dtype(), &x.device())?,
        };
        rows.push(g_i.reshape(&[n])?);
    }
    Tensor::stack(&rows, 0)
}

/// Computes the exact Hessian-Vector Product $\nabla^2 f(x) v$ of a scalar-valued function $f(x)$
/// in $O(\operatorname{numel}(x))$ time without materializing the full $n \times n$ Hessian.
pub fn hvp<F>(f: F, x: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    if x.shape() != v.shape() {
        return Err(Error::ShapeMismatch {
            op: "hvp",
            expected: x.shape().to_vec(),
            got: v.shape().to_vec(),
        });
    }
    let x_req = x.detach().requires_grad_(true);
    let loss = f(&x_req)?;
    if loss.numel() != 1 {
        return Err(Error::Msg(format!(
            "hvp requires a scalar function output, got shape {:?}",
            loss.shape()
        )));
    }
    let g = grad(
        std::slice::from_ref(&loss),
        std::slice::from_ref(&x_req),
        None,
        true,
    )?
    .pop()
    .flatten()
    .ok_or_else(|| Error::Msg("hvp: first derivative is None".into()))?;

    let gv = g.mul(v)?.sum()?;
    let hv = match grad(
        std::slice::from_ref(&gv),
        std::slice::from_ref(&x_req),
        None,
        true,
    )?
    .pop()
    .flatten()
    {
        Some(h) => h,
        None => Tensor::zeros(x.shape(), x.dtype(), &x.device())?,
    };

    Ok((g, hv))
}

/// Computes the exact analytical Hessian matrix $H \in \mathbb{R}^{n \times n}$ of a scalar-valued
/// function $f(x)$, where $n = \operatorname{numel}(x)$.
pub fn hessian<F>(f: F, x: &Tensor) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    let x_req = x.detach().requires_grad_(true);
    let loss = f(&x_req)?;
    if loss.numel() != 1 {
        return Err(Error::Msg(format!(
            "hessian requires a scalar function output, got shape {:?}",
            loss.shape()
        )));
    }
    let g = grad(
        std::slice::from_ref(&loss),
        std::slice::from_ref(&x_req),
        None,
        true,
    )?
    .pop()
    .flatten()
    .ok_or_else(|| Error::Msg("hessian: first derivative is None".into()))?;

    let g_flat = g.reshape(&[x_req.numel()])?;
    let n = x_req.numel();
    let eye = Tensor::eye(n, x.dtype(), &x.device())?;
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let e_i = eye.select(0, i as isize)?;
        let h_i = match grad(
            std::slice::from_ref(&g_flat),
            std::slice::from_ref(&x_req),
            Some(std::slice::from_ref(&e_i)),
            true,
        )?
        .pop()
        .flatten()
        {
            Some(h) => h,
            None => Tensor::zeros(x.shape(), x.dtype(), &x.device())?,
        };
        rows.push(h_i.reshape(&[n])?);
    }
    Tensor::stack(&rows, 0)
}

/// Computes the empirical Neural Tangent Kernel (NTK) Gram matrix
/// $\Theta(X_1, X_2) = J_\theta(X_1) J_\theta(X_2)^T \in \mathbb{R}^{N_1 \times N_2}$
/// for a scalar-output model (or summed over output dimensions per sample).
pub fn ntk_matrix<M: Module>(model: &M, x1: &Tensor, x2: &Tensor) -> Result<Tensor> {
    let params = model.parameters();
    let j1 = sample_param_jacobian(model, &params, x1)?;
    let j2 = if std::ptr::eq(x1, x2) {
        j1.clone()
    } else {
        sample_param_jacobian(model, &params, x2)?
    };
    j1.matmul(&j2.t()?)
}

fn sample_param_jacobian<M: Module>(model: &M, params: &[Tensor], x: &Tensor) -> Result<Tensor> {
    if x.ndim() < 1 {
        return Err(Error::Msg(
            "ntk_matrix: input needs at least one dim".into(),
        ));
    }
    let n_samples = x.shape()[0];
    let total_p: usize = params.iter().map(Tensor::numel).sum();
    let mut rows = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let xi = x.narrow(0, i, 1)?;
        let yi = model.forward(&xi)?.sum()?;
        let gs = grad(std::slice::from_ref(&yi), params, None, false)?;
        let mut flat_parts = Vec::with_capacity(params.len());
        for (p, g_opt) in params.iter().zip(gs) {
            let g = match g_opt {
                Some(gt) => gt.reshape(&[p.numel()])?,
                None => Tensor::zeros(&[p.numel()], p.dtype(), &p.device())?,
            };
            flat_parts.push(g);
        }
        let row = Tensor::cat(&flat_parts, 0)?;
        debug_assert_eq!(row.numel(), total_p);
        rows.push(row);
    }
    Tensor::stack(&rows, 0)
}

/// JAX / `torch.func.vmap`-style vectorizing map: applies a single-example function `f`
/// across slices of `x` along `in_axis` (using zero-copy `select` views) and stacks the
/// results along `out_axis`, preserving first- and higher-order autograd.
pub fn vmap<F>(f: F, x: &Tensor, in_axis: isize, out_axis: isize) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    let ndim = x.ndim() as isize;
    let ax = if in_axis < 0 { ndim + in_axis } else { in_axis };
    if ax < 0 || ax >= ndim {
        return Err(Error::InvalidAxis {
            op: "vmap",
            axis: in_axis,
            ndim: x.ndim(),
        });
    }
    let batch = x.shape()[ax as usize];
    let mut outputs = Vec::with_capacity(batch);
    for i in 0..batch {
        let slice_i = x.select(ax, i as isize)?;
        outputs.push(f(&slice_i)?);
    }
    Tensor::stack(&outputs, out_axis)
}

/// Binary vectorizing map (`vmap2`): applies `f(&a_i, &b_i)` across slices of `a` (along `in_axis_a`)
/// and `b` (along `in_axis_b`) and stacks the outputs along `out_axis`.
pub fn vmap2<F>(
    f: F,
    a: &Tensor,
    b: &Tensor,
    in_axis_a: isize,
    in_axis_b: isize,
    out_axis: isize,
) -> Result<Tensor>
where
    F: Fn(&Tensor, &Tensor) -> Result<Tensor>,
{
    let ax_a = if in_axis_a < 0 {
        (a.ndim() as isize) + in_axis_a
    } else {
        in_axis_a
    };
    let ax_b = if in_axis_b < 0 {
        (b.ndim() as isize) + in_axis_b
    } else {
        in_axis_b
    };
    if ax_a < 0 || ax_a >= a.ndim() as isize {
        return Err(Error::InvalidAxis {
            op: "vmap2",
            axis: in_axis_a,
            ndim: a.ndim(),
        });
    }
    if ax_b < 0 || ax_b >= b.ndim() as isize {
        return Err(Error::InvalidAxis {
            op: "vmap2",
            axis: in_axis_b,
            ndim: b.ndim(),
        });
    }
    let batch_a = a.shape()[ax_a as usize];
    let batch_b = b.shape()[ax_b as usize];
    if batch_a != batch_b {
        return Err(Error::ShapeMismatch {
            op: "vmap2",
            expected: vec![batch_a],
            got: vec![batch_b],
        });
    }
    let mut outputs = Vec::with_capacity(batch_a);
    for i in 0..batch_a {
        let sa = a.select(ax_a, i as isize)?;
        let sb = b.select(ax_b, i as isize)?;
        outputs.push(f(&sa, &sb)?);
    }
    Tensor::stack(&outputs, out_axis)
}
