//! Explainable AI (`XAI`) & Differentiable Feature Attribution:
//! - **Vanilla Saliency**: [`saliency_map`] ($|\nabla_x f_c(x)|$)
//! - **SmoothGrad**: [`smooth_grad`] (Smilkov et al., 2017 — Monte-Carlo noise-averaged gradient sensitivity)
//! - **Integrated Gradients**: [`integrated_gradients`] (Sundararajan et al., ICML 2017 — axiomatic path-integral attribution satisfying Completeness $\sum_i \operatorname{IG}_i(x) \approx f(x) - f(x')$)
//! - **Grad-CAM**: [`grad_cam_2d`] (Selvaraju et al., ICCV 2017 — gradient-weighted class activation mapping over 2D feature maps `[N, C, H, W]`)

use crate::error::{Error, Result};
use crate::tensor::{Tensor, grad};

/// Computes the input gradient attribution $\nabla_x f(x)$ (or $|\nabla_x f(x)|$ when `absolute = true`)
/// of scalar/summed target output `f(&x)` with respect to `input`.
pub fn saliency_map<F>(f: F, input: &Tensor, absolute: bool) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    let x = input.detach().requires_grad_(true);
    let out = f(&x)?.sum()?;
    let g = grad(&[out], std::slice::from_ref(&x), None, false)?
        .pop()
        .flatten()
        .ok_or_else(|| Error::Autograd("saliency_map: output did not depend on input".into()))?;
    if absolute { g.abs() } else { Ok(g) }
}

/// **SmoothGrad** (Smilkov et al., 2017):
/// Averages input gradients over `num_samples` Gaussian-perturbed copies $x + \mathcal{N}(0, \sigma^2 I)$
/// to sharpen visual attribution maps and suppress high-frequency gradient shatter.
pub fn smooth_grad<F>(
    f: F,
    input: &Tensor,
    num_samples: usize,
    noise_std: f64,
    absolute: bool,
) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    let n = crate::limits::check_iterations("smooth_grad num_samples", num_samples.max(1))?;
    let mut acc = input.zeros_like()?.detach();
    for _ in 0..n {
        let noise = Tensor::randn(input.shape())?
            .to_dtype(input.dtype())?
            .to_device(&input.device())?
            .mul_scalar(noise_std)?;
        let noisy_x = input.detach().add(&noise)?;
        let g = saliency_map(&f, &noisy_x, absolute)?;
        acc = acc.add(&g)?;
    }
    acc.div_scalar(n as f64)
}

/// **Integrated Gradients** (Sundararajan et al., ICML 2017):
/// Computes the axiomatic path-integral attribution along the straight line from `baseline` ($x'$)
/// to `input` ($x$) using an $M$-step Riemann trapezoidal/midpoint approximation:
/// $$\operatorname{IG}(x) = (x - x') \odot \frac{1}{M} \sum_{k=1}^{M} \nabla_x f\!\left(x' + \frac{k - 0.5}{M}(x - x')\right)$$
/// Satisfies the **Completeness Axiom**: $\sum_i \operatorname{IG}_i(x) \approx f(x) - f(x')$.
pub fn integrated_gradients<F>(
    f: F,
    input: &Tensor,
    baseline: Option<&Tensor>,
    steps: usize,
) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    let m = crate::limits::check_iterations("integrated_gradients steps", steps.max(1))?;
    let base = match baseline {
        Some(b) => {
            if b.shape() != input.shape() {
                return Err(Error::ShapeMismatch {
                    op: "integrated_gradients",
                    expected: input.shape().to_vec(),
                    got: b.shape().to_vec(),
                });
            }
            b.detach()
        }
        None => input.zeros_like()?.detach(),
    };

    let diff = input.detach().sub(&base)?;
    let mut grad_sum = input.zeros_like()?.detach();

    for k in 1..=m {
        let alpha = ((k as f64) - 0.5) / (m as f64);
        let interp = base
            .add(&diff.mul_scalar(alpha)?)?
            .detach()
            .requires_grad_(true);
        let y = f(&interp)?.sum()?;
        let g = grad(&[y], std::slice::from_ref(&interp), None, false)?
            .pop()
            .flatten()
            .ok_or_else(|| {
                Error::Autograd("integrated_gradients: output did not depend on input".into())
            })?;
        grad_sum = grad_sum.add(&g.detach())?;
    }

    let avg_grad = grad_sum.div_scalar(m as f64)?;
    diff.mul(&avg_grad)
}

/// **Grad-CAM** (Selvaraju et al., ICCV 2017):
/// Given a 4D convolutional feature activation map `activations` (`[N, C, H, W]`, with `requires_grad = true`)
/// and a downstream scalar/target score `score` derived from `activations`, computes the
/// global-average-pooled channel importance weights $\alpha_k^c = \frac{1}{H W}\sum_{i,j} \frac{\partial y^c}{\partial A_{i,j}^k}$
/// and returns the ReLU-rectified spatial heatmap $\operatorname{ReLU}\!\left(\sum_k \alpha_k^c A^k\right)$ of shape `[N, 1, H, W]`.
pub fn grad_cam_2d(activations: &Tensor, score: &Tensor) -> Result<Tensor> {
    if activations.ndim() != 4 {
        return Err(Error::Msg(format!(
            "grad_cam_2d expects 4D activations [N, C, H, W], got {:?}",
            activations.shape()
        )));
    }
    let s_sum = score.sum()?;
    let grads = grad(&[s_sum], std::slice::from_ref(activations), None, false)?
        .pop()
        .flatten()
        .ok_or_else(|| {
            Error::Autograd("grad_cam_2d: score did not depend on activations".into())
        })?;

    // Global average pooling over spatial dims (H=2, W=3) -> [N, C, 1, 1]
    let weights = grads.mean_axes(&[2, 3], true)?;
    let cam = activations
        .mul(&weights.broadcast_to(activations.shape())?)?
        .sum_axis(1, true)?
        .relu()?;
    Ok(cam)
}
