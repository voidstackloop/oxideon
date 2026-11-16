//! Loss modules in the style of `torch.nn`: [`CrossEntropyLoss`], [`NllLoss`], [`MseLoss`], [`L1Loss`],
//! [`SmoothL1Loss`], [`HuberLoss`], [`BceLoss`], [`BceWithLogitsLoss`], [`KlDivLoss`], [`PoissonNllLoss`],
//! [`GaussianNllLoss`], [`MultiLabelSoftMarginLoss`] and [`MultiMarginLoss`]. Each holds its options and exposes
//! `forward(input, target)` through the [`Loss`] trait; they wrap the functions of [`super::loss_ext`].

use super::loss_ext::{
    KlReduction, Reduction, Target, binary_cross_entropy, binary_cross_entropy_with_logits_r, cross_entropy_nd,
    gaussian_nll_loss, huber_loss_r, kl_div_r, l1_loss_r, mse_loss_r, multi_margin_loss, multilabel_soft_margin_loss,
    nll_loss_nd, poisson_nll_loss, smooth_l1_loss,
};
use crate::dtype::DType;
use crate::error::Result;
use crate::tensor::Tensor;

/// A loss between a prediction and a target.
pub trait Loss {
    fn forward(&self, input: &Tensor, target: &Tensor) -> Result<Tensor>;
}

fn is_index(t: &Tensor) -> bool {
    matches!(t.dtype(), DType::I32 | DType::I64 | DType::U8)
}

/// Cross entropy on logits; integer targets are class indices, floating-point targets are class probabilities.
#[derive(Clone)]
pub struct CrossEntropyLoss {
    pub weight: Option<Tensor>,
    pub ignore_index: Option<i64>,
    pub label_smoothing: f64,
    pub reduction: Reduction,
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self { weight: None, ignore_index: None, label_smoothing: 0.0, reduction: Reduction::Mean }
    }
}

impl Loss for CrossEntropyLoss {
    fn forward(&self, input: &Tensor, target: &Tensor) -> Result<Tensor> {
        let t = if is_index(target) { Target::Indices(target) } else { Target::Probs(target) };
        cross_entropy_nd(input, t, self.weight.as_ref(), self.ignore_index, self.label_smoothing, self.reduction)
    }
}

/// Negative log-likelihood on log-probabilities.
#[derive(Clone)]
pub struct NllLoss {
    pub weight: Option<Tensor>,
    pub ignore_index: Option<i64>,
    pub reduction: Reduction,
}

impl Default for NllLoss {
    fn default() -> Self {
        Self { weight: None, ignore_index: None, reduction: Reduction::Mean }
    }
}

impl Loss for NllLoss {
    fn forward(&self, input: &Tensor, target: &Tensor) -> Result<Tensor> {
        nll_loss_nd(input, target, self.weight.as_ref(), self.ignore_index, self.reduction)
    }
}

macro_rules! simple_loss {
    ($(#[$doc:meta])* $name:ident, |$s:ident, $i:ident, $t:ident| $body:expr) => {
        $(#[$doc])*
        #[derive(Clone)]
        pub struct $name {
            pub reduction: Reduction,
        }
        impl Default for $name {
            fn default() -> Self {
                Self { reduction: Reduction::Mean }
            }
        }
        impl Loss for $name {
            fn forward(&self, $i: &Tensor, $t: &Tensor) -> Result<Tensor> {
                let $s = self;
                $body
            }
        }
    };
}

simple_loss!(
    /// Mean squared error.
    MseLoss, |s, i, t| mse_loss_r(i, t, s.reduction)
);
simple_loss!(
    /// Mean absolute error.
    L1Loss, |s, i, t| l1_loss_r(i, t, s.reduction)
);
simple_loss!(
    /// Binary cross-entropy on probabilities.
    BceLoss, |s, i, t| binary_cross_entropy(i, t, None, s.reduction)
);
simple_loss!(
    /// Binary cross-entropy on logits.
    BceWithLogitsLoss, |s, i, t| binary_cross_entropy_with_logits_r(i, t, None, None, s.reduction)
);
simple_loss!(
    /// One-versus-all multi-label loss on logits.
    MultiLabelSoftMarginLoss, |s, i, t| multilabel_soft_margin_loss(i, t, None, s.reduction)
);

/// Smooth L1 loss with transition point `beta`.
#[derive(Clone)]
pub struct SmoothL1Loss {
    pub beta: f64,
    pub reduction: Reduction,
}

impl Default for SmoothL1Loss {
    fn default() -> Self {
        Self { beta: 1.0, reduction: Reduction::Mean }
    }
}

impl Loss for SmoothL1Loss {
    fn forward(&self, input: &Tensor, target: &Tensor) -> Result<Tensor> {
        smooth_l1_loss(input, target, self.beta, self.reduction)
    }
}

/// Huber loss with threshold `delta`.
#[derive(Clone)]
pub struct HuberLoss {
    pub delta: f64,
    pub reduction: Reduction,
}

impl Default for HuberLoss {
    fn default() -> Self {
        Self { delta: 1.0, reduction: Reduction::Mean }
    }
}

impl Loss for HuberLoss {
    fn forward(&self, input: &Tensor, target: &Tensor) -> Result<Tensor> {
        huber_loss_r(input, target, self.delta, self.reduction)
    }
}

/// KL divergence with log-probability input.
#[derive(Clone)]
pub struct KlDivLoss {
    pub reduction: KlReduction,
    pub log_target: bool,
}

impl Default for KlDivLoss {
    fn default() -> Self {
        Self { reduction: KlReduction::Mean, log_target: false }
    }
}

impl Loss for KlDivLoss {
    fn forward(&self, input: &Tensor, target: &Tensor) -> Result<Tensor> {
        kl_div_r(input, target, self.reduction, self.log_target)
    }
}

/// Poisson negative log-likelihood.
#[derive(Clone)]
pub struct PoissonNllLoss {
    pub log_input: bool,
    pub full: bool,
    pub eps: f64,
    pub reduction: Reduction,
}

impl Default for PoissonNllLoss {
    fn default() -> Self {
        Self { log_input: true, full: false, eps: 1e-8, reduction: Reduction::Mean }
    }
}

impl Loss for PoissonNllLoss {
    fn forward(&self, input: &Tensor, target: &Tensor) -> Result<Tensor> {
        poisson_nll_loss(input, target, self.log_input, self.full, self.eps, self.reduction)
    }
}

/// Gaussian negative log-likelihood; the variance is supplied through [`GaussianNllLoss::forward_var`].
#[derive(Clone)]
pub struct GaussianNllLoss {
    pub full: bool,
    pub eps: f64,
    pub reduction: Reduction,
}

impl Default for GaussianNllLoss {
    fn default() -> Self {
        Self { full: false, eps: 1e-6, reduction: Reduction::Mean }
    }
}

impl GaussianNllLoss {
    pub fn forward_var(&self, input: &Tensor, target: &Tensor, var: &Tensor) -> Result<Tensor> {
        gaussian_nll_loss(input, target, var, self.full, self.eps, self.reduction)
    }
}

/// Multi-class hinge loss.
#[derive(Clone)]
pub struct MultiMarginLoss {
    pub p: u32,
    pub margin: f64,
    pub weight: Option<Tensor>,
    pub reduction: Reduction,
}

impl Default for MultiMarginLoss {
    fn default() -> Self {
        Self { p: 1, margin: 1.0, weight: None, reduction: Reduction::Mean }
    }
}

impl Loss for MultiMarginLoss {
    fn forward(&self, input: &Tensor, target: &Tensor) -> Result<Tensor> {
        multi_margin_loss(input, target, self.p, self.margin, self.weight.as_ref(), self.reduction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(v: &[f64], shape: &[usize]) -> Tensor {
        Tensor::from_slice(v, shape).unwrap()
    }
    fn val(t: &Tensor) -> f64 {
        t.to_vec::<f64>().unwrap()[0]
    }

    #[test]
    fn modules_dispatch_to_the_functions() {
        let logits = f(&[1.0, 2.0, 3.0, 3.0, 1.0, 0.0], &[2, 3]);
        let idx = Tensor::from_slice(&[2i64, 0], &[2]).unwrap();
        let ce = CrossEntropyLoss::default();
        let by_index = val(&ce.forward(&logits, &idx).unwrap());
        let probs = f(&[0.0, 0.0, 1.0, 1.0, 0.0, 0.0], &[2, 3]);
        assert!((by_index - val(&ce.forward(&logits, &probs).unwrap())).abs() < 1e-9);
        let weighted = CrossEntropyLoss { ignore_index: Some(0), ..Default::default() };
        assert!((val(&weighted.forward(&logits, &idx).unwrap()) - ((1.0f64.exp() + 2.0f64.exp() + 3.0f64.exp()).ln() - 3.0)).abs() < 1e-9);
        let (p, t) = (f(&[0.0, 2.0], &[2]), f(&[1.0, 4.0], &[2]));
        assert_eq!(val(&MseLoss::default().forward(&p, &t).unwrap()), 2.5);
        assert_eq!(val(&L1Loss { reduction: Reduction::Sum }.forward(&p, &t).unwrap()), 3.0);
        assert_eq!(val(&SmoothL1Loss::default().forward(&p, &t).unwrap()), (0.5 + 1.5) / 2.0);
        assert_eq!(val(&HuberLoss { delta: 1.0, reduction: Reduction::Sum }.forward(&p, &t).unwrap()), 2.0);
        let lg = f(&[2.0, -1.0], &[2]);
        let y = f(&[1.0, 0.0], &[2]);
        let a = val(&BceWithLogitsLoss::default().forward(&lg, &y).unwrap());
        let b = val(&BceLoss::default().forward(&lg.sigmoid().unwrap(), &y).unwrap());
        assert!((a - b).abs() < 1e-9);
        let lp = f(&[0.5f64.ln(), 0.5f64.ln()], &[1, 2]);
        let kl = val(&KlDivLoss { reduction: KlReduction::BatchMean, log_target: false }.forward(&lp, &f(&[0.25, 0.75], &[1, 2])).unwrap());
        assert!(kl > 0.0);
        let ml = MultiMarginLoss::default().forward(&f(&[0.1, 0.2, 0.4, 0.8], &[1, 4]), &Tensor::from_slice(&[3i64], &[1]).unwrap()).unwrap();
        assert!((val(&ml) - 1.3 / 4.0).abs() < 1e-9);
        assert!(val(&PoissonNllLoss::default().forward(&f(&[1.0], &[1]), &f(&[2.0], &[1])).unwrap()).is_finite());
        assert!(val(&GaussianNllLoss::default().forward_var(&f(&[1.0], &[1]), &f(&[3.0], &[1]), &f(&[2.0], &[1])).unwrap()).is_finite());
        assert!(val(&MultiLabelSoftMarginLoss::default().forward(&f(&[0.0, 2.0], &[1, 2]), &f(&[1.0, 0.0], &[1, 2])).unwrap()) > 0.0);
        assert!(val(&NllLoss::default().forward(&logits.log_softmax(1).unwrap(), &idx).unwrap()) > 0.0);
    }
}
