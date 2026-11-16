//! Tensor Programs IV: Maximal Update Parameterization ($\mu\mathrm{P}$) & Zero-Shot Hyperparameter Transfer (Yang & Hu, 2021).
//!
//! ### 1. Width-Scaling Rules for Feature Learning in the Infinite-Width Limit
//! Unlike Standard Parameterization (SP) or NTK parameterization (which freezes representation learning as width $d \to \infty$),
//! Yang's **Maximal Update Parameterization ($\mu\mathrm{P}$)** ensures every hidden layer's activation update satisfies
//! $\|\Delta h_l\|_2 = \Theta(\sqrt{d})$ by scaling hidden Adam learning rates and output projection weights inversely with width ratio $m = d / d_{\mathrm{base}}$:
//! $$\eta_{\mathrm{hidden}}(d) = \frac{\eta_{\mathrm{base}}}{m}, \qquad \sigma_{\mathrm{out}}(d) = \frac{\sigma_{\mathrm{base}}}{m}$$

use crate::error::{Error, Result};

/// Yang-Hu Maximal Update Parameterization ($\mu\mathrm{P}$) Zero-Shot Transfer Engine.
#[derive(Debug, Clone, Copy)]
pub struct MaximalUpdateParamEngine {
    /// Base proxy model width $d_{\mathrm{base}} \ge 1$.
    pub base_width: usize,
    /// Target large model width $d_{\mathrm{target}} \ge 1$.
    pub target_width: usize,
}

impl MaximalUpdateParamEngine {
    /// Creates a new $\mu\mathrm{P}$ scaling engine.
    pub fn new(base_width: usize, target_width: usize) -> Result<Self> {
        if base_width == 0 || target_width == 0 {
            return Err(Error::Msg(
                "base_width and target_width must be >= 1".to_string(),
            ));
        }
        Ok(Self {
            base_width,
            target_width,
        })
    }

    /// Computes the $\mu\mathrm{P}$-transferred hidden Adam learning rate $\eta_{\mathrm{base}} / m$ and attention logit multiplier $1 / d_{\mathrm{head}}$ (vs $1 / \sqrt{d_{\mathrm{head}}}$).
    pub fn transfer_adam_lr_and_init_std(&self, base_lr: f64, base_init_std: f64) -> (f64, f64) {
        let mult = (self.target_width as f64) / (self.base_width as f64);
        (base_lr / mult, base_init_std / mult.sqrt())
    }
}
