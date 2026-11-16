//! Jukes-Cantor & Kimura 2-Parameter Continuous-Time Markov Substitution & Felsenstein Phylogenetic Distance (Jukes & Cantor, 1969; Felsenstein, 1981).
//!
//! ### 1. Jukes-Cantor Transition Probability & Evolutionary Distance $d_{\mathrm{JC}}$
//! Under equal nucleotide substitution rate across 4 bases, the probability of observing a difference after branch length $d = 3 \alpha t < 3/4$ is $p = \frac{3}{4}(1 - e^{-4 d / 3})$, inverted as:
//! $$d_{\mathrm{JC}}(p) = -\frac{3}{4} \ln\!\left(1 - \frac{4}{3} p\right), \qquad P_{ii}(d) = \frac{1}{4} + \frac{3}{4} e^{-4 d / 3}$$

use crate::error::{Error, Result};

/// Felsenstein Phylogenetic Likelihood & Jukes-Cantor Substitution Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct FelsensteinPhylogeneticLikelihood;

impl FelsensteinPhylogeneticLikelihood {
    /// Creates a new Felsenstein Phylogenetic Likelihood engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the Jukes-Cantor evolutionary distance $d_{\mathrm{JC}}(p)$ and asymptotic sampling variance $\mathrm{Var}(\hat{d}) = \frac{p(1 - p)}{L (1 - 4p/3)^2}$ for observed mismatch fraction $p \in [0, 0.75)$ over $L$ sites.
    pub fn jukes_cantor_distance_and_variance(
        mismatch_p: f64,
        sequence_length_l: usize,
    ) -> Result<(f64, f64)> {
        if !(0.0..0.75).contains(&mismatch_p) || sequence_length_l == 0 {
            return Err(Error::Msg(
                "mismatch_p in [0, 0.75) and sequence_length_l >= 1 required".to_string(),
            ));
        }
        let arg = 1.0 - (4.0 / 3.0) * mismatch_p;
        let d_jc = -0.75 * arg.ln();
        let var_d = mismatch_p * (1.0 - mismatch_p) / ((sequence_length_l as f64) * arg * arg);
        Ok((d_jc, var_d))
    }
}
