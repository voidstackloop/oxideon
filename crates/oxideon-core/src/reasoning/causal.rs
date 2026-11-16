//! Causal Representation Learning & Structural Causal Models (SCMs).
//!
//! Implements Pearl's do-calculus and counterfactual reasoning (Pearl 2009, Peters et al. 2017):
//! - Structural Causal Models (SCMs) defined over causal DAGs $X_i := f_i(\operatorname{PA}_i) + U_i$.
//! - Interventional sampling $\operatorname{do}(X_k = x^*)$ via graphical surgery / mutilated graphs $\mathcal{G}_{\overline{X_k}}$.
//! - Counterfactual reasoning via three-step abduction, action, and prediction.
//! - Average Treatment Effect (ATE): $\mathbb{E}[Y \mid \operatorname{do}(X = x_1)] - \mathbb{E}[Y \mid \operatorname{do}(X = x_0)]$.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Structural equation for a causal variable: $X_i = \sum_{j \in \operatorname{PA}_i} w_j X_j + b_i + U_i$.
#[derive(Debug, Clone)]
pub struct CausalVariable {
    pub id: usize,
    pub name: String,
    pub parents: Vec<usize>,
    pub weights: Vec<f64>,
    pub bias: f64,
    pub noise_std: f64,
}

/// Structural Causal Model (SCM) engine over a directed acyclic graph.
#[derive(Debug, Clone)]
pub struct StructuralCausalModel {
    pub variables: Vec<CausalVariable>,
}

impl Default for StructuralCausalModel {
    fn default() -> Self {
        Self::new()
    }
}

impl StructuralCausalModel {
    pub fn new() -> Self {
        Self {
            variables: Vec::new(),
        }
    }

    /// Adds an endogenous variable with linear-additive parent dependencies and Gaussian noise.
    pub fn add_variable(
        &mut self,
        name: &str,
        parents: Vec<usize>,
        weights: Vec<f64>,
        bias: f64,
        noise_std: f64,
    ) -> Result<usize> {
        let id = self.variables.len();
        if parents.len() != weights.len() {
            return Err(Error::Msg(format!(
                "Parents length {} != weights length {}",
                parents.len(),
                weights.len()
            )));
        }
        for &p in &parents {
            if p >= id {
                return Err(Error::Msg(format!(
                    "Parent id {p} must be strictly less than new node id {id} to guarantee DAG structure"
                )));
            }
        }
        self.variables.push(CausalVariable {
            id,
            name: name.to_string(),
            parents,
            weights,
            bias,
            noise_std: noise_std.max(1e-8),
        });
        Ok(id)
    }

    /// Returns the number of variables in the causal model.
    pub fn num_variables(&self) -> usize {
        self.variables.len()
    }

    /// Re-checks the DAG invariants `add_variable` enforces: `variables` is
    /// pub, so tampered parents/weights must fail here instead of indexing
    /// out of bounds in the sampling loops.
    fn check_graph(&self) -> Result<()> {
        let d = self.variables.len();
        for (v_idx, var) in self.variables.iter().enumerate() {
            if var.parents.len() != var.weights.len() {
                return Err(Error::Msg(format!(
                    "SCM variable {v_idx}: {} parents vs {} weights",
                    var.parents.len(),
                    var.weights.len()
                )));
            }
            for &p in &var.parents {
                if p >= d {
                    return Err(Error::Msg(format!(
                        "SCM variable {v_idx}: parent id {p} out of range (0..{d})"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Generates observational samples $X \sim P(X_0, \dots, X_{d-1})$.
    pub fn sample_observational(&self, n_samples: usize, seed: Option<u64>) -> Result<Tensor> {
        self.sample_interventional(&[], n_samples, seed)
    }

    /// Generates interventional samples under $\operatorname{do}(X_{k_1} = x_1^*, \dots, X_{k_m} = x_m^*)$.
    pub fn sample_interventional(
        &self,
        interventions: &[(usize, f64)],
        n_samples: usize,
        seed: Option<u64>,
    ) -> Result<Tensor> {
        let d = self.variables.len();
        if d == 0 {
            return Err(Error::Msg("SCM has no variables".into()));
        }
        for &(idx, _) in interventions {
            if idx >= d {
                return Err(Error::Msg(format!(
                    "intervention variable {idx} out of range (0..{d})"
                )));
            }
        }
        self.check_graph()?;

        let mut s = seed.unwrap_or(2026);
        let mut next_gauss = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u1 = ((s >> 33) as f64 + 1.0) / (u32::MAX as f64 + 2.0);
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u2 = ((s >> 33) as f64 + 1.0) / (u32::MAX as f64 + 2.0);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };

        // Checked: `n_samples * d` overflowed on giant counts.
        let total = crate::limits::check_elements("scm samples", &[n_samples, d])?;
        let mut data = vec![0.0f64; total];

        for i in 0..n_samples {
            for v_idx in 0..d {
                // Check if variable is subject to an intervention
                if let Some(&(_, fixed_val)) = interventions.iter().find(|(idx, _)| *idx == v_idx) {
                    data[i * d + v_idx] = fixed_val;
                    continue;
                }

                let var = &self.variables[v_idx];
                let mut val = var.bias;
                for (p_idx, &p) in var.parents.iter().enumerate() {
                    let parent_val = data[i * d + p];
                    val += var.weights[p_idx] * parent_val;
                }
                let u = next_gauss() * var.noise_std;
                data[i * d + v_idx] = val + u;
            }
        }

        Tensor::from_vec(data, &[n_samples, d])
    }

    /// Evaluates Pearl's 3-step counterfactual:
    /// 1. **Abduction**: calculate exact noise $u_i = x_i^{\text{obs}} - (\sum w_j x_j^{\text{obs}} + b_i)$.
    /// 2. **Action**: replace structural equations for intervened variables with $x^*$.
    /// 3. **Prediction**: compute counterfactual values $X^{\text{cf}}$ propagating the abducted noise $u_i$.
    pub fn compute_counterfactual(
        &self,
        factual_obs: &[f64],
        interventions: &[(usize, f64)],
    ) -> Result<Vec<f64>> {
        let d = self.variables.len();
        if factual_obs.len() != d {
            return Err(Error::Msg(format!(
                "factual_obs length {} != num_variables {d}",
                factual_obs.len()
            )));
        }
        self.check_graph()?;

        // 1. Abduction step: infer noise values U
        let mut abducted_noise = vec![0.0f64; d];
        for v_idx in 0..d {
            let var = &self.variables[v_idx];
            let mut pred_mean = var.bias;
            for (p_idx, &p) in var.parents.iter().enumerate() {
                pred_mean += var.weights[p_idx] * factual_obs[p];
            }
            abducted_noise[v_idx] = factual_obs[v_idx] - pred_mean;
        }

        // 2 & 3. Action and Prediction step
        let mut counterfactual_vals = vec![0.0f64; d];
        for v_idx in 0..d {
            if let Some(&(_, fixed_val)) = interventions.iter().find(|(idx, _)| *idx == v_idx) {
                counterfactual_vals[v_idx] = fixed_val;
                continue;
            }

            let var = &self.variables[v_idx];
            let mut val = var.bias;
            for (p_idx, &p) in var.parents.iter().enumerate() {
                val += var.weights[p_idx] * counterfactual_vals[p];
            }
            counterfactual_vals[v_idx] = val + abducted_noise[v_idx];
        }

        Ok(counterfactual_vals)
    }

    /// Computes the Average Treatment Effect (ATE):
    /// $\text{ATE}(T \to Y) = \mathbb{E}[Y \mid \operatorname{do}(T = t_1)] - \mathbb{E}[Y \mid \operatorname{do}(T = t_0)]$.
    pub fn average_treatment_effect(
        &self,
        treatment_var: usize,
        outcome_var: usize,
        val_treated: f64,
        val_control: f64,
        n_samples: usize,
        seed: Option<u64>,
    ) -> Result<f64> {
        let d = self.variables.len();
        if treatment_var >= d || outcome_var >= d {
            return Err(Error::Msg(format!(
                "Variables ({treatment_var}, {outcome_var}) out of bounds (0..{d})"
            )));
        }

        let s1 = self.sample_interventional(&[(treatment_var, val_treated)], n_samples, seed)?;
        // Wrapping: `u64::MAX` is a valid seed; `+ 1` overflowed on it.
        let s0 = self.sample_interventional(
            &[(treatment_var, val_control)],
            n_samples,
            seed.map(|s| s.wrapping_add(1)),
        )?;

        let data1 = s1.to_vec::<f64>()?;
        let data0 = s0.to_vec::<f64>()?;

        let mean1: f64 = (0..n_samples)
            .map(|i| data1[i * d + outcome_var])
            .sum::<f64>()
            / (n_samples as f64);
        let mean0: f64 = (0..n_samples)
            .map(|i| data0[i * d + outcome_var])
            .sum::<f64>()
            / (n_samples as f64);

        Ok(mean1 - mean0)
    }
}
