//! DeepSeek-V3 Auxiliary-Loss-Free MoE with Dynamic Bias Load Balancing and Shared Experts.
//!
//! Rather than degrading gradient representations with an artificial auxiliary load balancing loss,
//! DeepSeek-V3 introduces dynamic expert bias terms $b_e$ updated at each step:
//!
//! $$s_{i, e} = \text{sigmoid}(x_i \cdot W_e) + b_e$$
//! $$b_e \leftarrow b_e + \gamma \cdot (\bar{C} - C_e)$$
//!
//! Alongside routed experts, dedicated shared experts process every token unconditionally.

use crate::error::{Error, Result};
use crate::nn::{Linear, Module, SwiGLU};
use crate::tensor::Tensor;

/// DeepSeek-V3 Aux-Loss-Free Dynamic Expert Router.
#[derive(Clone)]
pub struct AuxFreeRouter {
    pub gate: Linear,
    pub dynamic_bias: Vec<f32>,
    pub num_experts: usize,
    pub top_k: usize,
    pub bias_update_rate: f32,
}

/// Routing result containing per-token expert choices and expert load distribution.
pub type MoERouteResult = (Vec<Vec<(usize, f32)>>, Vec<usize>);

impl AuxFreeRouter {
    pub fn new(
        d_model: usize,
        num_experts: usize,
        top_k: usize,
        bias_update_rate: f32,
    ) -> Result<Self> {
        if num_experts == 0 || top_k == 0 || top_k > num_experts || d_model == 0 {
            return Err(Error::Msg(format!(
                "AuxFreeRouter: invalid d_model={d_model}, num_experts={num_experts}, top_k={top_k}"
            )));
        }
        let gate = Linear::with_bias(d_model, num_experts, false)?;
        crate::limits::check_repeats("num_experts", num_experts)?;
        Ok(Self {
            gate,
            dynamic_bias: vec![0.0f32; num_experts],
            num_experts,
            top_k,
            bias_update_rate,
        })
    }

    /// Computes routing affinities, returns per-token chosen `(expert_id, normalized_gate)`
    /// and total token load distribution across all experts.
    pub fn route(&self, input: &Tensor) -> Result<MoERouteResult> {
        // All routing fields are pub: re-validate the self-consistency the indexing below assumes.
        if self.top_k == 0 || self.top_k > self.num_experts {
            return Err(Error::Msg(format!(
                "AuxFreeRouter: top_k={} needs 1..=num_experts={}",
                self.top_k, self.num_experts
            )));
        }
        if self.gate.out_features() != self.num_experts
            || self.dynamic_bias.len() != self.num_experts
        {
            return Err(Error::Msg(format!(
                "AuxFreeRouter: gate width {} / bias len {} disagree with num_experts={}",
                self.gate.out_features(),
                self.dynamic_bias.len(),
                self.num_experts
            )));
        }
        if self.gate.in_features() == 0 {
            return Err(Error::Msg(
                "AuxFreeRouter: gate in_features is 0 (token count would divide by zero)".into(),
            ));
        }
        let logits = self.gate.forward(input)?;
        let logits_vec = logits.to_vec::<f32>()?;
        let num_tokens = input.numel() / self.gate.in_features();
        let num_exp = self.num_experts;

        let mut routing_decisions = Vec::with_capacity(num_tokens);
        let mut expert_loads = vec![0usize; num_exp];

        for t in 0..num_tokens {
            let offset = t * num_exp;
            let mut affinities = Vec::with_capacity(num_exp);

            for e in 0..num_exp {
                let logit = logits_vec[offset + e];
                // Sigmoid gating + dynamic bias
                let sigmoid_score = 1.0 / (1.0 + (-logit).exp());
                let total_score = sigmoid_score + self.dynamic_bias[e];
                affinities.push((e, total_score));
            }

            // Top-K selection
            affinities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top = &affinities[..self.top_k];

            // Normalize top-k weights
            let sum_top: f32 = top.iter().map(|(_, s)| s.max(1e-6)).sum();
            let mut token_routes = Vec::with_capacity(self.top_k);

            for &(exp_id, s) in top {
                let norm_weight = s.max(1e-6) / sum_top;
                token_routes.push((exp_id, norm_weight));
                expert_loads[exp_id] += 1;
            }

            routing_decisions.push(token_routes);
        }

        Ok((routing_decisions, expert_loads))
    }

    /// Dynamically adjusts bias terms: underutilized experts get a boost, overloaded experts get lowered.
    pub fn update_biases(&mut self, expert_loads: &[usize], num_tokens: usize) {
        if num_tokens == 0 {
            return;
        }
        let target_load = (num_tokens as f32 * self.top_k as f32) / (self.num_experts as f32);
        // Infallible by API (Python calls this): never index past a shortened bias vec.
        let n = self.num_experts.min(self.dynamic_bias.len());
        for (e, &load_count) in expert_loads.iter().enumerate().take(n) {
            let actual_load = load_count as f32;
            let diff = target_load - actual_load;
            self.dynamic_bias[e] += self.bias_update_rate * (diff / (num_tokens as f32));
        }
    }
}

/// DeepSeek-V3 MoE Layer combining shared experts and aux-loss-free routed experts.
pub struct DeepSeekV3MoE {
    pub router: AuxFreeRouter,
    pub shared_experts: Vec<SwiGLU>,
    pub routed_experts: Vec<SwiGLU>,
    pub d_model: usize,
    pub num_shared: usize,
    pub num_routed: usize,
    pub top_k: usize,
}

impl DeepSeekV3MoE {
    pub fn new(
        d_model: usize,
        ffn_hidden: usize,
        num_shared: usize,
        num_routed: usize,
        top_k: usize,
        bias_update_rate: f32,
    ) -> Result<Self> {
        let router = AuxFreeRouter::new(d_model, num_routed, top_k, bias_update_rate)?;
        crate::limits::check_repeats("num_shared", num_shared)?;
        crate::limits::check_repeats("num_routed", num_routed)?;
        // every expert is a SwiGLU with three d_model x ffn_hidden matrices
        crate::limits::check_elements(
            "DeepSeekV3MoE experts",
            &[num_shared + num_routed, d_model, ffn_hidden, 3],
        )?;
        let mut shared_experts = Vec::with_capacity(num_shared);
        for _ in 0..num_shared {
            shared_experts.push(SwiGLU::new(d_model, ffn_hidden)?);
        }

        let mut routed_experts = Vec::with_capacity(num_routed);
        for _ in 0..num_routed {
            routed_experts.push(SwiGLU::new(d_model, ffn_hidden)?);
        }

        Ok(Self {
            router,
            shared_experts,
            routed_experts,
            d_model,
            num_shared,
            num_routed,
            top_k,
        })
    }

    /// Executes DeepSeek-V3 MoE forward pass and updates dynamic routing biases.
    pub fn forward_and_update(&mut self, input: &Tensor) -> Result<Tensor> {
        let orig_shape = input.shape().to_vec();
        let d = *orig_shape
            .last()
            .ok_or_else(|| Error::Msg("Empty input shape".into()))?;
        if d != self.d_model {
            return Err(Error::ShapeMismatch {
                op: "DeepSeekV3MoE",
                expected: vec![self.d_model],
                got: orig_shape,
            });
        }

        if d == 0 {
            return Err(Error::Msg(
                "DeepSeekV3MoE: d_model is 0 (token count would divide by zero)".into(),
            ));
        }
        let n_tokens = input.numel() / d;
        let x_flat = input.reshape(&[n_tokens, d])?;

        // 1. Shared experts contribution (all tokens go through shared experts)
        let mut combined_out = Tensor::zeros(&[n_tokens, d], input.dtype(), &input.device())?;
        for shared in &self.shared_experts {
            let s_out = shared.forward(&x_flat)?;
            combined_out = combined_out.add(&s_out)?;
        }

        // 2. Aux-Loss-Free Routing
        let (routes, expert_loads) = self.router.route(&x_flat)?;

        // 3. Routed expert evaluation
        let x_vec = x_flat.to_vec::<f32>()?;
        let mut routed_out_acc = vec![0.0f32; n_tokens * d];

        for (t, token_routes) in routes.iter().enumerate() {
            let token_in = &x_vec[t * d..(t + 1) * d];
            let t_tensor = Tensor::from_slice(token_in, &[1, d])?.to_device(&input.device())?;

            for &(exp_id, weight) in token_routes {
                let expert = self.routed_experts.get(exp_id).ok_or_else(|| {
                    Error::Msg(format!(
                        "DeepSeekV3MoE: routed expert {exp_id} missing (have {})",
                        self.routed_experts.len()
                    ))
                })?;
                let exp_res = expert.forward(&t_tensor)?;
                let res_vec = exp_res.to_vec::<f32>()?;
                for i in 0..d {
                    routed_out_acc[t * d + i] += weight * res_vec[i];
                }
            }
        }

        let routed_tensor =
            Tensor::from_slice(&routed_out_acc, &[n_tokens, d])?.to_device(&input.device())?;
        let total_flat = combined_out.add(&routed_tensor)?;

        // 4. Update dynamic biases
        self.router.update_biases(&expert_loads, n_tokens);

        total_flat.reshape(&orig_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn router_rejects_tampered_top_k() -> Result<()> {
        let mut r = AuxFreeRouter::new(8, 4, 2, 0.01)?;
        r.top_k = 99; // pub field: slicing affinities[..top_k] would panic
        let x = Tensor::zeros_f32(&[2, 8])?;
        let err = r
            .route(&x)
            .expect_err("top_k beyond num_experts must be an error, not a panic");
        assert!(err.to_string().contains("top_k"), "{err}");
        Ok(())
    }

    #[test]
    fn router_rejects_swapped_gate() -> Result<()> {
        let mut r = AuxFreeRouter::new(8, 4, 2, 0.01)?;
        r.gate = Linear::with_bias(8, 1, false)?; // fewer rows than num_experts: offset indexing would panic
        let x = Tensor::zeros_f32(&[2, 8])?;
        let err = r
            .route(&x)
            .expect_err("gate/router width mismatch must be an error, not a panic");
        assert!(err.to_string().contains("num_experts"), "{err}");
        Ok(())
    }

    #[test]
    fn moe_rejects_truncated_experts() -> Result<()> {
        let mut m = DeepSeekV3MoE::new(8, 16, 1, 4, 2, 0.01)?;
        m.routed_experts.truncate(1); // pub field: routed_experts[exp_id] would panic
        let x = Tensor::zeros_f32(&[1, 2, 8])?;
        let err = m
            .forward_and_update(&x)
            .expect_err("truncated experts must be an error, not a panic");
        assert!(err.to_string().contains("expert"), "{err}");
        Ok(())
    }
}
