//! Dynamic Autonomous Precision & Quantization Controller for LLM Inference.
//!
//! Provides automatic, runtime precision switching across FP32, FP8, Q8_0, Q4_0, and
//! 1.58-bit Ternary BitLinear in response to real-time telemetry (latency, memory headroom,
//! KV-cache capacity, and outlier activation kurtosis).

use crate::error::{Error, Result};
use crate::nn::{Linear, Module};
use crate::quant::{BitLinear, FP8Format, FP8Linear, QuantScheme, QuantizedLinear};
use crate::tensor::Tensor;
use std::collections::BTreeMap;

/// Quantization precision level for LLM layers during runtime inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DynamicPrecisionLevel {
    /// 1.58-bit Ternary compression ({-1, 0, +1} weights).
    BitLinear1_58 = 0,
    /// 4-bit standard symmetric block quantization (Q4_0).
    Q4_0 = 1,
    /// 8-bit standard symmetric block quantization (Q8_0).
    Q8_0 = 2,
    /// 8-bit floating-point (FP8 E4M3).
    FP8 = 3,
    /// Full 32-bit floating point uncompressed baseline.
    FP32 = 4,
}

impl DynamicPrecisionLevel {
    /// Nominal bits per weight parameter.
    pub fn bits_per_param(&self) -> f64 {
        match self {
            Self::BitLinear1_58 => 1.585,
            Self::Q4_0 => 4.0,
            Self::Q8_0 => 8.0,
            Self::FP8 => 8.0,
            Self::FP32 => 32.0,
        }
    }

    /// Theoretical memory compression ratio compared to FP32.
    pub fn compression_ratio(&self) -> f64 {
        32.0 / self.bits_per_param()
    }

    /// Relative compute/memory bandwidth speedup factor estimate.
    pub fn estimated_speedup(&self) -> f64 {
        match self {
            Self::BitLinear1_58 => 4.5,
            Self::Q4_0 => 3.2,
            Self::Q8_0 => 1.9,
            Self::FP8 => 2.0,
            Self::FP32 => 1.0,
        }
    }

    /// Checks if this level is lower precision (higher compression) than `other`.
    pub fn is_more_compressed_than(&self, other: Self) -> bool {
        (*self as usize) < (other as usize)
    }
}

/// Real-time telemetry snapshot collected during LLM token generation.
#[derive(Debug, Clone)]
pub struct DynamicTelemetry {
    /// Generation latency for recent token(s) in milliseconds.
    pub token_latency_ms: f64,
    /// Available system / accelerator memory headroom in megabytes.
    pub memory_headroom_mb: f64,
    /// Current KV cache buffer utilization percentage (0.0 to 100.0).
    pub kv_cache_usage_pct: f64,
    /// Activation kurtosis ($\mu_4 / \sigma^4$) measuring presence of heavy-tailed outliers.
    pub activation_kurtosis: f64,
    /// Monotonically increasing generated token index.
    pub token_index: usize,
}

/// Policy governing dynamic autonomous quantization shifts.
#[derive(Debug, Clone)]
pub struct DynamicQuantPolicy {
    /// Target per-token latency budget (ms).
    pub target_latency_ms: f64,
    /// Minimum desired memory headroom before triggering down-quantization (MB).
    pub min_memory_headroom_mb: f64,
    /// Critical emergency memory threshold (MB).
    pub critical_memory_headroom_mb: f64,
    /// Kurtosis threshold above which layers resist aggressive down-quantization.
    pub max_acceptable_kurtosis: f64,
    /// KV-cache utilization threshold triggering compression (0.0 - 100.0).
    pub kv_cache_threshold_pct: f64,
    /// Minimum token interval between precision transitions to prevent thrashing.
    pub cooldown_tokens: usize,
}

impl Default for DynamicQuantPolicy {
    fn default() -> Self {
        Self {
            target_latency_ms: 25.0,
            min_memory_headroom_mb: 512.0,
            critical_memory_headroom_mb: 128.0,
            max_acceptable_kurtosis: 6.0,
            kv_cache_threshold_pct: 85.0,
            cooldown_tokens: 10,
        }
    }
}

/// A planned autonomous quantization shift.
#[derive(Debug, Clone)]
pub struct DynamicTransitionPlan {
    pub target_precision: DynamicPrecisionLevel,
    pub affected_layers: Vec<String>,
    pub estimated_memory_saved_mb: f64,
    pub expected_speedup: f64,
    pub trigger_reason: String,
}

/// Multi-precision adaptive linear layer supporting dynamic zero-downtime precision switching.
pub struct AdaptiveQuantLinear {
    master_linear: Linear,
    active_precision: DynamicPrecisionLevel,
    cached_fp8: Option<FP8Linear>,
    cached_q8: Option<QuantizedLinear>,
    cached_q4: Option<QuantizedLinear>,
    cached_bit: Option<BitLinear>,
}

impl AdaptiveQuantLinear {
    /// Creates a new `AdaptiveQuantLinear` initialized at full FP32 precision.
    pub fn new(master_linear: Linear) -> Self {
        Self {
            master_linear,
            active_precision: DynamicPrecisionLevel::FP32,
            cached_fp8: None,
            cached_q8: None,
            cached_q4: None,
            cached_bit: None,
        }
    }

    /// Current active precision level.
    pub fn active_precision(&self) -> DynamicPrecisionLevel {
        self.active_precision
    }

    /// Sets the active precision level, lazily compiling quantized weight views if needed.
    pub fn set_precision(&mut self, target: DynamicPrecisionLevel) -> Result<()> {
        match target {
            DynamicPrecisionLevel::FP32 => {}
            DynamicPrecisionLevel::FP8 => {
                if self.cached_fp8.is_none() {
                    let fp8 = FP8Linear::from_linear(&self.master_linear, FP8Format::E4M3)?;
                    self.cached_fp8 = Some(fp8);
                }
            }
            DynamicPrecisionLevel::Q8_0 => {
                if self.cached_q8.is_none() {
                    let q8 = QuantizedLinear::from_linear(
                        &self.master_linear,
                        QuantScheme::DEFAULT_Q8_0,
                    )?;
                    self.cached_q8 = Some(q8);
                }
            }
            DynamicPrecisionLevel::Q4_0 => {
                if self.cached_q4.is_none() {
                    let q4 = QuantizedLinear::from_linear(
                        &self.master_linear,
                        QuantScheme::DEFAULT_Q4_0,
                    )?;
                    self.cached_q4 = Some(q4);
                }
            }
            DynamicPrecisionLevel::BitLinear1_58 => {
                if self.cached_bit.is_none() {
                    let bit = BitLinear::from_linear(&self.master_linear)?;
                    self.cached_bit = Some(bit);
                }
            }
        }

        self.active_precision = target;
        Ok(())
    }

    /// Executes the forward pass using the currently active precision representation.
    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        match self.active_precision {
            DynamicPrecisionLevel::FP32 => self.master_linear.forward(input),
            DynamicPrecisionLevel::FP8 => {
                if self.cached_fp8.is_none() {
                    self.set_precision(DynamicPrecisionLevel::FP8)?;
                }
                self.cached_fp8.as_ref().unwrap().forward(input)
            }
            DynamicPrecisionLevel::Q8_0 => {
                if self.cached_q8.is_none() {
                    self.set_precision(DynamicPrecisionLevel::Q8_0)?;
                }
                self.cached_q8.as_ref().unwrap().forward(input)
            }
            DynamicPrecisionLevel::Q4_0 => {
                if self.cached_q4.is_none() {
                    self.set_precision(DynamicPrecisionLevel::Q4_0)?;
                }
                self.cached_q4.as_ref().unwrap().forward(input)
            }
            DynamicPrecisionLevel::BitLinear1_58 => {
                if self.cached_bit.is_none() {
                    self.set_precision(DynamicPrecisionLevel::BitLinear1_58)?;
                }
                self.cached_bit.as_ref().unwrap().forward(input)
            }
        }
    }

    /// Measures the relative Frobenius norm error between active quantized output and FP32 baseline:
    /// $$\text{RelErr} = \frac{\| Y_{\text{quant}} - Y_{\text{FP32}} \|_F}{\| Y_{\text{FP32}} \|_F + \epsilon}$$
    pub fn quantization_relative_error(&mut self, sample_input: &Tensor) -> Result<f64> {
        let y_ref = self.master_linear.forward(sample_input)?;
        let y_quant = self.forward(sample_input)?;

        let diff = y_quant.sub(&y_ref)?;
        let diff_vec = diff.to_vec::<f32>()?;
        let ref_vec = y_ref.to_vec::<f32>()?;

        let norm_diff: f64 = diff_vec
            .iter()
            .map(|&x| (x as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let norm_ref: f64 = ref_vec
            .iter()
            .map(|&x| (x as f64).powi(2))
            .sum::<f64>()
            .sqrt();

        Ok(norm_diff / (norm_ref + 1e-8))
    }

    /// Estimated memory consumption of the active representation in bytes.
    pub fn active_memory_bytes(&self) -> usize {
        let num_params = self.master_linear.in_features() * self.master_linear.out_features();
        match self.active_precision {
            DynamicPrecisionLevel::FP32 => num_params * 4,
            DynamicPrecisionLevel::FP8 => num_params,
            DynamicPrecisionLevel::Q8_0 => self
                .cached_q8
                .as_ref()
                .map(|q| q.weight_bytes())
                .unwrap_or(num_params),
            DynamicPrecisionLevel::Q4_0 => self
                .cached_q4
                .as_ref()
                .map(|q| q.weight_bytes())
                .unwrap_or(num_params / 2),
            DynamicPrecisionLevel::BitLinear1_58 => num_params.div_ceil(4),
        }
    }
}

/// Dynamic Autonomous Quantization Controller for LLMs.
pub struct DynamicLlmQuantizer {
    pub layers: BTreeMap<String, AdaptiveQuantLinear>,
    pub policy: DynamicQuantPolicy,
    pub last_transition_token: usize,
}

impl DynamicLlmQuantizer {
    /// Creates a new `DynamicLlmQuantizer`.
    pub fn new(policy: DynamicQuantPolicy) -> Self {
        Self {
            layers: BTreeMap::new(),
            policy,
            last_transition_token: 0,
        }
    }

    /// Registers a new named linear layer under autonomous management.
    pub fn add_layer(&mut self, name: &str, linear: Linear) {
        self.layers
            .insert(name.to_string(), AdaptiveQuantLinear::new(linear));
    }

    /// Executes the forward pass of a named layer.
    pub fn forward_layer(&mut self, name: &str, input: &Tensor) -> Result<Tensor> {
        let layer = self.layers.get_mut(name).ok_or_else(|| {
            Error::Msg(format!(
                "Layer '{name}' not registered in DynamicLlmQuantizer"
            ))
        })?;
        layer.forward(input)
    }

    /// Evaluates current generation telemetry against policy to determine if an autonomous
    /// precision transition should be enacted.
    pub fn evaluate_telemetry(
        &mut self,
        telemetry: &DynamicTelemetry,
    ) -> Option<DynamicTransitionPlan> {
        // Cooldown check to eliminate flapping / thrashing
        let cooldown_end = self
            .last_transition_token
            .saturating_add(self.policy.cooldown_tokens);
        if telemetry.token_index < cooldown_end {
            return None;
        }

        // Determine prevailing precision across layers
        let current_precisions: Vec<DynamicPrecisionLevel> =
            self.layers.values().map(|l| l.active_precision()).collect();
        if current_precisions.is_empty() {
            return None;
        }
        let dominant_precision = *current_precisions
            .iter()
            .min()
            .unwrap_or(&DynamicPrecisionLevel::FP32);

        // 1. Critical Memory Exhaustion -> Emergency Downgrade to BitLinear1_58 or Q4_0
        if (telemetry.memory_headroom_mb < self.policy.critical_memory_headroom_mb
            || telemetry.kv_cache_usage_pct > 95.0)
            && dominant_precision > DynamicPrecisionLevel::BitLinear1_58
        {
            let target = DynamicPrecisionLevel::BitLinear1_58;
            return Some(self.build_plan(
                target,
                format!(
                    "CRITICAL_MEMORY_PRESSURE: headroom {:.1}MB < {:.1}MB or KV-cache {:.1}% > 95%",
                    telemetry.memory_headroom_mb,
                    self.policy.critical_memory_headroom_mb,
                    telemetry.kv_cache_usage_pct
                ),
            ));
        }

        // 2. High Memory Pressure / KV-Cache Capacity -> Downgrade to Q4_0
        if (telemetry.memory_headroom_mb < self.policy.min_memory_headroom_mb
            || telemetry.kv_cache_usage_pct > self.policy.kv_cache_threshold_pct)
            && dominant_precision > DynamicPrecisionLevel::Q4_0
        {
            let target = DynamicPrecisionLevel::Q4_0;
            return Some(self.build_plan(
                target,
                format!(
                    "MEMORY_HEADROOM_EXHAUSTION: headroom {:.1}MB or KV-cache {:.1}%",
                    telemetry.memory_headroom_mb, telemetry.kv_cache_usage_pct
                ),
            ));
        }

        // 3. High Latency Violation -> Step down precision unless activation kurtosis is excessive
        if telemetry.token_latency_ms > self.policy.target_latency_ms {
            let next_target = match dominant_precision {
                DynamicPrecisionLevel::FP32 => Some(DynamicPrecisionLevel::FP8),
                DynamicPrecisionLevel::FP8 => {
                    // If kurtosis is high, avoid down-quantizing below FP8 to preserve outlier sensitivity
                    if telemetry.activation_kurtosis <= self.policy.max_acceptable_kurtosis {
                        Some(DynamicPrecisionLevel::Q4_0)
                    } else {
                        None
                    }
                }
                DynamicPrecisionLevel::Q8_0 => Some(DynamicPrecisionLevel::Q4_0),
                DynamicPrecisionLevel::Q4_0 => Some(DynamicPrecisionLevel::BitLinear1_58),
                DynamicPrecisionLevel::BitLinear1_58 => None,
            };

            if let Some(target) = next_target {
                return Some(self.build_plan(
                    target,
                    format!(
                        "LATENCY_BUDGET_EXCEEDED: {:.2}ms > {:.2}ms (kurtosis: {:.2})",
                        telemetry.token_latency_ms,
                        self.policy.target_latency_ms,
                        telemetry.activation_kurtosis
                    ),
                ));
            }
        }

        // 4. Generous Headroom -> Upgrade precision for higher generative accuracy
        if telemetry.memory_headroom_mb > self.policy.min_memory_headroom_mb * 2.5
            && telemetry.token_latency_ms < self.policy.target_latency_ms * 0.65
            && dominant_precision < DynamicPrecisionLevel::FP32
        {
            let next_target = match dominant_precision {
                DynamicPrecisionLevel::BitLinear1_58 => DynamicPrecisionLevel::Q4_0,
                DynamicPrecisionLevel::Q4_0 => DynamicPrecisionLevel::FP8,
                DynamicPrecisionLevel::Q8_0 => DynamicPrecisionLevel::FP8,
                DynamicPrecisionLevel::FP8 => DynamicPrecisionLevel::FP32,
                DynamicPrecisionLevel::FP32 => DynamicPrecisionLevel::FP32,
            };

            if next_target != dominant_precision {
                return Some(self.build_plan(
                    next_target,
                    format!(
                        "GENEROUS_HEADROOM_FIDELITY_UPGRADE: headroom {:.1}MB, latency {:.2}ms",
                        telemetry.memory_headroom_mb, telemetry.token_latency_ms
                    ),
                ));
            }
        }

        None
    }

    /// Enacts an autonomous transition plan across managed layers.
    pub fn apply_plan(&mut self, plan: &DynamicTransitionPlan, current_token: usize) -> Result<()> {
        for name in &plan.affected_layers {
            if let Some(layer) = self.layers.get_mut(name) {
                layer.set_precision(plan.target_precision)?;
            }
        }
        self.last_transition_token = current_token;
        Ok(())
    }

    /// Total memory currently consumed across all managed layers in megabytes.
    pub fn total_memory_mb(&self) -> f64 {
        let total_bytes: usize = self.layers.values().map(|l| l.active_memory_bytes()).sum();
        total_bytes as f64 / (1024.0 * 1024.0)
    }

    fn build_plan(
        &self,
        target_precision: DynamicPrecisionLevel,
        reason: String,
    ) -> DynamicTransitionPlan {
        let affected_layers: Vec<String> = self.layers.keys().cloned().collect();
        let cur_bytes: usize = self.layers.values().map(|l| l.active_memory_bytes()).sum();
        let target_ratio = target_precision.compression_ratio();
        // Savings are estimated against the CURRENT mixed-precision footprint,
        // not a full-FP32 baseline.
        let baseline_bytes = cur_bytes.max(1);
        let est_saved_mb =
            ((baseline_bytes as f64) * (1.0 - 1.0 / target_ratio)) / (1024.0 * 1024.0);

        DynamicTransitionPlan {
            target_precision,
            affected_layers,
            estimated_memory_saved_mb: est_saved_mb.max(0.0),
            expected_speedup: target_precision.estimated_speedup(),
            trigger_reason: reason,
        }
    }
}
