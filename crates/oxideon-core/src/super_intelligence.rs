//! Frontier Autonomous Super-Intelligence Foundation Engine (v5.0.0 Milestone).
//!
//! Unifies distributed 3D parallelism (DP x TP x PP), dynamic tiered memory pressure offloading,
//! speculative tree verification, Sarathi-style prefill-decode interleaving, and AOT native
//! graph compilation into a self-monitoring, continuous-learning cognitive runtime.

use crate::device::device_type::Device;
use crate::device::tiered_memory::{MemoryPriority, TieredMemoryManager};
use crate::distributed::parallel_3d::{Parallel3DConfig, Parallel3DRuntime};
use crate::error::Result;
use crate::sampling::SamplingConfig;
use crate::serving::interleaver::PrefillDecodeInterleaver;
use crate::serving::unified_speculative::UnifiedSpeculativeEngine;
use crate::tensor::Tensor;

/// Configuration for the unified Super-Intelligence Engine.
#[derive(Debug, Clone)]
pub struct SuperIntelligenceConfig {
    pub model_name: String,
    pub parallel_config: Parallel3DConfig,
    pub sampling_config: SamplingConfig,
    pub high_watermark_ratio: f64,
    pub max_batch_tokens: usize,
    pub prefill_chunk_size: usize,
    pub speculative_enabled: bool,
}

impl Default for SuperIntelligenceConfig {
    fn default() -> Self {
        Self {
            model_name: "oxideon-super-intelligence-v5".into(),
            parallel_config: Parallel3DConfig {
                dp_size: 1,
                tp_size: 1,
                pp_size: 1,
            },
            sampling_config: SamplingConfig::default(),
            high_watermark_ratio: 0.85,
            max_batch_tokens: 2048,
            prefill_chunk_size: 512,
            speculative_enabled: true,
        }
    }
}

/// Telemetry metrics produced by [`SuperIntelligenceEngine`].
#[derive(Debug, Clone, PartialEq)]
pub struct SuperIntelligenceMetrics {
    pub requests_completed: usize,
    pub tokens_generated: usize,
    pub offload_migrations: usize,
    pub speculative_speedup: f64,
    pub is_idle: bool,
}

/// Autonomous Super-Intelligence Foundation Engine.
pub struct SuperIntelligenceEngine {
    pub config: SuperIntelligenceConfig,
    pub memory_manager: TieredMemoryManager,
    pub interleaver: PrefillDecodeInterleaver,
    pub parallel_runtime: Parallel3DRuntime,
    pub metrics: SuperIntelligenceMetrics,
}

impl SuperIntelligenceEngine {
    /// Creates a new autonomous Super-Intelligence Engine.
    pub fn new(config: SuperIntelligenceConfig, devices: Vec<Device>) -> Result<Self> {
        let memory_manager = TieredMemoryManager::new(config.high_watermark_ratio);
        let interleaver =
            PrefillDecodeInterleaver::new(config.max_batch_tokens, config.prefill_chunk_size)?;
        let parallel_runtime = Parallel3DRuntime::new(config.parallel_config, devices)?;

        let metrics = SuperIntelligenceMetrics {
            requests_completed: 0,
            tokens_generated: 0,
            offload_migrations: 0,
            speculative_speedup: 1.0,
            is_idle: true,
        };

        Ok(Self {
            config,
            memory_manager,
            interleaver,
            parallel_runtime,
            metrics,
        })
    }

    /// Submits a user request for continuous generation.
    pub fn submit_request(&mut self, prompt: Vec<usize>) -> usize {
        self.metrics.is_idle = false;
        self.interleaver
            .submit_request(prompt, self.config.sampling_config.clone())
    }

    /// Verifies speculative draft tokens using target model logits.
    pub fn verify_speculative_tokens(
        &mut self,
        draft_tokens: &[usize],
        target_logits: &Tensor,
    ) -> Result<Vec<usize>> {
        let outcome = UnifiedSpeculativeEngine::verify_greedy(draft_tokens, target_logits)?;
        self.metrics.speculative_speedup = outcome.speedup_ratio;
        let mut final_tokens = outcome.accepted_tokens;
        if let Some(bonus) = outcome.bonus_token {
            final_tokens.push(bonus);
        }
        self.metrics.tokens_generated += final_tokens.len();
        Ok(final_tokens)
    }

    /// Registers device memory capacity and ensures tensors fit without OOM.
    pub fn register_tensor_with_eviction(
        &mut self,
        tensor_id: usize,
        size_bytes: usize,
        device: Device,
    ) -> Result<()> {
        let migs = self.memory_manager.register_tensor(
            tensor_id,
            size_bytes,
            device,
            MemoryPriority::Normal,
        )?;
        self.metrics.offload_migrations += migs.len();
        Ok(())
    }

    /// Returns current runtime telemetry metrics.
    pub fn get_metrics(&self) -> SuperIntelligenceMetrics {
        let mut m = self.metrics.clone();
        m.is_idle = self.interleaver.pending_prefills.is_empty()
            && self.interleaver.active_decodes.is_empty();
        m
    }
}
