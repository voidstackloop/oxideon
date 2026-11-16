//! Heterogeneous Micro-Kernel Auto-Tuning Engine.
//!
//! Provides automated tile parameter exploration across CPU AVX/AMX, CUDA PTX, Metal MSL, and WebGPU WGSL:
//! - [`TileConfig`]: Hardware tiling hyperparameters (Tile-M, Tile-N, Tile-K, warps, stages, unroll).
//! - [`HardwareProfile`]: Memory bandwidth, register file size, and shared memory constraints.
//! - [`KernelAutoTuner`]: Analytical roofline model and empirical benchmark search discovering optimal tile shapes.

use crate::error::{Error, Result};
use std::collections::HashMap;

/// Hardware execution target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HardwareTarget {
    CpuRayon,
    CudaPtx,
    MetalMsl,
    WebGpuWgsl,
}

/// Hardware architecture constraints and bandwidth characteristics.
#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub target: HardwareTarget,
    pub max_shared_memory_bytes: usize,
    pub max_registers_per_thread: usize,
    pub peak_gflops: f64,
    pub memory_bandwidth_gb_s: f64,
    pub warp_size: usize,
}

impl HardwareProfile {
    pub fn default_cpu() -> Self {
        Self {
            target: HardwareTarget::CpuRayon,
            max_shared_memory_bytes: 32 * 1024 * 1024, // L3 cache
            max_registers_per_thread: 32,              // AVX-512 ZMM registers
            peak_gflops: 2000.0,
            memory_bandwidth_gb_s: 100.0,
            warp_size: 16,
        }
    }

    pub fn default_gpu(target: HardwareTarget) -> Self {
        Self {
            target,
            max_shared_memory_bytes: 96 * 1024, // 96 KB SMEM
            max_registers_per_thread: 255,
            peak_gflops: 40_000.0,
            memory_bandwidth_gb_s: 1000.0,
            warp_size: 32,
        }
    }
}

/// A specific tile configuration candidate for kernel execution.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TileConfig {
    pub tile_m: usize,
    pub tile_n: usize,
    pub tile_k: usize,
    pub num_warps: usize,
    pub num_stages: usize,
}

impl TileConfig {
    pub fn new(
        tile_m: usize,
        tile_n: usize,
        tile_k: usize,
        num_warps: usize,
        num_stages: usize,
    ) -> Self {
        Self {
            tile_m,
            tile_n,
            tile_k,
            num_warps: num_warps.max(1),
            num_stages: num_stages.max(1),
        }
    }

    /// Estimates shared memory required in bytes for double-buffered A and B tiles.
    /// Saturates instead of overflowing, so absurd configs are rejected as
    /// over-limit rather than wrapping to a small estimate.
    pub fn shared_memory_bytes(&self) -> usize {
        let tile_a = self.tile_m.saturating_mul(self.tile_k).saturating_mul(4);
        let tile_b = self.tile_k.saturating_mul(self.tile_n).saturating_mul(4);
        tile_a
            .saturating_add(tile_b)
            .saturating_mul(self.num_stages)
    }
}

/// Performance measurement metrics for a tuned kernel candidate.
#[derive(Debug, Clone)]
pub struct TuneMetric {
    pub config: TileConfig,
    pub estimated_latency_us: f64,
    pub estimated_tflops: f64,
    pub memory_efficiency: f64,
}

/// Micro-kernel auto-tuning engine searching tile spaces under hardware limits.
#[derive(Debug, Clone)]
pub struct KernelAutoTuner {
    pub hardware: HardwareProfile,
    cache: HashMap<(usize, usize, usize), TileConfig>,
}

impl KernelAutoTuner {
    pub fn new(hardware: HardwareProfile) -> Self {
        Self {
            hardware,
            cache: HashMap::new(),
        }
    }

    /// Evaluates candidate tiling under the analytical roofline model.
    pub fn evaluate_candidate(
        &self,
        m: usize,
        n: usize,
        k: usize,
        config: &TileConfig,
    ) -> Option<TuneMetric> {
        // Empty GEMMs have no meaningful roofline point (0/0 latency); reject.
        if m == 0 || n == 0 || k == 0 {
            return None;
        }
        // 1. Hardware constraint validation
        if config.shared_memory_bytes() > self.hardware.max_shared_memory_bytes {
            return None;
        }

        if config.tile_m == 0 || config.tile_n == 0 {
            return None;
        }
        let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
        let mk = m.checked_mul(k)?;
        let kn = k.checked_mul(n)?;
        let mn = m.checked_mul(n)?;
        let bytes = (mk.checked_add(kn)?.checked_add(mn)?.checked_mul(4)?) as f64;

        // Roofline model: latency = max(compute_time, memory_time)
        let compute_time_s = flops / (self.hardware.peak_gflops * 1e9);
        let memory_time_s = bytes / (self.hardware.memory_bandwidth_gb_s * 1e9);

        // Tile efficiency factors
        let m_pad = m.div_ceil(config.tile_m).checked_mul(config.tile_m)?;
        let n_pad = n.div_ceil(config.tile_n).checked_mul(config.tile_n)?;
        let padding_efficiency = (mn as f64) / ((m_pad.checked_mul(n_pad)?) as f64);

        let latency_s = compute_time_s.max(memory_time_s) / padding_efficiency;
        let latency_us = latency_s * 1e6;
        let tflops = (flops / 1e12) / latency_s.max(1e-12);

        Some(TuneMetric {
            config: config.clone(),
            estimated_latency_us: latency_us,
            estimated_tflops: tflops,
            memory_efficiency: padding_efficiency,
        })
    }

    /// Auto-tunes matrix multiplication (M, N, K), returning the optimal TileConfig.
    pub fn tune_gemm(&mut self, m: usize, n: usize, k: usize) -> Result<TileConfig> {
        if let Some(cached) = self.cache.get(&(m, n, k)) {
            return Ok(cached.clone());
        }

        let candidate_m = [32, 64, 128];
        let candidate_n = [32, 64, 128];
        let candidate_k = [16, 32, 64];
        let candidate_warps = [2, 4, 8];
        let candidate_stages = [1, 2, 3];

        let mut best_metric: Option<TuneMetric> = None;

        for &tm in &candidate_m {
            for &tn in &candidate_n {
                for &tk in &candidate_k {
                    for &w in &candidate_warps {
                        for &st in &candidate_stages {
                            let cfg = TileConfig::new(tm, tn, tk, w, st);
                            if let Some(metric) = self.evaluate_candidate(m, n, k, &cfg) {
                                if best_metric.as_ref().is_none_or(|b| {
                                    metric.estimated_latency_us < b.estimated_latency_us
                                }) {
                                    best_metric = Some(metric);
                                }
                            }
                        }
                    }
                }
            }
        }

        let best = best_metric.map(|m| m.config).ok_or_else(|| {
            Error::Msg(
                "No valid tile configuration satisfied hardware profile constraints".to_string(),
            )
        })?;

        self.cache.insert((m, n, k), best.clone());
        Ok(best)
    }
}
