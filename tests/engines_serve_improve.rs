// Engines plus serving improvement tests.
// Scope: serving, sampling, diffusion, quant, distributed, compression,
// tokenizer, oxideon-engines. Wrong calls must return Err, never panic,
// hang, or silently miscompute.

use oxideon::diffusion::EDMConfig;
use oxideon::distributed::{ZeRO3Parameter, sync_gradients_ddp};
use oxideon::error::Result;
use oxideon::nn::{Linear, Module};
use oxideon::prelude::*;
use oxideon::quant::{AWQLinear, BitLinear, FP8Format, FP8Linear};
use oxideon::quant::{GPTQLinear, QuantScheme, QuantizedLinear};
use oxideon::sampling::SamplingConfig;
use oxideon::serving::{BlockAllocator, ContinuousBatchScheduler, ServingEngine};

#[test]
fn engine_step_short_forward_output_is_error() -> Result<()> {
    let mut engine = ServingEngine::new(4, 8, 4, 2, 4)?;
    engine.submit(vec![1, 2, 3], 5);
    engine.submit(vec![4, 5], 5);
    let out = engine.step(|_reqs| Ok(vec![7]));
    assert!(out.is_err());
    Ok(())
}

#[test]
fn engine_step_matching_forward_output_still_works() -> Result<()> {
    let mut engine = ServingEngine::new(4, 8, 4, 2, 4)?;
    engine.submit(vec![1, 2, 3], 5);
    let out = engine.step(|reqs| Ok(vec![7; reqs.len()]))?;
    assert_eq!(out, vec![(1, 7)]);
    Ok(())
}

#[test]
fn engine_rejects_zero_block_size() {
    assert!(ServingEngine::new(4, 8, 0, 2, 4).is_err());
    let mut sched = ContinuousBatchScheduler::new(4, 512);
    let mut alloc = BlockAllocator::new(4, 0).unwrap();
    sched.submit(vec![1, 2, 3], 5);
    assert!(sched.schedule_step(&mut alloc).is_err());
}

#[test]
fn seeded_sampling_without_generator_advances_stream() -> Result<()> {
    let logits = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4])?;
    let cfg = SamplingConfig::new().with_temperature(1.0).with_seed(42);
    let draws: Vec<usize> = (0..16)
        .map(|_| logits.sample_token(&cfg, None, None))
        .collect::<Result<Vec<_>>>()?;
    let first = draws[0];
    assert!(draws.iter().any(|&d| d != first));
    Ok(())
}

#[test]
fn seeded_sampling_without_generator_is_reproducible() -> Result<()> {
    let logits = Tensor::from_slice(&[0.5f32, 2.0, 1.0, 0.25], &[4])?;
    let run = || -> Result<Vec<usize>> {
        let cfg = SamplingConfig::new().with_temperature(1.0).with_seed(7);
        (0..8)
            .map(|_| logits.sample_token(&cfg, None, None))
            .collect::<Result<Vec<_>>>()
    };
    assert_eq!(run()?, run()?);
    Ok(())
}

#[test]
fn zero3_rejects_zero_world_size() -> Result<()> {
    let w = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2])?;
    assert!(ZeRO3Parameter::from_parameter(&w, 0).is_err());
    let mut p = ZeRO3Parameter::from_parameter(&w, 2)?;
    assert_eq!(p.gather()?.shape(), &[2, 2]);
    Ok(())
}

#[test]
fn ddp_partial_grads_are_an_error() -> Result<()> {
    let a = Tensor::from_slice(&[1.0f32, 2.0], &[2])?;
    let b = Tensor::from_slice(&[3.0f32, 4.0], &[2])?;
    let g = Tensor::from_slice(&[0.5f32, 0.5], &[2])?;
    a.set_grad(Some(g));
    let replicas = vec![vec![a], vec![b]];
    assert!(sync_gradients_ddp(&replicas).is_err());
    Ok(())
}

#[test]
fn ddp_all_missing_grads_stays_ok() -> Result<()> {
    let a = Tensor::from_slice(&[1.0f32, 2.0], &[2])?;
    let b = Tensor::from_slice(&[3.0f32, 4.0], &[2])?;
    let replicas = vec![vec![a], vec![b]];
    assert!(sync_gradients_ddp(&replicas).is_ok());
    Ok(())
}

#[test]
fn ddp_full_grads_average() -> Result<()> {
    let a = Tensor::from_slice(&[1.0f32, 2.0], &[2])?;
    let b = Tensor::from_slice(&[3.0f32, 4.0], &[2])?;
    a.set_grad(Some(Tensor::from_slice(&[1.0f32, 1.0], &[2])?));
    b.set_grad(Some(Tensor::from_slice(&[3.0f32, 3.0], &[2])?));
    let replicas = vec![vec![a.clone()], vec![b.clone()]];
    sync_gradients_ddp(&replicas)?;
    assert_eq!(a.grad().unwrap().to_vec::<f32>()?, vec![2.0, 2.0]);
    assert_eq!(b.grad().unwrap().to_vec::<f32>()?, vec![2.0, 2.0]);
    Ok(())
}

fn scalar_f32() -> Result<Tensor> {
    Tensor::from_slice(&[1.0f32], &[])
}

#[test]
fn quantized_linear_rejects_scalar_and_mismatch() -> Result<()> {
    let lin = Linear::with_bias(4, 3, true)?;
    let q = QuantizedLinear::from_linear(&lin, QuantScheme::DEFAULT_Q8_0)?;
    assert!(q.forward(&scalar_f32()?).is_err());
    let bad = Tensor::from_slice(&[1.0f32; 6], &[2, 3])?;
    assert!(q.forward(&bad).is_err());
    let good = Tensor::from_slice(&[1.0f32; 8], &[2, 4])?;
    assert_eq!(q.forward(&good)?.shape(), &[2, 3]);
    Ok(())
}

#[test]
fn awq_linear_rejects_scalar_and_mismatch() -> Result<()> {
    let lin = Linear::with_bias(4, 3, true)?;
    let calib = Tensor::from_slice(&[0.5f32; 8], &[2, 4])?;
    let q = AWQLinear::from_linear_calibrated(&lin, &calib, QuantScheme::DEFAULT_Q8_0, 0.5)?;
    assert!(q.forward(&scalar_f32()?).is_err());
    let bad = Tensor::from_slice(&[1.0f32; 6], &[2, 3])?;
    assert!(q.forward(&bad).is_err());
    let good = Tensor::from_slice(&[1.0f32; 8], &[2, 4])?;
    assert_eq!(q.forward(&good)?.shape(), &[2, 3]);
    Ok(())
}

#[test]
fn fp8_linear_rejects_scalar_and_mismatch() -> Result<()> {
    let lin = Linear::with_bias(4, 3, true)?;
    let q = FP8Linear::from_linear(&lin, FP8Format::E4M3)?;
    assert!(q.forward(&scalar_f32()?).is_err());
    let bad = Tensor::from_slice(&[1.0f32; 6], &[2, 3])?;
    assert!(q.forward(&bad).is_err());
    let good = Tensor::from_slice(&[1.0f32; 8], &[2, 4])?;
    assert_eq!(q.forward(&good)?.shape(), &[2, 3]);
    Ok(())
}

#[test]
fn gptq_linear_rejects_scalar_and_mismatch() -> Result<()> {
    let lin = Linear::with_bias(4, 3, true)?;
    let calib = Tensor::from_slice(&[0.5f32; 12], &[3, 4])?;
    let q = GPTQLinear::from_linear_calibrated(&lin, &calib, QuantScheme::DEFAULT_Q8_0, 0.01)?;
    assert!(q.forward(&scalar_f32()?).is_err());
    let bad = Tensor::from_slice(&[1.0f32; 6], &[2, 3])?;
    assert!(q.forward(&bad).is_err());
    let good = Tensor::from_slice(&[1.0f32; 8], &[2, 4])?;
    assert_eq!(q.forward(&good)?.shape(), &[2, 3]);
    Ok(())
}

#[test]
fn bitlinear_rejects_scalar_mismatch_and_zero_features() -> Result<()> {
    let lin = Linear::with_bias(8, 4, true)?;
    let bit = BitLinear::from_linear(&lin)?;
    let flat = Tensor::from_slice(&[1.0f32; 16], &[16])?;
    assert!(bit.forward(&flat).is_err());
    assert!(bit.forward(&scalar_f32()?).is_err());
    let good = Tensor::from_slice(&[1.0f32; 16], &[2, 8])?;
    assert_eq!(bit.forward(&good)?.shape(), &[2, 4]);
    let mut broken = BitLinear::from_linear(&lin)?;
    broken.in_features = 0;
    assert!(broken.forward(&good).is_err());
    Ok(())
}

#[test]
fn edm_build_sigmas_degenerate_steps_have_no_nan() {
    for n in [0usize, 1] {
        let cfg = EDMConfig {
            num_steps: n,
            ..Default::default()
        };
        let sigmas = cfg.build_sigmas();
        assert!(sigmas.iter().all(|s| s.is_finite()));
        assert!(sigmas.len() >= 3);
    }
}

use oxideon::device::{Device, DeviceMesh};
use oxideon::diffusion::{BetaSchedule, DDPMScheduler, DiffusionForcingScheduler};
use oxideon::distributed::federated::{ByzantineAggregator, DpSgdConfig};
use oxideon::distributed::sharded_tensor::{AutoParallelPlanner, Placement, ShardedTensor};
use oxideon::quant::{QuantizedTensor, TernaryPackedTensor};
use oxideon::serving::{
    DoubleBufferedKVCache, EagleDraftHead, PagedKVCache, QuantKVDType, QuantizedPagedKVCache,
    UnifiedSpeculativeEngine,
};
use std::sync::Arc;

#[test]
fn stochastic_verify_short_probs_is_error() -> Result<()> {
    let draft = vec![1usize, 2];
    let full_target = vec![vec![0.1, 0.7, 0.2], vec![0.2, 0.2, 0.6]];
    let short_target = vec![vec![0.1, 0.7, 0.2]];
    let r1 = UnifiedSpeculativeEngine::verify_stochastic(&draft, &[0.5], &full_target, Some(1));
    assert!(r1.is_err());
    let r2 =
        UnifiedSpeculativeEngine::verify_stochastic(&draft, &[0.5, 0.5], &short_target, Some(1));
    assert!(r2.is_err());
    let ok =
        UnifiedSpeculativeEngine::verify_stochastic(&draft, &[0.5, 0.5], &full_target, Some(1))?;
    assert!(ok.num_accepted <= 2);
    Ok(())
}

#[test]
fn forcing_step_oob_timestep_is_error() -> Result<()> {
    let forcing = DiffusionForcingScheduler::new(100, BetaSchedule::default())?;
    let xt = Tensor::randn(&[1, 2, 4])?;
    let eps = Tensor::randn(&[1, 2, 4])?;
    assert!(
        forcing
            .step_per_token(&eps, &[10, 5000], &[5, 5], &xt)
            .is_err()
    );
    assert!(
        forcing
            .step_per_token(&eps, &[10, 20], &[5, 5000], &xt)
            .is_err()
    );
    assert!(
        forcing
            .step_per_token(&eps, &[10, 20], &[5, 5], &xt)
            .is_ok()
    );
    Ok(())
}

#[test]
fn krum_ragged_gradients_is_error() -> Result<()> {
    let a = Tensor::from_slice(&[1.0f32, 2.0], &[2])?;
    let b = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3])?;
    let c = Tensor::from_slice(&[0.0f32, 1.0], &[2])?;
    assert!(ByzantineAggregator::multi_krum(&[a.clone(), b.clone(), c], 0, 1).is_err());
    assert!(ByzantineAggregator::bulyan(&[a, b], 0).is_err());
    Ok(())
}

#[test]
fn trimmed_mean_nan_fraction_is_error() -> Result<()> {
    let a = Tensor::from_slice(&[1.0f32, 2.0], &[2])?;
    let b = Tensor::from_slice(&[3.0f32, 4.0], &[2])?;
    assert!(ByzantineAggregator::trimmed_mean(&[a, b], f64::NAN).is_err());
    Ok(())
}

#[test]
fn dpsgd_invalid_params_is_error() -> Result<()> {
    let g = Tensor::from_slice(&[1.0f32, 2.0], &[2])?;
    assert!(DpSgdConfig::new(-1.0, 0.1).clip_and_noise(&g).is_err());
    assert!(DpSgdConfig::new(1.0, f64::NAN).clip_and_noise(&g).is_err());
    assert!(DpSgdConfig::new(1.0, 0.0).clip_and_noise(&g).is_ok());
    Ok(())
}

#[test]
fn sharded_from_global_bad_axis_is_error() -> Result<()> {
    let mesh = DeviceMesh::new(&[2], &["x"], vec![Device::Cpu, Device::Cpu])?;
    let global = Tensor::zeros(&[4, 4], DType::F32, &Device::Cpu)?;
    assert!(ShardedTensor::from_global(&global, &mesh, &[Placement::Shard(7)], &[0]).is_err());
    let ok = ShardedTensor::from_global(&global, &mesh, &[Placement::Shard(0)], &[0])?;
    assert_eq!(ok.local_tensor.shape(), &[2, 4]);
    Ok(())
}

#[test]
fn sharded_matmul_mismatched_meta_is_error() -> Result<()> {
    let mesh = DeviceMesh::new(&[1], &["x"], vec![Device::Cpu])?;
    let local = Tensor::zeros(&[2, 2], DType::F32, &Device::Cpu)?;
    let mk = |placements: Vec<Placement>| ShardedTensor {
        global_shape: vec![2, 2],
        placements,
        mesh: mesh.clone(),
        local_tensor: local.clone(),
        device: Device::Cpu,
    };
    let a = mk(vec![Placement::Replicate, Placement::Replicate]);
    let b = mk(vec![Placement::Replicate]);
    assert!(a.sharded_matmul(&b, &[0]).is_err());
    Ok(())
}

#[test]
fn auto_parallel_empty_mesh_is_error() -> Result<()> {
    let mesh = DeviceMesh::new(&[0], &["x"], vec![])?;
    let planner = AutoParallelPlanner::new();
    assert!(planner.plan_optimal_sharding(&mesh, 1, 8, 64).is_err());
    Ok(())
}

#[test]
fn eagle_head_huge_hidden_dim_is_error() -> Result<()> {
    assert!(EagleDraftHead::new(usize::MAX / 2 + 1, 8).is_err());
    assert!(EagleDraftHead::new(8, 8).is_ok());
    Ok(())
}

#[test]
fn double_buffer_huge_dims_is_error() -> Result<()> {
    assert!(DoubleBufferedKVCache::new(0, 1 << 32, 1 << 32, 1, DType::F32).is_err());
    assert!(DoubleBufferedKVCache::new(2, 4, 2, 4, DType::F32).is_ok());
    Ok(())
}

#[test]
fn submit_swap_saturates_huge_bytes() -> Result<()> {
    let mut cache = DoubleBufferedKVCache::new(2, 4, 2, 4, DType::F32)?;
    cache.bytes_per_block = usize::MAX;
    let engine = run_pipeline_for_test(cache);
    assert!(engine >= 1);
    Ok(())
}

fn run_pipeline_for_test(cache: DoubleBufferedKVCache) -> usize {
    let engine = oxideon::serving::AsyncKVPipelineEngine::new(Arc::new(cache));
    let t = engine.submit_swap_in(1, 2);
    engine.wait_all();
    let events = engine.poll_completed();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].bytes_transferred, usize::MAX);
    t
}

#[test]
fn dequantize_corrupt_layout_is_error() -> Result<()> {
    let zero_block = QuantizedTensor {
        scheme: QuantScheme::Q8_0 { block_size: 0 },
        shape: vec![4],
        scales: vec![],
        data: vec![0; 4],
    };
    assert!(zero_block.dequantize(DType::F32).is_err());
    let short_scales = QuantizedTensor {
        scheme: QuantScheme::DEFAULT_Q8_0,
        shape: vec![64],
        scales: vec![],
        data: vec![0; 64],
    };
    assert!(short_scales.dequantize(DType::F32).is_err());
    Ok(())
}

#[test]
fn ternary_pack_nan_scale_is_error() -> Result<()> {
    let w = Tensor::from_slice(&[1.0f32, -2.0, 0.5], &[3])?;
    assert!(TernaryPackedTensor::pack(&w, f32::NAN).is_err());
    assert!(TernaryPackedTensor::pack(&w, 1.0).is_ok());
    Ok(())
}

#[test]
fn add_noise_scalar_is_error() -> Result<()> {
    let sched = DDPMScheduler::new(10, BetaSchedule::default())?;
    let x = scalar_f32()?;
    assert!(sched.add_noise(&x, &x, &[0]).is_err());
    Ok(())
}

#[test]
fn quant_kv_inconsistent_dims_is_error() -> Result<()> {
    let mut cache = QuantizedPagedKVCache::new(2, 4, 2, 4, QuantKVDType::Int8)?;
    cache.num_blocks = 100;
    let kv = Tensor::from_slice(&[1.0f32; 8], &[8])?;
    assert!(cache.write_token(50, 0, &kv, &kv).is_err());
    assert!(cache.read_token_dequantized(50, 0).is_err());
    Ok(())
}

#[test]
fn paged_write_inconsistent_dims_is_error() -> Result<()> {
    let mut cache = PagedKVCache::new(2, 4, 2, 4, DType::F32)?;
    cache.num_blocks = 100;
    let kv = Tensor::from_slice(&[1.0f32; 8], &[8])?;
    assert!(cache.write_token(50, 0, &kv, &kv).is_err());
    Ok(())
}

#[test]
fn awq_zero_features_calibration_is_error() -> Result<()> {
    let lin = Linear::with_bias(0, 2, false)?;
    let calib = Tensor::from_slice(&[0.5f32; 4], &[2, 2])?;
    assert!(
        AWQLinear::from_linear_calibrated(&lin, &calib, QuantScheme::DEFAULT_Q8_0, 0.5).is_err()
    );
    Ok(())
}

#[test]
fn sampling_config_clone_eq_ignore_counter() -> Result<()> {
    let logits = Tensor::from_slice(&[0.5f32, 2.0, 1.0, 0.25], &[4])?;
    let a = SamplingConfig::new().with_temperature(1.0).with_seed(9);
    let b = a.clone();
    assert_eq!(a, b);
    let d1 = logits.sample_token(&a, None, None)?;
    assert_eq!(a, b);
    let d2 = logits.sample_token(&b, None, None)?;
    assert_eq!(d1, d2);
    Ok(())
}

use oxideon::distributed::{
    ColumnParallelLinear, Parallel3DConfig, Parallel3DRuntime, RowParallelLinear, ZeROAdamW,
};
use oxideon::quant::{DynamicLlmQuantizer, DynamicQuantPolicy, DynamicTelemetry};

#[test]
fn parallel3d_rejects_overflowing_world_size() -> Result<()> {
    assert!(Parallel3DConfig::new(1 << 32, 1 << 32, 1 << 32).is_err());
    let cfg = Parallel3DConfig::new(2, 2, 1)?;
    assert_eq!(cfg.checked_world_size()?, 4);
    Ok(())
}

#[test]
fn parallel3d_absurd_config_runtime_is_error() -> Result<()> {
    let cfg = Parallel3DConfig {
        dp_size: 1 << 40,
        tp_size: 1 << 40,
        pp_size: 2,
    };
    assert!(cfg.checked_world_size().is_err());
    assert!(Parallel3DRuntime::new(cfg, vec![]).is_err());
    Ok(())
}

#[test]
fn parallel3d_mutated_config_coords_is_error() -> Result<()> {
    let cfg = Parallel3DConfig::new(2, 2, 1)?;
    let devices = vec![Device::Cpu, Device::Cpu, Device::Cpu, Device::Cpu];
    let mut rt = Parallel3DRuntime::new(cfg, devices)?;
    rt.config.dp_size = usize::MAX;
    assert!(rt.get_coords(0).is_err());
    assert!(rt.coords_to_rank(0, 0, 0).is_err());
    Ok(())
}

#[test]
fn mp_linear_shard_count_mismatch_is_error() -> Result<()> {
    let lin = Linear::with_bias(4, 4, true)?;
    let mut col = ColumnParallelLinear::from_linear(&lin, 2)?;
    col.shards.pop();
    let x = Tensor::from_slice(&[1.0f32; 8], &[2, 4])?;
    assert!(col.forward_sharded(&x).is_err());
    assert!(col.forward(&x).is_err());
    let mut row = RowParallelLinear::from_linear(&lin, 2)?;
    row.shards.pop();
    let shards = x.chunk(2, -1)?;
    assert!(row.forward_from_shards(&shards).is_err());
    Ok(())
}

#[test]
fn zero_adamw_rejects_zero_world_size() -> Result<()> {
    let p = Tensor::from_slice(&[1.0f32, 2.0], &[2])?;
    assert!(ZeROAdamW::new(vec![p], 0.01, 0.0, 0).is_err());
    Ok(())
}

#[test]
fn autonomous_cooldown_overflow_saturates() -> Result<()> {
    let policy = DynamicQuantPolicy {
        cooldown_tokens: usize::MAX,
        ..Default::default()
    };
    let mut q = DynamicLlmQuantizer::new(policy);
    let lin = Linear::with_bias(4, 4, true)?;
    q.add_layer("l0", lin);
    q.last_transition_token = 10;
    let telem = DynamicTelemetry {
        token_latency_ms: 1000.0,
        memory_headroom_mb: 1.0,
        kv_cache_usage_pct: 99.0,
        activation_kurtosis: 99.0,
        token_index: 5,
    };
    assert!(q.evaluate_telemetry(&telem).is_none());
    Ok(())
}
