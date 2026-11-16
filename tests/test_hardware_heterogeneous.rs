use oxideon::device::{
    AutoDeviceMixer, Device, HardwareManager, HeterogeneousPipeline, HeterogeneousShardedTensor,
    TpuCluster, TpuGeneration, TpuMxu,
};
use oxideon::error::Result;
use oxideon::tensor::Tensor;

#[test]
fn test_hardware_devices_and_predicates() -> Result<()> {
    let cpu = Device::cpu();
    let cuda0 = Device::cuda(0);
    let gpu1 = Device::gpu(1);
    let tpu00 = Device::tpu(0, 0);
    let tpu_chip1_core1 = Device::tpu(1, 1);

    assert!(cpu.is_cpu());
    assert!(!cpu.is_gpu());
    assert!(!cpu.is_tpu());

    assert!(cuda0.is_cuda());
    assert!(cuda0.is_gpu());
    assert!(!cuda0.is_cpu());

    assert!(gpu1.is_gpu());
    assert!(!gpu1.is_cpu());
    assert!(!gpu1.is_tpu());

    assert!(tpu00.is_tpu());
    assert!(!tpu00.is_cpu());
    assert!(!tpu00.is_gpu());

    // Parsing specs
    assert_eq!(Device::parse_spec("cpu")?, Device::Cpu);
    assert_eq!(Device::parse_spec("cuda:0")?, Device::Cuda(0));
    assert!(Device::parse_spec("gpu:2")?.is_gpu());
    assert_eq!(
        Device::parse_spec("tpu:3")?,
        Device::Tpu { chip: 3, core: 0 }
    );
    assert_eq!(
        Device::parse_spec("tpu:1:1")?,
        Device::Tpu { chip: 1, core: 1 }
    );

    assert_eq!(tpu_chip1_core1.to_string(), "tpu:1:1");

    Ok(())
}

#[test]
fn test_tensor_cross_device_placement_and_migration() -> Result<()> {
    let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2])?;
    assert!(t.is_cpu());

    let t_gpu = t.gpu(0)?;
    assert!(t_gpu.is_gpu());

    let t_cuda = t_gpu.cuda(1)?;
    assert!(t_cuda.is_cuda());
    assert_eq!(t_cuda.device(), Device::Cuda(1));

    let t_tpu = t_cuda.tpu(2, 1)?;
    assert!(t_tpu.is_tpu());
    assert_eq!(t_tpu.device(), Device::Tpu { chip: 2, core: 1 });

    let t_back_cpu = t_tpu.cpu()?;
    assert!(t_back_cpu.is_cpu());
    assert_eq!(t_back_cpu.to_vec::<f32>()?, vec![1.0, 2.0, 3.0, 4.0]);

    Ok(())
}

#[test]
fn test_heterogeneous_pipeline_and_cross_device_autograd() -> Result<()> {
    let mut pipeline = HeterogeneousPipeline::new();

    // Stage 1 on CPU: x * 2.0
    pipeline.add_stage("cpu_preproc", Device::Cpu, |x| x.mul_scalar(2.0));

    // Stage 2 on GPU (CUDA 0): x + 1.0
    pipeline.add_stage("gpu_dense", Device::cuda(0), |x| x.add_scalar(1.0));

    // Stage 3 on TPU: x * 3.0
    pipeline.add_stage("tpu_systolic", Device::tpu(0, 0), |x| x.mul_scalar(3.0));

    assert_eq!(pipeline.len(), 3);
    assert_eq!(
        pipeline.devices(),
        vec![
            Device::Cpu,
            Device::Cuda(0),
            Device::Tpu { chip: 0, core: 0 }
        ]
    );

    // Differentiable forward pass
    let input = Tensor::from_vec(vec![2.0f32, 5.0f32], &[1, 2])?;
    input.set_requires_grad(true)?;
    let output = pipeline.forward(&input)?;

    // Expected:
    // Stage 1 (CPU): [2.0, 5.0] * 2 = [4.0, 10.0]
    // Stage 2 (GPU): [4.0, 10.0] + 1 = [5.0, 11.0]
    // Stage 3 (TPU): [5.0, 11.0] * 3 = [15.0, 33.0]
    assert_eq!(output.device(), Device::Tpu { chip: 0, core: 0 });
    assert_eq!(output.to_vec::<f32>()?, vec![15.0, 33.0]);

    // Backprop: dLoss/dOutput = [1.0, 1.0]
    // dOutput/dInput = 2.0 * 3.0 = 6.0
    // Gradients should flow cross-device back to input on CPU!
    let loss = output.sum()?;
    loss.backward()?;

    let grad = input.grad().expect("input must have gradients populated");
    assert_eq!(grad.device(), Device::Cpu);
    assert_eq!(grad.to_vec::<f32>()?, vec![6.0, 6.0]);

    Ok(())
}

#[test]
fn test_heterogeneous_sharded_tensor() -> Result<()> {
    let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])?;
    let target_devices = vec![Device::Cpu, Device::cuda(0), Device::tpu(0, 0)];

    // Shard along axis 0
    let sharded = HeterogeneousShardedTensor::scatter_from(&t, &target_devices, 0)?;
    assert_eq!(sharded.shards.len(), 3);

    // Verify placement of each shard
    assert_eq!(sharded.shards[0].0, Device::Cpu);
    assert_eq!(sharded.shards[1].0, Device::Cuda(0));
    assert_eq!(sharded.shards[2].0, Device::Tpu { chip: 0, core: 0 });

    // Gather back to TPU
    let gathered_tpu = sharded.gather_to(&Device::tpu(1, 0))?;
    assert_eq!(gathered_tpu.device(), Device::Tpu { chip: 1, core: 0 });
    assert_eq!(gathered_tpu.shape(), &[3, 2]);
    assert_eq!(
        gathered_tpu.to_vec::<f32>()?,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );

    // Gather back to CPU
    let gathered_cpu = sharded.gather_to(&Device::Cpu)?;
    assert_eq!(gathered_cpu.device(), Device::Cpu);
    assert_eq!(
        gathered_cpu.to_vec::<f32>()?,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );

    Ok(())
}

#[test]
fn test_tpu_cluster_and_mxu_simulation() -> Result<()> {
    let cluster = TpuCluster::new(TpuGeneration::V5p, (2, 2, 2))?; // 8 chips, 16 cores
    assert_eq!(cluster.total_cores(), 16);
    assert_eq!(cluster.total_hbm_gb(), 16.0 * 95.0); // 1520 GB HBM3
    assert!(cluster.aggregate_peak_tflops() > 0.0);

    let devices = cluster.devices();
    assert_eq!(devices.len(), 16);
    assert_eq!(devices[0], Device::Tpu { chip: 0, core: 0 });
    assert_eq!(devices[1], Device::Tpu { chip: 0, core: 1 });

    // Systolic MXU matrix multiplication simulation
    let mxu = TpuMxu::new(TpuGeneration::V5p);
    let a = Tensor::randn(&[32, 32])?;
    let b = Tensor::randn(&[32, 32])?;
    let c = mxu.systolic_matmul(&a, &b)?;
    assert_eq!(c.shape(), &[32, 32]);

    Ok(())
}

#[test]
fn test_hardware_manager_and_inventory() -> Result<()> {
    let hw = HardwareManager::global();
    let summary = hw.summary();
    assert!(summary.contains("Hardware Inventory:"));
    assert!(summary.contains("CPU cores"));
    assert!(summary.contains("GPU accelerator"));
    assert!(summary.contains("TPU core"));

    // Micro benchmark on CPU
    let gflops = hw.benchmark_gemm(&Device::Cpu, 32)?;
    assert!(gflops > 0.0);

    // Track memory allocations
    let dev = Device::cuda(0);
    hw.track_allocation(&dev, 1024 * 1024 * 128); // 128 MB
    assert_eq!(hw.device_allocated_bytes(&dev), 1024 * 1024 * 128);
    hw.track_deallocation(&dev, 1024 * 1024 * 64);
    assert_eq!(hw.device_allocated_bytes(&dev), 1024 * 1024 * 64);
    hw.track_deallocation(&dev, 1024 * 1024 * 64);
    assert_eq!(hw.device_allocated_bytes(&dev), 0);

    Ok(())
}

#[test]
fn test_auto_device_mixer() -> Result<()> {
    // Dense GEMM / MLP -> TPU
    let dev_mlp = AutoDeviceMixer::recommend_placement("feed_forward_mlp", 10_000_000, true);
    assert!(dev_mlp.is_tpu());

    // Attention layer -> GPU
    let dev_attn = AutoDeviceMixer::recommend_placement("attention_layer", 1_000_000, false);
    assert!(dev_attn.is_gpu());

    // Embedding lookup -> CPU
    let dev_embed = AutoDeviceMixer::recommend_placement("embedding_tokens", 10_000_000, false);
    assert!(dev_embed.is_cpu());

    Ok(())
}
