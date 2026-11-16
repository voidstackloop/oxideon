//! Tests for zero-copy strided views, 8 dtypes (`f16`, `bf16`, `f32`, `f64`, `i32`, `i64`, `u8`, `bool`),
//! neural network modules (`Conv2d`, `BatchNorm2d`, `LayerNorm`, `MultiHeadAttention`, `Embedding`),
//! and `.safetensors` serialization.

use oxideon::prelude::*;

#[test]
fn zero_copy_views_share_storage_and_handle_negative_strides() -> Result<()> {
    let x = Tensor::arange(0.0, 24.0, 1.0, DType::F32)?.reshape(&[2, 3, 4])?;

    // Permute, transpose, narrow, step-slice, flip, select, expand, diagonal must all share storage!
    let p = x.permute(&[2, 0, 1])?;
    assert!(p.shares_storage(&x));
    assert!(!p.is_contiguous());
    assert_eq!(p.shape(), &[4, 2, 3]);

    let sub = x.i((.., 1..3, SliceElem::step(Some(3), None, -2)))?;
    assert!(sub.shares_storage(&x));
    assert_eq!(sub.shape(), &[2, 2, 2]);
    // Row 1 of batch 0 is [4, 5, 6, 7], indices 3, 1 -> [7, 5]
    // Row 2 of batch 0 is [8, 9, 10, 11], indices 3, 1 -> [11, 9]
    assert_eq!(
        sub.to_vec::<f32>()?,
        vec![7.0, 5.0, 11.0, 9.0, 19.0, 17.0, 23.0, 21.0]
    );

    let flipped = x.flip(&[2])?;
    assert!(flipped.shares_storage(&x));
    assert_eq!(flipped.strides()[2], -1);

    let expanded = Tensor::from_slice(&[10.0f32, 20.0, 30.0], &[1, 3])?.expand(&[4, 3])?;
    assert_eq!(expanded.strides(), &[0, 1]);
    assert_eq!(expanded.to_vec::<f32>()?.len(), 12);

    let m = Tensor::arange(1.0, 10.0, 1.0, DType::F32)?.reshape(&[3, 3])?;
    let diag = m.diagonal(0, 0, 1)?;
    assert!(diag.shares_storage(&m));
    assert_eq!(diag.to_vec::<f32>()?, vec![1.0, 5.0, 9.0]);
    assert_eq!(m.trace()?.item::<f32>()?, 15.0);
    Ok(())
}

#[test]
fn multi_dtype_conversions_and_f16_bf16_precision() -> Result<()> {
    let vals = [-3.5f32, 0.0, 1.25, 42.0];
    let t_f32 = Tensor::from_slice(&vals, &[2, 2])?;

    for dt in [DType::F16, DType::BF16, DType::F64, DType::I32, DType::I64] {
        let casted = t_f32.to_dtype(dt)?;
        assert_eq!(casted.dtype(), dt);
        let back = casted.to_dtype(DType::F32)?;
        if dt.is_floating_point() {
            assert!(t_f32.allclose(&back, 1e-2, 1e-2)?);
        }
    }

    let b = t_f32.gt(&Tensor::zeros_f32(&[2, 2])?)?;
    assert_eq!(b.dtype(), DType::Bool);
    assert_eq!(b.to_vec::<bool>()?, vec![false, false, true, true]);
    Ok(())
}

#[test]
fn blocked_simd_gemm_matches_reference_on_large_and_transposed_matrices() -> Result<()> {
    manual_seed(99);
    // Large enough (160 x 200 @ 200 x 144) to exercise MC/KC/NC blocking, AVX2 8x8 panels,
    // and odd-tail edge microkernels, including transposed strided inputs!
    let a = Tensor::randn(&[160, 200])?;
    let b_raw = Tensor::randn(&[144, 200])?;
    let b = b_raw.t()?; // non-contiguous [200, 144] with strides [1, 200]
    assert!(!b.is_contiguous());

    let c_f32 = a.matmul(&b)?;
    let c_f64 = a
        .to_dtype(DType::F64)?
        .matmul(&b.to_dtype(DType::F64)?)?
        .to_dtype(DType::F32)?;
    assert!(c_f32.allclose(&c_f64, 1e-4, 1e-4)?);
    Ok(())
}

#[test]
fn nn_modules_and_safetensors_roundtrip() -> Result<()> {
    manual_seed(123);

    // Conv2d -> BatchNorm2d -> GELU -> MaxPool2d -> Flatten -> Linear
    let cnn = Sequential::new()
        .add(Conv2d::new(2, 4, (3, 3), (1, 1), (1, 1))?)
        .add(BatchNorm2d::new(4)?)
        .add(GELU)
        .add(MaxPool2d::new((2, 2), (2, 2)))
        .add(Flatten)
        .add(Linear::new(4 * 4 * 4, 3)?);

    let x = Tensor::randn(&[2, 2, 8, 8])?;
    let logits = cnn.forward(&x)?;
    assert_eq!(logits.shape(), &[2, 3]);
    logits.sum()?.backward()?;

    // MultiHeadAttention + LayerNorm + RMSNorm + Embedding
    let emb = Embedding::new(20, 16)?;
    let mha = MultiHeadAttention::new(16, 4)?;
    let ln = LayerNorm::new(16, 1e-5)?;
    let rms = RMSNorm::new(16, 1e-5)?;

    let tokens = Tensor::from_slice(&[1i64, 5, 3, 12, 0, 7], &[2, 3])?;
    let h = emb.forward(&tokens)?;
    let attn_out = mha.forward_qkv(&h, &h, &h, true)?;
    let normed = rms.forward(&ln.forward(&attn_out)?)?;
    assert_eq!(normed.shape(), &[2, 3, 16]);
    normed.sum()?.backward()?;

    // Safetensors in-memory roundtrip of model state_dict
    let state = cnn.state_dict();
    let bytes = serialize_safetensors(&state)?;
    let loaded = deserialize_safetensors(&bytes)?;
    assert_eq!(state.len(), loaded.len());
    for (k, orig_t) in &state {
        assert!(orig_t.allclose(&loaded[k], 1e-6, 1e-6)?);
    }
    Ok(())
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_backend_unary_binary_matmul_and_autograd() -> Result<()> {
    let Ok(dev) = Device::wgpu() else {
        println!("No WGPU adapter available in this environment; skipping GPU execution.");
        return Ok(());
    };
    println!("Running WGPU test on adapter: {dev}");
    manual_seed(77);
    let a_cpu = Tensor::randn(&[32, 48])?.requires_grad_(true);
    let b_cpu = Tensor::randn(&[48, 24])?.requires_grad_(true);

    let a_gpu = a_cpu.detach().to_device(&dev)?.requires_grad_(true);
    let b_gpu = b_cpu.detach().to_device(&dev)?.requires_grad_(true);

    let y_cpu = a_cpu.matmul(&b_cpu)?.relu()?.add_scalar(1.5)?.sum()?;
    y_cpu.backward()?;

    let y_gpu = a_gpu.matmul(&b_gpu)?.relu()?.add_scalar(1.5)?.sum()?;
    y_gpu.backward()?;

    assert!(y_cpu.allclose(&y_gpu.to_device(&Device::Cpu)?, 1e-3, 1e-3)?);
    assert!(a_cpu.grad().unwrap().allclose(
        &a_gpu.grad().unwrap().to_device(&Device::Cpu)?,
        1e-3,
        1e-3
    )?);
    assert!(b_cpu.grad().unwrap().allclose(
        &b_gpu.grad().unwrap().to_device(&Device::Cpu)?,
        1e-3,
        1e-3
    )?);
    Ok(())
}

/// A 1-D kernel covers 256 elements per workgroup and one dispatch allows 65,535 of them (16.7 million elements); larger
/// tensors used to reach wgpu's validation layer, which raised a Rust panic. They now run on the CPU path.
#[cfg(feature = "wgpu")]
#[test]
fn wgpu_kernels_larger_than_one_dispatch_do_not_panic() -> Result<()> {
    let Ok(dev) = Device::wgpu() else {
        return Ok(());
    };
    let n = 20_000_000;
    let x = Tensor::ones(&[n], DType::F32, &Device::Cpu)?.to_device(&dev)?;
    let y = x.mul_scalar(3.0)?.add(&x)?.to_device(&Device::Cpu)?;
    assert_eq!(y.numel(), n);
    assert_eq!(y.narrow(0, n - 1, 1)?.item::<f32>()?, 4.0);
    assert_eq!(y.narrow(0, 0, 1)?.item::<f32>()?, 4.0);
    Ok(())
}
