//! Showcase of Oxideon v0.2's advanced tensor features:
//! 1. Zero-copy strided views (stepped slicing, negative strides, diagonal, broadcast)
//! 2. 8 DTypes (`f16`, `bf16`, `f32`, `f64`, `i32`, `i64`, `u8`, `bool`)
//! 3. Higher-order autograd (1st, 2nd, and 3rd derivatives + Hessian-vector products)
//! 4. Differentiable `einsum`, `Conv2d`, `MultiHeadAttention`, and `.safetensors`
//! 5. Blocked AVX2 + Rayon GEMM benchmark
//!
//! Run with: `cargo run --release --example showcase`

use std::time::Instant;

use oxideon::prelude::*;

fn main() -> Result<()> {
    println!("=== 1. Zero-Copy Strided Views & Indexing ===");
    let base = Tensor::arange(0.0, 24.0, 1.0, DType::F32)?.reshape(&[2, 3, 4])?;
    let view = base.i((.., 1..3, SliceElem::step(Some(3), None, -2)))?;
    println!(
        "base shape: {:?}, view shape: {:?}, view strides: {:?}, shares_storage: {}",
        base.shape(),
        view.shape(),
        view.strides(),
        view.shares_storage(&base)
    );

    println!("\n=== 2. Multi-DType Support (f16 / bf16 / f32 / f64 / i64 / bool) ===");
    let t_f16 = view.to_dtype(DType::F16)?;
    let t_bf16 = view.to_dtype(DType::BF16)?;
    println!("F16 tensor:  {:?}", t_f16);
    println!("BF16 tensor: {:?}", t_bf16);

    println!("\n=== 3. Higher-Order Autograd (1st, 2nd, 3rd Derivatives) ===");
    let x = Tensor::from_slice(&[2.0f32], &[1])?.requires_grad_(true);
    let y = x.powf(4.0)?; // y = x^4 at x = 2 => y = 16, y' = 32, y'' = 48, y''' = 48
    let dy = grad(&[y], std::slice::from_ref(&x), None, true)?
        .pop()
        .unwrap()
        .unwrap();
    let d2y = grad(
        std::slice::from_ref(&dy),
        std::slice::from_ref(&x),
        None,
        true,
    )?
    .pop()
    .unwrap()
    .unwrap();
    let d3y = grad(
        std::slice::from_ref(&d2y),
        std::slice::from_ref(&x),
        None,
        false,
    )?
    .pop()
    .unwrap()
    .unwrap();
    println!(
        "f(x) = x^4 at x=2 -> f'={:.1}, f''={:.1}, f'''={:.1}",
        dy.item::<f32>()?,
        d2y.item::<f32>()?,
        d3y.item::<f32>()?
    );

    println!("\n=== 4. Differentiable Einsum & Causal Multi-Head Attention ===");
    manual_seed(7);
    let q = Tensor::randn(&[2, 4, 8, 16])?.requires_grad_(true);
    let k = Tensor::randn(&[2, 4, 8, 16])?.requires_grad_(true);
    let attn_scores = Tensor::einsum("bhqd,bhkd->bhqk", &[&q, &k])?;
    attn_scores.sum()?.backward()?;
    println!(
        "Einsum 'bhqd,bhkd->bhqk' output shape: {:?}, q.grad shape: {:?}",
        attn_scores.shape(),
        q.grad().unwrap().shape()
    );

    let mha = MultiHeadAttention::new(32, 4)?;
    let seq = Tensor::randn(&[2, 6, 32])?;
    let mha_out = mha.forward_qkv(&seq, &seq, &seq, true)?;
    let bytes = serialize_safetensors(&mha.state_dict())?;
    println!(
        "Causal MHA output shape: {:?}, serialized safetensors size: {} bytes",
        mha_out.shape(),
        bytes.len()
    );

    println!("\n=== 5. Blocked AVX2+FMA & Rayon Strided GEMM Benchmark ===");
    let n = 512;
    let a = Tensor::randn(&[n, n])?;
    let b = Tensor::randn(&[n, n])?.t()?; // transposed view!
    let iters = 20;
    let t0 = Instant::now();
    for _ in 0..iters {
        let _c = no_grad(|| a.matmul(&b))?;
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let gflops = (2.0 * (n as f64).powi(3) * (iters as f64)) / elapsed / 1e9;
    println!(
        "{n}x{n} strided f32 GEMM ({iters} iters): {:.2} ms/iter ({:.1} GFLOP/s)",
        (elapsed * 1000.0) / (iters as f64),
        gflops
    );

    Ok(())
}
