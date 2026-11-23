//! GPU timings of the NVRTC-compiled kernels (best of several runs, microseconds), with the CPU time of the same operation
//! next to them. `cargo run --release --example cuda_bench` (set OXIDEON_NVRTC_SITE if NVRTC is not on the default path).

use oxideon::prelude::*;
use std::time::Instant;

fn rnd(shape: &[usize], seed: f32) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_slice(&(0..n).map(|i| ((i as f32 + seed) * 0.37).sin()).collect::<Vec<f32>>(), shape).unwrap()
}

fn best(mut f: impl FnMut(), sync: &dyn Fn()) -> f64 {
    f();
    sync();
    let mut b = f64::MAX;
    let start = Instant::now();
    let mut runs = 0;
    while runs < 100 && (runs < 5 || start.elapsed().as_secs_f64() < 0.4) {
        let t = Instant::now();
        f();
        sync();
        b = b.min(t.elapsed().as_secs_f64());
        runs += 1;
    }
    b * 1e6
}

fn main() -> Result<()> {
    let dev = Device::Cuda(0);
    let sync = || oxideon::backend::cuda_driver::cuda_synchronize(0).unwrap();
    let nosync = || {};
    println!("{:<30}{:>12}{:>12}{:>9}", "op", "cpu us", "cuda us", "speedup");
    macro_rules! row {
        ($name:expr, $cpu:expr, $gpu:expr) => {{
            let c = best($cpu, &nosync);
            let g = best($gpu, &sync);
            println!("{:<30}{:>12.1}{:>12.1}{:>8.1}x", $name, c, g, c / g);
        }};
    }
    let (a, b) = (rnd(&[4096, 4096], 0.0), rnd(&[4096, 4096], 1.0));
    let (ga, gb) = (a.to_device(&dev)?, b.to_device(&dev)?);
    row!("add 16M", || drop(a.add(&b).unwrap()), || drop(ga.add(&gb).unwrap()));
    row!("exp 16M", || drop(a.exp().unwrap()), || drop(ga.exp().unwrap()));
    row!("gelu 16M", || drop(a.gelu().unwrap()), || drop(ga.gelu().unwrap()));
    row!("sum 16M", || drop(a.sum().unwrap()), || drop(ga.sum().unwrap()));
    row!("sum axis0 4096x4096", || drop(a.sum_axis(0, false).unwrap()), || drop(ga.sum_axis(0, false).unwrap()));
    row!("softmax 4096x4096", || drop(a.softmax(-1).unwrap()), || drop(ga.softmax(-1).unwrap()));
    row!("transpose copy 4096^2", || drop(a.t().unwrap().contiguous().unwrap()), || drop(ga.t().unwrap().contiguous().unwrap()));
    let (m1, m2) = (rnd(&[1024, 1024], 2.0), rnd(&[1024, 1024], 3.0));
    let (gm1, gm2) = (m1.to_device(&dev)?, m2.to_device(&dev)?);
    row!("matmul 1024", || drop(m1.matmul(&m2).unwrap()), || drop(gm1.matmul(&gm2).unwrap()));
    row!("matmul 4096", || drop(a.matmul(&b).unwrap()), || drop(ga.matmul(&gb).unwrap()));
    // tensor cores: f16 / bf16 always, f32 only as TF32 when enabled
    let (ha, hb) = (ga.to_dtype(DType::F16)?, gb.to_dtype(DType::F16)?);
    let (ba, bb2) = (ga.to_dtype(DType::BF16)?, gb.to_dtype(DType::BF16)?);
    let c16 = best(|| drop(ha.matmul(&hb).unwrap()), &sync);
    let cbf = best(|| drop(ba.matmul(&bb2).unwrap()), &sync);
    oxideon::backend::cuda_ops::set_allow_tf32(true);
    let ctf = best(|| drop(ga.matmul(&gb).unwrap()), &sync);
    oxideon::backend::cuda_ops::set_allow_tf32(false);
    let tflops = |us: f64| 2.0 * 4096f64.powi(3) / (us * 1e-6) / 1e12;
    println!("matmul 4096 f16 tensor cores  {c16:>10.1} us  {:.1} TFLOP/s", tflops(c16));
    println!("matmul 4096 bf16 tensor cores {cbf:>10.1} us  {:.1} TFLOP/s", tflops(cbf));
    println!("matmul 4096 f32 as TF32       {ctf:>10.1} us  {:.1} TFLOP/s", tflops(ctf));
    let (q, k, v) = (rnd(&[8, 16, 1024, 64], 5.0).to_device(&dev)?, rnd(&[8, 16, 1024, 64], 6.0).to_device(&dev)?, rnd(&[8, 16, 1024, 64], 7.0).to_device(&dev)?);
    let fused = best(|| drop(oxideon::nn::functional_ext::scaled_dot_product_attention(&q, &k, &v, None, true, None).unwrap()), &sync);
    let qr = q.requires_grad_(true);
    let composed = best(|| drop(oxideon::nn::functional_ext::scaled_dot_product_attention(&qr, &k, &v, None, true, None).unwrap()), &sync);
    println!("attention 8x16x1024x64 causal  fused {fused:>9.1} us   composed (grad path) {composed:>9.1} us");
    let img = rnd(&[16, 32, 64, 64], 4.0);
    let w = rnd(&[64, 32, 3, 3], 5.0);
    let (gimg, gw) = (img.to_device(&dev)?, w.to_device(&dev)?);
    row!("conv2d 16x32x64x64 k3", || drop(img.conv2d(&w, None, (1, 1), (1, 1), (1, 1)).unwrap()), || drop(gimg.conv2d(&gw, None, (1, 1), (1, 1), (1, 1)).unwrap()));
    row!("max_pool2d 16x32x64x64", || drop(img.max_pool2d((2, 2), (2, 2), (0, 0)).unwrap()), || drop(gimg.max_pool2d((2, 2), (2, 2), (0, 0)).unwrap()));
    let (x, g, bb) = (rnd(&[8192, 1024], 6.0), rnd(&[1024], 7.0), rnd(&[1024], 8.0));
    let ln = oxideon::nn::LayerNorm::new(1024, 1e-5)?;
    let _ = (&g, &bb);
    let gx = x.to_device(&dev)?;
    let lng = oxideon::nn::LayerNorm::new(1024, 1e-5)?;
    for p in lng.parameters() {
        p.retarget(&p.detach().to_device(&dev)?)?;
    }
    row!("layer_norm 8192x1024", || drop(ln.forward(&x).unwrap()), || drop(lng.forward(&gx).unwrap()));
    println!("kernels launched: {}", oxideon::backend::cuda_rt::launch_count());
    Ok(())
}
