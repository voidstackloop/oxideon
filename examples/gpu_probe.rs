use oxideon::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    let dev = Device::wgpu()?;
    manual_seed(1);
    for n in [512usize, 1024, 2048] {
        let a = Tensor::randn(&[n, n])?;
        let b = Tensor::randn(&[n, n])?;
        let (ga, gb) = (a.to_device(&dev)?, b.to_device(&dev)?);
        let _ = ga.matmul(&gb)?.sum()?.to_vec::<f32>()?; // warm up + sync
        let t = Instant::now();
        let reps = 10;
        for _ in 0..reps {
            let _ = ga.matmul(&gb)?;
        }
        let _ = ga.matmul(&gb)?.sum()?.to_vec::<f32>()?;
        let gpu = t.elapsed().as_secs_f64() / (reps + 1) as f64;
        let t = Instant::now();
        let _ = a.matmul(&b)?;
        let cpu = t.elapsed().as_secs_f64();
        let fl = 2.0 * (n as f64).powi(3);
        println!("matmul {n}: gpu {:.1} GFLOP/s, cpu {:.1} GFLOP/s", fl / gpu / 1e9, fl / cpu / 1e9);
    }
    // autograd on the device: a small MLP step
    let w1 = Tensor::randn(&[256, 512])?.mul_scalar(0.05)?.to_device(&dev)?.requires_grad_(true);
    let w2 = Tensor::randn(&[512, 10])?.mul_scalar(0.05)?.to_device(&dev)?.requires_grad_(true);
    let x = Tensor::randn(&[128, 256])?.to_device(&dev)?;
    let y = x.matmul(&w1)?.relu()?.matmul(&w2)?;
    let loss = y.square()?.mean()?;
    loss.backward()?;
    let g = w1.grad().ok_or_else(|| Error::Msg("no grad".into()))?;
    println!("mlp loss {:.4}, w1.grad on {:?}, |g| {:.4}", loss.to_vec::<f32>()?[0], g.device(), g.abs()?.sum()?.to_device(&Device::Cpu)?.to_vec::<f32>()?[0]);
    Ok(())
}
