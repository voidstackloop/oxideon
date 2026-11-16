//! Square matmul timings on CPU vs the wgpu device.
//!
//! Release build:
//! ```bash
//! cargo run --release --features wgpu --example gpu_matmul_bench
//! ```
//!
//! Each timing ends with a scalar sum read back to the host, which is how the
//! benchmark knows the kernels have finished. Small sizes also run a CPU
//! baseline and verify GPU/CPU numerics agree. Sizes the device cannot hold
//! (e.g. over a software renderer's buffer limit) are reported and skipped.

#[cfg(not(feature = "wgpu"))]
fn main() {
    eprintln!("build with `--features wgpu`");
}

#[cfg(feature = "wgpu")]
fn bench_size(
    dev: &oxideon::prelude::Device,
    n: usize,
) -> oxideon::prelude::Result<(f64, String, String)> {
    use std::time::Instant;

    use oxideon::prelude::*;

    let a_host = Tensor::randn(&[n, n])?;
    let b_host = Tensor::randn(&[n, n])?;
    let a = a_host.to_device(dev)?;
    let b = b_host.to_device(dev)?;
    a.matmul(&b)?.sum()?.item::<f32>()?; // warm-up: shader compilation, allocations
    let reps = if n >= 4096 { 3 } else { 10 };
    let t0 = Instant::now();
    for _ in 0..reps {
        a.matmul(&b)?.sum()?.item::<f32>()?;
    }
    let gpu_secs = t0.elapsed().as_secs_f64() / reps as f64;

    // CPU baseline + correctness check for sizes that finish quickly on host.
    let (cpu_ms, max_diff) = if n <= 1024 {
        let t0 = Instant::now();
        let c_cpu = a_host.matmul(&b_host)?;
        let cpu_secs = t0.elapsed().as_secs_f64();
        let c_gpu = a.matmul(&b)?.to_device(&Device::cpu())?;
        let diff_vec = c_cpu.sub(&c_gpu)?.abs()?.to_vec::<f32>()?;
        let diff = diff_vec.iter().copied().fold(0.0f32, f32::max);
        (format!("{:.2}", cpu_secs * 1e3), format!("{diff:.2e}"))
    } else {
        ("-".to_string(), "-".to_string())
    };
    Ok((gpu_secs, cpu_ms, max_diff))
}

#[cfg(feature = "wgpu")]
fn main() -> oxideon::prelude::Result<()> {
    use oxideon::prelude::*;

    let dev = Device::wgpu()?;
    println!("device: {dev}");
    println!(
        "{:>6} {:>10} {:>10} {:>12} {:>10}",
        "n", "gpu ms", "cpu ms", "GFLOP/s", "max diff"
    );
    for n in [256usize, 512, 1024, 2048, 4096, 6144] {
        // One oversized case (e.g. over a software renderer's buffer limit) must not
        // abort the whole table: report it and continue with the remaining sizes.
        match bench_size(&dev, n) {
            Ok((gpu_secs, cpu_ms, max_diff)) => println!(
                "{n:>6} {:>10.2} {:>10} {:>12.1} {:>10}",
                gpu_secs * 1e3,
                cpu_ms,
                2.0 * (n as f64).powi(3) / gpu_secs / 1e9,
                max_diff,
            ),
            Err(e) => println!("{n:>6} {:>10} {:>10} {:>12}  skipped ({e})", "-", "-", "-"),
        }
    }
    Ok(())
}
