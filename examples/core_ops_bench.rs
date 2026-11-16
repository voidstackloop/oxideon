//! Wall-time of common CPU tensor operations (best of several runs, microseconds). `tools/core_ops_bench_torch.py` prints
//! the same table for PyTorch so the two can be compared on one machine.
//!
//! `cargo run --release --example core_ops_bench [name filter]`

use oxideon::nn::functional_ext::layer_norm;
use oxideon::prelude::*;
use std::time::Instant;

fn rnd(shape: &[usize], seed: f32) -> Tensor {
    let n: usize = shape.iter().product();
    let v: Vec<f32> = (0..n).map(|i| ((i as f32 + seed) * 0.37).sin()).collect();
    Tensor::from_slice(&v, shape).unwrap()
}

fn bench(name: &str, mut f: impl FnMut()) {
    // an optional argument keeps only the benchmarks whose name contains it
    if std::env::args().nth(1).is_some_and(|k| !k.split(',').any(|p| name.contains(p))) {
        return;
    }
    f(); // warm up
    let mut best = f64::INFINITY;
    let start = Instant::now();
    let mut runs = 0;
    while runs < 200 && (runs < 5 || start.elapsed().as_secs_f64() < 0.5) {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
        runs += 1;
    }
    println!("BENCH {name:<28} {:>12.1}", best * 1e6);
    if std::env::var("PROBE").is_ok() {
        thread_local! { static P: (Tensor, Tensor, Vec<f32>) = (rnd(&[1000, 1000], 1.0), rnd(&[1000, 1000], 2.0), vec![1.0; 1_000_000]); }
        P.with(|(x, y, v)| {
            let best = |f: &mut dyn FnMut()| { let mut b = f64::MAX; for _ in 0..20 { let t = Instant::now(); f(); b = b.min(t.elapsed().as_secs_f64()); } b * 1e6 };
            let add = best(&mut || drop(x.add(y).unwrap()));
            let sum = best(&mut || drop(x.sum().unwrap()));
            let serial = best(&mut || { std::hint::black_box(v.iter().sum::<f32>()); });
            let (h, m) = oxideon::pool::hit_stats();
            println!("   probe: add {add:.0}  sum {sum:.0}  serial-sum {serial:.0}  pool {} MiB hits {h} misses {m}", oxideon::pool::held_bytes() >> 20);
        });
    }
}

fn main() -> Result<()> {
    let a = rnd(&[1024, 1024], 0.0);
    let b = rnd(&[1024, 1024], 1.0);
    let row = rnd(&[1024], 2.0);
    let small = rnd(&[64], 3.0);
    let img = rnd(&[8, 16, 64, 64], 4.0);
    let w = rnd(&[32, 16, 3, 3], 5.0);
    let seq = rnd(&[32, 128, 256], 6.0);
    let (g, bb) = (rnd(&[256], 7.0), rnd(&[256], 8.0));
    let m = rnd(&[512, 512], 9.0);
    let idx = Tensor::from_slice(&(0..4096).map(|i| (i * 7 % 1024) as i64).collect::<Vec<_>>(), &[4096])?;

    bench("add 1M", || drop(a.add(&b).unwrap()));
    bench("mul 1M", || drop(a.mul(&b).unwrap()));
    bench("add broadcast row 1M", || drop(a.add(&row).unwrap()));
    bench("mul_scalar 1M", || drop(a.mul_scalar(1.5).unwrap()));
    bench("exp 1M", || drop(a.exp().unwrap()));
    bench("tanh 1M", || drop(a.tanh().unwrap()));
    bench("gelu 1M", || drop(a.gelu().unwrap()));
    bench("relu 1M", || drop(a.relu().unwrap()));
    bench("sum 1M", || drop(a.sum().unwrap()));
    bench("sum axis0 1M", || drop(a.sum_axis(0, false).unwrap()));
    bench("sum axis1 1M", || drop(a.sum_axis(1, false).unwrap()));
    bench("max axis1 1M", || drop(a.max_axis(1, false).unwrap()));
    bench("softmax 1M", || drop(a.softmax(-1).unwrap()));
    bench("transpose+contiguous 1M", || drop(a.transpose(0, 1).unwrap().contiguous().unwrap()));
    bench("add transposed 1M", || drop(a.transpose(0, 1).unwrap().add(&b).unwrap()));
    bench("matmul 512", || drop(m.matmul(&m).unwrap()));
    bench("matmul 1024", || drop(a.matmul(&b).unwrap()));
    bench("conv2d 8x16x64x64 k3", || drop(img.conv2d(&w, None, (1, 1), (1, 1), (1, 1)).unwrap()));
    bench("max_pool2d", || drop(img.max_pool2d((2, 2), (2, 2), (0, 0)).unwrap()));
    bench("layer_norm 32x128x256", || drop(layer_norm(&seq, &[256], Some(&g), Some(&bb), 1e-5).unwrap()));
    bench("index_select 4096 rows", || drop(a.index_select(0, &idx).unwrap()));
    bench("cat 2x1M", || drop(Tensor::cat(&[a.clone(), b.clone()], 0).unwrap()));
    bench("small add 64 (overhead)", || drop(small.add(&small).unwrap()));
    bench("small matmul 8x8", || {
        let s = small.reshape(&[8, 8]).unwrap();
        drop(s.matmul(&s).unwrap())
    });

    // one training step of a small MLP: forward, backward, SGD update
    let x = rnd(&[256, 784], 10.0);
    let w1 = rnd(&[784, 512], 11.0).mul_scalar(0.05)?.requires_grad_(true);
    let w2 = rnd(&[512, 10], 12.0).mul_scalar(0.05)?.requires_grad_(true);
    bench("mlp train step b256", || {
        w1.zero_grad();
        w2.zero_grad();
        let loss = x.matmul(&w1).unwrap().relu().unwrap().matmul(&w2).unwrap().log_softmax(-1).unwrap().mean().unwrap();
        loss.backward().unwrap();
    });
    let ag = rnd(&[1024, 1024], 13.0).requires_grad_(true);
    if let Ok(v) = std::env::var("SLEEP") { std::thread::sleep(std::time::Duration::from_secs(v.parse().unwrap())); }
    bench("add 1M again", || drop(a.add(&b).unwrap()));
    println!("pool hits/misses before chain: {:?}", oxideon::pool::hit_stats());
    println!("pool held before chain: {} MiB, live {} MiB", oxideon::pool::held_bytes() >> 20, oxideon::memory::allocated_bytes() >> 20);
    if std::env::var("CLEAR").is_ok() { oxideon::pool::clear(); }
    if std::env::var("DROPW").is_ok() { w1.zero_grad(); w2.zero_grad(); }
    bench("autograd elementwise chain", || {
        ag.zero_grad();
        ag.mul(&b).unwrap().tanh().unwrap().add(&b).unwrap().square().unwrap().sum().unwrap().backward().unwrap();
    });
    if std::env::var("PROF").is_ok() {
        let t = Instant::now();
        let (_, p) = oxideon::profiler::profile(|| {
            ag.zero_grad();
            ag.mul(&b).unwrap().tanh().unwrap().add(&b).unwrap().square().unwrap().sum().unwrap().backward().unwrap();
        });
        println!("chain wall {:.0} us, in ops {:.0} us", t.elapsed().as_secs_f64() * 1e6, p.total_op_time_us());
        let mut ev = p.events.clone();
        ev.sort_by(|a, b| a.start_us.partial_cmp(&b.start_us).unwrap());
        let mut prev_end = 0.0;
        for e in &ev {
            println!("  gap {:>8.0}  {:<12} {:>8.0} us {:?}", e.start_us - prev_end, e.name, e.dur_us, e.shape);
            prev_end = e.start_us + e.dur_us;
        }
    }
    Ok(())
}
