use oxideon::opgraph::OpGraph;
use oxideon::prelude::*;
use std::time::Instant;

fn time(reps: usize, mut f: impl FnMut()) -> f64 {
    f();
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    t.elapsed().as_secs_f64() / reps as f64 * 1e3
}

fn main() -> Result<()> {
    manual_seed(1);
    // 1. long elementwise chain on a large tensor (memory bound)
    for n in [1usize << 18, 1 << 22] {
        let x = Tensor::randn(&[n])?;
        let mut g = OpGraph::new();
        let xi = g.input(&[n], DType::F32)?;
        let a = g.mul_scalar(xi, 0.5)?;
        let b = g.add_scalar(a, 1.0)?;
        let c = g.mul(b, xi)?;
        let d = g.relu(c)?;
        let e = g.mul_scalar(d, 0.25)?;
        let f = g.sub(e, xi)?;
        let h = g.abs(f)?;
        let o = g.add_scalar(h, 0.1)?;
        g.set_outputs(&[o]);
        let c = g.compile()?;
        let eager = time(20, || {
            let _ = x.mul_scalar(0.5).unwrap().add_scalar(1.0).unwrap().mul(&x).unwrap().relu().unwrap().mul_scalar(0.25).unwrap().sub(&x).unwrap().abs().unwrap().add_scalar(0.1).unwrap();
        });
        let comp = time(20, || {
            let _ = c.run(std::slice::from_ref(&x)).unwrap();
        });
        println!("chain of 8 ops, n={n}: eager {eager:.2} ms, compiled {comp:.2} ms ({:.2}x) {:?}", eager / comp, c.stats());
    }
    // 2. transformer-style FFN + layer norm
    let (b, d, h) = (256usize, 512usize, 2048usize);
    let (w1, b1, w2) = (Tensor::randn(&[d, h])?.mul_scalar(0.03)?, Tensor::randn(&[h])?, Tensor::randn(&[h, d])?.mul_scalar(0.03)?);
    let (gam, bet) = (Tensor::ones(&[d], DType::F32, &Device::Cpu)?, Tensor::zeros(&[d], DType::F32, &Device::Cpu)?);
    let x = Tensor::randn(&[b, d])?;
    let mut g = OpGraph::new();
    let xi = g.input(&[b, d], DType::F32)?;
    let (cw1, cb1, cw2, cg, cb) = (g.constant(&w1)?, g.constant(&b1)?, g.constant(&w2)?, g.constant(&gam)?, g.constant(&bet)?);
    let y = g.matmul(xi, cw1)?;
    let y = g.add(y, cb1)?;
    let y = g.gelu(y)?;
    let y = g.matmul(y, cw2)?;
    let y = g.add(y, xi)?;
    let y = g.layer_norm(y, cg, cb, 1e-5)?;
    g.set_outputs(&[y]);
    let c = g.compile()?;
    let eager = time(20, || {
        let y = x.matmul(&w1).unwrap().add(&b1).unwrap().gelu().unwrap().matmul(&w2).unwrap().add(&x).unwrap();
        let mu = y.mean_axes(&[-1], true).unwrap();
        let cen = y.sub(&mu).unwrap();
        let var = cen.square().unwrap().mean_axes(&[-1], true).unwrap();
        let _ = cen.mul(&var.add_scalar(1e-5).unwrap().rsqrt().unwrap()).unwrap().mul(&gam).unwrap().add(&bet).unwrap();
    });
    let comp = time(20, || {
        let _ = c.run(std::slice::from_ref(&x)).unwrap();
    });
    println!("FFN + layer norm [{b}x{d}]: eager {eager:.2} ms, compiled {comp:.2} ms ({:.2}x) {:?}", eager / comp, c.stats());
    Ok(())
}
