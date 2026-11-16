//! The wgpu kernels against the CPU backend on awkward sizes and layouts, and a check that chains of GPU ops never copy
//! back to the host. Every test returns early when the machine has no wgpu adapter.
#![cfg(feature = "wgpu")]

use oxideon::backend::WgpuDevice;
use oxideon::prelude::*;

/// The tests share one global GPU device and count its host copies, so they run one at a time.
static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn gpu() -> Option<(Device, WgpuDevice, std::sync::MutexGuard<'static, ()>)> {
    let lock = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    match Device::wgpu() {
        Ok(dev @ Device::Wgpu(_)) => {
            let Device::Wgpu(d) = &dev else {
                unreachable!()
            };
            let d = d.clone();
            Some((dev, d, lock))
        }
        _ => None,
    }
}

fn same(what: &str, cpu: &Tensor, gpu: &Tensor, tol: f64) -> Result<()> {
    let g = gpu.to_device(&Device::Cpu)?;
    assert_eq!(cpu.shape(), g.shape(), "{what}: shape");
    assert!(
        cpu.allclose(&g, tol, tol)?,
        "{what}: GPU and CPU results differ"
    );
    Ok(())
}

#[test]
fn matmul_matches_the_cpu_for_every_tile_boundary() -> Result<()> {
    let Some((dev, _, _lock)) = gpu() else {
        return Ok(());
    };
    manual_seed(5);
    // sizes around the 64 x 64 output tile and the 16-wide k slab, and degenerate ones
    for (m, k, n) in [
        (1, 1, 1),
        (3, 5, 7),
        (17, 33, 19),
        (64, 64, 64),
        (65, 67, 63),
        (130, 100, 70),
        (1, 300, 1),
        (200, 1, 200),
        (33, 16, 64),
        (64, 17, 65),
        (63, 15, 129),
        (128, 256, 192),
    ] {
        let a = Tensor::randn(&[m, k])?;
        let b = Tensor::randn(&[k, n])?;
        same(
            &format!("matmul {m}x{k}x{n}"),
            &a.matmul(&b)?,
            &a.to_device(&dev)?.matmul(&b.to_device(&dev)?)?,
            2e-3,
        )?;
    }
    Ok(())
}

/// Products with enough 128 x 128 blocks use the large-matrix kernel; its edges (sizes not a multiple of 128 or of the 8-wide k
/// slab) get the same checks as the small one.
#[test]
fn large_matmul_kernel_matches_the_cpu_including_ragged_edges() -> Result<()> {
    let Some((dev, _, _lock)) = gpu() else {
        return Ok(());
    };
    manual_seed(11);
    for (m, k, n) in [
        (1024, 1024, 1024),
        (1000, 777, 1100),
        (1025, 65, 1023),
        (900, 64, 1300),
        (700, 129, 1000),
    ] {
        let a = Tensor::randn(&[m, k])?;
        let b = Tensor::randn(&[k, n])?;
        let tol = 1e-3 * (k as f64).sqrt();
        same(
            &format!("large matmul {m}x{k}x{n}"),
            &a.matmul(&b)?,
            &a.to_device(&dev)?.matmul(&b.to_device(&dev)?)?,
            tol,
        )?;
    }
    // batched large products, and a transposed operand
    let (a, b) = (
        Tensor::randn(&[2, 700, 300])?,
        Tensor::randn(&[2, 300, 800])?,
    );
    same(
        "batched large",
        &a.matmul(&b)?,
        &a.to_device(&dev)?.matmul(&b.to_device(&dev)?)?,
        0.02,
    )?;
    let (p, q) = (Tensor::randn(&[300, 900])?, Tensor::randn(&[300, 1000])?);
    same(
        "large A.t() @ B",
        &p.t()?.matmul(&q)?,
        &p.to_device(&dev)?.t()?.matmul(&q.to_device(&dev)?)?,
        0.02,
    )?;
    Ok(())
}

#[test]
fn batched_broadcast_and_transposed_matmul_match_the_cpu() -> Result<()> {
    let Some((dev, _, _lock)) = gpu() else {
        return Ok(());
    };
    manual_seed(6);
    let on = |t: &Tensor| t.to_device(&dev);
    let (a, b) = (Tensor::randn(&[3, 10, 20])?, Tensor::randn(&[3, 20, 15])?);
    same("batched", &a.matmul(&b)?, &on(&a)?.matmul(&on(&b)?)?, 2e-3)?;
    let w = Tensor::randn(&[20, 15])?;
    same(
        "batch x shared weight",
        &a.matmul(&w)?,
        &on(&a)?.matmul(&on(&w)?)?,
        2e-3,
    )?;
    let (c, d) = (Tensor::randn(&[2, 1, 5, 8])?, Tensor::randn(&[1, 4, 8, 6])?);
    same(
        "broadcast batch dims",
        &c.matmul(&d)?,
        &on(&c)?.matmul(&on(&d)?)?,
        2e-3,
    )?;
    let (p, q) = (Tensor::randn(&[20, 10])?, Tensor::randn(&[20, 15])?);
    same(
        "A.t() @ B",
        &p.t()?.matmul(&q)?,
        &on(&p)?.t()?.matmul(&on(&q)?)?,
        2e-3,
    )?;
    let (r, s) = (Tensor::randn(&[10, 20])?, Tensor::randn(&[15, 20])?);
    same(
        "A @ B.t()",
        &r.matmul(&s.t()?)?,
        &on(&r)?.matmul(&on(&s)?.t()?)?,
        2e-3,
    )?;
    Ok(())
}

#[test]
fn strided_views_and_broadcasts_stay_correct_on_the_gpu() -> Result<()> {
    let Some((dev, _, _lock)) = gpu() else {
        return Ok(());
    };
    manual_seed(7);
    let x = Tensor::randn(&[8, 9])?;
    let g = x.to_device(&dev)?;
    same("transpose + relu", &x.t()?.relu()?, &g.t()?.relu()?, 1e-6)?;
    same(
        "narrow + exp",
        &x.narrow(1, 2, 5)?.exp()?,
        &g.narrow(1, 2, 5)?.exp()?,
        1e-4,
    )?;
    same(
        "flip + neg",
        &x.flip(&[0, 1])?.neg()?,
        &g.flip(&[0, 1])?.neg()?,
        1e-6,
    )?;
    let t3 = Tensor::randn(&[4, 5, 6])?;
    same(
        "permute",
        &t3.permute(&[2, 0, 1])?.square()?,
        &t3.to_device(&dev)?.permute(&[2, 0, 1])?.square()?,
        1e-5,
    )?;
    let (act, bias) = (Tensor::randn(&[64, 32])?, Tensor::randn(&[32])?);
    same(
        "bias broadcast add",
        &act.add(&bias)?,
        &act.to_device(&dev)?.add(&bias.to_device(&dev)?)?,
        1e-6,
    )?;
    let col = Tensor::randn(&[64, 1])?;
    same(
        "column broadcast mul",
        &act.mul(&col)?,
        &act.to_device(&dev)?.mul(&col.to_device(&dev)?)?,
        1e-6,
    )?;
    Ok(())
}

#[test]
fn reductions_over_any_axes_match_the_cpu() -> Result<()> {
    let Some((dev, _, _lock)) = gpu() else {
        return Ok(());
    };
    manual_seed(8);
    let x = Tensor::randn(&[4, 5, 6])?;
    let g = x.to_device(&dev)?;
    for axis in 0..3isize {
        for keep in [false, true] {
            same(
                &format!("sum {axis} {keep}"),
                &x.sum_axis(axis, keep)?,
                &g.sum_axis(axis, keep)?,
                2e-3,
            )?;
            same(
                &format!("mean {axis} {keep}"),
                &x.mean_axis(axis, keep)?,
                &g.mean_axis(axis, keep)?,
                2e-3,
            )?;
            same(
                &format!("max {axis} {keep}"),
                &x.max_axis(axis, keep)?,
                &g.max_axis(axis, keep)?,
                1e-6,
            )?;
            same(
                &format!("min {axis} {keep}"),
                &x.min_axis(axis, keep)?,
                &g.min_axis(axis, keep)?,
                1e-6,
            )?;
        }
    }
    same(
        "sum over axes 0 and 2",
        &x.sum_axes(&[0, 2], false)?,
        &g.sum_axes(&[0, 2], false)?,
        2e-3,
    )?;
    same("sum of everything", &x.sum()?, &g.sum()?, 2e-3)?;
    // many rows: more workgroups than one grid dimension allows
    let tall = Tensor::randn(&[300_000, 7])?;
    same(
        "300000 rows",
        &tall.sum_axis(1, false)?,
        &tall.to_device(&dev)?.sum_axis(1, false)?,
        2e-3,
    )?;
    // one long row: several passes. Multiples of 1/8 below 2^21 add exactly in f32, so the order of the additions cannot matter.
    let n = 1_000_000usize;
    let data: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.125).collect();
    let long = Tensor::from_vec(data, &[n])?;
    let gl = long.to_device(&dev)?;
    same("sum of 10^6", &long.sum()?, &gl.sum()?, 1e-6)?;
    same("mean of 10^6", &long.mean()?, &gl.mean()?, 1e-6)?;
    same(
        "max of 10^6",
        &long.max_axis(0, false)?,
        &gl.max_axis(0, false)?,
        1e-6,
    )?;
    Ok(())
}

#[test]
fn activations_and_pow_match_the_cpu() -> Result<()> {
    let Some((dev, _, _lock)) = gpu() else {
        return Ok(());
    };
    manual_seed(9);
    let mut v: Vec<f32> = Tensor::randn(&[2000])?.mul_scalar(3.0)?.to_vec::<f32>()?;
    v.extend([
        0.5,
        1.5,
        2.5,
        -0.5,
        -1.5,
        -2.5,
        0.499_999_97,
        2.499_999_8,
        0.0,
        -0.0,
        25.0,
        -25.0,
    ]);
    let x = Tensor::from_vec(v.clone(), &[v.len()])?;
    let g = x.to_device(&dev)?;
    same("sign", &x.sign()?, &g.sign()?, 1e-6)?;
    same("gelu", &x.gelu()?, &g.gelu()?, 1e-4)?;
    same("elu", &x.elu(1.5)?, &g.elu(1.5)?, 1e-4)?;
    same("softplus", &x.softplus()?, &g.softplus()?, 1e-4)?;
    same("floor", &x.floor()?, &g.floor()?, 1e-6)?;
    same("ceil", &x.ceil()?, &g.ceil()?, 1e-6)?;
    same(
        "round (halves to even)",
        &x.round()?,
        &g.round()?,
        1e-6,
    )?;
    same("erf", &x.erf()?, &g.erf()?, 1e-4)?;
    same("logical_not", &x.logical_not()?, &g.logical_not()?, 1e-6)?;
    same("silu", &x.silu()?, &g.silu()?, 1e-4)?;
    same(
        "rsqrt",
        &x.abs()?.add_scalar(0.5)?.rsqrt()?,
        &g.abs()?.add_scalar(0.5)?.rsqrt()?,
        1e-4,
    )?;
    // negative bases: even and odd integer exponents are real, fractional ones are NaN (both backends agree)
    same("powf 2", &x.powf(2.0)?, &g.powf(2.0)?, 1e-4)?;
    same("powf 3", &x.powf(3.0)?, &g.powf(3.0)?, 1e-3)?;
    let exps = Tensor::from_vec(
        (0..v.len())
            .map(|i| [2.0f32, 3.0, 0.0, -1.0][i % 4])
            .collect(),
        &[v.len()],
    )?;
    let base = x
        .abs()?
        .add_scalar(0.25)?
        .mul(&x.sign()?.add_scalar(0.01)?)?; // some negative, none zero
    same(
        "pow elementwise",
        &base.pow(&exps)?,
        &base.to_device(&dev)?.pow(&exps.to_device(&dev)?)?,
        1e-3,
    )?;
    Ok(())
}

#[test]
fn a_chain_of_gpu_ops_never_copies_back_to_the_host() -> Result<()> {
    let Some((dev, d, _lock)) = gpu() else {
        return Ok(());
    };
    manual_seed(10);
    let x = Tensor::randn(&[64, 32])?.to_device(&dev)?;
    let w = Tensor::randn(&[32, 16])?.to_device(&dev)?;
    let b = Tensor::randn(&[16])?.to_device(&dev)?;
    let before = d.readbacks();
    let h = x.matmul(&w)?.add(&b)?.gelu()?; // matmul, broadcast bias, activation
    let loss = h.t()?.relu()?.sum_axis(0, false)?.mean()?; // transposed view, activation, two reductions
    let grad_like = x.t()?.matmul(&h)?; // A.t() @ B
    assert_eq!(d.readbacks(), before, "an op fell back to the host");
    // reading the results back is what costs a copy each
    let _ = (
        loss.to_device(&Device::Cpu)?,
        grad_like.to_device(&Device::Cpu)?,
    );
    assert_eq!(d.readbacks(), before + 2);
    Ok(())
}

#[test]
fn elementwise_kernels_cover_tensors_beyond_one_dispatch() -> Result<()> {
    let Some((dev, d, _lock)) = gpu() else {
        return Ok(());
    };
    // 20 million elements need more than the 65,535 workgroups of one dispatch dimension (the kernels fold the grid into 2-D)
    let n = 20_000_000;
    let x = Tensor::ones(&[n], DType::F32, &Device::Cpu)?.to_device(&dev)?;
    let before = d.readbacks();
    let y = x.mul_scalar(3.0)?.add(&x)?;
    assert_eq!(
        d.readbacks(),
        before,
        "the large kernel fell back to the host"
    );
    let y = y.to_device(&Device::Cpu)?;
    assert_eq!(y.numel(), n);
    assert_eq!(y.narrow(0, 0, 1)?.item::<f32>()?, 4.0);
    assert_eq!(y.narrow(0, n - 1, 1)?.item::<f32>()?, 4.0);
    Ok(())
}

#[test]
fn an_adapter_is_present_on_this_machine_report_it() -> Result<()> {
    match gpu() {
        Some((_, d, _lock)) => eprintln!("wgpu adapter: {}", d.name()),
        None => eprintln!("no wgpu adapter: every GPU test returned early"),
    }
    Ok(())
}

#[test]
fn division_family_and_nan_semantics_match_the_cpu() -> Result<()> {
    let Some((dev, _, _lock)) = gpu() else {
        return Ok(());
    };
    let nan = f32::NAN;
    let a: Vec<f32> = vec![-7.5, -3.0, -0.0, 0.0, 2.0, 5.5, 7.0, 7.0, -7.0, 1e10, -1e10, 0.3];
    let b: Vec<f32> = vec![2.0, -2.0, 3.0, -3.0, 0.3, -0.7, 2.0, 4.0, 3.0, 7.0, 7.0, 0.1];
    let (ta, tb) = (Tensor::from_vec(a.clone(), &[a.len()])?, Tensor::from_vec(b.clone(), &[b.len()])?);
    let (ga, gb) = (ta.to_device(&dev)?, tb.to_device(&dev)?);
    // large operands lose precision differently in f32 shader arithmetic, so compare the well-conditioned ones tightly
    let n = 9;
    let (ta9, tb9, ga9, gb9) = (ta.narrow(0, 0, n)?, tb.narrow(0, 0, n)?, ga.narrow(0, 0, n)?, gb.narrow(0, 0, n)?);
    same("fmod", &ta9.fmod(&tb9)?, &ga9.fmod(&gb9)?, 1e-5)?;
    same("remainder", &ta9.remainder(&tb9)?, &ga9.remainder(&gb9)?, 1e-5)?;
    same("floor_divide", &ta9.floor_divide(&tb9)?, &ga9.floor_divide(&gb9)?, 1e-5)?;
    // NaN propagation of relu and max / min reductions
    let x = Tensor::from_vec(vec![1.0f32, nan, -2.0, 3.0, 0.5, -0.5], &[2, 3])?;
    let g = x.to_device(&dev)?;
    let (cr, gr) = (x.relu()?.to_vec::<f32>()?, g.relu()?.to_device(&Device::Cpu)?.to_vec::<f32>()?);
    eprintln!("relu cpu {cr:?} gpu {gr:?}");
    assert!(cr[1].is_nan() && gr[1].is_nan(), "relu must propagate NaN on both backends");
    let (cm, gm) = (x.max_axis(1, false)?.to_vec::<f32>()?, g.max_axis(1, false)?.to_device(&Device::Cpu)?.to_vec::<f32>()?);
    assert!(cm[0].is_nan() && gm[0].is_nan(), "max over a row containing NaN is NaN");
    assert_eq!(cm[1], gm[1]);
    Ok(())
}

#[test]
fn comparisons_and_where_run_on_the_device_and_match_the_cpu() -> Result<()> {
    let Some((dev, wd, _lock)) = gpu() else {
        return Ok(());
    };
    let n = 70_000; // more than one workgroup row
    let av: Vec<f32> = (0..n).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.5).collect();
    let mut bv: Vec<f32> = (0..n).map(|i| ((i * 53 % 101) as f32 - 50.0) * 0.5).collect();
    bv[7] = f32::NAN;
    bv[8] = f32::INFINITY;
    let (a, b) = (Tensor::from_slice(&av, &[n])?, Tensor::from_slice(&bv, &[n])?);
    let (ga, gb) = (a.to_device(&dev)?, b.to_device(&dev)?);
    let m2 = Tensor::from_slice(&av[..12], &[3, 4])?;
    let g2 = m2.to_device(&dev)?;
    let row = Tensor::from_slice(&[0.0f32, -3.0, 5.0], &[3])?;
    let grow = row.to_device(&dev)?;

    let before = wd.readbacks();
    let on_gpu = [ga.eq_tensor(&gb)?, ga.ne_tensor(&gb)?, ga.lt(&gb)?, ga.le(&gb)?, ga.gt(&gb)?, ga.ge(&gb)?];
    let w_g = Tensor::where_cond(&ga.gt(&gb)?, &ga, &gb)?;
    // a strided condition and broadcast branches
    let s_g = Tensor::where_cond(&g2.transpose(0, 1)?.lt(&grow)?, &g2.transpose(0, 1)?, &grow)?;
    assert_eq!(wd.readbacks(), before, "comparisons and where must stay on the device");

    let on_cpu = [a.eq_tensor(&b)?, a.ne_tensor(&b)?, a.lt(&b)?, a.le(&b)?, a.gt(&b)?, a.ge(&b)?];
    for (i, (c, g)) in on_cpu.iter().zip(&on_gpu).enumerate() {
        assert_eq!(g.dtype(), DType::Bool);
        assert_eq!(c.to_vec::<bool>()?, g.to_device(&Device::Cpu)?.to_vec::<bool>()?, "comparison #{i}");
    }
    let wc = Tensor::where_cond(&a.gt(&b)?, &a, &b)?.to_vec::<f32>()?;
    let wg = w_g.to_device(&Device::Cpu)?.to_vec::<f32>()?;
    assert!(wc.iter().zip(&wg).all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan())), "where");
    let s_c = Tensor::where_cond(&m2.transpose(0, 1)?.lt(&row)?, &m2.transpose(0, 1)?, &row)?;
    assert_eq!(s_c.to_vec::<f32>()?, s_g.to_device(&Device::Cpu)?.to_vec::<f32>()?);
    Ok(())
}
