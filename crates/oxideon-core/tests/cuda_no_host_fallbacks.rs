// Ops that used to compute on the host even for CUDA tensors (conv_transpose2d, cumprod, isnan/isinf/isfinite) now run on
// the device: same values and gradients as the CPU backend, and no device-to-host transfer.
use oxideon_core::backend::{cuda_ops::download_count, cuda_rt};
use oxideon_core::{DType, Device, Tensor};
use std::sync::Mutex;

// the download counter is process-wide, so tests that assert on it must not overlap
static SERIAL: Mutex<()> = Mutex::new(());

fn rnd(shape: &[usize], seed: f32) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_slice(&(0..n).map(|i| ((i as f32 + seed) * 0.713).sin()).collect::<Vec<f32>>(), shape).unwrap()
}

fn close(what: &str, a: &Tensor, b: &Tensor) {
    let x = a.to_device(&Device::Cpu).unwrap().to_dtype(DType::F64).unwrap().to_vec::<f64>().unwrap();
    let y = b.to_device(&Device::Cpu).unwrap().to_dtype(DType::F64).unwrap().to_vec::<f64>().unwrap();
    assert_eq!(x.len(), y.len(), "{what}");
    for (p, q) in x.iter().zip(&y) {
        assert!((p - q).abs() <= 1e-4 * (1.0 + p.abs()), "{what}: {p} vs {q}");
    }
}

/// runs `f` on CPU and CUDA copies of `inputs`; compares the value and every input gradient; asserts no host traffic
fn check(what: &str, inputs: &[Tensor], f: impl Fn(&[Tensor]) -> Tensor) {
    let dev = Device::Cuda(0);
    let cpu: Vec<Tensor> = inputs.iter().map(|t| t.detach().requires_grad_(true)).collect();
    let gpu: Vec<Tensor> = inputs.iter().map(|t| t.detach().to_device(&dev).unwrap().requires_grad_(true)).collect();
    let yc = f(&cpu);
    let before = download_count();
    let yg = f(&gpu);
    let w = rnd(yc.shape(), 91.0);
    yc.mul(&w).unwrap().sum().unwrap().backward().unwrap();
    yg.mul(&w.to_device(&dev).unwrap()).unwrap().sum().unwrap().backward().unwrap();
    assert_eq!(download_count(), before, "{what}: touched the host");
    close(&format!("{what} value"), &yc, &yg);
    for (i, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
        close(&format!("{what} grad {i}"), &c.grad().unwrap(), &g.grad().unwrap());
    }
}

#[test]
fn conv_transpose2d_runs_on_the_gpu() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if !cuda_rt::available() {
        return;
    }
    let x = rnd(&[2, 3, 5, 4], 0.0);
    let w = rnd(&[3, 4, 3, 3], 1.0);
    let b = rnd(&[4], 2.0);
    for (stride, padding, out_pad, dil) in [((1, 1), (0, 0), (0, 0), (1, 1)), ((2, 2), (1, 1), (1, 1), (1, 1)), ((2, 1), (1, 0), (1, 0), (1, 1)), ((1, 2), (2, 1), (0, 1), (2, 1))] {
        check(&format!("conv_transpose2d s{stride:?} p{padding:?} op{out_pad:?} d{dil:?}"), &[x.clone(), w.clone(), b.clone()], |t| {
            t[0].conv_transpose2d(&t[1], Some(&t[2]), stride, padding, out_pad, dil).unwrap()
        });
    }
}

#[test]
fn cumprod_runs_on_the_gpu_including_zeros() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if !cuda_rt::available() {
        return;
    }
    let mut v: Vec<f32> = (0..2 * 6 * 3).map(|i| 0.5 + ((i as f32) * 0.37).sin().abs()).collect();
    v[7] = 0.0; // a zero in the middle of a scan
    v[24] = 0.0;
    let x = Tensor::from_slice(&v, &[2, 6, 3]).unwrap();
    for axis in [0isize, 1, 2, -1] {
        check(&format!("cumprod axis {axis}"), std::slice::from_ref(&x), |t| t[0].cumprod(axis).unwrap());
    }
}

#[test]
fn nan_inf_and_finite_predicates_run_on_the_gpu() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if !cuda_rt::available() {
        return;
    }
    let vals = [1.0f32, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -2.5];
    let cpu = Tensor::from_slice(&vals, &[2, 3]).unwrap();
    let gpu = cpu.to_device(&Device::Cuda(0)).unwrap();
    let before = download_count();
    let on_gpu = [gpu.isnan().unwrap(), gpu.isinf().unwrap(), gpu.isfinite().unwrap()];
    assert_eq!(download_count(), before, "predicates touched the host");
    for (name, c, g) in [("isnan", cpu.isnan().unwrap(), &on_gpu[0]), ("isinf", cpu.isinf().unwrap(), &on_gpu[1]), ("isfinite", cpu.isfinite().unwrap(), &on_gpu[2])] {
        assert_eq!(g.dtype(), DType::Bool, "{name}");
        assert_eq!(c.to_vec::<bool>().unwrap(), g.to_device(&Device::Cpu).unwrap().to_vec::<bool>().unwrap(), "{name}");
    }
}

#[test]
fn logical_ops_run_on_the_gpu() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if !cuda_rt::available() {
        return;
    }
    let a = Tensor::from_slice(&[0.0f32, 1.0, f32::NAN, -2.0, 0.0, 3.0], &[2, 3]).unwrap();
    let b = Tensor::from_slice(&[0.0f32, 0.0, 1.0, 1.0, 5.0, 0.0], &[2, 3]).unwrap();
    let (ga, gb) = (a.to_device(&Device::Cuda(0)).unwrap(), b.to_device(&Device::Cuda(0)).unwrap());
    let before = download_count();
    let on_gpu = [ga.logical_and(&gb).unwrap(), ga.logical_or(&gb).unwrap(), ga.logical_xor(&gb).unwrap()];
    assert_eq!(download_count(), before, "logical ops touched the host");
    let on_cpu = [a.logical_and(&b).unwrap(), a.logical_or(&b).unwrap(), a.logical_xor(&b).unwrap()];
    for (c, g) in on_cpu.iter().zip(&on_gpu) {
        assert_eq!(c.to_vec::<bool>().unwrap(), g.to_device(&Device::Cpu).unwrap().to_vec::<bool>().unwrap());
    }
}

#[test]
fn full_sort_of_a_long_row_runs_on_the_gpu() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if !cuda_rt::available() {
        return;
    }
    let x = rnd(&[3, 10_000], 3.0).to_device(&Device::Cuda(0)).unwrap();
    let (launches, downloads) = (cuda_rt::launch_count(), download_count());
    let (v, i) = x.sort(1, true).unwrap();
    let (v2, _) = x.topk(5000, 1, false, true).unwrap();
    assert!(cuda_rt::launch_count() > launches + 100, "the bitonic network should run as kernels");
    assert_eq!(download_count(), downloads, "sort touched the host");
    let c = x.to_device(&Device::Cpu).unwrap();
    close("sort values", &v, &c.sort(1, true).unwrap().0);
    close("sort indices", &i, &c.sort(1, true).unwrap().1);
    close("topk k=5000", &v2, &c.topk(5000, 1, false, true).unwrap().0);
}

#[test]
fn cross_entropy_with_tensor_targets_stays_on_the_gpu() {
    use oxideon_core::nn::loss_ext::{Reduction, Target, cross_entropy_nd, nll_loss_nd};
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if !cuda_rt::available() {
        return;
    }
    let logits = rnd(&[6, 5], 4.0);
    let targets = Tensor::from_slice(&[0i64, 4, -100, 2, 2, 1], &[6]).unwrap(); // one ignored entry
    let weight = Tensor::from_slice(&[1.0f32, 2.0, 0.5, 1.5, 1.0], &[5]).unwrap();
    for (ignore, w, smooth, r) in [
        (None, false, 0.0, Reduction::Mean),
        (Some(-100), false, 0.0, Reduction::Mean),
        (Some(-100), true, 0.0, Reduction::Sum),
        (Some(-100), true, 0.1, Reduction::Mean),
        (Some(-100), false, 0.2, Reduction::None),
    ] {
        // without ignore_index every target must be a valid class
        let tg = if ignore.is_none() { targets.mul_scalar(0.0).unwrap().add_scalar(3.0).unwrap().to_dtype(DType::I64).unwrap() } else { targets.clone() };
        let wt = w.then_some(&weight);
        let c = cross_entropy_nd(&logits.detach().requires_grad_(true), Target::Indices(&tg), wt, ignore, smooth, r).unwrap();
        let dev = Device::Cuda(0);
        let (gl, gt, gw) = (logits.to_device(&dev).unwrap().requires_grad_(true), tg.to_device(&dev).unwrap(), weight.to_device(&dev).unwrap());
        let before = download_count();
        let g = cross_entropy_nd(&gl, Target::Indices(&gt), w.then_some(&gw), ignore, smooth, r).unwrap();
        g.sum().unwrap().backward().unwrap();
        assert_eq!(download_count(), before, "cross_entropy_nd {ignore:?} w={w} ls={smooth} touched the host");
        close("cross_entropy_nd", &c, &g);
    }
    let lp = logits.log_softmax(1).unwrap();
    close("nll_loss_nd", &nll_loss_nd(&lp, &targets, None, Some(-100), Reduction::Mean).unwrap(), &nll_loss_nd(&lp.to_device(&Device::Cuda(0)).unwrap(), &targets.to_device(&Device::Cuda(0)).unwrap(), None, Some(-100), Reduction::Mean).unwrap());
    // an out-of-range class is still an error
    let bad = Tensor::from_slice(&[0i64, 9, 1, 2, 2, 1], &[6]).unwrap();
    assert!(nll_loss_nd(&lp, &bad, None, Some(-100), Reduction::Mean).is_err());
    let r = nll_loss_nd(&lp.to_device(&Device::Cuda(0)).unwrap(), &bad.to_device(&Device::Cuda(0)).unwrap(), None, Some(-100), Reduction::Mean).and_then(|t| t.item::<f32>());
    assert!(r.is_err(), "device-side validation must reject class 9 of 5: {r:?}");
}

#[test]
fn integer_matmul_runs_on_the_gpu_and_wraps_like_the_cpu() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if !cuda_rt::available() {
        return;
    }
    for dt in [DType::I32, DType::I64] {
        for (scale, shape_a, shape_b) in [(50.0f32, vec![37usize, 53], vec![53usize, 29]), (50.0, vec![3, 40, 17], vec![17, 21]), (60_000.0, vec![8, 300], vec![300, 8])] {
            let a = rnd(&shape_a, 1.0).mul_scalar(scale as f64).unwrap().to_dtype(dt).unwrap();
            let b = rnd(&shape_b, 9.0).mul_scalar(scale as f64).unwrap().to_dtype(dt).unwrap();
            let c = a.matmul(&b).unwrap();
            let before = download_count();
            let g = a.to_device(&Device::Cuda(0)).unwrap().matmul(&b.to_device(&Device::Cuda(0)).unwrap()).unwrap();
            assert_eq!(download_count(), before, "{dt:?} matmul touched the host");
            assert_eq!(c.to_vec::<i64>().unwrap(), g.to_device(&Device::Cpu).unwrap().to_vec::<i64>().unwrap(), "{dt:?} {shape_a:?}x{shape_b:?} scale {scale}");
        }
    }
}
