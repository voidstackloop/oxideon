use oxideon_core::backend::{cuda_ops::download_count, cuda_rt};
use oxideon_core::{Device, Tensor};
use std::sync::Mutex;

static SERIAL: Mutex<()> = Mutex::new(()); // the download counter is process-wide

fn rnd(shape: &[usize], seed: f32) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_slice(&(0..n).map(|i| ((i as f32 + seed) * 0.713).sin()).collect::<Vec<f32>>(), shape).unwrap()
}

// (equation, operand shapes)
const CASES: &[(&str, &[&[usize]])] = &[
    ("ij,jk->ik", &[&[3, 4], &[4, 5]]),
    ("bij,bkj->bik", &[&[2, 3, 4], &[2, 5, 4]]),
    ("bij,kj->bik", &[&[2, 3, 4], &[5, 4]]),
    ("i,j->ij", &[&[3], &[4]]),
    ("ij,ij->", &[&[3, 4], &[3, 4]]),
    ("ij->j", &[&[3, 4]]),
    ("ij->ji", &[&[3, 4]]),
    ("abc,cd,de->abe", &[&[2, 3, 4], &[4, 5], &[5, 2]]),
    ("bhqd,bhkd->bhqk", &[&[2, 2, 3, 4], &[2, 2, 5, 4]]),
    ("ab,cd->dcba", &[&[2, 3], &[4, 5]]),
    ("ijk,jl->il", &[&[2, 3, 4], &[3, 5]]),
    ("i,i,i->i", &[&[4], &[4], &[4]]),
    ("ii->i", &[&[4, 4]]),
    ("ii->", &[&[5, 5]]),
    ("bii->bi", &[&[2, 3, 3]]),
    ("iij,jk->ik", &[&[3, 3, 4], &[4, 2]]),
    ("ij,ij->ij", &[&[3, 3], &[3, 3]]),
];

#[test]
fn einsum_matches_the_cpu_values_and_gradients_on_cuda_without_host_trips() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if !cuda_rt::available() {
        return;
    }
    let dev = Device::Cuda(0);
    for (eq, shapes) in CASES {
        let cpu: Vec<Tensor> = shapes.iter().enumerate().map(|(i, s)| rnd(s, i as f32 * 5.0).requires_grad_(true)).collect();
        let gpu: Vec<Tensor> = cpu.iter().map(|t| t.detach().to_device(&dev).unwrap().requires_grad_(true)).collect();
        let yc = Tensor::einsum(eq, &cpu.iter().collect::<Vec<_>>()).unwrap();
        let before = download_count();
        let yg = Tensor::einsum(eq, &gpu.iter().collect::<Vec<_>>()).unwrap();
        let w = rnd(yc.shape(), 77.0);
        yc.mul(&w).unwrap().sum().unwrap().backward().unwrap();
        yg.mul(&w.to_device(&dev).unwrap()).unwrap().sum().unwrap().backward().unwrap();
        assert_eq!(download_count(), before, "{eq}: touched the host");
        let close = |a: &Tensor, b: &Tensor, what: &str| {
            let (x, y) = (a.to_vec::<f32>().unwrap(), b.to_device(&Device::Cpu).unwrap().to_vec::<f32>().unwrap());
            assert_eq!(x.len(), y.len(), "{eq} {what}");
            for (p, q) in x.iter().zip(&y) {
                assert!((p - q).abs() <= 1e-4 * (1.0 + p.abs()), "{eq} {what}: {p} vs {q}");
            }
        };
        close(&yc, &yg, "value");
        for (i, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
            close(&c.grad().unwrap(), &g.grad().unwrap(), &format!("grad {i}"));
        }
    }
}

#[test]
fn integer_and_mixed_dtype_einsum_runs_on_every_device() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    use oxideon_core::DType;
    let a = Tensor::from_slice(&[1i64, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
    let b = Tensor::from_slice(&[7i64, 8, 9, 10, 11, 12], &[3, 2]).unwrap();
    let want = vec![58i64, 64, 139, 154];
    assert_eq!(Tensor::einsum("ij,jk->ik", &[&a, &b]).unwrap().to_vec::<i64>().unwrap(), want);
    if cuda_rt::available() {
        let dev = Device::Cuda(0);
        let before = download_count();
        let y = Tensor::einsum("ij,jk->ik", &[&a.to_device(&dev).unwrap(), &b.to_device(&dev).unwrap()]).unwrap();
        assert_eq!(download_count(), before);
        assert_eq!(y.to_vec::<i64>().unwrap(), want);
    }
    // float x int promotes to the float dtype
    let f = Tensor::from_slice(&[0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0], &[2, 3]).unwrap();
    let y = Tensor::einsum("ij,jk->ik", &[&f, &b]).unwrap();
    assert_eq!(y.dtype(), DType::F32);
    assert_eq!(y.to_vec::<f32>().unwrap(), vec![29.0, 32.0, 69.5, 77.0]);
    // trace of an integer matrix
    let m = Tensor::from_slice(&[1i64, 2, 3, 4], &[2, 2]).unwrap();
    assert_eq!(Tensor::einsum("ii->", &[&m]).unwrap().to_vec::<i64>().unwrap(), vec![5]);
}
