// channels_last (NHWC physical, NCHW logical) must change only the memory order: every op sees the same logical values.
use oxideon_core::layout::MemoryFormat;
use oxideon_core::{DType, Tensor};

fn rnd(shape: &[usize], seed: f32) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_slice(&(0..n).map(|i| ((i as f32 + seed) * 0.713).sin()).collect::<Vec<f32>>(), shape).unwrap()
}
fn v(t: &Tensor) -> Vec<f32> {
    t.to_vec::<f32>().unwrap()
}
fn close(what: &str, a: &Tensor, b: &Tensor) {
    let (x, y) = (v(a), v(b));
    assert_eq!(x.len(), y.len(), "{what}");
    for (p, q) in x.iter().zip(&y) {
        assert!((p - q).abs() <= 1e-4 * (1.0 + p.abs()), "{what}: {p} vs {q}");
    }
}

#[test]
fn channels_last_is_a_pure_memory_reorder() {
    let x = rnd(&[2, 3, 4, 5], 0.0);
    assert!(!x.is_channels_last());
    let cl = x.to_channels_last().unwrap();
    assert!(cl.is_channels_last());
    assert_eq!(cl.shape(), x.shape());
    assert_eq!(v(&cl), v(&x), "logical values are unchanged");
    // converting back restores the contiguous layout with the same values
    let back = cl.to_memory_format(MemoryFormat::Contiguous).unwrap();
    assert!(!back.is_channels_last() && back.is_contiguous());
    assert_eq!(v(&back), v(&x));
    // idempotent, and a no-op for layouts that are both (C == 1)
    assert_eq!(v(&cl.to_channels_last().unwrap()), v(&x));
    let single = rnd(&[2, 1, 3, 3], 1.0);
    assert_eq!(v(&single.to_channels_last().unwrap()), v(&single));
    // wrong rank is an error
    assert!(rnd(&[2, 3, 4], 0.0).to_channels_last().is_err());
}

#[test]
fn ops_agree_between_memory_formats() {
    let x = rnd(&[2, 3, 7, 6], 2.0);
    let w = rnd(&[4, 3, 3, 3], 3.0).mul_scalar(0.4).unwrap();
    let b = rnd(&[4], 4.0);
    let xc = x.to_channels_last().unwrap();
    close("conv2d", &x.conv2d(&w, Some(&b), (1, 1), (1, 1), (1, 1)).unwrap(), &xc.conv2d(&w, Some(&b), (1, 1), (1, 1), (1, 1)).unwrap());
    close("strided conv2d", &x.conv2d(&w, None, (2, 1), (0, 1), (1, 1)).unwrap(), &xc.conv2d(&w, None, (2, 1), (0, 1), (1, 1)).unwrap());
    close("relu", &x.relu().unwrap(), &xc.relu().unwrap());
    close("add", &x.add(&x).unwrap(), &xc.add(&xc).unwrap());
    close("mixed add", &x.add(&xc).unwrap(), &xc.add(&x).unwrap());
    close("max_pool2d", &x.max_pool2d((2, 2), (2, 2), (0, 0)).unwrap(), &xc.max_pool2d((2, 2), (2, 2), (0, 0)).unwrap());
    close("sum over channels", &x.sum_axis(1, false).unwrap(), &xc.sum_axis(1, false).unwrap());
    close("reshape", &x.reshape(&[2, 126]).unwrap(), &xc.reshape(&[2, 126]).unwrap());
    close("to_dtype", &x.to_dtype(DType::F64).unwrap(), &xc.to_dtype(DType::F64).unwrap());
    close("batch norm style mean", &x.mean_axes(&[0, 2, 3], false).unwrap(), &xc.mean_axes(&[0, 2, 3], false).unwrap());
}

#[test]
fn gradients_flow_through_channels_last() {
    let w = rnd(&[4, 3, 3, 3], 5.0).mul_scalar(0.4).unwrap().requires_grad_(true);
    let x = rnd(&[2, 3, 6, 6], 6.0).requires_grad_(true);
    let weights = rnd(&[2, 4, 6, 6], 7.0);
    let run = |channels_last: bool| {
        w.set_grad(None);
        x.set_grad(None);
        let input = if channels_last { x.to_channels_last().unwrap() } else { x.clone() };
        let y = input.conv2d(&w, None, (1, 1), (1, 1), (1, 1)).unwrap();
        y.mul(&weights).unwrap().sum().unwrap().backward().unwrap();
        (x.grad().unwrap(), w.grad().unwrap())
    };
    let (gx, gw) = run(false);
    let (gx_cl, gw_cl) = run(true);
    close("grad x", &gx, &gx_cl);
    close("grad w", &gw, &gw_cl);
}
