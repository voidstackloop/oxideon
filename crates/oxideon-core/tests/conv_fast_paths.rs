// The fused CPU conv2d (Winograd 3x3, AVX2 direct, batched im2col) and its hand-written backward must agree with the generic
// im2col path, which the f64 dtype takes. Values and all three gradients are compared across shapes that trigger each kernel.
use oxideon_core::{DType, Tensor};

fn rnd(shape: &[usize], seed: f32) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_slice(&(0..n).map(|i| ((i as f32 + seed) * 0.713).sin()).collect::<Vec<f32>>(), shape).unwrap()
}

fn max_err(a: &Tensor, b: &Tensor) -> f64 {
    let x = a.to_dtype(DType::F64).unwrap().to_vec::<f64>().unwrap();
    let y = b.to_dtype(DType::F64).unwrap().to_vec::<f64>().unwrap();
    assert_eq!(x.len(), y.len());
    x.iter().zip(&y).map(|(p, q)| (p - q).abs() / (1.0 + q.abs())).fold(0.0, f64::max)
}

#[test]
fn fast_conv_paths_match_the_generic_path() {
    // (n, c_in, h, w, c_out, k, stride, pad, dil, bias)
    #[allow(clippy::type_complexity)]
    let cases: &[(usize, usize, usize, usize, usize, usize, usize, usize, usize, bool)] = &[
        (2, 8, 12, 12, 8, 3, 1, 1, 1, true),    // winograd
        (1, 16, 9, 11, 12, 3, 1, 0, 1, true),   // winograd, odd sizes, no padding
        (3, 8, 8, 8, 8, 3, 1, 2, 1, false),     // winograd, padding > 1, no bias
        (2, 4, 16, 16, 8, 3, 1, 1, 1, true),    // avx direct (c_in < 8)
        (2, 3, 17, 19, 6, 5, 1, 2, 1, true),    // avx direct, k=5, c_out % 4 != 0
        (2, 8, 10, 13, 9, 3, 1, 1, 1, true),    // winograd, odd output width
        (1, 8, 7, 7, 8, 3, 1, 1, 1, true),      // winograd, odd output height and width
        (1, 8, 6, 6, 8, 3, 1, 3, 1, false),    // winograd, padding wider than a tile
        (5, 9, 5, 30, 11, 3, 1, 0, 1, true),    // winograd, wide, batch > micro-batch
        (1, 72, 6, 6, 70, 3, 1, 1, 1, false),   // winograd, channels too large for the small-product kernel
        (2, 8, 15, 15, 8, 3, 2, 1, 1, true),    // strided
        (2, 6, 14, 14, 5, 3, 1, 2, 2, true),    // dilated
        (4, 5, 6, 6, 7, 1, 1, 0, 1, true),      // 1x1
        (1, 3, 5, 5, 4, 3, 1, 1, 1, false),     // tiny
    ];
    for &(n, ci, h, w, co, k, s, p, d, bias) in cases {
        let tag = format!("n{n} ci{ci} {h}x{w} co{co} k{k} s{s} p{p} d{d} bias{bias}");
        let x = rnd(&[n, ci, h, w], 1.0);
        let wt = rnd(&[co, ci, k, k], 2.0).mul_scalar(0.3).unwrap();
        let b = rnd(&[co], 3.0);
        let run = |dt: DType| {
            let leaf = |t: &Tensor| t.to_dtype(dt).unwrap().detach().requires_grad_(true);
            let (x, wt, b) = (leaf(&x), leaf(&wt), leaf(&b));
            let y = x.conv2d(&wt, bias.then_some(&b), (s, s), (p, p), (d, d)).unwrap();
            let g = rnd(y.shape(), 7.0).to_dtype(dt).unwrap();
            y.mul(&g).unwrap().sum().unwrap().backward().unwrap();
            (y, x.grad().unwrap(), wt.grad().unwrap(), bias.then(|| b.grad().unwrap()))
        };
        let (y32, gx32, gw32, gb32) = run(DType::F32);
        let (y64, gx64, gw64, gb64) = run(DType::F64);
        assert!(max_err(&y32, &y64) < 2e-4, "{tag}: value {}", max_err(&y32, &y64));
        assert!(max_err(&gx32, &gx64) < 2e-4, "{tag}: grad x {}", max_err(&gx32, &gx64));
        assert!(max_err(&gw32, &gw64) < 5e-4, "{tag}: grad w {}", max_err(&gw32, &gw64));
        if let (Some(a), Some(b)) = (gb32, gb64) {
            assert!(max_err(&a, &b) < 5e-4, "{tag}: grad bias {}", max_err(&a, &b));
        }
    }
}
