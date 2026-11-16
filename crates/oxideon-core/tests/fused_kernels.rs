//! The specialised `f32` kernels (direct max pooling, per-image convolution, vectorizable exp / sigmoid / gelu) against
//! the general paths they shortcut: the same operation in `f64`, or with gradient tracking on, still goes the old way.

use oxideon_core::{DType, Tensor};

fn img(shape: &[usize], seed: f32) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_slice(&(0..n).map(|i| ((i as f32 + seed) * 0.6180339).sin() * 3.0).collect::<Vec<f32>>(), shape).unwrap()
}

fn close(what: &str, a: &Tensor, b: &Tensor, tol: f64) {
    assert_eq!(a.shape(), b.shape(), "{what}: shape");
    let a = a.to_dtype(DType::F64).unwrap().to_vec::<f64>().unwrap();
    let b = b.to_dtype(DType::F64).unwrap().to_vec::<f64>().unwrap();
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert!((x - y).abs() <= tol * (1.0 + y.abs()) || (x.is_nan() && y.is_nan()), "{what}: element {i}: {x} vs {y}");
    }
}

#[test]
fn direct_max_pool_matches_the_general_path_in_values_and_gradients() {
    for (k, s, p) in [((2, 2), (2, 2), (0, 0)), ((3, 3), (1, 1), (1, 1)), ((3, 2), (2, 1), (1, 0)), ((5, 5), (3, 3), (2, 2)), ((2, 2), (3, 3), (1, 1))] {
        let x32 = img(&[2, 3, 11, 9], 1.0);
        let x64 = x32.to_dtype(DType::F64).unwrap();
        let what = format!("kernel {k:?} stride {s:?} padding {p:?}");
        close(&what, &x32.max_pool2d(k, s, p).unwrap(), &x64.max_pool2d(k, s, p).unwrap(), 0.0);

        let (g32, g64) = (x32.detach().requires_grad_(true), x64.detach().requires_grad_(true));
        let probe = img(x32.max_pool2d(k, s, p).unwrap().shape(), 9.0);
        g32.max_pool2d(k, s, p).unwrap().mul(&probe).unwrap().sum().unwrap().backward().unwrap();
        g64.max_pool2d(k, s, p).unwrap().mul(&probe.to_dtype(DType::F64).unwrap()).unwrap().sum().unwrap().backward().unwrap();
        close(&format!("{what} gradient"), &g32.grad().unwrap(), &g64.grad().unwrap(), 1e-6);
    }
    // all-negative windows next to padding: padding must not win; a NaN must propagate
    let mut v = vec![-5.0f32; 16];
    v[0] = f32::NAN; // the top-left corner belongs to the first window only
    let x = Tensor::from_slice(&v, &[1, 1, 4, 4]).unwrap();
    let y = x.max_pool2d((3, 3), (2, 2), (1, 1)).unwrap().to_vec::<f32>().unwrap();
    assert!(y[0].is_nan());
    assert_eq!(&y[1..], &[-5.0, -5.0, -5.0]);
}

#[test]
fn per_image_convolution_matches_the_autograd_path() {
    for (stride, padding, dilation) in [((1, 1), (1, 1), (1, 1)), ((2, 1), (0, 2), (1, 1)), ((1, 2), (2, 1), (2, 1)), ((3, 3), (1, 1), (1, 2))] {
        let x = img(&[3, 4, 13, 10], 0.0);
        let w = img(&[5, 4, 3, 3], 2.0).mul_scalar(0.2).unwrap();
        let b = img(&[5], 4.0);
        for bias in [None, Some(&b)] {
            let fast = x.conv2d(&w, bias, stride, padding, dilation).unwrap();
            // requiring a gradient sends the same call down the unfold + matmul path
            let slow = x.detach().requires_grad_(true).conv2d(&w, bias, stride, padding, dilation).unwrap();
            close(&format!("stride {stride:?} padding {padding:?} dilation {dilation:?} bias {}", bias.is_some()), &fast, &slow, 2e-5);
        }
    }
    // a strided view of the input falls back and still agrees
    let x = img(&[2, 3, 8, 8], 5.0);
    let w = img(&[2, 3, 3, 3], 6.0);
    let v = x.transpose(2, 3).unwrap();
    close("view", &v.conv2d(&w, None, (1, 1), (1, 1), (1, 1)).unwrap(), &v.contiguous().unwrap().conv2d(&w, None, (1, 1), (1, 1), (1, 1)).unwrap(), 1e-6);
}

#[test]
fn vectorizable_exp_sigmoid_and_gelu_stay_within_a_few_ulp_of_the_f64_results() {
    // dense sweep plus the edges: overflow, underflow into subnormals, zero, signed infinities, NaN
    let mut v: Vec<f32> = (-40_000..=40_000).map(|i| i as f32 * 0.0025).collect();
    v.extend([88.7, 88.73, 89.5, 200.0, -87.3, -95.0, -103.9, -104.5, -150.0, -1000.0, 0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 1e-20, -1e-20]);
    let x32 = Tensor::from_slice(&v, &[v.len()]).unwrap();
    // references in f64 from formulas that stay accurate in the tails (`erfc`, not `1 + erf`, for the normal CDF)
    type Op = fn(&Tensor) -> Tensor;
    type Ref = fn(f64) -> f64;
    let sig: Ref = |x| if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { x.exp() / (1.0 + x.exp()) };
    let ops: [(&str, Op, Ref, f64); 4] = [
        ("exp", |t| t.exp().unwrap(), |x| x.exp(), 4e-7),
        ("sigmoid", |t| t.sigmoid().unwrap(), sig, 4e-7),
        ("silu", |t| t.silu().unwrap(), |x| x * if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { x.exp() / (1.0 + x.exp()) }, 6e-7),
        ("gelu", |t| t.gelu().unwrap(), |x| x * 0.5 * libm::erfc(-x * std::f64::consts::FRAC_1_SQRT_2), 4e-6),
    ];
    for (name, f, reference, rel) in ops {
        let got = f(&x32).to_vec::<f32>().unwrap();
        for (&g, &xi) in got.iter().zip(&v) {
            let w = reference(xi as f64);
            // gelu's far left tail (|value| < 1e-10): the exponent -x^2/2 is itself rounded to f32 there, which costs
            // up to ~1e-5 relative on numbers that small
            let rel = if name == "gelu" && w.abs() < 1e-10 { 5e-5 } else { rel };
            let w32 = w as f32; // the correctly rounded single-precision answer
            let ok = if w.is_nan() {
                g.is_nan()
            } else if w32.is_infinite() || w32 == 0.0 {
                g == w32 || (g as f64 - w).abs() <= 3e-45 * (1.0 + xi.abs() as f64) // exact, or a subnormal grid step away
            } else {
                // relative on the normal range; in the subnormal range one grid step (1.4e-45) of the exponential,
                // scaled by the factor silu / gelu multiply it with
                ((g as f64 - w) / w).abs() <= rel || (g as f64 - w).abs() <= 3e-45 * (1.0 + xi.abs() as f64)
            };
            assert!(ok, "{name}({xi}) = {g}, expected {w}");
        }
    }
}
