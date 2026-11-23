// The vectorized f32 kernels must stay accurate over the whole input range, including tiny and huge arguments.
use oxideon_core::Tensor;

fn worst_rel(xs: &[f32], got: &[f32], f: impl Fn(f64) -> f64) -> (f64, f32) {
    let mut worst = (0.0f64, 0.0f32);
    for (&x, &g) in xs.iter().zip(got) {
        let want = f(x as f64);
        let err = if want == 0.0 { g.abs() as f64 } else { ((g as f64 - want) / want).abs() };
        if err > worst.0 {
            worst = (err, x);
        }
    }
    worst
}

fn inputs() -> Vec<f32> {
    // long enough to take the vector path; log-spaced magnitudes from 1e-7 to 80 plus a linear sweep, both signs
    let mut v: Vec<f32> = (0..40_000).map(|i| {
        let m = 10f32.powf(-7.0 + 8.9 * (i as f32) / 40_000.0); // up to about 79: beyond ~87 results are denormal
        if i % 2 == 0 { m } else { -m }
    }).collect();
    v.extend((0..40_000).map(|i| -10.0 + 20.0 * (i as f32) / 40_000.0));
    v
}

#[test]
fn tanh_is_accurate_for_every_magnitude() {
    let xs = inputs();
    let t = Tensor::from_slice(&xs, &[xs.len()]).unwrap();
    let got = t.tanh().unwrap().to_vec::<f32>().unwrap();
    let (err, at) = worst_rel(&xs, &got, f64::tanh);
    assert!(err < 2e-6, "tanh worst relative error {err:e} at x = {at}");
}

#[test]
fn exp_is_accurate_and_handles_the_ends() {
    let xs: Vec<f32> = (0..80_000).map(|i| -100.0 + 200.0 * (i as f32) / 80_000.0).collect();
    let t = Tensor::from_slice(&xs, &[xs.len()]).unwrap();
    let got = t.exp().unwrap().to_vec::<f32>().unwrap();
    for (&x, &g) in xs.iter().zip(&got) {
        if x > 88.73 {
            assert!(g.is_infinite(), "exp({x}) = {g}");
        } else if x > -87.0 && x < 88.7 {
            let want = (x as f64).exp();
            assert!(((g as f64 - want) / want).abs() < 2e-6, "exp({x}) = {g}, want {want}");
        }
    }
}

#[test]
fn gelu_sigmoid_and_silu_are_accurate() {
    let xs = inputs();
    let t = Tensor::from_slice(&xs, &[xs.len()]).unwrap();
    let sigmoid = |x: f64| 1.0 / (1.0 + (-x).exp());
    let (err, at) = worst_rel(&xs, &t.sigmoid().unwrap().to_vec::<f32>().unwrap(), sigmoid);
    assert!(err < 3e-6, "sigmoid worst relative error {err:e} at x = {at}");
    let (err, at) = worst_rel(&xs, &t.silu().unwrap().to_vec::<f32>().unwrap(), |x| x * sigmoid(x));
    assert!(err < 3e-6, "silu worst relative error {err:e} at x = {at}");
    // exact (erf) GELU; compare away from the zero crossing and the far negative tail, where relative error is meaningless
    let erf = |x: f64| libm_erf(x);
    let got = t.gelu().unwrap().to_vec::<f32>().unwrap();
    for (&x, &g) in xs.iter().zip(&got) {
        let xf = x as f64;
        let want = 0.5 * xf * (1.0 + erf(xf / std::f64::consts::SQRT_2));
        assert!((g as f64 - want).abs() <= 3e-6 * (1.0 + want.abs()), "gelu({x}) = {g}, want {want}");
    }
}

// Abramowitz-Stegun 7.1.26 is too coarse; use a high-accuracy series / continued fraction instead.
fn libm_erf(x: f64) -> f64 {
    let ax = x.abs();
    let r = if ax < 2.5 {
        // Taylor series
        let mut term = ax;
        let mut sum = ax;
        let mut n = 0.0;
        while term.abs() > 1e-17 * sum.abs() {
            n += 1.0;
            term *= -ax * ax / n;
            sum += term / (2.0 * n + 1.0);
        }
        2.0 / std::f64::consts::PI.sqrt() * sum
    } else {
        // continued fraction for erfc
        let mut f = 0.0;
        for k in (1..60).rev() {
            f = (k as f64 / 2.0) / (ax + f);
        }
        1.0 - (-ax * ax).exp() / (std::f64::consts::PI.sqrt() * (ax + f))
    };
    if x < 0.0 { -r } else { r }
}

#[test]
fn row_broadcast_binary_ops_match_a_plain_loop() {
    let (rows, cols) = (37usize, 211usize); // odd sizes: the vector kernels have tails
    let a: Vec<f32> = (0..rows * cols).map(|i| ((i as f32) * 0.37).sin() * 5.0 + 6.0).collect();
    let r: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.91).cos() * 2.0 + 3.0).collect();
    let ta = Tensor::from_slice(&a, &[rows, cols]).unwrap();
    let tr = Tensor::from_slice(&r, &[cols]).unwrap();
    type Case<'a> = (&'a str, Box<dyn Fn(&Tensor, &Tensor) -> Tensor + 'a>, Box<dyn Fn(f32, f32) -> f32>);
    let cases: Vec<Case> = vec![
        ("add", Box::new(|x, y| x.add(y).unwrap()), Box::new(|x, y| x + y)),
        ("sub", Box::new(|x, y| x.sub(y).unwrap()), Box::new(|x, y| x - y)),
        ("mul", Box::new(|x, y| x.mul(y).unwrap()), Box::new(|x, y| x * y)),
        ("div", Box::new(|x, y| x.div(y).unwrap()), Box::new(|x, y| x / y)),
    ];
    for (name, op, f) in &cases {
        // row on the right, and row on the left (matters for sub / div)
        let right = op(&ta, &tr).to_vec::<f32>().unwrap();
        let left = op(&tr, &ta).to_vec::<f32>().unwrap();
        for i in 0..rows * cols {
            assert_eq!(right[i], f(a[i], r[i % cols]), "{name} row on the right at {i}");
            assert_eq!(left[i], f(r[i % cols], a[i]), "{name} row on the left at {i}");
        }
    }
    // a 3-D tensor with a row over the last axis, and a row that is a view with an offset
    let t3 = Tensor::from_slice(&a[..4 * 9 * 211 / 4 * 4 / 4], &[1, 1, 4 * 9 * 211 / 4 * 4 / 4]).unwrap_or(ta.clone());
    let _ = t3;
    let big = Tensor::from_slice(&(0..8 * 64 * 100).map(|i| i as f32 * 0.01).collect::<Vec<_>>(), &[8, 64, 100]).unwrap();
    let base = Tensor::from_slice(&(0..300).map(|i| i as f32).collect::<Vec<_>>(), &[300]).unwrap();
    let row = base.narrow(0, 50, 100).unwrap();
    let got = big.add(&row).unwrap().to_vec::<f32>().unwrap();
    let want: Vec<f32> = (0..8 * 64 * 100).map(|i| i as f32 * 0.01 + (50 + i % 100) as f32).collect();
    assert_eq!(got, want);
}

#[test]
fn matmul_with_a_transposed_left_operand_matches_the_explicit_copy() {
    // big enough to take the repacking path (m, k >= 64 and m * k * n >= 2^23), odd sizes, plus a batched case
    for (m, k, n) in [(200usize, 150usize, 300usize), (131, 257, 259)] {
        let x: Vec<f32> = (0..k * m).map(|i| ((i as f32) * 0.37).sin()).collect();
        let g: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.91).cos()).collect();
        let tx = Tensor::from_slice(&x, &[k, m]).unwrap();
        let tg = Tensor::from_slice(&g, &[k, n]).unwrap();
        let fast = tx.transpose(0, 1).unwrap().matmul(&tg).unwrap().to_vec::<f32>().unwrap();
        let reference = tx.transpose(0, 1).unwrap().contiguous().unwrap().to_dtype(oxideon_core::DType::F64).unwrap()
            .matmul(&tg.to_dtype(oxideon_core::DType::F64).unwrap()).unwrap().to_vec::<f64>().unwrap();
        for (i, (&a, &b)) in fast.iter().zip(&reference).enumerate() {
            assert!((a as f64 - b).abs() <= 2e-4 * (1.0 + b.abs()), "{m}x{k}x{n} at {i}: {a} vs {b}");
        }
    }
    let x = Tensor::from_slice(&(0..3 * 128 * 96).map(|i| ((i as f32) * 0.37).sin()).collect::<Vec<_>>(), &[3, 128, 96]).unwrap();
    let g = Tensor::from_slice(&(0..3 * 128 * 160).map(|i| ((i as f32) * 0.91).cos()).collect::<Vec<_>>(), &[3, 128, 160]).unwrap();
    let batched = x.transpose(1, 2).unwrap().matmul(&g).unwrap().to_vec::<f32>().unwrap();
    let reference = x.transpose(1, 2).unwrap().contiguous().unwrap().matmul(&g).unwrap().to_vec::<f32>().unwrap();
    for (i, (&a, &b)) in batched.iter().zip(&reference).enumerate() {
        assert!((a - b).abs() <= 1e-3 * (1.0 + b.abs()), "batched at {i}: {a} vs {b}");
    }
}

#[test]
fn fused_cpu_log_softmax_matches_the_definition_values_and_gradients() {
    for (rows, cols) in [(256usize, 10usize), (7, 1000), (3, 1), (200, 300)] {
        let x: Vec<f32> = (0..rows * cols).map(|i| ((i as f32) * 0.37).sin() * 30.0).collect(); // large logits
        let t = Tensor::from_slice(&x, &[rows, cols]).unwrap().requires_grad_(true);
        let y = t.log_softmax(-1).unwrap();
        let got = y.to_vec::<f32>().unwrap();
        for r in 0..rows {
            let row = &x[r * cols..(r + 1) * cols];
            let m = row.iter().cloned().fold(f32::MIN, f32::max) as f64;
            let lse = row.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln() + m;
            for c in 0..cols {
                let want = row[c] as f64 - lse;
                assert!((got[r * cols + c] as f64 - want).abs() <= 2e-5 * (1.0 + want.abs()), "({r},{c})");
            }
        }
        // gradient of sum(w * log_softmax) is w - softmax * sum(w)
        let w: Vec<f32> = (0..rows * cols).map(|i| ((i as f32) * 0.11).cos()).collect();
        y.mul(&Tensor::from_slice(&w, &[rows, cols]).unwrap()).unwrap().sum().unwrap().backward().unwrap();
        let g = t.grad().unwrap().to_vec::<f32>().unwrap();
        for r in 0..rows {
            let wr = &w[r * cols..(r + 1) * cols];
            let sw: f64 = wr.iter().map(|&v| v as f64).sum();
            for c in 0..cols {
                let want = wr[c] as f64 - (got[r * cols + c] as f64).exp() * sw;
                assert!((g[r * cols + c] as f64 - want).abs() <= 1e-4, "grad ({r},{c}): {} vs {want}", g[r * cols + c]);
            }
        }
    }
    // non-finite rows still go through the careful path
    let t = Tensor::from_slice(&[1.0f32, f32::NEG_INFINITY, 2.0, 0.0], &[2, 2]).unwrap();
    let v = t.log_softmax(-1).unwrap().to_vec::<f32>().unwrap();
    assert!(v[1].is_infinite() && v[1] < 0.0 && (v[0] - 0.0).abs() < 1e-6);
}
