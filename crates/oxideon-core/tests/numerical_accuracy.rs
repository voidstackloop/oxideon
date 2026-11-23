// Numerical accuracy regression tests: float32 results against float64 references, in units of the float32 half-ulp (2^-24).
// The bounds sit just above what oxideon reaches; PyTorch's CPU kernels are far above them on the cases that matter
// (see benchmarks/accuracy_vs_pytorch.py: layer norm of offset data ~5000, saturated softmax ~70, log_softmax near a dominant
// logit ~3000, column sums ~2).
use oxideon_core::nn::functional_ext::layer_norm;
use oxideon_core::Tensor;

const EPS: f64 = 1.0 / 16_777_216.0; // 2^-24

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / (1u64 << 53) as f64 // [0, 1)
    }
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.next().max(1e-300), self.next());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

fn tensor(v: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_slice(v, shape).unwrap()
}

fn out(t: &Tensor) -> Vec<f64> {
    t.contiguous().unwrap().to_vec::<f32>().unwrap().into_iter().map(f64::from).collect()
}

#[cfg(target_arch = "x86_64")]
fn avx2() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

#[cfg(not(target_arch = "x86_64"))]
fn avx2() -> bool {
    false
}

/// Worst `|got - want| / scale` in float32 half-ulps.
fn worst(got: &[f64], want: &[f64], scale: impl Fn(usize) -> f64) -> f64 {
    assert_eq!(got.len(), want.len());
    got.iter().zip(want).enumerate().map(|(i, (g, w))| (g - w).abs() / scale(i) / EPS).fold(0.0, f64::max)
}

#[test]
fn layer_norm_of_data_with_a_large_offset_keeps_the_signal() {
    let (rows, d) = (64usize, 1024usize);
    let mut rng = Lcg(1);
    let x: Vec<f32> = (0..rows * d).map(|_| (1000.0 + rng.normal()) as f32).collect();
    let w: Vec<f32> = (0..d).map(|_| rng.uniform(0.5, 1.5) as f32).collect();
    let b: Vec<f32> = (0..d).map(|_| rng.normal() as f32).collect();
    let mut xhat = vec![0.0f64; rows * d];
    for r in 0..rows {
        let row: Vec<f64> = x[r * d..(r + 1) * d].iter().map(|&v| v as f64).collect();
        let mean = row.iter().sum::<f64>() / d as f64;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / d as f64;
        for i in 0..d {
            xhat[r * d + i] = (row[i] - mean) / (var + 1e-5).sqrt();
        }
    }
    for (with_w, with_b) in [(true, true), (true, false), (false, true), (false, false)] {
        let (xt, wt, bt) = (tensor(&x, &[rows, d]), tensor(&w, &[d]), tensor(&b, &[d]));
        let got = out(&layer_norm(&xt, &[d], with_w.then_some(&wt), with_b.then_some(&bt), 1e-5).unwrap());
        let reference: Vec<f64> = (0..rows * d)
            .map(|k| xhat[k] * if with_w { w[k % d] as f64 } else { 1.0 } + if with_b { b[k % d] as f64 } else { 0.0 })
            .collect();
        // PyTorch's f32 mean puts ~5000 half-ulps of error here; oxideon stays within a few dozen
        let e = worst(&got, &reference, |i| reference[i].abs() + 0.5);
        assert!(e < 120.0, "layer_norm w={with_w} b={with_b}: worst error {e} half-ulps");
    }
}

#[test]
fn column_sums_of_offset_data_do_not_drift() {
    let (rows, cols) = (1024usize, 256usize);
    let mut rng = Lcg(2);
    let x: Vec<f32> = (0..rows * cols).map(|_| (50.0 + rng.normal()) as f32).collect();
    let got = out(&tensor(&x, &[rows, cols]).sum_axis(0, false).unwrap());
    let want: Vec<f64> = (0..cols).map(|c| (0..rows).map(|r| x[r * cols + c] as f64).sum()).collect();
    let scale: Vec<f64> = (0..cols).map(|c| (0..rows).map(|r| (x[r * cols + c] as f64).abs()).sum()).collect();
    let e = worst(&got, &want, |i| scale[i]);
    assert!(e < 1.5, "column sums: worst error {e} half-ulps of the sum of magnitudes (a running f32 sum reaches ~30)");
    let mean = out(&tensor(&x, &[rows, cols]).mean_axis(0, false).unwrap());
    let e = worst(&mean, &want.iter().map(|s| s / rows as f64).collect::<Vec<_>>(), |i| scale[i] / rows as f64);
    assert!(e < 1.5, "column means: worst error {e}");
}

#[test]
fn long_matrix_products_accumulate_in_chunks() {
    let (m, k, n) = (96usize, 512usize, 96usize);
    let mut rng = Lcg(3);
    let a: Vec<f32> = (0..m * k).map(|_| rng.normal() as f32).collect();
    let b: Vec<f32> = (0..k * n).map(|_| rng.normal() as f32).collect();
    let got = out(&tensor(&a, &[m, k]).matmul(&tensor(&b, &[k, n])).unwrap());
    let (mut want, mut scale) = (vec![0.0f64; m * n], vec![0.0f64; m * n]);
    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                let t = a[i * k + p] as f64 * b[p * n + j] as f64;
                want[i * n + j] += t;
                scale[i * n + j] += t.abs();
            }
        }
    }
    let e = worst(&got, &want, |i| scale[i]);
    // one unbroken f32 chain over k = 512 reaches ~3.3 on a product this size (~6 on 512x512); 128-deep chains stay near 1.3
    assert!(e < 2.5, "matmul k=512: worst error {e} half-ulps of the sum of magnitudes");
}

#[test]
fn softmax_and_log_softmax_stay_accurate_for_spread_out_and_dominant_logits() {
    let (rows, cols) = (32usize, 1000usize);
    let mut rng = Lcg(4);
    let x: Vec<f32> = (0..rows * cols).map(|_| (rng.normal() * 10.0) as f32).collect();
    let (mut sm, mut lsm) = (vec![0.0f64; rows * cols], vec![0.0f64; rows * cols]);
    for r in 0..rows {
        let row: Vec<f64> = x[r * cols..(r + 1) * cols].iter().map(|&v| v as f64).collect();
        let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let z: f64 = row.iter().map(|v| (v - m).exp()).sum();
        for i in 0..cols {
            sm[r * cols + i] = (row[i] - m).exp() / z;
            lsm[r * cols + i] = row[i] - m - z.ln();
        }
    }
    let xt = tensor(&x, &[rows, cols]);
    let e = worst(&out(&xt.softmax(-1).unwrap()), &sm, |i| sm[i]);
    // the exponent's argument x - max is rounded unless its error is carried along: ~70 half-ulps without that
    assert!(e < if avx2() { 16.0 } else { 120.0 }, "softmax: worst relative error {e} half-ulps");
    let e = worst(&out(&xt.log_softmax(-1).unwrap()), &lsm, |i| lsm[i].abs() + 1e-3);
    assert!(e < 8.0, "log_softmax: worst error {e}");

    // one dominant logit: the largest log-probability is about -1e-6, which a rounded sum 1 + 1e-6 cannot represent
    let mut dom = vec![0.0f32; cols];
    dom[17] = 20.0;
    let want = -((cols - 1) as f64 * (-20.0f64).exp()).ln_1p();
    let got = out(&tensor(&dom, &[1, cols]).log_softmax(-1).unwrap())[17];
    assert!(((got - want) / want).abs() < 1e-6, "dominant logit: log_softmax {got} vs {want}");
}

#[test]
fn exp_tanh_and_sigmoid_are_within_a_few_half_ulps() {
    let mut rng = Lcg(5);
    let xs: Vec<f32> = (0..200_000).map(|_| rng.uniform(-20.0, 20.0) as f32).collect();
    let t = tensor(&xs, &[xs.len()]);
    let f = |g: fn(f64) -> f64| xs.iter().map(|&v| g(v as f64)).collect::<Vec<_>>();
    let (e, th, sg) = (f(f64::exp), f(f64::tanh), f(|x| 1.0 / (1.0 + (-x).exp())));
    let exp_err = worst(&out(&t.exp().unwrap()), &e, |i| e[i]);
    let tanh_err = worst(&out(&t.tanh().unwrap()), &th, |i| th[i].abs());
    let sig_err = worst(&out(&t.sigmoid().unwrap()), &sg, |i| sg[i]);
    let strict = avx2();
    assert!(exp_err < if strict { 1.5 } else { 6.0 }, "exp: {exp_err}");
    assert!(tanh_err < if strict { 1.25 } else { 6.0 }, "tanh: {tanh_err}");
    assert!(sig_err < if strict { 1.3 } else { 6.0 }, "sigmoid: {sig_err}");
}

#[test]
fn dot_norm_and_long_sums_accumulate_in_f64() {
    let n = 2_000_000usize;
    let mut rng = Lcg(6);
    let a: Vec<f32> = (0..n).map(|_| (3.0 + rng.normal()) as f32).collect();
    let b: Vec<f32> = (0..n).map(|_| (2.0 + rng.normal()) as f32).collect();
    let (ta, tb) = (tensor(&a, &[n]), tensor(&b, &[n]));
    let want: f64 = a.iter().zip(&b).map(|(&x, &y)| x as f64 * y as f64).sum();
    let got = ta.dot(&tb).unwrap().to_vec::<f32>().unwrap()[0] as f64;
    // relative error in half-ulps of 1.0: a correctly rounded float32 is within 1.0 of the exact value (0.5 ulp of its own binade), so
    // 1.0 is the floor; a running f32 sum over millions of terms reaches thousands
    assert!(((got - want) / want).abs() / EPS < 1.01, "dot: {} half-ulps", ((got - want) / want).abs() / EPS);
    let want: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let got = ta.norm().unwrap().to_vec::<f32>().unwrap()[0] as f64;
    assert!(((got - want) / want).abs() / EPS < 1.01, "norm: {} half-ulps", ((got - want) / want).abs() / EPS);
    let want: f64 = a.iter().map(|&x| x as f64).sum();
    let got = ta.sum().unwrap().to_vec::<f32>().unwrap()[0] as f64;
    assert!(((got - want) / want).abs() / EPS < 1.01, "sum: {} half-ulps", ((got - want) / want).abs() / EPS);
}

#[test]
fn erf_is_correctly_rounded_to_within_a_hair() {
    let mut rng = Lcg(7);
    let xs: Vec<f32> = (0..100_000).map(|_| rng.uniform(-4.0, 4.0) as f32).collect();
    let want: Vec<f64> = xs.iter().map(|&x| libm::erf(x as f64)).collect();
    let e = worst(&out(&tensor(&xs, &[xs.len()]).erf().unwrap()), &want, |i| want[i].abs());
    assert!(e < 1.01, "erf: worst relative error {e} half-ulps");
}

#[test]
fn variance_and_normalizations_survive_a_large_offset() {
    use oxideon_core::nn::functional_ext::{batch_norm, group_norm};
    let (n, c, l) = (16usize, 8usize, 64usize);
    let mut rng = Lcg(8);
    let x: Vec<f32> = (0..n * c * l).map(|_| (1000.0 + rng.normal()) as f32).collect();
    let xt = tensor(&x, &[n, c, l]);

    // variance over the last axis: relative to the true variance (~1)
    let got = out(&xt.var_axis(2, false, false).unwrap());
    let want: Vec<f64> = (0..n * c)
        .map(|r| {
            let row: Vec<f64> = x[r * l..(r + 1) * l].iter().map(|&v| v as f64).collect();
            let m = row.iter().sum::<f64>() / l as f64;
            row.iter().map(|v| (v - m).powi(2)).sum::<f64>() / l as f64
        })
        .collect();
    let e = worst(&got, &want, |i| want[i]);
    // the rounded mean alone puts ~1e-4 relative error here (thousands of half-ulps)
    assert!(e < 40.0, "var_axis: worst relative error {e} half-ulps");

    // batch norm over (N, L) per channel: x_hat has unit scale, so absolute error is the relative one
    let got = out(&batch_norm(&xt, None, None, None, None, true, 0.1, 1e-5).unwrap());
    let mut want = vec![0.0f64; x.len()];
    for ch in 0..c {
        let idx = |i: usize, k: usize| (i * c + ch) * l + k;
        let vals: Vec<f64> = (0..n).flat_map(|i| (0..l).map(move |k| (i, k))).map(|(i, k)| x[idx(i, k)] as f64).collect();
        let m = vals.iter().sum::<f64>() / vals.len() as f64;
        let v = vals.iter().map(|a| (a - m).powi(2)).sum::<f64>() / vals.len() as f64;
        for i in 0..n {
            for k in 0..l {
                want[idx(i, k)] = (x[idx(i, k)] as f64 - m) / (v + 1e-5).sqrt();
            }
        }
    }
    let e = worst(&got, &want, |_| 1.0);
    assert!(e < 200.0, "batch_norm: worst error {e} half-ulps of unit scale");

    // group norm, two groups of four channels
    let got = out(&group_norm(&xt, 2, None, None, 1e-5).unwrap());
    let mut want = vec![0.0f64; x.len()];
    let gsz = (c / 2) * l;
    for i in 0..n {
        for g in 0..2 {
            let base = i * c * l + g * gsz;
            let vals: Vec<f64> = x[base..base + gsz].iter().map(|&v| v as f64).collect();
            let m = vals.iter().sum::<f64>() / gsz as f64;
            let v = vals.iter().map(|a| (a - m).powi(2)).sum::<f64>() / gsz as f64;
            for (k, a) in vals.iter().enumerate() {
                want[base + k] = (a - m) / (v + 1e-5).sqrt();
            }
        }
    }
    let e = worst(&got, &want, |_| 1.0);
    assert!(e < 200.0, "group_norm: worst error {e} half-ulps of unit scale");
}

#[test]
fn cosine_similarity_is_correctly_rounded_and_matches_the_differentiable_path() {
    use oxideon_core::nn::loss::cosine_similarity;
    let (rows, d) = (300usize, 257usize);
    let mut rng = Lcg(9);
    let a: Vec<f32> = (0..rows * d).map(|_| rng.normal() as f32).collect();
    let b: Vec<f32> = (0..rows * d).map(|_| (rng.normal() + 0.3) as f32).collect();
    let (ta, tb) = (tensor(&a, &[rows, d]), tensor(&b, &[rows, d]));
    let fast = out(&cosine_similarity(&ta, &tb, 1, 1e-8).unwrap());
    let want: Vec<f64> = (0..rows)
        .map(|r| {
            let (x, y) = (&a[r * d..(r + 1) * d], &b[r * d..(r + 1) * d]);
            let dot: f64 = x.iter().zip(y).map(|(&p, &q)| p as f64 * q as f64).sum();
            let (nx, ny) = (x.iter().map(|&p| (p as f64).powi(2)).sum::<f64>().sqrt(), y.iter().map(|&p| (p as f64).powi(2)).sum::<f64>().sqrt());
            dot / (nx * ny)
        })
        .collect();
    assert!(worst(&fast, &want, |i| want[i].abs()) < 1.01, "cosine_similarity fast path is not correctly rounded");
    // the recorded (differentiable) path agrees to float32 accuracy and keeps its shape
    let slow = cosine_similarity(&ta.clone().requires_grad_(true), &tb, 1, 1e-8).unwrap();
    assert_eq!(slow.shape(), &[rows]);
    for (f, s) in fast.iter().zip(out(&slow)) {
        assert!((f - s).abs() < 1e-4, "fast {f} vs differentiable {s}");
    }
    // a 1-D pair gives a scalar
    let one = cosine_similarity(&tensor(&a[..d], &[d]), &tensor(&b[..d], &[d]), -1, 1e-8).unwrap();
    assert_eq!(one.numel(), 1);
}

#[test]
fn reductions_over_any_axes_accumulate_in_f64() {
    let shape = [12usize, 10, 14, 9];
    let n: usize = shape.iter().product();
    let mut rng = Lcg(10);
    let x: Vec<f32> = (0..n).map(|_| (400.0 + rng.normal()) as f32).collect();
    let t = tensor(&x, &shape);
    let strides = [shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1];
    for axes in [vec![0isize], vec![3], vec![0, 2], vec![1, 3], vec![0, 2, 3], vec![0, 1, 2, 3], vec![1, 2], vec![0, 3]] {
        let kept: Vec<usize> = (0..4).filter(|a| !axes.contains(&(*a as isize))).collect();
        let kept_len: usize = kept.iter().map(|&a| shape[a]).product();
        let (mut want, mut scale) = (vec![0.0f64; kept_len], vec![0.0f64; kept_len]);
        for (flat, &v) in x.iter().enumerate() {
            let mut idx = 0usize;
            for &a in &kept {
                idx = idx * shape[a] + (flat / strides[a]) % shape[a];
            }
            want[idx] += v as f64;
            scale[idx] += (v as f64).abs();
        }
        let got = out(&t.sum_axes(&axes, false).unwrap());
        let e = worst(&got, &want, |i| scale[i]);
        assert!(e < 1.5, "sum over axes {axes:?}: worst error {e} half-ulps of the sum of magnitudes");
        let count = (n / kept_len) as f64;
        let got = out(&t.mean_axes(&axes, false).unwrap());
        let wm: Vec<f64> = want.iter().map(|s| s / count).collect();
        let e = worst(&got, &wm, |i| scale[i] / count);
        assert!(e < 1.5, "mean over axes {axes:?}: worst error {e}");
    }
}

#[test]
fn sigmoid_tanh_silu_edge_values() {
    let xs: Vec<f32> = vec![
        0.0, -0.0, 1e-30, -1e-30, 0.24, 0.26, -0.26, 1.0, -1.0, 8.9, 9.2, -9.2, 20.0, -20.0, 80.0, -80.0, 87.0, -87.0, -88.0, -100.0, -103.0, -120.0, 200.0, -200.0,
        f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 1e-45, 3.4e38, -3.4e38,
    ];
    let n = xs.len() + 3; // keep the SIMD loop and its scalar tail both in play
    let mut padded = xs.clone();
    padded.resize(n, 0.5);
    let t = tensor(&padded, &[n]);
    let (sg, th, si) = (out(&t.sigmoid().unwrap()), out(&t.tanh().unwrap()), out(&t.silu().unwrap()));
    for (i, &x) in xs.iter().enumerate() {
        let xd = x as f64;
        let (ws, wt, wi) = (1.0 / (1.0 + (-xd).exp()), xd.tanh(), xd / (1.0 + (-xd).exp()));
        let check = |name: &str, got: f64, want: f64| {
            if want.is_nan() || xd.is_nan() {
                assert!(got.is_nan(), "{name}({x}) = {got}, want NaN");
            } else {
                // relative to the value, with the subnormal range compared absolutely (the references underflow differently)
                let tol = want.abs() * 4e-7 + 2e-44 * xd.abs().max(1.0);
                assert!(got == want || (got - want).abs() <= tol, "{name}({x}) = {got}, want {want}");
            }
        };
        check("sigmoid", sg[i], if xd.is_infinite() { if xd > 0.0 { 1.0 } else { 0.0 } } else { ws });
        check("tanh", th[i], wt);
        check("silu", si[i], if xd == f64::NEG_INFINITY { f64::NAN /* x * sigmoid(x) = -inf * 0, as before and as x / (1 + e^-x) in PyTorch */ } else if xd == f64::INFINITY { f64::INFINITY } else { wi });
    }
}

#[test]
fn normalize_and_variance_are_correctly_rounded_on_host_f32() {
    let (rows, d) = (200usize, 331usize);
    let mut rng = Lcg(12);
    let x: Vec<f32> = (0..rows * d).map(|_| (rng.normal() * 3.0 + 40.0) as f32).collect();
    let t = tensor(&x, &[rows, d]);
    let got = out(&t.normalize(2.0, 1, 1e-12).unwrap());
    let mut want = vec![0.0f64; rows * d];
    for r in 0..rows {
        let n = x[r * d..(r + 1) * d].iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
        for i in 0..d {
            want[r * d + i] = x[r * d + i] as f64 / n;
        }
    }
    assert!(worst(&got, &want, |i| want[i].abs()) < 1.01, "normalize is not correctly rounded");
    let var = out(&t.var_axis(1, true, false).unwrap());
    let want_var: Vec<f64> = (0..rows)
        .map(|r| {
            let row: Vec<f64> = x[r * d..(r + 1) * d].iter().map(|&v| v as f64).collect();
            let m = row.iter().sum::<f64>() / d as f64;
            row.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (d - 1) as f64
        })
        .collect();
    let ev = worst(&var, &want_var, |i| want_var[i]);
    assert!(ev < 1.01, "variance is not correctly rounded: {ev}");
    // the differentiable path agrees
    let slow = out(&tensor(&x, &[rows, d]).requires_grad_(true).normalize(2.0, 1, 1e-12).unwrap());
    for (a, b) in got.iter().zip(&slow) {
        assert!((a - b).abs() < 1e-5, "{a} vs {b}");
    }
    // an unbiased variance over one element is NaN, as in PyTorch
    assert!(tensor(&[1.0, 2.0], &[2, 1]).var_axis(1, true, false).unwrap().to_vec::<f32>().unwrap().iter().all(|v| v.is_nan()));
}
