// The matmul accuracy modes, one test so the process-wide switch is not raced by other tests.
use oxideon_core::{precision_mode, set_precision_mode, PrecisionMode};
use oxideon_core::Tensor;

const EPS: f64 = 1.0 / 16_777_216.0;

// the precision mode is process-wide, so the tests that switch it take turns
static MODE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lcg(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / (1u64 << 53) as f64 - 0.5
}

#[test]
fn precision_modes_order_by_error() {
    let _guard = MODE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (m, k, n) = (64usize, 1024usize, 64usize);
    let mut s = 11u64;
    let a: Vec<f32> = (0..m * k).map(|_| (lcg(&mut s) * 4.0) as f32).collect();
    let b: Vec<f32> = (0..k * n).map(|_| (lcg(&mut s) * 4.0) as f32).collect();
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
    let (ta, tb) = (Tensor::from_slice(&a, &[m, k]).unwrap(), Tensor::from_slice(&b, &[k, n]).unwrap());
    let mut errs = Vec::new();
    assert_eq!(precision_mode(), PrecisionMode::Balanced, "default");
    for mode in [PrecisionMode::Fastest, PrecisionMode::Balanced, PrecisionMode::Highest] {
        set_precision_mode(mode);
        assert_eq!(precision_mode(), mode);
        let got = ta.matmul(&tb).unwrap().contiguous().unwrap().to_vec::<f32>().unwrap();
        let e = (0..m * n).map(|i| (got[i] as f64 - want[i]).abs() / scale[i] / EPS).fold(0.0, f64::max);
        errs.push(e);
    }
    set_precision_mode(PrecisionMode::Balanced);
    // Highest is one final rounding of the f64 result; Balanced beats an unbroken f32 chain
    assert!(errs[2] < 0.51, "highest: {}", errs[2]);
    assert!(errs[1] < errs[0], "balanced {} should beat fast {}", errs[1], errs[0]);
    assert!(errs[1] < 3.0, "balanced: {}", errs[1]);
}

#[test]
fn mode_names_parse_and_round_trip() {
    for mode in [PrecisionMode::Fastest, PrecisionMode::Balanced, PrecisionMode::Highest] {
        assert_eq!(PrecisionMode::parse(mode.name()), Some(mode));
        assert_eq!(PrecisionMode::parse(&mode.name().to_uppercase()), Some(mode));
    }
    assert_eq!(PrecisionMode::parse(" Accurate "), Some(PrecisionMode::Highest));
    assert_eq!(PrecisionMode::parse("fast"), Some(PrecisionMode::Fastest));
    assert_eq!(PrecisionMode::parse("bogus"), None);
}

#[test]
fn every_mode_handles_odd_shapes_strides_and_empty_products() {
    let _guard = MODE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let naive = |a: &[f32], b: &[f32], m: usize, k: usize, n: usize| -> Vec<f64> {
        (0..m * n).map(|i| (0..k).map(|p| a[(i / n) * k + p] as f64 * b[p * n + i % n] as f64).sum()).collect()
    };
    let mut s = 5u64;
    for mode in [PrecisionMode::Fastest, PrecisionMode::Balanced, PrecisionMode::Highest] {
        set_precision_mode(mode);
        for &(m, k, n) in &[(1usize, 1usize, 1usize), (1, 7, 1), (5, 1, 5), (3, 0, 4), (0, 3, 4), (4, 3, 0), (17, 129, 33), (2, 700, 3), (65, 33, 1), (2, 70000, 1), (3, 100, 600), (8, 64, 513), (1, 1025, 40)] {
            let a: Vec<f32> = (0..m * k).map(|_| lcg(&mut s) as f32).collect();
            let b: Vec<f32> = (0..k * n).map(|_| lcg(&mut s) as f32).collect();
            let (ta, tb) = (Tensor::from_slice(&a, &[m, k]).unwrap(), Tensor::from_slice(&b, &[k, n]).unwrap());
            let Ok(prod) = ta.matmul(&tb) else { continue }; // an empty operand may be rejected, but must not panic
            let got = prod.contiguous().unwrap().to_vec::<f32>().unwrap();
            let want = naive(&a, &b, m, k, n);
            assert_eq!(got.len(), want.len(), "{mode:?} {m}x{k}x{n}");
            for (g, w) in got.iter().zip(&want) {
                assert!((*g as f64 - w).abs() <= 1e-4 * (1.0 + w.abs()), "{mode:?} {m}x{k}x{n}: {g} vs {w}");
            }
            // transposed operands take the strided route
            if m * k * n > 0 {
                let at = Tensor::from_slice(&(0..m * k).map(|i| a[(i % m) * k + i / m]).collect::<Vec<_>>(), &[k, m]).unwrap().t().unwrap();
                let bt = Tensor::from_slice(&(0..k * n).map(|i| b[(i % k) * n + i / k]).collect::<Vec<_>>(), &[n, k]).unwrap().t().unwrap();
                let got = at.matmul(&bt).unwrap().contiguous().unwrap().to_vec::<f32>().unwrap();
                for (g, w) in got.iter().zip(&want) {
                    assert!((*g as f64 - w).abs() <= 1e-4 * (1.0 + w.abs()), "{mode:?} strided {m}x{k}x{n}: {g} vs {w}");
                }
            }
        }
    }
    set_precision_mode(PrecisionMode::Balanced);
}

#[test]
fn highest_routes_for_big_weights_are_correctly_rounded() {
    let _guard = MODE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    set_precision_mode(PrecisionMode::Highest);
    let (k, n) = (2048usize, 2048usize); // a weight of 4M elements: the medium-row and column-strip routes
    let mut s = 21u64;
    let b: Vec<f32> = (0..k * n).map(|_| (lcg(&mut s) * 2.0) as f32).collect();
    let tb = Tensor::from_slice(&b, &[k, n]).unwrap();
    let tbt = Tensor::from_slice(&b, &[k, n]).unwrap().t().unwrap().contiguous().unwrap().t().unwrap(); // transposed layout, same values
    for m in [1usize, 20, 40, 100] {
        let a: Vec<f32> = (0..m * k).map(|_| (lcg(&mut s) * 2.0) as f32).collect();
        let ta = Tensor::from_slice(&a, &[m, k]).unwrap();
        for w in [&tb, &tbt] {
            let got = ta.matmul(w).unwrap().contiguous().unwrap().to_vec::<f32>().unwrap();
            for t in 0..40 {
                let (i, j) = ((t * 7) % m, (t * 389 + 5) % n);
                let (mut want, mut scale) = (0.0f64, 0.0f64);
                for p in 0..k {
                    let x = a[i * k + p] as f64 * b[p * n + j] as f64;
                    want += x;
                    scale += x.abs();
                }
                let e = (got[i * n + j] as f64 - want).abs() / scale / EPS;
                assert!(e < 0.55, "m={m} ({i},{j}): error {e} half-ulps of the sum of magnitudes");
            }
        }
    }
    set_precision_mode(PrecisionMode::Balanced);
}

/// The block-parallel chain loop of `Balanced` and `Fastest` (one parallel region over a grid of output blocks) gives, bit for bit, the
/// result of the serial loop (what a product inside a rayon worker takes): odd shapes, a skinny and a wide output, and a long `k` (the
/// f64 group combine).
#[test]
fn block_parallel_chains_are_bit_identical_to_serial() {
    let _guard = MODE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    for mode in [PrecisionMode::Balanced, PrecisionMode::Fastest] {
        set_precision_mode(mode);
        for (m, k, n) in [(130usize, 700usize, 270usize), (9, 2000, 517), (301, 400, 21), (200, 2500, 200), (1, 4096, 1024)] {
            let mut s = (m * 31 + k * 7 + n) as u64;
            let a: Vec<f32> = (0..m * k).map(|_| (lcg(&mut s) * 4.0) as f32).collect();
            let b: Vec<f32> = (0..k * n).map(|_| (lcg(&mut s) * 4.0) as f32).collect();
            let (ta, tb) = (Tensor::from_slice(&a, &[m, k]).unwrap(), Tensor::from_slice(&b, &[k, n]).unwrap());
            let parallel = ta.matmul(&tb).unwrap().contiguous().unwrap().to_vec::<f32>().unwrap();
            // inside a rayon worker the product takes the serial chain loop
            let (ta2, tb2) = (ta.clone(), tb.clone());
            let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
            let serial = pool.install(move || ta2.matmul(&tb2).unwrap().contiguous().unwrap().to_vec::<f32>().unwrap());
            assert_eq!(parallel, serial, "{mode:?} {m}x{k}x{n}");
        }
    }
    set_precision_mode(PrecisionMode::Balanced);
}

/// `Fastest` evaluates GELU with the cheaper five-term erfc; it must stay within a few float32 ulps of the exact value on the usual range
/// (and `Balanced` must stay at its tighter bound).
#[test]
fn fastest_gelu_stays_close_to_exact() {
    let _guard = MODE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let xs: Vec<f32> = (0..4001).map(|i| -4.0 + 8.0 * i as f32 / 4000.0).collect();
    let t = Tensor::from_slice(&xs, &[xs.len()]).unwrap();
    let erf = libm::erf;
    for mode in [PrecisionMode::Fastest, PrecisionMode::Balanced] {
        set_precision_mode(mode);
        let got = t.gelu().unwrap().to_vec::<f32>().unwrap();
        // the five-term erfc has an absolute error of about 1.5e-7 in the CDF, so the bound scales with |x|
        let worst = xs
            .iter()
            .zip(&got)
            .map(|(&x, &g)| {
                let want = x as f64 * 0.5 * (1.0 + erf(x as f64 / std::f64::consts::SQRT_2));
                (g as f64 - want).abs() / (1e-6 * (1.0 + x.abs() as f64))
            })
            .fold(0.0, f64::max);
        assert!(worst < 1.0, "{mode:?}: gelu error is {worst} times the bound");
        // infinities and NaN behave as PyTorch's: gelu(-inf) is NaN (-inf * 0), gelu(+inf) is +inf
        let special = Tensor::from_slice(&[f32::NEG_INFINITY, f32::INFINITY, f32::NAN], &[3]).unwrap().gelu().unwrap().to_vec::<f32>().unwrap();
        assert!(special[0].is_nan() && special[1] == f32::INFINITY && special[2].is_nan(), "{mode:?}: {special:?}");
    }
    set_precision_mode(PrecisionMode::Balanced);
}

/// `Fastest` stays plain f32 for any depth (no f64 group combine past 32 slices) and `Balanced` keeps its f64 combine: on a long `k` the
/// Balanced result is closer to the exact one, and both stay bit-identical between the block grid and the serial loop.
#[test]
fn long_k_modes_keep_their_accumulation_kind() {
    let _guard = MODE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (m, k, n) = (32usize, 20_000usize, 40usize); // 79 slices of 256, 417 chains of 48: both past the 32-chain group
    let mut s = 5u64;
    let a: Vec<f32> = (0..m * k).map(|_| (lcg(&mut s) * 4.0) as f32).collect();
    let b: Vec<f32> = (0..k * n).map(|_| (lcg(&mut s) * 4.0) as f32).collect();
    let (mut want, mut scale) = (vec![0.0f64; m * n], vec![0.0f64; m * n]);
    for i in 0..m {
        for p in 0..k {
            let x = a[i * k + p] as f64;
            for j in 0..n {
                let t = x * b[p * n + j] as f64;
                want[i * n + j] += t;
                scale[i * n + j] += t.abs();
            }
        }
    }
    let (ta, tb) = (Tensor::from_slice(&a, &[m, k]).unwrap(), Tensor::from_slice(&b, &[k, n]).unwrap());
    let worst = |mode: PrecisionMode| {
        set_precision_mode(mode);
        let got = ta.matmul(&tb).unwrap().contiguous().unwrap().to_vec::<f32>().unwrap();
        (0..m * n).map(|i| (got[i] as f64 - want[i]).abs() / scale[i] / EPS).fold(0.0, f64::max)
    };
    let (fast, balanced) = (worst(PrecisionMode::Fastest), worst(PrecisionMode::Balanced));
    set_precision_mode(PrecisionMode::Balanced);
    assert!(balanced < fast, "Balanced ({balanced}) should beat Fastest ({fast}) on a long k");
    assert!(fast < 8.0 && balanced < 2.0, "fast {fast}, balanced {balanced}");
}

/// Helper of [`results_do_not_depend_on_the_thread_count`]: prints a hash of the products of several shapes in every mode.
/// It does nothing unless the parent test started it.
#[test]
fn thread_count_hash_child() {
    if std::env::var("OXIDEON_HASH_CHILD").is_err() {
        return;
    }
    let _guard = MODE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut hash = 0xcbf29ce484222325u64;
    for mode in [PrecisionMode::Balanced, PrecisionMode::Fastest, PrecisionMode::Highest] {
        set_precision_mode(mode);
        for (m, k, n) in [(130usize, 700usize, 270usize), (9, 2000, 517), (301, 400, 21), (200, 2500, 200), (1, 4096, 1024)] {
            let mut s = (m * 31 + k * 7 + n) as u64;
            let a: Vec<f32> = (0..m * k).map(|_| (lcg(&mut s) * 4.0) as f32).collect();
            let b: Vec<f32> = (0..k * n).map(|_| (lcg(&mut s) * 4.0) as f32).collect();
            let (ta, tb) = (Tensor::from_slice(&a, &[m, k]).unwrap(), Tensor::from_slice(&b, &[k, n]).unwrap());
            for v in ta.matmul(&tb).unwrap().contiguous().unwrap().to_vec::<f32>().unwrap() {
                hash = (hash ^ u64::from(v.to_bits())).wrapping_mul(0x100000001b3);
            }
        }
    }
    println!("HASH={hash:016x}");
}

/// The block grid of the matmul splits the output differently for every thread count; the product must not notice.
#[test]
fn results_do_not_depend_on_the_thread_count() {
    let exe = std::env::current_exe().unwrap();
    let hash_with = |threads: usize| {
        let out = std::process::Command::new(&exe)
            .args(["thread_count_hash_child", "--exact", "--nocapture"])
            .env("OXIDEON_HASH_CHILD", "1")
            .env("RAYON_NUM_THREADS", threads.to_string())
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout).into_owned();
        assert!(out.status.success(), "child with {threads} threads failed: {text}");
        text.lines().find_map(|l| l.strip_prefix("HASH=")).unwrap_or_else(|| panic!("no hash from {threads} threads: {text}")).to_string()
    };
    let want = hash_with(1);
    for threads in [2, 3, 5, 8, 16] {
        assert_eq!(hash_with(threads), want, "{threads} threads");
    }
}
