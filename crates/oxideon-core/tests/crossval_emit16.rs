//! Cross-validation batch 16: multiple testing, power, causal estimators, clustering, feature selection, optimal
//! transport, numerics, root finding / optimization / quadrature, computational geometry
//! (compared by tools/crossval/check_batch16.py).
use oxideon_core::causal::*;
use oxideon_core::decomp::*;
use oxideon_core::ml_extras::*;
use oxideon_core::numerics::*;
use oxideon_core::scientific::*;
use oxideon_core::spatial::*;
use oxideon_core::text_ir::soundex;

fn u(i: u64) -> f64 {
    // splitmix64: decorrelated across offsets (a multiplicative Weyl sequence is not)
    let mut z = i.wrapping_add(0x9E37_79B9_7F4A_7C15).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 30)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}
fn g(i: u64) -> f64 {
    (-2.0 * u(2 * i + 1).ln()).sqrt() * (2.0 * std::f64::consts::PI * u(2 * i + 2)).cos()
}
fn labels(l: &[usize]) -> String {
    l.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")
}

#[test]
#[ignore]
fn emit() {
    // --- multiple testing
    let p: Vec<f64> = (0..25).map(|i| u(i + 3).powi(3)).collect();
    for (name, v) in [("bonf", bonferroni(&p).unwrap()), ("holm", holm(&p).unwrap()), ("bh", benjamini_hochberg(&p).unwrap())] {
        v.iter().enumerate().for_each(|(i, x)| println!("{name}_{i} {x:.15e}"));
    }
    p.iter().enumerate().for_each(|(i, x)| println!("p_{i} {x:.17e}"));

    // --- norm_ppf
    for (i, q) in [1e-12, 1e-8, 1e-5, 0.001, 0.02, 0.02425, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98, 0.999, 1.0 - 1e-9].iter().enumerate() {
        println!("ppf_{i} {q:.17e} {:.15e}", norm_ppf(*q));
    }

    // --- two-proportion z, sample size, power
    for (i, (x1, n1, x2, n2)) in [(120u64, 1000u64, 150u64, 1000u64), (30, 200, 55, 220), (5, 50, 9, 60), (480, 5000, 600, 5000)].iter().enumerate() {
        let r = two_proportion_ztest(*x1, *n1, *x2, *n2).unwrap();
        println!("zt_{i} {:.15e} {:.15e}", r.statistic, r.p_value);
    }
    for (i, (d, sd, a, pw)) in [(0.5, 1.0, 0.05, 0.8), (2.0, 10.0, 0.01, 0.9), (0.1, 1.0, 0.05, 0.95)].iter().enumerate() {
        println!("nm_{i} {}", sample_size_means(*d, *sd, *a, *pw).unwrap());
    }
    for (i, (p1, p2, a, pw)) in [(0.1, 0.12, 0.05, 0.8), (0.5, 0.55, 0.01, 0.9), (0.02, 0.03, 0.05, 0.8)].iter().enumerate() {
        println!("np_{i} {}", sample_size_proportions(*p1, *p2, *a, *pw).unwrap());
    }
    for (i, (d, sd, n, a)) in [(0.5, 1.0, 64u64, 0.05), (0.2, 1.0, 100, 0.05), (1.0, 2.0, 30, 0.01)].iter().enumerate() {
        println!("pw_{i} {:.15e}", power_means(*d, *sd, *n, *a));
    }

    // --- IPW and 2SLS on synthetic data
    let n = 400u64;
    let x: Vec<Vec<f64>> = (0..n).map(|i| vec![g(i), g(10_000 + i)]).collect();
    let treated: Vec<bool> = x.iter().enumerate().map(|(i, r)| u(500 + i as u64) < 1.0 / (1.0 + (-(0.8 * r[0] - 0.5 * r[1])).exp())).collect();
    let out: Vec<f64> = x.iter().zip(&treated).enumerate().map(|(i, (r, &t))| 2.0 * f64::from(u8::from(t)) + r[0] + 0.5 * r[1] + 0.3 * g(20_000 + i as u64)).collect();
    println!("ipw {:.12}", ipw_ate(&x, &treated, &out, 0.01).unwrap());
    for (i, r) in x.iter().enumerate() {
        println!("ipx_{i} {:.17e} {:.17e} {} {:.17e}", r[0], r[1], u8::from(treated[i]), out[i]);
    }
    let z: Vec<Vec<f64>> = (0..n).map(|i| vec![g(30_000 + i), g(31_000 + i)]).collect();
    let ctrl: Vec<Vec<f64>> = (0..n).map(|i| vec![g(32_000 + i)]).collect();
    let conf: Vec<f64> = (0..n).map(|i| g(33_000 + i)).collect();
    let endog: Vec<Vec<f64>> = (0..n as usize).map(|i| vec![0.7 * z[i][0] + 0.4 * z[i][1] + 0.5 * conf[i] + 0.2 * ctrl[i][0] + 0.3 * g(34_000 + i as u64)]).collect();
    let y: Vec<f64> = (0..n as usize).map(|i| 1.0 + 2.0 * endog[i][0] + 0.5 * ctrl[i][0] + 0.8 * conf[i] + 0.2 * g(35_000 + i as u64)).collect();
    let beta = two_sls(&y, &endog, &ctrl, &z).unwrap();
    beta.iter().enumerate().for_each(|(i, b)| println!("2sls_{i} {b:.12}"));
    for i in 0..n as usize {
        println!("iv_{i} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e}", y[i], endog[i][0], ctrl[i][0], z[i][0], z[i][1], 0.0);
    }
    println!("did {:.12}", diff_in_diff(&[1.0, 2.0, 3.5], &[4.0, 5.0], &[0.5, 1.5], &[1.0, 2.0, 2.5]).unwrap());

    // --- clustering on 3 gaussian blobs
    let centers = [(0.0, 0.0), (6.0, 0.0), (3.0, 5.0)];
    let pts: Vec<Vec<f64>> = (0..90u64).map(|i| { let c = centers[(i % 3) as usize]; vec![c.0 + 0.7 * g(40_000 + i), c.1 + 0.7 * g(41_000 + i)] }).collect();
    pts.iter().enumerate().for_each(|(i, r)| println!("pt_{i} {:.17e} {:.17e}", r[0], r[1]));
    for (name, l) in [("single", Linkage::Single), ("complete", Linkage::Complete), ("average", Linkage::Average), ("ward", Linkage::Ward)] {
        println!("agg_{name} {}", labels(&agglomerative(&pts, 3, l).unwrap()));
    }
    println!("spectral {}", labels(&spectral_clustering(&pts, 3, 0.5, 1).unwrap()));
    println!("meanshift {}", labels(&mean_shift(&pts, 2.5, 200).unwrap().0));
    // NMF: non-negative data of true rank 3
    let w: Vec<Vec<f64>> = (0..30u64).map(|i| (0..3).map(|k| u(50_000 + i * 3 + k)).collect()).collect();
    let h: Vec<Vec<f64>> = (0..3u64).map(|k| (0..12).map(|j| u(60_000 + k * 12 + j)).collect()).collect();
    let xm: Vec<Vec<f64>> = (0..30).map(|i| (0..12).map(|j| (0..3).map(|k| w[i][k] * h[k][j]).sum()).collect()).collect();
    xm.iter().enumerate().for_each(|(i, r)| println!("nmfx_{i} {}", r.iter().map(|v| format!("{v:.17e}")).collect::<Vec<_>>().join(" ")));
    let (_, _, err) = nmf(&xm, 3, 800, 1).unwrap();
    println!("nmf_err {err:.9e}");
    let (_, _, err2) = nmf(&xm, 1, 800, 1).unwrap();
    println!("nmf_err_rank1 {err2:.9e}");

    // --- feature selection
    let xf: Vec<Vec<f64>> = (0..120u64).map(|i| vec![g(70_000 + i) + (i % 3) as f64 * 0.8, g(71_000 + i), g(72_000 + i) + (i % 3) as f64 * 0.2]).collect();
    let yf: Vec<usize> = (0..120).map(|i| i % 3).collect();
    xf.iter().enumerate().for_each(|(i, r)| println!("fx_{i} {:.17e} {:.17e} {:.17e}", r[0], r[1], r[2]));
    f_classif(&xf, &yf).unwrap().iter().enumerate().for_each(|(j, v)| println!("fc_{j} {v:.12e}"));
    mutual_info_binned(&xf, &yf, 8).unwrap().iter().enumerate().for_each(|(j, v)| println!("mi_{j} {v:.12e}"));
    println!("vt {}", labels(&variance_threshold(&xf, 1.0)));

    // --- Sinkhorn vs exact optimal transport (LP via scipy) on a 5x6 problem
    let a = [0.1, 0.3, 0.2, 0.25, 0.15];
    let b = [0.2, 0.1, 0.25, 0.15, 0.2, 0.1];
    let cost: Vec<Vec<f64>> = (0..5).map(|i| (0..6).map(|j| (i as f64 * 0.7 - j as f64 * 0.5).powi(2) + 0.1 * u(80_000 + i * 6 + j)).collect()).collect();
    for (i, row) in cost.iter().enumerate() {
        println!("cost_{i} {}", row.iter().map(|v| format!("{v:.17e}")).collect::<Vec<_>>().join(" "));
    }
    for eps in [0.5, 0.05, 0.01] {
        let (plan, c) = sinkhorn(&a, &b, &cost, eps, 20_000).unwrap();
        println!("sink_{eps}_cost {c:.12}");
        let row_err = (0..5).map(|i| (plan[i].iter().sum::<f64>() - a[i]).abs()).fold(0.0, f64::max);
        let col_err = (0..6).map(|j| (plan.iter().map(|r| r[j]).sum::<f64>() - b[j]).abs()).fold(0.0, f64::max);
        println!("sink_{eps}_marg {:.3e}", row_err.max(col_err));
    }

    // --- numerics
    println!("lagr {:.15e}", lagrange_eval(&[0.0, 1.0, 2.0, 3.5], &[1.0, 3.0, 2.0, 7.0], 1.7).unwrap());
    println!("cheb {}", chebyshev_nodes(6, -2.0, 3.0).iter().map(|v| format!("{v:.15e}")).collect::<Vec<_>>().join(" "));
    println!("fd1 {:.12e}", derivative(|x: f64| x.sin() * x.exp(), 0.8, 1e-5));
    println!("rich {:.12e}", richardson_derivative(|x: f64| x.sin() * x.exp(), 0.8, 0.1, 5));
    let gf = gradient_fd(|v: &[f64]| v[0].powi(3) * v[1] + (v[1] * v[2]).sin(), &[1.1, 0.7, -0.4], 1e-5);
    println!("grad {}", gf.iter().map(|v| format!("{v:.10e}")).collect::<Vec<_>>().join(" "));
    let jf = jacobian_fd(|v: &[f64]| vec![v[0] * v[1], v[0].sin() + v[1] * v[1], v[0].exp()], &[0.9, -0.6], 1e-6);
    println!("jac {}", jf.iter().flatten().map(|v| format!("{v:.8e}")).collect::<Vec<_>>().join(" "));
    let hf = hessian_fd(|v: &[f64]| v[0].powi(3) * v[1] + (v[1] * v[2]).sin(), &[1.1, 0.7, -0.4], 1e-4);
    println!("hess {}", hf.iter().flatten().map(|v| format!("{v:.6e}")).collect::<Vec<_>>().join(" "));

    // --- root finding / minimization / quadrature
    println!("bisect {:.12}", bisect(|x| x.powi(3) - 2.0 * x - 5.0, 2.0, 3.0, 1e-12).unwrap());
    println!("brentq {:.12}", brentq(|x| x.cos() - x, 0.0, 1.0, 1e-12).unwrap());
    println!("newton {:.12}", newton(|x| x * x - 612.0, |x| 2.0 * x, 10.0, 1e-12).unwrap());
    let (xm1, fm1) = brent_min(|x| (x - 2.0).powi(2) + (3.0 * x).sin(), 1.0, 3.0, 1e-9);
    println!("brentmin {xm1:.8} {fm1:.10}");
    let rosen = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
    let (xn, fnm) = nelder_mead(rosen, &[-1.2, 1.0], 0.5, 1e-10, 2000);
    println!("nm {:.5} {:.5} {fnm:.3e}", xn[0], xn[1]);
    let (xb, fb) = bfgs(rosen, None::<fn(&[f64]) -> Vec<f64>>, &[-1.2, 1.0], 1e-8, 500);
    println!("bfgs {:.5} {:.5} {fb:.3e}", xb[0], xb[1]);
    let (qv, _) = quad(|x| (-x * x).exp() * x.cos(), -2.0, 3.0, 1e-10).unwrap();
    println!("quad {qv:.12}");
    println!("simpson {:.12}", simpson(|x| x.sin().powi(2) / (1.0 + x), 0.0, 4.0, 200).unwrap());
    println!("romberg {:.12}", romberg(|x| x.sin().powi(2) / (1.0 + x), 0.0, 4.0, 8));
    let xs: Vec<f64> = (0..30).map(|i| i as f64 * 0.2).collect();
    let ys: Vec<f64> = xs.iter().enumerate().map(|(i, x)| 2.5 * (-0.7 * x).exp() + 0.3 + 0.01 * g(90_000 + i as u64)).collect();
    xs.iter().zip(&ys).enumerate().for_each(|(i, (x, y))| println!("cf_{i} {x:.17e} {y:.17e}"));
    let (pf, _) = curve_fit(|x, p| p[0] * (-p[1] * x).exp() + p[2], &xs, &ys, &[1.0, 1.0, 0.0], 200).unwrap();
    println!("curvefit {}", pf.iter().map(|v| format!("{v:.8}")).collect::<Vec<_>>().join(" "));

    // --- geometry
    let gp: Vec<(f64, f64)> = (0..60u64).map(|i| (g(100_000 + i) * 3.0, g(101_000 + i) * 2.0)).collect();
    gp.iter().enumerate().for_each(|(i, p)| println!("gp_{i} {:.17e} {:.17e}", p.0, p.1));
    let hull = convex_hull(&gp);
    println!("hull_n {}", hull.len());
    println!("hull_area {:.12}", polygon_area(&hull));
    println!("signed_area_poly {:.12}", signed_area(&[(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (2.0, 5.0), (0.0, 3.0)]));
    let poly = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (2.5, 2.0), (0.0, 5.0)];
    let inside: Vec<u8> = (0..40u64).map(|i| u8::from(point_in_polygon((u(110_000 + i) * 7.0 - 1.0, u(111_000 + i) * 7.0 - 1.0), &poly))).collect();
    println!("pip {}", inside.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(","));
    let ((ci, cj), cd) = closest_pair(&gp).unwrap();
    println!("closest {} {} {cd:.12e}", ci.min(cj), ci.max(cj));

    // --- soundex (canonical vectors from the Soundex specification / Wikipedia)
    for w in ["Robert", "Rupert", "Rubin", "Ashcraft", "Ashcroft", "Tymczak", "Pfister", "Honeyman", "Lee", "Gutierrez", "Jackson", "VanDeusen", "Ekzampul"] {
        println!("sdx_{w} {}", soundex(w));
    }
}
