//! Twelfth cross-validation batch: the modules previously covered only by unit tests (GP, KDE, survival, HMM, Kalman,
//! AR / Ljung-Box, image processing, preprocessing, number theory, algebra, DataFrame, model selection, QMC).
//! Run: `cargo test -p oxideon-core --test crossval_emit12 -- --nocapture --ignored`.

use oxideon_core::{algebra, classical::*, dataio, frame, gp, imageproc, inference, model_selection, numtheory, preprocessing, qmc, survival, timeseries};

fn p(n: &str, v: f64) {
    println!("{n} {v:.12e}");
}

#[test]
#[ignore]
fn emit_gp_kde_ts() {
    // Gaussian process (RBF and Matern-5/2) posterior at query points
    let x: Matrix = (0..12).map(|i| vec![i as f64 * 0.5, ((i * 7) % 5) as f64 * 0.3]).collect();
    let y: Vec<f64> = x.iter().map(|r| (r[0] * 0.8).sin() + 0.3 * r[1]).collect();
    for (name, kern) in [("rbf", gp::GpKernel::Rbf), ("m52", gp::GpKernel::Matern52)] {
        let g = gp::GaussianProcess::fit(&x, &y, kern, 1.3, 1.0, 1e-3).unwrap();
        for (i, q) in [vec![0.7, 0.2], vec![2.9, 0.9], vec![5.0, 0.0], vec![8.0, 1.0]].iter().enumerate() {
            let (m, v) = g.predict(q);
            p(&format!("gp_{name}_mean_{i}"), m);
            p(&format!("gp_{name}_var_{i}"), v);
        }
        p(&format!("gp_{name}_lml"), g.log_marginal_likelihood());
    }
    // KDE
    let data: Vec<f64> = (0..40).map(|i| (i as f64 * 0.7).sin() * 2.0 + (i % 5) as f64 * 0.3).collect();
    let kde = oxideon_core::glm::Kde::fit(&data, None).unwrap();
    p("kde_bw", kde.bandwidth);
    for (i, q) in [-2.5, -1.0, 0.0, 0.7, 2.2, 4.0].iter().enumerate() {
        p(&format!("kde_{i}"), kde.density(*q));
    }
    // time series
    let s: Vec<f64> = (0..80).map(|i| (i as f64 * 0.3).sin() + 0.4 * ((i * i) as f64 * 0.17).cos() + 0.02 * i as f64).collect();
    let ar = timeseries::ArModel::fit(&s, 3).unwrap();
    ar.phi.iter().enumerate().for_each(|(i, v)| p(&format!("ar_phi{i}"), *v));
    p("ar_sigma2", ar.sigma2);
    p("ar_mean", ar.mean);
    p("ljung_box", timeseries::ljung_box(&s, 8).unwrap());
    timeseries::diff(&s, 2).iter().take(10).enumerate().for_each(|(i, v)| p(&format!("diff2_{i}"), *v));
    timeseries::ewma(&s, 0.3).unwrap().iter().take(10).enumerate().for_each(|(i, v)| p(&format!("ewma_{i}"), *v));
    timeseries::rolling_mean(&s, 5).iter().skip(4).take(10).enumerate().for_each(|(i, v)| p(&format!("rmean_{i}"), *v));
    timeseries::rolling_std(&s, 5).iter().skip(4).take(10).enumerate().for_each(|(i, v)| p(&format!("rstd_{i}"), *v));
    // seasonal decomposition (period 6)
    let d = timeseries::seasonal_decompose(&s, 6).unwrap();
    d.trend.iter().enumerate().filter(|(_, v)| !v.is_nan()).take(12).for_each(|(i, v)| p(&format!("trend_{i}"), *v));
    d.seasonal.iter().take(6).enumerate().for_each(|(i, v)| p(&format!("seasonal_{i}"), *v));
    // QMC discrepancy
    p("l2star_sobol", qmc::l2_star_discrepancy(&qmc::sobol(64, 3).unwrap()).unwrap());
    let pts = qmc::halton(50, 2).unwrap();
    p("l2star_halton", qmc::l2_star_discrepancy(&pts).unwrap());
    for (i, pt) in pts.iter().take(6).enumerate() {
        p(&format!("halton_{i}_0"), pt[0]);
        p(&format!("halton_{i}_1"), pt[1]);
    }
}

#[test]
#[ignore]
fn emit_survival_hmm_kalman() {
    // survival
    let time = [5.0, 6.0, 6.0, 2.5, 4.0, 4.0, 8.0, 9.0, 12.0, 12.0, 3.0, 7.5, 10.0, 1.0, 11.0];
    let event = [true, false, true, true, true, false, true, true, false, true, true, false, true, true, false];
    let km = survival::kaplan_meier(&time, &event).unwrap();
    for t in [0.5, 1.0, 2.5, 4.0, 5.0, 6.0, 8.0, 9.0, 12.0, 20.0] {
        p(&format!("km_{t}"), km.at(t));
    }
    let na = survival::nelson_aalen(&time, &event).unwrap();
    for t in [1.0, 4.0, 6.0, 9.0, 12.0] {
        p(&format!("na_{t}"), na.at(t));
    }
    let group: Vec<bool> = (0..15).map(|i| i % 2 == 0).collect();
    let (stat, pv) = survival::logrank_test(&time, &event, &group).unwrap();
    p("logrank_stat", stat);
    p("logrank_p", pv);
    let x: Vec<Vec<f64>> = (0..15).map(|i| vec![(i as f64 * 0.6).sin(), (i % 4) as f64 * 0.5]).collect();
    let cox = survival::CoxPh::fit(&x, &time, &event, 0.0).unwrap();
    cox.coef.iter().enumerate().for_each(|(i, v)| p(&format!("cox_coef{i}"), *v));
    cox.se.iter().enumerate().for_each(|(i, v)| p(&format!("cox_se{i}"), *v));
    p("cox_ll", cox.log_likelihood);
    let risk = cox.hazard_ratio(&x);
    p("cindex", survival::concordance_index(&time, &event, &risk).unwrap());
    // HMM: forward-backward posteriors, log-likelihood, Viterbi path
    let hmm = inference::Hmm { start: vec![0.6, 0.3, 0.1], trans: vec![vec![0.7, 0.2, 0.1], vec![0.2, 0.6, 0.2], vec![0.1, 0.3, 0.6]], emit: vec![vec![0.5, 0.4, 0.1], vec![0.1, 0.3, 0.6], vec![0.3, 0.3, 0.4]] };
    let obs = [0usize, 1, 2, 2, 1, 0, 0, 2, 1, 1];
    let (gamma, ll) = hmm.forward_backward(&obs).unwrap();
    p("hmm_ll", ll);
    gamma.iter().enumerate().for_each(|(t, g)| g.iter().enumerate().for_each(|(i, v)| p(&format!("hmm_gamma_{t}_{i}"), *v)));
    println!("hmm_viterbi {}", hmm.viterbi(&obs).unwrap().iter().map(usize::to_string).collect::<Vec<_>>().join(","));
    // Kalman filter + RTS smoother (constant-velocity model)
    let kf = inference::KalmanFilter { f: vec![vec![1.0, 1.0], vec![0.0, 1.0]], h: vec![vec![1.0, 0.0]], q: vec![vec![0.01, 0.0], vec![0.0, 0.02]], r: vec![vec![0.5]] };
    let ys: Vec<Vec<f64>> = (0..12).map(|i| vec![0.4 * i as f64 + 0.3 * (i as f64 * 1.7).sin()]).collect();
    let out = kf.filter(&ys, &[0.0, 0.0], &vec![vec![1.0, 0.0], vec![0.0, 1.0]]).unwrap();
    out.means.iter().enumerate().for_each(|(t, m)| m.iter().enumerate().for_each(|(i, v)| p(&format!("kf_mean_{t}_{i}"), *v)));
    out.covs.iter().enumerate().for_each(|(t, c)| p(&format!("kf_var_{t}"), c[0][0]));
    p("kf_ll", out.log_likelihood);
    let sm = kf.smooth(&out).unwrap();
    sm.means.iter().enumerate().for_each(|(t, m)| m.iter().enumerate().for_each(|(i, v)| p(&format!("rts_mean_{t}_{i}"), *v)));
}

#[test]
#[ignore]
fn emit_misc() {
    // image processing vs scipy.ndimage
    let img = imageproc::Image::from_fn(9, 7, |x, y| ((x * 3 + y * 5) % 7) as f64 / 7.0 + 0.1 * ((x * y) % 3) as f64);
    imageproc::gaussian_blur(&img, 1.2).unwrap().data.iter().enumerate().for_each(|(i, v)| p(&format!("gblur_{i}"), *v));
    let (gx, gy) = imageproc::sobel(&img).unwrap();
    gx.data.iter().enumerate().for_each(|(i, v)| p(&format!("sobel_x_{i}"), *v));
    gy.data.iter().enumerate().for_each(|(i, v)| p(&format!("sobel_y_{i}"), *v));
    let bin = imageproc::Image::from_fn(10, 8, |x, y| f64::from(u8::from((x / 3 + y / 3) % 2 == 0 && x != 4)));
    let (labels, n) = imageproc::connected_components(&bin, false);
    p("cc4_n", n as f64);
    println!("cc4 {}", labels.iter().map(usize::to_string).collect::<Vec<_>>().join(","));
    let (_, n8) = imageproc::connected_components(&bin, true);
    p("cc8_n", n8 as f64);
    imageproc::erode(&img, 1).data.iter().enumerate().for_each(|(i, v)| p(&format!("erode_{i}"), *v));
    imageproc::dilate(&img, 1).data.iter().enumerate().for_each(|(i, v)| p(&format!("dilate_{i}"), *v));
    let dt = imageproc::distance_transform(&bin);
    dt.data.iter().enumerate().for_each(|(i, v)| p(&format!("dt_{i}"), *v));
    let t = imageproc::integral_image(&img);
    p("box_sum", imageproc::box_sum(&t, 2, 1, 7, 5));
    p("otsu", imageproc::otsu_threshold(&img));
    imageproc::resize(&img, 5, 4).unwrap().data.iter().enumerate().for_each(|(i, v)| p(&format!("resize_{i}"), *v));
    // preprocessing vs scikit-learn
    let m: Matrix = (0..12).map(|i| vec![(i as f64 * 0.7).sin() * 3.0 + i as f64, ((i * 5) % 7) as f64, (i as f64).powi(2) / 10.0]).collect();
    let rs = preprocessing::RobustScaler::fit(&m).unwrap();
    rs.transform(&m).iter().flatten().enumerate().for_each(|(i, v)| p(&format!("robust_{i}"), *v));
    let kb = preprocessing::KBinsDiscretizer::fit(&m, 4, true).unwrap();
    kb.transform(&m).iter().flatten().enumerate().for_each(|(i, v)| p(&format!("kbins_q_{i}"), *v));
    let kbu = preprocessing::KBinsDiscretizer::fit(&m, 3, false).unwrap();
    kbu.transform(&m).iter().flatten().enumerate().for_each(|(i, v)| p(&format!("kbins_u_{i}"), *v));
    let small: Matrix = vec![vec![1.0, 2.0], vec![3.0, -1.0], vec![0.5, 4.0]];
    let pf = preprocessing::PolynomialFeatures { degree: 3, include_bias: true };
    pf.transform(&small).iter().flatten().enumerate().for_each(|(i, v)| p(&format!("poly_{i}"), *v));
    preprocessing::normalize(&small, 2.0).unwrap().iter().flatten().enumerate().for_each(|(i, v)| p(&format!("norm2_{i}"), *v));
    // number theory vs sympy
    for n in [2u64, 97, 561, 7919, 1_000_000_007, 3_215_031_751] {
        p(&format!("isprime_{n}"), f64::from(u8::from(numtheory::is_prime(n))));
    }
    for n in [360u64, 9_699_690, 600_851_475_143, 1_000_000_016_000_000_063] {
        println!("factor_{n} {}", numtheory::factorize(n).unwrap().iter().map(u64::to_string).collect::<Vec<_>>().join(","));
    }
    for n in [36u64, 97, 1001, 600_851_475_143] {
        p(&format!("totient_{n}"), numtheory::totient(n).unwrap() as f64);
    }
    p("modinv", numtheory::mod_inverse(17, 3120).unwrap() as f64);
    let (cx, cm) = numtheory::crt(&[2, 3, 2, 10], &[3, 5, 7, 11]).unwrap();
    p("crt_x", cx as f64);
    p("crt_m", cm as f64);
    p("binom", numtheory::binomial(60, 25).unwrap() as f64);
    p("stirling2", numtheory::stirling2(14, 5).unwrap() as f64);
    p("partitions", numtheory::partition_count(150).unwrap() as f64);
    p("catalan", numtheory::catalan(20).unwrap() as f64);
    p("modpow", numtheory::mod_pow(7, 1_000_003, 1_000_000_007).unwrap() as f64);
    // algebra vs sympy / fractions
    let a = algebra::Rational::new(355, 113).unwrap() * algebra::Rational::new(-7, 22).unwrap() + algebra::Rational::new(5, 6).unwrap();
    println!("rational {}", a);
    println!("approx {}", algebra::Rational::approximate(std::f64::consts::E, 500).unwrap());
    let q = algebra::Polynomial::new(vec![1.0, -3.0, 0.0, 2.0]);
    let r = algebra::Polynomial::new(vec![-2.0, 1.0, 5.0]);
    (&q * &r).c.iter().enumerate().for_each(|(i, v)| p(&format!("pmul_{i}"), *v));
    q.compose(&r).c.iter().enumerate().for_each(|(i, v)| p(&format!("pcomp_{i}"), *v));
    q.derivative().c.iter().enumerate().for_each(|(i, v)| p(&format!("pder_{i}"), *v));
    q.integral(0.0).c.iter().enumerate().for_each(|(i, v)| p(&format!("pint_{i}"), *v));
    // DataFrame vs pandas
    let t = dataio::parse_csv("region,product,units,price\nwest,a,10,2.5\neast,a,4,2.5\nwest,b,7,4.0\neast,b,3,4.0\nwest,a,3,2.5\nnorth,c,8,9.0\n", ',', true).unwrap();
    let df = frame::DataFrame::from_table(&t).unwrap();
    for (name, agg) in [("sum", frame::Agg::Sum), ("mean", frame::Agg::Mean), ("min", frame::Agg::Min), ("max", frame::Agg::Max), ("count", frame::Agg::Count)] {
        df.group_by("region", "units", agg).unwrap().num("units").unwrap().iter().enumerate().for_each(|(i, v)| p(&format!("groupby_{name}_{i}"), *v));
    }
    df.sort_by("units", true).unwrap().num("units").unwrap().iter().enumerate().for_each(|(i, v)| p(&format!("sort_{i}"), *v));
    for (c, row) in df.describe().iter().enumerate() {
        row.1.iter().enumerate().for_each(|(i, v)| p(&format!("describe_{c}_{i}"), *v));
    }
    // model selection partitions: fold sizes must agree with scikit-learn's contiguous scheme
    let labels: Vec<usize> = (0..23).map(|i| usize::from(i % 3 == 0)).collect();
    for (fi, f) in model_selection::stratified_kfold(&labels, 4, false, 0).unwrap().iter().enumerate() {
        p(&format!("skf_size_{fi}"), f.test.len() as f64);
        p(&format!("skf_pos_{fi}"), f.test.iter().filter(|&&i| labels[i] == 1).count() as f64);
    }
    for (fi, f) in model_selection::time_series_split(20, 4).unwrap().iter().enumerate() {
        p(&format!("tss_train_{fi}"), f.train.len() as f64);
        p(&format!("tss_test_{fi}"), f.test.len() as f64);
        p(&format!("tss_first_{fi}"), f.test[0] as f64);
    }
}
