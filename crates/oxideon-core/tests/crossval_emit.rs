//! Emits values computed by the pure-Rust statistics / metrics modules for a fixed set of inputs, one `name value`
//! per line, so `tools/crossval/check_against_scipy.py` can compare them with SciPy / scikit-learn.
//! Run: `cargo test -p oxideon-core --test crossval_emit -- --nocapture --ignored`.

use oxideon_core::metrics_more::*;
use oxideon_core::stats_tests::*;

#[test]
#[ignore]
fn emit() {
    let a = [5.1, 4.9, 5.6, 5.8, 6.0, 5.2, 5.5, 6.3, 4.7, 5.9];
    let b = [6.2, 6.8, 5.9, 7.1, 6.5, 6.0, 7.3, 6.7];
    let p = |n: &str, v: f64| println!("{n} {v:.12e}");
    let r = ttest_1samp(&a, 5.0).unwrap();
    p("ttest_1samp.stat", r.statistic);
    p("ttest_1samp.p", r.p_value);
    let r = ttest_ind(&a, &b).unwrap();
    p("ttest_ind.stat", r.statistic);
    p("ttest_ind.p", r.p_value);
    let r = mannwhitneyu(&a, &b).unwrap();
    p("mannwhitneyu.U", r.statistic);
    p("mannwhitneyu.p", r.p_value);
    let r = ks_2samp(&a, &b).unwrap();
    p("ks_2samp.stat", r.statistic);
    let x: Vec<f64> = (1..=12).map(f64::from).collect();
    let y = [2.0, 1.0, 4.0, 3.0, 7.0, 5.0, 6.0, 9.0, 8.0, 12.0, 10.0, 11.0];
    let r = pearsonr(&x, &y).unwrap();
    p("pearsonr.r", r.statistic);
    p("pearsonr.p", r.p_value);
    let r = spearmanr(&x, &y).unwrap();
    p("spearmanr.r", r.statistic);
    p("spearmanr.p", r.p_value);
    let r = kendalltau(&x, &y).unwrap();
    p("kendalltau.tau", r.statistic);
    let r = f_oneway(&[&a[..5], &a[5..], &b]).unwrap();
    p("f_oneway.F", r.statistic);
    p("f_oneway.p", r.p_value);
    let r = chi2_independence(&[vec![12.0, 5.0, 9.0], vec![7.0, 14.0, 6.0]]).unwrap();
    p("chi2_indep.stat", r.statistic);
    p("chi2_indep.p", r.p_value);
    for (n, v) in [("t_cdf(1.7,6)", t_cdf(1.7, 6.0)), ("t_cdf(-0.4,3.5)", t_cdf(-0.4, 3.5)), ("chi2_cdf(7.2,4)", chi2_cdf(7.2, 4.0)), ("f_cdf(2.3,3,17)", f_cdf(2.3, 3.0, 17.0)), ("norm_cdf(-1.3)", norm_cdf(-1.3)), ("beta_inc(2.5,1.5,0.35)", beta_inc(2.5, 1.5, 0.35)), ("gamma_p(3.2,2.1)", gamma_p(3.2, 2.1))] {
        p(n, v);
    }
    let l1 = [0usize, 0, 0, 1, 1, 1, 2, 2, 2, 2];
    let l2 = [0usize, 0, 1, 1, 1, 2, 2, 2, 0, 2];
    p("ari", adjusted_rand_index(&l1, &l2).unwrap());
    p("nmi", normalized_mutual_info(&l1, &l2).unwrap());
    let (h, c, v) = homogeneity_completeness_v(&l1, &l2).unwrap();
    p("homogeneity", h);
    p("completeness", c);
    p("v_measure", v);
    let pts: Vec<Vec<f64>> = (0..10).map(|i| vec![(i * i % 7) as f64 + 0.1 * i as f64, ((i * 3) % 5) as f64 - 0.05 * i as f64]).collect();
    p("silhouette", silhouette_score(&pts, &l1).unwrap());
    p("davies_bouldin", davies_bouldin(&pts, &l1).unwrap());
    p("calinski_harabasz", calinski_harabasz(&pts, &l1).unwrap());
    let scores = [0.1, 0.4, 0.35, 0.8, 0.65, 0.2, 0.9, 0.55, 0.3, 0.7];
    let tg = [0usize, 0, 1, 1, 1, 0, 1, 0, 0, 1];
    p("roc_auc", roc_auc(&scores, &tg).unwrap());
    p("log_loss", log_loss(&scores, &tg).unwrap());
    p("brier", brier_score(&scores, &tg).unwrap());
    let (bb, aa) = oxideon_core::dsp::butter(3, 0.35, false).unwrap();
    bb.iter().enumerate().for_each(|(i, v)| p(&format!("butter3_b{i}"), *v));
    aa.iter().enumerate().for_each(|(i, v)| p(&format!("butter3_a{i}"), *v));
    let (bh, ah) = oxideon_core::dsp::butter(4, 0.3, true).unwrap();
    bh.iter().enumerate().for_each(|(i, v)| p(&format!("butter4hp_b{i}"), *v));
    ah.iter().enumerate().for_each(|(i, v)| p(&format!("butter4hp_a{i}"), *v));
}

#[test]
#[ignore]
fn emit_batch2() {
    use oxideon_core::{classical::*, classical_linear::*, dsp, ml_extras, optimization, scientific, timeseries};
    let p = |n: &str, v: f64| println!("{n} {v:.12e}");
    let sig: Vec<f64> = (0..60).map(|i| (i as f64 * 0.3).sin() + 0.2 * ((i * i) as f64 * 0.7).cos()).collect();
    for (i, v) in dsp::savgol_filter(&sig, 7, 2, 0).unwrap().iter().enumerate() {
        p(&format!("savgol0_{i}"), *v);
    }
    for (i, v) in dsp::savgol_filter(&sig, 9, 3, 1).unwrap().iter().enumerate() {
        p(&format!("savgol1_{i}"), *v);
    }
    for (i, v) in dsp::medfilt(&sig, 5).unwrap().iter().enumerate().skip(2).take(55) {
        p(&format!("medfilt_{i}"), *v);
    }
    for (i, v) in dsp::firwin(41, &[0.25], false).unwrap().iter().enumerate() {
        p(&format!("firwin_lp_{i}"), *v);
    }
    for (i, v) in dsp::firwin(41, &[0.2, 0.5], false).unwrap().iter().enumerate() {
        p(&format!("firwin_bp_{i}"), *v);
    }
    for (i, v) in dsp::firwin(41, &[0.3], true).unwrap().iter().enumerate() {
        p(&format!("firwin_hp_{i}"), *v);
    }
    let long: Vec<f64> = (0..512).map(|i| (i as f64 * 0.4).sin() + 0.5 * (i as f64 * 1.3).sin() + 0.1 * ((i * 7919) % 13) as f64).collect();
    let (f, pw) = dsp::welch(&long, 100.0, 128, 64).unwrap();
    for (i, (ff, pp)) in f.iter().zip(&pw).enumerate() {
        p(&format!("welch_f{i}"), *ff);
        p(&format!("welch_p{i}"), *pp);
    }
    let pk = dsp::find_peaks(&sig, -10.0, 1);
    println!("peaks {}", pk.iter().map(usize::to_string).collect::<Vec<_>>().join(","));
    let xs: Vec<f64> = vec![0.0, 0.7, 1.5, 2.1, 3.4, 4.0, 5.5];
    let ys: Vec<f64> = vec![1.0, 0.2, 1.9, -0.5, 2.2, 1.1, 0.3];
    let cs = scientific::CubicSpline::new(&xs, &ys).unwrap();
    for q in [0.3, 1.0, 1.8, 2.7, 3.7, 4.8, 5.4] {
        p(&format!("spline_{q:.1}"), cs.eval(q));
        p(&format!("splined_{q:.1}"), cs.derivative(q));
    }
    let cost = vec![vec![4.0, 1.0, 3.0, 7.0], vec![2.0, 0.0, 5.0, 3.0], vec![3.0, 2.0, 2.0, 9.0], vec![6.0, 4.0, 8.0, 1.0]];
    let (cols, total) = optimization::assignment(&cost).unwrap();
    p("assign_total", total);
    println!("assign_cols {}", cols.iter().map(usize::to_string).collect::<Vec<_>>().join(","));
    let lp = optimization::linprog(&[-3.0, -5.0, -1.0], &[vec![1.0, 0.0, 1.0], vec![0.0, 2.0, 1.0], vec![3.0, 2.0, 2.0]], &[4.0, 12.0, 18.0], &[vec![1.0, 1.0, 1.0]], &[7.0]).unwrap();
    p("linprog_obj", lp.objective);
    let nn_a = vec![vec![1.0, 0.5, 2.0], vec![0.3, 1.2, 0.1], vec![2.0, 0.1, 0.4], vec![0.6, 0.9, 1.1], vec![1.5, 0.2, 0.7]];
    let (nx, nr) = optimization::nnls(&nn_a, &[1.0, -0.5, 2.0, 0.3, 1.1]).unwrap();
    nx.iter().enumerate().for_each(|(i, v)| p(&format!("nnls_x{i}"), *v));
    p("nnls_res", nr);
    let (q, _) = scientific::quad(|x| (x * x).sin() * (-x).exp(), 0.0, 4.0, 1e-12).unwrap();
    p("quad", q);
    p("brentq", scientific::brentq(|x| x.cos() - x * x * x, 0.0, 1.0, 1e-13).unwrap());
    let cx: Vec<f64> = (0..30).map(|i| i as f64 * 0.2).collect();
    let cy: Vec<f64> = cx.iter().map(|&x| 2.0 * (-0.7 * x).exp() + 0.3 + 0.01 * ((x * 37.0).sin())).collect();
    let (cp, sse) = scientific::curve_fit(|x, p| p[0] * (-p[1] * x).exp() + p[2], &cx, &cy, &[1.0, 1.0, 0.0], 500).unwrap();
    cp.iter().enumerate().for_each(|(i, v)| p(&format!("curvefit_p{i}"), *v));
    p("curvefit_sse", sse);
    let x: Matrix = (0..30).map(|i| vec![i as f64 / 5.0, ((i * 7) % 11) as f64 / 3.0, (i as f64 * 0.37).sin()]).collect();
    let y: Vec<f64> = x.iter().map(|r| 1.5 + 2.0 * r[0] - 1.0 * r[1] + 0.5 * r[2] + 0.1 * (r[0] * 13.0).cos()).collect();
    let ols = LinearRegression::fit(&x, &y, 0.0).unwrap();
    ols.coef.iter().enumerate().for_each(|(i, v)| p(&format!("ols_c{i}"), *v));
    p("ols_b", ols.intercept);
    let rd = LinearRegression::fit(&x, &y, 2.5).unwrap();
    rd.coef.iter().enumerate().for_each(|(i, v)| p(&format!("ridge_c{i}"), *v));
    p("ridge_b", rd.intercept);
    let pca = Pca::fit(&x, 2).unwrap();
    pca.explained_variance.iter().enumerate().for_each(|(i, v)| p(&format!("pca_var{i}"), *v));
    pca.explained_variance_ratio().iter().enumerate().for_each(|(i, v)| p(&format!("pca_ratio{i}"), *v));
    let iso = ml_extras::isotonic_regression(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1.0, 3.0, 2.0, 4.0, 3.5, 6.0, 5.0, 8.0]).unwrap();
    iso.iter().enumerate().for_each(|(i, v)| p(&format!("iso_{i}"), *v));
    for (i, v) in timeseries::acf(&sig, 8).unwrap().iter().enumerate() {
        p(&format!("acf_{i}"), *v);
    }
    for (i, v) in timeseries::pacf(&sig, 6).unwrap().iter().enumerate() {
        p(&format!("pacf_{i}"), *v);
    }
}
