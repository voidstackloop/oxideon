//! Third cross-validation batch: classical ML, graph, geometry, algebra, numerics, iterative, QMC, finance and
//! multiple-testing values, compared by `tools/crossval/check_batch3.py` against scikit-learn / SciPy / NetworkX / SymPy.
//! Run: `cargo test -p oxideon-core --test crossval_emit3 -- --nocapture --ignored`.

use oxideon_core::{algebra, anomaly, causal, classical::*, classical_more, decomp, finance, geometry3d, graph_algos::Graph, iterative, naive_bayes, numerics, qmc, text_ir};

fn ints(v: &[usize]) -> String {
    v.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
}

#[test]
#[ignore]
fn emit_batch3() {
    let p = |n: &str, v: f64| println!("{n} {v:.12e}");
    let pts: Matrix = (0..45)
        .map(|i| {
            let c = i % 3;
            let (cx, cy) = [(0.0, 0.0), (5.0, 1.0), (2.0, 6.0)][c];
            vec![cx + ((i * 37 % 17) as f64 - 8.0) / 9.0 + 1e-4 * (i as f64 * 1.3).sin(), cy + ((i * 53 % 19) as f64 - 9.0) / 9.0 + 1e-4 * (i as f64 * 2.1).cos()]
        })
        .collect();
    let y: Vec<usize> = (0..45).map(|i| i % 3).collect();
    let db = Dbscan { eps: 0.9, min_samples: 4 }.fit_predict(&pts).unwrap();
    println!("dbscan {}", db.iter().map(|l| l.map_or("-1".to_string(), |v| v.to_string())).collect::<Vec<_>>().join(","));
    for (name, l) in [("single", decomp::Linkage::Single), ("complete", decomp::Linkage::Complete), ("average", decomp::Linkage::Average), ("ward", decomp::Linkage::Ward)] {
        println!("agglo_{name} {}", ints(&decomp::agglomerative(&pts, 3, l).unwrap()));
    }
    println!("spectral {}", ints(&decomp::spectral_clustering(&pts, 3, 0.5, 1).unwrap()));
    let knn = KnnClassifier::fit(pts.clone(), y.clone(), 5).unwrap();
    let q: Matrix = (0..12).map(|i| vec![i as f64 * 0.5 - 1.0, (i * i % 7) as f64]).collect();
    println!("knn {}", ints(&knn.predict(&q)));
    let nb = GaussianNb::fit(&pts, &y).unwrap();
    println!("gnb {}", ints(&nb.predict(&q)));
    for (i, row) in nb.log_proba(&q).iter().enumerate().take(4) {
        row.iter().enumerate().for_each(|(c, v)| p(&format!("gnb_ll_{i}_{c}"), *v));
    }
    let counts: Matrix = (0..40).map(|i| (0..6).map(|j| ((i * (j + 3) + j * 5) % 7) as f64).collect()).collect();
    let cl: Vec<usize> = (0..40).map(|i| i * 7 % 3).collect();
    let mnb = naive_bayes::MultinomialNb::fit(&counts, &cl, 1.0).unwrap();
    println!("mnb {}", ints(&mnb.predict(&counts)));
    mnb.predict_proba(&counts).iter().take(3).enumerate().for_each(|(i, r)| r.iter().enumerate().for_each(|(c, v)| p(&format!("mnb_p_{i}_{c}"), *v)));
    println!("bnb {}", ints(&naive_bayes::BernoulliNb::fit(&counts, &cl, 1.0, 0.0).unwrap().predict(&counts)));
    println!("cnb {}", ints(&naive_bayes::ComplementNb::fit(&counts, &cl, 1.0, false).unwrap().predict(&counts)));
    let bx: Matrix = (0..60).map(|i| vec![(i as f64 * 0.17).sin() * 2.0, ((i * 11) % 9) as f64 / 4.0 - 1.0]).collect();
    let by: Vec<f64> = bx.iter().enumerate().map(|(i, r)| f64::from(u8::from(r[0] + 0.5 * r[1] + 1.5 * (i as f64 * 1.7).sin() > 0.1))).collect();
    let g = oxideon_core::glm::Glm::fit(&bx, &by, oxideon_core::glm::Family::Binomial, 100).unwrap();
    g.coef.iter().enumerate().for_each(|(i, v)| p(&format!("glm_logit_c{i}"), *v));
    let cnt: Vec<f64> = bx.iter().enumerate().map(|(i, r)| ((0.4 + 0.5 * r[0] - 0.3 * r[1]).exp() + (i % 3) as f64).round()).collect();
    let gp = oxideon_core::glm::Glm::fit(&bx, &cnt, oxideon_core::glm::Family::Poisson, 100).unwrap();
    gp.coef.iter().enumerate().for_each(|(i, v)| p(&format!("glm_pois_c{i}"), *v));
    let gx: Matrix = (0..80).map(|i| vec![i as f64 / 10.0, ((i * 13) % 17) as f64 / 5.0]).collect();
    let gy: Vec<f64> = gx.iter().map(|r| r[0].sin() * 2.0 + r[1] * 0.5).collect();
    let gb = classical_more::GradientBoosting::fit(&gx, &gy, 30, 0.1, 3, false).unwrap();
    gb.predict(&gx).iter().enumerate().for_each(|(i, v)| p(&format!("gbr_{i}"), *v));
    let rt = classical_more::RegressionTree::fit(&gx, &gy, 4, 1).unwrap();
    rt.predict(&gx).iter().enumerate().for_each(|(i, v)| p(&format!("rtree_{i}"), *v));
    anomaly::lof_scores(&pts, 6).unwrap().iter().enumerate().for_each(|(i, v)| p(&format!("lof_{i}"), *v));
    anomaly::mahalanobis_scores(&pts).unwrap().iter().enumerate().for_each(|(i, v)| p(&format!("maha_{i}"), *v));
    let docs = ["the cat sat on the mat", "the dog chased the cat", "dogs and cats are pets", "quantum field theory lectures"];
    let tf = text_ir::TfidfVectorizer::fit(&docs).unwrap();
    for (d, doc) in docs.iter().enumerate() {
        for (ix, v) in tf.transform_one(doc) {
            p(&format!("tfidf_{d}_{ix}"), v);
        }
    }
    let edges = [(0, 1, 7.0), (0, 2, 9.0), (0, 5, 14.0), (1, 2, 10.0), (1, 3, 15.0), (2, 3, 11.0), (2, 5, 2.0), (3, 4, 6.0), (4, 5, 9.0), (6, 7, 1.0), (7, 8, 2.0)];
    let gr = Graph::from_edges(9, false, &edges).unwrap();
    let (d, _) = gr.dijkstra(0).unwrap();
    d.iter().enumerate().for_each(|(i, v)| {
        if v.is_finite() {
            p(&format!("dijkstra_{i}"), *v)
        }
    });
    gr.pagerank(0.85, 200).iter().enumerate().for_each(|(i, v)| p(&format!("pagerank_{i}"), *v));
    gr.betweenness().iter().enumerate().for_each(|(i, v)| p(&format!("betweenness_{i}"), *v));
    gr.clustering().iter().enumerate().for_each(|(i, v)| p(&format!("clustering_{i}"), *v));
    p("mst_total", gr.kruskal().1);
    println!("components {}", ints(&gr.components()));
    let dg = Graph::from_edges(6, true, &[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 3, 1.0), (4, 5, 1.0)]).unwrap();
    println!("scc {}", ints(&dg.scc()));
    p("maxflow", oxideon_core::graph_algos::max_flow(6, &[(0, 1, 16.0), (0, 2, 13.0), (1, 2, 10.0), (2, 1, 4.0), (1, 3, 12.0), (3, 2, 9.0), (2, 4, 14.0), (4, 3, 7.0), (3, 5, 20.0), (4, 5, 4.0)], 0, 5).unwrap());
    let q1 = geometry3d::Quaternion::from_euler_zyx(0.4, -0.3, 0.9);
    p("quat_w", q1.w);
    p("quat_x", q1.x);
    p("quat_y", q1.y);
    p("quat_z", q1.z);
    let m = q1.to_matrix();
    (0..9).for_each(|i| p(&format!("rotm_{i}"), m[i / 3][i % 3]));
    let q2 = geometry3d::Quaternion::from_euler_zyx(-1.1, 0.7, 0.2);
    let s = q1.slerp(&q2, 0.37);
    p("slerp_w", s.w);
    p("slerp_x", s.x);
    p("slerp_y", s.y);
    p("slerp_z", s.z);
    let src: Vec<geometry3d::Vec3> = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [1.0, 1.0, 1.0], [-1.0, 0.5, 2.0]];
    let se = geometry3d::Se3 { rot: m, trans: [1.0, -2.0, 0.5] };
    let dst: Vec<geometry3d::Vec3> = src
        .iter()
        .enumerate()
        .map(|(i, &a)| {
            let r = se.apply(a);
            [r[0] + 0.01 * (i as f64).sin(), r[1] - 0.02 * (i as f64).cos(), r[2] + 0.015 * (i as f64 * 2.0).sin()]
        })
        .collect();
    let (est, rms) = geometry3d::kabsch(&src, &dst).unwrap();
    (0..9).for_each(|i| p(&format!("kabsch_r{i}"), est.rot[i / 3][i % 3]));
    est.trans.iter().enumerate().for_each(|(i, v)| p(&format!("kabsch_t{i}"), *v));
    p("kabsch_rms", rms);
    let poly = algebra::Polynomial::new(vec![-6.0, 11.0, -6.0, 1.0, 0.5]);
    let mut roots = poly.roots().unwrap();
    roots.sort_by(|a, b| ((a.0 * 1e8).round()).total_cmp(&(b.0 * 1e8).round()).then(a.1.total_cmp(&b.1)));
    roots.iter().enumerate().for_each(|(i, r)| {
        p(&format!("root_re{i}"), r.0);
        p(&format!("root_im{i}"), r.1);
    });
    let (dq, dr) = poly.div_rem(&algebra::Polynomial::new(vec![2.0, -1.0, 1.0])).unwrap();
    dq.c.iter().enumerate().for_each(|(i, v)| p(&format!("pdiv_q{i}"), *v));
    dr.c.iter().enumerate().for_each(|(i, v)| p(&format!("pdiv_r{i}"), *v));
    let ch = numerics::ChebyshevInterp::fit(|x| (3.0 * x).sin() * (-x).exp(), -1.0, 2.0, 12).unwrap();
    for t in [-0.9, -0.2, 0.4, 1.1, 1.9] {
        p(&format!("cheb_{t}"), ch.eval(t));
        p(&format!("chebd_{t}"), ch.derivative().eval(t));
    }
    let (lx, ly) = ([0.0, 1.0, 2.5, 4.0, 5.5], [1.0, -1.0, 2.0, 0.5, 3.0]);
    p("lagrange", numerics::lagrange_eval(&lx, &ly, 3.3).unwrap());
    let grid = numerics::RegularGrid::new(vec![vec![0.0, 1.0, 3.0], vec![-1.0, 0.5, 2.0, 4.0]], (0..12).map(|i| ((i * 7) % 5) as f64 + 0.25 * i as f64).collect()).unwrap();
    for (a, b) in [(0.5, 1.0), (2.2, 3.3), (1.0, -0.4)] {
        p(&format!("grid_{a:.1}_{b:.1}"), grid.eval(&[a, b]).unwrap());
    }
    let n = 40;
    let lap = |x: &[f64]| -> Vec<f64> { (0..x.len()).map(|i| 2.0 * x[i] - if i > 0 { x[i - 1] } else { 0.0 } - if i + 1 < x.len() { x[i + 1] } else { 0.0 } + 0.01 * (i as f64) * x[i]).collect() };
    let rhs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).cos()).collect();
    iterative::cg(lap, &rhs, None, None, 1e-12, 500).unwrap().x.iter().enumerate().for_each(|(i, v)| p(&format!("cg_{i}"), *v));
    let (lo, hi) = iterative::lanczos_extreme(lap, n, 40, 1).unwrap();
    p("lanczos_min", lo);
    p("lanczos_max", hi);
    for (i, pt) in qmc::sobol(32, 6).unwrap().iter().enumerate() {
        pt.iter().enumerate().for_each(|(d, v)| p(&format!("sobol_{i}_{d}"), *v));
    }
    for pp in [0.001, 0.025, 0.3, 0.5, 0.84, 0.975, 0.9999] {
        p(&format!("ppf_{pp}"), causal::norm_ppf(pp));
    }
    let o = finance::OptionSpec { s: 105.0, k: 98.0, r: 0.03, q: 0.01, sigma: 0.27, t: 0.75 };
    let c = finance::black_scholes(&o, true).unwrap();
    for (n, v) in [("bs_price", c.price), ("bs_delta", c.delta), ("bs_gamma", c.gamma), ("bs_vega", c.vega), ("bs_theta", c.theta), ("bs_rho", c.rho)] {
        p(n, v);
    }
    p("bs_put", finance::black_scholes(&o, false).unwrap().price);
    p("irr", finance::irr(&[-1000.0, 300.0, 400.0, 500.0, 120.0]).unwrap());
    p("loan", finance::loan_payment(250_000.0, 0.045 / 12.0, 300).unwrap());
    let pv = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216];
    causal::holm(&pv).unwrap().iter().enumerate().for_each(|(i, v)| p(&format!("holm_{i}"), *v));
    causal::benjamini_hochberg(&pv).unwrap().iter().enumerate().for_each(|(i, v)| p(&format!("bh_{i}"), *v));
}
