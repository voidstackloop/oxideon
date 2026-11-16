//! Cross-validation batch 14: RDP accountant, sketches, recsys, evolutionary optimizers
//! (compared by tools/crossval/check_batch14.py).
use oxideon_core::evolutionary::*;
use oxideon_core::privacy::RdpAccountant;
use oxideon_core::recsys::*;
use oxideon_core::sketches::*;

/// Deterministic pseudo-uniform in (0, 1): reproducible in Python with integers.
fn u(i: u64) -> f64 {
    (((i.wrapping_mul(2_654_435_761)) % 4_294_967_296) as f64 + 0.5) / 4_294_967_296.0
}

#[test]
#[ignore]
fn emit() {
    // --- RDP accountant
    for (q, sigma, steps, delta) in [(0.01, 1.1, 1000usize, 1e-5), (256.0 / 60000.0, 1.1, 14062, 1e-5), (0.05, 0.8, 200, 1e-6), (1.0, 2.0, 10, 1e-5), (0.002, 0.7, 5000, 1e-5)] {
        let mut a = RdpAccountant::new(q, sigma);
        a.step(steps);
        println!("rdp_{q}_{sigma}_{steps}_{delta} {:.12}", a.epsilon(delta));
    }

    // --- RunningStats (+ merge)
    let xs: Vec<f64> = (1..=500).map(|i| (u(i) * 20.0 - 7.0).powi(3) / 50.0).collect();
    let mut all = RunningStats::new();
    xs.iter().for_each(|&x| all.push(x));
    let (mut l, mut r) = (RunningStats::new(), RunningStats::new());
    xs[..137].iter().for_each(|&x| l.push(x));
    xs[137..].iter().for_each(|&x| r.push(x));
    l.merge(&r);
    for (tag, s) in [("all", &all), ("merged", &l)] {
        println!("rs_{tag}_mean {:.12}", s.mean());
        println!("rs_{tag}_var {:.12}", s.variance());
        println!("rs_{tag}_min {:.12}", s.min);
        println!("rs_{tag}_max {:.12}", s.max);
        println!("rs_{tag}_n {}", s.count());
    }

    // --- HyperLogLog relative error
    for (p, n) in [(10u32, 500u64), (10, 5000), (12, 100_000), (14, 300_000), (8, 20_000)] {
        let mut h = HyperLogLog::new(p).unwrap();
        for i in 0..n {
            h.add(format!("item-{i}").as_bytes());
        }
        println!("hll_{p}_{n} {:.6}", h.estimate());
    }

    // --- Count-Min: never under-estimates, error within epsilon * total w.h.p.
    let mut cm = CountMinSketch::new(0.001, 0.01).unwrap();
    let mut truth = vec![0u64; 2000];
    let mut total = 0u64;
    for i in 0..50_000u64 {
        let k = ((u(i) * u(i + 99_999)).powi(2) * 2000.0) as usize % 2000; // skewed
        cm.add(format!("k{k}").as_bytes(), 1);
        truth[k] += 1;
        total += 1;
    }
    let (mut under, mut over_eps) = (0, 0);
    for (k, &t) in truth.iter().enumerate() {
        let e = cm.estimate(format!("k{k}").as_bytes());
        if e < t {
            under += 1;
        }
        if (e - t.min(e)) as f64 > 0.001 * total as f64 {
            over_eps += 1;
        }
    }
    println!("cm_under {under}");
    println!("cm_over_eps_frac {:.6}", over_eps as f64 / 2000.0);

    // --- Bloom: no false negatives, empirical FP near target
    let mut bf = BloomFilter::new(10_000, 0.01).unwrap();
    for i in 0..10_000 {
        bf.insert(format!("in-{i}").as_bytes());
    }
    let neg = (0..10_000).filter(|i| !bf.contains(format!("in-{i}").as_bytes())).count();
    let fp = (0..100_000).filter(|i| bf.contains(format!("out-{i}").as_bytes())).count();
    println!("bloom_false_neg {neg}");
    println!("bloom_fp_rate {:.6}", fp as f64 / 100_000.0);
    println!("bloom_est_fp {:.6}", bf.estimated_fp_rate());

    // --- Streaming quantiles vs exact (exponential data)
    let data: Vec<f64> = (1..=100_000u64).map(|i| -u(i).ln() * 3.0).collect();
    let mut sq = StreamingQuantiles::new(100.0).unwrap();
    data.iter().for_each(|&x| sq.add(x));
    for q in [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999] {
        println!("tdigest_{q} {:.9}", sq.quantile(q).unwrap());
    }
    println!("tdigest_centroids {}", sq.n_centroids());

    // --- Reservoir: inclusion frequency uniform
    let (trials, n, k) = (4000u64, 40u64, 5usize);
    let mut hits = vec![0u32; n as usize];
    for s in 0..trials {
        let mut r = Reservoir::new(k, s + 1).unwrap();
        (0..n).for_each(|i| r.push(i));
        r.sample().iter().for_each(|&i| hits[i as usize] += 1);
    }
    for (i, h) in hits.iter().enumerate() {
        println!("res_{i} {h}");
    }

    // --- SpaceSaving: exact for frequent items, count upper bound >= true, bound - err <= true
    let mut ss = SpaceSaving::new(20).unwrap();
    let mut tc = std::collections::HashMap::new();
    for i in 0..30_000u64 {
        let k = if u(i) < 0.4 { (u(i + 7) * 3.0) as u64 } else { (u(i + 13) * 500.0) as u64 + 10 };
        ss.add(&k.to_le_bytes());
        *tc.entry(k).or_insert(0u64) += 1;
    }
    for (rank, (item, cnt, err)) in ss.top().into_iter().take(8).enumerate() {
        let key = u64::from_le_bytes(item.try_into().unwrap());
        println!("ss_{rank} {key} {cnt} {err} {}", tc[&key]);
    }

    // --- Apriori / association rules on fixed transactions
    let tx: Vec<Vec<usize>> = (0..200u64)
        .map(|t| {
            let mut v = vec![];
            for item in 0..8usize {
                let p = [0.7, 0.5, 0.45, 0.3, 0.25, 0.2, 0.1, 0.05][item];
                // correlate item 1 with item 0 and item 3 with item 2
                let boost = if (item == 1 && v.contains(&0)) || (item == 3 && v.contains(&2)) { 0.35 } else { 0.0 };
                if u(t * 8 + item as u64 + 1) < p + boost {
                    v.push(item);
                }
            }
            v
        })
        .collect();
    let sets = apriori(&tx, 0.08).unwrap();
    println!("ap_nsets {}", sets.len());
    for (s, sup) in &sets {
        println!("ap_{} {sup:.9}", s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("-"));
    }
    let rules = association_rules(&sets, 0.5);
    println!("rules_n {}", rules.len());
    for r in &rules {
        let a = r.antecedent.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("-");
        let c = r.consequent.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("-");
        println!("rule_{a}_to_{c}_conf {:.9}", r.confidence);
        println!("rule_{a}_to_{c}_lift {:.9}", r.lift);
        println!("rule_{a}_to_{c}_sup {:.9}", r.support);
    }

    // --- ItemKnn on fixed ratings
    let mut ratings = vec![];
    for user in 0..12usize {
        for item in 0..9usize {
            let h = u((user * 9 + item) as u64 + 5);
            if h < 0.6 {
                ratings.push(Rating { user, item, value: 1.0 + (u((user * 9 + item) as u64 + 777) * 5.0).floor().min(4.0) });
            }
        }
    }
    let knn = ItemKnn::fit(&ratings, 12, 9, 3).unwrap();
    for user in 0..12usize {
        for item in 0..9usize {
            println!("knn_{user}_{item} {:.9}", knn.predict(user, item));
        }
    }

    // --- Matrix factorization: low-rank synthetic ground truth, held-out RMSE
    let (nu, ni) = (30usize, 25usize);
    let truth = |a: usize, b: usize| 3.0 + ((a % 3) as f64 - 1.0) * ((b % 3) as f64 - 1.0) + 0.3 * (a % 5) as f64 / 5.0;
    let (mut train, mut test) = (vec![], vec![]);
    for a in 0..nu {
        for b in 0..ni {
            let r = Rating { user: a, item: b, value: truth(a, b) };
            if u((a * ni + b) as u64 + 31) < 0.7 { train.push(r) } else { test.push(r) }
        }
    }
    let mf = MatrixFactorization::fit(&train, nu, ni, 4, 300, 0.02, 0.02, 3).unwrap();
    println!("mf_train_rmse {:.6}", mf.rmse(&train));
    println!("mf_test_rmse {:.6}", mf.rmse(&test));
    let mean = train.iter().map(|r| r.value).sum::<f64>() / train.len() as f64;
    let base = (test.iter().map(|r| (r.value - mean).powi(2)).sum::<f64>() / test.len() as f64).sqrt();
    println!("mf_baseline_rmse {base:.6}");

    // --- Evolutionary optimizers on classic benchmarks (best value over 5 seeds)
    let sphere = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>();
    let rosen = |x: &[f64]| x.windows(2).map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2)).sum::<f64>();
    let rastrigin = |x: &[f64]| 10.0 * x.len() as f64 + x.iter().map(|v| v * v - 10.0 * (2.0 * std::f64::consts::PI * v).cos()).sum::<f64>();
    let ackley = |x: &[f64]| {
        let n = x.len() as f64;
        -20.0 * (-0.2 * (x.iter().map(|v| v * v).sum::<f64>() / n).sqrt()).exp() - ((x.iter().map(|v| (2.0 * std::f64::consts::PI * v).cos()).sum::<f64>()) / n).exp() + 20.0 + std::f64::consts::E
    };
    let b5 = vec![(-5.12, 5.12); 5];
    let b2 = vec![(-2.0, 2.0); 2];
    type Bench<'a> = (&'a str, &'a dyn Fn(&[f64]) -> f64, &'a Vec<(f64, f64)>);
    let fns: [Bench; 4] = [("sphere", &sphere, &b5), ("rosen", &rosen, &b2), ("rastrigin", &rastrigin, &b5), ("ackley", &ackley, &b5)];
    for (name, f, b) in fns {
        let x0: Vec<f64> = b.iter().map(|&(lo, hi)| lo + 0.8 * (hi - lo)).collect();
        for seed in 1..=5u64 {
            let rs = [
                ("de", differential_evolution(f, b, 40, 300, 0.7, 0.9, seed).unwrap()),
                ("cma", cma_es(f, &x0, 1.0, 600, seed).unwrap()),
                ("sa", simulated_annealing(f, &x0, 1.0, 5.0, 1e-4, 40_000, seed).unwrap()),
                ("pso", particle_swarm(f, b, 40, 300, seed).unwrap()),
                ("ga", genetic_algorithm(f, b, 60, 300, 0.2, seed).unwrap()),
            ];
            for (alg, r) in rs {
                let recomputed = f(&r.x);
                let mono = r.history.windows(2).all(|w| w[1] <= w[0] + 1e-15);
                println!("opt_{name}_{alg}_{seed} {:.9e} {:.9e} {}", r.value, recomputed, mono as u8);
            }
        }
    }
}
