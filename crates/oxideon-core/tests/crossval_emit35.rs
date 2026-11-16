//! Cross-validation batch 35: sampling filters (temperature, top-k, top-p, min-p, penalties, logit bias) compared with the
//! Hugging Face logits-processor order through empirical token frequencies (tools/crossval/check_batch35.py).
use oxideon_core::rng::Generator;
use oxideon_core::sampling::SamplingConfig;
use oxideon_core::tensor::Tensor;

const DRAWS: usize = 60_000;

fn logits(vocab: usize) -> Vec<f64> {
    (0..vocab).map(|i| (i as f64 * 1.7 + 0.4).sin() * 2.2 + (i as f64 * 0.31).cos()).collect()
}

fn run(tag: &str, cfg: &SamplingConfig, lg: &[f64], history: Option<&[usize]>) {
    let t = Tensor::from_slice(lg, &[lg.len()]).unwrap();
    let mut g = Generator::with_seed(12345);
    let mut counts = vec![0usize; lg.len()];
    for _ in 0..DRAWS {
        counts[cfg.sample_token(&t, history, Some(&mut g)).unwrap()] += 1;
    }
    println!("{tag} {}", counts.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(" "));
}

#[test]
#[ignore]
fn emit_sampling() {
    let lg = logits(12);
    run("plain", &SamplingConfig::new(), &lg, None);
    run("temp05", &SamplingConfig::new().with_temperature(0.5), &lg, None);
    run("temp17", &SamplingConfig::new().with_temperature(1.7), &lg, None);
    run("topk4", &SamplingConfig::new().with_top_k(4), &lg, None);
    run("topk1", &SamplingConfig::new().with_top_k(1), &lg, None);
    run("topp09", &SamplingConfig::new().with_top_p(0.9), &lg, None);
    run("topp05", &SamplingConfig::new().with_top_p(0.5), &lg, None);
    run("topp03", &SamplingConfig::new().with_top_p(0.3), &lg, None);
    run("minp01", &SamplingConfig::new().with_min_p(0.1), &lg, None);
    run("minp04", &SamplingConfig::new().with_min_p(0.4), &lg, None);
    run("combo", &SamplingConfig::new().with_temperature(0.8).with_top_k(6).with_top_p(0.85).with_min_p(0.05), &lg, None);
    let hist = [2usize, 2, 5, 7, 7, 7];
    run("rep13", &SamplingConfig::new().with_repetition_penalty(1.3), &lg, Some(&hist));
    run("presence", &SamplingConfig::new().with_presence_penalty(0.8), &lg, Some(&hist));
    run("frequency", &SamplingConfig::new().with_frequency_penalty(0.5), &lg, Some(&hist));
    run("bias", &SamplingConfig::new().with_logit_bias(3, 2.5).with_logit_bias(9, -4.0), &lg, None);
    // exact-boundary nucleus: probabilities 0.5 / 0.3 / 0.2 and top_p = 0.5 must keep only the first token
    let lp: Vec<f64> = [0.5f64, 0.3, 0.2].iter().map(|p| p.ln()).collect();
    run("boundary_p05", &SamplingConfig::new().with_top_p(0.5), &lp, None);
    run("boundary_p075", &SamplingConfig::new().with_top_p(0.75), &lp, None);
}
