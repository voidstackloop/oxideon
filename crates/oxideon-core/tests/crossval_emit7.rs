//! Seventh cross-validation batch: loss functions (values and gradients) vs PyTorch.
//! Run: `cargo test -p oxideon-core --test crossval_emit7 -- --nocapture --ignored`.

use oxideon_core::nn::loss::*;
use oxideon_core::nn::loss_ext::*;
use oxideon_core::tensor::Tensor;

fn m(v: &[f64], shape: &[usize]) -> Tensor {
    Tensor::from_slice(v, shape).unwrap()
}

fn scalar(name: &str, t: &Tensor) {
    println!("{name} {:.12e}", t.to_vec::<f64>().unwrap()[0]);
}

fn grad_of(name: &str, x: &Tensor, loss: &Tensor) {
    loss.backward().unwrap();
    x.grad().unwrap().to_vec::<f64>().unwrap().iter().enumerate().for_each(|(i, v)| println!("{name}_g{i} {v:.12e}"));
}

#[test]
#[ignore]
fn emit_losses() {
    let pred = [0.3, -1.2, 2.5, 0.9, -0.4, 1.8];
    let targ = [0.0, -0.7, 1.9, 2.4, -0.4, 0.5];
    let (p, t) = (m(&pred, &[2, 3]), m(&targ, &[2, 3]));
    scalar("mse", &mse_loss(&p, &t).unwrap());
    scalar("l1", &l1_loss(&p, &t).unwrap());
    scalar("huber", &huber_loss(&p, &t, 1.3).unwrap());
    scalar("smooth_l1", &smooth_l1_loss(&p, &t, 0.7, Reduction::Mean).unwrap());
    scalar("smooth_l1_sum", &smooth_l1_loss(&p, &t, 0.7, Reduction::Sum).unwrap());
    let labels01 = m(&[0.0, 1.0, 1.0, 0.0, 1.0, 0.0], &[2, 3]);
    scalar("bce_logits", &binary_cross_entropy_with_logits(&p, &labels01).unwrap());
    let probs = m(&[0.1, 0.7, 0.35, 0.92, 0.5, 0.03], &[2, 3]);
    scalar("bce", &binary_cross_entropy(&probs, &labels01, None, Reduction::Mean).unwrap());
    let logits = m(&[1.2, -0.3, 0.8, 2.0, 0.1, -1.5, 0.4, 0.4, 1.1], &[3, 3]);
    let cls = [2usize, 0, 1];
    scalar("ce", &cross_entropy(&logits, &cls).unwrap());
    scalar("ce_smooth", &cross_entropy_with_smoothing(&logits, &cls, 0.1).unwrap());
    scalar("nll", &nll_loss(&logits.log_softmax(1).unwrap(), &cls).unwrap());
    let lg = m(&[1.2, -0.3, 0.8, 2.0, 0.1, -1.5, 0.4, 0.4, 1.1], &[3, 3]).requires_grad_(true);
    grad_of("ce", &lg, &cross_entropy(&lg, &cls).unwrap());
    let tp = m(&[0.2, 0.5, 0.3, 0.1, 0.1, 0.8, 0.6, 0.3, 0.1], &[3, 3]);
    scalar("kl_div_mean", &kl_div_r(&logits.log_softmax(1).unwrap(), &tp, KlReduction::Mean, false).unwrap());
    scalar("kl_div_batchmean", &kl_div_r(&logits.log_softmax(1).unwrap(), &tp, KlReduction::BatchMean, false).unwrap());
    let a = m(&[0.3, 1.1, -0.4, 0.9, -0.2, 0.7], &[2, 3]);
    let b = m(&[0.1, 0.8, -0.1, -0.5, 0.6, 0.2], &[2, 3]);
    let c = m(&[1.0, -0.4, 0.3, 0.2, 0.5, -0.9], &[2, 3]);
    scalar("cosine_embedding", &cosine_embedding_loss(&a, &b, &m(&[1.0, -1.0], &[2]), 0.2).unwrap());
    scalar("triplet", &triplet_margin_loss(&a, &b, &c, 1.0).unwrap());
    let x1 = m(&[0.4, 1.2, -0.3, 0.8], &[4]);
    let x2 = m(&[0.1, 1.5, 0.2, -0.4], &[4]);
    let y = m(&[1.0, -1.0, 1.0, -1.0], &[4]);
    scalar("margin_ranking", &margin_ranking_loss(&x1, &x2, &y, 0.3).unwrap());
    scalar("hinge_embedding", &hinge_embedding_loss(&x1, &y, 0.9).unwrap());
    scalar("soft_margin", &soft_margin_loss(&x1, &y).unwrap());
    let ml_t = m(&[1.0, 2.0, -1.0, 0.0, 2.0, -1.0], &[2, 3]);
    scalar("multilabel_margin", &multilabel_margin_loss(&p, &ml_t).unwrap());
    scalar("multilabel_soft_margin", &multilabel_soft_margin_loss(&p, &labels01, None, Reduction::Mean).unwrap());
    let mm_t = Tensor::from_slice(&[2i64, 0, 1], &[3]).unwrap();
    scalar("multi_margin1", &multi_margin_loss(&logits, &mm_t, 1, 1.0, None, Reduction::Mean).unwrap());
    scalar("multi_margin2", &multi_margin_loss(&logits, &mm_t, 2, 0.8, None, Reduction::Mean).unwrap());
    let counts = m(&[0.0, 2.0, 5.0, 1.0], &[4]);
    let rate = m(&[0.4, 1.1, 1.6, -0.3], &[4]);
    scalar("poisson_nll_log", &poisson_nll_loss(&rate, &counts, true, false, 1e-8, Reduction::Mean).unwrap());
    scalar("poisson_nll_full", &poisson_nll_loss(&rate, &counts, true, true, 1e-8, Reduction::Mean).unwrap());
    let var = m(&[0.5, 1.2, 0.8, 2.0], &[4]);
    scalar("gaussian_nll", &gaussian_nll_loss(&x1, &x2, &var, false, 1e-6, Reduction::Mean).unwrap());
    scalar("gaussian_nll_full", &gaussian_nll_loss(&x1, &x2, &var, true, 1e-6, Reduction::Mean).unwrap());
    // CTC: T = 6, N = 2, C = 4 (blank = 0)
    let raw: Vec<f64> = (0..6 * 2 * 4).map(|i| ((i * 7 % 11) as f64 - 5.0) / 3.0).collect();
    let lp = m(&raw, &[6, 2, 4]).log_softmax(2).unwrap();
    let tg = Tensor::from_slice(&[1i64, 2, 3, 2, 2, 1], &[2, 3]).unwrap();
    scalar("ctc", &ctc_loss(&lp, &tg, 0, false).unwrap());
}
