//! Cross-validation batch 23: gradients of the loss functions vs PyTorch (tools/crossval/check_batch23.py).
use oxideon_core::nn::loss::*;
use oxideon_core::nn::loss_ext::*;
use oxideon_core::tensor::Tensor;

fn m(v: &[f64], shape: &[usize]) -> Tensor {
    Tensor::from_slice(v, shape).unwrap()
}

fn leaf(v: &[f64], shape: &[usize]) -> Tensor {
    m(v, shape).requires_grad_(true)
}

fn emit(name: &str, loss: &Tensor, inputs: &[(&str, &Tensor)]) {
    println!("{name}_loss {:.14e}", loss.to_vec::<f64>().unwrap()[0]);
    loss.backward().unwrap();
    for (tag, x) in inputs {
        let g = x.grad().unwrap().to_vec::<f64>().unwrap();
        println!("{name}_{tag} {}", g.iter().map(|v| format!("{v:.14e}")).collect::<Vec<_>>().join(" "));
    }
}

#[test]
#[ignore]
fn emit_loss_grads() {
    let pred = [0.3, -1.2, 2.5, 0.9, -0.4, 1.8];
    let targ = [0.0, -0.7, 1.9, 2.4, -0.4, 0.5];
    let t = m(&targ, &[2, 3]);
    let labels01 = m(&[0.0, 1.0, 1.0, 0.0, 1.0, 0.0], &[2, 3]);
    macro_rules! on_pred {
        ($name:literal, $f:expr) => {{
            let p = leaf(&pred, &[2, 3]);
            let l = $f(&p);
            emit($name, &l, &[("g", &p)]);
        }};
    }
    on_pred!("mse", |p: &Tensor| mse_loss(p, &t).unwrap());
    on_pred!("l1", |p: &Tensor| l1_loss(p, &t).unwrap());
    on_pred!("huber", |p: &Tensor| huber_loss(p, &t, 1.3).unwrap());
    on_pred!("smooth_l1", |p: &Tensor| smooth_l1_loss(p, &t, 0.7, Reduction::Mean).unwrap());
    on_pred!("smooth_l1_sum", |p: &Tensor| smooth_l1_loss(p, &t, 0.7, Reduction::Sum).unwrap());
    on_pred!("bce_logits", |p: &Tensor| binary_cross_entropy_with_logits(p, &labels01).unwrap());
    on_pred!("multilabel_soft_margin", |p: &Tensor| multilabel_soft_margin_loss(p, &labels01, None, Reduction::Mean).unwrap());
    let ml_t = m(&[1.0, 2.0, -1.0, 0.0, 2.0, -1.0], &[2, 3]);
    on_pred!("multilabel_margin", |p: &Tensor| multilabel_margin_loss(p, &ml_t).unwrap());

    let probs = [0.1, 0.7, 0.35, 0.92, 0.5, 0.03];
    let pr = leaf(&probs, &[2, 3]);
    emit("bce", &binary_cross_entropy(&pr, &labels01, None, Reduction::Mean).unwrap(), &[("g", &pr)]);

    let lg_v = [1.2, -0.3, 0.8, 2.0, 0.1, -1.5, 0.4, 0.4, 1.1];
    let cls = [2usize, 0, 1];
    let lg = leaf(&lg_v, &[3, 3]);
    emit("ce_smooth", &cross_entropy_with_smoothing(&lg, &cls, 0.1).unwrap(), &[("g", &lg)]);
    let lg = leaf(&lg_v, &[3, 3]);
    emit("nll", &nll_loss(&lg.log_softmax(1).unwrap(), &cls).unwrap(), &[("g", &lg)]);
    let tp = m(&[0.2, 0.5, 0.3, 0.1, 0.1, 0.8, 0.6, 0.3, 0.1], &[3, 3]);
    let lg = leaf(&lg_v, &[3, 3]);
    emit("kl_mean", &kl_div_r(&lg.log_softmax(1).unwrap(), &tp, KlReduction::Mean, false).unwrap(), &[("g", &lg)]);
    let lg = leaf(&lg_v, &[3, 3]);
    emit("kl_batchmean", &kl_div_r(&lg.log_softmax(1).unwrap(), &tp, KlReduction::BatchMean, false).unwrap(), &[("g", &lg)]);
    let mm_t = Tensor::from_slice(&[2i64, 0, 1], &[3]).unwrap();
    let lg = leaf(&lg_v, &[3, 3]);
    emit("multi_margin1", &multi_margin_loss(&lg, &mm_t, 1, 1.0, None, Reduction::Mean).unwrap(), &[("g", &lg)]);
    let lg = leaf(&lg_v, &[3, 3]);
    emit("multi_margin2", &multi_margin_loss(&lg, &mm_t, 2, 0.8, None, Reduction::Mean).unwrap(), &[("g", &lg)]);

    let zl = leaf(&[2.0, 1.0, 0.5, 0.0, 1.0, 3.0], &[2, 3]);
    let zt = Tensor::from_slice(&[0i64, 2], &[2]).unwrap();
    emit("multi_margin_zero", &multi_margin_loss(&zl, &zt, 1, 1.0, None, Reduction::Mean).unwrap(), &[("g", &zl)]);
    let (av, bv, cv) = ([0.3, 1.1, -0.4, 0.9, -0.2, 0.7], [0.1, 0.8, -0.1, -0.5, 0.6, 0.2], [1.0, -0.4, 0.3, 0.2, 0.5, -0.9]);
    let (a, b) = (leaf(&av, &[2, 3]), leaf(&bv, &[2, 3]));
    emit("cosine_embedding", &cosine_embedding_loss(&a, &b, &m(&[1.0, -1.0], &[2]), 0.2).unwrap(), &[("ga", &a), ("gb", &b)]);
    let (a, b, c) = (leaf(&av, &[2, 3]), leaf(&bv, &[2, 3]), leaf(&cv, &[2, 3]));
    emit("triplet", &triplet_margin_loss(&a, &b, &c, 1.0).unwrap(), &[("ga", &a), ("gb", &b), ("gc", &c)]);

    let (x1v, x2v) = ([0.4, 1.2, -0.3, 0.8], [0.1, 1.5, 0.2, -0.4]);
    let y = m(&[1.0, -1.0, 1.0, -1.0], &[4]);
    let (x1, x2) = (leaf(&x1v, &[4]), leaf(&x2v, &[4]));
    emit("margin_ranking", &margin_ranking_loss(&x1, &x2, &y, 0.3).unwrap(), &[("g1", &x1), ("g2", &x2)]);
    let x1 = leaf(&x1v, &[4]);
    emit("hinge_embedding", &hinge_embedding_loss(&x1, &y, 0.9).unwrap(), &[("g", &x1)]);
    let x1 = leaf(&x1v, &[4]);
    emit("soft_margin", &soft_margin_loss(&x1, &y).unwrap(), &[("g", &x1)]);

    let counts = m(&[0.0, 2.0, 5.0, 1.0], &[4]);
    let rate_v = [0.4, 1.1, 1.6, -0.3];
    let rate = leaf(&rate_v, &[4]);
    emit("poisson_log", &poisson_nll_loss(&rate, &counts, true, false, 1e-8, Reduction::Mean).unwrap(), &[("g", &rate)]);
    let rate = leaf(&rate_v, &[4]);
    emit("poisson_full", &poisson_nll_loss(&rate, &counts, true, true, 1e-8, Reduction::Mean).unwrap(), &[("g", &rate)]);
    let var_v = [0.5, 1.2, 0.8, 2.0];
    let (x1, var) = (leaf(&x1v, &[4]), leaf(&var_v, &[4]));
    emit("gaussian_nll", &gaussian_nll_loss(&x1, &m(&x2v, &[4]), &var, false, 1e-6, Reduction::Mean).unwrap(), &[("gx", &x1), ("gv", &var)]);

    // CTC gradient w.r.t. the log-probabilities (T = 6, N = 2, C = 4, blank = 0)
    let raw: Vec<f64> = (0..6 * 2 * 4).map(|i| ((i * 7 % 11) as f64 - 5.0) / 3.0).collect();
    let rawt = leaf(&raw, &[6, 2, 4]);
    let tg = Tensor::from_slice(&[1i64, 2, 3, 2, 2, 1], &[2, 3]).unwrap();
    emit("ctc", &ctc_loss(&rawt.log_softmax(2).unwrap(), &tg, 0, false).unwrap(), &[("g", &rawt)]);
}
