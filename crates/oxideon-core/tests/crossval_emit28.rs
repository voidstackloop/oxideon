//! Cross-validation batch 28: gradients of distribution log-probabilities and entropies w.r.t. their parameters
//! vs torch.distributions (tools/crossval/check_batch28.py).
use oxideon_core::distributions::{Laplace, Normal};
use oxideon_core::distributions_ext::*;
use oxideon_core::distributions_transforms::*;
use oxideon_core::tensor::Tensor;

fn t(v: &[f64]) -> Tensor {
    Tensor::from_slice(v, &[v.len()]).unwrap()
}

fn leaf(v: f64) -> Tensor {
    t(&[v]).requires_grad_(true)
}

/// Emits d/dparam of sum(log_prob(xs) * w) as `{name}_g{k}` and of sum(entropy) as `{name}_e{k}`.
fn grads<D: Distribution>(name: &str, d: &D, xs: &[f64], params: &[&Tensor]) {
    let w: Vec<f64> = (0..xs.len()).map(|i| 0.6 + 0.45 * i as f64).collect();
    let dump = |tag: &str, r: oxideon_core::error::Result<()>| match r {
        Ok(()) => {
            for (k, p) in params.iter().enumerate() {
                match p.grad() {
                    Some(g) => println!("{name}_{tag}{k} {:.13e}", g.to_vec::<f64>().unwrap()[0]),
                    None => println!("{name}_{tag}{k} NOGRAD"),
                }
                p.zero_grad();
            }
        }
        Err(e) => println!("{name}_{tag}0 NOGRAPH {e:?}"),
    };
    dump("g", d.log_prob(&t(xs)).and_then(|lp| lp.mul(&t(&w))?.sum()?.backward()));
    if let Ok(e) = d.entropy() {
        dump("e", e.sum().and_then(|s| s.backward()));
    }
}

#[test]
#[ignore]
fn emit_distribution_grads() {
    let (a, b) = (leaf(0.3), leaf(1.7));
    grads("normal", &Normal::new(a.clone(), b.clone()).unwrap(), &[-2.0, 0.0, 0.3, 1.5, 4.0], &[&a, &b]);
    let (a, b) = (leaf(-0.4), leaf(0.8));
    grads("laplace", &Laplace::new(a.clone(), b.clone()).unwrap(), &[-3.0, -0.1, 0.5, 2.0], &[&a, &b]);
    let (a, b) = (leaf(0.5), leaf(1.3));
    grads("gumbel", &Gumbel::new(a.clone(), b.clone()).unwrap(), &[-2.0, 0.0, 0.5, 3.0, 6.0], &[&a, &b]);
    let (a, b) = (leaf(1.8), leaf(2.3));
    grads("weibull", &Weibull::new(a.clone(), b.clone()).unwrap(), &[0.1, 0.9, 1.8, 3.5], &[&a, &b]);
    let (a, b) = (leaf(1.2), leaf(2.7));
    grads("pareto", &Pareto::new(a.clone(), b.clone()).unwrap(), &[1.3, 2.0, 5.0, 9.0], &[&a, &b]);
    let (a, b) = (leaf(0.2), leaf(0.9));
    grads("logistic", &Logistic::new(a.clone(), b.clone()).unwrap(), &[-4.0, 0.0, 0.2, 2.5], &[&a, &b]);
    let (a, b) = (leaf(-0.5), leaf(1.4));
    grads("cauchy", &Cauchy::new(a.clone(), b.clone()).unwrap(), &[-9.0, -0.5, 1.0, 20.0], &[&a, &b]);
    let (a, b, c) = (leaf(4.5), leaf(0.7), leaf(1.3));
    grads("studentt", &StudentT::new(a.clone(), b.clone(), c.clone()).unwrap(), &[-3.0, 0.7, 2.2, 8.0], &[&a, &b, &c]);
    let (a, b) = (leaf(0.1), leaf(0.6));
    grads("lognormal", &LogNormal::new(a.clone(), b.clone()).unwrap(), &[0.2, 0.9, 1.4, 5.0], &[&a, &b]);
    let a = leaf(1.5);
    grads("halfnormal", &HalfNormal::new(a.clone()).unwrap(), &[0.1, 0.8, 2.0, 4.5], &[&a]);
    let a = leaf(1.3);
    grads("exponential", &Exponential::new(a.clone()).unwrap(), &[0.05, 0.7, 2.5, 6.0], &[&a]);
    let (a, b) = (leaf(2.4), leaf(1.6));
    grads("gamma", &Gamma::new(a.clone(), b.clone()).unwrap(), &[0.1, 0.9, 2.0, 6.5], &[&a, &b]);
    let a = leaf(3.5);
    grads("chi2", &Chi2::new(a.clone()).unwrap(), &[0.3, 1.5, 4.0, 9.0], &[&a]);
    let (a, b) = (leaf(2.2), leaf(3.1));
    grads("beta", &Beta::new(a.clone(), b.clone()).unwrap(), &[0.05, 0.3, 0.6, 0.95], &[&a, &b]);
    let (a, b) = (leaf(4.0), leaf(7.0));
    grads("fisher", &FisherSnedecor::new(a.clone(), b.clone()).unwrap(), &[0.2, 0.9, 2.5, 6.0], &[&a, &b]);
    let (a, b) = (leaf(-1.0), leaf(2.5));
    grads("uniform", &Uniform::new(a.clone(), b.clone()).unwrap(), &[-0.9, 0.0, 1.7, 2.4], &[&a, &b]);
    let a = leaf(3.4);
    grads("poisson", &Poisson::new(a.clone()).unwrap(), &[0.0, 1.0, 3.0, 7.0], &[&a]);
    let a = leaf(0.3);
    grads("geometric", &Geometric::new(a.clone()).unwrap(), &[0.0, 1.0, 4.0, 9.0], &[&a]);
    let a = leaf(0.35);
    grads("binomial", &Binomial::new(t(&[12.0]), &a).unwrap(), &[0.0, 3.0, 6.0, 12.0], &[&a]);
    let a = leaf(0.4);
    grads("negbin", &NegativeBinomial::new(t(&[4.0]), a.clone()).unwrap(), &[0.0, 2.0, 5.0, 11.0], &[&a]);
    let a = leaf(0.27);
    grads("bernoulli", &Bernoulli::from_probs(&a).unwrap(), &[0.0, 1.0], &[&a]);

    let (a, b) = (leaf(1.7), leaf(2.4));
    grads("kumaraswamy", &Kumaraswamy::new(a.clone(), b.clone()).unwrap(), &[0.1, 0.4, 0.7, 0.95], &[&a, &b]);

    // vector / matrix parameters
    let vdump = |name: &str, r: oxideon_core::error::Result<()>, params: &[&Tensor]| match r {
        Ok(()) => {
            for (k, p) in params.iter().enumerate() {
                match p.grad() {
                    Some(g) => println!("{name}{k} {}", g.to_vec::<f64>().unwrap().iter().map(|v| format!("{v:.13e}")).collect::<Vec<_>>().join(" ")),
                    None => println!("{name}{k} NOGRAD"),
                }
                p.zero_grad();
            }
        }
        Err(e) => println!("{name}0 NOGRAPH {e:?}"),
    };
    let conc = t(&[1.5, 2.5, 0.8]).requires_grad_(true);
    let dir = Dirichlet::new(conc.clone()).unwrap();
    vdump("dirichlet_g", dir.log_prob(&t(&[0.2, 0.5, 0.3])).and_then(|l| l.sum()?.backward()), &[&conc]);
    vdump("dirichlet_e", dir.entropy().and_then(|l| l.sum()?.backward()), &[&conc]);

    let x = Tensor::from_slice(&[0.5f64, 0.3, -0.1], &[1, 3]).unwrap();
    let loc = t(&[0.1, -0.2, 0.4]).requires_grad_(true);
    let cov = Tensor::from_slice(&[2.0f64, 0.5, 0.2, 0.5, 1.0, -0.3, 0.2, -0.3, 1.5], &[3, 3]).unwrap().requires_grad_(true);
    let mvn = MultivariateNormal::from_covariance(loc.clone(), &cov).unwrap();
    vdump("mvncov_g", mvn.log_prob(&x).and_then(|l| l.sum()?.backward()), &[&loc, &cov]);
    vdump("mvncov_e", mvn.entropy().and_then(|l| l.sum()?.backward()), &[&loc, &cov]);
    let loc = t(&[0.1, -0.2, 0.4]).requires_grad_(true);
    let tril = Tensor::from_slice(&[1.2f64, 0.0, 0.0, 0.3, 0.9, 0.0, -0.2, 0.4, 1.1], &[3, 3]).unwrap().requires_grad_(true);
    let mvn = MultivariateNormal::from_scale_tril(loc.clone(), tril.clone()).unwrap();
    vdump("mvntril_g", mvn.log_prob(&x).and_then(|l| l.sum()?.backward()), &[&loc, &tril]);
    vdump("mvntril_e", mvn.entropy().and_then(|l| l.sum()?.backward()), &[&loc, &tril]);

    let logits = t(&[0.2, -1.0, 0.7, 0.1]).requires_grad_(true);
    let cat = oxideon_core::distributions::Categorical::from_logits(logits.clone());
    vdump("cat_g", cat.log_probs().and_then(|l| l.mul(&t(&[0.6, 1.05, 1.5, 1.95]))?.sum()?.backward()), &[&logits]);

    let probs = t(&[0.2, 0.5, 0.3]).requires_grad_(true);
    let mult = Multinomial::new(6, probs.clone()).unwrap();
    vdump("multinomial_g", mult.log_prob(&t(&[1.0, 3.0, 2.0])).and_then(|l| l.sum()?.backward()), &[&probs]);

    // KL divergences w.r.t. the parameters of p
    let kl = |name: &str, v: oxideon_core::error::Result<Tensor>, params: &[&Tensor]| match v.and_then(|k| k.sum()?.backward()) {
        Ok(()) => params.iter().enumerate().for_each(|(k, p)| match p.grad() {
            Some(g) => println!("{name}_g{k} {:.13e}", g.to_vec::<f64>().unwrap()[0]),
            None => println!("{name}_g{k} NOGRAD"),
        }),
        Err(e) => println!("{name}_g0 NOGRAPH {e:?}"),
    };
    let (a, b) = (leaf(2.4), leaf(1.6));
    kl("kl_gamma", Gamma::new(a.clone(), b.clone()).unwrap().kl_divergence(&Gamma::new(t(&[3.1]), t(&[0.9])).unwrap()), &[&a, &b]);
    let (a, b) = (leaf(2.2), leaf(3.1));
    kl("kl_beta", Beta::new(a.clone(), b.clone()).unwrap().kl_divergence(&Beta::new(t(&[1.4]), t(&[2.0])).unwrap()), &[&a, &b]);
    let a = leaf(1.3);
    kl("kl_exponential", Exponential::new(a.clone()).unwrap().kl_divergence(&Exponential::new(t(&[0.6])).unwrap()), &[&a]);
    let a = leaf(3.4);
    kl("kl_poisson", Poisson::new(a.clone()).unwrap().kl_divergence(&Poisson::new(t(&[5.1])).unwrap()), &[&a]);
    let (a, b) = (leaf(0.0), leaf(1.0));
    kl("kl_cauchy", Cauchy::new(a.clone(), b.clone()).unwrap().kl_divergence(&Cauchy::new(t(&[1.0]), t(&[2.5])).unwrap()), &[&a, &b]);
}
