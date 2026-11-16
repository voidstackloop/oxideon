//! Sixth cross-validation batch: distribution log-probabilities, entropies and KL divergences vs torch.distributions.
//! Run: `cargo test -p oxideon-core --test crossval_emit6 -- --nocapture --ignored`.

use oxideon_core::distributions::{Laplace, Normal};
use oxideon_core::distributions_ext::*;
use oxideon_core::distributions_transforms::*;
use oxideon_core::tensor::Tensor;

fn t(v: &[f64]) -> Tensor {
    Tensor::from_slice(v, &[v.len()]).unwrap()
}

fn emit(name: &str, v: &Tensor) {
    v.to_vec::<f64>().unwrap().iter().enumerate().for_each(|(i, x)| println!("{name}_{i} {x:.12e}"));
}

fn both<D: Distribution>(name: &str, d: &D, xs: &[f64]) {
    emit(&format!("lp_{name}"), &d.log_prob(&t(xs)).unwrap());
    if let Ok(e) = d.entropy() {
        emit(&format!("ent_{name}"), &e);
    }
}

#[test]
#[ignore]
fn emit_distributions() {
    both("normal", &Normal::new(t(&[0.3]), t(&[1.7])).unwrap(), &[-2.0, 0.0, 0.3, 1.5, 4.0]);
    both("laplace", &Laplace::new(t(&[-0.4]), t(&[0.8])).unwrap(), &[-3.0, -0.4, 0.5, 2.0]);
    both("gumbel", &Gumbel::new(t(&[0.5]), t(&[1.3])).unwrap(), &[-2.0, 0.0, 0.5, 3.0, 6.0]);
    both("weibull", &Weibull::new(t(&[1.8]), t(&[2.3])).unwrap(), &[0.1, 0.9, 1.8, 3.5]);
    both("pareto", &Pareto::new(t(&[1.2]), t(&[2.7])).unwrap(), &[1.3, 2.0, 5.0, 9.0]);
    both("logistic", &Logistic::new(t(&[0.2]), t(&[0.9])).unwrap(), &[-4.0, 0.0, 0.2, 2.5]);
    both("cauchy", &Cauchy::new(t(&[-0.5]), t(&[1.4])).unwrap(), &[-9.0, -0.5, 1.0, 20.0]);
    both("studentt", &StudentT::new(t(&[4.5]), t(&[0.7]), t(&[1.3])).unwrap(), &[-3.0, 0.7, 2.2, 8.0]);
    both("lognormal", &LogNormal::new(t(&[0.1]), t(&[0.6])).unwrap(), &[0.2, 0.9, 1.4, 5.0]);
    both("halfnormal", &HalfNormal::new(t(&[1.5])).unwrap(), &[0.1, 0.8, 2.0, 4.5]);
    both("exponential", &Exponential::new(t(&[1.3])).unwrap(), &[0.05, 0.7, 2.5, 6.0]);
    both("gamma", &Gamma::new(t(&[2.4]), t(&[1.6])).unwrap(), &[0.1, 0.9, 2.0, 6.5]);
    both("chi2", &Chi2::new(t(&[3.5])).unwrap(), &[0.3, 1.5, 4.0, 9.0]);
    both("beta", &Beta::new(t(&[2.2]), t(&[3.1])).unwrap(), &[0.05, 0.3, 0.6, 0.95]);
    both("fisher", &FisherSnedecor::new(t(&[4.0]), t(&[7.0])).unwrap(), &[0.2, 0.9, 2.5, 6.0]);
    both("uniform", &Uniform::new(t(&[-1.0]), t(&[2.5])).unwrap(), &[-0.9, 0.0, 1.7, 2.4]);
    // discrete
    both("poisson", &Poisson::new(t(&[3.4])).unwrap(), &[0.0, 1.0, 3.0, 7.0]);
    both("geometric", &Geometric::new(t(&[0.3])).unwrap(), &[0.0, 1.0, 4.0, 9.0]);
    both("binomial", &Binomial::new(t(&[12.0]), &t(&[0.35])).unwrap(), &[0.0, 3.0, 6.0, 12.0]);
    both("negbin", &NegativeBinomial::new(t(&[4.0]), t(&[0.4])).unwrap(), &[0.0, 2.0, 5.0, 11.0]);
    both("bernoulli", &Bernoulli::from_probs(&t(&[0.27])).unwrap(), &[0.0, 1.0]);
    // KL divergences
    emit("kl_bernoulli", &Bernoulli::from_probs(&t(&[0.27])).unwrap().kl_divergence(&Bernoulli::from_probs(&t(&[0.6])).unwrap()).unwrap());
    emit("kl_poisson", &Poisson::new(t(&[3.4])).unwrap().kl_divergence(&Poisson::new(t(&[5.1])).unwrap()).unwrap());
    emit("kl_uniform", &Uniform::new(t(&[0.0]), t(&[1.0])).unwrap().kl_divergence(&Uniform::new(t(&[-1.0]), t(&[3.0])).unwrap()).unwrap());
    emit("kl_exponential", &Exponential::new(t(&[1.3])).unwrap().kl_divergence(&Exponential::new(t(&[0.6])).unwrap()).unwrap());
    emit("kl_gamma", &Gamma::new(t(&[2.4]), t(&[1.6])).unwrap().kl_divergence(&Gamma::new(t(&[3.1]), t(&[0.9])).unwrap()).unwrap());
    emit("kl_beta", &Beta::new(t(&[2.2]), t(&[3.1])).unwrap().kl_divergence(&Beta::new(t(&[1.4]), t(&[2.0])).unwrap()).unwrap());
    emit("kl_dirichlet", &Dirichlet::new(t(&[1.5, 2.5, 0.8])).unwrap().kl_divergence(&Dirichlet::new(t(&[2.0, 1.0, 1.2])).unwrap()).unwrap());
    emit("kl_cauchy", &Cauchy::new(t(&[0.0]), t(&[1.0])).unwrap().kl_divergence(&Cauchy::new(t(&[1.0]), t(&[2.5])).unwrap()).unwrap());
    // Dirichlet / MVN
    let dir = Dirichlet::new(t(&[1.5, 2.5, 0.8])).unwrap();
    emit("lp_dirichlet", &dir.log_prob(&t(&[0.2, 0.5, 0.3])).unwrap());
    emit("ent_dirichlet", &dir.entropy().unwrap());
    let cov = Tensor::from_slice(&[2.0f64, 0.5, 0.2, 0.5, 1.0, -0.3, 0.2, -0.3, 1.5], &[3, 3]).unwrap();
    let mvn = MultivariateNormal::from_covariance(t(&[0.1, -0.2, 0.4]), &cov).unwrap();
    emit("lp_mvn", &mvn.log_prob(&Tensor::from_slice(&[0.5f64, 0.3, -0.1], &[1, 3]).unwrap()).unwrap());
    emit("ent_mvn", &mvn.entropy().unwrap());
}
