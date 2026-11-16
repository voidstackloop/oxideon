"""Sixth cross-validation batch: distributions (log_prob, entropy, KL) vs torch.distributions.

    cargo test -p oxideon-core --test crossval_emit6 -- --nocapture --ignored > rust_values6.txt
    py -3.11 tools/crossval/check_batch6.py rust_values6.txt
"""
import sys

import torch
from torch import distributions as D

rust = {}
for line in open(sys.argv[1], encoding="utf-8"):
    parts = line.split()
    if len(parts) == 2:
        try:
            rust[parts[0]] = float(parts[1])
        except ValueError:
            pass

T = lambda *v: torch.tensor(v, dtype=torch.float64)
cases = {
    "normal": (D.Normal(T(0.3), T(1.7)), [-2.0, 0.0, 0.3, 1.5, 4.0]),
    "laplace": (D.Laplace(T(-0.4), T(0.8)), [-3.0, -0.4, 0.5, 2.0]),
    "gumbel": (D.Gumbel(T(0.5), T(1.3)), [-2.0, 0.0, 0.5, 3.0, 6.0]),
    "weibull": (D.Weibull(T(1.8), T(2.3)), [0.1, 0.9, 1.8, 3.5]),
    "pareto": (D.Pareto(T(1.2), T(2.7)), [1.3, 2.0, 5.0, 9.0]),
    "logistic": (D.TransformedDistribution(D.Uniform(T(0.0), T(1.0)), [D.SigmoidTransform().inv, D.AffineTransform(T(0.2), T(0.9))]), [-4.0, 0.0, 0.2, 2.5]),
    "cauchy": (D.Cauchy(T(-0.5), T(1.4)), [-9.0, -0.5, 1.0, 20.0]),
    "studentt": (D.StudentT(T(4.5), T(0.7), T(1.3)), [-3.0, 0.7, 2.2, 8.0]),
    "lognormal": (D.LogNormal(T(0.1), T(0.6)), [0.2, 0.9, 1.4, 5.0]),
    "halfnormal": (D.HalfNormal(T(1.5)), [0.1, 0.8, 2.0, 4.5]),
    "exponential": (D.Exponential(T(1.3)), [0.05, 0.7, 2.5, 6.0]),
    "gamma": (D.Gamma(T(2.4), T(1.6)), [0.1, 0.9, 2.0, 6.5]),
    "chi2": (D.Chi2(T(3.5)), [0.3, 1.5, 4.0, 9.0]),
    "beta": (D.Beta(T(2.2), T(3.1)), [0.05, 0.3, 0.6, 0.95]),
    "fisher": (D.FisherSnedecor(T(4.0), T(7.0)), [0.2, 0.9, 2.5, 6.0]),
    "uniform": (D.Uniform(T(-1.0), T(2.5)), [-0.9, 0.0, 1.7, 2.4]),
    "poisson": (D.Poisson(T(3.4)), [0.0, 1.0, 3.0, 7.0]),
    "geometric": (D.Geometric(probs=T(0.3)), [0.0, 1.0, 4.0, 9.0]),
    "binomial": (D.Binomial(12, probs=T(0.35)), [0.0, 3.0, 6.0, 12.0]),
    "negbin": (D.NegativeBinomial(T(4.0), probs=T(0.4)), [0.0, 2.0, 5.0, 11.0]),
    "bernoulli": (D.Bernoulli(probs=T(0.27)), [0.0, 1.0]),
}
ref = {}
for name, (dist, xs) in cases.items():
    lp = dist.log_prob(torch.tensor(xs, dtype=torch.float64))
    for i, v in enumerate(lp.tolist()): ref[f"lp_{name}_{i}"] = v
    try:
        ref[f"ent_{name}_0"] = float(dist.entropy().reshape(-1)[0])
    except NotImplementedError:
        pass  # torch has no closed-form entropy for this family
pairs = {
    "kl_bernoulli": (D.Bernoulli(probs=T(0.27)), D.Bernoulli(probs=T(0.6))),
    "kl_poisson": (D.Poisson(T(3.4)), D.Poisson(T(5.1))),
    "kl_uniform": (D.Uniform(T(0.0), T(1.0)), D.Uniform(T(-1.0), T(3.0))),
    "kl_exponential": (D.Exponential(T(1.3)), D.Exponential(T(0.6))),
    "kl_gamma": (D.Gamma(T(2.4), T(1.6)), D.Gamma(T(3.1), T(0.9))),
    "kl_beta": (D.Beta(T(2.2), T(3.1)), D.Beta(T(1.4), T(2.0))),
    "kl_dirichlet": (D.Dirichlet(T(1.5, 2.5, 0.8)), D.Dirichlet(T(2.0, 1.0, 1.2))),
}
for name, (p, q) in pairs.items():
    ref[f"{name}_0"] = float(D.kl_divergence(p, q))
# Cauchy KL closed form (Chyzak & Nielsen): log(((g1+g2)^2 + (x1-x2)^2) / (4 g1 g2))
import math
ref["kl_cauchy_0"] = math.log(((1.0 + 2.5) ** 2 + (0.0 - 1.0) ** 2) / (4 * 1.0 * 2.5))
dr = D.Dirichlet(T(1.5, 2.5, 0.8))
ref["lp_dirichlet_0"] = float(dr.log_prob(T(0.2, 0.5, 0.3)))
ref["ent_dirichlet_0"] = float(dr.entropy())
cov = torch.tensor([[2.0, 0.5, 0.2], [0.5, 1.0, -0.3], [0.2, -0.3, 1.5]], dtype=torch.float64)
mv = D.MultivariateNormal(T(0.1, -0.2, 0.4), covariance_matrix=cov)
ref["lp_mvn_0"] = float(mv.log_prob(T(0.5, 0.3, -0.1)))
ref["ent_mvn_0"] = float(mv.entropy())

worst, bad, skipped = 0.0, [], 0
for k, want in ref.items():
    if k not in rust:
        bad.append((k, "missing", want)); continue
    err = abs(rust[k] - want) / max(1.0, abs(want))
    worst = max(worst, err)
    if err > 1e-6:
        bad.append((k, rust[k], want))
print(f"compared {len(ref)} values, worst relative error {worst:.2e}")
for k, got, want in bad:
    print(f"MISMATCH {k}: rust={got} torch={want}")
print(f"{len(bad)} mismatches")
sys.exit(1 if bad else 0)
