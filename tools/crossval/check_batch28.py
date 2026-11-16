"""Batch 28: gradients of distribution log-probs / entropies / KLs w.r.t. parameters vs torch.distributions."""
import re, sys, warnings
import torch
import torch.distributions as D
warnings.filterwarnings("ignore")

rust = {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.strip())
    k, _, rest = line.partition(" ")
    if rest: rust[k] = rest

P = lambda v: torch.tensor([v], dtype=torch.float64, requires_grad=True)
C = lambda v: torch.tensor([v], dtype=torch.float64)
bad = n = 0; worst = 0.0

def compare(name, k, want, tag="g"):
    global bad, n, worst
    key = f"{name}_{tag}{k}"; n += 1
    got = rust.get(key, "MISSING")
    if got.startswith(("NOGRAD", "NOGRAPH", "MISSING")):
        if abs(want) > 1e-12: bad += 1; print("NO GRADIENT", key, got[:60], "torch:", want)
        return
    got = float(got); err = abs(got - want) / max(1.0, abs(want)); worst = max(worst, err)
    if err > 1e-8: bad += 1; print("MISMATCH", key, got, want)

def grads(name, make, vals, xs):
    x = torch.tensor(xs, dtype=torch.float64)
    w = torch.tensor([0.6 + 0.45 * i for i in range(len(xs))], dtype=torch.float64)
    params = [P(v) for v in vals]
    (make(*params).log_prob(x) * w).sum().backward()
    for k, p in enumerate(params): compare(name, k, 0.0 if p.grad is None else p.grad.item())
    if f"{name}_e0" not in rust: return
    params = [P(v) for v in vals]
    try:
        # torch has no closed-form entropy for the transformed logistic: it is ln(scale) + 2
        ent = (params[1].log() + 2) if name == "logistic" else make(*params).entropy()
    except NotImplementedError:
        if name == "poisson":  # reference: central difference of scipy's Poisson entropy
            from scipy.stats import poisson
            h = 1e-5
            compare(name, 0, float(poisson(vals[0] + h).entropy() - poisson(vals[0] - h).entropy()) / (2 * h), "e")
        return
    if not ent.requires_grad: return
    ent.sum().backward()
    for k, p in enumerate(params): compare(name, k, 0.0 if p.grad is None else p.grad.item(), "e")

grads("normal", D.Normal, [0.3, 1.7], [-2.0, 0.0, 0.3, 1.5, 4.0])
grads("laplace", D.Laplace, [-0.4, 0.8], [-3.0, -0.1, 0.5, 2.0])
grads("gumbel", D.Gumbel, [0.5, 1.3], [-2.0, 0.0, 0.5, 3.0, 6.0])
grads("weibull", D.Weibull, [1.8, 2.3], [0.1, 0.9, 1.8, 3.5])
grads("pareto", D.Pareto, [1.2, 2.7], [1.3, 2.0, 5.0, 9.0])
grads("logistic", lambda loc, s: D.TransformedDistribution(D.Uniform(C(0.0), C(1.0)), [D.SigmoidTransform().inv, D.AffineTransform(loc, s)]), [0.2, 0.9], [-4.0, 0.0, 0.2, 2.5])
grads("cauchy", D.Cauchy, [-0.5, 1.4], [-9.0, -0.5, 1.0, 20.0])
grads("studentt", D.StudentT, [4.5, 0.7, 1.3], [-3.0, 0.7, 2.2, 8.0])
grads("lognormal", D.LogNormal, [0.1, 0.6], [0.2, 0.9, 1.4, 5.0])
grads("halfnormal", D.HalfNormal, [1.5], [0.1, 0.8, 2.0, 4.5])
grads("exponential", D.Exponential, [1.3], [0.05, 0.7, 2.5, 6.0])
grads("gamma", D.Gamma, [2.4, 1.6], [0.1, 0.9, 2.0, 6.5])
grads("chi2", D.Chi2, [3.5], [0.3, 1.5, 4.0, 9.0])
grads("beta", D.Beta, [2.2, 3.1], [0.05, 0.3, 0.6, 0.95])
grads("fisher", D.FisherSnedecor, [4.0, 7.0], [0.2, 0.9, 2.5, 6.0])
grads("uniform", D.Uniform, [-1.0, 2.5], [-0.9, 0.0, 1.7, 2.4])
grads("poisson", D.Poisson, [3.4], [0.0, 1.0, 3.0, 7.0])
grads("geometric", D.Geometric, [0.3], [0.0, 1.0, 4.0, 9.0])
grads("binomial", lambda p: D.Binomial(12, probs=p), [0.35], [0.0, 3.0, 6.0, 12.0])
grads("negbin", lambda p: D.NegativeBinomial(C(4.0), probs=p), [0.4], [0.0, 2.0, 5.0, 11.0])
grads("bernoulli", lambda p: D.Bernoulli(probs=p), [0.27], [0.0, 1.0])

grads("kumaraswamy", D.Kumaraswamy, [1.7, 2.4], [0.1, 0.4, 0.7, 0.95])

import numpy as np
def vcheck(key, want, sym=False):
    global bad, n, worst
    n += 1
    got = rust.get(key, "MISSING")
    if got.startswith(("NOGRAD", "NOGRAPH", "MISSING")):
        if want is not None and want.abs().max() > 1e-12: bad += 1; print("NO GRADIENT", key, got[:70])
        return
    g = np.array([float(v) for v in got.split()]); w = np.zeros_like(g) if want is None else want.detach().numpy().astype(float)
    if sym:
        g = g.reshape(w.shape); g = (g + g.T) / 2; w = (w + w.T) / 2
    err = np.abs(g.ravel() - w.ravel()).max() / max(1.0, np.abs(w).max()); worst = max(worst, err)
    if err > 1e-8: bad += 1; print("MISMATCH", key, g.ravel()[:4], w.ravel()[:4])
V = lambda v, shape=None: torch.tensor(v, dtype=torch.float64).reshape(shape if shape else (len(v),)).requires_grad_(True)

conc = V([1.5, 2.5, 0.8]); D.Dirichlet(conc).log_prob(torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64)).sum().backward(); vcheck("dirichlet_g0", conc.grad)
conc = V([1.5, 2.5, 0.8]); D.Dirichlet(conc).entropy().sum().backward(); vcheck("dirichlet_e0", conc.grad)
x = torch.tensor([[0.5, 0.3, -0.1]], dtype=torch.float64)
COV = [2.0, 0.5, 0.2, 0.5, 1.0, -0.3, 0.2, -0.3, 1.5]; TRIL = [1.2, 0.0, 0.0, 0.3, 0.9, 0.0, -0.2, 0.4, 1.1]
loc, cov = V([0.1, -0.2, 0.4]), V(COV, (3, 3)); D.MultivariateNormal(loc, covariance_matrix=cov).log_prob(x).sum().backward()
vcheck("mvncov_g0", loc.grad); vcheck("mvncov_g1", cov.grad, sym=True)
loc, cov = V([0.1, -0.2, 0.4]), V(COV, (3, 3)); D.MultivariateNormal(loc, covariance_matrix=cov).entropy().sum().backward()
vcheck("mvncov_e0", loc.grad); vcheck("mvncov_e1", cov.grad, sym=True)
loc, tr = V([0.1, -0.2, 0.4]), V(TRIL, (3, 3)); D.MultivariateNormal(loc, scale_tril=tr).log_prob(x).sum().backward()
vcheck("mvntril_g0", loc.grad); vcheck("mvntril_g1", tr.grad.tril())
loc, tr = V([0.1, -0.2, 0.4]), V(TRIL, (3, 3)); D.MultivariateNormal(loc, scale_tril=tr).entropy().sum().backward()
vcheck("mvntril_e0", loc.grad); vcheck("mvntril_e1", tr.grad.tril())
lg = V([0.2, -1.0, 0.7, 0.1]); (torch.log_softmax(lg, 0) * torch.tensor([0.6, 1.05, 1.5, 1.95], dtype=torch.float64)).sum().backward(); vcheck("cat_g0", lg.grad)
pr = V([0.2, 0.5, 0.3]); D.Multinomial(6, probs=pr).log_prob(torch.tensor([1.0, 3.0, 2.0], dtype=torch.float64)).sum().backward(); vcheck("multinomial_g0", pr.grad)

def kl(name, make, vals, q):
    params = [P(v) for v in vals]
    D.kl_divergence(make(*params), q).sum().backward()
    for k, p in enumerate(params): compare(name, k, p.grad.item())
kl("kl_gamma", D.Gamma, [2.4, 1.6], D.Gamma(C(3.1), C(0.9)))
kl("kl_beta", D.Beta, [2.2, 3.1], D.Beta(C(1.4), C(2.0)))
kl("kl_exponential", D.Exponential, [1.3], D.Exponential(C(0.6)))
kl("kl_poisson", D.Poisson, [3.4], D.Poisson(C(5.1)))
kl("kl_cauchy", D.Cauchy, [0.0, 1.0], D.Cauchy(C(1.0), C(2.5)))
print(f"{n} gradients compared, worst error {worst:.2e}, {bad} mismatches")
sys.exit(1 if bad else 0)
