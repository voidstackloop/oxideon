"""Tests of the distributions and transforms added to oxideon.distributions, and of the moments of the older ones.

The PyTorch comparisons are skipped where torch is not installed; the rest needs only NumPy."""
import math

import numpy as np
import pytest

import oxideon.distributions as D


def A(x):
    return np.asarray(x.to_numpy() if hasattr(x, "to_numpy") else x.detach().numpy() if hasattr(x, "detach") else x, dtype=np.float64)


def close(a, b, tol=1e-4):
    a, b = A(a), A(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    np.testing.assert_allclose(a, b, rtol=tol, atol=tol, equal_nan=True)


# ---- torch-free checks
@pytest.mark.parametrize("make", [lambda: D.Normal(0.0, 1.0), lambda: D.Gamma(2.0, 1.0), lambda: D.StudentT(3.0),
                                  lambda: D.HalfNormal(1.0), lambda: D.Pareto(1.0, 3.0), lambda: D.VonMises(0.0, 2.0),
                                  lambda: D.Kumaraswamy(2.0, 3.0), lambda: D.InverseGamma(3.0, 2.0)])
def test_scalar_parameters_give_scalar_samples_and_log_probs(make):
    d = make()
    s = d.sample()
    assert tuple(s.shape) == ()
    assert tuple(d.log_prob(s).shape) == ()
    assert tuple(d.sample((5,)).shape) == (5,)


def test_closed_forms():
    close(D.Pareto(2.0, 3.0).mean, 3.0)
    close(D.Pareto(2.0, 3.0).log_prob(4.0), math.log(3) + 3 * math.log(2) - 4 * math.log(4))
    close(D.Pareto(2.0, 3.0).log_prob(1.0), -np.inf)
    close(D.Chi2(4.0).mean, 4.0)
    close(D.Chi2(4.0).variance, 8.0)
    close(D.InverseGamma(3.0, 2.0).mean, 1.0)
    close(D.Kumaraswamy(1.0, 1.0).log_prob(0.3), 0.0)  # Kumaraswamy(1, 1) is uniform on (0, 1)
    close(D.NegativeBinomial(2.0, probs=0.5).log_prob(0.0), 2 * math.log(0.5))
    close(D.VonMises(0.0, 1e-9).log_prob(1.0), -math.log(2 * math.pi), 1e-6)  # concentration 0 is uniform on the circle
    close(D.Normal(1.0, 2.0).stddev, 2.0)
    with pytest.raises(NotImplementedError):
        D.Distribution().mean


def test_transform_round_trips():
    x = np.array([[-1.2, 0.3, 2.0], [0.5, -0.4, 0.1]], dtype=np.float32)
    for t in (D.ExpTransform(), D.SigmoidTransform(), D.TanhTransform(), D.SoftplusTransform(), D.AffineTransform(1.0, -3.0),
              D.StickBreakingTransform(), D.ComposeTransform([D.AffineTransform(0.5, 2.0), D.SigmoidTransform()])):
        close(t.inv(t(x)), x, 1e-4)
    y = A(D.StickBreakingTransform()(x))
    assert y.shape == (2, 4) and np.allclose(y.sum(-1), 1, atol=1e-6) and np.all(y > 0)
    c = A(D.CorrCholeskyTransform()(np.array([0.3, -0.7, 1.1])))
    assert np.allclose(np.linalg.norm(c, axis=-1), 1) and np.allclose(np.triu(c, 1), 0)


def test_transformed_distribution_log_prob_is_a_density():
    d = D.TransformedDistribution(D.Normal(0.0, 1.0), [D.ExpTransform()])  # a log-normal
    grid = np.linspace(1e-4, 60, 400001)
    p = np.exp(A(d.log_prob(grid)))
    assert abs(np.trapezoid(p, grid) - 1) < 1e-3 if hasattr(np, "trapezoid") else abs(np.trapz(p, grid) - 1) < 1e-3


def test_sampling_matches_moments():
    np.random.seed(0)
    for d in (D.Pareto(1.0, 4.0), D.Kumaraswamy(2.0, 3.0), D.FisherSnedecor(5.0, 12.0), D.NegativeBinomial(3.0, probs=0.4),
              D.GeneralizedPareto(0.0, 1.0, 0.2), D.InverseGamma(5.0, 2.0)):
        s = A(d.sample((60000,)))
        se = math.sqrt(float(A(d.variance))) / math.sqrt(60000)
        assert abs(s.mean() - float(A(d.mean))) < 6 * se, type(d).__name__


def test_mixture_and_one_hot():
    mix = D.MixtureSameFamily(D.Categorical(probs=np.array([0.25, 0.75])), D.Normal(np.array([-2.0, 2.0]), np.array([1.0, 1.0])))
    close(mix.mean, 1.0)
    close(mix.variance, 1.0 + 0.25 * 9 + 0.75 * 1)
    oh = D.OneHotCategorical(probs=np.array([0.2, 0.8]))
    s = A(oh.sample((1000,)))
    assert s.shape == (1000, 2) and np.all(s.sum(-1) == 1)
    assert tuple(oh.enumerate_support().shape) == (2, 2)


def test_wishart_and_lkj_samples_are_valid():
    w = A(D.Wishart(5.0, covariance_matrix=np.eye(3)).sample((200,)))
    assert np.allclose(w, np.swapaxes(w, -1, -2)) and np.all(np.linalg.eigvalsh(w) > 0)
    close(A(D.Wishart(5.0, covariance_matrix=np.eye(3)).mean), 5 * np.eye(3))
    with pytest.raises(ValueError):
        D.Wishart(1.0, covariance_matrix=np.eye(3))  # df must exceed p - 1
    L = A(D.LKJCholesky(4, 1.5).sample((100,)))
    assert np.allclose(np.linalg.norm(L, axis=-1), 1, atol=1e-6)


def test_straight_through_passes_gradients():
    import oxideon as ox
    p = ox.from_numpy(np.array([0.3, 0.7], dtype=np.float32)).requires_grad_(True)
    s = D.OneHotCategoricalStraightThrough(probs=p).rsample()
    np.testing.assert_allclose(sorted(A(s).tolist()), [0.0, 1.0], atol=1e-6)  # s + p - p.detach() rounds in float32, as in PyTorch
    (s * ox.from_numpy(np.array([1.0, 2.0], dtype=np.float32))).sum().backward()
    close(p.grad, [1.0, 2.0])


# ---- against PyTorch
torch = pytest.importorskip("torch")
TD = torch.distributions


def tt(x):
    return torch.tensor(np.asarray(x, dtype=np.float64))


C, R = np.array([2.5, 4.0, 7.0]), np.array([1.5, 0.5, 2.0])
P = np.array([[0.2, 0.5, 0.3], [0.6, 0.1, 0.3]])
CASES = [
    ("Chi2", lambda: D.Chi2(C), lambda: TD.Chi2(tt(C)), np.array([0.7, 3.0, 9.0]), ("mean", "variance", "entropy")),
    ("InverseGamma", lambda: D.InverseGamma(C, R), lambda: TD.InverseGamma(tt(C), tt(R)), np.array([0.3, 0.2, 0.5]), ("mean", "variance", "entropy")),
    ("Pareto", lambda: D.Pareto(R, C), lambda: TD.Pareto(tt(R), tt(C)), np.array([2.0, 0.9, 3.5]), ("mean", "variance", "entropy", "cdf")),
    ("Kumaraswamy", lambda: D.Kumaraswamy(C, R), lambda: TD.Kumaraswamy(tt(C), tt(R)), np.array([0.3, 0.7, 0.9]), ("mean", "variance", "entropy")),
    ("FisherSnedecor", lambda: D.FisherSnedecor(C, C + 4), lambda: TD.FisherSnedecor(tt(C), tt(C + 4)), np.array([0.5, 1.2, 3.0]), ("mean", "variance")),
    ("NegativeBinomial", lambda: D.NegativeBinomial(C, probs=np.array([0.3, 0.6, 0.1])),
     lambda: TD.NegativeBinomial(tt(C), probs=tt([0.3, 0.6, 0.1])), np.array([0.0, 4.0, 2.0]), ("mean", "variance")),
    ("OneHotCategorical", lambda: D.OneHotCategorical(probs=P), lambda: TD.OneHotCategorical(probs=tt(P)),
     np.array([[0, 1, 0], [1, 0, 0]], dtype=np.float64), ("mean", "variance", "entropy")),
    ("RelaxedBernoulli", lambda: D.RelaxedBernoulli(0.7, probs=np.array([0.3, 0.8])),
     lambda: TD.RelaxedBernoulli(tt(0.7), probs=tt([0.3, 0.8])), np.array([0.2, 0.9]), ()),
    ("RelaxedOneHotCategorical", lambda: D.RelaxedOneHotCategorical(0.5, probs=P), lambda: TD.RelaxedOneHotCategorical(tt(0.5), probs=tt(P)),
     np.array([[0.1, 0.7, 0.2], [0.5, 0.25, 0.25]]), ()),
    ("LogisticNormal", lambda: D.LogisticNormal(np.array([0.3, -0.2]), np.array([0.8, 1.2])),
     lambda: TD.LogisticNormal(tt([0.3, -0.2]), tt([0.8, 1.2])), np.array([0.2, 0.5, 0.3]), ()),
    ("MixtureSameFamily", lambda: D.MixtureSameFamily(D.Categorical(probs=np.array([0.3, 0.7])), D.Normal(np.array([-1.0, 2.0]), np.array([0.5, 1.5]))),
     lambda: TD.MixtureSameFamily(TD.Categorical(probs=tt([0.3, 0.7])), TD.Normal(tt([-1.0, 2.0]), tt([0.5, 1.5]))),
     np.array([0.0, 1.5, -2.0]), ("mean", "variance")),
    ("VonMises", lambda: D.VonMises(np.array([0.5, -1.0, 2.0]), np.array([0.3, 2.0, 20.0])),
     lambda: TD.VonMises(tt([0.5, -1.0, 2.0]), tt([0.3, 2.0, 20.0])), np.array([0.1, -0.8, 2.5]), ("mean", "variance")),
    ("Wishart", lambda: D.Wishart(6.0, covariance_matrix=np.array([[2.0, 0.3, 0.1], [0.3, 1.0, -0.2], [0.1, -0.2, 0.5]])),
     lambda: TD.Wishart(tt(6.0), covariance_matrix=tt([[2.0, 0.3, 0.1], [0.3, 1.0, -0.2], [0.1, -0.2, 0.5]])),
     np.array([[3.0, 0.5, 0.2], [0.5, 2.0, 0.1], [0.2, 0.1, 1.5]]), ("mean", "variance", "entropy")),
]


@pytest.mark.parametrize("name,make_o,make_t,values,props", CASES, ids=[c[0] for c in CASES])
def test_against_torch(name, make_o, make_t, values, props):
    od, td = make_o(), make_t()
    close(od.log_prob(values), td.log_prob(tt(values)))
    for p in props:
        if p == "cdf":
            close(od.cdf(values), td.cdf(tt(values)))
        elif p == "entropy":
            close(od.entropy(), td.entropy())
        else:
            close(getattr(od, p), getattr(td, p))


def test_generalized_pareto_against_torch():
    c = np.array([0.2, 0.0, -0.3])
    od = D.GeneralizedPareto(0.5, 2.0, c)
    td = TD.GeneralizedPareto(torch.tensor(0.5), torch.tensor(2.0), torch.tensor(c, dtype=torch.float32))
    v = np.array([1.0, 3.0, 2.5])
    close(od.log_prob(v), td.log_prob(torch.tensor(v, dtype=torch.float32)))
    close(od.cdf(v), td.cdf(torch.tensor(v, dtype=torch.float32)))
    for p in ("mean", "variance"):
        close(getattr(od, p), getattr(td, p))
    close(od.entropy(), td.entropy())


def test_lkj_cholesky_log_prob_against_torch():
    lkj = TD.LKJCholesky(3, tt(1.7))
    L = lkj.sample().numpy()
    close(D.LKJCholesky(3, 1.7).log_prob(L), lkj.log_prob(tt(L)))


@pytest.mark.parametrize("name,make", [
    ("Normal", lambda m: m.Normal(*(tt(1.0), tt(2.0)) if m is TD else (1.0, 2.0))),
    ("Gamma", lambda m: m.Gamma(*(tt(2.0), tt(3.0)) if m is TD else (2.0, 3.0))),
    ("Beta", lambda m: m.Beta(*(tt(2.0), tt(3.0)) if m is TD else (2.0, 3.0))),
    ("Gumbel", lambda m: m.Gumbel(*(tt(1.0), tt(2.0)) if m is TD else (1.0, 2.0))),
    ("LogNormal", lambda m: m.LogNormal(*(tt(0.2), tt(0.5)) if m is TD else (0.2, 0.5))),
    ("StudentT", lambda m: m.StudentT(*(tt(5.0), tt(1.0), tt(2.0)) if m is TD else (5.0, 1.0, 2.0))),
    ("Weibull", lambda m: m.Weibull(*(tt(2.0), tt(1.5)) if m is TD else (2.0, 1.5))),
    ("Geometric", lambda m: m.Geometric(tt(0.3) if m is TD else 0.3)),
])
def test_moments_of_existing_distributions(name, make):
    od, td = make(D), make(TD)
    for p in ("mean", "variance", "stddev"):
        close(getattr(od, p), getattr(td, p))


@pytest.mark.parametrize("name", ["Exp", "Affine", "Sigmoid", "Tanh", "Softplus", "StickBreaking", "Compose"])
def test_transforms_against_torch(name):
    make = {
        "Exp": lambda m: m.ExpTransform(), "Affine": lambda m: m.AffineTransform(1.5, -2.0), "Sigmoid": lambda m: m.SigmoidTransform(),
        "Tanh": lambda m: m.TanhTransform(), "Softplus": lambda m: m.SoftplusTransform(), "StickBreaking": lambda m: m.StickBreakingTransform(),
        "Compose": lambda m: m.ComposeTransform([m.AffineTransform(0.5, 2.0), m.SigmoidTransform()]),
    }[name]
    ot, tt_ = make(D), make(TD)
    x = np.array([[-1.2, 0.3, 2.0], [0.5, -0.4, 0.1]])
    yo, yt = ot(x), tt_(tt(x))
    close(yo, yt)
    close(ot.inv(yo), tt_.inv(yt))
    close(ot.log_abs_det_jacobian(x, yo), tt_.log_abs_det_jacobian(tt(x), yt))


def test_corr_cholesky_against_torch():
    v = np.array([0.3, -0.7, 1.1, 0.2, -0.4, 0.9])
    co, ct = D.CorrCholeskyTransform(), TD.CorrCholeskyTransform()
    close(co(v), ct(tt(v)))
    close(co.inv(co(v)), ct.inv(ct(tt(v))))
    close(co.log_abs_det_jacobian(v, co(v)), ct.log_abs_det_jacobian(tt(v), ct(tt(v))))
