"""Complete Probability & Distributions Library for Oxideon.

Provides probability distribution implementations supporting
variational inference, policy gradients, generative modeling, and closed-form KL divergences:
- `Distribution` (base)
- `Normal`, `Bernoulli`, `Categorical`, `Uniform`, `Exponential`
- `Laplace`, `Cauchy`, `Beta`, `Gamma`, `Dirichlet`, `Poisson`
- `Geometric`, `Gumbel`, `LogNormal`, `StudentT`
- `MultivariateNormal`, `LowRankMultivariateNormal`
- `Independent`, `TransformedDistribution` (with `log_prob` and `cdf` through the transforms)
- `Chi2`, `InverseGamma`, `Pareto`, `Kumaraswamy`, `FisherSnedecor`, `GeneralizedPareto`, `NegativeBinomial`
- `OneHotCategorical`, `OneHotCategoricalStraightThrough`, `RelaxedBernoulli`, `RelaxedOneHotCategorical`, `LogisticNormal`
- `MixtureSameFamily`, `VonMises`, `Wishart`, `LKJCholesky`, `ExponentialFamily` (base)
- transforms: `ExpTransform`, `AffineTransform`, `SigmoidTransform`, `TanhTransform`, `SoftplusTransform`, `AbsTransform`,
  `PowerTransform`, `SoftmaxTransform`, `StickBreakingTransform`, `ComposeTransform`, `IndependentTransform`,
  `ReshapeTransform`, `CatTransform`, `StackTransform`, `CumulativeDistributionTransform`, `CorrCholeskyTransform`,
  `LowerCholeskyTransform`, `PositiveDefiniteTransform`
- `mean`, `variance`, `stddev` on every distribution with a closed form
- `kl_divergence` & `@register_kl`
"""

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy


def _to_np(x: Any) -> np.ndarray:
    if isinstance(x, _Tensor):
        return x.to_numpy()
    return np.asarray(x, dtype=np.float32)


def _wrap(arr: np.ndarray) -> _Tensor:
    # np.ascontiguousarray would turn a 0-d result into shape (1,); keep 0-d scalars 0-d
    a = np.asarray(arr, dtype=np.float32)
    return _from_numpy(a if a.flags.c_contiguous else np.ascontiguousarray(a))


_KL_REGISTRY: Dict[Tuple[Type, Type], Callable[[Any, Any], _Tensor]] = {}


def register_kl(type_p: Type, type_q: Type):
    """Decorator to register a pairwise closed-form KL divergence formula."""
    def decorator(fn: Callable[[Any, Any], _Tensor]):
        _KL_REGISTRY[(type_p, type_q)] = fn
        return fn
    return decorator


def kl_divergence(p: Any, q: Any) -> _Tensor:
    """Computes the Kullback-Leibler divergence KL(p || q) between two distributions."""
    pair = (type(p), type(q))
    if pair in _KL_REGISTRY:
        return _KL_REGISTRY[pair](p, q)
    raise NotImplementedError(f"No closed-form KL divergence registered between {type(p).__name__} and {type(q).__name__}")


class Distribution:
    """Abstract base class for probability distributions."""

    has_rsample: bool = False
    has_enumerate_support: bool = False

    def __init__(self, batch_shape: Tuple[int, ...] = (), event_shape: Tuple[int, ...] = (), validate_args: Optional[bool] = None):
        self._batch_shape = tuple(batch_shape)
        self._event_shape = tuple(event_shape)
        self._validate_args = validate_args

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        return self._batch_shape

    @property
    def event_shape(self) -> Tuple[int, ...]:
        return self._event_shape

    def sample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        """Generates a sample_shape shaped sample or sample_shape shaped batch of samples."""
        raise NotImplementedError

    def rsample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        """Generates a sample_shape shaped reparameterized sample or sample_shape shaped batch of samples."""
        raise NotImplementedError

    def log_prob(self, value: Union[_Tensor, Any]) -> _Tensor:
        """Returns the log of the probability density/mass function evaluated at value."""
        raise NotImplementedError

    def cdf(self, value: Union[_Tensor, Any]) -> _Tensor:
        """Returns the cumulative density/mass function evaluated at value."""
        raise NotImplementedError

    def icdf(self, value: Union[_Tensor, Any]) -> _Tensor:
        """Returns the inverse cumulative density/mass function evaluated at value."""
        raise NotImplementedError

    def entropy(self) -> _Tensor:
        """Returns entropy of distribution, batched over batch_shape."""
        raise NotImplementedError


class Normal(Distribution):
    """Creates a normal (also called Gaussian) distribution parameterized by loc and scale."""

    has_rsample = True

    def __init__(self, loc: Union[_Tensor, float], scale: Union[_Tensor, float]):
        self.loc_np = _to_np(loc)
        self.scale_np = _to_np(scale)
        broadcast_shape = np.broadcast_shapes(self.loc_np.shape, self.scale_np.shape)
        self.loc_np = np.broadcast_to(self.loc_np, broadcast_shape).copy()
        self.scale_np = np.broadcast_to(self.scale_np, broadcast_shape).copy()

        self.loc = loc if (isinstance(loc, _Tensor) and loc.requires_grad) else _wrap(self.loc_np)
        self.scale = scale if (isinstance(scale, _Tensor) and scale.requires_grad) else _wrap(self.scale_np)
        super().__init__(batch_shape=broadcast_shape, event_shape=())

    def rsample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        shape = tuple(sample_shape) + self.batch_shape
        eps = np.random.standard_normal(shape).astype(np.float32)
        if (isinstance(self.loc, _Tensor) and self.loc.requires_grad) or (isinstance(self.scale, _Tensor) and self.scale.requires_grad):
            eps_t = _wrap(eps)
            return self.loc + eps_t * self.scale
        res = self.loc_np + eps * self.scale_np
        return _wrap(res)

    def sample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        shape = tuple(sample_shape) + self.batch_shape
        eps = np.random.standard_normal(shape).astype(np.float32)
        res = self.loc_np + eps * self.scale_np
        return _wrap(res)

    def log_prob(self, value: Union[_Tensor, Any]) -> _Tensor:
        has_grad = (
            (isinstance(value, _Tensor) and value.requires_grad)
            or (isinstance(self.loc, _Tensor) and self.loc.requires_grad)
            or (isinstance(self.scale, _Tensor) and self.scale.requires_grad)
        )
        if has_grad:
            v_t = value if isinstance(value, _Tensor) else _wrap(_to_np(value))
            var_t = self.scale * self.scale
            diff = v_t - self.loc
            return -(diff * diff) / (var_t * 2.0) - self.scale.log() - (0.5 * math.log(2.0 * math.pi))
        v = _to_np(value)
        var = self.scale_np ** 2
        log_scale = np.log(self.scale_np)
        res = -((v - self.loc_np) ** 2) / (2.0 * var) - log_scale - 0.5 * math.log(2.0 * math.pi)
        return _wrap(res)

    def cdf(self, value: Union[_Tensor, Any]) -> _Tensor:
        if (
            (isinstance(value, _Tensor) and value.requires_grad)
            or (isinstance(self.loc, _Tensor) and self.loc.requires_grad)
            or (isinstance(self.scale, _Tensor) and self.scale.requires_grad)
        ):
            v_t = value if isinstance(value, _Tensor) else _wrap(_to_np(value))
            z = (v_t - self.loc) / (self.scale * math.sqrt(2.0))
            return 0.5 * (1.0 + z.erf())
        v = _to_np(value)
        from .special import erf
        z = (v - self.loc_np) / (self.scale_np * math.sqrt(2.0))
        z_t = _wrap(z)
        erf_z = _to_np(erf(z_t))
        return _wrap(0.5 * (1.0 + erf_z))

    def icdf(self, value: Union[_Tensor, Any]) -> _Tensor:
        if (
            (isinstance(value, _Tensor) and value.requires_grad)
            or (isinstance(self.loc, _Tensor) and self.loc.requires_grad)
            or (isinstance(self.scale, _Tensor) and self.scale.requires_grad)
        ):
            v_t = value if isinstance(value, _Tensor) else _wrap(_to_np(value))
            y = 2.0 * v_t - 1.0
            return self.loc + self.scale * (math.sqrt(2.0) * y.erfinv())
        v = _to_np(value)
        from .special import erfinv
        y = 2.0 * v - 1.0
        erf_inv_val = _to_np(erfinv(_wrap(y)))
        res = self.loc_np + self.scale_np * math.sqrt(2.0) * erf_inv_val
        return _wrap(res)

    def entropy(self) -> _Tensor:
        if isinstance(self.scale, _Tensor) and self.scale.requires_grad:
            return self.scale.log() + (0.5 + 0.5 * math.log(2.0 * math.pi))
        res = 0.5 + 0.5 * math.log(2.0 * math.pi) + np.log(self.scale_np)
        return _wrap(res)


class Bernoulli(Distribution):
    """Creates a Bernoulli distribution parameterized by probs or logits."""

    def __init__(self, probs: Optional[Union[_Tensor, float]] = None, logits: Optional[Union[_Tensor, float]] = None):
        if (probs is None) == (logits is None):
            raise ValueError("Exactly one of probs or logits must be specified.")
        if probs is not None:
            self.probs_np = np.clip(_to_np(probs), 1e-12, 1.0 - 1e-12)
            self.logits_np = np.log(self.probs_np / (1.0 - self.probs_np))
        else:
            self.logits_np = _to_np(logits)
            self.probs_np = 1.0 / (1.0 + np.exp(-self.logits_np))

        self.probs = probs if (isinstance(probs, _Tensor) and probs.requires_grad) else _wrap(self.probs_np)
        self.logits = logits if (isinstance(logits, _Tensor) and logits.requires_grad) else _wrap(self.logits_np)
        super().__init__(batch_shape=self.probs_np.shape, event_shape=())

    def sample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        shape = tuple(sample_shape) + self.batch_shape
        u = np.random.uniform(0.0, 1.0, size=shape).astype(np.float32)
        res = (u < self.probs_np).astype(np.float32)
        return _wrap(res)

    def log_prob(self, value: Union[_Tensor, Any]) -> _Tensor:
        v = _to_np(value)
        res = v * np.log(self.probs_np) + (1.0 - v) * np.log(1.0 - self.probs_np)
        return _wrap(res)

    def entropy(self) -> _Tensor:
        p = self.probs_np
        res = -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)
        return _wrap(res)


class Categorical(Distribution):
    """Creates a categorical distribution parameterized by probs or logits."""

    def __init__(self, probs: Optional[Union[_Tensor, Sequence[float]]] = None, logits: Optional[Union[_Tensor, Sequence[float]]] = None):
        if (probs is None) == (logits is None):
            raise ValueError("Exactly one of probs or logits must be specified.")
        if probs is not None:
            p = _to_np(probs)
            p = p / np.sum(p, axis=-1, keepdims=True)
            self.probs_np = np.clip(p, 1e-12, 1.0)
            self.logits_np = np.log(self.probs_np)
        else:
            l = _to_np(logits)
            l_max = np.max(l, axis=-1, keepdims=True)
            exp_l = np.exp(l - l_max)
            self.probs_np = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
            self.logits_np = l - np.log(np.sum(exp_l, axis=-1, keepdims=True))

        self.probs = _wrap(self.probs_np)
        self.logits = _wrap(self.logits_np)
        super().__init__(batch_shape=self.probs_np.shape[:-1], event_shape=())

    def sample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        sample_shape = tuple(sample_shape)
        n_extra = int(np.prod(sample_shape)) if sample_shape else 1
        p_flat = self.probs_np.reshape(-1, self.probs_np.shape[-1])
        n_cats = p_flat.shape[-1]
        cols = [np.random.choice(n_cats, size=n_extra, p=row) for row in p_flat]
        res = np.stack(cols, axis=-1).reshape(sample_shape + self.batch_shape)
        return _wrap(res.astype(np.float32))

    def log_prob(self, value: Union[_Tensor, Any]) -> _Tensor:
        v = _to_np(value)
        if np.any(v != np.floor(v)):
            raise ValueError("Categorical log_prob values must be integers")
        vi = v.astype(int)
        n_cats = self.probs_np.shape[-1]
        if np.any(vi < 0) or np.any(vi >= n_cats):
            raise ValueError(f"category out of range [0, {n_cats})")
        one_hot = np.eye(n_cats)[vi]
        res = np.sum(one_hot * np.log(self.probs_np), axis=-1)
        return _wrap(res)

    def entropy(self) -> _Tensor:
        p = self.probs_np
        res = -np.sum(p * np.log(p), axis=-1)
        return _wrap(res)


class Uniform(Distribution):
    """Generates uniformly distributed random samples from the half-open interval [low, high)."""

    has_rsample = True

    def __init__(self, low: Union[_Tensor, float], high: Union[_Tensor, float]):
        self.low_np = _to_np(low)
        self.high_np = _to_np(high)
        broadcast_shape = np.broadcast_shapes(self.low_np.shape, self.high_np.shape)
        self.low_np = np.broadcast_to(self.low_np, broadcast_shape).copy()
        self.high_np = np.broadcast_to(self.high_np, broadcast_shape).copy()

        self.low = low if (isinstance(low, _Tensor) and low.requires_grad) else _wrap(self.low_np)
        self.high = high if (isinstance(high, _Tensor) and high.requires_grad) else _wrap(self.high_np)
        super().__init__(batch_shape=broadcast_shape, event_shape=())

    def rsample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        shape = tuple(sample_shape) + self.batch_shape
        u = np.random.uniform(0.0, 1.0, size=shape).astype(np.float32)
        if (isinstance(self.low, _Tensor) and self.low.requires_grad) or (isinstance(self.high, _Tensor) and self.high.requires_grad):
            return self.low + _wrap(u) * (self.high - self.low)
        res = self.low_np + u * (self.high_np - self.low_np)
        return _wrap(res)

    def sample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        return self.rsample(sample_shape)

    def log_prob(self, value: Union[_Tensor, Any]) -> _Tensor:
        v = _to_np(value)
        inside = (v >= self.low_np) & (v <= self.high_np)
        res = np.where(inside, -np.log(self.high_np - self.low_np), -np.inf)
        return _wrap(res)

    def entropy(self) -> _Tensor:
        res = np.log(self.high_np - self.low_np)
        return _wrap(res)


class Exponential(Distribution):
    """Parameterized by rate (inverse scale)."""

    has_rsample = True

    def __init__(self, rate: Union[_Tensor, float]):
        self.rate_np = _to_np(rate)
        self.rate = rate if (isinstance(rate, _Tensor) and rate.requires_grad) else _wrap(self.rate_np)
        super().__init__(batch_shape=self.rate_np.shape, event_shape=())

    def rsample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        shape = tuple(sample_shape) + self.batch_shape
        u = np.random.uniform(1e-12, 1.0, size=shape).astype(np.float32)
        if isinstance(self.rate, _Tensor) and self.rate.requires_grad:
            return _wrap(-np.log(u)) / self.rate
        res = -np.log(u) / self.rate_np
        return _wrap(res)

    def sample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        return self.rsample(sample_shape)

    def log_prob(self, value: Union[_Tensor, Any]) -> _Tensor:
        v = _to_np(value)
        res = np.where(v >= 0, np.log(self.rate_np) - self.rate_np * v, -np.inf)
        return _wrap(res)

    def entropy(self) -> _Tensor:
        res = 1.0 - np.log(self.rate_np)
        return _wrap(res)


class Laplace(Distribution):
    """Laplace distribution parameterized by loc and scale."""

    has_rsample = True

    def __init__(self, loc: Union[_Tensor, float], scale: Union[_Tensor, float]):
        self.loc_np = _to_np(loc)
        self.scale_np = _to_np(scale)
        broadcast_shape = np.broadcast_shapes(self.loc_np.shape, self.scale_np.shape)
        self.loc_np = np.broadcast_to(self.loc_np, broadcast_shape).copy()
        self.scale_np = np.broadcast_to(self.scale_np, broadcast_shape).copy()

        self.loc = loc if (isinstance(loc, _Tensor) and loc.requires_grad) else _wrap(self.loc_np)
        self.scale = scale if (isinstance(scale, _Tensor) and scale.requires_grad) else _wrap(self.scale_np)
        super().__init__(batch_shape=broadcast_shape, event_shape=())

    def rsample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        shape = tuple(sample_shape) + self.batch_shape
        u = np.random.uniform(-0.5, 0.5, size=shape).astype(np.float32)
        noise = -np.sign(u) * np.log(1.0 - 2.0 * np.abs(u))
        if (isinstance(self.loc, _Tensor) and self.loc.requires_grad) or (isinstance(self.scale, _Tensor) and self.scale.requires_grad):
            return self.loc + self.scale * _wrap(noise)
        res = self.loc_np + self.scale_np * noise
        return _wrap(res)

    def sample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        return self.rsample(sample_shape)

    def log_prob(self, value: Union[_Tensor, Any]) -> _Tensor:
        v = _to_np(value)
        res = -np.log(2.0 * self.scale_np) - np.abs(v - self.loc_np) / self.scale_np
        return _wrap(res)

    def entropy(self) -> _Tensor:
        res = 1.0 + np.log(2.0 * self.scale_np)
        return _wrap(res)


class Cauchy(Distribution):
    """Cauchy distribution parameterized by loc and scale."""

    has_rsample = True

    def __init__(self, loc: Union[_Tensor, float], scale: Union[_Tensor, float]):
        self.loc_np = _to_np(loc)
        self.scale_np = _to_np(scale)
        broadcast_shape = np.broadcast_shapes(self.loc_np.shape, self.scale_np.shape)
        self.loc = loc if (isinstance(loc, _Tensor) and loc.requires_grad) else _wrap(self.loc_np)
        self.scale = scale if (isinstance(scale, _Tensor) and scale.requires_grad) else _wrap(self.scale_np)
        super().__init__(batch_shape=broadcast_shape, event_shape=())

    def _has_grad(self, *extras: Any) -> bool:
        return any(isinstance(x, _Tensor) and x.requires_grad for x in (self.loc, self.scale, *extras))

    def rsample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        shape = tuple(sample_shape) + self.batch_shape
        u = np.random.uniform(1e-7, 1.0 - 1e-7, size=shape).astype(np.float32)
        if self._has_grad():
            u_t = _wrap(u)
            return self.loc + self.scale * ((u_t - 0.5) * math.pi).tan()
        res = self.loc_np + self.scale_np * np.tan(np.pi * (u - 0.5))
        return _wrap(res)

    def sample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        return self.rsample(sample_shape)

    def log_prob(self, value: Union[_Tensor, Any]) -> _Tensor:
        if self._has_grad(value):
            v_t = value if isinstance(value, _Tensor) else _wrap(_to_np(value))
            return -math.log(math.pi) - self.scale.log() - (1.0 + ((v_t - self.loc) / self.scale) ** 2).log()
        v = _to_np(value)
        res = -math.log(math.pi) - np.log(self.scale_np) - np.log(1.0 + ((v - self.loc_np) / self.scale_np) ** 2)
        return _wrap(res)

    def cdf(self, value: Union[_Tensor, Any]) -> _Tensor:
        if self._has_grad(value):
            v_t = value if isinstance(value, _Tensor) else _wrap(_to_np(value))
            return 0.5 + ((v_t - self.loc) / self.scale).atan() / math.pi
        v = _to_np(value)
        return _wrap(0.5 + np.arctan((v - self.loc_np) / self.scale_np) / math.pi)

    def icdf(self, value: Union[_Tensor, Any]) -> _Tensor:
        if self._has_grad(value):
            v_t = value if isinstance(value, _Tensor) else _wrap(_to_np(value))
            return self.loc + self.scale * ((v_t - 0.5) * math.pi).tan()
        v = _to_np(value)
        return _wrap(self.loc_np + self.scale_np * np.tan(math.pi * (v - 0.5)))

    def entropy(self) -> _Tensor:
        if self._has_grad():
            return self.scale.log() + math.log(4.0 * math.pi)
        return _wrap(np.log(4.0 * math.pi * self.scale_np))


class Beta(Distribution):
    """Beta distribution parameterized by concentration1 (alpha) and concentration0 (beta)."""

    has_rsample = True

    def __init__(self, concentration1: Union[_Tensor, float], concentration0: Union[_Tensor, float]):
        self.alpha_np = _to_np(concentration1)
        self.beta_np = _to_np(concentration0)
        shape = np.broadcast_shapes(self.alpha_np.shape, self.beta_np.shape)
        self.alpha_np = np.broadcast_to(self.alpha_np, shape).copy()
        self.beta_np = np.broadcast_to(self.beta_np, shape).copy()

        self.concentration1 = concentration1 if (isinstance(concentration1, _Tensor) and concentration1.requires_grad) else _wrap(self.alpha_np)
        self.concentration0 = concentration0 if (isinstance(concentration0, _Tensor) and concentration0.requires_grad) else _wrap(self.beta_np)
        super().__init__(batch_shape=shape, event_shape=())

    def rsample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        shape = tuple(sample_shape) + self.batch_shape
        g1 = np.random.gamma(self.alpha_np, 1.0, size=shape).astype(np.float32)
        g0 = np.random.gamma(self.beta_np, 1.0, size=shape).astype(np.float32)
        res = g1 / np.maximum(g1 + g0, 1e-12)
        return _wrap(res)

    def sample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        return self.rsample(sample_shape)

    def log_prob(self, value: Union[_Tensor, Any]) -> _Tensor:
        a, b = self.concentration1, self.concentration0
        if any(isinstance(x, _Tensor) and x.requires_grad for x in (a, b, value)):
            v_t = value if isinstance(value, _Tensor) else _wrap(_to_np(value))
            log_beta = a.lgamma() + b.lgamma() - (a + b).lgamma()
            return (a - 1.0) * v_t.log() + (b - 1.0) * (1.0 - v_t).log() - log_beta
        v_raw = _to_np(value).astype(np.float64)
        bad = (v_raw < 0.0) | (v_raw > 1.0)
        v = np.clip(np.where(bad, 0.5, v_raw), 1e-12, 1.0 - 1e-12)
        from .special import gammaln
        lgamma_a = _to_np(gammaln(self.concentration1))
        lgamma_b = _to_np(gammaln(self.concentration0))
        lgamma_ab = _to_np(gammaln(_wrap(self.alpha_np + self.beta_np)))
        log_beta = lgamma_a + lgamma_b - lgamma_ab
        res = (self.alpha_np - 1.0) * np.log(v) + (self.beta_np - 1.0) * np.log(1.0 - v) - log_beta
        return _wrap(np.where(bad, -np.inf, res))

    def entropy(self) -> _Tensor:
        a, b = self.concentration1, self.concentration0
        ab = a + b
        log_beta = a.lgamma() + b.lgamma() - ab.lgamma()
        return log_beta - (a - 1.0) * a.digamma() - (b - 1.0) * b.digamma() + (ab - 2.0) * ab.digamma()


class Gamma(Distribution):
    """Gamma distribution parameterized by concentration (shape) and rate (inverse scale)."""

    has_rsample = True

    def __init__(self, concentration: Union[_Tensor, float], rate: Union[_Tensor, float]):
        self.c_np = _to_np(concentration)
        self.r_np = _to_np(rate)
        shape = np.broadcast_shapes(self.c_np.shape, self.r_np.shape)
        self.c_np = np.broadcast_to(self.c_np, shape).copy()
        self.r_np = np.broadcast_to(self.r_np, shape).copy()

        self.concentration = concentration if (isinstance(concentration, _Tensor) and concentration.requires_grad) else _wrap(self.c_np)
        self.rate = rate if (isinstance(rate, _Tensor) and rate.requires_grad) else _wrap(self.r_np)
        super().__init__(batch_shape=shape, event_shape=())

    def rsample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        shape = tuple(sample_shape) + self.batch_shape
        res = np.random.gamma(self.c_np, 1.0 / self.r_np, size=shape).astype(np.float32)
        return _wrap(res)

    def sample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        return self.rsample(sample_shape)

    def log_prob(self, value: Union[_Tensor, Any]) -> _Tensor:
        c, r = self.concentration, self.rate
        if any(isinstance(x, _Tensor) and x.requires_grad for x in (c, r, value)):
            v_t = value if isinstance(value, _Tensor) else _wrap(_to_np(value))
            return c * r.log() + (c - 1.0) * v_t.log() - r * v_t - c.lgamma()
        v_raw = _to_np(value)
        bad = v_raw < 0
        v = np.where(bad, 1.0, v_raw)
        from .special import gammaln
        lg_c = _to_np(gammaln(self.concentration))
        res = self.c_np * np.log(self.r_np) + (self.c_np - 1.0) * np.log(np.maximum(v, 1e-12)) - self.r_np * v - lg_c
        return _wrap(np.where(bad, -np.inf, res))

    def entropy(self) -> _Tensor:
        c, r = self.concentration, self.rate
        return c - r.log() + c.lgamma() + (1.0 - c) * c.digamma()


class Dirichlet(Distribution):
    """Dirichlet distribution parameterized by concentration vector."""

    has_rsample = True

    def __init__(self, concentration: Union[_Tensor, Sequence[float]]):
        self.conc_np = _to_np(concentration)
        self.concentration = concentration if (isinstance(concentration, _Tensor) and concentration.requires_grad) else _wrap(self.conc_np)
        super().__init__(batch_shape=self.conc_np.shape[:-1], event_shape=(self.conc_np.shape[-1],))

    def rsample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        shape = tuple(sample_shape) + self.conc_np.shape
        gamma_samples = np.random.gamma(self.conc_np, 1.0, size=shape).astype(np.float32)
        res = gamma_samples / np.sum(gamma_samples, axis=-1, keepdims=True)
        return _wrap(res)

    def sample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        return self.rsample(sample_shape)

    def log_prob(self, value: Union[_Tensor, Any]) -> _Tensor:
        c = self.concentration
        if (isinstance(c, _Tensor) and c.requires_grad) or (isinstance(value, _Tensor) and value.requires_grad):
            v_t = value if isinstance(value, _Tensor) else _wrap(_to_np(value))
            c_sum = c.sum(dim=-1)
            log_b = c.lgamma().sum(dim=-1) - c_sum.lgamma()
            return ((c - 1.0) * v_t.log()).sum(dim=-1) - log_b
        v_raw = _to_np(value).astype(np.float64)
        bad = (
            np.any(v_raw < 0.0, axis=-1)
            | np.any(v_raw > 1.0, axis=-1)
            | ~np.isclose(np.sum(v_raw, axis=-1), 1.0)
        )
        v = np.clip(v_raw, 1e-12, 1.0)
        v = v / np.sum(v, axis=-1, keepdims=True)
        from .special import gammaln
        lg_alpha = _to_np(gammaln(self.concentration))
        lg_sum = _to_np(gammaln(_wrap(np.sum(self.conc_np, axis=-1))))
        log_b = np.sum(lg_alpha, axis=-1) - lg_sum
        res = np.sum((self.conc_np - 1.0) * np.log(v), axis=-1) - log_b
        return _wrap(np.where(bad, -np.inf, res))

    def entropy(self) -> _Tensor:
        c = self.concentration
        c_sum = c.sum(dim=-1)
        log_b = c.lgamma().sum(dim=-1) - c_sum.lgamma()
        k = self.event_shape[0]
        return log_b + (c_sum - float(k)) * c_sum.digamma() - ((c - 1.0) * c.digamma()).sum(dim=-1)


class Poisson(Distribution):
    """Poisson distribution parameterized by rate."""

    def __init__(self, rate: Union[_Tensor, float]):
        self.rate_np = _to_np(rate)
        self.rate = _wrap(self.rate_np)
        super().__init__(batch_shape=self.rate_np.shape, event_shape=())

    def sample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        shape = tuple(sample_shape) + self.batch_shape
        res = np.random.poisson(self.rate_np, size=shape).astype(np.float32)
        return _wrap(res)

    def log_prob(self, value: Union[_Tensor, Any]) -> _Tensor:
        k_raw = _to_np(value)
        bad = (k_raw < 0) | (k_raw != np.floor(k_raw))
        k = np.where(bad, 0.0, k_raw)
        from .special import gammaln
        lg_k1 = _to_np(gammaln(_wrap(k + 1.0)))
        res = k * np.log(self.rate_np) - self.rate_np - lg_k1
        return _wrap(np.where(bad, -np.inf, res))


class Geometric(Distribution):
    """Geometric distribution parameterized by probs or logits."""

    def __init__(self, probs: Optional[Union[_Tensor, float]] = None, logits: Optional[Union[_Tensor, float]] = None):
        if (probs is None) == (logits is None):
            raise ValueError("Exactly one of probs or logits must be specified.")
        if probs is not None:
            self.probs_np = np.clip(_to_np(probs), 1e-12, 1.0 - 1e-12)
        else:
            l = _to_np(logits)
            self.probs_np = 1.0 / (1.0 + np.exp(-l))
        self.probs = _wrap(self.probs_np)
        super().__init__(batch_shape=self.probs_np.shape, event_shape=())

    def sample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        shape = tuple(sample_shape) + self.batch_shape
        u = np.random.uniform(0.0, 1.0, size=shape).astype(np.float32)
        res = np.floor(np.log(u) / np.log(1.0 - self.probs_np)).astype(np.float32)
        return _wrap(res)

    def log_prob(self, value: Union[_Tensor, Any]) -> _Tensor:
        v_raw = _to_np(value)
        bad = (v_raw < 0) | (v_raw != np.floor(v_raw))
        v = np.where(bad, 0.0, v_raw)
        res = v * np.log(1.0 - self.probs_np) + np.log(self.probs_np)
        return _wrap(np.where(bad, -np.inf, res))


class Gumbel(Distribution):
    """Gumbel distribution parameterized by loc and scale."""

    has_rsample = True

    def __init__(self, loc: Union[_Tensor, float], scale: Union[_Tensor, float]):
        self.loc_np = _to_np(loc)
        self.scale_np = _to_np(scale)
        shape = np.broadcast_shapes(self.loc_np.shape, self.scale_np.shape)
        self.loc = loc if (isinstance(loc, _Tensor) and loc.requires_grad) else _wrap(self.loc_np)
        self.scale = scale if (isinstance(scale, _Tensor) and scale.requires_grad) else _wrap(self.scale_np)
        super().__init__(batch_shape=shape, event_shape=())

    def _has_grad(self, *extras: Any) -> bool:
        return any(isinstance(x, _Tensor) and x.requires_grad for x in (self.loc, self.scale, *extras))

    def rsample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        shape = tuple(sample_shape) + self.batch_shape
        u = np.random.uniform(1e-7, 1.0 - 1e-7, size=shape).astype(np.float32)
        if self._has_grad():
            u_t = _wrap(u)
            return self.loc - self.scale * (-u_t.log()).log()
        g = -np.log(-np.log(u))
        res = self.loc_np + self.scale_np * g
        return _wrap(res)

    def sample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        return self.rsample(sample_shape)

    def log_prob(self, value: Union[_Tensor, Any]) -> _Tensor:
        if self._has_grad(value):
            v_t = value if isinstance(value, _Tensor) else _wrap(_to_np(value))
            z = (v_t - self.loc) / self.scale
            return -z - (-z).exp() - self.scale.log()
        v = _to_np(value)
        z = (v - self.loc_np) / self.scale_np
        res = -z - np.exp(-z) - np.log(self.scale_np)
        return _wrap(res)

    def cdf(self, value: Union[_Tensor, Any]) -> _Tensor:
        if self._has_grad(value):
            v_t = value if isinstance(value, _Tensor) else _wrap(_to_np(value))
            z = (v_t - self.loc) / self.scale
            return (-(-z).exp()).exp()
        v = _to_np(value)
        z = (v - self.loc_np) / self.scale_np
        return _wrap(np.exp(-np.exp(-z)))

    def icdf(self, value: Union[_Tensor, Any]) -> _Tensor:
        if self._has_grad(value):
            v_t = value if isinstance(value, _Tensor) else _wrap(_to_np(value))
            return self.loc - self.scale * (-v_t.log()).log()
        v = _to_np(value)
        return _wrap(self.loc_np - self.scale_np * np.log(-np.log(v)))

    def entropy(self) -> _Tensor:
        if self._has_grad():
            return self.scale.log() + (1.0 + 0.5772156649015329)
        return _wrap(np.log(self.scale_np) + 1.0 + 0.5772156649015329)


class LogNormal(Distribution):
    """Log-normal distribution parameterized by loc and scale."""

    has_rsample = True

    def __init__(self, loc: Union[_Tensor, float], scale: Union[_Tensor, float]):
        self.base = Normal(loc, scale)
        super().__init__(batch_shape=self.base.batch_shape, event_shape=())

    def rsample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        base_s = self.base.rsample(sample_shape)
        if isinstance(base_s, _Tensor) and base_s.requires_grad:
            return base_s.exp()
        return _wrap(np.exp(_to_np(base_s)))

    def sample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        return self.rsample(sample_shape)

    def log_prob(self, value: Union[_Tensor, Any]) -> _Tensor:
        v_raw = _to_np(value)
        bad = v_raw <= 0.0
        v = np.where(bad, 1.0, v_raw)
        log_v = np.log(np.maximum(v, 1e-12))
        base_log_prob = _to_np(self.base.log_prob(_wrap(log_v)))
        res = base_log_prob - log_v
        return _wrap(np.where(bad, -np.inf, res))


class MultivariateNormal(Distribution):
    """Multivariate normal distribution parameterized by loc and covariance_matrix."""

    has_rsample = True

    def __init__(
        self,
        loc: Union[_Tensor, Sequence[float]],
        covariance_matrix: Optional[Union[_Tensor, Sequence[Sequence[float]]]] = None,
        scale_tril: Optional[Union[_Tensor, Sequence[Sequence[float]]]] = None,
    ):
        self.loc_np = _to_np(loc)
        dim = self.loc_np.shape[-1]

        if scale_tril is not None:
            self.tril_np = _to_np(scale_tril)
            self.cov_np = self.tril_np @ np.swapaxes(self.tril_np, -1, -2)
            self.scale_tril = scale_tril if (isinstance(scale_tril, _Tensor) and scale_tril.requires_grad) else _wrap(self.tril_np)
            self.covariance_matrix = self.scale_tril.matmul(self.scale_tril.transpose(-2, -1)) if self.scale_tril.requires_grad else _wrap(self.cov_np)
        elif covariance_matrix is not None:
            self.cov_np = _to_np(covariance_matrix)
            self.tril_np = np.linalg.cholesky(self.cov_np)
            self.covariance_matrix = covariance_matrix if (isinstance(covariance_matrix, _Tensor) and covariance_matrix.requires_grad) else _wrap(self.cov_np)
            self.scale_tril = self.covariance_matrix.cholesky() if self.covariance_matrix.requires_grad else _wrap(self.tril_np)
        else:
            self.cov_np = np.eye(dim, dtype=np.float32)
            self.tril_np = np.eye(dim, dtype=np.float32)
            self.covariance_matrix = _wrap(self.cov_np)
            self.scale_tril = _wrap(self.tril_np)

        self.loc = loc if (isinstance(loc, _Tensor) and loc.requires_grad) else _wrap(self.loc_np)
        super().__init__(batch_shape=self.loc_np.shape[:-1], event_shape=(dim,))

    def _has_grad(self, *extras: Any) -> bool:
        return any(
            isinstance(x, _Tensor) and x.requires_grad
            for x in (self.loc, self.covariance_matrix, self.scale_tril, *extras)
        )

    def rsample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        shape = tuple(sample_shape) + self.batch_shape + self.event_shape
        eps = np.random.standard_normal(shape).astype(np.float32)
        if self._has_grad():
            eps_t = _wrap(eps)
            return eps_t.unsqueeze(-2).matmul(self.scale_tril.transpose(-2, -1)).squeeze(-2) + self.loc
        res = (eps[..., None, :] @ np.swapaxes(self.tril_np, -1, -2)).squeeze(-2) + self.loc_np
        return _wrap(res)

    def sample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        return self.rsample(sample_shape)

    def log_prob(self, value: Union[_Tensor, Any]) -> _Tensor:
        dim = self.event_shape[0]
        if self._has_grad(value):
            v_t = value if isinstance(value, _Tensor) else _wrap(_to_np(value))
            diff = v_t - self.loc
            solve_val = self.covariance_matrix.solve(diff.unsqueeze(-1)).squeeze(-1)
            mahalanobis = (diff * solve_val).sum(dim=-1)
            ld = self.covariance_matrix.logdet()
            return -0.5 * (dim * math.log(2.0 * math.pi) + ld + mahalanobis)
        v = _to_np(value)
        diff = v - self.loc_np
        solve_val = np.linalg.solve(self.cov_np, diff[..., :, None]).squeeze(-1)
        mahalanobis = np.sum(diff * solve_val, axis=-1)
        sign, logdet = np.linalg.slogdet(self.cov_np)
        res = -0.5 * (dim * math.log(2.0 * math.pi) + logdet + mahalanobis)
        return _wrap(res)

    def entropy(self) -> _Tensor:
        dim = self.event_shape[0]
        if self._has_grad():
            return 0.5 * (dim * (1.0 + math.log(2.0 * math.pi)) + self.covariance_matrix.logdet())
        sign, logdet = np.linalg.slogdet(self.cov_np)
        res = 0.5 * (dim * (1.0 + math.log(2.0 * math.pi)) + logdet)
        return _wrap(res)


class LowRankMultivariateNormal(Distribution):
    """Multivariate normal distribution with low-rank covariance structure: cov = W W^T + D."""

    has_rsample = True

    def __init__(self, loc: _Tensor, cov_factor: _Tensor, cov_diag: _Tensor):
        self.loc_np = _to_np(loc)
        self.w_np = _to_np(cov_factor)
        self.d_np = _to_np(cov_diag)
        dim = self.loc_np.shape[-1]
        wwt = self.w_np @ np.swapaxes(self.w_np, -1, -2)
        diag = np.zeros_like(wwt)
        idx = np.arange(wwt.shape[-1])
        diag[..., idx, idx] = self.d_np
        cov = wwt + diag
        self.mvn = MultivariateNormal(loc, covariance_matrix=_wrap(cov))
        super().__init__(batch_shape=self.loc_np.shape[:-1], event_shape=(dim,))

    def rsample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        return self.mvn.rsample(sample_shape)

    def sample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        return self.mvn.sample(sample_shape)

    def log_prob(self, value: Union[_Tensor, Any]) -> _Tensor:
        return self.mvn.log_prob(value)


class StudentT(Distribution):
    """Student's t-distribution parameterized by degrees of freedom df, loc, and scale."""

    has_rsample = True

    def __init__(self, df: Union[_Tensor, float], loc: Union[_Tensor, float] = 0.0, scale: Union[_Tensor, float] = 1.0):
        self.df_np = _to_np(df)
        self.loc_np = _to_np(loc)
        self.scale_np = _to_np(scale)
        shape = np.broadcast_shapes(self.df_np.shape, self.loc_np.shape, self.scale_np.shape)
        self.df = df if (isinstance(df, _Tensor) and df.requires_grad) else _wrap(self.df_np)
        self.loc = loc if (isinstance(loc, _Tensor) and loc.requires_grad) else _wrap(self.loc_np)
        self.scale = scale if (isinstance(scale, _Tensor) and scale.requires_grad) else _wrap(self.scale_np)
        super().__init__(batch_shape=shape, event_shape=())

    def rsample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        shape = tuple(sample_shape) + self.batch_shape
        z = np.random.standard_normal(shape).astype(np.float32)
        v = np.random.gamma(self.df_np / 2.0, 2.0 / self.df_np, size=shape).astype(np.float32)
        res = self.loc_np + self.scale_np * (z / np.sqrt(np.maximum(v, 1e-12)))
        return _wrap(res)

    def sample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        return self.rsample(sample_shape)

    def log_prob(self, value: Union[_Tensor, Any]) -> _Tensor:
        if any(isinstance(x, _Tensor) and x.requires_grad for x in (self.df, self.loc, self.scale, value)):
            v_t = value if isinstance(value, _Tensor) else _wrap(_to_np(value))
            y = (v_t - self.loc) / self.scale
            half_df = self.df * 0.5
            half_df1 = (self.df + 1.0) * 0.5
            return (
                half_df1.lgamma()
                - half_df.lgamma()
                - 0.5 * (self.df.log() + math.log(math.pi))
                - self.scale.log()
                - half_df1 * (1.0 + (y * y) / self.df).log()
            )
        x = _to_np(value)
        y = (x - self.loc_np) / self.scale_np
        from .special import gammaln
        lg_half_df1 = _to_np(gammaln(_wrap((self.df_np + 1.0) / 2.0)))
        lg_half_df = _to_np(gammaln(_wrap(self.df_np / 2.0)))
        res = (
            lg_half_df1
            - lg_half_df
            - 0.5 * (np.log(math.pi) + np.log(self.df_np))
            - np.log(self.scale_np)
            - 0.5 * (self.df_np + 1.0) * np.log(1.0 + (y ** 2) / self.df_np)
        )
        return _wrap(res)

    def entropy(self) -> _Tensor:
        half_df = self.df * 0.5
        half_df1 = (self.df + 1.0) * 0.5
        log_b = half_df.lgamma() + 0.5 * math.log(math.pi) - half_df1.lgamma()
        return (
            half_df1 * (half_df1.digamma() - half_df.digamma())
            + 0.5 * self.df.log()
            + log_b
            + self.scale.log()
        )


class Independent(Distribution):
    """Reinterprets some of the batch dims of a base distribution as event dims."""

    def __init__(self, base_distribution: Distribution, reinterpreted_batch_ndims: int):
        self.base_dist = base_distribution
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        batch_shape = base_distribution.batch_shape[:-reinterpreted_batch_ndims] if reinterpreted_batch_ndims > 0 else base_distribution.batch_shape
        event_shape = base_distribution.batch_shape[-reinterpreted_batch_ndims:] + base_distribution.event_shape if reinterpreted_batch_ndims > 0 else base_distribution.event_shape
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def sample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        return self.base_dist.sample(sample_shape)

    def rsample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        return self.base_dist.rsample(sample_shape)

    def log_prob(self, value: Union[_Tensor, Any]) -> _Tensor:
        lp = self.base_dist.log_prob(value)
        if isinstance(lp, _Tensor) and lp.requires_grad:
            out = lp
            for _ in range(self.reinterpreted_batch_ndims):
                out = out.sum(dim=-1)
            return out
        lp_np = _to_np(lp)
        if self.reinterpreted_batch_ndims > 0:
            axes = tuple(range(-self.reinterpreted_batch_ndims, 0))
            lp_np = np.sum(lp_np, axis=axes)
        return _wrap(lp_np)


class TransformedDistribution(Distribution):
    """Extension of the Distribution class, which applies a sequence of Transforms to a base distribution."""

    def __init__(self, base_distribution: Distribution, transforms: Sequence[Any]):
        self.base_dist = base_distribution
        self.transforms = list(transforms)
        super().__init__(batch_shape=base_distribution.batch_shape, event_shape=base_distribution.event_shape)

    def sample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        x = self.base_dist.sample(sample_shape)
        for t in self.transforms:
            x = t(x)
        return x

    def rsample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        x = self.base_dist.rsample(sample_shape)
        for t in self.transforms:
            x = t(x)
        return x


# ---------------------------------------------------------------------------
# Closed-Form KL Divergences
# ---------------------------------------------------------------------------

@register_kl(Normal, Normal)
def _kl_normal_normal(p: Normal, q: Normal) -> _Tensor:
    has_grad = (
        (isinstance(p.loc, _Tensor) and p.loc.requires_grad)
        or (isinstance(p.scale, _Tensor) and p.scale.requires_grad)
        or (isinstance(q.loc, _Tensor) and q.loc.requires_grad)
        or (isinstance(q.scale, _Tensor) and q.scale.requires_grad)
    )
    if has_grad:
        var_p = p.scale * p.scale
        var_q = q.scale * q.scale
        diff = p.loc - q.loc
        return (q.scale / p.scale).log() + (var_p + diff * diff) / (var_q * 2.0) - 0.5
    var_p = p.scale_np ** 2
    var_q = q.scale_np ** 2
    kl = np.log(q.scale_np / p.scale_np) + (var_p + (p.loc_np - q.loc_np) ** 2) / (2.0 * var_q) - 0.5
    return _wrap(kl)


@register_kl(Bernoulli, Bernoulli)
def _kl_bernoulli_bernoulli(p: Bernoulli, q: Bernoulli) -> _Tensor:
    p_prob = p.probs_np
    q_prob = q.probs_np
    kl = p_prob * np.log(p_prob / q_prob) + (1.0 - p_prob) * np.log((1.0 - p_prob) / (1.0 - q_prob))
    return _wrap(kl)


@register_kl(Categorical, Categorical)
def _kl_categorical_categorical(p: Categorical, q: Categorical) -> _Tensor:
    p_prob = p.probs_np
    q_prob = q.probs_np
    kl = np.sum(p_prob * (np.log(p_prob) - np.log(q_prob)), axis=-1)
    return _wrap(kl)


@register_kl(Uniform, Uniform)
def _kl_uniform_uniform(p: Uniform, q: Uniform) -> _Tensor:
    kl = np.log((q.high_np - q.low_np) / (p.high_np - p.low_np))
    invalid = (p.low_np < q.low_np) | (p.high_np > q.high_np)
    kl = np.where(invalid, np.inf, kl)
    return _wrap(kl)


@register_kl(Laplace, Laplace)
def _kl_laplace_laplace(p: Laplace, q: Laplace) -> _Tensor:
    diff = np.abs(p.loc_np - q.loc_np)
    ratio = p.scale_np / q.scale_np
    kl = -np.log(ratio) + diff / q.scale_np + ratio * np.exp(-diff / p.scale_np) - 1.0
    return _wrap(kl)


@register_kl(Exponential, Exponential)
def _kl_exponential_exponential(p: Exponential, q: Exponential) -> _Tensor:
    rate_p = p.rate_np
    rate_q = q.rate_np
    kl = np.log(rate_p / rate_q) + rate_q / rate_p - 1.0
    return _wrap(kl)


class Binomial(Distribution):
    """Creates a Binomial distribution parameterized by `total_count` and `probs` or `logits`."""
    arg_constraints = {"total_count": "integer", "probs": "unit_interval"}

    def __init__(
        self,
        total_count: Union[int, _Tensor] = 1,
        probs: Optional[Union[float, _Tensor]] = None,
        logits: Optional[Union[float, _Tensor]] = None,
        validate_args: Optional[bool] = None,
    ):
        if probs is None and logits is None:
            raise ValueError("One of `probs` or `logits` must be specified.")
        self.count_np = _to_np(total_count).astype(np.int64)
        if probs is not None:
            self.probs_np = _to_np(probs).astype(np.float64)
            self.logits_np = np.log(np.maximum(1e-15, self.probs_np) / np.maximum(1e-15, 1.0 - self.probs_np))
        else:
            self.logits_np = _to_np(logits).astype(np.float64)
            self.probs_np = 1.0 / (1.0 + np.exp(-self.logits_np))
        batch_shape = np.broadcast_shapes(self.count_np.shape, self.probs_np.shape)
        super().__init__(batch_shape=batch_shape, event_shape=(), validate_args=validate_args)

    def sample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        shape = tuple(sample_shape) + self._batch_shape
        s = np.random.binomial(self.count_np, self.probs_np, size=shape)
        return _wrap(s)

    def log_prob(self, value: Union[float, _Tensor]) -> _Tensor:
        k_raw = _to_np(value).astype(np.float64)
        n = self.count_np
        p = self.probs_np
        bad = (k_raw < 0) | (k_raw > n) | (k_raw != np.floor(k_raw))
        k = np.clip(np.where(bad, 0.0, k_raw), 0.0, n)
        from math import lgamma
        _lgamma_v = np.vectorize(lgamma)
        log_comb = _lgamma_v(n + 1) - _lgamma_v(k + 1) - _lgamma_v(n - k + 1)
        lp = log_comb + k * np.log(np.maximum(1e-15, p)) + (n - k) * np.log(np.maximum(1e-15, 1.0 - p))
        return _wrap(np.where(bad, -np.inf, lp))


class Multinomial(Distribution):
    """Creates a Multinomial distribution parameterized by `total_count` and `probs` or `logits`."""

    def __init__(
        self,
        total_count: int = 1,
        probs: Optional[_Tensor] = None,
        logits: Optional[_Tensor] = None,
        validate_args: Optional[bool] = None,
    ):
        if probs is None and logits is None:
            raise ValueError("One of `probs` or `logits` must be specified.")
        self.total_count = int(total_count)
        if probs is not None:
            self.probs_np = _to_np(probs).astype(np.float64)
            self.probs_np = self.probs_np / np.sum(self.probs_np, axis=-1, keepdims=True)
        else:
            l = _to_np(logits).astype(np.float64)
            e = np.exp(l - np.max(l, axis=-1, keepdims=True))
            self.probs_np = e / np.sum(e, axis=-1, keepdims=True)
        super().__init__(batch_shape=self.probs_np.shape[:-1], event_shape=(self.probs_np.shape[-1],), validate_args=validate_args)

    def sample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        shape = tuple(sample_shape) + self._batch_shape
        res = np.zeros(shape + self._event_shape, dtype=np.float32)
        flat_p = self.probs_np.reshape(-1, self._event_shape[0])
        flat_res = res.reshape(-1, self._event_shape[0])
        for i in range(flat_res.shape[0]):
            p = flat_p[i % flat_p.shape[0]]
            flat_res[i] = np.random.multinomial(self.total_count, p)
        return _wrap(res)

    def log_prob(self, value: _Tensor) -> _Tensor:
        x_raw = _to_np(value).astype(np.float64)
        bad = (
            np.any(x_raw < 0, axis=-1)
            | np.any(x_raw != np.floor(x_raw), axis=-1)
            | ~np.isclose(np.sum(x_raw, axis=-1), self.total_count)
        )
        x = np.where(x_raw < 0, 0.0, x_raw)
        from math import lgamma
        _lgamma_v = np.vectorize(lgamma)
        log_n_fact = lgamma(self.total_count + 1)
        log_x_fact = np.sum(_lgamma_v(x + 1), axis=-1)
        log_p = np.sum(x * np.log(np.maximum(1e-15, self.probs_np)), axis=-1)
        return _wrap(np.where(bad, -np.inf, log_n_fact - log_x_fact + log_p))


class HalfNormal(Distribution):
    """Half-Normal distribution parameterized by `scale`."""

    has_rsample = True

    def __init__(self, scale: Union[float, _Tensor], validate_args: Optional[bool] = None):
        self.scale_np = _to_np(scale).astype(np.float64)
        self.scale = scale if (isinstance(scale, _Tensor) and scale.requires_grad) else _wrap(self.scale_np)
        super().__init__(batch_shape=self.scale_np.shape, event_shape=(), validate_args=validate_args)

    def rsample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        shape = tuple(sample_shape) + self._batch_shape
        eps = np.abs(np.random.standard_normal(shape)).astype(np.float32)
        if isinstance(self.scale, _Tensor) and self.scale.requires_grad:
            return _wrap(eps) * self.scale
        return _wrap(eps * self.scale_np)

    def sample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        return self.rsample(sample_shape)

    def log_prob(self, value: Union[float, _Tensor]) -> _Tensor:
        if (isinstance(self.scale, _Tensor) and self.scale.requires_grad) or (isinstance(value, _Tensor) and value.requires_grad):
            v_t = value if isinstance(value, _Tensor) else _wrap(_to_np(value))
            return 0.5 * math.log(2.0 / math.pi) - self.scale.log() - (v_t * v_t) / (2.0 * self.scale * self.scale)
        v = _to_np(value).astype(np.float64)
        lp = 0.5 * np.log(2.0 / np.pi) - np.log(self.scale_np) - (v * v) / (2.0 * self.scale_np * self.scale_np)
        lp = np.where(v < 0.0, -np.inf, lp)
        return _wrap(lp)


class HalfCauchy(Distribution):
    """Half-Cauchy distribution parameterized by `scale`."""

    def __init__(self, scale: Union[float, _Tensor], validate_args: Optional[bool] = None):
        self.scale_np = _to_np(scale).astype(np.float64)
        super().__init__(batch_shape=self.scale_np.shape, event_shape=(), validate_args=validate_args)

    def sample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        shape = tuple(sample_shape) + self._batch_shape
        s = np.abs(self.scale_np * np.tan(np.pi * (np.random.uniform(0.0, 1.0, size=shape) - 0.5)))
        return _wrap(s)

    def log_prob(self, value: Union[float, _Tensor]) -> _Tensor:
        v = _to_np(value).astype(np.float64)
        lp = np.log(2.0 / np.pi) - np.log(self.scale_np) - np.log(1.0 + (v / self.scale_np) ** 2)
        lp = np.where(v < 0.0, -np.inf, lp)
        return _wrap(lp)


class Weibull(Distribution):
    """Weibull distribution parameterized by `scale` and `concentration`."""

    has_rsample = True

    def __init__(self, scale: Union[float, _Tensor], concentration: Union[float, _Tensor], validate_args: Optional[bool] = None):
        self.scale_np = _to_np(scale).astype(np.float64)
        self.conc_np = _to_np(concentration).astype(np.float64)
        self.scale = scale if (isinstance(scale, _Tensor) and scale.requires_grad) else _wrap(self.scale_np)
        self.concentration = concentration if (isinstance(concentration, _Tensor) and concentration.requires_grad) else _wrap(self.conc_np)
        super().__init__(batch_shape=np.broadcast_shapes(self.scale_np.shape, self.conc_np.shape), event_shape=(), validate_args=validate_args)

    def rsample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        shape = tuple(sample_shape) + self._batch_shape
        u = np.random.uniform(1e-7, 1.0 - 1e-7, size=shape).astype(np.float32)
        if any(isinstance(x, _Tensor) and x.requires_grad for x in (self.scale, self.concentration)):
            u_t = _wrap(u)
            return self.scale * ((-u_t.log()) ** (1.0 / self.concentration))
        s = self.scale_np * (-np.log(u)) ** (1.0 / self.conc_np)
        return _wrap(s)

    def sample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        return self.rsample(sample_shape)

    def log_prob(self, value: Union[float, _Tensor]) -> _Tensor:
        if any(isinstance(x, _Tensor) and x.requires_grad for x in (self.scale, self.concentration, value)):
            v_t = value if isinstance(value, _Tensor) else _wrap(_to_np(value))
            k = self.concentration
            lam = self.scale
            return k.log() - lam.log() + (k - 1.0) * (v_t.log() - lam.log()) - (v_t / lam) ** k
        x = _to_np(value).astype(np.float64)
        k = self.conc_np
        lam = self.scale_np
        lp = np.log(k) - np.log(lam) + (k - 1.0) * (np.log(np.maximum(1e-15, x)) - np.log(lam)) - (x / lam) ** k
        lp = np.where(x <= 0.0, -np.inf, lp)
        return _wrap(lp)


class ContinuousBernoulli(Distribution):
    """Continuous Bernoulli distribution over [0, 1]."""

    def __init__(self, probs: Optional[Union[float, _Tensor]] = None, logits: Optional[Union[float, _Tensor]] = None, validate_args: Optional[bool] = None):
        if probs is None and logits is None:
            raise ValueError("One of `probs` or `logits` must be specified.")
        if probs is not None:
            self.probs_np = _to_np(probs).astype(np.float64)
        else:
            l = _to_np(logits).astype(np.float64)
            self.probs_np = 1.0 / (1.0 + np.exp(-l))
        super().__init__(batch_shape=self.probs_np.shape, event_shape=(), validate_args=validate_args)

    def sample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        shape = tuple(sample_shape) + self._batch_shape
        u = np.random.uniform(0.0, 1.0, size=shape)
        p = self.probs_np
        # Both where-branches evaluate: silence the 0/0 at p == 0.5 (discarded).
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(np.abs(p - 0.5) < 1e-6, u, np.log(1.0 + (2.0 * p - 1.0) * u / (1.0 - p)) / np.log(np.maximum(1e-15, p / (1.0 - p))))
        return _wrap(s)

    def log_prob(self, value: Union[float, _Tensor]) -> _Tensor:
        x = _to_np(value).astype(np.float64)
        p = self.probs_np
        # Both where-branches evaluate: silence the 0/0 at p == 0.5 (discarded).
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.where(np.abs(p - 0.5) < 1e-6, 2.0, 2.0 * np.arctanh(1.0 - 2.0 * p) / (1.0 - 2.0 * p))
        lp = np.log(np.maximum(1e-15, c)) + x * np.log(np.maximum(1e-15, p)) + (1.0 - x) * np.log(np.maximum(1e-15, 1.0 - p))
        return _wrap(lp)


__all__ = [
    "Distribution",
    "Normal",
    "Bernoulli",
    "Categorical",
    "Uniform",
    "Exponential",
    "Laplace",
    "Cauchy",
    "Beta",
    "Gamma",
    "Dirichlet",
    "Poisson",
    "Geometric",
    "Gumbel",
    "LogNormal",
    "MultivariateNormal",
    "LowRankMultivariateNormal",
    "StudentT",
    "Independent",
    "TransformedDistribution",
    "Binomial",
    "Multinomial",
    "HalfNormal",
    "HalfCauchy",
    "Weibull",
    "ContinuousBernoulli",
    "kl_divergence",
    "register_kl",
]


# ---------------------------------------------------------------------------
# mean / variance / stddev of the distributions above
# ---------------------------------------------------------------------------

def _stddev(self) -> _Tensor:
    return _wrap(np.sqrt(_to_np(self.variance)))


def _no_moment(name):
    def get(self):
        raise NotImplementedError(f"{type(self).__name__} has no closed-form {name}")
    return property(get)


Distribution.stddev = property(_stddev)
Distribution.mean = _no_moment("mean")
Distribution.variance = _no_moment("variance")


def _moments(cls, mean, variance):
    cls.mean = property(lambda self: _wrap(np.asarray(mean(self), dtype=np.float64)))
    cls.variance = property(lambda self: _wrap(np.asarray(variance(self), dtype=np.float64)))


def _lg(x):
    from math import lgamma
    return np.vectorize(lgamma, otypes=[np.float64])(np.asarray(x, dtype=np.float64))


def _dg(x):
    from .special import digamma
    return np.asarray(_to_np(digamma(_wrap(np.asarray(x, dtype=np.float64)))), dtype=np.float64)


_EULER = 0.5772156649015329
_moments(Normal, lambda d: d.loc_np, lambda d: d.scale_np ** 2)
_moments(Bernoulli, lambda d: d.probs_np, lambda d: d.probs_np * (1 - d.probs_np))
_moments(Uniform, lambda d: (d.low_np + d.high_np) / 2, lambda d: (d.high_np - d.low_np) ** 2 / 12)
_moments(Exponential, lambda d: 1 / d.rate_np, lambda d: d.rate_np ** -2.0)
_moments(Laplace, lambda d: d.loc_np, lambda d: 2 * d.scale_np ** 2)
_moments(Cauchy, lambda d: np.full(d.batch_shape, np.nan), lambda d: np.full(d.batch_shape, np.inf))
_moments(Beta, lambda d: d.alpha_np / (d.alpha_np + d.beta_np),
         lambda d: d.alpha_np * d.beta_np / ((d.alpha_np + d.beta_np) ** 2 * (d.alpha_np + d.beta_np + 1)))
_moments(Gamma, lambda d: d.c_np / d.r_np, lambda d: d.c_np / d.r_np ** 2)
_moments(Dirichlet, lambda d: d.conc_np / d.conc_np.sum(-1, keepdims=True),
         lambda d: (lambda a, a0: a * (a0 - a) / (a0 ** 2 * (a0 + 1)))(d.conc_np, d.conc_np.sum(-1, keepdims=True)))
_moments(Poisson, lambda d: d.rate_np, lambda d: d.rate_np)
_moments(Geometric, lambda d: 1 / d.probs_np - 1, lambda d: (1 / d.probs_np - 1) / d.probs_np)
_moments(Gumbel, lambda d: d.loc_np + d.scale_np * _EULER, lambda d: (math.pi * d.scale_np) ** 2 / 6)
_moments(LogNormal, lambda d: np.exp(d.base.loc_np + d.base.scale_np ** 2 / 2),
         lambda d: (np.exp(d.base.scale_np ** 2) - 1) * np.exp(2 * d.base.loc_np + d.base.scale_np ** 2))
_moments(MultivariateNormal, lambda d: np.broadcast_to(d.loc_np, d.batch_shape + d.event_shape),
         lambda d: np.broadcast_to(np.diagonal(d.cov_np, axis1=-2, axis2=-1), d.batch_shape + d.event_shape))
_moments(StudentT, lambda d: np.where(d.df_np > 1, d.loc_np, np.nan),
         lambda d: np.where(d.df_np > 2, d.scale_np ** 2 * d.df_np / np.where(d.df_np > 2, d.df_np - 2, 1),
                            np.where(d.df_np > 1, np.inf, np.nan)))
_moments(Binomial, lambda d: d.count_np * d.probs_np, lambda d: d.count_np * d.probs_np * (1 - d.probs_np))
_moments(Multinomial, lambda d: d.total_count * d.probs_np, lambda d: d.total_count * d.probs_np * (1 - d.probs_np))
_moments(HalfNormal, lambda d: d.scale_np * math.sqrt(2 / math.pi), lambda d: d.scale_np ** 2 * (1 - 2 / math.pi))
_moments(HalfCauchy, lambda d: np.full(np.shape(d.scale_np), np.inf), lambda d: np.full(np.shape(d.scale_np), np.inf))
_moments(Weibull, lambda d: d.scale_np * np.exp(_lg(1 + 1 / d.conc_np)),
         lambda d: d.scale_np ** 2 * (np.exp(_lg(1 + 2 / d.conc_np)) - np.exp(2 * _lg(1 + 1 / d.conc_np))))


def _lowrank_variance(d):
    return (d.w_np ** 2).sum(-1) + d.d_np


_moments(LowRankMultivariateNormal, lambda d: d.loc_np, _lowrank_variance)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def _as_t(x: Any) -> _Tensor:
    return x if isinstance(x, _Tensor) else _wrap(np.asarray(x, dtype=np.float32))


def _sum_rightmost(x: _Tensor, n: int) -> _Tensor:
    for _ in range(n):
        x = x.sum(-1)
    return x


def _softplus(x: _Tensor) -> _Tensor:
    # log(1 + exp(x)) without overflow
    return x.clamp_min(0.0) + (-(x.abs())).exp().log1p()


class Transform:
    """A differentiable map `y = t(x)` with `t.inv(y)` and, for bijections, `log_abs_det_jacobian(x, y)`.

    `event_dim` is how many rightmost dimensions form one event (0 for elementwise maps)."""

    bijective = True
    event_dim = 0
    sign = 1

    def __call__(self, x: Any) -> _Tensor:
        return self._call(_as_t(x))

    def _call(self, x: _Tensor) -> _Tensor:
        raise NotImplementedError

    def _inverse(self, y: _Tensor) -> _Tensor:
        raise NotImplementedError

    @property
    def inv(self) -> "Transform":
        return _InverseTransform(self)

    def log_abs_det_jacobian(self, x: Any, y: Any) -> _Tensor:
        raise NotImplementedError(f"{type(self).__name__} has no log_abs_det_jacobian")

    def forward_shape(self, shape: Sequence[int]) -> Tuple[int, ...]:
        return tuple(shape)

    def inverse_shape(self, shape: Sequence[int]) -> Tuple[int, ...]:
        return tuple(shape)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class _InverseTransform(Transform):
    def __init__(self, base: Transform):
        self._base = base
        self.bijective = base.bijective
        self.event_dim = base.event_dim
        self.sign = base.sign

    def _call(self, x):
        return self._base._inverse(x)

    def _inverse(self, y):
        return self._base._call(y)

    @property
    def inv(self):
        return self._base

    def log_abs_det_jacobian(self, x, y):
        return -self._base.log_abs_det_jacobian(_as_t(y), _as_t(x))

    def forward_shape(self, shape):
        return self._base.inverse_shape(shape)

    def inverse_shape(self, shape):
        return self._base.forward_shape(shape)


class ExpTransform(Transform):
    """y = exp(x)."""

    def _call(self, x):
        return x.exp()

    def _inverse(self, y):
        return y.log()

    def log_abs_det_jacobian(self, x, y):
        return _as_t(x)


class AffineTransform(Transform):
    """y = loc + scale * x (`event_dim` rightmost dimensions are summed in the log-determinant)."""

    def __init__(self, loc: Any, scale: Any, event_dim: int = 0):
        self.loc, self.scale, self.event_dim = loc, scale, event_dim
        self.sign = 1 if np.all(_to_np(scale) > 0) else (-1 if np.all(_to_np(scale) < 0) else 0)

    def _call(self, x):
        return x * self.scale + self.loc

    def _inverse(self, y):
        return (y - self.loc) / self.scale

    def log_abs_det_jacobian(self, x, y):
        x = _as_t(x)
        s = self.scale.abs().log() if isinstance(self.scale, _Tensor) else math.log(abs(float(self.scale)))
        out = x * 0 + s
        return _sum_rightmost(out, self.event_dim)


class SigmoidTransform(Transform):
    """y = sigmoid(x)."""

    def _call(self, x):
        return x.sigmoid()

    def _inverse(self, y):
        eps = float(np.finfo(np.float32).eps)
        y = y.clamp(eps, 1 - eps)
        return y.log() - (-y).log1p()

    def log_abs_det_jacobian(self, x, y):
        x = _as_t(x)
        return -_softplus(-x) - _softplus(x)


class TanhTransform(Transform):
    """y = tanh(x)."""

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return ((y.log1p()) - ((-y).log1p())) * 0.5

    def log_abs_det_jacobian(self, x, y):
        x = _as_t(x)
        return (math.log(2.0) - x - _softplus(x * -2.0)) * 2.0


class SoftplusTransform(Transform):
    """y = log(1 + exp(x))."""

    def _call(self, x):
        return _softplus(x)

    def _inverse(self, y):
        return y + (-((-y).expm1())).log()

    def log_abs_det_jacobian(self, x, y):
        return -_softplus(-_as_t(x))


class AbsTransform(Transform):
    """y = |x| (not a bijection: `inv` returns the non-negative preimage)."""

    bijective = False

    def _call(self, x):
        return x.abs()

    def _inverse(self, y):
        return y


class PowerTransform(Transform):
    """y = x ** exponent, for positive x."""

    def __init__(self, exponent: Any):
        self.exponent = exponent

    def _call(self, x):
        return x.pow(self.exponent)

    def _inverse(self, y):
        return y.pow(1.0 / self.exponent if not isinstance(self.exponent, _Tensor) else self.exponent.reciprocal())

    def log_abs_det_jacobian(self, x, y):
        x, y = _as_t(x), _as_t(y)
        return (y * self.exponent / x).abs().log()


class SoftmaxTransform(Transform):
    """y = softmax(x) over the last dimension (not a bijection; `inv` returns log(y))."""

    bijective = False
    event_dim = 1

    def _call(self, x):
        return x.softmax(-1)

    def _inverse(self, y):
        return y.log()


def _pad_last(x: _Tensor, value: float, before: bool) -> _Tensor:
    from .oxideon import cat
    shape = list(x.shape)
    shape[-1] = 1
    pad = _wrap(np.full(shape, value, dtype=np.float32))
    return cat([pad, x] if before else [x, pad], -1)


class StickBreakingTransform(Transform):
    """Maps a vector in R^(K-1) onto the K-simplex (positive entries summing to 1) by stick breaking."""

    event_dim = 1

    def _offset_log(self, k: int) -> _Tensor:
        return _wrap(np.log(np.arange(k, 0, -1, dtype=np.float64)))

    def _call(self, x):
        k = x.shape[-1]
        z = (x - self._offset_log(k)).sigmoid()
        z_cumprod = (1.0 - z).cumprod(-1)
        return _pad_last(z, 1.0, before=False) * _pad_last(z_cumprod, 1.0, before=True)

    def _inverse(self, y):
        k = y.shape[-1] - 1
        y_crop = y.narrow(len(y.shape) - 1, 0, k)
        sf = (1.0 - y_crop.cumsum(-1)).clamp_min(float(np.finfo(np.float32).tiny))
        return y_crop.log() - sf.log() + self._offset_log(k)

    def log_abs_det_jacobian(self, x, y):
        x, y = _as_t(x), _as_t(y)
        k = x.shape[-1]
        xs = x - self._offset_log(k)
        y_crop = y.narrow(len(y.shape) - 1, 0, k)
        return (-xs - _softplus(-xs) + y_crop.log()).sum(-1)

    def forward_shape(self, shape):
        return tuple(shape[:-1]) + (shape[-1] + 1,)

    def inverse_shape(self, shape):
        return tuple(shape[:-1]) + (shape[-1] - 1,)


class ComposeTransform(Transform):
    """Applies `parts` one after the other."""

    def __init__(self, parts: Sequence[Transform]):
        self.parts = list(parts)
        self.event_dim = max([p.event_dim for p in self.parts], default=0)
        self.bijective = all(p.bijective for p in self.parts)
        sign = 1
        for p in self.parts:
            sign *= p.sign
        self.sign = sign

    def _call(self, x):
        for p in self.parts:
            x = p(x)
        return x

    def _inverse(self, y):
        for p in reversed(self.parts):
            y = p.inv(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        x = _as_t(x)
        total = None
        for p in self.parts:
            y_p = p(x)
            term = _sum_rightmost(p.log_abs_det_jacobian(x, y_p), self.event_dim - p.event_dim)
            total = term if total is None else total + term
            x = y_p
        return total if total is not None else _wrap(np.zeros(tuple(x.shape[:len(x.shape) - self.event_dim])))

    def forward_shape(self, shape):
        for p in self.parts:
            shape = p.forward_shape(shape)
        return tuple(shape)

    def inverse_shape(self, shape):
        for p in reversed(self.parts):
            shape = p.inverse_shape(shape)
        return tuple(shape)


class IndependentTransform(Transform):
    """Treats `reinterpreted_batch_ndims` extra rightmost dimensions of `base_transform` as part of the event."""

    def __init__(self, base_transform: Transform, reinterpreted_batch_ndims: int):
        self.base_transform = base_transform
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        self.event_dim = base_transform.event_dim + reinterpreted_batch_ndims
        self.bijective, self.sign = base_transform.bijective, base_transform.sign

    def _call(self, x):
        return self.base_transform(x)

    def _inverse(self, y):
        return self.base_transform.inv(y)

    def log_abs_det_jacobian(self, x, y):
        return _sum_rightmost(self.base_transform.log_abs_det_jacobian(x, y), self.reinterpreted_batch_ndims)


class ReshapeTransform(Transform):
    """Reshapes the rightmost `in_shape` dimensions to `out_shape` (same number of elements)."""

    def __init__(self, in_shape: Sequence[int], out_shape: Sequence[int]):
        self.in_shape, self.out_shape = tuple(in_shape), tuple(out_shape)
        if int(np.prod(self.in_shape)) != int(np.prod(self.out_shape)):
            raise ValueError("in_shape and out_shape have different numbers of elements")
        self.event_dim = len(self.in_shape)

    def _call(self, x):
        batch = list(x.shape[:len(x.shape) - len(self.in_shape)])
        return x.reshape(batch + list(self.out_shape))

    def _inverse(self, y):
        batch = list(y.shape[:len(y.shape) - len(self.out_shape)])
        return y.reshape(batch + list(self.in_shape))

    def log_abs_det_jacobian(self, x, y):
        x = _as_t(x)
        return _wrap(np.zeros(tuple(x.shape[:len(x.shape) - len(self.in_shape)])))

    def forward_shape(self, shape):
        return tuple(shape[:len(shape) - len(self.in_shape)]) + self.out_shape

    def inverse_shape(self, shape):
        return tuple(shape[:len(shape) - len(self.out_shape)]) + self.in_shape


class CatTransform(Transform):
    """Applies `tseq[i]` to the i-th block of `lengths[i]` entries along `dim` and concatenates the results."""

    def __init__(self, tseq: Sequence[Transform], dim: int = 0, lengths: Optional[Sequence[int]] = None):
        self.transforms = list(tseq)
        self.dim = dim
        self.lengths = list(lengths) if lengths is not None else [1] * len(self.transforms)
        self.event_dim = max([t.event_dim for t in self.transforms], default=0)

    def _apply(self, x, fn):
        from .oxideon import cat
        parts, start = [], 0
        d = self.dim % len(x.shape)
        for t, n in zip(self.transforms, self.lengths):
            parts.append(fn(t, x.narrow(d, start, n)))
            start += n
        return cat(parts, d)

    def _call(self, x):
        return self._apply(x, lambda t, v: t(v))

    def _inverse(self, y):
        return self._apply(y, lambda t, v: t.inv(v))

    def log_abs_det_jacobian(self, x, y):
        from .oxideon import cat
        x, y = _as_t(x), _as_t(y)
        d = self.dim % len(x.shape)
        parts, start = [], 0
        for t, n in zip(self.transforms, self.lengths):
            parts.append(t.log_abs_det_jacobian(x.narrow(d, start, n), y.narrow(d, start, n)))
            start += n
        return cat(parts, d)


class StackTransform(Transform):
    """Applies `tseq[i]` to `x.select(dim, i)` and stacks the results along `dim`."""

    def __init__(self, tseq: Sequence[Transform], dim: int = 0):
        self.transforms = list(tseq)
        self.dim = dim

    def _apply(self, x, fn):
        from .oxideon import stack
        d = self.dim % len(x.shape)
        return stack([fn(t, x.narrow(d, i, 1).squeeze(d)) for i, t in enumerate(self.transforms)], d)

    def _call(self, x):
        return self._apply(x, lambda t, v: t(v))

    def _inverse(self, y):
        return self._apply(y, lambda t, v: t.inv(v))

    def log_abs_det_jacobian(self, x, y):
        from .oxideon import stack
        x, y = _as_t(x), _as_t(y)
        d = self.dim % len(x.shape)
        return stack([t.log_abs_det_jacobian(x.narrow(d, i, 1).squeeze(d), y.narrow(d, i, 1).squeeze(d))
                      for i, t in enumerate(self.transforms)], d)


class CumulativeDistributionTransform(Transform):
    """y = distribution.cdf(x), a map onto (0, 1); its log-determinant is the log-density."""

    def __init__(self, distribution: Distribution):
        self.distribution = distribution

    def _call(self, x):
        return self.distribution.cdf(x)

    def _inverse(self, y):
        return self.distribution.icdf(y)

    def log_abs_det_jacobian(self, x, y):
        return self.distribution.log_prob(x)


def _tril_vec_to_matrix(vec: np.ndarray, diag: int) -> np.ndarray:
    n_vec = vec.shape[-1]
    n = int(round((math.sqrt(1 + 8 * n_vec) - 1) / 2 - diag))  # matrix size so that tril(-diag) has n_vec entries
    while (n * (n + 1) // 2 + n * diag) != n_vec and n < 1000:
        n += 1
    rows, cols = np.tril_indices(n, diag)
    out = np.zeros(vec.shape[:-1] + (n, n), dtype=vec.dtype)
    out[..., rows, cols] = vec
    return out


def _tril_matrix_to_vec(mat: np.ndarray, diag: int) -> np.ndarray:
    n = mat.shape[-1]
    rows, cols = np.tril_indices(n, diag)
    return mat[..., rows, cols]


class CorrCholeskyTransform(Transform):
    """Maps a vector of D(D-1)/2 unconstrained numbers onto the Cholesky factor of a D x D correlation matrix
    (lower triangular, positive diagonal, unit-norm rows). No graph."""

    event_dim = 1

    def _call(self, x):
        v = np.tanh(_to_np(x).astype(np.float64))
        eps = np.finfo(np.float32).eps
        v = np.clip(v, -1 + eps, 1 - eps)
        r = _tril_vec_to_matrix(v, -1)
        z = r ** 2
        z1m_cumprod_sqrt = np.cumprod(np.sqrt(1 - z), axis=-1)
        r = r + np.eye(r.shape[-1])
        pad = np.concatenate([np.ones(z1m_cumprod_sqrt.shape[:-1] + (1,)), z1m_cumprod_sqrt[..., :-1]], -1)
        return _wrap(r * pad)

    def _inverse(self, y):
        y = _to_np(y).astype(np.float64)
        y_cumsum = 1 - np.cumsum(y * y, axis=-1)
        shifted = np.concatenate([np.ones(y_cumsum.shape[:-1] + (1,)), y_cumsum[..., :-1]], -1)
        t = _tril_matrix_to_vec(y, -1) / np.sqrt(_tril_matrix_to_vec(shifted, -1))
        return _wrap((np.log1p(t) - np.log1p(-t)) / 2)

    def log_abs_det_jacobian(self, x, y):
        x = _to_np(x).astype(np.float64)
        y = _to_np(y).astype(np.float64)
        y1m_cumsum = 1 - np.cumsum(y * y, axis=-1)
        stick = 0.5 * np.log(_tril_matrix_to_vec(y1m_cumsum, -2)).sum(-1)
        tanh_ld = -2 * (x + np.logaddexp(0, -2 * x) - math.log(2.0)).sum(-1)
        return _wrap(stick + tanh_ld)

    def forward_shape(self, shape):
        n = int(round((1 + math.sqrt(1 + 8 * shape[-1])) / 2))
        return tuple(shape[:-1]) + (n, n)

    def inverse_shape(self, shape):
        return tuple(shape[:-2]) + (shape[-1] * (shape[-1] - 1) // 2,)


class LowerCholeskyTransform(Transform):
    """Maps a square matrix onto a lower-triangular one with a positive diagonal (exp of the diagonal). No graph."""

    event_dim = 2

    def _call(self, x):
        a = _to_np(x).astype(np.float64)
        d = np.exp(np.diagonal(a, axis1=-2, axis2=-1))
        return _wrap(np.tril(a, -1) + d[..., None] * np.eye(a.shape[-1]))

    def _inverse(self, y):
        a = _to_np(y).astype(np.float64)
        d = np.log(np.diagonal(a, axis1=-2, axis2=-1))
        return _wrap(np.tril(a, -1) + d[..., None] * np.eye(a.shape[-1]))


class PositiveDefiniteTransform(Transform):
    """Maps a square matrix onto a symmetric positive-definite one: L L^T with L = LowerCholeskyTransform(x). No graph."""

    event_dim = 2

    def _call(self, x):
        low = _to_np(LowerCholeskyTransform()(x)).astype(np.float64)
        return _wrap(low @ np.swapaxes(low, -1, -2))

    def _inverse(self, y):
        low = np.linalg.cholesky(_to_np(y).astype(np.float64))
        return LowerCholeskyTransform().inv(_wrap(low))


# TransformedDistribution: log_prob through the transforms (and mean-free shapes)

def _td_init(self, base_distribution: Distribution, transforms: Any):
    self.base_dist = base_distribution
    self.transforms = [transforms] if isinstance(transforms, Transform) else list(transforms)
    shape = tuple(base_distribution.batch_shape) + tuple(base_distribution.event_shape)
    event_dim = len(base_distribution.event_shape)
    for t in self.transforms:
        if hasattr(t, "forward_shape"):
            shape = t.forward_shape(shape)
        event_dim = max(event_dim, getattr(t, "event_dim", 0))
    cut = len(shape) - event_dim
    Distribution.__init__(self, batch_shape=shape[:cut], event_shape=shape[cut:])
    self.has_rsample = getattr(base_distribution, "has_rsample", False)


def _td_log_prob(self, value: Any) -> _Tensor:
    y = _as_t(value)
    event_dim = len(self.event_shape)
    lp = None
    for t in reversed(self.transforms):
        x = t.inv(y)
        term = _sum_rightmost(t.log_abs_det_jacobian(x, y), event_dim - getattr(t, "event_dim", 0))
        lp = -term if lp is None else lp - term
        event_dim += len(x.shape) - len(y.shape) if hasattr(x, "shape") else 0
        y = x
    base = self.base_dist.log_prob(y)
    base = _sum_rightmost(base, event_dim - len(self.base_dist.event_shape))
    return base if lp is None else base + lp


TransformedDistribution.__init__ = _td_init
TransformedDistribution.log_prob = _td_log_prob


def _td_cdf(self, value):
    y = _as_t(value)
    for t in reversed(self.transforms):
        y = t.inv(y)
    out = self.base_dist.cdf(y)
    sign = 1
    for t in self.transforms:
        sign *= t.sign
    return out if sign == 1 else 1.0 - out


TransformedDistribution.cdf = _td_cdf


# ---------------------------------------------------------------------------
# More distributions
# ---------------------------------------------------------------------------

def _f64(x: Any) -> np.ndarray:
    return np.asarray(_to_np(x), dtype=np.float64)


def _bcast(*xs):
    arrs = [_f64(x) for x in xs]
    shape = np.broadcast_shapes(*[a.shape for a in arrs])
    return [np.broadcast_to(a, shape).copy() for a in arrs] + [shape]


class ExponentialFamily(Distribution):
    """Base class for exponential-family distributions p(x) = h(x) exp(<eta, T(x)> - A(eta)). Subclasses define
    `_natural_params`, `_log_normalizer(*eta)` and `_mean_carrier_measure`; the entropy then follows from A by
    numerical differentiation."""

    @property
    def _natural_params(self) -> Tuple[np.ndarray, ...]:
        raise NotImplementedError

    def _log_normalizer(self, *natural_params: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def _mean_carrier_measure(self) -> float:
        raise NotImplementedError

    def entropy(self) -> _Tensor:
        eta = [np.asarray(e, dtype=np.float64) for e in self._natural_params]
        a = self._log_normalizer(*eta)
        result = -self._mean_carrier_measure + a
        for i, e in enumerate(eta):
            h = 1e-6 * np.maximum(1.0, np.abs(e))
            up = list(eta)
            dn = list(eta)
            up[i], dn[i] = e + h, e - h
            grad = (self._log_normalizer(*up) - self._log_normalizer(*dn)) / (2 * h)
            result = result - e * grad
        return _wrap(result)


class Chi2(Gamma):
    """The chi-squared distribution with `df` degrees of freedom (Gamma(df / 2, 1 / 2))."""

    def __init__(self, df: Union[_Tensor, float]):
        self.df_np = _f64(df)
        super().__init__(df * 0.5 if isinstance(df, _Tensor) else self.df_np * 0.5, 0.5)
        self.df = df if isinstance(df, _Tensor) else _wrap(self.df_np)


class InverseGamma(Distribution):
    """The inverse-gamma distribution: 1 / X for X ~ Gamma(concentration, rate)."""

    has_rsample = True

    def __init__(self, concentration: Any, rate: Any):
        self.c_np, self.r_np, shape = _bcast(concentration, rate)
        self.concentration, self.rate = _wrap(self.c_np), _wrap(self.r_np)
        super().__init__(batch_shape=shape)


    @property
    def mean(self):
        c, r = self.c_np, self.r_np
        return _wrap(np.where(c > 1, r / np.where(c > 1, c - 1, 1), np.inf))

    @property
    def variance(self):
        c, r = self.c_np, self.r_np
        return _wrap(np.where(c > 2, r ** 2 / (np.where(c > 2, c - 1, 1) ** 2 * np.where(c > 2, c - 2, 1)), np.inf))

    def entropy(self):
        c, r = self.c_np, self.r_np
        return _wrap(c + np.log(r) + _lg(c) - (1 + c) * _dg(c))


class Pareto(Distribution):
    """The Pareto type I distribution: density alpha * scale^alpha / x^(alpha + 1) for x >= scale."""

    has_rsample = True

    def __init__(self, scale: Any, alpha: Any):
        self.scale_np, self.alpha_np, shape = _bcast(scale, alpha)
        self.scale, self.alpha = _wrap(self.scale_np), _wrap(self.alpha_np)
        super().__init__(batch_shape=shape)


    @property
    def mean(self):
        a = self.alpha_np
        return _wrap(np.where(a > 1, a * self.scale_np / np.where(a > 1, a - 1, 1), np.inf))

    @property
    def variance(self):
        a = self.alpha_np
        return _wrap(np.where(a > 2, self.scale_np ** 2 * a / (np.where(a > 2, a - 1, 1) ** 2 * np.where(a > 2, a - 2, 1)), np.inf))

    def entropy(self):
        return _wrap(np.log(self.scale_np / self.alpha_np) + 1 / self.alpha_np + 1)


class Kumaraswamy(Distribution):
    """The Kumaraswamy distribution on (0, 1): density a b x^(a-1) (1 - x^a)^(b-1), a Beta look-alike with a closed-form cdf."""

    has_rsample = True

    def __init__(self, concentration1: Any, concentration0: Any):
        self.a_np, self.b_np, shape = _bcast(concentration1, concentration0)
        self.concentration1, self.concentration0 = _wrap(self.a_np), _wrap(self.b_np)
        super().__init__(batch_shape=shape)


    def _moment(self, n):
        a, b = self.a_np, self.b_np
        return b * np.exp(_lg(1 + n / a) + _lg(b) - _lg(1 + n / a + b))

    @property
    def mean(self):
        return _wrap(self._moment(1))

    @property
    def variance(self):
        return _wrap(self._moment(2) - self._moment(1) ** 2)

    def entropy(self):
        a, b = self.a_np, self.b_np
        h0 = _dg(b + 1) + _EULER
        return _wrap((1 - 1 / b) + (1 - 1 / a) * h0 - np.log(b) - np.log(a))


class FisherSnedecor(Distribution):
    """The F distribution with `df1` and `df2` degrees of freedom: (chi2(df1) / df1) / (chi2(df2) / df2)."""

    has_rsample = True

    def __init__(self, df1: Any, df2: Any):
        self.d1, self.d2, shape = _bcast(df1, df2)
        self.df1, self.df2 = _wrap(self.d1), _wrap(self.d2)
        super().__init__(batch_shape=shape)


    @property
    def mean(self):
        d2 = self.d2
        return _wrap(np.where(d2 > 2, d2 / np.where(d2 > 2, d2 - 2, 1), np.nan))

    @property
    def variance(self):
        d1, d2 = self.d1, self.d2
        ok = d2 > 4
        s = np.where(ok, d2, 5)
        return _wrap(np.where(ok, 2 * s ** 2 * (d1 + s - 2) / (d1 * (s - 2) ** 2 * (s - 4)), np.nan))


class GeneralizedPareto(Distribution):
    """The generalized Pareto distribution with location, scale and shape (`concentration`) parameters."""

    has_rsample = True

    def __init__(self, loc: Any, scale: Any, concentration: Any):
        self.loc_np, self.scale_np, self.c_np, shape = _bcast(loc, scale, concentration)
        self.loc, self.scale, self.concentration = _wrap(self.loc_np), _wrap(self.scale_np), _wrap(self.c_np)
        super().__init__(batch_shape=shape)


    @property
    def mean(self):
        c = self.c_np
        return _wrap(np.where(c < 1, self.loc_np + self.scale_np / (1 - np.where(c < 1, c, 0)), np.inf))

    @property
    def variance(self):
        c = self.c_np
        ok = c < 0.5
        s = np.where(ok, c, 0)
        return _wrap(np.where(ok, self.scale_np ** 2 / ((1 - s) ** 2 * (1 - 2 * s)), np.inf))

    def entropy(self):
        return _wrap(np.log(self.scale_np) + self.c_np + 1)


def _probs_logits(probs, logits, categorical: bool):
    if (probs is None) == (logits is None):
        raise ValueError("pass exactly one of probs or logits")
    if probs is not None:
        p = _f64(probs)
        if categorical:
            p = p / p.sum(-1, keepdims=True)
            return p, np.log(np.clip(p, 1e-300, None))
        return p, np.log(np.clip(p, 1e-300, None)) - np.log1p(-np.clip(p, None, 1 - 1e-16))
    lg = _f64(logits)
    if categorical:
        lg = lg - np.log(np.exp(lg - lg.max(-1, keepdims=True)).sum(-1, keepdims=True)) - lg.max(-1, keepdims=True)
        return np.exp(lg), lg
    return 1 / (1 + np.exp(-lg)), lg


class NegativeBinomial(Distribution):
    """The number of successes (probability `probs`) before `total_count` failures."""

    def __init__(self, total_count: Any, probs: Any = None, logits: Any = None):
        p, lg = _probs_logits(probs, logits, categorical=False)
        self.count_np, self.probs_np, self.logits_np, shape = _bcast(total_count, p, lg)
        self.total_count, self.probs, self.logits = _wrap(self.count_np), _wrap(self.probs_np), _wrap(self.logits_np)
        super().__init__(batch_shape=shape)

    @property
    def mean(self):
        return _wrap(self.count_np * np.exp(self.logits_np))

    @property
    def variance(self):
        return _wrap(self.count_np * np.exp(self.logits_np) / (1 - self.probs_np))


class OneHotCategorical(Distribution):
    """A categorical distribution whose samples are one-hot vectors of length K."""

    has_enumerate_support = True

    def __init__(self, probs: Any = None, logits: Any = None):
        self.probs_np, self.logits_np = _probs_logits(probs, logits, categorical=True)
        self.probs, self.logits = _wrap(self.probs_np), _wrap(self.logits_np)
        super().__init__(batch_shape=self.probs_np.shape[:-1], event_shape=self.probs_np.shape[-1:])

    def entropy(self):
        p = self.probs_np
        return _wrap(-(np.where(p > 0, p * self.logits_np, 0.0)).sum(-1))

    @property
    def mean(self):
        return _wrap(self.probs_np)

    @property
    def variance(self):
        return _wrap(self.probs_np * (1 - self.probs_np))

    def enumerate_support(self, expand: bool = True):
        k = self.probs_np.shape[-1]
        vals = np.eye(k).reshape((k,) + (1,) * len(self.batch_shape) + (k,))
        if expand:
            vals = np.broadcast_to(vals, (k,) + self.batch_shape + (k,))
        return _wrap(vals)


class OneHotCategoricalStraightThrough(OneHotCategorical):
    """`OneHotCategorical` whose `rsample` passes gradients to `probs` with the straight-through estimator:
    the forward value is the one-hot sample, the backward pass treats it as `probs`."""

    has_rsample = True

    def __init__(self, probs: Any = None, logits: Any = None):
        self._probs_t = probs if isinstance(probs, _Tensor) else None
        super().__init__(probs=probs, logits=logits)

    def rsample(self, sample_shape=()):
        s = self.sample(sample_shape)
        p = self._probs_t if self._probs_t is not None else self.probs
        return s + p - p.detach()


class LogitRelaxedBernoulli(Distribution):
    """The logit of a relaxed (Concrete / Gumbel-sigmoid) Bernoulli sample at a given temperature."""

    has_rsample = True

    def __init__(self, temperature: Any, probs: Any = None, logits: Any = None):
        p, lg = _probs_logits(probs, logits, categorical=False)
        self.probs_np, self.logits_np, shape = _bcast(p, lg)
        self.temperature_np = _f64(temperature)
        self.temperature, self.probs, self.logits = _wrap(self.temperature_np), _wrap(self.probs_np), _wrap(self.logits_np)
        super().__init__(batch_shape=shape)


class RelaxedBernoulli(TransformedDistribution):
    """A Bernoulli relaxed to (0, 1) at a temperature (sigmoid of `LogitRelaxedBernoulli`); low temperatures approach
    hard 0/1 samples while staying differentiable in the parameters."""

    has_rsample = True

    def __init__(self, temperature: Any, probs: Any = None, logits: Any = None):
        base = LogitRelaxedBernoulli(temperature, probs, logits)
        super().__init__(base, SigmoidTransform())
        self.temperature, self.probs, self.logits = base.temperature, base.probs, base.logits


class ExpRelaxedCategorical(Distribution):
    """The log of a relaxed one-hot categorical (Concrete / Gumbel-softmax) sample."""

    has_rsample = True

    def __init__(self, temperature: Any, probs: Any = None, logits: Any = None):
        self.probs_np, self.logits_np = _probs_logits(probs, logits, categorical=True)
        self.temperature_np = _f64(temperature)
        self.temperature, self.probs, self.logits = _wrap(self.temperature_np), _wrap(self.probs_np), _wrap(self.logits_np)
        super().__init__(batch_shape=self.probs_np.shape[:-1], event_shape=self.probs_np.shape[-1:])


class RelaxedOneHotCategorical(TransformedDistribution):
    """A one-hot categorical relaxed onto the simplex at a temperature (Gumbel-softmax), for differentiable discrete choices."""

    has_rsample = True

    def __init__(self, temperature: Any, probs: Any = None, logits: Any = None):
        base = ExpRelaxedCategorical(temperature, probs, logits)
        super().__init__(base, ExpTransform())
        self.temperature, self.probs, self.logits = base.temperature, base.probs, base.logits


class LogisticNormal(TransformedDistribution):
    """A distribution on the simplex: stick-breaking of a normal vector (`loc`, `scale` of length K - 1 give K outputs)."""

    has_rsample = True

    def __init__(self, loc: Any, scale: Any):
        base = Independent(Normal(loc, scale), 1)
        super().__init__(base, StickBreakingTransform())
        self.loc, self.scale = base.base_dist.loc, base.base_dist.scale


class MixtureSameFamily(Distribution):
    """A mixture of components of one family: `mixture_distribution` (Categorical over K) picks one of the K
    components batched in the rightmost batch dimension of `component_distribution`."""

    def __init__(self, mixture_distribution: Categorical, component_distribution: Distribution):
        self.mixture_distribution = mixture_distribution
        self.component_distribution = component_distribution
        k = component_distribution.batch_shape[-1]
        if mixture_distribution.probs_np.shape[-1] != k:
            raise ValueError("the mixture has a different number of components than component_distribution's last batch dimension")
        super().__init__(batch_shape=component_distribution.batch_shape[:-1], event_shape=component_distribution.event_shape)

    def _mix_probs(self) -> np.ndarray:
        # batch + (K,) -> batch + (K,) + (1,) * event_ndims, to weight the component moments
        p = self.mixture_distribution.probs_np
        return p.reshape(p.shape + (1,) * len(self.event_shape))

    @property
    def mean(self):
        p = self._mix_probs()
        return _wrap((p * _f64(self.component_distribution.mean)).sum(-1 - len(self.event_shape)))

    @property
    def variance(self):
        ax = -1 - len(self.event_shape)
        p = self._mix_probs()
        m = _f64(self.component_distribution.mean)
        v = _f64(self.component_distribution.variance)
        mean = (p * m).sum(ax, keepdims=True)
        return _wrap((p * v).sum(ax) + (p * (m - mean) ** 2).sum(ax))


def _log_i0(k: np.ndarray) -> np.ndarray:
    k = np.asarray(k, dtype=np.float64)
    from .special import i0e
    return np.log(_f64(i0e(_wrap(k)))) + np.abs(k)


def _log_i1(k: np.ndarray) -> np.ndarray:
    k = np.asarray(k, dtype=np.float64)
    from .special import i1e
    return np.log(np.abs(_f64(i1e(_wrap(k))))) + np.abs(k)


class VonMises(Distribution):
    """The von Mises (circular normal) distribution on angles, with mean `loc` and concentration `concentration`."""

    def __init__(self, loc: Any, concentration: Any):
        self.loc_np, self.k_np, shape = _bcast(loc, concentration)
        self.loc, self.concentration = _wrap(self.loc_np), _wrap(self.k_np)
        super().__init__(batch_shape=shape)

    @property
    def mean(self):
        return _wrap(self.loc_np)

    @property
    def variance(self):
        """The circular variance 1 - I1(k) / I0(k)."""
        return _wrap(1 - np.exp(_log_i1(self.k_np) - _log_i0(self.k_np)))


def _mvlgamma_np(a: np.ndarray, p: int) -> np.ndarray:
    return p * (p - 1) / 4 * math.log(math.pi) + sum(_lg(a - i / 2) for i in range(p))


def _mvdigamma_np(a: np.ndarray, p: int) -> np.ndarray:
    return sum(_dg(a - i / 2) for i in range(p))


class Wishart(Distribution):
    """The Wishart distribution over p x p positive-definite matrices with `df` degrees of freedom and a scale matrix
    (pass one of `covariance_matrix`, `precision_matrix`, `scale_tril`)."""

    has_rsample = True

    def __init__(self, df: Any, covariance_matrix: Any = None, precision_matrix: Any = None, scale_tril: Any = None):
        if sum(x is not None for x in (covariance_matrix, precision_matrix, scale_tril)) != 1:
            raise ValueError("pass exactly one of covariance_matrix, precision_matrix, scale_tril")
        if scale_tril is not None:
            tril = _f64(scale_tril)
        elif covariance_matrix is not None:
            tril = np.linalg.cholesky(_f64(covariance_matrix))
        else:
            tril = np.linalg.cholesky(np.linalg.inv(_f64(precision_matrix)))
        self.tril_np = tril
        self.cov_np = tril @ np.swapaxes(tril, -1, -2)
        p = tril.shape[-1]
        self.df_np = _f64(df)
        if np.any(self.df_np <= p - 1):
            raise ValueError(f"df must be greater than p - 1 = {p - 1}")
        self.df, self.covariance_matrix, self.scale_tril = _wrap(self.df_np), _wrap(self.cov_np), _wrap(tril)
        batch = np.broadcast_shapes(self.df_np.shape, tril.shape[:-2])
        super().__init__(batch_shape=batch, event_shape=(p, p))


    @property
    def mean(self):
        return _wrap(self.df_np[..., None, None] * self.cov_np)

    @property
    def variance(self):
        d = np.diagonal(self.cov_np, axis1=-2, axis2=-1)
        return _wrap(self.df_np[..., None, None] * (self.cov_np ** 2 + d[..., :, None] * d[..., None, :]))

    def entropy(self):
        p = self.event_shape[-1]
        nu = self.df_np
        logdiag = np.log(np.diagonal(self.tril_np, axis1=-2, axis2=-1)).sum(-1)
        return _wrap((p + 1) * logdiag + 0.5 * p * (p + 1) * math.log(2) + _mvlgamma_np(0.5 * nu, p)
                     - 0.5 * (nu - p - 1) * _mvdigamma_np(0.5 * nu, p) + 0.5 * nu * p)


class LKJCholesky(Distribution):
    """The LKJ distribution over Cholesky factors of `dim` x `dim` correlation matrices; `concentration` > 1 favours
    matrices near the identity, < 1 strong correlations (Lewandowski, Kurowicka and Joe, 2009; onion sampler)."""

    def __init__(self, dim: int, concentration: Any = 1.0):
        if dim < 2:
            raise ValueError("dim must be at least 2")
        self.dim = dim
        self.conc_np = _f64(concentration)
        self.concentration = _wrap(self.conc_np)
        super().__init__(batch_shape=self.conc_np.shape, event_shape=(dim, dim))



# ---------------------------------------------------------------------------
# Sampling and log-densities of the distributions above, in Rust: draws come from the global oxideon generator
# (`oxideon.manual_seed`) through the Rust samplers (`_standard_gamma`, `_poisson`, `_von_mises`, `rand`, `randn`), and
# every density is a chain of Rust tensor operations. Parameters are evaluated in float64; results are float32.
# ---------------------------------------------------------------------------

def _P(a: Any) -> _Tensor:
    """A parameter or value as a float64 tensor."""
    if isinstance(a, _Tensor):
        return a.detach().double()
    arr = np.asarray(a, dtype=np.float64)  # keep 0-d values 0-d (np.ascontiguousarray would make them 1-d)
    return _from_numpy(arr if arr.flags.c_contiguous else np.ascontiguousarray(arr))


def _expand(t: _Tensor, shape: Tuple[int, ...]) -> _Tensor:
    return t if tuple(t.shape) == tuple(shape) else t.expand(list(shape)).contiguous()


def _shape(d: Distribution, sample_shape: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(s) for s in sample_shape) + tuple(d.batch_shape) + tuple(d.event_shape)


def _uniform(shape: Tuple[int, ...]) -> _Tensor:
    from .primitives import rand
    u = rand(list(shape)).double() if shape else rand([1]).double().reshape([])
    return u.clamp(1e-12, 1.0 - 1e-12)


def _normal(shape: Tuple[int, ...]) -> _Tensor:
    from .primitives import randn
    return randn(list(shape)).double() if shape else randn([1]).double().reshape([])


def _gamma(conc: _Tensor, shape: Tuple[int, ...]) -> _Tensor:
    from .oxideon import _standard_gamma
    return _standard_gamma(_expand(conc, shape))


def _f32(t: _Tensor) -> _Tensor:
    return t.float()


def _where(cond: _Tensor, a: Any, b: Any) -> _Tensor:
    from .primitives import where
    return where(cond, a, b)


def _neg_inf_like(t: _Tensor) -> _Tensor:
    return t * 0.0 - float("inf")


def _softplus_t(t: _Tensor) -> _Tensor:
    return t.clamp_min(0.0) + (-(t.abs())).exp().log1p()


def _logsumexp_last(t: _Tensor, keepdim: bool = False) -> _Tensor:
    from .tensor_utils import logsumexp
    return logsumexp(t, len(t.shape) - 1, keepdim)


# InverseGamma
def _ig_rsample(self, sample_shape=()):
    shape = _shape(self, sample_shape)
    return _f32(_expand(_P(self.r_np), shape) / _gamma(_P(self.c_np), shape))


def _ig_log_prob(self, value):
    v, c, r = _P(value), _P(self.c_np), _P(self.r_np)
    vs = _where(v > 0, v, v * 0.0 + 1.0)
    out = c * r.log() - c.lgamma() - (c + 1.0) * vs.log() - r / vs
    return _f32(_where(v > 0, out, _neg_inf_like(out)))


InverseGamma.rsample = InverseGamma.sample = _ig_rsample
InverseGamma.log_prob = _ig_log_prob


# Pareto
def _pareto_rsample(self, sample_shape=()):
    shape = _shape(self, sample_shape)
    u = _uniform(shape)
    return _f32(_expand(_P(self.scale_np), shape) * (1.0 - u).pow(_expand(-1.0 / _P(self.alpha_np), shape)))


def _pareto_log_prob(self, value):
    v, s, a = _P(value), _P(self.scale_np), _P(self.alpha_np)
    vs = _where(v >= s, v, s + v * 0.0)
    out = a.log() + a * s.log() - (a + 1.0) * vs.log()
    return _f32(_where(v >= s, out, _neg_inf_like(out)))


def _pareto_cdf(self, value):
    v, s, a = _P(value), _P(self.scale_np), _P(self.alpha_np)
    vs = _where(v >= s, v, s + v * 0.0)
    return _f32(_where(v >= s, 1.0 - (s / vs).pow(a), vs * 0.0))


def _pareto_icdf(self, value):
    return _f32(_P(self.scale_np) * (1.0 - _P(value)).pow(-1.0 / _P(self.alpha_np)))


Pareto.rsample = Pareto.sample = _pareto_rsample
Pareto.log_prob, Pareto.cdf, Pareto.icdf = _pareto_log_prob, _pareto_cdf, _pareto_icdf


# Kumaraswamy
def _kuma_rsample(self, sample_shape=()):
    shape = _shape(self, sample_shape)
    a, b = _expand(_P(self.a_np), shape), _expand(_P(self.b_np), shape)
    u = _uniform(shape)
    return _f32((1.0 - (1.0 - u).pow(1.0 / b)).pow(1.0 / a))


def _kuma_log_prob(self, value):
    v, a, b = _P(value), _P(self.a_np), _P(self.b_np)
    inside = (v > 0) & (v < 1)
    vs = _where(inside, v, v * 0.0 + 0.5)
    out = a.log() + b.log() + (a - 1.0) * vs.log() + (b - 1.0) * (-(vs.pow(a))).log1p()
    return _f32(_where(inside, out, _neg_inf_like(out)))


def _kuma_cdf(self, value):
    v = _P(value).clamp(0.0, 1.0)
    return _f32(1.0 - (1.0 - v.pow(_P(self.a_np))).pow(_P(self.b_np)))


Kumaraswamy.rsample = Kumaraswamy.sample = _kuma_rsample
Kumaraswamy.log_prob, Kumaraswamy.cdf = _kuma_log_prob, _kuma_cdf


# FisherSnedecor
def _f_rsample(self, sample_shape=()):
    shape = _shape(self, sample_shape)
    d1, d2 = _expand(_P(self.d1), shape), _expand(_P(self.d2), shape)
    x1 = _gamma(d1 * 0.5, shape) * 2.0 / d1
    x2 = _gamma(d2 * 0.5, shape) * 2.0 / d2
    return _f32(x1 / x2)


def _f_log_prob(self, value):
    v, d1, d2 = _P(value), _P(self.d1), _P(self.d2)
    c1, c2 = d1 * 0.5, d2 * 0.5
    ratio = d1 / d2
    vs = _where(v > 0, v, v * 0.0 + 1.0)
    out = (c1 + c2).lgamma() - c1.lgamma() - c2.lgamma() + c1 * ratio.log() + (c1 - 1.0) * vs.log() - (c1 + c2) * (ratio * vs).log1p()
    return _f32(_where(v > 0, out, _neg_inf_like(out)))


FisherSnedecor.rsample = FisherSnedecor.sample = _f_rsample
FisherSnedecor.log_prob = _f_log_prob


# GeneralizedPareto
def _gpd_icdf(self, value):
    u, c = _P(value), _P(self.c_np)
    zero = c == 0
    safe = _where(zero, c * 0.0 + 1.0, c)
    log1mu = (-u).log1p()
    out = _where(zero, -log1mu, (-(safe * log1mu)).expm1() / safe)
    return _f32(_P(self.loc_np) + _P(self.scale_np) * out)


def _gpd_rsample(self, sample_shape=()):
    shape = _shape(self, sample_shape)
    return self.icdf(_uniform(shape))


def _gpd_z(self, value):
    return (_P(value) - _P(self.loc_np)) / _P(self.scale_np)


def _gpd_support(self, z):
    c = _P(self.c_np)
    neg = c < 0
    upper = _where(neg, -1.0 / _where(neg, c, c * 0.0 - 1.0), c * 0.0 + float("inf"))
    return (z >= 0) & (z <= upper)


def _gpd_log_prob(self, value):
    z, c = _gpd_z(self, value), _P(self.c_np)
    ok = _gpd_support(self, z)
    zs = _where(ok, z, z * 0.0)
    zero = c == 0
    safe = _where(zero, c * 0.0 + 1.0, c)
    out = -(_P(self.scale_np).log()) - _where(zero, zs, (1.0 / safe + 1.0) * (c * zs).log1p())
    return _f32(_where(ok, out, _neg_inf_like(out)))


def _gpd_cdf(self, value):
    z, c = _gpd_z(self, value), _P(self.c_np)
    zs = z.clamp_min(0.0)
    zero = c == 0
    safe = _where(zero, c * 0.0 + 1.0, c)
    inside = (c * zs + 1.0).clamp_min(0.0)
    out = _where(zero, -((-zs).expm1()), -((-(inside.log() / safe)).expm1()))
    return _f32(_where(z < 0, z * 0.0, out).clamp(0.0, 1.0))


GeneralizedPareto.icdf, GeneralizedPareto.log_prob, GeneralizedPareto.cdf = _gpd_icdf, _gpd_log_prob, _gpd_cdf
GeneralizedPareto.rsample = GeneralizedPareto.sample = _gpd_rsample


# NegativeBinomial
def _nb_sample(self, sample_shape=()):
    from .oxideon import _poisson
    shape = _shape(self, sample_shape)
    p = _expand(_P(self.probs_np), shape)
    rate = _gamma(_P(self.count_np), shape) * (p / (1.0 - p))
    return _f32(_poisson(rate))


def _nb_log_prob(self, value):
    k, r, lg = _P(value), _P(self.count_np), _P(self.logits_np)
    log1mp = -_softplus_t(lg)
    logp = -_softplus_t(lg * -1.0)
    ks = _where(k >= 0, k, k * 0.0)
    out = (ks + r).lgamma() - r.lgamma() - (ks + 1.0).lgamma() + r * log1mp + _where(ks == 0, ks * 0.0, ks * logp)
    ok = (k >= 0) & (k == k.floor())
    return _f32(_where(ok, out, _neg_inf_like(out)))


NegativeBinomial.sample = _nb_sample
NegativeBinomial.log_prob = _nb_log_prob


# OneHotCategorical
def _ohc_sample(self, sample_shape=()):
    from .primitives import rand
    k = int(self.probs_np.shape[-1])
    shape = _shape(self, sample_shape)
    probs = _expand(_P(self.probs_np), shape)
    cum = probs.cumsum(len(shape) - 1)
    u = rand(list(shape[:-1]) + [1]).double() * cum.narrow(len(shape) - 1, k - 1, 1)
    idx = (u > cum).double().sum(len(shape) - 1, keepdim=True).clamp_max(k - 1)
    arange = _from_numpy(np.arange(k, dtype=np.float64))
    return _f32((idx == arange).double())


def _ohc_log_prob(self, value):
    prod = _P(value) * _P(self.logits_np)
    return _f32(prod.sum(len(prod.shape) - 1))


OneHotCategorical.sample = _ohc_sample
OneHotCategorical.log_prob = _ohc_log_prob


# LogitRelaxedBernoulli
def _lrb_rsample(self, sample_shape=()):
    shape = _shape(self, sample_shape)
    u = _uniform(shape)
    return _f32((_expand(_P(self.logits_np), shape) + u.log() - (-u).log1p()) / _P(self.temperature_np))


def _lrb_log_prob(self, value):
    t = _P(self.temperature_np)
    diff = _P(self.logits_np) - _P(value) * t
    return _f32(t.log() + diff - _softplus_t(diff) * 2.0)


LogitRelaxedBernoulli.rsample = LogitRelaxedBernoulli.sample = _lrb_rsample
LogitRelaxedBernoulli.log_prob = _lrb_log_prob


# ExpRelaxedCategorical
def _erc_rsample(self, sample_shape=()):
    shape = _shape(self, sample_shape)
    u = _uniform(shape)
    scores = (_expand(_P(self.logits_np), shape) - (-(u.log())).log()) / _P(self.temperature_np)
    return _f32(scores - _logsumexp_last(scores, keepdim=True))


def _erc_log_prob(self, value):
    v = _P(value)
    k = int(self.probs_np.shape[-1])
    t = _P(self.temperature_np)
    log_scale = math.lgamma(k) - t.log() * float(-(k - 1))
    score = _P(self.logits_np) - v * t
    score = (score - _logsumexp_last(score, keepdim=True)).sum(len(score.shape) - 1)
    return _f32(score + log_scale)


ExpRelaxedCategorical.rsample = ExpRelaxedCategorical.sample = _erc_rsample
ExpRelaxedCategorical.log_prob = _erc_log_prob


# MixtureSameFamily
def _msf_log_prob(self, value):
    n_ev = len(self.event_shape)
    v = _P(value)
    v = v.unsqueeze(len(v.shape) - n_ev)  # sample + batch + (1,) + event
    lp = _P(self.component_distribution.log_prob(v.float()))
    log_mix = _P(np.log(np.clip(self.mixture_distribution.probs_np, 1e-300, None)))
    return _f32(_logsumexp_last(lp + log_mix))


def _msf_sample(self, sample_shape=()):
    comps = _P(self.component_distribution.sample(sample_shape))  # sample + batch + (K,) + event
    n_ev = len(self.event_shape)
    axis = len(comps.shape) - n_ev - 1
    idx = _P(self.mixture_distribution.sample(sample_shape)).long()  # sample + batch
    for _ in range(n_ev + 1):
        idx = idx.unsqueeze(len(idx.shape))
    target = list(comps.shape)
    target[axis] = 1
    idx = idx.expand(target).contiguous()
    return _f32(comps.gather(axis, idx).squeeze(axis))


MixtureSameFamily.log_prob = _msf_log_prob
MixtureSameFamily.sample = _msf_sample


# VonMises
def _vm_log_prob(self, value):
    from .special import i0e
    k = _P(self.k_np)
    log_i0 = _P(i0e(k)).log() + k.abs()
    return _f32(k * (_P(value) - _P(self.loc_np)).cos() - math.log(2 * math.pi) - log_i0)


def _vm_sample(self, sample_shape=()):
    from .oxideon import _von_mises
    shape = _shape(self, sample_shape)
    return _f32(_von_mises(_expand(_P(self.loc_np), shape), _expand(_P(self.k_np), shape)))


VonMises.log_prob = _vm_log_prob
VonMises.sample = _vm_sample


# Wishart
def _mvlgamma_t(a: _Tensor, p: int) -> _Tensor:
    out = a * 0.0 + p * (p - 1) / 4 * math.log(math.pi)
    for i in range(p):
        out = out + (a - i / 2).lgamma()
    return out


def _wishart_rsample(self, sample_shape=()):
    # Bartlett decomposition: A lower triangular, chi(df - i) on the diagonal, N(0, 1) below; X = L A A^T L^T
    p = self.event_shape[-1]
    shape = tuple(int(s) for s in sample_shape) + tuple(self.batch_shape)
    df = _expand(_P(self.df_np), shape)
    below = _normal(shape + (p, p)).tril(-1)
    dfs = df.unsqueeze(len(shape)) - _from_numpy(np.arange(p, dtype=np.float64))
    chi = (_gamma(dfs * 0.5, shape + (p,)) * 2.0).sqrt()
    eye = _from_numpy(np.eye(p))
    a = below + chi.unsqueeze(len(shape) + 1) * eye
    la = _P(self.tril_np).matmul(a)
    return _f32(la.matmul(la.transpose(-2, -1)))


def _wishart_log_prob(self, value):
    from . import linalg
    x = _P(value)
    p = self.event_shape[-1]
    nu = _P(self.df_np)
    _, logdet_x = linalg.slogdet(x)
    diag = linalg.diagonal(_P(self.tril_np), 0, -2, -1)
    logdet_s = diag.log().sum(len(diag.shape) - 1) * 2.0
    tr = linalg.diagonal(linalg.solve(_P(self.cov_np), x), 0, -2, -1)
    tr = tr.sum(len(tr.shape) - 1)
    out = (nu - p - 1.0) * 0.5 * logdet_x - tr * 0.5 - nu * (p / 2 * math.log(2)) - nu * 0.5 * logdet_s - _mvlgamma_t(nu * 0.5, p)
    return _f32(out)


Wishart.rsample = Wishart.sample = _wishart_rsample
Wishart.log_prob = _wishart_log_prob


# LKJCholesky
def _lkj_sample(self, sample_shape=()):
    d = self.dim
    shape = tuple(int(s) for s in sample_shape) + tuple(self.batch_shape)
    conc = _expand(_P(self.conc_np), shape)
    marginal = conc + 0.5 * (d - 2)
    offset = _from_numpy(np.concatenate([[0.0], np.arange(d - 1, dtype=np.float64)]))
    c1 = _expand(offset + 0.5, shape + (d,))
    c0 = marginal.unsqueeze(len(shape)) - offset * 0.5
    ga, gb = _gamma(c1, shape + (d,)), _gamma(c0, shape + (d,))
    y = (ga / (ga + gb)).unsqueeze(len(shape) + 1)
    u = _normal(shape + (d, d)).tril(-1)
    norm = (u * u).sum(len(shape) + 1, keepdim=True).sqrt()
    u = _where(norm > 0, u / _where(norm > 0, norm, norm * 0.0 + 1.0), u * 0.0)
    first_row = _from_numpy(np.concatenate([[0.0], np.ones(d - 1)]).reshape(d, 1))
    u = u * first_row  # the first row of the onion is empty
    w = y.sqrt() * u
    diag = (1.0 - (w * w).sum(len(shape) + 1)).clamp_min(float(np.finfo(np.float64).tiny)).sqrt()
    return _f32(w + diag.unsqueeze(len(shape) + 1) * _from_numpy(np.eye(d)))


def _lkj_log_prob(self, value):
    from . import linalg
    d = self.dim
    v = _P(value)
    diag = linalg.diagonal(v, 0, -2, -1)
    diag = diag.narrow(len(diag.shape) - 1, 1, d - 1)
    conc = _P(self.conc_np)
    order = (conc - 1.0).unsqueeze(len(conc.shape)) * 2.0 + float(d) - _from_numpy(np.arange(2, d + 1, dtype=np.float64))
    unnorm = (order * diag.log()).sum(len(diag.shape) - 1)
    dm1 = d - 1
    alpha = conc + 0.5 * dm1
    norm = 0.5 * dm1 * math.log(math.pi) + _mvlgamma_t(alpha - 0.5, dm1) - alpha.lgamma() * float(dm1)
    return _f32(unnorm - norm)


LKJCholesky.sample = _lkj_sample
LKJCholesky.log_prob = _lkj_log_prob


__all__ += [
    "ExponentialFamily", "Chi2", "InverseGamma", "Pareto", "Kumaraswamy", "FisherSnedecor", "GeneralizedPareto",
    "NegativeBinomial", "OneHotCategorical", "OneHotCategoricalStraightThrough", "LogitRelaxedBernoulli",
    "RelaxedBernoulli", "ExpRelaxedCategorical", "RelaxedOneHotCategorical", "LogisticNormal", "MixtureSameFamily",
    "VonMises", "Wishart", "LKJCholesky",
    "Transform", "ExpTransform", "AffineTransform", "SigmoidTransform", "TanhTransform", "SoftplusTransform",
    "AbsTransform", "PowerTransform", "SoftmaxTransform", "StickBreakingTransform", "ComposeTransform",
    "IndependentTransform", "ReshapeTransform", "CatTransform", "StackTransform", "CumulativeDistributionTransform",
    "CorrCholeskyTransform", "LowerCholeskyTransform", "PositiveDefiniteTransform",
]
