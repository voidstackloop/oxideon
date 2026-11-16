"""Complete Probability & Distributions Library for Oxideon.

Provides PyTorch-compatible `torch.distributions` implementations supporting
variational inference, policy gradients, generative modeling, and closed-form KL divergences:
- `Distribution` (base)
- `Normal`, `Bernoulli`, `Categorical`, `Uniform`, `Exponential`
- `Laplace`, `Cauchy`, `Beta`, `Gamma`, `Dirichlet`, `Poisson`
- `Geometric`, `Gumbel`, `LogNormal`, `StudentT`
- `MultivariateNormal`, `LowRankMultivariateNormal`
- `Independent`, `TransformedDistribution`
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
    return _from_numpy(np.ascontiguousarray(arr, dtype=np.float32))


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
        eps = np.random.randn(*shape).astype(np.float32)
        if (isinstance(self.loc, _Tensor) and self.loc.requires_grad) or (isinstance(self.scale, _Tensor) and self.scale.requires_grad):
            eps_t = _wrap(eps)
            return self.loc + eps_t * self.scale
        res = self.loc_np + eps * self.scale_np
        return _wrap(res)

    def sample(self, sample_shape: Sequence[int] = ()) -> _Tensor:
        shape = tuple(sample_shape) + self.batch_shape
        eps = np.random.randn(*shape).astype(np.float32)
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
        eps = np.random.randn(*shape).astype(np.float32)
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
        z = np.random.randn(*shape).astype(np.float32)
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
        eps = np.abs(np.random.randn(*shape)).astype(np.float32)
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
