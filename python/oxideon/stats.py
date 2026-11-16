"""Classical (frequentist) statistics: hypothesis tests, correlation, and adjustments.

Implements:
61. ttest_1samp / ttest_ind / ttest_rel: Student t-tests (pooled and Welch)
62. chisquare / chi2_contingency: Goodness-of-fit and independence tests
63. pearsonr / spearmanr: Correlation coefficients with p-values
64. mannwhitneyu: Rank-sum test (normal approximation with tie correction)
65. kstest: One-sample Kolmogorov-Smirnov test (asymptotic p-value)
66. skew / kurtosis / jarque_bera: Moment statistics and normality test
67. ECDF: Empirical cumulative distribution function
68. p_adjust: Bonferroni/Holm/BH multiple-testing corrections

Conventions: inputs accept Tensor/ndarray/sequences (coerced to float64);
non-finite inputs raise ValueError (documented deviation from scipy's
nan-propagation); degenerate zero-variance cases return scipy-identical
NaN/inf values without warnings. P-values are computed by a self-contained
float64 core (regularized gamma/beta via series and continued fractions).
"""

from typing import Any, Callable, NamedTuple, Optional, Tuple, Union
import math
import operator
import numpy as np

from .oxideon import Tensor as _Tensor


__all__ = [
    "TtestResult",
    "Chi2Result",
    "CorrResult",
    "MannWhitneyResult",
    "KSTestResult",
    "JarqueBeraResult",
    "ttest_1samp",
    "ttest_ind",
    "ttest_rel",
    "chisquare",
    "chi2_contingency",
    "pearsonr",
    "spearmanr",
    "mannwhitneyu",
    "kstest",
    "skew",
    "kurtosis",
    "jarque_bera",
    "ECDF",
    "ecdf",
    "p_adjust",
]


class TtestResult(NamedTuple):
    statistic: float
    pvalue: float
    df: float


class Chi2Result(NamedTuple):
    statistic: float
    pvalue: float
    dof: int
    expected_freq: np.ndarray


class CorrResult(NamedTuple):
    statistic: float
    pvalue: float


class MannWhitneyResult(NamedTuple):
    statistic: float
    pvalue: float


class KSTestResult(NamedTuple):
    statistic: float
    pvalue: float


class JarqueBeraResult(NamedTuple):
    statistic: float
    pvalue: float


# ---------------------------------------------------------------------------
# float64 distribution-function core (series / continued fractions)
# ---------------------------------------------------------------------------


def _gser(a: float, x: float) -> float:
    """Regularized lower gamma P(a, x) by Taylor series (x < a + 1)."""
    if x <= 0.0:
        return 0.0
    gln = math.lgamma(a)
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(500):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * 3e-14:
            break
    return total * math.exp(-x + a * math.log(x) - gln)


def _gcf(a: float, x: float) -> float:
    """Regularized upper gamma Q(a, x) by continued fraction (x >= a + 1)."""
    gln = math.lgamma(a)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        dl = d * c
        h *= dl
        if abs(dl - 1.0) < 3e-14:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def _gammaincc_scalar(a: float, x: float) -> float:
    if x <= 0.0:
        return 1.0
    if x < a + 1.0:
        return max(0.0, min(1.0, 1.0 - _gser(a, x)))
    return max(0.0, min(1.0, _gcf(a, x)))


def _betacf(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        dl = d * c
        h *= dl
        if abs(dl - 1.0) < 3e-14:
            break
    return h


def _betainc_scalar(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    if x < (a + 1.0) / (a + b + 2.0):
        front = math.exp(a * math.log(x) + b * math.log1p(-x) - lbeta)
        return max(0.0, min(1.0, front * _betacf(a, b, x) / a))
    front = math.exp(b * math.log1p(-x) + a * math.log(x) - lbeta)
    return max(0.0, min(1.0, 1.0 - front * _betacf(b, a, 1.0 - x) / b))


def _t_sf(t: float, df: float) -> float:
    """Upper tail P(T > t) of Student's t (t >= 0)."""
    x = df / (df + t * t)
    return 0.5 * _betainc_scalar(0.5 * df, 0.5, x)


def _t_cdf(t: float, df: float) -> float:
    sf = _t_sf(abs(t), df)
    return 1.0 - sf if t >= 0.0 else sf


def _chi2_sf(x: float, k: float) -> float:
    if x <= 0.0:
        return 1.0
    return _gammaincc_scalar(0.5 * k, 0.5 * x)


def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _norm_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _kolmogorov_sf(d: float, n: int) -> float:
    """Asymptotic P(D_n > d): alternating series in lam = sqrt(n) * d."""
    if d <= 0.0:
        return 1.0
    lam = math.sqrt(n) * d
    if lam < 0.05:
        return 1.0
    total = 0.0
    for j in range(1, 200):
        term = math.exp(-2.0 * j * j * lam * lam)
        total += term if j % 2 == 1 else -term
        if term < 1e-16:
            break
    return max(0.0, min(1.0, 2.0 * total))


# ---------------------------------------------------------------------------
# input handling
# ---------------------------------------------------------------------------


def _as_1d(a: Any, name: str, minimum: int = 1) -> np.ndarray:
    arr = a.to_numpy() if isinstance(a, _Tensor) else np.asarray(a, dtype=np.float64)
    arr = np.asarray(arr, dtype=np.float64).ravel()
    if arr.shape[0] < minimum:
        raise ValueError(f"{name} needs at least {minimum} observations, got {arr.shape[0]}")
    if bool(np.any(~np.isfinite(arr))):
        raise ValueError(f"{name} must be finite (NaN/inf not supported)")
    return arr


def _check_alternative(alternative: str, name: str) -> str:
    if alternative not in ("two-sided", "less", "greater"):
        raise ValueError(f"{name}: alternative must be 'two-sided', 'less' or 'greater', got {alternative!r}")
    return alternative


def _tail_p(cdf: float, alternative: str) -> float:
    if alternative == "less":
        return cdf
    if alternative == "greater":
        return 1.0 - cdf
    return max(0.0, min(1.0, 2.0 * min(cdf, 1.0 - cdf)))


def _t_from_stats(diff: float, se: float, df: float, alternative: str) -> TtestResult:
    if se == 0.0:
        if diff == 0.0:
            return TtestResult(float("nan"), float("nan"), df)
        t = math.copysign(float("inf"), diff)
        return TtestResult(t, 0.0 if alternative == "two-sided" else (0.0 if (diff > 0) == (alternative == "greater") else 1.0), df)
    t = diff / se
    return TtestResult(t, _tail_p(_t_cdf(t, df), alternative), df)


def ttest_1samp(a: Any, popmean: float = 0.0, alternative: str = "two-sided") -> TtestResult:
    """One-sample t-test of mean(a) against `popmean`."""
    _check_alternative(alternative, "ttest_1samp")
    x = _as_1d(a, "ttest_1samp", minimum=2)
    try:
        mu = float(popmean)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ttest_1samp: popmean must be a real scalar, got {popmean!r}") from exc
    if not math.isfinite(mu):
        raise ValueError(f"ttest_1samp: popmean must be finite, got {mu}")
    n = x.shape[0]
    return _t_from_stats(float(np.mean(x)) - mu, float(np.std(x, ddof=1) / math.sqrt(n)), float(n - 1), alternative)


def ttest_ind(a: Any, b: Any, equal_var: bool = True, alternative: str = "two-sided") -> TtestResult:
    """Two-sample t-test (pooled variance, or Welch's test with `equal_var=False`)."""
    _check_alternative(alternative, "ttest_ind")
    x = _as_1d(a, "ttest_ind", minimum=2)
    y = _as_1d(b, "ttest_ind", minimum=2)
    nx, ny = x.shape[0], y.shape[0]
    mx, my = float(np.mean(x)), float(np.mean(y))
    vx, vy = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    if equal_var:
        df = float(nx + ny - 2)
        pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
        se = math.sqrt(pooled * (1.0 / nx + 1.0 / ny))
        return _t_from_stats(mx - my, se, df, alternative)
    wn = vx / nx + vy / ny
    se = math.sqrt(wn)
    denom = (vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1)
    df = wn * wn / denom if denom > 0.0 else float("nan")
    return _t_from_stats(mx - my, se, df, alternative)


def ttest_rel(a: Any, b: Any, alternative: str = "two-sided") -> TtestResult:
    """Paired t-test on the differences a - b."""
    _check_alternative(alternative, "ttest_rel")
    x = _as_1d(a, "ttest_rel", minimum=2)
    y = _as_1d(b, "ttest_rel", minimum=2)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"ttest_rel: paired samples must match, got {x.shape[0]} and {y.shape[0]}")
    return ttest_1samp(x - y, 0.0, alternative=alternative)


def chisquare(f_obs: Any, f_exp: Optional[Any] = None, ddof: int = 0) -> Chi2Result:
    """Chi-square goodness-of-fit test."""
    if isinstance(ddof, bool):
        raise ValueError(f"chisquare: ddof must be a non-negative integer, got {ddof!r}")
    try:
        ddof = operator.index(ddof)
    except TypeError as exc:
        raise TypeError(f"chisquare: ddof must be an integer, got {ddof!r}") from exc
    if ddof < 0:
        raise ValueError(f"chisquare: ddof must be non-negative, got {ddof}")
    obs = _as_1d(f_obs, "chisquare", minimum=2)
    if bool(np.any(obs < 0.0)):
        raise ValueError("chisquare: observed frequencies must be non-negative")
    if f_exp is None:
        exp = np.full_like(obs, float(np.mean(obs)))
    else:
        exp = np.asarray(f_exp, dtype=np.float64).ravel()
        if exp.shape != obs.shape:
            raise ValueError(f"chisquare: f_exp shape {exp.shape} must match f_obs {obs.shape}")
        if bool(np.any(~np.isfinite(exp))) or bool(np.any(exp <= 0.0)):
            raise ValueError("chisquare: expected frequencies must be finite and positive")
    if obs.shape[0] - 1 - ddof < 1:
        raise ValueError(f"chisquare: need at least 1 degree of freedom, got {obs.shape[0] - 1 - ddof}")
    dof = obs.shape[0] - 1 - ddof
    if f_exp is None and float(np.mean(obs)) == 0.0:
        return Chi2Result(float("nan"), float("nan"), dof, exp)
    stat = float(np.sum((obs - exp) ** 2 / exp))
    return Chi2Result(stat, _chi2_sf(stat, float(dof)), dof, exp)


def chi2_contingency(observed: Any, correction: bool = True) -> Chi2Result:
    """Chi-square test of independence for a 2-D contingency table.

    Yates' continuity correction applies to 2x2 tables when `correction`
    is True (scipy convention).
    """
    tab = observed.to_numpy() if isinstance(observed, _Tensor) else np.asarray(observed, dtype=np.float64)
    tab = np.asarray(tab, dtype=np.float64)
    if tab.ndim != 2 or tab.shape[0] < 2 or tab.shape[1] < 2:
        raise ValueError(f"chi2_contingency: need a 2-D table with shape >= (2, 2), got {tab.shape}")
    if bool(np.any(~np.isfinite(tab))) or bool(np.any(tab < 0.0)):
        raise ValueError("chi2_contingency: table entries must be finite and non-negative")
    total = float(np.sum(tab))
    if total <= 0.0:
        raise ValueError("chi2_contingency: table total must be positive")
    exp = np.outer(np.sum(tab, axis=1), np.sum(tab, axis=0)) / total
    if bool(np.any(exp == 0.0)):
        raise ValueError("chi2_contingency: an expected frequency is zero; test is undefined")
    dof = (tab.shape[0] - 1) * (tab.shape[1] - 1)
    if correction and tab.shape == (2, 2):
        clipped = np.maximum(np.abs(tab - exp) - 0.5, 0.0)
        stat = float(np.sum(clipped ** 2 / exp))
    else:
        stat = float(np.sum((tab - exp) ** 2 / exp))
    return Chi2Result(stat, _chi2_sf(stat, float(dof)), dof, exp)


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranked = np.empty(a.shape[0], dtype=np.float64)
    diff = np.diff(a[order])
    bounds = np.concatenate(([0], np.nonzero(diff != 0.0)[0] + 1, [a.shape[0]]))
    for s, e in zip(bounds[:-1], bounds[1:]):
        ranked[order[s:e]] = (s + 1.0 + e) / 2.0
    return ranked


def _corr_pvalue(r: float, n: int) -> float:
    if not math.isfinite(r):
        return float("nan")
    if abs(r) >= 1.0:
        return 0.0
    df = n - 2
    t = r * math.sqrt(df / max(1e-300, 1.0 - r * r))
    return _tail_p(_t_cdf(t, float(df)), "two-sided")


def pearsonr(x: Any, y: Any) -> CorrResult:
    """Pearson correlation coefficient with two-sided p-value (t distribution; NaN for n < 3)."""
    a = _as_1d(x, "pearsonr", minimum=2)
    b = _as_1d(y, "pearsonr", minimum=2)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"pearsonr: inputs must match, got {a.shape[0]} and {b.shape[0]}")
    n = a.shape[0]
    am, bm = a - float(np.mean(a)), b - float(np.mean(b))
    denom = math.sqrt(float(am @ am) * float(bm @ bm))
    if denom == 0.0:
        return CorrResult(float("nan"), float("nan"))
    r = float((am @ bm) / denom)
    r = max(-1.0, min(1.0, r))
    if n < 3:
        return CorrResult(r, float("nan"))
    return CorrResult(r, _corr_pvalue(r, n))


def spearmanr(x: Any, y: Any) -> CorrResult:
    """Spearman rank correlation with two-sided p-value (t distribution)."""
    a = _as_1d(x, "spearmanr", minimum=2)
    b = _as_1d(y, "spearmanr", minimum=2)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"spearmanr: inputs must match, got {a.shape[0]} and {b.shape[0]}")
    return pearsonr(_rankdata(a), _rankdata(b))


def mannwhitneyu(x: Any, y: Any, continuity: bool = True, alternative: str = "two-sided") -> MannWhitneyResult:
    """Mann-Whitney U test (normal approximation with tie correction).

    Returns U for the first sample, like scipy.
    """
    _check_alternative(alternative, "mannwhitneyu")
    a = _as_1d(x, "mannwhitneyu", minimum=1)
    b = _as_1d(y, "mannwhitneyu", minimum=1)
    n1, n2 = a.shape[0], b.shape[0]
    ranks = _rankdata(np.concatenate([a, b]))
    r1 = float(np.sum(ranks[:n1]))
    u1 = r1 - n1 * (n1 + 1.0) / 2.0
    mean = n1 * n2 / 2.0
    _, counts = np.unique(np.concatenate([a, b]), return_counts=True)
    tie_sum = float(np.sum(counts.astype(np.float64) ** 3 - counts.astype(np.float64)))
    n = n1 + n2
    var = n1 * n2 / 12.0 * ((n + 1.0) - tie_sum / (n * (n - 1.0))) if n > 1 else 0.0
    if var <= 0.0:
        return MannWhitneyResult(u1, 1.0)
    sd = math.sqrt(var)
    delta = abs(u1 - mean)
    if continuity:
        delta = max(0.0, delta - 0.5)
    z = delta / sd
    p_two = max(0.0, min(1.0, 2.0 * _norm_sf(z)))
    if alternative == "two-sided":
        return MannWhitneyResult(u1, p_two)
    # One-sided: like scipy, the continuity shift always opposes the
    # alternative direction, regardless of which side U falls on.
    shift = 0.5 if continuity else 0.0
    if alternative == "greater":
        p = _norm_sf((u1 - mean - shift) / sd)
        return MannWhitneyResult(u1, max(0.0, min(1.0, p)))
    p = _norm_cdf((u1 - mean + shift) / sd)
    return MannWhitneyResult(u1, max(0.0, min(1.0, p)))


def kstest(x: Any, cdf: Union[str, Callable[[np.ndarray], np.ndarray]] = "norm",
           args: Tuple[float, ...] = ()) -> KSTestResult:
    """One-sample Kolmogorov-Smirnov test (asymptotic p-value).

    `cdf='norm'` tests against Normal(args) (default standard normal); a
    callable receives the sorted sample and returns CDF values.
    """
    a = _as_1d(x, "kstest", minimum=1)
    if isinstance(cdf, str):
        if cdf != "norm":
            raise ValueError(f"kstest: unknown cdf {cdf!r} (expected 'norm' or a callable)")
        loc = float(args[0]) if len(args) > 0 else 0.0
        scale = float(args[1]) if len(args) > 1 else 1.0
        if not math.isfinite(loc) or not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"kstest: need finite loc and positive scale, got {args!r}")
        ordered = np.sort(a)
        expected = np.array([_norm_cdf((v - loc) / scale) for v in ordered])
    else:
        ordered = np.sort(a)
        try:
            expected = np.asarray(cdf(ordered), dtype=np.float64).ravel()
        except Exception as exc:
            raise ValueError(f"kstest: cdf callable failed ({exc})") from exc
        if expected.shape[0] != ordered.shape[0]:
            raise ValueError("kstest: cdf callable must return one value per observation")
        if bool(np.any(~np.isfinite(expected))):
            raise ValueError("kstest: cdf callable returned non-finite values")
    n = ordered.shape[0]
    empirical = np.arange(1, n + 1, dtype=np.float64) / n
    d_plus = float(np.max(empirical - expected))
    d_minus = float(np.max(expected - empirical + 1.0 / n))
    d = max(d_plus, d_minus)
    return KSTestResult(d, _kolmogorov_sf(d, n))


def skew(x: Any, bias: bool = True) -> float:
    """Sample skewness (bias=True matches the scipy default)."""
    a = _as_1d(x, "skew", minimum=3)
    n = a.shape[0]
    m2 = float(np.mean((a - float(np.mean(a))) ** 2))
    if m2 == 0.0:
        return float("nan")
    m3 = float(np.mean((a - float(np.mean(a))) ** 3))
    s = m3 / m2 ** 1.5
    if not bias:
        s *= math.sqrt(n * (n - 1.0)) / (n - 2.0)
    return s


def kurtosis(x: Any, fisher: bool = True, bias: bool = True) -> float:
    """Sample kurtosis (Fisher, bias=True matches the scipy defaults)."""
    a = _as_1d(x, "kurtosis", minimum=4)
    n = a.shape[0]
    m2 = float(np.mean((a - float(np.mean(a))) ** 2))
    if m2 == 0.0:
        return float("nan")
    m4 = float(np.mean((a - float(np.mean(a))) ** 4))
    k = m4 / m2 ** 2 - 3.0
    if not bias:
        k = ((n - 1.0) / ((n - 2.0) * (n - 3.0))) * ((n + 1.0) * k + 6.0)
    return k if fisher else k + 3.0


def jarque_bera(x: Any) -> JarqueBeraResult:
    """Jarque-Bera normality test (chi-square with 2 df; needs n >= 4)."""
    a = _as_1d(x, "jarque_bera", minimum=4)
    n = a.shape[0]
    s = skew(a)
    k = kurtosis(a)
    if not math.isfinite(s) or not math.isfinite(k):
        return JarqueBeraResult(float("nan"), float("nan"))
    jb = n / 6.0 * (s * s + k * k / 4.0)
    return JarqueBeraResult(jb, _chi2_sf(jb, 2.0))


class ECDF:
    """Empirical CDF: step function through sorted observations."""

    def __init__(self, x: Any):
        a = _as_1d(x, "ECDF", minimum=1)
        self.quantiles = np.sort(a)
        self.probabilities = np.arange(1, a.shape[0] + 1, dtype=np.float64) / a.shape[0]

    def __call__(self, q: Any) -> np.ndarray:
        qq = np.asarray(q, dtype=np.float64)
        return np.searchsorted(self.quantiles, qq, side="right") / self.quantiles.shape[0]


def ecdf(x: Any) -> ECDF:
    """Builds the empirical CDF of the observations `x`."""
    return ECDF(x)


def p_adjust(pvals: Any, method: str = "holm", alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Multiple-testing correction ('bonferroni', 'holm', 'bh', 'none').

    Returns `(rejected, adjusted)` with the BH method under both 'bh' and
    'fdr_bh' spellings.
    """
    methods = {"bonferroni": "bonferroni", "holm": "holm", "bh": "bh", "fdr_bh": "bh", "none": "none"}
    if method not in methods:
        raise ValueError(f"p_adjust: unknown method {method!r} (expected one of {sorted(methods)})")
    p = np.asarray(pvals, dtype=np.float64).ravel()
    if p.shape[0] < 1:
        raise ValueError("p_adjust: need at least one p-value")
    if bool(np.any(~np.isfinite(p))) or bool(np.any(p < 0.0)) or bool(np.any(p > 1.0)):
        raise ValueError("p_adjust: p-values must be finite and within [0, 1]")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError(f"p_adjust: alpha must lie in (0, 1), got {alpha!r}")
    m = p.shape[0]
    kind = methods[method]
    if kind == "none":
        adj = p.copy()
    elif kind == "bonferroni":
        adj = np.minimum(p * m, 1.0)
    else:
        order = np.argsort(p, kind="mergesort")
        ranked = p[order]
        if kind == "holm":
            adj_sorted = ranked * (m - np.arange(m))
            adj_sorted = np.maximum.accumulate(adj_sorted)
        else:  # Benjamini-Hochberg step-up
            adj_sorted = ranked * m / np.arange(1, m + 1)
            adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
        adj = np.empty(m)
        adj[order] = np.minimum(adj_sorted, 1.0)
    return adj <= alpha, adj

