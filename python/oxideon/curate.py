"""Pretraining data curation: dedup, decontamination, filters, domain mixing.

- ``MinHashLSH``: near-duplicate detection over character shingles with
  banded LSH indexing and union-find clustering.
- ``exact_jaccard``: character n-gram Jaccard similarity between two texts.
- ``NGramDecontaminator``: exact word n-gram overlap against eval sets.
- ``gopher_flags`` / ``passes_gopher``: Gopher-style heuristic quality rules.
- ``c4_flags`` / ``passes_c4``: C4-style terminal/minimal-content rules.
- ``doremi_weights``: exponentiated-gradient domain weight update on excess loss.
- ``regmix_fit`` / ``regmix_search``: ridge mixture-to-loss fit plus Dirichlet
  search for the predicted-best mixture (RegMix-lite).
- ``temperature_mix``: size-proportional sampling with temperature flattening.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import math
import re
import zlib

import numpy as np

_WORD = re.compile(r"[A-Za-z0-9']+")
_MERSENNE_61 = (1 << 61) - 1


def _shingles(text: str, n: int) -> List[str]:
    cleaned = " ".join(str(text).split())
    if len(cleaned) < n:
        return [cleaned] if cleaned else []
    return [cleaned[i : i + n] for i in range(len(cleaned) - n + 1)]


def exact_jaccard(a: str, b: str, ngram: int = 5) -> float:
    """Character n-gram Jaccard similarity between two texts."""
    if ngram < 1:
        raise ValueError("ngram must be >= 1")
    set_a = set(_shingles(a, ngram))
    set_b = set(_shingles(b, ngram))
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


class MinHashLSH:
    """MinHash signatures with banded LSH for near-duplicate discovery."""

    def __init__(self, num_perm: int = 128, bands: int = 16, ngram: int = 5, seed: int = 0):
        if num_perm < 1 or bands < 1 or ngram < 1:
            raise ValueError("num_perm, bands and ngram must be >= 1")
        if num_perm % bands:
            raise ValueError(f"num_perm={num_perm} must split evenly over bands={bands}")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self.num_perm = num_perm
        self.bands = bands
        self.rows = num_perm // bands
        self.ngram = ngram
        rng = np.random.default_rng(seed)
        self._a = (rng.integers(1, _MERSENNE_61, size=num_perm).astype(object) | 1)
        self._b = rng.integers(0, _MERSENNE_61, size=num_perm).astype(object)
        self._sigs: Dict[str, List[int]] = {}
        self._buckets: Dict[Any, List[str]] = {}

    def signature(self, text: str) -> List[int]:
        """MinHash signature (length ``num_perm``) of a text."""
        sig: Any = np.full(self.num_perm, _MERSENNE_61, dtype=object)
        for shingle in _shingles(text, self.ngram):
            code = zlib.crc32(shingle.encode("utf-8"))
            vals = (self._a * code + self._b) % _MERSENNE_61
            sig = np.minimum(sig, vals)
        return [int(v) for v in sig]

    def add(self, doc_id: str, text: str) -> None:
        """Indexes a document; duplicate ids are refused."""
        doc_id = str(doc_id)
        if doc_id in self._sigs:
            raise ValueError(f"duplicate doc_id '{doc_id}'")
        sig = self.signature(text)
        self._sigs[doc_id] = sig
        for band in range(self.bands):
            key = (band, tuple(sig[band * self.rows : (band + 1) * self.rows]))
            self._buckets.setdefault(key, []).append(doc_id)

    def query(self, text: str) -> List[str]:
        """Ids sharing at least one LSH band with the text (sorted)."""
        sig = self.signature(text)
        found = set()
        for band in range(self.bands):
            key = (band, tuple(sig[band * self.rows : (band + 1) * self.rows]))
            found.update(self._buckets.get(key, ()))
        return sorted(found)

    def duplicates(self) -> List[List[str]]:
        """Union-find clusters (size >= 2) over shared LSH bands."""
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for doc_id in self._sigs:
            parent[doc_id] = doc_id
        for members in self._buckets.values():
            for other in members[1:]:
                parent[find(members[0])] = find(other)
        clusters: Dict[str, List[str]] = {}
        for doc_id in self._sigs:
            clusters.setdefault(find(doc_id), []).append(doc_id)
        return sorted(
            [sorted(members) for members in clusters.values() if len(members) > 1]
        )


class NGramDecontaminator:
    """Exact word n-gram overlap check against an eval reference set."""

    def __init__(self, n: int = 13):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self._grams: set = set()

    @staticmethod
    def _grams_of(text: str, n: int) -> List[Tuple[str, ...]]:
        toks = _WORD.findall(str(text).lower())
        return [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]

    def fit(self, texts: Sequence[str]) -> "NGramDecontaminator":
        """Builds the reference n-gram set (refit replaces it)."""
        grams: set = set()
        for text in texts:
            grams.update(self._grams_of(text, self.n))
        if not grams:
            raise ValueError("fit needs texts holding at least one n-gram")
        self._grams = grams
        return self

    def is_contaminated(self, text: str) -> bool:
        """True when any eval n-gram appears in the text."""
        return any(g in self._grams for g in self._grams_of(text, self.n))

    def contamination_rate(self, texts: Sequence[str]) -> float:
        """Fraction of texts sharing at least one eval n-gram."""
        texts = list(texts)
        if not texts:
            raise ValueError("contamination_rate needs at least one text")
        return sum(1.0 for t in texts if self.is_contaminated(t)) / len(texts)


def _word_stats(text: str) -> Tuple[List[str], List[str]]:
    words = _WORD.findall(str(text))
    lines = str(text).splitlines() or [""]
    return words, lines


def gopher_flags(text: str) -> Dict[str, bool]:
    """Gopher-style (Rae et al., 2021) heuristic quality flags, all True = pass."""
    words, lines = _word_stats(text)
    n = len(words)
    mean_len = sum(len(w) for w in words) / n if n else 0.0
    hashes = str(text).count("#")
    ellipses = str(text).count("...")
    bullets = sum(1 for ln in lines if ln.strip().startswith(("- ", "* ", "• ")))
    ellipsis_lines = sum(1 for ln in lines if ln.strip().endswith("..."))
    uniq_lines = len({ln.strip() for ln in lines if ln.strip()})
    total_lines = sum(1 for ln in lines if ln.strip())
    dup_line_frac = 1.0 - uniq_lines / total_lines if total_lines else 0.0
    grams = [" ".join(words[i : i + 5]) for i in range(max(n - 4, 0))]
    dup_gram_frac = 1.0 - len(set(grams)) / len(grams) if len(grams) > 1 else 0.0
    return {
        "word_count_ok": 50 <= n <= 100000,
        "mean_word_length_ok": 3.0 <= mean_len <= 10.0,
        "hash_ratio_ok": hashes / max(n, 1) <= 0.1,
        "ellipsis_ratio_ok": ellipses / max(n, 1) <= 0.1,
        "bullet_ok": bullets / max(total_lines, 1) <= 0.9,
        "ellipsis_line_ok": ellipsis_lines / max(total_lines, 1) <= 0.3,
        "dup_line_ok": dup_line_frac <= 0.2,
        "dup_gram_ok": dup_gram_frac <= 0.2,
    }


def passes_gopher(text: str) -> bool:
    """True when every Gopher heuristic flag passes."""
    return all(gopher_flags(text).values())


_C4_BAD_WORDS = frozenset(
    ["porn", "xxx", "sex", "nude", "erotic", "fetish", "hentai", "boob", "penis", "vagina"]
)


def c4_flags(text: str) -> Dict[str, bool]:
    """C4-style (Raffel et al., 2020) minimal-content flags, all True = pass."""
    words, _ = _word_stats(text)
    lowered = str(text).lower()
    lowered_words = [w.lower() for w in words]
    return {
        "terminal_punct_ok": str(text).rstrip().endswith((".", "?", "!", '"', "'")),
        "min_words_ok": len(words) >= 5,
        "no_lorem_ok": "lorem ipsum" not in lowered,
        "no_code_braces_ok": "{" not in str(text) and "}" not in str(text),
        "no_badwords_ok": not any(w in _C4_BAD_WORDS for w in lowered_words),
    }


def passes_c4(text: str) -> bool:
    """True when every C4-style flag passes."""
    return all(c4_flags(text).values())


def doremi_weights(
    excess_losses: Sequence[float],
    prev_weights: Optional[Sequence[float]] = None,
    step_size: float = 0.1,
) -> List[float]:
    """DoReMi exponentiated-gradient update: w' ∝ w * exp(step * excess)."""
    excess = np.asarray(list(excess_losses), dtype=np.float64)
    if excess.ndim != 1 or excess.shape[0] < 1:
        raise ValueError("excess_losses must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(excess)):
        raise ValueError("excess_losses must be finite")
    if prev_weights is None:
        weights = np.ones_like(excess) / excess.shape[0]
    else:
        weights = np.asarray(list(prev_weights), dtype=np.float64)
        if weights.shape != excess.shape:
            raise ValueError("prev_weights must match excess_losses in length")
        if (weights < 0).any() or weights.sum() <= 0 or not np.all(np.isfinite(weights)):
            raise ValueError("prev_weights must be finite non-negative with positive sum")
        weights = weights / weights.sum()
    if step_size < 0:
        raise ValueError("step_size must be non-negative")
    updated = weights * np.exp(step_size * np.clip(excess, -30.0, 30.0))
    return (updated / updated.sum()).tolist()


def regmix_fit(
    mixtures: Sequence[Sequence[float]], losses: Sequence[float], l2: float = 1.0
) -> Tuple[List[float], float]:
    """Ridge fit of mixture -> loss; returns (coefficients, intercept) without penalizing intercept."""
    mat = np.asarray([list(m) for m in mixtures], dtype=np.float64)
    tgt = np.asarray(list(losses), dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] < 1 or mat.shape[1] < 1:
        raise ValueError("mixtures must be a non-empty 2-D array")
    if tgt.shape != (mat.shape[0],):
        raise ValueError("losses must hold one entry per mixture")
    if not np.all(np.isfinite(mat)) or not np.all(np.isfinite(tgt)):
        raise ValueError("mixtures and losses must be finite")
    if not np.allclose(mat.sum(axis=1), 1.0, atol=1e-6) or (mat < 0).any():
        raise ValueError("each mixture must be a probability vector")
    if l2 < 0:
        raise ValueError("l2 must be non-negative")
    design = np.concatenate([mat, np.ones((mat.shape[0], 1))], axis=1)
    reg = l2 * np.eye(design.shape[1])
    reg[-1, -1] = 1e-12 if l2 == 0 else 0.0
    gram = design.T @ design + reg
    try:
        sol = np.linalg.solve(gram, design.T @ tgt)
    except np.linalg.LinAlgError:
        # Fallback tiny ridge on intercept if collinear
        gram[-1, -1] += max(l2, 1e-8)
        try:
            sol = np.linalg.solve(gram, design.T @ tgt)
        except np.linalg.LinAlgError:
            raise ValueError("mixture design is singular; increase l2")
    return sol[:-1].tolist(), float(sol[-1])


def regmix_search(
    coef: Sequence[float],
    intercept: float,
    n_candidates: int = 1000,
    seed: int = 0,
    alpha: float = 1.0,
) -> Tuple[List[float], float]:
    """Dirichlet search for the mixture minimizing the fitted loss model."""
    coef_arr = np.asarray(list(coef), dtype=np.float64)
    if coef_arr.ndim != 1 or coef_arr.shape[0] < 1 or not np.all(np.isfinite(coef_arr)):
        raise ValueError("coef must be a non-empty finite vector")
    if not math.isfinite(intercept):
        raise ValueError("intercept must be finite")
    if n_candidates < 1:
        raise ValueError("n_candidates must be >= 1")
    if seed < 0 or alpha <= 0:
        raise ValueError("seed must be >= 0 and alpha positive")
    rng = np.random.default_rng(seed)
    cands = rng.dirichlet([alpha] * coef_arr.shape[0], size=n_candidates)
    preds = cands @ coef_arr + intercept
    best = int(np.argmin(preds))
    return cands[best].tolist(), float(preds[best])


def temperature_mix(sizes: Sequence[float], temperature: float) -> List[float]:
    """Size-proportional mixture flattened by temperature (mT5 style)."""
    arr = np.asarray(list(sizes), dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] < 1:
        raise ValueError("sizes must be a non-empty 1-D sequence")
    if (arr < 0).any() or arr.sum() <= 0 or not np.all(np.isfinite(arr)):
        raise ValueError("sizes must be finite non-negative with positive sum")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    with np.errstate(divide="ignore"):
        log_arr = np.where(arr > 0, np.log(arr) / temperature, -np.inf)
    max_log = np.max(log_arr)
    exp_arr = np.where(arr > 0, np.exp(log_arr - max_log), 0.0)
    s = exp_arr.sum()
    if s <= 0:
        return (np.ones_like(arr) / len(arr)).tolist()
    return (exp_arr / s).tolist()


__all__ = [
    "MinHashLSH",
    "exact_jaccard",
    "NGramDecontaminator",
    "gopher_flags",
    "passes_gopher",
    "c4_flags",
    "passes_c4",
    "doremi_weights",
    "regmix_fit",
    "regmix_search",
    "temperature_mix",
]
