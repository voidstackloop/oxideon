"""CTC decoding and OCR error metrics for Oxideon Vision."""

from typing import Any, Dict, List, Sequence, Tuple
import numpy as np

from ...oxideon import Tensor as _Tensor


def _to_np(t: Any) -> np.ndarray:
    if isinstance(t, _Tensor):
        return t.to_numpy()
    return np.asarray(t, dtype=np.float32)


def _check_log_probs(log_probs: Any, blank: int) -> Tuple[np.ndarray, int]:
    lp = np.asarray(_to_np(log_probs), dtype=np.float64)
    if lp.ndim != 2:
        raise ValueError(f"log_probs must be [T, C], got shape {lp.shape}")
    if lp.shape[0] == 0 or lp.shape[1] == 0:
        raise ValueError("log_probs must be non-empty in both dims")
    if not np.all(np.isfinite(lp)):
        raise ValueError("log_probs must be finite")
    blank = int(blank)
    if not 0 <= blank < lp.shape[1]:
        raise ValueError(f"blank must be in [0, C), got {blank}")
    return lp, blank


def ctc_greedy_decode(log_probs: Any, blank: int = 0) -> List[int]:
    """Best-path CTC decoding: argmax per frame, collapse repeats, drop blanks."""
    lp, blank = _check_log_probs(log_probs, blank)
    path = np.argmax(lp, axis=1).tolist()
    out: List[int] = []
    prev: int = -1
    for c in path:
        if c != blank and c != prev:
            out.append(int(c))
        prev = int(c)
    return out


def ctc_beam_decode(
    log_probs: Any, beam_width: int = 10, blank: int = 0
) -> List[Tuple[Tuple[int, ...], float]]:
    """Prefix beam-search CTC decoding (Graves 2006).

    Args:
        log_probs: [T, C] log-probabilities (natural log).
        beam_width: Number of prefixes retained per frame (>= 1).
        blank: Blank label id.

    Returns:
        List of (prefix, log_score) sorted by decreasing score.
    """
    lp, blank = _check_log_probs(log_probs, blank)
    if int(beam_width) < 1:
        raise ValueError("beam_width must be >= 1")
    beam_width = int(beam_width)
    n_inf = float("-inf")
    # prefix -> (log p ending in blank, log p ending in non-blank)
    beams: Dict[Tuple[int, ...], Tuple[float, float]] = {(): (0.0, n_inf)}
    for t in range(lp.shape[0]):
        nxt: Dict[Tuple[int, ...], List[float]] = {}

        def _add(prefix: Tuple[int, ...], idx: int, val: float) -> None:
            if val == n_inf:
                return  # zero-mass paths must not create phantom prefixes
            entry = nxt.get(prefix)
            if entry is None:
                entry = [n_inf, n_inf]
                nxt[prefix] = entry
            entry[idx] = float(np.logaddexp(entry[idx], val))

        for prefix, (pb, pnb) in beams.items():
            tot = float(np.logaddexp(pb, pnb))
            for c in range(lp.shape[1]):
                p = float(lp[t, c])
                if c == blank:
                    _add(prefix, 0, tot + p)
                else:
                    nprefix = prefix + (c,)
                    if prefix and c == prefix[-1]:
                        _add(nprefix, 1, pb + p)  # extend after blank only
                        _add(prefix, 1, pnb + p)  # repeat collapses: stay
                    else:
                        _add(nprefix, 1, tot + p)
        ranked = sorted(
            nxt.items(), key=lambda kv: float(np.logaddexp(kv[1][0], kv[1][1])), reverse=True
        )
        beams = {k: (v[0], v[1]) for k, v in ranked[:beam_width]}
    scored = [
        (prefix, float(np.logaddexp(pb, pnb))) for prefix, (pb, pnb) in beams.items()
    ]
    return sorted(scored, key=lambda kv: kv[1], reverse=True)


def edit_distance(a: Sequence[Any], b: Sequence[Any]) -> int:
    """Levenshtein distance between two sequences (strings or label lists)."""
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[m]


def cer(pred: str, ref: str) -> float:
    """Character error rate = edit_distance(pred, ref) / max(len(ref), 1)."""
    if not isinstance(pred, str) or not isinstance(ref, str):
        raise ValueError("cer expects two strings")
    if not ref:
        return float(edit_distance(pred, ref))
    return edit_distance(pred, ref) / len(ref)


def wer(pred: str, ref: str) -> float:
    """Word error rate = word-level edit distance / max(len(ref_words), 1)."""
    if not isinstance(pred, str) or not isinstance(ref, str):
        raise ValueError("wer expects two strings")
    rw = ref.split()
    if not rw:
        return float(edit_distance(pred.split(), rw))
    return edit_distance(pred.split(), rw) / len(rw)


__all__ = ["ctc_greedy_decode", "ctc_beam_decode", "edit_distance", "cer", "wer"]
