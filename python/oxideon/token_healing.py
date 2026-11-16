"""Token Healing, Nucleus (Top-P) / Top-K Sampling, and Repetition Penalty for Oxideon.

Provides production decoding utilities and tokenizer boundary healing:
  - repetition_penalty: Penalizes already generated tokens (Keskar et al., 2019)
  - top_k_top_p_filtering: Combined Top-K and Nucleus (Top-P) logit truncation (Holtzman et al., 2020)
  - greedy_search_step: Temperature-scaled categorical token step with optional deterministic seed
  - TokenBoundaryHealer: Fixes tokenizer boundary bias where prompts end in token prefixes
"""

from typing import List, Tuple, Optional, Any, Set, Dict
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .functional import softmax


def repetition_penalty(
    logits: Any, past_tokens: List[int], penalty: float = 1.2
) -> _Tensor:
    """Applies multiplicative repetition penalty to logits of previously generated tokens."""
    if float(penalty) <= 0.0:
        raise ValueError(f"penalty must be > 0, got {penalty}")
    arr = logits.to_numpy().copy() if isinstance(logits, _Tensor) else np.asarray(logits, dtype=np.float32).copy()
    if arr.size == 0:
        raise ValueError("logits cannot be empty")
    is_1d = arr.ndim == 1
    if is_1d:
        arr = arr[None, :]

    unique_tokens = set(int(t) for t in past_tokens)
    for t in unique_tokens:
        if 0 <= t < arr.shape[-1]:
            mask_pos = arr[:, t] > 0
            arr[mask_pos, t] /= penalty
            arr[~mask_pos, t] *= penalty

    if is_1d:
        arr = arr[0]
    return _from_numpy(np.ascontiguousarray(arr, dtype=np.float32))


def top_k_top_p_filtering(
    logits: Any, top_k: int = 0, top_p: float = 1.0, filter_val: float = -1e9
) -> _Tensor:
    """Filters logits using Top-K and Nucleus (Top-P) probability mass filtering."""
    if top_k < 0:
        raise ValueError(f"top_k must be >= 0, got {top_k}")
    if not (0.0 < float(top_p) <= 1.0):
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    arr = logits.to_numpy().copy() if isinstance(logits, _Tensor) else np.asarray(logits, dtype=np.float32).copy()
    if arr.size == 0:
        raise ValueError("logits cannot be empty")
    is_1d = arr.ndim == 1
    if is_1d:
        arr = arr[None, :]

    batch_size, vocab_size = arr.shape

    if 0 < top_k < vocab_size:
        for b in range(batch_size):
            kth_val = np.partition(arr[b], vocab_size - top_k)[vocab_size - top_k]
            arr[b, arr[b] < kth_val] = filter_val

    if top_p < 1.0:
        for b in range(batch_size):
            sorted_indices = np.argsort(arr[b])[::-1]
            sorted_logits = arr[b, sorted_indices]

            exp_l = np.exp(sorted_logits - np.max(sorted_logits))
            probs = exp_l / np.sum(exp_l)
            cum_probs = np.cumsum(probs)

            sorted_indices_to_remove = cum_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1]
            sorted_indices_to_remove[0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            arr[b, indices_to_remove] = filter_val

    if is_1d:
        arr = arr[0]
    return _from_numpy(np.ascontiguousarray(arr, dtype=np.float32))


def greedy_search_step(
    logits: Any, temperature: float = 1.0, seed: Optional[int] = None
) -> int:
    """Selects next token index via argmax or temperature sampling."""
    arr = logits.to_numpy() if isinstance(logits, _Tensor) else np.asarray(logits, dtype=np.float32)
    arr = arr.reshape(-1)
    if len(arr) == 0:
        raise ValueError("logits cannot be empty")
    if temperature < 0.0:
        raise ValueError(f"temperature must be non-negative, got {temperature}")

    if temperature <= 1e-4:
        return int(np.argmax(arr))

    scaled = arr / temperature
    exp_l = np.exp(scaled - np.max(scaled))
    probs = exp_l / np.sum(exp_l)
    rng = np.random.default_rng(seed)
    return int(rng.choice(len(probs), p=probs))


class TokenBoundaryHealer:
    """Heals prompt tokenization boundaries by trimming incomplete subword suffixes."""

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        self.vocab = vocab or {}

    def heal_prefix(self, prompt: str) -> Tuple[str, Optional[str]]:
        """If prompt ends with a non-boundary word character, separates into healed prompt and prefix."""
        if not prompt or prompt.endswith((" ", "\n", "\t", ".", ",", "?", "!")):
            return prompt, None

        parts = prompt.rsplit(" ", 1)
        if len(parts) == 2:
            return parts[0] + " ", parts[1]
        return "", prompt

    def matching_vocab_tokens(self, prefix: Optional[str]) -> List[Tuple[str, int]]:
        """Returns vocabulary (token_str, token_id) pairs matching the healed subword prefix."""
        if not prefix or not self.vocab:
            return []
        matches = [(tok, idx) for tok, idx in self.vocab.items() if tok.startswith(prefix)]
        matches.sort(key=lambda pair: (len(pair[0]), pair[1]))
        return matches


__all__ = [
    "repetition_penalty",
    "top_k_top_p_filtering",
    "greedy_search_step",
    "TokenBoundaryHealer",
]
