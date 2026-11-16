"""Speculative Decoding for accelerated autoregressive sequence generation.

Implements the draft-and-verify algorithm (Leviathan et al., 2023; Chen et al., 2023):
Uses a small, fast draft model to speculate K candidate tokens, followed by a single
parallel verification forward pass of the larger target model with exact distribution recovery.
"""

from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Any
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy


@dataclass
class SpeculativeStats:
    """Runtime statistics tracking speculative decoding efficiency."""
    total_generated: int = 0
    draft_tokens_proposed: int = 0
    draft_tokens_accepted: int = 0
    target_forward_passes: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.draft_tokens_proposed == 0:
            return 0.0
        return float(self.draft_tokens_accepted) / float(self.draft_tokens_proposed)

    @property
    def speedup(self) -> float:
        if self.target_forward_passes == 0:
            return 1.0
        return float(self.total_generated) / float(self.target_forward_passes)


class SpeculativeDraftDecoder:
    """Speculative decoding engine that coordinates draft and target models."""

    def __init__(
        self,
        draft_model_fn: Callable[[List[int]], np.ndarray],
        target_model_fn: Callable[[List[int]], np.ndarray],
        gamma: int = 4,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            draft_model_fn: Callable accepting sequence of token IDs and returning
                            logits [vocab_size] or [seq_len, vocab_size].
            target_model_fn: Callable accepting sequence of token IDs and returning
                             logits for next positions.
            gamma: Number of speculative lookahead tokens generated per round.
            temperature: Sampling temperature (1.0 for standard sampling, >0).
            seed: Optional RNG seed for reproducible speculative sampling.
        """
        self.draft_fn = draft_model_fn
        self.target_fn = target_model_fn
        self.gamma = gamma
        self.temperature = max(1e-5, temperature)
        self.stats = SpeculativeStats()
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def _extract_last_logits(logits: Any) -> np.ndarray:
        if hasattr(logits, "to_numpy"):
            arr = logits.to_numpy()
        else:
            arr = np.asarray(logits)
        if arr.ndim >= 2:
            return arr.reshape(-1, arr.shape[-1])[-1]
        return arr.flatten()

    @staticmethod
    def _softmax(logits: np.ndarray, temp: float = 1.0) -> np.ndarray:
        scaled = logits / temp
        shifted = scaled - np.max(scaled)
        exp_s = np.exp(shifted)
        return exp_s / (np.sum(exp_s) + 1e-12)

    def _sample(self, probs: np.ndarray) -> int:
        probs = np.maximum(probs, 0.0)
        s = np.sum(probs)
        if s <= 1e-12:
            return int(np.argmax(probs))
        norm_p = probs / s
        return int(self._rng.choice(len(norm_p), p=norm_p))

    def step(self, prompt: List[int]) -> List[int]:
        """Performs one speculative decoding step.

        Returns:
            List of accepted/sampled tokens in this step (at least 1, up to gamma + 1).
        """
        cur_tokens = list(prompt)
        draft_tokens: List[int] = []
        draft_probs: List[np.ndarray] = []

        # 1. Autoregressively draft gamma tokens
        for _ in range(self.gamma):
            d_logits = self._extract_last_logits(self.draft_fn(cur_tokens + draft_tokens))
            d_p = self._softmax(d_logits, self.temperature)
            next_d = self._sample(d_p)
            draft_tokens.append(next_d)
            draft_probs.append(d_p)

        self.stats.draft_tokens_proposed += len(draft_tokens)

        # 2. Parallel target evaluation
        accepted_tokens: List[int] = []
        self.stats.target_forward_passes += 1

        all_accepted = True
        for i, d_tok in enumerate(draft_tokens):
            seq = cur_tokens + accepted_tokens
            t_logits = self._extract_last_logits(self.target_fn(seq))
            t_p = self._softmax(t_logits, self.temperature)

            q_prob = draft_probs[i][d_tok]
            p_prob = t_p[d_tok]

            ratio = p_prob / (q_prob + 1e-12)
            u = float(self._rng.uniform(0.0, 1.0))

            if u <= ratio:
                accepted_tokens.append(d_tok)
                self.stats.draft_tokens_accepted += 1
            else:
                residual = np.maximum(0.0, t_p - draft_probs[i])
                resampled = self._sample(residual)
                accepted_tokens.append(resampled)
                all_accepted = False
                break

        # 3. If all gamma accepted, sample an extra token from target model
        if all_accepted:
            seq = cur_tokens + accepted_tokens
            t_logits = self._extract_last_logits(self.target_fn(seq))
            t_p = self._softmax(t_logits, self.temperature)
            extra_tok = self._sample(t_p)
            accepted_tokens.append(extra_tok)

        self.stats.total_generated += len(accepted_tokens)
        return accepted_tokens

    def generate(self, prompt: List[int], max_new_tokens: int) -> List[int]:
        """Generates up to max_new_tokens using speculative decoding."""
        generated = list(prompt)
        start_len = len(generated)

        while len(generated) - start_len < max_new_tokens:
            step_tokens = self.step(generated)
            needed = max_new_tokens - (len(generated) - start_len)
            generated.extend(step_tokens[:needed])

        return generated


__all__ = [
    "SpeculativeStats",
    "SpeculativeDraftDecoder",
]
