"""AI Safety, Guardrails, and Alignment Auditing for Oxideon.

Enables content filtering, jailbreak detection, and toxicity classification:
  - ToxicityClassifierHead: Multi-attribute safety classification head with Tensor autograd
  - JailbreakDetector: Identifies prompt injection and adversarial jailbreak patterns
  - OutputGuardrail: Intercepts and filters hazardous generations prior to output
"""

from typing import Tuple, Optional, Any, List, Dict, Set
import re
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .functional import sigmoid
from .nn import Module, Linear, Parameter


class ToxicityClassifierHead(Module):
    """Multi-label classification head predicting safety violations with Tensor autograd."""

    def __init__(self, in_features: int, num_labels: int = 4):
        super().__init__()
        if in_features < 1 or num_labels < 1:
            raise ValueError(f"in_features and num_labels must be >= 1, got {in_features}, {num_labels}")
        self.in_features = in_features
        self.num_labels = num_labels
        self.proj = Linear(in_features, num_labels)

    def forward(self, embeddings: Any) -> _Tensor:
        """Returns per-attribute violation probabilities in [0, 1] while preserving Tensor autograd."""
        x_t = embeddings if isinstance(embeddings, _Tensor) else _from_numpy(np.ascontiguousarray(embeddings, dtype=np.float32))
        sh = list(x_t.shape)
        is_1d = len(sh) == 1
        if is_1d:
            x_t = x_t.reshape([1, sh[0]])
            sh = list(x_t.shape)
        elif len(sh) != 2:
            raise ValueError(f"embeddings must be 1D [D] or 2D [B, D], got shape {tuple(sh)}")
        if sh[1] != self.in_features:
            raise ValueError(f"Feature dimension mismatch: expected {self.in_features}, got {sh[1]}")

        logits = self.proj(x_t)
        probs = logits.sigmoid()
        if is_1d:
            return probs.reshape([self.num_labels])
        return probs


class JailbreakDetector:
    """Detects adversarial prompt injections and jailbreak patterns."""

    INJECTION_PATTERNS = [
        r"ignore (all )?previous instructions",
        r"disregard (all )?prior prompts",
        r"you are now (unrestricted|DAN|jailbroken)",
        r"act as an uncensored AI",
        r"bypass (content )?filters",
        r"system prompt override",
    ]

    def __init__(self, custom_patterns: Optional[List[str]] = None):
        patterns = self.INJECTION_PATTERNS + (custom_patterns or [])
        self.regexes = [re.compile(p, re.IGNORECASE) for p in patterns]

    def detect(self, text: str) -> Dict[str, Any]:
        """Scans input prompt for jailbreak attempts."""
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        matches = []
        for reg in self.regexes:
            m = reg.search(text)
            if m:
                matches.append(m.group(0))

        is_jailbreak = len(matches) > 0
        return {
            "is_jailbreak": is_jailbreak,
            "matched_patterns": matches,
            "risk_score": 1.0 if is_jailbreak else 0.0,
        }

    def __call__(self, text: str) -> Dict[str, Any]:
        return self.detect(text)


class OutputGuardrail:
    """Validates candidate outputs against prohibited content rules."""

    def __init__(self, blocked_words: Optional[List[str]] = None, max_length: int = 4000):
        if max_length < 0:
            raise ValueError("max_length must be non-negative")
        self.blocked_words = sorted(set(w.lower().strip() for w in (blocked_words or []) if w.strip()))
        self.max_length = max_length

    @staticmethod
    def _word_pattern(word: str) -> str:
        # Use word boundaries if the blocked term starts/ends with word characters
        prefix = r"\b" if re.match(r"^\w", word) else ""
        suffix = r"\b" if re.search(r"\w$", word) else ""
        return prefix + re.escape(word) + suffix

    def check(self, candidate_text: str) -> Tuple[bool, Optional[str]]:
        """Returns (is_safe, violation_reason)."""
        if len(candidate_text) > self.max_length:
            return False, f"Output length {len(candidate_text)} exceeds maximum {self.max_length}"

        for word in self.blocked_words:
            if re.search(self._word_pattern(word), candidate_text, flags=re.IGNORECASE):
                return False, f"Prohibited keyword detected: '{word}'"

        return True, None

    def sanitize(self, candidate_text: str, replacement: str = "[REDACTED]") -> str:
        """Replaces any blocked words in candidate_text (case-insensitively) and truncates to max_length."""
        out = candidate_text
        for word in self.blocked_words:
            out = re.sub(self._word_pattern(word), replacement, out, flags=re.IGNORECASE)
        if len(out) > self.max_length:
            out = out[: self.max_length]
        return out

    def __call__(self, candidate_text: str) -> Tuple[bool, Optional[str]]:
        return self.check(candidate_text)


__all__ = [
    "ToxicityClassifierHead",
    "JailbreakDetector",
    "OutputGuardrail",
]
