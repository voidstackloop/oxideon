"""Offline-first evaluation harness for language models: metrics, task families, runner.

Covers the classic post-training eval loop without requiring dataset downloads:

- Metrics: ``accuracy``, ``exact_match``, ``token_f1``, ``pass_at_k``,
  ``math_equal`` (GSM8K/MATH style), ``rouge_l``, ``corpus_bleu``.
- Tasks: ``MultipleChoiceTask`` (MMLU/GPQA/ARC style, loglik or generate),
  ``MathTask`` (GSM8K style), ``CodeTask`` (HumanEval style, sandboxed
  execution), ``NeedleTask`` (synthetic needle-in-a-haystack long context).
- Runner: ``run_suite`` aggregates tasks into an ``EvalReport`` with dict/JSON
  export and a text summary.

Models are duck-typed. A model must provide ``generate(prompts)`` returning a
list of strings, and may additionally provide ``loglikelihoods(contexts,
continuations)`` (paired lists) or ``loglikelihood(context, continuation)``
returning log-probabilities, which unlocks the loglikelihood path of
``MultipleChoiceTask``.
"""

from dataclasses import dataclass, field
import math
import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .nn import _check_elements, _check_non_negative, _check_positive, _check_repeats


# ---------------------------------------------------------------------------
# Answer normalization and text metrics
# ---------------------------------------------------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    """Lowercases, removes articles/punctuation/extra whitespace (SQuAD style)."""
    text = str(text).lower()
    text = _ARTICLES.sub(" ", text)
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def exact_match(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Mean exact match over normalized prediction/reference pairs."""
    preds = [normalize_answer(p) for p in predictions]
    refs = [normalize_answer(r) for r in references]
    if len(preds) != len(refs):
        raise ValueError(f"count mismatch: {len(preds)} predictions vs {len(refs)} references")
    if not preds:
        raise ValueError("exact_match needs at least one pair")
    return sum(1.0 for p, r in zip(preds, refs) if p == r) / len(preds)


def token_f1(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Mean token F1 over normalized prediction/reference pairs (SQuAD style)."""
    preds = [normalize_answer(p).split() for p in predictions]
    refs = [normalize_answer(r).split() for r in references]
    if len(preds) != len(refs):
        raise ValueError(f"count mismatch: {len(preds)} predictions vs {len(refs)} references")
    if not preds:
        raise ValueError("token_f1 needs at least one pair")
    total = 0.0
    for p, r in zip(preds, refs):
        if not p and not r:
            total += 1.0
            continue
        if not p or not r:
            continue
        common = 0
        counts: Dict[str, int] = {}
        for tok in p:
            counts[tok] = counts.get(tok, 0) + 1
        for tok in r:
            if counts.get(tok, 0) > 0:
                counts[tok] -= 1
                common += 1
        if common == 0:
            continue
        total += 2.0 * common / (len(p) + len(r))
    return total / len(preds)


def accuracy(predictions: Sequence[Any], references: Sequence[Any]) -> float:
    """Fraction of predictions equal to their reference."""
    preds = list(predictions)
    refs = list(references)
    if len(preds) != len(refs):
        raise ValueError(f"count mismatch: {len(preds)} predictions vs {len(refs)} references")
    if not preds:
        raise ValueError("accuracy needs at least one pair")
    return sum(1.0 for p, r in zip(preds, refs) if p == r) / len(preds)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator: 1 - C(n-c, k) / C(n, k) (HumanEval style)."""
    n = _check_positive("n", n)
    c = _check_non_negative("c", c)
    k = _check_positive("k", k)
    if c > n:
        raise ValueError(f"c = {c} exceeds sample count n = {n}")
    if k > n:
        raise ValueError(f"k = {k} exceeds sample count n = {n}")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_NUMBER = re.compile(r"-?\d[\d,]*(?:/\d[\d,]*|\.?\d*)")


def _last_number(text: str) -> Optional[str]:
    found = _NUMBER.findall(text)
    return found[-1] if found else None


def _parse_numeric(s: str) -> float:
    s = s.strip()
    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2:
            denom = float(parts[1])
            if denom != 0.0:
                return float(parts[0]) / denom
    return float(s)


def math_equal(prediction: str, reference: str, rel_tol: float = 1e-6) -> bool:
    """GSM8K/MATH style answer check: boxed answer or last number/fraction, numeric compare.

    Prefers ``\\boxed{...}`` when present, else the last numeric token of the
    prediction. Compares as strings first, then as floats (including simple a/b fractions) within ``rel_tol``.
    """
    ref = str(reference).strip().replace(",", "")
    boxed = _BOXED.search(str(prediction))
    if boxed is not None:
        cand = boxed.group(1).strip().replace(",", "")
    else:
        num = _last_number(str(prediction))
        cand = num.replace(",", "") if num is not None else str(prediction).strip()
    if cand == ref:
        return True
    try:
        a, b = _parse_numeric(cand), _parse_numeric(ref)
    except ValueError:
        return False
    if not (math.isfinite(a) and math.isfinite(b)):
        return a == b
    return abs(a - b) <= rel_tol * max(1.0, abs(a), abs(b))


_LCS_CELLS = 1 << 24


def _lcs_length(a: List[str], b: List[str]) -> int:
    if len(a) * len(b) > _LCS_CELLS:
        raise ValueError(
            f"inputs too large for exact ROUGE-L ({len(a)}x{len(b)} cells "
            f"above the limit of {_LCS_CELLS})"
        )
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            if x == y:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = prev[j] if prev[j] >= cur[j - 1] else cur[j - 1]
        prev = cur
    return prev[len(b)]


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 over whitespace tokens (single pair)."""
    p = str(prediction).split()
    r = str(reference).split()
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    lcs = _lcs_length(p, r)
    if lcs == 0:
        return 0.0
    prec = lcs / len(p)
    rec = lcs / len(r)
    return 2.0 * prec * rec / (prec + rec)


def _ngram_counts(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    counts: Dict[Tuple[str, ...], int] = {}
    for i in range(len(tokens) - n + 1):
        key = tuple(tokens[i : i + n])
        counts[key] = counts.get(key, 0) + 1
    return counts


def corpus_bleu(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
    max_order: int = 4,
) -> float:
    """Corpus BLEU with add-one smoothing on orders 2+ (references are raw strings)."""
    max_order = _check_positive("max_order", max_order)
    if max_order > 4:
        raise ValueError(f"max_order = {max_order} exceeds the supported limit of 4")
    preds = [str(p).split() for p in predictions]
    ref_sets = [[str(r).split() for r in refs] for refs in references]
    if len(preds) != len(ref_sets):
        raise ValueError(f"count mismatch: {len(preds)} predictions vs {len(ref_sets)} references")
    if not preds:
        raise ValueError("corpus_bleu needs at least one pair")
    for refs in ref_sets:
        if not refs:
            raise ValueError("corpus_bleu needs at least one reference per prediction")
    pred_len = sum(len(p) for p in preds)
    if pred_len == 0:
        return 0.0
    ref_len = 0
    for p, refs in zip(preds, ref_sets):
        ref_len += min(refs, key=lambda r: (abs(len(r) - len(p)), len(r))).__len__()
    log_prec = 0.0
    for n in range(1, max_order + 1):
        match, total = 0, 0
        for p, refs in zip(preds, ref_sets):
            counts = _ngram_counts(p, n)
            total += sum(counts.values())
            if not counts:
                continue
            ceilings: Dict[Tuple[str, ...], int] = {}
            for r in refs:
                for key, cnt in _ngram_counts(r, n).items():
                    if cnt > ceilings.get(key, 0):
                        ceilings[key] = cnt
            for key, cnt in counts.items():
                match += min(cnt, ceilings.get(key, 0))
        if n == 1:
            prec = match / total if total else 0.0
        else:
            prec = (match + 1) / (total + 1)
        if prec <= 0.0:
            return 0.0
        log_prec += math.log(prec) / max_order
    bp = 1.0 if pred_len >= ref_len else math.exp(1.0 - ref_len / pred_len)
    return bp * math.exp(log_prec)


def perplexity_from_nll(mean_nll: float) -> float:
    """Perplexity from a mean negative log-likelihood."""
    if not math.isfinite(mean_nll):
        raise ValueError(f"mean_nll must be finite, got {mean_nll}")
    try:
        return math.exp(mean_nll)
    except OverflowError:
        raise ValueError(f"mean_nll = {mean_nll} overflows perplexity")


# ---------------------------------------------------------------------------
# Task results and model probing
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    """Scores for one task: metric name -> value, plus per-item details."""

    name: str
    n_items: int
    metrics: Dict[str, float] = field(default_factory=dict)
    details: List[Any] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_items": self.n_items,
            "metrics": dict(self.metrics),
            "details": list(self.details),
        }


def _generate(model: Any, prompts: List[str]) -> List[str]:
    generate = getattr(model, "generate", None)
    if not callable(generate):
        raise TypeError(f"model {type(model).__name__} has no callable generate(prompts) method")
    outputs = generate(prompts)
    if len(outputs) != len(prompts):
        raise ValueError(
            f"model returned {len(outputs)} outputs for {len(prompts)} prompts"
        )
    return [str(o) for o in outputs]


def _score_loglik(
    model: Any, contexts: List[str], continuations: List[str]
) -> Optional[List[float]]:
    """Paired loglikelihoods, or None when the model has no loglikelihood API."""
    if len(contexts) != len(continuations):
        raise ValueError("contexts and continuations must pair up")
    batch = getattr(model, "loglikelihoods", None)
    if callable(batch):
        scores = [float(s) for s in batch(contexts, continuations)]
    else:
        single = getattr(model, "loglikelihood", None)
        if not callable(single):
            return None
        scores = [float(single(c, t)) for c, t in zip(contexts, continuations)]
    if any(math.isnan(s) for s in scores):
        raise ValueError("model returned a NaN loglikelihood score")
    return scores


# ---------------------------------------------------------------------------
# Multiple choice (MMLU / GPQA / ARC style)
# ---------------------------------------------------------------------------


@dataclass
class ChoiceItem:
    """One multiple-choice item: prompt, choice list, index of the answer."""

    prompt: str
    choices: List[str]
    answer: int
    subject: str = ""

    def __post_init__(self) -> None:
        if len(self.choices) < 2:
            raise ValueError(f"need at least 2 choices, got {len(self.choices)}")
        if not 0 <= self.answer < len(self.choices):
            raise ValueError(f"answer = {self.answer} outside {len(self.choices)} choices")


class MultipleChoiceTask:
    """Lettered multiple choice; scores by loglikelihood when available."""

    def __init__(
        self,
        name: str,
        items: Sequence[ChoiceItem],
        fewshot: Sequence[ChoiceItem] = (),
    ) -> None:
        self.name = str(name)
        self.items = list(items)
        self.fewshot = list(fewshot)
        if not self.items:
            raise ValueError("MultipleChoiceTask needs at least one item")
        width = len(self.items[0].choices)
        if width > 26:
            raise ValueError(f"{width} choices exceed the 26 letter labels")
        for item in self.items:
            if len(item.choices) != width:
                raise ValueError("all items must have the same number of choices")

    def __len__(self) -> int:
        return len(self.items)

    def _render_item(self, item: ChoiceItem, with_answer: bool) -> str:
        lines = [item.prompt]
        for i, choice in enumerate(item.choices):
            lines.append(f"{chr(ord('A') + i)}. {choice}")
        lines.append("Answer:")
        if with_answer:
            lines.append(f" {chr(ord('A') + item.answer)}")
        return "\n".join(lines)

    def render(self, index: int) -> str:
        """Prompt for item ``index`` with few-shot examples prepended."""
        parts = [self._render_item(ex, True) for ex in self.fewshot]
        parts.append(self._render_item(self.items[index], False))
        return "\n\n".join(parts)

    def _parse_letter(self, text: str, width: int) -> Optional[int]:
        valid_letters = "".join(chr(ord("A") + i) for i in range(width))
        # 1. Explicit "answer is X" or "Answer: X" pattern
        m = re.search(rf"(?:answer(?:\s+is)?|option|choice)\s*[:\-]?\s*\(?([{valid_letters}])\)?\b", text, re.IGNORECASE)
        if m:
            return ord(m.group(1).upper()) - ord("A")
        # 2. Standalone word-boundary valid choice letter
        m_word = re.findall(rf"\b([{valid_letters}])\b", text.upper())
        if m_word:
            return ord(m_word[-1]) - ord("A")
        # 3. Fallback: first character if it is a valid choice letter
        stripped = text.strip().upper()
        if stripped and "A" <= stripped[0] <= "Z":
            idx = ord(stripped[0]) - ord("A")
            return idx if idx < width else None
        return None

    def evaluate(self, model: Any, limit: Optional[int] = None) -> TaskResult:
        """Score the task; ``limit`` caps the evaluated prefix of items."""
        items = self.items if limit is None else self.items[: _check_positive("limit", limit)]
        width = len(self.items[0].choices)
        contexts = [self.render(i) for i in range(len(items))]
        scores = _score_loglik(
            model,
            [c for ctx in contexts for c in [ctx] * width],
            [f" {chr(ord('A') + i)}" for _ in contexts for i in range(width)],
        )
        correct = 0
        details: List[Dict[str, Any]] = []
        if scores is not None:
            if len(scores) != len(items) * width:
                raise ValueError("model returned a miscounted loglikelihood batch")
            for row, item in enumerate(items):
                row_scores = scores[row * width : (row + 1) * width]
                pred = max(range(width), key=row_scores.__getitem__)
                hit = pred == item.answer
                correct += hit
                details.append({"pred": pred, "answer": item.answer, "correct": bool(hit)})
        else:
            for ctx, item in zip(contexts, items):
                pred = self._parse_letter(_generate(model, [ctx])[0], width)
                hit = pred == item.answer
                correct += hit
                details.append({"pred": pred, "answer": item.answer, "correct": bool(hit)})
        return TaskResult(
            name=self.name,
            n_items=len(items),
            metrics={"accuracy": correct / len(items)},
            details=details,
        )


# ---------------------------------------------------------------------------
# Math word problems (GSM8K style)
# ---------------------------------------------------------------------------


@dataclass
class MathItem:
    """One math item: prompt plus the reference answer."""

    prompt: str
    answer: str


class MathTask:
    """Grade-school math scoring generations with ``math_equal``."""

    def __init__(
        self,
        name: str,
        items: Sequence[MathItem],
        fewshot: Sequence[MathItem] = (),
    ) -> None:
        self.name = str(name)
        self.items = list(items)
        self.fewshot = list(fewshot)
        if not self.items:
            raise ValueError("MathTask needs at least one item")

    def __len__(self) -> int:
        return len(self.items)

    def render(self, index: int) -> str:
        parts = [f"Q: {ex.prompt}\nA: {ex.answer}" for ex in self.fewshot]
        parts.append(f"Q: {self.items[index].prompt}\nA:")
        return "\n\n".join(parts)

    def evaluate(self, model: Any, limit: Optional[int] = None) -> TaskResult:
        items = self.items if limit is None else self.items[: _check_positive("limit", limit)]
        prompts = [self.render(i) for i in range(len(items))]
        outputs = _generate(model, prompts)
        correct = 0
        details: List[Dict[str, Any]] = []
        for item, out in zip(items, outputs):
            hit = math_equal(out, item.answer)
            correct += hit
            details.append({"output": out[:200], "answer": item.answer, "correct": bool(hit)})
        return TaskResult(
            name=self.name,
            n_items=len(items),
            metrics={"accuracy": correct / len(items)},
            details=details,
        )


# ---------------------------------------------------------------------------
# Code generation (HumanEval style, sandboxed)
# ---------------------------------------------------------------------------


@dataclass
class CodeItem:
    """One code item: prompt plus test source appended after the completion."""

    prompt: str
    tests: str
    name: str = "task"


def _posix_limits() -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (1 << 31, 1 << 31))
    except (ImportError, ValueError, OSError):
        pass


def run_untrusted_tests(completion: str, tests: str, timeout: int = 10) -> bool:
    """Executes ``completion + tests`` in a subprocess; True on exit 0."""
    timeout = _check_positive("timeout", timeout)
    script = str(completion) + "\n" + str(tests) + "\n"
    with tempfile.TemporaryDirectory(prefix="oxevals") as tmp:
        prog = os.path.join(tmp, "solution.py")
        with open(prog, "w", encoding="utf-8") as f:
            f.write(script)
        try:
            proc = subprocess.run(
                [sys.executable, prog],
                cwd=tmp,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
                preexec_fn=_posix_limits if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired:
            return False
        except OSError:
            return False
        return proc.returncode == 0


class CodeTask:
    """Function completion scored by hidden tests; reports pass@k."""

    def __init__(self, name: str, items: Sequence[CodeItem]) -> None:
        self.name = str(name)
        self.items = list(items)
        if not self.items:
            raise ValueError("CodeTask needs at least one item")

    def __len__(self) -> int:
        return len(self.items)

    def evaluate(
        self,
        model: Any,
        limit: Optional[int] = None,
        num_samples: int = 1,
        timeout: int = 10,
    ) -> TaskResult:
        """Samples ``num_samples`` completions per prompt and scores pass@k."""
        items = self.items if limit is None else self.items[: _check_positive("limit", limit)]
        num_samples = _check_repeats("num_samples", _check_positive("num_samples", num_samples))
        timeout = _check_positive("timeout", timeout)
        top_k = min(num_samples, 10)
        pass1_total = 0.0
        passk_total = 0.0
        details: List[Dict[str, Any]] = []
        for item in items:
            completions = _generate(model, [item.prompt] * num_samples)
            passed = sum(
                1 for code in completions if run_untrusted_tests(code, item.tests, timeout)
            )
            pass1_total += pass_at_k(num_samples, passed, 1)
            passk_total += pass_at_k(num_samples, passed, top_k)
            details.append({"name": item.name, "passed": passed, "n": num_samples})
        metrics = {"pass@1": pass1_total / len(items)}
        if top_k != 1:
            metrics[f"pass@{top_k}"] = passk_total / len(items)
        return TaskResult(name=self.name, n_items=len(items), metrics=metrics, details=details)


# ---------------------------------------------------------------------------
# Long context needle-in-a-haystack (synthetic, offline)
# ---------------------------------------------------------------------------

_FILLER_SENTENCES = [
    "The city archives hold records of every building permit since 1921.",
    "Gardeners recommend watering deeply but infrequently during dry spells.",
    "The museum extended its evening hours through the end of the season.",
    "Engineers tested the bridge cables under twice the expected load.",
    "Local bakeries report rye loaves selling out before noon each day.",
    "The library digitized three thousand maps from the survey collection.",
    "Astronomers tracked the comet as it crossed the northern sky.",
    "The orchestra rehearsed the new symphony movement twice daily.",
    "Ferry service resumes on weekends starting with the spring tide.",
    "Botanists catalogued forty orchid species in the highland reserve.",
    "The council approved funds to repave the old market square.",
    "Volunteers planted two hundred saplings along the river bank.",
]


def make_haystack(needle: str, total_words: int, depth: float, seed: int = 0) -> str:
    """Builds filler context with ``needle`` planted at ``depth`` (0..1)."""
    total_words = _check_elements("total_words", _check_positive("total_words", total_words))
    if not 0.0 <= depth <= 1.0:
        raise ValueError(f"depth = {depth} must sit within [0, 1]")
    words: List[str] = []
    i = seed % max(len(_FILLER_SENTENCES), 1)
    while len(words) < total_words:
        words.extend(_FILLER_SENTENCES[i % len(_FILLER_SENTENCES)].split())
        i += 7
    words = words[:total_words]
    needle_words = str(needle).split()
    at = min(len(words), int(round(depth * len(words))))
    return " ".join(words[:at] + needle_words + words[at:])


class NeedleTask:
    """Synthetic long-context recall over (length, depth) grids."""

    def __init__(
        self,
        name: str = "needle",
        lengths: Sequence[int] = (512, 2048),
        depths: Sequence[float] = (0.0, 0.5, 1.0),
        needle: str = "The magic number for the gate is 7319.",
        question: str = "What is the magic number for the gate?",
        answer: str = "7319",
        seed: int = 0,
    ) -> None:
        self.name = str(name)
        self.lengths = [_check_positive("lengths", n) for n in lengths]
        self.depths = [float(d) for d in depths]
        for d in self.depths:
            if not 0.0 <= d <= 1.0:
                raise ValueError(f"depth = {d} must sit within [0, 1]")
        if not self.lengths or not self.depths:
            raise ValueError("NeedleTask needs at least one length and one depth")
        self.needle = str(needle)
        self.question = str(question)
        self.answer = str(answer)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.lengths) * len(self.depths)

    def render(self, index: int) -> str:
        length = self.lengths[index // len(self.depths)]
        depth = self.depths[index % len(self.depths)]
        context = make_haystack(self.needle, length, depth, self.seed + index)
        return f"{context}\n\nQuestion: {self.question}\nAnswer:"

    def evaluate(self, model: Any, limit: Optional[int] = None) -> TaskResult:
        n = len(self) if limit is None else min(len(self), _check_positive("limit", limit))
        prompts = [self.render(i) for i in range(n)]
        outputs = _generate(model, prompts)
        want = self.answer.casefold()
        correct = 0
        details: List[Dict[str, Any]] = []
        for i, out in enumerate(outputs):
            hit = want in out.casefold()
            correct += hit
            details.append(
                {
                    "length": self.lengths[i // len(self.depths)],
                    "depth": self.depths[i % len(self.depths)],
                    "correct": bool(hit),
                }
            )
        return TaskResult(
            name=self.name,
            n_items=n,
            metrics={"accuracy": correct / n},
            details=details,
        )


def demo_choices(n: int = 8, seed: int = 0) -> List[ChoiceItem]:
    """Deterministic arithmetic choice items for smoke tests and docs."""
    n = _check_positive("n", n)
    items = []
    for i in range(n):
        a = (seed * 31 + i * 7 + 3) % 20 + 2
        b = (seed * 17 + i * 13 + 5) % 20 + 2
        answer = a * b
        choices = [answer, answer + 1, answer - 1, answer + 10]
        items.append(
            ChoiceItem(
                prompt=f"What is {a} times {b}?",
                choices=[str(c) for c in choices],
                answer=0,
                subject="arithmetic",
            )
        )
    return items


def demo_math() -> List[MathItem]:
    """Tiny hand-written grade-school math set for smoke tests and docs."""
    return [
        MathItem("Maya has 3 boxes with 4 apples each. How many apples?", "12"),
        MathItem("A train travels 60 km in 2 hours. How far in 5 hours?", "150"),
        MathItem("Tom has 17 marbles and gives away 9. How many are left?", "8"),
        MathItem("A baker makes 24 cookies per tray. How many in 3 trays?", "72"),
    ]


def demo_code() -> List[CodeItem]:
    """Tiny hand-written code tasks: completion + tests run as one script."""
    return [
        CodeItem(
            prompt="def add(a, b):\n    return a + b\n",
            tests="assert add(1, 2) == 3\nassert add(-1, 1) == 0\n",
            name="add",
        ),
        CodeItem(
            prompt="def is_even(n):\n    return n % 2 == 0\n",
            tests="assert is_even(4) is True\nassert is_even(5) is False\n",
            name="is_even",
        ),
    ]


@dataclass
class EvalReport:
    """Aggregated results of ``run_suite`` with dict/JSON/text export."""

    model_name: str
    results: List[TaskResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, path: str) -> str:
        """Writes the report as JSON; returns the path written."""
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "EvalReport":
        """Reads a report written by ``save``."""
        import json

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        results = [
            TaskResult(
                name=r["name"],
                n_items=int(r["n_items"]),
                metrics={k: float(v) for k, v in r["metrics"].items()},
                details=list(r.get("details", [])),
            )
            for r in data["results"]
        ]
        return cls(model_name=str(data["model_name"]), results=results)

    def summary(self) -> str:
        lines = [f"eval report for {self.model_name}"]
        for r in self.results:
            scored = "  ".join(f"{k}={v:.4f}" for k, v in sorted(r.metrics.items()))
            lines.append(f"{r.name}  (n={r.n_items})  {scored}")
        return "\n".join(lines)


def run_suite(
    model: Any,
    tasks: Sequence[Any],
    limit: Optional[int] = None,
    model_name: str = "model",
    verbose: bool = False,
    task_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> EvalReport:
    """Evaluates every task and aggregates into an ``EvalReport``."""
    task_list = list(tasks)
    if not task_list:
        raise ValueError("run_suite needs at least one task")
    if limit is not None:
        _check_positive("limit", limit)
    task_kwargs = task_kwargs or {}
    report = EvalReport(model_name=str(model_name))
    printer: Callable[[str], None] = print if verbose else (lambda _s: None)
    for task in task_list:
        evaluate = getattr(task, "evaluate", None)
        if not callable(evaluate):
            raise TypeError(f"task {type(task).__name__} has no evaluate(model, ...) method")
        kwargs = dict(task_kwargs.get(getattr(task, "name", ""), {}))
        result = evaluate(model, limit=limit, **kwargs)
        if not isinstance(result, TaskResult):
            raise TypeError(
                f"task {getattr(task, 'name', '?')} returned {type(result).__name__}, "
                "expected TaskResult"
            )
        report.results.append(result)
        scored = "  ".join(f"{k}={v:.4f}" for k, v in sorted(result.metrics.items()))
        printer(f"{result.name} (n={result.n_items}) {scored}")
    return report


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Loads a JSONL file into a list of dicts (one per non-blank line)."""
    import json

    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{lineno} is not a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} holds no JSON rows")
    return rows


__all__ = [
    "TaskResult",
    "EvalReport",
    "ChoiceItem",
    "MultipleChoiceTask",
    "MathItem",
    "MathTask",
    "CodeItem",
    "CodeTask",
    "NeedleTask",
    "run_suite",
    "run_untrusted_tests",
    "make_haystack",
    "demo_choices",
    "demo_math",
    "demo_code",
    "load_jsonl",
    "normalize_answer",
    "exact_match",
    "token_f1",
    "accuracy",
    "pass_at_k",
    "math_equal",
    "rouge_l",
    "corpus_bleu",
    "perplexity_from_nll",
    "rag_context_precision",
    "rag_context_recall",
    "rag_faithfulness",
    "rag_answer_relevancy",
]


def _as_binary_flags(values: Any, name: str) -> List[bool]:
    flags = list(values)
    if not flags:
        raise ValueError(f"{name} needs at least one verdict")
    out = []
    for v in flags:
        if v is True or v == 1:
            out.append(True)
        elif v is False or v == 0:
            out.append(False)
        else:
            raise ValueError(f"{name} verdicts must be boolean (got {v!r})")
    return out


def rag_context_precision(relevant: Sequence[Any], k: Optional[int] = None) -> float:
    """RAGAS-style precision@k over ranked context relevance verdicts."""
    flags = _as_binary_flags(relevant, "relevant")
    k = len(flags) if k is None else _check_positive("k", k)
    if k > len(flags):
        raise ValueError(f"k = {k} exceeds {len(flags)} contexts")
    return sum(1.0 for v in flags[:k] if v) / k


def rag_context_recall(supported: Sequence[Any]) -> float:
    """RAGAS-style recall: fraction of ground-truth claims the contexts support."""
    flags = _as_binary_flags(supported, "supported")
    return sum(1.0 for v in flags if v) / len(flags)


def rag_faithfulness(supported: Sequence[Any]) -> float:
    """RAGAS-style faithfulness: fraction of answer claims contexts support."""
    flags = _as_binary_flags(supported, "supported")
    return sum(1.0 for v in flags if v) / len(flags)


def rag_answer_relevancy(similarities: Sequence[float]) -> float:
    """RAGAS-style relevancy: mean similarity of generated to asked questions."""
    sims = [float(s) for s in similarities]
    if not sims:
        raise ValueError("similarities needs at least one value")
    if not all(math.isfinite(s) for s in sims):
        raise ValueError("similarities must be finite")
    return sum(sims) / len(sims)
