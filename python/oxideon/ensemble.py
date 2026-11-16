"""Deep Ensembles, Model Soups, and Stochastic Weight Averaging for Oxideon.

Enables combining predictions and weights across checkpoints for superior generalization:
  - VotingClassifier: Aggregates model ensemble outputs via soft or hard voting
  - StochasticWeightAveraging (SWA): Running average of SGD trajectory weights (Izmailov et al., 2018)
  - model_soup: Uniform or weighted parameter space interpolation across fine-tuned checkpoints (Wortsman et al., 2022)
"""

from typing import List, Optional, Any, Dict, Tuple
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .functional import softmax
from .nn import Module, Parameter


def _iter_named_params(model: Any) -> List[Tuple[str, _Tensor]]:
    if hasattr(model, "named_parameters") and callable(model.named_parameters):
        return list(model.named_parameters())
    out: List[Tuple[str, _Tensor]] = []
    for attr in ("weight", "bias", "gamma", "beta"):
        val = getattr(model, attr, None)
        if isinstance(val, _Tensor):
            out.append((attr, val))
    if out:
        return out
    if hasattr(model, "parameters") and callable(model.parameters):
        return [(f"param_{i}", p) for i, p in enumerate(model.parameters())]
    return []


def _set_named_param(model: Any, name: str, arr: np.ndarray) -> None:
    parts = name.split(".")
    obj = model
    for p in parts[:-1]:
        obj = getattr(obj, p)
    t = _from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    if isinstance(obj, Module):
        setattr(obj, parts[-1], Parameter(t))
    else:
        t.requires_grad_(True)
        setattr(obj, parts[-1], t)


class VotingClassifier:
    """Ensemble container aggregating predictions across multiple trained models."""

    def __init__(self, models: List[Any], voting: str = "soft"):
        if not models:
            raise ValueError("VotingClassifier requires at least one model")
        if voting not in ("soft", "hard"):
            raise ValueError(f"voting must be 'soft' or 'hard', got '{voting}'")
        self.models = models
        self.voting = voting

    def predict_proba(self, x: Any) -> _Tensor:
        """Returns ensemble averaged class probabilities."""
        if not isinstance(x, _Tensor):
            x = _from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        probs_list = []
        for m in self.models:
            eval_fn = getattr(m, "eval", None)
            if callable(eval_fn):
                eval_fn()
            out = m(x)
            p = softmax(out, dim=-1)
            probs_list.append(p)

        total_p = probs_list[0]
        for p in probs_list[1:]:
            total_p = total_p + p
        return total_p / float(len(probs_list))

    def predict(self, x: Any) -> np.ndarray:
        """Returns ensemble predicted class indices."""
        if not isinstance(x, _Tensor):
            x = _from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        if self.voting == "soft":
            p = self.predict_proba(x).to_numpy()
            return np.argmax(p, axis=-1)
        else:
            preds = []
            for m in self.models:
                eval_fn = getattr(m, "eval", None)
                if callable(eval_fn):
                    eval_fn()
                preds.append(np.argmax(m(x).to_numpy(), axis=-1))
            preds = np.stack(preds, axis=0)
            if preds.ndim == 1:
                vals, counts = np.unique(preds, return_counts=True)
                return np.int64(vals[np.argmax(counts)])
            out = []
            for col in preds.T:
                vals, counts = np.unique(col, return_counts=True)
                out.append(vals[np.argmax(counts)])
            return np.array(out, dtype=np.int64)

    def __call__(self, x: Any) -> _Tensor:
        return self.predict_proba(x)


class StochasticWeightAveraging:
    """Maintains a running uniform average of model parameter weights."""

    def __init__(self, model: Any):
        self.avg_params: Dict[str, np.ndarray] = {
            name: p.to_numpy().copy() for name, p in _iter_named_params(model)
        }
        self.num_averaged = 1

    def update(self, model: Any):
        """Accumulates current model weights into running average: W_avg = (N * W_avg + W) / (N + 1)."""
        self.num_averaged += 1
        n = self.num_averaged

        curr_params = dict(_iter_named_params(model))
        for name, curr_p in curr_params.items():
            if name in self.avg_params:
                w_avg = self.avg_params[name]
                w_curr = curr_p.to_numpy()
                self.avg_params[name] = (w_avg * (n - 1) + w_curr) / n

    def step(self, model: Any):
        """Alias for update(model)."""
        self.update(model)

    def apply_to(self, model: Any):
        """Applies running average weights to the provided model."""
        for name, arr in self.avg_params.items():
            _set_named_param(model, name, arr)


def model_soup(
    models: List[Any],
    weights: Optional[List[float]] = None,
    target_model: Optional[Any] = None,
) -> Any:
    """Linearly merges parameter weights across multiple fine-tuned models."""
    if not models:
        raise ValueError("Must provide at least one model to merge")

    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    else:
        if len(weights) != len(models):
            raise ValueError(f"Length mismatch: weights has {len(weights)}, models has {len(models)}")
        norm = sum(weights)
        if norm <= 0:
            raise ValueError("sum of weights must be positive")
        weights = [w / norm for w in weights]

    soup = target_model if target_model is not None else models[0]
    eval_fn = getattr(soup, "eval", None)
    if callable(eval_fn):
        eval_fn()

    all_named = [dict(_iter_named_params(m)) for m in models]
    target_params = dict(_iter_named_params(soup))

    for name in target_params.keys():
        merged = np.zeros_like(target_params[name].to_numpy())
        for m_dict, w in zip(all_named, weights):
            merged += w * m_dict[name].to_numpy()
        _set_named_param(soup, name, merged)

    return soup


__all__ = [
    "VotingClassifier",
    "StochasticWeightAveraging",
    "model_soup",
]
