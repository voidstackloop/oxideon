"""Confidence calibration, uncertainty estimation, and Temperature Scaling for Oxideon.

Enables reliable probability estimation and Bayesian epistemic uncertainty:
  - expected_calibration_error (ECE): Quantifies gap between model confidence and accuracy
  - maximum_calibration_error (MCE): Worst-case confidence-accuracy gap across bins
  - reliability_diagram_data: Produces binned metrics for calibration assessment
  - TemperatureScaling: Post-hoc logits calibration optimizer (Guo et al., 2017)
  - MonteCarloDropout: Epistemic uncertainty estimation via MC dropout sampling
"""

from typing import Tuple, Optional, Any, List, Dict
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module, Parameter


def _get_probs_and_targets(
    probs_or_logits: Any, targets: Any
) -> Tuple[np.ndarray, np.ndarray]:
    p = probs_or_logits.to_numpy() if isinstance(probs_or_logits, _Tensor) else np.asarray(probs_or_logits, dtype=np.float32)
    t = targets.to_numpy() if isinstance(targets, _Tensor) else np.asarray(targets, dtype=np.int64).reshape(-1)

    if p.ndim == 1:
        p = np.expand_dims(p, 0)
    elif p.ndim == 0:
        p = p.reshape(-1, 1)

    if len(p) != len(t):
        raise ValueError(f"Length mismatch: probs has {len(p)} samples, targets has {len(t)}")

    if len(t) > 0 and p.shape[-1] > 0:
        if np.any(t < 0) or np.any(t >= p.shape[-1]):
            raise ValueError(f"target class index out of range [0, {p.shape[-1]})")

    # Convert logits to probs if not normalized
    if p.shape[-1] > 0 and (np.any(p < 0) or not np.allclose(np.sum(p, axis=-1), 1.0, atol=1e-2)):
        exp_p = np.exp(p - np.max(p, axis=-1, keepdims=True))
        sum_exp = np.sum(exp_p, axis=-1, keepdims=True)
        p = np.where(sum_exp > 0, exp_p / sum_exp, exp_p)

    return p, t


def expected_calibration_error(
    probs_or_logits: Any, targets: Any, num_bins: int = 10
) -> float:
    """Calculates the Expected Calibration Error (ECE) across confidence bins."""
    if num_bins < 1:
        raise ValueError(f"num_bins must be >= 1, got {num_bins}")
    probs, targets = _get_probs_and_targets(probs_or_logits, targets)

    n = len(probs)
    if n == 0 or probs.shape[-1] == 0:
        return 0.0

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets).astype(np.float32)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        if i == 0:
            bin_mask = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        else:
            bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            avg_conf = np.mean(confidences[bin_mask])
            avg_acc = np.mean(accuracies[bin_mask])
            ece += (bin_size / n) * abs(avg_acc - avg_conf)

    return float(ece)


def maximum_calibration_error(
    probs_or_logits: Any, targets: Any, num_bins: int = 10
) -> float:
    """Calculates the Maximum Calibration Error (MCE) across non-empty confidence bins."""
    if num_bins < 1:
        raise ValueError(f"num_bins must be >= 1, got {num_bins}")
    probs, targets = _get_probs_and_targets(probs_or_logits, targets)

    n = len(probs)
    if n == 0 or probs.shape[-1] == 0:
        return 0.0

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets).astype(np.float32)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    mce = 0.0

    for i in range(num_bins):
        if i == 0:
            bin_mask = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        else:
            bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if np.any(bin_mask):
            avg_conf = float(np.mean(confidences[bin_mask]))
            avg_acc = float(np.mean(accuracies[bin_mask]))
            mce = max(mce, abs(avg_acc - avg_conf))

    return float(mce)


def reliability_diagram_data(
    probs_or_logits: Any, targets: Any, num_bins: int = 10
) -> Dict[str, np.ndarray]:
    """Generates bin accuracies, bin confidences, and bin counts for plotting."""
    if num_bins < 1:
        raise ValueError(f"num_bins must be >= 1, got {num_bins}")
    probs, targets = _get_probs_and_targets(probs_or_logits, targets)

    confidences = np.max(probs, axis=1) if len(probs) > 0 and probs.shape[-1] > 0 else np.zeros(0, dtype=np.float32)
    predictions = np.argmax(probs, axis=1) if len(probs) > 0 and probs.shape[-1] > 0 else np.zeros(0, dtype=np.int64)
    accuracies = (predictions == targets).astype(np.float32) if len(probs) > 0 else np.zeros(0, dtype=np.float32)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    bin_accs = np.zeros(num_bins, dtype=np.float32)
    bin_confs = np.zeros(num_bins, dtype=np.float32)
    bin_counts = np.zeros(num_bins, dtype=np.int64)

    for i in range(num_bins):
        if i == 0:
            bin_mask = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        else:
            bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        count = int(np.sum(bin_mask))
        bin_counts[i] = count
        if count > 0:
            bin_accs[i] = float(np.mean(accuracies[bin_mask]))
            bin_confs[i] = float(np.mean(confidences[bin_mask]))

    return {
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts": bin_counts,
        "bin_boundaries": bin_boundaries,
    }


class TemperatureScaling(Module):
    """Post-hoc probability calibration via single scalar temperature parameter T."""

    def __init__(self, init_temperature: float = 1.5):
        super().__init__()
        if float(init_temperature) <= 0.0:
            raise ValueError(f"init_temperature must be > 0, got {init_temperature}")
        self.temperature = float(init_temperature)

    def forward(self, logits: Any) -> _Tensor:
        """Scales logits by 1.0 / T while preserving Tensor autograd."""
        t = max(float(self.temperature), 1e-4)
        if isinstance(logits, _Tensor):
            return logits / t
        raw = np.asarray(logits, dtype=np.float32)
        arr = raw / t
        return _from_numpy(np.ascontiguousarray(arr, dtype=np.float32))

    def fit(self, val_logits: Any, val_targets: Any, lr: float = 0.01, max_iter: int = 100) -> "TemperatureScaling":
        """Fits temperature T by optimizing cross-entropy loss on validation data using gradient descent."""
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        logits = val_logits.to_numpy() if isinstance(val_logits, _Tensor) else np.asarray(val_logits, dtype=np.float32)
        targets = val_targets.to_numpy() if isinstance(val_targets, _Tensor) else np.asarray(val_targets, dtype=np.int64).reshape(-1)

        if logits.ndim != 2 or len(logits) == 0:
            raise ValueError(f"val_logits must be non-empty 2D [N, C], got shape {logits.shape}")
        if len(logits) != len(targets):
            raise ValueError(f"Length mismatch: val_logits has {len(logits)} samples, val_targets has {len(targets)}")

        t = self.temperature
        n, c = logits.shape
        if np.any(targets < 0) or np.any(targets >= c):
            raise ValueError(f"val_targets class index out of range [0, {c})")
        one_hot = np.zeros((n, c), dtype=np.float32)
        one_hot[np.arange(n), targets] = 1.0

        for _ in range(max_iter):
            scaled_logits = logits / t
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # dLoss / dt = sum_i (probs_i - y_i) * (-logits_i / t^2)
            grad = np.sum((probs - one_hot) * (-logits / (t**2))) / n
            t -= lr * grad
            t = max(t, 0.05)

        self.temperature = float(t)
        return self


class MonteCarloDropout:
    """Performs Monte Carlo Dropout sampling to quantify epistemic prediction uncertainty."""

    def __init__(self, model: Module, num_samples: int = 20):
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        self.model = model
        self.num_samples = num_samples

    def predict(self, x: Any) -> Tuple[_Tensor, _Tensor]:
        """Runs multiple stochastic forward passes with dropout active.

        Returns (mean_predictions, predictive_variance).
        """
        x_tensor = x if isinstance(x, _Tensor) else _from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        # Ensure model is in train mode so dropout applies
        was_training = getattr(self.model, "training", True)
        train_fn = getattr(self.model, "train", None)
        if callable(train_fn):
            train_fn()

        samples = []
        for _ in range(self.num_samples):
            out = self.model(x_tensor).to_numpy()
            samples.append(out)

        if callable(train_fn):
            try:
                train_fn(was_training)
            except TypeError:
                if not was_training and hasattr(self.model, "eval"):
                    self.model.eval()

        samples_arr = np.stack(samples, axis=0)  # [S, B, C]
        mean = np.mean(samples_arr, axis=0)
        variance = np.var(samples_arr, axis=0)

        return (
            _from_numpy(np.ascontiguousarray(mean, dtype=np.float32)),
            _from_numpy(np.ascontiguousarray(variance, dtype=np.float32)),
        )

    def __call__(self, x: Any) -> Tuple[_Tensor, _Tensor]:
        return self.predict(x)


__all__ = [
    "expected_calibration_error",
    "maximum_calibration_error",
    "reliability_diagram_data",
    "TemperatureScaling",
    "MonteCarloDropout",
]
