"""Evaluation metrics for classification, regression, and ranking."""

import math
import numpy as np


def _to_numpy(x):
    if hasattr(x, "to_numpy"):
        return x.to_numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _to_binary_labels(a):
    """Thresholds an `[N, 1]` score column to {0, 1}: values already in [0, 1]
    are read as probabilities (0.5 cut), anything else as logits (0.0 cut)."""
    col = a.reshape(-1)
    if col.size > 0 and np.all(col >= 0.0) and np.all(col <= 1.0):
        return (col > 0.5).astype(np.int64)
    return (col > 0.0).astype(np.int64)


def _extract_class_indices(arr):
    a = _to_numpy(arr)
    if a.ndim > 1 and a.shape[-1] > 1:
        a = np.argmax(a, axis=-1)
    elif a.ndim == 2 and a.shape[-1] == 1:
        if np.issubdtype(a.dtype, np.floating):
            a = _to_binary_labels(a)
        else:
            a = a.reshape(-1)
    return a.reshape(-1)


def accuracy(preds, targets) -> float:
    """Computes top-1 classification accuracy.

    Supports:
      - Discrete class index arrays / Tensors.
      - 2D logits / probabilities `[N, C]` (argmax taken automatically).
      - Binary score columns `[N, 1]` (thresholded; see `_to_binary_labels`).
    """
    p = _extract_class_indices(preds)
    t = _extract_class_indices(targets)
    if len(t) == 0:
        return 0.0
    return float(np.mean(p == t))


def topk_accuracy(preds, targets, k: int = 5) -> float:
    """Computes Top-K classification accuracy."""
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"topk_accuracy requires k >= 1, got {k!r}")
    p = _to_numpy(preds)
    t = _to_numpy(targets).reshape(-1)
    if p.ndim < 2:
        raise ValueError("topk_accuracy requires 2D predictions of shape [N, C]")
    k = min(int(k), p.shape[-1])
    topk_preds = np.argpartition(p, -k, axis=-1)[:, -k:]
    if len(t) == 0:
        return 0.0
    correct = np.any(topk_preds == t[:, None], axis=-1)
    return float(np.mean(correct))


def confusion_matrix(preds, targets, num_classes=None):
    """Computes confusion matrix where entry (i, j) is ground truth i predicted as j."""
    p = _extract_class_indices(preds).astype(int)
    t = _extract_class_indices(targets).astype(int)
    if len(p) == 0:
        n = num_classes if num_classes is not None else 0
        return np.zeros((n, n), dtype=np.int64)

    if num_classes is None:
        num_classes = max(int(np.max(p)), int(np.max(t))) + 1 if len(p) > 0 else 0

    if num_classes == 0:
        return np.zeros((0, 0), dtype=np.int64)

    if not np.all((t >= 0) & (t < num_classes) & (p >= 0) & (p < num_classes)):
        bad_t = sorted(set(t[(t < 0) | (t >= num_classes)].tolist()))
        bad_p = sorted(set(p[(p < 0) | (p >= num_classes)].tolist()))
        raise ValueError(
            f"confusion_matrix: labels out of range for num_classes={num_classes} "
            f"(targets {bad_t}, preds {bad_p}); pass a larger num_classes"
        )
    indices = t * num_classes + p
    return np.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def precision_recall_f1(preds, targets, num_classes=None, average: str = "macro", eps: float = 1e-9):
    """Computes precision, recall, and F1-score across classes.

    Supported averages: 'macro', 'micro', 'weighted', 'binary'.
    Returns: dict with keys: 'precision', 'recall', 'f1'
    """
    p = _extract_class_indices(preds).astype(int)
    t = _extract_class_indices(targets).astype(int)
    if len(p) == 0:
        if average == "binary":
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if num_classes is None:
        num_classes = max(int(np.max(p)), int(np.max(t))) + 1 if len(p) > 0 else 0

    if num_classes == 0:
        if average == "binary":
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    cm = confusion_matrix(p, t, num_classes=num_classes)
    tp = np.diag(cm).astype(float)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    support = np.sum(cm, axis=1)

    precision_per_class = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1.0), 0.0)
    recall_per_class = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1.0), 0.0)
    f1_per_class = np.where(
        precision_per_class + recall_per_class > 0,
        2.0 * precision_per_class * recall_per_class / np.maximum(precision_per_class + recall_per_class, 1e-12),
        0.0,
    )

    if average == "binary":
        idx = 1 if num_classes > 1 else 0
        return {
            "precision": float(precision_per_class[idx]),
            "recall": float(recall_per_class[idx]),
            "f1": float(f1_per_class[idx]),
            "support": int(support[idx]),
        }
    elif average == "macro":
        return {
            "precision": float(np.mean(precision_per_class)),
            "recall": float(np.mean(recall_per_class)),
            "f1": float(np.mean(f1_per_class)),
        }
    elif average == "weighted":
        total_sup = float(np.sum(support))
        weights = support / total_sup if total_sup > 0 else np.zeros_like(support)
        return {
            "precision": float(np.sum(precision_per_class * weights)),
            "recall": float(np.sum(recall_per_class * weights)),
            "f1": float(np.sum(f1_per_class * weights)),
        }
    elif average == "micro":
        total_tp = float(np.sum(tp))
        total_fp = float(np.sum(fp))
        total_fn = float(np.sum(fn))
        micro_p = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
        micro_r = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
        micro_f1 = 2.0 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r > 0 else 0.0
        return {"precision": float(micro_p), "recall": float(micro_r), "f1": float(micro_f1)}
    else:
        raise ValueError(f"Unsupported average mode '{average}'; expected 'macro', 'micro', 'weighted', or 'binary'")


def matthews_corrcoef(preds, targets) -> float:
    """Computes the Matthews Correlation Coefficient (MCC) for binary or multiclass classification."""
    cm = confusion_matrix(preds, targets).astype(np.float64)
    if cm.size == 0:
        return 0.0
    t_k = np.sum(cm, axis=1)
    p_k = np.sum(cm, axis=0)
    c = np.trace(cm)
    s = np.sum(cm)
    cov_ytyp = c * s - float(np.dot(t_k, p_k))
    cov_ypyp = s * s - float(np.dot(p_k, p_k))
    cov_ytyt = s * s - float(np.dot(t_k, t_k))
    denom = math.sqrt(max(cov_ypyp * cov_ytyt, 0.0))
    if denom == 0.0:
        return 0.0
    return float(cov_ytyp / denom)


def cohen_kappa_score(preds, targets) -> float:
    """Computes Cohen's Kappa inter-annotator agreement score."""
    cm = confusion_matrix(preds, targets).astype(np.float64)
    if cm.size == 0:
        return 0.0
    n = float(np.sum(cm))
    if n == 0.0:
        return 0.0
    po = float(np.trace(cm)) / n
    pe = float(np.dot(np.sum(cm, axis=0), np.sum(cm, axis=1))) / (n * n)
    if abs(1.0 - pe) < 1e-12:
        return 1.0
    return float((po - pe) / (1.0 - pe))


def roc_auc_score(scores, targets) -> float:
    """Computes Area Under the Receiver Operating Characteristic Curve (ROC-AUC) with exact tie-handling."""
    s = _to_numpy(scores)
    t = _to_numpy(targets).reshape(-1).astype(int)
    if s.ndim > 1:
        s = s[:, 1] if s.shape[1] == 2 else s.reshape(-1)
    else:
        s = s.reshape(-1)

    if len(s) == 0:
        return 0.5

    pos_mask = (t == 1)
    neg_mask = (t != 1)
    pos_count = int(np.sum(pos_mask))
    neg_count = int(np.sum(neg_mask))

    if pos_count == 0 or neg_count == 0:
        return 0.5

    pos_scores = s[pos_mask]
    neg_scores = s[neg_mask]

    diff = pos_scores[:, None] - neg_scores[None, :]
    concordant = float(np.sum(diff > 0) + 0.5 * np.sum(diff == 0))
    return float(concordant / (pos_count * neg_count))


def mean_squared_error(preds, targets) -> float:
    """Computes Mean Squared Error (MSE) regression metric."""
    p = _to_numpy(preds)
    t = _to_numpy(targets)
    return float(np.mean((p - t) ** 2))


def mean_absolute_error(preds, targets) -> float:
    """Computes Mean Absolute Error (MAE) regression metric."""
    p = _to_numpy(preds)
    t = _to_numpy(targets)
    return float(np.mean(np.abs(p - t)))


def r2_score(preds, targets) -> float:
    """Computes R^2 (coefficient of determination) regression metric."""
    p = _to_numpy(preds)
    t = _to_numpy(targets)
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - (ss_res / ss_tot))


class MetricTracker:
    """Tracks running metric stats (avg, sum, count, min, max) for training & evaluation."""

    def __init__(self, name: str = "metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
        self.min = float("inf")
        self.max = float("-inf")

    def update(self, val, n: int = 1):
        if hasattr(val, "item"):
            val = val.item()
        val = float(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0
        if val < self.min:
            self.min = val
        if val > self.max:
            self.max = val

    def summary(self) -> dict:
        return {
            "name": self.name,
            "avg": self.avg,
            "last": self.val,
            "min": self.min,
            "max": self.max,
            "count": self.count,
        }

    def __repr__(self):
        return f"<MetricTracker '{self.name}': avg={self.avg:.4f}, min={self.min:.4f}, max={self.max:.4f}>"


__all__ = [
    "accuracy",
    "topk_accuracy",
    "confusion_matrix",
    "precision_recall_f1",
    "matthews_corrcoef",
    "cohen_kappa_score",
    "roc_auc_score",
    "mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    "MetricTracker",
]
