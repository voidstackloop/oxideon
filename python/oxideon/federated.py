"""
Federated Learning (FL) orchestration, Byzantine-robust aggregation, and client synchronization for Oxideon.
Includes FedAvg (McMahan et al.), FedProx (Li et al.), Coordinate-Wise Median (`fed_median`),
and Coordinate-Wise Trimmed Mean (`fed_trimmed_mean`).
"""
from typing import Any, Dict, List, Optional
import numpy as np
from .oxideon import Tensor, from_numpy as _from_numpy
from .nn import Module, Parameter


def _iter_named_parameters(model: Any):
    if hasattr(model, "named_parameters") and callable(model.named_parameters):
        yield from model.named_parameters()
    elif hasattr(model, "parameters") and callable(model.parameters):
        for i, p in enumerate(model.parameters()):
            name = "weight" if i == 0 else "bias" if i == 1 else f"param_{i}"
            yield name, p


def _extract_model_weights(model: Any) -> Dict[str, np.ndarray]:
    weights = {}
    for name, param in _iter_named_parameters(model):
        weights[name] = param.to_numpy().copy()
    return weights


def _load_model_weights(model: Any, weights: Dict[str, np.ndarray]):
    for name, arr in weights.items():
        parts = name.split(".")
        curr = model
        for p in parts[:-1]:
            curr = getattr(curr, p)
        leaf = parts[-1]
        new_t = _from_numpy(np.asarray(arr, dtype=np.float32).copy())
        existing = getattr(curr, leaf, None)
        if isinstance(existing, Tensor) and hasattr(existing, "copy_"):
            existing.copy_(new_t)
        else:
            setattr(curr, leaf, Parameter(new_t))


def fed_avg(client_weights: List[Dict[str, np.ndarray]], sample_counts: List[int]) -> Dict[str, np.ndarray]:
    """
    Standard Federated Averaging (FedAvg) aggregation.
    W_global = sum_k (n_k / N) * W_k
    """
    if not client_weights:
        raise ValueError("client_weights cannot be empty.")
    if len(client_weights) != len(sample_counts):
        raise ValueError(f"Length mismatch: {len(client_weights)} client updates vs {len(sample_counts)} sample counts.")
    if any(c < 0 for c in sample_counts):
        raise ValueError("sample_counts must be >= 0.")
    total_samples = sum(sample_counts)
    if total_samples == 0:
        raise ValueError("Total sample count across clients cannot be zero.")

    aggregated = {}
    param_names = client_weights[0].keys()

    for name in param_names:
        w_sum = np.zeros_like(client_weights[0][name], dtype=np.float64)
        for cw, count in zip(client_weights, sample_counts):
            weight_factor = count / total_samples
            w_sum += weight_factor * cw[name]
        aggregated[name] = w_sum.astype(client_weights[0][name].dtype)

    return aggregated


def fed_median(client_weights: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Byzantine-robust Coordinate-Wise Median aggregation (Yin et al., ICML 2018)."""
    if not client_weights:
        raise ValueError("client_weights cannot be empty.")
    aggregated = {}
    for name in client_weights[0].keys():
        stacked = np.stack([cw[name] for cw in client_weights], axis=0)
        aggregated[name] = np.median(stacked, axis=0).astype(client_weights[0][name].dtype)
    return aggregated


def fed_trimmed_mean(
    client_weights: List[Dict[str, np.ndarray]],
    trim_ratio: float = 0.2,
) -> Dict[str, np.ndarray]:
    """Byzantine-robust Coordinate-Wise Trimmed Mean aggregation (Yin et al., ICML 2018)."""
    if not client_weights:
        raise ValueError("client_weights cannot be empty.")
    if not (0.0 <= trim_ratio < 0.5):
        raise ValueError(f"trim_ratio must be in [0.0, 0.5), got {trim_ratio}")
    k = len(client_weights)
    trim_k = int(k * trim_ratio)
    aggregated = {}
    for name in client_weights[0].keys():
        stacked = np.sort(np.stack([cw[name] for cw in client_weights], axis=0), axis=0)
        if trim_k > 0 and (k - 2 * trim_k) > 0:
            sliced = stacked[trim_k : k - trim_k]
        else:
            sliced = stacked
        aggregated[name] = np.mean(sliced, axis=0).astype(client_weights[0][name].dtype)
    return aggregated


def compute_proximal_loss(
    current_weights: Dict[str, np.ndarray],
    global_weights: Dict[str, np.ndarray],
    mu: float = 0.01,
) -> float:
    """Computes the FedProx proximal penalty: (mu / 2) * sum_p ||w_p - w_global,p||^2."""
    if mu < 0.0:
        raise ValueError(f"FedProx mu must be non-negative (>= 0), got {mu}")
    penalty = 0.0
    for name in current_weights:
        if name in global_weights:
            diff = current_weights[name] - global_weights[name]
            penalty += np.sum(diff ** 2)
    return float(0.5 * mu * penalty)


class FederatedServer:
    """Orchestrates federated aggregation rounds across distributed clients."""
    def __init__(self, initial_model: Any):
        self.global_weights = _extract_model_weights(initial_model)
        self.round = 0

    def get_global_weights(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.global_weights.items()}

    def aggregate(
        self,
        client_updates: List[Dict[str, np.ndarray]],
        sample_counts: Optional[List[int]] = None,
        method: str = "fedavg",
        trim_ratio: float = 0.2,
    ):
        """Aggregates client parameters using 'fedavg', 'median', or 'trimmed_mean'."""
        m = method.strip().lower()
        if m == "fedavg":
            if sample_counts is None:
                sample_counts = [1] * len(client_updates)
            self.global_weights = fed_avg(client_updates, sample_counts)
        elif m == "median":
            self.global_weights = fed_median(client_updates)
        elif m in ("trimmed_mean", "trimmed"):
            self.global_weights = fed_trimmed_mean(client_updates, trim_ratio=trim_ratio)
        else:
            raise ValueError(f"Unknown aggregation method {method!r}")
        self.round += 1

    def sync_model(self, model: Any):
        """Loads updated global parameters into a local model instance in place."""
        _load_model_weights(model, self.global_weights)


class FederatedClient:
    """Simulates a local edge device/node in a federated learning network."""
    def __init__(self, client_id: str, model: Any):
        self.client_id = client_id
        self.model = model

    def set_weights(self, weights: Dict[str, np.ndarray]):
        _load_model_weights(self.model, weights)

    def get_weights(self) -> Dict[str, np.ndarray]:
        return _extract_model_weights(self.model)
