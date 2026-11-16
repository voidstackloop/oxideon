"""Training Callbacks for Oxideon.

Provides `EarlyStopping`, `ModelCheckpoint`, `CSVLogger`, `LearningRateLogger`,
`TerminateOnNaN`, and `ReduceLROnPlateauCallback`.
"""

import csv
import math
import os
from typing import Any, Dict, List, Optional

from .serialization import save_file


class Callback:
    """Base class for all training callbacks."""

    def on_train_begin(self, trainer: Any):
        pass

    def on_train_end(self, trainer: Any):
        pass

    def on_epoch_begin(self, epoch: int, trainer: Any):
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any], trainer: Any):
        pass

    def on_batch_begin(self, batch_idx: int, trainer: Any):
        pass

    def on_batch_end(self, batch_idx: int, loss: float, trainer: Any):
        pass


class EarlyStopping(Callback):
    """Early stops training when monitored quantity stops improving."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        mode: str = "min",
        monitor: str = "val_loss",
        restore_best_weights: bool = True,
    ):
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.monitor = monitor
        self.restore_best_weights = restore_best_weights

        self.best_score = math.inf if mode == "min" else -math.inf
        self.counter = 0
        self.early_stop = False
        self.best_state = None

    def __call__(self, current_metric: float, model: Optional[Any] = None) -> bool:
        """Direct call interface (manual training loop)."""
        current = float(current_metric)
        is_better = (
            (current < self.best_score - self.min_delta)
            if self.mode == "min"
            else (current > self.best_score + self.min_delta)
        )

        if is_better:
            self.best_score = current
            self.counter = 0
            if self.restore_best_weights and model is not None and hasattr(model, "state_dict"):
                self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and model is not None and self.best_state is not None:
                    model.load_state_dict(self.best_state)

        return self.early_stop

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any], trainer: Any):
        """Trainer hook interface."""
        val = metrics.get(self.monitor)
        if val is None:
            return
        stop = self(val, model=trainer.model)
        if stop:
            trainer.stop_training = True


class ModelCheckpoint(Callback):
    """Saves the model state_dict after every epoch if the monitored metric improved."""

    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        save_best_only: bool = True,
        mode: str = "min",
    ):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.best_score = math.inf if mode == "min" else -math.inf

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any], trainer: Any):
        current = metrics.get(self.monitor)
        if current is None:
            return

        is_better = (current < self.best_score) if self.mode == "min" else (current > self.best_score)
        if is_better or not self.save_best_only:
            if is_better:
                self.best_score = current

            # Save state dict or parameters
            if hasattr(trainer.model, "state_dict"):
                state = trainer.model.state_dict()
                save_file(state, self.filepath)
            elif hasattr(trainer.model, "parameters"):
                params = list(trainer.model.parameters())
                state = {f"param_{i}": p for i, p in enumerate(params)}
                save_file(state, self.filepath)


class CSVLogger(Callback):
    """Streams epoch training results into a CSV file."""

    def __init__(self, filename: str, separator: str = ","):
        self.filename = filename
        self.separator = separator
        self.file = None
        self.writer = None
        self.keys = None

    def on_train_begin(self, trainer: Any):
        os.makedirs(os.path.dirname(os.path.abspath(self.filename)) or ".", exist_ok=True)
        self.file = open(self.filename, mode="w", newline="", encoding="utf-8")
        self.writer = None

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any], trainer: Any):
        if self.file is None:
            self.file = open(self.filename, mode="w", newline="", encoding="utf-8")

        if self.writer is None:
            self.keys = list(metrics.keys())
            self.writer = csv.DictWriter(self.file, fieldnames=self.keys, delimiter=self.separator)
            self.writer.writeheader()

        self.writer.writerow({k: metrics.get(k, "") for k in self.keys})
        self.file.flush()

    def on_train_end(self, trainer: Any):
        if self.file is not None and not self.file.closed:
            self.file.close()


class LearningRateLogger(Callback):
    """Tracks learning rates across epochs."""

    def __init__(self):
        self.lrs: List[float] = []

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any], trainer: Any):
        lr = None
        if hasattr(trainer.optimizer, "lr"):
            lr = trainer.optimizer.lr
        elif hasattr(trainer.optimizer, "param_groups"):
            lr = trainer.optimizer.param_groups[0].get("lr")
        if lr is not None:
            self.lrs.append(float(lr))


class TerminateOnNaN(Callback):
    """Terminates training if loss is NaN or infinite."""

    def on_batch_end(self, batch_idx: int, loss: float, trainer: Any):
        if math.isnan(loss) or math.isinf(loss):
            print(f"TerminateOnNaN: Loss is {loss} at batch {batch_idx}. Halting training.")
            trainer.stop_training = True


class ReduceLROnPlateauCallback(Callback):
    """Reduces optimizer learning rate when a monitored metric stops improving."""

    def __init__(
        self,
        monitor: str = "val_loss",
        factor: float = 0.1,
        patience: int = 5,
        min_delta: float = 1e-4,
        mode: str = "min",
        min_lr: float = 0.0,
    ):
        if not (0.0 < float(factor) < 1.0):
            raise ValueError(f"factor must be in (0, 1), got {factor}")
        if int(patience) < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        self.monitor = monitor
        self.factor = float(factor)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.min_lr = float(min_lr)

        self.best_score = math.inf if mode == "min" else -math.inf
        self.counter = 0

    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any], trainer: Any):
        val = metrics.get(self.monitor)
        if val is None:
            return
        current = float(val)
        is_better = (
            (current < self.best_score - self.min_delta)
            if self.mode == "min"
            else (current > self.best_score + self.min_delta)
        )
        if is_better:
            self.best_score = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.counter = 0
                opt = getattr(trainer, "optimizer", None)
                if opt is not None:
                    if hasattr(opt, "lr"):
                        opt.lr = max(self.min_lr, float(opt.lr) * self.factor)
                    if hasattr(opt, "param_groups"):
                        for group in opt.param_groups:
                            if "lr" in group:
                                group["lr"] = max(self.min_lr, float(group["lr"]) * self.factor)


__all__ = [
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "CSVLogger",
    "LearningRateLogger",
    "TerminateOnNaN",
    "ReduceLROnPlateauCallback",
]
