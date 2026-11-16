"""High-level neural network training harness for Oxideon."""

from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import inspect
import time
import numpy as np

from .oxideon import Tensor as _Tensor
from .metrics import accuracy as _accuracy, MetricTracker as _MetricTracker
from .utils import clip_grad_norm_ as _clip_grad_norm_
from .logging import get_logger as _get_logger, is_debug as _is_debug, check_nan as _check_nan


def _step_takes_metric(step) -> bool:
    """True when a scheduler `step` requires a metric argument (plateau-style).

    Only REQUIRED positionals count: every epoch-style scheduler declares an
    optional ``epoch`` parameter, which must never be fed the validation loss.
    """
    try:
        params = inspect.signature(step).parameters.values()
    except (TypeError, ValueError):
        return False
    kinds = [p.kind for p in params]
    if any(k == inspect.Parameter.VAR_POSITIONAL for k in kinds):
        return True
    required_positional = [
        p
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and p.default is inspect.Parameter.empty
    ]
    return len(required_positional) >= 1


class TrainingArguments:
    """Configuration dataclass for Trainer hyperparameters."""

    def __init__(
        self,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
        max_grad_norm: Optional[float] = None,
        weight_decay: float = 0.0,
        debug: bool = False,
        log_interval: int = 10,
    ):
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.max_grad_norm = float(max_grad_norm) if max_grad_norm is not None else None
        self.weight_decay = float(weight_decay)
        self.debug = bool(debug)
        self.log_interval = int(log_interval)


class Trainer:
    """High-level training and evaluation harness for Oxideon models.

    Automates:
      - Training loop execution (`model.train()`, forward, loss, backward, step)
      - Validation and evaluation (`model.eval()`)
      - Learning rate scheduler stepping
      - Gradient norm clipping
      - Callback dispatch (`on_train_begin`, `on_epoch_begin`, `on_epoch_end`, etc.)
      - History metric recording
    """

    def __init__(
        self,
        model: Any,
        optimizer: Any,
        criterion: Any,
        lr_scheduler: Optional[Any] = None,
        callbacks: Optional[Sequence[Any]] = None,
        max_grad_norm: Optional[float] = None,
        debug: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.callbacks = list(callbacks) if callbacks else []
        self.max_grad_norm = max_grad_norm
        self.debug = debug
        self.stop_training = False
        self.logger = _get_logger("oxideon.trainer")

    def _dispatch(self, hook_name: str, *args, **kwargs):
        for cb in self.callbacks:
            if hasattr(cb, hook_name):
                getattr(cb, hook_name)(*args, **kwargs)

    def fit(
        self,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        epochs: int = 10,
        verbose: bool = True,
        debug: Optional[bool] = None,
        log_interval: int = 10,
    ) -> Dict[str, List[float]]:
        """Trains the model for the specified number of epochs.

        Args:
            train_loader: DataLoader or iterable yielding training batches.
            val_loader: Optional DataLoader for validation.
            epochs: Total epochs to train.
            verbose: If True, prints progress and summaries.
            debug: If True, outputs detailed tensor shapes, gradient norms, and NaN audits.
            log_interval: Step frequency for real-time progress logging.
        """
        is_dbg = self.debug if debug is None else debug
        is_dbg = is_dbg or _is_debug()

        if is_dbg:
            self.logger.debug(f"[Trainer Setup] Starting fit for {epochs} epochs (debug=True)")
            if hasattr(self.model, "parameters"):
                num_p = sum(p.to_numpy().size for p in self.model.parameters())
                self.logger.debug(f"[Model Parameter Count] Total parameters: {num_p:,}")

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
        }
        if val_loader is not None:
            history["val_loss"] = []
            history["val_acc"] = []

        total_batches = len(train_loader) if hasattr(train_loader, "__len__") else None

        self._dispatch("on_train_begin", self)

        for epoch in range(1, epochs + 1):
            if self.stop_training:
                if verbose:
                    self.logger.warning(f"Early stopping triggered at epoch {epoch}")
                break

            self._dispatch("on_epoch_begin", epoch, self)
            t0 = time.perf_counter()

            # --- Training Phase ---
            if hasattr(self.model, "train"):
                self.model.train()

            loss_tracker = _MetricTracker("train_loss")
            correct_samples = 0
            total_samples = 0

            for batch_idx, batch in enumerate(train_loader):
                self._dispatch("on_batch_begin", batch_idx, self)
                t_batch_start = time.perf_counter()

                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs, targets = batch, None

                if is_dbg and batch_idx == 0:
                    in_shape = inputs.shape if hasattr(inputs, "shape") else type(inputs)
                    tgt_shape = targets.shape if hasattr(targets, "shape") and targets is not None else None
                    self.logger.debug(f"[Batch 0 Audit] inputs={in_shape}, targets={tgt_shape}")

                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                if is_dbg and batch_idx == 0:
                    out_shape = outputs.shape if hasattr(outputs, "shape") else type(outputs)
                    self.logger.debug(f"[Forward Output 0] outputs={out_shape}")

                loss = self.criterion(outputs, targets)
                _check_nan(loss, name="criterion_loss")
                loss.backward()

                # In debug mode, inspect gradients
                if is_dbg:
                    params = list(self.model.parameters()) if hasattr(self.model, "parameters") else []
                    grad_norms = []
                    for p in params:
                        g_call = getattr(p, "grad", None)
                        g = g_call() if callable(g_call) else g_call
                        if g is not None:
                            grad_norms.append(float(np.linalg.norm(g.to_numpy())))
                    if grad_norms and (batch_idx == 0 or (total_batches and batch_idx == total_batches - 1)):
                        total_g_norm = np.sqrt(sum(gn ** 2 for gn in grad_norms))
                        self.logger.debug(f"[Grad Norm Audit] batch {batch_idx}: total_norm={total_g_norm:.4e}")

                if self.max_grad_norm is not None:
                    params = self.model.parameters() if hasattr(self.model, "parameters") else []
                    _clip_grad_norm_(params, self.max_grad_norm)

                self.optimizer.step()

                loss_val = float(loss.item()) if hasattr(loss, "item") else float(loss)
                b_size = inputs.shape[0] if hasattr(inputs, "shape") else 1
                loss_tracker.update(loss_val, n=b_size)

                batch_acc = None
                if targets is not None:
                    batch_acc = _accuracy(outputs, targets)
                    # Float accumulation: per-batch int(round()) drifts the
                    # epoch accuracy (up to ~0.5 sample per batch).
                    correct_samples += batch_acc * b_size
                    total_samples += b_size

                t_batch = max(time.perf_counter() - t_batch_start, 1e-6)
                samples_per_sec = b_size / t_batch

                # Step-level progress logging
                if verbose and total_batches and (batch_idx % log_interval == 0 or batch_idx == total_batches - 1):
                    metrics_now = {"acc": batch_acc} if batch_acc is not None else None
                    self.logger.log_training_step(
                        epoch=epoch,
                        step=batch_idx + 1,
                        total_steps=total_batches,
                        loss=loss_val,
                        metrics=metrics_now,
                        samples_per_sec=samples_per_sec,
                    )

                self._dispatch("on_batch_end", batch_idx, loss_val, self)

            epoch_train_loss = loss_tracker.avg
            epoch_train_acc = (correct_samples / total_samples) if total_samples > 0 else 0.0

            history["train_loss"].append(epoch_train_loss)
            history["train_acc"].append(epoch_train_acc)

            # --- Validation Phase ---
            epoch_val_loss = None
            epoch_val_acc = None

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                epoch_val_loss = val_metrics["loss"]
                epoch_val_acc = val_metrics.get("accuracy", 0.0)

                history["val_loss"].append(epoch_val_loss)
                history["val_acc"].append(epoch_val_acc)

            # Step LR scheduler if present. The call shape is decided by
            # signature inspection: blindly catching TypeError would also mask
            # real bugs raised inside a scheduler's own step().
            if self.lr_scheduler is not None:
                if hasattr(self.lr_scheduler, "step"):
                    step = self.lr_scheduler.step
                    if epoch_val_loss is not None and _step_takes_metric(step):
                        step(epoch_val_loss)
                    else:
                        step()

            elapsed = time.perf_counter() - t0
            throughput = total_samples / max(elapsed, 1e-6) if total_samples > 0 else None
            metrics_dict = {
                "epoch": epoch,
                "train_loss": epoch_train_loss,
                "train_acc": epoch_train_acc,
                "val_loss": epoch_val_loss,
                "val_acc": epoch_val_acc,
                "elapsed": elapsed,
                "samples_per_sec": throughput,
            }

            self._dispatch("on_epoch_end", epoch, metrics_dict, self)

            if verbose:
                self.logger.log_epoch_summary(
                    epoch=epoch,
                    total_epochs=epochs,
                    train_loss=epoch_train_loss,
                    train_acc=epoch_train_acc,
                    val_loss=epoch_val_loss,
                    val_acc=epoch_val_acc,
                    elapsed_sec=elapsed,
                    samples_per_sec=throughput,
                )

        self._dispatch("on_train_end", self)
        return history

    def evaluate(self, dataloader: Any) -> Dict[str, float]:
        """Evaluates the model on a dataset or DataLoader without gradient tracking."""
        if hasattr(self.model, "eval"):
            self.model.eval()

        loss_tracker = _MetricTracker("eval_loss")
        correct_samples = 0
        total_samples = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs, targets = batch[0], batch[1]
            else:
                inputs, targets = batch, None

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            loss_val = float(loss.item()) if hasattr(loss, "item") else float(loss)
            b_size = inputs.shape[0] if hasattr(inputs, "shape") else 1
            loss_tracker.update(loss_val, n=b_size)

            if targets is not None:
                acc = _accuracy(outputs, targets)
                correct_samples += acc * b_size
                total_samples += b_size

        return {
            "loss": loss_tracker.avg,
            "accuracy": (correct_samples / total_samples) if total_samples > 0 else 0.0,
        }

    def save_checkpoint(self, filepath: str) -> None:
        """Saves current model weights, optimizer state, and training metadata."""
        from .serialization import save
        ckpt = {
            "model_state_dict": self.model.state_dict() if hasattr(self.model, "state_dict") else None,
            "optimizer_state_dict": self.optimizer.state_dict() if hasattr(self.optimizer, "state_dict") else None,
        }
        save(ckpt, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """Loads model weights and optimizer state from a checkpoint file."""
        from .serialization import load
        ckpt = load(filepath, weights_only=False)
        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt and ckpt["model_state_dict"] and hasattr(self.model, "load_state_dict"):
                self.model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] and hasattr(self.optimizer, "load_state_dict"):
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    def predict(self, dataloader_or_inputs: Any) -> _Tensor:
        """Runs model inference on inputs or a dataloader, returning stacked predictions."""
        if hasattr(self.model, "eval"):
            self.model.eval()

        if hasattr(dataloader_or_inputs, "__iter__") and not isinstance(dataloader_or_inputs, _Tensor):
            preds = []
            for batch in dataloader_or_inputs:
                inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                out = self.model(inputs)
                preds.append(out)
            if not preds:
                raise ValueError("Trainer.predict received an empty dataloader (no batches)")
            from .oxideon import cat
            return cat(preds, dim=0)
        else:
            return self.model(dataloader_or_inputs)


__all__ = [
    "Trainer",
    "TrainingArguments",
]
