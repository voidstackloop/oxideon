"""Stochastic Weight Averaging (SWA) utilities (`AveragedModel`, `update_bn`, `SWALR`)."""

from typing import Any, Callable, Optional
import numpy as np

import oxideon as ox
from oxideon import nn


class AveragedModel(nn.Module):
    """Maintains moving averages of model parameters using an averaging function.

    Args:
        model: Model whose parameters will be averaged.
        avg_fn: Optional custom averaging function (defaults to running arithmetic average).
    """
    def __init__(self, model: nn.Module, avg_fn: Optional[Callable] = None):
        super().__init__()
        self.module = model
        self.register_buffer("n_averaged", ox.tensor(0, dtype=ox.int64))

        if avg_fn is None:
            def avg_fn(averaged_param, current_param, num_averaged):
                return averaged_param + (current_param - averaged_param) / float(num_averaged)
        self.avg_fn = avg_fn

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def update_parameters(self, model: nn.Module):
        """Updates the running average of the parameters with model's parameters."""
        self.n_averaged = self.n_averaged + 1
        n = int(self.n_averaged.to_numpy().item())
        with ox.no_grad():
            for p_swa, p_model in zip(self.module.parameters(), model.parameters()):
                new_val = self.avg_fn(p_swa, p_model, n)
                p_swa.copy_(new_val)


def update_bn(loader: Any, model: nn.Module, device: Optional[str] = None):
    """Updates BatchNorm running_mean and running_var with statistics from loader.

    Args:
        loader: Dataset loader producing input batches.
        model: Model containing BatchNorm layers to be updated.
        device: Optional target device name.
    """
    was_training = model.training
    model.train()
    with ox.no_grad():
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch
            model(x)
    model.train(was_training)


class SWALR:
    """Anneals the learning rate in SWA fashion."""
    def __init__(self, optimizer, swa_lr: float, anneal_epochs: int = 10, anneal_strategy: str = "cos"):
        self.optimizer = optimizer
        self.swa_lr = float(swa_lr)
        self.anneal_epochs = anneal_epochs
        self.anneal_strategy = anneal_strategy
        self.step_count = 0

    def step(self):
        self.step_count += 1
        for group in self.optimizer.param_groups:
            group["lr"] = self.swa_lr


__all__ = [
    "AveragedModel",
    "update_bn",
    "SWALR",
]
