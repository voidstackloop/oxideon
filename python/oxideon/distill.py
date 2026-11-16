"""Knowledge Distillation and Teacher-Student model compression for Oxideon.

Enables transferring knowledge from large teacher networks into compact student networks:
  - DistillationLoss: Hinton et al. (2015) soft-target dark knowledge + hard label cross-entropy
  - FeatureMatchingLoss: FitNets (Romero et al., 2014) intermediate representation matching
  - DistillationTrainer: Coordinated training loop for student-teacher distillation
"""

from typing import Tuple, Optional, Any, Callable
import numpy as np

from .oxideon import (
    Tensor as _Tensor,
    from_numpy as _from_numpy,
    cross_entropy,
)
from .functional import softmax, log_softmax
from .nn import Module


class DistillationLoss(Module):
    """Hinton-style Knowledge Distillation loss."""

    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: _Tensor,
        teacher_logits: _Tensor,
        targets: _Tensor,
    ) -> _Tensor:
        """Computes combined soft distillation loss + hard label cross entropy."""
        T = self.temperature

        # Soft probabilities and log-probabilities
        p_t = softmax(teacher_logits / T, dim=-1).detach()
        log_p_s = log_softmax(student_logits / T, dim=-1)

        # Cross entropy of soft targets scaled by T^2
        soft_loss = -(p_t * log_p_s).sum(dim=-1).mean() * (T * T)

        # Hard loss
        hard_loss = cross_entropy(student_logits, targets)

        total_loss = soft_loss * self.alpha + hard_loss * (1.0 - self.alpha)
        return total_loss


class FeatureMatchingLoss(Module):
    """FitNets intermediate representation matching loss."""

    def __init__(self, loss_type: str = "mse"):
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self, student_features: _Tensor, teacher_features: _Tensor
    ) -> _Tensor:
        if student_features.shape != teacher_features.shape:
            raise ValueError(f"Shape mismatch: student_features {student_features.shape} != teacher_features {teacher_features.shape}")
        if self.loss_type == "mse":
            diff = student_features - teacher_features.detach()
            loss = (diff * diff).mean()
        else:
            t_det = teacher_features.detach()
            s_norm = ((student_features * student_features).sum(dim=-1, keepdim=True) + 1e-12).sqrt()
            t_norm = ((t_det * t_det).sum(dim=-1, keepdim=True) + 1e-12).sqrt()
            cos_sim = ((student_features / s_norm) * (t_det / t_norm)).sum(dim=-1).mean()
            loss = 1.0 - cos_sim

        return loss


class DistillationTrainer:
    """Convenience coordinator for training student models with a frozen teacher."""

    def __init__(
        self,
        student: Module,
        teacher: Module,
        optimizer: Any,
        distill_loss: Optional[DistillationLoss] = None,
    ):
        self.student = student
        self.teacher = teacher
        self.optimizer = optimizer
        self.distill_loss = distill_loss or DistillationLoss()

        # Freeze teacher
        self.teacher.eval()
        for p in self.teacher.parameters():
            if hasattr(p, "requires_grad_"):
                p.requires_grad_(False)

    def train_step(self, x: _Tensor, y: _Tensor) -> float:
        self.student.train()
        self.optimizer.zero_grad()

        student_logits = self.student(x)
        teacher_logits = self.teacher(x)

        loss = self.distill_loss(student_logits, teacher_logits, y)
        loss.backward()
        self.optimizer.step()

        return float(loss.to_numpy().reshape(-1)[0])


__all__ = [
    "DistillationLoss",
    "FeatureMatchingLoss",
    "DistillationTrainer",
]
