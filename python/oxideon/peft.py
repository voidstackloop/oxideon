"""Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA) for Oxideon."""

from typing import Any, Dict, List, Optional, Sequence
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module, Parameter, Linear, Dropout


class LoRALinear(Module):
    """Linear layer wrapped with Low-Rank Adaptation (LoRA) adapter matrices:
    $$h = x W_0^T + \\frac{\\alpha}{r} (x A^T) B^T + b$$
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        bias: bool = True,
        base_linear: Optional[Any] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError(f"in_features and out_features must be >= 1, got in_features={in_features}, out_features={out_features}")
        if r < 0:
            raise ValueError(f"r must be >= 0, got {r}")
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / float(r) if r > 0 else 1.0
        self.merged = False
        self.dropout = Dropout(lora_dropout) if lora_dropout > 0.0 else None

        if base_linear is not None:
            self.weight = base_linear.weight
            self.bias = getattr(base_linear, "bias", None)
        else:
            self.weight = Parameter(_Tensor.randn([out_features, in_features]))
            self.bias = Parameter(_Tensor.zeros([out_features])) if bias else None

        # Freeze base weights
        if hasattr(self.weight, "requires_grad_"):
            self.weight.requires_grad_(False)
        elif hasattr(self.weight, "requires_grad"):
            self.weight.requires_grad = False

        if self.bias is not None:
            if hasattr(self.bias, "requires_grad_"):
                self.bias.requires_grad_(False)
            elif hasattr(self.bias, "requires_grad"):
                self.bias.requires_grad = False

        # LoRA low-rank adapters: A ~ N(0, 1/r), B = 0
        if r > 0:
            std_a = 1.0 / math.sqrt(r)
            rng = np.random.default_rng(seed) if seed is not None else np.random
            a_init = rng.normal(0.0, std_a, size=(r, in_features)).astype(np.float32)
            b_init = np.zeros((out_features, r), dtype=np.float32)

            self.lora_A = Parameter(_from_numpy(np.ascontiguousarray(a_init)), requires_grad=True)
            self.lora_B = Parameter(_from_numpy(np.ascontiguousarray(b_init)), requires_grad=True)
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x: _Tensor) -> _Tensor:
        result = x @ self.weight.t()
        if self.bias is not None:
            result = result + self.bias

        if self.r > 0 and not self.merged:
            x_in = self.dropout(x) if self.dropout is not None else x
            lora_out = (x_in @ self.lora_A.t()) @ self.lora_B.t()
            result = result + (lora_out * self.scaling)

        return result

    def merge(self):
        """Merges low-rank weights directly into base weight matrix: W = W + (alpha/r) * B @ A."""
        if not self.merged and self.r > 0:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            merged_w = self.weight + delta_w
            self.weight = Parameter(merged_w, requires_grad=False)
            self.merged = True

    def unmerge(self):
        """Subtracts merged low-rank adapter weights back from base weights."""
        if self.merged and self.r > 0:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            unmerged_w = self.weight - delta_w
            self.weight = Parameter(unmerged_w, requires_grad=False)
            self.merged = False


def apply_lora(
    model: Module,
    r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    target_modules: Optional[Sequence[str]] = None,
    seed: Optional[int] = None,
) -> Dict[str, int]:
    """Wraps target Linear layers with LoRALinear and freezes all non-LoRA parameters."""
    if r < 0:
        raise ValueError(f"r must be >= 0, got {r}")
    total_params = 0
    trainable_params = 0

    modules_dict = getattr(model, "_modules", {})
    for name, child in list(modules_dict.items()):
        is_target = target_modules is None or any(t == name or t in name for t in target_modules)

        if is_target and (isinstance(child, Linear) or child.__class__.__name__ == "Linear"):
            in_f = child.weight.shape[1] if len(child.weight.shape) > 1 else child.weight.shape[0]
            out_f = child.weight.shape[0]
            has_b = getattr(child, "bias", None) is not None

            lora_layer = LoRALinear(
                in_features=in_f,
                out_features=out_f,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=has_b,
                base_linear=child,
                seed=seed,
            )
            modules_dict[name] = lora_layer
            if hasattr(model, name):
                try:
                    setattr(model, name, lora_layer)
                except Exception:
                    pass
        else:
            apply_lora(
                child,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                seed=seed,
            )

    for p in model.parameters():
        total_params += p.numel()
        if getattr(p, "requires_grad", False):
            trainable_params += p.numel()

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_pct": (100.0 * trainable_params / total_params) if total_params > 0 else 0.0,
    }


def merge_lora(model: Module):
    """Recursively merges all LoRALinear layers across the entire model."""
    for child in getattr(model, "_modules", {}).values():
        if isinstance(child, LoRALinear):
            child.merge()
        else:
            merge_lora(child)


__all__ = [
    "LoRALinear",
    "apply_lora",
    "merge_lora",
]
