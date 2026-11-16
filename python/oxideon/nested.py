"""Nested Tensor data structures and jagged batch processing for Oxideon.

Enables efficient representation of variable-length sequence batches without zero-padding,
crucial for modern Transformer attention and ragged multimodal representations.
"""

from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np

from .oxideon import Tensor as _Tensor
from .primitives import tensor, zeros, full, Size, stack, cat


class NestedTensor:
    """Represents a jagged collection of tensors with varying lengths/shapes along sub-dimensions."""

    def __init__(self, tensors: Sequence[_Tensor]):
        if not tensors:
            raise ValueError("nested_tensor requires a non-empty sequence of tensors")
        self._tensors = [t.clone() if hasattr(t, "clone") else t for t in tensors]
        self.is_nested = True
        self.dtype = self._tensors[0].dtype
        self.device = self._tensors[0].device

    def __len__(self) -> int:
        return len(self._tensors)

    def __getitem__(self, idx: Union[int, slice]) -> Union[_Tensor, "NestedTensor"]:
        if isinstance(idx, slice):
            return NestedTensor(self._tensors[idx])
        return self._tensors[idx]

    def unbind(self, dim: int = 0) -> Tuple[_Tensor, ...]:
        if dim != 0:
            raise ValueError("nested_tensor only supports unbind along dim 0")
        return tuple(self._tensors)

    def values(self) -> _Tensor:
        """Flattens all jagged constituents into a single contiguous dense tensor along dim 0."""
        if len(self._tensors) == 1:
            return self._tensors[0]
        return cat(self._tensors, dim=0)

    def offsets(self) -> np.ndarray:
        """Returns cumulative element offsets for ragged partition boundaries."""
        lens = [t.shape[0] for t in self._tensors]
        return np.array([0] + list(np.cumsum(lens)), dtype=np.int64)

    def clone(self) -> "NestedTensor":
        """Returns a deep clone of the NestedTensor."""
        return NestedTensor([t.clone() for t in self._tensors])

    def to_padded_tensor(self, padding_value: float = 0.0, output_size: Optional[Tuple[int, ...]] = None) -> _Tensor:
        """Converts jagged nested tensor into a contiguous padded dense tensor preserving autograd."""
        shapes = [list(t.shape) for t in self._tensors]
        ndim = len(shapes[0])
        max_shape = [len(self._tensors)] + [max(s[d] for s in shapes) for d in range(ndim)]
        if output_size is not None:
            max_shape = list(output_size)

        has_autograd = any(getattr(t, "requires_grad", False) for t in self._tensors)

        if has_autograd:
            target_subshape = max_shape[1:]
            padded_items = []
            for t in self._tensors:
                cur = t
                for d, target_len in enumerate(target_subshape):
                    cur_len = cur.shape[d]
                    diff = target_len - cur_len
                    if diff > 0:
                        pad_shape = list(cur.shape)
                        pad_shape[d] = diff
                        pad_tensor = (
                            zeros(pad_shape, dtype=self.dtype, device=self.device)
                            if padding_value == 0.0
                            else full(pad_shape, float(padding_value), dtype=self.dtype, device=self.device)
                        )
                        cur = cat([cur, pad_tensor], dim=d)
                padded_items.append(cur)
            return stack(padded_items, dim=0)

        if padding_value != 0.0:
            out = full(max_shape, float(padding_value), dtype=self.dtype, device=self.device)
        else:
            out = zeros(max_shape, dtype=self.dtype, device=self.device)

        for i, t in enumerate(self._tensors):
            slc = [i] + [slice(0, dim_len) for dim_len in t.shape]
            out[tuple(slc)] = t
        return out

    @property
    def shape(self) -> List[Size]:
        return [t.shape for t in self._tensors]

    def __repr__(self) -> str:
        return f"nested_tensor([\n  " + ",\n  ".join(str(t) for t in self._tensors) + "\n])"


def nested_tensor(tensors: Sequence[_Tensor]) -> NestedTensor:
    """Constructs a NestedTensor from a sequence of tensors with varying sizes."""
    return NestedTensor(tensors)


import sys as _sys
_sys.modules["oxideon.nested"] = _sys.modules[__name__]

__all__ = ["nested_tensor", "NestedTensor"]
