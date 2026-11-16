"""Packed sequence utilities for recurrent neural networks.

Mirrors torch.nn.utils.rnn:
- PackedSequence
- pack_padded_sequence
- pad_packed_sequence
- pad_sequence
- pack_sequence
"""

from typing import List, Optional, Sequence, Tuple, Union
import numpy as np

from . import Tensor, from_numpy


class PackedSequence:
    """Holds the data and list of :attr:`batch_sizes` of a packed sequence."""

    def __init__(
        self,
        data: Tensor,
        batch_sizes: Optional[Union[Tensor, Sequence[int]]] = None,
        sorted_indices: Optional[Tensor] = None,
        unsorted_indices: Optional[Tensor] = None,
    ):
        self.data = data
        self.batch_sizes = batch_sizes
        self.sorted_indices = sorted_indices
        self.unsorted_indices = unsorted_indices

    def to_numpy(self) -> np.ndarray:
        return self.data.to_numpy()

    def __repr__(self) -> str:
        return f"PackedSequence(data={self.data}, batch_sizes={self.batch_sizes})"


def pack_padded_sequence(
    input: Tensor,
    lengths: Union[Tensor, Sequence[int]],
    batch_first: bool = False,
    enforce_sorted: bool = True,
) -> PackedSequence:
    """Packs a Tensor containing padded sequences of variable length."""
    if isinstance(lengths, Tensor):
        len_list = [int(x) for x in lengths.to_numpy().flatten()]
    else:
        len_list = [int(x) for x in lengths]

    if enforce_sorted:
        for i in range(len(len_list) - 1):
            if len_list[i] < len_list[i + 1]:
                raise RuntimeError(
                    f"`lengths` must be in decreasing order when `enforce_sorted` is True, got {len_list}"
                )

    data_np = input.to_numpy()
    if batch_first:
        # [B, T, *D] -> transpose to [T, B, *D]
        axes = (1, 0) + tuple(range(2, data_np.ndim))
        data_np = np.transpose(data_np, axes)

    max_len, batch_size = data_np.shape[0], data_np.shape[1]
    slices = []
    batch_sizes = []
    for step in range(max_len):
        active = sum(1 for l in len_list if l > step)
        if active == 0:
            break
        batch_sizes.append(active)
        slices.append(data_np[step, :active])

    if slices:
        packed_data = np.concatenate(slices, axis=0)
    else:
        packed_data = np.empty((0,) + data_np.shape[2:], dtype=data_np.dtype)

    out_tensor = from_numpy(np.ascontiguousarray(packed_data))
    if input.requires_grad:
        out_tensor.requires_grad_(True)
    return PackedSequence(out_tensor, batch_sizes=batch_sizes)


def pad_packed_sequence(
    sequence: PackedSequence,
    batch_first: bool = False,
    padding_value: float = 0.0,
    total_length: Optional[int] = None,
) -> Tuple[Tensor, List[int]]:
    """Pads a packed batch of variable length sequences."""
    data_np = sequence.data.to_numpy()
    batch_sizes = sequence.batch_sizes
    if batch_sizes is None:
        raise ValueError("sequence.batch_sizes cannot be None in pad_packed_sequence")

    if isinstance(batch_sizes, Tensor):
        bs_list = [int(x) for x in batch_sizes.to_numpy().flatten()]
    else:
        bs_list = [int(x) for x in batch_sizes]

    max_len = len(bs_list) if total_length is None else max(len(bs_list), total_length)
    batch_size = bs_list[0] if bs_list else 0
    feature_shape = data_np.shape[1:]

    padded_np = np.full((max_len, batch_size) + feature_shape, padding_value, dtype=data_np.dtype)
    lengths = [0] * batch_size

    offset = 0
    for step, bs in enumerate(bs_list):
        padded_np[step, :bs] = data_np[offset : offset + bs]
        offset += bs
        for b in range(bs):
            lengths[b] = step + 1

    if batch_first:
        axes = (1, 0) + tuple(range(2, padded_np.ndim))
        padded_np = np.transpose(padded_np, axes)

    out_tensor = from_numpy(np.ascontiguousarray(padded_np))
    if sequence.data.requires_grad:
        out_tensor.requires_grad_(True)
    return out_tensor, lengths


def pad_sequence(
    sequences: Sequence[Tensor],
    batch_first: bool = False,
    padding_value: float = 0.0,
) -> Tensor:
    """Pad a list of variable length Tensors with `padding_value`."""
    lengths = [s.shape[0] for s in sequences]
    max_len = max(lengths) if lengths else 0
    batch_size = len(sequences)
    if not sequences:
        return from_numpy(np.empty((0, 0), dtype=np.float32))

    sample = sequences[0].to_numpy()
    out = np.full((max_len, batch_size) + sample.shape[1:], padding_value, dtype=sample.dtype)
    for i, s in enumerate(sequences):
        s_np = s.to_numpy()
        out[: s_np.shape[0], i] = s_np

    if batch_first:
        axes = (1, 0) + tuple(range(2, out.ndim))
        out = np.transpose(out, axes)
    return from_numpy(np.ascontiguousarray(out))


def pack_sequence(
    sequences: Sequence[Tensor],
    enforce_sorted: bool = True,
) -> PackedSequence:
    """Packs a list of variable length Tensors."""
    lengths = [s.shape[0] for s in sequences]
    padded = pad_sequence(sequences)
    return pack_padded_sequence(padded, lengths, batch_first=False, enforce_sorted=enforce_sorted)
