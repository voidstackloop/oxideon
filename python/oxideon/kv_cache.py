"""Paged and contiguous Key-Value Cache systems for high-throughput LLM inference.

Provides:
  - PagedKVCache: Non-contiguous block-based memory allocator (vLLM style PagedAttention) with Copy-on-Write fork_sequence.
  - StandardKVCache: Contiguous sliding-window and append-only multi-head KV cache.
  - StreamingSinkCache: StreamingLLM attention sinks + recent window.
  - ScoredEvictionCache: H2O / TOVA / SnapKV score-guided eviction.
"""

from typing import List, Tuple, Optional, Dict
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import _check_non_negative, _check_positive, _check_repeats


class PagedKVCache:
    """Paged Key-Value memory manager inspired by vLLM's PagedAttention.

    Partitions KV memory into fixed-size physical blocks of tokens, eliminating
    memory fragmentation and allowing dynamic memory allocation and Copy-on-Write sequence sharing.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
    ):
        num_blocks = _check_repeats("num_blocks", num_blocks)
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        self._key_cache = np.zeros((num_blocks, block_size, num_heads, head_dim), dtype=np.float32)
        self._val_cache = np.zeros((num_blocks, block_size, num_heads, head_dim), dtype=np.float32)

        self._free_blocks = list(range(num_blocks))
        self._ref_counts: Dict[int, int] = {b: 0 for b in range(num_blocks)}

    @property
    def free_blocks_count(self) -> int:
        return len(self._free_blocks)

    def allocate(self, num_tokens: int) -> List[int]:
        """Allocates necessary physical blocks to store `num_tokens`."""
        needed_blocks = (num_tokens + self.block_size - 1) // self.block_size
        if needed_blocks > len(self._free_blocks):
            raise MemoryError(
                f"Out of KV cache blocks: needed {needed_blocks}, available {len(self._free_blocks)}"
            )
        allocated = []
        for _ in range(needed_blocks):
            b = self._free_blocks.pop(0)
            self._ref_counts[b] = 1
            allocated.append(b)
        return allocated

    def fork_sequence(self, block_table: List[int]) -> List[int]:
        """Creates a Copy-on-Write fork of `block_table`, incrementing physical block reference counts."""
        for b in block_table:
            self._ref_counts[b] = max(1, self._ref_counts.get(b, 1)) + 1
        return list(block_table)

    def append(self, block_table: List[int], current_len: int, key: _Tensor, value: _Tensor) -> List[int]:
        """Appends a single token key and value [num_heads, head_dim] into the sequence's blocks."""
        target_block_idx = current_len // self.block_size
        offset_in_block = current_len % self.block_size

        if target_block_idx >= len(block_table):
            if not self._free_blocks:
                raise MemoryError("Cannot append token: No free physical KV blocks left")
            b = self._free_blocks.pop(0)
            self._ref_counts[b] = 1
            block_table.append(b)
        else:
            phys_block = block_table[target_block_idx]
            if self._ref_counts.get(phys_block, 1) > 1:
                # Copy-on-Write: allocate a fresh block and copy existing entries
                if not self._free_blocks:
                    raise MemoryError("Cannot COW-copy block: No free physical KV blocks left")
                new_block = self._free_blocks.pop(0)
                self._key_cache[new_block] = self._key_cache[phys_block]
                self._val_cache[new_block] = self._val_cache[phys_block]
                self._ref_counts[phys_block] -= 1
                self._ref_counts[new_block] = 1
                block_table[target_block_idx] = new_block

        phys_block = block_table[target_block_idx]
        k_np = key.to_numpy()
        v_np = value.to_numpy()

        self._key_cache[phys_block, offset_in_block] = k_np
        self._val_cache[phys_block, offset_in_block] = v_np
        return block_table

    def read_sequence(self, block_table: List[int], seq_len: int) -> Tuple[_Tensor, _Tensor]:
        """Reconstructs contiguous key and value tensors [seq_len, num_heads, head_dim] for a sequence."""
        if seq_len == 0:
            empty = np.zeros((0, self.num_heads, self.head_dim), dtype=np.float32)
            return _from_numpy(empty), _from_numpy(empty)

        needed_blocks = (seq_len + self.block_size - 1) // self.block_size
        active_blocks = block_table[:needed_blocks]

        k_blocks = self._key_cache[active_blocks]
        v_blocks = self._val_cache[active_blocks]

        k_out = k_blocks.reshape(-1, self.num_heads, self.head_dim)[:seq_len]
        v_out = v_blocks.reshape(-1, self.num_heads, self.head_dim)[:seq_len]

        return _from_numpy(np.ascontiguousarray(k_out)), _from_numpy(np.ascontiguousarray(v_out))

    def paged_attention(
        self,
        query: _Tensor,
        block_table: List[int],
        seq_len: int,
        scale: Optional[float] = None,
    ) -> _Tensor:
        """Computes multi-head attention directly over paged KV memory for a query token [num_heads, head_dim].

        Preserves Tensor autograd w.r.t. `query`.
        """
        if seq_len == 0:
            return query * 0.0
        if scale is None:
            scale = float(1.0 / np.sqrt(self.head_dim))

        k_t, v_t = self.read_sequence(block_table, seq_len)
        # query: [H, D] -> [H, 1, D]
        # k_t: [S, H, D] -> transpose(0, 1) -> [H, S, D] -> transpose(1, 2) -> [H, D, S]
        # v_t: [S, H, D] -> transpose(0, 1) -> [H, S, D]
        q_3d = query.reshape([self.num_heads, 1, self.head_dim])
        k_3d = k_t.transpose(0, 1)
        v_3d = v_t.transpose(0, 1)

        scores = q_3d.matmul(k_3d.transpose(1, 2)) * float(scale)  # [H, 1, S]
        attn_weights = scores.softmax(-1)
        out_3d = attn_weights.matmul(v_3d)  # [H, 1, D]
        return out_3d.reshape([self.num_heads, self.head_dim])

    def free(self, block_table: List[int]):
        """Releases allocated physical blocks back to the free pool when reference count hits zero."""
        for b in block_table:
            rc = self._ref_counts.get(b, 1) - 1
            if rc <= 0:
                self._ref_counts[b] = 0
                if b not in self._free_blocks:
                    self._free_blocks.append(b)
            else:
                self._ref_counts[b] = rc


class StandardKVCache:
    """Standard contiguous append-only Key-Value cache with sliding window option."""

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        sliding_window: Optional[int] = None,
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window

        self.k_cache = np.zeros(
            (max_batch_size, max_seq_len, num_heads, head_dim), dtype=np.float32
        )
        self.v_cache = np.zeros(
            (max_batch_size, max_seq_len, num_heads, head_dim), dtype=np.float32
        )
        self.cur_len = 0

    def update(self, key: _Tensor, value: _Tensor) -> Tuple[_Tensor, _Tensor]:
        """Appends new key and value tensors [batch_size, seq_len, num_heads, head_dim]."""
        k_np = key.to_numpy()
        v_np = value.to_numpy()
        b_sz, s_len, _, _ = k_np.shape

        new_len = self.cur_len + s_len
        if new_len > self.max_seq_len:
            if self.sliding_window is not None:
                if s_len >= self.sliding_window:
                    k_np = k_np[:, -self.sliding_window:]
                    v_np = v_np[:, -self.sliding_window:]
                    s_len = self.sliding_window
                    self.cur_len = 0
                else:
                    keep = min(self.cur_len, self.sliding_window - s_len)
                    self.k_cache[:, :keep] = self.k_cache[:, self.cur_len - keep:self.cur_len]
                    self.v_cache[:, :keep] = self.v_cache[:, self.cur_len - keep:self.cur_len]
                    self.cur_len = keep
                new_len = self.cur_len + s_len
            else:
                raise ValueError(f"KV cache sequence length {new_len} exceeds max {self.max_seq_len}")

        self.k_cache[:b_sz, self.cur_len:new_len] = k_np
        self.v_cache[:b_sz, self.cur_len:new_len] = v_np
        self.cur_len = new_len

        return (
            _from_numpy(np.ascontiguousarray(self.k_cache[:b_sz, :self.cur_len])),
            _from_numpy(np.ascontiguousarray(self.v_cache[:b_sz, :self.cur_len])),
        )

    def reset(self):
        """Resets cache position."""
        self.cur_len = 0
        self.k_cache.fill(0)
        self.v_cache.fill(0)


class StreamingSinkCache:
    """StreamingLLM-style cache: first ``n_sink`` tokens + recent window (Xiao et al., 2024)."""

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        n_sink: int = 4,
    ):
        self.max_batch_size = _check_positive("max_batch_size", max_batch_size)
        self.max_seq_len = _check_positive("max_seq_len", max_seq_len)
        self.num_heads = _check_positive("num_heads", num_heads)
        self.head_dim = _check_positive("head_dim", head_dim)
        self.n_sink = _check_non_negative("n_sink", n_sink)
        if self.n_sink >= self.max_seq_len:
            raise ValueError(f"n_sink = {n_sink} must be below max_seq_len = {max_seq_len}")
        self.k_cache = np.zeros(
            (max_batch_size, max_seq_len, num_heads, head_dim), dtype=np.float32
        )
        self.v_cache = np.zeros(
            (max_batch_size, max_seq_len, num_heads, head_dim), dtype=np.float32
        )
        self.cur_len = 0

    def update(self, key: _Tensor, value: _Tensor) -> Tuple[_Tensor, _Tensor]:
        """Appends [batch, new, heads, dim] keys/values, evicting the middle."""
        k_np = key.to_numpy()
        v_np = value.to_numpy()
        b_sz, s_len = k_np.shape[0], k_np.shape[1]
        if b_sz > self.max_batch_size:
            raise ValueError(f"batch {b_sz} exceeds max_batch_size {self.max_batch_size}")
        full_k = np.concatenate([self.k_cache[:b_sz, : self.cur_len], k_np], axis=1)
        full_v = np.concatenate([self.v_cache[:b_sz, : self.cur_len], v_np], axis=1)
        if full_k.shape[1] > self.max_seq_len:
            keep_recent = self.max_seq_len - self.n_sink
            full_k = np.concatenate([full_k[:, : self.n_sink], full_k[:, -keep_recent:]], axis=1)
            full_v = np.concatenate([full_v[:, : self.n_sink], full_v[:, -keep_recent:]], axis=1)
        self.cur_len = full_k.shape[1]
        self.k_cache[:b_sz, : self.cur_len] = full_k
        self.v_cache[:b_sz, : self.cur_len] = full_v
        return (
            _from_numpy(np.ascontiguousarray(self.k_cache[:b_sz, : self.cur_len])),
            _from_numpy(np.ascontiguousarray(self.v_cache[:b_sz, : self.cur_len])),
        )

    def reset(self) -> None:
        """Resets cache position."""
        self.cur_len = 0
        self.k_cache.fill(0)
        self.v_cache.fill(0)


class ScoredEvictionCache:
    """Score-guided eviction over a fixed budget: H2O, TOVA, or SnapKV selection."""

    POLICIES = ("h2o", "tova", "snap")

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        recent_window: int = 32,
        policy: str = "h2o",
        observe_window: int = 8,
        pool_size: int = 4,
    ):
        self.max_batch_size = _check_positive("max_batch_size", max_batch_size)
        self.max_seq_len = _check_positive("max_seq_len", max_seq_len)
        self.num_heads = _check_positive("num_heads", num_heads)
        self.head_dim = _check_positive("head_dim", head_dim)
        self.recent_window = _check_positive("recent_window", recent_window)
        if self.recent_window >= self.max_seq_len:
            raise ValueError("recent_window must leave room for at least one heavy hitter")
        if policy not in self.POLICIES:
            raise ValueError(f"policy '{policy}' must be one of {self.POLICIES}")
        self.policy = policy
        self.observe_window = _check_positive("observe_window", observe_window)
        self.pool_size = _check_positive("pool_size", pool_size)
        self.k_cache = np.zeros(
            (max_batch_size, max_seq_len, num_heads, head_dim), dtype=np.float32
        )
        self.v_cache = np.zeros(
            (max_batch_size, max_seq_len, num_heads, head_dim), dtype=np.float32
        )
        self.cur_len = 0
        self._accum: Optional[np.ndarray] = None

    def _pooled(self, per_pos: np.ndarray) -> np.ndarray:
        out = np.empty_like(per_pos)
        for i in range(len(per_pos)):
            out[i] = per_pos[max(0, i - self.pool_size + 1) : i + 1].max()
        return out

    def update(
        self, key: _Tensor, value: _Tensor, scores: np.ndarray
    ) -> Tuple[_Tensor, _Tensor]:
        """Appends keys/values, evicting low-score positions beyond the budget."""
        k_np = key.to_numpy()
        v_np = value.to_numpy()
        b_sz, s_len = k_np.shape[0], k_np.shape[1]
        if b_sz > self.max_batch_size:
            raise ValueError(f"batch {b_sz} exceeds max_batch_size {self.max_batch_size}")
        total = self.cur_len + s_len
        expect = (b_sz, self.num_heads, s_len, total)
        if tuple(scores.shape) != expect:
            raise ValueError(f"scores shape {tuple(scores.shape)} != expected {expect}")
        if self._accum is None:
            self._accum = np.zeros(total, dtype=np.float64)
        else:
            self._accum = np.concatenate([self._accum, np.zeros(s_len)])
        self._accum = self._accum + scores.sum(axis=(0, 1, 2))
        full_k = np.concatenate([self.k_cache[:b_sz, : self.cur_len], k_np], axis=1)
        full_v = np.concatenate([self.v_cache[:b_sz, : self.cur_len], v_np], axis=1)
        if total > self.max_seq_len:
            if self.policy == "h2o":
                rank = self._accum.copy()
            elif self.policy == "tova":
                rank = scores[:, :, -1, :].mean(axis=(0, 1))
            else:
                window = scores[:, :, -self.observe_window :, :]
                rank = self._pooled(window.max(axis=(0, 1, 2)))
            recent = np.arange(total - self.recent_window, total)
            rest = np.setdiff1d(np.arange(total), recent, assume_unique=True)
            need = self.max_seq_len - self.recent_window
            heavy = rest[np.argsort(-rank[rest], kind="stable")[:need]]
            keep = np.sort(np.concatenate([heavy, recent]))
            full_k = full_k[:, keep]
            full_v = full_v[:, keep]
            self._accum = self._accum[keep]
            total = self.max_seq_len
        self.cur_len = total
        self.k_cache[:b_sz, : self.cur_len] = full_k
        self.v_cache[:b_sz, : self.cur_len] = full_v
        return (
            _from_numpy(np.ascontiguousarray(self.k_cache[:b_sz, : self.cur_len])),
            _from_numpy(np.ascontiguousarray(self.v_cache[:b_sz, : self.cur_len])),
        )

    def reset(self) -> None:
        """Resets cache position and accumulated scores."""
        self.cur_len = 0
        self._accum = None
        self.k_cache.fill(0)
        self.v_cache.fill(0)


__all__ = [
    "PagedKVCache",
    "StandardKVCache",
    "StreamingSinkCache",
    "ScoredEvictionCache",
]
