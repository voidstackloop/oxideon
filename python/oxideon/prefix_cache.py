"""Radix Tree Prefix KV-Cache for high-throughput LLM serving.

Implements vLLM / SGLang style prefix caching on token IDs:
- RadixNode: Node in radix tree storing token sequences and associated physical KV block IDs
- RadixPrefixCache: Manages prefix matching, node splitting/merging, reference locking, and LRU eviction
"""

import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class RadixNode:
    """A node in the prefix tree representing a contiguous token sub-sequence."""

    def __init__(
        self,
        token_ids: Optional[List[int]] = None,
        block_ids: Optional[List[int]] = None,
        parent: Optional["RadixNode"] = None,
    ):
        self.token_ids: List[int] = list(token_ids) if token_ids is not None else []
        self.block_ids: List[int] = list(block_ids) if block_ids is not None else []
        self.parent: Optional["RadixNode"] = parent
        self.children: Dict[int, "RadixNode"] = {}
        self.last_accessed: float = time.time()
        self.ref_count: int = 0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def touch(self):
        self.last_accessed = time.time()


class RadixPrefixCache:
    """Radix-Tree based prefix KV cache manager.

    Enables instant O(1) sharing of cached Key-Value blocks across prompts with common prefixes
    (e.g., shared system instructions, few-shot exemplars, multi-turn chat history).
    """

    def __init__(self, block_size: int = 16):
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        self.block_size = block_size
        self.root = RadixNode()
        self._total_cached_tokens = 0
        self._total_blocks = 0

    @property
    def total_cached_tokens(self) -> int:
        return self._total_cached_tokens

    @property
    def total_blocks(self) -> int:
        return self._total_blocks

    def __len__(self) -> int:
        return self._total_cached_tokens

    def clear(self) -> None:
        """Resets the radix prefix cache, removing all cached tokens and block mappings."""
        self.root = RadixNode()
        self._total_cached_tokens = 0
        self._total_blocks = 0

    def match_prefix(self, token_ids: List[int]) -> Tuple[List[int], List[int]]:
        """Finds the longest cached prefix for the given prompt tokens."""
        curr = self.root
        matched_tokens: List[int] = []
        matched_blocks: List[int] = []
        idx = 0

        while idx < len(token_ids):
            first_tok = token_ids[idx]
            if first_tok not in curr.children:
                break

            child = curr.children[first_tok]
            child.touch()
            node_toks = child.token_ids

            match_len = 0
            while (
                idx + match_len < len(token_ids)
                and match_len < len(node_toks)
                and token_ids[idx + match_len] == node_toks[match_len]
            ):
                match_len += 1

            if match_len == len(node_toks):
                matched_tokens.extend(node_toks)
                matched_blocks.extend(child.block_ids)
                idx += match_len
                curr = child
            else:
                usable_blocks = match_len // self.block_size
                if usable_blocks > 0 and usable_blocks <= len(child.block_ids):
                    usable_tokens = usable_blocks * self.block_size
                    matched_tokens.extend(node_toks[:usable_tokens])
                    matched_blocks.extend(child.block_ids[:usable_blocks])
                idx += match_len
                break

        return matched_tokens, matched_blocks

    def _adjust_ref_count(self, token_ids: List[int], delta: int) -> None:
        curr = self.root
        idx = 0
        while idx < len(token_ids):
            first_tok = token_ids[idx]
            if first_tok not in curr.children:
                break
            child = curr.children[first_tok]
            child.ref_count = max(0, child.ref_count + delta)
            node_toks = child.token_ids
            match_len = 0
            while (
                idx + match_len < len(token_ids)
                and match_len < len(node_toks)
                and token_ids[idx + match_len] == node_toks[match_len]
            ):
                match_len += 1
            if match_len < len(node_toks):
                break
            idx += match_len
            curr = child

    def lock_prefix(self, token_ids: List[int]) -> None:
        """Increments reference count along matched prefix nodes so they are pinned against LRU eviction."""
        self._adjust_ref_count(token_ids, +1)

    def unlock_prefix(self, token_ids: List[int]) -> None:
        """Decrements reference count along matched prefix nodes."""
        self._adjust_ref_count(token_ids, -1)

    def insert(self, token_ids: List[int], block_ids: List[int]) -> None:
        """Inserts a completed token sequence and its corresponding physical block IDs into the radix tree."""
        if not token_ids or not block_ids:
            return

        curr = self.root
        idx = 0

        while idx < len(token_ids):
            first_tok = token_ids[idx]

            if first_tok not in curr.children:
                blocks_used = idx // self.block_size
                new_node = RadixNode(
                    token_ids=token_ids[idx:],
                    block_ids=block_ids[blocks_used:],
                    parent=curr,
                )
                curr.children[first_tok] = new_node
                self._total_cached_tokens += len(new_node.token_ids)
                self._total_blocks += len(new_node.block_ids)
                return

            child = curr.children[first_tok]
            child.touch()
            node_toks = child.token_ids

            common_len = 0
            while (
                idx + common_len < len(token_ids)
                and common_len < len(node_toks)
                and token_ids[idx + common_len] == node_toks[common_len]
            ):
                common_len += 1

            if common_len < len(node_toks):
                split_node = RadixNode(
                    token_ids=node_toks[:common_len],
                    block_ids=child.block_ids[: (common_len // self.block_size)],
                    parent=curr,
                )
                split_node.ref_count = child.ref_count
                curr.children[first_tok] = split_node

                child.token_ids = node_toks[common_len:]
                child.block_ids = child.block_ids[(common_len // self.block_size) :]
                child.parent = split_node
                split_node.children[child.token_ids[0]] = child

                rem_toks = token_ids[idx + common_len :]
                if rem_toks:
                    rem_blocks = block_ids[(idx + common_len) // self.block_size :]
                    new_leaf = RadixNode(
                        token_ids=rem_toks,
                        block_ids=rem_blocks,
                        parent=split_node,
                    )
                    split_node.children[rem_toks[0]] = new_leaf
                    self._total_cached_tokens += len(rem_toks)
                    self._total_blocks += len(new_leaf.block_ids)
                return
            else:
                idx += common_len
                curr = child

    def evict_lru(self, num_blocks: int) -> List[int]:
        """Evicts unreferenced leaf nodes in LRU order until num_blocks are freed."""
        if num_blocks < 0:
            raise ValueError(f"num_blocks cannot be negative, got {num_blocks}")
        freed_blocks: List[int] = []

        def get_unreferenced_leaves(node: RadixNode) -> List[RadixNode]:
            leaves = []
            for child in node.children.values():
                if child.is_leaf and child.ref_count == 0:
                    leaves.append(child)
                else:
                    leaves.extend(get_unreferenced_leaves(child))
            return leaves

        while len(freed_blocks) < num_blocks:
            leaves = get_unreferenced_leaves(self.root)
            if not leaves:
                break
            oldest = min(leaves, key=lambda n: n.last_accessed)
            freed_blocks.extend(oldest.block_ids)
            self._total_cached_tokens -= len(oldest.token_ids)
            self._total_blocks -= len(oldest.block_ids)

            parent = oldest.parent
            if parent is not None:
                first_tok = oldest.token_ids[0]
                if first_tok in parent.children:
                    del parent.children[first_tok]

        return freed_blocks


__all__ = ["RadixNode", "RadixPrefixCache"]
