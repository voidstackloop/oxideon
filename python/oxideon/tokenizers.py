"""Byte-Pair Encoding (BPE) subword tokenizer for NLP and LLMs.

Implements:
  - BPETokenizer: Trainable byte-pair subword encoder with frequency-aggregated merge training,
    special tokens, batch encoding/decoding, and deterministic serialization.
"""

from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Set, Sequence
import json


class BPETokenizer:
    """Byte-Pair Encoding (BPE) subword tokenizer."""

    def __init__(self):
        self.special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
        self.vocab: Dict[str, int] = {tok: i for i, tok in enumerate(self.special_tokens)}
        self.inv_vocab: Dict[int, str] = {i: tok for i, tok in enumerate(self.special_tokens)}
        self.merges: List[Tuple[str, str]] = []

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.vocab["<pad>"]

    @property
    def unk_token_id(self) -> int:
        return self.vocab["<unk>"]

    @property
    def bos_token_id(self) -> int:
        return self.vocab["<bos>"]

    @property
    def eos_token_id(self) -> int:
        return self.vocab["<eos>"]

    def train(self, corpus: List[str], max_vocab_size: int = 500, min_freq: int = 1):
        """Trains BPE merge rules using word-frequency aggregation for fast O(V) subword learning."""
        # 1. Initialize character vocabulary (always include space so word boundaries survive)
        char_set: Set[str] = {" "}
        for text in corpus:
            char_set.update(list(text))

        for char in sorted(char_set):
            if char not in self.vocab:
                idx = len(self.vocab)
                self.vocab[char] = idx
                self.inv_vocab[idx] = char

        # 2. Count word frequencies across corpus once
        word_counts = Counter(word for doc in corpus for word in doc.split())
        splits = {w: list(w) for w in word_counts}

        while len(self.vocab) < max_vocab_size:
            pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
            for w, count in word_counts.items():
                syms = splits[w]
                if len(syms) < 2:
                    continue
                for i in range(len(syms) - 1):
                    pair_counts[(syms[i], syms[i + 1])] += count

            if not pair_counts:
                break

            best_pair = None
            best_freq = min_freq - 1
            for pair, freq in pair_counts.items():
                if freq > best_freq:
                    best_pair, best_freq = pair, freq
            if best_pair is None:
                break

            new_subword = "".join(best_pair)
            self.merges.append(best_pair)
            if new_subword not in self.vocab:
                idx = len(self.vocab)
                self.vocab[new_subword] = idx
                self.inv_vocab[idx] = new_subword

            p0, p1 = best_pair
            for w in list(word_counts.keys()):
                syms = splits[w]
                if len(syms) < 2:
                    continue
                n = len(syms)
                i = 0
                new_syms = []
                while i < n:
                    if i < n - 1 and syms[i] == p0 and syms[i + 1] == p1:
                        new_syms.append(new_subword)
                        i += 2
                    else:
                        new_syms.append(syms[i])
                        i += 1
                splits[w] = new_syms

    def _apply_merges(self, symbols: List[str], rank: Optional[Dict[Tuple[str, str], int]] = None) -> List[str]:
        """Greedily merges the lowest-rank pair first (Sennrich et al. reference BPE)."""
        if rank is None:
            rank = {pair: i for i, pair in enumerate(self.merges)}
        symbols = list(symbols)
        get = rank.get
        while len(symbols) >= 2:
            best_rank = None
            best_idx = -1
            for i in range(len(symbols) - 1):
                r = get((symbols[i], symbols[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_idx = i
            if best_idx < 0:
                break
            symbols[best_idx : best_idx + 2] = [symbols[best_idx] + symbols[best_idx + 1]]
        return symbols

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encodes text into a sequence of token IDs."""
        tokens: List[int] = []
        if add_special_tokens:
            tokens.append(self.bos_token_id)

        rank = {pair: i for i, pair in enumerate(self.merges)}
        unk = self.unk_token_id
        space_id = self.vocab.get(" ")
        words = text.split(" ")
        last_idx = len(words) - 1
        for w_idx, word in enumerate(words):
            subwords = self._apply_merges(list(word), rank)
            for sw in subwords:
                tokens.append(self.vocab.get(sw, unk))
            if w_idx < last_idx and space_id is not None:
                tokens.append(space_id)

        if add_special_tokens:
            tokens.append(self.eos_token_id)

        return tokens

    def encode_batch(self, texts: Sequence[str], add_special_tokens: bool = False) -> List[List[int]]:
        """Encodes a batch of strings into lists of token IDs."""
        return [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decodes token IDs back into string."""
        pieces = []
        for tid in token_ids:
            tok = self.inv_vocab.get(tid, "<unk>")
            if skip_special_tokens and tok in self.special_tokens:
                continue
            pieces.append(tok)
        return "".join(pieces)

    def decode_batch(self, batch_token_ids: Sequence[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """Decodes a batch of token ID sequences into strings."""
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch_token_ids]

    def save(self, file_path: str):
        """Saves tokenizer state to JSON."""
        data = {
            "vocab": self.vocab,
            "merges": self.merges,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, file_path: str) -> "BPETokenizer":
        """Loads tokenizer state from JSON."""
        tokenizer = cls()
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tokenizer.vocab = data["vocab"]
        tokenizer.inv_vocab = {int(v): k for k, v in tokenizer.vocab.items()}
        tokenizer.merges = [tuple(p) for p in data["merges"]]
        return tokenizer


__all__ = [
    "BPETokenizer",
]
