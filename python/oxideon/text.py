"""Natural language processing, tokenization, and sequence utilities for Oxideon."""

from collections import Counter
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import re
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module, Parameter, Linear, Embedding, Dropout, SiLU, RMSNorm, _check_repeats, _check_elements


class WordTokenizer:
    """Whitespace and punctuation word-level tokenizer with vocabulary management."""

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        lower: bool = True,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
    ):
        self.lower = lower
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        if vocab is not None:
            self.word2id = dict(vocab)
            self.id2word = {idx: word for word, idx in self.word2id.items()}
        else:
            specials = [pad_token, unk_token, bos_token, eos_token]
            self.word2id = {w: i for i, w in enumerate(specials)}
            self.id2word = {i: w for i, w in enumerate(specials)}

        self.pad_id = self.word2id[pad_token]
        self.unk_id = self.word2id[unk_token]
        self.bos_id = self.word2id[bos_token]
        self.eos_id = self.word2id[eos_token]

    def tokenize(self, text: str) -> List[str]:
        if self.lower:
            text = text.lower()
        return re.findall(r"\w+|[^\w\s]", text)

    def build_vocab(
        self,
        texts: Iterable[str],
        min_freq: int = 1,
        max_vocab_size: Optional[int] = None,
    ):
        counter = Counter()
        for text in texts:
            counter.update(self.tokenize(text))

        specials = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        self.word2id = {w: i for i, w in enumerate(specials)}
        self.id2word = {i: w for i, w in enumerate(specials)}

        for word, count in counter.most_common():
            if count < min_freq:
                break
            if max_vocab_size is not None and len(self.word2id) >= max_vocab_size:
                break
            if word not in self.word2id:
                idx = len(self.word2id)
                self.word2id[word] = idx
                self.id2word[idx] = word

        self.pad_id = self.word2id[self.pad_token]
        self.unk_id = self.word2id[self.unk_token]
        self.bos_id = self.word2id[self.bos_token]
        self.eos_id = self.word2id[self.eos_token]

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = self.tokenize(text)
        ids = [self.word2id.get(w, self.unk_id) for w in tokens]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        words = []
        specials = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
        for idx in ids:
            if skip_special_tokens and idx in specials:
                continue
            words.append(self.id2word.get(idx, self.unk_token))
        return " ".join(words)

    @property
    def vocab_size(self) -> int:
        return len(self.word2id)

    def __len__(self) -> int:
        return len(self.word2id)


class CharTokenizer:
    """Character-level tokenizer for character-based language modeling."""

    def __init__(
        self,
        chars: Optional[Sequence[str]] = None,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
    ):
        self.unk_token = unk_token
        self.pad_token = pad_token

        specials = [pad_token, unk_token]
        self.char2id = {c: i for i, c in enumerate(specials)}
        self.id2char = {i: c for i, c in enumerate(specials)}

        if chars is not None:
            for c in sorted(set(chars)):
                if c not in self.char2id:
                    idx = len(self.char2id)
                    self.char2id[c] = idx
                    self.id2char[idx] = c

        self.pad_id = self.char2id[pad_token]
        self.unk_id = self.char2id[unk_token]

    def build_vocab(self, texts: Iterable[str]):
        all_chars = set()
        for t in texts:
            all_chars.update(t)
        for c in sorted(all_chars):
            if c not in self.char2id:
                idx = len(self.char2id)
                self.char2id[c] = idx
                self.id2char[idx] = c

    def encode(self, text: str) -> List[int]:
        return [self.char2id.get(c, self.unk_id) for c in text]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        chars = []
        specials = {self.pad_id, self.unk_id}
        for idx in ids:
            if skip_special_tokens and idx in specials:
                continue
            chars.append(self.id2char.get(idx, ""))
        return "".join(chars)

    @property
    def vocab_size(self) -> int:
        return len(self.char2id)

    def __len__(self) -> int:
        return len(self.char2id)


def pad_sequence(
    sequences: Sequence[Union[List[int], _Tensor]],
    batch_first: bool = True,
    padding_value: int = 0,
) -> _Tensor:
    """Pads a list of variable length Tensors or lists with `padding_value` into a 2D Tensor."""
    seq_lists = [
        s.to_list() if hasattr(s, "to_list") else (list(s) if not isinstance(s, list) else s)
        for s in sequences
    ]
    max_len = max(len(s) for s in seq_lists) if seq_lists else 0
    padded = np.full((len(seq_lists), max_len), padding_value, dtype=np.int64)

    for i, s in enumerate(seq_lists):
        padded[i, : len(s)] = s

    if not batch_first:
        padded = np.transpose(padded)

    return _from_numpy(np.ascontiguousarray(padded, dtype=np.float32))


# ---------------------------------------------------------------------------
# Foundation Language Modeling & Modern Transformer Decoder Blocks
# ---------------------------------------------------------------------------

class RotaryEmbedding(Module):
    """Rotary Position Embedding (RoPE) (Su et al., 2021)."""

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        dim = _check_elements("dim", dim)
        max_seq_len = _check_elements("max_seq_len", max_seq_len)
        if dim % 2 != 0:
            raise ValueError(f"dim ({dim}) must be even for RotaryEmbedding")
        if dim > 65536:
            raise ValueError(f"dim ({dim}) exceeds maximum supported dimension 65536")
        if max_seq_len > 1_048_576:
            raise ValueError(f"max_seq_len ({max_seq_len}) exceeds maximum precomputed table length 1048576")
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        t = np.arange(max_seq_len, dtype=np.float32)
        freqs = np.outer(t, inv_freq)
        self.cos_cached = np.cos(freqs)
        self.sin_cached = np.sin(freqs)

    def forward(self, q: _Tensor, k: _Tensor, seq_len: int) -> Tuple[_Tensor, _Tensor]:
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]

        def _apply_rotary(x_tensor: _Tensor) -> _Tensor:
            arr = x_tensor.to_numpy()
            # arr: [B, H, T, D]
            orig_shape = arr.shape
            b, h, t, d = orig_shape
            x1 = arr[..., 0::2]
            x2 = arr[..., 1::2]
            c = cos[:t].reshape(1, 1, t, d // 2)
            s = sin[:t].reshape(1, 1, t, d // 2)

            out1 = x1 * c - x2 * s
            out2 = x1 * s + x2 * c
            out = np.empty_like(arr)
            out[..., 0::2] = out1
            out[..., 1::2] = out2
            return _from_numpy(np.ascontiguousarray(out, dtype=np.float32))

        return _apply_rotary(q), _apply_rotary(k)


class SwiGLU(Module):
    """Swish Gated Linear Unit (SwiGLU) feed-forward block."""

    def __init__(self, in_features: int, hidden_features: int, out_features: Optional[int] = None):
        super().__init__()
        out_dim = out_features or in_features
        self.w1 = Linear(in_features, hidden_features)
        self.w2 = Linear(in_features, hidden_features)
        self.w3 = Linear(hidden_features, out_dim)

    def forward(self, x: _Tensor) -> _Tensor:
        gate = (self.w2(x)).silu() if hasattr(self.w2(x), "silu") else (self.w2(x)).relu()
        return self.w3(self.w1(x) * gate)


class CausalSelfAttention(Module):
    """Causal Multi-Head Self-Attention with RoPE and Grouped-Query Attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = dim // num_heads

        self.q_proj = Linear(dim, num_heads * self.head_dim)
        self.k_proj = Linear(dim, self.num_kv_heads * self.head_dim)
        self.v_proj = Linear(dim, self.num_kv_heads * self.head_dim)
        self.out_proj = Linear(dim, dim)

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: _Tensor) -> _Tensor:
        b, t, d = x.shape
        if getattr(x, "requires_grad", False):
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = q.matmul(k.transpose(1, 2)) * scale
            mask = _from_numpy(np.triu(np.full((t, t), -1e9, dtype=np.float32), k=1))
            scores = scores + mask
            attn_weights = scores.softmax(-1)
            attn_out = attn_weights.matmul(v)
            return self.out_proj(attn_out)

        q = self.q_proj(x).to_numpy().reshape(b, t, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).to_numpy().reshape(b, t, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).to_numpy().reshape(b, t, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q_t, k_t = self.rope(_from_numpy(q), _from_numpy(k), t)
        q_np = q_t.to_numpy()
        k_np = k_t.to_numpy()

        # Repeat KV heads if GQA
        if self.num_kv_heads != self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            k_np = np.repeat(k_np, repeats, axis=1)
            v = np.repeat(v, repeats, axis=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q_np @ k_np.transpose(0, 1, 3, 2)) * scale

        # Causal mask: upper triangular
        mask = np.triu(np.full((t, t), -1e9, dtype=np.float32), k=1)
        scores += mask[None, None, :, :]

        exp_s = np.exp(np.clip(scores - np.max(scores, axis=-1, keepdims=True), -50.0, 0.0))
        attn_weights = exp_s / np.sum(exp_s, axis=-1, keepdims=True)

        attn_out = attn_weights @ v
        # [B, H, T, D_h] -> [B, T, D]
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(b, t, d)
        out_tensor = _from_numpy(np.ascontiguousarray(attn_out, dtype=np.float32))
        return self.out_proj(out_tensor)


class CausalTransformerDecoderLayer(Module):
    """Transformer decoder block with Pre-RMSNorm, Causal Attention, and SwiGLU FFN."""

    def __init__(self, dim: int, num_heads: int, num_kv_heads: Optional[int] = None, ffn_dim: Optional[int] = None):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads)
        self.norm2 = RMSNorm(dim)
        ffn_hidden = ffn_dim or int(4 * dim * 2 / 3)
        self.ffn = SwiGLU(dim, ffn_hidden)

    def forward(self, x: _Tensor) -> _Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class CausalTransformerDecoder(Module):
    """Full Autoregressive Transformer Decoder Language Model (LLaMA/Gemma style)."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        num_kv_heads: Optional[int] = None,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        num_layers = _check_repeats("num_layers", num_layers)
        self.vocab_size = vocab_size
        self.embed = Embedding(vocab_size, dim)
        self.layers = []
        for i in range(num_layers):
            layer = CausalTransformerDecoderLayer(dim, num_heads, num_kv_heads)
            self.layers.append(layer)
            setattr(self, f"layer_{i}", layer)
        self.norm = RMSNorm(dim)
        self.lm_head = Linear(dim, vocab_size)

    def forward(self, input_ids: _Tensor) -> _Tensor:
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.lm_head(h)


# ---------------------------------------------------------------------------
# Autoregressive Generation & Sampling Strategies
# ---------------------------------------------------------------------------

def top_k_top_p_filtering(
    logits: np.ndarray,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
) -> np.ndarray:
    """Filters logits using Top-K and Top-P (nucleus) sampling."""
    logits = logits.copy()
    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        indices_to_remove = logits < np.partition(logits, -top_k, axis=-1)[..., -top_k, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_indices = np.argsort(logits, axis=-1)[..., ::-1]
        sorted_logits = np.take_along_axis(logits, sorted_indices, axis=-1)
        exp_l = np.exp(sorted_logits - np.max(sorted_logits, axis=-1, keepdims=True))
        cumulative_probs = np.cumsum(exp_l / np.sum(exp_l, axis=-1, keepdims=True), axis=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].copy()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = np.empty_like(sorted_indices_to_remove)
        np.put_along_axis(indices_to_remove, sorted_indices, sorted_indices_to_remove, axis=-1)
        logits[indices_to_remove] = filter_value

    return logits


def generate(
    model: Module,
    prompt_ids: Union[List[int], _Tensor],
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    eos_token_id: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[int]:
    """Generates text tokens autoregressively from a language model."""
    if isinstance(prompt_ids, _Tensor):
        current_ids = [int(v) for v in prompt_ids.to_numpy().reshape(-1)]
    else:
        current_ids = list(prompt_ids)

    rng = np.random.RandomState(seed) if seed is not None else np.random

    for _ in range(max_new_tokens):
        inp_tensor = _from_numpy(np.array([current_ids], dtype=np.int64))
        logits_tensor = model(inp_tensor)
        logits = logits_tensor.to_numpy()[0, -1, :].copy()

        # Repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(current_ids):
                if logits[token_id] < 0:
                    logits[token_id] *= repetition_penalty
                else:
                    logits[token_id] /= repetition_penalty

        if temperature <= 0.0:
            next_token = int(np.argmax(logits))
        else:
            scaled_logits = logits / max(temperature, 1e-5)
            filtered_logits = top_k_top_p_filtering(scaled_logits, top_k=top_k, top_p=top_p)
            exp_l = np.exp(filtered_logits - np.max(filtered_logits))
            probs = exp_l / np.sum(exp_l)
            probs = np.nan_to_num(probs, nan=0.0)
            if np.sum(probs) > 0:
                probs /= np.sum(probs)
                next_token = int(rng.choice(len(probs), p=probs))
            else:
                next_token = int(np.argmax(logits))

        current_ids.append(next_token)
        if eos_token_id is not None and next_token == eos_token_id:
            break

    return current_ids


__all__ = [
    "WordTokenizer",
    "CharTokenizer",
    "pad_sequence",
    "RotaryEmbedding",
    "SwiGLU",
    "CausalSelfAttention",
    "CausalTransformerDecoderLayer",
    "CausalTransformerDecoder",
    "top_k_top_p_filtering",
    "generate",
]
