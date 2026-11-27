"""Canonical deep learning neural network architectures for Oxideon.

Includes ResNet, VisionTransformer (ViT), and CausalLM (Autoregressive GPT).
"""

from typing import List, Optional, Tuple
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import (
    Module,
    Parameter,
    Sequential,
    ModuleList,
    Linear,
    Conv2d,
    GroupNorm,
    LayerNorm,
    ReLU,
    Dropout,
    MultiheadAttention,
    AdaptiveAvgPool2d,
    Flatten,
    _check_repeats,
)


# ---------------------------------------------------------------------------
# 1. ResNet Architecture
# ---------------------------------------------------------------------------

class ResidualBlock(Module):
    """Basic Residual Block with identity skip connection: y = F(x) + x."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = LayerNorm(channels)
        self.relu = ReLU()
        self.conv2 = Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = LayerNorm(channels)

    def forward(self, x: _Tensor) -> _Tensor:
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + residual)


class ResNet18(Module):
    """ResNet-18 vision classifier model."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10, hidden_dim: int = 32):
        super().__init__()
        self.stem = Sequential(
            Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            LayerNorm(hidden_dim),
            ReLU(),
        )
        self.layer1 = ResidualBlock(hidden_dim)
        self.layer2 = ResidualBlock(hidden_dim)
        self.pool = AdaptiveAvgPool2d((1, 1))
        self.flatten = Flatten()
        self.fc = Linear(hidden_dim, num_classes)

    def forward(self, x: _Tensor) -> _Tensor:
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.pool(out)
        out = self.flatten(out)
        return self.fc(out)


# ---------------------------------------------------------------------------
# 2. Vision Transformer (ViT)
# ---------------------------------------------------------------------------

class VisionTransformer(Module):
    """Vision Transformer (ViT) with patch projection and transformer encoder blocks."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        depth = _check_repeats("depth", depth)
        if img_size <= 0 or patch_size <= 0 or in_channels <= 0 or num_classes <= 0 or embed_dim <= 0:
            raise ValueError("img_size, patch_size, in_channels, num_classes, and embed_dim must be positive")
        if img_size % patch_size != 0:
            raise ValueError(f"img_size ({img_size}) must be divisible by patch_size ({patch_size})")
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patch projection
        self.patch_proj = Linear(self.patch_dim, embed_dim)
        self.cls_token = Parameter(_Tensor.randn([1, 1, embed_dim]))
        self.pos_embed = Parameter(_Tensor.randn([1, self.num_patches + 1, embed_dim]))
        self.dropout = Dropout(dropout) if dropout > 0.0 else ReLU()

        # Encoder layers
        self.layers = ModuleList()
        mlp_dim = int(embed_dim * mlp_ratio)
        for _ in range(depth):
            layer = Module()
            layer.norm1 = LayerNorm(embed_dim)
            layer.attn = MultiheadAttention(embed_dim, num_heads, dropout=dropout)
            layer.norm2 = LayerNorm(embed_dim)
            layer.mlp = Sequential(
                Linear(embed_dim, mlp_dim),
                ReLU(),
                Linear(mlp_dim, embed_dim),
            )
            self.layers.append(layer)

        self.norm = LayerNorm(embed_dim)
        self.head = Linear(embed_dim, num_classes)

    def _extract_patches(self, x: _Tensor) -> _Tensor:
        if len(x.shape) != 4:
            raise ValueError(f"VisionTransformer requires 4D images [B, C, H, W], got shape {x.shape}")
        N, C, H, W = x.shape
        p = self.patch_size
        if H % p != 0 or W % p != 0:
            raise ValueError(f"Spatial dimensions (H={H}, W={W}) must be divisible by patch_size={p}")
        h_p, w_p = H // p, W // p

        if x.requires_grad:
            patch_list = []
            for i in range(h_p):
                row_slice = x.narrow(2, i * p, p)
                for j in range(w_p):
                    patch = row_slice.narrow(3, j * p, p).reshape([N, 1, C * p * p])
                    patch_list.append(patch)
            return _Tensor.cat(patch_list, dim=1) if len(patch_list) > 1 else patch_list[0]

        arr = x.to_numpy()
        stacked = (
            arr.reshape(N, C, h_p, p, w_p, p)
            .transpose(0, 2, 4, 1, 3, 5)
            .reshape(N, h_p * w_p, C * p * p)
        )
        return _from_numpy(np.ascontiguousarray(stacked, dtype=np.float32))

    def forward(self, x: _Tensor) -> _Tensor:
        patches = self._extract_patches(x)
        x_proj = self.patch_proj(patches)

        batch_size = x.shape[0]
        cls_tokens = (
            _Tensor.cat([self.cls_token] * batch_size, dim=0)
            if batch_size > 1
            else self.cls_token
        )
        tokens = _Tensor.cat([cls_tokens, x_proj], dim=1)

        x_seq = tokens + self.pos_embed

        for layer in self.layers:
            normed = layer.norm1(x_seq)
            attn_out, _ = layer.attn(normed)
            x_seq = x_seq + attn_out
            x_seq = x_seq + layer.mlp(layer.norm2(x_seq))

        x_seq = self.norm(x_seq)
        cls_out = x_seq.narrow(1, 0, 1).reshape([batch_size, self.embed_dim])
        return self.head(cls_out)


# ---------------------------------------------------------------------------
# 3. Causal LM (Mini-GPT / TransformerLM)
# ---------------------------------------------------------------------------

class MiniGPT(Module):
    """Autoregressive decoder-only Causal Language Model with text generation capability."""

    def __init__(
        self,
        vocab_size: int = 1000,
        max_seq_len: int = 128,
        embed_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        num_layers = _check_repeats("num_layers", num_layers)
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        self.tok_embed = Linear(vocab_size, embed_dim, bias=False)
        self.pos_embed = Parameter(_Tensor.randn([1, max_seq_len, embed_dim]))

        self.layers = ModuleList()
        for _ in range(num_layers):
            layer = Module()
            layer.norm1 = LayerNorm(embed_dim)
            layer.attn = MultiheadAttention(embed_dim, num_heads)
            layer.norm2 = LayerNorm(embed_dim)
            layer.ffn = Sequential(
                Linear(embed_dim, embed_dim * 4),
                ReLU(),
                Linear(embed_dim * 4, embed_dim),
            )
            self.layers.append(layer)

        self.ln_f = LayerNorm(embed_dim)
        self.lm_head = Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx: _Tensor) -> _Tensor:
        arr = idx.to_numpy().astype(int)
        single = False
        if arr.ndim == 1:
            arr = np.expand_dims(arr, 0)
            single = True
        elif arr.ndim != 2:
            raise ValueError(f"MiniGPT requires 1D or 2D token tensors [B, T], got shape {arr.shape}")
        B, T = arr.shape
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}")
        # One-hot embedding
        one_hot = np.zeros((B, T, self.vocab_size), dtype=np.float32)
        for b in range(B):
            for t in range(T):
                val = arr[b, t]
                if 0 <= val < self.vocab_size:
                    one_hot[b, t, val] = 1.0

        x = self.tok_embed(_from_numpy(one_hot))
        pos = self.pos_embed.narrow(1, 0, T)
        x = x + pos

        for layer in self.layers:
            attn_out, _ = layer.attn(layer.norm1(x))
            x = x + attn_out
            x = x + layer.ffn(layer.norm2(x))

        x = self.ln_f(x)
        logits = self.lm_head(x)
        if single:
            return logits.reshape([T, self.vocab_size])
        return logits

    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 20,
        temperature: float = 1.0,
    ) -> List[int]:
        """Autoregressively generates new token IDs."""
        if not prompt_ids:
            raise ValueError("prompt_ids cannot be empty, need at least 1 token")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive (> 0), got {temperature}")
        tokens = list(prompt_ids)
        for _ in range(max_new_tokens):
            cur_seq = tokens[-self.max_seq_len :]
            inp = _from_numpy(np.array([cur_seq], dtype=np.float32))
            logits = self(inp)
            last_logits = logits.to_numpy()[0, -1, :] / max(temperature, 1e-4)

            # Softmax
            exp_l = np.exp(last_logits - np.max(last_logits))
            probs = exp_l / np.sum(exp_l)
            next_token = int(np.argmax(probs))
            tokens.append(next_token)
        return tokens


# ---------------------------------------------------------------------------
# GPT: a trainable decoder-only language model made of tensor ops only (runs on the GPU backend)
# ---------------------------------------------------------------------------

class _GPTBlock(Module):
    def __init__(self, d: int, std: float, proj_std: float):
        super().__init__()
        n = lambda *shape, scale=std: Parameter(_Tensor.randn(list(shape)) * scale)
        self.ln1_w, self.ln1_b = Parameter(_Tensor.ones([d])), Parameter(_Tensor.zeros([d]))
        self.q_w, self.q_b = n(d, d), Parameter(_Tensor.zeros([d]))
        self.k_w, self.k_b = n(d, d), Parameter(_Tensor.zeros([d]))
        self.v_w, self.v_b = n(d, d), Parameter(_Tensor.zeros([d]))
        self.proj_w, self.proj_b = n(d, d, scale=proj_std), Parameter(_Tensor.zeros([d]))
        self.ln2_w, self.ln2_b = Parameter(_Tensor.ones([d])), Parameter(_Tensor.zeros([d]))
        self.fc_w, self.fc_b = n(d, 4 * d), Parameter(_Tensor.zeros([4 * d]))
        self.out_w, self.out_b = n(4 * d, d, scale=proj_std), Parameter(_Tensor.zeros([d]))


def _block_forward(x, blk, num_heads: int, attn_mask=None, causal: bool = False):
    """Pre-LayerNorm transformer block (self-attention + GELU MLP) over `[B, T, D]`, shared by `GPT` and `TinyTTS`."""
    from . import functional as F
    B, T, D = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    H = num_heads
    h = F.layer_norm(x, [D], blk.ln1_w, blk.ln1_b)
    q, k, v = ((h @ w + b).reshape([B, T, H, D // H]).permute([0, 2, 1, 3])
               for w, b in ((blk.q_w, blk.q_b), (blk.k_w, blk.k_b), (blk.v_w, blk.v_b)))
    a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=causal).permute([0, 2, 1, 3]).reshape([B, T, D])
    x = x + (a @ blk.proj_w + blk.proj_b)
    h = F.layer_norm(x, [D], blk.ln2_w, blk.ln2_b)
    return x + (F.gelu(h @ blk.fc_w + blk.fc_b) @ blk.out_w + blk.out_b)


class GPT(Module):
    """A GPT-2 style decoder-only language model: learned positions, pre-LayerNorm blocks, causal self-attention, GELU MLP and
    tied input / output embeddings. Built from tensor ops (no NumPy round trips), so it trains on the GPU backend:

        model = GPT(vocab_size=4096, max_seq_len=256, embed_dim=512, num_layers=6, num_heads=8).to("cuda")
        loss = model.loss(tokens[:, :-1], tokens[:, 1:])

    `num_params()` counts the parameters; `forward(idx)` returns logits `[B, T, vocab]`."""

    def __init__(self, vocab_size: int = 4096, max_seq_len: int = 256, embed_dim: int = 256, num_layers: int = 4, num_heads: int = 4,
                 init_std: float = 0.02):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")
        num_layers = _check_repeats("num_layers", num_layers)
        self.vocab_size, self.max_seq_len, self.embed_dim, self.num_heads = vocab_size, max_seq_len, embed_dim, num_heads
        self.tok_embed = Parameter(_Tensor.randn([vocab_size, embed_dim]) * init_std)
        self.pos_embed = Parameter(_Tensor.randn([max_seq_len, embed_dim]) * init_std)
        proj_std = init_std / math.sqrt(2 * num_layers)  # GPT-2's residual-branch scaling
        self.layers = ModuleList([_GPTBlock(embed_dim, init_std, proj_std) for _ in range(num_layers)])
        self.lnf_w, self.lnf_b = Parameter(_Tensor.ones([embed_dim])), Parameter(_Tensor.zeros([embed_dim]))

    def num_params(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())

    def forward(self, idx: _Tensor) -> _Tensor:
        from . import functional as F
        if idx.ndim != 2:
            raise ValueError(f"GPT expects token ids of shape [B, T], got {list(idx.shape)}")
        B, T = int(idx.shape[0]), int(idx.shape[1])
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}")
        D, H = self.embed_dim, self.num_heads
        x = self.tok_embed.index_select(0, idx.reshape([-1])).reshape([B, T, D]) + self.pos_embed.narrow(0, 0, T)
        for blk in self.layers:
            x = _block_forward(x, blk, H, causal=True)
        x = F.layer_norm(x, [D], self.lnf_w, self.lnf_b)
        return x @ self.tok_embed.transpose(0, 1)

    def loss(self, idx: _Tensor, targets: _Tensor) -> _Tensor:
        """Mean next-token cross-entropy of `forward(idx)` against `targets` (both `[B, T]`)."""
        logits = self.forward(idx).reshape([-1, self.vocab_size])
        # picks the target log-probability with a one-hot mask: gather's backward is slow on the GPU backends
        classes = _Tensor.arange(0, self.vocab_size, 1).to_dtype("i64").to(logits.device).reshape([1, self.vocab_size])
        onehot = (classes == targets.reshape([-1, 1])).to_dtype(logits.dtype)
        return -(logits.log_softmax(-1) * onehot).sum() / float(logits.shape[0])


# ---------------------------------------------------------------------------
# TinyTTS: phonemes -> log-mel frames (non-autoregressive, with a duration predictor), in the spirit of FastSpeech
# ---------------------------------------------------------------------------

def _local_conv(x, blk, fm, k: int) -> _Tensor:
    """Depthwise conv over time (zero padding; padded frames are zeroed first so they do not leak in) -> GELU -> pointwise.
    `blk` holds cv_ln_w / cv_ln_b / cv_w `[k, D]` / cv_pw / cv_b; `fm` is the float frame mask `[B, T, 1]`."""
    from . import functional as F
    B, T, D = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    p = k // 2
    h = F.layer_norm(x, [D], blk.cv_ln_w, blk.cv_ln_b) * fm
    z = h.narrow(1, 0, 1) * 0.0
    h = _Tensor.cat([z] * p + [h] + [z] * p, 1)
    y = None
    for j in range(k):
        t = h.narrow(1, j, T) * blk.cv_w.narrow(0, j, 1).reshape([1, 1, D])
        y = t if y is None else y + t
    return F.gelu(y) @ blk.cv_pw + blk.cv_b


def _add_conv_branch(blk, dim: int, k: int, zero_out: bool = True):
    blk.cv_ln_w, blk.cv_ln_b = Parameter(_Tensor.ones([dim])), Parameter(_Tensor.zeros([dim]))
    blk.cv_w = Parameter(_Tensor.randn([k, dim]) * (1.0 / math.sqrt(k)))
    blk.cv_pw = Parameter(_Tensor.zeros([dim, dim]) if zero_out else _Tensor.randn([dim, dim]) * 0.02)
    blk.cv_b = Parameter(_Tensor.zeros([dim]))


def _sinusoid(length: int, dim: int) -> np.ndarray:
    pos = np.arange(length)[:, None]
    i = np.arange(dim // 2)[None, :]
    ang = pos / (10000.0 ** (2 * i / dim))
    out = np.zeros((length, dim), dtype=np.float32)
    out[:, 0::2], out[:, 1::2] = np.sin(ang), np.cos(ang)
    return out


class TinyTTS(Module):
    """A small text-to-speech acoustic model (about 4 M parameters at the defaults).

    Phoneme ids (plus a word-start flag and a speaker id) go through a transformer encoder; a duration head predicts how many
    mel frames each phoneme lasts (`log(1 + frames)`); the encoder states are expanded to frame rate with an alignment matrix
    `A[B, frames, phonemes]` and a transformer decoder writes the log-mel frames. A vocoder (Griffin-Lim, `ox.audio.griffin_lim`)
    turns the mel frames into audio. Built from tensor ops only, so it trains on the GPU backend.

        mel, log_dur = model(ph, word_start, voice, ph_mask, A, phase, frame_mask)

    `ph`, `word_start`: int64 `[B, N]`; `voice`: int64 `[B]`; `ph_mask` / `frame_mask`: bool, True where real;
    `A`: float `[B, T, N]`; `phase`: float `[B, T, 2]` (position inside the phoneme, 1 / its length)."""

    def __init__(self, n_phones: int, n_mels: int = 80, dim: int = 192, heads: int = 4, enc_layers: int = 4, dec_layers: int = 5,
                 n_voices: int = 2, max_len: int = 1024, n_flags: int = 2, n_aux: int = 0, conv_kernel: int = 0, variance: bool = False,
                 prosody_layers: int = 0):
        super().__init__()
        self.n_phones, self.n_mels, self.dim, self.heads, self.max_len = n_phones, n_mels, dim, heads, max_len
        self.conv_kernel, self.variance = conv_kernel, variance
        std = 0.02
        n = lambda *shape, scale=std: Parameter(_Tensor.randn(list(shape)) * scale)
        self.n_aux = n_aux
        self.ph_embed, self.ws_embed, self.voice_embed = n(n_phones, dim), n(n_flags, dim), n(n_voices, dim)
        self.encoder = ModuleList([_GPTBlock(dim, std, std / math.sqrt(2 * enc_layers)) for _ in range(enc_layers)])
        self.enc_ln_w, self.enc_ln_b = Parameter(_Tensor.ones([dim])), Parameter(_Tensor.zeros([dim]))
        self.dur_w1, self.dur_b1 = n(dim, dim), Parameter(_Tensor.zeros([dim]))
        self.dur_w2, self.dur_b2 = n(dim, 1), Parameter(_Tensor.zeros([1]))
        self.phase_w = n(2, dim, scale=0.1)
        if variance:  # FastSpeech 2 style: per-phoneme pitch + energy are predicted, quantised (VAR_BINS) and added to the encoder
            self.var_w1, self.var_b1 = n(dim, dim), Parameter(_Tensor.zeros([dim]))  # states (true values while training);
            self.var_w2, self.var_b2 = n(dim, 2), Parameter(_Tensor.zeros([2]))  # the embeddings start at zero, so an older
            self.pitch_emb = Parameter(_Tensor.zeros([self.VAR_BINS, dim]))  # checkpoint loads (strict=False) unchanged
            self.energy_emb = Parameter(_Tensor.zeros([self.VAR_BINS, dim]))
        self.decoder = ModuleList([_GPTBlock(dim, std, std / math.sqrt(2 * dec_layers)) for _ in range(dec_layers)])
        if conv_kernel:  # a local (depthwise temporal conv -> GELU -> pointwise) residual branch after every decoder block;
            if conv_kernel % 2 == 0:  # the pointwise weight starts at zero, so an older checkpoint loads (strict=False) unchanged
                raise ValueError(f"conv_kernel must be odd, got {conv_kernel}")
            for blk in self.decoder:
                _add_conv_branch(blk, dim, conv_kernel)
        self.dec_ln_w, self.dec_ln_b = Parameter(_Tensor.ones([dim])), Parameter(_Tensor.zeros([dim]))
        self.mel_w, self.mel_b = n(dim, n_mels), Parameter(_Tensor.zeros([n_mels]))
        if n_aux:  # extra per-frame outputs, e.g. normalised log-F0 and a voicing logit
            self.aux_w, self.aux_b = n(dim, n_aux), Parameter(_Tensor.zeros([n_aux]))
        self.prosody_layers = prosody_layers
        if prosody_layers:  # tiny-voca style acoustic stage: the decoder states plus a projection of frame [log-F0, voiced, energy]
            self.energy_w, self.energy_b = n(dim, 1), Parameter(_Tensor.zeros([1]))  # go through more blocks, so what comes out
            self.pros_w, self.pros_b = Parameter(_Tensor.zeros([3, dim])), Parameter(_Tensor.zeros([dim]))  # is specific to one
            self.post = ModuleList([_GPTBlock(dim, std, 0.0) for _ in range(prosody_layers)])  # melody (zero output projections)
        self._pos = {}

    def num_params(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())

    def _positions(self, length: int, like) -> _Tensor:
        key = (length, str(like.device))
        if key not in self._pos:
            self._pos[key] = _from_numpy(_sinusoid(length, self.dim)).to(like.device)
        return self._pos[key]

    @staticmethod
    def _key_mask(mask: _Tensor) -> _Tensor:
        return mask.reshape([int(mask.shape[0]), 1, 1, int(mask.shape[1])])

    def encode(self, ph, word_start, voice, ph_mask) -> Tuple[_Tensor, _Tensor]:
        """Encoder states `[B, N, D]` and the predicted `log(1 + frames)` per phoneme `[B, N]`."""
        from . import functional as F
        B, N = int(ph.shape[0]), int(ph.shape[1])
        D = self.dim
        x = (self.ph_embed.index_select(0, ph.reshape([-1])).reshape([B, N, D])
             + self.ws_embed.index_select(0, word_start.reshape([-1])).reshape([B, N, D])
             + self.voice_embed.index_select(0, voice).reshape([B, 1, D]) + self._positions(N, self.ph_embed))
        mask = self._key_mask(ph_mask)
        for blk in self.encoder:
            x = _block_forward(x, blk, self.heads, attn_mask=mask)
        x = F.layer_norm(x, [D], self.enc_ln_w, self.enc_ln_b)
        log_dur = (F.gelu(x @ self.dur_w1 + self.dur_b1) @ self.dur_w2 + self.dur_b2).reshape([B, N])
        return x, log_dur

    VAR_BINS, VAR_RANGE = 32, 3.0  # bin 0 = below -VAR_RANGE (e.g. an unvoiced phoneme's pitch), bins 1..31 cover [-3, 3]

    @classmethod
    def var_bins(cls, v: np.ndarray) -> np.ndarray:
        """Quantises normalised per-phoneme pitch / energy values (any shape) to bin ids."""
        r, k = cls.VAR_RANGE, cls.VAR_BINS - 1
        return np.where(v < -r, 0, 1 + np.clip(np.floor((v + r) / (2 * r) * k), 0, k - 1)).astype(np.int64)

    def predict_variance(self, enc) -> _Tensor:
        """Predicted normalised per-phoneme `[pitch, energy]`, `[B, N, 2]`."""
        from . import functional as F
        return F.gelu(enc @ self.var_w1 + self.var_b1) @ self.var_w2 + self.var_b2

    def add_variance(self, enc, pitch_bins: _Tensor, energy_bins: _Tensor) -> _Tensor:
        """Adds the pitch / energy embeddings of int64 bin ids `[B, N]` to the encoder states."""
        B, N, D = int(enc.shape[0]), int(enc.shape[1]), self.dim
        return (enc + self.pitch_emb.index_select(0, pitch_bins.reshape([-1])).reshape([B, N, D])
                + self.energy_emb.index_select(0, energy_bins.reshape([-1])).reshape([B, N, D]))

    def _local_conv(self, x, blk, fm) -> _Tensor:
        return _local_conv(x, blk, fm, self.conv_kernel)

    def decode_hidden(self, enc, A, phase, frame_mask) -> _Tensor:
        """The decoder's final (layer-normed) frame states `[B, T, dim]`, before the mel / pitch heads."""
        from . import functional as F
        B, T = int(A.shape[0]), int(A.shape[1])
        D = self.dim
        x = A @ enc + phase @ self.phase_w + self._positions(T, self.ph_embed)
        mask = self._key_mask(frame_mask)
        fm = frame_mask.to_dtype(x.dtype).reshape([B, T, 1]) if self.conv_kernel else None
        for blk in self.decoder:
            x = _block_forward(x, blk, self.heads, attn_mask=mask)
            if self.conv_kernel:
                x = x + self._local_conv(x, blk, fm)
        return F.layer_norm(x, [D], self.dec_ln_w, self.dec_ln_b)

    def energy(self, x) -> _Tensor:
        """Predicted frame energy `[B, T]` (the mean normalised log-mel) from decoder states (`prosody_layers` models)."""
        return (x @ self.energy_w + self.energy_b).reshape([int(x.shape[0]), int(x.shape[1])])

    @staticmethod
    def hard_prosody(aux: np.ndarray, energy: np.ndarray) -> np.ndarray:
        """The acoustic stage's condition `[B, T, 3]` from predicted aux `[B, T, 2]` (normalised log-F0, voicing logit) and energy."""
        return np.stack([np.clip(aux[..., 0], -3.0, 3.0), (aux[..., 1] > 0).astype(np.float32), np.clip(energy, -3.0, 3.0)], -1).astype(np.float32)

    def acoustic(self, x, cond, frame_mask) -> _Tensor:
        """Acoustic-stage states `[B, T, dim]`: decoder states `x` conditioned on frame prosody `cond` `[B, T, 3]`. A new stage
        starts as the identity, so a model warm-started from one without it behaves the same at step 0."""
        y = x + cond @ self.pros_w + self.pros_b
        mask = self._key_mask(frame_mask)
        for blk in self.post:
            y = _block_forward(y, blk, self.heads, attn_mask=mask)
        return y

    def outputs(self, x):
        """Mel (and, with `n_aux`, pitch / voicing) predictions from decoder states `x` (`decode_hidden`)."""
        mel = x @ self.mel_w + self.mel_b
        return (mel, x @ self.aux_w + self.aux_b) if self.n_aux else mel

    def decode(self, enc, A, phase, frame_mask) -> _Tensor:
        return self.outputs(self.decode_hidden(enc, A, phase, frame_mask))

    def forward(self, ph, word_start, voice, ph_mask, A, phase, frame_mask, pitch_bins=None, energy_bins=None):
        """With `variance=True` the result gains a last item, the predicted `[pitch, energy]`; the decoder sees the given bins
        (the true values while training) or, when they are None, the quantised predictions."""
        enc, log_dur = self.encode(ph, word_start, voice, ph_mask)
        vpred = None
        if self.variance:
            vpred = self.predict_variance(enc)
            if pitch_bins is None:
                q = self.var_bins(vpred.to("cpu").to_numpy())
                pitch_bins, energy_bins = (_from_numpy(np.ascontiguousarray(q[..., i])).to(enc.device) for i in (0, 1))
            enc = self.add_variance(enc, pitch_bins, energy_bins)
        out = self.decode(enc, A, phase, frame_mask)
        res = (*out, log_dur) if self.n_aux else (out, log_dur)
        return (*res, vpred) if self.variance else res


class PitchFlow(Module):
    """Generative pitch for TinyTTS: flow matching (rectified flow) on per-frame `[normalised log-F0, voicing sign]`, conditioned on
    a (frozen) TinyTTS decoder's hidden states. A regression head predicts the *average* melody of a sentence, which is flat; this
    model *samples* one. The log-F0 channel is interpolated through unvoiced frames; the voicing channel is +1 / -1.

        v = flow(x_t, PitchFlow.time_features(t), cond, frame_mask)    # velocity (x1 - x0) at x_t = (1 - t) x0 + t x1
        x1 = flow.sample(cond, frame_mask, noise, steps=16)           # Euler from noise x0 ~ N(0, I)
    """

    def __init__(self, cond_dim: int = 192, dim: int = 128, layers: int = 2, heads: int = 4, conv_kernel: int = 9, channels: int = 2):
        super().__init__()
        std = 0.02
        n = lambda *shape, scale=std: Parameter(_Tensor.randn(list(shape)) * scale)
        self.dim, self.heads, self.conv_kernel, self.channels = dim, heads, conv_kernel, channels
        self.cond_w, self.cond_b = n(cond_dim, dim), Parameter(_Tensor.zeros([dim]))
        self.in_w, self.in_b = n(channels, dim, scale=0.1), Parameter(_Tensor.zeros([dim]))
        self.t_w1, self.t_b1 = n(dim, dim), Parameter(_Tensor.zeros([dim]))
        self.t_w2, self.t_b2 = n(dim, dim), Parameter(_Tensor.zeros([dim]))
        self.blocks = ModuleList([_GPTBlock(dim, std, std / math.sqrt(2 * layers)) for _ in range(layers)])
        for blk in self.blocks:
            _add_conv_branch(blk, dim, conv_kernel, zero_out=False)
        self.ln_w, self.ln_b = Parameter(_Tensor.ones([dim])), Parameter(_Tensor.zeros([dim]))
        self.out_w, self.out_b = Parameter(_Tensor.zeros([dim, channels])), Parameter(_Tensor.zeros([channels]))

    def num_params(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())

    @staticmethod
    def time_features(t, dim: int = 128) -> np.ndarray:
        """Sinusoidal features of flow times `t` in [0, 1], `[B] -> [B, dim]`."""
        half = dim // 2
        f = np.exp(-math.log(1000.0) * np.arange(half) / half)
        a = np.asarray(t, np.float64)[:, None] * 1000.0 * f[None, :]
        return np.concatenate([np.sin(a), np.cos(a)], 1).astype(np.float32)

    def forward(self, xt, tfeat, cond, frame_mask) -> _Tensor:
        from . import functional as F
        B, T = int(xt.shape[0]), int(xt.shape[1])
        D = self.dim
        temb = F.gelu(tfeat @ self.t_w1 + self.t_b1) @ self.t_w2 + self.t_b2
        x = cond @ self.cond_w + self.cond_b + xt @ self.in_w + self.in_b + temb.reshape([B, 1, D])
        mask = frame_mask.reshape([B, 1, 1, T])
        fm = frame_mask.to_dtype(x.dtype).reshape([B, T, 1])
        for blk in self.blocks:
            x = _block_forward(x, blk, self.heads, attn_mask=mask)
            x = x + _local_conv(x, blk, fm, self.conv_kernel)
        return F.layer_norm(x, [D], self.ln_w, self.ln_b) @ self.out_w + self.out_b

    def sample(self, cond, frame_mask, noise, steps: int = 16) -> _Tensor:
        """Euler integration of the learned flow from `noise` (`[B, T, channels]`, N(0, I) times a temperature) to a sample."""
        x = noise
        B = int(noise.shape[0])
        for i in range(steps):
            tf = _from_numpy(self.time_features(np.full(B, i / steps), self.dim)).to(noise.device)
            x = x + self.forward(x, tf, cond, frame_mask) * (1.0 / steps)
        return x


def pulse_train(f0, sample_rate: int = 24000, hop: int = 256, limit: float = 0.47, phase=None, jitter: float = 0.0,
                shimmer: float = 0.0, rng=None) -> np.ndarray:
    """Band-limited pulse train `[B, T * hop]` (unit power) for per-frame `f0` `[B, T]` in Hz: the closed-form sum of every
    harmonic below `limit * sample_rate`, pitch interpolated in the log domain between frame centres, the top harmonic faded in
    and out so the wave stays continuous while the pitch glides. Float64 keeps the phase exact. `jitter` / `shimmer` give each
    glottal cycle a random relative period / amplitude deviation of that standard deviation (real voices: ~0.005-0.01 / ~0.03-0.05),
    the cycle-to-cycle irregularity whose absence makes a perfectly periodic source sound buzzy."""
    from . import oxideon as _native
    _pulse_train = getattr(_native, "_pulse_train", None)  # absent in an older compiled extension
    if _pulse_train is not None and not (jitter or shimmer):  # the training path: the same sum in Rust, one row per thread
        f0 = np.asarray(f0, np.float32)
        ph = None if phase is None else [float(v) for v in np.asarray(phase, np.float64).reshape(-1)]
        return _pulse_train(f0.reshape(-1).tolist(), f0.shape[0], f0.shape[1], int(sample_rate), int(hop), float(limit), ph).to_numpy()
    f0 = np.asarray(f0, np.float64)
    B, T = f0.shape
    knots = np.log(np.maximum(f0, 1.0))
    knots = np.concatenate([knots, knots[:, -1:]], 1)
    pos = np.arange(T * hop) / hop
    low = np.floor(pos).astype(np.int64)
    w = pos - low
    freq = np.exp(knots[:, low] * (1 - w) + knots[:, low + 1] * w)
    if jitter or shimmer:
        rng = np.random.default_rng() if rng is None else rng
        cyc = np.floor(np.cumsum(freq / sample_rate, 1)).astype(np.int64)  # glottal cycle index of every sample
        rows = np.arange(B)[:, None]
        if jitter:
            freq = freq * (1.0 + jitter * rng.standard_normal((B, int(cyc.max()) + 1)))[rows, cyc]
        amp = (1.0 + shimmer * rng.standard_normal((B, int(cyc.max()) + 1)))[rows, cyc] if shimmer else None
    else:
        amp = None
    theta = 2 * np.pi * np.cumsum(freq / sample_rate, 1)
    if phase is not None:
        theta = theta + np.asarray(phase, np.float64)[:, None]
    theta = np.remainder(theta, 2 * np.pi)
    harm = np.maximum(limit * sample_rate / freq, 1.0)
    whole = np.floor(harm)
    den = 2 * np.sin(theta / 2)
    tiny = np.abs(den) < 1e-7
    core = np.where(tiny, whole, np.sin((whole + 0.5) * theta) / np.where(tiny, 1.0, den) - 0.5)
    out = (core + (harm - whole) * np.cos((whole + 1) * theta)) * np.sqrt(2.0 / harm)
    return (out if amp is None else out * amp).astype(np.float32)


class SourceFilterHead(Module):
    """Differentiable harmonic-plus-noise waveform head (DDSP / neural homomorphic vocoder): from decoder states `[B, T, dim]`
    two linear maps give, per frame, a complex cepstrum (causal = minimum phase, anticausal = the glottal pulse's maximum-phase
    part) for a pulse train at the given pitch and a cepstrum for white noise, plus per-sub-frame gains. Each source is
    filtered in the STFT domain (Hann, 4x overlap) and resynthesised; the DFTs are fixed strided (transposed) convolutions, so
    the whole path trains on GPU with gradients from a loss on the waveform. About 26k parameters at dim 192.

        wave = head(hidden, f0_hz, lengths)    # [B, T * hop]; frame t is centred on sample t * hop
    """

    def __init__(self, dim: int = 192, sample_rate: int = 24000, hop: int = 256, synth_fft: int = 512, synth_hop: int = 128,
                 noise_fft: int = 256, noise_hop: int = 64, cep_causal: int = 80, cep_anti: int = 16, noise_cep: int = 32,
                 harmonic_limit: float = 0.47):
        super().__init__()
        self.dim, self.sr, self.hop, self.limit = dim, sample_rate, hop, harmonic_limit
        self.sfft, self.shop, self.nfft, self.nhop = synth_fft, synth_hop, noise_fft, noise_hop
        self.cc, self.ca, self.nc = cep_causal, cep_anti, noise_cep
        self.hsteps, self.nsteps = hop // synth_hop, hop // noise_hop
        hb = np.zeros(cep_causal + cep_anti + self.hsteps, np.float32)
        hb[0] = -1.5  # start quiet and spectrally flat
        nb = np.zeros(noise_cep + self.nsteps, np.float32)
        nb[0] = -2.5
        self.h_w, self.h_b = Parameter(_Tensor.zeros([dim, len(hb)])), Parameter(_from_numpy(hb))
        self.n_w, self.n_b = Parameter(_Tensor.zeros([dim, len(nb)])), Parameter(_from_numpy(nb))
        self.jitter, self.shimmer = 0.0, 0.0  # source irregularity (see `pulse_train`); not weights, set per use
        self._const = {}

    def num_params(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())

    @staticmethod
    def _lifter(size, first):
        s = 1.0 / (1.0 + np.arange(first, first + size, dtype=np.float64) / 4)
        if first == 0:
            s[0] = 2.0  # coefficient 0 is the log gain and needs range
        return s.astype(np.float32)

    def _c(self, key, device, make):
        k = (key, str(device))
        if k not in self._const:
            self._const[k] = _from_numpy(np.ascontiguousarray(make())).to(device)
        return self._const[k]

    def _dft(self, n_coef, fft, first, device):
        """`(cos, sin)` `[n_coef, bins]` of quefrencies first .. first + n_coef - 1 (the rFFT of a cepstrum as a matmul)."""
        def make(fn):
            q = np.arange(first, first + n_coef)[:, None]
            k = np.arange(fft // 2 + 1)[None, :]
            return fn(2 * np.pi * q * k / fft).astype(np.float32)
        return (self._c(("cos", n_coef, fft, first), device, lambda: make(np.cos)), self._c(("sin", n_coef, fft, first), device, lambda: make(np.sin)))

    def stft(self, x, fft, hop):
        """Centred, zero-padded Hann STFT of `x` `[B, N]` -> `(re, im)` `[B, fft // 2 + 1, N // hop + 1]`."""
        from . import functional as F
        nb = fft // 2 + 1

        def basis():
            t, k = np.arange(fft), np.arange(nb)[:, None]
            w = 0.5 - 0.5 * np.cos(2 * np.pi * t / fft)
            return np.concatenate([np.cos(2 * np.pi * k * t / fft) * w, -np.sin(2 * np.pi * k * t / fft) * w])[:, None, :].astype(np.float32)
        B = int(x.shape[0])
        z = (x.narrow(1, 0, 1) * 0.0).expand([B, fft // 2])
        s = F.conv1d(_Tensor.cat([z, x, z], 1).reshape([B, 1, -1]), self._c(("ana", fft), x.device, basis), stride=hop)
        return s.narrow(1, 0, nb), s.narrow(1, nb, nb)

    def istft(self, re, im, fft, hop, length):
        """Inverse of `stft` (Hann OLA, normalised by the summed squared window), first `length` samples `[B, length]`."""
        from . import functional as F
        B, S = int(re.shape[0]), int(re.shape[2])
        y = F.conv_transpose1d(_Tensor.cat([re, im], 1), self._c(("syn", fft), re.device, lambda: NeuralVocoder.istft_basis(fft)), stride=hop).reshape([B, -1])

        def env():
            w2 = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(fft) / fft)) ** 2
            e = np.zeros((S - 1) * hop + fft)
            for i in range(S):
                e[i * hop:i * hop + fft] += w2
            return (1.0 / np.maximum(e[fft // 2:fft // 2 + length], 1e-3)).astype(np.float32)[None]
        return y.narrow(1, fft // 2, length) * self._c(("env", fft, hop, S, length), re.device, env)

    def _filter(self, source, logmag_phase, fft, hop):
        """Multiplies the STFT of `source` `[B, N]` by `exp(logmag + i * phase)` `[B, frames, bins]` and resynthesises `[B, N]`."""
        B, N = int(source.shape[0]), int(source.shape[1])
        z = (source.narrow(1, 0, 1) * 0.0).expand([B, fft // 2])  # half a window of trailing zeros, as the padded batch has
        sr_, si_ = self.stft(_Tensor.cat([source, z], 1), fft, hop)
        lm, ph = logmag_phase
        mag = lm.clamp(-14.0, 6.0).exp()
        hr, hi = (mag * ph.cos()).permute([0, 2, 1]), (mag * ph.sin()).permute([0, 2, 1])
        y = self.istft(sr_ * hr - si_ * hi, sr_ * hi + si_ * hr, fft, hop, N + fft // 2)
        return y.narrow(1, 0, N)

    def forward(self, hidden, f0_hz, lengths, noise=None, phase=None) -> _Tensor:
        return self.synth(hidden @ self.h_w + self.h_b, hidden @ self.n_w + self.n_b, f0_hz, lengths, noise, phase)

    def synth(self, hctrl, nctrl, f0_hz, lengths, noise=None, phase=None) -> _Tensor:
        """The waveform from per-frame controls: `hctrl` `[B, T, cep_causal + cep_anti + hop // synth_hop]` (harmonic cepstrum and
        gains) and `nctrl` `[B, T, noise_cep + hop // noise_hop]` (noise cepstrum and gains); `forward` maps decoder states to them."""
        B, T = int(hctrl.shape[0]), int(hctrl.shape[1])
        dev, hop = hctrl.device, self.hop
        lengths = np.asarray(lengths, np.int64)
        N = T * hop
        audible = (np.arange(N)[None, :] < lengths[:, None] * hop).astype(np.float32)
        last = np.maximum(lengths - 1, 0)
        put = lambda a: _from_numpy(np.ascontiguousarray(a)).to(dev)

        def take(flat, idx):  # rows of flat [B * rows, C] at per-utterance indices idx [B, S]
            rows = int(flat.shape[0]) // B
            return flat.index_select(0, put((idx + np.arange(B)[:, None] * rows).reshape(-1).astype(np.int64))).reshape([B, idx.shape[1], int(flat.shape[1])])

        # harmonic branch: envelope averaged between neighbouring frames, one gain per sub-frame
        h = hctrl.reshape([B * T, -1])
        st = self.hsteps
        sub = np.minimum(np.arange(st * T + 3)[None, :], np.maximum(st * lengths - 1, 0)[:, None])
        here, nxt = sub // st, np.minimum((sub + 1) // st, last[:, None])
        a, b_ = take(h, here), take(h, nxt)
        k = self.cc + self.ca
        cau = (a.narrow(2, 0, self.cc) + b_.narrow(2, 0, self.cc)) * 0.5 * self._c(("lc", self.cc), dev, lambda: self._lifter(self.cc, 0))
        ant = (a.narrow(2, self.cc, self.ca) + b_.narrow(2, self.cc, self.ca)) * 0.5 * self._c(("la", self.ca), dev, lambda: self._lifter(self.ca, 1))
        gain = take(h.narrow(1, k, st).reshape([B * T * st, 1]), sub)
        cau = _Tensor.cat([cau.narrow(2, 0, 1) + gain, cau.narrow(2, 1, self.cc - 1)], 2)
        cc, sc = self._dft(self.cc, self.sfft, 0, dev)
        ca, sa = self._dft(self.ca, self.sfft, 1, dev)
        harm_resp = (cau @ cc + ant @ ca, ant @ sa - cau @ sc)
        # noise branch: one envelope per frame, one gain per noise sub-frame
        n = nctrl.reshape([B * T, -1])
        st = self.nsteps
        sub = np.minimum(np.arange(st * T + 3)[None, :], np.maximum(st * lengths - 1, 0)[:, None])
        shp = take(n.narrow(1, 0, self.nc), sub // st) * self._c(("ln", self.nc), dev, lambda: self._lifter(self.nc, 0))
        gain = take(n.narrow(1, self.nc, st).reshape([B * T * st, 1]), sub)
        shp = _Tensor.cat([shp.narrow(2, 0, 1) + gain, shp.narrow(2, 1, self.nc - 1)], 2)
        cn, sn = self._dft(self.nc, self.nfft, 0, dev)
        noise_resp = (shp @ cn, shp @ sn * -1.0)
        # sources: pitch held through padding so a row's audio never depends on the batch
        f0 = np.asarray(f0_hz, np.float64)[np.arange(B)[:, None], np.minimum(np.arange(T)[None, :], last[:, None])]
        pulses = put(pulse_train(f0, self.sr, hop, self.limit, phase, self.jitter, self.shimmer) * audible)
        mask = put(audible)
        noise = _Tensor.randn([B, N]).to(dev) * mask if noise is None else put((np.asarray(noise, np.float32) * audible).astype(np.float32))
        wave = self._filter(pulses, harm_resp, self.sfft, self.shop) + self._filter(noise, noise_resp, self.nfft, self.nhop)
        return wave * mask


class NeuralVocoder(Module):
    """Neural vocoder (Vocos-style): frame features `[B, T, n_in]` (normalised log-mel, plus e.g. normalised log-F0 and voicing)
    -> ConvNeXt-like blocks at frame rate -> per-frame log-magnitude and phase of an `n_fft` STFT -> waveform by a fixed inverse
    DFT (`conv_transpose1d` with a windowed basis, overlap-add normalised). Trained with a mel loss plus discriminators
    (examples/tts/train_vocoder.py), it learns to turn a smooth predicted spectrogram into sharp speech.

        wav = voc(feats)    # [B, (T - 1) * hop]; sample 0 is the centre of frame 0
    """

    def __init__(self, n_in: int = 82, dim: int = 384, layers: int = 8, kernel: int = 7, n_fft: int = 1024, hop: int = 256):
        super().__init__()
        self.n_in, self.dim, self.kernel, self.n_fft, self.hop = n_in, dim, kernel, n_fft, hop
        nb = n_fft // 2 + 1
        n = lambda *shape, scale=0.02: Parameter(_Tensor.randn(list(shape)) * scale)
        self.in_w, self.in_b = n(n_in, dim, scale=1.0 / math.sqrt(n_in)), Parameter(_Tensor.zeros([dim]))
        self.blocks = ModuleList([Module() for _ in range(layers)])
        for blk in self.blocks:
            blk.dw = n(kernel, dim, scale=1.0 / math.sqrt(kernel))
            blk.ln_w, blk.ln_b = Parameter(_Tensor.ones([dim])), Parameter(_Tensor.zeros([dim]))
            blk.fc1, blk.b1 = n(dim, 3 * dim), Parameter(_Tensor.zeros([3 * dim]))
            blk.fc2, blk.b2 = n(3 * dim, dim), Parameter(_Tensor.zeros([dim]))
            blk.gamma = Parameter(_Tensor.ones([dim]) * (1.0 / layers))
        self.ln_w, self.ln_b = Parameter(_Tensor.ones([dim])), Parameter(_Tensor.zeros([dim]))
        self.head_w, self.head_b = n(dim, 2 * nb), Parameter(_Tensor.zeros([2 * nb]))
        self._basis = {}

    def num_params(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())

    @staticmethod
    def istft_basis(n_fft: int) -> np.ndarray:
        """`[2 * (n_fft // 2 + 1), 1, n_fft]`: rows k (real part) and nb + k (imaginary part) of a Hann-windowed inverse real DFT."""
        nb = n_fft // 2 + 1
        t = np.arange(n_fft)
        k = np.arange(nb)[:, None]
        c = np.where((k == 0) | (k == n_fft // 2), 1.0, 2.0) / n_fft
        w = 0.5 - 0.5 * np.cos(2 * np.pi * t / n_fft)
        ang = 2 * np.pi * k * t / n_fft
        return np.concatenate([c * np.cos(ang) * w, -c * np.sin(ang) * w])[:, None, :].astype(np.float32)

    def _inverse(self, re, im) -> _Tensor:
        """`[B, nb, T]` real / imaginary STFT -> `[B, (T - 1) * hop]` (centred frames, Hann OLA normalised)."""
        from . import functional as F
        B, T = int(re.shape[0]), int(re.shape[2])
        dev = str(re.device)
        if dev not in self._basis:
            self._basis[dev] = _from_numpy(self.istft_basis(self.n_fft)).to(re.device)
        y = F.conv_transpose1d(_Tensor.cat([re, im], 1), self._basis[dev], stride=self.hop).reshape([B, -1])
        n, h = self.n_fft, self.hop
        w2 = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)) ** 2
        env = np.zeros((T - 1) * h + n)
        for i in range(T):
            env[i * h:i * h + n] += w2
        env = env[n // 2:n // 2 + (T - 1) * h]
        inv = _from_numpy((1.0 / np.maximum(env, 1e-3)).astype(np.float32)).to(re.device).reshape([1, -1])
        return y.narrow(1, n // 2, (T - 1) * h) * inv

    def forward(self, feats) -> _Tensor:
        from . import functional as F
        B, T = int(feats.shape[0]), int(feats.shape[1])
        D, k = self.dim, self.kernel
        x = feats @ self.in_w + self.in_b
        for blk in self.blocks:
            p = k // 2
            z = x.narrow(1, 0, 1) * 0.0
            hp = _Tensor.cat([z] * p + [x] + [z] * p, 1)
            y = None
            for j in range(k):
                t = hp.narrow(1, j, T) * blk.dw.narrow(0, j, 1).reshape([1, 1, D])
                y = t if y is None else y + t
            y = F.layer_norm(y, [D], blk.ln_w, blk.ln_b)
            x = x + (F.gelu(y @ blk.fc1 + blk.b1) @ blk.fc2 + blk.b2) * blk.gamma
        h = F.layer_norm(x, [D], self.ln_w, self.ln_b) @ self.head_w + self.head_b
        nb = self.n_fft // 2 + 1
        mag = h.narrow(2, 0, nb).clamp(-12.0, 6.0).exp()
        ph = h.narrow(2, nb, nb)
        re, im = (mag * ph.cos()).permute([0, 2, 1]), (mag * ph.sin()).permute([0, 2, 1])
        return self._inverse(re, im)


TransformerLM = MiniGPT

# ---------------------------------------------------------------------------
# 5. FaceFlow: a small text-to-image flow-matching U-Net
# ---------------------------------------------------------------------------

class _FaceRes(Module):
    """GroupNorm -> SiLU -> 3x3 conv, a time-conditioned scale / shift, then a second conv, plus a (1x1) skip."""

    def __init__(self, cin: int, cout: int, tdim: int):
        super().__init__()
        self.n1, self.c1 = GroupNorm(8, cin), Conv2d(cin, cout, 3, padding=1)
        self.t = Linear(tdim, 2 * cout)
        self.n2, self.c2 = GroupNorm(8, cout), Conv2d(cout, cout, 3, padding=1)
        self.skip = Conv2d(cin, cout, 1) if cin != cout else None
        self.cout = cout

    def forward(self, x, temb):
        from . import functional as F
        B, C = int(x.shape[0]), self.cout
        h = self.c1(F.silu(self.n1(x)))
        ss = self.t(F.silu(temb)).reshape([B, 2 * C, 1, 1])
        h = self.n2(h) * (1.0 + ss[:, :C]) + ss[:, C:]
        h = self.c2(F.silu(h))
        return h + (x if self.skip is None else self.skip(x))


class _FaceAttn(Module):
    """Self-attention over the pixels followed by cross-attention to the caption tokens (and a small MLP)."""

    def __init__(self, c: int, text_dim: int, heads: int = 4):
        super().__init__()
        self.heads, self.c = heads, c
        self.n = GroupNorm(8, c)
        self.ln1, self.ln2, self.ln3 = LayerNorm(c), LayerNorm(c), LayerNorm(c)
        self.qkv = Linear(c, 3 * c)
        self.o1 = Linear(c, c)
        self.q2, self.kv2, self.o2 = Linear(c, c), Linear(text_dim, 2 * c), Linear(c, c)
        self.f1, self.f2 = Linear(c, 2 * c), Linear(2 * c, c)

    def _attend(self, q, k, v, mask=None):
        from . import functional as F
        B, Tq, C = int(q.shape[0]), int(q.shape[1]), self.c
        Tk, H = int(k.shape[1]), self.heads
        sp = lambda z, T: z.reshape([B, T, H, C // H]).permute([0, 2, 1, 3])
        a = F.scaled_dot_product_attention(sp(q, Tq), sp(k, Tk), sp(v, Tk), attn_mask=mask)
        return a.permute([0, 2, 1, 3]).reshape([B, Tq, C])

    def forward(self, x, text, key_mask):
        from . import functional as F
        B, C, H, W = (int(d) for d in x.shape)
        t = self.n(x).reshape([B, C, H * W]).permute([0, 2, 1])
        qkv = self.qkv(self.ln1(t))
        t = t + self.o1(self._attend(qkv[:, :, :C], qkv[:, :, C:2 * C], qkv[:, :, 2 * C:]))
        kv = self.kv2(text)
        t = t + self.o2(self._attend(self.q2(self.ln2(t)), kv[:, :, :C], kv[:, :, C:], key_mask))
        t = t + self.f2(F.gelu(self.f1(self.ln3(t))))
        return x + t.permute([0, 2, 1]).reshape([B, C, H, W])


class FaceFlow(Module):
    """A text-to-image rectified-flow model for small (64 x 64) images, about 10 M parameters at the defaults.

    A word-level transformer encodes the caption; a convolutional U-Net (time-conditioned residual blocks, self- and
    cross-attention at the low resolutions) predicts the velocity `v = noise - image` of the straight path
    `x_t = (1 - t) * image + t * noise`. `sample()` integrates it from noise to an image with Euler steps and classifier-free
    guidance (caption token id 1 is the "no caption" null token).

        v = model(x_t, t, tokens, token_mask)         # x_t [B,3,H,W], t [B], tokens int64 [B,N], token_mask bool [B,N]
        img = model.sample(tokens, token_mask, steps=24, guidance=3.0)"""

    def __init__(self, vocab_size: int, channels=(64, 96, 128, 224), attn_levels=(2, 3), text_dim: int = 128, text_layers: int = 3,
                 text_heads: int = 4, max_tokens: int = 48, size: int = 64):
        super().__init__()
        self.vocab_size, self.text_dim, self.text_heads, self.max_tokens, self.size = vocab_size, text_dim, text_heads, max_tokens, size
        self.channels, self.attn_levels = tuple(channels), tuple(attn_levels)
        std, tdim, L = 0.02, 4 * channels[0], len(channels)
        self.tok_embed = Parameter(_Tensor.randn([vocab_size, text_dim]) * std)
        self.pos_embed = Parameter(_Tensor.randn([max_tokens, text_dim]) * std)
        self.text_blocks = ModuleList([_GPTBlock(text_dim, std, std / math.sqrt(2 * text_layers)) for _ in range(text_layers)])
        self.text_ln = LayerNorm(text_dim)
        self.t1, self.t2 = Linear(channels[0], tdim), Linear(tdim, tdim)
        self.stem = Conv2d(3, channels[0], 3, padding=1)
        self.down_res = ModuleList([_FaceRes(channels[i - 1] if i else channels[0], c, tdim) for i, c in enumerate(channels)])
        self.down_attn = ModuleList([_FaceAttn(c, text_dim) for i, c in enumerate(channels) if i in self.attn_levels])
        self.down = ModuleList([Conv2d(c, c, 3, stride=2, padding=1) for c in channels[:-1]])
        cm = channels[-1]
        self.mid1, self.mid_attn, self.mid2 = _FaceRes(cm, cm, tdim), _FaceAttn(cm, text_dim), _FaceRes(cm, cm, tdim)
        self.up_res = ModuleList([_FaceRes((channels[i + 1] if i < L - 1 else cm) + c, c, tdim) for i, c in enumerate(channels)])
        self.up_attn = ModuleList([_FaceAttn(c, text_dim) for i, c in enumerate(channels) if i in self.attn_levels])
        self.up = ModuleList([Conv2d(c, c, 3, padding=1) for c in channels[1:]])
        self.out_norm, self.out = GroupNorm(8, channels[0]), Conv2d(channels[0], 3, 3, padding=1)
        self._cache = {}

    def num_params(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())

    def _consts(self, like):
        key = str(like.device)
        if key not in self._cache:
            half = self.channels[0] // 2
            f = np.exp(-math.log(10000.0) * np.arange(half, dtype=np.float32) / half)[None, :]
            self._cache[key] = _from_numpy(np.ascontiguousarray(f)).to(like.device)
        return self._cache[key]

    def encode_text(self, tokens, token_mask):
        B, N = int(tokens.shape[0]), int(tokens.shape[1])
        x = self.tok_embed.index_select(0, tokens.reshape([-1])).reshape([B, N, self.text_dim]) + self.pos_embed[:N]
        mask = token_mask.reshape([B, 1, 1, N])
        for blk in self.text_blocks:
            x = _block_forward(x, blk, self.text_heads, attn_mask=mask)
        return self.text_ln(x)

    def _attn_index(self, i):
        return sorted(self.attn_levels).index(i)

    def forward(self, x, t, tokens, token_mask):
        from . import functional as F
        B = int(x.shape[0])
        text = self.encode_text(tokens, token_mask)
        key_mask = token_mask.reshape([B, 1, 1, int(token_mask.shape[1])])
        ang = (t * 1000.0).reshape([B, 1]) * self._consts(x)
        temb = self.t2(F.silu(self.t1(_Tensor.cat([ang.cos(), ang.sin()], dim=1))))
        h = self.stem(x)
        skips = []
        for i in range(len(self.channels)):
            h = self.down_res[i](h, temb)
            if i in self.attn_levels:
                h = self.down_attn[self._attn_index(i)](h, text, key_mask)
            skips.append(h)
            if i < len(self.channels) - 1:
                h = self.down[i](h)
        h = self.mid2(self.mid_attn(self.mid1(h, temb), text, key_mask), temb)
        for i in reversed(range(len(self.channels))):
            h = self.up_res[i](_Tensor.cat([h, skips[i]], dim=1), temb)
            if i in self.attn_levels:
                h = self.up_attn[self._attn_index(i)](h, text, key_mask)
            if i > 0:
                h = self.up[i - 1](F.interpolate(h, scale_factor=2.0, mode="nearest"))
        return self.out(F.silu(self.out_norm(h)))

    def loss(self, x0, noise, t, tokens, token_mask):
        """Flow-matching loss: MSE between the predicted and the true velocity (`noise - x0`)."""
        tb = t.reshape([int(t.shape[0]), 1, 1, 1])
        v = self(x0 * (1.0 - tb) + noise * tb, t, tokens, token_mask)
        return ((v - (noise - x0)) ** 2).mean()

    def sample(self, tokens, token_mask, steps: int = 24, guidance: float = 3.0, seed: int = 0):
        """Euler integration from noise (t = 1) to an image (t = 0); `tokens` are int64 `[B, N]` on the model's device."""
        from . import no_grad
        B, dev = int(tokens.shape[0]), tokens.device
        null = _from_numpy(np.ones((B, 1), np.int64)).to(dev)
        null_mask = _from_numpy(np.ones((B, 1), bool)).to(dev)
        rng = np.random.default_rng(seed)
        x = _from_numpy(rng.standard_normal((B, 3, self.size, self.size)).astype(np.float32)).to(dev)
        with no_grad():
            for i in range(steps):
                t0, t1 = 1.0 - i / steps, 1.0 - (i + 1) / steps
                tt = _from_numpy(np.full((B,), t0, np.float32)).to(dev)
                v = self(x, tt, tokens, token_mask)
                if guidance != 1.0:
                    vu = self(x, tt, null, null_mask)
                    v = vu + (v - vu) * guidance
                x = x + v * (t1 - t0)
        return x


__all__ = [
    "ResidualBlock",
    "ResNet18",
    "VisionTransformer",
    "MiniGPT",
    "GPT",
    "TinyTTS",
    "PitchFlow",
    "NeuralVocoder",
    "SourceFilterHead",
    "FaceFlow",
    "TransformerLM",
]
