//! Quantized Paged KV-Cache for high-efficiency LLM serving.
//!
//! Provides FP8 (`E4M3`), INT8 symmetric, and INT4 packed block-scale KV cache pages,
//! slashing KV memory by 2x to 8x compared to standard FP32/FP16 and enabling 128k+ long context serving.

use crate::error::{Error, Result};
use crate::serving::paged_kv::{BlockTable, PhysicalBlockId};
use crate::tensor::Tensor;
use std::sync::RwLock;

// NOTE: every guard below uses unwrap_or_else(into_inner). A poisoned lock
// (some previous holder panicked) recovers its data instead of panicking
// the reader, matching the poison-recovering helpers used by python.rs.

/// Quantization format for Key and Value cache pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantKVDType {
    /// IEEE-style FP8 (1 sign bit, 4 exponent bits, 3 mantissa bits, bias = 7).
    /// Dynamic range ~[-448, 448], optimal for transformer KV representations.
    Fp8E4M3,
    /// Symmetric 8-bit integer quantization with per-token scale ($d = \max |x| / 127$).
    Int8,
    /// 4-bit packed symmetric quantization (2 elements per byte) with per-token scale ($d = \max |x| / 7$).
    Int4,
}

impl QuantKVDType {
    /// Number of bits per element.
    pub fn bits_per_element(&self) -> usize {
        match self {
            Self::Fp8E4M3 | Self::Int8 => 8,
            Self::Int4 => 4,
        }
    }

    /// Theoretical compression factor relative to FP32.
    pub fn compression_ratio(&self) -> f64 {
        match self {
            Self::Fp8E4M3 | Self::Int8 => 4.0,
            Self::Int4 => 8.0,
        }
    }
}

/// Converts an f32 into an 8-bit FP8 (E4M3) representation.
#[inline]
pub fn f32_to_fp8_e4m3(val: f32) -> u8 {
    if val.is_nan() {
        return 0x7F; // NaN representation
    }
    let bits = val.to_bits();
    let sign = (bits >> 31) & 0x1;
    let exp_f32 = ((bits >> 23) & 0xFF) as i32;
    let mant_f32 = bits & 0x7FFFFF;

    if val == 0.0 {
        return (sign << 7) as u8;
    }

    // FP8 E4M3 has bias 7. F32 has bias 127.
    let exp_fp8 = exp_f32 - 127 + 7;

    if exp_fp8 <= 0 {
        // Subnormal or underflow
        0
    } else if exp_fp8 >= 15 {
        // Overflow clamped to max finite 448.0 (exp = 14, mant = 7)
        let s = (sign << 7) as u8;
        s | 0x7E
    } else {
        // Normal number: take top 3 bits of mantissa with rounding
        let mant_fp8 = ((mant_f32 + (1 << 19)) >> 20) & 0x7;
        let s = (sign << 7) as u8;
        s | ((exp_fp8 as u8) << 3) | (mant_fp8 as u8)
    }
}

/// Converts an 8-bit FP8 (E4M3) byte back to f32.
#[inline]
pub fn fp8_e4m3_to_f32(byte: u8) -> f32 {
    let sign = (byte >> 7) & 0x1;
    let exp = (byte >> 3) & 0xF;
    let mant = byte & 0x7;

    if exp == 0 {
        if mant == 0 {
            if sign == 1 { -0.0 } else { 0.0 }
        } else {
            // Subnormal
            let sign_f = if sign == 1 { -1.0f32 } else { 1.0f32 };
            sign_f * (mant as f32 / 8.0) * (2.0f32).powi(-6)
        }
    } else if exp == 0xF && mant == 0x7 {
        f32::NAN
    } else {
        let sign_f = if sign == 1 { -1.0f32 } else { 1.0f32 };
        let mant_f = 1.0 + (mant as f32 / 8.0);
        let exp_f = (exp as i32) - 7;
        sign_f * mant_f * (2.0f32).powi(exp_f)
    }
}

/// Quantized physical Paged KV-Cache buffer holding memory pages across physical blocks.
/// Computes a token byte span, rejecting arithmetic overflow and spans
/// past the actual stores. Dims are public fields, so a mutated cache
/// must produce Err here instead of an out-of-bounds panic below.
fn token_span(token_idx: usize, per_token: usize, cap_k: usize, cap_v: usize) -> Result<usize> {
    let start = token_idx.checked_mul(per_token).ok_or(Error::Msg(
        "QuantizedPagedKVCache: token offset overflows".into(),
    ))?;
    let end = start.checked_add(per_token).ok_or(Error::Msg(
        "QuantizedPagedKVCache: token offset overflows".into(),
    ))?;
    if end > cap_k {
        return Err(Error::Msg(
            "QuantizedPagedKVCache: token offset exceeds key store".into(),
        ));
    }
    if end > cap_v {
        return Err(Error::Msg(
            "QuantizedPagedKVCache: token offset exceeds value store".into(),
        ));
    }
    Ok(start)
}

pub struct QuantizedPagedKVCache {
    pub num_blocks: usize,
    pub block_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub dtype: QuantKVDType,
    // Storage for Key & Value blocks
    k_data: RwLock<Vec<u8>>,
    v_data: RwLock<Vec<u8>>,
    // Per-token scaling factors for INT8 and INT4
    k_scales: RwLock<Vec<f32>>,
    v_scales: RwLock<Vec<f32>>,
}

impl QuantizedPagedKVCache {
    /// Creates a new quantized KV cache buffer.
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
        dtype: QuantKVDType,
    ) -> Result<Self> {
        let total_tokens = crate::layout::checked_numel(&[num_blocks, block_size])?;
        let elements_per_token = crate::layout::checked_numel(&[num_heads, head_dim])?;

        let bytes_per_token = match dtype {
            QuantKVDType::Fp8E4M3 | QuantKVDType::Int8 => elements_per_token,
            QuantKVDType::Int4 => elements_per_token.div_ceil(2),
        };

        let total_bytes = total_tokens
            .checked_mul(bytes_per_token)
            .ok_or_else(|| Error::Msg("QuantizedPagedKVCache size overflows usize".to_string()))?;
        let total_scales = total_tokens;

        Ok(Self {
            num_blocks,
            block_size,
            num_heads,
            head_dim,
            dtype,
            k_data: RwLock::new(crate::layout::try_filled(total_bytes, 0u8)?),
            v_data: RwLock::new(crate::layout::try_filled(total_bytes, 0u8)?),
            k_scales: RwLock::new(crate::layout::try_filled(total_scales, 1.0f32)?),
            v_scales: RwLock::new(crate::layout::try_filled(total_scales, 1.0f32)?),
        })
    }

    /// Total memory consumption in bytes.
    pub fn memory_usage_bytes(&self) -> usize {
        let k_bytes = self.k_data.read().unwrap_or_else(|e| e.into_inner()).len();
        let v_bytes = self.v_data.read().unwrap_or_else(|e| e.into_inner()).len();
        let scale_bytes = (self
            .k_scales
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .len()
            + self
                .v_scales
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .len())
            * 4;
        k_bytes + v_bytes + scale_bytes
    }

    /// Effective compression ratio compared to an uncompressed FP32 KV cache.
    pub fn compression_ratio(&self) -> f64 {
        let uncompressed_bytes = self.num_blocks as f64
            * self.block_size as f64
            * self.num_heads as f64
            * self.head_dim as f64
            * 8.0;
        let actual_bytes = self.memory_usage_bytes();
        uncompressed_bytes / actual_bytes.max(1) as f64
    }

    /// Writes key and value vectors for a single token at position `offset_in_block` inside `block_id`.
    pub fn write_token(
        &self,
        block_id: PhysicalBlockId,
        offset_in_block: usize,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<()> {
        if block_id >= self.num_blocks || offset_in_block >= self.block_size {
            return Err(Error::Msg(format!(
                "QuantizedPagedKVCache write out of bounds: block={block_id}, offset={offset_in_block}"
            )));
        }

        let k_vec = key.to_vec::<f32>()?;
        let v_vec = value.to_vec::<f32>()?;
        let expected_len = self.num_heads.checked_mul(self.head_dim).ok_or(Error::Msg(
            "QuantizedPagedKVCache: head dims overflow".into(),
        ))?;

        if k_vec.len() < expected_len || v_vec.len() < expected_len {
            return Err(Error::Msg(format!(
                "Key/Value tensor element count ({}) smaller than expected ({expected_len})",
                k_vec.len().min(v_vec.len())
            )));
        }

        let token_idx = block_id
            .checked_mul(self.block_size)
            .and_then(|v| v.checked_add(offset_in_block))
            .ok_or(Error::Msg(
                "QuantizedPagedKVCache: block index overflows".into(),
            ))?;

        match self.dtype {
            QuantKVDType::Fp8E4M3 => {
                let mut k_store = self.k_data.write().unwrap_or_else(|e| e.into_inner());
                let mut v_store = self.v_data.write().unwrap_or_else(|e| e.into_inner());
                let start = token_span(token_idx, expected_len, k_store.len(), v_store.len())?;
                for i in 0..expected_len {
                    k_store[start + i] = f32_to_fp8_e4m3(k_vec[i]);
                    v_store[start + i] = f32_to_fp8_e4m3(v_vec[i]);
                }
            }
            QuantKVDType::Int8 => {
                let max_k = k_vec[..expected_len]
                    .iter()
                    .fold(0.0f32, |m, &x| m.max(x.abs()))
                    .max(1e-8);
                let max_v = v_vec[..expected_len]
                    .iter()
                    .fold(0.0f32, |m, &x| m.max(x.abs()))
                    .max(1e-8);
                let scale_k = max_k / 127.0;
                let scale_v = max_v / 127.0;

                {
                    let mut ks = self.k_scales.write().unwrap_or_else(|e| e.into_inner());
                    let mut vs = self.v_scales.write().unwrap_or_else(|e| e.into_inner());
                    *ks.get_mut(token_idx).ok_or(Error::Msg(
                        "QuantizedPagedKVCache: token exceeds scale store".into(),
                    ))? = scale_k;
                    *vs.get_mut(token_idx).ok_or(Error::Msg(
                        "QuantizedPagedKVCache: token exceeds scale store".into(),
                    ))? = scale_v;
                }

                let mut k_store = self.k_data.write().unwrap_or_else(|e| e.into_inner());
                let mut v_store = self.v_data.write().unwrap_or_else(|e| e.into_inner());
                let start = token_span(token_idx, expected_len, k_store.len(), v_store.len())?;
                for i in 0..expected_len {
                    let qk = (k_vec[i] / scale_k).round().clamp(-127.0, 127.0) as i8;
                    let qv = (v_vec[i] / scale_v).round().clamp(-127.0, 127.0) as i8;
                    k_store[start + i] = qk as u8;
                    v_store[start + i] = qv as u8;
                }
            }
            QuantKVDType::Int4 => {
                let max_k = k_vec[..expected_len]
                    .iter()
                    .fold(0.0f32, |m, &x| m.max(x.abs()))
                    .max(1e-8);
                let max_v = v_vec[..expected_len]
                    .iter()
                    .fold(0.0f32, |m, &x| m.max(x.abs()))
                    .max(1e-8);
                let scale_k = max_k / 7.0;
                let scale_v = max_v / 7.0;

                {
                    let mut ks = self.k_scales.write().unwrap_or_else(|e| e.into_inner());
                    let mut vs = self.v_scales.write().unwrap_or_else(|e| e.into_inner());
                    *ks.get_mut(token_idx).ok_or(Error::Msg(
                        "QuantizedPagedKVCache: token exceeds scale store".into(),
                    ))? = scale_k;
                    *vs.get_mut(token_idx).ok_or(Error::Msg(
                        "QuantizedPagedKVCache: token exceeds scale store".into(),
                    ))? = scale_v;
                }

                let mut k_store = self.k_data.write().unwrap_or_else(|e| e.into_inner());
                let mut v_store = self.v_data.write().unwrap_or_else(|e| e.into_inner());
                let bytes_per_tok = expected_len.div_ceil(2);
                let start = token_span(token_idx, bytes_per_tok, k_store.len(), v_store.len())?;

                for pair_idx in 0..bytes_per_tok {
                    let i0 = pair_idx * 2;
                    let i1 = i0 + 1;

                    let qk0 = (k_vec[i0] / scale_k).round().clamp(-7.0, 7.0) as i8;
                    let qk1 = if i1 < expected_len {
                        (k_vec[i1] / scale_k).round().clamp(-7.0, 7.0) as i8
                    } else {
                        0
                    };
                    let byte_k = (((qk0 + 8) as u8 & 0x0F) << 4) | ((qk1 + 8) as u8 & 0x0F);
                    k_store[start + pair_idx] = byte_k;

                    let qv0 = (v_vec[i0] / scale_v).round().clamp(-7.0, 7.0) as i8;
                    let qv1 = if i1 < expected_len {
                        (v_vec[i1] / scale_v).round().clamp(-7.0, 7.0) as i8
                    } else {
                        0
                    };
                    let byte_v = (((qv0 + 8) as u8 & 0x0F) << 4) | ((qv1 + 8) as u8 & 0x0F);
                    v_store[start + pair_idx] = byte_v;
                }
            }
        }

        Ok(())
    }

    /// Dequantizes Key and Value representations for a single token at `(block_id, offset_in_block)`.
    pub fn read_token_dequantized(
        &self,
        block_id: PhysicalBlockId,
        offset_in_block: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        if block_id >= self.num_blocks || offset_in_block >= self.block_size {
            return Err(Error::Msg(format!(
                "QuantizedPagedKVCache read out of bounds: block={block_id}, offset={offset_in_block}"
            )));
        }
        let expected_len = self.num_heads.checked_mul(self.head_dim).ok_or(Error::Msg(
            "QuantizedPagedKVCache: head dims overflow".into(),
        ))?;
        let token_idx = block_id
            .checked_mul(self.block_size)
            .and_then(|v| v.checked_add(offset_in_block))
            .ok_or(Error::Msg(
                "QuantizedPagedKVCache: block index overflows".into(),
            ))?;
        let mut k_out = vec![0.0f32; expected_len];
        let mut v_out = vec![0.0f32; expected_len];

        match self.dtype {
            QuantKVDType::Fp8E4M3 => {
                let k_store = self.k_data.read().unwrap_or_else(|e| e.into_inner());
                let v_store = self.v_data.read().unwrap_or_else(|e| e.into_inner());
                let start = token_span(token_idx, expected_len, k_store.len(), v_store.len())?;
                for i in 0..expected_len {
                    k_out[i] = fp8_e4m3_to_f32(k_store[start + i]);
                    v_out[i] = fp8_e4m3_to_f32(v_store[start + i]);
                }
            }
            QuantKVDType::Int8 => {
                let scale_k = *self
                    .k_scales
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .get(token_idx)
                    .ok_or(Error::Msg(
                        "QuantizedPagedKVCache: token exceeds scale store".into(),
                    ))?;
                let scale_v = *self
                    .v_scales
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .get(token_idx)
                    .ok_or(Error::Msg(
                        "QuantizedPagedKVCache: token exceeds scale store".into(),
                    ))?;
                let k_store = self.k_data.read().unwrap_or_else(|e| e.into_inner());
                let v_store = self.v_data.read().unwrap_or_else(|e| e.into_inner());
                let start = token_span(token_idx, expected_len, k_store.len(), v_store.len())?;
                for i in 0..expected_len {
                    k_out[i] = (k_store[start + i] as i8 as f32) * scale_k;
                    v_out[i] = (v_store[start + i] as i8 as f32) * scale_v;
                }
            }
            QuantKVDType::Int4 => {
                let scale_k = *self
                    .k_scales
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .get(token_idx)
                    .ok_or(Error::Msg(
                        "QuantizedPagedKVCache: token exceeds scale store".into(),
                    ))?;
                let scale_v = *self
                    .v_scales
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .get(token_idx)
                    .ok_or(Error::Msg(
                        "QuantizedPagedKVCache: token exceeds scale store".into(),
                    ))?;
                let k_store = self.k_data.read().unwrap_or_else(|e| e.into_inner());
                let v_store = self.v_data.read().unwrap_or_else(|e| e.into_inner());
                let bytes_per_tok = expected_len.div_ceil(2);
                let start = token_span(token_idx, bytes_per_tok, k_store.len(), v_store.len())?;

                for pair_idx in 0..bytes_per_tok {
                    let byte_k = k_store[start + pair_idx];
                    let byte_v = v_store[start + pair_idx];

                    let i0 = pair_idx * 2;
                    let i1 = i0 + 1;

                    let qk0 = ((byte_k >> 4) & 0x0F) as i8 - 8;
                    let qk1 = (byte_k & 0x0F) as i8 - 8;
                    k_out[i0] = (qk0 as f32) * scale_k;
                    if i1 < expected_len {
                        k_out[i1] = (qk1 as f32) * scale_k;
                    }

                    let qv0 = ((byte_v >> 4) & 0x0F) as i8 - 8;
                    let qv1 = (byte_v & 0x0F) as i8 - 8;
                    v_out[i0] = (qv0 as f32) * scale_v;
                    if i1 < expected_len {
                        v_out[i1] = (qv1 as f32) * scale_v;
                    }
                }
            }
        }

        Ok((k_out, v_out))
    }
}

/// Evaluates PagedAttention directly over a Quantized Paged KV Cache.
///
/// Queries: `[batch_size, num_heads, head_dim]` (current decode step).
/// Returns: `[batch_size, num_heads, head_dim]` attention output tensor.
pub fn paged_attention_quantized(
    queries: &Tensor,
    cache: &QuantizedPagedKVCache,
    block_tables: &[BlockTable],
    seq_lens: &[usize],
    scale: f64,
) -> Result<Tensor> {
    let qshape = queries.shape().to_vec();
    if qshape.len() != 3 || qshape[1] != cache.num_heads || qshape[2] != cache.head_dim {
        return Err(Error::ShapeMismatch {
            op: "paged_attention_quantized",
            expected: vec![
                qshape.first().copied().unwrap_or(0),
                cache.num_heads,
                cache.head_dim,
            ],
            got: qshape,
        });
    }
    let batch_size = queries.shape()[0];
    if block_tables.len() != batch_size || seq_lens.len() != batch_size {
        return Err(Error::Msg(format!(
            "paged_attention_quantized: batch {batch_size} needs {batch_size} block tables and seq_lens, got {} and {}",
            block_tables.len(),
            seq_lens.len()
        )));
    }
    let num_heads = cache.num_heads;
    let head_dim = cache.head_dim;
    let b_size = cache.block_size;
    if b_size == 0 {
        return Err(Error::Msg(
            "paged_attention_quantized: block_size must be positive".into(),
        ));
    }

    let q_data = queries.to_vec::<f32>()?;
    // batch*heads*dim == queries numel (validated above), so this cannot overflow.
    let mut out_data = vec![0.0f32; batch_size * num_heads * head_dim];

    for b in 0..batch_size {
        let seq_len = seq_lens[b];
        if seq_len == 0 {
            continue;
        }
        let bt = &block_tables[b];
        let need_blocks = seq_len.div_ceil(b_size);
        if bt.physical_blocks.len() < need_blocks {
            return Err(Error::Msg(format!(
                "paged_attention_quantized: sequence {b} needs {need_blocks} blocks for {seq_len} tokens, table has {}",
                bt.physical_blocks.len()
            )));
        }

        for h in 0..num_heads {
            let q_offset = (b * num_heads + h) * head_dim;
            let q_head = &q_data[q_offset..q_offset + head_dim];

            // 1. Compute dot-product attention scores across all tokens in sequence
            let mut scores = Vec::with_capacity(seq_len);
            let mut v_heads = Vec::with_capacity(seq_len);

            for t in 0..seq_len {
                let logical_block = t / b_size;
                let offset_in_block = t % b_size;
                let phys_block = bt.physical_blocks[logical_block];

                let (k_tok, v_tok) = cache.read_token_dequantized(phys_block, offset_in_block)?;
                let head_offset = h * head_dim;
                let k_head = &k_tok[head_offset..head_offset + head_dim];
                let v_head = v_tok[head_offset..head_offset + head_dim].to_vec();

                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_head[d] * k_head[d];
                }
                scores.push((dot as f64 * scale) as f32);
                v_heads.push(v_head);
            }

            // 2. Numerically stable softmax
            let max_score = scores.iter().fold(f32::NEG_INFINITY, |m, &s| m.max(s));
            let mut sum_exp = 0.0f32;
            let mut exp_scores = Vec::with_capacity(seq_len);
            for &s in &scores {
                let e = (s - max_score).exp();
                exp_scores.push(e);
                sum_exp += e;
            }
            let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };

            // 3. Weighted sum over values
            let out_offset = (b * num_heads + h) * head_dim;
            for (t, &e) in exp_scores.iter().enumerate() {
                let w = e * inv_sum;
                for d in 0..head_dim {
                    out_data[out_offset + d] += w * v_heads[t][d];
                }
            }
        }
    }

    Tensor::from_slice_on(&out_data, &qshape, &queries.device())
}
