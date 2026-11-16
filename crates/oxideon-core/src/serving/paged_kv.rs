//! PagedAttention: Non-contiguous KV-cache memory management inspired by vLLM (Kwon et al., SOSP 2023).
//!
//! Replaces contiguous monolithic KV buffers with fixed-size physical memory pages (`block_size` tokens),
//! reducing memory fragmentation from ~60-80% down to under 4% and enabling shared prefixes / beam search
//! through copy-on-write page sharing.

use std::collections::{HashMap, VecDeque};

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::{Tensor, no_grad};

pub type PhysicalBlockId = usize;

/// Allocates and manages physical KV-cache memory blocks.
#[derive(Debug)]
pub struct BlockAllocator {
    pub num_blocks: usize,
    pub block_size: usize,
    free_blocks: VecDeque<PhysicalBlockId>,
    ref_counts: HashMap<PhysicalBlockId, usize>,
    allocated_count: usize,
}

impl BlockAllocator {
    /// Creates a new allocator managing `num_blocks` physical blocks, each storing `block_size` tokens.
    /// block_size must be positive; schedule_step and ServingEngine::new
    /// return Err for a zero-sized allocator instead of dividing by it.
    pub fn new(num_blocks: usize, block_size: usize) -> Result<Self> {
        // Same ceiling as ServingEngine::new: the freelists below are dense.
        crate::limits::check_at_most("BlockAllocator num_blocks", num_blocks, 1 << 24)?;
        let mut free = VecDeque::with_capacity(num_blocks);
        for id in 0..num_blocks {
            free.push_back(id);
        }
        Ok(Self {
            num_blocks,
            block_size,
            free_blocks: free,
            ref_counts: HashMap::with_capacity(num_blocks),
            allocated_count: 0,
        })
    }

    /// Allocates one free physical block.
    pub fn allocate(&mut self) -> Result<PhysicalBlockId> {
        let block_id = self
            .free_blocks
            .pop_front()
            .ok_or_else(|| Error::Msg("PagedKV: Out of physical KV blocks".into()))?;
        self.ref_counts.insert(block_id, 1);
        self.allocated_count += 1;
        Ok(block_id)
    }

    /// Increments reference count for Copy-on-Write sharing (e.g. beam search / prefix caching).
    pub fn inc_ref(&mut self, block_id: PhysicalBlockId) {
        if let Some(rc) = self.ref_counts.get_mut(&block_id) {
            *rc += 1;
        }
    }

    /// Decrements reference count and reclaims block when ref count drops to zero.
    pub fn free(&mut self, block_id: PhysicalBlockId) {
        if let Some(rc) = self.ref_counts.get_mut(&block_id) {
            if *rc > 1 {
                *rc -= 1;
            } else {
                self.ref_counts.remove(&block_id);
                self.free_blocks.push_back(block_id);
                self.allocated_count = self.allocated_count.saturating_sub(1);
            }
        }
    }

    /// How many owners (sequences, caches) currently hold `block_id`; 0 when it is free.
    pub fn ref_count(&self, block_id: PhysicalBlockId) -> usize {
        self.ref_counts.get(&block_id).copied().unwrap_or(0)
    }

    /// Returns the number of currently free physical blocks.
    pub fn num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    /// Memory utilization ratio $\in [0, 1]$.
    pub fn utilization(&self) -> f64 {
        if self.num_blocks == 0 {
            0.0
        } else {
            self.allocated_count as f64 / self.num_blocks as f64
        }
    }
}

/// Logical sequence to physical block mapping table for a single sequence.
#[derive(Debug, Clone, Default)]
pub struct BlockTable {
    pub physical_blocks: Vec<PhysicalBlockId>,
    pub num_tokens: usize,
}

impl BlockTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn append_block(&mut self, block_id: PhysicalBlockId) {
        self.physical_blocks.push(block_id);
    }
}

/// Physical Key-Value memory cache holding non-contiguous block pages.
/// Storage layout: `[num_blocks, block_size, num_heads, head_dim]`.
pub struct PagedKVCache {
    pub key_cache: Tensor,
    pub val_cache: Tensor,
    pub num_blocks: usize,
    pub block_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl PagedKVCache {
    /// Initializes physical memory buffers for `num_blocks` on `device`.
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
        dtype: DType,
    ) -> Result<Self> {
        let shape = [num_blocks, block_size, num_heads, head_dim];
        let key_cache = Tensor::zeros(&shape, dtype, &crate::backend::Device::Cpu)?;
        let val_cache = Tensor::zeros(&shape, dtype, &crate::backend::Device::Cpu)?;
        Ok(Self {
            key_cache,
            val_cache,
            num_blocks,
            block_size,
            num_heads,
            head_dim,
        })
    }

    /// Self-consistency guard: dims are public and hand-tamperable, and both
    /// `write_token` and `paged_attention` stride by them (overflow on giant
    /// values, silent mis-strides on wrong-but-fitting ones).
    fn check_layout(&self) -> Result<()> {
        let expect = [
            self.num_blocks,
            self.block_size,
            self.num_heads,
            self.head_dim,
        ];
        for (name, t) in [
            ("key_cache", &self.key_cache),
            ("val_cache", &self.val_cache),
        ] {
            if t.shape() != expect {
                return Err(Error::Msg(format!(
                    "PagedKVCache: {name} shape {:?} disagrees with dims {expect:?}",
                    t.shape()
                )));
            }
        }
        Ok(())
    }

    /// Writes key and value representations for a single token at sequence position `seq_pos`
    /// into the physical block `block_id`.
    pub fn write_token(
        &self,
        block_id: PhysicalBlockId,
        offset_in_block: usize,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<()> {
        self.check_layout()?;
        if block_id >= self.num_blocks {
            return Err(Error::IndexOutOfBounds {
                op: "PagedKVCache::write_token",
                index: block_id as isize,
                dim_size: self.num_blocks,
            });
        }
        if offset_in_block >= self.block_size {
            return Err(Error::IndexOutOfBounds {
                op: "PagedKVCache::write_token",
                index: offset_in_block as isize,
                dim_size: self.block_size,
            });
        }
        let key_vec = key.to_vec::<f32>()?;
        let val_vec = value.to_vec::<f32>()?;
        let stride = self.num_heads * self.head_dim;
        if key_vec.len() != stride || val_vec.len() != stride {
            return Err(Error::Msg(format!(
                "PagedKVCache::write_token: key/value must hold one token ({} elements), got {}/{}",
                stride,
                key_vec.len(),
                val_vec.len()
            )));
        }
        let token_idx = block_id
            .checked_mul(self.block_size)
            .and_then(|v| v.checked_add(offset_in_block))
            .ok_or(Error::Msg(
                "PagedKVCache::write_token: block index overflows".into(),
            ))?;
        let global_offset = token_idx.checked_mul(stride).ok_or(Error::Msg(
            "PagedKVCache::write_token: token offset overflows".into(),
        ))?;

        // In-place assignment into physical backing memory
        no_grad(|| {
            let mut k_data = self.key_cache.to_vec::<f32>()?;
            let mut v_data = self.val_cache.to_vec::<f32>()?;
            let len = key_vec.len().min(stride);
            let end = global_offset
                .checked_add(len)
                .ok_or(Error::IndexOutOfBounds {
                    op: "PagedKVCache::write_token",
                    index: global_offset as isize,
                    dim_size: k_data.len(),
                })?;
            if end > k_data.len() {
                return Err(Error::IndexOutOfBounds {
                    op: "PagedKVCache::write_token",
                    index: global_offset as isize,
                    dim_size: k_data.len(),
                });
            }
            if end > v_data.len() {
                return Err(Error::IndexOutOfBounds {
                    op: "PagedKVCache::write_token",
                    index: global_offset as isize,
                    dim_size: v_data.len(),
                });
            }
            k_data[global_offset..end].copy_from_slice(&key_vec[..len]);
            v_data[global_offset..end].copy_from_slice(&val_vec[..len]);
            let updated_k = Tensor::from_slice(&k_data, self.key_cache.shape())?;
            let updated_v = Tensor::from_slice(&v_data, self.val_cache.shape())?;
            self.key_cache.assign(&updated_k)?;
            self.val_cache.assign(&updated_v)?;
            Ok::<(), Error>(())
        })
    }
}

/// Evaluates multi-head PagedAttention over physical block tables without flattening
/// or allocating contiguous full-sequence KV buffers.
///
/// - `queries`: `[batch_size, num_heads, head_dim]` (current decode step)
/// - `cache`: Physical block KV cache
/// - `block_tables`: Slice of block tables (one per sequence in batch)
/// - `seq_lens`: Current context length for each sequence in batch
pub fn paged_attention(
    queries: &Tensor,
    cache: &PagedKVCache,
    block_tables: &[BlockTable],
    seq_lens: &[usize],
    scale: f64,
) -> Result<Tensor> {
    cache.check_layout()?;
    let qshape = queries.shape().to_vec();
    if qshape.len() != 3 || qshape[1] != cache.num_heads || qshape[2] != cache.head_dim {
        return Err(Error::ShapeMismatch {
            op: "paged_attention",
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
            "paged_attention: batch {batch_size} needs {batch_size} block tables and seq_lens, got {} and {}",
            block_tables.len(),
            seq_lens.len()
        )));
    }
    let num_heads = cache.num_heads;
    let head_dim = cache.head_dim;
    let b_size = cache.block_size;
    if b_size == 0 {
        return Err(Error::Msg(
            "paged_attention: block_size must be positive".into(),
        ));
    }

    let q_data = queries.to_vec::<f32>()?;
    let k_data = cache.key_cache.to_vec::<f32>()?;
    let v_data = cache.val_cache.to_vec::<f32>()?;

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
                "paged_attention: sequence {b} needs {need_blocks} blocks for {seq_len} tokens, table has {}",
                bt.physical_blocks.len()
            )));
        }

        for h in 0..num_heads {
            let q_offset = (b * num_heads + h) * head_dim;
            let q_vec = &q_data[q_offset..q_offset + head_dim];

            // Compute attention scores against all tokens across physical blocks
            let mut scores = Vec::with_capacity(seq_len);
            for t in 0..seq_len {
                let logical_block_idx = t / b_size;
                let offset_in_block = t % b_size;
                let phys_block = bt.physical_blocks[logical_block_idx];
                if phys_block >= cache.num_blocks {
                    return Err(Error::IndexOutOfBounds {
                        op: "paged_attention",
                        index: phys_block as isize,
                        dim_size: cache.num_blocks,
                    });
                }

                let tok = phys_block
                    .checked_mul(b_size)
                    .and_then(|v| v.checked_add(offset_in_block))
                    .ok_or(Error::Msg("paged_attention: block index overflows".into()))?;
                let kv_offset = tok
                    .checked_mul(num_heads)
                    .and_then(|v| v.checked_add(h))
                    .and_then(|v| v.checked_mul(head_dim))
                    .ok_or(Error::Msg("paged_attention: kv offset overflows".into()))?;
                let k_vec = k_data.get(kv_offset..kv_offset + head_dim).ok_or_else(|| {
                    Error::Msg("paged_attention: KV offset out of cache bounds".into())
                })?;

                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_vec[d] * k_vec[d];
                }
                scores.push((dot as f64 * scale) as f32);
            }

            // Softmax over sequence
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            let mut exp_scores = Vec::with_capacity(seq_len);
            for &sc in &scores {
                let e = (sc - max_score).exp();
                exp_scores.push(e);
                sum_exp += e;
            }
            let inv_sum = 1.0 / sum_exp.max(1e-12);

            // Weighted sum of values
            let out_offset = (b * num_heads + h) * head_dim;
            for t in 0..seq_len {
                let alpha = exp_scores[t] * inv_sum;
                let logical_block_idx = t / b_size;
                let offset_in_block = t % b_size;
                let phys_block = bt.physical_blocks[logical_block_idx];
                let tok = phys_block
                    .checked_mul(b_size)
                    .and_then(|v| v.checked_add(offset_in_block))
                    .ok_or(Error::Msg("paged_attention: block index overflows".into()))?;
                let kv_offset = tok
                    .checked_mul(num_heads)
                    .and_then(|v| v.checked_add(h))
                    .and_then(|v| v.checked_mul(head_dim))
                    .ok_or(Error::Msg("paged_attention: kv offset overflows".into()))?;
                let v_vec = v_data.get(kv_offset..kv_offset + head_dim).ok_or_else(|| {
                    Error::Msg("paged_attention: KV offset out of cache bounds".into())
                })?;

                for d in 0..head_dim {
                    out_data[out_offset + d] += alpha * v_vec[d];
                }
            }
        }
    }

    Tensor::from_slice(&out_data, &[batch_size, num_heads, head_dim])
}
