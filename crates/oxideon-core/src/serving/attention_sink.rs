//! StreamingLLM: Efficient Streaming Language Models with Attention Sinks (Xiao et al.).
//!
//! When sliding window attention drops initial prompt tokens, softmax normalization collapses
//! and perplexity explodes because the initial tokens act as "attention sinks" absorbing
//! large baseline attention scores.
//!
//! StreamingLLM preserves the first $S$ sink tokens (typically 4) alongside the rolling window
//! of recent $W$ tokens, enabling infinite-length continuous streaming inference with fixed $O(S + W)$ memory.

use crate::error::{Error, Result};
use crate::tensor::Tensor;
use std::collections::VecDeque;

/// Configuration for StreamingLLM Attention Sink Cache.
#[derive(Debug, Clone)]
pub struct AttentionSinkConfig {
    /// Number of initial sink tokens to preserve forever (typically 4).
    pub num_sink_tokens: usize,
    /// Size of the recent sliding window.
    pub window_size: usize,
}

impl AttentionSinkConfig {
    pub fn new(num_sink_tokens: usize, window_size: usize) -> Self {
        Self {
            num_sink_tokens: num_sink_tokens.max(1),
            window_size: window_size.max(4),
        }
    }
}

/// Streaming KV-Cache that keeps initial attention sinks and evicts middle tokens.
#[derive(Debug, Clone)]
pub struct StreamingKVCache {
    pub config: AttentionSinkConfig,
    sink_k: Vec<Tensor>,
    sink_v: Vec<Tensor>,
    window_k: VecDeque<Tensor>,
    window_v: VecDeque<Tensor>,
    total_tokens_seen: usize,
}

impl StreamingKVCache {
    pub fn new(config: AttentionSinkConfig) -> Self {
        Self {
            config,
            sink_k: Vec::new(),
            sink_v: Vec::new(),
            window_k: VecDeque::new(),
            window_v: VecDeque::new(),
            total_tokens_seen: 0,
        }
    }

    /// Appends a new token's Key and Value tensors.
    ///
    /// Expected shape: `[batch_size, num_heads, 1, head_dim]`.
    pub fn append(&mut self, key: &Tensor, value: &Tensor) -> Result<()> {
        if self.sink_k.len() < self.config.num_sink_tokens {
            self.sink_k.push(key.clone());
            self.sink_v.push(value.clone());
        } else {
            if self.window_k.len() >= self.config.window_size {
                self.window_k.pop_front();
                self.window_v.pop_front();
            }
            self.window_k.push_back(key.clone());
            self.window_v.push_back(value.clone());
        }

        self.total_tokens_seen += 1;
        Ok(())
    }

    /// Returns concatenated (Keys, Values) for the current attention computation:
    /// `[batch_size, num_heads, num_cached, head_dim]` where `num_cached <= num_sinks + window_size`.
    pub fn get_kv(&self) -> Result<(Tensor, Tensor)> {
        let mut all_k = Vec::with_capacity(self.sink_k.len() + self.window_k.len());
        let mut all_v = Vec::with_capacity(self.sink_v.len() + self.window_v.len());

        for k in &self.sink_k {
            all_k.push(k.clone());
        }
        for k in &self.window_k {
            all_k.push(k.clone());
        }

        for v in &self.sink_v {
            all_v.push(v.clone());
        }
        for v in &self.window_v {
            all_v.push(v.clone());
        }

        if all_k.is_empty() {
            return Err(Error::Msg("StreamingKVCache is empty".into()));
        }

        // Concatenate along sequence dimension (dim 2)
        let keys_tensor = Tensor::cat(&all_k, 2)?;
        let values_tensor = Tensor::cat(&all_v, 2)?;

        Ok((keys_tensor, values_tensor))
    }

    /// Total count of tokens passed into this streaming session.
    pub fn total_tokens_seen(&self) -> usize {
        self.total_tokens_seen
    }

    /// Current number of physical tokens retained in active KV memory.
    pub fn num_cached_tokens(&self) -> usize {
        self.sink_k.len() + self.window_k.len()
    }
}
