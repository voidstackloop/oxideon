//! Frontier Long-Context & Context Scaling Suite.
//!
//! Includes:
//! - FlashAttention-3 ([`flash_attention_3`], [`FlashAttention3Config`]) with asynchronous tiling & logit soft-capping
//! - Distributed RingAttention ([`RingAttention`]) for 1M+ token context sequences
//! - Chunked Prefill Engine ([`ChunkedPrefillEngine`])

pub mod chunked_prefill;
pub mod flash_attn2;
pub mod flash_attn3;
pub mod flash_decoding;
pub mod holographic;
pub mod ring_attn;

pub use chunked_prefill::ChunkedPrefillEngine;
pub use flash_attn2::{
    FlashAttention2Config, FlashAttention2ForwardOutput, flash_attention_2,
    flash_attention_2_backward, flash_attention_2_forward,
};
pub use flash_attn3::{FlashAttention3Config, flash_attention_3};
pub use flash_decoding::{FlashDecodingConfig, FlashDecodingEngine};
pub use holographic::HolographicMemoryEngine;
pub use ring_attn::RingAttention;
