//! JIT Compiler & Shader Code Generator.
//!
//! Includes:
//! - WebGPU Shading Language (`WGSL`) Compute Shader Synthesizer
//! - Fused Elementwise DAG builder ([`JitKernelBuilder`], [`JitOp`], [`JitNode`])

pub mod builder;

pub use builder::{JitKernelBuilder, JitNode, JitOp};
