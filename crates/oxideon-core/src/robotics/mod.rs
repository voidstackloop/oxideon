//! Multimodal Vision-Language-Action & Robotics Control Engine.
//!
//! Includes:
//! - Action Chunking Transformer ([`ActionChunkingTransformer`], [`ACTConfig`]) with temporal ensembling
//! - Continuous Flow Matching Robotics Policy ([`FlowMatchingRoboticsPolicy`])

pub mod act;
pub mod flow_policy;

pub use act::{ACTConfig, ActionChunkingTransformer};
pub use flow_policy::FlowMatchingRoboticsPolicy;
