//! Adaptive Compute & Dynamic Routing Engine.
//!
//! Includes:
//! - Mixture-of-Depths ([`MixtureOfDepths`]) for dynamic token routing and compute pruning
//! - Early-Exit Inference Router ([`EarlyExitRouter`])

pub mod autonomous_quant;
pub mod early_exit;
pub mod elastic;
pub mod mod_routing;
pub mod nas;

pub use autonomous_quant::{
    AutonomousQuantPolicy, AutonomousQuantizer, BlockQuantProfile, QuantFormat, QuantizedCausalLM,
    QuantizedLMBlock, QuantizedLayer,
};
pub use early_exit::EarlyExitRouter;
pub use elastic::ElasticSubnetExtractor;
pub use mod_routing::MixtureOfDepths;
pub use nas::{ArchitectureGenome, EvolutionaryAutoML, ZeroCostScore, compute_synflow};
