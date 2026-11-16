//! Autonomous Reasoning and Tree Search Engine.
//!
//! Includes:
//! - Monte Carlo Tree Search ([`MCTSSearchEngine`], [`MCTSNode`], [`SearchResult`])
//! - Self-Correction Verification ([`SelfCorrectionVerifier`], [`ReasoningTrace`], [`ReasoningStep`], [`VerificationReport`])

pub mod algorithmic;
pub mod causal;
pub mod conformal;
pub mod conformal_risk;
pub mod inductive_logic;
pub mod mcts;
pub mod prover;
pub mod swarm;
pub mod symbolic_gp;
pub mod topos;
pub mod verifier;

pub use algorithmic::{
    AlgorithmicEdge, AlgorithmicGraph, AlgorithmicSemiring, NeuralBellmanFord, PointerGraphNetwork,
};
pub use causal::{CausalVariable, StructuralCausalModel};
pub use conformal::{
    AdaptiveConformalClassifier, ConformalEvaluator, ConformalScoreType, PredictionInterval,
    SplitConformalRegressor,
};
pub use conformal_risk::{
    ConformalRiskController, LearnThenTest, RiskCalibrationResult, RiskMetric,
};
pub use inductive_logic::{DatalogRule, DifferentiableILPEngine, RuleType, TNorm};
pub use mcts::{MCTSNode, MCTSSearchEngine, SearchResult};
pub use prover::{PropExpr, Sequent, SequentProver};
pub use swarm::{AgentProposal, ConsensusReport, ConsensusStrategy, SwarmOrchestrator};
pub use symbolic_gp::{OpType, ParetoIndividual, SymbolicGpConfig, SymbolicGpEngine, SymbolicNode};
pub use verifier::{ReasoningStep, ReasoningTrace, SelfCorrectionVerifier, VerificationReport};

pub use topos::ToposHeytingEngine;

pub mod hott;
pub use hott::HomotopyTypeEngine;

pub mod solomonoff;
pub use solomonoff::{AlgorithmicHypothesis, SolomonoffLevinEngine};
