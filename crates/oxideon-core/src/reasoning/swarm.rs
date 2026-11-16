//! Multi-Agent Consensus & Swarm Intelligence Orchestration.
//!
//! Orchestrates decentralized reasoning agents through multi-round debate, peer critique,
//! and game-theoretic consensus resolution:
//! - [`AgentProposal`]: Structured reasoning proposition with confidence score and justification.
//! - [`ConsensusStrategy`]: Majority voting, weighted Borda count, and Nash equilibrium resolution.
//! - [`SwarmOrchestrator`]: Multi-agent debate engine driving distributed consensus.

use crate::error::{Error, Result};
use std::collections::HashMap;

/// Consensus resolution strategies across agent proposals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsensusStrategy {
    MajorityVote,
    ConfidenceWeighted,
    NashEquilibrium,
}

/// An individual agent's proposal and confidence.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentProposal {
    pub agent_id: String,
    pub solution: String,
    pub confidence: f64,
    pub justification: String,
}

impl AgentProposal {
    pub fn new(agent_id: &str, solution: &str, confidence: f64, justification: &str) -> Self {
        Self {
            agent_id: agent_id.to_string(),
            solution: solution.to_string(),
            confidence: confidence.clamp(0.0, 1.0),
            justification: justification.to_string(),
        }
    }
}

/// Final consensus report from swarm orchestration.
#[derive(Debug, Clone, PartialEq)]
pub struct ConsensusReport {
    pub final_solution: String,
    pub strategy_used: ConsensusStrategy,
    pub agreement_score: f64,
    pub total_rounds: usize,
    pub participating_agents: Vec<String>,
    pub vote_distribution: HashMap<String, f64>,
}

/// Multi-Agent Swarm Debate and Consensus Orchestrator.
#[derive(Debug, Clone)]
pub struct SwarmOrchestrator {
    pub max_debate_rounds: usize,
    pub convergence_threshold: f64,
    pub strategy: ConsensusStrategy,
}

impl Default for SwarmOrchestrator {
    fn default() -> Self {
        Self {
            max_debate_rounds: 3,
            convergence_threshold: 0.75,
            strategy: ConsensusStrategy::ConfidenceWeighted,
        }
    }
}

impl SwarmOrchestrator {
    pub fn new(
        max_debate_rounds: usize,
        convergence_threshold: f64,
        strategy: ConsensusStrategy,
    ) -> Self {
        Self {
            max_debate_rounds: max_debate_rounds.max(1),
            convergence_threshold: convergence_threshold.clamp(0.0, 1.0),
            strategy,
        }
    }

    /// Resolves consensus from a set of agent proposals according to the active strategy.
    pub fn resolve_consensus(&self, proposals: &[AgentProposal]) -> Result<ConsensusReport> {
        if proposals.is_empty() {
            return Err(Error::Msg(
                "Cannot resolve consensus on empty proposal set".to_string(),
            ));
        }

        let mut vote_distribution = HashMap::new();
        let mut agents = Vec::with_capacity(proposals.len());

        match self.strategy {
            ConsensusStrategy::MajorityVote => {
                for p in proposals {
                    agents.push(p.agent_id.clone());
                    *vote_distribution.entry(p.solution.clone()).or_insert(0.0) += 1.0;
                }
            }
            ConsensusStrategy::ConfidenceWeighted => {
                for p in proposals {
                    agents.push(p.agent_id.clone());
                    *vote_distribution.entry(p.solution.clone()).or_insert(0.0) += p.confidence;
                }
            }
            ConsensusStrategy::NashEquilibrium => {
                // In payoff matrix formulation: agents gain mutual payoff for agreeing on high-confidence solutions
                for (i, p1) in proposals.iter().enumerate() {
                    agents.push(p1.agent_id.clone());
                    let mut payoff = p1.confidence;
                    for (j, p2) in proposals.iter().enumerate() {
                        if i != j && p1.solution == p2.solution {
                            payoff += p1.confidence * p2.confidence;
                        }
                    }
                    *vote_distribution.entry(p1.solution.clone()).or_insert(0.0) += payoff;
                }
            }
        }

        let total_weight: f64 = vote_distribution.values().sum();
        let (winning_sol, max_weight) = vote_distribution
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, v)| (k.clone(), *v))
            .ok_or_else(|| Error::Msg("Failed to determine winner".to_string()))?;

        let agreement_score = if total_weight > 0.0 {
            max_weight / total_weight
        } else {
            0.0
        };

        Ok(ConsensusReport {
            final_solution: winning_sol,
            strategy_used: self.strategy,
            agreement_score,
            total_rounds: 1,
            participating_agents: agents,
            vote_distribution,
        })
    }

    /// Simulates multi-round debate where agents refine their proposals in response to peer solutions.
    ///
    /// With `max_debate_rounds == 0` there is no debate: the initial proposals
    /// are resolved directly with `total_rounds == 0`.
    pub fn debate(
        &self,
        initial_proposals: Vec<AgentProposal>,
        mut critique_and_revise: impl FnMut(usize, &[AgentProposal]) -> Vec<AgentProposal>,
    ) -> Result<ConsensusReport> {
        let mut current_proposals = initial_proposals;

        if self.max_debate_rounds == 0 {
            let mut report = self.resolve_consensus(&current_proposals)?;
            report.total_rounds = 0;
            return Ok(report);
        }
        // Each round costs a resolve pass; an uncapped budget spins millions
        // of rounds when agents never converge (real debates run dozens).
        crate::limits::check_repeats("max_debate_rounds", self.max_debate_rounds)?;

        for round in 1..=self.max_debate_rounds {
            let report = self.resolve_consensus(&current_proposals)?;

            if report.agreement_score >= self.convergence_threshold
                || round == self.max_debate_rounds
            {
                return Ok(ConsensusReport {
                    total_rounds: round,
                    ..report
                });
            }

            // Agents revise proposals based on debate context
            current_proposals = critique_and_revise(round, &current_proposals);
        }

        // Unreachable: the final iteration always returns via `round == max_debate_rounds`.
        unreachable!("debate loop must return on its final round")
    }
}
