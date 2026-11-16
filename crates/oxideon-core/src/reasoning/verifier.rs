//! Self-Correction Verification Engine for Autonomous Reasoning.
//!
//! Provides step-by-step reasoning validation and backtracking to eliminate
//! hallucinations and logical errors before final answer commitment.

/// A single step in a reasoning trace.
#[derive(Debug, Clone)]
pub struct ReasoningStep {
    pub step_index: usize,
    pub content: String,
    pub confidence: f64,
    pub is_verified: bool,
}

/// A structured multi-step reasoning trace with confidence tracking.
#[derive(Debug, Clone, Default)]
pub struct ReasoningTrace {
    pub steps: Vec<ReasoningStep>,
}

impl ReasoningTrace {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    pub fn add_step(&mut self, content: impl Into<String>, confidence: f64) {
        let idx = self.steps.len();
        self.steps.push(ReasoningStep {
            step_index: idx,
            content: content.into(),
            confidence,
            is_verified: false,
        });
    }

    /// Truncates the trace back to `step_idx` for self-correction backtracking.
    pub fn backtrack_to(&mut self, step_idx: usize) {
        if step_idx < self.steps.len() {
            self.steps.truncate(step_idx + 1);
        }
    }
}

/// Report detailing the outcome of reasoning verification.
#[derive(Debug, Clone)]
pub struct VerificationReport {
    pub is_valid: bool,
    pub failed_step_index: Option<usize>,
    pub recommended_backtrack_index: usize,
    pub average_confidence: f64,
}

/// Step-by-step reasoning verifier.
pub struct SelfCorrectionVerifier {
    pub confidence_threshold: f64,
}

impl SelfCorrectionVerifier {
    pub fn new(confidence_threshold: f64) -> Self {
        Self {
            confidence_threshold,
        }
    }

    /// Verifies each step in `trace` using an external checker function.
    pub fn verify<F>(&self, trace: &mut ReasoningTrace, mut checker: F) -> VerificationReport
    where
        F: FnMut(&str) -> (bool, f64),
    {
        let mut total_conf = 0.0;
        let n = trace.steps.len();

        for i in 0..n {
            let (passed, conf) = checker(&trace.steps[i].content);
            trace.steps[i].confidence = conf;
            trace.steps[i].is_verified = passed && conf >= self.confidence_threshold;
            total_conf += conf;

            if !trace.steps[i].is_verified {
                return VerificationReport {
                    is_valid: false,
                    failed_step_index: Some(i),
                    recommended_backtrack_index: i.saturating_sub(1),
                    average_confidence: if i > 0 {
                        total_conf / ((i + 1) as f64)
                    } else {
                        conf
                    },
                };
            }
        }

        VerificationReport {
            is_valid: true,
            failed_step_index: None,
            recommended_backtrack_index: n.saturating_sub(1),
            average_confidence: if n > 0 { total_conf / (n as f64) } else { 1.0 },
        }
    }
}
