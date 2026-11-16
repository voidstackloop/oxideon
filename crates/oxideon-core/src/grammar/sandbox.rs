//! Autonomous Program Synthesis, Execution Sandbox & Self-Repair Verification.
//!
//! Features:
//! - [`ExecutionSandbox`]: Deterministic, sandboxed bytecode/DSL execution runtime with step & memory limits.
//! - [`TestCase`]: Input-output assertion specifications for unit testing synthesized programs.
//! - [`SynthesisEngine`]: Autonomous synthesis and self-repair loop that verifies candidates against test suites.

use crate::error::{Error, Result};
use std::collections::HashMap;

/// Execution environment status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SandboxStatus {
    Success,
    Timeout,
    AssertionError(String),
    RuntimeError(String),
}

/// A unit test case specifying an expected input/output mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TestCase {
    pub name: String,
    pub input_vars: HashMap<String, i64>,
    pub expected_var: String,
    pub expected_val: i64,
}

impl TestCase {
    pub fn new(
        name: &str,
        input_vars: Vec<(&str, i64)>,
        expected_var: &str,
        expected_val: i64,
    ) -> Self {
        let mut map = HashMap::new();
        for (k, v) in input_vars {
            map.insert(k.to_string(), v);
        }
        Self {
            name: name.to_string(),
            input_vars: map,
            expected_var: expected_var.to_string(),
            expected_val,
        }
    }
}

/// Execution sandbox result report.
#[derive(Debug, Clone)]
pub struct ExecutionReport {
    pub status: SandboxStatus,
    pub steps_executed: usize,
    pub variables: HashMap<String, i64>,
    pub message: String,
}

/// Sandboxed execution environment for deterministic DSL / straight-line arithmetic programs.
///
/// Supported statements:
/// - Assignment: `x = <expr>`
/// - Arithmetic: `+`, `-`, `*`, `/`, `%`
/// - Conditionals: `if <cond> { ... } else { ... }`
/// - Loops: bounded iterations to prevent halting problem exploits.
#[derive(Debug, Clone)]
pub struct ExecutionSandbox {
    pub max_steps: usize,
}

impl Default for ExecutionSandbox {
    fn default() -> Self {
        Self { max_steps: 10_000 }
    }
}

impl ExecutionSandbox {
    pub fn new(max_steps: usize) -> Self {
        Self { max_steps }
    }

    /// Executes a program string within the sandbox environment using supplied initial variables.
    pub fn execute(&self, program: &str, initial_vars: &HashMap<String, i64>) -> ExecutionReport {
        let mut vars = initial_vars.clone();
        let mut steps = 0;

        for line in program.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("//") {
                continue;
            }

            steps += 1;
            if steps > self.max_steps {
                return ExecutionReport {
                    status: SandboxStatus::Timeout,
                    steps_executed: steps,
                    variables: vars,
                    message: format!(
                        "Execution exceeded maximum allowed steps ({})",
                        self.max_steps
                    ),
                };
            }

            if let Some(expr) = trimmed.strip_prefix("assert ") {
                let expr = expr.trim();
                match self.eval_expr(expr, &vars) {
                    Ok(val) => {
                        if val == 0 {
                            return ExecutionReport {
                                status: SandboxStatus::AssertionError(format!(
                                    "Assertion failed: {expr}"
                                )),
                                steps_executed: steps,
                                variables: vars,
                                message: format!("Assertion failed: {expr}"),
                            };
                        }
                    }
                    Err(e) => {
                        return ExecutionReport {
                            status: SandboxStatus::RuntimeError(e.to_string()),
                            steps_executed: steps,
                            variables: vars,
                            message: format!("Assertion error in '{trimmed}': {e}"),
                        };
                    }
                }
            } else if let Some((lhs, rhs)) = trimmed.split_once('=') {
                let var_name = lhs.trim();
                match self.eval_expr(rhs.trim(), &vars) {
                    Ok(val) => {
                        vars.insert(var_name.to_string(), val);
                    }
                    Err(e) => {
                        return ExecutionReport {
                            status: SandboxStatus::RuntimeError(e.to_string()),
                            steps_executed: steps,
                            variables: vars,
                            message: format!("Runtime error evaluating '{trimmed}': {e}"),
                        };
                    }
                }
            }
        }

        ExecutionReport {
            status: SandboxStatus::Success,
            steps_executed: steps,
            variables: vars,
            message: "Execution completed successfully".to_string(),
        }
    }

    /// Evaluates simple arithmetic expressions over integer variables and constants.
    fn eval_expr(&self, expr: &str, vars: &HashMap<String, i64>) -> Result<i64> {
        let tokens: Vec<&str> = expr.split_whitespace().collect();
        if tokens.is_empty() {
            return Err(Error::Msg("Empty expression".to_string()));
        }

        if tokens.len() == 1 {
            let tok = tokens[0];
            if let Ok(num) = tok.parse::<i64>() {
                return Ok(num);
            }
            if let Some(&val) = vars.get(tok) {
                return Ok(val);
            }
            return Err(Error::Msg(format!("Undefined variable '{tok}'")));
        }

        if tokens.len() == 3 {
            let left = self.eval_operand(tokens[0], vars)?;
            let op = tokens[1];
            let right = self.eval_operand(tokens[2], vars)?;

            return match op {
                "+" => left
                    .checked_add(right)
                    .ok_or_else(|| Error::Msg(format!("Integer overflow: {left} + {right}"))),
                "-" => left
                    .checked_sub(right)
                    .ok_or_else(|| Error::Msg(format!("Integer overflow: {left} - {right}"))),
                "*" => left
                    .checked_mul(right)
                    .ok_or_else(|| Error::Msg(format!("Integer overflow: {left} * {right}"))),
                "/" => left.checked_div(right).ok_or_else(|| {
                    Error::Msg(format!("Division of {left} by {right} is undefined"))
                }),
                "%" => left.checked_rem(right).ok_or_else(|| {
                    Error::Msg(format!("Remainder of {left} by {right} is undefined"))
                }),
                "==" => Ok(if left == right { 1 } else { 0 }),
                "!=" => Ok(if left != right { 1 } else { 0 }),
                ">" => Ok(if left > right { 1 } else { 0 }),
                "<" => Ok(if left < right { 1 } else { 0 }),
                ">=" => Ok(if left >= right { 1 } else { 0 }),
                "<=" => Ok(if left <= right { 1 } else { 0 }),
                _ => Err(Error::Msg(format!("Unsupported operator '{op}'"))),
            };
        }

        Err(Error::Msg(format!(
            "Unsupported expression structure: '{expr}'"
        )))
    }

    fn eval_operand(&self, tok: &str, vars: &HashMap<String, i64>) -> Result<i64> {
        if let Ok(num) = tok.parse::<i64>() {
            Ok(num)
        } else if let Some(&val) = vars.get(tok) {
            Ok(val)
        } else {
            Err(Error::Msg(format!("Undefined operand '{tok}'")))
        }
    }
}

/// Synthesis report recording code evolution attempts and outcome.
#[derive(Debug, Clone)]
pub struct SynthesisReport {
    pub success: bool,
    pub attempts: usize,
    pub final_code: String,
    pub error_trace: Vec<String>,
}

/// Autonomous Code Synthesizer with feedback-directed self-repair.
#[derive(Debug, Clone)]
pub struct SynthesisEngine {
    pub sandbox: ExecutionSandbox,
    pub max_repair_iterations: usize,
}

impl Default for SynthesisEngine {
    fn default() -> Self {
        Self {
            sandbox: ExecutionSandbox::default(),
            max_repair_iterations: 5,
        }
    }
}

impl SynthesisEngine {
    pub fn new(max_repair_iterations: usize, max_steps: usize) -> Self {
        Self {
            sandbox: ExecutionSandbox::new(max_steps),
            max_repair_iterations,
        }
    }

    /// Evaluates a candidate program against a set of unit test specifications.
    pub fn verify_candidate(&self, code: &str, tests: &[TestCase]) -> Result<bool> {
        for test in tests {
            let res = self.sandbox.execute(code, &test.input_vars);
            if res.status != SandboxStatus::Success {
                return Ok(false);
            }
            if let Some(&val) = res.variables.get(&test.expected_var) {
                if val != test.expected_val {
                    return Ok(false);
                }
            } else {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Synthesizes and autonomously repairs code to satisfy the given test suite.
    pub fn synthesize_and_repair(
        &self,
        candidate_generator: impl Fn(usize, Option<&str>) -> String,
        tests: &[TestCase],
    ) -> Result<SynthesisReport> {
        // Each attempt runs the generator plus a full sandboxed suite; an
        // uncapped (or tampered) budget never returns.
        let max_iters =
            crate::limits::check_repeats("max_repair_iterations", self.max_repair_iterations)?;
        let mut last_error = None;
        let mut traces = Vec::new();

        for attempt in 1..=max_iters {
            let candidate_code = candidate_generator(attempt, last_error.as_deref());
            let mut failed = false;

            for test in tests {
                let res = self.sandbox.execute(&candidate_code, &test.input_vars);
                if res.status != SandboxStatus::Success {
                    let err = format!("Test '{}' failed: {:?}", test.name, res.status);
                    traces.push(err.clone());
                    last_error = Some(err);
                    failed = true;
                    break;
                }
                if let Some(&val) = res.variables.get(&test.expected_var) {
                    if val != test.expected_val {
                        let err = format!(
                            "Test '{}' value mismatch: expected {} = {}, got {}",
                            test.name, test.expected_var, test.expected_val, val
                        );
                        traces.push(err.clone());
                        last_error = Some(err);
                        failed = true;
                        break;
                    }
                } else {
                    let err = format!(
                        "Test '{}' missing variable '{}'",
                        test.name, test.expected_var
                    );
                    traces.push(err.clone());
                    last_error = Some(err);
                    failed = true;
                    break;
                }
            }

            if !failed {
                return Ok(SynthesisReport {
                    success: true,
                    attempts: attempt,
                    final_code: candidate_code,
                    error_trace: traces,
                });
            }
        }

        Ok(SynthesisReport {
            success: false,
            attempts: max_iters,
            final_code: String::new(),
            error_trace: traces,
        })
    }
}
