//! Grammar-Constrained Decoding, JSON Schema Guided Sampling, and Agent Runtime.
//!
//! Enforces deterministic syntactic guarantees on LLM generation (similar to Outlines / Guidance):
//! - Masking invalid vocabulary tokens to $-\infty$ logit before softmax so the model can **never**
//!   produce invalid JSON, invalid types, or broken regex syntax.
//! - [`JsonSchema`]: High-level builder for structured JSON objects, arrays, strings, ints, floats, and booleans.
//! - [`GrammarLogitProcessor`]: High-performance bitmask / Trie filter evaluating token compatibility.
//! - [`AgentRuntime`]: Automated function calling, tool registration, argument schema validation, and dispatch.

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::tensor::Tensor;
use crate::tokenizer::BpeTokenizer;

pub mod regex;
pub mod sandbox;
pub use sandbox::{
    ExecutionReport, ExecutionSandbox, SandboxStatus, SynthesisEngine, SynthesisReport, TestCase,
};

/// Primitive JSON schema value type constraint.
#[derive(Debug, PartialEq)]
pub enum SchemaType {
    String,
    Integer,
    Number,
    Boolean,
    Object(Vec<(String, SchemaType)>),
    Array(Box<SchemaType>),
    Enum(Vec<String>),
}

impl Clone for SchemaType {
    fn clone(&self) -> Self {
        // Iterative post-order rebuild: the derived clone recursed per
        // nesting level, so cloning a deep schema overflowed the stack.
        enum Act<'a> {
            Visit(&'a SchemaType),
            BuildArray,
            BuildObject(Vec<String>),
        }
        let mut work: Vec<Act<'_>> = vec![Act::Visit(self)];
        let mut results: Vec<SchemaType> = Vec::new();
        while let Some(act) = work.pop() {
            match act {
                Act::Visit(schema) => match schema {
                    SchemaType::String => results.push(SchemaType::String),
                    SchemaType::Integer => results.push(SchemaType::Integer),
                    SchemaType::Number => results.push(SchemaType::Number),
                    SchemaType::Boolean => results.push(SchemaType::Boolean),
                    SchemaType::Enum(vals) => results.push(SchemaType::Enum(vals.clone())),
                    SchemaType::Array(sub) => {
                        work.push(Act::BuildArray);
                        work.push(Act::Visit(sub));
                    }
                    SchemaType::Object(fields) => {
                        let names: Vec<String> = fields.iter().map(|(n, _)| n.clone()).collect();
                        work.push(Act::BuildObject(names));
                        for (_, sub) in fields.iter().rev() {
                            work.push(Act::Visit(sub));
                        }
                    }
                },
                Act::BuildArray => {
                    let sub = results.pop().unwrap_or(SchemaType::Boolean);
                    results.push(SchemaType::Array(Box::new(sub)));
                }
                Act::BuildObject(names) => {
                    let subs: Vec<SchemaType> =
                        results.split_off(results.len().saturating_sub(names.len()));
                    let fields = names.into_iter().zip(subs).collect();
                    results.push(SchemaType::Object(fields));
                }
            }
        }
        results.pop().unwrap_or(SchemaType::Boolean)
    }
}

impl SchemaType {
    /// Moves the children out of `self` (leaving `Boolean` leaves behind)
    /// so `Drop` can tear a deep schema down iteratively.
    fn detach_children(&mut self, pending: &mut Vec<SchemaType>) {
        match self {
            SchemaType::Array(child) => {
                pending.push(std::mem::replace(&mut **child, SchemaType::Boolean));
            }
            SchemaType::Object(fields) => {
                for (_, sub) in fields.iter_mut() {
                    pending.push(std::mem::replace(sub, SchemaType::Boolean));
                }
            }
            SchemaType::String
            | SchemaType::Integer
            | SchemaType::Number
            | SchemaType::Boolean
            | SchemaType::Enum(_) => {}
        }
    }
}

impl Drop for SchemaType {
    fn drop(&mut self) {
        // Dropping a deep schema recursed per nesting level (process abort).
        let mut pending: Vec<SchemaType> = Vec::new();
        self.detach_children(&mut pending);
        while let Some(mut node) = pending.pop() {
            node.detach_children(&mut pending);
        }
    }
}

/// JSON Schema definition for tool arguments and structured generation.
#[derive(Debug, Clone)]
pub struct JsonSchema {
    pub root: SchemaType,
}

impl JsonSchema {
    pub fn object(fields: Vec<(&str, SchemaType)>) -> Self {
        Self {
            root: SchemaType::Object(
                fields
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect(),
            ),
        }
    }

    pub fn string() -> Self {
        Self {
            root: SchemaType::String,
        }
    }

    pub fn integer() -> Self {
        Self {
            root: SchemaType::Integer,
        }
    }

    pub fn number() -> Self {
        Self {
            root: SchemaType::Number,
        }
    }

    pub fn boolean() -> Self {
        Self {
            root: SchemaType::Boolean,
        }
    }

    pub fn enum_values(values: &[&str]) -> Self {
        Self {
            root: SchemaType::Enum(values.iter().map(|s| s.to_string()).collect()),
        }
    }
}

/// State of JSON grammar parsing during token generation.
#[derive(Debug, Clone)]
pub struct JsonParserState {
    pub buffer: String,
    pub schema: SchemaType,
    pub depth: usize,
    pub is_complete: bool,
}

impl JsonParserState {
    pub fn new(schema: SchemaType) -> Self {
        Self {
            buffer: String::new(),
            schema,
            depth: 0,
            is_complete: false,
        }
    }

    /// Evaluates whether candidate string chunk can validly extend current JSON buffer.
    pub fn can_accept(&self, chunk: &str) -> bool {
        // Quick prefix/character-level syntactic verification
        let mut test = self.buffer.clone();
        test.push_str(chunk);

        match &self.schema {
            SchemaType::Object(_) => {
                let trimmed = test.trim_start();
                if trimmed.is_empty() {
                    return true;
                }
                if !trimmed.starts_with('{') {
                    return false;
                }
                // Check balanced braces count so far
                let opens = trimmed.chars().filter(|&c| c == '{').count();
                let closes = trimmed.chars().filter(|&c| c == '}').count();
                closes <= opens
            }
            SchemaType::Integer => {
                let trimmed = test.trim();
                trimmed.is_empty() || trimmed.chars().all(|c| c.is_ascii_digit() || c == '-')
            }
            SchemaType::Number => {
                let trimmed = test.trim();
                trimmed.is_empty()
                    || trimmed
                        .chars()
                        .all(|c| c.is_ascii_digit() || c == '-' || c == '.')
            }
            SchemaType::Boolean => {
                let trimmed = test.trim().to_lowercase();
                "true".starts_with(&trimmed) || "false".starts_with(&trimmed)
            }
            SchemaType::String => true,
            SchemaType::Enum(vals) => {
                let trimmed = test.trim().trim_matches('"');
                vals.iter().any(|v| v.starts_with(trimmed))
            }
            SchemaType::Array(_) => {
                let trimmed = test.trim_start();
                trimmed.is_empty() || trimmed.starts_with('[')
            }
        }
    }

    /// Feeds token text into parser state and updates completion flag.
    pub fn feed(&mut self, chunk: &str) {
        self.buffer.push_str(chunk);
        let trimmed = self.buffer.trim();
        match &self.schema {
            SchemaType::Object(_) => {
                if trimmed.starts_with('{') && trimmed.ends_with('}') {
                    let opens = trimmed.chars().filter(|&c| c == '{').count();
                    let closes = trimmed.chars().filter(|&c| c == '}').count();
                    if opens > 0 && opens == closes {
                        self.is_complete = true;
                    }
                }
            }
            SchemaType::Boolean if trimmed == "true" || trimmed == "false" => {
                self.is_complete = true;
            }
            _ => {}
        }
    }
}

/// Logit processor that masks invalid tokens according to grammar state.
pub struct GrammarLogitProcessor {
    pub state: JsonParserState,
    vocab_strings: Vec<String>,
}

impl GrammarLogitProcessor {
    pub fn new(schema: JsonSchema, tokenizer: &BpeTokenizer) -> Self {
        let vocab_size = tokenizer.vocab_size();
        let mut vocab_strings = Vec::with_capacity(vocab_size);
        for id in 0..vocab_size {
            vocab_strings.push(tokenizer.decode(&[id]));
        }
        Self {
            state: JsonParserState::new(schema.root),
            vocab_strings,
        }
    }

    /// Filters unnormalized `logits` vector, setting disallowed token logits to `f32::NEG_INFINITY`.
    pub fn process_logits(&self, logits: &Tensor) -> Result<Tensor> {
        let mut logit_data = logits.to_vec::<f32>()?;
        for (tok_id, token_str) in self.vocab_strings.iter().enumerate() {
            if tok_id < logit_data.len() && !self.state.can_accept(token_str) {
                logit_data[tok_id] = f32::NEG_INFINITY;
            }
        }
        Tensor::from_slice(&logit_data, logits.shape())
    }

    /// Advances grammar state with sampled token ID.
    pub fn advance(&mut self, token_id: usize) {
        if let Some(s) = self.vocab_strings.get(token_id) {
            self.state.feed(s);
        }
    }
}

/// Tool execution handler closure signature.
pub type ToolHandler = Box<dyn Fn(&str) -> Result<String> + Send + Sync>;

/// Tool function signature definition for Autonomous Agent tool calling.
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: JsonSchema,
    pub handler: ToolHandler,
}

/// Autonomous agent runtime executing function calls and multi-turn tool loops.
pub struct AgentRuntime {
    tools: HashMap<String, ToolDef>,
}

impl Default for AgentRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentRuntime {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Registers a callable tool with a parameter JSON schema.
    pub fn register_tool<F>(
        &mut self,
        name: &str,
        description: &str,
        parameters: JsonSchema,
        handler: F,
    ) where
        F: Fn(&str) -> Result<String> + Send + Sync + 'static,
    {
        self.tools.insert(
            name.to_string(),
            ToolDef {
                name: name.to_string(),
                description: description.to_string(),
                parameters,
                handler: Box::new(handler),
            },
        );
    }

    /// Dispatches tool call given function name and JSON arguments string.
    pub fn call_tool(&self, name: &str, args_json: &str) -> Result<String> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| Error::Msg(format!("Tool '{name}' is not registered")))?;
        (tool.handler)(args_json)
    }

    /// Formats the tool definitions into a system prompt schema block.
    pub fn render_system_prompt(&self) -> String {
        let mut out = String::from("Available Tools:\n");
        for (name, def) in &self.tools {
            out.push_str(&format!("- {}: {}\n", name, def.description));
        }
        out
    }
}
