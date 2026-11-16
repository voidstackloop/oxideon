//! Symbolic Tensor Shapes and Ahead-Of-Time Dimension Propagation.
//!
//! Enables dynamic batching, variable sequence length inference, and compile-time shape verification
//! without executing concrete tensor allocations.
//!
//! Evaluates algebraic shape equations (e.g. `[B, S, H] @ [H, 4H] -> [B, S, 4H]`), binds runtime
//! dimensions, and computes symbolic broadcasting compatibility.

use crate::error::{Error, Result};
use std::collections::HashMap;
use std::fmt;

/// A symbolic dimension expression.
#[derive(Debug, PartialEq, Eq, Hash)]
pub enum SymDim {
    Const(usize),
    Var(String),
    Add(Box<SymDim>, Box<SymDim>),
    Mul(Box<SymDim>, Box<SymDim>),
}

/// Most expression depth the recursive evaluators below accept: real
/// symbolic dims nest a handful of levels (one per shape operation).
const MAX_SYM_DEPTH: usize = 1024;

/// Depth of a symbolic dimension (leaves are 0), measured iteratively:
/// the evaluators recurse per operator, so depth is checked first.
fn dim_depth(expr: &SymDim) -> usize {
    let mut max = 0usize;
    let mut stack: Vec<(&SymDim, usize)> = vec![(expr, 0)];
    while let Some((e, d)) = stack.pop() {
        max = max.max(d);
        if let SymDim::Add(l, r) | SymDim::Mul(l, r) = e {
            stack.push((l, d.saturating_add(1)));
            stack.push((r, d.saturating_add(1)));
        }
    }
    max
}

impl Clone for SymDim {
    fn clone(&self) -> Self {
        // Iterative post-order rebuild: the derived clone recursed per
        // operator, so cloning a deep dimension overflowed the stack.
        enum Act<'a> {
            Visit(&'a SymDim),
            BuildBinary(fn(Box<SymDim>, Box<SymDim>) -> SymDim),
        }
        let mut work: Vec<Act<'_>> = vec![Act::Visit(self)];
        let mut results: Vec<SymDim> = Vec::new();
        while let Some(act) = work.pop() {
            match act {
                Act::Visit(expr) => match expr {
                    SymDim::Const(v) => results.push(SymDim::Const(*v)),
                    SymDim::Var(name) => results.push(SymDim::Var(name.clone())),
                    SymDim::Add(l, r) => {
                        work.push(Act::BuildBinary(SymDim::Add));
                        work.push(Act::Visit(r));
                        work.push(Act::Visit(l));
                    }
                    SymDim::Mul(l, r) => {
                        work.push(Act::BuildBinary(SymDim::Mul));
                        work.push(Act::Visit(r));
                        work.push(Act::Visit(l));
                    }
                },
                Act::BuildBinary(ctor) => {
                    let right = results.pop().unwrap_or(SymDim::Const(0));
                    let left = results.pop().unwrap_or(SymDim::Const(0));
                    results.push(ctor(Box::new(left), Box::new(right)));
                }
            }
        }
        results.pop().unwrap_or(SymDim::Const(0))
    }
}

impl SymDim {
    /// Moves the children out of `self` (leaving `Const(0)` leaves behind)
    /// so `Drop` can tear a deep dimension down iteratively.
    fn detach_children(&mut self, pending: &mut Vec<SymDim>) {
        match self {
            SymDim::Add(l, r) | SymDim::Mul(l, r) => {
                pending.push(std::mem::replace(&mut **l, SymDim::Const(0)));
                pending.push(std::mem::replace(&mut **r, SymDim::Const(0)));
            }
            SymDim::Const(_) | SymDim::Var(_) => {}
        }
    }
}

impl Drop for SymDim {
    fn drop(&mut self) {
        // Dropping a deep dimension recursed per operator (process abort).
        let mut pending: Vec<SymDim> = Vec::new();
        self.detach_children(&mut pending);
        while let Some(mut node) = pending.pop() {
            node.detach_children(&mut pending);
        }
    }
}

impl SymDim {
    pub fn var(name: impl Into<String>) -> Self {
        Self::Var(name.into())
    }

    pub fn val(v: usize) -> Self {
        Self::Const(v)
    }

    fn check_depth(&self) -> Result<()> {
        if dim_depth(self) > MAX_SYM_DEPTH {
            return Err(Error::Msg(format!(
                "Symbolic dimension depth exceeds {MAX_SYM_DEPTH}"
            )));
        }
        Ok(())
    }

    /// Evaluates the symbolic dimension against an environment map of concrete variables.
    pub fn resolve(&self, env: &HashMap<&str, usize>) -> Result<usize> {
        self.check_depth()?;
        match self {
            Self::Const(v) => Ok(*v),
            Self::Var(name) => env
                .get(name.as_str())
                .copied()
                .ok_or_else(|| Error::Msg(format!("Unbound symbolic dimension variable '{name}'"))),
            Self::Add(lhs, rhs) => {
                let l = lhs.resolve(env)?;
                let r = rhs.resolve(env)?;
                l.checked_add(r).ok_or_else(|| {
                    Error::Msg(format!(
                        "Symbolic dimension overflow: {l} + {r} exceeds usize"
                    ))
                })
            }
            Self::Mul(lhs, rhs) => {
                let l = lhs.resolve(env)?;
                let r = rhs.resolve(env)?;
                l.checked_mul(r).ok_or_else(|| {
                    Error::Msg(format!(
                        "Symbolic dimension overflow: {l} * {r} exceeds usize"
                    ))
                })
            }
        }
    }

    /// Attempts to unify/bind a concrete dimension value to this symbolic expression.
    pub fn bind_dim(&self, concrete: usize, bindings: &mut HashMap<String, usize>) -> Result<()> {
        self.check_depth()?;
        match self {
            Self::Const(v) => {
                if *v != concrete {
                    return Err(Error::Msg(format!(
                        "Dimension mismatch: expected constant {v}, got {concrete}"
                    )));
                }
                Ok(())
            }
            Self::Var(name) => {
                if let Some(&existing) = bindings.get(name) {
                    if existing != concrete {
                        return Err(Error::Msg(format!(
                            "Symbolic variable '{name}' conflict: already bound to {existing}, tried to bind {concrete}"
                        )));
                    }
                } else {
                    bindings.insert(name.clone(), concrete);
                }
                Ok(())
            }
            // Linear compound cases solve for the variable side; anything
            // underdetermined is consistency-checked once fully bound.
            Self::Add(lhs, rhs) => match (lhs.as_ref(), rhs.as_ref()) {
                (Self::Const(c), other) | (other, Self::Const(c)) => {
                    let rest = concrete.checked_sub(*c).ok_or_else(|| {
                        Error::Msg(format!(
                            "Dimension mismatch: {self} cannot equal {concrete}"
                        ))
                    })?;
                    other.bind_dim(rest, bindings)
                }
                _ => Self::check_fully_bound(self, concrete, bindings),
            },
            Self::Mul(lhs, rhs) => match (lhs.as_ref(), rhs.as_ref()) {
                (Self::Const(c), other) | (other, Self::Const(c)) => {
                    if *c == 0 {
                        if concrete != 0 {
                            return Err(Error::Msg(format!(
                                "Dimension mismatch: {self} cannot equal {concrete}"
                            )));
                        }
                        return Ok(()); // 0 * x == 0 for any x: nothing to bind.
                    }
                    if concrete % c != 0 {
                        return Err(Error::Msg(format!(
                            "Dimension mismatch: {concrete} is not divisible by {c} in {self}"
                        )));
                    }
                    other.bind_dim(concrete / c, bindings)
                }
                _ => Self::check_fully_bound(self, concrete, bindings),
            },
        }
    }

    /// Consistency-checks a compound expression against `concrete` when every
    /// variable it mentions is already bound; underdetermined expressions
    /// (unbound variables no equation can solve for) are accepted as-is.
    fn check_fully_bound(
        expr: &SymDim,
        concrete: usize,
        bindings: &HashMap<String, usize>,
    ) -> Result<()> {
        let mut vars = Vec::new();
        expr.collect_vars(&mut vars);
        if vars.iter().all(|v| bindings.contains_key(v)) {
            let env: HashMap<&str, usize> =
                bindings.iter().map(|(k, v)| (k.as_str(), *v)).collect();
            let value = expr.resolve(&env)?;
            if value != concrete {
                return Err(Error::Msg(format!(
                    "Dimension mismatch: {expr} evaluates to {value}, got {concrete}"
                )));
            }
        }
        Ok(())
    }

    fn collect_vars(&self, out: &mut Vec<String>) {
        match self {
            Self::Const(_) => {}
            Self::Var(name) => {
                if !out.contains(name) {
                    out.push(name.clone());
                }
            }
            Self::Add(lhs, rhs) | Self::Mul(lhs, rhs) => {
                lhs.collect_vars(out);
                rhs.collect_vars(out);
            }
        }
    }
}

impl fmt::Display for SymDim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Iterative (explicit action stack): the recursive version
        // overflowed on deep hand-built dimensions.
        enum Act<'a> {
            Visit(&'a SymDim),
            Text(&'a str),
        }
        let mut out = String::new();
        let mut stack: Vec<Act<'_>> = vec![Act::Visit(self)];
        while let Some(act) = stack.pop() {
            match act {
                Act::Text(s) => out.push_str(s),
                Act::Visit(expr) => match expr {
                    Self::Const(v) => out.push_str(&v.to_string()),
                    Self::Var(name) => out.push_str(name),
                    Self::Add(a, b) => {
                        stack.push(Act::Text(")"));
                        stack.push(Act::Visit(b));
                        stack.push(Act::Text(" + "));
                        stack.push(Act::Visit(a));
                        stack.push(Act::Text("("));
                    }
                    Self::Mul(a, b) => {
                        stack.push(Act::Text(")"));
                        stack.push(Act::Visit(b));
                        stack.push(Act::Text(" * "));
                        stack.push(Act::Visit(a));
                        stack.push(Act::Text("("));
                    }
                },
            }
        }
        write!(f, "{out}")
    }
}

/// A multi-dimensional symbolic tensor shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolicShape {
    pub dims: Vec<SymDim>,
}

impl SymbolicShape {
    pub fn new(dims: Vec<SymDim>) -> Self {
        Self { dims }
    }

    /// Parses a string representation like `"[B, S, H]"` or `"B, S, 128"`.
    pub fn parse(s: &str) -> Self {
        let clean = s.trim().trim_start_matches('[').trim_end_matches(']');
        let dims = clean
            .split(',')
            .map(|part| {
                let p = part.trim();
                if let Ok(v) = p.parse::<usize>() {
                    SymDim::Const(v)
                } else {
                    SymDim::Var(p.to_string())
                }
            })
            .collect();
        Self { dims }
    }

    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Binds a concrete shape (e.g. `[2, 16, 512]`) and discovers variable values.
    pub fn bind(&self, concrete_shape: &[usize]) -> Result<HashMap<String, usize>> {
        if self.dims.len() != concrete_shape.len() {
            return Err(Error::Msg(format!(
                "Rank mismatch: symbolic has rank {}, concrete has rank {}",
                self.dims.len(),
                concrete_shape.len()
            )));
        }

        let mut bindings = HashMap::new();
        for (i, dim) in self.dims.iter().enumerate() {
            dim.bind_dim(concrete_shape[i], &mut bindings)?;
        }
        Ok(bindings)
    }

    /// Evaluates to concrete dimensions given variable environment.
    pub fn resolve(&self, env: &HashMap<&str, usize>) -> Result<Vec<usize>> {
        self.dims.iter().map(|d| d.resolve(env)).collect()
    }

    /// Computes symbolic broadcasting with another shape.
    pub fn broadcast_with(&self, other: &Self) -> Result<Self> {
        let rank_a = self.dims.len();
        let rank_b = other.dims.len();
        let max_rank = rank_a.max(rank_b);

        let mut out_dims = Vec::with_capacity(max_rank);

        for i in 0..max_rank {
            let dim_a = if i < rank_a {
                &self.dims[rank_a - 1 - i]
            } else {
                &SymDim::Const(1)
            };
            let dim_b = if i < rank_b {
                &other.dims[rank_b - 1 - i]
            } else {
                &SymDim::Const(1)
            };

            let unified = match (dim_a, dim_b) {
                (SymDim::Const(1), other) | (other, SymDim::Const(1)) => other.clone(),
                (SymDim::Const(a), SymDim::Const(b)) if a == b => SymDim::Const(*a),
                (SymDim::Const(a), SymDim::Const(b)) => {
                    return Err(Error::Msg(format!(
                        "broadcast_with: incompatible dims {a} and {b}"
                    )));
                }
                (SymDim::Var(a), SymDim::Var(b)) if a == b => SymDim::Var(a.clone()),
                (a, _) => a.clone(), // dynamic match assumed valid
            };
            out_dims.push(unified);
        }

        out_dims.reverse();
        Ok(Self { dims: out_dims })
    }

    /// Computes resulting symbolic shape of matrix multiplication `self @ other`.
    pub fn matmul_with(&self, other: &Self) -> Result<Self> {
        if self.dims.len() < 2 || other.dims.len() < 2 {
            return Err(Error::Msg(
                "Symbolic matmul requires both shapes to have rank >= 2".into(),
            ));
        }

        let a_batch = &self.dims[..self.dims.len() - 2];
        let b_batch = &other.dims[..other.dims.len() - 2];

        // Broadcast batch prefixes
        let batch_a_shape = Self {
            dims: a_batch.to_vec(),
        };
        let batch_b_shape = Self {
            dims: b_batch.to_vec(),
        };
        let mut out_batch = batch_a_shape.broadcast_with(&batch_b_shape)?.dims;

        // The contraction dim K must agree: provable constant mismatches
        // are rejected statically; symbolic K agreement is verified when the
        // shapes are bound to concrete values.
        let k_a = &self.dims[self.dims.len() - 1];
        let k_b = &other.dims[other.dims.len() - 2];
        if let (SymDim::Const(a), SymDim::Const(b)) = (k_a, k_b)
            && a != b
        {
            return Err(Error::Msg(format!(
                "matmul_with: contraction dim mismatch ({a} vs {b})"
            )));
        }

        let m_dim = self.dims[self.dims.len() - 2].clone();
        let n_dim = other.dims[other.dims.len() - 1].clone();

        out_batch.push(m_dim);
        out_batch.push(n_dim);

        Ok(Self { dims: out_batch })
    }
}

impl fmt::Display for SymbolicShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}
