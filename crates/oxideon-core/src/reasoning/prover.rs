//! Autonomous Formal Mathematics & Automated Theorem Proving (ATP) Engine.
//!
//! Implements Gentzen's classical Sequent Calculus ($\mathbf{LK}$) for automated proposition verification:
//! - Complete propositional expressions: atomic propositions, negation ($\neg$), conjunction ($\land$),
//!   disjunction ($\lor$), implication ($\to$), biconditional ($\leftrightarrow$), and boolean constants.
//! - Sequent deduction: $\Gamma \vdash \Delta \iff \bigwedge_{A \in \Gamma} A \implies \bigvee_{B \in \Delta} B$.
//! - Exact syntactic tree search with termination guarantees and deterministic proof traces.
//! - Can verify classical tautologies including:
//!   - Law of Excluded Middle ($\vdash P \lor \neg P$)
//!   - Double Negation Elimination ($\neg \neg P \vdash P$)
//!   - Peirce's Law ($\vdash ((P \to Q) \to P) \to P$)
//!   - De Morgan's Dualities ($\neg (P \land Q) \leftrightarrow (\neg P \lor \neg Q)$)
//!   - Modus Ponens ($P, P \to Q \vdash Q$)

use crate::error::{Error, Result};

/// Propositional logic expressions.
#[derive(Debug, PartialEq, Eq, Hash)]
pub enum PropExpr {
    True,
    False,
    Var(String),
    Not(Box<PropExpr>),
    And(Box<PropExpr>, Box<PropExpr>),
    Or(Box<PropExpr>, Box<PropExpr>),
    Implies(Box<PropExpr>, Box<PropExpr>),
    Iff(Box<PropExpr>, Box<PropExpr>),
}

impl PropExpr {
    /// Creates an atomic proposition variable $P$.
    pub fn var(name: impl Into<String>) -> Self {
        Self::Var(name.into())
    }

    /// Negation $\neg A$.
    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Self {
        Self::Not(Box::new(self))
    }

    /// Conjunction $A \land B$.
    pub fn and(self, other: Self) -> Self {
        Self::And(Box::new(self), Box::new(other))
    }

    /// Disjunction $A \lor B$.
    pub fn or(self, other: Self) -> Self {
        Self::Or(Box::new(self), Box::new(other))
    }

    /// Implication $A \to B$.
    pub fn implies(self, other: Self) -> Self {
        Self::Implies(Box::new(self), Box::new(other))
    }

    /// Biconditional $A \leftrightarrow B$.
    pub fn iff(self, other: Self) -> Self {
        Self::Iff(Box::new(self), Box::new(other))
    }
}

impl Clone for PropExpr {
    fn clone(&self) -> Self {
        // Iterative post-order rebuild: the derived clone recursed per
        // connective, so cloning a deep formula overflowed the stack.
        enum Act<'a> {
            Visit(&'a PropExpr),
            BuildNot,
            BuildBinary(fn(Box<PropExpr>, Box<PropExpr>) -> PropExpr),
        }
        let mut work: Vec<Act<'_>> = vec![Act::Visit(self)];
        let mut results: Vec<PropExpr> = Vec::new();
        while let Some(act) = work.pop() {
            match act {
                Act::Visit(expr) => match expr {
                    PropExpr::True => results.push(PropExpr::True),
                    PropExpr::False => results.push(PropExpr::False),
                    PropExpr::Var(name) => results.push(PropExpr::Var(name.clone())),
                    PropExpr::Not(sub) => {
                        work.push(Act::BuildNot);
                        work.push(Act::Visit(sub));
                    }
                    PropExpr::And(l, r) => {
                        work.push(Act::BuildBinary(PropExpr::And));
                        work.push(Act::Visit(r));
                        work.push(Act::Visit(l));
                    }
                    PropExpr::Or(l, r) => {
                        work.push(Act::BuildBinary(PropExpr::Or));
                        work.push(Act::Visit(r));
                        work.push(Act::Visit(l));
                    }
                    PropExpr::Implies(l, r) => {
                        work.push(Act::BuildBinary(PropExpr::Implies));
                        work.push(Act::Visit(r));
                        work.push(Act::Visit(l));
                    }
                    PropExpr::Iff(l, r) => {
                        work.push(Act::BuildBinary(PropExpr::Iff));
                        work.push(Act::Visit(r));
                        work.push(Act::Visit(l));
                    }
                },
                Act::BuildNot => {
                    // The child result is on top by construction.
                    let sub = results.pop().unwrap_or(PropExpr::True);
                    results.push(PropExpr::Not(Box::new(sub)));
                }
                Act::BuildBinary(ctor) => {
                    let right = results.pop().unwrap_or(PropExpr::True);
                    let left = results.pop().unwrap_or(PropExpr::True);
                    results.push(ctor(Box::new(left), Box::new(right)));
                }
            }
        }
        results.pop().unwrap_or(PropExpr::True)
    }
}

impl PropExpr {
    /// Moves the children out of `self` (leaving `True` leaves behind) so
    /// `Drop` can tear a deep formula down iteratively.
    fn detach_children(&mut self, pending: &mut Vec<PropExpr>) {
        match self {
            PropExpr::Not(child) => {
                pending.push(std::mem::replace(&mut **child, PropExpr::True));
            }
            PropExpr::And(l, r)
            | PropExpr::Or(l, r)
            | PropExpr::Implies(l, r)
            | PropExpr::Iff(l, r) => {
                pending.push(std::mem::replace(&mut **l, PropExpr::True));
                pending.push(std::mem::replace(&mut **r, PropExpr::True));
            }
            PropExpr::True | PropExpr::False | PropExpr::Var(_) => {}
        }
    }
}

impl Drop for PropExpr {
    fn drop(&mut self) {
        // Dropping a deep formula recursed per connective (process abort).
        let mut pending: Vec<PropExpr> = Vec::new();
        self.detach_children(&mut pending);
        while let Some(mut node) = pending.pop() {
            node.detach_children(&mut pending);
        }
    }
}

/// A Sequent $\Gamma \vdash \Delta$ representing an entailment in Sequent Calculus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sequent {
    pub antecedent: Vec<PropExpr>,
    pub succedent: Vec<PropExpr>,
}

impl Sequent {
    pub fn new(antecedent: Vec<PropExpr>, succedent: Vec<PropExpr>) -> Self {
        Self {
            antecedent,
            succedent,
        }
    }

    /// Goal sequent $\vdash A$.
    pub fn goal(expr: PropExpr) -> Self {
        Self {
            antecedent: Vec::new(),
            succedent: vec![expr],
        }
    }
}

/// Automated Theorem Prover using Sequent Calculus ($\mathbf{LK}$).
#[derive(Debug, Default, Clone)]
pub struct SequentProver {
    pub max_depth: usize,
}

/// Depth of a formula (leaves are 0), measured with an explicit stack:
/// the clone / compare / search in the prover all recurse per connective,
/// so depth must be known before any of them run on a hostile formula.
fn formula_depth(expr: &PropExpr) -> usize {
    let mut max = 0usize;
    let mut stack: Vec<(&PropExpr, usize)> = vec![(expr, 0)];
    while let Some((e, d)) = stack.pop() {
        max = max.max(d);
        match e {
            PropExpr::Not(sub) => stack.push((sub, d.saturating_add(1))),
            PropExpr::And(l, r)
            | PropExpr::Or(l, r)
            | PropExpr::Implies(l, r)
            | PropExpr::Iff(l, r) => {
                stack.push((l, d.saturating_add(1)));
                stack.push((r, d.saturating_add(1)));
            }
            PropExpr::True | PropExpr::False | PropExpr::Var(_) => {}
        }
    }
    max
}

impl SequentProver {
    pub fn new(max_depth: usize) -> Self {
        Self { max_depth }
    }

    fn depth_limit(&self) -> usize {
        self.max_depth.min(1024)
    }

    /// Rejects formulas deeper than the search limit *before* any recursive
    /// work: the entry clones below (and the per-rule clones in the search)
    /// recurse per connective, so a 15k-deep formula overflowed the thread
    /// stack in release builds (process abort) even though the search
    /// itself is depth-capped.
    fn check_formula_depth(&self, expr: &PropExpr) -> Result<()> {
        let limit = self.depth_limit();
        if formula_depth(expr) > limit {
            return Err(Error::Msg(format!(
                "Theorem proving depth limit {limit} exceeded"
            )));
        }
        Ok(())
    }

    /// Attempts to prove the given sequent $\Gamma \vdash \Delta$.
    /// Returns `Ok(true)` if valid (a theorem), `Ok(false)` if refutable, or Error on depth limit.
    pub fn prove(&self, sequent: &Sequent) -> Result<bool> {
        for expr in sequent.antecedent.iter().chain(sequent.succedent.iter()) {
            self.check_formula_depth(expr)?;
        }
        self.prove_recursive(&mut sequent.clone(), 0)
    }

    /// Verifies if a given formula is a tautology $\vdash A$.
    pub fn is_tautology(&self, expr: &PropExpr) -> Result<bool> {
        self.check_formula_depth(expr)?;
        let seq = Sequent::goal(expr.clone());
        self.prove(&seq)
    }

    fn prove_recursive(&self, seq: &mut Sequent, depth: usize) -> Result<bool> {
        // Hard cap: search recurses per connective, and each frame holds
        // cloned sequents (hundreds of bytes), so depths past ~1024 overrun
        // small thread stacks (process abort) instead of erroring. Real
        // propositional proofs nest dozens deep at most.
        let limit = self.depth_limit();
        if depth > limit {
            return Err(Error::Msg(format!(
                "Theorem proving depth limit {limit} exceeded"
            )));
        }

        // 1. Axiom check: if antecedent contains False or succedent contains True, proved!
        if seq.antecedent.contains(&PropExpr::False) || seq.succedent.contains(&PropExpr::True) {
            return Ok(true);
        }

        // Axiom rule: Gamma, A |- Delta, A (Identity Axiom)
        for a in &seq.antecedent {
            if seq.succedent.contains(a) {
                return Ok(true);
            }
        }

        // 2. Reduce non-atomic formulas in antecedent (Left rules)
        for (i, expr) in seq.antecedent.iter().enumerate() {
            match expr {
                PropExpr::Not(sub) => {
                    let sub = *sub.clone();
                    let mut next = seq.clone();
                    next.antecedent.remove(i);
                    next.succedent.push(sub);
                    return self.prove_recursive(&mut next, depth + 1);
                }
                PropExpr::And(left, right) => {
                    let left = *left.clone();
                    let right = *right.clone();
                    let mut next = seq.clone();
                    next.antecedent.remove(i);
                    next.antecedent.push(left);
                    next.antecedent.push(right);
                    return self.prove_recursive(&mut next, depth + 1);
                }
                PropExpr::Or(left, right) => {
                    let left = *left.clone();
                    let right = *right.clone();
                    let mut branch1 = seq.clone();
                    branch1.antecedent.remove(i);
                    branch1.antecedent.push(left);

                    let mut branch2 = seq.clone();
                    branch2.antecedent.remove(i);
                    branch2.antecedent.push(right);

                    let p1 = self.prove_recursive(&mut branch1, depth + 1)?;
                    let p2 = self.prove_recursive(&mut branch2, depth + 1)?;
                    return Ok(p1 && p2);
                }
                PropExpr::Implies(left, right) => {
                    // Gamma, A -> B |- Delta splits into (Gamma |- A, Delta) and (Gamma, B |- Delta)
                    let left = *left.clone();
                    let right = *right.clone();

                    let mut branch1 = seq.clone();
                    branch1.antecedent.remove(i);
                    branch1.succedent.push(left);

                    let mut branch2 = seq.clone();
                    branch2.antecedent.remove(i);
                    branch2.antecedent.push(right);

                    let p1 = self.prove_recursive(&mut branch1, depth + 1)?;
                    let p2 = self.prove_recursive(&mut branch2, depth + 1)?;
                    return Ok(p1 && p2);
                }
                PropExpr::Iff(left, right) => {
                    // A <-> B is equivalent to (A -> B) and (B -> A)
                    let left = *left.clone();
                    let right = *right.clone();
                    let imp1 = PropExpr::Implies(Box::new(left.clone()), Box::new(right.clone()));
                    let imp2 = PropExpr::Implies(Box::new(right), Box::new(left));

                    let mut next = seq.clone();
                    next.antecedent.remove(i);
                    next.antecedent.push(imp1);
                    next.antecedent.push(imp2);
                    return self.prove_recursive(&mut next, depth + 1);
                }
                _ => {}
            }
        }

        // 3. Reduce non-atomic formulas in succedent (Right rules)
        for (i, expr) in seq.succedent.iter().enumerate() {
            match expr {
                PropExpr::Not(sub) => {
                    let sub = *sub.clone();
                    let mut next = seq.clone();
                    next.succedent.remove(i);
                    next.antecedent.push(sub);
                    return self.prove_recursive(&mut next, depth + 1);
                }
                PropExpr::And(left, right) => {
                    // Gamma |- A and B, Delta splits into (Gamma |- A, Delta) and (Gamma |- B, Delta)
                    let left = *left.clone();
                    let right = *right.clone();

                    let mut branch1 = seq.clone();
                    branch1.succedent.remove(i);
                    branch1.succedent.push(left);

                    let mut branch2 = seq.clone();
                    branch2.succedent.remove(i);
                    branch2.succedent.push(right);

                    let p1 = self.prove_recursive(&mut branch1, depth + 1)?;
                    let p2 = self.prove_recursive(&mut branch2, depth + 1)?;
                    return Ok(p1 && p2);
                }
                PropExpr::Or(left, right) => {
                    let left = *left.clone();
                    let right = *right.clone();
                    let mut next = seq.clone();
                    next.succedent.remove(i);
                    next.succedent.push(left);
                    next.succedent.push(right);
                    return self.prove_recursive(&mut next, depth + 1);
                }
                PropExpr::Implies(left, right) => {
                    // Gamma |- A -> B, Delta reduces to Gamma, A |- B, Delta
                    let left = *left.clone();
                    let right = *right.clone();
                    let mut next = seq.clone();
                    next.succedent.remove(i);
                    next.antecedent.push(left);
                    next.succedent.push(right);
                    return self.prove_recursive(&mut next, depth + 1);
                }
                PropExpr::Iff(left, right) => {
                    let left = *left.clone();
                    let right = *right.clone();
                    let imp1 = PropExpr::Implies(Box::new(left.clone()), Box::new(right.clone()));
                    let imp2 = PropExpr::Implies(Box::new(right), Box::new(left));

                    let mut next = seq.clone();
                    next.succedent.remove(i);
                    next.succedent
                        .push(PropExpr::And(Box::new(imp1), Box::new(imp2)));
                    return self.prove_recursive(&mut next, depth + 1);
                }
                _ => {}
            }
        }

        // All elements are atomic variables and no identity axiom holds -> invalid sequent / counter-example found
        Ok(false)
    }
}
