//! Topos Theory, Intuitionistic Heyting Algebras, Subobject Classifiers $\Omega$ &
//! Lawvere-Tierney Modal Topologies.
//!
//! ### 1. Subobject Classifier $\Omega$ & Heyting Implication
//! In an elementary topos $\mathcal{E}$ (such as the category of sheaves $\operatorname{Sh}(X)$ on a topological space $X$),
//! truth values form a complete **Heyting Algebra** $(\Omega, \wedge, \vee, \Rightarrow, \bot, \top)$.
//! For open-set truth degrees $u, v \in [0, 1]$ (or finite Alexandroff/chain topologies $0 = v_0 < v_1 < \dots < v_m = 1$),
//! Heyting implication is defined by the adjunction $(w \wedge u \le v) \iff (w \le (u \Rightarrow v))$:
//! $$u \Rightarrow v = \begin{cases} 1.0 & \text{if } u \le v \\ v & \text{if } u > v \end{cases}$$
//! And intuitionistic negation is $\neg u = (u \Rightarrow 0)$.
//! For any intermediate truth value $0 < u < 1$:
//! • $\neg u = 0$, so $\neg\neg u = 1 > u$ (double negation introduction $u \le \neg\neg u$ holds, but $\neg\neg u = u$ fails!).
//! • $u \vee \neg u = u < 1$ (Law of Excluded Middle fails for non-clopen propositions!).
//!
//! ### 2. Lawvere-Tierney Topology $j: \Omega \to \Omega$
//! A modal closure operator $j: \Omega \to \Omega$ is a Lawvere-Tierney topology iff for all $u, v \in \Omega$:
//! • $j(\top) = \top$
//! • $j(j(u)) = j(u)$
//! • $j(u \wedge v) = j(u) \wedge j(v)$
//! The **Double-Negation Topology** $j_{\neg\neg}(u) = \neg\neg u$ is a valid Lawvere-Tierney topology whose
//! $j$-sheaves $\operatorname{Sh}_{\neg\neg}(\mathcal{E})$ form a classical Boolean topos (Glivenko's theorem).

use crate::error::{Error, Result};

/// Topos Subobject Classifier $\Omega$ & Intuitionistic Heyting Logic Engine.
#[derive(Debug, Clone)]
pub struct ToposHeytingEngine {
    /// Ordered finite set of truth degrees $0.0 = v_0 < v_1 < \dots < v_k = 1.0$ in $\Omega$.
    pub truth_lattice: Vec<f64>,
}

impl ToposHeytingEngine {
    /// Constructs a finite linear Heyting algebra subobject classifier $\Omega$ with `num_levels` truth values ($\ge 2$).
    pub fn new(num_levels: usize) -> Result<Self> {
        if num_levels < 2 {
            return Err(Error::Msg(
                "Heyting lattice requires at least 2 levels (0 and 1)".to_string(),
            ));
        }
        crate::limits::check_repeats("num_levels", num_levels)?;
        let truth_lattice = (0..num_levels)
            .map(|i| (i as f64) / ((num_levels - 1) as f64))
            .collect();
        Ok(Self { truth_lattice })
    }

    /// Conjunction (meet) $u \wedge v = \min(u, v)$.
    #[inline]
    pub fn meet(u: f64, v: f64) -> f64 {
        u.min(v)
    }

    /// Disjunction (join) $u \vee v = \max(u, v)$.
    #[inline]
    pub fn join(u: f64, v: f64) -> f64 {
        u.max(v)
    }

    /// Heyting intuitionistic implication $u \Rightarrow v = \sup \{ w \in \Omega \mid w \wedge u \le v \}$.
    #[inline]
    pub fn imply(u: f64, v: f64) -> f64 {
        if u <= v + 1e-14 { 1.0 } else { v }
    }

    /// Intuitionistic pseudo-complement (negation) $\neg u = (u \Rightarrow 0)$.
    #[inline]
    pub fn negate(u: f64) -> f64 {
        Self::imply(u, 0.0)
    }

    /// Lawvere-Tierney double-negation modal operator $j_{\neg\neg}(u) = \neg(\neg u)$.
    #[inline]
    pub fn double_negation_modality(u: f64) -> f64 {
        Self::negate(Self::negate(u))
    }

    /// Verifies the fundamental Heyting residuation adjunction across all triples $(w, u, v) \in \Omega^3$:
    /// $(w \wedge u \le v) \iff (w \le (u \Rightarrow v))$.
    ///
    /// The scan is cubic in the lattice size (measured 25 s at k=3000), so
    /// lattices past 1024 levels are rejected; real lattices are dozens wide.
    pub fn verify_heyting_adjunction(&self) -> Result<bool> {
        Self::check_verify_size(self.truth_lattice.len())?;
        for &w in &self.truth_lattice {
            for &u in &self.truth_lattice {
                for &v in &self.truth_lattice {
                    let lhs = Self::meet(w, u) <= v + 1e-12;
                    let rhs = w <= Self::imply(u, v) + 1e-12;
                    if lhs != rhs {
                        return Ok(false);
                    }
                }
            }
        }
        Ok(true)
    }

    /// Rejects lattices too large to verify: the adjunction scan is cubic.
    fn check_verify_size(levels: usize) -> Result<()> {
        if levels > 1024 {
            return Err(Error::Msg(format!(
                "Topos verify: {levels} lattice levels exceed the 1024 verification cap"
            )));
        }
        Ok(())
    }

    /// Verifies the three Lawvere-Tierney topology axioms for $j_{\neg\neg}$ across $\Omega$:
    /// 1. $j(1) == 1$
    /// 2. $j(j(u)) == j(u)$
    /// 3. $j(u \wedge v) == j(u) \wedge j(v)$
    ///
    /// Same cubic-scale guard as [`Self::verify_heyting_adjunction`].
    pub fn verify_lawvere_tierney_axioms(&self) -> Result<bool> {
        Self::check_verify_size(self.truth_lattice.len())?;
        if (Self::double_negation_modality(1.0) - 1.0).abs() > 1e-12 {
            return Ok(false);
        }
        for &u in &self.truth_lattice {
            let ju = Self::double_negation_modality(u);
            let jju = Self::double_negation_modality(ju);
            if (ju - jju).abs() > 1e-12 {
                return Ok(false);
            }
            for &v in &self.truth_lattice {
                let j_uv = Self::double_negation_modality(Self::meet(u, v));
                let ju_jv = Self::meet(ju, Self::double_negation_modality(v));
                if (j_uv - ju_jv).abs() > 1e-12 {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }
}
