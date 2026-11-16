//! Maxwell-Calladine Index Theorem & Kane-Lubensky Topological Mechanical Metamaterials (Calladine, 1978; Kane & Lubensky, 2014).
//!
//! ### 1. Maxwell-Calladine Index Theorem & Topological SSH Phonon Winding
//! In a $d$-dimensional ball-and-spring framework with $N_{\mathrm{sites}}$ vertices and $N_{\mathrm{bonds}}$ central-force bonds,
//! the rigidity matrix $Q$ and equilibrium matrix $Q^T$ obey the **Maxwell-Calladine Index Theorem**:
//! $$N_{\mathrm{zero}} - N_{\mathrm{self-stress}} = d \, N_{\mathrm{sites}} - N_{\mathrm{bonds}}$$
//! In an isostatic Maxwell lattice ($z = 2d$), Kane & Lubensky showed that the compatibility operator $Q(\mathbf{k})$ acts as a
//! supersymmetric Dirac partner of the dynamical matrix $D(\mathbf{k}) = Q^\dagger(\mathbf{k}) Q(\mathbf{k})$, with topological
//! winding number $\nu = \frac{1}{2\pi i} \oint d\ln \det Q(k)$ localizing zero-energy floppy modes or states of self-stress at boundaries!

/// Maxwell-Calladine Rigidity Index & Kane-Lubensky Topological Mechanics Engine.
#[derive(Debug, Clone, Copy)]
pub struct KaneLubenskyIsostaticEngine {
    /// Spatial dimension $d \ge 1$.
    pub dim_d: usize,
}

impl KaneLubenskyIsostaticEngine {
    /// Creates a new Kane-Lubensky Isostatic Mechanics engine.
    pub fn new(dim_d: usize) -> Self {
        Self {
            dim_d: dim_d.max(1),
        }
    }

    /// Evaluates the Maxwell-Calladine topological index $N_0 - N_s = d N_{\mathrm{sites}} - N_{\mathrm{bonds}}$ and checks the Maxwell isostatic coordination $z = 2 d$.
    pub fn maxwell_calladine_index(&self, num_sites: usize, num_bonds: usize) -> (i64, bool) {
        let dof = (self.dim_d * num_sites) as i64;
        let idx = dof - (num_bonds as i64);
        (idx, num_bonds == self.dim_d * num_sites)
    }

    /// Computes the 1D Kane-Lubensky topological winding number $\nu \in \{0, 1\}$ for dimerized compatibility matrix $q(k) = c_1 - c_2 e^{-i k}$.
    pub fn dimerized_compatibility_winding(c1: f64, c2: f64) -> i32 {
        if c2.abs() > c1.abs() { 1 } else { 0 }
    }
}
