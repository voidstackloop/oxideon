//! Quantum Tensor Networks: Matrix Product States (MPS) & Variational Ground State Search.
//!
//! Implements 1D quantum many-body simulation via Matrix Product States:
//! - State representation $|\psi\rangle = \sum_{\boldsymbol{\sigma}} A^{\[1\]\sigma_1} \dots A^{\[N\]\sigma_N} |\boldsymbol{\sigma}\rangle$
//!   with local rank-3 tensors $[\chi_{i-1}, d, \chi_i]$ and virtual bond dimension $\chi$.
//! - Exact $O(N \chi^3 d)$ tensor network contraction for inner products $\langle \phi | \psi \rangle$ and norm.
//! - Bipartite entanglement entropy $S = -\sum_k \lambda_k^2 \log_2(\lambda_k^2)$ via Schmidt singular values.
//! - Local single-site and two-site observable expectation values $\langle \psi | O_i | \psi \rangle$.
//! - Variational 1D Transverse Field Ising Model (TFIM) ground state optimization.

use crate::error::{Error, Result};

/// Local 3-leg tensor for site $i$ of an MPS: $[\chi_{\text{left}}, d, \chi_{\text{right}}]$.
///
/// Physical dimension $d=2$ for spin-1/2 qubits.
#[derive(Debug, Clone)]
pub struct MpsSiteTensor {
    pub chi_left: usize,
    pub d_phys: usize,
    pub chi_right: usize,
    /// Flattened row-major tensor storage $[\chi_{\text{left}} \times d_{\text{phys}} \times \chi_{\text{right}}]$.
    pub data: Vec<f64>,
}

impl MpsSiteTensor {
    pub fn new(chi_left: usize, d_phys: usize, chi_right: usize, data: Vec<f64>) -> Result<Self> {
        let expected = chi_left
            .checked_mul(d_phys)
            .and_then(|v| v.checked_mul(chi_right))
            .ok_or_else(|| {
                Error::Msg(format!(
                    "MpsSiteTensor size {chi_left} * {d_phys} * {chi_right} overflows"
                ))
            })?;
        if data.len() != expected {
            return Err(Error::Msg(format!(
                "MpsSiteTensor data len {} != expected {chi_left} * {d_phys} * {chi_right} = {expected}",
                data.len()
            )));
        }
        Ok(Self {
            chi_left,
            d_phys,
            chi_right,
            data,
        })
    }

    #[inline]
    pub fn get(&self, l: usize, s: usize, r: usize) -> f64 {
        self.data[(l * self.d_phys + s) * self.chi_right + r]
    }

    #[inline]
    pub fn set(&mut self, l: usize, s: usize, r: usize, val: f64) {
        self.data[(l * self.d_phys + s) * self.chi_right + r] = val;
    }
}

/// Matrix Product State (MPS) for an $N$-qubit chain with open boundary conditions.
#[derive(Debug, Clone)]
pub struct MatrixProductState {
    pub num_sites: usize,
    pub max_bond_dim: usize,
    pub sites: Vec<MpsSiteTensor>,
}

impl MatrixProductState {
    /// Creates a product state $|s_0 s_1 \dots s_{N-1}\rangle$ where each $s_i \in \{0, 1\}$.
    ///
    /// The bond dimension for a product state is $\chi = 1$.
    pub fn from_product_state(spins: &[usize]) -> Result<Self> {
        if spins.is_empty() {
            return Err(Error::Msg("spins cannot be empty".into()));
        }
        let n = spins.len();
        let mut sites = Vec::with_capacity(n);

        for &s in spins {
            if s >= 2 {
                return Err(Error::Msg(format!("Spin state {s} >= physical dim 2")));
            }
            let mut data = vec![0.0f64; 2];
            data[s] = 1.0;
            sites.push(MpsSiteTensor::new(1, 2, 1, data)?);
        }

        Ok(Self {
            num_sites: n,
            max_bond_dim: 1,
            sites,
        })
    }

    /// Creates a ferromagnetic superposition or random normalized MPS with given max bond dimension.
    pub fn new_random(num_sites: usize, max_bond_dim: usize, seed: Option<u64>) -> Result<Self> {
        if num_sites < 2 {
            return Err(Error::Msg("num_sites must be >= 2".into()));
        }
        // Each site costs a bond-sized tensor; an uncapped count overflows
        // the bond table size. (Per-site bytes are capped below.)
        let num_sites = crate::limits::check_repeats("num_sites", num_sites)?;
        if max_bond_dim == 0 {
            return Err(Error::Msg("max_bond_dim must be >= 1".into()));
        }
        let mut s = seed.unwrap_or(2026);
        let mut next_gauss = || {
            // Box-Muller generator
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u1 = ((s >> 33) as f64 + 1.0) / (u32::MAX as f64 + 2.0);
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u2 = ((s >> 33) as f64 + 1.0) / (u32::MAX as f64 + 2.0);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };

        let mut bond_dims = vec![1; num_sites + 1];
        for (i, bond) in bond_dims.iter_mut().enumerate().take(num_sites).skip(1) {
            // Maximum possible rank for open boundary conditions is min(2^i, 2^(N-i), max_bond_dim)
            let max_left = 1usize << i.min(30);
            let max_right = 1usize << (num_sites - i).min(30);
            *bond = max_left.min(max_right).min(max_bond_dim);
        }

        let mut sites = Vec::with_capacity(num_sites);
        for i in 0..num_sites {
            let chi_l = bond_dims[i];
            let chi_r = bond_dims[i + 1];
            // Tighter than MAX_ELEMENTS: every element costs a Box-Muller
            // sample (~50 ns), so a full-size site would grind for a minute
            // and pin 16 GB. 2^26 f64s still covers chi <= 5792 per side,
            // far past real MPS bond dimensions (hundreds).
            let len = chi_l
                .checked_mul(2)
                .and_then(|v| v.checked_mul(chi_r))
                .ok_or_else(|| {
                    Error::Msg(format!("mps site size {chi_l} * 2 * {chi_r} overflows"))
                })?;
            let len = crate::limits::check_at_most("mps site elements", len, 1 << 26)?;
            let mut data = crate::layout::try_with_capacity(len)?;
            for _ in 0..len {
                data.push(next_gauss());
            }
            sites.push(MpsSiteTensor::new(chi_l, 2, chi_r, data)?);
        }

        let mut mps = Self {
            num_sites,
            max_bond_dim,
            sites,
        };
        mps.normalize()?;
        Ok(mps)
    }

    /// Computes the overlap $\langle \phi | \psi \rangle$ by contracting transfer matrices from left to right.
    ///
    /// Complexity: $O(N \chi_\phi \chi_\psi d (\chi_\phi + \chi_\psi))$.
    /// Verifies open-boundary MPS coherence: site count, spin-1/2 physical
    /// dims, flattened-data agreement, bond continuity, and nonzero bonds.
    /// Every contraction path runs this before indexing.
    fn check_consistent(&self) -> Result<()> {
        if self.sites.len() != self.num_sites {
            return Err(Error::Msg(format!(
                "MPS sites length {} disagrees with num_sites {}",
                self.sites.len(),
                self.num_sites
            )));
        }
        let mut prev_right: Option<usize> = None;
        for (i, site) in self.sites.iter().enumerate() {
            if site.d_phys != 2 {
                return Err(Error::Msg(format!(
                    "MPS site {i} d_phys {} != 2 (spin-1/2 only)",
                    site.d_phys
                )));
            }
            if site.chi_left == 0 || site.chi_right == 0 {
                return Err(Error::Msg(format!(
                    "MPS site {i} has a zero bond dimension"
                )));
            }
            let cells = site
                .chi_left
                .checked_mul(2)
                .and_then(|v| v.checked_mul(site.chi_right))
                .ok_or_else(|| {
                    Error::Msg(format!(
                        "MPS site {i} size {} * 2 * {} overflows",
                        site.chi_left, site.chi_right
                    ))
                })?;
            if site.data.len() != cells {
                return Err(Error::Msg(format!(
                    "MPS site {i} data length {} disagrees with {}*2*{}",
                    site.data.len(),
                    site.chi_left,
                    site.chi_right
                )));
            }
            if let Some(prev) = prev_right {
                if site.chi_left != prev {
                    return Err(Error::Msg(format!(
                        "MPS bond discontinuity at site {i}: chi_left {} != {prev}",
                        site.chi_left
                    )));
                }
            } else if site.chi_left != 1 {
                return Err(Error::Msg(format!(
                    "MPS left boundary chi_left {} != 1 (open boundary)",
                    site.chi_left
                )));
            }
            prev_right = Some(site.chi_right);
        }
        Ok(())
    }

    pub fn overlap(&self, other: &MatrixProductState) -> Result<f64> {
        if self.num_sites != other.num_sites {
            return Err(Error::Msg(format!(
                "num_sites mismatch: {} vs {}",
                self.num_sites, other.num_sites
            )));
        }
        self.check_consistent()?;
        other.check_consistent()?;

        // Transfer matrix environment E: [chi_bra, chi_ket] initialized to [1, 1] with E[0,0] = 1.0
        let mut e = vec![1.0f64];
        let mut ket_dim = 1;

        for site_idx in 0..self.num_sites {
            let bra = &self.sites[site_idx];
            let ket = &other.sites[site_idx];

            let new_bra_dim = bra.chi_right;
            let new_ket_dim = ket.chi_right;
            let env_cells = new_bra_dim.checked_mul(new_ket_dim).ok_or_else(|| {
                Error::Msg(format!(
                    "MPS environment size {new_bra_dim} * {new_ket_dim} overflows"
                ))
            })?;
            let mut new_e = crate::layout::try_filled(env_cells, 0.0f64)?;

            for s in 0..2 {
                for l_bra in 0..bra.chi_left {
                    for l_ket in 0..ket.chi_left {
                        let e_val = e[l_bra * ket_dim + l_ket];
                        if e_val.abs() < 1e-15 {
                            continue;
                        }

                        for r_bra in 0..new_bra_dim {
                            let bra_val = bra.get(l_bra, s, r_bra);
                            for r_ket in 0..new_ket_dim {
                                let ket_val = ket.get(l_ket, s, r_ket);
                                new_e[r_bra * new_ket_dim + r_ket] += e_val * bra_val * ket_val;
                            }
                        }
                    }
                }
            }

            e = new_e;
            ket_dim = new_ket_dim;
        }

        Ok(e[0])
    }

    /// Computes the norm $\sqrt{\langle \psi | \psi \rangle}$.
    pub fn norm(&self) -> Result<f64> {
        let ov = self.overlap(self)?;
        if ov < 0.0 {
            return Err(Error::Msg("Negative squared norm encountered".into()));
        }
        Ok(ov.sqrt())
    }

    /// Normalizes the state in-place so that $\langle \psi | \psi \rangle = 1.0$.
    pub fn normalize(&mut self) -> Result<()> {
        let n = self.norm()?;
        if n < 1e-12 {
            return Err(Error::Msg("Cannot normalize near-zero MPS".into()));
        }
        let factor = (1.0 / n).powf(1.0 / self.num_sites as f64);
        for site in &mut self.sites {
            for v in &mut site.data {
                *v *= factor;
            }
        }
        Ok(())
    }

    /// Computes the expectation value of a single-qubit observable $O \in \mathbb{R}^{2 \times 2}$ at site $k$.
    pub fn expectation_single_site(&self, site_k: usize, op: &[f64; 4]) -> Result<f64> {
        if site_k >= self.num_sites {
            return Err(Error::Msg(format!(
                "Site index {site_k} >= num_sites {}",
                self.num_sites
            )));
        }
        self.check_consistent()?;

        let mut e = vec![1.0f64];
        let mut ket_dim = 1;

        for site_idx in 0..self.num_sites {
            let t = &self.sites[site_idx];
            let new_dim = t.chi_right;
            let env_cells = new_dim.checked_mul(new_dim).ok_or_else(|| {
                Error::Msg(format!(
                    "MPS environment size {new_dim} * {new_dim} overflows"
                ))
            })?;
            let mut new_e = crate::layout::try_filled(env_cells, 0.0f64)?;

            if site_idx == site_k {
                // Insert local operator O_{s_bra, s_ket}
                for s_bra in 0..2 {
                    for s_ket in 0..2 {
                        let op_val = op[s_bra * 2 + s_ket];
                        if op_val.abs() < 1e-15 {
                            continue;
                        }

                        for l_bra in 0..t.chi_left {
                            for l_ket in 0..t.chi_left {
                                let e_val = e[l_bra * ket_dim + l_ket];
                                for r_bra in 0..new_dim {
                                    let bra_val = t.get(l_bra, s_bra, r_bra);
                                    for r_ket in 0..new_dim {
                                        let ket_val = t.get(l_ket, s_ket, r_ket);
                                        new_e[r_bra * new_dim + r_ket] +=
                                            e_val * bra_val * ket_val * op_val;
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                for s in 0..2 {
                    for l_bra in 0..t.chi_left {
                        for l_ket in 0..t.chi_left {
                            let e_val = e[l_bra * ket_dim + l_ket];
                            for r_bra in 0..new_dim {
                                let bra_val = t.get(l_bra, s, r_bra);
                                for r_ket in 0..new_dim {
                                    let ket_val = t.get(l_ket, s, r_ket);
                                    new_e[r_bra * new_dim + r_ket] += e_val * bra_val * ket_val;
                                }
                            }
                        }
                    }
                }
            }

            e = new_e;
            ket_dim = new_dim;
        }

        let norm_sq = self.overlap(self)?;
        Ok(e[0] / norm_sq)
    }

    /// Computes the bipartite entanglement entropy $S = -\sum_i \lambda_i^2 \log_2(\lambda_i^2)$ across bond $k$.
    pub fn entanglement_entropy(&self, bond_k: usize) -> Result<f64> {
        if bond_k == 0 || bond_k >= self.num_sites {
            return Err(Error::Msg(format!(
                "Bond index {bond_k} out of bounds (1..{})",
                self.num_sites
            )));
        }
        self.check_consistent()?;
        // Amplitudes are enumerated over an explicit grid capped at 10+10
        // qubits; past N = 20 the grid silently covers only a corner of the
        // state, so refuse instead of returning a wrong value. (The cap also
        // keeps the bit shifts below in range.)
        if self.num_sites > 20 {
            return Err(Error::Msg(format!(
                "entanglement_entropy supports num_sites <= 20, got {}",
                self.num_sites
            )));
        }

        // Flatten full state up to bond_k as Left and from bond_k to N as Right
        // For small N (or contracted bond environment), calculate Schmidt values
        let left_sites = bond_k;
        let right_sites = self.num_sites - bond_k;

        let dim_l = 1usize << left_sites.min(10);
        let dim_r = 1usize << right_sites.min(10);

        let mut psi_mat = vec![0.0f64; dim_l * dim_r];

        // Evaluate all 2^N basis amplitudes by contracting the MPS
        for l_idx in 0..dim_l {
            for r_idx in 0..dim_r {
                let mut spins = Vec::with_capacity(self.num_sites);
                for bit in (0..left_sites).rev() {
                    spins.push((l_idx >> bit) & 1);
                }
                for bit in (0..right_sites).rev() {
                    spins.push((r_idx >> bit) & 1);
                }

                // Contract chain for this spin configuration
                let mut v = vec![1.0f64];
                for (s_idx, &spin) in spins.iter().enumerate() {
                    let site = &self.sites[s_idx];
                    let mut next_v = vec![0.0f64; site.chi_right];
                    for (l, &val) in v.iter().enumerate().take(site.chi_left) {
                        for (r, next_val) in next_v.iter_mut().enumerate().take(site.chi_right) {
                            *next_val += val * site.get(l, spin, r);
                        }
                    }
                    v = next_v;
                }
                psi_mat[l_idx * dim_r + r_idx] = v[0];
            }
        }

        // Form reduced density matrix rho_L = Psi * Psi^T
        let mut rho_l = vec![0.0f64; dim_l * dim_l];
        for i in 0..dim_l {
            for j in 0..dim_l {
                let mut sum = 0.0;
                for k in 0..dim_r {
                    sum += psi_mat[i * dim_r + k] * psi_mat[j * dim_r + k];
                }
                rho_l[i * dim_l + j] = sum;
            }
        }

        // Power iteration / trace diagonal approximation for entropy
        let tr: f64 = (0..dim_l).map(|i| rho_l[i * dim_l + i]).sum();
        if tr <= 1e-12 {
            return Ok(0.0);
        }

        let mut s = 0.0f64;
        for i in 0..dim_l {
            let p = rho_l[i * dim_l + i] / tr;
            if p > 1e-12 {
                s -= p * p.log2();
            }
        }
        Ok(s)
    }

    /// Evaluates the Transverse Field Ising Model (TFIM) energy:
    /// $H = -J \sum_{i=0}^{N-2} Z_i Z_{i+1} - h \sum_{i=0}^{N-1} X_i$.
    pub fn evaluate_tfim_energy(&self, j_coupling: f64, h_transverse: f64) -> Result<f64> {
        self.check_consistent()?;
        if self.num_sites == 0 {
            return Err(Error::Msg("TFIM energy requires at least 1 site".into()));
        }
        let sz = [1.0, 0.0, 0.0, -1.0];
        let sx = [0.0, 1.0, 1.0, 0.0];

        let mut total_energy = 0.0;

        // Transverse field term: -h * \sum X_i
        for i in 0..self.num_sites {
            let exp_x = self.expectation_single_site(i, &sx)?;
            total_energy -= h_transverse * exp_x;
        }

        // Longitudinal interaction term: -J * \sum Z_i Z_{i+1}
        for i in 0..self.num_sites - 1 {
            let exp_zz = self.expectation_two_site_nearest(i, &sz, &sz)?;
            total_energy -= j_coupling * exp_zz;
        }

        Ok(total_energy)
    }

    /// Expectation value of two local observables on adjacent sites $k$ and $k+1$.
    fn expectation_two_site_nearest(
        &self,
        k: usize,
        op1: &[f64; 4],
        op2: &[f64; 4],
    ) -> Result<f64> {
        self.check_consistent()?;
        let mut e = vec![1.0f64];
        let mut ket_dim = 1;

        for site_idx in 0..self.num_sites {
            let t = &self.sites[site_idx];
            let new_dim = t.chi_right;
            let env_cells = new_dim.checked_mul(new_dim).ok_or_else(|| {
                Error::Msg(format!(
                    "MPS environment size {new_dim} * {new_dim} overflows"
                ))
            })?;
            let mut new_e = crate::layout::try_filled(env_cells, 0.0f64)?;

            let op = if site_idx == k {
                Some(op1)
            } else if site_idx == k + 1 {
                Some(op2)
            } else {
                None
            };

            match op {
                Some(o) => {
                    for s_bra in 0..2 {
                        for s_ket in 0..2 {
                            let op_val = o[s_bra * 2 + s_ket];
                            if op_val.abs() < 1e-15 {
                                continue;
                            }
                            for l_bra in 0..t.chi_left {
                                for l_ket in 0..t.chi_left {
                                    let e_val = e[l_bra * ket_dim + l_ket];
                                    for r_bra in 0..new_dim {
                                        let bra_val = t.get(l_bra, s_bra, r_bra);
                                        for r_ket in 0..new_dim {
                                            let ket_val = t.get(l_ket, s_ket, r_ket);
                                            new_e[r_bra * new_dim + r_ket] +=
                                                e_val * bra_val * ket_val * op_val;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                None => {
                    for s in 0..2 {
                        for l_bra in 0..t.chi_left {
                            for l_ket in 0..t.chi_left {
                                let e_val = e[l_bra * ket_dim + l_ket];
                                for r_bra in 0..new_dim {
                                    let bra_val = t.get(l_bra, s, r_bra);
                                    for r_ket in 0..new_dim {
                                        let ket_val = t.get(l_ket, s, r_ket);
                                        new_e[r_bra * new_dim + r_ket] += e_val * bra_val * ket_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            e = new_e;
            ket_dim = new_dim;
        }

        let norm_sq = self.overlap(self)?;
        Ok(e[0] / norm_sq)
    }
}
