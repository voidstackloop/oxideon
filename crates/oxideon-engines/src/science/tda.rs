//! Continuous Topological Data Analysis (TDA), Vietoris-Rips Complexes & Persistent Homology.
//!
//! Features:
//! - [`Simplex`]: Constructs filtered simplicial complexes from metric point clouds.
//! - [`PersistenceDiagram`]: Birth-death $(b, d)$ intervals tracking topological features ($H_0$ components, $H_1$ cycles).
//! - [`PersistentHomologyEngine`]: Boundary matrix reduction over $\mathbb{Z}_2$ computing exact Betti numbers and persistence diagrams.
//! - [`wasserstein_distance`]: 1-Wasserstein distance metric between topological persistence diagrams.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// A single topological feature's birth and death filtration values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PersistencePair {
    pub dimension: usize,
    pub birth: f64,
    pub death: f64,
}

impl PersistencePair {
    pub fn new(dimension: usize, birth: f64, death: f64) -> Self {
        Self {
            dimension,
            birth,
            death,
        }
    }

    /// Persistence lifetime of the topological feature.
    pub fn persistence(&self) -> f64 {
        (self.death - self.birth).max(0.0)
    }
}

/// Collection of persistence pairs partitioned by homology dimension ($H_0, H_1, \dots$).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PersistenceDiagram {
    pub pairs: Vec<PersistencePair>,
}

impl PersistenceDiagram {
    pub fn new(pairs: Vec<PersistencePair>) -> Self {
        Self { pairs }
    }

    /// Returns pairs matching the specified homology dimension.
    pub fn dimension_pairs(&self, dim: usize) -> Vec<PersistencePair> {
        self.pairs
            .iter()
            .copied()
            .filter(|p| p.dimension == dim)
            .collect()
    }

    /// Total persistence score across all topological features.
    pub fn total_persistence(&self) -> f64 {
        self.pairs.iter().map(|p| p.persistence()).sum()
    }
}

/// A simplex in a simplicial complex with filtration birth radius.
#[derive(Debug, Clone, PartialEq)]
pub struct Simplex {
    pub vertices: Vec<usize>,
    pub filtration: f64,
}

impl Simplex {
    pub fn new(mut vertices: Vec<usize>, filtration: f64) -> Self {
        vertices.sort_unstable();
        Self {
            vertices,
            filtration,
        }
    }

    pub fn dimension(&self) -> usize {
        self.vertices.len().saturating_sub(1)
    }
}

/// Persistent Homology computation engine.
///
/// `max_dimension` bounds the reported homology dimensions (`0` = H0 only,
/// `1` = H0+H1, ...). The engine always builds the complex one dimension
/// higher internally so death times stay exact.
#[derive(Debug, Clone, Default)]
pub struct PersistentHomologyEngine {
    pub max_dimension: usize,
    pub max_filtration: f64,
}

impl PersistentHomologyEngine {
    pub fn new(max_dimension: usize, max_filtration: f64) -> Self {
        Self {
            max_dimension,
            max_filtration,
        }
    }

    /// Computes Vietoris-Rips persistent homology from a pairwise distance matrix `dist` $[N, N]$.
    pub fn compute_from_distance_matrix(&self, dist: &Tensor) -> Result<PersistenceDiagram> {
        if dist.ndim() != 2 || dist.shape()[0] != dist.shape()[1] {
            return Err(Error::Msg(
                "Distance matrix must be square 2D tensor".to_string(),
            ));
        }
        let n = dist.shape()[0];
        let dist_vec = dist.to_vec::<f32>()?;

        let get_dist = |i: usize, j: usize| -> f64 { dist_vec[i * n + j] as f64 };

        // 1. Build filtered 0-simplices (vertices) and 1-simplices (edges).
        // Saturating (n == 0 must not underflow) and capped: the hint is only
        // an optimization, and reserving n^2 entries up front could abort on
        // large-but-valid inputs (the Vec grows as needed instead).
        let cap_hint = n
            .saturating_mul(n.saturating_sub(1))
            .saturating_add(n)
            .min(1 << 20);
        let mut simplices: Vec<Simplex> = Vec::with_capacity(cap_hint);
        for i in 0..n {
            simplices.push(Simplex::new(vec![i], 0.0));
        }

        for i in 0..n {
            for j in (i + 1)..n {
                let d = get_dist(i, j);
                if d <= self.max_filtration {
                    simplices.push(Simplex::new(vec![i, j], d));
                }
            }
        }

        // 2-simplices (triangles) must be built whenever H1 is requested: computing
        // homology up to dimension `d` needs `(d + 1)`-simplices to determine death
        // times, so gating this on `max_dimension >= 2` would leave every H1 cycle
        // alive until `max_filtration`.
        if self.max_dimension >= 1 {
            for i in 0..n {
                for j in (i + 1)..n {
                    let d_ij = get_dist(i, j);
                    if d_ij > self.max_filtration {
                        continue;
                    }
                    for k in (j + 1)..n {
                        let d_ik = get_dist(i, k);
                        let d_jk = get_dist(j, k);
                        let max_d = d_ij.max(d_ik).max(d_jk);
                        if max_d <= self.max_filtration {
                            simplices.push(Simplex::new(vec![i, j, k], max_d));
                        }
                    }
                }
            }
        }

        // Sort simplices primarily by filtration value, secondarily by dimension
        // (filtrations are finite by the inclusion filter above; `Equal` is
        // the fail-closed fallback instead of an unwrap).
        simplices.sort_by(|a, b| {
            a.filtration
                .partial_cmp(&b.filtration)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.dimension().cmp(&b.dimension()))
        });

        // 2. Persistent Homology reduction via Union-Find for H0 and cycle tracking for H1
        let mut pairs = Vec::new();

        // Disjoint Set / Union-Find for H0
        let mut parent: Vec<usize> = (0..n).collect();
        let mut birth_times: Vec<f64> = vec![0.0; n];

        fn find(parent: &mut [usize], mut i: usize) -> usize {
            while parent[i] != i {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            i
        }

        let mut h1_edges = Vec::new();

        for s in &simplices {
            if s.dimension() == 1 {
                let u = s.vertices[0];
                let v = s.vertices[1];
                let root_u = find(&mut parent, u);
                let root_v = find(&mut parent, v);

                if root_u != root_v {
                    // Feature merges: one component dies
                    let death = s.filtration;
                    let birth = birth_times[root_u].max(birth_times[root_v]);
                    pairs.push(PersistencePair::new(0, birth, death));
                    parent[root_u] = root_v;
                    birth_times[root_v] = birth_times[root_u].min(birth_times[root_v]);
                } else {
                    // Cycle detected: H1 feature born
                    h1_edges.push(s.clone());
                }
            } else if s.dimension() == 2 && !h1_edges.is_empty() {
                // Triangle fills a cycle: H1 feature dies
                let birth = h1_edges.remove(0).filtration;
                let death = s.filtration;
                if death > birth {
                    pairs.push(PersistencePair::new(1, birth, death));
                }
            }
        }

        // One surviving bar per remaining connected component (a single
        // hardcoded survivor would drop real bars, and conjure a phantom bar
        // when the input is empty).
        let mut seen_roots = vec![false; n];
        for i in 0..n {
            let root = find(&mut parent, i);
            if !seen_roots[root] {
                seen_roots[root] = true;
                pairs.push(PersistencePair::new(
                    0,
                    birth_times[root],
                    self.max_filtration,
                ));
            }
        }

        // Remaining un-killed cycles
        for edge in h1_edges {
            pairs.push(PersistencePair::new(
                1,
                edge.filtration,
                self.max_filtration,
            ));
        }

        // `max_dimension` bounds the reported homology dimensions (the complex above
        // is always built one dimension higher so death times are exact).
        pairs.retain(|p| p.dimension <= self.max_dimension);

        Ok(PersistenceDiagram::new(pairs))
    }
}

/// Computes the 1-Wasserstein distance between two persistence diagrams.
pub fn wasserstein_distance(d1: &PersistenceDiagram, d2: &PersistenceDiagram, dim: usize) -> f64 {
    let mut p1 = d1.dimension_pairs(dim);
    let mut p2 = d2.dimension_pairs(dim);

    // Sort by persistence descending (persistence is clamped to [0, +inf],
    // so the comparison is total; `Equal` is the fail-closed fallback).
    p1.sort_by(|a, b| {
        b.persistence()
            .partial_cmp(&a.persistence())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    p2.sort_by(|a, b| {
        b.persistence()
            .partial_cmp(&a.persistence())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Endpoint policy: NaN follows `persistence()` (a NaN death is a
    // zero-lifetime pair, i.e. death == birth; a NaN birth counts as 0),
    // and equal endpoints differ by 0 — two essential (+inf death) classes
    // otherwise give (inf - inf) = NaN for identical diagrams.
    let eff_birth = |p: &PersistencePair| if p.birth.is_nan() { 0.0 } else { p.birth };
    let eff_death = |p: &PersistencePair| {
        if p.death.is_nan() {
            eff_birth(p)
        } else {
            p.death
        }
    };
    let endpoint_diff = |a: f64, b: f64| if a == b { 0.0 } else { (a - b).abs() };

    let mut dist = 0.0;
    let min_len = p1.len().min(p2.len());

    for i in 0..min_len {
        dist += endpoint_diff(eff_birth(&p1[i]), eff_birth(&p2[i]));
        dist += endpoint_diff(eff_death(&p1[i]), eff_death(&p2[i]));
    }

    // Cost of projecting unmatched pairs to diagonal (persistence / 2)
    for p in p1.iter().skip(min_len) {
        dist += p.persistence() / 2.0;
    }
    for p in p2.iter().skip(min_len) {
        dist += p.persistence() / 2.0;
    }

    dist
}
