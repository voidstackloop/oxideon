//! Neural Architecture Search (NAS) & Zero-Cost Architecture Scoring Engine.
//!
//! Evaluates candidate architectures at initialization without backprop training:
//! - [`compute_synflow`]: Synaptic Saliency Flow metric evaluating gradient flow capacity without data bias.
//! - [`ArchitectureGenome`]: Parameterized representation of layer dimensions, depths, and activations.
//! - [`EvolutionaryAutoML`]: Population-based genetic search discovering Pareto-optimal architectures under parameter budgets.

use crate::error::Result;
use crate::nn::Linear;
use crate::rng::Rng;
use crate::tensor::Tensor;

/// Genome encoding an architecture's structural hyperparameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchitectureGenome {
    pub hidden_dims: Vec<usize>,
    pub num_heads: usize,
    pub expansion_factor: usize,
}

impl ArchitectureGenome {
    pub fn new(hidden_dims: Vec<usize>, num_heads: usize, expansion_factor: usize) -> Self {
        Self {
            hidden_dims,
            num_heads: num_heads.max(1),
            expansion_factor: expansion_factor.max(1),
        }
    }

    /// Computes approximate parameter count for a multi-layer stack.
    ///
    /// Saturating: the estimate of an absurdly large stack is "at least `usize::MAX`",
    /// which callers treat as over budget. It is never used as an allocation size.
    pub fn estimate_parameters(&self, input_dim: usize, output_dim: usize) -> usize {
        let mut count = 0usize;
        let mut prev = input_dim;
        for &h in &self.hidden_dims {
            count = count.saturating_add(prev.saturating_mul(h).saturating_add(h));
            let exp = h.saturating_mul(self.expansion_factor);
            count = count.saturating_add(h.saturating_mul(exp).saturating_add(exp));
            prev = h;
        }
        count.saturating_add(prev.saturating_mul(output_dim).saturating_add(output_dim))
    }
}

/// Zero-cost proxy score metrics for an un-trained network.
#[derive(Debug, Clone)]
pub struct ZeroCostScore {
    pub synflow_score: f64,
    pub parameter_efficiency: f64,
    pub combined_fitness: f64,
}

/// Computes the SynFlow metric: R = sum |theta * dR/dtheta| on all-ones input.
///
/// SynFlow preserves expressivity and avoids gradient collapse by evaluating
/// contraction across weight matrices.
pub fn compute_synflow(weights: &[Tensor], input_dim: usize) -> Result<f64> {
    if weights.is_empty() {
        return Ok(0.0);
    }

    // Input: tensor of ones with requires_grad
    let mut current = Tensor::ones(
        &[1, input_dim],
        crate::dtype::DType::F32,
        &crate::device::Device::Cpu,
    )?;

    // Linear chain pass
    for w in weights {
        let abs_w = w.abs()?;
        // Forward through positive weight proxy
        current = current.matmul(&abs_w.transpose(0, 1)?)?;
    }

    // Scalar surrogate output
    let scalar = current.sum()?.item::<f32>()? as f64;
    // Logarithmic synflow score to avoid overflow on deep networks
    let score = (scalar.abs() + 1e-12).ln();
    Ok(score)
}

/// Continuous self-evolving population search for optimal architectures.
#[derive(Debug, Clone)]
pub struct EvolutionaryAutoML {
    pub population_size: usize,
    pub max_generations: usize,
    pub mutation_rate: f64,
    pub max_param_budget: usize,
    rng: Rng,
}

impl EvolutionaryAutoML {
    pub fn new(
        population_size: usize,
        max_generations: usize,
        mutation_rate: f64,
        max_param_budget: usize,
        seed: u64,
    ) -> Self {
        Self {
            population_size: population_size.max(4),
            max_generations: max_generations.max(1),
            mutation_rate: mutation_rate.clamp(0.0, 1.0),
            max_param_budget,
            rng: Rng::new(seed),
        }
    }

    /// Creates an initial random population.
    pub fn initialize_population(
        &mut self,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Vec<ArchitectureGenome>> {
        // `population_size` is pub with only a constructor floor: 0 empties
        // the population (`search` indexed [0] below) and MAX panics the
        // reserve (or hangs the loop), so re-check both ends here.
        if self.population_size == 0 {
            return Err(crate::error::Error::Msg(
                "EvolutionaryAutoML needs population_size >= 1".into(),
            ));
        }
        let pop_size = crate::limits::check_iterations("population_size", self.population_size)?;
        let mut pop = Vec::with_capacity(pop_size);
        for _ in 0..pop_size {
            let depth = 2 + (self.rng.next_u64() % 4) as usize; // 2 to 5 layers
            let mut dims = Vec::with_capacity(depth);
            for _ in 0..depth {
                let d = 32 * (1 + (self.rng.next_u64() % 8) as usize); // 32 to 256
                dims.push(d);
            }
            let heads = 1 << (self.rng.next_u64() % 4); // 1, 2, 4, 8
            let exp = 1 + (self.rng.next_u64() % 4) as usize; // 1 to 4
            let mut genome = ArchitectureGenome::new(dims, heads, exp);

            // Rescale if exceeds budget
            while genome.estimate_parameters(input_dim, output_dim) > self.max_param_budget
                && genome.hidden_dims.len() > 1
            {
                genome.hidden_dims.pop();
            }
            pop.push(genome);
        }
        Ok(pop)
    }

    /// Evaluates candidate fitness using zero-cost surrogate scores.
    pub fn evaluate_genome(
        &self,
        genome: &ArchitectureGenome,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<ZeroCostScore> {
        // Construct surrogate linear layers
        let mut weights = Vec::new();
        let mut prev = input_dim;
        for &h in &genome.hidden_dims {
            let layer = Linear::new(prev, h)?;
            weights.push(layer.weight);
            prev = h;
        }
        let final_layer = Linear::new(prev, output_dim)?;
        weights.push(final_layer.weight);

        let synflow = compute_synflow(&weights, input_dim)?;
        let params = genome.estimate_parameters(input_dim, output_dim) as f64;
        let budget = self.max_param_budget as f64;

        // Fitness penalizes parameter budget violations and rewards synflow gradient capacity
        let param_ratio = params / budget;
        let penalty = if param_ratio > 1.0 {
            (param_ratio - 1.0) * 10.0
        } else {
            0.0
        };
        let combined = synflow - penalty;

        Ok(ZeroCostScore {
            synflow_score: synflow,
            parameter_efficiency: budget / (params + 1.0),
            combined_fitness: combined,
        })
    }

    /// Runs evolutionary search across generations, returning the best genome and score.
    pub fn search(
        &mut self,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<(ArchitectureGenome, ZeroCostScore)> {
        // `max_generations` is pub with only a constructor floor: each
        // generation builds and scores whole populations, so cap it here.
        crate::limits::check_iterations("max_generations", self.max_generations)?;
        let mut population = self.initialize_population(input_dim, output_dim)?;
        let mut best_genome = population[0].clone();
        let mut best_score = self.evaluate_genome(&best_genome, input_dim, output_dim)?;

        for _ in 0..self.max_generations {
            let mut scored = Vec::with_capacity(population.len());
            for g in &population {
                let s = self.evaluate_genome(g, input_dim, output_dim)?;
                if s.combined_fitness > best_score.combined_fitness {
                    best_score = s.clone();
                    best_genome = g.clone();
                }
                scored.push((g.clone(), s));
            }

            // Sort descending by fitness; NaN fitness sorts last (never elite).
            scored.sort_by(|a, b| {
                let fa = if a.1.combined_fitness.is_nan() {
                    f64::NEG_INFINITY
                } else {
                    a.1.combined_fitness
                };
                let fb = if b.1.combined_fitness.is_nan() {
                    f64::NEG_INFINITY
                } else {
                    b.1.combined_fitness
                };
                fb.partial_cmp(&fa).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Elitism: retain top 50%
            let elite_count = (population.len() / 2).max(1);
            let mut next_gen = Vec::with_capacity(population.len());
            for item in scored.iter().take(elite_count) {
                next_gen.push(item.0.clone());
            }

            // Reproduce with mutation
            while next_gen.len() < population.len() {
                let parent_idx = (self.rng.next_u64() as usize) % elite_count;
                let mut child = scored[parent_idx].0.clone();

                // Mutation
                if (self.rng.next_u64() % 100) < (self.mutation_rate * 100.0) as u64
                    && !child.hidden_dims.is_empty()
                {
                    let dim_idx = (self.rng.next_u64() as usize) % child.hidden_dims.len();
                    let delta = if (self.rng.next_u64() % 2) == 0 {
                        32
                    } else {
                        0
                    };
                    if delta > 0 {
                        child.hidden_dims[dim_idx] += delta;
                    } else if child.hidden_dims[dim_idx] > 32 {
                        child.hidden_dims[dim_idx] -= 32;
                    }
                }

                next_gen.push(child);
            }

            population = next_gen;
        }

        Ok((best_genome, best_score))
    }
}
