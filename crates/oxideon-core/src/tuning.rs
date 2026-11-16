//! Automated Hyperparameter Tuning (`ParamValue`, `ParamSpace`, `TrialConfig`, `TrialResult`, `Study`):
//! - **Cartesian Grid Search**: [`Study::grid_search`]
//! - **Log-Uniform / Continuous Random Search**: [`Study::random_search`]

use std::collections::BTreeMap;

use crate::error::{Error, Result};
use crate::rng;

/// Most configurations [`ParamSpace::cartesian_grid`] will materialize. Grids grow
/// multiplicatively, so anything bigger is refused up front instead of OOMing.
pub const MAX_GRID_CONFIGS: usize = 1_000_000;

/// A concrete hyperparameter value sampled for a single trial.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamValue {
    Float(f64),
    Int(i64),
    Bool(bool),
}

impl ParamValue {
    pub fn as_float(&self) -> f64 {
        match *self {
            Self::Float(v) => v,
            Self::Int(v) => v as f64,
            Self::Bool(b) => {
                if b {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    pub fn as_int(&self) -> i64 {
        match *self {
            Self::Int(v) => v,
            Self::Float(v) => v.round() as i64,
            Self::Bool(b) => i64::from(b),
        }
    }

    pub fn as_bool(&self) -> bool {
        match *self {
            Self::Bool(b) => b,
            Self::Int(v) => v != 0,
            Self::Float(v) => v != 0.0,
        }
    }
}

/// Specification of a hyperparameter's search domain.
#[derive(Debug, Clone)]
pub enum ParamDomain {
    /// Discrete choices (used by both [`Study::grid_search`] and [`Study::random_search`]).
    Discrete(Vec<ParamValue>),
    /// Continuous uniform range $[low, high]$.
    Uniform { low: f64, high: f64 },
    /// Log-uniform range $[low, high]$ ($0 < low \le high$), ideal for learning rates and weight decay.
    LogUniform { low: f64, high: f64 },
}

impl ParamDomain {
    fn sample(&self) -> ParamValue {
        match self {
            Self::Discrete(choices) => {
                let idx = rng::with_rng(|r| (r.next_u64() as usize) % choices.len().max(1));
                choices.get(idx).cloned().unwrap_or(ParamValue::Float(0.0))
            }
            &Self::Uniform { low, high } => {
                let u = rng::with_rng(|r| r.next_f32() as f64);
                ParamValue::Float(low + u * (high - low))
            }
            &Self::LogUniform { low, high } => {
                let l_min = low.max(1e-12).ln();
                let l_max = high.max(low.max(1e-12)).ln();
                let u = rng::with_rng(|r| r.next_f32() as f64);
                ParamValue::Float((l_min + u * (l_max - l_min)).exp())
            }
        }
    }

    fn grid_points(&self, default_resolution: usize) -> Vec<ParamValue> {
        let res = default_resolution.max(2);
        match self {
            Self::Discrete(choices) => choices.clone(),
            &Self::Uniform { low, high } => (0..res)
                .map(|i| ParamValue::Float(low + (high - low) * (i as f64) / ((res - 1) as f64)))
                .collect(),
            &Self::LogUniform { low, high } => {
                let l_min = low.max(1e-12).ln();
                let l_max = high.max(low.max(1e-12)).ln();
                (0..res)
                    .map(|i| {
                        ParamValue::Float(
                            (l_min + (l_max - l_min) * (i as f64) / ((res - 1) as f64)).exp(),
                        )
                    })
                    .collect()
            }
        }
    }
}

pub type TrialConfig = BTreeMap<String, ParamValue>;

/// Named hyperparameter search space.
#[derive(Debug, Clone, Default)]
pub struct ParamSpace {
    pub domains: BTreeMap<String, ParamDomain>,
}

impl ParamSpace {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_discrete(mut self, name: impl Into<String>, choices: Vec<ParamValue>) -> Self {
        self.domains
            .insert(name.into(), ParamDomain::Discrete(choices));
        self
    }

    pub fn add_uniform(mut self, name: impl Into<String>, low: f64, high: f64) -> Self {
        self.domains
            .insert(name.into(), ParamDomain::Uniform { low, high });
        self
    }

    pub fn add_log_uniform(mut self, name: impl Into<String>, low: f64, high: f64) -> Self {
        self.domains
            .insert(name.into(), ParamDomain::LogUniform { low, high });
        self
    }

    pub fn sample_config(&self) -> TrialConfig {
        self.domains
            .iter()
            .map(|(k, d)| (k.clone(), d.sample()))
            .collect()
    }

    /// Generates the Cartesian product of all parameter domains.
    ///
    /// Refuses grids larger than [`MAX_GRID_CONFIGS`] instead of allocating
    /// until OOM (use [`Study::random_search`] for big spaces).
    pub fn cartesian_grid(&self, continuous_resolution: usize) -> Result<Vec<TrialConfig>> {
        let mut total = 1usize;
        for domain in self.domains.values() {
            // Count WITHOUT materializing: building the point list first would
            // allocate `continuous_resolution` values before the cap below runs.
            let n = match domain {
                ParamDomain::Discrete(choices) => choices.len().max(1),
                ParamDomain::Uniform { .. } | ParamDomain::LogUniform { .. } => {
                    continuous_resolution.max(2)
                }
            };
            total = total.checked_mul(n).ok_or_else(|| {
                Error::Msg("cartesian_grid: configuration count overflows usize".into())
            })?;
        }
        crate::limits::check_at_most("cartesian_grid configurations", total, MAX_GRID_CONFIGS)?;
        let mut configs = vec![TrialConfig::new()];
        for (name, domain) in &self.domains {
            let pts = domain.grid_points(continuous_resolution);
            let mut next = Vec::with_capacity(configs.len() * pts.len());
            for cfg in &configs {
                for p in &pts {
                    let mut c = cfg.clone();
                    c.insert(name.clone(), p.clone());
                    next.push(c);
                }
            }
            configs = next;
        }
        Ok(configs)
    }
}

#[derive(Debug, Clone)]
pub struct TrialResult {
    pub trial_id: usize,
    pub config: TrialConfig,
    pub objective: f64,
}

/// Hyperparameter optimization study tracking all evaluated trials and the best configuration.
#[derive(Debug, Clone, Default)]
pub struct Study {
    pub trials: Vec<TrialResult>,
    pub best_trial: Option<TrialResult>,
}

impl Study {
    /// Runs an exhaustive Cartesian Grid Search over `space` (with `continuous_resolution` points
    /// per continuous axis), minimizing `objective_fn`.
    pub fn grid_search<F>(
        space: &ParamSpace,
        continuous_resolution: usize,
        mut objective_fn: F,
    ) -> Result<Self>
    where
        F: FnMut(&TrialConfig) -> Result<f64>,
    {
        let configs = space.cartesian_grid(continuous_resolution)?;
        if configs.is_empty() {
            return Err(Error::Msg(
                "Study::grid_search produced 0 configurations".into(),
            ));
        }
        let mut trials = Vec::with_capacity(configs.len());
        let mut best: Option<TrialResult> = None;

        for (trial_id, config) in configs.into_iter().enumerate() {
            let obj = objective_fn(&config)?;
            let tr = TrialResult {
                trial_id,
                config,
                objective: obj,
            };
            // NaN objectives are recorded but can never be "best": NaN compares
            // false against everything, so a NaN best would stick forever.
            if !obj.is_nan() && best.as_ref().is_none_or(|b| obj < b.objective) {
                best = Some(tr.clone());
            }
            trials.push(tr);
        }

        Ok(Self {
            trials,
            best_trial: best,
        })
    }

    /// Runs `num_trials` of Random Search over `space`, minimizing `objective_fn`.
    pub fn random_search<F>(
        space: &ParamSpace,
        num_trials: usize,
        mut objective_fn: F,
    ) -> Result<Self>
    where
        F: FnMut(&TrialConfig) -> Result<f64>,
    {
        let n =
            crate::limits::check_iterations("Study::random_search num_trials", num_trials.max(1))?;
        // Reserve boundedly: `n` can be large-but-legal (up to 1e8).
        let mut trials = Vec::with_capacity(n.min(1024));
        let mut best: Option<TrialResult> = None;

        for trial_id in 0..n {
            let config = space.sample_config();
            let obj = objective_fn(&config)?;
            let tr = TrialResult {
                trial_id,
                config,
                objective: obj,
            };
            // NaN objectives are recorded but can never be "best" (see grid_search).
            if !obj.is_nan() && best.as_ref().is_none_or(|b| obj < b.objective) {
                best = Some(tr.clone());
            }
            trials.push(tr);
        }

        Ok(Self {
            trials,
            best_trial: best,
        })
    }
}
