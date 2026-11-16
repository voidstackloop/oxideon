"""Hyperparameter Optimization (HPO) and parameter search engine for Oxideon.

Enables automated hyperparameter discovery and tuning:
  - SearchSpace: Configurable distributions (choice, uniform, log_uniform, int_uniform)
  - Trial: Tracks parameter configurations and evaluated objective values
  - RandomSearch: Stochastic sampling across bounded continuous/discrete search spaces
  - GridSearch: Exhaustive Cartesian product evaluation over discrete parameters
  - TPESearch: Tree-structured Parzen Estimator (Bergstra et al., 2011) Bayesian HPO
"""

from typing import Dict, Any, List, Optional, Callable, Union
import itertools
import math
import random
import numpy as np


class SearchSpace:
    """Defines tunable hyperparameter distributions."""

    def __init__(self):
        self.distributions: Dict[str, Dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.distributions)

    def __contains__(self, name: str) -> bool:
        return name in self.distributions

    def choice(self, name: str, choices: List[Any]) -> "SearchSpace":
        if not name or not isinstance(name, str):
            raise ValueError("parameter name must be a non-empty string")
        if len(choices) == 0:
            raise ValueError("choices cannot be empty")
        self.distributions[name] = {"type": "choice", "choices": list(choices)}
        return self

    def uniform(self, name: str, low: float, high: float) -> "SearchSpace":
        if not name or not isinstance(name, str):
            raise ValueError("parameter name must be a non-empty string")
        low_f, high_f = float(low), float(high)
        if low_f > high_f:
            raise ValueError(f"low ({low_f}) must be <= high ({high_f})")
        self.distributions[name] = {"type": "uniform", "low": low_f, "high": high_f}
        return self

    def log_uniform(self, name: str, low: float, high: float) -> "SearchSpace":
        if not name or not isinstance(name, str):
            raise ValueError("parameter name must be a non-empty string")
        low_f, high_f = float(low), float(high)
        if low_f <= 0.0 or high_f <= 0.0:
            raise ValueError("low and high must be positive for log_uniform")
        if low_f > high_f:
            raise ValueError(f"low ({low_f}) must be <= high ({high_f})")
        self.distributions[name] = {
            "type": "log_uniform",
            "low": math.log(low_f),
            "high": math.log(high_f),
        }
        return self

    def int_uniform(self, name: str, low: int, high: int) -> "SearchSpace":
        if not name or not isinstance(name, str):
            raise ValueError("parameter name must be a non-empty string")
        low_i, high_i = int(low), int(high)
        if low_i > high_i:
            raise ValueError(f"low ({low_i}) must be <= high ({high_i})")
        self.distributions[name] = {"type": "int_uniform", "low": low_i, "high": high_i}
        return self

    def sample(self, rng: Optional[random.Random] = None) -> Dict[str, Any]:
        """Draws one random configuration from the search space."""
        r = rng if rng is not None else random
        config = {}
        for name, spec in self.distributions.items():
            t = spec["type"]
            if t == "choice":
                config[name] = r.choice(spec["choices"])
            elif t == "uniform":
                config[name] = r.uniform(spec["low"], spec["high"])
            elif t == "log_uniform":
                config[name] = math.exp(r.uniform(spec["low"], spec["high"]))
            elif t == "int_uniform":
                config[name] = r.randint(spec["low"], spec["high"])
        return config


class Trial:
    """Summary of a single hyperparameter evaluation experiment."""

    def __init__(self, trial_id: int, params: Dict[str, Any]):
        self.trial_id = trial_id
        self.params = params
        self.score: Optional[float] = None
        self.status: str = "PENDING"

    def __repr__(self) -> str:
        return f"Trial(id={self.trial_id}, score={self.score}, params={self.params})"


class RandomSearch:
    """Executes stochastic hyperparameter search across SearchSpace."""

    def __init__(
        self,
        search_space: SearchSpace,
        objective_fn: Callable[[Dict[str, Any]], float],
        n_trials: int = 20,
        maximize: bool = True,
        seed: Optional[int] = None,
    ):
        if n_trials < 1:
            raise ValueError("n_trials must be >= 1")
        self.search_space = search_space
        self.objective_fn = objective_fn
        self.n_trials = n_trials
        self.maximize = maximize
        self._rng = random.Random(seed)

        self.trials: List[Trial] = []
        self.best_trial: Optional[Trial] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None

    def run(self) -> "RandomSearch":
        """Executes search trials sequentially."""
        if len(self.search_space) == 0:
            raise ValueError("search_space cannot be empty")
        for i in range(self.n_trials):
            params = self.search_space.sample(rng=self._rng)
            trial = Trial(trial_id=i, params=params)

            score = float(self.objective_fn(params))
            trial.score = score
            trial.status = "COMPLETED"
            self.trials.append(trial)

            is_better = False
            if self.best_score is None:
                is_better = True
            elif self.maximize and score > self.best_score:
                is_better = True
            elif not self.maximize and score < self.best_score:
                is_better = True

            if is_better:
                self.best_score = score
                self.best_params = params
                self.best_trial = trial

        return self


class GridSearch:
    """Exhaustive search over discrete parameter configurations."""

    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        objective_fn: Callable[[Dict[str, Any]], float],
        maximize: bool = True,
    ):
        if not param_grid:
            raise ValueError("param_grid cannot be empty")
        for k, vals in param_grid.items():
            if len(vals) == 0:
                raise ValueError(f"choices for '{k}' cannot be empty")
        self.param_grid = param_grid
        self.objective_fn = objective_fn
        self.maximize = maximize

        self.trials: List[Trial] = []
        self.best_trial: Optional[Trial] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None

    def run(self) -> "GridSearch":
        keys = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]

        for i, combination in enumerate(itertools.product(*values)):
            params = dict(zip(keys, combination))
            trial = Trial(trial_id=i, params=params)

            score = float(self.objective_fn(params))
            trial.score = score
            trial.status = "COMPLETED"
            self.trials.append(trial)

            is_better = False
            if self.best_score is None:
                is_better = True
            elif self.maximize and score > self.best_score:
                is_better = True
            elif not self.maximize and score < self.best_score:
                is_better = True

            if is_better:
                self.best_score = score
                self.best_params = params
                self.best_trial = trial

        return self


class TPESearch:
    """Tree-structured Parzen Estimator (TPE) Bayesian hyperparameter optimizer (Bergstra et al., 2011)."""

    def __init__(
        self,
        search_space: SearchSpace,
        objective_fn: Callable[[Dict[str, Any]], float],
        n_trials: int = 20,
        maximize: bool = True,
        gamma: float = 0.25,
        n_startup_trials: int = 5,
        n_ei_candidates: int = 24,
        seed: Optional[int] = None,
    ):
        if n_trials < 1:
            raise ValueError("n_trials must be >= 1")
        if not (0.0 < float(gamma) < 1.0):
            raise ValueError("gamma must be in (0, 1)")
        self.search_space = search_space
        self.objective_fn = objective_fn
        self.n_trials = int(n_trials)
        self.maximize = maximize
        self.gamma = float(gamma)
        self.n_startup_trials = max(2, int(n_startup_trials))
        self.n_ei_candidates = max(4, int(n_ei_candidates))
        self._rng = random.Random(seed)

        self.trials: List[Trial] = []
        self.best_trial: Optional[Trial] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None

    def _suggest_tpe(self) -> Dict[str, Any]:
        if len(self.trials) < self.n_startup_trials:
            return self.search_space.sample(rng=self._rng)

        sorted_trials = sorted(
            self.trials,
            key=lambda t: float(t.score) if t.score is not None else -float("inf"),
            reverse=self.maximize,
        )
        n_good = max(1, int(math.ceil(self.gamma * len(sorted_trials))))
        good_trials = sorted_trials[:n_good]
        bad_trials = sorted_trials[n_good:]
        if not bad_trials:
            bad_trials = sorted_trials[-1:]

        best_cand = None
        best_ratio = -float("inf")

        for _ in range(self.n_ei_candidates):
            cand: Dict[str, Any] = {}
            log_l_minus_g = 0.0

            for name, spec in self.search_space.distributions.items():
                t = spec["type"]
                good_vals = [tr.params[name] for tr in good_trials]
                bad_vals = [tr.params[name] for tr in bad_trials]

                if t == "choice":
                    choices = spec["choices"]
                    g_counts = {c: 1.0 for c in choices}
                    b_counts = {c: 1.0 for c in choices}
                    for v in good_vals:
                        g_counts[v] = g_counts.get(v, 1.0) + 1.0
                    for v in bad_vals:
                        b_counts[v] = b_counts.get(v, 1.0) + 1.0
                    g_tot = sum(g_counts.values())
                    b_tot = sum(b_counts.values())
                    # Sample proportional to good distribution
                    r = self._rng.uniform(0.0, g_tot)
                    acc = 0.0
                    chosen = choices[-1]
                    for c in choices:
                        acc += g_counts[c]
                        if r <= acc:
                            chosen = c
                            break
                    cand[name] = chosen
                    log_l_minus_g += math.log(g_counts[chosen] / g_tot) - math.log(b_counts[chosen] / b_tot)
                else:
                    low, high = float(spec["low"]), float(spec["high"])
                    span = max(high - low, 1e-6)
                    sigma = max(span / max(len(good_vals), 2), 1e-4)

                    if t == "log_uniform":
                        g_pts = [math.log(max(float(v), 1e-12)) for v in good_vals]
                        b_pts = [math.log(max(float(v), 1e-12)) for v in bad_vals]
                    else:
                        g_pts = [float(v) for v in good_vals]
                        b_pts = [float(v) for v in bad_vals]

                    mu = self._rng.choice(g_pts)
                    val = min(high, max(low, self._rng.gauss(mu, sigma)))
                    if t == "int_uniform":
                        val_i = int(round(val))
                        cand[name] = val_i
                        eval_x = float(val_i)
                    elif t == "log_uniform":
                        cand[name] = math.exp(val)
                        eval_x = val
                    else:
                        cand[name] = val
                        eval_x = val

                    l_prob = sum(math.exp(-0.5 * ((eval_x - p) / sigma) ** 2) for p in g_pts) / len(g_pts) + 1e-12
                    g_prob = sum(math.exp(-0.5 * ((eval_x - p) / sigma) ** 2) for p in b_pts) / len(b_pts) + 1e-12
                    log_l_minus_g += math.log(l_prob) - math.log(g_prob)

            if log_l_minus_g > best_ratio:
                best_ratio = log_l_minus_g
                best_cand = cand

        return best_cand if best_cand is not None else self.search_space.sample(rng=self._rng)

    def run(self) -> "TPESearch":
        """Executes Bayesian TPE trials sequentially."""
        if len(self.search_space) == 0:
            raise ValueError("search_space cannot be empty")
        for i in range(self.n_trials):
            params = self._suggest_tpe()
            trial = Trial(trial_id=i, params=params)

            score = float(self.objective_fn(params))
            trial.score = score
            trial.status = "COMPLETED"
            self.trials.append(trial)

            is_better = False
            if self.best_score is None:
                is_better = True
            elif self.maximize and score > self.best_score:
                is_better = True
            elif not self.maximize and score < self.best_score:
                is_better = True

            if is_better:
                self.best_score = score
                self.best_params = params
                self.best_trial = trial

        return self


__all__ = [
    "SearchSpace",
    "Trial",
    "RandomSearch",
    "GridSearch",
    "TPESearch",
]
