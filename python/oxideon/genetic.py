"""Neuroevolution and Genetic Algorithms (GA / ES) for Oxideon.

Enables gradient-free optimization and direct weight evolution for non-differentiable objectives:
  - GeneticAlgorithm: Vectorized genetic algorithm with tournament selection, crossover, and mutation
  - NeuroEvolution: Natural Evolution Strategies (Salimans et al., 2017) directly evolving model weights
"""

from typing import Callable, List, Tuple, Optional, Any, Dict
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module, Parameter


def _collect_param_slots(model: Any) -> List[Tuple[Any, str, _Tensor]]:
    """Collects (owner_obj, attr_name, tensor) for both Python Modules and native PyO3 layers."""
    slots: List[Tuple[Any, str, _Tensor]] = []
    visited = set()

    def _walk(obj: Any):
        if id(obj) in visited:
            return
        visited.add(id(obj))
        params_dict = getattr(obj, "_parameters", None)
        if isinstance(params_dict, dict):
            for k, v in params_dict.items():
                if isinstance(v, _Tensor):
                    slots.append((obj, k, v))
        else:
            for attr in ("weight", "bias", "gamma", "beta"):
                v = getattr(obj, attr, None)
                if isinstance(v, _Tensor):
                    slots.append((obj, attr, v))

        mods_dict = getattr(obj, "_modules", None)
        if isinstance(mods_dict, dict):
            for sub in mods_dict.values():
                if sub is not None:
                    _walk(sub)

    _walk(model)
    return slots


class GeneticAlgorithm:
    """Vectorized Genetic Algorithm for continuous real-valued optimization."""

    def __init__(
        self,
        population_size: int = 50,
        genome_len: int = 10,
        mutation_rate: float = 0.05,
        mutation_scale: float = 0.1,
        crossover_rate: float = 0.8,
        bounds: Tuple[float, float] = (-5.0, 5.0),
        seed: Optional[int] = None,
    ):
        if population_size < 3:
            raise ValueError("population_size must be >= 3")
        if genome_len < 1:
            raise ValueError(f"genome_len must be >= 1, got {genome_len}")
        if not (0.0 <= float(mutation_rate) <= 1.0):
            raise ValueError(f"mutation_rate must be in [0, 1], got {mutation_rate}")
        if float(mutation_scale) < 0.0:
            raise ValueError(f"mutation_scale must be >= 0, got {mutation_scale}")
        if not (0.0 <= float(crossover_rate) <= 1.0):
            raise ValueError(f"crossover_rate must be in [0, 1], got {crossover_rate}")
        if len(bounds) != 2 or float(bounds[0]) >= float(bounds[1]):
            raise ValueError(f"bounds must be (low, high) with low < high, got {bounds}")

        self.population_size = population_size
        self.genome_len = genome_len
        self.mutation_rate = float(mutation_rate)
        self.mutation_scale = float(mutation_scale)
        self.crossover_rate = float(crossover_rate)
        self.bounds = (float(bounds[0]), float(bounds[1]))
        self._rng = np.random.RandomState(seed)

        self.population = self._rng.uniform(
            self.bounds[0], self.bounds[1], (population_size, genome_len)
        ).astype(np.float32)

        self.best_individual: Optional[np.ndarray] = None
        self.best_fitness: float = -float("inf")

    def step(self, fitness_scores: np.ndarray) -> np.ndarray:
        """Runs a single generation selection, crossover, and mutation step."""
        fitness = np.asarray(fitness_scores, dtype=np.float32)
        if len(fitness) != self.population_size:
            raise ValueError(f"fitness_scores length {len(fitness)} must match population_size {self.population_size}")
        best_idx = np.argmax(fitness)
        if fitness[best_idx] > self.best_fitness:
            self.best_fitness = float(fitness[best_idx])
            self.best_individual = self.population[best_idx].copy()

        new_pop = []
        new_pop.append(self.population[best_idx].copy())

        while len(new_pop) < self.population_size:
            t1 = self._rng.choice(self.population_size, 3, replace=False)
            p1 = self.population[t1[np.argmax(fitness[t1])]]
            t2 = self._rng.choice(self.population_size, 3, replace=False)
            p2 = self.population[t2[np.argmax(fitness[t2])]]

            if self._rng.rand() < self.crossover_rate:
                mask = self._rng.rand(self.genome_len) < 0.5
                child = np.where(mask, p1, p2)
            else:
                child = p1.copy()

            mut_mask = self._rng.rand(self.genome_len) < self.mutation_rate
            child += mut_mask * self._rng.normal(0.0, self.mutation_scale, size=self.genome_len)
            child = np.clip(child, self.bounds[0], self.bounds[1])

            new_pop.append(child)

        self.population = np.array(new_pop, dtype=np.float32)
        return self.population

    def evolve(self, fitness_fn: Callable[[np.ndarray], float], generations: int = 20) -> np.ndarray:
        if generations < 1:
            raise ValueError(f"generations must be >= 1, got {generations}")
        for _ in range(generations):
            scores = np.array([fitness_fn(ind) for ind in self.population])
            self.step(scores)
        return self.best_individual


class NeuroEvolution:
    """Evolution Strategies (ES) for evolving neural network Module weights."""

    def __init__(
        self,
        model: Any,
        population_size: int = 20,
        sigma: float = 0.05,
        lr: float = 0.01,
        seed: Optional[int] = None,
    ):
        if population_size < 2:
            raise ValueError("population_size must be >= 2")
        if float(sigma) <= 0.0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if float(lr) <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        self.model = model
        self.population_size = population_size
        self.sigma = float(sigma)
        self.lr = float(lr)
        self._rng = np.random.RandomState(seed)

    def step(self, fitness_fn: Callable[[Any], float]) -> float:
        """Perturbs weights in opposite directions (antithetic sampling) and steps towards gradient estimate."""
        slots = _collect_param_slots(self.model)
        base_weights = [p.to_numpy().copy() for _, _, p in slots]

        half_pop = self.population_size // 2
        noises = []
        for _ in range(half_pop):
            noise = [self._rng.randn(*w.shape).astype(np.float32) for w in base_weights]
            noises.append(noise)

        scores_pos = []
        scores_neg = []

        for noise in noises:
            # Positive
            self._set_weights(slots, [w + self.sigma * eps for w, eps in zip(base_weights, noise)])
            scores_pos.append(fitness_fn(self.model))
            # Negative
            self._set_weights(slots, [w - self.sigma * eps for w, eps in zip(base_weights, noise)])
            scores_neg.append(fitness_fn(self.model))

        all_scores = np.array(scores_pos + scores_neg, dtype=np.float32)
        mean_score = np.mean(all_scores)
        std_score = np.std(all_scores) + 1e-8
        norm_pos = (np.array(scores_pos) - mean_score) / std_score
        norm_neg = (np.array(scores_neg) - mean_score) / std_score

        new_weights = [w.copy() for w in base_weights]
        for i, noise in enumerate(noises):
            weight_diff = norm_pos[i] - norm_neg[i]
            for j in range(len(new_weights)):
                new_weights[j] += (self.lr / (half_pop * self.sigma)) * weight_diff * noise[j]

        self._set_weights(slots, new_weights)
        return float(np.max(all_scores))

    def _set_weights(self, slots: List[Tuple[Any, str, _Tensor]], weights: List[np.ndarray]):
        for (owner, attr, _), w in zip(slots, weights):
            t = _from_numpy(np.ascontiguousarray(w, dtype=np.float32))
            if isinstance(owner, Module):
                setattr(owner, attr, Parameter(t))
            else:
                t.requires_grad_(True)
                setattr(owner, attr, t)


__all__ = [
    "GeneticAlgorithm",
    "NeuroEvolution",
]
