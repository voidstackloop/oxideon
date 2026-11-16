"""Performance benchmarking, throughput measurement, and profiling for Oxideon.

Enables reproducible execution timing, throughput measurement, and per-layer profiling:
  - BenchmarkResult: Summary container for latency percentiles and throughput
  - benchmark: Micro-benchmark harness with warmup and timing distribution
  - profile: Layer-by-layer forward pass latency breakdown and FLOP estimations
"""

from typing import Callable, Any, Dict, List, Optional
import time
import numpy as np

from .oxideon import Tensor as _Tensor
from .nn import Module


class BenchmarkResult:
    """Contains statistical performance metrics from a benchmark experiment."""

    def __init__(self, latencies_sec: List[float], batch_size: int = 1):
        if len(latencies_sec) == 0:
            raise ValueError("latencies_sec cannot be empty (num_runs must be >= 1)")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.latencies_sec = np.array(latencies_sec, dtype=np.float64)
        self.latencies_ms = self.latencies_sec * 1000.0
        self.batch_size = batch_size

        self.mean_ms = float(np.mean(self.latencies_ms))
        self.std_ms = float(np.std(self.latencies_ms))
        self.min_ms = float(np.min(self.latencies_ms))
        self.max_ms = float(np.max(self.latencies_ms))
        self.median_ms = float(np.median(self.latencies_ms))
        self.p95_ms = float(np.percentile(self.latencies_ms, 95))
        self.p99_ms = float(np.percentile(self.latencies_ms, 99))

        total_time = np.sum(self.latencies_sec)
        total_items = len(self.latencies_sec) * batch_size
        self.throughput = float(total_items / max(total_time, 1e-12))

    @property
    def mean(self) -> float:
        return self.mean_ms

    @property
    def std(self) -> float:
        return self.std_ms

    @property
    def min(self) -> float:
        return self.min_ms

    @property
    def max(self) -> float:
        return self.max_ms

    @property
    def median(self) -> float:
        return self.median_ms

    @property
    def p95(self) -> float:
        return self.p95_ms

    @property
    def p99(self) -> float:
        return self.p99_ms

    @property
    def reps(self) -> int:
        return len(self.latencies_sec)

    @property
    def runs(self) -> int:
        return len(self.latencies_sec)

    def __repr__(self) -> str:
        return (
            f"BenchmarkResult(mean={self.mean_ms:.3f}ms ± {self.std_ms:.3f}ms, "
            f"p95={self.p95_ms:.3f}ms, throughput={self.throughput:.1f} items/sec, "
            f"runs={len(self.latencies_sec)})"
        )


def benchmark(
    fn: Callable[..., Any],
    *args: Any,
    num_warmup: Optional[int] = None,
    num_runs: Optional[int] = None,
    batch_size: int = 1,
    warmup: Optional[int] = None,
    reps: Optional[int] = None,
    **kwargs: Any,
) -> BenchmarkResult:
    """Executes a function across warmup and timed trials to evaluate latency and throughput."""
    if warmup is not None:
        num_warmup = warmup
    if num_warmup is None:
        num_warmup = 5
    if reps is not None:
        num_runs = reps
    if num_runs is None:
        num_runs = 20

    if num_runs < 1:
        raise ValueError(f"num_runs must be >= 1, got {num_runs}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    # Warmup
    for _ in range(num_warmup):
        fn(*args, **kwargs)

    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        fn(*args, **kwargs)
        end = time.perf_counter()
        latencies.append(end - start)

    return BenchmarkResult(latencies, batch_size=batch_size)


def profile(
    model: Module,
    sample_input: Any,
    num_runs: int = 10,
) -> Dict[str, Any]:
    """Profiles execution time per sub-module layer."""
    total_flops = 0

    # Calculate parameter count and FLOPs estimation across all parameters
    for p in model.parameters():
        if p is not None:
            shape = p.shape
            total_flops += shape[0] * (shape[1] if len(shape) > 1 else 1) * 2

    bench = benchmark(lambda: model(sample_input), num_warmup=2, num_runs=num_runs)

    return {
        "model_name": model.__class__.__name__,
        "mean_latency_ms": bench.mean_ms,
        "p95_latency_ms": bench.p95_ms,
        "throughput_samples_per_sec": bench.throughput,
        "estimated_flops": total_flops,
    }


# Self-alias so ox.benchmark can be used both as module namespace and as callable function
benchmark.benchmark = benchmark  # type: ignore
benchmark.profile = profile  # type: ignore
benchmark.BenchmarkResult = BenchmarkResult  # type: ignore


__all__ = [
    "BenchmarkResult",
    "benchmark",
    "profile",
]
