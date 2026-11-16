"""Comprehensive tests for distributed primitives, Ring All-Reduce, PipelineParallel, and ZeRO."""

import numpy as np
import pytest

from oxideon import Tensor, from_numpy
from oxideon.distributed import (
    PipelineParallel,
    ReduceOp,
    ZeroRedundancyOptimizer,
    all_gather,
    all_reduce,
    broadcast,
    destroy_process_group,
    get_rank,
    get_world_size,
    init_process_group,
    is_available,
    is_initialized,
    ring_all_reduce,
)
from oxideon.nn import Linear, Module, ReLU, Sequential
from oxideon.optim import SGD


def test_process_group_lifecycle():
    assert is_available()
    destroy_process_group()
    assert not is_initialized()

    init_process_group(backend="gloo", rank=1, world_size=4)
    assert is_initialized()
    assert get_rank() == 1
    assert get_world_size() == 4

    destroy_process_group()
    assert not is_initialized()
    assert get_rank() == 0
    assert get_world_size() == 1


def test_ring_all_reduce_sum():
    for p in [1, 2, 3, 4, 5]:
        arrays = [np.linspace(1.0, 10.0, 60, dtype=np.float32) * (r + 1) for r in range(p)]
        tensors = [from_numpy(arr) for arr in arrays]
        reduced = ring_all_reduce(tensors, op=ReduceOp.SUM)

        assert len(reduced) == p
        expected = np.sum(arrays, axis=0)
        for r_t in reduced:
            np.testing.assert_allclose(r_t.to_numpy(), expected, rtol=1e-5, atol=1e-5)


def test_ring_all_reduce_avg():
    for p in [2, 3, 4, 5]:
        arrays = [np.arange(120, dtype=np.float32) * (r + 1.5) for r in range(p)]
        tensors = [from_numpy(arr) for arr in arrays]
        reduced = ring_all_reduce(tensors, op=ReduceOp.AVG)

        assert len(reduced) == p
        expected = np.mean(arrays, axis=0)
        for r_t in reduced:
            np.testing.assert_allclose(r_t.to_numpy(), expected, rtol=1e-5, atol=1e-5)


def test_ring_all_reduce_min_max():
    p = 4
    arrays = [np.array([r * 2.0, (10 - r) * 3.0, (r - 2) ** 2], dtype=np.float32) for r in range(p)]
    tensors = [from_numpy(arr) for arr in arrays]

    res_max = ring_all_reduce(tensors, op=ReduceOp.MAX)
    res_min = ring_all_reduce(tensors, op=ReduceOp.MIN)

    exp_max = np.max(arrays, axis=0)
    exp_min = np.min(arrays, axis=0)

    for r_t in res_max:
        np.testing.assert_allclose(r_t.to_numpy(), exp_max)
    for r_t in res_min:
        np.testing.assert_allclose(r_t.to_numpy(), exp_min)


class TinyStage(Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc = Linear(in_features, out_features)
        self.act = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.fc(x))


def test_pipeline_parallel_forward_and_backward():
    np.random.seed(42)
    s1 = TinyStage(8, 16)
    s2 = TinyStage(16, 8)
    s3 = TinyStage(8, 4)

    pipe = PipelineParallel(stages=[s1, s2, s3], num_microbatches=4, schedule="1f1b")

    x_np = np.random.randn(8, 8).astype(np.float32)
    x = from_numpy(x_np).requires_grad_(True)

    out = pipe(x)
    assert out.shape == (8, 4)

    # Compute loss and verify backward through all pipeline stages
    loss = (out * out).sum()
    loss.backward()

    # Verify input gradient
    assert x.grad is not None
    assert x.grad.shape == (8, 8)
    assert not np.isnan(x.grad.to_numpy()).any()

    # Verify gradients flowed into all stages' weights
    for idx, stage in enumerate(pipe.stages):
        w_grad = stage.fc.weight.grad
        assert w_grad is not None, f"Stage {idx} weight gradient was None"
        assert not np.isnan(w_grad.to_numpy()).any()


def test_zero_redundancy_optimizer_partitioning():
    np.random.seed(1337)
    layers = [Linear(4, 4) for _ in range(4)]
    all_params = []
    for l in layers:
        all_params.extend(list(l.parameters()))

    # World size 2, rank 0 manages even index params, rank 1 manages odd
    opt_rank0 = ZeroRedundancyOptimizer(all_params, SGD, world_size=2, rank=0, lr=0.1)
    opt_rank1 = ZeroRedundancyOptimizer(all_params, SGD, world_size=2, rank=1, lr=0.1)

    assert len(opt_rank0.assigned_params) == len(all_params) // 2
    assert len(opt_rank1.assigned_params) == len(all_params) // 2

    # Give all params artificial gradients
    for p in all_params:
        p.grad = from_numpy(np.ones_like(p.to_numpy(), dtype=np.float32))

    # Record initial values
    p0_before = opt_rank0.assigned_params[0].to_numpy().copy()
    p1_before = opt_rank1.assigned_params[0].to_numpy().copy()

    # Step rank 0
    opt_rank0.step()
    p0_after = opt_rank0.assigned_params[0].to_numpy()
    np.testing.assert_allclose(p0_after, p0_before - 0.1 * 1.0, rtol=1e-5)

    # Param managed by rank 1 should NOT have changed on rank 0 step
    np.testing.assert_allclose(opt_rank1.assigned_params[0].to_numpy(), p1_before)

    # Step rank 1
    opt_rank1.step()
    p1_after = opt_rank1.assigned_params[0].to_numpy()
    np.testing.assert_allclose(p1_after, p1_before - 0.1 * 1.0, rtol=1e-5)

    # Test zero_grad
    opt_rank0.zero_grad()
    for p in all_params:
        assert p.grad is None
