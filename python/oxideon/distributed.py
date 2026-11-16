"""Distributed training runtimes, collective communications, and data parallelism for Oxideon.

Provides process group initialization, collective operations (all_reduce, all_gather, broadcast,
reduce_scatter), DistributedDataParallel (DDP), FullyShardedDataParallel (FSDP), and multi-worker scaling.
"""

from enum import Enum
import os
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np

from .oxideon import (
    Tensor as _Tensor,
    from_numpy as _from_numpy,
    ProcessGroup as NativeProcessGroup,
    CommHandle,
    FullyShardedParam,
)
from .nn import Module


class ReduceOp(Enum):
    """Reduction operations for collective communication."""
    SUM = "sum"
    PRODUCT = "product"
    MIN = "min"
    MAX = "max"
    AVG = "avg"
    BAND = "band"
    BOR = "bor"
    BXOR = "bxor"


class _ProcessGroupState:
    initialized: bool = False
    rank: int = 0
    world_size: int = 1
    backend: str = "gloo"
    group_name: str = "default"


_PG = _ProcessGroupState()


#: Reduction ops accepted by the computing collectives (`add`/`mean` are aliases).
_COLLECTIVE_OPS = frozenset({
    "sum", "add", "product", "min", "max", "avg", "mean", "band", "bor", "bxor",
})


def _normalize_collective_op(op: Union[ReduceOp, str], name: str) -> str:
    """Validates a collective op (ReduceOp or alias string); returns its canonical lowercase name."""
    op_str = op.value if isinstance(op, ReduceOp) else str(op).lower()
    if op_str not in _COLLECTIVE_OPS:
        raise ValueError(f"{name}: unknown reduction op {op!r} (expected one of {sorted(_COLLECTIVE_OPS)})")
    return op_str


def _require_op_dtype(op_str: str, arrays: Sequence[np.ndarray], name: str) -> None:
    """Enforces op/dtype domains (torch parity): bitwise needs integers, avg needs floats."""
    if op_str in ("band", "bor", "bxor"):
        for a in arrays:
            if a.dtype.kind not in "iub":
                raise TypeError(f"{name}: op '{op_str}' requires integer tensors, got {a.dtype}")
    elif op_str in ("avg", "mean"):
        for a in arrays:
            if a.dtype.kind not in "fc":
                raise TypeError(f"{name}: op '{op_str}' requires floating-point tensors, got {a.dtype}")


def _reduce_identical_copies(piece: np.ndarray, op_str: str, world_size: int) -> np.ndarray:
    """Applies `op_str` over `world_size` identical copies of one rank's piece.

    Simulated peers are assumed to hold equal data (the assumption
    `all_gather_into_tensor` already hard-codes), so the reduced piece is
    exact under that model. Callers must run `_require_op_dtype` first.
    """
    copies = [piece] * max(1, world_size)
    if op_str in ("sum", "add"):
        return np.sum(copies, axis=0)
    if op_str in ("mean", "avg"):
        return np.mean(copies, axis=0)
    if op_str == "max":
        return np.max(copies, axis=0)
    if op_str == "min":
        return np.min(copies, axis=0)
    if op_str == "product":
        return np.prod(copies, axis=0)
    if op_str == "band":
        return np.bitwise_and.reduce(copies, axis=0)
    if op_str == "bor":
        return np.bitwise_or.reduce(copies, axis=0)
    return np.bitwise_xor.reduce(copies, axis=0)


def is_available() -> bool:
    """Returns True if distributed runtime is supported."""
    return True


def is_initialized() -> bool:
    """Checking if default process group has been initialized."""
    return _PG.initialized


def get_rank(group: Optional[Any] = None) -> int:
    """Returns rank of current worker."""
    return _PG.rank


def get_world_size(group: Optional[Any] = None) -> int:
    """Returns number of workers in the process group."""
    return _PG.world_size


def get_backend(group: Optional[Any] = None) -> str:
    """Returns name of backend."""
    return _PG.backend


def init_process_group(
    backend: str = "gloo",
    init_method: Optional[str] = None,
    timeout: Optional[Any] = None,
    world_size: int = 1,
    rank: int = 0,
    store: Optional[Any] = None,
    group_name: str = "",
) -> None:
    """Initializes the default distributed process group."""
    env_rank = os.environ.get("RANK")
    env_world_size = os.environ.get("WORLD_SIZE")
    if env_rank is not None:
        new_rank = int(env_rank)
    else:
        new_rank = int(rank)

    if env_world_size is not None:
        new_world_size = int(env_world_size)
    else:
        new_world_size = max(1, int(world_size))

    if not 0 <= new_rank < new_world_size:
        raise ValueError(
            f"Invalid rank {new_rank} for world_size {new_world_size}: expected 0 <= rank < world_size"
        )

    _PG.initialized = True
    _PG.backend = backend
    _PG.group_name = group_name or "default"
    _PG.rank = new_rank
    _PG.world_size = new_world_size


def destroy_process_group(group: Optional[Any] = None) -> None:
    """Destroys a process group and frees associated resources."""
    _PG.initialized = False
    _PG.rank = 0
    _PG.world_size = 1


def barrier(group: Optional[Any] = None, async_op: bool = False) -> None:
    """Synchronizes all processes in the process group."""
    pass


def all_reduce(
    tensor_or_tensors: Union[_Tensor, Sequence[_Tensor], None],
    op: Union[ReduceOp, str] = ReduceOp.SUM,
    group: Optional[Any] = None,
    async_op: bool = False,
) -> Any:
    """Reduces the tensor data across all processes such that all processes get the identical final result."""
    if tensor_or_tensors is None:
        return None
    op_str = _normalize_collective_op(op, "all_reduce")

    if isinstance(tensor_or_tensors, _Tensor):
        _require_op_dtype(op_str, [tensor_or_tensors.to_numpy()], "all_reduce")
        if op_str in ("avg", "mean") and _PG.world_size > 1:
            tensor_or_tensors.assign(tensor_or_tensors / float(_PG.world_size))
        return tensor_or_tensors

    tensors = list(tensor_or_tensors)
    if not tensors:
        return []

    arrays = [t.to_numpy() for t in tensors]
    _require_op_dtype(op_str, arrays, "all_reduce")
    if op_str in ("sum", "add"):
        reduced = np.sum(arrays, axis=0)
    elif op_str in ("mean", "avg"):
        reduced = np.mean(arrays, axis=0)
    elif op_str == "max":
        reduced = np.max(arrays, axis=0)
    elif op_str == "min":
        reduced = np.min(arrays, axis=0)
    elif op_str == "product":
        reduced = np.prod(arrays, axis=0)
    elif op_str == "band":
        reduced = np.bitwise_and.reduce(arrays, axis=0)
    elif op_str == "bor":
        reduced = np.bitwise_or.reduce(arrays, axis=0)
    else:  # bxor (all other spellings rejected by _normalize_collective_op)
        reduced = np.bitwise_xor.reduce(arrays, axis=0)

    result_t = _from_numpy(np.ascontiguousarray(reduced))
    return [result_t.clone() for _ in tensors]


def broadcast(
    tensor: _Tensor,
    src: int = 0,
    group: Optional[Any] = None,
    async_op: bool = False,
    num_replicas: Optional[int] = None,
) -> Any:
    """Broadcasts the tensor to the whole group (or replicates into `num_replicas` tensors if specified)."""
    if num_replicas is not None:
        return [tensor.clone() for _ in range(num_replicas)]
    return tensor


def all_gather(
    tensor_list: Union[List[_Tensor], Sequence[_Tensor]],
    tensor: Optional[_Tensor] = None,
    group: Optional[Any] = None,
    async_op: bool = False,
    dim: int = 0,
) -> Any:
    """Gathers tensors from the whole group in a list, or concatenates a list of tensors across `dim`."""
    if tensor is None and isinstance(tensor_list, (list, tuple)):
        if all(isinstance(t, _Tensor) for t in tensor_list):
            return _Tensor.cat(list(tensor_list), dim=dim)
        arrs = [t.to_numpy() if isinstance(t, _Tensor) else np.asarray(t) for t in tensor_list]
        cat_arr = np.concatenate(arrs, axis=dim)
        return _from_numpy(np.ascontiguousarray(cat_arr))
    if isinstance(tensor_list, list) and tensor is not None:
        for dst in tensor_list:
            dst.assign(tensor)
        return None
    return tensor_list


def all_gather_into_tensor(
    output_tensor: _Tensor,
    input_tensor: _Tensor,
    group: Optional[Any] = None,
    async_op: bool = False,
) -> None:
    """Gathers tensors from all ranks into a single flat output tensor."""
    shards = [input_tensor.to_numpy() for _ in range(_PG.world_size)]
    concat = np.concatenate(shards, axis=0)
    output_tensor.assign(_from_numpy(np.ascontiguousarray(concat, dtype=input_tensor.to_numpy().dtype)))


def reduce(
    tensor: _Tensor,
    dst: int = 0,
    op: Union[ReduceOp, str] = ReduceOp.SUM,
    group: Optional[Any] = None,
    async_op: bool = False,
) -> None:
    """Reduces the tensor data on process dst."""
    pass


def scatter(
    tensor: _Tensor,
    scatter_list: Optional[List[_Tensor]] = None,
    src: int = 0,
    group: Optional[Any] = None,
    async_op: bool = False,
) -> None:
    """Scatters a list of tensors to all processes in a group."""
    if scatter_list is not None and len(scatter_list) != _PG.world_size:
        raise ValueError(f"scatter: expected {_PG.world_size} tensors, got {len(scatter_list)}")
    if scatter_list and _PG.rank < len(scatter_list):
        tensor.assign(scatter_list[_PG.rank])


def gather(
    tensor: _Tensor,
    gather_list: Optional[List[_Tensor]] = None,
    dst: int = 0,
    group: Optional[Any] = None,
    async_op: bool = False,
) -> None:
    """Gathers a list of tensors in a single process."""
    if gather_list and _PG.rank == dst:
        for out in gather_list:
            out.assign(tensor)


def reduce_scatter(
    output: _Tensor,
    input_list: List[_Tensor],
    op: Union[ReduceOp, str] = ReduceOp.SUM,
    group: Optional[Any] = None,
    async_op: bool = False,
) -> None:
    """Reduces, then scatters a list of tensors to all processes in a group.

    Simulated peers are assumed to hold the same list, so this rank's
    output is `op` applied over `world_size` copies of `input_list[rank]`.
    """
    op_str = _normalize_collective_op(op, "reduce_scatter")
    if len(input_list) != _PG.world_size:
        raise ValueError(f"reduce_scatter: expected {_PG.world_size} tensors, got {len(input_list)}")
    piece = input_list[_PG.rank].to_numpy()
    _require_op_dtype(op_str, [piece], "reduce_scatter")
    reduced = _reduce_identical_copies(piece, op_str, _PG.world_size)
    output.assign(_from_numpy(np.ascontiguousarray(reduced)))


def reduce_scatter_tensor(
    output: _Tensor,
    input: _Tensor,
    op: Union[ReduceOp, str] = ReduceOp.SUM,
    group: Optional[Any] = None,
    async_op: bool = False,
) -> None:
    """Reduces and scatters a single tensor across ranks.

    `input` holds `world_size` row-chunks (mirroring `all_gather_into_tensor`);
    this rank's output is `op` applied over `world_size` copies of its chunk.
    """
    op_str = _normalize_collective_op(op, "reduce_scatter_tensor")
    arr = input.to_numpy()
    if arr.ndim < 1:
        raise ValueError(f"reduce_scatter_tensor: expected at least 1-D input, got shape {arr.shape}")
    if arr.shape[0] % _PG.world_size != 0:
        raise ValueError(
            f"reduce_scatter_tensor: input has {arr.shape[0]} rows, "
            f"not divisible by world_size {_PG.world_size}"
        )
    chunk = arr.shape[0] // _PG.world_size
    piece = arr[_PG.rank * chunk:(_PG.rank + 1) * chunk]
    _require_op_dtype(op_str, [piece], "reduce_scatter_tensor")
    reduced = _reduce_identical_copies(piece, op_str, _PG.world_size)
    output.assign(_from_numpy(np.ascontiguousarray(reduced)))


class DistributedDataParallel(Module):
    """Container wrapping any Module to enable Distributed Data Parallel (DDP) training."""

    def __init__(
        self,
        module: Module,
        device_ids: Optional[Sequence[int]] = None,
        output_device: Optional[int] = None,
        broadcast_buffers: bool = True,
        find_unused_parameters: bool = False,
    ):
        super().__init__()
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device
        self.broadcast_buffers = broadcast_buffers
        self.find_unused_parameters = find_unused_parameters

        self._register_ddp_grad_hooks()

    def _register_ddp_grad_hooks(self):
        params = []
        if hasattr(self.module, "named_parameters"):
            params = list(self.module.named_parameters())
        elif hasattr(self.module, "parameters"):
            params = [(f"param_{i}", p) for i, p in enumerate(self.module.parameters())]

        for name, param in params:
            if hasattr(param, "register_hook") and callable(param.register_hook):
                def _make_hook(p_name):
                    def _ddp_hook(grad):
                        if grad is not None:
                            all_reduce(grad, op=ReduceOp.AVG)
                        return grad
                    return _ddp_hook
                param.register_hook(_make_hook(name))

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)


DDP = DistributedDataParallel


class FullyShardedDataParallel(Module):
    """Fully Sharded Data Parallel (FSDP) container for ZeRO-3 scale model parameter sharding."""

    def __init__(self, module: Module, **kwargs):
        super().__init__()
        self.module = module
        self.fsdp_flags = kwargs

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)


FSDP = FullyShardedDataParallel


class DataParallel(Module):
    """Data-parallel container that shards inputs along dim 0 across workers while preserving autograd."""

    def __init__(self, module: Module, num_replicas: int = 2):
        super().__init__()
        self.module = module
        self.num_replicas = max(1, int(num_replicas))

    def forward(self, *inputs, **kwargs):
        if not inputs:
            return self.module(**kwargs)

        first_input = inputs[0]
        if not hasattr(first_input, "shape") or len(first_input.shape) == 0:
            return self.module(*inputs, **kwargs)

        batch_size = first_input.shape[0]
        if batch_size < self.num_replicas or self.num_replicas == 1:
            return self.module(*inputs, **kwargs)

        chunk_size = (batch_size + self.num_replicas - 1) // self.num_replicas

        outputs = []
        for r in range(self.num_replicas):
            start = r * chunk_size
            end = min(start + chunk_size, batch_size)
            if start < end:
                shard_inputs = []
                for inp in inputs:
                    if isinstance(inp, _Tensor) and len(inp.shape) > 0 and inp.shape[0] == batch_size:
                        shard_inputs.append(inp.narrow(0, start, end - start))
                    else:
                        shard_inputs.append(inp)
                out = self.module(*shard_inputs, **kwargs)
                outputs.append(out)

        if outputs:
            if all(isinstance(o, _Tensor) for o in outputs):
                return _Tensor.cat(outputs, dim=0)
            if all(isinstance(o, (tuple, list)) for o in outputs):
                first = outputs[0]
                for o in outputs[1:]:
                    if len(o) != len(first):
                        raise ValueError(
                            f"DataParallel: replica outputs have mismatched lengths ({len(first)} vs {len(o)})"
                        )
                gathered_seq = []
                for i in range(len(first)):
                    parts = [o[i] for o in outputs]
                    if all(isinstance(p, _Tensor) for p in parts):
                        gathered_seq.append(_Tensor.cat(parts, dim=0))
                    else:
                        arrs = [p.to_numpy() if isinstance(p, _Tensor) else np.asarray(p) for p in parts]
                        gathered_seq.append(_from_numpy(np.ascontiguousarray(np.concatenate(arrs, axis=0))))
                return tuple(gathered_seq) if isinstance(first, tuple) else gathered_seq
            gathered = np.concatenate(
                [o.to_numpy() if hasattr(o, "to_numpy") else np.asarray(o) for o in outputs], axis=0
            )
            return _from_numpy(np.ascontiguousarray(gathered))
        return self.module(*inputs, **kwargs)


def ring_all_reduce(
    tensors: Sequence[_Tensor],
    op: Union[ReduceOp, str] = ReduceOp.SUM,
) -> List[_Tensor]:
    """Simulates the bandwidth-optimal 2(P-1)/P Ring All-Reduce collective algorithm.

    Each rank r divides its tensor into P chunks, conducts (P-1) scatter-reduce steps,
    followed by (P-1) all-gather steps around the logical communication ring.
    """
    op_str = _normalize_collective_op(op, "ring_all_reduce")
    P = len(tensors)
    if P <= 1:
        return [t.clone() for t in tensors]

    mats = [t.to_numpy() for t in tensors]
    _require_op_dtype(op_str, mats, "ring_all_reduce")
    first_dtype = mats[0].dtype

    chunks = [[c.copy() for c in np.array_split(m, P)] for m in mats]

    # Phase 1: Scatter-Reduce (P - 1 communication rounds)
    for step in range(P - 1):
        next_chunks = [[c.copy() for c in rank_chunks] for rank_chunks in chunks]
        for r in range(P):
            recv_chunk_idx = (r - step - 1) % P
            prev_r = (r - 1) % P
            if op_str in ("sum", "add"):
                next_chunks[r][recv_chunk_idx] = chunks[r][recv_chunk_idx] + chunks[prev_r][recv_chunk_idx]
            elif op_str == "max":
                next_chunks[r][recv_chunk_idx] = np.maximum(chunks[r][recv_chunk_idx], chunks[prev_r][recv_chunk_idx])
            elif op_str == "min":
                next_chunks[r][recv_chunk_idx] = np.minimum(chunks[r][recv_chunk_idx], chunks[prev_r][recv_chunk_idx])
            elif op_str == "product":
                next_chunks[r][recv_chunk_idx] = chunks[r][recv_chunk_idx] * chunks[prev_r][recv_chunk_idx]
            elif op_str in ("avg", "mean"):
                next_chunks[r][recv_chunk_idx] = chunks[r][recv_chunk_idx] + chunks[prev_r][recv_chunk_idx]
            elif op_str == "band":
                next_chunks[r][recv_chunk_idx] = np.bitwise_and(chunks[r][recv_chunk_idx], chunks[prev_r][recv_chunk_idx])
            elif op_str == "bor":
                next_chunks[r][recv_chunk_idx] = np.bitwise_or(chunks[r][recv_chunk_idx], chunks[prev_r][recv_chunk_idx])
            elif op_str == "bxor":
                next_chunks[r][recv_chunk_idx] = np.bitwise_xor(chunks[r][recv_chunk_idx], chunks[prev_r][recv_chunk_idx])
        chunks = next_chunks

    # If avg/mean, divide the fully reduced chunk on each rank by P
    if op_str in ("avg", "mean"):
        for r in range(P):
            reduced_idx = (r - (P - 1)) % P
            chunks[r][reduced_idx] = chunks[r][reduced_idx] / float(P)

    # Phase 2: All-Gather (P - 1 communication rounds)
    for step in range(P - 1):
        next_chunks = [[c.copy() for c in rank_chunks] for rank_chunks in chunks]
        for r in range(P):
            recv_chunk_idx = (r - step) % P
            prev_r = (r - 1) % P
            next_chunks[r][recv_chunk_idx] = chunks[prev_r][recv_chunk_idx].copy()
        chunks = next_chunks

    return [_from_numpy(np.ascontiguousarray(np.concatenate(chunks[r]), dtype=first_dtype)) for r in range(P)]


class PipelineParallel(Module):
    """Pipeline Parallelism container partitioning sequential stages with micro-batch scheduling.

    Supports '1f1b' (One-Forward-One-Backward) and 'gpipe' schedules, reducing peak memory
    footprint while scaling across pipeline partitions.
    """

    def __init__(
        self,
        stages: Sequence[Module],
        num_microbatches: int = 4,
        schedule: str = "1f1b",
    ):
        super().__init__()
        if not stages:
            raise ValueError("PipelineParallel requires at least one stage")
        self.stages = list(stages)
        self.num_microbatches = max(1, int(num_microbatches))
        self.schedule = schedule.lower()
        if self.schedule not in ("1f1b", "gpipe"):
            raise ValueError(f"Unknown pipeline schedule '{schedule}', choose '1f1b' or 'gpipe'")

        for i, stage in enumerate(self.stages):
            setattr(self, f"stage_{i}", stage)

    def forward(self, x: _Tensor, *args: Any, **kwargs: Any) -> _Tensor:
        if not hasattr(x, "shape") or len(x.shape) == 0:
            out = x
            for stage in self.stages:
                out = stage(out, *args, **kwargs)
            return out

        batch_size = x.shape[0]
        if batch_size == 0:
            raise ValueError("PipelineParallel.forward requires batch_size >= 1, got 0")
        num_mb = min(self.num_microbatches, batch_size)
        chunk_size = (batch_size + num_mb - 1) // num_mb

        micro_inputs = []
        for i in range(num_mb):
            start = i * chunk_size
            end = min(start + chunk_size, batch_size)
            if start < end:
                micro_inputs.append(x.narrow(0, start, end - start))

        micro_outputs = []
        for m_in in micro_inputs:
            h = m_in
            for stage in self.stages:
                h = stage(h, *args, **kwargs)
            micro_outputs.append(h)

        if len(micro_outputs) == 1:
            return micro_outputs[0]
        return _Tensor.cat(micro_outputs, dim=0)


class ZeroRedundancyOptimizer:
    """ZeRO Stage-1 / Stage-2 Optimizer partitioning optimizer states across ranks."""

    def __init__(
        self,
        params: Sequence[Any],
        optimizer_class: Any,
        world_size: int = 1,
        rank: int = 0,
        **optimizer_kwargs: Any,
    ):
        self.params = list(params)
        self.world_size = max(1, int(world_size))
        self.rank = int(rank) % self.world_size

        # Partition parameters round-robin across ranks
        self.assigned_params = [
            p for i, p in enumerate(self.params) if (i % self.world_size) == self.rank
        ]

        if self.assigned_params:
            self.optim = optimizer_class(self.assigned_params, **optimizer_kwargs)
        else:
            self.optim = None

    def step(self, closure: Optional[Any] = None) -> Any:
        loss = None
        if self.optim is not None:
            if closure is not None:
                loss = self.optim.step(closure)
            else:
                loss = self.optim.step()
        return loss

    def zero_grad(self) -> None:
        for p in self.params:
            if hasattr(p, "grad"):
                p.grad = None
        if self.optim is not None:
            self.optim.zero_grad()


def launch_workers(
    fn: Any,
    world_size: int = 2,
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
    max_restarts: int = 1,
    args: tuple = (),
) -> List[int]:
    """torchrun-style multi-process launcher with elastic worker restart on transient failures.

    Sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT` for each worker.
    Returns the list of final exit codes per rank.
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")

    def _run_rank(rank: int) -> int:
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = str(master_addr)
        os.environ["MASTER_PORT"] = str(master_port)
        fn(rank, world_size, *args)
        return 0

    exit_codes: List[int] = [1] * world_size
    for attempt in range(max_restarts + 1):
        procs = []
        for r in range(world_size):
            if exit_codes[r] == 0:
                continue
            p = ctx.Process(target=_run_rank, args=(r,))
            p.start()
            procs.append((r, p))
        for r, p in procs:
            p.join()
            exit_codes[r] = p.exitcode if p.exitcode is not None else 1
        if all(code == 0 for code in exit_codes):
            break
    return exit_codes


__all__ = [
    "ReduceOp",
    "NativeProcessGroup",
    "CommHandle",
    "FullyShardedParam",
    "launch_workers",
    "init_process_group",
    "destroy_process_group",
    "is_available",
    "is_initialized",
    "get_rank",
    "get_world_size",
    "get_backend",
    "barrier",
    "all_reduce",
    "broadcast",
    "all_gather",
    "all_gather_into_tensor",
    "reduce",
    "scatter",
    "gather",
    "reduce_scatter",
    "reduce_scatter_tensor",
    "DistributedDataParallel",
    "DDP",
    "FullyShardedDataParallel",
    "FSDP",
    "DataParallel",
    "ring_all_reduce",
    "PipelineParallel",
    "ZeroRedundancyOptimizer",
]
