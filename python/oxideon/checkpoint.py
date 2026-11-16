"""Activation recomputation and gradient checkpointing for memory-efficient deep learning.

Enables training deep neural network architectures by trading compute for memory:
activations are freed during the forward pass and recomputed on-the-fly during backpropagation.
"""

from typing import Any, Callable, List, Sequence, Tuple, Union
import contextlib
import random
import numpy as np

from .oxideon import Tensor as _Tensor, get_rng_state as _get_ox_rng_state, set_rng_state as _set_ox_rng_state
from .primitives import Function, no_grad, enable_grad, is_grad_enabled


class _CheckpointFunction(Function):
    """Custom autograd Function executing forward without storing internal activations,

    then recomputing the forward pass during the backward pass to generate exact gradients.
    """

    @staticmethod
    def forward(ctx, run_function: Callable[..., Any], preserve_rng_state: bool, *args):
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state

        if preserve_rng_state:
            ctx.py_rng_state = random.getstate()
            ctx.np_rng_state = np.random.get_state()
            ctx.ox_rng_state = _get_ox_rng_state()

        # Detach inputs for the recomputation forward pass
        detached_args = []
        for arg in args:
            if isinstance(arg, _Tensor):
                detached_args.append(arg.detach())
            else:
                detached_args.append(arg)
        ctx.saved_inputs = tuple(detached_args)

        # Run forward under no_grad to discard intermediate graph activations
        with no_grad():
            outputs = run_function(*args)

        ctx.is_tuple = isinstance(outputs, tuple)
        if not ctx.is_tuple:
            outputs = (outputs,)
        return outputs if ctx.is_tuple else outputs[0]

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Retrieve detached inputs and enable requires_grad on eligible tensors
        recompute_inputs = []
        for inp in ctx.saved_inputs:
            if isinstance(inp, _Tensor):
                t = inp.clone().detach().requires_grad_(True)
                recompute_inputs.append(t)
            else:
                recompute_inputs.append(inp)

        if getattr(ctx, "preserve_rng_state", False):
            py_state = random.getstate()
            np_state = np.random.get_state()
            ox_state = _get_ox_rng_state()
            random.setstate(ctx.py_rng_state)
            np.random.set_state(ctx.np_rng_state)
            _set_ox_rng_state(ctx.ox_rng_state)

        try:
            # Recompute forward pass under active autograd
            with enable_grad():
                outputs = ctx.run_function(*recompute_inputs)
                if not isinstance(outputs, tuple):
                    outputs = (outputs,)
        finally:
            if getattr(ctx, "preserve_rng_state", False):
                random.setstate(py_state)
                np.random.set_state(np_state)
                _set_ox_rng_state(ox_state)

        # Propagate upstream gradients through the recomputed subgraph
        from .primitives import autograd
        valid_grads = [g for g in grad_outputs if g is not None]
        inputs_with_grad = [inp for inp in recompute_inputs if isinstance(inp, _Tensor)]

        grads = autograd.grad(
            outputs=outputs[:len(valid_grads)],
            inputs=inputs_with_grad,
            grad_outputs=valid_grads,
            allow_unused=True,
        )

        # Map computed gradients back to original input signature (first 2 args are run_fn and preserve_rng)
        result_grads = [None, None]
        grad_idx = 0
        for inp in ctx.saved_inputs:
            if isinstance(inp, _Tensor):
                result_grads.append(grads[grad_idx] if grad_idx < len(grads) else None)
                grad_idx += 1
            else:
                result_grads.append(None)

        return tuple(result_grads)


def checkpoint(
    function: Callable[..., Any],
    *args: Any,
    use_reentrant: bool = True,
    preserve_rng_state: bool = True,
    **kwargs: Any,
) -> Any:
    """Computes a forward pass of `function(*args, **kwargs)` with activation checkpointing."""
    if kwargs:
        def _wrapper(*a):
            return function(*a, **kwargs)
        fn_to_run = _wrapper
    else:
        fn_to_run = function

    if not is_grad_enabled() or not any(isinstance(a, _Tensor) and a.requires_grad for a in args):
        return fn_to_run(*args)

    return _CheckpointFunction.apply(fn_to_run, preserve_rng_state, *args)


def checkpoint_sequential(
    functions: Any,
    segments: int,
    input: Any,
    preserve_rng_state: bool = True,
) -> Any:
    """Checkpoints a Sequential container or list of modules across `segments` chunks."""
    if segments < 1:
        raise ValueError(f"segments must be >= 1, got {segments}")
    if hasattr(functions, "_modules") and isinstance(functions._modules, dict) and functions._modules:
        mods = list(functions._modules.values())
    elif hasattr(functions, "layers"):
        mods = list(functions.layers)
    elif isinstance(functions, (list, tuple)):
        mods = list(functions)
    else:
        mods = list(functions)

    if len(mods) == 0:
        return input

    def _run_segment(start: int, end: int):
        def _forward(x):
            for m in mods[start:end]:
                x = m(x)
            return x
        return _forward

    n = len(mods)
    seg_size = max(1, (n + segments - 1) // segments)
    curr = input
    for start in range(0, n, seg_size):
        end = min(start + seg_size, n)
        if end < n:
            curr = checkpoint(_run_segment(start, end), curr, preserve_rng_state=preserve_rng_state)
        else:
            curr = _run_segment(start, end)(curr)
    return curr


checkpoint.checkpoint = checkpoint
checkpoint.checkpoint_sequential = checkpoint_sequential


__all__ = [
    "checkpoint",
    "checkpoint_sequential",
]
