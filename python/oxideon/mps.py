"""Apple Silicon Metal (MPS) acceleration interface for Oxideon."""

from typing import Any, Optional


def is_available() -> bool:
    """Returns a bool indicating if Metal Performance Shaders (MPS) is available."""
    return False


def is_built() -> bool:
    """Returns whether Oxideon was compiled with MPS support."""
    return False


def synchronize() -> None:
    """Waits for all MPS streams to complete."""
    pass


def empty_cache() -> None:
    """Frees all cached memory allocations held by MPS allocator."""
    pass


def current_allocated_memory() -> int:
    """Returns current allocated memory on MPS device in bytes."""
    return 0


def driver_allocated_memory() -> int:
    """Returns driver allocated memory on MPS device in bytes."""
    return 0


def manual_seed(seed: int) -> None:
    """Sets random seed for MPS device."""
    from .oxideon import manual_seed as _manual_seed
    _manual_seed(seed)


__all__ = [
    "is_available",
    "is_built",
    "synchronize",
    "empty_cache",
    "current_allocated_memory",
    "driver_allocated_memory",
    "manual_seed",
]
