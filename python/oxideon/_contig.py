"""`np.ascontiguousarray` wrapper that keeps 0-d arrays 0-d (NumPy promotes them to shape (1,), which would turn scalar
reduction results into 1-element tensors instead of 0-d scalars)."""
import numpy as np


def contig(a, dtype=None):
    return np.asarray(a, dtype=dtype, order="C")
