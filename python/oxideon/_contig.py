"""`np.ascontiguousarray` that keeps 0-d arrays 0-d (NumPy promotes them to shape (1,), which made scalar results of
reductions come back as 1-element tensors instead of PyTorch's 0-d ones)."""
import numpy as np


def contig(a, dtype=None):
    return np.asarray(a, dtype=dtype, order="C")
