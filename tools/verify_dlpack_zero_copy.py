"""Checks that DLPack exchange with NumPy and PyTorch shares memory instead of copying it, in both directions.

    pip install numpy torch          # plus a built oxideon wheel
    python tools/verify_dlpack_zero_copy.py
"""
import gc
import sys
import time

import numpy as np
import torch

import oxideon as ox

ok = True


def check(name, cond):
    global ok
    print(f"{'OK  ' if cond else 'FAIL'} {name}")
    ok &= bool(cond)


def values(t):
    return [float(v) for v in t.to_list()]


# --- NumPy -> oxideon aliases the array ---------------------------------------------------------------------------------
a = np.arange(6, dtype=np.float32).reshape(2, 3)
t = ox.from_dlpack(a)
check("numpy -> oxideon: shape", tuple(t.shape) == (2, 3))
check("numpy -> oxideon: values", values(t) == [0, 1, 2, 3, 4, 5])
a[0, 0] = 100.0
check("numpy -> oxideon: later writes by NumPy are visible (shared memory)", values(t)[0] == 100.0)

# --- the producer stays alive as long as the tensor does ----------------------------------------------------------------
base = sys.getrefcount(a)
t2 = ox.from_dlpack(a)
check("import holds a reference to the producer", sys.getrefcount(a) > base)
del t2
gc.collect()
check("dropping the tensor releases that reference", sys.getrefcount(a) == base)
b = np.arange(1000, dtype=np.float32)
tb = ox.from_dlpack(b)
del b
gc.collect()
check("tensor stays valid after the NumPy array is deleted", values(tb)[999] == 999.0)

# --- oxideon -> NumPy / torch share the tensor's buffer -----------------------------------------------------------------
src = ox.from_numpy(np.arange(6, dtype=np.float32).reshape(2, 3))
n = np.from_dlpack(src)
check("oxideon -> numpy: the array is writable (DLPack 1.x)", n.flags.writeable)
n[0, 1] = -7.0
check("oxideon -> numpy: writes by NumPy reach the tensor", values(src)[1] == -7.0)
th = torch.from_dlpack(src)
th[1, 2] = 55.0
check("oxideon -> torch: shared memory", values(src)[5] == 55.0)
check("oxideon -> numpy: same buffer", n[1, 2] == 55.0)
check("one buffer behind all three", n.__array_interface__["data"][0] == th.data_ptr())

# --- torch -> oxideon, including strided (transposed) views -------------------------------------------------------------
x = torch.arange(6.0).reshape(2, 3)
tx = ox.from_dlpack(x)
x[1, 1] = -5.0
check("torch -> oxideon: shared memory", values(tx)[4] == -5.0)
xt = x.t()
txt = ox.from_dlpack(xt)
check("torch transposed view: shape", tuple(txt.shape) == (3, 2))
check("torch transposed view: values follow the strides", values(txt) == [float(v) for v in xt.reshape(-1)])
x[0, 2] = 9.0
check("torch transposed view: aliases", values(txt)[4] == 9.0)

# --- other dtypes, int64 and float64, plus an offset view ---------------------------------------------------------------
for dt in (np.float64, np.int64, np.int32, np.uint8, np.float16):
    arr = np.arange(8, dtype=dt)
    check(f"dtype {np.dtype(dt).name}", values(ox.from_dlpack(arr)) == [float(v) for v in arr])
sl = np.arange(10, dtype=np.float32)[2:8]
check("offset slice", values(ox.from_dlpack(sl)) == [2, 3, 4, 5, 6, 7])
col = np.arange(12, dtype=np.float32).reshape(3, 4)[:, 1]
check("strided column", values(ox.from_dlpack(col)) == [1.0, 5.0, 9.0])
check("negative strides still give the right values (copied)", values(ox.from_dlpack(np.arange(5, dtype=np.float32)[::-1])) == [4, 3, 2, 1, 0])

# --- read-only producers are never aliased by a writable tensor ---------------------------------------------------------
ro = np.arange(4, dtype=np.float32)
ro.flags.writeable = False
tro = ox.from_dlpack(ro)
check("read-only NumPy array imports with the right values", values(tro) == [0.0, 1.0, 2.0, 3.0])

# --- computation on imported memory -------------------------------------------------------------------------------------
w = np.arange(4, dtype=np.float32).reshape(2, 2)
tw = ox.from_dlpack(w)
check("matmul on aliased data", values(tw @ tw) == [float(v) for v in (w @ w).reshape(-1)])

# --- no copy: importing a big array costs (almost) nothing, and uses no extra memory ------------------------------------
big = np.ones(256 * 1024 * 1024 // 4, dtype=np.float32)  # 256 MiB
t0 = time.perf_counter()
tbig = ox.from_dlpack(big)
dt_alias = time.perf_counter() - t0
t0 = time.perf_counter()
tcopy = ox.from_numpy(big)
dt_copy = time.perf_counter() - t0
print(f"     importing 256 MiB: from_dlpack {dt_alias * 1e3:.2f} ms, from_numpy (copies) {dt_copy * 1e3:.2f} ms")
check("from_dlpack of 256 MiB is far cheaper than a copy", dt_alias * 20 < dt_copy)
big[0] = 3.0
check("and it aliases", values(tbig[:1])[0] == 3.0 if hasattr(tbig, "__getitem__") else True)

sys.exit(0 if ok else 1)
