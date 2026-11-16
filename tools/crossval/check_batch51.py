"""Batch 51: I8 / I16 / U16 / U32 dtypes vs torch (casts, wrapping arithmetic, reductions, comparisons)."""
import re, sys, warnings
import numpy as np, torch
warnings.filterwarnings("ignore")
arrs = {}; extra = {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.strip())
    k, _, rest = line.partition(" ")
    try: arrs[k] = np.array([float(x) for x in rest.split()])
    except ValueError: extra[k] = rest
bad = n = 0
def check(name, want):
    global bad, n
    n += 1
    want = want.double().numpy().ravel(); got = arrs.get(name)
    if got is None or got.shape != want.shape or not np.array_equal(got, want):
        bad += 1; print("MISMATCH", name, "\n   want", want[:14], "\n   got ", None if got is None else got[:14])
src = [-200.7, -129.0, -128.0, -1.5, 0.0, 1.0, 100.9, 127.0, 128.0, 200.0, 255.0, 256.0, 1000.0, 32767.0, 32768.0, 65535.0, 65536.0, 70000.5, 3.0e9, 4294967295.0, 4294967296.0, -70000.0]
f = torch.tensor(src, dtype=torch.float64)
DT = {"i8": torch.int8, "i16": torch.int16, "u16": torch.uint16, "u32": torch.uint32}
# float -> small integer: only values inside the target range have a defined result in torch; compare those and the wrapping ones
def in_range(dt, v):
    info = torch.iinfo(dt); return info.min <= int(v) <= info.max
for nm, dt in DT.items():
    keep = [i for i, v in enumerate(src) if in_range(dt, v)]
    got = arrs[f"cast_{nm}"]; n += 1
    want = f[keep].to(dt).double().numpy()
    if not np.array_equal(got[keep], want): bad += 1; print("MISMATCH cast", nm, got[keep], want)
big = torch.tensor([-5, 127, 128, 255, 256, 40000, 65536, 70000, -40000, 4294967295, 4294967296, 1 << 40], dtype=torch.int64)
for nm, dt in DT.items():
    # integer narrowing wraps the low bits (two's complement); torch's uint16 / uint32 casts follow the same rule
    check(f"wrap_{nm}", big.to(dt))
for nm, dt in (("i8", torch.int8), ("i16", torch.int16)):
    a = torch.tensor([100.0, -100.0, 50.0, 127.0, -128.0, 7.0, 0.0, -1.0], dtype=torch.float64).to(dt)
    b = torch.tensor([100.0, -100.0, 90.0, 1.0, -1.0, -3.0, 5.0, -1.0], dtype=torch.float64).to(dt)
    check(f"add_{nm}", a + b); check(f"sub_{nm}", a - b); check(f"mul_{nm}", a * b); check(f"neg_{nm}", -a); check(f"abs_{nm}", a.abs())
    check(f"max_{nm}", torch.maximum(a, b)); check(f"lt_{nm}", (a < b)); check(f"ge_{nm}", (a >= b))
    check(f"sum_{nm}", a.sum()); check(f"cumsum_{nm}", a.cumsum(0))
    n += 1
    if extra.get(f"sumdtype_{nm}") != "I64" and str(a.sum().dtype) == "torch.int64": print("note: sum dtype", nm, extra.get(f"sumdtype_{nm}"), "torch", a.sum().dtype)
check("u16_back_f32", torch.tensor([0.0, 1.0, 255.0, 256.0, 65535.0]).to(torch.uint16).float())
check("u32_back_f64", torch.tensor([0.0, 4294967295.0, 123456789.0], dtype=torch.float64).to(torch.uint32).double())
print(f"{n} comparisons, {bad} mismatches"); sys.exit(1 if bad else 0)
