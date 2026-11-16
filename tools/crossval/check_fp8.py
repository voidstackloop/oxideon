"""Compare oxideon's f32 -> float8 codes with torch. Usage: check_fp8.py <emit file> (lines 'FP8 <f32 bits> <e4m3 code> <e5m2 code>')."""
import struct, sys
import torch

bad = n = 0
for line in open(sys.argv[1]):
    p = line.split()
    if len(p) != 4 or p[0] != "FP8":
        continue
    v = struct.unpack("<f", struct.pack("<I", int(p[1])))[0]
    t = torch.tensor([v], dtype=torch.float32)
    for name, dt, got in (("e4m3", torch.float8_e4m3fn, int(p[2])), ("e5m2", torch.float8_e5m2, int(p[3]))):
        want = t.to(dt).view(torch.uint8).item()
        n += 1
        if want != got:
            bad += 1
            if bad <= 20:
                print(f"MISMATCH {name} v={v!r} torch={want:#04x} oxideon={got:#04x}")
print(f"checked {n} codes, {bad} mismatches")
sys.exit(1 if bad else 0)
