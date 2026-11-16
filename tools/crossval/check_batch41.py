"""Batch 41: Unicode general categories, combining classes and the four normalization forms vs unicodedata."""
import re, sys, unicodedata as ud

got = {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.strip())
    k, _, v = line.partition(" ")
    if v.isdigit(): got[k] = int(v)

def fnv(h, b):
    for x in b: h = ((h ^ x) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return ((h ^ 0xff) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF

OFF = 0xcbf29ce484222325
hc = hcc = OFF
hs = {f: OFF for f in ("NFC", "NFD", "NFKC", "NFKD")}
for cp in range(0x110000):
    if 0xD800 <= cp <= 0xDFFF: continue
    c = chr(cp)
    hc = fnv(hc, ud.category(c).encode()); hcc = fnv(hcc, bytes([ud.combining(c)]))
    for f in hs: hs[f] = fnv(hs[f], ud.normalize(f, c).encode("utf-8", "surrogatepass"))
pool = list("aeiouAEIOUnNcCsSzẒ゙ְًุ̧̨̱́̀̈ͅ각가각ᄒᆫﬁ①㎏ǆǅÅÅẛ̣Ωöạཱིྀུḋ̇̈́क़োাାେ")
M = 0xFFFFFFFFFFFFFFFF
x = 88172645463325252
hq = {f: OFF for f in hs}
for _ in range(40000):
    x ^= (x << 13) & M; x ^= x >> 7; x ^= (x << 17) & M
    y = x; s = ""
    for _ in range(1 + y % 7):
        y = (y * 6364136223846793005 + 1442695040888963407) & M
        s += pool[(y >> 33) % len(pool)]
    for f in hq: hq[f] = fnv(hq[f], ud.normalize(f, s).encode())
bad = n = 0
def check(name, want):
    global bad, n
    n += 1
    if got.get(name) != want: bad += 1; print("MISMATCH", name, got.get(name), want)
check("cat", hc); check("ccc", hcc)
for f in hs: check("single_" + f.lower(), hs[f]); check("seq_" + f.lower(), hq[f])
print(f"{n} checks (ud {ud.unidata_version}), {bad} mismatches"); sys.exit(1 if bad else 0)
