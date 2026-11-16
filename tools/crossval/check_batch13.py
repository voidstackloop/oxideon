"""Batch 13: string metrics vs rapidfuzz, BM25 vs an independent numpy implementation."""
import math, re, sys
from collections import Counter
from rapidfuzz.distance import JaroWinkler, OSA, LCSseq

vals = {}
for line in open(sys.argv[1]):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.strip())
    p = line.split()
    if len(p) == 2:
        try: vals[p[0]] = float(p[1])
        except ValueError: pass

PAIRS = [("abcxxxxxxx","abcyyyyyyy"),("martha","marhta"),("dixon","dicksonx"),("jellyfish","smellyfish"),("abc","xyz"),("kitten","sitting"),
         ("ca","abc"),("hello","hello"),("crate","trace"),("a","b"),("prefixmatch","prefixmiss"),("dwayne","duane"),
         ("shackleford","shackelford"),("abcdef","badcfe"),("","abc")]
bad = n = 0
def cmp(name, got, want, tol=1e-9):
    global bad, n
    n += 1
    if abs(got - want) > tol * max(1, abs(want)):
        bad += 1
        print("MISMATCH", name, got, want)

for i, (a, b) in enumerate(PAIRS):
    cmp(f"jw_{i}", vals[f"jw_{i}"], JaroWinkler.similarity(a, b))
    cmp(f"dl_{i}", vals[f"dl_{i}"], OSA.distance(a, b))
    cmp(f"lcs_{i}", vals[f"lcs_{i}"], LCSseq.similarity(a, b))

docs = ["the quick brown fox jumps over the lazy dog","never jump over the lazy dog quickly",
        "a quick brown dog outpaces a quick fox","lorem ipsum dolor sit amet","the the the the fox"]
tok = lambda s: [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]
T = [tok(d) for d in docs]
N = len(T); avg = sum(map(len, T)) / N
df = Counter(t for d in T for t in set(d))
def bm25(q, i, k1, b):
    tf = Counter(T[i]); s = 0.0
    for t in tok(q):
        f = tf.get(t, 0)
        if f == 0: continue
        idf = math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1)
        s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(T[i]) / avg))
    return s
for k1, b in [(1.5,0.75),(1.2,0.3),(0.0,0.5),(2.0,1.0)]:
    for qi, q in enumerate(["quick fox","lazy dog the","ipsum","the fox fox"]):
        for i in range(N):
            cmp(f"bm25_{k1}_{b}_{qi}_{i}", vals[f"bm25_{k1:g}_{b:g}_{qi}_{i}"], bm25(q, i, k1, b))
print(f"{n} checks, {bad} mismatches")
sys.exit(1 if bad else 0)
