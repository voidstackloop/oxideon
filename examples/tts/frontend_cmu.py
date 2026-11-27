"""A text front end for a TinyTTS trained on the tiny-voca labels, built on the CMU Pronouncing Dictionary (BSD licence).

The labels use IPA-like symbols, CMUdict uses ARPABET. The mapping between them depends on context (`AH` is `ə`, `ʌ`, `ɪ` or `ᵻ`; `T`
is `t` or the flap `ɾ`; `AO`+`R` merge into `ɔːɹ`), so it is *learned* from the words that appear in both: each CMU pronunciation is
aligned with the labelled tokens (Viterbi, re-estimated twice) and a back-off table (phone, stress, previous phone, next phone) keeps
the commonest symbol. Stress follows the same pattern (a function word's primary stress becomes 0). Words missing from CMUdict are
tried with a suffix stripped (-s, -ed, -ing, 's), then taken from the labelled lexicon, then built from known pieces.

    fe = CmuFrontEnd.learn("runs/tinyvoca/annotations/v4", "cmudict.dict", lexicon)   # or CmuFrontEnd.load("frontend.json", "cmudict.dict", lexicon)
    tokens, stress, unknown = fe.phonemize("Dr. Reynolds baked 350 cakes.")
"""
import collections
import json
import math
import os
import re

BOS, EOS, SP = 1, 2, 3
QUOTE = 11
PUNCT_IDS = {",": 5, ".": 6, "?": 7, "!": 8, ";": 9, ":": 10}
SKIP = set(",.?!;:\"'()—")

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
ABBR = {"dr": "doctor", "mr": "mister", "mrs": "missus", "ms": "miss", "st": "saint", "vs": "versus", "etc": "et cetera", "no": "number"}


def load_cmudict(path):
    """`cmudict.dict` (lines `word PH1 PH2 ...`, variants as `word(2)`) -> {word: [[phones], ...]}."""
    out = {}
    for line in open(path, encoding="utf8", errors="ignore"):
        parts = line.split("#")[0].split()
        if len(parts) >= 2:
            out.setdefault(re.sub(r"\(\d+\)$", "", parts[0]).lower(), []).append(parts[1:])
    return out


def number_words(n):
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + number_words(n % 100))
    for size, name in ((10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")):
        if n >= size:
            return number_words(n // size) + " " + name + ("" if n % size == 0 else " " + number_words(n % size))


def normalize(text):
    """Digits, a few abbreviations and symbols -> plain words (keeps the punctuation the model knows)."""
    text = re.sub(r"\b([Dd]r|[Mm]rs?|[Mm]s|[Ss]t|vs|etc)\.", lambda m: ABBR[m.group(1).lower()] + ("." if m.group(1).lower() == "etc" else ""), text)
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    text = re.sub(r"\$(\d+)\.(\d\d)\b", r"\1 dollars and \2 cents", text)
    text = re.sub(r"\$(\d+)", lambda m: m.group(1) + (" dollar" if m.group(1) == "1" else " dollars"), text)
    text = re.sub(r"(\d+(?:\.\d+)?)%", r"\1 percent", text)
    text = re.sub(r"(\d+)\.(\d+)", lambda m: number_words(int(m.group(1))) + " point " + " ".join(ONES[int(c)] for c in m.group(2)), text)
    text = re.sub(r"\b(1[1-9]\d\d|20\d\d)\b", lambda m: number_words(int(m.group(1)[:2])) + " " + (("oh " + ONES[int(m.group(1)[2:])]) if m.group(1)[2:3] == "0" and m.group(1)[3:] != "0" else number_words(int(m.group(1)[2:])) if m.group(1)[2:] != "00" else "hundred"), text)
    text = re.sub(r"\d+", lambda m: number_words(int(m.group(0))) if len(m.group(0)) < 10 else m.group(0), text)
    return text.replace("&", " and ")


def base(p):
    return re.sub(r"\d", "", p)


def digit(p):
    m = re.search(r"\d", p)
    return m.group(0) if m else ""


def _align(P, L, c1, c2, big=7.0):
    """Viterbi alignment of CMU phones P (str) with label tokens L (sym, stress). Ops: 1:1, 2:1 merge, delete phone, insert token.
    Returns (cost, [(i0, i1, j or None)])."""
    n, m = len(P), len(L)
    INF = 1e9
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    B = [[None] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            d = D[i][j]
            if d >= INF:
                continue
            if i < n and j < m:
                c = d + c1.get((base(P[i]), L[j][0]), big)
                if c < D[i + 1][j + 1]:
                    D[i + 1][j + 1], B[i + 1][j + 1] = c, (i, j, i + 1)
            if i + 1 < n and j < m:
                c = d + c2.get((base(P[i]), base(P[i + 1]), L[j][0]), 99.0)
                if c < D[i + 2][j + 1]:
                    D[i + 2][j + 1], B[i + 2][j + 1] = c, (i, j, i + 2)
            if i < n:
                c = d + big
                if c < D[i + 1][j]:
                    D[i + 1][j], B[i + 1][j] = c, (i, j, i + 1)
            if j < m:
                c = d + big
                if c < D[i][j + 1]:
                    D[i][j + 1], B[i][j + 1] = c, (i, j, i)
    ops, i, j = [], n, m
    while (i, j) != (0, 0):
        pi, pj, _ = B[i][j]
        ops.append((pi, i, pj if j > pj else None))
        i, j = pi, pj
    return D[n][m], ops[::-1]


def _sentence_words(r, syms):
    words = r["text"].split()
    segs, cur = [], []
    for s, x in zip(syms[1:-1], r["stress"][1:-1]):
        if s == "<sp>":
            segs.append(cur), (cur := [])
        else:
            cur.append((s, x))
    segs.append(cur)
    if len(segs) != len(words):
        return
    for w, seg in zip(words, segs):
        w = re.sub(r"^[^a-z']+|[^a-z']+$", "", w.lower())
        body = [(a, b) for a, b in seg if a not in SKIP]
        if w and body:
            yield w, body


class CmuFrontEnd:
    def __init__(self, rules, cmu, lex, sym2id):
        self.rules, self.cmu, self.lex, self.sym2id = rules, cmu, lex, sym2id

    # ------------------------------------------------------------------ learning
    @classmethod
    def learn(cls, ann_dir, cmu, lex, holdout_every=10):
        symtab = {}
        for l in open(os.path.join(ann_dir, "phonemes.jsonl"), encoding="utf8"):
            j = json.loads(l)
            symtab[j["text_hash"]] = j["symbols"]
        sym2id, words = {}, collections.defaultdict(collections.Counter)
        for line in open(os.path.join(ann_dir, "train.jsonl"), encoding="utf8"):
            r = json.loads(line)
            syms = symtab.get(r["text_hash"])
            if syms is None or len(syms) != len(r["tokens"]):
                continue
            for t, s in zip(r["tokens"], syms):
                sym2id[s] = t
            for w, body in _sentence_words(r, syms):
                if w in cmu:
                    words[w][tuple(body)] += 1
        keys = sorted(words)
        held = {w for i, w in enumerate(keys) if holdout_every and i % holdout_every == 0}  # never used for learning: honest PER
        train = {w: c for w, c in words.items() if w not in held}
        # symbol costs: start from words whose phone count equals the token count
        c1, c2 = {}, {}
        cnt = collections.Counter()
        for w, c in train.items():
            body = c.most_common(1)[0][0]
            for pron in cmu[w]:
                if len(pron) == len(body):
                    for a, (s, _) in zip(pron, body):
                        cnt[(base(a), s)] += 1
                    break
        tot = collections.Counter()
        for (a, s), n in cnt.items():
            tot[a] += n
        c1 = {k: -math.log(n / tot[k[0]]) + 0.05 for k, n in cnt.items()}
        for a in ("AO", "AA", "OW", "IH", "EH", "UH", "ER", "UW", "AY", "AE"):
            c2[(a, "R", "ɔːɹ")] = c2[(a, "R", "ɑːɹ")] = c2[(a, "R", "oːɹ")] = c2[(a, "R", "ɪɹ")] = c2[(a, "R", "ɛɹ")] = c2[(a, "R", "ʊɹ")] = 1.0
        c2[("AY", "ER", "aɪɚ")] = c2[("AY", "AH", "aɪə")] = c2[("AH", "L", "əl")] = c2[("AH", "N", "n̩")] = c2[("IY", "AH", "iə")] = c2[("AY", "AH", "aɪə")] = 1.0
        best = {}
        for it in range(3):
            n1, n2 = collections.Counter(), collections.Counter()
            best = {}
            for w, c in train.items():
                body = c.most_common(1)[0][0]
                cand = []
                for pron in cmu[w][:4]:
                    cost, ops = _align(pron, body, c1, c2)
                    cand.append((cost, pron, ops))
                cost, pron, ops = min(cand, key=lambda x: x[0])
                best[w] = (pron, body, ops)
                for i0, i1, j in ops:
                    if j is None:
                        continue
                    if i1 - i0 == 1:
                        n1[(base(pron[i0]), body[j][0])] += 1
                    elif i1 - i0 == 2:
                        n2[(base(pron[i0]), base(pron[i0 + 1]), body[j][0])] += 1
            tot = collections.Counter()
            for (a, s), n in n1.items():
                tot[a] += n
            c1 = {k: -math.log(n / tot[k[0]]) + 0.05 for k, n in n1.items()}
            c2 = {k: max(0.2, 2.0 - math.log(1 + n)) for k, n in n2.items()}
        # context tables
        tabs = [collections.defaultdict(collections.Counter) for _ in range(5)]
        pair_total, pair_merged = collections.Counter(), collections.Counter()
        for w, (pron, body, ops) in best.items():
            ph = [base(p) for p in pron]
            for i0, i1, j in ops:
                if i1 - i0 == 2:
                    pair_merged[(ph[i0], ph[i0 + 1], body[j][0])] += 1
                if i1 - i0 == 0:
                    continue
                if i1 - i0 == 1:
                    b, d = ph[i0], digit(pron[i0])
                    pv, nx = (ph[i0 - 1] if i0 else "#"), (ph[i0 + 1] if i0 + 1 < len(ph) else "#")
                    sym = body[j][0] if j is not None else ""
                    for t, k in zip(tabs, (f"{b}|{d}|{pv}|{nx}", f"{b}|{d}|{nx}", f"{b}|{d}|{pv}", f"{b}|{d}", b)):
                        t[k][sym] += 1
                    if i0 + 1 < len(ph):
                        pair_total[(b, ph[i0 + 1])] += 1
        rules = {"tables": [{k: [c.most_common(1)[0][0], sum(c.values())] for k, c in t.items()} for t in tabs]}
        merges = {}
        for (a, b, s), n in pair_merged.items():
            tot_ab = n + pair_total[(a, b)]
            if n / tot_ab > 0.5 and n >= 4:
                merges[f"{a}|{b}"] = s
        rules["merge"] = merges
        # stress: (syllables, cmu digit, function word) -> label stress; function words = monosyllables whose primary stress is dropped
        votes = collections.defaultdict(lambda: [0, 0])
        stab = collections.defaultdict(collections.Counter)
        for w, (pron, body, ops) in best.items():
            digs = [digit(p) for p in pron if digit(p)]
            if not digs:
                continue
            labs = []
            for i0, i1, j in ops:
                if j is not None and i1 - i0 >= 1 and any(digit(pron[i]) for i in range(i0, i1)):
                    labs.append((digit(pron[i0]) or digit(pron[i0 + 1]), body[j][1]))
            if len(labs) != len(digs):
                continue
            if len(digs) == 1 and digs[0] == "1":
                votes[w][0] += labs[0][1] == 0
                votes[w][1] += 1
        func = sorted(w for w, (z, n) in votes.items() if z / n >= 0.7)
        funcset = set(func)
        for w, (pron, body, ops) in best.items():
            digs = [digit(p) for p in pron if digit(p)]
            labs = [(digit(pron[i0]) or (digit(pron[i0 + 1]) if i1 - i0 > 1 else ""), body[j][1]) for i0, i1, j in ops if j is not None and i1 - i0 >= 1 and any(digit(pron[i]) for i in range(i0, i1))]
            if len(labs) == len(digs) and digs:
                for d, (_, l) in zip(digs, labs):
                    stab[f"{min(len(digs), 3)}|{d}|{int(w in funcset)}"][l] += 1
        rules["stress"] = {k: c.most_common(1)[0][0] for k, c in stab.items()}
        rules["function_words"] = func
        # held-out accuracy (words never used above)
        fe = cls(rules, cmu, lex, sym2id)
        err = tot_n = ok = 0
        for w in held:
            body = [s for s, _ in words[w].most_common(1)[0][0]]
            best_e = min(_edit(fe.word_tokens(w, v)[0], body) for v in range(min(len(cmu[w]), 3)))
            first = _edit(fe.word_tokens(w, 0)[0], body)
            err += first
            tot_n += len(body)
            ok += first == 0
        rules["heldout"] = {"words": len(held), "token_error_rate": err / max(tot_n, 1), "exact_word_rate": ok / max(len(held), 1)}
        return fe

    def save(self, path):
        json.dump({"rules": self.rules, "sym2id": self.sym2id}, open(path, "w", encoding="utf8"), ensure_ascii=False)

    @classmethod
    def load(cls, path, cmu, lex):
        j = json.load(open(path, encoding="utf8"))
        return cls(j["rules"], cmu, lex, j["sym2id"])

    # ------------------------------------------------------------------ applying
    def _lookup(self, b, d, pv, nx):
        t = self.rules["tables"]
        for ti, k, need in ((0, f"{b}|{d}|{pv}|{nx}", 3), (1, f"{b}|{d}|{nx}", 3), (2, f"{b}|{d}|{pv}", 3), (3, f"{b}|{d}", 1), (4, b, 1)):
            e = t[ti].get(k)
            if e and e[1] >= need:
                return e[0]
        return None

    def word_tokens(self, word, variant=0):
        """`(symbols, stress)` for a word in CMUdict (variant picks the pronunciation)."""
        pron = self.cmu[word][variant]
        ph = [base(p) for p in pron]
        n_syll = sum(1 for p in pron if digit(p))
        func = int(word in self._funcs)
        syms, st, i = [], [], 0
        while i < len(pron):
            m = self.rules["merge"].get(f"{ph[i]}|{ph[i + 1]}") if i + 1 < len(pron) else None
            pv, nx = (ph[i - 1] if i else "#"), (ph[i + 1] if i + 1 < len(ph) else "#")
            sym = m if m else self._lookup(ph[i], digit(pron[i]), pv, nx)
            d = digit(pron[i]) or (digit(pron[i + 1]) if m else "")
            if sym is None:
                sym = ph[i].lower()
            if sym:
                syms.append(sym)
                st.append(int(self.rules["stress"].get(f"{min(n_syll, 3)}|{d}|{func}", 0)) if d else 0)
            i += 2 if m else 1
        return syms, st

    @property
    def _funcs(self):
        if not hasattr(self, "_f"):
            self._f = set(self.rules["function_words"])
        return self._f

    def _with_suffix(self, w):
        """`(symbols, stress)` for inflected words missing from CMUdict (plural / past / -ing / possessive), else None."""
        for suf, rule in (("'s", "s"), ("s", "s"), ("es", "s"), ("ed", "ed"), ("d", "ed"), ("ing", "ing")):
            if w.endswith(suf) and len(w) > len(suf) + 1:
                stem = w[: -len(suf)]
                for cand in (stem, stem + "e", stem[:-1] if len(stem) > 1 and stem[-1] == stem[-2] else None):
                    if cand and cand in self.cmu:
                        s, st = self.word_tokens(cand)
                        last = s[-1] if s else ""
                        if rule == "s":
                            add = ["ɪz"] if last in ("s", "z", "ʃ", "ʒ", "tʃ", "dʒ") else (["s"] if last in ("p", "t", "k", "f", "θ") else ["z"])
                        elif rule == "ed":
                            add = ["ɪd"] if last in ("t", "d") else (["t"] if last in ("p", "k", "f", "s", "ʃ", "tʃ", "θ") else ["d"])
                        else:
                            add = ["ɪŋ"]
                        return s + add, st + [0]
        return None

    def add_pronunciation(self, word, arpabet):
        """Teach a word (a product name, say) by its ARPABET spelling, e.g. `add_pronunciation("oxideon", "AA1 K S AH0 D IY0 AA2 N")`."""
        word = word.lower()
        self.cmu[word] = [arpabet.split()]
        self.lex.pop(word, None)

    def _pieces(self, word):
        """Last resort for names and rare words: greedy longest known lexicon pieces (no pronunciation rules for spelling)."""
        ids, st, i = [], [], 0
        while i < len(word):
            for j in range(len(word), i + 1, -1):
                if word[i:j] in self.lex:
                    ids += list(self.lex[word[i:j]][0])
                    st += list(self.lex[word[i:j]][1])
                    i = j
                    break
            else:
                i += 1
        return ids, st

    def _word(self, body, unknown):
        """`(ids, stress)` of one word (no hyphen)."""
        if body in self.lex:  # seen in the labels: exactly the pronunciation the model was trained on
            return list(self.lex[body][0]), list(self.lex[body][1])
        r = (self.word_tokens(body) if body in self.cmu else None) or self._with_suffix(body)
        if r:
            ids = [self.sym2id[s] for s in r[0] if s in self.sym2id]
            return ids, [x for s, x in zip(*r) if s in self.sym2id]
        ids, st = self._pieces(body)
        if not ids:
            unknown.append(body)
        return ids, st

    def phonemize(self, text):
        """`(tokens, stress, unknown_words)` in the label vocabulary, laid out like the labels: <bos>, words joined by <sp>, hyphenated
        compounds joined directly, punctuation / quote tokens attached with no <sp> after them, <eos>."""
        toks, stress, unknown = [BOS], [0], []
        after_phone = False  # the previous token is a phone (a word is open), so the next word needs a <sp>
        for m in re.finditer(r"[A-Za-z']+(?:-[A-Za-z']+)*|[,.;:?!]|[\"\u201c\u201d]", normalize(text)):
            w = m.group(0)
            if w in PUNCT_IDS:
                toks.append(PUNCT_IDS[w]), stress.append(0)
                after_phone = False
                continue
            if w in ('"', "\u201c", "\u201d"):
                if after_phone and w != "\u201d":
                    toks.append(SP), stress.append(0)
                toks.append(QUOTE), stress.append(0)
                after_phone = False
                continue
            ids, st = [], []
            for part in w.split("-"):
                body = part.strip("'").lower()
                if body:
                    i, s_ = self._word(body, unknown)
                    ids += i
                    st += s_
            if not ids:
                continue
            if after_phone:
                toks.append(SP), stress.append(0)
            toks += ids
            stress += list(st)
            after_phone = True
        toks.append(EOS), stress.append(0)
        return toks, stress, unknown


def _edit(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]
