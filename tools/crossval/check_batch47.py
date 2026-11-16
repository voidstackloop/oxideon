"""Batch 47: chat templates of real models: the Rust Jinja subset vs jinja2 with the transformers environment.
Run after: py -3.11 tools/crossval/fetch_chat_templates.py target/chat_configs"""
import json, os, re, sys, warnings
warnings.filterwarnings("ignore")
from transformers.utils.chat_template_utils import render_jinja_template

here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, "..", "..")
got = {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.rstrip("\n"))
    k, _, rest = line.partition(" ")
    try: got[k] = json.loads(rest)
    except ValueError: pass
cases = json.load(open(f"{here}/tokenizer_fixtures/chat_cases.json", encoding="utf-8"))
d = os.path.join(root, "target", "chat_configs")
def tok(v): return v.get("content") if isinstance(v, dict) else v
bad = n = skipped = 0
per = {}
for fn in sorted(os.listdir(d)):
    name = fn[:-5]; cfg = json.load(open(os.path.join(d, fn), encoding="utf-8"))
    tpl = cfg.get("chat_template")
    if isinstance(tpl, list): tpl = next((t["template"] for t in tpl if t["name"] == "default"), tpl[0]["template"])
    if not tpl: continue
    if f"{name}.parse" in got: bad += 1; n += 1; print("PARSE FAILED", name, got[f"{name}.parse"]["err"][:300]); continue
    timedep = "strftime_now" in tpl
    for i, c in enumerate(cases):
        g = got.get(f"{name}.c{i}")
        if g is None: bad += 1; n += 1; print("MISSING", name, i); continue
        kwargs = dict(bos_token=tok(cfg.get("bos_token")), eos_token=tok(cfg.get("eos_token")), unk_token=tok(cfg.get("unk_token")), pad_token=tok(cfg.get("pad_token")))
        try:
            want, _ = render_jinja_template([json.loads(json.dumps(c["messages"]))], c.get("tools"), None, tpl, False, False, c["gen"], **kwargs)
            want = want[0]; werr = None
        except Exception as e: want = None; werr = str(e)
        n += 1
        if timedep:
            skipped += 1
            if (want is None) != ("err" in g): bad += 1; per[name] = per.get(name, 0) + 1; print("ERR-NESS (time-dependent template)", name, i)
            continue
        if werr is not None:
            if "err" not in g: bad += 1; per[name] = per.get(name, 0) + 1; print("EXPECTED ERROR", name, i, werr[:80], "| got output", repr(g["out"][:60]))
        elif "err" in g:
            bad += 1; per[name] = per.get(name, 0) + 1
            if per[name] <= 3: print("UNEXPECTED ERROR", name, i, g["err"][:200])
        elif g["out"] != want:
            bad += 1; per[name] = per.get(name, 0) + 1
            if per[name] <= 3:
                k = next((j for j in range(min(len(want), len(g["out"]))) if want[j] != g["out"][j]), min(len(want), len(g["out"])))
                print("MISMATCH", name, i, "first diff at", k, "\n   want", repr(want[max(0, k-40):k+60]), "\n   got ", repr(g["out"][max(0, k-40):k+60]))
print(f"{n} renders ({skipped} time-dependent compared for error-ness only), {bad} mismatches"); sys.exit(1 if bad else 0)
