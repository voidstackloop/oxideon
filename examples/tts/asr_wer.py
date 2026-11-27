"""Word error rate of the speech written by `evaluate_voca.py`, measured with a small speech recogniser (faster-whisper tiny.en).

    python examples/tts/asr_wer.py --dir runs/voca/eval

The real recording of the same sentence is scored too: its WER is the floor (what the recogniser gets wrong on natural speech), so
`ours` and `gl` are judged against it. Needs `pip install faster-whisper jiwer`.
"""
import argparse
import json
import os
import re
import sys

import jiwer
from faster_whisper import WhisperModel


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frontend_cmu import normalize  # noqa: E402  (digits -> words, so "3" and "three" match; pure Python, no oxideon needed)


def norm(s):
    """Lower case, digits spelled out, quotes dropped (inner apostrophes such as don't stay), single spaces."""
    s = re.sub(r"[^a-z' ]", " ", normalize(s.lower().replace(",", "").replace("-", " ")))
    return re.sub(r"\s+", " ", " ".join(w.strip("'") for w in s.split())).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--model", default="tiny.en")
    a = ap.parse_args()
    rows = json.load(open(os.path.join(a.dir, "eval.json")))
    asr = WhisperModel(a.model, device="cpu", compute_type="int8")
    hyp = {t: [] for t in ("ours", "ours_lex", "ours_gt", "gl", "real", "oracle_ours", "oracle_gl")}
    ref = []
    for r in rows:
        ref.append(norm(r["text"]))
        for t in hyp:
            segs, _ = asr.transcribe(os.path.join(a.dir, f"{r['id']}_{t}.wav"), language="en", beam_size=5)
            hyp[t].append(norm(" ".join(s.text for s in segs)))
    res = {"clips": len(rows), "model": a.model}
    for t in hyp:
        res[t] = {"wer": round(jiwer.wer(ref, hyp[t]), 4), "cer": round(jiwer.cer(ref, hyp[t]), 4)}
    for v, name in ((0, "F2"), (1, "M2")):
        idx = [i for i, r in enumerate(rows) if r["voice"] == v]
        res[name] = {t: round(jiwer.wer([ref[i] for i in idx], [hyp[t][i] for i in idx]), 4) for t in hyp}
    res["examples"] = [dict(text=ref[i], **{t: hyp[t][i] for t in hyp}) for i in range(min(8, len(rows)))]
    res["all"] = [dict(id=r["id"], text=ref[i], **{t: hyp[t][i] for t in hyp}) for i, r in enumerate(rows)]
    json.dump(res, open(os.path.join(a.dir, "asr.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("examples", "all")}, indent=1))


if __name__ == "__main__":
    main()
