"""Turns synthesized clips (see Synth.cs: <id>.wav + <id>.ev) into training arrays for `train_tts.py`:
log-mel frames, SAPI phoneme ids with their durations in frames, word-start flags, the speaker, and a word -> phonemes lexicon.

    python examples/tts/prepare_tts.py --clips runs/tts/clips --sentences runs/tts/sentences.txt --out runs/tts/data
"""
import argparse
import collections
import json
import os
import re
import wave

import numpy as np

import oxideon as ox

SAMPLE_RATE, N_FFT, HOP, N_MELS = 16000, 1024, 256, 80
SILENCE = "7"  # the SAPI phone for silence


def read_wav(path):
    with wave.open(path) as w:
        assert w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def parse_events(path):
    words, phones = [], []
    for line in open(path):
        t = line.split()
        if t[0] == "W":
            words.append((int(t[1]), int(t[2]), int(t[3])))
        else:
            phones.append((int(t[1]), int(t[2]), t[3]))
    return words, phones


def log_mel(x, spec_fn, fb):
    mag = spec_fn(ox.from_numpy(x)).to_numpy()  # [freq, frames]
    return np.log(np.maximum(fb @ mag, 1e-5)).T.astype(np.float32)  # [frames, mels]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-sentences", type=int, default=100)
    ap.add_argument("--max-frames", type=int, default=700)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sents = [l.strip() for l in open(a.sentences, encoding="utf8") if l.strip()]
    spec_fn = ox.audio.Spectrogram(n_fft=N_FFT, hop_length=HOP, power=1.0)
    fb = ox.audio.mel_filter_bank(n_mels=N_MELS, n_fft=N_FFT, sample_rate=SAMPLE_RATE).astype(np.float32)
    frame_ms = 1000.0 * HOP / SAMPLE_RATE
    code_ids = {SILENCE: 1}
    lexicon = collections.defaultdict(collections.Counter)
    rows = []
    for i, s in enumerate(sents):
        for tag, voice in (("F", 0), ("M", 1)):
            cid = f"u{i:05d}{tag}"
            wav, ev = os.path.join(a.clips, cid + ".wav"), os.path.join(a.clips, cid + ".ev")
            if not (os.path.exists(wav) and os.path.exists(ev)):
                continue
            words, phones = parse_events(ev)
            if not phones:
                continue
            mel = log_mel(read_wav(wav), spec_fn, fb)
            n_frames = mel.shape[0]
            if n_frames > a.max_frames:
                continue
            for _, _, c in phones:
                code_ids.setdefault(c, len(code_ids) + 1)
            starts = [p[0] for p in phones]
            bounds = [min(n_frames, int(round(st / frame_ms))) for st in starts] + [n_frames]
            dur = np.diff(bounds).astype(np.int16)
            ids = np.array([code_ids[c] for _, _, c in phones], dtype=np.int16)
            ws = np.zeros(len(phones), dtype=np.uint8)
            word_ends = [w[0] for w in words[1:]] + [10 ** 9]
            for (w_start, cpos, clen), nxt in zip(words, word_ends):
                idx = [k for k, st in enumerate(starts) if w_start <= st < nxt]
                if not idx:
                    continue
                ws[idx[0]] = 1
                word = s[cpos:cpos + clen].lower()
                body = [c for k, c in ((k, phones[k][2]) for k in idx) if c != SILENCE]
                if re.fullmatch(r"[a-z']+", word) and body:
                    lexicon[word][tuple(code_ids[c] for c in body)] += 1
            rows.append(dict(i=i, voice=voice, mel=mel.astype(np.float16), ids=ids, dur=dur, ws=ws))
        if i % 500 == 0:
            print(f"{i}/{len(sents)} sentences, {len(rows)} clips", flush=True)
    allmel = np.concatenate([r["mel"].astype(np.float32) for r in rows[::7]])
    mean, std = float(allmel.mean()), float(allmel.std())
    print(f"mel mean {mean:.3f} std {std:.3f}; {len(code_ids)} phone codes; {len(lexicon)} lexicon words")

    def save(name, subset):
        off = lambda key: np.cumsum([0] + [len(r[key]) for r in subset]).astype(np.int64)
        np.savez(os.path.join(a.out, name + ".npz"), mel=np.concatenate([r["mel"] for r in subset]), mel_off=off("mel"),
                 ids=np.concatenate([r["ids"] for r in subset]), dur=np.concatenate([r["dur"] for r in subset]),
                 ws=np.concatenate([r["ws"] for r in subset]), ph_off=off("ids"), voice=np.array([r["voice"] for r in subset], dtype=np.int8))
        print(f"{name}: {len(subset)} clips, {sum(len(r['mel']) for r in subset) * HOP / SAMPLE_RATE / 3600:.2f} hours")

    save("val", [r for r in rows if r["i"] >= len(sents) - a.val_sentences])
    save("train", [r for r in rows if r["i"] < len(sents) - a.val_sentences])
    lex = {w: list(c.most_common(1)[0][0]) for w, c in lexicon.items()}
    meta = dict(sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop=HOP, n_mels=N_MELS, mel_mean=mean, mel_std=std, silence_id=code_ids[SILENCE],
                code_ids=code_ids, n_phones=len(code_ids) + 1, voices=["Microsoft Zira Desktop", "Microsoft David Desktop"])
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    json.dump(lex, open(os.path.join(a.out, "lexicon.json"), "w"))


if __name__ == "__main__":
    main()
