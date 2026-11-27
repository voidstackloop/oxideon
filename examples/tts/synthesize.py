"""Speaks text with a trained TinyTTS: text -> phonemes (lexicon) -> durations + log-mel frames -> Griffin-Lim -> a 16 kHz wave file.

    python examples/tts/synthesize.py --data runs/tts/data --model runs/tts/model/tinytts.oxd --voice 0 --text "Hello there." --out hello.wav
"""
import argparse
import json
import os
import re
import wave

import numpy as np

import oxideon as ox

PUNCT = ",.;:?!"


def _put(t, device):
    """'cuda' is the native CUDA backend (`to_device`), 'gpu' the wgpu one (`to`)."""
    return t if device == "cpu" else t.to_device(device) if device.startswith("cuda") else t.to(device)


def phonemize(text, lexicon, silence, pieces=2):
    """`(ids, word_start)` for `text`: the lexicon pronunciation of every word, a silence token after punctuation and at both
    ends. A word that is not in the lexicon is split greedily into the longest known pieces."""
    ids, starts = [silence], [0]
    unknown = []

    def lookup(word):
        if word in lexicon:
            return lexicon[word]
        out, i = [], 0
        while i < len(word):
            for j in range(len(word), i + pieces - 1, -1):
                if word[i:j] in lexicon:
                    out += lexicon[word[i:j]]
                    i = j
                    break
            else:
                i += 1
        if not out:
            unknown.append(word)
        return out

    for tok in re.findall(r"[a-z']+|[,.;:?!]", text.lower()):
        if tok in PUNCT:
            if ids[-1] != silence:
                ids.append(silence), starts.append(0)
            continue
        ph = lookup(tok)
        if ph:
            ids += ph
            starts += [1] + [0] * (len(ph) - 1)
    if ids[-1] != silence:
        ids.append(silence), starts.append(0)
    return ids, starts, unknown


def build_alignment(frames, device):
    n = len(frames)
    rows = np.repeat(np.arange(n), frames)
    T = len(rows)
    A = np.zeros((1, T, n), np.float32)
    A[0, np.arange(T), rows] = 1.0
    start = np.concatenate([[0], np.cumsum(frames)])[:-1]
    length = np.maximum(np.bincount(rows, minlength=n)[rows], 1)
    phase = np.zeros((1, T, 2), np.float32)
    phase[0, :, 0], phase[0, :, 1] = (np.arange(T) - start[rows]) / length, 1.0 / length
    to = lambda x: _put(ox.from_numpy(np.ascontiguousarray(x)), device)
    return to(A), to(phase), to(np.ones((1, T), bool))


def synthesize(model, meta, lexicon, text, voice=0, speed=1.0, device="cpu", gl_iters=100, return_mel=False):
    ids, starts, unknown = phonemize(text, lexicon, meta["silence_id"])
    to = lambda x, dt=None: _put(ox.from_numpy(np.ascontiguousarray(x if dt is None else x.astype(dt))), device)
    ph, ws = to(np.array([ids], np.int64)), to(np.array([starts], np.int64))
    vo, pm = to(np.array([voice], np.int64)), to(np.ones((1, len(ids)), bool))
    with ox.no_grad():
        enc, log_dur = model.encode(ph, ws, vo, pm)
        frames = np.maximum(np.rint((np.expm1(log_dur.to("cpu").to_numpy()[0]) / speed)), 0).astype(np.int64)
        frames[np.array(ids) != meta["silence_id"]] = np.maximum(frames[np.array(ids) != meta["silence_id"]], 1)
        A, phase, fm = build_alignment(frames, device)
        mel = model.decode(enc, A, phase, fm).to("cpu").to_numpy()[0]
    mel = np.exp(mel * meta["mel_std"] + meta["mel_mean"])  # [frames, mels] magnitudes
    fb = ox.audio.mel_filter_bank(n_mels=meta["n_mels"], n_fft=meta["n_fft"], sample_rate=meta["sample_rate"]).astype(np.float32)
    mag = np.maximum(np.linalg.pinv(fb) @ mel.T, 0.0).astype(np.float32)
    wav = ox.audio.griffin_lim(ox.from_numpy(np.ascontiguousarray(mag)), n_fft=meta["n_fft"], hop_length=meta["hop"], n_iter=gl_iters, seed=0).to_numpy()
    wav = wav / max(1e-6, float(np.abs(wav).max())) * 0.9
    return (wav, mel, unknown) if return_mel else (wav, unknown)


def write_wav(path, wav, sr):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(wav, -1, 1) * 32767).astype(np.int16).tobytes())


def load(data_dir, model_path, device="cpu"):
    meta = json.load(open(os.path.join(data_dir, "meta.json")))
    lex_path = os.path.join(data_dir, "lexicon_big.json")
    lexicon = json.load(open(lex_path if os.path.exists(lex_path) else os.path.join(data_dir, "lexicon.json")))
    model = ox.models.TinyTTS(meta["n_phones"], n_mels=meta["n_mels"])
    model.load_file(model_path)
    if device != "cpu":
        model.to(device=device)
    model.eval()
    return model, meta, lexicon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="speech.wav")
    ap.add_argument("--voice", type=int, default=0, help="0 = female, 1 = male")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    model, meta, lexicon = load(a.data, a.model, a.device)
    wav, unknown = synthesize(model, meta, lexicon, a.text, a.voice, a.speed, a.device)
    write_wav(a.out, wav, meta["sample_rate"])
    print(f"wrote {a.out}: {len(wav) / meta['sample_rate']:.2f} s" + (f"; no pronunciation for {unknown}" if unknown else ""))


if __name__ == "__main__":
    main()
