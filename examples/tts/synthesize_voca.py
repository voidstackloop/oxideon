"""Speaks text with a TinyTTS trained on the tiny-voca data (`prepare_voca.py` + `train_tts.py`): text -> phoneme tokens (a
lexicon learned from the dataset's own aligned labels) -> durations, log-mel, pitch and voicing -> the Rust source-filter
vocoder (`ox.audio.harmonic_synth`). `--compare` also writes the Griffin-Lim rendering of the same mel for an A/B listen.

    python examples/tts/synthesize_voca.py --data runs/voca/data --model runs/voca/model/tinytts.oxd \
        --annotations runs/tinyvoca/annotations/v4 --voice 0 --text "Hello there." --out hello.wav --compare
"""
import argparse
import collections
import json
import os
import re
import sys
import wave

import numpy as np

import oxideon as ox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthesize import _put, build_alignment  # noqa: E402

BOS, EOS, SP = 1, 2, 3
PUNCT_IDS = {",": 5, ".": 6, "?": 7, "!": 8, ";": 9, ":": 10}  # the phonemizer's punctuation tokens
PUNCT_SET = set(PUNCT_IDS.values()) | {11, 12}


def build_lexicon(ann_dir, cache):
    """word -> (tokens, stress): the most common pronunciation in the aligned training labels."""
    if os.path.exists(cache):
        return {w: (v[0], v[1]) for w, v in json.load(open(cache)).items()}
    seen = collections.defaultdict(collections.Counter)
    with open(os.path.join(ann_dir, "train.jsonl"), encoding="utf8") as f:
        for line in f:
            r = json.loads(line)
            words = r["text"].split()
            tok, st = r["tokens"][1:-1], r["stress"][1:-1]  # drop <bos> / <eos>
            segs, cur = [], ([], [])
            for t, s in zip(tok, st):
                if t == SP:
                    segs.append(cur), (cur := ([], []))
                else:
                    cur[0].append(t), cur[1].append(s)
            segs.append(cur)
            if len(segs) != len(words):
                continue
            for w, (t, s) in zip(words, segs):
                w = re.sub(r"^[^a-z']+|[^a-z']+$", "", w.lower())
                keep = [i for i, x in enumerate(t) if x not in PUNCT_SET]
                if w and keep and re.fullmatch(r"[a-z']+", w):
                    seen[w][(tuple(t[i] for i in keep), tuple(s[i] for i in keep))] += 1
    lex = {w: [list(c.most_common(1)[0][0][0]), list(c.most_common(1)[0][0][1])] for w, c in seen.items()}
    json.dump(lex, open(cache, "w"))
    return {w: (v[0], v[1]) for w, v in lex.items()}


def phonemize(text, lex):
    """`(tokens, stress, unknown_words)`: <bos>, the words joined by <sp> (each followed by its punctuation token), <eos>."""
    toks, stress, unknown = [BOS], [0], []

    def lookup(word):
        if word in lex:
            return lex[word]
        t, s, i = [], [], 0  # greedy longest known pieces
        while i < len(word):
            for j in range(len(word), i + 1, -1):
                if word[i:j] in lex:
                    t += lex[word[i:j]][0]
                    s += lex[word[i:j]][1]
                    i = j
                    break
            else:
                i += 1
        if not t:
            unknown.append(word)
        return t, s

    words = re.findall(r"[A-Za-z']+[,.;:?!]*|[,.;:?!]+", text)
    first = True
    for w in words:
        raw = re.match(r"[A-Za-z']*", w).group(0)
        body = raw.strip("'").lower()  # quotes around a word ('madame') are not part of it; inner apostrophes (don't) stay
        trail = [PUNCT_IDS[c] for c in w[len(raw):] if c in PUNCT_IDS][:1]
        t, s = lookup(body) if body else ([], [])
        if not t and not trail:
            continue
        if not first:
            toks.append(SP), stress.append(0)
        toks += list(t) + trail
        stress += list(s) + [0] * len(trail)
        first = False
    toks.append(EOS), stress.append(0)
    return toks, stress, unknown


def write_wav(path, wav, sr):
    wav = wav / max(1e-6, float(np.abs(wav).max())) * 0.9
    with wave.open(path, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(sr)
        w.writeframes((np.clip(wav, -1, 1) * 32767).astype(np.int16).tobytes())


def load(data_dir, model_path, device="cpu", prosody_layers=0):
    """A TinyTTS checkpoint, its shape read from the weights; `prosody_layers` adds a fresh (identity) acoustic stage to a
    checkpoint that has none."""
    meta = json.load(open(os.path.join(data_dir, "meta.json")))
    sd = ox.load_file(model_path)
    k = int(sd["decoder.0.cv_w"].shape[0]) if "decoder.0.cv_w" in sd else 0  # checkpoints trained with --conv-kernel
    n_blocks = lambda pre: len({key.split(".")[1] for key in sd if key.startswith(pre + ".")})
    post = n_blocks("post")
    model = ox.models.TinyTTS(meta["n_phones"], n_mels=meta["n_mels"], n_flags=meta["n_flags"], n_aux=2, conv_kernel=k, variance="pitch_emb" in sd,
                              dim=int(sd["ph_embed"].shape[1]), enc_layers=n_blocks("encoder"), dec_layers=n_blocks("decoder"),
                              prosody_layers=post or prosody_layers)
    model.load_state_dict(sd, strict=bool(post) or not prosody_layers)
    if device != "cpu":
        model.to(device=device)
    model.eval()
    return model, meta


MEL_FLOW_SCALE = 2.5  # a SpecFlow models (true - predicted) normalised log-mel times this (the residual's std is ~0.4)


def load_pitchflow(path, device="cpu"):
    sd = ox.load_file(path)
    cond_dim, dim = (int(n) for n in sd["cond_w"].shape)
    layers = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
    flow = ox.models.PitchFlow(cond_dim=cond_dim, dim=dim, layers=layers, conv_kernel=int(sd["blocks.0.cv_w"].shape[0]), channels=int(sd["in_w"].shape[0]))
    flow.load_state_dict(sd)
    if device != "cpu":
        flow.to(device=device)
    flow.eval()
    return flow


def flow_pitch(flow, hidden, frame_mask, temperature=0.7, steps=16, seed=0, device="cpu"):
    """Samples `(normalised log-F0, voicing 0..1)` per frame from a PitchFlow, given the TinyTTS decoder states."""
    T = int(hidden.shape[1])
    noise = np.random.default_rng(seed).standard_normal((1, T, flow.channels)).astype(np.float32) * temperature
    with ox.no_grad():
        x = flow.sample(hidden, frame_mask, _put(ox.from_numpy(noise), device), steps).to("cpu").to_numpy()[0]
    return x[:, 0], np.clip(0.5 + x[:, 1], 0, 1)


def load_head(path, device="cpu", jitter=0.0, shimmer=0.0, sharpen=1.0):
    """`sharpen` > 1 scales the harmonic envelope's cepstral coefficients 1.. (a formant post-filter: deeper peaks and valleys
    against the over-smoothed spectrum a regression-trained model predicts; 1.25 matches the real clips' spectral contrast)."""
    sd = ox.load_file(path)
    if sharpen != 1.0:
        cc = ox.models.SourceFilterHead().cc
        s = np.ones(int(sd["h_b"].shape[0]), np.float32)
        s[1:cc] = sharpen
        sd["h_w"] = ox.from_numpy(np.ascontiguousarray(sd["h_w"].to_numpy() * s))
        sd["h_b"] = ox.from_numpy(np.ascontiguousarray(sd["h_b"].to_numpy() * s))
    head = ox.models.SourceFilterHead(dim=int(sd["h_w"].shape[0]))
    head.load_state_dict(sd)
    head.jitter, head.shimmer = jitter, shimmer
    if device != "cpu":
        head.to(device=device)
    return head


def load_vocoder(path, device="cpu"):
    sd = ox.load_file(path)
    n_in, dim = (int(n) for n in sd["in_w"].shape)
    layers = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
    voc = ox.models.NeuralVocoder(n_in=n_in, dim=dim, layers=layers, kernel=int(sd["blocks.0.dw"].shape[0]))
    voc.load_state_dict(sd)
    if device != "cpu":
        voc.to(device=device)
    voc.eval()
    return voc


def vocoder_features(mel_norm, lf0_norm, voiced):
    """NeuralVocoder input `[T, 82]`: normalised log-mel, normalised log-F0 (interpolated through unvoiced frames), voicing 0..1."""
    return np.concatenate([np.asarray(mel_norm, np.float32), np.asarray(lf0_norm, np.float32)[:, None],
                           np.asarray(voiced, np.float32)[:, None]], 1)


def median(x, k=5):
    p = np.pad(x, (k // 2, k // 2), mode="edge")
    return np.median(np.stack([p[i:i + len(x)] for i in range(k)]), axis=0)


def synthesize(model, meta, lex, text, voice=0, speed=1.0, device="cpu", compare=False, gl_iters=60, pitch_shift=0.0, phonemes=None, frontend=None, pitch_range=1.0,
               pitchflow=None, temperature=0.7, flow_steps=16, seed=0, specflow=None, spec_temperature=0.7, eq=None, vocoder=None, head=None, pitch_jitter=0.0, voiced_vowels=False):
    """`phonemes=(tokens, stress)` (e.g. the dataset's own labels, with <bos>/<eos>) skips the text front end; `frontend` is a
    `frontend_cmu.CmuFrontEnd` (CMUdict based) used instead of the label lexicon. `pitchflow` (from `load_pitchflow`) samples the
    melody and voicing instead of the regression head (which is flat); `temperature` scales its noise."""
    if phonemes is not None:
        toks, stress, unknown = list(phonemes[0]), list(phonemes[1]), []
    else:
        toks, stress, unknown = frontend.phonemize(text) if frontend else phonemize(text, lex)
    to = lambda a: _put(ox.from_numpy(np.ascontiguousarray(a)), device)
    ph, ws = to(np.array([toks], np.int64)), to(np.array([stress], np.int64))
    vo, pm = to(np.array([voice], np.int64)), to(np.ones((1, len(toks)), bool))
    with ox.no_grad():
        enc, log_dur = model.encode(ph, ws, vo, pm)
        if getattr(model, "variance", False):  # predicted per-phoneme pitch / energy, quantised, feed the decoder
            q = model.var_bins(model.predict_variance(enc).to("cpu").to_numpy())
            enc = model.add_variance(enc, to(np.ascontiguousarray(q[..., 0])), to(np.ascontiguousarray(q[..., 1])))
        frames = np.maximum(np.rint(np.expm1(log_dur.to("cpu").to_numpy()[0]) / speed), 0).astype(np.int64)
        real = ~np.isin(np.array(toks), [BOS, EOS, SP] + list(PUNCT_SET))
        frames[real] = np.maximum(frames[real], 1)
        A, phase, fm = build_alignment(frames, device)
        hidden = model.decode_hidden(enc, A, phase, fm)
        mel, aux = (t.to("cpu").to_numpy()[0] for t in model.outputs(hidden))
        if specflow is not None:  # sampled spectral detail on top of the (smooth) predicted mel
            T = int(A.shape[1])
            noise = np.random.default_rng(seed + 1).standard_normal((1, T, specflow.channels)).astype(np.float32) * spec_temperature
            r = specflow.sample(model.decode_hidden(enc, A, phase, fm), fm, _put(ox.from_numpy(noise), device), flow_steps)
            mel = mel + r.to("cpu").to_numpy()[0] / MEL_FLOW_SCALE
        if pitchflow is not None:
            lf_flow, vo_flow = flow_pitch(pitchflow, model.decode_hidden(enc, A, phase, fm), fm, temperature, flow_steps, seed, device)
    sr, n_fft, hop, nm = meta["sample_rate"], meta["n_fft"], meta["hop"], meta["n_mels"]
    logmel = mel * meta["mel_std"] + meta["mel_mean"]
    mu, sd = meta["f0_stats"][str(voice)]
    p = 1 / (1 + np.exp(-aux[:, 1]))
    lf = median(aux[:, 0])  # normalised log-F0; a model trained on average error is too flat, so `pitch_range` > 1 widens the
    centre = float((lf * p).sum() / max(p.sum(), 1e-6))  # melody around the voiced frames' mean (1.0 = as predicted)
    f0 = np.exp((centre + (lf - centre) * pitch_range) * sd + mu) * 2 ** (pitch_shift / 12)
    voiced = np.clip((median(p, 3) - 0.35) / 0.3, 0, 1)
    if pitchflow is not None:
        lf, voiced = median(lf_flow, 3), median(vo_flow, 3)
        centre = float((lf * voiced).sum() / max(voiced.sum(), 1e-6))
        f0 = np.exp((centre + (lf - centre) * pitch_range) * sd + mu) * 2 ** (pitch_shift / 12)
    if voiced_vowels:  # tiny-voca: a vowel is voiced by definition (models trained with train_e2e.py --voiced-vowels)
        vowel = np.isin(np.array(toks), np.arange(43, 76))[np.repeat(np.arange(len(toks)), frames)][:len(voiced)]
        voiced = np.where(vowel, np.maximum(voiced, 1.0), voiced)
    if pitch_jitter > 0:  # natural micro-variation: a predicted melody is smoother than a real voice (frame-to-frame wobble ~20 %
        r = np.random.default_rng(seed + 7).standard_normal(len(f0) + 2)  # too low); lightly smoothed noise, in semitones
        f0 = f0 * 2 ** (np.convolve(r, [0.25, 0.5, 0.25], "valid") / 0.61 * pitch_jitter / 12)
    if head is not None:  # end-to-end SourceFilterHead (train_e2e.py): the decoder states drive the waveform head directly
        on = np.flatnonzero(voiced > 0.5)
        f0i = np.clip(np.interp(np.arange(len(f0)), on, f0[on]) if len(on) else f0, 50.0, 600.0)
        with ox.no_grad():
            if getattr(model, "prosody_layers", 0):  # acoustic stage, conditioned on the very pitch / voicing the head will use
                en = model.energy(hidden).to("cpu").to_numpy()[0]
                cond = np.stack([np.clip((np.log(f0i) - mu) / sd, -3, 3), (voiced > 0.5).astype(np.float32), np.clip(en, -3, 3)], -1)
                hidden = model.acoustic(hidden, to(cond[None].astype(np.float32)), fm)
            wav = head(hidden, f0i[None], [len(f0i)], noise=np.random.default_rng(seed).standard_normal((1, len(f0i) * hop)), phase=np.zeros(1))
        wav = wav.to("cpu").to_numpy()[0]
    elif vocoder is not None:  # neural vocoder (train_vocoder.py) instead of the harmonic one: mel + the same pitch / voicing
        with ox.no_grad():
            feats = vocoder_features(mel, (np.log(f0) - mu) / sd, voiced)
            wav = vocoder(to(feats[None])).to("cpu").to_numpy()[0]
        return {"wav": wav, "unknown": unknown}
    else:
        env = ox.audio.mel_envelope(logmel, nm, n_fft, sr)
        if eq is not None:  # fit_vocoder_eq.py: restores the 1-8 kHz energy the harmonic vocoder loses (it sounds muffled without)
            env = env * eq[None, :]
        wav = ox.audio.harmonic_synth(f0.astype(np.float32), voiced.astype(np.float32), env, sr, n_fft, hop).to_numpy()
    out = {"wav": wav, "unknown": unknown}
    if compare:
        fb = ox.audio.mel_filter_bank(n_mels=nm, n_fft=n_fft, sample_rate=sr).astype(np.float32)
        mag = np.maximum(np.linalg.pinv(fb) @ np.exp(logmel).T, 0).astype(np.float32)
        out["griffinlim"] = ox.audio.griffin_lim(ox.from_numpy(np.ascontiguousarray(mag)), n_fft=n_fft, hop_length=hop, n_iter=gl_iters, seed=0).to_numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="speech.wav")
    ap.add_argument("--voice", type=int, default=0, help="0 = F2 (female), 1 = M2 (male)")
    ap.add_argument("--speed", type=float, default=0.91, help="speaking rate (<1 = slower). 0.91 makes the median duration equal the real recordings (the model alone speaks ~9 %% fast)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pitch-range", type=float, default=1.0, help="widen (>1) or narrow the predicted melody around its mean; the model alone is ~19 %% too flat")
    ap.add_argument("--pitchflow", default=None, help="a PitchFlow checkpoint (train_pitchflow.py): sampled, livelier melody")
    ap.add_argument("--temperature", type=float, default=0.7, help="PitchFlow noise scale (0 = its most typical melody)")
    ap.add_argument("--seed", type=int, default=0, help="PitchFlow noise seed (another seed = another intonation)")
    ap.add_argument("--specflow", default=None, help="a SpecFlow checkpoint (train_pitchflow.py --target mel): sharper spectral detail")
    ap.add_argument("--spec-temperature", type=float, default=0.7, help="SpecFlow noise scale. Measured: it HURTS intelligibility (male WER 2.87%% -> 3.69%% at 0.7, 11%% at 1.0)")
    ap.add_argument("--vocoder", default=None, help="a NeuralVocoder checkpoint (train_vocoder.py) instead of the harmonic vocoder")
    ap.add_argument("--eq", default=None, help="vocoder correction filter (.npy) from fit_vocoder_eq.py: less muffled")
    ap.add_argument("--pitch-jitter", type=float, default=0.0, help="random pitch micro-variation, semitones (e.g. 0.1); a predicted melody is too smooth")
    ap.add_argument("--sf-head", default=None, help="a SourceFilterHead (train_e2e.py; pair it with that run's --model): waveform straight from the decoder")
    ap.add_argument("--jitter", type=float, default=0.0, help="--sf-head: per-cycle relative period deviation of the voice source (less buzzy)")
    ap.add_argument("--shimmer", type=float, default=0.0, help="--sf-head: per-cycle relative amplitude deviation of the voice source")
    ap.add_argument("--voiced-vowels", action="store_true", help="keep vowel frames voiced (models trained with train_e2e.py --voiced-vowels)")
    ap.add_argument("--sharpen", type=float, default=1.0, help="--sf-head: formant post-filter strength (1 = off, 1.25 = real clips' spectral contrast)")
    ap.add_argument("--compare", action="store_true", help="also write <out>_griffinlim.wav")
    ap.add_argument("--frontend", default=None, help="frontend.json from learn_frontend.py: CMUdict based text front end (needs --cmudict); much better on new words")
    ap.add_argument("--cmudict", default=None, help="cmudict.dict (CMU Pronouncing Dictionary)")
    ap.add_argument("--pron", action="append", default=[], help="WORD=ARPABET override, e.g. oxideon=\"AA1 K S AH0 D IY0 AA2 N\" (repeatable)")
    a = ap.parse_args()
    model, meta = load(a.data, a.model, a.device)
    lex = build_lexicon(a.annotations, os.path.join(a.data, "lexicon.json"))
    fe = None
    if a.frontend:
        from frontend_cmu import CmuFrontEnd, load_cmudict
        fe = CmuFrontEnd.load(a.frontend, load_cmudict(a.cmudict), lex)
        for item in a.pron:
            fe.add_pronunciation(*item.split("=", 1))
    r = synthesize(model, meta, lex, a.text, a.voice, a.speed, a.device, a.compare, frontend=fe, pitch_range=a.pitch_range,
                   pitchflow=load_pitchflow(a.pitchflow, a.device) if a.pitchflow else None, temperature=a.temperature, seed=a.seed,
                   specflow=load_pitchflow(a.specflow, a.device) if a.specflow else None, spec_temperature=a.spec_temperature,
                   eq=np.load(a.eq) if a.eq else None, vocoder=load_vocoder(a.vocoder, a.device) if a.vocoder else None,
                   head=load_head(a.sf_head, a.device, a.jitter, a.shimmer, a.sharpen) if a.sf_head else None, pitch_jitter=a.pitch_jitter, voiced_vowels=a.voiced_vowels)
    write_wav(a.out, r["wav"], meta["sample_rate"])
    if a.compare:
        write_wav(a.out.replace(".wav", "_griffinlim.wav"), r["griffinlim"], meta["sample_rate"])
    print(f"wrote {a.out}: {len(r['wav']) / meta['sample_rate']:.2f} s" + (f"; no pronunciation for {r['unknown']}" if r["unknown"] else ""))


if __name__ == "__main__":
    main()
