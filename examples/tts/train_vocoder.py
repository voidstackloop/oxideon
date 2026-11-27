"""Trains `ox.models.NeuralVocoder` (GAN, HiFi-GAN / Vocos recipe) on tiny-voca, so the TTS voice stops sounding muffled.

    python examples/tts/prepare_voca.py --root tinyvoca --out voca/data --audio --workers 8
    python examples/tts/train_vocoder.py --data voca/data --tts tinytts_BEST.oxd --out voca/voc --device cuda --minutes 150

Input per frame: [normalised log-mel (80), normalised log-F0 interpolated through unvoiced frames, voicing 0/1]. Half of the
training segments use the frozen TinyTTS's own teacher-forced (GTA) mel instead of the real one, so the vocoder learns to turn
the TTS's smooth spectrogram into sharp speech. Losses: 45 x log-mel L1 + least-squares GAN + 2 x feature matching against a
multi-period (waveform) and a multi-resolution (spectrogram) discriminator; the GAN switches on after --gan-start steps.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

import oxideon as ox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_tts import Data  # noqa: E402
from synthesize_voca import load, vocoder_features, write_wav  # noqa: E402

F = ox.functional
Parameter, Module, ModuleList = ox.nn.Parameter, ox.nn.Module, ox.nn.ModuleList


def put(x, device):
    t = ox.from_numpy(np.ascontiguousarray(x))
    return t if device == "cpu" else t.to_device(device) if device.startswith("cuda") else t.to(device)


def stft_basis(n):
    nb = n // 2 + 1
    t, k = np.arange(n), np.arange(nb)[:, None]
    w = 0.5 - 0.5 * np.cos(2 * np.pi * t / n)
    ang = 2 * np.pi * k * t / n
    return np.concatenate([np.cos(ang) * w, -np.sin(ang) * w])[:, None, :].astype(np.float32)


class Spectra:
    """Differentiable magnitude STFT (a strided conv with a fixed DFT basis) and log-mel, cached per resolution."""

    def __init__(self, device, sr, n_mels=80):
        self.device, self.sr, self.n_mels, self.cache = device, sr, n_mels, {}

    def mag(self, x, n, hop):  # x [B, N] -> [B, nb, T]
        if n not in self.cache:
            self.cache[n] = put(stft_basis(n), self.device)
        B = int(x.shape[0])
        z = x.narrow(1, 0, 1) * 0.0
        xp = ox.Tensor.cat([z.expand([B, n // 2])] + [x] + [z.expand([B, n // 2])], 1).reshape([B, 1, -1])
        s = F.conv1d(xp, self.cache[n], stride=hop)
        nb = n // 2 + 1
        re, im = s.narrow(1, 0, nb), s.narrow(1, nb, nb)
        return (re * re + im * im + 1e-7).sqrt()

    def logmel(self, x, n=1024, hop=256):
        key = ("fb", n)
        if key not in self.cache:
            self.cache[key] = put(ox.audio.mel_filter_bank(n_mels=self.n_mels, n_fft=n, sample_rate=self.sr).astype(np.float32).T.copy(), self.device)
        return (self.mag(x, n, hop).permute([0, 2, 1]) @ self.cache[key]).clamp(1e-5, 1e9).log()


def _conv_stack(mod, chans, kernels, strides, pads):
    mod.cfg = []
    convs = []
    for (ci, co), k, s, p in zip(zip(chans[:-1], chans[1:]), kernels, strides, pads):
        fan = ci * k[0] * k[1]
        convs.append((Parameter(ox.Tensor.randn([co, ci, k[0], k[1]]) * (1.0 / math.sqrt(fan))), Parameter(ox.Tensor.zeros([co])), s, p))
    for i, (w, b, s, p) in enumerate(convs):
        setattr(mod, f"w{i}", w)
        setattr(mod, f"b{i}", b)
        mod.cfg.append((s, p))


def _run_stack(mod, x):
    feats = []
    n = len(mod.cfg)
    for i, (s, p) in enumerate(mod.cfg):
        x = F.conv2d(x, getattr(mod, f"w{i}"), getattr(mod, f"b{i}"), stride=s, padding=p)
        if i < n - 1:
            x = F.leaky_relu(x, 0.1)
        feats.append(x)
    return x, feats


class PeriodDisc(Module):
    def __init__(self, period, ch=(1, 32, 64, 128, 256, 256, 1)):
        super().__init__()
        self.period = period
        _conv_stack(self, list(ch), [(5, 1)] * 5 + [(3, 1)], [(3, 1)] * 4 + [(1, 1), (1, 1)], [(2, 0)] * 5 + [(1, 0)])

    def forward(self, x):  # [B, N]
        B, N = int(x.shape[0]), int(x.shape[1])
        p = self.period
        if N % p:
            x = ox.Tensor.cat([x, (x.narrow(1, 0, 1) * 0.0).expand([B, p - N % p])], 1)
        return _run_stack(self, x.reshape([B, 1, -1, p]))


class ResDisc(Module):
    def __init__(self, n_fft, hop, spectra, ch=(1, 32, 32, 32, 32, 32, 1)):
        super().__init__()
        self.n_fft, self.hop, self.spectra = n_fft, hop, spectra
        _conv_stack(self, list(ch), [(7, 5), (5, 3), (5, 3), (5, 3), (3, 3), (3, 3)], [(1, 1), (2, 1), (2, 1), (2, 1), (1, 1), (1, 1)],
                    [(3, 2), (2, 1), (2, 1), (2, 1), (1, 1), (1, 1)])

    def forward(self, x):
        m = (self.spectra.mag(x, self.n_fft, self.hop) + 1.0).log()  # [B, nb, T]
        return _run_stack(self, m.reshape([int(m.shape[0]), 1, int(m.shape[1]), int(m.shape[2])]))


class Discriminators(Module):
    def __init__(self, spectra):
        super().__init__()
        self.ds = ModuleList([PeriodDisc(p) for p in (2, 3, 5, 7, 11)] + [ResDisc(n, n // 4, spectra) for n in (512, 1024, 2048)])

    def forward(self, x):
        return [d(x) for d in self.ds]


class Segments:
    """Random fixed-length training segments: features (real or GTA mel + pitch) and the matching real waveform."""

    def __init__(self, data, audio, gta, meta, frames):
        self.d, self.wav, self.woff, self.gta, self.meta, self.L = data, audio["wav"], audio["wav_off"], gta, meta, frames
        self.ok = np.flatnonzero(data.frames >= frames + 2)

    def feats(self, i, a, L, use_gta):
        d = self.d
        lo = d.mel_off[i]
        mel = self.gta[lo + a:lo + a + L].astype(np.float32) if use_gta else (d.mel[lo + a:lo + a + L].astype(np.float32) - d.mean) / d.std
        mu, sd = self.meta["f0_stats"][str(int(d.voice[i]))]
        f0, conf = d.f0[lo:lo + int(d.frames[i])], d.conf[lo:lo + int(d.frames[i])]
        voiced = np.flatnonzero(conf > 0.8)
        lf = (np.log(np.maximum(f0, 1.0)) - mu) / sd
        lfi = np.interp(np.arange(len(f0)), voiced, lf[voiced]) if len(voiced) else np.zeros(len(f0))
        return vocoder_features(mel, lfi[a:a + L], (conf[a:a + L] > 0.7).astype(np.float32))

    def batch(self, rng, B, gta_prob):
        X, Y = [], []
        hop = self.meta["hop"]
        for i in rng.choice(self.ok, B):
            a = int(rng.integers(0, int(self.d.frames[i]) - self.L))
            X.append(self.feats(i, a, self.L, rng.random() < gta_prob))
            w0 = self.woff[i] + a * hop
            Y.append(self.wav[w0:w0 + (self.L - 1) * hop].astype(np.float32) / 32768.0)
        return np.stack(X), np.stack(Y)


def compute_gta(tts, data, device, path, max_frames=12000):
    """The frozen TinyTTS's teacher-forced normalised mel for every frame of `data` (cached at `path`)."""
    if os.path.exists(path):
        return np.load(path, mmap_mode="r")
    out = np.zeros(data.mel.shape, np.float16)
    with ox.no_grad():
        for idx in data.batches(np.random.default_rng(0), max_frames, shuffle=False):
            b = data.collate(idx, device)
            enc, _ = tts.encode(b["ph"], b["ws"], b["voice"], b["ph_mask"])
            mel = tts.decode(enc, b["A"], b["phase"], b["fmask"])[0].to("cpu").to_numpy()
            for k, i in enumerate(idx):
                t = int(data.frames[i])
                out[data.mel_off[i]:data.mel_off[i] + t] = mel[k, :t]
    np.save(path, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tts", required=True, help="frozen TinyTTS checkpoint whose own (GTA) mels the vocoder learns to clean up")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--minutes", type=float, default=150)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--frames", type=int, default=48, help="frames per training segment")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--gan-start", type=int, default=2000)
    ap.add_argument("--gta-prob", type=float, default=0.5)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--init", default=None, help="resume generator (and <out>/disc.oxd if present)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "samples"), exist_ok=True)
    meta = json.load(open(os.path.join(a.data, "meta.json")))
    sr = meta["sample_rate"]
    val_name = "val" if os.path.exists(os.path.join(a.data, "val.npz")) else "validation"
    train, val = Data(os.path.join(a.data, "train.npz"), meta), Data(os.path.join(a.data, val_name + ".npz"), meta)
    tts, _ = load(a.data, a.tts, a.device)
    t0 = time.time()
    gta_tr = compute_gta(tts, train, a.device, os.path.join(a.out, "gta_train.npy"))
    gta_va = compute_gta(tts, val, a.device, os.path.join(a.out, "gta_val.npy"))
    print(f"GTA mels ready in {time.time() - t0:.0f}s", flush=True)
    del tts
    seg_tr = Segments(train, np.load(os.path.join(a.data, "audio_train.npz")), gta_tr, meta, a.frames)
    seg_va = Segments(val, np.load(os.path.join(a.data, f"audio_{val_name}.npz")), gta_va, meta, a.frames)
    ox.manual_seed(a.seed)
    gen = ox.models.NeuralVocoder(n_in=82, dim=a.dim, layers=a.layers, n_fft=meta["n_fft"], hop=meta["hop"])
    spectra = Spectra(a.device, sr, meta["n_mels"])
    disc = Discriminators(spectra)
    if a.init:
        gen.load_file(a.init)
        if os.path.exists(os.path.join(a.out, "disc.oxd")):
            disc.load_file(os.path.join(a.out, "disc.oxd"))
    if a.device != "cpu":
        gen.to(device=a.device)
        disc.to(device=a.device)
    og = ox.optim.AdamW(list(gen.parameters()), lr=a.lr, betas=(0.8, 0.99), weight_decay=0.01)
    od = ox.optim.AdamW(list(disc.parameters()), lr=a.lr, betas=(0.8, 0.99), weight_decay=0.01)
    print(f"NeuralVocoder {gen.num_params() / 1e6:.2f} M, discriminators {sum(int(p.numel()) for p in disc.parameters()) / 1e6:.2f} M", flush=True)
    rng = np.random.default_rng(a.seed)
    vrng = np.random.default_rng(123)
    vX, vY = seg_va.batch(vrng, 32, 1.0)  # fixed validation segments, GTA input (what the vocoder sees at inference)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0, step, run, best = time.time(), 0, None, 1e9
    while time.time() - t0 < a.minutes * 60:
        frac = (time.time() - t0) / (a.minutes * 60)
        lr = a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for o in (og, od):
            for g in o.param_groups:
                g["lr"] = lr
        X, Y = seg_tr.batch(rng, a.batch, a.gta_prob)
        x, y = put(X, a.device), put(Y, a.device)
        yh = gen(x)
        gan = step >= a.gan_start
        ld = 0.0
        if gan:  # discriminator step on detached fakes
            od.zero_grad()
            loss_d = None
            for (r, _), (f, _) in zip(disc(y), disc(yh.detach())):
                l_ = ((r - 1.0) ** 2).mean() + (f ** 2).mean()
                loss_d = l_ if loss_d is None else loss_d + l_
            loss_d.backward()
            od.step()
            ld = float(loss_d.item())
        og.zero_grad()
        lmel = (spectra.logmel(yh) - spectra.logmel(y)).abs().mean()
        loss = lmel * 45.0
        la = lf = 0.0
        if gan:
            real = disc(y)
            fake = disc(yh)
            adv, fm = None, None
            for (r, rf), (f, ff) in zip(real, fake):
                l_ = ((f - 1.0) ** 2).mean()
                adv = l_ if adv is None else adv + l_
                for u, v in zip(rf, ff):
                    m_ = (u.detach() - v).abs().mean()
                    fm = m_ if fm is None else fm + m_
            loss = loss + adv + fm * 2.0
            la, lf = float(adv.item()), float(fm.item())
        loss.backward()
        ox.nn.utils.clip_grad_norm_(list(gen.parameters()), 100.0)
        og.step()
        step += 1
        v = (float(lmel.item()), la, lf, ld)
        run = v if run is None else tuple(0.95 * r + 0.05 * q for r, q in zip(run, v))
        if step % 100 == 0:
            rec = dict(step=step, mel_l1=round(run[0], 4), adv=round(run[1], 3), fm=round(run[2], 3), disc=round(run[3], 3), lr=lr,
                       minutes=round((time.time() - t0) / 60, 2), steps_per_s=round(step / (time.time() - t0), 2))
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if step % a.eval_every == 0 or time.time() - t0 >= a.minutes * 60:
            with ox.no_grad():
                vm = float(np.mean([float((spectra.logmel(gen(put(vX[k:k + 8], a.device))) - spectra.logmel(put(vY[k:k + 8], a.device))).abs().mean().item())
                                    for k in range(0, len(vX), 8)]))
                i = int(seg_va.ok[0])
                full = seg_va.feats(i, 0, int(val.frames[i]), True)
                wav = gen(put(full[None], a.device)).to("cpu").to_numpy()[0]
            write_wav(os.path.join(a.out, "samples", f"step{step:06d}.wav"), wav, sr)
            gen.save_file(os.path.join(a.out, "vocoder.oxd"))
            disc.save_file(os.path.join(a.out, "disc.oxd"))
            tag = ""
            if vm < best and gan:  # before the GAN starts, a low mel loss is not what we want to keep
                best = vm
                gen.save_file(os.path.join(a.out, "vocoder_best.oxd"))
                tag = " (best)"
            print(json.dumps(dict(step=step, val_mel_l1=round(vm, 4))) + tag, flush=True)
    print("DONE", json.dumps(dict(steps=step, best_val_mel_l1=round(best, 4), minutes=round((time.time() - t0) / 60, 1))), flush=True)


if __name__ == "__main__":
    main()
