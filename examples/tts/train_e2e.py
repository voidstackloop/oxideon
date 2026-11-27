"""End-to-end waveform training of TinyTTS + `ox.models.SourceFilterHead` (the tiny-voca recipe): the decoder states drive a
differentiable harmonic-plus-noise head, and the loss is taken on the waveform itself, not on a mel spectrogram.

    python examples/tts/prepare_voca.py --root tinyvoca --out voca/data --audio --workers 8
    python examples/tts/train_e2e.py --data voca/data --tts tinytts_BEST.oxd --out voca/e2e --device cuda --minutes 240

Losses (tiny-voca weights): 2 x multi-resolution STFT (log-magnitude L1 + spectral convergence, 512/1024/2048) + 10 x log-mel L1 of
the waveform + 2 x the TinyTTS mel head + duration / pitch / voicing. From --gan-start: least-squares GAN + 30 x feature matching
on 8192-sample crops, scaled so that inside the crops its waveform gradient is 0.7 x the reconstruction gradient (ramped in).
In training the oscillator runs at the true pitch (random start phase); at inference at the predicted pitch.
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
from train_vocoder import Discriminators, Spectra, put  # noqa: E402
from synthesize_voca import load, write_wav  # noqa: E402

RES = ((512, 128), (1024, 256), (2048, 512))


def true_f0(d, i, t, sd=None):
    """Frame pitch in Hz, interpolated (log domain) through unvoiced frames, clipped to 50..600."""
    lo = d.mel_off[i]
    f0, conf = d.f0[lo:lo + t].astype(np.float64), d.conf[lo:lo + t]
    v = np.flatnonzero(conf > 0.8)
    if len(v) == 0:
        return np.full(t, 120.0)
    lf = np.interp(np.arange(t), v, np.log(np.maximum(f0[v], 1.0)))
    return np.clip(np.exp(lf), 50.0, 600.0)


def wave_batch(d, audio, idx, hop, meta=None):
    """Real waveform `[B, T * hop]`, true pitch in Hz `[B, T]`, frame counts and (with `meta`) the true frame prosody `[B, T, 3]`
    (normalised log-F0 interpolated through unvoiced frames, voiced 0/1, energy = mean normalised log-mel) for the acoustic stage."""
    T = max(int(d.frames[i]) for i in idx)
    y, f0 = np.zeros((len(idx), T * hop), np.float32), np.full((len(idx), T), 120.0)
    cond = np.zeros((len(idx), T, 3), np.float32)
    for b, i in enumerate(idx):
        t = int(d.frames[i])
        w0 = audio["wav_off"][i]
        y[b, :t * hop] = audio["wav"][w0:w0 + t * hop].astype(np.float32) / 32768.0
        f0[b, :t] = true_f0(d, i, t)
        if meta is not None:
            mu, sd = meta["f0_stats"][str(int(d.voice[i]))]
            lo = d.mel_off[i]
            cond[b, :t, 0] = np.clip((np.log(f0[b, :t]) - mu) / sd, -3.0, 3.0)
            cond[b, :t, 1] = (d.conf[lo:lo + t] > 0.7) | d.vowel_frames(i, t)
            cond[b, :t, 2] = np.clip(((d.mel[lo:lo + t].astype(np.float32) - d.mean) / d.std).mean(1), -3.0, 3.0)
    return (y, f0, d.frames[idx].astype(np.int64)) + ((cond,) if meta is not None else ())


def spectral(sp, p, t, lens, hop_frames, dev):
    """(multi-resolution STFT loss, log-mel L1), frames past an utterance's end masked."""
    total = None
    B = int(p.shape[0])
    for n, hop in RES:
        P, Tm = sp.mag(p, n, hop), sp.mag(t, n, hop)
        S, nb = int(P.shape[2]), int(P.shape[1])
        m = put((np.arange(S)[None, :] * hop < lens[:, None] * hop_frames).astype(np.float32)[:, None, :], dev)
        lg = ((P.clamp(1e-5, 1e9).log() - Tm.clamp(1e-5, 1e9).log()).abs() * m).sum() / (float(m.sum().item()) * nb)
        num = (((P - Tm) ** 2) * m).reshape([B, -1]).sum(1)
        den = ((Tm * Tm) * m).reshape([B, -1]).sum(1)
        sc = ((num + 1e-9).sqrt() / den.sqrt().clamp(1.0, 1e12)).mean()
        total = lg + sc if total is None else total + lg + sc
    P, Tm = sp.logmel(p), sp.logmel(t)  # [B, S, 80]
    S = int(P.shape[1])
    m = put((np.arange(S)[None, :] * 256 < lens[:, None] * hop_frames).astype(np.float32)[:, :, None], dev)
    mel = ((P - Tm).abs() * m).sum() / (float(m.sum().item()) * int(P.shape[2]))
    return total * (1.0 / len(RES)), mel


def crop_index(lens, hop, row_len, size, count, max_crops, rng):
    """Flat sample indices into a [B, row_len] batch for `count` random windows on each of at most `max_crops // count` random
    utterances (the critics' memory grows with the number of windows), and the window size."""
    rows = np.sort(rng.choice(len(lens), min(len(lens), max(1, max_crops // count)), replace=False))
    n = lens[rows] * hop
    size = int(min(size, n.min()))
    start = (rng.random((len(rows), count)) * (n[:, None] - size + 1)).astype(np.int64)
    return (rows[:, None, None] * row_len + start[..., None] + np.arange(size)).reshape(-1), size


def take_crops(w, flat, size, dev):  # w [B, N], flat sample indices -> [crops, size]
    return w.reshape([-1]).index_select(0, flat).reshape([-1, size])


def forward(tts, head, b, f0, lens, phase, true_cond=None, teacher=None):
    """Waveform and the TinyTTS outputs. With an acoustic stage (`prosody_layers`), utterances flagged in `teacher` are conditioned
    on `true_cond`, the rest on the model's own (detached, hardened) prosody prediction, as at inference."""
    enc, log_dur = tts.encode(b["ph"], b["ws"], b["voice"], b["ph_mask"])
    vpred = None
    if tts.variance:
        vpred = tts.predict_variance(enc)
        enc = tts.add_variance(enc, b["pbin"], b["ebin"])
    x = tts.decode_hidden(enc, b["A"], b["phase"], b["fmask"])
    mel, aux = tts.outputs(x)
    energy = None
    if tts.prosody_layers:
        energy = tts.energy(x)
        cond = tts.hard_prosody(aux.detach().to("cpu").to_numpy(), energy.detach().to("cpu").to_numpy())
        if true_cond is not None and teacher is not None:
            cond = np.where(np.asarray(teacher)[:, None, None], true_cond, cond).astype(np.float32)
        x = tts.acoustic(x, _put_like(cond, x), b["fmask"])
    return head(x, f0, lens, phase=phase), mel, aux, log_dur, vpred, energy


def _put_like(a, like):
    t = ox.from_numpy(np.ascontiguousarray(a))
    return t if str(like.device) == "cpu" else t.to(like.device)


def aux_losses(b, mel, aux, log_dur, vpred):
    B, T = int(b["fmask"].shape[0]), int(b["fmask"].shape[1])
    fm = b["fmask"].to_dtype("f32").reshape([B, T, 1])
    pm = b["ph_mask"].to_dtype("f32")
    l1 = ((mel - b["mel"]).abs() * fm).sum() / (fm.sum() * float(mel.shape[-1]))
    ld = (((log_dur - b["dur"]) ** 2) * pm).sum() / pm.sum()
    if vpred is not None:
        ld = ld + (((vpred - b["pv"]) ** 2) * pm.reshape([B, -1, 1])).sum() / (2.0 * pm.sum())
    f0p, vl = aux[:, :, 0], aux[:, :, 1]
    vm = b["vmask"].to_dtype("f32")
    lf = ((f0p - b["f0"]).abs() * vm).sum() / vm.sum().clamp(1.0, 1e12)
    y, fm2 = b["vtgt"], fm.reshape([B, T])
    lv = ((vl.clamp(0.0, 1e9) - vl * y + (1.0 + (-vl.abs()).exp()).log()) * fm2).sum() / fm2.sum()
    # pitch spread (tiny-voca): each utterance's predicted melody must vary as much as the real one; an L1 head alone is flat
    n = vm.sum(1).clamp(1.0, 1e12)
    def spread(v):
        mu = (v * vm).sum(1) / n
        return ((((v - mu.reshape([B, 1])) ** 2) * vm).sum(1) / n + 1e-6).sqrt()
    ok = (vm.sum(1) > 2.0).to_dtype("f32")
    ls = ((spread(f0p) - spread(b["f0"])).abs() * ok).sum() / ok.sum().clamp(1.0, 1e12)
    return l1, ld, lf, lv, ls


def gan_terms(disc, real, fake):
    adv, fm, n = None, None, 0
    for (_, rf), (f, ff) in zip(real, fake):
        l_ = ((f - 1.0) ** 2).mean()
        adv = l_ if adv is None else adv + l_
        for u, v in zip(rf, ff):
            m_ = (u.detach() - v).abs().mean()
            fm = m_ if fm is None else fm + m_
            n += 1
    return adv * (1.0 / len(fake)), fm * (1.0 / n)


def predicted_f0(aux, voice, meta, t):
    """Inference pitch from the TinyTTS aux head: normalised log-F0 -> Hz, interpolated through predicted-unvoiced frames."""
    a = aux.to("cpu").to_numpy()[0, :t]
    mu, sd = meta["f0_stats"][str(int(voice))]
    hz = np.exp(a[:, 0] * sd + mu)
    v = np.flatnonzero(a[:, 1] > 0)
    return np.clip(np.interp(np.arange(t), v, hz[v]) if len(v) else hz, 50.0, 600.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tts", default=None, help="TinyTTS checkpoint to start from (or --scratch; both = resume a scratch run)")
    ap.add_argument("--scratch", default=None, help="start from random weights with this many 'encoder,frame,acoustic' blocks; "
                    "tiny-voca's layout is 4,2,5 (2 frame blocks predict pitch / voicing / energy, 5 more blocks predict the filters)")
    ap.add_argument("--head", default=None, help="resume a SourceFilterHead (and <out>/disc.oxd if present)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--minutes", type=float, default=240)
    ap.add_argument("--max-frames", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tts-lr-scale", type=float, default=0.3, help="the warm-started TinyTTS moves slower than the new head")
    ap.add_argument("--prosody-layers", type=int, default=0, help="add a tiny-voca acoustic stage of this many blocks conditioned on "
                    "frame [pitch, voiced, energy] (new parameters train at the full rate)")
    ap.add_argument("--teacher-hold", type=int, default=2000, help="steps where the acoustic stage sees only the recorded prosody")
    ap.add_argument("--teacher-ramp", type=int, default=5000, help="steps over which that share falls to 0 (then: own predictions)")
    ap.add_argument("--spread-weight", type=float, default=0.0, help="per-utterance pitch-spread loss weight (tiny-voca: 2)")
    ap.add_argument("--half-life", type=int, default=0, help="tiny-voca schedule: lr halves every this many steps after --warmup "
                    "(0 = cosine over --minutes); independent of the run length, so a run cut short is just a shorter one")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--start-step", type=int, default=0, help="resume: continue the step count (GAN / teacher / lr schedules) "
                    "from here; with --scratch, --tts loads that run's weights and --head (+ <out>/disc.oxd) the rest")
    ap.add_argument("--gan-start", type=int, default=3000)
    ap.add_argument("--gan-ramp", type=int, default=3000)
    ap.add_argument("--max-crops", type=int, default=16, help="discriminator windows per step (8192 samples each)")
    ap.add_argument("--crop-chunk", type=int, default=16, help="windows per critic pass (memory); results do not depend on it")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--gan-weight", type=float, default=0.7, help="adversarial gradient norm as a fraction of the reconstruction one (inside the crops)")
    ap.add_argument("--jitter", type=float, default=0.0, help="per-cycle relative period deviation of the head's voice source (e.g. 0.01)")
    ap.add_argument("--shimmer", type=float, default=0.0, help="per-cycle relative amplitude deviation of the head's voice source (e.g. 0.04)")
    ap.add_argument("--keep-evals", action="store_true", help="also keep every evaluation's checkpoint as tts_e2e_s<step>.oxd / head_s<step>.oxd")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--voiced-vowels", action="store_true", help="tiny-voca: vowel frames are voiced by definition (voicing target "
                    "and condition 1), so the model never learns to whisper a vowel the pitch tracker missed")
    ap.add_argument("--profile", type=int, default=0, help="time N steps after 3 warm-up steps (per phase, and per CUDA kernel when "
                    "OXIDEON_CUDA_PROFILE=1), print the table and exit")
    a = ap.parse_args()
    phases, last, peaks = {}, [0.0], {}

    def tick(name):  # --profile: wall time since the previous tick, the GPU drained first
        if a.profile and a.profile + 3 > step >= 3:
            if dev != "cpu":
                ox.cuda_synchronize()
                if hasattr(ox, "cuda_memory_stats"):  # the most this phase held at once, and what it left behind
                    live, peak, _ = ox.cuda_memory_stats(True)
                    pk = peaks.get(name, (0, 0))
                    peaks[name] = (max(pk[0], peak), max(pk[1], live))
            now = time.perf_counter()
            phases[name] = phases.get(name, 0.0) + now - last[0]
            last[0] = now
    os.makedirs(os.path.join(a.out, "samples"), exist_ok=True)
    meta = json.load(open(os.path.join(a.data, "meta.json")))
    sr, hop, dev = meta["sample_rate"], meta["hop"], a.device
    vn = "val" if os.path.exists(os.path.join(a.data, "val.npz")) else "validation"
    train, val = Data(os.path.join(a.data, "train.npz"), meta), Data(os.path.join(a.data, vn + ".npz"), meta)
    au_tr, au_va = np.load(os.path.join(a.data, "audio_train.npz")), np.load(os.path.join(a.data, f"audio_{vn}.npz"))
    au_tr = {k: au_tr[k] for k in ("wav", "wav_off")}
    if a.voiced_vowels:
        train.voiced_ids = val.voiced_ids = np.arange(43, 76)  # ə .. əl, n̩ in the tiny-voca v4 token table
    au_va = {k: au_va[k] for k in ("wav", "wav_off")}
    if a.scratch:
        enc_l, dec_l, post_l = (int(v) for v in a.scratch.split(","))
        tts = ox.models.TinyTTS(meta["n_phones"], n_mels=meta["n_mels"], n_flags=meta.get("n_flags", 2), n_aux=2, conv_kernel=7,
                                enc_layers=enc_l, dec_layers=dec_l, prosody_layers=post_l)
        if a.tts:  # resume a run that started from scratch (same layout)
            tts.load_file(a.tts)
        if dev != "cpu":
            tts.to(device=dev)
        a.prosody_layers = post_l
    else:
        tts, _ = load(a.data, a.tts, dev, prosody_layers=a.prosody_layers)
    tts.train()
    ox.manual_seed(a.seed)
    head = ox.models.SourceFilterHead(dim=tts.dim, sample_rate=sr, hop=hop)
    head.jitter, head.shimmer = a.jitter, a.shimmer
    sp = Spectra(dev, sr, meta["n_mels"])
    disc = Discriminators(sp)
    if a.head:
        head.load_file(a.head)
        if os.path.exists(os.path.join(a.out, "disc.oxd")):
            disc.load_file(os.path.join(a.out, "disc.oxd"))
    if dev != "cpu":
        head.to(device=dev)
        disc.to(device=dev)
    new = list(tts.post.parameters()) + [tts.energy_w, tts.energy_b, tts.pros_w, tts.pros_b] if tts.prosody_layers else []
    old = [p for p in tts.parameters() if all(p is not q for q in new)]
    opt = ox.optim.AdamW([{"params": old, "lr": a.lr * a.tts_lr_scale}, {"params": list(head.parameters()) + new, "lr": a.lr}],
                         lr=a.lr, betas=(0.8, 0.99), weight_decay=0.01)
    od = ox.optim.AdamW(list(disc.parameters()), lr=a.lr, betas=(0.8, 0.99), weight_decay=0.0)
    params = list(tts.parameters()) + list(head.parameters())
    print(f"TinyTTS {tts.num_params() / 1e6:.2f} M + SourceFilterHead {head.num_params()} params; discriminators "
          f"{sum(int(p.numel()) for p in disc.parameters()) / 1e6:.2f} M", flush=True)
    rng = np.random.default_rng(a.seed)
    vidx = [list(x) for x in val.batches(np.random.default_rng(0), a.max_frames, shuffle=False)][:6]
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0, step, run, best = time.time(), a.start_step, None, 1e9
    done = False
    while not done:
        for idx in train.batches(rng, a.max_frames):
            frac = (time.time() - t0) / (a.minutes * 60)
            if frac >= 1.0:
                done = True
                break
            decay = 0.5 ** (step / a.half_life) if a.half_life else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac))
            sched = min(1.0, (step + 1) / a.warmup) * decay
            opt.param_groups[0]["lr"], opt.param_groups[1]["lr"] = a.lr * a.tts_lr_scale * sched, a.lr * sched
            for g in od.param_groups:
                g["lr"] = a.lr * sched
            tick("(between steps)")
            b = train.collate(idx, dev)
            Y, F0, lens, TC = wave_batch(train, au_tr, idx, hop, meta)
            y = put(Y, dev)
            tick("data")
            p_teacher = 1.0 if step < a.teacher_hold else max(0.0, 1.0 - (step - a.teacher_hold) / a.teacher_ramp)
            teacher = rng.random(len(idx)) < p_teacher
            wave, mel, aux, log_dur, vpred, energy = forward(tts, head, b, F0, lens, rng.uniform(0, 2 * np.pi, len(idx)), TC, teacher)
            tick("forward (model + head)")
            l1, ld, lf, lv, ls = aux_losses(b, mel, aux, log_dur, vpred)
            loss_aux = l1 * 2.0 + ld + lf * 2.0 + lv + ls * a.spread_weight
            le = 0.0
            if energy is not None:
                fm2 = b["fmask"].to_dtype("f32")
                le_t = ((energy - _put_like(TC[..., 2], energy)).abs() * fm2).sum() / fm2.sum()
                loss_aux = loss_aux + le_t
                le = float(le_t.item())
            # reconstruction gradient at the waveform. Every backward below is the only one through its graph, so each
            # frees it as it goes (retain_graph=False): activations no longer live on into the next step's forward
            wl = wave.detach().requires_grad_(True)
            stft, wmel = spectral(sp, wl, y, lens, hop, dev)
            (stft * 2.0 + wmel * 10.0).backward(retain_graph=False)
            G = wl.grad
            rec_g = G
            tick("aux losses + spectral loss fwd/bwd")
            gan = step >= a.gan_start
            ld_ = la = lfm = scale = 0.0
            if gan:
                cflat, size = crop_index(lens, hop, int(wave.shape[1]), 8192, 2, a.max_crops, rng)
                cidx = put(cflat, dev)
                # the critics see the windows in chunks of --crop-chunk, each chunk's loss weighted by its share of the windows and
                # its gradients accumulated: the same sums as one pass, with memory that no longer grows with --max-crops
                ncrop = len(cflat) // size
                chunks = [(c0, min(ncrop, c0 + a.crop_chunk)) for c0 in range(0, ncrop, a.crop_chunk)]
                cflat_np = np.asarray(cflat)
                cid = [put(cflat_np[c0 * size:c1 * size], dev) for c0, c1 in chunks]
                od.zero_grad()
                ld_ = 0.0
                wd = wave.detach()
                for (c0, c1), ci in zip(chunks, cid):
                    loss_d = None
                    for (r, _), (f, _) in zip(disc(take_crops(y, ci, size, dev)), disc(take_crops(wd, ci, size, dev))):
                        l_ = ((r - 1.0) ** 2).mean() + (f ** 2).mean()
                        loss_d = l_ if loss_d is None else loss_d + l_
                    loss_d = loss_d * ((c1 - c0) / ncrop / len(disc.ds))
                    loss_d.backward(retain_graph=False)
                    ld_ += float(loss_d.item())
                ox.nn.utils.clip_grad_norm_(list(disc.parameters()), 10.0)
                od.step()
                od.zero_grad()
                tick("critic update")
                w2 = wave.detach().requires_grad_(True)
                la = lfm = 0.0
                for (c0, c1), ci in zip(chunks, cid):
                    with ox.no_grad():
                        real = disc(take_crops(y, ci, size, dev))
                    adv, fm = gan_terms(disc, real, disc(take_crops(w2, ci, size, dev)))
                    wgt = (c1 - c0) / ncrop
                    ((adv + fm * 30.0) * wgt).backward(retain_graph=False)  # w2.grad accumulates over the chunks
                    la, lfm = la + wgt * float(adv.item()), lfm + wgt * float(fm.item())
                Gg = w2.grad
                od.zero_grad()  # the generator pass must not leave gradients on the critics
                flat = cidx
                nr = float((rec_g.reshape([-1]).index_select(0, flat) ** 2).sum().sqrt().item())
                ng = float((Gg.reshape([-1]).index_select(0, flat) ** 2).sum().sqrt().item())
                ramp = min(1.0, (step - a.gan_start + 1) / a.gan_ramp)
                scale = min(a.gan_weight * ramp * nr / max(ng, 1e-12), 2000.0)
                G = rec_g + Gg * scale
                tick("generator GAN grads + scale")
            opt.zero_grad()
            (loss_aux + (wave * G.detach()).sum()).backward(retain_graph=False)
            tick("model backward")
            ox.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            tick("clip + optimizer")
            step += 1
            if a.profile and step == 3:
                if dev != "cpu":
                    ox.cuda_synchronize()
                ox.cuda_kernel_profile()  # drop the warm-up launches
                last[0] = time.perf_counter()
            if a.profile and step == a.profile + 3:
                tot = sum(phases.values())
                print(f"PROFILE {a.profile} steps, {tot / a.profile:.3f} s/step, {int(sum(train.frames[i] for i in idx))} frames in the last batch")
                for k, v in sorted(phases.items(), key=lambda kv: -kv[1]):
                    pk = peaks.get(k, (0, 0))
                    print(f"  {k:40s} {v / a.profile * 1000:9.1f} ms/step {100 * v / tot:5.1f} %   peak {pk[0] / 2**30:6.2f} GiB, "
                          f"left {pk[1] / 2**30:6.2f} GiB")
                kp = ox.cuda_kernel_profile()
                if kp:
                    ks = sum(x[2] for x in kp)
                    print(f"KERNELS: {ks / a.profile:.3f} s/step of GPU time over {sum(x[1] for x in kp) // a.profile} launches/step")
                    for name, n, sec in kp[:30]:
                        print(f"  {name:28s} {n // a.profile:7d} launches/step {sec / a.profile * 1000:9.1f} ms/step {100 * sec / ks:5.1f} %")
                return
            v = (float(stft.item()), float(wmel.item()), float(l1.item()), float(lf.item()), la, lfm, ld_, scale, float(ls.item()), le, p_teacher)
            run = v if run is None else tuple(0.95 * r + 0.05 * q for r, q in zip(run, v))
            if step % 50 == 0:
                rec = dict(step=step, stft=round(run[0], 4), wave_mel=round(run[1], 4), mel_head=round(run[2], 4), f0=round(run[3], 4),
                           adv=round(run[4], 3), fm=round(run[5], 3), disc=round(run[6], 3), gan_scale=round(run[7], 2), spread=round(run[8], 4), energy=round(run[9], 4), teacher=round(run[10], 2),
                           minutes=round((time.time() - t0) / 60, 2), steps_per_s=round((step - a.start_step) / (time.time() - t0), 3))
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
            if step % a.eval_every == 0:
                tts.eval()
                vm = []
                with ox.no_grad():
                    for vi in vidx:
                        vb = val.collate(vi, dev)
                        VY, VF, vl = wave_batch(val, au_va, vi, hop)
                        w = forward(tts, head, vb, VF, vl, np.zeros(len(vi)))[0]  # own prosody (as at inference)
                        vm.append(float(spectral(sp, w, put(VY, dev), vl, hop, dev)[1].item()))
                    for k, vi in enumerate((vidx[0][0], vidx[-1][0])):
                        vb = val.collate([vi], dev)
                        t = int(val.frames[vi])
                        aux_v = forward(tts, head, vb, np.full((1, t), 120.0), np.array([t]), np.zeros(1))[2]
                        wv = forward(tts, head, vb, predicted_f0(aux_v, val.voice[vi], meta, t)[None], np.array([t]), np.zeros(1))[0]
                        write_wav(os.path.join(a.out, "samples", f"step{step:06d}_{k}_v{int(val.voice[vi])}.wav"), wv.to("cpu").to_numpy()[0], sr)
                tts.train()
                vmv = float(np.mean(vm))
                tts.save_file(os.path.join(a.out, "tts_e2e.oxd"))
                head.save_file(os.path.join(a.out, "head.oxd"))
                disc.save_file(os.path.join(a.out, "disc.oxd"))
                if a.keep_evals:
                    tts.save_file(os.path.join(a.out, f"tts_e2e_s{step}.oxd"))
                    head.save_file(os.path.join(a.out, f"head_s{step}.oxd"))
                tag = ""
                if vmv < best:
                    best = vmv
                    tts.save_file(os.path.join(a.out, "tts_e2e_best.oxd"))
                    head.save_file(os.path.join(a.out, "head_best.oxd"))
                    tag = " (best)"
                print(json.dumps(dict(step=step, val_wave_mel=round(vmv, 4))) + tag, flush=True)
    tts.save_file(os.path.join(a.out, "tts_e2e.oxd"))
    head.save_file(os.path.join(a.out, "head.oxd"))
    print("DONE", json.dumps(dict(steps=step, best_val_wave_mel=round(best, 4), minutes=round((time.time() - t0) / 60, 1))), flush=True)


if __name__ == "__main__":
    main()
