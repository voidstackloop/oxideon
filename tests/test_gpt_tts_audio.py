"""GPT, TinyTTS, Rust Griffin-Lim and `Tensor.to(device)` placement (CPU only, small and fast)."""
import numpy as np

import oxideon as ox


def _ids(rng, b, t, v):
    return ox.from_numpy(rng.integers(0, v, (b, t)).astype(np.int64))


def test_gpt_shapes_params_and_causality():
    rng = np.random.default_rng(0)
    m = ox.models.GPT(vocab_size=50, max_seq_len=16, embed_dim=32, num_layers=2, num_heads=4)
    # tied embeddings: tok 50*32 + pos 16*32 + 2 blocks + final LayerNorm
    assert m.num_params() == sum(int(np.prod(p.shape)) for p in m.parameters())
    x = _ids(rng, 3, 10, 50)
    base = m(x).to_numpy()
    assert base.shape == (3, 10, 50)
    x2 = x.to_numpy().copy()
    x2[:, 7:] = (x2[:, 7:] + 1) % 50  # change only the future
    again = m(ox.from_numpy(x2)).to_numpy()
    np.testing.assert_allclose(base[:, :7], again[:, :7], atol=1e-5)  # earlier positions cannot see it
    assert np.abs(base[:, 7:] - again[:, 7:]).max() > 1e-4


def test_gpt_loss_matches_cross_entropy_and_trains():
    rng = np.random.default_rng(1)
    ox.manual_seed(0)
    m = ox.models.GPT(vocab_size=20, max_seq_len=12, embed_dim=32, num_layers=1, num_heads=2)
    x = _ids(rng, 4, 9, 20)
    idx, tgt = x[:, :-1], x[:, 1:]
    logits = m(idx).to_numpy().reshape(-1, 20)
    lp = logits - np.log(np.exp(logits - logits.max(-1, keepdims=True)).sum(-1, keepdims=True)) - logits.max(-1, keepdims=True)
    want = -lp[np.arange(len(lp)), tgt.to_numpy().reshape(-1)].mean()
    np.testing.assert_allclose(float(m.loss(idx, tgt).item()), want, rtol=1e-4)
    opt = ox.optim.AdamW(list(m.parameters()), lr=1e-2)
    first = float(m.loss(idx, tgt).item())
    for _ in range(30):
        opt.zero_grad()
        m.loss(idx, tgt).backward()
        opt.step()
    assert float(m.loss(idx, tgt).item()) < first * 0.5  # it can memorise one batch


def test_tinytts_forward_shapes_and_padding_is_ignored():
    n_ph, n, t = 12, 5, 9
    m = ox.models.TinyTTS(n_ph, n_mels=8, dim=32, heads=4, enc_layers=1, dec_layers=1)
    assert m.num_params() == sum(int(np.prod(p.shape)) for p in m.parameters())
    rng = np.random.default_rng(2)
    ph = rng.integers(1, n_ph, (2, n)).astype(np.int64)
    ws = rng.integers(0, 2, (2, n)).astype(np.int64)
    A = np.zeros((2, t, n), np.float32)
    A[:, np.arange(t), np.minimum(np.arange(t) // 2, n - 1)] = 1.0
    phase = rng.random((2, t, 2)).astype(np.float32)
    pm = np.ones((2, n), bool)
    fm = np.ones((2, t), bool)
    T = lambda a: ox.from_numpy(np.ascontiguousarray(a))
    mel, log_dur = m(T(ph), T(ws), T(np.array([0, 1], np.int64)), T(pm), T(A), T(phase), T(fm))
    assert mel.shape == [2, t, 8] and log_dur.shape == [2, n]
    # a padded phoneme / frame (mask False) must not change the real outputs
    ph2, pm2 = ph.copy(), pm.copy()
    ph2[0, -1], pm2[0, -1] = 3, False
    fm2 = fm.copy()
    fm2[0, -1] = False
    A2 = A.copy()
    A2[0, :, -1] = 0.0
    a = m.encode(T(ph), T(ws), T(np.array([0, 1], np.int64)), T(pm2))[1].to_numpy()
    b = m.encode(T(ph2), T(ws), T(np.array([0, 1], np.int64)), T(pm2))[1].to_numpy()
    np.testing.assert_allclose(a[0, :-1], b[0, :-1], atol=1e-5)


def test_tinytts_conv_branch_starts_as_a_no_op_and_ignores_padding():
    n_ph, n, t = 12, 4, 10
    kw = dict(n_mels=8, dim=32, heads=4, enc_layers=1, dec_layers=2)
    ox.manual_seed(3)
    base, conv = ox.models.TinyTTS(n_ph, **kw), ox.models.TinyTTS(n_ph, conv_kernel=5, **kw)
    conv.load_state_dict(base.state_dict(), strict=False)  # an older checkpoint has no conv weights
    rng = np.random.default_rng(4)
    T = lambda a: ox.from_numpy(np.ascontiguousarray(a))
    enc = T(rng.standard_normal((1, n, 32)).astype(np.float32))
    A = np.zeros((1, t, n), np.float32)
    A[0, np.arange(t), np.arange(t) * n // t] = 1.0
    phase = T(rng.random((1, t, 2)).astype(np.float32))
    fm = np.ones((1, t), bool)
    np.testing.assert_allclose(conv.decode(enc, T(A), phase, T(fm)).to_numpy(), base.decode(enc, T(A), phase, T(fm)).to_numpy(), atol=1e-6)
    for blk in conv.decoder:  # once trained (non-zero pointwise weight), a padded tail frame must not change the real frames
        blk.cv_pw = ox.nn.Parameter(ox.from_numpy(rng.standard_normal((32, 32)).astype(np.float32) * 0.1))
    fm2 = fm.copy()
    fm2[0, -1] = False
    a = conv.decode(enc, T(A), phase, T(fm2)).to_numpy()
    A3 = A.copy()
    A3[0, -1] = 0.0
    A3[0, -1, 0] = 5.0  # different content in the padded frame
    b = conv.decode(enc, T(A3), phase, T(fm2)).to_numpy()
    np.testing.assert_allclose(a[0, :-1], b[0, :-1], atol=1e-5)
    assert conv.num_params() > base.num_params()


def test_faceflow_shapes_gradients_and_padding():
    ox.manual_seed(0)
    m = ox.models.FaceFlow(vocab_size=30, channels=(8, 16, 16, 16), attn_levels=(2, 3), text_dim=16, text_layers=1, text_heads=2, size=16)
    rng = np.random.default_rng(0)
    T = lambda a: ox.from_numpy(np.ascontiguousarray(a))
    x0, nz = (T(rng.standard_normal((2, 3, 16, 16)).astype(np.float32)) for _ in range(2))
    t = T(np.array([0.3, 0.8], np.float32))
    tok = rng.integers(2, 30, (2, 6)).astype(np.int64)
    msk = np.ones((2, 6), bool)
    msk[:, -2:] = False
    loss = m.loss(x0, nz, t, T(tok), T(msk))
    loss.backward()
    assert np.isfinite(float(loss.item())) and all(p.grad is not None for p in m.parameters())
    a = m(x0, t, T(tok), T(msk)).to_numpy()
    tok[:, -2:] = 9  # padded positions cannot change the output
    np.testing.assert_allclose(a, m(x0, t, T(tok), T(msk)).to_numpy(), atol=1e-5)
    assert m.sample(T(tok[:1]), T(msk[:1]), steps=2).shape == [1, 3, 16, 16]


def test_cmu_frontend_layout_numbers_and_overrides():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "tts"))
    from frontend_cmu import CmuFrontEnd, normalize, number_words

    assert [number_words(n) for n in (0, 21, 100, 1200, 2005)] == ["zero", "twenty one", "one hundred", "one thousand two hundred", "two thousand five"]
    assert normalize("It cost $2.50 and 3.5%.") == "It cost two dollars and fifty cents and three point five percent."
    rules = {"tables": [{}] * 5, "merge": {}, "stress": {}, "function_words": []}
    lex = {"hi": ([31, 60], [1, 0]), "there": ([26, 70], [1, 0])}
    fe = CmuFrontEnd(rules, {}, lex, {})
    t = lambda s: fe.phonemize(s)[0]
    assert t("Hi there") == [1, 31, 60, 3, 26, 70, 2]
    assert t("Hi, there!") == [1, 31, 60, 5, 26, 70, 8, 2]  # no <sp> after punctuation
    assert t('"Hi there."') == [1, 11, 31, 60, 3, 26, 70, 6, 11, 2]  # quote tokens, no <sp> beside them
    assert t("hi-there") == [1, 31, 60, 26, 70, 2]  # a hyphenated compound is one word
    assert fe.phonemize("zzz")[2] == ["zzz"]  # nothing known: reported, not guessed


def test_griffin_lim_recovers_a_tone():
    sr, n_fft, hop = 8000, 256, 64
    x = np.sin(2 * np.pi * 440 * np.arange(sr // 2) / sr).astype(np.float32)
    spec = ox.audio.Spectrogram(n_fft=n_fft, hop_length=hop, power=1.0)
    mag = spec(ox.from_numpy(x)).to_numpy()
    y = ox.audio.griffin_lim(ox.from_numpy(np.ascontiguousarray(mag)), n_fft=n_fft, hop_length=hop, n_iter=60, seed=0).to_numpy()
    mag2 = spec(ox.from_numpy(np.ascontiguousarray(y.astype(np.float32)))).to_numpy()
    f = min(mag.shape[1], mag2.shape[1])
    assert np.linalg.norm(mag[:, :f] - mag2[:, :f]) / np.linalg.norm(mag[:, :f]) < 0.1  # spectral convergence
    # deterministic for a fixed seed
    y2 = ox.audio.griffin_lim(ox.from_numpy(np.ascontiguousarray(mag)), n_fft=n_fft, hop_length=hop, n_iter=60, seed=0).to_numpy()
    np.testing.assert_array_equal(y, y2)


def test_to_with_a_device_handle_keeps_the_exact_device():
    a = ox.from_numpy(np.ones((2, 3), np.float32))
    assert str(a.to(a.device).device) == "cpu"
    try:
        g = a.to("gpu")
    except Exception:
        return  # no GPU backend on this machine
    assert str(g.to(g.device).device) == str(g.device)
    assert str(a.to(g.device).device) == str(g.device)
    assert str(a.to(g).device) == str(g.device)
    assert str(g.to(a.device).device) == "cpu"


def test_pitch_track_finds_f0_and_silence_is_unvoiced():
    sr, hop, f = 24000, 256, 187.0
    n = np.arange(sr)
    x = sum(np.sin(2 * np.pi * f * h * n / sr) / h for h in range(1, 6)).astype(np.float32)
    f0, conf = (t.to_numpy() for t in ox.audio.pitch_track(x, sr, hop, 60.0, 500.0))
    assert len(f0) == len(x) // hop + 1
    mid = slice(10, len(f0) - 10)
    assert abs(float(np.median(f0[mid])) - f) < 2.0 and float(conf[mid].min()) > 0.8
    _, c0 = ox.audio.pitch_track(np.zeros(4096, np.float32), sr, hop)
    assert float(c0.to_numpy().max()) == 0.0


def test_harmonic_synth_follows_the_pitch_and_is_deterministic():
    sr, n_fft, hop, frames = 24000, 1024, 256, 60
    f0 = np.full(frames, 210.0, np.float32)
    voiced = np.ones(frames, np.float32)
    env = np.tile((1.0 / (1.0 + np.arange(n_fft // 2 + 1) / 20.0)).astype(np.float32), (frames, 1))
    y = ox.audio.harmonic_synth(f0, voiced, env, sr, n_fft, hop, breath=0.0).to_numpy()
    assert y.shape == ((frames - 1) * hop,) and np.isfinite(y).all() and np.abs(y).max() > 0
    p, c = (t.to_numpy() for t in ox.audio.pitch_track(y, sr, hop, 60.0, 500.0))
    assert abs(float(np.median(p[10:-10])) - 210.0) < 3.0 and float(c[10:-10].min()) > 0.8
    np.testing.assert_array_equal(y, ox.audio.harmonic_synth(f0, voiced, env, sr, n_fft, hop, breath=0.0).to_numpy())
    # an unvoiced frame range is noise only: no harmonic peak, so low pitch confidence
    u = ox.audio.harmonic_synth(f0, np.zeros(frames, np.float32), env, sr, n_fft, hop).to_numpy()
    assert float(ox.audio.pitch_track(u, sr, hop)[1].to_numpy()[10:-10].mean()) < 0.6


def test_mel_envelope_returns_a_flat_spectrum():
    sr, n_fft, n_mels = 24000, 1024, 80
    fb = ox.audio.mel_filter_bank(n_mels=n_mels, n_fft=n_fft, sample_rate=sr).astype(np.float64)
    mel = np.log(fb @ np.full(n_fft // 2 + 1, 3.0))[None, :]  # the mel of a flat magnitude-3 spectrum
    env = ox.audio.mel_envelope(mel, n_mels, n_fft, sr)
    covered = fb.sum(0) > 1e-3
    np.testing.assert_allclose(env[0, covered], 3.0, rtol=1e-3)


def test_pitchflow_ignores_padding_and_learns_a_fixed_target():
    rng = np.random.default_rng(3)
    ox.manual_seed(0)
    f = ox.models.PitchFlow(cond_dim=16, dim=32, layers=1, heads=2, conv_kernel=3)
    B, T = 2, 7
    cond = ox.from_numpy(rng.standard_normal((B, T, 16)).astype(np.float32))
    fm = np.ones((B, T), bool)
    fm[1, 5:] = False
    xt = rng.standard_normal((B, T, 2)).astype(np.float32)
    tf = ox.from_numpy(f.time_features(np.array([0.2, 0.7]), 32))
    f.out_w.copy_(ox.from_numpy(rng.standard_normal((32, 2)).astype(np.float32)))  # zero-init output would hide leaks
    base = f(ox.from_numpy(xt), tf, cond, ox.from_numpy(fm)).to_numpy()
    xt2 = xt.copy()
    xt2[1, 5:] += 10.0  # padded frames must not change the real ones
    again = f(ox.from_numpy(xt2), tf, cond, ox.from_numpy(fm)).to_numpy()
    np.testing.assert_allclose(base[1, :5], again[1, :5], atol=1e-4)
    # flow matching to a single fixed target: the sampler must land near it
    x1 = np.tile(np.array([0.5, -1.0], np.float32), (1, T, 1))
    c1, m1 = ox.from_numpy(np.zeros((1, T, 16), np.float32)), ox.from_numpy(np.ones((1, T), bool))
    opt = ox.optim.Adam(list(f.parameters()), lr=3e-3)
    for _ in range(800):
        x0 = rng.standard_normal((1, T, 2)).astype(np.float32)
        t = rng.random(1).astype(np.float32)
        v = f(ox.from_numpy((1 - t) * x0 + t * x1), ox.from_numpy(f.time_features(t, 32)), c1, m1)
        opt.zero_grad()
        ((v - ox.from_numpy(x1 - x0)) ** 2).mean().backward()
        opt.step()
    with ox.no_grad():
        s = f.sample(c1, m1, ox.from_numpy(rng.standard_normal((1, T, 2)).astype(np.float32)), steps=16).to_numpy()
    assert np.abs(s - x1).mean() < 0.15, np.abs(s - x1).mean()


def test_neural_vocoder_inverse_stft_is_exact_and_trains():
    n, h = 1024, 256
    rng = np.random.default_rng(0)
    x = rng.standard_normal(20 * h).astype(np.float32)
    xp = np.pad(x, n // 2)
    T = len(x) // h + 1
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
    S = np.stack([np.fft.rfft(xp[i * h:i * h + n] * w) for i in range(T)], 1)
    v = ox.models.NeuralVocoder(n_in=4, dim=16, layers=1)
    y = v._inverse(ox.from_numpy(S.real[None].astype(np.float32)), ox.from_numpy(S.imag[None].astype(np.float32))).to_numpy()[0]
    np.testing.assert_allclose(y, x[:len(y)], atol=1e-4)  # centred Hann STFT -> exact signal
    out = v(ox.from_numpy(rng.standard_normal((2, 6, 4)).astype(np.float32)))
    assert tuple(out.shape) == (2, 5 * h)
    out.abs().mean().backward()
    assert v.head_w.grad is not None and v.blocks[0].dw.grad is not None


def test_source_filter_head_flat_filters_padding_and_gradients():
    from oxideon.models import SourceFilterHead, pulse_train
    rng = np.random.default_rng(0)
    h = SourceFilterHead(dim=8)
    x = rng.standard_normal((2, 4096)).astype(np.float32)
    re, im = h.stft(ox.from_numpy(x), 512, 128)
    np.testing.assert_allclose(h.istft(re, im, 512, 128, 4096).to_numpy(), x, atol=1e-5)
    p = pulse_train(np.full((1, 40), 150.0))
    assert abs(float((p ** 2).mean()) - 1.0) < 0.02  # unit power
    T, f0, lens = 20, np.full((2, 20), 140.0), np.array([20, 14])
    noise = rng.standard_normal((2, T * 256))
    w = h(ox.from_numpy(np.zeros((2, T, 8), np.float32)), f0, lens, noise=noise, phase=np.zeros(2)).to_numpy()
    # zero weights = flat filters at the lifted bias gains exp(2 * -1.5) and exp(2 * -2.5)
    want = np.exp(-3.0) * pulse_train(f0, phase=np.zeros(2)) + np.exp(-5.0) * noise
    np.testing.assert_allclose(w[0, 600:4000], want[0, 600:4000], atol=1e-4)
    assert np.abs(w[1, 14 * 256:]).max() == 0.0  # padding is silent
    hid = ox.nn.Parameter(ox.from_numpy(rng.standard_normal((2, T, 8)).astype(np.float32) * 0.1))
    (h(hid, f0, lens, noise=noise) ** 2).mean().backward()
    assert h.h_w.grad is not None and h.n_w.grad is not None and hid.grad is not None


def test_tinytts_acoustic_stage_starts_as_identity_and_learns():
    rng = np.random.default_rng(0)
    m = ox.models.TinyTTS(10, n_mels=8, dim=16, heads=2, enc_layers=1, dec_layers=1, n_aux=2, prosody_layers=1)
    x = ox.nn.Parameter(ox.from_numpy(rng.standard_normal((2, 5, 16)).astype(np.float32)))
    cond = ox.from_numpy(rng.standard_normal((2, 5, 3)).astype(np.float32))
    fm = ox.from_numpy(np.ones((2, 5), bool))
    np.testing.assert_allclose(m.acoustic(x, cond, fm).to_numpy(), x.to_numpy(), atol=1e-6)  # warm start is unchanged
    (m.acoustic(x, cond, fm) ** 2).mean().backward()
    assert m.pros_w.grad is not None and float(np.abs(m.pros_w.grad.to_numpy()).sum()) > 0
    hp = ox.models.TinyTTS.hard_prosody(np.array([[[5.0, 0.3], [-1.0, -2.0]]], np.float32), np.array([[0.5, -9.0]], np.float32))
    np.testing.assert_allclose(hp, [[[3.0, 1.0, 0.5], [-1.0, 0.0, -3.0]]])
    assert tuple(m.energy(x).shape) == (2, 5)
