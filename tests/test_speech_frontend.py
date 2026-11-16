"""Tests for the speech front-end (oxideon.speech)."""
import numpy as np
import pytest

import oxideon as ox
from oxideon import (
    SpeakerDiarizer,
    apply_mask,
    audio_fingerprint,
    energy_vad,
    fingerprint_matches,
    frame_energies_db,
    ideal_ratio_mask,
    spectral_subtraction,
    wiener_filter,
)

SR = 8000


def tone(freq, seconds, amp=0.5, sr=SR, seed=None):
    rng = np.random.default_rng(seed) if seed is not None else None
    t = np.arange(int(seconds * sr)) / sr
    wave = amp * np.sin(2 * np.pi * freq * t)
    if rng is not None:
        wave = wave + 0.05 * rng.normal(size=wave.shape)
    return wave


def snr_db(clean, estimate):
    clean = np.asarray(clean, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)[: clean.shape[0]]
    return 10 * np.log10(np.mean(clean**2) / max(np.mean((clean - est) ** 2), 1e-12))


def test_top_level_exports():
    assert ox.energy_vad is energy_vad
    assert ox.SpeakerDiarizer is SpeakerDiarizer
    assert ox.speech.wiener_filter is wiener_filter


def test_frame_energies_db():
    db = frame_energies_db(tone(440, 1.0), SR).to_numpy()
    assert float(np.mean(db)) == pytest.approx(-9.03, abs=0.5)
    silence = frame_energies_db(np.zeros(SR), SR).to_numpy()
    assert bool((silence <= -200).all())
    with pytest.raises(ValueError):
        frame_energies_db(np.zeros(10), SR)
    with pytest.raises(ValueError):
        frame_energies_db([], SR)


def test_energy_vad_segments_and_options():
    wav = np.concatenate([np.zeros(SR // 2), tone(440, 0.5), np.zeros(SR // 2)])
    segs = energy_vad(wav, SR)
    assert len(segs) == 1
    assert segs[0][0] == pytest.approx(0.5, abs=0.05)
    assert segs[0][1] == pytest.approx(1.0, abs=0.05)
    assert energy_vad(np.zeros(SR), SR) == []
    burst = np.concatenate([tone(440, 0.05), np.zeros(240), tone(440, 0.05)])
    assert len(energy_vad(burst, SR, hangover=5, min_duration_ms=1.0)) == 1
    blip = np.concatenate([np.zeros(4000), tone(440, 0.02), np.zeros(4000)])
    assert energy_vad(blip, SR, min_duration_ms=100.0) == []


def test_diarizer_two_speakers():
    wav = np.concatenate([tone(440, 2.0), tone(880, 2.0)])

    def dominant_freq(segment):
        spec = np.abs(np.fft.rfft(segment * np.hanning(len(segment))))
        low = float(np.argmax(spec)) < 330.0
        return np.array([1.0, 0.0] if low else [0.0, 1.0])

    diar = SpeakerDiarizer(dominant_freq, window_s=0.5, hop_s=0.5, threshold=0.5)
    segs = diar.diarize(wav, SR)
    assert len(segs) == 2
    assert segs[0][2] != segs[1][2]
    assert segs[0][0] == pytest.approx(0.0, abs=0.01)
    assert segs[1][0] == pytest.approx(2.0, abs=0.01)
    with pytest.raises(TypeError):
        SpeakerDiarizer("nope")
    with pytest.raises(ValueError):
        SpeakerDiarizer(dominant_freq, threshold=3.0)


def test_spectral_subtraction_improves_snr():
    rng = np.random.default_rng(0)
    clean = tone(440, 1.0, amp=0.5)
    noise = rng.normal(scale=0.5, size=clean.shape)
    noisy = np.concatenate([noise[:800], clean[800:] + noise[800:]])
    before = snr_db(clean[800:], noisy[800:])
    out = spectral_subtraction(noisy, noise_frames=5).to_numpy()
    after = snr_db(clean[800:], out[800:])
    assert after > before + 2.0
    with pytest.raises(ValueError):
        spectral_subtraction(noisy, noise_frames=10000)


def test_wiener_filter_improves_snr():
    rng = np.random.default_rng(1)
    clean = tone(330, 1.0, amp=0.5)
    noise = rng.normal(scale=0.4, size=clean.shape)
    noisy = np.concatenate([noise[:800], clean[800:] + noise[800:]])
    before = snr_db(clean[800:], noisy[800:])
    out = wiener_filter(noisy, noise_frames=5, iterations=2).to_numpy()
    assert snr_db(clean[800:], out[800:]) > before + 2.0
    with pytest.raises(ValueError):
        wiener_filter(noisy, iterations=0)


def test_ideal_ratio_mask_recovers_target():
    sr = SR
    t = np.arange(sr) / sr
    s1 = np.sin(2 * np.pi * 440 * t)
    s2 = 0.8 * np.sin(2 * np.pi * 1200 * t)
    mix = s1 + s2
    from oxideon.signal import istft, stft

    mix_spec = stft(ox.from_numpy(mix.astype(np.float32)), 512, center=False).to_numpy()
    tgt_spec = stft(ox.from_numpy(s1.astype(np.float32)), 512, center=False).to_numpy()
    mix_c = mix_spec[..., 0] + 1j * mix_spec[..., 1]
    tgt_c = tgt_spec[..., 0] + 1j * tgt_spec[..., 1]
    mask = ideal_ratio_mask(mix_c, tgt_c)
    assert mask.shape == mix_c.shape
    assert bool(((mask >= 0) & (mask <= 1)).all())
    est = apply_mask(mix_c, mask)
    back = istft(
        ox.from_numpy(np.stack([est.real, est.imag], axis=-1).astype(np.float32)),
        512, center=False, length=sr,
    ).to_numpy()
    corr = float(np.corrcoef(back, s1)[0, 1])
    assert corr > 0.95
    with pytest.raises(ValueError):
        ideal_ratio_mask(mix_c, tgt_c[:, :10])
    with pytest.raises(ValueError):
        apply_mask(mix_c, np.ones((4, 4)))


def test_fingerprint_matches_self_not_other():
    rng = np.random.default_rng(2)
    t = np.arange(2 * SR) / SR
    clip = np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 660 * t)
    clip = clip + 0.02 * rng.normal(size=clip.shape)
    other = np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 900 * t)
    fp = audio_fingerprint(clip, SR)
    same = audio_fingerprint(clip, SR)
    assert len(fp) > 50
    offset, count = fingerprint_matches(fp, same)
    assert offset == 0
    assert count == len(fp)
    _, cross = fingerprint_matches(fp, audio_fingerprint(other, SR))
    assert cross < 0.1 * len(fp)
    assert fingerprint_matches([], fp) == (0, 0)
    with pytest.raises(ValueError):
        audio_fingerprint(clip, SR, top_k=0)
