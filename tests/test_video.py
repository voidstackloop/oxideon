"""Focused tests for Group 8: video analytics."""

import numpy as np
import pytest

import oxideon as ox


def test_mog_segments_moving_square():
    bg = np.full((20, 20), 100.0)
    mog = ox.MOGBackgroundSubtractor(n_modes=3, learning_rate=0.1)
    for _ in range(30):
        mog.apply(bg)
    frame = bg.copy()
    frame[5:11, 7:13] = 200.0
    mask = mog.apply(frame).to_numpy()
    expect = np.zeros((20, 20), dtype=np.uint8)
    expect[5:11, 7:13] = 1
    assert np.array_equal(mask, expect)
    assert mog.get_background().to_numpy().reshape(-1).tolist() == pytest.approx(
        [100.0] * 400
    )


def test_mog_adapts_to_static_object():
    bg = np.full((16, 16), 50.0)
    mog = ox.MOGBackgroundSubtractor(learning_rate=0.1)
    for _ in range(10):
        mog.apply(bg)
    obj = bg.copy()
    obj[4:8, 4:8] = 150.0
    assert mog.apply(obj).to_numpy().sum() == 16
    for _ in range(60):
        mog.apply(obj)
    assert mog.apply(obj).to_numpy().sum() == 0


def test_mog_validation():
    mog = ox.MOGBackgroundSubtractor()
    with pytest.raises(ValueError):
        mog.get_background()
    with pytest.raises(ValueError):
        ox.MOGBackgroundSubtractor(learning_rate=0.0)
    mog.apply(np.zeros((8, 8)))
    with pytest.raises(ValueError):
        mog.apply(np.zeros((8, 9)))


def test_shot_cut_detection():
    red = np.zeros((8, 8, 3))
    red[:, :] = [255, 0, 0]
    blue = np.zeros((8, 8, 3))
    blue[:, :] = [0, 0, 255]
    out = ox.detect_shot_boundaries([red, red, red, blue, blue])
    assert out["cuts"] == [3]
    assert out["flashes"] == []
    out = ox.detect_shot_boundaries([red, red, red])
    assert out == {"cuts": [], "flashes": []}


def test_shot_flash_veto():
    red = np.zeros((8, 8, 3))
    red[:, :] = [255, 0, 0]
    white = np.full((8, 8, 3), 255.0)
    out = ox.detect_shot_boundaries([red, red, white, red, red])
    assert out["cuts"] == []
    assert out["flashes"] == [2]
    with pytest.raises(ValueError):
        ox.detect_shot_boundaries([])
    with pytest.raises(ValueError):
        ox.detect_shot_boundaries([red, np.zeros((4, 4, 3))])


def test_interpolate_shifted_noise_exact_interior():
    rng = np.random.default_rng(0)
    prev = rng.uniform(0, 255, (64, 64))
    nxt = np.zeros_like(prev)
    nxt[:, 4:] = prev[:, :-4]
    mid = ox.interpolate_frames(prev, nxt, block=16, search=7).to_numpy()
    truth = np.zeros_like(prev)
    truth[:, 2:] = prev[:, :-2]
    # Right-edge blocks cannot see their match (search clipped): exclude them.
    assert float(((mid - truth)[:, 8:48] ** 2).mean()) < 1e-9
    with pytest.raises(ValueError):
        ox.interpolate_frames(prev, prev[:, :32])
    with pytest.raises(ValueError):
        ox.interpolate_frames(prev, nxt, block=0)


def test_estimate_translation_exact():
    rng = np.random.default_rng(4)
    ref = rng.normal(size=(48, 48))
    assert ox.estimate_translation(ref, np.roll(ref, (5, -3), axis=(0, 1))) == pytest.approx(
        (5.0, -3.0)
    )
    assert ox.estimate_translation(ref, ref) == pytest.approx((0.0, 0.0))
    with pytest.raises(ValueError):
        ox.estimate_translation(ref, ref[:, :10])


def test_stabilize_jittered_sequence():
    rng = np.random.default_rng(9)
    base = rng.normal(size=(48, 48))
    jitters = [(0, 0), (2, -1), (-1, 3), (1, 1), (0, -2)]
    frames = [np.roll(base, j, axis=(0, 1)) for j in jitters]
    stab, corr = (v.to_numpy() for v in ox.stabilize_frames(frames, radius=10))
    assert stab.shape[0] == 5 and corr.shape == (5, 2)
    assert stab.shape[1] < 48 and stab.shape[2] < 48
    assert float(((stab[1:] - stab[:-1]) ** 2).mean()) < 1e-12
    with pytest.raises(ValueError):
        ox.stabilize_frames(frames, radius=-1)
