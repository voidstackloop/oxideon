"""Focused tests for Group 6b vision toolkits (decode/metrics/track/flow/stereo/OCR/seg)."""

import numpy as np
import pytest

from oxideon.vision import models, ops


def test_yolo_decode_exact_box():
    pred = np.zeros((1, 1, 1 * (5 + 1)), dtype=np.float32)
    pred[0, 0, :] = [0.0, 0.0, 0.0, 0.0, 10.0, 10.0]
    boxes, scores, classes = ops.yolo_decode(pred, [[10.0, 20.0]], stride=32, num_classes=1)
    assert boxes.to_numpy().reshape(-1).tolist() == pytest.approx([11.0, 6.0, 21.0, 26.0])
    assert float(scores.to_numpy()[0]) == pytest.approx(1.0, abs=1e-3)
    assert classes.to_numpy().tolist() == [0]


def test_yolo_decode_batch_and_empty():
    pred = np.zeros((2, 1, 1, 1 * (5 + 2)), dtype=np.float32)
    pred[0, 0, 0, :] = [0, 0, 0, 0, 10.0, 10.0, -10.0]
    pred[1, 0, 0, :] = [0, 0, 0, 0, -10.0, 10.0, 10.0]
    out = ops.yolo_decode_batch(pred, [[8.0, 8.0]], stride=16, num_classes=2)
    assert len(out) == 2
    assert out[0]["classes"].to_numpy().tolist() == [0]
    assert out[1]["boxes"].to_numpy().shape == (0, 4)
    with pytest.raises(ValueError):
        ops.yolo_decode(np.zeros((2, 2, 8), dtype=np.float32), [[8.0, 8.0]], 16, 2)


def test_map_perfect_partial_empty():
    gt = np.array([[0.0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    r = ops.mean_average_precision(gt, [0.9, 0.8], [0, 0], gt, [0, 0])
    assert r["map"] == pytest.approx(1.0)
    assert r["aps"] == {0: pytest.approx(1.0)}
    pb = np.array([[0.0, 0, 10, 10], [100, 100, 110, 110], [20, 20, 30, 30]], dtype=np.float32)
    r = ops.mean_average_precision(pb, [0.9, 0.8, 0.7], [0, 0, 0], gt, [0, 0])
    assert r["map"] == pytest.approx(0.835, abs=1e-3)
    r = ops.mean_average_precision(
        np.zeros((0, 4), dtype=np.float32), [], [], gt, [0, 0], iou_thresholds=(0.5, 0.75)
    )
    assert r["map"] == 0.0
    assert r["thresholds"] == [0.5, 0.75]


def test_mask_iou_and_panoptic():
    m = np.zeros((1, 4, 4), dtype=bool)
    m[0, :2, :2] = True
    assert ops.mask_iou(m, m).to_numpy().reshape(-1).tolist() == pytest.approx([1.0])
    other = np.zeros((1, 4, 4), dtype=bool)
    other[0, 2:, 2:] = True
    assert ops.mask_iou(m, other).to_numpy().tolist() == [[0.0]]
    assert ops.panoptic_quality(m, m)["pq"] == pytest.approx(1.0)
    pq = ops.panoptic_quality(np.concatenate([m, other]), m)
    assert pq["pq"] == pytest.approx(2.0 / 3.0)
    assert (pq["tp"], pq["fp"], pq["fn"]) == (1.0, 1.0, 0.0)


def test_oks_pck_heatmap():
    gt = np.array([[0.0, 0], [5, 5]], dtype=np.float32)
    assert ops.oks(gt, gt, [0.1, 0.2], area=100.0) == pytest.approx(1.0)
    pred = np.array([[1.0, 0], [5, 5]], dtype=np.float32)
    assert ops.oks(pred, gt, [0.1, 0.1], area=100.0) == pytest.approx(
        (np.exp(-0.5) + 1.0) / 2.0, rel=1e-6
    )
    assert ops.oks(pred, gt, [0.1, 0.1], area=100.0, visible=[True, False]) == pytest.approx(
        np.exp(-0.5), rel=1e-6
    )
    assert ops.pck(pred, gt, threshold=0.5) == pytest.approx(0.5)
    assert ops.pck(pred, gt, threshold=2.0, scale=2.0) == pytest.approx(1.0)
    hm = np.zeros((1, 6, 8), dtype=np.float32)
    hm[0, 1, 3] = 1.0
    hm[0, 1, 4] = 0.8
    hm[0, 1, 2] = 0.2
    assert ops.heatmap_to_keypoints(hm).to_numpy().tolist() == [[3.25, 1.0]]
    with pytest.raises(ValueError):
        ops.oks(gt, gt, [0.1, 0.1], area=0.0)


def test_linear_sum_assignment():
    r, c = ops.linear_sum_assignment([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    assert r.to_numpy().tolist() == [0, 1, 2]
    assert c.to_numpy().tolist() == [1, 0, 2]
    r, c = ops.linear_sum_assignment([[1, 2, 3], [2, 4, 1]])
    assert c.to_numpy().tolist() == [0, 2]
    r, c = ops.linear_sum_assignment([[1, 5], [2, 6], [3, 1]])
    assert r.to_numpy().tolist() == [0, 2]
    assert c.to_numpy().tolist() == [0, 1]
    with pytest.raises(ValueError):
        ops.linear_sum_assignment([[1, np.inf], [2, 3]])


def test_kalman_box_tracker():
    kf = ops.KalmanBoxTracker([10.0, 10, 20, 20])
    assert kf.predict().tolist() == pytest.approx([10, 10, 20, 20], abs=1e-6)
    before = float(np.trace(kf.P))
    kf.update([12.0, 10, 22, 20])
    after = float(np.trace(kf.P))
    assert after < before
    cx = (kf.get_state()[0] + kf.get_state()[2]) / 2
    assert 15.0 < cx < 17.0


def test_iou_tracker_lifecycle():
    tr = ops.IoUTracker(iou_threshold=0.3, max_age=1)
    f1 = tr.update([[0.0, 0, 10, 10], [50, 50, 60, 60]])
    assert [i for i, _ in f1] == [0, 1]
    f2 = tr.update([[1.0, 0, 11, 10], [51, 50, 61, 60]])
    assert [i for i, _ in f2] == [0, 1]
    f3 = tr.update([[2.0, 0, 12, 10]])
    assert [i for i, _ in f3] == [0]
    tr.update([[3.0, 0, 13, 10]])
    assert all(t["id"] != 1 for t in tr.tracks)


def test_byte_tracker_two_stage():
    tr = ops.ByteTracker(high_threshold=0.6, low_threshold=0.1, max_age=2)
    assert [i for i, _ in tr.update([[0.0, 0, 10, 10]], [0.9])] == [0]
    # Low-score repeat of the same box re-associates in stage 2.
    live = tr.update([[0.0, 0, 10, 10]], [0.2])
    assert [i for i, _ in live] == [0]
    # A lone low-score detection never spawns a track.
    tr2 = ops.ByteTracker()
    assert tr2.update([[0.0, 0, 10, 10]], [0.2]) == []
    with pytest.raises(ValueError):
        tr.update([[0.0, 0, 10, 10]], [2.0])


def test_lucas_kanade_translation():
    rng = np.random.default_rng(7)
    dots = (rng.random((64, 64)) > 0.96).astype(np.float64)
    prev = dots.copy()
    for _ in range(2):
        prev = (prev + np.roll(prev, 1, 0) + np.roll(prev, 1, 1)) / 3.0
    nxt = np.zeros_like(prev)
    nxt[2:, 3:] = prev[:-2, :-3]
    gy, gx = np.gradient(prev)
    mag = gx**2 + gy**2
    # Coarse-level windows need a ~half * 2**(levels-1) border margin.
    mag[:20, :] = mag[-20:, :] = 0
    mag[:, :20] = mag[:, -20:] = 0
    idx = np.argsort(mag.ravel())[-8:]
    pts = np.stack([idx % 64, idx // 64], axis=1).astype(np.float32)
    new, status = ops.lucas_kanade(prev, nxt, pts, window=9, levels=2)
    assert status.to_numpy().tolist() == [1] * 8
    flow = new.to_numpy() - pts
    # Median is robust to an occasional wrong-dot lock (LK aliasing).
    assert np.median(flow, axis=0).tolist() == pytest.approx([3.0, 2.0], abs=0.3)
    assert int(np.sum(np.abs(flow - [3.0, 2.0]).max(axis=1) < 0.6)) >= 6


def test_lucas_kanade_out_of_bounds():
    img = np.random.default_rng(0).random((32, 32))
    _, status = ops.lucas_kanade(img, img, [[0.0, 0.0]], window=15, levels=1)
    assert status.to_numpy().tolist() == [0]


def test_block_match_disparity():
    rng = np.random.default_rng(3)
    left = rng.random((24, 32))
    right = np.zeros_like(left)
    right[:, :27] = left[:, 5:]
    d = ops.block_match_disparity(left, right, max_disparity=8, window=5).to_numpy()
    assert d.dtype == np.int32
    assert d[0, 0] == -1 and d[10, 8] == -1
    valid = d[2:22, 9:30]
    assert valid.shape == (20, 21)
    assert float(np.mean(valid == 5)) > 0.95


def test_ctc_decoding():
    lp = np.full((5, 3), -10.0)
    for t, c in enumerate([0, 1, 1, 0, 2]):
        lp[t, c] = 0.0
    assert ops.ctc_greedy_decode(lp, blank=0) == [1, 2]
    beam = ops.ctc_beam_decode(lp, beam_width=2, blank=0)
    assert len(beam) <= 2
    assert beam[0][0] == (1, 2)
    assert all(beam[i][1] >= beam[i + 1][1] for i in range(len(beam) - 1))
    assert ops.ctc_greedy_decode(np.zeros((3, 2)), blank=1) == [0]


def test_edit_cer_wer():
    assert ops.edit_distance("kitten", "sitting") == 3
    assert ops.cer("kitten", "sitting") == pytest.approx(3 / 7)
    assert ops.wer("the cat sat", "the cat mat") == pytest.approx(1 / 3)
    assert ops.wer("", "") == 0.0
    assert ops.cer("a", "") == 1.0


def test_deeplab_shapes_and_determinism():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(2, 8, 12, 12)).astype(np.float32)
    h1, h2 = models.DeepLabHead(8, 4, rates=(2, 4), aspp_channels=16, seed=0), models.DeepLabHead(
        8, 4, rates=(2, 4), aspp_channels=16, seed=0
    )
    l1, l2 = h1(x).to_numpy(), h2(x).to_numpy()
    assert l1.shape == (2, 4, 12, 12)
    assert np.array_equal(l1, l2)
    h3 = models.DeepLabHead(8, 4, rates=(2, 4), aspp_channels=16, seed=1)
    assert not np.array_equal(l1, h3(x).to_numpy())
    pred = h1.predict(l1)
    assert pred.to_numpy().shape == (2, 12, 12)
    assert str(pred.to_numpy().dtype) == "int64"
    assert len(h1.parameters()) == len(h2.parameters()) > 0


def test_dilated_conv2d_matches_pointwise():
    rng = np.random.default_rng(5)
    conv = models.DilatedConv2d(3, 2, kernel_size=1, dilation=1, seed=5)
    x = rng.normal(size=(1, 3, 4, 4))
    y = conv(x).to_numpy()[0]
    w = conv.weight.to_numpy()[:, :, 0, 0]
    b = conv.bias.to_numpy()
    expect = np.einsum("oi,ihw->ohw", w, x[0]) + b[:, None, None]
    assert y.reshape(-1).tolist() == pytest.approx(expect.reshape(-1).tolist(), rel=1e-5)
    with pytest.raises(ValueError):
        models.DilatedConv2d(3, 2, kernel_size=2)
    with pytest.raises(ValueError):
        conv(np.zeros((1, 5, 4, 4), dtype=np.float32))
