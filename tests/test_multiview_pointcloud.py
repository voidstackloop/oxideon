"""Focused tests for Group 7: multiview geometry + point-cloud registration."""

import numpy as np
import pytest

import oxideon as ox
from oxideon.vision import ops


def _two_view(seed=0, n=20):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, (n, 3))
    X[:, 2] = rng.uniform(4, 8, n)
    a = np.deg2rad(10.0)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    t = np.array([1.0, 0.2, 0.1])
    K = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])
    P1 = K @ np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)
    P2 = K @ np.concatenate([R, t[:, None]], axis=1)
    Xh = np.concatenate([X, np.ones((n, 1))], axis=1)
    x1 = ((P1 @ Xh.T).T[:, :2].T / (P1 @ Xh.T)[2]).T
    x2 = ((P2 @ Xh.T).T[:, :2].T / (P2 @ Xh.T)[2]).T
    n1 = np.concatenate([X, np.ones((n, 1))], axis=1)[:, :2] / X[:, 2:3]
    X2 = X @ R.T + t[None, :]
    n2 = X2[:, :2] / X2[:, 2:3]
    E_true = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]]) @ R
    return X, R, t, K, P1, P2, x1, x2, n1, n2, E_true


def _epipolar_residual(x1, x2, E):
    X1 = np.concatenate([x1, np.ones((len(x1), 1))], axis=1)
    X2 = np.concatenate([x2, np.ones((len(x2), 1))], axis=1)
    return np.abs(np.sum(X2 * (E @ X1.T).T, axis=1))


def test_rodrigues_known_and_roundtrip():
    R = ops.rodrigues([0.0, 0, np.pi / 2]).to_numpy()
    assert R.reshape(-1).tolist() == pytest.approx([0, -1, 0, 1, 0, 0, 0, 0, 1], abs=1e-6)
    rng = np.random.default_rng(0)
    for v in [rng.normal(size=3) * 0.5, np.array([1e-9, 0.0, 0.0]), np.array([np.pi, 0, 0]),
              np.pi / np.sqrt(2) * np.array([1.0, 1.0, 0.0])]:
        R = ops.rodrigues(v).to_numpy().astype(np.float64)
        back = ops.rodrigues(ops.rodrigues_inv(R)).to_numpy().astype(np.float64)
        assert np.linalg.norm(R - back) < 1e-6
    with pytest.raises(ValueError):
        ops.rodrigues_inv(np.eye(3) * 2.0)


def test_normalize_by_K_and_project():
    K = np.array([[2.0, 0, 10], [0, 2, 20], [0, 0, 1]])
    n = ops.normalize_by_K([[12.0, 22]], K).to_numpy()
    assert n.reshape(-1).tolist() == pytest.approx([1.0, 1.0])
    uv, depth = ops.project_points([[0.0, 0, 5], [1, 2, 5]], K)
    assert uv.to_numpy()[0].tolist() == pytest.approx([10.0, 20.0])
    assert uv.to_numpy()[1].tolist() == pytest.approx([10.4, 20.8])
    assert depth.to_numpy().tolist() == pytest.approx([5.0, 5.0])
    with pytest.raises(ValueError):
        ops.normalize_by_K([[0.0, 0]], np.zeros((3, 3)))


def test_normalize_points_2d():
    pts = np.array([[0.0, 0], [4, 0], [0, 4], [4, 4]])
    q, T = (v.to_numpy() for v in ops.normalize_points_2d(pts))
    assert np.linalg.norm(q, axis=1).tolist() == pytest.approx([np.sqrt(2)] * 4)
    assert q.mean(axis=0).tolist() == pytest.approx([0.0, 0.0], abs=1e-9)
    with pytest.raises(ValueError):
        ops.normalize_points_2d(np.zeros((3, 2)))


def test_essential_8point_exact():
    X, R, t, K, P1, P2, x1, x2, n1, n2, E_true = _two_view()
    E = ops.essential_8point(n1, n2).to_numpy().astype(np.float64)
    assert E.shape == (3, 3)
    assert np.linalg.svd(E, compute_uv=False)[2] < 1e-9
    assert float(_epipolar_residual(n1, n2, E).max()) < 1e-6
    assert abs(np.linalg.norm(E) - 1.0) < 1e-6
    Ek = ops.essential_8point(x1, x2, K, K).to_numpy().astype(np.float64)
    assert float(_epipolar_residual(n1, n2, Ek).max()) < 1e-6
    with pytest.raises(ValueError):
        ops.essential_8point(n1[:7], n2[:7])


def test_select_essential_pose_recovers_motion():
    X, R, t, K, P1, P2, x1, x2, n1, n2, E_true = _two_view()
    E = ops.essential_8point(n1, n2)
    assert len(ops.essential_twisted_pair(E)) == 4
    pose = ops.select_essential_pose(E, n1, n2)
    assert pose["in_front"] == len(n1)
    assert np.linalg.norm(pose["R"].to_numpy().astype(np.float64) - R) < 1e-6
    te = pose["t"].to_numpy().astype(np.float64)
    assert float(te @ t / np.linalg.norm(te) / np.linalg.norm(t)) > 1 - 1e-9


def test_triangulate_dlt_exact():
    X, R, t, K, P1, P2, x1, x2, n1, n2, E_true = _two_view()
    Xr = ops.triangulate_dlt(x1, x2, P1, P2).to_numpy()
    assert np.abs(Xr - X).max() < 1e-5
    count, _ = ops.cheirality_count(R, t, n1, n2)
    assert count == len(n1)


def test_sampson_and_ransac_essential():
    X, R, t, K, P1, P2, x1, x2, n1, n2, E_true = _two_view(n=40)
    E = ops.essential_8point(n1, n2)
    assert float(ops.sampson_error(n1, n2, E).to_numpy().max()) < 1e-9
    rng = np.random.default_rng(1)
    noisy2 = n2.copy()
    # Guaranteed-separated outliers: Sampson error >> threshold under true E.
    bad = np.empty((12, 2))
    for i in range(12):
        for _ in range(1000):
            c = rng.uniform(-1, 1, 2)
            e = float(ops.sampson_error([n1[i]], [c], E).to_numpy()[0])
            if e > 0.1:
                bad[i] = c
                break
        else:
            raise AssertionError("no separated outlier found")
    noisy2[:12] = bad
    E_r, mask = ops.ransac_essential(n1, noisy2, thresh=1e-3, max_iters=500, seed=0)
    assert int(mask[12:].sum()) >= 26  # true-inlier recall
    assert int(mask[:12].sum()) < 6  # majority of gross outliers rejected
    assert float(_epipolar_residual(n1[12:], n2[12:], E_r.to_numpy()).max()) < 0.05
    E_c, mask_c = ops.ransac_essential(n1, n2, thresh=1e-3, max_iters=50, seed=0)
    assert bool(mask_c.all())
    assert float(_epipolar_residual(n1, n2, E_c.to_numpy()).max()) < 1e-6
    with pytest.raises(ValueError):
        ops.sampson_error(n1, n2[:10], E)


def test_voxel_downsample():
    p = np.array([[0.1, 0, 0], [0.2, 0, 0], [1.5, 0, 0]])
    out = ox.voxel_downsample(p, 1.0).to_numpy()
    assert out.reshape(-1).tolist() == pytest.approx([0.15, 0.0, 0.0, 1.5, 0.0, 0.0])
    with pytest.raises(ValueError):
        ox.voxel_downsample(p, 0.0)


def test_estimate_normals_plane():
    g = np.stack(np.meshgrid(np.arange(5.0), np.arange(5.0), [0.0]), axis=-1).reshape(-1, 3)
    n = ox.estimate_normals(g, k=4, viewpoint=[0.0, 0, 5]).to_numpy()
    assert np.abs(n - [0.0, 0, 1.0]).max() < 1e-6
    n2 = ox.estimate_normals(g, k=4).to_numpy()
    assert np.abs(n2 - [0.0, 0, 1.0]).max() < 1e-6
    with pytest.raises(ValueError):
        ox.estimate_normals(g, k=2)


def test_umeyama_similarity():
    rng = np.random.default_rng(2)
    s = rng.normal(size=(20, 3))
    a = np.deg2rad(30.0)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    t = np.array([1.0, 2, 3])
    T = ox.umeyama(s, 2.0 * s @ R.T + t[None, :]).to_numpy().astype(np.float64)
    assert np.abs(T[:3, :3] - 2.0 * R).max() < 1e-6
    assert np.abs(T[:3, 3] - t).max() < 1e-6
    T = ox.umeyama(s, s @ R.T + t[None, :], with_scale=False).to_numpy().astype(np.float64)
    assert np.abs(T[:3, :3] - R).max() < 1e-6
    with pytest.raises(ValueError):
        ox.umeyama(s, s[:10])


def test_icp_converges():
    rng = np.random.default_rng(3)
    tgt = rng.uniform(-3, 3, (60, 3))
    a = np.deg2rad(5.0)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    src = tgt @ R.T + np.array([0.5, -0.2, 0.1])[None, :]
    res = ox.icp(src, tgt, max_iters=100, tol=1e-9)
    assert res["converged"]
    assert res["rmse"] < 0.05
    assert res["iterations"] < 100
    ident = ox.icp(tgt, tgt)
    assert np.linalg.norm(ident["transform"].to_numpy().astype(np.float64) - np.eye(4)) < 1e-6
    assert ident["rmse"] < 1e-9


def test_emd_and_hausdorff():
    line = np.array([[0.0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0]])
    assert ox.earth_mover_distance(line, line) == pytest.approx(0.0, abs=1e-5)
    assert ox.earth_mover_distance(line, line + [1.0, 0, 0]) == pytest.approx(1.0)
    sq = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    with_center = np.concatenate([sq, [[0.5, 0.5, 0]]])
    assert ox.hausdorff_distance(sq, with_center) == pytest.approx(np.sqrt(0.5))
    with pytest.raises(ValueError):
        ox.earth_mover_distance(line, line[:2])


def test_depth_metrics():
    gt = np.array([[1.0, 2], [3, 4]])
    m = ox.vision.ops.depth_metrics(gt, gt)
    assert m["rmse"] == 0.0 and m["delta1"] == 1.0 and m["abs_rel"] == 0.0
    m = ox.vision.ops.depth_metrics(2 * gt, gt)
    assert m["abs_rel"] == pytest.approx(1.0)
    assert m["sq_rel"] == pytest.approx(2.5)
    assert m["rmse"] == pytest.approx(np.sqrt(7.5))
    assert m["delta1"] == 0.0
    with pytest.raises(ValueError):
        ox.vision.ops.depth_metrics(gt, np.zeros((2, 2)))
