"""Focused tests for Group 6c: NeRF (differentiable) + Gaussian splat renderer."""

import numpy as np
import pytest

import oxideon as ox


def test_positional_encoding_values():
    x = ox.from_numpy(np.zeros((2, 3), dtype=np.float32))
    pe = ox.positional_encoding(x, 2).to_numpy()
    assert pe.shape == (2, 15)
    assert pe[0].tolist() == [0.0] * 3 + [0.0, 0.0, 1.0, 1.0] * 3
    x = ox.from_numpy(np.array([[np.pi / 2, 0.0, 0.0]], dtype=np.float32))
    row = ox.positional_encoding(x, 2).to_numpy()[0]
    assert row[:3].tolist() == pytest.approx([np.pi / 2, 0.0, 0.0])
    assert row[3:7].tolist() == pytest.approx([1.0, 0.0, 0.0, -1.0], abs=1e-6)
    assert ox.positional_encoding(x, 2, include_input=False).to_numpy().shape == (1, 12)
    with pytest.raises(ValueError):
        ox.positional_encoding(x, 0)


def test_get_rays_center_and_pose():
    pose = np.zeros((3, 4))
    pose[:3, :3] = np.eye(3)
    o, d = ox.get_rays(5, 5, 5.0, pose)
    assert o.to_numpy().shape == (5, 5, 3)
    assert d.to_numpy()[2, 2].tolist() == pytest.approx([0.0, 0.0, -1.0])
    corner = np.array([-0.4, 0.4, -1.0])
    assert d.to_numpy()[0, 0].tolist() == pytest.approx((corner / np.linalg.norm(corner)).tolist())
    pose4 = np.eye(4)
    pose4[:3, 3] = [1.0, 2.0, 3.0]
    o4, _ = ox.get_rays(2, 3, 2.0, pose4)
    assert o4.to_numpy()[0, 0].tolist() == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        ox.get_rays(4, 4, 0.0, pose)
    with pytest.raises(ValueError):
        ox.get_rays(4, 4, 1.0, np.eye(2))


def test_stratified_samples():
    t = ox.stratified_samples(2.0, 4.0, 4, 3, perturb=False).to_numpy()
    assert t.tolist() == [[2.25, 2.75, 3.25, 3.75]] * 3
    a = ox.stratified_samples(2.0, 4.0, 8, 5, seed=0).to_numpy()
    b = ox.stratified_samples(2.0, 4.0, 8, 5, seed=0).to_numpy()
    assert np.array_equal(a, b)
    assert bool(np.all(a[:, 1:] > a[:, :-1]))
    assert float(a.min()) >= 2.0 and float(a.max()) <= 4.0
    with pytest.raises(ValueError):
        ox.stratified_samples(4.0, 2.0, 4, 1)


def test_volume_render_opaque_and_empty():
    t = [1.0, 2.0, 3.0]
    rgb = [[[1.0, 0, 0]], [[0, 1.0, 0]], [[0, 0, 1.0]]]
    out = ox.volume_render([[0.0, 1000.0, 0.0]], [np.concatenate(rgb, axis=0)], [t])
    assert out["rgb"].to_numpy()[0].tolist() == pytest.approx([0.0, 1.0, 0.0], abs=1e-4)
    assert float(out["acc"].to_numpy()[0]) == pytest.approx(1.0, abs=1e-4)
    assert float(out["depth"].to_numpy()[0]) == pytest.approx(2.0, abs=1e-3)
    out = ox.volume_render([[0.0, 0.0]], [[[0.0, 0, 0], [1.0, 1.0, 1.0]]], [0.0, 1.0])
    assert float(out["acc"].to_numpy()[0]) == 0.0
    assert out["rgb"].to_numpy()[0].tolist() == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        ox.volume_render([[1.0]], [[[0.0, 0, 0]]], [0.0, 1.0])


def test_volume_render_differentiable_end_to_end():
    sig = ox.from_numpy(np.array([[0.5, 2.0]], dtype=np.float32))
    sig.requires_grad_(True)
    rgb = ox.from_numpy(np.zeros((1, 2, 3), dtype=np.float32) + 0.5)
    out = ox.volume_render(sig, rgb, [[0.0, 1.0]])
    out["rgb"].sum().backward()
    assert sig.grad is not None
    field = ox.NeRFField(depth=2, width=8, pos_freqs=2, dir_freqs=2, skip_layer=1)
    render = ox.render_rays(
        field,
        [[0.0, 0, 0], [0.1, 0, 0]],
        [[0.0, 0, -1], [0.0, 0, -1]],
        1.0,
        3.0,
        n_coarse=4,
        n_fine=4,
        seed=0,
    )
    render["rgb_fine"].sum().backward()
    params = field.parameters()
    assert len(params) > 0
    assert all(p.grad is not None for p in params)


def test_nerf_field_forward():
    field = ox.NeRFField(depth=2, width=16, pos_freqs=2, dir_freqs=2, skip_layer=1)
    pts = ox.from_numpy(np.random.default_rng(0).normal(size=(5, 3)).astype(np.float32))
    sig, rgb = field(pts, [0.0, 0.0, -1.0])
    assert sig.to_numpy().shape == (5,)
    assert rgb.to_numpy().shape == (5, 3)
    assert float(sig.to_numpy().min()) >= 0.0
    assert float(rgb.to_numpy().min()) >= 0.0 and float(rgb.to_numpy().max()) <= 1.0
    sig2, rgb2 = field(pts, np.tile([0.0, 0.0, -1.0], (5, 1)).astype(np.float32))
    assert np.array_equal(sig.to_numpy(), sig2.to_numpy())
    assert np.array_equal(rgb.to_numpy(), rgb2.to_numpy())
    with pytest.raises(ValueError):
        field(pts, [0.0, 0.0])
    with pytest.raises(ValueError):
        ox.NeRFField(depth=1)


def test_sample_pdf_concentrates():
    t = [[0.0, 1.0, 2.0, 3.0, 4.0]]
    w = [[0.0, 0.0, 1.0, 0.0, 0.0]]
    s = ox.sample_pdf(t, w, 400, seed=1).to_numpy()
    assert s.shape == (1, 400)
    assert float(s.min()) >= 0.0 and float(s.max()) <= 4.0
    assert 1.9 < float(s.mean()) < 3.1
    assert float(np.mean((s > 1.99) & (s < 3.01))) > 0.9
    with pytest.raises(ValueError):
        ox.sample_pdf(t, [[-1.0, 0, 0, 0, 0]], 4)


def test_render_rays_smoke_and_determinism():
    field = ox.NeRFField(depth=2, width=8, pos_freqs=2, dir_freqs=2, skip_layer=0)
    kw = dict(n_coarse=8, n_fine=8, seed=3)
    o = [[0.0, 0, 0], [0.5, 0, 0]]
    d = [[0.0, 0, -1], [0.0, 0, -1]]
    r1 = ox.render_rays(field, o, d, 1.0, 3.0, **kw)
    r2 = ox.render_rays(field, o, d, 1.0, 3.0, **kw)
    assert r1["rgb_fine"].to_numpy().shape == (2, 3)
    assert r1["weights"].to_numpy().shape == (2, 16)
    assert np.array_equal(r1["rgb_fine"].to_numpy(), r2["rgb_fine"].to_numpy())
    t = r1["t_vals"].to_numpy()
    assert bool(np.all(t[:, 1:] >= t[:, :-1]))
    assert float(r1["rgb_fine"].to_numpy().min()) >= 0.0
    assert float(r1["rgb_fine"].to_numpy().max()) <= 1.0
    wsum = r1["weights"].to_numpy().sum(axis=1)
    assert wsum.tolist() == pytest.approx(r1["acc_fine"].to_numpy().tolist(), rel=1e-4)
    assert r1["depth_coarse"].to_numpy().shape == (2,)


def test_quat_to_matrix():
    m = ox.quat_to_matrix([[1.0, 0, 0, 0]]).to_numpy()
    assert m.reshape(-1).tolist() == pytest.approx(np.eye(3).reshape(-1).tolist())
    q = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
    m = ox.quat_to_matrix([q]).to_numpy()[0]
    assert m.reshape(-1).tolist() == pytest.approx(
        [0.0, -1, 0, 1, 0, 0, 0, 0, 1], abs=1e-6
    )
    with pytest.raises(ValueError):
        ox.quat_to_matrix([[0.0, 0, 0, 0]])
    with pytest.raises(ValueError):
        ox.quat_to_matrix(np.zeros((2, 3)))


def test_project_center_and_clip():
    proj = ox.project_gaussians(
        [[0.0, 0, -5], [0.0, 0, 5]],
        [[1.0, 1, 1], [1.0, 1, 1]],
        [[1.0, 0, 0, 0], [1.0, 0, 0, 0]],
        np.eye(4),
        10.0,
        10.0,
        8.0,
        8.0,
        16,
        16,
    )
    assert proj["uv"].to_numpy()[0].tolist() == pytest.approx([8.0, 8.0])
    assert proj["depth"].to_numpy().tolist() == pytest.approx([5.0, -5.0])
    assert proj["visible"].to_numpy().tolist() == [1, 0]
    assert proj["cov2d"].to_numpy()[0].reshape(-1).tolist() == pytest.approx(
        [4.3, 0.0, 0.0, 4.3], rel=1e-5
    )
    with pytest.raises(ValueError):
        ox.project_gaussians(
            [[0.0, 0, -5]], [[-1.0, 1, 1]], [[1.0, 0, 0, 0]], np.eye(4),
            10.0, 10.0, 8.0, 8.0, 16, 16,
        )


def test_rasterize_single_splat():
    img = ox.rasterize(
        [[8.0, 8.0]],
        [[[4.3, 0.0], [0.0, 4.3]]],
        [[1.0, 0.0, 0.0]],
        [1.0],
        [5.0],
        [1],
        16,
        16,
        bg=(0.0, 0.0, 0.0),
    ).to_numpy()
    assert img.shape == (16, 16, 3)
    assert img[8, 8].tolist() == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)
    assert img[0, 0].tolist() == [0.0, 0.0, 0.0]
    assert img[8, 10, 0] == pytest.approx(np.exp(-0.5 * 4.0 / 4.3), rel=1e-5)
    assert img[8, 9, 0] > img[8, 11, 0] > img[8, 13, 0]
    with pytest.raises(ValueError):
        ox.rasterize([[8.0, 8.0]], [[[1.0, 0], [0, 1.0]]], [[2.0, 0, 0]], [1.0], [1.0], [1], 4, 4)


def test_rasterize_depth_order():
    kw = dict(
        uv=[[4.0, 4.0], [4.0, 4.0]],
        cov2d=[[[2.0, 0], [0, 2.0]], [[2.0, 0], [0, 2.0]]],
        colors=[[1.0, 0, 0], [0.0, 0, 1.0]],
        depth=[1.0, 5.0],
        visible=[1, 1],
        width=9,
        height=9,
    )
    img = ox.rasterize(opacities=[1.0, 1.0], **kw).to_numpy()
    assert img[4, 4].tolist() == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)
    img = ox.rasterize(opacities=[0.5, 0.5], **kw).to_numpy()
    assert img[4, 4].tolist() == pytest.approx([0.5, 0.0, 0.25], abs=1e-9)


def test_scene_render_and_validation():
    scene = ox.GaussianSplatScene(
        means=[[0.0, 0, -4], [1.0, 0, -6]],
        scales=[[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]],
        quats=[[1.0, 0, 0, 0], [1.0, 0, 0, 0]],
        colors=[[1.0, 0, 0], [0.0, 1.0, 0]],
        opacities=[0.9, 0.8],
    )
    assert len(scene) == 2
    img = scene.render(np.eye(4), 12.0, 12.0, 8.0, 8.0, 16, 16).to_numpy()
    assert img.shape == (16, 16, 3)
    assert str(img.dtype) == "float32"
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0
    assert float(img.max()) > 0.1
    empty = scene.render(
        np.eye(4), 12.0, 12.0, 8.0, 8.0, 8, 8, bg=(0.2, 0.3, 0.4), near=100.0
    ).to_numpy()
    assert empty.reshape(-1, 3)[0].tolist() == pytest.approx([0.2, 0.3, 0.4])
    with pytest.raises(ValueError):
        ox.GaussianSplatScene([[0.0, 0, 0]], [[0.0, 1, 1]], [[1.0, 0, 0, 0]], [[1.0, 0, 0]], [1.0])
    with pytest.raises(ValueError):
        ox.GaussianSplatScene([[0.0, 0, 0]], [[1.0, 1, 1]], [[1.0, 0, 0, 0]], [[1.0, 0, 0]], [2.0])
