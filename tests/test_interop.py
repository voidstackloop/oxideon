"""Integration test suite for Oxideon Python interoperability with popular libraries."""

import numpy as np
import torch
import pytest

import oxideon as ox


def test_numpy_bidirectional_conversion():
    dtypes = [
        (np.float32, "f32"),
        (np.float64, "f64"),
        (np.int32, "i32"),
        (np.int64, "i64"),
        (np.uint8, "u8"),
        (np.bool_, "bool"),
    ]
    for np_dt, expected_dtype in dtypes:
        orig = np.array([[1, 2, 3], [4, 5, 6]], dtype=np_dt)
        t = ox.from_numpy(orig)
        assert t.shape == [2, 3]
        assert t.dtype == expected_dtype
        back = t.numpy()
        assert back.dtype == np_dt
        assert np.array_equal(orig, back)


def test_numpy_array_protocol():
    ox_t = ox.Tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2], dtype="f32")
    arr = np.asarray(ox_t)
    assert isinstance(arr, np.ndarray)
    assert np.allclose(arr, [[1.0, 2.0], [3.0, 4.0]])

    arr_copy = np.array(ox_t)
    assert isinstance(arr_copy, np.ndarray)
    assert np.allclose(arr_copy, [[1.0, 2.0], [3.0, 4.0]])

    # Mathematical functions directly accepting ox.Tensor
    sin_arr = np.sin(ox_t)
    assert isinstance(sin_arr, np.ndarray)
    assert np.allclose(sin_arr, np.sin([[1.0, 2.0], [3.0, 4.0]]))


def test_numpy_export_matches_values_across_layouts_and_dtypes():
    cases = [
        (np.float32, (4, 5)),
        (np.float64, (3, 7)),
        (np.int32, (6,)),
        (np.int64, (2, 3, 4)),
        (np.uint8, (5, 5)),
    ]
    for np_dt, shape in cases:
        n = int(np.prod(shape))
        base = (np.arange(n).reshape(shape) % 251).astype(np_dt)
        t = ox.from_numpy(base)
        assert np.array_equal(t.numpy(), base)
        if len(shape) > 1:
            axes = list(range(base.ndim))
            axes[0], axes[1] = axes[1], axes[0]
            assert np.array_equal(t.transpose(0, 1).numpy(), base.transpose(axes))
        assert np.array_equal(
            t.narrow(0, 1, shape[0] - 1).numpy(), base.take(range(1, shape[0]), axis=0)
        )
        assert t.numpy().flags.writeable

    f16 = (np.arange(12).reshape(3, 4) % 17).astype(np.float16)
    back = ox.from_numpy(f16).numpy()
    assert back.dtype == np.float32
    assert np.allclose(back, f16.astype(np.float32))


def test_from_numpy_honours_non_native_byteorder():
    bef = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=">f4")
    assert ox.from_numpy(bef).to_list() == [1.0, 2.0, 3.0, 4.0]
    bei = np.array([1, 256, 65536], dtype=">i4")
    assert ox.from_numpy(bei).to_list() == [1, 256, 65536]


def test_from_numpy_large_roundtrip_exact():
    for np_dt, shape in [(np.float32, (1000, 1000)), (np.int64, (999, 1001)), (np.float64, (12345,))]:
        base = (np.arange(int(np.prod(shape))).reshape(shape) % 100003).astype(np_dt)
        assert np.array_equal(ox.from_numpy(base).numpy(), base)
    sq = (np.arange(250000).reshape(500, 500) % 1009).astype(np.float32)
    assert np.array_equal(ox.from_numpy(sq.T).numpy(), sq.T)


def test_array_interface_strided_views():
    base = (np.arange(20).reshape(4, 5) % 251).astype(np.float32)
    t = ox.from_numpy(base)
    assert np.array_equal(np.asarray(t), base)
    assert np.array_equal(np.asarray(t.transpose(0, 1)), base.T)
    assert np.array_equal(np.asarray(t.narrow(0, 1, 3)), base[1:4])
    assert np.array_equal(np.asarray(t.transpose(0, 1).narrow(0, 1, 3)), base.T[1:4])
    assert not np.asarray(t).flags.writeable


def test_array_interface():
    ox_t = ox.Tensor([5.0, 6.0], shape=[2], dtype="f32")
    iface = ox_t.__array_interface__
    assert iface["shape"] == (2,)
    assert iface["typestr"] == "<f4"
    assert "data" in iface


def test_pytorch_interop_and_dlpack():
    ox_t = ox.Tensor([1.0, -1.0, 2.0, -2.0], shape=[2, 2], dtype="f32")

    # to_torch
    torch_t = ox_t.to_torch()
    assert isinstance(torch_t, torch.Tensor)
    assert torch.allclose(torch_t, torch.tensor([[1.0, -1.0], [2.0, -2.0]]))

    # from_torch
    orig_torch = torch.tensor([[10.0, 20.0], [30.0, 40.0]], requires_grad=True)
    t2 = ox.from_torch(orig_torch)
    assert t2.shape == [2, 2]
    assert np.allclose(t2.numpy(), [[10.0, 20.0], [30.0, 40.0]])

    # DLPack protocol
    torch_dl = torch.from_dlpack(ox_t)
    assert isinstance(torch_dl, torch.Tensor)
    assert torch.allclose(torch_dl, torch_t)
    assert ox_t.__dlpack_device__() == (1, 0)


def test_pillow_interop():
    Image = pytest.importorskip("PIL.Image")
    img_arr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    pil_in = Image.fromarray(img_arr, mode="RGB")

    # from_pil
    t = ox.from_pil(pil_in, normalize=True, as_chw=True)
    assert t.shape == [3, 32, 32]
    assert 0.0 <= t.numpy().min() and t.numpy().max() <= 1.0

    # to_pil
    pil_out = t.to_pil(mode="RGB")
    assert isinstance(pil_out, Image.Image)
    assert pil_out.size == (32, 32)
    assert pil_out.mode == "RGB"


def test_pandas_and_polars_interop():
    pd = pytest.importorskip("pandas")
    # Pandas
    t = ox.Tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2], dtype="f64")
    df_pd = t.to_pandas(columns=["Col1", "Col2"])
    assert isinstance(df_pd, pd.DataFrame)
    assert list(df_pd.columns) == ["Col1", "Col2"]
    t_from_pd = ox.from_pandas(df_pd)
    assert t_from_pd.shape == [2, 2]
    assert np.allclose(t_from_pd.numpy(), df_pd.to_numpy())

    # Polars
    pl = pytest.importorskip("polars")
    df_pl = t.to_polars(schema=["X", "Y"])
    assert isinstance(df_pl, pl.DataFrame)
    assert df_pl.columns == ["X", "Y"]
    t_from_pl = ox.from_polars(df_pl)
    assert t_from_pl.shape == [2, 2]
    assert np.allclose(t_from_pl.numpy(), df_pl.to_numpy())


def test_networkx_interop():
    import networkx as nx

    # Undirected graph
    G = nx.cycle_graph(5)
    t = ox.from_networkx(G)
    assert t.shape == [5, 5]
    G_rec = ox.to_networkx(t)
    assert isinstance(G_rec, nx.Graph)
    assert G_rec.number_of_nodes() == 5
    assert G_rec.number_of_edges() == 5

    # Directed graph
    DiG = nx.path_graph(4, create_using=nx.DiGraph)
    t_dir = ox.from_networkx(DiG)
    assert t_dir.shape == [4, 4]
    DiG_rec = ox.to_networkx(t_dir)
    assert isinstance(DiG_rec, nx.DiGraph)
    assert DiG_rec.number_of_edges() == 3


def test_sympy_interop():
    import sympy as sp

    x, y = sp.symbols("x y")
    expr = x**2 + 2 * y

    grid_t = ox.from_sympy(expr, symbols=[x, y], ranges=[(0.0, 1.0), (0.0, 2.0)], steps=5)
    assert grid_t.shape == [5, 5]
    assert np.isclose(grid_t.numpy()[0, 0], 0.0)
    assert np.isclose(grid_t.numpy()[-1, -1], 1.0 + 2 * 2.0)  # 5.0

    # to_sympy matrix
    mat_t = ox.Tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    sp_mat = ox.to_sympy(mat_t)
    assert isinstance(sp_mat, sp.Matrix)
    assert sp_mat.shape == (2, 2)
    assert float(sp_mat[0, 1]) == 2.0


def test_torch_state_dict_interop():
    model = ox.nn.Sequential(
        ox.nn.Linear(4, 8),
        ox.nn.GELU(),
        ox.nn.Linear(8, 2),
    )
    # Export state dict to PyTorch tensors
    torch_state = ox.export_torch_state_dict(model)
    assert isinstance(torch_state, dict)
    for k, v in torch_state.items():
        assert isinstance(v, torch.Tensor)

    # Modify one weight tensor and reload
    first_key = list(torch_state.keys())[0]
    torch_state[first_key] = torch_state[first_key] + 1.5

    missing = ox.load_torch_state_dict(model, torch_state)
    assert missing == []
    # Verify parameter changed in Oxideon model
    param_arr = dict(model.named_parameters())[first_key].to_numpy()
    assert np.allclose(param_arr, torch_state[first_key].numpy())


def test_optional_library_fallbacks():
    # JAX fallback
    t = ox.Tensor.randn([2, 2])
    try:
        import jax  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="requires JAX"):
            ox.to_jax(t)

    # SciPy fallback
    try:
        import scipy.sparse  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="requires SciPy"):
            ox.to_scipy_sparse(t)


def test_dlpack_zero_copy_and_pinning():
    t = ox.Tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2], dtype="f32")
    assert t.__dlpack_device__() == (1, 0)

    np_view = np.from_dlpack(t)
    assert np_view.shape == (2, 2)
    assert np_view.dtype == np.float32
    assert np_view.__array_interface__["data"][0] == t.__array_interface__["data"][0]

    # Mutating through compatible in-place assign updates the exported DLPack view without reallocating.
    t.assign(ox.Tensor([10.0, 20.0, 30.0, 40.0], shape=[2, 2], dtype="f32"))
    assert np.allclose(np_view, [[10.0, 20.0], [30.0, 40.0]])
    assert np_view.__array_interface__["data"][0] == t.__array_interface__["data"][0]

    # PyTorch from_dlpack also shares the exact same backing pointer.
    torch_view = torch.from_dlpack(t)
    assert torch_view.data_ptr() == t.__array_interface__["data"][0]
    assert torch.allclose(torch_view, torch.tensor([[10.0, 20.0], [30.0, 40.0]]))

    del np_view, torch_view


def test_native_complex_dtype_roundtrip_and_ops():
    for np_dt, ox_dt in [(np.complex64, "c64"), (np.complex128, "c128")]:
        a_np = np.array([[1 + 2j, 3 - 4j], [-5 + 0.5j, 0 + 1j]], dtype=np_dt)
        b_np = np.array([[2 - 1j, -1 + 2j], [4 + 3j, 1 - 2j]], dtype=np_dt)

        a = ox.from_numpy(a_np)
        b = ox.from_numpy(b_np)
        assert a.dtype == ox_dt
        assert b.dtype == ox_dt
        assert np.allclose(a.numpy(), a_np)
        assert np.allclose(np.asarray(a), a_np)

        # Arithmetic, conjugate, real/imag, abs, matmul, reductions
        assert np.allclose((a + b).numpy(), a_np + b_np)
        assert np.allclose((a - b).numpy(), a_np - b_np)
        assert np.allclose((a * b).numpy(), a_np * b_np)
        assert np.allclose((a / b).numpy(), a_np / b_np)
        assert np.allclose(a.conj().numpy(), np.conj(a_np))
        assert np.allclose(a.real.numpy(), a_np.real)
        assert np.allclose(a.imag.numpy(), a_np.imag)
        assert np.allclose(a.abs().numpy(), np.abs(a_np))
        assert np.allclose(a.matmul(b).numpy(), a_np @ b_np)
        assert np.allclose(a.sum().numpy(), np.sum(a_np))

        # Zero-copy DLPack roundtrip with NumPy and PyTorch
        np_dl = np.from_dlpack(a)
        assert np_dl.dtype == np_dt
        assert np_dl.__array_interface__["data"][0] == a.__array_interface__["data"][0]
        assert np.allclose(np_dl, a_np)

        torch_dl = torch.from_dlpack(a)
        assert torch_dl.data_ptr() == a.__array_interface__["data"][0]
        assert np.allclose(torch_dl.numpy(), a_np)




