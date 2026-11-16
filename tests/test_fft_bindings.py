"""Z7: Rust differentiable FFT path (Tensor methods) vs torch.fft ground truth.

The `oxideon.fft` *module* shims are numpy-backed; these tests pin the *Rust*
`Tensor.fft/ifft/fft2d/rfft/irfft/stft/complex_*` methods (the autograd-capable
path) against torch, plus the two fail-closed guards (non-float FFT input and
non-pair trailing dims) and the documented full-spectrum `rfft` divergence.
"""
import numpy as np
import pytest
import torch

import oxideon as ox

T = ox.from_numpy


def test_rust_fft_ifft_match_torch():
    rng = np.random.default_rng(1)
    for n in [1, 3, 8, 33, 64]:
        xr = rng.standard_normal((n, 2)).astype(np.float32)
        t = T(xr)
        th = torch.from_numpy(xr[..., 0] + 1j * xr[..., 1])
        f = t.fft().to_numpy()
        np.testing.assert_allclose(f[..., 0] + 1j * f[..., 1],
                                   torch.fft.fft(th).numpy(), rtol=1e-4, atol=1e-5)
        fi = t.ifft().to_numpy()
        np.testing.assert_allclose(fi[..., 0] + 1j * fi[..., 1],
                                   torch.fft.ifft(th).numpy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(t.fft().ifft().to_numpy(), xr, rtol=1e-5, atol=1e-5)


def test_rust_fft2d_matches_torch():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, 5, 2)).astype(np.float32)
    th = torch.from_numpy(x[..., 0] + 1j * x[..., 1])
    f = T(x).fft2d().to_numpy()
    np.testing.assert_allclose(f[..., 0] + 1j * f[..., 1],
                               torch.fft.fft2(th).numpy(), rtol=1e-4, atol=1e-5)
    b = T(x).fft2d().ifft2d().to_numpy()
    np.testing.assert_allclose(b, x, rtol=1e-5, atol=1e-5)


def test_rust_stft_matches_torch_center_false():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(64).astype(np.float32)
    th = torch.from_numpy(x)
    s = T(x).stft(16, 8, True).to_numpy()
    assert s.shape == (1, 7, 16, 2)
    ref = torch.stft(th, n_fft=16, hop_length=8, window=torch.hann_window(16),
                     center=False, return_complex=True)
    want = torch.view_as_real(ref).numpy().transpose(1, 0, 2)
    np.testing.assert_allclose(s[0, :, :9, :], want, rtol=1e-3, atol=1e-4)
    # Hann DC-bin property: ones signal -> bin0 == n_fft/2.
    dc = T(np.ones(8, dtype=np.float32)).stft(8, 8, True).to_numpy()
    np.testing.assert_allclose(dc[0, 0, 0], [4.0, 0.0], rtol=1e-6, atol=1e-6)


def test_rust_complex_ops_match_torch():
    rng = np.random.default_rng(4)
    a = rng.standard_normal((6, 2)).astype(np.float32)
    b = rng.standard_normal((6, 2)).astype(np.float32)
    tha = torch.from_numpy(a[..., 0] + 1j * a[..., 1])
    thb = torch.from_numpy(b[..., 0] + 1j * b[..., 1])
    m = T(a).complex_mul(T(b)).to_numpy()
    np.testing.assert_allclose(m[..., 0] + 1j * m[..., 1],
                               (tha * thb).numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(T(a).complex_abs().to_numpy(),
                               torch.abs(tha).numpy(), rtol=1e-6, atol=1e-6)
    with pytest.raises(Exception, match="complex-pair|complex_mul"):
        T(rng.standard_normal((4, 3)).astype(np.float32)).complex_mul(T(b))
    with pytest.raises(Exception, match="complex-pair|complex_abs"):
        T(rng.standard_normal((4, 3)).astype(np.float32)).complex_abs()


def test_rust_rfft_full_spectrum_and_guards():
    # Deliberate divergence: Rust rfft returns the FULL two-sided spectrum,
    # unlike torch.fft.rfft one-sided. Python ox.fft.rfft matches torch.
    r = T(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)).rfft().to_numpy()
    assert r.shape == (4, 2)
    np.testing.assert_allclose(r, [[10, 0], [-2, 2], [-2, 0], [-2, -2]],
                               rtol=1e-5, atol=1e-5)
    back = T(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)).rfft().irfft().to_numpy()
    np.testing.assert_allclose(back, [1, 2, 3, 4], rtol=1e-5, atol=1e-5)
    # Non-float FFT input fails closed like torch.
    with pytest.raises(Exception, match="floating-point"):
        T(np.arange(8, dtype=np.int32).reshape(4, 2)).fft()
    with pytest.raises(Exception, match="floating-point"):
        T(np.arange(8, dtype=np.int32).reshape(4, 2)).ifft()
