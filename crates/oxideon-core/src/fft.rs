//! Differentiable Fast Fourier Transform (`fft`, `ifft`, `rfft`, `irfft`, `fft2d`, `ifft2d`),
//! complex arithmetic (`complex_mul`, `complex_abs`), and Short-Time Fourier Transform (`stft`).
//!
//! Complex tensors of logical shape `[..., N]` are stored with a trailing dimension of size 2
//! (`[..., N, 2]` holding `[real, imag]`, matching PyTorch's `view_as_real` convention).
//! Because $\text{DFT}^H = N \cdot \text{IDFT}$, the backward pass of [`Tensor::fft`] is
//! `grad.ifft() * N` and the backward pass of [`Tensor::ifft`] is `grad.fft() / N`,
//! making both operations higher-order differentiable!

use std::f64::consts::TAU;

use crate::backend::{CpuStorage, Storage};
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::tensor::Tensor;

impl Tensor {
    /// 1D Fast Fourier Transform along the second-to-last axis of a complex tensor
    /// `[..., N, 2]` -> `[..., N, 2]`.
    pub fn fft(&self) -> Result<Tensor> {
        self.fft_impl(false)
    }

    /// 1D Inverse Fast Fourier Transform along the second-to-last axis of a complex tensor
    /// `[..., N, 2]` -> `[..., N, 2]`.
    pub fn ifft(&self) -> Result<Tensor> {
        self.fft_impl(true)
    }

    fn fft_impl(&self, inverse: bool) -> Result<Tensor> {
        let r = self.ndim();
        if r < 2 || self.shape()[r - 1] != 2 {
            return Err(Error::Msg(format!(
                "fft/ifft expects complex tensor of shape [..., N, 2], got {:?}",
                self.shape()
            )));
        }
        if !self.dtype().is_floating_point() {
            return Err(Error::Msg(format!(
                "fft/ifft expects a floating-point dtype, got {:?} (like torch, which \
                 rejects integer input instead of truncating the spectrum)",
                self.dtype()
            )));
        }
        let n = self.shape()[r - 2];
        let batch: usize = self.shape()[..r - 2].iter().product();
        let data = self.to_vec::<f64>()?;
        let mut out = vec![0.0f64; data.len()];

        for b in 0..batch {
            let slice = &data[b * n * 2..(b + 1) * n * 2];
            let dst = &mut out[b * n * 2..(b + 1) * n * 2];
            fft_1d_slice(slice, dst, n, inverse);
        }

        let shape = self.shape().to_vec();
        let cpu =
            CpuStorage::F64(out).to_dtype(&Layout::contiguous(shape.clone()), self.dtype())?;
        let storage =
            Storage::new_cpu(cpu).to_device(&Layout::contiguous(shape.clone()), &self.device())?;

        Ok(Tensor::from_op(
            storage,
            Layout::contiguous(shape),
            if inverse { "ifft" } else { "fft" },
            vec![self.clone()],
            move |g| {
                let grad_in = if inverse {
                    g.fft()?.div_scalar(n as f64)?
                } else {
                    g.ifft()?.mul_scalar(n as f64)?
                };
                Ok(vec![Some(grad_in)])
            },
        ))
    }

    /// Real-to-complex 1D FFT: converts a real tensor `[..., N]` into complex `[..., N, 2]`
    /// and applies [`Tensor::fft`].
    ///
    /// NOTE — deliberate `torch.fft.rfft` divergence: this returns the FULL two-sided
    /// `[..., N, 2]` spectrum, not torch's one-sided `[..., N/2+1]`. The Rust-side
    /// [`Tensor::stft`] and mel frontend consume the full spectrum; Python users get true
    /// torch semantics from the numpy-backed `oxideon.fft.rfft` shim instead.
    pub fn rfft(&self) -> Result<Tensor> {
        let imag = self.zeros_like()?;
        let complex = Self::stack(&[self.clone(), imag], -1)?;
        complex.fft()
    }

    /// Complex-to-real 1D IFFT: applies [`Tensor::ifft`] to `[..., N, 2]` and extracts the
    /// real part `[..., N]`.
    pub fn irfft(&self) -> Result<Tensor> {
        self.ifft()?.select(-1, 0)
    }

    /// 2D Fast Fourier Transform over the last two spatial dimensions of a complex tensor
    /// `[..., H, W, 2]` -> `[..., H, W, 2]`.
    pub fn fft2d(&self) -> Result<Tensor> {
        if self.ndim() < 3 || self.shape()[self.ndim() - 1] != 2 {
            return Err(Error::Msg(
                "fft2d expects complex tensor [..., H, W, 2]".into(),
            ));
        }
        let r = self.ndim();
        let w_fft = self.fft()?;
        w_fft
            .transpose((r - 3) as isize, (r - 2) as isize)?
            .fft()?
            .transpose((r - 3) as isize, (r - 2) as isize)
    }

    /// 2D Inverse Fast Fourier Transform over `[..., H, W, 2]` -> `[..., H, W, 2]`.
    pub fn ifft2d(&self) -> Result<Tensor> {
        if self.ndim() < 3 || self.shape()[self.ndim() - 1] != 2 {
            return Err(Error::Msg(
                "ifft2d expects complex tensor [..., H, W, 2]".into(),
            ));
        }
        let r = self.ndim();
        let w_ifft = self.ifft()?;
        w_ifft
            .transpose((r - 3) as isize, (r - 2) as isize)?
            .ifft()?
            .transpose((r - 3) as isize, (r - 2) as isize)
    }

    /// Element-wise complex multiplication of two `[..., 2]` complex tensors:
    /// $(a_r + i a_i)(b_r + i b_i) = (a_r b_r - a_i b_i) + i (a_r b_i + a_i b_r)$.
    pub fn complex_mul(&self, other: &Tensor) -> Result<Tensor> {
        for (what, t) in [("lhs", self), ("rhs", other)] {
            if t.ndim() < 1 || t.shape()[t.ndim() - 1] != 2 {
                return Err(Error::Msg(format!(
                    "complex_mul expects {what} with trailing complex-pair dim 2, got {:?}",
                    t.shape()
                )));
            }
        }
        let ar = self.select(-1, 0)?;
        let ai = self.select(-1, 1)?;
        let br = other.select(-1, 0)?;
        let bi = other.select(-1, 1)?;

        let real = ar.mul(&br)?.sub(&ai.mul(&bi)?)?;
        let imag = ar.mul(&bi)?.add(&ai.mul(&br)?)?;
        Self::stack(&[real, imag], -1)
    }

    /// Magnitude $|z| = \sqrt{\text{re}^2 + \text{im}^2}$ of a `[..., 2]` complex tensor.
    pub fn complex_abs(&self) -> Result<Tensor> {
        if self.ndim() < 1 || self.shape()[self.ndim() - 1] != 2 {
            return Err(Error::Msg(format!(
                "complex_abs expects trailing complex-pair dim 2, got {:?}",
                self.shape()
            )));
        }
        self.norm_p(2.0, &[-1], false)
    }

    /// Short-Time Fourier Transform (`STFT`) on a 1D `[L]` or 2D `[batch, L]` real signal,
    /// returning complex frames `[batch, num_frames, n_fft, 2]`.
    pub fn stft(&self, n_fft: usize, hop_length: usize, hann_window: bool) -> Result<Tensor> {
        let x = if self.ndim() == 1 {
            self.unsqueeze(0)?
        } else if self.ndim() == 2 {
            self.clone()
        } else {
            return Err(Error::Msg(
                "stft expects 1D [L] or 2D [batch, L] signal".into(),
            ));
        };
        let len = x.shape()[1];
        if len < n_fft || hop_length == 0 {
            return Err(Error::Msg(
                "stft: signal shorter than n_fft or hop_length == 0".into(),
            ));
        }

        let window = if hann_window {
            let w_vec: Vec<f64> = (0..n_fft)
                .map(|i| 0.5 * (1.0 - (TAU * (i as f64) / (n_fft as f64)).cos()))
                .collect();
            Some(Tensor::from_slice_on(&w_vec, &[1, n_fft], &x.device())?.to_dtype(x.dtype())?)
        } else {
            None
        };

        let mut frames = Vec::new();
        let mut start = 0usize;
        // `len >= n_fft` was validated above, so `len - n_fft` cannot
        // underflow; the saturating step keeps a giant `hop_length` from
        // overflowing `start` on the final iteration.
        let last_start = len - n_fft;
        while start <= last_start {
            let mut frame = x.narrow(1, start, n_fft)?;
            if let Some(w) = &window {
                frame = frame.mul(w)?;
            }
            frames.push(frame);
            start = start.saturating_add(hop_length);
        }
        let stacked = Self::stack(&frames, 1)?; // [batch, num_frames, n_fft]
        stacked.rfft()
    }
}

fn fft_1d_slice(src: &[f64], dst: &mut [f64], n: usize, inverse: bool) {
    if n == 0 {
        return;
    }
    if n.is_power_of_two() {
        dst.copy_from_slice(src);
        cooley_tukey_in_place(dst, n, inverse);
    } else {
        dft_direct(src, dst, n, inverse);
    }
}

fn cooley_tukey_in_place(buf: &mut [f64], n: usize, inverse: bool) {
    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(2 * i, 2 * j);
            buf.swap(2 * i + 1, 2 * j + 1);
        }
    }

    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let angle = sign * TAU / (len as f64);
        let (wlen_r, wlen_i) = (angle.cos(), angle.sin());

        let mut i = 0;
        while i < n {
            let (mut wr, mut wi) = (1.0f64, 0.0f64);
            for k in 0..half {
                let u_idx = 2 * (i + k);
                let v_idx = 2 * (i + k + half);
                let (ur, ui) = (buf[u_idx], buf[u_idx + 1]);
                let (vr, vi) = (
                    buf[v_idx] * wr - buf[v_idx + 1] * wi,
                    buf[v_idx] * wi + buf[v_idx + 1] * wr,
                );
                buf[u_idx] = ur + vr;
                buf[u_idx + 1] = ui + vi;
                buf[v_idx] = ur - vr;
                buf[v_idx + 1] = ui - vi;

                let next_wr = wr * wlen_r - wi * wlen_i;
                wi = wr * wlen_i + wi * wlen_r;
                wr = next_wr;
            }
            i += len;
        }
        len <<= 1;
    }

    if inverse {
        let inv_n = 1.0 / (n as f64);
        for v in buf.iter_mut() {
            *v *= inv_n;
        }
    }
}

fn dft_direct(src: &[f64], dst: &mut [f64], n: usize, inverse: bool) {
    let sign = if inverse { 1.0 } else { -1.0 };
    let scale = if inverse { 1.0 / (n as f64) } else { 1.0 };
    for k in 0..n {
        let mut sum_r = 0.0f64;
        let mut sum_i = 0.0f64;
        for t in 0..n {
            let angle = sign * TAU * ((k * t) as f64) / (n as f64);
            let (c, s) = (angle.cos(), angle.sin());
            let (xr, xi) = (src[2 * t], src[2 * t + 1]);
            sum_r += xr * c - xi * s;
            sum_i += xr * s + xi * c;
        }
        dst[2 * k] = sum_r * scale;
        dst[2 * k + 1] = sum_i * scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic values in `[-1, 1)`.
    fn values(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f64 / (1u64 << 23) as f64) - 1.0
            })
            .collect()
    }

    fn complex_tensor(n: usize, seed: u64) -> Tensor {
        Tensor::from_slice(&values(2 * n, seed), &[n, 2]).unwrap()
    }

    /// O(n^2) DFT/IDFT over interleaved `[re, im]` pairs, independent of the
    /// Cooley-Tukey path: fresh angles per element, no twiddle recurrence.
    fn naive_dft(src: &[f64], inverse: bool) -> Vec<f64> {
        let n = src.len() / 2;
        let sign = if inverse { 1.0 } else { -1.0 };
        let scale = if inverse { 1.0 / (n as f64) } else { 1.0 };
        let mut dst = vec![0.0; src.len()];
        for k in 0..n {
            let mut sr = 0.0;
            let mut si = 0.0;
            for t in 0..n {
                let angle = sign * TAU * (k * t) as f64 / (n as f64);
                let (c, s) = (angle.cos(), angle.sin());
                sr += src[2 * t] * c - src[2 * t + 1] * s;
                si += src[2 * t] * s + src[2 * t + 1] * c;
            }
            dst[2 * k] = sr * scale;
            dst[2 * k + 1] = si * scale;
        }
        dst
    }

    fn assert_close(got: &[f64], want: &[f64], tol: f64, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() <= tol,
                "{what} element {i}: {g} vs {w} (tol {tol})"
            );
        }
    }

    /// Power-of-two sizes take the Cooley-Tukey path, the rest the direct
    /// DFT; both must match an independent naive reference.
    #[test]
    fn fft_matches_naive_dft_for_many_sizes() {
        for n in [
            0usize, 1, 2, 3, 4, 5, 7, 8, 9, 12, 16, 31, 32, 33, 64, 100, 127, 128, 129, 256,
        ] {
            let x = complex_tensor(n, 7);
            let src = x.to_vec::<f64>().unwrap();
            let got = x.fft().unwrap().to_vec::<f64>().unwrap();
            let want = naive_dft(&src, false);
            // the twiddle recurrence drifts a little for large power-of-two sizes
            let tol = if n.is_power_of_two() {
                1e-9 * (n as f64)
            } else {
                1e-9
            };
            assert_close(&got, &want, tol, &format!("fft n={n}"));
        }
    }

    #[test]
    fn ifft_matches_naive_inverse_dft() {
        for n in [0usize, 1, 2, 3, 5, 8, 9, 16, 33, 64, 100, 128] {
            let x = complex_tensor(n, 11);
            let src = x.to_vec::<f64>().unwrap();
            let got = x.ifft().unwrap().to_vec::<f64>().unwrap();
            let want = naive_dft(&src, true);
            let tol = if n.is_power_of_two() {
                1e-9 * (n as f64)
            } else {
                1e-9
            };
            assert_close(&got, &want, tol, &format!("ifft n={n}"));
        }
    }

    #[test]
    fn ifft_inverts_fft_roundtrip() {
        for n in [0usize, 1, 4, 7, 16, 33, 128] {
            let x = complex_tensor(n, 13);
            let src = x.to_vec::<f64>().unwrap();
            let back = x.fft().unwrap().ifft().unwrap().to_vec::<f64>().unwrap();
            assert_close(&back, &src, 1e-9, &format!("roundtrip n={n}"));
        }
    }

    #[test]
    fn fft_maps_batches_independently() {
        let n = 9;
        let rows: Vec<f64> = (0..2).flat_map(|b| values(2 * n, 100 + b)).collect();
        let x = Tensor::from_slice(&rows, &[2, n, 2]).unwrap();
        let got = x.fft().unwrap().to_vec::<f64>().unwrap();
        for b in 0..2 {
            let want = naive_dft(&rows[b * 2 * n..(b + 1) * 2 * n], false);
            assert_close(
                &got[b * 2 * n..(b + 1) * 2 * n],
                &want,
                1e-9,
                &format!("batch {b}"),
            );
        }
    }

    #[test]
    fn rfft_then_irfft_recovers_real_signal() {
        let x = Tensor::from_slice(&values(24, 17), &[24]).unwrap();
        let back = x.rfft().unwrap().irfft().unwrap().to_vec::<f64>().unwrap();
        assert_close(&back, &values(24, 17), 1e-9, "rfft/irfft");
    }

    #[test]
    fn fft2d_matches_separable_reference() {
        for (h, w) in [(4usize, 4usize), (3, 5)] {
            let x = Tensor::from_slice(&values(2 * h * w, 19), &[h, w, 2]).unwrap();
            let src = x.to_vec::<f64>().unwrap();
            // rows, then columns
            let mut tmp = vec![0.0; src.len()];
            for r in 0..h {
                let row = naive_dft(&src[r * 2 * w..(r + 1) * 2 * w], false);
                tmp[r * 2 * w..(r + 1) * 2 * w].copy_from_slice(&row);
            }
            let mut want = vec![0.0; src.len()];
            for c in 0..w {
                let col: Vec<f64> = (0..h)
                    .flat_map(|r| [tmp[(r * w + c) * 2], tmp[(r * w + c) * 2 + 1]])
                    .collect();
                let col_fft = naive_dft(&col, false);
                for r in 0..h {
                    want[(r * w + c) * 2] = col_fft[2 * r];
                    want[(r * w + c) * 2 + 1] = col_fft[2 * r + 1];
                }
            }
            let got = x.fft2d().unwrap().to_vec::<f64>().unwrap();
            assert_close(&got, &want, 1e-9, &format!("fft2d {h}x{w}"));
            let back = x
                .fft2d()
                .unwrap()
                .ifft2d()
                .unwrap()
                .to_vec::<f64>()
                .unwrap();
            assert_close(&back, &src, 1e-9, &format!("fft2d roundtrip {h}x{w}"));
        }
    }

    #[test]
    fn complex_mul_and_abs_match_hand_computed_values() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, -4.0], &[2, 2]).unwrap();
        let b = Tensor::from_slice(&[3.0, 4.0, 1.0, 1.0], &[2, 2]).unwrap();
        let got = a.complex_mul(&b).unwrap().to_vec::<f64>().unwrap();
        // (1+2i)(3+4i) = -5+10i, (3-4i)(1+i) = 7-i
        assert_close(&got, &[-5.0, 10.0, 7.0, -1.0], 0.0, "complex_mul");
        let mag = a.complex_abs().unwrap().to_vec::<f64>().unwrap();
        assert_close(&mag, &[5.0f64.sqrt(), 5.0], 1e-12, "complex_abs");
    }

    #[test]
    fn stft_frames_match_manual_framing() {
        let signal = values(16, 23);
        let x = Tensor::from_slice(&signal, &[16]).unwrap();
        let out = x.stft(8, 4, false).unwrap();
        assert_eq!(out.shape(), &[1usize, 3, 8, 2]);
        let got = out.to_vec::<f64>().unwrap();
        for (f, start) in [0usize, 4, 8].iter().enumerate() {
            let frame: Vec<f64> = signal[*start..*start + 8]
                .iter()
                .flat_map(|&v| [v, 0.0])
                .collect();
            let want = naive_dft(&frame, false);
            assert_close(
                &got[f * 16..(f + 1) * 16],
                &want,
                1e-9,
                &format!("frame {f}"),
            );
        }
    }

    #[test]
    fn stft_hann_window_differs_and_stays_finite() {
        let x = Tensor::from_slice(&values(16, 29), &[16]).unwrap();
        let plain = x.stft(8, 4, false).unwrap().to_vec::<f64>().unwrap();
        let hann = x.stft(8, 4, true).unwrap().to_vec::<f64>().unwrap();
        assert!(hann.iter().all(|v| v.is_finite()));
        let diff: f64 = hann
            .iter()
            .zip(plain.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-6, "hann window must change the spectrum");
    }

    #[test]
    fn fft_and_stft_reject_bad_shapes() {
        assert!(
            Tensor::from_slice(&[1.0, 2.0], &[2])
                .unwrap()
                .fft()
                .is_err()
        );
        assert!(
            Tensor::from_slice(&[1.0; 6], &[2, 3])
                .unwrap()
                .fft()
                .is_err()
        );
        assert!(
            Tensor::from_slice(&[1.0; 4], &[2, 2])
                .unwrap()
                .fft2d()
                .is_err()
        );
        assert!(
            Tensor::from_slice(&[1.0; 8], &[2, 2, 2])
                .unwrap()
                .ifft2d()
                .is_ok()
        );
        let short = Tensor::from_slice(&[1.0; 4], &[4]).unwrap();
        assert!(short.stft(8, 4, false).is_err());
        let ok = Tensor::from_slice(&[1.0; 16], &[16]).unwrap();
        assert!(ok.stft(8, 0, false).is_err());
        let bad_rank = Tensor::from_slice(&[1.0; 16], &[2, 2, 4]).unwrap();
        assert!(bad_rank.stft(4, 2, false).is_err());
    }

    /// The DFT is linear: check the autograd rule against central differences.
    #[test]
    fn fft_backward_matches_finite_differences() {
        for n in [7usize, 8] {
            let base = values(2 * n, 31);
            let x = Tensor::from_slice(&base, &[n, 2])
                .unwrap()
                .requires_grad_(true);
            x.fft().unwrap().sum().unwrap().backward().unwrap();
            let analytic = x.grad().unwrap().to_vec::<f64>().unwrap();
            let e = 1e-6;
            for i in 0..base.len() {
                let mut up = base.clone();
                up[i] += e;
                let mut dn = base.clone();
                dn[i] -= e;
                let lp = Tensor::from_slice(&up, &[n, 2])
                    .unwrap()
                    .fft()
                    .unwrap()
                    .sum()
                    .unwrap()
                    .item::<f64>()
                    .unwrap();
                let lm = Tensor::from_slice(&dn, &[n, 2])
                    .unwrap()
                    .fft()
                    .unwrap()
                    .sum()
                    .unwrap()
                    .item::<f64>()
                    .unwrap();
                let numeric = (lp - lm) / (2.0 * e);
                assert!(
                    (analytic[i] - numeric).abs() <= 1e-5,
                    "n={n} element {i}: {} vs {numeric}",
                    analytic[i]
                );
            }
        }
    }

    /// Hand-computed DFT values (independent of any FFT code path, verifiable by hand
    /// from the DFT sum): impulse -> ones, and DFT([1,2,3,4]) = [10, -2+2i, -2, -2-2i].
    #[test]
    fn fft_matches_hand_computed_dft_values() {
        let imp = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[4, 2]).unwrap();
        let got = imp.fft().unwrap().to_vec::<f64>().unwrap();
        assert_close(
            &got,
            &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            1e-12,
            "impulse",
        );
        let x = Tensor::from_slice(&[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0], &[4, 2]).unwrap();
        let got = x.fft().unwrap().to_vec::<f64>().unwrap();
        assert_close(
            &got,
            &[10.0, 0.0, -2.0, 2.0, -2.0, 0.0, -2.0, -2.0],
            1e-12,
            "dft 1..4",
        );
        // ifft inverts to the original samples.
        let back = x.fft().unwrap().ifft().unwrap().to_vec::<f64>().unwrap();
        assert_close(
            &back,
            &[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0],
            1e-12,
            "dft 1..4 roundtrip",
        );
    }

    /// The DC bin of a spectrum equals the sum of the windowed signal. For an all-ones
    /// signal under the periodic Hann window, that sum is exactly n_fft / 2 (each window
    /// value averages 1/2), which pins both the window formula and its normalization
    /// without duplicating the formula.
    #[test]
    fn stft_hann_dc_bin_equals_window_sum() {
        let x = Tensor::from_slice(&[1.0f64; 8], &[8]).unwrap();
        let out = x.stft(8, 8, true).unwrap();
        assert_eq!(out.shape(), &[1usize, 1, 8, 2]);
        let got = out.to_vec::<f64>().unwrap();
        assert_close(&got[0..2], &[4.0, 0.0], 1e-12, "hann dc bin");
    }

    /// Like torch (`torch.fft.fft` on an integer tensor raises), the FFT rejects
    /// non-floating dtypes instead of silently truncating the spectrum back to int.
    #[test]
    fn fft_rejects_non_float_dtypes() {
        use crate::dtype::{DType, bf16, f16};
        for dtype in [DType::U8, DType::I32, DType::I64, DType::Bool] {
            let x = Tensor::zeros(&[4, 2], dtype, &crate::backend::Device::Cpu).unwrap();
            assert!(x.fft().is_err(), "fft must reject {dtype:?}");
            assert!(x.ifft().is_err(), "ifft must reject {dtype:?}");
        }
        let f16x = Tensor::from_slice(&[f16::from_bits(0x3C00); 8], &[4, 2]).unwrap();
        assert!(f16x.fft().is_ok());
        let bf16x = Tensor::from_slice(&[bf16::from_bits(0x3F80); 8], &[4, 2]).unwrap();
        assert!(bf16x.ifft().is_ok());
        let f32x = Tensor::zeros(&[4, 2], DType::F32, &crate::backend::Device::Cpu).unwrap();
        assert!(f32x.fft().is_ok());
        let real = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
        assert!(real.rfft().is_ok());
    }

    /// `complex_mul`/`complex_abs` promise complex-pair semantics, so a trailing dim
    /// other than 2 must error rather than silently operate on the wrong lanes.
    #[test]
    fn complex_ops_reject_non_pair_trailing_dim() {
        let a = Tensor::from_slice(&[1.0f64; 12], &[4, 3]).unwrap();
        let b = Tensor::from_slice(&[2.0f64; 12], &[4, 3]).unwrap();
        assert!(a.complex_mul(&b).is_err());
        assert!(a.complex_abs().is_err());
        let ok_a = Tensor::from_slice(&[1.0f64; 8], &[4, 2]).unwrap();
        let ok_b = Tensor::from_slice(&[2.0f64; 8], &[4, 2]).unwrap();
        assert!(ok_a.complex_mul(&ok_b).is_ok());
        assert!(ok_a.complex_abs().is_ok());
    }

    /// DELIBERATE torch divergence, pinned: Rust `Tensor::rfft` returns the FULL
    /// two-sided `[..., N, 2]` spectrum (plain `fft` of a zero-imag signal), NOT the
    /// one-sided `[..., N/2+1]` of `torch.fft.rfft`. Rationale: the Rust-side `stft`
    /// and mel frontend consume the full spectrum and slice what they need, while the
    /// Python `oxideon.fft.rfft` shim is numpy-backed with true torch semantics — so no
    /// Python user ever sees this shape.
    #[test]
    fn rfft_returns_full_two_sided_spectrum() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let got = x.rfft().unwrap();
        assert_eq!(got.shape(), &[4usize, 2]);
        let flat = got.to_vec::<f64>().unwrap();
        assert_close(
            &flat,
            &[10.0, 0.0, -2.0, 2.0, -2.0, 0.0, -2.0, -2.0],
            1e-12,
            "rfft full spectrum",
        );
        // ...and irfft inverts it back to the real signal.
        let back = got.irfft().unwrap().to_vec::<f64>().unwrap();
        assert_close(&back, &[1.0, 2.0, 3.0, 4.0], 1e-12, "irfft");
    }
}
