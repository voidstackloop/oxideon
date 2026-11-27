//! Signal-processing helpers from `torch.fft` and `torch.signal`: window functions ([`hann_window`], [`hamming_window`],
//! [`blackman_window`], [`bartlett_window`], [`kaiser_window`]), frequency grids ([`fftfreq`], [`rfftfreq`]),
//! [`fftshift`] / [`ifftshift`], N-dimensional FFTs with normalization modes on [`Complex`], a full-featured
//! [`stft`] / [`istft`] pair (centering, custom window, normalization, one-sided spectra) and the discrete cosine
//! transform ([`dct`], [`idct`]). Everything is differentiable.

use crate::complex::Complex;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::layout::normalize_axis;
use crate::tensor::Tensor;
use std::f64::consts::{PI, TAU};

fn window_from(n: usize, periodic: bool, dtype: DType, f: impl Fn(f64) -> f64) -> Result<Tensor> {
    if n == 0 {
        return Tensor::zeros(&[0], dtype, &crate::backend::Device::Cpu);
    }
    if n == 1 {
        return Tensor::ones(&[1], dtype, &crate::backend::Device::Cpu);
    }
    let denom = if periodic { n as f64 } else { (n - 1) as f64 };
    let v: Vec<f64> = (0..n).map(|i| f(i as f64 / denom)).collect();
    Tensor::from_slice(&v, &[n])?.to_dtype(dtype)
}

/// Hann window `0.5 (1 - cos(2 pi k / N))` (`periodic` = DFT-even, the default of `torch.hann_window`).
pub fn hann_window(n: usize, periodic: bool, dtype: DType) -> Result<Tensor> {
    window_from(n, periodic, dtype, |x| 0.5 - 0.5 * (TAU * x).cos())
}

/// Hamming window `0.54 - 0.46 cos(2 pi k / N)`.
pub fn hamming_window(n: usize, periodic: bool, dtype: DType) -> Result<Tensor> {
    window_from(n, periodic, dtype, |x| 0.54 - 0.46 * (TAU * x).cos())
}

/// Blackman window `0.42 - 0.5 cos(2 pi k / N) + 0.08 cos(4 pi k / N)`.
pub fn blackman_window(n: usize, periodic: bool, dtype: DType) -> Result<Tensor> {
    window_from(n, periodic, dtype, |x| 0.42 - 0.5 * (TAU * x).cos() + 0.08 * (2.0 * TAU * x).cos())
}

/// Triangular (Bartlett) window.
pub fn bartlett_window(n: usize, periodic: bool, dtype: DType) -> Result<Tensor> {
    window_from(n, periodic, dtype, |x| 1.0 - (2.0 * x - 1.0).abs())
}

/// Kaiser window with shape parameter `beta`.
pub fn kaiser_window(n: usize, periodic: bool, beta: f64, dtype: DType) -> Result<Tensor> {
    if n <= 1 {
        return window_from(n, periodic, dtype, |_| 1.0);
    }
    let denom = if periodic { n as f64 } else { (n - 1) as f64 };
    let args: Vec<f64> = (0..n)
        .map(|i| {
            let r = 2.0 * i as f64 / denom - 1.0;
            beta * (1.0 - r * r).max(0.0).sqrt()
        })
        .collect();
    let num = Tensor::from_slice(&args, &[n])?.i0()?;
    let den = Tensor::from_slice(&[beta], &[1])?.i0()?;
    num.div(&den)?.to_dtype(dtype)
}

/// Sample frequencies of an `n`-point FFT with sample spacing `d` (`torch.fft.fftfreq`).
pub fn fftfreq(n: usize, d: f64, dtype: DType) -> Result<Tensor> {
    let v: Vec<f64> = (0..n)
        .map(|i| {
            let k = if i < n.div_ceil(2) { i as f64 } else { i as f64 - n as f64 };
            k / (n as f64 * d)
        })
        .collect();
    Tensor::from_slice(&v, &[n])?.to_dtype(dtype)
}

/// Frequencies of the non-negative half of a real signal's spectrum (`torch.fft.rfftfreq`).
pub fn rfftfreq(n: usize, d: f64, dtype: DType) -> Result<Tensor> {
    let v: Vec<f64> = (0..=n / 2).map(|i| i as f64 / (n as f64 * d)).collect();
    let len = v.len();
    Tensor::from_slice(&v, &[len])?.to_dtype(dtype)
}

fn shift_dims(x: &Tensor, dims: &[isize], inverse: bool) -> Result<Tensor> {
    let all: Vec<isize> = (0..x.ndim() as isize).collect();
    let dims = if dims.is_empty() { &all } else { dims };
    let mut out = x.clone();
    for &d in dims {
        let ax = normalize_axis("fftshift", d, x.ndim())?;
        let n = x.shape()[ax] as isize;
        let by = if inverse { -(n / 2) } else { n / 2 };
        out = out.roll(by, ax as isize)?;
    }
    Ok(out)
}

/// Moves the zero-frequency component to the centre (`torch.fft.fftshift`); `dims` empty = all dimensions.
pub fn fftshift(x: &Tensor, dims: &[isize]) -> Result<Tensor> {
    shift_dims(x, dims, false)
}

/// Inverse of [`fftshift`].
pub fn ifftshift(x: &Tensor, dims: &[isize]) -> Result<Tensor> {
    shift_dims(x, dims, true)
}

/// Scaling applied to FFT results (`torch.fft`'s `norm` argument).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FftNorm {
    /// No scaling forward, `1/n` inverse.
    Backward,
    /// `1/sqrt(n)` both ways (unitary).
    Ortho,
    /// `1/n` forward, none inverse.
    Forward,
}

impl FftNorm {
    fn factor(self, n: usize, inverse: bool) -> f64 {
        match (self, inverse) {
            (FftNorm::Ortho, _) => 1.0 / (n as f64).sqrt(),
            (FftNorm::Backward, true) | (FftNorm::Forward, false) => 1.0 / n as f64,
            _ => 1.0,
        }
    }
}

impl Complex {
    /// FFT along one logical dimension with a normalization mode. The library's FFT already scales the inverse by
    /// `1/n`; this adjusts that to the requested mode.
    pub fn fft_dim(&self, dim: isize, inverse: bool, norm: FftNorm) -> Result<Complex> {
        let nd = self.re.ndim();
        let ax = normalize_axis("fft_dim", dim, nd)?;
        let n = self.re.shape()[ax];
        let moved = Complex::new(self.re.transpose(ax as isize, -1)?.contiguous()?, self.im.transpose(ax as isize, -1)?.contiguous()?)?;
        let y = if inverse { moved.ifft()? } else { moved.fft()? };
        // base scaling: forward 1, inverse 1/n
        let base = if inverse { 1.0 / n as f64 } else { 1.0 };
        let scale = norm.factor(n, inverse) / base;
        let y = if (scale - 1.0).abs() > 1e-15 { y.scale(scale)? } else { y };
        Complex::new(y.re.transpose(ax as isize, -1)?, y.im.transpose(ax as isize, -1)?)
    }

    /// N-dimensional FFT over `dims` (all dimensions when empty), `torch.fft.fftn`.
    pub fn fftn(&self, dims: &[isize], norm: FftNorm) -> Result<Complex> {
        self.nd_transform(dims, false, norm)
    }

    /// Inverse of [`fftn`](Self::fftn), `torch.fft.ifftn`.
    pub fn ifftn(&self, dims: &[isize], norm: FftNorm) -> Result<Complex> {
        self.nd_transform(dims, true, norm)
    }

    fn nd_transform(&self, dims: &[isize], inverse: bool, norm: FftNorm) -> Result<Complex> {
        let all: Vec<isize> = (0..self.re.ndim() as isize).collect();
        let dims = if dims.is_empty() { &all } else { dims };
        let mut z = self.clone();
        for &d in dims {
            z = z.fft_dim(d, inverse, norm)?;
        }
        Ok(z)
    }
}

/// Discrete cosine transform (type II, or III when `inverse`) along the last axis (`scipy.fft.dct`). With `ortho`
/// the transform is orthonormal.
fn dct_matrix(n: usize, ortho: bool, inverse: bool) -> Vec<f64> {
    // forward: X_k = 2 sum_n x_n cos(pi k (2n + 1) / (2N)); the orthonormal scaling makes the matrix orthogonal
    let mut m = vec![0.0; n * n];
    for k in 0..n {
        for i in 0..n {
            let c = (PI * k as f64 * (2.0 * i as f64 + 1.0) / (2.0 * n as f64)).cos();
            let scale = if ortho {
                if k == 0 { (1.0 / n as f64).sqrt() } else { (2.0 / n as f64).sqrt() }
            } else {
                2.0
            };
            // rows index the output frequency for the forward transform
            if inverse {
                m[i * n + k] = c * if ortho { scale } else if k == 0 { 0.5 / n as f64 } else { 1.0 / n as f64 };
            } else {
                m[k * n + i] = c * scale;
            }
        }
    }
    m
}

/// DCT-II along the last axis (`ortho` selects the orthonormal scaling).
pub fn dct(x: &Tensor, ortho: bool) -> Result<Tensor> {
    let n = *x.shape().last().ok_or_else(|| Error::Msg("dct: needs at least one dimension".into()))?;
    let m = Tensor::from_slice(&dct_matrix(n, ortho, false), &[n, n])?.to_dtype(x.dtype())?.to_device(&x.device())?;
    x.matmul(&m.t()?)
}

/// Inverse of [`dct`] (a scaled DCT-III).
pub fn idct(x: &Tensor, ortho: bool) -> Result<Tensor> {
    let n = *x.shape().last().ok_or_else(|| Error::Msg("idct: needs at least one dimension".into()))?;
    let m = Tensor::from_slice(&dct_matrix(n, ortho, true), &[n, n])?.to_dtype(x.dtype())?.to_device(&x.device())?;
    x.matmul(&m.t()?)
}

/// Options of [`stft`] / [`istft`].
#[derive(Clone)]
pub struct StftOptions {
    pub n_fft: usize,
    pub hop_length: usize,
    /// Length of the analysis window (`<= n_fft`, zero-padded symmetrically). Defaults to `n_fft`.
    pub win_length: Option<usize>,
    /// Analysis window of `win_length` samples; a rectangular window when `None`.
    pub window: Option<Tensor>,
    /// Pad the signal by `n_fft / 2` on both sides (reflect) so frames are centred on multiples of `hop_length`.
    pub center: bool,
    /// Scale by `1/sqrt(n_fft)`.
    pub normalized: bool,
    /// Return only the `n_fft / 2 + 1` non-negative frequencies.
    pub onesided: bool,
}

impl StftOptions {
    pub fn new(n_fft: usize, hop_length: usize) -> Self {
        Self { n_fft, hop_length, win_length: None, window: None, center: true, normalized: false, onesided: true }
    }

    fn full_window(&self, dtype: DType, device: &crate::backend::Device) -> Result<Tensor> {
        let wl = self.win_length.unwrap_or(self.n_fft);
        if wl > self.n_fft {
            return Err(Error::Msg("stft: win_length must not exceed n_fft".into()));
        }
        let w = match &self.window {
            Some(w) => {
                if w.shape() != [wl] {
                    return Err(Error::Msg(format!("stft: window must have win_length = {wl} samples")));
                }
                w.to_dtype(dtype)?
            }
            None => Tensor::ones(&[wl], dtype, device)?,
        };
        let left = (self.n_fft - wl) / 2;
        w.pad(&[(left, self.n_fft - wl - left)], 0.0)
    }
}

/// Short-time Fourier transform of `[L]` or `[B, L]` into a complex spectrogram `[B, freq, frames]` (or
/// `[freq, frames]` for an unbatched signal), like `torch.stft(..., return_complex=True)`.
pub fn stft(x: &Tensor, o: &StftOptions) -> Result<Complex> {
    let unbatched = x.ndim() == 1;
    let x = if unbatched { x.unsqueeze(0)? } else if x.ndim() == 2 { x.clone() } else { return Err(Error::Msg("stft: expects [L] or [B, L]".into())) };
    if o.hop_length == 0 || o.n_fft == 0 {
        return Err(Error::Msg("stft: n_fft and hop_length must be positive".into()));
    }
    let x = if o.center {
        let p = o.n_fft / 2;
        crate::nn::functional_ext::pad_mode(&x, &[(p, p)], crate::nn::functional_ext::PadMode::Reflect)?
    } else {
        x
    };
    let len = x.shape()[1];
    if len < o.n_fft {
        return Err(Error::Msg("stft: signal shorter than n_fft".into()));
    }
    let w = o.full_window(x.dtype(), &x.device())?;
    let frames: Vec<Tensor> = (0..=(len - o.n_fft) / o.hop_length)
        .map(|f| x.narrow(1, f * o.hop_length, o.n_fft)?.mul(&w))
        .collect::<Result<_>>()?;
    let framed = Tensor::stack(&frames, 1)?; // [B, frames, n_fft]
    let mut spec = Complex::rfft(&framed)?;
    if o.onesided {
        let k = o.n_fft / 2 + 1;
        spec = Complex::new(spec.re.narrow(-1, 0, k)?, spec.im.narrow(-1, 0, k)?)?;
    }
    if o.normalized {
        spec = spec.scale(1.0 / (o.n_fft as f64).sqrt())?;
    }
    let spec = Complex::new(spec.re.transpose(1, 2)?.contiguous()?, spec.im.transpose(1, 2)?.contiguous()?)?; // [B, freq, frames]
    if unbatched {
        Complex::new(spec.re.squeeze_axis(0)?, spec.im.squeeze_axis(0)?)
    } else {
        Ok(spec)
    }
}

/// Inverse short-time Fourier transform by windowed overlap-add (`torch.istft`). `length` trims or pads the output.
pub fn istft(spec: &Complex, o: &StftOptions, length: Option<usize>) -> Result<Tensor> {
    let unbatched = spec.re.ndim() == 2;
    let spec = if unbatched { Complex::new(spec.re.unsqueeze(0)?, spec.im.unsqueeze(0)?)? } else { spec.clone() };
    let (b, freq, frames) = (spec.re.shape()[0], spec.re.shape()[1], spec.re.shape()[2]);
    let n = o.n_fft;
    let mut s = Complex::new(spec.re.transpose(1, 2)?.contiguous()?, spec.im.transpose(1, 2)?.contiguous()?)?; // [B, frames, freq]
    if o.normalized {
        s = s.scale((n as f64).sqrt())?;
    }
    if o.onesided {
        if freq != n / 2 + 1 {
            return Err(Error::Msg(format!("istft: a one-sided spectrum needs n_fft/2 + 1 = {} bins, got {freq}", n / 2 + 1)));
        }
        // rebuild the conjugate-symmetric upper half: bins n - k = conj(bin k) for k = 1 .. (n - 1) / 2
        let mirror: Vec<i64> = (1..=(n - 1) / 2).rev().map(|k| k as i64).collect();
        if !mirror.is_empty() {
            let idx = Tensor::from_slice_on(&mirror, &[mirror.len()], &s.re.device())?;
            let upper = Complex::new(s.re.index_select(-1, &idx)?, s.im.index_select(-1, &idx)?.neg()?)?;
            s = Complex::new(Tensor::cat(&[s.re.clone(), upper.re], -1)?, Tensor::cat(&[s.im.clone(), upper.im], -1)?)?;
        }
    } else if freq != n {
        return Err(Error::Msg(format!("istft: a two-sided spectrum needs n_fft = {n} bins, got {freq}")));
    }
    let time = s.ifft()?.re; // [B, frames, n]
    let w = o.full_window(time.dtype(), &time.device())?;
    let windowed = time.mul(&w)?;
    let total = n + o.hop_length * (frames - 1);
    // overlap-add through index_add over a flat time axis
    let mut idx = Vec::with_capacity(frames * n);
    for f in 0..frames {
        idx.extend((0..n).map(|i| (f * o.hop_length + i) as i64));
    }
    let idx_t = Tensor::from_slice_on(&idx, &[idx.len()], &time.device())?;
    let flat = windowed.reshape(&[b, frames * n])?.transpose(0, 1)?.contiguous()?; // [frames*n, B]
    let signal = Tensor::zeros(&[total, b], time.dtype(), &time.device())?.index_add(0, &idx_t, &flat)?.transpose(0, 1)?; // [B, total]
    let w2 = w.square()?.reshape(&[1, n])?.broadcast_to(&[frames, n])?.contiguous()?.reshape(&[frames * n, 1])?;
    let env = Tensor::zeros(&[total, 1], time.dtype(), &time.device())?.index_add(0, &idx_t, &w2)?.transpose(0, 1)?; // [1, total]
    let start = if o.center { n / 2 } else { 0 };
    let stop = match length {
        Some(l) => start + l,
        None => if o.center { total - n / 2 } else { total },
    };
    let stop = stop.min(total);
    let sig = signal.narrow(1, start, stop - start)?;
    let env = env.narrow(1, start, stop - start)?;
    let out = sig.div(&env.clamp(1e-11, f64::INFINITY)?)?;
    let out = match length {
        Some(l) if out.shape()[1] < l => out.pad(&[(0, 0), (0, l - out.shape()[1])], 0.0)?,
        _ => out,
    };
    if unbatched { out.squeeze_axis(0) } else { Ok(out) }
}

/// Fast Griffin-Lim phase reconstruction (Perraudin et al., 2013): the waveform whose STFT magnitude best matches `mag`
/// (`[freq, frames]` or `[B, freq, frames]`, one-sided, made with the same `o`). Starts from a random phase (`seed`) and
/// alternates the projection onto consistent spectrograms (`stft(istft(c))`, accelerated with `momentum`) with re-imposing the
/// magnitude. Returns `[(frames - 1) * hop_length]` samples (`[B, ..]` when batched).
pub fn griffin_lim(mag: &Tensor, o: &StftOptions, n_iter: usize, momentum: f64, seed: u64) -> Result<Tensor> {
    let unbatched = mag.ndim() == 2;
    let mag = if unbatched { mag.unsqueeze(0)? } else if mag.ndim() == 3 { mag.clone() } else { return Err(Error::Msg("griffin_lim: expects [freq, frames] or [B, freq, frames]".into())) };
    let (n, frames) = (mag.shape()[0] * mag.shape()[1] * mag.shape()[2], mag.shape()[2]);
    if frames == 0 || n == 0 {
        return Err(Error::Msg("griffin_lim: empty spectrogram".into()));
    }
    let length = (frames - 1) * o.hop_length;
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        // splitmix64 -> uniform [0, 1)
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    let (cos, sin): (Vec<f64>, Vec<f64>) = (0..n).map(|_| { let a = next() * TAU; (a.cos(), a.sin()) }).unzip();
    let dev = mag.device();
    let shape = mag.shape().to_vec();
    let mk = |v: Vec<f64>| -> Result<Tensor> { Tensor::from_slice_on(&v, &shape, &dev)?.to_dtype(mag.dtype()) };
    let mut c = Complex::new(mag.mul(&mk(cos)?)?, mag.mul(&mk(sin)?)?)?;
    let mut prev = c.clone();
    for _ in 0..n_iter {
        let t = stft(&istft(&c, o, Some(length))?, o)?;
        let extrapolated = t.add(&t.sub(&prev)?.scale(momentum)?)?;
        let inv = extrapolated.abs()?.clamp(1e-12, f64::INFINITY)?.recip()?;
        c = extrapolated.mul_real(&inv.mul(&mag)?)?;
        prev = t;
    }
    let out = istft(&c, o, Some(length))?;
    if unbatched { out.squeeze_axis(0) } else { Ok(out) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(t: &Tensor) -> Vec<f64> {
        t.to_vec::<f64>().unwrap()
    }
    fn near(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len(), "{a:?} vs {b:?}");
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < tol, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn griffin_lim_recovers_a_consistent_spectrogram() {
        let n = 2048usize;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.21).sin() + 0.5 * (i as f64 * 0.047).sin()).collect();
        let x = Tensor::from_slice_on(&x, &[n], &crate::backend::Device::Cpu).unwrap();
        let mut o = StftOptions::new(256, 64);
        o.window = Some(hann_window(256, true, DType::F64).unwrap());
        let mag = stft(&x, &o).unwrap().abs().unwrap();
        let y = griffin_lim(&mag, &o, 60, 0.99, 7).unwrap();
        let again = stft(&y, &o).unwrap().abs().unwrap();
        let (a, b) = (v(&mag), v(&again));
        let err: f64 = a.iter().zip(&b).map(|(p, q)| (p - q) * (p - q)).sum::<f64>().sqrt() / a.iter().map(|p| p * p).sum::<f64>().sqrt();
        assert!(err < 0.2, "spectral convergence {err}");
        assert_eq!(y.shape(), &[(mag.shape()[1] - 1) * 64]);
    }

    #[test]
    fn windows() {
        near(&v(&hann_window(4, true, DType::F64).unwrap()), &[0.0, 0.5, 1.0, 0.5], 1e-12);
        near(&v(&hann_window(5, false, DType::F64).unwrap()), &[0.0, 0.5, 1.0, 0.5, 0.0], 1e-12);
        near(&v(&hamming_window(3, false, DType::F64).unwrap()), &[0.08, 1.0, 0.08], 1e-12);
        near(&v(&blackman_window(3, false, DType::F64).unwrap()), &[0.0, 1.0, 0.0], 1e-12);
        near(&v(&bartlett_window(5, false, DType::F64).unwrap()), &[0.0, 0.5, 1.0, 0.5, 0.0], 1e-12);
        let k = v(&kaiser_window(5, false, 8.0, DType::F64).unwrap());
        assert!((k[2] - 1.0).abs() < 1e-12 && k[0] < 0.01 && (k[1] - k[3]).abs() < 1e-12);
        assert_eq!(v(&hann_window(1, true, DType::F64).unwrap()), vec![1.0]);
    }

    #[test]
    fn frequencies_and_shifts() {
        near(&v(&fftfreq(5, 1.0, DType::F64).unwrap()), &[0.0, 0.2, 0.4, -0.4, -0.2], 1e-12);
        near(&v(&fftfreq(4, 0.5, DType::F64).unwrap()), &[0.0, 0.5, -1.0, -0.5], 1e-12);
        near(&v(&rfftfreq(6, 1.0, DType::F64).unwrap()), &[0.0, 1.0 / 6.0, 2.0 / 6.0, 0.5], 1e-12);
        let x = Tensor::from_slice(&[0.0f64, 1.0, 2.0, 3.0, 4.0], &[5]).unwrap();
        near(&v(&fftshift(&x, &[]).unwrap()), &[3.0, 4.0, 0.0, 1.0, 2.0], 1e-12);
        near(&v(&ifftshift(&fftshift(&x, &[]).unwrap(), &[]).unwrap()), &v(&x), 1e-12);
        let e = Tensor::from_slice(&[0.0f64, 1.0, 2.0, 3.0], &[4]).unwrap();
        near(&v(&fftshift(&e, &[0]).unwrap()), &[2.0, 3.0, 0.0, 1.0], 1e-12);
    }

    #[test]
    fn nd_fft_and_norms() {
        let re = Tensor::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]).unwrap();
        let z = Complex::from_real(re.clone()).unwrap();
        // fftn over both dims equals fft2d
        let a = z.fftn(&[], FftNorm::Backward).unwrap();
        let b = z.fft2d().unwrap();
        near(&v(&a.re), &v(&b.re), 1e-9);
        near(&v(&a.im), &v(&b.im), 1e-9);
        // round trip in every norm mode, and ortho preserves energy
        for norm in [FftNorm::Backward, FftNorm::Ortho, FftNorm::Forward] {
            let back = z.fftn(&[], norm).unwrap().ifftn(&[], norm).unwrap();
            near(&v(&back.re), &v(&re), 1e-9);
        }
        let e0: f64 = v(&z.abs_sq().unwrap()).iter().sum();
        let e1: f64 = v(&z.fftn(&[], FftNorm::Ortho).unwrap().abs_sq().unwrap()).iter().sum();
        assert!((e0 - e1).abs() < 1e-8);
        // forward norm divides by n: the DC bin is the mean
        let dc = z.fft_dim(1, false, FftNorm::Forward).unwrap();
        assert!((v(&dc.re)[0] - 2.5).abs() < 1e-9);
    }

    #[test]
    fn cosine_transform() {
        let x = Tensor::from_slice(&[1.0f64, 2.0, 0.5, -1.0], &[4]).unwrap();
        // definition (unnormalized, scipy type II): X_k = 2 sum x_n cos(pi k (2n+1) / 2N)
        let naive: Vec<f64> = (0..4)
            .map(|k| 2.0 * (0..4).map(|n| v(&x)[n] * (PI * k as f64 * (2.0 * n as f64 + 1.0) / 8.0).cos()).sum::<f64>())
            .collect();
        near(&v(&dct(&x, false).unwrap()), &naive, 1e-9);
        near(&v(&idct(&dct(&x, false).unwrap(), false).unwrap()), &v(&x), 1e-9);
        near(&v(&idct(&dct(&x, true).unwrap(), true).unwrap()), &v(&x), 1e-9);
        // orthonormal: energy preserved
        let e: f64 = v(&dct(&x, true).unwrap()).iter().map(|a| a * a).sum();
        assert!((e - v(&x).iter().map(|a| a * a).sum::<f64>()).abs() < 1e-9);
        // gradient flows
        let p = x.requires_grad_(true);
        dct(&p, true).unwrap().square().unwrap().sum().unwrap().backward().unwrap();
        near(&v(&p.grad().unwrap()), &v(&p.detach().mul_scalar(2.0).unwrap()), 1e-9); // orthogonal: d||Qx||^2 = 2x
    }

    #[test]
    fn stft_inverts() {
        let sig: Vec<f64> = (0..256).map(|i| (i as f64 * 0.21).sin() + 0.3 * (i as f64 * 0.9).cos()).collect();
        let x = Tensor::from_slice(&sig, &[256]).unwrap();
        for (center, onesided, normalized) in [(true, true, false), (true, false, true), (true, true, true)] {
            let mut o = StftOptions::new(32, 8);
            o.window = Some(hann_window(32, true, DType::F64).unwrap());
            o.center = center;
            o.onesided = onesided;
            o.normalized = normalized;
            let s = stft(&x, &o).unwrap();
            assert_eq!(s.re.shape(), [if onesided { 17 } else { 32 }, 1 + 256 / 8]);
            let y = istft(&s, &o, Some(256)).unwrap();
            near(&v(&y), &sig, 1e-9);
        }
        // batched, shorter window zero-padded to n_fft
        let xb = Tensor::stack(&[x.clone(), x.mul_scalar(0.5).unwrap()], 0).unwrap();
        let mut o = StftOptions::new(32, 16);
        o.win_length = Some(24);
        o.window = Some(hann_window(24, true, DType::F64).unwrap());
        let s = stft(&xb, &o).unwrap();
        assert_eq!(s.re.shape()[0], 2);
        near(&v(&istft(&s, &o, Some(256)).unwrap().narrow(0, 1, 1).unwrap().reshape(&[256]).unwrap()), &sig.iter().map(|a| a * 0.5).collect::<Vec<_>>(), 1e-9);
        // a pure tone peaks in the right bin
        let tone: Vec<f64> = (0..128).map(|i| (TAU * 4.0 * i as f64 / 32.0).sin()).collect();
        let mut o = StftOptions::new(32, 32);
        o.center = false;
        let s = stft(&Tensor::from_slice(&tone, &[128]).unwrap(), &o).unwrap();
        let mag = v(&s.abs().unwrap());
        let frame0: Vec<f64> = (0..17).map(|k| mag[k * 4]).collect();
        let peak = frame0.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        assert_eq!(peak, 4);
    }
}
