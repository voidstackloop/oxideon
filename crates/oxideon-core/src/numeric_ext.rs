//! NumPy / SciPy style numerical helpers: piecewise-linear [`interp`], polynomials ([`polyval`], [`polyfit`],
//! [`polyder`], [`polyint`], [`roots`]), 1-D [`convolve`] / [`correlate`] with `full` / `same` / `valid` modes and an
//! FFT-based [`fftconvolve`], and matrix functions ([`sqrtm`] by the Denman-Beavers iteration, [`logm`] by inverse
//! scaling and squaring, [`sqrtm_spd`] / [`funm_spd`] through the eigendecomposition). Polynomial coefficients are
//! ordered from the highest power down, like `numpy.polyval`.

use crate::complex::Complex;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Linear interpolation of `(xp, fp)` at the points `x` (`numpy.interp`); outside `xp` the end values are used.
/// `xp` must be increasing. Differentiable with respect to `fp` (and to `x` between the knots).
pub fn interp(x: &Tensor, xp: &Tensor, fp: &Tensor) -> Result<Tensor> {
    if xp.ndim() != 1 || fp.shape() != xp.shape() || xp.numel() == 0 {
        return Err(Error::Msg("interp: xp and fp must be 1-D arrays of the same non-zero length".into()));
    }
    let knots = xp.to_vec::<f64>()?;
    if knots.windows(2).any(|w| w[1] <= w[0]) {
        return Err(Error::Msg("interp: xp must be strictly increasing".into()));
    }
    let n = knots.len();
    if n == 1 {
        return fp.broadcast_to(x.shape())?.contiguous();
    }
    let xs = x.to_vec::<f64>()?;
    let (mut lo, mut hi, mut w) = (Vec::with_capacity(xs.len()), Vec::with_capacity(xs.len()), Vec::with_capacity(xs.len()));
    for &v in &xs {
        let i = knots.partition_point(|&k| k <= v).clamp(1, n - 1);
        let (a, b) = (knots[i - 1], knots[i]);
        lo.push((i - 1) as i64);
        hi.push(i as i64);
        w.push(((v - a) / (b - a)).clamp(0.0, 1.0));
    }
    let dev = x.device();
    let (f_lo, f_hi) = (
        fp.index_select(0, &Tensor::from_slice_on(&lo, &[lo.len()], &dev)?)?,
        fp.index_select(0, &Tensor::from_slice_on(&hi, &[hi.len()], &dev)?)?,
    );
    let wt = Tensor::from_slice_on(&w, &[w.len()], &dev)?.to_dtype(fp.dtype())?;
    f_lo.mul(&wt.neg()?.add_scalar(1.0)?)?.add(&f_hi.mul(&wt)?)?.reshape(x.shape())
}

/// Evaluates the polynomial with `coeffs` (highest power first) at `x` by Horner's rule.
pub fn polyval(coeffs: &Tensor, x: &Tensor) -> Result<Tensor> {
    if coeffs.ndim() != 1 || coeffs.numel() == 0 {
        return Err(Error::Msg("polyval: coeffs must be a non-empty 1-D tensor".into()));
    }
    let mut acc = x.zeros_like()?.add(&coeffs.narrow(0, 0, 1)?.reshape(&[])?.to_dtype(x.dtype())?)?;
    for i in 1..coeffs.numel() {
        acc = acc.mul(x)?.add(&coeffs.narrow(0, i, 1)?.reshape(&[])?.to_dtype(x.dtype())?)?;
    }
    Ok(acc)
}

/// Least-squares polynomial fit of the given `degree` to `(x, y)` (`numpy.polyfit`), highest power first.
pub fn polyfit(x: &Tensor, y: &Tensor, degree: usize) -> Result<Tensor> {
    if x.ndim() != 1 || x.shape() != y.shape() || x.numel() <= degree {
        return Err(Error::Msg(format!("polyfit: need 1-D x and y of equal length greater than the degree {degree}")));
    }
    let cols: Vec<Tensor> = (0..=degree).rev().map(|p| x.powf(p as f64)).collect::<Result<_>>()?;
    let vander = Tensor::stack(&cols, 1)?; // [n, degree + 1]
    vander.lstsq(y, 1e-12)
}

/// Derivative coefficients of a polynomial.
pub fn polyder(coeffs: &Tensor) -> Result<Tensor> {
    let n = coeffs.numel();
    if n <= 1 {
        return coeffs.zeros_like()?.narrow(0, 0, 1);
    }
    let powers: Vec<f64> = (1..n).rev().map(|p| p as f64).collect();
    let pw = Tensor::from_slice(&powers, &[n - 1])?.to_dtype(coeffs.dtype())?;
    coeffs.narrow(0, 0, n - 1)?.mul(&pw)
}

/// Antiderivative coefficients with integration constant `constant`.
pub fn polyint(coeffs: &Tensor, constant: f64) -> Result<Tensor> {
    let n = coeffs.numel();
    let divisors: Vec<f64> = (1..=n).rev().map(|p| p as f64).collect();
    let dv = Tensor::from_slice(&divisors, &[n])?.to_dtype(coeffs.dtype())?;
    let tail = Tensor::from_slice(&[constant], &[1])?.to_dtype(coeffs.dtype())?;
    Tensor::cat(&[coeffs.div(&dv)?, tail], 0)
}

/// Complex roots of the polynomial as eigenvalues of its companion matrix (`numpy.roots`).
pub fn roots(coeffs: &Tensor) -> Result<Complex> {
    let c = coeffs.to_vec::<f64>()?;
    let first = c.iter().position(|&v| v != 0.0).ok_or_else(|| Error::Msg("roots: all coefficients are zero".into()))?;
    let trailing = c.iter().rev().take_while(|&&v| v == 0.0).count();
    let core = &c[first..c.len() - trailing];
    let deg = core.len() - 1;
    let zero_roots = trailing;
    let (mut re, mut im) = (vec![0.0; zero_roots], vec![0.0; zero_roots]);
    if deg >= 1 {
        let mut m = vec![0.0; deg * deg];
        for j in 0..deg {
            m[j] = -core[j + 1] / core[0];
        }
        for i in 1..deg {
            m[i * deg + i - 1] = 1.0;
        }
        let (wr, wi) = Tensor::from_slice(&m, &[deg, deg])?.eigvals()?;
        re.extend(wr.to_vec::<f64>()?);
        im.extend(wi.to_vec::<f64>()?);
    }
    let len = re.len();
    Complex::new(Tensor::from_slice(&re, &[len])?, Tensor::from_slice(&im, &[len])?)
}

/// Output extent of [`convolve`] / [`correlate`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvMode {
    /// All `len(a) + len(v) - 1` overlaps.
    Full,
    /// Output as long as the longer input, centred.
    Same,
    /// Only positions where the inputs fully overlap.
    Valid,
}

/// Discrete 1-D convolution (`numpy.convolve`), differentiable with respect to both inputs.
pub fn convolve(a: &Tensor, v: &Tensor, mode: ConvMode) -> Result<Tensor> {
    if a.ndim() != 1 || v.ndim() != 1 || a.numel() == 0 || v.numel() == 0 {
        return Err(Error::Msg("convolve: both inputs must be non-empty 1-D tensors".into()));
    }
    // swap so that the kernel is the shorter input, then use a full conv1d on a padded signal
    let (sig, ker) = if a.numel() >= v.numel() { (a, v) } else { (v, a) };
    let k = ker.numel();
    let padded = sig.pad(&[(k - 1, k - 1)], 0.0)?;
    let flipped = ker.flip(&[0])?;
    let full = padded.reshape(&[1, 1, padded.numel()])?.conv1d(&flipped.reshape(&[1, 1, k])?, None, 1, 0, 1)?.reshape(&[sig.numel() + k - 1])?;
    let n_sig = sig.numel();
    match mode {
        ConvMode::Full => Ok(full),
        ConvMode::Same => full.narrow(0, (k - 1) / 2, n_sig),
        ConvMode::Valid => full.narrow(0, k - 1, n_sig - k + 1),
    }
}

/// Cross-correlation `c[k] = sum_n a[n + k] * conj(v[n])` (`numpy.correlate`).
pub fn correlate(a: &Tensor, v: &Tensor, mode: ConvMode) -> Result<Tensor> {
    convolve(a, &v.flip(&[0])?, mode)
}

/// Convolution through the FFT, `O(n log n)` for long signals; equals [`convolve`] up to rounding.
pub fn fftconvolve(a: &Tensor, v: &Tensor) -> Result<Tensor> {
    let n = a.numel() + v.numel() - 1;
    let size = n.next_power_of_two();
    let fa = Complex::rfft(&a.pad(&[(0, size - a.numel())], 0.0)?)?;
    let fv = Complex::rfft(&v.pad(&[(0, size - v.numel())], 0.0)?)?;
    fa.mul(&fv)?.ifft()?.re.narrow(0, 0, n)
}

/// Principal square root of a matrix with no eigenvalues on the closed negative real axis, by the Denman-Beavers
/// iteration (differentiable through the iteration).
pub fn sqrtm(a: &Tensor) -> Result<Tensor> {
    if a.ndim() < 2 || a.shape()[a.ndim() - 1] != a.shape()[a.ndim() - 2] {
        return Err(Error::Msg("sqrtm: expects square matrices".into()));
    }
    let n = a.shape()[a.ndim() - 1];
    let eye = Tensor::eye(n, a.dtype(), &a.device())?;
    let (mut y, mut z) = (a.clone(), eye.broadcast_to(a.shape())?.contiguous()?);
    for _ in 0..60 {
        let (yi, zi) = (y.inv()?, z.inv()?);
        let (ny, nz) = (y.add(&zi)?.mul_scalar(0.5)?, z.add(&yi)?.mul_scalar(0.5)?);
        let delta = ny.sub(&y)?.abs()?.amax(&[], false)?.item::<f64>()?;
        y = ny;
        z = nz;
        if delta < 1e-13 {
            break;
        }
    }
    Ok(y)
}

/// Principal matrix logarithm by inverse scaling and squaring: repeated square roots bring `a` near the identity,
/// where a Taylor series of `log(I + X)` converges quickly.
pub fn logm(a: &Tensor) -> Result<Tensor> {
    let n = a.shape()[a.ndim() - 1];
    let eye = Tensor::eye(n, a.dtype(), &a.device())?;
    let (mut m, mut squarings) = (a.clone(), 0u32);
    while squarings < 40 {
        let dist = m.sub(&eye)?.abs()?.amax(&[], false)?.item::<f64>()?;
        if dist < 0.1 {
            break;
        }
        m = sqrtm(&m)?;
        squarings += 1;
    }
    let x = m.sub(&eye)?;
    let (mut term, mut sum) = (x.clone(), x.clone());
    for k in 2..40 {
        term = term.matmul(&x)?;
        let c = if k % 2 == 0 { -1.0 } else { 1.0 } / k as f64;
        sum = sum.add(&term.mul_scalar(c)?)?;
    }
    sum.mul_scalar(f64::from(2u32.pow(squarings)))
}

/// Applies `f` to the eigenvalues of a symmetric matrix: `V f(diag(w)) V^T`. Differentiable.
pub fn funm_spd(a: &Tensor, f: impl Fn(&Tensor) -> Result<Tensor>) -> Result<Tensor> {
    let (w, v) = a.eigh()?;
    let fw = f(&w)?;
    v.mul(&fw.unsqueeze(-2)?)?.matmul(&v.transpose(-2, -1)?)
}

/// Square root of a symmetric positive-semidefinite matrix through the eigendecomposition.
pub fn sqrtm_spd(a: &Tensor) -> Result<Tensor> {
    funm_spd(a, |w| w.clamp(0.0, f64::INFINITY)?.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(v: &[f64]) -> Tensor {
        Tensor::from_slice(v, &[v.len()]).unwrap()
    }
    fn vals(x: &Tensor) -> Vec<f64> {
        x.to_vec::<f64>().unwrap()
    }
    fn near(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len(), "{a:?} vs {b:?}");
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < tol, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn interpolation() {
        let (xp, fp) = (t(&[0.0, 1.0, 3.0]), t(&[0.0, 10.0, 30.0]));
        near(&vals(&interp(&t(&[-1.0, 0.5, 1.0, 2.0, 5.0]), &xp, &fp).unwrap()), &[0.0, 5.0, 10.0, 20.0, 30.0], 1e-12);
        let f = t(&[1.0, 2.0, 4.0]).requires_grad_(true);
        interp(&t(&[0.5, 2.0]), &xp, &f).unwrap().sum().unwrap().backward().unwrap();
        near(&vals(&f.grad().unwrap()), &[0.5, 0.5 + 0.5, 0.5], 1e-12);
        assert!(interp(&t(&[0.0]), &t(&[1.0, 1.0]), &t(&[0.0, 1.0])).is_err());
    }

    #[test]
    fn polynomials() {
        let p = t(&[2.0, -3.0, 1.0]); // 2x^2 - 3x + 1
        near(&vals(&polyval(&p, &t(&[0.0, 1.0, 2.0])).unwrap()), &[1.0, 0.0, 3.0], 1e-12);
        near(&vals(&polyder(&p).unwrap()), &[4.0, -3.0], 1e-12);
        near(&vals(&polyint(&p, 5.0).unwrap()), &[2.0 / 3.0, -1.5, 1.0, 5.0], 1e-12);
        // exact quadratic data is recovered
        let xs: Vec<f64> = (0..8).map(|i| i as f64 * 0.5 - 1.0).collect();
        let ys: Vec<f64> = xs.iter().map(|x| 2.0 * x * x - 3.0 * x + 1.0).collect();
        near(&vals(&polyfit(&t(&xs), &t(&ys), 2).unwrap()), &[2.0, -3.0, 1.0], 1e-8);
        // roots of (x - 1)(x - 2)(x + 3) and of x^2 + 1
        let r = roots(&t(&[1.0, 0.0, -7.0, 6.0])).unwrap();
        let mut re = vals(&r.re);
        re.sort_by(f64::total_cmp);
        near(&re, &[-3.0, 1.0, 2.0], 1e-8);
        let c = roots(&t(&[1.0, 0.0, 1.0])).unwrap();
        near(&vals(&c.im), &[1.0, -1.0], 1e-9);
        // leading and trailing zeros: x^2 - x has roots 0 and 1
        let mut z = vals(&roots(&t(&[0.0, 1.0, -1.0, 0.0])).unwrap().re);
        z.sort_by(f64::total_cmp);
        near(&z, &[0.0, 1.0], 1e-9);
    }

    #[test]
    fn convolution_modes_match_numpy() {
        let (a, v) = (t(&[1.0, 2.0, 3.0]), t(&[0.0, 1.0, 0.5]));
        near(&vals(&convolve(&a, &v, ConvMode::Full).unwrap()), &[0.0, 1.0, 2.5, 4.0, 1.5], 1e-12);
        near(&vals(&convolve(&a, &v, ConvMode::Same).unwrap()), &[1.0, 2.5, 4.0], 1e-12);
        near(&vals(&convolve(&a, &v, ConvMode::Valid).unwrap()), &[2.5], 1e-12);
        // correlate: np.correlate([1,2,3],[0,1,0.5],'full') = [0.5, 2, 3.5, 3, 0]
        near(&vals(&correlate(&a, &v, ConvMode::Full).unwrap()), &[0.5, 2.0, 3.5, 3.0, 0.0], 1e-12);
        // the longer input is the signal regardless of argument order
        near(&vals(&convolve(&v, &a, ConvMode::Full).unwrap()), &vals(&convolve(&a, &v, ConvMode::Full).unwrap()), 1e-12);
        // even-length kernel in 'same' mode
        near(&vals(&convolve(&t(&[1.0, 2.0, 3.0, 4.0]), &t(&[1.0, 1.0]), ConvMode::Same).unwrap()), &[1.0, 3.0, 5.0, 7.0], 1e-12);
        // fft path agrees
        let long: Vec<f64> = (0..37).map(|i| (i as f64 * 0.3).sin()).collect();
        let ker: Vec<f64> = (0..9).map(|i| 1.0 / (1.0 + i as f64)).collect();
        near(&vals(&fftconvolve(&t(&long), &t(&ker)).unwrap()), &vals(&convolve(&t(&long), &t(&ker), ConvMode::Full).unwrap()), 1e-9);
        // gradient flows to the kernel
        let k = t(&[1.0, 2.0]).requires_grad_(true);
        convolve(&t(&[1.0, 1.0, 1.0]), &k, ConvMode::Full).unwrap().sum().unwrap().backward().unwrap();
        near(&vals(&k.grad().unwrap()), &[3.0, 3.0], 1e-12);
    }

    #[test]
    fn matrix_functions() {
        let a = Tensor::from_slice(&[4.0f64, 1.0, 1.0, 3.0], &[2, 2]).unwrap();
        let s = sqrtm(&a).unwrap();
        near(&vals(&s.matmul(&s).unwrap()), &vals(&a), 1e-9);
        near(&vals(&sqrtm_spd(&a).unwrap()), &vals(&s), 1e-8);
        let l = logm(&a).unwrap();
        near(&vals(&l.matrix_exp().unwrap()), &vals(&a), 1e-7);
        // a non-symmetric matrix with positive eigenvalues
        let b = Tensor::from_slice(&[2.0f64, 1.0, 0.0, 3.0], &[2, 2]).unwrap();
        near(&vals(&sqrtm(&b).unwrap().matmul(&sqrtm(&b).unwrap()).unwrap()), &vals(&b), 1e-9);
        near(&vals(&logm(&b).unwrap().matrix_exp().unwrap()), &vals(&b), 1e-7);
        // funm_spd(x^3) = A^3
        near(&vals(&funm_spd(&a, |w| w.powf(3.0)).unwrap()), &vals(&a.matmul(&a).unwrap().matmul(&a).unwrap()), 1e-8);
        // batched
        let batch = Tensor::stack(&[a.clone(), a.mul_scalar(2.0).unwrap()], 0).unwrap();
        assert_eq!(sqrtm(&batch).unwrap().shape(), [2, 2, 2]);
    }
}
