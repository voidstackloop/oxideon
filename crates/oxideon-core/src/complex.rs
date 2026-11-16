//! Complex tensors.
//!
//! [`Complex`] pairs a real and an imaginary [`Tensor`] of equal shape. Every operation is composed from real,
//! differentiable tensor ops, so autograd works through complex arithmetic (gradients flow to `re` and `im`
//! separately, the convention of treating a complex number as two reals). The pair converts to and from the
//! `[..., 2]` "real view" the FFT functions use ([`Complex::view_as_real`], [`Complex::from_view_as_real`]), and
//! [`Complex::fft`] / [`Complex::ifft`] / [`Complex::fft2d`] / [`Complex::ifft2d`] / [`Complex::rfft`] operate on it
//! directly. [`Tensor::eig_complex`] returns the eigen-decomposition of a general matrix as complex tensors.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// A complex tensor `re + i * im`.
#[derive(Clone)]
pub struct Complex {
    pub re: Tensor,
    pub im: Tensor,
}

impl Complex {
    /// Builds a complex tensor; `re` and `im` must have the same shape.
    pub fn new(re: Tensor, im: Tensor) -> Result<Self> {
        if re.shape() != im.shape() {
            return Err(Error::ShapeMismatch { op: "Complex::new", expected: re.shape().to_vec(), got: im.shape().to_vec() });
        }
        Ok(Self { re, im })
    }

    /// A real tensor viewed as complex with zero imaginary part.
    pub fn from_real(re: Tensor) -> Result<Self> {
        let im = re.zeros_like()?;
        Ok(Self { re, im })
    }

    /// `abs * (cos angle + i sin angle)` (`torch.polar`).
    pub fn polar(abs: &Tensor, angle: &Tensor) -> Result<Self> {
        Self::new(abs.mul(&angle.cos()?)?, abs.mul(&angle.sin()?)?)
    }

    /// Interprets a real tensor `[..., 2]` as complex (`torch.view_as_complex`).
    pub fn from_view_as_real(t: &Tensor) -> Result<Self> {
        let nd = t.ndim();
        if nd == 0 || t.shape()[nd - 1] != 2 {
            return Err(Error::Msg(format!("view_as_complex: the last dimension must be 2, got shape {:?}", t.shape())));
        }
        let ax = nd as isize - 1;
        Self::new(t.narrow(ax, 0, 1)?.squeeze_axis(ax)?, t.narrow(ax, 1, 1)?.squeeze_axis(ax)?)
    }

    /// The `[..., 2]` real view (`torch.view_as_real`).
    pub fn view_as_real(&self) -> Result<Tensor> {
        Tensor::stack(&[self.re.clone(), self.im.clone()], -1)
    }

    /// Packs this `Complex { re, im }` into a native `DType::C64` or `DType::C128` tensor.
    pub fn to_native(&self) -> Result<Tensor> {
        use crate::dtype::DType;
        if self.re.dtype() == DType::F64 || self.im.dtype() == DType::F64 {
            let re = self.re.to_dtype(DType::F64)?.to_vec::<f64>()?;
            let im = self.im.to_dtype(DType::F64)?.to_vec::<f64>()?;
            let pairs = re.into_iter().zip(im).map(|(r, i)| [r, i]).collect();
            Tensor::from_c128_vec(pairs, self.re.shape())
        } else {
            let re = self.re.to_dtype(DType::F32)?.to_vec::<f32>()?;
            let im = self.im.to_dtype(DType::F32)?.to_vec::<f32>()?;
            let pairs = re.into_iter().zip(im).map(|(r, i)| [r, i]).collect();
            Tensor::from_c64_vec(pairs, self.re.shape())
        }
    }

    /// Unpacks a native `DType::C64` or `DType::C128` tensor into `Complex { re, im }`.
    pub fn from_native(t: &Tensor) -> Result<Self> {
        Self::new(t.real()?, t.imag()?)
    }

    pub fn shape(&self) -> &[usize] {
        self.re.shape()
    }

    pub fn conj(&self) -> Result<Self> {
        Self::new(self.re.clone(), self.im.neg()?)
    }

    pub fn neg(&self) -> Result<Self> {
        Self::new(self.re.neg()?, self.im.neg()?)
    }

    /// `|z|^2 = re^2 + im^2` (differentiable everywhere, unlike `abs`).
    pub fn abs_sq(&self) -> Result<Tensor> {
        self.re.square()?.add(&self.im.square()?)
    }

    /// Modulus `|z|`.
    pub fn abs(&self) -> Result<Tensor> {
        self.re.hypot(&self.im)
    }

    /// Argument `atan2(im, re)` (`torch.angle`).
    pub fn angle(&self) -> Result<Tensor> {
        self.im.atan2(&self.re)
    }

    pub fn add(&self, o: &Complex) -> Result<Self> {
        Self::new(self.re.add(&o.re)?, self.im.add(&o.im)?)
    }

    pub fn sub(&self, o: &Complex) -> Result<Self> {
        Self::new(self.re.sub(&o.re)?, self.im.sub(&o.im)?)
    }

    pub fn mul(&self, o: &Complex) -> Result<Self> {
        Self::new(
            self.re.mul(&o.re)?.sub(&self.im.mul(&o.im)?)?,
            self.re.mul(&o.im)?.add(&self.im.mul(&o.re)?)?,
        )
    }

    pub fn div(&self, o: &Complex) -> Result<Self> {
        let d = o.abs_sq()?;
        let num = self.mul(&o.conj()?)?;
        Self::new(num.re.div(&d)?, num.im.div(&d)?)
    }

    /// Multiplies by a real tensor (broadcasting).
    pub fn mul_real(&self, r: &Tensor) -> Result<Self> {
        Self::new(self.re.mul(r)?, self.im.mul(r)?)
    }

    pub fn scale(&self, s: f64) -> Result<Self> {
        Self::new(self.re.mul_scalar(s)?, self.im.mul_scalar(s)?)
    }

    pub fn add_real(&self, r: &Tensor) -> Result<Self> {
        Self::new(self.re.add(r)?, self.im.clone())
    }

    pub fn reciprocal(&self) -> Result<Self> {
        let d = self.abs_sq()?;
        Self::new(self.re.div(&d)?, self.im.neg()?.div(&d)?)
    }

    pub fn exp(&self) -> Result<Self> {
        let m = self.re.exp()?;
        Self::new(m.mul(&self.im.cos()?)?, m.mul(&self.im.sin()?)?)
    }

    /// Principal natural logarithm.
    pub fn ln(&self) -> Result<Self> {
        Self::new(self.abs_sq()?.ln()?.mul_scalar(0.5)?, self.angle()?)
    }

    /// Principal square root.
    pub fn sqrt(&self) -> Result<Self> {
        let r = self.abs()?.sqrt()?;
        let half = self.angle()?.mul_scalar(0.5)?;
        Self::new(r.mul(&half.cos()?)?, r.mul(&half.sin()?)?)
    }

    /// Integer power by repeated squaring (negative powers via the reciprocal).
    pub fn powi(&self, n: i32) -> Result<Self> {
        if n < 0 {
            return self.reciprocal()?.powi(-n);
        }
        let mut result = Self::new(self.re.ones_like()?, self.re.zeros_like()?)?;
        let mut base = self.clone();
        let mut e = n as u32;
        while e > 0 {
            if e & 1 == 1 {
                result = result.mul(&base)?;
            }
            e >>= 1;
            if e > 0 {
                base = base.mul(&base)?;
            }
        }
        Ok(result)
    }

    /// `self ^ w = exp(w ln self)` for complex exponents.
    pub fn pow(&self, w: &Complex) -> Result<Self> {
        w.mul(&self.ln()?)?.exp()
    }

    pub fn sin(&self) -> Result<Self> {
        Self::new(self.re.sin()?.mul(&self.im.cosh()?)?, self.re.cos()?.mul(&self.im.sinh()?)?)
    }

    pub fn cos(&self) -> Result<Self> {
        Self::new(self.re.cos()?.mul(&self.im.cosh()?)?, self.re.sin()?.mul(&self.im.sinh()?)?.neg()?)
    }

    pub fn sum_axis(&self, axis: isize, keepdim: bool) -> Result<Self> {
        Self::new(self.re.sum_axis(axis, keepdim)?, self.im.sum_axis(axis, keepdim)?)
    }

    pub fn mean_axis(&self, axis: isize, keepdim: bool) -> Result<Self> {
        Self::new(self.re.mean_axis(axis, keepdim)?, self.im.mean_axis(axis, keepdim)?)
    }

    /// Sum of all elements as a 0-d complex tensor.
    pub fn sum(&self) -> Result<Self> {
        Self::new(self.re.sum()?, self.im.sum()?)
    }

    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        Self::new(self.re.reshape(shape)?, self.im.reshape(shape)?)
    }

    pub fn transpose(&self, a: isize, b: isize) -> Result<Self> {
        Self::new(self.re.transpose(a, b)?, self.im.transpose(a, b)?)
    }

    /// Conjugate transpose of the last two dimensions (`.mH`).
    pub fn adjoint(&self) -> Result<Self> {
        self.transpose(-2, -1)?.conj()
    }

    /// Complex matrix product (batched like `Tensor::matmul`).
    pub fn matmul(&self, o: &Complex) -> Result<Self> {
        Self::new(
            self.re.matmul(&o.re)?.sub(&self.im.matmul(&o.im)?)?,
            self.re.matmul(&o.im)?.add(&self.im.matmul(&o.re)?)?,
        )
    }

    /// Hermitian inner product `sum(conj(self) * other)` over all elements.
    pub fn vdot(&self, o: &Complex) -> Result<Self> {
        self.conj()?.mul(o)?.sum()
    }

    fn through_fft(&self, f: impl Fn(&Tensor) -> Result<Tensor>) -> Result<Self> {
        Self::from_view_as_real(&f(&self.view_as_real()?)?)
    }

    /// FFT along the last logical dimension of a `[..., N]` complex tensor.
    pub fn fft(&self) -> Result<Self> {
        self.through_fft(Tensor::fft)
    }

    pub fn ifft(&self) -> Result<Self> {
        self.through_fft(Tensor::ifft)
    }

    /// 2-D FFT over the last two logical dimensions.
    pub fn fft2d(&self) -> Result<Self> {
        self.through_fft(Tensor::fft2d)
    }

    pub fn ifft2d(&self) -> Result<Self> {
        self.through_fft(Tensor::ifft2d)
    }

    /// FFT of a real signal `[..., N]`, returned as the full complex spectrum `[..., N]`.
    pub fn rfft(x: &Tensor) -> Result<Self> {
        Self::from_real(x.clone())?.fft()
    }
}

impl Tensor {
    /// Builds a complex tensor from real and imaginary parts (`torch.complex`).
    pub fn complex(re: &Tensor, im: &Tensor) -> Result<Complex> {
        Complex::new(re.clone(), im.clone())
    }

    /// Reads a real `[..., 2]` tensor as complex.
    pub fn view_as_complex(&self) -> Result<Complex> {
        Complex::from_view_as_real(self)
    }

    /// Eigenvalues `[..., n]` and eigenvectors `[..., n, n]` (columns) of a general real matrix as complex tensors.
    pub fn eig_complex(&self) -> Result<(Complex, Complex)> {
        let (wr, wi, vr, vi) = self.eig()?;
        Ok((Complex::new(wr, wi)?, Complex::new(vr, vi)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: &[f64], im: &[f64]) -> Complex {
        Complex::new(Tensor::from_slice(re, &[re.len()]).unwrap(), Tensor::from_slice(im, &[im.len()]).unwrap()).unwrap()
    }
    fn pair(z: &Complex) -> (Vec<f64>, Vec<f64>) {
        (z.re.to_vec::<f64>().unwrap(), z.im.to_vec::<f64>().unwrap())
    }
    fn close(z: &Complex, re: &[f64], im: &[f64]) {
        let (r, i) = pair(z);
        for k in 0..re.len() {
            assert!((r[k] - re[k]).abs() < 1e-9 && (i[k] - im[k]).abs() < 1e-9, "{r:?} {i:?} vs {re:?} {im:?}");
        }
    }

    #[test]
    fn arithmetic() {
        let (a, b) = (c(&[1.0], &[2.0]), c(&[3.0], &[-1.0]));
        close(&a.mul(&b).unwrap(), &[5.0], &[5.0]);
        close(&a.add(&b).unwrap(), &[4.0], &[1.0]);
        close(&a.sub(&b).unwrap(), &[-2.0], &[3.0]);
        close(&a.div(&b).unwrap(), &[0.1], &[0.7]); // (1+2i)(3+i)/10
        close(&a.reciprocal().unwrap(), &[0.2], &[-0.4]);
        close(&a.conj().unwrap(), &[1.0], &[-2.0]);
        assert!((a.abs().unwrap().to_vec::<f64>().unwrap()[0] - 5f64.sqrt()).abs() < 1e-12);
        assert!((a.angle().unwrap().to_vec::<f64>().unwrap()[0] - 2f64.atan2(1.0)).abs() < 1e-12);
        close(&a.powi(3).unwrap(), &[-11.0], &[-2.0]); // (1+2i)^3
        close(&a.powi(-1).unwrap(), &[0.2], &[-0.4]);
        close(&a.powi(0).unwrap(), &[1.0], &[0.0]);
    }

    #[test]
    fn transcendental() {
        let pi = std::f64::consts::PI;
        close(&c(&[0.0], &[pi]).exp().unwrap(), &[-1.0], &[0.0]); // Euler
        close(&c(&[-1.0], &[0.0]).sqrt().unwrap(), &[0.0], &[1.0]);
        close(&c(&[3.0], &[4.0]).sqrt().unwrap(), &[2.0], &[1.0]);
        close(&c(&[-1.0], &[0.0]).ln().unwrap(), &[0.0], &[pi]);
        close(&c(&[0.0], &[1.0]).pow(&c(&[0.0], &[1.0])).unwrap(), &[(-pi / 2.0).exp()], &[0.0]); // i^i
        let z = c(&[0.3], &[0.7]);
        let s = z.sin().unwrap();
        let co = z.cos().unwrap();
        close(&s.mul(&s).unwrap().add(&co.mul(&co).unwrap()).unwrap(), &[1.0], &[0.0]); // sin^2 + cos^2 = 1
        let p = Complex::polar(&Tensor::from_slice(&[2.0f64], &[1]).unwrap(), &Tensor::from_slice(&[pi / 2.0], &[1]).unwrap()).unwrap();
        close(&p, &[0.0], &[2.0]);
    }

    #[test]
    fn real_view_and_fft() {
        let z = c(&[1.0, 0.0, 0.0, 0.0], &[0.0; 4]);
        close(&z.fft().unwrap(), &[1.0; 4], &[0.0; 4]); // impulse -> flat spectrum
        let x = c(&[1.0, 2.0, 3.0, 4.0], &[0.5, -1.0, 0.0, 2.0]);
        let back = x.fft().unwrap().ifft().unwrap();
        close(&back, &pair(&x).0, &pair(&x).1);
        let real = Tensor::from_slice(&[1.0f64, 0.0, -1.0, 0.0], &[4]).unwrap();
        close(&Complex::rfft(&real).unwrap(), &[0.0, 2.0, 0.0, 2.0], &[0.0; 4]); // cosine at bin 1
        let v = x.view_as_real().unwrap();
        assert_eq!(v.shape(), [4, 2]);
        close(&v.view_as_complex().unwrap(), &pair(&x).0, &pair(&x).1);
        assert!(Tensor::zeros(&[3, 3], crate::dtype::DType::F32, &crate::backend::Device::Cpu).unwrap().view_as_complex().is_err());
        // Parseval: sum |x|^2 = sum |X|^2 / N
        let e1: f64 = x.abs_sq().unwrap().sum().unwrap().item().unwrap();
        let e2: f64 = x.fft().unwrap().abs_sq().unwrap().sum().unwrap().item::<f64>().unwrap() / 4.0;
        assert!((e1 - e2).abs() < 1e-9);
    }

    #[test]
    fn gradients_flow_through_complex_arithmetic() {
        // loss = Re(z * z) = x^2 - y^2  ->  (2x, -2y)
        let (x, y) = (Tensor::from_slice(&[1.5f64], &[1]).unwrap().requires_grad_(true), Tensor::from_slice(&[-0.5f64], &[1]).unwrap().requires_grad_(true));
        let z = Complex::new(x.clone(), y.clone()).unwrap();
        z.mul(&z).unwrap().re.sum().unwrap().backward().unwrap();
        assert_eq!(x.grad().unwrap().to_vec::<f64>().unwrap(), vec![3.0]);
        assert_eq!(y.grad().unwrap().to_vec::<f64>().unwrap(), vec![1.0]);
        // |z|^2 of the DFT (Parseval) has gradient 2 N x on the input
        let sig = Tensor::from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[4]).unwrap().requires_grad_(true);
        Complex::rfft(&sig).unwrap().abs_sq().unwrap().sum().unwrap().backward().unwrap();
        assert_eq!(sig.grad().unwrap().to_vec::<f64>().unwrap(), vec![8.0, 16.0, 24.0, 32.0]);
    }

    #[test]
    fn matmul_and_eigen_decomposition() {
        let a = Complex::new(Tensor::from_slice(&[1.0f64, 0.0, 0.0, 1.0], &[2, 2]).unwrap(), Tensor::from_slice(&[0.0f64, 1.0, 1.0, 0.0], &[2, 2]).unwrap()).unwrap();
        let sq = a.matmul(&a).unwrap(); // [[1,i],[i,1]]^2 = [[0,2i],[2i,0]]
        close(&sq.reshape(&[4]).unwrap(), &[0.0; 4], &[0.0, 2.0, 2.0, 0.0]);
        // A v = lambda v for a real matrix with complex eigenvalues
        let m = Tensor::from_slice(&[0.0f64, -2.0, 1.0, 0.0, 1.0, 3.0, 2.0, -1.0, 0.5], &[3, 3]).unwrap();
        let (w, v) = m.eig_complex().unwrap();
        let av = Complex::from_real(m.clone()).unwrap().matmul(&v).unwrap();
        let lv = v.mul(&Complex::new(w.re.unsqueeze(0).unwrap(), w.im.unsqueeze(0).unwrap()).unwrap()).unwrap();
        let diff = av.sub(&lv).unwrap().abs_sq().unwrap().sum().unwrap().item::<f64>().unwrap();
        assert!(diff < 1e-12, "{diff}");
        assert!(a.adjoint().unwrap().matmul(&a).unwrap().re.to_vec::<f64>().unwrap()[0] > 0.0);
    }
}
