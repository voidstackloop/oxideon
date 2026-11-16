//! Linear algebra that `torch.linalg` has beyond the core set: batched application of the 2-D factorizations,
//! `svdvals`, `eigvalsh`, `logdet`, `vector_norm`, `matrix_norm`, `vecdot`, `cholesky_inverse`, `lu_solve`,
//! `tensorinv`, `tensorsolve`, `householder_product`, and the general (non-symmetric) eigenvalue problem
//! `eigvals` / `eig`, which returns real and imaginary parts as separate tensors because the library has no complex
//! dtype.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    /// Applies a per-matrix function over the leading batch dimensions of a `[..., m, n]` tensor, stacking every
    /// output. The function sees plain 2-D matrices, so autograd flows through it unchanged.
    pub(crate) fn batched_mats(
        &self,
        f: impl Fn(&Tensor) -> Result<Vec<Tensor>>,
    ) -> Result<Vec<Tensor>> {
        let nd = self.ndim();
        if nd < 2 {
            return Err(Error::Msg("expected a tensor of matrices ([..., m, n])".into()));
        }
        let batch: Vec<usize> = self.shape()[..nd - 2].to_vec();
        let b: usize = batch.iter().product();
        if b == 0 {
            return Err(Error::Msg("batched linear algebra: empty batch".into()));
        }
        let flat = self.reshape(&[b, self.shape()[nd - 2], self.shape()[nd - 1]])?;
        let mut collected: Vec<Vec<Tensor>> = Vec::new();
        for i in 0..b {
            let outs = f(&flat.select(0, i as isize)?)?;
            if collected.is_empty() {
                collected = vec![Vec::with_capacity(b); outs.len()];
            }
            for (slot, o) in collected.iter_mut().zip(outs) {
                slot.push(o);
            }
        }
        collected
            .into_iter()
            .map(|parts| {
                let stacked = Tensor::stack(&parts, 0)?;
                let mut shape = batch.clone();
                shape.extend(&stacked.shape()[1..]);
                stacked.reshape(&shape)
            })
            .collect()
    }

    /// Singular values in descending order, differentiable (`torch.linalg.svdvals`).
    pub fn svdvals(&self) -> Result<Tensor> {
        Ok(self.svd()?.1)
    }

    /// Eigenvalues of a symmetric matrix in ascending order (`torch.linalg.eigvalsh`).
    pub fn eigvalsh(&self) -> Result<Tensor> {
        Ok(self.eigh()?.0)
    }

    /// `ln det(A)` of the last two dimensions; `NaN` for a negative determinant (`torch.logdet`).
    pub fn logdet(&self) -> Result<Tensor> {
        let (sign, logabs) = self.slogdet()?;
        let neg = sign.lt(&sign.zeros_like()?)?;
        Tensor::where_cond(&neg, &logabs.full_like(f64::NAN)?, &logabs)
    }

    /// Vector norm of order `ord` over `dims` (all dims when empty): any finite `ord`, `0` (count of non-zeros),
    /// and `+-inf` (`torch.linalg.vector_norm`).
    pub fn vector_norm(&self, ord: f64, dims: &[isize], keepdim: bool) -> Result<Tensor> {
        if ord.is_infinite() {
            let a = self.abs()?;
            return if ord > 0.0 { a.amax(dims, keepdim) } else { a.amin(dims, keepdim) };
        }
        let axes: Vec<isize> = if dims.is_empty() {
            (0..self.ndim() as isize).collect()
        } else {
            dims.to_vec()
        };
        if ord == 0.0 {
            let nz = self.ne_tensor(&self.zeros_like()?)?.to_dtype(self.dtype())?;
            return nz.sum_axes(&axes, keepdim);
        }
        self.norm_p(ord, &axes, keepdim)
    }

    /// Matrix norm over the last two dimensions: `"fro"`, `"nuc"`, `"1"`, `"-1"`, `"2"`, `"-2"`, `"inf"` or
    /// `"-inf"` (`torch.linalg.matrix_norm`).
    pub fn matrix_norm(&self, ord: &str, keepdim: bool) -> Result<Tensor> {
        if self.ndim() < 2 {
            return Err(Error::Msg("matrix_norm: expects at least 2 dimensions".into()));
        }
        let (r, c) = (self.ndim() as isize - 2, self.ndim() as isize - 1);
        let out = match ord {
            "fro" => self.square()?.sum_axes(&[r, c], false)?.sqrt()?,
            "nuc" => self.svdvals()?.sum_axis(-1, false)?,
            "2" => self.svdvals()?.amax(&[-1], false)?,
            "-2" => self.svdvals()?.amin(&[-1], false)?,
            "1" => self.abs()?.sum_axis(r, false)?.amax(&[-1], false)?,
            "-1" => self.abs()?.sum_axis(r, false)?.amin(&[-1], false)?,
            "inf" => self.abs()?.sum_axis(c, false)?.amax(&[-1], false)?,
            "-inf" => self.abs()?.sum_axis(c, false)?.amin(&[-1], false)?,
            other => return Err(Error::Msg(format!("matrix_norm: unknown ord '{other}'"))),
        };
        if keepdim {
            out.unsqueeze(-1)?.unsqueeze(-1)
        } else {
            Ok(out)
        }
    }

    /// Dot product along `dim` (`torch.linalg.vecdot`; real inputs).
    pub fn vecdot(&self, other: &Tensor, dim: isize) -> Result<Tensor> {
        self.mul(other)?.sum_axis(dim, false)
    }

    /// Inverse of `L L^T` (or `U^T U` with `upper`) from its Cholesky factor (`torch.cholesky_inverse`).
    pub fn cholesky_inverse(&self, upper: bool) -> Result<Tensor> {
        let t = self.transpose(-2, -1)?;
        if upper {
            t.matmul(self)?.inv()
        } else {
            self.matmul(&t)?.inv()
        }
    }

    /// Solves `A X = B` from the unpacked factorization `A = P L U` returned by [`lu`](Self::lu)
    /// (`torch.linalg.lu_solve`).
    pub fn lu_solve(p: &Tensor, l: &Tensor, u: &Tensor, b: &Tensor) -> Result<Tensor> {
        let pb = p.transpose(-2, -1)?.matmul(b)?;
        let y = l.triangular_solve(&pb, true, false)?;
        u.triangular_solve(&y, false, false)
    }

    /// Inverse of a tensor viewed as a matrix that maps the trailing dims to the leading `ind` dims
    /// (`torch.linalg.tensorinv`).
    pub fn tensorinv(&self, ind: usize) -> Result<Tensor> {
        let (lead, trail) = self.shape().split_at(ind.min(self.ndim()));
        let (p, q): (usize, usize) = (lead.iter().product(), trail.iter().product());
        if p != q {
            return Err(Error::Msg("tensorinv: the leading and trailing dimensions must have equal size".into()));
        }
        let mut out_shape = trail.to_vec();
        out_shape.extend(lead);
        self.reshape(&[p, q])?.inv()?.reshape(&out_shape)
    }

    /// Solves `tensordot(A, X, X.ndim) = B` for `X` (`torch.linalg.tensorsolve`).
    pub fn tensorsolve(&self, b: &Tensor) -> Result<Tensor> {
        let nb = b.ndim();
        if self.ndim() < nb || self.shape()[..nb] != *b.shape() {
            return Err(Error::Msg("tensorsolve: A must have shape B.shape + X.shape".into()));
        }
        let rest: Vec<usize> = self.shape()[nb..].to_vec();
        let n = b.numel();
        if n != rest.iter().product::<usize>() {
            return Err(Error::Msg("tensorsolve: A must be square when viewed as a matrix".into()));
        }
        self.reshape(&[n, n])?
            .solve(&b.reshape(&[n, 1])?)?
            .reshape(&rest)
    }

    /// First `n` columns of the product of Householder reflectors `H_i = I - tau_i v_i v_i^T`, where `v_i` is stored
    /// in column `i` below an implicit unit diagonal (`torch.linalg.householder_product`).
    pub fn householder_product(&self, tau: &Tensor) -> Result<Tensor> {
        if self.ndim() != 2 || tau.ndim() != 1 || tau.shape()[0] > self.shape()[1] {
            return Err(Error::Msg("householder_product: expects a [m, n] matrix and at most n scalars tau".into()));
        }
        let (m, n) = (self.shape()[0], self.shape()[1]);
        let dev = self.device();
        let mut q = Tensor::eye(m, self.dtype(), &dev)?;
        for i in 0..tau.shape()[0] {
            let col = self.narrow(1, i, 1)?.reshape(&[m])?;
            let mut parts = Vec::new();
            if i > 0 {
                parts.push(Tensor::zeros(&[i], self.dtype(), &dev)?);
            }
            parts.push(Tensor::ones(&[1], self.dtype(), &dev)?);
            if i + 1 < m {
                parts.push(col.narrow(0, i + 1, m - i - 1)?);
            }
            let v = Tensor::cat(&parts, 0)?;
            let t = tau.narrow(0, i, 1)?.reshape(&[])?;
            let h = Tensor::eye(m, self.dtype(), &dev)?.sub(&v.outer(&v)?.mul(&t)?)?;
            q = q.matmul(&h)?;
        }
        q.narrow(1, 0, n.min(m))
    }

    /// Eigenvalues of a general real square matrix as `(real, imag)` tensors of shape `[..., n]` (not
    /// differentiable). Complex pairs are adjacent, positive imaginary part first.
    pub fn eigvals(&self) -> Result<(Tensor, Tensor)> {
        let (wr, wi, _, _) = self.eig_impl(false)?;
        Ok((wr, wi))
    }

    /// Eigen-decomposition of a general real square matrix: `(w_re, w_im, v_re, v_im)` with the unit-norm
    /// eigenvector for `w[j]` in column `j` of `v` (not differentiable; repeated defective eigenvalues repeat their
    /// vector).
    pub fn eig(&self) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        self.eig_impl(true)
    }

    fn eig_impl(&self, vectors: bool) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let nd = self.ndim();
        if nd < 2 || self.shape()[nd - 1] != self.shape()[nd - 2] {
            return Err(Error::Msg("eig: expects a (batch of) square matrix [..., n, n]".into()));
        }
        let n = self.shape()[nd - 1];
        let batch: Vec<usize> = self.shape()[..nd - 2].to_vec();
        let b: usize = batch.iter().product();
        let data = self.to_vec::<f64>()?;
        let (mut wr, mut wi) = (Vec::with_capacity(b * n), Vec::with_capacity(b * n));
        let (mut vr, mut vi) = (Vec::new(), Vec::new());
        for bi in 0..b {
            let a = &data[bi * n * n..(bi + 1) * n * n];
            let vals = eigenvalues(a, n)?;
            for z in &vals {
                wr.push(z.re);
                wi.push(z.im);
            }
            if vectors {
                let vecs = eigenvectors(a, n, &vals);
                // row-major [n, n] with eigenvector j in column j
                for row in 0..n {
                    for col in 0..n {
                        vr.push(vecs[col][row].re);
                        vi.push(vecs[col][row].im);
                    }
                }
            }
        }
        let dev = self.device();
        let dt = self.dtype();
        let mut vshape = batch.clone();
        vshape.extend([n, n]);
        let mut wshape = batch;
        wshape.push(n);
        let mk = |v: &[f64], s: &[usize]| -> Result<Tensor> {
            Tensor::from_slice_on(v, s, &dev)?.to_dtype(dt)
        };
        if vectors {
            Ok((mk(&wr, &wshape)?, mk(&wi, &wshape)?, mk(&vr, &vshape)?, mk(&vi, &vshape)?))
        } else {
            let empty = Tensor::zeros(&[0], dt, &dev)?;
            Ok((mk(&wr, &wshape)?, mk(&wi, &wshape)?, empty.clone(), empty))
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Complex scalar helper and the eigenvalue algorithm
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
struct C {
    re: f64,
    im: f64,
}

impl C {
    const ZERO: C = C { re: 0.0, im: 0.0 };
    const ONE: C = C { re: 1.0, im: 0.0 };
    fn new(re: f64, im: f64) -> C {
        C { re, im }
    }
    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    fn conj(self) -> C {
        C::new(self.re, -self.im)
    }
    fn add(self, o: C) -> C {
        C::new(self.re + o.re, self.im + o.im)
    }
    fn sub(self, o: C) -> C {
        C::new(self.re - o.re, self.im - o.im)
    }
    fn mul(self, o: C) -> C {
        C::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
    fn scale(self, s: f64) -> C {
        C::new(self.re * s, self.im * s)
    }
    fn div(self, o: C) -> C {
        let d = o.re * o.re + o.im * o.im;
        C::new(
            (self.re * o.re + self.im * o.im) / d,
            (self.im * o.re - self.re * o.im) / d,
        )
    }
    fn sqrt(self) -> C {
        let m = self.abs();
        let re = ((m + self.re) / 2.0).max(0.0).sqrt();
        let im = ((m - self.re) / 2.0).max(0.0).sqrt();
        C::new(re, if self.im < 0.0 { -im } else { im })
    }
}

/// Reduces a real matrix (row-major `n x n`) to upper Hessenberg form by Householder similarity transforms.
fn hessenberg(a: &[f64], n: usize) -> Vec<f64> {
    let mut h = a.to_vec();
    for k in 0..n.saturating_sub(2) {
        let mut x: Vec<f64> = (k + 1..n).map(|i| h[i * n + k]).collect();
        let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm == 0.0 {
            continue;
        }
        let alpha = if x[0] > 0.0 { -norm } else { norm };
        x[0] -= alpha;
        let vn = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        if vn == 0.0 {
            continue;
        }
        for v in &mut x {
            *v /= vn;
        }
        // H <- P H P with P = I - 2 x x^T acting on rows / columns k+1..n
        for j in 0..n {
            let dot: f64 = (0..x.len()).map(|i| x[i] * h[(k + 1 + i) * n + j]).sum();
            for i in 0..x.len() {
                h[(k + 1 + i) * n + j] -= 2.0 * x[i] * dot;
            }
        }
        for i in 0..n {
            let dot: f64 = (0..x.len()).map(|j| h[i * n + k + 1 + j] * x[j]).sum();
            for j in 0..x.len() {
                h[i * n + k + 1 + j] -= 2.0 * dot * x[j];
            }
        }
    }
    h
}

/// Eigenvalues of a real matrix by single-shift complex QR iteration on the Hessenberg form.
fn eigenvalues(a: &[f64], n: usize) -> Result<Vec<C>> {
    if n == 0 {
        return Ok(Vec::new());
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(Error::Msg("eig: the matrix contains non-finite values".into()));
    }
    let hr = hessenberg(a, n);
    let mut h: Vec<C> = hr.iter().map(|&v| C::new(v, 0.0)).collect();
    let mut eig = vec![C::ZERO; n];
    let mut hi = n - 1;
    let mut iter = 0usize;
    let eps = f64::EPSILON;
    loop {
        // find the start of the active unreduced block
        let mut l = 0;
        for k in (1..=hi).rev() {
            let s = h[(k - 1) * n + k - 1].abs() + h[k * n + k].abs();
            let s = if s == 0.0 { 1.0 } else { s };
            if h[k * n + k - 1].abs() <= eps * s {
                h[k * n + k - 1] = C::ZERO;
                l = k;
                break;
            }
        }
        if l == hi {
            eig[hi] = h[hi * n + hi];
            if hi == 0 {
                break;
            }
            hi -= 1;
            iter = 0;
            continue;
        }
        iter += 1;
        if iter > 60 * n.max(4) {
            return Err(Error::Msg("eig: the QR iteration did not converge".into()));
        }
        // Wilkinson shift from the trailing 2x2 block (exceptional shifts every 10 iterations)
        let (a11, a12, a21, a22) = (
            h[(hi - 1) * n + hi - 1],
            h[(hi - 1) * n + hi],
            h[hi * n + hi - 1],
            h[hi * n + hi],
        );
        let mu = if iter % 10 == 0 {
            a22.add(C::new(a21.abs(), 0.0))
        } else {
            let tr_half = a11.add(a22).scale(0.5);
            let det = a11.mul(a22).sub(a12.mul(a21));
            let disc = tr_half.mul(tr_half).sub(det).sqrt();
            let (l1, l2) = (tr_half.add(disc), tr_half.sub(disc));
            if l1.sub(a22).abs() < l2.sub(a22).abs() { l1 } else { l2 }
        };
        for k in l..=hi {
            h[k * n + k] = h[k * n + k].sub(mu);
        }
        // QR by Givens rotations on rows k, k+1
        let mut rots: Vec<(C, C)> = Vec::with_capacity(hi - l);
        for k in l..hi {
            let (x, y) = (h[k * n + k], h[(k + 1) * n + k]);
            let r = (x.abs().powi(2) + y.abs().powi(2)).sqrt();
            let (c, s) = if r == 0.0 { (C::ONE, C::ZERO) } else { (x.scale(1.0 / r), y.scale(1.0 / r)) };
            // rotation G = [[conj(c), conj(s)], [-s, c]] applied to rows
            for j in k..=hi {
                let (u, v) = (h[k * n + j], h[(k + 1) * n + j]);
                h[k * n + j] = c.conj().mul(u).add(s.conj().mul(v));
                h[(k + 1) * n + j] = c.mul(v).sub(s.mul(u));
            }
            rots.push((c, s));
        }
        // multiply by Q from the right: columns k, k+1 with G^H
        for (idx, &(c, s)) in rots.iter().enumerate() {
            let k = l + idx;
            let top = (k + 2).min(hi);
            for i in l..=top {
                let (u, v) = (h[i * n + k], h[i * n + k + 1]);
                h[i * n + k] = u.mul(c).add(v.mul(s));
                h[i * n + k + 1] = v.mul(c.conj()).sub(u.mul(s.conj()));
            }
        }
        for k in l..=hi {
            h[k * n + k] = h[k * n + k].add(mu);
        }
    }
    // pair conjugates: positive imaginary part first
    let tol = 1e-10;
    let mut i = 0;
    while i + 1 < n {
        if eig[i].im.abs() > tol * (1.0 + eig[i].re.abs()) && (eig[i].im + eig[i + 1].im).abs() <= tol * (1.0 + eig[i].im.abs()) {
            if eig[i].im < 0.0 {
                eig.swap(i, i + 1);
            }
            i += 2;
        } else {
            i += 1;
        }
    }
    // real matrices have exactly real eigenvalues up to rounding: clean the tiny imaginary parts
    for z in &mut eig {
        if z.im.abs() <= 1e-13 * (1.0 + z.re.abs()) {
            z.im = 0.0;
        }
    }
    Ok(eig)
}

/// Unit-norm eigenvectors by complex inverse iteration.
fn eigenvectors(a: &[f64], n: usize, vals: &[C]) -> Vec<Vec<C>> {
    let scale = a.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1.0);
    vals.iter()
        .map(|&lam| {
            // slightly perturbed shift so that (A - shift I) is invertible
            let shift = lam.add(C::new(scale * 1e-10, scale * 1e-10 * 0.5));
            let mut m = vec![C::ZERO; n * n];
            for i in 0..n {
                for j in 0..n {
                    m[i * n + j] = C::new(a[i * n + j], 0.0);
                }
                m[i * n + i] = m[i * n + i].sub(shift);
            }
            let lu = complex_lu(&mut m, n);
            let mut x: Vec<C> = (0..n).map(|i| C::new(1.0 + 0.1 * i as f64, 0.0)).collect();
            for _ in 0..4 {
                x = complex_solve(&m, &lu, n, &x);
                let norm = x.iter().map(|z| z.abs().powi(2)).sum::<f64>().sqrt();
                if norm == 0.0 || !norm.is_finite() {
                    break;
                }
                for z in &mut x {
                    *z = z.scale(1.0 / norm);
                }
            }
            // fix the phase: the largest component becomes real and positive
            if let Some(big) = x.iter().copied().max_by(|p, q| p.abs().total_cmp(&q.abs())) {
                if big.abs() > 0.0 {
                    let rot = big.conj().scale(1.0 / big.abs());
                    for z in &mut x {
                        *z = z.mul(rot);
                    }
                }
            }
            x
        })
        .collect()
}

/// In-place LU with partial pivoting; returns the row permutation.
fn complex_lu(m: &mut [C], n: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..n).collect();
    for k in 0..n {
        let mut p = k;
        for i in k + 1..n {
            if m[i * n + k].abs() > m[p * n + k].abs() {
                p = i;
            }
        }
        if p != k {
            for j in 0..n {
                m.swap(k * n + j, p * n + j);
            }
            perm.swap(k, p);
        }
        if m[k * n + k].abs() < 1e-300 {
            m[k * n + k] = C::new(1e-300, 0.0);
        }
        for i in k + 1..n {
            let f = m[i * n + k].div(m[k * n + k]);
            m[i * n + k] = f;
            for j in k + 1..n {
                m[i * n + j] = m[i * n + j].sub(f.mul(m[k * n + j]));
            }
        }
    }
    perm
}

fn complex_solve(lu: &[C], perm: &[usize], n: usize, b: &[C]) -> Vec<C> {
    let mut y: Vec<C> = perm.iter().map(|&p| b[p]).collect();
    for i in 0..n {
        for j in 0..i {
            y[i] = y[i].sub(lu[i * n + j].mul(y[j]));
        }
    }
    for i in (0..n).rev() {
        for j in i + 1..n {
            y[i] = y[i].sub(lu[i * n + j].mul(y[j]));
        }
        y[i] = y[i].div(lu[i * n + i]);
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(v: &[f64], shape: &[usize]) -> Tensor {
        Tensor::from_slice(v, shape).unwrap()
    }
    fn vals(t: &Tensor) -> Vec<f64> {
        t.to_vec::<f64>().unwrap()
    }
    fn close(t: &Tensor, want: &[f64], tol: f64) {
        let got = vals(t);
        assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() <= tol * (1.0 + w.abs()), "{got:?} vs {want:?}");
        }
    }

    #[test]
    fn batched_factorizations_match_per_matrix() {
        let a = Tensor::randn(&[2, 3, 4]).unwrap().to_dtype(crate::dtype::DType::F64).unwrap();
        let (u, s, vt) = a.svd().unwrap();
        assert_eq!(u.shape(), [2, 3, 3]);
        assert_eq!(s.shape(), [2, 3]);
        assert_eq!(vt.shape(), [2, 3, 4]);
        let recon = u.mul(&s.unsqueeze(-2).unwrap()).unwrap().matmul(&vt).unwrap();
        close(&recon, &vals(&a), 1e-9);
        let sym = a.matmul(&a.transpose(-2, -1).unwrap()).unwrap();
        let (w, q) = sym.eigh().unwrap();
        let back = q.mul(&w.unsqueeze(-2).unwrap()).unwrap().matmul(&q.transpose(-2, -1).unwrap()).unwrap();
        close(&back, &vals(&sym), 1e-9);
        let (qq, r) = a.transpose(-2, -1).unwrap().qr().unwrap();
        close(&qq.matmul(&r).unwrap(), &vals(&a.transpose(-2, -1).unwrap()), 1e-9);
        close(&sym.matrix_power(2).unwrap(), &vals(&sym.matmul(&sym).unwrap()), 1e-9);
        let (p, l, uu) = sym.lu().unwrap();
        close(&p.matmul(&l).unwrap().matmul(&uu).unwrap(), &vals(&sym), 1e-9);
    }

    #[test]
    fn norms_and_determinants() {
        let m = f(&[1.0, -2.0, 3.0, 4.0], &[2, 2]);
        close(&m.matrix_norm("fro", false).unwrap(), &[30f64.sqrt()], 1e-12);
        close(&m.matrix_norm("1", false).unwrap(), &[6.0], 1e-12);
        close(&m.matrix_norm("inf", false).unwrap(), &[7.0], 1e-12);
        let s = m.svdvals().unwrap();
        close(&m.matrix_norm("nuc", false).unwrap(), &[vals(&s).iter().sum::<f64>()], 1e-9);
        close(&m.matrix_norm("2", false).unwrap(), &[vals(&s)[0]], 1e-9);
        let v = f(&[3.0, -4.0, 0.0], &[3]);
        close(&v.vector_norm(f64::INFINITY, &[], false).unwrap(), &[4.0], 1e-12);
        close(&v.vector_norm(f64::NEG_INFINITY, &[], false).unwrap(), &[0.0], 1e-12);
        close(&v.vector_norm(0.0, &[], false).unwrap(), &[2.0], 1e-12);
        close(&v.vector_norm(1.0, &[], false).unwrap(), &[7.0], 1e-12);
        close(&v.vector_norm(2.0, &[], false).unwrap(), &[5.0], 1e-12);
        close(&m.logdet().unwrap(), &[10f64.ln()], 1e-12);
        assert!(vals(&f(&[0.0, 1.0, 1.0, 0.0], &[2, 2]).logdet().unwrap())[0].is_nan());
        close(&f(&[1.0, 2.0], &[2]).vecdot(&f(&[3.0, 4.0], &[2]), 0).unwrap(), &[11.0], 1e-12);
    }

    #[test]
    fn solvers_and_inverses() {
        let a = f(&[4.0, 1.0, 2.0, 3.0], &[2, 2]);
        let (p, l, u) = a.lu().unwrap();
        let b = f(&[1.0, 2.0], &[2, 1]);
        close(&Tensor::lu_solve(&p, &l, &u, &b).unwrap(), &vals(&a.solve(&b).unwrap()), 1e-10);
        let spd = f(&[4.0, 2.0, 2.0, 3.0], &[2, 2]);
        let chol = spd.cholesky().unwrap();
        close(&chol.cholesky_inverse(false).unwrap(), &vals(&spd.inv().unwrap()), 1e-10);
        let t = f(&[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[2, 2, 2, 2]);
        let _ = t; // shape smoke test below
        let eye = Tensor::eye(4, crate::dtype::DType::F64, &crate::backend::Device::Cpu).unwrap().reshape(&[2, 2, 2, 2]).unwrap();
        close(&eye.tensorinv(2).unwrap(), &vals(&eye), 1e-12);
        let rhs = f(&[2.0, 4.0, 6.0, 8.0], &[2, 2]);
        close(&eye.tensorsolve(&rhs).unwrap(), &vals(&rhs), 1e-12);
        // one reflector: H = I - tau v v^T with v = [1, 0.5, 0.25]
        let v = f(&[0.0, 0.5, 0.25], &[3, 1]);
        let h = v.householder_product(&f(&[0.8], &[1])).unwrap();
        let vv = [1.0, 0.5, 0.25];
        let want: Vec<f64> = (0..3).map(|i| (if i == 0 { 1.0 } else { 0.0 }) - 0.8 * vv[i] * vv[0]).collect();
        close(&h, &want, 1e-12);
    }

    #[test]
    fn general_eigenproblem() {
        // companion matrix of (x - 1)(x - 2)(x - 3)
        let c = f(&[0.0, 0.0, 6.0, 1.0, 0.0, -11.0, 0.0, 1.0, 6.0], &[3, 3]);
        let (wr, wi) = c.eigvals().unwrap();
        let mut re = vals(&wr);
        re.sort_by(f64::total_cmp);
        for (g, w) in re.iter().zip([1.0, 2.0, 3.0]) {
            assert!((g - w).abs() < 1e-9, "{re:?}");
        }
        assert!(vals(&wi).iter().all(|v| v.abs() < 1e-9));
        // rotation by 90 degrees: +-i
        let rot = f(&[0.0, -1.0, 1.0, 0.0], &[2, 2]);
        let (wr, wi) = rot.eigvals().unwrap();
        close(&wr, &[0.0, 0.0], 1e-12);
        close(&wi, &[1.0, -1.0], 1e-12);
        // a general matrix: A v = lambda v, trace and determinant
        let a = f(&[1.0, 2.0, 0.5, -1.0, 0.5, 3.0, 2.0, -2.0, 0.3, 1.0, 1.0, 0.2, 0.7, -0.4, 0.9, 2.0], &[4, 4]);
        let (wr, wi, vr, vi) = a.eig().unwrap();
        let (wr_v, wi_v, vr_v, vi_v) = (vals(&wr), vals(&wi), vals(&vr), vals(&vi));
        let av = vals(&a);
        for j in 0..4 {
            for i in 0..4 {
                let (mut re, mut im) = (0.0, 0.0);
                for k in 0..4 {
                    re += av[i * 4 + k] * vr_v[k * 4 + j];
                    im += av[i * 4 + k] * vi_v[k * 4 + j];
                }
                let (lre, lim) = (wr_v[j] * vr_v[i * 4 + j] - wi_v[j] * vi_v[i * 4 + j], wr_v[j] * vi_v[i * 4 + j] + wi_v[j] * vr_v[i * 4 + j]);
                assert!((re - lre).abs() < 1e-7 && (im - lim).abs() < 1e-7, "residual at ({i},{j}): {re} {im} vs {lre} {lim}");
            }
        }
        let trace: f64 = (0..4).map(|i| av[i * 5]).sum();
        assert!((wr_v.iter().sum::<f64>() - trace).abs() < 1e-8);
        let det = vals(&a.det().unwrap())[0];
        let (mut pr, mut pi) = (1.0, 0.0);
        for j in 0..4 {
            (pr, pi) = (pr * wr_v[j] - pi * wi_v[j], pr * wi_v[j] + pi * wr_v[j]);
        }
        assert!((pr - det).abs() < 1e-7 * (1.0 + det.abs()) && pi.abs() < 1e-7, "det {det} vs {pr}+{pi}i");
    }
}
