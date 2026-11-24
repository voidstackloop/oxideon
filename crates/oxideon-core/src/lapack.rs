//! LAPACK-style dense factorizations that report failure through an `info` code instead of an error, in the
//! conventions `torch.linalg` uses: partially pivoted LU (`getrf`, 1-based row interchanges), Cholesky (`potrf`),
//! the Bunch-Kaufman `L D L^T` factorization of symmetric indefinite matrices (`sytrf`, LAPACK's compact format with
//! 1x1 and 2x2 pivots) and the solves that use them. Every function works on a batch `[..., m, n]`; the arithmetic
//! is done in `f64` and the results come back in the input's floating-point dtype. These are forward-only (no
//! autograd graph).

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Unblocked LU with partial pivoting of a row-major `m x n` matrix, in place (LAPACK `dgetf2`). `piv[j]` is the
/// 1-based row swapped with row `j`. Returns `info`: 0, or `j + 1` for the first exactly-zero pivot `U[j, j]`.
pub fn getrf(a: &mut [f64], m: usize, n: usize, piv: &mut [i32]) -> i32 {
    let k = m.min(n);
    let mut info = 0;
    for j in 0..k {
        let mut p = j;
        let mut best = a[j * n + j].abs();
        for i in j + 1..m {
            let v = a[i * n + j].abs();
            if v > best {
                best = v;
                p = i;
            }
        }
        piv[j] = (p + 1) as i32;
        if a[p * n + j] != 0.0 {
            if p != j {
                for c in 0..n {
                    a.swap(j * n + c, p * n + c);
                }
            }
            let d = a[j * n + j];
            for i in j + 1..m {
                a[i * n + j] /= d;
            }
        } else if info == 0 {
            info = (j + 1) as i32;
        }
        for i in j + 1..m {
            let l = a[i * n + j];
            if l != 0.0 {
                for c in j + 1..n {
                    a[i * n + c] -= l * a[j * n + c];
                }
            }
        }
    }
    info
}

/// Solves `A X = B` (or `A^T X = B` with `transpose`) in place in the row-major `n x nrhs` `b`, from `getrf`'s output.
pub fn getrs(lu: &[f64], piv: &[i32], n: usize, b: &mut [f64], nrhs: usize, transpose: bool) {
    let row_swap = |b: &mut [f64], i: usize, j: usize| {
        if i != j {
            for c in 0..nrhs {
                b.swap(i * nrhs + c, j * nrhs + c);
            }
        }
    };
    if !transpose {
        for (i, &p) in piv.iter().enumerate() {
            row_swap(b, i, (p - 1) as usize);
        }
        for i in 0..n {
            for k in 0..i {
                let l = lu[i * n + k];
                if l != 0.0 {
                    for c in 0..nrhs {
                        b[i * nrhs + c] -= l * b[k * nrhs + c];
                    }
                }
            }
        }
        for i in (0..n).rev() {
            for k in i + 1..n {
                let u = lu[i * n + k];
                if u != 0.0 {
                    for c in 0..nrhs {
                        b[i * nrhs + c] -= u * b[k * nrhs + c];
                    }
                }
            }
            let d = lu[i * n + i];
            for c in 0..nrhs {
                b[i * nrhs + c] /= d;
            }
        }
    } else {
        // A = P^T L U, so A^T = U^T L^T P: solve U^T, then L^T, then undo the interchanges in reverse order
        for i in 0..n {
            for k in 0..i {
                let u = lu[k * n + i];
                if u != 0.0 {
                    for c in 0..nrhs {
                        b[i * nrhs + c] -= u * b[k * nrhs + c];
                    }
                }
            }
            let d = lu[i * n + i];
            for c in 0..nrhs {
                b[i * nrhs + c] /= d;
            }
        }
        for i in (0..n).rev() {
            for k in i + 1..n {
                let l = lu[k * n + i];
                if l != 0.0 {
                    for c in 0..nrhs {
                        b[i * nrhs + c] -= l * b[k * nrhs + c];
                    }
                }
            }
        }
        for (i, &p) in piv.iter().enumerate().rev() {
            row_swap(b, i, (p - 1) as usize);
        }
    }
}

/// Lower Cholesky factor of a row-major `n x n` matrix (only its lower triangle is read), in place (LAPACK `dpotf2`).
/// Returns `info`: 0, or `j + 1` when the leading minor of order `j + 1` is not positive definite; the non-positive
/// pivot is then left on the diagonal and the rest of the matrix is not touched. The strict upper triangle is zeroed.
pub fn potrf_lower(a: &mut [f64], n: usize) -> i32 {
    for i in 0..n {
        for j in i + 1..n {
            a[i * n + j] = 0.0;
        }
    }
    for j in 0..n {
        let mut ajj = a[j * n + j];
        for k in 0..j {
            ajj -= a[j * n + k] * a[j * n + k];
        }
        if ajj.is_nan() || ajj <= 0.0 {
            a[j * n + j] = ajj;
            return (j + 1) as i32;
        }
        let ajj = ajj.sqrt();
        a[j * n + j] = ajj;
        for i in j + 1..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s / ajj;
        }
    }
    0
}

/// Bunch-Kaufman `L D L^T` factorization of a symmetric row-major `n x n` matrix (only its lower triangle is read), in
/// place (LAPACK `dsytf2`, lower). On return the lower triangle holds L (unit diagonal implied) and the 1x1 / 2x2
/// blocks of D; `ipiv` gets LAPACK's 1-based interchanges, negative (and repeated) for a 2x2 block. Returns `info`:
/// 0, or `k + 1` when `D[k, k]` is exactly zero (D singular). The strict upper triangle is zeroed.
pub fn sytrf_lower(a: &mut [f64], n: usize, ipiv: &mut [i32]) -> i32 {
    let at = |i: usize, j: usize| i * n + j;
    // symmetric working copy, so row and column reads below agree
    for i in 0..n {
        for j in i + 1..n {
            a[at(i, j)] = a[at(j, i)];
        }
    }
    let alpha = (1.0 + 17f64.sqrt()) / 8.0;
    let mut info = 0;
    let mut k = 0;
    while k < n {
        let mut kstep = 1;
        let absakk = a[at(k, k)].abs();
        let (imax, colmax) = if k + 1 < n {
            let mut im = k + 1;
            let mut cm = a[at(k + 1, k)].abs();
            for i in k + 2..n {
                if a[at(i, k)].abs() > cm {
                    cm = a[at(i, k)].abs();
                    im = i;
                }
            }
            (im, cm)
        } else {
            (k, 0.0)
        };
        let kp;
        if absakk.max(colmax) == 0.0 || absakk.is_nan() {
            if info == 0 {
                info = (k + 1) as i32;
            }
            kp = k;
        } else {
            if absakk >= alpha * colmax {
                kp = k;
            } else {
                let mut rowmax = 0.0f64;
                for j in k..imax {
                    rowmax = rowmax.max(a[at(imax, j)].abs());
                }
                for j in imax + 1..n {
                    rowmax = rowmax.max(a[at(j, imax)].abs());
                }
                if absakk >= alpha * colmax * (colmax / rowmax) {
                    kp = k;
                } else if a[at(imax, imax)].abs() >= alpha * rowmax {
                    kp = imax;
                } else {
                    kp = imax;
                    kstep = 2;
                }
            }
            let kk = k + kstep - 1;
            if kp != kk {
                for i in kp + 1..n {
                    a.swap(at(i, kk), at(i, kp));
                }
                for j in kk + 1..kp {
                    a.swap(at(j, kk), at(kp, j));
                }
                a.swap(at(kk, kk), at(kp, kp));
                if kstep == 2 {
                    a.swap(at(k + 1, k), at(kp, k));
                }
            }
            if kstep == 1 {
                if k + 1 < n {
                    let d11 = 1.0 / a[at(k, k)];
                    for j in k + 1..n {
                        let cj = a[at(j, k)];
                        for i in j..n {
                            a[at(i, j)] -= d11 * a[at(i, k)] * cj;
                        }
                    }
                    for i in k + 1..n {
                        a[at(i, k)] *= d11;
                    }
                }
            } else if k + 2 < n {
                let mut d21 = a[at(k + 1, k)];
                let d11 = a[at(k + 1, k + 1)] / d21;
                let d22 = a[at(k, k)] / d21;
                let t = 1.0 / (d11 * d22 - 1.0);
                d21 = t / d21;
                for j in k + 2..n {
                    let wk = d21 * (d11 * a[at(j, k)] - a[at(j, k + 1)]);
                    let wkp1 = d21 * (d22 * a[at(j, k + 1)] - a[at(j, k)]);
                    for i in j..n {
                        a[at(i, j)] -= a[at(i, k)] * wk + a[at(i, k + 1)] * wkp1;
                    }
                    a[at(j, k)] = wk;
                    a[at(j, k + 1)] = wkp1;
                }
            }
        }
        if kstep == 1 {
            ipiv[k] = (kp + 1) as i32;
        } else {
            ipiv[k] = -((kp + 1) as i32);
            ipiv[k + 1] = -((kp + 1) as i32);
        }
        k += kstep;
    }
    for i in 0..n {
        for j in i + 1..n {
            a[at(i, j)] = 0.0;
        }
    }
    info
}

/// Solves `A X = B` in place in the row-major `n x nrhs` `b` from `sytrf_lower`'s output (LAPACK `dsytrs`, lower).
pub fn sytrs_lower(a: &[f64], ipiv: &[i32], n: usize, b: &mut [f64], nrhs: usize) {
    let at = |i: usize, j: usize| i * n + j;
    let swap_rows = |b: &mut [f64], i: usize, j: usize| {
        if i != j {
            for c in 0..nrhs {
                b.swap(i * nrhs + c, j * nrhs + c);
            }
        }
    };
    let mut k = 0;
    while k < n {
        if ipiv[k] > 0 {
            swap_rows(b, k, (ipiv[k] - 1) as usize);
            for i in k + 1..n {
                let l = a[at(i, k)];
                for c in 0..nrhs {
                    b[i * nrhs + c] -= l * b[k * nrhs + c];
                }
            }
            let d = a[at(k, k)];
            for c in 0..nrhs {
                b[k * nrhs + c] /= d;
            }
            k += 1;
        } else {
            swap_rows(b, k + 1, (-ipiv[k] - 1) as usize);
            for i in k + 2..n {
                let (l0, l1) = (a[at(i, k)], a[at(i, k + 1)]);
                for c in 0..nrhs {
                    b[i * nrhs + c] -= l0 * b[k * nrhs + c] + l1 * b[(k + 1) * nrhs + c];
                }
            }
            let akm1k = a[at(k + 1, k)];
            let akm1 = a[at(k, k)] / akm1k;
            let ak = a[at(k + 1, k + 1)] / akm1k;
            let denom = akm1 * ak - 1.0;
            for c in 0..nrhs {
                let bkm1 = b[k * nrhs + c] / akm1k;
                let bk = b[(k + 1) * nrhs + c] / akm1k;
                b[k * nrhs + c] = (ak * bkm1 - bk) / denom;
                b[(k + 1) * nrhs + c] = (akm1 * bk - bkm1) / denom;
            }
            k += 2;
        }
    }
    let mut k = n as isize - 1;
    while k >= 0 {
        let ku = k as usize;
        if ipiv[ku] > 0 {
            for c in 0..nrhs {
                let mut s = 0.0;
                for i in ku + 1..n {
                    s += a[at(i, ku)] * b[i * nrhs + c];
                }
                b[ku * nrhs + c] -= s;
            }
            swap_rows(b, ku, (ipiv[ku] - 1) as usize);
            k -= 1;
        } else {
            for c in 0..nrhs {
                let (mut s0, mut s1) = (0.0, 0.0);
                for i in ku + 1..n {
                    s0 += a[at(i, ku)] * b[i * nrhs + c];
                    s1 += a[at(i, ku - 1)] * b[i * nrhs + c];
                }
                b[ku * nrhs + c] -= s0;
                b[(ku - 1) * nrhs + c] -= s1;
            }
            swap_rows(b, ku, (-ipiv[ku] - 1) as usize);
            k -= 2;
        }
    }
}

/// `[..., m, n]` split into (batch shape, m, n), with a readable error for anything else.
fn mats(t: &Tensor, op: &str) -> Result<(Vec<usize>, usize, usize)> {
    if t.ndim() < 2 {
        return Err(Error::Msg(format!("{op}: expected a tensor of matrices ([..., m, n]), got shape {:?}", t.shape())));
    }
    let s = t.shape();
    Ok((s[..s.len() - 2].to_vec(), s[s.len() - 2], s[s.len() - 1]))
}

fn square(t: &Tensor, op: &str) -> Result<(Vec<usize>, usize)> {
    let (batch, m, n) = mats(t, op)?;
    if m != n {
        return Err(Error::Msg(format!("{op}: expected square matrices, got {m} x {n}")));
    }
    Ok((batch, n))
}

/// The floating-point dtype results come back in (`f32` for integer input).
fn out_dtype(t: &Tensor) -> DType {
    if t.dtype().is_floating_point() { t.dtype() } else { DType::F32 }
}

fn finish(data: &[f64], shape: &[usize], like: &Tensor) -> Result<Tensor> {
    Tensor::from_slice_on(data, shape, &like.device())?.to_dtype(out_dtype(like))
}

fn ints(data: &[i32], shape: &[usize], like: &Tensor) -> Result<Tensor> {
    Tensor::from_slice_on(data, shape, &like.device())
}

/// Right-hand sides for the solves: `[..., n, k]` or (when `vector`) `[..., n]`, broadcast to the matrices' batch.
fn rhs(b: &Tensor, batch: &[usize], n: usize, op: &str) -> Result<(Vec<f64>, usize, bool)> {
    // as in PyTorch: B is a (batch of) vector(s) when it is 1-D or has exactly A's batch shape plus n
    let vector = b.ndim() == 1 || (b.ndim() == batch.len() + 1 && b.shape()[..batch.len()] == *batch && b.shape()[batch.len()] == n);
    let b2 = if vector { b.unsqueeze(-1)? } else { b.clone() };
    let (bb, rows, k) = mats(&b2, op)?;
    if rows != n {
        return Err(Error::Msg(format!("{op}: B has {rows} rows, A is {n} x {n}")));
    }
    let count: usize = batch.iter().product();
    let data = b2.to_vec::<f64>()?;
    let per = n * k;
    let out = if bb.as_slice() == batch {
        data
    } else if bb.is_empty() {
        data.iter().copied().cycle().take(count * per).collect()
    } else {
        return Err(Error::Msg(format!("{op}: the batch shape of B ({bb:?}) does not match A's ({batch:?})")));
    };
    Ok((out, k, vector))
}

impl Tensor {
    /// `torch.linalg.lu_factor_ex`: the packed LU, the 1-based pivots (`i32`) and `info` (`i32`, one per matrix).
    pub fn lu_factor_ex(&self) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch, m, n) = mats(self, "lu_factor_ex")?;
        let k = m.min(n);
        let mut a = self.to_vec::<f64>()?;
        let count: usize = batch.iter().product();
        let mut piv = vec![0i32; count * k];
        let mut info = vec![0i32; count];
        for b in 0..count {
            info[b] = getrf(&mut a[b * m * n..(b + 1) * m * n], m, n, &mut piv[b * k..(b + 1) * k]);
        }
        let mut pshape = batch.clone();
        pshape.push(k);
        Ok((finish(&a, self.shape(), self)?, ints(&piv, &pshape, self)?, ints(&info, &batch, self)?))
    }

    /// Solves `A X = B` (or `X A = B` with `left == false`) from `lu_factor_ex`'s packed LU and 1-based pivots.
    pub fn lu_solve_packed(lu: &Tensor, pivots: &Tensor, b: &Tensor, left: bool) -> Result<Tensor> {
        let (batch, n) = square(lu, "lu_solve")?;
        let a = lu.to_vec::<f64>()?;
        let piv: Vec<i32> = pivots.to_vec::<i64>()?.into_iter().map(|p| p as i32).collect();
        if piv.len() != batch.iter().product::<usize>() * n || piv.iter().any(|&p| p < 1 || p as usize > n) {
            return Err(Error::Msg("lu_solve: pivots must be 1-based row interchanges as returned by lu_factor".into()));
        }
        let count: usize = batch.iter().product();
        if left {
            let (mut x, k, vector) = rhs(b, &batch, n, "lu_solve")?;
            for i in 0..count {
                getrs(&a[i * n * n..(i + 1) * n * n], &piv[i * n..(i + 1) * n], n, &mut x[i * n * k..(i + 1) * n * k], k, false);
            }
            let mut shape = batch.clone();
            shape.push(n);
            if !vector {
                shape.push(k);
            }
            finish(&x, &shape, b)
        } else {
            // X A = B  <=>  A^T X^T = B^T
            let bt = b.transpose(-2, -1)?.contiguous()?;
            let (mut x, k, _) = rhs(&bt, &batch, n, "lu_solve")?;
            for i in 0..count {
                getrs(&a[i * n * n..(i + 1) * n * n], &piv[i * n..(i + 1) * n], n, &mut x[i * n * k..(i + 1) * n * k], k, true);
            }
            let mut shape = batch.clone();
            shape.push(n);
            shape.push(k);
            finish(&x, &shape, b)?.transpose(-2, -1)?.contiguous()
        }
    }

    /// `torch.linalg.inv_ex`: the inverse and `info` (non-zero for a singular matrix, whose "inverse" is then inf / nan).
    pub fn inv_ex(&self) -> Result<(Tensor, Tensor)> {
        let (batch, n) = square(self, "inv_ex")?;
        let (lu, piv, info) = self.lu_factor_ex()?;
        let a = lu.to_dtype(DType::F64)?.to_vec::<f64>()?;
        let p: Vec<i32> = piv.to_vec::<i64>()?.into_iter().map(|v| v as i32).collect();
        let infos = info.to_vec::<i64>()?;
        let count: usize = batch.iter().product();
        let mut out = vec![0.0f64; count * n * n];
        for i in 0..count {
            let x = &mut out[i * n * n..(i + 1) * n * n];
            if infos[i] != 0 {
                x.iter_mut().for_each(|v| *v = f64::INFINITY);
                continue;
            }
            for r in 0..n {
                x[r * n + r] = 1.0;
            }
            getrs(&a[i * n * n..(i + 1) * n * n], &p[i * n..(i + 1) * n], n, x, n, false);
        }
        Ok((finish(&out, self.shape(), self)?, info))
    }

    /// `torch.linalg.solve_ex`: solves `A X = B` (`left`) or `X A = B`, returning `info` instead of failing on a
    /// singular `A` (whose solution is then nan).
    pub fn solve_ex(a: &Tensor, b: &Tensor, left: bool) -> Result<(Tensor, Tensor)> {
        let (lu, piv, info) = a.lu_factor_ex()?;
        let lu64 = lu.to_dtype(DType::F64)?;
        let x = Tensor::lu_solve_packed(&lu64, &piv, b, left)?;
        let bad: Vec<i64> = info.to_vec::<i64>()?;
        if bad.iter().all(|&v| v == 0) {
            return Ok((x.to_dtype(out_dtype(a))?, info));
        }
        let mut data = x.to_vec::<f64>()?;
        let per = data.len() / bad.len().max(1);
        for (i, &v) in bad.iter().enumerate() {
            if v != 0 {
                data[i * per..(i + 1) * per].iter_mut().for_each(|e| *e = f64::NAN);
            }
        }
        Ok((finish(&data, x.shape(), a)?, info))
    }

    /// `torch.linalg.cholesky_ex`: the lower (or, with `upper`, upper) Cholesky factor and `info` (0, or the order of
    /// the first leading minor that is not positive definite).
    pub fn cholesky_ex(&self, upper: bool) -> Result<(Tensor, Tensor)> {
        let (batch, n) = square(self, "cholesky_ex")?;
        let mut a = self.to_vec::<f64>()?;
        let count: usize = batch.iter().product();
        let mut info = vec![0i32; count];
        for i in 0..count {
            info[i] = potrf_lower(&mut a[i * n * n..(i + 1) * n * n], n);
        }
        let l = finish(&a, self.shape(), self)?;
        let l = if upper { l.transpose(-2, -1)?.contiguous()? } else { l };
        Ok((l, ints(&info, &batch, self)?))
    }

    /// `torch.linalg.ldl_factor_ex`: the compact Bunch-Kaufman factorization `LD`, LAPACK's 1-based pivots (negative
    /// for 2x2 blocks) and `info` (non-zero when D is singular). Only the lower triangle of the input is read.
    pub fn ldl_factor_ex(&self) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch, n) = square(self, "ldl_factor_ex")?;
        let mut a = self.to_vec::<f64>()?;
        let count: usize = batch.iter().product();
        let mut piv = vec![0i32; count * n];
        let mut info = vec![0i32; count];
        for i in 0..count {
            info[i] = sytrf_lower(&mut a[i * n * n..(i + 1) * n * n], n, &mut piv[i * n..(i + 1) * n]);
        }
        let mut pshape = batch.clone();
        pshape.push(n);
        Ok((finish(&a, self.shape(), self)?, ints(&piv, &pshape, self)?, ints(&info, &batch, self)?))
    }

    /// `torch.linalg.ldl_solve`: solves `A X = B` from `ldl_factor_ex`'s result.
    pub fn ldl_solve(ld: &Tensor, pivots: &Tensor, b: &Tensor) -> Result<Tensor> {
        let (batch, n) = square(ld, "ldl_solve")?;
        let a = ld.to_vec::<f64>()?;
        let piv: Vec<i32> = pivots.to_vec::<i64>()?.into_iter().map(|p| p as i32).collect();
        let count: usize = batch.iter().product();
        if piv.len() != count * n || piv.iter().any(|&p| p == 0 || p.unsigned_abs() as usize > n) {
            return Err(Error::Msg("ldl_solve: pivots must be LAPACK interchanges as returned by ldl_factor".into()));
        }
        let (mut x, k, vector) = rhs(b, &batch, n, "ldl_solve")?;
        for i in 0..count {
            sytrs_lower(&a[i * n * n..(i + 1) * n * n], &piv[i * n..(i + 1) * n], n, &mut x[i * n * k..(i + 1) * n * k], k);
        }
        let mut shape = batch.clone();
        shape.push(n);
        if !vector {
            shape.push(k);
        }
        finish(&x, &shape, b)
    }
}

/// Householder QR of a row-major `m x n` matrix in LAPACK's convention (`dgeqr2` + `dorg2r`, so the signs agree with
/// NumPy and PyTorch): returns `Q` (`m x m` when `complete`, else `m x k`) and `R` (`m x n` / `k x n`), `k = min(m, n)`.
pub fn householder_qr(a: &[f64], m: usize, n: usize, complete: bool) -> (Vec<f64>, Vec<f64>) {
    let k = m.min(n);
    let mut r = a.to_vec();
    let mut taus = vec![0.0; k];
    for j in 0..k {
        let alpha = r[j * n + j];
        let xnorm = (j + 1..m).map(|i| r[i * n + j] * r[i * n + j]).sum::<f64>().sqrt();
        if xnorm == 0.0 {
            continue; // H = I
        }
        let beta = -alpha.signum() * alpha.hypot(xnorm);
        let beta = if alpha == 0.0 { -alpha.hypot(xnorm) } else { beta };
        taus[j] = (beta - alpha) / beta;
        let scale = 1.0 / (alpha - beta);
        for i in j + 1..m {
            r[i * n + j] *= scale;
        }
        r[j * n + j] = beta;
        // apply H = I - tau v v^T (v = [1, r[j+1.., j]]) to the trailing columns
        for c in j + 1..n {
            let mut dot = r[j * n + c];
            for i in j + 1..m {
                dot += r[i * n + j] * r[i * n + c];
            }
            let t = taus[j] * dot;
            r[j * n + c] -= t;
            for i in j + 1..m {
                r[i * n + c] -= t * r[i * n + j];
            }
        }
    }
    let qc = if complete { m } else { k };
    let mut q = vec![0.0; m * qc];
    for i in 0..qc.min(m) {
        q[i * qc + i] = 1.0;
    }
    for j in (0..k).rev() {
        if taus[j] == 0.0 {
            continue;
        }
        for c in 0..qc {
            let mut dot = q[j * qc + c];
            for i in j + 1..m {
                dot += r[i * n + j] * q[i * qc + c];
            }
            let t = taus[j] * dot;
            q[j * qc + c] -= t;
            for i in j + 1..m {
                q[i * qc + c] -= t * r[i * n + j];
            }
        }
    }
    let rr = if complete { m } else { k };
    let mut rout = vec![0.0; rr * n];
    for i in 0..rr.min(k) {
        for c in i..n {
            rout[i * n + c] = r[i * n + c];
        }
    }
    (q, rout)
}

impl Tensor {
    /// `torch.linalg.qr` for any `[..., m, n]` shape: `mode` reduced (`complete == false`) or complete, computed with
    /// LAPACK-convention Householder reflections (forward-only).
    pub fn qr_householder(&self, complete: bool) -> Result<(Tensor, Tensor)> {
        let (batch, m, n) = mats(self, "qr")?;
        let k = m.min(n);
        let a = self.to_vec::<f64>()?;
        let count: usize = batch.iter().product();
        let (qc, rr) = if complete { (m, m) } else { (k, k) };
        let mut qs = Vec::with_capacity(count * m * qc);
        let mut rs = Vec::with_capacity(count * rr * n);
        for i in 0..count {
            let (q, r) = householder_qr(&a[i * m * n..(i + 1) * m * n], m, n, complete);
            qs.extend(q);
            rs.extend(r);
        }
        let mut qshape = batch.clone();
        qshape.extend([m, qc]);
        let mut rshape = batch.clone();
        rshape.extend([rr, n]);
        Ok((finish(&qs, &qshape, self)?, finish(&rs, &rshape, self)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matmul(a: &[f64], b: &[f64], n: usize, k: usize) -> Vec<f64> {
        let mut out = vec![0.0; n * k];
        for i in 0..n {
            for j in 0..k {
                out[i * k + j] = (0..n).map(|t| a[i * n + t] * b[t * k + j]).sum();
            }
        }
        out
    }

    fn sym(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let v = next();
                a[i * n + j] = v;
                a[j * n + i] = v;
            }
        }
        a
    }

    #[test]
    fn lu_and_ldl_solve_symmetric_indefinite_systems() {
        for n in [1usize, 2, 3, 5, 9, 16] {
            for seed in 0..20u64 {
                let mut a = sym(n, seed);
                if seed % 4 == 0 {
                    a[0] = 0.0; // force a 2x2 pivot or an interchange
                }
                let x_true: Vec<f64> = (0..n * 2).map(|i| (i as f64 * 0.37).sin()).collect();
                let b = matmul(&a, &x_true, n, 2);
                let mut ld = a.clone();
                let mut piv = vec![0i32; n];
                if sytrf_lower(&mut ld, n, &mut piv) == 0 {
                    let mut x = b.clone();
                    sytrs_lower(&ld, &piv, n, &mut x, 2);
                    for (u, v) in x.iter().zip(&x_true) {
                        assert!((u - v).abs() < 1e-6, "ldl n={n} seed={seed}");
                    }
                }
                let mut lu = a.clone();
                let mut p = vec![0i32; n];
                if getrf(&mut lu, n, n, &mut p) == 0 {
                    let mut x = b.clone();
                    getrs(&lu, &p, n, &mut x, 2, false);
                    for (u, v) in x.iter().zip(&x_true) {
                        assert!((u - v).abs() < 1e-6, "lu n={n} seed={seed}");
                    }
                    // A^T X = A^T x_true
                    let mut at = vec![0.0; n * n];
                    for i in 0..n {
                        for j in 0..n {
                            at[i * n + j] = a[j * n + i];
                        }
                    }
                    let mut y = matmul(&at, &x_true, n, 2);
                    getrs(&lu, &p, n, &mut y, 2, true);
                    for (u, v) in y.iter().zip(&x_true) {
                        assert!((u - v).abs() < 1e-6, "lu transpose n={n} seed={seed}");
                    }
                }
            }
        }
    }

    #[test]
    fn info_codes() {
        let mut z = vec![0.0; 9];
        let mut piv = vec![0i32; 3];
        assert_eq!(sytrf_lower(&mut z.clone(), 3, &mut piv), 1);
        assert_eq!(getrf(&mut z, 3, 3, &mut piv), 1);
        let mut not_pd = vec![1.0, 2.0, 2.0, 1.0];
        assert_eq!(potrf_lower(&mut not_pd, 2), 2);
        assert_eq!(not_pd[3], -3.0); // the failing pivot is left in place, as LAPACK does
        let mut spd = vec![4.0, 2.0, 2.0, 3.0];
        assert_eq!(potrf_lower(&mut spd, 2), 0);
        assert!((spd[0] - 2.0).abs() < 1e-15 && (spd[2] - 1.0).abs() < 1e-15 && spd[1] == 0.0);
    }
}
