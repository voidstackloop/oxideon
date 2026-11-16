//! Advanced differentiable linear algebra: norms, pairwise distances (`cdist`),
//! `cholesky`, `triangular_solve`, `solve`, `inv`, `det`, `slogdet`, and `qr`.

use crate::backend::{CpuStorage, Storage};
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::tensor::{Tensor, no_grad};

impl Tensor {
    /// $L_p$ vector/tensor norm along `axes` (Frobenius norm when $p = 2$).
    pub fn norm_p(&self, p: f64, axes: &[isize], keepdim: bool) -> Result<Tensor> {
        if p.is_nan() {
            return Err(Error::Msg("norm_p: p must not be NaN".into()));
        }
        if p == 0.0 {
            // Zero-"norm": count of nonzero elements (torch parity).
            let zero = self.zeros_like()?;
            return self
                .abs()?
                .gt(&zero)?
                .to_dtype(self.dtype())?
                .sum_axes(axes, keepdim);
        }
        if p == 2.0 {
            // No epsilon floor: the norm of a zero vector is exactly 0
            // (like torch; the gradient there is NaN, also like torch).
            self.square()?.sum_axes(axes, keepdim)?.sqrt()
        } else if p == 1.0 {
            self.abs()?.sum_axes(axes, keepdim)
        } else if p.is_infinite() && p > 0.0 {
            if axes.len() == 1 {
                self.abs()?.amax(&[axes[0]], keepdim) // ties share the gradient, like torch
            } else {
                Err(Error::Msg(
                    "inf-norm currently supports a single axis".into(),
                ))
            }
        } else {
            self.abs()?.powf(p)?.sum_axes(axes, keepdim)?.powf(1.0 / p)
        }
    }

    /// Global Frobenius / L2 norm of all elements.
    pub fn norm(&self) -> Result<Tensor> {
        let axes: Vec<isize> = (0..self.ndim() as isize).collect();
        self.norm_p(2.0, &axes, false)
    }

    /// Normalizes `self` along `axis` to unit $L_p$ norm: `x / max(||x||_p, eps)`.
    pub fn normalize(&self, p: f64, axis: isize, eps: f64) -> Result<Tensor> {
        let denom = self.norm_p(p, &[axis], true)?.clamp(eps, f64::INFINITY)?;
        self.div(&denom)
    }

    /// Pairwise Euclidean distance matrix between `[..., n, d]` and `[..., m, d]` -> `[..., n, m]`.
    pub fn cdist(&self, other: &Tensor) -> Result<Tensor> {
        self.cdist_p(other, 2.0)
    }

    /// Lower-triangular Cholesky decomposition $A = L L^T$ for symmetric positive-definite
    /// matrix (or batch of matrices) `[..., n, n]`.
    ///
    /// Differentiable! Backward uses the closed-form SPD Cholesky adjoint:
    /// $\nabla A = L^{-T} \Phi(L^T \nabla L) L^{-1}$, where $\Phi(M) = \text{tril}(M) - \frac{1}{2}\text{diag}(M)$.
    pub fn cholesky(&self) -> Result<Tensor> {
        let (batch, n) = check_square_batch("cholesky", self.shape())?;
        let data = self.to_vec::<f64>()?;
        let mut out = vec![0.0f64; batch * n * n];

        for b in 0..batch {
            let a = &data[b * n * n..(b + 1) * n * n];
            let l = &mut out[b * n * n..(b + 1) * n * n];
            for i in 0..n {
                for j in 0..=i {
                    let mut sum = a[i * n + j];
                    for k in 0..j {
                        sum -= l[i * n + k] * l[j * n + k];
                    }
                    if i == j {
                        if sum <= 0.0 {
                            return Err(Error::Msg(format!(
                                "cholesky: matrix at batch {b} is not positive definite (pivot {i} = {sum})"
                            )));
                        }
                        l[i * n + j] = sum.sqrt();
                    } else {
                        l[i * n + j] = sum / l[j * n + j];
                    }
                }
            }
        }

        let shape = self.shape().to_vec();
        let cpu =
            CpuStorage::F64(out).to_dtype(&Layout::contiguous(shape.clone()), self.dtype())?;
        let storage =
            Storage::new_cpu(cpu).to_device(&Layout::contiguous(shape.clone()), &self.device())?;
        let l_tensor = Tensor::from_storage(storage.clone(), Layout::contiguous(shape.clone()));

        Ok(Tensor::from_op(
            storage,
            Layout::contiguous(shape),
            "cholesky",
            vec![self.clone()],
            move |g| {
                // Phi(M) = tril(M, -1) + 0.5 * diag(M) = tril(M, 0) - 0.5 * diag(M)
                let lt_g = l_tensor.t()?.matmul(g)?;
                let strict_lower = lt_g.tril(-1)?;
                let diag_only = lt_g.tril(0)?.triu(0)?.mul_scalar(0.5)?;
                let phi = strict_lower.add(&diag_only)?;
                // Solve L^T X = phi  => X = L^{-T} phi, then solve L^T Y = X^T => grad_A = 0.5 * (Y + Y^T)
                let x = l_tensor.triangular_solve(&phi, true, true)?;
                let y = l_tensor.triangular_solve(&x.t()?, true, true)?;
                let sym = y.add(&y.t()?)?.mul_scalar(0.5)?;
                Ok(vec![Some(sym)])
            },
        ))
    }

    /// Solves $op(L) X = B$ where $L$ (`self`) is `[..., n, n]` triangular (`lower = true` or
    /// `false`) and optionally transposed (`transpose_a = true` solves $L^T X = B$).
    /// Differentiable with respect to both `self` and `rhs`!
    pub fn triangular_solve(&self, rhs: &Tensor, lower: bool, transpose_a: bool) -> Result<Tensor> {
        let (batch, n) = check_square_batch("triangular_solve", self.shape())?;
        if rhs.ndim() < 2 || rhs.shape()[rhs.ndim() - 2] != n {
            return Err(Error::ShapeMismatch {
                op: "triangular_solve",
                expected: self.shape().to_vec(),
                got: rhs.shape().to_vec(),
            });
        }
        let rhs_batch: usize = rhs.shape()[..rhs.ndim() - 2].iter().product();
        if rhs_batch != batch {
            return Err(Error::ShapeMismatch {
                op: "triangular_solve",
                expected: self.shape().to_vec(),
                got: rhs.shape().to_vec(),
            });
        }
        let k_cols = rhs.shape()[rhs.ndim() - 1];
        let a_data = self.to_vec::<f64>()?;
        let b_data = rhs.to_vec::<f64>()?;
        let mut x_data = vec![0.0f64; batch * n * k_cols];

        let effective_lower = lower ^ transpose_a;
        for b in 0..batch {
            let a = &a_data[b * n * n..(b + 1) * n * n];
            let rhs_b = &b_data[b * n * k_cols..(b + 1) * n * k_cols];
            let x = &mut x_data[b * n * k_cols..(b + 1) * n * k_cols];

            let get_a = |r: usize, c: usize| -> f64 {
                if transpose_a {
                    a[c * n + r]
                } else {
                    a[r * n + c]
                }
            };

            if effective_lower {
                for i in 0..n {
                    let diag = get_a(i, i);
                    if diag == 0.0 {
                        return Err(Error::Msg(format!(
                            "triangular_solve: singular matrix (zero diagonal at row {i} of batch {b})"
                        )));
                    }
                    for col in 0..k_cols {
                        let mut sum = rhs_b[i * k_cols + col];
                        for j in 0..i {
                            sum -= get_a(i, j) * x[j * k_cols + col];
                        }
                        x[i * k_cols + col] = sum / diag;
                    }
                }
            } else {
                for i in (0..n).rev() {
                    let diag = get_a(i, i);
                    if diag == 0.0 {
                        return Err(Error::Msg(format!(
                            "triangular_solve: singular matrix (zero diagonal at row {i} of batch {b})"
                        )));
                    }
                    for col in 0..k_cols {
                        let mut sum = rhs_b[i * k_cols + col];
                        for j in (i + 1)..n {
                            sum -= get_a(i, j) * x[j * k_cols + col];
                        }
                        x[i * k_cols + col] = sum / diag;
                    }
                }
            }
        }

        let out_shape = rhs.shape().to_vec();
        let cpu = CpuStorage::F64(x_data)
            .to_dtype(&Layout::contiguous(out_shape.clone()), self.dtype())?;
        let storage = Storage::new_cpu(cpu)
            .to_device(&Layout::contiguous(out_shape.clone()), &self.device())?;
        let x_tensor = Tensor::from_storage(storage.clone(), Layout::contiguous(out_shape.clone()));
        let a_in = self.clone();
        let b_in = rhs.clone();

        Ok(Tensor::from_op(
            storage,
            Layout::contiguous(out_shape),
            "triangular_solve",
            vec![self.clone(), rhs.clone()],
            move |g| {
                // grad_B = op(A)^{-T} g
                let grad_b = a_in.triangular_solve(g, lower, !transpose_a)?;
                let gb = if b_in.requires_grad() {
                    Some(grad_b.clone())
                } else {
                    None
                };
                let ga = if a_in.requires_grad() {
                    let raw = if transpose_a {
                        x_tensor.matmul(&grad_b.t()?)?.neg()?
                    } else {
                        grad_b.matmul(&x_tensor.t()?)?.neg()?
                    };
                    let masked = if lower { raw.tril(0)? } else { raw.triu(0)? };
                    Some(masked)
                } else {
                    None
                };
                Ok(vec![ga, gb])
            },
        ))
    }

    /// Solves a general linear system $A X = B$ for square (or batched square) $A: [..., n, n]$
    /// and $B: [..., n, k]$ using Gaussian elimination with partial pivoting.
    /// Differentiable with respect to both $A$ and $B$:
    /// $\nabla B = A^{-T} \nabla X$, $\nabla A = -(\nabla B) X^T$.
    pub fn solve(&self, rhs: &Tensor) -> Result<Tensor> {
        let (batch, n) = check_square_batch("solve", self.shape())?;
        if rhs.ndim() < 2 || rhs.shape()[rhs.ndim() - 2] != n {
            return Err(Error::ShapeMismatch {
                op: "solve",
                expected: self.shape().to_vec(),
                got: rhs.shape().to_vec(),
            });
        }
        let rhs_batch: usize = rhs.shape()[..rhs.ndim() - 2].iter().product();
        if rhs_batch != batch {
            return Err(Error::ShapeMismatch {
                op: "solve",
                expected: self.shape().to_vec(),
                got: rhs.shape().to_vec(),
            });
        }
        let k_cols = rhs.shape()[rhs.ndim() - 1];
        let a_data = self.to_vec::<f64>()?;
        let b_data = rhs.to_vec::<f64>()?;
        let mut x_data = vec![0.0f64; batch * n * k_cols];

        for b in 0..batch {
            let a = &a_data[b * n * n..(b + 1) * n * n];
            let rhs_b = &b_data[b * n * k_cols..(b + 1) * n * k_cols];
            let sol = lu_solve_single(a, rhs_b, n, k_cols)?;
            x_data[b * n * k_cols..(b + 1) * n * k_cols].copy_from_slice(&sol);
        }

        let out_shape = rhs.shape().to_vec();
        let cpu = CpuStorage::F64(x_data)
            .to_dtype(&Layout::contiguous(out_shape.clone()), self.dtype())?;
        let storage = Storage::new_cpu(cpu)
            .to_device(&Layout::contiguous(out_shape.clone()), &self.device())?;
        let x_tensor = Tensor::from_storage(storage.clone(), Layout::contiguous(out_shape.clone()));
        let a_in = self.clone();
        let b_in = rhs.clone();

        Ok(Tensor::from_op(
            storage,
            Layout::contiguous(out_shape),
            "solve",
            vec![self.clone(), rhs.clone()],
            move |g| {
                let grad_b = a_in.t()?.solve(g)?;
                let gb = if b_in.requires_grad() {
                    Some(grad_b.clone())
                } else {
                    None
                };
                let ga = if a_in.requires_grad() {
                    Some(grad_b.matmul(&x_tensor.t()?)?.neg()?)
                } else {
                    None
                };
                Ok(vec![ga, gb])
            },
        ))
    }

    /// Matrix inverse $A^{-1}$ for square (or batched square) $A: [..., n, n]$.
    /// Differentiable ($\nabla A = -A^{-T} (\nabla A^{-1}) A^{-T}$ via [`Tensor::solve`])!
    pub fn inv(&self) -> Result<Tensor> {
        let (_, n) = check_square_batch("inv", self.shape())?;
        let eye = Tensor::eye(n, self.dtype(), &self.device())?.broadcast_to(self.shape())?;
        self.solve(&eye)
    }

    /// Sign and natural log of absolute determinant `(sign, logabsdet)` of `[..., n, n]`.
    /// `logabsdet` is differentiable ($\nabla_A \ln|\det A| = A^{-T}$)!
    pub fn slogdet(&self) -> Result<(Tensor, Tensor)> {
        let (batch, n) = check_square_batch("slogdet", self.shape())?;
        let a_data = self.to_vec::<f64>()?;
        let mut signs = vec![0.0f64; batch];
        let mut logdets = vec![0.0f64; batch];

        for b in 0..batch {
            let a = &a_data[b * n * n..(b + 1) * n * n];
            let (s, ld) = lu_slogdet_single(a, n);
            signs[b] = s;
            logdets[b] = ld;
        }

        let batch_shape = self.shape()[..self.ndim() - 2].to_vec();
        let sign_cpu = CpuStorage::F64(signs)
            .to_dtype(&Layout::contiguous(batch_shape.clone()), self.dtype())?;
        let ld_cpu = CpuStorage::F64(logdets)
            .to_dtype(&Layout::contiguous(batch_shape.clone()), self.dtype())?;

        let sign_t = Tensor::from_storage(
            Storage::new_cpu(sign_cpu)
                .to_device(&Layout::contiguous(batch_shape.clone()), &self.device())?,
            Layout::contiguous(batch_shape.clone()),
        );
        let ld_storage = Storage::new_cpu(ld_cpu)
            .to_device(&Layout::contiguous(batch_shape.clone()), &self.device())?;
        let a_in = self.clone();

        let logdet_t = Tensor::from_op(
            ld_storage,
            Layout::contiguous(batch_shape),
            "slogdet",
            vec![self.clone()],
            move |g| {
                let mut g_expanded_shape = a_in.shape().to_vec();
                let r = g_expanded_shape.len();
                g_expanded_shape[r - 2] = 1;
                g_expanded_shape[r - 1] = 1;
                let g_exp = g.reshape(&g_expanded_shape)?;
                let inv_t = a_in.inv()?.t()?;
                Ok(vec![Some(inv_t.mul(&g_exp)?)])
            },
        );
        Ok((sign_t, logdet_t))
    }

    /// Determinant $\det(A)$ of a square (or batched square) matrix `[..., n, n]`.
    /// Differentiable ($\nabla_A \det(A) = \det(A) A^{-T}$)!
    pub fn det(&self) -> Result<Tensor> {
        let (sign, logabsdet) = no_grad(|| self.slogdet())?;
        let det_val = sign.mul(&logabsdet.exp()?)?;
        let a_in = self.clone();
        let det_for_bwd = det_val.clone();
        Ok(Tensor::from_op(
            det_val.storage().clone(),
            det_val.layout().clone(),
            "det",
            vec![self.clone()],
            move |g| {
                let mut exp_shape = a_in.shape().to_vec();
                let r = exp_shape.len();
                exp_shape[r - 2] = 1;
                exp_shape[r - 1] = 1;
                let scale = g.mul(&det_for_bwd)?.reshape(&exp_shape)?;
                let inv_t = a_in.inv()?.t()?;
                Ok(vec![Some(inv_t.mul(&scale)?)])
            },
        ))
    }

    /// Reduced QR decomposition $A = Q R$ of a 2D matrix `[m, n]` ($m \ge n$), returning
    /// `(Q: [m, n], R: [n, n])` via Householder reflections. `Q` is orthonormal
    /// to machine precision, including for rank-deficient inputs (whose `R`
    /// diagonal entries come out ~0 there).
    pub fn qr(&self) -> Result<(Tensor, Tensor)> {
        if self.ndim() > 2 {
            let mut v = self.batched_mats(|m| {
                let (q, r) = m.qr()?;
                Ok(vec![q, r])
            })?;
            let r = v.remove(1);
            return Ok((v.remove(0), r));
        }
        if self.ndim() != 2 {
            return Err(Error::Msg("qr requires a 2D matrix [m, n]".into()));
        }
        let (m, n) = (self.shape()[0], self.shape()[1]);
        if m < n {
            return Err(Error::Msg("qr reduced mode requires m >= n".into()));
        }
        let a = self.to_vec::<f64>()?;
        // R starts as A with reflectors H_0..H_{n-1} applied from the left;
        // Q is their product applied to the first n columns of I_m (in
        // reverse, so only the reduced [m, n] block is ever materialized).
        let mut r = a;
        let mut vs = vec![0.0f64; m * n]; // unit Householder vectors, one per column
        for k in 0..n {
            let mut sigma_sq = 0.0f64;
            for i in k..m {
                sigma_sq += r[i * n + k] * r[i * n + k];
            }
            let sigma = sigma_sq.sqrt();
            if sigma == 0.0 {
                // Already-orthogonal column (rank deficiency): H_k = I.
                // The zero-initialized reflector below acts as identity.
                continue;
            }
            // u = x - alpha e_0 with alpha = -sign(x_0) ||x||; v = u / ||u||.
            // ||u||^2 >= 2 sigma^2 > 0, so this division always succeeds.
            let alpha = -sigma.copysign(r[k * n + k]);
            let mut norm_u_sq = 0.0f64;
            for i in k..m {
                let mut u = r[i * n + k];
                if i == k {
                    u -= alpha;
                }
                vs[i * n + k] = u;
                norm_u_sq += u * u;
            }
            let norm_u = norm_u_sq.sqrt();
            for i in k..m {
                vs[i * n + k] /= norm_u;
            }
            // R[k.., k..] -= 2 v (v^T R).
            for j in k..n {
                let mut dot = 0.0f64;
                for i in k..m {
                    dot += vs[i * n + k] * r[i * n + j];
                }
                let c = 2.0 * dot;
                for i in k..m {
                    r[i * n + j] -= c * vs[i * n + k];
                }
            }
            // Explicitly zero the eliminated subdiagonal.
            for i in (k + 1)..m {
                r[i * n + k] = 0.0;
            }
        }
        let mut q = vec![0.0f64; m * n];
        for i in 0..m {
            if i < n {
                q[i * n + i] = 1.0;
            }
        }
        for k in (0..n).rev() {
            for j in 0..n {
                let mut dot = 0.0f64;
                for i in k..m {
                    dot += vs[i * n + k] * q[i * n + j];
                }
                let c = 2.0 * dot;
                for i in k..m {
                    q[i * n + j] -= c * vs[i * n + k];
                }
            }
        }

        let q_leaf = Tensor::from_slice_on(&q, &[m, n], &self.device())?.to_dtype(self.dtype())?;
        // Reduced R is the top n x n block (rows below are exactly zero).
        let mut r_top = vec![0.0f64; n * n];
        for i in 0..n {
            r_top[i * n..(i + 1) * n].copy_from_slice(&r[i * n..(i + 1) * n]);
        }
        let r_leaf =
            Tensor::from_slice_on(&r_top, &[n, n], &self.device())?.to_dtype(self.dtype())?;
        // Differentiable via Giles (2008): dA = (dQ + Q @ E) @ R^{-T} with
        // E = tril(M) + tril(M,-1)^T (diagonal kept) and M = R dR^T - dQ^T Q.
        // Each output carries the branch with the other cotangent zeroed.
        let q_for_q = q_leaf.clone();
        let r_for_q = r_leaf.clone();
        let qt = Tensor::from_op(
            q_leaf.storage().clone(),
            q_leaf.layout().clone(),
            "qr_Q",
            vec![self.clone()],
            move |gq| {
                let c = gq.t()?.matmul(&q_for_q)?.neg()?;
                let e = c.tril(0)?.add(&c.tril(-1)?.t()?)?;
                let d = gq.add(&q_for_q.matmul(&e)?)?;
                let z = r_for_q.triangular_solve(&d.t()?, false, false)?;
                Ok(vec![Some(z.t()?)])
            },
        );
        let q_for_r = q_leaf.clone();
        let r_for_r = r_leaf.clone();
        let rt = Tensor::from_op(
            r_leaf.storage().clone(),
            r_leaf.layout().clone(),
            "qr_R",
            vec![self.clone()],
            move |gr| {
                // Giles (2008): dA = Q E R^{-T} with E = tril(C) + tril(C,-1)^T
                // and C = R Gr^T (the Q cotangent is zero on this branch).
                let c = r_for_r.matmul(&gr.t()?)?;
                let e = c.tril(0)?.add(&c.tril(-1)?.t()?)?;
                let d = q_for_r.matmul(&e)?;
                let z = r_for_r.triangular_solve(&d.t()?, false, false)?;
                Ok(vec![Some(z.t()?)])
            },
        );
        Ok((qt, rt))
    }

    /// Symmetric eigenvalue decomposition $A = V \operatorname{diag}(\lambda) V^T$ for a 2D
    /// symmetric matrix `[n, n]`, returning `(eigenvalues: [n], eigenvectors: [n, n])` sorted
    /// in ascending order of eigenvalues.
    ///
    /// `eigenvalues` are differentiable ($\nabla_A \lambda_k = v_k v_k^T$, i.e.
    /// $\nabla A = V \operatorname{diag}(\nabla \lambda) V^T$)!
    pub fn eigh(&self) -> Result<(Tensor, Tensor)> {
        if self.ndim() > 2 {
            let mut v = self.batched_mats(|m| {
                let (w, q) = m.eigh()?;
                Ok(vec![w, q])
            })?;
            let q = v.remove(1);
            return Ok((v.remove(0), q));
        }
        if self.ndim() != 2 || self.shape()[0] != self.shape()[1] {
            return Err(Error::Msg(
                "eigh requires a 2D square symmetric matrix [n, n]".into(),
            ));
        }
        let n = self.shape()[0];
        let a = self.to_vec::<f64>()?;
        let (vals, vecs) = jacobi_eigh(&a, n);

        let val_cpu =
            CpuStorage::F64(vals.clone()).to_dtype(&Layout::contiguous(vec![n]), self.dtype())?;
        let vec_cpu =
            CpuStorage::F64(vecs).to_dtype(&Layout::contiguous(vec![n, n]), self.dtype())?;

        // Build F_{ij} = 1 / (lambda_j - lambda_i) for i != j, 0 on diagonal
        let mut f_mat = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let diff = vals[j] - vals[i];
                    if diff.abs() > 1e-12 {
                        f_mat[i * n + j] = 1.0 / diff;
                    }
                }
            }
        }
        let f_tensor =
            Tensor::from_slice_on(&f_mat, &[n, n], &self.device())?.to_dtype(self.dtype())?;

        let v_storage =
            Storage::new_cpu(vec_cpu).to_device(&Layout::contiguous(vec![n, n]), &self.device())?;
        let v_detached = Tensor::from_storage(v_storage.clone(), Layout::contiguous(vec![n, n]));
        let v_for_vec_bwd = v_detached.clone();
        let v_tensor = Tensor::from_op(
            v_storage,
            Layout::contiguous(vec![n, n]),
            "eigh_vectors",
            vec![self.clone()],
            move |gv| {
                // grad_A = 0.5 * V @ (F * (V^T @ gv) + (F * (V^T @ gv))^T) @ V^T
                let vt_gv = v_for_vec_bwd.t()?.matmul(gv)?;
                let m = f_tensor.mul(&vt_gv)?;
                let m_sym = m.add(&m.t()?)?.mul_scalar(0.5)?;
                let grad_a = v_for_vec_bwd.matmul(&m_sym)?.matmul(&v_for_vec_bwd.t()?)?;
                Ok(vec![Some(grad_a)])
            },
        );
        let val_storage =
            Storage::new_cpu(val_cpu).to_device(&Layout::contiguous(vec![n]), &self.device())?;
        let v_for_bwd = v_detached;

        let val_tensor = Tensor::from_op(
            val_storage,
            Layout::contiguous(vec![n]),
            "eigh_values",
            vec![self.clone()],
            move |g| {
                // grad_A = V @ diag(g) @ V^T = (V * g.unsqueeze(0)) @ V^T
                let scaled_v = v_for_bwd.mul(&g.unsqueeze(0)?)?;
                Ok(vec![Some(scaled_v.matmul(&v_for_bwd.t()?)?)])
            },
        );
        Ok((val_tensor, v_tensor))
    }

    /// Reduced Singular Value Decomposition $A = U \operatorname{diag}(S) V^T$ of a 2D matrix
    /// `[m, n]`, returning `(U: [m, k], S: [k], Vt: [k, n])` where $k = \min(m, n)$ and singular
    /// values $S$ are sorted in descending order.
    ///
    /// All three outputs are differentiable: $\nabla_A S = U \operatorname{diag}(\nabla S) V^T$, and `U` / `Vt` use the
    /// standard (Townsend 2016) backward, valid for distinct nonzero singular values.
    /// NaN entries propagate into the outputs (PyTorch raises on non-finite input instead).
    pub fn svd(&self) -> Result<(Tensor, Tensor, Tensor)> {
        if self.ndim() > 2 {
            let mut v = self.batched_mats(|m| {
                let (u, s, vt) = m.svd()?;
                Ok(vec![u, s, vt])
            })?;
            let vt = v.remove(2);
            let s = v.remove(1);
            return Ok((v.remove(0), s, vt));
        }
        if self.ndim() != 2 {
            return Err(Error::Msg("svd requires a 2D matrix [m, n]".into()));
        }
        let (m, n) = (self.shape()[0], self.shape()[1]);
        let k = m.min(n);
        let a_f64 = self.to_dtype(crate::dtype::DType::F64)?.detach();

        let (u_f64, s_vec, vt_f64) = if m >= n {
            // A^T A = V S^2 V^T
            let ata = a_f64.t()?.matmul(&a_f64)?;
            let (evals, v_asc) = jacobi_eigh(&ata.to_vec::<f64>()?, n);
            let mut s = Vec::with_capacity(k);
            let mut v_cols = vec![0.0f64; n * k];
            for col in 0..k {
                let src_col = n - 1 - col;
                // Clamp only small negative roundoff: NaN must propagate,
                // not be silenced to 0 by `f64::max`.
                let eval = evals[src_col];
                let sigma = if eval.is_nan() {
                    f64::NAN
                } else {
                    eval.max(0.0).sqrt()
                };
                s.push(sigma);
                for r in 0..n {
                    v_cols[r * k + col] = v_asc[r * n + src_col];
                }
            }
            let v_mat = Tensor::from_slice_on(&v_cols, &[n, k], &self.device())?;
            let av = a_f64.matmul(&v_mat)?.to_vec::<f64>()?;
            let mut u_data = vec![0.0f64; m * k];
            for col in 0..k {
                let inv_s = if s[col] > 1e-12 { 1.0 / s[col] } else { 0.0 };
                for r in 0..m {
                    u_data[r * k + col] = av[r * k + col] * inv_s;
                }
            }
            let u_mat = Tensor::from_slice_on(&u_data, &[m, k], &self.device())?;
            (u_mat, s, v_mat.t()?)
        } else {
            // Transposed case: m < n
            let (u_t, s_t, vt_t) = self.t()?.svd()?;
            return Ok((vt_t.t()?, s_t, u_t.t()?));
        };

        let u_out = u_f64.to_dtype(self.dtype())?.detach();
        let vt_out = vt_f64.to_dtype(self.dtype())?.detach();
        let dev = self.device();
        let s_cpu = CpuStorage::F64(s_vec.clone()).to_dtype(&Layout::contiguous(vec![k]), self.dtype())?;
        let s_storage = Storage::new_cpu(s_cpu).to_device(&Layout::contiguous(vec![k]), &dev)?;

        // Backward (Townsend 2016): the gradient is linear in (gU, gS, gV), so each output is its own graph node.
        //   gA = U [ (F o (U^T gU - gU^T U)) S + S (F o (V^T gV - gV^T V)) + diag(gS) ] V^T
        //        + (I - U U^T) gU S^-1 V^T + U S^-1 gV^T (I - V V^T),   F_ij = 1 / (s_j^2 - s_i^2)
        let mut f_mat = vec![0.0f64; k * k];
        for i in 0..k {
            for j in 0..k {
                let d = s_vec[j] * s_vec[j] - s_vec[i] * s_vec[i];
                if i != j {
                    // repeated singular values give inf here, so the U / Vt gradient becomes NaN (as in PyTorch)
                    // instead of a silently wrong finite value
                    f_mat[i * k + j] = 1.0 / d;
                }
            }
        }
        let f_t = Tensor::from_slice_on(&f_mat, &[k, k], &dev)?.to_dtype(self.dtype())?;
        let s_t = Tensor::from_slice_on(&s_vec, &[k], &dev)?.to_dtype(self.dtype())?;
        let sinv_t = Tensor::from_slice_on(&s_vec.iter().map(|&v| 1.0 / v).collect::<Vec<_>>(), &[k], &dev)?
            .to_dtype(self.dtype())?;

        let u_bwd = u_out.clone();
        let vt_bwd = vt_out.clone();
        let s_tensor = Tensor::from_op(
            s_storage,
            Layout::contiguous(vec![k]),
            "svd_s",
            vec![self.clone()],
            move |g| {
                // grad_A = U @ diag(g) @ Vt
                let scaled_u = u_bwd.mul(&g.unsqueeze(0)?)?;
                Ok(vec![Some(scaled_u.matmul(&vt_bwd)?)])
            },
        );

        let (u_c, vt_c, f_c, s_c, sinv_c) = (u_out.clone(), vt_out.clone(), f_t.clone(), s_t.clone(), sinv_t.clone());
        let u_tensor = Tensor::from_op(
            u_out.storage().clone(),
            u_out.layout().clone(),
            "svd_u",
            vec![self.clone()],
            move |gu| {
                let utgu = u_c.t()?.matmul(gu)?;
                let j = f_c.mul(&utgu)?;
                let sym = j.add(&j.t()?)?.mul(&s_c.unsqueeze(0)?)?; // (J + J^T) S
                let t1 = u_c.matmul(&sym)?.matmul(&vt_c)?;
                let proj = gu.sub(&u_c.matmul(&utgu)?)?; // (I - U U^T) gU
                let t3 = proj.mul(&sinv_c.unsqueeze(0)?)?.matmul(&vt_c)?;
                Ok(vec![Some(t1.add(&t3)?)])
            },
        );

        let (u_c, vt_c, f_c, s_c, sinv_c) = (u_out.clone(), vt_out.clone(), f_t, s_t, sinv_t);
        let vt_tensor = Tensor::from_op(
            vt_out.storage().clone(),
            vt_out.layout().clone(),
            "svd_vt",
            vec![self.clone()],
            move |gvt| {
                // gvt: [k, n]; with V = Vt^T, gV = gvt^T and V^T gV = Vt gvt^T
                let vtgv = vt_c.matmul(&gvt.t()?)?;
                let kmat = f_c.mul(&vtgv)?;
                let sym = kmat.add(&kmat.t()?)?.mul(&s_c.unsqueeze(1)?)?; // S (K + K^T)
                let t2 = u_c.matmul(&sym)?.matmul(&vt_c)?;
                let proj = gvt.sub(&gvt.matmul(&vt_c.t()?)?.matmul(&vt_c)?)?; // gV^T (I - V V^T)
                let t4 = u_c.matmul(&proj.mul(&sinv_c.unsqueeze(1)?)?)?;
                Ok(vec![Some(t2.add(&t4)?)])
            },
        );
        Ok((u_tensor, s_tensor, vt_tensor))
    }

    /// Moore-Penrose pseudoinverse `A^+` (`torch.linalg.pinv`) via SVD with relative cutoff `rcond`,
    /// equipped with the exact Golub-Pereyra analytical backward pass.
    /// NaN entries propagate into the outputs (PyTorch raises on non-finite input instead).
    pub fn pinv(&self, rcond: f64) -> Result<Tensor> {
        if self.ndim() > 2 {
            return Ok(self.batched_mats(|m| Ok(vec![m.pinv(rcond)?]))?.remove(0));
        }
        if self.ndim() != 2 {
            return Err(Error::Msg("pinv requires a 2D matrix".into()));
        }
        let m = self.shape()[0];
        let n = self.shape()[1];
        let (u, s, vt) = self.svd()?;
        let s_vals = s.to_vec::<f64>()?;
        let max_s = s_vals.iter().copied().fold(0.0f64, f64::max);
        let cutoff = rcond * max_s;
        let s_inv: Vec<f64> = s_vals
            .iter()
            .map(|&sv| if sv > cutoff { 1.0 / sv } else { 0.0 })
            .collect();
        let k = s_inv.len();
        let s_inv_t =
            Tensor::from_slice_on(&s_inv, &[k], &self.device())?.to_dtype(self.dtype())?;
        // A^+ = V @ diag(s_inv) @ U^T, where V = Vt^T [n, k] and U^T = [k, m]
        let v = vt.t()?;
        let v_scaled = v.mul(&s_inv_t.unsqueeze(0)?)?;
        let pinv_primal = v_scaled.matmul(&u.t()?)?;
        let storage = pinv_primal.storage().clone();
        let layout = pinv_primal.layout().clone();

        let a_saved = self.detach();
        let pinv_saved = pinv_primal.detach();
        Ok(Tensor::from_op(
            storage,
            layout,
            "pinv",
            vec![self.clone()],
            move |g| {
                // Golub & Pereyra (1973):
                // dA = -(A^+)^T G (A^+)^T
                //      + (I_m - A A^+) G^T A^+ (A^+)^T
                //      + (A^+)^T A^+ G^T (I_n - A^+ A)
                let pinv_t = pinv_saved.t()?;
                let gt = g.t()?;
                let term1 = pinv_t.matmul(g)?.matmul(&pinv_t)?.neg()?;

                let eye_m = Tensor::eye(m, a_saved.dtype(), &a_saved.device())?;
                let proj_m = eye_m.sub(&a_saved.matmul(&pinv_saved)?)?;
                let term2 = proj_m.matmul(&gt)?.matmul(&pinv_saved)?.matmul(&pinv_t)?;

                let eye_n = Tensor::eye(n, a_saved.dtype(), &a_saved.device())?;
                let proj_n = eye_n.sub(&pinv_saved.matmul(&a_saved)?)?;
                let term3 = pinv_t.matmul(&pinv_saved)?.matmul(&gt)?.matmul(&proj_n)?;

                Ok(vec![Some(term1.add(&term2)?.add(&term3)?)])
            },
        ))
    }

    /// Differentiable minimum-norm least squares solver `min_X ||A X - B||_F` (`torch.linalg.lstsq`).
    /// Supports both 1D (`[m]`) and 2D (`[m, k]`) right-hand sides `b`.
    pub fn lstsq(&self, b: &Tensor, rcond: f64) -> Result<Tensor> {
        let a_pinv = self.pinv(rcond)?;
        if b.ndim() == 1 {
            a_pinv.matmul(&b.unsqueeze(1)?)?.squeeze_axis(1)
        } else {
            a_pinv.matmul(b)
        }
    }

    /// Differentiable matrix exponential `exp(A)` (`torch.linalg.matrix_exp`) via
    /// scaling-and-squaring with a 10th-order Taylor approximation.
    pub fn matrix_exp(&self) -> Result<Tensor> {
        if self.ndim() > 2 {
            return Ok(self.batched_mats(|m| Ok(vec![m.matrix_exp()?]))?.remove(0));
        }
        if self.ndim() != 2 || self.shape()[0] != self.shape()[1] {
            return Err(Error::Msg("matrix_exp requires a 2D square matrix".into()));
        }
        let n = self.shape()[0];
        let vals = self.to_vec::<f64>()?;
        let inf_norm = (0..n)
            .map(|i| (0..n).map(|j| vals[i * n + j].abs()).sum::<f64>())
            .fold(0.0f64, f64::max);
        let s = if inf_norm > 0.5 {
            (inf_norm * 2.0).log2().ceil().max(0.0) as usize
        } else {
            0
        };
        let scale = (1u64 << s.min(30)) as f64;
        let a_scaled = self.div_scalar(scale)?;

        let mut result = Tensor::eye(n, self.dtype(), &self.device())?;
        let mut term = Tensor::eye(n, self.dtype(), &self.device())?;
        for k in 1..=10 {
            term = term.matmul(&a_scaled)?.div_scalar(k as f64)?;
            result = result.add(&term)?;
        }
        for _ in 0..s.min(30) {
            result = result.matmul(&result)?;
        }
        Ok(result)
    }

    /// Differentiable integer matrix power `A^p` (`torch.linalg.matrix_power`).
    /// Supports `p == 0` (`I`), `p > 0` (binary exponentiation), and `p < 0` (`(A^-1)^|p|`).
    pub fn matrix_power(&self, p: isize) -> Result<Tensor> {
        if self.ndim() > 2 {
            return Ok(self.batched_mats(|m| Ok(vec![m.matrix_power(p)?]))?.remove(0));
        }
        if self.ndim() != 2 || self.shape()[0] != self.shape()[1] {
            return Err(Error::Msg(
                "matrix_power requires a 2D square matrix".into(),
            ));
        }
        let n = self.shape()[0];
        if p == 0 {
            return Tensor::eye(n, self.dtype(), &self.device());
        }
        let mut base = if p > 0 { self.clone() } else { self.inv()? };
        let mut exp = p.unsigned_abs();
        let mut acc: Option<Tensor> = None;
        while exp > 0 {
            if exp & 1 == 1 {
                acc = Some(match acc {
                    None => base.clone(),
                    Some(prev) => prev.matmul(&base)?,
                });
            }
            exp >>= 1;
            if exp > 0 {
                base = base.matmul(&base)?;
            }
        }
        acc.ok_or_else(|| Error::Msg("matrix_power failed".into()))
    }

    /// 2-norm condition number `sigma_max / sigma_min` (`torch.linalg.cond`).
    pub fn cond(&self) -> Result<Tensor> {
        let (_u, s, _vt) = self.svd()?;
        let k = s.shape()[0];
        if k == 0 {
            return Err(Error::Msg(
                "cond: empty matrix has no condition number".into(),
            ));
        }
        let s_max = s.narrow(0, 0, 1)?;
        let s_min = s.narrow(0, k - 1, 1)?;
        s_max.div(&s_min)
    }

    /// Numerical rank of a 2D matrix counting singular values strictly above `tol` (`torch.linalg.matrix_rank`).
    pub fn matrix_rank(&self, tol: f64) -> Result<usize> {
        let (_u, s, _vt) = self.svd()?;
        let s_vals = s.to_vec::<f64>()?;
        let max_s = s_vals.iter().copied().fold(0.0f64, f64::max);
        let threshold = if tol > 0.0 { tol } else { max_s * 1e-6 };
        Ok(s_vals.iter().filter(|&&v| v > threshold).count())
    }

    /// Differentiable Kronecker product of two tensors (`torch.kron`).
    pub fn kron(&self, other: &Tensor) -> Result<Tensor> {
        let max_ndim = self.ndim().max(other.ndim());
        let a = if self.ndim() < max_ndim {
            let mut new_shape = vec![1usize; max_ndim - self.ndim()];
            new_shape.extend_from_slice(self.shape());
            self.reshape(&new_shape)?
        } else {
            self.clone()
        };
        let b = if other.ndim() < max_ndim {
            let mut new_shape = vec![1usize; max_ndim - other.ndim()];
            new_shape.extend_from_slice(other.shape());
            other.reshape(&new_shape)?
        } else {
            other.clone()
        };

        let mut a_expand_shape = Vec::with_capacity(max_ndim * 2);
        let mut b_expand_shape = Vec::with_capacity(max_ndim * 2);
        let mut out_shape = Vec::with_capacity(max_ndim);

        for (&da, &db) in a.shape().iter().zip(b.shape()) {
            a_expand_shape.push(da);
            a_expand_shape.push(1);
            b_expand_shape.push(1);
            b_expand_shape.push(db);
            out_shape.push(da * db);
        }

        let a_expanded = a.reshape(&a_expand_shape)?;
        let b_expanded = b.reshape(&b_expand_shape)?;
        let prod = a_expanded.mul(&b_expanded)?;
        prod.reshape(&out_shape)
    }

    /// Differentiable Cholesky linear solver (`torch.cholesky_solve`):
    /// solves $A X = B$ (`self` = $B$) given Cholesky factor $L$ ($A = L L^T$, `upper = false`)
    /// or $U$ ($A = U^T U$, `upper = true`) via two chained differentiable triangular solves.
    pub fn cholesky_solve(&self, factor: &Tensor, upper: bool) -> Result<Tensor> {
        if upper {
            // U^T (U X) = B -> solve U^T Y = B, then U X = Y
            let y = factor.triangular_solve(self, false, true)?;
            factor.triangular_solve(&y, false, false)
        } else {
            // L (L^T X) = B -> solve L Y = B, then L^T X = Y
            let y = factor.triangular_solve(self, true, false)?;
            factor.triangular_solve(&y, true, true)
        }
    }

    /// Computes the $P, L, U$ decomposition with partial pivoting such that $A = P L U$.
    pub fn lu(&self) -> Result<(Tensor, Tensor, Tensor)> {
        if self.ndim() > 2 {
            let mut v = self.batched_mats(|m| {
                let (p, l, u) = m.lu()?;
                Ok(vec![p, l, u])
            })?;
            let u = v.remove(2);
            let l = v.remove(1);
            return Ok((v.remove(0), l, u));
        }
        if self.ndim() != 2 || self.shape()[0] != self.shape()[1] {
            return Err(Error::Msg(format!(
                "lu expects a square 2D matrix, got {:?}",
                self.shape()
            )));
        }
        let n = self.shape()[0];
        let mut u = self.to_vec::<f64>()?;
        let mut l = vec![0.0f64; n * n];
        let mut perm: Vec<usize> = (0..n).collect();

        for k in 0..n {
            let mut max_row = k;
            let mut max_val = u[k * n + k].abs();
            for i in (k + 1)..n {
                let v = u[i * n + k].abs();
                if v > max_val {
                    max_val = v;
                    max_row = i;
                }
            }
            if max_row != k {
                perm.swap(k, max_row);
                for j in 0..n {
                    u.swap(k * n + j, max_row * n + j);
                }
                for j in 0..k {
                    l.swap(k * n + j, max_row * n + j);
                }
            }
            l[k * n + k] = 1.0;
            let pivot = u[k * n + k];
            if pivot.abs() > 1e-15 {
                for i in (k + 1)..n {
                    let factor = u[i * n + k] / pivot;
                    l[i * n + k] = factor;
                    u[i * n + k] = 0.0;
                    for j in (k + 1)..n {
                        u[i * n + j] -= factor * u[k * n + j];
                    }
                }
            }
        }
        // A = P @ L @ U -> P[perm[i], i] = 1
        let mut p_mat = vec![0.0f64; n * n];
        for (col, &row) in perm.iter().enumerate() {
            p_mat[row * n + col] = 1.0;
        }
        let dt = self.dtype();
        let dev = self.device();
        let p_t = Tensor::from_slice_on(&p_mat, &[n, n], &dev)?.to_dtype(dt)?;
        let l_t = Tensor::from_slice_on(&l, &[n, n], &dev)?.to_dtype(dt)?;
        let u_t = Tensor::from_slice_on(&u, &[n, n], &dev)?.to_dtype(dt)?;
        Ok((p_t, l_t, u_t))
    }

    /// Computes the Polar decomposition $A = U P$ where $U = W V^T$ is orthogonal/isometric
    /// and $P = V \Sigma V^T$ is symmetric positive semi-definite.
    pub fn polar(&self) -> Result<(Tensor, Tensor)> {
        let (w, s, vt) = self.svd()?;
        let v = vt.t()?;
        let u_pol = w.matmul(&vt)?;
        let s_diag = s.diag_embed()?;
        let p_pol = v.matmul(&s_diag)?.matmul(&vt)?;
        Ok((u_pol, p_pol))
    }

    /// Generalized differentiable $L_p$ pairwise distance matrix between `[..., N, D]` and `[..., M, D]`.
    pub fn cdist_p(&self, other: &Tensor, p: f64) -> Result<Tensor> {
        if self.ndim() < 2 || other.ndim() < 2 {
            return Err(Error::Msg("cdist_p requires rank >= 2".into()));
        }
        let d1 = self.shape()[self.ndim() - 1];
        let d2 = other.shape()[other.ndim() - 1];
        if d1 != d2 {
            return Err(Error::Msg(format!(
                "cdist_p: feature dimension mismatch ({} vs {})",
                d1, d2
            )));
        }
        let diff = self.unsqueeze(-2)?.sub(&other.unsqueeze(-3)?)?;
        if (p - 2.0).abs() < 1e-12 {
            diff.square()?.sum_axis(-1, false)?.sqrt()
        } else {
            diff.norm_p(p, &[-1], false)
        }
    }

    /// Differentiable sample covariance matrix of a 1D or 2D `[variables, observations]` tensor (`torch.cov`).
    pub fn cov(&self, correction: usize) -> Result<Tensor> {
        let x = if self.ndim() == 1 {
            self.unsqueeze(0)?
        } else if self.ndim() == 2 {
            self.clone()
        } else {
            return Err(Error::Msg(format!(
                "cov expects a 1D or 2D tensor, got {:?}",
                self.shape()
            )));
        };
        let n_obs = x.shape()[1];
        if n_obs <= correction {
            return Err(Error::Msg(format!(
                "cov: degrees of freedom <= 0 (n_obs={n_obs}, correction={correction})"
            )));
        }
        let mean = x.mean_axis(1, true)?.broadcast_to(x.shape())?;
        let centered = x.sub(&mean)?;
        let denom = (n_obs - correction) as f64;
        centered.matmul(&centered.t()?)?.div_scalar(denom)
    }

    /// Differentiable Pearson product-moment correlation coefficient matrix (`torch.corrcoef`).
    pub fn corrcoef(&self) -> Result<Tensor> {
        let c = self.cov(1)?;
        let diag = c.diagonal(0, 0, 1)?.clamp(1e-12, f64::INFINITY)?.sqrt()?;
        let denom = diag.unsqueeze(1)?.matmul(&diag.unsqueeze(0)?)?;
        c.div(&denom)?.clamp(-1.0, 1.0)
    }
}

fn jacobi_eigh(a_in: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = a_in.to_vec();
    // Symmetrize
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (a[i * n + j] + a[j * n + i]);
            a[i * n + j] = avg;
            a[j * n + i] = avg;
        }
    }
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    for _sweep in 0..80 {
        let mut max_off = 0.0f64;
        for p in 0..n {
            for q in (p + 1)..n {
                max_off = max_off.max(a[p * n + q].abs());
            }
        }
        if max_off < 1e-14 {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-15 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                a[p * n + p] = app - t * apq;
                a[q * n + q] = aqq + t * apq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;

                for r in 0..n {
                    if r != p && r != q {
                        let arp = a[r * n + p];
                        let arq = a[r * n + q];
                        a[r * n + p] = c * arp - s * arq;
                        a[p * n + r] = a[r * n + p];
                        a[r * n + q] = s * arp + c * arq;
                        a[q * n + r] = a[r * n + q];
                    }
                    let vrp = v[r * n + p];
                    let vrq = v[r * n + q];
                    v[r * n + p] = c * vrp - s * vrq;
                    v[r * n + q] = s * vrp + c * vrq;
                }
            }
        }
    }

    let mut order: Vec<usize> = (0..n).collect();
    // `total_cmp` keeps the ordering deterministic, with NaN sorting last.
    order.sort_by(|&i, &j| a[i * n + i].total_cmp(&a[j * n + j]));

    let mut vals = Vec::with_capacity(n);
    let mut vecs = vec![0.0f64; n * n];
    for (new_col, &old_col) in order.iter().enumerate() {
        vals.push(a[old_col * n + old_col]);
        for r in 0..n {
            vecs[r * n + new_col] = v[r * n + old_col];
        }
    }
    (vals, vecs)
}

fn check_square_batch(op: &'static str, shape: &[usize]) -> Result<(usize, usize)> {
    let r = shape.len();
    if r < 2 {
        return Err(Error::Msg(format!(
            "{op}: expected a batch of square matrices ([..., n, n]), got rank-{r} shape {shape:?}"
        )));
    }
    if shape[r - 2] != shape[r - 1] {
        let mut expected = shape.to_vec();
        expected[r - 1] = expected[r - 2];
        return Err(Error::ShapeMismatch {
            op,
            expected,
            got: shape.to_vec(),
        });
    }
    let n = shape[r - 1];
    let batch: usize = shape[..r - 2].iter().product();
    Ok((batch, n))
}

fn lu_solve_single(a: &[f64], b: &[f64], n: usize, k_cols: usize) -> Result<Vec<f64>> {
    let mut lu = a.to_vec();
    let mut x = b.to_vec();

    for i in 0..n {
        let mut pivot_row = i;
        let mut max_val = lu[i * n + i].abs();
        for r in (i + 1)..n {
            let v = lu[r * n + i].abs();
            if v > max_val {
                max_val = v;
                pivot_row = r;
            }
        }
        if max_val < 1e-14 {
            return Err(Error::Msg("solve: singular matrix encountered".into()));
        }
        if pivot_row != i {
            for col in 0..n {
                lu.swap(i * n + col, pivot_row * n + col);
            }
            for col in 0..k_cols {
                x.swap(i * k_cols + col, pivot_row * k_cols + col);
            }
        }
        let piv = lu[i * n + i];
        for r in (i + 1)..n {
            let factor = lu[r * n + i] / piv;
            lu[r * n + i] = factor;
            for c in (i + 1)..n {
                lu[r * n + c] -= factor * lu[i * n + c];
            }
            for col in 0..k_cols {
                x[r * k_cols + col] -= factor * x[i * k_cols + col];
            }
        }
    }

    for i in (0..n).rev() {
        let piv = lu[i * n + i];
        for col in 0..k_cols {
            let mut sum = x[i * k_cols + col];
            for j in (i + 1)..n {
                sum -= lu[i * n + j] * x[j * k_cols + col];
            }
            x[i * k_cols + col] = sum / piv;
        }
    }
    Ok(x)
}

fn lu_slogdet_single(a: &[f64], n: usize) -> (f64, f64) {
    let mut lu = a.to_vec();
    let mut sign = 1.0f64;
    let mut logabsdet = 0.0f64;

    for i in 0..n {
        let mut pivot_row = i;
        let mut max_val = lu[i * n + i].abs();
        for r in (i + 1)..n {
            let v = lu[r * n + i].abs();
            if v > max_val {
                max_val = v;
                pivot_row = r;
            }
        }
        if max_val < 1e-15 {
            return (0.0, f64::NEG_INFINITY);
        }
        if pivot_row != i {
            for col in 0..n {
                lu.swap(i * n + col, pivot_row * n + col);
            }
            sign = -sign;
        }
        let piv = lu[i * n + i];
        if piv < 0.0 {
            sign = -sign;
        }
        logabsdet += piv.abs().ln();
        for r in (i + 1)..n {
            let factor = lu[r * n + i] / piv;
            for c in (i + 1)..n {
                lu[r * n + c] -= factor * lu[i * n + c];
            }
        }
    }
    (sign, logabsdet)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn svd_vector_gradient_is_nan_for_repeated_singular_values() {
        // identity: all singular values equal, so the U / Vt gradient is undefined; it must not look valid
        let x = Tensor::eye(3, crate::dtype::DType::F64, &crate::device::Device::Cpu).unwrap().requires_grad_(true);
        let (u, _s, _vt) = x.svd().unwrap();
        let w = Tensor::from_slice(&[0.3f64, -1.0, 0.7, 0.2, 0.9, -0.4, 1.1, 0.5, -0.8], &[3, 3]).unwrap();
        u.mul(&w).unwrap().sum().unwrap().backward().unwrap();
        assert!(x.grad().unwrap().to_vec::<f64>().unwrap().iter().any(|v| !v.is_finite()));
    }

    #[test]
    fn svd_singular_vectors_are_differentiable() {
        // sign-invariant function of U, S, V^T: U sqrt(S) V^T; check the gradient by central differences (tall and wide)
        for (rows, cols) in [(5usize, 3usize), (3, 5)] {
            let a: Vec<f64> = (0..rows * cols).map(|k| (0.83 * k as f64 + 0.3).sin() + if k / cols == k % cols { 2.0 } else { 0.0 }).collect();
            let w: Vec<f64> = (0..rows * cols).map(|k| (1.3 * k as f64).cos()).collect();
            let wt = Tensor::from_slice(&w, &[rows, cols]).unwrap();
            let f = |t: &Tensor| {
                let (u, s, vt) = t.svd().unwrap();
                let sq = s.sqrt().unwrap().reshape(&[1, rows.min(cols)]).unwrap();
                u.mul(&sq).unwrap().matmul(&vt).unwrap().mul(&wt).unwrap().sum().unwrap()
            };
            let x = Tensor::from_slice(&a, &[rows, cols]).unwrap().requires_grad_(true);
            f(&x).backward().unwrap();
            let got = x.grad().unwrap().to_vec::<f64>().unwrap();
            for i in 0..a.len() {
                let (mut hi, mut lo) = (a.clone(), a.clone());
                hi[i] += 1e-6;
                lo[i] -= 1e-6;
                let at = |v: &[f64]| f(&Tensor::from_slice(v, &[rows, cols]).unwrap()).to_vec::<f64>().unwrap()[0];
                let num = (at(&hi) - at(&lo)) / 2e-6;
                assert!((got[i] - num).abs() < 1e-5, "{rows}x{cols} element {i}: {} vs {num}", got[i]);
            }
        }
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

    #[test]
    fn norm_p_matches_hand_values() {
        let x = Tensor::from_slice(&[3.0, 4.0], &[2]).unwrap();
        assert_eq!(
            x.norm_p(0.0, &[0], false).unwrap().item::<f64>().unwrap(),
            2.0
        );
        assert_eq!(
            x.norm_p(1.0, &[0], false).unwrap().item::<f64>().unwrap(),
            7.0
        );
        assert_eq!(
            x.norm_p(2.0, &[0], false).unwrap().item::<f64>().unwrap(),
            5.0
        );
        let linf = x
            .norm_p(f64::INFINITY, &[0], false)
            .unwrap()
            .item::<f64>()
            .unwrap();
        assert_eq!(linf, 4.0);
        // the norm of a zero vector is exactly 0 (like torch)
        let z = Tensor::from_slice(&[0.0, 0.0], &[2]).unwrap();
        assert_eq!(
            z.norm_p(2.0, &[0], false).unwrap().item::<f64>().unwrap(),
            0.0
        );
        assert!(x.norm_p(f64::NAN, &[0], false).is_err());
    }

    #[test]
    fn cdist_matches_hand_values() {
        let a = Tensor::from_slice(&[0.0, 0.0], &[1, 2]).unwrap();
        let b = Tensor::from_slice(&[3.0, 4.0], &[1, 2]).unwrap();
        let d = a.cdist(&b).unwrap().to_vec::<f64>().unwrap();
        assert_close(&d, &[5.0], 1e-9, "3-4-5");
        // identical points have exact distance 0.0 (matching torch.cdist)
        let d0 = a.cdist(&a).unwrap().to_vec::<f64>().unwrap();
        assert_eq!(d0[0], 0.0, "identical points: {}", d0[0]);
    }

    #[test]
    fn cholesky_roundtrip_and_rejects_non_spd() {
        // A = [[4,2],[2,3]] -> L = [[2,0],[1,sqrt(2)]]
        let a = Tensor::from_slice(&[4.0, 2.0, 2.0, 3.0], &[2, 2]).unwrap();
        let l = a.cholesky().unwrap();
        let got = l.to_vec::<f64>().unwrap();
        assert_close(&got, &[2.0, 0.0, 1.0, 2.0f64.sqrt()], 1e-12, "cholesky");
        let back = l.matmul(&l.t().unwrap()).unwrap().to_vec::<f64>().unwrap();
        assert_close(&back, &[4.0, 2.0, 2.0, 3.0], 1e-12, "L L^T");
        // not positive definite -> error, not NaN
        let bad = Tensor::from_slice(&[1.0, 2.0, 2.0, 1.0], &[2, 2]).unwrap();
        assert!(bad.cholesky().is_err());
        let nonsq = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        assert!(nonsq.cholesky().is_err());
    }

    #[test]
    fn triangular_solve_matches_hand_values_and_rejects_singular() {
        // L = [[2,0],[1,3]], B = [4,11] -> X = [2,3]
        let l = Tensor::from_slice(&[2.0, 0.0, 1.0, 3.0], &[2, 2]).unwrap();
        let b = Tensor::from_slice(&[4.0, 11.0], &[2, 1]).unwrap();
        let x = l
            .triangular_solve(&b, true, false)
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(&x, &[2.0, 3.0], 1e-12, "forward substitution");
        // transposed path solves U^T X = B with U upper
        let u = Tensor::from_slice(&[2.0, 1.0, 0.0, 3.0], &[2, 2]).unwrap();
        let x2 = u
            .triangular_solve(&b, false, true)
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(&x2, &[2.0, 3.0], 1e-12, "transposed");
        // zero diagonal -> error
        let s = Tensor::from_slice(&[0.0, 0.0, 1.0, 3.0], &[2, 2]).unwrap();
        assert!(s.triangular_solve(&b, true, false).is_err());
    }

    #[test]
    fn solve_and_inv_match_hand_values_and_reject_singular() {
        // A = [[3,1],[1,2]], B = [9,8] -> X = [2,3]
        let a = Tensor::from_slice(&[3.0, 1.0, 1.0, 2.0], &[2, 2]).unwrap();
        let b = Tensor::from_slice(&[9.0, 8.0], &[2, 1]).unwrap();
        let x = a.solve(&b).unwrap().to_vec::<f64>().unwrap();
        assert_close(&x, &[2.0, 3.0], 1e-12, "solve");
        // A^-1 = (1/5)[[2,-1],[-1,3]]
        let inv = a.inv().unwrap().to_vec::<f64>().unwrap();
        assert_close(&inv, &[0.4, -0.2, -0.2, 0.6], 1e-12, "inv");
        let eye = a
            .matmul(&a.inv().unwrap())
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(&eye, &[1.0, 0.0, 0.0, 1.0], 1e-12, "A A^-1");
        // singular -> error, not NaN
        let s = Tensor::from_slice(&[1.0, 2.0, 2.0, 4.0], &[2, 2]).unwrap();
        assert!(s.solve(&b).is_err());
        assert!(s.inv().is_err());
    }

    #[test]
    fn slogdet_and_det_match_hand_values_and_singular() {
        // det [[3,1],[1,2]] = 5
        let a = Tensor::from_slice(&[3.0, 1.0, 1.0, 2.0], &[2, 2]).unwrap();
        let (sign, ld) = a.slogdet().unwrap();
        assert_eq!(sign.item::<f64>().unwrap(), 1.0);
        assert!((ld.item::<f64>().unwrap() - 5.0f64.ln()).abs() <= 1e-12);
        assert!((a.det().unwrap().item::<f64>().unwrap() - 5.0).abs() <= 1e-12);
        // negative determinant: [[0,1],[1,0]] -> -1
        let f = Tensor::from_slice(&[0.0, 1.0, 1.0, 0.0], &[2, 2]).unwrap();
        let (s2, ld2) = f.slogdet().unwrap();
        assert_eq!(s2.item::<f64>().unwrap(), -1.0);
        assert!(ld2.item::<f64>().unwrap().abs() <= 1e-12);
        // singular -> (0, -inf) and det 0, like torch
        let s = Tensor::from_slice(&[1.0, 2.0, 2.0, 4.0], &[2, 2]).unwrap();
        let (s0, ld0) = s.slogdet().unwrap();
        assert_eq!(s0.item::<f64>().unwrap(), 0.0);
        assert_eq!(ld0.item::<f64>().unwrap(), f64::NEG_INFINITY);
        assert_eq!(s.det().unwrap().item::<f64>().unwrap(), 0.0);
        // ...but differentiating through a singular det fails closed
        let sg = Tensor::from_slice(&[1.0, 2.0, 2.0, 4.0], &[2, 2])
            .unwrap()
            .requires_grad_(true);
        assert!(sg.det().unwrap().backward().is_err());
    }

    #[test]
    fn qr_reconstructs_and_orthogonalizes_full_rank() {
        // A = [[12,-51,4],[6,167,-68],[-4,24,-41]] (classic MGS example)
        let a = Tensor::from_slice(
            &[12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0],
            &[3, 3],
        )
        .unwrap();
        let (q, r) = a.qr().unwrap();
        let qtq = q.t().unwrap().matmul(&q).unwrap().to_vec::<f64>().unwrap();
        assert_close(
            &qtq,
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            1e-9,
            "Q orthonormal",
        );
        let back = q.matmul(&r).unwrap().to_vec::<f64>().unwrap();
        assert_close(
            &back,
            &[12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0],
            1e-9,
            "Q R = A",
        );
        let rv = r.to_vec::<f64>().unwrap();
        assert!(
            rv[3] == 0.0 && rv[6] == 0.0 && rv[7] == 0.0,
            "R upper: {rv:?}"
        );
        // wide matrices are refused (reduced mode needs m >= n)
        let wide = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        assert!(wide.qr().is_err());
        // rank-deficient input still yields orthonormal Q with reconstruction
        let dep = Tensor::from_slice(&[1.0, 2.0, 2.0, 4.0], &[2, 2]).unwrap();
        let (qd, rd) = dep.qr().unwrap();
        let qtq_d = qd
            .t()
            .unwrap()
            .matmul(&qd)
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(
            &qtq_d,
            &[1.0, 0.0, 0.0, 1.0],
            1e-12,
            "deficient Q orthonormal",
        );
        let back_d = qd.matmul(&rd).unwrap().to_vec::<f64>().unwrap();
        assert_close(&back_d, &[1.0, 2.0, 2.0, 4.0], 1e-9, "deficient Q R = A");
        // ...with a ~0 R diagonal entry marking the dependent column
        let rdv = rd.to_vec::<f64>().unwrap();
        assert!(rdv[3].abs() <= 1e-9, "R[1,1] ~ 0: {rdv:?}");
        // NaN propagates instead of erroring
        let nan = Tensor::from_slice(&[f64::NAN, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let (qn, rn) = nan.qr().unwrap();
        let has_nan = qn
            .to_vec::<f64>()
            .unwrap()
            .into_iter()
            .chain(rn.to_vec::<f64>().unwrap())
            .any(|v| v.is_nan());
        assert!(has_nan, "qr propagates NaN");
    }

    #[test]
    fn eigh_returns_sorted_spectrum_with_orthonormal_vectors() {
        // [[2,1],[1,2]] has eigenvalues 1, 3
        let a = Tensor::from_slice(&[2.0, 1.0, 1.0, 2.0], &[2, 2]).unwrap();
        let (vals, vecs) = a.eigh().unwrap();
        assert_close(
            &vals.to_vec::<f64>().unwrap(),
            &[1.0, 3.0],
            1e-12,
            "ascending",
        );
        let vtv = vecs
            .t()
            .unwrap()
            .matmul(&vecs)
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(&vtv, &[1.0, 0.0, 0.0, 1.0], 1e-12, "orthonormal");
        // A = V diag V^T (sign-proof reconstruction)
        let d = Tensor::from_slice(&[1.0, 0.0, 0.0, 3.0], &[2, 2]).unwrap();
        let back = vecs
            .matmul(&d)
            .unwrap()
            .matmul(&vecs.t().unwrap())
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(&back, &[2.0, 1.0, 1.0, 2.0], 1e-9, "spectral");
        // NaN eigenvalues sort last (deterministic)
        let n = Tensor::from_slice(&[f64::NAN, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let (nv, _) = n.eigh().unwrap();
        let got = nv.to_vec::<f64>().unwrap();
        assert_eq!(got[0], 1.0);
        assert!(got[1].is_nan(), "NaN sorts last: {got:?}");
    }

    #[test]
    fn svd_reconstructs_with_descending_singular_values() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let (u, s, vt) = a.svd().unwrap();
        let sv = s.to_vec::<f64>().unwrap();
        assert!(sv[0] >= sv[1] && sv[1] > 0.0, "descending positive: {sv:?}");
        let d = Tensor::from_slice(&[sv[0], 0.0, 0.0, sv[1]], &[2, 2]).unwrap();
        let back = u
            .matmul(&d)
            .unwrap()
            .matmul(&vt)
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(&back, &[1.0, 2.0, 3.0, 4.0], 1e-8, "U S Vt");
        // tall (m > n) and wide (m < n) shapes
        let tall = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let (ut, st, vtt) = tall.svd().unwrap();
        assert_eq!(ut.shape(), &[3, 2]);
        assert_eq!(st.shape(), &[2]);
        assert_eq!(vtt.shape(), &[2, 2]);
        let wide = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let (uw, sw, vtw) = wide.svd().unwrap();
        assert_eq!(uw.shape(), &[2, 2]);
        assert_eq!(sw.shape(), &[2]);
        assert_eq!(vtw.shape(), &[2, 3]);
        // NaN propagates into S (never silenced to 0)
        let n = Tensor::from_slice(&[f64::NAN, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let (_, sn, _) = n.svd().unwrap();
        assert!(
            sn.to_vec::<f64>().unwrap().iter().any(|v| v.is_nan()),
            "S keeps NaN"
        );
    }

    #[test]
    fn pinv_satisfies_moore_penrose_and_propagates_nan() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let p = a.pinv(1e-6).unwrap();
        let back = a
            .matmul(&p)
            .unwrap()
            .matmul(&a)
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(&back, &[1.0, 2.0, 3.0, 4.0], 1e-6, "A pinv A");
        // rank-deficient: pinv still satisfies A pinv A = A
        let r = Tensor::from_slice(&[1.0, 2.0, 2.0, 4.0], &[2, 2]).unwrap();
        let pr = r.pinv(1e-6).unwrap();
        let back2 = r
            .matmul(&pr)
            .unwrap()
            .matmul(&r)
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(&back2, &[1.0, 2.0, 2.0, 4.0], 1e-6, "rank-deficient");
        // NaN propagates (never silenced to zeros)
        let n = Tensor::from_slice(&[f64::NAN, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let pn = n.pinv(1e-6).unwrap().to_vec::<f64>().unwrap();
        assert!(pn.iter().any(|v| v.is_nan()), "pinv keeps NaN: {pn:?}");
    }

    #[test]
    fn matrix_exp_power_kron_and_cond_match_hand_values() {
        // exp(0) = I; exp of nilpotent [[0,1],[0,0]] = I + A
        let z = Tensor::from_slice(&[0.0, 0.0, 0.0, 0.0], &[2, 2]).unwrap();
        let e0 = z.matrix_exp().unwrap().to_vec::<f64>().unwrap();
        assert_close(&e0, &[1.0, 0.0, 0.0, 1.0], 1e-12, "exp(0)");
        let n = Tensor::from_slice(&[0.0, 1.0, 0.0, 0.0], &[2, 2]).unwrap();
        let en = n.matrix_exp().unwrap().to_vec::<f64>().unwrap();
        assert_close(&en, &[1.0, 1.0, 0.0, 1.0], 1e-12, "exp(nilpotent)");
        // powers of diag([1,2]): ^3 = diag([1,8]), ^0 = I, ^-1 inverts
        let d = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], &[2, 2]).unwrap();
        let p3 = d.matrix_power(3).unwrap().to_vec::<f64>().unwrap();
        assert_close(&p3, &[1.0, 0.0, 0.0, 8.0], 0.0, "pow3");
        let p0 = d.matrix_power(0).unwrap().to_vec::<f64>().unwrap();
        assert_close(&p0, &[1.0, 0.0, 0.0, 1.0], 0.0, "pow0");
        let pm1 = d.matrix_power(-1).unwrap().to_vec::<f64>().unwrap();
        assert_close(&pm1, &[1.0, 0.0, 0.0, 0.5], 1e-12, "pow-1");
        // kron([[1,2],[3,4]], [[0,5],[6,7]]) hand value
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::from_slice(&[0.0, 5.0, 6.0, 7.0], &[2, 2]).unwrap();
        let k = a.kron(&b).unwrap().to_vec::<f64>().unwrap();
        assert_close(
            &k,
            &[
                0.0, 5.0, 0.0, 10.0, 6.0, 7.0, 12.0, 14.0, 0.0, 15.0, 0.0, 20.0, 18.0, 21.0, 24.0,
                28.0,
            ],
            1e-12,
            "kron",
        );
        // cond(diag([2,1])) = 2
        let c = Tensor::from_slice(&[2.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        assert!((c.cond().unwrap().item::<f64>().unwrap() - 2.0).abs() <= 1e-9);
    }

    #[test]
    fn matrix_rank_counts_above_threshold_and_lstsq_solves() {
        let d = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0], &[2, 2]).unwrap();
        assert_eq!(d.matrix_rank(0.5).unwrap(), 1);
        let z = Tensor::from_slice(&[0.0, 0.0, 0.0, 0.0], &[2, 2]).unwrap();
        assert_eq!(z.matrix_rank(0.0).unwrap(), 0);
        // min-norm least squares: [[1],[1]] x = [2,4] -> x = 3
        let a = Tensor::from_slice(&[1.0, 1.0], &[2, 1]).unwrap();
        let b = Tensor::from_slice(&[2.0, 4.0], &[2]).unwrap();
        let x = a.lstsq(&b, 1e-6).unwrap().to_vec::<f64>().unwrap();
        assert_close(&x, &[3.0], 1e-9, "lstsq");
    }

    /// The Giles (2008) QR backward must match central differences on both
    /// the Q and R branches.
    #[test]
    fn qr_backward_matches_finite_differences() {
        // tall full-rank input keeps every R diagonal safely from zero
        let base = vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0];
        for branch in 0..2 {
            let x = Tensor::from_slice(&base, &[3, 2])
                .unwrap()
                .requires_grad_(true);
            let (q, r) = x.qr().unwrap();
            let loss = if branch == 0 {
                q.sum().unwrap()
            } else {
                r.sum().unwrap()
            };
            loss.backward().unwrap();
            let analytic = x.grad().unwrap().to_vec::<f64>().unwrap();
            let e = 1e-6;
            for i in 0..base.len() {
                let mut up = base.clone();
                up[i] += e;
                let mut dn = base.clone();
                dn[i] -= e;
                let lp = Tensor::from_slice(&up, &[3, 2]).unwrap().qr().unwrap();
                let lm = Tensor::from_slice(&dn, &[3, 2]).unwrap().qr().unwrap();
                let lpv = if branch == 0 {
                    lp.0.sum().unwrap()
                } else {
                    lp.1.sum().unwrap()
                };
                let lmv = if branch == 0 {
                    lm.0.sum().unwrap()
                } else {
                    lm.1.sum().unwrap()
                };
                let numeric = (lpv.item::<f64>().unwrap() - lmv.item::<f64>().unwrap()) / (2.0 * e);
                assert!(
                    (analytic[i] - numeric).abs() <= 1e-4,
                    "branch {branch} element {i}: {} vs {numeric}",
                    analytic[i]
                );
            }
        }
    }

    #[test]
    fn advanced_linalg_values_match_torch() {
        // pinv + lstsq on a 3x2 full-rank factor (torch float64, rcond=1e-7)
        let a = Tensor::from_slice(&[2.0, 1.0, 0.5, 3.0, -1.0, 1.5], &[3, 2]).unwrap();
        assert_close(
            &a.pinv(1e-7).unwrap().to_vec::<f64>().unwrap(),
            &[
                0.3730569948186528,
                0.0020725388601036746,
                -0.25284974093264245,
                0.020725388601036288,
                0.2445595854922279,
                0.16373056994818652,
            ],
            1e-9,
            "pinv",
        );
        let b1 = Tensor::from_slice(&[4.0, 2.5, 1.0], &[3]).unwrap();
        assert_close(
            &a.lstsq(&b1, 1e-7).unwrap().to_vec::<f64>().unwrap(),
            &[1.2445595854922282, 0.8580310880829013],
            1e-9,
            "lstsq1",
        );
        // rank-deficient pinv respects rcond (torch-verified)
        let rd =
            Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]).unwrap();
        assert_close(
            &rd.pinv(1e-7).unwrap().to_vec::<f64>().unwrap(),
            &[
                -0.6388888888888886,
                -0.16666666666666594,
                0.305555555555555,
                -0.05555555555555532,
                1.3877787807814457e-16,
                0.05555555555555542,
                0.5277777777777773,
                0.166666666666666,
                -0.19444444444444384,
            ],
            1e-9,
            "pinv-def",
        );
        // matrix_exp: 10th-order Taylor + scaling/squaring (torch-verified)
        let s = Tensor::from_slice(&[0.2, 0.1, -0.1, 0.3], &[2, 2]).unwrap();
        assert_close(
            &s.matrix_exp().unwrap().to_vec::<f64>().unwrap(),
            &[
                1.2150922807226585,
                0.12824209866963274,
                -0.12824209866963274,
                1.3433343793922912,
            ],
            1e-9,
            "exp-small",
        );
        let big = Tensor::from_slice(
            &[
                0.8066936952585986,
                1.6760526659953197,
                -1.4385151569387185,
                -0.8066870498643491,
                -1.1932707252302546,
                0.3640729701226111,
                -1.7133491865927486,
                2.201208341806854,
                -2.1423747262182946,
            ],
            &[3, 3],
        )
        .unwrap();
        assert_close(
            &big.matrix_exp().unwrap().to_vec::<f64>().unwrap(),
            &[
                2.9116376562532484,
                1.0437118010042326,
                -1.119669455809908,
                -1.1502064858841625,
                0.024785508799804505,
                0.4603592927160276,
                -2.0884330556575303,
                -0.1991664969561021,
                0.888452390492152,
            ],
            1e-9,
            "exp-big",
        );
        // matrix_power incl. p=0 identity (torch-verified)
        assert_close(
            &s.matrix_power(0).unwrap().to_vec::<f64>().unwrap(),
            &[1.0, 0.0, 0.0, 1.0],
            0.0,
            "pow0",
        );
        assert_close(
            &s.matrix_power(3).unwrap().to_vec::<f64>().unwrap(),
            &[0.001, 0.018, -0.018, 0.019],
            1e-12,
            "pow3",
        );
        // cond + rank (explicit tol; torch's *default* tol scales by eps while
        // oxideon's tol<=0 fallback scales by 1e-6: deliberate, documented)
        assert!((a.cond().unwrap().item::<f64>().unwrap() - 1.6457563846065935).abs() <= 1e-9);
        assert!((s.cond().unwrap().item::<f64>().unwrap() - 1.4560832005096074).abs() <= 1e-9);
        assert_eq!(rd.matrix_rank(1e-6).unwrap(), 2);
        assert_eq!(a.matrix_rank(1e-6).unwrap(), 2);
        // kron 1D + 2D (torch-verified)
        let k1 = Tensor::from_slice(&[1.0, 2.0], &[2]).unwrap();
        let k2 = Tensor::from_slice(&[3.0, 4.0, 5.0], &[3]).unwrap();
        assert_close(
            &k1.kron(&k2).unwrap().to_vec::<f64>().unwrap(),
            &[3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
            0.0,
            "kron1",
        );
        let e2 = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        assert_close(
            &e2.kron(&s).unwrap().to_vec::<f64>().unwrap(),
            &[
                0.2, 0.1, 0.0, 0.0, -0.1, 0.3, 0.0, 0.0, 0.0, 0.0, 0.2, 0.1, 0.0, 0.0, -0.1, 0.3,
            ],
            1e-12,
            "kron2",
        );
        // cdist general case (torch-verified)
        let c1 = Tensor::from_slice(
            &[
                0.12270123916331216,
                -0.5663174751548546,
                0.37311463418249957,
                -0.8919953148065625,
                -1.509107744632202,
                0.3703935333738153,
                1.4565026067903366,
                0.9398099079535257,
                0.7748488177844168,
                0.19186942747902466,
                1.2637947253235853,
                -1.29043510317847,
            ],
            &[4, 3],
        )
        .unwrap();
        let c2 = Tensor::from_slice(
            &[
                -0.7911026902762878,
                -0.020879472995974358,
                -0.7184800423600348,
                0.5186367489510352,
                -1.3125219619835629,
                0.19199507185683629,
                0.5427713494727768,
                -2.2187793699300005,
                0.2589845413662202,
                -1.029702181843019,
                -0.5007584123764369,
                0.27336773842007633,
                -0.9180999735255815,
                -0.040421087473651175,
                0.28811682688855406,
            ],
            &[5, 3],
        )
        .unwrap();
        assert_close(
            &c1.cdist(&c2).unwrap().to_vec::<f64>().unwrap(),
            &[
                1.5245062064531991,
                0.8639388635379239,
                1.7088343656848441,
                1.158568547248595,
                1.1692128985585084,
                1.8467940800515312,
                1.4353936046646656,
                1.604556374883796,
                1.0223236092236394,
                1.4712210597161997,
                2.8643820873988526,
                2.5084476691325976,
                3.328318943347283,
                2.916836404312409,
                2.614669816592646,
                1.715737325964609,
                2.9902815552276145,
                3.827813438856606,
                2.6554405095690488,
                2.329127981334475,
            ],
            1e-9,
            "cdist",
        );
        // slogdet/det with negative determinant (torch-verified)
        let n = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let (sign, logdet) = n.slogdet().unwrap();
        assert_eq!(sign.item::<f64>().unwrap(), -1.0);
        assert!((logdet.item::<f64>().unwrap() - std::f64::consts::LN_2).abs() <= 1e-12);
        assert!((n.det().unwrap().item::<f64>().unwrap() - -2.0).abs() <= 1e-12);
    }
}
