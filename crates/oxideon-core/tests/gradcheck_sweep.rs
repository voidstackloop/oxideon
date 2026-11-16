//! Finite-difference gradient checks over differentiable linear-algebra ops (a sweep for backward-pass bugs).

use oxideon_core::autograd_utils::gradcheck;
use oxideon_core::{Result, Tensor};

fn t(v: &[f64], shape: &[usize]) -> Tensor {
    Tensor::from_slice(v, shape).unwrap()
}

fn spd() -> Tensor {
    t(&[4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0], &[3, 3])
}
fn gmat() -> Tensor {
    t(&[2.0, 0.3, -0.5, 0.7, 1.5, 0.4, -0.2, 0.9, 1.8], &[3, 3])
}
fn tall() -> Tensor {
    t(&[1.0, 0.5, 0.2, 1.3, -0.4, 0.8, 0.6, -0.9], &[4, 2])
}

fn sym(x: &Tensor) -> Result<Tensor> {
    x.add(&x.transpose(0, 1)?)?.mul_scalar(0.5)
}

macro_rules! check {
    ($name:ident, $inputs:expr, $f:expr) => {
        #[test]
        fn $name() {
            let r = gradcheck($f, &$inputs, 1e-6, 1e-5, 1e-4);
            assert!(r.is_ok(), "{}: {:?}", stringify!($name), r.err());
        }
    };
}

check!(inv, [gmat()], |x| x[0].inv());
check!(det, [gmat()], |x| x[0].det());
check!(slogdet_logabs, [gmat()], |x| Ok(x[0].slogdet()?.1));
check!(solve, [gmat(), t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])], |x| x[0].solve(&x[1]));
check!(cholesky, [spd()], |x| sym(&x[0])?.cholesky());
check!(qr_r, [tall()], |x| Ok(x[0].qr()?.1));
check!(qr_q, [tall()], |x| Ok(x[0].qr()?.0));
check!(eigh_values, [spd()], |x| Ok(sym(&x[0])?.eigh()?.0));
check!(eigh_vectors_gauge_free, [spd()], |x| {
    // eigenvectors are defined up to sign: check a sign-invariant function, the projector V V^T scaled by values
    let (w, v) = sym(&x[0])?.eigh()?;
    v.mul(&w.unsqueeze(0)?)?.matmul(&v.transpose(0, 1)?)
});
check!(svd_values, [tall()], |x| Ok(x[0].svd()?.1));
check!(svd_reconstruction_gauge_free, [tall()], |x| {
    let (u, s, vt) = x[0].svd()?;
    u.mul(&s.unsqueeze(0)?)?.matmul(&vt)
});
check!(pinv, [tall()], |x| x[0].pinv(1e-12));
check!(lstsq, [tall(), t(&[1.0, 2.0, 3.0, 4.0], &[4, 1])], |x| x[0].lstsq(&x[1], 1e-12));
check!(matrix_exp, [t(&[0.1, 0.3, -0.2, 0.05], &[2, 2])], |x| x[0].matrix_exp());
check!(cholesky_solve, [spd(), t(&[1.0, 2.0, 3.0], &[3, 1])], |x| {
    let l = sym(&x[0])?.cholesky()?;
    x[1].cholesky_solve(&l, false)
});
