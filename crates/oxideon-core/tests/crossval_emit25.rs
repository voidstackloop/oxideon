//! Cross-validation batch 25: values and gradients of linear-algebra / contraction ops vs PyTorch (float64)
//! (tools/crossval/check_batch25.py).
use oxideon_core::tensor::Tensor;

fn mat(rows: usize, cols: usize, salt: f64) -> Vec<f64> {
    (0..rows * cols).map(|k| (0.83 * k as f64 + salt).sin() + if k / cols == k % cols { 2.5 } else { 0.0 }).collect()
}

fn wts(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_slice(&(0..n).map(|k| (1.3 * k as f64 + 0.4).sin()).collect::<Vec<_>>(), shape).unwrap()
}

fn leaf(v: &[f64], shape: &[usize]) -> Tensor {
    Tensor::from_slice(v, shape).unwrap().requires_grad_(true)
}

fn dump(name: &str, t: &Tensor) {
    let v = t.to_vec::<f64>().unwrap();
    println!("{name} {}", v.iter().map(|x| format!("{x:.14e}")).collect::<Vec<_>>().join(" "));
}

/// Emits the value(s) and the gradient of sum(out * w) for each input.
fn case(name: &str, inputs: &[&Tensor], outs: &[Tensor]) {
    let mut loss: Option<Tensor> = None;
    for (k, o) in outs.iter().enumerate() {
        dump(&format!("{name}_out{k}"), o);
        let term = o.mul(&wts(o.shape())).unwrap().sum().unwrap();
        loss = Some(match loss {
            Some(l) => l.add(&term).unwrap(),
            None => term,
        });
    }
    loss.unwrap().backward().unwrap();
    for (k, x) in inputs.iter().enumerate() {
        dump(&format!("{name}_g{k}"), &x.grad().unwrap());
    }
}

#[test]
#[ignore]
fn emit_linalg() {
    let (av, bv, tv, bv2) = (mat(4, 4, 0.2), mat(4, 3, 1.1), mat(5, 3, 2.3), mat(5, 2, 3.1));
    // symmetric positive definite
    let sv: Vec<f64> = {
        let m = mat(4, 4, 4.4);
        (0..16).map(|k| (m[(k / 4) * 4 + k % 4] + m[(k % 4) * 4 + k / 4]) * 0.5 + if k / 4 == k % 4 { 3.0 } else { 0.0 }).collect()
    };
    macro_rules! with {
        ($name:literal, [$(($x:ident, $v:expr, $shape:expr)),*], $body:expr) => {{
            $( let $x = leaf(&$v, $shape); )*
            let outs: Vec<Tensor> = $body;
            case($name, &[$(&$x),*], &outs);
        }};
    }
    with!("matmul", [(a, av, &[4, 4]), (b, bv, &[4, 3])], vec![a.matmul(&b).unwrap()]);
    with!("det", [(a, av, &[4, 4])], vec![a.det().unwrap().reshape(&[1]).unwrap()]);
    with!("inv", [(a, av, &[4, 4])], vec![a.inv().unwrap()]);
    with!("solve", [(a, av, &[4, 4]), (b, bv, &[4, 3])], vec![a.solve(&b).unwrap()]);
    with!("slogdet", [(a, av, &[4, 4])], vec![a.slogdet().unwrap().1.reshape(&[1]).unwrap()]);
    with!("cholesky", [(s, sv, &[4, 4])], vec![s.cholesky().unwrap()]);
    with!("logdet", [(s, sv, &[4, 4])], vec![s.logdet().unwrap().reshape(&[1]).unwrap()]);
    with!("eigvalsh", [(s, sv, &[4, 4])], vec![s.eigvalsh().unwrap()]);
    // eigh through an eigenvector-sign-invariant function: V diag(exp(L)) V^T
    with!("eigh_fn", [(s, sv, &[4, 4])], {
        let (l, v) = s.eigh().unwrap();
        let d = l.exp().unwrap().reshape(&[1, 4]).unwrap();
        vec![v.mul(&d).unwrap().matmul(&v.t().unwrap()).unwrap()]
    });
    with!("qr", [(t, tv, &[5, 3])], {
        let (q, r) = t.qr().unwrap();
        vec![q, r]
    });
    // singular values and a sign-invariant function of (U, S, V^T)
    with!("svd_s", [(t, tv, &[5, 3])], vec![t.svd().unwrap().1]);
    with!("svd_fn", [(t, tv, &[5, 3])], {
        let (u, s, vt) = t.svd().unwrap();
        let sq = s.sqrt().unwrap().reshape(&[1, 3]).unwrap();
        vec![u.mul(&sq).unwrap().matmul(&vt).unwrap()]
    });
    with!("svd_fn_wide", [(t, mat(3, 5, 2.7), &[3, 5])], {
        let (u, s, vt) = t.svd().unwrap();
        let sq = s.sqrt().unwrap().reshape(&[1, 3]).unwrap();
        vec![u.mul(&sq).unwrap().matmul(&vt).unwrap()]
    });
    with!("pinv", [(t, tv, &[5, 3])], vec![t.pinv(1e-15).unwrap()]);
    with!("lstsq", [(t, tv, &[5, 3]), (b, bv2, &[5, 2])], vec![t.lstsq(&b, 1e-15).unwrap()]);
    with!("matrix_exp", [(a, av, &[4, 4])], vec![a.mul_scalar(0.2).unwrap().matrix_exp().unwrap()]);
    with!("matrix_power3", [(a, av, &[4, 4])], vec![a.mul_scalar(0.3).unwrap().matrix_power(3).unwrap()]);
    with!("matrix_power-2", [(a, av, &[4, 4])], vec![a.matrix_power(-2).unwrap()]);
    with!("trace", [(a, av, &[4, 4])], vec![a.trace().unwrap().reshape(&[1]).unwrap()]);
    with!("kron", [(a, mat(2, 2, 0.5), &[2, 2]), (b, mat(2, 3, 1.5), &[2, 3])], vec![a.kron(&b).unwrap()]);
    with!("cross", [(a, mat(4, 3, 0.7), &[4, 3]), (b, mat(4, 3, 1.9), &[4, 3])], vec![a.cross(&b, 1).unwrap()]);
    with!("einsum_mm", [(a, av, &[4, 4]), (b, bv, &[4, 3])], vec![Tensor::einsum("ij,jk->ik", &[&a, &b]).unwrap()]);
    with!("einsum_batch", [(a, mat(6, 4, 2.0), &[2, 3, 4]), (b, mat(8, 3, 3.0), &[2, 4, 3])], vec![Tensor::einsum("bij,bjk->bik", &[&a, &b]).unwrap()]);
    with!("einsum_trace_outer", [(a, av, &[4, 4]), (b, bv, &[4, 3])], vec![Tensor::einsum("ii->", &[&a]).unwrap().reshape(&[1]).unwrap(), Tensor::einsum("ij,kl->ikjl", &[&a, &b]).unwrap()]);
    with!("tensordot", [(a, mat(6, 4, 2.2), &[2, 3, 4]), (b, mat(12, 2, 3.3), &[3, 4, 2])], vec![a.tensordot(&b, &[1, 2], &[0, 1]).unwrap()]);
    with!("cdist", [(a, mat(5, 3, 0.9), &[5, 3]), (b, mat(4, 3, 2.9), &[4, 3])], vec![a.cdist(&b).unwrap()]);
    with!("cov", [(a, mat(3, 6, 1.7), &[3, 6])], vec![a.cov(1).unwrap()]);
    with!("tri_solve", [(a, av, &[4, 4]), (b, bv, &[4, 3])], vec![a.triangular_solve(&b, true, false).unwrap()]);
    with!("tri_solve_t", [(a, av, &[4, 4]), (b, bv, &[4, 3])], vec![a.triangular_solve(&b, false, true).unwrap()]);
}
