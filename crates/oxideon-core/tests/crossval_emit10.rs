//! Tenth cross-validation batch: general tensor operations vs PyTorch.
//! Run: `cargo test -p oxideon-core --test crossval_emit10 -- --nocapture --ignored`.

use oxideon_core::tensor::Tensor;

fn m(v: &[f64], shape: &[usize]) -> Tensor {
    Tensor::from_slice(v, shape).unwrap()
}

fn i64s(v: &[i64], shape: &[usize]) -> Tensor {
    Tensor::from_slice(v, shape).unwrap()
}

fn emit(name: &str, x: &Tensor) {
    x.to_dtype(oxideon_core::dtype::DType::F64).unwrap().to_vec::<f64>().unwrap().iter().enumerate().for_each(|(i, v)| println!("{name}_{i} {v:.12e}"));
}

#[test]
#[ignore]
fn emit_ops() {
    let a = m(&[0.5, -1.2, 3.3, 2.0, -0.7, 1.1], &[2, 3]);
    let b = m(&[1.5, 0.4, -2.2, 0.9, 3.0, -0.1], &[2, 3]);
    emit("logaddexp", &a.logaddexp(&b).unwrap());
    emit("logaddexp2", &a.logaddexp2(&b).unwrap());
    emit("float_power", &a.abs().unwrap().float_power(&b).unwrap());
    emit("ldexp", &a.ldexp(&i64s(&[1, -2, 3, 0, 4, -1], &[2, 3])).unwrap());
    let (mant, exp) = a.frexp().unwrap();
    emit("frexp_m", &mant);
    emit("frexp_e", &exp);
    emit("nextafter", &m(&[1.0, -1.0, 0.0, 2.5], &[4]).nextafter(&m(&[2.0, -2.0, 1.0, -3.0], &[4])).unwrap());
    emit("gcd", &i64s(&[12, -18, 7, 0, 100], &[5]).gcd(&i64s(&[18, 12, 5, 9, 75], &[5])).unwrap());
    emit("lcm", &i64s(&[4, 6, 7, 3], &[4]).lcm(&i64s(&[6, 8, 5, 0], &[4])).unwrap());
    emit("hypot", &a.hypot(&b).unwrap());
    emit("copysign", &a.copysign(&b).unwrap());
    emit("trapezoid", &b.trapezoid(1, 0.5).unwrap());
    emit("cumtrapz", &b.cumulative_trapezoid(1, 0.5).unwrap());
    for k in [1i64, 2, 3, -1] {
        emit(&format!("rot90_{}", if k < 0 { format!("neg{}", -k) } else { k.to_string() }), &a.rot90(k, [0, 1]).unwrap());
    }
    emit("take_along_dim", &a.take_along_dim(&i64s(&[2, 0, 1, 1, 1, 0], &[2, 3]), 1).unwrap());
    emit("index_put_acc", &m(&[0.0; 6], &[6]).index_put(&[i64s(&[1, 3, 1, 5], &[4])], &m(&[1.0, 2.0, 3.0, 4.0], &[4]), true).unwrap());
    emit("index_copy", &m(&[0.0; 6], &[3, 2]).index_copy(0, &i64s(&[2, 0], &[2]), &m(&[1.0, 2.0, 3.0, 4.0], &[2, 2])).unwrap());
    let idx = i64s(&[0, 1, 0, 1, 2, 1, 0, 2, 2], &[3, 3]);
    let src = m(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]);
    let base = m(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0], &[3, 3]);
    for r in ["sum", "prod", "mean", "amax", "amin"] {
        for inc in [true, false] {
            emit(&format!("scatter_reduce_{r}_{inc}"), &base.scatter_reduce(0, &idx, &src, r, inc).unwrap());
        }
    }
    emit("unique", &m(&[3.0, 1.0, 2.0, 3.0, 1.0, 5.0], &[6]).unique().unwrap());
    let bounds = m(&[1.0, 2.5, 4.0], &[3]);
    let vals = m(&[0.5, 1.0, 2.5, 3.0, 4.0, 5.0], &[6]);
    emit("bucketize_left", &vals.bucketize(&bounds, false).unwrap());
    emit("bucketize_right", &vals.bucketize(&bounds, true).unwrap());
    emit("searchsorted_left", &Tensor::searchsorted(&bounds, &vals, false).unwrap());
    emit("searchsorted_right", &Tensor::searchsorted(&bounds, &vals, true).unwrap());
    emit("tril_idx", &Tensor::tril_indices(4, 3, 0).unwrap());
    emit("tril_idx_off", &Tensor::tril_indices(3, 4, 1).unwrap());
    emit("triu_idx", &Tensor::triu_indices(4, 3, 0).unwrap());
    emit("triu_idx_off", &Tensor::triu_indices(3, 4, -1).unwrap());
    emit("combinations", &m(&[1.0, 2.0, 3.0, 4.0], &[4]).combinations(2, false).unwrap());
    emit("combinations_rep", &m(&[1.0, 2.0, 3.0], &[3]).combinations(2, true).unwrap());
    emit("renorm", &a.renorm(2.0, 0, 1.5).unwrap());
    let (tv, ti) = m(&[3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 5.0, 3.5], &[2, 4]).topk(2, 1, true, true).unwrap();
    emit("topk_v", &tv);
    emit("topk_i", &ti);
    let (tv, ti) = m(&[3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 5.0, 3.5], &[2, 4]).topk(3, 1, false, true).unwrap();
    emit("topk_small_v", &tv);
    emit("topk_small_i", &ti);
    let (kv, ki) = m(&[3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 5.0, 3.5], &[2, 4]).kthvalue(2, 1, false).unwrap();
    emit("kth_v", &kv);
    emit("kth_i", &ki);
    emit("diff2", &m(&[1.0, 4.0, 9.0, 16.0, 25.0, 36.0], &[6]).diff(2, 0).unwrap());
    emit("bincount", &i64s(&[0, 1, 1, 3, 3, 3, 6], &[7]).bincount(None, 0).unwrap());
    emit("bincount_w", &i64s(&[0, 1, 1, 3], &[4]).bincount(Some(&m(&[0.5, 1.0, 2.0, 0.25], &[4])), 6).unwrap());
    emit("histc", &m(&[0.1, 0.5, 0.9, 1.4, 2.2, 2.9, 3.0, 3.5], &[8]).histc(4, 0.0, 3.5).unwrap());
    let cm = m(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], &[2, 4]);
    let (cv, ci) = cm.cummax(1).unwrap();
    emit("cummax_v", &cv);
    emit("cummax_i", &ci);
    let (cv, ci) = cm.cummin(1).unwrap();
    emit("cummin_v", &cv);
    emit("cummin_i", &ci);
    emit("logcumsumexp", &cm.logcumsumexp(1).unwrap());
    emit("cross", &m(&[1.0, 2.0, 3.0, -1.0, 0.5, 2.0], &[2, 3]).cross(&m(&[0.5, -1.0, 2.0, 3.0, 1.0, -2.0], &[2, 3]), 1).unwrap());
    emit("tensordot", &m(&(0..24).map(|i| (i as f64 * 0.37).sin()).collect::<Vec<_>>(), &[2, 3, 4]).tensordot(&m(&(0..12).map(|i| (i as f64 * 0.21).cos()).collect::<Vec<_>>(), &[3, 4]), &[1, 2], &[0, 1]).unwrap());
    emit("block_diag", &Tensor::block_diag(&[&m(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), &m(&[5.0, 6.0, 7.0], &[1, 3])]).unwrap());
    emit("vander", &m(&[1.0, 2.0, 3.0], &[3]).vander(Some(4), true).unwrap());
    emit("cartesian", &Tensor::cartesian_prod(&[&m(&[1.0, 2.0], &[2]), &m(&[3.0, 4.0, 5.0], &[3])]).unwrap());
    emit("kron", &m(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).kron(&m(&[0.0, 5.0, 6.0, 7.0], &[2, 2])).unwrap());
    let obs = m(&(0..15).map(|i| ((i * 7 % 11) as f64) / 3.0 - 1.0).collect::<Vec<_>>(), &[3, 5]);
    emit("cov", &obs.cov(1).unwrap());
    emit("cov0", &obs.cov(0).unwrap());
    emit("corrcoef", &obs.corrcoef().unwrap());
    emit("cdist", &m(&[0.0, 0.0, 1.0, 2.0, -1.0, 0.5], &[3, 2]).cdist(&m(&[1.0, 1.0, 0.0, -2.0], &[2, 2])).unwrap());
    let e1 = m(&(0..12).map(|i| (i as f64 * 0.3).sin()).collect::<Vec<_>>(), &[3, 4]);
    let e2 = m(&(0..8).map(|i| (i as f64 * 0.5).cos()).collect::<Vec<_>>(), &[4, 2]);
    emit("einsum_mm", &Tensor::einsum("ij,jk->ik", &[&e1, &e2]).unwrap());
    emit("einsum_trace", &Tensor::einsum("ii->", &[&m(&[1.0, 2.0, 3.0, 4.0], &[2, 2])]).unwrap());
    emit("einsum_outer", &Tensor::einsum("i,j->ij", &[&m(&[1.0, 2.0], &[2]), &m(&[3.0, 4.0, 5.0], &[3])]).unwrap());
    emit("einsum_batch", &Tensor::einsum("bij,bjk->bik", &[&m(&(0..12).map(|i| i as f64 * 0.1).collect::<Vec<_>>(), &[2, 2, 3]), &m(&(0..12).map(|i| 1.0 - i as f64 * 0.05).collect::<Vec<_>>(), &[2, 3, 2])]).unwrap());
    emit("nanmedian", &m(&[3.0, f64::NAN, 1.0, 2.0, 7.0], &[5]).nanmedian().unwrap());
}
