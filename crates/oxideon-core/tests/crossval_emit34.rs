//! Cross-validation batch 34: sparse COO / CSR operations vs scipy.sparse (tools/crossval/check_batch34.py).
use oxideon_core::device::Device;
use oxideon_core::dtype::DType;
use oxideon_core::sparse::SparseCoo;
use oxideon_core::tensor::Tensor;

fn u(i: u64) -> f64 {
    let mut z = i.wrapping_add(0x9E37_79B9_7F4A_7C15).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 30)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// Triplets (with a few duplicate coordinates and an exact-zero entry) of a rows x cols matrix.
fn triplets(rows: usize, cols: usize, density: f64, seed: u64) -> (Vec<(usize, usize)>, Vec<f64>) {
    let (mut idx, mut val) = (vec![], vec![]);
    for r in 0..rows {
        for c in 0..cols {
            if u(seed * 100_000 + (r * cols + c) as u64) < density {
                idx.push((r, c));
                val.push((u(seed * 100_000 + 50_000 + (r * cols + c) as u64) - 0.5) * 6.0);
            }
        }
    }
    // duplicates that must be summed, and an explicit zero
    idx.push(idx[0]);
    val.push(1.25);
    idx.push((rows - 1, cols - 1));
    val.push(0.0);
    (idx, val)
}

fn dump(name: &str, m: &SparseCoo) {
    let d = m.to_dense(DType::F64, &Device::Cpu).unwrap().to_vec::<f64>().unwrap();
    println!("{name} {:?} {}", m.shape, d.iter().map(|v| format!("{v:.13e}")).collect::<Vec<_>>().join(" "));
}

fn dump_vec(name: &str, v: &[f64]) {
    println!("{name} [{}] {}", v.len(), v.iter().map(|x| format!("{x:.13e}")).collect::<Vec<_>>().join(" "));
}

fn dump_t(name: &str, t: &Tensor) {
    let v = t.to_vec::<f64>().unwrap();
    println!("{name} {:?} {}", t.shape(), v.iter().map(|x| format!("{x:.13e}")).collect::<Vec<_>>().join(" "));
}

fn build(rows: usize, cols: usize, density: f64, seed: u64) -> SparseCoo {
    let (i, v) = triplets(rows, cols, density, seed);
    SparseCoo::new(i, v, [rows, cols]).unwrap()
}

#[test]
#[ignore]
fn emit_sparse() {
    let a = build(7, 6, 0.35, 1);
    let b = build(6, 5, 0.4, 2);
    let c = build(7, 6, 0.3, 3);
    let sq = build(6, 6, 0.4, 4);
    dump("a", &a);
    dump("b", &b);
    dump("c", &c);
    dump("sq", &sq);
    println!("a_nnz {}", a.nnz());
    dump("a_t", &a.transpose().unwrap());
    dump("a_csr_t", &a.to_csr().unwrap().transpose().unwrap().to_coo().unwrap());
    dump("a_plus_c", &a.add(&c).unwrap());
    dump("a_minus_c", &a.sub(&c).unwrap());
    dump("a_scaled", &a.mul_scalar(-2.5).unwrap());
    dump("ab", &a.spspmm(&b).unwrap());
    dump("ab_csr", &a.to_csr().unwrap().spspmm(&b.to_csr().unwrap()).unwrap().to_coo().unwrap());
    dump("a_had_c", &a.hadamard(&c).unwrap());
    dump("a_kron_b", &build(2, 3, 0.7, 5).kron(&build(3, 2, 0.7, 6)).unwrap());
    for (up, k) in [(true, 0i64), (true, 1), (true, -2), (false, 0), (false, 2), (false, -1)] {
        dump(&format!("tri_{}_{k}", u8::from(up)), &sq.triangle(up, k).unwrap());
    }
    dump("rows_sel", &a.select_rows(&[3, 0, 3, 6]).unwrap());
    dump("eye5", &SparseCoo::eye(5).unwrap());
    dump("diag_from", &SparseCoo::from_diagonal(&[1.5, 0.0, -2.0, 4.0]).unwrap());
    dump_vec("sq_diag", &sq.diag());
    dump_vec("sum_axis0", &a.sum_axis(0).unwrap());
    dump_vec("sum_axis1", &a.sum_axis(1).unwrap());
    let x: Vec<f64> = (0..6).map(|i| (i as f64 * 0.7).sin() + 0.3).collect();
    dump_vec("spmv", &a.spmv(&x).unwrap());
    println!("a_norm {:.13e}", a.norm());
    println!("a_sparsity {:.13e}", a.sparsity());
    // row-wise softmax over stored entries (absent entries behave like -inf)
    dump("a_softmax", &a.softmax_rows().unwrap());

    // spmm and its gradient (COO and CSR)
    let dense = Tensor::from_slice(&(0..6 * 4).map(|i| (0.31 * i as f64).cos()).collect::<Vec<_>>(), &[6, 4]).unwrap().requires_grad_(true);
    let out = a.spmm(&dense).unwrap();
    dump_t("spmm_out", &out);
    let w = Tensor::from_slice(&(0..7 * 4).map(|i| (0.77 * i as f64).sin()).collect::<Vec<_>>(), &[7, 4]).unwrap();
    out.mul(&w).unwrap().sum().unwrap().backward().unwrap();
    dump_t("spmm_dx", &dense.grad().unwrap());
    let dense2 = Tensor::from_slice(&(0..6 * 4).map(|i| (0.31 * i as f64).cos()).collect::<Vec<_>>(), &[6, 4]).unwrap().requires_grad_(true);
    let out = a.to_csr().unwrap().spmm(&dense2).unwrap();
    dump_t("spmm_csr_out", &out);
    out.mul(&w).unwrap().sum().unwrap().backward().unwrap();
    dump_t("spmm_csr_dx", &dense2.grad().unwrap());
    // from_dense round trip with a threshold
    let d = Tensor::from_slice(&[0.0f64, 2.0, 0.001, -3.0, 0.0, 0.5, 1.0, 0.0, -0.0004], &[3, 3]).unwrap();
    dump("from_dense", &SparseCoo::from_dense(&d, 0.01).unwrap());
}
