//! Regression tests for the core-tensor improvement job (2026-11-04 bug-hunt follow-up).
//!
//! Locks in the two genuinely open items found during triage:
//!
//! - `gemm.rs` MEDIUM: the strided/batched GEMM entry points did no bounds
//!   validation, so bad caller args panicked deep in the kernels. The raw
//!   entry points keep their signatures (and now fail loudly at the gate),
//!   while new `try_` wrappers return `Err` instead.
//! - `sparse.rs` R2 gap: `SparseCoo::transpose` was still infallible while its
//!   CSR sibling returns `Result`; hand-built invalid COO structs now error.

use oxideon_core::gemm::{try_batched_gemm_f32, try_batched_gemm_f64, try_gemm_f32_strided};
use oxideon_core::prelude::*;

fn assert_f32_close(got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "element {i}: got {g}, want {w} (tol {tol})"
        );
    }
}

fn assert_f64_close(got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "element {i}: got {g}, want {w} (tol {tol})"
        );
    }
}

// ---------------------------------------------------------------------------
// Fallible GEMM entry points
// ---------------------------------------------------------------------------

#[test]
fn try_gemm_f32_valid_matches_tensor_matmul() {
    // 4x3 @ 3x5 with nonzero offsets and a flipped-B layout.
    let a: Vec<f32> = (0..14).map(|i| i as f32 - 6.0).collect();
    let b: Vec<f32> = (0..17).map(|i| (i as f32) * 0.5 - 2.0).collect();
    let (m, k, n) = (4, 3, 5);
    let mut c = vec![f32::NAN; m * n];
    try_gemm_f32_strided(&a, 2, k as isize, 1, &b, 1, n as isize, 1, &mut c, m, k, n).unwrap();
    assert!(c.iter().all(|v| !v.is_nan()), "output not fully written");

    // Cross-check against the high-level matmul on the same logical data.
    let a_t = Tensor::from_slice(&a[2..2 + m * k], &[m, k]).unwrap();
    let b_t = Tensor::from_slice(&b[1..1 + k * n], &[k, n]).unwrap();
    let want = a_t.matmul(&b_t).unwrap().to_vec::<f32>().unwrap();
    assert_f32_close(&c, &want, 1e-4);
}

#[test]
fn try_gemm_f32_rejects_short_output() {
    let a = vec![1.0f32; 6];
    let b = vec![1.0f32; 6];
    let mut c = vec![0.0f32; 3]; // needs 2x2 = 4
    let err = try_gemm_f32_strided(&a, 0, 3, 1, &b, 0, 2, 1, &mut c, 2, 3, 2).unwrap_err();
    assert!(err.to_string().contains("gemm_f32_strided"), "{err}");
}

#[test]
fn try_gemm_f32_rejects_out_of_bounds_a() {
    let a = vec![1.0f32; 6];
    let b = vec![1.0f32; 6];
    let mut c = vec![0.0f32; 4];
    // Offset past the end of `a`.
    assert!(try_gemm_f32_strided(&a, 5, 3, 1, &b, 0, 2, 1, &mut c, 2, 3, 2).is_err());
    // Negative row stride from offset 0 leaves the slice.
    assert!(try_gemm_f32_strided(&a, 0, -3, 1, &b, 0, 2, 1, &mut c, 2, 3, 2).is_err());
    // Overflowing stride is refused, not wrapped.
    assert!(try_gemm_f32_strided(&a, 0, isize::MAX, 1, &b, 0, 2, 1, &mut c, 2, 3, 2).is_err());
}

#[test]
fn try_gemm_f32_rejects_out_of_bounds_b() {
    let a = vec![1.0f32; 6];
    let b = vec![1.0f32; 6];
    let mut c = vec![0.0f32; 4];
    assert!(try_gemm_f32_strided(&a, 0, 3, 1, &b, 4, 2, 1, &mut c, 2, 3, 2).is_err());
    assert!(try_gemm_f32_strided(&a, 0, 3, 1, &b, 0, 2, 7, &mut c, 2, 3, 2).is_err());
}

#[test]
fn try_gemm_f32_empty_dims_succeed_without_reading() {
    // m = 0 with valid B geometry: the A view reads nothing, so its absurd
    // offset is fine.
    let a: Vec<f32> = vec![];
    let b = vec![0.0f32; 6];
    let mut c: Vec<f32> = vec![];
    try_gemm_f32_strided(&a, usize::MAX, 1, 1, &b, 0, 2, 1, &mut c, 0, 3, 2).unwrap();
    // A garbage B view (k, n > 0) is still refused when m == 0: every
    // declared view must fit its slice.
    let bad_b: Vec<f32> = vec![];
    assert!(try_gemm_f32_strided(&a, 0, 1, 1, &bad_b, usize::MAX, 1, 1, &mut c, 0, 3, 2).is_err());
    // k = 0: the output is zeros.
    let a = vec![1.0f32; 4];
    let b: Vec<f32> = vec![];
    let mut c = vec![9.0f32; 4];
    try_gemm_f32_strided(&a, 0, 0, 1, &b, 0, 2, 1, &mut c, 2, 0, 2).unwrap();
    assert_eq!(c, vec![0.0f32; 4]);
}

#[test]
fn try_gemm_f32_overflowing_dims_are_errors_not_panics() {
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];
    let mut c: Vec<f32> = vec![];
    // m * n overflows usize: must be Err, never a panic or wrap.
    assert!(try_gemm_f32_strided(&a, 0, 1, 1, &b, 0, 1, 1, &mut c, usize::MAX, 1, 2).is_err());
}

#[test]
fn try_batched_gemm_f32_valid_matches_per_batch_reference() {
    let (m, k, n, nb) = (3, 4, 2, 3);
    let a: Vec<f32> = (0..(nb * m * k)).map(|i| i as f32 * 0.25 - 3.0).collect();
    let b: Vec<f32> = (0..(nb * k * n)).map(|i| i as f32 * 0.5 - 1.0).collect();
    let offsets: Vec<(usize, usize)> = (0..nb).map(|i| (i * m * k, i * k * n)).collect();
    let c = try_batched_gemm_f32(&a, k as isize, 1, &b, n as isize, 1, &offsets, m, k, n).unwrap();
    assert_eq!(c.len(), nb * m * n);
    // Each batch must equal the single-product wrapper on the same slices.
    for (i, &(oa, ob)) in offsets.iter().enumerate() {
        let mut one = vec![0.0f32; m * n];
        try_gemm_f32_strided(
            &a, oa, k as isize, 1, &b, ob, n as isize, 1, &mut one, m, k, n,
        )
        .unwrap();
        assert_f32_close(&c[i * m * n..(i + 1) * m * n], &one, 1e-5);
    }
}

#[test]
fn try_batched_gemm_f32_rejects_bad_batch_offset() {
    let (m, k, n) = (3, 4, 2);
    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];
    // Second batch reads past the end of `a`.
    let offsets = vec![(0usize, 0usize), (1usize, 0usize)];
    let err =
        try_batched_gemm_f32(&a, k as isize, 1, &b, n as isize, 1, &offsets, m, k, n).unwrap_err();
    assert!(err.to_string().contains("batched_gemm_f32"), "{err}");
}

#[test]
fn try_batched_gemm_f32_refuses_giant_output_without_allocating() {
    // 2^20 x 2^12 = 2^32 elements > MAX_ELEMENTS: refused up front.
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];
    let err = try_batched_gemm_f32(&a, 1, 1, &b, 1, 1, &[(0, 0)], 1 << 20, 1, 1 << 12).unwrap_err();
    assert!(err.to_string().contains("batched_gemm_f32"), "{err}");
}

#[test]
fn try_batched_gemm_f64_valid_and_rejects_bad_batch() {
    let (m, k, n, nb) = (2, 3, 4, 2);
    let a: Vec<f64> = (0..(nb * m * k)).map(|i| i as f64 - 5.0).collect();
    let b: Vec<f64> = (0..(nb * k * n)).map(|i| i as f64 * 0.5).collect();
    let offsets: Vec<(usize, usize)> = (0..nb).map(|i| (i * m * k, i * k * n)).collect();
    let c = try_batched_gemm_f64(&a, k as isize, 1, &b, n as isize, 1, &offsets, m, k, n).unwrap();
    assert_eq!(c.len(), nb * m * n);
    // Spot-check batch 0 against Tensor::matmul.
    let a0 = Tensor::from_slice(&a[..m * k], &[m, k]).unwrap();
    let b0 = Tensor::from_slice(&b[..k * n], &[k, n]).unwrap();
    let want = a0.matmul(&b0).unwrap().to_vec::<f64>().unwrap();
    assert_f64_close(&c[..m * n], &want, 1e-12);

    let bad = vec![(0usize, 0usize), (usize::MAX, 0usize)];
    let err =
        try_batched_gemm_f64(&a, k as isize, 1, &b, n as isize, 1, &bad, m, k, n).unwrap_err();
    assert!(err.to_string().contains("batched_gemm_f64"), "{err}");
}

// ---------------------------------------------------------------------------
// Fallible SparseCoo::transpose
// ---------------------------------------------------------------------------

#[test]
fn coo_transpose_valid_swaps_shape_and_entries() {
    let coo = SparseCoo::new(vec![(0, 1), (1, 0), (1, 2)], vec![2.0, 3.0, 4.0], [2, 3]).unwrap();
    let t = coo.transpose().unwrap();
    assert_eq!(t.shape, [3, 2]);
    let dense = t
        .to_dense(DType::F64, &Device::Cpu)
        .unwrap()
        .to_vec::<f64>()
        .unwrap();
    assert_eq!(dense, vec![0.0, 3.0, 2.0, 0.0, 0.0, 4.0]);
}

#[test]
fn coo_transpose_rejects_hand_built_invalid_structs() {
    // Out-of-bounds index, bypassing SparseCoo::new validation.
    let bad = SparseCoo {
        indices: vec![(5, 5)],
        values: vec![1.0],
        shape: [2, 2],
    };
    assert!(bad.transpose().is_err());
    // Length mismatch (zip would silently truncate).
    let skew = SparseCoo {
        indices: vec![(0, 0), (1, 1)],
        values: vec![1.0],
        shape: [2, 2],
    };
    assert!(skew.transpose().is_err());
}

#[test]
fn coo_transpose_roundtrip_restores_coalesced_entries() {
    let coo = SparseCoo::new(vec![(1, 0), (0, 2), (1, 0)], vec![1.0, 2.0, 3.0], [2, 3]).unwrap();
    // (1, 0) coalesces to 4.0; double transpose must restore it.
    let back = coo.transpose().unwrap().transpose().unwrap();
    assert_eq!(back.shape, [2, 3]);
    let dense = back
        .to_dense(DType::F64, &Device::Cpu)
        .unwrap()
        .to_vec::<f64>()
        .unwrap();
    assert_eq!(dense, vec![0.0, 0.0, 2.0, 4.0, 0.0, 0.0]);
}


// ---------------------------------------------------------------------------
// Core Engine Enrichment Tests (Core-1..Core-10)
// ---------------------------------------------------------------------------

#[test]
fn core_enrichment_trig_hyperbolic_and_scans_grad() {
    let x = Tensor::from_slice(&[0.0f64, 0.5, -0.5], &[3])
        .unwrap()
        .requires_grad_(true);
    let y = x
        .tan()
        .unwrap()
        .add(&x.asin().unwrap())
        .unwrap()
        .add(&x.acos().unwrap())
        .unwrap()
        .add(&x.atan().unwrap())
        .unwrap()
        .add(&x.sinc().unwrap())
        .unwrap();
    y.sum().unwrap().backward().unwrap();
    let g = x.grad().unwrap().to_vec::<f64>().unwrap();
    assert!((g[0] - 2.0).abs() < 1e-10);

    // cummax & cummin autograd
    let s = Tensor::from_slice(&[2.0f64, 1.0, 4.0, 3.0], &[4])
        .unwrap()
        .requires_grad_(true);
    let (cmax, cidx) = s.cummax(0).unwrap();
    assert_eq!(cidx.to_vec::<i64>().unwrap(), vec![0, 0, 2, 2]);
    cmax.sum().unwrap().backward().unwrap();
    assert_eq!(s.grad().unwrap().to_vec::<f64>().unwrap(), vec![2.0, 0.0, 2.0, 0.0]);

    // Zero-safe prod autograd
    let p_in = Tensor::from_slice(&[3.0f64, 0.0, 4.0], &[3])
        .unwrap()
        .requires_grad_(true);
    let p_out = p_in.prod().unwrap();
    p_out.backward().unwrap();
    assert_eq!(p_in.grad().unwrap().to_vec::<f64>().unwrap(), vec![0.0, 12.0, 0.0]);
}

#[test]
fn core_enrichment_linalg_and_fused_kernels() {
    use oxideon_core::{fused_layernorm, fused_rope, fused_sdpa};

    let a = Tensor::from_slice(&[4.0f64, 1.0, 1.0, 3.0], &[2, 2])
        .unwrap()
        .requires_grad_(true);
    let b = Tensor::from_slice(&[1.0f64, 2.0], &[2, 1])
        .unwrap()
        .requires_grad_(true);
    let l = a.cholesky().unwrap();
    let sol = b.cholesky_solve(&l, false).unwrap();
    let recon = a.matmul(&sol).unwrap().to_vec::<f64>().unwrap();
    assert_f64_close(&recon, &[1.0, 2.0], 1e-10);

    let (p_mat, l_mat, u_mat) = a.lu().unwrap();
    let plu = p_mat
        .matmul(&l_mat)
        .unwrap()
        .matmul(&u_mat)
        .unwrap()
        .to_vec::<f64>()
        .unwrap();
    assert_f64_close(&plu, &[4.0, 1.0, 1.0, 3.0], 1e-10);

    let x = Tensor::from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[1, 4])
        .unwrap()
        .requires_grad_(true);
    let w = Tensor::from_slice(&[1.0f64, 1.0, 1.0, 1.0], &[4])
        .unwrap()
        .requires_grad_(true);
    let ln = fused_layernorm(&x, Some(&w), None, 1e-5).unwrap();
    ln.sum().unwrap().backward().unwrap();
    assert!(x.grad().is_some() && w.grad().is_some());

    let cos = Tensor::from_slice(&[0.3f64.cos(), 0.7f64.cos()], &[1, 2]).unwrap();
    let sin = Tensor::from_slice(&[0.3f64.sin(), 0.7f64.sin()], &[1, 2]).unwrap();
    let rope = fused_rope(&x, &cos, &sin).unwrap();
    let norm_before = x.square().unwrap().sum().unwrap().item::<f64>().unwrap();
    let norm_after = rope.square().unwrap().sum().unwrap().item::<f64>().unwrap();
    assert!((norm_before - norm_after).abs() < 1e-10);

    let q = Tensor::from_slice(&[1.0f64, 0.0, 0.0, 1.0], &[2, 2])
        .unwrap()
        .requires_grad_(true);
    let k = Tensor::from_slice(&[1.0f64, 0.0, 0.0, 1.0], &[2, 2])
        .unwrap()
        .requires_grad_(true);
    let v = Tensor::from_slice(&[2.0f64, 3.0, 4.0, 5.0], &[2, 2])
        .unwrap()
        .requires_grad_(true);
    let attn = fused_sdpa(&q, &k, &v, None, true).unwrap();
    let attn_vec = attn.to_vec::<f64>().unwrap();
    assert_f64_close(&attn_vec[..2], &[2.0, 3.0], 1e-10);
}
