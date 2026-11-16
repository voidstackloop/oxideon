//! Eighth cross-validation batch: linear algebra, windows/DCT, interpolation/polynomial and convolution routines vs
//! NumPy / SciPy. Run: `cargo test -p oxideon-core --test crossval_emit8 -- --nocapture --ignored`.

use oxideon_core::dtype::DType;
use oxideon_core::numeric_ext::*;
use oxideon_core::signal::*;
use oxideon_core::tensor::Tensor;

fn m(v: &[f64], shape: &[usize]) -> Tensor {
    Tensor::from_slice(v, shape).unwrap()
}

fn emit(name: &str, t: &Tensor) {
    t.to_vec::<f64>().unwrap().iter().enumerate().for_each(|(i, x)| println!("{name}_{i} {x:.12e}"));
}

#[test]
#[ignore]
fn emit_numeric() {
    let a = m(&[4.0, 1.0, 0.5, 1.0, 3.0, -0.2, 0.5, -0.2, 2.0], &[3, 3]); // symmetric positive definite
    let g = m(&[1.0, 2.0, -1.0, 0.5, -3.0, 2.2, 4.0, 0.1, 1.5], &[3, 3]);
    emit("svdvals", &g.svdvals().unwrap());
    emit("eigvalsh", &a.eigvalsh().unwrap());
    emit("logdet", &a.logdet().unwrap());
    emit("norm_fro", &g.matrix_norm("fro", false).unwrap());
    emit("norm_nuc", &g.matrix_norm("nuc", false).unwrap());
    emit("norm_2", &g.matrix_norm("2", false).unwrap());
    emit("norm_1", &g.matrix_norm("1", false).unwrap());
    emit("norm_inf", &g.matrix_norm("inf", false).unwrap());
    emit("vnorm3", &m(&[1.0, -2.0, 3.0, 0.5], &[4]).vector_norm(3.0, &[0], false).unwrap());
    let (wr, wi) = m(&[0.0, 1.0, 0.0, -2.0, 0.0, 1.0, 0.5, 0.3, 0.1], &[3, 3]).eigvals().unwrap();
    let mut ev: Vec<(f64, f64)> = wr.to_vec::<f64>().unwrap().into_iter().zip(wi.to_vec::<f64>().unwrap()).collect();
    ev.sort_by(|x, y| ((x.0 * 1e8).round()).total_cmp(&(y.0 * 1e8).round()).then(x.1.total_cmp(&y.1)));
    ev.iter().enumerate().for_each(|(i, (re, im))| {
        println!("eigvals_re_{i} {re:.12e}");
        println!("eigvals_im_{i} {im:.12e}");
    });
    emit("chol_inv", &a.cholesky().unwrap().cholesky_inverse(false).unwrap());
    emit("sqrtm", &sqrtm(&a).unwrap());
    emit("logm", &logm(&a).unwrap());
    // windows
    for (name, w) in [
        ("hann_p", hann_window(9, true, DType::F64)),
        ("hann_s", hann_window(9, false, DType::F64)),
        ("hamming_p", hamming_window(9, true, DType::F64)),
        ("hamming_s", hamming_window(9, false, DType::F64)),
        ("blackman_s", blackman_window(9, false, DType::F64)),
        ("bartlett_s", bartlett_window(9, false, DType::F64)),
        ("kaiser_s", kaiser_window(9, false, 8.6, DType::F64)),
    ] {
        emit(name, &w.unwrap());
    }
    emit("fftfreq", &fftfreq(8, 0.5, DType::F64).unwrap());
    emit("rfftfreq", &rfftfreq(8, 0.5, DType::F64).unwrap());
    let sig = m(&[0.3, -1.2, 2.5, 0.9, -0.4, 1.8, 0.0, 0.7], &[8]);
    emit("dct", &dct(&sig, false).unwrap());
    emit("dct_ortho", &dct(&sig, true).unwrap());
    emit("idct_ortho", &idct(&dct(&sig, true).unwrap(), true).unwrap());
    emit("idct", &idct(&sig, false).unwrap());
    // polynomials / interpolation / convolution
    let xs = m(&[0.0, 1.0, 2.0, 3.5, 5.0, 6.0], &[6]);
    let ys = m(&[1.0, 0.5, 2.2, 1.9, 4.0, 3.1], &[6]);
    emit("polyfit2", &polyfit(&xs, &ys, 2).unwrap());
    emit("polyfit3", &polyfit(&xs, &ys, 3).unwrap());
    emit("polyval", &polyval(&m(&[2.0, -1.0, 0.5, 3.0], &[4]), &m(&[-1.5, 0.0, 2.2], &[3])).unwrap());
    emit("interp", &interp(&m(&[-1.0, 0.5, 2.7, 4.2, 7.0], &[5]), &xs, &ys).unwrap());
    let rts = roots(&m(&[1.0, -6.0, 11.0, -6.0], &[4])).unwrap();
    println!("roots_n {}", rts.re.numel());
    let mut rr: Vec<f64> = rts.re.to_vec::<f64>().unwrap();
    rr.sort_by(f64::total_cmp);
    rr.iter().enumerate().for_each(|(i, v)| println!("roots_{i} {v:.12e}"));
    let (u, v) = (m(&[1.0, 2.0, 3.0, -1.0, 0.5], &[5]), m(&[0.5, -1.0, 2.0], &[3]));
    emit("conv_full", &convolve(&u, &v, ConvMode::Full).unwrap());
    emit("conv_same", &convolve(&u, &v, ConvMode::Same).unwrap());
    emit("conv_valid", &convolve(&u, &v, ConvMode::Valid).unwrap());
    emit("corr_full", &correlate(&u, &v, ConvMode::Full).unwrap());
    emit("corr_same", &correlate(&u, &v, ConvMode::Same).unwrap());
    emit("corr_valid", &correlate(&u, &v, ConvMode::Valid).unwrap());
    emit("fftconv", &fftconvolve(&u, &v).unwrap());
}
