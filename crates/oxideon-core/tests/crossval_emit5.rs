//! Fifth cross-validation batch: the tensor special functions (value and gradient), window functions, loss
//! functions and distribution log-probabilities vs SciPy / PyTorch.
//! Run: `cargo test -p oxideon-core --test crossval_emit5 -- --nocapture --ignored`.

use oxideon_core::tensor::Tensor;

fn t(v: &[f64]) -> Tensor {
    Tensor::from_slice(v, &[v.len()]).unwrap()
}

fn emit(name: &str, v: &Tensor) {
    v.to_vec::<f64>().unwrap().iter().enumerate().for_each(|(i, x)| println!("{name}_{i} {x:.12e}"));
}

#[test]
#[ignore]
fn emit_special() {
    let pos = [0.05, 0.3, 0.9, 1.0, 1.7, 2.5, 4.2, 7.5, 15.0, 40.0];
    let unit = [-0.97, -0.6, -0.2, 0.0, 0.15, 0.5, 0.8, 0.99];
    let any = [-12.0, -6.5, -3.1, -1.2, -0.3, 0.0, 0.4, 1.3, 3.6, 7.2, 11.0];
    let (p, u, a) = (t(&pos), t(&unit), t(&any));
    emit("erfc", &a.erfc().unwrap());
    emit("erfinv", &u.erfinv().unwrap());
    emit("lgamma", &p.lgamma().unwrap());
    emit("digamma", &p.digamma().unwrap());
    emit("polygamma1", &p.polygamma(1).unwrap());
    emit("polygamma2", &p.polygamma(2).unwrap());
    emit("polygamma3", &p.polygamma(3).unwrap());
    emit("ndtr", &a.ndtr().unwrap());
    emit("log_ndtr", &a.log_ndtr().unwrap());
    emit("ndtri", &t(&[0.001, 0.02, 0.2, 0.5, 0.77, 0.95, 0.9999]).ndtri().unwrap());
    emit("entr", &t(&[0.0, 0.1, 0.5, 1.0, 2.7]).entr().unwrap());
    emit("zeta", &t(&[1.5, 2.0, 3.5, 6.0]).zeta(&t(&[1.0, 2.0, 0.5, 3.0])).unwrap());
    emit("igamma", &t(&[0.5, 1.0, 2.5, 5.0, 12.0]).igamma(&t(&[0.3, 1.0, 3.0, 4.0, 15.0])).unwrap());
    emit("igammac", &t(&[0.5, 1.0, 2.5, 5.0, 12.0]).igammac(&t(&[0.3, 1.0, 3.0, 4.0, 15.0])).unwrap());
    emit("i0", &a.i0().unwrap());
    emit("i0e", &a.i0e().unwrap());
    emit("i1", &a.i1().unwrap());
    emit("i1e", &a.i1e().unwrap());
    emit("j0", &a.bessel_j0().unwrap());
    emit("j1", &a.bessel_j1().unwrap());
    emit("y0", &p.bessel_y0().unwrap());
    emit("y1", &p.bessel_y1().unwrap());
    emit("mvlgamma", &t(&[2.5, 3.3, 6.0]).mvlgamma(3).unwrap());
    emit("xlog1py", &t(&[0.0, 1.5, 2.0, 3.0]).xlog1py(&t(&[0.5, -0.4, 2.0, 9.0])).unwrap());
    // gradients of a few of them (compared with torch autograd)
    for (name, f) in [("g_lgamma", 0usize), ("g_digamma", 1), ("g_erfinv", 2), ("g_i0", 3), ("g_log_ndtr", 4)] {
        let x = match f {
            2 => t(&[-0.8, -0.1, 0.4, 0.9]),
            4 => t(&[-6.0, -1.0, 0.5, 3.0]),
            _ => t(&[0.4, 1.1, 2.3, 5.0]),
        }
        .requires_grad_(true);
        let y = match f {
            0 => x.lgamma(),
            1 => x.digamma(),
            2 => x.erfinv(),
            3 => x.i0(),
            _ => x.log_ndtr(),
        }
        .unwrap();
        y.sum().unwrap().backward().unwrap();
        emit(name, &x.grad().unwrap());
    }
}
