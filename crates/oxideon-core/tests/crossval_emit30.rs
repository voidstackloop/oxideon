//! Cross-validation batch 30: Maxout, Snake, Swish, ConvLstmCell, SpectralNormLinear (values and gradients) and the
//! structural properties of DropPath / AlphaDropout, vs PyTorch (tools/crossval/check_batch30.py).
use oxideon_core::dtype::DType;
use oxideon_core::nn::layers_misc::ConvLstmCell;
use oxideon_core::nn::sequence_ext::{Maxout, Snake, Swish};
use oxideon_core::nn::*;
use oxideon_core::tensor::Tensor;

fn genv(n: usize, salt: f64) -> Vec<f64> {
    (0..n).map(|i| (0.37 * i as f64 + salt).sin() * 0.8 + 0.1 * (0.11 * i as f64 * salt).cos()).collect()
}

fn t(shape: &[usize], salt: f64) -> Tensor {
    Tensor::from_slice(&genv(shape.iter().product(), salt), shape).unwrap().to_dtype(DType::F32).unwrap()
}

fn emit(name: &str, x: &Tensor) {
    let v = x.to_dtype(DType::F64).unwrap().to_vec::<f64>().unwrap();
    println!("{name} {}", v.iter().map(|a| format!("{a:.9e}")).collect::<Vec<_>>().join(" "));
}

fn loss(outs: &[&Tensor]) {
    let mut total: Option<Tensor> = None;
    for (k, o) in outs.iter().enumerate() {
        let term = o.mul(&t(o.shape(), 90.0 + k as f64)).unwrap().sum().unwrap();
        total = Some(match total {
            Some(l) => l.add(&term).unwrap(),
            None => term,
        });
    }
    total.unwrap().backward().unwrap();
}

#[test]
#[ignore]
fn emit_misc_layers() {
    // Maxout (with an exact tie in the first group)
    let mut xv = genv(24, 1.0);
    xv[1] = xv[0];
    let x = Tensor::from_slice(&xv, &[3, 8]).unwrap().to_dtype(DType::F32).unwrap().requires_grad_(true);
    let out = Maxout(2).forward(&x).unwrap();
    emit("maxout_out", &out);
    loss(&[&out]);
    emit("maxout_dx", &x.grad().unwrap());

    // Snake / Swish with learnable parameters
    let x = t(&[3, 4], 2.0).requires_grad_(true);
    let sn = Snake::new(4, 0.7).unwrap();
    sn.alpha.assign(&t(&[4], 2.5).abs().unwrap().add_scalar(0.4).unwrap()).unwrap();
    let out = sn.forward(&x).unwrap();
    emit("snake_out", &out);
    loss(&[&out]);
    emit("snake_dx", &x.grad().unwrap());
    emit("snake_da", &sn.alpha.grad().unwrap());
    let x = t(&[3, 4], 3.0).requires_grad_(true);
    let sw = Swish::new(1.3).unwrap();
    let out = sw.forward(&x).unwrap();
    emit("swish_out", &out);
    loss(&[&out]);
    emit("swish_dx", &x.grad().unwrap());
    emit("swish_db", &sw.beta.grad().unwrap());

    // ConvLstmCell
    for with_state in [true, false] {
        let tag = if with_state { "state" } else { "zero" };
        let cell = ConvLstmCell::new(2, 3, 3).unwrap();
        cell.conv.weight.assign(&t(cell.conv.weight.shape(), 4.0)).unwrap();
        cell.conv.bias.as_ref().unwrap().assign(&t(&[12], 4.5)).unwrap();
        let (x, h, c) = (t(&[2, 2, 4, 5], 5.0).requires_grad_(true), t(&[2, 3, 4, 5], 6.0).requires_grad_(true), t(&[2, 3, 4, 5], 7.0).requires_grad_(true));
        let (hn, cn) = cell.step(&x, if with_state { Some((&h, &c)) } else { None }).unwrap();
        emit(&format!("clstm_{tag}_h"), &hn);
        emit(&format!("clstm_{tag}_c"), &cn);
        loss(&[&hn, &cn]);
        emit(&format!("clstm_{tag}_dx"), &x.grad().unwrap());
        emit(&format!("clstm_{tag}_dw"), &cell.conv.weight.grad().unwrap());
        emit(&format!("clstm_{tag}_db"), &cell.conv.bias.as_ref().unwrap().grad().unwrap());
        if with_state {
            emit("clstm_state_dh", &h.grad().unwrap());
            emit("clstm_state_dc", &c.grad().unwrap());
        }
    }

    // SpectralNormLinear: fixed u / v, one power iteration in training mode, then an eval-mode call
    let sl = SpectralNormLinear::new(5, 3, true, 1).unwrap();
    sl.weight.assign(&t(&[3, 5], 8.0)).unwrap();
    sl.bias.as_ref().unwrap().assign(&t(&[3], 8.5)).unwrap();
    *sl.u_vec.write().unwrap() = t(&[3, 1], 9.0).normalize(2.0, 0, 1e-12).unwrap();
    *sl.v_vec.write().unwrap() = t(&[5, 1], 9.5).normalize(2.0, 0, 1e-12).unwrap();
    let x = t(&[4, 5], 10.0).requires_grad_(true);
    let out = sl.forward(&x).unwrap();
    emit("sn_train_out", &out);
    loss(&[&out]);
    emit("sn_train_dx", &x.grad().unwrap());
    emit("sn_train_dw", &sl.weight.grad().unwrap());
    emit("sn_u", &sl.u_vec.read().unwrap());
    emit("sn_v", &sl.v_vec.read().unwrap());
    sl.eval();
    emit("sn_eval_out", &sl.forward(&t(&[4, 5], 11.0)).unwrap());

    // DropPath / AlphaDropout: eval is the identity, p = 1 gives zeros, training outputs have a fixed structure
    let x = t(&[64, 16], 12.0);
    emit("drop_in", &x);
    let dp = DropPath::new(0.3);
    emit("droppath_train", &dp.forward(&x).unwrap());
    dp.eval();
    emit("droppath_eval", &dp.forward(&x).unwrap());
    emit("droppath_p1", &DropPath::new(1.0).forward(&x).unwrap());
    let ad = AlphaDropout::new(0.25);
    emit("alpha_train", &ad.forward(&x).unwrap());
    ad.eval();
    emit("alpha_eval", &ad.forward(&x).unwrap());
    emit("alpha_p1", &AlphaDropout::new(1.0).forward(&x).unwrap());
}
