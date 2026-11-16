//! Cross-validation batch 31: single-step RNN/GRU/LSTM cells and Upsample2d, values and gradients vs PyTorch (f32)
//! (tools/crossval/check_batch31.py).
use oxideon_core::dtype::DType;
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

fn set(l: &Linear, salt: f64) {
    l.weight.assign(&t(l.weight.shape(), salt)).unwrap();
    l.bias.as_ref().unwrap().assign(&t(l.bias.as_ref().unwrap().shape(), salt + 0.5)).unwrap();
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
fn emit_cells() {
    for with_state in [true, false] {
        let tag = if with_state { "state" } else { "zero" };
        // RNN cell
        let c = RNNCell::new(3, 4).unwrap();
        set(&c.ih, 1.0);
        set(&c.hh, 2.0);
        let (x, h) = (t(&[5, 3], 3.0).requires_grad_(true), t(&[5, 4], 4.0).requires_grad_(true));
        let out = c.step(&x, if with_state { Some(&h) } else { None }).unwrap();
        emit(&format!("rnn_{tag}_out"), &out);
        loss(&[&out]);
        emit(&format!("rnn_{tag}_dx"), &x.grad().unwrap());
        emit(&format!("rnn_{tag}_dwih"), &c.ih.weight.grad().unwrap());
        emit(&format!("rnn_{tag}_dwhh"), &c.hh.weight.grad().unwrap());
        // GRU cell
        let c = GRUCell::new(3, 4).unwrap();
        set(&c.ih, 5.0);
        set(&c.hh, 6.0);
        let (x, h) = (t(&[5, 3], 7.0).requires_grad_(true), t(&[5, 4], 8.0).requires_grad_(true));
        let out = c.step(&x, if with_state { Some(&h) } else { None }).unwrap();
        emit(&format!("gru_{tag}_out"), &out);
        loss(&[&out]);
        emit(&format!("gru_{tag}_dx"), &x.grad().unwrap());
        emit(&format!("gru_{tag}_dwih"), &c.ih.weight.grad().unwrap());
        emit(&format!("gru_{tag}_dbhh"), &c.hh.bias.as_ref().unwrap().grad().unwrap());
        // LSTM cell
        let c = LSTMCell::new(3, 4).unwrap();
        set(&c.ih, 9.0);
        set(&c.hh, 10.0);
        let (x, h, cc) = (t(&[5, 3], 11.0).requires_grad_(true), t(&[5, 4], 12.0).requires_grad_(true), t(&[5, 4], 13.0).requires_grad_(true));
        let (ho, co) = c.step(&x, if with_state { Some((&h, &cc)) } else { None }).unwrap();
        emit(&format!("lstm_{tag}_h"), &ho);
        emit(&format!("lstm_{tag}_c"), &co);
        loss(&[&ho, &co]);
        emit(&format!("lstm_{tag}_dx"), &x.grad().unwrap());
        emit(&format!("lstm_{tag}_dwih"), &c.ih.weight.grad().unwrap());
        emit(&format!("lstm_{tag}_dwhh"), &c.hh.weight.grad().unwrap());
        if with_state {
            emit("lstm_state_dh", &h.grad().unwrap());
            emit("lstm_state_dc", &cc.grad().unwrap());
        }
    }
    // Upsample2d
    let x0 = t(&[2, 3, 4, 5], 14.0);
    for (name, up) in [
        ("up_nearest", Upsample2d::nearest((7, 9))),
        ("up_nearest_down", Upsample2d::nearest((2, 3))),
        ("up_bilinear_f", Upsample2d::bilinear((7, 9), false)),
        ("up_bilinear_t", Upsample2d::bilinear((6, 8), true)),
        ("up_bilinear_down", Upsample2d::bilinear((2, 3), false)),
    ] {
        let x = x0.detach().requires_grad_(true);
        let out = up.forward(&x).unwrap();
        emit(&format!("{name}_out"), &out);
        loss(&[&out]);
        emit(&format!("{name}_dx"), &x.grad().unwrap());
    }
}
