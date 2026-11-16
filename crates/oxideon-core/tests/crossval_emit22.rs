//! Cross-validation batch 22: backward passes (input and parameter gradients) of conv / pooling / norm / recurrent
//! layers vs PyTorch, float32 modules (tools/crossval/check_batch22.py).
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

fn set_linear(l: &Linear, salt: f64) {
    l.weight.assign(&t(l.weight.shape(), salt)).unwrap();
    if let Some(b) = &l.bias {
        b.assign(&t(b.shape(), salt + 0.5)).unwrap();
    }
}

/// loss = sum(out * w) with a fixed weighting so every output element gets a distinct upstream gradient.
fn backward(out: &Tensor) {
    out.mul(&t(out.shape(), 99.0)).unwrap().sum().unwrap().backward().unwrap();
}

fn grad(x: &Tensor) -> Tensor {
    x.grad().expect("gradient")
}

#[test]
#[ignore]
fn emit_grads() {
    let x = t(&[2, 4, 7, 6], 0.3);
    let conv_case = |name: &str, c: Conv2d, wshape: &[usize], wsalt: f64, bshape: usize, bsalt: f64| {
        c.weight.assign(&t(wshape, wsalt)).unwrap();
        c.bias.as_ref().unwrap().assign(&t(&[bshape], bsalt)).unwrap();
        let xi = x.detach().requires_grad_(true);
        backward(&c.forward(&xi).unwrap());
        emit(&format!("{name}_dx"), &grad(&xi));
        emit(&format!("{name}_dw"), &grad(&c.weight));
        emit(&format!("{name}_db"), &grad(c.bias.as_ref().unwrap()));
    };
    conv_case("conv2d", Conv2d::new(4, 6, (3, 3), (1, 1), (0, 0)).unwrap(), &[6, 4, 3, 3], 1.0, 6, 1.5);
    conv_case("conv2d_sp", Conv2d::new(4, 6, (3, 2), (2, 1), (1, 2)).unwrap(), &[6, 4, 3, 2], 2.0, 6, 2.5);
    conv_case("conv2d_groups", Conv2d::with_groups(4, 6, (3, 3), (1, 1), (1, 1), 2).unwrap(), &[6, 2, 3, 3], 3.0, 6, 3.5);
    let mut c = Conv2d::new(4, 3, (3, 3), (1, 1), (2, 2)).unwrap();
    c.dilation = (2, 2);
    conv_case("conv2d_dil", c, &[3, 4, 3, 3], 4.0, 3, 4.5);

    let x1 = t(&[2, 3, 11], 5.0).requires_grad_(true);
    let c1 = Conv1d::new(3, 4, 3, 2, 1).unwrap();
    c1.weight.assign(&t(&[4, 3, 3], 6.0)).unwrap();
    c1.bias.as_ref().unwrap().assign(&t(&[4], 6.5)).unwrap();
    backward(&c1.forward(&x1).unwrap());
    emit("conv1d_dx", &grad(&x1));
    emit("conv1d_dw", &grad(&c1.weight));
    emit("conv1d_db", &grad(c1.bias.as_ref().unwrap()));

    let xt = t(&[2, 3, 4, 5], 7.0).requires_grad_(true);
    let ct = ConvTranspose2d::new(3, 2, (3, 3), (2, 2), (1, 1)).unwrap();
    ct.weight.assign(&t(&[3, 2, 3, 3], 8.0)).unwrap();
    ct.bias.as_ref().unwrap().assign(&t(&[2], 8.5)).unwrap();
    backward(&ct.forward(&xt).unwrap());
    emit("convt2d_dx", &grad(&xt));
    emit("convt2d_dw", &grad(&ct.weight));
    emit("convt2d_db", &grad(ct.bias.as_ref().unwrap()));

    // pooling backward
    let pool = |name: &str, f: &dyn Fn(&Tensor) -> Tensor| {
        let xi = x.detach().requires_grad_(true);
        backward(&f(&xi));
        emit(&format!("{name}_dx"), &grad(&xi));
    };
    pool("maxpool", &|v| MaxPool2d::new((2, 2), (2, 2)).forward(v).unwrap());
    pool("maxpool_k3s1", &|v| MaxPool2d::new((3, 3), (1, 1)).forward(v).unwrap());
    pool("avgpool", &|v| AvgPool2d::new((3, 2), (2, 2)).forward(v).unwrap());
    pool("adaptive_avg", &|v| AdaptiveAvgPool2d((3, 4)).forward(v).unwrap());

    // norms
    let xl = t(&[3, 5, 6], 10.0).requires_grad_(true);
    let ln = LayerNorm::new(6, 1e-5).unwrap();
    ln.weight.assign(&t(&[6], 9.0)).unwrap();
    ln.bias.assign(&t(&[6], 9.5)).unwrap();
    backward(&ln.forward(&xl).unwrap());
    emit("layernorm_dx", &grad(&xl));
    emit("layernorm_dw", &grad(&ln.weight));
    emit("layernorm_db", &grad(&ln.bias));

    let xr = t(&[3, 5, 6], 12.0).requires_grad_(true);
    let rn = RMSNorm::new(6, 1e-6).unwrap();
    rn.weight.assign(&t(&[6], 11.0)).unwrap();
    backward(&rn.forward(&xr).unwrap());
    emit("rmsnorm_dx", &grad(&xr));
    emit("rmsnorm_dw", &grad(&rn.weight));

    let xg = x.detach().requires_grad_(true);
    let gn = GroupNorm::new(2, 4, 1e-5).unwrap();
    gn.weight.assign(&t(&[4], 13.0)).unwrap();
    gn.bias.assign(&t(&[4], 13.5)).unwrap();
    backward(&gn.forward(&xg).unwrap());
    emit("groupnorm_dx", &grad(&xg));
    emit("groupnorm_dw", &grad(&gn.weight));
    emit("groupnorm_db", &grad(&gn.bias));

    let xb = x.detach().requires_grad_(true);
    let bn = BatchNorm2d::new(4).unwrap();
    bn.weight.assign(&t(bn.weight.shape(), 15.0)).unwrap();
    bn.bias.assign(&t(bn.bias.shape(), 15.5)).unwrap();
    backward(&bn.forward(&xb).unwrap());
    emit("batchnorm_dx", &grad(&xb));
    emit("batchnorm_dw", &grad(&bn.weight));
    emit("batchnorm_db", &grad(&bn.bias));
    emit("batchnorm_rm", &bn.running_mean);
    emit("batchnorm_rv", &bn.running_var);

    // recurrent layers
    let seq = t(&[2, 5, 3], 14.0);
    let lstm = LSTM::new(3, 4, 2).unwrap();
    for (l, cell) in lstm.cells.iter().enumerate() {
        set_linear(&cell.ih, 20.0 + 4.0 * l as f64);
        set_linear(&cell.hh, 22.0 + 4.0 * l as f64);
    }
    let si = seq.detach().requires_grad_(true);
    let (out, (hn, cn)) = lstm.forward_seq(&si, None).unwrap();
    // gradient flows through the sequence output and both final states
    out.mul(&t(out.shape(), 99.0)).unwrap().sum().unwrap().add(&hn.mul(&t(hn.shape(), 98.0)).unwrap().sum().unwrap()).unwrap().add(&cn.mul(&t(cn.shape(), 97.0)).unwrap().sum().unwrap()).unwrap().backward().unwrap();
    emit("lstm_dx", &grad(&si));
    for (l, cell) in lstm.cells.iter().enumerate() {
        emit(&format!("lstm_dwih{l}"), &grad(&cell.ih.weight));
        emit(&format!("lstm_dwhh{l}"), &grad(&cell.hh.weight));
        emit(&format!("lstm_dbih{l}"), &grad(cell.ih.bias.as_ref().unwrap()));
    }
    let gru = GRU::new(3, 4, 2).unwrap();
    for (l, cell) in gru.cells.iter().enumerate() {
        set_linear(&cell.ih, 30.0 + 4.0 * l as f64);
        set_linear(&cell.hh, 32.0 + 4.0 * l as f64);
    }
    let sg = seq.detach().requires_grad_(true);
    let (gout, _) = gru.forward_seq(&sg, None).unwrap();
    backward(&gout);
    emit("gru_dx", &grad(&sg));
    for (l, cell) in gru.cells.iter().enumerate() {
        emit(&format!("gru_dwih{l}"), &grad(&cell.ih.weight));
        emit(&format!("gru_dwhh{l}"), &grad(&cell.hh.weight));
        emit(&format!("gru_dbhh{l}"), &grad(cell.hh.bias.as_ref().unwrap()));
    }
    let rnn = RNN::new(3, 4, 1).unwrap();
    for cell in rnn.cells.iter() {
        set_linear(&cell.ih, 40.0);
        set_linear(&cell.hh, 42.0);
    }
    let sr = seq.detach().requires_grad_(true);
    let (rout, _) = rnn.forward_seq(&sr, None).unwrap();
    backward(&rout);
    emit("rnn_dx", &grad(&sr));
    emit("rnn_dwih", &grad(&rnn.cells[0].ih.weight));
    emit("rnn_dwhh", &grad(&rnn.cells[0].hh.weight));
}
