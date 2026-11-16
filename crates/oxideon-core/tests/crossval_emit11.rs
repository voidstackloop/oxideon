//! Eleventh cross-validation batch: convolution, pooling and recurrent layers vs PyTorch (float32 modules).
//! Run: `cargo test -p oxideon-core --test crossval_emit11 -- --nocapture --ignored`.

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
    x.to_dtype(DType::F64).unwrap().to_vec::<f64>().unwrap().iter().enumerate().for_each(|(i, v)| println!("{name}_{i} {v:.9e}"));
}

fn set_linear(l: &Linear, salt: f64) {
    l.weight.assign(&t(l.weight.shape(), salt)).unwrap();
    if let Some(b) = &l.bias {
        b.assign(&t(b.shape(), salt + 0.5)).unwrap();
    }
}

#[test]
#[ignore]
fn emit_layers() {
    let x = t(&[2, 4, 7, 6], 0.3);
    // Conv2d: plain, strided+padded, grouped, dilated
    let c = Conv2d::new(4, 6, (3, 3), (1, 1), (0, 0)).unwrap();
    c.weight.assign(&t(&[6, 4, 3, 3], 1.0)).unwrap();
    c.bias.as_ref().unwrap().assign(&t(&[6], 1.5)).unwrap();
    emit("conv2d", &c.forward(&x).unwrap());
    let c = Conv2d::new(4, 6, (3, 2), (2, 1), (1, 2)).unwrap();
    c.weight.assign(&t(&[6, 4, 3, 2], 2.0)).unwrap();
    c.bias.as_ref().unwrap().assign(&t(&[6], 2.5)).unwrap();
    emit("conv2d_sp", &c.forward(&x).unwrap());
    let c = Conv2d::with_groups(4, 6, (3, 3), (1, 1), (1, 1), 2).unwrap();
    c.weight.assign(&t(&[6, 2, 3, 3], 3.0)).unwrap();
    c.bias.as_ref().unwrap().assign(&t(&[6], 3.5)).unwrap();
    emit("conv2d_groups", &c.forward(&x).unwrap());
    let mut c = Conv2d::new(4, 3, (3, 3), (1, 1), (2, 2)).unwrap();
    c.dilation = (2, 2);
    c.weight.assign(&t(&[3, 4, 3, 3], 4.0)).unwrap();
    c.bias.as_ref().unwrap().assign(&t(&[3], 4.5)).unwrap();
    emit("conv2d_dil", &c.forward(&x).unwrap());
    // Conv1d
    let x1 = t(&[2, 3, 11], 5.0);
    let c1 = Conv1d::new(3, 4, 3, 2, 1).unwrap();
    c1.weight.assign(&t(&[4, 3, 3], 6.0)).unwrap();
    c1.bias.as_ref().unwrap().assign(&t(&[4], 6.5)).unwrap();
    emit("conv1d", &c1.forward(&x1).unwrap());
    // ConvTranspose2d
    let xt = t(&[2, 3, 4, 5], 7.0);
    let ct = ConvTranspose2d::new(3, 2, (3, 3), (2, 2), (1, 1)).unwrap();
    ct.weight.assign(&t(&[3, 2, 3, 3], 8.0)).unwrap();
    ct.bias.as_ref().unwrap().assign(&t(&[2], 8.5)).unwrap();
    emit("convt2d", &ct.forward(&xt).unwrap());
    // pooling
    emit("maxpool", &MaxPool2d::new((2, 2), (2, 2)).forward(&x).unwrap());
    emit("maxpool_k3s1", &MaxPool2d::new((3, 3), (1, 1)).forward(&x).unwrap());
    emit("avgpool", &AvgPool2d::new((3, 2), (2, 2)).forward(&x).unwrap());
    emit("adaptive_avg", &AdaptiveAvgPool2d((3, 4)).forward(&x).unwrap());
    // norms
    let ln = LayerNorm::new(6, 1e-5).unwrap();
    ln.weight.assign(&t(&[6], 9.0)).unwrap();
    ln.bias.assign(&t(&[6], 9.5)).unwrap();
    emit("layernorm", &ln.forward(&t(&[3, 5, 6], 10.0)).unwrap());
    let rn = RMSNorm::new(6, 1e-6).unwrap();
    rn.weight.assign(&t(&[6], 11.0)).unwrap();
    emit("rmsnorm", &rn.forward(&t(&[3, 5, 6], 12.0)).unwrap());
    let gn = GroupNorm::new(2, 4, 1e-5).unwrap();
    gn.weight.assign(&t(&[4], 13.0)).unwrap();
    gn.bias.assign(&t(&[4], 13.5)).unwrap();
    emit("groupnorm", &gn.forward(&x).unwrap());
    // recurrent layers (batch_first [B, T, F])
    let seq = t(&[2, 5, 3], 14.0);
    let lstm = LSTM::new(3, 4, 2).unwrap();
    for (l, cell) in lstm.cells.iter().enumerate() {
        set_linear(&cell.ih, 20.0 + 4.0 * l as f64);
        set_linear(&cell.hh, 22.0 + 4.0 * l as f64);
    }
    let (out, (hn, cn)) = lstm.forward_seq(&seq, None).unwrap();
    emit("lstm_out", &out);
    emit("lstm_h", &hn);
    emit("lstm_c", &cn);
    let gru = GRU::new(3, 4, 2).unwrap();
    for (l, cell) in gru.cells.iter().enumerate() {
        set_linear(&cell.ih, 30.0 + 4.0 * l as f64);
        set_linear(&cell.hh, 32.0 + 4.0 * l as f64);
    }
    let (gout, ghn) = gru.forward_seq(&seq, None).unwrap();
    emit("gru_out", &gout);
    emit("gru_h", &ghn);
    let rnn = RNN::new(3, 4, 1).unwrap();
    for cell in rnn.cells.iter() {
        set_linear(&cell.ih, 40.0);
        set_linear(&cell.hh, 42.0);
    }
    let (rout, rhn) = rnn.forward_seq(&seq, None).unwrap();
    emit("rnn_out", &rout);
    emit("rnn_h", &rhn);
}
