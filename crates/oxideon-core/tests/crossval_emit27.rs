//! Cross-validation batch 27: MultiheadAttention backward (inputs and projection weights) and embedding / linear
//! gradients vs PyTorch, float32 modules (tools/crossval/check_batch27.py).
use oxideon_core::dtype::DType;
use oxideon_core::nn::attention_ext::MultiheadAttention;
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

fn backward(out: &Tensor) {
    out.mul(&t(out.shape(), 99.0)).unwrap().sum().unwrap().backward().unwrap();
}

fn build() -> MultiheadAttention {
    let mut mha = MultiheadAttention::new(8, 2, 0.0, true, None, None, true).unwrap();
    let assign = |lin: &mut Linear, salt: f64| {
        lin.weight.assign(&t(&[8, 8], salt)).unwrap();
        lin.bias.as_ref().unwrap().assign(&t(&[8], salt + 0.5)).unwrap();
    };
    assign(&mut mha.q_proj, 10.0);
    assign(&mut mha.k_proj, 11.0);
    assign(&mut mha.v_proj, 12.0);
    assign(&mut mha.out_proj, 13.0);
    mha
}

fn dump_params(tag: &str, mha: &MultiheadAttention) {
    for (n, l) in [("q", &mha.q_proj), ("k", &mha.k_proj), ("v", &mha.v_proj), ("o", &mha.out_proj)] {
        emit(&format!("{tag}_dw{n}"), &l.weight.grad().unwrap());
        emit(&format!("{tag}_db{n}"), &l.bias.as_ref().unwrap().grad().unwrap());
    }
}

#[test]
#[ignore]
fn emit_attention_grads() {
    // cross attention, no masks
    let mha = build();
    let (q, k, v) = (t(&[2, 5, 8], 20.0).requires_grad_(true), t(&[2, 7, 8], 21.0).requires_grad_(true), t(&[2, 7, 8], 22.0).requires_grad_(true));
    let (o, _) = mha.forward_full(&q, &k, &v, None, false, None, true, false).unwrap();
    backward(&o);
    emit("x_dq", &q.grad().unwrap());
    emit("x_dk", &k.grad().unwrap());
    emit("x_dv", &v.grad().unwrap());
    dump_params("x", &mha);

    // key padding mask
    let mha = build();
    let (q, k, v) = (t(&[2, 5, 8], 20.0).requires_grad_(true), t(&[2, 7, 8], 21.0).requires_grad_(true), t(&[2, 7, 8], 22.0).requires_grad_(true));
    let kpm = Tensor::from_slice(&[0u8, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0], &[2, 7]).unwrap().to_dtype(DType::Bool).unwrap();
    let (o, _) = mha.forward_full(&q, &k, &v, Some(&kpm), false, None, true, false).unwrap();
    backward(&o);
    emit("kpm_dq", &q.grad().unwrap());
    emit("kpm_dk", &k.grad().unwrap());
    emit("kpm_dv", &v.grad().unwrap());
    dump_params("kpm", &mha);

    // causal self attention
    let mha = build();
    let s = t(&[2, 5, 8], 23.0).requires_grad_(true);
    backward(&mha.forward_self(&s, None, None, true).unwrap());
    emit("causal_dx", &s.grad().unwrap());
    dump_params("causal", &mha);

    // Linear on a 3-D input
    let lin = Linear::new(8, 3).unwrap();
    lin.weight.assign(&t(&[3, 8], 30.0)).unwrap();
    lin.bias.as_ref().unwrap().assign(&t(&[3], 30.5)).unwrap();
    let x = t(&[2, 5, 8], 31.0).requires_grad_(true);
    backward(&lin.forward(&x).unwrap());
    emit("lin_dx", &x.grad().unwrap());
    emit("lin_dw", &lin.weight.grad().unwrap());
    emit("lin_db", &lin.bias.as_ref().unwrap().grad().unwrap());

    // Embedding with repeated indices
    let emb = Embedding::new(7, 4).unwrap();
    emb.weight.assign(&t(&[7, 4], 40.0)).unwrap();
    let idx = Tensor::from_slice(&[1i64, 3, 3, 0, 6, 1, 1, 2], &[2, 4]).unwrap();
    let e = emb.forward(&idx).unwrap();
    emit("emb_out", &e);
    backward(&e);
    emit("emb_dw", &emb.weight.grad().unwrap());
}
