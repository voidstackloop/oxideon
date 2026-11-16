//! Cross-validation batch 32: the full encoder-decoder Transformer (post-norm ReLU and pre-norm GELU, with padding and
//! causal masks) vs torch.nn.Transformer, outputs and gradients (tools/crossval/check_batch32.py).
use oxideon_core::dtype::DType;
use oxideon_core::nn::attention_ext::{FfnActivation, Transformer, TransformerConfig};
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

#[test]
#[ignore]
fn emit_transformer() {
    for (tag, activation, norm_first) in [("post_relu", FfnActivation::Relu, false), ("pre_gelu", FfnActivation::Gelu, true)] {
        let model = Transformer::new(TransformerConfig {
            d_model: 8,
            nhead: 2,
            num_encoder_layers: 2,
            num_decoder_layers: 2,
            dim_feedforward: 16,
            dropout: 0.0,
            activation,
            norm_first,
            batch_first: true,
            layer_norm_eps: 1e-5,
        })
        .unwrap();
        let params = model.named_parameters();
        for (k, (name, p)) in params.iter().enumerate() {
            // scaled down so activations stay in a sensible range
            p.assign(&t(p.shape(), 1.0 + 0.61 * k as f64).mul_scalar(0.5).unwrap()).unwrap();
            println!("param_{tag} {k} {name} {:?}", p.shape());
        }
        let src = t(&[2, 6, 8], 50.0).requires_grad_(true);
        let tgt = t(&[2, 5, 8], 51.0).requires_grad_(true);
        // no masks
        let out = model.forward_pair(&src, &tgt).unwrap();
        emit(&format!("{tag}_plain_out"), &out);
        out.mul(&t(out.shape(), 99.0)).unwrap().sum().unwrap().backward().unwrap();
        emit(&format!("{tag}_plain_dsrc"), &src.grad().unwrap());
        emit(&format!("{tag}_plain_dtgt"), &tgt.grad().unwrap());
        for k in [0usize, params.len() / 2, params.len() - 1] {
            emit(&format!("{tag}_plain_dparam{k}"), &params[k].1.grad().unwrap());
        }
        // source padding mask (also applied to the memory) + causal target
        let kpm = Tensor::from_slice(&[0u8, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1], &[2, 6]).unwrap().to_dtype(DType::Bool).unwrap();
        let (src2, tgt2) = (t(&[2, 6, 8], 52.0).requires_grad_(true), t(&[2, 5, 8], 53.0).requires_grad_(true));
        let out = model.forward_full(&src2, &tgt2, None, None, None, Some(&kpm), None, Some(&kpm), true).unwrap();
        emit(&format!("{tag}_masked_out"), &out);
        out.mul(&t(out.shape(), 99.0)).unwrap().sum().unwrap().backward().unwrap();
        emit(&format!("{tag}_masked_dsrc"), &src2.grad().unwrap());
        emit(&format!("{tag}_masked_dtgt"), &tgt2.grad().unwrap());
    }
}
