//! Ninth cross-validation batch: neural-network functional ops and MultiheadAttention vs PyTorch.
//! Run: `cargo test -p oxideon-core --test crossval_emit9 -- --nocapture --ignored`.

use oxideon_core::nn::attention_ext::MultiheadAttention;
use oxideon_core::nn::functional_ext::*;
use oxideon_core::tensor::Tensor;

fn genv(n: usize, salt: f64) -> Vec<f64> {
    (0..n).map(|i| (0.37 * i as f64 + salt).sin() * 0.8 + 0.1 * (0.11 * i as f64 * salt).cos()).collect()
}

fn t(shape: &[usize], salt: f64) -> Tensor {
    Tensor::from_slice(&genv(shape.iter().product(), salt), shape).unwrap()
}

fn emit(name: &str, x: &Tensor) {
    x.to_vec::<f64>().unwrap().iter().enumerate().for_each(|(i, v)| println!("{name}_{i} {v:.12e}"));
}

#[test]
#[ignore]
fn emit_nn() {
    let q = t(&[2, 3, 5, 4], 0.3);
    let k = t(&[2, 3, 6, 4], 1.1);
    let v = t(&[2, 3, 6, 4], 2.2);
    emit("sdpa", &scaled_dot_product_attention(&q, &k, &v, None, false, None).unwrap());
    emit("sdpa_scale", &scaled_dot_product_attention(&q, &k, &v, None, false, Some(0.37)).unwrap());
    let k5 = t(&[2, 3, 5, 4], 1.4);
    let v5 = t(&[2, 3, 5, 4], 2.6);
    emit("sdpa_causal", &scaled_dot_product_attention(&q, &k5, &v5, None, true, None).unwrap());
    let mask = Tensor::from_slice(&genv(5 * 6, 3.3), &[5, 6]).unwrap();
    emit("sdpa_mask", &scaled_dot_product_attention(&q, &k, &v, Some(&mask), false, None).unwrap());
    let x = t(&[2, 6, 3, 3], 0.9);
    let w6 = Tensor::from_slice(&genv(6, 4.0), &[6]).unwrap();
    let b6 = Tensor::from_slice(&genv(6, 5.0), &[6]).unwrap();
    emit("group_norm", &group_norm(&x, 3, Some(&w6), Some(&b6), 1e-5).unwrap());
    let ln_in = t(&[4, 6], 1.7);
    emit("layer_norm", &layer_norm(&ln_in, &[6], Some(&w6), Some(&b6), 1e-5).unwrap());
    let rm = Tensor::from_slice(&genv(6, 6.0), &[6]).unwrap();
    let rv = Tensor::from_slice(&genv(6, 7.0).iter().map(|v| v.abs() + 0.3).collect::<Vec<_>>(), &[6]).unwrap();
    emit("batch_norm_eval", &batch_norm(&x, Some(&rm), Some(&rv), Some(&w6), Some(&b6), false, 0.1, 1e-5).unwrap());
    let rm2 = Tensor::from_slice(&genv(6, 6.0), &[6]).unwrap();
    let rv2 = Tensor::from_slice(&genv(6, 7.0).iter().map(|v| v.abs() + 0.3).collect::<Vec<_>>(), &[6]).unwrap();
    emit("batch_norm_train", &batch_norm(&x, Some(&rm2), Some(&rv2), Some(&w6), Some(&b6), true, 0.1, 1e-5).unwrap());
    emit("bn_running_mean", &rm2);
    emit("bn_running_var", &rv2);
    emit("lrn", &local_response_norm(&x, 3, 1e-3, 0.75, 1.5).unwrap());
    let img = t(&[1, 2, 4, 5], 0.2);
    for (name, mode, ac) in [("nearest", InterpMode::Nearest, false), ("linear", InterpMode::Linear, false), ("linear_ac", InterpMode::Linear, true), ("bicubic", InterpMode::Bicubic, false), ("bicubic_ac", InterpMode::Bicubic, true), ("area", InterpMode::Area, false)] {
        emit(&format!("interp_{name}_up"), &interpolate(&img, Some(&[7, 8]), None, mode, ac).unwrap());
        emit(&format!("interp_{name}_down"), &interpolate(&img, Some(&[3, 2]), None, mode, ac).unwrap());
    }
    emit("interp_scale", &interpolate(&img, None, Some(&[2.0, 1.5]), InterpMode::Linear, false).unwrap());
    for (name, mode) in [("reflect", PadMode::Reflect), ("replicate", PadMode::Replicate), ("circular", PadMode::Circular), ("constant", PadMode::Constant(0.7))] {
        emit(&format!("pad_{name}"), &pad_mode(&img, &[(2, 1), (1, 2)], mode).unwrap());
    }
    emit("channel_shuffle", &channel_shuffle(&t(&[1, 6, 2, 2], 0.6), 3).unwrap());
    // MultiheadAttention with deterministic weights (copied into torch by the checker)
let t = |shape: &[usize], salt: f64| t(shape, salt).to_dtype(oxideon_core::dtype::DType::F32).unwrap();
    let mut mha = MultiheadAttention::new(8, 2, 0.0, true, None, None, true).unwrap();
    let assign = |lin: &mut oxideon_core::nn::Linear, salt: f64| {
        lin.weight.assign(&t(&[8, 8], salt)).unwrap();
        if let Some(b) = &lin.bias {
            b.assign(&t(&[8], salt + 0.5)).unwrap();
        }
    };
    assign(&mut mha.q_proj, 10.0);
    assign(&mut mha.k_proj, 11.0);
    assign(&mut mha.v_proj, 12.0);
    assign(&mut mha.out_proj, 13.0);
    let qin = t(&[2, 5, 8], 20.0);
    let kin = t(&[2, 7, 8], 21.0);
    let vin = t(&[2, 7, 8], 22.0);
    let (o, w) = mha.forward_full(&qin, &kin, &vin, None, true, None, true, false).unwrap();
    emit("mha_out", &o);
    emit("mha_w", &w.unwrap());
    let kpm = Tensor::from_slice(&[0u8, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0], &[2, 7]).unwrap().to_dtype(oxideon_core::dtype::DType::Bool).unwrap();
    let (o2, w2) = mha.forward_full(&qin, &kin, &vin, Some(&kpm), true, None, false, false).unwrap();
    emit("mha_kpm_out", &o2);
    emit("mha_kpm_w", &w2.unwrap());
    let sq = t(&[2, 5, 8], 23.0);
    emit("mha_causal", &mha.forward_self(&sq, None, None, true).unwrap());
}
