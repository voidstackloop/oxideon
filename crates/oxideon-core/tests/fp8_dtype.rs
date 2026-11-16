//! 8-bit float tensors: casts, arithmetic (computed in f32, stored rounded), safetensors round trip.
//! `fp8_emit` prints every f32 -> fp8 code for a value sweep; tools/crossval/check_fp8.py compares with torch.

use oxideon_core::dtype::{f8e4m3, f8e5m2};
use oxideon_core::{DType, Tensor};

fn sweep() -> Vec<f32> {
    let mut v = vec![0.0f32, -0.0, 1.0, -1.0, 448.0, 449.0, 464.0, 465.0, 480.0, 1e9, -1e9, 57344.0, 61439.0, 61440.0, 65000.0, 1e-3, 1.9e-3, 2.0e-3, 7.6e-6, 1.5e-5, 3e-5];
    for i in 0..400 {
        let x = ((i as f32) * 0.731).sin() * 10f32.powf(((i % 13) as f32) - 6.0);
        v.push(x);
        v.push(x * 1.0625);
    }
    v
}

#[test]
fn casts_arithmetic_and_safetensors_round_trip() {
    let x = Tensor::from_slice(&[0.3f32, 1.0625, -2.7, 500.0], &[4]).unwrap();
    let q = x.to_dtype(DType::F8E4M3).unwrap();
    assert_eq!(q.dtype(), DType::F8E4M3);
    let back = q.to_dtype(DType::F32).unwrap().to_vec::<f32>().unwrap();
    assert_eq!(back, vec![0.3125, 1.0, -2.75, 448.0]); // E4M3FN has no infinity: overflow saturates, like torch
    let r = x.to_dtype(DType::F8E5M2).unwrap();
    let sum = r.add(&r).unwrap();
    assert_eq!(sum.dtype(), DType::F8E5M2);
    assert_eq!(sum.to_dtype(DType::F32).unwrap().to_vec::<f32>().unwrap(), vec![0.625, 2.0, -5.0, 1024.0]);

    let m = std::collections::HashMap::from([("a".to_string(), q.clone()), ("b".to_string(), r.clone())]);
    let bytes = oxideon_core::serialize::serialize_safetensors(&m).unwrap();
    assert!(String::from_utf8_lossy(&bytes).contains("F8_E4M3"));
    let l = oxideon_core::serialize::deserialize_safetensors(&bytes).unwrap();
    assert_eq!(l["a"].to_vec::<f8e4m3>().unwrap(), q.to_vec::<f8e4m3>().unwrap());
    assert_eq!(l["b"].to_vec::<f8e5m2>().unwrap(), r.to_vec::<f8e5m2>().unwrap());
}

#[test]
#[ignore]
fn fp8_emit() {
    for v in sweep() {
        println!("FP8 {} {} {}", v.to_bits(), f8e4m3::from_f32(v).to_bits(), f8e5m2::from_f32(v).to_bits());
    }
}
