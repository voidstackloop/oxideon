//! U64 tensors: exact values above 2^53 survive arithmetic, comparison and serialization.

use oxideon_core::{DType, Tensor};

const BIG: u64 = (1 << 63) + 12345; // not representable in f64 / i64

#[test]
fn big_values_round_trip_through_ops_and_files() {
    let a = Tensor::from_slice(&[BIG, 1, u64::MAX], &[3]).unwrap();
    assert_eq!(a.dtype(), DType::U64);
    let b = Tensor::from_slice(&[1u64, 2, 1], &[3]).unwrap();
    // wrapping arithmetic like torch / numpy
    assert_eq!(a.add(&b).unwrap().to_vec::<u64>().unwrap(), vec![BIG + 1, 3, 0]);
    assert_eq!(a.sub(&b).unwrap().to_vec::<u64>().unwrap(), vec![BIG - 1, u64::MAX, u64::MAX - 1]);
    assert_eq!(a.maximum(&b).unwrap().to_vec::<u64>().unwrap(), vec![BIG, 2, u64::MAX]);
    // comparisons are done on the integers, not on lossy f64 images
    let near = Tensor::from_slice(&[BIG + 1, 1, u64::MAX], &[3]).unwrap();
    assert_eq!(a.eq_tensor(&near).unwrap().to_vec::<bool>().unwrap(), vec![false, true, true]);

    let dir = std::env::temp_dir().join("oxideon_u64_test");
    std::fs::create_dir_all(&dir).unwrap();
    oxideon_core::serialize::npy::save_npy(&a, dir.join("a.npy")).unwrap();
    let back = oxideon_core::serialize::npy::load_npy(dir.join("a.npy")).unwrap();
    assert_eq!(back.dtype(), DType::U64);
    assert_eq!(back.to_vec::<u64>().unwrap(), vec![BIG, 1, u64::MAX]);
    let m = std::collections::HashMap::from([("a".to_string(), a.clone())]);
    oxideon_core::serialize::save_safetensors(&m, dir.join("a.safetensors")).unwrap();
    let st = oxideon_core::serialize::load_safetensors(dir.join("a.safetensors")).unwrap();
    assert_eq!(st["a"].to_vec::<u64>().unwrap(), vec![BIG, 1, u64::MAX]);
    let _ = std::fs::remove_dir_all(&dir);
}
