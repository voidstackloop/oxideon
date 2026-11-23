use oxideon_core::{DType, Device, Tensor};

#[test]
fn to_accepts_device_dtype_str_pair_and_tensor() {
    let x = Tensor::from_slice(&[1.5f32, 2.5, -3.0], &[3]).unwrap();
    let a = x.to(DType::F64).unwrap();
    assert_eq!(a.dtype(), DType::F64);
    let b = x.to(&Device::Cpu).unwrap();
    assert_eq!(b.to_vec::<f32>().unwrap(), vec![1.5, 2.5, -3.0]);
    let c = x.to("cuda:0").unwrap();
    assert_eq!(c.device(), Device::Cuda(0));
    let d = x.to((Device::Cuda(0), DType::F16)).unwrap();
    assert_eq!((d.device(), d.dtype()), (Device::Cuda(0), DType::F16));
    let e = x.to(&d).unwrap();
    assert_eq!((e.device(), e.dtype()), (Device::Cuda(0), DType::F16));
    assert_eq!(e.to("cpu").unwrap().to_dtype(DType::F32).unwrap().to_vec::<f32>().unwrap(), vec![1.5, 2.5, -3.0]);
    assert!(x.to("nonsense:9").is_err());
}
