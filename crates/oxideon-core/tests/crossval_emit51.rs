//! Cross-validation batch 51: the I8 / I16 / U16 / U32 dtypes (casts, wrapping arithmetic, reductions, comparisons) vs torch
//! (tools/crossval/check_batch51.py).
use oxideon_core::dtype::DType;
use oxideon_core::tensor::Tensor;

fn p(name: &str, t: &Tensor) {
    let v = t.to_dtype(DType::F64).unwrap().to_vec::<f64>().unwrap();
    println!("{name} {}", v.iter().map(|a| format!("{a:e}")).collect::<Vec<_>>().join(" "));
}

#[test]
#[ignore]
fn emit_small_ints() {
    let src = [-200.7, -129.0, -128.0, -1.5, 0.0, 1.0, 100.9, 127.0, 128.0, 200.0, 255.0, 256.0, 1000.0, 32767.0, 32768.0, 65535.0, 65536.0, 70000.5, 3.0e9, 4294967295.0, 4294967296.0, -70000.0];
    let f = Tensor::from_slice(&src, &[src.len()]).unwrap();
    for (n, dt) in [("i8", DType::I8), ("i16", DType::I16), ("u16", DType::U16), ("u32", DType::U32)] {
        p(&format!("cast_{n}"), &f.to_dtype(dt).unwrap());
    }
    // integer to integer wrapping through the intermediate i64
    let big = Tensor::from_slice(&[-5i64, 127, 128, 255, 256, 40000, 65536, 70000, -40000, 4294967295, 4294967296, 1 << 40], &[12]).unwrap();
    for (n, dt) in [("i8", DType::I8), ("i16", DType::I16), ("u16", DType::U16), ("u32", DType::U32)] {
        p(&format!("wrap_{n}"), &big.to_dtype(dt).unwrap());
    }
    // arithmetic on i8 / i16 (wraps like torch)
    for (n, dt) in [("i8", DType::I8), ("i16", DType::I16)] {
        let a = Tensor::from_slice(&[100.0f64, -100.0, 50.0, 127.0, -128.0, 7.0, 0.0, -1.0], &[8]).unwrap().to_dtype(dt).unwrap();
        let b = Tensor::from_slice(&[100.0f64, -100.0, 90.0, 1.0, -1.0, -3.0, 5.0, -1.0], &[8]).unwrap().to_dtype(dt).unwrap();
        p(&format!("add_{n}"), &a.add(&b).unwrap());
        p(&format!("sub_{n}"), &a.sub(&b).unwrap());
        p(&format!("mul_{n}"), &a.mul(&b).unwrap());
        p(&format!("neg_{n}"), &a.neg().unwrap());
        p(&format!("abs_{n}"), &a.abs().unwrap());
        p(&format!("max_{n}"), &a.maximum(&b).unwrap());
        p(&format!("lt_{n}"), &a.lt(&b).unwrap());
        p(&format!("ge_{n}"), &a.ge(&b).unwrap());
        p(&format!("sum_{n}"), &a.sum().unwrap());
        println!("sumdtype_{n} {:?}", a.sum().unwrap().dtype());
        p(&format!("cumsum_{n}"), &a.cumsum(0).unwrap());
    }
    // u16 / u32 round trips through float and int
    let u = Tensor::from_slice(&[0.0f64, 1.0, 255.0, 256.0, 65535.0], &[5]).unwrap();
    p("u16_back_f32", &u.to_dtype(DType::U16).unwrap().to_dtype(DType::F32).unwrap());
    p("u32_back_f64", &Tensor::from_slice(&[0.0f64, 4294967295.0, 123456789.0], &[3]).unwrap().to_dtype(DType::U32).unwrap().to_dtype(DType::F64).unwrap());
}
