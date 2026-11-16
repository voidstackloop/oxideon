//! Cross-validation batch 36: fake quantization (values, straight-through gradients, rounding of exact halves,
//! saturation) and min-max observers vs torch / torch.ao.quantization (tools/crossval/check_batch36.py).
use oxideon_core::quant_aware::*;
use oxideon_core::tensor::Tensor;

fn dump(name: &str, t: &Tensor) {
    let v = t.to_dtype(oxideon_core::dtype::DType::F64).unwrap().to_vec::<f64>().unwrap();
    println!("{name} {}", v.iter().map(|x| format!("{x:.13e}")).collect::<Vec<_>>().join(" "));
}

fn data(n: usize, salt: f64, amp: f64) -> Vec<f64> {
    (0..n).map(|i| ((0.61 * i as f64 + salt).sin() + 0.3 * (0.23 * i as f64).cos()) * amp).collect()
}

#[test]
#[ignore]
fn emit_quant() {
    // fake_quantize: scale 0.25, zero point 3; inputs include exact halves of the grid and values beyond both ends
    let mut xv = data(24, 0.4, 3.0);
    xv.extend([0.125, 0.375, -0.125, -0.375, 0.625, 2.5 * 0.25, -1.5 * 0.25, 40.0, -40.0, 31.0, 31.125, -32.75, -32.875]);
    for (tag, scale, zp, range) in [("u8", 0.25, 3.0, QuantRange::unsigned(8)), ("i8", 0.25, 0.0, QuantRange::signed(8)), ("i4", 0.6, -2.0, QuantRange::signed(4))] {
        let x = Tensor::from_slice(&xv, &[xv.len()]).unwrap().requires_grad_(true);
        let q = QParams { scale: Tensor::from_slice(&[scale], &[1]).unwrap(), zero_point: Tensor::from_slice(&[zp], &[1]).unwrap() };
        let y = fake_quantize(&x, &q, range).unwrap();
        dump(&format!("fq_{tag}_out"), &y);
        let w = Tensor::from_slice(&(0..xv.len()).map(|i| 0.5 + 0.1 * i as f64).collect::<Vec<_>>(), &[xv.len()]).unwrap();
        y.mul(&w).unwrap().sum().unwrap().backward().unwrap();
        dump(&format!("fq_{tag}_dx"), &x.grad().unwrap());
    }

    // observers -> qparams
    let batches: Vec<Tensor> = (0..4).map(|k| Tensor::from_slice(&data(30, 1.0 + k as f64, 1.0 + 0.7 * k as f64), &[5, 6]).unwrap()).collect();
    for (tag, symmetric, range) in [("affine_u8", false, QuantRange::unsigned(8)), ("sym_i8", true, QuantRange::signed(8)), ("sym_u8", true, QuantRange::unsigned(8)), ("affine_i8", false, QuantRange::signed(8))] {
        let obs = MinMaxObserver::per_tensor();
        batches.iter().for_each(|b| obs.observe(b).unwrap());
        let q = obs.qparams(range, symmetric, 2).unwrap();
        dump(&format!("obs_{tag}_scale"), &q.scale);
        dump(&format!("obs_{tag}_zp"), &q.zero_point);
        let ma = MinMaxObserver::per_tensor().moving_average(0.2);
        batches.iter().for_each(|b| ma.observe(b).unwrap());
        let q = ma.qparams(range, symmetric, 2).unwrap();
        dump(&format!("ma_{tag}_scale"), &q.scale);
        dump(&format!("ma_{tag}_zp"), &q.zero_point);
    }
    // per-channel (axis 1), symmetric int8 and affine uint8
    for (tag, symmetric, range) in [("pc_sym_i8", true, QuantRange::signed(8)), ("pc_affine_u8", false, QuantRange::unsigned(8))] {
        let obs = MinMaxObserver::per_channel(1);
        batches.iter().for_each(|b| obs.observe(b).unwrap());
        let q = obs.qparams(range, symmetric, 2).unwrap();
        dump(&format!("{tag}_scale"), &q.scale);
        dump(&format!("{tag}_zp"), &q.zero_point);
        // per-channel fake quantization of the last batch
        let x = batches[3].detach().requires_grad_(true);
        let y = fake_quantize(&x, &q, range).unwrap();
        dump(&format!("{tag}_fq"), &y);
        y.sum().unwrap().backward().unwrap();
        dump(&format!("{tag}_fq_dx"), &x.grad().unwrap());
    }
}
