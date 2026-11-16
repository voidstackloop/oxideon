//! Cross-validation batch 37: elementwise ops at edge values (halves, signed zeros, infinities, NaN, subnormals,
//! zero / negative divisors) vs PyTorch (tools/crossval/check_batch37.py).
use oxideon_core::dtype::DType;
use oxideon_core::tensor::Tensor;

fn dump(name: &str, t: &Tensor) {
    let v = t.to_dtype(DType::F64).unwrap().to_vec::<f64>().unwrap();
    // {:e} prints NaN / inf as "NaN" / "inf"; the sign of zero is kept
    println!("{name} {}", v.iter().map(|x| format!("{x:e}")).collect::<Vec<_>>().join(" "));
}

fn t(v: &[f64]) -> Tensor {
    Tensor::from_slice(v, &[v.len()]).unwrap()
}

#[test]
#[ignore]
fn emit_edges() {
    let (inf, nan) = (f64::INFINITY, f64::NAN);
    let xs = [-2.5, -1.5, -0.5, -0.0, 0.0, 0.5, 1.5, 2.5, 3.7, -3.7, 0.49999999999999994, 4503599627370497.5, inf, -inf, nan, 1e-320, -1e-320, 1e308];
    let x = t(&xs);
    dump("round", &x.round().unwrap());
    dump("floor", &x.floor().unwrap());
    dump("ceil", &x.ceil().unwrap());
    dump("trunc", &x.trunc().unwrap());
    dump("frac", &x.frac().unwrap());
    dump("sign", &x.sign().unwrap());
    dump("signbit", &x.signbit().unwrap());
    dump("abs", &x.abs().unwrap());
    dump("neg", &x.neg().unwrap());
    dump("recip", &x.recip().unwrap());
    dump("sqrt", &x.sqrt().unwrap());
    dump("exp", &x.exp().unwrap());
    dump("ln", &x.ln().unwrap());
    dump("log1p", &x.log1p().unwrap());
    dump("expm1", &x.expm1().unwrap());
    dump("tanh", &x.tanh().unwrap());
    dump("sigmoid", &x.sigmoid().unwrap());
    dump("relu", &x.relu().unwrap());
    dump("isfinite", &x.isfinite().unwrap());
    dump("isinf", &x.isinf().unwrap());
    dump("isnan", &x.isnan().unwrap());
    dump("nan_to_num", &x.nan_to_num(7.0, 1e9, -1e9).unwrap());
    dump("heaviside", &x.heaviside(&t(&[0.25])).unwrap());
    dump("clamp", &x.clamp(-1.0, 2.0).unwrap());
    dump("square", &x.square().unwrap());
    dump("count_nonzero", &x.count_nonzero(None, false).unwrap().reshape(&[1]).unwrap());

    // binary ops over a grid of numerators / denominators (including zero and negative divisors, infinities)
    let nums = [-7.5, -3.0, -0.0, 0.0, 2.0, 5.5, 7.0, inf, -inf, nan];
    let dens = [-2.0, -0.5, 0.0, -0.0, 0.3, 2.0, inf, nan];
    let (mut a, mut b) = (vec![], vec![]);
    for &p in &nums {
        for &q in &dens {
            a.push(p);
            b.push(q);
        }
    }
    let (ta, tb) = (t(&a), t(&b));
    dump("div", &ta.div(&tb).unwrap());
    dump("floor_divide", &ta.floor_divide(&tb).unwrap());
    dump("fmod", &ta.fmod(&tb).unwrap());
    dump("remainder", &ta.remainder(&tb).unwrap());
    dump("pow", &ta.pow(&tb).unwrap());
    dump("atan2", &ta.atan2(&tb).unwrap());
    dump("maximum", &ta.maximum(&tb).unwrap());
    dump("minimum", &ta.minimum(&tb).unwrap());
    dump("copysign", &ta.copysign(&tb).unwrap());
    dump("hypot", &ta.hypot(&tb).unwrap());
    dump("eq", &ta.eq_tensor(&tb).unwrap());
    dump("lt", &ta.lt(&tb).unwrap());
    dump("le", &ta.le(&tb).unwrap());
    dump("gt", &ta.gt(&tb).unwrap());
    dump("ge", &ta.ge(&tb).unwrap());

    // integer arithmetic semantics
    let ia = Tensor::from_slice(&[-7i64, -7, 7, 7, -8, 8, 0, 5, -5, 9], &[10]).unwrap();
    let ib = Tensor::from_slice(&[2i64, -2, 2, -2, 3, -3, 5, 5, 5, 4], &[10]).unwrap();
    dump("i_floor_divide", &ia.floor_divide(&ib).unwrap());
    dump("i_remainder", &ia.remainder(&ib).unwrap());
    dump("i_fmod", &ia.fmod(&ib).unwrap());
    dump("i_and", &ia.bitwise_and(&ib).unwrap());
    dump("i_xor", &ia.bitwise_xor(&ib).unwrap());
    // membership / reductions
    dump("isin", &ia.isin(&Tensor::from_slice(&[7i64, 0, 9], &[3]).unwrap(), false).unwrap());
    dump("isin_inv", &ia.isin(&Tensor::from_slice(&[7i64, 0, 9], &[3]).unwrap(), true).unwrap());
    dump("nonzero", &ia.reshape(&[2, 5]).unwrap().nonzero().unwrap());
    dump("count_nz_axis", &ia.reshape(&[2, 5]).unwrap().count_nonzero(Some(1), false).unwrap());
}
