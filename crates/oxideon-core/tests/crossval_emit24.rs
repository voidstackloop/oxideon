//! Cross-validation batch 24: gradients of elementwise / binary / reduction tensor ops vs PyTorch
//! (tools/crossval/check_batch24.py).
use oxideon_core::tensor::Tensor;

fn dump(name: &str, v: &[f64]) {
    println!("{name} {}", v.iter().map(|x| format!("{x:.14e}")).collect::<Vec<_>>().join(" "));
}

const UNIT: [f64; 8] = [0.15, 0.3, 0.42, 0.55, 0.61, 0.7, 0.82, 0.9];
const WIDE: [f64; 8] = [-2.7, -1.3, -0.45, -0.1, 0.2, 0.9, 1.6, 2.8];
const POS: [f64; 8] = [0.3, 0.7, 1.1, 1.9, 2.6, 3.4, 4.5, 6.1];
const GT1: [f64; 8] = [1.2, 1.5, 2.0, 2.6, 3.1, 4.0, 5.5, 7.0];

fn un(name: &str, x: &[f64], f: &dyn Fn(&Tensor) -> Tensor) {
    let t = Tensor::from_slice(x, &[x.len()]).unwrap().requires_grad_(true);
    let y = f(&t);
    let w = Tensor::from_slice(&(0..x.len()).map(|i| 0.5 + 0.37 * i as f64).collect::<Vec<_>>(), &[x.len()]).unwrap();
    y.mul(&w).unwrap().sum().unwrap().backward().unwrap();
    dump(&format!("{name}_v"), &y.to_vec::<f64>().unwrap());
    dump(&format!("{name}_g"), &t.grad().unwrap().to_vec::<f64>().unwrap());
}

fn bin(name: &str, a: &[f64], b: &[f64], f: &dyn Fn(&Tensor, &Tensor) -> Tensor) {
    let ta = Tensor::from_slice(a, &[a.len()]).unwrap().requires_grad_(true);
    let tb = Tensor::from_slice(b, &[b.len()]).unwrap().requires_grad_(true);
    let y = f(&ta, &tb);
    let w = Tensor::from_slice(&(0..a.len()).map(|i| 0.5 + 0.37 * i as f64).collect::<Vec<_>>(), &[a.len()]).unwrap();
    y.mul(&w).unwrap().sum().unwrap().backward().unwrap();
    dump(&format!("{name}_v"), &y.to_vec::<f64>().unwrap());
    dump(&format!("{name}_ga"), &ta.grad().unwrap().to_vec::<f64>().unwrap());
    dump(&format!("{name}_gb"), &tb.grad().unwrap().to_vec::<f64>().unwrap());
}

#[test]
#[ignore]
fn emit_unary() {
    un("exp", &WIDE, &|x| x.exp().unwrap());
    un("ln", &POS, &|x| x.ln().unwrap());
    un("sqrt", &POS, &|x| x.sqrt().unwrap());
    un("rsqrt", &POS, &|x| x.rsqrt().unwrap());
    un("sin", &WIDE, &|x| x.sin().unwrap());
    un("cos", &WIDE, &|x| x.cos().unwrap());
    un("tan", &WIDE, &|x| x.tan().unwrap());
    un("asin", &UNIT, &|x| x.asin().unwrap());
    un("acos", &UNIT, &|x| x.acos().unwrap());
    un("atan", &WIDE, &|x| x.atan().unwrap());
    un("sinh", &WIDE, &|x| x.sinh().unwrap());
    un("cosh", &WIDE, &|x| x.cosh().unwrap());
    un("tanh", &WIDE, &|x| x.tanh().unwrap());
    un("asinh", &WIDE, &|x| x.asinh().unwrap());
    un("acosh", &GT1, &|x| x.acosh().unwrap());
    un("atanh", &UNIT, &|x| x.atanh().unwrap());
    un("erf", &WIDE, &|x| x.erf().unwrap());
    un("erfc", &WIDE, &|x| x.erfc().unwrap());
    un("erfinv", &UNIT, &|x| x.erfinv().unwrap());
    un("lgamma", &POS, &|x| x.lgamma().unwrap());
    un("digamma", &POS, &|x| x.digamma().unwrap());
    un("expm1", &WIDE, &|x| x.expm1().unwrap());
    un("log1p", &POS, &|x| x.log1p().unwrap());
    un("log2", &POS, &|x| x.log2().unwrap());
    un("log10", &POS, &|x| x.log10().unwrap());
    un("abs", &WIDE, &|x| x.abs().unwrap());
    un("reciprocal", &WIDE, &|x| x.recip().unwrap());
    un("square", &WIDE, &|x| x.square().unwrap());
    un("exp2", &WIDE, &|x| x.exp2().unwrap());
    un("sigmoid", &WIDE, &|x| x.sigmoid().unwrap());
    un("logit", &UNIT, &|x| x.logit(None).unwrap());
    un("sinc", &WIDE, &|x| x.sinc().unwrap());
    un("pow_scalar", &POS, &|x| x.powf(2.5).unwrap());
    un("pow_neg", &POS, &|x| x.powf(-1.3).unwrap());
    un("cumsum", &WIDE, &|x| x.cumsum(0).unwrap());
    un("cumprod", &POS, &|x| x.cumprod(0).unwrap());
    un("logcumsumexp", &WIDE, &|x| x.logcumsumexp(0).unwrap());
    un("logsumexp", &WIDE, &|x| x.logsumexp(0, true).unwrap());
    un("softmax", &WIDE, &|x| x.softmax(0).unwrap());
    un("log_softmax", &WIDE, &|x| x.log_softmax(0).unwrap());
    un("prod", &POS, &|x| x.prod().unwrap().reshape(&[1]).unwrap());
    un("var", &WIDE, &|x| x.var_axis(0, true, false).unwrap().reshape(&[1]).unwrap());
    un("norm2", &WIDE, &|x| x.norm_p(2.0, &[0], false).unwrap().reshape(&[1]).unwrap());
    un("norm3", &WIDE, &|x| x.norm_p(3.0, &[0], false).unwrap().reshape(&[1]).unwrap());
    un("norm1", &WIDE, &|x| x.norm_p(1.0, &[0], false).unwrap().reshape(&[1]).unwrap());
    un("amax", &WIDE, &|x| x.amax(&[0], false).unwrap().reshape(&[1]).unwrap());
    un("flip", &WIDE, &|x| x.flip(&[0]).unwrap());
    un("roll", &WIDE, &|x| x.roll(3, 0).unwrap());
}

#[test]
#[ignore]
fn emit_binary() {
    let a = [0.3, 1.4, -0.7, 2.2, -1.5, 0.8, 3.1, -2.4];
    let b = [1.1, -0.6, 2.3, 0.4, -1.9, 1.7, 0.9, 2.6];
    let ap = [0.3, 1.4, 0.7, 2.2, 1.5, 0.8, 3.1, 2.4];
    bin("add", &a, &b, &|x, y| x.add(y).unwrap());
    bin("sub", &a, &b, &|x, y| x.sub(y).unwrap());
    bin("mul", &a, &b, &|x, y| x.mul(y).unwrap());
    bin("div", &a, &b, &|x, y| x.div(y).unwrap());
    bin("pow", &ap, &b, &|x, y| x.pow(y).unwrap());
    bin("atan2", &a, &b, &|x, y| x.atan2(y).unwrap());
    bin("hypot", &a, &b, &|x, y| x.hypot(y).unwrap());
    bin("maximum", &a, &b, &|x, y| x.maximum(y).unwrap());
    bin("minimum", &a, &b, &|x, y| x.minimum(y).unwrap());
    bin("logaddexp", &a, &b, &|x, y| x.logaddexp(y).unwrap());
    bin("logaddexp2", &a, &b, &|x, y| x.logaddexp2(y).unwrap());
    bin("fmod", &a, &b, &|x, y| x.fmod(y).unwrap());
    bin("remainder", &a, &b, &|x, y| x.remainder(y).unwrap());
    bin("copysign", &a, &b, &|x, y| x.copysign(y).unwrap());
    bin("xlogy", &ap, &b.map(f64::abs), &|x, y| x.xlogy(y).unwrap());
    bin("lerp", &a, &b, &|x, y| x.lerp(y, 0.3).unwrap());
}
