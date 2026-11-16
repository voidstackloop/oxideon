//! Cross-validation batch 52: function transforms vs torch.func (tools/crossval/check_batch52.py).
use oxideon_core::functional::{hessian, hvp, jvp, vjp};
use oxideon_core::nn::{Linear, Module};
use oxideon_core::tensor::Tensor;
use oxideon_core::transforms::{jacfwd, jacrev, per_sample_grads, value_and_grad};

fn genv(n: usize, salt: f64) -> Vec<f64> {
    (0..n).map(|i| (0.37 * i as f64 + salt).sin() * 1.3 + 0.1 * (0.11 * i as f64 * salt).cos()).collect()
}
fn t(shape: &[usize], salt: f64) -> Tensor {
    Tensor::from_slice(&genv(shape.iter().product(), salt), shape).unwrap()
}
fn p(name: &str, x: &Tensor) {
    let v = x.to_vec::<f64>().unwrap();
    println!("{name} {}", v.iter().map(|a| format!("{a:e}")).collect::<Vec<_>>().join(" "));
}

#[test]
#[ignore]
fn emit_transforms() {
    let (x, y) = (t(&[4], 0.4), t(&[4], 1.1));
    let f = |a: &[Tensor]| a[0].sin()?.mul(&a[1])?.sum()?.add(&a[0].square()?.mean()?.mul(&a[1].exp()?.mean()?)?);
    let (val, g) = value_and_grad(f, &[x.clone(), y.clone()], &[0, 1]).unwrap();
    p("value", &val);
    p("grad_x", &g[0]);
    p("grad_y", &g[1]);
    let w = t(&[3, 4], 2.0);
    let h = |a: &[Tensor]| a[0].matmul(&w.t().unwrap())?.tanh()?.mul(&a[0].sum_axes(&[1], true)?);
    let xm = t(&[2, 4], 0.9);
    let jr = jacrev(h, std::slice::from_ref(&xm), &[0]).unwrap().remove(0);
    p("jacrev", &jr);
    let jf = jacfwd(|a| a.matmul(&w.t().unwrap())?.tanh()?.mul(&a.sum_axes(&[1], true)?), &xm).unwrap();
    p("jacfwd", &jf);
    let scalar = |a: &Tensor| a.sin()?.mul(a)?.sum()?.add(&a.pow(&a.square()?.add_scalar(1.0)?)?.sum()?);
    p("hessian", &hessian(scalar, &x).unwrap());
    let v = t(&[4], 3.3);
    let (_, hv) = hvp(scalar, &x, &v).unwrap();
    p("hvp", &hv);
    let vec_f = |a: &Tensor| a.matmul(&w.t().unwrap())?.tanh();
    let xv = t(&[2, 4], 0.9);
    let (_, jv) = jvp(vec_f, &xv, &t(&[2, 4], 1.5)).unwrap();
    p("jvp", &jv);
    let (_, vf) = vjp(vec_f, &xv).unwrap();
    p("vjp", &vf(&t(&[2, 3], 0.2)).unwrap());
    // per-sample gradients of a linear regression loss
    let l = Linear::new(4, 3).unwrap();
    l.weight.assign(&t(&[3, 4], 0.6)).unwrap();
    l.bias.as_ref().unwrap().assign(&t(&[3], 1.4)).unwrap();
    let (xb, yb) = (t(&[5, 4], 2.2).to_dtype(oxideon_core::dtype::DType::F32).unwrap(), t(&[5, 3], 0.7).to_dtype(oxideon_core::dtype::DType::F32).unwrap());
    let params = vec![l.weight.clone(), l.bias.clone().unwrap()];
    let g = per_sample_grads(&params, 5, |i| l.forward(&xb.narrow(0, i, 1)?)?.sub(&yb.narrow(0, i, 1)?)?.square()?.sum()).unwrap();
    p("psg_w", &g[0]);
    p("psg_b", &g[1]);
}
