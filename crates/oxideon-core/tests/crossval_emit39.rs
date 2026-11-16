//! Cross-validation batch 39: GeGlu/ReGlu/SquaredRelu/Maxout, CosineLinear, WeightNormLinear, ConvPadded and the T5
//! relative-position bucketing vs PyTorch (tools/crossval/check_batch39.py).
use oxideon_core::nn::*;
use oxideon_core::nn::sequence_ext::*;
use oxideon_core::nn::layers_ext::ConvPadded;
use oxideon_core::nn::functional_ext::PadMode;
use oxideon_core::tensor::Tensor;

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
/// value, then gradient of sum(out * weights) wrt the input
fn run(name: &str, x: &Tensor, f: impl Fn(&Tensor) -> Tensor) {
    let x = x.detach().requires_grad_(true);
    let y = f(&x);
    p(name, &y);
    let w = t(y.shape(), 2.9);
    y.mul(&w).unwrap().sum().unwrap().backward().unwrap();
    p(&format!("{name}_grad"), &x.grad().unwrap());
}

#[test]
#[ignore]
fn emit_modules() {
    let x = t(&[3, 8], 0.4);
    run("geglu", &x, |x| GeGlu.forward(x).unwrap());
    run("reglu", &x, |x| ReGlu.forward(x).unwrap());
    run("sqrelu", &x, |x| SquaredRelu.forward(x).unwrap());
    run("maxout", &x, |x| Maxout(4).forward(x).unwrap());

    let mut cl = CosineLinear::new(8, 5, 12.0).unwrap();
    cl.weight = t(&[5, 8], 1.7).requires_grad_(true);
    cl.scale = Tensor::from_slice(&[12.0f64], &[1]).unwrap();
    run("cosine_linear", &x, |x| cl.forward(x).unwrap());

    let mut wn = WeightNormLinear::new(8, 5, true).unwrap();
    wn.weight_v = t(&[5, 8], 0.9);
    wn.weight_g = t(&[5, 1], 3.3).abs().unwrap().add_scalar(0.5).unwrap();
    wn.bias = Some(t(&[5], 4.1));
    p("wn_weight", &wn.effective_weight().unwrap());
    run("wn_forward", &x, |x| wn.forward(x).unwrap());

    let img = t(&[2, 3, 6, 7], 0.2);
    for (nm, mode) in [("reflect", PadMode::Reflect), ("replicate", PadMode::Replicate), ("circular", PadMode::Circular)] {
        let conv = Conv2d { weight: t(&[4, 3, 3, 3], 1.1), bias: Some(t(&[4], 0.3)), stride: (1, 2), padding: (2, 1), dilation: (1, 1), groups: 1 };
        let cp = ConvPadded::new(conv, mode);
        run(&format!("convpad_{nm}"), &img, |x| cp.forward(x).unwrap());
    }

    for (bi, nb, md) in [(true, 32usize, 128usize), (false, 32, 128), (true, 16, 50), (false, 8, 20)] {
        let r = RelativePositionBias::new(nb, md, 2, bi).unwrap();
        let v: Vec<f64> = (-40..=40).map(|d: i64| r.bucket(d) as f64).collect();
        println!("t5bucket_{}_{nb}_{md} {}", bi as u8, v.iter().map(|a| format!("{a}")).collect::<Vec<_>>().join(" "));
    }
    let mut r = RelativePositionBias::new(8, 20, 2, true).unwrap();
    r.embedding = t(&[8, 2], 0.6);
    p("t5bias", &r.forward(3, 4).unwrap());
}
