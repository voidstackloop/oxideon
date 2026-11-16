//! Cross-validation batch 29: layers not covered before (3-D / transposed convolutions, 1-D / 3-D pooling, batch norm
//! in eval mode after training steps, instance norm, PReLU, GLU, weight norm) vs PyTorch, values and gradients
//! (tools/crossval/check_batch29.py).
use oxideon_core::device::Device;
use oxideon_core::dtype::DType;
use oxideon_core::nn::*;
use oxideon_core::tensor::Tensor;

fn genv(n: usize, salt: f64) -> Vec<f64> {
    (0..n).map(|i| (0.37 * i as f64 + salt).sin() * 0.8 + 0.1 * (0.11 * i as f64 * salt).cos()).collect()
}

fn t(shape: &[usize], salt: f64) -> Tensor {
    Tensor::from_slice(&genv(shape.iter().product(), salt), shape).unwrap().to_dtype(DType::F32).unwrap()
}

fn emit(name: &str, x: &Tensor) {
    let v = x.to_dtype(DType::F64).unwrap().to_vec::<f64>().unwrap();
    println!("{name} {:?} {}", x.shape(), v.iter().map(|a| format!("{a:.9e}")).collect::<Vec<_>>().join(" "));
}

fn fill(p: &Tensor, salt: f64) {
    p.assign(&t(p.shape(), salt)).unwrap();
}

/// Emits the output, then input / parameter gradients of sum(out * w).
fn run(name: &str, x: &Tensor, out: &Tensor, params: &[(&str, &Tensor)]) {
    emit(&format!("{name}_out"), out);
    out.mul(&t(out.shape(), 99.0)).unwrap().sum().unwrap().backward().unwrap();
    emit(&format!("{name}_dx"), &x.grad().expect("input gradient"));
    for (tag, p) in params {
        match p.grad() {
            Some(g) => emit(&format!("{name}_d{tag}"), &g),
            None => println!("{name}_d{tag} NOGRAD"),
        }
    }
}

#[test]
#[ignore]
fn emit_layers() {
    // Conv3d
    let x = t(&[2, 2, 4, 5, 4], 1.0).requires_grad_(true);
    let c = Conv3d::new(2, 3, (2, 3, 2), (1, 2, 1), (1, 0, 1), true, DType::F32, Device::Cpu).unwrap();
    fill(&c.weight, 2.0);
    fill(c.bias.as_ref().unwrap(), 2.5);
    run("conv3d", &x, &c.forward(&x).unwrap(), &[("w", &c.weight), ("b", c.bias.as_ref().unwrap())]);

    // ConvTranspose1d
    let x = t(&[2, 3, 6], 3.0).requires_grad_(true);
    let c = ConvTranspose1d::new(3, 2, 3, 2, 1).unwrap();
    fill(&c.weight, 4.0);
    if let Some(b) = &c.bias {
        fill(b, 4.5);
    }
    println!("convt1d_wshape {:?} bias {}", c.weight.shape(), c.bias.is_some());
    let mut ps = vec![("w", &c.weight)];
    if let Some(b) = &c.bias {
        ps.push(("b", b));
    }
    run("convt1d", &x, &c.forward(&x).unwrap(), &ps);

    // ConvTranspose3d
    let x = t(&[1, 2, 3, 3, 2], 5.0).requires_grad_(true);
    let c = ConvTranspose3d::new(2, 2, (2, 2, 2), (2, 1, 2), (0, 1, 0), (1, 0, 1), true, DType::F32, Device::Cpu).unwrap();
    fill(&c.weight, 6.0);
    fill(c.bias.as_ref().unwrap(), 6.5);
    println!("convt3d_wshape {:?}", c.weight.shape());
    run("convt3d", &x, &c.forward(&x).unwrap(), &[("w", &c.weight), ("b", c.bias.as_ref().unwrap())]);

    // pooling 1-D / 3-D
    let x1 = t(&[2, 3, 11], 7.0);
    let pool1 = |name: &str, f: &dyn Fn(&Tensor) -> Tensor| {
        let xi = x1.detach().requires_grad_(true);
        run(name, &xi, &f(&xi), &[]);
    };
    pool1("maxpool1d", &|v| MaxPool1d::new(3, 2).forward(v).unwrap());
    pool1("avgpool1d", &|v| AvgPool1d::new(3, 2).forward(v).unwrap());
    pool1("adaavg1d", &|v| AdaptiveAvgPool1d::new(4).forward(v).unwrap());
    pool1("adamax1d", &|v| AdaptiveMaxPool1d::new(4).forward(v).unwrap());
    pool1("lppool1d", &|v| LPPool1d::new(2.0, 3, 2).forward(&v.abs().unwrap().add_scalar(0.1).unwrap()).unwrap());
    let x3 = t(&[1, 2, 4, 6, 4], 8.0);
    let pool3 = |name: &str, f: &dyn Fn(&Tensor) -> Tensor| {
        let xi = x3.detach().requires_grad_(true);
        run(name, &xi, &f(&xi), &[]);
    };
    pool3("maxpool3d", &|v| MaxPool3d::new((2, 2, 2), (2, 2, 2)).forward(v).unwrap());
    pool3("avgpool3d", &|v| AvgPool3d::new((2, 3, 2), (2, 3, 2)).forward(v).unwrap());

    // BatchNorm1d: three training steps on different batches, then eval
    for (tag, shape) in [("bn1d_2d", vec![6usize, 4]), ("bn1d_3d", vec![3, 4, 5])] {
        let bn = BatchNorm1d::new(4).unwrap();
        fill(&bn.weight, 10.0);
        fill(&bn.bias, 10.5);
        for step in 0..3 {
            let xi = t(&shape, 11.0 + step as f64).requires_grad_(true);
            let out = bn.forward(&xi).unwrap();
            if step == 2 {
                run(&format!("{tag}_train"), &xi, &out, &[("w", &bn.weight), ("b", &bn.bias)]);
            }
        }
        emit(&format!("{tag}_rm"), &bn.running_mean);
        emit(&format!("{tag}_rv"), &bn.running_var);
        bn.eval();
        let xe = t(&shape, 20.0).requires_grad_(true);
        bn.weight.zero_grad();
        bn.bias.zero_grad();
        run(&format!("{tag}_eval"), &xe, &bn.forward(&xe).unwrap(), &[("w", &bn.weight), ("b", &bn.bias)]);
    }
    // BatchNorm2d eval after two training steps
    let bn = BatchNorm2d::new(3).unwrap();
    fill(&bn.weight, 12.0);
    fill(&bn.bias, 12.5);
    for step in 0..2 {
        bn.forward(&t(&[2, 3, 4, 3], 13.0 + step as f64)).unwrap();
    }
    emit("bn2d_rm", &bn.running_mean);
    emit("bn2d_rv", &bn.running_var);
    bn.eval();
    let xe = t(&[2, 3, 4, 3], 21.0).requires_grad_(true);
    run("bn2d_eval", &xe, &bn.forward(&xe).unwrap(), &[("w", &bn.weight), ("b", &bn.bias)]);
    // BatchNorm3d train + eval
    let bn = BatchNorm3d::new(2, 1e-5, 0.1, DType::F32, Device::Cpu).unwrap();
    fill(&bn.weight, 14.0);
    fill(&bn.bias, 14.5);
    let xi = t(&[2, 2, 3, 2, 3], 15.0).requires_grad_(true);
    run("bn3d_train", &xi, &bn.forward(&xi).unwrap(), &[("w", &bn.weight), ("b", &bn.bias)]);
    emit("bn3d_rm", &bn.running_mean.read().unwrap());
    emit("bn3d_rv", &bn.running_var.read().unwrap());
    bn.eval();
    let xe = t(&[2, 2, 3, 2, 3], 22.0).requires_grad_(true);
    emit("bn3d_eval_out", &bn.forward(&xe).unwrap());

    // InstanceNorm
    let x = t(&[2, 3, 7], 16.0).requires_grad_(true);
    run("in1d", &x, &InstanceNorm1d::new(3, 1e-5).unwrap().forward(&x).unwrap(), &[]);
    let x = t(&[2, 3, 4, 5], 17.0).requires_grad_(true);
    run("in2d", &x, &InstanceNorm2d::new(3, 1e-5).unwrap().forward(&x).unwrap(), &[]);

    // PReLU (per-channel) and GLU
    let x = t(&[2, 3, 5], 18.0).requires_grad_(true);
    let pr = PReLU::new(3, 0.25).unwrap();
    fill(&pr.weight, 18.5);
    println!("prelu_wshape {:?}", pr.weight.shape());
    run("prelu", &x, &pr.forward(&x).unwrap(), &[("w", &pr.weight)]);
    let x = t(&[2, 6, 5], 19.0).requires_grad_(true);
    run("glu", &x, &GLU::new(1).forward(&x).unwrap(), &[]);

    // weight-normalized linear
    let wn = WeightNormLinear::new(5, 3, true).unwrap();
    fill(&wn.weight_v, 23.0);
    fill(&wn.weight_g, 23.5);
    fill(wn.bias.as_ref().unwrap(), 23.7);
    let x = t(&[4, 5], 24.0).requires_grad_(true);
    run("wnorm", &x, &wn.forward(&x).unwrap(), &[("v", &wn.weight_v), ("g", &wn.weight_g), ("b", wn.bias.as_ref().unwrap())]);
}
