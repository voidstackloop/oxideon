//! Cross-validation batch 20: activation functions, values and gradients, including exact boundary points
//! (tools/crossval/check_batch20.py).
use oxideon_core::tensor::Tensor;

fn grid() -> Vec<f64> {
    vec![-30.0, -8.0, -6.0, -4.5, -3.0, -2.0, -1.5, -1.0, -0.6, -0.5, -0.1, -1e-9, 0.0, 1e-9, 0.1, 0.5, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 4.5, 6.0, 8.0, 19.0, 21.0, 30.0]
}

fn emit(name: &str, f: &dyn Fn(&Tensor) -> Tensor) {
    let g = grid();
    let x = Tensor::from_slice(&g, &[g.len()]).unwrap().requires_grad_(true);
    let y = f(&x);
    y.sum().unwrap().backward().unwrap();
    let yv = y.to_vec::<f64>().unwrap();
    let gv = x.grad().unwrap().to_vec::<f64>().unwrap();
    for i in 0..g.len() {
        println!("{name}_v_{i} {:.15e}", yv[i]);
        println!("{name}_g_{i} {:.15e}", gv[i]);
    }
}

#[test]
#[ignore]
fn emit_all() {
    emit("relu", &|x| x.relu().unwrap());
    emit("relu6", &|x| x.relu6().unwrap());
    emit("leaky", &|x| x.leaky_relu(0.07).unwrap());
    emit("elu", &|x| x.elu(1.3).unwrap());
    emit("selu", &|x| x.selu().unwrap());
    emit("celu", &|x| x.celu(0.7).unwrap());
    emit("celu_neg", &|x| x.celu(-0.7).unwrap());
    emit("gelu", &|x| x.gelu().unwrap());
    emit("silu", &|x| x.silu().unwrap());
    emit("mish", &|x| x.mish().unwrap());
    emit("softplus", &|x| x.softplus().unwrap());
    emit("softplus_b2", &|x| x.softplus_beta(2.0).unwrap());
    emit("softplus_b05", &|x| x.softplus_beta(0.5).unwrap());
    emit("softsign", &|x| x.softsign().unwrap());
    emit("hardsigmoid", &|x| x.hardsigmoid().unwrap());
    emit("hardswish", &|x| x.hardswish().unwrap());
    emit("hardtanh", &|x| x.hardtanh(-1.5, 2.0).unwrap());
    emit("hardshrink", &|x| x.hardshrink(0.6).unwrap());
    emit("softshrink", &|x| x.softshrink(0.6).unwrap());
    emit("tanhshrink", &|x| x.tanhshrink().unwrap());
    emit("logsigmoid", &|x| x.log_sigmoid().unwrap());
    emit("sigmoid", &|x| x.sigmoid().unwrap());
    emit("tanh", &|x| x.tanh().unwrap());
}
