//! Cross-validation batch 19: LBFGS (one iteration per step) and Adafactor vs PyTorch (tools/crossval/check_batch19.py).
use oxideon_core::optim::*;
use oxideon_core::optim_extra::*;
use oxideon_core::tensor::Tensor;

fn target() -> Vec<f64> {
    vec![1.5, -0.7, 0.2, 2.5, -1.2, 0.9, 0.4, -2.0, 1.1, 0.3, -0.5, 1.8]
}

fn grad_of(p: &Tensor) -> Tensor {
    // loss = sum (p - t)^2 + 0.075 p^4 + 0.1 sin(p) p   =>  g = 2 (p - t) + 0.3 p^3 + 0.1 (cos(p) p + sin(p))
    let n: usize = p.shape().iter().product();
    let t = Tensor::from_slice(&target()[..n], p.shape()).unwrap();
    let d = p.detach();
    d.sub(&t).unwrap().mul_scalar(2.0).unwrap().add(&d.powf(3.0).unwrap().mul_scalar(0.3).unwrap()).unwrap().add(&d.cos().unwrap().mul(&d).unwrap().add(&d.sin().unwrap()).unwrap().mul_scalar(0.1).unwrap()).unwrap()
}

fn start(shape: &[usize]) -> Tensor {
    let v: Vec<f64> = [0.9, 1.4, -1.1, 0.3, 0.7, -0.4, 1.9, 0.2, -0.8, 1.2, 0.5, -1.5].iter().copied().take(shape.iter().product()).collect();
    Tensor::from_slice(&v, shape).unwrap().requires_grad_(true)
}

fn run(name: &str, mut opt: Box<dyn Optimizer>, p: &Tensor, steps: usize) {
    for s in 1..=steps {
        p.set_grad(Some(grad_of(p)));
        opt.step().unwrap();
        if [1, 2, 3, 5, 10, steps].contains(&s) {
            p.to_vec::<f64>().unwrap().iter().enumerate().for_each(|(i, v)| println!("{name}_s{s}_{i} {v:.14e}"));
        }
    }
}

#[test]
#[ignore]
fn emit() {
    macro_rules! go {
        ($name:literal, $shape:expr, $make:expr) => {{
            let p = start($shape);
            let opt: Box<dyn Optimizer> = Box::new($make(vec![p.clone()]));
            run($name, opt, &p, 25);
        }};
    }
    go!("lbfgs", &[12], |v| LBFGS::new(v, 1.0, 10));
    go!("lbfgs_h3", &[12], |v| LBFGS::new(v, 0.5, 3));
    go!("lbfgs_lr", &[4], |v| LBFGS::new(v, 0.2, 5));
    go!("adafactor_2d", &[3, 4], |v| Adafactor::new(v, 0.05));
    go!("adafactor_1d", &[12], |v| Adafactor::new(v, 0.05));
    go!("adafactor_wd", &[3, 4], |v| Adafactor::new(v, 0.1).weight_decay(0.05).beta2_decay(-0.6).clip_threshold(1.5));
    go!("adafactor_eps", &[4, 3], |v| Adafactor::new(v, 0.02).eps(Some(1e-6), 1e-2));
}
