//! Fourth cross-validation batch: optimizers and learning-rate schedules vs PyTorch.
//! Run: `cargo test -p oxideon-core --test crossval_emit4 -- --nocapture --ignored`.

use oxideon_core::optim::*;
use oxideon_core::optim_extra::*;
use oxideon_core::tensor::Tensor;

fn grad_of(p: &Tensor, target: &[f64]) -> Tensor {
    // g = 2 (p - target) + 0.3 p^3
    let t = Tensor::from_slice(target, &[target.len()]).unwrap();
    p.detach().sub(&t).unwrap().mul_scalar(2.0).unwrap().add(&p.detach().powf(3.0).unwrap().mul_scalar(0.3).unwrap()).unwrap()
}

fn run(name: &str, mut opt: Box<dyn Optimizer>, p: &Tensor, steps: usize) {
    let target = [1.5, -0.7, 0.2, 2.5];
    for s in 1..=steps {
        p.set_grad(Some(grad_of(p, &target)));
        opt.step().unwrap();
        if [1, 2, 5, 12, steps].contains(&s) {
            p.to_vec::<f64>().unwrap().iter().enumerate().for_each(|(i, v)| println!("opt_{name}_s{s}_{i} {v:.12e}"));
        }
    }
}

fn start() -> Tensor {
    Tensor::from_slice(&[0.9f64, 1.4, -1.1, 0.3], &[4]).unwrap().requires_grad_(true)
}

#[test]
#[ignore]
fn emit_optimizers() {
    macro_rules! go {
        ($name:literal, $make:expr) => {{
            let p = start();
            let opt: Box<dyn Optimizer> = Box::new($make(vec![p.clone()]));
            run($name, opt, &p, 30);
        }};
    }
    go!("sgd", |v| Sgd::new(v, 0.05));
    go!("sgd_mom", |v| Sgd::new(v, 0.05).momentum(0.9));
    go!("sgd_nesterov", |v| Sgd::new(v, 0.05).momentum(0.9).nesterov(true));
    go!("sgd_damp", |v| Sgd::new(v, 0.05).momentum(0.8).dampening(0.3).weight_decay(0.01));
    go!("sgd_max", |v| Sgd::new(v, 0.05).momentum(0.5).maximize(true));
    go!("adam", |v| Adam::new(v, 0.05));
    go!("adam_ams", |v| Adam::new(v, 0.05).amsgrad(true));
    go!("adam_wd_ams", |v| Adam::new(v, 0.05).weight_decay(0.02).amsgrad(true).betas(0.8, 0.9));
    go!("adamw_ams", |v| Adam::adamw(v, 0.05, 0.05).amsgrad(true));
    go!("adam_max", |v| Adam::new(v, 0.05).maximize(true));
    go!("adagrad", |v| Adagrad::new(v, 0.1));
    go!("rmsprop", |v| RmsProp::new(v, 0.01));
    go!("rmsprop_centered", |v| RmsProp::new(v, 0.01).centered(true));
    go!("rmsprop_mom", |v| RmsProp::new(v, 0.01).momentum(0.9).alpha(0.9).eps(1e-6).weight_decay(0.01));
    go!("rmsprop_cmom", |v| RmsProp::new(v, 0.01).centered(true).momentum(0.5));
    go!("adagrad_decay", |v| Adagrad::new(v, 0.1).lr_decay(0.05).initial_accumulator_value(0.2).weight_decay(0.01).eps(1e-8));
    go!("adadelta", |v| Adadelta::new(v, 1.0));
    go!("adamax", |v| Adamax::new(v, 0.05));
    go!("nadam", |v| NAdam::new(v, 0.05));
    go!("radam", |v| RAdam::new(v, 0.05));
    go!("rprop", |v| Rprop::new(v, 0.05));
    go!("asgd", |v| Asgd::new(v, 0.05));
}

#[test]
#[ignore]
fn emit_schedules() {
    let steps = 30;
    let mk = |lr: f64| Sgd::new(vec![Tensor::zeros_f32(&[1]).unwrap().requires_grad_(true)], lr);
    println!("sched_step_0 {:.12e}", 0.1);
    let (mut o, mut s) = (mk(0.1), StepLr::new(0.1, 7, 0.5));
    for e in 1..steps {
        println!("sched_step_{e} {:.12e}", s.step(&mut o));
    }
    println!("sched_cosine_0 {:.12e}", 0.1);
    let (mut o, mut s) = (mk(0.1), CosineAnnealingLr::new(0.1, 20, 0.001));
    for e in 1..steps {
        println!("sched_cosine_{e} {:.12e}", s.step(&mut o));
    }
    println!("sched_exp_0 {:.12e}", 0.1);
    let (mut o, mut s) = (mk(0.1), ExponentialLr::new(0.93));
    for e in 1..steps {
        println!("sched_exp_{e} {:.12e}", s.step(&mut o));
    }
    println!("sched_warm_0 {:.12e}", 0.1);
    let (mut o, mut s) = (mk(0.1), CosineAnnealingWarmRestarts::new(0.1, 6, 2, 0.001));
    for e in 1..steps {
        println!("sched_warm_{e} {:.12e}", s.step(&mut o));
    }
}
