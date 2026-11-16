//! Cross-validation batch 18: remaining learning-rate schedules vs PyTorch (tools/crossval/check_batch18.py).
use oxideon_core::optim::*;
use oxideon_core::optim_extra::*;
use oxideon_core::tensor::Tensor;

const N: usize = 30;

fn opt(lr: f64) -> Sgd {
    Sgd::new(vec![Tensor::zeros_f32(&[1]).unwrap().requires_grad_(true)], lr)
}

fn run<S: Schedule>(name: &str, base: f64, sched: S) {
    let mut o = opt(base);
    let mut s = LrScheduler::new(base, sched);
    println!("{name}_0 {:.14e}", s.lr());
    for e in 1..N {
        println!("{name}_{e} {:.14e}", s.step(&mut o));
    }
}

#[test]
#[ignore]
fn emit() {
    run("constant", 0.1, ConstantLr { factor: 0.3, total_iters: 4 });
    run("linear", 0.1, LinearLr { start_factor: 0.2, end_factor: 0.9, total_iters: 7 });
    run("multistep", 0.1, MultiStepLr { milestones: vec![4, 9, 9, 15], gamma: 0.5 });
    run("poly", 0.1, PolynomialLr { total_iters: 12, power: 2.0 });
    run("lambda", 0.1, LambdaLr(Box::new(|e| 0.95f64.powi(e as i32))));
    run("cyc_tri", 0.01, CyclicLr { base_lr: 0.01, max_lr: 0.1, step_size_up: 4, step_size_down: 6, mode: CyclicMode::Triangular });
    run("cyc_tri2", 0.01, CyclicLr { base_lr: 0.01, max_lr: 0.1, step_size_up: 4, step_size_down: 6, mode: CyclicMode::Triangular2 });
    run("cyc_exp", 0.01, CyclicLr { base_lr: 0.01, max_lr: 0.1, step_size_up: 3, step_size_down: 3, mode: CyclicMode::ExpRange(0.95) });
    run(
        "seq",
        0.1,
        SequentialLr {
            schedules: vec![Box::new(LinearLr { start_factor: 0.1, end_factor: 1.0, total_iters: 4 }), Box::new(MultiStepLr { milestones: vec![3, 8], gamma: 0.5 })],
            milestones: vec![4],
        },
    );
    run("chain", 0.1, ChainedLr(vec![Box::new(ConstantLr { factor: 0.5, total_iters: 5 }), Box::new(LambdaLr(Box::new(|e| 0.95f64.powi(e as i32))))]));

    for (name, max_lr, total, pct, div, fdiv) in [("onecycle_a", 0.1, 30usize, 0.3, 25.0, 1e4), ("onecycle_b", 0.5, 30, 0.45, 10.0, 100.0)] {
        let mut o = opt(max_lr);
        let mut s = OneCycleLR::new(max_lr, total, pct, div, fdiv);
        println!("{name}_0 {:.14e}", s.current_lr());
        for e in 1..total {
            println!("{name}_{e} {:.14e}", s.step(&mut o));
        }
    }

    // warmup + cosine: linear warmup then cosine decay (compared with a closed-form LambdaLR reference)
    let mut o = opt(0.1);
    let mut s = LinearWarmupCosineLr::new(0.1, 5, 25, 0.001);
    for e in 1..N {
        println!("warmcos_{e} {:.14e}", s.step(&mut o));
    }

    // ReduceLROnPlateau
    let metrics = [1.0, 0.9, 0.9, 0.9, 0.9, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
    let mut o = opt(0.1);
    let mut s = ReduceLROnPlateau::new(0.5, 2, 1e-4);
    for (e, m) in metrics.iter().enumerate() {
        println!("plateau_{e} {:.14e}", s.step(*m, &mut o));
    }
    // option combinations: (tag, maximize, threshold, relative, cooldown)
    let combos = [("pa", false, 0.02, false, 0usize), ("pb", false, 1e-4, true, 3), ("pc", true, 0.05, true, 0), ("pd", true, 0.01, false, 2)];
    for (tag, mx, th, rel, cd) in combos {
        let mut o = opt(0.1);
        let mut s = ReduceLROnPlateau::new(0.5, 1, 1e-3).maximize(mx).threshold(th, rel).cooldown(cd);
        for (e, m) in metrics.iter().enumerate() {
            let v = if mx { 2.0 - m } else { *m };
            println!("{tag}_{e} {:.14e}", s.step(v, &mut o));
        }
    }
}
