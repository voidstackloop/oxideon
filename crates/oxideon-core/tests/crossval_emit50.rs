//! Cross-validation batch 50: autocast matmul / Linear numerics and the GradScaler scale trajectory vs torch.autocast and
//! torch.amp.GradScaler (tools/crossval/check_batch50.py).
use oxideon_core::dtype::DType;
use oxideon_core::nn::{Linear, Module};
use oxideon_core::optim::Sgd;
use oxideon_core::tensor::Tensor;
use oxideon_core::{GradScaler, autocast};

fn genv(n: usize, salt: f64) -> Vec<f32> {
    (0..n).map(|i| ((0.37 * i as f64 + salt).sin() * 1.3 + 0.1 * (0.11 * i as f64 * salt).cos()) as f32).collect()
}
fn p(name: &str, v: &[f32]) {
    println!("{name} {}", v.iter().map(|a| format!("{a:e}")).collect::<Vec<_>>().join(" "));
}

#[test]
#[ignore]
fn emit_amp() {
    let a = Tensor::from_slice(&genv(32 * 64, 0.4), &[32, 64]).unwrap();
    let b = Tensor::from_slice(&genv(64 * 48, 1.7), &[64, 48]).unwrap();
    for (name, dt) in [("bf16", DType::BF16), ("f16", DType::F16)] {
        let y = autocast(dt, || a.matmul(&b).unwrap());
        p(&format!("matmul_{name}"), &y.to_dtype(DType::F32).unwrap().to_vec::<f32>().unwrap());
    }
    // Linear with float32 parameters under bf16 autocast
    let l = Linear::new(64, 48).unwrap();
    l.weight.assign(&Tensor::from_slice(&genv(48 * 64, 0.9), &[48, 64]).unwrap()).unwrap();
    l.bias.as_ref().unwrap().assign(&Tensor::from_slice(&genv(48, 2.2), &[48]).unwrap()).unwrap();
    let out = autocast(DType::BF16, || l.forward(&a).unwrap());
    p("linear_bf16", &out.to_dtype(DType::F32).unwrap().to_vec::<f32>().unwrap());
    // scale trajectory: init 1024, growth every 3 clean steps, overflow on steps 4 and 5 (0-based) and 11
    let w = Tensor::from_slice(&[1.0f32, 2.0], &[2]).unwrap().requires_grad_(true);
    let mut opt = Sgd::new(vec![w.clone()], 0.01);
    let mut sc = GradScaler::new(1024.0).unwrap().with_intervals(2.0, 0.5, 3).unwrap();
    let mut scales = vec![];
    let mut taken = vec![];
    for step in 0..16 {
        let g = if [4, 5, 11].contains(&step) { f32::INFINITY } else { 0.5 * sc.scale_factor() as f32 };
        w.set_grad(Some(Tensor::from_slice(&[g, g], &[2]).unwrap()));
        taken.push(sc.step_and_update(&mut opt, std::slice::from_ref(&w)).unwrap() as u8 as f32);
        scales.push(sc.scale_factor() as f32);
    }
    p("scale_trajectory", &scales);
    p("steps_taken", &taken);
    p("final_weight", &w.to_vec::<f32>().unwrap());
}
