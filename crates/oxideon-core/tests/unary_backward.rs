// The fused single-pass backward of tanh / sigmoid / exp / relu / square (CPU F32) must give the same gradients as the composed
// formulas, for contiguous and scalar-broadcast upstream gradients, below and above the parallel threshold, and a second
// derivative taken with create_graph must still work (that path stays composed).
use oxideon_core::{DType, Tensor, grad};

fn rnd(n: usize, seed: f32) -> Tensor {
    Tensor::from_slice(&(0..n).map(|i| ((i as f32 + seed) * 0.0731).sin() * 1.7).collect::<Vec<f32>>(), &[n]).unwrap()
}

fn close(a: &Tensor, b: &Tensor, tol: f64, tag: &str) {
    let x = a.to_dtype(DType::F64).unwrap().to_vec::<f64>().unwrap();
    let y = b.to_dtype(DType::F64).unwrap().to_vec::<f64>().unwrap();
    assert_eq!(x.len(), y.len(), "{tag}");
    let e = x.iter().zip(&y).map(|(p, q)| (p - q).abs() / (1.0 + q.abs())).fold(0.0, f64::max);
    assert!(e < tol, "{tag}: max error {e}");
}

type Act = fn(&Tensor) -> Tensor;

#[test]
fn fused_backward_matches_the_f64_composed_path() {
    let ops: [(&str, Act); 5] = [
        ("tanh", |t| t.tanh().unwrap()),
        ("sigmoid", |t| t.sigmoid().unwrap()),
        ("exp", |t| t.exp().unwrap()),
        ("relu", |t| t.relu().unwrap()),
        ("square", |t| t.square().unwrap()),
    ];
    for n in [37usize, 20_000, 70_001] {
        for (name, f) in ops {
            for weighted in [false, true] {
                let run = |dt: DType| {
                    let x = rnd(n, 1.0).to_dtype(dt).unwrap().detach().requires_grad_(true);
                    let y = f(&x);
                    // `sum` hands down a broadcast scalar; the weighted loss hands down a real tensor
                    let loss = if weighted { y.mul(&rnd(n, 5.0).to_dtype(dt).unwrap()).unwrap().sum().unwrap() } else { y.sum().unwrap() };
                    loss.backward().unwrap();
                    x.grad().unwrap()
                };
                close(&run(DType::F32), &run(DType::F64), 2e-5, &format!("{name} n={n} weighted={weighted}"));
            }
        }
    }
}

#[test]
fn second_derivatives_still_work() {
    // d²/dx² tanh(x) = -2 tanh(x) (1 - tanh(x)²)
    let x = rnd(100, 2.0).detach().requires_grad_(true);
    let y = x.tanh().unwrap().sum().unwrap();
    let g = grad(&[y], std::slice::from_ref(&x), None, true).unwrap().remove(0).unwrap();
    let gg = grad(&[g.sum().unwrap()], std::slice::from_ref(&x), None, false).unwrap().remove(0).unwrap();
    let t = x.detach().tanh().unwrap();
    let want = t.mul_scalar(-2.0).unwrap().mul(&t.square().unwrap().neg().unwrap().add_scalar(1.0).unwrap()).unwrap();
    close(&gg, &want, 1e-5, "tanh second derivative");
}
