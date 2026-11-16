//! Tests for higher-order autograd (2nd & 3rd derivatives, Hessian-vector products),
//! hooks, `retain_grad`, custom ops, gradient checkpointing, and optimizers.

use oxideon::nn::loss::mse_loss;
use oxideon::prelude::*;

#[test]
fn second_and_third_order_derivatives() -> Result<()> {
    // f(x) = x^4 + 2 * sin(x)
    // f'(x) = 4x^3 + 2 * cos(x)
    // f''(x) = 12x^2 - 2 * sin(x)
    // f'''(x) = 24x - 2 * cos(x)
    let x0 = 1.3f32;
    let x = Tensor::from_slice(&[x0], &[1])?.requires_grad_(true);
    let y = (&x.powf(4.0)? + &x.sin()?.mul_scalar(2.0)?)?;

    let g1 = grad(&[y], std::slice::from_ref(&x), None, true)?
        .pop()
        .unwrap()
        .unwrap();
    let expected_g1 = 4.0 * x0.powi(3) + 2.0 * x0.cos();
    assert!((g1.item::<f32>()? - expected_g1).abs() < 1e-4);

    let g2 = grad(&[g1], std::slice::from_ref(&x), None, true)?
        .pop()
        .unwrap()
        .unwrap();
    let expected_g2 = 12.0 * x0.powi(2) - 2.0 * x0.sin();
    assert!((g2.item::<f32>()? - expected_g2).abs() < 1e-4);

    let g3 = grad(&[g2], std::slice::from_ref(&x), None, false)?
        .pop()
        .unwrap()
        .unwrap();
    let expected_g3 = 24.0 * x0 - 2.0 * x0.cos();
    assert!((g3.item::<f32>()? - expected_g3).abs() < 1e-3);
    Ok(())
}

#[test]
fn hessian_vector_product_via_higher_order_grad() -> Result<()> {
    // f(x) = 0.5 * x^T A x, where A = [[3, 1], [1, 2]]
    // grad f(x) = A x,  H * v = A v
    let a = Tensor::from_slice(&[3.0f32, 1.0, 1.0, 2.0], &[2, 2])?;
    let x = Tensor::from_slice(&[1.5f32, -0.5], &[2, 1])?.requires_grad_(true);
    let v = Tensor::from_slice(&[2.0f32, 3.0], &[2, 1])?;

    let fx = x.t()?.matmul(&a)?.matmul(&x)?.mul_scalar(0.5)?;
    let gx = grad(&[fx], std::slice::from_ref(&x), None, true)?
        .pop()
        .unwrap()
        .unwrap();
    let gv = gx.mul(&v)?.sum()?;
    let hv = grad(&[gv], &[x], None, false)?.pop().unwrap().unwrap();
    assert_eq!(hv.to_vec::<f32>()?, vec![9.0, 8.0]); // A @ [2, 3]^T = [9, 8]^T
    Ok(())
}

#[test]
fn hooks_and_retain_grad() -> Result<()> {
    let x = Tensor::from_slice(&[3.0f32], &[1])?.requires_grad_(true);
    let z = x.square()?; // z = x^2 = 9, dz/dx = 2x = 6
    z.retain_grad();

    // Register a hook on `x` that doubles its gradient.
    x.register_hook(|g| Ok(Some(g.mul_scalar(2.0)?)));

    let out = z.mul_scalar(5.0)?; // out = 5 * x^2
    out.backward()?;

    assert_eq!(z.grad().unwrap().to_vec::<f32>()?, vec![5.0]);
    // Unhooked dx would be 5 * 6 = 30; hook doubles it to 60.
    assert_eq!(x.grad().unwrap().to_vec::<f32>()?, vec![60.0]);
    Ok(())
}

#[test]
fn gradient_checkpointing_matches_standard_backward() -> Result<()> {
    manual_seed(11);
    let x1 = Tensor::randn(&[4, 6])?.requires_grad_(true);
    let x2 = x1.detach().requires_grad_(true);
    let w = Tensor::randn(&[6, 6])?.requires_grad_(true);
    let w2 = w.detach().requires_grad_(true);

    let block = |inputs: &[Tensor]| -> Result<Tensor> {
        let h = inputs[0].matmul(&inputs[1])?.gelu()?;
        h.matmul(&inputs[1].t()?)?.tanh()
    };

    // Standard forward + backward
    let y_std = block(&[x1.clone(), w.clone()])?.sum()?;
    y_std.backward()?;

    // Checkpointed forward + backward
    let y_ckpt = checkpoint(block, &[x2.clone(), w2.clone()])?.sum()?;
    y_ckpt.backward()?;

    assert!(
        x1.grad()
            .unwrap()
            .allclose(&x2.grad().unwrap(), 1e-5, 1e-5)?
    );
    assert!(
        w.grad()
            .unwrap()
            .allclose(&w2.grad().unwrap(), 1e-5, 1e-5)?
    );
    Ok(())
}

struct CubeOp;
impl CustomOp for CubeOp {
    fn name(&self) -> &'static str {
        "cube"
    }
    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor> {
        inputs[0].powf(3.0)
    }
    fn backward(&self, inputs: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        Ok(vec![Some(
            grad_output.mul(&inputs[0].square()?.mul_scalar(3.0)?)?,
        )])
    }
}

#[test]
fn custom_op_forward_and_backward() -> Result<()> {
    let x = Tensor::from_slice(&[2.0f32, -3.0], &[2])?.requires_grad_(true);
    let y = Tensor::apply_custom(CubeOp, std::slice::from_ref(&x))?;
    assert_eq!(y.to_vec::<f32>()?, vec![8.0, -27.0]);
    y.sum()?.backward()?;
    assert_eq!(x.grad().unwrap().to_vec::<f32>()?, vec![12.0, 27.0]);
    Ok(())
}

#[test]
fn optimizers_and_schedulers_converge() -> Result<()> {
    manual_seed(42);
    let x = Tensor::randn(&[32, 3])?;
    let true_w = Tensor::from_slice(&[1.5f32, -2.0, 0.5], &[3, 1])?;
    let y = no_grad(|| x.matmul(&true_w)?.add_scalar(0.3))?;

    let model = Linear::new(3, 1)?;
    let mut opt = Adam::adamw(model.parameters(), 0.08, 1e-4);
    let mut sched = CosineAnnealingLr::new(0.08, 250, 0.005);

    for _ in 0..250 {
        let loss = mse_loss(&model.forward(&x)?, &y)?;
        opt.zero_grad();
        loss.backward()?;
        clip_grad_norm(&model.parameters(), 5.0)?;
        opt.step()?;
        sched.step(&mut opt);
    }

    let final_loss = mse_loss(&model.forward(&x)?, &y)?.item::<f32>()?;
    assert!(final_loss < 1e-3, "final_loss = {final_loss}");
    Ok(())
}

#[test]
fn grad_fn_validates_outputs_seeds_and_hooks() -> Result<()> {
    let x = Tensor::from_slice(&[2.0f32], &[1])?.requires_grad_(true);
    let y = x.square()?.sum()?;
    // non-grad outputs are rejected like `backward()` rejects them
    let plain = Tensor::from_slice(&[2.0f32], &[1])?;
    let err = grad(
        std::slice::from_ref(&plain),
        std::slice::from_ref(&x),
        None,
        false,
    )
    .unwrap_err();
    assert!(
        format!("{err:?}").contains("does not require grad"),
        "{err:?}"
    );
    // wrong-count and wrong-size seeds fail with a pointer, not downstream noise
    let seed = Tensor::ones(&[1], x.dtype(), &x.device())?;
    let err = grad(
        std::slice::from_ref(&y),
        std::slice::from_ref(&x),
        Some(&[seed.clone(), seed]),
        false,
    )
    .unwrap_err();
    assert!(format!("{err:?}").contains("grad_outputs"), "{err:?}");
    let big = Tensor::ones(&[2], x.dtype(), &x.device())?;
    let err = grad(
        std::slice::from_ref(&y),
        std::slice::from_ref(&x),
        Some(std::slice::from_ref(&big)),
        false,
    )
    .unwrap_err();
    assert!(
        format!("{err:?}").contains("grad() grad_output 0"),
        "{err:?}"
    );
    // a hook that changes the gradient size is caught by both engines
    x.register_hook(|g| Ok(Some(Tensor::ones(&[2], g.dtype(), &g.device())?)));
    assert!(y.backward().is_err());
    assert!(grad(&[y], &[x], None, false).is_err());
    Ok(())
}

#[test]
fn backward_accumulates_and_diamonds_sum() -> Result<()> {
    // repeated backward() accumulates; zero_grad() clears
    let x = Tensor::from_slice(&[2.0f32], &[1])?.requires_grad_(true);
    let y = x.square()?.sum()?;
    y.backward()?;
    assert_eq!(x.grad().unwrap().to_vec::<f32>()?, vec![4.0]);
    y.backward()?;
    assert_eq!(x.grad().unwrap().to_vec::<f32>()?, vec![8.0]);
    x.zero_grad();
    assert!(x.grad().is_none());
    // diamond: y = x*x + x, dy/dx = 2x + 1 = 5 at x = 2
    let d = Tensor::from_slice(&[2.0f32], &[1])?.requires_grad_(true);
    let yd = Tensor::add(&Tensor::mul(&d, &d)?, &d)?.sum()?;
    yd.backward()?;
    assert_eq!(d.grad().unwrap().to_vec::<f32>()?, vec![5.0]);
    Ok(())
}

#[test]
fn engine_edges_lenient_strict_detach_and_nograd() -> Result<()> {
    // grad flags: lenient builder no-ops on ints, strict setter errors
    let ints = Tensor::from_slice(&[1i64, 2], &[2])?.requires_grad_(true);
    assert!(!ints.requires_grad());
    assert!(ints.set_requires_grad(true).is_err());
    // backward() rejects non-scalar and non-grad roots
    let v = Tensor::from_slice(&[1.0f32, 2.0], &[2])?.requires_grad_(true);
    assert!(v.backward().is_err());
    let plain = Tensor::from_slice(&[1.0f32], &[1])?;
    assert!(plain.backward().is_err());
    // intermediates only keep grads with retain_grad()
    let x = Tensor::from_slice(&[3.0f32], &[1])?.requires_grad_(true);
    let h = x.square()?;
    h.sum()?.backward()?;
    assert!(h.grad().is_none());
    assert_eq!(x.grad().unwrap().to_vec::<f32>()?, vec![6.0]);
    // detach shares storage but drops the graph
    let y = x.mul_scalar(2.0)?;
    let d = y.detach();
    assert!(d.is_leaf() && !d.requires_grad());
    assert!(y.shares_storage(&d));
    assert_eq!(d.to_vec::<f32>()?, vec![6.0]);
    // no_grad outputs never track, even from grad parents
    let z = no_grad(|| x.mul_scalar(2.0))?;
    assert!(!z.requires_grad());
    Ok(())
}

#[test]
fn grad_duplicate_inputs_each_get_a_gradient() -> Result<()> {
    // Like torch, asking for the same input twice yields two copies of its
    // gradient (a remove-based lookup starved the second entry to None).
    let x = Tensor::from_slice(&[2.0f32], &[1])?.requires_grad_(true);
    let y = x.square()?.sum()?;
    let gs = grad(&[y], &[x.clone(), x], None, false)?;
    assert_eq!(gs.len(), 2);
    assert_eq!(gs[0].as_ref().unwrap().to_vec::<f32>()?, vec![4.0]);
    assert_eq!(gs[1].as_ref().unwrap().to_vec::<f32>()?, vec![4.0]);
    Ok(())
}

#[test]
fn grad_edges_dup_outputs_unrelated_and_no_slot_writes() -> Result<()> {
    let x = Tensor::from_slice(&[2.0f32], &[1])?.requires_grad_(true);
    let y = x.square()?.sum()?;
    // duplicate outputs accumulate their seeds (ones + ones)
    let gs = grad(&[y.clone(), y], std::slice::from_ref(&x), None, false)?;
    assert_eq!(gs[0].as_ref().unwrap().to_vec::<f32>()?, vec![8.0]);
    // inputs outside the graph yield None (torch parity)
    let outsider = Tensor::from_slice(&[5.0f32], &[1])?.requires_grad_(true);
    let y2 = x.square()?.sum()?;
    let gs = grad(&[y2], std::slice::from_ref(&outsider), None, false)?;
    assert!(gs[0].is_none());
    // grad() never writes .grad slots (torch parity)
    assert!(x.grad().is_none());
    assert!(outsider.grad().is_none());
    Ok(())
}

#[test]
fn backward_with_grad_create_graph_enables_second_order() -> Result<()> {
    // y = x^3 at x = 2: first backward (create_graph) leaves a grad that
    // still requires grad; differentiating it gives 6x = 12.
    let x = Tensor::from_slice(&[2.0f32], &[1])?.requires_grad_(true);
    let y = x.powf(3.0)?.sum()?;
    let seed = Tensor::ones(&[1], x.dtype(), &x.device())?;
    y.backward_with_grad(&seed, true)?;
    let g1 = x.grad().unwrap();
    assert_eq!(g1.to_vec::<f32>()?, vec![12.0]);
    assert!(g1.requires_grad());
    x.zero_grad();
    g1.sum()?.backward()?;
    assert_eq!(x.grad().unwrap().to_vec::<f32>()?, vec![12.0]);
    Ok(())
}

#[test]
fn grad_mode_nesting_restores_and_stays_thread_local() -> Result<()> {
    use std::thread;
    assert!(is_grad_enabled());
    no_grad(|| {
        assert!(!is_grad_enabled());
        // inner enable_grad re-enables recording inside no_grad ...
        let y = enable_grad(|| {
            assert!(is_grad_enabled());
            let x = Tensor::from_slice(&[2.0f32], &[1]).unwrap().requires_grad_(true);
            x.square().unwrap()
        });
        assert!(y.requires_grad());
        // ... and dropping it restores the outer disabled state.
        assert!(!is_grad_enabled());
        // other threads are unaffected (mode is thread-local like torch)
        let h = thread::spawn(is_grad_enabled);
        assert!(h.join().unwrap());
    });
    assert!(is_grad_enabled());
    Ok(())
}

#[test]
fn hook_returning_none_keeps_gradient() -> Result<()> {
    let x = Tensor::from_slice(&[2.0f32], &[1])?.requires_grad_(true);
    x.register_hook(|_g| Ok(None));
    let y = x.square()?.sum()?;
    y.backward()?;
    assert_eq!(x.grad().unwrap().to_vec::<f32>()?, vec![4.0]);
    Ok(())
}
