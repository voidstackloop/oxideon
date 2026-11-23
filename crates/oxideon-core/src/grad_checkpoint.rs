//! Activation gradient checkpointing (`checkpoint`) for memory-efficient training.
//!
//! Runs the forward closure under `no_grad` so intermediate activations are not retained,
//! then recomputes them on-the-fly during backward to compute exact input gradients.

use std::sync::Arc;

use crate::error::Result;
use crate::tensor::{Tensor, is_grad_enabled, set_grad_enabled};

/// Runs `f(inputs)` without saving intermediate activations, recomputing them during backward.
pub fn checkpoint<F>(f: F, inputs: &[Tensor]) -> Result<Tensor>
where
    F: Fn(&[Tensor]) -> Result<Tensor> + Send + Sync + 'static,
{
    let any_req = inputs.iter().any(Tensor::requires_grad);
    if !is_grad_enabled() || !any_req {
        return f(inputs);
    }

    // the forward is replayed in backward, so random layers (dropout) must see the same random stream both times
    let rng_state = crate::rng::get_rng_state();
    let f_arc = Arc::new(f);
    let detached: Vec<Tensor> = inputs.iter().map(Tensor::detach).collect();
    let out = {
        let _ng = set_grad_enabled(false);
        f_arc(&detached)?
    };

    let saved_inputs: Vec<Tensor> = inputs.to_vec();
    let saved_req: Vec<bool> = inputs.iter().map(Tensor::requires_grad).collect();
    let f_bwd = Arc::clone(&f_arc);

    Ok(Tensor::from_op(
        out.storage().clone(),
        out.layout().clone(),
        "checkpoint",
        inputs.to_vec(),
        move |gy: &Tensor| {
            let create_graph = is_grad_enabled();
            let mut leaves: Vec<Tensor> = Vec::with_capacity(saved_inputs.len());
            for (inp, &req) in saved_inputs.iter().zip(saved_req.iter()) {
                let d = inp.detach();
                if req {
                    d.set_requires_grad(true)?;
                }
                leaves.push(d);
            }
            // replay under the forward's random stream; the global stream is restored afterwards
            crate::rng::fork_rng(|| -> Result<()> {
                crate::rng::set_rng_state(&rng_state)?;
                let recomputed = {
                    let _g = set_grad_enabled(true);
                    f_bwd(&leaves)?
                };
                recomputed.backward_with_grad(gy, create_graph)
            })?;
            let mut grads = Vec::with_capacity(leaves.len());
            for (leaf, &req) in leaves.iter().zip(saved_req.iter()) {
                if req {
                    grads.push(leaf.grad());
                } else {
                    grads.push(None);
                }
            }
            Ok(grads)
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_replays_the_same_dropout_mask_in_backward() {
        use crate::nn::{Dropout, Module};
        crate::rng::manual_seed(123);
        let drop = std::sync::Arc::new(Dropout::new(0.5));
        let x = Tensor::from_slice(&[1.0f32; 64], &[64]).unwrap();
        x.set_requires_grad(true).unwrap();
        let d = std::sync::Arc::clone(&drop);
        let y = checkpoint(move |ins| d.forward(&ins[0]), &[x.clone()]).unwrap();
        let before = crate::rng::get_rng_state();
        y.sum().unwrap().backward().unwrap();
        // x = 1, so y is exactly the mask and d(sum y)/dx must be that same mask
        assert_eq!(y.to_vec::<f32>().unwrap(), x.grad().unwrap().to_vec::<f32>().unwrap());
        assert!(y.to_vec::<f32>().unwrap().contains(&0.0), "dropout should have dropped something");
        assert_eq!(before, crate::rng::get_rng_state(), "backward must leave the global random stream untouched");
    }

    #[test]
    fn gradient_checkpoint_matches_standard_backward() {
        let x_ref = Tensor::from_slice(&[1.0f32, -2.0, 3.0, 0.5], &[2, 2]).unwrap();
        x_ref.set_requires_grad(true).unwrap();
        let w_ref = Tensor::from_slice(&[0.5f32, -0.25, 1.0, 0.75], &[2, 2]).unwrap();
        w_ref.set_requires_grad(true).unwrap();

        let y_ref = x_ref.matmul(&w_ref).unwrap().relu().unwrap().matmul(&w_ref).unwrap().sum().unwrap();
        y_ref.backward().unwrap();

        let x_cp = Tensor::from_slice(&[1.0f32, -2.0, 3.0, 0.5], &[2, 2]).unwrap();
        x_cp.set_requires_grad(true).unwrap();
        let w_cp = Tensor::from_slice(&[0.5f32, -0.25, 1.0, 0.75], &[2, 2]).unwrap();
        w_cp.set_requires_grad(true).unwrap();

        let y_cp = checkpoint(
            |ins| ins[0].matmul(&ins[1])?.relu()?.matmul(&ins[1]),
            &[x_cp.clone(), w_cp.clone()],
        )
        .unwrap()
        .sum()
        .unwrap();
        y_cp.backward().unwrap();

        let gx_ref = x_ref.grad().unwrap().to_vec::<f32>().unwrap();
        let gx_cp = x_cp.grad().unwrap().to_vec::<f32>().unwrap();
        let gw_ref = w_ref.grad().unwrap().to_vec::<f32>().unwrap();
        let gw_cp = w_cp.grad().unwrap().to_vec::<f32>().unwrap();
        for (a, b) in gx_ref.iter().zip(gx_cp.iter()) {
            assert!((a - b).abs() < 1e-5, "gx mismatch: {a} vs {b}");
        }
        for (a, b) in gw_ref.iter().zip(gw_cp.iter()) {
            assert!((a - b).abs() < 1e-5, "gw mismatch: {a} vs {b}");
        }
    }
}
