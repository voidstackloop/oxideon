//! 3-class spiral classification (the classic CS231n dataset) with an MLP,
//! cross-entropy loss, mini-batches, and AdamW.
//!
//! Run with: `cargo run --release --example spiral`

use oxideon::data::batch_indices;
use oxideon::nn::loss::cross_entropy;
use oxideon::prelude::*;
use oxideon::rng::with_rng;

const POINTS_PER_CLASS: usize = 200;
const CLASSES: usize = 3;

fn make_spiral() -> Result<(Tensor, Vec<usize>)> {
    let mut xs = Vec::with_capacity(POINTS_PER_CLASS * CLASSES * 2);
    let mut ys = Vec::with_capacity(POINTS_PER_CLASS * CLASSES);
    for class in 0..CLASSES {
        for i in 0..POINTS_PER_CLASS {
            let r = i as f32 / POINTS_PER_CLASS as f32;
            let theta = class as f32 * 4.0 + r * 4.0 + with_rng(|g| g.normal()) * 0.2;
            xs.extend([r * theta.sin(), r * theta.cos()]);
            ys.push(class);
        }
    }
    Ok((Tensor::from_vec(xs, &[POINTS_PER_CLASS * CLASSES, 2])?, ys))
}

fn accuracy(logits: &Tensor, labels: &[usize]) -> Result<f32> {
    let pred = logits.argmax_axis(1, false)?.to_vec::<i64>()?;
    let correct = pred
        .iter()
        .zip(labels)
        .filter(|&(&p, &y)| p as usize == y)
        .count();
    Ok(correct as f32 / labels.len() as f32)
}

fn main() -> Result<()> {
    manual_seed(42);
    let (x, y) = make_spiral()?;
    let n = y.len();

    let model = Sequential::new()
        .add(Linear::new(2, 64)?)
        .add(GELU)
        .add(Linear::new(64, 64)?)
        .add(GELU)
        .add(Linear::new(64, CLASSES)?);
    println!("samples: {n}, parameters: {}", model.num_parameters());

    let mut opt = Adam::adamw(model.parameters(), 0.01, 1e-4);
    for epoch in 1..=160 {
        let mut epoch_loss = 0.0f32;
        let batches = batch_indices(n, 32, true);
        for idx in &batches {
            let idx_i64: Vec<i64> = idx.iter().map(|&i| i as i64).collect();
            let idx_t = Tensor::from_slice(&idx_i64, &[idx.len()])?;
            let xb = x.index_select(0, &idx_t)?;
            let yb: Vec<usize> = idx.iter().map(|&i| y[i]).collect();

            let loss = cross_entropy(&model.forward(&xb)?, &yb)?;
            opt.zero_grad();
            loss.backward()?;
            opt.step()?;
            epoch_loss += loss.item::<f32>()?;
        }
        if epoch % 20 == 0 {
            let logits = no_grad(|| model.forward(&x))?;
            println!(
                "epoch {epoch:3}  loss {:.4}  train acc {:.1}%",
                epoch_loss / batches.len() as f32,
                100.0 * accuracy(&logits, &y)?
            );
        }
    }
    Ok(())
}
