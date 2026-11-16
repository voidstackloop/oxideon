//! Learn XOR with a tiny MLP.
//!
//! Run with: `cargo run --release --example xor`

use oxideon::nn::loss::mse_loss;
use oxideon::prelude::*;

fn main() -> Result<()> {
    manual_seed(0);

    let x = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], &[4, 2])?;
    let y = Tensor::from_slice(&[0.0f32, 1.0, 1.0, 0.0], &[4, 1])?;

    let model = Sequential::new()
        .add(Linear::new(2, 8)?)
        .add(Tanh)
        .add(Linear::new(8, 1)?)
        .add(Sigmoid);
    println!("parameters: {}", model.num_parameters());

    let mut opt = Adam::new(model.parameters(), 0.05);
    for epoch in 0..=800 {
        let loss = mse_loss(&model.forward(&x)?, &y)?;
        opt.zero_grad();
        loss.backward()?;
        opt.step()?;
        if epoch % 200 == 0 {
            println!("epoch {epoch:4}  loss {:.6}", loss.item::<f32>()?);
        }
    }

    let pred = no_grad(|| model.forward(&x))?.to_vec::<f32>()?;
    let inputs = x.to_vec::<f32>()?;
    println!("\npredictions:");
    for i in 0..4 {
        println!(
            "  {} XOR {} = {:.4}",
            inputs[2 * i],
            inputs[2 * i + 1],
            pred[i]
        );
    }
    Ok(())
}
