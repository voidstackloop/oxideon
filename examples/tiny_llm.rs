//! Train a LLaMA-style Causal Transformer Language Model (`CausalLM` with Rotary
//! Position Embeddings, `RMSNorm`, and `SwiGLU` MLP) on a synthetic repeating-sequence
//! task using `AdamW` + `LinearWarmupCosineLr`, then autoregressively generate tokens!
//!
//! Run with: `cargo run --release --example tiny_llm`

use oxideon::nn::loss::cross_entropy_with_smoothing;
use oxideon::prelude::*;

fn main() -> Result<()> {
    manual_seed(2026);

    let vocab_size = 16;
    let d_model = 32;
    let num_heads = 4;
    let num_layers = 2;
    let ffn_hidden = 64;
    let max_seq_len = 32;

    let lm = CausalLM::new(
        vocab_size,
        d_model,
        num_heads,
        num_layers,
        ffn_hidden,
        max_seq_len,
    )?;
    println!("{}", lm.summary());

    // Synthetic pattern: repeating cycle [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, ...]
    let pattern: Vec<i64> = (0..16).map(|i| (i % 8) + 1).collect();
    let seq_len = 15;
    let input_tokens = Tensor::from_slice(&pattern[..seq_len], &[1, seq_len])?;
    let target_tokens: Vec<usize> = pattern[1..=seq_len].iter().map(|&t| t as usize).collect();

    let total_steps = 120;
    let mut opt = Adam::adamw(lm.parameters(), 0.015, 1e-3);
    let mut sched = LinearWarmupCosineLr::new(0.015, 10, total_steps, 1e-4);

    for step in 1..=total_steps {
        let logits = lm.forward(&input_tokens)?.reshape(&[seq_len, vocab_size])?;
        let loss = cross_entropy_with_smoothing(&logits, &target_tokens, 0.02)?;
        opt.zero_grad();
        loss.backward()?;
        clip_grad_norm(&lm.parameters(), 1.0)?;
        opt.step()?;
        let lr = sched.step(&mut opt);

        if step % 30 == 0 {
            let l_val = loss.item::<f64>()?;
            let acc = metrics::accuracy(&logits, &target_tokens)?;
            println!(
                "step {step:3}/{total_steps}  lr={lr:.5}  loss={l_val:.4}  ppl={:.2}  next-token acc={:.1}%",
                metrics::perplexity(l_val),
                100.0 * acc
            );
        }
    }

    let prompt = [1i64, 2, 3];
    let generated = lm.generate(&prompt, 10, 0.0, None, None)?;
    println!("\nPrompt:    {prompt:?}");
    println!("Generated: {generated:?}");
    Ok(())
}
