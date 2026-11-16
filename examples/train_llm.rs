//! Complete End-to-End LLM Creation, Pre-Training, Alignment & Inference in Oxideon.
//!
//! Run with:
//! ```bash
//! cargo run --release --example train_llm
//! ```

use oxideon::nn::loss::simpo_loss;
use oxideon::prelude::*;

fn sequence_log_prob(model: &CausalLM, seq: &[usize]) -> Result<Tensor> {
    let t_len = seq.len() - 1;
    let inp_ids: Vec<i64> = seq[..t_len].iter().map(|&x| x as i64).collect();
    let tgt_ids: Vec<i64> = seq[1..].iter().map(|&x| x as i64).collect();
    let inp_t = Tensor::from_slice(&inp_ids, &[1, t_len])?;
    let tgt_t = Tensor::from_slice(&tgt_ids, &[t_len, 1])?;

    let logits = model.forward(&inp_t)?.reshape(&[t_len, model.vocab_size])?;
    let log_probs = logits.log_softmax(-1)?;
    let selected = log_probs.gather(1, &tgt_t)?;
    selected.sum()
}

fn main() -> Result<()> {
    manual_seed(42);
    println!("==================================================================");
    println!("  Oxideon End-to-End Large Language Model (LLM) Creation Pipeline");
    println!("==================================================================\n");

    // -------------------------------------------------------------------------
    // Stage 1: Train Byte-Level BPE Tokenizer on Domain Corpus
    // -------------------------------------------------------------------------
    let qa_samples = [
        "Q: What is Oxideon? A: Oxideon is a frontier deep learning and tensor library in Rust.",
        "Q: How does Oxideon run fast? A: Fast execution comes from fused SIMD kernels and zero-copy views.",
        "Q: Can Oxideon train LLMs? A: Yes, Oxideon trains CausalLM and DeepSeek MoE models.",
    ];
    let corpus = qa_samples.join("\n");
    let tokenizer = BpeTokenizer::train(&corpus, 320);
    let vocab_size = tokenizer.vocab_size();

    let encoded_samples: Vec<Vec<usize>> = qa_samples
        .iter()
        .map(|s| tokenizer.encode_with_special(s, true, true))
        .collect();

    println!(
        "[Stage 1] BPE Tokenizer trained on domain corpus:\n    Learned vocab_size = {} (256 base bytes + 4 special + {} BPE merges)\n    Sample 0 compression: {} UTF-8 bytes -> {} BPE tokens",
        vocab_size,
        vocab_size - 260,
        qa_samples[0].len(),
        encoded_samples[0].len()
    );

    // -------------------------------------------------------------------------
    // Stage 2: Pre-Train LLaMA-style CausalLM (RoPE + RMSNorm + SwiGLU)
    // -------------------------------------------------------------------------
    let max_seq_len = 96;
    let model = CausalLM::new(vocab_size, 48, 4, 2, 128, max_seq_len)?;
    let params = model.parameters();
    let mut optimizer = Adam::adamw(params.clone(), 1.5e-2, 1e-3);
    let epochs = 40;
    let total_steps = epochs * encoded_samples.len();
    let mut scheduler = OneCycleLR::new(1.5e-2, total_steps, 0.2, 5.0, 20.0);

    let mut initial_loss = 0.0f64;
    let mut final_loss = 0.0f64;

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0f64;
        for seq in &encoded_samples {
            let t_len = seq.len() - 1;
            let inp_ids: Vec<i64> = seq[..t_len].iter().map(|&x| x as i64).collect();
            let tgt_ids: Vec<usize> = seq[1..].to_vec();
            let inp_t = Tensor::from_slice(&inp_ids, &[1, t_len])?;

            optimizer.zero_grad();
            let logits = model.forward(&inp_t)?.reshape(&[t_len, vocab_size])?;
            // Fused online log-sum-exp Cross-Entropy + PaLM/DeepSeek z-loss
            let loss = fused_cross_entropy_with_z_loss(&logits, &tgt_ids, 1e-4)?;
            epoch_loss += loss.item::<f64>()?;

            loss.backward()?;
            clip_grad_norm(&params, 1.0)?;
            optimizer.step()?;
            scheduler.step(&mut optimizer);
        }
        epoch_loss /= encoded_samples.len() as f64;
        if epoch == 0 {
            initial_loss = epoch_loss;
        }
        if epoch + 1 == epochs {
            final_loss = epoch_loss;
        }
        if (epoch + 1) % 10 == 0 || epoch == 0 {
            println!(
                "    [Pre-Train Epoch {:2}/{}] NLL+z_loss = {:.4} | Perplexity = {:6.2} | LR = {:.5}",
                epoch + 1,
                epochs,
                epoch_loss,
                epoch_loss.exp(),
                optimizer.learning_rate()
            );
        }
    }
    println!(
        "[Stage 2] CausalLM ({} params) pre-training complete: loss {:.4} -> {:.4} (PPL {:.2} -> {:.2})\n",
        model.num_parameters(),
        initial_loss,
        final_loss,
        initial_loss.exp(),
        final_loss.exp()
    );

    // -------------------------------------------------------------------------
    // Stage 3: Preference Alignment with SimPO (Simple Preference Optimization)
    // -------------------------------------------------------------------------
    let chosen_text =
        "Q: What is Oxideon? A: Oxideon is a frontier deep learning and tensor library in Rust.";
    let rejected_text = "Q: What is Oxideon? A: Oxideon is a slow spreadsheet script.";
    let chosen_seq = tokenizer.encode_with_special(chosen_text, true, true);
    let rejected_seq = tokenizer.encode_with_special(rejected_text, true, true);

    let mut align_opt = Adam::adamw(params.clone(), 3e-4, 0.0);
    let mut simpo_initial = 0.0f64;
    let mut simpo_final = 0.0f64;

    for step in 0..8 {
        align_opt.zero_grad();
        let lp_chosen = sequence_log_prob(&model, &chosen_seq)?
            .div_scalar((chosen_seq.len() - 1) as f64)?
            .reshape(&[1])?;
        let lp_rejected = sequence_log_prob(&model, &rejected_seq)?
            .div_scalar((rejected_seq.len() - 1) as f64)?
            .reshape(&[1])?;
        let loss = simpo_loss(&lp_chosen, &lp_rejected, 1.0, 6.5)?;
        let lv = loss.item::<f64>()?;
        if step == 0 {
            simpo_initial = lv;
        }
        if step == 7 {
            simpo_final = lv;
        }
        loss.backward()?;
        clip_grad_norm(&params, 1.0)?;
        align_opt.step()?;
    }
    println!(
        "[Stage 3] SimPO Alignment complete (8 steps): preference loss {:.4} -> {:.4}\n",
        simpo_initial, simpo_final
    );

    // -------------------------------------------------------------------------
    // Stage 4: Autoregressive Generation with Stateful KVCache & Decode
    // -------------------------------------------------------------------------
    println!("[Stage 4] Autoregressive Generation (Greedy KVCache Decoding):");
    for prompt_str in [
        "Q: What is Oxideon? A: ",
        "Q: How does Oxideon run fast? A: ",
        "Q: Can Oxideon train LLMs? A: ",
    ] {
        let prompt_ids_usize = tokenizer.encode_with_special(prompt_str, true, false);
        let prompt_ids_i64: Vec<i64> = prompt_ids_usize.iter().map(|&x| x as i64).collect();
        let gen_i64 = model.generate(&prompt_ids_i64, 50, 0.0, None, None)?;

        // Stop at EOS_TOKEN_ID if generated
        let mut gen_usize = Vec::with_capacity(gen_i64.len());
        for id in gen_i64 {
            let u = id as usize;
            if u == EOS_TOKEN_ID {
                break;
            }
            gen_usize.push(u);
        }
        let decoded = tokenizer.decode(&gen_usize);
        println!("    Prompt : {prompt_str}");
        println!("    Output : {decoded}\n");
    }

    // -------------------------------------------------------------------------
    // Stage 5: DeepSeek-V3 Frontier MoE + Multi-Token Prediction (MTP) Step
    // -------------------------------------------------------------------------
    let deepseek = DeepSeekMoELM::new(vocab_size, 32, 2, 4, 8, 64, max_seq_len, 1, 4, 2, 2)?;
    let sample_ids: Vec<i64> = encoded_samples[0].iter().map(|&x| x as i64).collect();
    let ds_inp = Tensor::from_slice(&sample_ids, &[1, sample_ids.len()])?;
    let (ds_main, ds_mtp, ds_aux) = deepseek.forward_with_mtp(&ds_inp)?;
    println!(
        "[Stage 5] DeepSeek-V3 MoE + Multi-Token Prediction (MTP) verified on same BPE vocab:\n    Params = {}, main logits = {:?}, MTP t+1 = {:?}, MTP t+2 = {:?}, MoE balance loss = {:.4}",
        deepseek.num_parameters(),
        ds_main.shape(),
        ds_mtp[0].shape(),
        ds_mtp[1].shape(),
        ds_aux.item::<f32>()?
    );

    Ok(())
}
