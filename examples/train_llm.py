#!/usr/bin/env python3
"""
End-to-End Large Language Model (LLM) Creation & Training in Python with Oxideon (`import oxideon as ox`).
Python acts purely as the high-level script while BPE tokenization, next-token prediction
pre-training (`fused_cross_entropy_with_z_loss` + `AdamW` + gradient clipping), and KVCache
autoregressive decoding run 100% in native Rust with the Python GIL released.
"""

import oxideon as ox


def main() -> None:
    ox.manual_seed(42)
    print("=========================================================================")
    print("  Oxideon Python -> Native Rust LLM Creation, Training & Generation")
    print("=========================================================================\n")

    qa_samples = [
        "Q: What is Oxideon? A: Oxideon is a frontier deep learning library in Rust.",
        "Q: Who executes the LLM math? A: Rust executes all CPU SIMD and GPU kernels.",
        "Q: Can Python train an Oxideon LLM? A: Yes, the entire training loop runs in Rust.",
    ]
    corpus = "\n".join(qa_samples)

    # 1. Train Byte-Level BPE Tokenizer in Rust
    tokenizer = ox.BpeTokenizer.train(corpus, 310)
    vocab_size = tokenizer.vocab_size()
    sequences = [tokenizer.encode_with_special(s, True, True) for s in qa_samples]
    print(f"[1] BPE Tokenizer trained in Rust: vocab_size={vocab_size}, sequences={len(sequences)}")

    # 2. Instantiate RoPE + RMSNorm + SwiGLU CausalLM & Pre-Train 100% in Rust
    llm = ox.CausalLM(vocab_size, 48, 4, 2, 128, 96)
    print(f"[2] Initialized CausalLM ({llm.num_parameters()} parameters). Offloading 35 epochs to Rust...")
    losses, ppls = llm.train_ntp_epochs_in_rust(
        sequences,
        epochs=35,
        lr=1.5e-2,
        weight_decay=1e-3,
        max_grad_norm=1.0,
        z_loss_weight=1e-4,
    )
    print(
        f"    Epoch  1: Loss = {losses[0]:.4f} | Perplexity = {ppls[0]:6.2f}\n"
        f"    Epoch 18: Loss = {losses[17]:.4f} | Perplexity = {ppls[17]:6.2f}\n"
        f"    Epoch 35: Loss = {losses[-1]:.4f} | Perplexity = {ppls[-1]:6.2f}\n"
    )

    # 3. Generate Answers Using Stateful KVCache in Rust
    print("[3] Autoregressive Generation from Trained LLM:")
    for prompt in [
        "Q: What is Oxideon? A: ",
        "Q: Who executes the LLM math? A: ",
        "Q: Can Python train an Oxideon LLM? A: ",
    ]:
        prompt_ids = tokenizer.encode_with_special(prompt, True, False)
        gen_ids = llm.generate(prompt_ids, 40, temperature=0.0)
        # Stop at EOS token (258)
        clean_ids = []
        for tid in gen_ids:
            if tid == 258:
                break
            clean_ids.append(tid)
        print(f"    Prompt : {prompt}")
        print(f"    Output : {tokenizer.decode(clean_ids)}\n")


if __name__ == "__main__":
    main()
