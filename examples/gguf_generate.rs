//! GGUF text generation: tokenize a prompt with the model's own GGUF vocabulary,
//! greedily decode with the KV cache, and print token ids, timing, and text.
//!
//! Run with:
//! ```bash
//! cargo run --release --example gguf_generate -- model.gguf "prompt" [new_tokens]
//! ```
//!
//! Examples:
//! ```bash
//! cargo run --release --example gguf_generate -- tinyllama.gguf "Hello world" 32
//! ```

use oxideon::nn::{GGUFCausalLM, GGUFTokenizer};
use oxideon::serialize::GGUFFile;
use std::time::Instant;

fn usage() -> ! {
    eprintln!("usage: gguf_generate model.gguf \"prompt\" [new_tokens]");
    eprintln!();
    eprintln!("  model.gguf   path to a GGUF checkpoint readable by oxideon");
    eprintln!("  prompt       text prompt to tokenize with the model's vocabulary");
    eprintln!("  new_tokens   max tokens to generate greedily (default: 24)");
    std::process::exit(2);
}

/// Prints `what: error` and exits with status 1 (a bad model path or an unsupported tokenizer is not a panic).
fn fail(what: &str, e: impl std::fmt::Display) -> ! {
    eprintln!("{what}: {e}");
    std::process::exit(1);
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| usage());
    if path == "-h" || path == "--help" {
        usage();
    }
    let prompt = args.next().unwrap_or_else(|| usage());
    let new_tokens: usize = args
        .next()
        .map(|n| n.parse().unwrap_or_else(|_| usage()))
        .unwrap_or(24);

    let t0 = Instant::now();
    let file = GGUFFile::open(&path).unwrap_or_else(|e| fail("cannot read GGUF file", e));
    let tokenizer = GGUFTokenizer::from_gguf(&file).unwrap_or_else(|e| fail("cannot build tokenizer", e));
    let model = GGUFCausalLM::from_gguf(&file).unwrap_or_else(|e| fail("cannot load model", e));
    let load_secs = t0.elapsed().as_secs_f64();
    println!("loaded {path} in {load_secs:.2}s");

    let ids = tokenizer.encode(&prompt, tokenizer.add_bos_default);
    println!("prompt ids ({} tokens): {ids:?}", ids.len());

    let t1 = Instant::now();
    let all = model
        .generate_greedy(&ids, new_tokens, tokenizer.eos_id.map(i64::from))
        .unwrap_or_else(|e| fail("generation failed", e));
    let gen_secs = t1.elapsed().as_secs_f64();
    let fresh = &all[ids.len()..];
    let stopped_early = fresh.len() < new_tokens;
    println!("new ids ({} tokens): {fresh:?}", fresh.len());
    println!("text: {}", tokenizer.decode(&all));
    println!(
        "generated {} tokens in {gen_secs:.2}s ({:.1} tok/s){}",
        fresh.len(),
        fresh.len() as f64 / gen_secs.max(1e-9),
        if stopped_early {
            " [stopped at EOS]"
        } else {
            ""
        },
    );
}
