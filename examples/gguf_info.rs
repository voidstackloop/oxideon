//! GGUF inspector: print non-trivial metadata, tensor list, and model size summary.
//!
//! Run with:
//! ```bash
//! cargo run --release --example gguf_info -- model.gguf
//! ```

use oxideon::serialize::{GGUFFile, GGUFValue};

fn usage() -> ! {
    eprintln!("usage: gguf_info model.gguf");
    eprintln!();
    eprintln!("Prints non-trivial GGUF metadata, every tensor (name, shape, dtype),");
    eprintln!("and a total parameter / size summary.");
    std::process::exit(2);
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| usage());
    if path == "-h" || path == "--help" {
        usage();
    }
    let f = GGUFFile::open(&path).expect("cannot read GGUF file");

    println!("file: {path}");
    println!("metadata ({} keys):", f.metadata.len());
    let mut keys: Vec<_> = f.metadata.iter().collect();
    keys.sort_by_key(|(k, _)| (*k).clone());
    let mut shown = 0usize;
    for (k, v) in keys {
        match v {
            GGUFValue::Uint32(0) | GGUFValue::Bool(false) => {}
            GGUFValue::Float32(x) if *x == 0.0 => {}
            GGUFValue::Array(a) => {
                println!(
                    "  {k} = [array of {}]{}",
                    a.len(),
                    if a.len() <= 8 {
                        format!(" {a:?}")
                    } else {
                        String::new()
                    }
                );
                shown += 1;
            }
            other => {
                println!("  {k} = {other:?}");
                shown += 1;
            }
        }
    }
    println!("  ({shown} non-trivial keys shown)");

    let mut ts: Vec<_> = f.tensors.values().collect();
    ts.sort_by_key(|t| t.name.clone());
    println!("\ntensors ({}):", ts.len());
    let mut total_params: usize = 0;
    for t in ts {
        let params: usize = t.shape.iter().product();
        total_params += params;
        println!(
            "  T {} {:?} {:?} ({} params)",
            t.name, t.shape, t.ggml_type, params
        );
    }
    println!("\ntotal parameters: {total_params}");
    println!(
        "approx fp32 size: {:.1} MB",
        total_params as f64 * 4.0 / 1e6
    );
}
