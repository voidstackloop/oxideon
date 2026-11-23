//! Runs the operations of `benchmarks/accuracy_vs_pytorch.py` on the inputs that script wrote and stores the results, so Python can
//! compare oxideon and PyTorch against a float64 reference.
//!
//! `cargo run --release --example accuracy_dump -- <dir>`; `<dir>/manifest.txt` has one line per input: `case name d0,d1,.. file`.

use oxideon::nn::functional_ext::{batch_norm, group_norm, layer_norm};
use oxideon::nn::loss::{cosine_similarity, mse_loss};
use oxideon::prelude::*;
use std::collections::HashMap;

fn read_f32(path: &str) -> Vec<f32> {
    std::fs::read(path).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn run(case: &str, t: &HashMap<String, Tensor>) -> Tensor {
    let x = |n: &str| t.get(n).unwrap_or_else(|| panic!("{case}: input {n} missing"));
    match case {
        "sum_1M" | "sum_16M_offset" => x("x").sum().unwrap(),
        "mean_4M_offset" => x("x").mean().unwrap(),
        "sum_axis0" => x("x").sum_axis(0, false).unwrap(),
        "sum_axis1" => x("x").sum_axis(1, false).unwrap(),
        "dot_4M" => x("a").dot(x("b")).unwrap(),
        "norm_4M" => x("x").norm().unwrap(),
        "matmul_512" | "matmul_k8192" => x("a").matmul(x("b")).unwrap(),
        "softmax" => x("x").softmax(-1).unwrap(),
        "log_softmax" => x("x").log_softmax(-1).unwrap(),
        "layer_norm_offset" => {
            let d = *x("x").shape().last().unwrap();
            layer_norm(x("x"), &[d], Some(x("w")), Some(x("b")), 1e-5).unwrap()
        }
        "exp" => x("x").exp().unwrap(),
        "tanh" => x("x").tanh().unwrap(),
        "sigmoid" => x("x").sigmoid().unwrap(),
        "gelu" => x("x").gelu().unwrap(),
        "erf" => x("x").erf().unwrap(),
        "ln" => x("x").ln().unwrap(),
        "sqrt" => x("x").sqrt().unwrap(),
        "conv2d_16ch" | "conv2d_64ch" => x("x").conv2d(x("w"), None, (1, 1), (1, 1), (1, 1)).unwrap(),
        "sin" => x("x").sin().unwrap(),
        "cos" => x("x").cos().unwrap(),
        "log1p" => x("x").log1p().unwrap(),
        "expm1" => x("x").expm1().unwrap(),
        "softplus" => x("x").softplus().unwrap(),
        "silu" => x("x").silu().unwrap(),
        "sinh" => x("x").sinh().unwrap(),
        "cosh" => x("x").cosh().unwrap(),
        "atan" => x("x").atan().unwrap(),
        "rsqrt" => x("x").rsqrt().unwrap(),
        "pow_2_5" => x("x").powf(2.5).unwrap(),
        "var_offset" => x("x").var_axis(1, false, false).unwrap(),
        "batch_norm_offset" => batch_norm(x("x"), None, None, None, None, true, 0.1, 1e-5).unwrap(),
        "group_norm_offset" => group_norm(x("x"), 4, None, None, 1e-5).unwrap(),
        "cosine_sim" => cosine_similarity(x("a"), x("b"), 1, 1e-8).unwrap(),
        "normalize_l2" => x("x").normalize(2.0, 1, 1e-12).unwrap(),
        "mse" => mse_loss(x("a"), x("b")).unwrap(),
        "cumsum_1M" => x("x").cumsum(0).unwrap(),
        "logsumexp" => x("x").logsumexp(-1, false).unwrap(),
        other => panic!("unknown case {other}"),
    }
}

fn main() {
    let Some(dir) = std::env::args().nth(1) else {
        eprintln!("usage: accuracy_dump <dir>   (the directory benchmarks/accuracy_vs_pytorch.py wrote its inputs to)");
        std::process::exit(2);
    };
    // the precision mode comes from OXIDEON_PRECISION (fastest|balanced|highest, default balanced)
    // optional experiment knob: OXI_KCHUNK=<depth> sets the matmul accumulation chain depth (0 = unsplit)
    if let Some(d) = std::env::var("OXI_KCHUNK").ok().and_then(|v| v.parse().ok()) {
        oxideon_core::gemm::set_k_chunk(d);
    }
    let manifest = std::fs::read_to_string(format!("{dir}/manifest.txt")).unwrap();
    let mut cases: Vec<(String, HashMap<String, Tensor>)> = Vec::new();
    for line in manifest.lines().filter(|l| !l.trim().is_empty()) {
        let p: Vec<&str> = line.split_whitespace().collect();
        let shape: Vec<usize> = p[2].split(',').map(|d| d.parse().unwrap()).collect();
        let tensor = Tensor::from_slice(&read_f32(&format!("{dir}/{}", p[3])), &shape).unwrap();
        match cases.iter_mut().find(|c| c.0 == p[0]) {
            Some(c) => {
                c.1.insert(p[1].to_string(), tensor);
            }
            None => cases.push((p[0].to_string(), HashMap::from([(p[1].to_string(), tensor)]))),
        }
    }
    for (name, inputs) in &cases {
        let out = run(name, inputs).contiguous().unwrap().to_vec::<f32>().unwrap();
        let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(format!("{dir}/{name}.oxideon.f32"), bytes).unwrap();
    }
    println!("ran {} cases", cases.len());
}
