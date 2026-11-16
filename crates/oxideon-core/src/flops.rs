//! Analytic FLOP / memory estimates (1 multiply-add = 2 FLOPs) for common layers and whole transformers, plus a
//! [`FlopCounter`] accumulator and a simple roofline helper.

/// FLOPs of `[m, k] x [k, n]`.
pub fn matmul_flops(m: usize, k: usize, n: usize) -> u64 {
    2 * (m * k * n) as u64
}

pub fn linear_flops(batch: usize, in_f: usize, out_f: usize, bias: bool) -> u64 {
    matmul_flops(batch, in_f, out_f) + if bias { (batch * out_f) as u64 } else { 0 }
}

/// 2-D convolution over an `h_out x w_out` output map.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_flops(batch: usize, c_in: usize, c_out: usize, kh: usize, kw: usize, h_out: usize, w_out: usize, groups: usize) -> u64 {
    2 * (batch * c_out * h_out * w_out * (c_in / groups.max(1)) * kh * kw) as u64
}

/// Multi-head attention: Q/K/V/output projections plus the score and value products (`causal` halves the latter two).
pub fn attention_flops(batch: usize, seq_q: usize, seq_kv: usize, d_model: usize, causal: bool) -> u64 {
    let proj = 2 * matmul_flops(batch * seq_q, d_model, d_model) + 2 * matmul_flops(batch * seq_kv, d_model, d_model);
    let mut core = 2 * matmul_flops(batch * seq_q, d_model, seq_kv);
    if causal {
        core /= 2;
    }
    proj + core
}

/// Decoder-only transformer forward pass: attention + gated/plain MLP per layer and the LM head.
pub fn transformer_flops(batch: usize, seq: usize, layers: usize, d_model: usize, d_ff: usize, vocab: usize, gated_mlp: bool) -> u64 {
    let mlp_mats = if gated_mlp { 3 } else { 2 };
    let per_layer = attention_flops(batch, seq, seq, d_model, true) + mlp_mats * matmul_flops(batch * seq, d_model, d_ff);
    layers as u64 * per_layer + matmul_flops(batch * seq, d_model, vocab)
}

/// Training step estimate: forward + backward (about 2x forward).
pub fn training_flops(forward: u64) -> u64 {
    3 * forward
}

/// Attainable FLOP/s under the roofline model for arithmetic intensity `flops / bytes`.
pub fn roofline(peak_flops: f64, bandwidth_bytes: f64, flops: f64, bytes: f64) -> f64 {
    peak_flops.min(bandwidth_bytes * flops / bytes.max(1.0))
}

/// Accumulates named FLOP counts.
#[derive(Default, Debug)]
pub struct FlopCounter {
    entries: Vec<(String, u64)>,
}

impl FlopCounter {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add(&mut self, name: &str, flops: u64) -> &mut Self {
        self.entries.push((name.to_string(), flops));
        self
    }
    pub fn total(&self) -> u64 {
        self.entries.iter().map(|e| e.1).sum()
    }
    /// One line per entry with its share of the total.
    pub fn report(&self) -> String {
        let t = self.total().max(1) as f64;
        self.entries.iter().map(|(n, f)| format!("{n}: {f} ({:.1}%)\n", *f as f64 / t * 100.0)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formulas() {
        assert_eq!(matmul_flops(2, 3, 4), 48);
        assert_eq!(linear_flops(2, 3, 4, true), 48 + 8);
        // 1x1 conv equals a matmul over positions
        assert_eq!(conv2d_flops(1, 8, 16, 1, 1, 4, 4, 1), matmul_flops(16, 8, 16));
        assert_eq!(conv2d_flops(1, 8, 16, 3, 3, 4, 4, 8), 2 * 16 * 16 * 9);
        assert!(attention_flops(1, 8, 8, 16, true) < attention_flops(1, 8, 8, 16, false));
        // dominant term of a big transformer is ~ 2 * params * tokens
        let f = transformer_flops(1, 1, 12, 768, 3072, 50000, false) as f64;
        let params = (12 * (4 * 768 * 768 + 2 * 768 * 3072) + 768 * 50000) as f64;
        assert!((f / (2.0 * params) - 1.0).abs() < 0.01);
        assert_eq!(training_flops(10), 30);
        assert_eq!(roofline(100.0, 10.0, 1.0, 1.0), 10.0);
        assert_eq!(roofline(100.0, 10.0, 1000.0, 1.0), 100.0);
        let mut c = FlopCounter::new();
        c.add("a", 30).add("b", 70);
        assert_eq!(c.total(), 100);
        assert!(c.report().contains("b: 70 (70.0%)"));
    }
}
