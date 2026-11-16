//! Python bindings for the analytic FLOP estimates in `oxideon_core::flops`.

use pyo3::prelude::*;

/// FLOPs of an `[m, k] x [k, n]` matrix product.
#[pyfunction]
fn matmul_flops(m: usize, k: usize, n: usize) -> u64 {
    crate::flops::matmul_flops(m, k, n)
}

#[pyfunction]
#[pyo3(signature = (batch, in_features, out_features, bias=true))]
fn linear_flops(batch: usize, in_features: usize, out_features: usize, bias: bool) -> u64 {
    crate::flops::linear_flops(batch, in_features, out_features, bias)
}

#[pyfunction]
#[pyo3(signature = (batch, c_in, c_out, kh, kw, h_out, w_out, groups=1))]
#[allow(clippy::too_many_arguments)]
fn conv2d_flops(batch: usize, c_in: usize, c_out: usize, kh: usize, kw: usize, h_out: usize, w_out: usize, groups: usize) -> u64 {
    crate::flops::conv2d_flops(batch, c_in, c_out, kh, kw, h_out, w_out, groups)
}

#[pyfunction]
#[pyo3(signature = (batch, seq_q, seq_kv, d_model, causal=false))]
fn attention_flops(batch: usize, seq_q: usize, seq_kv: usize, d_model: usize, causal: bool) -> u64 {
    crate::flops::attention_flops(batch, seq_q, seq_kv, d_model, causal)
}

#[pyfunction]
#[pyo3(signature = (batch, seq, layers, d_model, d_ff, vocab, gated_mlp=true))]
fn transformer_flops(batch: usize, seq: usize, layers: usize, d_model: usize, d_ff: usize, vocab: usize, gated_mlp: bool) -> u64 {
    crate::flops::transformer_flops(batch, seq, layers, d_model, d_ff, vocab, gated_mlp)
}

#[pyfunction]
fn training_flops(forward: u64) -> u64 {
    crate::flops::training_flops(forward)
}

#[pyfunction]
fn roofline(peak_flops: f64, bandwidth_bytes: f64, flops: f64, bytes: f64) -> f64 {
    crate::flops::roofline(peak_flops, bandwidth_bytes, flops, bytes)
}

pub fn register_flops_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(matmul_flops, m)?)?;
    m.add_function(wrap_pyfunction!(linear_flops, m)?)?;
    m.add_function(wrap_pyfunction!(conv2d_flops, m)?)?;
    m.add_function(wrap_pyfunction!(attention_flops, m)?)?;
    m.add_function(wrap_pyfunction!(transformer_flops, m)?)?;
    m.add_function(wrap_pyfunction!(training_flops, m)?)?;
    m.add_function(wrap_pyfunction!(roofline, m)?)?;
    Ok(())
}
