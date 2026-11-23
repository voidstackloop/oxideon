//! `no_std`-compatible (`core`-only, zero-heap) embedded inference kernels.
//!
//! Operates strictly on caller-supplied slices (`&[f32]`, `&[i8]`, `&mut [f32]`)
//! without any dynamic memory allocation (`Vec` / `Box`), suitable for bare-metal
//! microcontrollers (Cortex-M, RISC-V RV32IMAC, ESP32) and safety-critical RTOS targets.

#![allow(clippy::too_many_arguments)]

/// Zero-allocation matrix multiplication `out[m, n] = a[m, k] @ b[k, n]`.
pub fn embedded_matmul_f32(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    debug_assert!(a.len() >= m * k && b.len() >= k * n && out.len() >= m * n);
    for i in 0..m {
        let arow = &a[i * k..(i + 1) * k];
        let orow = &mut out[i * n..(i + 1) * n];
        for v in orow.iter_mut() {
            *v = 0.0;
        }
        for p in 0..k {
            let ap = arow[p];
            let brow = &b[p * n..(p + 1) * n];
            for j in 0..n {
                orow[j] += ap * brow[j];
            }
        }
    }
}

/// Zero-allocation Linear layer `out[m, n] = x[m, k] @ w[n, k]^T + bias[n]`.
pub fn embedded_linear_f32(
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    debug_assert!(x.len() >= m * k && w.len() >= n * k && out.len() >= m * n);
    for i in 0..m {
        let xrow = &x[i * k..(i + 1) * k];
        for j in 0..n {
            let wrow = &w[j * k..(j + 1) * k];
            let mut sum = bias.map_or(0.0, |b| b[j]);
            for p in 0..k {
                sum += xrow[p] * wrow[p];
            }
            out[i * n + j] = sum;
        }
    }
}

/// Zero-allocation Int8 quantized Linear layer `out[m, n] = (x_q[m, k] @ w_q[n, k]^T) * scale + bias`.
pub fn embedded_qlinear_i8(
    x_q: &[i8],
    x_scale: f32,
    w_q: &[i8],
    w_scale: f32,
    bias: Option<&[f32]>,
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    let scale = x_scale * w_scale;
    for i in 0..m {
        let xrow = &x_q[i * k..(i + 1) * k];
        for j in 0..n {
            let wrow = &w_q[j * k..(j + 1) * k];
            let mut acc = 0i32;
            for p in 0..k {
                acc += (xrow[p] as i32) * (wrow[p] as i32);
            }
            out[i * n + j] = (acc as f32) * scale + bias.map_or(0.0, |b| b[j]);
        }
    }
}

/// Zero-allocation in-place ReLU `x[i] = max(x[i], 0)`.
pub fn embedded_relu_inplace_f32(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// Zero-allocation in-place row Softmax using `libm::expf` (no `std` required).
pub fn embedded_softmax_inplace_f32(row: &mut [f32]) {
    if row.is_empty() {
        return;
    }
    let mut max_v = row[0];
    for &v in &row[1..] {
        if v > max_v {
            max_v = v;
        }
    }
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        *v = libm::expf(*v - max_v);
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in row.iter_mut() {
        *v *= inv;
    }
}

/// Zero-allocation LayerNorm over a single row `src[cols] -> dst[cols]`.
pub fn embedded_layer_norm_f32(
    src: &[f32],
    gamma: &[f32],
    beta: &[f32],
    dst: &mut [f32],
    eps: f32,
) {
    let n = src.len();
    if n == 0 {
        return;
    }
    let mut sum = 0.0f32;
    for &v in src {
        sum += v;
    }
    let mean = sum / (n as f32);
    let mut var_sum = 0.0f32;
    for &v in src {
        let d = v - mean;
        var_sum += d * d;
    }
    let inv_std = 1.0 / libm::sqrtf(var_sum / (n as f32) + eps);
    for i in 0..n {
        dst[i] = (src[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_zero_heap_kernels_execute_accurately() {
        let x = [1.0f32, -1.0, 2.0, 0.5];
        let w = [0.5f32, 1.0, -0.5, 2.0];
        let b = [0.1f32, -0.2];
        let mut out = [0.0f32; 4];
        embedded_linear_f32(&x, &w, Some(&b), &mut out, 2, 2, 2);
        // row 0: [1,-1] -> [-0.4, -2.7]; row 1: [2,0.5] -> [1.6, -0.2]
        for (got, want) in out.iter().zip([-0.4f32, -2.7, 1.6, -0.2]) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
        embedded_relu_inplace_f32(&mut out);
        assert_eq!(out, [0.0, 0.0, 1.6, 0.0]);
        embedded_softmax_inplace_f32(&mut out[0..2]);
        assert!((out[0] - 0.5).abs() < 1e-6 && (out[1] - 0.5).abs() < 1e-6);
    }
}
