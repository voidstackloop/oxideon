//! Parameter initialization schemes.

use crate::error::Result;
use crate::tensor::Tensor;

pub fn uniform(shape: &[usize], low: f32, high: f32) -> Result<Tensor> {
    Tensor::rand_uniform(shape, low, high)
}

pub fn normal(shape: &[usize], mean: f64, std: f64) -> Result<Tensor> {
    Tensor::randn(shape)?.mul_scalar(std)?.add_scalar(mean)
}

/// Glorot/Xavier uniform initialization `U(-a, a)` with `a = sqrt(6 / (fan_in + fan_out))`.
pub fn xavier_uniform(shape: &[usize], fan_in: usize, fan_out: usize) -> Result<Tensor> {
    if fan_in == 0 || fan_out == 0 {
        return Err(crate::error::Error::Msg(format!(
            "xavier_uniform: fans must be positive, got {fan_in}/{fan_out}"
        )));
    }
    let fan_sum = fan_in.checked_add(fan_out).ok_or_else(|| {
        crate::error::Error::Msg(format!(
            "xavier_uniform: fan_in + fan_out overflows ({fan_in} + {fan_out})"
        ))
    })?;
    let a = (6.0 / fan_sum as f32).sqrt();
    uniform(shape, -a, a)
}

/// Glorot/Xavier normal initialization `N(0, sqrt(2 / (fan_in + fan_out)))`.
pub fn xavier_normal(shape: &[usize], fan_in: usize, fan_out: usize) -> Result<Tensor> {
    if fan_in == 0 || fan_out == 0 {
        return Err(crate::error::Error::Msg(format!(
            "xavier_normal: fans must be positive, got {fan_in}/{fan_out}"
        )));
    }
    let fan_sum = fan_in.checked_add(fan_out).ok_or_else(|| {
        crate::error::Error::Msg(format!(
            "xavier_normal: fan_in + fan_out overflows ({fan_in} + {fan_out})"
        ))
    })?;
    let std = (2.0 / fan_sum as f64).sqrt();
    normal(shape, 0.0, std)
}

/// He/Kaiming uniform initialization `U(-a, a)` with `a = sqrt(6 / fan_in)`.
pub fn kaiming_uniform(shape: &[usize], fan_in: usize) -> Result<Tensor> {
    if fan_in == 0 {
        return Err(crate::error::Error::Msg(
            "kaiming_uniform: fan_in must be positive".into(),
        ));
    }
    let a = (6.0 / fan_in as f32).sqrt();
    uniform(shape, -a, a)
}

/// He/Kaiming normal initialization `N(0, sqrt(2 / fan_in))`.
pub fn kaiming_normal(shape: &[usize], fan_in: usize) -> Result<Tensor> {
    if fan_in == 0 {
        return Err(crate::error::Error::Msg(
            "kaiming_normal: fan_in must be positive".into(),
        ));
    }
    let std = (2.0 / fan_in as f64).sqrt();
    normal(shape, 0.0, std)
}

/// LeCun normal initialization `N(0, sqrt(1 / fan_in))` (for `SELU` self-normalizing networks).
pub fn lecun_normal(shape: &[usize], fan_in: usize) -> Result<Tensor> {
    let std = (1.0 / fan_in.max(1) as f64).sqrt();
    normal(shape, 0.0, std)
}

/// Orthogonal matrix initialization (`torch.nn.init.orthogonal_`) with gain `gain` via QR decomposition.
pub fn orthogonal(rows: usize, cols: usize, gain: f64) -> Result<Tensor> {
    let (m, n) = if rows >= cols {
        (rows, cols)
    } else {
        (cols, rows)
    };
    let z = Tensor::randn(&[m, n])?;
    let (q, _r) = z.qr()?;
    let q_shaped = if rows >= cols {
        q
    } else {
        q.t()?.contiguous()?
    };
    q_shaped.mul_scalar(gain)
}
