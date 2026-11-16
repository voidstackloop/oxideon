//! `torch.nn.init` features missing from the basic initializers: [`calculate_gain`], fan computation,
//! kaiming / xavier with explicit gain, mode and nonlinearity, [`trunc_normal`], [`constant`], [`dirac`], [`sparse`]
//! and a generic [`eye_matrix`].

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::rng;
use crate::tensor::Tensor;

/// Nonlinearity names accepted by [`calculate_gain`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Nonlinearity {
    Linear,
    Sigmoid,
    Tanh,
    Relu,
    /// Leaky ReLU with the given negative slope.
    LeakyRelu(f64),
    Selu,
}

/// Recommended gain of a nonlinearity (`torch.nn.init.calculate_gain`).
pub fn calculate_gain(nl: Nonlinearity) -> f64 {
    match nl {
        Nonlinearity::Linear | Nonlinearity::Sigmoid => 1.0,
        Nonlinearity::Tanh => 5.0 / 3.0,
        Nonlinearity::Relu => 2f64.sqrt(),
        Nonlinearity::LeakyRelu(s) => (2.0 / (1.0 + s * s)).sqrt(),
        Nonlinearity::Selu => 0.75,
    }
}

/// `(fan_in, fan_out)` of a weight of shape `[out, in, k...]` (`torch.nn.init._calculate_fan_in_and_fan_out`).
pub fn fan_in_and_fan_out(shape: &[usize]) -> Result<(usize, usize)> {
    if shape.len() < 2 {
        return Err(Error::Msg("fan in / fan out need a tensor with at least 2 dimensions".into()));
    }
    let receptive: usize = shape[2..].iter().product();
    Ok((shape[1] * receptive, shape[0] * receptive))
}

/// Which fan [`kaiming_uniform_with`] / [`kaiming_normal_with`] preserve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FanMode {
    FanIn,
    FanOut,
}

fn kaiming_std(shape: &[usize], mode: FanMode, nl: Nonlinearity) -> Result<f64> {
    let (fi, fo) = fan_in_and_fan_out(shape)?;
    let fan = if mode == FanMode::FanIn { fi } else { fo };
    if fan == 0 {
        return Err(Error::Msg("kaiming: the selected fan is zero".into()));
    }
    Ok(calculate_gain(nl) / (fan as f64).sqrt())
}

/// Kaiming uniform with explicit fan mode and nonlinearity: `U(-b, b)`, `b = sqrt(3) * gain / sqrt(fan)`.
pub fn kaiming_uniform_with(shape: &[usize], mode: FanMode, nl: Nonlinearity) -> Result<Tensor> {
    let bound = 3f64.sqrt() * kaiming_std(shape, mode, nl)?;
    Tensor::rand_uniform(shape, -bound as f32, bound as f32)
}

/// Kaiming normal with explicit fan mode and nonlinearity: `N(0, gain / sqrt(fan))`.
pub fn kaiming_normal_with(shape: &[usize], mode: FanMode, nl: Nonlinearity) -> Result<Tensor> {
    super::init::normal(shape, 0.0, kaiming_std(shape, mode, nl)?)
}

/// Xavier uniform with a gain, taking the fans from the shape.
pub fn xavier_uniform_gain(shape: &[usize], gain: f64) -> Result<Tensor> {
    let (fi, fo) = fan_in_and_fan_out(shape)?;
    let a = gain * (6.0 / (fi + fo) as f64).sqrt();
    Tensor::rand_uniform(shape, -a as f32, a as f32)
}

/// Xavier normal with a gain, taking the fans from the shape.
pub fn xavier_normal_gain(shape: &[usize], gain: f64) -> Result<Tensor> {
    let (fi, fo) = fan_in_and_fan_out(shape)?;
    super::init::normal(shape, 0.0, gain * (2.0 / (fi + fo) as f64).sqrt())
}

/// Normal samples truncated to `[a, b]` by rejection (`torch.nn.init.trunc_normal_`).
pub fn trunc_normal(shape: &[usize], mean: f64, std: f64, a: f64, b: f64) -> Result<Tensor> {
    if a >= b || a.is_nan() || b.is_nan() || std <= 0.0 {
        return Err(Error::Msg("trunc_normal: needs a < b and a positive std".into()));
    }
    let n: usize = shape.iter().product();
    let (lo, hi) = ((a - mean) / std, (b - mean) / std);
    let mut mass_ok = true;
    let v: Vec<f32> = rng::with_rng(|r| {
        (0..n)
            .map(|_| {
                for _ in 0..10_000 {
                    let z = f64::from(r.normal());
                    if (lo..=hi).contains(&z) {
                        return (mean + std * z) as f32;
                    }
                }
                mass_ok = false;
                (mean + std * (lo + hi) / 2.0) as f32
            })
            .collect()
    });
    if !mass_ok {
        return Err(Error::Msg("trunc_normal: [a, b] holds almost no probability mass".into()));
    }
    Tensor::from_slice(&v, shape)
}

pub fn constant(shape: &[usize], value: f64) -> Result<Tensor> {
    Tensor::full(shape, value, DType::F32, &crate::backend::Device::Cpu)
}

pub fn zeros(shape: &[usize]) -> Result<Tensor> {
    constant(shape, 0.0)
}

pub fn ones(shape: &[usize]) -> Result<Tensor> {
    constant(shape, 1.0)
}

/// Identity on the leading two dimensions of a 2-D shape (`torch.nn.init.eye_`).
pub fn eye_matrix(rows: usize, cols: usize) -> Result<Tensor> {
    let mut v = vec![0.0f32; rows * cols];
    for i in 0..rows.min(cols) {
        v[i * cols + i] = 1.0;
    }
    Tensor::from_slice(&v, &[rows, cols])
}

/// Dirac delta initialization of a convolution weight `[out, in, k...]` (`torch.nn.init.dirac_`): each output channel
/// copies one input channel (repeated across `groups`).
pub fn dirac(shape: &[usize], groups: usize) -> Result<Tensor> {
    if shape.len() < 3 || shape.len() > 5 {
        return Err(Error::Msg("dirac: expects a 3-, 4- or 5-dimensional convolution weight".into()));
    }
    if groups == 0 || shape[0] % groups != 0 {
        return Err(Error::Msg("dirac: the output channels must be divisible by groups".into()));
    }
    let out_per_group = shape[0] / groups;
    let min_dim = out_per_group.min(shape[1]);
    let numel: usize = shape.iter().product();
    let ksize: usize = shape[2..].iter().product();
    let center: usize = shape[2..].iter().fold(0, |acc, &k| acc * k + k / 2);
    let mut v = vec![0.0f32; numel];
    for g in 0..groups {
        for d in 0..min_dim {
            let o = g * out_per_group + d;
            v[(o * shape[1] + d) * ksize + center] = 1.0;
        }
    }
    Tensor::from_slice(&v, shape)
}

/// Sparse matrix `[rows, cols]` with `ceil(sparsity * rows)` zeros per column and `N(0, std)` elsewhere
/// (`torch.nn.init.sparse_`).
pub fn sparse(rows: usize, cols: usize, sparsity: f64, std: f64) -> Result<Tensor> {
    if !(0.0..=1.0).contains(&sparsity) {
        return Err(Error::Msg("sparse: sparsity must lie in [0, 1]".into()));
    }
    let zeros_per_col = (sparsity * rows as f64).ceil() as usize;
    let mut v = vec![0.0f32; rows * cols];
    rng::with_rng(|r| {
        for c in 0..cols {
            let mut rows_idx: Vec<usize> = (0..rows).collect();
            r.shuffle(&mut rows_idx);
            for &row in &rows_idx[zeros_per_col.min(rows)..] {
                v[row * cols + c] = (std * f64::from(r.normal())) as f32;
            }
        }
    });
    Tensor::from_slice(&v, &[rows, cols])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vals(t: &Tensor) -> Vec<f64> {
        t.to_vec::<f64>().unwrap()
    }

    #[test]
    fn gains_and_fans() {
        assert!((calculate_gain(Nonlinearity::Relu) - 2f64.sqrt()).abs() < 1e-12);
        assert!((calculate_gain(Nonlinearity::LeakyRelu(0.2)) - (2.0f64 / 1.04).sqrt()).abs() < 1e-12);
        assert!((calculate_gain(Nonlinearity::Tanh) - 5.0 / 3.0).abs() < 1e-12);
        assert_eq!(fan_in_and_fan_out(&[8, 3, 5, 5]).unwrap(), (75, 200));
        assert!(fan_in_and_fan_out(&[4]).is_err());
    }

    #[test]
    fn kaiming_and_xavier_scales() {
        rng::manual_seed(1);
        let w = kaiming_normal_with(&[400, 100], FanMode::FanIn, Nonlinearity::Relu).unwrap();
        let v = vals(&w);
        let var = v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64;
        assert!((var - 2.0 / 100.0).abs() < 0.002, "{var}");
        let u = vals(&kaiming_uniform_with(&[400, 100], FanMode::FanOut, Nonlinearity::Linear).unwrap());
        let bound = 3f64.sqrt() / 400f64.sqrt();
        assert!(u.iter().all(|x| x.abs() <= bound + 1e-6) && u.iter().cloned().fold(0.0, f64::max) > 0.9 * bound);
        let x = vals(&xavier_uniform_gain(&[50, 150], 2.0).unwrap());
        let a = 2.0 * (6.0f64 / 200.0).sqrt();
        assert!(x.iter().all(|v| v.abs() <= a + 1e-6));
        assert!((vals(&xavier_normal_gain(&[300, 100], 1.0).unwrap()).iter().map(|v| v * v).sum::<f64>() / 30_000.0 - 2.0 / 400.0).abs() < 0.001);
    }

    #[test]
    fn truncation_constants_and_structured_inits() {
        rng::manual_seed(2);
        let t = vals(&trunc_normal(&[5000], 1.0, 0.5, 0.5, 1.5).unwrap());
        assert!(t.iter().all(|v| (0.5..=1.5).contains(v)));
        assert!((t.iter().sum::<f64>() / 5000.0 - 1.0).abs() < 0.02);
        assert!(trunc_normal(&[3], 0.0, 1.0, 50.0, 51.0).is_err());
        assert_eq!(vals(&constant(&[2], 3.5).unwrap()), vec![3.5, 3.5]);
        assert_eq!(vals(&eye_matrix(2, 3).unwrap()), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        // dirac on [2, 2, 3]: channel i maps to channel i at the kernel center
        let d = vals(&dirac(&[2, 2, 3], 1).unwrap());
        assert_eq!(d, vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        // sparse: every column has ceil(0.5 * 6) = 3 zeros
        let s = vals(&sparse(6, 4, 0.5, 0.01).unwrap());
        for c in 0..4 {
            assert_eq!((0..6).filter(|&r| s[r * 4 + c] == 0.0).count(), 3);
        }
    }
}
