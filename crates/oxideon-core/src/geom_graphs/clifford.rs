//! Clifford Geometric Algebra $Cl(3, 0)$ & Multivector Neural Layer.
//!
//! Features:
//! - Complete 8-dimensional graded multivector algebra in $Cl(3, 0)$:
//!   - Grade 0 (Scalar): $1$
//!   - Grade 1 (Vectors): $e_1, e_2, e_3$
//!   - Grade 2 (Bivectors): $e_{12}, e_{23}, e_{31}$
//!   - Grade 3 (Pseudoscalar / Trivector): $e_{123} = e_1 e_2 e_3$
//! - Universal geometric product $ab = a \cdot b + a \wedge b$.
//! - Exact multivector reversion $A^\dagger$ and rotor group action $R v R^\dagger$
//!   for continuous 3D rotations without coordinate singularities or gimbal lock.
//! - Equivariant Clifford linear layer [`CliffordLinear`] operating on multivector channels.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// 8-dimensional Multivector in 3D Euclidean Clifford Algebra $Cl(3, 0)$.
///
/// Component order: `[s, e1, e2, e3, e12, e23, e31, e123]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Multivector3D {
    pub components: [f64; 8],
}

impl Default for Multivector3D {
    fn default() -> Self {
        Self {
            components: [0.0; 8],
        }
    }
}

impl Multivector3D {
    /// Creates a multivector from an 8-element component array.
    pub fn new(components: [f64; 8]) -> Self {
        Self { components }
    }

    /// Pure scalar multivector (Grade 0).
    pub fn scalar(s: f64) -> Self {
        let mut c = [0.0; 8];
        c[0] = s;
        Self { components: c }
    }

    /// Pure vector multivector (Grade 1): $v_1 e_1 + v_2 e_2 + v_3 e_3$.
    pub fn vector(x: f64, y: f64, z: f64) -> Self {
        let mut c = [0.0; 8];
        c[1] = x;
        c[2] = y;
        c[3] = z;
        Self { components: c }
    }

    /// Pure bivector multivector (Grade 2): $b_{12} e_{12} + b_{23} e_{23} + b_{31} e_{31}$.
    pub fn bivector(xy: f64, yz: f64, zx: f64) -> Self {
        let mut c = [0.0; 8];
        c[4] = xy;
        c[5] = yz;
        c[6] = zx;
        Self { components: c }
    }

    /// Pure pseudoscalar / trivector multivector (Grade 3): $p e_{123}$.
    pub fn trivector(p: f64) -> Self {
        let mut c = [0.0; 8];
        c[7] = p;
        Self { components: c }
    }

    /// Creates a unit Rotor for rotation by `angle` radians around unit `axis` $[u_x, u_y, u_z]$.
    ///
    /// The rotation plane bivector dual to axis is $B = u_x e_{23} + u_y e_{31} + u_z e_{12}$.
    /// $$R = \cos(\theta / 2) - \sin(\theta / 2) B$$
    pub fn rotor(axis: [f64; 3], angle: f64) -> Self {
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2])
            .sqrt()
            .max(1e-12);
        let ux = axis[0] / norm;
        let uy = axis[1] / norm;
        let uz = axis[2] / norm;

        let half = 0.5 * angle;
        let cos_h = half.cos();
        let sin_h = half.sin();

        let mut c = [0.0; 8];
        c[0] = cos_h;
        // Bivectors: e12 is dual to uz, e23 dual to ux, e31 dual to uy
        c[4] = -sin_h * uz;
        c[5] = -sin_h * ux;
        c[6] = -sin_h * uy;

        Self { components: c }
    }

    /// Multivector reversion $(\cdot)^\dagger$:
    /// Reverses the product order of basis vectors for each grade:
    /// - Grade 0 (scalar): $+1$
    /// - Grade 1 (vectors): $+1$
    /// - Grade 2 (bivectors): $-1$
    /// - Grade 3 (trivector): $-1$
    pub fn reverse(&self) -> Self {
        let mut c = self.components;
        c[4] = -c[4];
        c[5] = -c[5];
        c[6] = -c[6];
        c[7] = -c[7];
        Self { components: c }
    }

    /// Full Clifford geometric product $A B$ in $Cl(3, 0)$.
    pub fn geometric_product(&self, other: &Self) -> Self {
        let a = self.components;
        let b = other.components;

        let mut out = [0.0; 8];

        // Grade 0 (Scalar)
        out[0] = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
            - a[4] * b[4]
            - a[5] * b[5]
            - a[6] * b[6]
            - a[7] * b[7];

        // Grade 1 (e1, e2, e3)
        out[1] = a[0] * b[1] + a[1] * b[0] - a[2] * b[4] + a[4] * b[2] + a[3] * b[6]
            - a[6] * b[3]
            - a[5] * b[7]
            - a[7] * b[5];

        out[2] = a[0] * b[2] + a[2] * b[0] - a[3] * b[5] + a[5] * b[3] + a[1] * b[4]
            - a[4] * b[1]
            - a[6] * b[7]
            - a[7] * b[6];

        out[3] = a[0] * b[3] + a[3] * b[0] - a[1] * b[6] + a[6] * b[1] + a[2] * b[5]
            - a[5] * b[2]
            - a[4] * b[7]
            - a[7] * b[4];

        // Grade 2 (e12, e23, e31)
        out[4] = a[0] * b[4] + a[4] * b[0] + a[1] * b[2] - a[2] * b[1] - a[3] * b[7] - a[7] * b[3]
            + a[5] * b[6]
            - a[6] * b[5];

        out[5] = a[0] * b[5] + a[5] * b[0] + a[2] * b[3] - a[3] * b[2] - a[1] * b[7] - a[7] * b[1]
            + a[6] * b[4]
            - a[4] * b[6];

        out[6] = a[0] * b[6] + a[6] * b[0] + a[3] * b[1] - a[1] * b[3] - a[2] * b[7] - a[7] * b[2]
            + a[4] * b[5]
            - a[5] * b[4];

        // Grade 3 (Pseudoscalar e123)
        out[7] = a[0] * b[7]
            + a[7] * b[0]
            + a[1] * b[5]
            + a[5] * b[1]
            + a[2] * b[6]
            + a[6] * b[2]
            + a[3] * b[4]
            + a[4] * b[3];

        Self { components: out }
    }

    /// Sandwich product transformation by rotor $R$:
    /// $$M' = R M R^\dagger$$
    pub fn sandwich(&self, rotor: &Self) -> Self {
        let rev = rotor.reverse();
        rotor.geometric_product(self).geometric_product(&rev)
    }

    /// Multivector Euclidean magnitude squared $\sum_{i=0}^7 c_i^2$.
    pub fn norm_squared(&self) -> f64 {
        self.components.iter().map(|&x| x * x).sum()
    }

    /// Converts into a 1D tensor of shape `[8]`.
    pub fn to_tensor(&self) -> Result<Tensor> {
        Tensor::from_vec(self.components.to_vec(), &[8])
    }
}

/// Clifford Geometric Linear Layer over multivector channels.
///
/// Maps $C_{\text{in}}$ multivector channels to $C_{\text{out}}$ multivector channels:
/// Input shape: `[Batch, C_in, 8]`. Output shape: `[Batch, C_out, 8]`.
pub struct CliffordLinear {
    pub in_channels: usize,
    pub out_channels: usize,
    pub weights: Tensor,      // [out_channels, in_channels, 8]
    pub bias: Option<Tensor>, // [out_channels, 8]
}

impl CliffordLinear {
    /// Creates a new `CliffordLinear` layer.
    pub fn new(in_channels: usize, out_channels: usize, has_bias: bool) -> Result<Self> {
        let bound = 1.0 / (8.0 * in_channels as f64).sqrt();
        let weights = Tensor::randn(&[out_channels, in_channels, 8])?.mul_scalar(bound)?;
        let bias = if has_bias {
            Some(Tensor::from_vec(
                vec![0.0f64; out_channels * 8],
                &[out_channels, 8],
            )?)
        } else {
            None
        };

        Ok(Self {
            in_channels,
            out_channels,
            weights,
            bias,
        })
    }

    /// Forward pass through Clifford multivector channels.
    /// Input `x`: `[Batch, in_channels, 8]`.
    /// Output: `[Batch, out_channels, 8]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.ndim() != 3 || x.shape()[2] != 8 {
            return Err(Error::Msg(format!(
                "CliffordLinear expects [Batch, InChannels, 8], got {:?}",
                x.shape()
            )));
        }

        let batch = x.shape()[0];
        let in_c = x.shape()[1];
        if in_c != self.in_channels {
            return Err(Error::Msg(format!(
                "Input channels {} != layer in_channels {}",
                in_c, self.in_channels
            )));
        }

        let x_data = x.to_vec::<f64>()?;
        let w_data = self.weights.to_vec::<f64>()?;
        let b_data = match &self.bias {
            Some(b) => Some(b.to_vec::<f64>()?),
            None => None,
        };
        // Channel agreement (weight/bias tensors vs tamperable out_channels).
        let w_need = self
            .out_channels
            .checked_mul(in_c)
            .and_then(|t| t.checked_mul(8))
            .ok_or_else(|| {
                Error::Msg(format!(
                    "CliffordLinear weights size {} * {in_c} * 8 overflows",
                    self.out_channels
                ))
            })?;
        if w_data.len() != w_need {
            return Err(Error::ShapeMismatch {
                op: "CliffordLinear::forward",
                expected: vec![self.out_channels, in_c, 8],
                got: vec![w_data.len()],
            });
        }
        if let Some(b) = &b_data {
            let b_need = self.out_channels.checked_mul(8).ok_or_else(|| {
                Error::Msg(format!(
                    "CliffordLinear bias size {} * 8 overflows",
                    self.out_channels
                ))
            })?;
            if b.len() != b_need {
                return Err(Error::ShapeMismatch {
                    op: "CliffordLinear::forward",
                    expected: vec![self.out_channels, 8],
                    got: vec![b.len()],
                });
            }
        }

        let mut out = vec![0.0f64; batch * self.out_channels * 8];

        for b in 0..batch {
            for oc in 0..self.out_channels {
                let mut accum = Multivector3D::default();

                for ic in 0..in_c {
                    let mut x_mv_c = [0.0; 8];
                    let x_offset = (b * in_c + ic) * 8;
                    x_mv_c.copy_from_slice(&x_data[x_offset..x_offset + 8]);
                    let x_mv = Multivector3D::new(x_mv_c);

                    let mut w_mv_c = [0.0; 8];
                    let w_offset = (oc * in_c + ic) * 8;
                    w_mv_c.copy_from_slice(&w_data[w_offset..w_offset + 8]);
                    let w_mv = Multivector3D::new(w_mv_c);

                    let prod = w_mv.geometric_product(&x_mv);
                    for k in 0..8 {
                        accum.components[k] += prod.components[k];
                    }
                }

                if let Some(ref bias_vec) = b_data {
                    for k in 0..8 {
                        accum.components[k] += bias_vec[oc * 8 + k];
                    }
                }

                let out_offset = (b * self.out_channels + oc) * 8;
                out[out_offset..out_offset + 8].copy_from_slice(&accum.components);
            }
        }

        Tensor::from_vec(out, &[batch, self.out_channels, 8])
    }
}
