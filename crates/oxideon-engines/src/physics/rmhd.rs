//! Relativistic Magnetohydrodynamics (RMHD) & Ideal Magnetized Plasma Shock Solvers.
//!
//! # Physical & Mathematical Foundations
//!
//! 1. **Covariant RMHD Stress-Energy Tensor ($c = 1$, $\eta^{\mu\nu} = \operatorname{diag}(-1, 1, 1, 1)$)**:
//!    Given rest-mass density $\rho$, thermal pressure $p$, 3-velocity $\mathbf{v}$ ($|\mathbf{v}| < 1$),
//!    and laboratory magnetic field $\mathbf{B}$:
//!    - Lorentz factor: $\Gamma = (1 - \mathbf{v}^2)^{-1/2}$
//!    - 4-velocity: $u^\mu = \Gamma (1, \mathbf{v})$
//!    - Covariant magnetic 4-vector $b^\mu$ orthogonal to $u^\mu$ ($u_\mu b^\mu = 0$):
//!      $$b^0 = \Gamma (\mathbf{v} \cdot \mathbf{B}), \qquad \mathbf{b} = \frac{\mathbf{B}}{\Gamma} + \Gamma (\mathbf{v} \cdot \mathbf{B}) \mathbf{v}$$
//!    - Lorentz-invariant magnetic pressure:
//!      $$b^2 = b_\mu b^\mu = -(b^0)^2 + |\mathbf{b}|^2 = \frac{|\mathbf{B}|^2}{\Gamma^2} + (\mathbf{v} \cdot \mathbf{B})^2 \ge 0$$
//!    - Ideal gas relativistic specific enthalpy: $h = 1 + \frac{\gamma}{\gamma - 1} \frac{p}{\rho}$
//!    - Total RMHD Energy-Momentum Tensor:
//!      $$T^{\mu\nu} = (\rho h + b^2) u^\mu u^\nu + \left(p + \frac{1}{2} b^2\right) \eta^{\mu\nu} - b^\mu b^\nu$$
//!
//! 2. **Relativistic Characteristic Wave Speeds**:
//!    - Relativistic sound speed squared: $c_s^2 = \frac{\gamma p}{\rho h} < 1$
//!    - Relativistic Alfvén speed squared: $c_A^2 = \frac{b^2}{\rho h + b^2} < 1$
//!    - Fast magnetosonic wave speed bound: $c_f^2 = c_s^2 + c_A^2 - c_s^2 c_A^2 < 1$ (strictly causal subluminal propagation).

use crate::error::{Error, Result};

/// Primitive physical state of an ideal relativistic magnetized plasma element.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RmhdPrimitiveState {
    /// Rest-mass density $\rho > 0$.
    pub rho: f64,
    /// Thermal gas pressure $p \ge 0$.
    pub pressure: f64,
    /// 3-velocity $(v_x, v_y, v_z)$ in units of $c$ ($|\mathbf{v}| < 1$).
    pub velocity: [f64; 3],
    /// Laboratory magnetic field $(B_x, B_y, B_z)$.
    pub b_field: [f64; 3],
    /// Adiabatic index $\gamma \in (1, 2]$ (e.g., $4/3$ for ultrarelativistic, $5/3$ for non-relativistic).
    pub gamma_adiabatic: f64,
}

impl RmhdPrimitiveState {
    /// Constructs and validates a primitive RMHD plasma state.
    pub fn new(
        rho: f64,
        pressure: f64,
        velocity: [f64; 3],
        b_field: [f64; 3],
        gamma_adiabatic: f64,
    ) -> Result<Self> {
        if rho <= 0.0 {
            return Err(Error::Msg(format!(
                "Rest-mass density rho must be > 0, got {rho}"
            )));
        }
        if pressure < 0.0 {
            return Err(Error::Msg(format!(
                "Thermal pressure must be >= 0, got {pressure}"
            )));
        }
        let v2 = velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2];
        if v2 >= 1.0 {
            return Err(Error::Msg(format!(
                "Superluminal velocity |v|^2 = {v2} >= 1 violates special relativity"
            )));
        }
        if gamma_adiabatic <= 1.0 || gamma_adiabatic > 2.0 {
            return Err(Error::Msg(format!(
                "Adiabatic index gamma must be in (1.0, 2.0], got {gamma_adiabatic}"
            )));
        }
        Ok(Self {
            rho,
            pressure,
            velocity,
            b_field,
            gamma_adiabatic,
        })
    }

    /// Computes the Lorentz factor $\Gamma = (1 - |\mathbf{v}|^2)^{-1/2}$.
    ///
    /// Trusted computation: the caller must uphold $|\mathbf{v}| < 1$ (as
    /// [`Self::new`] enforces). A superluminal `velocity` set through the pub
    /// fields yields `inf`/`NaN` (and trips this debug assertion) rather than
    /// a silently clamped factor, so broken simulation states stay visible.
    #[inline]
    pub fn lorentz_factor(&self) -> f64 {
        let v2 = self.velocity[0] * self.velocity[0]
            + self.velocity[1] * self.velocity[1]
            + self.velocity[2] * self.velocity[2];
        debug_assert!(v2 < 1.0, "lorentz_factor requires |v|^2 < 1, got {v2}");
        1.0 / (1.0 - v2).sqrt()
    }

    /// Computes thefluid 4-velocity $u^\mu = \Gamma (1, v_x, v_y, v_z)$.
    pub fn four_velocity(&self) -> [f64; 4] {
        let g = self.lorentz_factor();
        [
            g,
            g * self.velocity[0],
            g * self.velocity[1],
            g * self.velocity[2],
        ]
    }

    /// Computes the comoving magnetic 4-vector $b^\mu = (b^0, b^x, b^y, b^z)$.
    pub fn magnetic_four_vector(&self) -> [f64; 4] {
        let g = self.lorentz_factor();
        let v_dot_b = self.velocity[0] * self.b_field[0]
            + self.velocity[1] * self.b_field[1]
            + self.velocity[2] * self.b_field[2];
        let b0 = g * v_dot_b;
        [
            b0,
            self.b_field[0] / g + b0 * self.velocity[0],
            self.b_field[1] / g + b0 * self.velocity[1],
            self.b_field[2] / g + b0 * self.velocity[2],
        ]
    }

    /// Computes the Minkowski inner product $u_\mu b^\mu = -u^0 b^0 + \mathbf{u} \cdot \mathbf{b}$
    /// (identically zero by relativistic ideal MHD orthogonality).
    pub fn orthogonality_residual(&self) -> f64 {
        let u = self.four_velocity();
        let b = self.magnetic_four_vector();
        -u[0] * b[0] + u[1] * b[1] + u[2] * b[2] + u[3] * b[3]
    }

    /// Computes the Lorentz-invariant comoving magnetic energy density $b^2 = b_\mu b^\mu$:
    ///
    /// $$b^2 = \frac{|\mathbf{B}|^2}{\Gamma^2} + (\mathbf{v} \cdot \mathbf{B})^2$$
    pub fn magnetic_energy_squared(&self) -> f64 {
        let b = self.magnetic_four_vector();
        -b[0] * b[0] + b[1] * b[1] + b[2] * b[2] + b[3] * b[3]
    }

    /// Computes the relativistic specific enthalpy $h = 1 + \frac{\gamma}{\gamma - 1} \frac{p}{\rho}$.
    pub fn specific_enthalpy(&self) -> f64 {
        1.0 + (self.gamma_adiabatic / (self.gamma_adiabatic - 1.0)) * (self.pressure / self.rho)
    }

    /// Computes the full $4 \times 4$ symmetric Stress-Energy-Momentum Tensor $T^{\mu\nu}$:
    ///
    /// $$T^{\mu\nu} = (\rho h + b^2) u^\mu u^\nu + \left(p + \frac{1}{2} b^2\right) \eta^{\mu\nu} - b^\mu b^\nu$$
    pub fn stress_energy_tensor(&self) -> [[f64; 4]; 4] {
        let u = self.four_velocity();
        let b = self.magnetic_four_vector();
        let b2 = self.magnetic_energy_squared();
        let w_tot = self.rho * self.specific_enthalpy() + b2;
        let p_tot = self.pressure + 0.5 * b2;
        let eta = [-1.0, 1.0, 1.0, 1.0];

        let mut t = [[0.0; 4]; 4];
        for mu in 0..4 {
            for nu in 0..4 {
                let metric_term = if mu == nu { p_tot * eta[mu] } else { 0.0 };
                t[mu][nu] = w_tot * u[mu] * u[nu] + metric_term - b[mu] * b[nu];
            }
        }
        t
    }

    /// Computes the relativistic Alfvén wave speed $c_A = \sqrt{\frac{b^2}{\rho h + b^2}} \in [0, 1)$.
    pub fn alfven_speed(&self) -> f64 {
        let b2 = self.magnetic_energy_squared();
        let w_tot = self.rho * self.specific_enthalpy() + b2;
        (b2 / w_tot).clamp(0.0, 1.0).sqrt()
    }

    /// Computes the relativistic acoustic sound speed $c_s = \sqrt{\frac{\gamma p}{\rho h}} \in [0, 1)$.
    pub fn sound_speed(&self) -> f64 {
        let h = self.specific_enthalpy();
        ((self.gamma_adiabatic * self.pressure) / (self.rho * h))
            .clamp(0.0, 1.0)
            .sqrt()
    }

    /// Computes the upper bound on the fast magnetosonic wave speed $c_f = \sqrt{c_s^2 + c_A^2 - c_s^2 c_A^2} < 1$.
    pub fn fast_magnetosonic_speed(&self) -> f64 {
        let cs2 = self.sound_speed().powi(2);
        let ca2 = self.alfven_speed().powi(2);
        (cs2 + ca2 - cs2 * ca2).clamp(0.0, 1.0 - 1e-15).sqrt()
    }

    /// Computes the 5-component conserved state vector $U = (D, S_x, S_y, S_z, E)$ for 1D shock evolution along $x$:
    /// - $D = \Gamma \rho$
    /// - $S_i = T^{0i}$
    /// - $E = T^{00}$
    pub fn conserved_vector(&self) -> [f64; 5] {
        let t = self.stress_energy_tensor();
        let d = self.lorentz_factor() * self.rho;
        [d, t[0][1], t[0][2], t[0][3], t[0][0]]
    }

    /// Computes the 5-component hyperbolic flux vector $F^x(U) = (D v_x, T^{x x}, T^{x y}, T^{x z}, T^{0 x})$ along $x$.
    pub fn flux_x(&self) -> [f64; 5] {
        let t = self.stress_energy_tensor();
        let d = self.lorentz_factor() * self.rho;
        [d * self.velocity[0], t[1][1], t[1][2], t[1][3], t[0][1]]
    }

    /// Computes the Harten-Lax-van Leer (HLL) approximate Riemann solver intercell flux
    /// between a left plasma state and a right plasma state across an interface normal to $x$.
    pub fn hll_riemann_flux(left: &Self, right: &Self) -> [f64; 5] {
        let u_l = left.conserved_vector();
        let u_r = right.conserved_vector();
        let f_l = left.flux_x();
        let f_r = right.flux_x();

        // Relativistic velocity-addition formula for bounding signal speeds
        let vx_l = left.velocity[0];
        let cf_l = left.fast_magnetosonic_speed();
        let lambda_min_l = (vx_l - cf_l) / (1.0 - vx_l * cf_l);
        let lambda_max_l = (vx_l + cf_l) / (1.0 + vx_l * cf_l);

        let vx_r = right.velocity[0];
        let cf_r = right.fast_magnetosonic_speed();
        let lambda_min_r = (vx_r - cf_r) / (1.0 - vx_r * cf_r);
        let lambda_max_r = (vx_r + cf_r) / (1.0 + vx_r * cf_r);

        let s_l = lambda_min_l.min(lambda_min_r).min(0.0);
        let s_r = lambda_max_l.max(lambda_max_r).max(0.0);

        if (s_r - s_l).abs() < 1e-14 {
            let mut f = [0.0; 5];
            for k in 0..5 {
                f[k] = 0.5 * (f_l[k] + f_r[k]);
            }
            return f;
        }

        let mut f_hll = [0.0; 5];
        for k in 0..5 {
            f_hll[k] = (s_r * f_l[k] - s_l * f_r[k] + s_l * s_r * (u_r[k] - u_l[k])) / (s_r - s_l);
        }
        f_hll
    }
}
