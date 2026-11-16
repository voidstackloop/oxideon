//! Geometric Mechanics & Lie-Poisson Hamiltonian Systems on $SE(3)$.
//!
//! Implements non-canonical Hamiltonian mechanics on the Lie algebra dual $\mathfrak{se}(3)^*$:
//! - Lie algebra $\mathfrak{se}(3) \cong \mathbb{R}^3 \ltimes \mathbb{R}^3$ with bracket $[(v_1, \omega_1), (v_2, \omega_2)]$.
//! - Dual momentum space $\mu = (\mathbf{p}, \boldsymbol{\Pi}) \in \mathfrak{se}(3)^*$ (linear & angular momentum).
//! - Coadjoint action $\operatorname{ad}^*_{(v, \omega)} (\mathbf{p}, \boldsymbol{\Pi}) = (\mathbf{p} \times \omega, \boldsymbol{\Pi} \times \omega + \mathbf{p} \times v)$.
//! - Lie-Poisson dynamics $\dot{\mu} = -\operatorname{ad}^*_{\nabla H(\mu)} \mu$.
//! - Exact Casimir invariant preservation: $C_1(\mu) = \|\mathbf{p}\|^2$ and $C_2(\mu) = \mathbf{p} \cdot \boldsymbol{\Pi}$.

use crate::error::{Error, Result};

/// 3D Vector cross product: $a \times b$.
#[inline]
pub fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// 3D Vector dot product: $a \cdot b$.
#[inline]
pub fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Rigid body inertia tensor and total mass.
#[derive(Debug, Clone)]
pub struct RigidBodyInertia {
    pub mass: f64,
    /// Principal moments of inertia $[I_{xx}, I_{yy}, I_{zz}]$.
    pub principal_moments: [f64; 3],
}

impl RigidBodyInertia {
    pub fn new(mass: f64, principal_moments: [f64; 3]) -> Result<Self> {
        if mass <= 0.0 {
            return Err(Error::Msg("mass must be positive".into()));
        }
        if principal_moments[0] <= 0.0 || principal_moments[1] <= 0.0 || principal_moments[2] <= 0.0
        {
            return Err(Error::Msg("principal moments must be positive".into()));
        }
        // Triangle inequality for physical moments of inertia: I_x + I_y >= I_z etc.
        let (ix, iy, iz) = (
            principal_moments[0],
            principal_moments[1],
            principal_moments[2],
        );
        if ix + iy < iz || ix + iz < iy || iy + iz < ix {
            return Err(Error::Msg(
                "principal moments violate triangle inequality".into(),
            ));
        }
        Ok(Self {
            mass,
            principal_moments,
        })
    }
}

/// State on the dual Lie algebra $\mathfrak{se}(3)^*$: linear momentum $\mathbf{p}$ and angular momentum $\boldsymbol{\Pi}$.
#[derive(Debug, Clone)]
pub struct LiePoissonState {
    pub linear_momentum: [f64; 3],
    pub angular_momentum: [f64; 3],
}

impl LiePoissonState {
    pub fn new(linear_momentum: [f64; 3], angular_momentum: [f64; 3]) -> Self {
        Self {
            linear_momentum,
            angular_momentum,
        }
    }

    /// First Casimir invariant $C_1(\mu) = \|\mathbf{p}\|^2$. Constant along all coadjoint orbits!
    pub fn casimir_c1(&self) -> f64 {
        dot3(self.linear_momentum, self.linear_momentum)
    }

    /// Second Casimir invariant $C_2(\mu) = \mathbf{p} \cdot \boldsymbol{\Pi}$. Constant along all coadjoint orbits!
    pub fn casimir_c2(&self) -> f64 {
        dot3(self.linear_momentum, self.angular_momentum)
    }

    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.linear_momentum[0],
            self.linear_momentum[1],
            self.linear_momentum[2],
            self.angular_momentum[0],
            self.angular_momentum[1],
            self.angular_momentum[2],
        ]
    }
}

/// Lie-Poisson Dynamical System on $SE(3)$.
#[derive(Debug, Clone)]
pub struct LiePoissonSystem {
    pub inertia: RigidBodyInertia,
}

impl LiePoissonSystem {
    pub fn new(inertia: RigidBodyInertia) -> Self {
        Self { inertia }
    }

    /// Computes the Hamiltonian energy $H(\mu) = \frac{1}{2m} \|\mathbf{p}\|^2 + \frac{1}{2} \boldsymbol{\Pi}^T I^{-1} \boldsymbol{\Pi}$.
    pub fn hamiltonian(&self, state: &LiePoissonState) -> f64 {
        let p_sq = dot3(state.linear_momentum, state.linear_momentum);
        let t_trans = 0.5 * p_sq / self.inertia.mass;

        let (pi_x, pi_y, pi_z) = (
            state.angular_momentum[0],
            state.angular_momentum[1],
            state.angular_momentum[2],
        );
        let (ix, iy, iz) = (
            self.inertia.principal_moments[0],
            self.inertia.principal_moments[1],
            self.inertia.principal_moments[2],
        );
        let t_rot = 0.5 * (pi_x * pi_x / ix + pi_y * pi_y / iy + pi_z * pi_z / iz);

        t_trans + t_rot
    }

    /// Gradient $\nabla H(\mu) = (\mathbf{v}, \boldsymbol{\omega}) \in \mathfrak{se}(3)$.
    pub fn velocities(&self, state: &LiePoissonState) -> ([f64; 3], [f64; 3]) {
        let inv_m = 1.0 / self.inertia.mass;
        let v = [
            state.linear_momentum[0] * inv_m,
            state.linear_momentum[1] * inv_m,
            state.linear_momentum[2] * inv_m,
        ];

        let omega = [
            state.angular_momentum[0] / self.inertia.principal_moments[0],
            state.angular_momentum[1] / self.inertia.principal_moments[1],
            state.angular_momentum[2] / self.inertia.principal_moments[2],
        ];

        (v, omega)
    }

    /// Computes the Lie-Poisson vector field $\dot{\mu} = -\operatorname{ad}^*_{(\mathbf{v}, \boldsymbol{\omega})} \mu$:
    /// $$\dot{\mathbf{p}} = \mathbf{p} \times \boldsymbol{\omega}, \quad \dot{\boldsymbol{\Pi}} = \boldsymbol{\Pi} \times \boldsymbol{\omega} + \mathbf{p} \times \mathbf{v}$$
    pub fn vector_field(&self, state: &LiePoissonState) -> ([f64; 3], [f64; 3]) {
        let (v, omega) = self.velocities(state);
        let p = state.linear_momentum;
        let pi = state.angular_momentum;

        let p_cross_omega = cross3(p, omega);
        let pi_cross_omega = cross3(pi, omega);
        let p_cross_v = cross3(p, v);

        let p_dot = p_cross_omega;
        let pi_dot = [
            pi_cross_omega[0] + p_cross_v[0],
            pi_cross_omega[1] + p_cross_v[1],
            pi_cross_omega[2] + p_cross_v[2],
        ];

        (p_dot, pi_dot)
    }

    /// Runge-Kutta 4th-Order (RK4) Step for Lie-Poisson state propagation.
    pub fn step_rk4(&self, state: &LiePoissonState, dt: f64) -> LiePoissonState {
        let (k1_p, k1_pi) = self.vector_field(state);

        let s1 = LiePoissonState {
            linear_momentum: [
                state.linear_momentum[0] + 0.5 * dt * k1_p[0],
                state.linear_momentum[1] + 0.5 * dt * k1_p[1],
                state.linear_momentum[2] + 0.5 * dt * k1_p[2],
            ],
            angular_momentum: [
                state.angular_momentum[0] + 0.5 * dt * k1_pi[0],
                state.angular_momentum[1] + 0.5 * dt * k1_pi[1],
                state.angular_momentum[2] + 0.5 * dt * k1_pi[2],
            ],
        };
        let (k2_p, k2_pi) = self.vector_field(&s1);

        let s2 = LiePoissonState {
            linear_momentum: [
                state.linear_momentum[0] + 0.5 * dt * k2_p[0],
                state.linear_momentum[1] + 0.5 * dt * k2_p[1],
                state.linear_momentum[2] + 0.5 * dt * k2_p[2],
            ],
            angular_momentum: [
                state.angular_momentum[0] + 0.5 * dt * k2_pi[0],
                state.angular_momentum[1] + 0.5 * dt * k2_pi[1],
                state.angular_momentum[2] + 0.5 * dt * k2_pi[2],
            ],
        };
        let (k3_p, k3_pi) = self.vector_field(&s2);

        let s3 = LiePoissonState {
            linear_momentum: [
                state.linear_momentum[0] + dt * k3_p[0],
                state.linear_momentum[1] + dt * k3_p[1],
                state.linear_momentum[2] + dt * k3_p[2],
            ],
            angular_momentum: [
                state.angular_momentum[0] + dt * k3_pi[0],
                state.angular_momentum[1] + dt * k3_pi[1],
                state.angular_momentum[2] + dt * k3_pi[2],
            ],
        };
        let (k4_p, k4_pi) = self.vector_field(&s3);

        let dt_6 = dt / 6.0;
        LiePoissonState {
            linear_momentum: [
                state.linear_momentum[0]
                    + dt_6 * (k1_p[0] + 2.0 * k2_p[0] + 2.0 * k3_p[0] + k4_p[0]),
                state.linear_momentum[1]
                    + dt_6 * (k1_p[1] + 2.0 * k2_p[1] + 2.0 * k3_p[1] + k4_p[1]),
                state.linear_momentum[2]
                    + dt_6 * (k1_p[2] + 2.0 * k2_p[2] + 2.0 * k3_p[2] + k4_p[2]),
            ],
            angular_momentum: [
                state.angular_momentum[0]
                    + dt_6 * (k1_pi[0] + 2.0 * k2_pi[0] + 2.0 * k3_pi[0] + k4_pi[0]),
                state.angular_momentum[1]
                    + dt_6 * (k1_pi[1] + 2.0 * k2_pi[1] + 2.0 * k3_pi[1] + k4_pi[1]),
                state.angular_momentum[2]
                    + dt_6 * (k1_pi[2] + 2.0 * k2_pi[2] + 2.0 * k3_pi[2] + k4_pi[2]),
            ],
        }
    }
}
