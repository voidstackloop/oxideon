//! Continuous-time deep learning: Differentiable ODE solvers (`odeint` with `Euler`, `Midpoint`, `RK4`),
//! [`NeuralODE`] blocks (Chen et al. 2018), and Optimal-Transport Conditional Flow Matching /
//! Rectified Flow ([`FlowMatcher`], Lipman et al. 2023).

pub mod jump_sde;
pub mod latent_sde;
pub mod mean_field_games;
pub mod riemannian_cnf;

pub use jump_sde::{JumpDiffusionEngine, NeuralJumpSDE};
pub use latent_sde::{LatentSdeConfig, LatentSdeEngine, SdeTrajectory};
pub use mean_field_games::MeanFieldGameSolver1D;
pub use riemannian_cnf::{PoincareCnfFlow, RiemannianTrajectory};

use crate::error::{Error, Result};
use crate::nn::Module;
use crate::nn::loss::mse_loss;
use crate::tensor::Tensor;

/// Numerical integration scheme for ordinary differential equations $\frac{dy}{dt} = f(t, y)$.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdeMethod {
    /// First-order explicit Euler step: $y_{k+1} = y_k + h f(t_k, y_k)$.
    Euler,
    /// Second-order explicit midpoint (RK2) step.
    Midpoint,
    /// Classic fourth-order Runge–Kutta (`RK4`) step.
    RK4,
}

/// Integrates the initial value problem $\frac{dy}{dt} = f(t, y)$ from `t0` to `t1` using
/// `steps` steps of `method`, returning the final state $y(t_1)$.
///
/// Because every step is built from differentiable [`Tensor`] ops, `odeint` supports
/// reverse-mode and higher-order automatic differentiation through the solver trajectory.
pub fn odeint<F>(
    f: F,
    y0: &Tensor,
    t0: f64,
    t1: f64,
    steps: usize,
    method: OdeMethod,
) -> Result<Tensor>
where
    F: Fn(f64, &Tensor) -> Result<Tensor>,
{
    if steps == 0 {
        return Err(Error::Msg("odeint requires steps >= 1".into()));
    }
    crate::limits::check_iterations("steps", steps)?;
    let dt = (t1 - t0) / (steps as f64);
    let mut y = y0.clone();
    let mut t = t0;

    for _ in 0..steps {
        y = ode_step(&f, t, &y, dt, method)?;
        t += dt;
    }
    Ok(y)
}

/// Integrates $\frac{dy}{dt} = f(t, y)$ from `t0` to `t1` over `steps` steps and returns the
/// full trajectory `[y(t_0), y(t_1), ..., y(t_steps)]` of length `steps + 1`.
pub fn odeint_trajectory<F>(
    f: F,
    y0: &Tensor,
    t0: f64,
    t1: f64,
    steps: usize,
    method: OdeMethod,
) -> Result<Vec<Tensor>>
where
    F: Fn(f64, &Tensor) -> Result<Tensor>,
{
    if steps == 0 {
        return Err(Error::Msg("odeint_trajectory requires steps >= 1".into()));
    }
    crate::limits::check_iterations("steps", steps)?;
    let dt = (t1 - t0) / (steps as f64);
    let mut traj = Vec::with_capacity(steps + 1);
    let mut y = y0.clone();
    let mut t = t0;
    traj.push(y.clone());

    for _ in 0..steps {
        y = ode_step(&f, t, &y, dt, method)?;
        t += dt;
        traj.push(y.clone());
    }
    Ok(traj)
}

fn ode_step<F>(f: &F, t: f64, y: &Tensor, dt: f64, method: OdeMethod) -> Result<Tensor>
where
    F: Fn(f64, &Tensor) -> Result<Tensor>,
{
    match method {
        OdeMethod::Euler => {
            let k1 = f(t, y)?;
            y.add(&k1.mul_scalar(dt)?)
        }
        OdeMethod::Midpoint => {
            let k1 = f(t, y)?;
            let y_mid = y.add(&k1.mul_scalar(0.5 * dt)?)?;
            let k2 = f(t + 0.5 * dt, &y_mid)?;
            y.add(&k2.mul_scalar(dt)?)
        }
        OdeMethod::RK4 => {
            let k1 = f(t, y)?;
            let y2 = y.add(&k1.mul_scalar(0.5 * dt)?)?;
            let k2 = f(t + 0.5 * dt, &y2)?;
            let y3 = y.add(&k2.mul_scalar(0.5 * dt)?)?;
            let k3 = f(t + 0.5 * dt, &y3)?;
            let y4 = y.add(&k3.mul_scalar(dt)?)?;
            let k4 = f(t + dt, &y4)?;

            let weighted = k1
                .add(&k2.mul_scalar(2.0)?)?
                .add(&k3.mul_scalar(2.0)?)?
                .add(&k4)?;
            y.add(&weighted.mul_scalar(dt / 6.0)?)
        }
    }
}

/// Continuous-depth [`NeuralODE`] block wrapping an autonomous vector field `dynamics: M`
/// ($\frac{dz}{dt} = f_\theta(z)$) integrated over $[t_0, t_1]$.
pub struct NeuralODE<M: Module> {
    pub dynamics: M,
    pub t0: f64,
    pub t1: f64,
    pub steps: usize,
    pub method: OdeMethod,
}

impl<M: Module> NeuralODE<M> {
    pub fn new(dynamics: M, t0: f64, t1: f64, steps: usize, method: OdeMethod) -> Self {
        Self {
            dynamics,
            t0,
            t1,
            steps: steps.max(1),
            method,
        }
    }
}

impl<M: Module> Module for NeuralODE<M> {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        odeint(
            |_t, z| self.dynamics.forward(z),
            input,
            self.t0,
            self.t1,
            self.steps,
            self.method,
        )
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        self.dynamics
            .named_parameters()
            .into_iter()
            .map(|(k, v)| (format!("dynamics.{k}"), v))
            .collect()
    }

    fn train(&self, mode: bool) {
        self.dynamics.train(mode);
    }
}

/// Optimal-Transport Conditional Flow Matching / Rectified Flow (`FlowMatcher`, Lipman et al. 2023).
///
/// Constructs straight transport paths between prior noise $x_0 \sim \mathcal{N}(0, I)$ and data $x_1$:
/// $$x_t = (1 - (1 - \sigma_{\min}) t)\, x_0 + t\, x_1, \quad u_t(x_t \mid x_0, x_1) = x_1 - (1 - \sigma_{\min})\, x_0$$
#[derive(Debug, Clone, Copy)]
pub struct FlowMatcher {
    pub sigma_min: f64,
}

impl Default for FlowMatcher {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl FlowMatcher {
    pub fn new(sigma_min: f64) -> Self {
        Self { sigma_min }
    }

    /// Computes the interpolated state $x_t$ and target conditional velocity $u_t$ for batch
    /// times `t` (`[batch]` in $[0, 1]$).
    pub fn sample_location_and_target(
        &self,
        x0: &Tensor,
        x1: &Tensor,
        t: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        if x0.shape() != x1.shape() {
            return Err(Error::ShapeMismatch {
                op: "FlowMatcher::sample_location_and_target",
                expected: x0.shape().to_vec(),
                got: x1.shape().to_vec(),
            });
        }
        if x0.ndim() < 1 {
            return Err(Error::Msg(
                "FlowMatcher needs [batch, ...] inputs, got 0-dim tensor".into(),
            ));
        }
        let b = x0.shape()[0];
        let mut t_shape = vec![1usize; x0.ndim()];
        t_shape[0] = b;
        let t_view = t.reshape(&t_shape)?;

        let one_minus_sigma = 1.0 - self.sigma_min;
        // coeff0 = 1 - (1 - sigma_min) * t
        let coeff0 = t_view.mul_scalar(-one_minus_sigma)?.add_scalar(1.0)?;
        let xt = x0.mul(&coeff0)?.add(&x1.mul(&t_view)?)?;
        let ut = x1.sub(&x0.mul_scalar(one_minus_sigma)?)?;
        Ok((xt, ut))
    }

    /// Conditional Flow Matching MSE loss $\mathbb{E}[\|v_\theta(t, x_t) - u_t\|_2^2]$.
    pub fn cfm_loss(&self, pred_velocity: &Tensor, target_velocity: &Tensor) -> Result<Tensor> {
        mse_loss(pred_velocity, target_velocity)
    }

    /// Generates samples by integrating the learned velocity field $v_\theta(t, x)$ from
    /// $t = 0$ (noise `x0`) to $t = 1$ (data) in `steps` steps using `method`.
    pub fn sample<F>(
        &self,
        velocity_fn: F,
        x0: &Tensor,
        steps: usize,
        method: OdeMethod,
    ) -> Result<Tensor>
    where
        F: Fn(f64, &Tensor) -> Result<Tensor>,
    {
        odeint(velocity_fn, x0, 0.0, 1.0, steps, method)
    }
}

pub mod rough_paths;
pub use rough_paths::RoughPathSignature;

pub mod koopman;
pub use koopman::KoopmanSpectralEngine;

pub mod malliavin_calculus;
pub use malliavin_calculus::MalliavinBismutEngine;

pub mod schramm_loewner_evolution;
pub use schramm_loewner_evolution::SchrammLoewnerEngine;

pub mod kpz_stochastic_burgers;
pub use kpz_stochastic_burgers::KpzColeHopfEngine;

pub mod rough_heston_volatility;
pub use rough_heston_volatility::RoughVolatilityHestonEngine;

pub mod freidlin_wentzell_ldp;
pub use freidlin_wentzell_ldp::FreidlinWentzellLdpEngine;

pub mod mc_kean_vlasov_sde;
pub use mc_kean_vlasov_sde::McKeanVlasovPropagationChaos;

pub mod levy_jump_ito;
pub use levy_jump_ito::LevyItodiffusionEngine;

pub mod backward_sde_pardoux_peng;
pub use backward_sde_pardoux_peng::PardouxPengBsdeEngine;

pub mod stein_method_wasserstein;
pub use stein_method_wasserstein::SteinPoissonCharacterization;

pub mod pontryagin_maximum_principle;
pub use pontryagin_maximum_principle::PontryaginMaximumEngine;

pub mod chow_rashevskii_sub_riemannian;
pub use chow_rashevskii_sub_riemannian::SubRiemannianHeisenbergEngine;

pub mod kalman_bucy_riccati_lqg;
pub use kalman_bucy_riccati_lqg::KalmanBucyRiccatiEngine;

pub mod zames_h_infinity_rob_control;
pub use zames_h_infinity_rob_control::HInfinityRobustControlEngine;

pub mod sontag_control_lyapunov_cbf;
pub use sontag_control_lyapunov_cbf::ControlBarrierLyapunovEngine;
