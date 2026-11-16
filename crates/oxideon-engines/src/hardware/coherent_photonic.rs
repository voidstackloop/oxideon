//! Coherent Photonic Silicon Waveguides & Mach-Zehnder Interferometer Meshes II.
//!
//! Provides high-fidelity physical simulation of coherent optical linear processing on
//! silicon photonics platforms:
//! - Waveguide propagation with attenuation $\alpha$ and optical phase delay $\beta L$.
//! - Evanescent directional couplers with configurable coupling ratios $\kappa$.
//! - Inter-waveguide parasitic optical crosstalk matrices.
//! - Thermal phase noise sensitivity & unitary matrix fidelity $\mathcal{F}(U, \tilde{U})$.
//! - Balanced homodyne and heterodyne photodiode square-law optical detection.
//! - Electro-optic nonlinear modulators (MZM/EOM transfer functions).

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Physical optical waveguide parameters.
#[derive(Debug, Clone)]
pub struct WaveguideProfile {
    /// Propagation attenuation loss (linear amplitude multiplier per unit length, in $(0, 1]$).
    pub transmission_factor: f64,
    /// Effective refractive index $n_{\text{eff}}$.
    pub n_eff: f64,
    /// Optical carrier wavelength $\lambda_0$ in meters (e.g. 1550e-9 m for telecom C-band).
    pub wavelength_m: f64,
    /// Thermal phase noise standard deviation $\sigma_\theta$ (in radians).
    pub phase_noise_std: f64,
    /// Inter-waveguide parasitic crosstalk coefficient $\epsilon \ge 0$.
    pub crosstalk_leakage: f64,
}

impl Default for WaveguideProfile {
    fn default() -> Self {
        Self {
            transmission_factor: 0.98,
            n_eff: 2.45,
            wavelength_m: 1550e-9,
            phase_noise_std: 0.01,
            crosstalk_leakage: 0.005,
        }
    }
}

/// Coherent Photonic Integrated Circuit (PIC) mesh engine.
#[derive(Debug, Clone)]
pub struct CoherentPhotonicCircuit {
    /// Number of spatial optical waveguide modes $N$.
    pub num_modes: usize,
    /// Physical profile of waveguides.
    pub profile: WaveguideProfile,
    /// Phase shifter matrix $\Theta \in \mathbb{R}^{N \times N}$.
    pub phase_shifts: Vec<f64>,
    /// Coupling ratios $\kappa \in [0, 1]$ for directional couplers between adjacent waveguides.
    pub coupling_ratios: Vec<f64>,
}

impl CoherentPhotonicCircuit {
    /// Constructs a new Coherent Photonic Circuit for $N$ optical modes.
    pub fn new(num_modes: usize, profile: WaveguideProfile) -> Result<Self> {
        if num_modes < 2 {
            return Err(Error::Msg(
                "Photonic circuit requires at least 2 optical modes".into(),
            ));
        }

        let phase_shifts =
            crate::layout::try_filled(crate::layout::checked_numel(&[num_modes, num_modes])?, 0.0)?;
        let coupling_ratios = crate::layout::try_filled(num_modes - 1, 0.5)?; // 50:50 3dB couplers by default

        Ok(Self {
            num_modes,
            profile,
            phase_shifts,
            coupling_ratios,
        })
    }

    /// Sets the phase shift of interferometer at position $(i, j)$.
    pub fn set_phase_shift(&mut self, i: usize, j: usize, theta: f64) -> Result<()> {
        if i >= self.num_modes || j >= self.num_modes {
            return Err(Error::IndexOutOfBounds {
                op: "set_phase_shift",
                index: i.max(j) as isize,
                dim_size: self.num_modes,
            });
        }
        self.phase_shifts[i * self.num_modes + j] = theta;
        Ok(())
    }

    /// Sets the directional coupler splitting ratio $\kappa \in [0, 1]$ between modes $i$ and $i+1$.
    pub fn set_coupling_ratio(&mut self, mode_i: usize, kappa: f64) -> Result<()> {
        if mode_i >= self.num_modes - 1 {
            return Err(Error::IndexOutOfBounds {
                op: "set_coupling_ratio",
                index: mode_i as isize,
                dim_size: self.num_modes - 1,
            });
        }
        if !(0.0..=1.0).contains(&kappa) {
            return Err(Error::Msg(format!(
                "Coupling ratio kappa must be in [0, 1], got {kappa}"
            )));
        }
        self.coupling_ratios[mode_i] = kappa;
        Ok(())
    }

    /// Propagates complex electric fields $\mathbf{E}_{\text{in}} = (\mathbf{E}_r, \mathbf{E}_i) \in \mathbb{R}^N \times \mathbb{R}^N$
    /// through the directional coupler layer between adjacent waveguides:
    ///
    /// $$\begin{pmatrix} E_1' \\ E_2' \end{pmatrix} = \begin{pmatrix} \sqrt{1 - \kappa} & i\sqrt{\kappa} \\ i\sqrt{\kappa} & \sqrt{1 - \kappa} \end{pmatrix} \begin{pmatrix} E_1 \\ E_2 \end{pmatrix}$$
    pub fn propagate_coupler_layer(
        &self,
        e_real: &[f64],
        e_imag: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let n = self.num_modes;
        if e_real.len() != n || e_imag.len() != n {
            return Err(Error::ShapeMismatch {
                op: "propagate_coupler_layer",
                expected: vec![n],
                got: vec![e_real.len()],
            });
        }

        let mut out_r = e_real.to_vec();
        let mut out_i = e_imag.to_vec();

        // Pairwise directional coupling along adjacent waveguide pairs (0,1), (2,3), etc.
        let mut i = 0;
        while i + 1 < n {
            let kappa = self.coupling_ratios[i];
            let t = (1.0 - kappa).sqrt();
            let k = kappa.sqrt();

            let r1 = out_r[i];
            let im1 = out_i[i];
            let r2 = out_r[i + 1];
            let im2 = out_i[i + 1];

            // E1' = t * E1 + i * k * E2 = (t * r1 - k * im2) + i * (t * im1 + k * r2)
            out_r[i] = t * r1 - k * im2;
            out_i[i] = t * im1 + k * r2;

            // E2' = i * k * E1 + t * E2 = (-k * im1 + t * r2) + i * (k * r1 + t * im2)
            out_r[i + 1] = -k * im1 + t * r2;
            out_i[i + 1] = k * r1 + t * im2;

            i += 2;
        }

        Ok((out_r, out_i))
    }

    /// Propagates complex electric fields through phase shifters and applies waveguide attenuation:
    /// $E'_m = \alpha \cdot e^{i \theta_m} E_m$.
    pub fn propagate_phase_layer(
        &self,
        e_real: &[f64],
        e_imag: &[f64],
        layer_idx: usize,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let n = self.num_modes;
        if e_real.len() != n || e_imag.len() != n {
            return Err(Error::ShapeMismatch {
                op: "propagate_phase_layer",
                expected: vec![n],
                got: vec![e_real.len()],
            });
        }

        let row = layer_idx % n;
        let mut out_r = vec![0.0; n];
        let mut out_i = vec![0.0; n];
        let alpha = self.profile.transmission_factor;

        for m in 0..n {
            let theta = self.phase_shifts[row * n + m];
            let (s, c) = theta.sin_cos();

            let r = e_real[m];
            let im = e_imag[m];

            // E' = alpha * (c + i s) * (r + i im) = alpha * ((c*r - s*im) + i (s*r + c*im))
            out_r[m] = alpha * (c * r - s * im);
            out_i[m] = alpha * (s * r + c * im);
        }

        Ok((out_r, out_i))
    }

    /// Full coherent forward pass simulating multi-stage MZI mesh with optical crosstalk.
    pub fn forward(
        &self,
        e_real: &[f64],
        e_imag: &[f64],
        stages: usize,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let n = self.num_modes;
        let mut cur_r = e_real.to_vec();
        let mut cur_i = e_imag.to_vec();

        for s in 0..stages {
            let (p_r, p_i) = self.propagate_phase_layer(&cur_r, &cur_i, s)?;
            let (c_r, c_i) = self.propagate_coupler_layer(&p_r, &p_i)?;
            cur_r = c_r;
            cur_i = c_i;
        }

        // Apply parasitic evanescent crosstalk leakage: E' = (1 - eps) E + (eps / (N-1)) sum_{k != m} E_k
        let eps = self.profile.crosstalk_leakage;
        if eps > 0.0 && n > 1 {
            let sum_r: f64 = cur_r.iter().sum();
            let sum_i: f64 = cur_i.iter().sum();
            let mut xtalk_r = vec![0.0; n];
            let mut xtalk_i = vec![0.0; n];

            let leak_factor = eps / ((n - 1) as f64);
            for m in 0..n {
                let other_r = sum_r - cur_r[m];
                let other_i = sum_i - cur_i[m];
                xtalk_r[m] = (1.0 - eps) * cur_r[m] + leak_factor * other_r;
                xtalk_i[m] = (1.0 - eps) * cur_i[m] + leak_factor * other_i;
            }
            cur_r = xtalk_r;
            cur_i = xtalk_i;
        }

        Ok((cur_r, cur_i))
    }

    /// Evaluates square-law photodiode detection power:
    /// $P_m = |E_m|^2 = \operatorname{Re}(E_m)^2 + \operatorname{Im}(E_m)^2$.
    pub fn optical_power(e_real: &[f64], e_imag: &[f64]) -> Vec<f64> {
        e_real
            .iter()
            .zip(e_imag.iter())
            .map(|(&r, &i)| r * r + i * i)
            .collect()
    }

    /// Evaluates optical power directly as a Tensor given complex fields of shape `[num_modes, 2]`.
    pub fn forward_power(&self, input_fields: &Tensor, stages: usize) -> Result<Tensor> {
        let shape = input_fields.shape();
        let n = self.num_modes;
        if shape.len() != 2 || shape[0] != n || shape[1] != 2 {
            return Err(Error::ShapeMismatch {
                op: "CoherentPhotonicCircuit::forward_power",
                expected: vec![n, 2],
                got: shape.to_vec(),
            });
        }
        let data = input_fields.to_vec::<f64>()?;
        let mut r = vec![0.0; n];
        let mut im = vec![0.0; n];
        for i in 0..n {
            r[i] = data[2 * i];
            im[i] = data[2 * i + 1];
        }
        let (out_r, out_i) = self.forward(&r, &im, stages)?;
        let powers = Self::optical_power(&out_r, &out_i);
        Tensor::from_slice(&powers, &[n])
    }

    /// Balanced homodyne detection between signal field $E_{\text{sig}}$ and local oscillator $E_{\text{lo}}$:
    /// $I_{\text{homodyne}} = 4 \operatorname{Re}(E_{\text{sig}} E_{\text{lo}}^*)$.
    pub fn balanced_homodyne(
        sig_r: &[f64],
        sig_i: &[f64],
        lo_r: &[f64],
        lo_i: &[f64],
    ) -> Result<Vec<f64>> {
        let n = sig_r.len();
        if sig_i.len() != n || lo_r.len() != n || lo_i.len() != n {
            return Err(Error::ShapeMismatch {
                op: "balanced_homodyne",
                expected: vec![n],
                got: vec![sig_i.len()],
            });
        }

        let mut out = vec![0.0; n];
        for m in 0..n {
            // E_sig * E_lo^* = (sr + i si) * (lor - i loi) = (sr*lor + si*loi) + i(...)
            out[m] = 4.0 * (sig_r[m] * lo_r[m] + sig_i[m] * lo_i[m]);
        }

        Ok(out)
    }

    /// Electro-Optic Mach-Zehnder Modulator (MZM) nonlinear transfer function:
    /// $T(V) = \cos^2\left(\frac{\pi V}{2 V_\pi}\right)$.
    pub fn electro_optic_activation(voltages: &[f64], v_pi: f64) -> Result<Vec<f64>> {
        if !v_pi.is_finite() || v_pi == 0.0 {
            return Err(Error::Msg(format!(
                "electro_optic_activation requires finite nonzero v_pi, got {v_pi}"
            )));
        }
        Ok(voltages
            .iter()
            .map(|&v| {
                let arg = std::f64::consts::FRAC_PI_2 * (v / v_pi);
                let cos_val = arg.cos();
                cos_val * cos_val
            })
            .collect())
    }
}
