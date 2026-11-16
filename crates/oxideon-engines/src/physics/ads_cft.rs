//! Holographic Bulk-Boundary Duality, $\text{AdS}_{d+1}/\text{CFT}_d$ Geodesics &
//! Ryu-Takayanagi Entanglement Entropy Engine.
//!
//! # Mathematical & Physical Foundations
//!
//! 1. **Poincaré Patch of Anti-de Sitter Space ($\text{AdS}_3$)**:
//!    With AdS curvature radius $L_{\text{AdS}} > 0$ and holographic radial coordinate $z > 0$
//!    (conformal boundary at $z \to \epsilon > 0$), the equal-time spatial metric is:
//!    $$ds^2 = \frac{L_{\text{AdS}}^2}{z^2} (dz^2 + dx^2)$$
//!    The exact bulk geodesic distance between $(x_1, z_1)$ and $(x_2, z_2)$ is given by the chordal invariant:
//!    $$d_{\text{AdS}}((x_1, z_1), (x_2, z_2)) = L_{\text{AdS}} \, \operatorname{arccosh}\left(1 + \frac{(x_1 - x_2)^2 + (z_1 - z_2)^2}{2 z_1 z_2}\right)$$
//!
//! 2. **Brown-Henneaux Central Charge & Ryu-Takayanagi Formula (2006)**:
//!    Given gravitational constant $G_N$, the dual 2D CFT central charge is:
//!    $$c = \frac{3 L_{\text{AdS}}}{2 G_N}$$
//!    For a boundary subregion $A$ of length $\ell$ and UV cutoff $\epsilon \ll \ell$, the extremal bulk surface
//!    $\gamma_A$ is a semicircle $z(x) = \sqrt{(\ell / 2)^2 - (x - x_0)^2}$ penetrating to IR depth $z_{\max} = \ell / 2$,
//!    with regularized length $\operatorname{Length}(\gamma_A) = 2 L_{\text{AdS}} \ln(\ell / \epsilon)$, giving:
//!    $$S_A = \frac{\operatorname{Area}(\gamma_A)}{4 G_N} = \frac{c}{3} \ln\left(\frac{\ell}{\epsilon}\right)$$
//!
//! 3. **Holographic Mutual Information & Entanglement Phase Transition**:
//!    For two disjoint boundary intervals $A$ (length $\ell_A$) and $B$ (length $\ell_B$) separated by distance $d$:
//!    $$S(A \cup B) = \min\Big( S(\ell_A) + S(\ell_B), \; S(\ell_A + \ell_B + d) + S(d) \Big)$$
//!    $$I(A : B) = S(A) + S(B) - S(A \cup B) \ge 0$$

use crate::error::{Error, Result};

/// Holographic $\text{AdS}_3 / \text{CFT}_2$ Bulk-Boundary Duality & Ryu-Takayanagi Engine.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdsCftHolographicEngine {
    /// AdS curvature radius $L_{\text{AdS}} > 0$.
    pub ads_radius: f64,
    /// 3D Newton gravitational constant $G_N > 0$.
    pub newton_g: f64,
    /// UV regularization cutoff $\epsilon > 0$ near the conformal boundary $z = \epsilon$.
    pub uv_cutoff: f64,
}

impl AdsCftHolographicEngine {
    /// Constructs a validated $\text{AdS}_3 / \text{CFT}_2$ holographic engine.
    pub fn new(ads_radius: f64, newton_g: f64, uv_cutoff: f64) -> Result<Self> {
        if ads_radius <= 0.0 {
            return Err(Error::Msg(format!(
                "ads_radius must be > 0, got {ads_radius}"
            )));
        }
        if newton_g <= 0.0 {
            return Err(Error::Msg(format!("newton_g must be > 0, got {newton_g}")));
        }
        if uv_cutoff <= 0.0 {
            return Err(Error::Msg(format!(
                "uv_cutoff must be > 0, got {uv_cutoff}"
            )));
        }
        Ok(Self {
            ads_radius,
            newton_g,
            uv_cutoff,
        })
    }

    /// Computes the exact Brown-Henneaux CFT central charge:
    ///
    /// $$c = \frac{3 L_{\text{AdS}}}{2 G_N}$$
    #[inline]
    pub fn central_charge(&self) -> f64 {
        (3.0 * self.ads_radius) / (2.0 * self.newton_g)
    }

    /// Computes the exact bulk geodesic distance between two points $(x_1, z_1)$ and $(x_2, z_2)$
    /// in the Poincaré upper half-plane ($z_1, z_2 > 0$).
    pub fn bulk_geodesic_distance(&self, x1: f64, z1: f64, x2: f64, z2: f64) -> Result<f64> {
        if z1 <= 0.0 || z2 <= 0.0 {
            return Err(Error::Msg(
                "Bulk radial coordinates z1, z2 must be > 0".to_string(),
            ));
        }
        let dx = x1 - x2;
        let dz = z1 - z2;
        let chordal = 1.0 + (dx * dx + dz * dz) / (2.0 * z1 * z2);
        Ok(self.ads_radius * chordal.max(1.0).acosh())
    }

    /// Computes the regularized minimal surface (geodesic) area $\operatorname{Area}(\gamma_A)$
    /// anchored on a boundary interval of length $\ell$:
    ///
    /// $$\operatorname{Area}(\gamma_A) = 2 L_{\text{AdS}} \ln\left(\frac{\ell}{\epsilon}\right)$$
    pub fn minimal_surface_area(&self, interval_length: f64) -> Result<f64> {
        if interval_length <= self.uv_cutoff {
            return Err(Error::Msg(format!(
                "Interval length {interval_length} must exceed UV cutoff {}",
                self.uv_cutoff
            )));
        }
        Ok(2.0 * self.ads_radius * (interval_length / self.uv_cutoff).ln())
    }

    /// Computes the Ryu-Takayanagi holographic entanglement entropy $S_A$ for a boundary interval of length $\ell$:
    ///
    /// $$S_A = \frac{\operatorname{Area}(\gamma_A)}{4 G_N} = \frac{c}{3} \ln\left(\frac{\ell}{\epsilon}\right)$$
    pub fn ryu_takayanagi_entropy(&self, interval_length: f64) -> Result<f64> {
        let area = self.minimal_surface_area(interval_length)?;
        Ok(area / (4.0 * self.newton_g))
    }

    /// Samples $N$ points $(x_k, z_k)$ along the bulk Ryu-Takayanagi semicircle geodesic $\gamma_A$
    /// anchored at $[x_{\text{left}}, x_{\text{right}}]$ and returns its maximum bulk radial depth $z_* = \ell / 2$.
    pub fn sample_rt_geodesic(
        &self,
        x_left: f64,
        x_right: f64,
        num_points: usize,
    ) -> Result<(Vec<(f64, f64)>, f64)> {
        if x_right <= x_left || num_points < 2 {
            return Err(Error::Msg(
                "Invalid interval or num_points < 2 in sample_rt_geodesic".to_string(),
            ));
        }
        let x_mid = 0.5 * (x_left + x_right);
        let radius = 0.5 * (x_right - x_left);
        let mut pts = Vec::with_capacity(num_points);

        for k in 0..num_points {
            let theta = std::f64::consts::PI * (k as f64) / ((num_points - 1) as f64);
            let x = x_mid - radius * theta.cos();
            let z = (radius * theta.sin()).max(self.uv_cutoff);
            pts.push((x, z));
        }
        Ok((pts, radius))
    }

    /// Computes the holographic mutual information $I(A : B) = S(A) + S(B) - S(A \cup B) \ge 0$
    /// between two disjoint intervals of lengths $\ell_A, \ell_B$ separated by distance $d > \epsilon$,
    /// capturing the Ryu-Takayanagi connected/disconnected bulk surface phase transition.
    pub fn holographic_mutual_information(
        &self,
        len_a: f64,
        len_b: f64,
        separation_d: f64,
    ) -> Result<f64> {
        let s_a = self.ryu_takayanagi_entropy(len_a)?;
        let s_b = self.ryu_takayanagi_entropy(len_b)?;
        let s_disconnected = s_a + s_b;

        let s_bridge = self.ryu_takayanagi_entropy(len_a + len_b + separation_d)?;
        let s_gap = self.ryu_takayanagi_entropy(separation_d)?;
        let s_connected = s_bridge + s_gap;

        let s_union = s_disconnected.min(s_connected);
        Ok((s_a + s_b - s_union).max(0.0))
    }

    /// Verifies Strong Subadditivity (SSA) of holographic entanglement entropy:
    /// $S(A \cup B) + S(B \cup C) \ge S(A \cup B \cup C) + S(B)$ for adjacent intervals $\ell_A, \ell_B, \ell_C$.
    /// Returns the non-negative SSA information gap $\Delta_{\text{SSA}} \ge 0$.
    pub fn strong_subadditivity_gap(&self, len_a: f64, len_b: f64, len_c: f64) -> Result<f64> {
        let s_ab = self.ryu_takayanagi_entropy(len_a + len_b)?;
        let s_bc = self.ryu_takayanagi_entropy(len_b + len_c)?;
        let s_abc = self.ryu_takayanagi_entropy(len_a + len_b + len_c)?;
        let s_b = self.ryu_takayanagi_entropy(len_b)?;
        Ok(s_ab + s_bc - s_abc - s_b)
    }
}
