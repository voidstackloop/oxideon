//! Penrose Twistor Theory $\mathbb{PT} \cong \mathbb{CP}^3$, Incidence Relation & Null Ray Helicity (Penrose, 1967).
//!
//! ### 1. Twistor Space $\mathbb{T} \cong \mathbb{C}^4$ & Incidence Relation
//! A twistor $Z^\alpha = (\omega^A, \pi_{A'}) \in \mathbb{C}^2 \oplus \mathbb{C}^2$ connects Minkowski spacetime points
//! $x^\mu = (t, x, y, z)$ with spinor matrix:
//! $$x^{AA'} = \frac{1}{\sqrt{2}} \begin{pmatrix} t + z & x - i y \\ x + i y & t - z \end{pmatrix}$$
//! via **Penrose's Incidence Relation**:
//! $$\omega^A = i x^{AA'} \pi_{A'}$$
//!
//! ### 2. Twistor Pseudo-Hermitian Norm & Massless Helicity $s$
//! Under the $SU(2, 2)$-invariant pseudo-Hermitian inner product:
//! $$Z^\alpha \bar{Z}_\alpha = \omega^A \bar{\pi}_A + \pi_{A'} \bar{\omega}^{A'} = 2 \operatorname{Re}\left(\omega^0 \bar{\pi}_0 + \omega^1 \bar{\pi}_1\right)$$
//! Any twistor satisfying the real incidence relation at a real Minkowski point $x^\mu \in \mathbb{R}^{1,3}$ lies
//! strictly on the **Null Twistor Quadric** $\mathbb{PN} = \{Z^\alpha \bar{Z}_\alpha = 0\}$ (representing a light ray
//! of helicity $s = \frac{1}{2} Z^\alpha \bar{Z}_\alpha = 0$!).

/// Penrose Twistor $Z^\alpha = (\omega^0, \omega^1, \pi_{0'}, \pi_{1'}) \in \mathbb{C}^4$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PenroseTwistorEngine {
    /// 2-spinor $\omega^A = (\omega^0, \omega^1)$ as `(re, im)` pairs.
    pub omega: [(f64, f64); 2],
    /// 2-spinor $\pi_{A'} = (\pi_{0'}, \pi_{1'})$ as `(re, im)` pairs.
    pub pi: [(f64, f64); 2],
}

impl PenroseTwistorEngine {
    #[inline]
    fn cmul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
        (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
    }

    #[inline]
    fn cadd(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
        (a.0 + b.0, a.1 + b.1)
    }

    /// Constructs a null twistor $Z^\alpha = (\omega^A, \pi_{A'})$ incident with a real Minkowski event
    /// $x^\mu = [t, x, y, z]$ and spinor $\pi_{A'}$ via $\omega^A = i x^{AA'} \pi_{A'}$.
    pub fn from_minkowski_incidence(spacetime_x: [f64; 4], pi: [(f64, f64); 2]) -> Self {
        let [t, x, y, z] = spacetime_x;
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;

        // Hermitian matrix x^{AA'}
        let x00 = (inv_sqrt2 * (t + z), 0.0);
        let x01 = (inv_sqrt2 * x, -inv_sqrt2 * y);
        let x10 = (inv_sqrt2 * x, inv_sqrt2 * y);
        let x11 = (inv_sqrt2 * (t - z), 0.0);

        let v0 = Self::cadd(Self::cmul(x00, pi[0]), Self::cmul(x01, pi[1]));
        let v1 = Self::cadd(Self::cmul(x10, pi[0]), Self::cmul(x11, pi[1]));

        // Multiply by i: i * (re + i im) = (-im, re)
        let omega = [(-v0.1, v0.0), (-v1.1, v1.0)];
        Self { omega, pi }
    }

    /// Evaluates the $SU(2,2)$-invariant twistor norm $Z^\alpha \bar{Z}_\alpha = 2 \operatorname{Re}(\omega^0 \bar{\pi}_0 + \omega^1 \bar{\pi}_1)$.
    /// Must vanish identically ($0.0$) for every real Minkowski light ray!
    pub fn su22_hermitian_norm(&self) -> f64 {
        let term0 = self.omega[0].0 * self.pi[0].0 + self.omega[0].1 * self.pi[0].1;
        let term1 = self.omega[1].0 * self.pi[1].0 + self.omega[1].1 * self.pi[1].1;
        2.0 * (term0 + term1)
    }

    /// Computes the future-pointing null 4-momentum $p^\mu = (E, p_x, p_y, p_z)$ from the flagpole spinor $\pi_{A'} \bar{\pi}_A$,
    /// which automatically satisfies the massless Einstein dispersion relation $E^2 - p_x^2 - p_y^2 - p_z^2 = 0$.
    pub fn null_four_momentum(&self) -> [f64; 4] {
        let p0 = self.pi[0];
        let p1 = self.pi[1];
        let norm0 = p0.0 * p0.0 + p0.1 * p0.1;
        let norm1 = p1.0 * p1.0 + p1.1 * p1.1;
        // pi_0 * conj(pi_1)
        let cross_re = p0.0 * p1.0 + p0.1 * p1.1;
        let cross_im = p0.1 * p1.0 - p0.0 * p1.1;

        let e = norm0 + norm1;
        let px = 2.0 * cross_re;
        let py = 2.0 * cross_im;
        let pz = norm0 - norm1;
        [e, px, py, pz]
    }
}
