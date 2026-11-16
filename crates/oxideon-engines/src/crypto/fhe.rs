//! Fully Homomorphic Encryption (FHE / CKKS Scheme) for Encrypted Neural Inference.
//!
//! Implements the Cheon-Kim-Kim-Song (CKKS) approximate homomorphic encryption scheme:
//! - Arithmetic in the cyclotomic polynomial quotient ring $R_q = \mathbb{Z}_q\[X\] / (X^N + 1)$ with $N = 2^k$.
//! - Negacyclic polynomial multiplication:
//!   $$(a \cdot b)_k = \sum_{j=0}^k a_j b_{k-j} - \sum_{j=k+1}^{N-1} a_j b_{N+k-j} \pmod q$$
//! - Encoding of real-valued vectors with fixed-point scaling factor $\Delta$.
//! - Homomorphic operations directly on ciphertexts:
//!   - Addition: $\text{Enc}(m_1) \oplus \text{Enc}(m_2) = \text{Enc}(m_1 + m_2)$
//!   - Plaintext Scalar Multiplication: $\text{Enc}(m) \odot \alpha = \text{Enc}(\alpha \cdot m)$
//!   - Homomorphic Linear Dot Product: $\sum_i \text{Enc}(x_i) \odot w_i = \text{Enc}(\mathbf{w} \cdot \mathbf{x})$
//!
//! **Insecure, for demos and tests only:** key generation and encryption use fixed patterns instead of
//! random sampling, so the secret key is public knowledge and encryption is deterministic.
//! See the security warning in [`crate::crypto`].

use crate::error::{Error, Result};

/// Modulus $q$ used for CKKS polynomial ring ($q \approx 2^{40}$).
pub const CKKS_DEFAULT_MODULUS: i64 = 1_099_511_627_777; // Prime close to 2^40
/// Default fixed-point scaling factor $\Delta = 2^{20}$.
pub const CKKS_DEFAULT_SCALE: f64 = 1_048_576.0;
/// Largest supported ring degree: polynomial products are schoolbook $O(N^2)$, so larger degrees only hang.
pub const CKKS_MAX_DEGREE: usize = 1 << 14;

/// Polynomial in $R_q = \mathbb{Z}_q\[X\] / (X^N + 1)$ with coefficients in $\[0, q)$.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FhePoly {
    pub coeffs: Vec<i64>,
    pub degree: usize,
    pub modulus: i64,
}

impl FhePoly {
    /// Creates a zero polynomial of degree $N$.
    pub fn zero(degree: usize, modulus: i64) -> Self {
        Self {
            coeffs: vec![0; degree],
            degree,
            modulus,
        }
    }

    /// Polynomial addition in $R_q$: $(a + b) \pmod q$.
    pub fn add(&self, other: &Self) -> Result<Self> {
        if self.degree != other.degree || self.modulus != other.modulus {
            return Err(Error::Msg(
                "FhePoly dimension/modulus mismatch in add".into(),
            ));
        }
        let q = self.modulus;
        let coeffs = self
            .coeffs
            .iter()
            .zip(other.coeffs.iter())
            .map(|(&a, &b)| (a + b).rem_euclid(q))
            .collect();
        Ok(Self {
            coeffs,
            degree: self.degree,
            modulus: q,
        })
    }

    /// Polynomial subtraction in $R_q$: $(a - b) \pmod q$.
    pub fn sub(&self, other: &Self) -> Result<Self> {
        if self.degree != other.degree || self.modulus != other.modulus {
            return Err(Error::Msg(
                "FhePoly dimension/modulus mismatch in sub".into(),
            ));
        }
        let q = self.modulus;
        let coeffs = self
            .coeffs
            .iter()
            .zip(other.coeffs.iter())
            .map(|(&a, &b)| (a - b).rem_euclid(q))
            .collect();
        Ok(Self {
            coeffs,
            degree: self.degree,
            modulus: q,
        })
    }

    /// Negacyclic polynomial multiplication in $R_q = \mathbb{Z}_q\[X\] / (X^N + 1)$.
    pub fn mul(&self, other: &Self) -> Result<Self> {
        if self.degree != other.degree || self.modulus != other.modulus {
            return Err(Error::Msg(
                "FhePoly dimension/modulus mismatch in mul".into(),
            ));
        }
        let n = self.degree;
        let q = self.modulus as i128;
        let mut res = vec![0i128; n];

        for i in 0..n {
            let ai = self.coeffs[i] as i128;
            for j in 0..n {
                let bj = other.coeffs[j] as i128;
                let prod = (ai * bj) % q;
                if i + j < n {
                    res[i + j] = (res[i + j] + prod) % q;
                } else {
                    // X^N = -1 mod (X^N + 1)
                    res[i + j - n] = (res[i + j - n] - prod) % q;
                }
            }
        }

        let coeffs = res.into_iter().map(|c| c.rem_euclid(q) as i64).collect();

        Ok(Self {
            coeffs,
            degree: n,
            modulus: self.modulus,
        })
    }

    /// Multiplies polynomial by integer scalar modulo $q$.
    pub fn mul_scalar(&self, scalar: i64) -> Self {
        let q = self.modulus as i128;
        let s = scalar as i128;
        let coeffs = self
            .coeffs
            .iter()
            .map(|&c| ((c as i128 * s) % q).rem_euclid(q) as i64)
            .collect();
        Self {
            coeffs,
            degree: self.degree,
            modulus: self.modulus,
        }
    }
}

/// Secret key $s \in R_q$ with ternary coefficients $\{-1, 0, 1\}$.
#[derive(Debug, Clone)]
pub struct FheSecretKey {
    pub poly: FhePoly,
}

/// Public key $(b, a) \in R_q \times R_q$ where $b = -(a \cdot s + e) \pmod q$.
#[derive(Debug, Clone)]
pub struct FhePublicKey {
    pub b: FhePoly,
    pub a: FhePoly,
}

/// CKKS Ciphertext $(c_0, c_1) \in R_q \times R_q$.
#[derive(Debug, Clone)]
pub struct FheCiphertext {
    pub c0: FhePoly,
    pub c1: FhePoly,
    pub scale: f64,
}

/// CKKS Homomorphic Encryption Context and Parameter Suite.
#[derive(Debug, Clone)]
pub struct CkksContext {
    pub degree: usize,
    pub modulus: i64,
    pub scale: f64,
}

impl CkksContext {
    /// Creates a new CKKS context with polynomial degree $N$ (must be power of 2, e.g. 8, 16, 64).
    pub fn new(degree: usize) -> Result<Self> {
        if degree == 0 || (degree & (degree - 1)) != 0 {
            return Err(Error::Msg(format!("Degree {degree} must be a power of 2")));
        }
        if degree > CKKS_MAX_DEGREE {
            return Err(Error::Msg(format!(
                "Degree {degree} exceeds the supported maximum {CKKS_MAX_DEGREE}"
            )));
        }
        Ok(Self {
            degree,
            modulus: CKKS_DEFAULT_MODULUS,
            scale: CKKS_DEFAULT_SCALE,
        })
    }

    /// Generates a key pair `(SecretKey, PublicKey)`.
    ///
    /// **Insecure:** fully deterministic. The secret key, the public polynomial `a` and the noise are fixed
    /// patterns, so every call returns the same key pair and anyone can recompute the secret key.
    pub fn keygen(&self) -> Result<(FheSecretKey, FhePublicKey)> {
        // Deterministic ternary secret key s: pattern [1, 0, -1, 1, ...]
        let mut s_coeffs = vec![0i64; self.degree];
        for i in 0..self.degree {
            s_coeffs[i] = match i % 3 {
                0 => 1,
                1 => 0,
                _ => -1,
            };
        }
        let s_poly = FhePoly {
            coeffs: s_coeffs
                .into_iter()
                .map(|c| c.rem_euclid(self.modulus))
                .collect(),
            degree: self.degree,
            modulus: self.modulus,
        };

        // Uniform pseudo-random polynomial a
        let mut a_coeffs = vec![0i64; self.degree];
        for i in 0..self.degree {
            let val = ((i as i128 * 6364136223846793005i128 + 1) % self.modulus as i128)
                .rem_euclid(self.modulus as i128) as i64;
            a_coeffs[i] = val;
        }
        let a_poly = FhePoly {
            coeffs: a_coeffs,
            degree: self.degree,
            modulus: self.modulus,
        };

        // Small noise e
        let mut e_coeffs = vec![0i64; self.degree];
        for i in 0..self.degree {
            e_coeffs[i] = if i % 2 == 0 { 1 } else { -1 };
        }
        let e_poly = FhePoly {
            coeffs: e_coeffs
                .into_iter()
                .map(|c| c.rem_euclid(self.modulus))
                .collect(),
            degree: self.degree,
            modulus: self.modulus,
        };

        // b = -(a * s + e) mod q
        let a_s = a_poly.mul(&s_poly)?;
        let a_s_e = a_s.add(&e_poly)?;
        let b_poly = a_s_e.mul_scalar(-1);

        let sk = FheSecretKey { poly: s_poly };
        let pk = FhePublicKey {
            b: b_poly,
            a: a_poly,
        };

        Ok((sk, pk))
    }

    /// Encodes a vector of real numbers $\mathbf{m}$ into a plaintext polynomial $p(X) = \sum \lfloor m_i \cdot \Delta \rceil X^i$.
    pub fn encode(&self, values: &[f64]) -> Result<FhePoly> {
        if values.len() > self.degree {
            return Err(Error::Msg(format!(
                "CkksContext::encode: {} values exceed ring degree {}",
                values.len(),
                self.degree
            )));
        }
        let mut coeffs = vec![0i64; self.degree];
        for (i, &v) in values.iter().enumerate() {
            let scaled = (v * self.scale).round() as i64;
            coeffs[i] = scaled.rem_euclid(self.modulus);
        }
        Ok(FhePoly {
            coeffs,
            degree: self.degree,
            modulus: self.modulus,
        })
    }

    /// Decodes a plaintext polynomial back to real numbers.
    pub fn decode(&self, poly: &FhePoly, scale: f64) -> Vec<f64> {
        let half_q = self.modulus / 2;
        poly.coeffs
            .iter()
            .map(|&c| {
                let centered = if c > half_q { c - self.modulus } else { c };
                centered as f64 / scale
            })
            .collect()
    }

    /// Encrypts a real-valued vector $\mathbf{m}$ using the public key:
    /// $$c = (v \cdot b + e_0 + m, \; v \cdot a + e_1)$$
    ///
    /// **Insecure:** the ephemeral key `v` and the noise are fixed patterns, so encryption is deterministic
    /// (equal plaintexts give equal ciphertexts).
    pub fn encrypt(&self, values: &[f64], pk: &FhePublicKey) -> Result<FheCiphertext> {
        let m_poly = self.encode(values)?;

        // Ephemeral key v with small coefficients
        let mut v_coeffs = vec![0i64; self.degree];
        for i in 0..self.degree {
            v_coeffs[i] = if i % 4 == 0 { 1 } else { 0 };
        }
        let v = FhePoly {
            coeffs: v_coeffs
                .into_iter()
                .map(|c| c.rem_euclid(self.modulus))
                .collect(),
            degree: self.degree,
            modulus: self.modulus,
        };

        // Small noise terms e0, e1
        let mut e0_coeffs = vec![0i64; self.degree];
        let mut e1_coeffs = vec![0i64; self.degree];
        for i in 0..self.degree {
            e0_coeffs[i] = if i % 3 == 0 { 1 } else { 0 };
            e1_coeffs[i] = if i % 3 == 1 { 1 } else { 0 };
        }
        let e0 = FhePoly {
            coeffs: e0_coeffs
                .into_iter()
                .map(|c| c.rem_euclid(self.modulus))
                .collect(),
            degree: self.degree,
            modulus: self.modulus,
        };
        let e1 = FhePoly {
            coeffs: e1_coeffs
                .into_iter()
                .map(|c| c.rem_euclid(self.modulus))
                .collect(),
            degree: self.degree,
            modulus: self.modulus,
        };

        // c0 = v * b + e0 + m
        let vb = v.mul(&pk.b)?;
        let c0 = vb.add(&e0)?.add(&m_poly)?;

        // c1 = v * a + e1
        let va = v.mul(&pk.a)?;
        let c1 = va.add(&e1)?;

        Ok(FheCiphertext {
            c0,
            c1,
            scale: self.scale,
        })
    }

    /// Decrypts ciphertext $c = (c_0, c_1)$ using secret key $s$:
    /// $$m \approx c_0 + c_1 \cdot s \pmod q$$
    pub fn decrypt(&self, ct: &FheCiphertext, sk: &FheSecretKey) -> Result<Vec<f64>> {
        let c1_s = ct.c1.mul(&sk.poly)?;
        let decrypted_poly = ct.c0.add(&c1_s)?;
        Ok(self.decode(&decrypted_poly, ct.scale))
    }

    /// Homomorphic Addition: $\text{Enc}(m_1) \oplus \text{Enc}(m_2)$.
    pub fn add(&self, ct1: &FheCiphertext, ct2: &FheCiphertext) -> Result<FheCiphertext> {
        if (ct1.scale - ct2.scale).abs() > 1e-4 {
            return Err(Error::Msg("Scale mismatch in homomorphic addition".into()));
        }
        let c0 = ct1.c0.add(&ct2.c0)?;
        let c1 = ct1.c1.add(&ct2.c1)?;
        Ok(FheCiphertext {
            c0,
            c1,
            scale: ct1.scale,
        })
    }

    /// Homomorphic Plaintext Multiplication: $\text{Enc}(m) \odot \alpha$.
    pub fn mul_plain(&self, ct: &FheCiphertext, scalar: f64) -> Result<FheCiphertext> {
        let scalar_int = (scalar * 1000.0).round() as i64;
        let c0 = ct.c0.mul_scalar(scalar_int);
        let c1 = ct.c1.mul_scalar(scalar_int);
        Ok(FheCiphertext {
            c0,
            c1,
            scale: ct.scale * 1000.0,
        })
    }

    /// Homomorphic Linear Layer / Dot Product over encrypted ciphertexts and plain weights:
    /// $$\text{Result} = \sum_i \text{Enc}(x_i) \odot w_i$$
    pub fn eval_linear_combination(
        &self,
        ciphertexts: &[FheCiphertext],
        weights: &[f64],
    ) -> Result<FheCiphertext> {
        if ciphertexts.is_empty() || ciphertexts.len() != weights.len() {
            return Err(Error::Msg("Ciphertext and weight length mismatch".into()));
        }

        let mut acc = self.mul_plain(&ciphertexts[0], weights[0])?;
        for (ct, &w) in ciphertexts[1..].iter().zip(weights[1..].iter()) {
            let term = self.mul_plain(ct, w)?;
            acc = self.add(&acc, &term)?;
        }

        Ok(acc)
    }
}
