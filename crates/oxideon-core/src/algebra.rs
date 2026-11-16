//! Exact and automatic arithmetic types: [`Polynomial`] (real coefficients, division, calculus, Durand-Kerner
//! roots), [`Rational`] (exact `i128` fractions), [`Dual`] (forward-mode automatic differentiation of scalar
//! functions), and [`Interval`] arithmetic with guaranteed enclosures.

use crate::error::{Error, Result};
use std::ops::{Add, Div, Mul, Neg, Sub};

// ------------------------------------------------------------------ polynomial

/// Polynomial with coefficients in ascending order: `c[0] + c[1] x + c[2] x^2 + ...` (trailing zeros trimmed).
#[derive(Debug, Clone, PartialEq)]
pub struct Polynomial {
    pub c: Vec<f64>,
}

impl Polynomial {
    pub fn new(mut c: Vec<f64>) -> Self {
        while c.len() > 1 && *c.last().unwrap() == 0.0 {
            c.pop();
        }
        if c.is_empty() {
            c.push(0.0);
        }
        Self { c }
    }

    /// Monic polynomial with the given real roots.
    pub fn from_roots(roots: &[f64]) -> Self {
        roots.iter().fold(Self::new(vec![1.0]), |p, &r| &p * &Self::new(vec![-r, 1.0]))
    }

    pub fn degree(&self) -> usize {
        self.c.len() - 1
    }

    /// Horner evaluation.
    pub fn eval(&self, x: f64) -> f64 {
        self.c.iter().rev().fold(0.0, |acc, &k| acc * x + k)
    }

    pub fn derivative(&self) -> Self {
        Self::new(self.c.iter().enumerate().skip(1).map(|(i, &k)| k * i as f64).collect())
    }

    /// Antiderivative with integration constant `k`.
    pub fn integral(&self, k: f64) -> Self {
        Self::new(std::iter::once(k).chain(self.c.iter().enumerate().map(|(i, &v)| v / (i + 1) as f64)).collect())
    }

    /// Definite integral over `[a, b]`.
    pub fn integrate(&self, a: f64, b: f64) -> f64 {
        let p = self.integral(0.0);
        p.eval(b) - p.eval(a)
    }

    /// Polynomial composition `self(other(x))`.
    pub fn compose(&self, other: &Polynomial) -> Self {
        self.c.iter().rev().fold(Self::new(vec![0.0]), |acc, &k| &(&acc * other) + &Self::new(vec![k]))
    }

    /// Long division: `(quotient, remainder)`.
    pub fn div_rem(&self, d: &Polynomial) -> Result<(Self, Self)> {
        if d.c.iter().all(|&v| v == 0.0) {
            return Err(Error::Msg("Polynomial::div_rem: division by the zero polynomial".into()));
        }
        let mut r = self.c.clone();
        let (dn, lead) = (d.degree(), *d.c.last().unwrap());
        if r.len() < d.c.len() {
            return Ok((Self::new(vec![0.0]), self.clone()));
        }
        let mut q = vec![0.0; r.len() - dn];
        for i in (0..q.len()).rev() {
            let coef = r[i + dn] / lead;
            q[i] = coef;
            for (j, &dv) in d.c.iter().enumerate() {
                r[i + j] -= coef * dv;
            }
        }
        r.truncate(dn.max(1));
        Ok((Self::new(q), Self::new(r)))
    }

    /// All complex roots `(re, im)` by Durand-Kerner iteration (leading coefficient must be non-zero).
    pub fn roots(&self) -> Result<Vec<(f64, f64)>> {
        let n = self.degree();
        if n == 0 {
            return Ok(Vec::new());
        }
        let lead = *self.c.last().unwrap();
        let monic: Vec<f64> = self.c.iter().map(|v| v / lead).collect();
        let ev = |z: (f64, f64)| -> (f64, f64) { monic.iter().rev().fold((0.0, 0.0), |a, &k| (a.0 * z.0 - a.1 * z.1 + k, a.0 * z.1 + a.1 * z.0)) };
        let radius = 1.0 + monic[..n].iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let mut z: Vec<(f64, f64)> = (0..n).map(|k| { let a = 2.0 * std::f64::consts::PI * k as f64 / n as f64 + 0.4; (0.5 * radius * a.cos(), 0.5 * radius * a.sin()) }).collect();
        for _ in 0..500 {
            let mut delta = 0.0f64;
            for i in 0..n {
                let mut denom = (1.0, 0.0);
                for j in 0..n {
                    if i != j {
                        let d = (z[i].0 - z[j].0, z[i].1 - z[j].1);
                        denom = (denom.0 * d.0 - denom.1 * d.1, denom.0 * d.1 + denom.1 * d.0);
                    }
                }
                let p = ev(z[i]);
                let dd = denom.0 * denom.0 + denom.1 * denom.1;
                if dd == 0.0 {
                    continue;
                }
                let step = ((p.0 * denom.0 + p.1 * denom.1) / dd, (p.1 * denom.0 - p.0 * denom.1) / dd);
                z[i] = (z[i].0 - step.0, z[i].1 - step.1);
                delta = delta.max(step.0.hypot(step.1));
            }
            if delta < 1e-14 {
                break;
            }
        }
        z.iter_mut().for_each(|r| { if r.1.abs() < 1e-9 { r.1 = 0.0; } });
        z.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
        Ok(z)
    }
}

impl Add for &Polynomial {
    type Output = Polynomial;
    fn add(self, o: &Polynomial) -> Polynomial {
        Polynomial::new((0..self.c.len().max(o.c.len())).map(|i| self.c.get(i).unwrap_or(&0.0) + o.c.get(i).unwrap_or(&0.0)).collect())
    }
}

impl Sub for &Polynomial {
    type Output = Polynomial;
    fn sub(self, o: &Polynomial) -> Polynomial {
        Polynomial::new((0..self.c.len().max(o.c.len())).map(|i| self.c.get(i).unwrap_or(&0.0) - o.c.get(i).unwrap_or(&0.0)).collect())
    }
}

impl Mul for &Polynomial {
    type Output = Polynomial;
    fn mul(self, o: &Polynomial) -> Polynomial {
        let mut out = vec![0.0; self.c.len() + o.c.len() - 1];
        for (i, a) in self.c.iter().enumerate() {
            for (j, b) in o.c.iter().enumerate() {
                out[i + j] += a * b;
            }
        }
        Polynomial::new(out)
    }
}

// ------------------------------------------------------------------ rational

fn gcd(a: i128, b: i128) -> i128 {
    if b == 0 { a.abs() } else { gcd(b, a % b) }
}

/// Exact fraction in lowest terms with a positive denominator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rational {
    pub num: i128,
    pub den: i128,
}

impl Rational {
    pub fn new(num: i128, den: i128) -> Result<Self> {
        if den == 0 {
            return Err(Error::Msg("Rational: zero denominator".into()));
        }
        let g = gcd(num, den).max(1);
        let s = if den < 0 { -1 } else { 1 };
        Ok(Self { num: s * num / g, den: s * den / g })
    }

    pub fn int(n: i128) -> Self {
        Self { num: n, den: 1 }
    }

    pub fn to_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }

    pub fn recip(self) -> Result<Self> {
        Self::new(self.den, self.num)
    }

    /// Best rational approximation of `x` with denominator at most `max_den` (continued fractions).
    pub fn approximate(x: f64, max_den: i128) -> Result<Self> {
        if !x.is_finite() || max_den < 1 {
            return Err(Error::Msg("Rational::approximate: need a finite value and max_den >= 1".into()));
        }
        let (mut p0, mut q0, mut p1, mut q1) = (0i128, 1i128, 1i128, 0i128);
        let mut v = x;
        for _ in 0..64 {
            let a = v.floor();
            let ai = a as i128;
            let (p2, q2) = (ai * p1 + p0, ai * q1 + q0);
            if q2 > max_den {
                break;
            }
            (p0, q0, p1, q1) = (p1, q1, p2, q2);
            let frac = v - a;
            if frac.abs() < 1e-15 {
                break;
            }
            v = 1.0 / frac;
        }
        Self::new(p1, q1.max(1))
    }
}

impl Add for Rational {
    type Output = Rational;
    fn add(self, o: Rational) -> Rational {
        Rational::new(self.num * o.den + o.num * self.den, self.den * o.den).unwrap()
    }
}

impl Sub for Rational {
    type Output = Rational;
    fn sub(self, o: Rational) -> Rational {
        self + (-o)
    }
}

impl Mul for Rational {
    type Output = Rational;
    fn mul(self, o: Rational) -> Rational {
        Rational::new(self.num * o.num, self.den * o.den).unwrap()
    }
}

impl Div for Rational {
    type Output = Rational;
    fn div(self, o: Rational) -> Rational {
        Rational::new(self.num * o.den, self.den * o.num).expect("Rational: division by zero")
    }
}

impl Neg for Rational {
    type Output = Rational;
    fn neg(self) -> Rational {
        Rational { num: -self.num, den: self.den }
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        (self.num * o.den).partial_cmp(&(o.num * self.den))
    }
}

impl std::fmt::Display for Rational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.den == 1 { write!(f, "{}", self.num) } else { write!(f, "{}/{}", self.num, self.den) }
    }
}

// ------------------------------------------------------------------ dual numbers

/// Dual number `value + deriv * eps` for forward-mode differentiation: evaluate any composition of the supplied
/// operations at `Dual::var(x)` and read the derivative from `.deriv`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual {
    pub value: f64,
    pub deriv: f64,
}

impl Dual {
    pub fn var(x: f64) -> Self {
        Self { value: x, deriv: 1.0 }
    }
    pub fn constant(x: f64) -> Self {
        Self { value: x, deriv: 0.0 }
    }
    pub fn sin(self) -> Self {
        Self { value: self.value.sin(), deriv: self.deriv * self.value.cos() }
    }
    pub fn cos(self) -> Self {
        Self { value: self.value.cos(), deriv: -self.deriv * self.value.sin() }
    }
    pub fn exp(self) -> Self {
        let e = self.value.exp();
        Self { value: e, deriv: self.deriv * e }
    }
    pub fn ln(self) -> Self {
        Self { value: self.value.ln(), deriv: self.deriv / self.value }
    }
    pub fn sqrt(self) -> Self {
        let s = self.value.sqrt();
        Self { value: s, deriv: self.deriv / (2.0 * s) }
    }
    pub fn powi(self, n: i32) -> Self {
        Self { value: self.value.powi(n), deriv: self.deriv * n as f64 * self.value.powi(n - 1) }
    }
    pub fn tanh(self) -> Self {
        let t = self.value.tanh();
        Self { value: t, deriv: self.deriv * (1.0 - t * t) }
    }
}

impl Add for Dual {
    type Output = Dual;
    fn add(self, o: Dual) -> Dual {
        Dual { value: self.value + o.value, deriv: self.deriv + o.deriv }
    }
}
impl Sub for Dual {
    type Output = Dual;
    fn sub(self, o: Dual) -> Dual {
        Dual { value: self.value - o.value, deriv: self.deriv - o.deriv }
    }
}
impl Mul for Dual {
    type Output = Dual;
    fn mul(self, o: Dual) -> Dual {
        Dual { value: self.value * o.value, deriv: self.deriv * o.value + self.value * o.deriv }
    }
}
impl Div for Dual {
    type Output = Dual;
    fn div(self, o: Dual) -> Dual {
        Dual { value: self.value / o.value, deriv: (self.deriv * o.value - self.value * o.deriv) / (o.value * o.value) }
    }
}
impl Neg for Dual {
    type Output = Dual;
    fn neg(self) -> Dual {
        Dual { value: -self.value, deriv: -self.deriv }
    }
}

/// Derivative of `f` at `x` by forward-mode AD.
pub fn diff<F: Fn(Dual) -> Dual>(f: F, x: f64) -> (f64, f64) {
    let y = f(Dual::var(x));
    (y.value, y.deriv)
}

// ------------------------------------------------------------------ intervals

/// Closed interval `[lo, hi]`; arithmetic returns intervals that contain every possible result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
}

impl Interval {
    pub fn new(lo: f64, hi: f64) -> Result<Self> {
        if lo > hi || lo.is_nan() || hi.is_nan() {
            return Err(Error::Msg("Interval: need lo <= hi".into()));
        }
        Ok(Self { lo, hi })
    }

    pub fn point(x: f64) -> Self {
        Self { lo: x, hi: x }
    }

    pub fn contains(&self, x: f64) -> bool {
        self.lo <= x && x <= self.hi
    }

    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }

    pub fn square(self) -> Self {
        if self.lo >= 0.0 { Self { lo: self.lo * self.lo, hi: self.hi * self.hi } } else if self.hi <= 0.0 { Self { lo: self.hi * self.hi, hi: self.lo * self.lo } } else { Self { lo: 0.0, hi: (self.lo * self.lo).max(self.hi * self.hi) } }
    }
}

impl Add for Interval {
    type Output = Interval;
    fn add(self, o: Interval) -> Interval {
        Interval { lo: self.lo + o.lo, hi: self.hi + o.hi }
    }
}
impl Sub for Interval {
    type Output = Interval;
    fn sub(self, o: Interval) -> Interval {
        Interval { lo: self.lo - o.hi, hi: self.hi - o.lo }
    }
}
impl Mul for Interval {
    type Output = Interval;
    fn mul(self, o: Interval) -> Interval {
        let p = [self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi];
        Interval { lo: p.iter().cloned().fold(f64::INFINITY, f64::min), hi: p.iter().cloned().fold(f64::NEG_INFINITY, f64::max) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} vs {b}");
    }

    #[test]
    fn polynomial_arithmetic_and_calculus() {
        let p = Polynomial::from_roots(&[1.0, 2.0, 3.0]); // x^3 - 6x^2 + 11x - 6
        assert_eq!(p.c, vec![-6.0, 11.0, -6.0, 1.0]);
        close(p.eval(4.0), 6.0, 1e-12);
        assert_eq!(p.derivative().c, vec![11.0, -12.0, 3.0]);
        close(p.integrate(0.0, 2.0), 2.0f64.powi(4) / 4.0 - 2.0 * 8.0 + 11.0 * 2.0 - 12.0, 1e-12);
        let (q, r) = p.div_rem(&Polynomial::new(vec![-1.0, 1.0])).unwrap(); // divide by (x - 1)
        assert_eq!(q.c, vec![6.0, -5.0, 1.0]);
        assert_eq!(r.c, vec![0.0]);
        let (q2, r2) = Polynomial::new(vec![1.0, 0.0, 1.0]).div_rem(&Polynomial::new(vec![1.0, 1.0])).unwrap(); // (x^2 + 1) / (x + 1)
        assert_eq!((q2.c, r2.c), (vec![-1.0, 1.0], vec![2.0]));
        assert!(p.div_rem(&Polynomial::new(vec![0.0])).is_err());
        // composition: (x^2)(x + 1) = x^2 + 2x + 1
        assert_eq!(Polynomial::new(vec![0.0, 0.0, 1.0]).compose(&Polynomial::new(vec![1.0, 1.0])).c, vec![1.0, 2.0, 1.0]);
        assert_eq!((&p - &p).c, vec![0.0]);
    }

    #[test]
    fn polynomial_roots_real_and_complex() {
        let roots = Polynomial::from_roots(&[-2.0, 0.5, 3.0]).roots().unwrap();
        for (got, want) in roots.iter().zip([-2.0, 0.5, 3.0]) {
            close(got.0, want, 1e-8);
            assert_eq!(got.1, 0.0);
        }
        // x^2 + 1 -> +-i
        let c = Polynomial::new(vec![1.0, 0.0, 1.0]).roots().unwrap();
        assert!(c.iter().all(|r| r.0.abs() < 1e-9 && (r.1.abs() - 1.0).abs() < 1e-9));
        // quartic with mixed roots: (x - 1)(x + 1)(x^2 + 4)
        let q = &Polynomial::from_roots(&[1.0, -1.0]) * &Polynomial::new(vec![4.0, 0.0, 1.0]);
        let r = q.roots().unwrap();
        assert_eq!(r.len(), 4);
        assert!(r.iter().all(|z| q_eval(&q, *z) < 1e-8));
        assert!(Polynomial::new(vec![5.0]).roots().unwrap().is_empty());
    }

    fn q_eval(p: &Polynomial, z: (f64, f64)) -> f64 {
        let v = p.c.iter().rev().fold((0.0, 0.0), |a, &k| (a.0 * z.0 - a.1 * z.1 + k, a.0 * z.1 + a.1 * z.0));
        v.0.hypot(v.1)
    }

    #[test]
    fn exact_fractions() {
        let (a, b) = (Rational::new(1, 3).unwrap(), Rational::new(1, 6).unwrap());
        assert_eq!(a + b, Rational::new(1, 2).unwrap());
        assert_eq!(a - b, Rational::new(1, 6).unwrap());
        assert_eq!(a * b, Rational::new(1, 18).unwrap());
        assert_eq!(a / b, Rational::int(2));
        assert_eq!(Rational::new(6, -4).unwrap(), Rational::new(-3, 2).unwrap());
        assert_eq!(format!("{} {}", Rational::new(-3, 2).unwrap(), Rational::int(4)), "-3/2 4");
        assert!(a > b && Rational::new(1, 0).is_err());
        // exact where floats are not: 0.1 + 0.2 == 3/10
        assert_eq!(Rational::new(1, 10).unwrap() + Rational::new(1, 5).unwrap(), Rational::new(3, 10).unwrap());
        assert_eq!(Rational::approximate(std::f64::consts::PI, 1000).unwrap(), Rational::new(355, 113).unwrap());
        assert_eq!(Rational::approximate(0.75, 100).unwrap(), Rational::new(3, 4).unwrap());
        assert!(Rational::approximate(f64::NAN, 10).is_err());
    }

    #[test]
    fn dual_numbers_differentiate_exactly() {
        // d/dx [x^2 sin(x)] = 2x sin x + x^2 cos x
        let (v, d) = diff(|x| x.powi(2) * x.sin(), 1.3);
        close(v, 1.3f64.powi(2) * 1.3f64.sin(), 1e-14);
        close(d, 2.0 * 1.3 * 1.3f64.sin() + 1.3f64.powi(2) * 1.3f64.cos(), 1e-14);
        // quotient and chain rule: d/dx exp(x) / (1 + x^2)
        let (_, d) = diff(|x| x.exp() / (Dual::constant(1.0) + x * x), 0.7);
        let want = (0.7f64.exp() * (1.0 + 0.49) - 0.7f64.exp() * 1.4) / (1.49f64 * 1.49);
        close(d, want, 1e-14);
        let (_, d) = diff(|x| (x * Dual::constant(2.0)).tanh().sqrt(), 0.4);
        close(d, (1.0 - 0.8f64.tanh().powi(2)) * 2.0 / (2.0 * 0.8f64.tanh().sqrt()), 1e-13);
        // agrees with the finite-difference helper to its accuracy
        close(diff(|x| x.ln() + x.cos(), 2.0).1, crate::numerics::derivative(|x| x.ln() + x.cos(), 2.0, 1e-5), 1e-8);
    }

    #[test]
    fn interval_arithmetic_encloses_results() {
        let (x, y) = (Interval::new(-1.0, 2.0).unwrap(), Interval::new(3.0, 4.0).unwrap());
        assert_eq!(x + y, Interval { lo: 2.0, hi: 6.0 });
        assert_eq!(x - y, Interval { lo: -5.0, hi: -1.0 });
        assert_eq!(x * y, Interval { lo: -4.0, hi: 8.0 });
        assert_eq!(x.square(), Interval { lo: 0.0, hi: 4.0 });
        // every sampled evaluation of x*y - x*x lies in the computed enclosure
        let enc = x * y - x.square();
        for i in 0..=30 {
            for j in 0..=30 {
                let (a, b) = (-1.0 + 3.0 * i as f64 / 30.0, 3.0 + j as f64 / 30.0);
                assert!(enc.contains(a * b - a * a));
            }
        }
        assert!(Interval::new(2.0, 1.0).is_err() && Interval::point(3.0).width() == 0.0);
    }
}
