//! Scalar (`f64`) special functions that complement [`crate::special`]: complementary error function with full tail
//! accuracy ([`erfc`], [`erfcx`]), complete elliptic integrals ([`ellipk`], [`ellipe`]), exponential integrals
//! ([`expint_e1`], [`expint_ei`]), the Lambert W function ([`lambertw`]), Fresnel integrals ([`fresnel`]), Dawson's
//! integral ([`dawsn`]), sine / cosine integrals ([`sici`]), the dilogarithm ([`spence`]), Bessel functions of integer
//! order ([`jn`], [`ive`]) and real order ([`kve`], [`kv`]), and classical orthogonal polynomials
//! ([`legendre`], [`hermite`], [`hermite_e`], [`laguerre`], [`genlaguerre`], [`chebyt`], [`chebyu`], [`gegenbauer`],
//! [`jacobi`]). Domain errors return `NaN`. Definitions follow `scipy.special`.

use crate::dtype::DType;
use crate::error::Result;
use crate::tensor::Tensor;
use std::f64::consts::{FRAC_PI_2, PI};

const EULER: f64 = 0.577_215_664_901_532_9;
const FPMIN: f64 = 1e-300;
const SQRT_PI: f64 = 1.772_453_850_905_516;

/// Complementary error function `erfc(x)` with small relative error in both tails.
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    if x < 2.0 {
        // erf(x) = 2/sqrt(pi) e^{-x^2} sum 2^n x^{2n+1} / (2n+1)!!   (all terms positive)
        let (mut term, mut sum, mut n) = (x, x, 0.0);
        while term > 1e-17 * sum {
            n += 1.0;
            term *= 2.0 * x * x / (2.0 * n + 1.0);
            sum += term;
        }
        1.0 - 2.0 / SQRT_PI * (-x * x).exp() * sum
    } else if x > 27.3 {
        0.0
    } else {
        (-x * x).exp() * erfcx_cf(x)
    }
}

/// `sqrt(pi)^-1 / (x + (1/2)/(x + 1/(x + (3/2)/(x + ...))))`, the continued fraction of `erfcx` for `x >= 2`.
fn erfcx_cf(x: f64) -> f64 {
    let mut f = x;
    for k in (1..=300).rev() {
        f = x + (k as f64 / 2.0) / f;
    }
    1.0 / (SQRT_PI * f)
}

/// Scaled complementary error function `exp(x^2) * erfc(x)` (no overflow or underflow for large `x`).
pub fn erfcx(x: f64) -> f64 {
    if x.is_nan() {
        x
    } else if x >= 2.0 {
        erfcx_cf(x)
    } else if x > -26.0 {
        (x * x).exp() * erfc(x)
    } else {
        f64::INFINITY
    }
}

/// Complete elliptic integral of the first kind `K(m)` with parameter `m = k^2 <= 1` (`scipy.special.ellipk`).
pub fn ellipk(m: f64) -> f64 {
    if m.is_nan() || m > 1.0 {
        return f64::NAN;
    }
    if m == 1.0 {
        return f64::INFINITY;
    }
    if m.is_infinite() {
        return 0.0; // m -> -inf
    }
    let (mut a, mut b) = (1.0, (1.0 - m).sqrt());
    for _ in 0..60 {
        if (a - b).abs() <= 1e-16 * a {
            break;
        }
        let an = 0.5 * (a + b);
        b = (a * b).sqrt();
        a = an;
    }
    PI / (2.0 * a)
}

/// Complete elliptic integral of the second kind `E(m)` with `m <= 1` (`scipy.special.ellipe`).
pub fn ellipe(m: f64) -> f64 {
    if m.is_nan() || m > 1.0 {
        return f64::NAN;
    }
    if m == 1.0 {
        return 1.0;
    }
    if m.is_infinite() {
        return f64::INFINITY; // m -> -inf
    }
    let (mut a, mut b, mut c2) = (1.0, (1.0 - m).sqrt(), m);
    let (mut sum, mut pow2) = (0.5 * c2, 0.5);
    for _ in 0..60 {
        if (a - b).abs() <= 1e-16 * a {
            break;
        }
        let c = 0.5 * (a - b);
        let an = 0.5 * (a + b);
        b = (a * b).sqrt();
        a = an;
        c2 = c * c;
        pow2 *= 2.0;
        sum += pow2 * c2;
    }
    ellipk(m) * (1.0 - sum)
}

/// Exponential integral `E1(x) = int_x^inf e^-t / t dt` for `x > 0` (`scipy.special.exp1` on the reals).
pub fn expint_e1(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY;
    }
    if x <= 1.0 {
        let (mut term, mut sum, mut k) = (1.0, 0.0, 0.0);
        loop {
            k += 1.0;
            term *= -x / k;
            let add = -term / k;
            sum += add;
            if add.abs() < 1e-17 * sum.abs().max(1e-300) {
                break;
            }
        }
        -EULER - x.ln() + sum
    } else {
        let (mut b, mut c, mut d) = (x + 1.0, 1.0 / FPMIN, 1.0 / (x + 1.0));
        let mut h = d;
        for i in 1..1000 {
            let a = -((i * i) as f64);
            b += 2.0;
            d = 1.0 / (a * d + b);
            c = b + a / c;
            let del = c * d;
            h *= del;
            if (del - 1.0).abs() < 1e-16 {
                break;
            }
        }
        h * (-x).exp()
    }
}

/// Exponential integral `Ei(x)` (principal value for `x > 0`; `Ei(-x) = -E1(x)`).
pub fn expint_ei(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x < 0.0 {
        return -expint_e1(-x);
    }
    if x > 710.0 {
        return f64::INFINITY;
    }
    let (mut term, mut sum, mut k) = (1.0, 0.0, 0.0);
    loop {
        k += 1.0;
        term *= x / k;
        let add = term / k;
        sum += add;
        if add < 1e-17 * sum {
            break;
        }
    }
    EULER + x.ln() + sum
}

/// Lambert W: the solution `w` of `w e^w = x`. `branch` 0 is the principal branch (`x >= -1/e`), `-1` the lower
/// real branch (`-1/e <= x < 0`).
pub fn lambertw(x: f64, branch: i32) -> f64 {
    let e = std::f64::consts::E;
    if x.is_nan() || x < -1.0 / e || !(branch == 0 || branch == -1) {
        return f64::NAN;
    }
    if x == -1.0 / e {
        return -1.0;
    }
    if branch == -1 && x >= 0.0 {
        return if x == 0.0 { f64::NEG_INFINITY } else { f64::NAN };
    }
    if x == 0.0 {
        return 0.0;
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }
    let p2 = 2.0 * (e * x + 1.0);
    let mut w = if branch == -1 {
        if x < -0.25 {
            let p = -p2.sqrt();
            -1.0 + p - p * p / 3.0 + 11.0 / 72.0 * p * p * p
        } else {
            let l = (-x).ln();
            l - (-l).ln()
        }
    } else if x < -0.25 {
        let p = p2.sqrt();
        -1.0 + p - p * p / 3.0 + 11.0 / 72.0 * p * p * p
    } else if x < 3.0 {
        let l = x.ln_1p();
        0.665 * (1.0 + 0.0195 * l) * l + 0.04
    } else {
        let (l1, l2) = (x.ln(), x.ln().ln());
        l1 - l2 + l2 / l1
    };
    for _ in 0..60 {
        let ew = w.exp();
        let f = w * ew - x;
        let denom = ew * (w + 1.0) - (w + 2.0) * f / (2.0 * w + 2.0);
        if denom == 0.0 || !denom.is_finite() {
            break;
        }
        let dw = f / denom;
        w -= dw;
        if dw.abs() <= 1e-15 * (1.0 + w.abs()) {
            break;
        }
    }
    w
}

/// Fresnel integrals `(S(x), C(x))` with `S(x) = int_0^x sin(pi t^2 / 2) dt` (`scipy.special.fresnel`).
pub fn fresnel(x: f64) -> (f64, f64) {
    if x.is_nan() {
        return (x, x);
    }
    if x.is_infinite() {
        return (0.5f64.copysign(x), 0.5f64.copysign(x));
    }
    let ax = x.abs();
    let (s, c) = if ax < 1.6 {
        let t = FRAC_PI_2 * ax * ax;
        // C = sum (-1)^n (pi/2)^{2n} x^{4n+1} / ((2n)! (4n+1)),  S = sum (-1)^n (pi/2)^{2n+1} x^{4n+3} / ((2n+1)! (4n+3))
        let (mut cs, mut ss) = (0.0, 0.0);
        let (mut pc, mut ps) = (ax, FRAC_PI_2 * ax * ax * ax); // x^(4n+1) * (pi/2)^(2n) / (2n)!  and the S analogue
        let t2 = t * t;
        for n in 0..60 {
            let nf = n as f64;
            let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
            cs += sign * pc / (4.0 * nf + 1.0);
            ss += sign * ps / (4.0 * nf + 3.0);
            pc *= t2 / ((2.0 * nf + 1.0) * (2.0 * nf + 2.0));
            ps *= t2 / ((2.0 * nf + 2.0) * (2.0 * nf + 3.0));
            if pc < 1e-18 && ps < 1e-18 {
                break;
            }
        }
        (ss, cs)
    } else {
        // Lentz continued fraction for the complex auxiliary function.
        let (mut br, bi) = (1.0, -PI * ax * ax);
        let (mut cr, mut ci) = (1.0 / FPMIN, 0.0);
        let cdiv = |ar: f64, ai: f64, br: f64, bi: f64| {
            let d = br * br + bi * bi;
            ((ar * br + ai * bi) / d, (ai * br - ar * bi) / d)
        };
        let (mut dr, mut di) = cdiv(1.0, 0.0, br, bi);
        let (mut hr, mut hi) = (dr, di);
        let mut n = -1.0;
        for _ in 0..2000 {
            n += 2.0;
            let a = -n * (n + 1.0);
            br += 4.0;
            // d = 1 / (a d + b)
            let (tr, ti) = (a * dr + br, a * di + bi);
            (dr, di) = cdiv(1.0, 0.0, tr, ti);
            // c = b + a / c
            let (qr, qi) = cdiv(a, 0.0, cr, ci);
            (cr, ci) = (br + qr, bi + qi);
            let (delr, deli) = (cr * dr - ci * di, cr * di + ci * dr);
            let (nhr, nhi) = (hr * delr - hi * deli, hr * deli + hi * delr);
            (hr, hi) = (nhr, nhi);
            if (delr - 1.0).abs() + deli.abs() < 1e-16 {
                break;
            }
        }
        // h *= (ax, -ax)
        let (hr2, hi2) = (ax * hr + ax * hi, ax * hi - ax * hr);
        let ang = FRAC_PI_2 * ax * ax;
        let (cs_, sn_) = (ang.cos(), ang.sin());
        // cs = (0.5 + 0.5i) * (1 - (cos + i sin) * h)
        let (pr, pi_) = (cs_ * hr2 - sn_ * hi2, cs_ * hi2 + sn_ * hr2);
        let (ur, ui) = (1.0 - pr, -pi_);
        let (c, s) = (0.5 * ur - 0.5 * ui, 0.5 * ur + 0.5 * ui);
        (s, c)
    };
    if x < 0.0 { (-s, -c) } else { (s, c) }
}

/// Dawson's integral `D(x) = exp(-x^2) int_0^x exp(t^2) dt` (`scipy.special.dawsn`).
pub fn dawsn(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let ax = x.abs();
    let v = if ax < 0.5 {
        // D(x) = sum (-2)^k x^{2k+1} / (2k+1)!!
        let (mut term, mut sum) = (ax, ax);
        for k in 1..40 {
            term *= -2.0 * ax * ax / (2.0 * k as f64 + 1.0);
            sum += term;
        }
        sum
    } else if ax > 1e8 {
        0.5 / ax
    } else {
        // Rybicki: D(x) = pi^{-1/2} sum_{n odd} exp(-(x - n h)^2) / n  with h = 0.2 (error ~ e^{-(pi/2h)^2})
        let h = 0.2;
        let n0 = (2.0 * (ax / (2.0 * h)).round()) as i64 + 1; // an odd integer near ax / h
        let mut sum = 0.0;
        for k in -40..=40 {
            let n = n0 + 2 * k;
            if n == 0 {
                continue;
            }
            let d = ax - n as f64 * h;
            sum += (-d * d).exp() / n as f64;
        }
        sum / SQRT_PI
    };
    v.copysign(x)
}

/// Sine and cosine integrals `(Si(x), Ci(x))` for `x >= 0` (`Ci` is `NaN` for `x < 0`, `-inf` at 0).
pub fn sici(x: f64) -> (f64, f64) {
    if x.is_nan() {
        return (x, x);
    }
    let t = x.abs();
    let (si, ci) = if t.is_infinite() {
        (FRAC_PI_2, 0.0)
    } else if t == 0.0 {
        (0.0, f64::NEG_INFINITY)
    } else if t <= 2.0 {
        let (mut si, mut ci) = (0.0, 0.0);
        let mut term = t; // t^{2k+1}/(2k+1)!
        let mut k = 0.0;
        // Si = sum (-1)^k t^{2k+1} / ((2k+1) (2k+1)!)
        loop {
            let a = term / (2.0 * k + 1.0);
            si += if (k as i64) % 2 == 0 { a } else { -a };
            term *= t * t / ((2.0 * k + 2.0) * (2.0 * k + 3.0));
            k += 1.0;
            if a < 1e-18 {
                break;
            }
        }
        // Ci = gamma + ln t + sum_{k>=1} (-1)^k t^{2k} / (2k (2k)!)
        let mut term = 1.0;
        let mut k = 0.0;
        loop {
            k += 1.0;
            term *= t * t / ((2.0 * k - 1.0) * (2.0 * k));
            let a = term / (2.0 * k);
            ci += if (k as i64) % 2 == 0 { a } else { -a };
            if a < 1e-18 {
                break;
            }
        }
        (si, EULER + t.ln() + ci)
    } else {
        // continued fraction for E1(i t)
        let (mut br, bi) = (1.0, t);
        let cdiv = |ar: f64, ai: f64, br: f64, bi: f64| {
            let d = br * br + bi * bi;
            ((ar * br + ai * bi) / d, (ai * br - ar * bi) / d)
        };
        let (mut cr, mut ci_) = (1.0 / FPMIN, 0.0);
        let (mut dr, mut di) = cdiv(1.0, 0.0, br, bi);
        let (mut hr, mut hi) = (dr, di);
        for i in 2..5000 {
            let a = -(((i - 1) * (i - 1)) as f64);
            br += 2.0;
            let (tr, ti) = (a * dr + br, a * di + bi);
            (dr, di) = cdiv(1.0, 0.0, tr, ti);
            let (qr, qi) = cdiv(a, 0.0, cr, ci_);
            (cr, ci_) = (br + qr, bi + qi);
            let (delr, deli) = (cr * dr - ci_ * di, cr * di + ci_ * dr);
            (hr, hi) = (hr * delr - hi * deli, hr * deli + hi * delr);
            if (delr - 1.0).abs() + deli.abs() < 1e-16 {
                break;
            }
        }
        // h *= (cos t - i sin t)
        let (c, s) = (t.cos(), t.sin());
        let (rr, ri) = (c * hr + s * hi, c * hi - s * hr);
        (FRAC_PI_2 + ri, -rr)
    };
    if x < 0.0 { (-si, f64::NAN) } else { (si, ci) }
}

/// Dilogarithm in `scipy.special.spence` convention: `spence(z) = Li2(1 - z)` for `z >= 0`.
pub fn spence(z: f64) -> f64 {
    if z.is_nan() || z < 0.0 {
        return f64::NAN;
    }
    li2(1.0 - z)
}

/// Real dilogarithm `Li2(y)` for `y <= 1`.
fn li2(y: f64) -> f64 {
    if y == 1.0 {
        return PI * PI / 6.0;
    }
    if y == 0.0 {
        return 0.0;
    }
    if y < -1.0 {
        // inversion
        return -PI * PI / 6.0 - 0.5 * (-y).ln().powi(2) - li2(1.0 / y);
    }
    if y < -0.5 {
        // Landen
        return -li2(y / (y - 1.0)) - 0.5 * (1.0 - y).ln().powi(2);
    }
    if y > 0.5 {
        return PI * PI / 6.0 - y.ln() * (1.0 - y).ln() - li2(1.0 - y);
    }
    let (mut pw, mut sum) = (y, y);
    for k in 2..200 {
        pw *= y;
        let t = pw / (k * k) as f64;
        sum += t;
        if t.abs() < 1e-18 * sum.abs() {
            break;
        }
    }
    sum
}

/// Bessel function of the first kind of integer order `J_n(x)`, from the periodic integral representation
/// `J_n(x) = (1/2pi) int_0^{2pi} cos(n t - x sin t) dt` (trapezoid rule converges geometrically).
pub fn jn(n: i32, x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x.is_infinite() {
        return 0.0;
    }
    // J_{-n}(x) = (-1)^n J_n(x) and J_n(-x) = (-1)^n J_n(x)
    let odd = n % 2 != 0;
    let sign = if odd && ((n < 0) != (x < 0.0)) { -1.0 } else { 1.0 };
    let (n, ax) = (n.unsigned_abs() as f64, x.abs());
    if ax * ax <= 4.0 * (n + 1.0) {
        // power series: terms decrease from the start, so tiny values keep full relative accuracy
        let half = 0.5 * ax;
        let mut term = 1.0f64;
        for i in 1..=n as u64 {
            term *= half / i as f64;
            if term == 0.0 {
                return 0.0;
            }
        }
        let (mut sum, mut k) = (term, 0.0);
        loop {
            k += 1.0;
            term *= -half * half / (k * (k + n));
            sum += term;
            if term.abs() <= 1e-17 * sum.abs() || k > 500.0 {
                break;
            }
        }
        return sign * sum;
    }
    if ax >= 25.0 + n * n {
        // Hankel asymptotic expansion: sqrt(2 / (pi x)) (P cos(chi) - Q sin(chi)), chi = x - (n/2 + 1/4) pi
        let mu = 4.0 * n * n;
        let (mut p, mut q, mut term, mut k) = (1.0f64, 0.0f64, 1.0f64, 0.0f64);
        loop {
            k += 1.0;
            let next = term * (mu - (2.0 * k - 1.0).powi(2)) / (k * 8.0 * ax);
            if next.abs() >= term.abs() || k > 200.0 {
                break; // the series is asymptotic: stop at the smallest term
            }
            term = next;
            // terms alternate between Q (odd k) and P (even k), with signs + - - + ...
            match (k as u64) % 4 {
                1 => q += term,
                2 => p -= term,
                3 => q -= term,
                _ => p += term,
            }
            if term.abs() < 1e-17 {
                break;
            }
        }
        // reduce x modulo 2 pi before subtracting the phase to keep the angle accurate
        // (fused multiply-add keeps k * 2pi exact, so the reduction stays accurate for very large x)
        let kk = (ax / (2.0 * PI)).round();
        let r = (-kk).mul_add(std::f64::consts::TAU, ax) - kk * 2.449_293_598_294_706_4e-16;
        let chi = r - (0.5 * n + 0.25) % 8.0 * PI;
        return sign * (2.0 / (PI * ax)).sqrt() * (p * chi.cos() - q * chi.sin());
    }
    // in between: periodic trapezoid rule on the integral representation (absolute accuracy ~1e-16)
    let m = (ax + n).ceil() + 50.0;
    if m > 5e7 {
        return f64::NAN; // order too large for this method (cost grows with |x| + n)
    }
    let m = m as usize;
    let h = 2.0 * PI / m as f64;
    let s: f64 = (0..m).map(|k| (n * k as f64 * h - ax * (k as f64 * h).sin()).cos()).sum();
    sign * s / m as f64
}

/// Exponentially scaled modified Bessel function `exp(-|x|) I_n(x)` of integer order (`scipy.special.ive`).
pub fn ive(n: i32, x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let n = n.unsigned_abs() as f64;
    let ax = x.abs();
    if ax <= 500.0 && (ax <= 50.0 || n > 3.0 * ax.sqrt()) {
        // power series (all terms positive); keeps full relative accuracy when I_n is tiny
        let half = 0.5 * ax;
        let mut term = (1..=n as u32).fold(1.0, |t, i| t * half / i as f64);
        let mut sum = term;
        let mut k = 0.0;
        while term > 1e-18 * sum && k < 5000.0 {
            k += 1.0;
            term *= half * half / (k * (k + n));
            sum += term;
        }
        let v = sum * (-ax).exp();
        return if x < 0.0 && (n as i64) % 2 == 1 { -v } else { v };
    }
    let m = (n + (100.0 * ax).sqrt()).ceil() as usize + 64;
    let h = 2.0 * PI / m as f64;
    let s: f64 = (0..m).map(|k| { let t = k as f64 * h; (ax * (t.cos() - 1.0)).exp() * (n * t).cos() }).sum();
    let v = s / m as f64;
    if x < 0.0 && (n as i64) % 2 == 1 { -v } else { v }
}

/// Exponentially scaled modified Bessel function of the second kind `exp(x) K_v(x)` for real order and `x > 0`
/// (`scipy.special.kve`), via `K_v(x) = int_0^inf e^{-x cosh t} cosh(v t) dt` (double-exponentially convergent).
pub fn kve(v: f64, x: f64) -> f64 {
    if x.is_nan() || v.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY;
    }
    // the integrand peaks with width ~ 1 / sqrt(|v|); keep at least four steps per width
    let h = 0.1f64.min(0.25 / (1.0 + v.abs()).sqrt());
    // cosh(v t) is evaluated in log space so a large order does not overflow before the damping factor wins
    let f = |t: f64| {
        let a = (v * t).abs();
        (-x * (t.cosh() - 1.0) + a + (-2.0 * a).exp().ln_1p() - std::f64::consts::LN_2).exp()
    };
    let mut sum = 0.5 * f(0.0);
    for k in 1..100_000 {
        let term = f(k as f64 * h);
        sum += term;
        if term < 1e-18 * sum && k > 5 {
            break;
        }
    }
    sum * h
}

/// Modified Bessel function of the second kind `K_v(x)` for real `v` and `x > 0`.
pub fn kv(v: f64, x: f64) -> f64 {
    kve(v, x) * (-x).exp()
}

/// Legendre polynomial `P_n(x)`.
pub fn legendre(n: u32, x: f64) -> f64 {
    let (mut p0, mut p1) = (1.0, x);
    match n {
        0 => 1.0,
        _ => {
            for k in 1..n {
                let kf = k as f64;
                (p0, p1) = (p1, ((2.0 * kf + 1.0) * x * p1 - kf * p0) / (kf + 1.0));
            }
            p1
        }
    }
}

/// Physicists' Hermite polynomial `H_n(x)`.
pub fn hermite(n: u32, x: f64) -> f64 {
    let (mut h0, mut h1) = (1.0, 2.0 * x);
    if n == 0 {
        return 1.0;
    }
    for k in 1..n {
        (h0, h1) = (h1, 2.0 * x * h1 - 2.0 * k as f64 * h0);
    }
    h1
}

/// Probabilists' Hermite polynomial `He_n(x)`.
pub fn hermite_e(n: u32, x: f64) -> f64 {
    let (mut h0, mut h1) = (1.0, x);
    if n == 0 {
        return 1.0;
    }
    for k in 1..n {
        (h0, h1) = (h1, x * h1 - k as f64 * h0);
    }
    h1
}

/// Laguerre polynomial `L_n(x)`.
pub fn laguerre(n: u32, x: f64) -> f64 {
    genlaguerre(n, 0.0, x)
}

/// Generalized Laguerre polynomial `L_n^{(alpha)}(x)`.
pub fn genlaguerre(n: u32, alpha: f64, x: f64) -> f64 {
    let (mut l0, mut l1) = (1.0, 1.0 + alpha - x);
    if n == 0 {
        return 1.0;
    }
    for k in 1..n {
        let kf = k as f64;
        (l0, l1) = (l1, ((2.0 * kf + 1.0 + alpha - x) * l1 - (kf + alpha) * l0) / (kf + 1.0));
    }
    l1
}

/// Chebyshev polynomial of the first kind `T_n(x)`.
pub fn chebyt(n: u32, x: f64) -> f64 {
    let (mut t0, mut t1) = (1.0, x);
    if n == 0 {
        return 1.0;
    }
    for _ in 1..n {
        (t0, t1) = (t1, 2.0 * x * t1 - t0);
    }
    t1
}

/// Chebyshev polynomial of the second kind `U_n(x)`.
pub fn chebyu(n: u32, x: f64) -> f64 {
    let (mut u0, mut u1) = (1.0, 2.0 * x);
    if n == 0 {
        return 1.0;
    }
    for _ in 1..n {
        (u0, u1) = (u1, 2.0 * x * u1 - u0);
    }
    u1
}

/// Gegenbauer (ultraspherical) polynomial `C_n^{(alpha)}(x)`.
pub fn gegenbauer(n: u32, alpha: f64, x: f64) -> f64 {
    let (mut c0, mut c1) = (1.0, 2.0 * alpha * x);
    if n == 0 {
        return 1.0;
    }
    for k in 1..n {
        let kf = k as f64;
        (c0, c1) = (c1, (2.0 * x * (kf + alpha) * c1 - (kf + 2.0 * alpha - 1.0) * c0) / (kf + 1.0));
    }
    c1
}

/// Jacobi polynomial `P_n^{(alpha, beta)}(x)`.
pub fn jacobi(n: u32, alpha: f64, beta: f64, x: f64) -> f64 {
    let (mut p0, mut p1) = (1.0, 0.5 * (alpha - beta + (alpha + beta + 2.0) * x));
    if n == 0 {
        return 1.0;
    }
    for k in 1..n {
        let kf = k as f64;
        let c = 2.0 * kf + alpha + beta;
        let a1 = 2.0 * (kf + 1.0) * (kf + alpha + beta + 1.0) * c;
        let a2 = (c + 1.0) * (alpha * alpha - beta * beta);
        let a3 = c * (c + 1.0) * (c + 2.0);
        let a4 = 2.0 * (kf + alpha) * (kf + beta) * (c + 2.0);
        (p0, p1) = (p1, ((a2 + a3 * x) * p1 - a4 * p0) / a1);
    }
    p1
}

fn unary_ext(
    t: &Tensor,
    name: &'static str,
    fwd: impl Fn(f64) -> f64,
    deriv: impl Fn(f64, f64) -> f64 + Send + Sync + 'static,
) -> Result<Tensor> {
    let dt = if matches!(t.dtype(), DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
        t.dtype()
    } else {
        DType::F32
    };
    let shape = t.shape().to_vec();
    let xv = t.to_vec::<f64>()?;
    let yv: Vec<f64> = xv.iter().map(|&x| fwd(x)).collect();
    let out = Tensor::from_slice_on(&yv, &shape, &t.device())?.to_dtype(dt)?;
    let (storage, layout) = (out.storage().clone(), out.layout().clone());
    let dev = t.device();
    Ok(Tensor::from_op(storage, layout, name, vec![t.clone()], move |g| {
        let gv = g.to_vec::<f64>()?;
        let grad_vals: Vec<f64> = gv
            .iter()
            .zip(xv.iter().zip(&yv))
            .map(|(&gi, (&x, &y))| gi * deriv(x, y))
            .collect();
        Ok(vec![Some(Tensor::from_slice_on(&grad_vals, &shape, &dev)?.to_dtype(dt)?)])
    }))
}

fn legendre_deriv(n: u32, x: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return 1.0;
    }
    let (mut p0, mut p1) = (1.0, x);
    let mut d1 = 1.0;
    for k in 2..=n {
        let kf = k as f64;
        let pk = ((2.0 * kf - 1.0) * x * p1 - (kf - 1.0) * p0) / kf;
        d1 = kf * p1 + x * d1;
        p0 = p1;
        p1 = pk;
    }
    d1
}

fn chebyu_deriv(n: u32, x: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return 2.0;
    }
    let (mut u0, mut u1) = (1.0, 2.0 * x);
    let (mut d0, mut d1) = (0.0, 2.0);
    for _ in 2..=n {
        let uk = 2.0 * x * u1 - u0;
        let dk = 2.0 * u1 + 2.0 * x * d1 - d0;
        u0 = u1;
        u1 = uk;
        d0 = d1;
        d1 = dk;
    }
    d1
}

fn laguerre_deriv(n: u32, x: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return -1.0;
    }
    let (mut l0, mut l1) = (1.0, 1.0 - x);
    let mut d1 = -1.0;
    for k in 2..=n {
        let kf = k as f64;
        let lk = ((2.0 * kf - 1.0 - x) * l1 - (kf - 1.0) * l0) / kf;
        d1 -= l1;
        l0 = l1;
        l1 = lk;
    }
    d1
}

impl Tensor {
    /// Scaled complementary error function `exp(x^2) * erfc(x)`.
    pub fn erfcx(&self) -> Result<Tensor> {
        let two_over_sqrt_pi = 2.0 / SQRT_PI;
        unary_ext(self, "erfcx", erfcx, move |x, y| 2.0 * x * y - two_over_sqrt_pi)
    }

    /// Principal branch of the Lambert W function `W_0(x)`.
    pub fn lambert_w(&self) -> Result<Tensor> {
        unary_ext(
            self,
            "lambert_w",
            |x| lambertw(x, 0),
            |x, w| {
                if x == 0.0 {
                    1.0
                } else {
                    w / (x * (1.0 + w))
                }
            },
        )
    }

    /// Modified Bessel function of the second kind of order 0, `K_0(x)`.
    pub fn modified_bessel_k0(&self) -> Result<Tensor> {
        unary_ext(self, "modified_bessel_k0", |x| kv(0.0, x), |x, _| -kv(1.0, x))
    }

    /// Modified Bessel function of the second kind of order 1, `K_1(x)`.
    pub fn modified_bessel_k1(&self) -> Result<Tensor> {
        unary_ext(
            self,
            "modified_bessel_k1",
            |x| kv(1.0, x),
            |x, k1| -kv(0.0, x) - k1 / x,
        )
    }

    /// Exponentially scaled modified Bessel function of the second kind of order 0, `exp(x) * K_0(x)`.
    pub fn scaled_modified_bessel_k0(&self) -> Result<Tensor> {
        unary_ext(
            self,
            "scaled_modified_bessel_k0",
            |x| kve(0.0, x),
            |x, k0e| k0e - kve(1.0, x),
        )
    }

    /// Exponentially scaled modified Bessel function of the second kind of order 1, `exp(x) * K_1(x)`.
    pub fn scaled_modified_bessel_k1(&self) -> Result<Tensor> {
        unary_ext(
            self,
            "scaled_modified_bessel_k1",
            |x| kve(1.0, x),
            |x, k1e| k1e - kve(0.0, x) - k1e / x,
        )
    }

    /// Physicist's Hermite polynomial `H_n(x)`.
    pub fn hermite_polynomial_h(&self, n: u32) -> Result<Tensor> {
        unary_ext(
            self,
            "hermite_polynomial_h",
            move |x| hermite(n, x),
            move |x, _| {
                if n == 0 {
                    0.0
                } else {
                    2.0 * (n as f64) * hermite(n - 1, x)
                }
            },
        )
    }

    /// Legendre polynomial `P_n(x)`.
    pub fn legendre_polynomial_p(&self, n: u32) -> Result<Tensor> {
        unary_ext(
            self,
            "legendre_polynomial_p",
            move |x| legendre(n, x),
            move |x, _| legendre_deriv(n, x),
        )
    }

    /// Chebyshev polynomial of the first kind `T_n(x)`.
    pub fn chebyshev_polynomial_t(&self, n: u32) -> Result<Tensor> {
        unary_ext(
            self,
            "chebyshev_polynomial_t",
            move |x| chebyt(n, x),
            move |x, _| {
                if n == 0 {
                    0.0
                } else {
                    (n as f64) * chebyu(n - 1, x)
                }
            },
        )
    }

    /// Chebyshev polynomial of the second kind `U_n(x)`.
    pub fn chebyshev_polynomial_u(&self, n: u32) -> Result<Tensor> {
        unary_ext(
            self,
            "chebyshev_polynomial_u",
            move |x| chebyu(n, x),
            move |x, _| chebyu_deriv(n, x),
        )
    }

    /// Laguerre polynomial `L_n(x)`.
    pub fn laguerre_polynomial_l(&self, n: u32) -> Result<Tensor> {
        unary_ext(
            self,
            "laguerre_polynomial_l",
            move |x| laguerre(n, x),
            move |x, _| laguerre_deriv(n, x),
        )
    }

    /// Complete elliptic integral of the first kind `K(m)`.
    pub fn ellipk(&self) -> Result<Tensor> {
        unary_ext(self, "ellipk", ellipk, |m, k| {
            if m.abs() < 1e-8 {
                PI / 8.0
            } else {
                (ellipe(m) - (1.0 - m) * k) / (2.0 * m * (1.0 - m))
            }
        })
    }

    /// Complete elliptic integral of the second kind `E(m)`.
    pub fn ellipe(&self) -> Result<Tensor> {
        unary_ext(self, "ellipe", ellipe, |m, e| {
            if m.abs() < 1e-8 {
                -PI / 8.0
            } else {
                (e - ellipk(m)) / (2.0 * m)
            }
        })
    }

    /// Exponential integral `E_1(x)`.
    pub fn expint_e1(&self) -> Result<Tensor> {
        unary_ext(self, "expint_e1", expint_e1, |x, _| -(-x).exp() / x)
    }

    /// Exponential integral `Ei(x)`.
    pub fn expint_ei(&self) -> Result<Tensor> {
        unary_ext(self, "expint_ei", expint_ei, |x, _| x.exp() / x)
    }

    /// Dawson's integral `D_+(x)`.
    pub fn dawsn(&self) -> Result<Tensor> {
        unary_ext(self, "dawsn", dawsn, |x, d| 1.0 - 2.0 * x * d)
    }

    /// Dilogarithm / Spence's function `spence(z)`.
    pub fn spence(&self) -> Result<Tensor> {
        unary_ext(self, "spence", spence, |z, _| {
            if (z - 1.0).abs() < 1e-12 {
                -1.0
            } else {
                z.ln() / (1.0 - z)
            }
        })
    }

    /// Fresnel sine and cosine integrals `(S(x), C(x))`.
    pub fn fresnel(&self) -> Result<(Tensor, Tensor)> {
        let s = unary_ext(
            self,
            "fresnel_s",
            |x| fresnel(x).0,
            |x, _| (FRAC_PI_2 * x * x).sin(),
        )?;
        let c = unary_ext(
            self,
            "fresnel_c",
            |x| fresnel(x).1,
            |x, _| (FRAC_PI_2 * x * x).cos(),
        )?;
        Ok((s, c))
    }

    /// Sine and cosine integrals `(Si(x), Ci(x))`.
    pub fn sici(&self) -> Result<(Tensor, Tensor)> {
        let si = unary_ext(
            self,
            "sici_si",
            |x| sici(x).0,
            |x, _| if x.abs() < 1e-8 { 1.0 - x * x / 6.0 } else { x.sin() / x },
        )?;
        let ci = unary_ext(
            self,
            "sici_ci",
            |x| sici(x).1,
            |x, _| x.cos() / x,
        )?;
        Ok((si, ci))
    }

    /// Gegenbauer (ultraspherical) polynomial `C_n^{(alpha)}(x)`.
    pub fn gegenbauer(&self, n: u32, alpha: f64) -> Result<Tensor> {
        unary_ext(
            self,
            "gegenbauer",
            move |x| gegenbauer(n, alpha, x),
            move |x, _| {
                if n == 0 {
                    0.0
                } else {
                    2.0 * alpha * gegenbauer(n - 1, alpha + 1.0, x)
                }
            },
        )
    }

    /// Jacobi polynomial `P_n^{(alpha, beta)}(x)`.
    pub fn jacobi(&self, n: u32, alpha: f64, beta: f64) -> Result<Tensor> {
        unary_ext(
            self,
            "jacobi",
            move |x| jacobi(n, alpha, beta, x),
            move |x, _| {
                if n == 0 {
                    0.0
                } else {
                    0.5 * ((n as f64) + alpha + beta + 1.0)
                        * jacobi(n - 1, alpha + 1.0, beta + 1.0, x)
                }
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn known_values() {
        close(erfc(0.5), 0.479_500_122_186_953_5, 1e-14);
        close(erfc(5.0), 1.537_459_794_428_035e-12, 1e-12);
        close(erfcx(10.0), 0.056_140_992_743_822, 1e-12);
        close(ellipk(0.5), 1.854_074_677_301_372, 1e-14);
        close(ellipe(0.5), 1.350_643_881_047_675, 1e-14);
        close(expint_e1(1.0), 0.219_383_934_395_520_3, 1e-14);
        close(expint_ei(1.0), 1.895_117_816_355_936_8, 1e-14);
        close(lambertw(1.0, 0), 0.567_143_290_409_783_8, 1e-14);
        close(lambertw(-0.2, -1), -2.542_641_357_773_527, 1e-13);
        let (s, c) = fresnel(1.0);
        close(s, 0.438_259_147_390_354_8, 1e-13);
        close(c, 0.779_893_400_376_822_8, 1e-13);
        close(dawsn(1.0), 0.538_079_506_912_768_4, 1e-13);
        let (si, ci) = sici(3.0);
        close(si, 1.848_652_527_999_468, 1e-13);
        close(ci, 0.119_629_786_008_000_3, 1e-12);
        close(spence(0.5), 0.582_240_526_465_012_6, 1e-13);
        close(jn(2, 3.5), 0.458_629_184_194_307_5, 1e-12);
        close(kv(0.0, 1.0), 0.421_024_438_240_708_3, 1e-13);
        close(legendre(3, 0.4), -0.44, 1e-14);
        close(hermite(3, 0.7), -5.656, 1e-12);
        close(chebyt(4, 0.3), 0.3448, 1e-13); // 8x^4 - 8x^2 + 1
    }
}
