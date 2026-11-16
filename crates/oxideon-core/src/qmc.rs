//! Low-discrepancy and stratified sampling: [`halton`], [`sobol`] (Joe-Kuo direction numbers for the first 6
//! dimensions), [`latin_hypercube`], [`poisson_disk`], and Monte-Carlo integration with [`mc_integrate`] (plain,
//! antithetic) and [`qmc_integrate`], plus the star-discrepancy proxy [`l2_star_discrepancy`].

use crate::error::{Error, Result};
use crate::rng::Rng;

const PRIMES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

fn radical_inverse(mut i: u64, base: u64) -> f64 {
    let (mut f, mut r) = (1.0, 0.0);
    while i > 0 {
        f /= base as f64;
        r += f * (i % base) as f64;
        i /= base;
    }
    r
}

/// First `n` points (skipping the origin) of the Halton sequence in `dim <= 12` dimensions.
pub fn halton(n: usize, dim: usize) -> Result<Vec<Vec<f64>>> {
    if dim == 0 || dim > PRIMES.len() {
        return Err(Error::Msg(format!("halton: dim must be in 1..={}", PRIMES.len())));
    }
    Ok((1..=n as u64).map(|i| (0..dim).map(|d| radical_inverse(i, PRIMES[d])).collect()).collect())
}

/// Sobol sequence (Gray-code construction) in `dim <= 6` dimensions; starts from the origin like the reference
/// implementations.
pub fn sobol(n: usize, dim: usize) -> Result<Vec<Vec<f64>>> {
    // (degree s, polynomial coefficient a, initial direction numbers m) for dimensions 2..=6 (Joe & Kuo)
    const PARAMS: [(u32, u32, [u32; 4]); 5] = [(1, 0, [1, 0, 0, 0]), (2, 1, [1, 3, 0, 0]), (3, 1, [1, 3, 1, 0]), (3, 2, [1, 1, 1, 0]), (4, 1, [1, 1, 3, 3])];
    if dim == 0 || dim > 6 {
        return Err(Error::Msg("sobol: dim must be in 1..=6".into()));
    }
    const BITS: usize = 32;
    let mut v: Vec<[u32; BITS]> = Vec::new();
    // dimension 1: van der Corput
    let mut first = [0u32; BITS];
    (0..BITS).for_each(|i| first[i] = 1u32 << (31 - i));
    v.push(first);
    for &(s, a, m) in PARAMS.iter().take(dim - 1) {
        let mut dir = [0u32; BITS];
        for i in 0..s as usize {
            dir[i] = m[i] << (31 - i);
        }
        for i in s as usize..BITS {
            let mut x = dir[i - s as usize] ^ (dir[i - s as usize] >> s);
            for k in 1..s {
                if a >> (s - 1 - k) & 1 == 1 {
                    x ^= dir[i - k as usize];
                }
            }
            dir[i] = x;
        }
        v.push(dir);
    }
    let mut state = vec![0u32; dim];
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(state.iter().map(|&s| f64::from(s) / 4_294_967_296.0).collect());
        let c = (i as u32 + 1).trailing_zeros() as usize; // index of the lowest zero bit of i
        for d in 0..dim {
            state[d] ^= v[d][c.min(BITS - 1)];
        }
    }
    Ok(out)
}

/// Latin hypercube sample: each of the `n` strata along every axis contains exactly one point.
pub fn latin_hypercube(n: usize, dim: usize, seed: u64) -> Result<Vec<Vec<f64>>> {
    if n == 0 || dim == 0 {
        return Err(Error::Msg("latin_hypercube: n and dim must be positive".into()));
    }
    let mut rng = Rng::new(seed);
    let mut pts = vec![vec![0.0; dim]; n];
    for d in 0..dim {
        let mut perm: Vec<usize> = (0..n).collect();
        rng.shuffle(&mut perm);
        for (i, &stratum) in perm.iter().enumerate() {
            pts[i][d] = (stratum as f64 + rng.next_f64()) / n as f64;
        }
    }
    Ok(pts)
}

/// Poisson-disk sampling in the unit square (dart throwing with a minimum distance), up to `max_points`.
pub fn poisson_disk(min_dist: f64, max_points: usize, attempts: usize, seed: u64) -> Result<Vec<(f64, f64)>> {
    if min_dist <= 0.0 || min_dist >= 1.0 || max_points == 0 {
        return Err(Error::Msg("poisson_disk: need 0 < min_dist < 1 and max_points >= 1".into()));
    }
    let mut rng = Rng::new(seed);
    let mut pts: Vec<(f64, f64)> = Vec::new();
    for _ in 0..attempts {
        if pts.len() >= max_points {
            break;
        }
        let c = (rng.next_f64(), rng.next_f64());
        if pts.iter().all(|p| (p.0 - c.0).hypot(p.1 - c.1) >= min_dist) {
            pts.push(c);
        }
    }
    Ok(pts)
}

/// Plain Monte-Carlo estimate of the integral of `f` over the unit hypercube with its standard error;
/// `antithetic` pairs every point `u` with `1 - u` (variance reduction for monotone integrands).
pub fn mc_integrate<F: Fn(&[f64]) -> f64>(f: F, dim: usize, n: usize, antithetic: bool, seed: u64) -> Result<(f64, f64)> {
    if dim == 0 || n < 2 {
        return Err(Error::Msg("mc_integrate: need dim >= 1 and n >= 2".into()));
    }
    let mut rng = Rng::new(seed);
    let samples: Vec<f64> = (0..n)
        .map(|_| {
            let u: Vec<f64> = (0..dim).map(|_| rng.next_f64()).collect();
            if antithetic { 0.5 * (f(&u) + f(&u.iter().map(|v| 1.0 - v).collect::<Vec<_>>())) } else { f(&u) }
        })
        .collect();
    let mean = samples.iter().sum::<f64>() / n as f64;
    let var = samples.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    Ok((mean, (var / n as f64).sqrt()))
}

/// Quasi-Monte-Carlo integral over the unit hypercube using Sobol points (`dim <= 6`) or Halton points.
pub fn qmc_integrate<F: Fn(&[f64]) -> f64>(f: F, dim: usize, n: usize) -> Result<f64> {
    let pts = if dim <= 6 { sobol(n, dim)? } else { halton(n, dim)? };
    Ok(pts.iter().map(|p| f(p)).sum::<f64>() / n as f64)
}

/// Warnock's L2 star discrepancy of a point set in `[0, 1]^d` (lower = more uniform).
pub fn l2_star_discrepancy(points: &[Vec<f64>]) -> Result<f64> {
    let d = points.first().map_or(0, Vec::len);
    if points.is_empty() || d == 0 || points.iter().any(|p| p.len() != d || p.iter().any(|v| !(0.0..=1.0).contains(v))) {
        return Err(Error::Msg("l2_star_discrepancy: need points in the unit hypercube".into()));
    }
    let n = points.len() as f64;
    let term1 = 3f64.powi(-(d as i32));
    let term2 = points.iter().map(|p| p.iter().map(|x| 1.0 - x * x).product::<f64>()).sum::<f64>() / n * 2f64.powi(1 - d as i32);
    let term3 = points.iter().map(|p| points.iter().map(|q| p.iter().zip(q).map(|(a, b)| 1.0 - a.max(*b)).product::<f64>()).sum::<f64>()).sum::<f64>() / (n * n);
    Ok((term1 - term2 + term3).max(0.0).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequences_have_the_expected_prefixes() {
        // base-2 radical inverses
        let h = halton(4, 2).unwrap();
        assert_eq!(h[0], vec![0.5, 1.0 / 3.0]);
        assert_eq!(h[1], vec![0.25, 2.0 / 3.0]);
        assert_eq!(h[2], vec![0.75, 1.0 / 9.0]);
        // Sobol (scipy, unscrambled): first points of dimension 1 and 2
        let s = sobol(5, 2).unwrap();
        assert_eq!(s[0], vec![0.0, 0.0]);
        assert_eq!(s[1], vec![0.5, 0.5]);
        assert_eq!(s[2], vec![0.75, 0.25]);
        assert_eq!(s[3], vec![0.25, 0.75]);
        assert_eq!(s[4], vec![0.375, 0.375]);
        assert!(halton(3, 0).is_err() && halton(3, 13).is_err() && sobol(3, 7).is_err());
    }

    #[test]
    fn stratification_and_spacing_guarantees() {
        let lhs = latin_hypercube(20, 3, 1).unwrap();
        for d in 0..3 {
            let mut strata: Vec<usize> = lhs.iter().map(|p| (p[d] * 20.0) as usize).collect();
            strata.sort_unstable();
            assert_eq!(strata, (0..20).collect::<Vec<_>>(), "axis {d}");
        }
        let pd = poisson_disk(0.1, 200, 20_000, 2).unwrap();
        assert!(pd.len() > 30);
        assert!(pd.iter().enumerate().all(|(i, p)| pd[..i].iter().all(|q| (p.0 - q.0).hypot(p.1 - q.1) >= 0.1)));
        assert!(latin_hypercube(0, 2, 0).is_err() && poisson_disk(1.5, 10, 10, 0).is_err());
    }

    #[test]
    fn low_discrepancy_beats_random_points() {
        let n = 256;
        let sob = l2_star_discrepancy(&sobol(n, 3).unwrap()).unwrap();
        let hal = l2_star_discrepancy(&halton(n, 3).unwrap()).unwrap();
        let mut rng = Rng::new(7);
        let rand: Vec<Vec<f64>> = (0..n).map(|_| (0..3).map(|_| rng.next_f64()).collect()).collect();
        let rnd = l2_star_discrepancy(&rand).unwrap();
        assert!(sob < rnd * 0.6 && hal < rnd * 0.8, "sobol {sob}, halton {hal}, random {rnd}");
        assert!(l2_star_discrepancy(&[vec![2.0]]).is_err());
    }

    #[test]
    fn integration_accuracy() {
        // integral of x0 * x1 * x2 over the unit cube is 1/8; of exp(x) (1-dim) is e - 1
        let f = |x: &[f64]| x.iter().product::<f64>();
        let qmc = qmc_integrate(f, 3, 4096).unwrap();
        assert!((qmc - 0.125).abs() < 2e-3, "{qmc}");
        let (mc, se) = mc_integrate(f, 3, 4096, false, 3).unwrap();
        assert!((mc - 0.125).abs() < 4.0 * se, "{mc} +- {se}");
        assert!((qmc - 0.125).abs() < (mc - 0.125).abs() + 1e-3, "QMC is at least as accurate here");
        // antithetic variates cut the standard error for a monotone integrand
        let g = |x: &[f64]| x[0].exp();
        let (a, se_a) = mc_integrate(g, 1, 4000, true, 4).unwrap();
        let (_, se_p) = mc_integrate(g, 1, 4000, false, 4).unwrap();
        assert!((a - (std::f64::consts::E - 1.0)).abs() < 4.0 * se_a + 1e-3);
        assert!(se_a < se_p * 0.2, "{se_a} vs {se_p}");
        assert!(mc_integrate(g, 0, 10, false, 0).is_err() && qmc_integrate(g, 0, 10).is_err());
    }
}
