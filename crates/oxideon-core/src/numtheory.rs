//! Number theory and combinatorics on integers: [`mod_pow`], [`ext_gcd`], [`mod_inverse`], [`crt`],
//! deterministic [`is_prime`] (Miller-Rabin for all `u64`), [`factorize`] (Pollard rho), [`sieve`], [`totient`],
//! checked [`binomial`], [`catalan`], [`stirling2`], [`partition_count`], and [`next_permutation`].

use crate::error::{Error, Result};

fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

/// `base^exp mod m` without overflow.
pub fn mod_pow(mut base: u64, mut exp: u64, m: u64) -> Result<u64> {
    if m == 0 {
        return Err(Error::Msg("mod_pow: modulus must be positive".into()));
    }
    let mut r = 1 % m;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            r = mul_mod(r, base, m);
        }
        base = mul_mod(base, base, m);
        exp >>= 1;
    }
    Ok(r)
}

/// Extended Euclid: `(g, x, y)` with `a x + b y = g = gcd(a, b)`.
pub fn ext_gcd(a: i128, b: i128) -> (i128, i128, i128) {
    if b == 0 {
        (a.abs(), a.signum(), 0)
    } else {
        let (g, x, y) = ext_gcd(b, a.rem_euclid(b));
        (g, y, x - (a.div_euclid(b)) * y)
    }
}

/// Modular inverse of `a` mod `m` (errors when `gcd(a, m) != 1`).
pub fn mod_inverse(a: u64, m: u64) -> Result<u64> {
    if m == 0 {
        return Err(Error::Msg("mod_inverse: modulus must be positive".into()));
    }
    let (g, x, _) = ext_gcd(a as i128, m as i128);
    if g != 1 {
        return Err(Error::Msg(format!("mod_inverse: {a} has no inverse modulo {m}")));
    }
    Ok(x.rem_euclid(m as i128) as u64)
}

/// Chinese remainder theorem for pairwise-coprime moduli: the unique `x` mod their product with `x = r_i mod m_i`.
pub fn crt(residues: &[u64], moduli: &[u64]) -> Result<(u64, u64)> {
    if residues.len() != moduli.len() || moduli.is_empty() {
        return Err(Error::Msg("crt: need matching non-empty residues and moduli".into()));
    }
    let (mut x, mut m) = (0u128, 1u128);
    for (&r, &mi) in residues.iter().zip(moduli) {
        let inv = mod_inverse((m % mi as u128) as u64, mi).map_err(|_| Error::Msg("crt: moduli must be pairwise coprime".into()))? as u128;
        let t = ((r as u128 % mi as u128) + mi as u128 - x % mi as u128) % mi as u128 * inv % mi as u128;
        x += m * t;
        m *= mi as u128;
        if m > u64::MAX as u128 {
            return Err(Error::Msg("crt: the combined modulus overflows u64".into()));
        }
    }
    Ok((x as u64, m as u64))
}

/// Deterministic primality test for every `u64` (Miller-Rabin with a fixed witness set).
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for p in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n % p == 0 {
            return n == p;
        }
    }
    let (mut d, mut s) = (n - 1, 0);
    while d % 2 == 0 {
        d /= 2;
        s += 1;
    }
    'witness: for a in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        let mut x = mod_pow(a, d, n).unwrap();
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 0..s - 1 {
            x = mul_mod(x, x, n);
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn pollard_rho(n: u64) -> u64 {
    if n % 2 == 0 {
        return 2;
    }
    let mut c = 1u64;
    loop {
        let f = |x: u64| (mul_mod(x, x, n) + c) % n;
        let (mut x, mut y, mut d) = (2u64, 2u64, 1u64);
        while d == 1 {
            x = f(x);
            y = f(f(y));
            d = gcd(x.abs_diff(y), n);
        }
        if d != n {
            return d;
        }
        c += 1;
    }
}

/// Prime factorization with multiplicity, ascending (`factorize(1)` is empty).
pub fn factorize(n: u64) -> Result<Vec<u64>> {
    if n == 0 {
        return Err(Error::Msg("factorize: 0 has no factorization".into()));
    }
    fn rec(n: u64, out: &mut Vec<u64>) {
        if n == 1 {
            return;
        }
        if is_prime(n) {
            out.push(n);
            return;
        }
        let d = pollard_rho(n);
        rec(d, out);
        rec(n / d, out);
    }
    let mut out = Vec::new();
    rec(n, &mut out);
    out.sort_unstable();
    Ok(out)
}

/// All primes `<= n` (sieve of Eratosthenes).
pub fn sieve(n: usize) -> Vec<usize> {
    if n < 2 {
        return Vec::new();
    }
    let mut is = vec![true; n + 1];
    is[0] = false;
    is[1] = false;
    let mut i = 2;
    while i * i <= n {
        if is[i] {
            (i * i..=n).step_by(i).for_each(|j| is[j] = false);
        }
        i += 1;
    }
    (0..=n).filter(|&i| is[i]).collect()
}

/// Euler's totient.
pub fn totient(n: u64) -> Result<u64> {
    let f = factorize(n)?;
    let mut distinct = f.clone();
    distinct.dedup();
    Ok(distinct.iter().fold(n, |acc, &p| acc / p * (p - 1)))
}

/// `C(n, k)` with overflow detection.
pub fn binomial(n: u64, k: u64) -> Result<u128> {
    if k > n {
        return Ok(0);
    }
    let k = k.min(n - k);
    let mut r: u128 = 1;
    for i in 1..=k as u128 {
        r = r.checked_mul(n as u128 - k as u128 + i).ok_or_else(|| Error::Msg("binomial: overflow".into()))? / i;
    }
    Ok(r)
}

/// The `n`-th Catalan number.
pub fn catalan(n: u64) -> Result<u128> {
    Ok(binomial(2 * n, n)? / (n as u128 + 1))
}

/// Stirling number of the second kind `S(n, k)` (partitions of `n` items into `k` non-empty blocks).
pub fn stirling2(n: usize, k: usize) -> Result<u128> {
    let mut row = vec![0u128; k + 1];
    row[0] = 1; // S(0, 0)
    for i in 1..=n {
        for j in (1..=k.min(i)).rev() {
            row[j] = row[j].checked_mul(j as u128).and_then(|v| v.checked_add(row[j - 1])).ok_or_else(|| Error::Msg("stirling2: overflow".into()))?;
        }
        row[0] = 0;
    }
    Ok(row[k])
}

/// Number of integer partitions of `n` (Euler's pentagonal recurrence).
pub fn partition_count(n: usize) -> Result<u128> {
    let mut p = vec![0u128; n + 1];
    p[0] = 1;
    for i in 1..=n {
        let (mut k, mut sum) = (1i64, 0i128);
        loop {
            let g1 = (k * (3 * k - 1) / 2) as usize;
            if g1 > i {
                break;
            }
            let sign = if k % 2 == 1 { 1 } else { -1 };
            sum += sign * p[i - g1] as i128;
            let g2 = (k * (3 * k + 1) / 2) as usize;
            if g2 <= i {
                sum += sign * p[i - g2] as i128;
            }
            k += 1;
        }
        p[i] = u128::try_from(sum).map_err(|_| Error::Msg("partition_count: overflow".into()))?;
    }
    Ok(p[n])
}

/// Rearranges `a` into the next lexicographic permutation; `false` (and sorted ascending) after the last one.
pub fn next_permutation<T: Ord>(a: &mut [T]) -> bool {
    if a.len() < 2 {
        return false;
    }
    let mut i = a.len() - 1;
    while i > 0 && a[i - 1] >= a[i] {
        i -= 1;
    }
    if i == 0 {
        a.reverse();
        return false;
    }
    let mut j = a.len() - 1;
    while a[j] <= a[i - 1] {
        j -= 1;
    }
    a.swap(i - 1, j);
    a[i..].reverse();
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modular_arithmetic() {
        assert_eq!(mod_pow(2, 10, 1000).unwrap(), 24);
        assert_eq!(mod_pow(7, 0, 13).unwrap(), 1);
        assert!(mod_pow(u64::MAX - 1, u64::MAX - 2, u64::MAX).unwrap() < u64::MAX);
        assert!(mod_pow(2, 3, 0).is_err());
        assert_eq!(mod_inverse(3, 11).unwrap(), 4);
        assert_eq!(mod_inverse(17, 3120).unwrap(), 2753); // RSA textbook example
        assert!(mod_inverse(6, 9).is_err());
        let (g, x, y) = ext_gcd(240, 46);
        assert_eq!((g, 240 * x + 46 * y), (2, 2));
        // Sunzi: x = 2 mod 3, 3 mod 5, 2 mod 7 -> 23 mod 105
        assert_eq!(crt(&[2, 3, 2], &[3, 5, 7]).unwrap(), (23, 105));
        assert!(crt(&[1, 1], &[4, 6]).is_err());
    }

    #[test]
    fn primes_and_factorization() {
        assert!([2u64, 3, 5, 7919, 1_000_000_007, 18_446_744_073_709_551_557].iter().all(|&p| is_prime(p)));
        assert!([0u64, 1, 4, 561, 1_000_000_007 * 3, 3_215_031_751].iter().all(|&c| !is_prime(c))); // Carmichael / strong pseudoprime
        assert_eq!(sieve(30), vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
        assert_eq!(sieve(100_000).len(), 9592);
        assert_eq!(factorize(360).unwrap(), vec![2, 2, 2, 3, 3, 5]);
        assert_eq!(factorize(1).unwrap(), Vec::<u64>::new());
        let big = 1_000_000_007u64 * 998_244_353;
        assert_eq!(factorize(big).unwrap(), vec![998_244_353, 1_000_000_007]);
        assert!(factorize(0).is_err());
        assert_eq!(totient(36).unwrap(), 12);
        assert_eq!(totient(97).unwrap(), 96);
        // Euler's theorem: a^phi(n) = 1 mod n
        assert_eq!(mod_pow(5, totient(36).unwrap(), 36).unwrap(), 1);
    }

    #[test]
    fn combinatorics() {
        assert_eq!(binomial(10, 3).unwrap(), 120);
        assert_eq!(binomial(52, 5).unwrap(), 2_598_960);
        assert_eq!(binomial(5, 7).unwrap(), 0);
        assert_eq!(binomial(100, 50).unwrap(), 100_891_344_545_564_193_334_812_497_256);
        assert!(binomial(1000, 500).is_err());
        assert_eq!((0..8).map(|n| catalan(n).unwrap()).collect::<Vec<_>>(), vec![1, 1, 2, 5, 14, 42, 132, 429]);
        assert_eq!(stirling2(5, 2).unwrap(), 15);
        assert_eq!(stirling2(10, 3).unwrap(), 9330);
        assert_eq!(stirling2(0, 0).unwrap(), 1);
        assert_eq!(partition_count(5).unwrap(), 7);
        assert_eq!(partition_count(100).unwrap(), 190_569_292);
        let mut p = [1, 2, 3];
        let mut all = vec![p.to_vec()];
        while next_permutation(&mut p) {
            all.push(p.to_vec());
        }
        assert_eq!(all, vec![vec![1, 2, 3], vec![1, 3, 2], vec![2, 1, 3], vec![2, 3, 1], vec![3, 1, 2], vec![3, 2, 1]]);
        assert_eq!(p, [1, 2, 3], "wraps to sorted order");
    }
}
