//! Streaming and probabilistic data structures: [`BloomFilter`], [`CountMinSketch`], [`HyperLogLog`],
//! [`Reservoir`] sampling, [`TDigest`]-style quantile summaries ([`StreamingQuantiles`]), [`SpaceSaving`]
//! heavy hitters, and Welford [`RunningStats`].

use crate::error::{Error, Result};
use crate::rng::Rng;

/// 64-bit FNV-1a with a seed mixed in, finalized with a splitmix step for good bit dispersion.
fn hash64(data: &[u8], seed: u64) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h = (h ^ (h >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^ (h >> 31)
}

/// Bloom filter sized for `capacity` items at false-positive rate `fp_rate`.
#[derive(Debug, Clone)]
pub struct BloomFilter {
    bits: Vec<u64>,
    n_bits: usize,
    k: usize,
    inserted: usize,
}

impl BloomFilter {
    pub fn new(capacity: usize, fp_rate: f64) -> Result<Self> {
        if capacity == 0 || !(0.0 < fp_rate && fp_rate < 1.0) {
            return Err(Error::Msg("BloomFilter: need capacity >= 1 and 0 < fp_rate < 1".into()));
        }
        let ln2 = std::f64::consts::LN_2;
        let n_bits = ((-(capacity as f64) * fp_rate.ln() / (ln2 * ln2)).ceil() as usize).max(8);
        let k = ((n_bits as f64 / capacity as f64 * ln2).round() as usize).clamp(1, 32);
        Ok(Self { bits: vec![0; n_bits.div_ceil(64)], n_bits, k, inserted: 0 })
    }

    fn positions<'a>(&'a self, item: &'a [u8]) -> impl Iterator<Item = usize> + 'a {
        // double hashing: h1 + i * h2
        let (h1, h2) = (hash64(item, 1), hash64(item, 2) | 1);
        (0..self.k).map(move |i| (h1.wrapping_add((i as u64).wrapping_mul(h2)) % self.n_bits as u64) as usize)
    }

    pub fn insert(&mut self, item: &[u8]) {
        let pos: Vec<usize> = self.positions(item).collect();
        pos.into_iter().for_each(|p| self.bits[p / 64] |= 1 << (p % 64));
        self.inserted += 1;
    }

    /// `false` means definitely absent; `true` means probably present.
    pub fn contains(&self, item: &[u8]) -> bool {
        self.positions(item).all(|p| self.bits[p / 64] >> (p % 64) & 1 == 1)
    }

    /// Estimated current false-positive probability.
    pub fn estimated_fp_rate(&self) -> f64 {
        (1.0 - (-(self.k as f64) * self.inserted as f64 / self.n_bits as f64).exp()).powi(self.k as i32)
    }
}

/// Count-Min sketch: over-estimates item counts within `epsilon * total` with probability `1 - delta`.
#[derive(Debug, Clone)]
pub struct CountMinSketch {
    table: Vec<Vec<u64>>,
    width: usize,
}

impl CountMinSketch {
    pub fn new(epsilon: f64, delta: f64) -> Result<Self> {
        if !(0.0 < epsilon && epsilon < 1.0 && 0.0 < delta && delta < 1.0) {
            return Err(Error::Msg("CountMinSketch: epsilon and delta must lie in (0, 1)".into()));
        }
        let width = (std::f64::consts::E / epsilon).ceil() as usize;
        let depth = (1.0 / delta).ln().ceil().max(1.0) as usize;
        Ok(Self { table: vec![vec![0; width]; depth], width })
    }

    pub fn add(&mut self, item: &[u8], count: u64) {
        for (row, t) in self.table.iter_mut().enumerate() {
            let c = (hash64(item, row as u64 + 11) % self.width as u64) as usize;
            t[c] += count;
        }
    }

    pub fn estimate(&self, item: &[u8]) -> u64 {
        self.table.iter().enumerate().map(|(row, t)| t[(hash64(item, row as u64 + 11) % self.width as u64) as usize]).min().unwrap_or(0)
    }
}

/// HyperLogLog cardinality estimator with `2^precision` registers (standard error about `1.04 / sqrt(m)`).
#[derive(Debug, Clone)]
pub struct HyperLogLog {
    registers: Vec<u8>,
    p: u32,
}

impl HyperLogLog {
    pub fn new(precision: u32) -> Result<Self> {
        if !(4..=16).contains(&precision) {
            return Err(Error::Msg("HyperLogLog: precision must be in 4..=16".into()));
        }
        Ok(Self { registers: vec![0; 1 << precision], p: precision })
    }

    pub fn add(&mut self, item: &[u8]) {
        let h = hash64(item, 7);
        let idx = (h >> (64 - self.p)) as usize;
        let rest = (h << self.p) | (1u64 << (self.p - 1)); // sentinel bit bounds the rank
        let rank = rest.leading_zeros() as u8 + 1;
        if rank > self.registers[idx] {
            self.registers[idx] = rank;
        }
    }

    /// Merges another sketch of the same precision (union of the two streams).
    pub fn merge(&mut self, other: &Self) -> Result<()> {
        if self.p != other.p {
            return Err(Error::Msg("HyperLogLog::merge: precision mismatch".into()));
        }
        self.registers.iter_mut().zip(&other.registers).for_each(|(a, b)| *a = (*a).max(*b));
        Ok(())
    }

    pub fn estimate(&self) -> f64 {
        let m = self.registers.len() as f64;
        let alpha = match self.registers.len() {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / m),
        };
        let raw = alpha * m * m / self.registers.iter().map(|&r| 2f64.powi(-i32::from(r))).sum::<f64>();
        let zeros = self.registers.iter().filter(|&&r| r == 0).count() as f64;
        if raw <= 2.5 * m && zeros > 0.0 { m * (m / zeros).ln() } else { raw }
    }
}

/// Reservoir sampling (Algorithm R): a uniform sample of `k` items from a stream of unknown length.
#[derive(Debug, Clone)]
pub struct Reservoir<T> {
    items: Vec<T>,
    k: usize,
    seen: u64,
    rng: Rng,
}

impl<T> Reservoir<T> {
    pub fn new(k: usize, seed: u64) -> Result<Self> {
        if k == 0 {
            return Err(Error::Msg("Reservoir: k must be positive".into()));
        }
        Ok(Self { items: Vec::with_capacity(k), k, seen: 0, rng: Rng::new(seed) })
    }

    pub fn push(&mut self, item: T) {
        self.seen += 1;
        if self.items.len() < self.k {
            self.items.push(item);
        } else {
            let j = self.rng.next_u64() % self.seen;
            if (j as usize) < self.k {
                self.items[j as usize] = item;
            }
        }
    }

    pub fn sample(&self) -> &[T] {
        &self.items
    }
}

/// Merging-centroid quantile summary (a simplified t-digest): constant memory, accurate at the tails.
#[derive(Debug, Clone)]
pub struct StreamingQuantiles {
    centroids: Vec<(f64, f64)>, // (mean, weight), sorted by mean after compress
    buffer: Vec<f64>,
    compression: f64,
    total: f64,
}

impl StreamingQuantiles {
    pub fn new(compression: f64) -> Result<Self> {
        if compression < 10.0 {
            return Err(Error::Msg("StreamingQuantiles: compression must be at least 10".into()));
        }
        Ok(Self { centroids: Vec::new(), buffer: Vec::new(), compression, total: 0.0 })
    }

    pub fn add(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        self.buffer.push(x);
        self.total += 1.0;
        if self.buffer.len() >= (self.compression * 10.0) as usize {
            self.compress();
        }
    }

    fn compress(&mut self) {
        let mut all: Vec<(f64, f64)> = std::mem::take(&mut self.centroids);
        all.extend(self.buffer.drain(..).map(|x| (x, 1.0)));
        all.sort_by(|a, b| a.0.total_cmp(&b.0));
        let total: f64 = all.iter().map(|c| c.1).sum();
        let mut out: Vec<(f64, f64)> = Vec::new();
        let mut cum = 0.0;
        for (m, w) in all {
            if let Some(last) = out.last_mut() {
                let q = (cum + last.1 + w / 2.0) / total;
                // size limit shrinks near the tails so extreme quantiles stay precise
                let limit = 4.0 * total * q * (1.0 - q) / self.compression;
                if last.1 + w <= limit.max(1.0) {
                    last.0 = (last.0 * last.1 + m * w) / (last.1 + w);
                    last.1 += w;
                    continue;
                }
                cum += last.1;
            }
            out.push((m, w));
        }
        self.centroids = out;
    }

    /// Estimated `q`-quantile (`q` in `[0, 1]`); `None` before any data.
    pub fn quantile(&mut self, q: f64) -> Option<f64> {
        if self.total == 0.0 {
            return None;
        }
        self.compress();
        let target = q.clamp(0.0, 1.0) * self.total;
        let mut cum = 0.0;
        for (i, &(m, w)) in self.centroids.iter().enumerate() {
            if cum + w >= target || i + 1 == self.centroids.len() {
                // interpolate between neighbouring centroid means by position inside this centroid
                let prev = if i == 0 { m } else { self.centroids[i - 1].0 };
                let frac = ((target - cum) / w).clamp(0.0, 1.0);
                return Some(if i == 0 && target <= w / 2.0 { m } else { prev + (m - prev) * frac });
            }
            cum += w;
        }
        None
    }

    pub fn n_centroids(&self) -> usize {
        self.centroids.len() + self.buffer.len()
    }
}

/// Space-Saving heavy hitters: tracks the `k` most frequent items of a stream with bounded over-estimation.
#[derive(Debug, Clone)]
pub struct SpaceSaving {
    counters: Vec<(Vec<u8>, u64, u64)>, // (item, count, error)
    k: usize,
}

impl SpaceSaving {
    pub fn new(k: usize) -> Result<Self> {
        if k == 0 {
            return Err(Error::Msg("SpaceSaving: k must be positive".into()));
        }
        Ok(Self { counters: Vec::new(), k })
    }

    pub fn add(&mut self, item: &[u8]) {
        if let Some(c) = self.counters.iter_mut().find(|c| c.0 == item) {
            c.1 += 1;
        } else if self.counters.len() < self.k {
            self.counters.push((item.to_vec(), 1, 0));
        } else {
            let min = (0..self.counters.len()).min_by_key(|&i| self.counters[i].1).unwrap();
            let floor = self.counters[min].1;
            self.counters[min] = (item.to_vec(), floor + 1, floor);
        }
    }

    /// Tracked items as `(item, count upper bound, max error)`, most frequent first.
    pub fn top(&self) -> Vec<(&[u8], u64, u64)> {
        let mut v: Vec<(&[u8], u64, u64)> = self.counters.iter().map(|c| (c.0.as_slice(), c.1, c.2)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        v
    }
}

/// Welford's numerically stable running mean / variance, with min / max and parallel merge.
#[derive(Debug, Clone, Copy, Default)]
pub struct RunningStats {
    n: f64,
    mean: f64,
    m2: f64,
    pub min: f64,
    pub max: f64,
}

impl RunningStats {
    pub fn new() -> Self {
        Self { min: f64::INFINITY, max: f64::NEG_INFINITY, ..Default::default() }
    }

    pub fn push(&mut self, x: f64) {
        self.n += 1.0;
        let d = x - self.mean;
        self.mean += d / self.n;
        self.m2 += d * (x - self.mean);
        self.min = self.min.min(x);
        self.max = self.max.max(x);
    }

    pub fn count(&self) -> u64 {
        self.n as u64
    }

    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Sample variance (`n - 1` denominator); `NaN` with fewer than two values.
    pub fn variance(&self) -> f64 {
        if self.n < 2.0 { f64::NAN } else { self.m2 / (self.n - 1.0) }
    }

    /// Chan et al. parallel combination of two summaries.
    pub fn merge(&mut self, o: &Self) {
        if o.n == 0.0 {
            return;
        }
        let n = self.n + o.n;
        let d = o.mean - self.mean;
        self.m2 += o.m2 + d * d * self.n * o.n / n;
        self.mean += d * o.n / n;
        self.n = n;
        self.min = self.min.min(o.min);
        self.max = self.max.max(o.max);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(i: u64) -> Vec<u8> {
        format!("item-{i}").into_bytes()
    }

    #[test]
    fn bloom_filter_has_no_false_negatives_and_bounded_false_positives() {
        let mut b = BloomFilter::new(2000, 0.01).unwrap();
        (0..2000).for_each(|i| b.insert(&key(i)));
        assert!((0..2000).all(|i| b.contains(&key(i))));
        let fp = (10_000..20_000).filter(|&i| b.contains(&key(i))).count() as f64 / 10_000.0;
        assert!(fp < 0.025, "false positive rate {fp}");
        assert!((b.estimated_fp_rate() - 0.01).abs() < 0.01);
        assert!(BloomFilter::new(0, 0.1).is_err() && BloomFilter::new(10, 1.5).is_err());
    }

    #[test]
    fn count_min_overestimates_within_bound() {
        let mut s = CountMinSketch::new(0.01, 0.01).unwrap();
        let mut total = 0u64;
        for i in 0..500u64 {
            let c = if i < 5 { 1000 } else { 1 + i % 3 };
            s.add(&key(i), c);
            total += c;
        }
        for i in 0..5 {
            let est = s.estimate(&key(i));
            assert!(est >= 1000 && est as f64 <= 1000.0 + 0.01 * total as f64, "{est}");
        }
        assert!(s.estimate(&key(99_999)) as f64 <= 0.01 * total as f64);
        assert!(CountMinSketch::new(0.0, 0.1).is_err());
    }

    #[test]
    fn hyperloglog_estimates_cardinality_and_merges() {
        let mut h = HyperLogLog::new(12).unwrap();
        (0..50_000u64).for_each(|i| h.add(&key(i)));
        let est = h.estimate();
        assert!((est - 50_000.0).abs() / 50_000.0 < 0.04, "{est}");
        (0..50_000u64).for_each(|i| h.add(&key(i))); // duplicates change nothing
        assert!((h.estimate() - est).abs() < 1e-9);
        let (mut a, mut b) = (HyperLogLog::new(12).unwrap(), HyperLogLog::new(12).unwrap());
        (0..30_000u64).for_each(|i| a.add(&key(i)));
        (20_000..50_000u64).for_each(|i| b.add(&key(i)));
        a.merge(&b).unwrap();
        assert!((a.estimate() - 50_000.0).abs() / 50_000.0 < 0.04);
        let mut small = HyperLogLog::new(10).unwrap();
        (0..100u64).for_each(|i| small.add(&key(i)));
        assert!((small.estimate() - 100.0).abs() < 8.0, "{}", small.estimate());
        assert!(a.merge(&small).is_err() && HyperLogLog::new(2).is_err());
    }

    #[test]
    fn reservoir_is_uniform() {
        let mut hits = vec![0u32; 10];
        for seed in 0..2000 {
            let mut r = Reservoir::new(3, seed).unwrap();
            (0..10usize).for_each(|i| r.push(i));
            assert_eq!(r.sample().len(), 3);
            r.sample().iter().for_each(|&i| hits[i] += 1);
        }
        // every element is kept with probability 3/10
        assert!(hits.iter().all(|&h| (500..700).contains(&h)), "{hits:?}");
        assert!(Reservoir::<u8>::new(0, 0).is_err());
    }

    #[test]
    fn streaming_quantiles_and_heavy_hitters() {
        let mut q = StreamingQuantiles::new(100.0).unwrap();
        assert!(q.quantile(0.5).is_none());
        let mut rng = Rng::new(3);
        let mut data: Vec<f64> = (0..100_000).map(|_| rng.next_f64()).collect();
        data.iter().for_each(|&x| q.add(x));
        data.sort_by(f64::total_cmp);
        for p in [0.01, 0.25, 0.5, 0.75, 0.99, 0.999] {
            let est = q.quantile(p).unwrap();
            assert!((est - data[(p * 99_999.0) as usize]).abs() < 0.01, "q{p}: {est}");
        }
        assert!(q.n_centroids() < 1500, "bounded memory: {}", q.n_centroids());
        // Space-Saving finds the hot items of a skewed stream
        let mut ss = SpaceSaving::new(10).unwrap();
        for i in 0..20_000u64 {
            let item = if i % 4 == 0 { 0 } else if i % 5 == 0 { 1 } else { 100 + i };
            ss.add(&key(item));
        }
        let top = ss.top();
        assert_eq!(top[0].0, key(0).as_slice());
        assert_eq!(top[1].0, key(1).as_slice());
        assert!(top[0].1 >= 5000 && top[0].1 - top[0].2 <= 5000);
        assert!(SpaceSaving::new(0).is_err() && StreamingQuantiles::new(1.0).is_err());
    }

    #[test]
    fn running_stats_match_two_pass_and_merge() {
        let data: Vec<f64> = (0..1000).map(|i| 1e9 + (i as f64 * 0.37).sin() * 3.0).collect(); // large offset stresses naive sums
        let mut s = RunningStats::new();
        data.iter().for_each(|&x| s.push(x));
        let mean = data.iter().sum::<f64>() / 1000.0;
        let var = data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 999.0;
        assert!((s.mean() - mean).abs() < 1e-5 && (s.variance() - var).abs() < 1e-6 * var.max(1.0), "{} {} {} {}", s.mean(), mean, s.variance(), var);
        let (mut a, mut b) = (RunningStats::new(), RunningStats::new());
        data[..400].iter().for_each(|&x| a.push(x));
        data[400..].iter().for_each(|&x| b.push(x));
        a.merge(&b);
        assert!((a.mean() - s.mean()).abs() < 1e-5 && (a.variance() - s.variance()).abs() < 1e-6 * s.variance() && a.count() == 1000);
        assert!(a.min <= a.max && RunningStats::new().variance().is_nan());
    }
}
