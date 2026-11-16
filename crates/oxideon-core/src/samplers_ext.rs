//! More samplers: [`DistributedSampler`] (deterministic per-rank shards), [`StratifiedSampler`] (class-proportional
//! interleaving), and [`batch_indices`] (`BatchSampler`).

use crate::data_ext::Sampler;
use crate::error::{Error, Result};
use crate::rng::Rng;

/// Splits a sampler's order into batches of `batch_size`; the last short batch is kept unless `drop_last`.
pub fn batch_indices(order: &[usize], batch_size: usize, drop_last: bool) -> Vec<Vec<usize>> {
    if batch_size == 0 {
        return Vec::new();
    }
    order.chunks(batch_size).filter(|c| !drop_last || c.len() == batch_size).map(<[usize]>::to_vec).collect()
}

/// Shard of a dataset for process `rank` of `world_size`: every rank sees the same shuffled order (derived from
/// `seed + epoch`) and takes every `world_size`-th index. Without `drop_last` the order is padded by wrapping so
/// all ranks get the same number of samples.
pub struct DistributedSampler {
    pub world_size: usize,
    pub rank: usize,
    pub shuffle: bool,
    pub seed: u64,
    pub drop_last: bool,
    epoch: u64,
}

impl DistributedSampler {
    pub fn new(world_size: usize, rank: usize, shuffle: bool, seed: u64, drop_last: bool) -> Result<Self> {
        if world_size == 0 || rank >= world_size {
            return Err(Error::Msg(format!("DistributedSampler: rank {rank} invalid for world size {world_size}")));
        }
        Ok(Self { world_size, rank, shuffle, seed, drop_last, epoch: 0 })
    }

    /// Call at the start of every epoch so the shuffle differs between epochs (but not between ranks).
    pub fn set_epoch(&mut self, epoch: u64) {
        self.epoch = epoch;
    }

    /// Samples each rank draws from a dataset of `len` items.
    pub fn num_samples(&self, len: usize) -> usize {
        if self.drop_last { len / self.world_size } else { len.div_ceil(self.world_size) }
    }
}

impl Sampler for DistributedSampler {
    fn indices(&self, len: usize) -> Vec<usize> {
        let mut order: Vec<usize> = (0..len).collect();
        if self.shuffle {
            Rng::new(self.seed.wrapping_add(self.epoch.wrapping_mul(0x9E37_79B9_7F4A_7C15))).shuffle(&mut order);
        }
        let per = self.num_samples(len);
        let total = per * self.world_size;
        if total > len && !order.is_empty() {
            let pad: Vec<usize> = order.iter().copied().cycle().take(total - len).collect();
            order.extend(pad);
        }
        order.truncate(total);
        order.into_iter().skip(self.rank).step_by(self.world_size).collect()
    }
}

/// Visits every class in proportion to its size, evenly interleaved, shuffling within each class — so every prefix
/// of the order (and hence every batch) mirrors the dataset's class balance.
pub struct StratifiedSampler {
    pub labels: Vec<usize>,
    pub seed: u64,
}

impl Sampler for StratifiedSampler {
    fn indices(&self, _len: usize) -> Vec<usize> {
        let classes = self.labels.iter().max().map_or(0, |m| m + 1);
        let mut rng = Rng::new(self.seed);
        let mut keyed: Vec<(f64, usize)> = Vec::with_capacity(self.labels.len());
        for c in 0..classes {
            let mut members: Vec<usize> = (0..self.labels.len()).filter(|&i| self.labels[i] == c).collect();
            rng.shuffle(&mut members);
            let n = members.len() as f64;
            let offset = rng.next_f64();
            // the k-th member of a class of size n sits at (k + offset) / n of the way through the epoch
            keyed.extend(members.into_iter().enumerate().map(|(k, i)| ((k as f64 + offset) / n, i)));
        }
        keyed.sort_by(|a, b| a.0.total_cmp(&b.0));
        keyed.into_iter().map(|p| p.1).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batching() {
        let o: Vec<usize> = (0..7).collect();
        assert_eq!(batch_indices(&o, 3, false), vec![vec![0, 1, 2], vec![3, 4, 5], vec![6]]);
        assert_eq!(batch_indices(&o, 3, true).len(), 2);
        assert!(batch_indices(&o, 0, false).is_empty());
    }

    #[test]
    fn distributed_shards_partition_the_dataset() {
        let shards: Vec<Vec<usize>> = (0..3).map(|r| DistributedSampler::new(3, r, true, 7, false).unwrap().indices(10)).collect();
        assert!(shards.iter().all(|s| s.len() == 4)); // ceil(10 / 3), padded by wrapping
        let mut all: Vec<usize> = shards.concat();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all, (0..10).collect::<Vec<_>>());
        let dropped: Vec<usize> = (0..3).flat_map(|r| DistributedSampler::new(3, r, true, 7, true).unwrap().indices(10)).collect();
        assert_eq!(dropped.len(), 9);
        let mut s = DistributedSampler::new(3, 0, true, 7, false).unwrap();
        let first = s.indices(10);
        assert_eq!(first, s.indices(10)); // deterministic
        s.set_epoch(1);
        assert_ne!(first, s.indices(10));
        assert!(DistributedSampler::new(2, 2, false, 0, false).is_err());
    }

    #[test]
    fn stratified_prefixes_keep_class_balance() {
        let labels: Vec<usize> = (0..100).map(|i| if i < 80 { 0 } else { 1 }).collect();
        let order = StratifiedSampler { labels: labels.clone(), seed: 1 }.indices(100);
        assert_eq!(order.len(), 100);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..100).collect::<Vec<_>>());
        for prefix in [10, 25, 50, 75] {
            let minority = order[..prefix].iter().filter(|&&i| labels[i] == 1).count() as f64;
            assert!((minority - prefix as f64 * 0.2).abs() <= 1.5, "prefix {prefix}: {minority}");
        }
    }
}
