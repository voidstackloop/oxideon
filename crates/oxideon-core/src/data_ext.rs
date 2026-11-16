//! Dataset composition and sampling in the style of `torch.utils.data`: [`Subset`], [`ConcatDataset`],
//! [`MapDataset`], [`random_split`], the samplers ([`SequentialSampler`], [`RandomSampler`],
//! [`WeightedRandomSampler`], [`SubsetRandomSampler`]) and a [`SamplerLoader`] that batches whatever a sampler
//! yields. All of them build on the `Dataset` trait of `crate::data`.

use crate::data::{Dataset, Transform};
use crate::error::{Error, Result};
use crate::rng;
use crate::tensor::Tensor;

/// A view of a dataset restricted to `indices`.
pub struct Subset<'a, D: Dataset> {
    dataset: &'a D,
    indices: Vec<usize>,
}

impl<'a, D: Dataset> Subset<'a, D> {
    pub fn new(dataset: &'a D, indices: Vec<usize>) -> Result<Self> {
        if let Some(&bad) = indices.iter().find(|&&i| i >= dataset.len()) {
            return Err(Error::Msg(format!("Subset: index {bad} is out of range for a dataset of {}", dataset.len())));
        }
        Ok(Self { dataset, indices })
    }
}

impl<D: Dataset> Dataset for Subset<'_, D> {
    fn len(&self) -> usize {
        self.indices.len()
    }
    fn get(&self, index: usize) -> Result<(Tensor, Tensor)> {
        let i = *self
            .indices
            .get(index)
            .ok_or_else(|| Error::Msg(format!("Subset: index {index} is out of range for {} items", self.indices.len())))?;
        self.dataset.get(i)
    }
    fn get_batch(&self, indices: &[usize]) -> Result<(Tensor, Tensor)> {
        let mapped: Vec<usize> = indices
            .iter()
            .map(|&i| {
                self.indices
                    .get(i)
                    .copied()
                    .ok_or_else(|| Error::Msg(format!("Subset: index {i} is out of range for {} items", self.indices.len())))
            })
            .collect::<Result<_>>()?;
        self.dataset.get_batch(&mapped)
    }
}

/// Datasets laid end to end.
pub struct ConcatDataset<'a> {
    parts: Vec<&'a dyn Dataset>,
}

impl<'a> ConcatDataset<'a> {
    pub fn new(parts: Vec<&'a dyn Dataset>) -> Self {
        Self { parts }
    }
    fn locate(&self, mut index: usize) -> Result<(usize, usize)> {
        for (p, d) in self.parts.iter().enumerate() {
            if index < d.len() {
                return Ok((p, index));
            }
            index -= d.len();
        }
        Err(Error::Msg("ConcatDataset: index out of range".into()))
    }
}

impl Dataset for ConcatDataset<'_> {
    fn len(&self) -> usize {
        self.parts.iter().map(|d| d.len()).sum()
    }
    fn get(&self, index: usize) -> Result<(Tensor, Tensor)> {
        let (p, i) = self.locate(index)?;
        self.parts[p].get(i)
    }
}

/// Applies `transform` to the features of every item.
pub struct MapDataset<'a, D: Dataset, T: Transform> {
    dataset: &'a D,
    transform: T,
}

impl<'a, D: Dataset, T: Transform> MapDataset<'a, D, T> {
    pub fn new(dataset: &'a D, transform: T) -> Self {
        Self { dataset, transform }
    }
}

impl<D: Dataset, T: Transform> Dataset for MapDataset<'_, D, T> {
    fn len(&self) -> usize {
        self.dataset.len()
    }
    fn get(&self, index: usize) -> Result<(Tensor, Tensor)> {
        let (x, y) = self.dataset.get(index)?;
        Ok((self.transform.apply(&x)?, y))
    }
}

/// Splits `0..len` into random disjoint index sets of the given `lengths` (which must sum to `len`).
pub fn random_split(len: usize, lengths: &[usize]) -> Result<Vec<Vec<usize>>> {
    if lengths.iter().sum::<usize>() != len {
        return Err(Error::Msg(format!("random_split: lengths {lengths:?} do not sum to {len}")));
    }
    let mut all: Vec<usize> = (0..len).collect();
    rng::with_rng(|r| r.shuffle(&mut all));
    let mut out = Vec::with_capacity(lengths.len());
    let mut at = 0;
    for &l in lengths {
        out.push(all[at..at + l].to_vec());
        at += l;
    }
    Ok(out)
}

/// Like [`random_split`] with fractions: each part gets `floor(len * fraction)` items and the remainder is spread
/// round-robin from the first part, matching PyTorch.
pub fn random_split_fractions(len: usize, fractions: &[f64]) -> Result<Vec<Vec<usize>>> {
    if (fractions.iter().sum::<f64>() - 1.0).abs() > 1e-9 || fractions.iter().any(|&f| f < 0.0) {
        return Err(Error::Msg("random_split_fractions: fractions must be non-negative and sum to 1".into()));
    }
    let mut lengths: Vec<usize> = fractions.iter().map(|f| (len as f64 * f).floor() as usize).collect();
    let mut rem = len - lengths.iter().sum::<usize>();
    let mut i = 0;
    while rem > 0 && !lengths.is_empty() {
        lengths[i % fractions.len()] += 1;
        rem -= 1;
        i += 1;
    }
    random_split(len, &lengths)
}

/// Produces the order in which a dataset of `len` items is visited in one epoch.
pub trait Sampler {
    fn indices(&self, len: usize) -> Vec<usize>;
}

/// `0, 1, 2, ...`
pub struct SequentialSampler;

impl Sampler for SequentialSampler {
    fn indices(&self, len: usize) -> Vec<usize> {
        (0..len).collect()
    }
}

/// A random permutation, or `num_samples` draws with replacement.
pub struct RandomSampler {
    pub replacement: bool,
    pub num_samples: Option<usize>,
}

impl Sampler for RandomSampler {
    fn indices(&self, len: usize) -> Vec<usize> {
        let n = self.num_samples.unwrap_or(len);
        if len == 0 {
            return Vec::new();
        }
        rng::with_rng(|r| {
            if self.replacement {
                (0..n).map(|_| r.below(len).unwrap_or(0)).collect()
            } else {
                let mut out = Vec::with_capacity(n);
                while out.len() < n {
                    let mut p: Vec<usize> = (0..len).collect();
                    r.shuffle(&mut p);
                    out.extend(p.into_iter().take(n - out.len()));
                }
                out
            }
        })
    }
}

/// Draws indices with probability proportional to `weights`.
pub struct WeightedRandomSampler {
    pub weights: Vec<f64>,
    pub num_samples: usize,
    pub replacement: bool,
}

impl WeightedRandomSampler {
    pub fn new(weights: Vec<f64>, num_samples: usize, replacement: bool) -> Result<Self> {
        if weights.iter().any(|&w| w < 0.0 || !w.is_finite()) || weights.iter().sum::<f64>() <= 0.0 {
            return Err(Error::Msg("WeightedRandomSampler: weights must be finite, non-negative and not all zero".into()));
        }
        if !replacement && num_samples > weights.iter().filter(|&&w| w > 0.0).count() {
            return Err(Error::Msg("WeightedRandomSampler: cannot draw more samples than positive weights without replacement".into()));
        }
        Ok(Self { weights, num_samples, replacement })
    }
}

impl Sampler for WeightedRandomSampler {
    fn indices(&self, _len: usize) -> Vec<usize> {
        rng::with_rng(|r| {
            let mut w = self.weights.clone();
            let mut out = Vec::with_capacity(self.num_samples);
            for _ in 0..self.num_samples {
                let total: f64 = w.iter().sum();
                let mut u = r.next_f64() * total;
                let mut pick = w.len() - 1;
                for (i, &wi) in w.iter().enumerate() {
                    if u < wi {
                        pick = i;
                        break;
                    }
                    u -= wi;
                }
                out.push(pick);
                if !self.replacement {
                    w[pick] = 0.0;
                }
            }
            out
        })
    }
}

/// A random permutation of the fixed `indices`.
pub struct SubsetRandomSampler {
    pub indices: Vec<usize>,
}

impl Sampler for SubsetRandomSampler {
    fn indices(&self, _len: usize) -> Vec<usize> {
        let mut v = self.indices.clone();
        rng::with_rng(|r| r.shuffle(&mut v));
        v
    }
}

/// Mini-batches of a dataset in the order chosen by a [`Sampler`].
pub struct SamplerLoader<'a, D: Dataset, S: Sampler> {
    dataset: &'a D,
    sampler: S,
    batch_size: usize,
    drop_last: bool,
}

impl<'a, D: Dataset, S: Sampler> SamplerLoader<'a, D, S> {
    pub fn new(dataset: &'a D, sampler: S, batch_size: usize) -> Self {
        Self { dataset, sampler, batch_size: batch_size.max(1), drop_last: false }
    }
    pub fn drop_last(mut self, on: bool) -> Self {
        self.drop_last = on;
        self
    }
    /// The batches of one epoch, each fetched with [`Dataset::get_batch`].
    pub fn iter(&self) -> impl Iterator<Item = Result<(Tensor, Tensor)>> + '_ {
        let order = self.sampler.indices(self.dataset.len());
        let bs = self.batch_size;
        let drop_last = self.drop_last;
        let chunks: Vec<Vec<usize>> = order
            .chunks(bs)
            .filter(|c| !drop_last || c.len() == bs)
            .map(<[usize]>::to_vec)
            .collect();
        chunks.into_iter().map(move |idx| self.dataset.get_batch(&idx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::TensorDataset;

    fn ds(n: usize) -> TensorDataset {
        let x = Tensor::from_slice(&(0..n).map(|i| i as f32).collect::<Vec<_>>(), &[n, 1]).unwrap();
        let y = Tensor::from_slice(&(0..n).map(|i| (i * 10) as f32).collect::<Vec<_>>(), &[n]).unwrap();
        TensorDataset::try_new(x, y).unwrap()
    }
    fn ys(b: &(Tensor, Tensor)) -> Vec<f32> {
        b.1.to_vec::<f32>().unwrap()
    }

    #[test]
    fn subset_concat_map() {
        let d = ds(6);
        let sub = Subset::new(&d, vec![4, 1]).unwrap();
        assert_eq!(sub.len(), 2);
        assert_eq!(ys(&sub.get_batch(&[0, 1]).unwrap()), vec![40.0, 10.0]);
        assert!(Subset::new(&d, vec![6]).is_err());
        let (a, b) = (ds(2), ds(3));
        let cat = ConcatDataset::new(vec![&a, &b]);
        assert_eq!(cat.len(), 5);
        assert_eq!(cat.get(2).unwrap().1.to_vec::<f32>().unwrap(), vec![0.0]); // first item of `b`
        assert!(cat.get(5).is_err());
        let mapped = MapDataset::new(&d, crate::data::Normalize::new(1.0, 2.0));
        assert_eq!(mapped.get(3).unwrap().0.to_vec::<f32>().unwrap(), vec![1.0]); // (3 - 1) / 2
    }

    #[test]
    fn splits_partition_the_indices() {
        rng::manual_seed(5);
        let parts = random_split(10, &[6, 3, 1]).unwrap();
        let mut all: Vec<usize> = parts.iter().flatten().copied().collect();
        all.sort_unstable();
        assert_eq!(all, (0..10).collect::<Vec<_>>());
        assert_eq!(parts.iter().map(Vec::len).collect::<Vec<_>>(), vec![6, 3, 1]);
        assert!(random_split(10, &[6, 3]).is_err());
        let fr = random_split_fractions(10, &[0.45, 0.45, 0.1]).unwrap();
        assert_eq!(fr.iter().map(Vec::len).sum::<usize>(), 10);
        assert_eq!(fr[2].len(), 1);
    }

    #[test]
    fn samplers_and_loader() {
        rng::manual_seed(9);
        assert_eq!(SequentialSampler.indices(4), vec![0, 1, 2, 3]);
        let mut p = RandomSampler { replacement: false, num_samples: None }.indices(20);
        p.sort_unstable();
        assert_eq!(p, (0..20).collect::<Vec<_>>());
        assert_eq!(RandomSampler { replacement: true, num_samples: Some(50) }.indices(3).len(), 50);
        // weights concentrate the draws
        let s = WeightedRandomSampler::new(vec![0.0, 0.0, 1.0, 9.0], 2000, true).unwrap();
        let draws = s.indices(4);
        let threes = draws.iter().filter(|&&i| i == 3).count() as f64 / 2000.0;
        assert!(draws.iter().all(|&i| i >= 2) && (threes - 0.9).abs() < 0.05, "{threes}");
        let nr = WeightedRandomSampler::new(vec![1.0, 1.0, 1.0], 3, false).unwrap().indices(3);
        let mut sorted = nr.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2]);
        assert!(WeightedRandomSampler::new(vec![1.0], 2, false).is_err());
        let sub = SubsetRandomSampler { indices: vec![5, 6, 7] }.indices(10);
        assert_eq!(sub.len(), 3);
        // the loader batches in sampler order
        let d = ds(5);
        let loader = SamplerLoader::new(&d, SequentialSampler, 2);
        let batches: Vec<Vec<f32>> = loader.iter().map(|b| ys(&b.unwrap())).collect();
        assert_eq!(batches, vec![vec![0.0, 10.0], vec![20.0, 30.0], vec![40.0]]);
        assert_eq!(SamplerLoader::new(&d, SequentialSampler, 2).drop_last(true).iter().count(), 2);
    }
}
