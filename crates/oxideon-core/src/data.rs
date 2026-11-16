//! Complete data pipeline: [`Dataset`], [`TensorDataset`], [`DataLoader`], composable
//! [`Transform`]s, `train_test_split`, and synthetic dataset generators.

use crate::error::{Error, Result};
use crate::rng;
use crate::tensor::Tensor;

/// Splits `0..n` into mini-batches of indices (the last batch may be smaller).
///
/// A `batch_size` of 0 is clamped to 1 (same convention as
/// [`DataLoader::new`]) instead of panicking.
pub fn batch_indices(n: usize, batch_size: usize, shuffle: bool) -> Vec<Vec<usize>> {
    let batch_size = batch_size.max(1);
    let mut idx: Vec<usize> = (0..n).collect();
    if shuffle {
        rng::with_rng(|r| r.shuffle(&mut idx));
    }
    idx.chunks(batch_size).map(<[usize]>::to_vec).collect()
}

/// A dataset yielding `(feature, target)` tensor pairs.
pub trait Dataset {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn get(&self, index: usize) -> Result<(Tensor, Tensor)>;

    /// Efficient batch fetch along axis 0 (defaults to stacking `get(i)` calls).
    fn get_batch(&self, indices: &[usize]) -> Result<(Tensor, Tensor)> {
        let mut xs = Vec::with_capacity(indices.len());
        let mut ys = Vec::with_capacity(indices.len());
        for &i in indices {
            let (x, y) = self.get(i)?;
            xs.push(x);
            ys.push(y);
        }
        Ok((Tensor::stack(&xs, 0)?, Tensor::stack(&ys, 0)?))
    }
}

/// An in-memory dataset wrapping aligned `features` and `targets` tensors (matching dim 0).
#[derive(Clone)]
pub struct TensorDataset {
    pub features: Tensor,
    pub targets: Tensor,
}

impl TensorDataset {
    /// Fallible constructor: rejects rank-0 inputs (which have no axis 0) and
    /// features/targets whose lengths along axis 0 disagree.
    pub fn try_new(features: Tensor, targets: Tensor) -> Result<Self> {
        if features.ndim() < 1 || targets.ndim() < 1 {
            return Err(crate::error::Error::Msg(format!(
                "TensorDataset: features and targets must have rank >= 1, got {:?} and {:?}",
                features.shape(),
                targets.shape()
            )));
        }
        if features.shape()[0] != targets.shape()[0] {
            return Err(crate::error::Error::Msg(format!(
                "TensorDataset: features and targets must have the same length along axis 0, got {} vs {}",
                features.shape()[0],
                targets.shape()[0]
            )));
        }
        Ok(Self { features, targets })
    }

    /// Infallible constructor (panics with a clear message on invalid input).
    /// Prefer [`TensorDataset::try_new`] when inputs are user-controlled.
    pub fn new(features: Tensor, targets: Tensor) -> Self {
        Self::try_new(features, targets).expect("TensorDataset::new: invalid features/targets")
    }
}

impl Dataset for TensorDataset {
    fn len(&self) -> usize {
        self.features.shape()[0]
    }

    fn get(&self, index: usize) -> Result<(Tensor, Tensor)> {
        // `as isize` would wrap huge indices to negatives (silent last-row
        // select); reject them instead, mirroring `sat_isize` in indexing.
        let idx = isize::try_from(index).map_err(|_| {
            crate::error::Error::Msg(format!(
                "TensorDataset::get: index {index} exceeds isize::MAX"
            ))
        })?;
        Ok((self.features.select(0, idx)?, self.targets.select(0, idx)?))
    }

    fn get_batch(&self, indices: &[usize]) -> Result<(Tensor, Tensor)> {
        let mut idx_i64: Vec<i64> = Vec::with_capacity(indices.len());
        for &i in indices {
            idx_i64.push(i64::try_from(i).map_err(|_| {
                crate::error::Error::Msg(format!(
                    "TensorDataset::get_batch: index {i} exceeds i64::MAX"
                ))
            })?);
        }
        let idx_t = Tensor::from_slice_on(&idx_i64, &[indices.len()], &self.features.device())?;
        Ok((
            self.features.index_select(0, &idx_t)?,
            self.targets.index_select(0, &idx_t)?,
        ))
    }
}

/// Mini-batch iterator over any [`Dataset`].
pub struct DataLoader<'a, D: Dataset> {
    dataset: &'a D,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
}

impl<'a, D: Dataset> DataLoader<'a, D> {
    pub fn new(dataset: &'a D, batch_size: usize, shuffle: bool) -> Self {
        Self {
            dataset,
            batch_size: batch_size.max(1),
            shuffle,
            drop_last: false,
        }
    }

    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    pub fn num_batches(&self) -> usize {
        if self.drop_last {
            self.dataset.len() / self.batch_size
        } else {
            self.dataset.len().div_ceil(self.batch_size)
        }
    }

    /// Returns an iterator yielding `Result<(Tensor, Tensor)>` mini-batches for one epoch.
    pub fn iter(&self) -> DataLoaderIter<'a, D> {
        let mut batches = batch_indices(self.dataset.len(), self.batch_size, self.shuffle);
        if self.drop_last {
            batches.retain(|b| b.len() == self.batch_size);
        }
        DataLoaderIter {
            dataset: self.dataset,
            batches,
            pos: 0,
        }
    }
}

pub struct DataLoaderIter<'a, D: Dataset> {
    dataset: &'a D,
    batches: Vec<Vec<usize>>,
    pos: usize,
}

impl<'a, D: Dataset> Iterator for DataLoaderIter<'a, D> {
    type Item = Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.batches.len() {
            return None;
        }
        let idx = &self.batches[self.pos];
        self.pos += 1;
        Some(self.dataset.get_batch(idx))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.batches.len() - self.pos;
        (rem, Some(rem))
    }
}

impl<'a, D: Dataset> ExactSizeIterator for DataLoaderIter<'a, D> {}

// ---------------------------------------------------------------------------
// Data Transforms (`Normalize`, `AddGaussianNoise`, `Compose`)
// ---------------------------------------------------------------------------
pub trait Transform {
    fn apply(&self, x: &Tensor) -> Result<Tensor>;
}

pub struct Normalize {
    pub mean: f64,
    pub std: f64,
}

impl Normalize {
    pub fn new(mean: f64, std: f64) -> Self {
        Self { mean, std }
    }
}

impl Transform for Normalize {
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        if !self.std.is_finite() || self.std == 0.0 {
            return Err(crate::error::Error::Msg(format!(
                "Normalize: std must be finite and non-zero, got {}",
                self.std
            )));
        }
        x.sub_scalar(self.mean)?.div_scalar(self.std)
    }
}

pub struct AddGaussianNoise {
    pub std: f64,
}

impl Transform for AddGaussianNoise {
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let noise = Tensor::randn_with(x.shape(), x.dtype(), &x.device())?.mul_scalar(self.std)?;
        x.add(&noise)
    }
}

#[derive(Default)]
pub struct Compose {
    transforms: Vec<Box<dyn Transform>>,
}

impl Compose {
    pub fn new() -> Self {
        Self::default()
    }

    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, t: impl Transform + 'static) -> Self {
        self.transforms.push(Box::new(t));
        self
    }
}

impl Transform for Compose {
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let mut cur = x.clone();
        for t in &self.transforms {
            cur = t.apply(&cur)?;
        }
        Ok(cur)
    }
}

// ---------------------------------------------------------------------------
// Dataset Splitting & Synthetic Generators (`make_moons`, `make_circles`, `make_blobs`)
// ---------------------------------------------------------------------------

/// Splits `(features, targets)` into `(train_x, train_y, test_x, test_y)` according to `test_ratio`.
pub fn train_test_split(
    features: &Tensor,
    targets: &Tensor,
    test_ratio: f64,
    shuffle: bool,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    if features.ndim() < 1 || targets.ndim() < 1 {
        return Err(crate::error::Error::Msg(format!(
            "train_test_split: features and targets must have rank >= 1, got {:?} and {:?}",
            features.shape(),
            targets.shape()
        )));
    }
    let n = features.shape()[0];
    if targets.shape()[0] != n {
        return Err(crate::error::Error::Msg(format!(
            "train_test_split: features/targets length mismatch ({} vs {})",
            n,
            targets.shape()[0]
        )));
    }
    if !test_ratio.is_finite() {
        return Err(crate::error::Error::Msg(format!(
            "train_test_split: test_ratio must be finite, got {test_ratio}"
        )));
    }
    // `n as i64` wraps past i64::MAX (stride-0 views can hold such dims);
    // and collecting `n` indices unbounded aborts on giant `n` (8 TB at
    // 2^40 rows). Reject the wrap loudly and reserve fallibly instead.
    let n_i64 = i64::try_from(n)
        .map_err(|_| Error::Msg(format!("train_test_split: n_samples {n} exceeds i64::MAX")))?;
    // Cap first: try_reserve alone can succeed under memory overcommit and
    // the extend below would then OOM-touch terabytes.
    crate::limits::check_elements("train_test_split indices", &[n])?;
    let mut idx = Vec::new();
    idx.try_reserve(n).map_err(|_| {
        Error::Msg(format!(
            "train_test_split: cannot reserve index buffer for {n} samples"
        ))
    })?;
    idx.extend(0..n_i64);
    if shuffle {
        rng::with_rng(|r| r.shuffle(&mut idx));
    }
    let test_len = ((n as f64) * test_ratio.clamp(0.0, 1.0)).round() as usize;
    let train_len = n - test_len;

    let train_idx = Tensor::from_slice_on(&idx[..train_len], &[train_len], &features.device())?;
    let test_idx = Tensor::from_slice_on(&idx[train_len..], &[test_len], &features.device())?;

    Ok((
        features.index_select(0, &train_idx)?,
        targets.index_select(0, &train_idx)?,
        features.index_select(0, &test_idx)?,
        targets.index_select(0, &test_idx)?,
    ))
}

/// Generates the two-interleaving-half-moons 2D classification dataset `(X: [n, 2], y: [n])`.
pub fn make_moons(n_samples: usize, noise: f32) -> Result<(Tensor, Tensor)> {
    // Refuse overflowing/giant buffers up front (alloc + fill loops below).
    crate::limits::check_elements("make_moons", &[n_samples, 2])?;
    let n0 = n_samples / 2;
    let n1 = n_samples - n0;
    let mut xs = Vec::with_capacity(n_samples * 2);
    let mut ys = Vec::with_capacity(n_samples);

    rng::with_rng(|r| {
        for i in 0..n0 {
            let theta = std::f32::consts::PI * (i as f32) / (n0.max(1) as f32);
            xs.push(theta.cos() + r.normal() * noise);
            xs.push(theta.sin() + r.normal() * noise);
            ys.push(0i64);
        }
        for i in 0..n1 {
            let theta = std::f32::consts::PI * (i as f32) / (n1.max(1) as f32);
            xs.push(1.0 - theta.cos() + r.normal() * noise);
            xs.push(0.5 - theta.sin() + r.normal() * noise);
            ys.push(1i64);
        }
    });
    Ok((
        Tensor::from_vec(xs, &[n_samples, 2])?,
        Tensor::from_vec(ys, &[n_samples])?,
    ))
}

/// Generates concentric circles `(X: [n, 2], y: [n])` with inner/outer scale `factor`.
pub fn make_circles(n_samples: usize, factor: f32, noise: f32) -> Result<(Tensor, Tensor)> {
    // Refuse overflowing/giant buffers up front (alloc + fill loops below).
    crate::limits::check_elements("make_circles", &[n_samples, 2])?;
    let n0 = n_samples / 2;
    let n1 = n_samples - n0;
    let mut xs = Vec::with_capacity(n_samples * 2);
    let mut ys = Vec::with_capacity(n_samples);

    rng::with_rng(|r| {
        for i in 0..n0 {
            let theta = std::f32::consts::TAU * (i as f32) / (n0.max(1) as f32);
            xs.push(theta.cos() + r.normal() * noise);
            xs.push(theta.sin() + r.normal() * noise);
            ys.push(0i64);
        }
        for i in 0..n1 {
            let theta = std::f32::consts::TAU * (i as f32) / (n1.max(1) as f32);
            xs.push(factor * theta.cos() + r.normal() * noise);
            xs.push(factor * theta.sin() + r.normal() * noise);
            ys.push(1i64);
        }
    });
    Ok((
        Tensor::from_vec(xs, &[n_samples, 2])?,
        Tensor::from_vec(ys, &[n_samples])?,
    ))
}

/// Generates isotropic Gaussian blobs `(X: [n_samples, n_features], y: [n_samples])`.
pub fn make_blobs(
    n_samples: usize,
    n_features: usize,
    centers: usize,
    cluster_std: f32,
) -> Result<(Tensor, Tensor)> {
    if centers == 0 {
        return Err(crate::error::Error::Msg(
            "make_blobs: centers must be positive".into(),
        ));
    }
    if n_features == 0 {
        return Err(crate::error::Error::Msg(
            "make_blobs: n_features must be positive".into(),
        ));
    }
    // checked_mul below stops overflow but not giant-but-fitting allocs.
    crate::limits::check_elements("make_blobs", &[n_samples, n_features])?;
    let n_coords = centers.checked_mul(n_features).ok_or_else(|| {
        crate::error::Error::Msg("make_blobs: centers * n_features overflows".into())
    })?;
    let n_xs = n_samples.checked_mul(n_features).ok_or_else(|| {
        crate::error::Error::Msg("make_blobs: n_samples * n_features overflows".into())
    })?;
    let mut center_coords = vec![0.0f32; n_coords];
    let mut xs = Vec::with_capacity(n_xs);
    let mut ys = Vec::with_capacity(n_samples);

    rng::with_rng(|r| {
        for c in &mut center_coords {
            *c = r.uniform(-5.0, 5.0);
        }
        for i in 0..n_samples {
            let c = i % centers;
            for f in 0..n_features {
                let mu = center_coords[c * n_features + f];
                xs.push(mu + r.normal() * cluster_std);
            }
            ys.push(c as i64);
        }
    });
    Ok((
        Tensor::from_vec(xs, &[n_samples, n_features])?,
        Tensor::from_vec(ys, &[n_samples])?,
    ))
}

/// Splits `(features, targets)` along axis 0 into `k` cross-validation folds,
/// returning `k` tuples of `(train_features, train_targets, val_features, val_targets)`.
pub fn k_fold_split(
    features: &Tensor,
    targets: &Tensor,
    k: usize,
    shuffle: bool,
) -> Result<Vec<(Tensor, Tensor, Tensor, Tensor)>> {
    if features.ndim() < 1 || targets.ndim() < 1 {
        return Err(crate::error::Error::Msg(format!(
            "k_fold_split: features and targets must have rank >= 1, got {:?} and {:?}",
            features.shape(),
            targets.shape()
        )));
    }
    let n = features.shape()[0];
    if targets.shape()[0] != n {
        return Err(crate::error::Error::Msg(format!(
            "k_fold_split: features/targets length mismatch ({} vs {})",
            n,
            targets.shape()[0]
        )));
    }
    if k < 2 || k > n {
        return Err(crate::error::Error::Msg(format!(
            "k_fold_split requires 2 <= k <= n_samples ({n}), got k = {k}"
        )));
    }
    // Same wrap + unbounded-reserve hazards as train_test_split above.
    let n_i64 = i64::try_from(n)
        .map_err(|_| Error::Msg(format!("k_fold_split: n_samples {n} exceeds i64::MAX")))?;
    crate::limits::check_elements("k_fold_split indices", &[n])?;
    let mut idx = Vec::new();
    idx.try_reserve(n).map_err(|_| {
        Error::Msg(format!(
            "k_fold_split: cannot reserve index buffer for {n} samples"
        ))
    })?;
    idx.extend(0..n_i64);
    if shuffle {
        rng::with_rng(|r| {
            for i in (1..n).rev() {
                let j = (r.next_u64() as usize) % (i + 1);
                idx.swap(i, j);
            }
        });
    }

    let base_fold_size = n / k;
    let remainder = n % k;
    let mut folds = Vec::with_capacity(k);
    let mut val_start = 0usize;

    for fold in 0..k {
        let val_len = base_fold_size + usize::from(fold < remainder);
        let val_end = val_start + val_len;

        let val_indices = &idx[val_start..val_end];
        let mut train_indices = Vec::with_capacity(n - val_len);
        train_indices.extend_from_slice(&idx[..val_start]);
        train_indices.extend_from_slice(&idx[val_end..]);

        let train_idx_t =
            Tensor::from_slice_on(&train_indices, &[train_indices.len()], &features.device())?;
        let val_idx_t =
            Tensor::from_slice_on(val_indices, &[val_indices.len()], &features.device())?;

        folds.push((
            features.index_select(0, &train_idx_t)?,
            targets.index_select(0, &train_idx_t)?,
            features.index_select(0, &val_idx_t)?,
            targets.index_select(0, &val_idx_t)?,
        ));
        val_start = val_end;
    }

    Ok(folds)
}
