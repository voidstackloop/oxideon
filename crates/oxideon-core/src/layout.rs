//! Strided tensor [`Layout`] and affine [`ViewSpec`] for zero-copy views.
//!
//! `Layout` tracks `shape: Vec<usize>`, `strides: Vec<isize>`, and `offset: usize`
//! (in elements). Because strides are signed (`isize`) and may be `0` or negative,
//! all of the following are zero-copy view transformations on `Layout`:
//! - `permute`, `transpose`, `t`, `movedim`
//! - `reshape` (whenever the current layout can be viewed with new shape without copy)
//! - `expand` / `broadcast_to` (stride = `0`)
//! - `narrow`, `slice_step` (step > 0 or step < 0), `flip` (negative stride)
//! - `select`, `squeeze`, `unsqueeze`
//! - `diagonal` (stride = `s_d1 + s_d2`)
//! - `as_strided`
//!
//! In addition, [`ViewSpec`] encodes any slicing/indexing/flip/unsqueeze/select chain
//! as an explicit affine map from output indices to input indices so that its
//! exact adjoint (`view_scatter`) can be computed during backpropagation.

use crate::error::{Error, Result};

/// Number of elements of a tensor with this `shape`, or an error if the count overflows `usize`.
/// (An unchecked product wraps silently in release builds and yields a tensor whose shape and storage disagree.)
pub fn checked_numel(shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| Error::Msg(format!("shape {shape:?}: element count overflows usize")))
}

/// `Vec::with_capacity(n)` that reports an allocation failure as an error instead of aborting the process.
pub fn try_with_capacity<T>(n: usize) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| {
        Error::Msg(format!(
            "cannot allocate {n} elements ({} bytes)",
            n.saturating_mul(std::mem::size_of::<T>())
        ))
    })?;
    Ok(v)
}

/// Fallible `vec![value; n]`.
pub fn try_filled<T: Clone>(n: usize, value: T) -> Result<Vec<T>> {
    let mut v = try_with_capacity(n)?;
    v.resize(n, value);
    Ok(v)
}

/// Normalizes a (possibly negative) axis in `[-ndim, ndim)` to `0..ndim`.
pub fn normalize_axis(op: &'static str, axis: isize, ndim: usize) -> Result<usize> {
    let idx = if axis < 0 { ndim as isize + axis } else { axis };
    if idx < 0 || idx as usize >= ndim {
        Err(Error::InvalidAxis { op, axis, ndim })
    } else {
        Ok(idx as usize)
    }
}

/// Normalizes an axis for insertion (e.g. `unsqueeze`, `stack`), valid in `[-ndim-1, ndim]`.
pub fn normalize_insert_axis(op: &'static str, axis: isize, ndim: usize) -> Result<usize> {
    let limit = ndim + 1;
    let idx = if axis < 0 {
        limit as isize + axis
    } else {
        axis
    };
    if idx < 0 || idx as usize >= limit {
        Err(Error::InvalidAxis { op, axis, ndim })
    } else {
        Ok(idx as usize)
    }
}

/// Computes standard row-major (C-contiguous) strides for `shape`.
pub fn contiguous_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![1isize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * (shape[i + 1] as isize).max(1);
    }
    strides
}

/// Computes the NumPy broadcast shape of `a` and `b`.
pub fn broadcast_shapes(op: &'static str, a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let n = a.len().max(b.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let da = if i < n - a.len() {
            1
        } else {
            a[i - (n - a.len())]
        };
        let db = if i < n - b.len() {
            1
        } else {
            b[i - (n - b.len())]
        };
        if da == db {
            out.push(da);
        } else if da == 1 {
            out.push(db);
        } else if db == 1 {
            out.push(da);
        } else {
            return Err(Error::BroadcastIncompatible {
                op,
                lhs: a.to_vec(),
                rhs: b.to_vec(),
            });
        }
    }
    Ok(out)
}

/// Describes how a multi-dimensional tensor maps logical indices to flat storage offsets.
/// Physical memory layout convention for 4D tensors (`NCHW` vs `NHWC`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryFormat {
    Contiguous,
    ChannelsLast,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: usize,
}

impl Layout {
    pub fn contiguous(shape: impl Into<Vec<usize>>) -> Self {
        let shape = shape.into();
        let strides = contiguous_strides(&shape);
        Self {
            shape,
            strides,
            offset: 0,
        }
    }

    /// Builds a layout directly, panicking when `shape`/`strides` ranks differ.
    ///
    /// For layouts derived from user input use [`Layout::try_new`], which
    /// reports the mismatch as `Err` instead of panicking.
    pub fn new(shape: Vec<usize>, strides: Vec<isize>, offset: usize) -> Self {
        assert_eq!(
            shape.len(),
            strides.len(),
            "shape and strides rank mismatch"
        );
        Self {
            shape,
            strides,
            offset,
        }
    }

    /// Fallible `new` for layouts built from user-controlled input
    /// (e.g. [`crate::Tensor::as_strided`]).
    pub fn try_new(
        shape: Vec<usize>,
        strides: Vec<isize>,
        offset: usize,
    ) -> crate::error::Result<Self> {
        if shape.len() != strides.len() {
            return Err(crate::error::Error::Msg(format!(
                "shape and strides rank mismatch: {} vs {}",
                shape.len(),
                strides.len()
            )));
        }
        Ok(Self {
            shape,
            strides,
            offset,
        })
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Element count. Saturates at `usize::MAX` on overflow (debug builds assert)
    /// instead of wrapping to a small count that would disagree with the shape;
    /// use [`checked_numel`] for a fallible count.
    #[inline]
    pub fn numel(&self) -> usize {
        let mut n = 1usize;
        for &d in &self.shape {
            n = match n.checked_mul(d) {
                Some(v) => v,
                None => {
                    debug_assert!(false, "numel overflow for shape {:?}", self.shape);
                    return usize::MAX;
                }
            };
        }
        n
    }

    /// True if elements are stored contiguously in row-major order starting at `self.offset()`.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        if self.numel() <= 1 {
            return true;
        }
        let mut expected = 1isize;
        for (&d, &s) in self.shape.iter().zip(&self.strides).rev() {
            if d == 1 {
                continue;
            }
            if s != expected {
                return false;
            }
            // On overflow the true expected stride exceeds any `isize` stride,
            // so no real `s` can match it: not contiguous.
            expected = match expected.checked_mul(d as isize) {
                Some(v) => v,
                None => return false,
            };
        }
        true
    }

    /// True if a 4D tensor is laid out in `channels_last` (`NHWC`) physical order
    /// `[H*W*C, 1, W*C, C]` while preserving logical `[N, C, H, W]` shape.
    #[inline]
    pub fn is_channels_last(&self) -> bool {
        if self.shape.len() != 4 {
            return false;
        }
        let (n, c, h, w) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        if n == 0 || c == 0 || h == 0 || w == 0 {
            return true;
        }
        let s = &self.strides;
        let ec = 1isize;
        let ew = c as isize;
        let eh = (w * c) as isize;
        let en = (h * w * c) as isize;
        (c == 1 || s[1] == ec)
            && (w == 1 || s[3] == ew)
            && (h == 1 || s[2] == eh)
            && (n == 1 || s[0] == en)
    }

    /// Creates a `channels_last` (`NHWC` physical, `[N, C, H, W]` logical) layout.
    pub fn channels_last(shape: Vec<usize>) -> crate::error::Result<Self> {
        if shape.len() != 4 {
            return Err(crate::error::Error::Msg(format!(
                "channels_last layout requires 4D [N, C, H, W] shape, got {:?}",
                shape
            )));
        }
        let (_, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let strides = vec![(h * w * c) as isize, 1, (w * c) as isize, c as isize];
        Ok(Self {
            shape,
            strides,
            offset: 0,
        })
    }

    /// Returns `Some(start..end)` if the tensor is contiguous so its storage slice
    /// is simply `&buf[start..end]`.
    #[inline]
    pub fn contiguous_range(&self) -> Option<std::ops::Range<usize>> {
        if self.is_contiguous() {
            let n = self.numel();
            let end = self.offset.checked_add(n)?;
            Some(self.offset..end)
        } else {
            None
        }
    }

    /// [`Self::contiguous_range`], validated against a storage length for slicing.
    ///
    /// An empty view can legally carry a nonzero offset (e.g. narrowing axis 1
    /// of a `[0, 4]` tensor), and slicing storage with the resulting
    /// `offset..offset` range panics once `offset > 0`. Returns `None` in that
    /// case so callers fall back to the strided path, which correctly yields
    /// no elements for empty layouts.
    pub fn contiguous_range_in(&self, storage_len: usize) -> Option<std::ops::Range<usize>> {
        let range = self.contiguous_range()?;
        if range.start > storage_len || range.end > storage_len {
            return None;
        }
        Some(range)
    }

    /// Flat storage index for a multi-dimensional logical coordinate.
    #[inline]
    pub fn storage_index(&self, coords: &[usize]) -> usize {
        debug_assert_eq!(coords.len(), self.shape.len());
        let mut idx = self.offset as isize;
        for (&c, &s) in coords.iter().zip(&self.strides) {
            idx += (c as isize) * s;
        }
        debug_assert!(idx >= 0);
        idx as usize
    }

    /// Attempts to compute new strides for viewing `self` with `new_shape` without copying.
    /// Implements PyTorch's `computeStride` algorithm for arbitrary strided views.
    pub fn reshape_strides(&self, new_shape: &[usize]) -> Option<Vec<isize>> {
        let n = self.numel();
        let new_n = checked_numel(new_shape).ok()?;
        if n != new_n {
            return None;
        }
        if n == 0 {
            return Some(contiguous_strides(new_shape));
        }
        if self.is_contiguous() {
            return Some(contiguous_strides(new_shape));
        }

        // Group old dimensions into contiguous chunks (ignoring size-1 dims).
        let mut old_dims: Vec<(usize, isize)> = self
            .shape
            .iter()
            .copied()
            .zip(self.strides.iter().copied())
            .filter(|&(d, _)| d != 1)
            .collect();

        if old_dims.is_empty() {
            return Some(vec![1; new_shape.len()]);
        }

        let mut chunks: Vec<(usize, isize)> = Vec::new();
        let mut cur_size = old_dims[0].0;
        let mut cur_stride = old_dims[0].1;
        for &(d, s) in &old_dims[1..] {
            if cur_stride == (d as isize) * s {
                cur_size *= d;
                cur_stride = s;
            } else {
                chunks.push((cur_size, cur_stride));
                cur_size = d;
                cur_stride = s;
            }
        }
        chunks.push((cur_size, cur_stride));
        let _ = &mut old_dims;

        // Match new_shape dimensions from right to left against chunks.
        let mut new_strides = vec![0isize; new_shape.len()];
        let mut chunk_idx = chunks.len() as isize - 1;
        let mut rem_size = chunks.last().unwrap().0;
        let mut base_stride = chunks.last().unwrap().1;

        for i in (0..new_shape.len()).rev() {
            let d = new_shape[i];
            if d == 1 {
                new_strides[i] = base_stride;
                continue;
            }
            if chunk_idx < 0 || rem_size % d != 0 {
                return None;
            }
            new_strides[i] = base_stride;
            rem_size /= d;
            base_stride *= d as isize;
            if rem_size == 1 {
                chunk_idx -= 1;
                if chunk_idx >= 0 {
                    rem_size = chunks[chunk_idx as usize].0;
                    base_stride = chunks[chunk_idx as usize].1;
                }
            }
        }
        Some(new_strides)
    }

    pub fn permute(&self, axes: &[usize]) -> Result<Self> {
        let nd = self.ndim();
        if axes.len() != nd {
            return Err(Error::Msg(format!(
                "permute: expected {nd} axes, got {axes:?}"
            )));
        }
        let mut seen = vec![false; nd];
        for &a in axes {
            if a >= nd || seen[a] {
                return Err(Error::Msg(format!("permute: invalid permutation {axes:?}")));
            }
            seen[a] = true;
        }
        let shape = axes.iter().map(|&a| self.shape[a]).collect();
        let strides = axes.iter().map(|&a| self.strides[a]).collect();
        Ok(Self {
            shape,
            strides,
            offset: self.offset,
        })
    }

    pub fn transpose(&self, d0: isize, d1: isize) -> Result<Self> {
        let a0 = normalize_axis("transpose", d0, self.ndim())?;
        let a1 = normalize_axis("transpose", d1, self.ndim())?;
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        shape.swap(a0, a1);
        strides.swap(a0, a1);
        Ok(Self {
            shape,
            strides,
            offset: self.offset,
        })
    }

    pub fn broadcast_to(&self, target: &[usize]) -> Result<Self> {
        if self.shape == target {
            return Ok(self.clone());
        }
        if target.len() < self.shape.len() {
            return Err(Error::BroadcastIncompatible {
                op: "broadcast_to",
                lhs: self.shape.clone(),
                rhs: target.to_vec(),
            });
        }
        let pad = target.len() - self.shape.len();
        let mut strides = vec![0isize; target.len()];
        for i in 0..target.len() {
            if i < pad {
                strides[i] = 0;
            } else {
                let src_d = self.shape[i - pad];
                let dst_d = target[i];
                if src_d == dst_d {
                    strides[i] = self.strides[i - pad];
                } else if src_d == 1 {
                    strides[i] = 0;
                } else {
                    return Err(Error::BroadcastIncompatible {
                        op: "broadcast_to",
                        lhs: self.shape.clone(),
                        rhs: target.to_vec(),
                    });
                }
            }
        }
        Ok(Self {
            shape: target.to_vec(),
            strides,
            offset: self.offset,
        })
    }

    /// Zero-copy narrow along `axis`: `[start .. start + len]`.
    pub fn narrow(&self, axis: isize, start: usize, len: usize) -> Result<Self> {
        let ax = normalize_axis("narrow", axis, self.ndim())?;
        let dim = self.shape[ax];
        if len > dim || start > dim - len {
            return Err(Error::IndexOutOfBounds {
                op: "narrow",
                index: start.saturating_add(len) as isize,
                dim_size: dim,
            });
        }
        let mut shape = self.shape.clone();
        shape[ax] = len;
        // Empty views keep the original offset: the computed one is never
        // dereferenced and can be negative for valid inputs (e.g. start == dim
        // with a negative stride), which must not wrap to a huge `usize`.
        let new_offset = if len == 0 {
            self.offset
        } else {
            let off = self.offset as isize + (start as isize) * self.strides[ax];
            if off < 0 {
                return Err(Error::Msg(format!(
                    "narrow: negative storage offset {off} (invalid layout)"
                )));
            }
            off as usize
        };
        Ok(Self {
            shape,
            strides: self.strides.clone(),
            offset: new_offset,
        })
    }

    /// Zero-copy strided slice along `axis` with `start`, `end` (exclusive, in `[0..=dim]`
    /// for positive step or `[-1..dim]` for negative step), and non-zero `step`.
    pub fn slice_axis(&self, axis: isize, start: isize, end: isize, step: isize) -> Result<Self> {
        if step == 0 {
            return Err(Error::Msg("slice_axis: step cannot be 0".into()));
        }
        let ax = normalize_axis("slice_axis", axis, self.ndim())?;
        let dim = isize::try_from(self.shape[ax]).unwrap_or(isize::MAX);
        // Lengths use the division form `(span - 1) / step + 1`: the naive
        // `(span + step - 1) / step` overflows for huge steps on small dims.
        let (s, len) = if step > 0 {
            let s = start.clamp(0, dim);
            let e = end.clamp(s, dim);
            let len = if e <= s {
                0
            } else {
                ((e - s - 1) / step + 1) as usize
            };
            (s, len)
        } else {
            let s = start.clamp(-1, dim - 1);
            let e = end.clamp(-1, s);
            let neg_step = step
                .checked_neg()
                .ok_or_else(|| Error::Msg("slice_axis: step isize::MIN is not supported".into()))?;
            let len = if s <= e {
                0
            } else {
                ((s - e - 1) / neg_step + 1) as usize
            };
            (s, len)
        };
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        shape[ax] = len;
        let new_offset = if len == 0 {
            self.offset
        } else {
            let base = isize::try_from(self.offset)
                .map_err(|_| Error::Msg("slice_axis: storage offset exceeds isize::MAX".into()))?;
            let off =
                base.checked_add(s.checked_mul(self.strides[ax]).ok_or_else(|| {
                    Error::Msg("slice_axis: offset computation overflows".into())
                })?)
                .ok_or_else(|| Error::Msg("slice_axis: offset computation overflows".into()))?;
            if off < 0 {
                return Err(Error::Msg(format!(
                    "slice_axis: negative storage offset {off} (invalid layout)"
                )));
            }
            off as usize
        };
        strides[ax] *= step;
        Ok(Self {
            shape,
            strides,
            offset: new_offset,
        })
    }

    /// Reverses elements along the given axes by negating their strides (zero-copy).
    pub fn flip(&self, axes: &[isize]) -> Result<Self> {
        let mut strides = self.strides.clone();
        let mut offset = self.offset as isize;
        let mut seen = vec![false; self.ndim()];
        for &a in axes {
            let ax = normalize_axis("flip", a, self.ndim())?;
            if seen[ax] {
                return Err(Error::Msg(format!("flip: duplicate axis {a}")));
            }
            seen[ax] = true;
            let d = self.shape[ax];
            if d > 1 {
                offset += (d as isize - 1) * strides[ax];
                strides[ax] = -strides[ax];
            }
        }
        if offset < 0 {
            return Err(Error::Msg(format!(
                "flip: negative storage offset {offset} (invalid layout)"
            )));
        }
        Ok(Self {
            shape: self.shape.clone(),
            strides,
            offset: offset as usize,
        })
    }

    /// Selects a single index along `axis`, removing that dimension (zero-copy).
    pub fn select(&self, axis: isize, index: isize) -> Result<Self> {
        let ax = normalize_axis("select", axis, self.ndim())?;
        let dim = self.shape[ax];
        let idx = if index < 0 {
            (dim as isize).saturating_add(index)
        } else {
            index
        };
        if idx < 0 || idx as usize >= dim {
            return Err(Error::IndexOutOfBounds {
                op: "select",
                index,
                dim_size: dim,
            });
        }
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        let off = self.offset as isize + idx * strides[ax];
        if off < 0 {
            return Err(Error::Msg(format!(
                "select: negative storage offset {off} (invalid layout)"
            )));
        }
        let offset = off as usize;
        shape.remove(ax);
        strides.remove(ax);
        Ok(Self {
            shape,
            strides,
            offset,
        })
    }

    /// Inserts a size-1 dimension at `axis` (zero-copy).
    pub fn unsqueeze(&self, axis: isize) -> Result<Self> {
        let ax = normalize_insert_axis("unsqueeze", axis, self.ndim())?;
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        let s = if ax < strides.len() {
            strides[ax] * (shape[ax] as isize).max(1)
        } else {
            1
        };
        shape.insert(ax, 1);
        strides.insert(ax, s);
        Ok(Self {
            shape,
            strides,
            offset: self.offset,
        })
    }

    /// Removes a size-1 dimension at `axis` (or no-op if `shape[axis] != 1`).
    pub fn squeeze_axis(&self, axis: isize) -> Result<Self> {
        let ax = normalize_axis("squeeze", axis, self.ndim())?;
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        if shape[ax] == 1 {
            shape.remove(ax);
            strides.remove(ax);
        }
        Ok(Self {
            shape,
            strides,
            offset: self.offset,
        })
    }

    /// Removes all size-1 dimensions.
    pub fn squeeze_all(&self) -> Self {
        let mut shape = Vec::new();
        let mut strides = Vec::new();
        for (&d, &s) in self.shape.iter().zip(&self.strides) {
            if d != 1 {
                shape.push(d);
                strides.push(s);
            }
        }
        Self {
            shape,
            strides,
            offset: self.offset,
        }
    }

    /// Extracts a diagonal view across `dim1` and `dim2` with `diag_offset`, appending
    /// the diagonal dimension at the end (just like `torch.diagonal`). Zero-copy!
    pub fn diagonal(&self, diag_offset: isize, dim1: isize, dim2: isize) -> Result<Self> {
        let d1 = normalize_axis("diagonal", dim1, self.ndim())?;
        let d2 = normalize_axis("diagonal", dim2, self.ndim())?;
        if d1 == d2 {
            return Err(Error::Msg(
                "diagonal: dim1 and dim2 must be distinct".into(),
            ));
        }
        let n1 = self.shape[d1] as isize;
        let n2 = self.shape[d2] as isize;
        let (len, start1, start2) = if diag_offset >= 0 {
            let len = (n1).min(n2.saturating_sub(diag_offset)).max(0) as usize;
            (len, 0isize, diag_offset)
        } else {
            let len = (n1.saturating_add(diag_offset)).min(n2).max(0) as usize;
            (len, diag_offset.saturating_neg(), 0isize)
        };
        let mut shape = Vec::with_capacity(self.ndim() - 1);
        let mut strides = Vec::with_capacity(self.ndim() - 1);
        for i in 0..self.ndim() {
            if i != d1 && i != d2 {
                shape.push(self.shape[i]);
                strides.push(self.strides[i]);
            }
        }
        shape.push(len);
        strides.push(self.strides[d1] + self.strides[d2]);
        let offset = if len == 0 {
            self.offset
        } else {
            let off = self.offset as isize + start1 * self.strides[d1] + start2 * self.strides[d2];
            if off < 0 {
                return Err(Error::Msg(format!(
                    "diagonal: negative storage offset {off} (invalid layout)"
                )));
            }
            off as usize
        };
        Ok(Self {
            shape,
            strides,
            offset,
        })
    }

    /// Iterator over flat storage offsets in logical row-major order.
    pub fn storage_offsets(&self) -> StridedOffsets {
        StridedOffsets::new(self)
    }

    /// Verifies every element this layout touches lives inside a storage of
    /// `storage_len` elements. Empty layouts touch nothing and always pass;
    /// otherwise the touched range is `[min ..= max]` with `min`/`max` derived
    /// from the offset plus the negative/positive stride extents (all checked).
    pub fn check_storage_bounds(&self, storage_len: usize, op: &str) -> Result<()> {
        if checked_numel(&self.shape).is_err() {
            return Err(Error::Msg(format!(
                "{op}: shape {:?} overflows the element count",
                self.shape
            )));
        }
        if self.shape.contains(&0) {
            return Ok(());
        }
        let mut min = self.offset as i128;
        let mut max = self.offset as i128;
        for (&dim, &stride) in self.shape.iter().zip(self.strides.iter()) {
            let extent = (dim as i128)
                .checked_sub(1)
                .and_then(|m| m.checked_mul(stride as i128))
                .ok_or_else(|| Error::Msg(format!("{op}: stride extent overflows")))?;
            if stride < 0 {
                min = min
                    .checked_add(extent)
                    .ok_or_else(|| Error::Msg(format!("{op}: offset range overflows")))?;
            } else {
                max = max
                    .checked_add(extent)
                    .ok_or_else(|| Error::Msg(format!("{op}: offset range overflows")))?;
            }
        }
        if min < 0 || max >= storage_len as i128 {
            return Err(Error::Msg(format!(
                "{op}: layout touches storage [{min}..={max}] outside [0..{storage_len}]"
            )));
        }
        Ok(())
    }
}

/// Efficient odometer iterator yielding flat storage indices for any `Layout`.
pub struct StridedOffsets {
    shape: Vec<usize>,
    strides: Vec<isize>,
    coord: Vec<usize>,
    cur_offset: isize,
    remaining: usize,
}

impl StridedOffsets {
    pub fn new(layout: &Layout) -> Self {
        let n = layout.numel();
        Self {
            shape: layout.shape.clone(),
            strides: layout.strides.clone(),
            coord: vec![0; layout.ndim()],
            cur_offset: layout.offset as isize,
            remaining: n,
        }
    }
}

impl Iterator for StridedOffsets {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let out = self.cur_offset as usize;
        for d in (0..self.shape.len()).rev() {
            self.coord[d] += 1;
            self.cur_offset += self.strides[d];
            if self.coord[d] < self.shape[d] {
                break;
            }
            self.cur_offset -= self.strides[d] * (self.shape[d] as isize);
            self.coord[d] = 0;
        }
        Some(out)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for StridedOffsets {}

// ---------------------------------------------------------------------------
// ViewSpec: Invertible affine index mapping for differentiable slicing/views
// ---------------------------------------------------------------------------

/// How one dimension of a source tensor is mapped by a [`ViewSpec`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DimMap {
    /// Source dimension is indexed by `start + step * out_coord[out_dim]`.
    Sliced {
        out_dim: usize,
        start: isize,
        step: isize,
    },
    /// Source dimension is fixed to a single index (`select` / integer index).
    Selected { index: usize },
}

/// An affine view transformation from a tensor of `src_shape` to a tensor of `dst_shape`.
///
/// This unifies `narrow`, `slice`, `flip`, `select`, `unsqueeze`, `squeeze`, and multi-axis
/// indexing `t.i((..))` into a single invertible descriptor:
/// - Forward (`view`): zero-copy layout adjustment on the source tensor.
/// - Backward (`view_scatter`): scatters upstream gradient into a zero tensor of `src_shape`,
///   and the backward of `view_scatter` is simply `view` again!
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewSpec {
    pub src_shape: Vec<usize>,
    pub dst_shape: Vec<usize>,
    pub dim_maps: Vec<DimMap>,
}

impl ViewSpec {
    /// Identity view spec for `shape`.
    pub fn identity(shape: &[usize]) -> Self {
        let dim_maps = (0..shape.len())
            .map(|i| DimMap::Sliced {
                out_dim: i,
                start: 0,
                step: 1,
            })
            .collect();
        Self {
            src_shape: shape.to_vec(),
            dst_shape: shape.to_vec(),
            dim_maps,
        }
    }

    /// Applies this `ViewSpec` to a `Layout` (zero-copy!).
    pub fn apply_layout(&self, src: &Layout) -> Result<Layout> {
        if src.shape() != self.src_shape.as_slice() {
            return Err(Error::ShapeMismatch {
                op: "ViewSpec::apply_layout",
                expected: self.src_shape.clone(),
                got: src.shape().to_vec(),
            });
        }
        // `ViewSpec` fields are pub, so hand-built specs must be validated
        // instead of panicking on out-of-range indices below.
        if self.dim_maps.len() != self.src_shape.len() {
            return Err(Error::Msg(format!(
                "ViewSpec::apply_layout: {} dim maps for {} source dims",
                self.dim_maps.len(),
                self.src_shape.len()
            )));
        }
        let mut offset = src.offset() as isize;
        let mut dst_strides = vec![0isize; self.dst_shape.len()];
        for (src_ax, map) in self.dim_maps.iter().enumerate() {
            let s = src.strides()[src_ax];
            match *map {
                DimMap::Sliced {
                    out_dim,
                    start,
                    step,
                } => {
                    if out_dim >= self.dst_shape.len() {
                        return Err(Error::Msg(format!(
                            "ViewSpec::apply_layout: out_dim {out_dim} out of range for {} dst dims",
                            self.dst_shape.len()
                        )));
                    }
                    if self.dst_shape[out_dim] > 0 {
                        offset = offset
                            .checked_add(start.checked_mul(s).ok_or_else(|| {
                                Error::Msg("ViewSpec::apply_layout: offset overflows".into())
                            })?)
                            .ok_or_else(|| {
                                Error::Msg("ViewSpec::apply_layout: offset overflows".into())
                            })?;
                    }
                    dst_strides[out_dim] = dst_strides[out_dim]
                        .checked_add(step.checked_mul(s).ok_or_else(|| {
                            Error::Msg("ViewSpec::apply_layout: stride overflows".into())
                        })?)
                        .ok_or_else(|| {
                            Error::Msg("ViewSpec::apply_layout: stride overflows".into())
                        })?;
                }
                DimMap::Selected { index } => {
                    if index >= self.src_shape[src_ax] {
                        return Err(Error::IndexOutOfBounds {
                            op: "ViewSpec::apply_layout",
                            index: index as isize,
                            dim_size: self.src_shape[src_ax],
                        });
                    }
                    offset = offset
                        .checked_add((index as isize).checked_mul(s).ok_or_else(|| {
                            Error::Msg("ViewSpec::apply_layout: offset overflows".into())
                        })?)
                        .ok_or_else(|| {
                            Error::Msg("ViewSpec::apply_layout: offset overflows".into())
                        })?;
                }
            }
        }
        if offset < 0 {
            return Err(Error::Msg(format!(
                "ViewSpec::apply_layout: negative storage offset {offset}"
            )));
        }
        Ok(Layout::new(
            self.dst_shape.clone(),
            dst_strides,
            offset as usize,
        ))
    }
}
