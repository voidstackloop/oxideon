//! Ergonomic multi-dimensional slicing and indexing (`t.i((.., 1..4, -1))` and `s![..]`).
//!
//! All indexing via [`Tensor::i`] builds a [`ViewSpec`] and returns a **zero-copy**
//! strided view whose backward pass (`view_scatter`) is also differentiable!

use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

use crate::error::{Error, Result};
use crate::layout::{DimMap, ViewSpec};
use crate::tensor::Tensor;

/// A single-axis indexing descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceElem {
    /// Selects a single index `i` (supports negative indexing), removing the axis.
    Index(isize),
    /// Slices `[start .. end]` with `step` (supports negative indices and negative steps).
    Slice {
        start: Option<isize>,
        end: Option<isize>,
        step: isize,
    },
    /// Inserts a new axis of length 1 (`None` / `NewAxis`).
    NewAxis,
}

impl SliceElem {
    pub const FULL: Self = Self::Slice {
        start: None,
        end: None,
        step: 1,
    };

    pub const fn step(start: Option<isize>, end: Option<isize>, step: isize) -> Self {
        Self::Slice { start, end, step }
    }
}

impl From<isize> for SliceElem {
    fn from(i: isize) -> Self {
        Self::Index(i)
    }
}

/// Saturating `usize -> isize`: values past `isize::MAX` become `isize::MAX`
/// (rejected downstream as out-of-bounds) instead of wrapping to negatives.
fn sat_isize(i: usize) -> isize {
    isize::try_from(i).unwrap_or(isize::MAX)
}

impl From<usize> for SliceElem {
    fn from(i: usize) -> Self {
        Self::Index(sat_isize(i))
    }
}

impl From<i32> for SliceElem {
    fn from(i: i32) -> Self {
        Self::Index(i as isize)
    }
}

impl From<RangeFull> for SliceElem {
    fn from(_: RangeFull) -> Self {
        Self::FULL
    }
}

impl From<Range<usize>> for SliceElem {
    fn from(r: Range<usize>) -> Self {
        Self::Slice {
            start: Some(sat_isize(r.start)),
            end: Some(sat_isize(r.end)),
            step: 1,
        }
    }
}

impl From<Range<isize>> for SliceElem {
    fn from(r: Range<isize>) -> Self {
        Self::Slice {
            start: Some(r.start),
            end: Some(r.end),
            step: 1,
        }
    }
}

impl From<Range<i32>> for SliceElem {
    fn from(r: Range<i32>) -> Self {
        Self::Slice {
            start: Some(r.start as isize),
            end: Some(r.end as isize),
            step: 1,
        }
    }
}

impl From<RangeFrom<usize>> for SliceElem {
    fn from(r: RangeFrom<usize>) -> Self {
        Self::Slice {
            start: Some(sat_isize(r.start)),
            end: None,
            step: 1,
        }
    }
}

impl From<RangeFrom<isize>> for SliceElem {
    fn from(r: RangeFrom<isize>) -> Self {
        Self::Slice {
            start: Some(r.start),
            end: None,
            step: 1,
        }
    }
}

impl From<RangeTo<usize>> for SliceElem {
    fn from(r: RangeTo<usize>) -> Self {
        Self::Slice {
            start: None,
            end: Some(sat_isize(r.end)),
            step: 1,
        }
    }
}

impl From<RangeTo<isize>> for SliceElem {
    fn from(r: RangeTo<isize>) -> Self {
        Self::Slice {
            start: None,
            end: Some(r.end),
            step: 1,
        }
    }
}

impl From<RangeInclusive<usize>> for SliceElem {
    fn from(r: RangeInclusive<usize>) -> Self {
        Self::Slice {
            start: Some(sat_isize(*r.start())),
            end: Some(sat_isize(*r.end()).saturating_add(1)),
            step: 1,
        }
    }
}

impl From<RangeToInclusive<usize>> for SliceElem {
    fn from(r: RangeToInclusive<usize>) -> Self {
        Self::Slice {
            start: None,
            end: Some(sat_isize(r.end).saturating_add(1)),
            step: 1,
        }
    }
}

/// Converts a single index or a tuple of range/index arguments into a `Vec<SliceElem>`.
pub trait IndexArg {
    fn into_slice_elems(self) -> Vec<SliceElem>;
}

impl<A: Into<SliceElem>> IndexArg for A {
    fn into_slice_elems(self) -> Vec<SliceElem> {
        vec![self.into()]
    }
}

impl IndexArg for &[SliceElem] {
    fn into_slice_elems(self) -> Vec<SliceElem> {
        self.to_vec()
    }
}

impl IndexArg for Vec<SliceElem> {
    fn into_slice_elems(self) -> Vec<SliceElem> {
        self
    }
}

macro_rules! impl_tuple_index {
    ($($T:ident : $idx:tt),+) => {
        impl<$($T: Into<SliceElem>),+> IndexArg for ($($T,)+) {
            fn into_slice_elems(self) -> Vec<SliceElem> {
                vec![$(self.$idx.into()),+]
            }
        }
    };
}

impl_tuple_index!(A: 0);
impl_tuple_index!(A: 0, B: 1);
impl_tuple_index!(A: 0, B: 1, C: 2);
impl_tuple_index!(A: 0, B: 1, C: 2, D: 3);
impl_tuple_index!(A: 0, B: 1, C: 2, D: 3, E: 4);
impl_tuple_index!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5);

/// Compiles a slice of [`SliceElem`]s against `src_shape` into a [`ViewSpec`].
pub fn build_view_spec(src_shape: &[usize], elems: &[SliceElem]) -> Result<ViewSpec> {
    let mut src_ax = 0usize;
    let mut dst_shape = Vec::new();
    let mut dim_maps = Vec::with_capacity(src_shape.len());

    for &elem in elems {
        match elem {
            SliceElem::NewAxis => {
                dst_shape.push(1);
            }
            SliceElem::Index(idx) => {
                if src_ax >= src_shape.len() {
                    return Err(Error::Msg("too many indices for tensor".into()));
                }
                let dim = src_shape[src_ax];
                // `i128`: `dim as isize + idx` overflows for dims past
                // `isize::MAX` (stride-0 views can legally hold them).
                let norm = if idx < 0 {
                    dim as i128 + idx as i128
                } else {
                    idx as i128
                };
                if norm < 0 || norm >= dim as i128 {
                    return Err(Error::IndexOutOfBounds {
                        op: "index",
                        index: idx,
                        dim_size: dim,
                    });
                }
                dim_maps.push(DimMap::Selected {
                    index: norm as usize,
                });
                src_ax += 1;
            }
            SliceElem::Slice { start, end, step } => {
                if src_ax >= src_shape.len() {
                    return Err(Error::Msg("too many slice dimensions for tensor".into()));
                }
                if step == 0 {
                    return Err(Error::Msg("slice step cannot be zero".into()));
                }
                // The stride/offset algebra below is `isize`-based: a dim past
                // `isize::MAX` wraps negative and either panics in `clamp`
                // (min > max, all profiles) or yields a wrong empty view.
                if src_shape[src_ax] > isize::MAX as usize {
                    return Err(Error::Msg(format!(
                        "slice: dimension {} exceeds the addressable isize range",
                        src_shape[src_ax]
                    )));
                }
                let dim = src_shape[src_ax] as isize;
                let (s, len) = if step > 0 {
                    let s = match start {
                        Some(v) => (if v < 0 { dim.saturating_add(v) } else { v }).clamp(0, dim),
                        None => 0,
                    };
                    let e = match end {
                        Some(v) => (if v < 0 { dim.saturating_add(v) } else { v }).clamp(s, dim),
                        None => dim,
                    };
                    let len = (e.saturating_sub(s).saturating_add(step.saturating_sub(1)) / step)
                        .max(0) as usize;
                    (s, len)
                } else {
                    let s = match start {
                        Some(v) => {
                            (if v < 0 { dim.saturating_add(v) } else { v }).clamp(-1, dim - 1)
                        }
                        None => dim - 1,
                    };
                    let e = match end {
                        Some(v) => (if v < 0 { dim.saturating_add(v) } else { v }).clamp(-1, s),
                        None => -1,
                    };
                    let neg_step = step
                        .checked_neg()
                        .ok_or_else(|| Error::Msg("slice step isize::MIN overflows".into()))?;
                    let len = (s
                        .saturating_sub(e)
                        .saturating_add(neg_step.saturating_sub(1))
                        / neg_step)
                        .max(0) as usize;
                    (s, len)
                };
                let out_dim = dst_shape.len();
                dst_shape.push(len);
                dim_maps.push(DimMap::Sliced {
                    out_dim,
                    start: s,
                    step,
                });
                src_ax += 1;
            }
        }
    }

    // Any remaining unmentioned trailing dimensions are sliced in full (`..`).
    while src_ax < src_shape.len() {
        let out_dim = dst_shape.len();
        dst_shape.push(src_shape[src_ax]);
        dim_maps.push(DimMap::Sliced {
            out_dim,
            start: 0,
            step: 1,
        });
        src_ax += 1;
    }

    Ok(ViewSpec {
        src_shape: src_shape.to_vec(),
        dst_shape,
        dim_maps,
    })
}

impl Tensor {
    /// Indexes/slices the tensor using a single range/index or a tuple of ranges/indices,
    /// returning a **zero-copy** strided view that tracks gradients.
    ///
    /// ```
    /// use oxideon_core::prelude::*;
    /// let x = Tensor::arange(0.0, 12.0, 1.0, DType::F32)?.reshape(&[3, 4])?;
    /// let sub = x.i((1..3, 1..4))?;
    /// assert_eq!(sub.shape(), &[2, 3]);
    /// assert!(sub.shares_storage(&x));
    /// # Ok::<(), oxideon_core::Error>(())
    /// ```
    pub fn i(&self, index: impl IndexArg) -> Result<Tensor> {
        let elems = index.into_slice_elems();
        let spec = build_view_spec(self.shape(), &elems)?;
        self.apply_view(spec)
    }
}

/// Macro for building a `Vec<SliceElem>`, including stepped slices via
/// `SliceElem::step(start, end, step)`.
#[macro_export]
macro_rules! s {
    ($($elem:expr),* $(,)?) => {
        vec![$($crate::indexing::SliceElem::from($elem)),*]
    };
}
