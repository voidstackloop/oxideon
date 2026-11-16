//! Error and Result types for Oxideon.

use std::fmt;

use crate::dtype::DType;

/// All fallible operations in Oxideon return `Result<T, Error>`.
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub enum Error {
    ShapeMismatch {
        op: &'static str,
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    BroadcastIncompatible {
        op: &'static str,
        lhs: Vec<usize>,
        rhs: Vec<usize>,
    },
    InvalidAxis {
        op: &'static str,
        axis: isize,
        ndim: usize,
    },
    IndexOutOfBounds {
        op: &'static str,
        index: isize,
        dim_size: usize,
    },
    DTypeMismatch {
        op: &'static str,
        expected: DType,
        got: DType,
    },
    UnsupportedDType {
        op: &'static str,
        dtype: DType,
    },
    DeviceMismatch {
        op: &'static str,
        lhs: String,
        rhs: String,
    },
    NotContiguous {
        op: &'static str,
    },
    Autograd(String),
    Serialization(String),
    Backend(String),
    Msg(String),
}

impl Error {
    pub fn msg(s: impl Into<String>) -> Self {
        Self::Msg(s.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { op, expected, got } => {
                write!(
                    f,
                    "{op}: shape mismatch (expected {expected:?}, got {got:?})"
                )
            }
            Self::BroadcastIncompatible { op, lhs, rhs } => {
                write!(
                    f,
                    "{op}: shapes {lhs:?} and {rhs:?} are not broadcast-compatible"
                )
            }
            Self::InvalidAxis { op, axis, ndim } => {
                write!(f, "{op}: invalid axis {axis} for tensor with {ndim} dims")
            }
            Self::IndexOutOfBounds {
                op,
                index,
                dim_size,
            } => {
                write!(
                    f,
                    "{op}: index {index} out of bounds for dimension of size {dim_size}"
                )
            }
            Self::DTypeMismatch { op, expected, got } => {
                write!(f, "{op}: dtype mismatch (expected {expected}, got {got})")
            }
            Self::UnsupportedDType { op, dtype } => {
                write!(f, "{op}: unsupported dtype {dtype}")
            }
            Self::DeviceMismatch { op, lhs, rhs } => {
                write!(f, "{op}: device mismatch ({lhs} vs {rhs})")
            }
            Self::NotContiguous { op } => {
                write!(f, "{op}: operation requires a contiguous tensor")
            }
            Self::Autograd(s) => write!(f, "autograd error: {s}"),
            Self::Serialization(s) => write!(f, "serialization error: {s}"),
            Self::Backend(s) => write!(f, "backend error: {s}"),
            Self::Msg(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for Error {}

#[macro_export]
macro_rules! bail {
    ($($arg:tt)*) => {
        return Err($crate::error::Error::Msg(format!($($arg)*)))
    };
}
