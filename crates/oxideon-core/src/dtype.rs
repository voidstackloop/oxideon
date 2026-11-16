//! Data types (`DType`), scalar traits, and pure-Rust IEEE-754 `f16` & `bf16`.

use std::fmt;

use crate::error::{Error, Result};

/// Runtime data type identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    U8,
    I8,
    I16,
    U16,
    I32,
    U32,
    U64,
    I64,
    F16,
    BF16,
    F8E4M3,
    F8E5M2,
    F32,
    F64,
    C64,
    C128,
}

impl DType {
    /// Size of a single element in bytes.
    pub const fn size_in_bytes(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 | Self::F8E4M3 | Self::F8E5M2 => 1,
            Self::F16 | Self::BF16 | Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 | Self::C64 => 8,
            Self::C128 => 16,
        }
    }

    pub const fn is_floating_point(self) -> bool {
        matches!(self, Self::F16 | Self::BF16 | Self::F8E4M3 | Self::F8E5M2 | Self::F32 | Self::F64)
    }

    pub const fn is_complex(self) -> bool {
        matches!(self, Self::C64 | Self::C128)
    }

    pub const fn is_signed_int(self) -> bool {
        matches!(self, Self::I8 | Self::I16 | Self::I32 | Self::I64)
    }

    pub const fn is_int(self) -> bool {
        matches!(self, Self::U8 | Self::I8 | Self::I16 | Self::U16 | Self::I32 | Self::U32 | Self::U64 | Self::I64)
    }

    /// Standard safetensors dtype string.
    pub const fn safetensors_name(self) -> &'static str {
        match self {
            Self::Bool => "BOOL",
            Self::U8 => "U8",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::U16 => "U16",
            Self::U32 => "U32",
            Self::U64 => "U64",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::F8E4M3 => "F8_E4M3",
            Self::F8E5M2 => "F8_E5M2",
            Self::F32 => "F32",
            Self::F64 => "F64",
            Self::C64 => "C64",
            Self::C128 => "C128",
        }
    }

    pub fn from_safetensors_name(name: &str) -> Result<Self> {
        match name {
            "BOOL" => Ok(Self::Bool),
            "U8" => Ok(Self::U8),
            "I8" => Ok(Self::I8),
            "I16" => Ok(Self::I16),
            "U16" => Ok(Self::U16),
            "U32" => Ok(Self::U32),
            "U64" => Ok(Self::U64),
            "I32" => Ok(Self::I32),
            "I64" => Ok(Self::I64),
            "F16" => Ok(Self::F16),
            "BF16" => Ok(Self::BF16),
            "F8_E4M3" | "F8E4M3" => Ok(Self::F8E4M3),
            "F8_E5M2" | "F8E5M2" => Ok(Self::F8E5M2),
            "F32" => Ok(Self::F32),
            "F64" => Ok(Self::F64),
            "C64" => Ok(Self::C64),
            "C128" => Ok(Self::C128),
            _ => Err(Error::Serialization(format!(
                "unknown safetensors dtype: {name}"
            ))),
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Bool => "bool",
            Self::U8 => "u8",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::F8E4M3 => "f8e4m3",
            Self::F8E5M2 => "f8e5m2",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::C64 => "c64",
            Self::C128 => "c128",
        };
        write!(f, "{s}")
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust IEEE 754 binary16 (f16) and bfloat16 (bf16)
// ---------------------------------------------------------------------------

/// IEEE 754 half-precision floating-point (`f16`).
#[allow(non_camel_case_types)]
#[repr(transparent)]
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct f16(pub u16);

impl f16 {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(0x3C00);

    #[inline]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    #[inline]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    #[inline]
    pub fn from_f32(val: f32) -> Self {
        let b = val.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xFF) as i32;
        let mant = b & 0x007F_FFFF;

        if exp == 255 {
            // Inf or NaN
            if mant == 0 {
                return Self(sign | 0x7C00);
            } else {
                let nan_mant = (mant >> 13) as u16;
                return Self(sign | 0x7C00 | nan_mant.max(1));
            }
        }

        let new_exp = exp - 127 + 15;
        if new_exp >= 31 {
            return Self(sign | 0x7C00); // overflow -> Inf
        }
        if new_exp <= 0 {
            if new_exp < -10 {
                return Self(sign); // underflow -> 0
            }
            // Subnormal
            let full_mant = mant | 0x0080_0000;
            let shift = 14 - new_exp;
            let half_mant = (full_mant >> shift) as u16;
            let round_bit = (full_mant >> (shift - 1)) & 1;
            let sticky = full_mant & ((1 << (shift - 1)) - 1);
            let rounded = if round_bit != 0 && (sticky != 0 || (half_mant & 1) != 0) {
                half_mant + 1
            } else {
                half_mant
            };
            return Self(sign | rounded);
        }

        let half_exp = (new_exp as u16) << 10;
        let half_mant = (mant >> 13) as u16;
        let round_bit = (mant >> 12) & 1;
        let sticky = mant & 0x0FFF;
        let bits = sign | half_exp | half_mant;
        if round_bit != 0 && (sticky != 0 || (half_mant & 1) != 0) {
            Self(bits.wrapping_add(1))
        } else {
            Self(bits)
        }
    }

    #[inline]
    pub fn to_f32(self) -> f32 {
        let sign = ((self.0 & 0x8000) as u32) << 16;
        let exp = ((self.0 >> 10) & 0x1F) as u32;
        let mant = (self.0 & 0x03FF) as u32;

        if exp == 0 {
            if mant == 0 {
                return f32::from_bits(sign);
            }
            // Subnormal
            let mut m = mant;
            let mut e = 127 - 14;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03FF;
            return f32::from_bits(sign | ((e as u32) << 23) | (m << 13));
        } else if exp == 31 {
            return f32::from_bits(sign | 0x7F80_0000 | (mant << 13));
        }
        f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
    }
}

impl fmt::Debug for f16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.to_f32())
    }
}

impl fmt::Display for f16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

/// Brain floating-point (`bf16`): upper 16 bits of IEEE 754 `f32`.
#[allow(non_camel_case_types)]
#[repr(transparent)]
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct bf16(pub u16);

impl bf16 {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(0x3F80);

    #[inline]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    #[inline]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    #[inline]
    pub fn from_f32(val: f32) -> Self {
        let b = val.to_bits();
        if (b & 0x7F80_0000) == 0x7F80_0000 && (b & 0x007F_FFFF) != 0 {
            // Preserve quiet NaN
            return Self(((b >> 16) as u16) | 0x0040);
        }
        // Round to nearest even
        let lsb = (b >> 16) & 1;
        let rounding_bias = 0x7FFF + lsb;
        Self((b.wrapping_add(rounding_bias) >> 16) as u16)
    }

    #[inline]
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }
}

impl fmt::Debug for bf16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.to_f32())
    }
}

impl fmt::Display for bf16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

// ---------------------------------------------------------------------------
// Element trait for generic host buffers and conversions
// ---------------------------------------------------------------------------

/// A scalar type that can be stored inside an Oxideon tensor.
/// Truncating float→int conversion with torch-CPU semantics: values that are
/// NaN, infinite, or outside the `i64` range map to `i64::MIN` (the x86
/// `cvttsd2si` invalid result, which is what torch produces on CPU);
/// otherwise truncate toward zero.
#[inline]
fn float_to_i64_torch(v: f64) -> i64 {
    if v.is_finite() && v > (i64::MIN as f64) && v < (i64::MAX as f64) {
        v as i64
    } else {
        i64::MIN
    }
}

/// 32-bit twin: invalid maps to `i32::MIN` directly (routing through `i64::MIN`
/// would wrap the low bits to 0 instead).
#[inline]
fn float_to_i32_torch(v: f64) -> i32 {
    if v.is_finite() && v > (i32::MIN as f64) && v < (i32::MAX as f64) {
        v as i32
    } else {
        i32::MIN
    }
}

pub trait Element: Copy + Default + Send + Sync + 'static + fmt::Debug + fmt::Display {
    const DTYPE: DType;

    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn from_f32(v: f32) -> Self {
        Self::from_f64(v as f64)
    }
    fn to_f32(self) -> f32 {
        self.to_f64() as f32
    }
    fn is_zero(self) -> bool {
        self.to_f64() == 0.0
    }
    /// Conversion from `i64`; the default goes through `f64`, which is exact only up to 2^53.
    fn from_i64(v: i64) -> Self {
        Self::from_f64(v as f64)
    }
    /// Conversion to `i64`; the default goes through `f64`, which is exact only up to 2^53.
    fn to_i64(self) -> i64 {
        self.to_f64() as i64
    }
    /// Conversion from `u64`; the default goes through `f64`, which is exact only up to 2^53.
    fn from_u64(v: u64) -> Self {
        Self::from_f64(v as f64)
    }
    /// Conversion to `u64`; the default goes through `f64`, which is exact only up to 2^53.
    fn to_u64(self) -> u64 {
        self.to_f64() as u64
    }
}

impl Element for bool {
    const DTYPE: DType = DType::Bool;
    #[inline]
    fn from_f64(v: f64) -> Self {
        v != 0.0
    }
    #[inline]
    fn to_f64(self) -> f64 {
        if self { 1.0 } else { 0.0 }
    }
}

impl Element for u8 {
    const DTYPE: DType = DType::U8;
    #[inline]
    fn from_f64(v: f64) -> Self {
        // Torch routes float->u8 through i64 (truncate, then wrap low bits).
        float_to_i64_torch(v) as u8
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn from_i64(v: i64) -> Self {
        // Exact low-bit wrap (the f64 default loses bits beyond 2^53).
        v as u8
    }
}

impl Element for i8 {
    const DTYPE: DType = DType::I8;
    #[inline]
    fn from_f64(v: f64) -> Self {
        // Torch routes float -> small integer through i64 (truncate, then wrap the low bits).
        float_to_i64_torch(v) as i8
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn from_i64(v: i64) -> Self {
        v as i8
    }
}

impl Element for i16 {
    const DTYPE: DType = DType::I16;
    #[inline]
    fn from_f64(v: f64) -> Self {
        // Torch routes float -> small integer through i64 (truncate, then wrap the low bits).
        float_to_i64_torch(v) as i16
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn from_i64(v: i64) -> Self {
        v as i16
    }
}

impl Element for u16 {
    const DTYPE: DType = DType::U16;
    #[inline]
    fn from_f64(v: f64) -> Self {
        // Torch routes float -> small integer through i64 (truncate, then wrap the low bits).
        float_to_i64_torch(v) as u16
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn from_i64(v: i64) -> Self {
        v as u16
    }
}

impl Element for u32 {
    const DTYPE: DType = DType::U32;
    #[inline]
    fn from_f64(v: f64) -> Self {
        // Torch routes float -> small integer through i64 (truncate, then wrap the low bits).
        float_to_i64_torch(v) as u32
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn from_i64(v: i64) -> Self {
        v as u32
    }
}
impl Element for u64 {
    const DTYPE: DType = DType::U64;
    #[inline]
    fn from_f64(v: f64) -> Self {
        // Torch routes float -> small integer through i64 (truncate, then wrap the low bits).
        if v >= 9.223_372_036_854_776e18 { v as u64 } else { float_to_i64_torch(v) as u64 }
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn from_i64(v: i64) -> Self {
        v as u64
    }
    #[inline]
    fn to_i64(self) -> i64 {
        self as i64
    }
    #[inline]
    fn from_u64(v: u64) -> Self {
        v
    }
    #[inline]
    fn to_u64(self) -> u64 {
        self
    }
}

impl Element for i32 {
    const DTYPE: DType = DType::I32;
    #[inline]
    fn from_f64(v: f64) -> Self {
        // Truncate toward zero; NaN/inf/overflow land on INT32_MIN like torch.
        float_to_i32_torch(v)
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn from_i64(v: i64) -> Self {
        // Exact low-bit wrap (the f64 default loses bits beyond 2^53).
        v as i32
    }
}

impl Element for i64 {
    const DTYPE: DType = DType::I64;
    #[inline]
    fn from_f64(v: f64) -> Self {
        float_to_i64_torch(v)
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn from_i64(v: i64) -> Self {
        v
    }
    #[inline]
    fn to_i64(self) -> i64 {
        self
    }
    #[inline]
    fn from_u64(v: u64) -> Self {
        v as i64
    }
    #[inline]
    fn to_u64(self) -> u64 {
        self as u64
    }
}

impl Element for f16 {
    const DTYPE: DType = DType::F16;
    #[inline]
    fn from_f64(v: f64) -> Self {
        f16::from_f32(v as f32)
    }
    #[inline]
    fn to_f64(self) -> f64 {
        f16::to_f32(self) as f64
    }
    #[inline]
    fn from_f32(v: f32) -> Self {
        f16::from_f32(v)
    }
    #[inline]
    fn to_f32(self) -> f32 {
        f16::to_f32(self)
    }
}

impl Element for bf16 {
    const DTYPE: DType = DType::BF16;
    #[inline]
    fn from_f64(v: f64) -> Self {
        bf16::from_f32(v as f32)
    }
    #[inline]
    fn to_f64(self) -> f64 {
        bf16::to_f32(self) as f64
    }
    #[inline]
    fn from_f32(v: f32) -> Self {
        bf16::from_f32(v)
    }
    #[inline]
    fn to_f32(self) -> f32 {
        bf16::to_f32(self)
    }
}
impl Element for f8e4m3 {
    const DTYPE: DType = DType::F8E4M3;
    #[inline]
    fn from_f64(v: f64) -> Self {
        f8e4m3::from_f32(v as f32)
    }
    #[inline]
    fn to_f64(self) -> f64 {
        f8e4m3::to_f32(self) as f64
    }
    #[inline]
    fn from_f32(v: f32) -> Self {
        f8e4m3::from_f32(v)
    }
    #[inline]
    fn to_f32(self) -> f32 {
        f8e4m3::to_f32(self)
    }
}
impl Element for f8e5m2 {
    const DTYPE: DType = DType::F8E5M2;
    #[inline]
    fn from_f64(v: f64) -> Self {
        f8e5m2::from_f32(v as f32)
    }
    #[inline]
    fn to_f64(self) -> f64 {
        f8e5m2::to_f32(self) as f64
    }
    #[inline]
    fn from_f32(v: f32) -> Self {
        f8e5m2::from_f32(v)
    }
    #[inline]
    fn to_f32(self) -> f32 {
        f8e5m2::to_f32(self)
    }
}

impl Element for f32 {
    const DTYPE: DType = DType::F32;
    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn from_f32(v: f32) -> Self {
        v
    }
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }
}

impl Element for f64 {
    const DTYPE: DType = DType::F64;
    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn int_narrowing_wraps_like_torch() {
        // low-bit truncation, exact beyond 2^53 (torch-verified)
        let x = Tensor::from_slice(
            &[(1i64 << 60) + 7, -(1i64 << 60) - 7, 1i64 << 40, -1, 0],
            &[5],
        )
        .unwrap();
        assert_eq!(
            x.to_dtype(DType::I32).unwrap().to_vec::<i32>().unwrap(),
            vec![7, -7, 0, -1, 0]
        );
        assert_eq!(
            x.to_dtype(DType::U8).unwrap().to_vec::<u8>().unwrap(),
            vec![7, 249, 0, 255, 0]
        );
    }

    #[test]
    fn float_to_int_matches_torch_cpu() {
        // truncation toward zero; NaN/inf/out-of-range land on INT_MIN like
        // x86 cvttsd2si; u8 wraps through i64 (all torch-verified).
        let x = Tensor::from_slice(
            &[
                1.9,
                -1.9,
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                1e20,
                -1e20,
                256.7,
            ],
            &[8],
        )
        .unwrap();
        assert_eq!(
            x.to_dtype(DType::I64).unwrap().to_vec::<i64>().unwrap(),
            vec![1, -1, i64::MIN, i64::MIN, i64::MIN, i64::MIN, i64::MIN, 256]
        );
        assert_eq!(
            x.to_dtype(DType::I32).unwrap().to_vec::<i32>().unwrap(),
            vec![1, -1, i32::MIN, i32::MIN, i32::MIN, i32::MIN, i32::MIN, 256]
        );
        assert_eq!(
            x.to_dtype(DType::U8).unwrap().to_vec::<u8>().unwrap(),
            vec![1, 255, 0, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn float_to_bool_and_half_edges() {
        // only exact zeros are false; NaN/inf are true (torch-verified)
        let x = Tensor::from_slice(
            &[
                1.9,
                -1.9,
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                0.0,
                -0.0,
            ],
            &[7],
        )
        .unwrap();
        assert_eq!(
            x.to_dtype(DType::Bool).unwrap().to_vec::<bool>().unwrap(),
            vec![true, true, true, true, true, false, false]
        );
        // float16 overflow saturates to inf, NaN survives (torch-verified)
        let h = Tensor::from_slice(&[1e10, -1e10, f64::NAN], &[3]).unwrap();
        let back = h
            .to_dtype(DType::F16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(&back[..2], &[f32::INFINITY, f32::NEG_INFINITY]);
        assert!(back[2].is_nan());
    }

    #[test]
    fn int_to_float_rounds_like_torch() {
        // round-to-nearest on values f32 cannot represent (torch-verified 2^60)
        let x = Tensor::from_slice(&[(1i64 << 60) + 7, -((1i64 << 60) + 7)], &[2]).unwrap();
        let bits: Vec<u32> = x
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(bits, vec![0x5D80_0000, 0xDD80_0000]);
        // float identity casts preserve every bit incl. NaN payloads
        let f = Tensor::from_slice(&[1.5f32, f32::NAN], &[2]).unwrap();
        let a: Vec<u32> = f
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let b: Vec<u32> = f
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(a, b);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// 8-bit floats: `f8e4m3` (E4M3FN: no infinities, one NaN pattern, max 448) and `f8e5m2` (IEEE-like, max 57344), with
// round-to-nearest-even and torch's overflow behaviour (E4M3FN saturates at +-448, E5M2 goes to infinity).
// ---------------------------------------------------------------------------------------------------------------------

/// Decodes one code of a minifloat with `ebits` exponent bits and `mbits` mantissa bits.
fn minifloat_decode(code: u8, ebits: u32, mbits: u32, bias: i32, fn_variant: bool) -> f32 {
    let sign = if code >> (ebits + mbits) & 1 == 1 { -1.0f32 } else { 1.0 };
    let exp = ((code >> mbits) & ((1 << ebits) - 1)) as i32;
    let man = (code & ((1 << mbits) - 1)) as i32;
    let top = (1 << ebits) - 1;
    if exp == top {
        if fn_variant {
            if man == (1 << mbits) - 1 {
                return f32::NAN;
            }
        } else {
            return if man == 0 { sign * f32::INFINITY } else { f32::NAN };
        }
    }
    let v = if exp == 0 { man as f32 / (1 << mbits) as f32 * (2.0f32).powi(1 - bias) } else { (1.0 + man as f32 / (1 << mbits) as f32) * (2.0f32).powi(exp - bias) };
    sign * v
}

/// Encodes with round-to-nearest-even by searching the sorted table of non-negative finite codes.
fn minifloat_encode(v: f32, ebits: u32, mbits: u32, bias: i32, fn_variant: bool) -> u8 {
    let sign_bit = 1u8 << (ebits + mbits);
    let nan = (((1u8 << ebits) - 1) << mbits) | ((1u8 << mbits) - 1); // 0x7F for both formats, like torch
    if v.is_nan() {
        return nan;
    }
    let (neg, a) = (v.is_sign_negative(), v.abs());
    let max_code: u8 = if fn_variant { sign_bit - 2 } else { (((1u8 << ebits) - 1) << mbits) - 1 };
    let table = |c: u8| minifloat_decode(c, ebits, mbits, bias, fn_variant);
    // torch saturates E4M3FN (no infinities) at its largest finite value; E5M2 overflows to infinity
    let overflow = if fn_variant { max_code } else { ((1u8 << ebits) - 1) << mbits };
    let code = if a.is_infinite() {
        overflow
    } else {
        // first code whose value is >= a
        let (mut lo, mut hi) = (0u8, max_code);
        if a > table(max_code) {
            // beyond the largest finite value: round to it only if closer than the next (virtual) step, else overflow
            let step = table(max_code) - table(max_code - 1);
            let boundary = table(max_code) + step / 2.0;
            if a > boundary || (a == boundary && max_code & 1 == 1) { overflow } else { max_code }
        } else {
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if table(mid) >= a { hi = mid } else { lo = mid + 1 }
            }
            if lo == 0 || table(lo) == a {
                lo
            } else {
                let (below, above) = (table(lo - 1), table(lo));
                let (d_lo, d_hi) = (a - below, above - a);
                if d_lo < d_hi || (d_lo == d_hi && (lo - 1) & 1 == 0) { lo - 1 } else { lo }
            }
        }
    };
    if neg { code | sign_bit } else { code }
}

macro_rules! minifloat_type {
    ($name:ident, $doc:expr, $ebits:expr, $mbits:expr, $bias:expr, $fnv:expr, $one:expr) => {
        #[doc = $doc]
        #[allow(non_camel_case_types)]
        #[repr(transparent)]
        #[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
        pub struct $name(pub u8);

        impl $name {
            pub const ZERO: Self = Self(0);
            pub const ONE: Self = Self($one);
            #[inline]
            pub const fn from_bits(bits: u8) -> Self {
                Self(bits)
            }
            #[inline]
            pub const fn to_bits(self) -> u8 {
                self.0
            }
            #[inline]
            pub fn from_f32(v: f32) -> Self {
                Self(minifloat_encode(v, $ebits, $mbits, $bias, $fnv))
            }
            #[inline]
            pub fn to_f32(self) -> f32 {
                minifloat_decode(self.0, $ebits, $mbits, $bias, $fnv)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{:?}", self.to_f32())
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.to_f32())
            }
        }
    };
}

minifloat_type!(f8e4m3, "8-bit float E4M3FN (`torch.float8_e4m3fn`): 1 sign, 4 exponent, 3 mantissa bits, no infinities.", 4, 3, 7, true, 0x38);
minifloat_type!(f8e5m2, "8-bit float E5M2 (`torch.float8_e5m2`): 1 sign, 5 exponent, 2 mantissa bits, IEEE-style infinities.", 5, 2, 15, false, 0x3C);

#[cfg(test)]
mod fp8_tests {
    use super::*;

    #[test]
    fn decode_known_values_and_round_trip_every_code() {
        assert_eq!(f8e4m3(0x38).to_f32(), 1.0);
        assert_eq!(f8e4m3(0x7E).to_f32(), 448.0);
        assert!(f8e4m3(0x7F).to_f32().is_nan() && f8e4m3(0xFF).to_f32().is_nan());
        assert_eq!(f8e4m3(0x01).to_f32(), 2.0f32.powi(-9)); // smallest subnormal
        assert_eq!(f8e5m2(0x3C).to_f32(), 1.0);
        assert_eq!(f8e5m2(0x7B).to_f32(), 57344.0);
        assert_eq!(f8e5m2(0x7C).to_f32(), f32::INFINITY);
        assert_eq!(f8e5m2(0xFC).to_f32(), f32::NEG_INFINITY);
        assert!(f8e5m2(0x7D).to_f32().is_nan());
        assert_eq!(f8e5m2(0x01).to_f32(), 2.0f32.powi(-16));
        for c in 0..=255u8 {
            let v = f8e4m3(c).to_f32();
            if !v.is_nan() {
                assert_eq!(f8e4m3::from_f32(v).0 & 0x7F, c & 0x7F, "e4m3 code {c:#x}");
            }
            let w = f8e5m2(c).to_f32();
            if !w.is_nan() {
                assert_eq!(f8e5m2::from_f32(w).0, c, "e5m2 code {c:#x}");
            }
        }
    }

    #[test]
    fn rounding_ties_to_even_and_overflow_follow_torch() {
        // between 1.0 (0x38) and 1.125 (0x39): the midpoint 1.0625 ties to the even code 0x38
        assert_eq!(f8e4m3::from_f32(1.0625).0, 0x38);
        assert_eq!(f8e4m3::from_f32(1.0626).0, 0x39);
        assert_eq!(f8e4m3::from_f32(1.1875).0, 0x3A); // tie between 0x39 and 0x3A rounds to even 0x3A
        assert_eq!(f8e4m3::from_f32(-1.0).0, 0xB8);
        assert_eq!(f8e4m3::from_f32(1000.0).to_f32(), 448.0); // E4M3FN saturates
        assert_eq!(f8e4m3::from_f32(f32::NEG_INFINITY).to_f32(), -448.0);
        assert!(f8e4m3::from_f32(f32::NAN).to_f32().is_nan());
        assert_eq!(f8e4m3::from_f32(0.0).0, 0);
        assert_eq!(f8e5m2::from_f32(1.0e6).to_f32(), f32::INFINITY);
        assert_eq!(f8e5m2::from_f32(-1.0e6).to_f32(), f32::NEG_INFINITY);
        assert_eq!(f8e5m2::from_f32(1.125).0, 0x3C); // 1.125 ties between 1.0 and 1.25: even code 0x3C
        assert_eq!(f8e5m2::from_f32(1e-30).0, 0);
    }
}
