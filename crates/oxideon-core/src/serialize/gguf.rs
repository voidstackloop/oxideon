//! Native GGUF (GGML Universal Format) v2/v3 binary parser, writer, and dequantizer.
//!
//! Provides zero-copy parsing of GGUF headers, metadata key-value trees, tensor directories, and
//! dequantization for every type GGUF commonly carries (`F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1,
//! Q8_0, Q4_K, Q5_K, Q6_K`), plus fused GEMV for `Q4_0`/`Q8_0` (others fall back to dequantize+matmul).
//!
//! Dequantization for `Q4_1`, `Q5_0`, `Q5_1`, `Q4_K`, `Q5_K` and `Q6_K` follows GGML's reference
//! bit layouts exactly (block struct field order, split-half nibble packing, `get_scale_min_k4`),
//! so files produced by llama.cpp / GGML tooling decode correctly, not just this crate's own
//! round-trip. See `tests` below for layout regression checks against hand-built reference bytes.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::dtype::{bf16, f16};
use crate::error::{Error, Result};
use crate::tensor::Tensor;

pub const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian
pub const GGUF_DEFAULT_ALIGNMENT: usize = 32;

/// GGML data types supported in GGUF tensor directories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39,
    NVFP4 = 40,
    Q1_0 = 41,
    Q2_0 = 42,
}

impl GGMLType {
    pub fn from_u32(val: u32) -> Result<Self> {
        match val {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            10 => Ok(Self::Q2_K),
            11 => Ok(Self::Q3_K),
            12 => Ok(Self::Q4_K),
            13 => Ok(Self::Q5_K),
            14 => Ok(Self::Q6_K),
            16 => Ok(Self::IQ2_XXS),
            17 => Ok(Self::IQ2_XS),
            18 => Ok(Self::IQ3_XXS),
            19 => Ok(Self::IQ1_S),
            20 => Ok(Self::IQ4_NL),
            21 => Ok(Self::IQ3_S),
            22 => Ok(Self::IQ2_S),
            23 => Ok(Self::IQ4_XS),
            24 => Ok(Self::I8),
            25 => Ok(Self::I16),
            26 => Ok(Self::I32),
            27 => Ok(Self::I64),
            28 => Ok(Self::F64),
            29 => Ok(Self::IQ1_M),
            30 => Ok(Self::BF16),
            34 => Ok(Self::TQ1_0),
            35 => Ok(Self::TQ2_0),
            39 => Ok(Self::MXFP4),
            40 => Ok(Self::NVFP4),
            41 => Ok(Self::Q1_0),
            42 => Ok(Self::Q2_0),
            other => Err(Error::Msg(format!("Unsupported GGMLType id: {other}"))),
        }
    }

    /// Block size (number of weights represented by one block of bytes).
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32
            | Self::F16
            | Self::BF16
            | Self::F64
            | Self::I8
            | Self::I16
            | Self::I32
            | Self::I64 => 1,
            Self::Q4_0
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::IQ4_NL
            | Self::MXFP4 => 32,
            Self::NVFP4 | Self::Q2_0 => 64,
            Self::Q1_0 => 128,
            Self::Q2_K
            | Self::Q3_K
            | Self::Q4_K
            | Self::Q5_K
            | Self::Q6_K
            | Self::IQ4_XS
            | Self::TQ1_0
            | Self::TQ2_0
            | Self::IQ2_XXS
            | Self::IQ2_XS
            | Self::IQ2_S
            | Self::IQ3_XXS
            | Self::IQ3_S
            | Self::IQ1_S
            | Self::IQ1_M => 256,
        }
    }

    /// Number of bytes per block.
    pub fn block_size_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 2 + 16, // f16 scale (2 bytes) + 16 bytes (32 x 4-bit nibbles)
            Self::Q4_1 => 2 + 2 + 16, // f16 scale + f16 min + 16 bytes
            Self::Q5_0 => 2 + 4 + 16,
            Self::Q5_1 => 2 + 2 + 4 + 16,
            Self::Q8_0 => 2 + 32, // f16 scale (2 bytes) + 32 bytes (32 x i8)
            Self::IQ2_XXS => 2 + 64,
            Self::IQ2_XS => 2 + 64 + 8,
            Self::IQ2_S => 2 + 64 + 8 + 8,
            Self::IQ3_XXS => 2 + 96,
            Self::IQ3_S => 2 + 64 + 8 + 32 + 4,
            Self::IQ1_S => 2 + 32 + 16,
            Self::IQ1_M => 32 + 16 + 8,
            Self::F64 | Self::I64 => 8,
            Self::I32 => 4,
            Self::I16 => 2,
            Self::I8 => 1,
            Self::TQ1_0 => 48 + 4 + 2,
            Self::TQ2_0 => 64 + 2,
            Self::MXFP4 => 1 + 16,
            Self::NVFP4 => 4 + 32,
            Self::Q1_0 | Self::Q2_0 => 2 + 16,
            Self::IQ4_NL => 2 + 16,              // d + 16 bytes of nibbles
            Self::IQ4_XS => 2 + 2 + 4 + 128,     // d + scales_h + scales_l + qs = 136
            Self::Q2_K => 16 + 64 + 2 + 2,       // scales + qs + d + dmin = 84
            Self::Q3_K => 32 + 64 + 12 + 2,      // hmask + qs + scales + d = 110
            Self::Q4_K => 2 + 2 + 12 + 128,      // 144 bytes for 256 weights
            Self::Q5_K => 2 + 2 + 12 + 128 + 32, // 176 bytes
            Self::Q6_K => 128 + 64 + 16 + 2,     // 210 bytes
        }
    }

    /// Computes the total byte footprint for a tensor with `numel` elements.
    pub fn tensor_bytes(&self, numel: usize) -> Result<usize> {
        let b = self.block_size();
        let num_blocks = numel.div_ceil(b);
        num_blocks
            .checked_mul(self.block_size_bytes())
            .ok_or_else(|| {
                Error::Msg(format!(
                    "GGMLType::{self:?} byte footprint overflows for {numel} elements"
                ))
            })
    }
}

/// Metadata value stored in GGUF header.
#[derive(Debug, Clone, PartialEq)]
pub enum GGUFValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GGUFValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GGUFValue {
    pub fn type_id(&self) -> u32 {
        match self {
            Self::Uint8(_) => 0,
            Self::Int8(_) => 1,
            Self::Uint16(_) => 2,
            Self::Int16(_) => 3,
            Self::Uint32(_) => 4,
            Self::Int32(_) => 5,
            Self::Float32(_) => 6,
            Self::Bool(_) => 7,
            Self::String(_) => 8,
            Self::Array(_) => 9,
            Self::Uint64(_) => 10,
            Self::Int64(_) => 11,
            Self::Float64(_) => 12,
        }
    }
}

/// Information describing a single tensor inside a GGUF file.
#[derive(Debug, Clone)]
pub struct GGUFTensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub ggml_type: GGMLType,
    pub offset: u64,
    pub size_bytes: usize,
}

/// Deserialized GGUF container holding parsed metadata, tensor catalog, and raw tensor buffers.
#[derive(Debug, Clone)]
pub struct GGUFFile {
    pub version: u32,
    pub alignment: usize,
    pub metadata: HashMap<String, GGUFValue>,
    pub tensors: HashMap<String, GGUFTensorInfo>,
    tensor_data: Vec<u8>,
}

impl GGUFFile {
    /// Parses a GGUF file from raw memory bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut cursor = 0usize;

        // 1. Header Magic & Version
        if bytes.len() < 16 {
            return Err(Error::Msg("GGUF buffer too short for header".into()));
        }

        let magic = read_u32_le(bytes, &mut cursor)?;
        if magic != GGUF_MAGIC {
            return Err(Error::Msg(format!(
                "Invalid GGUF magic: expected 0x{:08X}, got 0x{:08X}",
                GGUF_MAGIC, magic
            )));
        }

        let version = read_u32_le(bytes, &mut cursor)?;
        if version != 2 && version != 3 {
            return Err(Error::Msg(format!(
                "Unsupported GGUF version: {version} (expected 2 or 3)"
            )));
        }

        let tensor_count = read_u64_le(bytes, &mut cursor)? as usize;
        let metadata_kv_count = read_u64_le(bytes, &mut cursor)? as usize;

        // 2. Metadata Key-Value pairs
        if metadata_kv_count > bytes.len() || tensor_count > bytes.len() {
            return Err(Error::Msg("GGUF header counts exceed file size".into()));
        }
        // Fallible reserve: the count only needs to fit the file, so a hostile
        // count aborts the upfront reserve instead of failing on the first read.
        let mut metadata = HashMap::new();
        metadata
            .try_reserve(metadata_kv_count)
            .map_err(|_| Error::Msg("GGUF metadata count exceeds memory".into()))?;
        for _ in 0..metadata_kv_count {
            let key = read_string(bytes, &mut cursor)?;
            let val_type = read_u32_le(bytes, &mut cursor)?;
            let val = read_value(bytes, &mut cursor, val_type)?;
            metadata.insert(key, val);
        }

        let alignment = match metadata.get("general.alignment") {
            Some(GGUFValue::Uint32(a)) => *a as usize,
            // A u64 alignment that does not fit usize (32-bit targets) must
            // not truncate into a wrong data offset.
            Some(GGUFValue::Uint64(a)) => usize::try_from(*a)
                .map_err(|_| Error::Msg("GGUF general.alignment exceeds usize".into()))?,
            _ => GGUF_DEFAULT_ALIGNMENT,
        };

        // 3. Tensor Catalog
        let mut tensors = HashMap::new();
        tensors
            .try_reserve(tensor_count)
            .map_err(|_| Error::Msg("GGUF tensor count exceeds memory".into()))?;
        for _ in 0..tensor_count {
            let name = read_string(bytes, &mut cursor)?;
            let n_dims = read_u32_le(bytes, &mut cursor)? as usize;
            if n_dims > 8 {
                return Err(Error::Msg(format!(
                    "GGUF tensor '{name}' has {n_dims} dimensions"
                )));
            }
            let mut shape = Vec::with_capacity(n_dims);
            let mut numel = 1usize;
            for _ in 0..n_dims {
                let d = read_u64_le(bytes, &mut cursor)? as usize;
                shape.push(d);
                numel = numel.checked_mul(d).ok_or_else(|| {
                    Error::Msg(format!("GGUF tensor '{name}' element count overflows"))
                })?;
            }
            // GGUF stores dimensions in column-major [columns, rows, ...].
            // Reverse to conventional row-major [batch, ..., channels] for deep learning libraries.
            shape.reverse();

            let type_id = read_u32_le(bytes, &mut cursor)?;
            let ggml_type = GGMLType::from_u32(type_id)?;
            let offset = read_u64_le(bytes, &mut cursor)?;
            let size_bytes = numel
                .div_ceil(ggml_type.block_size())
                .checked_mul(ggml_type.block_size_bytes())
                .ok_or_else(|| Error::Msg(format!("GGUF tensor '{name}' byte size overflows")))?;

            tensors.insert(
                name.clone(),
                GGUFTensorInfo {
                    name,
                    shape,
                    ggml_type,
                    offset,
                    size_bytes,
                },
            );
        }

        // Align cursor to tensor data section
        if alignment == 0 {
            return Err(Error::Msg("GGUF general.alignment must be non-zero".into()));
        }
        let rem = cursor % alignment;
        // A wrapped data_start would silently slice the wrong tensor bytes.
        let data_start = if rem == 0 {
            cursor
        } else {
            cursor
                .checked_add(alignment - rem)
                .ok_or_else(|| Error::Msg("GGUF tensor data offset overflows".into()))?
        };

        if data_start > bytes.len() {
            return Err(Error::Msg(
                "GGUF buffer truncated before tensor data section".into(),
            ));
        }

        let tensor_data = bytes[data_start..].to_vec();

        Ok(Self {
            version,
            alignment,
            metadata,
            tensors,
            tensor_data,
        })
    }

    /// Reads and parses a GGUF model file from disk.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let bytes =
            fs::read(path).map_err(|e| Error::Msg(format!("Failed to read GGUF file: {e}")))?;
        Self::from_bytes(&bytes)
    }

    /// Returns a string metadata value if present.
    pub fn get_metadata_str(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key) {
            Some(GGUFValue::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Returns an unsigned 32-bit integer metadata value if present.
    pub fn get_metadata_u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key) {
            Some(GGUFValue::Uint32(v)) => Some(*v),
            Some(GGUFValue::Int32(v)) if *v >= 0 => Some(*v as u32),
            // Never truncate: an out-of-range u64 falls back to None (and the
            // caller's default) instead of silently becoming a wrong small dim.
            Some(GGUFValue::Uint64(v)) => u32::try_from(*v).ok(),
            _ => None,
        }
    }

    /// Returns a 32-bit float metadata value if present.
    pub fn get_metadata_f32(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key) {
            Some(GGUFValue::Float32(v)) => Some(*v),
            Some(GGUFValue::Float64(v)) => Some(*v as f32),
            _ => None,
        }
    }

    /// Retrieves raw byte slice for the named tensor.
    pub fn get_tensor_data(&self, name: &str) -> Result<&[u8]> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::Msg(format!("Tensor '{name}' not found in GGUF file")))?;

        let start = usize::try_from(info.offset).unwrap_or(usize::MAX);
        let end = start.saturating_add(info.size_bytes);

        if end > self.tensor_data.len() {
            return Err(Error::Msg(format!(
                "Tensor '{name}' data slice [{start}..{end}] out of bounds (total data len {})",
                self.tensor_data.len()
            )));
        }

        Ok(&self.tensor_data[start..end])
    }

    /// Dequantizes the specified tensor into a standard continuous [`Tensor`].
    pub fn dequantize_tensor(&self, name: &str) -> Result<Tensor> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::Msg(format!("Tensor '{name}' not found in GGUF file")))?;

        let data = self.get_tensor_data(name)?;
        // Tensor infos are pub: a tampered shape overflowed the product
        // (panic) or aborted the reserve below; cap it like any user buffer.
        let numel: usize = crate::limits::check_elements("gguf dequantize", &info.shape)?;
        let mut out_f32 = Vec::with_capacity(numel);

        match info.ggml_type {
            GGMLType::F32 => {
                for chunk in data.chunks_exact(4) {
                    out_f32.push(f32::from_le_bytes(chunk.try_into().unwrap()));
                }
            }
            GGMLType::F16 => {
                for chunk in data.chunks_exact(2) {
                    let h = f16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap()));
                    out_f32.push(h.to_f32());
                }
            }
            GGMLType::BF16 => {
                for chunk in data.chunks_exact(2) {
                    let b = bf16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap()));
                    out_f32.push(b.to_f32());
                }
            }
            GGMLType::Q8_0 => {
                let block_bytes = info.ggml_type.block_size_bytes(); // 34
                for block in data.chunks_exact(block_bytes) {
                    let scale = f16::from_bits(u16::from_le_bytes(block[0..2].try_into().unwrap()))
                        .to_f32();
                    for &q in &block[2..34] {
                        let val = (q as i8 as f32) * scale;
                        out_f32.push(val);
                    }
                }
            }
            GGMLType::Q4_0 => {
                // d(2) | qs[16]: each byte packs x[j] (low nibble) and x[j+16] (high nibble) of the
                // 32-wide block, matching GGML's reference layout (not an adjacent-pair packing).
                let block_bytes = info.ggml_type.block_size_bytes(); // 18
                for block in data.chunks_exact(block_bytes) {
                    let scale = f16::from_bits(u16::from_le_bytes(block[0..2].try_into().unwrap()))
                        .to_f32();
                    let qs = &block[2..18];
                    let mut lo = [0f32; 16];
                    let mut hi = [0f32; 16];
                    for (j, &b) in qs.iter().enumerate() {
                        lo[j] = ((b & 0x0F) as i8 - 8) as f32 * scale;
                        hi[j] = (((b >> 4) & 0x0F) as i8 - 8) as f32 * scale;
                    }
                    out_f32.extend_from_slice(&lo);
                    out_f32.extend_from_slice(&hi);
                }
            }
            GGMLType::Q4_1 => {
                // d(2) | m(2) | qs[16]: split-half nibble packing like Q4_0; x = q*d + m, q in [0,15].
                let block_bytes = info.ggml_type.block_size_bytes(); // 20
                for block in data.chunks_exact(block_bytes) {
                    let d = f16::from_bits(u16::from_le_bytes(block[0..2].try_into().unwrap()))
                        .to_f32();
                    let m = f16::from_bits(u16::from_le_bytes(block[2..4].try_into().unwrap()))
                        .to_f32();
                    let qs = &block[4..20];
                    let mut lo = [0f32; 16];
                    let mut hi = [0f32; 16];
                    for (j, &b) in qs.iter().enumerate() {
                        lo[j] = (b & 0x0F) as f32 * d + m;
                        hi[j] = ((b >> 4) & 0x0F) as f32 * d + m;
                    }
                    out_f32.extend_from_slice(&lo);
                    out_f32.extend_from_slice(&hi);
                }
            }
            GGMLType::Q5_0 => {
                // d(2) | qh[4] (5th bit per weight) | qs[16] (low nibbles), split-half: x = (q - 16) * d.
                let block_bytes = info.ggml_type.block_size_bytes(); // 22
                for block in data.chunks_exact(block_bytes) {
                    let d = f16::from_bits(u16::from_le_bytes(block[0..2].try_into().unwrap()))
                        .to_f32();
                    let qh = u32::from_le_bytes(block[2..6].try_into().unwrap());
                    let qs = &block[6..22];
                    let mut lo = [0f32; 16];
                    let mut hi = [0f32; 16];
                    for (j, &b) in qs.iter().enumerate() {
                        let xh_0 = ((qh >> j) << 4) & 0x10;
                        let xh_1 = (qh >> (j + 12)) & 0x10;
                        let x0 = (((b & 0x0F) as u32 | xh_0) as i32) - 16;
                        let x1 = (((b >> 4) as u32 | xh_1) as i32) - 16;
                        lo[j] = x0 as f32 * d;
                        hi[j] = x1 as f32 * d;
                    }
                    out_f32.extend_from_slice(&lo);
                    out_f32.extend_from_slice(&hi);
                }
            }
            GGMLType::Q5_1 => {
                // d(2) | m(2) | qh[4] | qs[16]: x = q*d + m, q = low nibble | 5th bit.
                let block_bytes = info.ggml_type.block_size_bytes(); // 24
                for block in data.chunks_exact(block_bytes) {
                    let d = f16::from_bits(u16::from_le_bytes(block[0..2].try_into().unwrap()))
                        .to_f32();
                    let m = f16::from_bits(u16::from_le_bytes(block[2..4].try_into().unwrap()))
                        .to_f32();
                    let qh = u32::from_le_bytes(block[4..8].try_into().unwrap());
                    let qs = &block[8..24];
                    let mut lo = [0f32; 16];
                    let mut hi = [0f32; 16];
                    for (j, &b) in qs.iter().enumerate() {
                        let xh_0 = ((qh >> j) << 4) & 0x10;
                        let xh_1 = (qh >> (j + 12)) & 0x10;
                        lo[j] = (((b & 0x0F) as u32) | xh_0) as f32 * d + m;
                        hi[j] = (((b >> 4) as u32) | xh_1) as f32 * d + m;
                    }
                    out_f32.extend_from_slice(&lo);
                    out_f32.extend_from_slice(&hi);
                }
            }
            GGMLType::F64 => {
                for chunk in data.chunks_exact(8) {
                    out_f32.push(f64::from_le_bytes(chunk.try_into().unwrap()) as f32);
                }
            }
            GGMLType::I8 => out_f32.extend(data.iter().map(|&b| b as i8 as f32)),
            GGMLType::I16 => {
                for chunk in data.chunks_exact(2) {
                    out_f32.push(i16::from_le_bytes(chunk.try_into().unwrap()) as f32);
                }
            }
            GGMLType::I32 => {
                for chunk in data.chunks_exact(4) {
                    out_f32.push(i32::from_le_bytes(chunk.try_into().unwrap()) as f32);
                }
            }
            GGMLType::I64 => {
                for chunk in data.chunks_exact(8) {
                    out_f32.push(i64::from_le_bytes(chunk.try_into().unwrap()) as f32);
                }
            }
            GGMLType::TQ1_0 => {
                for block in data.chunks_exact(info.ggml_type.block_size_bytes()) {
                    dequant_tq1_0_block(block, &mut out_f32);
                }
            }
            GGMLType::TQ2_0 => {
                for block in data.chunks_exact(info.ggml_type.block_size_bytes()) {
                    let d = half_at(block, 64);
                    for j in (0..64).step_by(32) {
                        for l in 0..4 {
                            for m in 0..32 {
                                out_f32
                                    .push((((block[j + m] >> (2 * l)) & 3) as i32 - 1) as f32 * d);
                            }
                        }
                    }
                }
            }
            GGMLType::MXFP4 => {
                for block in data.chunks_exact(info.ggml_type.block_size_bytes()) {
                    let d = e8m0_to_f32_half(block[0]);
                    let qs = &block[1..17];
                    out_f32.extend(
                        qs.iter()
                            .map(|&b| KVALUES_FP4[(b & 0x0F) as usize] as f32 * d),
                    );
                    out_f32.extend(
                        qs.iter()
                            .map(|&b| KVALUES_FP4[(b >> 4) as usize] as f32 * d),
                    );
                }
            }
            GGMLType::NVFP4 => {
                for block in data.chunks_exact(info.ggml_type.block_size_bytes()) {
                    for s in 0..4 {
                        let d = ue4m3_to_f32(block[s]);
                        let qs = &block[4 + 8 * s..4 + 8 * s + 8];
                        out_f32.extend(
                            qs.iter()
                                .map(|&b| KVALUES_FP4[(b & 0x0F) as usize] as f32 * d),
                        );
                        out_f32.extend(
                            qs.iter()
                                .map(|&b| KVALUES_FP4[(b >> 4) as usize] as f32 * d),
                        );
                    }
                }
            }
            GGMLType::Q1_0 => {
                for block in data.chunks_exact(info.ggml_type.block_size_bytes()) {
                    let d = half_at(block, 0);
                    for j in 0..128 {
                        out_f32.push(if (block[2 + j / 8] >> (j % 8)) & 1 != 0 {
                            d
                        } else {
                            -d
                        });
                    }
                }
            }
            GGMLType::Q2_0 => {
                for block in data.chunks_exact(info.ggml_type.block_size_bytes()) {
                    let d = half_at(block, 0);
                    for j in 0..64 {
                        // 00 = -1, 01 = 0, 10 = +1, 11 = +2
                        out_f32.push(
                            (((block[2 + j / 4] >> ((j % 4) * 2)) & 3) as i32 - 1) as f32 * d,
                        );
                    }
                }
            }
            GGMLType::IQ2_XXS => {
                for block in data.chunks_exact(info.ggml_type.block_size_bytes()) {
                    dequant_iq2_xxs_block(block, &mut out_f32);
                }
            }
            GGMLType::IQ2_XS => {
                for block in data.chunks_exact(info.ggml_type.block_size_bytes()) {
                    dequant_iq2_xs_block(block, &mut out_f32);
                }
            }
            GGMLType::IQ2_S => {
                for block in data.chunks_exact(info.ggml_type.block_size_bytes()) {
                    dequant_iq2_s_block(block, &mut out_f32);
                }
            }
            GGMLType::IQ3_XXS => {
                for block in data.chunks_exact(info.ggml_type.block_size_bytes()) {
                    dequant_iq3_xxs_block(block, &mut out_f32);
                }
            }
            GGMLType::IQ3_S => {
                for block in data.chunks_exact(info.ggml_type.block_size_bytes()) {
                    dequant_iq3_s_block(block, &mut out_f32);
                }
            }
            GGMLType::IQ1_S => {
                for block in data.chunks_exact(info.ggml_type.block_size_bytes()) {
                    dequant_iq1_s_block(block, &mut out_f32);
                }
            }
            GGMLType::IQ1_M => {
                for block in data.chunks_exact(info.ggml_type.block_size_bytes()) {
                    dequant_iq1_m_block(block, &mut out_f32);
                }
            }
            GGMLType::IQ4_NL => {
                for block in data.chunks_exact(info.ggml_type.block_size_bytes()) {
                    let d = f16::from_bits(u16::from_le_bytes(block[0..2].try_into().unwrap()))
                        .to_f32();
                    dequant_iq4_nibbles(d, &block[2..18], &mut out_f32);
                }
            }
            GGMLType::IQ4_XS => {
                for block in data.chunks_exact(info.ggml_type.block_size_bytes()) {
                    dequant_iq4_xs_block(block, &mut out_f32);
                }
            }
            GGMLType::Q2_K => {
                for block in data.chunks_exact(info.ggml_type.block_size_bytes()) {
                    dequant_q2_k_block(block, &mut out_f32);
                }
            }
            GGMLType::Q3_K => {
                for block in data.chunks_exact(info.ggml_type.block_size_bytes()) {
                    dequant_q3_k_block(block, &mut out_f32);
                }
            }
            GGMLType::Q4_K => {
                for block in data.chunks_exact(info.ggml_type.block_size_bytes()) {
                    dequant_q4_k_block(block, &mut out_f32);
                }
            }
            GGMLType::Q5_K => {
                for block in data.chunks_exact(info.ggml_type.block_size_bytes()) {
                    dequant_q5_k_block(block, &mut out_f32);
                }
            }
            GGMLType::Q6_K => {
                for block in data.chunks_exact(info.ggml_type.block_size_bytes()) {
                    dequant_q6_k_block(block, &mut out_f32);
                }
            }
        }

        out_f32.truncate(numel);
        Tensor::from_slice(&out_f32, &info.shape)
    }

    /// Fast matrix-vector product ($y = W x$) directly over quantized GGUF weights without full unpacking.
    pub fn fused_gemv(&self, name: &str, x: &Tensor) -> Result<Tensor> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::Msg(format!("Tensor '{name}' not found in GGUF file")))?;

        if info.shape.len() != 2 {
            return Err(Error::Msg(format!(
                "fused_gemv requires 2D weight matrix, found shape {:?}",
                info.shape
            )));
        }

        let rows = info.shape[0];
        let cols = info.shape[1];

        let x_vec = x.to_vec::<f32>()?;
        if x_vec.len() != cols {
            return Err(Error::ShapeMismatch {
                op: "gguf_fused_gemv",
                expected: vec![cols],
                got: vec![x_vec.len()],
            });
        }

        let data = self.get_tensor_data(name)?;
        // Shape/rows are pub (tamperable): validate the fused byte math with
        // checked arithmetic BEFORE reserving `rows` outputs (giant rows
        // aborted the reserve, and row strides overflowed below).
        if matches!(info.ggml_type, GGMLType::Q8_0 | GGMLType::Q4_0) {
            if cols % 32 != 0 {
                return Err(Error::Msg(format!(
                    "fused_gemv: {cols} columns is not a multiple of the 32-wide quantization block"
                )));
            }
            let expect = rows
                .checked_mul(cols / 32)
                .and_then(|b| b.checked_mul(info.ggml_type.block_size_bytes()))
                .ok_or_else(|| Error::Msg("fused_gemv: weight byte size overflows".into()))?;
            if expect > data.len() {
                return Err(Error::Msg(format!(
                    "fused_gemv: [{rows}x{cols}] needs {expect} weight bytes, tensor has {}",
                    data.len()
                )));
            }
        }
        let mut y_out = vec![0.0f32; rows];

        match info.ggml_type {
            GGMLType::Q8_0 => {
                let blocks_per_row = cols / 32;
                let row_bytes = blocks_per_row * 34;

                for r in 0..rows {
                    let row_data = &data[r * row_bytes..(r + 1) * row_bytes];
                    let mut acc = 0.0f32;
                    for b in 0..blocks_per_row {
                        let block = &row_data[b * 34..(b + 1) * 34];
                        let scale =
                            f16::from_bits(u16::from_le_bytes(block[0..2].try_into().unwrap()))
                                .to_f32();
                        let x_chunk = &x_vec[b * 32..(b + 1) * 32];
                        for i in 0..32 {
                            let q = block[2 + i] as i8 as i32;
                            // Multiply-accumulate
                            acc += (q as f32) * scale * x_chunk[i];
                        }
                    }
                    y_out[r] = acc;
                }
            }
            GGMLType::Q4_0 => {
                // Split-half layout (see dequantize_tensor): byte i's low nibble is x[i], high
                // nibble is x[i+16] of the 32-wide block, not an adjacent pair.
                let blocks_per_row = cols / 32;
                let row_bytes = blocks_per_row * 18;

                for r in 0..rows {
                    let row_data = &data[r * row_bytes..(r + 1) * row_bytes];
                    let mut acc = 0.0f32;
                    for b in 0..blocks_per_row {
                        let block = &row_data[b * 18..(b + 1) * 18];
                        let scale =
                            f16::from_bits(u16::from_le_bytes(block[0..2].try_into().unwrap()))
                                .to_f32();
                        let x_chunk = &x_vec[b * 32..(b + 1) * 32];
                        for i in 0..16 {
                            let byte = block[2 + i];
                            let q0 = (byte & 0x0F) as i8 - 8;
                            let q1 = ((byte >> 4) & 0x0F) as i8 - 8;
                            acc +=
                                ((q0 as f32) * x_chunk[i] + (q1 as f32) * x_chunk[i + 16]) * scale;
                        }
                    }
                    y_out[r] = acc;
                }
            }
            _ => {
                // Fallback to dequantize + standard matmul
                let w = self.dequantize_tensor(name)?;
                let wt = w.t()?;
                let is_1d = x.ndim() == 1;
                let x_2d = if is_1d {
                    x.reshape(&[1, cols])?
                } else {
                    x.clone()
                };
                let y = x_2d.matmul(&wt)?;
                return if is_1d { y.reshape(&[rows]) } else { Ok(y) };
            }
        }

        Tensor::from_slice_on(&y_out, &[rows], &x.device())
    }
}

/// Serializes metadata and tensors into GGUF v3 binary format.
pub fn serialize_gguf(
    metadata: &HashMap<String, GGUFValue>,
    tensors: &[(&str, GGMLType, &Tensor)],
    alignment: usize,
) -> Result<Vec<u8>> {
    // The parser rejects 0 the same way; 0 divided by zero below.
    if alignment == 0 {
        return Err(Error::Msg("GGUF alignment must be non-zero".into()));
    }
    let mut header_buf = Vec::new();

    // 1. Magic + Version
    header_buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    header_buf.extend_from_slice(&3u32.to_le_bytes()); // version 3
    header_buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    header_buf.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

    // 2. Metadata KV
    for (k, v) in metadata {
        write_string(&mut header_buf, k);
        header_buf.extend_from_slice(&v.type_id().to_le_bytes());
        write_value(&mut header_buf, v)?;
    }

    // 3. Tensor Catalog & Data Body
    let mut data_body = Vec::new();
    let mut current_offset = 0u64;

    for &(name, ggml_type, tensor) in tensors {
        write_string(&mut header_buf, name);
        let shape = tensor.shape();
        header_buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());

        // GGUF stores dimensions reversed (column-major)
        for &d in shape.iter().rev() {
            header_buf.extend_from_slice(&(d as u64).to_le_bytes());
        }

        header_buf.extend_from_slice(&(ggml_type as u32).to_le_bytes());

        // Align each tensor within the data body
        let rem = current_offset as usize % alignment;
        if rem != 0 {
            let pad = alignment - rem;
            // Fallible: a giant alignment aborted the pad below.
            data_body
                .try_reserve(pad)
                .map_err(|_| Error::Msg("GGUF alignment padding exceeds memory".into()))?;
            data_body.resize(data_body.len() + pad, 0u8);
            current_offset += pad as u64;
        }

        header_buf.extend_from_slice(&current_offset.to_le_bytes());

        // Quantize or copy tensor bytes
        let encoded_bytes = encode_tensor_for_gguf(tensor, ggml_type)?;
        current_offset += encoded_bytes.len() as u64;
        data_body.extend_from_slice(&encoded_bytes);
    }

    // 4. Align header before data body
    let rem = header_buf.len() % alignment;
    if rem != 0 {
        let pad = alignment - rem;
        header_buf
            .try_reserve(pad)
            .map_err(|_| Error::Msg("GGUF alignment padding exceeds memory".into()))?;
        header_buf.resize(header_buf.len() + pad, 0u8);
    }

    header_buf.extend_from_slice(&data_body);
    Ok(header_buf)
}

/// Saves metadata and tensors into a `.gguf` file on disk.
pub fn save_gguf(
    path: impl AsRef<Path>,
    metadata: &HashMap<String, GGUFValue>,
    tensors: &[(&str, GGMLType, &Tensor)],
) -> Result<()> {
    let bytes = serialize_gguf(metadata, tensors, GGUF_DEFAULT_ALIGNMENT)?;
    fs::write(path, bytes).map_err(|e| Error::Msg(format!("Failed to write GGUF: {e}")))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn encode_tensor_for_gguf(t: &Tensor, dtype: GGMLType) -> Result<Vec<u8>> {
    let raw_f32 = t.to_vec::<f32>()?;
    let mut out = Vec::new();

    match dtype {
        GGMLType::F32 => {
            for &x in &raw_f32 {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        GGMLType::F16 => {
            for &x in &raw_f32 {
                out.extend_from_slice(&f16::from_f32(x).to_bits().to_le_bytes());
            }
        }
        GGMLType::BF16 => {
            for &x in &raw_f32 {
                out.extend_from_slice(&bf16::from_f32(x).to_bits().to_le_bytes());
            }
        }
        GGMLType::Q8_0 => {
            for chunk in raw_f32.chunks(32) {
                let max_abs = chunk.iter().fold(0.0f32, |m, &x| m.max(x.abs())).max(1e-8);
                let d = max_abs / 127.0;
                let id = if d > 0.0 { 1.0 / d } else { 0.0 };
                out.extend_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
                for &x in chunk {
                    let q = (x * id).round().clamp(-127.0, 127.0) as i8;
                    out.push(q as u8);
                }
                if chunk.len() < 32 {
                    out.resize(out.len() + (32 - chunk.len()), 0);
                }
            }
        }
        GGMLType::Q4_0 => {
            // Split-half packing (byte j = x[j] | x[j+16] << 4), matching GGML's reference layout
            // and the split-half decode above — not an adjacent-pair packing.
            for chunk in raw_f32.chunks(32) {
                let max_abs = chunk.iter().fold(0.0f32, |m, &x| m.max(x.abs())).max(1e-8);
                let d = max_abs / 7.0;
                let id = if d > 0.0 { 1.0 / d } else { 0.0 };
                out.extend_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());

                let mut padded = chunk.to_vec();
                padded.resize(32, 0.0);

                for j in 0..16 {
                    let q0 = (padded[j] * id).round().clamp(-7.0, 7.0) as i8;
                    let q1 = (padded[j + 16] * id).round().clamp(-7.0, 7.0) as i8;
                    let byte = ((q0 + 8) as u8 & 0x0F) | (((q1 + 8) as u8 & 0x0F) << 4);
                    out.push(byte);
                }
            }
        }
        other => {
            return Err(Error::Msg(format!(
                "Encoding for {other:?} not yet implemented"
            )));
        }
    }

    Ok(out)
}

/// Unpacks the 6-bit (scale, min) pair for sub-block `j` (0..8) from a K-quant super-block's
/// 12-byte `scales` array. Matches GGML's `get_scale_min_k4`.
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

use super::gguf_iq_tables::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KSIGNS_IQ2XS,
};

const IQ1S_DELTA: f32 = 0.125;

/// FP4 (E2M1) codebook shared by `MXFP4` and `NVFP4` (GGML `kvalues_fp4`, values doubled).
const KVALUES_FP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// `2^(x - 128)`, GGML's `e8m0_to_fp32_half` (the MXFP4 block scale, halved to undo the doubled codebook).
fn e8m0_to_f32_half(x: u8) -> f32 {
    f32::from_bits(if x < 2 {
        0x0020_0000u32 << x
    } else {
        (x as u32 - 1) << 23
    })
}

/// NVFP4's unsigned E4M3 sub-block scale, halved (GGML `ggml_ue4m3_to_fp32`).
fn ue4m3_to_f32(x: u8) -> f32 {
    if x == 0 || x == 0x7F {
        return 0.0;
    }
    let (exp, man) = (((x >> 3) & 0xF) as i32, (x & 7) as f32);
    let raw = if exp == 0 {
        man * 2f32.powi(-9)
    } else {
        (1.0 + man / 8.0) * 2f32.powi(exp - 7)
    };
    raw * 0.5
}

/// TQ1_0 (54 bytes, 256 ternary weights): base-3 packed qs[48] (5 per byte), qh[4] (4 per byte), d.
fn dequant_tq1_0_block(block: &[u8], out: &mut Vec<f32>) {
    const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];
    let (qs, qh, d) = (&block[0..48], &block[48..52], half_at(block, 52));
    let digit = |byte: u8, n: usize| -> f32 {
        let q = byte.wrapping_mul(POW3[n]);
        ((((q as u16) * 3) >> 8) as i32 - 1) as f32 * d
    };
    for j in (0..32).step_by(32) {
        for n in 0..5 {
            for m in 0..32 {
                out.push(digit(qs[j + m], n));
            }
        }
    }
    for j in (32..48).step_by(16) {
        for n in 0..5 {
            for m in 0..16 {
                out.push(digit(qs[j + m], n));
            }
        }
    }
    for n in 0..4 {
        for &b in qh {
            out.push(digit(b, n));
        }
    }
}

fn half_at(b: &[u8], at: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes(b[at..at + 2].try_into().unwrap())).to_f32()
}

/// `±1.0` for weight `j` of a sign byte (GGML `kmask_iq2xs[j] = 1 << j`).
fn sgn(signs: u8, j: usize) -> f32 {
    if signs & (1 << j) != 0 { -1.0 } else { 1.0 }
}

/// 8 weights from a 64-bit grid entry (unsigned bytes) scaled by `db` with sign byte `signs`.
fn push_grid8(out: &mut Vec<f32>, db: f32, grid: u64, signs: u8) {
    for (j, g) in grid.to_le_bytes().into_iter().enumerate() {
        out.push(db * g as f32 * sgn(signs, j));
    }
}

/// 2 x 4 weights from two 32-bit grid entries sharing one sign byte (IQ3).
fn push_grid4x2(out: &mut Vec<f32>, db: f32, g1: u32, g2: u32, signs: u8) {
    for (j, g) in g1.to_le_bytes().into_iter().enumerate() {
        out.push(db * g as f32 * sgn(signs, j));
    }
    for (j, g) in g2.to_le_bytes().into_iter().enumerate() {
        out.push(db * g as f32 * sgn(signs, j + 4));
    }
}

/// IQ2_XXS (66 bytes): d | u16 qs[32]; per 32 weights: 4 grid indices (bytes), 4x7-bit sign codes, 4-bit scale.
fn dequant_iq2_xxs_block(block: &[u8], out: &mut Vec<f32>) {
    let d = half_at(block, 0);
    let qs = &block[2..66];
    for ib32 in 0..8 {
        let idx = &qs[8 * ib32..8 * ib32 + 4];
        let aux1 = u32::from_le_bytes(qs[8 * ib32 + 4..8 * ib32 + 8].try_into().unwrap());
        let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
        for l in 0..4 {
            let signs = KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
            push_grid8(out, db, IQ2XXS_GRID[idx[l] as usize], signs);
        }
    }
}

/// IQ2_XS (74 bytes): d | u16 qs[32] (9-bit grid index + 7-bit signs) | scales[8] (two 4-bit scales each).
fn dequant_iq2_xs_block(block: &[u8], out: &mut Vec<f32>) {
    let d = half_at(block, 0);
    let scales = &block[66..74];
    for ib32 in 0..8 {
        let db = [
            d * (0.5 + (scales[ib32] & 0x0F) as f32) * 0.25,
            d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25,
        ];
        for l in 0..4 {
            let v = u16::from_le_bytes(
                block[2 + 2 * (4 * ib32 + l)..4 + 2 * (4 * ib32 + l)]
                    .try_into()
                    .unwrap(),
            );
            let signs = KSIGNS_IQ2XS[(v >> 9) as usize];
            push_grid8(out, db[l / 2], IQ2XS_GRID[(v & 511) as usize], signs);
        }
    }
}

/// IQ2_S (82 bytes): d | qs[64] (32 grid-index bytes, then 32 sign bytes) | qh[8] | scales[8].
fn dequant_iq2_s_block(block: &[u8], out: &mut Vec<f32>) {
    let d = half_at(block, 0);
    let qs = &block[2..66];
    let qh = &block[66..74];
    let scales = &block[74..82];
    for ib32 in 0..8 {
        let db = [
            d * (0.5 + (scales[ib32] & 0x0F) as f32) * 0.25,
            d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25,
        ];
        for l in 0..4 {
            let idx =
                qs[4 * ib32 + l] as usize | ((((qh[ib32] as u32) << (8 - 2 * l)) & 0x300) as usize);
            push_grid8(out, db[l / 2], IQ2S_GRID[idx], qs[32 + 4 * ib32 + l]);
        }
    }
}

/// IQ3_XXS (98 bytes): d | qs[96] (64 grid-index bytes, then per-32 u32 of 4x7-bit signs + 4-bit scale).
fn dequant_iq3_xxs_block(block: &[u8], out: &mut Vec<f32>) {
    let d = half_at(block, 0);
    let qs = &block[2..98];
    for ib32 in 0..8 {
        let aux = u32::from_le_bytes(qs[64 + 4 * ib32..68 + 4 * ib32].try_into().unwrap());
        let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
        for l in 0..4 {
            let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
            let g1 = IQ3XXS_GRID[qs[8 * ib32 + 2 * l] as usize];
            let g2 = IQ3XXS_GRID[qs[8 * ib32 + 2 * l + 1] as usize];
            push_grid4x2(out, db, g1, g2, signs);
        }
    }
}

/// IQ3_S (110 bytes): d | qs[64] | qh[8] | signs[32] | scales[4].
fn dequant_iq3_s_block(block: &[u8], out: &mut Vec<f32>) {
    let d = half_at(block, 0);
    let qs = &block[2..66];
    let qh = &block[66..74];
    let signs = &block[74..106];
    let scales = &block[106..110];
    for p in 0..4 {
        let dbs = [
            d * (1 + 2 * (scales[p] & 0x0F) as u32) as f32,
            d * (1 + 2 * (scales[p] >> 4) as u32) as f32,
        ];
        for (half, db) in dbs.into_iter().enumerate() {
            let h = qh[2 * p + half] as u32;
            for l in 0..4 {
                let q = 16 * p + 8 * half + 2 * l;
                let i1 = qs[q] as usize | (((h << (8 - 2 * l)) & 256) as usize);
                let i2 = qs[q + 1] as usize | (((h << (7 - 2 * l)) & 256) as usize);
                push_grid4x2(
                    out,
                    db,
                    IQ3S_GRID[i1],
                    IQ3S_GRID[i2],
                    signs[8 * p + 4 * half + l],
                );
            }
        }
    }
}

fn push_iq1(out: &mut Vec<f32>, dl: f32, idx: usize, delta: f32) {
    for g in IQ1S_GRID[idx].to_le_bytes() {
        out.push(dl * (g as i8 as f32 + delta));
    }
}

/// IQ1_S (50 bytes): d | qs[32] | u16 qh[8] (3 high index bits per group, 3-bit scale, delta sign).
fn dequant_iq1_s_block(block: &[u8], out: &mut Vec<f32>) {
    let d = half_at(block, 0);
    for ib in 0..8 {
        let qh = u16::from_le_bytes(block[34 + 2 * ib..36 + 2 * ib].try_into().unwrap()) as usize;
        let dl = d * (2 * ((qh >> 12) & 7) + 1) as f32;
        let delta = if qh & 0x8000 != 0 {
            -IQ1S_DELTA
        } else {
            IQ1S_DELTA
        };
        for l in 0..4 {
            push_iq1(
                out,
                dl,
                block[2 + 4 * ib + l] as usize | (((qh >> (3 * l)) & 7) << 8),
                delta,
            );
        }
    }
}

/// IQ1_M (56 bytes): qs[32] | qh[16] | scales[8]; the f16 super-scale is spread over the top nibbles of
/// the four u16 scale words.
fn dequant_iq1_m_block(block: &[u8], out: &mut Vec<f32>) {
    let sc: [u16; 4] = std::array::from_fn(|i| {
        u16::from_le_bytes(block[48 + 2 * i..50 + 2 * i].try_into().unwrap())
    });
    let scale =
        (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    let d = f16::from_bits(scale).to_f32();
    for ib in 0..8 {
        let dl1 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2))) & 7) + 1) as f32;
        let dl2 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 7) + 1) as f32;
        let qs = &block[4 * ib..4 * ib + 4];
        let (qh0, qh1) = (block[32 + 2 * ib] as usize, block[33 + 2 * ib] as usize);
        let idx = [
            qs[0] as usize | ((qh0 << 8) & 0x700),
            qs[1] as usize | ((qh0 << 4) & 0x700),
            qs[2] as usize | ((qh1 << 8) & 0x700),
            qs[3] as usize | ((qh1 << 4) & 0x700),
        ];
        let delta = |qh: usize, bit: usize| {
            if qh & bit != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            }
        };
        let deltas = [
            delta(qh0, 0x08),
            delta(qh0, 0x80),
            delta(qh1, 0x08),
            delta(qh1, 0x80),
        ];
        for l in 0..4 {
            push_iq1(out, if l < 2 { dl1 } else { dl2 }, idx[l], deltas[l]);
        }
    }
}

/// The non-linear 4-bit codebook shared by `IQ4_NL` and `IQ4_XS` (GGML `kvalues_iq4nl`).
const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// 32 weights from 16 bytes: low nibbles are x[0..16], high nibbles x[16..32] (split-half).
fn dequant_iq4_nibbles(scale: f32, qs: &[u8], out: &mut Vec<f32>) {
    out.extend(
        qs.iter()
            .map(|&b| scale * KVALUES_IQ4NL[(b & 0x0F) as usize] as f32),
    );
    out.extend(
        qs.iter()
            .map(|&b| scale * KVALUES_IQ4NL[(b >> 4) as usize] as f32),
    );
}

/// Dequantizes one 136-byte IQ4_XS super-block: d | scales_h (u16) | scales_l[4] | qs[128].
fn dequant_iq4_xs_block(block: &[u8], out: &mut Vec<f32>) {
    let d = f16::from_bits(u16::from_le_bytes(block[0..2].try_into().unwrap())).to_f32();
    let scales_h = u16::from_le_bytes(block[2..4].try_into().unwrap());
    let scales_l = &block[4..8];
    let qs = &block[8..136];
    for ib in 0..8 {
        let low = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F;
        let high = ((scales_h >> (2 * ib)) & 3) as u8;
        let ls = (low | (high << 4)) as i32 - 32;
        dequant_iq4_nibbles(d * ls as f32, &qs[ib * 16..ib * 16 + 16], out);
    }
}

/// Dequantizes one 84-byte Q2_K super-block (256 weights): scales[16] | qs[64] | d | dmin.
fn dequant_q2_k_block(block: &[u8], out: &mut Vec<f32>) {
    let scales = &block[0..16];
    let qs = &block[16..80];
    let d = f16::from_bits(u16::from_le_bytes(block[80..82].try_into().unwrap())).to_f32();
    let dmin = f16::from_bits(u16::from_le_bytes(block[82..84].try_into().unwrap())).to_f32();

    let mut is = 0;
    for n in 0..2 {
        let q = &qs[n * 32..n * 32 + 32];
        for j in 0..4 {
            let shift = 2 * j;
            for half in 0..2 {
                let sc = scales[is];
                is += 1;
                let (dl, ml) = (d * (sc & 0x0F) as f32, dmin * (sc >> 4) as f32);
                for &b in &q[half * 16..half * 16 + 16] {
                    out.push(dl * ((b >> shift) & 3) as f32 - ml);
                }
            }
        }
    }
}

/// Dequantizes one 110-byte Q3_K super-block (256 weights): hmask[32] | qs[64] | scales[12] | d.
fn dequant_q3_k_block(block: &[u8], out: &mut Vec<f32>) {
    let hmask = &block[0..32];
    let qs = &block[32..96];
    let d_all = f16::from_bits(u16::from_le_bytes(block[108..110].try_into().unwrap())).to_f32();

    // Unpack sixteen 6-bit scales: low nibbles from bytes 0..8, top two bits from bytes 8..12.
    let raw = &block[96..108];
    let mut scales = [0i32; 16];
    for (i, s) in scales.iter_mut().enumerate() {
        let low = if i < 8 {
            raw[i] & 0x0F
        } else {
            raw[i - 8] >> 4
        };
        let high = (raw[8 + i % 4] >> (2 * (i / 4))) & 3;
        *s = (low | (high << 4)) as i32 - 32;
    }

    let mut is = 0;
    let mut m = 1u8;
    for n in 0..2 {
        let q = &qs[n * 32..n * 32 + 32];
        for j in 0..4 {
            let shift = 2 * j;
            for half in 0..2 {
                let dl = d_all * scales[is] as f32;
                is += 1;
                for l in 0..16 {
                    let idx = half * 16 + l;
                    let low2 = ((q[idx] >> shift) & 3) as i32;
                    let v = low2 - if hmask[idx] & m != 0 { 0 } else { 4 };
                    out.push(dl * v as f32);
                }
            }
            m <<= 1;
        }
    }
}

/// Dequantizes one 144-byte Q4_K super-block (256 weights) into `out`.
fn dequant_q4_k_block(block: &[u8], out: &mut Vec<f32>) {
    let d = f16::from_bits(u16::from_le_bytes(block[0..2].try_into().unwrap())).to_f32();
    let dmin = f16::from_bits(u16::from_le_bytes(block[2..4].try_into().unwrap())).to_f32();
    let scales = &block[4..16];
    let qs = &block[16..144];

    let mut is = 0;
    for j in (0..256).step_by(64) {
        let (sc1, m1) = get_scale_min_k4(is, scales);
        let (sc2, m2) = get_scale_min_k4(is + 1, scales);
        let (d1, mn1) = (d * sc1 as f32, dmin * m1 as f32);
        let (d2, mn2) = (d * sc2 as f32, dmin * m2 as f32);
        let q = &qs[j / 2..j / 2 + 32];
        for &b in q {
            out.push(d1 * (b & 0x0F) as f32 - mn1);
        }
        for &b in q {
            out.push(d2 * (b >> 4) as f32 - mn2);
        }
        is += 2;
    }
}

/// Dequantizes one 176-byte Q5_K super-block (256 weights) into `out`.
fn dequant_q5_k_block(block: &[u8], out: &mut Vec<f32>) {
    let d = f16::from_bits(u16::from_le_bytes(block[0..2].try_into().unwrap())).to_f32();
    let dmin = f16::from_bits(u16::from_le_bytes(block[2..4].try_into().unwrap())).to_f32();
    let scales = &block[4..16];
    let qh = &block[16..48];
    let qs = &block[48..176];

    let mut is = 0;
    let (mut u1, mut u2) = (1u8, 2u8);
    for j in (0..256).step_by(64) {
        let (sc1, m1) = get_scale_min_k4(is, scales);
        let (sc2, m2) = get_scale_min_k4(is + 1, scales);
        let (d1, mn1) = (d * sc1 as f32, dmin * m1 as f32);
        let (d2, mn2) = (d * sc2 as f32, dmin * m2 as f32);
        let ql = &qs[j / 2..j / 2 + 32];
        for (l, &b) in ql.iter().enumerate() {
            let hi_bit = if qh[l] & u1 != 0 { 16 } else { 0 };
            out.push(d1 * ((b & 0x0F) as i32 + hi_bit) as f32 - mn1);
        }
        for (l, &b) in ql.iter().enumerate() {
            let hi_bit = if qh[l] & u2 != 0 { 16 } else { 0 };
            out.push(d2 * ((b >> 4) as i32 + hi_bit) as f32 - mn2);
        }
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

/// Dequantizes one 210-byte Q6_K super-block (256 weights) into `out`.
fn dequant_q6_k_block(block: &[u8], out: &mut Vec<f32>) {
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let d = f16::from_bits(u16::from_le_bytes(block[208..210].try_into().unwrap())).to_f32();

    for n in (0..256).step_by(128) {
        let ql = &ql[n / 2..n / 2 + 64];
        let qh = &qh[n / 4..n / 4 + 32];
        let sc = &scales[n / 16..n / 16 + 8];
        let mut vals = [0f32; 128];
        for l in 0..32 {
            let is = l / 16;
            let q1 = (((ql[l] & 0x0F) | ((qh[l] & 3) << 4)) as i32) - 32;
            let q2 = (((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i32) - 32;
            let q3 = (((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32) - 32;
            let q4 = (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32) - 32;
            vals[l] = d * (sc[is] as i8) as f32 * q1 as f32;
            vals[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
            vals[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
            vals[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
        }
        out.extend_from_slice(&vals);
    }
}

/// Returns the next `len` bytes and advances `cursor`, erroring (never panicking or overflowing)
/// when the file is shorter than its own headers claim.
fn take<'a>(buf: &'a [u8], cursor: &mut usize, len: usize) -> Result<&'a [u8]> {
    let end = cursor
        .checked_add(len)
        .filter(|&e| e <= buf.len())
        .ok_or_else(|| Error::Msg("Unexpected EOF in GGUF data".into()))?;
    let out = &buf[*cursor..end];
    *cursor = end;
    Ok(out)
}

fn read_u32_le(buf: &[u8], cursor: &mut usize) -> Result<u32> {
    if *cursor + 4 > buf.len() {
        return Err(Error::Msg("Unexpected EOF reading u32".into()));
    }
    let val = u32::from_le_bytes(buf[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    Ok(val)
}

fn read_u64_le(buf: &[u8], cursor: &mut usize) -> Result<u64> {
    if *cursor + 8 > buf.len() {
        return Err(Error::Msg("Unexpected EOF reading u64".into()));
    }
    let val = u64::from_le_bytes(buf[*cursor..*cursor + 8].try_into().unwrap());
    *cursor += 8;
    Ok(val)
}

fn read_string(buf: &[u8], cursor: &mut usize) -> Result<String> {
    let len = read_u64_le(buf, cursor)? as usize;
    let s = String::from_utf8(take(buf, cursor, len)?.to_vec())
        .map_err(|e| Error::Msg(format!("Invalid UTF-8 in GGUF string: {e}")))?;
    Ok(s)
}

fn write_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn read_value(buf: &[u8], cursor: &mut usize, type_id: u32) -> Result<GGUFValue> {
    match type_id {
        0 => {
            let v = buf
                .get(*cursor)
                .copied()
                .ok_or_else(|| Error::Msg("EOF".into()))?;
            *cursor += 1;
            Ok(GGUFValue::Uint8(v))
        }
        1 => {
            let v = buf
                .get(*cursor)
                .copied()
                .ok_or_else(|| Error::Msg("EOF".into()))? as i8;
            *cursor += 1;
            Ok(GGUFValue::Int8(v))
        }
        2 => {
            let v = u16::from_le_bytes(take(buf, cursor, 2)?.try_into().unwrap());
            Ok(GGUFValue::Uint16(v))
        }
        3 => {
            let v = i16::from_le_bytes(take(buf, cursor, 2)?.try_into().unwrap());
            Ok(GGUFValue::Int16(v))
        }
        4 => Ok(GGUFValue::Uint32(read_u32_le(buf, cursor)?)),
        5 => {
            let v = i32::from_le_bytes(take(buf, cursor, 4)?.try_into().unwrap());
            Ok(GGUFValue::Int32(v))
        }
        6 => {
            let v = f32::from_le_bytes(take(buf, cursor, 4)?.try_into().unwrap());
            Ok(GGUFValue::Float32(v))
        }
        7 => {
            let v = buf
                .get(*cursor)
                .copied()
                .ok_or_else(|| Error::Msg("EOF".into()))?
                != 0;
            *cursor += 1;
            Ok(GGUFValue::Bool(v))
        }
        8 => Ok(GGUFValue::String(read_string(buf, cursor)?)),
        9 => {
            let item_type = read_u32_le(buf, cursor)?;
            // GGUF arrays are flat and homogeneous (the writer refuses to emit
            // nested arrays); accepting item_type 9 here would recurse with no
            // depth limit, so a small crafted file could stack-overflow the
            // process. Reject nesting outright instead.
            if item_type == 9 {
                return Err(Error::Msg("GGUF arrays cannot be nested".into()));
            }
            let item_count = read_u64_le(buf, cursor)? as usize;
            // Every element needs at least one byte, so a count larger than the remaining buffer
            // is corrupt (and must not drive a multi-terabyte `with_capacity`).
            if item_count > buf.len().saturating_sub(*cursor) {
                return Err(Error::Msg(format!(
                    "GGUF array of {item_count} items exceeds file size"
                )));
            }
            let mut arr: Vec<GGUFValue> = crate::layout::try_with_capacity(item_count)?;
            for _ in 0..item_count {
                arr.push(read_value(buf, cursor, item_type)?);
            }
            Ok(GGUFValue::Array(arr))
        }
        10 => Ok(GGUFValue::Uint64(read_u64_le(buf, cursor)?)),
        11 => {
            let v = i64::from_le_bytes(take(buf, cursor, 8)?.try_into().unwrap());
            Ok(GGUFValue::Int64(v))
        }
        12 => {
            let v = f64::from_le_bytes(take(buf, cursor, 8)?.try_into().unwrap());
            Ok(GGUFValue::Float64(v))
        }
        other => Err(Error::Msg(format!("Unknown GGUF value type id: {other}"))),
    }
}

fn write_value(buf: &mut Vec<u8>, val: &GGUFValue) -> Result<()> {
    match val {
        GGUFValue::Uint8(v) => buf.push(*v),
        GGUFValue::Int8(v) => buf.push(*v as u8),
        GGUFValue::Uint16(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GGUFValue::Int16(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GGUFValue::Uint32(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GGUFValue::Int32(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GGUFValue::Float32(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GGUFValue::Bool(v) => buf.push(if *v { 1 } else { 0 }),
        GGUFValue::String(s) => write_string(buf, s),
        GGUFValue::Array(arr) => {
            if arr.is_empty() {
                buf.extend_from_slice(&8u32.to_le_bytes()); // string default
                buf.extend_from_slice(&0u64.to_le_bytes());
            } else {
                // GGUF arrays are homogeneous and non-nested: refuse to write
                // corrupt bytes for mixed-type or nested arrays.
                let item_type = arr[0].type_id();
                if matches!(arr[0], GGUFValue::Array(_)) {
                    return Err(Error::Msg("GGUF arrays cannot be nested".into()));
                }
                if arr.iter().any(|item| item.type_id() != item_type) {
                    return Err(Error::Msg(
                        "GGUF arrays must be homogeneous; refusing to write mixed-type array"
                            .into(),
                    ));
                }
                buf.extend_from_slice(&item_type.to_le_bytes());
                buf.extend_from_slice(&(arr.len() as u64).to_le_bytes());
                for item in arr {
                    write_value(buf, item)?;
                }
            }
        }
        GGUFValue::Uint64(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GGUFValue::Int64(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GGUFValue::Float64(v) => buf.extend_from_slice(&v.to_le_bytes()),
    }
    Ok(())
}

#[cfg(test)]
mod quant_layout_tests {
    use super::*;

    /// Builds a minimal one-tensor GGUF buffer (no metadata) wrapping a single hand-built block,
    /// bypassing `serialize_gguf` so decode is checked against independently-assembled bytes
    /// rather than against this crate's own encoder. `shape` is the row-major shape the parsed
    /// `GGUFTensorInfo` should end up with (GGUF stores dims column-major, so they're written reversed).
    fn single_tensor_gguf_shaped(ggml_type: GGMLType, shape: &[usize], block: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count

        write_string(&mut buf, "w");
        buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for &d in shape.iter().rev() {
            buf.extend_from_slice(&(d as u64).to_le_bytes());
        }
        buf.extend_from_slice(&(ggml_type as u32).to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset

        let rem = buf.len() % GGUF_DEFAULT_ALIGNMENT;
        if rem != 0 {
            buf.resize(buf.len() + (GGUF_DEFAULT_ALIGNMENT - rem), 0);
        }
        buf.extend_from_slice(block);
        buf
    }

    fn single_tensor_gguf(ggml_type: GGMLType, numel: usize, block: &[u8]) -> Vec<u8> {
        single_tensor_gguf_shaped(ggml_type, &[numel], block)
    }

    fn dequant(ggml_type: GGMLType, numel: usize, block: &[u8]) -> Vec<f32> {
        let bytes = single_tensor_gguf(ggml_type, numel, block);
        let file = GGUFFile::from_bytes(&bytes).unwrap();
        file.dequantize_tensor("w")
            .unwrap()
            .to_vec::<f32>()
            .unwrap()
    }

    /// Q4_0 packs byte j as x[j] | x[j+16] << 4 (split-half), not adjacent pairs (x[2j] | x[2j+1] << 4).
    /// This is the real GGML on-disk layout; getting it wrong silently scrambles every real-world
    /// Q4_0 GGUF file while still round-tripping against this crate's own (symmetrically wrong) encoder.
    #[test]
    fn q4_0_uses_split_half_nibble_layout() {
        let d = 2.0f32;
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        for j in 0..16u8 {
            let lo = j; // -> j - 8
            let hi = 15 - j; // -> (15-j) - 8
            block[2 + j as usize] = (lo & 0x0F) | ((hi & 0x0F) << 4);
        }
        let out = dequant(GGMLType::Q4_0, 32, &block);
        assert_eq!(out.len(), 32);
        for j in 0..16usize {
            assert_eq!(out[j], (j as f32 - 8.0) * d, "lo half index {j}");
            assert_eq!(
                out[j + 16],
                ((15 - j) as f32 - 8.0) * d,
                "hi half index {j}"
            );
        }
    }

    /// Q5_1 (d, m, qh, qs) must also split-half, with the 5th bit folded in from `qh` and an
    /// additive `min`, not a multiplicative-only scale.
    #[test]
    fn q5_1_layout_matches_reference() {
        let (d, m) = (0.5f32, 1.5f32);
        let mut block = vec![0u8; 24];
        block[0..2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        block[2..4].copy_from_slice(&f16::from_f32(m).to_bits().to_le_bytes());
        // qh all zero -> no 5th bit set; qs[0] low nibble = 3, high nibble = 9.
        block[8] = 3 | (9 << 4);
        let out = dequant(GGMLType::Q5_1, 32, &block);
        assert_eq!(out[0], 3.0 * d + m);
        assert_eq!(out[16], 9.0 * d + m);
        assert_eq!(out[1], 0.0 * d + m); // untouched slots still decode to just `m`
    }

    /// K-quant super-blocks (Q4_K/Q5_K/Q6_K) must read `d`/`dmin`/`scales` from the right byte
    /// ranges: an all-zero `scales` (and, for Q6_K, all-zero per-sub-block int8 scale) collapses
    /// every output to exactly zero regardless of the quant nibbles or `d`, which pins down the
    /// super-block's field offsets without having to hand-reproduce GGML's full bit math.
    #[test]
    fn k_quant_zero_scale_zeroes_whole_superblock() {
        for (ty, len) in [
            (GGMLType::Q4_K, 144),
            (GGMLType::Q5_K, 176),
            (GGMLType::Q6_K, 210),
        ] {
            let mut block = vec![0xFFu8; len]; // nonzero quant bits everywhere
            match ty {
                GGMLType::Q4_K | GGMLType::Q5_K => {
                    block[0..2].copy_from_slice(&f16::from_f32(3.0).to_bits().to_le_bytes());
                    block[2..4].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
                    for b in &mut block[4..16] {
                        *b = 0; // scales[12] = 0 -> every (scale, min) pair decodes to (0, 0)
                    }
                }
                GGMLType::Q6_K => {
                    for b in &mut block[192..208] {
                        *b = 0; // int8 scales[16] = 0
                    }
                    block[208..210].copy_from_slice(&f16::from_f32(5.0).to_bits().to_le_bytes());
                }
                _ => unreachable!(),
            }
            let out = dequant(ty, 256, &block);
            assert_eq!(out.len(), 256, "{ty:?} output length");
            assert!(
                out.iter().all(|&v| v == 0.0),
                "{ty:?} expected an all-zero super-block, got {out:?}"
            );
        }
    }

    /// A concrete nonzero Q4_K case: sub-block 0's (scale, min) = (1, 0) and one explicit nibble,
    /// so `get_scale_min_k4` and the nibble unpack are both checked against real numbers.
    #[test]
    fn q4_k_nonzero_subblock_matches_hand_computed_value() {
        let d = 1.0f32;
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        block[2..4].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes()); // dmin
        block[4] = 1; // scales[0] & 63 = 1 -> sub-block 0 scale = d*1 = 1; min(=scales[4]&63=0) = 0
        block[16] = 5; // qs[0] low nibble = 5
        let out = dequant(GGMLType::Q4_K, 256, &block);
        assert_eq!(out[0], 5.0); // d1 * 5 - mn1 = 1*1*5 - 0
    }

    /// Malformed input must be an `Err`, never a panic, overflow or giant allocation: every
    /// truncation of a valid file, plus headers claiming absurd counts / sizes.
    #[test]
    fn malformed_files_are_errors_not_panics() {
        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".to_string(),
            GGUFValue::String("llama".into()),
        );
        meta.insert(
            "a".to_string(),
            GGUFValue::Array(vec![GGUFValue::Uint16(7), GGUFValue::Uint16(9)]),
        );
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let good = serialize_gguf(&meta, &[("w", GGMLType::F32, &t)], 32).unwrap();
        for cut in 0..good.len() {
            // Must not panic; most prefixes are errors (a tensor-data cut is caught at read time).
            if let Ok(f) = GGUFFile::from_bytes(&good[..cut]) {
                let _ = f.dequantize_tensor("w");
            }
        }
        // Header claiming 2^62 metadata entries / tensors.
        let mut huge = good.clone();
        huge[8..16].copy_from_slice(&(1u64 << 62).to_le_bytes());
        assert!(GGUFFile::from_bytes(&huge).is_err());
        let mut huge = good.clone();
        huge[16..24].copy_from_slice(&(1u64 << 62).to_le_bytes());
        assert!(GGUFFile::from_bytes(&huge).is_err());
        // Tensor whose dims multiply past usize.
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        write_string(&mut buf, "w");
        buf.extend_from_slice(&3u32.to_le_bytes());
        for _ in 0..3 {
            buf.extend_from_slice(&(1u64 << 40).to_le_bytes());
        }
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        assert!(GGUFFile::from_bytes(&buf).is_err());
    }

    /// Bit-exact check against llama.cpp itself: `testdata/ggml_reference_blocks.txt` holds random
    /// blocks of every supported type and the output of GGML's own `dequantize_row_*` for them
    /// (an FNV-1a hash of all f32 bit patterns plus the first eight values for diagnostics).
    /// Regenerate with `tools/ggml_reference/harness.c` (see the README there).
    #[test]
    fn matches_ggml_reference_dequantizers_bit_for_bit() {
        let mut seen = std::collections::BTreeSet::new();
        for line in include_str!("testdata/ggml_reference_blocks.txt").lines() {
            let mut f = line.split(' ');
            let id: u32 = f.next().unwrap().parse().unwrap();
            let block: Vec<u8> = f
                .next()
                .unwrap()
                .as_bytes()
                .chunks(2)
                .map(|h| u8::from_str_radix(std::str::from_utf8(h).unwrap(), 16).unwrap())
                .collect();
            let first8: Vec<u32> = f
                .next()
                .unwrap()
                .split(',')
                .map(|h| u32::from_str_radix(h, 16).unwrap())
                .collect();
            let want_hash = u64::from_str_radix(f.next().unwrap(), 16).unwrap();

            let ty = GGMLType::from_u32(id).unwrap();
            assert_eq!(ty.block_size_bytes(), block.len(), "{ty:?} block size");
            let out = dequant(ty, ty.block_size(), &block);
            let mut h = 0xcbf29ce484222325u64;
            for v in &out {
                for b in v.to_bits().to_le_bytes() {
                    h = (h ^ b as u64).wrapping_mul(0x100000001b3);
                }
            }
            let got8: Vec<u32> = out.iter().take(8).map(|v| v.to_bits()).collect();
            assert_eq!(
                h, want_hash,
                "{ty:?} differs from GGML: first8 got {got8:08x?} want {first8:08x?}"
            );
            seen.insert(id);
        }
        assert_eq!(
            seen.len(),
            25,
            "fixture should cover every block-quantized type"
        );
    }

    /// Real GGML numbers BF16 as 30 (16 is IQ2_XXS); this crate used 16, which made genuine BF16
    /// files unreadable and IQ2_XXS files decode as BF16 garbage.
    #[test]
    fn ggml_type_ids_match_ggml() {
        for (ty, id) in [
            (GGMLType::IQ2_XXS, 16),
            (GGMLType::BF16, 30),
            (GGMLType::IQ4_NL, 20),
            (GGMLType::IQ1_M, 29),
        ] {
            assert_eq!(ty as u32, id);
            assert_eq!(GGMLType::from_u32(id).unwrap(), ty);
        }
    }

    /// Q2_K / Q3_K: field offsets pinned by zero-scale blocks, values by one hand-computed weight.
    #[test]
    fn q2_k_and_q3_k_match_reference_layout() {
        // Q2_K: scales[0] = 0x21 -> dl = d*1, ml = dmin*2; qs[0] low 2 bits = 3 -> 1*3 - 2 = 1.
        let mut b = vec![0xFFu8; 84];
        b[0..16].fill(0);
        b[0] = 0x21;
        b[16] = 0b11; // qs[0]
        b[80..82].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        b[82..84].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        let out = dequant(GGMLType::Q2_K, 256, &b);
        assert_eq!(out.len(), 256);
        assert_eq!(out[0], 3.0 - 2.0);
        // Scale 0 covers only the first 16 weights; qs bytes are 0xFF elsewhere but scale is 0.
        assert!(out[16..].iter().all(|&v| v == 0.0));

        // Q3_K: every 6-bit scale = 32 (-> 0) except scale 0 = 33 (-> 1).
        let mut b = vec![0u8; 110];
        b[96] = 1; // scale 0 low nibble
        for byte in &mut b[104..108] {
            *byte = 0xAA; // top two bits = 2 for every scale => 32
        }
        b[108..110].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        b[32] = 0b11; // qs[0] low 2 bits = 3
        b[0] = 0b1; // hmask[0] bit 0 set => no -4 offset
        let out = dequant(GGMLType::Q3_K, 256, &b);
        assert_eq!(out.len(), 256);
        assert_eq!(out[0], 3.0);
        b[0] = 0; // high bit clear => 3 - 4
        assert_eq!(dequant(GGMLType::Q3_K, 256, &b)[0], -1.0);
        assert!(out[16..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn iq4_nl_and_iq4_xs_use_the_nonlinear_codebook() {
        // IQ4_NL: d=1; byte 0 = low nibble 0 (-127), high nibble 8 (+1).
        let mut b = vec![0u8; 18];
        b[0..2].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        b[2] = 8 << 4;
        let out = dequant(GGMLType::IQ4_NL, 32, &b);
        assert_eq!((out[0], out[16], out[1]), (-127.0, 1.0, -127.0));

        // IQ4_XS: every 6-bit scale = 32 (zero) except block 0 = 33 (x1).
        let mut b = vec![0u8; 136];
        b[0..2].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        b[2..4].copy_from_slice(&0xAAAAu16.to_le_bytes()); // high 2 bits = 2 for all 8 blocks
        b[4] = 1; // scales_l[0] low nibble -> block 0 low bits = 1
        b[8] = (8 << 4) | 15; // qs[0]: low nibble 15 (113), high nibble 8 (1)
        let out = dequant(GGMLType::IQ4_XS, 256, &b);
        assert_eq!(out.len(), 256);
        assert_eq!((out[0], out[16]), (113.0, 1.0));
        assert!(out[32..].iter().all(|&v| v == 0.0));
    }

    /// Plain (non-block) F64 and integer tensors decode to the same numbers as f32.
    #[test]
    fn f64_and_integer_tensors_decode() {
        let f64s: Vec<u8> = [1.5f64, -2.25]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(dequant(GGMLType::F64, 2, &f64s), vec![1.5, -2.25]);
        assert_eq!(
            dequant(GGMLType::I8, 3, &[1u8, 255, 3]),
            vec![1.0, -1.0, 3.0]
        );
        let i16s: Vec<u8> = [-300i16, 7].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(dequant(GGMLType::I16, 2, &i16s), vec![-300.0, 7.0]);
        let i32s: Vec<u8> = [-70000i32].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(dequant(GGMLType::I32, 1, &i32s), vec![-70000.0]);
        let i64s: Vec<u8> = [123456789012i64]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(
            dequant(GGMLType::I64, 1, &i64s),
            vec![123456789012i64 as f32]
        );
    }

    /// `fused_gemv`'s Q4_0 fast path unpacks nibbles with its own independent loop (not shared
    /// with `dequantize_tensor`), and it previously disagreed on nibble order (adjacent-pair vs.
    /// split-half) after only the decoder was fixed. Cross-check the fast path against the
    /// generic dequantize-then-matmul path, which is the invariant that broke.
    #[test]
    fn q4_0_fused_gemv_matches_dequantize_then_matmul() {
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        for j in 0..16u8 {
            block[2 + j as usize] = (j & 0x0F) | (((15 - j) & 0x0F) << 4);
        }
        let bytes = single_tensor_gguf_shaped(GGMLType::Q4_0, &[1, 32], &block);
        let file = GGUFFile::from_bytes(&bytes).unwrap();

        let x_vals: Vec<f32> = (0..32).map(|i| i as f32 * 0.1 - 1.0).collect();
        let x = Tensor::from_slice(&x_vals, &[32]).unwrap();

        let y_fast = file.fused_gemv("w", &x).unwrap().to_vec::<f32>().unwrap();

        let w = file.dequantize_tensor("w").unwrap();
        let y_ref = x
            .reshape(&[1, 32])
            .unwrap()
            .matmul(&w.t().unwrap())
            .unwrap()
            .to_vec::<f32>()
            .unwrap();

        assert!(
            (y_fast[0] - y_ref[0]).abs() < 1e-4,
            "fast={y_fast:?} ref={y_ref:?}"
        );
    }
}
