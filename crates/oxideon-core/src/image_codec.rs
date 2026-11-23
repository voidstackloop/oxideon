//! Pure-Rust PNG and JPEG (baseline and progressive) image decoders returning `[C, H, W]` [`Tensor`]s.
//!
//! Zero external image crate dependencies: decodes PNG (including Adam7 interlacing; gray / RGB / palette / alpha, 1-16 bit, all 5 scanline filters,
//! stored + fixed Huffman + dynamic Huffman DEFLATE) and baseline DCT JPEG (`SOF0`) directly
//! in `oxideon-core`.

use std::path::Path;

use crate::backend::{CpuStorage, Storage};
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// RFC 1951 DEFLATE / RFC 1950 zlib decompressor (stored + fixed + dynamic)
// ---------------------------------------------------------------------------

struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_buf: u32,
    bit_cnt: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_buf: 0,
            bit_cnt: 0,
        }
    }

    fn ensure(&mut self, n: u8) -> Result<()> {
        while self.bit_cnt < n {
            if self.byte_pos >= self.data.len() {
                return Err(Error::Msg("deflate: unexpected EOF".into()));
            }
            self.bit_buf |= (self.data[self.byte_pos] as u32) << self.bit_cnt;
            self.byte_pos += 1;
            self.bit_cnt += 8;
        }
        Ok(())
    }

    fn read_bits(&mut self, n: u8) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        self.ensure(n)?;
        let val = self.bit_buf & ((1u32 << n) - 1);
        self.bit_buf >>= n;
        self.bit_cnt -= n;
        Ok(val)
    }

    fn align_byte(&mut self) {
        let drop = self.bit_cnt & 7;
        self.bit_buf >>= drop;
        self.bit_cnt -= drop;
    }

    fn read_u16_aligned(&mut self) -> Result<u16> {
        self.align_byte();
        let lo = self.read_bits(8)? as u16;
        let hi = self.read_bits(8)? as u16;
        Ok(lo | (hi << 8))
    }
}

struct HuffTable {
    counts: [u16; 16],
    symbols: Vec<u16>,
}

impl HuffTable {
    fn from_lengths(lengths: &[u8]) -> Result<Self> {
        let mut counts = [0u16; 16];
        for &l in lengths {
            if l > 15 {
                return Err(Error::Msg("deflate: code length > 15".into()));
            }
            if l > 0 {
                counts[l as usize] += 1;
            }
        }
        let mut offsets = [0u16; 16];
        let mut sum = 0u16;
        for bits in 1..=15 {
            offsets[bits] = sum;
            sum = sum.saturating_add(counts[bits]);
        }
        let mut symbols = vec![0u16; sum as usize];
        for (sym, &len) in lengths.iter().enumerate() {
            if len > 0 {
                let idx = offsets[len as usize] as usize;
                offsets[len as usize] += 1;
                symbols[idx] = sym as u16;
            }
        }
        Ok(Self { counts, symbols })
    }

    fn decode(&self, br: &mut BitReader<'_>) -> Result<u16> {
        let mut code = 0i32;
        let mut first = 0i32;
        let mut index = 0i32;
        for len in 1..=15usize {
            code |= br.read_bits(1)? as i32;
            let count = self.counts[len] as i32;
            if code - count < first {
                return Ok(self.symbols[(index + (code - first)) as usize]);
            }
            index += count;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        Err(Error::Msg("deflate: invalid huffman symbol".into()))
    }
}

fn inflate_zlib(zlib: &[u8]) -> Result<Vec<u8>> {
    if zlib.len() < 6 {
        return Err(Error::Msg("zlib: stream too short".into()));
    }
    let cmf = zlib[0];
    let flg = zlib[1];
    if (cmf & 0x0F) != 8 || (((cmf as u16) << 8 | flg as u16) % 31 != 0) {
        return Err(Error::Msg("zlib: invalid header".into()));
    }
    let start = if (flg & 0x20) != 0 { 6 } else { 2 };
    let mut br = BitReader::new(&zlib[start..]);
    let mut out = Vec::new();

    const LEN_BASE: [u16; 29] = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99,
        115, 131, 163, 195, 227, 258,
    ];
    const LEN_EXTRA: [u8; 29] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
    ];
    const DIST_BASE: [u16; 30] = [
        1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025,
        1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
    ];
    const DIST_EXTRA: [u8; 30] = [
        0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12,
        12, 13, 13,
    ];
    const CL_ORDER: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];

    loop {
        let bfinal = br.read_bits(1)?;
        let btype = br.read_bits(2)?;
        if btype == 0 {
            let len = br.read_u16_aligned()? as usize;
            let nlen = br.read_u16_aligned()?;
            if (len as u16) != !nlen {
                return Err(Error::Msg("deflate: stored block length mismatch".into()));
            }
            for _ in 0..len {
                out.push(br.read_bits(8)? as u8);
            }
        } else if btype == 1 || btype == 2 {
            let (lit_tbl, dist_tbl) = if btype == 1 {
                let mut ll = vec![0u8; 288];
                ll[0..144].fill(8);
                ll[144..256].fill(9);
                ll[256..280].fill(7);
                ll[280..288].fill(8);
                let dl = vec![5u8; 32];
                (HuffTable::from_lengths(&ll)?, HuffTable::from_lengths(&dl)?)
            } else {
                let hlit = br.read_bits(5)? as usize + 257;
                let hdist = br.read_bits(5)? as usize + 1;
                let hclen = br.read_bits(4)? as usize + 4;
                let mut cl_lens = [0u8; 19];
                for i in 0..hclen {
                    cl_lens[CL_ORDER[i]] = br.read_bits(3)? as u8;
                }
                let cl_tbl = HuffTable::from_lengths(&cl_lens)?;
                let total = hlit + hdist;
                let mut lens = Vec::with_capacity(total);
                while lens.len() < total {
                    let sym = cl_tbl.decode(&mut br)?;
                    match sym {
                        0..=15 => lens.push(sym as u8),
                        16 => {
                            let prev = *lens
                                .last()
                                .ok_or_else(|| Error::Msg("deflate: repeat with no prev".into()))?;
                            let rep = br.read_bits(2)? as usize + 3;
                            lens.resize(lens.len() + rep, prev);
                        }
                        17 => {
                            let rep = br.read_bits(3)? as usize + 3;
                            lens.resize(lens.len() + rep, 0);
                        }
                        18 => {
                            let rep = br.read_bits(7)? as usize + 11;
                            lens.resize(lens.len() + rep, 0);
                        }
                        _ => return Err(Error::Msg("deflate: bad cl symbol".into())),
                    }
                }
                (
                    HuffTable::from_lengths(&lens[..hlit])?,
                    HuffTable::from_lengths(&lens[hlit..])?,
                )
            };

            loop {
                let sym = lit_tbl.decode(&mut br)? as usize;
                if sym < 256 {
                    out.push(sym as u8);
                } else if sym == 256 {
                    break;
                } else {
                    let idx = sym - 257;
                    if idx >= 29 {
                        return Err(Error::Msg("deflate: invalid length code".into()));
                    }
                    let length = LEN_BASE[idx] as usize + br.read_bits(LEN_EXTRA[idx])? as usize;
                    let dsym = dist_tbl.decode(&mut br)? as usize;
                    if dsym >= 30 {
                        return Err(Error::Msg("deflate: invalid dist code".into()));
                    }
                    let dist = DIST_BASE[dsym] as usize + br.read_bits(DIST_EXTRA[dsym])? as usize;
                    if dist == 0 || dist > out.len() {
                        return Err(Error::Msg("deflate: backreference out of range".into()));
                    }
                    for _ in 0..length {
                        let b = out[out.len() - dist];
                        out.push(b);
                    }
                }
            }
        } else {
            return Err(Error::Msg("deflate: reserved block type".into()));
        }
        if bfinal != 0 {
            break;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// PNG Decoder
// ---------------------------------------------------------------------------

fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let a = a as i32;
    let b = b as i32;
    let c = c as i32;
    let p = a + b - c;
    let pa = (p - a).abs();
    let pb = (p - b).abs();
    let pc = (p - c).abs();
    if pa <= pb && pa <= pc {
        a as u8
    } else if pb <= pc {
        b as u8
    } else {
        c as u8
    }
}

/// Decodes a PNG byte buffer into interleaved `HWC` `u8` pixels + `(width, height, channels)`.
pub fn decode_png(bytes: &[u8]) -> Result<(Vec<u8>, usize, usize, usize)> {
    const SIG: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];
    if bytes.len() < 8 || bytes[..8] != SIG {
        return Err(Error::Msg("png: invalid signature".into()));
    }
    let mut pos = 8usize;
    let (mut width, mut height, mut bit_depth, mut color_type) = (0usize, 0usize, 0u8, 0u8);
    let mut idat = Vec::new();
    let mut seen_ihdr = false;
    let mut palette: Vec<u8> = Vec::new();
    let mut trns: Vec<u8> = Vec::new();
    let mut interlace = false;

    while pos + 12 <= bytes.len() {
        let len = u32::from_be_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        let ctype = &bytes[pos + 4..pos + 8];
        let data_start = pos + 8;
        let data_end = data_start + len;
        if data_end + 4 > bytes.len() {
            return Err(Error::Msg("png: truncated chunk".into()));
        }
        let chunk = &bytes[data_start..data_end];
        if ctype == b"IHDR" {
            if len < 13 {
                return Err(Error::Msg("png: invalid IHDR".into()));
            }
            width = u32::from_be_bytes(chunk[0..4].try_into().unwrap()) as usize;
            height = u32::from_be_bytes(chunk[4..8].try_into().unwrap()) as usize;
            bit_depth = chunk[8];
            color_type = chunk[9];
            interlace = match chunk[12] {
                0 => false,
                1 => true,
                m => return Err(Error::Msg(format!("png: unknown interlace method {m}"))),
            };
            seen_ihdr = true;
        } else if ctype == b"PLTE" {
            palette = chunk.to_vec();
        } else if ctype == b"tRNS" {
            trns = chunk.to_vec();
        } else if ctype == b"IDAT" {
            idat.extend_from_slice(chunk);
        } else if ctype == b"IEND" {
            break;
        }
        pos = data_end + 4;
    }
    if !seen_ihdr || width == 0 || height == 0 {
        return Err(Error::Msg("png: missing or empty IHDR".into()));
    }
    let samples: usize = match (color_type, bit_depth) {
        (0, 1 | 2 | 4 | 8 | 16) => 1,
        (2, 8 | 16) => 3,
        (3, 1 | 2 | 4 | 8) => 1,
        (4, 8 | 16) => 2,
        (6, 8 | 16) => 4,
        _ => return Err(Error::Msg(format!("png: unsupported color type {color_type} with bit depth {bit_depth}"))),
    };
    let depth = bit_depth as usize;
    crate::limits::check_elements("png pixels", &[height, width, samples.max(4)])?;
    let bits_per_pixel = samples * depth;
    let step = bits_per_pixel.div_ceil(8); // distance to the "left" byte for the filters
    let row_bytes = (width * bits_per_pixel).div_ceil(8);
    let raw = inflate_zlib(&idat)?;
    // undoes the scanline filters of one (sub-)image of `w` x `h` pixels stored at the start of `data`; returns the
    // unfiltered rows and the number of bytes consumed
    let unfilter = |data: &[u8], w: usize, h: usize| -> Result<(Vec<u8>, usize)> {
        let rb = (w * bits_per_pixel).div_ceil(8);
        let stride = rb + 1;
        if data.len() < h * stride {
            return Err(Error::Msg("png: decompressed data shorter than expected".into()));
        }
        let mut scan = vec![0u8; h * rb];
        for y in 0..h {
            let filter = data[y * stride];
            let src_row = &data[y * stride + 1..y * stride + 1 + rb];
            let dst_off = y * rb;
            for x in 0..rb {
                let a = if x >= step { scan[dst_off + x - step] } else { 0 };
                let b = if y > 0 { scan[dst_off - rb + x] } else { 0 };
                let c = if y > 0 && x >= step { scan[dst_off - rb + x - step] } else { 0 };
                scan[dst_off + x] = match filter {
                    0 => src_row[x],
                    1 => src_row[x].wrapping_add(a),
                    2 => src_row[x].wrapping_add(b),
                    3 => src_row[x].wrapping_add((((a as u16) + (b as u16)) >> 1) as u8),
                    4 => src_row[x].wrapping_add(paeth(a, b, c)),
                    _ => return Err(Error::Msg(format!("png: unknown filter {filter}"))),
                };
            }
        }
        Ok((scan, h * stride))
    };
    let scan = if !interlace {
        unfilter(&raw, width, height)?.0
    } else {
        // Adam7: seven reduced images, each with its own filtered scanlines, interleaved into the full image
        const PASSES: [(usize, usize, usize, usize); 7] = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)];
        let mut full = vec![0u8; height * row_bytes];
        let mut at = 0usize;
        for (x0, y0, dx, dy) in PASSES {
            if width <= x0 || height <= y0 {
                continue; // an empty pass has no scanlines at all
            }
            let (pw, ph) = ((width - x0).div_ceil(dx), (height - y0).div_ceil(dy));
            let (pass, used) = unfilter(&raw[at.min(raw.len())..], pw, ph)?;
            at += used;
            let prb = (pw * bits_per_pixel).div_ceil(8);
            for py in 0..ph {
                let src = &pass[py * prb..(py + 1) * prb];
                let dst_row = (y0 + py * dy) * row_bytes;
                for px in 0..pw {
                    let dxp = x0 + px * dx;
                    if bits_per_pixel % 8 == 0 {
                        let nb = bits_per_pixel / 8;
                        full[dst_row + dxp * nb..dst_row + (dxp + 1) * nb].copy_from_slice(&src[px * nb..(px + 1) * nb]);
                    } else {
                        // sub-byte pixels (gray / palette with 1, 2 or 4 bits): move the bit group
                        let (sb, db) = (px * bits_per_pixel, dxp * bits_per_pixel);
                        let v = (src[sb / 8] >> (8 - bits_per_pixel - sb % 8)) & ((1u8 << bits_per_pixel) - 1);
                        full[dst_row + db / 8] |= v << (8 - bits_per_pixel - db % 8);
                    }
                }
            }
        }
        full
    };
    // unpack to 8-bit samples: palette images become RGB (RGBA when a tRNS chunk gives alpha), 16-bit keeps the high byte
    // and low bit depths are scaled to the full 0..=255 range
    let sample_at = |row: &[u8], i: usize| -> u32 {
        match depth {
            8 => row[i] as u32,
            16 => row[2 * i] as u32,
            d => {
                let off = i * d;
                ((row[off / 8] >> (8 - d - off % 8)) as u32) & ((1 << d) - 1)
            }
        }
    };
    if color_type == 3 {
        if palette.is_empty() || palette.len() % 3 != 0 {
            return Err(Error::Msg("png: palette image without a valid PLTE chunk".into()));
        }
        let channels = if trns.is_empty() { 3 } else { 4 };
        let mut out = vec![0u8; width * height * channels];
        for y in 0..height {
            let row = &scan[y * row_bytes..(y + 1) * row_bytes];
            for x in 0..width {
                let idx = sample_at(row, x) as usize;
                if idx * 3 + 3 > palette.len() {
                    return Err(Error::Msg("png: palette index out of range".into()));
                }
                let o = (y * width + x) * channels;
                out[o..o + 3].copy_from_slice(&palette[idx * 3..idx * 3 + 3]);
                if channels == 4 {
                    out[o + 3] = trns.get(idx).copied().unwrap_or(255);
                }
            }
        }
        return Ok((out, width, height, channels));
    }
    if depth == 8 {
        return Ok((scan, width, height, samples));
    }
    let max = (1u32 << depth.min(8)) - 1;
    let mut out = vec![0u8; width * height * samples];
    for y in 0..height {
        let row = &scan[y * row_bytes..(y + 1) * row_bytes];
        for i in 0..width * samples {
            let v = sample_at(row, i);
            out[y * width * samples + i] = if depth == 16 { v as u8 } else { (v * 255 / max) as u8 };
        }
    }
    Ok((out, width, height, samples))
}

// ---------------------------------------------------------------------------
// Baseline JPEG Decoder (SOF0, 8-bit 1 or 3 components)
// ---------------------------------------------------------------------------

#[derive(Clone)]
/// Canonical Huffman table in the "mincode / maxcode / valptr" form of the JPEG specification (decoding needs no search).
struct JpegHuff {
    mincode: [i32; 17],
    maxcode: [i32; 17], // -1 when there is no code of that length
    valptr: [usize; 17],
    vals: Vec<u8>,
}

impl JpegHuff {
    fn from_dht(bits: &[u8; 16], syms: &[u8]) -> Result<Self> {
        let total: usize = bits.iter().map(|&b| b as usize).sum();
        if total > 256 || syms.len() < total {
            return Err(Error::Msg("jpeg: invalid DHT table".into()));
        }
        let (mut mincode, mut maxcode, mut valptr) = ([0i32; 17], [-1i32; 17], [0usize; 17]);
        let (mut code, mut k) = (0i32, 0usize);
        for len in 1..=16usize {
            let cnt = bits[len - 1] as usize;
            valptr[len] = k;
            mincode[len] = code;
            if cnt > 0 {
                maxcode[len] = code + cnt as i32 - 1;
            }
            code += cnt as i32;
            k += cnt;
            if code > (1 << len) {
                return Err(Error::Msg("jpeg: over-subscribed Huffman table".into()));
            }
            code <<= 1;
        }
        Ok(Self { mincode, maxcode, valptr, vals: syms[..total].to_vec() })
    }
}

/// Entropy-coded segment reader: undoes `FF 00` byte stuffing, stops at markers (feeding zero bits like libjpeg) and
/// resynchronizes at restart markers.
struct JpegBitReader<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u32,
    cnt: u8,
    marker: Option<u8>,
    padding_bytes: usize,
}

impl<'a> JpegBitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0, buf: 0, cnt: 0, marker: None, padding_bytes: 0 }
    }

    fn next_byte(&mut self) -> Result<u8> {
        if self.marker.is_some() {
            // a marker ends the entropy data; a decoder that still wants bits reads zeros (corrupt or truncated data)
            self.padding_bytes += 1;
            if self.padding_bytes > 64 {
                return Err(Error::Msg("jpeg: entropy data ended early".into()));
            }
            return Ok(0);
        }
        let Some(&b) = self.data.get(self.pos) else {
            return Err(Error::Msg("jpeg: unexpected EOF in scan".into()));
        };
        self.pos += 1;
        if b != 0xFF {
            return Ok(b);
        }
        // 0xFF: stuffed data byte (FF 00), fill bytes (FF FF ..) or a marker
        while self.data.get(self.pos) == Some(&0xFF) {
            self.pos += 1;
        }
        match self.data.get(self.pos) {
            Some(0) => {
                self.pos += 1;
                Ok(0xFF)
            }
            Some(&m) => {
                self.marker = Some(m);
                self.pos += 1;
                self.padding_bytes = 1;
                Ok(0)
            }
            None => Err(Error::Msg("jpeg: unexpected EOF in scan".into())),
        }
    }

    fn read_bits(&mut self, n: u8) -> Result<i32> {
        while self.cnt < n {
            let b = self.next_byte()?;
            self.buf = (self.buf << 8) | (b as u32);
            self.cnt += 8;
        }
        let shift = self.cnt - n;
        let v = (self.buf >> shift) & ((1u32 << n) - 1);
        self.cnt -= n;
        self.buf &= (1u32 << self.cnt).wrapping_sub(1);
        Ok(v as i32)
    }

    fn decode_huff(&mut self, h: &JpegHuff) -> Result<u8> {
        let mut code = 0i32;
        for len in 1..=16usize {
            code = (code << 1) | self.read_bits(1)?;
            if h.maxcode[len] >= 0 && code <= h.maxcode[len] && code >= h.mincode[len] {
                return Ok(h.vals[h.valptr[len] + (code - h.mincode[len]) as usize]);
            }
        }
        Err(Error::Msg("jpeg: invalid huffman code".into()))
    }

    fn receive_extend(&mut self, n: u8) -> Result<i32> {
        if n == 0 {
            return Ok(0);
        }
        if n > 16 {
            return Err(Error::Msg("jpeg: invalid coefficient size".into()));
        }
        let v = self.read_bits(n)?;
        let vt = 1i32 << (n - 1);
        Ok(if v < vt { v + ((-1i32) << n) + 1 } else { v })
    }

    /// Discards the padding bits of the finished restart interval and consumes the `RSTn` marker that follows.
    fn restart(&mut self) -> Result<()> {
        self.buf = 0;
        self.cnt = 0;
        self.padding_bytes = 0;
        if self.marker.is_none() {
            // not seen yet: the marker sits right behind the last data byte (possibly after fill bytes)
            loop {
                match self.data.get(self.pos) {
                    Some(0xFF) => {
                        self.pos += 1;
                        while self.data.get(self.pos) == Some(&0xFF) {
                            self.pos += 1;
                        }
                        match self.data.get(self.pos) {
                            Some(&m) => {
                                self.pos += 1;
                                self.marker = Some(m);
                                break;
                            }
                            None => return Err(Error::Msg("jpeg: missing restart marker".into())),
                        }
                    }
                    Some(_) => self.pos += 1, // garbage inside a damaged interval: skip to the next marker
                    None => return Err(Error::Msg("jpeg: missing restart marker".into())),
                }
            }
        }
        match self.marker.take() {
            Some(m) if (0xD0..=0xD7).contains(&m) => Ok(()),
            _ => Err(Error::Msg("jpeg: expected a restart marker".into())),
        }
    }
}

const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27,
    20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58,
    59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// `t[x][u] = c(u) / 2 * cos((2x + 1) u pi / 16)`: one 1-D inverse DCT stage is `out[x] = sum_u t[x][u] * in[u]`.
fn idct_table() -> [[f32; 8]; 8] {
    let mut t = [[0.0f32; 8]; 8];
    for (x, row) in t.iter_mut().enumerate() {
        for (u, v) in row.iter_mut().enumerate() {
            let cu = if u == 0 { core::f32::consts::FRAC_1_SQRT_2 } else { 1.0 };
            *v = 0.5 * cu * (((2 * x + 1) * u) as f32 * core::f32::consts::PI / 16.0).cos();
        }
    }
    t
}

fn idct_8x8(block: &[i32; 64], t: &[[f32; 8]; 8]) -> [u8; 64] {
    let mut out = [0u8; 64];
    if block[1..].iter().all(|&c| c == 0) {
        // DC only: a flat block
        let v = (block[0] as f32 * 0.125 + 128.0).round().clamp(0.0, 255.0) as u8;
        out.fill(v);
        return out;
    }
    let mut tmp = [0.0f32; 64];
    for y in 0..8 {
        for x in 0..8 {
            let mut sum = 0.0f32;
            for u in 0..8 {
                sum += t[x][u] * block[y * 8 + u] as f32;
            }
            tmp[y * 8 + x] = sum;
        }
    }
    for y in 0..8 {
        for x in 0..8 {
            let mut sum = 0.0f32;
            for v in 0..8 {
                sum += t[y][v] * tmp[v * 8 + x];
            }
            out[y * 8 + x] = (sum + 128.0).round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// Resamples a component plane (`pw` wide, valid area `cw x ch`) to `width x height`, factor `sx x sy`, with the centered
/// linear ("triangle") filter libjpeg uses for chroma upsampling.
#[allow(clippy::too_many_arguments)]
fn upsample_plane(plane: &[u8], pw: usize, cw: usize, ch: usize, width: usize, height: usize, sx: usize, sy: usize) -> Vec<u8> {
    let mut out = vec![0u8; width * height];
    if sx == 1 && sy == 1 {
        for y in 0..height {
            out[y * width..(y + 1) * width].copy_from_slice(&plane[y * pw..y * pw + width]);
        }
        return out;
    }
    let axis = |n: usize, s: usize, limit: usize| -> Vec<(usize, usize, f32)> {
        (0..n)
            .map(|i| {
                let f = (i as f32 + 0.5) / s as f32 - 0.5;
                let f0 = f.floor();
                let (a, b) = ((f0 as isize).clamp(0, limit as isize - 1) as usize, ((f0 as isize) + 1).clamp(0, limit as isize - 1) as usize);
                (a, b, f - f0)
            })
            .collect()
    };
    let (xs, ys) = (axis(width, sx, cw), axis(height, sy, ch));
    for (y, &(y0, y1, fy)) in ys.iter().enumerate() {
        let (r0, r1) = (&plane[y0 * pw..], &plane[y1 * pw..]);
        for (x, &(x0, x1, fx)) in xs.iter().enumerate() {
            let top = r0[x0] as f32 * (1.0 - fx) + r0[x1] as f32 * fx;
            let bot = r1[x0] as f32 * (1.0 - fx) + r1[x1] as f32 * fx;
            out[y * width + x] = (top * (1.0 - fy) + bot * fy + 0.5) as u8;
        }
    }
    out
}

/// Geometry and coefficient storage of the frame being decoded (all scans of a progressive file accumulate into it).
struct JpegFrame {
    width: usize,
    height: usize,
    comps: Vec<(u8, usize, usize, usize)>, // (id, hsamp, vsamp, qt_id)
    max_h: usize,
    max_v: usize,
    mcus_x: usize,
    mcus_y: usize,
    blocks_w: Vec<usize>, // per component, padded to whole MCUs
    blocks_h: Vec<usize>,
    coefs: Vec<Vec<i16>>, // per component: blocks_w * blocks_h blocks of 64 coefficients in natural (not zigzag) order, not dequantized
    qts: Vec<Option<[u16; 64]>>,
}

fn clamp_i16(v: i32) -> i16 {
    v.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

/// One block of a sequential (baseline / extended) scan.
fn jpeg_block_sequential(br: &mut JpegBitReader, dc_t: &JpegHuff, ac_t: &JpegHuff, pred: &mut i32, block: &mut [i16]) -> Result<()> {
    let s = br.decode_huff(dc_t)?;
    if s > 11 {
        return Err(Error::Msg("jpeg: invalid DC coefficient size".into()));
    }
    *pred += br.receive_extend(s)?;
    block[0] = clamp_i16(*pred);
    let mut k = 1usize;
    while k < 64 {
        let rs = br.decode_huff(ac_t)?;
        if rs == 0 {
            break;
        }
        let (run, size) = ((rs >> 4) as usize, rs & 0x0F);
        if rs == 0xF0 {
            k += 16; // ZRL: sixteen zero coefficients
            continue;
        }
        k += run;
        if k >= 64 {
            break;
        }
        block[ZIGZAG[k]] = clamp_i16(br.receive_extend(size)?);
        k += 1;
    }
    Ok(())
}

/// First pass over the AC band `ss..=se` of a progressive scan (coefficients shifted left by `al`, with end-of-band runs).
fn jpeg_ac_first(br: &mut JpegBitReader, ac_t: &JpegHuff, block: &mut [i16], (ss, se, al): (usize, usize, u8), eobrun: &mut u32) -> Result<()> {
    if *eobrun > 0 {
        *eobrun -= 1;
        return Ok(());
    }
    let mut k = ss;
    while k <= se {
        let rs = br.decode_huff(ac_t)?;
        let (r, s) = ((rs >> 4) as usize, rs & 0x0F);
        if s == 0 {
            if r < 15 {
                // EOBn: this block and the next (2^r + bits - 1) have no more coefficients in the band
                *eobrun = 1u32 << r;
                if r > 0 {
                    *eobrun += br.read_bits(r as u8)? as u32;
                }
                *eobrun -= 1;
                break;
            }
            k += 16; // ZRL
            continue;
        }
        k += r;
        if k > se {
            break;
        }
        block[ZIGZAG[k]] = clamp_i16(br.receive_extend(s)? << al);
        k += 1;
    }
    Ok(())
}

/// Refinement pass over the AC band: one more bit of every already non-zero coefficient plus newly non-zero ones (+-1 << al).
fn jpeg_ac_refine(br: &mut JpegBitReader, ac_t: &JpegHuff, block: &mut [i16], (ss, se, al): (usize, usize, u8), eobrun: &mut u32) -> Result<()> {
    let p1 = 1i32 << al;
    let m1 = -p1;
    // correction bit for an already non-zero coefficient
    let refine = |br: &mut JpegBitReader, c: &mut i16| -> Result<()> {
        if br.read_bits(1)? == 1 && (*c as i32 & p1) == 0 {
            *c = clamp_i16(*c as i32 + if *c >= 0 { p1 } else { m1 });
        }
        Ok(())
    };
    let mut k = ss;
    if *eobrun == 0 {
        while k <= se {
            let rs = br.decode_huff(ac_t)?;
            let (mut r, s) = ((rs >> 4) as i32, rs & 0x0F);
            let mut value = 0i32;
            if s != 0 {
                if s != 1 {
                    return Err(Error::Msg("jpeg: invalid refinement coefficient size".into()));
                }
                value = if br.read_bits(1)? == 1 { p1 } else { m1 };
            } else if r != 15 {
                *eobrun = 1u32 << r;
                if r > 0 {
                    *eobrun += br.read_bits(r as u8)? as u32;
                }
                break; // the remaining coefficients of this block are refined below
            }
            // skip `r` zero-history coefficients (refining the non-zero ones on the way) and stop on the next zero
            while k <= se {
                let c = &mut block[ZIGZAG[k]];
                if *c != 0 {
                    refine(br, c)?;
                } else {
                    r -= 1;
                    if r < 0 {
                        break;
                    }
                }
                k += 1;
            }
            if value != 0 && k <= se {
                block[ZIGZAG[k]] = clamp_i16(value);
            }
            k += 1;
        }
    }
    if *eobrun > 0 {
        while k <= se {
            let c = &mut block[ZIGZAG[k]];
            if *c != 0 {
                refine(br, c)?;
            }
            k += 1;
        }
        *eobrun -= 1;
    }
    Ok(())
}

/// Index of the next real marker (not stuffing `FF 00`, not `RSTn`, not fill bytes) at or after `start`, or `bytes.len()`.
fn jpeg_scan_end(bytes: &[u8], start: usize) -> usize {
    let mut i = start;
    while i + 1 < bytes.len() {
        if bytes[i] == 0xFF {
            let m = bytes[i + 1];
            if m == 0xFF {
                i += 1;
            } else if m == 0 || (0xD0..=0xD7).contains(&m) {
                i += 2;
            } else {
                return i;
            }
        } else {
            i += 1;
        }
    }
    bytes.len()
}

/// Decodes one scan (sequential or progressive) into the frame's coefficient storage.
#[allow(clippy::too_many_arguments)]
fn decode_jpeg_scan(
    frame: &mut JpegFrame,
    data: &[u8],
    scan: &[(usize, usize, usize)], // (component index, dc table id, ac table id)
    band: (usize, usize, u8, u8),   // (Ss, Se, Ah, Al)
    progressive: bool,
    restart_interval: usize,
    dc_huff: &[Option<JpegHuff>; 4],
    ac_huff: &[Option<JpegHuff>; 4],
) -> Result<()> {
    let bad = |m: &str| Error::Msg(format!("jpeg: {m}"));
    let (ss, se, ah, al) = band;
    let interleaved = scan.len() > 1;
    let (units_x, units_y) = if interleaved {
        (frame.mcus_x, frame.mcus_y)
    } else {
        // a lone component is scanned block by block over its own (unpadded) size
        let (_, hs, vs, _) = frame.comps[scan[0].0];
        let cw = (frame.width * hs).div_ceil(frame.max_h);
        let ch = (frame.height * vs).div_ceil(frame.max_v);
        (cw.div_ceil(8), ch.div_ceil(8))
    };
    let mut br = JpegBitReader::new(data);
    let mut pred = [0i32; 4];
    let mut eobrun = 0u32;
    let mut unit = 0usize;
    for uy in 0..units_y {
        for ux in 0..units_x {
            if restart_interval > 0 && unit > 0 && unit % restart_interval == 0 {
                br.restart()?;
                pred = [0; 4];
                eobrun = 0;
            }
            unit += 1;
            for &(cidx, dc_id, ac_id) in scan {
                let (_, hs, vs, _) = frame.comps[cidx];
                let (nh, nv) = if interleaved { (hs, vs) } else { (1, 1) };
                let bw = frame.blocks_w[cidx];
                for v in 0..nv {
                    for h in 0..nh {
                        let (bx, by) = if interleaved { (ux * hs + h, uy * vs + v) } else { (ux, uy) };
                        let off = (by * bw + bx) * 64;
                        let block = &mut frame.coefs[cidx][off..off + 64];
                        if !progressive {
                            let dc_t = dc_huff[dc_id].as_ref().ok_or_else(|| bad("missing DC Huffman table"))?;
                            let ac_t = ac_huff[ac_id].as_ref().ok_or_else(|| bad("missing AC Huffman table"))?;
                            jpeg_block_sequential(&mut br, dc_t, ac_t, &mut pred[cidx], block)?;
                        } else if ss == 0 {
                            if ah == 0 {
                                let dc_t = dc_huff[dc_id].as_ref().ok_or_else(|| bad("missing DC Huffman table"))?;
                                let s = br.decode_huff(dc_t)?;
                                if s > 11 {
                                    return Err(bad("invalid DC coefficient size"));
                                }
                                pred[cidx] += br.receive_extend(s)?;
                                block[0] = clamp_i16(pred[cidx] << al);
                            } else if br.read_bits(1)? == 1 {
                                block[0] |= 1 << al;
                            }
                        } else {
                            let ac_t = ac_huff[ac_id].as_ref().ok_or_else(|| bad("missing AC Huffman table"))?;
                            if ah == 0 {
                                jpeg_ac_first(&mut br, ac_t, block, (ss, se, al), &mut eobrun)?;
                            } else {
                                jpeg_ac_refine(&mut br, ac_t, block, (ss, se, al), &mut eobrun)?;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Decodes a Huffman JPEG (baseline `SOF0`, extended `SOF1` or progressive `SOF2`; 8-bit, grayscale or 3 components, any
/// chroma subsampling, restart intervals, interleaved or one-component-per-scan layouts, Adobe RGB or YCbCr) into `HWC` `u8`
/// pixels + `(width, height, channels)`. Arithmetic-coded, lossless, 12-bit, hierarchical and CMYK files are rejected with an error.
pub fn decode_jpeg(bytes: &[u8]) -> Result<(Vec<u8>, usize, usize, usize)> {
    let bad = |m: &str| Error::Msg(format!("jpeg: {m}"));
    if bytes.len() < 4 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
        return Err(bad("missing SOI marker"));
    }
    let mut pos = 2usize;
    let mut dqt = [[0u16; 64]; 4];
    let mut dc_huff: [Option<JpegHuff>; 4] = [None, None, None, None];
    let mut ac_huff: [Option<JpegHuff>; 4] = [None, None, None, None];
    let (mut width, mut height, mut ncomp) = (0usize, 0usize, 0usize);
    let mut comps = Vec::<(u8, usize, usize, usize)>::new(); // (id, hsamp, vsamp, qt_id)
    let mut progressive = false;
    let mut restart_interval = 0usize;
    let mut adobe_transform: Option<u8> = None;
    let mut frame: Option<JpegFrame> = None;
    let mut seen_sof = false;

    while pos + 1 < bytes.len() {
        if bytes[pos] != 0xFF {
            pos += 1;
            continue;
        }
        while pos < bytes.len() && bytes[pos] == 0xFF {
            pos += 1;
        }
        if pos >= bytes.len() {
            break;
        }
        let marker = bytes[pos];
        pos += 1;
        if marker == 0xD9 {
            break;
        }
        if marker == 0x00 || marker == 0x01 || (0xD0..=0xD8).contains(&marker) {
            continue;
        }
        if pos + 2 > bytes.len() {
            break;
        }
        let len = u16::from_be_bytes([bytes[pos], bytes[pos + 1]]) as usize;
        if len < 2 || pos + len > bytes.len() {
            return Err(bad("invalid segment length"));
        }
        let seg = &bytes[pos + 2..pos + len];
        pos += len;

        match marker {
            0xDB => {
                let mut sp = 0;
                while sp < seg.len() {
                    let info = seg[sp];
                    sp += 1;
                    let (prec, id) = (info >> 4, (info & 0x0F) as usize);
                    if id >= 4 || prec > 1 {
                        return Err(bad("invalid DQT"));
                    }
                    let need = if prec == 0 { 64 } else { 128 };
                    if sp + need > seg.len() {
                        return Err(bad("truncated DQT"));
                    }
                    for i in 0..64 {
                        dqt[id][ZIGZAG[i]] = if prec == 0 { seg[sp + i] as u16 } else { u16::from_be_bytes([seg[sp + 2 * i], seg[sp + 2 * i + 1]]) };
                    }
                    sp += need;
                }
            }
            0xC0..=0xC2 => {
                if frame.is_some() {
                    return Err(bad("a second frame header after the first scan"));
                }
                if seg.len() < 6 {
                    return Err(bad("truncated SOF"));
                }
                if seg[0] != 8 {
                    return Err(bad("only 8-bit samples are supported"));
                }
                progressive = marker == 0xC2;
                height = u16::from_be_bytes([seg[1], seg[2]]) as usize;
                width = u16::from_be_bytes([seg[3], seg[4]]) as usize;
                ncomp = seg[5] as usize;
                if seg.len() < 6 + 3 * ncomp {
                    return Err(bad("truncated SOF"));
                }
                comps.clear();
                for c in 0..ncomp {
                    let (id, samp, qt) = (seg[6 + c * 3], seg[7 + c * 3], seg[8 + c * 3] as usize);
                    let (hs, vs) = ((samp >> 4) as usize, (samp & 0x0F) as usize);
                    if hs == 0 || vs == 0 || hs > 4 || vs > 4 || qt >= 4 {
                        return Err(bad("invalid component description"));
                    }
                    comps.push((id, hs, vs, qt));
                }
                seen_sof = true;
            }
            0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF => {
                return Err(bad("only Huffman-coded baseline / progressive JPEGs are supported (lossless, hierarchical and arithmetic coding are not)"));
            }
            0xC4 => {
                let mut sp = 0;
                while sp < seg.len() {
                    if sp + 17 > seg.len() {
                        return Err(bad("truncated DHT"));
                    }
                    let info = seg[sp];
                    sp += 1;
                    let (cls, id) = (info >> 4, (info & 0x0F) as usize);
                    if cls > 1 || id >= 4 {
                        return Err(bad("invalid DHT id"));
                    }
                    let mut bits = [0u8; 16];
                    bits.copy_from_slice(&seg[sp..sp + 16]);
                    sp += 16;
                    let total: usize = bits.iter().map(|&b| b as usize).sum();
                    if sp + total > seg.len() {
                        return Err(bad("truncated DHT"));
                    }
                    let h = JpegHuff::from_dht(&bits, &seg[sp..sp + total])?;
                    sp += total;
                    if cls == 0 {
                        dc_huff[id] = Some(h);
                    } else {
                        ac_huff[id] = Some(h);
                    }
                }
            }
            0xDD => {
                if seg.len() < 2 {
                    return Err(bad("truncated DRI"));
                }
                restart_interval = u16::from_be_bytes([seg[0], seg[1]]) as usize;
            }
            0xEE if seg.len() >= 12 && &seg[..5] == b"Adobe" => adobe_transform = Some(seg[11]),
            0xDA => {
                if !seen_sof {
                    return Err(bad("scan before the frame header"));
                }
                if seg.is_empty() {
                    return Err(bad("truncated SOS"));
                }
                let ns = seg[0] as usize;
                if ns == 0 || ns > ncomp || seg.len() < 1 + 2 * ns + 3 {
                    return Err(bad("truncated or invalid SOS"));
                }
                if width == 0 || height == 0 {
                    return Err(bad("missing or empty frame header (SOF)"));
                }
                if ncomp != 1 && ncomp != 3 {
                    return Err(bad("only grayscale and 3-component images are supported"));
                }
                if frame.is_none() {
                    crate::limits::check_elements("jpeg pixels", &[width, height, ncomp])?;
                    if ncomp == 1 {
                        comps[0].1 = 1; // a single component is never subsampled: one 8x8 block per MCU
                        comps[0].2 = 1;
                    }
                    let max_h = comps.iter().map(|c| c.1).max().unwrap_or(1);
                    let max_v = comps.iter().map(|c| c.2).max().unwrap_or(1);
                    let (mcus_x, mcus_y) = (width.div_ceil(max_h * 8), height.div_ceil(max_v * 8));
                    let blocks_w: Vec<usize> = comps.iter().map(|c| mcus_x * c.1).collect();
                    let blocks_h: Vec<usize> = comps.iter().map(|c| mcus_y * c.2).collect();
                    let coefs = (0..ncomp).map(|c| vec![0i16; blocks_w[c] * blocks_h[c] * 64]).collect();
                    frame = Some(JpegFrame { width, height, comps: comps.clone(), max_h, max_v, mcus_x, mcus_y, blocks_w, blocks_h, coefs, qts: vec![None; ncomp] });
                }
                let fr = frame.as_mut().expect("frame was just created");
                let mut scan = Vec::<(usize, usize, usize)>::new(); // (component index, dc table id, ac table id)
                for i in 0..ns {
                    let (cid, tbl) = (seg[1 + i * 2], seg[2 + i * 2]);
                    let cidx = fr.comps.iter().position(|c| c.0 == cid).ok_or_else(|| bad("scan refers to an unknown component"))?;
                    if scan.iter().any(|s| s.0 == cidx) {
                        return Err(bad("a component appears twice in one scan"));
                    }
                    let (dc_id, ac_id) = ((tbl >> 4) as usize, (tbl & 0x0F) as usize);
                    if dc_id >= 4 || ac_id >= 4 {
                        return Err(bad("invalid Huffman table id in scan"));
                    }
                    if fr.qts[cidx].is_none() {
                        fr.qts[cidx] = Some(dqt[fr.comps[cidx].3]);
                    }
                    scan.push((cidx, dc_id, ac_id));
                }
                let (ss, se, a) = (seg[1 + 2 * ns] as usize, seg[2 + 2 * ns] as usize, seg[3 + 2 * ns]);
                let (ah, al) = (a >> 4, a & 0x0F);
                if progressive && (ss > se || se > 63 || (ss == 0 && se != 0) || (ss > 0 && ns != 1) || al > 13 || ah > 13) {
                    return Err(bad("invalid progressive scan parameters"));
                }
                let end = jpeg_scan_end(bytes, pos);
                // keep the terminating marker in the slice so a short final scan reads zero bits like a baseline file does
                let data = &bytes[pos..(end + 2).min(bytes.len())];
                decode_jpeg_scan(fr, data, &scan, (ss, se, ah, al), progressive, restart_interval, &dc_huff, &ac_huff)?;
                pos = end;
            }
            _ => {}
        }
    }

    if !seen_sof {
        return Err(bad("missing or empty frame header (SOF0/SOF1/SOF2)"));
    }
    let fr = frame.ok_or_else(|| bad("missing SOS"))?;
    let (max_h, max_v) = (fr.max_h, fr.max_v);
    let comps = &fr.comps;

    // dequantize + inverse DCT every block into component planes covering whole MCUs
    let idct_t = idct_table();
    let plane_w: Vec<usize> = fr.blocks_w.iter().map(|b| b * 8).collect();
    let mut planes: Vec<Vec<u8>> = (0..ncomp).map(|c| vec![0u8; plane_w[c] * fr.blocks_h[c] * 8]).collect();
    for c in 0..ncomp {
        let qt = fr.qts[c].unwrap_or(dqt[comps[c].3]);
        for by in 0..fr.blocks_h[c] {
            for bx in 0..fr.blocks_w[c] {
                let off = (by * fr.blocks_w[c] + bx) * 64;
                let mut coeff = [0i32; 64];
                for i in 0..64 {
                    coeff[i] = fr.coefs[c][off + i] as i32 * qt[i] as i32;
                }
                let blk = idct_8x8(&coeff, &idct_t);
                for row in 0..8 {
                    let at = (by * 8 + row) * plane_w[c] + bx * 8;
                    planes[c][at..at + 8].copy_from_slice(&blk[row * 8..row * 8 + 8]);
                }
            }
        }
    }

    // chroma planes back to full resolution
    let full: Vec<Vec<u8>> = (0..ncomp)
        .map(|c| {
            let (sx, sy) = (max_h / comps[c].1, max_v / comps[c].2);
            let (cw, ch) = (width.div_ceil(sx), height.div_ceil(sy));
            upsample_plane(&planes[c], plane_w[c], cw, ch, width, height, sx, sy)
        })
        .collect();
    let mut pixels = vec![0u8; width * height * ncomp];
    if ncomp == 1 {
        pixels.copy_from_slice(&full[0]);
    } else {
        // RGB stored directly: Adobe transform 0, or component ids 'R' 'G' 'B'; everything else is YCbCr
        let is_rgb = adobe_transform == Some(0) || (comps[0].0 == b'R' && comps[1].0 == b'G' && comps[2].0 == b'B');
        for i in 0..width * height {
            let (a, b, c) = (full[0][i], full[1][i], full[2][i]);
            let o = i * 3;
            if is_rgb {
                pixels[o] = a;
                pixels[o + 1] = b;
                pixels[o + 2] = c;
            } else {
                let (yv, cb, cr) = (a as f32, b as f32 - 128.0, c as f32 - 128.0);
                pixels[o] = (yv + 1.402 * cr).round().clamp(0.0, 255.0) as u8;
                pixels[o + 1] = (yv - 0.34414 * cb - 0.71414 * cr).round().clamp(0.0, 255.0) as u8;
                pixels[o + 2] = (yv + 1.772 * cb).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok((pixels, width, height, ncomp))
}

/// Auto-detects PNG or JPEG from magic bytes and returns a `[C, H, W]` `u8` [`Tensor`].
pub fn decode_image_tensor(bytes: &[u8]) -> Result<Tensor> {
    let (hwc, w, h, c) = if bytes.len() >= 8 && bytes[..8] == [137, 80, 78, 71, 13, 10, 26, 10] {
        decode_png(bytes)?
    } else if bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] == 0xD8 {
        decode_jpeg(bytes)?
    } else {
        return Err(Error::Msg("image_codec: unrecognized image format (expected PNG or JPEG)".into()));
    };
    let t_hwc = Tensor::from_storage(
        Storage::new_cpu(CpuStorage::U8(hwc)),
        Layout::contiguous(vec![h, w, c]),
    );
    t_hwc.permute(&[2, 0, 1])?.contiguous()
}

/// Reads a PNG or JPEG file from disk into a `[C, H, W]` `u8` [`Tensor`].
pub fn load_image(path: impl AsRef<Path>) -> Result<Tensor> {
    let bytes = std::fs::read(path.as_ref())
        .map_err(|e| Error::Msg(format!("load_image '{}': {e}", path.as_ref().display())))?;
    decode_image_tensor(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crc32(data: &[u8]) -> u32 {
        let mut c = 0xFFFF_FFFFu32;
        for &b in data {
            c ^= b as u32;
            for _ in 0..8 {
                let m = (c & 1).wrapping_neg();
                c = (c >> 1) ^ (0xEDB8_8320 & m);
            }
        }
        !c
    }

    fn adler32(data: &[u8]) -> u32 {
        let (mut s1, mut s2) = (1u32, 0u32);
        for &b in data {
            s1 = (s1 + b as u32) % 65521;
            s2 = (s2 + s1) % 65521;
        }
        (s2 << 16) | s1
    }

    fn png_chunk(ctype: &[u8; 4], data: &[u8], out: &mut Vec<u8>) {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(ctype);
        out.extend_from_slice(data);
        let mut crc_in = Vec::with_capacity(4 + data.len());
        crc_in.extend_from_slice(ctype);
        crc_in.extend_from_slice(data);
        out.extend_from_slice(&crc32(&crc_in).to_be_bytes());
    }

    fn png_with(width: u32, height: u32, depth: u8, color_type: u8, extra: &[(&[u8; 4], Vec<u8>)], raw: &[u8]) -> Vec<u8> {
        let mut png = vec![137, 80, 78, 71, 13, 10, 26, 10];
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&width.to_be_bytes());
        ihdr.extend_from_slice(&height.to_be_bytes());
        ihdr.extend_from_slice(&[depth, color_type, 0, 0, 0]);
        png_chunk(b"IHDR", &ihdr, &mut png);
        for (t, d) in extra {
            png_chunk(t, d, &mut png);
        }
        let mut zlib = vec![0x78, 0x01, 0x01];
        let len = raw.len() as u16;
        zlib.extend_from_slice(&len.to_le_bytes());
        zlib.extend_from_slice(&(!len).to_le_bytes());
        zlib.extend_from_slice(raw);
        zlib.extend_from_slice(&adler32(raw).to_be_bytes());
        png_chunk(b"IDAT", &zlib, &mut png);
        png_chunk(b"IEND", &[], &mut png);
        png
    }

    #[test]
    fn decodes_palette_low_bit_depth_and_16_bit_pngs() {
        // 2-bit palette 3x2 with transparency: indices [0,1,2] / [3,2,1]
        let plte = vec![10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42];
        let png = png_with(3, 2, 2, 3, &[(b"PLTE", plte), (b"tRNS", vec![255, 128, 64])], &[0, 0b0001_1000, 0, 0b1110_0100]);
        let (px, w, h, c) = decode_png(&png).unwrap();
        assert_eq!((w, h, c), (3, 2, 4));
        assert_eq!(&px[..12], &[10, 11, 12, 255, 20, 21, 22, 128, 30, 31, 32, 64]);
        assert_eq!(&px[12..16], &[40, 41, 42, 255], "indices beyond tRNS are opaque");
        // 4-bit grayscale scales to 0..=255
        let (px, ..) = decode_png(&png_with(2, 1, 4, 0, &[], &[0, 0x0F])).unwrap();
        assert_eq!(px, vec![0, 255]);
        // 1-bit grayscale, 8 pixels in one byte
        let (px, ..) = decode_png(&png_with(8, 1, 1, 0, &[], &[0, 0b1010_0101])).unwrap();
        assert_eq!(px, vec![255, 0, 255, 0, 0, 255, 0, 255]);
        // 16-bit gray keeps the high byte
        let (px, ..) = decode_png(&png_with(2, 1, 16, 0, &[], &[0, 0x12, 0x34, 0xAB, 0xCD])).unwrap();
        assert_eq!(px, vec![0x12, 0xAB]);
        // palette image without PLTE and a bad bit depth for the color type are errors (Adam7 is covered by tests/png_adam7.rs)
        assert!(decode_png(&png_with(1, 1, 8, 3, &[], &[0, 0])).is_err());
        assert!(decode_png(&png_with(1, 1, 4, 2, &[], &[0, 0, 0, 0])).is_err());
    }

    #[test]
    fn hostile_png_headers_are_rejected_without_huge_allocations() {
        // claims a 4-billion x 4-billion image but carries a few bytes
        assert!(decode_png(&png_with(u32::MAX, u32::MAX, 8, 6, &[], &[0, 0])).is_err());
        assert!(decode_png(&png_with(0, 5, 8, 2, &[], &[0])).is_err());
        assert!(decode_png(&png_with(70_000, 70_000, 8, 0, &[], &[0, 0])).is_err());
    }

    #[test]
    fn decodes_valid_2x2_rgb_png() {
        let mut png = vec![137, 80, 78, 71, 13, 10, 26, 10];
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&2u32.to_be_bytes());
        ihdr.extend_from_slice(&2u32.to_be_bytes());
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit RGB
        png_chunk(b"IHDR", &ihdr, &mut png);

        // Scanlines: filter=0 + 2 RGB pixels per row
        let raw = vec![
            0, 255, 0, 0, 0, 255, 0, // row 0: red, green
            0, 0, 0, 255, 255, 255, 0, // row 1: blue, yellow
        ];
        let mut zlib = vec![0x78, 0x01, 0x01]; // stored DEFLATE final block
        let len = raw.len() as u16;
        zlib.extend_from_slice(&len.to_le_bytes());
        zlib.extend_from_slice(&(!len).to_le_bytes());
        zlib.extend_from_slice(&raw);
        zlib.extend_from_slice(&adler32(&raw).to_be_bytes());
        png_chunk(b"IDAT", &zlib, &mut png);
        png_chunk(b"IEND", &[], &mut png);

        let t = decode_image_tensor(&png).unwrap();
        assert_eq!(t.shape(), &[3, 2, 2]);
        let v = t.to_vec::<u8>().unwrap();
        // Channel 0 (R): [255, 0, 0, 255]
        assert_eq!(&v[0..4], &[255, 0, 0, 255]);
        // Channel 1 (G): [0, 255, 0, 255]
        assert_eq!(&v[4..8], &[0, 255, 0, 255]);
        // Channel 2 (B): [0, 0, 255, 0]
        assert_eq!(&v[8..12], &[0, 0, 255, 0]);
    }
}
