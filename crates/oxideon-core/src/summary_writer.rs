//! TensorBoard event files (`tfevents`): scalars, histograms and text, readable by TensorBoard, W&B and anything that reads
//! TFRecord event logs. No protobuf or TensorFlow dependency: the few messages involved are encoded by hand.
//!
//! ```no_run
//! # use oxideon_core::summary_writer::SummaryWriter;
//! let mut w = SummaryWriter::new("runs/exp1").unwrap();
//! w.add_scalar("loss/train", 0.5, 100).unwrap();
//! w.flush().unwrap();
//! ```

use crate::error::{Error, Result};
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

fn crc32c_table() -> &'static [u32; 256] {
    static T: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0u32; 256];
        for (i, slot) in t.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 == 1 { (c >> 1) ^ 0x82F6_3B78 } else { c >> 1 };
            }
            *slot = c;
        }
        t
    })
}

/// CRC-32C (Castagnoli), the checksum of the TFRecord format.
pub fn crc32c(data: &[u8]) -> u32 {
    let t = crc32c_table();
    !data.iter().fold(!0u32, |c, &b| t[((c ^ b as u32) & 0xff) as usize] ^ (c >> 8))
}

fn masked_crc(data: &[u8]) -> u32 {
    let c = crc32c(data);
    c.rotate_right(15).wrapping_add(0xa282_ead8)
}

fn varint(mut v: u64, out: &mut Vec<u8>) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}
fn tag(field: u32, wire: u8, out: &mut Vec<u8>) {
    varint(((field as u64) << 3) | wire as u64, out);
}
fn bytes_field(field: u32, data: &[u8], out: &mut Vec<u8>) {
    tag(field, 2, out);
    varint(data.len() as u64, out);
    out.extend_from_slice(data);
}

fn event(wall_time: f64, step: i64, body: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
    let mut e = Vec::new();
    tag(1, 1, &mut e);
    e.extend_from_slice(&wall_time.to_le_bytes());
    tag(2, 0, &mut e);
    varint(step as u64, &mut e);
    body(&mut e);
    e
}

fn summary_value(tag_name: &str, f: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
    let mut v = Vec::new();
    bytes_field(1, tag_name.as_bytes(), &mut v);
    f(&mut v);
    let mut summary = Vec::new();
    bytes_field(1, &v, &mut summary);
    summary
}

pub struct SummaryWriter {
    file: std::io::BufWriter<std::fs::File>,
    path: std::path::PathBuf,
}

impl SummaryWriter {
    /// Creates `dir` if needed and starts a new `events.out.tfevents.*` file in it.
    pub fn new(dir: impl AsRef<std::path::Path>) -> Result<Self> {
        let io = |e: std::io::Error| Error::Msg(format!("summary writer: {e}"));
        std::fs::create_dir_all(dir.as_ref()).map_err(io)?;
        let now = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64());
        let host = std::env::var("COMPUTERNAME").or_else(|_| std::env::var("HOSTNAME")).unwrap_or_else(|_| "localhost".into());
        let path = dir.as_ref().join(format!("events.out.tfevents.{}.{}.{}", now as u64, host, std::process::id()));
        let file = std::io::BufWriter::new(std::fs::File::create(&path).map_err(io)?);
        let mut w = Self { file, path };
        let version = event(now, 0, |e| bytes_field(3, b"brain.Event:2", e));
        w.record(&version)?;
        Ok(w)
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    fn record(&mut self, data: &[u8]) -> Result<()> {
        let io = |e: std::io::Error| Error::Msg(format!("summary writer: {e}"));
        let len = (data.len() as u64).to_le_bytes();
        self.file.write_all(&len).map_err(io)?;
        self.file.write_all(&masked_crc(&len).to_le_bytes()).map_err(io)?;
        self.file.write_all(data).map_err(io)?;
        self.file.write_all(&masked_crc(data).to_le_bytes()).map_err(io)
    }

    fn now() -> f64 {
        SystemTime::now().duration_since(UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
    }

    /// A scalar curve point (`tag` may contain `/` to group charts).
    pub fn add_scalar(&mut self, tag_name: &str, value: f64, step: i64) -> Result<()> {
        let s = summary_value(tag_name, |v| {
            tag(2, 5, v);
            v.extend_from_slice(&(value as f32).to_le_bytes());
        });
        let e = event(Self::now(), step, |e| bytes_field(5, &s, e));
        self.record(&e)
    }

    /// A histogram of `values` with `bins` equal-width buckets.
    pub fn add_histogram(&mut self, tag_name: &str, values: &[f64], bins: usize, step: i64) -> Result<()> {
        let vals: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
        if vals.is_empty() || bins == 0 {
            return Err(Error::Msg("summary writer: a histogram needs at least one finite value and one bin".into()));
        }
        let (lo, hi) = vals.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
        let width = ((hi - lo) / bins as f64).max(f64::MIN_POSITIVE);
        let mut counts = vec![0.0f64; bins];
        for &v in &vals {
            counts[(((v - lo) / width) as usize).min(bins - 1)] += 1.0;
        }
        let limits: Vec<f64> = (1..=bins).map(|i| lo + width * i as f64).collect();
        let packed = |xs: &[f64]| xs.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let (sum, sum_sq) = (vals.iter().sum::<f64>(), vals.iter().map(|v| v * v).sum::<f64>());
        let mut h = Vec::new();
        for (field, x) in [(1u32, lo), (2, hi), (3, vals.len() as f64), (4, sum), (5, sum_sq)] {
            tag(field, 1, &mut h);
            h.extend_from_slice(&x.to_le_bytes());
        }
        bytes_field(6, &packed(&limits), &mut h);
        bytes_field(7, &packed(&counts), &mut h);
        let s = summary_value(tag_name, |v| bytes_field(5, &h, v));
        let e = event(Self::now(), step, |e| bytes_field(5, &s, e));
        self.record(&e)
    }

    /// Several scalars under one tag prefix: `main_tag/<name>` for every entry.
    pub fn add_scalars(&mut self, main_tag: &str, values: &[(&str, f64)], step: i64) -> Result<()> {
        for (k, v) in values {
            self.add_scalar(&format!("{main_tag}/{k}"), *v, step)?;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        self.file.flush().map_err(|e| Error::Msg(format!("summary writer: {e}")))
    }
}

impl Drop for SummaryWriter {
    fn drop(&mut self) {
        let _ = self.file.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32c_known_vectors_and_a_readable_file() {
        // RFC 3720 test vectors
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
        assert_eq!(crc32c(&[0u8; 32]), 0x8A91_36AA);
        let dir = std::env::temp_dir().join("oxideon_tb_test");
        let _ = std::fs::remove_dir_all(&dir);
        let mut w = SummaryWriter::new(&dir).unwrap();
        w.add_scalar("loss", 0.5, 1).unwrap();
        w.add_histogram("w", &[1.0, 2.0, 3.0, 4.0], 2, 1).unwrap();
        w.flush().unwrap();
        let bytes = std::fs::read(w.path()).unwrap();
        // every record carries valid length and data checksums
        let mut at = 0;
        let mut n = 0;
        while at < bytes.len() {
            let len = u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap()) as usize;
            assert_eq!(u32::from_le_bytes(bytes[at + 8..at + 12].try_into().unwrap()), masked_crc(&bytes[at..at + 8]));
            let data = &bytes[at + 12..at + 12 + len];
            assert_eq!(u32::from_le_bytes(bytes[at + 12 + len..at + 16 + len].try_into().unwrap()), masked_crc(data));
            at += 16 + len;
            n += 1;
        }
        assert_eq!(n, 3);
        drop(w);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
