// JPEG decoder regression tests. A tiny baseline encoder (all-ones quantization, fixed-length Huffman codes) produces files
// with and without restart intervals, with chroma subsampling, and so on; the decoder must reproduce the image and survive
// truncated and corrupted input without panicking.
use oxideon_core::image_codec::decode_jpeg;

struct Bits {
    out: Vec<u8>,
    acc: u32,
    n: u32,
}
impl Bits {
    fn put(&mut self, code: u32, len: u32) {
        for i in (0..len).rev() {
            self.acc = (self.acc << 1) | ((code >> i) & 1);
            self.n += 1;
            if self.n == 8 {
                self.flush_byte();
            }
        }
    }
    fn flush_byte(&mut self) {
        let b = self.acc as u8;
        self.out.push(b);
        if b == 0xFF {
            self.out.push(0);
        }
        self.acc = 0;
        self.n = 0;
    }
    fn pad_ones(&mut self) {
        while self.n != 0 {
            self.put(1, 1);
        }
    }
}

const ZZ: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
];

fn fdct(block: &[f32; 64]) -> [i32; 64] {
    let mut out = [0i32; 64];
    for v in 0..8 {
        for u in 0..8 {
            let mut s = 0.0f32;
            for y in 0..8 {
                for x in 0..8 {
                    s += (block[y * 8 + x] - 128.0)
                        * (((2 * x + 1) * u) as f32 * std::f32::consts::PI / 16.0).cos()
                        * (((2 * y + 1) * v) as f32 * std::f32::consts::PI / 16.0).cos();
                }
            }
            let (cu, cv) = (if u == 0 { std::f32::consts::FRAC_1_SQRT_2 } else { 1.0 }, if v == 0 { std::f32::consts::FRAC_1_SQRT_2 } else { 1.0 });
            out[v * 8 + u] = (0.25 * cu * cv * s).round() as i32;
        }
    }
    out
}

fn category(v: i32) -> u32 {
    32 - v.unsigned_abs().leading_zeros()
}

fn put_value(b: &mut Bits, v: i32, size: u32) {
    if size > 0 {
        let bits = if v >= 0 { v } else { v + (1 << size) - 1 };
        b.put(bits as u32, size);
    }
}

/// AC symbol -> code index in the 162-entry table: EOB, ZRL, then (run, size) pairs
fn ac_index(sym: u8) -> u32 {
    match sym {
        0x00 => 0,
        0xF0 => 1,
        s => 2 + (s >> 4) as u32 * 10 + ((s & 15) as u32 - 1),
    }
}

/// `planes[c]` is the full-resolution plane of component c (w x h); `samp[c]` its (h, v) sampling factors
fn encode(planes: &[Vec<u8>], w: usize, h: usize, samp: &[(usize, usize)], restart: usize, ids: &[u8], adobe: Option<u8>) -> Vec<u8> {
    let nc = planes.len();
    let (mh, mv) = (samp.iter().map(|s| s.0).max().unwrap(), samp.iter().map(|s| s.1).max().unwrap());
    let mut o = vec![0xFF, 0xD8];
    if let Some(t) = adobe {
        o.extend_from_slice(&[0xFF, 0xEE, 0, 14, b'A', b'd', b'o', b'b', b'e', 0, 100, 0, 0, 0, 0, t]);
    }
    o.extend_from_slice(&[0xFF, 0xDB, 0, 67, 0]);
    o.extend(std::iter::repeat_n(1u8, 64));
    o.extend_from_slice(&[0xFF, 0xC0, 0, (8 + 3 * nc) as u8, 8, (h >> 8) as u8, h as u8, (w >> 8) as u8, w as u8, nc as u8]);
    for c in 0..nc {
        o.extend_from_slice(&[ids[c], ((samp[c].0 << 4) | samp[c].1) as u8, 0]);
    }
    // DC table: 12 symbols of length 4; AC table: 162 symbols of length 8
    o.extend_from_slice(&[0xFF, 0xC4, 0, 31, 0x00, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    o.extend(0u8..12);
    o.extend_from_slice(&[0xFF, 0xC4, 0, (19 + 162) as u8, 0x10, 0, 0, 0, 0, 0, 0, 0, 162, 0, 0, 0, 0, 0, 0, 0, 0]);
    o.push(0x00);
    o.push(0xF0);
    for r in 0..16u8 {
        for s in 1..=10u8 {
            o.push((r << 4) | s);
        }
    }
    if restart > 0 {
        o.extend_from_slice(&[0xFF, 0xDD, 0, 4, (restart >> 8) as u8, restart as u8]);
    }
    o.extend_from_slice(&[0xFF, 0xDA, 0, (6 + 2 * nc) as u8, nc as u8]);
    for c in 0..nc {
        o.extend_from_slice(&[ids[c], 0x00]);
    }
    o.extend_from_slice(&[0, 63, 0]);

    let (mcus_x, mcus_y) = (w.div_ceil(8 * mh), h.div_ceil(8 * mv));
    let mut bits = Bits { out: vec![], acc: 0, n: 0 };
    let mut pred = vec![0i32; nc];
    let mut count = 0usize;
    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            if restart > 0 && count > 0 && count % restart == 0 {
                bits.pad_ones();
                bits.out.extend_from_slice(&[0xFF, 0xD0 + ((count / restart - 1) % 8) as u8]);
                pred.iter_mut().for_each(|p| *p = 0);
            }
            count += 1;
            for c in 0..nc {
                let (hs, vs) = samp[c];
                let (sx, sy) = (mh / hs, mv / vs);
                for v in 0..vs {
                    for hh in 0..hs {
                        let mut blk = [0.0f32; 64];
                        for by in 0..8 {
                            for bx in 0..8 {
                                // box-average the full-resolution plane over the subsampling window (clamped at the edges)
                                let (mut sum, mut cnt) = (0.0f32, 0.0f32);
                                for py in 0..sy {
                                    for px in 0..sx {
                                        let x = ((mx * hs + hh) * 8 + bx) * sx + px;
                                        let y = ((my * vs + v) * 8 + by) * sy + py;
                                        let (x, y) = (x.min(w - 1), y.min(h - 1));
                                        sum += planes[c][y * w + x] as f32;
                                        cnt += 1.0;
                                    }
                                }
                                blk[by * 8 + bx] = sum / cnt;
                            }
                        }
                        let q = fdct(&blk);
                        let diff = q[0] - pred[c];
                        pred[c] = q[0];
                        let s = category(diff);
                        bits.put(s, 4);
                        put_value(&mut bits, diff, s);
                        let mut run = 0u32;
                        let last = (1..64).rev().find(|&k| q[ZZ[k]] != 0).unwrap_or(0);
                        for k in 1..=last {
                            let val = q[ZZ[k]];
                            if val == 0 {
                                run += 1;
                                continue;
                            }
                            while run > 15 {
                                bits.put(ac_index(0xF0), 8);
                                run -= 16;
                            }
                            let size = category(val);
                            bits.put(ac_index(((run << 4) | size) as u8), 8);
                            put_value(&mut bits, val, size);
                            run = 0;
                        }
                        if last < 63 {
                            bits.put(ac_index(0x00), 8);
                        }
                    }
                }
            }
        }
    }
    bits.pad_ones();
    o.extend_from_slice(&bits.out);
    o.extend_from_slice(&[0xFF, 0xD9]);
    o
}

fn pattern(w: usize, h: usize, k: usize) -> Vec<u8> {
    (0..w * h).map(|i| ((i % w) * 5 + (i / w) * 3 + k * 40 + ((i % w) * (i / w)) % 13) as u8 % 200 + 20).collect()
}

fn max_diff(a: &[u8], b: &[u8]) -> i32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(&x, &y)| (x as i32 - y as i32).abs()).max().unwrap_or(0)
}

#[test]
fn grayscale_round_trip_with_and_without_restart_intervals() {
    for (w, h) in [(8, 8), (21, 13), (50, 33)] {
        let img = pattern(w, h, 0);
        let plain = encode(&[img.clone()], w, h, &[(1, 1)], 0, &[1], None);
        let (px, dw, dh, c) = decode_jpeg(&plain).unwrap();
        assert_eq!((dw, dh, c), (w, h, 1));
        assert!(max_diff(&px, &img) <= 2, "{w}x{h} grayscale round trip");
        for ri in [1usize, 2, 3, 5] {
            let rst = encode(&[img.clone()], w, h, &[(1, 1)], ri, &[1], None);
            assert_eq!(decode_jpeg(&rst).unwrap().0, px, "{w}x{h} restart interval {ri} must decode identically");
        }
    }
}

#[test]
fn color_with_subsampling_restart_markers_and_component_ids() {
    let (w, h) = (45, 29);
    let planes: Vec<Vec<u8>> = (0..3).map(|k| pattern(w, h, k)).collect();
    // YCbCr stored as is; the decoder converts to RGB, so compare against its own no-restart decode
    for (samp, name) in [(vec![(1, 1), (1, 1), (1, 1)], "4:4:4"), (vec![(2, 1), (1, 1), (1, 1)], "4:2:2"), (vec![(2, 2), (1, 1), (1, 1)], "4:2:0")] {
        for ids in [[1u8, 2, 3], [0, 1, 2]] {
            let plain = encode(&planes, w, h, &samp, 0, &ids, None);
            let (base, dw, dh, c) = decode_jpeg(&plain).unwrap();
            assert_eq!((dw, dh, c), (w, h, 3), "{name}");
            for ri in [1usize, 2, 4] {
                let rst = encode(&planes, w, h, &samp, ri, &ids, None);
                assert_eq!(decode_jpeg(&rst).unwrap().0, base, "{name} ids {ids:?} restart {ri}");
            }
        }
    }
    // 4:4:4 is lossless apart from rounding: check the YCbCr -> RGB conversion itself on a flat gray image
    let flat: Vec<Vec<u8>> = vec![vec![100; 16 * 16], vec![128; 16 * 16], vec![128; 16 * 16]];
    let (px, ..) = decode_jpeg(&encode(&flat, 16, 16, &[(1, 1), (1, 1), (1, 1)], 0, &[1, 2, 3], None)).unwrap();
    assert!(px.iter().all(|&v| (v as i32 - 100).abs() <= 1), "neutral chroma gives gray");
    // Adobe transform 0 means the three components are R, G, B themselves
    let rgb: Vec<Vec<u8>> = vec![vec![200; 16 * 16], vec![50; 16 * 16], vec![90; 16 * 16]];
    let (px, ..) = decode_jpeg(&encode(&rgb, 16, 16, &[(1, 1), (1, 1), (1, 1)], 0, &[1, 2, 3], Some(0))).unwrap();
    assert!(px.chunks(3).all(|p| (p[0] as i32 - 200).abs() <= 1 && (p[1] as i32 - 50).abs() <= 1 && (p[2] as i32 - 90).abs() <= 1), "Adobe RGB is not color-converted");
}

#[test]
fn unsupported_and_damaged_files_are_errors_never_panics() {
    let (w, h) = (30, 20);
    let planes: Vec<Vec<u8>> = (0..3).map(|k| pattern(w, h, k)).collect();
    let good = encode(&planes, w, h, &[(2, 2), (1, 1), (1, 1)], 2, &[1, 2, 3], None);
    assert!(decode_jpeg(&good).is_ok());
    // lossless (SOF3) and arithmetic-coded (SOF9) frames are named as unsupported
    let sof = good.windows(2).position(|p| p == [0xFF, 0xC0]).unwrap();
    for marker in [0xC3u8, 0xC9] {
        let mut other = good.clone();
        other[sof + 1] = marker;
        let e = decode_jpeg(&other).unwrap_err().to_string();
        assert!(e.contains("baseline / progressive"), "unsupported coding needs a clear message, got: {e}");
    }
    // a sequential scan header under a progressive frame marker is inconsistent: an error, not garbage or a panic
    let mut mislabeled = good.clone();
    mislabeled[sof + 1] = 0xC2;
    assert!(decode_jpeg(&mislabeled).is_err());
    // every truncation of the file
    for len in 0..good.len() {
        let _ = std::panic::catch_unwind(|| decode_jpeg(&good[..len])).unwrap_or_else(|_| panic!("panicked on a file truncated to {len} bytes"));
    }
    // corrupted bytes everywhere
    let mut seed = 12345u32;
    for _ in 0..400 {
        let mut bad = good.clone();
        for _ in 0..3 {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let i = (seed >> 8) as usize % bad.len();
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            bad[i] = (seed >> 16) as u8;
        }
        let r = std::panic::catch_unwind(|| decode_jpeg(&bad));
        assert!(r.is_ok(), "panicked on a corrupted file");
    }
}
