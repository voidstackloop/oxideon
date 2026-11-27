//! Voice analysis and synthesis: a YIN pitch tracker and a harmonic-plus-noise (source-filter) synthesiser.
//!
//! A spectral envelope alone cannot carry the harmonics of a voice, so a vocoder that only sees a mel spectrogram has to guess
//! them (and the phase). Here the pitch is explicit: voiced speech is a sum of harmonics of F0 whose amplitudes follow the
//! spectral envelope and whose phases follow its minimum-phase response, unvoiced speech is noise shaped by the same envelope.

use crate::error::{Error, Result};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------------------------------------------------
// YIN pitch tracking
// ---------------------------------------------------------------------------------------------------------------------

/// Pitch of `wave` at frame centres `t * hop` (`wave.len() / hop + 1` frames): `(f0 in Hz, confidence in [0, 1])`.
///
/// YIN (de Cheveigne and Kawahara, 2002) on a copy of the signal decimated to about 8 kHz. `f0` is always the best candidate
/// inside `[fmin, fmax]`; `confidence = 1 - d'(tau)` of the cumulative-mean-normalised difference, so values above about 0.7
/// are clearly voiced. Silent frames get confidence 0. Frames are processed on all cores.
pub fn pitch_track(wave: &[f32], sample_rate: usize, hop: usize, fmin: f32, fmax: f32) -> Result<(Vec<f32>, Vec<f32>)> {
    if sample_rate == 0 || hop == 0 || !(fmin > 0.0 && fmax > fmin) {
        return Err(Error::Msg("pitch_track: need sample_rate, hop > 0 and 0 < fmin < fmax".into()));
    }
    let q = (sample_rate / 8000).max(1);
    let sr = sample_rate as f64 / q as f64;
    // decimation with a triangular low-pass of 2q - 1 taps
    let n = wave.len() / q;
    let taps: Vec<f32> = (0..2 * q - 1).map(|i| (q as f32 - (i as f32 - (q as f32 - 1.0)).abs()) / (q * q) as f32).collect();
    let x: Vec<f32> = (0..n)
        .map(|i| {
            let c = (i * q) as i64;
            taps.iter().enumerate().map(|(k, w)| {
                let j = c + k as i64 - (q as i64 - 1);
                if j >= 0 && (j as usize) < wave.len() { w * wave[j as usize] } else { 0.0 }
            }).sum()
        })
        .collect();
    let tau_min = ((sr / fmax as f64).floor() as usize).max(2);
    let tau_max = ((sr / fmin as f64).ceil() as usize).max(tau_min + 2);
    let win = (tau_max * 2).max(256);
    let frames = wave.len() / hop + 1;
    let mut f0 = vec![0f32; frames];
    let mut conf = vec![0f32; frames];
    let workers = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1).min(frames.max(1));
    let per = frames.div_ceil(workers);
    std::thread::scope(|s| {
        for (w, (f0c, cc)) in f0.chunks_mut(per).zip(conf.chunks_mut(per)).enumerate() {
            let x = &x;
            s.spawn(move || {
                let mut d = vec![0f64; tau_max + 2];
                for (k, (fo, co)) in f0c.iter_mut().zip(cc.iter_mut()).enumerate() {
                    let t = w * per + k;
                    let centre = (t * hop) as i64 / q as i64;
                    let start = centre - (win / 2) as i64;
                    // difference function over `win` samples; samples outside the signal read as zero
                    let at = |i: i64| -> f64 { if i >= 0 && (i as usize) < x.len() { x[i as usize] as f64 } else { 0.0 } };
                    let energy: f64 = (0..win as i64).map(|j| at(start + j).powi(2)).sum::<f64>() / win as f64;
                    if energy < 1e-9 {
                        continue;
                    }
                    for tau in 1..=tau_max {
                        let mut acc = 0.0;
                        for j in 0..win as i64 {
                            let diff = at(start + j) - at(start + j + tau as i64);
                            acc += diff * diff;
                        }
                        d[tau] = acc;
                    }
                    // cumulative mean normalised difference
                    let mut dn = vec![1.0f64; tau_max + 2];
                    let mut run = 0.0;
                    for tau in 1..=tau_max {
                        run += d[tau];
                        dn[tau] = if run > 0.0 { d[tau] * tau as f64 / run } else { 1.0 };
                    }
                    // first dip under the threshold, descended to its local minimum; otherwise the global minimum
                    let thresh = 0.15;
                    let mut best = 0usize;
                    let mut tau = tau_min;
                    while tau < tau_max {
                        if dn[tau] < thresh {
                            while tau + 1 < tau_max && dn[tau + 1] < dn[tau] {
                                tau += 1;
                            }
                            best = tau;
                            break;
                        }
                        tau += 1;
                    }
                    if best == 0 {
                        best = (tau_min..tau_max).min_by(|a, b| dn[*a].partial_cmp(&dn[*b]).unwrap()).unwrap_or(tau_min);
                    }
                    let (a, b, c) = (dn[best - 1], dn[best], dn[best + 1]);
                    let denom = a - 2.0 * b + c;
                    let refined = if denom.abs() > 1e-12 { best as f64 + (a - c) / (2.0 * denom) } else { best as f64 };
                    *fo = (sr / refined.clamp(tau_min as f64, tau_max as f64)) as f32;
                    *co = (1.0 - dn[best]).clamp(0.0, 1.0) as f32;
                }
            });
        }
    });
    Ok((f0, conf))
}

// ---------------------------------------------------------------------------------------------------------------------
// Minimum-phase response of a magnitude envelope
// ---------------------------------------------------------------------------------------------------------------------

fn fft_in_place(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = 2.0 * PI / len as f64 * if inverse { 1.0 } else { -1.0 };
        let (wr, wi) = (ang.cos(), ang.sin());
        for start in (0..n).step_by(len) {
            let (mut cr, mut ci) = (1.0, 0.0);
            for k in 0..len / 2 {
                let (a, b) = (start + k, start + k + len / 2);
                let (tr, ti) = (re[b] * cr - im[b] * ci, re[b] * ci + im[b] * cr);
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                let nr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = nr;
            }
        }
        len <<= 1;
    }
    if inverse {
        for k in 0..n {
            re[k] /= n as f64;
            im[k] /= n as f64;
        }
    }
}

/// Phase (radians, unwrapped, one value per one-sided bin) of the minimum-phase filter whose magnitude is `mag`
/// (`n_fft / 2 + 1` bins): the folded real cepstrum of `ln mag`.
fn min_phase(mag: &[f32], n_fft: usize) -> Vec<f64> {
    let half = n_fft / 2;
    let mut re = vec![0f64; n_fft];
    let mut im = vec![0f64; n_fft];
    for k in 0..=half {
        let l = (mag[k].max(1e-8) as f64).ln();
        re[k] = l;
        if k > 0 && k < half {
            re[n_fft - k] = l;
        }
    }
    fft_in_place(&mut re, &mut im, true); // real cepstrum
    for k in 0..n_fft {
        // fold: c[0], 2 c[k] for 0 < k < n/2, c[n/2], 0 above
        let g = if k == 0 || k == half { 1.0 } else if k < half { 2.0 } else { 0.0 };
        re[k] *= g;
        im[k] = 0.0;
    }
    fft_in_place(&mut re, &mut im, false); // ln |H| + j phase
    (0..=half).map(|k| im[k]).collect()
}

// ---------------------------------------------------------------------------------------------------------------------
// Harmonic-plus-noise synthesis
// ---------------------------------------------------------------------------------------------------------------------

/// Parameters of [`harmonic_synth`].
#[derive(Clone, Debug)]
pub struct SynthOptions {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub hop: usize,
    /// Scale of the harmonic amplitudes against the envelope (an envelope averages the peaks and valleys of a spectrum).
    pub gain: f32,
    /// Noise kept in voiced frames above a few kHz (0 = clean harmonics up to the Nyquist frequency).
    pub breath: f32,
    /// Scale of the noise part (white noise shaped by an envelope comes out louder than the real unvoiced speech it models).
    pub noise_gain: f32,
    pub seed: u64,
}

impl SynthOptions {
    pub fn new(sample_rate: usize, n_fft: usize, hop: usize) -> Self {
        Self { sample_rate, n_fft, hop, gain: 1.9, breath: 0.4, noise_gain: 0.65, seed: 0 }
    }
}

fn splitmix(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
}

/// `noise` (length `len`) filtered by a real, per-frame, per-bin gain `w[k * frames + t]` in the STFT domain: a centred, reflect-padded
/// Hann STFT, the gain on the one-sided spectrum, the inverse transform, and weighted overlap-add normalised by the summed squared
/// window. Same result as `stft` -> `mul_real` -> `istft`, but frame by frame on all cores without building tensors.
fn shape_noise(noise: &[f64], w: &[f64], n: usize, hop: usize, frames: usize) -> Vec<f64> {
    let len = noise.len();
    let pad = n / 2;
    let win: Vec<f64> = (0..n).map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos()).collect(); // periodic Hann
    let at = |i: usize| -> f64 {
        // reflect padding (the edge sample is not repeated)
        let j = i as isize - pad as isize;
        let l = len as isize;
        let j = if j < 0 { -j } else if j >= l { 2 * (l - 1) - j } else { j };
        noise[j.clamp(0, l - 1) as usize]
    };
    let frame = |f: usize| -> Vec<f64> {
        let mut re: Vec<f64> = (0..n).map(|i| at(f * hop + i) * win[i]).collect();
        let mut im = vec![0f64; n];
        fft_in_place(&mut re, &mut im, false);
        for k in 0..=n / 2 {
            let g = w[k * frames + f];
            re[k] *= g;
            im[k] *= g;
        }
        im[0] = 0.0;
        im[n / 2] = 0.0;
        for k in 1..=(n - 1) / 2 {
            re[n - k] = re[k];
            im[n - k] = -im[k];
        }
        fft_in_place(&mut re, &mut im, true);
        (0..n).map(|i| re[i] * win[i]).collect()
    };
    #[cfg(feature = "rayon")]
    let parts: Vec<Vec<f64>> = {
        use rayon::prelude::*;
        (0..frames).into_par_iter().map(frame).collect()
    };
    #[cfg(not(feature = "rayon"))]
    let parts: Vec<Vec<f64>> = (0..frames).map(frame).collect();
    let total = n + hop * (frames - 1);
    let mut sig = vec![0f64; total];
    let mut env = vec![0f64; total];
    for (f, part) in parts.iter().enumerate() {
        for i in 0..n {
            sig[f * hop + i] += part[i];
            env[f * hop + i] += win[i] * win[i];
        }
    }
    (0..len).map(|j| sig[pad + j] / env[pad + j].max(1e-11)).collect()
}

/// Waveform of `(frames - 1) * hop` samples from per-frame `f0` (Hz), `voiced` in `[0, 1]` and magnitude envelope
/// `env[frames, n_fft / 2 + 1]` (linear STFT magnitude of a Hann window, power 1; frame `t` is centred on sample `t * hop`).
pub fn harmonic_synth(f0: &[f32], voiced: &[f32], env: &[f32], o: &SynthOptions) -> Result<Vec<f32>> {
    let frames = f0.len();
    let nf = o.n_fft / 2 + 1;
    if frames < 2 || voiced.len() != frames || env.len() != frames * nf {
        return Err(Error::Msg(format!("harmonic_synth: need f0 / voiced of {frames} (>= 2) frames and env of {frames} x {nf}")));
    }
    if !o.n_fft.is_power_of_two() {
        return Err(Error::Msg("harmonic_synth: n_fft must be a power of two".into()));
    }
    let sr = o.sample_rate as f64;
    let len = (frames - 1) * o.hop;
    // an unvoiced frame keeps the pitch of its nearest voiced neighbour so the phase does not jump when voicing starts
    let mut pitch: Vec<f64> = f0.iter().zip(voiced).map(|(f, v)| if *v > 0.0 && *f > 20.0 { *f as f64 } else { 0.0 }).collect();
    let mut last = 0.0;
    for p in pitch.iter_mut() {
        if *p > 0.0 { last = *p } else { *p = last }
    }
    let mut next = 0.0;
    for p in pitch.iter_mut().rev() {
        if *p > 0.0 { next = *p } else { *p = next }
    }
    for p in pitch.iter_mut() {
        if *p <= 0.0 { *p = 120.0 }
    }
    #[cfg(feature = "rayon")]
    let phase: Vec<Vec<f64>> = {
        use rayon::prelude::*;
        (0..frames).into_par_iter().map(|t| min_phase(&env[t * nf..(t + 1) * nf], o.n_fft)).collect()
    };
    #[cfg(not(feature = "rayon"))]
    let phase: Vec<Vec<f64>> = (0..frames).map(|t| min_phase(&env[t * nf..(t + 1) * nf], o.n_fft)).collect();
    let hz_per_bin = sr / o.n_fft as f64;
    let amp_scale = o.gain as f64 * 2.0 / (o.n_fft as f64 / 2.0); // a sinusoid of amplitude A gives a Hann-window peak of A * n_fft / 4
    let mut out = vec![0f32; len];
    // the fundamental's phase at every sample (a running sum, so it is the only serial part)
    let interp = |n: usize| -> (usize, f64, f64, f64) {
        let u = n as f64 / o.hop as f64;
        let t0 = (u.floor() as usize).min(frames - 2);
        let a = u - t0 as f64;
        (t0, a, pitch[t0] * (1.0 - a) + pitch[t0 + 1] * a, voiced[t0] as f64 * (1.0 - a) + voiced[t0 + 1] as f64 * a)
    };
    let mut phis = vec![0f64; len];
    let mut phi = 0.0f64;
    for (n, slot) in phis.iter_mut().enumerate() {
        phi += 2.0 * PI * interp(n).2 / sr;
        if phi > 2.0 * PI {
            phi -= 2.0 * PI;
        }
        *slot = phi;
    }
    let look = |t: usize, b: f64| -> (f64, f64) {
        let i = (b.floor() as usize).min(nf - 2);
        let fr = (b - i as f64).clamp(0.0, 1.0);
        let (a0, p0) = (env[t * nf + i], phase[t][i]);
        let (a1, p1) = (env[t * nf + i + 1], phase[t][i + 1]);
        (a0 as f64 * (1.0 - fr) + a1 as f64 * fr, p0 * (1.0 - fr) + p1 * fr)
    };
    // every sample is a sum of harmonics that depends only on its own position and phase, so they are computed in parallel
    let one = |n: usize| -> f32 {
        let (t0, a, f, v) = interp(n);
        if v <= 1e-3 {
            return 0.0;
        }
        let phi = phis[n];
        let mut acc = 0.0;
        let mut h = 1usize;
        while (h as f64) * f < 0.49 * sr {
            let b = (h as f64 * f / hz_per_bin).min((nf - 1) as f64 - 1e-6);
            let (m0, p0) = look(t0, b);
            let (m1, p1) = look(t0 + 1, b);
            let (mag, ph) = (m0 * (1.0 - a) + m1 * a, p0 * (1.0 - a) + p1 * a);
            // above a few kHz real voices are partly noise: fade the harmonics out as `breath` grows
            let hf = ((h as f64 * f - 2000.0) / 4000.0).clamp(0.0, 1.0);
            let keep = 1.0 - o.breath as f64 * hf;
            acc += mag * keep * (h as f64 * phi + ph).sin();
            h += 1;
        }
        (v * amp_scale * acc) as f32
    };
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        let chunk = o.hop.max(1) * 4;
        out.par_chunks_mut(chunk).enumerate().for_each(|(ci, c)| {
            for (i, x) in c.iter_mut().enumerate() {
                *x = one(ci * chunk + i);
            }
        });
    }
    #[cfg(not(feature = "rayon"))]
    for (n, x) in out.iter_mut().enumerate() {
        *x = one(n);
    }
    // noise part: white noise shaped by the envelope in the STFT domain, per frame and bin
    let mut state = o.seed ^ 0xD1B5_4A32_D192_ED03;
    let sqrt3 = 3f64.sqrt();
    let noise: Vec<f64> = (0..len).map(|_| (splitmix(&mut state) * 2.0 - 1.0) * sqrt3).collect();
    let norm = 1.0 / (0.886 * (3.0 * o.n_fft as f64 / 8.0).sqrt()); // E|X| of unit-variance white noise under a Hann window
    let mut w = vec![0f64; nf * frames];
    for t in 0..frames {
        let v = voiced[t] as f64;
        for k in 0..nf {
            let hz = k as f64 * hz_per_bin;
            let b = o.breath as f64 * (0.3 + 0.7 * ((hz - 2000.0) / 4000.0).clamp(0.0, 1.0));
            let share = ((1.0 - v) * (1.0 - v) + b * b * v * v).sqrt();
            w[k * frames + t] = env[t * nf + k] as f64 * share * norm * o.noise_gain as f64;
        }
    }
    let sv = shape_noise(&noise, &w, o.n_fft, o.hop, frames);
    for (o_, s) in out.iter_mut().zip(sv) {
        *o_ += s as f32;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------------------------------
// Forced-alignment boundary refinement
// ---------------------------------------------------------------------------------------------------------------------

/// Moves the phone boundaries of a forced alignment (each by at most `max_shift` frames) to where the features change.
///
/// `feats` is `frames x dim` (row-major), `durations` the frames of every phone (summing to `frames`). Returns the durations of
/// the segmentation into the same phones, in the same order, that minimises the squared deviation of every frame from its
/// phone's mean, plus `penalty` per frame a boundary moves. A phone may not end up shorter than `min(min_len, its old length)`;
/// zero-length phones (silence / punctuation tokens) stay zero. CTC aligners place boundaries a few frames off and leave many
/// 1-2 frame phones; an acoustic model trained on such timing learns blurred transitions. Exact dynamic programming over the
/// allowed boundary positions: O(phones * (2 * max_shift + 1)^2 * dim).
pub fn refine_boundaries(feats: &[f32], dim: usize, durations: &[i64], max_shift: usize, min_len: usize, penalty: f64) -> Result<Vec<i64>> {
    if dim == 0 || feats.len() % dim != 0 {
        return Err(Error::Msg("refine_boundaries: feats must be frames x dim".into()));
    }
    let t = feats.len() / dim;
    if durations.iter().any(|&d| d < 0) || durations.iter().sum::<i64>() as usize != t {
        return Err(Error::Msg(format!("refine_boundaries: durations must be >= 0 and sum to the {t} frames")));
    }
    let nz: Vec<usize> = (0..durations.len()).filter(|&i| durations[i] > 0).collect();
    let m = nz.len();
    if m <= 1 {
        return Ok(durations.to_vec());
    }
    let mut s1 = vec![0f64; (t + 1) * dim];
    let mut s2 = vec![0f64; t + 1];
    for i in 0..t {
        let mut q = 0.0;
        for k in 0..dim {
            let v = feats[i * dim + k] as f64;
            s1[(i + 1) * dim + k] = s1[i * dim + k] + v;
            q += v * v;
        }
        s2[i + 1] = s2[i] + q;
    }
    let cost = |a: usize, b: usize| -> f64 {
        let mut q = 0.0;
        for k in 0..dim {
            let d = s1[b * dim + k] - s1[a * dim + k];
            q += d * d;
        }
        (s2[b] - s2[a]) - q / (b - a) as f64
    };
    let mut b0 = vec![0usize; m + 1];
    for j in 0..m {
        b0[j + 1] = b0[j] + durations[nz[j]] as usize;
    }
    let cands: Vec<Vec<usize>> = (0..=m)
        .map(|j| {
            if j == 0 || j == m {
                return vec![b0[j]];
            }
            let lo = b0[j].saturating_sub(max_shift).max(j);
            let hi = (b0[j] + max_shift).min(t - (m - j));
            let mut c: Vec<usize> = (lo..=hi.max(lo)).collect();
            if !c.contains(&b0[j]) {
                c.push(b0[j]);
                c.sort_unstable();
            }
            c
        })
        .collect();
    let mut dp: Vec<Vec<f64>> = cands.iter().map(|c| vec![f64::INFINITY; c.len()]).collect();
    let mut back: Vec<Vec<usize>> = cands.iter().map(|c| vec![0; c.len()]).collect();
    dp[0][0] = 0.0;
    for j in 1..=m {
        let need = min_len.max(1).min(durations[nz[j - 1]] as usize);
        for (qi, &q) in cands[j].iter().enumerate() {
            let moved = (q as f64 - b0[j] as f64).abs() * penalty;
            for (pi, &p) in cands[j - 1].iter().enumerate() {
                if q < p + need || !dp[j - 1][pi].is_finite() {
                    continue;
                }
                let v = dp[j - 1][pi] + cost(p, q) + moved;
                if v < dp[j][qi] {
                    dp[j][qi] = v;
                    back[j][qi] = pi;
                }
            }
        }
    }
    let mut pos = vec![0usize; m + 1];
    let mut qi = 0;
    for j in (1..=m).rev() {
        pos[j] = cands[j][qi];
        qi = back[j][qi];
    }
    let mut out = durations.to_vec();
    for j in 0..m {
        out[nz[j]] = (pos[j + 1] - pos[j]) as i64;
    }
    Ok(out)
}

/// Band-limited pulse train for `rows` rows of `frames` per-frame pitches `f0` (Hz) -> `rows * frames * hop` samples of unit
/// power; the Rust twin of `oxideon.models.pulse_train` without jitter / shimmer (the training path). Every harmonic below
/// `limit * sample_rate` in closed form (Dirichlet kernel), pitch interpolated in the log domain between frame centres, the top
/// harmonic faded in and out; the phase accumulates in f64. `phase` (one per row, radians) offsets each row's start.
pub fn pulse_train(f0: &[f32], rows: usize, frames: usize, sample_rate: usize, hop: usize, limit: f64, phase: Option<&[f64]>) -> Result<Vec<f32>> {
    if f0.len() != rows * frames || hop == 0 || phase.is_some_and(|p| p.len() != rows) {
        return Err(Error::Msg("pulse_train: f0 must be rows * frames, phase one per row, hop > 0".into()));
    }
    let n = frames * hop;
    let sr = sample_rate as f64;
    let row = |r: usize, out: &mut [f32]| {
        let knot = |t: usize| (f0[r * frames + t.min(frames - 1)] as f64).max(1.0).ln();
        let (mut acc, off, tau) = (0f64, phase.map_or(0.0, |p| p[r]), 2.0 * PI);
        for (i, o) in out.iter_mut().enumerate() {
            let (low, w) = (i / hop, (i % hop) as f64 / hop as f64);
            let freq = (knot(low) * (1.0 - w) + knot(low + 1) * w).exp();
            acc += freq / sr;
            let theta = (tau * acc + off).rem_euclid(tau);
            let harm = (limit * sr / freq).max(1.0);
            let whole = harm.floor();
            let den = 2.0 * (theta / 2.0).sin();
            let core = if den.abs() < 1e-7 { whole } else { ((whole + 0.5) * theta).sin() / den - 0.5 };
            *o = ((core + (harm - whole) * ((whole + 1.0) * theta).cos()) * (2.0 / harm).sqrt()) as f32;
        }
    };
    let mut out = vec![0f32; rows * n];
    if n == 0 {
        return Ok(out);
    }
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(n).enumerate().for_each(|(r, o)| row(r, o));
    }
    #[cfg(not(feature = "rayon"))]
    out.chunks_mut(n).enumerate().for_each(|(r, o)| row(r, o));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shaped_noise_matches_the_tensor_stft_pipeline() {
        use crate::{backend::Device, dtype::DType, signal::{hann_window, istft, stft, StftOptions}, tensor::Tensor};
        let (n, hop, frames) = (64usize, 16usize, 12usize);
        let len = (frames - 1) * hop;
        let nf = n / 2 + 1;
        let mut state = 7u64;
        let noise: Vec<f64> = (0..len).map(|_| splitmix(&mut state) * 2.0 - 1.0).collect();
        let w: Vec<f64> = (0..nf * frames).map(|i| 0.2 + ((i * 7) % 10) as f64 / 10.0).collect();
        let got = shape_noise(&noise, &w, n, hop, frames);
        let mut so = StftOptions::new(n, hop);
        so.window = Some(hann_window(n, true, DType::F64).unwrap());
        let spec = stft(&Tensor::from_slice_on(&noise, &[len], &Device::Cpu).unwrap(), &so).unwrap();
        let wt = Tensor::from_slice_on(&w, &[nf, frames], &Device::Cpu).unwrap();
        let want = istft(&spec.mul_real(&wt).unwrap(), &so, Some(len)).unwrap().to_vec::<f64>().unwrap();
        assert_eq!(got.len(), want.len());
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!((a - b).abs() < 1e-9, "sample {i}: {a} vs {b}");
        }
    }

    #[test]
    fn yin_finds_the_pitch_of_a_harmonic_tone() {
        let sr = 24000usize;
        let f = 187.0f64;
        let x: Vec<f32> = (0..sr).map(|n| (1..6).map(|h| (2.0 * PI * f * h as f64 * n as f64 / sr as f64).sin() / h as f64).sum::<f64>() as f32).collect();
        let (f0, conf) = pitch_track(&x, sr, 256, 60.0, 500.0).unwrap();
        let mid = f0.len() / 2;
        assert!((f0[mid] as f64 - f).abs() < 2.0, "{} vs {f}", f0[mid]);
        assert!(conf[mid] > 0.9, "{}", conf[mid]);
        // silence is unvoiced
        let (_, c) = pitch_track(&vec![0.0; 4096], sr, 256, 60.0, 500.0).unwrap();
        assert!(c.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn refine_boundaries_snaps_to_feature_changes_and_keeps_empty_phones() {
        // three phones with constant, distinct features; true boundaries at 10 and 25 of 40 frames
        let dim = 3;
        let feats: Vec<f32> = (0..40).flat_map(|i| { let v = if i < 10 { 0.0 } else if i < 25 { 2.0 } else { -1.5 }; vec![v, v * 0.5, 1.0 - v] }).collect();
        let got = refine_boundaries(&feats, dim, &[0, 12, 11, 0, 17], 4, 2, 0.01).unwrap();
        assert_eq!(got, vec![0, 10, 15, 0, 15]);
        // out of reach (shift limit) -> stays as close as allowed
        assert_eq!(refine_boundaries(&feats, dim, &[16, 9, 15], 2, 2, 0.01).unwrap(), vec![14, 11, 15]);
        // featureless input: nothing moves (the penalty keeps the old timing)
        assert_eq!(refine_boundaries(&vec![1.0; 40 * dim], dim, &[12, 11, 17], 4, 2, 0.01).unwrap(), vec![12, 11, 17]);
        assert!(refine_boundaries(&feats, dim, &[12, 11], 4, 2, 0.0).is_err());
    }

    #[test]
    fn min_phase_of_a_flat_envelope_is_zero() {
        let p = min_phase(&vec![1.0; 513], 1024);
        assert!(p.iter().all(|v| v.abs() < 1e-9));
    }

    #[test]
    fn synthesised_harmonics_follow_the_pitch_and_envelope() {
        let (sr, n_fft, hop, frames) = (24000usize, 1024usize, 256usize, 40usize);
        let nf = n_fft / 2 + 1;
        let f0 = vec![200.0f32; frames];
        let voiced = vec![1.0f32; frames];
        let env: Vec<f32> = (0..frames * nf).map(|i| 1.0 / (1.0 + (i % nf) as f32 / 20.0)).collect();
        let mut o = SynthOptions::new(sr, n_fft, hop);
        o.breath = 0.0;
        let y = harmonic_synth(&f0, &voiced, &env, &o).unwrap();
        assert_eq!(y.len(), (frames - 1) * hop);
        let (p, c) = pitch_track(&y, sr, hop, 60.0, 500.0).unwrap();
        assert!((p[frames / 2] - 200.0).abs() < 3.0, "{}", p[frames / 2]);
        assert!(c[frames / 2] > 0.8);
        assert!(y.iter().all(|v| v.is_finite()));
    }
}
