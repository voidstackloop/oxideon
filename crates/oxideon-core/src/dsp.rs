//! Digital signal processing on `&[f64]`: [`lfilter`] / [`filtfilt`], FIR design ([`firwin`]), Butterworth IIR design
//! ([`butter`], low/high-pass via the bilinear transform), [`freqz_mag`], [`savgol_filter`], [`medfilt`], [`detrend`],
//! [`find_peaks`], and spectral estimates ([`periodogram`], [`welch`]).

use crate::error::{Error, Result};

type C = (f64, f64);

fn cmul(a: C, b: C) -> C {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn cdiv(a: C, b: C) -> C {
    let d = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
}

/// Direct-form II transposed IIR/FIR filter: `a[0] y[n] = sum b[k] x[n-k] - sum_{k>=1} a[k] y[n-k]`.
pub fn lfilter(b: &[f64], a: &[f64], x: &[f64]) -> Result<Vec<f64>> {
    if b.is_empty() || a.is_empty() || a[0] == 0.0 {
        return Err(Error::Msg("lfilter: need non-empty b and a with a[0] != 0".into()));
    }
    let n = b.len().max(a.len());
    let norm = |v: &[f64]| -> Vec<f64> { (0..n).map(|i| v.get(i).copied().unwrap_or(0.0) / a[0]).collect() };
    let (b, a) = (norm(b), norm(a));
    let mut z = vec![0.0; n];
    Ok(x.iter()
        .map(|&xn| {
            let y = b[0] * xn + z[0];
            for i in 0..n - 1 {
                z[i] = z[i + 1] + b[i + 1] * xn - a[i + 1] * y;
            }
            y
        })
        .collect())
}

/// Zero-phase filtering: forward then backward pass with odd-extension padding to tame edge transients.
pub fn filtfilt(b: &[f64], a: &[f64], x: &[f64]) -> Result<Vec<f64>> {
    let pad = (3 * b.len().max(a.len())).min(x.len().saturating_sub(1));
    if x.len() < 2 {
        return Err(Error::Msg("filtfilt: need at least 2 samples".into()));
    }
    let first = x[0];
    let last = x[x.len() - 1];
    let mut ext: Vec<f64> = (1..=pad).rev().map(|i| 2.0 * first - x[i]).collect();
    ext.extend_from_slice(x);
    ext.extend((1..=pad).map(|i| 2.0 * last - x[x.len() - 1 - i]));
    let fwd = lfilter(b, a, &ext)?;
    let rev: Vec<f64> = fwd.into_iter().rev().collect();
    let back = lfilter(b, a, &rev)?;
    let out: Vec<f64> = back.into_iter().rev().collect();
    Ok(out[pad..pad + x.len()].to_vec())
}

/// Windowed-sinc FIR design (Hamming window). `cutoff` values are fractions of Nyquist in `(0, 1)`:
/// one value for low-pass (or high-pass with `highpass`), two for band-pass.
pub fn firwin(taps: usize, cutoff: &[f64], highpass: bool) -> Result<Vec<f64>> {
    if taps < 3 || cutoff.is_empty() || cutoff.len() > 2 || cutoff.iter().any(|&c| !(c > 0.0 && c < 1.0)) || (cutoff.len() == 2 && cutoff[0] >= cutoff[1]) {
        return Err(Error::Msg("firwin: need taps >= 3 and 1 or 2 increasing cutoffs in (0, 1)".into()));
    }
    if highpass && cutoff.len() == 1 && taps % 2 == 0 {
        return Err(Error::Msg("firwin: a high-pass filter needs an odd number of taps".into()));
    }
    let m = (taps - 1) as f64 / 2.0;
    let sinc = |x: f64| if x == 0.0 { 1.0 } else { (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x) };
    // ideal response: low-pass, band-pass, or high-pass (all-pass minus low-pass), then windowed
    let mut h: Vec<f64> = (0..taps)
        .map(|i| {
            let t = i as f64 - m;
            let w = 0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (taps - 1) as f64).cos();
            let ideal = match (cutoff.len(), highpass) {
                (2, _) => cutoff[1] * sinc(cutoff[1] * t) - cutoff[0] * sinc(cutoff[0] * t),
                (_, true) => sinc(t) - cutoff[0] * sinc(cutoff[0] * t),
                _ => cutoff[0] * sinc(cutoff[0] * t),
            };
            ideal * w
        })
        .collect();
    // unit gain at DC (low-pass), band centre (band-pass) or Nyquist (high-pass)
    let freq = if cutoff.len() == 2 { 0.5 * (cutoff[0] + cutoff[1]) } else if highpass { 1.0 } else { 0.0 };
    let gain: f64 = h.iter().enumerate().map(|(i, v)| v * (std::f64::consts::PI * freq * (i as f64 - m)).cos()).sum();
    h.iter_mut().for_each(|v| *v /= gain);
    Ok(h)
}

/// Butterworth IIR filter of `order` with cutoff `wn` (fraction of Nyquist in `(0, 1)`); returns `(b, a)`.
pub fn butter(order: usize, wn: f64, highpass: bool) -> Result<(Vec<f64>, Vec<f64>)> {
    if order == 0 || order > 12 || !(wn > 0.0 && wn < 1.0) {
        return Err(Error::Msg("butter: need 1 <= order <= 12 and 0 < wn < 1".into()));
    }
    // pre-warped analog cutoff for the bilinear transform with fs = 2
    let w = 4.0 * (std::f64::consts::PI * wn / 2.0).tan();
    let poles: Vec<C> = (0..order)
        .map(|k| {
            let th = std::f64::consts::PI * (2 * k + order + 1) as f64 / (2 * order) as f64;
            (w * th.cos(), w * th.sin())
        })
        .collect();
    // analog low-pass prototype -> optional high-pass (s -> w/s), then bilinear z = (4 + s) / (4 - s)
    let (apoles, azeros): (Vec<C>, Vec<C>) = if highpass { (poles.iter().map(|&p| cdiv((w * w, 0.0), p)).collect(), vec![(0.0, 0.0); order]) } else { (poles.clone(), Vec::new()) };
    let bil = |s: C| -> C { cdiv((4.0 + s.0, s.1), (4.0 - s.0, -s.1)) };
    let zp: Vec<C> = apoles.iter().map(|&p| bil(p)).collect();
    let mut zz: Vec<C> = azeros.iter().map(|&z| bil(z)).collect();
    while zz.len() < order {
        zz.push((-1.0, 0.0));
    }
    let poly = |roots: &[C]| -> Vec<f64> {
        let mut c: Vec<C> = vec![(1.0, 0.0)];
        for &r in roots {
            let mut next = vec![(0.0, 0.0); c.len() + 1];
            for (i, &ci) in c.iter().enumerate() {
                next[i].0 += ci.0;
                next[i].1 += ci.1;
                let t = cmul(ci, r);
                next[i + 1].0 -= t.0;
                next[i + 1].1 -= t.1;
            }
            c = next;
        }
        c.iter().map(|v| v.0).collect()
    };
    let (mut b, a) = (poly(&zz), poly(&zp));
    // scale for unit gain at DC (low-pass) or Nyquist (high-pass)
    let zr = if highpass { -1.0 } else { 1.0 };
    let eval = |p: &[f64]| -> f64 { p.iter().enumerate().map(|(i, c)| c * f64::powi(zr, i as i32)).sum::<f64>() };
    let g = eval(&a) / eval(&b);
    b.iter_mut().for_each(|v| *v *= g);
    Ok((b, a))
}

/// Magnitude response `|H(e^{jw})|` at `n` evenly spaced frequencies in `[0, pi]`.
pub fn freqz_mag(b: &[f64], a: &[f64], n: usize) -> Vec<f64> {
    (0..n)
        .map(|k| {
            let w = std::f64::consts::PI * k as f64 / (n.max(2) - 1) as f64;
            let ev = |p: &[f64]| -> C { p.iter().enumerate().fold((0.0, 0.0), |acc, (i, c)| (acc.0 + c * (w * i as f64).cos(), acc.1 - c * (w * i as f64).sin())) };
            let (nb, na) = (ev(b), ev(a));
            let h = cdiv(nb, na);
            (h.0 * h.0 + h.1 * h.1).sqrt()
        })
        .collect()
}

fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[p][c].abs() < 1e-14 {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        for r in c + 1..n {
            let w = a[r][c] / a[c][c];
            for k in c..n {
                a[r][k] -= w * a[c][k];
            }
            b[r] -= w * b[c];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = (b[i] - (i + 1..n).map(|k| a[i][k] * x[k]).sum::<f64>()) / a[i][i];
    }
    Some(x)
}

/// Savitzky-Golay smoothing (or the `deriv`-th derivative) with an odd `window` and polynomial `order`.
/// Edges use the polynomial fitted to the first/last window.
pub fn savgol_filter(x: &[f64], window: usize, order: usize, deriv: usize) -> Result<Vec<f64>> {
    if window % 2 == 0 || window < 3 || order >= window || deriv > order || x.len() < window {
        return Err(Error::Msg("savgol_filter: need an odd window >= 3, order < window, deriv <= order and enough samples".into()));
    }
    let half = (window / 2) as i64;
    // weights that evaluate the deriv-th derivative at offset `at` of the least-squares polynomial over the window
    let weights = |at: f64| -> Option<Vec<f64>> {
        let n = order + 1;
        let vander: Vec<Vec<f64>> = (-half..=half).map(|t| (0..n).map(|p| (t as f64).powi(p as i32)).collect()).collect();
        let ata: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| vander.iter().map(|r| r[i] * r[j]).sum()).collect()).collect();
        // derivative of t^p at `at`: p!/(p-deriv)! at^(p-deriv)
        let basis: Vec<f64> = (0..n).map(|p| if p < deriv { 0.0 } else { (p - deriv + 1..=p).product::<usize>() as f64 * at.powi((p - deriv) as i32) }).collect();
        let sol = solve(ata, basis)?; // (A^T A)^-1 basis
        Some(vander.iter().map(|row| row.iter().zip(&sol).map(|(a, b)| a * b).sum()).collect())
    };
    let apply = |w: &[f64], start: usize| -> f64 { w.iter().enumerate().map(|(i, c)| c * x[start + i]).sum() };
    let mid = weights(0.0).ok_or_else(|| Error::Msg("savgol_filter: singular design".into()))?;
    let mut out = vec![0.0; x.len()];
    for i in 0..x.len() {
        out[i] = if i as i64 >= half && (i as i64) < x.len() as i64 - half {
            apply(&mid, i - half as usize)
        } else if (i as i64) < half {
            apply(&weights(i as f64 - half as f64).ok_or_else(|| Error::Msg("savgol_filter: singular design".into()))?, 0)
        } else {
            let start = x.len() - window;
            apply(&weights(i as f64 - (start as f64 + half as f64)).ok_or_else(|| Error::Msg("savgol_filter: singular design".into()))?, start)
        };
    }
    Ok(out)
}

/// Running median with an odd `window` (edges shrink the window symmetrically).
pub fn medfilt(x: &[f64], window: usize) -> Result<Vec<f64>> {
    if window % 2 == 0 || window == 0 {
        return Err(Error::Msg("medfilt: window must be odd".into()));
    }
    let h = window / 2;
    Ok((0..x.len())
        .map(|i| {
            let (lo, hi) = (i.saturating_sub(h), (i + h + 1).min(x.len()));
            let mut w = x[lo..hi].to_vec();
            w.sort_by(f64::total_cmp);
            w[w.len() / 2]
        })
        .collect())
}

/// Removes the least-squares line (or the mean when `linear` is false).
pub fn detrend(x: &[f64], linear: bool) -> Vec<f64> {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    if !linear || x.len() < 2 {
        return x.iter().map(|v| v - mean).collect();
    }
    let tm = (n - 1.0) / 2.0;
    let slope = x.iter().enumerate().map(|(i, v)| (i as f64 - tm) * (v - mean)).sum::<f64>() / (0..x.len()).map(|i| (i as f64 - tm).powi(2)).sum::<f64>();
    x.iter().enumerate().map(|(i, v)| v - mean - slope * (i as f64 - tm)).collect()
}

/// Indices of local maxima that are at least `height` high and at least `distance` samples apart (taller peaks win).
pub fn find_peaks(x: &[f64], height: f64, distance: usize) -> Vec<usize> {
    let mut cand: Vec<usize> = Vec::new();
    for i in 1..x.len().saturating_sub(1) {
        // plateau-aware: strictly rises into the point, and does not rise after the plateau
        let mut j = i;
        while j + 1 < x.len() && x[j + 1] == x[i] {
            j += 1;
        }
        if x[i] > x[i - 1] && j + 1 < x.len() && x[j + 1] < x[i] && x[i] >= height {
            cand.push((i + j) / 2);
        }
    }
    cand.dedup();
    let mut order = cand.clone();
    order.sort_by(|&a, &b| x[b].total_cmp(&x[a]));
    let mut keep = vec![true; cand.len()];
    for &p in &order {
        let pos = cand.iter().position(|&c| c == p).unwrap();
        if !keep[pos] {
            continue;
        }
        for (k, &c) in cand.iter().enumerate() {
            if k != pos && keep[k] && c.abs_diff(p) < distance.max(1) {
                keep[k] = false;
            }
        }
    }
    cand.into_iter().zip(keep).filter(|(_, k)| *k).map(|(c, _)| c).collect()
}

fn dft_power(seg: &[f64], window: &[f64]) -> Vec<f64> {
    let n = seg.len();
    let xw: Vec<f64> = seg.iter().zip(window).map(|(a, b)| a * b).collect();
    (0..=n / 2)
        .map(|k| {
            let (mut re, mut im) = (0.0, 0.0);
            for (t, v) in xw.iter().enumerate() {
                let ang = -2.0 * std::f64::consts::PI * (k * t) as f64 / n as f64;
                re += v * ang.cos();
                im += v * ang.sin();
            }
            re * re + im * im
        })
        .collect()
}

/// One-sided power spectral density of the whole signal (Hann window, density scaling): `(freqs, psd)`.
pub fn periodogram(x: &[f64], fs: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    welch(x, fs, x.len(), 0)
}

/// Welch's averaged-periodogram PSD with Hann windows, segment length `nperseg` and `noverlap` samples of overlap.
pub fn welch(x: &[f64], fs: f64, nperseg: usize, noverlap: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    if nperseg < 4 || nperseg > x.len() || noverlap >= nperseg || fs <= 0.0 {
        return Err(Error::Msg("welch: need 4 <= nperseg <= len, noverlap < nperseg and fs > 0".into()));
    }
    let window: Vec<f64> = (0..nperseg).map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / nperseg as f64).cos()).collect();
    let scale = 1.0 / (fs * window.iter().map(|w| w * w).sum::<f64>());
    let step = nperseg - noverlap;
    let mut acc = vec![0.0; nperseg / 2 + 1];
    let mut count = 0.0;
    let mut start = 0;
    while start + nperseg <= x.len() {
        let mean = x[start..start + nperseg].iter().sum::<f64>() / nperseg as f64;
        let seg: Vec<f64> = x[start..start + nperseg].iter().map(|v| v - mean).collect();
        dft_power(&seg, &window).iter().zip(acc.iter_mut()).for_each(|(p, a)| *a += p);
        count += 1.0;
        start += step;
    }
    let last = acc.len() - 1;
    let psd: Vec<f64> = acc.iter().enumerate().map(|(k, v)| v * scale / count * if k == 0 || (nperseg % 2 == 0 && k == last) { 1.0 } else { 2.0 }).collect();
    let freqs = (0..acc.len()).map(|k| k as f64 * fs / nperseg as f64).collect();
    Ok((freqs, psd))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(n: usize, f: f64, fs: f64) -> Vec<f64> {
        (0..n).map(|i| (2.0 * std::f64::consts::PI * f * i as f64 / fs).sin()).collect()
    }

    fn rms(x: &[f64]) -> f64 {
        (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt()
    }

    #[test]
    fn lfilter_matches_hand_computation() {
        // y[n] = x[n] + 0.5 y[n-1]  (impulse response 1, 0.5, 0.25, ...)
        let y = lfilter(&[1.0], &[1.0, -0.5], &[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(y, vec![1.0, 0.5, 0.25, 0.125]);
        // 3-tap moving average FIR
        let m = lfilter(&[1.0 / 3.0; 3], &[1.0], &[3.0, 3.0, 3.0, 3.0]).unwrap();
        assert!((m[0] - 1.0).abs() < 1e-12 && (m[2] - 3.0).abs() < 1e-12);
        assert!(lfilter(&[1.0], &[0.0], &[1.0]).is_err());
    }

    #[test]
    fn butterworth_design_properties() {
        // scipy.signal.butter(2, 0.2): b = [0.0674553, 0.1349106, 0.0674553], a = [1, -1.1429805, 0.4128016]
        let (b, a) = butter(2, 0.2, false).unwrap();
        for (got, want) in b.iter().zip([0.067_455_3, 0.134_910_6, 0.067_455_3]) {
            assert!((got - want).abs() < 1e-6, "{b:?}");
        }
        for (got, want) in a.iter().zip([1.0, -1.142_980_5, 0.412_801_6]) {
            assert!((got - want).abs() < 1e-6, "{a:?}");
        }
        let mag = freqz_mag(&b, &a, 1001);
        assert!((mag[0] - 1.0).abs() < 1e-9 && mag[1000] < 1e-3);
        // -3 dB at the cutoff (0.2 of Nyquist -> index 200)
        assert!((mag[200] - 1.0 / 2f64.sqrt()).abs() < 5e-3, "{}", mag[200]);
        let (bh, ah) = butter(4, 0.3, true).unwrap();
// scipy.signal.butter(4, 0.3, 'high') (this check caught a wrong analog high-pass pole scaling)        for (got, want) in bh.iter().zip([0.275_413_288_072_304, -1.101_653_152_289_217, 1.652_479_728_433_826, -1.101_653_152_289_217, 0.275_413_288_072_304]) {            assert!((got - want).abs() < 1e-9, "{bh:?}");        }        assert!((ah[1] + 1.570_398_851_228_172).abs() < 1e-9 && (ah[4] - 0.076_197_064_61).abs() < 1e-9, "{ah:?}");
        let mh = freqz_mag(&bh, &ah, 1001);
        assert!(mh[0] < 1e-6 && (mh[1000] - 1.0).abs() < 1e-6);
        assert!(butter(0, 0.5, false).is_err() && butter(2, 1.5, false).is_err());
    }

    #[test]
    fn filters_pass_and_reject_tones() {
        let fs = 1000.0;
        let x: Vec<f64> = sine(2000, 10.0, fs).iter().zip(sine(2000, 200.0, fs)).map(|(a, b)| a + b).collect();
        let (b, a) = butter(4, 0.1, false).unwrap(); // 50 Hz low-pass
        let y = filtfilt(&b, &a, &x).unwrap();
        let clean = sine(2000, 10.0, fs);
        let err: Vec<f64> = y.iter().zip(&clean).skip(200).take(1600).map(|(p, q)| p - q).collect();
        assert!(rms(&err) < 0.02, "residual {}", rms(&err));
        // zero phase: no lag on the 10 Hz tone, unlike one-pass filtering
        let one_pass = lfilter(&b, &a, &x).unwrap();
        let lag_err: Vec<f64> = one_pass.iter().zip(&clean).skip(200).take(1600).map(|(p, q)| p - q).collect();
        assert!(rms(&lag_err) > rms(&err));
        // FIR low/high/band
        let lp = firwin(101, &[0.2], false).unwrap();
        assert!((lp.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        let hp = firwin(101, &[0.2], true).unwrap();
        let hp_mag = freqz_mag(&hp, &[1.0], 1001);
        assert!(hp_mag[0] < 0.01 && (hp_mag[1000] - 1.0).abs() < 1e-9, "{} {}", hp_mag[0], hp_mag[1000]); // unit gain at Nyquist, as scipy.signal.firwin
        let bp = firwin(101, &[0.3, 0.5], false).unwrap();
        let mag = freqz_mag(&bp, &[1.0], 1001);
        assert!((mag[400] - 1.0).abs() < 0.02 && mag[50] < 0.02 && mag[900] < 0.02);
        assert!(firwin(100, &[0.2], true).is_err() && firwin(2, &[0.2], false).is_err());
    }

    #[test]
    fn smoothing_and_peaks() {
        // savgol reproduces a quadratic exactly (value and derivative), including at the edges
        let x: Vec<f64> = (0..20).map(|i| 0.5 * (i as f64).powi(2) - 3.0 * i as f64 + 1.0).collect();
        let s = savgol_filter(&x, 7, 2, 0).unwrap();
        assert!(s.iter().zip(&x).all(|(a, b)| (a - b).abs() < 1e-8));
        let d = savgol_filter(&x, 7, 2, 1).unwrap();
        assert!(d.iter().enumerate().all(|(i, v)| (v - (i as f64 - 3.0)).abs() < 1e-8), "{d:?}");
        assert!(savgol_filter(&x, 4, 2, 0).is_err());
        // median filter removes a spike
        assert_eq!(medfilt(&[1.0, 1.0, 100.0, 1.0, 1.0], 3).unwrap(), vec![1.0, 1.0, 1.0, 1.0, 1.0]);
        assert!(medfilt(&[1.0], 2).is_err());
        let d = detrend(&(0..10).map(|i| 2.0 * i as f64 + 5.0).collect::<Vec<_>>(), true);
        assert!(d.iter().all(|v| v.abs() < 1e-9));
        assert!(detrend(&[1.0, 2.0, 3.0], false).iter().sum::<f64>().abs() < 1e-12);
        // peaks: respects height and distance
        let sig = [0.0, 1.0, 0.0, 3.0, 0.0, 2.0, 2.0, 0.0, 0.5, 0.0];
        assert_eq!(find_peaks(&sig, 0.0, 1), vec![1, 3, 5, 8]);
        assert_eq!(find_peaks(&sig, 1.5, 1), vec![3, 5]);
        assert_eq!(find_peaks(&sig, 0.0, 3), vec![3, 8]);
    }

    #[test]
    fn spectral_estimates_find_the_tone() {
        let fs = 256.0;
        let x = sine(1024, 32.0, fs);
        let (f, p) = welch(&x, fs, 256, 128).unwrap();
        let peak = (0..p.len()).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap();
        assert!((f[peak] - 32.0).abs() < 1.0);
        // Parseval: integrating the PSD recovers the signal power (1/2 for a unit sine)
        let power: f64 = p.iter().sum::<f64>() * (f[1] - f[0]);
        assert!((power - 0.5).abs() < 0.05, "{power}");
        let (pf, pp) = periodogram(&x, fs).unwrap();
        assert!((pf[(0..pp.len()).max_by(|&a, &b| pp[a].total_cmp(&pp[b])).unwrap()] - 32.0).abs() < 1.0);
        assert!(welch(&x, fs, 5000, 0).is_err() && welch(&x, fs, 64, 64).is_err());
    }
}
