//! Mel-spectrogram audio feature extraction for speech foundation models.
//!
//! Converts raw 16kHz audio waveforms into 80 or 128 channel log-Mel spectrograms
//! matching OpenAI Whisper and modern speech recognition frontends.

use crate::error::Result;
use crate::tensor::Tensor;

/// Constructs triangular Mel-scale filterbank weighting matrix `[n_mels, n_freqs]`.
pub fn mel_filterbank_matrix(
    n_mels: usize,
    n_fft: usize,
    sample_rate: usize,
    f_min: f64,
    f_max: f64,
) -> Result<Tensor> {
    let n_freqs = n_fft / 2 + 1;
    crate::limits::check_repeats("n_mels", n_mels)?;
    crate::limits::check_elements("mel filterbank", &[n_mels, n_freqs])?;
    let hz_to_mel = |hz: f64| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f64| 700.0 * (10.0f64.powf(mel / 2595.0) - 1.0);

    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    let mut mel_points = Vec::with_capacity(n_mels + 2);
    for i in 0..=(n_mels + 1) {
        let frac = i as f64 / (n_mels + 1) as f64;
        mel_points.push(mel_min + frac * (mel_max - mel_min));
    }

    let mut bin_indices = Vec::with_capacity(n_mels + 2);
    for m in &mel_points {
        let hz = mel_to_hz(*m);
        let bin = ((n_fft + 1) as f64 * hz / sample_rate as f64).floor() as usize;
        bin_indices.push(bin.min(n_freqs - 1));
    }

    let mut weights = crate::layout::try_filled(n_mels * n_freqs, 0.0f32)?;
    for m in 0..n_mels {
        let left = bin_indices[m];
        let center = bin_indices[m + 1];
        let right = bin_indices[m + 2];

        for f in left..center {
            if center > left {
                weights[m * n_freqs + f] = (f - left) as f32 / (center - left) as f32;
            }
        }
        for f in center..right {
            if right > center {
                weights[m * n_freqs + f] = (right - f) as f32 / (right - center) as f32;
            }
        }
    }

    Tensor::from_slice(&weights, &[n_mels, n_freqs])
}

/// Mel-spectrogram audio feature extractor.
pub struct MelSpectrogram {
    pub n_mels: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub filterbank: Tensor,
}

impl MelSpectrogram {
    /// Creates a Mel spectrogram extractor matching standard Whisper settings (16kHz audio, n_mels=80, n_fft=400, hop=160).
    pub fn new(n_mels: usize, n_fft: usize, hop_length: usize, sample_rate: usize) -> Result<Self> {
        if n_mels == 0 || n_fft == 0 || hop_length == 0 || sample_rate == 0 {
            return Err(crate::error::Error::Msg(format!(
                "MelSpectrogram needs positive n_mels/n_fft/hop_length/sample_rate, got {n_mels}/{n_fft}/{hop_length}/{sample_rate}"
            )));
        }
        let filterbank = mel_filterbank_matrix(n_mels, n_fft, sample_rate, 0.0, 8000.0)?;
        Ok(Self {
            n_mels,
            n_fft,
            hop_length,
            filterbank,
        })
    }

    /// Transforms raw audio waveform `[batch, time]` or `[time]` into log-Mel spectrogram `[batch, n_mels, frames]`.
    pub fn forward(&self, waveform: &Tensor) -> Result<Tensor> {
        let is_1d = waveform.ndim() == 1;
        let x = if is_1d {
            waveform.unsqueeze(0)?
        } else {
            waveform.clone()
        };

        // STFT: [batch, num_frames, n_fft, 2] (complex real/imag)
        let stft_out = x.stft(self.n_fft, self.hop_length, true)?;
        let n_freqs = self.n_fft / 2 + 1;
        let real = stft_out.select(3, 0)?.narrow(2, 0, n_freqs)?;
        let imag = stft_out.select(3, 1)?.narrow(2, 0, n_freqs)?;

        // Power spectrogram = real^2 + imag^2: [batch, num_frames, n_freqs]
        let power = real.square()?.add(&imag.square()?)?;

        let b = power.shape()[0];
        let frames = power.shape()[1];

        // Apply Mel filterbank: [n_mels, n_freqs] @ [batch, frames, n_freqs].T -> [batch, n_mels, frames]
        let power_flat = power.reshape(&[b * frames, n_freqs])?;
        let mel_spec = power_flat
            .matmul(&self.filterbank.t()?)?
            .reshape(&[b, frames, self.n_mels])?
            .permute(&[0, 2, 1])?;

        // Log compression: log10(clamp(mel, 1e-5))
        let clamped = mel_spec.clamp(1e-5, f64::INFINITY)?;
        let log_mel = clamped.ln()?.mul_scalar(std::f64::consts::LOG10_E)?;

        if is_1d {
            log_mel.select(0, 0)
        } else {
            Ok(log_mel)
        }
    }
}
