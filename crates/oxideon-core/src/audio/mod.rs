//! Audio processing and speech foundation models (Whisper / Speech-to-Text).

pub mod mel;
pub mod rvq;
pub mod whisper;

pub use mel::{MelSpectrogram, mel_filterbank_matrix};
pub use rvq::{RVQConfig, RVQOutput, ResidualVectorQuantizer, VQCodebookStage};
pub use whisper::{AudioEncoder, Whisper, WhisperConfig};
