//! Unified Interleaved Multimodal Tokenizer (Text, Vision, and Audio).
//!
//! Maps arbitrary interleaved sequences of text, vision patch tokens, and RVQ acoustic tokens
//! into a unified discrete sequence with modality tracking masks.

use crate::error::Result;
use crate::tokenizer::BpeTokenizer;

/// The modality category of an interleaved token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModalityKind {
    Text,
    Vision,
    Audio,
}

/// A segment of tokens belonging to a single modality.
#[derive(Debug, Clone)]
pub struct OmniTokenSegment {
    pub modality: ModalityKind,
    pub token_ids: Vec<usize>,
}

/// An interleaved multimodal token sequence.
#[derive(Debug, Clone)]
pub struct OmniSequence {
    pub token_ids: Vec<usize>,
    pub modalities: Vec<ModalityKind>,
}

impl OmniSequence {
    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }
}

/// Unified Multimodal Tokenizer.
#[derive(Debug, Clone)]
pub struct OmniTokenizer {
    pub bpe: BpeTokenizer,
    pub image_start_id: usize,
    pub image_end_id: usize,
    pub audio_start_id: usize,
    pub audio_end_id: usize,
    pub vision_offset: usize,
    pub audio_offset: usize,
}

impl Default for OmniTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl OmniTokenizer {
    pub fn new() -> Self {
        let bpe = BpeTokenizer::new_byte_level();
        let base_vocab = 50_000;

        let image_start_id = base_vocab;
        let image_end_id = base_vocab + 1;
        let audio_start_id = base_vocab + 2;
        let audio_end_id = base_vocab + 3;

        let vision_offset = base_vocab + 100;
        let audio_offset = base_vocab + 10_000;

        Self {
            bpe,
            image_start_id,
            image_end_id,
            audio_start_id,
            audio_end_id,
            vision_offset,
            audio_offset,
        }
    }

    /// Encodes UTF-8 text into a Text segment.
    pub fn encode_text(&self, text: &str) -> OmniTokenSegment {
        let ids = self.bpe.encode(text);
        OmniTokenSegment {
            modality: ModalityKind::Text,
            token_ids: ids,
        }
    }

    /// Wraps discrete visual patch token IDs into a Vision segment with `<image_start>` and `<image_end>`.
    ///
    /// Patch IDs must fit the vision range below `audio_offset` so vision and
    /// audio IDs can never collide.
    pub fn encode_vision_patches(&self, patch_ids: &[usize]) -> Result<OmniTokenSegment> {
        if self.audio_offset <= self.vision_offset {
            return Err(crate::error::Error::Msg(
                "OmniTokenizer: audio_offset must exceed vision_offset".into(),
            ));
        }
        let max_patch = self.audio_offset - self.vision_offset - 1;
        let mut ids = Vec::with_capacity(patch_ids.len() + 2);
        ids.push(self.image_start_id);
        for &p in patch_ids {
            if p > max_patch {
                return Err(crate::error::Error::Msg(format!(
                    "vision patch id {p} collides with the audio id range (max {max_patch})"
                )));
            }
            ids.push(self.vision_offset + p);
        }
        ids.push(self.image_end_id);

        Ok(OmniTokenSegment {
            modality: ModalityKind::Vision,
            token_ids: ids,
        })
    }

    /// Wraps discrete audio RVQ token IDs into an Audio segment with `<audio_start>` and `<audio_end>`.
    pub fn encode_audio_tokens(&self, audio_tokens: &[usize]) -> Result<OmniTokenSegment> {
        let mut ids = Vec::with_capacity(audio_tokens.len() + 2);
        ids.push(self.audio_start_id);
        for &a in audio_tokens {
            let id = self.audio_offset.checked_add(a).ok_or_else(|| {
                crate::error::Error::Msg(format!("audio token id {a} overflows the id space"))
            })?;
            ids.push(id);
        }
        ids.push(self.audio_end_id);

        Ok(OmniTokenSegment {
            modality: ModalityKind::Audio,
            token_ids: ids,
        })
    }

    /// Interleaves multiple multimodal segments into a single unified sequence.
    pub fn interleave(&self, segments: &[OmniTokenSegment]) -> Result<OmniSequence> {
        let mut token_ids = Vec::new();
        let mut modalities = Vec::new();

        for seg in segments {
            for &tok in &seg.token_ids {
                token_ids.push(tok);
                modalities.push(seg.modality);
            }
        }

        Ok(OmniSequence {
            token_ids,
            modalities,
        })
    }
}
