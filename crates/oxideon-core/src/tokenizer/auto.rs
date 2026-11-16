//! One entry point over every supported tokenizer format: a model directory (or a single file) is inspected and loaded as a
//! Hugging Face `tokenizer.json`, a SentencePiece `.model`, GPT-2 style `vocab.json` + `merges.txt`, or BERT style
//! `vocab.txt`; a tiktoken encoding can be wrapped explicitly. `tokenizer_config.json` supplies the special tokens, the
//! chat template and the maximum length.

use super::chat_template::{ChatTemplate, Val};
use super::hf::Tokenizer;
use super::json::Json;
use super::sentencepiece::SentencePiece;
use super::tiktoken::{Special, Tiktoken};
use crate::error::{Error, Result};
use std::path::Path;

pub enum Backend {
    Hf(Box<Tokenizer>),
    Spm(SentencePiece),
    Tiktoken(Tiktoken),
}

#[derive(Debug, Clone, Default)]
pub struct TokenizerConfig {
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub unk_token: Option<String>,
    pub pad_token: Option<String>,
    pub chat_template: Option<String>,
    pub model_max_length: Option<usize>,
    pub add_bos_token: Option<bool>,
    pub add_eos_token: Option<bool>,
}

fn token_field(j: &Json, key: &str) -> Option<String> {
    match j.get(key)? {
        Json::Str(s) => Some(s.clone()),
        o @ Json::Obj(_) => o.get("content").and_then(Json::as_str).map(str::to_string),
        _ => None,
    }
}

impl TokenizerConfig {
    pub fn from_json(j: &Json) -> Self {
        // the chat template is a string, or a list of {name, template} (the "default" one is used)
        let chat_template = match j.get("chat_template") {
            Some(Json::Str(s)) => Some(s.clone()),
            Some(Json::Arr(a)) => a
                .iter()
                .find(|t| t.get("name").and_then(Json::as_str) == Some("default"))
                .or_else(|| a.first())
                .and_then(|t| t.get("template").and_then(Json::as_str))
                .map(str::to_string),
            _ => None,
        };
        Self {
            bos_token: token_field(j, "bos_token"),
            eos_token: token_field(j, "eos_token"),
            unk_token: token_field(j, "unk_token"),
            pad_token: token_field(j, "pad_token"),
            chat_template,
            // some files carry a huge sentinel (1e30) meaning "no limit"
            model_max_length: j.get("model_max_length").and_then(Json::as_f64).filter(|&n| n > 0.0 && n < 1e15).map(|n| n as usize),
            add_bos_token: j.get("add_bos_token").and_then(Json::as_bool),
            add_eos_token: j.get("add_eos_token").and_then(Json::as_bool),
        }
    }
}

pub struct AutoTokenizer {
    pub backend: Backend,
    pub config: TokenizerConfig,
}

fn read(path: &Path) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

impl AutoTokenizer {
    /// Loads whatever tokenizer files the directory holds (preference: `tokenizer.json`, `tokenizer.model`,
    /// `vocab.json` + `merges.txt`, `vocab.txt`).
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let d = dir.as_ref();
        let config = read(&d.join("tokenizer_config.json")).and_then(|t| Json::parse(&t).ok()).map(|j| TokenizerConfig::from_json(&j)).unwrap_or_default();
        let class = read(&d.join("tokenizer_config.json")).and_then(|t| Json::parse(&t).ok()).and_then(|j| j.get("tokenizer_class").and_then(Json::as_str).map(str::to_string)).unwrap_or_default();
        let backend = if d.join("tokenizer.json").exists() {
            Backend::Hf(Box::new(Tokenizer::from_file(d.join("tokenizer.json"))?))
        } else if let Some(m) = ["tokenizer.model", "spiece.model", "sentencepiece.bpe.model", "spm.model"].iter().map(|f| d.join(f)).find(|p| p.exists()) {
            Backend::Spm(SentencePiece::from_file(m)?)
        } else if let (Some(v), Some(m)) = (read(&d.join("vocab.json")), read(&d.join("merges.txt"))) {
            let roberta = class.to_lowercase().contains("roberta") || class.to_lowercase().contains("bart");
            let specials: Vec<&str> = [config.bos_token.as_deref(), config.eos_token.as_deref(), config.unk_token.as_deref(), config.pad_token.as_deref(), Some("<mask>")].into_iter().flatten().collect();
            Backend::Hf(Box::new(Tokenizer::from_vocab_and_merges(&v, &m, false, &specials, roberta)?))
        } else if let Some(v) = read(&d.join("vocab.txt")) {
            let lower = std::fs::read_to_string(d.join("tokenizer_config.json")).ok().and_then(|t| Json::parse(&t).ok()).and_then(|j| j.get("do_lower_case").and_then(Json::as_bool)).unwrap_or(!class.contains("Cased"));
            Backend::Hf(Box::new(Tokenizer::from_vocab_txt(&v, lower, None)?))
        } else {
            return Err(Error::Msg(format!("tokenizer: no tokenizer files found in {}", d.display())));
        };
        Ok(Self { backend, config })
    }

    /// Loads one file: `*.json` (tokenizer.json), `*.model` / `*.spm` (SentencePiece).
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let p = path.as_ref();
        let backend = match p.extension().and_then(|e| e.to_str()) {
            Some("json") => Backend::Hf(Box::new(Tokenizer::from_file(p)?)),
            Some("model") | Some("spm") => Backend::Spm(SentencePiece::from_file(p)?),
            _ => return Err(Error::Msg("tokenizer: unsupported file type (expected .json, .model or .spm)".into())),
        };
        Ok(Self { backend, config: TokenizerConfig::default() })
    }

    pub fn from_tiktoken(t: Tiktoken) -> Self {
        Self { backend: Backend::Tiktoken(t), config: TokenizerConfig::default() }
    }

    /// Token ids of `text`. With `add_special_tokens` the model's usual specials are added (BOS / EOS or the
    /// post-processor's template); tiktoken text containing special token strings encodes them.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        match &self.backend {
            Backend::Hf(t) => Ok(t.encode(text, add_special_tokens)?.ids),
            Backend::Spm(s) => {
                let bos = add_special_tokens && self.config.add_bos_token.unwrap_or(true);
                let eos = add_special_tokens && self.config.add_eos_token.unwrap_or(false);
                s.encode_with_special(text, bos, eos)
            }
            Backend::Tiktoken(t) => t.encode(text, if add_special_tokens { Special::All } else { Special::Ignore }),
        }
    }

    pub fn encode_batch(&self, texts: &[&str], add_special_tokens: bool) -> Result<Vec<Vec<u32>>> {
        texts.iter().map(|t| self.encode(t, add_special_tokens)).collect()
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        match &self.backend {
            Backend::Hf(t) => t.decode(ids, skip_special_tokens),
            Backend::Spm(s) => {
                // control pieces (BOS, EOS, PAD) are dropped by the decoder itself; unknown pieces stay visible
                let _ = skip_special_tokens;
                s.decode(ids)
            }
            Backend::Tiktoken(t) => {
                if skip_special_tokens {
                    let special: std::collections::HashSet<u32> = t.special_tokens().values().copied().collect();
                    let kept: Vec<u32> = ids.iter().copied().filter(|i| !special.contains(i)).collect();
                    t.decode(&kept)
                } else {
                    t.decode(ids)
                }
            }
        }
    }

    pub fn vocab_size(&self) -> usize {
        match &self.backend {
            Backend::Hf(t) => t.get_vocab_size(true),
            Backend::Spm(s) => s.vocab_size(),
            Backend::Tiktoken(t) => t.n_vocab(),
        }
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        match &self.backend {
            Backend::Hf(t) => t.token_to_id(token),
            Backend::Spm(s) => s.piece_to_id(token),
            Backend::Tiktoken(t) => t.special_tokens().get(token).copied(),
        }
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        match &self.backend {
            Backend::Hf(t) => t.id_to_token(id),
            Backend::Spm(s) => s.id_to_piece(id).map(str::to_string),
            Backend::Tiktoken(t) => t.token_bytes(id).map(|b| String::from_utf8_lossy(&b).into_owned()),
        }
    }

    /// Renders the model's chat template (`tokenizer_config.json` / `chat_template.jinja`) over `messages`, a JSON array of
    /// `{"role": ..., "content": ...}` objects. `tools` is the optional JSON list of tool definitions and `extra` any
    /// further template variables (for example `enable_thinking`).
    pub fn apply_chat_template(&self, messages: &Json, tools: Option<&Json>, add_generation_prompt: bool, extra: &[(&str, Json)]) -> Result<String> {
        let src = self.config.chat_template.as_deref().ok_or_else(|| Error::Msg("tokenizer: this tokenizer has no chat template".into()))?;
        let tok = |o: &Option<String>| o.clone().map_or(Val::None, Val::Str);
        let mut vars: Vec<(&str, Val)> = vec![
            ("messages", Val::from_json(messages)),
            ("add_generation_prompt", Val::Bool(add_generation_prompt)),
            ("bos_token", tok(&self.config.bos_token)),
            ("eos_token", tok(&self.config.eos_token)),
            ("unk_token", tok(&self.config.unk_token)),
            ("pad_token", tok(&self.config.pad_token)),
        ];
        if let Some(t) = tools {
            vars.push(("tools", Val::from_json(t)));
        }
        for (k, v) in extra {
            vars.push((k, Val::from_json(v)));
        }
        ChatTemplate::new(src)?.render(&vars)
    }

    pub fn bos_token_id(&self) -> Option<u32> {
        self.config.bos_token.as_deref().and_then(|t| self.token_to_id(t))
    }
    pub fn eos_token_id(&self) -> Option<u32> {
        self.config.eos_token.as_deref().and_then(|t| self.token_to_id(t))
    }
    pub fn pad_token_id(&self) -> Option<u32> {
        self.config.pad_token.as_deref().and_then(|t| self.token_to_id(t))
    }
    pub fn unk_token_id(&self) -> Option<u32> {
        self.config.unk_token.as_deref().and_then(|t| self.token_to_id(t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_a_bert_vocab_directory() {
        let dir = std::env::temp_dir().join("oxideon_auto_tok_test");
        let _ = std::fs::create_dir_all(&dir);
        std::fs::write(dir.join("vocab.txt"), "[PAD]
[UNK]
[CLS]
[SEP]
[MASK]
hello
world
##s
").unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), r#"{"do_lower_case": true, "unk_token": "[UNK]", "cls_token": {"content": "[CLS]"}, "model_max_length": 1e30, "chat_template": "{{ x }}"}"#).unwrap();
        let t = AutoTokenizer::from_dir(&dir).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(t.encode("Hello worlds", true).unwrap(), [2, 5, 6, 7, 3]);
        assert_eq!(t.decode(&[5, 6, 7], true).unwrap(), "hello worlds");
        assert_eq!(t.unk_token_id(), Some(1));
        assert_eq!(t.config.model_max_length, None);
        assert_eq!(t.config.chat_template.as_deref(), Some("{{ x }}"));
        assert_eq!(t.vocab_size(), 8);
    }
}
