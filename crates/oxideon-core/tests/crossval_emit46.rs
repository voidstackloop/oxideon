//! Cross-validation batch 46: BPE / WordPiece / WordLevel training vs the `tokenizers` trainers
//! (tools/crossval/check_batch46.py).
use oxideon_core::tokenizer::hf::pretokenizer::byte_to_char;
use oxideon_core::tokenizer::hf::trainer::{BpeTrainer, WordLevelTrainer};
use oxideon_core::tokenizer::hf::Tokenizer;
use oxideon_core::tokenizer::json::Json;

fn corpus() -> Vec<String> {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let j = Json::parse(&std::fs::read_to_string(format!("{root}/tools/crossval/tokenizer_fixtures/train_corpus.json")).unwrap()).unwrap();
    j.as_array().unwrap().iter().map(|t| t.as_str().unwrap().to_string()).collect()
}

fn pre(s: &str) -> Json {
    Json::parse(s).unwrap()
}

fn show(name: &str, t: &Tokenizer) {
    let j = t.to_json();
    println!("{name} {}", j.get("model").unwrap().to_text());
    println!("{name}.added {}", j.get("added_tokens").unwrap().to_text());
}

#[test]
#[ignore]
fn emit_training() {
    let c = corpus();
    let bytes: Vec<char> = (0..=255u8).map(byte_to_char).collect();

    let mut t = Tokenizer::untrained("BPE", None).unwrap();
    t.set_pre_tokenizer_json(&pre(r#"{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}"#)).unwrap();
    t.train_bpe(&c, &BpeTrainer { vocab_size: 500, special_tokens: vec!["<|endoftext|>".into()], initial_alphabet: bytes, ..Default::default() }).unwrap();
    show("bpe_bytelevel", &t);

    let mut t = Tokenizer::untrained("BPE", Some("[UNK]")).unwrap();
    t.set_pre_tokenizer_json(&pre(r#"{"type":"Whitespace"}"#)).unwrap();
    t.train_bpe(&c, &BpeTrainer { vocab_size: 300, min_frequency: 2, special_tokens: vec!["[UNK]".into()], ..Default::default() }).unwrap();
    show("bpe_ws", &t);

    let mut t = Tokenizer::untrained("BPE", Some("[UNK]")).unwrap();
    t.set_pre_tokenizer_json(&pre(r#"{"type":"Whitespace"}"#)).unwrap();
    t.train_bpe(&c, &BpeTrainer { vocab_size: 300, special_tokens: vec!["[UNK]".into()], end_of_word_suffix: Some("</w>".into()), ..Default::default() }).unwrap();
    show("bpe_suffix", &t);

    let mut t = Tokenizer::untrained("BPE", Some("[UNK]")).unwrap();
    t.set_pre_tokenizer_json(&pre(r#"{"type":"Whitespace"}"#)).unwrap();
    t.train_bpe(&c, &BpeTrainer { vocab_size: 250, special_tokens: vec!["[UNK]".into()], limit_alphabet: Some(30), ..Default::default() }).unwrap();
    show("bpe_limit", &t);

    let mut t = Tokenizer::untrained("WordLevel", Some("[UNK]")).unwrap();
    t.set_pre_tokenizer_json(&pre(r#"{"type":"Whitespace"}"#)).unwrap();
    t.train_word_level(&c, &WordLevelTrainer { vocab_size: 80, min_frequency: 2, special_tokens: vec!["[UNK]".into(), "[PAD]".into()] }).unwrap();
    show("wordlevel", &t);

    let mut t = Tokenizer::untrained("BPE", Some("[UNK]")).unwrap();
    t.set_normalizer_json(&pre(r#"{"type":"BertNormalizer","clean_text":true,"handle_chinese_chars":true,"strip_accents":null,"lowercase":true}"#)).unwrap();
    t.set_pre_tokenizer_json(&pre(r#"{"type":"BertPreTokenizer"}"#)).unwrap();
    t.train_bpe(&c, &BpeTrainer { vocab_size: 300, special_tokens: vec!["[UNK]".into()], ..Default::default() }).unwrap();
    show("bpe_bert", &t);

    let mut t = Tokenizer::untrained("WordPiece", Some("[UNK]")).unwrap();
    t.set_normalizer_json(&pre(r#"{"type":"BertNormalizer","clean_text":true,"handle_chinese_chars":true,"strip_accents":null,"lowercase":true}"#)).unwrap();
    t.set_pre_tokenizer_json(&pre(r#"{"type":"BertPreTokenizer"}"#)).unwrap();
    t.train_bpe(&c, &BpeTrainer { special_tokens: vec!["[UNK]".into(), "[CLS]".into(), "[SEP]".into()], ..BpeTrainer::word_piece(300) }).unwrap();
    show("wordpiece", &t);
    // the trained tokenizer must be usable end to end
    let e = t.encode("Hello, world! naïve café", false).unwrap();
    println!("wordpiece.encode [{}]", e.ids.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
}
