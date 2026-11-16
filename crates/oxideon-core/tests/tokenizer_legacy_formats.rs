//! Legacy file formats (`vocab.json` + `merges.txt`, `vocab.txt`) must tokenize exactly like the matching `tokenizer.json`.
//! Needs the files fetched by tools/crossval/fetch_real_tokenizers.py (`--legacy`) into target/; skipped when absent.
use oxideon_core::tokenizer::hf::Tokenizer;
use oxideon_core::tokenizer::json::Json;

fn read(root: &str, rel: &str) -> Option<String> {
    std::fs::read_to_string(format!("{root}/{rel}")).ok()
}

#[test]
fn legacy_files_match_tokenizer_json() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let texts = Json::parse(&read(root, "tools/crossval/tokenizer_fixtures/real_texts.json").unwrap()).unwrap();
    let texts: Vec<&str> = texts.get("texts").unwrap().as_array().unwrap().iter().map(|t| t.as_str().unwrap()).collect();
    let mut checked = 0;
    // GPT-2
    if let (Some(v), Some(m), Some(j)) = (read(root, "target/legacy/gpt2_vocab.json"), read(root, "target/legacy/gpt2_merges.txt"), read(root, "target/real_tokenizers/openai-community__gpt2.json")) {
        let legacy = Tokenizer::from_vocab_and_merges(&v, &m, false, &["<|endoftext|>"], false).unwrap();
        let full = Tokenizer::from_str(&j).unwrap();
        for t in &texts {
            assert_eq!(legacy.encode(t, true).unwrap().ids, full.encode(t, true).unwrap().ids, "gpt2: {t:?}");
        }
        checked += 1;
    }
    // RoBERTa
    if let (Some(v), Some(m), Some(j)) = (read(root, "target/legacy/roberta_vocab.json"), read(root, "target/legacy/roberta_merges.txt"), read(root, "target/real_tokenizers/FacebookAI__roberta-base.json")) {
        let legacy = Tokenizer::from_vocab_and_merges(&v, &m, false, &["<s>", "<pad>", "</s>", "<unk>", "<mask>"], true).unwrap();
        let full = Tokenizer::from_str(&j).unwrap();
        for t in &texts {
            assert_eq!(legacy.encode(t, true).unwrap().ids, full.encode(t, true).unwrap().ids, "roberta: {t:?}");
        }
        checked += 1;
    }
    // BERT uncased and cased
    for (vocab, json, lower) in [("target/legacy/bert_vocab.txt", "target/real_tokenizers/google-bert__bert-base-uncased.json", true), ("target/legacy/bertcased_vocab.txt", "target/real_tokenizers/google-bert__bert-base-cased.json", false)] {
        if let (Some(v), Some(j)) = (read(root, vocab), read(root, json)) {
            let legacy = Tokenizer::from_vocab_txt(&v, lower, None).unwrap();
            let full = Tokenizer::from_str(&j).unwrap();
            for t in &texts {
                assert_eq!(legacy.encode(t, true).unwrap().ids, full.encode(t, true).unwrap().ids, "{vocab}: {t:?}");
            }
            checked += 1;
        }
    }
    eprintln!("legacy format comparisons run: {checked}");
}

#[test]
fn save_and_reload_round_trip() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let Some(j) = read(root, "tools/crossval/tokenizer_fixtures/llama_bpe.json") else { return };
    let mut t = Tokenizer::from_str(&j).unwrap();
    t.add_special_tokens(&["<|extra|>"]);
    let path = std::env::temp_dir().join("oxideon_tok_roundtrip.json");
    t.save(&path).unwrap();
    let back = Tokenizer::from_file(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let text = "Hello <|extra|> world, naïve café 日本語";
    let (a, b) = (t.encode(text, true).unwrap(), back.encode(text, true).unwrap());
    assert_eq!(a.ids, b.ids);
    assert_eq!(a.offsets, b.offsets);
    assert!(a.tokens.iter().any(|x| x == "<|extra|>"));
    assert_eq!(t.decode(&a.ids, false).unwrap(), back.decode(&b.ids, false).unwrap());
}
