//! Cross-validation batch 42: the pure-Rust Hugging Face tokenizer pipeline vs the `tokenizers` library on trained
//! fixtures of every family (tools/crossval/check_batch42.py, fixtures from make_tokenizer_fixtures.py).
use oxideon_core::tokenizer::hf::{Direction, Encoding, Padding, PaddingStrategy, Tokenizer, Truncation, TruncationStrategy};
use oxideon_core::tokenizer::json::Json;

fn js(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}
fn arr<T: ToString>(v: &[T]) -> String {
    format!("[{}]", v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","))
}
fn opt(v: &[Option<impl ToString>]) -> String {
    format!("[{}]", v.iter().map(|w| w.as_ref().map_or("null".to_string(), |x| x.to_string())).collect::<Vec<_>>().join(","))
}
fn dump(name: &str, e: &Encoding, text: &str, t: &Tokenizer) {
    let off: Vec<String> = e.char_offsets(text).iter().map(|(a, b)| format!("[{a},{b}]")).collect();
    let toks: Vec<String> = e.tokens.iter().map(|s| js(s)).collect();
    let dec = t.decode(&e.ids, false).unwrap();
    let dec_skip = t.decode(&e.ids, true).unwrap();
    println!(
        "{name} {{\"ids\":{},\"tokens\":[{}],\"offsets\":[{}],\"type_ids\":{},\"special\":{},\"attn\":{},\"words\":{},\"seqs\":{},\"dec\":{},\"dec_skip\":{}}}",
        arr(&e.ids), toks.join(","), off.join(","), arr(&e.type_ids), arr(&e.special_tokens_mask), arr(&e.attention_mask), opt(&e.word_ids), opt(&e.sequence_ids), js(&dec), js(&dec_skip)
    );
}

#[test]
#[ignore]
fn emit_tokenizers() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools/crossval/tokenizer_fixtures");
    let texts = Json::parse(&std::fs::read_to_string(format!("{dir}/texts.json")).unwrap()).unwrap();
    let names = ["gpt2_bpe", "roberta_bpe", "bert_wordpiece", "bert_cased", "unigram_metaspace", "llama_bpe", "wordlevel", "cl100k_like", "pipeline_wp"];
    for f in names {
        let t = match Tokenizer::from_file(format!("{dir}/{f}.json")) {
            Ok(t) => t,
            Err(e) => {
                println!("{f}.load {{\"err\":{}}}", js(&format!("{e:?}")));
                continue;
            }
        };
        for (i, tx) in texts.get("texts").unwrap().as_array().unwrap().iter().enumerate() {
            let tx = tx.as_str().unwrap();
            for special in [true, false] {
                match t.encode(tx, special) {
                    Ok(e) => dump(&format!("{f}.t{i}.s{}", special as u8), &e, tx, &t),
                    Err(err) => println!("{f}.t{i}.s{} {{\"err\":{}}}", special as u8, js(&format!("{err:?}"))),
                }
            }
        }
        for (i, p) in texts.get("pairs").unwrap().as_array().unwrap().iter().enumerate() {
            let (a, b) = (p.as_array().unwrap()[0].as_str().unwrap(), p.as_array().unwrap()[1].as_str().unwrap());
            match t.encode_pair(a, b, true) {
                Ok(e) => println!("{f}.p{i} {{\"ids\":{},\"type_ids\":{},\"special\":{},\"seqs\":{},\"words\":{}}}", arr(&e.ids), arr(&e.type_ids), arr(&e.special_tokens_mask), opt(&e.sequence_ids), opt(&e.word_ids)),
                Err(err) => println!("{f}.p{i} {{\"err\":{}}}", js(&format!("{err:?}"))),
            }
        }
    }
    // truncation + padding + batch on BERT
    let mut t = Tokenizer::from_file(format!("{dir}/bert_wordpiece.json")).unwrap();
    t.with_truncation(Some(Truncation { max_length: 9, strategy: TruncationStrategy::LongestFirst, stride: 2, direction: Direction::Right }));
    t.with_padding(Some(Padding { strategy: PaddingStrategy::BatchLongest, direction: Direction::Left, pad_to_multiple_of: Some(4), pad_id: 3, pad_type_id: 0, pad_token: "[PAD]".into() }));
    let batch = ["Hello, world! It's a test of truncation and padding.", "short", "The quick brown fox jumps over the lazy dog."];
    let encs = t.encode_batch(&batch, true).unwrap();
    for (i, e) in encs.iter().enumerate() {
        println!(
            "trunc.b{i} {{\"ids\":{},\"attn\":{},\"special\":{},\"overflow\":[{}]}}",
            arr(&e.ids), arr(&e.attention_mask), arr(&e.special_tokens_mask),
            e.overflowing.iter().map(|o| arr(&o.ids)).collect::<Vec<_>>().join(",")
        );
    }
    let pe = t.encode_pair("Hello, world! It's a test of truncation and padding.", "The quick brown fox jumps over", true).unwrap();
    println!("trunc.pair {{\"ids\":{},\"type_ids\":{},\"attn\":{}}}", arr(&pe.ids), arr(&pe.type_ids), arr(&pe.attention_mask));
}
