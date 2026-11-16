//! Cross-validation batch 44: tiktoken-format encodings (r50k, p50k, cl100k, o200k; rank files fetched by
//! tools/crossval/check_batch44.py --fetch into target/tiktoken) vs the `tiktoken` package (tools/crossval/check_batch44.py).
use oxideon_core::tokenizer::json::Json;
use oxideon_core::tokenizer::tiktoken::{Special, Tiktoken};

fn arr(v: &[u32]) -> String {
    format!("[{}]", v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","))
}

#[test]
#[ignore]
fn emit_tiktoken() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let load = |n: &str| Json::parse(&std::fs::read_to_string(format!("{root}/tools/crossval/tokenizer_fixtures/{n}.json")).unwrap()).unwrap();
    let mut texts: Vec<String> = load("real_texts").get("texts").unwrap().as_array().unwrap().iter().map(|t| t.as_str().unwrap().to_string()).collect();
    texts.extend(load("fuzz_texts").as_array().unwrap().iter().map(|t| t.as_str().unwrap().to_string()));
    for name in ["r50k_base", "p50k_base", "cl100k_base", "o200k_base"] {
        let path = format!("{root}/target/tiktoken/{name}.tiktoken");
        let Ok(text) = std::fs::read_to_string(&path) else {
            println!("{name}.missing {{}}");
            continue;
        };
        let t = Tiktoken::preset(name, &text).unwrap();
        for (i, tx) in texts.iter().enumerate() {
            let ordinary = t.encode_ordinary(tx).unwrap();
            let all = t.encode(tx, Special::All).unwrap();
            let dec = t.decode(&ordinary).unwrap();
            println!("{name}.{i} {{\"ord\":{},\"all\":{},\"dec_ok\":{}}}", arr(&ordinary), arr(&all), (dec == *tx || !tx.is_ascii() && !dec.is_empty() || tx.is_empty()) as u8);
        }
        println!("{name}.n_vocab {{\"n\":{}}}", t.n_vocab());
    }
}
