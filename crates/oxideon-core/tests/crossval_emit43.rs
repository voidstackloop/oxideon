//! Cross-validation batch 43: real model tokenizers (downloaded with tools/crossval/fetch_real_tokenizers.py into
//! target/real_tokenizers) vs the `tokenizers` library (tools/crossval/check_batch43.py).
use oxideon_core::tokenizer::hf::{Encoding, Tokenizer};
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
fn dump(name: &str, e: &Encoding, text: &str, t: &Tokenizer) {
    let off: Vec<String> = e.char_offsets(text).iter().map(|(a, b)| format!("[{a},{b}]")).collect();
    println!("{name} {{\"ids\":{},\"offsets\":[{}],\"type_ids\":{},\"special\":{},\"dec\":{},\"dec_skip\":{}}}", arr(&e.ids), off.join(","), arr(&e.type_ids), arr(&e.special_tokens_mask), js(&t.decode(&e.ids, false).unwrap()), js(&t.decode(&e.ids, true).unwrap()));
}

#[test]
#[ignore]
fn emit_real_tokenizers() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let texts = Json::parse(&std::fs::read_to_string(format!("{root}/tools/crossval/tokenizer_fixtures/real_texts.json")).unwrap()).unwrap();
    let mut files: Vec<_> = std::fs::read_dir(format!("{root}/target/real_tokenizers")).unwrap().filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "json")).collect();
    files.sort();
    for p in files {
        let name = p.file_stem().unwrap().to_string_lossy().to_string();
        let t = match Tokenizer::from_file(&p) {
            Ok(t) => t,
            Err(e) => {
                println!("{name}.load {{\"err\":{}}}", js(&format!("{e:?}")));
                continue;
            }
        };
        for (i, tx) in texts.get("texts").unwrap().as_array().unwrap().iter().enumerate() {
            let tx = tx.as_str().unwrap();
            for special in [true, false] {
                match t.encode(tx, special) {
                    Ok(e) => dump(&format!("{name}.t{i}.s{}", special as u8), &e, tx, &t),
                    Err(err) => println!("{name}.t{i}.s{} {{\"err\":{}}}", special as u8, js(&format!("{err:?}"))),
                }
            }
        }
        if let (Some(a), Some(b)) = (texts.get("pairs").and_then(|p| p.as_array()), texts.get("texts").and_then(|p| p.as_array())) {
            let _ = (a, b);
        }
        for (i, p) in texts.get("pairs").unwrap().as_array().unwrap().iter().enumerate() {
            let (a, b) = (p.as_array().unwrap()[0].as_str().unwrap(), p.as_array().unwrap()[1].as_str().unwrap());
            match t.encode_pair(a, b, true) {
                Ok(e) => println!("{name}.p{i} {{\"ids\":{},\"type_ids\":{},\"special\":{}}}", arr(&e.ids), arr(&e.type_ids), arr(&e.special_tokens_mask)),
                Err(err) => println!("{name}.p{i} {{\"err\":{}}}", js(&format!("{err:?}"))),
            }
        }
    }
}
