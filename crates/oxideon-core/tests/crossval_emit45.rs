//! Cross-validation batch 45: SentencePiece `.model` files (fetched by tools/crossval/check_batch45.py --fetch into
//! target/spm) vs the `sentencepiece` package (tools/crossval/check_batch45.py).
use oxideon_core::tokenizer::json::Json;
use oxideon_core::tokenizer::sentencepiece::SentencePiece;

fn arr(v: &[u32]) -> String {
    format!("[{}]", v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","))
}
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

#[test]
#[ignore]
fn emit_sentencepiece() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let load = |n: &str| Json::parse(&std::fs::read_to_string(format!("{root}/tools/crossval/tokenizer_fixtures/{n}.json")).unwrap()).unwrap();
    let mut texts: Vec<String> = load("real_texts").get("texts").unwrap().as_array().unwrap().iter().map(|t| t.as_str().unwrap().to_string()).collect();
    texts.extend(load("fuzz_texts").as_array().unwrap().iter().map(|t| t.as_str().unwrap().to_string()));
    let mut files: Vec<_> = std::fs::read_dir(format!("{root}/target/spm")).unwrap().filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "model")).collect();
    files.sort();
    for p in files {
        let name = p.file_stem().unwrap().to_string_lossy().to_string();
        let sp = match SentencePiece::from_file(&p) {
            Ok(s) => s,
            Err(e) => {
                println!("{name}.load {{\"err\":{}}}", js(&format!("{e:?}")));
                continue;
            }
        };
        println!("{name}.info {{\"n\":{}}}", sp.vocab_size());
        for (i, tx) in texts.iter().enumerate() {
            match sp.encode(tx) {
                Ok(ids) => {
                    let dec = sp.decode(&ids).unwrap_or_default();
                    println!("{name}.{i} {{\"ids\":{},\"norm\":{},\"dec\":{}}}", arr(&ids), js(&sp.normalize(tx)), js(&dec));
                }
                Err(e) => println!("{name}.{i} {{\"err\":{}}}", js(&format!("{e:?}"))),
            }
        }
    }
}
