//! Cross-validation batch 47: chat templates of real models (fetched by tools/crossval/fetch_chat_templates.py into
//! target/chat_configs) rendered by the Rust Jinja subset vs jinja2 with the transformers environment
//! (tools/crossval/check_batch47.py).
use oxideon_core::tokenizer::auto::TokenizerConfig;
use oxideon_core::tokenizer::chat_template::{ChatTemplate, Val};
use oxideon_core::tokenizer::json::Json;

fn js(s: &str) -> String {
    Json::Str(s.to_string()).to_text()
}

#[test]
#[ignore]
fn emit_chat_templates() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let cases = Json::parse(&std::fs::read_to_string(format!("{root}/tools/crossval/tokenizer_fixtures/chat_cases.json")).unwrap()).unwrap();
    let mut files: Vec<_> = std::fs::read_dir(format!("{root}/target/chat_configs")).unwrap().filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "json")).collect();
    files.sort();
    for p in files {
        let name = p.file_stem().unwrap().to_string_lossy().to_string();
        let cfg_json = Json::parse(&std::fs::read_to_string(&p).unwrap()).unwrap();
        let cfg = TokenizerConfig::from_json(&cfg_json);
        let Some(src) = cfg.chat_template.clone() else { continue };
        let tpl = match ChatTemplate::new(&src) {
            Ok(t) => t,
            Err(e) => {
                println!("{name}.parse {{\"err\":{}}}", js(&format!("{e:?}")));
                continue;
            }
        };
        let tok = |o: &Option<String>| o.clone().map_or(Val::None, Val::Str);
        for (i, c) in cases.as_array().unwrap().iter().enumerate() {
            let mut vars: Vec<(&str, Val)> = vec![
                ("messages", Val::from_json(c.get("messages").unwrap())),
                ("add_generation_prompt", Val::Bool(c.get("gen").and_then(Json::as_bool).unwrap_or(false))),
                ("bos_token", tok(&cfg.bos_token)),
                ("eos_token", tok(&cfg.eos_token)),
                ("unk_token", tok(&cfg.unk_token)),
                ("pad_token", tok(&cfg.pad_token)),
            ];
            if let Some(t) = c.get("tools") {
                vars.push(("tools", Val::from_json(t)));
            }
            match tpl.render(&vars) {
                Ok(s) => println!("{name}.c{i} {{\"out\":{}}}", js(&s)),
                Err(e) => println!("{name}.c{i} {{\"err\":{}}}", js(&format!("{e:?}"))),
            }
        }
    }
}
