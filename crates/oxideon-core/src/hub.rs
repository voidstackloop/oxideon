//! `from_pretrained`-style model loading: resolve a Hugging Face repo (or a local directory) to files on disk, and read
//! (sharded) safetensors weights from it.
//!
//! The crate has no HTTP dependency, so downloads shell out to the system `curl` (present on Windows 10+, macOS and
//! practically every Linux). Files are cached under `$OXIDEON_HOME/hub` (default `~/.cache/oxideon/hub`) and never
//! re-downloaded. `HF_TOKEN` is sent as a bearer token when set; `HF_ENDPOINT` overrides `https://huggingface.co`.
//!
//! ```no_run
//! # use oxideon_core::hub;
//! let dir = hub::snapshot("org/model", "main", &["config.json", "model.safetensors"]).unwrap();
//! let weights = hub::load_weights(&dir).unwrap();
//! ```

use crate::error::{Error, Result};
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn err(m: impl Into<String>) -> Error {
    Error::Msg(format!("hub: {}", m.into()))
}

/// Root of the download cache.
pub fn cache_dir() -> PathBuf {
    if let Some(p) = std::env::var_os("OXIDEON_HOME") {
        return PathBuf::from(p).join("hub");
    }
    let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")).map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    home.join(".cache").join("oxideon").join("hub")
}

fn safe_component(s: &str) -> Result<()> {
    if s.is_empty() || s.contains("..") || s.starts_with('/') || s.starts_with('\\') || s.contains(':') {
        return Err(err(format!("unsafe path component {s:?}")));
    }
    Ok(())
}

/// Downloads `url` to `dest` with `curl` (atomically: written to `<dest>.part`, then renamed).
pub fn download(url: &str, dest: &Path, token: Option<&str>) -> Result<()> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| err(format!("creating {}: {e}", parent.display())))?;
    }
    let part = PathBuf::from(format!("{}.part", dest.display()));
    let mut cmd = std::process::Command::new("curl");
    cmd.args(["--fail", "--location", "--silent", "--show-error", "--output"]).arg(&part);
    if let Some(t) = token {
        cmd.arg("--header").arg(format!("Authorization: Bearer {t}"));
    }
    cmd.arg(url);
    let out = cmd.output().map_err(|e| err(format!("cannot run curl: {e}")))?;
    if !out.status.success() {
        let _ = std::fs::remove_file(&part);
        return Err(err(format!("download of {url} failed: {}", String::from_utf8_lossy(&out.stderr).trim())));
    }
    std::fs::rename(&part, dest).map_err(|e| err(format!("saving {}: {e}", dest.display())))
}

/// A directory holding `files` of `repo` at `revision`: `repo` may also be an existing local directory, which is
/// returned as is. Missing files are downloaded into the cache; files already there are left alone.
pub fn snapshot(repo: &str, revision: &str, files: &[&str]) -> Result<PathBuf> {
    let local = Path::new(repo);
    if local.is_dir() {
        return Ok(local.to_path_buf());
    }
    for part in repo.split('/') {
        safe_component(part)?;
    }
    safe_component(revision)?;
    let dir = cache_dir().join(repo.replace('/', "--")).join(revision);
    let endpoint = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".into());
    let token = std::env::var("HF_TOKEN").ok();
    for f in files {
        safe_component(f)?;
        let dest = dir.join(f);
        if !dest.exists() {
            download(&format!("{}/{repo}/resolve/{revision}/{f}", endpoint.trim_end_matches('/')), &dest, token.as_deref())?;
        }
    }
    Ok(dir)
}

/// The weight files named by `model.safetensors.index.json` (sharded checkpoints), or the single `model.safetensors`.
pub fn weight_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        let text = std::fs::read_to_string(&index).map_err(|e| err(format!("{}: {e}", index.display())))?;
        let v = crate::tokenizer::json::Json::parse(&text).map_err(|e| err(format!("bad index json: {e}")))?;
        let map = v.get("weight_map").and_then(|m| m.as_object()).ok_or_else(|| err("index has no weight_map"))?;
        let mut names: Vec<String> = map.iter().filter_map(|(_, f)| f.as_str().map(str::to_string)).collect();
        names.sort();
        names.dedup();
        for n in &names {
            safe_component(n)?;
        }
        return Ok(names.into_iter().map(|n| dir.join(n)).collect());
    }
    let single = dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }
    Err(err(format!("no model.safetensors or model.safetensors.index.json in {}", dir.display())))
}

/// Reads every weight in `dir` (single file or shards) into one name → tensor map.
pub fn load_weights(dir: impl AsRef<Path>) -> Result<HashMap<String, Tensor>> {
    let mut all = HashMap::new();
    for f in weight_files(dir.as_ref())? {
        all.extend(crate::serialize::load_safetensors(&f)?);
    }
    Ok(all)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sharded_and_single_checkpoints_load_and_unsafe_paths_are_refused() {
        let dir = std::env::temp_dir().join("oxideon_hub_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let a: HashMap<String, Tensor> = [("w1".to_string(), Tensor::from_slice(&[1.0f32, 2.0], &[2]).unwrap())].into();
        let b: HashMap<String, Tensor> = [("w2".to_string(), Tensor::from_slice(&[3.0f32], &[1]).unwrap())].into();
        crate::serialize::save_safetensors(&a, dir.join("model-00001-of-00002.safetensors")).unwrap();
        crate::serialize::save_safetensors(&b, dir.join("model-00002-of-00002.safetensors")).unwrap();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"metadata":{},"weight_map":{"w1":"model-00001-of-00002.safetensors","w2":"model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();
        let w = load_weights(&dir).unwrap();
        assert_eq!(w["w1"].to_vec::<f32>().unwrap(), vec![1.0, 2.0]);
        assert_eq!(w["w2"].to_vec::<f32>().unwrap(), vec![3.0]);
        // a local directory is its own snapshot
        assert_eq!(snapshot(dir.to_str().unwrap(), "main", &[]).unwrap(), dir);
        // an index must not point outside the directory
        std::fs::write(dir.join("model.safetensors.index.json"), r#"{"weight_map":{"w":"../evil.safetensors"}}"#).unwrap();
        assert!(weight_files(&dir).is_err());
        assert!(snapshot("../x", "main", &["a"]).is_err());
        assert!(snapshot("org/model", "main", &["../../etc/passwd"]).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn download_works_through_curl_and_is_atomic() {
        let dir = std::env::temp_dir().join("oxideon_hub_dl");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("src.txt");
        std::fs::write(&src, b"hello").unwrap();
        let dest = dir.join("out").join("copy.txt");
        let url = format!("file:///{}", src.display().to_string().replace('\\', "/"));
        match download(&url, &dest, None) {
            Ok(()) => {
                assert_eq!(std::fs::read(&dest).unwrap(), b"hello");
                assert!(!dir.join("out").join("copy.txt.part").exists());
            }
            Err(e) if e.to_string().contains("cannot run curl") => {} // curl absent: nothing to test
            Err(e) => panic!("{e}"),
        }
        // a failing download leaves no partial file behind
        let bad = dir.join("out").join("missing.txt");
        if download("file:///definitely/not/here", &bad, None).is_err() {
            assert!(!bad.exists() && !dir.join("out").join("missing.txt.part").exists());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
