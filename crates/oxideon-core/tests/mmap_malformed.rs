// Memory-mapped safetensors loading must reject malformed files with errors, never panic or read out of bounds.
use oxideon_core::serialize::SafetensorsMmap;
use std::path::PathBuf;

fn write(name: &str, bytes: &[u8]) -> PathBuf {
    let p = std::env::temp_dir().join(format!("oxideon_mmap_malformed_{}_{name}.safetensors", std::process::id()));
    std::fs::write(&p, bytes).unwrap();
    p
}

fn file(header: &str, body: &[u8]) -> Vec<u8> {
    let mut v = (header.len() as u64).to_le_bytes().to_vec();
    v.extend_from_slice(header.as_bytes());
    v.extend_from_slice(body);
    v
}

/// opens the file and tries every tensor; true when nothing panicked and everything was rejected or valid
fn open_and_read(name: &str, bytes: &[u8]) -> Result<Vec<Vec<f32>>, String> {
    let p = write(name, bytes);
    let r = std::panic::catch_unwind(|| {
        let m = SafetensorsMmap::open(&p).map_err(|e| e.to_string())?;
        let mut out = vec![];
        for n in m.tensor_names() {
            out.push(m.get(n).map_err(|e| e.to_string())?.to_vec::<f32>().map_err(|e| e.to_string())?);
            m.raw_bytes(n).map_err(|e| e.to_string())?;
        }
        Ok(out)
    });
    let _ = std::fs::remove_file(&p);
    r.unwrap_or_else(|_| panic!("{name}: panicked instead of returning an error"))
}

#[test]
fn well_formed_file_reads_back() {
    let body: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    let h = r#"{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
    assert_eq!(open_and_read("ok", &file(h, &body)).unwrap(), vec![vec![1.0, 2.0, 3.0, 4.0]]);
    // a __metadata__ entry is skipped
    let h = r#"{"__metadata__":{"format":"pt"},"w":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#;
    assert_eq!(open_and_read("meta", &file(h, &body)).unwrap().len(), 1);
}

#[test]
fn malformed_files_are_errors_not_panics() {
    let body = vec![0u8; 16];
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("empty", vec![]),
        ("short", vec![1, 2, 3]),
        ("huge_header_len", { let mut v = u64::MAX.to_le_bytes().to_vec(); v.extend_from_slice(b"{}"); v }),
        ("header_past_eof", { let mut v = 1000u64.to_le_bytes().to_vec(); v.extend_from_slice(b"{}"); v }),
        ("not_json", file("hello", &body)),
        ("offsets_past_body", file(r#"{"w":{"dtype":"F32","shape":[4],"data_offsets":[0,64]}}"#, &body)),
        ("offsets_reversed", file(r#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[8,4]}}"#, &body)),
        ("shape_bigger_than_data", file(r#"{"w":{"dtype":"F32","shape":[100],"data_offsets":[0,16]}}"#, &body)),
        ("shape_smaller_than_data", file(r#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,16]}}"#, &body)),
        ("shape_overflow", file(r#"{"w":{"dtype":"F32","shape":[4294967296,4294967296,4294967296],"data_offsets":[0,16]}}"#, &body)),
        ("unknown_dtype", file(r#"{"w":{"dtype":"Q7","shape":[4],"data_offsets":[0,16]}}"#, &body)),
        ("missing_fields", file(r#"{"w":{"dtype":"F32"}}"#, &body)),
        ("unaligned_f32_len", file(r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,3]}}"#, &body)),
        ("utf8_garbage_header", { let mut v = 4u64.to_le_bytes().to_vec(); v.extend_from_slice(&[0xff, 0xfe, 0xfd, 0xfc]); v }),
    ];
    for (name, bytes) in cases {
        assert!(open_and_read(name, &bytes).is_err(), "{name}: a malformed file was accepted");
    }
}

#[test]
fn missing_tensor_and_missing_file_are_errors() {
    let body: Vec<u8> = [1.0f32].iter().flat_map(|v| v.to_le_bytes()).collect();
    let p = write("lookup", &file(r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#, &body));
    let m = SafetensorsMmap::open(&p).unwrap();
    assert!(m.get("nope").is_err());
    assert!(m.raw_bytes("nope").is_err());
    drop(m);
    let _ = std::fs::remove_file(&p);
    assert!(SafetensorsMmap::open(std::env::temp_dir().join("oxideon_definitely_not_here.safetensors")).is_err());
}
