//! Cross-validation batch 48: PyTorch checkpoints (target/pt_files, written by tools/crossval/make_pt_files.py) loaded by the
//! pure-Rust reader vs torch.load (tools/crossval/check_batch48.py).
use oxideon_core::dtype::DType;
use oxideon_core::serialize::load_torch_tensors;

#[test]
#[ignore]
fn emit_checkpoints() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/pt_files");
    let mut files: Vec<_> = std::fs::read_dir(dir).unwrap().filter_map(|e| e.ok()).map(|e| e.path()).collect();
    files.sort();
    for p in files {
        let name = p.file_stem().unwrap().to_string_lossy().to_string();
        match load_torch_tensors(&p) {
            Err(e) => println!("{name} {{\"err\":\"{e:?}\"}}"),
            Ok(ts) => {
                for (k, t) in ts {
                    let v = t.to_dtype(DType::F64).unwrap().to_vec::<f64>().unwrap();
                    let mut h = 0xcbf29ce484222325u64;
                    for x in &v {
                        for b in x.to_bits().to_le_bytes() {
                            h = (h ^ b as u64).wrapping_mul(0x100000001b3);
                        }
                    }
                    println!("{name}::{k} {{\"shape\":{:?},\"dtype\":\"{:?}\",\"hash\":{h}}}", t.shape(), t.dtype());
                }
            }
        }
    }
}
