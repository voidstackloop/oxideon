//! Cross-validation batch 49: ONNX models (built by tools/crossval/make_onnx_models.py into target/onnx_models) executed by the
//! pure-Rust importer vs onnxruntime (tools/crossval/check_batch49.py).
use oxideon_core::dtype::DType;
use oxideon_core::serialize::{OnnxModel, load_npy};

#[test]
#[ignore]
fn emit_onnx() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/onnx_models");
    for name in ["mlp", "cnn", "transformer", "embed", "misc"] {
        let model = match OnnxModel::from_file(format!("{dir}/{name}.onnx")) {
            Ok(m) => m,
            Err(e) => {
                println!("{name} {{\"err\":\"{e:?}\"}}");
                continue;
            }
        };
        let feeds: Vec<(String, _)> = model.input_names().iter().enumerate().map(|(i, n)| (n.clone(), load_npy(format!("{dir}/{name}_in{i}.npy")).unwrap())).collect();
        let refs: Vec<(&str, _)> = feeds.iter().map(|(n, t)| (n.as_str(), t.clone())).collect();
        match model.run(&refs) {
            Err(e) => println!("{name} {{\"err\":\"{e:?}\"}}"),
            Ok(outs) => {
                for (i, got) in outs.iter().enumerate() {
                    let want = load_npy(format!("{dir}/{name}_out{i}.npy")).unwrap();
                    let (g, w) = (got.to_dtype(DType::F64).unwrap(), want.to_dtype(DType::F64).unwrap());
                    let same_shape = got.shape() == want.shape();
                    let diff = if same_shape { g.sub(&w).unwrap().abs().unwrap().to_vec::<f64>().unwrap().into_iter().fold(0.0, f64::max) } else { -1.0 };
                    println!("{name}.out{i} {{\"same_shape\":{same_shape},\"maxdiff\":{diff:e},\"got\":{:?},\"want\":{:?}}}", got.shape(), want.shape());
                }
            }
        }
    }
}
