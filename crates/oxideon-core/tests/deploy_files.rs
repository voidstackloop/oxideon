// Writes deployment files for `tools/verify_deploy_exports.py`, which loads them with the real runtimes (LiteRT, coremltools,
// OpenVINO) and compares the results with NumPy.
//   OXIDEON_DEPLOY_DIR=out cargo test -p oxideon-core --test deploy_files -- --ignored
use oxideon_core::serialize::{ExportLayer, export_coreml_mlmodel, export_openvino_ir, export_tflite_flatbuffer};
use oxideon_core::Tensor;

fn layers(gelu: bool) -> Vec<ExportLayer> {
    let w1: Vec<f32> = (0..12).map(|i| ((i as f32) * 0.37).sin()).collect();
    let w2: Vec<f32> = (0..8).map(|i| ((i as f32) * 0.91).cos()).collect();
    vec![
        ExportLayer::Linear { name: "fc1".into(), weight: Tensor::from_slice(&w1, &[4, 3]).unwrap(), bias: Some(Tensor::from_slice(&[0.1f32, -0.2, 0.3, 0.0], &[4]).unwrap()) },
        if gelu { ExportLayer::Gelu { name: "act".into() } } else { ExportLayer::Relu { name: "act".into() } },
        ExportLayer::Linear { name: "fc2".into(), weight: Tensor::from_slice(&w2, &[2, 4]).unwrap(), bias: Some(Tensor::from_slice(&[0.5f32, -0.5], &[2]).unwrap()) },
        ExportLayer::Softmax { name: "sm".into(), axis: -1 },
    ]
}

#[test]
#[ignore = "writes files for the runtime check; set OXIDEON_DEPLOY_DIR"]
fn write_deploy_files() {
    let Ok(dir) = std::env::var("OXIDEON_DEPLOY_DIR") else { return };
    let d = std::path::Path::new(&dir);
    export_tflite_flatbuffer(d.join("relu.tflite"), &[1, 3], &layers(false)).unwrap();
    export_tflite_flatbuffer(d.join("gelu.tflite"), &[2, 3], &layers(true)).unwrap();
    export_coreml_mlmodel(d.join("model.mlmodel"), &[3], &layers(false)).unwrap();
    export_openvino_ir(d.join("model"), "mlp", &[1, 3], &layers(false)).unwrap();
    oxideon_core::serialize::export_tensorrt_bundle(d, "mlp_relu", &[1, 3], &layers(false), true, false).unwrap();
    oxideon_core::serialize::export_tensorrt_bundle(d, "mlp_gelu", &[2, 3], &layers(true), false, false).unwrap();
}
