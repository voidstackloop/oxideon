// Writes a TorchScript archive for `tools/verify_torchscript_export.py`, which loads it with `torch.jit.load` and compares the
// outputs with eager PyTorch. Opt-in: `OXIDEON_TS_OUT=model.pt cargo test --test torchscript_export -- --ignored`
use oxideon_core::serialize::{ExportLayer, export_torchscript_sequential};
use oxideon_core::Tensor;

#[test]
#[ignore = "writes a file for the PyTorch check; set OXIDEON_TS_OUT"]
fn write_torchscript_for_pytorch() {
    let Ok(out) = std::env::var("OXIDEON_TS_OUT") else { return };
    let w1: Vec<f32> = (0..12).map(|i| ((i as f32) * 0.37).sin()).collect();
    let w2: Vec<f32> = (0..8).map(|i| ((i as f32) * 0.91).cos()).collect();
    let layers = vec![
        ExportLayer::Linear { name: "fc1".into(), weight: Tensor::from_slice(&w1, &[4, 3]).unwrap(), bias: Some(Tensor::from_slice(&[0.1f32, -0.2, 0.3, 0.0], &[4]).unwrap()) },
        ExportLayer::Relu { name: "r".into() },
        ExportLayer::Linear { name: "fc.2".into(), weight: Tensor::from_slice(&w2, &[2, 4]).unwrap(), bias: None },
        ExportLayer::Gelu { name: "g".into() },
        ExportLayer::Softmax { name: "s".into(), axis: -1 },
    ];
    export_torchscript_sequential(out, &layers).unwrap();
}
