//! Deployment exporters, with their verification status (`tools/verify_torchscript_export.py` and
//! `tools/verify_deploy_exports.py` re-run these checks against the real runtimes):
//! - **TorchScript** (`.pt`): loads with `torch.jit.load` and reproduces eager PyTorch.
//! - **TensorFlow Lite** (`.tflite`): a FlatBuffer writer for `Linear` / `ReLU` / `GELU` / `Softmax`; runs in the LiteRT
//!   interpreter and matches NumPy.
//! - **OpenVINO IR** (`.xml` + `.bin`): reads and runs in the OpenVINO runtime and matches NumPy.
//! - **Core ML** (`.mlmodel`): parses with Core ML's own protobuf schema (layers, weights, shapes verified); predictions
//!   need macOS, so numerical output was not executed. `Gelu` is rejected (no such layer in this specification).
//! - **ONNX / TensorRT**: [`serialize_onnx_sequential`] writes a real ONNX graph (the interchange format TensorRT builds
//!   engines from); [`export_tensorrt_bundle`] adds the `trtexec` manifest and [`build_tensorrt_engine`] runs `trtexec` when
//!   TensorRT is installed.

use std::collections::HashMap;
use std::path::Path;

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Layer description for deployment graph export (`Linear` / `ReLU` / `Gelu` / `Softmax`).
#[derive(Debug, Clone)]
pub enum ExportLayer {
    Linear {
        name: String,
        weight: Tensor,
        bias: Option<Tensor>,
    },
    Relu {
        name: String,
    },
    Gelu {
        name: String,
    },
    Softmax {
        name: String,
        axis: isize,
    },
}

// ---------------------------------------------------------------------------
// 1. OpenVINO IR v11 (.xml + .bin)
// ---------------------------------------------------------------------------

/// Exports a sequential model to OpenVINO IR v11 (`(xml_string, bin_bytes)`).
pub fn serialize_openvino_ir(
    model_name: &str,
    input_shape: &[usize],
    layers: &[ExportLayer],
) -> Result<(String, Vec<u8>)> {
    let mut bin = Vec::new();
    let mut xml = String::new();
    xml.push_str("<?xml version=\"1.0\"?>\n");
    xml.push_str(&format!("<net name=\"{model_name}\" version=\"11\">\n\t<layers>\n"));

    let shape_tags = |shp: &[usize]| -> String {
        shp.iter().map(|d| format!("<dim>{d}</dim>")).collect::<Vec<_>>().join("")
    };

    let mut cur_layer_id = 0usize;
    let mut cur_port = 0usize;
    let mut cur_shape = input_shape.to_vec();
    let mut edges = Vec::<(usize, usize, usize, usize)>::new();

    // Input Parameter layer
    xml.push_str(&format!(
        "\t\t<layer id=\"0\" name=\"input\" type=\"Parameter\" version=\"opset1\">\n\t\t\t<data shape=\"{}\" element_type=\"f32\"/>\n\t\t\t<output><port id=\"0\" precision=\"FP32\">{}</port></output>\n\t\t</layer>\n",
        input_shape.iter().map( usize::to_string).collect::<Vec<_>>().join(","),
        shape_tags(&cur_shape)
    ));
    let mut next_id = 1usize;

    for layer in layers {
        match layer {
            ExportLayer::Linear { name, weight, bias } => {
                let w_cpu = weight.to_device(&crate::backend::Device::Cpu)?.contiguous()?;
                let w_f32 = w_cpu.to_vec::<f32>()?;
                let w_off = bin.len();
                for &v in &w_f32 {
                    bin.extend_from_slice(&v.to_le_bytes());
                }
                let w_sz = bin.len() - w_off;
                let w_id = next_id;
                next_id += 1;
                let out_features = weight.shape()[0];
                let in_features = weight.shape()[1];
                xml.push_str(&format!(
                    "\t\t<layer id=\"{w_id}\" name=\"{name}_weight\" type=\"Const\" version=\"opset1\">\n\t\t\t<data offset=\"{w_off}\" size=\"{w_sz}\" shape=\"{out_features},{in_features}\" element_type=\"f32\"/>\n\t\t\t<output><port id=\"0\" precision=\"FP32\">{}</port></output>\n\t\t</layer>\n",
                    shape_tags(weight.shape())
                ));

                let mm_id = next_id;
                next_id += 1;
                let mut next_shape = cur_shape.clone();
                if let Some(last) = next_shape.last_mut() {
                    *last = out_features;
                }
                xml.push_str(&format!(
                    "\t\t<layer id=\"{mm_id}\" name=\"{name}_matmul\" type=\"MatMul\" version=\"opset1\">\n\t\t\t<data transpose_a=\"false\" transpose_b=\"true\"/>\n\t\t\t<input><port id=\"0\">{}</port><port id=\"1\">{}</port></input>\n\t\t\t<output><port id=\"2\" precision=\"FP32\">{}</port></output>\n\t\t</layer>\n",
                    shape_tags(&cur_shape),
                    shape_tags(weight.shape()),
                    shape_tags(&next_shape)
                ));
                edges.push((cur_layer_id, cur_port, mm_id, 0));
                edges.push((w_id, 0, mm_id, 1));
                cur_layer_id = mm_id;
                cur_port = 2;
                cur_shape = next_shape;

                if let Some(b) = bias {
                    let b_f32 = b.to_device(&crate::backend::Device::Cpu)?.to_vec::<f32>()?;
                    let b_off = bin.len();
                    for &v in &b_f32 {
                        bin.extend_from_slice(&v.to_le_bytes());
                    }
                    let b_sz = bin.len() - b_off;
                    let b_id = next_id;
                    next_id += 1;
                    xml.push_str(&format!(
                        "\t\t<layer id=\"{b_id}\" name=\"{name}_bias\" type=\"Const\" version=\"opset1\">\n\t\t\t<data offset=\"{b_off}\" size=\"{b_sz}\" shape=\"1,{out_features}\" element_type=\"f32\"/>\n\t\t\t<output><port id=\"0\" precision=\"FP32\"><dim>1</dim><dim>{out_features}</dim></port></output>\n\t\t</layer>\n"
                    ));
                    let add_id = next_id;
                    next_id += 1;
                    xml.push_str(&format!(
                        "\t\t<layer id=\"{add_id}\" name=\"{name}_add\" type=\"Add\" version=\"opset1\">\n\t\t\t<input><port id=\"0\">{}</port><port id=\"1\"><dim>1</dim><dim>{out_features}</dim></port></input>\n\t\t\t<output><port id=\"2\" precision=\"FP32\">{}</port></output>\n\t\t</layer>\n",
                        shape_tags(&cur_shape),
                        shape_tags(&cur_shape)
                    ));
                    edges.push((cur_layer_id, cur_port, add_id, 0));
                    edges.push((b_id, 0, add_id, 1));
                    cur_layer_id = add_id;
                    cur_port = 2;
                }
            }
            ExportLayer::Relu { name } | ExportLayer::Gelu { name } | ExportLayer::Softmax { name, .. } => {
                let (op_type, version, data) = match layer {
                    ExportLayer::Relu { .. } => ("ReLU", "opset1", String::new()),
                    ExportLayer::Gelu { .. } => ("Gelu", "opset7", "
			<data approximation_mode=\"ERF\"/>".to_string()),
                    ExportLayer::Softmax { axis, .. } => {
                        let rank = cur_shape.len() as isize;
                        let ax = if *axis < 0 { rank + *axis } else { *axis };
                        if ax < 0 || ax >= rank {
                            return Err(Error::Serialization(format!("openvino: softmax axis {axis} is out of range for rank {rank}")));
                        }
                        ("SoftMax", "opset1", format!("
			<data axis=\"{ax}\"/>"))
                    }
                    _ => unreachable!(),
                };
                let act_id = next_id;
                next_id += 1;
                xml.push_str(&format!(
                    "\t\t<layer id=\"{act_id}\" name=\"{name}\" type=\"{op_type}\" version=\"{version}\">{data}\n\t\t\t<input><port id=\"0\">{}</port></input>\n\t\t\t<output><port id=\"1\" precision=\"FP32\">{}</port></output>\n\t\t</layer>\n",
                    shape_tags(&cur_shape),
                    shape_tags(&cur_shape)
                ));
                edges.push((cur_layer_id, cur_port, act_id, 0));
                cur_layer_id = act_id;
                cur_port = 1;
            }
        }
    }

    let res_id = next_id;
    xml.push_str(&format!(
        "\t\t<layer id=\"{res_id}\" name=\"output\" type=\"Result\" version=\"opset1\">\n\t\t\t<input><port id=\"0\">{}</port></input>\n\t\t</layer>\n\t</layers>\n\t<edges>\n",
        shape_tags(&cur_shape)
    ));
    edges.push((cur_layer_id, cur_port, res_id, 0));
    for (fl, fp, tl, tp) in edges {
        xml.push_str(&format!(
            "\t\t<edge from-layer=\"{fl}\" from-port=\"{fp}\" to-layer=\"{tl}\" to-port=\"{tp}\"/>\n"
        ));
    }
    xml.push_str("\t</edges>\n</net>\n");
    Ok((xml, bin))
}

/// Writes `<base_path>.xml` and `<base_path>.bin` for OpenVINO runtime execution.
pub fn export_openvino_ir(
    base_path: impl AsRef<Path>,
    model_name: &str,
    input_shape: &[usize],
    layers: &[ExportLayer],
) -> Result<()> {
    let (xml, bin) = serialize_openvino_ir(model_name, input_shape, layers)?;
    let xml_path = base_path.as_ref().with_extension("xml");
    let bin_path = base_path.as_ref().with_extension("bin");
    std::fs::write(&xml_path, xml).map_err(|e| Error::Serialization(e.to_string()))?;
    std::fs::write(&bin_path, bin).map_err(|e| Error::Serialization(e.to_string()))
}

// ---------------------------------------------------------------------------
// 2. Core ML (.mlmodel protobuf v4)
// ---------------------------------------------------------------------------

fn pb_varint(mut v: u64, out: &mut Vec<u8>) {
    while v >= 0x80 {
        out.push(((v as u8) & 0x7F) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn pb_tag(field: u32, wire: u8, out: &mut Vec<u8>) {
    pb_varint(((field as u64) << 3) | (wire as u64), out);
}

fn pb_bytes(field: u32, data: &[u8], out: &mut Vec<u8>) {
    pb_tag(field, 2, out);
    pb_varint(data.len() as u64, out);
    out.extend_from_slice(data);
}

fn pb_str(field: u32, s: &str, out: &mut Vec<u8>) {
    pb_bytes(field, s.as_bytes(), out);
}

/// `WeightParams { floatValue = 1 (packed repeated float) }`
fn coreml_weights(t: &Tensor) -> Result<Vec<u8>> {
    let v = t.to_device(&crate::backend::Device::Cpu)?.contiguous()?.to_vec::<f32>()?;
    let mut raw = Vec::with_capacity(v.len() * 4);
    for x in &v {
        raw.extend_from_slice(&x.to_le_bytes());
    }
    let mut w = Vec::new();
    pb_bytes(1, &raw, &mut w);
    Ok(w)
}

/// `FeatureDescription { name = 1, type = 3 { multiArrayType = 5 { shape = 1 (packed), dataType = 2 (FLOAT32) } } }`
fn coreml_feature(name: &str, shape: &[usize]) -> Vec<u8> {
    let mut dims = Vec::new();
    for &d in shape {
        pb_varint(d as u64, &mut dims);
    }
    let mut arr = Vec::new();
    pb_bytes(1, &dims, &mut arr);
    pb_tag(2, 0, &mut arr);
    pb_varint(65568, &mut arr); // ArrayDataType.FLOAT32
    let mut ty = Vec::new();
    pb_bytes(5, &arr, &mut ty);
    let mut f = Vec::new();
    pb_str(1, name, &mut f);
    pb_bytes(3, &ty, &mut f);
    f
}

/// Serializes a sequential model (`Linear`, `Relu`, `Softmax` over the last axis) into a Core ML `.mlmodel` protobuf
/// (specification version 4, `neuralNetwork` = field 500). `Gelu` has no layer in this specification and is rejected.
///
/// The field numbers follow Core ML's `Model.proto` / `NeuralNetwork.proto` (innerProduct = 140, activation = 130 with
/// `ReLU` = 10, softmax = 175, `InnerProductLayerParams.weights` = 20). The result has not been validated against Apple's
/// runtime.
pub fn serialize_coreml_mlmodel(
    input_shape: &[usize],
    layers: &[ExportLayer],
) -> Result<Vec<u8>> {
    if layers.is_empty() {
        return Err(Error::Serialization("coreml: no layers to export".into()));
    }
    // output shape: the last Linear's width replaces the last input dimension
    let mut out_shape = input_shape.to_vec();
    for l in layers {
        if let ExportLayer::Linear { weight, .. } = l
            && let Some(last) = out_shape.last_mut()
        {
            *last = weight.shape()[0];
        }
    }
    let mut model = Vec::new();
    pb_tag(1, 0, &mut model); // specificationVersion
    pb_varint(4, &mut model);

    let mut desc = Vec::new();
    pb_bytes(1, &coreml_feature("input", input_shape), &mut desc); // ModelDescription.input
    pb_bytes(10, &coreml_feature("output", &out_shape), &mut desc); // ModelDescription.output
    pb_bytes(2, &desc, &mut model);

    let mut nn = Vec::new();
    let mut prev_blob = "input".to_string();
    for (idx, layer) in layers.iter().enumerate() {
        let next_blob = if idx + 1 == layers.len() { "output".to_string() } else { format!("blob_{idx}") };
        let mut lbuf = Vec::new();
        let name = match layer {
            ExportLayer::Linear { name, .. } | ExportLayer::Relu { name } | ExportLayer::Gelu { name } | ExportLayer::Softmax { name, .. } => name,
        };
        pb_str(1, name, &mut lbuf); // NeuralNetworkLayer.name
        pb_str(2, &prev_blob, &mut lbuf); // input
        pb_str(3, &next_blob, &mut lbuf); // output
        match layer {
            ExportLayer::Linear { weight, bias, .. } => {
                if weight.ndim() != 2 {
                    return Err(Error::Serialization("coreml: Linear weights must be [out, in]".into()));
                }
                let mut ip = Vec::new();
                pb_tag(1, 0, &mut ip); // inputChannels
                pb_varint(weight.shape()[1] as u64, &mut ip);
                pb_tag(2, 0, &mut ip); // outputChannels
                pb_varint(weight.shape()[0] as u64, &mut ip);
                pb_tag(10, 0, &mut ip); // hasBias
                pb_varint(u64::from(bias.is_some()), &mut ip);
                pb_bytes(20, &coreml_weights(weight)?, &mut ip); // weights
                if let Some(b) = bias {
                    pb_bytes(21, &coreml_weights(b)?, &mut ip); // bias
                }
                pb_bytes(140, &ip, &mut lbuf); // innerProduct
            }
            ExportLayer::Relu { .. } => {
                let mut act = Vec::new();
                pb_bytes(10, &[], &mut act); // ActivationParams.ReLU (empty message)
                pb_bytes(130, &act, &mut lbuf); // activation
            }
            ExportLayer::Softmax { axis, .. } => {
                if *axis != -1 && *axis as usize != input_shape.len().saturating_sub(1) {
                    return Err(Error::Serialization("coreml: Softmax is only supported over the last axis".into()));
                }
                pb_bytes(175, &[], &mut lbuf); // softmax (empty SoftmaxLayerParams)
            }
            ExportLayer::Gelu { .. } => return Err(Error::Serialization("coreml: Gelu is not supported by this exporter".into())),
        }
        pb_bytes(1, &lbuf, &mut nn); // NeuralNetwork.layers
        prev_blob = next_blob;
    }
    pb_bytes(500, &nn, &mut model); // neuralNetwork
    Ok(model)
}

pub fn export_coreml_mlmodel(
    path: impl AsRef<Path>,
    input_shape: &[usize],
    layers: &[ExportLayer],
) -> Result<()> {
    let bytes = serialize_coreml_mlmodel(input_shape, layers)?;
    std::fs::write(path, bytes).map_err(|e| Error::Serialization(e.to_string()))
}

// ---------------------------------------------------------------------------
// 3. TFLite (.tflite "TFL3" FlatBuffer container)
// ---------------------------------------------------------------------------

/// Serializes a sequential model (`Linear`, `Relu`, `Gelu`, `Softmax`) into a TensorFlow Lite flatbuffer (`.tflite`,
/// schema version 3) using the `FULLY_CONNECTED`, `RELU`, `GELU` and `SOFTMAX` builtin operators, float32 throughout.
/// Softmax runs over the last axis (the only axis TFLite supports).
pub fn serialize_tflite_flatbuffer(input_shape: &[usize], layers: &[ExportLayer]) -> Result<Vec<u8>> {
    use super::flatbuffer::Fb;
    const FULLY_CONNECTED: i32 = 9;
    const RELU: i32 = 19;
    const SOFTMAX: i32 = 25;
    const GELU: i32 = 150;
    if layers.is_empty() {
        return Err(Error::Serialization("tflite: no layers to export".into()));
    }
    let ser = |m: &str| Error::Serialization(format!("tflite: {m}"));
    if let Some(ExportLayer::Softmax { axis, .. }) = layers.iter().find(|l| matches!(l, ExportLayer::Softmax { .. })) {
        let _ = axis;
    }
    let mut fb = Fb::new();

    // buffers: 0 is the empty sentinel, then one per constant tensor
    struct Tn {
        shape: Vec<i32>,
        name: String,
        buffer: u32,
    }
    let mut buffer_data: Vec<Vec<u8>> = vec![vec![]];
    let mut tensors: Vec<Tn> = vec![Tn { shape: input_shape.iter().map(|&d| d as i32).collect(), name: "input".into(), buffer: 0 }];
    // (op code, inputs, output, options kind)
    struct Op {
        code: i32,
        inputs: Vec<i32>,
        output: i32,
    }
    let mut ops: Vec<Op> = Vec::new();
    let mut cur = 0i32;
    let mut cur_shape = input_shape.to_vec();
    for layer in layers {
        match layer {
            ExportLayer::Linear { name, weight, bias } => {
                if weight.ndim() != 2 || cur_shape.last() != Some(&weight.shape()[1]) {
                    return Err(ser(&format!("Linear '{name}' weight {:?} does not match input width of {cur_shape:?}", weight.shape())));
                }
                let mut add_const = |t: &Tensor, tname: String, tensors: &mut Vec<Tn>| -> Result<i32> {
                    let v = t.to_device(&crate::backend::Device::Cpu)?.contiguous()?.to_vec::<f32>()?;
                    buffer_data.push(v.iter().flat_map(|x| x.to_le_bytes()).collect());
                    tensors.push(Tn { shape: t.shape().iter().map(|&d| d as i32).collect(), name: tname, buffer: (buffer_data.len() - 1) as u32 });
                    Ok((tensors.len() - 1) as i32)
                };
                let w = add_const(weight, format!("{name}.weight"), &mut tensors)?;
                let b = match bias {
                    Some(b) => add_const(b, format!("{name}.bias"), &mut tensors)?,
                    None => -1,
                };
                *cur_shape.last_mut().unwrap() = weight.shape()[0];
                tensors.push(Tn { shape: cur_shape.iter().map(|&d| d as i32).collect(), name: format!("{name}.out"), buffer: 0 });
                let out = (tensors.len() - 1) as i32;
                ops.push(Op { code: FULLY_CONNECTED, inputs: vec![cur, w, b], output: out });
                cur = out;
            }
            ExportLayer::Relu { name } | ExportLayer::Gelu { name } | ExportLayer::Softmax { name, .. } => {
                let code = match layer {
                    ExportLayer::Relu { .. } => RELU,
                    ExportLayer::Gelu { .. } => GELU,
                    _ => SOFTMAX,
                };
                if let ExportLayer::Softmax { axis, .. } = layer
                    && *axis != -1
                    && *axis as usize != cur_shape.len().saturating_sub(1)
                {
                    return Err(ser("Softmax is only supported over the last axis"));
                }
                tensors.push(Tn { shape: cur_shape.iter().map(|&d| d as i32).collect(), name: format!("{name}.out"), buffer: 0 });
                let out = (tensors.len() - 1) as i32;
                ops.push(Op { code, inputs: vec![cur], output: out });
                cur = out;
            }
        }
    }

    // operator codes (deduplicated, in order of first use)
    let mut codes: Vec<i32> = Vec::new();
    for op in &ops {
        if !codes.contains(&op.code) {
            codes.push(op.code);
        }
    }
    let code_offs: Vec<usize> = codes
        .iter()
        .map(|&c| {
            fb.start_table();
            fb.add_u8(0, if c < 127 { c as u8 } else { 127 }); // deprecated_builtin_code
            fb.add_i32(2, 1); // version
            fb.add_i32(3, c); // builtin_code
            fb.end_table()
        })
        .collect();

    // operators (options tables first: children are written before their parents)
    let op_offs: Vec<usize> = ops
        .iter()
        .map(|op| {
            let opts = match op.code {
                FULLY_CONNECTED => {
                    fb.start_table();
                    Some((8u8, fb.end_table())) // FullyConnectedOptions: all defaults (no fused activation)
                }
                SOFTMAX => {
                    fb.start_table();
                    fb.add_f32(0, 1.0);
                    Some((9u8, fb.end_table())) // SoftmaxOptions { beta = 1 }
                }
                _ => None,
            };
            let ins = fb.create_i32_vec(&op.inputs);
            let outs = fb.create_i32_vec(&[op.output]);
            fb.start_table();
            fb.add_u32(0, codes.iter().position(|&c| c == op.code).unwrap() as u32);
            fb.add_offset(1, ins);
            fb.add_offset(2, outs);
            if let Some((tag, off)) = opts {
                fb.add_u8(3, tag);
                fb.add_offset(4, off);
            }
            fb.end_table()
        })
        .collect();

    // tensors
    let tensor_offs: Vec<usize> = tensors
        .iter()
        .map(|t| {
            let shape = fb.create_i32_vec(&t.shape);
            let name = fb.create_string(&t.name);
            fb.start_table();
            fb.add_offset(0, shape);
            fb.add_u8(1, 0); // FLOAT32
            fb.add_u32(2, t.buffer);
            fb.add_offset(3, name);
            fb.end_table()
        })
        .collect();

    let tensors_vec = fb.create_offset_vec(&tensor_offs);
    let inputs_vec = fb.create_i32_vec(&[0]);
    let outputs_vec = fb.create_i32_vec(&[cur]);
    let ops_vec = fb.create_offset_vec(&op_offs);
    let sg_name = fb.create_string("main");
    fb.start_table();
    fb.add_offset(0, tensors_vec);
    fb.add_offset(1, inputs_vec);
    fb.add_offset(2, outputs_vec);
    fb.add_offset(3, ops_vec);
    fb.add_offset(4, sg_name);
    let subgraph = fb.end_table();
    let subgraphs_vec = fb.create_offset_vec(&[subgraph]);

    let buffer_offs: Vec<usize> = buffer_data
        .iter()
        .map(|d| {
            if d.is_empty() {
                fb.start_table();
                fb.end_table()
            } else {
                let data = fb.create_bytes(d);
                fb.start_table();
                fb.add_offset(0, data);
                fb.end_table()
            }
        })
        .collect();
    let buffers_vec = fb.create_offset_vec(&buffer_offs);
    let codes_vec = fb.create_offset_vec(&code_offs);
    let desc = fb.create_string("oxideon");
    fb.start_table();
    fb.add_u32(0, 3); // schema version
    fb.add_offset(1, codes_vec);
    fb.add_offset(2, subgraphs_vec);
    fb.add_offset(3, desc);
    fb.add_offset(4, buffers_vec);
    let model = fb.end_table();
    Ok(fb.finish(model, b"TFL3"))
}

pub fn export_tflite_flatbuffer(
    path: impl AsRef<Path>,
    input_shape: &[usize],
    layers: &[ExportLayer],
) -> Result<()> {
    let bytes = serialize_tflite_flatbuffer(input_shape, layers)?;
    std::fs::write(path, bytes).map_err(|e| Error::Serialization(e.to_string()))
}

// ---------------------------------------------------------------------------
// 4. ONNX and TensorRT
// ---------------------------------------------------------------------------

/// Serializes a sequential model (`Linear`, `Relu`, `Gelu`, `Softmax`) as an ONNX model (IR 8, opset 17) with one float
/// input named `input` and one output named `output`.
pub fn serialize_onnx_sequential(input_shape: &[usize], layers: &[ExportLayer]) -> Result<Vec<u8>> {
    use super::onnx::{Dim, OnnxBuilder};
    if layers.is_empty() {
        return Err(Error::Serialization("onnx: no layers to export".into()));
    }
    let mut b = OnnxBuilder::new("input", input_shape.iter().map(|&d| Dim::Fixed(d)).collect());
    let mut out_shape = input_shape.to_vec();
    for layer in layers {
        match layer {
            ExportLayer::Linear { weight, bias, .. } => {
                if weight.ndim() != 2 || out_shape.last() != Some(&weight.shape()[1]) {
                    return Err(Error::Serialization(format!("onnx: Linear weight {:?} does not match input width of {out_shape:?}", weight.shape())));
                }
                let (out_f, in_f) = (weight.shape()[0], weight.shape()[1]);
                let mut l = crate::nn::Linear::with_bias(in_f, out_f, bias.is_some())?;
                l.weight = weight.to_device(&crate::backend::Device::Cpu)?.to_dtype(crate::dtype::DType::F32)?;
                l.bias = match bias {
                    Some(t) => Some(t.to_device(&crate::backend::Device::Cpu)?.to_dtype(crate::dtype::DType::F32)?),
                    None => None,
                };
                b.linear(&l)?;
                *out_shape.last_mut().unwrap() = out_f;
            }
            ExportLayer::Relu { .. } => {
                b.relu();
            }
            ExportLayer::Gelu { .. } => {
                b.gelu();
            }
            ExportLayer::Softmax { axis, .. } => {
                b.softmax(*axis as i64);
            }
        }
    }
    b.finish("output", false, out_shape.iter().map(|&d| Dim::Fixed(d)).collect())
}

/// Generates the `trtexec` command manifest (JSON) for a model file `<model_name>.onnx`.
pub fn serialize_tensorrt_manifest(
    model_name: &str,
    input_shape: &[usize],
    fp16: bool,
    int8: bool,
) -> String {
    let dims = input_shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join("x");
    let mut flags = vec![format!("--onnx={model_name}.onnx"), format!("--saveEngine={model_name}.plan"), format!("--shapes=input:{dims}")];
    if fp16 {
        flags.push("--fp16".into());
    }
    if int8 {
        flags.push("--int8".into());
    }
    format!(
        "{{\n  \"format\": \"tensorrt-manifest-v1\",\n  \"model\": \"{model_name}\",\n  \"input_shape\": {:?},\n  \"fp16\": {fp16},\n  \"int8\": {int8},\n  \"trtexec_cmd\": \"trtexec {}\"\n}}\n",
        input_shape,
        flags.join(" ")
    )
}

/// Writes `<dir>/<model_name>.onnx` (a real ONNX model) and `<dir>/<model_name>.trt.json` (the `trtexec` command that
/// builds the engine from it). Returns the ONNX path.
pub fn export_tensorrt_bundle(
    dir: impl AsRef<Path>,
    model_name: &str,
    input_shape: &[usize],
    layers: &[ExportLayer],
    fp16: bool,
    int8: bool,
) -> Result<std::path::PathBuf> {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir).map_err(|e| Error::Serialization(e.to_string()))?;
    let onnx = dir.join(format!("{model_name}.onnx"));
    std::fs::write(&onnx, serialize_onnx_sequential(input_shape, layers)?).map_err(|e| Error::Serialization(e.to_string()))?;
    std::fs::write(dir.join(format!("{model_name}.trt.json")), serialize_tensorrt_manifest(model_name, input_shape, fp16, int8))
        .map_err(|e| Error::Serialization(e.to_string()))?;
    Ok(onnx)
}

/// Builds a TensorRT engine (`<model_name>.plan`) from a bundle written by [`export_tensorrt_bundle`] by running `trtexec`
/// (which must be on `PATH` or named by the `TRTEXEC` environment variable). Needs an NVIDIA GPU and TensorRT.
pub fn build_tensorrt_engine(dir: impl AsRef<Path>, model_name: &str, input_shape: &[usize], fp16: bool, int8: bool) -> Result<std::path::PathBuf> {
    let dir = dir.as_ref();
    let dims = input_shape.iter().map(usize::to_string).collect::<Vec<_>>().join("x");
    let plan = dir.join(format!("{model_name}.plan"));
    let exe = std::env::var("TRTEXEC").unwrap_or_else(|_| "trtexec".into());
    let mut cmd = std::process::Command::new(&exe);
    cmd.arg(format!("--onnx={}", dir.join(format!("{model_name}.onnx")).display()))
        .arg(format!("--saveEngine={}", plan.display()))
        .arg(format!("--shapes=input:{dims}"));
    if fp16 {
        cmd.arg("--fp16");
    }
    if int8 {
        cmd.arg("--int8");
    }
    let out = cmd.output().map_err(|e| Error::Serialization(format!("could not run '{exe}' (is TensorRT installed? set TRTEXEC): {e}")))?;
    if !out.status.success() || !plan.exists() {
        let log = String::from_utf8_lossy(&out.stdout);
        let tail: String = log.lines().rev().take(8).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n");
        return Err(Error::Serialization(format!("trtexec failed ({}):\n{tail}", out.status)));
    }
    Ok(plan)
}

// ---------------------------------------------------------------------------
// 5. TorchScript Mobile / Lite (.ptl / .pt TorchScript archive)
// ---------------------------------------------------------------------------

/// Serializes `tensors` and a TorchScript `forward` body into a `.pt` archive that `torch.jit.load` accepts (see
/// [`super::torchpt::serialize_torchscript_module`] for the body conventions). `torchscript_code` is required: a bare state
/// dict is not a TorchScript module.
pub fn serialize_torchscript_mobile(
    tensors: &HashMap<String, Tensor>,
    torchscript_code: Option<&str>,
) -> Result<Vec<u8>> {
    let body = torchscript_code.ok_or_else(|| Error::Serialization("TorchScript export needs the forward body; use serialize_torchscript_sequential for a layer list, or serialize_torch_state_dict for plain weights".into()))?;
    super::torchpt::serialize_torchscript_module(tensors, body)
}

fn script_attr(name: &str) -> String {
    let mut a: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' }).collect();
    if a.is_empty() || a.starts_with(|c: char| c.is_ascii_digit()) {
        a.insert(0, '_');
    }
    a
}

/// Exports a sequential model as a TorchScript module (`torch.jit.load`-able): each `Linear` becomes `torch.linear`,
/// `Relu` / `Gelu` / `Softmax` the matching op. Weights are stored as `<layer>.weight` / `<layer>.bias`.
pub fn serialize_torchscript_sequential(layers: &[ExportLayer]) -> Result<Vec<u8>> {
    let mut tensors = HashMap::new();
    let mut body = String::new();
    let mut cur = "x".to_string();
    for (i, layer) in layers.iter().enumerate() {
        let next = format!("h{i}");
        match layer {
            ExportLayer::Linear { name, weight, bias } => {
                let w = format!("{name}.weight");
                tensors.insert(w.clone(), weight.clone());
                let call = if let Some(b) = bias {
                    let bn = format!("{name}.bias");
                    tensors.insert(bn.clone(), b.clone());
                    format!("torch.linear({cur}, {}, {})", script_attr(&w), script_attr(&bn))
                } else {
                    format!("torch.linear({cur}, {})", script_attr(&w))
                };
                body.push_str(&format!("{next} = {call}\n"));
            }
            ExportLayer::Relu { .. } => body.push_str(&format!("{next} = torch.relu({cur})\n")),
            ExportLayer::Gelu { .. } => body.push_str(&format!("{next} = torch.gelu({cur})\n")),
            ExportLayer::Softmax { axis, .. } => body.push_str(&format!("{next} = torch.softmax({cur}, {axis}, None)\n")),
        }
        cur = next;
    }
    body.push_str(&format!("return {cur}\n"));
    super::torchpt::serialize_torchscript_module(&tensors, &body)
}

pub fn export_torchscript_mobile(
    path: impl AsRef<Path>,
    tensors: &HashMap<String, Tensor>,
    torchscript_code: Option<&str>,
) -> Result<()> {
    let bytes = serialize_torchscript_mobile(tensors, torchscript_code)?;
    std::fs::write(path, bytes).map_err(|e| Error::Serialization(e.to_string()))
}

/// Writes a sequential model as a TorchScript `.pt` file.
pub fn export_torchscript_sequential(path: impl AsRef<Path>, layers: &[ExportLayer]) -> Result<()> {
    let bytes = serialize_torchscript_sequential(layers)?;
    std::fs::write(path, bytes).map_err(|e| Error::Serialization(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exports_openvino_coreml_tflite_tensorrt_and_torchscript() {
        let w = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = Tensor::from_slice(&[0.1f32, -0.1], &[2]).unwrap();
        let layers = vec![
            ExportLayer::Linear { name: "fc1".into(), weight: w.clone(), bias: Some(b.clone()) },
            ExportLayer::Relu { name: "relu1".into() },
        ];

        let (xml, bin) = serialize_openvino_ir("test_mlp", &[1, 3], &layers).unwrap();
        assert!(xml.contains("<net name=\"test_mlp\" version=\"11\">"));
        assert!(xml.contains("type=\"MatMul\""));
        assert_eq!(bin.len(), (6 + 2) * 4);

        let mlmodel = serialize_coreml_mlmodel(&[1, 3], &layers).unwrap();
        assert!(mlmodel.len() > 20);
        assert_eq!(mlmodel[0], 0x08); // field 1 varint tag
        assert_eq!(mlmodel[1], 4); // spec version 4
        // walk the protobuf: model.neuralNetwork (500) -> layers (1) -> innerProduct (140) / activation (130)
        fn fields(mut b: &[u8]) -> Vec<(u64, Vec<u8>)> {
            fn varint(b: &mut &[u8]) -> u64 {
                let (mut v, mut shift) = (0u64, 0);
                loop {
                    let byte = b[0];
                    *b = &b[1..];
                    v |= ((byte & 0x7F) as u64) << shift;
                    if byte & 0x80 == 0 {
                        return v;
                    }
                    shift += 7;
                }
            }
            let mut out = vec![];
            while !b.is_empty() {
                let tag = varint(&mut b);
                match tag & 7 {
                    0 => out.push((tag >> 3, varint(&mut b).to_le_bytes().to_vec())),
                    2 => {
                        let n = varint(&mut b) as usize;
                        out.push((tag >> 3, b[..n].to_vec()));
                        b = &b[n..];
                    }
                    w => panic!("unexpected wire type {w}"),
                }
            }
            out
        }
        let top = fields(&mlmodel);
        let nn = &top.iter().find(|(f, _)| *f == 500).expect("neuralNetwork").1;
        let layer_msgs: Vec<_> = fields(nn).into_iter().filter(|(f, _)| *f == 1).map(|(_, m)| fields(&m)).collect();
        assert_eq!(layer_msgs.len(), 2);
        let ip = &layer_msgs[0].iter().find(|(f, _)| *f == 140).expect("innerProduct layer").1;
        let ipf = fields(ip);
        assert_eq!(u64::from_le_bytes(ipf.iter().find(|(f, _)| *f == 1).unwrap().1[..8].try_into().unwrap()), 3, "inputChannels");
        assert_eq!(u64::from_le_bytes(ipf.iter().find(|(f, _)| *f == 2).unwrap().1[..8].try_into().unwrap()), 2, "outputChannels");
        assert_eq!(fields(&ipf.iter().find(|(f, _)| *f == 20).unwrap().1)[0].1.len(), 6 * 4, "weights.floatValue");
        assert_eq!(fields(&ipf.iter().find(|(f, _)| *f == 21).unwrap().1)[0].1.len(), 2 * 4, "bias.floatValue");
        let act = &layer_msgs[1].iter().find(|(f, _)| *f == 130).expect("activation layer").1;
        assert!(fields(act).iter().any(|(f, _)| *f == 10), "ReLU parameter");
        assert!(serialize_coreml_mlmodel(&[1, 3], &[ExportLayer::Gelu { name: "g".into() }]).is_err());

        let mut map = HashMap::new();
        map.insert("fc1.weight".into(), w);
        map.insert("fc1.bias".into(), b);
        let tfl = serialize_tflite_flatbuffer(&[1, 3], &layers).unwrap();
        assert_eq!(&tfl[4..8], b"TFL3");

        let trt = serialize_tensorrt_manifest("test_mlp", &[1, 3], true, false);
        assert!(trt.contains("trtexec"));
        assert!(trt.contains("--fp16"));

        assert!(serialize_torchscript_mobile(&map, None).is_err(), "a state dict alone is not a TorchScript module");
        let ptl = serialize_torchscript_sequential(&layers).unwrap();
        assert_eq!(&ptl[0..2], b"PK");
    }
}
