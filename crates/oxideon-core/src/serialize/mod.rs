//! Model serialization and deserialization formats:
//! - Hugging Face `.safetensors`
//! - GGML Universal Format `.gguf` (v2 and v3)

pub mod deploy_export;
pub(crate) mod flatbuffer;
pub mod gguf;
pub mod mmap;
pub mod npy;
pub mod onnx;
pub mod onnx_import;
mod gguf_iq_tables;
pub mod safetensors;
pub mod torchpt;

pub use deploy_export::{
    ExportLayer, export_coreml_mlmodel, export_openvino_ir, export_tflite_flatbuffer,
    export_torchscript_mobile, export_torchscript_sequential, serialize_coreml_mlmodel, serialize_openvino_ir,
    serialize_torchscript_sequential, serialize_onnx_sequential, export_tensorrt_bundle, build_tensorrt_engine,
    serialize_tensorrt_manifest, serialize_tflite_flatbuffer, serialize_torchscript_mobile,
};
pub use mmap::{GgufMmap, MmapFile, SafetensorsMmap};
pub use npy::{from_npy_bytes, from_npz_bytes, load_npy, load_npz, npy_bytes, npz_bytes, save_npy, save_npz};
pub use onnx_import::OnnxModel;
pub use torchpt::{
    serialize_torchscript_module,
    load_torch_state_dict, load_torch_tensors, load_torch_tensors_from_bytes,
    save_torch_state_dict, serialize_torch_state_dict,
};
pub use gguf::{GGMLType, GGUFFile, GGUFTensorInfo, GGUFValue, save_gguf, serialize_gguf};
pub use safetensors::{
    deserialize_safetensors, deserialize_safetensors_with_metadata, load_safetensors,
    load_safetensors_with_metadata, save_safetensors, save_safetensors_with_metadata,
    serialize_safetensors, serialize_safetensors_with_metadata,
};
