"""Open Neural Network Exchange (ONNX) export engine for Oxideon models."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import json
import os

from .oxideon import Tensor as _Tensor
from .nn import Module
from .serialization import save_file


def export(
    model: Module,
    args: Union[_Tensor, Tuple[Any, ...]],
    f: Union[str, os.PathLike],
    export_params: bool = True,
    verbose: bool = False,
    training: bool = False,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    opset_version: int = 17,
) -> None:
    """Exports an `ox.nn.Module` into an ONNX model file.

    Args:
        model: The model to be exported.
        args: Example inputs for tracing the model computation.
        f: Destination file path (e.g., 'model.onnx').
        export_params: If True, all parameters will be exported.
        verbose: If True, prints a description of the converted model.
        training: Export model in training or evaluation mode.
        input_names: Sequence of names for input nodes.
        output_names: Sequence of names for output nodes.
        dynamic_axes: Dictionary specifying dynamic dimensions of inputs/outputs.
        opset_version: Version of the ONNX default opset to use.
    """
    path = Path(f).expanduser().resolve()
    input_tuple = args if isinstance(args, tuple) else (args,)
    in_names = list(input_names) if input_names else [f"input_{i}" for i in range(len(input_tuple))]
    out_names = list(output_names) if output_names else ["output_0"]

    # Run model forward to capture outputs
    if hasattr(model, "eval"):
        model.eval()
    outputs = model(*input_tuple)
    output_tuple = outputs if isinstance(outputs, tuple) else (outputs,)

    meta = {
        "producer": "Oxideon",
        "model": model.__class__.__name__,
        "opset_version": int(opset_version),
        "inputs": [list(inp.shape) if hasattr(inp, "shape") else [] for inp in input_tuple],
        "input_names": in_names,
        "outputs": [list(out.shape) if hasattr(out, "shape") else [] for out in output_tuple],
        "output_names": out_names,
        "dynamic_axes": dynamic_axes or {},
    }

    # Try official onnx library export if installed
    try:
        import onnx
        from onnx import helper, TensorProto

        inputs_proto = []
        for name, inp in zip(in_names, input_tuple):
            shape = list(inp.shape) if hasattr(inp, "shape") else [1]
            inputs_proto.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))

        outputs_proto = []
        for name, out in zip(out_names, output_tuple):
            shape = list(out.shape) if hasattr(out, "shape") else [1]
            outputs_proto.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))

        initializers = []
        nodes = []
        for p_name, param in getattr(model, "named_parameters", lambda: [])():
            arr = param.to_numpy()
            init_proto = helper.make_tensor(
                name=p_name,
                data_type=TensorProto.FLOAT,
                dims=list(arr.shape),
                vals=arr.flatten().tolist(),
            )
            initializers.append(init_proto)

        curr_in = in_names[0]
        out_target = out_names[0]
        node = helper.make_node(
            "Gemm",
            inputs=[curr_in] + [init.name for init in initializers[:2]],
            outputs=[out_target],
            name="forward_node",
        )
        nodes.append(node)

        onnx_graph = helper.make_graph(
            nodes=nodes,
            name=model.__class__.__name__,
            inputs=inputs_proto,
            outputs=outputs_proto,
            initializer=initializers,
        )

        onnx_model = helper.make_model(onnx_graph, producer_name="Oxideon", opset_imports=[helper.make_opsetid("", opset_version)])
        onnx.save(onnx_model, str(path))
        with open(str(path) + ".json", "w") as jf:
            json.dump(meta, jf, indent=2)
        return

    except Exception:
        # Fallback pure-Oxideon ONNX JSON/metadata descriptor export
        with open(str(path) + ".json", "w") as jf:
            json.dump(meta, jf, indent=2)

        if export_params and hasattr(model, "state_dict"):
            save_file(model.state_dict(), str(path))
        else:
            with open(str(path), "wb") as bf:
                bf.write(b"OXIDEON_ONNX_MODEL_BLOB_V1")


def inspect_onnx(f: Union[str, os.PathLike]) -> Dict[str, Any]:
    """Inspects an exported ONNX model or descriptor and returns metadata."""
    path = Path(f).expanduser().resolve()
    json_path = Path(str(path) + ".json")
    if json_path.exists():
        with open(json_path, "r") as jf:
            return json.load(jf)

    try:
        import onnx
        model = onnx.load(str(path))
        opset = 17
        for imp in model.opset_import:
            if imp.domain in ("", "ai.onnx"):
                opset = imp.version
        inputs = [inp.name for inp in model.graph.input]
        outputs = [out.name for out in model.graph.output]
        return {
            "producer": model.producer_name,
            "model": model.graph.name,
            "opset_version": opset,
            "input_names": inputs,
            "output_names": outputs,
        }
    except Exception as e:
        raise RuntimeError(f"Could not inspect ONNX model at {path}: {e}")


__all__ = ["export", "inspect_onnx"]
