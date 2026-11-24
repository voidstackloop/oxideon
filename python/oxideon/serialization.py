"""Model serialization and checkpointing for Oxideon.

Provides HuggingFace-compatible `save_file` and `load_file` functions,
full model checkpointing via `save` and `load`, safetensors metadata inspection,
and high-performance in-memory buffer serialization.
"""

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import json
import os
import pickle
import tempfile

from .oxideon import (
    Tensor,
    save_safetensors as _save_safetensors_raw,
    load_safetensors as _load_safetensors_raw,
    load_safetensors_metadata as _load_safetensors_metadata_raw,
    from_numpy,
)


def _reconstruct_tensor(arr, requires_grad: bool = False):
    """Reconstruct an Oxideon Tensor from pickled NumPy data."""
    t = from_numpy(arr)
    if requires_grad:
        t = t.requires_grad_(True)
    return t


def save_file(
    tensors: Union[Dict[str, Tensor], Sequence[Tuple[str, Tensor]], Any],
    filename: Union[str, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """Saves a dictionary or module of tensors to a file in `.safetensors` format."""
    if isinstance(tensors, (str, os.PathLike)) and not isinstance(filename, (str, os.PathLike)):
        tensors, filename = filename, tensors

    raw_path_str = str(filename)
    is_cloud = raw_path_str.startswith("s3://") or raw_path_str.startswith("gs://")

    if not is_cloud:
        path_str = str(Path(filename).expanduser().resolve())
    else:
        path_str = raw_path_str

    if hasattr(tensors, "state_dict") and callable(tensors.state_dict):
        obj_name = type(tensors).__name__
        tensors = tensors.state_dict()
        if metadata is None:
            metadata = {}
        metadata.setdefault("oxideon_class", obj_name)
        metadata.setdefault("format", "safetensors")

    if metadata is not None:
        clean_metadata = {str(k): str(v) for k, v in metadata.items()}
    else:
        clean_metadata = None

    if is_cloud:
        suffix = Path(path_str).suffix or ".safetensors"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
        try:
            _save_safetensors_raw(tmp_path, tensors, clean_metadata)
            if path_str.startswith("s3://"):
                from .cloud.aws import s3_upload
                s3_upload(tmp_path, path_str)
            else:
                from .cloud.gcp import gcs_upload
                gcs_upload(tmp_path, path_str)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        return

    _save_safetensors_raw(path_str, tensors, clean_metadata)


def load_file(
    filename: Union[str, os.PathLike],
    device: Optional[str] = None,
    model: Optional[Any] = None,
    strict: bool = True,
) -> Union[Dict[str, Tensor], Any]:
    """Loads a `.safetensors` file into an OrderedDict of tensors, or directly into a model."""
    raw_path_str = str(filename)
    if raw_path_str.startswith("s3://") or raw_path_str.startswith("gs://"):
        suffix = Path(raw_path_str).suffix or ".safetensors"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
        try:
            if raw_path_str.startswith("s3://"):
                from .cloud.aws import s3_download
                s3_download(raw_path_str, tmp_path)
            else:
                from .cloud.gcp import gcs_download
                gcs_download(raw_path_str, tmp_path)
            return load_file(tmp_path, device=device, model=model, strict=strict)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    path_str = str(Path(filename).expanduser().resolve())
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"No such file: '{path_str}'")

    raw_pairs = _load_safetensors_raw(path_str)
    result = OrderedDict()
    for k, v in raw_pairs:
        if device is not None and hasattr(v, "to_device"):
            v = v.to_device(device)
        result[k] = v

    if model is not None:
        if hasattr(model, "load_state_dict") and callable(model.load_state_dict):
            model.load_state_dict(result, strict=strict)
            if device is not None and hasattr(model, "to"):
                model.to(device=device)
            return model
        else:
            raise TypeError(f"Target model of type {type(model).__name__} lacks load_state_dict() method.")

    return result


def load_file_metadata(filename: Union[str, os.PathLike]) -> Dict[str, str]:
    """Reads only the header metadata from a `.safetensors` file without loading tensor buffers."""
    raw_path_str = str(filename)
    if raw_path_str.startswith("s3://") or raw_path_str.startswith("gs://"):
        suffix = Path(raw_path_str).suffix or ".safetensors"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
        try:
            if raw_path_str.startswith("s3://"):
                from .cloud.aws import s3_download
                s3_download(raw_path_str, tmp_path)
            else:
                from .cloud.gcp import gcs_download
                gcs_download(raw_path_str, tmp_path)
            return load_file_metadata(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    path_str = str(Path(filename).expanduser().resolve())
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"No such file: '{path_str}'")
    return dict(_load_safetensors_metadata_raw(path_str))


def save_safetensors_bytes(
    tensors: Union[Dict[str, Tensor], Sequence[Tuple[str, Tensor]], Any],
    metadata: Optional[Dict[str, str]] = None,
) -> bytes:
    """Serializes tensors into standard safetensors binary bytes in-memory."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_name = tmp.name
    try:
        save_file(tensors, tmp_name, metadata=metadata)
        with open(tmp_name, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def load_safetensors_bytes(
    data: bytes,
    device: Optional[str] = None,
) -> Dict[str, Tensor]:
    """Loads an in-memory safetensors byte buffer into an OrderedDict of tensors."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_name = tmp.name
    try:
        with open(tmp_name, "wb") as f:
            f.write(data)
        return load_file(tmp_name, device=device)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def inspect_safetensors(filename: Union[str, os.PathLike]) -> Dict[str, Any]:
    """Inspects header metadata, tensor shapes, and datatypes without loading weight tensors into memory."""
    path = Path(filename).expanduser().resolve()
    path_str = str(path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"No such file: '{path_str}'")

    with open(path_str, "rb") as f:
        header_size_bytes = f.read(8)
        if len(header_size_bytes) < 8:
            raise ValueError(f"Invalid safetensors header in '{path_str}'")
        header_size = int.from_bytes(header_size_bytes, byteorder="little")
        header_json_bytes = f.read(header_size)
        header = json.loads(header_json_bytes.decode("utf-8"))

    metadata = header.pop("__metadata__", {})
    return {
        "filename": path_str,
        "metadata": metadata,
        "tensors": header,
    }


def save(
    obj: Any,
    filename: Union[str, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """Saves an object (Module, state dict, Tensor, optimizer, or checkpoint dict) to a file."""
    path = Path(filename).expanduser().resolve()
    path_str = str(path)

    if str(path).endswith(".safetensors"):
        save_file(obj, path_str, metadata=metadata)
        return

    if hasattr(obj, "state_dict") and callable(obj.state_dict) and not str(path).endswith((".pt", ".pth", ".pkl", ".ckpt")):
        save_file(obj, path_str, metadata=metadata)
        return

    if path.suffix.lower() not in PICKLE_EXTENSIONS:
        # `load` only unpickles files with a checkpoint extension, so a pickle written under any other name could never
        # be read back: store tensor collections as safetensors, refuse the rest.
        if isinstance(obj, dict) and obj and all(isinstance(v, Tensor) for v in obj.values()):
            save_file(obj, path_str, metadata=metadata)
            return
        raise ValueError(
            f"save: '{path.name}' would be written with pickle, which load() only reads from "
            f"{', '.join(sorted(PICKLE_EXTENSIONS))} files; use one of those extensions (or .safetensors for tensors)")

    with open(path_str, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


#: File extensions that may contain pickled checkpoints (see :func:`save`).
PICKLE_EXTENSIONS = frozenset({".pt", ".pth", ".pkl", ".ckpt", ".ox", ".bin"})


def load(
    filename: Union[str, os.PathLike],
    model: Optional[Any] = None,
    device: Optional[str] = None,
    strict: bool = True,
    weights_only: bool = False,
) -> Any:
    """Loads a saved AI model, tensor collection, or checkpoint from a file."""
    path = Path(filename).expanduser().resolve()
    path_str = str(path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"No such file: '{path_str}'")

    # Only PARSE failures fall back to pickle for torch-style checkpoint
    # extensions; model-application errors (strict mismatches) raised by
    # `load_file` must propagate, never masquerade as unpickle errors.
    try:
        result = load_file(path_str, device=device)
    except Exception as exc:
        if Path(path_str).suffix.lower() not in PICKLE_EXTENSIONS:
            raise
        if weights_only:
            raise ValueError(
                f"Refusing to unpickle '{path_str}' with weights_only=True; "
                "pass weights_only=False only for files you trust."
            ) from exc
        with open(path_str, "rb") as f:
            obj = pickle.load(f)
        if model is not None:
            if isinstance(obj, dict) and "state_dict" in obj:
                model.load_state_dict(obj["state_dict"], strict=strict)
            elif isinstance(obj, dict):
                model.load_state_dict(obj, strict=strict)
            else:
                raise TypeError(f"Cannot load {type(obj).__name__} into model; expected a state dict.")
            if device is not None and hasattr(model, "to"):
                model.to(device=device)
            return model
        return obj

    if model is not None:
        if hasattr(model, "load_state_dict") and callable(model.load_state_dict):
            model.load_state_dict(result, strict=strict)
            if device is not None and hasattr(model, "to"):
                model.to(device=device)
            return model
        raise TypeError(f"Target model of type {type(model).__name__} lacks load_state_dict() method.")
    return result


__all__ = [
    "save_file",
    "load_file",
    "load_file_metadata",
    "save_safetensors_bytes",
    "load_safetensors_bytes",
    "inspect_safetensors",
    "save",
    "load",
    "PICKLE_EXTENSIONS",
]
