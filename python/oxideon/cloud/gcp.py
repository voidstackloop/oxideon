"""
GCP (Google Cloud Platform) Integration for Oxideon.
Provides Google Cloud Storage (GCS) checkpoint streaming and persistence,
Vertex AI custom training environment context detection, and Vertex AI callbacks.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os

_MOCK_GCS_STORAGE: Dict[str, Dict[str, bytes]] = {}

def parse_gcs_uri(uri: str) -> Tuple[str, str]:
    """Parses a gs://bucket/key URI into (bucket, blob_name)."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI (expected prefix 'gs://'): '{uri}'")
    without_prefix = uri[5:]
    parts = without_prefix.split("/", 1)
    bucket = parts[0]
    blob = parts[1] if len(parts) > 1 else ""
    return bucket, blob

class GCSClient:
    """High-level GCS client with automatic google-cloud-storage integration and in-memory mock fallback."""
    def __init__(self, project: Optional[str] = None):
        self.project = project or os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("CLOUD_ML_PROJECT_ID")
        self._gcs_client = None

        try:
            from google.cloud import storage
            self._gcs_client = storage.Client(project=self.project)
        except Exception:
            self._gcs_client = None

    def upload_file(self, local_path: str, gcs_uri: str) -> str:
        """Uploads a local file to GCS."""
        bucket_name, blob_name = parse_gcs_uri(gcs_uri)
        local_p = Path(local_path).resolve()
        if not local_p.exists():
            raise FileNotFoundError(f"Local file does not exist: '{local_path}'")

        if self._gcs_client is not None:
            bucket = self._gcs_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(local_p))
        else:
            with open(local_p, "rb") as f:
                data = f.read()
            if bucket_name not in _MOCK_GCS_STORAGE:
                _MOCK_GCS_STORAGE[bucket_name] = {}
            _MOCK_GCS_STORAGE[bucket_name][blob_name] = data

        return gcs_uri

    def download_file(self, gcs_uri: str, local_path: str) -> str:
        """Downloads a GCS blob to a local destination file."""
        bucket_name, blob_name = parse_gcs_uri(gcs_uri)
        dest_p = Path(local_path).resolve()
        dest_p.parent.mkdir(parents=True, exist_ok=True)

        if self._gcs_client is not None:
            bucket = self._gcs_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(str(dest_p))
        else:
            if bucket_name not in _MOCK_GCS_STORAGE or blob_name not in _MOCK_GCS_STORAGE[bucket_name]:
                raise FileNotFoundError(f"GCS object '{gcs_uri}' not found in storage.")
            data = _MOCK_GCS_STORAGE[bucket_name][blob_name]
            with open(dest_p, "wb") as f:
                f.write(data)

        return str(dest_p)

    def exists(self, gcs_uri: str) -> bool:
        """Checks if a GCS object exists."""
        bucket_name, blob_name = parse_gcs_uri(gcs_uri)
        if self._gcs_client is not None:
            try:
                bucket = self._gcs_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                return blob.exists()
            except Exception:
                return False
        else:
            return bucket_name in _MOCK_GCS_STORAGE and blob_name in _MOCK_GCS_STORAGE[bucket_name]

    def delete(self, gcs_uri: str) -> bool:
        """Deletes a GCS blob."""
        bucket_name, blob_name = parse_gcs_uri(gcs_uri)
        if self._gcs_client is not None:
            try:
                bucket = self._gcs_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                blob.delete()
                return True
            except Exception:
                return False
        else:
            if bucket_name in _MOCK_GCS_STORAGE and blob_name in _MOCK_GCS_STORAGE[bucket_name]:
                del _MOCK_GCS_STORAGE[bucket_name][blob_name]
                return True
            return False

    def list_objects(self, gcs_prefix: str) -> List[str]:
        """Lists GCS blobs matching a prefix."""
        bucket_name, prefix = parse_gcs_uri(gcs_prefix)
        if self._gcs_client is not None:
            bucket = self._gcs_client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=prefix)
            return [f"gs://{bucket_name}/{b.name}" for b in blobs]
        else:
            if bucket_name not in _MOCK_GCS_STORAGE:
                return []
            return [
                f"gs://{bucket_name}/{k}"
                for k in _MOCK_GCS_STORAGE[bucket_name].keys()
                if k.startswith(prefix)
            ]

_DEFAULT_GCS_CLIENT = GCSClient()

def gcs_upload(local_path: str, gcs_uri: str) -> str:
    return _DEFAULT_GCS_CLIENT.upload_file(local_path, gcs_uri)

def gcs_download(gcs_uri: str, local_path: str) -> str:
    return _DEFAULT_GCS_CLIENT.download_file(gcs_uri, local_path)

def gcs_exists(gcs_uri: str) -> bool:
    return _DEFAULT_GCS_CLIENT.exists(gcs_uri)

def clear_mock_gcs_storage():
    """Clears in-memory simulated GCS storage."""
    _MOCK_GCS_STORAGE.clear()


@dataclass
class VertexAIContext:
    """Runtime context for jobs executing inside GCP Vertex AI Custom Training."""
    model_dir: str = "/gcs/model"
    checkpoint_dir: Optional[str] = "/gcs/checkpoints"
    tensorboard_log_dir: Optional[str] = None
    project_id: Optional[str] = None
    region: Optional[str] = None

def is_vertex_ai() -> bool:
    """Returns True if running within a Google Cloud Vertex AI training container."""
    return any(
        k in os.environ
        for k in ("AIP_MODEL_DIR", "AIP_CHECKPOINT_DIR", "CLOUD_ML_PROJECT_ID", "VERTEX_AI_ENV")
    )

def is_tpu_available() -> bool:
    """Checks whether GCP TPU VM accelerators are present."""
    return "TPU_NAME" in os.environ or Path("/sys/class/accel/accel0").exists()

def get_vertex_ai_context() -> VertexAIContext:
    """Reads environment variables configured by GCP Vertex AI."""
    model_dir = os.environ.get("AIP_MODEL_DIR", "/gcs/model")
    ckpt_dir = os.environ.get("AIP_CHECKPOINT_DIR", "/gcs/checkpoints")
    tb_dir = os.environ.get("AIP_TENSORBOARD_LOG_DIR")
    proj = os.environ.get("CLOUD_ML_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    region = os.environ.get("CLOUD_ML_REGION") or os.environ.get("GOOGLE_CLOUD_REGION")

    return VertexAIContext(
        model_dir=model_dir,
        checkpoint_dir=ckpt_dir,
        tensorboard_log_dir=tb_dir,
        project_id=proj,
        region=region,
    )

class VertexAICallback:
    """
    Oxideon training callback for GCP Vertex AI.
    Handles artifact uploading to AIP_MODEL_DIR and periodic checkpoints to AIP_CHECKPOINT_DIR.
    """
    def __init__(self, model_name: str = "model.safetensors", save_checkpoints: bool = True):
        self.context = get_vertex_ai_context()
        self.model_name = model_name
        self.save_checkpoints = save_checkpoints
        self.logged_metrics: List[Dict[str, float]] = []

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], model=None):
        self.logged_metrics.append(metrics)
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"[Vertex AI Epoch {epoch}] {metric_str}")

        if self.save_checkpoints and model is not None and self.context.checkpoint_dir:
            from ..serialization import save_file
            ckpt_uri = f"{self.context.checkpoint_dir.rstrip('/')}/checkpoint_epoch_{epoch}.safetensors"
            try:
                if not ckpt_uri.startswith("gs://"):
                    Path(ckpt_uri).parent.mkdir(parents=True, exist_ok=True)
                save_file(model, ckpt_uri)
            except Exception as e:
                print(f"Warning: Failed to save Vertex AI checkpoint to {ckpt_uri}: {e}")

    def on_train_end(self, model=None):
        if model is not None and self.context.model_dir:
            from ..serialization import save_file
            out_uri = f"{self.context.model_dir.rstrip('/')}/{self.model_name}"
            try:
                if not out_uri.startswith("gs://"):
                    Path(out_uri).parent.mkdir(parents=True, exist_ok=True)
                save_file(model, out_uri)
                print(f"Saved final Vertex AI model artifact to: {out_uri}")
            except Exception as e:
                print(f"Warning: Failed to save Vertex AI model artifact to {out_uri}: {e}")
