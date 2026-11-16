"""
AWS Cloud Integration for Oxideon.
Provides Amazon S3 checkpoint streaming and persistence,
AWS SageMaker environment context detection, and SageMaker training callbacks.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import os
import shutil

_MOCK_S3_STORAGE: Dict[str, Dict[str, bytes]] = {}

def parse_s3_uri(uri: str) -> Tuple[str, str]:
    """Parses an s3://bucket/key URI into (bucket, key)."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI (expected prefix 's3://'): '{uri}'")
    without_prefix = uri[5:]
    parts = without_prefix.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key

class S3Client:
    """High-level S3 client with automatic boto3 integration and in-memory mock fallback."""
    def __init__(self, region_name: Optional[str] = None):
        self.region_name = region_name or os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")
        self._boto_client = None

        try:
            import boto3
            self._boto_client = boto3.client("s3", region_name=self.region_name)
        except Exception:
            self._boto_client = None

    def upload_file(self, local_path: str, s3_uri: str) -> str:
        """Uploads a local file to S3."""
        bucket, key = parse_s3_uri(s3_uri)
        local_p = Path(local_path).resolve()
        if not local_p.exists():
            raise FileNotFoundError(f"Local file does not exist: '{local_path}'")

        if self._boto_client is not None:
            self._boto_client.upload_file(str(local_p), bucket, key)
        else:
            with open(local_p, "rb") as f:
                data = f.read()
            if bucket not in _MOCK_S3_STORAGE:
                _MOCK_S3_STORAGE[bucket] = {}
            _MOCK_S3_STORAGE[bucket][key] = data

        return s3_uri

    def download_file(self, s3_uri: str, local_path: str) -> str:
        """Downloads an S3 object to a local destination file."""
        bucket, key = parse_s3_uri(s3_uri)
        dest_p = Path(local_path).resolve()
        dest_p.parent.mkdir(parents=True, exist_ok=True)

        if self._boto_client is not None:
            self._boto_client.download_file(bucket, key, str(dest_p))
        else:
            if bucket not in _MOCK_S3_STORAGE or key not in _MOCK_S3_STORAGE[bucket]:
                raise FileNotFoundError(f"S3 object '{s3_uri}' not found in storage.")
            data = _MOCK_S3_STORAGE[bucket][key]
            with open(dest_p, "wb") as f:
                f.write(data)

        return str(dest_p)

    def exists(self, s3_uri: str) -> bool:
        """Checks if an S3 object exists."""
        bucket, key = parse_s3_uri(s3_uri)
        if self._boto_client is not None:
            try:
                self._boto_client.head_object(Bucket=bucket, Key=key)
                return True
            except Exception:
                return False
        else:
            return bucket in _MOCK_S3_STORAGE and key in _MOCK_S3_STORAGE[bucket]

    def delete(self, s3_uri: str) -> bool:
        """Deletes an S3 object."""
        bucket, key = parse_s3_uri(s3_uri)
        if self._boto_client is not None:
            try:
                self._boto_client.delete_object(Bucket=bucket, Key=key)
                return True
            except Exception:
                return False
        else:
            if bucket in _MOCK_S3_STORAGE and key in _MOCK_S3_STORAGE[bucket]:
                del _MOCK_S3_STORAGE[bucket][key]
                return True
            return False

    def list_objects(self, s3_prefix: str) -> List[str]:
        """Lists S3 keys matching prefix."""
        bucket, prefix = parse_s3_uri(s3_prefix)
        if self._boto_client is not None:
            resp = self._boto_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            contents = resp.get("Contents", [])
            return [f"s3://{bucket}/{item['Key']}" for item in contents]
        else:
            if bucket not in _MOCK_S3_STORAGE:
                return []
            return [
                f"s3://{bucket}/{k}"
                for k in _MOCK_S3_STORAGE[bucket].keys()
                if k.startswith(prefix)
            ]

_DEFAULT_S3_CLIENT = S3Client()

def s3_upload(local_path: str, s3_uri: str) -> str:
    return _DEFAULT_S3_CLIENT.upload_file(local_path, s3_uri)

def s3_download(s3_uri: str, local_path: str) -> str:
    return _DEFAULT_S3_CLIENT.download_file(s3_uri, local_path)

def s3_exists(s3_uri: str) -> bool:
    return _DEFAULT_S3_CLIENT.exists(s3_uri)

def clear_mock_s3_storage():
    """Clears in-memory simulated S3 storage (useful for unit tests)."""
    _MOCK_S3_STORAGE.clear()


@dataclass
class SageMakerContext:
    """Runtime context for jobs executing inside AWS SageMaker."""
    model_dir: str = "/opt/ml/model"
    checkpoint_dir: Optional[str] = "/opt/ml/checkpoints"
    training_channel: Optional[str] = "/opt/ml/input/data/training"
    validation_channel: Optional[str] = "/opt/ml/input/data/validation"
    num_gpus: int = 0
    num_cpus: int = 1
    current_host: str = "algo-1"
    hosts: List[str] = field(default_factory=lambda: ["algo-1"])

def is_sagemaker() -> bool:
    """Returns True if running within an AWS SageMaker container."""
    return any(
        k in os.environ
        for k in ("SM_MODEL_DIR", "SM_CHANNEL_TRAINING", "SM_CURRENT_HOST", "SAGEMAKER_JOB_NAME")
    )

def get_sagemaker_context() -> SageMakerContext:
    """Reads environment variables configured by AWS SageMaker."""
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    ckpt_dir = os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints")
    train_ch = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    val_ch = os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation")
    num_gpus = int(os.environ.get("SM_NUM_GPUS", "0"))
    num_cpus = int(os.environ.get("SM_NUM_CPUS", "1"))
    curr_host = os.environ.get("SM_CURRENT_HOST", "algo-1")

    hosts_raw = os.environ.get("SM_HOSTS")
    if hosts_raw:
        try:
            hosts = json.loads(hosts_raw)
        except Exception:
            hosts = [curr_host]
    else:
        hosts = [curr_host]

    return SageMakerContext(
        model_dir=model_dir,
        checkpoint_dir=ckpt_dir,
        training_channel=train_ch,
        validation_channel=val_ch,
        num_gpus=num_gpus,
        num_cpus=num_cpus,
        current_host=curr_host,
        hosts=hosts,
    )

class SageMakerCallback:
    """
    Oxideon training callback for AWS SageMaker.
    Emits CloudWatch-parseable metric strings and saves checkpoints to SageMaker artifact paths.
    """
    def __init__(self, model_name: str = "model.safetensors", save_checkpoints: bool = True):
        self.context = get_sagemaker_context()
        self.model_name = model_name
        self.save_checkpoints = save_checkpoints
        self.logged_metrics: List[Dict[str, float]] = []

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], model=None):
        """Logs metrics in standard CloudWatch Regex format: metric_name=value."""
        self.logged_metrics.append(metrics)
        metric_str = " ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        print(f"[SageMaker Epoch {epoch}] {metric_str}")

        if self.save_checkpoints and model is not None and self.context.checkpoint_dir:
            from ..serialization import save_file
            ckpt_path = Path(self.context.checkpoint_dir) / f"checkpoint_epoch_{epoch}.safetensors"
            try:
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                save_file(model, str(ckpt_path))
            except Exception as e:
                print(f"Warning: Failed to save SageMaker checkpoint to {ckpt_path}: {e}")

    def on_train_end(self, model=None):
        """Saves final model weights directly to SM_MODEL_DIR for SageMaker model packaging."""
        if model is not None and self.context.model_dir:
            from ..serialization import save_file
            out_path = Path(self.context.model_dir) / self.model_name
            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                save_file(model, str(out_path))
                print(f"Saved final SageMaker model artifact to: {out_path}")
            except Exception as e:
                print(f"Warning: Failed to save SageMaker model artifact to {out_path}: {e}")
