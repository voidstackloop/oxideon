"""
Unit tests for AWS and GCP Cloud Integrations in Oxideon:
- AWS S3 streaming persistence and URI parsing
- GCP GCS streaming persistence and URI parsing
- Seamless ox.save_file and ox.load_file directly to/from s3:// and gs:// URIs
- AWS SageMaker runtime context and training callbacks
- GCP Vertex AI runtime context and training callbacks
- Cloud environment and accelerator detection
"""
import os
from pathlib import Path
import numpy as np
import pytest
import oxideon as ox
from oxideon.cloud import aws, gcp
from oxideon.cloud.aws import clear_mock_s3_storage
from oxideon.cloud.gcp import clear_mock_gcs_storage

@pytest.fixture(autouse=True)
def cleanup_cloud_storage():
    clear_mock_s3_storage()
    clear_mock_gcs_storage()
    yield
    clear_mock_s3_storage()
    clear_mock_gcs_storage()

def test_s3_uri_parsing_and_mock_io(tmp_path):
    # URI parsing
    b, k = aws.parse_s3_uri("s3://my-data-bucket/models/checkpoint.safetensors")
    assert b == "my-data-bucket"
    assert k == "models/checkpoint.safetensors"

    with pytest.raises(ValueError):
        aws.parse_s3_uri("https://s3.amazonaws.com/bucket/key")

    # Local file upload/download
    test_file = tmp_path / "data.bin"
    test_file.write_bytes(b"HELLO AWS S3 DATA")

    uri = "s3://my-data-bucket/test_data.bin"
    assert not aws.s3_exists(uri)
    aws.s3_upload(str(test_file), uri)
    assert aws.s3_exists(uri)

    dest_file = tmp_path / "downloaded.bin"
    aws.s3_download(uri, str(dest_file))
    assert dest_file.read_bytes() == b"HELLO AWS S3 DATA"

    # List objects
    client = aws.S3Client()
    objs = client.list_objects("s3://my-data-bucket/test")
    assert uri in objs

    # Delete
    assert client.delete(uri) is True
    assert not aws.s3_exists(uri)

def test_gcs_uri_parsing_and_mock_io(tmp_path):
    # URI parsing
    b, k = gcp.parse_gcs_uri("gs://my-vertex-bucket/checkpoints/epoch_1.safetensors")
    assert b == "my-vertex-bucket"
    assert k == "checkpoints/epoch_1.safetensors"

    with pytest.raises(ValueError):
        gcp.parse_gcs_uri("s3://bucket/key")

    # Local file upload/download
    test_file = tmp_path / "gcs_sample.txt"
    test_file.write_text("HELLO GCP GCS DATA")

    uri = "gs://my-vertex-bucket/gcs_sample.txt"
    assert not gcp.gcs_exists(uri)
    gcp.gcs_upload(str(test_file), uri)
    assert gcp.gcs_exists(uri)

    dest_file = tmp_path / "gcs_downloaded.txt"
    gcp.gcs_download(uri, str(dest_file))
    assert dest_file.read_text() == "HELLO GCP GCS DATA"

    # List objects
    client = gcp.GCSClient()
    objs = client.list_objects("gs://my-vertex-bucket/gcs")
    assert uri in objs

    # Delete
    assert client.delete(uri) is True
    assert not gcp.gcs_exists(uri)

def test_save_file_and_load_file_direct_s3_uri():
    # Model
    model = ox.nn.Sequential(
        ox.nn.Linear(4, 8),
        ox.nn.ReLU(),
        ox.nn.Linear(8, 2),
    )
    s3_path = "s3://prod-weights-bucket/networks/classifier.safetensors"

    # Direct save to S3
    ox.save_file(model, s3_path, metadata={"project": "oxideon-aws", "accuracy": "0.985"})
    assert aws.s3_exists(s3_path)

    # Inspect metadata from S3
    meta = ox.load_file_metadata(s3_path)
    assert meta.get("project") == "oxideon-aws"
    assert meta.get("accuracy") == "0.985"

    # Direct load into new model
    new_model = ox.nn.Sequential(
        ox.nn.Linear(4, 8),
        ox.nn.ReLU(),
        ox.nn.Linear(8, 2),
    )
    ox.load_file(s3_path, model=new_model)

    x = ox.randn([3, 4])
    y_orig = model(x).to_numpy()
    y_loaded = new_model(x).to_numpy()
    assert np.allclose(y_orig, y_loaded)

def test_save_file_and_load_file_direct_gcs_uri():
    # State dictionary
    w1 = ox.randn([10, 5])
    w2 = ox.randn([5, 2])
    state = {"layer1.weight": w1, "layer2.weight": w2}

    gcs_path = "gs://prod-vertex-bucket/models/custom_net.safetensors"

    # Direct save to GCS
    ox.save_file(state, gcs_path, metadata={"cloud": "gcp-vertex-ai"})
    assert gcp.gcs_exists(gcs_path)

    # Direct load from GCS into state dict
    loaded_state = ox.load_file(gcs_path)
    assert "layer1.weight" in loaded_state
    assert np.allclose(loaded_state["layer1.weight"].to_numpy(), w1.to_numpy())
    assert np.allclose(loaded_state["layer2.weight"].to_numpy(), w2.to_numpy())

    # Metadata
    meta = ox.load_file_metadata(gcs_path)
    assert meta.get("cloud") == "gcp-vertex-ai"

def test_sagemaker_context_and_callback(tmp_path, monkeypatch):
    sm_model_dir = str(tmp_path / "sagemaker_model")
    sm_ckpt_dir = str(tmp_path / "sagemaker_ckpts")

    monkeypatch.setenv("SM_MODEL_DIR", sm_model_dir)
    monkeypatch.setenv("SM_CHECKPOINT_DIR", sm_ckpt_dir)
    monkeypatch.setenv("SM_CURRENT_HOST", "algo-1")
    monkeypatch.setenv("SM_NUM_GPUS", "4")

    assert aws.is_sagemaker() is True
    ctx = aws.get_sagemaker_context()
    assert ctx.model_dir == sm_model_dir
    assert ctx.num_gpus == 4
    assert ctx.current_host == "algo-1"

    model = ox.nn.Sequential(ox.nn.Linear(2, 2))
    cb = aws.SageMakerCallback(save_checkpoints=True)

    # Epoch end callback
    cb.on_epoch_end(epoch=1, metrics={"loss": 0.45, "acc": 0.92}, model=model)
    assert len(cb.logged_metrics) == 1
    assert (Path(sm_ckpt_dir) / "checkpoint_epoch_1.safetensors").exists()

    # Training end callback
    cb.on_train_end(model=model)
    assert (Path(sm_model_dir) / "model.safetensors").exists()

def test_vertex_ai_context_and_callback(tmp_path, monkeypatch):
    aip_model_dir = str(tmp_path / "vertex_model")
    aip_ckpt_dir = str(tmp_path / "vertex_ckpts")

    monkeypatch.setenv("AIP_MODEL_DIR", aip_model_dir)
    monkeypatch.setenv("AIP_CHECKPOINT_DIR", aip_ckpt_dir)
    monkeypatch.setenv("CLOUD_ML_PROJECT_ID", "my-gcp-ai-project")
    monkeypatch.setenv("CLOUD_ML_REGION", "us-central1")

    assert gcp.is_vertex_ai() is True
    ctx = gcp.get_vertex_ai_context()
    assert ctx.model_dir == aip_model_dir
    assert ctx.project_id == "my-gcp-ai-project"
    assert ctx.region == "us-central1"

    model = ox.nn.Sequential(ox.nn.Linear(2, 2))
    cb = gcp.VertexAICallback(save_checkpoints=True)

    # Epoch end callback
    cb.on_epoch_end(epoch=2, metrics={"val_loss": 0.31, "val_acc": 0.95}, model=model)
    assert len(cb.logged_metrics) == 1
    assert (Path(aip_ckpt_dir) / "checkpoint_epoch_2.safetensors").exists()

    # Training end callback
    cb.on_train_end(model=model)
    assert (Path(aip_model_dir) / "model.safetensors").exists()

def test_cloud_environment_provider_detection(monkeypatch):
    # Base local state
    monkeypatch.delenv("SM_MODEL_DIR", raising=False)
    monkeypatch.delenv("SM_CHANNEL_TRAINING", raising=False)
    monkeypatch.delenv("SM_CURRENT_HOST", raising=False)
    monkeypatch.delenv("SAGEMAKER_JOB_NAME", raising=False)
    monkeypatch.delenv("AIP_MODEL_DIR", raising=False)
    monkeypatch.delenv("AIP_CHECKPOINT_DIR", raising=False)
    monkeypatch.delenv("CLOUD_ML_PROJECT_ID", raising=False)
    monkeypatch.delenv("VERTEX_AI_ENV", raising=False)
    monkeypatch.delenv("TPU_NAME", raising=False)

    assert ox.cloud.is_cloud_environment() is False
    assert ox.cloud.get_cloud_provider() is None

    # Simulate AWS SageMaker
    monkeypatch.setenv("SM_MODEL_DIR", "/opt/ml/model")
    assert ox.cloud.is_cloud_environment() is True
    assert ox.cloud.get_cloud_provider() == "aws"

    # Simulate GCP Vertex AI
    monkeypatch.delenv("SM_MODEL_DIR")
    monkeypatch.setenv("AIP_MODEL_DIR", "/gcs/model")
    assert ox.cloud.is_cloud_environment() is True
    assert ox.cloud.get_cloud_provider() == "gcp"

    # Simulate TPU VM
    monkeypatch.delenv("AIP_MODEL_DIR")
    monkeypatch.setenv("TPU_NAME", "v4-8-pod-node")
    assert ox.cloud.is_tpu_available() is True
    assert ox.cloud.get_cloud_provider() == "gcp"
