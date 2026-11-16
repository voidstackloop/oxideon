# Native Cloud Integration & Deployment in Oxideon

Oxideon natively integrates with **Amazon Web Services (AWS)** and **Google Cloud Platform (GCP)** to provide direct streaming I/O to object stores (`s3://` and `gs://`), automatic cloud container detection (SageMaker and Vertex AI), and managed training callbacks for distributed and TPU/GPU pipelines.

---

## Table of Contents
1. [Unified Cloud Object Storage Streaming (`s3://` and `gs://`)](#1-unified-cloud-object-storage-streaming)
2. [Amazon Web Services (AWS) Integration (`ox.cloud.aws`)](#2-amazon-web-services-aws-integration)
3. [Google Cloud Platform (GCP) Integration (`ox.cloud.gcp`)](#3-google-cloud-platform-gcp-integration)
4. [Cloud Environment Telemetry & Detection](#4-cloud-environment-telemetry--detection)
5. [End-to-End Managed Cloud Training Pipeline](#5-end-to-end-managed-cloud-training-pipeline)

---

## 1. Unified Cloud Object Storage Streaming

Oxideon's serialization engine transparently streams tensor dictionaries to and from remote cloud URIs without requiring intermediate local disk management:

```python
import oxideon as ox

tensors = {
    "weight": ox.Tensor.randn([512, 512]),
    "bias": ox.Tensor.zeros([512]),
}

# 1. Direct Save to AWS S3
ox.save_file(tensors, "s3://my-model-bucket/checkpoints/epoch_1.ox")

# 2. Direct Save to Google Cloud Storage (GCS)
ox.save_file(tensors, "gs://my-gcs-bucket/models/production_v1.ox")

# 3. Stream Checkpoints directly from Cloud Object Stores
s3_weights = ox.load_file("s3://my-model-bucket/checkpoints/epoch_1.ox")
gcs_weights = ox.load_file("gs://my-gcs-bucket/models/production_v1.ox")

# 4. Inspect Cloud Metadata without downloading full tensor payloads
metadata = ox.load_file_metadata("s3://my-model-bucket/checkpoints/epoch_1.ox")
print("Saved tensor keys:", list(metadata.keys()))
```

---

## 2. Amazon Web Services (AWS) Integration

The `ox.cloud.aws` submodule provides native tooling for S3 object manipulation and automatic AWS SageMaker training container coordination.

### S3 Direct Client
```python
from oxideon.cloud.aws import S3Client

s3 = S3Client()

# Upload a local file first (uses the local mock store when offline)
with open("results.json", "w") as f:
    f.write('{"acc": 0.99}')
s3.upload_file("results.json", "s3://my-bucket/results/run_1.json")

# Check existence
if s3.exists("s3://my-bucket/results/run_1.json"):
    print("Dataset found!")

# Download the object back to a local file
s3.download_file("s3://my-bucket/results/run_1.json", "results_back.json")
```

### AWS SageMaker Training Support
Oxideon automatically detects when code executes inside an AWS SageMaker training container via environment variables (`SM_MODEL_DIR`, `SM_CHANNEL_TRAINING`, `SM_NUM_GPUS`) and configures output paths accordingly.

```python
from oxideon.cloud.aws import is_sagemaker, get_sagemaker_context, SageMakerCallback

if is_sagemaker():
    ctx = get_sagemaker_context()
    print("SageMaker Job Name:", ctx.job_name)
    print("Model Directory:   ", ctx.model_dir)
    print("Number of GPUs:    ", ctx.num_gpus)

# Attach SageMaker Callback to Trainer for automatic artifact synchronization
callback = SageMakerCallback()  # syncs model.safetensors to the SageMaker model dir
```

---

## 3. Google Cloud Platform (GCP) Integration

The `ox.cloud.gcp` submodule provides Cloud Storage (GCS) clients, Vertex AI custom job detection, Cloud TPU detection, and Vertex AI callbacks.

### GCS Direct Client
```python
from oxideon.cloud.gcp import GCSClient

gcs = GCSClient()

# Write a local checkpoint first, then upload it (local mock store when offline)
import oxideon as ox
ox.save_file({"w": ox.Tensor.randn([4, 4])}, "model.safetensors")
gcs.upload_file("model.safetensors", "gs://my-vertex-bucket/models/model.safetensors")

# Check existence, then download back
if gcs.exists("gs://my-vertex-bucket/models/model.safetensors"):
    print("Embeddings ready.")
gcs.download_file("gs://my-vertex-bucket/models/model.safetensors", "downloaded.safetensors")
```

### Google Cloud Vertex AI & TPU Detection
```python
from oxideon.cloud.gcp import is_vertex_ai, get_vertex_ai_context, is_tpu_available, VertexAICallback

# Detect Vertex AI Custom Job environment
if is_vertex_ai():
    ctx = get_vertex_ai_context()
    print("Vertex AI Project:", ctx.project_id)
    print("Cloud Storage Output:", ctx.gcs_output_dir)

# Check Cloud TPU hardware availability
if is_tpu_available():
    print("Google Cloud TPU VM environment detected!")

# Attach Vertex AI Callback for automatic GCS model checkpointing
callback = VertexAICallback()  # syncs model.safetensors to the Vertex AI output dir
```

---

## 4. Cloud Environment Telemetry & Detection

Write cloud-agnostic code that automatically branches based on the host cloud infrastructure:

```python
import oxideon as ox

# Identify execution environment
if ox.cloud.is_cloud_environment():
    provider = ox.cloud.get_cloud_provider()
    print(f"Running on managed cloud infrastructure: {provider}")
    # Returns "aws" on SageMaker/EC2, "gcp" on Vertex AI/Compute Engine, or "unknown"
else:
    print("Running in local development environment.")
```

---

## 5. End-to-End Managed Cloud Training Pipeline

Integrate cloud storage and automated checkpoint synchronization into Oxideon's `Trainer`:

```python
import numpy as np
import oxideon as ox

# 1. Build Model, Loss & Optimizer
model = ox.Sequential(
    ox.Linear(784, 256),
    ox.ReLU(),
    ox.Linear(256, 10),
)
criterion = ox.cross_entropy
optimizer = ox.Adam(model.parameters(), lr=1e-3)

# 2. Select Cloud Callback based on Active Provider
callbacks = []
if ox.cloud.aws.is_sagemaker():
    callbacks.append(ox.cloud.aws.SageMakerCallback())
elif ox.cloud.gcp.is_vertex_ai():
    callbacks.append(ox.cloud.gcp.VertexAICallback())

# 3. Launch Training
trainer = ox.Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    callbacks=callbacks,
    debug=False,
)

# Trainer automatically logs sample throughput, step metrics, and streams
# epoch checkpoints directly to the configured cloud bucket
train_loader = ox.DataLoader(
    ox.TensorDataset(
        ox.Tensor.randn([64, 784]),
        ox.from_numpy(np.zeros(64, dtype=np.int64)),
    ),
    batch_size=16,
)
trainer.fit(train_loader, epochs=1)
```
