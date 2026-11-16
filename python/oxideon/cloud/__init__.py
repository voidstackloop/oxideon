"""
Oxideon Cloud Integration Package for AWS and GCP.
Provides seamless cloud storage persistence (S3, GCS) and environment orchestration (SageMaker, Vertex AI).
"""
from typing import Optional
import os

from . import aws as aws
from . import gcp as gcp

from .aws import (
    S3Client,
    s3_upload,
    s3_download,
    s3_exists,
    is_sagemaker,
    get_sagemaker_context,
    SageMakerCallback,
    SageMakerContext,
)

from .gcp import (
    GCSClient,
    gcs_upload,
    gcs_download,
    gcs_exists,
    is_vertex_ai,
    get_vertex_ai_context,
    VertexAICallback,
    VertexAIContext,
    is_tpu_available,
)

def is_cloud_environment() -> bool:
    """Returns True if running in an AWS (SageMaker/EC2) or GCP (Vertex AI/TPU VM) environment."""
    return is_sagemaker() or is_vertex_ai()

def get_cloud_provider() -> Optional[str]:
    """Returns 'aws' if running on AWS, 'gcp' if running on GCP, or None."""
    if is_sagemaker():
        return "aws"
    if is_vertex_ai() or is_tpu_available():
        return "gcp"
    return None

__all__ = [
    "aws",
    "gcp",
    "S3Client",
    "s3_upload",
    "s3_download",
    "s3_exists",
    "is_sagemaker",
    "get_sagemaker_context",
    "SageMakerCallback",
    "SageMakerContext",
    "GCSClient",
    "gcs_upload",
    "gcs_download",
    "gcs_exists",
    "is_vertex_ai",
    "get_vertex_ai_context",
    "VertexAICallback",
    "VertexAIContext",
    "is_tpu_available",
    "is_cloud_environment",
    "get_cloud_provider",
]
