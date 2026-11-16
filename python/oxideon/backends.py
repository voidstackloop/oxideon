"""Backend acceleration engines and driver introspection for Oxideon."""

from typing import List
from . import cuda
from . import mps


class _CudnnBackend:
    enabled: bool = True
    benchmark: bool = False
    deterministic: bool = False

    @staticmethod
    def is_available() -> bool:
        return cuda.is_available()

    @staticmethod
    def version() -> int:
        return 90000


class _MklBackend:
    @staticmethod
    def is_available() -> bool:
        return True


class _OpenmpBackend:
    @staticmethod
    def is_available() -> bool:
        return True


class _QuantizedBackend:
    engine: str = "fbgemm"
    supported_engines = ["fbgemm", "qnnpack", "none"]


class _MlxBackend:
    @staticmethod
    def is_available() -> bool:
        return mps.is_available()


class _VulkanBackend:
    @staticmethod
    def is_available() -> bool:
        return False


class _WebGpuBackend:
    @staticmethod
    def is_available() -> bool:
        return False


cudnn = _CudnnBackend()
mkl = _MklBackend()
openmp = _OpenmpBackend()
quantized = _QuantizedBackend()
mlx = _MlxBackend()
vulkan = _VulkanBackend()
webgpu = _WebGpuBackend()


def get_available_backends() -> List[str]:
    """Returns a list of currently available compute and acceleration backends."""
    available = ["cpu"]
    if openmp.is_available():
        available.append("openmp")
    if mkl.is_available():
        available.append("mkl")
    if cuda.is_available():
        available.append("cuda")
    if cudnn.is_available():
        available.append("cudnn")
    if mps.is_available():
        available.append("mps")
    if mlx.is_available():
        available.append("mlx")
    if vulkan.is_available():
        available.append("vulkan")
    if webgpu.is_available():
        available.append("webgpu")
    return available


__all__ = [
    "cuda",
    "mps",
    "cudnn",
    "mkl",
    "openmp",
    "quantized",
    "mlx",
    "vulkan",
    "webgpu",
    "get_available_backends",
]
