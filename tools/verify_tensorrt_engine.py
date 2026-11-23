"""Builds a TensorRT engine from the ONNX files written by tests/deploy_files.rs and runs it on the GPU against NumPy.

    OXIDEON_DEPLOY_DIR=out cargo test -p oxideon-core --test deploy_files -- --ignored
    pip install tensorrt-cu12 cuda-python numpy        # needs an NVIDIA GPU and driver
    python tools/verify_tensorrt_engine.py out
"""
import math
import os
import sys

import numpy as np
import tensorrt as trt

try:
    from cuda.bindings import runtime as cudart
except ImportError:  # older cuda-python
    from cuda import cudart

d = sys.argv[1]
w1 = np.array([math.sin(i * 0.37) for i in range(12)], dtype=np.float32).reshape(4, 3)
w2 = np.array([math.cos(i * 0.91) for i in range(8)], dtype=np.float32).reshape(2, 4)
b1 = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float32)
b2 = np.array([0.5, -0.5], dtype=np.float32)
erf = np.vectorize(math.erf)
gelu = lambda x: 0.5 * x * (1 + erf(x / math.sqrt(2)))
softmax = lambda x: np.exp(x - x.max(-1, keepdims=True)) / np.exp(x - x.max(-1, keepdims=True)).sum(-1, keepdims=True)
relu = lambda x: np.maximum(x, 0)
ref = lambda x, act: softmax(act(x @ w1.T + b1) @ w2.T + b2)


def check_cuda(result):
    err = result[0]
    if int(err) != 0:
        raise RuntimeError(f"CUDA error {err}")
    return result[1] if len(result) > 1 else None


logger = trt.Logger(trt.Logger.WARNING)
ok = True
for name, act, shape in (("mlp_relu", relu, (1, 3)), ("mlp_gelu", gelu, (2, 3))):
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(open(os.path.join(d, f"{name}.onnx"), "rb").read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        sys.exit(f"TensorRT could not parse {name}.onnx")
    config = builder.create_builder_config()
    plan = builder.build_serialized_network(network, config)
    assert plan is not None, f"engine build failed for {name}"
    open(os.path.join(d, f"{name}.plan"), "wb").write(bytes(plan))
    engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
    ctx = engine.create_execution_context()

    x = np.random.default_rng(1).standard_normal(shape).astype(np.float32)
    out = np.zeros_like(ref(x, act), dtype=np.float32)
    d_in = check_cuda(cudart.cudaMalloc(x.nbytes))
    d_out = check_cuda(cudart.cudaMalloc(out.nbytes))
    check_cuda(cudart.cudaMemcpy(d_in, x.ctypes.data, x.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    inp = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    outp = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
    ctx.set_tensor_address(inp, int(d_in))
    ctx.set_tensor_address(outp, int(d_out))
    stream = check_cuda(cudart.cudaStreamCreate())
    assert ctx.execute_async_v3(int(stream)), "execution failed"
    check_cuda(cudart.cudaStreamSynchronize(stream))
    check_cuda(cudart.cudaMemcpy(out.ctypes.data, d_out, out.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
    err = float(np.abs(out - ref(x, act)).max())
    print(f"{name}: TensorRT engine ({len(bytes(plan))} bytes) max error vs NumPy {err:.2e}", "OK" if err < 1e-4 else "FAIL")
    ok &= err < 1e-4
sys.exit(0 if ok else 1)
