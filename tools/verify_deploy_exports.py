"""Loads the files written by tests/deploy_files.rs with LiteRT (TFLite), coremltools and OpenVINO and compares the
results with NumPy.

    OXIDEON_DEPLOY_DIR=out cargo test -p oxideon-core --test deploy_files -- --ignored
    pip install ai-edge-litert coremltools openvino numpy
    python tools/verify_deploy_exports.py out
"""
import math, sys, os
import numpy as np

d = sys.argv[1]
w1 = np.array([math.sin(i * 0.37) for i in range(12)], dtype=np.float32).reshape(4, 3)
w2 = np.array([math.cos(i * 0.91) for i in range(8)], dtype=np.float32).reshape(2, 4)
b1 = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float32)
b2 = np.array([0.5, -0.5], dtype=np.float32)
gelu = lambda x: 0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))
softmax = lambda x: np.exp(x - x.max(-1, keepdims=True)) / np.exp(x - x.max(-1, keepdims=True)).sum(-1, keepdims=True)
ref = lambda x, act: softmax(act(x @ w1.T + b1) @ w2.T + b2)
relu = lambda x: np.maximum(x, 0)
rng = np.random.default_rng(0)
ok = True

def check(name, got, want, tol=1e-4):
    global ok
    err = float(np.abs(np.asarray(got, dtype=np.float64) - want).max())
    print(f"{name:10s} max error {err:.2e}", "OK" if err < tol else "FAIL")
    ok &= err < tol

from ai_edge_litert.interpreter import Interpreter
for f, act, shape in (("relu.tflite", relu, (1, 3)), ("gelu.tflite", gelu, (2, 3))):
    it = Interpreter(model_path=os.path.join(d, f)); it.allocate_tensors()
    x = rng.standard_normal(shape).astype(np.float32)
    it.set_tensor(it.get_input_details()[0]["index"], x); it.invoke()
    check(f, it.get_tensor(it.get_output_details()[0]["index"]), ref(x, act))

import openvino as ov
core = ov.Core()
m = core.read_model(os.path.join(d, "model.xml"))
x = rng.standard_normal((1, 3)).astype(np.float32)
check("openvino", core.compile_model(m, "CPU")(x)[0], ref(x, relu))

import onnx, onnxruntime as ort
for f, act, shape in (("mlp_relu.onnx", relu, (1, 3)), ("mlp_gelu.onnx", gelu, (2, 3))):
    onnx.checker.check_model(onnx.load(os.path.join(d, f)))
    sess = ort.InferenceSession(os.path.join(d, f), providers=["CPUExecutionProvider"])
    x = rng.standard_normal(shape).astype(np.float32)
    check(f, sess.run(None, {"input": x})[0], ref(x, act))

import coremltools as ct
spec = ct.models.MLModel(os.path.join(d, "model.mlmodel"), skip_model_load=True).get_spec()
kinds = [l.WhichOneof("layer") for l in spec.neuralNetwork.layers]
print("coreml layers:", kinds)
ok &= kinds == ["innerProduct", "activation", "innerProduct", "softmax"]
# Core ML predictions need macOS: compare the parsed weights and shapes with the NumPy reference instead
L = spec.neuralNetwork.layers
ip1, ip2 = L[0].innerProduct, L[2].innerProduct
ok &= (ip1.inputChannels, ip1.outputChannels, ip2.inputChannels, ip2.outputChannels) == (3, 4, 4, 2) and ip1.hasBias and ip2.hasBias
ok &= np.allclose(np.array(ip1.weights.floatValue), w1.reshape(-1)) and np.allclose(np.array(ip2.weights.floatValue), w2.reshape(-1))
ok &= np.allclose(np.array(ip1.bias.floatValue), b1) and np.allclose(np.array(ip2.bias.floatValue), b2)
ok &= L[1].activation.WhichOneof("NonlinearityType") == "ReLU"
ok &= [list(f.type.multiArrayType.shape) for f in list(spec.description.input) + list(spec.description.output)] == [[3], [2]]
print("coreml spec matches the reference:", ok)
sys.exit(0 if ok else 1)
