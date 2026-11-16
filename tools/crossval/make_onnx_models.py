"""Builds ONNX models with onnx.helper (random weights) and records onnxruntime's outputs in target/onnx_models for batch 49.
Run with the venv that has onnx + onnxruntime: target/tokvenv/Scripts/python.exe tools/crossval/make_onnx_models.py"""
import os
import numpy as np, onnx, onnxruntime as ort
from onnx import helper as h, numpy_helper as nh, TensorProto as TP

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "target", "onnx_models"); os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(7)
def f(*s, scale=0.3): return (rng.standard_normal(s) * scale).astype(np.float32)
def init(name, arr): return nh.from_array(arr, name)
def vi(name, dt, shape): return h.make_tensor_value_info(name, dt, shape)

def finish(name, nodes, inits, inputs, outputs, feeds):
    g = h.make_graph(nodes, name, inputs, outputs, inits)
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 17)]); m.ir_version = 8
    onnx.checker.check_model(m)
    path = os.path.join(out, name + ".onnx"); onnx.save(m, path)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    res = sess.run(None, feeds)
    for i, (k, v) in enumerate(feeds.items()): np.save(os.path.join(out, f"{name}_in{i}.npy"), v)
    for i, v in enumerate(res): np.save(os.path.join(out, f"{name}_out{i}.npy"), v)
    print(name, [r.shape for r in res])

# 1. MLP
x = f(4, 16, scale=1.0)
finish("mlp", [h.make_node("Gemm", ["x", "w1", "b1"], ["a"], transB=1, alpha=0.5, beta=2.0), h.make_node("Relu", ["a"], ["r"]),
               h.make_node("Gemm", ["r", "w2", "b2"], ["l"], transB=1), h.make_node("Softmax", ["l"], ["y"], axis=-1)],
       [init("w1", f(32, 16)), init("b1", f(32)), init("w2", f(5, 32)), init("b2", f(5))], [vi("x", TP.FLOAT, [4, 16])], [vi("y", TP.FLOAT, [4, 5])], {"x": x})

# 2. CNN
x = f(2, 4, 12, 12, scale=1.0)
finish("cnn", [h.make_node("Conv", ["x", "w1", "b1"], ["c1"], pads=[1, 1, 1, 1], strides=[1, 1], kernel_shape=[3, 3]),
               h.make_node("BatchNormalization", ["c1", "g", "be", "mu", "var"], ["bn"], epsilon=1e-3), h.make_node("Relu", ["bn"], ["r"]),
               h.make_node("MaxPool", ["r"], ["p"], kernel_shape=[2, 2], strides=[2, 2]),
               h.make_node("Conv", ["p", "w2"], ["c2"], group=2, dilations=[2, 2], pads=[2, 2, 2, 2], strides=[1, 1]),
               h.make_node("AveragePool", ["c2"], ["ap"], kernel_shape=[2, 2], strides=[2, 2]),
               h.make_node("GlobalAveragePool", ["ap"], ["gp"]), h.make_node("Flatten", ["gp"], ["fl"], axis=1), h.make_node("Gemm", ["fl", "wf", "bf"], ["y"], transB=1)],
       [init("w1", f(8, 4, 3, 3)), init("b1", f(8)), init("g", np.abs(f(8)) + 0.5), init("be", f(8)), init("mu", f(8)), init("var", np.abs(f(8)) + 0.5), init("w2", f(6, 4, 3, 3)), init("wf", f(3, 6)), init("bf", f(3))],
       [vi("x", TP.FLOAT, [2, 4, 12, 12])], [vi("y", TP.FLOAT, [2, 3])], {"x": x})

# 3. transformer block (2 heads) with LayerNormalization, GELU via erf, residuals
B, T, D, H = 2, 5, 8, 2
x = f(B, T, D, scale=1.0)
nodes = [h.make_node("LayerNormalization", ["x", "ln_g", "ln_b"], ["ln"], axis=-1, epsilon=1e-5)]
for nm in "qkv":
    nodes += [h.make_node("MatMul", ["ln", "w" + nm], [nm + "p"]), h.make_node("Add", [nm + "p", "b" + nm], [nm + "b"]),
              h.make_node("Reshape", [nm + "b", "split"], [nm + "r"]), h.make_node("Transpose", [nm + "r"], [nm + "t"], perm=[0, 2, 1, 3])]
nodes += [h.make_node("Transpose", ["kt"], ["ktt"], perm=[0, 1, 3, 2]), h.make_node("MatMul", ["qt", "ktt"], ["sc"]), h.make_node("Div", ["sc", "scale"], ["scs"]),
          h.make_node("Softmax", ["scs"], ["att"], axis=-1), h.make_node("MatMul", ["att", "vt"], ["ctx"]), h.make_node("Transpose", ["ctx"], ["ctxt"], perm=[0, 2, 1, 3]),
          h.make_node("Reshape", ["ctxt", "merge"], ["ctxm"]), h.make_node("MatMul", ["ctxm", "wo"], ["o"]), h.make_node("Add", ["x", "o"], ["res"]),
          h.make_node("MatMul", ["res", "w1"], ["h1"]), h.make_node("Div", ["h1", "sqrt2"], ["hs"]), h.make_node("Erf", ["hs"], ["er"]), h.make_node("Add", ["er", "one"], ["e1"]),
          h.make_node("Mul", ["h1", "e1"], ["hm"]), h.make_node("Mul", ["hm", "half"], ["gelu"]), h.make_node("MatMul", ["gelu", "w2"], ["f"]), h.make_node("Add", ["res", "f"], ["y"]),
          h.make_node("ReduceMean", ["y"], ["pool"], axes=[1], keepdims=0)]
inits = [init("ln_g", np.abs(f(D)) + 0.5), init("ln_b", f(D)), init("split", np.array([B, T, H, D // H], dtype=np.int64)), init("merge", np.array([B, T, D], dtype=np.int64)),
         init("scale", np.array(np.sqrt(D // H), dtype=np.float32)), init("sqrt2", np.array(np.sqrt(2), dtype=np.float32)), init("one", np.array(1, dtype=np.float32)), init("half", np.array(0.5, dtype=np.float32)),
         init("wo", f(D, D)), init("w1", f(D, 16)), init("w2", f(16, D))] + [init("w" + nm, f(D, D)) for nm in "qkv"] + [init("b" + nm, f(D)) for nm in "qkv"]
finish("transformer", nodes, inits, [vi("x", TP.FLOAT, [B, T, D])], [vi("y", TP.FLOAT, [B, T, D]), vi("pool", TP.FLOAT, [B, D])], {"x": x})

# 4. embedding lookup and reductions
ids = rng.integers(0, 50, size=(2, 6)).astype(np.int64)
finish("embed", [h.make_node("Gather", ["table", "ids"], ["e"], axis=0), h.make_node("Add", ["e", "pos"], ["ep"]), h.make_node("LeakyRelu", ["ep"], ["lr"], alpha=0.2),
                 h.make_node("ReduceMean", ["lr"], ["y"], axes=[1], keepdims=0), h.make_node("ReduceMax", ["lr"], ["mx"], axes=[1, 2], keepdims=1), h.make_node("ReduceSum", ["lr", "ax"], ["sm"], keepdims=0)],
       [init("table", f(50, 8, scale=1.0)), init("pos", f(6, 8)), init("ax", np.array([-1], dtype=np.int64))], [vi("ids", TP.INT64, [2, 6])],
       [vi("y", TP.FLOAT, [2, 8]), vi("mx", TP.FLOAT, [2, 1, 1]), vi("sm", TP.FLOAT, [2, 6])], {"ids": ids})

# 5. shape manipulation and elementwise odds and ends
x = f(3, 4, 5, scale=1.0); cond = (rng.random((3, 4, 5)) > 0.5)
finish("misc", [h.make_node("Slice", ["x", "st", "en", "axs", "stp"], ["sl"]), h.make_node("Unsqueeze", ["sl", "uax"], ["un"]), h.make_node("Squeeze", ["un", "uax"], ["sq"]),
                h.make_node("Concat", ["sq", "sq"], ["cc"], axis=1), h.make_node("Cast", ["cc"], ["ci"], to=TP.INT64), h.make_node("Cast", ["ci"], ["cf"], to=TP.FLOAT),
                h.make_node("Clip", ["x", "lo", "hi"], ["cl"]), h.make_node("Where", ["cond", "x", "cl"], ["wh"]), h.make_node("Expand", ["col", "eshape"], ["ex"]), h.make_node("Add", ["wh", "ex"], ["we"]),
                h.make_node("Abs", ["x"], ["ab"]), h.make_node("Add", ["ab", "eps"], ["ab1"]), h.make_node("Log", ["ab1"], ["lg"]), h.make_node("Sqrt", ["ab1"], ["sr"]), h.make_node("Pow", ["ab1", "two"], ["pw"]),
                h.make_node("Reciprocal", ["ab1"], ["rc"]), h.make_node("Neg", ["rc"], ["ng"]), h.make_node("Tanh", ["ng"], ["th"]), h.make_node("Sigmoid", ["lg"], ["sg"]), h.make_node("Exp", ["th"], ["ep"]),
                h.make_node("Shape", ["x"], ["shp"]), h.make_node("Sub", ["ep", "sg"], ["d1"]), h.make_node("Mul", ["d1", "pw"], ["d2"]), h.make_node("Add", ["d2", "sr"], ["y"])],
       [init("st", np.array([1, 0], dtype=np.int64)), init("en", np.array([3, 4], dtype=np.int64)), init("axs", np.array([0, 2], dtype=np.int64)), init("stp", np.array([1, 2], dtype=np.int64)),
        init("uax", np.array([1], dtype=np.int64)), init("lo", np.array(-0.5, dtype=np.float32)), init("hi", np.array(0.7, dtype=np.float32)), init("col", f(3, 1, 1)),
        init("eshape", np.array([3, 4, 5], dtype=np.int64)), init("eps", np.array(1.0, dtype=np.float32)), init("two", np.array(2.0, dtype=np.float32))],
       [vi("x", TP.FLOAT, [3, 4, 5]), vi("cond", TP.BOOL, [3, 4, 5])],
       [vi("y", TP.FLOAT, [3, 4, 5]), vi("cf", TP.FLOAT, [2, 8, 2]), vi("we", TP.FLOAT, [3, 4, 5]), vi("shp", TP.INT64, [3])], {"x": x, "cond": cond})
