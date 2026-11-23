// Loads the wasm-bindgen build of the `wasm` feature in Node and checks the JavaScript API against plain JS math.
//   cargo build --target wasm32-unknown-unknown --no-default-features --features wasm --release
//   wasm-bindgen target/wasm32-unknown-unknown/release/oxideon.wasm --target nodejs --out-dir out
//   node tools/verify_wasm_bindings.js out
const path = require("path");
const { WasmTensor, WasmInferenceSession } = require(path.resolve(process.argv[2], "oxideon.js"));
let ok = true;
const check = (name, cond) => { console.log(`${cond ? "OK  " : "FAIL"} ${name}`); ok = ok && !!cond; };
const near = (a, b, tol = 1e-5) => a.length === b.length && a.every((v, i) => Math.abs(v - b[i]) <= tol * (1 + Math.abs(b[i])));

const a = WasmTensor.from_f32_array(new Float32Array([1, 2, 3, 4, 5, 6]), new Uint32Array([2, 3]));
const b = WasmTensor.from_f32_array(new Float32Array([1, 0, 0, 1, 1, 1]), new Uint32Array([3, 2]));
check("shape()", near(Array.from(a.shape()), [2, 3]) && a.numel() === 6);
check("to_f32_vec returns a Float32Array", a.to_f32_vec() instanceof Float32Array);
check("matmul", near(Array.from(a.matmul(b).to_f32_vec()), [4, 5, 10, 11]));
check("add/sub/mul/div", near(Array.from(a.add(a).to_f32_vec()), [2, 4, 6, 8, 10, 12])
  && near(Array.from(a.sub(a).to_f32_vec()), [0, 0, 0, 0, 0, 0])
  && near(Array.from(a.mul(a).to_f32_vec()), [1, 4, 9, 16, 25, 36])
  && near(Array.from(a.div(a).to_f32_vec()), [1, 1, 1, 1, 1, 1]));
const x = WasmTensor.from_f32_array(new Float32Array([-1, 0, 1, 2]), new Uint32Array([4]));
check("relu", near(Array.from(x.relu().to_f32_vec()), [0, 0, 1, 2]));
check("exp / tanh", near(Array.from(x.exp().to_f32_vec()), [-1, 0, 1, 2].map(Math.exp)) && near(Array.from(x.tanh().to_f32_vec()), [-1, 0, 1, 2].map(Math.tanh)));
const erf = (z) => { const t = 1 / (1 + 0.3275911 * Math.abs(z)); const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-z * z); return z >= 0 ? y : -y; };
check("gelu", near(Array.from(x.gelu().to_f32_vec()), [-1, 0, 1, 2].map((v) => 0.5 * v * (1 + erf(v / Math.SQRT2))), 1e-3));
check("silu", near(Array.from(x.silu().to_f32_vec()), [-1, 0, 1, 2].map((v) => v / (1 + Math.exp(-v)))));
const sm = a.softmax(-1).to_f32_vec();
check("softmax rows sum to 1", Math.abs(sm[0] + sm[1] + sm[2] - 1) < 1e-5 && Math.abs(sm[3] + sm[4] + sm[5] - 1) < 1e-5);
check("reshape / transpose", near(Array.from(a.reshape(new Uint32Array([3, 2])).shape()), [3, 2]) && near(Array.from(a.transpose(0, 1).to_f32_vec()), [1, 4, 2, 5, 3, 6]));
let threw = false;
try { a.matmul(a); } catch (e) { threw = e instanceof Error && /shape|matmul/i.test(e.message); }
check("errors become JS exceptions with a message", threw);
const s = new WasmInferenceSession();
check("WasmInferenceSession", typeof s.has_webgpu() === "boolean");
[a, b, x, s].forEach((o) => o.free());
process.exit(ok ? 0 : 1);
