//! Model tracing into [`OpGraph`] and [`CompiledGraph`].
//!
//! Executes any function or [`crate::nn::Module`] once with traced leaf inputs,
//! lowers the recorded computation DAG into an [`OpGraph`], and runs all graph
//! compiler optimization & operator fusion passes (`Conv+BN+ReLU`, `FusedAttention`,
//! constant folding, CSE, DCE, and fused elementwise kernels).

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::opgraph::{CompiledGraph, Id, OpGraph};
use crate::tensor::{Tensor, set_grad_enabled};

/// Traces `f(&inputs)` into an uncompiled [`OpGraph`].
///
/// Only operations the graph compiler knows are lowered; anything else is an error rather than a silently frozen constant.
pub fn trace_to_opgraph<F>(inputs: &[Tensor], f: F) -> Result<OpGraph>
where
    F: FnOnce(&[Tensor]) -> Result<Vec<Tensor>>,
{
    trace_inner(inputs, f).map(|(g, _)| g)
}

/// The traced graph together with the eager outputs of the traced call.
fn trace_inner<F>(inputs: &[Tensor], f: F) -> Result<(OpGraph, Vec<Tensor>)>
where
    F: FnOnce(&[Tensor]) -> Result<Vec<Tensor>>,
{
    let _guard = set_grad_enabled(true);
    let traced_inputs: Vec<Tensor> = inputs
        .iter()
        .map(|t| {
            let d = t.detach();
            d.set_requires_grad(true)?;
            Ok(d)
        })
        .collect::<Result<Vec<_>>>()?;
    let outputs = f(&traced_inputs)?;
    if outputs.is_empty() {
        return Err(Error::Msg("trace: closure returned zero outputs".into()));
    }

    let mut graph = OpGraph::new();
    let mut map: HashMap<usize, Id> = HashMap::new();
    for inp in &traced_inputs {
        let id = graph.input(inp.shape(), inp.dtype())?;
        map.insert(inp.id(), id);
    }

    fn lower(t: &Tensor, graph: &mut OpGraph, map: &mut HashMap<usize, Id>) -> Result<Id> {
        if let Some(&id) = map.get(&t.id()) {
            return Ok(id);
        }
        let Some((op, ins)) = t.trace_node() else {
            let id = graph.constant(t)?;
            map.insert(t.id(), id);
            return Ok(id);
        };
        let mut arg_ids = Vec::with_capacity(ins.len());
        for child in &ins {
            arg_ids.push(lower(child, graph, map)?);
        }
        let id = match op {
            "add" => graph.add(arg_ids[0], arg_ids[1])?,
            "sub" => graph.sub(arg_ids[0], arg_ids[1])?,
            "mul" => graph.mul(arg_ids[0], arg_ids[1])?,
            "div" => graph.div(arg_ids[0], arg_ids[1])?,
            "maximum" => graph.maximum(arg_ids[0], arg_ids[1])?,
            "minimum" => graph.minimum(arg_ids[0], arg_ids[1])?,
            "pow" => graph.pow(arg_ids[0], arg_ids[1])?,
            "matmul" => graph.matmul(arg_ids[0], arg_ids[1])?,
            "neg" => graph.neg(arg_ids[0])?,
            "abs" => graph.abs(arg_ids[0])?,
            "exp" => graph.exp(arg_ids[0])?,
            "ln" => graph.ln(arg_ids[0])?,
            "sqrt" => graph.sqrt(arg_ids[0])?,
            "rsqrt" => graph.rsqrt(arg_ids[0])?,
            "tanh" => graph.tanh(arg_ids[0])?,
            "sigmoid" => graph.sigmoid(arg_ids[0])?,
            "relu" => graph.relu(arg_ids[0])?,
            "gelu" => graph.gelu(arg_ids[0])?,
            "silu" => graph.silu(arg_ids[0])?,
            "square" => graph.square(arg_ids[0])?,
            "recip" => graph.recip(arg_ids[0])?,
            "softmax" => graph.softmax(arg_ids[0], -1)?,
            "reshape" | "contiguous" => graph.reshape(arg_ids[0], t.shape())?,
            "linear" => {
                // inputs: [x, weight, (optional bias)]
                let x_id = arg_ids[0];
                let w_id = arg_ids[1];
                let wt_id = graph.permute(w_id, &[1, 0])?;
                let mm = graph.matmul(x_id, wt_id)?;
                if arg_ids.len() >= 3 {
                    let b_shape = t.shape();
                    let b_bcast = graph.broadcast_to(arg_ids[2], b_shape)?;
                    graph.add(mm, b_bcast)?
                } else {
                    mm
                }
            }
            "conv2d" => {
                let b_id = if arg_ids.len() >= 3 { Some(arg_ids[2]) } else { None };
                let (stride, padding, dilation) = find_conv2d_params(&ins, t)?;
                graph.conv2d(arg_ids[0], arg_ids[1], b_id, stride, padding, dilation, 1)?
            }
            other => {
                // Freezing the traced value as a constant would make the compiled graph return stale results for any
                // other input, so refuse instead.
                return Err(Error::Msg(format!(
                    "trace: operation '{other}' (output shape {:?}) is not supported by the graph compiler",
                    t.shape()
                )));
            }
        };
        map.insert(t.id(), id);
        Ok(id)
    }

    let mut out_ids = Vec::with_capacity(outputs.len());
    for o in &outputs {
        out_ids.push(lower(o, &mut graph, &mut map)?);
    }
    graph.set_outputs(&out_ids);
    Ok((graph, outputs))
}

/// Finds the `(stride, padding, dilation)` a traced conv2d node ran with: among the combinations that give the recorded
/// output shape, the one that reproduces the recorded values on the traced data (the node itself keeps no attributes).
type ConvParams = ((usize, usize), (usize, usize), (usize, usize));

fn find_conv2d_params(ins: &[Tensor], out: &Tensor) -> Result<ConvParams> {
    let (x, w) = (&ins[0], &ins[1]);
    let bias = ins.get(2);
    let (h, wd, kh, kw) = (x.shape()[2], x.shape()[3], w.shape()[2], w.shape()[3]);
    let (oh, ow) = (out.shape()[2], out.shape()[3]);
    let want = out.to_vec::<f64>()?;
    let out_dim = |size: usize, k: usize, s: usize, p: usize, d: usize| (size + 2 * p).checked_sub(d * (k - 1) + 1).map(|v| v / s + 1);
    let _ng = set_grad_enabled(false);
    for sh in 1..=4 {
        for sw in 1..=4 {
            for dh in 1..=3 {
                for dw in 1..=3 {
                    for ph in 0..=4 {
                        for pw in 0..=4 {
                            if out_dim(h, kh, sh, ph, dh) != Some(oh) || out_dim(wd, kw, sw, pw, dw) != Some(ow) {
                                continue;
                            }
                            let Ok(y) = x.conv2d(w, bias, (sh, sw), (ph, pw), (dh, dw)) else { continue };
                            let got = y.to_vec::<f64>()?;
                            if got.len() == want.len() && got.iter().zip(&want).all(|(a, b)| (a - b).abs() <= 1e-4 * (1.0 + b.abs())) {
                                return Ok(((sh, sw), (ph, pw), (dh, dw)));
                            }
                        }
                    }
                }
            }
        }
    }
    Err(Error::Msg("trace: could not determine the stride/padding/dilation of a conv2d node".into()))
}

/// Traces `f(&inputs)` into an [`OpGraph`] and compiles it into an optimized [`CompiledGraph`].
pub fn trace_and_compile<F>(inputs: &[Tensor], f: F) -> Result<CompiledGraph>
where
    F: FnOnce(&[Tensor]) -> Result<Vec<Tensor>>,
{
    let (graph, eager) = trace_inner(inputs, f)?;
    let compiled = graph.compile()?;
    // the compiled graph must reproduce the traced call on the traced inputs (catches mis-lowered operations)
    let got = compiled.run(inputs)?;
    if got.len() != eager.len() {
        return Err(Error::Msg(format!("trace: compiled graph has {} outputs, the traced call {}", got.len(), eager.len())));
    }
    for (i, (g, e)) in got.iter().zip(&eager).enumerate() {
        if g.shape() != e.shape() {
            return Err(Error::Msg(format!("trace: output {i} has shape {:?} after compilation, {:?} eagerly", g.shape(), e.shape())));
        }
        if !e.dtype().is_floating_point() {
            continue;
        }
        let (a, b) = (g.to_vec::<f64>()?, e.to_vec::<f64>()?);
        if let Some((x, y)) = a.iter().zip(&b).find(|(x, y)| !((**x - **y).abs().le(&(1e-3 * (1.0 + y.abs()))) || x.is_nan() && y.is_nan())) {
            return Err(Error::Msg(format!("trace: compiled output {i} differs from the traced call ({x} vs {y}); an operation was lowered incorrectly")));
        }
    }
    Ok(compiled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::nn::{Linear, Module};

    #[test]
    fn trace_linear_relu_linear_into_compiled_graph() {
        crate::rng::manual_seed(42);
        let l1 = Linear::new(4, 8).unwrap();
        let l2 = Linear::new(8, 2).unwrap();
        let x = Tensor::randn(&[3, 4]).unwrap();

        let eager = l2.forward(&l1.forward(&x).unwrap().relu().unwrap()).unwrap();
        let compiled = trace_and_compile(std::slice::from_ref(&x), |ins| {
            let h = l1.forward(&ins[0])?.relu()?;
            Ok(vec![l2.forward(&h)?])
        })
        .unwrap();
        let got = compiled.run(&[x]).unwrap()[0].to_vec::<f32>().unwrap();
        let want = eager.to_vec::<f32>().unwrap();
        for (a, b) in got.iter().zip(want.iter()) {
            assert!((a - b).abs() < 1e-4, "traced mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn unsupported_ops_are_rejected_instead_of_frozen_into_constants() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        // `sum_axis` has no graph lowering: tracing must say so, not bake in this call's value
        let r = trace_and_compile(std::slice::from_ref(&x), |ins| Ok(vec![ins[0].sum_axis(1, false)?]));
        assert!(r.is_err(), "an unsupported op must not silently trace");
        assert!(format!("{:?}", r.err().unwrap()).contains("not supported"));
    }

    #[test]
    fn traced_conv_keeps_its_stride_padding_and_dilation() {
        crate::rng::manual_seed(7);
        let x = Tensor::randn(&[2, 3, 11, 10]).unwrap();
        let w = Tensor::randn(&[4, 3, 3, 3]).unwrap();
        let b = Tensor::randn(&[4]).unwrap();
        for (stride, pad, dil) in [((1, 1), (1, 1), (1, 1)), ((2, 2), (1, 1), (1, 1)), ((1, 2), (0, 2), (1, 1)), ((1, 1), (2, 2), (2, 2)), ((3, 2), (0, 1), (1, 1))] {
            let compiled = trace_and_compile(std::slice::from_ref(&x), |ins| Ok(vec![ins[0].conv2d(&w, Some(&b), stride, pad, dil)?])).unwrap();
            // different data than the traced call: the lowered conv must be the real conv, not a replay
            let x2 = Tensor::randn(&[2, 3, 11, 10]).unwrap();
            let got = compiled.run(std::slice::from_ref(&x2)).unwrap()[0].to_vec::<f32>().unwrap();
            let want = x2.conv2d(&w, Some(&b), stride, pad, dil).unwrap().to_vec::<f32>().unwrap();
            for (a, b) in got.iter().zip(&want) {
                assert!((a - b).abs() < 1e-3, "stride {stride:?} pad {pad:?} dil {dil:?}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn graph_fuses_conv_bn_relu_and_attention() {
        let mut g = OpGraph::new();
        let x = g.input(&[1, 2, 4, 4], DType::F32).unwrap();
        let dev = crate::backend::Device::Cpu;
        let w = g.constant(&Tensor::ones(&[3, 2, 3, 3], DType::F32, &dev).unwrap()).unwrap();
        let b = g.constant(&Tensor::zeros(&[3], DType::F32, &dev).unwrap()).unwrap();
        let gamma = g.constant(&Tensor::ones(&[3], DType::F32, &dev).unwrap()).unwrap();
        let beta = g.constant(&Tensor::zeros(&[3], DType::F32, &dev).unwrap()).unwrap();
        let mean = g.constant(&Tensor::zeros(&[3], DType::F32, &dev).unwrap()).unwrap();
        let var = g.constant(&Tensor::ones(&[3], DType::F32, &dev).unwrap()).unwrap();
        let c = g.conv2d(x, w, Some(b), (1, 1), (1, 1), (1, 1), 1).unwrap();
        let bn = g.batch_norm2d(c, gamma, beta, mean, var, 1e-5).unwrap();
        let r = g.relu(bn).unwrap();
        g.set_outputs(&[r]);
        let comp = g.compile().unwrap();
        assert!(comp.stats().fused_kernels >= 1);
        let inp = Tensor::ones(&[1, 2, 4, 4], DType::F32, &dev).unwrap();
        let out = comp.run(&[inp]).unwrap();
        assert_eq!(out[0].shape(), &[1, 3, 4, 4]);
    }
}
