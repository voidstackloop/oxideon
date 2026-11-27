//! Autograd computation graph introspection, FLOP & activation memory estimation,
//! execution profiling, and Graphviz `.dot` export.

use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::time::{Duration, Instant};

use crate::error::Result;
use crate::tensor::Tensor;

/// Static analysis summary of an autograd computation graph.
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub total_nodes: usize,
    pub leaf_nodes: usize,
    pub op_nodes: usize,
    pub op_counts: BTreeMap<&'static str, usize>,
    pub total_activation_bytes: usize,
    pub total_leaf_bytes: usize,
    pub estimated_flops: u64,
}

impl fmt::Display for GraphStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "GraphStats {{ nodes: {} (leaves: {}, ops: {}), activations: {} B, leaves: {} B, est_flops: {} }}",
            self.total_nodes,
            self.leaf_nodes,
            self.op_nodes,
            self.total_activation_bytes,
            self.total_leaf_bytes,
            self.estimated_flops
        )?;
        for (op, count) in &self.op_counts {
            writeln!(f, "  - {op:<20}: {count}")?;
        }
        Ok(())
    }
}

/// Traverses the autograd DAG rooted at `outputs` in topological order and computes
/// node counts, op histogram, memory footprint, and estimated FLOPs.
pub fn analyze_graph(outputs: &[Tensor]) -> GraphStats {
    let topo = collect_topological(outputs);
    let mut leaf_nodes = 0usize;
    let mut op_nodes = 0usize;
    let mut op_counts = BTreeMap::new();
    let mut total_activation_bytes = 0usize;
    let mut total_leaf_bytes = 0usize;
    let mut estimated_flops = 0u64;

    for t in &topo {
        // Saturate: meta tensors can describe models that would not fit.
        let bytes = t.numel().saturating_mul(t.dtype().size_in_bytes());
        if let Some(op) = t.op_name() {
            op_nodes += 1;
            *op_counts.entry(op).or_insert(0) += 1;
            total_activation_bytes = total_activation_bytes.saturating_add(bytes);
            estimated_flops =
                estimated_flops.saturating_add(estimate_node_flops(op, t, &t.op_inputs()));
        } else {
            leaf_nodes += 1;
            total_leaf_bytes = total_leaf_bytes.saturating_add(bytes);
        }
    }

    GraphStats {
        total_nodes: topo.len(),
        leaf_nodes,
        op_nodes,
        op_counts,
        total_activation_bytes,
        total_leaf_bytes,
        estimated_flops,
    }
}

/// Exports the computation graph rooted at `outputs` as a Graphviz `.dot` string.
pub fn export_graphviz_dot(outputs: &[Tensor]) -> String {
    let topo = collect_topological(outputs);
    let mut dot = String::from(
        "digraph OxideonAutograd {\n  rankdir=LR;\n  node [fontname=\" monospace\"];\n",
    );

    for t in &topo {
        let id = t.id();
        let shape_str = format!("{:?}", t.shape());
        let dtype_str = t.dtype().to_string();
        if let Some(op) = t.op_name() {
            dot.push_str(&format!(
                "  t{id} [shape=record, style=\"filled,rounded\", fillcolor=\"#E8F0FE\", label=\"{{ {op} | id={id} | {shape_str} {dtype_str} }}\"];\n"
            ));
            for inp in t.op_inputs() {
                dot.push_str(&format!("  t{} -> t{id};\n", inp.id()));
            }
        } else {
            let color = if t.requires_grad() {
                "#E6F4EA"
            } else {
                "#F1F3F4"
            };
            let tag = if t.requires_grad() {
                "param/leaf"
            } else {
                "const"
            };
            dot.push_str(&format!(
                "  t{id} [shape=box, style=\"filled\", fillcolor=\"{color}\", label=\"{tag} (id={id})\\n{shape_str} {dtype_str}\"];\n"
            ));
        }
    }

    dot.push_str("}\n");
    dot
}

/// Timing and throughput profile of a forward + backward execution pass.
#[derive(Debug, Clone)]
pub struct ExecutionProfile {
    pub forward_duration: Duration,
    pub backward_duration: Duration,
    pub graph_stats: GraphStats,
}

impl ExecutionProfile {
    /// Effective forward throughput in GFLOP/s.
    pub fn forward_gflops(&self) -> f64 {
        let secs = self.forward_duration.as_secs_f64().max(1e-12);
        (self.graph_stats.estimated_flops as f64) / secs / 1e9
    }
}

/// Profiles a forward closure `f` (which must return a scalar loss or tensor to backpropagate)
/// and its `.backward()` pass, returning timing and [`GraphStats`].
pub fn profile_forward_backward<F>(f: F) -> Result<ExecutionProfile>
where
    F: FnOnce() -> Result<Tensor>,
{
    let t0 = Instant::now();
    let out = f()?;
    let forward_duration = t0.elapsed();

    let graph_stats = analyze_graph(std::slice::from_ref(&out));
    let loss = if out.numel() == 1 { out } else { out.sum()? };

    let t1 = Instant::now();
    loss.backward()?;
    let backward_duration = t1.elapsed();

    Ok(ExecutionProfile {
        forward_duration,
        backward_duration,
        graph_stats,
    })
}

fn collect_topological(outputs: &[Tensor]) -> Vec<Tensor> {
    // Iterative post-order traversal with an explicit heap stack: the
    // recursive DFS overflowed the call stack on long chains (a 100k-add
    // graph aborts the process). Same shape as `Tensor::topological_order`.
    let mut visited = HashSet::new();
    let mut order = Vec::new();
    let mut stack: Vec<(Tensor, bool)> = outputs.iter().map(|t| (t.clone(), false)).collect();
    while let Some((t, expanded)) = stack.pop() {
        if expanded {
            order.push(t);
            continue;
        }
        if !visited.insert(t.id()) {
            continue;
        }
        stack.push((t.clone(), true));
        for inp in t.op_inputs() {
            if !visited.contains(&inp.id()) {
                stack.push((inp.clone(), false));
            }
        }
    }
    order
}

fn estimate_node_flops(op: &'static str, out: &Tensor, inputs: &[Tensor]) -> u64 {
    match op {
        "matmul" if inputs.len() == 2 => {
            let a_shape = inputs[0].shape();
            let out_shape = out.shape();
            if a_shape.len() >= 2 && out_shape.len() >= 2 {
                let k = a_shape[a_shape.len() - 1] as u64;
                let out_elems = out.numel() as u64;
                2u64.saturating_mul(out_elems).saturating_mul(k)
            } else {
                2 * (out.numel() as u64)
            }
        }
        "view" | "contiguous" | "to_dtype" | "to_device" | "cat" | "pad" => 0,
        "fft" | "ifft" => {
            let n = out.shape().iter().rev().nth(1).copied().unwrap_or(1) as f64;
            let batches = (out.numel() / 2) as f64 / n.max(1.0);
            (5.0 * batches * n * n.max(1.0).log2().max(1.0)) as u64
        }
        "sum_axes" | "extrema_axis" => inputs
            .first()
            .map_or(out.numel() as u64, |x| x.numel() as u64),
        _ => out.numel() as u64,
    }
}
