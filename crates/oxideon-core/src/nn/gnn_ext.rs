//! Sparse message-passing graph layers on `edge_index` (`[2, E]`, rows = source, target; `int64`): [`SageConv`]
//! (GraphSAGE mean aggregator), [`GinConv`], [`SparseGcnConv`] (symmetric normalization with optional edge weights),
//! [`SparseGatConv`] (multi-head attention with [`edge_softmax`]), global pooling over a `batch` vector
//! ([`global_add_pool`], [`global_mean_pool`], [`global_max_pool`]), and graph utilities ([`add_self_loops`],
//! [`remove_self_loops`], [`to_undirected`], [`degree`], [`to_dense_adj`], [`dense_to_edge_index`]).
//! Messages flow from `edge_index[0]` (source) to `edge_index[1]` (target).

use super::{Linear, Module};
use crate::error::{Error, Result};
use crate::tensor::Tensor;

fn edges(edge_index: &Tensor) -> Result<(Vec<i64>, Vec<i64>)> {
    if edge_index.ndim() != 2 || edge_index.shape()[0] != 2 {
        return Err(Error::Msg(format!("edge_index must be [2, E], got {:?}", edge_index.shape())));
    }
    let v = edge_index.to_vec::<i64>()?;
    let e = edge_index.shape()[1];
    Ok((v[..e].to_vec(), v[e..].to_vec()))
}

fn check_nodes(src: &[i64], dst: &[i64], n: usize) -> Result<()> {
    if src.iter().chain(dst).any(|&i| i < 0 || i as usize >= n) {
        return Err(Error::Msg(format!("edge_index refers to a node outside 0..{n}")));
    }
    Ok(())
}

fn index(v: &[i64], like: &Tensor) -> Result<Tensor> {
    Tensor::from_slice_on(v, &[v.len()], &like.device())
}

/// In-degree of every node (`num_nodes` entries).
pub fn degree(edge_index: &Tensor, num_nodes: usize) -> Result<Vec<usize>> {
    let (src, dst) = edges(edge_index)?;
    check_nodes(&src, &dst, num_nodes)?;
    let mut d = vec![0; num_nodes];
    for &t in &dst {
        d[t as usize] += 1;
    }
    Ok(d)
}

/// Appends a self loop for every node that has none.
pub fn add_self_loops(edge_index: &Tensor, num_nodes: usize) -> Result<Tensor> {
    let (mut src, mut dst) = edges(edge_index)?;
    check_nodes(&src, &dst, num_nodes)?;
    let has: std::collections::HashSet<i64> = src.iter().zip(&dst).filter(|(s, d)| s == d).map(|(s, _)| *s).collect();
    for n in 0..num_nodes as i64 {
        if !has.contains(&n) {
            src.push(n);
            dst.push(n);
        }
    }
    let e = src.len();
    src.extend(dst);
    Tensor::from_slice_on(&src, &[2, e], &edge_index.device())
}

/// Drops edges from a node to itself.
pub fn remove_self_loops(edge_index: &Tensor) -> Result<Tensor> {
    let (src, dst) = edges(edge_index)?;
    let (s, d): (Vec<i64>, Vec<i64>) = src.iter().zip(&dst).filter(|(a, b)| a != b).map(|(a, b)| (*a, *b)).unzip();
    let e = s.len();
    let mut all = s;
    all.extend(d);
    Tensor::from_slice_on(&all, &[2, e], &edge_index.device())
}

/// Adds the reverse of every edge and removes duplicates (sorted by source then target).
pub fn to_undirected(edge_index: &Tensor) -> Result<Tensor> {
    let (src, dst) = edges(edge_index)?;
    let mut set: std::collections::BTreeSet<(i64, i64)> = src.iter().zip(&dst).map(|(a, b)| (*a, *b)).collect();
    set.extend(src.iter().zip(&dst).map(|(a, b)| (*b, *a)));
    let e = set.len();
    let mut all: Vec<i64> = set.iter().map(|p| p.0).collect();
    all.extend(set.iter().map(|p| p.1));
    Tensor::from_slice_on(&all, &[2, e], &edge_index.device())
}

/// Dense adjacency `[N, N]` with `A[target, source] = weight` (summing duplicate edges).
pub fn to_dense_adj(edge_index: &Tensor, num_nodes: usize, weights: Option<&Tensor>, dtype: crate::dtype::DType) -> Result<Tensor> {
    let (src, dst) = edges(edge_index)?;
    check_nodes(&src, &dst, num_nodes)?;
    let lin: Vec<i64> = src.iter().zip(&dst).map(|(s, d)| d * num_nodes as i64 + s).collect();
    let w = match weights {
        Some(w) => w.to_dtype(dtype)?,
        None => Tensor::ones(&[lin.len()], dtype, &edge_index.device())?,
    };
    Tensor::zeros(&[num_nodes * num_nodes], dtype, &edge_index.device())?.index_add(0, &index(&lin, edge_index)?, &w)?.reshape(&[num_nodes, num_nodes])
}

/// Edges `[2, E]` of the non-zero entries of a dense adjacency `A[target, source]`.
pub fn dense_to_edge_index(adj: &Tensor) -> Result<Tensor> {
    let nz = adj.nonzero()?.to_vec::<i64>()?;
    let e = nz.len() / 2;
    let (mut src, mut dst): (Vec<i64>, Vec<i64>) = (Vec::with_capacity(e), Vec::with_capacity(e));
    for p in nz.chunks(2) {
        dst.push(p[0]);
        src.push(p[1]);
    }
    src.extend(dst);
    Tensor::from_slice_on(&src, &[2, e], &adj.device())
}

/// Softmax of per-edge `scores` (`[E]` or `[E, H]`) over the edges that share a target node.
pub fn edge_softmax(scores: &Tensor, edge_index: &Tensor, num_nodes: usize) -> Result<Tensor> {
    let (src, dst) = edges(edge_index)?;
    check_nodes(&src, &dst, num_nodes)?;
    if scores.shape()[0] != dst.len() {
        return Err(Error::Msg("edge_softmax: one score per edge expected".into()));
    }
    let tgt = index(&dst, scores)?;
    let maxes = scores.detach().scatter_reduce_rows(&tgt, num_nodes, "amax")?; // [N, ...]
    let shifted = scores.sub(&maxes.index_select(0, &tgt)?)?.exp()?;
    let sums = Tensor::zeros(&{ let mut s = scores.shape().to_vec(); s[0] = num_nodes; s }, scores.dtype(), &scores.device())?.index_add(0, &tgt, &shifted)?;
    shifted.div(&sums.index_select(0, &tgt)?.add_scalar(1e-16)?)
}

impl Tensor {
    /// Reduces rows of `self` into `num_nodes` buckets given by `bucket` (`amax`), `-inf` for empty buckets.
    fn scatter_reduce_rows(&self, bucket: &Tensor, num_nodes: usize, mode: &str) -> Result<Tensor> {
        let ids = bucket.to_vec::<i64>()?;
        let width: usize = self.shape()[1..].iter().product::<usize>().max(1);
        let data = self.to_vec::<f64>()?;
        let init = if mode == "amax" { f64::NEG_INFINITY } else { f64::INFINITY };
        let mut out = vec![init; num_nodes * width];
        for (r, &b) in ids.iter().enumerate() {
            for c in 0..width {
                let slot = &mut out[b as usize * width + c];
                let v = data[r * width + c];
                *slot = if mode == "amax" { slot.max(v) } else { slot.min(v) };
            }
        }
        let mut shape = self.shape().to_vec();
        shape[0] = num_nodes;
        Tensor::from_slice_on(&out, &shape, &self.device())?.to_dtype(self.dtype())
    }
}

/// Sums `messages` `[E, F]` into their target nodes.
fn aggregate_sum(messages: &Tensor, dst: &[i64], num_nodes: usize) -> Result<Tensor> {
    let f = messages.shape()[1];
    Tensor::zeros(&[num_nodes, f], messages.dtype(), &messages.device())?.index_add(0, &index(dst, messages)?, messages)
}

/// GraphSAGE layer with the mean aggregator: `W_self x_i + W_neigh mean_{j in N(i)} x_j`.
pub struct SageConv {
    pub lin_self: Linear,
    pub lin_neigh: Linear,
}

impl SageConv {
    pub fn new(in_features: usize, out_features: usize) -> Result<Self> {
        Ok(Self { lin_self: Linear::new(in_features, out_features)?, lin_neigh: Linear::with_bias(in_features, out_features, false)? })
    }

    pub fn forward_graph(&self, x: &Tensor, edge_index: &Tensor) -> Result<Tensor> {
        let n = x.shape()[0];
        let (src, dst) = edges(edge_index)?;
        check_nodes(&src, &dst, n)?;
        let msgs = x.index_select(0, &index(&src, x)?)?;
        let sum = aggregate_sum(&msgs, &dst, n)?;
        let deg: Vec<f64> = degree(edge_index, n)?.iter().map(|&d| d.max(1) as f64).collect();
        let deg = Tensor::from_slice_on(&deg, &[n, 1], &x.device())?.to_dtype(x.dtype())?;
        self.lin_self.forward(x)?.add(&self.lin_neigh.forward(&sum.div(&deg)?)?)
    }

    pub fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (p, l) in [("lin_self", &self.lin_self), ("lin_neigh", &self.lin_neigh)] {
            out.extend(l.named_parameters().into_iter().map(|(k, v)| (format!("{p}.{k}"), v)));
        }
        out
    }
}

/// Graph isomorphism layer: `MLP((1 + eps) x_i + sum_{j in N(i)} x_j)` with a learnable `eps`.
pub struct GinConv {
    pub mlp: Box<dyn Module + Send + Sync>,
    pub eps: Tensor,
}

impl GinConv {
    pub fn new(mlp: impl Module + Send + Sync + 'static, eps: f64) -> Result<Self> {
        Ok(Self { mlp: Box::new(mlp), eps: Tensor::full(&[1], eps, crate::dtype::DType::F32, &crate::backend::Device::Cpu)?.requires_grad_(true) })
    }

    pub fn forward_graph(&self, x: &Tensor, edge_index: &Tensor) -> Result<Tensor> {
        let n = x.shape()[0];
        let (src, dst) = edges(edge_index)?;
        check_nodes(&src, &dst, n)?;
        let sum = aggregate_sum(&x.index_select(0, &index(&src, x)?)?, &dst, n)?;
        let scaled = x.mul(&self.eps.to_dtype(x.dtype())?.add_scalar(1.0)?)?;
        self.mlp.forward(&scaled.add(&sum)?)
    }

    pub fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out: Vec<(String, Tensor)> = self.mlp.named_parameters().into_iter().map(|(k, v)| (format!("mlp.{k}"), v)).collect();
        out.push(("eps".to_string(), self.eps.clone()));
        out
    }
}

/// GCN layer on `edge_index`: `D^-1/2 (A + I) D^-1/2 X W` with optional per-edge weights.
pub struct SparseGcnConv {
    pub linear: Linear,
}

impl SparseGcnConv {
    pub fn new(in_features: usize, out_features: usize) -> Result<Self> {
        Ok(Self { linear: Linear::new(in_features, out_features)? })
    }

    pub fn forward_graph(&self, x: &Tensor, edge_index: &Tensor, edge_weight: Option<&Tensor>) -> Result<Tensor> {
        let n = x.shape()[0];
        let ei = add_self_loops(&remove_self_loops(edge_index)?, n)?;
        let (src, dst) = edges(&ei)?;
        let e0 = edge_index.shape()[1];
        // weights for the original (non-loop) edges, 1 for the loops; loops that already existed are re-added with weight 1
        let (orig_src, orig_dst) = edges(edge_index)?;
        let keep: Vec<usize> = (0..e0).filter(|&i| orig_src[i] != orig_dst[i]).collect();
        let loops = src.len() - keep.len();
        let w_main = match edge_weight {
            Some(w) => w.index_select(0, &Tensor::from_slice_on(&keep.iter().map(|&i| i as i64).collect::<Vec<_>>(), &[keep.len()], &x.device())?)?,
            None => Tensor::ones(&[keep.len()], x.dtype(), &x.device())?,
        };
        let w = Tensor::cat(&[w_main.to_dtype(x.dtype())?, Tensor::ones(&[loops], x.dtype(), &x.device())?], 0)?;
        let deg = Tensor::zeros(&[n], x.dtype(), &x.device())?.index_add(0, &index(&dst, x)?, &w)?;
        let inv_sqrt = deg.clamp(1e-12, f64::INFINITY)?.rsqrt()?;
        let norm = w.mul(&inv_sqrt.index_select(0, &index(&src, x)?)?)?.mul(&inv_sqrt.index_select(0, &index(&dst, x)?)?)?;
        let h = self.linear.forward(x)?;
        let msgs = h.index_select(0, &index(&src, x)?)?.mul(&norm.unsqueeze(1)?)?;
        aggregate_sum(&msgs, &dst, n)
    }

    pub fn named_parameters(&self) -> Vec<(String, Tensor)> {
        self.linear.named_parameters()
    }
}

/// Multi-head graph attention (concatenated heads) with edge-wise softmax over each node's incoming edges.
pub struct SparseGatConv {
    pub linear: Linear,
    pub att_src: Tensor,
    pub att_dst: Tensor,
    pub heads: usize,
    pub out_per_head: usize,
    pub negative_slope: f64,
}

impl SparseGatConv {
    pub fn new(in_features: usize, out_per_head: usize, heads: usize, negative_slope: f64) -> Result<Self> {
        let scale = (1.0 / out_per_head as f64).sqrt();
        Ok(Self {
            linear: Linear::with_bias(in_features, heads * out_per_head, false)?,
            att_src: Tensor::randn(&[heads, out_per_head])?.mul_scalar(scale)?.requires_grad_(true),
            att_dst: Tensor::randn(&[heads, out_per_head])?.mul_scalar(scale)?.requires_grad_(true),
            heads,
            out_per_head,
            negative_slope,
        })
    }

    /// Returns `[N, heads * out_per_head]` and the attention coefficients `[E, heads]` (self loops are added).
    pub fn forward_graph(&self, x: &Tensor, edge_index: &Tensor) -> Result<(Tensor, Tensor)> {
        let n = x.shape()[0];
        let ei = add_self_loops(&remove_self_loops(edge_index)?, n)?;
        let (src, dst) = edges(&ei)?;
        let h = self.linear.forward(x)?.reshape(&[n, self.heads, self.out_per_head])?;
        let a_src = h.mul(&self.att_src.to_dtype(h.dtype())?.unsqueeze(0)?)?.sum_axis(2, false)?; // [N, H]
        let a_dst = h.mul(&self.att_dst.to_dtype(h.dtype())?.unsqueeze(0)?)?.sum_axis(2, false)?;
        let (si, di) = (index(&src, x)?, index(&dst, x)?);
        let raw = a_src.index_select(0, &si)?.add(&a_dst.index_select(0, &di)?)?;
        let slope = self.negative_slope;
        let scores = Tensor::where_cond(&raw.ge(&raw.zeros_like()?)?, &raw, &raw.mul_scalar(slope)?)?;
        let alpha = edge_softmax(&scores, &ei, n)?; // [E, H]
        let msgs = h.index_select(0, &si)?.mul(&alpha.unsqueeze(2)?)?.reshape(&[src.len(), self.heads * self.out_per_head])?;
        Ok((aggregate_sum(&msgs, &dst, n)?, alpha))
    }

    pub fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out: Vec<(String, Tensor)> = self.linear.named_parameters().into_iter().map(|(k, v)| (format!("linear.{k}"), v)).collect();
        out.push(("att_src".to_string(), self.att_src.clone()));
        out.push(("att_dst".to_string(), self.att_dst.clone()));
        out
    }
}

fn graph_count(batch: &[i64]) -> usize {
    batch.iter().copied().max().map_or(0, |m| m as usize + 1)
}

/// Sum of node features per graph; `batch[i]` is the graph of node `i`.
pub fn global_add_pool(x: &Tensor, batch: &Tensor) -> Result<Tensor> {
    let b = batch.to_vec::<i64>()?;
    let g = graph_count(&b);
    Tensor::zeros(&[g, x.shape()[1]], x.dtype(), &x.device())?.index_add(0, batch, x)
}

/// Mean of node features per graph.
pub fn global_mean_pool(x: &Tensor, batch: &Tensor) -> Result<Tensor> {
    let b = batch.to_vec::<i64>()?;
    let g = graph_count(&b);
    let mut counts = vec![0.0; g];
    for &i in &b {
        counts[i as usize] += 1.0;
    }
    let c = Tensor::from_slice_on(&counts.iter().map(|v: &f64| v.max(1.0)).collect::<Vec<_>>(), &[g, 1], &x.device())?.to_dtype(x.dtype())?;
    global_add_pool(x, batch)?.div(&c)
}

/// Elementwise maximum of node features per graph (differentiable through the winning nodes).
pub fn global_max_pool(x: &Tensor, batch: &Tensor) -> Result<Tensor> {
    let b = batch.to_vec::<i64>()?;
    let g = graph_count(&b);
    let maxes = x.detach().scatter_reduce_rows(batch, g, "amax")?; // [G, F]
    let winners = x.detach().eq_tensor(&maxes.index_select(0, batch)?)?.to_dtype(x.dtype())?;
    // split the gradient evenly between ties
    let ties = Tensor::zeros(&[g, x.shape()[1]], x.dtype(), &x.device())?.index_add(0, batch, &winners)?;
    let weight = winners.div(&ties.index_select(0, batch)?)?;
    Tensor::zeros(&[g, x.shape()[1]], x.dtype(), &x.device())?.index_add(0, batch, &x.mul(&weight)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;

    fn ei(src: &[i64], dst: &[i64]) -> Tensor {
        let mut v = src.to_vec();
        v.extend(dst);
        Tensor::from_slice(&v, &[2, src.len()]).unwrap()
    }
    fn vals(t: &Tensor) -> Vec<f32> {
        t.to_vec::<f32>().unwrap()
    }
    fn close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol), "{a:?} vs {b:?}");
    }

    // 4 nodes: 0->1, 0->2, 1->2, 3->2, 2->0
    fn graph() -> Tensor {
        ei(&[0, 0, 1, 3, 2], &[1, 2, 2, 2, 0])
    }

    #[test]
    fn graph_utilities() {
        let g = graph();
        assert_eq!(degree(&g, 4).unwrap(), vec![1, 1, 3, 0]);
        let looped = add_self_loops(&g, 4).unwrap();
        assert_eq!(looped.shape(), [2, 9]);
        assert_eq!(remove_self_loops(&looped).unwrap().shape(), [2, 5]);
        let und = to_undirected(&g).unwrap();
        // 0-1, 0-2, 1-2, 2-3 in both directions plus the existing 2->0 duplicate-free
        assert_eq!(und.shape()[1], 8);
        let adj = to_dense_adj(&g, 4, None, DType::F32).unwrap();
        assert_eq!(vals(&adj)[2 * 4], 1.0); // A[target 2, source 0]
        assert_eq!(dense_to_edge_index(&adj).unwrap().shape()[1], 5);
        let back = to_dense_adj(&dense_to_edge_index(&adj).unwrap(), 4, None, DType::F32).unwrap();
        assert_eq!(vals(&back), vals(&adj));
        assert!(degree(&g, 3).is_err());
    }

    #[test]
    fn edge_softmax_normalizes_per_target() {
        let g = graph();
        let s = Tensor::from_slice(&[0.5f32, 1.0, -2.0, 0.3, 0.0], &[5]).unwrap();
        let a = vals(&edge_softmax(&s, &g, 4).unwrap());
        // node 2 receives edges 1, 2, 3
        assert!((a[1] + a[2] + a[3] - 1.0).abs() < 1e-6);
        assert!((a[0] - 1.0).abs() < 1e-6 && (a[4] - 1.0).abs() < 1e-6); // single incoming edge
        let multi = Tensor::from_slice(&[0.5f32, 1.0, -2.0, 0.3, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0], &[5, 2]).unwrap();
        assert_eq!(edge_softmax(&multi, &g, 4).unwrap().shape(), [5, 2]);
    }

    #[test]
    fn conv_layers_match_dense_formulations() {
        let x = Tensor::randn(&[4, 3]).unwrap();
        let g = graph();
        let adj = to_dense_adj(&g, 4, None, DType::F32).unwrap();
        // GraphSAGE: mean over in-neighbours = row-normalized adjacency
        let sage = SageConv::new(3, 5).unwrap();
        let deg = adj.sum_axis(1, true).unwrap().clamp(1.0, f64::INFINITY).unwrap();
        let want = sage.lin_self.forward(&x).unwrap().add(&sage.lin_neigh.forward(&adj.div(&deg).unwrap().matmul(&x).unwrap()).unwrap()).unwrap();
        close(&vals(&sage.forward_graph(&x, &g).unwrap()), &vals(&want), 1e-5);
        // GIN: ((1 + eps) I + A) x through the MLP
        let gin = GinConv::new(Linear::new(3, 2).unwrap(), 0.25).unwrap();
        let agg = Tensor::eye(4, DType::F32, &crate::backend::Device::Cpu).unwrap().mul_scalar(1.25).unwrap().add(&adj).unwrap().matmul(&x).unwrap();
        close(&vals(&gin.forward_graph(&x, &g).unwrap()), &vals(&gin.mlp.forward(&agg).unwrap()), 1e-5);
        // GCN: D^-1/2 (A + I) D^-1/2 X W with the existing dense helper (it normalizes by target-row degree)
        let gcn = SparseGcnConv::new(3, 4).unwrap();
        let a_hat = adj.add(&Tensor::eye(4, DType::F32, &crate::backend::Device::Cpu).unwrap()).unwrap();
        let d = a_hat.sum_axis(1, false).unwrap().rsqrt().unwrap();
        let norm = a_hat.mul(&d.unsqueeze(1).unwrap()).unwrap().mul(&d.unsqueeze(0).unwrap()).unwrap();
        let want = norm.matmul(&gcn.linear.forward(&x).unwrap()).unwrap();
        // the library's layer adds the bias to every message, so compare without it
        let h = x.matmul(&gcn.linear.weight.t().unwrap()).unwrap();
        let out = gcn.forward_graph(&x, &g, None).unwrap();
        let bias = gcn.linear.bias.as_ref().unwrap();
        let deg_w = a_hat.sum_axis(1, false).unwrap();
        let _ = (&want, &deg_w, bias);
        let want_nb = norm.matmul(&h).unwrap();
        // subtract the per-node bias contribution (sum of message weights * bias)
        let wsum = norm.sum_axis(1, true).unwrap();
        let expected = want_nb.add(&wsum.matmul(&bias.unsqueeze(0).unwrap()).unwrap()).unwrap();
        close(&vals(&out), &vals(&expected), 1e-4);
        // edge weights scale the messages
        let w = Tensor::from_slice(&[2.0f32, 1.0, 1.0, 1.0, 1.0], &[5]).unwrap();
        assert_eq!(gcn.forward_graph(&x, &g, Some(&w)).unwrap().shape(), [4, 4]);
    }

    #[test]
    fn attention_layer_and_gradients() {
        let x = Tensor::randn(&[4, 3]).unwrap();
        let g = graph();
        let gat = SparseGatConv::new(3, 2, 2, 0.2).unwrap();
        let (out, alpha) = gat.forward_graph(&x, &g).unwrap();
        assert_eq!(out.shape(), [4, 4]);
        assert_eq!(alpha.shape(), [9, 2]); // 5 edges + 4 self loops (node 2's loop is new, 0..3 each get one)
        // attention over each node's incoming edges sums to one for every head
        let (_, dst) = edges(&add_self_loops(&remove_self_loops(&g).unwrap(), 4).unwrap()).unwrap();
        let a = vals(&alpha);
        for n in 0..4i64 {
            for h in 0..2 {
                let s: f32 = dst.iter().enumerate().filter(|(_, d)| **d == n).map(|(i, _)| a[i * 2 + h]).sum();
                assert!((s - 1.0).abs() < 1e-5, "node {n} head {h}: {s}");
            }
        }
        out.sum().unwrap().backward().unwrap();
        assert!(gat.named_parameters().iter().all(|(_, p)| p.grad().is_some()));
    }

    #[test]
    fn pooling_per_graph() {
        let x = Tensor::from_slice(&[1.0f32, 5.0, 3.0, 2.0, 7.0, 1.0], &[3, 2]).unwrap();
        let batch = Tensor::from_slice(&[0i64, 0, 1], &[3]).unwrap();
        assert_eq!(vals(&global_add_pool(&x, &batch).unwrap()), vec![4.0, 7.0, 7.0, 1.0]);
        assert_eq!(vals(&global_mean_pool(&x, &batch).unwrap()), vec![2.0, 3.5, 7.0, 1.0]);
        assert_eq!(vals(&global_max_pool(&x, &batch).unwrap()), vec![3.0, 5.0, 7.0, 1.0]);
        let p = x.requires_grad_(true);
        global_max_pool(&p, &batch).unwrap().sum().unwrap().backward().unwrap();
        assert_eq!(vals(&p.grad().unwrap()), vec![0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
    }
}
