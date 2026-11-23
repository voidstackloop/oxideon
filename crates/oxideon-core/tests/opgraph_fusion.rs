// Graph fusion must never change results: each fused graph is compared with the same graph evaluated eagerly.
use oxideon_core::opgraph::OpGraph;
use oxideon_core::{DType, Device, Tensor};

fn rnd(shape: &[usize], seed: f32) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_slice(&(0..n).map(|i| ((i as f32 + seed) * 0.713).sin()).collect::<Vec<f32>>(), shape).unwrap()
}
fn close(what: &str, a: &Tensor, b: &Tensor) {
    let (x, y) = (a.to_vec::<f32>().unwrap(), b.to_vec::<f32>().unwrap());
    assert_eq!(x.len(), y.len(), "{what}");
    for (p, q) in x.iter().zip(&y) {
        assert!((p - q).abs() <= 1e-4 * (1.0 + q.abs()), "{what}: {p} vs {q}");
    }
}

#[allow(clippy::too_many_arguments)]
fn eager_conv_bn(x: &Tensor, w: &Tensor, b: &Tensor, g: &Tensor, be: &Tensor, m: &Tensor, v: &Tensor, eps: f64) -> Tensor {
    let y = x.conv2d(w, Some(b), (1, 1), (1, 1), (1, 1)).unwrap();
    let c = y.shape()[1];
    let r = |t: &Tensor| t.reshape(&[1, c, 1, 1]).unwrap();
    y.sub(&r(m)).unwrap().mul(&r(&v.add_scalar(eps).unwrap().rsqrt().unwrap())).unwrap().mul(&r(g)).unwrap().add(&r(be)).unwrap()
}

#[test]
fn conv_bn_relu_followed_by_more_ops_and_with_other_consumers() {
    let dev = Device::Cpu;
    let (x, w, b) = (rnd(&[2, 3, 6, 6], 0.0), rnd(&[4, 3, 3, 3], 1.0).mul_scalar(0.3).unwrap(), rnd(&[4], 2.0));
    let (g, be, m) = (rnd(&[4], 3.0).add_scalar(2.0).unwrap(), rnd(&[4], 4.0), rnd(&[4], 5.0));
    let v = rnd(&[4], 6.0).abs().unwrap().add_scalar(0.5).unwrap();
    let eps = 1e-5;
    let _ = &dev;
    let mut gr = OpGraph::new();
    let xi = gr.input(&[2, 3, 6, 6], DType::F32).unwrap();
    let (wi, bi) = (gr.constant(&w).unwrap(), gr.constant(&b).unwrap());
    let (gi, bei, mi, vi) = (gr.constant(&g).unwrap(), gr.constant(&be).unwrap(), gr.constant(&m).unwrap(), gr.constant(&v).unwrap());
    let c = gr.conv2d(xi, wi, Some(bi), (1, 1), (1, 1), (1, 1), 1).unwrap();
    let n = gr.batch_norm2d(c, gi, bei, mi, vi, eps).unwrap();
    let r = gr.relu(n).unwrap();
    // consumers that come after the fused node: a further op, and a second use of the relu output
    let tail = gr.mul_scalar(r, 3.0).unwrap();
    let both = gr.add(tail, r).unwrap();
    gr.set_outputs(&[both]);
    let comp = gr.compile().unwrap();
    assert!(comp.stats().fused_kernels >= 1, "the pattern should fuse");
    let want = {
        let y = eager_conv_bn(&x, &w, &b, &g, &be, &m, &v, eps).relu().unwrap();
        y.mul_scalar(3.0).unwrap().add(&y).unwrap()
    };
    close("conv+bn+relu then more ops", &comp.run(&[x.clone()]).unwrap()[0], &want);
    // a second input must work too (the fusion folded constants, not data)
    let x2 = rnd(&[2, 3, 6, 6], 50.0);
    let want2 = {
        let y = eager_conv_bn(&x2, &w, &b, &g, &be, &m, &v, eps).relu().unwrap();
        y.mul_scalar(3.0).unwrap().add(&y).unwrap()
    };
    close("different input", &comp.run(&[x2]).unwrap()[0], &want2);
}

#[test]
fn an_intermediate_used_twice_blocks_the_fusion_and_stays_correct() {
    let (x, w) = (rnd(&[1, 2, 5, 5], 0.0), rnd(&[3, 2, 3, 3], 1.0));
    let mut gr = OpGraph::new();
    let xi = gr.input(&[1, 2, 5, 5], DType::F32).unwrap();
    let wi = gr.constant(&w).unwrap();
    let c = gr.conv2d(xi, wi, None, (1, 1), (1, 1), (1, 1), 1).unwrap();
    let r = gr.relu(c).unwrap();
    let out = gr.add(r, c).unwrap(); // `c` also feeds the add: it must still be computed un-activated
    gr.set_outputs(&[out]);
    let comp = gr.compile().unwrap();
    let y = x.conv2d(&w, None, (1, 1), (1, 1), (1, 1)).unwrap();
    close("conv used twice", &comp.run(&[x]).unwrap()[0], &y.relu().unwrap().add(&y).unwrap());
}

fn attention_graph(softmax_axis: isize, perm: &[usize]) -> (oxideon_core::opgraph::CompiledGraph, usize) {
    let mut gr = OpGraph::new();
    let q = gr.input(&[2, 2, 4, 3], DType::F32).unwrap();
    let k = gr.input(&[2, 2, 4, 3], DType::F32).unwrap();
    let v = gr.input(&[2, 2, 4, 3], DType::F32).unwrap();
    let kt = gr.permute(k, perm).unwrap();
    let s = gr.matmul(q, kt).unwrap();
    let sc = gr.mul_scalar(s, 0.5).unwrap();
    let sm = gr.softmax(sc, softmax_axis).unwrap();
    let o = gr.matmul(sm, v).unwrap();
    gr.set_outputs(&[o]);
    let comp = gr.compile().unwrap();
    let fused = comp.stats().fused_kernels;
    (comp, fused)
}

#[test]
fn attention_fuses_only_when_it_is_really_attention() {
    let (q, k, v) = (rnd(&[2, 2, 4, 3], 0.0), rnd(&[2, 2, 4, 3], 7.0), rnd(&[2, 2, 4, 3], 13.0));
    let attn = |kt: &Tensor, axis: isize| {
        q.matmul(kt).unwrap().mul_scalar(0.5).unwrap().softmax(axis).unwrap().matmul(&v).unwrap()
    };
    // the real thing: K transposed over its last two axes, softmax over the key axis
    let (comp, fused) = attention_graph(-1, &[0, 1, 3, 2]);
    assert_eq!(fused, 1, "genuine attention fuses");
    close("attention", &comp.run(&[q.clone(), k.clone(), v.clone()]).unwrap()[0], &attn(&k.transpose(2, 3).unwrap(), -1));
    // softmax over another axis is NOT attention
    let (comp, fused) = attention_graph(2, &[0, 1, 3, 2]);
    assert_eq!(fused, 0, "softmax over the query axis must not fuse");
    close("softmax over axis 2", &comp.run(&[q.clone(), k.clone(), v.clone()]).unwrap()[0], &attn(&k.transpose(2, 3).unwrap(), 2));
    // a permute that also swaps the batch axes is not a plain transpose
    let (comp, fused) = attention_graph(-1, &[1, 0, 3, 2]);
    assert_eq!(fused, 0, "a permute that moves batch axes must not fuse");
    let kperm = k.permute(&[1, 0, 3, 2]).unwrap();
    close("permuted batch axes", &comp.run(&[q.clone(), k, v.clone()]).unwrap()[0], &attn(&kperm, -1));
}

#[test]
fn batch_norm_with_non_constant_parameters_still_folds_and_follows_the_inputs() {
    let mut gr = OpGraph::new();
    let xi = gr.input(&[2, 3, 6, 6], DType::F32).unwrap();
    let wi = gr.input(&[4, 3, 3, 3], DType::F32).unwrap();
    let (gi, bei, mi, vi) = (gr.input(&[4], DType::F32).unwrap(), gr.input(&[4], DType::F32).unwrap(), gr.input(&[4], DType::F32).unwrap(), gr.input(&[4], DType::F32).unwrap());
    let c = gr.conv2d(xi, wi, None, (1, 1), (1, 1), (1, 1), 1).unwrap();
    let n = gr.batch_norm2d(c, gi, bei, mi, vi, 1e-5).unwrap();
    let r = gr.relu(n).unwrap();
    gr.set_outputs(&[r]);
    let comp = gr.compile().unwrap();
    assert!(comp.stats().fused_kernels >= 1, "the pattern folds even though the parameters are runtime values");
    // two different parameter sets through the same compiled graph
    for seed in [0.0f32, 40.0] {
        let x = rnd(&[2, 3, 6, 6], seed);
        let w = rnd(&[4, 3, 3, 3], seed + 1.0).mul_scalar(0.3).unwrap();
        let g = rnd(&[4], seed + 2.0).add_scalar(2.0).unwrap();
        let be = rnd(&[4], seed + 3.0);
        let m = rnd(&[4], seed + 4.0);
        let v = rnd(&[4], seed + 5.0).abs().unwrap().add_scalar(0.5).unwrap();
        let zeros = Tensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap();
        let want = eager_conv_bn(&x, &w, &zeros, &g, &be, &m, &v, 1e-5).relu().unwrap();
        close("runtime-parameter fold", &comp.run(&[x, w, g, be, m, v]).unwrap()[0], &want);
    }
}
