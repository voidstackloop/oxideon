//! Tensors on the CUDA device against the CPU backend: values and gradients of whole computations (elementwise chains,
//! broadcasting, reductions, softmax / log-softmax, layer norm, convolution, pooling, matmul on views) and a short training
//! run. Skipped when there is no CUDA device or no NVRTC to compile the `.cu` kernels (see `backend::nvrtc` for where it is
//! searched; set `OXIDEON_NVRTC_SITE` to a Python `site-packages` that has `nvidia-cuda-nvrtc-cu12`).

use oxideon_core::nn::{LayerNorm, Linear, Module};
use oxideon_core::{DType, Device, Tensor};

fn ready() -> bool {
    let ok = oxideon_core::backend::cuda_rt::available();
    if !ok {
        eprintln!("skipped: no CUDA device / NVRTC");
    }
    ok
}

fn rnd(shape: &[usize], seed: f32) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_slice(&(0..n).map(|i| ((i as f32 + seed) * 0.713).sin() * 1.5 + 0.1 * ((i % 5) as f32 - 2.0)).collect::<Vec<f32>>(), shape).unwrap()
}

fn cuda() -> Device {
    Device::Cuda(0)
}

fn close(what: &str, got: &Tensor, want: &Tensor, tol: f64) {
    let got = got.to_device(&Device::Cpu).unwrap();
    assert_eq!(got.shape(), want.shape(), "{what}: shape");
    let (g, w) = (got.to_dtype(DType::F64).unwrap().to_vec::<f64>().unwrap(), want.to_dtype(DType::F64).unwrap().to_vec::<f64>().unwrap());
    for (i, (x, y)) in g.iter().zip(&w).enumerate() {
        assert!((x.is_nan() && y.is_nan()) || x == y || (x - y).abs() <= tol * (1.0 + y.abs()), "{what}: element {i}: gpu {x} vs cpu {y}");
    }
}

/// Runs `f` on CPU leaves and on the same data moved to CUDA; compares the scalar loss and every input gradient.
fn same_on_both(name: &str, inputs: &[Tensor], f: impl Fn(&[Tensor]) -> Tensor, tol: f64) {
    let leaf = |t: &Tensor, dev: &Device| t.detach().to_device(dev).unwrap().requires_grad_(true);
    let cpu_in: Vec<Tensor> = inputs.iter().map(|t| leaf(t, &Device::Cpu)).collect();
    let gpu_in: Vec<Tensor> = inputs.iter().map(|t| leaf(t, &cuda())).collect();
    let before = oxideon_core::backend::cuda_rt::launch_count();
    let (lc, lg) = (f(&cpu_in), f(&gpu_in));
    assert!(oxideon_core::backend::cuda_rt::launch_count() > before, "{name}: no .cu kernel ran, the GPU path was not used");
    assert_eq!(lg.device(), cuda(), "{name}: the result must stay on the device");
    close(&format!("{name} value"), &lg, &lc, tol);
    lc.sum().unwrap().backward().unwrap();
    lg.sum().unwrap().backward().unwrap();
    for (i, (c, g)) in cpu_in.iter().zip(&gpu_in).enumerate() {
        match (c.grad(), g.grad()) {
            (Some(cg), Some(gg)) => close(&format!("{name} grad {i}"), &gg, &cg, tol * 4.0),
            (None, None) => {}
            _ => panic!("{name}: gradient presence differs for input {i}"),
        }
    }
}

#[test]
fn elementwise_broadcast_reduction_and_matmul_graphs_agree() {
    if !ready() {
        return;
    }
    same_on_both("elementwise chain", &[rnd(&[64, 33], 0.0), rnd(&[33], 1.0), rnd(&[64, 33], 2.0)], |x| x[0].mul(&x[1]).unwrap().add(&x[2]).unwrap().tanh().unwrap().gelu().unwrap().mul(&x[0]).unwrap(), 2e-5);
    same_on_both("broadcast + view matmul", &[rnd(&[8, 40, 24], 3.0), rnd(&[24, 40], 4.0), rnd(&[40], 5.0)], |x| x[0].matmul(&x[1].transpose(0, 1).unwrap().transpose(0, 1).unwrap()).unwrap().add(&x[2]).unwrap().relu().unwrap(), 5e-5);
    same_on_both("reductions", &[rnd(&[6, 50, 7], 6.0)], |x| {
        let m = x[0].mean_axis(1, false).unwrap();
        let v = x[0].var_axis(2, true, false).unwrap();
        m.sum_axis(0, false).unwrap().mul_scalar(0.5).unwrap().add(&v.sum_axis(0, false).unwrap().narrow(0, 0, 7).unwrap().mean().unwrap()).unwrap().add(&x[0].max_axis(1, false).unwrap().sum().unwrap()).unwrap()
    }, 5e-5);
    same_on_both("logsumexp + where", &[rnd(&[20, 31], 7.0), rnd(&[20, 31], 8.0)], |x| {
        let mask = x[0].gt(&x[1]).unwrap();
        Tensor::where_cond(&mask, &x[0], &x[1]).unwrap().logsumexp(1, false).unwrap()
    }, 5e-5);
}

#[test]
fn softmax_layer_norm_conv_and_pooling_agree() {
    if !ready() {
        return;
    }
    same_on_both("softmax", &[rnd(&[16, 100], 9.0)], |x| x[0].softmax(-1).unwrap().mul(&rnd(&[16, 100], 1.0).to_device(&x[0].device()).unwrap()).unwrap(), 5e-5);
    same_on_both("log_softmax", &[rnd(&[16, 100], 10.0)], |x| x[0].log_softmax(-1).unwrap().mul(&rnd(&[16, 100], 2.0).to_device(&x[0].device()).unwrap()).unwrap(), 5e-5);
    same_on_both("softmax over a middle axis", &[rnd(&[4, 9, 6], 11.0)], |x| x[0].softmax(1).unwrap().square().unwrap(), 5e-5);

    // layer norm module with trainable affine parameters
    let ln_cpu = LayerNorm::new(48, 1e-5).unwrap();
    let ln_gpu = LayerNorm::new(48, 1e-5).unwrap();
    for (a, b) in ln_gpu.parameters().iter().zip(ln_cpu.parameters()) {
        b.assign(&b.add_scalar(0.3).unwrap()).unwrap();
        a.retarget(&b.detach().to_device(&cuda()).unwrap()).unwrap();
    }
    let x = rnd(&[10, 48], 12.0);
    let (xc, xg) = (x.detach().requires_grad_(true), x.detach().to_device(&cuda()).unwrap().requires_grad_(true));
    let (yc, yg) = (ln_cpu.forward(&xc).unwrap(), ln_gpu.forward(&xg).unwrap());
    assert_eq!(yg.device(), cuda());
    close("layer_norm value", &yg, &yc, 3e-5);
    yc.mul(&rnd(&[10, 48], 3.0)).unwrap().sum().unwrap().backward().unwrap();
    yg.mul(&rnd(&[10, 48], 3.0).to_device(&cuda()).unwrap()).unwrap().sum().unwrap().backward().unwrap();
    close("layer_norm grad x", &xg.grad().unwrap(), &xc.grad().unwrap(), 2e-4);
    for (g, c) in ln_gpu.parameters().iter().zip(ln_cpu.parameters()) {
        close("layer_norm grad param", &g.grad().unwrap(), &c.grad().unwrap(), 2e-4);
    }

    // convolution and pooling, forward and backward
    same_on_both("conv2d", &[rnd(&[2, 3, 13, 11], 13.0), rnd(&[5, 3, 3, 3], 14.0), rnd(&[5], 15.0)], |x| x[0].conv2d(&x[1], Some(&x[2]), (2, 1), (1, 2), (1, 1)).unwrap().relu().unwrap(), 2e-4);
    same_on_both("conv2d dilated", &[rnd(&[1, 2, 15, 15], 16.0), rnd(&[3, 2, 3, 3], 17.0)], |x| x[0].conv2d(&x[1], None, (1, 1), (2, 2), (2, 2)).unwrap(), 2e-4);
    same_on_both("max_pool2d", &[rnd(&[2, 4, 11, 10], 18.0)], |x| x[0].max_pool2d((3, 2), (2, 1), (1, 0)).unwrap().square().unwrap(), 5e-5);
    same_on_both("max_pool2d padded", &[rnd(&[1, 2, 8, 8], 19.0)], |x| x[0].max_pool2d((3, 3), (2, 2), (1, 1)).unwrap(), 5e-5);
    same_on_both("avg_pool2d", &[rnd(&[2, 3, 9, 9], 20.0)], |x| x[0].avg_pool2d((3, 3), (2, 2), (1, 1)).unwrap(), 5e-5);
    same_on_both("embedding-style index_select", &[rnd(&[30, 12], 21.0)], |x| {
        let idx = Tensor::from_slice(&[3i64, 7, 7, 0, 29], &[5]).unwrap().to_device(&x[0].device()).unwrap();
        x[0].index_select(0, &idx).unwrap().square().unwrap()
    }, 5e-5);
}

#[test]
fn a_small_network_trains_to_the_same_weights_on_the_gpu() {
    if !ready() {
        return;
    }
    use oxideon_core::optim::{Optimizer, Sgd};
    let build = |dev: &Device| {
        oxideon_core::rng::manual_seed(42);
        let l1 = Linear::new(16, 24).unwrap();
        let l2 = Linear::new(24, 4).unwrap();
        for p in l1.parameters().iter().chain(l2.parameters().iter()) {
            p.retarget(&p.detach().to_device(dev).unwrap()).unwrap();
        }
        (l1, l2)
    };
    let x = rnd(&[32, 16], 30.0);
    let labels = Tensor::from_slice(&(0..32).map(|i| (i * 7 % 4) as i64).collect::<Vec<_>>(), &[32]).unwrap();
    let run = |dev: &Device| {
        let (l1, l2) = build(dev);
        let params: Vec<Tensor> = l1.parameters().into_iter().chain(l2.parameters()).collect();
        let mut opt = Sgd::new(params, 0.1);
        let (xd, yd) = (x.to_device(dev).unwrap(), labels.to_device(dev).unwrap());
        let mut losses = Vec::new();
        for _ in 0..25 {
            opt.zero_grad();
            let logits = l2.forward(&l1.forward(&xd).unwrap().gelu().unwrap()).unwrap();
            let lp = logits.log_softmax(-1).unwrap();
            let picked = lp.gather(1, &yd.unsqueeze(1).unwrap()).unwrap();
            let loss = picked.mean().unwrap().neg().unwrap();
            loss.backward().unwrap();
            opt.step().unwrap();
            losses.push(loss.item::<f32>().unwrap());
        }
        (losses, l1.parameters()[0].detach().to_device(&Device::Cpu).unwrap())
    };
    let (lc, wc) = run(&Device::Cpu);
    let before = oxideon_core::backend::cuda_rt::launch_count();
    let (lg, wg) = run(&cuda());
    assert!(oxideon_core::backend::cuda_rt::launch_count() > before + 25 * 10, "training must run on .cu kernels");
    assert!(lc.last().unwrap() < &lc[0], "the CPU run should be learning: {lc:?}");
    for (i, (a, b)) in lc.iter().zip(&lg).enumerate() {
        assert!((a - b).abs() <= 2e-4 * (1.0 + a.abs()), "loss at step {i}: cpu {a} vs gpu {b}");
    }
    close("trained weights", &wg, &wc, 5e-4);
}

#[test]
fn data_movement_ops_agree_in_values_and_gradients() {
    if !ready() {
        return;
    }
    let idx_i = |v: &[i64], shape: &[usize], dev: &Device| Tensor::from_slice(v, shape).unwrap().to_device(dev).unwrap();
    same_on_both("cat axes + stack", &[rnd(&[3, 4, 5], 1.0), rnd(&[3, 2, 5], 2.0), rnd(&[3, 4, 5], 3.0)], |x| {
        let a = Tensor::cat(&[x[0].clone(), x[1].clone()], 1).unwrap();
        let b = Tensor::cat(&[x[0].clone(), x[2].clone()], 0).unwrap();
        let s = Tensor::stack(&[x[0].clone(), x[2].clone()], 2).unwrap();
        a.sum_axis(1, false).unwrap().sum().unwrap().add(&b.square().unwrap().sum().unwrap()).unwrap().add(&s.mul(&s).unwrap().sum().unwrap()).unwrap()
    }, 5e-5);
    same_on_both("cat of views", &[rnd(&[6, 8], 4.0)], |x| {
        let t = x[0].transpose(0, 1).unwrap(); // [8, 6]
        let parts = [t.narrow(0, 1, 3).unwrap(), t.flip(&[0]).unwrap().narrow(0, 0, 2).unwrap(), t.narrow(1, 0, 6).unwrap().narrow(0, 5, 3).unwrap()];
        Tensor::cat(&parts, 0).unwrap().square().unwrap()
    }, 5e-5);
    same_on_both("pad / repeat / roll / flip", &[rnd(&[4, 5], 5.0)], |x| {
        let p = x[0].pad(&[(1, 2), (0, 3)], 1.5).unwrap();
        let r = x[0].repeat(&[2, 3]).unwrap();
        let l = x[0].roll(2, 1).unwrap().flip(&[0]).unwrap();
        p.sum().unwrap().add(&r.square().unwrap().sum().unwrap()).unwrap().add(&l.mul(&l).unwrap().sum().unwrap()).unwrap()
    }, 5e-5);
    // indexing: indices are rebuilt on each device (the helper moves them with the tensors)
    for (name, f) in [
        ("gather", Box::new(|x: &[Tensor]| {
            let i = Tensor::from_slice(&[0i64, 2, 1, 1, 3, 0, 2, 2, 1, 0, 3, 1], &[3, 4]).unwrap().to_device(&x[0].device()).unwrap();
            x[0].gather(1, &i).unwrap().square().unwrap()
        }) as Box<dyn Fn(&[Tensor]) -> Tensor>),
        ("scatter_add", Box::new(|x: &[Tensor]| {
            let i = Tensor::from_slice(&[0i64, 2, 1, 1, 3, 0], &[3, 2]).unwrap().to_device(&x[0].device()).unwrap();
            x[0].scatter_add(1, &i, &x[1]).unwrap().square().unwrap()
        })),
        ("index_add", Box::new(|x: &[Tensor]| {
            let i = Tensor::from_slice(&[2i64, 0, 2], &[3]).unwrap().to_device(&x[0].device()).unwrap();
            x[0].index_add(0, &i, &x[2]).unwrap().square().unwrap()
        })),
        ("scatter", Box::new(|x: &[Tensor]| {
            let i = Tensor::from_slice(&[3i64, 0, 1, 2, 2, 1], &[3, 2]).unwrap().to_device(&x[0].device()).unwrap();
            x[0].scatter(1, &i, &x[1]).unwrap().square().unwrap()
        })),
    ] {
        same_on_both(name, &[rnd(&[3, 4], 6.0), rnd(&[3, 2], 7.0), rnd(&[3, 4], 8.0)], &f, 5e-5);
    }
    // out-of-range indices must be reported as errors, like on the CPU, not fault the GPU
    let x = rnd(&[3, 4], 9.0).to_device(&cuda()).unwrap();
    let bad = idx_i(&[0, 1, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[3, 4], &cuda());
    assert!(x.gather(1, &bad).is_err());
    assert!(x.add_scalar(1.0).is_ok(), "the device must still work after a rejected index");
}

#[test]
fn sort_topk_cumsum_and_device_indices_agree() {
    if !ready() {
        return;
    }
    let exact = |what: &str, a: &Tensor, b: &Tensor| close(what, a, b, 0.0);
    // values with duplicates, NaN, infinities and both zeros so the tie-break and the total order are exercised
    let spiky = |n: usize, seed: f32| {
        let mut v: Vec<f32> = (0..n).map(|i| (((i as f32 + seed) * 0.37).sin() * 3.0).round() / 2.0).collect();
        for (k, s) in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0, f32::NAN].into_iter().enumerate() {
            if k * 7 < n {
                v[k * 7] = s;
            }
        }
        Tensor::from_slice(&v, &[n]).unwrap()
    };
    for (rows, n) in [(1usize, 37usize), (5, 300), (3, 4096), (2, 5000), (2, 20_000), (1, 70_000)] {
        let flat = spiky(rows * n, 1.0).reshape(&[rows, n]).unwrap();
        let g = flat.to_device(&cuda()).unwrap();
        for largest in [true, false] {
            for k in [1usize, 7, n.min(100), n.min(2049), n.min(4096), n] {
                let (cv, ci) = flat.topk(k, 1, largest, true).unwrap();
                let (gv, gi) = g.topk(k, 1, largest, true).unwrap();
                exact(&format!("topk values {rows}x{n} k={k} largest={largest}"), &gv, &cv);
                exact(&format!("topk indices {rows}x{n} k={k} largest={largest}"), &gi, &ci);
            }
        }
    }
    // full sort, ascending and descending, on another axis and on integers
    let m = spiky(6 * 40 * 3, 2.0).reshape(&[6, 40, 3]).unwrap();
    let gm = m.to_device(&cuda()).unwrap();
    for ax in [0isize, 1, 2] {
        for desc in [false, true] {
            let (cv, ci) = m.sort(ax, desc).unwrap();
            let (gv, gi) = gm.sort(ax, desc).unwrap();
            exact(&format!("sort axis {ax} desc {desc} values"), &gv, &cv);
            exact(&format!("sort axis {ax} desc {desc} indices"), &gi, &ci);
        }
    }
    let ints = Tensor::from_slice(&(0..500).map(|i| ((i * 37) % 23) as i64 - 11).collect::<Vec<_>>(), &[5, 100]).unwrap();
    let (ci, cx) = ints.sort(1, true).unwrap();
    let (gi, gx) = ints.to_device(&cuda()).unwrap().sort(1, true).unwrap();
    exact("sort i64 values", &gi, &ci);
    exact("sort i64 indices", &gx, &cx);

    // cumulative sums over the last axis, a middle axis and a long single row; gradients flow back through them
    same_on_both("cumsum last", &[rnd(&[7, 500], 3.0)], |x| x[0].cumsum(1).unwrap().mul(&x[0]).unwrap(), 1e-4);
    same_on_both("cumsum middle", &[rnd(&[4, 90, 5], 4.0)], |x| x[0].cumsum(1).unwrap().square().unwrap(), 1e-4);
    same_on_both("cumsum long row", &[rnd(&[1, 30_000], 5.0)], |x| x[0].cumsum(1).unwrap(), 5e-4);

    // indices that live on the device are validated there: errors match the CPU, the device stays usable
    let data = rnd(&[10, 4], 6.0).to_device(&cuda()).unwrap();
    let bad = Tensor::from_slice(&[1i64, 12, 3], &[3]).unwrap().to_device(&cuda()).unwrap();
    let err = data.index_select(0, &bad).unwrap_err().to_string();
    assert!(err.contains("12") || err.to_lowercase().contains("out of"), "{err}");
    assert!(data.index_select(0, &Tensor::from_slice(&[-1i64, 0], &[2]).unwrap().to_device(&cuda()).unwrap()).is_ok(), "negative indices wrap");
    assert!(data.add_scalar(1.0).unwrap().sum().unwrap().item::<f32>().unwrap().is_finite());
}

#[test]
fn adamw_with_gradient_clipping_trains_identically_on_the_gpu() {
    if !ready() {
        return;
    }
    use oxideon_core::optim::{Adam, Optimizer, clip_grad_norm};
    let x = rnd(&[24, 12], 40.0);
    let target = rnd(&[24, 3], 41.0);
    let run = |dev: &Device| {
        oxideon_core::rng::manual_seed(7);
        let (l1, l2) = (Linear::new(12, 16).unwrap(), Linear::new(16, 3).unwrap());
        let params: Vec<Tensor> = l1.parameters().into_iter().chain(l2.parameters()).collect();
        for p in &params {
            p.retarget(&p.detach().to_device(dev).unwrap()).unwrap();
        }
        let mut opt = Adam::adamw(params.clone(), 0.01, 0.05);
        let (xd, td) = (x.to_device(dev).unwrap(), target.to_device(dev).unwrap());
        let mut losses = Vec::new();
        for _ in 0..30 {
            opt.zero_grad();
            let pred = l2.forward(&l1.forward(&xd).unwrap().tanh().unwrap()).unwrap();
            let loss = pred.sub(&td).unwrap().square().unwrap().mean().unwrap();
            loss.backward().unwrap();
            clip_grad_norm(&params, 0.5).unwrap();
            opt.step().unwrap();
            losses.push(loss.item::<f32>().unwrap());
        }
        (losses, params[0].detach().to_device(&Device::Cpu).unwrap())
    };
    let (lc, wc) = run(&Device::Cpu);
    let before = oxideon_core::backend::cuda_rt::launch_count();
    let (lg, wg) = run(&cuda());
    assert!(oxideon_core::backend::cuda_rt::launch_count() > before + 30 * 4);
    assert!(lc.last().unwrap() < &lc[0]);
    for (i, (a, b)) in lc.iter().zip(&lg).enumerate() {
        assert!((a - b).abs() <= 3e-4 * (1.0 + a.abs()), "loss at step {i}: cpu {a} vs gpu {b}");
    }
    close("AdamW weights", &wg, &wc, 1e-3);
}

#[test]
fn fused_attention_matches_the_composed_path() {
    if !ready() {
        return;
    }
    use oxideon_core::nn::functional_ext::scaled_dot_product_attention as sdpa;
    oxideon_core::backend::cuda_ops::set_fused_attention_always(true);
    let before = oxideon_core::backend::cuda_rt::launch_count();
    // (leading dims, L, S, head dim, value dim)
    for (lead, l, s, d, dv) in [(&[2usize, 3][..], 37usize, 37usize, 32usize, 32usize), (&[1, 4][..], 100, 100, 64, 64), (&[5][..], 1, 130, 16, 24), (&[][..], 50, 70, 128, 8), (&[2, 2][..], 3, 1, 8, 8)] {
        let shape = |last2: [usize; 2]| lead.iter().copied().chain(last2).collect::<Vec<_>>();
        let (q, k, v) = (rnd(&shape([l, d]), 1.0), rnd(&shape([s, d]), 2.0), rnd(&shape([s, dv]), 3.0));
        let (gq, gk, gv) = (q.to_device(&cuda()).unwrap(), k.to_device(&cuda()).unwrap(), v.to_device(&cuda()).unwrap());
        let causal_ok = l == s;
        // no mask, causal, a boolean mask, and an additive float mask broadcast over the batch
        let bool_mask = Tensor::from_slice(&(0..l * s).map(|i| (i * 7 + 3) % 5 != 0 || i % s == 0).collect::<Vec<_>>(), &[l, s]).unwrap();
        let add_mask = rnd(&[l, s], 9.0);
        let cases: Vec<(&str, Option<&Tensor>, bool)> = vec![("plain", None, false), ("causal", None, causal_ok), ("bool mask", Some(&bool_mask), false), ("additive mask", Some(&add_mask), false)];
        for (name, mask, causal) in cases {
            if name == "causal" && !causal_ok {
                continue;
            }
            let gmask = mask.map(|m| m.to_device(&cuda()).unwrap());
            let want = sdpa(&q, &k, &v, mask, causal, None).unwrap();
            let got = sdpa(&gq, &gk, &gv, gmask.as_ref(), causal, None).unwrap();
            close(&format!("attention {name} lead {lead:?} L={l} S={s} D={d}"), &got, &want, 2e-4);
        }
    }
    assert!(oxideon_core::backend::cuda_rt::launch_count() > before + 10, "the fused kernel must have run");
    oxideon_core::backend::cuda_ops::set_fused_attention_always(false);
    // gradients are still available: with requires_grad the composed path is used and matches the CPU
    same_on_both("attention with gradients", &[rnd(&[2, 5, 8], 11.0), rnd(&[2, 6, 8], 12.0), rnd(&[2, 6, 8], 13.0)], |x| sdpa(&x[0], &x[1], &x[2], None, false, None).unwrap(), 5e-5);
}

/// A masked-fill on the CUDA device whose GPU kernel declines (broadcast mask) must still return CUDA storage: its result feeds a
/// matmul, which fails with "cuda:0 vs cuda:0" if the fallback hands back a host-backed tensor.
#[test]
fn where_cond_fallback_stays_on_the_cuda_storage() {
    if !ready() {
        return;
    }
    let (l, e) = (64usize, 16usize);
    let keep: Vec<bool> = (0..l * l).map(|ix| ix % l <= ix / l).collect();
    let (q, k, v) = (rnd(&[2, 3, l, e], 1.0), rnd(&[2, 3, l, e], 2.0), rnd(&[2, 3, l, e], 3.0));
    let run = |dev: &Device| {
        let (q, k, v) = (q.to_device(dev).unwrap(), k.to_device(dev).unwrap(), v.to_device(dev).unwrap());
        let keep = Tensor::from_slice_on(&keep, &[l, l], dev).unwrap();
        let scores = q.matmul(&k.transpose(-2, -1).unwrap()).unwrap();
        let masked = Tensor::where_cond(&keep, &scores, &scores.full_like(f64::NEG_INFINITY).unwrap()).unwrap();
        masked.softmax(-1).unwrap().matmul(&v).unwrap()
    };
    close("masked attention", &run(&cuda()), &run(&Device::Cpu), 2e-4);
}
