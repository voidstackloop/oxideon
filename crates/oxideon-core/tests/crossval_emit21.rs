//! Cross-validation batch 21: spatial ops and misc layers vs PyTorch (tools/crossval/check_batch21.py).
use oxideon_core::nn::conv_ext::{max_pool2d_with_indices, max_unpool2d};
use oxideon_core::nn::loss::cosine_similarity;
use oxideon_core::nn::*;
use oxideon_core::tensor::Tensor;
use oxideon_core::vision_ops::{affine_grid, grid_sample};

fn u(i: u64) -> f64 {
    let mut z = i.wrapping_add(0x9E37_79B9_7F4A_7C15).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 30)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// Deterministic tensor with values in (-2, 2).
fn data(seed: u64, shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    let v: Vec<f64> = (0..n as u64).map(|i| (u(seed * 100_000 + i) - 0.5) * 4.0).collect();
    Tensor::from_slice(&v, shape).unwrap()
}

fn dump(name: &str, t: &Tensor) {
    let v = t.to_dtype(oxideon_core::dtype::DType::F64).unwrap().to_vec::<f64>().unwrap();
    println!("{name} {}", v.iter().map(|x| format!("{x:.15e}")).collect::<Vec<_>>().join(" "));
}

#[test]
#[ignore]
fn emit() {
    // ---- cosine similarity / pairwise distance / normalize / softmin
    let (a, b) = (data(1, &[5, 6]), data(2, &[5, 6]));
    // a zero row exercises the eps clamp
    let mut av = a.to_vec::<f64>().unwrap();
    av[6..12].iter_mut().for_each(|x| *x = 0.0);
    let a = Tensor::from_slice(&av, &[5, 6]).unwrap();
    dump("cos_axis1", &cosine_similarity(&a, &b, 1, 1e-8).unwrap());
    dump("cos_axis0", &cosine_similarity(&a, &b, 0, 1e-8).unwrap());
    for (p, tag) in [(2.0, "p2"), (1.0, "p1"), (3.0, "p3")] {
        dump(&format!("pdist_{tag}"), &a.pairwise_distance(&b, p, 1e-6, false).unwrap());
        dump(&format!("pdist_keep_{tag}"), &a.pairwise_distance(&b, p, 1e-6, true).unwrap());
        dump(&format!("normalize_{tag}"), &a.normalize(p, 1, 1e-12).unwrap());
    }
    dump("normalize_ax0", &a.normalize(2.0, 0, 1e-12).unwrap());
    dump("softmin1", &b.softmin(1).unwrap());

    // ---- unfold / fold (im2col)
    let x = data(3, &[2, 3, 7, 6]);
    for (tag, k, s, p, d) in [("u_a", (3, 2), (1, 1), (0, 0), (1, 1)), ("u_b", (3, 3), (2, 1), (1, 2), (1, 1)), ("u_c", (2, 2), (1, 2), (1, 0), (2, 1))] {
        let cols = x.unfold2d(k, s, p, d).unwrap();
        println!("{tag}_shape {:?}", cols.shape());
        dump(tag, &cols);
        let folded = cols.fold2d((7, 6), k, s, p, d).unwrap();
        dump(&format!("{tag}_fold"), &folded);
    }

    // ---- pooling
    let (vals, idx) = max_pool2d_with_indices(&x, (3, 2), (2, 2), (1, 1)).unwrap();
    dump("mp_vals", &vals);
    dump("mp_idx", &idx.to_dtype(oxideon_core::dtype::DType::F64).unwrap());
    dump("unpool", &max_unpool2d(&vals, &idx, (7, 6)).unwrap());
    dump("avgpool_pad", &x.avg_pool2d((3, 3), (2, 2), (1, 1)).unwrap());
    dump("lppool2", &x.abs().unwrap().lp_pool2d(2.0, (2, 3), (2, 1)).unwrap());
    dump("lppool15", &x.abs().unwrap().lp_pool2d(1.5, (3, 3), (2, 2)).unwrap());
    for (h, w) in [(1, 1), (3, 4), (4, 5), (7, 6), (5, 3)] {
        dump(&format!("adamax_{h}x{w}"), &x.adaptive_max_pool2d((h, w)).unwrap());
        dump(&format!("adaavg_{h}x{w}"), &x.adaptive_avg_pool2d((h, w)).unwrap());
    }

    // ---- affine_grid / grid_sample (values and gradients)
    for (tag, align) in [("gs_t", true), ("gs_f", false)] {
        let theta = Tensor::from_slice(&[0.9f64, 0.2, 0.1, -0.15, 1.1, -0.05, 0.7, -0.3, 0.2, 0.25, 0.8, 0.1], &[2, 2, 3]).unwrap().requires_grad_(true);
        let inp = data(4, &[2, 3, 5, 6]).requires_grad_(true);
        let grid = affine_grid(&theta, [2, 3, 4, 5], align).unwrap();
        dump(&format!("{tag}_grid"), &grid);
        let out = grid_sample(&inp, &grid, align).unwrap();
        dump(&format!("{tag}_out"), &out);
        let w = data(5, out.shape());
        out.mul(&w).unwrap().sum().unwrap().backward().unwrap();
        dump(&format!("{tag}_dinput"), &inp.grad().unwrap());
        dump(&format!("{tag}_dtheta"), &theta.grad().unwrap());
    }

    // ---- embedding bag
    let eb = |mode: EmbeddingBagMode| {
        let e = EmbeddingBag::new(10, 4, mode).unwrap();
        e.weight.assign(&data(6, &[10, 4])).unwrap();
        e
    };
    let idx = Tensor::from_slice(&[1i64, 4, 4, 9, 0, 2, 7, 7, 3], &[9]).unwrap();
    let offsets = [0usize, 3, 4, 8];
    for (tag, mode) in [("sum", EmbeddingBagMode::Sum), ("mean", EmbeddingBagMode::Mean), ("max", EmbeddingBagMode::Max)] {
        dump(&format!("ebag_{tag}"), &eb(mode).forward_offsets(&idx, &offsets).unwrap());
    }

    // ---- bilinear layer
    let bl = Bilinear::new(3, 4, 2, true).unwrap();
    let f32d = |s: u64, sh: &[usize]| data(s, sh).to_dtype(oxideon_core::dtype::DType::F32).unwrap();
    bl.weight.assign(&f32d(7, &[2, 3, 4])).unwrap();
    bl.bias.as_ref().unwrap().assign(&f32d(8, &[2])).unwrap();
    dump("bilinear", &bl.forward_pair(&f32d(9, &[5, 3]), &f32d(10, &[5, 4])).unwrap());

    // ---- pixel shuffle / unshuffle
    let ps = PixelShuffle::new(2);
    dump("pixshuf", &ps.forward(&data(11, &[2, 8, 3, 4])).unwrap());
    let pu = PixelUnshuffle::new(2);
    dump("pixunshuf", &pu.forward(&data(12, &[2, 3, 6, 4])).unwrap());
}
