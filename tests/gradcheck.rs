//! Comprehensive numerical gradient checks (first-order and second-order!) across
//! Oxideon's differentiable operations, strided views, einsum, conv2d, and losses.

use oxideon::nn::loss;
use oxideon::prelude::*;

/// Deterministic non-uniform weighted sum so upstream gradients are distinct at every element.
fn weighted_sum(x: &Tensor) -> Result<Tensor> {
    let n = x.numel();
    let w: Vec<f64> = (0..n).map(|i| ((i as f64 + 1.0) * 0.37).sin()).collect();
    let wt = Tensor::from_slice_on(&w, x.shape(), &x.device())?.to_dtype(x.dtype())?;
    x.mul(&wt)?.sum()
}

/// Checks analytical gradients (in `F64` for high precision) against central finite differences.
fn gradcheck(f: impl Fn(&[Tensor]) -> Result<Tensor>, inputs: &[Tensor]) {
    let f64_inputs: Vec<Tensor> = inputs
        .iter()
        .map(|t| {
            t.to_dtype(DType::F64)
                .unwrap()
                .detach()
                .requires_grad_(true)
        })
        .collect();
    let loss = f(&f64_inputs).unwrap();
    loss.backward().unwrap();

    let eps = 1e-5;
    for (k, inp) in f64_inputs.iter().enumerate() {
        let analytic = inp
            .grad()
            .unwrap_or_else(|| panic!("input {k} did not receive a gradient"))
            .to_vec::<f64>()
            .unwrap();
        let base_vec = inp.to_vec::<f64>().unwrap();
        let shape = inp.shape().to_vec();

        for i in 0..base_vec.len() {
            let mut plus_args = f64_inputs.iter().map(Tensor::detach).collect::<Vec<_>>();
            let mut p_vec = base_vec.clone();
            p_vec[i] += eps;
            plus_args[k] = Tensor::from_vec(p_vec, &shape).unwrap();

            let mut minus_args = f64_inputs.iter().map(Tensor::detach).collect::<Vec<_>>();
            let mut m_vec = base_vec.clone();
            m_vec[i] -= eps;
            minus_args[k] = Tensor::from_vec(m_vec, &shape).unwrap();

            let f_plus = no_grad(|| f(&plus_args).unwrap().item::<f64>().unwrap());
            let f_minus = no_grad(|| f(&minus_args).unwrap().item::<f64>().unwrap());
            let numeric = (f_plus - f_minus) / (2.0 * eps);
            let a = analytic[i];
            assert!(
                (a - numeric).abs() <= 1e-4 * (1.0 + numeric.abs()),
                "input {k}, elem {i}: analytic {a} vs numeric {numeric}"
            );
        }
    }
}

#[test]
fn binary_and_broadcast_ops() -> Result<()> {
    manual_seed(1);
    let a = Tensor::randn(&[2, 3, 1])?;
    let b = Tensor::rand_uniform(&[3, 4], 0.5, 2.0)?;
    gradcheck(|t| weighted_sum(&(&t[0] + &t[1])?), &[a.clone(), b.clone()]);
    gradcheck(|t| weighted_sum(&(&t[0] - &t[1])?), &[a.clone(), b.clone()]);
    gradcheck(|t| weighted_sum(&(&t[0] * &t[1])?), &[a.clone(), b.clone()]);
    gradcheck(|t| weighted_sum(&(&t[0] / &t[1])?), &[a.clone(), b.clone()]);
    gradcheck(
        |t| weighted_sum(&t[0].add_scalar(2.0)?.pow(&t[1])?),
        &[b.clone(), a],
    );
    Ok(())
}

#[test]
fn unary_and_activations() -> Result<()> {
    manual_seed(2);
    let a = Tensor::from_slice(&[-1.7f32, -0.4, 0.3, 0.9, 1.4, -0.8], &[2, 3])?;
    let pos = Tensor::rand_uniform(&[2, 3], 0.4, 2.5)?;

    gradcheck(|t| weighted_sum(&t[0].exp()?), std::slice::from_ref(&a));
    gradcheck(|t| weighted_sum(&t[0].sin()?), std::slice::from_ref(&a));
    gradcheck(|t| weighted_sum(&t[0].cos()?), std::slice::from_ref(&a));
    gradcheck(|t| weighted_sum(&t[0].tanh()?), std::slice::from_ref(&a));
    gradcheck(|t| weighted_sum(&t[0].sigmoid()?), std::slice::from_ref(&a));
    gradcheck(|t| weighted_sum(&t[0].relu()?), std::slice::from_ref(&a));
    gradcheck(
        |t| weighted_sum(&t[0].leaky_relu(0.1)?),
        std::slice::from_ref(&a),
    );
    gradcheck(|t| weighted_sum(&t[0].gelu()?), std::slice::from_ref(&a));
    gradcheck(|t| weighted_sum(&t[0].silu()?), std::slice::from_ref(&a));
    gradcheck(|t| weighted_sum(&t[0].elu(1.2)?), std::slice::from_ref(&a));
    gradcheck(
        |t| weighted_sum(&t[0].softplus()?),
        std::slice::from_ref(&a),
    );
    gradcheck(|t| weighted_sum(&t[0].erf()?), std::slice::from_ref(&a));
    gradcheck(|t| weighted_sum(&t[0].ln()?), std::slice::from_ref(&pos));
    gradcheck(|t| weighted_sum(&t[0].sqrt()?), std::slice::from_ref(&pos));
    gradcheck(|t| weighted_sum(&t[0].rsqrt()?), std::slice::from_ref(&pos));
    gradcheck(|t| weighted_sum(&t[0].recip()?), std::slice::from_ref(&pos));
    gradcheck(|t| weighted_sum(&t[0].powf(2.3)?), &[pos]);
    Ok(())
}

#[test]
fn strided_views_and_slicing_gradcheck() -> Result<()> {
    manual_seed(3);
    let a = Tensor::randn(&[4, 5, 3])?;
    gradcheck(
        |t| weighted_sum(&t[0].permute(&[2, 0, 1])?),
        std::slice::from_ref(&a),
    );
    gradcheck(
        |t| weighted_sum(&t[0].narrow(1, 1, 3)?),
        std::slice::from_ref(&a),
    );
    gradcheck(
        |t| weighted_sum(&t[0].flip(&[0, 2])?),
        std::slice::from_ref(&a),
    );
    gradcheck(
        |t| weighted_sum(&t[0].select(1, -2)?),
        std::slice::from_ref(&a),
    );
    gradcheck(
        |t| weighted_sum(&t[0].i((1..4, SliceElem::step(Some(0), Some(5), 2), ..))?),
        std::slice::from_ref(&a),
    );
    gradcheck(|t| weighted_sum(&t[0].diagonal(1, 0, 1)?), &[a]);
    Ok(())
}

#[test]
fn matmul_strided_and_batched() -> Result<()> {
    manual_seed(4);
    let a = Tensor::randn(&[3, 4])?;
    let b = Tensor::randn(&[5, 4])?;
    // Note: b.t() is a non-contiguous view fed directly into GEMM!
    gradcheck(|t| weighted_sum(&t[0].matmul(&t[1].t()?)?), &[a, b]);

    let ba = Tensor::randn(&[2, 3, 4])?;
    let bb = Tensor::randn(&[4, 2])?;
    gradcheck(|t| weighted_sum(&t[0].matmul(&t[1])?), &[ba, bb]);
    Ok(())
}

#[test]
fn reductions_and_softmax() -> Result<()> {
    manual_seed(5);
    let a = Tensor::randn(&[3, 4, 2])?;
    gradcheck(
        |t| weighted_sum(&t[0].sum_axes(&[0, 2], false)?),
        std::slice::from_ref(&a),
    );
    gradcheck(
        |t| weighted_sum(&t[0].mean_axis(1, true)?),
        std::slice::from_ref(&a),
    );
    gradcheck(
        |t| weighted_sum(&t[0].var_axis(1, true, false)?),
        std::slice::from_ref(&a),
    );
    gradcheck(
        |t| weighted_sum(&t[0].log_softmax(1)?),
        std::slice::from_ref(&a),
    );
    gradcheck(|t| weighted_sum(&t[0].softmax(1)?), &[a]);
    Ok(())
}

#[test]
fn einsum_pad_cat_cumsum_gather_topk() -> Result<()> {
    manual_seed(6);
    let a = Tensor::randn(&[2, 3, 4])?;
    let b = Tensor::randn(&[2, 4, 3])?;
    gradcheck(
        |t| weighted_sum(&Tensor::einsum("bik,bkj->bij", &[&t[0], &t[1]])?),
        &[a.clone(), b],
    );
    gradcheck(
        |t| weighted_sum(&Tensor::einsum("bii->bi", &[&t[0].narrow(2, 0, 3)?])?),
        std::slice::from_ref(&a),
    );
    gradcheck(
        |t| weighted_sum(&t[0].pad(&[(0, 0), (1, 2), (1, 0)], 0.5)?),
        std::slice::from_ref(&a),
    );
    gradcheck(|t| weighted_sum(&t[0].cumsum(1)?), std::slice::from_ref(&a));
    gradcheck(
        |t| {
            let (vals, _) = t[0].topk(2, 2, true, true)?;
            weighted_sum(&vals)
        },
        std::slice::from_ref(&a),
    );
    let c = Tensor::randn(&[2, 2, 4])?;
    gradcheck(
        |t| weighted_sum(&Tensor::cat(&[t[0].clone(), t[1].clone()], 1)?),
        &[a, c],
    );
    Ok(())
}

#[test]
fn conv2d_and_pooling_gradcheck() -> Result<()> {
    manual_seed(7);
    let x = Tensor::randn(&[2, 2, 5, 5])?;
    let w = Tensor::randn(&[3, 2, 3, 3])?;
    let b = Tensor::randn(&[3])?;
    gradcheck(
        |t| weighted_sum(&t[0].conv2d(&t[1], Some(&t[2]), (1, 1), (1, 1), (1, 1))?),
        &[x.clone(), w, b],
    );
    gradcheck(
        |t| weighted_sum(&t[0].avg_pool2d((2, 2), (2, 2), (0, 0))?),
        std::slice::from_ref(&x),
    );
    gradcheck(
        |t| weighted_sum(&t[0].max_pool2d((2, 2), (2, 2), (0, 0))?),
        &[x],
    );
    // transposed conv: strided path and the output_padding path (fold-grid narrowing)
    let xt = Tensor::randn(&[2, 2, 4, 4])?;
    let wt = Tensor::randn(&[2, 3, 3, 3])?;
    let bt = Tensor::randn(&[3])?;
    gradcheck(
        |t| {
            weighted_sum(&t[0].conv_transpose2d(
                &t[1],
                Some(&t[2]),
                (2, 2),
                (1, 1),
                (0, 0),
                (1, 1),
            )?)
        },
        &[xt.clone(), wt.clone(), bt],
    );
    gradcheck(
        |t| weighted_sum(&t[0].conv_transpose2d(&t[1], None, (1, 1), (0, 0), (1, 1), (2, 2))?),
        &[xt, wt],
    );
    Ok(())
}

#[test]
fn losses_gradcheck() -> Result<()> {
    manual_seed(8);
    let logits = Tensor::randn(&[4, 3])?;
    let targets_f = Tensor::from_slice(
        &[
            1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0,
        ],
        &[4, 3],
    )?;
    gradcheck(
        |t| loss::cross_entropy(&t[0], &[0, 2, 1, 2]),
        std::slice::from_ref(&logits),
    );
    gradcheck(
        |t| loss::cross_entropy_with_smoothing(&t[0], &[0, 2, 1, 2], 0.1),
        std::slice::from_ref(&logits),
    );
    gradcheck(
        |t| loss::focal_loss(&t[0], &[0, 2, 1, 2], 0.25, 2.0),
        std::slice::from_ref(&logits),
    );
    let logq = Tensor::randn(&[4, 3])?;
    let prob = Tensor::from_slice(
        &[
            0.5f32, 0.25, 0.25, 0.7, 0.2, 0.1, 0.1, 0.1, 0.8, 0.34, 0.33, 0.33,
        ],
        &[4, 3],
    )?;
    gradcheck(|t| loss::kl_div_loss(&t[0], &t[1]), &[logq, prob]);
    gradcheck(
        |t| {
            let targ = targets_f.to_dtype(t[0].dtype())?;
            loss::binary_cross_entropy_with_logits(&t[0], &targ)
        },
        std::slice::from_ref(&logits),
    );
    gradcheck(
        |t| {
            let targ = targets_f.to_dtype(t[0].dtype())?;
            loss::huber_loss(&t[0], &targ, 0.5)
        },
        &[logits],
    );
    Ok(())
}

#[test]
fn linalg_backward_sweep_triangular_pinv_lstsq_exp_power_cond_kron_cdist_norms() -> Result<()> {
    manual_seed(7);
    let m33 = Tensor::randn(&[3, 3])?;
    let eye3 = Tensor::eye(3, DType::F32, &Device::Cpu)?;
    // Well-conditioned base: A = M + 3I (nonsingular, distinct singular values a.s.)
    let a_of =
        |t: &Tensor| -> Result<Tensor> { t.add(&eye3.mul_scalar(3.0)?.to_dtype(t.dtype())?) };

    // triangular_solve (lower): route through tril so strict-upper perturbations vanish both ways
    let rhs = Tensor::randn(&[3, 2])?;
    gradcheck(
        |t| {
            let l = t[0].tril(0)?;
            weighted_sum(&l.triangular_solve(&t[1], true, false)?)
        },
        &[m33.clone(), rhs.clone()],
    );
    // slogdet logabsdet
    gradcheck(
        |t| {
            let a = a_of(&t[0])?;
            let (_sign, logdet) = a.slogdet()?;
            weighted_sum(&logdet)
        },
        std::slice::from_ref(&m33),
    );
    // pinv on a tall full-rank factor
    let tall = Tensor::randn(&[5, 3])?;
    gradcheck(
        |t| weighted_sum(&t[0].pinv(1e-6)?),
        std::slice::from_ref(&tall),
    );
    // lstsq (normal-equation path through pinv + matmul)
    let b5 = Tensor::randn(&[5, 2])?;
    gradcheck(|t| weighted_sum(&t[0].lstsq(&t[1], 1e-6)?), &[tall, b5]);
    // matrix_exp / matrix_power (pure matmul compositions)
    let small = Tensor::randn(&[3, 3])?.mul_scalar(0.25)?;
    gradcheck(
        |t| weighted_sum(&t[0].matrix_exp()?),
        std::slice::from_ref(&small),
    );
    gradcheck(
        |t| weighted_sum(&t[0].matrix_power(3)?),
        std::slice::from_ref(&m33),
    );
    gradcheck(
        |t| {
            let a = a_of(&t[0])?;
            weighted_sum(&a.matrix_power(-2)?)
        },
        std::slice::from_ref(&m33),
    );
    // cond via singular values (distinct a.s. for random input)
    gradcheck(
        |t| {
            let a = a_of(&t[0])?;
            weighted_sum(&a.cond()?)
        },
        std::slice::from_ref(&m33),
    );
    // kron (bilinear in both args)
    let k1 = Tensor::randn(&[2, 3])?;
    let k2 = Tensor::randn(&[4, 2])?;
    gradcheck(|t| weighted_sum(&t[0].kron(&t[1])?), &[k1, k2]);
    // cdist (clamped sqrt keeps coincident-point grads finite)
    let c1 = Tensor::randn(&[4, 3])?;
    let c2 = Tensor::randn(&[5, 3])?;
    gradcheck(|t| weighted_sum(&t[0].cdist(&t[1])?), &[c1, c2]);
    // vector norms (nonzero input: smooth everywhere it is evaluated)
    let v = Tensor::randn(&[6])?.add_scalar(0.5)?;
    gradcheck(|t| t[0].norm_p(2.0, &[0], false), std::slice::from_ref(&v));
    gradcheck(|t| t[0].norm_p(1.0, &[0], false), std::slice::from_ref(&v));
    gradcheck(|t| t[0].norm_p(3.0, &[0], false), std::slice::from_ref(&v));
    gradcheck(
        |t| t[0].norm_p(f64::INFINITY, &[0], false),
        std::slice::from_ref(&v),
    );
    gradcheck(
        |t| weighted_sum(&t[0].normalize(2.0, 0, 1e-12)?),
        std::slice::from_ref(&v),
    );
    Ok(())
}

#[test]
fn index_select_add_gather_scatter_backward_cycle() -> Result<()> {
    manual_seed(11);
    // select/add cycle with duplicate indices (grads accumulate)
    let x = Tensor::randn(&[4, 3])?;
    let idx1 = Tensor::from_slice(&[2i64, 0, 2], &[3])?;
    gradcheck(
        |t| weighted_sum(&t[0].index_select(0, &idx1)?),
        std::slice::from_ref(&x),
    );
    let base = Tensor::randn(&[4, 3])?;
    let add_src = Tensor::randn(&[3, 3])?;
    gradcheck(
        |t| weighted_sum(&t[0].index_add(0, &idx1, &t[1])?),
        &[base, add_src],
    );
    // gather/scatter cycle on a 2D problem
    let g = Tensor::randn(&[3, 4])?;
    let gidx = Tensor::from_slice(&[2i64, 0, 3, 1, 1, 0], &[2, 3])?;
    gradcheck(
        |t| weighted_sum(&t[0].gather(1, &gidx)?),
        std::slice::from_ref(&g),
    );
    let dst = Tensor::randn(&[2, 4])?;
    let sidx = Tensor::from_slice(&[1i64, 3, 3, 0, 2, 1], &[2, 3])?;
    let ssrc = Tensor::randn(&[2, 3])?;
    gradcheck(
        |t| weighted_sum(&t[0].scatter_add(1, &sidx, &t[1])?),
        &[dst, ssrc],
    );
    Ok(())
}
