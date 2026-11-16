//! Finite-difference gradient checks over core differentiable ops (activations, reductions, shape ops, conv/pool, norms,
//! attention, indexing). Inputs avoid kinks (ties, zero) so central differences are valid.

use oxideon_core::autograd_utils::gradcheck;
use oxideon_core::nn::functional_ext::{group_norm, layer_norm, scaled_dot_product_attention};
use oxideon_core::Tensor;

fn r(n: usize, shape: &[usize], seed: f64) -> Tensor {
    let v: Vec<f64> = (0..n).map(|i| ((i as f64 + seed) * 0.913).sin() * 1.3 + 0.17 * (i as f64 % 3.0) + 0.05).collect();
    Tensor::from_slice(&v, shape).unwrap()
}
fn pos(n: usize, shape: &[usize]) -> Tensor {
    r(n, shape, 1.0).abs().unwrap().add_scalar(0.5).unwrap()
}

macro_rules! check {
    ($name:ident, $inputs:expr, $f:expr) => {
        #[test]
        fn $name() {
            let r = gradcheck($f, &$inputs, 1e-6, 1e-5, 1e-4);
            assert!(r.is_ok(), "{}: {:?}", stringify!($name), r.err());
        }
    };
}

check!(gelu, [r(6, &[6], 0.0)], |x| x[0].gelu());
check!(silu, [r(6, &[6], 0.0)], |x| x[0].silu());
check!(elu, [r(6, &[6], 0.0)], |x| x[0].elu(1.0));
check!(softplus, [r(6, &[6], 0.0)], |x| x[0].softplus());
check!(tanh, [r(6, &[6], 0.0)], |x| x[0].tanh());
check!(sigmoid, [r(6, &[6], 0.0)], |x| x[0].sigmoid());
check!(erf, [r(6, &[6], 0.0)], |x| x[0].erf());
check!(sqrt_rsqrt, [pos(6, &[6])], |x| x[0].sqrt()?.add(&x[0].rsqrt()?));
check!(asin_atan, [r(6, &[6], 0.0).mul_scalar(0.4).unwrap()], |x| x[0].asin()?.add(&x[0].atan()?));
check!(sinh_expm1_log1p, [pos(6, &[6])], |x| x[0].sinh()?.add(&x[0].expm1()?)?.add(&x[0].log1p()?));
check!(pow_tensor, [pos(6, &[6]), r(6, &[6], 3.0)], |x| x[0].pow(&x[1]));
check!(atan2, [r(6, &[6], 0.0), pos(6, &[6])], |x| x[0].atan2(&x[1]));
check!(maximum, [r(6, &[6], 0.0), r(6, &[6], 9.0)], |x| x[0].maximum(&x[1]));
check!(softmax, [r(12, &[3, 4], 0.0)], |x| x[0].softmax(-1));
check!(log_softmax, [r(12, &[3, 4], 0.0)], |x| x[0].log_softmax(0));
check!(logsumexp, [r(12, &[3, 4], 0.0)], |x| x[0].logsumexp(1, false));
check!(var_std, [r(12, &[3, 4], 0.0)], |x| x[0].var_axis(1, true, false)?.add(&x[0].std_axis(1, false, false)?));
check!(prod, [pos(5, &[5])], |x| x[0].prod());
check!(cumprod, [pos(6, &[2, 3])], |x| x[0].cumprod(1));
check!(cumsum, [r(6, &[2, 3], 0.0)], |x| x[0].cumsum(0));
check!(cummax_values, [r(6, &[6], 0.0)], |x| Ok(x[0].cummax(0)?.0));
check!(max_axis, [r(12, &[3, 4], 0.0)], |x| x[0].max_axis(1, false));
check!(sort_values, [r(6, &[6], 0.0)], |x| Ok(x[0].sort(0, true)?.0));
check!(topk_values, [r(8, &[2, 4], 0.0)], |x| Ok(x[0].topk(2, 1, true, true)?.0));
check!(gather, [r(12, &[3, 4], 0.0)], |x| {
    let idx = Tensor::from_slice(&[0i64, 3, 1, 2, 2, 0], &[3, 2]).unwrap();
    x[0].gather(1, &idx)
});
check!(index_select, [r(12, &[3, 4], 0.0)], |x| {
    let idx = Tensor::from_slice(&[2i64, 0, 2], &[3]).unwrap();
    x[0].index_select(0, &idx)
});
check!(scatter_src, [r(6, &[2, 3], 0.0), r(4, &[2, 2], 5.0)], |x| {
    let idx = Tensor::from_slice(&[0i64, 2, 1, 0], &[2, 2]).unwrap();
    x[0].scatter(1, &idx, &x[1])
});
check!(where_cond, [r(6, &[6], 0.0), r(6, &[6], 4.0)], |x| {
    let c = Tensor::from_slice(&[true, false, true, true, false, false], &[6]).unwrap();
    Tensor::where_cond(&c, &x[0], &x[1])
});
check!(clamp, [r(6, &[6], 0.0).mul_scalar(2.0).unwrap()], |x| x[0].clamp(-1.1, 0.9));
check!(shape_ops, [r(12, &[3, 4], 0.0)], |x| {
    x[0].narrow(1, 1, 2)?.flip(&[0])?.roll(1, 1)?.pad(&[(1, 0), (0, 2)], 0.5)?.repeat(&[1, 2])
});
check!(expand_sum, [r(4, &[4, 1], 0.0)], |x| x[0].expand(&[4, 3])?.sum_axes(&[0], false));
check!(tril_triu_trace, [r(9, &[3, 3], 0.0)], |x| x[0].tril(0)?.add(&x[0].triu(1)?)?.mul_scalar(1.0)?.add_scalar(0.0)?.add(&x[0].trace()?.unsqueeze(0)?));
check!(einsum, [r(6, &[2, 3], 0.0), r(12, &[3, 4], 2.0)], |x| Tensor::einsum("ij,jk->ik", &[&x[0], &x[1]]));
check!(matmul_batched, [r(12, &[2, 2, 3], 0.0), r(6, &[3, 2], 2.0)], |x| x[0].matmul(&x[1]));
check!(conv1d, [r(16, &[1, 2, 8], 0.0), r(12, &[2, 2, 3], 3.0), r(2, &[2], 4.0)], |x| x[0].conv1d(&x[1], Some(&x[2]), 1, 1, 1));
check!(conv2d, [r(50, &[1, 2, 5, 5], 0.0), r(36, &[2, 2, 3, 3], 3.0), r(2, &[2], 4.0)], |x| x[0].conv2d(&x[1], Some(&x[2]), (2, 1), (1, 0), (1, 1)));
check!(max_pool2d, [r(36, &[1, 1, 6, 6], 0.0)], |x| x[0].max_pool2d((2, 2), (2, 2), (0, 0)));
check!(avg_pool2d, [r(36, &[1, 1, 6, 6], 0.0)], |x| x[0].avg_pool2d((3, 3), (1, 1), (1, 1)));
check!(layer_norm_all, [r(12, &[3, 4], 0.0), r(4, &[4], 1.0), r(4, &[4], 2.0)], |x| layer_norm(&x[0], &[4], Some(&x[1]), Some(&x[2]), 1e-5));
check!(group_norm_all, [r(32, &[2, 4, 2, 2], 0.0), r(4, &[4], 1.0), r(4, &[4], 2.0)], |x| group_norm(&x[0], 2, Some(&x[1]), Some(&x[2]), 1e-5));
check!(attention, [r(12, &[1, 3, 4], 0.0), r(12, &[1, 3, 4], 1.0), r(12, &[1, 3, 4], 2.0)], |x| scaled_dot_product_attention(&x[0], &x[1], &x[2], None, false, None));
check!(attention_causal, [r(12, &[1, 3, 4], 0.0), r(12, &[1, 3, 4], 1.0), r(12, &[1, 3, 4], 2.0)], |x| scaled_dot_product_attention(&x[0], &x[1], &x[2], None, true, None));
