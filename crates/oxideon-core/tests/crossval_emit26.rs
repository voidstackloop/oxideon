//! Cross-validation batch 26: values and gradients of indexing / selection / shape ops vs PyTorch (float64)
//! (tools/crossval/check_batch26.py).
use oxideon_core::nn::functional_ext::{InterpMode, interpolate};
use oxideon_core::tensor::Tensor;

fn base(rows: usize, cols: usize, salt: f64) -> Vec<f64> {
    (0..rows * cols).map(|k| (0.77 * k as f64 + salt).sin() * 1.5).collect()
}

fn wts(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_slice(&(0..n).map(|k| (1.3 * k as f64 + 0.4).sin()).collect::<Vec<_>>(), shape).unwrap()
}

fn leaf(v: &[f64], shape: &[usize]) -> Tensor {
    Tensor::from_slice(v, shape).unwrap().requires_grad_(true)
}

fn dump(name: &str, t: &Tensor) {
    let v = t.to_dtype(oxideon_core::dtype::DType::F64).unwrap().to_vec::<f64>().unwrap();
    println!("{name} {}", v.iter().map(|x| format!("{x:.14e}")).collect::<Vec<_>>().join(" "));
}

fn case(name: &str, inputs: &[&Tensor], outs: &[Tensor]) {
    let mut loss: Option<Tensor> = None;
    for (k, o) in outs.iter().enumerate() {
        dump(&format!("{name}_out{k}"), o);
        let term = o.mul(&wts(o.shape())).unwrap().sum().unwrap();
        loss = Some(match loss {
            Some(l) => l.add(&term).unwrap(),
            None => term,
        });
    }
    loss.unwrap().backward().unwrap();
    for (k, x) in inputs.iter().enumerate() {
        match x.grad() {
            Some(g) => dump(&format!("{name}_g{k}"), &g),
            None => println!("{name}_g{k} NOGRAD"),
        }
    }
}

fn idx(v: &[i64], shape: &[usize]) -> Tensor {
    Tensor::from_slice(v, shape).unwrap()
}

#[test]
#[ignore]
fn emit_indexing() {
    let mut xv = base(4, 5, 0.3);
    // ties for the max/min/median tests
    xv[1] = 2.5;
    xv[3] = 2.5;
    xv[10] = -2.0;
    xv[12] = -2.0;
    let yv = base(4, 5, 1.7);
    let xt = base(4, 5, 0.3); // tie-free copy for order-statistic ops
    macro_rules! with {
        ($name:literal, [$(($x:ident, $v:expr, $shape:expr)),*], $body:expr) => {{
            $( let $x = leaf(&$v, $shape); )*
            let outs: Vec<Tensor> = $body;
            case($name, &[$(&$x),*], &outs);
        }};
    }
    with!("gather", [(x, xv, &[4, 5])], vec![x.gather(1, &idx(&[0, 4, 2, 1, 3, 3, 0, 0, 2, 2, 4, 1], &[4, 3])).unwrap()]);
    with!("index_select", [(x, xv, &[4, 5])], vec![x.index_select(0, &idx(&[2, 0, 3, 0], &[4])).unwrap()]);
    let mask = Tensor::from_slice(&(0..20).map(|k| u8::from(k % 3 == 0)).collect::<Vec<_>>(), &[4, 5]).unwrap().to_dtype(oxideon_core::dtype::DType::Bool).unwrap();
    with!("masked_fill", [(x, xv, &[4, 5])], vec![x.masked_fill(&mask, 7.0).unwrap()]);
    with!("where", [(x, xv, &[4, 5]), (y, yv, &[4, 5])], vec![Tensor::where_cond(&mask, &x, &y).unwrap()]);
    with!("masked_select", [(x, xv, &[4, 5])], vec![x.masked_select(&mask).unwrap()]);
    with!("scatter", [(x, xv, &[4, 5]), (s, base(4, 3, 2.2), &[4, 3])], vec![x.scatter(1, &idx(&[0, 4, 2, 1, 3, 0, 4, 0, 2, 1, 2, 3], &[4, 3]), &s).unwrap()]);
    with!("scatter_add", [(x, xv, &[4, 5]), (s, base(4, 3, 2.2), &[4, 3])], vec![x.scatter_add(1, &idx(&[0, 0, 2, 1, 3, 3, 4, 0, 2, 1, 1, 1], &[4, 3]), &s).unwrap()]);
    with!("index_add", [(x, xv, &[4, 5]), (s, base(3, 5, 3.3), &[3, 5])], vec![x.index_add(0, &idx(&[1, 3, 1], &[3]), &s).unwrap()]);
    with!("index_copy", [(x, xv, &[4, 5]), (s, base(2, 5, 3.9), &[2, 5])], vec![x.index_copy(0, &idx(&[3, 0], &[2]), &s).unwrap()]);
    with!("cat0", [(x, xv, &[4, 5]), (y, yv, &[4, 5])], vec![Tensor::cat(&[x.clone(), y.clone()], 0).unwrap()]);
    with!("cat1", [(x, xv, &[4, 5]), (y, yv, &[4, 5])], vec![Tensor::cat(&[x.clone(), y.clone()], 1).unwrap()]);
    with!("stack", [(x, xv, &[4, 5]), (y, yv, &[4, 5])], vec![Tensor::stack(&[x.clone(), y.clone()], 1).unwrap()]);
    with!("repeat", [(x, xv, &[4, 5])], vec![x.repeat(&[2, 3]).unwrap()]);
    with!("tile", [(x, xv, &[4, 5])], vec![x.tile(&[2, 1, 2]).unwrap()]);
    with!("repeat_interleave", [(x, xv, &[4, 5])], vec![x.repeat_interleave(3, 1).unwrap()]);
    with!("tril", [(x, xv, &[4, 5])], vec![x.tril(1).unwrap()]);
    with!("triu", [(x, xv, &[4, 5])], vec![x.triu(-1).unwrap()]);
    with!("diag_embed", [(x, xv, &[4, 5])], vec![x.diag_embed().unwrap()]);
    with!("diagonal", [(x, xv, &[4, 5])], vec![x.diagonal(1, 0, 1).unwrap()]);
    with!("sort_desc", [(x, xt, &[4, 5])], vec![x.sort(1, true).unwrap().0]);
    with!("sort_asc0", [(x, xt, &[4, 5])], vec![x.sort(0, false).unwrap().0]);
    with!("topk", [(x, xt, &[4, 5])], vec![x.topk(3, 1, true, true).unwrap().0]);
    with!("topk_small", [(x, xt, &[4, 5])], vec![x.topk(2, 0, false, true).unwrap().0]);
    with!("cummax", [(x, xt, &[4, 5])], vec![x.cummax(1).unwrap().0]);
    with!("cummin", [(x, xt, &[4, 5])], vec![x.cummin(0).unwrap().0]);
    with!("kthvalue", [(x, xt, &[4, 5])], vec![x.kthvalue(2, 1, false).unwrap().0]);
    with!("median", [(x, xt, &[4, 5])], vec![x.median().unwrap().reshape(&[1]).unwrap()]);
    with!("max_axis_ties", [(x, xv, &[4, 5])], vec![x.max_axis(1, false).unwrap()]);
    with!("min_axis_ties", [(x, xv, &[4, 5])], vec![x.min_axis(1, true).unwrap()]);
    with!("amax_ties", [(x, xv, &[4, 5])], vec![x.amax(&[1], false).unwrap()]);
    with!("amin_ties", [(x, xv, &[4, 5])], vec![x.amin(&[1], false).unwrap()]);
    with!("narrow", [(x, xv, &[4, 5])], vec![x.narrow(1, 1, 3).unwrap()]);
    with!("expand", [(x, base(1, 5, 0.9), &[1, 5])], vec![x.expand(&[4, 5]).unwrap()]);
    with!("take", [(x, xv, &[4, 5])], vec![x.take(&idx(&[19, 0, 7, 7, 3], &[5])).unwrap()]);
    with!("clamp", [(x, xv, &[4, 5])], vec![x.clamp(-1.0, 1.2).unwrap()]);
    with!("clamp_edge", [(x, xv, &[4, 5])], vec![x.clamp(2.5, 2.5).unwrap()]);
    with!("clamp_min", [(x, xv, &[4, 5])], vec![x.clamp_min(0.3).unwrap()]);
    with!("permute", [(x, base(6, 4, 0.1), &[2, 3, 4])], vec![x.permute(&[2, 0, 1]).unwrap()]);
    with!("transpose", [(x, xv, &[4, 5])], vec![x.transpose(0, 1).unwrap()]);
    with!("movedim", [(x, base(6, 4, 0.1), &[2, 3, 4])], vec![x.movedim(0, 2).unwrap()]);
    with!("chunk", [(x, xv, &[4, 5])], x.chunk(3, 1).unwrap());
    with!("split", [(x, xv, &[4, 5])], x.split(2, 0).unwrap());
    with!("std_axis", [(x, xv, &[4, 5])], vec![x.std_axis(1, true, false).unwrap()]);
    with!("var_axis0", [(x, xv, &[4, 5])], vec![x.var_axis(0, false, true).unwrap()]);
    with!("pad", [(x, xv, &[4, 5])], vec![x.pad(&[(1, 2), (0, 3)], 0.5).unwrap()]);
    with!("unfold_dim", [(x, xv, &[4, 5])], vec![x.unfold(1, 3, 1).unwrap()]);
    // nan-aware reductions with a NaN element in the data (gradient of the NaN position must be zero)
    let mut xn = xv.clone();
    xn[7] = f64::NAN;
    with!("nanmean", [(x, xn, &[4, 5])], vec![x.nanmean().unwrap().reshape(&[1]).unwrap()]);
    with!("nansum", [(x, xn, &[4, 5])], vec![x.nansum().unwrap().reshape(&[1]).unwrap()]);
    with!("clamp_nan", [(x, xn, &[4, 5])], vec![x.clamp(-1.0, 1.0).unwrap()]);
    with!("norm_inf_ties", [(x, xv, &[4, 5])], vec![x.norm_p(f64::INFINITY, &[1], false).unwrap()]);
    with!("aminmax_ties", [(x, xv, &[4, 5])], {
        let (mn, mx) = x.aminmax(Some(1), false).unwrap();
        vec![mn, mx]
    });
    with!("matrix_norm_inf_ties", [(x, [1.0, -2.0, 2.0, 1.0, 0.5, 0.25], &[3, 2])], vec![x.matrix_norm("inf", false).unwrap().reshape(&[1]).unwrap()]);
    // interpolation backward
    let img = base(2 * 3 * 4, 5, 0.6);
    with!("interp_nearest", [(x, img, &[2, 3, 4, 5])], vec![interpolate(&x, Some(&[7, 9]), None, InterpMode::Nearest, false).unwrap()]);
    with!("interp_bilinear_f", [(x, img, &[2, 3, 4, 5])], vec![interpolate(&x, Some(&[7, 9]), None, InterpMode::Linear, false).unwrap()]);
    with!("interp_bilinear_t", [(x, img, &[2, 3, 4, 5])], vec![interpolate(&x, Some(&[6, 8]), None, InterpMode::Linear, true).unwrap()]);
    with!("interp_bicubic", [(x, img, &[2, 3, 4, 5])], vec![interpolate(&x, Some(&[7, 9]), None, InterpMode::Bicubic, false).unwrap()]);
    with!("interp_down", [(x, img, &[2, 3, 4, 5])], vec![interpolate(&x, Some(&[2, 3]), None, InterpMode::Linear, false).unwrap()]);
    with!("interp_area", [(x, img, &[2, 3, 4, 5])], vec![interpolate(&x, Some(&[2, 3]), None, InterpMode::Area, false).unwrap()]);
}
