//! Cross-validation batch 38: reductions and order statistics on awkward inputs (NaN, infinities, ties, single
//! elements, all -inf rows) vs PyTorch (tools/crossval/check_batch38.py).
use oxideon_core::dtype::DType;
use oxideon_core::tensor::Tensor;

fn dump(name: &str, r: oxideon_core::error::Result<Tensor>) {
    match r {
        Ok(t) => {
            let v = t.to_dtype(DType::F64).unwrap().to_vec::<f64>().unwrap();
            println!("{name} {}", if v.is_empty() { "EMPTY".to_string() } else { v.iter().map(|x| format!("{x:e}")).collect::<Vec<_>>().join(" ") });
        }
        Err(e) => println!("{name} ERR {e:?}"),
    }
}

fn pair(name: &str, r: oxideon_core::error::Result<(Tensor, Tensor)>) {
    match r {
        Ok((a, b)) => {
            dump(&format!("{name}_v"), Ok(a));
            dump(&format!("{name}_i"), Ok(b));
        }
        Err(e) => println!("{name}_v ERR {e:?}"),
    }
}

#[test]
#[ignore]
fn emit_reductions() {
    let (inf, nan) = (f64::INFINITY, f64::NAN);
    let rows: Vec<(&str, Vec<f64>)> = vec![
        ("plain", vec![3.0, -1.0, 2.0, 7.0, 0.5, -4.0]),
        ("ties", vec![2.0, 5.0, 5.0, -3.0, 5.0, -3.0]),
        ("nan_mid", vec![1.0, nan, 3.0, -2.0, 0.0, 4.0]),
        ("nan_first", vec![nan, 1.0, 3.0, -2.0, 0.0, 4.0]),
        ("infs", vec![inf, 1.0, -inf, 2.0, 0.0, 5.0]),
        ("pos_inf2", vec![inf, 1.0, inf, 2.0, 0.0, 5.0]),
        ("all_neg_inf", vec![-inf, -inf, -inf, -inf, -inf, -inf]),
        ("zeros", vec![0.0, -0.0, 0.0, 0.0, -0.0, 0.0]),
        ("single", vec![4.25]),
        ("big", vec![1e308, 1e308, -1e308, 3.0, 2.0, 1.0]),
    ];
    for (tag, v) in &rows {
        let x = Tensor::from_slice(v, &[v.len()]).unwrap();
        dump(&format!("{tag}_sum"), x.sum());
        dump(&format!("{tag}_mean"), x.mean());
        dump(&format!("{tag}_prod"), x.prod());
        dump(&format!("{tag}_max"), x.max_axis(0, false));
        dump(&format!("{tag}_min"), x.min_axis(0, false));
        dump(&format!("{tag}_amax"), x.amax(&[0], false));
        dump(&format!("{tag}_argmax"), x.argmax_axis(0, false));
        dump(&format!("{tag}_argmin"), x.argmin_axis(0, false));
        dump(&format!("{tag}_median"), x.median());
        dump(&format!("{tag}_var"), x.var_axis(0, true, false));
        dump(&format!("{tag}_std0"), x.std_axis(0, false, false));
        dump(&format!("{tag}_logsumexp"), x.logsumexp(0, false));
        dump(&format!("{tag}_softmax"), x.softmax(0));
        dump(&format!("{tag}_log_softmax"), x.log_softmax(0));
        dump(&format!("{tag}_cumsum"), x.cumsum(0));
        dump(&format!("{tag}_cumprod"), x.cumprod(0));
        pair(&format!("{tag}_cummax"), x.cummax(0));
        pair(&format!("{tag}_cummin"), x.cummin(0));
        pair(&format!("{tag}_sort"), x.sort(0, false));
        pair(&format!("{tag}_sort_desc"), x.sort(0, true));
        if v.len() >= 3 {
            pair(&format!("{tag}_topk"), x.topk(3, 0, true, true));
            pair(&format!("{tag}_kth"), x.kthvalue(2, 0, false));
        }
        dump(&format!("{tag}_norm2"), x.norm_p(2.0, &[0], false));
        dump(&format!("{tag}_norminf"), x.norm_p(inf, &[0], false));
        dump(&format!("{tag}_nansum"), x.nansum());
        dump(&format!("{tag}_nanmean"), x.nanmean());
        dump(&format!("{tag}_any"), x.any());
        dump(&format!("{tag}_all"), x.all());
    }
}
