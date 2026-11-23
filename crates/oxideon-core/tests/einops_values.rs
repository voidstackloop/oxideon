use oxideon_core::einops::{ReduceKind, rearrange, reduce, repeat};
use oxideon_core::Tensor;

fn arange(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_slice(&(0..n).map(|i| i as f32).collect::<Vec<_>>(), shape).unwrap()
}
fn v(t: &Tensor) -> Vec<f32> {
    t.contiguous().unwrap().to_vec::<f32>().unwrap()
}

#[test]
fn rearrange_moves_the_right_elements() {
    let x = arange(&[2, 3, 4]); // x[b, c, hw]
    let r = rearrange(&x, "b c (h w) -> b h w c", &[("h", 2)]).unwrap();
    assert_eq!(r.shape(), &[2, 2, 2, 3]);
    let got = v(&r);
    for b in 0..2 {
        for h in 0..2 {
            for w in 0..2 {
                for c in 0..3 {
                    assert_eq!(got[((b * 2 + h) * 2 + w) * 3 + c], (b * 12 + c * 4 + h * 2 + w) as f32, "b{b} h{h} w{w} c{c}");
                }
            }
        }
    }
    // merging axes: (c h) is c-major
    let m = rearrange(&arange(&[2, 3, 2]), "b c h -> b (c h)", &[]).unwrap();
    assert_eq!(v(&m), (0..12).map(|i| i as f32).collect::<Vec<_>>());
    let m2 = rearrange(&arange(&[2, 3, 2]), "b c h -> b (h c)", &[]).unwrap();
    assert_eq!(v(&m2)[..6], [0.0, 2.0, 4.0, 1.0, 3.0, 5.0]); // h-major
    // round trip
    let back = rearrange(&r, "b h w c -> b c (h w)", &[]).unwrap();
    assert_eq!(v(&back), v(&x));
    // split with the known size on the left-hand group
    let s = rearrange(&arange(&[6]), "(a b) -> a b", &[("a", 2)]).unwrap();
    assert_eq!((s.shape().to_vec(), v(&s)), (vec![2, 3], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]));
    // errors
    assert!(rearrange(&x, "b c h -> b c", &[]).is_err(), "axis h dropped");
    assert!(rearrange(&x, "b c (h w) -> b h w c", &[]).is_err(), "two unknown axes in a group");
    assert!(rearrange(&x, "b c (h w) -> b h w c", &[("h", 3)]).is_err(), "4 is not divisible by 3");
    assert!(rearrange(&x, "b c d -> b c x", &[]).is_err());
}

#[test]
fn reduce_computes_the_right_values() {
    let x = arange(&[2, 3, 4]);
    assert_eq!(v(&reduce(&x, "b c h -> b c", ReduceKind::Sum, &[]).unwrap()), vec![6.0, 22.0, 38.0, 54.0, 70.0, 86.0]);
    assert_eq!(v(&reduce(&x, "b c h -> b", ReduceKind::Mean, &[]).unwrap()), vec![5.5, 17.5]);
    assert_eq!(v(&reduce(&x, "b c h -> c", ReduceKind::Max, &[]).unwrap()), vec![15.0, 19.0, 23.0]);
    assert_eq!(v(&reduce(&x, "b c h -> h", ReduceKind::Min, &[]).unwrap()), vec![0.0, 1.0, 2.0, 3.0]);
    // reduce and transpose at once, and a 2x2 average pooling expressed with groups
    let t = reduce(&x, "b c h -> h b", ReduceKind::Sum, &[]).unwrap();
    assert_eq!(t.shape(), &[4, 2]);
    let img = arange(&[1, 1, 4, 4]);
    let pooled = reduce(&img, "b c (h h2) (w w2) -> b c h w", ReduceKind::Mean, &[("h2", 2), ("w2", 2)]).unwrap();
    assert_eq!(v(&pooled), vec![2.5, 4.5, 10.5, 12.5]);
}

#[test]
fn repeat_tiles_in_einops_order() {
    let y = arange(&[2, 2]);
    let r = repeat(&y, "h w -> b h w", &[("b", 2)]).unwrap();
    assert_eq!(v(&r), vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]);
    // (h r): each row repeated consecutively; (r h): the whole block repeated
    assert_eq!(v(&repeat(&y, "h w -> (h r) w", &[("r", 2)]).unwrap()), vec![0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 3.0]);
    assert_eq!(v(&repeat(&y, "h w -> (r h) w", &[("r", 2)]).unwrap()), vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]);
    assert!(repeat(&y, "h w -> h w c", &[]).is_err(), "unknown size for c");
}

#[test]
fn einops_is_differentiable() {
    let x = arange(&[2, 3]).requires_grad_(true);
    let y = rearrange(&x, "a b -> b a", &[]).unwrap();
    y.mul(&arange(&[3, 2])).unwrap().sum().unwrap().backward().unwrap();
    // y[b, a] = x[a, b] is weighted by arange[b, a] = 2b + a
    assert_eq!(v(&x.grad().unwrap()), vec![0.0, 2.0, 4.0, 1.0, 3.0, 5.0]);
}

#[test]
fn ellipsis_and_unit_axes() {
    let x = arange(&[2, 3, 4, 5]);
    // `...` stands for the axes in the middle, on both sides
    let moved = rearrange(&x, "b ... c -> c b ...", &[]).unwrap();
    assert_eq!(moved.shape(), &[5, 2, 3, 4]);
    let want = rearrange(&x, "b h w c -> c b h w", &[]).unwrap();
    assert_eq!(v(&moved), v(&want));
    // the same pattern works for another rank
    assert_eq!(rearrange(&arange(&[2, 3, 5]), "b ... c -> c b ...", &[]).unwrap().shape(), &[5, 2, 3]);
    assert_eq!(rearrange(&arange(&[2, 5]), "b ... c -> c b ...", &[]).unwrap().shape(), &[5, 2], "an ellipsis can be empty");
    // flatten everything after the first axis
    assert_eq!(rearrange(&x, "b ... -> b (...)", &[]).unwrap().shape(), &[2, 60]);
    // reduce over the ellipsis axes
    assert_eq!(v(&reduce(&x, "b ... c -> b c", ReduceKind::Sum, &[]).unwrap()), v(&reduce(&x, "b h w c -> b c", ReduceKind::Sum, &[]).unwrap()));
    // repeat keeps the ellipsis axes
    assert_eq!(repeat(&arange(&[2, 3]), "... -> k ...", &[("k", 4)]).unwrap().shape(), &[4, 2, 3]);
    // unit axes: `()` and `1`
    let y = arange(&[3, 4]);
    assert_eq!(rearrange(&y, "h w -> h () w", &[]).unwrap().shape(), &[3, 1, 4]);
    assert_eq!(rearrange(&y, "h w -> 1 h w 1", &[]).unwrap().shape(), &[1, 3, 4, 1]);
    assert_eq!(v(&rearrange(&arange(&[1, 3, 1]), "1 h () -> h", &[]).unwrap()), vec![0.0, 1.0, 2.0]);
    assert!(rearrange(&y, "h () w -> h w", &[]).is_err(), "a unit axis must have size 1");
    // errors
    assert!(rearrange(&x, "... ... -> ...", &[]).is_err());
    assert!(rearrange(&arange(&[2]), "a b ... -> a b", &[]).is_err(), "too few dimensions");
}
