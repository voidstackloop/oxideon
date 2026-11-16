//! Accuracy of the CPU reductions on long inputs.

use oxideon::prelude::*;

/// A running `f32` total adds the same rounding error at every step once it is large: summing a million 0.1s used to give a
/// result 0.85% too high. Sums and means accumulate in `f64` now.
#[test]
fn f32_sums_and_means_over_a_million_values_are_accurate() -> Result<()> {
    let tenth = f64::from(0.1f32);
    let x = Tensor::full(&[1_000_000], 0.1, DType::F32, &Device::Cpu)?;
    let sum = f64::from(x.sum()?.item::<f32>()?);
    assert!(
        (sum - 1_000_000.0 * tenth).abs() / (1_000_000.0 * tenth) < 1e-6,
        "sum {sum}"
    );
    let mean = f64::from(x.mean()?.item::<f32>()?);
    assert!((mean - tenth).abs() / tenth < 1e-6, "mean {mean}");

    // many rows (one task per row) and many columns (one long reduction per output)
    let y = Tensor::full(&[1000, 1000], 0.1, DType::F32, &Device::Cpu)?;
    for (axis, what) in [(1, "rows"), (0, "columns")] {
        let s = y.sum_axis(axis, false)?.to_vec::<f32>()?;
        assert_eq!(s.len(), 1000);
        assert!(
            s.iter()
                .all(|&v| (f64::from(v) - 1000.0 * tenth).abs() / (1000.0 * tenth) < 1e-6),
            "sums over {what}"
        );
    }
    Ok(())
}

#[test]
fn the_sum_does_not_depend_on_how_many_threads_ran() -> Result<()> {
    // blocks are added in a fixed order, so repeated runs give the same bits
    manual_seed(3);
    let x = Tensor::randn(&[3_000_000])?;
    let first = x.sum()?.item::<f32>()?;
    for _ in 0..5 {
        assert_eq!(x.sum()?.item::<f32>()?.to_bits(), first.to_bits());
    }
    Ok(())
}
