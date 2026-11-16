//! NN/training hardening: data pipeline guards, metric validation, distribution
//! validation, trainer empty-loader behavior, seeding determinism, and
//! state_dict round-trips.

use oxideon::data::{
    DataLoader, Dataset, Normalize, TensorDataset, Transform, batch_indices, k_fold_split,
    make_blobs, make_circles, make_moons, train_test_split,
};
use oxideon::distributions::{Categorical, Laplace, Normal};
use oxideon::metrics::{accuracy, r2_score, top_k_accuracy};
use oxideon::nn::loss::mse_loss;
use oxideon::nn::{Linear, Module, Sequential};
use oxideon::optim::Sgd;
use oxideon::prelude::*;
use oxideon::rng;
use oxideon::trainer::{Trainer, TrainerConfig};

// ---------------------------------------------------------------------------
// data: make_blobs
// ---------------------------------------------------------------------------

#[test]
fn make_blobs_rejects_zero_centers() {
    assert!(make_blobs(10, 2, 0, 1.0).is_err());
}

#[test]
fn make_blobs_rejects_zero_features() {
    assert!(make_blobs(10, 0, 3, 1.0).is_err());
}

#[test]
fn make_blobs_basic_shapes_ok() {
    let (x, y) = make_blobs(12, 2, 3, 0.5).unwrap();
    assert_eq!(x.shape(), &[12, 2]);
    assert_eq!(y.shape(), &[12]);
}

// ---------------------------------------------------------------------------
// data: batch_indices / TensorDataset
// ---------------------------------------------------------------------------

#[test]
fn batch_indices_zero_size_does_not_panic() {
    // Clamped to 1 (same convention as DataLoader::new): singles, no panic.
    let batches = batch_indices(5, 0, false);
    assert_eq!(batches, vec![vec![0], vec![1], vec![2], vec![3], vec![4]]);
}

#[test]
fn tensor_dataset_try_new_validates() {
    let good_x = Tensor::zeros(&[4, 2], DType::F32, &Device::Cpu).unwrap();
    let good_y = Tensor::zeros(&[4], DType::I64, &Device::Cpu).unwrap();
    assert!(TensorDataset::try_new(good_x.clone(), good_y.clone()).is_ok());
    // Length mismatch.
    let bad_y = Tensor::zeros(&[3], DType::I64, &Device::Cpu).unwrap();
    assert!(TensorDataset::try_new(good_x.clone(), bad_y).is_err());
    // Scalar (0-dim) inputs have no axis 0.
    let scalar = Tensor::from_slice::<f32>(&[1.0], &[]).unwrap();
    assert!(TensorDataset::try_new(scalar.clone(), scalar).is_err());
}

#[test]
#[should_panic(expected = "TensorDataset")]
fn tensor_dataset_new_panics_with_clear_message_on_scalar() {
    let scalar = Tensor::from_slice::<f32>(&[1.0], &[]).unwrap();
    let _ = TensorDataset::new(scalar.clone(), scalar);
}

// ---------------------------------------------------------------------------
// data: train_test_split / k_fold_split
// ---------------------------------------------------------------------------

#[test]
fn train_test_split_rejects_nan_ratio() {
    let x = Tensor::zeros(&[8, 2], DType::F32, &Device::Cpu).unwrap();
    let y = Tensor::zeros(&[8], DType::I64, &Device::Cpu).unwrap();
    assert!(train_test_split(&x, &y, f64::NAN, true).is_err());
}

#[test]
fn train_test_split_rejects_scalar_and_mismatched_inputs() {
    let x = Tensor::zeros(&[8, 2], DType::F32, &Device::Cpu).unwrap();
    let bad_y = Tensor::zeros(&[7], DType::I64, &Device::Cpu).unwrap();
    assert!(train_test_split(&x, &bad_y, 0.25, false).is_err());
    let scalar = Tensor::from_slice::<f32>(&[1.0], &[]).unwrap();
    let y = Tensor::zeros(&[8], DType::I64, &Device::Cpu).unwrap();
    assert!(train_test_split(&scalar, &y, 0.25, false).is_err());
}

#[test]
fn train_test_split_basic_ok() {
    let x = Tensor::zeros(&[8, 2], DType::F32, &Device::Cpu).unwrap();
    let y = Tensor::zeros(&[8], DType::I64, &Device::Cpu).unwrap();
    let (trx, _try, tex, _tey) = train_test_split(&x, &y, 0.25, false).unwrap();
    assert_eq!(trx.shape()[0] + tex.shape()[0], 8);
}

#[test]
fn k_fold_split_validates_inputs() {
    let x = Tensor::zeros(&[8, 2], DType::F32, &Device::Cpu).unwrap();
    let bad_y = Tensor::zeros(&[7], DType::I64, &Device::Cpu).unwrap();
    assert!(k_fold_split(&x, &bad_y, 4, false).is_err());
    let scalar = Tensor::from_slice::<f32>(&[1.0], &[]).unwrap();
    let y = Tensor::zeros(&[8], DType::I64, &Device::Cpu).unwrap();
    assert!(k_fold_split(&scalar, &y, 4, false).is_err());
    let folds = k_fold_split(&x, &y, 4, false).unwrap();
    assert_eq!(folds.len(), 4);
}

// ---------------------------------------------------------------------------
// data: Normalize
// ---------------------------------------------------------------------------

#[test]
fn normalize_rejects_zero_and_nonfinite_std() {
    let x = Tensor::ones(&[2, 2], DType::F32, &Device::Cpu).unwrap();
    assert!(Normalize::new(0.0, 0.0).apply(&x).is_err());
    assert!(Normalize::new(0.0, f64::NAN).apply(&x).is_err());
    assert!(Normalize::new(0.0, f64::INFINITY).apply(&x).is_err());
    let y = Normalize::new(1.0, 2.0).apply(&x).unwrap();
    let v = y.to_vec::<f32>().unwrap();
    assert!(v.iter().all(|&a| a == 0.0));
}

// ---------------------------------------------------------------------------
// metrics
// ---------------------------------------------------------------------------

fn tiny_logits() -> Tensor {
    // Row 0 -> class 0, row 1 -> class 2, row 2 -> class 1.
    Tensor::from_slice::<f32>(&[5.0, 1.0, 0.0, 0.0, 1.0, 4.0, 0.0, 3.0, 1.0], &[3, 3]).unwrap()
}

#[test]
fn accuracy_validates_and_scores() {
    let logits = tiny_logits();
    assert!((accuracy(&logits, &[0, 2, 1]).unwrap() - 1.0).abs() < 1e-9);
    assert!((accuracy(&logits, &[0, 2, 0]).unwrap() - 2.0 / 3.0).abs() < 1e-9);
    assert!(accuracy(&logits, &[]).is_err());
    assert!(accuracy(&logits, &[0, 2]).is_err());
    assert!(accuracy(&logits, &[0, 2, 1, 0]).is_err());
}

#[test]
fn top_k_accuracy_validates_k_and_lengths() {
    let logits = tiny_logits();
    assert!((top_k_accuracy(&logits, &[0, 2, 1], 1).unwrap() - 1.0).abs() < 1e-9);
    assert!((top_k_accuracy(&logits, &[2, 0, 1], 2).unwrap() - 1.0 / 3.0).abs() < 1e-9);
    assert!(top_k_accuracy(&logits, &[0, 2, 1], 0).is_err());
    assert!(top_k_accuracy(&logits, &[0, 2, 1], 4).is_err());
    assert!(top_k_accuracy(&logits, &[], 1).is_err());
    // More targets than rows used to slice out of bounds.
    assert!(top_k_accuracy(&logits, &[0, 2, 1, 0], 1).is_err());
    assert!(top_k_accuracy(&logits, &[0, 2], 1).is_err());
}

#[test]
fn r2_score_handles_constant_targets_like_sklearn() {
    let perfect = Tensor::from_slice::<f32>(&[5.0, 5.0, 5.0], &[3]).unwrap();
    assert!((r2_score(&perfect, &perfect).unwrap() - 1.0).abs() < 1e-9);
    let bad = Tensor::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3]).unwrap();
    assert!(r2_score(&bad, &perfect).unwrap().abs() < 1e-9);
    let pred = Tensor::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3]).unwrap();
    let targ = Tensor::from_slice::<f32>(&[1.0, 2.0, 4.0], &[3]).unwrap();
    let r2 = r2_score(&pred, &targ).unwrap();
    assert!((r2 - 11.0 / 14.0).abs() < 1e-6, "unexpected r2 {r2}");
}

// ---------------------------------------------------------------------------
// distributions
// ---------------------------------------------------------------------------

#[test]
fn gumbel_softmax_rejects_bad_temperature() {
    let logits = Tensor::zeros(&[2, 4], DType::F32, &Device::Cpu).unwrap();
    let cat = Categorical::from_logits(logits);
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(cat.gumbel_softmax(bad, false).is_err(), "temp {bad}");
    }
    let y = cat.gumbel_softmax(0.5, false).unwrap();
    assert_eq!(y.shape(), &[2, 4]);
}

#[test]
fn normal_and_laplace_reject_nonpositive_scale() {
    let loc = Tensor::zeros(&[3], DType::F32, &Device::Cpu).unwrap();
    let good = Tensor::ones(&[3], DType::F32, &Device::Cpu).unwrap();
    assert!(Normal::new(loc.clone(), good.clone()).is_ok());
    assert!(Laplace::new(loc.clone(), good.clone()).is_ok());
    let zero = Tensor::zeros(&[3], DType::F32, &Device::Cpu).unwrap();
    assert!(Normal::new(loc.clone(), zero.clone()).is_err());
    assert!(Laplace::new(loc.clone(), zero.clone()).is_err());
    let neg = Tensor::from_slice::<f32>(&[1.0, -1.0, 1.0], &[3]).unwrap();
    assert!(Normal::new(loc.clone(), neg.clone()).is_err());
    assert!(Laplace::new(loc.clone(), neg.clone()).is_err());
    let nan = Tensor::from_slice::<f32>(&[1.0, f32::NAN, 1.0], &[3]).unwrap();
    assert!(Normal::new(loc.clone(), nan.clone()).is_err());
    assert!(Laplace::new(loc, nan).is_err());
}

// ---------------------------------------------------------------------------
// trainer
// ---------------------------------------------------------------------------

fn lin_reg_data(n: usize) -> (Tensor, Tensor) {
    rng::manual_seed(7);
    let mut xs = Vec::with_capacity(n * 2);
    let mut ys = Vec::with_capacity(n);
    rng::with_rng(|r| {
        for _ in 0..n {
            let a = r.uniform(-2.0, 2.0);
            let b = r.uniform(-2.0, 2.0);
            xs.push(a);
            xs.push(b);
            ys.push(2.0 * a - b + 0.5);
        }
    });
    (
        Tensor::from_slice::<f32>(&xs, &[n, 2]).unwrap(),
        Tensor::from_slice::<f32>(&ys, &[n, 1]).unwrap(),
    )
}

#[test]
fn trainer_rejects_empty_train_loader() {
    let x = Tensor::zeros(&[0, 2], DType::F32, &Device::Cpu).unwrap();
    let y = Tensor::zeros(&[0, 1], DType::F32, &Device::Cpu).unwrap();
    let ds = TensorDataset::try_new(x, y).unwrap();
    let loader = DataLoader::new(&ds, 4, false);
    let model = Linear::new(2, 1).unwrap();
    let mut opt = Sgd::new(model.parameters(), 0.01);
    let trainer = Trainer::new(TrainerConfig::new(1));
    assert!(
        trainer
            .fit(&model, &mut opt, &loader, None, mse_loss)
            .is_err()
    );
}

#[test]
fn trainer_reports_none_val_loss_for_empty_val_loader() {
    let (x, y) = lin_reg_data(16);
    let ds = TensorDataset::try_new(x, y).unwrap();
    let loader = DataLoader::new(&ds, 8, false);
    let vx = Tensor::zeros(&[0, 2], DType::F32, &Device::Cpu).unwrap();
    let vy = Tensor::zeros(&[0, 1], DType::F32, &Device::Cpu).unwrap();
    let vds = TensorDataset::try_new(vx, vy).unwrap();
    let vloader = DataLoader::new(&vds, 8, false);
    let model = Linear::new(2, 1).unwrap();
    let mut opt = Sgd::new(model.parameters(), 0.01);
    let trainer = Trainer::new(TrainerConfig::new(1));
    let hist = trainer
        .fit(&model, &mut opt, &loader, Some(&vloader), mse_loss)
        .unwrap();
    assert_eq!(hist.metrics.len(), 1);
    assert!(hist.metrics[0].val_loss.is_none());
}

#[test]
fn trainer_reduces_linear_regression_loss() {
    let (x, y) = lin_reg_data(64);
    let ds = TensorDataset::try_new(x, y).unwrap();
    let loader = DataLoader::new(&ds, 16, true);
    let model = Linear::new(2, 1).unwrap();
    let mut opt = Sgd::new(model.parameters(), 0.05);
    let trainer = Trainer::new(TrainerConfig::new(15));
    let hist = trainer
        .fit(&model, &mut opt, &loader, None, mse_loss)
        .unwrap();
    assert_eq!(hist.metrics.len(), 15);
    let first = hist.metrics[0].train_loss;
    let last = hist.metrics[14].train_loss;
    assert!(last < first, "loss did not decrease: {first} -> {last}");
    assert!(hist.best_loss <= last + 1e-9);
}

// ---------------------------------------------------------------------------
// seeding determinism
// ---------------------------------------------------------------------------

#[test]
fn data_pipeline_is_deterministic_under_manual_seed() {
    rng::manual_seed(1234);
    let (x1, y1) = make_moons(32, 0.1).unwrap();
    let b1 = batch_indices(32, 8, true);
    rng::manual_seed(1234);
    let (x2, y2) = make_moons(32, 0.1).unwrap();
    let b2 = batch_indices(32, 8, true);
    assert_eq!(x1.to_vec::<f32>().unwrap(), x2.to_vec::<f32>().unwrap());
    assert_eq!(y1.to_vec::<i64>().unwrap(), y2.to_vec::<i64>().unwrap());
    assert_eq!(b1, b2);
}

// ---------------------------------------------------------------------------
// state_dict round-trips
// ---------------------------------------------------------------------------

fn tiny_mlp() -> Sequential {
    Sequential::new()
        .add(Linear::new(2, 3).unwrap())
        .add(Linear::new(3, 1).unwrap())
}

fn param_vecs(m: &Sequential) -> Vec<Vec<f32>> {
    m.parameters()
        .iter()
        .map(|p| p.to_vec::<f32>().unwrap())
        .collect()
}

#[test]
fn sequential_state_dict_roundtrip_restores_params() {
    let model = tiny_mlp();
    let before = param_vecs(&model);
    let bytes = oxideon::serialize::serialize_safetensors(&model.state_dict()).unwrap();
    // Perturb every parameter, then restore.
    for p in model.parameters() {
        let noisy = p.add_scalar(1.0).unwrap();
        p.assign(&noisy).unwrap();
    }
    assert_ne!(param_vecs(&model), before);
    let state = oxideon::serialize::deserialize_safetensors(&bytes).unwrap();
    model.load_state_dict(&state).unwrap();
    assert_eq!(param_vecs(&model), before);
}

#[test]
fn load_state_dict_rejects_missing_keys() {
    let model = tiny_mlp();
    let mut state = model.state_dict();
    let dropped = state.keys().next().unwrap().clone();
    state.remove(&dropped);
    assert!(model.load_state_dict(&state).is_err());
}

/// Extra keys in the incoming state are silently ignored (torch `strict=True` would raise;
/// oxideon deliberately behaves like `strict=False` for extras while staying strict on
/// missing keys and shape mismatches, so partial checkpoints can load).
#[test]
fn load_state_dict_ignores_unexpected_keys() {
    let model = tiny_mlp();
    let before = param_vecs(&model);
    let mut state = model.state_dict();
    state.insert(
        "not.a.real.param".to_string(),
        Tensor::from_slice(&[1.0f32], &[1]).unwrap(),
    );
    model.load_state_dict(&state).unwrap();
    assert_eq!(param_vecs(&model), before);
}

#[test]
fn load_state_dict_rejects_shape_mismatch() {
    let model = tiny_mlp();
    let mut state = model.state_dict();
    let name = state.keys().next().unwrap().clone();
    // [7, 7] matches no tiny_mlp parameter ([3, 2], [3], [1, 3], [1]): deterministic.
    let bad = Tensor::from_slice(&[1.0f32; 49], &[7, 7]).unwrap();
    state.insert(name, bad);
    assert!(model.load_state_dict(&state).is_err());
}

/// Dtype mismatches cast (torch `load_state_dict` parity: `param.copy_` casts too), so an
/// f64 checkpoint loads into an f32 model with values intact.
#[test]
fn load_state_dict_casts_dtype_mismatch() {
    let model = tiny_mlp();
    let before = param_vecs(&model);
    let mut state = model.state_dict();
    for t in state.values_mut() {
        let cast = t.to_dtype(DType::F64).unwrap();
        *t = cast;
    }
    for p in model.parameters() {
        let noisy = p.add_scalar(1.0).unwrap();
        p.assign(&noisy).unwrap();
    }
    model.load_state_dict(&state).unwrap();
    for p in model.parameters() {
        assert_eq!(p.dtype(), DType::F32);
    }
    assert_eq!(param_vecs(&model), before);
}

// ---------------------------------------------------------------------------
// empty-batch NaN guards
// ---------------------------------------------------------------------------

#[test]
fn batch_losses_reject_empty_batches() {
    use oxideon::nn::loss::{cross_entropy, focal_loss, nll_loss};
    let empty = Tensor::zeros(&[0, 3], DType::F32, &Device::Cpu).unwrap();
    assert!(nll_loss(&empty, &[]).is_err());
    assert!(cross_entropy(&empty, &[]).is_err());
    assert!(focal_loss(&empty, &[], 0.25, 2.0).is_err());
    // Non-empty behavior unchanged.
    let logits = Tensor::from_slice::<f32>(&[2.0, 1.0, 0.5, 0.5, 3.0, 1.0], &[2, 3]).unwrap();
    assert!(nll_loss(&logits.softmax(1).unwrap().ln().unwrap(), &[0, 1]).is_ok());
    assert!(cross_entropy(&logits, &[0, 1]).is_ok());
    assert!(focal_loss(&logits, &[0, 1], 0.25, 2.0).is_ok());
}

#[test]
fn regression_metrics_reject_empty_inputs() {
    use oxideon::metrics::rmse;
    let empty = Tensor::zeros(&[0], DType::F32, &Device::Cpu).unwrap();
    assert!(rmse(&empty, &empty).is_err());
    assert!(r2_score(&empty, &empty).is_err());
    let pred = Tensor::from_slice::<f32>(&[1.0, 2.0], &[2]).unwrap();
    let targ = Tensor::from_slice::<f32>(&[1.0, 3.0], &[2]).unwrap();
    assert!(rmse(&pred, &targ).unwrap() > 0.0);
}

// ---------------------------------------------------------------------------
// data (pt.12/H): huge-index wrap + giant generator sizes
// ---------------------------------------------------------------------------

#[test]
fn tensor_dataset_get_rejects_huge_indices_without_wrapping() {
    let x = Tensor::from_slice::<f32>(&[1.0, 2.0, 3.0], &[3]).unwrap();
    let y = Tensor::from_slice::<i64>(&[0, 0, 0], &[3]).unwrap();
    let ds = TensorDataset::try_new(x, y).unwrap();
    // usize::MAX must be an error, not `as isize` wrapping to -1 (last row).
    assert!(ds.get(usize::MAX).is_err());
    assert!(ds.get_batch(&[usize::MAX]).is_err());
    assert!(ds.get_batch(&[0, usize::MAX]).is_err());
    // Ordinary out-of-bounds still errors; in-bounds still works.
    assert!(ds.get(3).is_err());
    let (gx, _) = ds.get(0).unwrap();
    assert_eq!(gx.to_vec::<f32>().unwrap(), vec![1.0]);
}

#[test]
fn make_moons_and_circles_reject_giant_sizes() {
    // Overflowing (usize::MAX * 2) and merely giant (2^41 elements) alike.
    assert!(make_moons(usize::MAX, 0.1).is_err());
    assert!(make_moons(1 << 40, 0.1).is_err());
    assert!(make_circles(usize::MAX, 0.5, 0.1).is_err());
    assert!(make_circles(1 << 40, 0.5, 0.1).is_err());
    // Ordinary sizes still work.
    let (x, y) = make_moons(10, 0.1).unwrap();
    assert_eq!(x.shape(), &[10, 2]);
    assert_eq!(y.shape(), &[10]);
}

#[test]
fn make_blobs_rejects_giant_sizes() {
    // 2^60 elements: fits usize (passes checked_mul) but can never allocate.
    assert!(make_blobs(1 << 40, 1 << 20, 3, 0.5).is_err());
}

#[test]
fn trainer_fit_rejects_absurd_epoch_counts() {
    let (x, y) = lin_reg_data(8);
    let ds = TensorDataset::try_new(x, y).unwrap();
    let loader = DataLoader::new(&ds, 4, false);
    let model = Linear::new(2, 1).unwrap();
    let mut opt = Sgd::new(model.parameters(), 0.01);
    let trainer = Trainer::new(TrainerConfig::new(usize::MAX));
    assert!(
        trainer
            .fit(&model, &mut opt, &loader, None, mse_loss)
            .is_err()
    );
}
