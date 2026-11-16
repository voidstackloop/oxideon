//! Mistakes a user can make by accident must come back as `Err`, never abort or hang the process.

use oxideon::amp::GradScaler;
use oxideon::audio::{RVQConfig, ResidualVectorQuantizer};
use oxideon::diffusion::{
    BetaSchedule, DDPMScheduler, DiffusionForcingScheduler, EDMConfig, RectifiedFlowConfig,
    RectifiedFlowEngine, timestep_embedding,
};
use oxideon::explain::{integrated_gradients, smooth_grad};
use oxideon::metrics::{auroc, confusion_matrix, precision_recall_f1};
use oxideon::nn::loss::{compute_gae, ctc_loss, info_nce_loss};
use oxideon::nn::{
    AlphaDropout, Bilinear, CausalLMBlock, ConvNeXtBlock, DropPath, Dropout, Dropout2d, GRUCell,
    GroupNorm, GroupedQueryAttention, LSTMCell, Linear, Module, MultiHeadAttention, RNNCell,
    RotaryEmbedding, SinusoidalPositionalEncoding, SqueezeExcitation2d, TransfusionConfig,
    TransfusionModel, VisionLanguageActionModel,
};
use oxideon::prelude::*;
use oxideon::quant::{QuantScheme, QuantizedTensor, TernaryPackedTensor};
use oxideon::serving::{BlockAllocator, BlockTable, PagedKVCache, paged_attention};
use oxideon::tokenizer::BpeTokenizer;
use oxideon::video::flow_dit::continuous_timestep_embedding;
use oxideon::video::{RectifiedFlowVideoDiT, RotaryEmbedding3D, VideoDiTConfig};
use oxideon::{CkksContext, crypto::fhe::CKKS_MAX_DEGREE};

#[test]
fn absurd_shapes_are_errors_not_aborts() {
    // The element count overflows usize (it used to wrap silently in release builds).
    assert!(Tensor::zeros(&[usize::MAX / 2, 4], DType::F32, &Device::Cpu).is_err());
    assert!(Tensor::from_slice::<f32>(&[], &[1 << 40, 1 << 40]).is_err());
    // The allocation itself fails (it used to abort the whole process).
    assert!(Tensor::zeros(&[1 << 50], DType::F32, &Device::Cpu).is_err());
    assert!(Tensor::ones(&[1 << 50], DType::F64, &Device::Cpu).is_err());
    assert!(Tensor::randn(&[1 << 50]).is_err());
    assert!(Tensor::rand_uniform(&[1 << 50], 0.0, 1.0).is_err());
    assert!(Tensor::arange(0.0, 1e15, 1.0, DType::F32).is_err());
    assert!(Tensor::arange(0.0, f64::INFINITY, 1.0, DType::F32).is_err());
    // A size the OS may well hand out (Linux with permissive overcommit does, then the zero-fill is OOM-killed) is refused up
    // front by the element limit, so the outcome does not depend on this machine's memory or overcommit setting.
    let over = oxideon::limits::MAX_ELEMENTS + 1;
    assert!(Tensor::zeros(&[over], DType::F32, &Device::Cpu).is_err());
    assert!(Tensor::zeros(&[1_000_000_000_000], DType::F32, &Device::Cpu).is_err());
    assert!(Tensor::randn(&[over]).is_err());
    assert!(Tensor::arange(0.0, over as f64, 1.0, DType::F32).is_err());
    assert!(Tensor::linspace(0.0, 1.0, over, DType::F32).is_err());
    assert!(Tensor::eye(1 << 16, DType::F32, &Device::Cpu).is_err()); // 2^32 elements
    // Meta tensors hold no data and keep describing models that would not fit.
    assert!(Tensor::zeros(&[over], DType::F32, &Device::Meta).is_ok());
    // Ordinary sizes still work.
    assert_eq!(
        Tensor::zeros(&[3, 4], DType::F32, &Device::Cpu)
            .unwrap()
            .numel(),
        12
    );
}

#[test]
fn i64_values_survive_a_round_trip_exactly() {
    // 2^60 + 1 is not representable as f64; the storage used to be read and written through f64.
    let big = (1i64 << 60) + 1;
    let t = Tensor::from_slice(&[big, -big, i64::MAX, i64::MIN], &[4]).unwrap();
    assert_eq!(
        t.to_vec::<i64>().unwrap(),
        vec![big, -big, i64::MAX, i64::MIN]
    );
    assert_eq!(t.narrow(0, 0, 1).unwrap().item::<i64>().unwrap(), big);
}

#[test]
fn reshape_with_overflowing_shape_is_an_error() {
    let t = Tensor::zeros(&[0], DType::F32, &Device::Cpu).unwrap();
    assert!(t.reshape(&[1 << 40, 1 << 40]).is_err());
}

#[test]
fn integer_division_by_zero_is_an_error() {
    let a = Tensor::from_slice(&[1i64, 2], &[2]).unwrap();
    let zero = Tensor::from_slice(&[0i64, 1], &[2]).unwrap();
    assert!(a.div(&zero).is_err());
    let ok = Tensor::from_slice(&[2i64, 1], &[2]).unwrap();
    assert_eq!(a.div(&ok).unwrap().to_vec::<i64>().unwrap(), vec![0, 2]);
    // i64::MIN / -1 overflows; it must not panic.
    let min = Tensor::from_slice(&[i64::MIN], &[1]).unwrap();
    let neg1 = Tensor::from_slice(&[-1i64], &[1]).unwrap();
    assert!(min.div(&neg1).is_ok());
    // Wrapping semantics: the wrapped quotient is MIN itself.
    assert_eq!(
        min.div(&neg1).unwrap().to_vec::<i64>().unwrap(),
        vec![i64::MIN]
    );
    // A zero that is outside the broadcast view does not count.
    let wide = Tensor::from_slice(&[5i32, 0], &[2])
        .unwrap()
        .narrow(0, 0, 1)
        .unwrap();
    let num = Tensor::from_slice(&[10i32], &[1]).unwrap();
    assert_eq!(num.div(&wide).unwrap().to_vec::<i32>().unwrap(), vec![2]);
}

#[test]
fn reductions_over_an_empty_axis_are_errors() {
    let empty = Tensor::zeros(&[0], DType::F32, &Device::Cpu).unwrap();
    assert!(empty.max_axis(0, false).is_err());
    assert!(empty.min_axis(0, false).is_err());
    assert!(empty.argmax_axis(0, false).is_err());
    assert!(empty.argmin_axis(0, false).is_err());
    // sum / mean of nothing are still defined.
    assert!(empty.sum().is_ok());
    let t = Tensor::from_slice(&[3.0f32, 1.0, 2.0], &[3]).unwrap();
    assert_eq!(
        t.argmax_axis(0, false).unwrap().to_vec::<i64>().unwrap(),
        vec![0]
    );
}

#[test]
fn ckks_degree_is_bounded() {
    assert!(CkksContext::new(1 << 30).is_err());
    assert!(CkksContext::new(CKKS_MAX_DEGREE * 2).is_err());
    assert!(CkksContext::new(64).is_ok());
}

/// Typical configuration mistakes: an odd RoPE head dim, heads that do not divide the width, zero heads, groups or ranks.
/// These used to be `assert!`s, i.e. panics (a `BaseException` for Python callers) inside constructors that already return `Result`.
#[test]
fn bad_layer_configuration_is_an_error() {
    assert!(RotaryEmbedding::new(7, 16, 10_000.0).is_err());
    assert!(RotaryEmbedding::new(8, 16, 10_000.0).is_ok());
    assert!(MultiHeadAttention::new(10, 3).is_err());
    assert!(MultiHeadAttention::new(10, 0).is_err());
    assert!(MultiHeadAttention::new(12, 3).is_ok());
    assert!(GroupNorm::new(3, 8, 1e-5).is_err());
    assert!(GroupNorm::new(0, 8, 1e-5).is_err());
    assert!(GroupNorm::new(2, 8, 1e-5).is_ok());
    assert!(CausalLMBlock::new(16, 3, 32).is_err());
    assert!(CausalLMBlock::new(16, 0, 32).is_err());
    assert!(GroupedQueryAttention::new(16, 4, 3).is_err());
    assert!(GroupedQueryAttention::new(16, 4, 0).is_err());
    assert!(GroupedQueryAttention::new(16, 4, 2).is_ok());
}

/// Dimension products that overflow `usize` (found by edge-value fuzzing over the Python API).
/// Each used to panic with "attempt to multiply/add with overflow" in debug builds (and wrap in release).
#[test]
fn overflowing_layer_dims_are_errors_not_panics() {
    use oxideon::adaptive::nas::{ArchitectureGenome, EvolutionaryAutoML};
    use oxideon::device::{DeviceMesh, DeviceType};
    use oxideon::geom_graphs::{GraphLaplacian, LaplacianType, TemporalInteractionGNN};
    use oxideon::nn::{
        Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d, GGUFCausalLM,
        GGUFDecoderBlock, GGUFModelConfig, GRUCell, LSTMCell, PatchEmbedding, QkNorm,
    };
    use oxideon::quantum::continuous_variable::matrix_hafnian;
    use oxideon::robotics::{ACTConfig, ActionChunkingTransformer, FlowMatchingRoboticsPolicy};
    use oxideon::tile_ir::{HardwareProfile, KernelAutoTuner, TileConfig};
    use oxideon::video::{CausalConv3d, RectifiedFlowVideoDiT, VideoDiTConfig};
    use oxideon::vision::MiniImageClassifier;
    use oxideon::world_model::{UnifiedWorldModel, WorldModelConfig};

    assert!(Conv2d::new(1 << 62, 2, (2, 2), (1, 1), (0, 0)).is_err());
    assert!(Conv1d::new(1 << 62, 2, 2, 1, 0).is_err());
    assert!(ConvTranspose2d::new(1 << 62, 2, (2, 2), (1, 1), (0, 0)).is_err());
    assert!(ConvTranspose1d::new(1 << 62, 2, 2, 1, 0).is_err());
    assert!(
        Conv3d::new(
            1 << 62,
            2,
            (2, 2, 2),
            (1, 1, 1),
            (0, 0, 0),
            true,
            DType::F32,
            Device::Cpu
        )
        .is_err()
    );
    assert!(
        ConvTranspose3d::new(
            1 << 62,
            2,
            (2, 2, 2),
            (1, 1, 1),
            (0, 0, 0),
            (0, 0, 0),
            true,
            DType::F32,
            Device::Cpu
        )
        .is_err()
    );
    assert!(LSTMCell::new(2, 1 << 62).is_err());
    assert!(GRUCell::new(2, 1 << 62).is_err());
    assert!(RotaryEmbedding::new(8, 1 << 62, 10_000.0).is_err());
    assert!(RotaryEmbedding::new(1 << 62, 16, 10_000.0).is_err());
    assert!(RotaryEmbedding::from_fn(8, 1 << 62, |_, _| (1.0, 0.0)).is_err());
    assert!(RotaryEmbedding::new_yarn(8, 1 << 62, 16, 10_000.0, 32.0, 1.0).is_err());
    assert!(SinusoidalPositionalEncoding::new(1 << 62, 8).is_err());
    assert!(SinusoidalPositionalEncoding::new(8, 1 << 62).is_err());
    assert!(PatchEmbedding::new(1 << 40, 2, 2, 2).is_err());
    assert!(DeviceMesh::new(&[1 << 32, 1 << 32], &["x", "y"], vec![]).is_err());
    assert!(DeviceMesh::homogeneous(&[1 << 32, 1 << 32], &["x", "y"], DeviceType::Cpu).is_err());
    assert!(DeviceMesh::homogeneous(&[1 << 40], &["x"], DeviceType::Cpu).is_err());
    assert!(GGUFModelConfig::basic(16, 1 << 62, 2, 1, 32, 10_000.0).is_err());
    assert!(GGUFDecoderBlock::new(8, 4, 2, 1 << 62, 16, 1e-5).is_err());
    let mut huge_heads = GGUFModelConfig::basic(16, 8, 4, 1, 32, 10_000.0).unwrap();
    huge_heads.head_dim = 1 << 62;
    assert!(GGUFCausalLM::from_config(huge_heads).is_err());
    let mut huge_mla_full = GGUFModelConfig::basic(16, 8, 4, 1, 32, 10_000.0).unwrap();
    huge_mla_full.head_dim = 1 << 62;
    huge_mla_full.kv_lora_rank = 4;
    huge_mla_full.qk_norm = QkNorm::Full;
    assert!(GGUFCausalLM::from_config(huge_mla_full).is_err());
    assert!(TemporalInteractionGNN::new(2, usize::MAX, 16, 8).is_err());
    assert!(
        ActionChunkingTransformer::new(ACTConfig {
            action_dim: 1 << 62,
            ..Default::default()
        })
        .is_err()
    );
    assert!(FlowMatchingRoboticsPolicy::new(usize::MAX, 32, 64).is_err());
    assert!(
        RectifiedFlowVideoDiT::new(VideoDiTConfig {
            in_channels: 1 << 62,
            ..Default::default()
        })
        .is_err()
    );
    assert!(CausalConv3d::new(1 << 62, 2, (2, 2, 2), (1, 1, 1)).is_err());
    assert!(MiniImageClassifier::new(1, 1 << 62, 16, 4).is_err());
    assert!(
        UnifiedWorldModel::new(WorldModelConfig {
            action_chunk: 1 << 62,
            ..Default::default()
        })
        .is_err()
    );
    assert!(
        GraphLaplacian::new(&[1.0, 2.0, 3.0, 4.0], 1 << 32, LaplacianType::Combinatorial).is_err()
    );
    assert!(matrix_hafnian(&[1.0, 2.0, 3.0, 4.0], 1 << 32).is_err());
    let mut tuner = KernelAutoTuner::new(HardwareProfile::default_cpu());
    assert!(tuner.tune_gemm(1 << 62, 2, 2).is_err());
    assert!(
        tuner
            .evaluate_candidate(64, 64, 64, &TileConfig::new(0, 32, 16, 2, 1))
            .is_none()
    );
    let mut searcher = EvolutionaryAutoML::new(4, 1, 0.2, 100_000, 42);
    assert!(searcher.search(1 << 62, 2).is_err());
    // Ordinary sizes still work.
    assert!(Conv2d::new(3, 8, (3, 3), (1, 1), (1, 1)).is_ok());
    assert!(SinusoidalPositionalEncoding::new(16, 8).is_ok());
    assert!(DeviceMesh::homogeneous(&[2, 2], &["x", "y"], DeviceType::Cpu).is_ok());
    assert!(LSTMCell::new(4, 8).is_ok());
    assert!(GGUFModelConfig::basic(16, 8, 2, 1, 32, 10_000.0).is_ok());
    let genome = ArchitectureGenome::new(vec![64, 32], 2, 2);
    assert_eq!(genome.estimate_parameters(16, 4), 13_732);
}

/// Infallible math on absurd inputs: saturated, widened, or floating-point, never a panic.
#[test]
fn absurd_math_inputs_do_not_panic() {
    use oxideon::adaptive::nas::ArchitectureGenome;
    use oxideon::physics::causal_dynamical_triangulations::CausalTriangulationEngine;
    use oxideon::physics::loop_quantum_gravity::SpinNetworkLqgEngine;
    use oxideon::science::heegaard_floer_ozsvath_szabo::HeegaardFloerKnotEngine;
    use oxideon::science::intersection_theory::ChowHirzebruchRiemannRoch;

    // Saturating estimate: "at least usize::MAX", treated as over budget by callers.
    let genome = ArchitectureGenome::new(vec![64, 32], 2, 2);
    assert_eq!(genome.estimate_parameters(usize::MAX, 2), usize::MAX);
    // Widened predicates stay exact for every input.
    assert!(!HeegaardFloerKnotEngine::satisfies_legendrian_tau_bound(
        i64::MAX,
        2,
        2
    ));
    assert!(HeegaardFloerKnotEngine::satisfies_legendrian_tau_bound(
        0, 0, 1
    ));
    assert!(!SpinNetworkLqgEngine::is_admissible_trivalent_intertwiner(
        u32::MAX,
        u32::MAX,
        u32::MAX
    ));
    assert!(SpinNetworkLqgEngine::is_admissible_trivalent_intertwiner(
        2, 2, 2
    ));
    // Exact 128-bit Euler characteristic.
    assert_eq!(
        ChowHirzebruchRiemannRoch::riemann_roch_curve(i64::MAX, 2, 2),
        -18_446_744_073_709_551_610i128
    );
    assert_eq!(ChowHirzebruchRiemannRoch::riemann_roch_curve(2, 1, 5), 4);
    // Floating-point sums round instead of overflowing.
    let cdt = CausalTriangulationEngine::new(1.0, 1.0).unwrap();
    assert!(cdt.regge_action(2, usize::MAX, 2).is_finite());
    assert!(ChowHirzebruchRiemannRoch::riemann_roch_surface(i64::MAX, 2, 2, 2).is_finite());
    // Saturating intersection number.
    assert_eq!(
        ChowHirzebruchRiemannRoch::bezout_intersection_number(&[u64::MAX, 2]),
        u64::MAX
    );
    assert_eq!(
        ChowHirzebruchRiemannRoch::bezout_intersection_number(&[2, 3, 4]),
        24
    );
}

/// `split(0)` used to loop forever pushing empty views until OOM, and
/// `chunk()` reached it via `div_ceil` on zero-length axes.
#[test]
fn split_chunk_reject_zero_sizes() -> Result<()> {
    let t = Tensor::arange(0.0, 6.0, 1.0, DType::F32)?;
    assert!(t.split(0, 0).is_err());
    assert!(t.chunk(0, 0).is_err());
    // Zero-length axis: split yields a single empty view, chunk yields one
    // empty view per requested chunk (torch parity), no hang.
    let empty = Tensor::zeros(&[0], DType::F32, &Device::Cpu)?;
    let parts = empty.chunk(2, 0)?;
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].shape(), &[0]);
    assert_eq!(parts[1].shape(), &[0]);
    let sparts = empty.split(2, 0)?;
    assert_eq!(sparts.len(), 1);
    // Ordinary behavior unchanged.
    assert_eq!(t.split(2, 0)?.len(), 3);
    assert_eq!(t.chunk(2, 0)?.len(), 2);
    Ok(())
}

/// `index_add` used to accept non-1D indices, wrap negative OOB indices via
/// `as usize`, and read `source` at unchecked offsets (deep panics).
#[test]
fn index_add_validates_indices_and_source() -> Result<()> {
    let dst = Tensor::zeros(&[2, 3], DType::F32, &Device::Cpu)?;
    let idx = Tensor::from_slice(&[0i64, 2], &[2])?;
    let src = Tensor::ones(&[2, 2], DType::F32, &Device::Cpu)?;
    let out = dst.index_add(1, &idx, &src)?;
    assert_eq!(out.to_vec::<f32>()?, vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
    let oob = Tensor::from_slice(&[0i64, 5], &[2])?;
    assert!(dst.index_add(1, &oob, &src).is_err());
    let neg_oob = Tensor::from_slice(&[0i64, -4], &[2])?;
    assert!(dst.index_add(1, &neg_oob, &src).is_err());
    let idx2d = Tensor::from_slice(&[0i64, 1], &[1, 2])?;
    assert!(dst.index_add(1, &idx2d, &src).is_err());
    let bad_src = Tensor::ones(&[2, 3], DType::F32, &Device::Cpu)?;
    assert!(dst.index_add(1, &idx, &bad_src).is_err());
    Ok(())
}

/// `scatter_add` used to skip rank checks, wrap negative OOB indices, and
/// zip-truncate `src`/`index` length mismatches into silent wrong results.
#[test]
fn scatter_add_validates_rank_shapes_and_bounds() -> Result<()> {
    let dst = Tensor::zeros(&[2, 3], DType::F32, &Device::Cpu)?;
    let index = Tensor::from_slice(&[0i64, 1, 2, 2, 1, 0], &[2, 3])?;
    let src = Tensor::ones(&[2, 3], DType::F32, &Device::Cpu)?;
    let out = dst.scatter_add(1, &index, &src)?;
    assert_eq!(out.to_vec::<f32>()?, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let flat = Tensor::from_slice(&[0i64, 1], &[2])?;
    assert!(dst.scatter_add(1, &flat, &src).is_err());
    let small = Tensor::ones(&[2, 2], DType::F32, &Device::Cpu)?;
    assert!(dst.scatter_add(1, &index, &small).is_err());
    let oob = Tensor::from_slice(&[0i64, 1, 9, 2, 1, 0], &[2, 3])?;
    assert!(dst.scatter_add(1, &oob, &src).is_err());
    let neg = Tensor::from_slice(&[0i64, 1, -4, 2, 1, 0], &[2, 3])?;
    assert!(dst.scatter_add(1, &neg, &src).is_err());
    Ok(())
}

/// `as_strided` backward used to index the grad buffer by absolute storage
/// offsets, panicking OOB (or scattering wrong grads) whenever the source was
/// itself a nonzero-offset/strided view.
#[test]
fn as_strided_backward_flows_through_offset_views() -> Result<()> {
    let x = Tensor::from_slice(&[0f32, 1., 2., 3., 4., 5.], &[6])?.requires_grad_(true);
    let n = x.narrow(0, 2, 3)?; // logical [2,3,4] living at storage 2..5
    let v = n.as_strided(&[2], &[1], 3)?; // absolute storage 3..5
    v.sum()?.backward()?;
    assert_eq!(
        x.grad().unwrap().to_vec::<f32>()?,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    );
    // A view reading storage outside the source footprint is an autograd
    // error, not a panic.
    let bad = n.as_strided(&[3], &[1], 3)?;
    assert!(bad.sum()?.backward().is_err());
    // Rank-mismatched shape/strides are rejected up front, not via assert.
    assert!(n.as_strided(&[2, 2], &[1], 0).is_err());
    Ok(())
}

/// The conv family used to divide by zero on `stride == 0`, underflow on
/// `kernel == 0` / oversized kernels, and index `shape()[0..3]` on rank<4
/// inputs; `fold2d` additionally ignored `L` and dropped channel remainders.
#[test]
fn conv_family_rejects_degenerate_params() -> Result<()> {
    let x = Tensor::zeros(&[1, 1, 5, 5], DType::F32, &Device::Cpu)?;
    assert!(x.unfold2d((3, 3), (0, 1), (0, 0), (1, 1)).is_err());
    assert!(x.unfold2d((0, 3), (1, 1), (0, 0), (1, 1)).is_err());
    assert!(x.unfold2d((7, 7), (1, 1), (0, 0), (1, 1)).is_err());
    assert!(x.unfold2d((3, 3), (1, 1), (0, 0), (1, 1)).is_ok());

    let cols = Tensor::zeros(&[1, 9, 9], DType::F32, &Device::Cpu)?;
    assert!(cols.fold2d((5, 5), (3, 3), (0, 1), (0, 0), (1, 1)).is_err());
    assert!(cols.fold2d((5, 5), (0, 3), (1, 1), (0, 0), (1, 1)).is_err());
    let bad_l = Tensor::zeros(&[1, 9, 8], DType::F32, &Device::Cpu)?;
    assert!(
        bad_l
            .fold2d((5, 5), (3, 3), (1, 1), (0, 0), (1, 1))
            .is_err()
    );
    let bad_c = Tensor::zeros(&[1, 10, 9], DType::F32, &Device::Cpu)?;
    assert!(
        bad_c
            .fold2d((5, 5), (3, 3), (1, 1), (0, 0), (1, 1))
            .is_err()
    );
    assert!(cols.fold2d((5, 5), (3, 3), (1, 1), (0, 0), (1, 1)).is_ok());

    let w = Tensor::zeros(&[2, 1, 3, 3], DType::F32, &Device::Cpu)?;
    assert!(x.conv2d(&w, None, (0, 1), (0, 0), (1, 1)).is_err());
    let w_big = Tensor::zeros(&[2, 1, 7, 7], DType::F32, &Device::Cpu)?;
    assert!(x.conv2d(&w_big, None, (1, 1), (0, 0), (1, 1)).is_err());
    assert!(x.conv2d(&w, None, (1, 1), (0, 0), (1, 1)).is_ok());

    assert!(x.max_pool2d((2, 2), (0, 2), (0, 0)).is_err());
    assert!(x.avg_pool2d((2, 2), (2, 0), (0, 0)).is_err());
    let flat = Tensor::zeros(&[25], DType::F32, &Device::Cpu)?;
    assert!(flat.max_pool2d((2, 2), (2, 2), (0, 0)).is_err());
    assert!(flat.avg_pool2d((2, 2), (2, 2), (0, 0)).is_err());
    assert!(x.max_pool2d((2, 2), (2, 2), (0, 0)).is_ok());
    assert!(x.avg_pool2d((2, 2), (2, 2), (0, 0)).is_ok());
    Ok(())
}

/// `grad()` used to index `grad_outputs` without a length check (slice OOB
/// panic when fewer grads than outputs were passed).
#[test]
fn grad_rejects_mismatched_grad_outputs() -> Result<()> {
    let a = Tensor::from_slice(&[2.0f32], &[1])?.requires_grad_(true);
    let b = Tensor::from_slice(&[3.0f32], &[1])?.requires_grad_(true);
    let g = Tensor::ones(&[1], DType::F32, &Device::Cpu)?;
    assert!(
        grad(
            &[a.clone(), b.clone()],
            std::slice::from_ref(&a),
            Some(std::slice::from_ref(&g)),
            false
        )
        .is_err()
    );
    let gos = [g.clone(), g.clone()];
    assert!(
        grad(
            &[a.clone(), b.clone()],
            std::slice::from_ref(&a),
            Some(&gos),
            false
        )
        .is_ok()
    );
    Ok(())
}

/// `triangular_solve`/`solve` used to slice `rhs` per batch of `A` without
/// checking batch counts (OOB panic on mismatch, silent truncation on extra).
#[test]
fn linalg_solvers_reject_batch_mismatch() -> Result<()> {
    let a = Tensor::from_slice(&[1f32, 0., 0., 1., 2., 0., 0., 2.], &[2, 2, 2])?;
    let rhs_flat = Tensor::from_slice(&[1f32, 1.], &[2, 1])?;
    assert!(a.triangular_solve(&rhs_flat, true, false).is_err());
    assert!(a.solve(&rhs_flat).is_err());
    let rhs = Tensor::from_slice(&[1f32, 1., 2., 2.], &[2, 2, 1])?;
    assert!(a.triangular_solve(&rhs, true, false).is_ok());
    assert!(a.solve(&rhs).is_ok());
    Ok(())
}

/// `MixtureOfDepths` used to detach router weights, compute outputs, and the
/// input into plain `f32` buffers, so nothing in the layer could ever learn.
#[test]
fn mixture_of_depths_router_learns() -> Result<()> {
    use oxideon::MixtureOfDepths;
    let m = MixtureOfDepths::new(8, 0.5)?;
    let x = Tensor::randn(&[2, 6, 8])?.requires_grad_(true);
    let (out, weights, k) = m.forward(&x)?;
    assert_eq!(k, 3);
    assert_eq!(out.shape(), &[2, 6, 8]);
    assert_eq!(weights.shape(), &[2, 6, 1]);
    out.sum()?.backward()?;
    assert!(m.router.weight.grad().is_some(), "router got no gradient");
    assert!(
        m.compute_block.weight.grad().is_some(),
        "compute block got no gradient"
    );
    assert!(x.grad().is_some(), "input got no gradient");
    // Empty sequences are an error, not a clamp(1, 0) panic.
    let empty = Tensor::zeros(&[2, 0, 8], DType::F32, &Device::Cpu)?;
    assert!(m.forward(&empty).is_err());
    Ok(())
}

/// AOT `emit_c` used to size every buffer as `in_0_len` with no null checks,
/// leak every temporary, and emit comments (broken code) for unsupported ops.
#[test]
fn aot_emit_validates_and_cleans_up() -> Result<()> {
    use oxideon::{AotCompiler, IrGraph, IrOp};
    let mut g = IrGraph::new();
    let a = g.add_input();
    let b = g.add_input();
    let s = g.push_op(IrOp::Add(a, b));
    g.set_outputs(vec![s]);
    let c = AotCompiler::emit_c(&g, "f")?;
    assert!(c.contains("if (!v_2)"));
    assert!(c.contains("free(v_2);"));
    assert!(c.contains("memcpy(out_0, v_2"));
    assert!(AotCompiler::emit_c(&g, "bad name!").is_err());
    assert!(AotCompiler::emit_c(&g, "0abc").is_err());
    let mut g2 = IrGraph::new();
    let x = g2.add_input();
    let y = g2.add_input();
    let mm = g2.push_op(IrOp::Matmul(x, y));
    g2.set_outputs(vec![mm]);
    assert!(AotCompiler::emit_c(&g2, "f").is_err());
    assert!(AotCompiler::emit_rust(&g2, "f").is_err());
    let r = AotCompiler::emit_rust(&g, "f")?;
    assert!(r.contains("v_0[i] + v_1[i]"));
    Ok(())
}

/// Tile-IR `emit_ptx`/`emit_msl` used to ignore the plan and always emit the
/// canned add_silu kernel; unlowerable plans now fail loudly instead.
#[test]
fn tile_fusion_emission_follows_plan() -> Result<()> {
    use oxideon::{KernelFusionEngine, TileKernelPlan, TileOp};
    let mut p = TileKernelPlan::new("k1", 16, 16, 16);
    p.add_input("a").add_input("b").set_output("o");
    p.push_op(TileOp::Add).push_op(TileOp::SiLU);
    let ptx = p.emit_ptx("sm_90");
    assert!(ptx.contains("ex2") && !ptx.contains("trap;"));
    let mut q = TileKernelPlan::new("k2", 8, 8, 8);
    q.add_input("xx").add_input("yy").set_output("zz");
    q.push_op(TileOp::Mul).push_op(TileOp::ReLU);
    let msl = q.emit_msl();
    assert!(msl.contains("xx[tid]") && msl.contains("zz[tid]"));
    assert!(!msl.contains("#error"));
    let sw = KernelFusionEngine::plan_fused_linear_swiglu("sw", 8, 8, 8);
    let ptx_sw = sw.emit_ptx("sm_90");
    assert!(ptx_sw.contains("sw") && ptx_sw.contains("trap;"));
    assert!(sw.emit_msl().contains("#error"));
    assert!(p.emit_ptx("sm_XX; evil").contains("trap;"));
    let mut bad = TileKernelPlan::new("evil\n.entry x", 1, 1, 1);
    bad.add_input("a").set_output("o");
    bad.push_op(TileOp::Add).push_op(TileOp::SiLU);
    assert!(bad.emit_msl().contains("#error"));
    Ok(())
}

/// Karp's minimum-cycle-mean used to index `matrix` with no consistency check
/// (OOB panic on any state the pub fields allow) and `new` multiplied `n * n`
/// unchecked (overflow panic for huge `n`).
#[test]
fn tropical_karp_validates_square_state() -> Result<()> {
    use oxideon::science::tropical_spectral::TropicalSpectralEngine;
    let good = TropicalSpectralEngine::new(2, vec![5.0, 1.0, 3.0, 6.0])?;
    assert!((good.karp_minimum_cycle_mean()? - 2.0).abs() < 1e-12);
    assert!(TropicalSpectralEngine::new(1usize << 32, Vec::new()).is_err());
    let bad = TropicalSpectralEngine {
        n: 4,
        matrix: vec![1.0; 9],
    };
    assert!(bad.karp_minimum_cycle_mean().is_err());
    assert!(bad.min_plus_matvec(&[1.0; 4]).is_err());
    Ok(())
}

/// The GPU registry is keyed by arbitrary indices, but enumeration assumed
/// `0..count()`, silently dropping non-contiguous GPUs; and a Wgpu entry
/// without the `wgpu` feature reported a nonexistent CUDA device.
#[test]
fn gpu_registry_enumerates_all_devices_honestly() {
    use oxideon::device::{GpuBackendKind, GpuDescriptor, GpuRegistry};
    let reg = GpuRegistry::new();
    // Fresh registries already hold index 0; add non-contiguous customs.
    reg.register_custom_gpu(GpuDescriptor::new(
        5,
        "gpu5",
        GpuBackendKind::Cuda,
        8192,
        64,
        32.0,
        true,
    ));
    reg.register_custom_gpu(GpuDescriptor::new(
        7,
        "gpu7",
        GpuBackendKind::Cuda,
        8192,
        64,
        32.0,
        true,
    ));
    let all = reg.all_gpus();
    assert_eq!(all.len(), 3);
    assert_eq!(
        all.iter().map(|g| g.index).collect::<Vec<_>>(),
        vec![0, 5, 7]
    );
    let wgpu = GpuDescriptor::new(0, "w", GpuBackendKind::Wgpu, 1024, 8, 1.0, false);
    #[cfg(not(feature = "wgpu"))]
    assert_eq!(wgpu.device(), Device::Cpu);
    #[cfg(feature = "wgpu")]
    let _ = wgpu.device();
}

/// GGUF metadata arrays must be homogeneous and non-nested; the writer used to
/// stamp the first element's type id and emit corrupt bytes for the rest.
#[test]
fn gguf_writer_rejects_bad_arrays() -> Result<()> {
    use oxideon::{GGUFValue, serialize_gguf};
    use std::collections::HashMap;
    let mixed = HashMap::from([(
        "k".to_string(),
        GGUFValue::Array(vec![GGUFValue::Uint32(1), GGUFValue::String("x".into())]),
    )]);
    assert!(serialize_gguf(&mixed, &[], 32).is_err());
    let nested = HashMap::from([(
        "k".to_string(),
        GGUFValue::Array(vec![GGUFValue::Array(vec![GGUFValue::Uint32(1)])]),
    )]);
    assert!(serialize_gguf(&nested, &[], 32).is_err());
    let homo = HashMap::from([(
        "k".to_string(),
        GGUFValue::Array(vec![GGUFValue::Uint32(1), GGUFValue::Uint32(2)]),
    )]);
    assert!(serialize_gguf(&homo, &[], 32).is_ok());
    Ok(())
}

/// The `vjp` closure used to accept any cotangent shape (unlike `jvp`/`hvp`),
/// failing deep inside `grad` instead of at the boundary.
#[test]
fn vjp_rejects_mismatched_cotangent() -> Result<()> {
    use oxideon::vjp;
    let x = Tensor::from_slice(&[1.0f32, 2.0], &[2])?;
    let (_y, pullback) = vjp(|t| t.mul_scalar(2.0), &x)?;
    let bad = Tensor::ones(&[3], DType::F32, &Device::Cpu)?;
    assert!(pullback(&bad).is_err());
    let good = Tensor::ones(&[2], DType::F32, &Device::Cpu)?;
    let g = pullback(&good)?;
    assert_eq!(g.to_vec::<f32>()?, vec![2.0, 2.0]);
    Ok(())
}

/// `logical_not` on numeric input used to return 1.0/0.0 in the input dtype;
/// like torch it must always return Bool.
#[test]
fn logical_not_always_returns_bool() -> Result<()> {
    let x = Tensor::from_slice(&[0.0f32, 1.0, -2.5, f32::NAN], &[4])?;
    let y = x.logical_not()?;
    assert_eq!(y.dtype(), DType::Bool);
    assert_eq!(y.to_vec::<bool>()?, vec![true, false, false, false]);
    let b = Tensor::from_slice(&[true, false], &[2])?;
    let nb = b.logical_not()?;
    assert_eq!(nb.dtype(), DType::Bool);
    assert_eq!(nb.to_vec::<bool>()?, vec![false, true]);
    // Bool <-> half-precision casts must be exact (0.0/1.0 round-trip).
    for dt in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let h = b.to_dtype(dt)?;
        assert_eq!(h.dtype(), dt);
        assert_eq!(
            h.to_dtype(DType::Bool)?.to_vec::<bool>()?,
            vec![true, false]
        );
    }
    Ok(())
}

/// `set_requires_grad(true)` on a non-float tensor used to panic via assert.
#[test]
fn set_requires_grad_rejects_non_float() -> Result<()> {
    let i = Tensor::from_slice(&[1i64, 2], &[2])?;
    assert!(i.set_requires_grad(true).is_err());
    assert!(!i.requires_grad());
    // The lenient builder form stays a no-op there (matches `from_op`).
    let i2 = Tensor::from_slice(&[1i64, 2], &[2])?.requires_grad_(true);
    assert!(!i2.requires_grad());
    let x = Tensor::from_slice(&[1.0f32, 2.0], &[2])?;
    x.set_requires_grad(true)?;
    assert!(x.requires_grad());
    Ok(())
}

/// `assign` used to retarget the destination's dtype/device to the source's;
/// like `torch.copy_` it must cast the source instead.
#[test]
fn assign_casts_source_to_destination() -> Result<()> {
    let dst = Tensor::zeros(&[2], DType::F32, &Device::Cpu)?;
    let src = Tensor::from_slice(&[1.5f64, 2.5], &[2])?;
    dst.assign(&src)?;
    assert_eq!(dst.dtype(), DType::F32);
    assert_eq!(dst.to_vec::<f32>()?, vec![1.5, 2.5]);
    let bad = Tensor::zeros(&[3], DType::F32, &Device::Cpu)?;
    assert!(dst.assign(&bad).is_err());
    Ok(())
}

/// Printing a 0-d tensor whose data cannot be read (Meta) used to panic.
#[test]
fn display_meta_scalar_does_not_panic() {
    let t = Tensor::zeros(&[], DType::F32, &Device::Meta).unwrap();
    let s = format!("{t}");
    assert!(s.contains("nan") || s.contains("NaN"));
    let d = format!("{t:?}");
    assert!(d.contains("shape=[]"));
}

/// `as_strided` with mismatched shape/stride ranks must be an Err, not a panic.
#[test]
fn as_strided_rank_mismatch_is_an_error() -> Result<()> {
    let t = Tensor::zeros(&[6], DType::F32, &Device::Cpu)?;
    let err = t.as_strided(&[2, 3], &[3], 0).unwrap_err();
    assert!(format!("{err}").contains("as_strided"));
    Ok(())
}

/// `atan2` used to reject broadcasting while every sibling binary op broadcasts.
#[test]
fn atan2_broadcasts_like_siblings() -> Result<()> {
    let y = Tensor::from_slice(&[1.0f32, 1.0], &[2])?.requires_grad_(true);
    let x = Tensor::from_slice(&[1.0f32], &[1])?.requires_grad_(true);
    let z = y.atan2(&x)?;
    assert_eq!(z.shape(), &[2]);
    for v in z.to_vec::<f32>()? {
        assert!((v - std::f32::consts::FRAC_PI_4).abs() < 1e-6);
    }
    z.sum()?.backward()?;
    assert_eq!(y.grad().unwrap().shape(), &[2]);
    assert_eq!(x.grad().unwrap().shape(), &[1]);
    Ok(())
}

/// `bincount` used to silently truncate float input through `to_vec::<i64>`.
#[test]
fn bincount_rejects_floats() -> Result<()> {
    let f = Tensor::from_slice(&[0.5f32, 1.5], &[2])?;
    assert!(f.bincount(None, 0).is_err());
    let i = Tensor::from_slice(&[0i64, 1, 1, 3], &[4])?;
    assert_eq!(i.bincount(None, 0)?.to_vec::<i64>()?, vec![1, 2, 0, 1]);
    Ok(())
}

/// `einsum` used to panic on unknown output labels and miscompute repeated ones.
#[test]
fn einsum_validates_output_subscripts() -> Result<()> {
    let v = Tensor::from_slice(&[1.0f32, 2.0], &[2])?;
    assert!(Tensor::einsum("i->ij", &[&v]).is_err());
    assert!(Tensor::einsum("i->ii", &[&v]).is_err());
    let d = Tensor::einsum("i,i->", &[&v, &v])?;
    assert_eq!(d.to_vec::<f32>()?, vec![5.0]);
    Ok(())
}

/// `conv_transpose2d` output-size arithmetic used to underflow on h==0 / big padding.
#[test]
fn conv_transpose2d_rejects_degenerate_geometry() -> Result<()> {
    let w = Tensor::ones(&[1, 1, 2, 2], DType::F32, &Device::Cpu)?;
    let empty = Tensor::zeros(&[1, 1, 0, 2], DType::F32, &Device::Cpu)?;
    assert!(
        empty
            .conv_transpose2d(&w, None, (1, 1), (0, 0), (0, 0), (1, 1))
            .is_err()
    );
    let x = Tensor::ones(&[1, 1, 2, 2], DType::F32, &Device::Cpu)?;
    assert!(
        x.conv_transpose2d(&w, None, (1, 1), (4, 4), (0, 0), (1, 1))
            .is_err()
    );
    let y = x.conv_transpose2d(&w, None, (1, 1), (0, 0), (0, 0), (1, 1))?;
    assert_eq!(y.shape(), &[1, 1, 3, 3]);
    Ok(())
}

/// `tile` used to reject `reps` longer than `ndim`, unlike `torch.tile`.
#[test]
fn tile_promotes_rank() -> Result<()> {
    let v = Tensor::from_slice(&[1.0f32, 2.0], &[2])?;
    let t = v.tile(&[2, 3])?;
    assert_eq!(t.shape(), &[2, 6]);
    assert_eq!(
        t.to_vec::<f32>()?,
        vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    );
    let m = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2])?;
    let t2 = m.tile(&[3])?;
    assert_eq!(t2.shape(), &[2, 6]);
    Ok(())
}

/// `nan_to_num` used to drop the autograd graph; grads must flow through finite entries.
#[test]
fn nan_to_num_keeps_autograd() -> Result<()> {
    let x = Tensor::from_slice(&[1.0f32, f32::NAN, f32::INFINITY], &[3])?.requires_grad_(true);
    let y = x.nan_to_num(0.0, 5.0, -5.0)?;
    assert_eq!(y.to_vec::<f32>()?, vec![1.0, 0.0, 5.0]);
    y.sum()?.backward()?;
    assert_eq!(x.grad().unwrap().to_vec::<f32>()?, vec![1.0, 0.0, 0.0]);
    Ok(())
}

/// `triangular_solve` used to divide by a zero diagonal, yielding silent inf/NaN.
#[test]
fn triangular_solve_rejects_singular() -> Result<()> {
    let a = Tensor::from_slice(&[1.0f32, 0.0, 1.0, 0.0], &[2, 2])?;
    let b = Tensor::from_slice(&[1.0f32, 1.0], &[2, 1])?;
    assert!(a.triangular_solve(&b, true, false).is_err());
    assert!(a.triangular_solve(&b, false, false).is_err());
    let ok = Tensor::from_slice(&[2.0f32, 0.0, 1.0, 2.0], &[2, 2])?;
    let x = ok.triangular_solve(&b, true, false)?;
    assert_eq!(x.to_vec::<f32>()?, vec![0.5, 0.25]);
    Ok(())
}

/// `norm_p` used to add 1e-12, so a zero vector had norm 1e-6 instead of 0.
#[test]
fn norm_p_has_no_epsilon_floor() -> Result<()> {
    let z = Tensor::zeros(&[4], DType::F32, &Device::Cpu)?;
    assert_eq!(z.norm_p(2.0, &[0], false)?.to_vec::<f32>()?, vec![0.0]);
    assert_eq!(z.norm_p(3.0, &[0], false)?.to_vec::<f32>()?, vec![0.0]);
    assert_eq!(z.norm_p(0.0, &[0], false)?.to_vec::<f32>()?, vec![0.0]);
    let v = Tensor::from_slice(&[3.0f32, 4.0], &[2])?;
    assert_eq!(v.norm_p(2.0, &[0], false)?.to_vec::<f32>()?, vec![5.0]);
    Ok(())
}

/// `compute_gae` used to `assert_eq!` length equality, panicking on bad input.
#[test]
fn compute_gae_rejects_length_mismatch() {
    assert!(compute_gae(&[1.0], &[1.0, 2.0], &[false], 0.0, 0.99, 0.95).is_err());
    assert!(compute_gae(&[1.0], &[1.0], &[false, true], 0.0, 0.99, 0.95).is_err());
    let (adv, ret) =
        compute_gae(&[1.0, 0.5], &[0.8, 0.9], &[false, true], 0.0, 0.99, 0.95).unwrap();
    assert_eq!(adv.len(), 2);
    assert_eq!(ret.len(), 2);
}

/// `info_nce_loss` with temperature 0 used to silently produce Inf/NaN.
#[test]
fn info_nce_rejects_nonpositive_temperature() -> Result<()> {
    let q = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2])?;
    let k = q.clone();
    assert!(info_nce_loss(&q, &k, 0.0).is_err());
    assert!(info_nce_loss(&q, &k, -1.0).is_err());
    assert!(info_nce_loss(&q, &k, f64::NAN).is_err());
    assert!(info_nce_loss(&q, &k, 0.5)?.to_vec::<f32>()?[0].is_finite());
    Ok(())
}

/// A NaN objective used to permanently poison `Study::best_trial`.
#[test]
fn study_best_ignores_nan_objectives() -> Result<()> {
    let space =
        ParamSpace::new().add_discrete("x", vec![ParamValue::Float(0.0), ParamValue::Float(1.0)]);
    let study = Study::grid_search(&space, 2, |cfg| {
        let x = cfg["x"].as_float();
        Ok(if x == 0.0 { f64::NAN } else { 1.0 })
    })?;
    assert_eq!(study.trials.len(), 2);
    assert_eq!(study.best_trial.unwrap().objective, 1.0);
    Ok(())
}

/// `Study::random_search` used to `with_capacity(num_trials)` blindly (abort on absurd counts).
#[test]
fn study_random_search_rejects_absurd_trial_counts() -> Result<()> {
    let space = ParamSpace::new().add_discrete("x", vec![ParamValue::Float(0.0)]);
    assert!(Study::random_search(&space, usize::MAX, |_| Ok(1.0)).is_err());
    let study = Study::random_search(&space, 4, |_| Ok(1.0))?;
    assert_eq!(study.trials.len(), 4);
    Ok(())
}

/// `encode_batch_padded` used to `vec![PAD; batch * max_len]` blindly (abort on giant `max_len`).
#[test]
fn bpe_encode_batch_padded_rejects_giant_max_len() -> Result<()> {
    let tok = BpeTokenizer::new_byte_level();
    assert!(
        tok.encode_batch_padded(&["hello"], Some(usize::MAX))
            .is_err()
    );
    assert!(
        tok.encode_batch_padded(&["hello", "world"], Some(1 << 40))
            .is_err()
    );
    // Ordinary padding still works.
    let (ids, mask) = tok.encode_batch_padded(&["hi", "hello"], None)?;
    assert_eq!(ids.shape(), &[2, 5]);
    assert_eq!(mask.shape(), &[2, 5]);
    Ok(())
}

/// Trailing data bytes used to sail through `check_layout` and then index `scales` out of bounds.
#[test]
fn quantized_tensor_rejects_trailing_data_bytes() -> Result<()> {
    let t = Tensor::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4])?;
    let mut q = QuantizedTensor::quantize(&t, QuantScheme::DEFAULT_Q8_0)?;
    q.data.extend_from_slice(&[0u8; 32]); // one bogus extra block
    assert!(q.dequantize(DType::F32).is_err());
    Ok(())
}

/// `check_layout` used to demand full-block Q4 bytes, rejecting `quantize`'s own partial-block output.
#[test]
fn quantized_q4_partial_block_roundtrips() -> Result<()> {
    let vals: Vec<f32> = (0..35).map(|i| i as f32 * 0.5).collect();
    let t = Tensor::from_slice(&vals, &[35])?;
    let q = QuantizedTensor::quantize(&t, QuantScheme::DEFAULT_Q4_0)?;
    let back = q.dequantize(DType::F32)?;
    assert_eq!(back.shape(), &[35]);
    let v = back.to_vec::<f32>()?;
    assert!(v.iter().all(|x| x.is_finite()));
    assert!((v[0] - vals[0]).abs() < 2.0);
    Ok(())
}

/// `compression_ratio_f32` used to `product()` the (public) shape, panicking on overflow.
#[test]
fn quantized_compression_ratio_survives_absurd_shapes() -> Result<()> {
    let t = Tensor::zeros(&[4], DType::F32, &Device::Cpu)?;
    let mut q = QuantizedTensor::quantize(&t, QuantScheme::DEFAULT_Q8_0)?;
    q.shape = vec![usize::MAX, 2];
    assert!(q.compression_ratio_f32().is_finite());
    Ok(())
}

/// `unpack` used to `with_capacity(num_elements)` on the public count (abort when tampered).
#[test]
fn ternary_unpack_validates_element_count() -> Result<()> {
    let t = Tensor::from_slice::<f32>(&[1.0, -2.0, 3.0], &[3])?;
    let mut p = TernaryPackedTensor::pack(&t, 1.0)?;
    p.num_elements = usize::MAX;
    assert!(p.unpack().is_err());
    // Untouched packing still round-trips.
    let p2 = TernaryPackedTensor::pack(&t, 1.0)?;
    assert_eq!(p2.unpack()?.shape(), &[3]);
    Ok(())
}

/// `SparseCoo::transpose` used to overwrite (not sum) duplicate hand-built entries.
#[test]
fn sparse_coo_transpose_sums_duplicate_entries() -> Result<()> {
    let coo = SparseCoo {
        indices: vec![(0, 0), (0, 0)],
        values: vec![1.0, 2.0],
        shape: [2, 2],
    };
    // to_dense sums duplicates; transpose must agree with it.
    let direct = coo.to_dense(DType::F64, &Device::Cpu)?.to_vec::<f64>()?;
    assert_eq!(direct, vec![3.0, 0.0, 0.0, 0.0]);
    let via_t = coo
        .transpose()?
        .to_dense(DType::F64, &Device::Cpu)?
        .to_vec::<f64>()?;
    assert_eq!(via_t, direct);
    Ok(())
}

/// `analyze_graph` used to multiply `numel * dtype_bytes` raw (overflow panic on meta-scale shapes).
#[test]
fn analyze_graph_saturates_on_meta_scale_tensors() -> Result<()> {
    // requires_grad so the ops actually record graph nodes (from_op skips
    // grad-free subgraphs); meta shapes need no storage.
    let big = Tensor::zeros(&[usize::MAX], DType::F32, &Device::Meta)?.requires_grad_(true);
    let doubled = big.add(&big)?;
    let stats = analyze_graph(&[doubled]);
    assert_eq!(stats.leaf_nodes, 1);
    assert_eq!(stats.op_nodes, 1);
    assert_eq!(stats.total_leaf_bytes, usize::MAX);
    assert_eq!(stats.total_activation_bytes, usize::MAX);
    // Matmul FLOP estimate: 2 * out_elems * k must saturate, not overflow.
    let a = Tensor::zeros(&[1 << 40, 1 << 23], DType::F32, &Device::Meta)?.requires_grad_(true);
    let b = Tensor::zeros(&[1 << 23, 1], DType::F32, &Device::Meta)?.requires_grad_(true);
    let m = a.matmul(&b)?;
    let mstats = analyze_graph(&[m]);
    assert_eq!(mstats.estimated_flops, u64::MAX);
    Ok(())
}

/// `stft` used to add `start + n_fft` / `start + hop_length` raw (overflow panic on giant hops).
#[test]
fn stft_survives_giant_hop_length() -> Result<()> {
    let x = Tensor::randn(&[64])?;
    let s = x.stft(16, usize::MAX, false)?;
    assert_eq!(s.shape(), &[1, 1, 16, 2]); // exactly one frame
    Ok(())
}

/// `confusion_matrix`/`precision_recall_f1` used to allocate `num_classes^2` blindly (abort).
#[test]
fn confusion_matrix_rejects_absurd_class_counts() -> Result<()> {
    assert!(confusion_matrix(&[0], &[0], usize::MAX).is_err());
    assert!(precision_recall_f1(&[0], &[0], usize::MAX).is_err());
    // Ordinary sizes still work.
    let cm = confusion_matrix(&[0, 1, 1], &[0, 0, 1], 2)?;
    assert_eq!(cm, vec![vec![1, 1], vec![0, 1]]);
    let (p, r, f1) = precision_recall_f1(&[0, 1, 1], &[0, 0, 1], 2)?;
    assert!((p - 0.75).abs() < 1e-12);
    assert!((r - 0.75).abs() < 1e-12);
    assert!((f1 - 2.0 / 3.0).abs() < 1e-12);
    Ok(())
}

/// `auroc` totals used to count untruncated `targets` while scores were zipped (wrong on mismatch).
#[test]
fn auroc_scores_truncated_pairs_consistently() {
    // Zip truncates to 2 perfectly-ranked pairs: must score 1.0, not 1/4.
    assert!((auroc(&[0.9, 0.1], &[1, 0, 1, 0]) - 1.0).abs() < 1e-12);
    assert!((auroc(&[0.9, 0.1], &[1, 0]) - 1.0).abs() < 1e-12);
    // Single-class (over pairs) still scores 0.5.
    assert!((auroc(&[0.9], &[1, 1]) - 0.5).abs() < 1e-12);
}

/// `smooth_grad` used to loop `num_samples` times unbounded (hang on absurd counts).
#[test]
fn smooth_grad_rejects_absurd_sample_counts() -> Result<()> {
    let x = Tensor::randn(&[4])?;
    assert!(smooth_grad(|t| t.square()?.sum(), &x, usize::MAX, 0.1, false).is_err());
    let g = smooth_grad(|t| t.square()?.sum(), &x, 2, 0.1, false)?;
    assert_eq!(g.shape(), &[4]);
    Ok(())
}

/// `integrated_gradients` used to loop `steps` times unbounded (hang on absurd counts).
#[test]
fn integrated_gradients_rejects_absurd_step_counts() -> Result<()> {
    let x = Tensor::from_slice::<f32>(&[1.0, 2.0], &[2])?;
    assert!(integrated_gradients(|t| t.square()?.sum(), &x, None, usize::MAX).is_err());
    let ig = integrated_gradients(|t| t.square()?.sum(), &x, None, 4)?;
    assert_eq!(ig.shape(), &[2]);
    Ok(())
}

/// Tampered RVQ stage dims used to index the centroid out of bounds (quantize path).
#[test]
fn rvq_quantize_rejects_dim_mismatched_stages() -> Result<()> {
    let mut q = ResidualVectorQuantizer::new(RVQConfig::new(2, 4, 8))?;
    q.stages[0].dim = 4;
    let lat = Tensor::randn(&[1, 2, 8])?;
    assert!(q.quantize(&lat).is_err());
    Ok(())
}

/// Tampered RVQ stage counts used to index the codes table out of bounds.
#[test]
fn rvq_rejects_stage_count_mismatch() -> Result<()> {
    let mut q = ResidualVectorQuantizer::new(RVQConfig::new(2, 4, 8))?;
    q.stages.push(q.stages[0].clone());
    let lat = Tensor::randn(&[1, 2, 8])?;
    assert!(q.quantize(&lat).is_err());
    let codes = vec![vec![0usize; 2], vec![0usize; 2]];
    assert!(q.decode_codes(&codes).is_err());
    // Untouched quantizer still round-trips.
    let q2 = ResidualVectorQuantizer::new(RVQConfig::new(2, 4, 8))?;
    assert_eq!(q2.quantize(&lat)?.quantized.shape(), &[1, 2, 8]);
    Ok(())
}

/// Tampered RoPE partitions used to index across head boundaries (panic or cross-head corruption).
#[test]
fn rope3d_forward_validates_partitions() -> Result<()> {
    let mut rope = RotaryEmbedding3D::new(8, 10000.0)?;
    rope.dim_t = 8;
    rope.dim_h = 8;
    rope.dim_w = 8;
    let x = Tensor::randn(&[1, 8, 2, 8])?;
    assert!(rope.forward(&x, 2, 2, 2).is_err());
    // Untouched embedding still works.
    let rope2 = RotaryEmbedding3D::new(8, 10000.0)?;
    assert_eq!(rope2.forward(&x, 2, 2, 2)?.shape(), &[1, 8, 2, 8]);
    Ok(())
}

/// `continuous_timestep_embedding` used to allocate `batch * dim` blindly (abort on giant `dim`).
#[test]
fn timestep_embedding_rejects_absurd_dim() -> Result<()> {
    let t = Tensor::from_slice::<f32>(&[0.5], &[1])?;
    assert!(continuous_timestep_embedding(&t, usize::MAX).is_err());
    let e = continuous_timestep_embedding(&t, 8)?;
    assert_eq!(e.shape(), &[1, 8]);
    Ok(())
}

/// `sample_euler` used to loop `num_steps` unbounded (hang on absurd counts).
#[test]
fn flow_dit_euler_rejects_absurd_step_counts() -> Result<()> {
    let dit = RectifiedFlowVideoDiT::new(VideoDiTConfig::default())?;
    let noise = Tensor::randn(&[1, 4, 2, 4, 4])?;
    let cond = Tensor::randn(&[1, 2, 32])?;
    assert!(dit.sample_euler(&noise, &cond, usize::MAX).is_err());
    let out = dit.sample_euler(&noise, &cond, 1)?;
    assert_eq!(out.shape(), &[1, 4, 2, 4, 4]);
    Ok(())
}

/// `MiniImageDataset::generate` used to multiply its dims blindly (overflow panic / abort).
#[test]
fn mini_dataset_generate_rejects_absurd_sizes() -> Result<()> {
    use oxideon::vision::MiniImageDataset;
    assert!(MiniImageDataset::generate(1 << 40, 1 << 20, 16, 16, 1).is_err());
    let ds = MiniImageDataset::generate(8, 1, 4, 4, 1)?;
    assert_eq!(ds.len(), 8);
    Ok(())
}

/// `split`/`batch` used to trust the public layout fields (slicing panic when tampered).
#[test]
fn mini_dataset_split_batch_validate_layout() -> Result<()> {
    use oxideon::vision::MiniImageDataset;
    let mut ds = MiniImageDataset::generate(8, 1, 4, 4, 1)?;
    ds.channels = 1000;
    assert!(ds.split(0.5).is_err());
    assert!(ds.batch(0, 4).is_err());
    // Untouched dataset still splits and batches.
    let ds2 = MiniImageDataset::generate(8, 1, 4, 4, 1)?;
    let (train, val) = ds2.split(0.5)?;
    assert_eq!(train.len() + val.len(), 8);
    assert_eq!(ds2.batch(0, 4)?.0.shape(), &[4, 1, 4, 4]);
    Ok(())
}

/// `MiniImageDataset::batch` used to multiply/add indices blindly (overflow panic).
#[test]
fn mini_dataset_batch_survives_huge_indices() -> Result<()> {
    use oxideon::vision::MiniImageDataset;
    let ds = MiniImageDataset::generate(8, 1, 4, 4, 1)?;
    assert!(ds.batch(1 << 32, 1 << 32).is_err());
    assert!(ds.batch(1, usize::MAX).is_err());
    // Whole-dataset batch via huge batch_size keeps working.
    assert_eq!(ds.batch(0, usize::MAX)?.0.shape(), &[8, 1, 4, 4]);
    Ok(())
}

/// `MiniImageClassifier::forward` used to index `shape()[0]` before validating rank (scalar panic).
#[test]
fn classifier_forward_rejects_scalar() -> Result<()> {
    use oxideon::nn::Module;
    use oxideon::vision::MiniImageClassifier;
    let m = MiniImageClassifier::standard()?;
    let s = Tensor::scalar(1.0f32)?;
    assert!(m.forward(&s).is_err());
    Ok(())
}

/// `MiniResNet::forward` used to index `shape()[0]` before validating rank (scalar panic).
#[test]
fn resnet_forward_rejects_scalar() -> Result<()> {
    use oxideon::nn::Module;
    use oxideon::vision::MiniResNet;
    let m = MiniResNet::standard()?;
    let s = Tensor::scalar(1.0f32)?;
    assert!(m.forward(&s).is_err());
    Ok(())
}

/// `MiniAutoEncoder::encode` used to index `shape()[0]` before validating rank (scalar panic).
#[test]
fn autoencoder_encode_rejects_scalar() -> Result<()> {
    use oxideon::vision::MiniAutoEncoder;
    let m = MiniAutoEncoder::standard()?;
    let s = Tensor::scalar(1.0f32)?;
    assert!(m.encode(&s).is_err());
    Ok(())
}

/// `MiniAutoEncoder::decode` used to index `shape()[0]` before validating rank (scalar panic).
#[test]
fn autoencoder_decode_rejects_scalar() -> Result<()> {
    use oxideon::vision::MiniAutoEncoder;
    let m = MiniAutoEncoder::standard()?;
    let s = Tensor::scalar(1.0f32)?;
    assert!(m.decode(&s).is_err());
    Ok(())
}

/// `MiniImageClassifier::predict` used the tamperable `num_classes` to slice logits (panic).
#[test]
fn classifier_predict_uses_logit_width() -> Result<()> {
    use oxideon::vision::MiniImageClassifier;
    let mut m = MiniImageClassifier::standard()?;
    m.num_classes = 1000;
    let images = Tensor::randn(&[2, 1, 16, 16])?;
    assert_eq!(m.predict(&images)?.len(), 2);
    Ok(())
}

/// `MiniResNet::predict` used the tamperable `num_classes` to slice logits (panic).
#[test]
fn resnet_predict_uses_logit_width() -> Result<()> {
    use oxideon::vision::MiniResNet;
    let mut m = MiniResNet::standard()?;
    m.num_classes = 1000;
    let images = Tensor::randn(&[2, 1, 16, 16])?;
    assert_eq!(m.predict(&images)?.len(), 2);
    Ok(())
}

/// `MiniImageClassifier::fit` used to `with_capacity(epochs)` blindly (abort on absurd counts).
#[test]
fn classifier_fit_rejects_absurd_epochs() -> Result<()> {
    use oxideon::vision::{MiniImageClassifier, MiniImageDataset};
    let mut m = MiniImageClassifier::standard()?;
    let train = MiniImageDataset::generate(8, 1, 16, 16, 1)?;
    assert!(m.fit(&train, None, usize::MAX, 4, 0.01).is_err());
    Ok(())
}

/// `MiniAutoEncoder::fit` used to `with_capacity(epochs)` blindly (abort on absurd counts).
#[test]
fn autoencoder_fit_rejects_absurd_epochs() -> Result<()> {
    use oxideon::vision::{MiniAutoEncoder, MiniImageDataset};
    let mut m = MiniAutoEncoder::standard()?;
    let train = MiniImageDataset::generate(8, 1, 16, 16, 1)?;
    assert!(m.fit(&train, None, usize::MAX, 4, 0.01).is_err());
    Ok(())
}

/// `MiniResNet::fit` used to `with_capacity(epochs)` blindly (abort on absurd counts).
#[test]
fn resnet_fit_rejects_absurd_epochs() -> Result<()> {
    use oxideon::vision::{MiniImageDataset, MiniResNet};
    let mut m = MiniResNet::standard()?;
    let train = MiniImageDataset::generate(8, 1, 16, 16, 1)?;
    assert!(m.fit(&train, None, usize::MAX, 4, 0.01).is_err());
    Ok(())
}

/// `MiniAutoEncoder::interpolate` used to reserve/loop `steps` blindly (abort on absurd counts).
#[test]
fn autoencoder_interpolate_rejects_absurd_steps() -> Result<()> {
    use oxideon::vision::MiniAutoEncoder;
    let m = MiniAutoEncoder::standard()?;
    let img = Tensor::randn(&[1, 1, 16, 16])?;
    assert!(m.interpolate(&img, &img, usize::MAX).is_err());
    assert_eq!(m.interpolate(&img, &img, 2)?.shape(), &[2, 1, 16, 16]);
    Ok(())
}

/// `affine_grid` used to multiply `h * w` blindly (overflow panic on giant sizes).
#[test]
fn affine_grid_rejects_absurd_size() -> Result<()> {
    use oxideon::vision_ops::affine_grid;
    let theta = Tensor::randn(&[1, 2, 3])?;
    assert!(affine_grid(&theta, [1, 1, 1 << 32, 1 << 32], false).is_err());
    assert_eq!(
        affine_grid(&theta, [1, 1, 4, 4], false)?.shape(),
        &[1, 4, 4, 2]
    );
    Ok(())
}

/// `roi_align` used to loop `sampling_ratio^2` unbounded (hang on absurd counts).
#[test]
fn roi_align_rejects_absurd_sampling() -> Result<()> {
    use oxideon::vision_ops::roi_align;
    let input = Tensor::randn(&[1, 1, 4, 4])?;
    let rois = Tensor::from_slice::<f32>(&[0.0, 0.0, 0.0, 2.0, 2.0], &[1, 5])?;
    assert!(roi_align(&input, &rois, [2, 2], 1.0, usize::MAX).is_err());
    assert_eq!(
        roi_align(&input, &rois, [2, 2], 1.0, 2)?.shape(),
        &[1, 1, 2, 2]
    );
    Ok(())
}

/// `roi_align` used to multiply its output dims blindly (overflow panic on giant sizes).
#[test]
fn roi_align_rejects_absurd_output() -> Result<()> {
    use oxideon::vision_ops::roi_align;
    let input = Tensor::randn(&[1, 1, 4, 4])?;
    let rois = Tensor::from_slice::<f32>(&[0.0, 0.0, 0.0, 2.0, 2.0], &[1, 5])?;
    assert!(roi_align(&input, &rois, [1 << 32, 1 << 32], 1.0, 2).is_err());
    Ok(())
}

/// `roi_align` used to accept NaN coordinates (silent batch-0 pooling, or a `clamp` panic).
#[test]
fn roi_align_rejects_nan_coords() -> Result<()> {
    use oxideon::vision_ops::roi_align;
    let input = Tensor::randn(&[1, 1, 4, 4])?;
    let nan_batch = Tensor::from_slice::<f32>(&[f32::NAN, 0.0, 0.0, 2.0, 2.0], &[1, 5])?;
    assert!(roi_align(&input, &nan_batch, [2, 2], 1.0, 2).is_err());
    let nan_coord = Tensor::from_slice::<f32>(&[0.0, f32::NAN, 0.0, 2.0, 2.0], &[1, 5])?;
    assert!(roi_align(&input, &nan_coord, [2, 2], 1.0, 2).is_err());
    Ok(())
}

/// `DDPMScheduler::new` used to `assert!` on zero (panic) and allocate blindly (abort on giant).
#[test]
fn ddpm_new_rejects_degenerate_counts() -> Result<()> {
    assert!(DDPMScheduler::new(0, BetaSchedule::default()).is_err());
    assert!(DDPMScheduler::new(usize::MAX, BetaSchedule::default()).is_err());
    assert!(DDPMScheduler::new(100, BetaSchedule::default()).is_ok());
    assert!(DDPMScheduler::new(100, BetaSchedule::Cosine { s: 0.008 }).is_ok());
    Ok(())
}

/// Tampered scheduler tables used to be indexed out of bounds by stepping methods.
#[test]
fn ddpm_methods_reject_inconsistent_tables() -> Result<()> {
    let mut sched = DDPMScheduler::new(10, BetaSchedule::default())?;
    sched.alphas.truncate(5);
    let x = Tensor::randn(&[1, 4])?;
    let n = Tensor::randn(&[1, 4])?;
    assert!(sched.step_ddpm(&n, 8, &x).is_err());
    Ok(())
}

/// An emptied schedule used to underflow `num_train_timesteps - 1` in per-token noising.
#[test]
fn forcing_add_noise_rejects_empty_schedule() -> Result<()> {
    let mut forcing = DiffusionForcingScheduler::new(10, BetaSchedule::default())?;
    forcing.ddpm.num_train_timesteps = 0;
    forcing.ddpm.betas.clear();
    forcing.ddpm.alphas.clear();
    forcing.ddpm.alphas_cumprod.clear();
    forcing.ddpm.sqrt_alphas_cumprod.clear();
    forcing.ddpm.sqrt_one_minus_alphas_cumprod.clear();
    forcing.ddpm.posterior_variance.clear();
    let seq = Tensor::randn(&[1, 2, 4])?;
    assert!(forcing.add_noise_per_token(&seq, &[1, 1]).is_err());
    Ok(())
}

/// `timestep_embedding` used to multiply `len * dim` blindly (overflow panic / abort).
#[test]
fn ddpm_timestep_embedding_rejects_absurd_dim() -> Result<()> {
    assert!(timestep_embedding(&[1, 2], usize::MAX).is_err());
    assert_eq!(timestep_embedding(&[1, 2], 8)?.shape(), &[2, 8]);
    Ok(())
}

/// `try_build_sigmas` used to loop/push `num_steps` unbounded (hang on absurd counts).
#[test]
fn edm_sigmas_reject_absurd_steps() -> Result<()> {
    let cfg = EDMConfig {
        num_steps: usize::MAX,
        ..EDMConfig::default()
    };
    assert!(cfg.try_build_sigmas().is_err());
    assert_eq!(EDMConfig::default().try_build_sigmas()?.len(), 16);
    Ok(())
}

/// Rectified-flow `sample_euler` used to loop `num_steps` unbounded (hang on absurd counts).
#[test]
fn rectified_euler_rejects_absurd_steps() -> Result<()> {
    let cfg = RectifiedFlowConfig {
        num_steps: usize::MAX,
        ..RectifiedFlowConfig::default()
    };
    let x0 = Tensor::randn(&[1, 4])?;
    assert!(RectifiedFlowEngine::sample_euler(&x0, &cfg, |v, _, _| Ok(v.clone())).is_err());
    let cfg2 = RectifiedFlowConfig {
        num_steps: 2,
        ..RectifiedFlowConfig::default()
    };
    assert_eq!(
        RectifiedFlowEngine::sample_euler(&x0, &cfg2, |v, _, _| Ok(v.clone()))?.shape(),
        &[1, 4]
    );
    Ok(())
}

/// `BlockAllocator::new` used to reserve freelists blindly (abort on giant counts).
#[test]
fn block_allocator_rejects_absurd_block_counts() -> Result<()> {
    assert!(BlockAllocator::new(usize::MAX, 16).is_err());
    assert!(BlockAllocator::new(4, 16).is_ok());
    // Zero block_size still constructs (rejected downstream instead).
    assert!(BlockAllocator::new(4, 0).is_ok());
    Ok(())
}

/// Tampered cache dims used to overflow the token stride in `write_token`.
#[test]
fn paged_kv_write_rejects_tampered_dims() -> Result<()> {
    let mut cache = PagedKVCache::new(4, 16, 2, 8, DType::F32)?;
    cache.num_heads = usize::MAX;
    let k = Tensor::randn(&[16])?;
    let v = Tensor::randn(&[16])?;
    assert!(cache.write_token(0, 0, &k, &v).is_err());
    Ok(())
}

/// Tampered cache dims used to silently mis-stride `paged_attention` reads.
#[test]
fn paged_attention_rejects_tampered_cache_dims() -> Result<()> {
    let mut cache = PagedKVCache::new(4, 16, 2, 8, DType::F32)?;
    cache.num_heads = 4;
    let q = Tensor::randn(&[1, 4, 8])?;
    let mut bt = BlockTable::new();
    bt.append_block(0);
    assert!(paged_attention(&q, &cache, &[bt], &[4], 1.0).is_err());
    // Untouched cache still attends.
    let cache2 = PagedKVCache::new(4, 16, 2, 8, DType::F32)?;
    let q2 = Tensor::randn(&[1, 2, 8])?;
    let mut bt2 = BlockTable::new();
    bt2.append_block(0);
    assert_eq!(
        paged_attention(&q2, &cache2, &[bt2], &[4], 1.0)?.shape(),
        &[1, 2, 8]
    );
    Ok(())
}

/// `Bilinear::forward_pair` used to reject the documented 1-D/ND input ranks.
#[test]
fn bilinear_accepts_documented_ranks() -> Result<()> {
    let b = Bilinear::new(3, 4, 2, false)?;
    let x1 = Tensor::randn(&[5, 3])?;
    let x2 = Tensor::randn(&[5, 4])?;
    assert_eq!(b.forward_pair(&x1, &x2)?.shape(), &[5, 2]);
    let x1d = Tensor::randn(&[3])?;
    let y1d = Tensor::randn(&[4])?;
    assert_eq!(b.forward_pair(&x1d, &y1d)?.shape(), &[2]);
    let xnd = Tensor::randn(&[2, 5, 3])?;
    let ynd = Tensor::randn(&[5, 4])?;
    assert_eq!(b.forward_pair(&xnd, &ynd)?.shape(), &[2, 5, 2]);
    let bad = Tensor::randn(&[5, 7])?;
    assert!(b.forward_pair(&x1, &bad).is_err());
    Ok(())
}

/// Backend kernels used to panic or zip-truncate on invalid axes / mismatched shapes.
#[test]
fn backend_kernels_validate_shapes() -> Result<()> {
    use oxideon::Layout;
    use oxideon::backend::{CmpOp, CpuStorage, ReduceOp};
    let t = Tensor::zeros(&[2, 3], DType::F32, &Device::Cpu)?;
    assert!(t.sum_axes(&[5], false).is_err());
    let l2 = Layout::contiguous(vec![2]);
    let l3 = Layout::contiguous(vec![3]);
    let a = CpuStorage::F32(vec![1.0, 2.0]);
    let b = CpuStorage::F32(vec![1.0, 2.0, 3.0]);
    assert!(a.reduce_op(&l2, &[7], ReduceOp::Sum).is_err());
    assert!(a.cmp_op(&l2, &b, &l3, CmpOp::Eq).is_err());
    let c = CpuStorage::Bool(vec![true, false, true]);
    assert!(CpuStorage::where_cond(&c, &l3, &a, &l2, &a, &l2).is_err());
    Ok(())
}

/// Bool `div`/`pow` used to silently return the lhs operand.
#[test]
fn bool_div_pow_are_errors() -> Result<()> {
    let a = Tensor::from_slice(&[true, false], &[2])?;
    let b = Tensor::from_slice(&[true, true], &[2])?;
    assert!(a.div(&b).is_err());
    assert!(a.pow(&b).is_err());
    // Boolean logic ops keep working.
    assert_eq!(a.mul(&b)?.to_vec::<bool>()?, vec![true, false]);
    Ok(())
}

/// `alloc_f32` used to panic (or wrap accounting) on absurd sizes.
#[test]
fn allocator_rejects_absurd_sizes() -> Result<()> {
    use oxideon::CachingAllocator;
    let alloc = CachingAllocator::new();
    assert!(alloc.alloc_f32(usize::MAX).is_err());
    assert!(alloc.alloc_f32(oxideon::limits::MAX_ELEMENTS + 1).is_err());
    assert_eq!(CachingAllocator::bucket_capacity(usize::MAX), usize::MAX);
    let buf = alloc.alloc_f32(100)?;
    assert_eq!(buf.len(), 100);
    Ok(())
}

/// Re-registering a tensor id used to double-count device memory.
#[test]
fn tiered_reregister_does_not_double_count() -> Result<()> {
    use oxideon::{MemoryPriority, TieredMemoryManager};
    let mut mgr = TieredMemoryManager::new(0.9);
    mgr.register_device_capacity(Device::Cpu, 1 << 30);
    mgr.register_tensor(7, 100, Device::Cpu, MemoryPriority::Normal)?;
    mgr.register_tensor(7, 100, Device::Cpu, MemoryPriority::Normal)?;
    assert_eq!(mgr.get_device_usage(&Device::Cpu).0, 100);
    Ok(())
}

/// `with_device` used to leak the scoped device when the closure panicked.
#[test]
fn with_device_restores_after_panic() {
    use oxideon::{get_default_device, set_default_device, with_device};
    set_default_device(Device::Cpu);
    let r = std::panic::catch_unwind(|| {
        with_device(Device::Cuda(0), || panic!("boom"));
    });
    assert!(r.is_err());
    assert_eq!(get_default_device(), Device::Cpu);
}

/// `StreamEvent::synchronize` used to hot-spin forever; the timeout variant errors.
#[test]
fn stream_event_timeout_errors() {
    use oxideon::backend::StreamEvent;
    let ev = StreamEvent::new(1);
    assert!(
        ev.synchronize_timeout(std::time::Duration::from_millis(20))
            .is_err()
    );
    ev.mark_completed();
    assert!(
        ev.synchronize_timeout(std::time::Duration::from_millis(20))
            .is_ok()
    );
}

/// Malformed `IrGraph`s (pub fields) used to panic DCE/`live_set`/`execute`.
#[test]
fn malformed_ir_graphs_are_errors() -> Result<()> {
    use oxideon::compiler::{IrGraph, IrNode, IrOp};
    let mut g = IrGraph::new();
    let x = g.add_input();
    g.nodes.push(IrNode {
        id: 1,
        op: IrOp::Add(x, 99),
    });
    g.set_outputs(vec![1]);
    assert!(g.validate().is_err());
    assert!(g.optimize().is_err());
    let inp = Tensor::ones(&[2], DType::F32, &Device::Cpu)?;
    assert!(g.execute(&[inp]).is_err());
    // Bad output id alone is also an error, not a panic.
    let mut g2 = IrGraph::new();
    g2.add_input();
    g2.set_outputs(vec![7]);
    assert!(g2.validate().is_err());
    assert!(g2.optimize().is_err());
    Ok(())
}

/// `differentiate` used to reject `MeanAll` and fused linears.
#[test]
fn differentiate_handles_mean_and_fused() -> Result<()> {
    use oxideon::compiler::{IrGraph, IrOp};
    let mut g = IrGraph::new();
    let x = g.add_input();
    let m = g.push_op(IrOp::MeanAll(x));
    g.set_outputs(vec![m]);
    let grads = g.differentiate(m, &[x])?;
    assert_eq!(grads.len(), 1);
    g.set_outputs(vec![grads[0]]);
    g.optimize()?;
    let inp = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4])?;
    let out = g.execute(&[inp])?.pop().unwrap();
    assert_eq!(out.to_vec::<f32>()?, vec![0.25, 0.25, 0.25, 0.25]);

    // A fused graph unfuses and differentiates through silu.
    let mut f = IrGraph::new();
    let fx = f.add_input();
    let fw = f.add_input();
    let fb = f.add_input();
    let lin = f.push_op(IrOp::FusedLinearSilu {
        x: fx,
        w: fw,
        b: fb,
    });
    let s = f.push_op(IrOp::SumAll(lin));
    f.set_outputs(vec![s]);
    assert!(f.differentiate(s, &[fx]).is_ok());

    // Relu still errors, loudly and specifically.
    let mut r = IrGraph::new();
    let rx = r.add_input();
    let rr = r.push_op(IrOp::Relu(rx));
    let rs = r.push_op(IrOp::SumAll(rr));
    r.set_outputs(vec![rs]);
    let err = r.differentiate(rs, &[rx]).unwrap_err();
    assert!(format!("{err}").contains("Relu"));
    Ok(())
}

/// `x + 0 -> x` used to fire under broadcasting, changing the output shape.
#[test]
fn simplify_keeps_broadcast_shapes() -> Result<()> {
    use oxideon::compiler::{IrGraph, IrOp};
    let mut g = IrGraph::new();
    let x = g.add_input();
    let y = g.add_input();
    let z = g.push_op(IrOp::ZeroLike { src: y });
    let a = g.push_op(IrOp::Add(x, z));
    g.set_outputs(vec![a]);
    g.optimize()?;
    // x=[1] plus zeros shaped [3] must stay [3], not fold to x's [1].
    let x1 = Tensor::from_slice(&[2.0f32], &[1])?;
    let y3 = Tensor::zeros(&[3], DType::F32, &Device::Cpu)?;
    let out = g.execute(&[x1, y3])?.pop().unwrap();
    assert_eq!(out.shape(), &[3]);
    assert_eq!(out.to_vec::<f32>()?, vec![2.0, 2.0, 2.0]);
    Ok(())
}

/// `bind_dim` used to accept anything for `Add`/`Mul`; `matmul_with` skipped K.
#[test]
fn symbolic_shapes_bind_and_check_k() {
    use oxideon::compiler::{SymDim, SymbolicShape};
    let s = SymbolicShape::new(vec![
        SymDim::Add(
            Box::new(SymDim::Var("S".into())),
            Box::new(SymDim::Const(1)),
        ),
        SymDim::Mul(
            Box::new(SymDim::Const(2)),
            Box::new(SymDim::Var("H".into())),
        ),
    ]);
    let b = s.bind(&[5, 8]).unwrap();
    assert_eq!(b["S"], 4);
    assert_eq!(b["H"], 4);
    assert!(s.bind(&[5, 7]).is_err());
    let a = SymbolicShape::parse("[B, M, 3]");
    let c = SymbolicShape::parse("[B, 4, N]");
    assert!(a.matmul_with(&c).is_err());
    let ok = SymbolicShape::parse("[B, 3, N]");
    let out = a.matmul_with(&ok).unwrap();
    assert_eq!(out.to_string(), "[B, M, N]");
}

/// `build_wgsl` used to panic on malformed kernels or emit invalid WGSL.
#[test]
fn jit_builder_validates() {
    use oxideon::jit::{JitKernelBuilder, JitOp};
    let empty = JitKernelBuilder::new("k");
    assert!(empty.build_wgsl(64).is_err());
    let mut b = JitKernelBuilder::new("k");
    let x = b.add_input("x");
    let bad = b.add_op(JitOp::Add, &[x]);
    b.set_output(bad);
    assert!(b.build_wgsl(64).is_err());
    assert!(b.build_wgsl(0).is_err());
    let mut g = JitKernelBuilder::new("k");
    let gx = g.add_input("x");
    let gy = g.add_input("y");
    let s = g.add_op(JitOp::Add, &[gx, gy]);
    g.set_output(s);
    let wgsl = g.build_wgsl(64).unwrap();
    assert!(wgsl.contains("@compute @workgroup_size(64)"));
}

/// Autotune shared-memory math used to overflow; empty GEMMs gave NaN metrics.
#[test]
fn autotune_saturates_and_rejects_empty() {
    use oxideon::tile_ir::{HardwareProfile, KernelAutoTuner, TileConfig};
    let huge = TileConfig::new(usize::MAX, 1024, 1024, 4, 2);
    assert_eq!(huge.shared_memory_bytes(), usize::MAX);
    let tuner = KernelAutoTuner::new(HardwareProfile::default_cpu());
    let cfg = TileConfig::new(64, 64, 32, 4, 2);
    assert!(tuner.evaluate_candidate(0, 64, 64, &cfg).is_none());
    let m = tuner.evaluate_candidate(128, 128, 128, &cfg).unwrap();
    assert!(m.estimated_latency_us.is_finite());
}

/// The SPIR-V emitter used to produce a header-only stub with a bogus `Bound`.
#[test]
fn spirv_emits_structured_module() {
    use oxideon::driver::VulkanSpirvKernel;
    let k = VulkanSpirvKernel::new("add_kernel", 2);
    let words = k.emit_spirv_binary().unwrap();
    assert_eq!(words[0], VulkanSpirvKernel::SPIRV_MAGIC);
    assert_eq!(words[1], VulkanSpirvKernel::SPIRV_VERSION);
    // 9 types/globals + gid + 2 bindings + main + label = 14 ids -> bound 15.
    assert_eq!(words[3], 15);
    // Walk the instruction stream: every instruction must end exactly at the
    // stream end, and the module must contain an entry point, an execution
    // mode, and one function with a label/return/end.
    let mut ops = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let wc = (words[i] >> 16) as usize;
        let op = words[i] & 0xffff;
        assert!(wc >= 1, "empty instruction at word {i}");
        ops.push(op);
        if op == 16 {
            // OpExecutionMode LocalSize carries the workgroup size.
            assert_eq!(&words[i + 3..i + 6], &[64, 1, 1]);
        }
        i += wc;
    }
    assert_eq!(i, words.len());
    for need in [15, 16, 54, 56, 248, 253] {
        assert!(ops.contains(&need), "missing opcode {need}");
    }
    // Entry-point name bytes are embedded.
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    assert!(bytes.windows(10).any(|w| w == b"add_kernel"));
}

/// WGSL swiglu used to index buffers with no bounds check; rmsnorm reduction
/// was wrong for non-power-of-2 workgroup sizes.
#[test]
fn wgsl_shaders_guard_and_reduce() {
    use oxideon::driver::WgslKernelBuilder;
    let b = WgslKernelBuilder::new("k", 6);
    let sw = b.emit_fused_swiglu_wgsl();
    assert!(sw.contains("arrayLength(&gate_a)"));
    let rn = b.emit_fused_rmsnorm_wgsl(40, 1e-5);
    assert!(rn.contains("(tid + s) < 6u"));
    assert!(rn.contains("row_offset >= arrayLength(&input_x)"));
}

/// Fused CE with z-loss used to return 0/0 NaN on an empty batch.
#[test]
fn fused_ce_rejects_empty_batch() -> Result<()> {
    use oxideon::fused_cross_entropy_with_z_loss;
    let empty = Tensor::zeros(&[0, 8], DType::F32, &Device::Cpu)?;
    assert!(fused_cross_entropy_with_z_loss(&empty, &[], 0.0).is_err());
    let logits = Tensor::from_slice(&[1.0f32, 2.0, 0.5, -1.0], &[2, 2])?;
    let loss = fused_cross_entropy_with_z_loss(&logits, &[1, 0], 0.0)?;
    assert!(loss.item::<f32>()?.is_finite());
    Ok(())
}

/// Every `Display` device form must parse back to the same device.
#[test]
fn device_display_roundtrips() {
    use oxideon::Device;
    let cases = [
        Device::Cpu,
        Device::Cuda(2),
        Device::Metal(1),
        Device::Vulkan(0),
        Device::Tpu { chip: 1, core: 3 },
        Device::Npu {
            name: "ascend".into(),
            index: 1,
        },
        Device::Meta,
    ];
    for d in cases {
        let s = d.to_string();
        let back: Device = s.parse().unwrap();
        assert_eq!(back, d, "roundtrip failed for {s}");
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let w = Device::Wgpu(3);
        let back: Device = w.to_string().parse().unwrap();
        assert_eq!(back, w);
    }
}

/// Concurrent deallocations used to race the check-then-act and wrap the counter.
#[test]
fn hardware_dealloc_is_race_free() {
    use oxideon::device::HardwareManager;
    use std::sync::Arc;
    let mgr = Arc::new(HardwareManager::new());
    mgr.track_allocation(&Device::Cpu, 1000);
    let mut handles = Vec::new();
    for _ in 0..8 {
        let m = Arc::clone(&mgr);
        handles.push(std::thread::spawn(move || {
            for _ in 0..250 {
                m.track_deallocation(&Device::Cpu, 1);
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    // 2000 deallocated against 1000 allocated: saturates at exactly 0.
    assert_eq!(mgr.device_allocated_bytes(&Device::Cpu), 0);
}

/// `trimmed_mean` used to index ragged gradient vecs, panicking.
#[test]
fn trimmed_mean_rejects_ragged() -> Result<()> {
    use oxideon::distributed::ByzantineAggregator;
    let a = Tensor::from_slice(&[1.0f32, 2.0], &[2])?;
    let b = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3])?;
    assert!(ByzantineAggregator::trimmed_mean(&[a.clone(), b], 0.0).is_err());
    let c = Tensor::from_slice(&[3.0f32, 4.0], &[2])?;
    let m = ByzantineAggregator::trimmed_mean(&[a, c], 0.0)?;
    assert_eq!(m.to_vec::<f32>()?, vec![2.0, 3.0]);
    Ok(())
}

/// FlashDecoding/ring attention used to slice-index K/V without shape checks.
#[test]
fn long_context_engines_validate_kv() -> Result<()> {
    use oxideon::long_context::{FlashDecodingConfig, FlashDecodingEngine, RingAttention};
    let eng = FlashDecodingEngine::new(FlashDecodingConfig::new(2, 4));
    let q = Tensor::randn(&[1, 2, 1, 4])?;
    let k_bad = Tensor::randn(&[1, 3, 8, 4])?;
    let v = Tensor::randn(&[1, 2, 8, 4])?;
    assert!(eng.decode_attention(&q, &k_bad, &v).is_err());
    let k = Tensor::randn(&[1, 2, 8, 4])?;
    let v_bad = Tensor::randn(&[1, 2, 4, 4])?;
    assert!(eng.decode_attention(&q, &k, &v_bad).is_err());
    assert_eq!(eng.decode_attention(&q, &k, &v)?.shape(), &[1, 2, 1, 4]);

    let ring = RingAttention::new(2, 0, false).unwrap();
    let ql = Tensor::randn(&[1, 2, 4, 4])?;
    let k0 = Tensor::randn(&[1, 2, 4, 4])?;
    let v0 = Tensor::randn(&[1, 2, 4, 4])?;
    let k1_bad = Tensor::randn(&[2])?;
    assert!(
        ring.forward_simulated(&ql, &[k0.clone(), k1_bad], &[v0.clone(), v0.clone()])
            .is_err()
    );
    assert_eq!(
        ring.forward_simulated(&ql, &[k0.clone(), k0], &[v0.clone(), v0])?
            .shape(),
        &[1, 2, 4, 4]
    );
    Ok(())
}

/// GPTQ calibration with a scrambled last dim used to be silently accepted.
#[test]
fn gptq_validates_calibration() -> Result<()> {
    use oxideon::nn::Linear;
    use oxideon::quant::{GPTQLinear, QuantScheme};
    let lin = Linear::new(8, 4)?;
    let bad = Tensor::randn(&[7, 5])?;
    assert!(
        GPTQLinear::from_linear_calibrated(&lin, &bad, QuantScheme::Q8_0 { block_size: 32 }, 0.1)
            .is_err()
    );
    let good = Tensor::randn(&[16, 8])?;
    let q =
        GPTQLinear::from_linear_calibrated(&lin, &good, QuantScheme::Q8_0 { block_size: 32 }, 0.1)?;
    assert_eq!(q.to_linear(DType::F32)?.forward(&good)?.shape(), &[16, 4]);
    Ok(())
}

/// Tree mask building hung forever on parent cycles; greedy verify silently
/// used batch 0 of multi-batch logits.
#[test]
fn speculative_verify_validates() -> Result<()> {
    use oxideon::serving::{TreeAttentionMask, UnifiedSpeculativeEngine};
    assert!(TreeAttentionMask::build(&[1, 0]).is_err());
    assert!(TreeAttentionMask::build(&[0, 9]).is_err());
    assert_eq!(TreeAttentionMask::build(&[0, 0, 1])?.shape(), &[1, 1, 3, 3]);
    let logits = Tensor::randn(&[2, 4, 8])?;
    assert!(UnifiedSpeculativeEngine::verify_greedy(&[1, 2], &logits).is_err());
    let logits1 = Tensor::randn(&[1, 4, 8])?;
    let out = UnifiedSpeculativeEngine::verify_greedy(&[1, 2], &logits1)?;
    assert!(out.num_accepted <= 2);
    Ok(())
}

/// `roi_align` used to silently clamp out-of-range batch indices.
#[test]
fn roi_align_rejects_bad_batch() -> Result<()> {
    use oxideon::roi_align;
    let input = Tensor::randn(&[2, 3, 8, 8])?;
    let bad = Tensor::from_slice(&[5.0f32, 0.0, 0.0, 4.0, 4.0], &[1, 5])?;
    assert!(roi_align(&input, &bad, [2, 2], 1.0, 1).is_err());
    assert!(roi_align(&input, &bad, [0, 2], 1.0, 1).is_err());
    let good = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 4.0, 4.0], &[1, 5])?;
    assert_eq!(
        roi_align(&input, &good, [2, 2], 1.0, 1)?.shape(),
        &[1, 3, 2, 2]
    );
    Ok(())
}

/// GRPO's KL term used to overflow to inf on extreme log-ratios.
#[test]
fn grpo_loss_stays_finite() -> Result<()> {
    use oxideon::rl::grpo_loss;
    let policy = Tensor::from_slice(&[-1000.0f32, 0.0, -1000.0, 0.0], &[2, 2])?;
    let old = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[2, 2])?;
    let reference = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[2, 2])?;
    let adv = Tensor::from_slice(&[1.0f32, 1.0], &[2])?;
    let loss = grpo_loss(&policy, &old, &reference, &adv, None, 0.2, 0.04)?;
    assert!(loss.item::<f32>()?.is_finite());
    Ok(())
}

/// TCN blocks must be causal: perturbing future inputs cannot change early outputs.
#[test]
fn tcn_block_is_causal() -> Result<()> {
    use oxideon::nn::Module;
    use oxideon::nn::architectures::TCNBlock;
    let tcn = TCNBlock::new(4, 4, 3)?;
    let x = Tensor::randn(&[2, 4, 16])?;
    let y0 = tcn.forward(&x)?;
    let mut perturbed = x.to_vec::<f32>()?;
    for (i, v) in perturbed.iter_mut().enumerate() {
        if i % 16 >= 14 {
            *v += 10.0;
        }
    }
    let xp = Tensor::from_vec(perturbed, &[2, 4, 16])?;
    let y1 = tcn.forward(&xp)?;
    let (a, b) = (y0.to_vec::<f32>()?, y1.to_vec::<f32>()?);
    // First two timesteps see only inputs [0..=1]; the tail perturbation
    // (last 2 timesteps of each sequence) must not reach them.
    for n in 0..2 {
        for t in 0..2 {
            for c in 0..4 {
                let idx = (n * 4 + c) * 16 + t;
                assert!((a[idx] - b[idx]).abs() < 1e-5, "future leaked into t={t}");
            }
        }
    }
    Ok(())
}

/// Mel extractor construction used to accept zero n_fft/hop/sample-rate.
#[test]
fn mel_rejects_degenerate_params() {
    use oxideon::audio::MelSpectrogram;
    assert!(MelSpectrogram::new(80, 0, 160, 16000).is_err());
    assert!(MelSpectrogram::new(80, 400, 0, 16000).is_err());
    assert!(MelSpectrogram::new(80, 400, 160, 0).is_err());
    assert!(MelSpectrogram::new(0, 400, 160, 16000).is_err());
    assert!(MelSpectrogram::new(8, 64, 16, 16000).is_ok());
}

/// Whisper encoder used to fail deep in conv1 on wrong mel dims.
#[test]
fn whisper_validates_mel() -> Result<()> {
    use oxideon::audio::{AudioEncoder, WhisperConfig};
    let cfg = WhisperConfig {
        n_mels: 8,
        d_model: 16,
        encoder_layers: 1,
        encoder_heads: 2,
        decoder_layers: 0,
        decoder_heads: 2,
        d_ff: 32,
        vocab_size: 32,
        max_source_positions: 16,
        max_target_positions: 8,
    };
    let enc = AudioEncoder::new(&cfg)?;
    let bad = Tensor::randn(&[1, 6, 16])?;
    assert!(enc.forward(&bad).is_err());
    let good = Tensor::randn(&[1, 8, 16])?;
    assert_eq!(enc.forward(&good)?.shape(), &[1, 8, 16]);
    Ok(())
}

/// DiT AdaLN chunking used to index dim 2 without a rank check.
#[test]
fn dit_block_validates_ranks() -> Result<()> {
    use oxideon::video::VideoDiTBlock;
    let b = VideoDiTBlock::new(8, 2, 16)?;
    let flat = Tensor::randn(&[2, 8])?;
    let t = Tensor::randn(&[2, 8])?;
    let txt = Tensor::randn(&[2, 3, 16])?;
    assert!(b.forward(&flat, &t, &txt, (2, 2, 1)).is_err());
    let x = Tensor::randn(&[2, 4, 8])?;
    assert_eq!(b.forward(&x, &t, &txt, (2, 2, 1))?.shape(), &[2, 4, 8]);
    Ok(())
}

/// Hyperbolic GAT used to index shape()[0] on any rank.
#[test]
fn hgat_requires_2d() -> Result<()> {
    use oxideon::geom_graphs::HyperbolicGATLayer;
    let g = HyperbolicGATLayer::new(4, 2, 0.5, 0.1)?;
    let scalar = Tensor::from_slice(&[1.0f32], &[])?;
    assert!(g.forward(&scalar, &[]).is_err());
    let wrong = Tensor::randn(&[3, 3])?;
    assert!(g.forward(&wrong, &[]).is_err());
    Ok(())
}

/// RoPE-3D grid math used to divide by zero / overflow on degenerate grids.
#[test]
fn rope3d_rejects_degenerate_grid() -> Result<()> {
    use oxideon::video::RotaryEmbedding3D;
    let r = RotaryEmbedding3D::new(8, 10000.0)?;
    let x = Tensor::randn(&[1, 8, 2, 8])?;
    assert!(r.forward(&x, 2, 0, 2).is_err());
    assert!(r.forward(&x, 2, 2, 2).is_ok());
    Ok(())
}

/// McKean-Vlasov variance used to overflow on negative time.
#[test]
fn mkv_rejects_negative_time() {
    use oxideon::ode::mc_kean_vlasov_sde::McKeanVlasovPropagationChaos;
    let m = McKeanVlasovPropagationChaos::new(1.0, 2.0).unwrap();
    assert!(m.mean_field_variance(2.0, -1.0).is_err());
    assert!(m.mean_field_variance(-1.0, 1.0).is_err());
    assert!(m.mean_field_variance(2.0, 10.0).unwrap().is_finite());
}

/// ACT temporal ensembling used to interleave batch>1 trajectories silently.
#[test]
fn act_ensemble_is_single_stream() -> Result<()> {
    use oxideon::robotics::{ACTConfig, ActionChunkingTransformer};
    let mut act = ActionChunkingTransformer::new(ACTConfig {
        state_dim: 4,
        action_dim: 3,
        chunk_size: 4,
        hidden_dim: 8,
        temporal_ensemble_decay: 0.01,
    })?;
    let batch = Tensor::randn(&[2, 4])?;
    assert!(act.step_temporal_ensemble(&batch).is_err());
    let single = Tensor::randn(&[1, 4])?;
    assert_eq!(act.step_temporal_ensemble(&single)?.len(), 3);
    Ok(())
}

/// Mismatched GP operator arity used to silently compute the wrong function.
#[test]
fn symbolic_gp_strict_arity() {
    use oxideon::reasoning::{OpType, SymbolicNode};
    let bad = SymbolicNode::Binary(
        OpType::Sin,
        Box::new(SymbolicNode::Variable(0)),
        Box::new(SymbolicNode::Variable(0)),
    );
    assert!(bad.try_evaluate(&[1.0]).is_err());
    assert!(bad.evaluate(&[1.0]).is_nan());
    let good = SymbolicNode::Binary(
        OpType::Add,
        Box::new(SymbolicNode::Variable(0)),
        Box::new(SymbolicNode::Constant(2.0)),
    );
    assert_eq!(good.try_evaluate(&[1.0]).unwrap(), 3.0);
}

/// Ragged EDMD snapshots used to panic on row indexing.
#[test]
fn koopman_rejects_ragged() {
    use oxideon::ode::koopman::KoopmanSpectralEngine;
    let x = vec![vec![1.0, 2.0], vec![3.0]];
    let y = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    assert!(KoopmanSpectralEngine::fit_diagonal_edmd(&x, &y).is_err());
    let xr = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    assert!(KoopmanSpectralEngine::fit_diagonal_edmd(&xr, &y).is_ok());
}

/// Cluster mutation used to divide by a zero cluster variable.
#[test]
fn cluster_mutation_rejects_zero_var() {
    use oxideon::ClusterAlgebraQuiverEngine;
    let mut q = ClusterAlgebraQuiverEngine::new(2, vec![0, 1, -1, 0], vec![1.0, 0.0]).unwrap();
    assert!(q.mutate(1).is_err());
    assert!(q.mutate(0).is_ok());
}

/// Shamir reconstruction used to silently accept duplicate x coordinates.
#[test]
fn shamir_rejects_duplicate_x() {
    use oxideon::ShamirSecretSharingFp;
    let s = ShamirSecretSharingFp::new(101).unwrap();
    assert!(s.reconstruct_secret(&[1, 1, 3], &[5, 5, 9]).is_err());
    // Distinct shares reconstruct fine.
    let secret = s.reconstruct_secret(&[1, 2, 3], &[8, 8, 8]).unwrap();
    assert_eq!(secret, 8);
}

#[test]
fn low_slice_min_step_errors_loudly() {
    use oxideon::indexing::SliceElem;
    let t = Tensor::arange(0.0, 5.0, 1.0, DType::F32).unwrap();
    let err = t.i(SliceElem::step(None, None, isize::MIN)).unwrap_err();
    assert!(
        format!("{:?}", err).contains("MIN"),
        "unexpected error: {:?}",
        err
    );
    // Sanity: ordinary negative steps still work.
    let rev = t.i(SliceElem::step(None, None, -1)).unwrap();
    assert_eq!(rev.shape(), &[5]);
}

#[test]
fn r2_hand_built_sparse_structs_are_errors_not_panics() {
    // COO: out-of-bounds index, bypassing SparseCoo::new validation.
    let bad_coo = SparseCoo {
        indices: vec![(5, 5)],
        values: vec![1.0],
        shape: [2, 2],
    };
    assert!(bad_coo.to_dense(DType::F64, &Device::Cpu).is_err());
    assert!(bad_coo.to_csr().is_err());
    // COO: length mismatch (zip would silently truncate).
    let skew_coo = SparseCoo {
        indices: vec![(0, 0), (1, 1)],
        values: vec![1.0],
        shape: [2, 2],
    };
    assert!(skew_coo.to_dense(DType::F64, &Device::Cpu).is_err());
    // CSR: short row_ptr.
    let bad_csr = SparseCsr {
        row_ptr: vec![0],
        col_idx: vec![0],
        values: vec![1.0],
        shape: [2, 2],
    };
    assert!(bad_csr.to_dense(DType::F64, &Device::Cpu).is_err());
    assert!(bad_csr.transpose().is_err());
    // CSR: non-monotonic row_ptr.
    let nonmono = SparseCsr {
        row_ptr: vec![0, 2, 1],
        col_idx: vec![0, 1],
        values: vec![1.0, 1.0],
        shape: [2, 2],
    };
    assert!(
        nonmono
            .spmm(&Tensor::ones(&[2, 1], DType::F64, &Device::Cpu).unwrap())
            .is_err()
    );
    // CSR: column index out of range.
    let bad_col = SparseCsr {
        row_ptr: vec![0, 1],
        col_idx: vec![7],
        values: vec![1.0],
        shape: [1, 2],
    };
    assert!(bad_col.to_dense(DType::F64, &Device::Cpu).is_err());
    // Valid structs still convert.
    let good = SparseCoo::new(vec![(0, 1)], vec![2.0], [2, 2]).unwrap();
    let csr = good.to_csr().unwrap();
    let d = csr.to_dense(DType::F64, &Device::Cpu).unwrap();
    assert_eq!(d.to_vec::<f64>().unwrap(), vec![0.0, 2.0, 0.0, 0.0]);
}

#[test]
fn r2_spmm_huge_output_is_an_error_not_an_abort() {
    // 2^36 x 1 output (~512 GiB) must be refused up front, not allocated.
    let big = SparseCsr {
        row_ptr: vec![0, 0],
        col_idx: vec![],
        values: vec![],
        shape: [1 << 36, 1],
    };
    let x = Tensor::ones(&[1, 1], DType::F64, &Device::Cpu).unwrap();
    assert!(big.spmm(&x).is_err());
    // Same via the element limit with a fully VALID struct: 2^20 rows x
    // n=4096 cols = 2^32 elements > MAX_ELEMENTS, refused before the alloc.
    let wide = SparseCsr {
        row_ptr: vec![0; (1 << 20) + 1],
        col_idx: vec![],
        values: vec![],
        shape: [1 << 20, 1],
    };
    let xw = Tensor::ones(&[1, 4096], DType::F64, &Device::Cpu).unwrap();
    assert!(wide.spmm(&xw).is_err());
}

#[test]
fn r2_as_strided_rejects_out_of_storage_views() {
    let t = Tensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap();
    // Reads past the end of storage.
    assert!(t.as_strided(&[8], &[1], 0).is_err());
    // Negative-stride underflow below offset 0.
    assert!(t.as_strided(&[4], &[-1], 0).is_err());
    // Offset beyond storage.
    assert!(t.as_strided(&[1], &[1], 100).is_err());
    // Stride-extent overflow.
    assert!(t.as_strided(&[4], &[isize::MAX], 0).is_err());
    // Valid views still work.
    let v = t.as_strided(&[2], &[2], 1).unwrap();
    assert_eq!(v.shape(), &[2]);
    let flip = t.as_strided(&[4], &[-1], 3).unwrap();
    assert_eq!(flip.shape(), &[4]);
    // Empty views touch nothing: any offset is fine.
    assert!(t.as_strided(&[0], &[1], 100).is_ok());
}

#[test]
fn r2_try_from_storage_validates_layout() {
    use oxideon::Layout;
    let t = Tensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap();
    let bad = Layout::try_new(vec![8], vec![1], 0).unwrap();
    assert!(Tensor::try_from_storage(t.storage().clone(), bad).is_err());
    let good = Layout::try_new(vec![2], vec![2], 0).unwrap();
    assert!(Tensor::try_from_storage(t.storage().clone(), good).is_ok());
}

#[test]
fn r2_narrow_huge_start_is_an_error_not_a_panic() {
    let t = Tensor::zeros(&[5], DType::F32, &Device::Cpu).unwrap();
    // start + len overflows usize: must Err (debug builds panicked here).
    assert!(t.narrow(0, usize::MAX, 1).is_err());
    assert!(t.narrow(0, 0, usize::MAX).is_err());
    assert!(t.narrow(0, 4, 2).is_err());
    assert_eq!(t.narrow(0, 1, 3).unwrap().shape(), &[3]);
}

#[test]
fn r2_repeat_huge_output_is_an_error_not_a_hang() {
    let t = Tensor::zeros(&[2], DType::F32, &Device::Cpu).unwrap();
    assert!(t.repeat(&[1 << 40]).is_err());
    assert!(t.repeat(&[usize::MAX]).is_err());
    assert_eq!(t.repeat(&[3]).unwrap().shape(), &[6]);
    // Zero-repeat torch parity preserved.
    assert_eq!(t.repeat(&[0]).unwrap().shape(), &[0]);
}

#[test]
fn r2_pad_overflowing_width_is_an_error_not_a_panic() {
    let t = Tensor::zeros(&[2], DType::F32, &Device::Cpu).unwrap();
    assert!(t.pad(&[(usize::MAX, 0)], 0.0).is_err());
    assert!(t.pad(&[(0, usize::MAX)], 0.0).is_err());
    assert_eq!(t.pad(&[(1, 2)], 0.0).unwrap().shape(), &[5]);
}

#[test]
fn r2_bincount_huge_range_is_an_error_not_an_abort() {
    // 3e9 + 1 > MAX_ELEMENTS (2^31): refused before any allocation.
    let big = Tensor::from_slice(&[3_000_000_000i64], &[1]).unwrap();
    assert!(big.bincount(None, 0).is_err());
    let max = Tensor::from_slice(&[i64::MAX], &[1]).unwrap();
    assert!(max.bincount(None, 0).is_err());
    let t = Tensor::from_slice(&[0i64, 2, 2], &[3]).unwrap();
    assert!(t.bincount(None, usize::MAX).is_err());
    // Negatives still rejected (now before any allocation).
    let neg = Tensor::from_slice(&[0i64, -1], &[2]).unwrap();
    assert!(neg.bincount(None, 0).is_err());
    // Valid path unchanged.
    assert_eq!(
        t.bincount(None, 0).unwrap().to_vec::<i64>().unwrap(),
        vec![1, 0, 2]
    );
}

#[test]
fn r2_histc_huge_bins_is_an_error_not_an_abort() {
    let t = Tensor::from_slice(&[0.5f32], &[1]).unwrap();
    assert!(t.histc(usize::MAX, 0.0, 1.0).is_err());
    assert_eq!(t.histc(4, 0.0, 1.0).unwrap().shape(), &[4]);
}

#[test]
fn r2_customop_backward_arity_mismatch_is_an_error() {
    use oxideon::CustomOp;
    struct ShortBackward;
    impl CustomOp for ShortBackward {
        fn forward(&self, inputs: &[Tensor]) -> Result<Tensor> {
            inputs[0].add(&inputs[1])
        }
        fn backward(
            &self,
            _inputs: &[Tensor],
            grad_output: &Tensor,
        ) -> Result<Vec<Option<Tensor>>> {
            // Wrong: 2 inputs but only 1 grad (used to silently drop input 2).
            Ok(vec![Some(grad_output.clone())])
        }
    }
    let a = Tensor::from_slice(&[1.0f32], &[1])
        .unwrap()
        .requires_grad_(true);
    let b = Tensor::from_slice(&[2.0f32], &[1])
        .unwrap()
        .requires_grad_(true);
    let y = Tensor::apply_custom(ShortBackward, &[a.clone(), b.clone()]).unwrap();
    assert!(
        grad(
            std::slice::from_ref(&y),
            &[a.clone(), b.clone()],
            None,
            false
        )
        .is_err()
    );
    let seed = Tensor::ones(&[1], DType::F32, &Device::Cpu).unwrap();
    assert!(y.backward_with_grad(&seed, false).is_err());
}

#[test]
fn r2_huge_usize_indices_do_not_wrap_to_negatives() {
    let t = Tensor::arange(0.0, 5.0, 1.0, DType::F32).unwrap();
    // Was: wrapped to Index(-1) and silently returned the LAST element.
    assert!(t.i(usize::MAX).is_err());
    // Saturating ranges: no panic, clamped to the valid region.
    assert_eq!(t.i(0..=usize::MAX).unwrap().shape(), &[5]);
    assert_eq!(t.i(usize::MAX..).unwrap().shape(), &[0]);
    assert_eq!(t.i(..=usize::MAX).unwrap().shape(), &[5]);
    // Ordinary indexing unchanged.
    assert_eq!(t.i(4usize).unwrap().to_vec::<f32>().unwrap(), vec![4.0]);
    assert_eq!(t.i(1usize..3).unwrap().shape(), &[2]);
}

#[test]
fn r2_norm_and_block_rank_mismatches_are_errors() {
    let gn = GroupNorm::new(2, 4, 1e-5).unwrap();
    assert!(
        gn.forward(&Tensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap())
            .is_err()
    );
    assert!(
        gn.forward(&Tensor::zeros(&[2, 8, 3], DType::F32, &Device::Cpu).unwrap())
            .is_err()
    );
    assert_eq!(
        gn.forward(&Tensor::zeros(&[2, 4, 3], DType::F32, &Device::Cpu).unwrap())
            .unwrap()
            .shape(),
        &[2, 4, 3]
    );
    let se = SqueezeExcitation2d::new(4, 2).unwrap();
    assert!(
        se.forward(&Tensor::zeros(&[4, 4, 4], DType::F32, &Device::Cpu).unwrap())
            .is_err()
    );
    assert_eq!(
        se.forward(&Tensor::zeros(&[1, 4, 4, 4], DType::F32, &Device::Cpu).unwrap())
            .unwrap()
            .shape(),
        &[1, 4, 4, 4]
    );
    let nx = ConvNeXtBlock::new(4, 3, 0.0, 1e-6).unwrap();
    assert!(
        nx.forward(&Tensor::zeros(&[4, 4, 4], DType::F32, &Device::Cpu).unwrap())
            .is_err()
    );
}

#[test]
fn r2_rnn_cell_rank_mismatches_are_errors() {
    let bad = Tensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap();
    let good = Tensor::zeros(&[2, 4], DType::F32, &Device::Cpu).unwrap();
    let rnn = RNNCell::new(4, 8).unwrap();
    assert!(rnn.step(&bad, None).is_err());
    assert_eq!(rnn.step(&good, None).unwrap().shape(), &[2, 8]);
    let lstm = LSTMCell::new(4, 8).unwrap();
    assert!(lstm.step(&bad, None).is_err());
    let (h, c) = lstm.step(&good, None).unwrap();
    assert_eq!(h.shape(), &[2, 8]);
    assert_eq!(c.shape(), &[2, 8]);
    let gru = GRUCell::new(4, 8).unwrap();
    assert!(gru.step(&bad, None).is_err());
    assert_eq!(gru.step(&good, None).unwrap().shape(), &[2, 8]);
}

#[test]
fn r2_vla_empty_text_sequence_is_an_error() {
    let vla = VisionLanguageActionModel::new(8, 4, 1, 16, 16, 10, 2, 3).unwrap();
    let images = Tensor::zeros(&[1, 1, 8, 8], DType::F32, &Device::Cpu).unwrap();
    let empty = Tensor::zeros(&[1, 0, 10], DType::F32, &Device::Cpu).unwrap();
    assert!(vla.forward(&images, &empty).is_err());
    let text = Tensor::zeros(&[1, 3, 10], DType::F32, &Device::Cpu).unwrap();
    let (logits, actions) = vla.forward(&images, &text).unwrap();
    assert_eq!(logits.shape(), &[1, 3, 10]);
    assert_eq!(actions.shape(), &[1, 2, 3]);
}

#[test]
fn r2_transfusion_lm_loss_is_cross_entropy_over_targets() {
    let config = TransfusionConfig {
        vocab_size: 20,
        d_model: 16,
        patch_dim: 8,
        num_layers: 1,
        num_heads: 2,
        d_ff: 32,
        dropout: 0.0,
    };
    let model = TransfusionModel::new(config).unwrap();
    let clean = Tensor::randn(&[1, 2, 8]).unwrap();
    let noise = Tensor::randn(&[1, 2, 8]).unwrap();
    let t = Tensor::from_slice(&[0.5f32], &[1, 1]).unwrap();
    let tok_a = Tensor::from_slice(&[1i64, 2, 3, 4], &[1, 4]).unwrap();
    let tok_b = Tensor::from_slice(&[5i64, 6, 7, 8], &[1, 4]).unwrap();
    let la = model.compute_loss(&tok_a, &clean, &noise, &t, 1.0).unwrap();
    let lb = model.compute_loss(&tok_b, &clean, &noise, &t, 1.0).unwrap();
    assert_eq!(la.ndim(), 0);
    assert!(la.item::<f32>().unwrap() > 0.0);
    // The old mean(logits^2) term never consulted the targets; CE must vary.
    assert_ne!(la.item::<f32>().unwrap(), lb.item::<f32>().unwrap());
    // Fractional float ids are rejected, never truncated.
    let frac = Tensor::from_slice(&[1.5f32, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
    assert!(model.compute_loss(&frac, &clean, &noise, &t, 1.0).is_err());
    // Single-token sequences cannot shift.
    let one = Tensor::from_slice(&[1i64], &[1, 1]).unwrap();
    assert!(model.compute_loss(&one, &clean, &noise, &t, 1.0).is_err());
    // Out-of-vocab ids are rejected.
    let oob = Tensor::from_slice(&[1i64, 2, 3, 99], &[1, 4]).unwrap();
    assert!(model.compute_loss(&oob, &clean, &noise, &t, 1.0).is_err());
}

#[test]
fn r2_gradscaler_rejects_non_positive_scale() {
    assert!(GradScaler::new(0.0).is_err());
    assert!(GradScaler::new(-100.0).is_err());
    assert!(GradScaler::new(f64::NAN).is_err());
    assert!(GradScaler::new(f64::INFINITY).is_err());
    assert!(
        GradScaler::new(128.0)
            .unwrap()
            .with_intervals(-2.0, 0.5, 10)
            .is_err()
    );
    assert!(
        GradScaler::new(128.0)
            .unwrap()
            .with_intervals(2.0, 0.0, 10)
            .is_err()
    );
    let s = GradScaler::new(128.0)
        .unwrap()
        .with_intervals(2.0, 0.5, 10)
        .unwrap();
    assert_eq!(s.scale_factor(), 128.0);
}

#[test]
fn r2_ctc_empty_batch_is_an_error_not_nan() {
    let lp = Tensor::zeros(&[2, 0, 5], DType::F32, &Device::Cpu).unwrap();
    let tgt = Tensor::zeros(&[0, 1], DType::I64, &Device::Cpu).unwrap();
    assert!(ctc_loss(&lp, &tgt, 0, false).is_err());
}

#[test]
fn r2_paged_attention_quantized_validates_inputs() {
    use oxideon::serving::{
        BlockTable, QuantKVDType, QuantizedPagedKVCache, paged_attention_quantized,
    };
    let cache = QuantizedPagedKVCache::new(8, 16, 4, 16, QuantKVDType::Fp8E4M3).unwrap();
    let table = || BlockTable {
        physical_blocks: vec![0],
        num_tokens: 1,
    };
    let q = Tensor::randn(&[1, 4, 16]).unwrap();
    // Wrong head count / head dim (used to slice q_data out of bounds).
    let bad_heads = Tensor::randn(&[1, 3, 16]).unwrap();
    assert!(paged_attention_quantized(&bad_heads, &cache, &[table()], &[1], 1.0).is_err());
    let bad_dim = Tensor::randn(&[1, 4, 8]).unwrap();
    assert!(paged_attention_quantized(&bad_dim, &cache, &[table()], &[1], 1.0).is_err());
    // Short tables / seq_lens (used to index [b] out of bounds).
    assert!(paged_attention_quantized(&q, &cache, &[], &[1], 1.0).is_err());
    assert!(paged_attention_quantized(&q, &cache, &[table()], &[], 1.0).is_err());
    // Table too short for the sequence (used to index physical_blocks OOB).
    let short = BlockTable {
        physical_blocks: vec![],
        num_tokens: 0,
    };
    assert!(paged_attention_quantized(&q, &cache, &[short], &[20], 1.0).is_err());
    // Zero block size (used to divide by zero).
    let zero_bs = QuantizedPagedKVCache::new(8, 0, 4, 16, QuantKVDType::Fp8E4M3).unwrap();
    assert!(paged_attention_quantized(&q, &zero_bs, &[table()], &[1], 1.0).is_err());
    // Valid call still works.
    let out = paged_attention_quantized(&q, &cache, &[table()], &[1], 0.25).unwrap();
    assert_eq!(out.shape(), &[1, 4, 16]);
}

#[test]
fn r2_accept_longest_path_cycle_is_an_error() {
    use oxideon::serving::TreeSpeculativeDecoder;
    // Cyclic parents (used to loop forever, growing the path unboundedly).
    assert!(TreeSpeculativeDecoder::accept_longest_path(&[7, 7], &[7, 7], &[1, 0]).is_err());
    let (path, n) =
        TreeSpeculativeDecoder::accept_longest_path(&[10, 20, 30], &[10, 20, 99], &[0, 0, 1])
            .unwrap();
    assert_eq!((path, n), (vec![10, 20], 2));
}

#[test]
fn r2_jit_multi_output_cpu_execution() {
    let mut b = JitKernelBuilder::new("two_out");
    let x = b.add_input("x");
    let y = b.add_input("y");
    let sum = b.add_op(JitOp::Add, &[x, y]);
    let prod = b.add_op(JitOp::Mul, &[x, y]);
    b.set_output(sum);
    b.set_output(prod);
    let t1 = Tensor::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
    let t2 = Tensor::from_slice(&[3.0f32, 4.0], &[2]).unwrap();
    // Single-output entry point refuses multi-output kernels loudly.
    assert!(b.execute_cpu(&[&t1, &t2]).is_err());
    // execute_cpu_all returns every output in registration order.
    let outs = b.execute_cpu_all(&[&t1, &t2]).unwrap();
    assert_eq!(outs.len(), 2);
    assert_eq!(outs[0].to_vec::<f32>().unwrap(), vec![4.0, 6.0]);
    assert_eq!(outs[1].to_vec::<f32>().unwrap(), vec![3.0, 8.0]);
}

#[test]
fn r2_gguf_nested_array_is_an_error_not_a_stack_overflow() {
    use oxideon::serialize::gguf::{GGUF_MAGIC, GGUFFile};
    // Minimal container: magic, version 3, 0 tensors, 1 metadata KV whose
    // value is an array (type 9) of arrays (item type 9).
    let mut buf = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes());
    buf.push(b'k');
    buf.extend_from_slice(&9u32.to_le_bytes());
    buf.extend_from_slice(&9u32.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes());
    assert!(GGUFFile::from_bytes(&buf).is_err());
}

#[test]
fn r2_tpu_cluster_huge_grid_is_an_error_not_a_hang() {
    use oxideon::device::{TpuCluster, TpuGeneration};
    assert!(TpuCluster::new(TpuGeneration::V5e, (1_000_000, 1_000_000, 1_000_000)).is_err());
    assert!(TpuCluster::new(TpuGeneration::V5e, (1 << 32, 1 << 32, 1)).is_err());
    let c = TpuCluster::new(TpuGeneration::V5e, (2, 2, 1)).unwrap();
    assert_eq!(c.total_cores(), 8);
    let devs = c.devices();
    assert_eq!(devs.len(), 8);
    assert_eq!(devs[0], Device::Tpu { chip: 0, core: 0 });
}

#[test]
fn r2_tiered_memory_accounting_overflow_is_an_error() {
    use oxideon::device::{MemoryPriority, TieredMemoryManager};
    let gpu = Device::Cuda(0);
    let mut m = TieredMemoryManager::new(0.9);
    m.register_device_capacity(gpu.clone(), usize::MAX);
    m.register_tensor(1, usize::MAX - 10, gpu.clone(), MemoryPriority::Normal)
        .unwrap();
    // (MAX - 10) + 20 overflows: must Err, not panic or wrap.
    assert!(
        m.register_tensor(2, 20, gpu, MemoryPriority::Normal)
            .is_err()
    );
    // Same for the CPU fast path, which skips the pressure check.
    let mut m2 = TieredMemoryManager::new(0.9);
    m2.register_tensor(1, usize::MAX - 10, Device::Cpu, MemoryPriority::Normal)
        .unwrap();
    assert!(
        m2.register_tensor(2, 20, Device::Cpu, MemoryPriority::Normal)
            .is_err()
    );
}

#[test]
fn r2_double_buffered_swap_is_atomic_and_self_healing() {
    use oxideon::serving::DoubleBufferedKVCache;
    use std::sync::atomic::Ordering;
    let cache = std::sync::Arc::new(DoubleBufferedKVCache::new(2, 4, 1, 4, DType::F32).unwrap());
    // Concurrent swaps must not lose updates: 4 threads x 500 swaps = 2000
    // (even) must land exactly back on bank A.
    let mut handles = Vec::new();
    for _ in 0..4 {
        let c = cache.clone();
        handles.push(std::thread::spawn(move || {
            for _ in 0..500 {
                c.swap_banks();
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(cache.active_bank.load(Ordering::SeqCst), 0);
    // Out-of-range stored values self-heal instead of underflowing.
    cache.active_bank.store(2, Ordering::SeqCst);
    assert!(std::ptr::eq(cache.active_cache(), &cache.bank_a));
    cache.swap_banks();
    assert_eq!(cache.active_bank.load(Ordering::SeqCst), 1);
}

#[test]
fn r2_retarget_follows_source_dtype_and_moves_grad() {
    let t = Tensor::from_slice(&[1.0f32, 2.0], &[2])
        .unwrap()
        .requires_grad_(true);
    t.set_grad(Some(Tensor::from_slice(&[0.5f32, 0.5], &[2]).unwrap()));
    let moved = t.to_dtype(DType::F64).unwrap();
    t.retarget(&moved).unwrap();
    assert_eq!(t.dtype(), DType::F64);
    assert_eq!(t.to_vec::<f64>().unwrap(), vec![1.0, 2.0]);
    assert!(t.requires_grad());
    let g = t.grad().expect("grad moved alongside");
    assert_eq!(g.dtype(), DType::F64);
    // Layout mismatch is rejected (a &self method cannot swap layouts).
    let other = Tensor::zeros(&[1, 2], DType::F64, &Device::Cpu).unwrap();
    assert!(t.retarget(&other).is_err());
}

#[test]
fn r2_persistent_homology_reports_every_h0_survivor() {
    use oxideon::science::tda::PersistentHomologyEngine;
    // Three mutually distant points: three surviving H0 bars, no merges.
    let dist = Tensor::from_slice(&[0.0, 9.0, 9.0, 9.0, 0.0, 9.0, 9.0, 9.0, 0.0], &[3, 3]).unwrap();
    let diag = PersistentHomologyEngine::new(1, 2.5)
        .compute_from_distance_matrix(&dist)
        .unwrap();
    let h0 = diag.dimension_pairs(0);
    assert_eq!(h0.len(), 3);
    assert!(h0.iter().all(|p| p.birth == 0.0 && p.death == 2.5));
    // Empty input: no phantom survivor bar.
    let empty = Tensor::zeros(&[0, 0], DType::F64, &Device::Cpu).unwrap();
    let diag_empty = PersistentHomologyEngine::new(1, 2.5)
        .compute_from_distance_matrix(&empty)
        .unwrap();
    assert!(diag_empty.dimension_pairs(0).is_empty());
}

#[test]
fn r2_dropout_p1_zeroes_everything_torch_parity() {
    let x4 = Tensor::ones(&[1, 2, 2, 2], DType::F32, &Device::Cpu).unwrap();
    let d2 = Dropout2d::new(1.0);
    let y = d2.forward(&x4).unwrap();
    assert!(y.to_vec::<f32>().unwrap().iter().all(|&v| v == 0.0));
    let x2 = Tensor::ones(&[2, 4], DType::F32, &Device::Cpu).unwrap();
    let ad = AlphaDropout::new(1.0);
    let ya = ad.forward(&x2).unwrap();
    assert!(ya.to_vec::<f32>().unwrap().iter().all(|&v| v == 0.0));
    let dp = DropPath::new(1.0);
    let yd = dp.forward(&x4).unwrap();
    assert!(yd.to_vec::<f32>().unwrap().iter().all(|&v| v == 0.0));
    // Rank validation still applies.
    let bad = Tensor::ones(&[4], DType::F32, &Device::Cpu).unwrap();
    assert!(d2.forward(&bad).is_err());
}

#[test]
fn r2_symplectic_topology_rejects_absurd_dof() {
    use oxideon::physics::symplectic_topology::SymplecticTopologyEngine;
    assert!(SymplecticTopologyEngine::new(0).is_err());
    assert!(SymplecticTopologyEngine::new(usize::MAX).is_err());
    assert!(SymplecticTopologyEngine::new(1 << 40).is_err());
    // Hand-mutated num_dof (public field) must Err, not wrap/panic.
    let mut eng = SymplecticTopologyEngine::new(2).unwrap();
    eng.num_dof = 1 << 62;
    assert!(eng.symplecticity_defect(&[]).is_err());
    // Valid small case still computes (identity is symplectic: defect ~ 0).
    let eng2 = SymplecticTopologyEngine::new(1).unwrap();
    let ident = [1.0, 0.0, 0.0, 1.0];
    assert!(eng2.symplecticity_defect(&ident).unwrap() < 1e-12);
}

#[test]
fn r2_lbm_rejects_nan_omega() {
    use oxideon::physics::relativistic_lbm::RelativisticLbm1D;
    let rho = vec![1.0; 4];
    let vel = vec![0.0; 4];
    assert!(RelativisticLbm1D::new(&rho, &vel, f64::NAN).is_err());
    assert!(RelativisticLbm1D::new(&rho, &vel, 0.0).is_err());
    assert!(RelativisticLbm1D::new(&rho, &vel, 2.0).is_err());
    assert!(RelativisticLbm1D::new(&rho, &vel, 1.0).is_ok());
}

// ---------------------------------------------------------------------------
// Round 3: reserve-before-check, config/tensor arithmetic mixing, Meta parity.
// ---------------------------------------------------------------------------

#[test]
fn r3_repeat_interleave_guards_and_zero_is_empty() {
    let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
    // Overflowing / over-limit repeats are refused before any reservation.
    assert!(t.repeat_interleave(usize::MAX, 0).is_err());
    assert!(t.repeat_interleave(1 << 40, 0).is_err());
    // torch parity (probed on torch 2.15): zero repeats -> empty dim.
    let e = t.repeat_interleave(0, 0).unwrap();
    assert_eq!(e.shape(), &[0]);
    let ok = t.repeat_interleave(2, 0).unwrap();
    assert_eq!(
        ok.to_vec::<f32>().unwrap(),
        vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    );
}

#[test]
fn r3_repeat_skips_empty_axes_and_doubles() {
    // An empty axis stays empty under astronomic repeats (no hang).
    let e = Tensor::zeros(&[0, 4], DType::F32, &Device::Cpu).unwrap();
    assert_eq!(e.repeat(&[1 << 40, 2]).unwrap().shape(), &[0, 8]);
    let e2 = Tensor::zeros(&[2, 0], DType::F32, &Device::Cpu).unwrap();
    assert_eq!(e2.repeat(&[3, 1 << 40]).unwrap().shape(), &[6, 0]);
    // Doubling still produces the right values.
    let t = Tensor::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
    assert_eq!(
        t.repeat(&[3]).unwrap().to_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    );
}

#[test]
fn r3_one_hot_refuses_huge_classes_up_front() {
    let t = Tensor::from_slice(&[0i64, 1], &[2]).unwrap();
    assert!(t.one_hot(usize::MAX, DType::F32).is_err());
    assert!(t.one_hot(1 << 40, DType::F32).is_err());
    let ok = t.one_hot(3, DType::F32).unwrap();
    assert_eq!(ok.shape(), &[2, 3]);
}

#[test]
fn r3_unbind_split_fanout_is_capped() {
    let big = Tensor::zeros(&[70000], DType::F32, &Device::Cpu).unwrap();
    assert!(big.unbind(0).is_err());
    let big2 = Tensor::zeros(&[100000], DType::F32, &Device::Cpu).unwrap();
    assert!(big2.split(1, 0).is_err());
    let small = Tensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap();
    assert_eq!(small.unbind(0).unwrap().len(), 4);
    assert_eq!(small.split(2, 0).unwrap().len(), 2);
}

#[test]
fn r3_einsum_outer_product_is_capped() {
    // Disjoint labels multiply INDEPENDENT sizes: 2^16 * 2^16 exceeds the limit.
    let a = Tensor::zeros(&[1 << 16], DType::F32, &Device::Cpu).unwrap();
    let b = Tensor::zeros(&[1 << 16], DType::F32, &Device::Cpu).unwrap();
    assert!(Tensor::einsum("a,b->ab", &[&a, &b]).is_err());
    let x = Tensor::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
    let y = Tensor::from_slice(&[3.0f32, 4.0], &[2]).unwrap();
    let d = Tensor::einsum("i,i->", &[&x, &y]).unwrap();
    assert_eq!(d.to_vec::<f32>().unwrap(), vec![11.0]);
}

#[test]
fn r3_meta_matmul_matches_cpu_validation() {
    // Contracting-dim mismatch must fail on Meta exactly like on Cpu.
    let a = Tensor::zeros(&[2, 3], DType::F32, &Device::Meta).unwrap();
    let b = Tensor::zeros(&[5, 4], DType::F32, &Device::Meta).unwrap();
    assert!(a.matmul(&b).is_err());
    // Batch dims broadcast ([1,..] x [7,..] -> [7,..]), not copied from lhs.
    let c = Tensor::zeros(&[1, 2, 3], DType::F32, &Device::Meta).unwrap();
    let d = Tensor::zeros(&[7, 3, 5], DType::F32, &Device::Meta).unwrap();
    assert_eq!(c.matmul(&d).unwrap().shape(), &[7, 2, 5]);
    // Dtype mismatch must fail as well.
    let e = Tensor::zeros(&[2, 3], DType::F64, &Device::Meta).unwrap();
    let f = Tensor::zeros(&[3, 4], DType::F32, &Device::Meta).unwrap();
    assert!(e.matmul(&f).is_err());
}

#[test]
fn r3_moe_rejects_zero_d_model() {
    use oxideon::nn::AuxFreeRouter;
    use oxideon::nn::SparseMoE;
    assert!(SparseMoE::new(0, 8, 2, 1).is_err());
    assert!(AuxFreeRouter::new(0, 4, 2, 0.01).is_err());
    assert!(AuxFreeRouter::new(8, 1 << 30, 1, 0.01).is_err());
    assert!(SparseMoE::new(8, 16, 2, 1).is_ok());
}

#[test]
fn r3_kan_handles_zero_features() {
    use oxideon::nn::KANLinear;
    // in_features=0 used to divide by zero (numel / d_in) in forward.
    let kan = KANLinear::new(0, 4, 8, (0.0, 1.0)).unwrap();
    let x = Tensor::zeros(&[5, 0], DType::F32, &Device::Cpu).unwrap();
    assert_eq!(kan.forward(&x).unwrap().shape(), &[5, 4]);
    assert!(KANLinear::new(usize::MAX, 4, 8, (0.0, 1.0)).is_err());
}

#[test]
fn r3_titans_rejects_unmaterializable_memory() {
    use oxideon::nn::{TitansMemoryConfig, TitansNeuralMemory};
    assert!(TitansNeuralMemory::new(TitansMemoryConfig::new(4, 1 << 40)).is_err());
    assert!(TitansNeuralMemory::new(TitansMemoryConfig::new(0, usize::MAX)).is_err());
    assert!(TitansNeuralMemory::new(TitansMemoryConfig::new(4, 8)).is_ok());
}

#[test]
fn r3_deq_rejects_absurd_iteration_caps() {
    use oxideon::nn::{DeepEquilibriumModel, DeqConfig};
    let cfg = DeqConfig {
        max_forward_iters: usize::MAX,
        ..DeqConfig::default()
    };
    assert!(DeepEquilibriumModel::new(4, 4, cfg).is_err());
    let cfg2 = DeqConfig {
        max_backward_iters: usize::MAX,
        ..DeqConfig::default()
    };
    assert!(DeepEquilibriumModel::new(4, 4, cfg2).is_err());
    assert!(DeepEquilibriumModel::new(4, 4, DeqConfig::default()).is_ok());
}

#[test]
fn r3_upsample_rejects_empty_spatial() {
    let t = Tensor::zeros(&[1, 2, 0, 4], DType::F32, &Device::Cpu).unwrap();
    assert!(t.upsample_nearest2d((2, 2)).is_err());
    assert!(t.upsample_bilinear2d((2, 2), false).is_err());
    let ok = Tensor::zeros(&[1, 2, 4, 4], DType::F32, &Device::Cpu).unwrap();
    assert_eq!(
        ok.upsample_nearest2d((2, 2)).unwrap().shape(),
        &[1, 2, 2, 2]
    );
}

#[test]
fn r3_unfold_rejects_overflowing_step() {
    let v = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
    // A wrapped stride could pass bounds checks and silently return wrong windows.
    assert!(v.unfold(0, 2, usize::MAX).is_err());
    assert_eq!(v.unfold(0, 3, 1).unwrap().shape(), &[2, 3]);
}

#[test]
fn r3_fold_and_unfold2d_reject_overflowing_geometry() {
    let cols = Tensor::zeros(&[1, 4, 1], DType::F32, &Device::Cpu).unwrap();
    // kh*kw wraps to 0 -> would divide by zero in the divisibility check.
    assert!(
        cols.fold2d((2, 2), (1 << 32, 1 << 32), (1, 1), (0, 0), (1, 1))
            .is_err()
    );
    let img = Tensor::zeros(&[1, 1, 4, 4], DType::F32, &Device::Cpu).unwrap();
    // dh*(kh-1) overflows -> wrapped eff_k would defeat the kernel check.
    assert!(
        img.unfold2d((1 << 32, 1), (1, 1), (0, 0), (1 << 33, 1))
            .is_err()
    );
    // Sane geometry still works.
    assert_eq!(
        img.unfold2d((2, 2), (1, 1), (0, 0), (1, 1))
            .unwrap()
            .shape(),
        &[1, 4, 9]
    );
}

fn r3_gguf_bytes(entries: &[(String, u32, Vec<u8>)]) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    b.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    for (k, ty, v) in entries {
        b.extend_from_slice(&(k.len() as u64).to_le_bytes());
        b.extend_from_slice(k.as_bytes());
        b.extend_from_slice(&ty.to_le_bytes());
        b.extend_from_slice(v);
    }
    // Pad to the default alignment so the (empty) tensor data section starts in-bounds.
    while b.len() % 32 != 0 {
        b.push(0);
    }
    b
}

#[test]
fn r3_gguf_metadata_u64_never_truncates_to_u32() {
    use oxideon::serialize::gguf::GGUFFile;
    // 2^32 + 5 used to come back as Some(5); now it is None (caller default).
    let bytes = r3_gguf_bytes(&[(
        "k".into(),
        10,
        (u64::from(u32::MAX) + 6).to_le_bytes().to_vec(),
    )]);
    let f = GGUFFile::from_bytes(&bytes).unwrap();
    assert_eq!(f.get_metadata_u32("k"), None);
    let bytes2 = r3_gguf_bytes(&[("k".into(), 10, 5u64.to_le_bytes().to_vec())]);
    let f2 = GGUFFile::from_bytes(&bytes2).unwrap();
    assert_eq!(f2.get_metadata_u32("k"), Some(5));
}

#[test]
fn r3_matmul_output_is_capped_before_gemm_reserve() {
    // Each operand is validated, but the broadcast output (2^8 x 2^12 x 2^12
    // = 2^32) exceeds the limit: refuse before the GEMM reserves it.
    let a = Tensor::zeros(&[1 << 8, 1 << 12, 1], DType::F32, &Device::Cpu).unwrap();
    let b = Tensor::zeros(&[1 << 8, 1, 1 << 12], DType::F32, &Device::Cpu).unwrap();
    assert!(a.matmul(&b).is_err());
    // Empty contraction still yields zeros (KANLinear with 0 features relies on this).
    let c = Tensor::zeros(&[2, 0], DType::F32, &Device::Cpu).unwrap();
    let d = Tensor::zeros(&[0, 3], DType::F32, &Device::Cpu).unwrap();
    let z = c.matmul(&d).unwrap();
    assert_eq!(z.shape(), &[2, 3]);
    assert_eq!(z.to_vec::<f32>().unwrap(), vec![0.0; 6]);
}

#[test]
fn r3_gguf_config_rejects_zero_embedding_length() {
    use oxideon::nn::gguf_model::GGUFModelConfig;
    use oxideon::serialize::gguf::GGUFFile;
    let mut arch_val = (5u64).to_le_bytes().to_vec();
    arch_val.extend_from_slice(b"llama");
    let bytes = r3_gguf_bytes(&[
        ("general.architecture".into(), 8, arch_val),
        (
            "llama.embedding_length".into(),
            4,
            0u32.to_le_bytes().to_vec(),
        ),
    ]);
    let file = GGUFFile::from_bytes(&bytes).unwrap();
    let err = GGUFModelConfig::from_gguf(&file).unwrap_err().to_string();
    assert!(err.contains("embedding_length"), "unexpected error: {err}");
}

#[test]
fn s2_dropout_p1_zeroes_and_clamps() {
    // Base Dropout used to panic on p == 1.0 (assert); torch zeroes instead.
    let d = Dropout::new(1.0);
    let x = Tensor::ones(&[2, 3], DType::F32, &Device::Cpu).unwrap();
    assert_eq!(
        d.forward(&x).unwrap().to_vec::<f32>().unwrap(),
        vec![0.0; 6]
    );
    // Out-of-range p clamps like Dropout2d.
    let d2 = Dropout::new(2.0);
    assert_eq!(
        d2.forward(&x).unwrap().to_vec::<f32>().unwrap(),
        vec![0.0; 6]
    );
    let d0 = Dropout::new(0.0);
    assert_eq!(
        d0.forward(&x).unwrap().to_vec::<f32>().unwrap(),
        vec![1.0; 6]
    );
}

#[test]
fn s2_mha_forward_qkv_rejects_bad_rank() {
    let mha = MultiHeadAttention::new(8, 2).unwrap();
    let bad = Tensor::zeros(&[2, 8], DType::F32, &Device::Cpu).unwrap();
    assert!(mha.forward_qkv(&bad, &bad, &bad, false).is_err());
    let good = Tensor::zeros(&[2, 4, 8], DType::F32, &Device::Cpu).unwrap();
    assert_eq!(
        mha.forward_qkv(&good, &good, &good, false).unwrap().shape(),
        &[2, 4, 8]
    );
}

#[test]
fn s2_linear_nd_zero_last_dim_yields_zeros() {
    // [..., 0] input used to divide 0/0 computing rows; now checked.
    let lin = Linear::with_bias(0, 4, false).unwrap();
    let x = Tensor::zeros(&[2, 5, 0], DType::F32, &Device::Cpu).unwrap();
    let y = lin.forward(&x).unwrap();
    assert_eq!(y.shape(), &[2, 5, 4]);
    assert!(y.to_vec::<f32>().unwrap().iter().all(|&v| v == 0.0));
}

#[test]
fn s2_ckks_encode_rejects_oversize_vectors() {
    // encode() used to silently truncate past ring degree.
    let ctx = CkksContext::new(8).unwrap();
    assert!(ctx.encode(&[1.0; 8]).is_ok());
    assert!(ctx.encode(&[1.0; 9]).is_err());
}

#[test]
fn s3_grpo_group_size_zero_is_safe() {
    use oxideon::rl::{GRPOConfig, GRPOTrainer, compute_group_advantages};
    // Raw fn: no groups form (used to panic inside chunks).
    assert!(compute_group_advantages(&[1.0, 2.0], 0, 1e-6).is_empty());
    assert_eq!(compute_group_advantages(&[1.0, 2.0], 2, 1e-6).len(), 2);
    // Trainer path rejects loudly.
    let trainer = GRPOTrainer::new(GRPOConfig {
        group_size: 0,
        ..Default::default()
    });
    let logps = Tensor::zeros(&[1, 2], DType::F32, &Device::Cpu).unwrap();
    assert!(
        trainer
            .compute_step(&logps, &logps, &logps, &[1.0], None)
            .is_err()
    );
}

#[test]
fn s3_grpo_loss_averages_over_unmasked_tokens() {
    use oxideon::rl::grpo_loss;
    let policy = Tensor::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]).unwrap();
    let old = Tensor::from_slice(&[0.0f32, 0.1, 0.2, 0.3], &[2, 2]).unwrap();
    let refr = Tensor::from_slice(&[0.05f32, 0.15, 0.25, 0.35], &[2, 2]).unwrap();
    let adv = Tensor::from_slice(&[1.0f32, -1.0], &[2]).unwrap();
    let loss_none = grpo_loss(&policy, &old, &refr, &adv, None, 0.2, 0.04)
        .unwrap()
        .item::<f32>()
        .unwrap();
    // All-ones mask must match the unmasked mean.
    let ones = Tensor::ones(&[2, 2], DType::F32, &Device::Cpu).unwrap();
    let loss_ones = grpo_loss(&policy, &old, &refr, &adv, Some(&ones), 0.2, 0.04)
        .unwrap()
        .item::<f32>()
        .unwrap();
    assert!(
        (loss_none - loss_ones).abs() < 1e-6,
        "{loss_none} vs {loss_ones}"
    );
    // All-masked batch yields 0 instead of NaN (0/0).
    let zeros = Tensor::zeros(&[2, 2], DType::F32, &Device::Cpu).unwrap();
    let loss_zero = grpo_loss(&policy, &old, &refr, &adv, Some(&zeros), 0.2, 0.04)
        .unwrap()
        .item::<f32>()
        .unwrap();
    assert_eq!(loss_zero, 0.0);
}

#[test]
fn s3_flash_decoding_rejects_zero_chunk_size() {
    use oxideon::long_context::{FlashDecodingConfig, FlashDecodingEngine};
    let q = Tensor::zeros(&[1, 2, 1, 4], DType::F32, &Device::Cpu).unwrap();
    let kv = Tensor::zeros(&[1, 2, 8, 4], DType::F32, &Device::Cpu).unwrap();
    let mut cfg = FlashDecodingConfig::new(2, 4);
    cfg.split_kv_chunk_size = 0;
    let eng = FlashDecodingEngine::new(cfg);
    assert!(eng.decode_attention(&q, &kv, &kv).is_err());
    let good = FlashDecodingEngine::new(FlashDecodingConfig::new(2, 4));
    assert_eq!(
        good.decode_attention(&q, &kv, &kv).unwrap().shape(),
        &[1, 2, 1, 4]
    );
}

#[test]
fn s3_kzg_opening_quotient_handles_huge_points() {
    use oxideon::crypto::KzgPolynomialCommitment;
    // f = 1 + 2X at z = i64::MAX over F_3: the i64 Horner product
    // overflows (debug panic); the i128 path gives (1 + 2z) mod 3 = 0.
    let kzg = KzgPolynomialCommitment::new(3).unwrap();
    let (y, q) = kzg.compute_opening_quotient(&[1, 2], i64::MAX).unwrap();
    assert_eq!((y, q), (0, vec![2]));
    assert!(KzgPolynomialCommitment::new(2).is_err());
}

#[test]
fn s3_fri_fold_handles_huge_challenges() {
    use oxideon::crypto::FastReedSolomonIopEngine;
    // [1, 2] folded with alpha = i64::MAX over F_3: (1 + 2*MAX) mod 3 = 0.
    let fri = FastReedSolomonIopEngine::new(3).unwrap();
    assert_eq!(
        fri.fold_polynomial_coeffs(&[1, 2], i64::MAX).unwrap(),
        vec![0]
    );
}

#[test]
fn s3_lattice_decrypt_handles_huge_modulus() {
    use oxideon::crypto::LatticeLweEngine;
    // q past i64::MAX/3 overflows (3*q)/4 (debug panic); decrypt must cope.
    let q = i64::MAX / 2 + 1;
    let eng = LatticeLweEngine::new(2, q).unwrap();
    let bits = eng.decrypt(&[0, 0], &[0, 0], &[0, 0]).unwrap();
    assert_eq!(bits.len(), 2);
}

#[test]
fn s4_rvq_rejects_degenerate_geometry() {
    use oxideon::audio::{RVQConfig, ResidualVectorQuantizer, VQCodebookStage};
    // Zero-size stages used to build fine, then panic in quantize_vec.
    assert!(VQCodebookStage::new(0, 4).is_err());
    assert!(VQCodebookStage::new(4, 0).is_err());
    assert!(VQCodebookStage::new(4, 4).is_ok());
    // A zero-codebook quantizer cannot infer sequence length (codes[0] panic).
    let rvq = ResidualVectorQuantizer::new(RVQConfig::new(0, 4, 4)).unwrap();
    assert!(rvq.decode_codes(&[]).is_err());
    // Ordinary geometry still round-trips.
    let rvq2 = ResidualVectorQuantizer::new(RVQConfig::new(2, 4, 4)).unwrap();
    let codes = vec![vec![0, 1], vec![2, 3]];
    assert_eq!(rvq2.decode_codes(&codes).unwrap().shape(), &[1, 2, 4]);
}

#[test]
fn s5_groth16_r1cs_handles_huge_entries() {
    use oxideon::crypto::QuadraticArithmeticProgramEngine;
    // MAX*MAX overflows i64 (debug panic); the i128 path verifies fine.
    // (MAX^2 mod 3, MAX mod 3, MAX mod 3) = (1, 1, 1): 1*1-1 == 0.
    let eng = QuadraticArithmeticProgramEngine::new(3, 1, 1).unwrap();
    let big = i64::MAX;
    assert!(eng.verify_r1cs_witness(&[big], &[1], &[1], &[big]).unwrap());
    // A genuinely unsatisfied gate still fails.
    assert!(!eng.verify_r1cs_witness(&[1], &[1], &[2], &[1]).unwrap());
}

#[test]
fn s5_ttcore_dims_overflow_is_an_error() {
    use oxideon::science::TtCore;
    assert!(TtCore::new(1 << 32, 1 << 32, 2, Vec::new()).is_err());
    assert!(TtCore::new(1, 2, 1, vec![0.0; 2]).is_ok());
}

#[test]
fn s6_vae3d_groups_accept_nonmultiple_channels() {
    use oxideon::video::VideoResBlock3d;
    // 6 % 4 != 0 used to fail GroupNorm divisibility; groups now divide C.
    let block = VideoResBlock3d::new(6, 6).unwrap();
    let x = Tensor::zeros(&[1, 6, 2, 4, 4], DType::F32, &Device::Cpu).unwrap();
    assert_eq!(block.forward(&x).unwrap().shape(), &[1, 6, 2, 4, 4]);
}

#[test]
fn s6_whisper_named_parameters_cover_all_layers() {
    use oxideon::audio::{Whisper, WhisperConfig};
    let cfg = WhisperConfig {
        n_mels: 8,
        d_model: 8,
        encoder_layers: 1,
        encoder_heads: 2,
        decoder_layers: 1,
        decoder_heads: 2,
        d_ff: 16,
        vocab_size: 16,
        max_source_positions: 8,
        max_target_positions: 8,
    };
    let names: Vec<String> = Whisper::new(cfg)
        .unwrap()
        .named_parameters()
        .into_iter()
        .map(|p| p.0)
        .collect();
    for probe in [
        "encoder.layers.0.self_attn.q_proj.weight",
        "encoder.layers.0.fc1.weight",
        "encoder.ln_post.weight",
        "decoder_layers.0.self_attn.q_proj.weight",
        "decoder_layers.0.cross_attn.q_proj.weight",
        "decoder_layers.0.fc1.weight",
        "ln_post.weight",
    ] {
        assert!(names.iter().any(|n| n == probe), "missing {probe}");
    }
}

#[test]
fn s6_videodit_block_applies_rope() {
    use oxideon::video::VideoDiTBlock;
    let block = VideoDiTBlock::new(32, 4, 16).unwrap();
    let x = Tensor::randn(&[1, 8, 32]).unwrap();
    let t_emb = Tensor::randn(&[1, 32]).unwrap();
    let text = Tensor::randn(&[1, 2, 16]).unwrap();
    let y = block.forward(&x, &t_emb, &text, (1, 2, 4)).unwrap();
    assert_eq!(y.shape(), &[1, 8, 32]);
    assert!(y.to_vec::<f32>().unwrap().iter().all(|v| v.is_finite()));
}

#[test]
fn s6_arkani_bcfw_rejects_bypassed_fields() {
    use oxideon::physics::AmplituhedronGrassmannianVolume;
    let good = AmplituhedronGrassmannianVolume::new(6, 1).unwrap();
    assert!(good.bcfw_cell_count_and_dimension().is_ok());
    // Pub fields bypass new(): n < 4 underflowed n - 3, m == 0 divided by 0.
    for (n, k) in [(2, 0), (3, 0), (6, 99)] {
        let mut bad = AmplituhedronGrassmannianVolume::new(6, 1).unwrap();
        bad.num_particles_n = n;
        bad.nmhv_degree_k = k;
        assert!(bad.bcfw_cell_count_and_dimension().is_err());
    }
}

#[test]
fn s6_perfectoid_rejects_huge_prime() {
    use oxideon::science::PerfectoidTiltingEngine;
    let eng = PerfectoidTiltingEngine::new(3).unwrap();
    assert!(eng.teichmuller_sharp_lift(7, 2).is_ok());
    assert!(eng.verify_frobenius_tower(&[1, 1]));
    // u64 prime truncated by `as u32` / overflowed checked_pow before.
    let huge = PerfectoidTiltingEngine::new(u64::MAX).unwrap();
    assert!(huge.teichmuller_sharp_lift(7, 2).is_err());
    assert!(!huge.verify_frobenius_tower(&[1, 1]));
}

#[test]
fn s6_vae3d_named_parameters_are_unique_and_complete() {
    use oxideon::video::VideoVAE3d;
    let names: Vec<String> = VideoVAE3d::new(2, 4, 4)
        .unwrap()
        .named_parameters()
        .into_iter()
        .map(|p| p.0)
        .collect();
    for probe in [
        "enc_conv_in.weight",
        "enc_block.conv1.weight",
        "enc_block.norm1.weight",
        "enc_down.weight",
        "enc_out.weight",
        "dec_conv_in.weight",
        "dec_block.conv2.weight",
        "dec_out.weight",
    ] {
        assert!(names.iter().any(|n| n == probe), "missing {probe}");
    }
    let mut dedup = names.clone();
    dedup.sort();
    dedup.dedup();
    assert_eq!(dedup.len(), names.len(), "colliding parameter names");
}

#[test]
fn s6_causal_oob_intervention_is_an_error() {
    use oxideon::reasoning::StructuralCausalModel;
    let mut scm = StructuralCausalModel::new();
    scm.add_variable("x", Vec::new(), Vec::new(), 0.0, 1.0)
        .unwrap();
    // Out-of-range do() targets used to be silently ignored.
    assert!(scm.sample_interventional(&[(5, 1.0)], 4, None).is_err());
    assert!(scm.sample_interventional(&[(0, 1.0)], 4, None).is_ok());
}

#[test]
fn s6_christoffel_validates_point_and_metric() {
    use oxideon::geom_graphs::RiemannianManifold;
    let m = RiemannianManifold::new(2);
    let eye = Tensor::from_slice(&[1.0f64, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
    let bad_x = Tensor::zeros(&[3], DType::F64, &Device::Cpu).unwrap();
    assert!(
        m.christoffel_symbols(&bad_x, &|_| Ok(eye.clone()), 1e-5)
            .is_err()
    );
    let good_x = Tensor::zeros(&[2], DType::F64, &Device::Cpu).unwrap();
    assert_eq!(
        m.christoffel_symbols(&good_x, &|_| Ok(eye.clone()), 1e-5)
            .unwrap()
            .shape(),
        &[2, 2, 2]
    );
}

#[test]
fn s6_ilp_loss_rejects_oob_targets() {
    use oxideon::reasoning::{DifferentiableILPEngine, TNorm};
    let eng = DifferentiableILPEngine::new(2, 1, TNorm::Product);
    let facts = Tensor::full(&[1, 2, 2], 0.5, DType::F64, &Device::Cpu).unwrap();
    // (predicate, i, j) = (4, 0, 0) used to index out of bounds.
    assert!(eng.evaluate_loss(&facts, &[(4, 0, 0)], &[]).is_err());
    assert!(
        eng.evaluate_loss(&facts, &[(0, 0, 0)], &[(0, 1, 1)])
            .is_ok()
    );
}

#[test]
fn s7_int_pow_is_exact_and_checked() {
    // Exactness incl. 0^0 and (2^20)^3 (naive squaring overflows there).
    let b = Tensor::from_slice(&[2i64, 0, -2, 1 << 20], &[4]).unwrap();
    let e = Tensor::from_slice(&[10i64, 0, 3, 3], &[4]).unwrap();
    assert_eq!(
        b.pow(&e).unwrap().to_vec::<i64>().unwrap(),
        vec![1024, 1, -8, 1 << 60]
    );
    let b32 = Tensor::from_slice(&[3i32, 10], &[2]).unwrap();
    let e32 = Tensor::from_slice(&[5i32, 2], &[2]).unwrap();
    assert_eq!(
        b32.pow(&e32).unwrap().to_vec::<i32>().unwrap(),
        vec![243, 100]
    );
    let b8 = Tensor::from_slice(&[2u8, 3], &[2]).unwrap();
    let e8 = Tensor::from_slice(&[7u8, 3], &[2]).unwrap();
    assert_eq!(b8.pow(&e8).unwrap().to_vec::<u8>().unwrap(), vec![128, 27]);
    // Negative exponents have no integer value (u8 exponents can't be negative).
    let neg64 = Tensor::from_slice(&[-1i64], &[1]).unwrap();
    assert!(
        Tensor::from_slice(&[2i64], &[1])
            .unwrap()
            .pow(&neg64)
            .is_err()
    );
    let neg32 = Tensor::from_slice(&[-1i32], &[1]).unwrap();
    assert!(
        Tensor::from_slice(&[2i32], &[1])
            .unwrap()
            .pow(&neg32)
            .is_err()
    );
    // Overflows are errors, not saturation/wrap.
    let big = Tensor::from_slice(&[1i64 << 62], &[1]).unwrap();
    let two = Tensor::from_slice(&[2i64], &[1]).unwrap();
    assert!(big.pow(&two).is_err());
    let big32 = Tensor::from_slice(&[100_000i32], &[1]).unwrap();
    let two32 = Tensor::from_slice(&[2i32], &[1]).unwrap();
    assert!(big32.pow(&two32).is_err());
    let big8 = Tensor::from_slice(&[16u8], &[1]).unwrap();
    let two8 = Tensor::from_slice(&[2u8], &[1]).unwrap();
    assert!(big8.pow(&two8).is_err());
    // Float pow keeps its powf semantics.
    let f = Tensor::from_slice(&[4.0f32], &[1]).unwrap();
    let h = Tensor::from_slice(&[0.5f32], &[1]).unwrap();
    assert_eq!(f.pow(&h).unwrap().to_vec::<f32>().unwrap(), vec![2.0]);
}

#[test]
fn s7_jump_trigger_is_random_not_state_hash() {
    use oxideon::ode::JumpDiffusionEngine;
    // No Wiener noise, huge jumps: output == x + drift*dt iff no jump.
    // A state-hash trigger would decide identically on all 20 runs.
    let engine = JumpDiffusionEngine::new(1.0, 0.0, 1.0, 99.0);
    let x = Tensor::from_slice(&[1.0f32, -0.5], &[2]).unwrap();
    let drift = x.mul_scalar(-1.0).unwrap();
    let plain = x
        .add(&drift.mul_scalar(0.5).unwrap())
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    oxideon::manual_seed(42);
    let mut jumped = 0;
    for _ in 0..20 {
        let y = engine
            .step(&x, &drift, 0.5)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        if y != plain {
            jumped += 1;
        }
    }
    assert!(jumped > 0 && jumped < 20, "jumps: {jumped}/20");
}

#[test]
fn s7_neural_sde_diffusion_net_is_wired() {
    use oxideon::ode::{JumpDiffusionEngine, NeuralJumpSDE};
    // Zeroed diffusion net + no jumps: output must equal x + drift*dt
    // exactly even with a nonzero scalar scale (the old path added
    // scalar Wiener noise and never consulted the net).
    let engine = JumpDiffusionEngine::new(1.0, 0.7, 0.0, 0.0);
    let sde = NeuralJumpSDE::new(2, engine).unwrap();
    let zw = sde.diffusion_net.weight.zeros_like().unwrap();
    sde.diffusion_net.weight.assign(&zw).unwrap();
    if let Some(b) = &sde.diffusion_net.bias {
        let zb = b.zeros_like().unwrap();
        b.assign(&zb).unwrap();
    }
    let x = Tensor::from_slice(&[1.0f32, -0.5], &[2]).unwrap();
    let drift = sde.drift_net.forward(&x).unwrap();
    let expected = x
        .add(&drift.mul_scalar(0.5).unwrap())
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let y = sde.forward_step(&x, 0.5).unwrap().to_vec::<f32>().unwrap();
    assert_eq!(y, expected);
}

#[test]
fn s5_powi_range_guards() {
    use oxideon::science::{
        AltCaffarelliFreeBoundaryWeiss, GrothendieckLefschetzTraceFormula,
        HardyLittlewoodCircleEngine,
    };
    // u32/usize exponents silently wrapped via `as i32`.
    let g = GrothendieckLefschetzTraceFormula::new(2, 1).unwrap();
    assert!(g.hasse_weil_deligne_point_bounds(1).is_ok());
    assert!(g.hasse_weil_deligne_point_bounds(u32::MAX).is_err());
    assert!(HardyLittlewoodCircleEngine::weyl_exponential_sum_modulus(0.5, 2, 2).is_ok());
    assert!(HardyLittlewoodCircleEngine::weyl_exponential_sum_modulus(0.5, u32::MAX, 2).is_err());
    let ac = AltCaffarelliFreeBoundaryWeiss::new(1.0, 2).unwrap();
    assert!(
        ac.bernoulli_slope_and_weiss_energy(1.0, 1.0, 1.0, 1.0)
            .is_ok()
    );
    let huge = AltCaffarelliFreeBoundaryWeiss::new(1.0, usize::MAX).unwrap();
    assert!(
        huge.bernoulli_slope_and_weiss_energy(1.0, 1.0, 1.0, 1.0)
            .is_err()
    );
}

#[test]
fn eagle_draft_rejects_tampered_depth() {
    // `draft_depth` is a pub field: `with_capacity(draft_depth)` aborted on giant counts.
    let mut engine = EagleSpeculativeEngine::new(EagleConfig::new(8, 16, 2)).unwrap();
    let hidden = Tensor::zeros(&[1, 1, 8], DType::F32, &Device::Cpu).unwrap();
    assert!(engine.generate_draft(&hidden, 0).is_ok());
    engine.config.draft_depth = usize::MAX;
    assert!(engine.generate_draft(&hidden, 0).is_err());
}

#[test]
fn eagle_draft_rejects_unrepresentable_token_id() {
    // `last_token as i64` wrapped ids >= 2^63 to negative, which embedding lookup
    // silently wraps again to a valid row (wrong draft, no error).
    let engine = EagleSpeculativeEngine::new(EagleConfig::new(8, 16, 2)).unwrap();
    let hidden = Tensor::zeros(&[1, 1, 8], DType::F32, &Device::Cpu).unwrap();
    assert!(engine.generate_draft(&hidden, 0).is_ok());
    assert!(engine.generate_draft(&hidden, usize::MAX).is_err());
}

#[test]
fn chunked_prefill_rejects_unrepresentable_token_id() {
    // Same `as i64` wrap as the Eagle draft loop, on the prefill token path.
    let model = CausalLM::new(32, 32, 4, 2, 64, 128).unwrap();
    let mut kv_cache = KVCache::new(2).unwrap();
    let engine = ChunkedPrefillEngine::new(4).unwrap();
    assert!(engine.prefill(&model, &[1, 2, 3], &mut kv_cache).is_ok());
    assert!(
        engine
            .prefill(&model, &[usize::MAX], &mut kv_cache)
            .is_err()
    );
}

#[test]
fn interleaver_chunk_ack_saturates_huge_counts() {
    // `processed_tokens += chunk_len` overflowed (debug panic) once the running
    // total was nonzero and a huge ack arrived.
    let mut sched = PrefillDecodeInterleaver::new(8, 4).unwrap();
    let id = sched.submit_request(vec![1, 2, 3, 4], SamplingConfig::default());
    sched.on_chunk_processed(id, 2);
    sched.on_chunk_processed(id, usize::MAX);
    assert!(sched.pending_prefills.iter().all(|p| p.request_id != id));
    assert!(sched.active_decodes.iter().any(|d| d.request_id == id));
}

#[test]
fn column_parallel_new_rejects_absurd_world() {
    // `Vec::with_capacity(world_size)` aborted before any shard was built.
    assert!(ColumnParallelLinear::new(1, 0, usize::MAX).is_err());
    assert!(ColumnParallelLinear::new(4, 8, 2).is_ok());
}

#[test]
fn parallel_from_linear_rejects_absurd_world() {
    // The from_linear constructors inherit the chunk/split fan-out cap; this pins it.
    let col = Linear::new(1, 140_000).unwrap();
    assert!(ColumnParallelLinear::from_linear(&col, 70_000).is_err());
    assert!(ColumnParallelLinear::from_linear(&col, 2).is_ok());
    let row = Linear::new(140_000, 4).unwrap();
    assert!(RowParallelLinear::from_linear(&row, 70_000).is_err());
    assert!(RowParallelLinear::from_linear(&row, 2).is_ok());
}

#[test]
fn dualpipe_schedule_rejects_tampered_counts() {
    // `num_microbatches` is pub: the steady-state + cooldown loops materialized
    // one step per microbatch with no use-site cap (hang/OOM on tampered values).
    let mut sched = DualPipeSchedule::new(2, 4).unwrap();
    assert!(sched.generate_schedule_for_rank(0).is_ok());
    sched.num_microbatches = 70_000;
    assert!(sched.generate_schedule_for_rank(0).is_err());
}

#[test]
fn zero_adamw_rejects_absurd_world() {
    // `vec![Vec::new(); world_size]` aborted on giant rank counts.
    let p = Tensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap();
    assert!(ZeROAdamW::new(vec![], 0.01, 0.0, usize::MAX).is_err());
    assert!(ZeROAdamW::new(vec![p], 0.01, 0.0, 1).is_ok());
}

#[test]
fn zero_adamw_step_rejects_tampered_owner() {
    // `rank_states` is pub: a tampered `(param_idx, m, v)` entry indexed
    // `params` out of bounds instead of erroring.
    let p = Tensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap();
    p.set_grad(Some(Tensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap()));
    let mut opt = ZeROAdamW::new(vec![p], 0.01, 0.0, 1).unwrap();
    assert!(opt.step().is_ok());
    let m = Tensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap();
    let v = Tensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap();
    opt.rank_states[0].push((usize::MAX, m, v));
    assert!(opt.step().is_err());
}

#[test]
fn zero3_rejects_absurd_world() {
    // `Vec::with_capacity(world_size)` aborted on giant rank counts.
    let t = Tensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap();
    assert!(ZeRO3Parameter::from_parameter(&t, usize::MAX).is_err());
    assert!(ZeRO3Parameter::from_parameter(&t, 2).is_ok());
}

#[test]
fn zero3_gather_rejects_shape_product_overflow() {
    // `original_shape` is pub: `.product::<usize>()` overflowed (debug panic)
    // on tampered shapes instead of erroring.
    let t = Tensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap();
    // Fresh instance per case: a successful gather caches and would mask the product.
    let mut legit = ZeRO3Parameter::from_parameter(&t, 2).unwrap();
    assert!(legit.gather().is_ok());
    let mut z = ZeRO3Parameter::from_parameter(&t, 2).unwrap();
    z.original_shape = vec![usize::MAX, usize::MAX];
    assert!(z.gather().is_err());
}

#[test]
fn multi_krum_rejects_absurd_byzantine_count() {
    use oxideon::distributed::ByzantineAggregator;
    // `2 * num_byzantine + 2` overflowed (debug panic) on giant counts.
    let t = Tensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap();
    assert!(ByzantineAggregator::multi_krum(&[t.clone(), t.clone()], 0, 1).is_ok());
    // Absurd f saturates into the single-worker fallback path instead of panicking.
    assert!(ByzantineAggregator::multi_krum(&[t], usize::MAX, 1).is_ok());
}

#[test]
fn bulyan_rejects_absurd_byzantine_count() {
    use oxideon::distributed::ByzantineAggregator;
    // `4 * num_byzantine + 3` overflowed (debug panic) on giant counts.
    let t = Tensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap();
    let grads = vec![t.clone(), t.clone(), t.clone()];
    assert!(ByzantineAggregator::bulyan(&grads, 0).is_ok());
    // Absurd f saturates into the documented trimmed-mean fallback instead of panicking.
    assert!(ByzantineAggregator::bulyan(&grads, usize::MAX).is_ok());
}

#[test]
fn parallel3d_groups_reject_tampered_dims() {
    // Config dims are pub: the group loops re-validate instead of trusting them.
    let config = Parallel3DConfig::new(2, 2, 2).unwrap();
    let mut rt = Parallel3DRuntime::new(config, vec![Device::Cpu; 8]).unwrap();
    assert!(rt.get_dp_group(0).is_ok());
    rt.config.dp_size = 70_000;
    assert!(rt.get_dp_group(0).is_err());
}

#[test]
fn flash_attention_2_backward_rejects_low_rank_inputs() {
    // `q.shape()[2]` indexed directly: 2D inputs panicked instead of erroring.
    let cfg = FlashAttention2Config::default();
    let bad = Tensor::randn(&[4, 4]).unwrap();
    let good = Tensor::randn(&[1, 1, 2, 4]).unwrap();
    let lse = Tensor::zeros(&[1, 1, 2, 1], DType::F32, &Device::Cpu).unwrap();
    assert!(flash_attention_2_backward(&good, &bad, &good, &good, &good, &lse, &cfg).is_err());
    assert!(flash_attention_2_backward(&good, &good, &good, &good, &good, &lse, &cfg).is_ok());
}

#[test]
fn ring_attention_rejects_tampered_rank() {
    // `rank` is pub: `rank + world_size - step` overflowed (debug panic) on
    // tampered ranks instead of erroring.
    let mut ring = RingAttention::new(2, 0, false).unwrap();
    let q = Tensor::randn(&[1, 2, 8, 16]).unwrap();
    let k = Tensor::randn(&[1, 2, 8, 16]).unwrap();
    let v = Tensor::randn(&[1, 2, 8, 16]).unwrap();
    assert!(
        ring.forward_simulated(&q, &[k.clone(), k.clone()], &[v.clone(), v.clone()])
            .is_ok()
    );
    ring.rank = usize::MAX;
    assert!(
        ring.forward_simulated(&q, &[k.clone(), k], &[v.clone(), v])
            .is_err()
    );
}

#[test]
fn warmup_cosine_rejects_absurd_warmup() {
    // `total_steps.max(warmup_steps + 1)` overflowed (debug panic) on giant warmups.
    let mut sched = LinearWarmupCosineLr::new(0.1, usize::MAX, 10, 0.0);
    let mut opt = Sgd::new(vec![], 0.1);
    assert!(sched.step(&mut opt).is_finite());
}

#[test]
fn lookahead_step_tolerates_zeroed_interval() {
    // `k_steps` is pub: `step_counter % k_steps` divided by zero (panic)
    // once tampered to 0.
    let p = Tensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap();
    p.set_grad(Some(Tensor::ones(&[4], DType::F32, &Device::Cpu).unwrap()));
    let inner = Sgd::new(vec![p.clone()], 0.01);
    let mut la = Lookahead::new(inner, vec![p], 5, 0.5);
    la.k_steps = 0;
    assert!(la.step().is_ok());
}

#[test]
fn stiefel_retract_handles_empty_rows() {
    // `col[imax]` indexed an empty column (panic) for 0-row inputs.
    let m = ManifoldType::Stiefel { p: 3 };
    let x = Tensor::zeros(&[0, 3], DType::F32, &Device::Cpu).unwrap();
    let xi = Tensor::zeros(&[0, 3], DType::F32, &Device::Cpu).unwrap();
    let q = m.retract(&x, &xi).unwrap();
    assert_eq!(q.shape(), &[0, 3]);
}

#[test]
fn riemannian_adam_step_rejects_length_mismatch() {
    // `params`/`manifolds`/`m`/`v`/`t` are all pub: a shortened `manifolds`
    // indexed out of bounds (panic) instead of erroring.
    let p1 = Tensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap();
    let p2 = Tensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap();
    for p in [&p1, &p2] {
        p.set_grad(Some(Tensor::ones(&[4], DType::F32, &Device::Cpu).unwrap()));
    }
    let mut opt = RiemannianAdam::new(
        vec![p1, p2],
        vec![ManifoldType::Sphere, ManifoldType::Sphere],
        0.01,
    )
    .unwrap();
    assert!(opt.step().is_ok());
    opt.manifolds.pop();
    assert!(opt.step().is_err());
}

#[test]
fn riemannian_adam_step_saturates_huge_counter() {
    // `t` is pub: `t += 1` overflowed (debug panic) at i32::MAX.
    // Counters are per-param now; saturating still holds elementwise.
    let p = Tensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap();
    p.set_grad(Some(Tensor::ones(&[4], DType::F32, &Device::Cpu).unwrap()));
    let mut opt = RiemannianAdam::new(vec![p], vec![ManifoldType::Sphere], 0.01).unwrap();
    opt.t = vec![i32::MAX];
    assert!(opt.step().is_ok());
    assert_eq!(opt.t, vec![i32::MAX]);
}

#[test]
fn gguf_dequantize_rejects_shape_product_overflow() {
    use std::collections::HashMap;
    // Tensor infos are pub: `.product::<usize>()` on a tampered shape
    // overflowed (debug panic), and fitting-but-huge shapes aborted the reserve.
    let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
    let bytes = serialize_gguf(&HashMap::new(), &[("w", GGMLType::F32, &t)], 32).unwrap();
    let mut file = GGUFFile::from_bytes(&bytes).unwrap();
    assert!(file.dequantize_tensor("w").is_ok());
    file.tensors.get_mut("w").unwrap().shape = vec![usize::MAX, usize::MAX];
    assert!(file.dequantize_tensor("w").is_err());
}

#[test]
fn gguf_fused_gemv_rejects_tampered_rows() {
    use std::collections::HashMap;
    // `vec![0.0; rows]` on a tampered row count aborted before any bounds check ran.
    let t = Tensor::zeros(&[2, 32], DType::F32, &Device::Cpu).unwrap();
    let bytes = serialize_gguf(&HashMap::new(), &[("w", GGMLType::Q8_0, &t)], 32).unwrap();
    let mut file = GGUFFile::from_bytes(&bytes).unwrap();
    let x = Tensor::zeros(&[32], DType::F32, &Device::Cpu).unwrap();
    assert!(file.fused_gemv("w", &x).is_ok());
    file.tensors.get_mut("w").unwrap().shape[0] = usize::MAX;
    assert!(file.fused_gemv("w", &x).is_err());
}

#[test]
fn gguf_serialize_rejects_zero_alignment() {
    use std::collections::HashMap;
    // `len % alignment` divided by zero (panic) for alignment 0.
    assert!(serialize_gguf(&HashMap::new(), &[], 0).is_err());
}

#[test]
fn gguf_serialize_rejects_absurd_alignment() {
    use std::collections::HashMap;
    // Padding `resize` to a near-usize::MAX length aborted the process.
    let t = Tensor::from_slice(&[1.0f32], &[1]).unwrap();
    assert!(serialize_gguf(&HashMap::new(), &[("w", GGMLType::F32, &t)], usize::MAX).is_err());
}

#[test]
fn moe_forward_rejects_zeroed_d_model() {
    // `d_model` is pub: `numel / d` divided by zero (panic) once tampered
    // to 0 with an empty-last-dim input.
    let mut moe = SparseMoE::new(4, 8, 2, 1).unwrap();
    let x = Tensor::zeros(&[2, 4], DType::F32, &Device::Cpu).unwrap();
    assert!(moe.forward(&x).is_ok());
    moe.d_model = 0;
    let empty = Tensor::zeros(&[2, 0], DType::F32, &Device::Cpu).unwrap();
    assert!(moe.forward(&empty).is_err());
}

#[test]
fn moe_forward_rejects_expert_count_mismatch() {
    // `num_experts` is pub: `vec![0.0; num_experts]` aborted (capacity
    // panic) once tampered past the real expert vec.
    let mut moe = SparseMoE::new(4, 8, 2, 1).unwrap();
    let x = Tensor::zeros(&[2, 4], DType::F32, &Device::Cpu).unwrap();
    assert!(moe.forward(&x).is_ok());
    moe.num_experts = usize::MAX;
    assert!(moe.forward(&x).is_err());
}

#[test]
fn moe_forward_rejects_zeroed_top_k() {
    // `top_k` is pub: `topk(0)` yields empty indices, and the aux-loss
    // loop indexed `idx_vec[0]` of the empty vec (panic).
    let mut moe = SparseMoE::new(4, 8, 2, 1).unwrap();
    let x = Tensor::zeros(&[2, 4], DType::F32, &Device::Cpu).unwrap();
    moe.top_k = 0;
    assert!(moe.forward(&x).is_err());
}

#[test]
fn apply_view_rejects_out_of_bounds_spec() {
    use oxideon::layout::{DimMap, ViewSpec};
    // `ViewSpec` fields are pub and `apply_view` never bounds-checked the
    // resulting layout: a hand-built strided spec produced an OOB view
    // that panicked on first read instead of erroring here.
    let t = Tensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap();
    let spec = ViewSpec {
        src_shape: vec![4],
        dst_shape: vec![4],
        dim_maps: vec![DimMap::Sliced {
            out_dim: 0,
            start: 0,
            step: 100,
        }],
    };
    assert!(t.apply_view(spec).is_err());
}

#[test]
fn view_scatter_rejects_out_of_bounds_spec() {
    use oxideon::layout::{DimMap, ViewSpec};
    // Same hand-built spec class on the scatter path: `out[dst_off]`
    // indexed out of bounds (panic) instead of erroring.
    let t = Tensor::zeros(&[4], DType::F32, &Device::Cpu).unwrap();
    let spec = ViewSpec {
        src_shape: vec![4],
        dst_shape: vec![4],
        dim_maps: vec![DimMap::Sliced {
            out_dim: 0,
            start: 0,
            step: 100,
        }],
    };
    assert!(t.view_scatter(spec).is_err());
}

#[test]
fn causal_generate_rejects_absurd_token_budget() {
    // The decode loop ran one full forward per requested token with no cap
    // (pre-fix: hangs; verified with `timeout` before the fix). Empty stop
    // set: the default config stops at EOS, which would end the run early.
    let lm = CausalLM::new(16, 8, 2, 1, 16, 32).unwrap();
    let cfg = SamplingConfig {
        max_new_tokens: usize::MAX,
        stop_tokens: vec![],
        ..Default::default()
    };
    assert!(lm.generate_with_config(&[1, 2], &cfg).is_err());
}

#[test]
fn speculative_generate_rejects_absurd_token_budget() {
    // `seq.len() + max_new_tokens` overflowed (debug panic) on giant budgets.
    let target = CausalLM::new(16, 8, 2, 1, 16, 32).unwrap();
    let draft = CausalLM::new(16, 8, 2, 1, 16, 32).unwrap();
    assert!(
        target
            .speculative_generate(&draft, &[1, 2], usize::MAX, 2)
            .is_err()
    );
}

#[test]
fn beam_search_rejects_absurd_beam_width() {
    // `Vec::with_capacity(beams.len() * bw)` aborted (capacity panic), and
    // each step runs `bw` full forwards; vocab-sized widths must Err.
    let lm = CausalLM::new(70_000, 8, 2, 1, 16, 32).unwrap();
    assert!(lm.beam_search(&[1], 1, 70_000, 0.0).is_err());
}

#[test]
fn beam_search_rejects_absurd_token_budget() {
    // The step loop ran one `bw`-wide expansion per requested token with no
    // cap (pre-fix: hangs; verified with `timeout` before the fix).
    let lm = CausalLM::new(16, 8, 2, 1, 16, 32).unwrap();
    assert!(lm.beam_search(&[1], usize::MAX, 1, 0.0).is_err());
}

#[test]
fn beam_search_rejects_zeroed_max_seq_len() {
    // `max_seq_len` is pub: a tampered 0 built an empty window that panicked
    // in the backend (pre-fix) and would underflow `ctx_len - 1` past that.
    let mut lm = CausalLM::new(16, 8, 2, 1, 16, 32).unwrap();
    lm.max_seq_len = 0;
    assert!(lm.beam_search(&[1], 1, 1, 0.0).is_err());
}

#[test]
fn kv_cache_new_rejects_absurd_layer_count() {
    // `(0..num_layers).collect()` aborted (capacity panic) on giant counts.
    assert!(KVCache::new(usize::MAX).is_err());
    assert!(KVCache::new(2).is_ok());
}

#[test]
fn empty_view_with_offset_ops_do_not_panic() {
    // Narrowing axis 1 of a `[0, 4]` tensor yields a valid empty view with
    // offset 2; slicing storage `2..2` panicked (backend `binary_op`,
    // `to_vec`, `to_contiguous`, unary fast paths) instead of yielding empty.
    let t = Tensor::zeros(&[0, 4], DType::F32, &Device::Cpu).unwrap();
    let n = t.narrow(1, 2, 2).unwrap();
    assert_eq!(n.to_vec::<f32>().unwrap(), Vec::<f32>::new());
    let s = n.add(&n).unwrap();
    assert_eq!(s.shape(), &[0, 2]);
    assert_eq!(n.contiguous().unwrap().shape(), &[0, 2]);
}

#[test]
fn causal_forward_rejects_empty_sequence() {
    // End-to-end pin for the empty-view root cause: an empty prompt window
    // used to panic in backend `binary_op`; the ops now flow and the model
    // fails closed with a clear reduction error instead.
    let lm = CausalLM::new(16, 8, 2, 1, 16, 32).unwrap();
    let empty = Tensor::from_slice(&[] as &[i64], &[1, 0]).unwrap();
    let err = lm
        .forward(&empty)
        .expect_err("empty forward must fail closed");
    assert!(err.to_string().contains("empty"), "{err}");
}

#[test]
fn prover_rejects_extreme_nesting() {
    use oxideon::reasoning::{PropExpr, SequentProver};
    // Proof search recurses per connective: a 15k-deep formula overflowed the
    // thread stack (process abort) instead of erroring on a depth limit.
    let mut expr = PropExpr::var("P");
    for _ in 0..15_000 {
        expr = expr.not();
    }
    let prover = SequentProver::new(1_000_000_000);
    assert!(prover.is_tautology(&expr).is_err());
    // A shallow tautology still proves.
    let em = PropExpr::var("P").or(PropExpr::var("P").not());
    assert!(prover.is_tautology(&em).unwrap());
}

#[test]
fn swarm_debate_rejects_absurd_round_budget() {
    // Each round costs a resolve pass; an uncapped budget spun 70k+ rounds.
    let orch = SwarmOrchestrator::new(70_000, 1.0, ConsensusStrategy::MajorityVote);
    let initial = vec![
        AgentProposal::new("a", "X", 0.9, ""),
        AgentProposal::new("b", "Y", 0.9, ""),
    ];
    assert!(orch.debate(initial, |_, props| props.to_vec()).is_err());
}

#[test]
fn mcts_search_rejects_absurd_simulation_budget() {
    // Each simulation walks/expands the tree with no early exit (pre-fix:
    // hangs; verified with `timeout` before the fix).
    let engine = MCTSSearchEngine::new(1.414, usize::MAX, 4);
    assert!(
        engine
            .search("root", |_| vec![("a".to_string(), 0.5)], |_| 0.5)
            .is_err()
    );
    let sane = MCTSSearchEngine::new(1.414, 10, 4);
    assert!(
        sane.search("root", |_| vec![("a".to_string(), 0.5)], |_| 0.5)
            .is_ok()
    );
}

#[test]
fn conformal_regressors_reject_nan_alpha() {
    use oxideon::reasoning::ConformalScoreType;
    // `alpha <= 0.0 || alpha >= 1.0` is false for NaN, so NaN slipped
    // through and silently calibrated to the minimum score.
    assert!(SplitConformalRegressor::new(ConformalScoreType::AbsoluteResidual, f64::NAN).is_err());
    assert!(AdaptiveConformalClassifier::new(f64::NAN).is_err());
    assert!(SplitConformalRegressor::new(ConformalScoreType::AbsoluteResidual, 0.1).is_ok());
}

#[test]
fn conformal_predict_rejects_dispersion_mismatch() {
    use oxideon::reasoning::ConformalScoreType;
    // `calibrate` errors on length mismatch but `predict_intervals` silently
    // truncated via `zip`, returning fewer intervals than inputs.
    let mut reg =
        SplitConformalRegressor::new(ConformalScoreType::NormalizedResidual, 0.1).unwrap();
    let yp = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();
    let yt = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();
    reg.calibrate(&yp, &yt, None).unwrap();
    let bad_sig = Tensor::from_slice(&[1.0, 1.0], &[2]).unwrap();
    assert!(reg.predict_intervals(&yp, Some(&bad_sig)).is_err());
}

#[test]
fn crc_loss_rejects_tampered_bound() {
    use oxideon::reasoning::{ConformalRiskController, RiskMetric};
    // `loss_bound_b` is pub: NaN/negative values panicked inside `f64::clamp`.
    let mut crc = ConformalRiskController::new(RiskMetric::FalseNegativeRate, 0.25, 1.0).unwrap();
    assert!(crc.compute_loss(&[0], &[0]).is_ok());
    crc.loss_bound_b = f64::NAN;
    assert!(crc.compute_loss(&[0], &[0]).is_err());
}

#[test]
fn crc_calibrate_rejects_absurd_grid() {
    use oxideon::reasoning::{ConformalRiskController, RiskMetric};
    // Each grid step costs a full empirical-risk pass; 70k steps spun freely.
    let crc = ConformalRiskController::new(RiskMetric::FalseNegativeRate, 0.25, 1.0).unwrap();
    let scores = vec![vec![0.9, 0.1], vec![0.2, 0.8]];
    let truth = vec![vec![0], vec![1]];
    assert!(crc.calibrate(&scores, &truth, 70_000).is_err());
}

#[test]
fn scm_sampling_rejects_absurd_sample_count() {
    // `n_samples * d` overflowed (debug panic) on giant counts.
    let mut scm = StructuralCausalModel::new();
    scm.add_variable("x", vec![], vec![], 0.0, 1.0).unwrap();
    scm.add_variable("y", vec![0], vec![1.0], 0.0, 1.0).unwrap();
    assert!(scm.sample_observational(usize::MAX, None).is_err());
}

#[test]
fn scm_sampling_rejects_tampered_graph() {
    // `variables` is pub: a tampered parent index read out of bounds (panic),
    // and truncated weights indexed out of bounds too.
    let mut scm = StructuralCausalModel::new();
    scm.add_variable("x", vec![], vec![], 0.0, 1.0).unwrap();
    scm.add_variable("y", vec![0], vec![1.0], 0.0, 1.0).unwrap();
    assert!(scm.sample_observational(4, None).is_ok());
    scm.variables[1].parents = vec![5];
    assert!(scm.sample_observational(4, None).is_err());
    let mut scm2 = StructuralCausalModel::new();
    scm2.add_variable("x", vec![], vec![], 0.0, 1.0).unwrap();
    scm2.add_variable("y", vec![0], vec![1.0], 0.0, 1.0)
        .unwrap();
    scm2.variables[1].weights = vec![];
    assert!(scm2.sample_observational(4, None).is_err());
}

#[test]
fn scm_ate_handles_max_seed() {
    // `seed + 1` overflowed (debug panic) for `Some(u64::MAX)` (a valid seed).
    let mut scm = StructuralCausalModel::new();
    scm.add_variable("x", vec![], vec![], 0.0, 1.0).unwrap();
    scm.add_variable("y", vec![0], vec![1.0], 0.0, 1.0).unwrap();
    let ate = scm
        .average_treatment_effect(0, 1, 1.0, 0.0, 8, Some(u64::MAX))
        .unwrap();
    assert!(ate.is_finite());
}

#[test]
fn nbf_solve_rejects_tampered_node_count() {
    // `num_nodes` is pub: `vec![..; n]` aborted (capacity panic) once
    // tampered past the real edge set.
    let mut graph = AlgorithmicGraph::new(2);
    graph.add_edge(0, 1, 1.0, vec![]).unwrap();
    let nbf = NeuralBellmanFord::new(AlgorithmicSemiring::MinPlus, 4);
    assert!(nbf.solve(&graph, 0, 3).is_ok());
    graph.num_nodes = usize::MAX;
    assert!(nbf.solve(&graph, 0, 3).is_err());
}

#[test]
fn nbf_solve_rejects_absurd_step_budget() {
    // Relaxation rounds never converge on negative cycles; 1e8+ rounds spun.
    let mut graph = AlgorithmicGraph::new(1);
    graph.add_edge(0, 0, -1.0, vec![]).unwrap();
    let nbf = NeuralBellmanFord::new(AlgorithmicSemiring::MinPlus, 4);
    assert!(nbf.solve(&graph, 0, 100_000_001).is_err());
}

#[test]
fn nbf_solve_rejects_tampered_edges() {
    // `edges` is pub: a tampered endpoint indexed `current` out of bounds.
    let mut graph = AlgorithmicGraph::new(2);
    graph.add_edge(0, 1, 1.0, vec![]).unwrap();
    let nbf = NeuralBellmanFord::new(AlgorithmicSemiring::MinPlus, 4);
    graph.edges.push(AlgorithmicEdge {
        source: usize::MAX,
        target: 0,
        weight: 1.0,
        features: vec![],
    });
    assert!(nbf.solve(&graph, 0, 3).is_err());
}

#[test]
fn nbf_reconstruct_rejects_out_of_bounds_nodes() {
    // `target` and stored predecessors are plain usizes: both indexed the
    // slice out of bounds (panic) instead of erroring.
    let nbf = NeuralBellmanFord::new(AlgorithmicSemiring::MinPlus, 4);
    assert!(nbf.reconstruct_path(0, usize::MAX, &[Some(0)]).is_err());
    assert!(nbf.reconstruct_path(0, 1, &[Some(0), Some(9)]).is_err());
    assert_eq!(
        nbf.reconstruct_path(0, 1, &[None, Some(0)]).unwrap(),
        vec![0, 1]
    );
}

#[test]
fn pgn_forward_rejects_tampered_edges() {
    // Same pub-edges tamper on the pointer path: `k_data[u*d..]` sliced out
    // of bounds (panic) instead of erroring.
    let mut graph = AlgorithmicGraph::new(2);
    graph.add_edge(0, 1, 1.0, vec![]).unwrap();
    let pgn = PointerGraphNetwork::new(4, None).unwrap();
    let states = Tensor::zeros(&[2, 4], DType::F32, &Device::Cpu).unwrap();
    assert!(pgn.forward(&states, &graph).is_ok());
    graph.edges.push(AlgorithmicEdge {
        source: usize::MAX,
        target: 0,
        weight: 1.0,
        features: vec![],
    });
    assert!(pgn.forward(&states, &graph).is_err());
}

#[test]
fn symbolic_eval_rejects_extreme_nesting() {
    use oxideon::reasoning::OpType;
    // Evaluation recurses per tree level: a 100k-deep hand-built tree
    // overflowed the thread stack (process abort) instead of erroring.
    let mut tree = SymbolicNode::Variable(0);
    for _ in 0..100_000 {
        tree = SymbolicNode::Unary(OpType::Sin, Box::new(tree));
    }
    assert!(tree.try_evaluate(&[1.0]).is_err());
    // The recovered Err leaves a 100k-deep Box chain whose *drop glue*
    // would recurse past the stack; leak it (see bug log R1).
    std::mem::forget(tree);
    let shallow = SymbolicNode::Binary(
        OpType::Add,
        Box::new(SymbolicNode::Variable(0)),
        Box::new(SymbolicNode::Constant(1.0)),
    );
    assert_eq!(shallow.try_evaluate(&[2.0]).unwrap(), 3.0);
}

#[test]
fn symbolic_fit_rejects_wide_targets() {
    // Only `y_shape[0]` was checked: a `[N, K]` target silently paired the
    // first N of N*K values.
    let engine = SymbolicGpEngine::new(SymbolicGpConfig {
        population_size: 4,
        num_generations: 1,
        ..Default::default()
    });
    let x = Tensor::from_slice(&[1.0, 2.0], &[2, 1]).unwrap();
    let bad_y = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    assert!(engine.fit(&x, &bad_y, Some(7)).is_err());
}

#[test]
fn symbolic_fit_rejects_absurd_population() {
    // `with_capacity(population_size)` plus per-tree work per generation.
    let engine = SymbolicGpEngine::new(SymbolicGpConfig {
        population_size: 70_000,
        num_generations: 1,
        ..Default::default()
    });
    let x = Tensor::from_slice(&[1.0, 2.0], &[2, 1]).unwrap();
    let y = Tensor::from_slice(&[1.0, 2.0], &[2]).unwrap();
    assert!(engine.fit(&x, &y, Some(7)).is_err());
}

#[test]
fn symbolic_fit_rejects_absurd_generations() {
    // Each generation re-evaluates the whole population.
    let engine = SymbolicGpEngine::new(SymbolicGpConfig {
        population_size: 2,
        num_generations: 70_000,
        ..Default::default()
    });
    let x = Tensor::from_slice(&[1.0, 2.0], &[2, 1]).unwrap();
    let y = Tensor::from_slice(&[1.0, 2.0], &[2]).unwrap();
    assert!(engine.fit(&x, &y, Some(7)).is_err());
}

#[test]
fn topos_verify_rejects_huge_lattice() {
    use oxideon::reasoning::ToposHeytingEngine;
    // The O(k^3) adjunction scan took 25 s at k=3000 pre-fix (timed probe)
    // and would take years at the 65536 construction cap.
    let topos = ToposHeytingEngine::new(1100).unwrap();
    assert!(topos.verify_heyting_adjunction().is_err());
    assert!(topos.verify_lawvere_tierney_axioms().is_err());
    let small = ToposHeytingEngine::new(5).unwrap();
    assert!(small.verify_heyting_adjunction().unwrap());
    assert!(small.verify_lawvere_tierney_axioms().unwrap());
}

#[test]
fn solomonoff_new_rejects_nan_lengths() {
    use oxideon::reasoning::{AlgorithmicHypothesis, SolomonoffLevinEngine};
    // `NaN <= 0.0` is false, so NaN lengths slipped through and poisoned
    // every downstream aggregate with silent NaNs.
    let bad = AlgorithmicHypothesis {
        code_length_bits: f64::NAN,
        runtime_steps: 10,
        data_misfit_bits: 1.0,
    };
    assert!(SolomonoffLevinEngine::new(vec![bad]).is_err());
    let good = AlgorithmicHypothesis {
        code_length_bits: 8.0,
        runtime_steps: 10,
        data_misfit_bits: 1.0,
    };
    assert!(SolomonoffLevinEngine::new(vec![good]).is_ok());
}

#[test]
fn ilp_infer_rejects_tampered_rules() {
    use oxideon::reasoning::RuleType;
    // `rules` is pub: a tampered predicate id overflowed `head * n * n`.
    let mut engine = DifferentiableILPEngine::new(2, 1, TNorm::Product);
    engine.add_base_rule(0, 0, 1.0).unwrap();
    let facts = Tensor::zeros(&[1, 2, 2], DType::F32, &Device::Cpu).unwrap();
    assert!(engine.infer_step(&facts).is_ok());
    engine.rules[0].rule_type = RuleType::Base {
        head: usize::MAX,
        body: 0,
    };
    assert!(engine.infer_step(&facts).is_err());
}

#[test]
fn ilp_deductive_rejects_absurd_steps() {
    // Each step runs a full consequence pass; 70k steps spun freely.
    let engine = DifferentiableILPEngine::new(2, 1, TNorm::Product);
    let facts = Tensor::zeros(&[1, 2, 2], DType::F32, &Device::Cpu).unwrap();
    assert!(engine.deductive_forward(&facts, 70_000).is_err());
}

#[test]
fn ilp_loss_rejects_tampered_counts() {
    // `num_entities` is pub: `num_predicates * n * n` overflowed on tampering.
    let mut engine = DifferentiableILPEngine::new(2, 1, TNorm::Product);
    let facts = Tensor::zeros(&[1, 2, 2], DType::F32, &Device::Cpu).unwrap();
    assert!(engine.evaluate_loss(&facts, &[(0, 0, 1)], &[]).is_ok());
    engine.num_entities = usize::MAX;
    assert!(engine.evaluate_loss(&facts, &[(0, 0, 1)], &[]).is_err());
}

#[test]
fn gqa_forward_rejects_zeroed_kv_heads() {
    // `num_kv_heads` is pub: `h / kv_h` divided by zero (panic) once tampered to 0.
    let mut gqa = GroupedQueryAttention::new(8, 2, 1).unwrap();
    let x = Tensor::zeros(&[1, 2, 8], DType::F32, &Device::Cpu).unwrap();
    assert!(gqa.forward_gqa(&x, None, None, true).is_ok());
    gqa.num_kv_heads = 0;
    assert!(gqa.forward_gqa(&x, None, None, true).is_err());
}

#[test]
fn hott_odd_dim_transport_pins_last_lane() {
    use oxideon::reasoning::HomotopyTypeEngine;
    // Odd-length vectors keep the final lane: the map is block-diagonal
    // 2D rotations plus a 1x1 identity, still orthogonal.
    let v = vec![1.0, 0.0, 5.0];
    let out = HomotopyTypeEngine::univalent_orthogonal_transport(&v, 0.0).unwrap();
    assert_eq!(out, vec![1.0, 0.0, 5.0]);
    let out = HomotopyTypeEngine::univalent_orthogonal_transport(&v, std::f64::consts::FRAC_PI_2)
        .unwrap();
    assert!((out[0] - 0.0).abs() < 1e-12);
    assert!((out[1] - 1.0).abs() < 1e-12);
    assert_eq!(out[2], 5.0);
    let empty: Vec<f64> = vec![];
    assert!(
        HomotopyTypeEngine::univalent_orthogonal_transport(&empty, 1.0)
            .unwrap()
            .is_empty()
    );
    assert!(HomotopyTypeEngine::verify_univalent_circle_coherence(3).unwrap());
    assert!(HomotopyTypeEngine::decode_winding_number(i64::MAX).is_err());
}

#[test]
fn odeint_rejects_absurd_steps() {
    use oxideon::ode::{OdeMethod, odeint};
    let y0 = Tensor::from_slice(&[1.0f32], &[1]).unwrap();
    assert!(
        odeint(
            |_, y| Ok(y.clone()),
            &y0,
            0.0,
            1.0,
            usize::MAX,
            OdeMethod::Euler
        )
        .is_err()
    );
    assert!(odeint(|_, y| Ok(y.clone()), &y0, 0.0, 1.0, 8, OdeMethod::RK4).is_ok());
}

#[test]
fn odeint_trajectory_rejects_absurd_steps() {
    use oxideon::ode::{OdeMethod, odeint_trajectory};
    let y0 = Tensor::from_slice(&[1.0f32], &[1]).unwrap();
    assert!(
        odeint_trajectory(
            |_, y| Ok(y.clone()),
            &y0,
            0.0,
            1.0,
            usize::MAX,
            OdeMethod::Euler
        )
        .is_err()
    );
    assert!(odeint_trajectory(|_, y| Ok(y.clone()), &y0, 0.0, 1.0, 4, OdeMethod::Midpoint).is_ok());
}

#[test]
fn flow_matcher_rejects_scalar_inputs() {
    use oxideon::ode::FlowMatcher;
    let m = FlowMatcher::new(0.0);
    let s = Tensor::from_slice(&[1.0f32], &[]).unwrap();
    let t = Tensor::from_slice(&[0.5f32], &[1]).unwrap();
    assert!(m.sample_location_and_target(&s, &s, &t).is_err());
}

#[test]
fn latent_simulate_rejects_absurd_steps() {
    use oxideon::ode::{LatentSdeConfig, LatentSdeEngine};
    let cfg = LatentSdeConfig {
        num_steps: usize::MAX,
        ..Default::default()
    };
    let engine = LatentSdeEngine::new(cfg).unwrap();
    let z0 = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[4]).unwrap();
    assert!(engine.simulate(&z0, false, Some(1)).is_err());
}

#[test]
fn latent_simulate_rejects_truncated_weights() {
    use oxideon::ode::{LatentSdeConfig, LatentSdeEngine};
    let mut engine = LatentSdeEngine::new(LatentSdeConfig::default()).unwrap();
    let z0 = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[4]).unwrap();
    assert!(engine.simulate(&z0, false, Some(1)).is_ok());
    engine.prior_drift_w.truncate(3);
    assert!(engine.simulate(&z0, false, Some(1)).is_err());
}

#[test]
fn latent_simulate_rejects_shrunk_latent_dim() {
    use oxideon::ode::{LatentSdeConfig, LatentSdeEngine};
    let mut engine = LatentSdeEngine::new(LatentSdeConfig::default()).unwrap();
    engine.config.latent_dim = 2;
    let z0 = Tensor::from_slice(&[0.0f32, 0.0], &[2]).unwrap();
    assert!(engine.simulate(&z0, false, Some(1)).is_err());
}

#[test]
fn jump_trajectory_rejects_absurd_steps() {
    use oxideon::ode::JumpDiffusionEngine;
    let engine = JumpDiffusionEngine::new(1.0, 0.0, 0.0, 1.0);
    let x0 = Tensor::from_slice(&[1.0f32], &[1]).unwrap();
    assert!(
        engine
            .simulate_trajectory(|x| Ok(x.clone()), &x0, 0.1, usize::MAX)
            .is_err()
    );
    assert!(
        engine
            .simulate_trajectory(|x| Ok(x.clone()), &x0, 0.1, 3)
            .is_ok()
    );
}

#[test]
fn hutchinson_rejects_absurd_samples() {
    use oxideon::ode::JumpDiffusionEngine;
    let x = Tensor::from_slice(&[1.0f32], &[1]).unwrap();
    assert!(JumpDiffusionEngine::hutchinson_divergence(|x| Ok(x.clone()), &x, usize::MAX).is_err());
    assert!(JumpDiffusionEngine::hutchinson_divergence(|x| Ok(x.clone()), &x, 2).is_ok());
}

#[test]
fn jump_step_rejects_nan_intensity() {
    use oxideon::ode::JumpDiffusionEngine;
    let mut engine = JumpDiffusionEngine::new(1.0, 0.0, 0.5, 1.0);
    let x = Tensor::from_slice(&[1.0f32], &[1]).unwrap();
    let drift = Tensor::from_slice(&[0.0f32], &[1]).unwrap();
    assert!(engine.step(&x, &drift, 0.1).is_ok());
    engine.jump_intensity = f64::NAN;
    assert!(engine.step(&x, &drift, 0.1).is_err());
    assert!(engine.step_full(&x, &drift, &drift, 0.1).is_err());
}

#[test]
fn latent_new_rejects_huge_dim() {
    use oxideon::ode::{LatentSdeConfig, LatentSdeEngine};
    let cfg = LatentSdeConfig {
        latent_dim: usize::MAX,
        ..Default::default()
    };
    assert!(LatentSdeEngine::new(cfg).is_err());
    assert!(LatentSdeEngine::new(LatentSdeConfig::default()).is_ok());
}

#[test]
fn latent_eval_drift_rejects_short_slices() {
    use oxideon::ode::LatentSdeEngine;
    assert!(LatentSdeEngine::eval_drift(&[1.0], &[1.0], 4).is_err());
    let w = vec![1.0; 16];
    let z = vec![1.0; 4];
    assert_eq!(
        LatentSdeEngine::eval_drift(&w, &z, 4).unwrap(),
        vec![4.0; 4]
    );
}

#[test]
fn cnf_new_rejects_zero_dim() {
    use oxideon::ode::riemannian_cnf::PoincareCnfFlow;
    assert!(PoincareCnfFlow::new(0).is_err());
    assert!(PoincareCnfFlow::new(2).is_ok());
}

#[test]
fn cnf_integrate_rejects_absurd_steps() {
    use oxideon::ode::riemannian_cnf::PoincareCnfFlow;
    let flow = PoincareCnfFlow::new(2).unwrap();
    let x0 = Tensor::from_slice(&[0.2f32, -0.3], &[2]).unwrap();
    assert!(flow.integrate(&x0, usize::MAX, 1.0).is_err());
    assert!(flow.integrate(&x0, 4, 1.0).is_ok());
}

#[test]
fn cnf_velocity_rejects_truncated_weights() {
    use oxideon::ode::riemannian_cnf::PoincareCnfFlow;
    let mut flow = PoincareCnfFlow::new(2).unwrap();
    assert!(flow.velocity(&[0.0, 0.0], 0.0).is_ok());
    flow.weight_v.truncate(1);
    assert!(flow.velocity(&[0.0, 0.0], 0.0).is_err());
    flow.weight_v = vec![0.0; 4];
    flow.omega.truncate(1);
    assert!(flow.riemannian_divergence(&[0.0, 0.0], 0.0).is_err());
}

#[test]
fn mfg_new_rejects_nonfinite_viscosity() {
    use oxideon::ode::MeanFieldGameSolver1D;
    assert!(MeanFieldGameSolver1D::new(4, f64::NAN, 0.5, vec![0.0; 4]).is_err());
    assert!(MeanFieldGameSolver1D::new(4, f64::INFINITY, 0.5, vec![0.0; 4]).is_err());
    assert!(MeanFieldGameSolver1D::new(4, 0.1, 0.5, vec![0.0; 4]).is_ok());
}

#[test]
fn mfg_solve_rejects_absurd_iters() {
    use oxideon::ode::MeanFieldGameSolver1D;
    let s = MeanFieldGameSolver1D::new(4, 0.1, 0.5, vec![0.0; 4]).unwrap();
    let m = vec![0.25; 4];
    let u = vec![0.0; 4];
    assert!(
        s.solve_nash_equilibrium(&m, &u, 0.01, 0.25, usize::MAX)
            .is_err()
    );
    assert!(s.solve_nash_equilibrium(&m, &u, 0.01, 0.25, 2).is_ok());
}

#[test]
fn mfg_steps_reject_truncated_potential() {
    use oxideon::ode::MeanFieldGameSolver1D;
    let mut s = MeanFieldGameSolver1D::new(4, 0.1, 0.5, vec![0.0; 4]).unwrap();
    let m = vec![0.25; 4];
    let u = vec![0.0; 4];
    assert!(s.backward_hjb_step(&u, &m, 0.01, 0.25).is_ok());
    s.potential.truncate(1);
    assert!(s.backward_hjb_step(&u, &m, 0.01, 0.25).is_err());
    assert!(s.forward_fpk_step(&m, &u, 0.01, 0.25).is_ok());
    s.potential.truncate(1);
    assert!(s.forward_fpk_step(&m, &u, 0.01, 0.25).is_ok());
}

#[test]
fn koopman_predict_rejects_absurd_steps() {
    use oxideon::ode::KoopmanSpectralEngine;
    let e = KoopmanSpectralEngine::fit_diagonal_edmd(&[vec![1.0, 2.0]], &[vec![1.0, 2.0]]).unwrap();
    assert!(e.predict_observable(&[1.0, 2.0], usize::MAX).is_err());
    assert!(e.predict_observable(&[1.0, 2.0], 3).is_ok());
}

#[test]
fn koopman_predict_rejects_truncated_matrix() {
    use oxideon::ode::KoopmanSpectralEngine;
    let mut e =
        KoopmanSpectralEngine::fit_diagonal_edmd(&[vec![1.0, 2.0]], &[vec![1.0, 2.0]]).unwrap();
    assert!(e.predict_observable(&[1.0, 2.0], 1).is_ok());
    e.koopman_matrix.truncate(1);
    assert!(e.predict_observable(&[1.0, 2.0], 1).is_err());
}

#[test]
fn chen_multiply_rejects_ragged_levels() {
    use oxideon::ode::RoughPathSignature;
    let good = RoughPathSignature::from_path(&[vec![0.0, 0.0], vec![1.0, 1.0]]).unwrap();
    assert!(good.chen_multiply(&good).is_ok());
    let mut bad = good.clone();
    bad.level1.truncate(1);
    assert!(bad.chen_multiply(&good).is_err());
    let mut bad2 = good.clone();
    bad2.level2.truncate(1);
    assert!(good.chen_multiply(&bad2).is_err());
}

#[test]
fn levy_area_rejects_short_level2() {
    use oxideon::ode::RoughPathSignature;
    let mut sig = RoughPathSignature::from_path(&[vec![0.0, 0.0], vec![1.0, 1.0]]).unwrap();
    assert!(sig.levy_area(0, 1).is_ok());
    sig.level2.truncate(1);
    assert!(sig.levy_area(0, 1).is_err());
}

// Batch T (quantum circuit/qgnn/qaoa/vqe). Pre-fix each fails at the noted
// site; post-fix each passes. t4b aborts pre-fix and t10 hangs pre-fix, so
// those two are run solo (timeout-guarded), never in the suite pre-fix.
#[test]
fn t1_tampered_qubits_gate_is_err() {
    use oxideon::quantum::QuantumCircuit;
    let mut qc = QuantumCircuit::new(2).unwrap();
    assert!(qc.h(0).is_ok());
    qc.num_qubits = 30;
    assert!(qc.h(0).is_err()); // pre-fix: index-OOB panic in state_real
}

#[test]
fn t1b_tampered_qubits_shift_is_err() {
    use oxideon::quantum::QuantumCircuit;
    let mut qc = QuantumCircuit::new(2).unwrap();
    qc.num_qubits = 64;
    assert!(qc.expectation_z(0).is_err()); // pre-fix: shl-overflow panic
}

#[test]
fn t1c_truncated_state_gate_is_err() {
    use oxideon::quantum::QuantumCircuit;
    let mut qc = QuantumCircuit::new(2).unwrap();
    qc.state_real.truncate(2);
    assert!(qc.x(0).is_err()); // pre-fix: index-OOB panic in swap
}

#[test]
fn t2_prob_truncation_is_err() {
    use oxideon::quantum::QuantumCircuit;
    let mut qc = QuantumCircuit::new(2).unwrap();
    assert_eq!(qc.probabilities().unwrap().len(), 4);
    qc.state_imag.truncate(2);
    assert!(qc.probabilities().is_err()); // pre-fix: silent 2-elem zip truncation
}

#[test]
fn t3_reset_empty_is_err() {
    use oxideon::quantum::QuantumCircuit;
    let mut qc = QuantumCircuit::new(2).unwrap();
    assert!(qc.reset().is_ok());
    qc.state_real.clear();
    qc.state_imag.clear();
    assert!(qc.reset().is_err()); // pre-fix: index panic on state_real[0]
}

#[test]
fn t4_qgnn_layers_overflow_is_err() {
    use oxideon::quantum::QuantumGraphNeuralNetwork;
    assert!(QuantumGraphNeuralNetwork::new(2, vec![], usize::MAX).is_err()); // pre-fix: mul-overflow panic
}

#[test]
fn t4b_qgnn_layers_abort_is_err() {
    use oxideon::quantum::QuantumGraphNeuralNetwork;
    // SOLO pre-fix: 20 * 2^40 f64s = 128 TiB collect -> alloc abort. Run solo with timeout.
    assert!(QuantumGraphNeuralNetwork::new(20, vec![], 1 << 40).is_err());
}

#[test]
fn t5_qgnn_setparams_overflow_is_err() {
    use oxideon::quantum::QuantumGraphNeuralNetwork;
    let mut g = QuantumGraphNeuralNetwork::new(2, vec![(0, 1)], 2).unwrap();
    g.num_layers = usize::MAX;
    assert!(g.set_parameters(vec![], vec![], vec![]).is_err()); // pre-fix: mul-overflow panic
}

#[test]
fn t6_qgnn_buildcircuit_tamper_is_err() {
    use oxideon::quantum::QuantumGraphNeuralNetwork;
    let mut g = QuantumGraphNeuralNetwork::new(2, vec![(0, 1)], 2).unwrap();
    g.num_layers = 5;
    assert!(g.forward(&[0.1, 0.2]).is_err()); // pre-fix: index-OOB panic on theta
}

#[test]
fn t6b_qgnn_edge_shift_is_err() {
    use oxideon::quantum::QuantumGraphNeuralNetwork;
    let mut g = QuantumGraphNeuralNetwork::new(2, vec![(0, 1)], 1).unwrap();
    g.num_layers = 0;
    g.edges = vec![(99, 98)];
    assert!(g.forward_graph_energy(&[0.1, 0.2]).is_err()); // pre-fix: shl-overflow panic
}

#[test]
fn t7_qaoa_edge_shift_is_err() {
    use oxideon::quantum::QaoaCircuit;
    let mut q = QaoaCircuit::new(2, 1, vec![(0, 1, 1.0)]).unwrap();
    q.p_layers = 0;
    q.edges = vec![(99, 98, 1.0)];
    assert!(q.evaluate_energy(&[], &[]).is_err()); // pre-fix: shr-overflow panic
}

#[test]
fn t8_qaoa_grad_alloc_is_err() {
    use oxideon::quantum::QaoaCircuit;
    let mut q = QaoaCircuit::new(2, 1, vec![(0, 1, 1.0)]).unwrap();
    q.p_layers = usize::MAX;
    assert!(q.compute_gradients(&[0.1], &[0.1]).is_err()); // pre-fix: capacity-overflow panic
}

#[test]
fn t9_qaoa_optimize_alloc_is_err() {
    use oxideon::quantum::QaoaCircuit;
    let mut q = QaoaCircuit::new(2, 1, vec![(0, 1, 1.0)]).unwrap();
    q.p_layers = usize::MAX;
    assert!(q.optimize(1, 0.01).is_err()); // pre-fix: capacity-overflow panic
}

#[test]
fn t10_qaoa_steps_hang_is_err() {
    use oxideon::quantum::QaoaCircuit;
    let q = QaoaCircuit::new(2, 1, vec![(0, 1, 1.0)]).unwrap();
    // SOLO pre-fix: unbounded gradient-ascent loop. Run solo with timeout.
    assert!(q.optimize_from(&[0.2], &[0.4], usize::MAX, 0.01).is_err());
}

#[test]
fn t11_vqe_layers_overflow_is_err() {
    use oxideon::quantum::VqeAnsatz;
    // Pre-fix new() accepted and num_parameters() add-overflow-panicked.
    assert!(VqeAnsatz::new(2, usize::MAX).is_err());
    let mut v = VqeAnsatz::new(2, 3).unwrap();
    assert_eq!(v.num_parameters(), 8);
    v.num_layers = usize::MAX;
    assert!(v.evaluate_energy(&[], &[]).is_err());
}

// Batch U (quantum qec/mitigation/mps). u4b aborts pre-fix: solo with timeout.
#[test]
fn u1_stabilizer_tampered_circuit_is_err() {
    use oxideon::quantum::circuit::QuantumCircuit;
    use oxideon::quantum::qec::{Pauli, Stabilizer};
    let mut qc = QuantumCircuit::new(2).unwrap();
    qc.num_qubits = 5;
    let stab = Stabilizer::new(vec![Pauli::Z; 5]);
    assert!(stab.expectation(&qc).is_err()); // pre-fix: index-OOB panic
}

#[test]
fn u2_zne_nonfinite_scale_is_err() {
    use oxideon::quantum::mitigation::{ExtrapolationModel, ZneExtrapolator};
    assert!(ZneExtrapolator::new(vec![f64::NAN], ExtrapolationModel::Richardson).is_err());
    assert!(ZneExtrapolator::new(vec![f64::INFINITY], ExtrapolationModel::Richardson).is_err());
    assert!(ZneExtrapolator::new(vec![1.0, 2.0], ExtrapolationModel::Linear).is_ok());
}

#[test]
fn u2b_zne_tampered_empty_is_err() {
    use oxideon::quantum::mitigation::{ExtrapolationModel, ZneExtrapolator};
    let mut z = ZneExtrapolator::new(vec![1.0], ExtrapolationModel::Richardson).unwrap();
    z.scale_factors.clear();
    assert!(z.extrapolate(&[]).is_err()); // pre-fix: silent Ok(0.0)
}

#[test]
fn u3_mps_site_overflow_is_err() {
    use oxideon::quantum::mps::MpsSiteTensor;
    assert!(MpsSiteTensor::new(1 << 32, 1 << 32, 2, vec![]).is_err()); // pre-fix: mul-overflow panic
}

#[test]
fn u4a_mps_random_absurd_sites_is_err() {
    use oxideon::quantum::mps::MatrixProductState;
    assert!(MatrixProductState::new_random(usize::MAX, 2, None).is_err()); // pre-fix: add-overflow panic
}

#[test]
fn u4b_mps_random_absurd_bond_is_err() {
    use oxideon::quantum::mps::MatrixProductState;
    // SOLO pre-fix: 16 TB+ site alloc -> abort. Run solo with timeout.
    assert!(MatrixProductState::new_random(100, usize::MAX, None).is_err());
}

#[test]
fn u5a_mps_overlap_truncated_data_is_err() {
    use oxideon::quantum::mps::MatrixProductState;
    let mut a = MatrixProductState::from_product_state(&[0, 0]).unwrap();
    let b = a.clone();
    a.sites[0].data.truncate(1);
    assert!(a.overlap(&b).is_err()); // pre-fix: index-OOB panic in get
}

#[test]
fn u5b_mps_overlap_short_sites_is_err() {
    use oxideon::quantum::mps::MatrixProductState;
    let mut a = MatrixProductState::from_product_state(&[0, 0]).unwrap();
    let b = a.clone();
    a.sites.pop();
    assert!(a.overlap(&b).is_err()); // pre-fix: index-OOB panic on sites[1]
}

#[test]
fn u5c_mps_overlap_bond_tamper_is_err() {
    use oxideon::quantum::mps::MatrixProductState;
    let mut a = MatrixProductState::from_product_state(&[0, 0]).unwrap();
    let b = a.clone();
    a.sites[0].chi_right = 5;
    assert!(a.overlap(&b).is_err()); // pre-fix: index-OOB panic in get
}

#[test]
fn u6_mps_overlap_absurd_bond_is_err() {
    use oxideon::quantum::mps::MatrixProductState;
    let mut a = MatrixProductState::from_product_state(&[0, 0]).unwrap();
    let b = a.clone();
    a.sites[0].chi_right = usize::MAX;
    assert!(a.overlap(&b).is_err()); // pre-fix: capacity-overflow panic sizing new_e
}

#[test]
fn u7_mps_tfim_empty_is_err() {
    use oxideon::quantum::mps::MatrixProductState;
    let mut a = MatrixProductState::from_product_state(&[0, 0]).unwrap();
    a.num_sites = 0;
    a.sites.clear();
    assert!(a.evaluate_tfim_energy(1.0, 0.5).is_err()); // pre-fix: sub-underflow panic
}

#[test]
fn u8_mps_entropy_large_n_is_err() {
    use oxideon::quantum::mps::MatrixProductState;
    let a = MatrixProductState::from_product_state(&[0; 100]).unwrap();
    assert!(a.entanglement_entropy(70).is_err()); // pre-fix: shr-overflow panic
}

// Batch V (quantum continuous_variable/tqft). v2b hangs pre-fix: solo.
#[test]
fn v1_cv_gate_tampered_modes_is_err() {
    use oxideon::quantum::continuous_variable::GaussianCvCircuit;
    let mut cv = GaussianCvCircuit::new(2).unwrap();
    cv.num_modes = 40;
    assert!(cv.displace(5, 0.1, 0.2).is_err()); // pre-fix: index-OOB panic
}

#[test]
fn v1b_cv_squeeze_absurd_modes_is_err() {
    use oxideon::quantum::continuous_variable::GaussianCvCircuit;
    let mut cv = GaussianCvCircuit::new(2).unwrap();
    cv.num_modes = usize::MAX;
    assert!(cv.squeeze(0, 0.5).is_err()); // pre-fix: mul-overflow panic
}

#[test]
fn v1c_cv_photon_truncated_disp_is_err() {
    use oxideon::quantum::continuous_variable::GaussianCvCircuit;
    let mut cv = GaussianCvCircuit::new(2).unwrap();
    cv.displacement.truncate(2);
    assert!(cv.mean_photon_number(1).is_err()); // pre-fix: index-OOB panic
}

#[test]
fn v1d_cv_purity_absurd_modes_is_err() {
    use oxideon::quantum::continuous_variable::GaussianCvCircuit;
    let mut cv = GaussianCvCircuit::new(2).unwrap();
    cv.num_modes = usize::MAX;
    assert!(cv.purity().is_err()); // pre-fix: mul-overflow panic
}

#[test]
fn v1e_cv_vacuum_tampered_modes_is_err() {
    use oxideon::quantum::continuous_variable::GaussianCvCircuit;
    let mut cv = GaussianCvCircuit::new(2).unwrap();
    cv.num_modes = 40;
    assert!(cv.vacuum_probability().is_err()); // pre-fix: index-OOB panic
}

#[test]
fn v1f_cv_total_is_err() {
    use oxideon::quantum::continuous_variable::GaussianCvCircuit;
    let mut cv = GaussianCvCircuit::new(2).unwrap();
    assert!(cv.total_mean_photon_number().is_ok());
    cv.covariance.clear();
    cv.displacement.clear();
    assert!(cv.total_mean_photon_number().is_err()); // pre-fix: index panic
}

#[test]
fn v2a_tqft_num_anyons_is_err() {
    use oxideon::quantum::tqft::Su2ChernSimonsTqft;
    let mut t = Su2ChernSimonsTqft::new(2).unwrap();
    assert_eq!(t.num_anyons().unwrap(), 3);
    t.level_k = usize::MAX;
    assert!(t.num_anyons().is_err()); // pre-fix: add-overflow panic
}

#[test]
fn v2b_tqft_total_hang_is_err() {
    use oxideon::quantum::tqft::Su2ChernSimonsTqft;
    let mut t = Su2ChernSimonsTqft::new(2).unwrap();
    assert!(t.total_quantum_dimension().is_ok());
    t.level_k = usize::MAX;
    assert!(t.total_quantum_dimension().is_err()); // pre-fix: add-overflow panic
}

#[test]
fn v2c_tqft_smatrix_is_err() {
    use oxideon::quantum::tqft::Su2ChernSimonsTqft;
    let mut t = Su2ChernSimonsTqft::new(2).unwrap();
    assert_eq!(t.modular_s_matrix().unwrap().len(), 9);
    t.level_k = 1 << 32;
    assert!(t.modular_s_matrix().is_err()); // pre-fix: mul-overflow panic
}

#[test]
fn v2d_tqft_qdim_absurd_level_is_err() {
    use oxideon::quantum::tqft::Su2ChernSimonsTqft;
    let mut t = Su2ChernSimonsTqft::new(2).unwrap();
    t.level_k = usize::MAX;
    assert!(t.quantum_dimension(0).is_err()); // pre-fix: add-overflow panic
}

#[test]
fn v2e_tqft_verlinde_absurd_level_is_err() {
    use oxideon::quantum::tqft::Su2ChernSimonsTqft;
    let mut t = Su2ChernSimonsTqft::new(2).unwrap();
    t.level_k = usize::MAX;
    assert!(t.verlinde_fusion_coefficient(0, 0, 0).is_err()); // pre-fix: add-overflow panic
}

// Batch W (quantum shadow_tomography/qldpc).
#[test]
fn w1_shadow_nan_eps_is_err() {
    use oxideon::quantum::shadow_tomography::ClassicalShadowTomography;
    assert!(ClassicalShadowTomography::sample_complexity_bound(2, 10, f64::NAN, 0.05).is_err());
    assert!(ClassicalShadowTomography::sample_complexity_bound(2, 10, 0.1, 0.05).is_ok());
}

#[test]
fn w2_ldpc_zero_dim_is_err() {
    use oxideon::quantum::quantum_ldpc::QuantumLdpcCodeEngine;
    let mut q = QuantumLdpcCodeEngine::new(2, vec![1, 0], vec![0, 1]).unwrap();
    assert!(q.verify_css_commutativity().is_ok());
    q.num_qubits = 0;
    assert!(q.verify_css_commutativity().is_err()); // pre-fix: div-by-zero panic
}

#[test]
fn w3_ldpc_truncated_matrix_is_err() {
    use oxideon::quantum::quantum_ldpc::QuantumLdpcCodeEngine;
    let mut q = QuantumLdpcCodeEngine::new(2, vec![1, 0], vec![0, 1]).unwrap();
    q.hz.truncate(1);
    assert!(q.compute_z_syndrome(&[1, 0]).is_err()); // pre-fix: silent Ok([]) on ragged hz
}

#[test]
fn w4_ldpc_syndrome_zero_dim_is_err() {
    use oxideon::quantum::quantum_ldpc::QuantumLdpcCodeEngine;
    let mut q = QuantumLdpcCodeEngine::new(2, vec![1, 0], vec![0, 1]).unwrap();
    q.num_qubits = 0;
    assert!(q.compute_z_syndrome(&[]).is_err()); // pre-fix: div-by-zero panic
}

// Batch X (robotics flow_policy, grammar sandbox). x2/x4 hang pre-fix: solo.
#[test]
fn x1_flow_scalar_obs_is_err() {
    use oxideon::robotics::flow_policy::FlowMatchingRoboticsPolicy;
    let p = FlowMatchingRoboticsPolicy::new(2, 4, 8).unwrap();
    let scalar = Tensor::from_slice(&[1.0f32], &[]).unwrap();
    assert!(p.sample_action(&scalar, 3).is_err()); // pre-fix: index panic on shape()[0]
}

#[test]
fn x2_flow_absurd_steps_is_err() {
    use oxideon::robotics::flow_policy::FlowMatchingRoboticsPolicy;
    let p = FlowMatchingRoboticsPolicy::new(2, 4, 8).unwrap();
    let obs = Tensor::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[1, 4]).unwrap();
    // SOLO pre-fix: unbounded Euler loop. Run solo with timeout.
    assert!(p.sample_action(&obs, usize::MAX).is_err());
}

#[test]
fn x3_sandbox_int_overflow_is_runtime_error() {
    use oxideon::grammar::sandbox::{ExecutionSandbox, SandboxStatus};
    use std::collections::HashMap;
    let sb = ExecutionSandbox::new(1000);
    let vars = HashMap::new();
    // Pre-fix each of these panicked the host (overflow); post-fix the
    // sandbox reports RuntimeError and stays up.
    for prog in [
        "x = 9223372036854775807 + 1",
        "x = -9223372036854775808 - 1",
        "x = 3037000500 * 3037000500",
    ] {
        assert!(
            matches!(
                sb.execute(prog, &vars).status,
                SandboxStatus::RuntimeError(_)
            ),
            "overflowing program must report, not panic: {prog}"
        );
    }
    let mut min_vars = HashMap::new();
    min_vars.insert("x".to_string(), i64::MIN);
    assert!(matches!(
        sb.execute("y = x / -1", &min_vars).status,
        SandboxStatus::RuntimeError(_)
    ));
    assert!(matches!(
        sb.execute("y = x % -1", &min_vars).status,
        SandboxStatus::RuntimeError(_)
    ));
    // Healthy programs still succeed.
    let ok = sb.execute("y = x + 1", &HashMap::from([("x".to_string(), 41)]));
    assert_eq!(ok.status, SandboxStatus::Success);
    assert_eq!(ok.variables["y"], 42);
}

#[test]
fn x4_synth_absurd_iters_is_err() {
    use oxideon::grammar::sandbox::{SynthesisEngine, TestCase};
    let mut e = SynthesisEngine::new(3, 100);
    let tests = [TestCase::new("t", vec![], "y", 1)];
    assert!(
        e.synthesize_and_repair(|_, _| "y = 1".to_string(), &tests)
            .is_ok()
    );
    e.max_repair_iterations = usize::MAX;
    assert!(
        e.synthesize_and_repair(|_, _| "y = 0".to_string(), &tests)
            .is_err()
    );
}

// Batch Y (geom_graphs wasserstein/tgn/spiking/gat/sheaf). y7 hangs pre-fix: solo.
#[test]
fn y1_wasserstein_zero_nodes_is_err() {
    use oxideon::geom_graphs::wasserstein_flow::GraphWassersteinGeodesicEngine;
    let mut e = GraphWassersteinGeodesicEngine::new(2, 0.5).unwrap();
    e.num_nodes = 0;
    assert!(e.displacement_interpolate(&[], &[], 0.5).is_err()); // pre-fix: sub-underflow panic
}

#[test]
fn y2_tgn_bad_node_is_err() {
    use oxideon::geom_graphs::temporal_gnn::TemporalInteractionGNN;
    let mut g = TemporalInteractionGNN::new(3, 4, 4, 2).unwrap();
    let edge = Tensor::from_slice(&[0.0f32, 0.0], &[1, 2]).unwrap();
    assert!(g.process_event(99, 0, 1.0, &edge).is_err()); // pre-fix: assert panic
}

#[test]
fn y3_tgn_popped_memory_is_err() {
    use oxideon::geom_graphs::temporal_gnn::TemporalInteractionGNN;
    let mut g = TemporalInteractionGNN::new(3, 4, 4, 2).unwrap();
    let edge = Tensor::from_slice(&[0.0f32, 0.0], &[1, 2]).unwrap();
    g.node_memories.pop();
    assert!(g.process_event(2, 2, 1.0, &edge).is_err()); // pre-fix: index-OOB panic
}

#[test]
fn y4_spiking_scalar_v_is_err() {
    use oxideon::geom_graphs::spiking_gnn::SpikingGnnLayer;
    let l = SpikingGnnLayer::new(2, 3, 0.5, 0.9).unwrap();
    let v = Tensor::from_slice(&[1.0f32], &[]).unwrap();
    let s = Tensor::from_slice(&[1.0f32, 0.0], &[1, 2]).unwrap();
    assert!(l.step(&v, &s, &[], None).is_err()); // pre-fix: index panic on shape()[0]
}

#[test]
fn y4b_spiking_temporal_scalar_is_err() {
    use oxideon::geom_graphs::spiking_gnn::SpikingGnnLayer;
    let l = SpikingGnnLayer::new(2, 3, 0.5, 0.9).unwrap();
    let s = Tensor::from_slice(&[1.0f32], &[]).unwrap();
    // Same shape()[0] panic class as y4 (class-red-proven there).
    assert!(l.forward_temporal(&s, &[], 2).is_err());
}

#[test]
fn y5_spiking_row_mismatch_is_err() {
    use oxideon::geom_graphs::spiking_gnn::SpikingGnnLayer;
    let l = SpikingGnnLayer::new(2, 3, 0.5, 0.9).unwrap();
    let v = Tensor::zeros(&[4, 3], DType::F32, &Device::Cpu).unwrap();
    let s = Tensor::zeros(&[2, 2], DType::F32, &Device::Cpu).unwrap();
    assert!(l.step(&v, &s, &[], None).is_err()); // pre-fix: index-OOB panic
}

#[test]
fn y6_spiking_out_tamper_is_err() {
    use oxideon::geom_graphs::spiking_gnn::SpikingGnnLayer;
    let mut l = SpikingGnnLayer::new(2, 3, 0.5, 0.9).unwrap();
    let v = Tensor::zeros(&[2, 3], DType::F32, &Device::Cpu).unwrap();
    let s = Tensor::zeros(&[2, 2], DType::F32, &Device::Cpu).unwrap();
    l.out_dim = usize::MAX;
    assert!(l.step(&v, &s, &[], None).is_err()); // pre-fix: mul-overflow panic
}

#[test]
fn y7_spiking_absurd_steps_is_err() {
    use oxideon::geom_graphs::spiking_gnn::SpikingGnnLayer;
    let l = SpikingGnnLayer::new(2, 3, 0.5, 0.9).unwrap();
    let s = Tensor::zeros(&[2, 2], DType::F32, &Device::Cpu).unwrap();
    // SOLO pre-fix: unbounded temporal loop. Run solo with timeout.
    assert!(l.forward_temporal(&s, &[], usize::MAX).is_err());
}

#[test]
fn y10_gat_in_tamper_is_err() {
    use oxideon::geom_graphs::hyperbolic_gat::HyperbolicGATLayer;
    let mut l = HyperbolicGATLayer::new(4, 4, 1.0, 0.5).unwrap();
    l.in_dim = usize::MAX;
    let x = Tensor::zeros(&[3, 5], DType::F32, &Device::Cpu).unwrap();
    assert!(l.forward(&x, &[]).is_err()); // pre-fix: add-overflow panic in the check
}

#[test]
fn y11a_gat_bad_edge_is_err() {
    use oxideon::geom_graphs::hyperbolic_gat::HyperbolicGATLayer;
    let l = HyperbolicGATLayer::new(4, 4, 1.0, 0.5).unwrap();
    let x = Tensor::zeros(&[3, 5], DType::F32, &Device::Cpu).unwrap();
    assert!(l.forward(&x, &[(0, 99)]).is_err()); // pre-fix: silently dropped edge
}

#[test]
fn y11b_spiking_bad_edge_is_err() {
    use oxideon::geom_graphs::spiking_gnn::SpikingGnnLayer;
    let l = SpikingGnnLayer::new(2, 3, 0.5, 0.9).unwrap();
    let v = Tensor::zeros(&[2, 3], DType::F32, &Device::Cpu).unwrap();
    let s = Tensor::zeros(&[2, 2], DType::F32, &Device::Cpu).unwrap();
    assert!(l.step(&v, &s, &[(0, 99)], None).is_err()); // pre-fix: silently dropped edge
}

#[test]
fn y12_gat_out_tamper_is_err() {
    use oxideon::geom_graphs::hyperbolic_gat::HyperbolicGATLayer;
    let mut l = HyperbolicGATLayer::new(4, 4, 1.0, 0.5).unwrap();
    l.out_dim = 8;
    let x = Tensor::zeros(&[3, 5], DType::F32, &Device::Cpu).unwrap();
    assert!(l.forward(&x, &[]).is_err()); // pre-fix: index-OOB panic
}

#[test]
fn y13_sheaf_dim_tamper_is_err() {
    use oxideon::geom_graphs::sheaf::CellularSheaf;
    let mut s = CellularSheaf::new(3, 2, 2, vec![(0, 1), (1, 2)]).unwrap();
    s.node_dim = usize::MAX;
    assert!(s.compute_sheaf_laplacian().is_err()); // pre-fix: mul-overflow panic
}

#[test]
fn y14_sheaf_popped_map_is_err() {
    use oxideon::geom_graphs::sheaf::CellularSheaf;
    let mut s = CellularSheaf::new(3, 2, 2, vec![(0, 1), (1, 2)]).unwrap();
    s.restriction_maps.pop();
    assert!(s.compute_sheaf_laplacian().is_err()); // pre-fix: index-OOB panic
}

#[test]
fn y14b_sheaf_edge_tamper_is_err() {
    use oxideon::geom_graphs::sheaf::CellularSheaf;
    let mut s = CellularSheaf::new(3, 2, 2, vec![(0, 1), (1, 2)]).unwrap();
    s.edges[0] = (99, 0);
    assert!(s.compute_sheaf_laplacian().is_err()); // pre-fix: index-OOB panic
}

#[test]
fn y15_sheaf_map_shape_is_err() {
    use oxideon::geom_graphs::sheaf::CellularSheaf;
    let mut s = CellularSheaf::new(3, 2, 2, vec![(0, 1), (1, 2)]).unwrap();
    let wrong = Tensor::zeros(&[1, 1], DType::F64, &Device::Cpu).unwrap();
    s.restriction_maps[0] = (wrong.clone(), wrong);
    assert!(s.compute_sheaf_laplacian().is_err()); // pre-fix: index-OOB panic
}

// Batch Z (geom_graphs riemannian/clifford/yangmills/fiber/wavelet).
#[test]
fn z2_riemann_short_metric_is_err() {
    use oxideon::geom_graphs::riemannian::RiemannianManifold;
    let m = RiemannianManifold::new(2);
    let x = Tensor::from_slice(&[1.0, 2.0], &[2]).unwrap();
    let metric = |t: &Tensor| {
        let v = t.to_vec::<f64>().unwrap();
        if v == vec![1.0, 2.0] {
            Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2])
        } else {
            Tensor::from_slice(&[1.0], &[1, 1])
        }
    };
    assert!(m.christoffel_symbols(&x, &metric, 1e-4).is_err()); // pre-fix: index-OOB panic
}

#[test]
fn z3_riemann_short_velocity_is_err() {
    use oxideon::geom_graphs::riemannian::RiemannianManifold;
    let m = RiemannianManifold::new(2);
    let v = Tensor::from_slice(&[1.0], &[1]).unwrap();
    let g = Tensor::zeros(&[2, 2, 2], DType::F64, &Device::Cpu).unwrap();
    assert!(m.geodesic_acceleration(&v, &g).is_err()); // pre-fix: index-OOB panic
}

#[test]
fn z4_riemann_short_transport_is_err() {
    use oxideon::geom_graphs::riemannian::RiemannianManifold;
    let m = RiemannianManifold::new(2);
    let w = Tensor::from_slice(&[1.0], &[1]).unwrap();
    let v = Tensor::from_slice(&[1.0, 0.0], &[2]).unwrap();
    let g = Tensor::zeros(&[2, 2, 2], DType::F64, &Device::Cpu).unwrap();
    assert!(m.parallel_transport_step(&w, &v, &g, 0.01).is_err()); // pre-fix: index-OOB panic
}

#[test]
fn z5_riemann_absurd_steps_is_err() {
    use oxideon::geom_graphs::riemannian::RiemannianManifold;
    let m = RiemannianManifold::new(2);
    let x0 = Tensor::from_slice(&[0.1, 0.2], &[2]).unwrap();
    let v0 = Tensor::from_slice(&[0.0, 0.1], &[2]).unwrap();
    let metric = |_: &Tensor| Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    assert!(
        m.integrate_geodesic(&x0, &v0, &metric, 0.01, usize::MAX)
            .is_err()
    );
}

#[test]
fn z6_clifford_out_tamper_is_err() {
    use oxideon::geom_graphs::clifford::CliffordLinear;
    let mut l = CliffordLinear::new(2, 3, true).unwrap();
    l.out_channels = usize::MAX;
    let x = Tensor::zeros(&[1, 2, 8], DType::F32, &Device::Cpu).unwrap();
    assert!(l.forward(&x).is_err()); // pre-fix: mul-overflow panic
}

#[test]
fn z7_clifford_short_weights_is_err() {
    use oxideon::geom_graphs::clifford::CliffordLinear;
    let mut l = CliffordLinear::new(2, 3, true).unwrap();
    l.weights = Tensor::zeros(&[1, 1, 1], DType::F32, &Device::Cpu).unwrap();
    let x = Tensor::zeros(&[1, 2, 8], DType::F32, &Device::Cpu).unwrap();
    assert!(l.forward(&x).is_err()); // pre-fix: copy_from_slice panic
}

#[test]
fn z8_yangmills_popped_link_is_err() {
    use oxideon::geom_graphs::yang_mills::YangMillsLatticeGauge;
    let mut g = YangMillsLatticeGauge::new(2, vec![(0, 1)]).unwrap();
    g.links.pop();
    let spinors = [[(1.0, 0.0); 2]; 2];
    assert!(g.gauge_conv_step(&spinors).is_err()); // pre-fix: index-OOB panic
}

#[test]
fn z8b_yangmills_edge_tamper_is_err() {
    use oxideon::geom_graphs::yang_mills::{Su2Element, YangMillsLatticeGauge};
    let mut g = YangMillsLatticeGauge::new(2, vec![(0, 1)]).unwrap();
    g.edges[0] = (99, 0);
    let omegas = [Su2Element::identity(), Su2Element::identity()];
    assert!(g.apply_gauge_transformation(&omegas).is_err()); // pre-fix: index-OOB panic
}

#[test]
fn z9_fiber_transport_bounds_is_err() {
    use oxideon::geom_graphs::fiber_bundle::PrincipalBundleConnection;
    let c = PrincipalBundleConnection::new(2, 2).unwrap();
    assert!(c.transport_vector(99, 0, &[1.0, 2.0]).is_err()); // pre-fix: index-OOB panic
}

#[test]
fn z10_fiber_cycle_bounds_is_err() {
    use oxideon::geom_graphs::fiber_bundle::PrincipalBundleConnection;
    let mut c = PrincipalBundleConnection::new(2, 2).unwrap();
    c.set_connection(0, 1, 0.5).unwrap();
    assert!(c.cycle_holonomy(&[0, 1, 99]).is_err()); // pre-fix: index-OOB panic
}

#[test]
fn z12_gauge_conn_tamper_is_err() {
    use oxideon::geom_graphs::fiber_bundle::{
        GaugeEquivariantConvLayer, PrincipalBundleConnection,
    };
    let c = PrincipalBundleConnection::new(2, 2).unwrap();
    let mut l = GaugeEquivariantConvLayer::new(c);
    l.connection.num_nodes = 99;
    let x = Tensor::from_slice(&vec![0.0; 198], &[99, 2]).unwrap();
    assert!(l.forward(&x).is_err()); // pre-fix: index-OOB panic
}

#[test]
fn z13_gauge_weights_tamper_is_err() {
    use oxideon::geom_graphs::fiber_bundle::{
        GaugeEquivariantConvLayer, PrincipalBundleConnection,
    };
    let c = PrincipalBundleConnection::new(2, 2).unwrap();
    let mut l = GaugeEquivariantConvLayer::new(c);
    l.weight_self.clear();
    let x = Tensor::zeros(&[2, 2], DType::F64, &Device::Cpu).unwrap();
    assert!(l.forward(&x).is_err()); // pre-fix: index-OOB panic
}

#[test]
fn z14_wavelet_matvec_short_is_err() {
    use oxideon::geom_graphs::wavelet::{GraphLaplacian, LaplacianType};
    let mut lap =
        GraphLaplacian::new(&[0.0, -1.0, -1.0, 0.0], 2, LaplacianType::Combinatorial).unwrap();
    assert!(lap.matvec(&[1.0, 0.0]).is_ok());
    assert!(lap.matvec(&[1.0]).is_err());
    lap.laplacian.truncate(1);
    assert!(lap.matvec(&[1.0, 0.0]).is_err()); // pre-fix: index-OOB panic
}

#[test]
fn z15_chebyshev_absurd_order_is_err() {
    use oxideon::geom_graphs::wavelet::ChebyshevWaveletFilter;
    assert!(ChebyshevWaveletFilter::new(1.0, 4, 2.0).is_ok());
    assert!(ChebyshevWaveletFilter::new(1.0, usize::MAX, 2.0).is_err());
}

#[test]
fn z16_wavelet_coeffs_tamper_is_err() {
    use oxideon::geom_graphs::wavelet::{ChebyshevWaveletFilter, GraphLaplacian, LaplacianType};
    let lap =
        GraphLaplacian::new(&[0.0, -1.0, -1.0, 0.0], 2, LaplacianType::Combinatorial).unwrap();
    let mut f = ChebyshevWaveletFilter::new(1.0, 4, lap.lambda_max).unwrap();
    f.coeffs.truncate(1);
    assert!(f.filter(&lap, &[1.0, 0.0]).is_err()); // pre-fix: index-OOB panic
}

#[test]
fn z17_multiscale_scales_tamper_is_err() {
    use oxideon::geom_graphs::wavelet::{LaplacianType, MultiscaleGraphWaveletTransform};
    let mut t = MultiscaleGraphWaveletTransform::new(
        &[0.0, -1.0, -1.0, 0.0],
        2,
        &[0.5, 1.0],
        3,
        LaplacianType::Combinatorial,
    )
    .unwrap();
    t.scales.pop();
    let x = Tensor::zeros(&[2, 2], DType::F64, &Device::Cpu).unwrap();
    assert!(t.forward(&x).is_err()); // pre-fix: index-OOB panic
}

#[test]
fn z19_spatiotemporal_absurd_hidden_is_err() {
    use oxideon::geom_graphs::wavelet::SpatioTemporalWaveletLayer;
    // SOLO pre-fix: 2^44-element recurrent collect -> alloc abort (hidden = MAX
    // itself is already caught loudly by the spatial check). Run solo with timeout.
    assert!(
        SpatioTemporalWaveletLayer::new(&[0.0, -1.0, -1.0, 0.0], 2, &[0.5], 2, 2, 1 << 22).is_err()
    );
}

#[test]
fn z20_spatiotemporal_cols_mismatch_is_err() {
    use oxideon::geom_graphs::wavelet::SpatioTemporalWaveletLayer;
    let l = SpatioTemporalWaveletLayer::new(&[0.0, -1.0, -1.0, 0.0], 2, &[0.5], 2, 2, 4).unwrap();
    let x = Tensor::zeros(&[2, 5], DType::F64, &Device::Cpu).unwrap();
    assert!(l.step(&x, None).is_err()); // pre-fix: silent wrong values
}

#[test]
fn z21_spatiotemporal_weights_tamper_is_err() {
    use oxideon::geom_graphs::wavelet::SpatioTemporalWaveletLayer;
    let mut l =
        SpatioTemporalWaveletLayer::new(&[0.0, -1.0, -1.0, 0.0], 2, &[0.5], 2, 2, 4).unwrap();
    l.weight_spatial.truncate(1);
    let x = Tensor::zeros(&[2, 2], DType::F64, &Device::Cpu).unwrap();
    assert!(l.step(&x, None).is_err()); // pre-fix: index-OOB panic
}

#[test]
fn z22_spatiotemporal_hprev_shape_is_err() {
    use oxideon::geom_graphs::wavelet::SpatioTemporalWaveletLayer;
    let l = SpatioTemporalWaveletLayer::new(&[0.0, -1.0, -1.0, 0.0], 2, &[0.5], 2, 2, 4).unwrap();
    let x = Tensor::zeros(&[2, 2], DType::F64, &Device::Cpu).unwrap();
    let h = Tensor::zeros(&[2, 1], DType::F64, &Device::Cpu).unwrap();
    assert!(l.step(&x, Some(&h)).is_err()); // pre-fix: index-OOB panic
}

// ===== Batch AA: fused kernels + autograd upstream-grad validation =====

#[test]
fn aa1_backward_with_grad_wrong_shape_is_err_not_panic() {
    use oxideon::fused_kernels::fused_swiglu;
    let g = Tensor::from_slice(&[1f32, 2., 3., 4.], &[2, 2])
        .unwrap()
        .requires_grad_(true);
    let u = Tensor::from_slice(&[1f32, 0., 0., 1.], &[2, 2])
        .unwrap()
        .requires_grad_(true);
    let y = fused_swiglu(&g, &u).unwrap();
    let wrong = Tensor::from_slice(&[1f32, 1., 1.], &[3]).unwrap();
    assert!(y.backward_with_grad(&wrong, false).is_err()); // pre-fix: index-OOB panic in closure
}

#[test]
fn aa1b_hook_returning_wrong_shape_is_err_not_panic() {
    use oxideon::fused_kernels::fused_swiglu;
    let g = Tensor::from_slice(&[1f32, 2., 3., 4.], &[2, 2])
        .unwrap()
        .requires_grad_(true);
    let u = Tensor::from_slice(&[1f32, 0., 0., 1.], &[2, 2])
        .unwrap()
        .requires_grad_(true);
    let y = fused_swiglu(&g, &u).unwrap();
    y.register_hook(|_g| {
        Ok(Some(
            Tensor::from_slice(&[1f32], &[1]).expect("hook tensor"),
        ))
    });
    assert!(y.sum().unwrap().backward().is_err()); // pre-fix: index-OOB panic in closure
}

#[test]
fn ab1_differentiate_through_unfused_silu_is_nonzero() {
    use oxideon::{IrGraph, IrOp};
    let mut g = IrGraph::new();
    let x = g.add_input();
    let w = g.add_input();
    let b = g.add_input();
    let lin = g.push_op(IrOp::FusedLinearSilu { x, w, b });
    let out = g.push_op(IrOp::SumAll(lin));
    g.set_outputs(vec![out]);
    let grads = g.differentiate(out, &[x]).unwrap();
    g.set_outputs(vec![grads[0]]);
    let xt = Tensor::from_slice(&[1f32, 2., 3., 4.], &[2, 2]).unwrap();
    let wt = Tensor::from_slice(&[1f32, 0., 0., 1.], &[2, 2]).unwrap();
    let bt = Tensor::from_slice(&[0f32, 0.], &[2]).unwrap();
    let got = g.execute(&[xt, wt, bt]).unwrap();
    let v: Vec<f32> = got[0].to_vec().unwrap();
    assert!(v.iter().any(|e| *e != 0.0)); // pre-fix: silent ZeroLike grad
}

#[test]
fn ab2_differentiate_then_optimize_executes_nonzero() {
    use oxideon::{IrGraph, IrOp};
    let mut g = IrGraph::new();
    let x = g.add_input();
    let w = g.add_input();
    let b = g.add_input();
    let lin = g.push_op(IrOp::FusedLinearSilu { x, w, b });
    let out = g.push_op(IrOp::SumAll(lin));
    g.set_outputs(vec![out]);
    let grads = g.differentiate(out, &[x]).unwrap();
    g.set_outputs(vec![grads[0]]);
    g.optimize().unwrap();
    // pre-fix: silent all-zero grad flowed through optimize+execute
    let xt = Tensor::from_slice(&[1f32, 2., 3., 4.], &[2, 2]).unwrap();
    let wt = Tensor::from_slice(&[1f32, 0., 0., 1.], &[2, 2]).unwrap();
    let bt = Tensor::from_slice(&[0f32, 0.], &[2]).unwrap();
    let got = g.execute(&[xt, wt, bt]).unwrap();
    let v: Vec<f32> = got[0].to_vec().unwrap();
    assert!(v.iter().any(|e| *e != 0.0));
}

#[test]
fn ab3_optimize_nontopological_handbuilt_is_err_not_panic() {
    use oxideon::{IrGraph, IrNode, IrOp};
    // Node 1 references node 2 (defined below it): valid ids, bad order.
    let mut g = IrGraph {
        nodes: vec![
            IrNode {
                id: 0,
                op: IrOp::Input { index: 0 },
            },
            IrNode {
                id: 1,
                op: IrOp::SumAll(2),
            },
            IrNode {
                id: 2,
                op: IrOp::Neg(0),
            },
        ],
        outputs: vec![1],
        num_inputs: 1,
    };
    assert!(g.optimize().is_err()); // pre-fix: DCE indexing panic
}

#[test]
fn ab4_symdim_resolve_overflow_is_err_not_panic() {
    use oxideon::SymDim;
    use std::collections::HashMap;
    let env: HashMap<&str, usize> = HashMap::new();
    let mul = SymDim::Mul(Box::new(SymDim::val(usize::MAX)), Box::new(SymDim::val(2)));
    assert!(mul.resolve(&env).is_err()); // pre-fix: overflow panic (debug) / wrap (release)
    let add = SymDim::Add(Box::new(SymDim::val(usize::MAX)), Box::new(SymDim::val(1)));
    assert!(add.resolve(&env).is_err());
}

#[test]
fn ab5_plan_memory_oob_operand_no_panic() {
    use oxideon::{IrGraph, IrNode, IrOp};
    let g = IrGraph {
        nodes: vec![
            IrNode {
                id: 0,
                op: IrOp::Input { index: 0 },
            },
            IrNode {
                id: 1,
                op: IrOp::Neg(99),
            },
        ],
        outputs: vec![1],
        num_inputs: 1,
    };
    let plan = g.plan_memory(); // pre-fix: index-OOB panic
    assert_eq!(plan.slot_of.len(), 2);
}

#[test]
fn ab6_trace_deep_chain_no_stack_overflow() {
    use oxideon::trace_to_ir;
    let x0 = Tensor::from_slice(&[1f32], &[1]).unwrap();
    let one = Tensor::from_slice(&[1f32], &[1]).unwrap();
    let g = trace_to_ir(
        |ts| {
            let mut acc = ts[0].clone();
            for _ in 0..20_000 {
                acc = acc.add(&ts[1]).unwrap();
            }
            Ok(acc)
        },
        &[x0, one],
    )
    .unwrap(); // pre-fix: stack overflow abort
    assert!(g.nodes.len() > 20_000);
}

#[test]
fn ab6b_deep_chain_build_and_drop_no_stack_overflow() {
    let x0 = Tensor::from_slice(&[1f32], &[1]).unwrap();
    let one = Tensor::from_slice(&[1f32], &[1]).unwrap();
    let mut acc = x0.requires_grad_(true);
    for _ in 0..20_000 {
        acc = acc.add(&one).unwrap();
    }
    drop(acc); // pre-fix: stack overflow abort in recursive drop
    drop(one);
}

#[test]
fn ac1_interleaver_zero_config_is_err_not_panic() {
    assert!(PrefillDecodeInterleaver::new(0, 4).is_err()); // pre-fix: assert panic
    assert!(PrefillDecodeInterleaver::new(8, 0).is_err());
    assert!(PrefillDecodeInterleaver::new(8, 4).is_ok());
}

#[test]
fn ac2a_index_huge_dim_ok_not_panic() {
    let x = Tensor::arange(0.0, 5.0, 1.0, DType::F32).unwrap();
    assert!(x.i(isize::MAX).is_err());
    assert!(x.i(isize::MIN).is_err());
    assert_eq!(x.i(-1).unwrap().numel(), 1);
    // dim = 2^63 (stride-0 view): `dim as isize + idx` overflows pre-fix;
    // post-fix the downstream view application rejects it loudly.
    let base = Tensor::zeros(&[1], DType::F32, &Device::Cpu).unwrap();
    let v = base.as_strided(&[1usize << 63], &[0], 0).unwrap();
    let sel = v.i(-1).unwrap(); // pre-fix: overflow panic
    assert_eq!(sel.shape(), &[] as &[usize]);
    // every isize is a valid index into a 2^63 dim (MIN == -dim -> 0)
    assert_eq!(v.i(0).unwrap().numel(), 1);
    assert_eq!(v.i(isize::MIN).unwrap().numel(), 1);
}

#[test]
fn ac2b_slice_huge_dim_is_err_not_panic_or_empty() {
    let base = Tensor::zeros(&[1], DType::F32, &Device::Cpu).unwrap();
    let v = base.as_strided(&[1usize << 63], &[0], 0).unwrap();
    assert!(v.i(0..5).is_err()); // pre-fix: clamp(min > max) panic incl. release
    assert!(v.i(..).is_err()); // pre-fix: silently wrong empty view
}

#[test]
fn ac3_ring_zero_world_is_err_not_panic() {
    use oxideon::long_context::RingAttention;
    assert!(RingAttention::new(0, 0, false).is_err()); // pre-fix: assert panic
    assert!(RingAttention::new(2, 2, false).is_err());
    assert!(RingAttention::new(2, 0, false).is_ok());
}

#[test]
fn ae1_sharded_matmul_1d_global_no_underflow_panic() {
    use oxideon::device::DeviceMesh;
    use oxideon::distributed::{Placement, ShardedTensor};
    let mesh = DeviceMesh::new(&[1], &["x"], vec![Device::Cpu]).unwrap();
    let a = Tensor::from_slice(&[1f32, 0., 0., 1.], &[2, 2]).unwrap();
    let mk = |t: Tensor| ShardedTensor {
        global_shape: vec![2],
        placements: vec![Placement::Shard(0)],
        mesh: mesh.clone(),
        local_tensor: t,
        device: Device::Cpu,
    };
    let r = mk(a.clone()).sharded_matmul(&mk(a), &[0]); // pre-fix: len()-2 underflow panic
    assert!(r.is_ok());
}

#[test]
fn ae2_ring_allreduce_short_dim_exact_not_panic() {
    use oxideon::distributed::{Parallel3DConfig, Parallel3DRuntime};
    let cfg = Parallel3DConfig::new(2, 1, 1).unwrap();
    let rt = Parallel3DRuntime::new(cfg, vec![Device::Cpu, Device::Cpu]).unwrap();
    // dim0 (1) < group (2): chunk yields 1 piece, ring indexes [1] pre-fix.
    let t1 = Tensor::from_slice(&[1f32, 2., 3., 4.], &[1, 4]).unwrap();
    let t2 = Tensor::from_slice(&[10f32, 20., 30., 40.], &[1, 4]).unwrap();
    let out = rt.ring_all_reduce(&[0, 1], &[t1, t2]).unwrap(); // pre-fix: index panic
    for o in &out {
        assert_eq!(o.to_vec::<f32>().unwrap(), vec![11., 22., 33., 44.]);
    }
    // Fully empty dim0 also reduces exactly (to empty).
    let e1 = Tensor::zeros(&[0, 4], DType::F32, &Device::Cpu).unwrap();
    let e2 = Tensor::zeros(&[0, 4], DType::F32, &Device::Cpu).unwrap();
    let out_e = rt.ring_all_reduce(&[0, 1], &[e1, e2]).unwrap();
    for o in &out_e {
        assert_eq!(o.shape(), &[0, 4]);
    }
}

#[test]
fn ae3_tree_mask_huge_is_err_not_10gb_alloc() {
    use oxideon::serving::TreeAttentionMask;
    // 50k nodes -> 2.5e9 mask elements > MAX_ELEMENTS (2^31).
    let parents = vec![0usize; 50_000];
    assert!(TreeAttentionMask::build(&parents).is_err()); // pre-fix: uncapped 10 GB alloc (abort-class)
}

#[test]
fn ae4_ema_new_bool_is_err_not_panic() {
    use oxideon::amp::EMA;
    let b = Tensor::from_slice(&[true, false], &[2]).unwrap();
    assert!(EMA::new(&[b], 0.9).is_err()); // pre-fix: `.expect("EMA clone")` panic
    let f = Tensor::from_slice(&[1f32, 2.], &[2]).unwrap();
    assert!(EMA::new(&[f], 0.9).is_ok());
}

#[test]
fn ae5_gradscaler_scale_stays_finite() {
    use oxideon::amp::GradScaler;
    use oxideon::optim::Sgd;
    let p = Tensor::from_slice(&[1f32], &[1])
        .unwrap()
        .requires_grad_(true);
    p.mul(&p).unwrap().sum().unwrap().backward().unwrap();
    let mut opt = Sgd::new(vec![p.clone()], 0.01);
    let mut scaler = GradScaler::new(1e308)
        .unwrap()
        .with_intervals(2.0, 0.5, 1)
        .unwrap();
    assert!(scaler.step_and_update(&mut opt, &[p]).unwrap());
    assert!(scaler.scale_factor().is_finite()); // pre-fix: 1e308 * 2 = inf
}

#[test]
fn ae7a_train_test_split_wrap_n_is_err_not_panic() {
    use oxideon::data::train_test_split;
    let base = Tensor::zeros(&[1], DType::F32, &Device::Cpu).unwrap();
    // dim past i64::MAX: `0..n as i64` wraps to an empty range pre-fix,
    // then slicing panics.
    let n = (i64::MAX as usize) + 5;
    let x = base.as_strided(&[n], &[0], 0).unwrap();
    let y = base.as_strided(&[n], &[0], 0).unwrap();
    assert!(train_test_split(&x, &y, 0.25, false).is_err()); // pre-fix: slice panic
}

#[test]
fn ae7b_train_test_split_giant_n_is_err_not_abort() {
    use oxideon::data::train_test_split;
    let base = Tensor::zeros(&[1], DType::F32, &Device::Cpu).unwrap();
    // 2^40 index rows: 8 TB index buffer pre-fix (abort-class; demonstrated
    // under `ulimit -v` containment, never run unbounded pre-fix).
    let n = 1usize << 40;
    let x = base.as_strided(&[n], &[0], 0).unwrap();
    let y = base.as_strided(&[n], &[0], 0).unwrap();
    assert!(train_test_split(&x, &y, 0.25, false).is_err());
}

#[test]
fn ae7c_kfold_wrap_n_is_err_not_panic() {
    use oxideon::data::k_fold_split;
    let base = Tensor::zeros(&[1], DType::F32, &Device::Cpu).unwrap();
    let n = (i64::MAX as usize) + 5;
    let x = base.as_strided(&[n], &[0], 0).unwrap();
    let y = base.as_strided(&[n], &[0], 0).unwrap();
    assert!(k_fold_split(&x, &y, 2, false).is_err()); // pre-fix: slice panic
}

#[test]
fn ae7d_kfold_giant_n_is_err_not_abort() {
    use oxideon::data::k_fold_split;
    let base = Tensor::zeros(&[1], DType::F32, &Device::Cpu).unwrap();
    let n = 1usize << 40;
    let x = base.as_strided(&[n], &[0], 0).unwrap();
    let y = base.as_strided(&[n], &[0], 0).unwrap();
    assert!(k_fold_split(&x, &y, 2, false).is_err()); // pre-fix: 8 TB index buffer (abort-class)
}

#[test]
fn ae8_loss_huge_target_is_err_not_silent_wrap() {
    use oxideon::nn::loss::{focal_loss, nll_loss};
    let lp = Tensor::from_slice(&[0.0f32, -1.0], &[1, 2]).unwrap();
    // usize::MAX `as i64` wraps to -1, which gather reads as last-class.
    assert!(nll_loss(&lp, &[usize::MAX]).is_err()); // pre-fix: silently wrong Ok
    assert!(focal_loss(&lp, &[usize::MAX], 0.25, 2.0).is_err());
    assert!(nll_loss(&lp, &[1]).is_ok());
}

#[test]
fn ae9a_speculative_huge_prompt_is_err_not_silent_wrap() {
    use oxideon::nn::transformer::CausalLM;
    let target = CausalLM::new(8, 8, 2, 1, 16, 16).unwrap();
    let draft = CausalLM::new(8, 8, 2, 1, 16, 16).unwrap();
    // usize::MAX `as i64` wraps to -1, which embedding reads as last-row.
    assert!(
        target
            .speculative_generate(&draft, &[usize::MAX], 2, 2)
            .is_err()
    ); // pre-fix: silently wrong Ok
    assert!(target.speculative_generate(&draft, &[1, 2], 2, 2).is_ok());
}

#[test]
fn ae9b_beam_huge_prompt_is_err_not_silent_wrap() {
    use oxideon::nn::transformer::CausalLM;
    let m = CausalLM::new(8, 8, 2, 1, 16, 16).unwrap();
    assert!(m.beam_search(&[usize::MAX], 2, 2, 1.0).is_err()); // pre-fix: silently wrong Ok
    assert!(m.beam_search(&[1, 2], 2, 2, 1.0).is_ok());
}

#[test]
fn ae9c_multimodal_huge_prompt_is_err_not_silent_wrap() {
    use oxideon::nn::frontier::MultimodalVLM;
    let vlm = MultimodalVLM::new(8, 4, 1, 8, 1, 2, 16, 8, 1, 2, 2, 16, 32).unwrap();
    let img = Tensor::zeros(&[1, 1, 8, 8], DType::F32, &Device::Cpu).unwrap();
    assert!(vlm.generate_multimodal(&img, &[usize::MAX], 2).is_err()); // pre-fix: silently wrong Ok
    assert!(vlm.generate_multimodal(&img, &[1, 2], 2).is_ok());
}

#[test]
fn af1_graph_stats_deep_chain_no_stack_overflow() {
    use oxideon::graph::analyze_graph;
    let x0 = Tensor::from_slice(&[1f32], &[1]).unwrap();
    let one = Tensor::from_slice(&[1f32], &[1]).unwrap();
    let mut acc = x0.requires_grad_(true);
    for _ in 0..100_000 {
        acc = acc.add(&one).unwrap();
    }
    let stats = analyze_graph(&[acc]); // pre-fix: stack overflow abort
    assert_eq!(stats.total_nodes, 100_002);
    assert_eq!(stats.op_nodes, 100_000);
}

/// Builds a minimal GGUF v3 file holding one 256-element super-block of `type_id`.
fn al_gguf_one_kblock(type_id: u32, block_len: usize) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&0x4655_4747u32.to_le_bytes()); // magic
    b.extend_from_slice(&3u32.to_le_bytes()); // version
    b.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
    b.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count
    b.extend_from_slice(&1u64.to_le_bytes()); // name len
    b.push(b'w'); // name
    b.extend_from_slice(&1u32.to_le_bytes()); // n_dims
    b.extend_from_slice(&256u64.to_le_bytes()); // dim
    b.extend_from_slice(&type_id.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes()); // offset
    while b.len() % 32 != 0 {
        b.push(0); // default alignment pad
    }
    b.extend_from_slice(&vec![0u8; block_len]);
    b
}

/// Minimal Mamba-kind block: only `d_conv` is hostile (panics before any tensor use).
fn an_ssm_mamba_tiny() -> oxideon::nn::gguf_ssm::SsmBlock {
    use oxideon::nn::gguf_ssm::{SsmBlock, SsmConfig, SsmKind};
    use oxideon::tensor::Tensor;
    SsmBlock {
        cfg: SsmConfig {
            kind: SsmKind::Mamba,
            d_conv: 4,
            d_inner: 8,
            d_state: 4,
            dt_rank: 2,
            n_group: 1,
            dt_b_c_rms: false,
        },
        in_proj: Tensor::zeros_f32(&[16, 4]).unwrap(),
        conv_w: vec![0.0; 8 * 4],
        conv_b: vec![0.0; 8],
        x_proj: None,
        dt_proj: None,
        dt_bias: vec![0.0; 8],
        a: vec![0.0; 8 * 4],
        d: vec![0.0; 8],
        norm_w: None,
        dt_b_c_norm: None,
        out_proj: Tensor::zeros_f32(&[4, 8]).unwrap(),
        eps: 1e-5,
        in_scale: None,
        out_scale: None,
        gdn: None,
        kda: None,
    }
}

fn an_ssm_mamba2_tiny(dt_rank: usize, n_group: usize) -> oxideon::nn::gguf_ssm::SsmBlock {
    use oxideon::nn::gguf_ssm::{SsmBlock, SsmConfig, SsmKind};
    use oxideon::tensor::Tensor;
    let (di, ds) = (8usize, 4usize);
    let ch = di + 2 * n_group * ds;
    SsmBlock {
        cfg: SsmConfig {
            kind: SsmKind::Mamba2,
            d_conv: 4,
            d_inner: di,
            d_state: ds,
            dt_rank,
            n_group,
            dt_b_c_rms: false,
        },
        in_proj: Tensor::zeros_f32(&[64, 4]).unwrap(),
        conv_w: vec![0.0; ch * 4],
        conv_b: vec![0.0; ch],
        x_proj: None,
        dt_proj: None,
        dt_bias: vec![0.0; dt_rank.max(2)],
        a: vec![0.0; dt_rank.max(2)],
        d: vec![0.0; dt_rank.max(2)],
        norm_w: None,
        dt_b_c_norm: None,
        out_proj: Tensor::zeros_f32(&[4, 8]).unwrap(),
        eps: 1e-5,
        in_scale: None,
        out_scale: None,
        gdn: None,
        kda: None,
    }
}

fn an_ssm_lightning_tiny(
    d_inner: usize,
    dt_rank: usize,
    d_state: usize,
) -> oxideon::nn::gguf_ssm::SsmBlock {
    use oxideon::nn::gguf_ssm::{SsmBlock, SsmConfig, SsmKind};
    use oxideon::tensor::Tensor;
    SsmBlock {
        cfg: SsmConfig {
            kind: SsmKind::Lightning,
            d_conv: 4,
            d_inner,
            d_state,
            dt_rank,
            n_group: 1,
            dt_b_c_rms: false,
        },
        in_proj: Tensor::zeros_f32(&[24, 4]).unwrap(),
        conv_w: vec![],
        conv_b: vec![],
        x_proj: Some(Tensor::zeros_f32(&[8, 4]).unwrap()),
        dt_proj: None,
        dt_bias: vec![],
        a: vec![0.0; 2],
        d: vec![],
        norm_w: Some(vec![1.0; 8]),
        dt_b_c_norm: None,
        out_proj: Tensor::zeros_f32(&[4, 8]).unwrap(),
        eps: 1e-5,
        in_scale: None,
        out_scale: None,
        gdn: None,
        kda: None,
    }
}

#[test]
fn an15_ssm_zero_heads_is_err_not_panic() {
    use oxideon::tensor::Tensor;
    // Mamba2 `di / nh` with tampered `dt_rank = 0`: pre-fix div-by-zero.
    let blk = an_ssm_mamba2_tiny(0, 1);
    let x = Tensor::zeros_f32(&[1, 2, 4]).unwrap();
    assert!(blk.forward(&x, &mut None).is_err());
}

#[test]
fn an16_ssm_fewer_heads_than_groups_is_err_not_panic() {
    use oxideon::tensor::Tensor;
    // Mamba2 `h / (nh / g)` with nh=2 < g=4: pre-fix `nh / g == 0` traps.
    let blk = an_ssm_mamba2_tiny(2, 4);
    let x = Tensor::zeros_f32(&[1, 2, 4]).unwrap();
    assert!(blk.forward(&x, &mut None).is_err());
}

#[test]
fn an17_ssm_lightning_state_len_overflow_is_err_not_panic() {
    use oxideon::tensor::Tensor;
    // `nh * hd * hd` with 2^22 heads/channels: pre-fix multiply overflow.
    let blk = an_ssm_lightning_tiny(8, 1usize << 22, 1usize << 22);
    let x = Tensor::zeros_f32(&[1, 2, 4]).unwrap();
    assert!(blk.forward(&x, &mut None).is_err());
}

#[test]
fn an18_ssm_lightning_output_overflow_is_err_not_panic() {
    use oxideon::tensor::Tensor;
    // `b * s * di` with di = 2^62: pre-fix multiply overflow.
    let blk = an_ssm_lightning_tiny(1usize << 62, 2, 2);
    let x = Tensor::zeros_f32(&[1, 2, 4]).unwrap();
    assert!(blk.forward(&x, &mut None).is_err());
}

/// Valid Mamba block (di=8, ds=4, dt_rank=2) over x [1, 2, 4].
fn ap_ssm_mamba_ok() -> oxideon::nn::gguf_ssm::SsmBlock {
    use oxideon::nn::gguf_ssm::{SsmBlock, SsmConfig, SsmKind};
    use oxideon::tensor::Tensor;
    SsmBlock {
        cfg: SsmConfig {
            kind: SsmKind::Mamba,
            d_conv: 4,
            d_inner: 8,
            d_state: 4,
            dt_rank: 2,
            n_group: 1,
            dt_b_c_rms: false,
        },
        in_proj: Tensor::zeros_f32(&[16, 4]).unwrap(),
        conv_w: vec![0.0; 32],
        conv_b: vec![0.0; 8],
        x_proj: Some(Tensor::zeros_f32(&[10, 8]).unwrap()),
        dt_proj: Some(Tensor::zeros_f32(&[8, 2]).unwrap()),
        dt_bias: vec![0.0; 8],
        a: vec![0.0; 32],
        d: vec![0.0; 8],
        norm_w: None,
        dt_b_c_norm: None,
        out_proj: Tensor::zeros_f32(&[4, 8]).unwrap(),
        eps: 1e-5,
        in_scale: None,
        out_scale: None,
        gdn: None,
        kda: None,
    }
}

/// Valid Mamba-2 block (di=8, ds=4, heads=2, groups=1) over x [1, 2, 4].
fn ap_ssm_mamba2_ok() -> oxideon::nn::gguf_ssm::SsmBlock {
    use oxideon::nn::gguf_ssm::{SsmBlock, SsmConfig, SsmKind};
    use oxideon::tensor::Tensor;
    SsmBlock {
        cfg: SsmConfig {
            kind: SsmKind::Mamba2,
            d_conv: 4,
            d_inner: 8,
            d_state: 4,
            dt_rank: 2,
            n_group: 1,
            dt_b_c_rms: false,
        },
        in_proj: Tensor::zeros_f32(&[26, 4]).unwrap(),
        conv_w: vec![0.0; 64],
        conv_b: vec![0.0; 16],
        x_proj: None,
        dt_proj: None,
        dt_bias: vec![0.0; 2],
        a: vec![0.0; 2],
        d: vec![0.0; 2],
        norm_w: None,
        dt_b_c_norm: None,
        out_proj: Tensor::zeros_f32(&[4, 8]).unwrap(),
        eps: 1e-5,
        in_scale: None,
        out_scale: None,
        gdn: None,
        kda: None,
    }
}

/// Valid PLaMo-2 block (di=8, ds=4, heads=2, dt_dim=2) over x [1, 2, 4].
fn ap_ssm_plamo2_ok() -> oxideon::nn::gguf_ssm::SsmBlock {
    use oxideon::nn::gguf_ssm::{SsmBlock, SsmConfig, SsmKind};
    use oxideon::tensor::Tensor;
    SsmBlock {
        cfg: SsmConfig {
            kind: SsmKind::Plamo2,
            d_conv: 4,
            d_inner: 8,
            d_state: 4,
            dt_rank: 2,
            n_group: 1,
            dt_b_c_rms: false,
        },
        in_proj: Tensor::zeros_f32(&[16, 4]).unwrap(),
        conv_w: vec![0.0; 32],
        conv_b: vec![0.0; 8],
        x_proj: Some(Tensor::zeros_f32(&[10, 8]).unwrap()),
        dt_proj: Some(Tensor::zeros_f32(&[2, 2]).unwrap()),
        dt_bias: vec![0.0; 2],
        a: vec![0.0; 2],
        d: vec![0.0; 2],
        norm_w: None,
        dt_b_c_norm: Some([vec![1.0; 2], vec![1.0; 4], vec![1.0; 4]]),
        out_proj: Tensor::zeros_f32(&[4, 8]).unwrap(),
        eps: 1e-5,
        in_scale: None,
        out_scale: None,
        gdn: None,
        kda: None,
    }
}

/// Valid KDA block (di=8, hd=4, heads=2) over x [1, 2, 4].
fn ap_ssm_kda_ok() -> oxideon::nn::gguf_ssm::SsmBlock {
    use oxideon::nn::gguf_ssm::{KdaParts, SsmBlock, SsmConfig, SsmKind};
    use oxideon::tensor::Tensor;
    SsmBlock {
        cfg: SsmConfig {
            kind: SsmKind::Kda,
            d_conv: 4,
            d_inner: 8,
            d_state: 4,
            dt_rank: 2,
            n_group: 1,
            dt_b_c_rms: false,
        },
        in_proj: Tensor::zeros_f32(&[24, 4]).unwrap(),
        conv_w: vec![0.0; 96],
        conv_b: vec![0.0; 24],
        x_proj: None,
        dt_proj: None,
        dt_bias: vec![0.0; 8],
        a: vec![0.0; 2],
        d: vec![],
        norm_w: Some(vec![1.0; 4]),
        dt_b_c_norm: None,
        out_proj: Tensor::zeros_f32(&[4, 8]).unwrap(),
        eps: 1e-5,
        in_scale: None,
        out_scale: None,
        gdn: None,
        kda: Some(KdaParts {
            f_a: Tensor::zeros_f32(&[8, 4]).unwrap(),
            f_b: None,
            beta_proj: Tensor::zeros_f32(&[2, 4]).unwrap(),
            g_a: Tensor::zeros_f32(&[8, 4]).unwrap(),
            g_b: None,
            lower_bound: None,
            bound_negated: false,
            l2_eps: None,
        }),
    }
}

/// Valid gated delta-net block (di=8, sk=sv=4, value heads=2, key heads=2).
fn ap_ssm_gdn_ok() -> oxideon::nn::gguf_ssm::SsmBlock {
    use oxideon::nn::gguf_ssm::{GdnParts, SsmBlock, SsmConfig, SsmKind};
    use oxideon::tensor::Tensor;
    SsmBlock {
        cfg: SsmConfig {
            kind: SsmKind::GatedDelta,
            d_conv: 4,
            d_inner: 8,
            d_state: 4,
            dt_rank: 2,
            n_group: 2,
            dt_b_c_rms: false,
        },
        in_proj: Tensor::zeros_f32(&[24, 4]).unwrap(),
        conv_w: vec![0.0; 96],
        conv_b: vec![],
        x_proj: None,
        dt_proj: None,
        dt_bias: vec![0.0; 2],
        a: vec![0.0; 2],
        d: vec![],
        norm_w: Some(vec![1.0; 4]),
        dt_b_c_norm: None,
        out_proj: Tensor::zeros_f32(&[4, 8]).unwrap(),
        eps: 1e-5,
        in_scale: None,
        out_scale: None,
        gdn: Some(GdnParts {
            z_proj: Tensor::zeros_f32(&[8, 4]).unwrap(),
            beta_proj: Tensor::zeros_f32(&[2, 4]).unwrap(),
            alpha_proj: Tensor::zeros_f32(&[2, 4]).unwrap(),
            z_scale: None,
            beta_scale: None,
            alpha_scale: None,
            sigmoid_gate: false,
        }),
        kda: None,
    }
}

#[test]
fn ap1_ssm_mamba_valid_block_forwards_ok() {
    use oxideon::tensor::Tensor;
    let blk = ap_ssm_mamba_ok();
    let x = Tensor::zeros_f32(&[1, 2, 4]).unwrap();
    let out = blk.forward(&x, &mut None).unwrap();
    assert_eq!(out.shape(), &[1, 2, 4]);
}

#[test]
fn ap2_ssm_mamba2_valid_block_forwards_ok() {
    use oxideon::tensor::Tensor;
    let blk = ap_ssm_mamba2_ok();
    let x = Tensor::zeros_f32(&[1, 2, 4]).unwrap();
    let out = blk.forward(&x, &mut None).unwrap();
    assert_eq!(out.shape(), &[1, 2, 4]);
}

#[test]
fn ap3_ssm_plamo2_valid_block_forwards_ok() {
    use oxideon::tensor::Tensor;
    let blk = ap_ssm_plamo2_ok();
    let x = Tensor::zeros_f32(&[1, 2, 4]).unwrap();
    let out = blk.forward(&x, &mut None).unwrap();
    assert_eq!(out.shape(), &[1, 2, 4]);
}

#[test]
fn ap4_ssm_kda_valid_block_forwards_ok() {
    use oxideon::tensor::Tensor;
    let blk = ap_ssm_kda_ok();
    let x = Tensor::zeros_f32(&[1, 2, 4]).unwrap();
    let out = blk.forward(&x, &mut None).unwrap();
    assert_eq!(out.shape(), &[1, 2, 4]);
}

#[test]
fn ap5_ssm_gdn_valid_block_forwards_ok() {
    use oxideon::tensor::Tensor;
    let blk = ap_ssm_gdn_ok();
    let x = Tensor::zeros_f32(&[1, 2, 4]).unwrap();
    let out = blk.forward(&x, &mut None).unwrap();
    assert_eq!(out.shape(), &[1, 2, 4]);
}

#[test]
fn ap6_ssm_lightning_valid_block_forwards_ok() {
    use oxideon::tensor::Tensor;
    let blk = an_ssm_lightning_tiny(8, 2, 4);
    let x = Tensor::zeros_f32(&[1, 2, 4]).unwrap();
    let out = blk.forward(&x, &mut None).unwrap();
    assert_eq!(out.shape(), &[1, 2, 4]);
}

#[test]
fn ap7_ssm_shortconv_loader_config_forwards_ok() {
    use oxideon::nn::gguf_ssm::{SsmBlock, SsmConfig, SsmKind};
    use oxideon::tensor::Tensor;
    // The GGUF loader builds ShortConv configs with d_state/dt_rank == 0
    // (the block reads only d_conv and the width of x): pre-fix the entry
    // positivity check rejected every such block.
    let blk = SsmBlock {
        cfg: SsmConfig {
            kind: SsmKind::ShortConv,
            d_conv: 4,
            d_inner: 4,
            d_state: 0,
            dt_rank: 0,
            n_group: 1,
            dt_b_c_rms: false,
        },
        in_proj: Tensor::zeros_f32(&[12, 4]).unwrap(),
        conv_w: vec![0.0; 16],
        conv_b: vec![0.0; 4],
        x_proj: None,
        dt_proj: None,
        dt_bias: vec![],
        a: vec![],
        d: vec![],
        norm_w: None,
        dt_b_c_norm: None,
        out_proj: Tensor::zeros_f32(&[4, 4]).unwrap(),
        eps: 1e-5,
        in_scale: None,
        out_scale: None,
        gdn: None,
        kda: None,
    };
    let x = Tensor::zeros_f32(&[1, 2, 4]).unwrap();
    let out = blk.forward(&x, &mut None).unwrap();
    assert_eq!(out.shape(), &[1, 2, 4]);
}

#[test]
fn ap8_ssm_mamba2_narrow_in_proj_is_err_not_panic() {
    use oxideon::tensor::Tensor;
    // in_proj rows 24 < di + ch + heads = 26: the scan slices
    // `row[di + ch..di + ch + nh]` past the row end pre-fix.
    let mut blk = ap_ssm_mamba2_ok();
    blk.in_proj = Tensor::zeros_f32(&[24, 4]).unwrap();
    let x = Tensor::zeros_f32(&[1, 2, 4]).unwrap();
    assert!(blk.forward(&x, &mut None).is_err());
}

#[test]
fn ap9_ssm_mamba_narrow_x_proj_is_err_not_panic() {
    use oxideon::tensor::Tensor;
    // x_proj rows 9 < dt_rank + 2 * d_state = 10: the B/C/dt copies read
    // `x_db[r * wd..(r + 1) * wd]` with the config width pre-fix.
    let mut blk = ap_ssm_mamba_ok();
    blk.x_proj = Some(Tensor::zeros_f32(&[9, 8]).unwrap());
    let x = Tensor::zeros_f32(&[1, 2, 4]).unwrap();
    assert!(blk.forward(&x, &mut None).is_err());
}

#[test]
fn ap10_ssm_mamba_truncated_conv_w_is_err_not_panic() {
    use oxideon::tensor::Tensor;
    // conv_w one tap short: `conv_w[ci * d_conv + k]` indexes past the end.
    let mut blk = ap_ssm_mamba_ok();
    blk.conv_w = vec![0.0; 31];
    let x = Tensor::zeros_f32(&[1, 2, 4]).unwrap();
    assert!(blk.forward(&x, &mut None).is_err());
}

#[test]
fn ap11_ssm_lightning_narrow_in_proj_is_err_not_panic() {
    use oxideon::tensor::Tensor;
    // in_proj rows 16 < 3 * d_inner = 24: the per-head qkv slices overrun.
    let mut blk = an_ssm_lightning_tiny(8, 2, 4);
    blk.in_proj = Tensor::zeros_f32(&[16, 4]).unwrap();
    let x = Tensor::zeros_f32(&[1, 2, 4]).unwrap();
    assert!(blk.forward(&x, &mut None).is_err());
}

#[test]
fn ap12_ssm_kda_narrow_beta_proj_is_err_not_panic() {
    use oxideon::tensor::Tensor;
    // beta_proj rows 1 < heads = 2: `beta[r * nh + h]` overruns the buffer.
    let mut blk = ap_ssm_kda_ok();
    blk.kda.as_mut().unwrap().beta_proj = Tensor::zeros_f32(&[1, 4]).unwrap();
    let x = Tensor::zeros_f32(&[1, 2, 4]).unwrap();
    assert!(blk.forward(&x, &mut None).is_err());
}

#[test]
fn ap13_ssm_kda_wrong_norm_len_is_err_not_panic() {
    use oxideon::tensor::Tensor;
    // The KDA output norm is per-head (head_dim wide, as the loader stores
    // it); a full d_inner-wide norm silently truncated in the zip pre-fix.
    let mut blk = ap_ssm_kda_ok();
    blk.norm_w = Some(vec![1.0; 8]);
    let x = Tensor::zeros_f32(&[1, 2, 4]).unwrap();
    assert!(blk.forward(&x, &mut None).is_err());
}

#[test]
fn ap14_ssm_gdn_narrow_z_proj_is_err_not_panic() {
    use oxideon::tensor::Tensor;
    // z_proj rows 4 < d_inner = 8: the gate slice `z[r * di + h * sv..]`
    // overruns on the second token pre-fix.
    let mut blk = ap_ssm_gdn_ok();
    blk.gdn.as_mut().unwrap().z_proj = Tensor::zeros_f32(&[4, 4]).unwrap();
    let x = Tensor::zeros_f32(&[1, 2, 4]).unwrap();
    assert!(blk.forward(&x, &mut None).is_err());
}

#[test]
fn ap15_ssm_gdn_truncated_dt_bias_is_err_not_panic() {
    use oxideon::tensor::Tensor;
    // dt_bias shorter than the value-head count: `dt_bias[h]` overruns.
    let mut blk = ap_ssm_gdn_ok();
    blk.dt_bias = vec![0.0; 1];
    let x = Tensor::zeros_f32(&[1, 2, 4]).unwrap();
    assert!(blk.forward(&x, &mut None).is_err());
}

#[test]
fn ap16_ssm_plamo2_narrow_dt_proj_is_err_not_panic() {
    use oxideon::tensor::Tensor;
    // dt_proj rows 1 < heads = 2: `dt[r * nh + h]` overruns on token two.
    let mut blk = ap_ssm_plamo2_ok();
    blk.dt_proj = Some(Tensor::zeros_f32(&[1, 2]).unwrap());
    let x = Tensor::zeros_f32(&[1, 2, 4]).unwrap();
    assert!(blk.forward(&x, &mut None).is_err());
}

#[test]
fn ap17_ssm_shortconv_narrow_in_proj_is_err_not_panic() {
    use oxideon::nn::gguf_ssm::{SsmBlock, SsmConfig, SsmKind};
    use oxideon::tensor::Tensor;
    // in_proj rows 8 < 3 * d = 12: the [b | c | x] row split overruns.
    // (Positive d_state/dt_rank so the pre-fix entry check lets it through
    // to the panicking slice.)
    let blk = SsmBlock {
        cfg: SsmConfig {
            kind: SsmKind::ShortConv,
            d_conv: 4,
            d_inner: 4,
            d_state: 4,
            dt_rank: 2,
            n_group: 1,
            dt_b_c_rms: false,
        },
        in_proj: Tensor::zeros_f32(&[8, 4]).unwrap(),
        conv_w: vec![0.0; 16],
        conv_b: vec![0.0; 4],
        x_proj: None,
        dt_proj: None,
        dt_bias: vec![],
        a: vec![],
        d: vec![],
        norm_w: None,
        dt_b_c_norm: None,
        out_proj: Tensor::zeros_f32(&[4, 4]).unwrap(),
        eps: 1e-5,
        in_scale: None,
        out_scale: None,
        gdn: None,
        kda: None,
    };
    let x = Tensor::zeros_f32(&[1, 2, 4]).unwrap();
    assert!(blk.forward(&x, &mut None).is_err());
}

#[test]
fn ap18_ssm_mamba2_indivisible_inner_is_err_not_panic() {
    use oxideon::tensor::Tensor;
    // d_inner = 8 with 3 heads: head_dim truncates to 2 and the scan
    // silently drops the last two channels pre-fix (the GGUF config
    // loader rejects this geometry; the pub-field path did not).
    let mut blk = ap_ssm_mamba2_ok();
    blk.cfg.dt_rank = 3;
    blk.in_proj = Tensor::zeros_f32(&[27, 4]).unwrap();
    blk.dt_bias = vec![0.0; 3];
    blk.a = vec![0.0; 3];
    blk.d = vec![0.0; 3];
    let x = Tensor::zeros_f32(&[1, 2, 4]).unwrap();
    assert!(blk.forward(&x, &mut None).is_err());
}

#[test]
fn ap19_ssm_lightning_head_width_mismatch_is_err_not_panic() {
    use oxideon::tensor::Tensor;
    // heads * head_dim = 2 * 2 != d_inner = 8: the per-head slices leave
    // most of each qkv row unread pre-fix instead of erroring.
    let blk = an_ssm_lightning_tiny(8, 2, 2);
    let x = Tensor::zeros_f32(&[1, 2, 4]).unwrap();
    assert!(blk.forward(&x, &mut None).is_err());
}

#[test]
fn aq1_mod_nan_capacity_is_err_not_ok() {
    use oxideon::adaptive::MixtureOfDepths;
    // NaN fails both range comparisons in `new`: pre-fix it was accepted
    // and forward silently routed a single token (capacity saturates to
    // k = 1) instead of honouring the documented (0, 1] contract.
    assert!(MixtureOfDepths::new(8, f64::NAN).is_err());
    assert!(MixtureOfDepths::new(8, f64::INFINITY).is_err());
    assert!(MixtureOfDepths::new(8, 0.5).is_ok());
}

#[test]
fn as1_symbolic_gp_deep_tree_methods_do_not_overflow() {
    use oxideon::reasoning::{OpType, SymbolicNode};
    // 200k-deep hand-built chain: depth/complexity/to_formula recursed per
    // level pre-fix (process abort), and dropping the tree recursed too.
    let mut tree = SymbolicNode::Variable(0);
    for _ in 0..200_000 {
        tree = SymbolicNode::Unary(OpType::Sin, Box::new(tree));
    }
    assert_eq!(tree.depth(), 200_001);
    assert_eq!(tree.complexity(), 200_001);
    let f = tree.to_formula();
    assert!(f.starts_with("sin(sin(") && f.ends_with("))"));
    // No mem::forget: iterative Drop must let this fall out of scope.
}

#[test]
fn as2_symbolic_gp_small_tree_semantics_unchanged() {
    use oxideon::reasoning::{OpType, SymbolicNode};
    // (sin(x0) + 2): locks depth/complexity/formula conventions.
    let tree = SymbolicNode::Binary(
        OpType::Add,
        Box::new(SymbolicNode::Unary(
            OpType::Sin,
            Box::new(SymbolicNode::Variable(0)),
        )),
        Box::new(SymbolicNode::Constant(2.0)),
    );
    assert_eq!(tree.depth(), 3);
    assert_eq!(tree.complexity(), 4);
    assert_eq!(tree.to_formula(), "(sin(x0) + 2)");
}

#[test]
fn au1_tda_nan_distance_matrix_is_ok_not_panic() {
    use oxideon::science::tda::PersistentHomologyEngine;
    use oxideon::tensor::Tensor;
    // NaN distances fail the `d <= max_filtration` inclusion filter, so the
    // filtration sort never sees them; this locks that defended behavior
    // (the comparator is also unwrap-free as of batch AU).
    let dist = Tensor::from_slice(
        &[
            0.0f32,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            0.0,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            0.0,
        ],
        &[3, 3],
    )
    .unwrap();
    let diag = PersistentHomologyEngine::new(1, 2.5)
        .compute_from_distance_matrix(&dist)
        .unwrap();
    assert_eq!(diag.dimension_pairs(0).len(), 3);
}

#[test]
fn au2_tda_wasserstein_nan_death_is_finite_not_panic() {
    use oxideon::science::tda::{PersistenceDiagram, PersistencePair, wasserstein_distance};
    // `persistence()` clamps to [0, +inf] (a NaN death becomes 0.0), so the
    // descending sort is total; locks the defended behavior.
    let diag = PersistenceDiagram::new(vec![
        PersistencePair::new(0, 0.0, f64::NAN),
        PersistencePair::new(0, 1.0, 3.0),
    ]);
    let d = wasserstein_distance(&diag, &diag, 0);
    assert!(d.is_finite() && d >= 0.0, "got {d}");
    // Essential classes (death = +inf, the standard TDA encoding) in
    // identical diagrams: (inf - inf) made the self-distance NaN pre-fix.
    let essential = PersistenceDiagram::new(vec![PersistencePair::new(0, 0.0, f64::INFINITY)]);
    assert_eq!(wasserstein_distance(&essential, &essential, 0), 0.0);
}

#[test]
fn av1_prop_expr_deep_chain_clone_and_drop_do_not_overflow() {
    use oxideon::reasoning::PropExpr;
    // 200k-deep Not chain: derived Clone and the derived Drop both recursed
    // per level pre-fix (process abort). The prover entry points reject
    // such depth before cloning (batch AR), but the type itself did not.
    let mut expr = PropExpr::var("P");
    for _ in 0..200_000 {
        expr = expr.not();
    }
    let cloned = expr.clone();
    // Walk both chains iteratively (derived PartialEq recurses too).
    let count_nots = |mut e: &PropExpr| -> usize {
        let mut n = 0;
        while let PropExpr::Not(sub) = e {
            n += 1;
            e = sub;
        }
        n
    };
    assert_eq!(count_nots(&expr), 200_000);
    assert_eq!(count_nots(&cloned), 200_000);
    // Both fall out of scope here: iterative Drop must not recurse.
}

#[test]
fn av2_prop_expr_shallow_semantics_unchanged() {
    use oxideon::reasoning::PropExpr;
    // (P -> Q) <-> P: locks Clone/PartialEq structure preservation.
    let expr = PropExpr::var("P")
        .implies(PropExpr::var("Q"))
        .iff(PropExpr::var("P"));
    assert_eq!(expr.clone(), expr);
}

#[test]
fn aw1_symdim_deep_chain_clone_and_drop_do_not_overflow() {
    use oxideon::compiler::symbolic_shape::SymDim;
    // 200k-deep Add chain: derived Clone/Drop recursed per level pre-fix.
    let mut expr = SymDim::val(1);
    for _ in 0..200_000 {
        expr = SymDim::Add(Box::new(expr), Box::new(SymDim::val(1)));
    }
    let cloned = expr.clone();
    let count_adds = |mut e: &SymDim| -> usize {
        let mut n = 0;
        while let SymDim::Add(sub, _) = e {
            n += 1;
            e = sub;
        }
        n
    };
    assert_eq!(count_adds(&expr), 200_000);
    assert_eq!(count_adds(&cloned), 200_000);
    // Both fall out of scope here: iterative Drop must not recurse.
}

#[test]
fn aw2_symdim_resolve_depth_guard() {
    use oxideon::compiler::symbolic_shape::SymDim;
    use std::collections::HashMap;
    // A 500-deep chain resolves (each Add adds the Var leaf value 1).
    let mut expr = SymDim::var("x");
    for _ in 0..500 {
        expr = SymDim::Add(Box::new(expr), Box::new(SymDim::val(0)));
    }
    let env: HashMap<&str, usize> = [("x", 7usize)].into_iter().collect();
    assert_eq!(expr.resolve(&env).unwrap(), 7);
    // A 2000-deep chain exceeds the depth guard: Err, not a stack abort.
    let mut deep = SymDim::var("x");
    for _ in 0..2000 {
        deep = SymDim::Add(Box::new(deep), Box::new(SymDim::val(0)));
    }
    assert!(deep.resolve(&env).is_err());
    assert!(deep.bind_dim(7, &mut HashMap::new()).is_err());
}

#[test]
fn aw3_symdim_display_semantics_unchanged() {
    use oxideon::compiler::symbolic_shape::SymDim;
    let expr = SymDim::Add(Box::new(SymDim::var("B")), Box::new(SymDim::val(2)));
    assert_eq!(format!("{expr}"), "(B + 2)");
    let expr = SymDim::Mul(Box::new(SymDim::var("S")), Box::new(SymDim::var("H")));
    assert_eq!(format!("{expr}"), "(S * H)");
}

#[test]
fn aw4_schema_type_deep_chain_clone_and_drop_do_not_overflow() {
    use oxideon::grammar::SchemaType;
    // 200k-deep Array nesting: derived Clone/Drop recursed pre-fix.
    let mut schema = SchemaType::String;
    for _ in 0..200_000 {
        schema = SchemaType::Array(Box::new(schema));
    }
    let cloned = schema.clone();
    let count_arrays = |mut s: &SchemaType| -> usize {
        let mut n = 0;
        while let SchemaType::Array(sub) = s {
            n += 1;
            s = sub;
        }
        n
    };
    assert_eq!(count_arrays(&schema), 200_000);
    assert_eq!(count_arrays(&cloned), 200_000);
    // Both fall out of scope here: iterative Drop must not recurse.
}

#[test]
fn ao6_spirv_huge_bindings_is_err_not_panic() {
    use oxideon::driver::VulkanSpirvKernel;
    // `num_bindings` is pub: pre-fix `with_capacity(MAX)` panicked
    // ("capacity overflow", demonstrated pre-fix via temp probe).
    let k = VulkanSpirvKernel::new("k", usize::MAX);
    assert!(k.emit_spirv_binary().is_err());
    let ok = VulkanSpirvKernel::new("k", 2);
    assert!(ok.emit_spirv_binary().is_ok());
}

#[test]
fn ao2_affine_grid_absurd_size_is_err_not_panic() {
    use oxideon::tensor::Tensor;
    use oxideon::vision_ops::affine_grid;
    // DEFENDED (kept as a lock): `check_elements` caps n*h*w*2 at 2^31,
    // so h*w*3 (<= 3*2^30) cannot overflow; the absurd size Errs up front.
    let theta = Tensor::zeros_f32(&[1, 2, 3]).unwrap();
    assert!(affine_grid(&theta, [1, 1, 1usize << 31, 1usize << 31], false).is_err());
}

#[test]
fn ao3_grid_sample_infinite_grid_is_ok_not_panic() {
    use oxideon::tensor::Tensor;
    use oxideon::vision_ops::grid_sample;
    // `inf.floor() as isize` saturates to MAX, then `+1` overflows pre-fix
    // (debug panic); post-fix out-of-range corners sample zeros.
    let input = Tensor::ones_f32(&[1, 1, 2, 2]).unwrap();
    let grid = Tensor::from_slice(&[f32::INFINITY, 0.0f32], &[1, 1, 1, 2]).unwrap();
    let out = grid_sample(&input, &grid, false).unwrap();
    assert_eq!(out.shape(), &[1, 1, 1, 1]);
}

#[test]
fn ao2b_grid_sample_smoke() {
    use oxideon::tensor::Tensor;
    use oxideon::vision_ops::grid_sample;
    // Locks forward + backward over the checked output buffer.
    let input = Tensor::ones_f32(&[1, 1, 2, 2])
        .unwrap()
        .requires_grad_(true);
    let grid = Tensor::zeros_f32(&[1, 2, 2, 2])
        .unwrap()
        .requires_grad_(true);
    let out = grid_sample(&input, &grid, true).unwrap();
    out.sum_axes(&[0, 1, 2, 3], false)
        .unwrap()
        .backward()
        .unwrap();
    assert!(input.grad().is_some() && grid.grad().is_some());
}

#[test]
fn ao4_roi_align_empty_spatial_is_err_not_panic() {
    use oxideon::tensor::Tensor;
    use oxideon::vision_ops::roi_align;
    // Zero-height input: the pixel table is empty but the sampler still
    // indexes [0] pre-fix.
    let input = Tensor::zeros_f32(&[1, 1, 0, 4]).unwrap();
    let rois = Tensor::zeros_f32(&[1, 5]).unwrap();
    assert!(roi_align(&input, &rois, [2, 2], 1.0, 1).is_err());
}

#[test]
fn ao5_box_iou_smoke() {
    use oxideon::tensor::Tensor;
    use oxideon::vision_ops::box_iou;
    // Locks the checked n*m output over a tiny pair.
    let a = Tensor::from_slice(&[0.0f32, 0.0, 1.0, 1.0], &[1, 4]).unwrap();
    let b = Tensor::from_slice(&[0.0f32, 0.0, 1.0, 1.0], &[1, 4]).unwrap();
    let iou = box_iou(&a, &b).unwrap();
    assert_eq!(iou.shape(), &[1, 1]);
}

#[test]
fn ao1_fused_linear_scalar_bias_is_err_not_panic() {
    use oxideon::fused_kernels::{FusedActivation, fused_linear_activation};
    use oxideon::tensor::Tensor;
    // If scalar matmul succeeds, `y` is 0-D and `b_shape.last_mut().unwrap()`
    // panics; if matmul rejects scalars this Errs either way (defended).
    let x = Tensor::from_slice(&[2.0f32], &[]).unwrap();
    let w = Tensor::from_slice(&[3.0f32], &[]).unwrap();
    let b = Tensor::from_slice(&[1.0f32], &[1]).unwrap();
    assert!(fused_linear_activation(&x, &w, Some(&b), FusedActivation::None).is_err());
}

#[test]
fn an19_mlp_nan_dropout_is_err_not_panic() {
    use oxideon::nn::architectures::{ActivationKind, MLP};
    // `dropout_p` reaches the asserting `Dropout::new` (as f32): pre-fix NaN panics.
    assert!(MLP::new(&[4, 4, 2], ActivationKind::ReLU, true, f64::NAN, false).is_err());
}

#[test]
fn an14_ssm_zero_d_conv_is_err_not_panic() {
    use oxideon::tensor::Tensor;
    let mut blk = an_ssm_mamba_tiny();
    // `d_conv - 1` appears in all five forwards: pre-fix it underflows.
    blk.cfg.d_conv = 0;
    let x = Tensor::zeros_f32(&[1, 2, 4]).unwrap();
    assert!(blk.forward(&x, &mut None).is_err());
}

#[test]
fn an13_transformer_nan_dropout_is_err_not_panic() {
    use oxideon::nn::transformer::{FeedForward, TransformerDecoderLayer, TransformerEncoderLayer};
    use oxideon::nn::transfusion::{TransfusionConfig, TransfusionModel};
    // All three `new`s forward dropout_p to the asserting `Dropout::new`; the
    // transfusion model inherits the panic through the encoder layer.
    assert!(FeedForward::new(8, 16, f32::NAN).is_err());
    assert!(TransformerEncoderLayer::new(8, 2, 16, f32::NAN).is_err());
    assert!(TransformerDecoderLayer::new(8, 2, 16, f32::NAN).is_err());
    let cfg = TransfusionConfig {
        dropout: f32::NAN,
        ..TransfusionConfig::default()
    };
    assert!(TransfusionModel::new(cfg).is_err());
}

#[test]
fn an9_polychronous_nan_max_delay_is_err_not_panic() {
    use oxideon::nn::neuromorphic::PolychronousNeuralLayer;
    // NaN passes `max_delay <= 0.0`; the NaN bound later panics `clamp`.
    assert!(PolychronousNeuralLayer::new(4, 4, f64::NAN).is_err());
}

#[test]
fn an10_set_weights_tampered_dims_is_err_not_panic() {
    use oxideon::nn::neuromorphic::PolychronousNeuralLayer;
    let mut layer = PolychronousNeuralLayer::new(4, 4, 0.01).unwrap();
    // `num_inputs * num_outputs` on pub dims: pre-fix this overflows.
    layer.num_inputs = usize::MAX;
    assert!(layer.set_weights(vec![]).is_err());
}

#[test]
fn an11_event_driven_truncated_weights_is_err_not_panic() {
    use oxideon::nn::neuromorphic::PolychronousNeuralLayer;
    let mut layer = PolychronousNeuralLayer::new(4, 4, 0.01).unwrap();
    // `weights` is pub and never re-validated: pre-fix this indexes OOB.
    layer.weights.truncate(3);
    assert!(layer.forward_event_driven(&[(0.001, 0)], 0.1).is_err());
}

#[test]
fn an12_adapt_stddp_oob_indices_is_err_not_panic() {
    use oxideon::nn::neuromorphic::PolychronousNeuralLayer;
    // Spike indices come straight from the caller: pre-fix both panicked
    // with index OOB (demonstrated pre-fix via catch_unwind temp).
    let mut layer = PolychronousNeuralLayer::new(4, 4, 0.01).unwrap();
    assert!(
        layer
            .adapt_stddp(&[(0.0, 999)], &[(0.0, 0)], 0.001, 0.01, 0.005)
            .is_err()
    );
    assert!(
        layer
            .adapt_stddp(&[(0.0, 0)], &[(0.012, 999)], 0.001, 0.01, 0.005)
            .is_err()
    );
    assert!(
        layer
            .adapt_stddp(&[(0.0, 0)], &[(0.012, 0)], 0.001, 0.01, 0.005)
            .is_ok()
    );
}

#[test]
fn an9b_adapt_stddp_nan_bound_is_err_not_panic() {
    use oxideon::nn::neuromorphic::PolychronousNeuralLayer;
    // `max_delay` is pub: NaN panics `clamp` pre-fix (demonstrated pre-fix).
    let mut layer = PolychronousNeuralLayer::new(4, 4, 0.01).unwrap();
    layer.max_delay = f64::NAN;
    assert!(
        layer
            .adapt_stddp(&[(0.0, 0)], &[(0.012, 0)], 0.001, 0.01, 0.005)
            .is_err()
    );
}

#[test]
fn an8_qlora_nan_dropout_is_err_not_panic() {
    use oxideon::nn::Linear;
    use oxideon::nn::qlora::QLoRALinear;
    use oxideon::quant::{QuantScheme, QuantizedLinear};
    let lin = Linear::with_bias(4, 4, false).unwrap();
    let base = QuantizedLinear::from_linear(&lin, QuantScheme::DEFAULT_Q8_0).unwrap();
    // Same inherited `Dropout::new` assert as an7.
    assert!(QLoRALinear::new(base, 2, 1.0, f32::NAN).is_err());
}

#[test]
fn an7_lora_nan_dropout_is_err_not_panic() {
    use oxideon::nn::Linear;
    use oxideon::nn::lora::LoRALinear;
    let lin = Linear::with_bias(4, 4, false).unwrap();
    // `Dropout::new` asserts on NaN; `from_linear` is fallible, so it must
    // reject NaN itself instead of inheriting the panic.
    assert!(LoRALinear::from_linear(&lin, 2, 1.0, f32::NAN).is_err());
}

#[test]
fn an3_lif_poisoned_state_is_recovered_not_panic() {
    use oxideon::nn::spiking::{LIFNode, ResetMode};
    use oxideon::tensor::Tensor;
    use std::sync::Arc;
    let node = Arc::new(LIFNode::new(0.9, 1.0, 0.0, ResetMode::Hard));
    // Poison the pub membrane lock from another thread.
    let victim = Arc::clone(&node);
    let _ = std::thread::spawn(move || {
        let _g = victim.v.lock().unwrap();
        panic!("intentional poison for an3");
    })
    .join();
    // Pre-fix `.lock().unwrap()` panics; post-fix it recovers.
    let x = Tensor::zeros_f32(&[2, 4]).unwrap();
    assert!(node.forward_step(&x).is_ok());
}

#[test]
fn an4_poisson_absurd_steps_is_err_not_panic() {
    use oxideon::nn::spiking::PoissonEncoder;
    use oxideon::tensor::Tensor;
    let mut enc = PoissonEncoder::new(4);
    // `time_steps` is pub with only a `.max(1)` floor: pre-fix
    // `with_capacity(MAX)` panics ("capacity overflow") before the loop.
    enc.time_steps = usize::MAX;
    let x = Tensor::zeros_f32(&[2, 4]).unwrap();
    assert!(enc.encode(&x).is_err());
}

#[test]
fn an5_stdp_truncated_weights_is_err_not_panic() {
    use oxideon::nn::stdp::{STDPConfig, STDPSynapse};
    use oxideon::tensor::Tensor;
    let mut s = STDPSynapse::new(4, 4, STDPConfig::default()).unwrap();
    // `weights` is pub: a short swap panics the update indexing pre-fix.
    use oxideon::backend::Device;
    use oxideon::dtype::DType;
    s.weights = Tensor::zeros(&[1], DType::F64, &Device::Cpu).unwrap();
    let pre = Tensor::zeros(&[4], DType::F64, &Device::Cpu).unwrap();
    let post = Tensor::zeros(&[4], DType::F64, &Device::Cpu).unwrap();
    assert!(s.step_stdp(&pre, &post).is_err());
}

#[test]
fn an6_stdp_inverted_bounds_is_err_not_panic() {
    use oxideon::nn::stdp::{STDPConfig, STDPSynapse};
    use oxideon::tensor::Tensor;
    let mut s = STDPSynapse::new(4, 4, STDPConfig::default()).unwrap();
    // `config` is pub: min > max panics f64::clamp pre-fix (same as am9).
    s.config.w_min = 5.0;
    s.config.w_max = 1.0;
    use oxideon::backend::Device;
    use oxideon::dtype::DType;
    let pre = Tensor::zeros(&[4], DType::F64, &Device::Cpu).unwrap();
    let post = Tensor::zeros(&[4], DType::F64, &Device::Cpu).unwrap();
    assert!(s.step_stdp(&pre, &post).is_err());
}

#[test]
fn an1_titans_memory_smoke() {
    use oxideon::nn::titans::{TitansBlock, TitansMemoryConfig};
    use oxideon::tensor::Tensor;
    // Locks the m*m scratch / retrieved-buffer path (products are checked).
    let blk = TitansBlock::new(TitansMemoryConfig::new(8, 4)).unwrap();
    let x = Tensor::zeros_f32(&[1, 3, 8]).unwrap();
    let y = blk.forward_block(&x).unwrap();
    assert_eq!(y.shape(), &[1, 3, 8]);
}

#[test]
fn an2_titans_swapped_narrow_projection_is_err_not_panic() {
    use oxideon::nn::Linear;
    use oxideon::nn::titans::{TitansBlock, TitansMemoryConfig};
    use oxideon::tensor::Tensor;
    let mut blk = TitansBlock::new(TitansMemoryConfig::new(8, 4)).unwrap();
    // `w_out` is pub: a narrower swap shortens mem_out, and the gate loop
    // indexes past it pre-fix.
    blk.memory.w_out = Linear::with_bias(4, 7, false).unwrap();
    let x = Tensor::zeros_f32(&[1, 3, 8]).unwrap();
    assert!(blk.forward_block(&x).is_err());
}

#[test]
fn am10b_nas_huge_population_is_err_not_panic() {
    use oxideon::adaptive::nas::EvolutionaryAutoML;
    // `population_size` is pub: pre-fix `with_capacity(MAX)` panicked with
    // "capacity overflow" (demonstrated pre-fix); post-fix this is an Err.
    let mut evo = EvolutionaryAutoML::new(4, 2, 0.1, 100_000, 7);
    evo.population_size = usize::MAX;
    assert!(evo.initialize_population(8, 4).is_err());
}

#[test]
fn am10_nas_tampered_generations_is_err_not_hang() {
    use oxideon::adaptive::nas::EvolutionaryAutoML;
    let mut evo = EvolutionaryAutoML::new(4, 2, 0.1, 100_000, 7);
    // `max_generations` is pub with only a constructor floor: each generation
    // builds and scores whole populations. Pre-fix this loops ~forever
    // (demonstrated with a solo `timeout` kill, never run unbounded pre-fix).
    evo.max_generations = usize::MAX;
    assert!(evo.search(8, 4).is_err());
}

#[test]
fn am11_nas_zero_population_is_err_not_panic() {
    use oxideon::adaptive::nas::EvolutionaryAutoML;
    let mut evo = EvolutionaryAutoML::new(4, 2, 0.1, 100_000, 7);
    // `population_size` is pub: zeroing it makes `population[0]` panic pre-fix.
    evo.population_size = 0;
    assert!(evo.search(8, 4).is_err());
}

#[test]
fn am9_elastic_slice_empty_features_is_err_not_panic() {
    use oxideon::adaptive::elastic::ElasticSubnetExtractor;
    use oxideon::tensor::Tensor;
    // `target.clamp(1, features)` with features == 0 panics (min > max) pre-fix.
    let w_in0 = Tensor::zeros_f32(&[5, 0]).unwrap();
    assert!(ElasticSubnetExtractor::slice_linear_width(&w_in0, None, 0.5, 0.5).is_err());
    let w_out0 = Tensor::zeros_f32(&[0, 5]).unwrap();
    assert!(ElasticSubnetExtractor::slice_linear_width(&w_out0, None, 0.5, 0.5).is_err());
}

#[test]
fn am6_dit_tampered_patch_size_is_err_not_panic() {
    use oxideon::tensor::Tensor;
    use oxideon::video::flow_dit::{RectifiedFlowVideoDiT, VideoDiTConfig};
    let mut m = RectifiedFlowVideoDiT::new(VideoDiTConfig::default()).unwrap();
    // `config` is pub: absurd patches with an all-empty video pass the
    // divisibility checks (0 % p == 0), then `c*pt*ph*pw` overflows pre-fix.
    m.config.patch_size = (1usize << 32, 1usize << 32, 1usize << 32);
    let x = Tensor::zeros_f32(&[1, 4, 0, 0, 0]).unwrap();
    let t = Tensor::zeros_f32(&[1, 1]).unwrap();
    let c = Tensor::zeros_f32(&[1, 2, 32]).unwrap();
    assert!(m.forward(&x, &t, &c).is_err());
}

#[test]
fn am7_dit_swapped_final_proj_is_err_not_panic() {
    use oxideon::nn::Linear;
    use oxideon::tensor::Tensor;
    use oxideon::video::flow_dit::{RectifiedFlowVideoDiT, VideoDiTConfig};
    let mut m = RectifiedFlowVideoDiT::new(VideoDiTConfig::default()).unwrap();
    // `final_proj` is pub: a narrower swap makes unpatchify index past the
    // token buffer pre-fix (default patch_dim is 4*1*2*2 = 16).
    m.final_proj = Linear::new(64, 1).unwrap();
    let x = Tensor::zeros_f32(&[1, 4, 2, 4, 4]).unwrap();
    let t = Tensor::zeros_f32(&[1, 1]).unwrap();
    let c = Tensor::zeros_f32(&[1, 2, 32]).unwrap();
    assert!(m.forward(&x, &t, &c).is_err());
}

#[test]
fn am8_dit_forward_scalar_video_is_err_not_panic() {
    use oxideon::tensor::Tensor;
    use oxideon::video::flow_dit::{RectifiedFlowVideoDiT, VideoDiTConfig};
    let m = RectifiedFlowVideoDiT::new(VideoDiTConfig::default()).unwrap();
    // `shape()[0]` runs before patchify's rank check: pre-fix this panics.
    let x = Tensor::zeros_f32(&[]).unwrap();
    let t = Tensor::zeros_f32(&[1, 1]).unwrap();
    let c = Tensor::zeros_f32(&[1, 2, 32]).unwrap();
    assert!(m.forward(&x, &t, &c).is_err());
}

#[test]
fn am8b_dit_sample_scalar_noise_is_err_not_panic() {
    use oxideon::tensor::Tensor;
    use oxideon::video::flow_dit::{RectifiedFlowVideoDiT, VideoDiTConfig};
    let m = RectifiedFlowVideoDiT::new(VideoDiTConfig::default()).unwrap();
    // Same shape-first pattern in sample_euler: pre-fix this panics.
    let x = Tensor::zeros_f32(&[]).unwrap();
    let c = Tensor::zeros_f32(&[1, 2, 32]).unwrap();
    assert!(m.sample_euler(&x, &c, 2).is_err());
}

#[test]
fn am4_causal_conv3d_zero_channels_is_err_not_silent_nan() {
    use oxideon::video::causal_conv3d::CausalConv3d;
    // fan = 0 makes k = +inf; the bias samples U(-inf, +inf) = NaN.
    // Pre-fix this returns Ok with a NaN bias (same shape as ah2).
    assert!(CausalConv3d::new(0, 2, (3, 3, 3), (1, 1, 1)).is_err());
    assert!(CausalConv3d::new(2, 2, (0, 3, 3), (1, 1, 1)).is_err());
}

#[test]
fn am2_audio_encoder_swapped_conv_weight_is_err_not_panic() {
    use oxideon::audio::whisper::{AudioEncoder, WhisperConfig};
    use oxideon::tensor::Tensor;
    let cfg = WhisperConfig {
        n_mels: 4,
        d_model: 8,
        encoder_layers: 1,
        encoder_heads: 2,
        decoder_layers: 1,
        decoder_heads: 2,
        d_ff: 16,
        vocab_size: 32,
        max_source_positions: 8,
        max_target_positions: 8,
    };
    let mut enc = AudioEncoder::new(&cfg).unwrap();
    // `conv1.weight` is pub: a 1-D swap panics `shape()[1]` pre-fix.
    enc.conv1.weight = Tensor::zeros_f32(&[5]).unwrap();
    let mel = Tensor::zeros_f32(&[1, 4, 16]).unwrap();
    assert!(enc.forward(&mel).is_err());
}

#[test]
fn am1_quantize_vec_truncated_is_err_not_panic() {
    use oxideon::audio::VQCodebookStage;
    // `embeddings` is pub: pre-fix a truncated table panicked with index OOB
    // (demonstrated pre-fix at rvq.rs:76); an emptied codebook sliced OOB too.
    let mut st = VQCodebookStage::new(4, 4).unwrap();
    st.embeddings.truncate(3);
    assert!(st.quantize_vec(&[0.0f32; 4]).is_err());
    st.embeddings.clear();
    st.codebook_size = 0;
    assert!(st.quantize_vec(&[0.0f32; 4]).is_err());
    let ok = VQCodebookStage::new(4, 4).unwrap();
    assert!(ok.quantize_vec(&[0.0f32; 4]).is_ok());
}

#[test]
fn al1_tensor_bytes_huge_numel_is_err_not_panic() {
    use oxideon::serialize::gguf::GGMLType;
    // `num_blocks * block_bytes` on a bare numel: pre-fix this panics
    // ("attempt to multiply with overflow", demonstrated pre-fix).
    assert!(GGMLType::F64.tensor_bytes(usize::MAX).is_err());
    assert_eq!(GGMLType::F32.tensor_bytes(10).unwrap(), 40);
}

#[test]
fn al2_gguf_hostile_kv_count_is_err_not_abort() {
    use oxideon::serialize::gguf::GGUFFile;
    // 5M claimed entries in a 5MB file: pre-fix `with_capacity(5M)` on the
    // metadata map reserves ~160MB up front (abort-class under `ulimit -v`,
    // demonstrated pre-fix with containment); post-fix this is a fast Err.
    let mut b = Vec::new();
    b.extend_from_slice(&0x4655_4747u32.to_le_bytes());
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    b.extend_from_slice(&5_000_000u64.to_le_bytes()); // metadata_kv_count
    b.resize(5_000_000, 0);
    assert!(GGUFFile::from_bytes(&b).is_err());
}

#[test]
fn al4_gguf_big_bool_array_parses() {
    use oxideon::serialize::gguf::{GGUFFile, GGUFValue};
    // Locks valid-big-array behavior (2M bools): the `with_capacity(count)`
    // below the count-vs-remainder check is fallible post-fix (same abort
    // class as al2, fixed in the same pass).
    let mut b = Vec::new();
    b.extend_from_slice(&0x4655_4747u32.to_le_bytes());
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes()); // one KV pair
    b.extend_from_slice(&1u64.to_le_bytes());
    b.push(b'a'); // key
    b.extend_from_slice(&9u32.to_le_bytes()); // array type
    b.extend_from_slice(&7u32.to_le_bytes()); // bool items
    b.extend_from_slice(&2_000_000u64.to_le_bytes()); // count
    b.resize(b.len() + 2_000_000, 0); // 2M false bytes
    while b.len() % 32 != 0 {
        b.push(0); // data-section alignment pad
    }
    let f = GGUFFile::from_bytes(&b).unwrap();
    match f.metadata.get("a") {
        Some(GGUFValue::Array(v)) => assert_eq!(v.len(), 2_000_000),
        other => panic!("expected 2M-bool array, got {other:?}"),
    }
}

#[test]
fn al3_gguf_q3k_dequant_does_not_panic() {
    use oxideon::serialize::gguf::GGUFFile;
    // The hmask selector `m <<= 1` runs one dead extra shift (128 << 1):
    // pre-fix this panics with "attempt to shift left with overflow" in debug.
    let bytes = al_gguf_one_kblock(11, 110); // Q3_K
    let f = GGUFFile::from_bytes(&bytes).unwrap();
    let t = f.dequantize_tensor("w").unwrap();
    assert_eq!(t.numel(), 256);
}

#[test]
fn al3b_gguf_q5k_dequant_does_not_panic() {
    use oxideon::serialize::gguf::GGUFFile;
    // Same dead-shift shape: `u1 <<= 2` ends at 64 << 2 (and u2 at 128 << 2).
    let bytes = al_gguf_one_kblock(13, 176); // Q5_K
    let f = GGUFFile::from_bytes(&bytes).unwrap();
    let t = f.dequantize_tensor("w").unwrap();
    assert_eq!(t.numel(), 256);
}

#[test]
fn ak1_router_swapped_zero_in_gate_is_err_not_divzero() {
    use oxideon::nn::Linear;
    use oxideon::nn::deepseek_moe::AuxFreeRouter;
    use oxideon::tensor::Tensor;
    let mut r = AuxFreeRouter::new(8, 4, 2, 0.01).unwrap();
    // Only the gate *output* width is re-validated: a zero-input-width gate
    // passes the check, forwards an empty batch fine, then divides 0/0 below.
    r.gate = Linear::with_bias(0, 4, false).unwrap();
    let x = Tensor::zeros_f32(&[2, 0]).unwrap();
    assert!(r.route(&x).is_err());
}

#[test]
fn ak3_moe_zero_d_model_is_err_not_divzero() {
    use oxideon::nn::deepseek_moe::DeepSeekV3MoE;
    use oxideon::tensor::Tensor;
    let mut m = DeepSeekV3MoE::new(8, 16, 1, 4, 2, 0.01).unwrap();
    // `d_model` is pub: zeroing it admits an empty-feature input whose
    // `numel / d` divides 0/0 below.
    m.d_model = 0;
    let x = Tensor::zeros_f32(&[1, 2, 0]).unwrap();
    assert!(m.forward_and_update(&x).is_err());
}

#[test]
fn ai2_bilinear_tampered_out_features_is_err_not_panic() {
    use oxideon::nn::activations::Bilinear;
    use oxideon::tensor::Tensor;
    let mut m = Bilinear::new(2, 2, 2, false).unwrap();
    // `out_features` is pub and unchecked before `out * in2` below: pre-fix
    // this overflows (debug panic) instead of returning Err.
    m.out_features = usize::MAX / 2 + 1;
    let x1 = Tensor::zeros_f32(&[1, 2]).unwrap();
    let x2 = Tensor::zeros_f32(&[1, 2]).unwrap();
    assert!(m.forward_pair(&x1, &x2).is_err());
}

#[test]
fn ai3_spectral_norm_poisoned_lock_is_recovered_not_panic() {
    use oxideon::nn::activations::SpectralNormLinear;
    use std::sync::Arc;
    let m = Arc::new(SpectralNormLinear::new(4, 4, false, 2).unwrap());
    // Poison `u_vec` from another thread (panic while holding the write guard).
    let victim = Arc::clone(&m);
    let _ = std::thread::spawn(move || {
        let _g = victim.u_vec.write().unwrap();
        panic!("intentional poison for ai3");
    })
    .join();
    // Pre-fix the `.read().unwrap()` below panics; post-fix it recovers.
    assert!(m.spectral_weight().is_ok());
}

#[test]
fn ai4_spectral_norm_tampered_iters_is_err_not_hang() {
    use oxideon::nn::activations::SpectralNormLinear;
    let mut m = SpectralNormLinear::new(4, 4, false, 2).unwrap();
    // `n_power_iterations` is pub with only a constructor `.max(1)` floor and no
    // ceiling: pre-fix this loops ~forever (demonstrated with a solo `timeout`
    // kill, never run unbounded pre-fix).
    m.n_power_iterations = usize::MAX;
    assert!(m.spectral_weight().is_err());
}

#[test]
fn ah1_xavier_uniform_huge_fans_is_err_not_panic() {
    use oxideon::nn::init;
    // `fan_in + fan_out` on bare usizes: pre-fix this wraps (release) or panics
    // on overflow (debug) instead of returning Err.
    assert!(init::xavier_uniform(&[2, 2], usize::MAX, 1).is_err());
}

#[test]
fn ah1b_xavier_normal_huge_fans_is_err_not_panic() {
    use oxideon::nn::init;
    assert!(init::xavier_normal(&[2, 2], usize::MAX, usize::MAX).is_err());
}

#[test]
fn ah2_conv_transpose3d_zero_in_channels_is_err_not_silent_nan() {
    use oxideon::backend::Device;
    use oxideon::dtype::DType;
    use oxideon::nn::conv3d::ConvTranspose3d;
    // fan_in = 0 makes bound = +inf; the bias samples U(-inf, +inf) = NaN.
    // Pre-fix this returns Ok with a NaN bias.
    let r = ConvTranspose3d::new(
        0,
        2,
        (1, 1, 1),
        (1, 1, 1),
        (0, 0, 0),
        (0, 0, 0),
        true,
        DType::F32,
        Device::Cpu,
    );
    assert!(r.is_err());
}

#[test]
fn ag2_conv3d_backward_smoke() {
    use oxideon::tensor::Tensor;
    // Locks the conv3d backward closure (fallible grad-buffer allocation).
    let x = Tensor::ones_f32(&[1, 1, 2, 2, 2])
        .unwrap()
        .requires_grad_(true);
    let w = Tensor::ones_f32(&[1, 1, 1, 1, 1])
        .unwrap()
        .requires_grad_(true);
    let y = x.conv3d(&w, None, (1, 1, 1), (0, 0, 0)).unwrap();
    y.sum_axes(&[0, 1, 2, 3, 4], false)
        .unwrap()
        .backward()
        .unwrap();
    assert!(x.grad().is_some() && w.grad().is_some());
}

#[test]
fn ag2b_conv_transpose3d_backward_smoke() {
    use oxideon::tensor::Tensor;
    // Locks the conv_transpose3d backward closure (fallible grad-buffer allocation).
    let x = Tensor::ones_f32(&[1, 1, 2, 2, 2])
        .unwrap()
        .requires_grad_(true);
    let w = Tensor::ones_f32(&[1, 1, 1, 1, 1])
        .unwrap()
        .requires_grad_(true);
    let y = x
        .conv_transpose3d(&w, None, (1, 1, 1), (0, 0, 0), (0, 0, 0), (1, 1, 1))
        .unwrap();
    y.sum_axes(&[0, 1, 2, 3, 4], false)
        .unwrap()
        .backward()
        .unwrap();
    assert!(x.grad().is_some() && w.grad().is_some());
}

#[test]
fn ag1_deq_tampered_forward_iters_is_err_not_hang() {
    use oxideon::nn::deq::{DeepEquilibriumModel, DeqConfig};
    use oxideon::tensor::Tensor;
    let mut m = DeepEquilibriumModel::new(4, 4, DeqConfig::default()).unwrap();
    // `config` is a pub field: `new` caps it, post-construction tampering bypasses
    // the constructor cap. Pre-fix this loops ~forever (demonstrated with a solo
    // `timeout` kill, never run unbounded pre-fix).
    m.config.max_forward_iters = usize::MAX;
    // Force a stable 2-cycle (relaxed Picard can never converge): strong negative
    // diagonal feedback plus a bias that keeps z0=0 away from the fixed point.
    m.weight_z = vec![0.0; 16];
    for i in 0..4 {
        m.weight_z[i * 4 + i] = -100.0;
    }
    m.bias = vec![5.0; 4];
    let x = Tensor::zeros_f32(&[4]).unwrap();
    assert!(m.forward_equilibrium(&x).is_err());
}

#[test]
fn ag1b_deq_tampered_backward_iters_is_err_not_hang() {
    use oxideon::nn::deq::{DeepEquilibriumModel, DeqConfig};
    let mut m = DeepEquilibriumModel::new(4, 4, DeqConfig::default()).unwrap();
    m.config.max_backward_iters = usize::MAX;
    // Spectral radius 2 at the origin (dtanh=1 there): the Neumann iterates double
    // every step, overflow to inf, and `inf < tol` never holds.
    m.weight_z = vec![0.0; 16];
    for i in 0..4 {
        m.weight_z[i * 4 + i] = 2.0;
    }
    assert!(m.solve_adjoint(&[0.1; 4], &[0.0; 4], &[0.0; 4]).is_err());
}

#[test]
fn af2_cartesian_grid_absurd_resolution_is_err_not_abort() {
    use oxideon::tuning::ParamSpace;
    let space = ParamSpace::new().add_uniform("lr", 1e-5, 1e-1);
    // 2^40 grid points: pre-fix materializes them while *counting* (abort-class;
    // demonstrated under `ulimit -v` containment, never run unbounded pre-fix).
    assert!(space.cartesian_grid(1usize << 40).is_err());
}

#[test]
fn aa2_backward_with_grad_longer_grad_is_err_not_silent() {
    use oxideon::fused_kernels::fused_geglu;
    let g = Tensor::from_slice(&[1f32, 2., 3., 4.], &[2, 2])
        .unwrap()
        .requires_grad_(true);
    let u = Tensor::from_slice(&[1f32, 0., 0., 1.], &[2, 2])
        .unwrap()
        .requires_grad_(true);
    let y = fused_geglu(&g, &u).unwrap();
    let long = Tensor::from_slice(&[1f32, 1., 1., 1., 1., 1.], &[2, 3]).unwrap();
    assert!(y.backward_with_grad(&long, false).is_err()); // pre-fix: Ok(()) with silently wrong grads
}
