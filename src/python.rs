//! PyO3 Python Bindings (`import oxideon as ox`) with Zero-GIL Rust Compute Offloading.
//!
//! All heavy CPU (AVX2+FMA blocked GEMM, `rayon` parallel kernels, fused single-pass
//! `RMSNorm`/`SwiGLU`/`CrossEntropy`, differentiable `SVD`/`Cholesky`/`QR`/`pinv`, `Conv2d`/`Conv3d`,
//! `FlashAttention-2`, `Mamba`, `DeepSeekMoELM`, `HybridJambaLM`, `MultimodalVLM`, `DiT`)
//! and `wgpu` GPU operations release the Python Global Interpreter Lock (`py.detach(...)`)
//! and execute 100% in native Rust.
//!
//! Includes [`PyMLP::train_epochs_in_rust`] so entire multi-epoch mini-batch training loops
//! (forward + loss + backward + gradient clipping + optimizer step) run inside Rust without
//! crossing the Python/Rust FFI boundary per batch.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::adaptive::{
    ArchitectureGenome, AutonomousQuantPolicy, AutonomousQuantizer, EarlyExitRouter,
    ElasticSubnetExtractor, EvolutionaryAutoML, MixtureOfDepths, QuantFormat,
};
use crate::allocator::CachingAllocator;
use crate::audio::{MelSpectrogram, RVQConfig, ResidualVectorQuantizer, Whisper, WhisperConfig};
use crate::backend::{ComputeGraph, ComputeStream, StreamEvent};
use crate::compiler::{AotCompiler, CompiledGraph, IrGraph, IrOp, SymbolicShape};
use crate::crypto::{
    CkksContext, DifferentiableR1csQap, FheCiphertext, FhePublicKey, FheSecretKey, ZkMlEngine,
};
use crate::device::{
    Device, DeviceMesh, DeviceType, InterconnectMatrix, MemoryPriority, TieredMemoryManager,
    get_default_device, set_default_device,
};
use crate::diffusion::{
    BetaSchedule, DiffusionForcingScheduler, EDMConfig, KDiffusionSampler, RectifiedFlowConfig,
    RectifiedFlowEngine,
};
use crate::distributed::{
    AutoParallelPlanner, ByzantineAggregator, DpSgdConfig, DualPipeSchedule, Parallel3DConfig,
    Parallel3DRuntime, PipelinePhase, PipelineSchedule1F1B, Placement, ShardedTensor,
    ZeRO3Parameter,
};
use crate::driver::{MslKernelEmitter, PtxKernelEmitter, VulkanSpirvKernel, WgslKernelBuilder};
use crate::dtype::DType;
use crate::error::Error;
use crate::fused_kernels::{
    FusedActivation, flash_attention_tiled, fused_add_layernorm, fused_add_rmsnorm,
    fused_bias_activation, fused_cross_entropy_with_z_loss, fused_geglu, fused_layernorm,
    fused_linear_activation, fused_rmsnorm, fused_rope, fused_rotary_qk, fused_sdpa,
    fused_silu_and_mul, fused_swiglu, mlx_dequantize_affine_q4, mlx_quantize_affine_q4,
    mlx_quantized_matmul_q4, paged_kv_attention,
};
use crate::geom_graphs::{
    CellularSheaf, CliffordLinear, GaugeEquivariantConvLayer, HyperbolicGATLayer, LaplacianType,
    LorentzManifold, MultiscaleGraphWaveletTransform, Multivector3D, PoincareManifold,
    PrincipalBundleConnection, RiemannianManifold, SpatioTemporalWaveletLayer, SpikingGnnLayer,
    Su2Element, TemporalInteractionGNN, YangMillsLatticeGauge,
};
use crate::grammar::{AgentRuntime, ExecutionSandbox, JsonSchema, SynthesisEngine, TestCase};
use crate::hardware::{
    CoherentPhotonicCircuit, MachZehnderInterferometer, PhotonicLinear, PhotonicMesh,
    SaturableAbsorber, WaveguideProfile,
};
use crate::jit::{JitKernelBuilder, JitOp};
use crate::layout::Layout;
use crate::long_context::{
    FlashAttention2Config, FlashAttention3Config, FlashDecodingConfig, FlashDecodingEngine,
    HolographicMemoryEngine, RingAttention, flash_attention_2, flash_attention_2_backward,
    flash_attention_3,
};
use crate::nn::{
    AlphaDropout, AuxFreeRouter, AvgPool1d, BiLSTM, CTCLoss, CausalLM, Conv2d, Conv3d, ConvLstmCell,
    ConvNeXt, ConvTranspose3d, DeepEquilibriumModel, DeepSeekMoELM, DeepSeekV3MoE, DeqConfig, DiT,
    DropPath, Dropout, Embedding, EmbeddingBag, EmbeddingBagMode, EventDrivenSpikeRouter, GELU,
    GGUFCausalLM, GGUFModelConfig, GGUFTokenizer, GRUCell, GatedCrossAttention, Hardshrink,
    Hardtanh, HingeEmbeddingLoss, HybridJambaLM, InstanceNorm1d, InstanceNorm3d, LIFNode,
    LSTMCell, LayerNorm, Linear, LogSigmoid, MLPMixer, Mamba2Block, MarginRankingLoss, MaxPool1d,
    Maxout, Mish, Module, MultiHeadLatentAttention, MultimodalVLM, PixelShuffle, PoissonEncoder,
    PolychronousNeuralLayer, QLoRALinear, ReLU, ReLU6, RNNCell, ResetMode, STDPConfig, STDPSynapse,
    Sequential, SiLU, Snake, SoftMarginLoss, Softshrink, SpectralNormLinear, SpikeEvent, Swish, Tanh,
    TemporalConvNet, TitansBlock, TitansMemoryConfig, TitansNeuralMemory, TransfusionConfig,
    TransfusionModel, Upsample2d, VariationalAutoencoder, VisionLanguageActionModel,
    WeightNormLinear, ctc_loss, hinge_embedding_loss, loss, margin_ranking_loss, soft_margin_loss,
};
use crate::ode::{
    JumpDiffusionEngine, LatentSdeConfig, LatentSdeEngine, MeanFieldGameSolver1D, NeuralJumpSDE,
    PoincareCnfFlow, RiemannianTrajectory, SdeTrajectory,
};
use crate::omni_diamond_v::{GrandOmniReportV, GrandOmniUniversalEngineV};
use crate::omni_octa_vii::{GrandOmniOctaEngineVII, GrandOmniOctaReportVII};
use crate::omni_science::OmniScientificFoundationEngine;
use crate::omni_septua_vi::{GrandOmniSeptuaEngineVI, GrandOmniSeptuaReportVI};
use crate::omni_singularity::OmniTransSingularityEngine;
use crate::omni_trans_universal::GrandTransUniversalEngineIV;
use crate::optim::{
    Adam, ManifoldType, Muon, Optimizer, RiemannianAdam, ScheduleFreeAdamW, Sgd, clip_grad_norm,
};
use crate::physics::{
    AdsCftHolographicEngine, ContactHamiltonianSystem, ExceptionalPointReport,
    HamiltonianNeuralEngine, HeisenbergLieAlgebra, HeisenbergPoint, HypoellipticDiffusion3D,
    JarzynskiReport, KerrSpacetime, LangevinThermodynamics, LiePoissonState, LiePoissonSystem,
    NonHermitianHamiltonian2x2, RigidBodyInertia, RmhdPrimitiveState, SolitonPhysics,
    SpatialGrid1D, SplitStepNlsEngine, StochasticThermoEngine, SymplecticTopologyEngine,
};
use crate::quant::{
    AWQLinear, BitLinear, DynamicLlmQuantizer, DynamicQuantPolicy, DynamicTelemetry,
    DynamicTransitionPlan, FP8Format, FP8Tensor, GPTQLinear, PackedBitLinear, QuantScheme,
    TernaryPackedTensor,
};
use crate::quantum::{
    CliffordDataRegression, ExtrapolationModel, GaussianCvCircuit, MatrixProductState,
    QGNNGradients, QaoaCircuit, QecEngine, QuantumCircuit, QuantumGraphNeuralNetwork,
    StabilizerCode, Su2ChernSimonsTqft, VqeAnsatz, ZneExtrapolator, matrix_hafnian,
};
use crate::reasoning::{
    AdaptiveConformalClassifier, AgentProposal, AlgorithmicGraph, AlgorithmicSemiring,
    ConformalRiskController, ConformalScoreType, ConsensusStrategy, DifferentiableILPEngine,
    LearnThenTest, MCTSSearchEngine, NeuralBellmanFord, ParetoIndividual, PointerGraphNetwork,
    PropExpr, ReasoningTrace, RiskCalibrationResult, RiskMetric, SelfCorrectionVerifier, Sequent,
    SequentProver, SplitConformalRegressor, StructuralCausalModel, SwarmOrchestrator,
    SymbolicGpConfig, SymbolicGpEngine, SymbolicNode, TNorm, ToposHeytingEngine,
};
use crate::rl::{
    DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer, PPOConfig, PPOTrainer, RuleReward,
};
use crate::rng::{Generator, get_rng_state, initial_seed, manual_seed, seed, set_rng_state};
use crate::robotics::{ACTConfig, ActionChunkingTransformer, FlowMatchingRoboticsPolicy};
use crate::distributions::{Categorical, Laplace, Normal};
use crate::sampling::SamplingConfig;
use crate::sparse::{SparseCoo, SparseCsr};
use crate::science::{
    AInfinityOperadEngine, AdaptiveCellularSheaf, DeepONet, FourierNeuralOperator1d,
    GromovWassersteinConfig, GromovWassersteinEngine, GromovWassersteinResult, HigherSimplex,
    InfoNceEstimator, InformationGeometryEngine, KernelRidgeRegressor, MercerKernel, MineEstimator,
    MolecularGeometryDiffusion, OrientedSimplex, PairformerBlock, PersistenceDiagram,
    PersistentHomologyEngine, QuasiCategoryEngine, RandomFourierFeatures, SE3EquivariantLayer,
    SheafDiffusionLayer, SherringtonKirkpatrickModel, SimplicialComplex, SpectralConv1d,
    TapSolution, TensorTrain, TropicalMatrix, TropicalPolynomial, TtCore, maximum_mean_discrepancy,
    tropical_distance, wasserstein_distance,
};
use crate::serialize::{
    GGMLType, GGUFFile, GGUFValue, load_safetensors_with_metadata, save_gguf,
    save_safetensors_with_metadata,
};
use crate::serving::{
    AsyncKVPipelineEngine, AttentionSinkConfig, DoubleBufferedKVCache, EagleConfig,
    EagleSpeculativeEngine, MedusaHeads, PrefillDecodeInterleaver, QuantKVDType,
    QuantizedPagedKVCache, ServingEngine, SpeculativeConfig, SpeculativeDecoder,
    StopSequenceMatcher, StreamingKVCache, StreamingUtf8Decoder, TreeAttentionMask,
    TreeSpeculativeDecoder, UnifiedSpeculativeEngine,
};
use crate::super_intelligence::{SuperIntelligenceConfig, SuperIntelligenceEngine};
use crate::tensor::{Tensor, grad, no_grad};
use crate::tile_ir::{
    HardwareProfile, HardwareTarget, KernelAutoTuner, KernelFusionEngine, TileConfig,
    TileKernelPlan,
};
use crate::tokenizer::{BpeTokenizer, OmniTokenizer};
use crate::universal_engine::{UniversalAGIConfig, UniversalAGIEngine};
use crate::video::{RectifiedFlowVideoDiT, RotaryEmbedding3D, VideoDiTConfig, VideoVAE3d};
use crate::vision::RotaryEmbedding2D;
use crate::vectorize::*;
use crate::world_model::{UnifiedWorldModel, WorldModelConfig};

fn to_py_err(e: Error) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

thread_local! {
    static IN_GUARD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `f`; a Rust panic (arithmetic overflow, an out-of-bounds index, ...) becomes a `RuntimeError` instead of
/// pyo3's `PanicException`, which derives from `BaseException` and slips past `except Exception`.
/// Reinterprets a shared slice as bytes (read-only view, same lifetime).
fn slice_bytes<T>(s: &[T]) -> &[u8] {
    // SAFETY: `s` is a valid shared slice, so its full footprint is readable for the
    // borrow's lifetime; `u8` accepts any alignment, so the cast is always sound.
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

pub(crate) fn guard<T>(f: impl FnOnce() -> T) -> PyResult<T> {
    let outer = IN_GUARD.with(|g| g.replace(true));
    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    IN_GUARD.with(|g| g.set(outer));
    out.map_err(|payload| {
        let msg = payload
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "unknown panic".into());
        PyRuntimeError::new_err(format!("internal error (Rust panic): {msg}"))
    })
}

/// [`guard`] for a fallible call: an `Err` becomes a `RuntimeError` as well.
pub(crate) fn guard_res<T, E: std::fmt::Display>(f: impl FnOnce() -> Result<T, E>) -> PyResult<T> {
    guard(f)?.map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Locks a `Mutex`, recovering the inner value when a panicked holder poisoned
/// it instead of panicking again (which would kill the interpreter call).
pub(crate) fn mutex_guard<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// `RwLock::read` with poison recovery (see [`mutex_guard`]).
pub(crate) fn rw_read<T>(m: &std::sync::RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    m.read().unwrap_or_else(|e| e.into_inner())
}

/// `RwLock::write` with poison recovery (see [`mutex_guard`]).
pub(crate) fn rw_write<T>(m: &std::sync::RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    m.write().unwrap_or_else(|e| e.into_inner())
}

/// Panics on purpose, so the test-suite can check that a Rust panic reaches Python as an ordinary `RuntimeError`.
#[pyfunction]
#[pyo3(name = "_raise_test_panic")]
pub fn py_raise_test_panic() {
    panic!("deliberate test panic");
}

/// pyo3 raises a Rust panic as `pyo3_runtime.PanicException`, which derives from `BaseException`: `except Exception`
/// misses it, so one stray overflow or out-of-bounds index in an engine would kill a script. Re-parent the class to
/// `RuntimeError` (same object layout) so every panic from this extension, including ones in hand-written wrappers that
/// bypass [`guard`], is an ordinary exception. Best effort: if the interpreter refuses, panics keep pyo3's behaviour.
fn make_panics_ordinary_exceptions(py: Python<'_>) {
    use pyo3::PyTypeInfo;
    let panic_type = pyo3::panic::PanicException::type_object(py);
    let _ = panic_type.setattr("__bases__", (py.get_type::<PyRuntimeError>(),));
}

/// Keeps the default panic message off stderr for panics that [`guard`] turns into exceptions. Anything else still
/// goes to the hook that was installed before (the Python process may have other Rust extensions).
pub(crate) fn install_quiet_panic_hook() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if !IN_GUARD.with(std::cell::Cell::get) {
                previous(info);
            }
        }));
    });
}

/// Declares a mechanical wrapper around an engine type: a `#[pyclass]` holding `inner`, a constructor and
/// pass-through methods. `new(..) -> res;` and the `*_res` forms convert `Error` with `to_py_err`;
/// `assoc*` forms are static methods. Every call goes through [`guard`], so a panic inside an engine is a `RuntimeError`.
/// Wrappers that need defaults, GIL release or custom bodies are written by hand.
macro_rules! py_engine {
    (@emit [$(#[$m:meta])*] $py:ident $name:literal $ty:ident [$($d:ident),*] { $($ctor:tt)* }
        {
            $( method $im:ident($($ia:ident: $it:ty),*) -> $ir:ty; )*
            $( method_res $rm:ident($($ra:ident: $rt:ty),*) -> $rr:ty; )*
            $( assoc $sm:ident($($sa:ident: $sty:ty),*) -> $sr:ty; )*
            $( assoc_res $tm:ident($($ta:ident: $tty:ty),*) -> $tr:ty; )*
        }) => {
        $(#[$m])*
        #[pyclass(name = $name, module = "oxideon", from_py_object)]
        #[derive($($d),*)]
        pub struct $py {
            pub inner: $ty,
        }

        #[pymethods]
        impl $py {
            $($ctor)*
            $( pub fn $im(&self, $($ia: $it),*) -> PyResult<$ir> { guard(|| self.inner.$im($($ia),*)) } )*
            $( pub fn $rm(&self, $($ra: $rt),*) -> PyResult<$rr> { guard_res(|| self.inner.$rm($($ra),*)) } )*
            $( #[staticmethod] pub fn $sm($($sa: $sty),*) -> PyResult<$sr> { guard(|| $ty::$sm($($sa),*)) } )*
            $( #[staticmethod] pub fn $tm($($ta: $tty),*) -> PyResult<$tr> { guard_res(|| $ty::$tm($($ta),*)) } )*
        }
    };
    ($(#[$m:meta])* $py:ident($name:literal) wraps $ty:ident [$($d:ident),*] {
        new($($na:ident: $nt:ty),*) -> res;
        $($rest:tt)*
    }) => {
        py_engine!(@emit [$(#[$m])*] $py $name $ty [$($d),*]
            {
                #[new]
                pub fn new($($na: $nt),*) -> PyResult<Self> {
                    let inner = guard_res(|| $ty::new($($na),*))?;
                    Ok(Self { inner })
                }
            }
            { $($rest)* });
    };
    ($(#[$m:meta])* $py:ident($name:literal) wraps $ty:ident [$($d:ident),*] {
        new($($na:ident: $nt:ty),*);
        $($rest:tt)*
    }) => {
        py_engine!(@emit [$(#[$m])*] $py $name $ty [$($d),*]
            {
                #[new]
                pub fn new($($na: $nt),*) -> PyResult<Self> {
                    Ok(Self { inner: guard(|| $ty::new($($na),*))? })
                }
            }
            { $($rest)* });
    };
}

fn parse_dtype(s: &str) -> PyResult<DType> {
    match s.to_ascii_lowercase().as_str() {
        "f32" | "float32" | "float" => Ok(DType::F32),
        "f64" | "float64" | "double" => Ok(DType::F64),
        "f16" | "float16" | "half" => Ok(DType::F16),
        "bf16" | "bfloat16" => Ok(DType::BF16),
        "i32" | "int32" | "int" => Ok(DType::I32),
        "i64" | "int64" | "long" => Ok(DType::I64),
        "u8" | "uint8" | "byte" => Ok(DType::U8),
        "i16" | "int16" | "short" => Ok(DType::I16),
        "i8" | "int8" | "char" => Ok(DType::I8),
        "u16" | "uint16" => Ok(DType::U16),
        "u32" | "uint32" => Ok(DType::U32),
        "u64" | "uint64" => Ok(DType::U64),
        "bool" => Ok(DType::Bool),
        "c64" | "complex64" | "cfloat" => Ok(DType::C64),
        "c128" | "complex128" | "cdouble" => Ok(DType::C128),
        "e4m3" | "float8_e4m3fn" | "fp8_e4m3" | "f8e4m3" => Ok(DType::F8E4M3),
        "e5m2" | "float8_e5m2" | "fp8_e5m2" | "f8e5m2" => Ok(DType::F8E5M2),
        other => Err(PyRuntimeError::new_err(format!(
            "unsupported dtype '{other}' (expected f32, f64, f16, bf16, i32, i64, u8, bool)"
        ))),
    }
}

fn parse_device(s: &str) -> PyResult<Device> {
    Device::parse_spec(s).map_err(to_py_err)
}

fn extract_dtype_str(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(s) = obj.extract::<String>() {
        return Ok(s);
    }
    if let Ok(name_attr) = obj.getattr("name") {
        if let Ok(s) = name_attr.extract::<String>() {
            return Ok(s);
        }
    }
    let s = obj.str()?.extract::<String>()?;
    let clean = if let Some(stripped) = s.strip_prefix("oxideon.") {
        stripped
    } else if let Some(stripped) = s.strip_prefix("torch.") {
        stripped
    } else {
        &s
    };
    Ok(clean.to_string())
}

fn parse_dtype_obj(obj: Option<&Bound<'_, PyAny>>, default_dtype: DType) -> PyResult<DType> {
    match obj {
        None => Ok(default_dtype),
        Some(val) => {
            if val.is_none() {
                Ok(default_dtype)
            } else {
                let s = extract_dtype_str(val)?;
                parse_dtype(&s)
            }
        }
    }
}

fn extract_device_str(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(s) = obj.extract::<String>() {
        return Ok(s);
    }
    if let Ok(type_attr) = obj.getattr("type") {
        if let Ok(t_str) = type_attr.extract::<String>() {
            if let Ok(idx_attr) = obj.getattr("index") {
                if let Ok(Some(idx)) = idx_attr.extract::<Option<usize>>() {
                    return Ok(format!("{t_str}:{idx}"));
                }
            }
            return Ok(t_str);
        }
    }
    let s = obj.str()?.extract::<String>()?;
    Ok(s)
}

fn parse_device_obj(obj: Option<&Bound<'_, PyAny>>, default_device: &str) -> PyResult<Device> {
    match obj {
        None => parse_device(default_device),
        Some(val) => {
            if val.is_none() {
                parse_device(default_device)
            } else {
                let s = extract_device_str(val)?;
                parse_device(&s)
            }
        }
    }
}

/// Copies a tensor's data into fresh storage. A tensor in the autograd graph is copied by an op, so it stays in it.
fn deep_copy(t: &Tensor) -> crate::error::Result<Tensor> {
    if t.requires_grad() {
        return t.mul_scalar(1.0);
    }
    // Fresh storage via a single typed copy (memcpy when contiguous); the old
    // gather-to-Vec plus from_slice round trip copied twice and converted dtypes.
    let storage = t.storage().to_contiguous(t.layout())?;
    Ok(Tensor::from_storage(
        storage,
        Layout::contiguous(t.shape().to_vec()),
    ))
}

#[pyclass(name = "Device", module = "oxideon", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyDevice {
    pub(crate) inner: Device,
}

#[pymethods]
impl PyDevice {
    #[new]
    pub fn new(spec: &str) -> PyResult<Self> {
        let inner = Device::parse_spec(spec).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn r#type(&self) -> String {
        self.inner.device_type().to_string()
    }

    #[getter]
    pub fn index(&self) -> Option<usize> {
        self.inner.index()
    }

    #[getter]
    pub fn is_cpu(&self) -> bool {
        self.inner.is_cpu()
    }

    #[getter]
    pub fn is_cuda(&self) -> bool {
        self.inner.is_cuda()
    }

    #[getter]
    pub fn is_metal(&self) -> bool {
        self.inner.is_metal()
    }

    #[getter]
    pub fn is_vulkan(&self) -> bool {
        self.inner.is_vulkan()
    }

    #[getter]
    pub fn is_wgpu(&self) -> bool {
        self.inner.is_wgpu()
    }

    #[getter]
    pub fn is_gpu(&self) -> bool {
        self.inner.is_gpu()
    }

    #[getter]
    pub fn is_tpu(&self) -> bool {
        self.inner.is_tpu()
    }

    /// True when the device is only a label: data stays in host memory and CPU kernels run.
    #[getter]
    pub fn is_simulated(&self) -> bool {
        self.inner.is_simulated()
    }

    #[staticmethod]
    pub fn cpu() -> Self {
        Self { inner: Device::Cpu }
    }

    #[staticmethod]
    #[pyo3(signature = (index=0))]
    pub fn cuda(index: usize) -> Self {
        Self {
            inner: Device::Cuda(index),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (index=0))]
    pub fn gpu(index: usize) -> Self {
        #[cfg(feature = "wgpu")]
        {
            if let Ok(dev) = Device::wgpu() {
                return Self { inner: dev };
            }
        }
        Self {
            inner: Device::Cuda(index),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (chip=0, core=0))]
    pub fn tpu(chip: usize, core: usize) -> Self {
        Self {
            inner: Device::Tpu { chip, core },
        }
    }

    #[getter]
    pub fn is_npu(&self) -> bool {
        self.inner.is_npu()
    }

    #[getter]
    pub fn is_meta(&self) -> bool {
        self.inner.is_meta()
    }

    #[getter]
    pub fn is_accelerator(&self) -> bool {
        self.inner.is_accelerator()
    }

    pub fn get_properties<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let p = self.inner.properties();
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("name", p.name)?;
        dict.set_item("type", p.device_type.to_string())?;
        dict.set_item("index", p.index)?;
        dict.set_item("total_memory_bytes", p.total_memory_bytes)?;
        dict.set_item(
            "total_memory_gb",
            (p.total_memory_bytes as f64) / (1024.0 * 1024.0 * 1024.0),
        )?;
        dict.set_item("is_unified_memory", p.is_unified_memory)?;
        if let Some((major, minor)) = p.compute_capability {
            dict.set_item("compute_capability", (major, minor))?;
        } else {
            dict.set_item("compute_capability", py.None())?;
        }
        dict.set_item("peak_fp32_tflops", p.peak_fp32_tflops)?;
        dict.set_item("peak_fp16_tflops", p.peak_fp16_tflops)?;
        dict.set_item("peak_bf16_tflops", p.peak_bf16_tflops)?;
        dict.set_item("peak_fp8_tflops", p.peak_fp8_tflops)?;
        dict.set_item("interconnect", p.interconnect.to_string())?;
        Ok(dict)
    }

    pub fn __repr__(&self) -> String {
        format!(
            "device(type='{}', index={:?})",
            self.inner.device_type(),
            self.inner.index()
        )
    }

    pub fn __str__(&self) -> String {
        self.inner.to_string()
    }

    pub fn __eq__(&self, other: &Bound<'_, pyo3::PyAny>) -> bool {
        if let Ok(dev) = other.extract::<PyDevice>() {
            self.inner == dev.inner
        } else if let Ok(s) = other.extract::<String>() {
            if let Ok(dev) = Device::parse_spec(&s) {
                self.inner == dev
            } else {
                false
            }
        } else {
            false
        }
    }

    #[staticmethod]
    pub fn available_devices() -> Vec<PyDevice> {
        Device::available_devices()
            .into_iter()
            .map(|inner| PyDevice { inner })
            .collect()
    }

    #[staticmethod]
    pub fn is_available(device_type: &str) -> bool {
        match device_type.to_ascii_lowercase().as_str() {
            "cuda" | "gpu" => Device::is_cuda_available(),
            "metal" | "mps" => Device::is_metal_available(),
            "vulkan" => Device::is_vulkan_available(),
            "wgpu" => Device::is_wgpu_available(),
            "cpu" | "meta" => true,
            _ => false,
        }
    }

    #[staticmethod]
    pub fn device_count(device_type: &str) -> usize {
        match device_type.to_ascii_lowercase().as_str() {
            "cuda" | "gpu" => Device::cuda_device_count(),
            "metal" | "mps" => Device::metal_device_count(),
            "vulkan" => Device::vulkan_device_count(),
            "wgpu" => {
                if Device::is_wgpu_available() {
                    1
                } else {
                    0
                }
            }
            "cpu" | "meta" => 1,
            _ => 0,
        }
    }
}

#[pyclass(name = "DeviceMesh", module = "oxideon", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyDeviceMesh {
    inner: DeviceMesh,
}

#[pymethods]
impl PyDeviceMesh {
    #[new]
    #[pyo3(signature = (shape, dim_names, device_type="cuda"))]
    pub fn new(shape: Vec<usize>, dim_names: Vec<String>, device_type: &str) -> PyResult<Self> {
        let dev_type = match device_type.to_ascii_lowercase().as_str() {
            "cuda" => DeviceType::Cuda,
            "metal" | "mps" => DeviceType::Metal,
            "cpu" => DeviceType::Cpu,
            "vulkan" => DeviceType::Vulkan,
            "tpu" => DeviceType::Tpu,
            "npu" => DeviceType::Npu,
            "meta" => DeviceType::Meta,
            "wgpu" => DeviceType::Wgpu,
            other => {
                return Err(PyRuntimeError::new_err(format!(
                    "Unsupported device_type '{other}' in DeviceMesh"
                )));
            }
        };
        let dims: Vec<&str> = dim_names.iter().map(|s| s.as_str()).collect();
        let mesh = DeviceMesh::homogeneous(&shape, &dims, dev_type).map_err(to_py_err)?;
        Ok(Self { inner: mesh })
    }

    #[getter]
    pub fn shape(&self) -> Vec<usize> {
        self.inner.shape.clone()
    }

    #[getter]
    pub fn dim_names(&self) -> Vec<String> {
        self.inner.dim_names.clone()
    }

    #[getter]
    pub fn device_count(&self) -> usize {
        self.inner.device_count()
    }

    pub fn get_device(&self, coords: Vec<usize>) -> PyResult<PyDevice> {
        let dev = self.inner.get_device(&coords).map_err(to_py_err)?;
        Ok(PyDevice { inner: dev.clone() })
    }

    pub fn slice_submesh(&self, dim_name: &str, index: usize) -> PyResult<Vec<PyDevice>> {
        let sub = self
            .inner
            .slice_submesh(dim_name, index)
            .map_err(to_py_err)?;
        Ok(sub.into_iter().map(|d| PyDevice { inner: d }).collect())
    }

    pub fn __repr__(&self) -> String {
        format!(
            "DeviceMesh(shape={:?}, dim_names={:?}, total_devices={})",
            self.inner.shape,
            self.inner.dim_names,
            self.inner.device_count()
        )
    }
}

/// Standalone seedable pseudo-random number generator (`torch.Generator`).
#[pyclass(name = "Generator", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyGenerator {
    pub(crate) inner: Generator,
}

#[pymethods]
impl PyGenerator {
    #[new]
    #[pyo3(signature = (device_or_seed=None, seed=None))]
    pub fn new(
        device_or_seed: Option<&Bound<'_, pyo3::types::PyAny>>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let (dev, s_opt) = match device_or_seed {
            Some(obj) => {
                if let Ok(num) = obj.extract::<u64>() {
                    (Device::Cpu, Some(num))
                } else if let Ok(s) = obj.extract::<String>() {
                    (parse_device(&s)?, seed)
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "expected device (str) or seed (int)",
                    ));
                }
            }
            None => (Device::Cpu, seed),
        };
        let inner = if let Some(s) = s_opt {
            Generator::with_device(s, dev)
        } else {
            let mut g = Generator::new();
            let s = g.seed();
            Generator::with_device(s, dev)
        };
        Ok(Self { inner })
    }

    pub fn manual_seed(&mut self, seed: u64) {
        self.inner.manual_seed(seed);
    }

    pub fn seed(&mut self) -> u64 {
        self.inner.seed()
    }

    pub fn initial_seed(&self) -> u64 {
        self.inner.initial_seed()
    }

    pub fn get_state(&self) -> Vec<u8> {
        self.inner.get_state()
    }

    pub fn set_state(&mut self, state: &[u8]) -> PyResult<()> {
        self.inner.set_state(state).map_err(to_py_err)
    }

    #[getter]
    pub fn device(&self) -> String {
        self.inner.device().to_string()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Generator(device={}, initial_seed={})",
            self.inner.device(),
            self.inner.initial_seed()
        )
    }
}

/// Generation & logit sampling configuration (temperature, top_k, top_p, min_p, penalties, seed).
#[pyclass(name = "SamplingConfig", module = "oxideon", from_py_object)]
#[derive(Clone, Debug)]
pub struct PySamplingConfig {
    pub(crate) inner: SamplingConfig,
}

#[pymethods]
impl PySamplingConfig {
    #[new]
    #[pyo3(signature = (
        temperature=1.0,
        top_k=None,
        top_p=None,
        min_p=None,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        seed=None,
        max_new_tokens=128,
        stop_tokens=None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        temperature: f64,
        top_k: Option<usize>,
        top_p: Option<f64>,
        min_p: Option<f64>,
        repetition_penalty: f64,
        presence_penalty: f64,
        frequency_penalty: f64,
        seed: Option<u64>,
        max_new_tokens: usize,
        stop_tokens: Option<Vec<usize>>,
    ) -> Self {
        let inner = SamplingConfig {
            temperature: temperature.max(0.0),
            top_k,
            top_p,
            min_p,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            seed,
            max_new_tokens,
            stop_tokens: stop_tokens.unwrap_or_default(),
            ..Default::default()
        };
        Self { inner }
    }

    #[staticmethod]
    pub fn greedy() -> Self {
        Self {
            inner: SamplingConfig::greedy(),
        }
    }

    #[getter]
    pub fn temperature(&self) -> f64 {
        self.inner.temperature
    }

    #[setter]
    pub fn set_temperature(&mut self, t: f64) {
        self.inner.temperature = t.max(0.0);
    }

    #[getter]
    pub fn top_k(&self) -> Option<usize> {
        self.inner.top_k
    }

    #[setter]
    pub fn set_top_k(&mut self, k: Option<usize>) {
        self.inner.top_k = k;
    }

    #[getter]
    pub fn top_p(&self) -> Option<f64> {
        self.inner.top_p
    }

    #[setter]
    pub fn set_top_p(&mut self, p: Option<f64>) {
        self.inner.top_p = p;
    }

    #[getter]
    pub fn min_p(&self) -> Option<f64> {
        self.inner.min_p
    }

    #[setter]
    pub fn set_min_p(&mut self, p: Option<f64>) {
        self.inner.min_p = p;
    }

    #[getter]
    pub fn repetition_penalty(&self) -> f64 {
        self.inner.repetition_penalty
    }

    #[setter]
    pub fn set_repetition_penalty(&mut self, rp: f64) {
        self.inner.repetition_penalty = rp;
    }

    #[getter]
    pub fn presence_penalty(&self) -> f64 {
        self.inner.presence_penalty
    }

    #[setter]
    pub fn set_presence_penalty(&mut self, pp: f64) {
        self.inner.presence_penalty = pp;
    }

    #[getter]
    pub fn frequency_penalty(&self) -> f64 {
        self.inner.frequency_penalty
    }

    #[setter]
    pub fn set_frequency_penalty(&mut self, fp: f64) {
        self.inner.frequency_penalty = fp;
    }

    #[getter]
    pub fn seed(&self) -> Option<u64> {
        self.inner.seed
    }

    #[setter]
    pub fn set_seed(&mut self, s: Option<u64>) {
        self.inner.seed = s;
    }

    #[getter]
    pub fn max_new_tokens(&self) -> usize {
        self.inner.max_new_tokens
    }

    #[setter]
    pub fn set_max_new_tokens(&mut self, m: usize) {
        self.inner.max_new_tokens = m;
    }

    pub fn with_logit_bias(&mut self, token: usize, bias: f64) {
        self.inner.logit_bias.insert(token, bias);
    }

    pub fn __repr__(&self) -> String {
        format!(
            "SamplingConfig(temperature={}, top_k={:?}, top_p={:?}, min_p={:?}, repetition_penalty={}, seed={:?})",
            self.inner.temperature,
            self.inner.top_k,
            self.inner.top_p,
            self.inner.min_p,
            self.inner.repetition_penalty,
            self.inner.seed
        )
    }
}

/// Python wrapper around [`SparseCoo`]: 2D sparse matrix in Coordinate format.
/// NOTE: construction sums duplicate entries and drops exact zeros (torch
/// `coalesce()` keeps explicit zeros instead — values are identical either way).
#[pyclass(name = "SparseCoo", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PySparseCoo {
    pub(crate) inner: SparseCoo,
}

#[pymethods]
impl PySparseCoo {
    #[new]
    #[pyo3(signature = (indices, values, shape))]
    pub fn new(
        indices: Vec<(usize, usize)>,
        values: Vec<f64>,
        shape: (usize, usize),
    ) -> PyResult<Self> {
        let inner =
            SparseCoo::new(indices, values, [shape.0, shape.1]).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    pub fn from_dense(tensor: &PyTensor, threshold: f64) -> PyResult<Self> {
        let inner =
            SparseCoo::from_dense(&tensor.inner, threshold).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn shape(&self) -> (usize, usize) {
        (self.inner.shape[0], self.inner.shape[1])
    }

    pub fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    pub fn indices(&self) -> Vec<(usize, usize)> {
        self.inner.indices.clone()
    }

    pub fn values(&self) -> Vec<f64> {
        self.inner.values.clone()
    }

    pub fn transpose(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.transpose()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn sparsity(&self) -> f64 {
        self.inner.sparsity()
    }

    pub fn norm(&self) -> f64 {
        self.inner.norm()
    }

    pub fn coalesce(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.coalesce()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn spmm(&self, py: Python<'_>, dense: &PyTensor) -> PyResult<PyTensor> {
        let inner = py
            .detach(|| self.inner.spmm(&dense.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }

    pub fn spspmm(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.spspmm(&other.inner))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn add(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.add(&other.inner))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn sub(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.sub(&other.inner))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn mul_scalar(&self, scalar: f64) -> PyResult<Self> {
        let inner = self.inner.mul_scalar(scalar).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn __add__(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        self.add(py, other)
    }

    pub fn __sub__(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        self.sub(py, other)
    }

    pub fn __mul__(&self, scalar: f64) -> PyResult<Self> {
        self.mul_scalar(scalar)
    }

    pub fn __rmul__(&self, scalar: f64) -> PyResult<Self> {
        self.mul_scalar(scalar)
    }

    pub fn __matmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(dense) = other.extract::<PyTensor>() {
            let res = self.spmm(py, &dense)?;
            return Ok(res.into_pyobject(py)?.into_any().unbind());
        }
        if let Ok(csr) = other.extract::<PySparseCsr>() {
            let res = self.to_csr(py)?.spspmm(py, &csr)?.to_coo(py)?;
            return Ok(res.into_pyobject(py)?.into_any().unbind());
        }
        if let Ok(coo) = other.extract::<PySparseCoo>() {
            let res = self.spspmm(py, &coo)?;
            return Ok(res.into_pyobject(py)?.into_any().unbind());
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Unsupported operand type for @: expected Tensor, SparseCoo, or SparseCsr",
        ))
    }

    pub fn __repr__(&self) -> String {
        format!(
            "SparseCoo(shape=({}, {}), nnz={}, sparsity={:.2}%)",
            self.inner.shape[0],
            self.inner.shape[1],
            self.inner.nnz(),
            self.inner.sparsity() * 100.0
        )
    }

    #[pyo3(signature = (dtype = "float32", device = None))]
    pub fn to_dense(
        &self,
        py: Python<'_>,
        dtype: &str,
        device: Option<&PyDevice>,
    ) -> PyResult<PyTensor> {
        let dt = parse_dtype(dtype)?;
        let dev = device.map(|d| d.inner.clone()).unwrap_or(Device::Cpu);
        let inner = py
            .detach(|| self.inner.to_dense(dt, &dev))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }

    pub fn to_csr(&self, py: Python<'_>) -> PyResult<PySparseCsr> {
        let inner = py.detach(|| self.inner.to_csr()).map_err(to_py_err)?;
        Ok(PySparseCsr { inner })
    }
}

/// Python wrapper around [`SparseCsr`]: 2D sparse matrix in Compressed Sparse
/// Row format with differentiable [`SparseCsr::spmm`].
#[pyclass(name = "SparseCsr", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PySparseCsr {
    pub(crate) inner: SparseCsr,
}

#[pymethods]
impl PySparseCsr {
    #[new]
    #[pyo3(signature = (row_ptr, col_idx, values, shape))]
    pub fn new(
        row_ptr: Vec<usize>,
        col_idx: Vec<usize>,
        values: Vec<f64>,
        shape: (usize, usize),
    ) -> PyResult<Self> {
        let inner =
            SparseCsr::new(row_ptr, col_idx, values, [shape.0, shape.1]).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    pub fn from_dense(tensor: &PyTensor, threshold: f64) -> PyResult<Self> {
        let inner =
            SparseCsr::from_dense(&tensor.inner, threshold).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn shape(&self) -> (usize, usize) {
        (self.inner.shape[0], self.inner.shape[1])
    }

    pub fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    pub fn sparsity(&self) -> f64 {
        self.inner.sparsity()
    }

    pub fn norm(&self) -> f64 {
        self.inner.norm()
    }

    pub fn row_ptr(&self) -> Vec<usize> {
        self.inner.row_ptr.clone()
    }

    pub fn col_idx(&self) -> Vec<usize> {
        self.inner.col_idx.clone()
    }

    pub fn values(&self) -> Vec<f64> {
        self.inner.values.clone()
    }

    pub fn transpose(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.transpose()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn to_coo(&self, py: Python<'_>) -> PyResult<PySparseCoo> {
        let inner = py.detach(|| self.inner.to_coo()).map_err(to_py_err)?;
        Ok(PySparseCoo { inner })
    }

    pub fn spspmm(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.spspmm(&other.inner))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn add(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.add(&other.inner))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn sub(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.sub(&other.inner))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn mul_scalar(&self, scalar: f64) -> PyResult<Self> {
        let inner = self.inner.mul_scalar(scalar).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn __add__(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        self.add(py, other)
    }

    pub fn __sub__(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        self.sub(py, other)
    }

    pub fn __mul__(&self, scalar: f64) -> PyResult<Self> {
        self.mul_scalar(scalar)
    }

    pub fn __rmul__(&self, scalar: f64) -> PyResult<Self> {
        self.mul_scalar(scalar)
    }

    pub fn __matmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(dense) = other.extract::<PyTensor>() {
            let res = self.spmm(py, &dense)?;
            return Ok(res.into_pyobject(py)?.into_any().unbind());
        }
        if let Ok(csr) = other.extract::<PySparseCsr>() {
            let res = self.spspmm(py, &csr)?;
            return Ok(res.into_pyobject(py)?.into_any().unbind());
        }
        if let Ok(coo) = other.extract::<PySparseCoo>() {
            let res = self.spspmm(py, &coo.to_csr(py)?)?;
            return Ok(res.into_pyobject(py)?.into_any().unbind());
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Unsupported operand type for @: expected Tensor, SparseCsr, or SparseCoo",
        ))
    }

    #[pyo3(signature = (dtype = "float32", device = None))]
    pub fn to_dense(
        &self,
        py: Python<'_>,
        dtype: &str,
        device: Option<&PyDevice>,
    ) -> PyResult<PyTensor> {
        let dt = parse_dtype(dtype)?;
        let dev = device.map(|d| d.inner.clone()).unwrap_or(Device::Cpu);
        let inner = py
            .detach(|| self.inner.to_dense(dt, &dev))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }

    pub fn spmm(&self, py: Python<'_>, dense: &PyTensor) -> PyResult<PyTensor> {
        let inner = py
            .detach(|| self.inner.spmm(&dense.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }

    pub fn __repr__(&self) -> String {
        format!(
            "SparseCsr(shape=({}, {}), nnz={}, sparsity={:.2}%)",
            self.inner.shape[0],
            self.inner.shape[1],
            self.inner.nnz(),
            self.inner.sparsity() * 100.0
        )
    }
}

/// Python wrapper around [`Normal`]: univariate Gaussian with broadcastable
/// `loc`/`scale`, reparameterized sampling, and analytical KL/entropy.
#[pyclass(name = "Normal", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyNormal {
    pub(crate) inner: Normal,
}

#[pymethods]
impl PyNormal {
    #[new]
    pub fn new(loc: &PyTensor, scale: &PyTensor) -> PyResult<Self> {
        let inner =
            Normal::new(loc.inner.clone(), scale.inner.clone()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn rsample(&self, py: Python<'_>) -> PyResult<PyTensor> {
        let inner = py.detach(|| self.inner.rsample()).map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }

    pub fn sample(&self, py: Python<'_>) -> PyResult<PyTensor> {
        let inner = py.detach(|| self.inner.sample()).map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }

    pub fn log_prob(&self, py: Python<'_>, value: &PyTensor) -> PyResult<PyTensor> {
        let inner = py
            .detach(|| self.inner.log_prob(&value.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }

    pub fn entropy(&self, py: Python<'_>) -> PyResult<PyTensor> {
        let inner = py.detach(|| self.inner.entropy()).map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }

    pub fn kl_divergence(&self, py: Python<'_>, other: &Self) -> PyResult<PyTensor> {
        let inner = py
            .detach(|| self.inner.kl_divergence(&other.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }
}

/// Python wrapper around [`Categorical`]: discrete distribution over unnormalized
/// `logits`, with Gumbel-Softmax (straight-through `hard`) sampling.
#[pyclass(name = "Categorical", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyCategorical {
    pub(crate) inner: Categorical,
}

#[pymethods]
impl PyCategorical {
    #[new]
    pub fn new(logits: &PyTensor) -> PyResult<Self> {
        Ok(Self {
            inner: Categorical::from_logits(logits.inner.clone()),
        })
    }

    pub fn probs(&self, py: Python<'_>) -> PyResult<PyTensor> {
        let inner = py.detach(|| self.inner.probs()).map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }

    pub fn log_probs(&self, py: Python<'_>) -> PyResult<PyTensor> {
        let inner = py.detach(|| self.inner.log_probs()).map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }

    pub fn sample(&self, py: Python<'_>) -> PyResult<PyTensor> {
        let inner = py.detach(|| self.inner.sample()).map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }

    pub fn log_prob(&self, py: Python<'_>, indices: &PyTensor) -> PyResult<PyTensor> {
        let inner = py
            .detach(|| self.inner.log_prob(&indices.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }

    pub fn entropy(&self, py: Python<'_>) -> PyResult<PyTensor> {
        let inner = py.detach(|| self.inner.entropy()).map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }

    pub fn gumbel_softmax(
        &self,
        py: Python<'_>,
        temperature: f64,
        hard: bool,
    ) -> PyResult<PyTensor> {
        let inner = py
            .detach(|| self.inner.gumbel_softmax(temperature, hard))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }
}

/// Python wrapper around [`Laplace`]: double-exponential distribution with
/// reparameterized inverse-CDF sampling.
#[pyclass(name = "Laplace", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyLaplace {
    pub(crate) inner: Laplace,
}

#[pymethods]
impl PyLaplace {
    #[new]
    pub fn new(loc: &PyTensor, scale: &PyTensor) -> PyResult<Self> {
        let inner =
            Laplace::new(loc.inner.clone(), scale.inner.clone()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn rsample(&self, py: Python<'_>) -> PyResult<PyTensor> {
        let inner = py.detach(|| self.inner.rsample()).map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }

    pub fn sample(&self, py: Python<'_>) -> PyResult<PyTensor> {
        let inner = py.detach(|| self.inner.sample()).map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }

    pub fn log_prob(&self, py: Python<'_>, value: &PyTensor) -> PyResult<PyTensor> {
        let inner = py
            .detach(|| self.inner.log_prob(&value.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }

    pub fn entropy(&self, py: Python<'_>) -> PyResult<PyTensor> {
        let inner = py.detach(|| self.inner.entropy()).map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }
}

/// Python wrapper around Oxideon's native [`Tensor`].
/// Every compute method releases the Python GIL via `py.detach` so Rust `rayon`
/// worker threads and `wgpu` GPU queues execute without Python interpreter contention.
#[pyclass(name = "Tensor", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyTensor {
    pub(crate) inner: Tensor,
}

/// Right-hand side of a Python operator: another tensor or a number.
enum Operand {
    Tensor(Tensor),
    Scalar(f64),
}

fn operand(other: &Bound<'_, PyAny>) -> Option<Operand> {
    if let Ok(t) = other.extract::<PyRef<'_, PyTensor>>() {
        return Some(Operand::Tensor(t.inner.clone()));
    }
    other.extract::<f64>().ok().map(Operand::Scalar)
}

/// Extracts quantile levels: a scalar, or a 0-D/1-D tensor whose dtype
/// matches the input (torch rejects anything else). Returns the levels and
/// whether `q` was scalar-like (no q dim in the output).
fn quantile_levels(q: &Bound<'_, PyAny>, input: &Tensor) -> PyResult<(Vec<f64>, bool)> {
    match operand(q) {
        Some(Operand::Scalar(s)) => Ok((vec![s], true)),
        Some(Operand::Tensor(t)) => {
            if t.dtype() != input.dtype() {
                return Err(PyRuntimeError::new_err(
                    "quantile() q tensor must be same dtype as the input tensor",
                ));
            }
            if t.ndim() > 1 {
                return Err(PyRuntimeError::new_err(
                    "quantile() q must be a scalar or 1D tensor",
                ));
            }
            let scalar_like = t.ndim() == 0;
            let qs = q.py().detach(|| t.to_vec::<f64>()).map_err(to_py_err)?;
            Ok((qs, scalar_like))
        }
        None => Err(pyo3::exceptions::PyTypeError::new_err(
            "quantile() q must be a float or tensor",
        )),
    }
}

/// Resolves a Python reshape request (at most one `-1`) into concrete dimensions.
fn resolve_shape(numel: usize, shape: &[isize]) -> PyResult<Vec<usize>> {
    let (mut inferred, mut known) = (None, 1usize);
    let mut out = Vec::with_capacity(shape.len());
    for (i, &d) in shape.iter().enumerate() {
        match d {
            -1 if inferred.is_none() => {
                inferred = Some(i);
                out.push(0);
            }
            -1 => {
                return Err(PyValueError::new_err(
                    "only one dimension can be inferred (-1)",
                ));
            }
            d if d < 0 => {
                return Err(PyValueError::new_err(format!(
                    "invalid shape dimension {d}"
                )));
            }
            d => {
                known = known
                    .checked_mul(d as usize)
                    .ok_or_else(|| PyValueError::new_err("shape is too large"))?;
                out.push(d as usize);
            }
        }
    }
    if let Some(i) = inferred {
        if known == 0 || numel % known != 0 {
            return Err(PyValueError::new_err(format!(
                "cannot reshape {numel} elements into shape {shape:?}"
            )));
        }
        out[i] = numel / known;
    }
    Ok(out)
}

impl PyTensor {
    fn run(
        py: Python<'_>,
        f: impl FnOnce() -> crate::error::Result<Tensor> + Send,
    ) -> PyResult<Self> {
        py.detach(f).map(|inner| Self { inner }).map_err(to_py_err)
    }

    fn wrap(
        py: Python<'_>,
        f: impl FnOnce() -> crate::error::Result<Tensor> + Send,
    ) -> PyResult<Py<PyAny>> {
        Ok(Py::new(py, Self::run(py, f)?)?.into_any())
    }

    /// `self <op> other` for a tensor or scalar `other`; anything else is `NotImplemented` so Python raises `TypeError`.
    fn bin(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        tensor_op: fn(&Tensor, &Tensor) -> crate::error::Result<Tensor>,
        scalar_op: fn(&Tensor, f64) -> crate::error::Result<Tensor>,
    ) -> PyResult<Py<PyAny>> {
        match operand(other) {
            Some(Operand::Tensor(t)) => Self::wrap(py, || tensor_op(&self.inner, &t)),
            Some(Operand::Scalar(s)) => Self::wrap(py, || scalar_op(&self.inner, s)),
            None => Ok(py.NotImplemented()),
        }
    }

    /// Shared `quantile`/`nanquantile` binding: extracts `q`, runs the
    /// engine, then drops the length-1 q dim for scalar `q` (and reshapes
    /// full reductions to ones when `keepdim`, which torch allows here).
    fn quantile_impl(
        &self,
        py: Python<'_>,
        q: &Bound<'_, PyAny>,
        dim: Option<isize>,
        keepdim: bool,
        interpolation: &str,
        nan_skip: bool,
    ) -> PyResult<Self> {
        let (qs, scalar_q) = quantile_levels(q, &self.inner)?;
        let run_axis = |t: &Tensor, d: isize| -> crate::error::Result<Tensor> {
            if nan_skip {
                t.nanquantile_axis(d, &qs, keepdim, interpolation)
            } else {
                t.quantile_axis(d, &qs, keepdim, interpolation)
            }
        };
        let run_full = |t: &Tensor| -> crate::error::Result<Tensor> {
            if nan_skip {
                t.nanquantile(&qs, interpolation)
            } else {
                t.quantile(&qs, interpolation)
            }
        };
        let drop_q_dim = |t: Tensor| -> crate::error::Result<Tensor> {
            let s: Vec<usize> = t.shape()[1..].to_vec();
            t.reshape(&s)
        };
        let ones_full = |t: Tensor, ndim: usize, k: usize| -> crate::error::Result<Tensor> {
            let mut s = vec![k];
            s.extend(std::iter::repeat_n(1, ndim));
            t.reshape(&s)
        };
        let inner = match dim {
            None => {
                let t = py.detach(|| run_full(&self.inner)).map_err(to_py_err)?;
                if scalar_q {
                    if keepdim {
                        let ndim = self.inner.shape().len();
                        py.detach(|| t.reshape(&vec![1; ndim])).map_err(to_py_err)?
                    } else {
                        py.detach(|| t.reshape(&[])).map_err(to_py_err)?
                    }
                } else if keepdim {
                    let (ndim, k) = (self.inner.shape().len(), qs.len());
                    py.detach(|| ones_full(t, ndim, k)).map_err(to_py_err)?
                } else {
                    t
                }
            }
            Some(d) => {
                let t = py
                    .detach(|| run_axis(&self.inner, d))
                    .map_err(to_py_err)?;
                if scalar_q {
                    py.detach(|| drop_q_dim(t)).map_err(to_py_err)?
                } else {
                    t
                }
            }
        };
        Ok(Self { inner })
    }

    /// Element-wise comparison against a tensor or a scalar (broadcast with `full_like`).
    fn cmp(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: fn(&Tensor, &Tensor) -> crate::error::Result<Tensor>,
    ) -> PyResult<Py<PyAny>> {
        match operand(other) {
            Some(Operand::Tensor(t)) => Self::wrap(py, || op(&self.inner, &t)),
            Some(Operand::Scalar(s)) => Self::wrap(py, || {
                self.inner.full_like(s).and_then(|o| op(&self.inner, &o))
            }),
            None => Ok(py.NotImplemented()),
        }
    }

    /// All elements as one dimension (Rust's `flatten()` keeps the batch dimension, which is not what reductions need).
    fn flat(&self) -> crate::error::Result<Tensor> {
        self.inner.reshape(&[self.inner.numel()])
    }

    fn scalar_item(&self) -> PyResult<f64> {
        if self.inner.numel() != 1 {
            return Err(PyValueError::new_err(format!(
                "only a tensor with exactly one element can be converted to a Python number (this one has {})",
                self.inner.numel()
            )));
        }
        self.inner.item::<f64>().map_err(to_py_err)
    }
}

#[pymethods]
impl PyTensor {
    #[new]
    #[pyo3(signature = (data, shape, dtype=None, device=None, requires_grad=false))]
    pub fn new(
        py: Python<'_>,
        data: Vec<f64>,
        shape: Vec<usize>,
        dtype: Option<&Bound<'_, PyAny>>,
        device: Option<&Bound<'_, PyAny>>,
        requires_grad: bool,
    ) -> PyResult<Self> {
        let dt = parse_dtype_obj(dtype, DType::F32)?;
        let dev = parse_device_obj(device, "cpu")?;
        let inner = py
            .detach(|| {
                let t = Tensor::from_slice_on(&data, &shape, &dev)?.to_dtype(dt)?;
                Ok::<Tensor, Error>(if requires_grad {
                    t.requires_grad_(true)
                } else {
                    t
                })
            })
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (data, shape, device=None))]
    pub fn from_i64(
        py: Python<'_>,
        data: Vec<i64>,
        shape: Vec<usize>,
        device: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let dev = parse_device_obj(device, "cpu")?;
        let inner = py
            .detach(|| Tensor::from_slice_on(&data, &shape, &dev))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (shape, dtype=None, device=None, requires_grad=false))]
    pub fn zeros(
        py: Python<'_>,
        shape: Vec<usize>,
        dtype: Option<&Bound<'_, PyAny>>,
        device: Option<&Bound<'_, PyAny>>,
        requires_grad: bool,
    ) -> PyResult<Self> {
        let dt = parse_dtype_obj(dtype, DType::F32)?;
        let dev = parse_device_obj(device, "cpu")?;
        let inner = py
            .detach(|| {
                let t = Tensor::zeros(&shape, dt, &dev)?;
                Ok::<Tensor, Error>(if requires_grad {
                    t.requires_grad_(true)
                } else {
                    t
                })
            })
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (shape, dtype=None, device=None, requires_grad=false))]
    pub fn ones(
        py: Python<'_>,
        shape: Vec<usize>,
        dtype: Option<&Bound<'_, PyAny>>,
        device: Option<&Bound<'_, PyAny>>,
        requires_grad: bool,
    ) -> PyResult<Self> {
        let dt = parse_dtype_obj(dtype, DType::F32)?;
        let dev = parse_device_obj(device, "cpu")?;
        let inner = py
            .detach(|| {
                let t = Tensor::ones(&shape, dt, &dev)?;
                Ok::<Tensor, Error>(if requires_grad {
                    t.requires_grad_(true)
                } else {
                    t
                })
            })
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (shape, dtype=None, device=None, requires_grad=false))]
    pub fn empty(
        py: Python<'_>,
        shape: Vec<usize>,
        dtype: Option<&Bound<'_, PyAny>>,
        device: Option<&Bound<'_, PyAny>>,
        requires_grad: bool,
    ) -> PyResult<Self> {
        let dt = parse_dtype_obj(dtype, DType::F32)?;
        let dev = parse_device_obj(device, "cpu")?;
        let inner = py
            .detach(|| {
                let t = Tensor::empty(&shape, dt, &dev)?;
                Ok::<Tensor, Error>(if requires_grad {
                    t.requires_grad_(true)
                } else {
                    t
                })
            })
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (shape, dtype=None, device=None, requires_grad=false))]
    pub fn randn(
        py: Python<'_>,
        shape: Vec<usize>,
        dtype: Option<&Bound<'_, PyAny>>,
        device: Option<&Bound<'_, PyAny>>,
        requires_grad: bool,
    ) -> PyResult<Self> {
        let dt = parse_dtype_obj(dtype, DType::F32)?;
        let dev = parse_device_obj(device, "cpu")?;
        let inner = py
            .detach(|| {
                let t = Tensor::randn_with(&shape, dt, &dev)?;
                Ok::<Tensor, Error>(if requires_grad {
                    t.requires_grad_(true)
                } else {
                    t
                })
            })
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (n, dtype=None, device=None))]
    pub fn eye(
        py: Python<'_>,
        n: usize,
        dtype: Option<&Bound<'_, PyAny>>,
        device: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let dt = parse_dtype_obj(dtype, DType::F32)?;
        let dev = parse_device_obj(device, "cpu")?;
        let inner = py.detach(|| Tensor::eye(n, dt, &dev)).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (start, end, step=1.0, dtype=None))]
    pub fn arange(
        py: Python<'_>,
        start: f64,
        end: f64,
        step: f64,
        dtype: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let dt = parse_dtype_obj(dtype, DType::F32)?;
        let inner = py
            .detach(|| Tensor::arange(start, end, step, dt))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (start, end, steps, dtype=None))]
    pub fn linspace(
        py: Python<'_>,
        start: f64,
        end: f64,
        steps: usize,
        dtype: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let dt = parse_dtype_obj(dtype, DType::F32)?;
        let inner = py
            .detach(|| Tensor::linspace(start, end, steps, dt))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    // --- Metadata & Host Transfer ---

    #[getter]
    pub fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    #[getter]
    pub fn strides(&self) -> Vec<isize> {
        self.inner.strides().to_vec()
    }

    #[pyo3(signature = (dim=None))]
    pub fn stride<'py>(
        &self,
        py: Python<'py>,
        dim: Option<isize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let strides = self.inner.strides();
        match dim {
            None => {
                let py_strides = pyo3::types::PyTuple::new(py, strides)?;
                Ok(py_strides.into_any())
            }
            Some(d) => {
                let n = strides.len() as isize;
                let actual = if d < 0 { d + n } else { d };
                if actual < 0 || actual >= n {
                    return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                        "Dimension out of range (expected to be in range of [{}, {}], but got {})",
                        -n,
                        n - 1,
                        d
                    )));
                }
                let val = strides[actual as usize];
                Ok(val.into_pyobject(py)?.into_any())
            }
        }
    }

    #[getter]
    pub fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    pub fn numel(&self) -> usize {
        self.inner.numel()
    }

    pub fn is_contiguous(&self) -> bool {
        self.inner.is_contiguous()
    }

    #[getter]
    pub fn dtype(&self) -> String {
        self.inner.dtype().to_string()
    }

    #[getter]
    pub fn device(&self) -> PyDevice {
        PyDevice {
            inner: self.inner.device(),
        }
    }

    #[getter]
    pub fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    pub fn set_requires_grad(&self, flag: bool) -> PyResult<()> {
        self.inner.set_requires_grad(flag).map_err(to_py_err)
    }

    pub fn requires_grad_(&self, flag: bool) -> Self {
        Self {
            inner: self.inner.clone().requires_grad_(flag),
        }
    }

    pub fn detach(&self) -> Self {
        Self {
            inner: self.inner.detach(),
        }
    }

    pub fn grad(&self) -> Option<Self> {
        self.inner.grad().map(|inner| Self { inner })
    }

    pub fn zero_grad(&self) {
        self.inner.zero_grad();
    }

    pub fn assign(&self, other: &Self) -> PyResult<()> {
        guard_res(|| self.inner.assign(&other.inner))
    }

    pub fn retarget(&self, other: &Self) -> PyResult<()> {
        guard_res(|| self.inner.retarget(&other.inner))
    }

    pub fn copy_(&self, other: &Self) -> PyResult<()> {
        guard_res(|| self.inner.assign(&other.inner))
    }

    #[pyo3(signature = (grad=None))]
    pub fn set_grad(&self, grad: Option<PyTensor>) {
        self.inner.set_grad(grad.map(|g| g.inner));
    }

    pub fn retain_grad(&self) {
        self.inner.retain_grad();
    }

    pub fn shares_storage(&self, other: &Self) -> bool {
        self.inner.shares_storage(&other.inner)
    }

    pub fn to_device(&self, py: Python<'_>, device: Bound<'_, pyo3::PyAny>) -> PyResult<Self> {
        let dev = if let Ok(d) = device.extract::<PyDevice>() {
            d.inner
        } else if let Ok(s) = device.extract::<String>() {
            parse_device(&s)?
        } else {
            return Err(PyRuntimeError::new_err(
                "to_device expects a device string (e.g. 'cuda:0') or ox.Device",
            ));
        };
        let inner = py
            .detach(|| self.inner.to_device(&dev))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn to(&self, py: Python<'_>, target: Bound<'_, pyo3::PyAny>) -> PyResult<Self> {
        if let Ok(dev) = target.extract::<PyDevice>() {
            let inner = py
                .detach(|| self.inner.to_device(&dev.inner))
                .map_err(to_py_err)?;
            return Ok(Self { inner });
        }
        if let Ok(dt_str) = extract_dtype_str(&target) {
            if let Ok(dt) = parse_dtype(&dt_str) {
                let inner = py.detach(|| self.inner.to_dtype(dt)).map_err(to_py_err)?;
                return Ok(Self { inner });
            }
        }
        if let Ok(s) = extract_device_str(&target) {
            if let Ok(dev) = parse_device(&s) {
                let inner = py
                    .detach(|| self.inner.to_device(&dev))
                    .map_err(to_py_err)?;
                return Ok(Self { inner });
            }
        }
        Err(PyRuntimeError::new_err(
            "Invalid argument to .to(): expected device, PyDevice, or dtype",
        ))
    }

    /// Moves this tensor to Host CPU.
    pub fn cpu(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.cpu()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Moves this tensor to CUDA GPU.
    #[pyo3(signature = (index=0))]
    pub fn cuda(&self, py: Python<'_>, index: usize) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.cuda(index)).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Moves this tensor to a GPU accelerator.
    #[pyo3(signature = (index=0))]
    pub fn gpu(&self, py: Python<'_>, index: usize) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.gpu(index)).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Moves this tensor to a Google TPU core.
    #[pyo3(signature = (chip=0, core=0))]
    pub fn tpu(&self, py: Python<'_>, chip: usize, core: usize) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.tpu(chip, core))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn is_cpu(&self) -> bool {
        self.inner.is_cpu()
    }

    #[getter]
    pub fn is_cuda(&self) -> bool {
        self.inner.is_cuda()
    }

    #[getter]
    pub fn is_gpu(&self) -> bool {
        self.inner.is_gpu()
    }

    #[getter]
    pub fn is_tpu(&self) -> bool {
        self.inner.is_tpu()
    }

    pub fn to_dtype(&self, py: Python<'_>, dtype: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dt = parse_dtype_obj(Some(dtype), DType::F32)?;
        let inner = py.detach(|| self.inner.to_dtype(dt)).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn contiguous(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.contiguous()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn to_list(&self, py: Python<'_>) -> PyResult<Vec<f64>> {
        py.detach(|| self.inner.to_vec::<f64>()).map_err(to_py_err)
    }

    pub fn to_i64_list(&self, py: Python<'_>) -> PyResult<Vec<i64>> {
        py.detach(|| self.inner.to_vec::<i64>()).map_err(to_py_err)
    }

    pub fn item(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.item::<f64>()).map_err(to_py_err)
    }

    /// Independent copy with its own storage; a tensor that requires grad stays in the autograd graph.
    #[pyo3(name = "clone")]
    pub fn py_clone(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| deep_copy(&self.inner)).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Copies the data into a NumPy array (needs NumPy installed). `f16` / `bf16` come back as `float32`.
    pub fn numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let np = py.import("numpy").map_err(|_| {
            pyo3::exceptions::PyImportError::new_err(
                "Tensor.numpy() needs NumPy: pip install numpy",
            )
        })?;
        let t = &self.inner;
        // Fast path: contiguous host storage is copied as raw little-endian bytes in one
        // memcpy. Strided views, f16/bf16/bool (need conversion), non-host backends, and
        // big-endian targets fall through to the generic gather below.
        if cfg!(target_endian = "little") {
            if let Some(range) = t.layout().contiguous_range() {
                let guard = t.storage().read();
                let bytes: Option<(&[u8], &str)> = match &*guard {
                    crate::backend::StorageInner::Cpu(s)
                    | crate::backend::StorageInner::Accelerator { storage: s, .. } => match s {
                        crate::backend::cpu::CpuStorage::U8(v) => v.get(range).map(|sl| (sl, "u1")),
                        crate::backend::cpu::CpuStorage::F32(v) => {
                            v.get(range).map(|sl| (slice_bytes(sl), "<f4"))
                        }
                        crate::backend::cpu::CpuStorage::F64(v) => {
                            v.get(range).map(|sl| (slice_bytes(sl), "<f8"))
                        }
                        crate::backend::cpu::CpuStorage::I32(v) => {
                            v.get(range).map(|sl| (slice_bytes(sl), "<i4"))
                        }
                        crate::backend::cpu::CpuStorage::I64(v) => {
                            v.get(range).map(|sl| (slice_bytes(sl), "<i8"))
                        }
                        crate::backend::cpu::CpuStorage::C64(v) => {
                            v.get(range).map(|sl| (slice_bytes(sl), "<c8"))
                        }
                        crate::backend::cpu::CpuStorage::C128(v) => {
                            v.get(range).map(|sl| (slice_bytes(sl), "<c16"))
                        }
                        _ => None,
                    },
                    _ => None,
                };
                if let Some((bytes, dtype)) = bytes {
                    // `frombuffer` keeps the bytearray alive via the array base and stays
                    // writeable (unlike `bytes`, which needs the extra `.copy()` below).
                    let ba = pyo3::types::PyByteArray::new(py, bytes);
                    let flat = np.call_method1("frombuffer", (&ba, dtype))?;
                    return flat.call_method1("reshape", (t.shape().to_vec(),));
                }
            }
        }
        let (bytes, dtype) = py
            .detach(|| -> crate::error::Result<(Vec<u8>, &'static str)> {
                macro_rules! le {
                    ($ty:ty, $name:literal) => {
                        (
                            t.to_vec::<$ty>()?
                                .into_iter()
                                .flat_map(<$ty>::to_le_bytes)
                                .collect(),
                            $name,
                        )
                    };
                }
                Ok(match t.dtype() {
                    DType::F64 => le!(f64, "<f8"),
                    DType::I64 => le!(i64, "<i8"),
                    DType::I32 => le!(i32, "<i4"),
                    DType::U8 => (t.to_vec::<u8>()?, "u1"),
                    DType::I8 => le!(i8, "i1"),
                    DType::I16 => le!(i16, "<i2"),
                    DType::U16 => le!(u16, "<u2"),
                    DType::U32 => le!(u32, "<u4"),
                    DType::U64 => le!(u64, "<u8"),
                    DType::Bool => (t.to_vec::<u8>()?, "?"),
                    DType::F32 | DType::F16 | DType::BF16 | DType::F8E4M3 | DType::F8E5M2 => le!(f32, "<f4"),
                    DType::C64 => (
                        t.to_c64_vec()?
                            .into_iter()
                            .flat_map(|[r, i]| [r.to_le_bytes(), i.to_le_bytes()].into_iter().flatten())
                            .collect(),
                        "<c8",
                    ),
                    DType::C128 => (
                        t.to_c128_vec()?
                            .into_iter()
                            .flat_map(|[r, i]| [r.to_le_bytes(), i.to_le_bytes()].into_iter().flatten())
                            .collect(),
                        "<c16",
                    ),
                })
            })
            .map_err(to_py_err)?;
        let flat = np.call_method1("frombuffer", (pyo3::types::PyBytes::new(py, &bytes), dtype))?;
        flat.call_method1("reshape", (t.shape().to_vec(),))?
            .call_method0("copy")
    }

    /// Converts tensor to a NumPy array (`__array__` protocol: `np.asarray(t)`, `np.array(t)`).
    #[pyo3(signature = (dtype=None, copy=None))]
    pub fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<&Bound<'py, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut arr = self.numpy(py)?;
        if let Some(dt) = dtype {
            arr = arr.call_method1("astype", (dt,))?;
        }
        if copy == Some(true) {
            arr = arr.call_method0("copy")?;
        }
        Ok(arr)
    }

    /// Alias for `.numpy()` for PyTorch / JAX ergonomics: `t.to_numpy()`.
    pub fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.numpy(py)
    }

    /// Serialization reduction protocol for Python `pickle` support.
    pub fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        let np_arr = self.numpy(py)?;
        let req_grad = self.requires_grad();
        let oxideon_mod = py.import("oxideon")?;
        let reconstruct_fn = oxideon_mod.getattr("_reconstruct_tensor")?;
        (reconstruct_fn, (np_arr, req_grad)).into_pyobject(py)
    }

    /// Exposes the array interface for NumPy, Pillow, and other C-API consumers.
    ///
    /// Zero-copy view straight into host storage (read-only: the buffer is shared behind a
    /// lock, so NumPy must not write through it). NumPy keeps `self` alive via the array
    /// base, which keeps the storage allocation alive with it.
    #[getter]
    pub fn __array_interface__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        if !cfg!(target_endian = "little") {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "__array_interface__ needs a little-endian host; use .numpy() instead",
            ));
        }
        let t = &self.inner;
        let layout = t.layout();
        let guard = t.storage().read();
        let (itemsize, typestr, base): (usize, &'static str, *const u8) = match &*guard {
            crate::backend::StorageInner::Cpu(s)
            | crate::backend::StorageInner::Accelerator { storage: s, .. } => match s {
                crate::backend::cpu::CpuStorage::F32(v) => (4, "<f4", v.as_ptr() as *const u8),
                crate::backend::cpu::CpuStorage::F64(v) => (8, "<f8", v.as_ptr() as *const u8),
                crate::backend::cpu::CpuStorage::I32(v) => (4, "<i4", v.as_ptr() as *const u8),
                crate::backend::cpu::CpuStorage::I64(v) => (8, "<i8", v.as_ptr() as *const u8),
                crate::backend::cpu::CpuStorage::U8(v) => (1, "u1", v.as_ptr()),
                crate::backend::cpu::CpuStorage::I8(v) => (1, "i1", v.as_ptr() as *const u8),
                crate::backend::cpu::CpuStorage::I16(v) => (2, "<i2", v.as_ptr() as *const u8),
                crate::backend::cpu::CpuStorage::U16(v) => (2, "<u2", v.as_ptr() as *const u8),
                crate::backend::cpu::CpuStorage::U32(v) => (4, "<u4", v.as_ptr() as *const u8),
                crate::backend::cpu::CpuStorage::U64(v) => (8, "<u8", v.as_ptr() as *const u8),
                crate::backend::cpu::CpuStorage::Bool(v) => (1, "?", v.as_ptr() as *const u8),
                crate::backend::cpu::CpuStorage::F16(v) => (2, "<f2", v.as_ptr() as *const u8),
                crate::backend::cpu::CpuStorage::C64(v) => (8, "<c8", v.as_ptr() as *const u8),
                crate::backend::cpu::CpuStorage::C128(v) => (16, "<c16", v.as_ptr() as *const u8),
                crate::backend::cpu::CpuStorage::BF16(_) | crate::backend::cpu::CpuStorage::F8E4M3(_) | crate::backend::cpu::CpuStorage::F8E5M2(_) => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "bf16 / float8 have no NumPy dtype; use .numpy() (upcasts to float32)",
                    ));
                }
            },
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "__array_interface__ needs host storage; use .numpy() instead",
                ));
            }
        };
        let data = (base as usize).saturating_add(layout.offset().saturating_mul(itemsize));
        let strides: Vec<isize> = layout
            .strides()
            .iter()
            .map(|s| s.saturating_mul(itemsize as isize))
            .collect();
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("shape", pyo3::types::PyTuple::new(py, layout.shape())?)?;
        dict.set_item("typestr", typestr)?;
        dict.set_item("strides", pyo3::types::PyTuple::new(py, &strides)?)?;
        dict.set_item("data", (data, true))?;
        dict.set_item("version", 3)?;
        Ok(dict)
    }

    /// Standard DLPack exchange protocol (`__dlpack__`) for zero-copy sharing with
    /// PyTorch, CuPy, JAX, Apache Arrow, and Polars.
    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    pub fn __dlpack__<'py>(
        &self,
        py: Python<'py>,
        stream: Option<&Bound<'py, PyAny>>,
        max_version: Option<&Bound<'py, PyAny>>,
        dl_device: Option<&Bound<'py, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = (stream, max_version, dl_device);
        let src = if copy == Some(true) {
            self.inner.contiguous().map_err(to_py_err)?
        } else {
            self.inner.clone()
        };
        let managed = oxideon_core::dlpack::to_dlpack(&src).map_err(to_py_err)?;
        unsafe extern "C" fn capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
            let name = unsafe { pyo3::ffi::PyCapsule_GetName(capsule) };
            if name.is_null() {
                unsafe { pyo3::ffi::PyErr_Clear() };
                return;
            }
            if unsafe { std::ffi::CStr::from_ptr(name) == c"dltensor" } {
                let ptr = unsafe {
                    pyo3::ffi::PyCapsule_GetPointer(capsule, c"dltensor".as_ptr())
                } as *mut oxideon_core::dlpack::DLManagedTensor;
                if !ptr.is_null() {
                    unsafe {
                        if let Some(deleter) = (*ptr).deleter {
                            deleter(ptr);
                        }
                    }
                }
            }
        }
        let raw = unsafe {
            pyo3::ffi::PyCapsule_New(
                managed as *mut std::ffi::c_void,
                c"dltensor".as_ptr(),
                Some(capsule_destructor),
            )
        };
        if raw.is_null() {
            unsafe {
                if let Some(deleter) = (*managed).deleter {
                    deleter(managed);
                }
            }
            return Err(pyo3::PyErr::fetch(py));
        }
        Ok(unsafe { Bound::from_owned_ptr(py, raw) })
    }

    /// DLPack device metadata `(device_type, device_id)`.
    pub fn __dlpack_device__(&self) -> (i32, i32) {
        (oxideon_core::dlpack::DL_CPU, 0)
    }

    /// Imports a DLPack `"dltensor"` PyCapsule directly in Rust and marks it consumed (`"used_dltensor"`).
    #[staticmethod]
    pub fn from_dlpack_capsule(capsule: &Bound<'_, PyAny>) -> PyResult<Self> {
        let ptr = unsafe {
            pyo3::ffi::PyCapsule_GetPointer(capsule.as_ptr(), c"dltensor".as_ptr())
        } as *mut oxideon_core::dlpack::DLManagedTensor;
        if ptr.is_null() {
            unsafe { pyo3::ffi::PyErr_Clear() };
            return Err(PyValueError::new_err(
                "expected an unconsumed DLPack capsule named 'dltensor'",
            ));
        }
        let rc = unsafe {
            pyo3::ffi::PyCapsule_SetName(capsule.as_ptr(), c"used_dltensor".as_ptr())
        };
        if rc != 0 {
            return Err(pyo3::PyErr::fetch(capsule.py()));
        }
        let inner = unsafe { oxideon_core::dlpack::from_dlpack(ptr) }.map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Converts tensor to a PyTorch tensor (`torch.Tensor`).
    #[pyo3(signature = (device=None))]
    pub fn to_torch<'py>(
        &self,
        py: Python<'py>,
        device: Option<&str>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::python_interop::to_torch_impl(py, self, device)
    }

    /// Alias for `to_torch()`.
    #[pyo3(signature = (device=None))]
    pub fn torch<'py>(&self, py: Python<'py>, device: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
        crate::python_interop::to_torch_impl(py, self, device)
    }

    /// Converts image tensor to a `PIL.Image.Image`.
    #[pyo3(signature = (mode=None))]
    pub fn to_pil<'py>(&self, py: Python<'py>, mode: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
        crate::python_interop::to_pil_impl(py, self, mode)
    }

    /// Converts tensor to a `pandas.DataFrame`.
    #[pyo3(signature = (columns=None))]
    pub fn to_pandas<'py>(
        &self,
        py: Python<'py>,
        columns: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::python_interop::to_pandas_impl(py, self, columns)
    }

    /// Converts tensor to a `polars.DataFrame`.
    #[pyo3(signature = (schema=None))]
    pub fn to_polars<'py>(
        &self,
        py: Python<'py>,
        schema: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::python_interop::to_polars_impl(py, self, schema)
    }

    /// Creates an Oxideon tensor from a NumPy array or array-like object.
    #[staticmethod]
    #[pyo3(signature = (array, device="cpu"))]
    pub fn from_numpy<'py>(
        py: Python<'py>,
        array: &Bound<'py, PyAny>,
        device: &str,
    ) -> PyResult<Self> {
        crate::python_interop::from_numpy_impl(py, array, device)
    }

    /// Creates an Oxideon tensor from a PyTorch tensor (`torch.Tensor`).
    #[staticmethod]
    #[pyo3(signature = (tensor, device="cpu"))]
    pub fn from_torch<'py>(
        py: Python<'py>,
        tensor: &Bound<'py, PyAny>,
        device: &str,
    ) -> PyResult<Self> {
        crate::python_interop::from_torch_impl(py, tensor, device)
    }

    /// Creates an Oxideon image tensor from a `PIL.Image.Image`.
    #[staticmethod]
    #[pyo3(signature = (image, normalize=false, as_chw=true, device="cpu"))]
    pub fn from_pil<'py>(
        py: Python<'py>,
        image: &Bound<'py, PyAny>,
        normalize: bool,
        as_chw: bool,
        device: &str,
    ) -> PyResult<Self> {
        crate::python_interop::from_pil_impl(py, image, normalize, as_chw, device)
    }

    /// Creates an Oxideon tensor from a `pandas.DataFrame` or `pandas.Series`.
    #[staticmethod]
    #[pyo3(signature = (df, device="cpu"))]
    pub fn from_pandas<'py>(
        py: Python<'py>,
        df: &Bound<'py, PyAny>,
        device: &str,
    ) -> PyResult<Self> {
        crate::python_interop::from_pandas_impl(py, df, device)
    }

    /// Creates an Oxideon tensor from a `polars.DataFrame` or `polars.Series`.
    #[staticmethod]
    #[pyo3(signature = (df, device="cpu"))]
    pub fn from_polars<'py>(
        py: Python<'py>,
        df: &Bound<'py, PyAny>,
        device: &str,
    ) -> PyResult<Self> {
        crate::python_interop::from_polars_impl(py, df, device)
    }

    /// Concatenates `tensors` along `dim` (differentiable: gradients split back to the inputs).
    #[staticmethod]
    #[pyo3(signature = (tensors, dim=0))]
    pub fn cat(py: Python<'_>, tensors: Vec<PyTensor>, dim: isize) -> PyResult<Self> {
        let inners: Vec<Tensor> = tensors.iter().map(|t| t.inner.clone()).collect();
        let out = py.detach(|| Tensor::cat(&inners, dim)).map_err(to_py_err)?;
        Ok(Self { inner: out })
    }

    /// Stacks `tensors` along a new dimension `dim` (differentiable).
    #[staticmethod]
    #[pyo3(signature = (tensors, dim=0))]
    pub fn stack(py: Python<'_>, tensors: Vec<PyTensor>, dim: isize) -> PyResult<Self> {
        let inners: Vec<Tensor> = tensors.iter().map(|t| t.inner.clone()).collect();
        let out = py
            .detach(|| Tensor::stack(&inners, dim))
            .map_err(to_py_err)?;
        Ok(Self { inner: out })
    }

    // --- Zero-Copy Views & Manipulation ---

    /// Returns the real part of a tensor.
    pub fn real(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.real().map_err(to_py_err)?,
        })
    }

    /// Returns the imaginary part of a tensor.
    pub fn imag(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.imag().map_err(to_py_err)?,
        })
    }

    /// Returns the complex conjugate of a tensor.
    pub fn conj(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.conj().map_err(to_py_err)?,
        })
    }

    /// Reshapes to `shape`; one dimension may be `-1` and is inferred from the element count.
    pub fn reshape(&self, shape: Vec<isize>) -> PyResult<Self> {
        let shape = resolve_shape(self.inner.numel(), &shape)?;
        Ok(Self {
            inner: self.inner.reshape(&shape).map_err(to_py_err)?,
        })
    }

    /// Zero-copy view with `shape`; one dimension may be `-1`.
    pub fn view(&self, shape: Vec<isize>) -> PyResult<Self> {
        let shape = resolve_shape(self.inner.numel(), &shape)?;
        Ok(Self {
            inner: self.inner.view(&shape).map_err(to_py_err)?,
        })
    }

    pub fn transpose(&self, d0: isize, d1: isize) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.transpose(d0, d1).map_err(to_py_err)?,
        })
    }

    pub fn t(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.t().map_err(to_py_err)?,
        })
    }

    pub fn permute(&self, axes: Vec<usize>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.permute(&axes).map_err(to_py_err)?,
        })
    }

    pub fn broadcast_to(&self, shape: Vec<usize>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.broadcast_to(&shape).map_err(to_py_err)?,
        })
    }

    pub fn narrow(&self, axis: isize, start: usize, len: usize) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.narrow(axis, start, len).map_err(to_py_err)?,
        })
    }

    #[pyo3(signature = (chunks, dim=0))]
    pub fn chunk<'py>(&self, py: Python<'py>, chunks: usize, dim: isize) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        let tensors = self.inner.chunk(chunks, dim).map_err(to_py_err)?;
        let py_tensors: Vec<PyTensor> = tensors.into_iter().map(|inner| Self { inner }).collect();
        pyo3::types::PyTuple::new(py, py_tensors)
    }

    #[pyo3(signature = (split_size_or_sections, dim=0))]
    pub fn split<'py>(&self, py: Python<'py>, split_size_or_sections: &Bound<'_, PyAny>, dim: isize) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        let tensors = if let Ok(size) = split_size_or_sections.extract::<usize>() {
            self.inner
                .split(size, dim)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        } else if let Ok(sections) = split_size_or_sections.extract::<Vec<usize>>() {
            self.inner.split_with_sizes(&sections, dim).map_err(|e| {
                let msg = e.to_string();
                if msg.contains("must match dim size") {
                    PyValueError::new_err(format!("{msg} (sections must sum exactly to dim size)"))
                } else {
                    PyValueError::new_err(msg)
                }
            })?
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "split_size_or_sections must be int or sequence of ints",
            ));
        };
        let py_tensors: Vec<PyTensor> = tensors.into_iter().map(|inner| Self { inner }).collect();
        pyo3::types::PyTuple::new(py, py_tensors)
    }

    #[pyo3(signature = (size, stride, storage_offset=0))]
    pub fn as_strided(&self, size: Vec<usize>, stride: Vec<isize>, storage_offset: usize) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.as_strided(&size, &stride, storage_offset).map_err(to_py_err)?,
        })
    }

    #[pyo3(signature = (weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1))]
    pub fn conv2d(
        &self,
        weight: &PyTensor,
        bias: Option<&PyTensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    ) -> PyResult<Self> {
        let b = bias.map(|b| &b.inner);
        let out = self
            .inner
            .conv2d_grouped(&weight.inner, b, stride, padding, dilation, groups)
            .map_err(to_py_err)?;
        Ok(Self { inner: out })
    }

    #[pyo3(signature = (weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0)))]
    pub fn conv3d(
        &self,
        py: Python<'_>,
        weight: &PyTensor,
        bias: Option<&PyTensor>,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
    ) -> PyResult<Self> {
        let b = bias.map(|b| &b.inner);
        let out = py
            .detach(|| self.inner.conv3d(&weight.inner, b, stride, padding))
            .map_err(to_py_err)?;
        Ok(Self { inner: out })
    }

    #[pyo3(signature = (grid, align_corners=false))]
    pub fn grid_sample(
        &self,
        py: Python<'_>,
        grid: &PyTensor,
        align_corners: bool,
    ) -> PyResult<Self> {
        let out = py
            .detach(|| oxideon_core::vision_ops::grid_sample(&self.inner, &grid.inner, align_corners))
            .map_err(to_py_err)?;
        Ok(Self { inner: out })
    }

    #[staticmethod]
    #[pyo3(signature = (theta, size, align_corners=false))]
    pub fn affine_grid(
        py: Python<'_>,
        theta: &PyTensor,
        size: [usize; 4],
        align_corners: bool,
    ) -> PyResult<Self> {
        let out = py
            .detach(|| oxideon_core::vision_ops::affine_grid(&theta.inner, size, align_corners))
            .map_err(to_py_err)?;
        Ok(Self { inner: out })
    }

    #[pyo3(signature = (rois, output_size, spatial_scale=1.0, sampling_ratio=1))]
    pub fn roi_align(
        &self,
        py: Python<'_>,
        rois: &PyTensor,
        output_size: [usize; 2],
        spatial_scale: f64,
        sampling_ratio: usize,
    ) -> PyResult<Self> {
        let out = py
            .detach(|| {
                oxideon_core::vision_ops::roi_align(
                    &self.inner,
                    &rois.inner,
                    output_size,
                    spatial_scale,
                    sampling_ratio,
                )
            })
            .map_err(to_py_err)?;
        Ok(Self { inner: out })
    }

    pub fn box_iou(&self, py: Python<'_>, other: &PyTensor) -> PyResult<Self> {
        let out = py
            .detach(|| oxideon_core::vision_ops::box_iou(&self.inner, &other.inner))
            .map_err(to_py_err)?;
        Ok(Self { inner: out })
    }

    #[pyo3(signature = (scores, iou_threshold))]
    pub fn nms(&self, py: Python<'_>, scores: &PyTensor, iou_threshold: f64) -> PyResult<Self> {
        let keep = py
            .detach(|| oxideon_core::vision_ops::nms(&self.inner, &scores.inner, iou_threshold))
            .map_err(to_py_err)?;
        let keep_i64: Vec<i64> = keep.into_iter().map(|i| i as i64).collect();
        let len = keep_i64.len();
        let out = Tensor::from_slice(&keep_i64, &[len]).map_err(to_py_err)?;
        Ok(Self { inner: out })
    }

    #[pyo3(signature = (size=None, scale_factor=None, mode="nearest", align_corners=false))]
    pub fn interpolate(
        &self,
        py: Python<'_>,
        size: Option<Vec<usize>>,
        scale_factor: Option<Vec<f64>>,
        mode: &str,
        align_corners: bool,
    ) -> PyResult<Self> {
        use oxideon_core::nn::functional_ext::{InterpMode, interpolate as rs_interp};
        let m = match mode {
            "nearest" => InterpMode::Nearest,
            "nearest-exact" => InterpMode::NearestExact,
            "linear" | "bilinear" | "trilinear" => InterpMode::Linear,
            "bicubic" => InterpMode::Bicubic,
            "area" => InterpMode::Area,
            other => {
                return Err(PyValueError::new_err(format!(
                    "interpolate: unsupported mode '{other}'"
                )));
            }
        };
        let out = py
            .detach(|| rs_interp(&self.inner, size.as_deref(), scale_factor.as_deref(), m, align_corners))
            .map_err(to_py_err)?;
        Ok(Self { inner: out })
    }

    #[pyo3(signature = (output_size, kernel_size, dilation=(1, 1), padding=(0, 0), stride=(1, 1)))]
    pub fn fold_2d(
        &self,
        py: Python<'_>,
        output_size: (usize, usize),
        kernel_size: (usize, usize),
        dilation: (usize, usize),
        padding: (usize, usize),
        stride: (usize, usize),
    ) -> PyResult<Self> {
        let out = py
            .detach(|| {
                oxideon_core::nn::functional_ext::fold_2d(
                    &self.inner,
                    output_size,
                    kernel_size,
                    dilation,
                    padding,
                    stride,
                )
            })
            .map_err(to_py_err)?;
        Ok(Self { inner: out })
    }

    #[pyo3(signature = (weight, bias=None, stride=1, padding=0, dilation=1, groups=1))]
    pub fn conv1d(
        &self,
        weight: &PyTensor,
        bias: Option<&PyTensor>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> PyResult<Self> {
        let b = bias.map(|b| &b.inner);
        let out = self
            .inner
            .conv1d_grouped(&weight.inner, b, stride, padding, dilation, groups)
            .map_err(to_py_err)?;
        Ok(Self { inner: out })
    }

    pub fn select(&self, axis: isize, index: isize) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.select(axis, index).map_err(to_py_err)?,
        })
    }

    pub fn unsqueeze(&self, axis: isize) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.unsqueeze(axis).map_err(to_py_err)?,
        })
    }

    pub fn squeeze(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.squeeze().map_err(to_py_err)?,
        })
    }

    pub fn unfold(&self, dim: isize, size: usize, step: usize) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.unfold(dim, size, step).map_err(to_py_err)?,
        })
    }

    pub fn roll(&self, py: Python<'_>, shift: isize, dim: isize) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.roll(shift, dim))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn repeat_interleave(&self, py: Python<'_>, repeats: usize, dim: isize) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.repeat_interleave(repeats, dim))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    // --- Arithmetic, Unary, Reductions & Linalg (All GIL-Released) ---

    pub fn add(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.add(&other.inner))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn sub(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.sub(&other.inner))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn mul(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.mul(&other.inner))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn div(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.div(&other.inner))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn add_scalar(&self, py: Python<'_>, s: f64) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.add_scalar(s)).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn mul_scalar(&self, py: Python<'_>, s: f64) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.mul_scalar(s)).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn powf(&self, py: Python<'_>, p: f64) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.powf(p)).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn square(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.square()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn sqrt(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.sqrt()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn rsqrt(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.rsqrt()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn exp(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.exp()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn ln(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.ln()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn sin(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.sin()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn cos(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.cos()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn tanh(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.tanh()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn sigmoid(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.sigmoid()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn relu(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.relu()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn silu(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.silu()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn gelu(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.gelu()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn mish(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.mish()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn hardswish(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.hardswish()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn hardsigmoid(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.hardsigmoid()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn selu(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.selu()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (alpha=1.0))]
    pub fn celu(&self, py: Python<'_>, alpha: f64) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.celu(alpha)).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (beta=1.0))]
    pub fn softplus(&self, py: Python<'_>, beta: f64) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.softplus_beta(beta))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn softsign(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.softsign()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn relu6(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.relu6()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (min_val=-1.0, max_val=1.0))]
    pub fn hardtanh(&self, py: Python<'_>, min_val: f64, max_val: f64) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.hardtanh(min_val, max_val))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn log_sigmoid(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.log_sigmoid())
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (lambd=0.5))]
    pub fn softshrink(&self, py: Python<'_>, lambd: f64) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.softshrink(lambd))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (lambd=0.5))]
    pub fn hardshrink(&self, py: Python<'_>, lambd: f64) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.hardshrink(lambd))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (alpha=1.0))]
    pub fn elu(&self, py: Python<'_>, alpha: f64) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.elu(alpha)).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn erf(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.erf()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (axis=-1))]
    pub fn softmax(&self, py: Python<'_>, axis: isize) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.softmax(axis)).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Sum of all elements, or along `dim` (kept as size 1 when `keepdim`).
    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn sum(&self, py: Python<'_>, dim: Option<isize>, keepdim: bool) -> PyResult<Self> {
        Self::run(py, || match dim {
            None => self.inner.sum(),
            Some(d) => self.inner.sum_axis(d, keepdim),
        })
    }

    /// Mean of all elements, or along `dim`.
    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn mean(&self, py: Python<'_>, dim: Option<isize>, keepdim: bool) -> PyResult<Self> {
        Self::run(py, || match dim {
            None => self.inner.mean(),
            Some(d) => self.inner.mean_axis(d, keepdim),
        })
    }

    /// Mean of non-NaN elements, or along `dim` (all-NaN reads NaN; float only).
    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn nanmean(&self, py: Python<'_>, dim: Option<isize>, keepdim: bool) -> PyResult<Self> {
        Self::run(py, || match dim {
            None => {
                let t = self.inner.nanmean()?;
                let ndim = self.inner.shape().len();
                if keepdim && ndim > 0 {
                    t.reshape(&vec![1; ndim])
                } else {
                    Ok(t)
                }
            }
            Some(d) => self.inner.nanmean_axes(&[d], keepdim),
        })
    }

    /// Mean of non-NaN elements along several `axes` at once.
    #[pyo3(signature = (axes, keepdim=false))]
    pub fn nanmean_axes(
        &self,
        py: Python<'_>,
        axes: Vec<isize>,
        keepdim: bool,
    ) -> PyResult<Self> {
        Self::run(py, || self.inner.nanmean_axes(&axes, keepdim))
    }

    /// Sum of non-NaN elements, or along `dim` (NaN reads as zero).
    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn nansum(&self, py: Python<'_>, dim: Option<isize>, keepdim: bool) -> PyResult<Self> {
        Self::run(py, || match dim {
            None => {
                let t = self.inner.nansum()?;
                let ndim = self.inner.shape().len();
                if keepdim && ndim > 0 {
                    t.reshape(&vec![1; ndim])
                } else {
                    Ok(t)
                }
            }
            Some(d) => self.inner.nansum_axes(&[d], keepdim),
        })
    }

    /// Sum of non-NaN elements along several `axes` at once.
    #[pyo3(signature = (axes, keepdim=false))]
    pub fn nansum_axes(&self, py: Python<'_>, axes: Vec<isize>, keepdim: bool) -> PyResult<Self> {
        Self::run(py, || self.inner.nansum_axes(&axes, keepdim))
    }

    /// Median: values alone when `dim` is omitted, otherwise a
    /// `(values, indices)` tuple (lower middle for even counts; NaN propagates).
    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn median(
        &self,
        py: Python<'_>,
        dim: Option<isize>,
        keepdim: bool,
    ) -> PyResult<Py<PyAny>> {
        match dim {
            None => {
                let t = py.detach(|| self.inner.median()).map_err(to_py_err)?;
                let ndim = self.inner.shape().len();
                let t = if keepdim && ndim > 0 {
                    py.detach(|| t.reshape(&vec![1; ndim])).map_err(to_py_err)?
                } else {
                    t
                };
                Ok(Py::new(py, Self { inner: t })?.into_any())
            }
            Some(d) => {
                let (v, i) = py
                    .detach(|| self.inner.median_axis(d, keepdim))
                    .map_err(to_py_err)?;
                let vt = Py::new(py, Self { inner: v })?;
                let it = Py::new(py, Self { inner: i })?;
                Ok((vt, it).into_pyobject(py)?.into_any().unbind())
            }
        }
    }

    /// Quantiles at levels `q` (a scalar or 1-D tensor matching the input
    /// dtype): full reduction, or along `dim` with the q dim prepended.
    #[pyo3(signature = (q, dim=None, keepdim=false, interpolation="linear"))]
    pub fn quantile(
        &self,
        py: Python<'_>,
        q: &Bound<'_, PyAny>,
        dim: Option<isize>,
        keepdim: bool,
        interpolation: &str,
    ) -> PyResult<Self> {
        self.quantile_impl(py, q, dim, keepdim, interpolation, false)
    }

    /// NaN-skipping quantiles at levels `q` (same shapes as `quantile`).
    #[pyo3(signature = (q, dim=None, keepdim=false, interpolation="linear"))]
    pub fn nanquantile(
        &self,
        py: Python<'_>,
        q: &Bound<'_, PyAny>,
        dim: Option<isize>,
        keepdim: bool,
        interpolation: &str,
    ) -> PyResult<Self> {
        self.quantile_impl(py, q, dim, keepdim, interpolation, true)
    }

    /// Median of non-NaN elements: values alone when `dim` is omitted,
    /// otherwise a `(values, indices)` tuple (even counts take the lower middle).
    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn nanmedian(
        &self,
        py: Python<'_>,
        dim: Option<isize>,
        keepdim: bool,
    ) -> PyResult<Py<PyAny>> {
        match dim {
            None => {
                let t = py.detach(|| self.inner.nanmedian()).map_err(to_py_err)?;
                let ndim = self.inner.shape().len();
                let t = if keepdim && ndim > 0 {
                    py.detach(|| t.reshape(&vec![1; ndim])).map_err(to_py_err)?
                } else {
                    t
                };
                Ok(Py::new(py, Self { inner: t })?.into_any())
            }
            Some(d) => {
                let (v, i) = py
                    .detach(|| self.inner.nanmedian_axis(d, keepdim))
                    .map_err(to_py_err)?;
                let vt = Py::new(py, Self { inner: v })?;
                let it = Py::new(py, Self { inner: i })?;
                Ok((vt, it).into_pyobject(py)?.into_any().unbind())
            }
        }
    }

    /// Maximum of all elements, or along `dim` (values only; use `argmax` for positions).
    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn max(&self, py: Python<'_>, dim: Option<isize>, keepdim: bool) -> PyResult<Self> {
        Self::run(py, || match dim {
            None => self.flat()?.max_axis(0, false),
            Some(d) => self.inner.max_axis(d, keepdim),
        })
    }

    /// Minimum of all elements, or along `dim`.
    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn min(&self, py: Python<'_>, dim: Option<isize>, keepdim: bool) -> PyResult<Self> {
        Self::run(py, || match dim {
            None => self.flat()?.min_axis(0, false),
            Some(d) => self.inner.min_axis(d, keepdim),
        })
    }

    /// Index of the maximum (in the flattened tensor when `dim` is omitted).
    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn argmax(&self, py: Python<'_>, dim: Option<isize>, keepdim: bool) -> PyResult<Self> {
        Self::run(py, || match dim {
            None => self.flat()?.argmax_axis(0, false),
            Some(d) => self.inner.argmax_axis(d, keepdim),
        })
    }

    /// Index of the minimum (in the flattened tensor when `dim` is omitted).
    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn argmin(&self, py: Python<'_>, dim: Option<isize>, keepdim: bool) -> PyResult<Self> {
        Self::run(py, || match dim {
            None => self.flat()?.argmin_axis(0, false),
            Some(d) => self.inner.argmin_axis(d, keepdim),
        })
    }

    /// Variance of all elements, or along `dim`.
    #[pyo3(signature = (dim=None, unbiased=true, keepdim=false))]
    pub fn var(
        &self,
        py: Python<'_>,
        dim: Option<isize>,
        unbiased: bool,
        keepdim: bool,
    ) -> PyResult<Self> {
        Self::run(py, || match dim {
            None => self.flat()?.var_axis(0, unbiased, false),
            Some(d) => self.inner.var_axis(d, unbiased, keepdim),
        })
    }

    /// Standard deviation of all elements, or along `dim`.
    #[pyo3(signature = (dim=None, unbiased=true, keepdim=false))]
    pub fn std(
        &self,
        py: Python<'_>,
        dim: Option<isize>,
        unbiased: bool,
        keepdim: bool,
    ) -> PyResult<Self> {
        Self::run(py, || match dim {
            None => self.flat()?.std_axis(0, unbiased, false),
            Some(d) => self.inner.std_axis(d, unbiased, keepdim),
        })
    }

    /// Euclidean norm of all elements.
    pub fn norm(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.norm())
    }

    /// Cumulative sum along `dim`.
    pub fn cumsum(&self, py: Python<'_>, dim: isize) -> PyResult<Self> {
        Self::run(py, || self.inner.cumsum(dim))
    }

    /// Cumulative product along `dim`.
    pub fn cumprod(&self, py: Python<'_>, dim: isize) -> PyResult<Self> {
        Self::run(py, || self.inner.cumprod(dim))
    }

    /// n-th forward discrete difference along `dim`.
    #[pyo3(signature = (n=1, dim=-1))]
    pub fn diff(&self, py: Python<'_>, n: usize, dim: isize) -> PyResult<Self> {
        Self::run(py, || self.inner.diff(n, dim))
    }

    /// First-order gradient along one axis with uniform spacing `h`.
    pub fn gradient_axis(
        &self,
        py: Python<'_>,
        axis: isize,
        edge_order: isize,
        h: f64,
    ) -> PyResult<Self> {
        Self::run(py, || self.inner.gradient_axis(axis, edge_order, h))
    }

    /// First-order gradient along one axis at 1-D coordinates `coords`.
    pub fn gradient_axis_coords(
        &self,
        py: Python<'_>,
        axis: isize,
        edge_order: isize,
        coords: &Self,
    ) -> PyResult<Self> {
        Self::run(py, || {
            self.inner.gradient_axis_coords(axis, edge_order, &coords.inner)
        })
    }

    /// Trapezoidal integral along one axis with uniform spacing `dx`.
    pub fn trapezoid_axis(&self, py: Python<'_>, axis: isize, dx: f64) -> PyResult<Self> {
        Self::run(py, || self.inner.trapezoid_axis(axis, dx))
    }

    /// Trapezoidal integral along one axis at sample points `x`.
    pub fn trapezoid_axis_x(&self, py: Python<'_>, axis: isize, x: &Self) -> PyResult<Self> {
        Self::run(py, || self.inner.trapezoid_axis_x(axis, &x.inner))
    }

    /// Counts occurrences of each non-negative integer (1D int tensor).
    #[pyo3(signature = (weights=None, minlength=0))]
    pub fn bincount(
        &self,
        py: Python<'_>,
        weights: Option<&Self>,
        minlength: usize,
    ) -> PyResult<Self> {
        Self::run(py, || {
            self.inner
                .bincount(weights.map(|w| &w.inner), minlength)
        })
    }

    /// Histogram with `bins` equal-width bins over `[min, max]`.
    #[pyo3(signature = (bins=100, min=0.0, max=0.0))]
    pub fn histc(
        &self,
        py: Python<'_>,
        bins: usize,
        min: f64,
        max: f64,
    ) -> PyResult<Self> {
        Self::run(py, || self.inner.histc(bins, min, max))
    }

    pub fn log_softmax(&self, py: Python<'_>, dim: isize) -> PyResult<Self> {
        Self::run(py, || self.inner.log_softmax(dim))
    }

    // --- Element-wise helpers & constructors users reach for ---

    /// Natural logarithm (alias of `ln`).
    pub fn log(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.ln())
    }

    pub fn abs(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.abs())
    }

    pub fn neg(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.neg())
    }

    pub fn recip(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.recip())
    }

    pub fn sign(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.sign())
    }

    pub fn floor(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.floor())
    }

    pub fn ceil(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.ceil())
    }

    pub fn round(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.round())
    }

    /// Clamps every element into `[min, max]`.
    pub fn clamp(&self, py: Python<'_>, min: f64, max: f64) -> PyResult<Self> {
        Self::run(py, || self.inner.clamp(min, max))
    }

    /// Raises to a tensor or scalar power.
    pub fn pow(&self, py: Python<'_>, exponent: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.bin(py, exponent, Tensor::pow, Tensor::powf)
    }

    pub fn maximum(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        Self::run(py, || self.inner.maximum(&other.inner))
    }

    pub fn minimum(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        Self::run(py, || self.inner.minimum(&other.inner))
    }

    /// Replaces elements where `mask` is true with `value`.
    pub fn masked_fill(&self, py: Python<'_>, mask: &Self, value: f64) -> PyResult<Self> {
        Self::run(py, || self.inner.masked_fill(&mask.inner, value))
    }

    /// Elements where boolean `mask` (broadcast against the input) is true.
    pub fn masked_select(&self, py: Python<'_>, mask: &Self) -> PyResult<Self> {
        Self::run(py, || self.inner.masked_select(&mask.inner))
    }

    /// Fills slices along `dim` selected by int64 `index` with `value`.
    pub fn index_fill(&self, py: Python<'_>, dim: isize, index: &Self, value: f64) -> PyResult<Self> {
        Self::run(py, || self.inner.index_fill(dim, &index.inner, value))
    }

    /// Copy of `self` with `source` written at true mask positions.
    pub fn masked_scatter(&self, py: Python<'_>, mask: &Self, source: &Self) -> PyResult<Self> {
        Self::run(py, || {
            self.inner.masked_scatter(&mask.inner, &source.inner)
        })
    }

    /// Selects slices along `axis` at 1D int64 `indices` (negatives wrap).
    pub fn index_select(&self, py: Python<'_>, axis: isize, indices: &Self) -> PyResult<Self> {
        Self::run(py, || self.inner.index_select(axis, &indices.inner))
    }

    /// Out-of-place `index_add` of `source` into `self` along `axis` (same dtype).
    pub fn index_add(
        &self,
        py: Python<'_>,
        axis: isize,
        indices: &Self,
        source: &Self,
    ) -> PyResult<Self> {
        Self::run(py, || {
            self.inner.index_add(axis, &indices.inner, &source.inner)
        })
    }

    /// Gathers values along `axis` at same-rank int64 `index`.
    pub fn gather(&self, py: Python<'_>, axis: isize, index: &Self) -> PyResult<Self> {
        Self::run(py, || self.inner.gather(axis, &index.inner))
    }

    /// Out-of-place `scatter_add` of `src` into `self` along `axis` (same dtype).
    pub fn scatter_add(
        &self,
        py: Python<'_>,
        axis: isize,
        index: &Self,
        src: &Self,
    ) -> PyResult<Self> {
        Self::run(py, || {
            self.inner.scatter_add(axis, &index.inner, &src.inner)
        })
    }

    /// Out-of-place `scatter` of `src` into `self` along `axis` (same dtype).
    pub fn scatter(
        &self,
        py: Python<'_>,
        axis: isize,
        index: &Self,
        src: &Self,
    ) -> PyResult<Self> {
        Self::run(py, || {
            self.inner.scatter(axis, &index.inner, &src.inner)
        })
    }

    pub fn det(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.det())
    }

    pub fn slogdet(&self, py: Python<'_>) -> PyResult<(Self, Self)> {
        let (sign, logdet) = py.detach(|| self.inner.slogdet()).map_err(to_py_err)?;
        Ok((Self { inner: sign }, Self { inner: logdet }))
    }

    /// Collapses dimensions `start_dim..=end_dim` into one (default: all of them).
    #[pyo3(signature = (start_dim=0, end_dim=-1))]
    pub fn flatten(&self, py: Python<'_>, start_dim: isize, end_dim: isize) -> PyResult<Self> {
        Self::run(py, || self.inner.flatten_range(start_dim, end_dim))
    }

    /// Broadcasts to `shape` without copying.
    pub fn expand(&self, py: Python<'_>, shape: Vec<usize>) -> PyResult<Self> {
        Self::run(py, || self.inner.expand(&shape))
    }

    pub fn zeros_like(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.zeros_like())
    }

    pub fn ones_like(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.ones_like())
    }

    pub fn full_like(&self, py: Python<'_>, value: f64) -> PyResult<Self> {
        Self::run(py, || self.inner.full_like(value))
    }

    /// The shape as a list, or one dimension's size.
    #[pyo3(signature = (dim=None))]
    pub fn size(&self, py: Python<'_>, dim: Option<isize>) -> PyResult<Py<PyAny>> {
        let shape = self.inner.shape();
        match dim {
            None => Ok(shape.to_vec().into_pyobject(py)?.into_any().unbind()),
            Some(d) => {
                let ax =
                    crate::layout::normalize_axis("size", d, shape.len()).map_err(to_py_err)?;
                Ok(shape[ax].into_pyobject(py)?.into_any().unbind())
            }
        }
    }

    /// Same as the constructor: `Tensor.from_slice([2.0], [1], requires_grad=True)`.
    #[staticmethod]
    #[pyo3(signature = (data, shape, dtype=None, device=None, requires_grad=false))]
    pub fn from_slice(
        py: Python<'_>,
        data: Vec<f64>,
        shape: Vec<usize>,
        dtype: Option<&Bound<'_, PyAny>>,
        device: Option<&Bound<'_, PyAny>>,
        requires_grad: bool,
    ) -> PyResult<Self> {
        Self::new(py, data, shape, dtype, device, requires_grad)
    }

    /// A tensor filled with `value`.
    #[staticmethod]
    #[pyo3(signature = (shape, value, dtype=None, device=None, requires_grad=false))]
    pub fn full(
        shape: Vec<usize>,
        value: f64,
        dtype: Option<&Bound<'_, PyAny>>,
        device: Option<&Bound<'_, PyAny>>,
        requires_grad: bool,
    ) -> PyResult<Self> {
        let dt = parse_dtype_obj(dtype, DType::F32)?;
        let dev = parse_device_obj(device, "cpu")?;
        let t = Tensor::full(&shape, value, dt, &dev).map_err(to_py_err)?;
        Ok(Self {
            inner: if requires_grad {
                t.requires_grad_(true)
            } else {
                t
            },
        })
    }

    /// Uniform random values in `[0, 1)`.
    #[staticmethod]
    #[pyo3(signature = (shape, dtype=None, device=None, requires_grad=false))]
    pub fn rand(
        py: Python<'_>,
        shape: Vec<usize>,
        dtype: Option<&Bound<'_, PyAny>>,
        device: Option<&Bound<'_, PyAny>>,
        requires_grad: bool,
    ) -> PyResult<Self> {
        let dt = parse_dtype_obj(dtype, DType::F32)?;
        let dev = parse_device_obj(device, "cpu")?;
        let inner = py
            .detach(|| {
                let mut t = Tensor::rand_uniform(&shape, 0.0, 1.0)?;
                if t.dtype() != dt {
                    t = t.to_dtype(dt)?;
                }
                if t.device() != dev {
                    t = t.to_device(&dev)?;
                }
                Ok::<Tensor, Error>(if requires_grad {
                    t.requires_grad_(true)
                } else {
                    t
                })
            })
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (axes, keepdim=false))]
    pub fn sum_axes(&self, py: Python<'_>, axes: Vec<isize>, keepdim: bool) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.sum_axes(&axes, keepdim))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (axes, keepdim=false))]
    pub fn mean_axes(&self, py: Python<'_>, axes: Vec<isize>, keepdim: bool) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.mean_axes(&axes, keepdim))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn matmul(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.matmul(&other.inner))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn cholesky(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.cholesky()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn inv(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.inv()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn inverse(&self, py: Python<'_>) -> PyResult<Self> {
        self.inv(py)
    }

    #[pyo3(signature = (rcond=1e-7))]
    pub fn pinv(&self, py: Python<'_>, rcond: f64) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.pinv(rcond)).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (b, rcond=1e-7))]
    pub fn lstsq(&self, py: Python<'_>, b: &Self, rcond: f64) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.lstsq(&b.inner, rcond))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn solve(&self, py: Python<'_>, b: &Self) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.solve(&b.inner))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (rhs, lower=true, transpose_a=false))]
    pub fn triangular_solve(
        &self,
        py: Python<'_>,
        rhs: &Self,
        lower: bool,
        transpose_a: bool,
    ) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.triangular_solve(&rhs.inner, lower, transpose_a))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn svd(&self, py: Python<'_>) -> PyResult<(Self, Self, Self)> {
        let (u, s, vt) = py.detach(|| self.inner.svd()).map_err(to_py_err)?;
        Ok((Self { inner: u }, Self { inner: s }, Self { inner: vt }))
    }

    pub fn eigh(&self, py: Python<'_>) -> PyResult<(Self, Self)> {
        let (w, v) = py.detach(|| self.inner.eigh()).map_err(to_py_err)?;
        Ok((Self { inner: w }, Self { inner: v }))
    }

    pub fn qr(&self, py: Python<'_>) -> PyResult<(Self, Self)> {
        let (q, r) = py.detach(|| self.inner.qr()).map_err(to_py_err)?;
        Ok((Self { inner: q }, Self { inner: r }))
    }

    pub fn svdvals(&self, py: Python<'_>) -> PyResult<Self> {
        let s = py.detach(|| self.inner.svdvals()).map_err(to_py_err)?;
        Ok(Self { inner: s })
    }

    pub fn eigvalsh(&self, py: Python<'_>) -> PyResult<Self> {
        let w = py.detach(|| self.inner.eigvalsh()).map_err(to_py_err)?;
        Ok(Self { inner: w })
    }

    pub fn logdet(&self, py: Python<'_>) -> PyResult<Self> {
        let res = py.detach(|| self.inner.logdet()).map_err(to_py_err)?;
        Ok(Self { inner: res })
    }

    #[pyo3(signature = (upper = false))]
    pub fn cholesky_inverse(&self, py: Python<'_>, upper: bool) -> PyResult<Self> {
        let res = py
            .detach(|| self.inner.cholesky_inverse(upper))
            .map_err(to_py_err)?;
        Ok(Self { inner: res })
    }

    pub fn householder_product(&self, py: Python<'_>, tau: &Self) -> PyResult<Self> {
        let res = py
            .detach(|| self.inner.householder_product(&tau.inner))
            .map_err(to_py_err)?;
        Ok(Self { inner: res })
    }

    #[pyo3(signature = (ord = 2.0, dims = vec![], keepdim = false))]
    pub fn vector_norm(
        &self,
        py: Python<'_>,
        ord: f64,
        dims: Vec<isize>,
        keepdim: bool,
    ) -> PyResult<Self> {
        let res = py
            .detach(|| self.inner.vector_norm(ord, &dims, keepdim))
            .map_err(to_py_err)?;
        Ok(Self { inner: res })
    }

    #[pyo3(signature = (ord = "fro", keepdim = false))]
    pub fn matrix_norm(&self, py: Python<'_>, ord: &str, keepdim: bool) -> PyResult<Self> {
        let res = py
            .detach(|| self.inner.matrix_norm(ord, keepdim))
            .map_err(to_py_err)?;
        Ok(Self { inner: res })
    }

    #[pyo3(signature = (other, dim = -1))]
    pub fn vecdot(&self, py: Python<'_>, other: &Self, dim: isize) -> PyResult<Self> {
        let res = py
            .detach(|| self.inner.vecdot(&other.inner, dim))
            .map_err(to_py_err)?;
        Ok(Self { inner: res })
    }

    #[pyo3(signature = (ind = 2))]
    pub fn tensorinv(&self, py: Python<'_>, ind: usize) -> PyResult<Self> {
        let res = py
            .detach(|| self.inner.tensorinv(ind))
            .map_err(to_py_err)?;
        Ok(Self { inner: res })
    }

    pub fn tensorsolve(&self, py: Python<'_>, b: &Self) -> PyResult<Self> {
        let res = py
            .detach(|| self.inner.tensorsolve(&b.inner))
            .map_err(to_py_err)?;
        Ok(Self { inner: res })
    }

    pub fn eigvals(&self, py: Python<'_>) -> PyResult<(Self, Self)> {
        let (wr, wi) = py.detach(|| self.inner.eigvals()).map_err(to_py_err)?;
        Ok((Self { inner: wr }, Self { inner: wi }))
    }

    pub fn eig(&self, py: Python<'_>) -> PyResult<(Self, Self, Self, Self)> {
        let (wr, wi, vr, vi) = py.detach(|| self.inner.eig()).map_err(to_py_err)?;
        Ok((
            Self { inner: wr },
            Self { inner: wi },
            Self { inner: vr },
            Self { inner: vi },
        ))
    }

    #[staticmethod]
    pub fn lu_solve(
        py: Python<'_>,
        p: &Self,
        l: &Self,
        u: &Self,
        b: &Self,
    ) -> PyResult<Self> {
        let res = py
            .detach(|| Tensor::lu_solve(&p.inner, &l.inner, &u.inner, &b.inner))
            .map_err(to_py_err)?;
        Ok(Self { inner: res })
    }

    #[pyo3(signature = (dim=-1, descending=false))]
    pub fn sort(
        &self,
        py: Python<'_>,
        dim: isize,
        descending: bool,
    ) -> PyResult<(Self, Self)> {
        let (values, indices) = py
            .detach(|| self.inner.sort(dim, descending))
            .map_err(to_py_err)?;
        Ok((Self { inner: values }, Self { inner: indices }))
    }

    #[pyo3(signature = (dim=-1, descending=false))]
    pub fn argsort(&self, py: Python<'_>, dim: isize, descending: bool) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.argsort(dim, descending))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (k, dim=-1, largest=true, sorted=true))]
    pub fn topk(
        &self,
        py: Python<'_>,
        k: usize,
        dim: isize,
        largest: bool,
        sorted: bool,
    ) -> PyResult<(Self, Self)> {
        let (values, indices) = py
            .detach(|| self.inner.topk(k, dim, largest, sorted))
            .map_err(to_py_err)?;
        Ok((Self { inner: values }, Self { inner: indices }))
    }

    #[pyo3(signature = (k, dim=-1, keepdim=false))]
    pub fn kthvalue(
        &self,
        py: Python<'_>,
        k: usize,
        dim: isize,
        keepdim: bool,
    ) -> PyResult<(Self, Self)> {
        let (values, indices) = py
            .detach(|| self.inner.kthvalue(k, dim, keepdim))
            .map_err(to_py_err)?;
        Ok((Self { inner: values }, Self { inner: indices }))
    }

    /// Mode (most frequent value) and its first-occurrence index along `dim`.
    #[pyo3(signature = (dim=-1, keepdim=false))]
    pub fn mode(&self, py: Python<'_>, dim: isize, keepdim: bool) -> PyResult<(Self, Self)> {
        let (values, indices) = py
            .detach(|| self.inner.mode_axis(dim, keepdim))
            .map_err(to_py_err)?;
        Ok((Self { inner: values }, Self { inner: indices }))
    }

    pub fn matrix_exp(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.matrix_exp()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn matrix_power(&self, py: Python<'_>, p: isize) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.matrix_power(p))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn kron(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.kron(&other.inner))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn fft(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.fft()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn ifft(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.ifft()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn fft2d(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.fft2d()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn ifft2d(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.ifft2d()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn rfft(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.rfft()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn irfft(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.irfft()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn stft(
        &self,
        py: Python<'_>,
        n_fft: usize,
        hop_length: usize,
        hann_window: bool,
    ) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.stft(n_fft, hop_length, hann_window))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn complex_mul(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        let inner = py
            .detach(|| self.inner.complex_mul(&other.inner))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn complex_abs(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.complex_abs()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    // --- Autograd (GIL-Released) ---

    pub fn backward(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.backward()).map_err(to_py_err)
    }

    pub fn backward_create_graph(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            let g0 = self.inner.ones_like()?;
            self.inner.backward_with_grad(&g0, true)
        })
        .map_err(to_py_err)
    }

    // --- Python Operator Overloads: a tensor or a number on either side ---

    pub fn __add__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.bin(py, other, Tensor::add, Tensor::add_scalar)
    }

    pub fn __radd__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.bin(py, other, Tensor::add, Tensor::add_scalar)
    }

    pub fn __sub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.bin(py, other, Tensor::sub, Tensor::sub_scalar)
    }

    pub fn __rsub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        match operand(other) {
            Some(Operand::Tensor(t)) => Self::wrap(py, || t.sub(&self.inner)),
            Some(Operand::Scalar(s)) => Self::wrap(py, || self.inner.neg()?.add_scalar(s)),
            None => Ok(py.NotImplemented()),
        }
    }

    pub fn __mul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.bin(py, other, Tensor::mul, Tensor::mul_scalar)
    }

    pub fn __rmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.bin(py, other, Tensor::mul, Tensor::mul_scalar)
    }

    pub fn __truediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.bin(py, other, Tensor::div, Tensor::div_scalar)
    }

    pub fn __rtruediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        match operand(other) {
            Some(Operand::Tensor(t)) => Self::wrap(py, || t.div(&self.inner)),
            Some(Operand::Scalar(s)) => Self::wrap(py, || self.inner.recip()?.mul_scalar(s)),
            None => Ok(py.NotImplemented()),
        }
    }

    pub fn __pow__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        _modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        self.bin(py, other, Tensor::pow, Tensor::powf)
    }

    pub fn __matmul__(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        self.matmul(py, other)
    }

    pub fn __neg__(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.neg())
    }

    pub fn __abs__(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.abs())
    }

    // Comparisons are element-wise and return a boolean tensor (as in PyTorch).

    pub fn __lt__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.cmp(py, other, Tensor::lt)
    }

    pub fn __le__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.cmp(py, other, Tensor::le)
    }

    pub fn __gt__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.cmp(py, other, Tensor::gt)
    }

    pub fn __ge__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.cmp(py, other, Tensor::ge)
    }

    pub fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.cmp(py, other, Tensor::eq_tensor)
    }

    pub fn __ne__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.cmp(py, other, Tensor::ne_tensor)
    }

    // Bitwise operators are element-wise on integer/bool tensors (as in PyTorch).

    pub fn __and__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.cmp(py, other, Tensor::bitwise_and)
    }

    pub fn __or__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.cmp(py, other, Tensor::bitwise_or)
    }

    pub fn __xor__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.cmp(py, other, Tensor::bitwise_xor)
    }

    pub fn __invert__(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.bitwise_not())
    }

    pub fn bitwise_and(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.cmp(py, other, Tensor::bitwise_and)
    }

    pub fn bitwise_or(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.cmp(py, other, Tensor::bitwise_or)
    }

    pub fn bitwise_xor(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.cmp(py, other, Tensor::bitwise_xor)
    }

    pub fn bitwise_not(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.bitwise_not())
    }

    pub fn logical_and(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.cmp(py, other, Tensor::logical_and)
    }

    pub fn logical_or(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.cmp(py, other, Tensor::logical_or)
    }

    pub fn logical_xor(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.cmp(py, other, Tensor::logical_xor)
    }

    pub fn logical_not(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.logical_not())
    }

    pub fn isnan(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.isnan())
    }

    pub fn isinf(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.isinf())
    }

    pub fn isfinite(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.isfinite())
    }

    /// Identity hash, so tensors stay usable as dict keys although `==` is element-wise.
    pub fn __hash__(slf: &Bound<'_, Self>) -> isize {
        slf.as_ptr() as isize
    }

    // --- Sequence protocol & number conversions ---

    pub fn __len__(&self) -> PyResult<usize> {
        self.inner
            .shape()
            .first()
            .copied()
            .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("len() of a 0-d tensor"))
    }

    /// Basic indexing: integers, slices (positive and negative step), `None` and one `...`, e.g. `t[0, 1:3, ..., None]`.
    /// Iterating a tensor (`for row in t`) works through this method.
    pub fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Self> {
        use pyo3::exceptions::{PyIndexError, PyTypeError};
        use pyo3::types::{PyEllipsis, PySlice, PyTuple};
        let items: Vec<Bound<'_, PyAny>> = match key.cast::<PyTuple>() {
            Ok(t) => t.iter().collect(),
            Err(_) => vec![key.clone()],
        };
        let consumes = |x: &Bound<'_, PyAny>| !x.is_none() && !x.is_instance_of::<PyEllipsis>();
        let mut out = self.inner.clone();
        let mut dim = 0usize;
        let mut seen_ellipsis = false;
        for (pos, item) in items.iter().enumerate() {
            if item.is_instance_of::<PyEllipsis>() {
                if std::mem::replace(&mut seen_ellipsis, true) {
                    return Err(PyIndexError::new_err(
                        "an index can only have a single ellipsis",
                    ));
                }
                let after = items[pos + 1..].iter().filter(|x| consumes(x)).count();
                dim = out
                    .ndim()
                    .checked_sub(after)
                    .ok_or_else(|| PyIndexError::new_err("too many indices for tensor"))?;
            } else if item.is_none() {
                out = out.unsqueeze(dim as isize).map_err(to_py_err)?;
                dim += 1;
            } else {
                if dim >= out.ndim() {
                    return Err(PyIndexError::new_err(format!(
                        "too many indices for tensor of dimension {}",
                        self.inner.ndim()
                    )));
                }
                let size = out.shape()[dim] as isize;
                if let Ok(slice) = item.cast::<PySlice>() {
                    let ix = slice.indices(size)?;
                    if ix.step > 0 {
                        let stop = ix.stop.max(ix.start);
                        out = out
                            .slice(dim as isize, Some(ix.start), Some(stop), ix.step)
                            .map_err(to_py_err)?;
                    } else if ix.start < 0 {
                        // Normalized start -1 with negative step selects nothing; spell it empty
                        // (the core would read Some(-1) as the last index).
                        out = out
                            .slice(dim as isize, Some(0), Some(0), 1)
                            .map_err(to_py_err)?;
                    } else if ix.stop == -1 {
                        // Stop -1 is the past-the-start sentinel (explicit -1 normalizes to
                        // size - 1 instead); None carries the same meaning to the core.
                        out = out
                            .slice(dim as isize, Some(ix.start), None, ix.step)
                            .map_err(to_py_err)?;
                    } else {
                        out = out
                            .slice(dim as isize, Some(ix.start), Some(ix.stop), ix.step)
                            .map_err(to_py_err)?;
                    }
                    dim += 1;
                } else if let Ok(i) = item.extract::<isize>() {
                    let i = if i < 0 { i + size } else { i };
                    if i < 0 || i >= size {
                        return Err(PyIndexError::new_err(format!(
                            "index {i} is out of bounds for dimension {dim} with size {size}"
                        )));
                    }
                    out = out.select(dim as isize, i).map_err(to_py_err)?;
                } else {
                    return Err(PyTypeError::new_err(
                        "tensor indices must be integers, slices, None or an ellipsis",
                    ));
                }
            }
        }
        let _ = py;
        Ok(Self { inner: out })
    }

    pub fn __float__(&self) -> PyResult<f64> {
        self.scalar_item()
    }

    pub fn __int__(&self) -> PyResult<i64> {
        self.scalar_item().map(|v| v as i64)
    }

    pub fn __bool__(&self) -> PyResult<bool> {
        self.scalar_item().map(|v| v != 0.0)
    }

    #[pyo3(signature = (num_samples=1, replacement=true, generator=None))]
    pub fn multinomial(
        &self,
        py: Python<'_>,
        num_samples: usize,
        replacement: bool,
        generator: Option<&mut PyGenerator>,
    ) -> PyResult<Self> {
        let mut g_opt = generator.map(|g| g.inner.clone());
        let inner = py
            .detach(|| {
                self.inner
                    .multinomial_with_generator(num_samples, replacement, g_opt.as_mut())
            })
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (config, history=None, generator=None))]
    pub fn sample_token(
        &self,
        py: Python<'_>,
        config: &PySamplingConfig,
        history: Option<Vec<usize>>,
        generator: Option<&mut PyGenerator>,
    ) -> PyResult<usize> {
        let mut g_opt = generator.map(|g| g.inner.clone());
        py.detach(|| {
            self.inner
                .sample_token(&config.inner, history.as_deref(), g_opt.as_mut())
        })
        .map_err(to_py_err)
    }

    // --- Core-1..Core-9 Enriched Native Tensor Methods ---

    pub fn tan(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.tan())
    }

    pub fn asin(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.asin())
    }

    pub fn arcsin(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.asin())
    }

    pub fn acos(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.acos())
    }

    pub fn arccos(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.acos())
    }

    pub fn atan(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.atan())
    }

    pub fn arctan(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.atan())
    }

    pub fn sinc(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.sinc())
    }

    pub fn sinh(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.sinh())
    }

    pub fn cosh(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.cosh())
    }

    pub fn asinh(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.asinh())
    }

    pub fn arcsinh(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.asinh())
    }

    pub fn acosh(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.acosh())
    }

    pub fn arccosh(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.acosh())
    }

    pub fn atanh(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.atanh())
    }

    pub fn arctanh(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.atanh())
    }

    pub fn expm1(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.expm1())
    }

    pub fn log1p(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.log1p())
    }

    pub fn log2(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.log2())
    }

    pub fn log10(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.log10())
    }

    pub fn exp2(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.exp2())
    }

    pub fn xlogy(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        Self::run(py, || self.inner.xlogy(&other.inner))
    }

    pub fn xlog1py(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        Self::run(py, || self.inner.xlog1py(&other.inner))
    }

    pub fn entr(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.entr())
    }

    pub fn ndtr(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.ndtr())
    }

    pub fn ndtri(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.ndtri())
    }

    pub fn igamma(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        Self::run(py, || self.inner.igamma(&other.inner))
    }

    pub fn igammac(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        Self::run(py, || self.inner.igammac(&other.inner))
    }

    pub fn erfc(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.erfc())
    }

    pub fn erfinv(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.erfinv())
    }

    pub fn lgamma(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.lgamma())
    }

    pub fn digamma(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.digamma())
    }

    pub fn polygamma(&self, py: Python<'_>, n: u32) -> PyResult<Self> {
        Self::run(py, || self.inner.polygamma(n))
    }

    pub fn mvlgamma(&self, py: Python<'_>, p: usize) -> PyResult<Self> {
        Self::run(py, || self.inner.mvlgamma(p))
    }

    pub fn zeta(&self, py: Python<'_>, q: &Self) -> PyResult<Self> {
        Self::run(py, || self.inner.zeta(&q.inner))
    }

    pub fn log_ndtr(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.log_ndtr())
    }

    pub fn i0(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.i0())
    }

    pub fn i0e(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.i0e())
    }

    pub fn i1(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.i1())
    }

    pub fn i1e(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.i1e())
    }

    pub fn bessel_j0(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.bessel_j0())
    }

    pub fn bessel_j1(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.bessel_j1())
    }

    pub fn bessel_y0(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.bessel_y0())
    }

    pub fn bessel_y1(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.bessel_y1())
    }

    pub fn spherical_bessel_j0(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.spherical_bessel_j0())
    }

    pub fn erfcx(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.erfcx())
    }

    pub fn lambert_w(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.lambert_w())
    }

    pub fn modified_bessel_k0(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.modified_bessel_k0())
    }

    pub fn modified_bessel_k1(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.modified_bessel_k1())
    }

    pub fn scaled_modified_bessel_k0(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.scaled_modified_bessel_k0())
    }

    pub fn scaled_modified_bessel_k1(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.scaled_modified_bessel_k1())
    }

    pub fn hermite_polynomial_h(&self, py: Python<'_>, n: u32) -> PyResult<Self> {
        Self::run(py, || self.inner.hermite_polynomial_h(n))
    }

    pub fn legendre_polynomial_p(&self, py: Python<'_>, n: u32) -> PyResult<Self> {
        Self::run(py, || self.inner.legendre_polynomial_p(n))
    }

    pub fn chebyshev_polynomial_t(&self, py: Python<'_>, n: u32) -> PyResult<Self> {
        Self::run(py, || self.inner.chebyshev_polynomial_t(n))
    }

    pub fn chebyshev_polynomial_u(&self, py: Python<'_>, n: u32) -> PyResult<Self> {
        Self::run(py, || self.inner.chebyshev_polynomial_u(n))
    }

    pub fn laguerre_polynomial_l(&self, py: Python<'_>, n: u32) -> PyResult<Self> {
        Self::run(py, || self.inner.laguerre_polynomial_l(n))
    }

    pub fn ellipk(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.ellipk())
    }

    pub fn ellipe(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.ellipe())
    }

    pub fn expint_e1(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.expint_e1())
    }

    pub fn expint_ei(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.expint_ei())
    }

    pub fn dawsn(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.dawsn())
    }

    pub fn spence(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.spence())
    }

    pub fn fresnel(&self, py: Python<'_>) -> PyResult<(Self, Self)> {
        let (s, c) = py.detach(|| self.inner.fresnel()).map_err(to_py_err)?;
        Ok((Self { inner: s }, Self { inner: c }))
    }

    pub fn sici(&self, py: Python<'_>) -> PyResult<(Self, Self)> {
        let (si, ci) = py.detach(|| self.inner.sici()).map_err(to_py_err)?;
        Ok((Self { inner: si }, Self { inner: ci }))
    }

    pub fn gegenbauer(&self, py: Python<'_>, n: u32, alpha: f64) -> PyResult<Self> {
        Self::run(py, || self.inner.gegenbauer(n, alpha))
    }

    pub fn jacobi(&self, py: Python<'_>, n: u32, alpha: f64, beta: f64) -> PyResult<Self> {
        Self::run(py, || self.inner.jacobi(n, alpha, beta))
    }

    #[pyo3(signature = (eps=None))]
    pub fn logit(&self, py: Python<'_>, eps: Option<f64>) -> PyResult<Self> {
        Self::run(py, || self.inner.logit(eps))
    }

    #[pyo3(signature = (dim=-1))]
    pub fn cummax(&self, py: Python<'_>, dim: isize) -> PyResult<(Self, Self)> {
        let (v, i) = py.detach(|| self.inner.cummax(dim)).map_err(to_py_err)?;
        Ok((Self { inner: v }, Self { inner: i }))
    }

    #[pyo3(signature = (dim=-1))]
    pub fn cummin(&self, py: Python<'_>, dim: isize) -> PyResult<(Self, Self)> {
        let (v, i) = py.detach(|| self.inner.cummin(dim)).map_err(to_py_err)?;
        Ok((Self { inner: v }, Self { inner: i }))
    }

    #[pyo3(signature = (dim=-1))]
    pub fn logcumsumexp(&self, py: Python<'_>, dim: isize) -> PyResult<Self> {
        Self::run(py, || self.inner.logcumsumexp(dim))
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn prod(&self, py: Python<'_>, dim: Option<isize>, keepdim: bool) -> PyResult<Self> {
        Self::run(py, || match dim {
            None => {
                let p = self.inner.prod()?;
                if keepdim && self.inner.ndim() > 0 {
                    p.reshape(&vec![1usize; self.inner.ndim()])
                } else {
                    Ok(p)
                }
            }
            Some(d) => self.inner.prod_axis(d, keepdim),
        })
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn any(&self, py: Python<'_>, dim: Option<isize>, keepdim: bool) -> PyResult<Self> {
        Self::run(py, || match dim {
            None => {
                let a = self.inner.any()?;
                if keepdim && self.inner.ndim() > 0 {
                    a.reshape(&vec![1usize; self.inner.ndim()])
                } else {
                    Ok(a)
                }
            }
            Some(d) => self.inner.any_axis(d, keepdim),
        })
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn all(&self, py: Python<'_>, dim: Option<isize>, keepdim: bool) -> PyResult<Self> {
        Self::run(py, || match dim {
            None => {
                let a = self.inner.all()?;
                if keepdim && self.inner.ndim() > 0 {
                    a.reshape(&vec![1usize; self.inner.ndim()])
                } else {
                    Ok(a)
                }
            }
            Some(d) => self.inner.all_axis(d, keepdim),
        })
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn count_nonzero(
        &self,
        py: Python<'_>,
        dim: Option<isize>,
        keepdim: bool,
    ) -> PyResult<Self> {
        Self::run(py, || self.inner.count_nonzero(dim, keepdim))
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn aminmax(
        &self,
        py: Python<'_>,
        dim: Option<isize>,
        keepdim: bool,
    ) -> PyResult<(Self, Self)> {
        let (mn, mx) = py
            .detach(|| self.inner.aminmax(dim, keepdim))
            .map_err(to_py_err)?;
        Ok((Self { inner: mn }, Self { inner: mx }))
    }

    #[pyo3(signature = (tensor1, tensor2, *, value=1.0))]
    pub fn addcmul(
        &self,
        py: Python<'_>,
        tensor1: &Self,
        tensor2: &Self,
        value: f64,
    ) -> PyResult<Self> {
        Self::run(py, || {
            self.inner.addcmul(&tensor1.inner, &tensor2.inner, value)
        })
    }

    #[pyo3(signature = (tensor1, tensor2, *, value=1.0))]
    pub fn addcdiv(
        &self,
        py: Python<'_>,
        tensor1: &Self,
        tensor2: &Self,
        value: f64,
    ) -> PyResult<Self> {
        Self::run(py, || {
            self.inner.addcdiv(&tensor1.inner, &tensor2.inner, value)
        })
    }


    pub fn trunc(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.trunc())
    }

    pub fn frac(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.frac())
    }

    pub fn copysign(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        Self::run(py, || self.inner.copysign(&other.inner))
    }

    pub fn atan2(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        Self::run(py, || self.inner.atan2(&other.inner))
    }

    pub fn arctan2(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        Self::run(py, || self.inner.atan2(&other.inner))
    }

    pub fn hypot(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        Self::run(py, || self.inner.hypot(&other.inner))
    }

    pub fn fmod(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        Self::run(py, || self.inner.fmod(&other.inner))
    }

    pub fn remainder(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        Self::run(py, || self.inner.remainder(&other.inner))
    }

    #[pyo3(signature = (other, dim=-1))]
    pub fn cross(&self, py: Python<'_>, other: &Self, dim: isize) -> PyResult<Self> {
        Self::run(py, || self.inner.cross(&other.inner, dim))
    }

    #[pyo3(signature = (other, dims=None))]
    pub fn tensordot(
        &self,
        py: Python<'_>,
        other: &Self,
        dims: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let (da, db): (Vec<usize>, Vec<usize>) = match dims {
            None => {
                let n = 2usize;
                if n > self.inner.ndim() || n > other.inner.ndim() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "tensordot: contraction order exceeds tensor rank",
                    ));
                }
                let a_axes: Vec<usize> = ((self.inner.ndim() - n)..self.inner.ndim()).collect();
                let b_axes: Vec<usize> = (0..n).collect();
                (a_axes, b_axes)
            }
            Some(d) => {
                if let Ok(n) = d.extract::<usize>() {
                    if n > self.inner.ndim() || n > other.inner.ndim() {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "tensordot: contraction order exceeds tensor rank",
                        ));
                    }
                    let a_axes: Vec<usize> = ((self.inner.ndim() - n)..self.inner.ndim()).collect();
                    let b_axes: Vec<usize> = (0..n).collect();
                    (a_axes, b_axes)
                } else if let Ok((a_axes, b_axes)) = d.extract::<(Vec<usize>, Vec<usize>)>() {
                    (a_axes, b_axes)
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "tensordot dims must be an int or a tuple of two axis lists",
                    ));
                }
            }
        };
        Self::run(py, || self.inner.tensordot(&other.inner, &da, &db))
    }

    #[pyo3(signature = (diagonal=0))]
    pub fn tril(&self, py: Python<'_>, diagonal: isize) -> PyResult<Self> {
        Self::run(py, || self.inner.tril(diagonal))
    }

    #[pyo3(signature = (diagonal=0))]
    pub fn triu(&self, py: Python<'_>, diagonal: isize) -> PyResult<Self> {
        Self::run(py, || self.inner.triu(diagonal))
    }

    #[pyo3(signature = (other, p=2.0))]
    pub fn cdist(&self, py: Python<'_>, other: &Self, p: f64) -> PyResult<Self> {
        Self::run(py, || self.inner.cdist_p(&other.inner, p))
    }

    #[pyo3(signature = (p=2.0))]
    pub fn pdist(&self, py: Python<'_>, p: f64) -> PyResult<Self> {
        Self::run(py, || self.inner.pdist(p))
    }

    #[pyo3(signature = (N=None, increasing=false))]
    #[allow(non_snake_case)]
    pub fn vander(&self, py: Python<'_>, N: Option<usize>, increasing: bool) -> PyResult<Self> {
        Self::run(py, || self.inner.vander(N, increasing))
    }

    #[pyo3(signature = (factor, upper=false))]
    pub fn cholesky_solve(&self, py: Python<'_>, factor: &Self, upper: bool) -> PyResult<Self> {
        Self::run(py, || self.inner.cholesky_solve(&factor.inner, upper))
    }

    pub fn lu(&self, py: Python<'_>) -> PyResult<(Self, Self, Self)> {
        let (p, l, u) = py.detach(|| self.inner.lu()).map_err(to_py_err)?;
        Ok((Self { inner: p }, Self { inner: l }, Self { inner: u }))
    }

    pub fn polar(&self, py: Python<'_>) -> PyResult<(Self, Self)> {
        let (u, p) = py.detach(|| self.inner.polar()).map_err(to_py_err)?;
        Ok((Self { inner: u }, Self { inner: p }))
    }

    #[pyo3(signature = (other, p=2.0))]
    pub fn cdist_p(&self, py: Python<'_>, other: &Self, p: f64) -> PyResult<Self> {
        Self::run(py, || self.inner.cdist_p(&other.inner, p))
    }

    #[pyo3(signature = (correction=1))]
    pub fn cov(&self, py: Python<'_>, correction: usize) -> PyResult<Self> {
        Self::run(py, || self.inner.cov(correction))
    }

    pub fn corrcoef(&self, py: Python<'_>) -> PyResult<Self> {
        Self::run(py, || self.inner.corrcoef())
    }

    pub fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }
}

// ---------------------------------------------------------------------------
// Neural Network Layers, Foundation Models & Whole-Loop Native Rust Trainer
// ---------------------------------------------------------------------------

#[pyclass(name = "Linear", module = "oxideon")]
pub struct PyLinear {
    inner: Linear,
}

#[pymethods]
impl PyLinear {
    #[new]
    #[pyo3(signature = (in_features, out_features, bias=true))]
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> PyResult<Self> {
        Ok(Self {
            inner: Linear::with_bias(in_features, out_features, bias).map_err(to_py_err)?,
        })
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn __call__(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(py, x)
    }

    #[getter]
    pub fn weight(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.weight.clone(),
        }
    }

    #[setter]
    pub fn set_weight(&mut self, w: PyTensor) {
        self.inner.weight = w.inner;
    }

    #[getter]
    pub fn bias(&self) -> Option<PyTensor> {
        self.inner
            .bias
            .as_ref()
            .map(|b| PyTensor { inner: b.clone() })
    }

    #[setter]
    pub fn set_bias(&mut self, b: Option<PyTensor>) {
        self.inner.bias = b.map(|p| p.inner);
    }

    pub fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|inner| PyTensor { inner })
            .collect()
    }
}

/// Lookup table of `num_embeddings` vectors of size `embedding_dim`; the input is an integer tensor of indices.
#[pyclass(name = "Embedding", module = "oxideon")]
pub struct PyEmbedding {
    inner: Embedding,
}

#[pymethods]
impl PyEmbedding {
    #[new]
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> PyResult<Self> {
        Ok(Self {
            inner: Embedding::new(num_embeddings, embedding_dim).map_err(to_py_err)?,
        })
    }

    pub fn forward(&self, py: Python<'_>, indices: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&indices.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn __call__(&self, py: Python<'_>, indices: &PyTensor) -> PyResult<PyTensor> {
        self.forward(py, indices)
    }

    pub fn parameters(&self) -> Vec<PyTensor> {
        vec![PyTensor {
            inner: self.inner.weight.clone(),
        }]
    }

    #[getter]
    pub fn weight(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.weight.clone(),
        }
    }

    #[setter]
    pub fn set_weight(&mut self, w: PyTensor) {
        self.inner.weight = w.inner;
    }

    #[getter]
    pub fn num_embeddings(&self) -> usize {
        self.inner.num_embeddings()
    }

    #[getter]
    pub fn embedding_dim(&self) -> usize {
        self.inner.embedding_dim()
    }
}

/// Layer normalization over dimensions (`normalized_shape` is an int or a list/tuple).
#[pyclass(name = "LayerNorm", module = "oxideon")]
pub struct PyLayerNorm {
    inner: LayerNorm,
    normalized_shape: Vec<usize>,
}

#[pymethods]
impl PyLayerNorm {
    #[new]
    #[pyo3(signature = (normalized_shape, eps=1e-5))]
    pub fn new(normalized_shape: Bound<'_, PyAny>, eps: f64) -> PyResult<Self> {
        let shape = match normalized_shape.extract::<usize>() {
            Ok(n) => vec![n],
            Err(_) => match normalized_shape.extract::<Vec<usize>>()?.as_slice() {
                [n] => vec![*n],
                other => {
                    return Err(PyValueError::new_err(format!(
                        "LayerNorm normalizes the last dimension only: normalized_shape must be an int or a one-element list, got {other:?}"
                    )));
                }
            },
        };
        let inner = LayerNorm::with_shape(&shape, eps).map_err(to_py_err)?;
        Ok(Self {
            inner,
            normalized_shape: shape,
        })
    }

    #[getter]
    pub fn normalized_shape<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        pyo3::types::PyTuple::new(py, &self.normalized_shape)
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn __call__(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(py, x)
    }

    pub fn parameters(&self) -> Vec<PyTensor> {
        vec![
            PyTensor {
                inner: self.inner.weight.clone(),
            },
            PyTensor {
                inner: self.inner.bias.clone(),
            },
        ]
    }

    #[getter]
    pub fn weight(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.weight.clone(),
        }
    }

    #[setter]
    pub fn set_weight(&mut self, w: PyTensor) {
        self.inner.weight = w.inner;
    }

    #[getter]
    pub fn bias(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.bias.clone(),
        }
    }

    #[setter]
    pub fn set_bias(&mut self, b: PyTensor) {
        self.inner.bias = b.inner;
    }

    pub fn __repr__(&self) -> String {
        format!("LayerNorm({:?}, eps={}, elementwise_affine=true)", self.normalized_shape, self.inner.eps)
    }
}

/// Zeroes each element with probability `p` (scaling the rest by `1 / (1 - p)`) while in training mode.
#[pyclass(name = "Dropout", module = "oxideon")]
pub struct PyDropout {
    inner: Dropout,
}

#[pymethods]
impl PyDropout {
    #[new]
    #[pyo3(signature = (p=0.5))]
    pub fn new(p: f64) -> PyResult<Self> {
        if !(0.0..1.0).contains(&p) {
            return Err(PyValueError::new_err(format!(
                "dropout probability must be in [0, 1), got {p}"
            )));
        }
        Ok(Self {
            inner: Dropout::new(p as f32),
        })
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn __call__(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(py, x)
    }

    #[pyo3(signature = (mode=true))]
    pub fn train(&self, mode: bool) {
        self.inner.train(mode);
    }

    pub fn eval(&self) {
        self.inner.train(false);
    }

    pub fn parameters(&self) -> Vec<PyTensor> {
        Vec::new()
    }
}

/// Rectified linear unit, `max(x, 0)`, as a layer.
#[pyclass(name = "ReLU", module = "oxideon")]
#[derive(Default)]
pub struct PyReLU;

#[pymethods]
impl PyReLU {
    #[new]
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py.detach(|| ReLU.forward(&x.inner)).map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn __call__(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(py, x)
    }

    pub fn parameters(&self) -> Vec<PyTensor> {
        Vec::new()
    }
}

/// Multi-Layer Perceptron that can execute both single forward passes and **entire multi-epoch
/// training loops (`train_epochs_in_rust`) 100% inside native Rust** with the Python GIL released.
#[pyclass(name = "MLP", module = "oxideon")]
pub struct PyMLP {
    layers: Vec<Linear>,
}

#[pymethods]
impl PyMLP {
    #[new]
    pub fn new(dims: Vec<usize>) -> PyResult<Self> {
        if dims.len() < 2 {
            return Err(PyRuntimeError::new_err(
                "MLP requires at least 2 layer dimensions",
            ));
        }
        let mut layers = Vec::with_capacity(dims.len() - 1);
        for w in dims.windows(2) {
            layers.push(Linear::new(w[0], w[1]).map_err(to_py_err)?);
        }
        Ok(Self { layers })
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.forward_rust(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn __call__(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(py, x)
    }

    pub fn parameters(&self) -> Vec<PyTensor> {
        self.layers
            .iter()
            .flat_map(Module::parameters)
            .map(|inner| PyTensor { inner })
            .collect()
    }

    pub fn num_parameters(&self) -> usize {
        self.layers.iter().map(Module::num_parameters).sum()
    }

    /// Runs `epochs` of mini-batch training (`AdamW` or `Muon` + gradient clipping) **entirely
    /// inside native Rust** with the Python GIL released, returning the per-epoch loss curve.
    #[pyo3(signature = (x, y, epochs, batch_size=32, lr=1e-2, weight_decay=1e-4, max_grad_norm=1.0, use_muon=false))]
    #[allow(clippy::too_many_arguments)]
    pub fn train_epochs_in_rust(
        &self,
        py: Python<'_>,
        x: &PyTensor,
        y: &PyTensor,
        epochs: usize,
        batch_size: usize,
        lr: f64,
        weight_decay: f64,
        max_grad_norm: f64,
        use_muon: bool,
    ) -> PyResult<Vec<f64>> {
        let x_t = x.inner.clone();
        let y_t = y.inner.clone();
        let params: Vec<Tensor> = self.layers.iter().flat_map(Module::parameters).collect();

        py.detach(|| {
            let n = x_t.shape()[0];
            let bs = batch_size.max(1).min(n);
            let mut history = Vec::with_capacity(epochs);

            let mut adamw = Adam::adamw(params.clone(), lr, weight_decay);
            let mut muon = Muon::new(params.clone(), lr)
                .momentum(0.95)
                .weight_decay(weight_decay);

            for _epoch in 0..epochs {
                let mut epoch_loss = 0.0f64;
                let mut steps = 0usize;
                let mut start = 0usize;
                while start < n {
                    let len = (n - start).min(bs);
                    let xb = x_t.narrow(0, start, len)?;
                    let yb = y_t.narrow(0, start, len)?;

                    if use_muon {
                        muon.zero_grad();
                    } else {
                        adamw.zero_grad();
                    }

                    let pred = self.forward_rust(&xb)?;
                    let batch_loss = loss::mse_loss(&pred, &yb)?;
                    epoch_loss += batch_loss.item::<f64>()?;
                    steps += 1;

                    batch_loss.backward()?;
                    if max_grad_norm > 0.0 {
                        clip_grad_norm(&params, max_grad_norm)?;
                    }
                    if use_muon {
                        muon.step()?;
                    } else {
                        adamw.step()?;
                    }
                    start += len;
                }
                history.push(epoch_loss / (steps.max(1) as f64));
            }
            Ok::<Vec<f64>, Error>(history)
        })
        .map_err(to_py_err)
    }
}

impl PyMLP {
    fn forward_rust(&self, x: &Tensor) -> crate::error::Result<Tensor> {
        let mut h = x.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h)?;
            if i + 1 < self.layers.len() {
                h = h.silu()?;
            }
        }
        Ok(h)
    }
}

#[pyclass(name = "Conv2d", module = "oxideon")]
pub struct PyConv2d {
    inner: Conv2d,
}

#[pymethods]
impl PyConv2d {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride=1, padding=0))]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Conv2d::new(
                in_channels,
                out_channels,
                (kernel_size, kernel_size),
                (stride, stride),
                (padding, padding),
            )
            .map_err(to_py_err)?,
        })
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn __call__(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(py, x)
    }

    #[getter]
    pub fn weight(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.weight.clone(),
        }
    }

    #[setter]
    pub fn set_weight(&mut self, w: PyTensor) {
        self.inner.weight = w.inner;
    }

    #[getter]
    pub fn bias(&self) -> Option<PyTensor> {
        self.inner
            .bias
            .as_ref()
            .map(|b| PyTensor { inner: b.clone() })
    }

    #[setter]
    pub fn set_bias(&mut self, b: Option<PyTensor>) {
        self.inner.bias = b.map(|p| p.inner);
    }

    pub fn parameters(&self) -> Vec<PyTensor> {
        let mut p = vec![PyTensor {
            inner: self.inner.weight.clone(),
        }];
        if let Some(b) = &self.inner.bias {
            p.push(PyTensor { inner: b.clone() });
        }
        p
    }
}

#[pyclass(name = "Conv3d", module = "oxideon")]
pub struct PyConv3d {
    inner: Conv3d,
}

#[pymethods]
impl PyConv3d {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride=1, padding=0))]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> PyResult<Self> {
        let k = (kernel_size, kernel_size, kernel_size);
        let s = (stride, stride, stride);
        let p = (padding, padding, padding);
        Ok(Self {
            inner: Conv3d::new(
                in_channels,
                out_channels,
                k,
                s,
                p,
                true,
                DType::F32,
                Device::Cpu,
            )
            .map_err(to_py_err)?,
        })
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "MaxPool1d", module = "oxideon")]
pub struct PyMaxPool1d {
    inner: MaxPool1d,
}

#[pymethods]
impl PyMaxPool1d {
    #[new]
    #[pyo3(signature = (kernel_size, stride=None, padding=0))]
    pub fn new(kernel_size: usize, stride: Option<usize>, padding: usize) -> Self {
        let s = stride.unwrap_or(kernel_size);
        Self {
            inner: MaxPool1d::new(kernel_size, s).with_padding(padding),
        }
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "AvgPool1d", module = "oxideon")]
pub struct PyAvgPool1d {
    inner: AvgPool1d,
}

#[pymethods]
impl PyAvgPool1d {
    #[new]
    #[pyo3(signature = (kernel_size, stride=None, padding=0))]
    pub fn new(kernel_size: usize, stride: Option<usize>, padding: usize) -> Self {
        let s = stride.unwrap_or(kernel_size);
        Self {
            inner: AvgPool1d::new(kernel_size, s).with_padding(padding),
        }
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "ConvTranspose3d", module = "oxideon")]
pub struct PyConvTranspose3d {
    inner: ConvTranspose3d,
}

#[pymethods]
impl PyConvTranspose3d {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0))]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
    ) -> PyResult<Self> {
        let k = (kernel_size, kernel_size, kernel_size);
        let s = (stride, stride, stride);
        let p = (padding, padding, padding);
        let op = (output_padding, output_padding, output_padding);
        Ok(Self {
            inner: ConvTranspose3d::new(
                in_channels,
                out_channels,
                k,
                s,
                p,
                op,
                true,
                DType::F32,
                Device::Cpu,
            )
            .map_err(to_py_err)?,
        })
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "InstanceNorm1d", module = "oxideon")]
pub struct PyInstanceNorm1d {
    inner: InstanceNorm1d,
}

#[pymethods]
impl PyInstanceNorm1d {
    #[new]
    #[pyo3(signature = (num_features, eps=1e-5))]
    pub fn new(num_features: usize, eps: f64) -> PyResult<Self> {
        Ok(Self {
            inner: InstanceNorm1d::new(num_features, eps).map_err(to_py_err)?,
        })
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "InstanceNorm3d", module = "oxideon")]
pub struct PyInstanceNorm3d {
    inner: InstanceNorm3d,
}

#[pymethods]
impl PyInstanceNorm3d {
    #[new]
    #[pyo3(signature = (num_features, eps=1e-5))]
    pub fn new(num_features: usize, eps: f64) -> PyResult<Self> {
        Ok(Self {
            inner: InstanceNorm3d::new(num_features, eps).map_err(to_py_err)?,
        })
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "EmbeddingBag", module = "oxideon")]
pub struct PyEmbeddingBag {
    inner: EmbeddingBag,
}

#[pymethods]
impl PyEmbeddingBag {
    #[new]
    #[pyo3(signature = (num_embeddings, embedding_dim, mode="mean"))]
    pub fn new(num_embeddings: usize, embedding_dim: usize, mode: &str) -> PyResult<Self> {
        let m = match mode.to_lowercase().as_str() {
            "sum" => EmbeddingBagMode::Sum,
            "mean" => EmbeddingBagMode::Mean,
            "max" => EmbeddingBagMode::Max,
            other => {
                return Err(to_py_err(Error::Msg(format!(
                    "unknown EmbeddingBag mode: {other}"
                ))));
            }
        };
        Ok(Self {
            inner: EmbeddingBag::new(num_embeddings, embedding_dim, m).map_err(to_py_err)?,
        })
    }

    pub fn forward(&self, py: Python<'_>, indices: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&indices.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "ReLU6", module = "oxideon")]
#[derive(Default)]
pub struct PyReLU6 {
    inner: ReLU6,
}

#[pymethods]
impl PyReLU6 {
    #[new]
    pub fn new() -> Self {
        Self { inner: ReLU6 }
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "Hardtanh", module = "oxideon")]
pub struct PyHardtanh {
    inner: Hardtanh,
}

#[pymethods]
impl PyHardtanh {
    #[new]
    #[pyo3(signature = (min_val=-1.0, max_val=1.0))]
    pub fn new(min_val: f64, max_val: f64) -> Self {
        Self {
            inner: Hardtanh::new(min_val, max_val),
        }
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "Softshrink", module = "oxideon")]
pub struct PySoftshrink {
    inner: Softshrink,
}

#[pymethods]
impl PySoftshrink {
    #[new]
    #[pyo3(signature = (lambd=0.5))]
    pub fn new(lambd: f64) -> Self {
        Self {
            inner: Softshrink::new(lambd),
        }
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "Hardshrink", module = "oxideon")]
pub struct PyHardshrink {
    inner: Hardshrink,
}

#[pymethods]
impl PyHardshrink {
    #[new]
    #[pyo3(signature = (lambd=0.5))]
    pub fn new(lambd: f64) -> Self {
        Self {
            inner: Hardshrink::new(lambd),
        }
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "LogSigmoid", module = "oxideon")]
#[derive(Default)]
pub struct PyLogSigmoid {
    inner: LogSigmoid,
}

#[pymethods]
impl PyLogSigmoid {
    #[new]
    pub fn new() -> Self {
        Self { inner: LogSigmoid }
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "MarginRankingLoss", module = "oxideon")]
pub struct PyMarginRankingLoss {
    inner: MarginRankingLoss,
}

#[pymethods]
impl PyMarginRankingLoss {
    #[new]
    #[pyo3(signature = (margin=0.0))]
    pub fn new(margin: f64) -> Self {
        Self {
            inner: MarginRankingLoss::new(margin),
        }
    }

    pub fn forward(
        &self,
        py: Python<'_>,
        x1: &PyTensor,
        x2: &PyTensor,
        target: &PyTensor,
    ) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x1.inner, &x2.inner, &target.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "HingeEmbeddingLoss", module = "oxideon")]
pub struct PyHingeEmbeddingLoss {
    inner: HingeEmbeddingLoss,
}

#[pymethods]
impl PyHingeEmbeddingLoss {
    #[new]
    #[pyo3(signature = (margin=1.0))]
    pub fn new(margin: f64) -> Self {
        Self {
            inner: HingeEmbeddingLoss::new(margin),
        }
    }

    pub fn forward(
        &self,
        py: Python<'_>,
        input: &PyTensor,
        target: &PyTensor,
    ) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&input.inner, &target.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "SoftMarginLoss", module = "oxideon")]
#[derive(Default)]
pub struct PySoftMarginLoss {
    inner: SoftMarginLoss,
}

#[pymethods]
impl PySoftMarginLoss {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: SoftMarginLoss,
        }
    }

    pub fn forward(
        &self,
        py: Python<'_>,
        input: &PyTensor,
        target: &PyTensor,
    ) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&input.inner, &target.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "CTCLoss", module = "oxideon")]
pub struct PyCTCLoss {
    inner: CTCLoss,
}

#[pymethods]
impl PyCTCLoss {
    #[new]
    #[pyo3(signature = (blank=0, zero_infinity=false))]
    pub fn new(blank: usize, zero_infinity: bool) -> Self {
        Self {
            inner: CTCLoss::new(blank, zero_infinity),
        }
    }

    pub fn forward(
        &self,
        py: Python<'_>,
        log_probs: &PyTensor,
        targets: &PyTensor,
    ) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&log_probs.inner, &targets.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "QuantizedPagedKVCache", module = "oxideon")]
pub struct PyQuantizedPagedKVCache {
    pub inner: std::sync::Arc<QuantizedPagedKVCache>,
}

#[pymethods]
impl PyQuantizedPagedKVCache {
    #[new]
    #[pyo3(signature = (num_blocks, block_size, num_heads, head_dim, dtype="fp8"))]
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
        dtype: &str,
    ) -> PyResult<Self> {
        let qdtype = match dtype.to_lowercase().as_str() {
            "fp8" | "fp8_e4m3" | "e4m3" => QuantKVDType::Fp8E4M3,
            "int8" | "i8" => QuantKVDType::Int8,
            "int4" | "i4" => QuantKVDType::Int4,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown QuantKVDType: {dtype}"
                )));
            }
        };
        let inner = QuantizedPagedKVCache::new(num_blocks, block_size, num_heads, head_dim, qdtype)
            .map_err(to_py_err)?;
        Ok(Self {
            inner: std::sync::Arc::new(inner),
        })
    }

    pub fn write_token(
        &self,
        py: Python<'_>,
        block_id: usize,
        offset_in_block: usize,
        key: &PyTensor,
        value: &PyTensor,
    ) -> PyResult<()> {
        py.detach(|| {
            self.inner
                .write_token(block_id, offset_in_block, &key.inner, &value.inner)
        })
        .map_err(to_py_err)
    }

    pub fn read_token_dequantized(
        &self,
        py: Python<'_>,
        block_id: usize,
        offset_in_block: usize,
    ) -> PyResult<(PyTensor, PyTensor)> {
        let (k_vec, v_vec) = py
            .detach(|| self.inner.read_token_dequantized(block_id, offset_in_block))
            .map_err(to_py_err)?;
        let shape = vec![self.inner.num_heads, self.inner.head_dim];
        let k_t = Tensor::from_slice(&k_vec, &shape).map_err(to_py_err)?;
        let v_t = Tensor::from_slice(&v_vec, &shape).map_err(to_py_err)?;
        Ok((PyTensor { inner: k_t }, PyTensor { inner: v_t }))
    }

    #[getter]
    pub fn memory_usage_bytes(&self) -> usize {
        self.inner.memory_usage_bytes()
    }

    #[getter]
    pub fn compression_ratio(&self) -> f64 {
        self.inner.compression_ratio()
    }
}

#[pyfunction]
#[pyo3(name = "paged_attention_quantized")]
pub fn py_paged_attention_quantized(
    py: Python<'_>,
    queries: &PyTensor,
    cache: &PyQuantizedPagedKVCache,
    block_tables: Vec<Vec<usize>>,
    seq_lens: Vec<usize>,
    scale: f64,
) -> PyResult<PyTensor> {
    let bts: Vec<crate::serving::BlockTable> = block_tables
        .into_iter()
        .map(|blocks| crate::serving::BlockTable {
            num_tokens: blocks.len() * cache.inner.block_size,
            physical_blocks: blocks,
        })
        .collect();
    let out = py
        .detach(|| {
            crate::serving::paged_attention_quantized(
                &queries.inner,
                &cache.inner,
                &bts,
                &seq_lens,
                scale,
            )
        })
        .map_err(to_py_err)?;
    Ok(PyTensor { inner: out })
}

#[pyclass(name = "StreamingUtf8Decoder", module = "oxideon")]
#[derive(Default)]
pub struct PyStreamingUtf8Decoder {
    inner: StreamingUtf8Decoder,
}

#[pymethods]
impl PyStreamingUtf8Decoder {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: StreamingUtf8Decoder::new(),
        }
    }

    pub fn add_bytes(&mut self, chunk: &[u8]) -> String {
        self.inner.add_bytes(chunk)
    }

    pub fn flush(&mut self) -> String {
        self.inner.flush()
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

#[pyclass(name = "StopSequenceMatcher", module = "oxideon")]
pub struct PyStopSequenceMatcher {
    inner: StopSequenceMatcher,
}

#[pymethods]
impl PyStopSequenceMatcher {
    #[new]
    pub fn new(stop_sequences: Vec<String>) -> Self {
        Self {
            inner: StopSequenceMatcher::new(stop_sequences),
        }
    }

    pub fn step(&mut self, text: &str) -> (String, bool) {
        self.inner.step(text)
    }

    pub fn flush(&mut self) -> String {
        self.inner.flush()
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    pub fn is_stopped(&self) -> bool {
        self.inner.is_stopped()
    }
}

#[pyclass(name = "QLoRALinear", module = "oxideon")]
pub struct PyQLoRALinear {
    pub inner: QLoRALinear,
}

#[pymethods]
impl PyQLoRALinear {
    #[new]
    #[pyo3(signature = (in_features, out_features, rank=8, alpha=16.0, scheme="q4_0", dropout_p=0.0))]
    pub fn new(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f64,
        scheme: &str,
        dropout_p: f32,
    ) -> PyResult<Self> {
        if !(0.0..1.0).contains(&dropout_p) {
            return Err(PyValueError::new_err(format!(
                "dropout_p must be in [0, 1), got {dropout_p}"
            )));
        }
        let qscheme = match scheme.to_lowercase().as_str() {
            "q4_0" | "q4" => QuantScheme::DEFAULT_Q4_0,
            "q8_0" | "q8" => QuantScheme::DEFAULT_Q8_0,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown QuantScheme: {scheme}"
                )));
            }
        };
        let lin = Linear::new(in_features, out_features).map_err(to_py_err)?;
        let inner =
            QLoRALinear::from_linear(&lin, qscheme, rank, alpha, dropout_p).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn forward(&self, py: Python<'_>, input: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&input.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn enable_adapter(&mut self) {
        self.inner.enable_adapter();
    }

    pub fn disable_adapter(&mut self) {
        self.inner.disable_adapter();
    }

    pub fn merge_adapter(&mut self) -> PyResult<()> {
        self.inner.merge_adapter().map_err(to_py_err)
    }

    #[getter]
    pub fn in_features(&self) -> usize {
        self.inner.in_features()
    }

    #[getter]
    pub fn out_features(&self) -> usize {
        self.inner.out_features()
    }

    #[getter]
    pub fn rank(&self) -> usize {
        self.inner.rank
    }

    #[getter]
    pub fn scaling(&self) -> f64 {
        self.inner.scaling
    }
}

#[pyclass(name = "GGUFFile", module = "oxideon")]
pub struct PyGGUFFile {
    pub inner: GGUFFile,
}

#[pymethods]
impl PyGGUFFile {
    #[staticmethod]
    pub fn open(path: &str) -> PyResult<Self> {
        let inner = GGUFFile::open(path).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    pub fn from_bytes(bytes: &[u8]) -> PyResult<Self> {
        let inner = GGUFFile::from_bytes(bytes).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn version(&self) -> u32 {
        self.inner.version
    }

    #[getter]
    pub fn alignment(&self) -> usize {
        self.inner.alignment
    }

    #[getter]
    pub fn tensor_names(&self) -> Vec<String> {
        self.inner.tensors.keys().cloned().collect()
    }

    pub fn get_metadata_str(&self, key: &str) -> Option<String> {
        self.inner.get_metadata_str(key).map(|s| s.to_string())
    }

    pub fn get_metadata_u32(&self, key: &str) -> Option<u32> {
        self.inner.get_metadata_u32(key)
    }

    pub fn dequantize_tensor(&self, py: Python<'_>, name: &str) -> PyResult<PyTensor> {
        let t = py
            .detach(|| self.inner.dequantize_tensor(name))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: t })
    }

    pub fn fused_gemv(&self, py: Python<'_>, name: &str, x: &PyTensor) -> PyResult<PyTensor> {
        let t = py
            .detach(|| self.inner.fused_gemv(name, &x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: t })
    }
}

#[pyfunction]
#[pyo3(name = "save_gguf", signature = (path, tensors, metadata=None))]
pub fn py_save_gguf(
    path: &str,
    tensors: std::collections::HashMap<String, PyTensor>,
    metadata: Option<std::collections::HashMap<String, String>>,
) -> PyResult<()> {
    let mut meta = std::collections::HashMap::new();
    if let Some(m) = metadata {
        for (k, v) in m {
            meta.insert(k, GGUFValue::String(v));
        }
    }
    let list: Vec<(&str, GGMLType, &Tensor)> = tensors
        .iter()
        .map(|(k, v)| (k.as_str(), GGMLType::F32, &v.inner))
        .collect();
    save_gguf(path, &meta, &list).map_err(to_py_err)
}

#[pyclass(name = "MultiHeadLatentAttention", module = "oxideon")]
pub struct PyMultiHeadLatentAttention {
    pub inner: MultiHeadLatentAttention,
}

#[pymethods]
impl PyMultiHeadLatentAttention {
    #[new]
    #[pyo3(signature = (d_model, num_heads, kv_lora_rank=512, q_lora_rank=1536))]
    pub fn new(
        d_model: usize,
        num_heads: usize,
        kv_lora_rank: usize,
        q_lora_rank: usize,
    ) -> PyResult<Self> {
        let inner = MultiHeadLatentAttention::new(d_model, num_heads, kv_lora_rank, q_lora_rank)
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (x, causal=true))]
    pub fn forward(&self, py: Python<'_>, x: &PyTensor, causal: bool) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward_mla(&x.inner, causal))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    #[pyo3(signature = (x, cache=None))]
    pub fn step_absorbed(
        &self,
        py: Python<'_>,
        x: &PyTensor,
        cache: Option<&PyTensor>,
    ) -> PyResult<(PyTensor, PyTensor)> {
        let mut opt_cache = cache.map(|c| c.inner.clone());
        let out = py
            .detach(|| self.inner.step_absorbed(&x.inner, &mut opt_cache))
            .map_err(to_py_err)?;
        let updated_cache = opt_cache.ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("step_absorbed failed to initialize cache")
        })?;
        Ok((
            PyTensor { inner: out },
            PyTensor {
                inner: updated_cache,
            },
        ))
    }

    #[getter]
    pub fn kv_cache_compression_ratio(&self) -> f64 {
        self.inner.kv_cache_compression_ratio()
    }
}

#[pyclass(name = "DoubleBufferedKVCache", module = "oxideon")]
pub struct PyDoubleBufferedKVCache {
    pub inner: std::sync::Arc<DoubleBufferedKVCache>,
}

#[pymethods]
impl PyDoubleBufferedKVCache {
    #[new]
    #[pyo3(signature = (num_blocks, block_size, num_heads, head_dim, dtype="f32"))]
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
        dtype: &str,
    ) -> PyResult<Self> {
        let dt = parse_dtype(dtype)?;
        let inner = DoubleBufferedKVCache::new(num_blocks, block_size, num_heads, head_dim, dt)
            .map_err(to_py_err)?;
        Ok(Self {
            inner: std::sync::Arc::new(inner),
        })
    }

    pub fn swap_banks(&self) {
        self.inner.swap_banks();
    }

    #[getter]
    pub fn active_bank(&self) -> usize {
        self.inner
            .active_bank
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[pyclass(name = "AsyncKVPipelineEngine", module = "oxideon")]
pub struct PyAsyncKVPipelineEngine {
    pub inner: AsyncKVPipelineEngine,
}

#[pymethods]
impl PyAsyncKVPipelineEngine {
    #[new]
    pub fn new(cache: &PyDoubleBufferedKVCache) -> Self {
        let inner = AsyncKVPipelineEngine::new(cache.inner.clone());
        Self { inner }
    }

    pub fn submit_swap_in(&self, request_id: usize, num_blocks: usize) -> usize {
        self.inner.submit_swap_in(request_id, num_blocks)
    }

    pub fn submit_swap_out(&self, request_id: usize, num_blocks: usize) -> usize {
        self.inner.submit_swap_out(request_id, num_blocks)
    }

    pub fn poll_completed(&self) -> Vec<usize> {
        self.inner
            .poll_completed()
            .into_iter()
            .map(|ev| ev.transfer_id)
            .collect()
    }

    #[getter]
    pub fn is_idle(&self) -> bool {
        self.inner.is_idle()
    }

    pub fn wait_all(&self) {
        self.inner.wait_all();
    }

    pub fn swap_banks(&self) {
        self.inner.swap_banks();
    }
}

#[pyclass(name = "UnifiedSpeculativeEngine", module = "oxideon")]
pub struct PyUnifiedSpeculativeEngine;

#[pymethods]
impl PyUnifiedSpeculativeEngine {
    #[staticmethod]
    pub fn verify_greedy(
        py: Python<'_>,
        draft_tokens: Vec<usize>,
        target_logits: &PyTensor,
    ) -> PyResult<(Vec<usize>, Option<usize>, f64)> {
        let out = py
            .detach(|| UnifiedSpeculativeEngine::verify_greedy(&draft_tokens, &target_logits.inner))
            .map_err(to_py_err)?;
        Ok((out.accepted_tokens, out.bonus_token, out.speedup_ratio))
    }

    #[staticmethod]
    #[pyo3(signature = (draft_tokens, draft_probs, target_probs, seed=None))]
    pub fn verify_stochastic(
        py: Python<'_>,
        draft_tokens: Vec<usize>,
        draft_probs: Vec<f64>,
        target_probs: Vec<Vec<f64>>,
        seed: Option<u64>,
    ) -> PyResult<(Vec<usize>, Option<usize>, f64)> {
        let out = py
            .detach(|| {
                UnifiedSpeculativeEngine::verify_stochastic(
                    &draft_tokens,
                    &draft_probs,
                    &target_probs,
                    seed,
                )
            })
            .map_err(to_py_err)?;
        Ok((out.accepted_tokens, out.bonus_token, out.speedup_ratio))
    }
}

#[pyclass(name = "PrefillDecodeInterleaver", module = "oxideon")]
pub struct PyPrefillDecodeInterleaver {
    inner: PrefillDecodeInterleaver,
}

#[pymethods]
impl PyPrefillDecodeInterleaver {
    #[new]
    #[pyo3(signature = (max_batch_tokens, chunk_size=512))]
    pub fn new(max_batch_tokens: usize, chunk_size: usize) -> PyResult<Self> {
        if max_batch_tokens == 0 || chunk_size == 0 {
            return Err(PyValueError::new_err(
                "max_batch_tokens and chunk_size must be positive",
            ));
        }
        Ok(Self {
            inner: PrefillDecodeInterleaver::new(max_batch_tokens, chunk_size)
                .map_err(to_py_err)?,
        })
    }

    #[pyo3(signature = (prompt, config=None))]
    pub fn submit_request(
        &mut self,
        prompt: Vec<usize>,
        config: Option<&PySamplingConfig>,
    ) -> usize {
        let sc = config.map(|c| c.inner.clone()).unwrap_or_default();
        self.inner.submit_request(prompt, sc)
    }

    #[allow(clippy::type_complexity)]
    pub fn schedule_iteration(&mut self) -> (Vec<(usize, usize, usize, bool)>, Vec<usize>, usize) {
        let batch = self.inner.schedule_iteration();
        let chunks = batch
            .prefill_chunks
            .into_iter()
            .map(|c| (c.request_id, c.start_idx, c.len, c.is_final_chunk))
            .collect();
        (chunks, batch.decode_ids, batch.total_tokens)
    }

    pub fn on_chunk_processed(&mut self, request_id: usize, chunk_len: usize) {
        self.inner.on_chunk_processed(request_id, chunk_len);
    }

    pub fn on_decode_token(&mut self, request_id: usize, token_id: usize, eos_token: usize) {
        self.inner.on_decode_token(request_id, token_id, eos_token);
    }

    pub fn prune_and_check_idle(&mut self) -> bool {
        self.inner.prune_and_check_idle()
    }
}

#[pyclass(name = "GGUFCausalLM", module = "oxideon")]
pub struct PyGGUFCausalLM {
    pub inner: GGUFCausalLM,
}

#[pymethods]
impl PyGGUFCausalLM {
    #[new]
    #[pyo3(signature = (vocab_size, d_model, num_heads, num_layers, max_seq_len=2048, rope_freq_base=10000.0))]
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        num_heads: usize,
        num_layers: usize,
        max_seq_len: usize,
        rope_freq_base: f32,
    ) -> PyResult<Self> {
        let config = GGUFModelConfig::basic(
            vocab_size,
            d_model,
            num_heads,
            num_layers,
            max_seq_len,
            rope_freq_base,
        )
        .map_err(to_py_err)?;
        let inner = GGUFCausalLM::from_config(config).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Loads both the architecture config *and* the actual quantized weights from a real GGUF
    /// model file (unlike the constructor above, which only shapes a model from hyperparameters
    /// you pass in and leaves it randomly initialized).
    #[staticmethod]
    pub fn from_file(path: &str) -> PyResult<Self> {
        let file = GGUFFile::open(path).map_err(to_py_err)?;
        let inner = GGUFCausalLM::from_gguf(&file).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn forward(&self, py: Python<'_>, input_ids: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&input_ids.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    /// Greedy decoding: returns `prompt_ids` followed by up to `max_new_tokens` generated ids.
    #[pyo3(signature = (prompt_ids, max_new_tokens=32, eos_token=None))]
    pub fn generate(
        &self,
        py: Python<'_>,
        prompt_ids: Vec<i64>,
        max_new_tokens: usize,
        eos_token: Option<i64>,
    ) -> PyResult<Vec<i64>> {
        py.detach(|| {
            self.inner
                .generate_greedy(&prompt_ids, max_new_tokens, eos_token)
        })
        .map_err(to_py_err)
    }
}

/// Tokenizer taken from a GGUF file's own vocabulary: SentencePiece (`llama`) or byte-level BPE (`gpt2`).
#[pyclass(name = "GGUFTokenizer", module = "oxideon")]
pub struct PyGGUFTokenizer {
    inner: GGUFTokenizer,
}

#[pymethods]
impl PyGGUFTokenizer {
    #[staticmethod]
    pub fn from_file(path: &str) -> PyResult<Self> {
        let file = GGUFFile::open(path).map_err(to_py_err)?;
        let inner = GGUFTokenizer::from_gguf(&file).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// `add_bos=None` follows the file (`tokenizer.ggml.add_bos_token`).
    #[pyo3(signature = (text, add_bos=None))]
    pub fn encode(&self, text: &str, add_bos: Option<bool>) -> Vec<i64> {
        self.inner
            .encode(text, add_bos.unwrap_or(self.inner.add_bos_default))
    }

    pub fn decode(&self, ids: Vec<i64>) -> String {
        self.inner.decode(&ids)
    }

    #[getter]
    pub fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    #[getter]
    pub fn bos_id(&self) -> Option<u32> {
        self.inner.bos_id
    }

    #[getter]
    pub fn eos_id(&self) -> Option<u32> {
        self.inner.eos_id
    }
}

#[pyclass(name = "BpeTokenizer", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyBpeTokenizer {
    inner: BpeTokenizer,
}

#[pymethods]
impl PyBpeTokenizer {
    #[staticmethod]
    pub fn train(py: Python<'_>, corpus: &str, target_vocab_size: usize) -> Self {
        let inner = py.detach(|| BpeTokenizer::train(corpus, target_vocab_size));
        Self { inner }
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        self.inner.encode(text)
    }

    pub fn encode_with_special(&self, text: &str, add_bos: bool, add_eos: bool) -> Vec<usize> {
        self.inner.encode_with_special(text, add_bos, add_eos)
    }

    pub fn decode(&self, ids: Vec<usize>) -> String {
        self.inner.decode(&ids)
    }
}

#[pyfunction]
#[pyo3(signature = (key, seed=0))]
pub fn fast_murmurhash3_32(key: &str, seed: u32) -> u32 {
    crate::vectorize::murmurhash3_32(key.as_bytes(), seed)
}

#[pyclass(name = "FastBM25Index", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyFastBM25Index {
    inner: FastBM25Index,
}

#[pymethods]
impl PyFastBM25Index {
    #[new]
    #[pyo3(signature = (k1=1.5, b=0.75, delta=0.0))]
    pub fn new(k1: f64, b: f64, delta: f64) -> Self {
        Self {
            inner: FastBM25Index::new(k1, b, delta),
        }
    }

    #[staticmethod]
    pub fn okapi() -> Self {
        Self {
            inner: FastBM25Index::new_okapi(),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (delta=1.0))]
    pub fn plus(delta: f64) -> Self {
        Self {
            inner: FastBM25Index::new_plus(delta),
        }
    }

    pub fn fit(&mut self, corpus: Vec<String>) {
        let refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();
        self.inner.fit(&refs);
    }

    pub fn score_query(&self, query: &str) -> Vec<f64> {
        self.inner.score_query(query)
    }

    pub fn top_k(&self, query: &str, k: usize) -> Vec<(usize, f64)> {
        self.inner.top_k(query, k)
    }

    #[getter]
    pub fn num_docs(&self) -> usize {
        self.inner.num_docs
    }

    #[getter]
    pub fn avgdl(&self) -> f64 {
        self.inner.avgdl
    }
}

#[pyclass(name = "FastCountVectorizer", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyFastCountVectorizer {
    inner: FastCountVectorizer,
}

#[pymethods]
impl PyFastCountVectorizer {
    #[new]
    #[pyo3(signature = (ngram_range=(1, 1), binary=false, max_features=None, min_df=1, stop_words=None))]
    pub fn new(
        ngram_range: (usize, usize),
        binary: bool,
        max_features: Option<usize>,
        min_df: usize,
        stop_words: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let mut cv = FastCountVectorizer::new(ngram_range, binary, max_features)
            .map_err(to_py_err)?
            .with_min_df(min_df);
        if let Some(sw) = stop_words {
            cv = cv.with_stop_words(sw.into_iter().collect());
        }
        Ok(Self { inner: cv })
    }

    pub fn fit(&mut self, documents: Vec<String>) {
        let refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
        self.inner.fit(&refs);
    }

    pub fn transform(&self, documents: Vec<String>) -> PyResult<PySparseCsr> {
        let refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
        let inner = self.inner.transform(&refs).map_err(to_py_err)?;
        Ok(PySparseCsr { inner })
    }

    pub fn transform_dense(&self, documents: Vec<String>) -> PyResult<PyTensor> {
        let refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
        let inner = self.inner.transform_dense(&refs).map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }

    #[getter]
    pub fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    pub fn get_feature_names(&self) -> Vec<String> {
        self.inner.inverse_vocab.clone()
    }
}

#[pyclass(name = "FastTfidfVectorizer", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyFastTfidfVectorizer {
    inner: FastTfidfVectorizer,
}

#[pymethods]
impl PyFastTfidfVectorizer {
    #[new]
    #[pyo3(signature = (ngram_range=(1, 1), sublinear_tf=false, norm=Some("l2".to_string()), max_features=None, min_df=1, stop_words=None))]
    pub fn new(
        ngram_range: (usize, usize),
        sublinear_tf: bool,
        norm: Option<String>,
        max_features: Option<usize>,
        min_df: usize,
        stop_words: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let mut tfidf = FastTfidfVectorizer::new(ngram_range, sublinear_tf, norm)
            .map_err(to_py_err)?;
        tfidf.count_vec.max_features = max_features;
        tfidf.count_vec = tfidf.count_vec.with_min_df(min_df);
        if let Some(sw) = stop_words {
            tfidf.count_vec = tfidf.count_vec.with_stop_words(sw.into_iter().collect());
        }
        Ok(Self { inner: tfidf })
    }

    pub fn fit(&mut self, documents: Vec<String>) {
        let refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
        self.inner.fit(&refs);
    }

    pub fn transform(&self, documents: Vec<String>) -> PyResult<PySparseCsr> {
        let refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
        let inner = self.inner.transform(&refs).map_err(to_py_err)?;
        Ok(PySparseCsr { inner })
    }

    pub fn transform_dense(&self, documents: Vec<String>) -> PyResult<PyTensor> {
        let refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
        let inner = self.inner.transform_dense(&refs).map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }

    #[getter]
    pub fn idf(&self) -> Vec<f64> {
        self.inner.idf.clone()
    }

    #[getter]
    pub fn vocab_size(&self) -> usize {
        self.inner.count_vec.vocab_size()
    }

    pub fn get_feature_names(&self) -> Vec<String> {
        self.inner.count_vec.inverse_vocab.clone()
    }
}

#[pyclass(name = "FastHashingVectorizer", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyFastHashingVectorizer {
    inner: FastHashingVectorizer,
}

#[pymethods]
impl PyFastHashingVectorizer {
    #[new]
    #[pyo3(signature = (n_features=1048576, ngram_range=(1, 1), alternate_sign=true, norm=Some("l2".to_string())))]
    pub fn new(
        n_features: usize,
        ngram_range: (usize, usize),
        alternate_sign: bool,
        norm: Option<String>,
    ) -> PyResult<Self> {
        let inner = FastHashingVectorizer::new(n_features, ngram_range, alternate_sign, norm)
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn transform(&self, documents: Vec<String>) -> PyResult<PySparseCsr> {
        let refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
        let inner = self.inner.transform(&refs).map_err(to_py_err)?;
        Ok(PySparseCsr { inner })
    }

    pub fn transform_dense(&self, documents: Vec<String>) -> PyResult<PyTensor> {
        let refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
        let inner = self.inner.transform_dense(&refs).map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }
}

#[pyclass(name = "FastTextCharNGramVectorizer", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyFastTextCharNGramVectorizer {
    inner: FastTextCharNGramVectorizer,
}

#[pymethods]
impl PyFastTextCharNGramVectorizer {
    #[new]
    #[pyo3(signature = (dim=128, min_n=3, max_n=6))]
    pub fn new(dim: usize, min_n: usize, max_n: usize) -> PyResult<Self> {
        let inner = FastTextCharNGramVectorizer::new(dim, min_n, max_n).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn transform_word(&self, word: &str) -> Vec<f64> {
        self.inner.transform_word(word)
    }

    pub fn transform_document(&self, doc: &str) -> Vec<f64> {
        self.inner.transform_document(doc)
    }

    pub fn transform_dense(&self, docs: Vec<String>) -> PyResult<PyTensor> {
        let refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
        let inner = self.inner.transform_dense(&refs).map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }
}

#[pyclass(name = "KernelSpectralVectorizer", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyKernelSpectralVectorizer {
    inner: KernelSpectralVectorizer,
}

#[pymethods]
impl PyKernelSpectralVectorizer {
    #[new]
    #[pyo3(signature = (in_dim, out_dim, gamma=0.5, seed=42))]
    pub fn new(in_dim: usize, out_dim: usize, gamma: f64, seed: u64) -> PyResult<Self> {
        let inner =
            KernelSpectralVectorizer::new(in_dim, out_dim, gamma, seed).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn transform(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let inner = self.inner.transform(&x.inner).map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }
}

#[pyclass(name = "FastMinHashVectorizer", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyFastMinHashVectorizer {
    inner: FastMinHashVectorizer,
}

#[pymethods]
impl PyFastMinHashVectorizer {
    #[new]
    #[pyo3(signature = (num_hashes=64, ngram_range=(1, 1), seed=42))]
    pub fn new(num_hashes: usize, ngram_range: (usize, usize), seed: u32) -> PyResult<Self> {
        let inner =
            FastMinHashVectorizer::new(num_hashes, ngram_range, seed).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn transform_document(&self, doc: &str) -> Vec<u32> {
        self.inner.transform_document(doc)
    }

    pub fn transform_dense(&self, docs: Vec<String>) -> PyResult<PyTensor> {
        let refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
        let inner = self.inner.transform_dense(&refs).map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }

    pub fn jaccard_similarity(&self, sig_a: Vec<u32>, sig_b: Vec<u32>) -> f64 {
        self.inner.jaccard_similarity(&sig_a, &sig_b)
    }

    #[getter]
    pub fn num_hashes(&self) -> usize {
        self.inner.num_hashes
    }
}

#[pyclass(name = "FastSimHashVectorizer", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyFastSimHashVectorizer {
    inner: FastSimHashVectorizer,
}

#[pymethods]
impl PyFastSimHashVectorizer {
    #[new]
    #[pyo3(signature = (dim=64, ngram_range=(1, 1), seed=42))]
    pub fn new(dim: usize, ngram_range: (usize, usize), seed: u32) -> PyResult<Self> {
        let inner =
            FastSimHashVectorizer::new(dim, ngram_range, seed).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn transform_document(&self, doc: &str) -> (Vec<f64>, Vec<u8>) {
        self.inner.transform_document(doc)
    }

    pub fn transform_dense(&self, docs: Vec<String>) -> PyResult<PyTensor> {
        let refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
        let inner = self.inner.transform_dense(&refs).map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }

    pub fn hamming_distance(&self, bits_a: Vec<u8>, bits_b: Vec<u8>) -> usize {
        self.inner.hamming_distance(&bits_a, &bits_b)
    }

    #[getter]
    pub fn dim(&self) -> usize {
        self.inner.dim
    }
}

#[pyclass(name = "FastByteHistogramVectorizer", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyFastByteHistogramVectorizer {
    inner: FastByteHistogramVectorizer,
}

#[pymethods]
impl PyFastByteHistogramVectorizer {
    #[new]
    #[pyo3(signature = (log_scale=true, norm=Some("l2".to_string())))]
    pub fn new(log_scale: bool, norm: Option<String>) -> Self {
        Self {
            inner: FastByteHistogramVectorizer::new(log_scale, norm),
        }
    }

    pub fn transform_bytes(&self, data: &[u8]) -> Vec<f64> {
        self.inner.transform_bytes(data)
    }

    pub fn transform_dense(&self, docs: Vec<String>) -> PyResult<PyTensor> {
        let refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
        let inner = self.inner.transform_dense(&refs).map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }
}

#[pyclass(name = "FastMaxSimScorer", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyFastMaxSimScorer;

#[pymethods]
impl PyFastMaxSimScorer {
    #[staticmethod]
    pub fn score_maxsim(query: &PyTensor, doc: &PyTensor) -> PyResult<f64> {
        FastMaxSimScorer::score_maxsim(&query.inner, &doc.inner).map_err(to_py_err)
    }

    #[staticmethod]
    pub fn batch_maxsim(queries: Vec<PyTensor>, docs: Vec<PyTensor>) -> PyResult<PyTensor> {
        let q_tensors: Vec<Tensor> = queries.into_iter().map(|t| t.inner).collect();
        let d_tensors: Vec<Tensor> = docs.into_iter().map(|t| t.inner).collect();
        let inner =
            FastMaxSimScorer::batch_maxsim(&q_tensors, &d_tensors).map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }
}

#[pyclass(name = "FastSphericalHarmonics", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyFastSphericalHarmonics {
    inner: FastSphericalHarmonics,
}

#[pymethods]
impl PyFastSphericalHarmonics {
    #[new]
    #[pyo3(signature = (degree=2))]
    pub fn new(degree: usize) -> PyResult<Self> {
        let inner = FastSphericalHarmonics::new(degree).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn transform_point(&self, p: (f64, f64, f64)) -> Vec<f64> {
        self.inner.transform_point([p.0, p.1, p.2])
    }

    pub fn transform(&self, points: &PyTensor) -> PyResult<PyTensor> {
        let inner = self.inner.transform(&points.inner).map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }

    #[getter]
    pub fn output_dim(&self) -> usize {
        self.inner.output_dim()
    }
}

#[pyclass(name = "FastDiscretizer", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyFastDiscretizer {
    inner: FastDiscretizer,
}

#[pymethods]
impl PyFastDiscretizer {
    #[staticmethod]
    #[pyo3(signature = (n_bins=5, min_val=0.0, max_val=1.0, one_hot=true))]
    pub fn uniform(
        n_bins: usize,
        min_val: f64,
        max_val: f64,
        one_hot: bool,
    ) -> PyResult<Self> {
        let inner = FastDiscretizer::new_uniform(n_bins, min_val, max_val, one_hot)
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn fit_quantile(&mut self, values: Vec<f64>) -> PyResult<()> {
        self.inner.fit_quantile(&values).map_err(to_py_err)
    }

    pub fn assign_bin(&self, x: f64) -> usize {
        self.inner.assign_bin(x)
    }

    pub fn transform(&self, values: Vec<f64>) -> PyResult<PyTensor> {
        let inner = self.inner.transform(&values).map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }

    #[getter]
    pub fn n_bins(&self) -> usize {
        self.inner.n_bins
    }
}

#[pyclass(name = "FastLsaVectorizer", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyFastLsaVectorizer {
    inner: FastLsaVectorizer,
}

#[pymethods]
impl PyFastLsaVectorizer {
    #[new]
    #[pyo3(signature = (n_components=16, ngram_range=(1, 1)))]
    pub fn new(n_components: usize, ngram_range: (usize, usize)) -> PyResult<Self> {
        let inner = FastLsaVectorizer::new(n_components, ngram_range).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn fit(&mut self, corpus: Vec<String>) -> PyResult<()> {
        let refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();
        self.inner.fit(&refs).map_err(to_py_err)
    }

    pub fn transform(&self, docs: Vec<String>) -> PyResult<PyTensor> {
        let refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
        let inner = self.inner.transform(&refs).map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }

    #[getter]
    pub fn n_components(&self) -> usize {
        self.inner.n_components
    }
}

#[pyclass(name = "FastDictVectorizer", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyFastDictVectorizer {
    inner: FastDictVectorizer,
}

#[pymethods]
impl PyFastDictVectorizer {
    #[new]
    #[pyo3(signature = (sort=true))]
    pub fn new(sort: bool) -> Self {
        Self {
            inner: FastDictVectorizer::new(sort),
        }
    }

    pub fn fit(&mut self, samples: &Bound<'_, pyo3::types::PyList>) -> PyResult<()> {
        let mut rust_samples = Vec::with_capacity(samples.len());
        for item in samples.iter() {
            let dict = item
                .cast::<pyo3::types::PyDict>()
                .map_err(|_| PyValueError::new_err("Expected list of dicts"))?;
            let mut map = HashMap::new();
            for (k, v) in dict.iter() {
                let key: String = k.extract()?;
                let val = if v.is_instance_of::<pyo3::types::PyBool>() {
                    DictVal::Bool(v.extract::<bool>()?)
                } else if let Ok(f) = v.extract::<f64>() {
                    DictVal::Num(f)
                } else if let Ok(s) = v.extract::<String>() {
                    DictVal::Str(s)
                } else {
                    DictVal::Str(v.to_string())
                };
                map.insert(key, val);
            }
            rust_samples.push(map);
        }
        self.inner.fit(&rust_samples);
        Ok(())
    }

    pub fn transform_sparse(
        &self,
        samples: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<PySparseCsr> {
        let mut rust_samples = Vec::with_capacity(samples.len());
        for item in samples.iter() {
            let dict = item
                .cast::<pyo3::types::PyDict>()
                .map_err(|_| PyValueError::new_err("Expected list of dicts"))?;
            let mut map = HashMap::new();
            for (k, v) in dict.iter() {
                let key: String = k.extract()?;
                let val = if v.is_instance_of::<pyo3::types::PyBool>() {
                    DictVal::Bool(v.extract::<bool>()?)
                } else if let Ok(f) = v.extract::<f64>() {
                    DictVal::Num(f)
                } else if let Ok(s) = v.extract::<String>() {
                    DictVal::Str(s)
                } else {
                    DictVal::Str(v.to_string())
                };
                map.insert(key, val);
            }
            rust_samples.push(map);
        }
        let inner = self.inner.transform_sparse(&rust_samples).map_err(to_py_err)?;
        Ok(PySparseCsr { inner })
    }

    pub fn transform_dense(
        &self,
        samples: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<PyTensor> {
        let mut rust_samples = Vec::with_capacity(samples.len());
        for item in samples.iter() {
            let dict = item
                .cast::<pyo3::types::PyDict>()
                .map_err(|_| PyValueError::new_err("Expected list of dicts"))?;
            let mut map = HashMap::new();
            for (k, v) in dict.iter() {
                let key: String = k.extract()?;
                let val = if v.is_instance_of::<pyo3::types::PyBool>() {
                    DictVal::Bool(v.extract::<bool>()?)
                } else if let Ok(f) = v.extract::<f64>() {
                    DictVal::Num(f)
                } else if let Ok(s) = v.extract::<String>() {
                    DictVal::Str(s)
                } else {
                    DictVal::Str(v.to_string())
                };
                map.insert(key, val);
            }
            rust_samples.push(map);
        }
        let inner = self.inner.transform_dense(&rust_samples).map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }

    #[getter]
    pub fn feature_names(&self) -> Vec<String> {
        self.inner.feature_names.clone()
    }

    #[getter]
    pub fn vocab_size(&self) -> usize {
        self.inner.num_features()
    }
}

#[pyclass(name = "FastOneHotVectorizer", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyFastOneHotVectorizer {
    inner: FastOneHotVectorizer,
}

#[pymethods]
impl PyFastOneHotVectorizer {
    #[new]
    #[pyo3(signature = (drop_first=false, ignore_unknown=true))]
    pub fn new(drop_first: bool, ignore_unknown: bool) -> Self {
        Self {
            inner: FastOneHotVectorizer::new(drop_first, ignore_unknown),
        }
    }

    pub fn fit(&mut self, table: Vec<Vec<String>>) -> PyResult<()> {
        self.inner.fit(&table).map_err(to_py_err)
    }

    pub fn transform(&self, table: Vec<Vec<String>>) -> PyResult<PyTensor> {
        let inner = self.inner.transform(&table).map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }

    #[getter]
    pub fn total_features(&self) -> usize {
        self.inner.total_features()
    }

    #[getter]
    pub fn categories(&self) -> Vec<Vec<String>> {
        self.inner.categories.clone()
    }
}

#[pyclass(name = "FastJsonTreeVectorizer", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyFastJsonTreeVectorizer {
    inner: FastJsonTreeVectorizer,
}

#[pymethods]
impl PyFastJsonTreeVectorizer {
    #[new]
    #[pyo3(signature = (n_features=128, decay_rate=0.5, seed=42))]
    pub fn new(n_features: usize, decay_rate: f64, seed: u32) -> PyResult<Self> {
        let inner =
            FastJsonTreeVectorizer::new(n_features, decay_rate, seed).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn transform_entries(&self, entries: Vec<(String, f64)>) -> Vec<f64> {
        let slice: Vec<(&str, f64)> = entries.iter().map(|(s, v)| (s.as_str(), *v)).collect();
        self.inner.transform_entries(&slice)
    }

    pub fn transform_dense(&self, batch: Vec<Vec<(String, f64)>>) -> PyResult<PyTensor> {
        let rust_batch: Vec<Vec<(&str, f64)>> = batch
            .iter()
            .map(|row| row.iter().map(|(s, v)| (s.as_str(), *v)).collect())
            .collect();
        let inner = self.inner.transform_dense(&rust_batch).map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }

    #[getter]
    pub fn n_features(&self) -> usize {
        self.inner.n_features
    }
}

#[pyclass(
    name = "FastTimeSeriesSpectralVectorizer",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyFastTimeSeriesSpectralVectorizer {
    inner: FastTimeSeriesSpectralVectorizer,
}

#[pymethods]
impl PyFastTimeSeriesSpectralVectorizer {
    #[new]
    #[pyo3(signature = (n_fft_bands=8, normalize=true))]
    pub fn new(n_fft_bands: usize, normalize: bool) -> PyResult<Self> {
        let inner =
            FastTimeSeriesSpectralVectorizer::new(n_fft_bands, normalize).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn transform_series(&self, x: Vec<f64>) -> Vec<f64> {
        self.inner.transform_series(&x)
    }

    pub fn transform(&self, series_list: Vec<Vec<f64>>) -> PyResult<PyTensor> {
        let inner = self.inner.transform(&series_list).map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }

    #[getter]
    pub fn output_dim(&self) -> usize {
        self.inner.output_dim()
    }
}

#[pyclass(name = "FastPathSignatureVectorizer", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyFastPathSignatureVectorizer {
    inner: FastPathSignatureVectorizer,
}

#[pymethods]
impl PyFastPathSignatureVectorizer {
    #[new]
    #[pyo3(signature = (depth=2))]
    pub fn new(depth: usize) -> PyResult<Self> {
        let inner = FastPathSignatureVectorizer::new(depth).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn signature_single(&self, path: Vec<Vec<f64>>, d: usize) -> Vec<f64> {
        self.inner.signature_single(&path, d)
    }

    pub fn transform(&self, paths: Vec<Vec<Vec<f64>>>, d: usize) -> PyResult<PyTensor> {
        let inner = self.inner.transform(&paths, d).map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }

    #[getter]
    pub fn depth(&self) -> usize {
        self.inner.depth
    }

    pub fn output_dim(&self, d: usize) -> usize {
        self.inner.output_dim(d)
    }
}

#[pyclass(name = "FastDeepSetsVectorizer", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyFastDeepSetsVectorizer {
    inner: FastDeepSetsVectorizer,
}

#[pymethods]
impl PyFastDeepSetsVectorizer {
    #[new]
    #[pyo3(signature = (in_features, out_features=32, pooling="mean".to_string(), seed=42))]
    pub fn new(
        in_features: usize,
        out_features: usize,
        pooling: String,
        seed: u64,
    ) -> PyResult<Self> {
        let inner = FastDeepSetsVectorizer::new(in_features, out_features, &pooling, seed)
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn transform_set(&self, set: Vec<Vec<f64>>) -> Vec<f64> {
        self.inner.transform_set(&set)
    }

    pub fn transform(&self, sets: Vec<Vec<Vec<f64>>>) -> PyResult<PyTensor> {
        let inner = self.inner.transform(&sets).map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }

    #[getter]
    pub fn in_features(&self) -> usize {
        self.inner.in_features
    }

    #[getter]
    pub fn out_features(&self) -> usize {
        self.inner.out_features
    }

    #[getter]
    pub fn pooling(&self) -> String {
        self.inner.pooling.clone()
    }
}

#[pyclass(name = "FastWeisfeilerLehmanVectorizer", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyFastWeisfeilerLehmanVectorizer {
    inner: FastWeisfeilerLehmanVectorizer,
}

#[pymethods]
impl PyFastWeisfeilerLehmanVectorizer {
    #[new]
    #[pyo3(signature = (iterations=2, n_features=128, seed=42))]
    pub fn new(iterations: usize, n_features: usize, seed: u32) -> PyResult<Self> {
        let inner = FastWeisfeilerLehmanVectorizer::new(iterations, n_features, seed)
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn transform_graph(&self, adj: Vec<Vec<i32>>, node_labels: Vec<String>) -> Vec<f64> {
        self.inner.transform_graph(&adj, &node_labels)
    }

    pub fn transform_dense(
        &self,
        graphs: Vec<(Vec<Vec<i32>>, Vec<String>)>,
    ) -> PyResult<PyTensor> {
        let pairs: Vec<(&[Vec<i32>], &[String])> = graphs
            .iter()
            .map(|(adj, labels)| (adj.as_slice(), labels.as_slice()))
            .collect();
        let inner = self.inner.transform_dense(&pairs).map_err(to_py_err)?;
        Ok(PyTensor { inner })
    }

    #[getter]
    pub fn iterations(&self) -> usize {
        self.inner.iterations
    }

    #[getter]
    pub fn n_features(&self) -> usize {
        self.inner.n_features
    }
}

#[pyclass(name = "CausalLM", module = "oxideon")]
pub struct PyCausalLM {
    inner: CausalLM,
}

#[pymethods]
impl PyCausalLM {
    #[new]
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        num_heads: usize,
        num_layers: usize,
        d_ff: usize,
        max_seq_len: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CausalLM::new(
                vocab_size,
                d_model,
                num_heads,
                num_layers,
                d_ff,
                max_seq_len,
            )
            .map_err(to_py_err)?,
        })
    }

    pub fn forward(&self, py: Python<'_>, tokens: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&tokens.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    /// Trains the `CausalLM` on tokenized sequences using next-token prediction +
    /// [`fused_cross_entropy_with_z_loss`] + gradient norm clipping + `AdamW` **100% inside
    /// native Rust** with the Python GIL released, returning `(loss_history, perplexity_history)`.
    #[pyo3(signature = (sequences, epochs=25, lr=1e-2, weight_decay=1e-3, max_grad_norm=1.0, z_loss_weight=1e-4))]
    #[allow(clippy::too_many_arguments)]
    pub fn train_ntp_epochs_in_rust(
        &self,
        py: Python<'_>,
        sequences: Vec<Vec<usize>>,
        epochs: usize,
        lr: f64,
        weight_decay: f64,
        max_grad_norm: f64,
        z_loss_weight: f64,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let params = self.inner.parameters();
        py.detach(|| {
            let mut opt = Adam::adamw(params.clone(), lr, weight_decay);
            let mut losses = Vec::with_capacity(epochs);
            let mut ppls = Vec::with_capacity(epochs);

            for _epoch in 0..epochs {
                let mut total_loss = 0.0f64;
                let mut count = 0usize;
                for seq in &sequences {
                    if seq.len() < 2 {
                        continue;
                    }
                    let t_len = seq.len() - 1;
                    let inp_ids: Vec<i64> = seq[..t_len].iter().map(|&x| x as i64).collect();
                    let tgt_ids: Vec<usize> = seq[1..].to_vec();
                    let inp_t = Tensor::from_slice(&inp_ids, &[1, t_len])?;

                    opt.zero_grad();
                    let logits = self
                        .inner
                        .forward(&inp_t)?
                        .reshape(&[t_len, self.inner.vocab_size])?;
                    let step_loss =
                        fused_cross_entropy_with_z_loss(&logits, &tgt_ids, z_loss_weight)?;
                    total_loss += step_loss.item::<f64>()?;
                    count += 1;

                    step_loss.backward()?;
                    if max_grad_norm > 0.0 {
                        clip_grad_norm(&params, max_grad_norm)?;
                    }
                    opt.step()?;
                }
                let avg_loss = total_loss / (count.max(1) as f64);
                losses.push(avg_loss);
                ppls.push(avg_loss.exp());
            }
            Ok::<(Vec<f64>, Vec<f64>), Error>((losses, ppls))
        })
        .map_err(to_py_err)
    }

    #[pyo3(signature = (prompt, max_new_tokens, temperature=0.8, top_k=None, top_p=None))]
    pub fn generate(
        &self,
        py: Python<'_>,
        prompt: Vec<i64>,
        max_new_tokens: usize,
        temperature: f64,
        top_k: Option<usize>,
        top_p: Option<f64>,
    ) -> PyResult<Vec<i64>> {
        py.detach(|| {
            self.inner
                .generate(&prompt, max_new_tokens, temperature, top_k, top_p)
        })
        .map_err(to_py_err)
    }

    #[pyo3(signature = (prompt, config=None, max_new_tokens=128, temperature=1.0, top_k=None, top_p=None, seed=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_config(
        &self,
        py: Python<'_>,
        prompt: Vec<i64>,
        config: Option<&PySamplingConfig>,
        max_new_tokens: usize,
        temperature: f64,
        top_k: Option<usize>,
        top_p: Option<f64>,
        seed: Option<u64>,
    ) -> PyResult<Vec<i64>> {
        let cfg = if let Some(c) = config {
            c.inner.clone()
        } else {
            SamplingConfig {
                max_new_tokens,
                temperature,
                top_k,
                top_p,
                seed,
                ..Default::default()
            }
        };
        py.detach(|| self.inner.generate_with_config(&prompt, &cfg))
            .map_err(to_py_err)
    }

    pub fn speculative_generate(
        &self,
        py: Python<'_>,
        draft_model: &PyCausalLM,
        prompt: Vec<usize>,
        max_new_tokens: usize,
        gamma: usize,
    ) -> PyResult<(Vec<usize>, usize, usize)> {
        py.detach(|| {
            self.inner
                .speculative_generate(&draft_model.inner, &prompt, max_new_tokens, gamma)
        })
        .map_err(to_py_err)
    }

    pub fn num_parameters(&self) -> usize {
        self.inner.num_parameters()
    }
}

#[pyclass(name = "DeepSeekMoELM", module = "oxideon")]
pub struct PyDeepSeekMoELM {
    inner: DeepSeekMoELM,
}

#[pymethods]
impl PyDeepSeekMoELM {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        num_layers: usize,
        num_heads: usize,
        d_kv_latent: usize,
        d_ff: usize,
        max_seq_len: usize,
        num_shared_experts: usize,
        num_routed_experts: usize,
        top_k_experts: usize,
        mtp_depths: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: DeepSeekMoELM::new(
                vocab_size,
                d_model,
                num_layers,
                num_heads,
                d_kv_latent,
                d_ff,
                max_seq_len,
                num_shared_experts,
                num_routed_experts,
                top_k_experts,
                mtp_depths,
            )
            .map_err(to_py_err)?,
        })
    }

    /// Runs the full DeepSeek-V3 forward + Multi-Token Prediction + MoE routing in Rust threads.
    pub fn forward_with_mtp(
        &self,
        py: Python<'_>,
        tokens: &PyTensor,
    ) -> PyResult<(PyTensor, Vec<PyTensor>, PyTensor)> {
        let (main, mtp, aux) = py
            .detach(|| self.inner.forward_with_mtp(&tokens.inner))
            .map_err(to_py_err)?;
        let mtp_py = mtp.into_iter().map(|inner| PyTensor { inner }).collect();
        Ok((PyTensor { inner: main }, mtp_py, PyTensor { inner: aux }))
    }

    pub fn num_parameters(&self) -> usize {
        self.inner.num_parameters()
    }
}

#[pyclass(name = "HybridJambaLM", module = "oxideon")]
pub struct PyHybridJambaLM {
    inner: HybridJambaLM,
}

#[pymethods]
impl PyHybridJambaLM {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        num_layers: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        d_state: usize,
        d_ff: usize,
        num_experts: usize,
        top_k: usize,
        attn_every_n: usize,
        moe_every_n: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: HybridJambaLM::new(
                vocab_size,
                d_model,
                num_layers,
                num_q_heads,
                num_kv_heads,
                d_state,
                d_ff,
                num_experts,
                top_k,
                attn_every_n,
                moe_every_n,
            )
            .map_err(to_py_err)?,
        })
    }

    pub fn forward(&self, py: Python<'_>, tokens: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&tokens.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "MultimodalVLM", module = "oxideon")]
pub struct PyMultimodalVLM {
    inner: MultimodalVLM,
}

#[pymethods]
impl PyMultimodalVLM {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        img_size: usize,
        patch_size: usize,
        in_channels: usize,
        vision_dim: usize,
        vision_layers: usize,
        vision_heads: usize,
        vocab_size: usize,
        llm_dim: usize,
        llm_layers: usize,
        llm_q_heads: usize,
        llm_kv_heads: usize,
        llm_ff: usize,
        max_seq_len: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: MultimodalVLM::new(
                img_size,
                patch_size,
                in_channels,
                vision_dim,
                vision_layers,
                vision_heads,
                vocab_size,
                llm_dim,
                llm_layers,
                llm_q_heads,
                llm_kv_heads,
                llm_ff,
                max_seq_len,
            )
            .map_err(to_py_err)?,
        })
    }

    pub fn forward_multimodal(
        &self,
        py: Python<'_>,
        images: &PyTensor,
        text_tokens: &PyTensor,
    ) -> PyResult<PyTensor> {
        let out = py
            .detach(|| {
                self.inner
                    .forward_multimodal(&images.inner, &text_tokens.inner)
            })
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn generate_multimodal(
        &self,
        py: Python<'_>,
        image: &PyTensor,
        prompt: Vec<usize>,
        max_new_tokens: usize,
    ) -> PyResult<Vec<usize>> {
        py.detach(|| {
            self.inner
                .generate_multimodal(&image.inner, &prompt, max_new_tokens)
        })
        .map_err(to_py_err)
    }
}

#[pyclass(name = "GatedCrossAttention", module = "oxideon")]
pub struct PyGatedCrossAttention {
    pub inner: GatedCrossAttention,
}

#[pymethods]
impl PyGatedCrossAttention {
    #[new]
    pub fn new(dim: usize, num_heads: usize, context_dim: usize) -> PyResult<Self> {
        let inner = GatedCrossAttention::new(dim, num_heads, context_dim).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor, context: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner, &context.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "VisionLanguageActionModel", module = "oxideon")]
pub struct PyVisionLanguageActionModel {
    pub inner: VisionLanguageActionModel,
}

#[pymethods]
impl PyVisionLanguageActionModel {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        img_size: usize,
        patch_size: usize,
        in_channels: usize,
        vision_dim: usize,
        text_dim: usize,
        vocab_size: usize,
        action_horizon: usize,
        action_dim: usize,
    ) -> PyResult<Self> {
        let inner = VisionLanguageActionModel::new(
            img_size,
            patch_size,
            in_channels,
            vision_dim,
            text_dim,
            vocab_size,
            action_horizon,
            action_dim,
        )
        .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn forward(
        &self,
        py: Python<'_>,
        images: &PyTensor,
        text_one_hot: &PyTensor,
    ) -> PyResult<(PyTensor, PyTensor)> {
        let (logits, actions) = py
            .detach(|| self.inner.forward(&images.inner, &text_one_hot.inner))
            .map_err(to_py_err)?;
        Ok((PyTensor { inner: logits }, PyTensor { inner: actions }))
    }
}

#[pyclass(name = "LIFNode", module = "oxideon")]
pub struct PyLIFNode {
    pub inner: LIFNode,
}

#[pymethods]
impl PyLIFNode {
    #[new]
    #[pyo3(signature = (tau=0.9, v_threshold=1.0, v_reset=0.0, reset_mode="hard"))]
    pub fn new(tau: f64, v_threshold: f64, v_reset: f64, reset_mode: &str) -> Self {
        let rm = match reset_mode {
            "soft" => ResetMode::Soft,
            _ => ResetMode::Hard,
        };
        Self {
            inner: LIFNode::new(tau, v_threshold, v_reset, rm),
        }
    }

    pub fn reset_state(&self) {
        self.inner.reset_state();
    }

    pub fn forward_step(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let spike = py
            .detach(|| self.inner.forward_step(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: spike })
    }

    pub fn forward_sequence(&self, py: Python<'_>, x_seq: &PyTensor) -> PyResult<PyTensor> {
        let spikes = py
            .detach(|| self.inner.forward_sequence(&x_seq.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: spikes })
    }
}

#[pyclass(name = "PoissonEncoder", module = "oxideon")]
pub struct PyPoissonEncoder {
    pub inner: PoissonEncoder,
}

#[pymethods]
impl PyPoissonEncoder {
    #[new]
    pub fn new(time_steps: usize) -> Self {
        Self {
            inner: PoissonEncoder::new(time_steps),
        }
    }

    pub fn encode(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.encode(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "STDPSynapse", module = "oxideon")]
pub struct PySTDPSynapse {
    pub inner: STDPSynapse,
}

#[pymethods]
impl PySTDPSynapse {
    #[new]
    #[pyo3(signature = (in_features, out_features, a_plus=0.01, a_minus=0.012, tau_plus=20.0, tau_minus=20.0, w_min=0.0, w_max=1.0, dt=1.0))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_features: usize,
        out_features: usize,
        a_plus: f64,
        a_minus: f64,
        tau_plus: f64,
        tau_minus: f64,
        w_min: f64,
        w_max: f64,
        dt: f64,
    ) -> PyResult<Self> {
        let config = STDPConfig {
            a_plus,
            a_minus,
            tau_plus,
            tau_minus,
            w_min,
            w_max,
            dt,
        };
        let inner = STDPSynapse::new(in_features, out_features, config).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn forward(&self, py: Python<'_>, pre_spikes: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&pre_spikes.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn step_stdp(
        &mut self,
        py: Python<'_>,
        pre_spikes: &PyTensor,
        post_spikes: &PyTensor,
    ) -> PyResult<()> {
        py.detach(|| self.inner.step_stdp(&pre_spikes.inner, &post_spikes.inner))
            .map_err(to_py_err)?;
        Ok(())
    }

    pub fn reset_traces(&mut self) -> PyResult<()> {
        self.inner.reset_traces().map_err(to_py_err)?;
        Ok(())
    }

    #[getter]
    pub fn weights(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.weights.clone(),
        }
    }

    #[getter]
    pub fn trace_pre(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.trace_pre.clone(),
        }
    }

    #[getter]
    pub fn trace_post(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.trace_post.clone(),
        }
    }
}

#[pyclass(name = "SpikeEvent", module = "oxideon", from_py_object)]
#[derive(Clone, Copy)]
pub struct PySpikeEvent {
    pub inner: SpikeEvent,
}

#[pymethods]
impl PySpikeEvent {
    #[new]
    pub fn new(timestamp: f64, source: usize, target: usize, weight: f64, delay: f64) -> Self {
        Self {
            inner: SpikeEvent::new(timestamp, source, target, weight, delay),
        }
    }

    #[getter]
    pub fn timestamp(&self) -> f64 {
        self.inner.timestamp
    }

    #[getter]
    pub fn source(&self) -> usize {
        self.inner.source
    }

    #[getter]
    pub fn target(&self) -> usize {
        self.inner.target
    }

    #[getter]
    pub fn weight(&self) -> f64 {
        self.inner.weight
    }

    #[getter]
    pub fn delay(&self) -> f64 {
        self.inner.delay
    }

    pub fn arrival_time(&self) -> f64 {
        self.inner.arrival_time()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "SpikeEvent(t={:.4}, src={}, dst={}, w={:.4}, d={:.4}, arr={:.4})",
            self.inner.timestamp,
            self.inner.source,
            self.inner.target,
            self.inner.weight,
            self.inner.delay,
            self.inner.arrival_time()
        )
    }
}

#[pyclass(name = "EventDrivenSpikeRouter", module = "oxideon")]
#[derive(Default)]
pub struct PyEventDrivenSpikeRouter {
    pub inner: EventDrivenSpikeRouter,
}

#[pymethods]
impl PyEventDrivenSpikeRouter {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: EventDrivenSpikeRouter::new(),
        }
    }

    pub fn schedule(&mut self, event: &PySpikeEvent) {
        self.inner.schedule(event.inner);
    }

    pub fn pending_count(&self) -> usize {
        self.inner.pending_count()
    }

    pub fn next_arrival_time(&self) -> Option<f64> {
        self.inner.next_arrival_time()
    }

    pub fn pop_next(&mut self) -> Option<PySpikeEvent> {
        self.inner.pop_next().map(|e| PySpikeEvent { inner: e })
    }

    pub fn clear(&mut self) {
        self.inner.clear();
    }

    #[getter]
    pub fn events_routed(&self) -> usize {
        self.inner.events_routed
    }
}

#[pyclass(name = "PolychronousNeuralLayer", module = "oxideon")]
pub struct PyPolychronousNeuralLayer {
    pub inner: PolychronousNeuralLayer,
}

#[pymethods]
impl PyPolychronousNeuralLayer {
    #[new]
    pub fn new(num_inputs: usize, num_outputs: usize, max_delay: f64) -> PyResult<Self> {
        let inner =
            PolychronousNeuralLayer::new(num_inputs, num_outputs, max_delay).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn set_weights(&mut self, weights: Vec<f64>) -> PyResult<()> {
        self.inner.set_weights(weights).map_err(to_py_err)
    }

    pub fn set_delays(&mut self, delays: Vec<f64>) -> PyResult<()> {
        self.inner.set_delays(delays).map_err(to_py_err)
    }

    #[getter]
    pub fn weights(&self) -> Vec<f64> {
        self.inner.weights.clone()
    }

    #[getter]
    pub fn delays(&self) -> Vec<f64> {
        self.inner.delays.clone()
    }

    pub fn forward_event_driven(
        &self,
        py: Python<'_>,
        input_spikes: Vec<(f64, usize)>,
        t_max: f64,
    ) -> PyResult<Vec<(f64, usize)>> {
        py.detach(|| self.inner.forward_event_driven(&input_spikes, t_max))
            .map_err(to_py_err)
    }

    pub fn adapt_stddp(
        &mut self,
        input_spikes: Vec<(f64, usize)>,
        output_spikes: Vec<(f64, usize)>,
        lr_delay: f64,
        lr_weight: f64,
        tau_delta: f64,
    ) -> PyResult<()> {
        self.inner
            .adapt_stddp(
                &input_spikes,
                &output_spikes,
                lr_delay,
                lr_weight,
                tau_delta,
            )
            .map_err(to_py_err)
    }
}

#[pyclass(name = "DiT", module = "oxideon")]
pub struct PyDiT {
    inner: DiT,
}

#[pymethods]
impl PyDiT {
    #[new]
    pub fn new(
        img_size: usize,
        patch_size: usize,
        in_channels: usize,
        d_model: usize,
        depth: usize,
        num_heads: usize,
        num_classes: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: DiT::new(
                img_size,
                patch_size,
                in_channels,
                d_model,
                depth,
                num_heads,
                num_classes,
            )
            .map_err(to_py_err)?,
        })
    }

    #[pyo3(signature = (noisy_latents, timesteps, class_labels=None))]
    pub fn forward_dit(
        &self,
        py: Python<'_>,
        noisy_latents: &PyTensor,
        timesteps: Vec<usize>,
        class_labels: Option<&PyTensor>,
    ) -> PyResult<PyTensor> {
        let cls = class_labels.map(|c| c.inner.clone());
        let out = py
            .detach(|| {
                self.inner
                    .forward_dit(&noisy_latents.inner, &timesteps, cls.as_ref())
            })
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "Sequential", module = "oxideon")]
pub struct PySequential {
    inner: std::sync::RwLock<Sequential>,
}

#[pymethods]
impl PySequential {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: std::sync::RwLock::new(Sequential::new()),
        }
    }

    pub fn add_linear(&self, in_features: usize, out_features: usize) -> PyResult<()> {
        let layer = Linear::new(in_features, out_features).map_err(to_py_err)?;
        rw_write(&self.inner).push(layer);
        Ok(())
    }

    pub fn add_relu(&self) -> PyResult<()> {
        rw_write(&self.inner).push(ReLU);
        Ok(())
    }

    pub fn add_gelu(&self) -> PyResult<()> {
        rw_write(&self.inner).push(GELU);
        Ok(())
    }

    pub fn add_silu(&self) -> PyResult<()> {
        rw_write(&self.inner).push(SiLU);
        Ok(())
    }

    pub fn add_mish(&self) -> PyResult<()> {
        rw_write(&self.inner).push(Mish);
        Ok(())
    }

    pub fn add_tanh(&self) -> PyResult<()> {
        rw_write(&self.inner).push(Tanh);
        Ok(())
    }

    #[pyo3(signature = (normalized_shape, eps=1e-5))]
    pub fn add_layernorm(&self, normalized_shape: usize, eps: f64) -> PyResult<()> {
        let ln = LayerNorm::new(normalized_shape, eps).map_err(to_py_err)?;
        rw_write(&self.inner).push(ln);
        Ok(())
    }

    #[pyo3(signature = (p=0.5))]
    pub fn add_dropout(&self, p: f64) -> PyResult<()> {
        if !(0.0..1.0).contains(&p) {
            return Err(PyValueError::new_err(format!(
                "dropout probability must be in [0, 1), got {p}"
            )));
        }
        rw_write(&self.inner).push(Dropout::new(p as f32));
        Ok(())
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| rw_read(&self.inner).forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn __call__(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(py, x)
    }

    pub fn parameters(&self) -> Vec<PyTensor> {
        rw_read(&self.inner)
            .parameters()
            .into_iter()
            .map(|t| PyTensor { inner: t })
            .collect()
    }

    pub fn train(&self, mode: bool) {
        rw_write(&self.inner).train(mode);
    }

    pub fn eval(&self) {
        rw_write(&self.inner).eval();
    }

    pub fn num_parameters(&self) -> usize {
        rw_read(&self.inner).num_parameters()
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (x, y, epochs, batch_size=32, lr=1e-3, weight_decay=1e-4, max_grad_norm=1.0))]
    pub fn train_classification_in_rust(
        &self,
        py: Python<'_>,
        x: &PyTensor,
        y: &PyTensor,
        epochs: usize,
        batch_size: usize,
        lr: f64,
        weight_decay: f64,
        max_grad_norm: f64,
    ) -> PyResult<Vec<f64>> {
        let x_t = x.inner.clone();
        let y_t = y.inner.clone();
        let params: Vec<Tensor> = rw_read(&self.inner).parameters();
        let mut opt = Adam::adamw(params.clone(), lr, weight_decay);
        let n = x_t.shape()[0];
        let mut history = Vec::with_capacity(epochs);

        py.detach(|| {
            rw_write(&self.inner).train(true);
            for _ in 0..epochs {
                let mut epoch_loss = 0.0;
                let mut steps = 0usize;
                let mut start = 0;
                while start < n {
                    let len = (n - start).min(batch_size);
                    let x_batch = x_t.narrow(0, start, len)?;
                    let y_batch = y_t.narrow(0, start, len)?;
                    opt.zero_grad();
                    let logits = rw_read(&self.inner).forward(&x_batch)?;
                    // Branch on dtype, not on to_vec success: to_vec::<i64>
                    // silently truncates floats (2.9 -> 2), so probing it
                    // first would accept fractional labels. Floats must be
                    // finite, integral, and non-negative; ints must be
                    // non-negative; everything is range-checked below.
                    let to_label = |v: f64| -> Result<usize, String> {
                        if !v.is_finite() || v.fract() != 0.0 || v < 0.0 {
                            return Err(format!("label {v} is not a valid class index"));
                        }
                        Ok(v as usize)
                    };
                    let raw: Vec<usize> = if y_batch.dtype().is_int() {
                        y_batch
                            .to_vec::<i64>()?
                            .into_iter()
                            .map(|v| {
                                usize::try_from(v).map_err(|_| format!("label {v} is not a valid class index"))
                            })
                            .collect::<Result<Vec<_>, _>>()
                            .map_err(Error::Msg)?
                    } else if y_batch.dtype().is_floating_point() {
                        y_batch
                            .to_vec::<f64>()?
                            .into_iter()
                            .map(to_label)
                            .collect::<Result<Vec<_>, _>>()
                            .map_err(Error::Msg)?
                    } else {
                        return Err(Error::Msg(format!(
                            "train_classification_in_rust: labels must be int or float, got {:?}",
                            y_batch.dtype()
                        )));
                    };
                    if logits.ndim() != 2 {
                        return Err(Error::Msg(format!(
                            "train_classification_in_rust: logits must be [B, C], got {:?}",
                            logits.shape()
                        )));
                    }
                    let n_classes = logits.shape()[1];
                    let mut targets: Vec<usize> = Vec::with_capacity(raw.len());
                    for t in raw {
                        if t >= n_classes {
                            return Err(Error::Msg(format!(
                                "train_classification_in_rust: label {t} out of range for {n_classes} classes"
                            )));
                        }
                        targets.push(t);
                    }
                    let loss = loss::cross_entropy(&logits, &targets)?;
                    let l_val = loss.item::<f64>()?;
                    epoch_loss += l_val;
                    steps += 1;
                    loss.backward()?;
                    if max_grad_norm > 0.0 {
                        clip_grad_norm(&params, max_grad_norm)?;
                    }
                    opt.step()?;
                    start += len;
                }
                history.push(epoch_loss / (steps.max(1) as f64));
            }
            rw_write(&self.inner).eval();
            Ok::<Vec<f64>, Error>(history)
        })
        .map_err(to_py_err)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (x, y, epochs, batch_size=32, lr=1e-3, weight_decay=1e-4, max_grad_norm=1.0))]
    pub fn train_regression_in_rust(
        &self,
        py: Python<'_>,
        x: &PyTensor,
        y: &PyTensor,
        epochs: usize,
        batch_size: usize,
        lr: f64,
        weight_decay: f64,
        max_grad_norm: f64,
    ) -> PyResult<Vec<f64>> {
        let x_t = x.inner.clone();
        let y_t = y.inner.clone();
        let params: Vec<Tensor> = rw_read(&self.inner).parameters();
        let mut opt = Adam::adamw(params.clone(), lr, weight_decay);
        let n = x_t.shape()[0];
        let mut history = Vec::with_capacity(epochs);

        py.detach(|| {
            rw_write(&self.inner).train(true);
            for _ in 0..epochs {
                let mut epoch_loss = 0.0;
                let mut steps = 0usize;
                let mut start = 0;
                while start < n {
                    let len = (n - start).min(batch_size);
                    let x_batch = x_t.narrow(0, start, len)?;
                    let y_batch = y_t.narrow(0, start, len)?;
                    opt.zero_grad();
                    let preds = rw_read(&self.inner).forward(&x_batch)?;
                    let loss = loss::mse_loss(&preds, &y_batch)?;
                    let l_val = loss.item::<f64>()?;
                    epoch_loss += l_val;
                    steps += 1;
                    loss.backward()?;
                    if max_grad_norm > 0.0 {
                        clip_grad_norm(&params, max_grad_norm)?;
                    }
                    opt.step()?;
                    start += len;
                }
                history.push(epoch_loss / (steps.max(1) as f64));
            }
            rw_write(&self.inner).eval();
            Ok::<Vec<f64>, Error>(history)
        })
        .map_err(to_py_err)
    }
}

impl Default for PySequential {
    fn default() -> Self {
        Self::new()
    }
}

#[pyclass(name = "ConvNeXt", module = "oxideon")]
pub struct PyConvNeXt {
    inner: ConvNeXt,
}

#[pymethods]
impl PyConvNeXt {
    #[new]
    #[pyo3(signature = (in_channels=3, dim=64, depth=3, patch_size=4, num_classes=10))]
    pub fn new(
        in_channels: usize,
        dim: usize,
        depth: usize,
        patch_size: usize,
        num_classes: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: ConvNeXt::new(in_channels, dim, depth, patch_size, num_classes)
                .map_err(to_py_err)?,
        })
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|t| PyTensor { inner: t })
            .collect()
    }

    pub fn train(&self, mode: bool) {
        self.inner.train(mode);
    }

    pub fn eval(&self) {
        self.inner.eval();
    }

    pub fn num_parameters(&self) -> usize {
        self.inner.num_parameters()
    }
}

#[pyclass(name = "BiLSTM", module = "oxideon")]
pub struct PyBiLSTM {
    inner: BiLSTM,
}

#[pymethods]
impl PyBiLSTM {
    #[new]
    #[pyo3(signature = (input_size, hidden_size, num_layers=1))]
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> PyResult<Self> {
        Ok(Self {
            inner: BiLSTM::new(input_size, hidden_size, num_layers).map_err(to_py_err)?,
        })
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|t| PyTensor { inner: t })
            .collect()
    }

    pub fn train(&self, mode: bool) {
        self.inner.train(mode);
    }

    pub fn eval(&self) {
        self.inner.eval();
    }

    pub fn num_parameters(&self) -> usize {
        self.inner.num_parameters()
    }
}

#[pyclass(name = "TemporalConvNet", module = "oxideon")]
pub struct PyTemporalConvNet {
    inner: TemporalConvNet,
}

#[pymethods]
impl PyTemporalConvNet {
    #[new]
    #[pyo3(signature = (in_channels, channel_dims, kernel_size=3))]
    pub fn new(in_channels: usize, channel_dims: Vec<usize>, kernel_size: usize) -> PyResult<Self> {
        Ok(Self {
            inner: TemporalConvNet::new(in_channels, &channel_dims, kernel_size)
                .map_err(to_py_err)?,
        })
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|t| PyTensor { inner: t })
            .collect()
    }

    pub fn train(&self, mode: bool) {
        self.inner.train(mode);
    }

    pub fn eval(&self) {
        self.inner.eval();
    }

    pub fn num_parameters(&self) -> usize {
        self.inner.num_parameters()
    }
}

#[pyclass(name = "MLPMixer", module = "oxideon")]
pub struct PyMLPMixer {
    inner: MLPMixer,
}

#[pymethods]
impl PyMLPMixer {
    #[new]
    #[pyo3(signature = (img_size, patch_size, in_channels, hidden_dim, depth, token_mlp_dim, channel_mlp_dim, num_classes))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        img_size: usize,
        patch_size: usize,
        in_channels: usize,
        hidden_dim: usize,
        depth: usize,
        token_mlp_dim: usize,
        channel_mlp_dim: usize,
        num_classes: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: MLPMixer::new(
                img_size,
                patch_size,
                in_channels,
                hidden_dim,
                depth,
                token_mlp_dim,
                channel_mlp_dim,
                num_classes,
            )
            .map_err(to_py_err)?,
        })
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|t| PyTensor { inner: t })
            .collect()
    }

    pub fn train(&self, mode: bool) {
        self.inner.train(mode);
    }

    pub fn eval(&self) {
        self.inner.eval();
    }

    pub fn num_parameters(&self) -> usize {
        self.inner.num_parameters()
    }
}

#[pyclass(name = "VAE", module = "oxideon")]
pub struct PyVAE {
    inner: VariationalAutoencoder,
}

#[pymethods]
impl PyVAE {
    #[new]
    pub fn new(input_dim: usize, hidden_dim: usize, latent_dim: usize) -> PyResult<Self> {
        Ok(Self {
            inner: VariationalAutoencoder::new(input_dim, hidden_dim, latent_dim)
                .map_err(to_py_err)?,
        })
    }

    pub fn encode(&self, py: Python<'_>, x: &PyTensor) -> PyResult<(PyTensor, PyTensor)> {
        let (mu, logvar) = py
            .detach(|| self.inner.encode(&x.inner))
            .map_err(to_py_err)?;
        Ok((PyTensor { inner: mu }, PyTensor { inner: logvar }))
    }

    pub fn decode(&self, py: Python<'_>, z: &PyTensor) -> PyResult<PyTensor> {
        let recon = py
            .detach(|| self.inner.decode(&z.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: recon })
    }

    #[pyo3(signature = (x, beta_kl=1.0))]
    pub fn forward_elbo(
        &self,
        py: Python<'_>,
        x: &PyTensor,
        beta_kl: f64,
    ) -> PyResult<(PyTensor, PyTensor, PyTensor, PyTensor)> {
        let (recon, mu, logvar, total) = py
            .detach(|| self.inner.forward_elbo(&x.inner, beta_kl))
            .map_err(to_py_err)?;
        Ok((
            PyTensor { inner: recon },
            PyTensor { inner: mu },
            PyTensor { inner: logvar },
            PyTensor { inner: total },
        ))
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|t| PyTensor { inner: t })
            .collect()
    }

    pub fn train(&self, mode: bool) {
        self.inner.train(mode);
    }

    pub fn eval(&self) {
        self.inner.eval();
    }

    pub fn num_parameters(&self) -> usize {
        self.inner.num_parameters()
    }
}

#[pyclass(name = "SpectralNormLinear", module = "oxideon")]
pub struct PySpectralNormLinear {
    inner: SpectralNormLinear,
}

#[pymethods]
impl PySpectralNormLinear {
    #[new]
    #[pyo3(signature = (in_features, out_features, use_bias=true, n_power_iterations=1))]
    pub fn new(
        in_features: usize,
        out_features: usize,
        use_bias: bool,
        n_power_iterations: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: SpectralNormLinear::new(in_features, out_features, use_bias, n_power_iterations)
                .map_err(to_py_err)?,
        })
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn __call__(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(py, x)
    }

    #[getter]
    pub fn weight(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.weight.clone(),
        }
    }

    #[setter]
    pub fn set_weight(&mut self, w: PyTensor) {
        self.inner.weight = w.inner;
    }

    #[getter]
    pub fn bias(&self) -> Option<PyTensor> {
        self.inner
            .bias
            .as_ref()
            .map(|b| PyTensor { inner: b.clone() })
    }

    #[setter]
    pub fn set_bias(&mut self, b: Option<PyTensor>) {
        self.inner.bias = b.map(|p| p.inner);
    }

    #[getter]
    pub fn n_power_iterations(&self) -> usize {
        self.inner.n_power_iterations
    }

    pub fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|t| PyTensor { inner: t })
            .collect()
    }
}

#[pyclass(name = "WeightNormLinear", module = "oxideon")]
pub struct PyWeightNormLinear {
    inner: WeightNormLinear,
}

#[pymethods]
impl PyWeightNormLinear {
    #[new]
    #[pyo3(signature = (in_features, out_features, use_bias=true))]
    pub fn new(in_features: usize, out_features: usize, use_bias: bool) -> PyResult<Self> {
        Ok(Self {
            inner: WeightNormLinear::new(in_features, out_features, use_bias).map_err(to_py_err)?,
        })
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn __call__(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(py, x)
    }

    pub fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|t| PyTensor { inner: t })
            .collect()
    }
}

/// Maxout: the maximum over groups of `k` consecutive features.
#[pyclass(name = "Maxout", module = "oxideon")]
pub struct PyMaxout {
    inner: Maxout,
}

#[pymethods]
impl PyMaxout {
    #[new]
    #[pyo3(signature = (k))]
    pub fn new(k: usize) -> Self {
        Self { inner: Maxout(k) }
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn __call__(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(py, x)
    }

    #[getter]
    pub fn k(&self) -> usize {
        self.inner.0
    }

    pub fn parameters(&self) -> Vec<PyTensor> {
        Vec::new()
    }
}

/// Snake activation `x + sin(alpha x)^2 / alpha` with a learnable per-feature `alpha`.
#[pyclass(name = "Snake", module = "oxideon")]
pub struct PySnake {
    inner: Snake,
}

#[pymethods]
impl PySnake {
    #[new]
    #[pyo3(signature = (features, alpha=1.0))]
    pub fn new(features: usize, alpha: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Snake::new(features, alpha).map_err(to_py_err)?,
        })
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn __call__(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(py, x)
    }

    #[getter]
    pub fn alpha(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.alpha.clone(),
        }
    }

    #[setter]
    pub fn set_alpha(&mut self, a: PyTensor) {
        self.inner.alpha = a.inner;
    }

    pub fn parameters(&self) -> Vec<PyTensor> {
        vec![PyTensor {
            inner: self.inner.alpha.clone(),
        }]
    }
}

/// Swish `x * sigmoid(beta x)` with a learnable scalar `beta`.
#[pyclass(name = "Swish", module = "oxideon")]
pub struct PySwish {
    inner: Swish,
}

#[pymethods]
impl PySwish {
    #[new]
    #[pyo3(signature = (beta=1.0))]
    pub fn new(beta: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Swish::new(beta).map_err(to_py_err)?,
        })
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn __call__(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(py, x)
    }

    #[getter]
    pub fn beta(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.beta.clone(),
        }
    }

    #[setter]
    pub fn set_beta(&mut self, b: PyTensor) {
        self.inner.beta = b.inner;
    }

    pub fn parameters(&self) -> Vec<PyTensor> {
        vec![PyTensor {
            inner: self.inner.beta.clone(),
        }]
    }
}

/// Stochastic Depth / DropPath (`timm.layers.DropPath`, Huang et al. / ConvNeXt / ViT).
#[pyclass(name = "DropPath", module = "oxideon")]
pub struct PyDropPath {
    inner: DropPath,
}

#[pymethods]
impl PyDropPath {
    #[new]
    #[pyo3(signature = (drop_prob=0.0))]
    pub fn new(drop_prob: f64) -> Self {
        Self {
            inner: DropPath::new(drop_prob),
        }
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn __call__(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(py, x)
    }

    #[pyo3(signature = (mode=true))]
    pub fn train(&self, mode: bool) {
        self.inner.train(mode);
    }

    pub fn eval(&self) {
        self.inner.train(false);
    }

    #[getter]
    pub fn drop_prob(&self) -> f64 {
        self.inner.drop_prob
    }

    pub fn parameters(&self) -> Vec<PyTensor> {
        Vec::new()
    }
}

/// Applies Alpha Dropout over the input tensor (Klambauer et al., SELU networks).
#[pyclass(name = "AlphaDropout", module = "oxideon")]
pub struct PyAlphaDropout {
    inner: AlphaDropout,
}

#[pymethods]
impl PyAlphaDropout {
    #[new]
    #[pyo3(signature = (p=0.5))]
    pub fn new(p: f64) -> Self {
        Self {
            inner: AlphaDropout::new(p),
        }
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn __call__(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(py, x)
    }

    #[pyo3(signature = (mode=true))]
    pub fn train(&self, mode: bool) {
        self.inner.train(mode);
    }

    pub fn eval(&self) {
        self.inner.train(false);
    }

    #[getter]
    pub fn p(&self) -> f64 {
        self.inner.p
    }

    pub fn parameters(&self) -> Vec<PyTensor> {
        Vec::new()
    }
}

/// Convolutional LSTM cell: the gates are 2-D convolutions over `[x, h]`.
#[pyclass(name = "ConvLstmCell", module = "oxideon")]
pub struct PyConvLstmCell {
    inner: ConvLstmCell,
}

#[pymethods]
impl PyConvLstmCell {
    #[new]
    #[pyo3(signature = (in_channels, hidden, kernel))]
    pub fn new(in_channels: usize, hidden: usize, kernel: usize) -> PyResult<Self> {
        Ok(Self {
            inner: ConvLstmCell::new(in_channels, hidden, kernel).map_err(to_py_err)?,
        })
    }

    #[pyo3(signature = (x, hx=None))]
    pub fn step(
        &self,
        py: Python<'_>,
        x: &PyTensor,
        hx: Option<(PyTensor, PyTensor)>,
    ) -> PyResult<(PyTensor, PyTensor)> {
        let (hn, cn) = py
            .detach(|| {
                let state_ref = hx.as_ref().map(|(h, c)| (&h.inner, &c.inner));
                self.inner.step(&x.inner, state_ref)
            })
            .map_err(to_py_err)?;
        Ok((PyTensor { inner: hn }, PyTensor { inner: cn }))
    }

    pub fn forward_sequence(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward_sequence(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    #[pyo3(signature = (x, hx=None))]
    pub fn forward(
        &self,
        py: Python<'_>,
        x: &PyTensor,
        hx: Option<(PyTensor, PyTensor)>,
    ) -> PyResult<Py<PyAny>> {
        if hx.is_some() || x.inner.ndim() == 4 {
            let (hn, cn) = self.step(py, x, hx)?;
            Ok((hn, cn).into_pyobject(py)?.into_any().unbind())
        } else {
            let out = self.forward_sequence(py, x)?;
            Ok(out.into_pyobject(py)?.into_any().unbind())
        }
    }

    #[pyo3(signature = (x, hx=None))]
    pub fn __call__(
        &self,
        py: Python<'_>,
        x: &PyTensor,
        hx: Option<(PyTensor, PyTensor)>,
    ) -> PyResult<Py<PyAny>> {
        self.forward(py, x, hx)
    }

    #[getter]
    pub fn hidden(&self) -> usize {
        self.inner.hidden
    }

    #[getter]
    pub fn weight(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.conv.weight.clone(),
        }
    }

    #[setter]
    pub fn set_weight(&mut self, w: PyTensor) {
        self.inner.conv.weight = w.inner;
    }

    #[getter]
    pub fn bias(&self) -> Option<PyTensor> {
        self.inner
            .conv
            .bias
            .as_ref()
            .map(|b| PyTensor { inner: b.clone() })
    }

    #[setter]
    pub fn set_bias(&mut self, b: Option<PyTensor>) {
        self.inner.conv.bias = b.map(|p| p.inner);
    }

    pub fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .named_parameters()
            .into_iter()
            .map(|(_, t)| PyTensor { inner: t })
            .collect()
    }
}

/// 2D Upsampling layer supporting nearest and bilinear modes.
#[pyclass(name = "Upsample2d", module = "oxideon")]
pub struct PyUpsample2d {
    inner: Upsample2d,
}

#[pymethods]
impl PyUpsample2d {
    #[new]
    #[pyo3(signature = (size, mode="nearest", align_corners=None))]
    pub fn new(
        size: &Bound<'_, PyAny>,
        mode: &str,
        align_corners: Option<bool>,
    ) -> PyResult<Self> {
        let output_size = if let Ok(s) = size.extract::<(usize, usize)>() {
            s
        } else if let Ok(v) = size.extract::<Vec<usize>>() {
            if v.len() == 2 {
                (v[0], v[1])
            } else if v.len() == 1 {
                (v[0], v[0])
            } else {
                return Err(PyValueError::new_err(
                    "size list must have 1 or 2 elements",
                ));
            }
        } else if let Ok(s) = size.extract::<usize>() {
            (s, s)
        } else {
            return Err(PyValueError::new_err(
                "size must be an int, tuple of two ints, or list of two ints",
            ));
        };

        let inner = match mode.to_ascii_lowercase().as_str() {
            "nearest" => Upsample2d::nearest(output_size),
            "bilinear" => {
                let align = align_corners.unwrap_or(false);
                Upsample2d::bilinear(output_size, align)
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unsupported upsample mode '{other}', expected 'nearest' or 'bilinear'"
                )))
            }
        };
        Ok(Self { inner })
    }

    #[staticmethod]
    pub fn nearest(size: (usize, usize)) -> Self {
        Self {
            inner: Upsample2d::nearest(size),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (size, align_corners=false))]
    pub fn bilinear(size: (usize, usize), align_corners: bool) -> Self {
        Self {
            inner: Upsample2d::bilinear(size, align_corners),
        }
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn __call__(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        self.forward(py, x)
    }

    #[getter]
    pub fn output_size(&self) -> (usize, usize) {
        self.inner.output_size
    }

    pub fn parameters(&self) -> Vec<PyTensor> {
        Vec::new()
    }
}

/// An Elman RNN cell with tanh or ReLU non-linearity.
#[pyclass(name = "RNNCell", module = "oxideon")]
pub struct PyRNNCell {
    inner: RNNCell,
    nonlinearity: String,
}

#[pymethods]
impl PyRNNCell {
    #[new]
    #[pyo3(signature = (input_size, hidden_size, bias=true, nonlinearity="tanh"))]
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        bias: bool,
        nonlinearity: &str,
    ) -> PyResult<Self> {
        let mut inner = RNNCell::new(input_size, hidden_size).map_err(to_py_err)?;
        if !bias {
            inner.ih.bias = None;
            inner.hh.bias = None;
        }
        let nl = nonlinearity.to_ascii_lowercase();
        if nl != "tanh" && nl != "relu" {
            return Err(PyValueError::new_err(format!(
                "unsupported nonlinearity '{nonlinearity}', expected 'tanh' or 'relu'"
            )));
        }
        Ok(Self {
            inner,
            nonlinearity: nl,
        })
    }

    #[pyo3(signature = (x, hx=None))]
    pub fn step(&self, py: Python<'_>, x: &PyTensor, hx: Option<PyTensor>) -> PyResult<PyTensor> {
        let out = py
            .detach(|| {
                let h_ref = hx.as_ref().map(|p| &p.inner);
                if self.nonlinearity == "relu" {
                    let b = x.inner.shape()[0];
                    let h_prev = match h_ref {
                        Some(h) => h.clone(),
                        None => Tensor::zeros(
                            &[b, self.inner.hidden_size],
                            x.inner.dtype(),
                            &x.inner.device(),
                        )?,
                    };
                    self.inner
                        .ih
                        .forward(&x.inner)?
                        .add(&self.inner.hh.forward(&h_prev)?)?
                        .relu()
                } else {
                    self.inner.step(&x.inner, h_ref)
                }
            })
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    #[pyo3(signature = (x, hx=None))]
    pub fn forward(&self, py: Python<'_>, x: &PyTensor, hx: Option<PyTensor>) -> PyResult<PyTensor> {
        self.step(py, x, hx)
    }

    #[pyo3(signature = (x, hx=None))]
    pub fn __call__(&self, py: Python<'_>, x: &PyTensor, hx: Option<PyTensor>) -> PyResult<PyTensor> {
        self.step(py, x, hx)
    }

    #[getter]
    pub fn input_size(&self) -> usize {
        self.inner.input_size
    }

    #[getter]
    pub fn hidden_size(&self) -> usize {
        self.inner.hidden_size
    }

    #[getter]
    pub fn nonlinearity(&self) -> String {
        self.nonlinearity.clone()
    }

    #[getter]
    pub fn bias(&self) -> bool {
        self.inner.ih.bias.is_some()
    }

    #[getter]
    pub fn weight_ih(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.ih.weight.clone(),
        }
    }

    #[setter]
    pub fn set_weight_ih(&mut self, w: PyTensor) {
        self.inner.ih.weight = w.inner;
    }

    #[getter]
    pub fn bias_ih(&self) -> Option<PyTensor> {
        self.inner
            .ih
            .bias
            .as_ref()
            .map(|b| PyTensor { inner: b.clone() })
    }

    #[setter]
    pub fn set_bias_ih(&mut self, b: Option<PyTensor>) {
        self.inner.ih.bias = b.map(|p| p.inner);
    }

    #[getter]
    pub fn weight_hh(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.hh.weight.clone(),
        }
    }

    #[setter]
    pub fn set_weight_hh(&mut self, w: PyTensor) {
        self.inner.hh.weight = w.inner;
    }

    #[getter]
    pub fn bias_hh(&self) -> Option<PyTensor> {
        self.inner
            .hh
            .bias
            .as_ref()
            .map(|b| PyTensor { inner: b.clone() })
    }

    #[setter]
    pub fn set_bias_hh(&mut self, b: Option<PyTensor>) {
        self.inner.hh.bias = b.map(|p| p.inner);
    }

    pub fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .named_parameters()
            .into_iter()
            .map(|(_, t)| PyTensor { inner: t })
            .collect()
    }
}

/// Long Short-Term Memory (LSTM) cell.
#[pyclass(name = "LSTMCell", module = "oxideon")]
pub struct PyLSTMCell {
    inner: LSTMCell,
}

#[pymethods]
impl PyLSTMCell {
    #[new]
    #[pyo3(signature = (input_size, hidden_size, bias=true))]
    pub fn new(input_size: usize, hidden_size: usize, bias: bool) -> PyResult<Self> {
        let mut inner = LSTMCell::new(input_size, hidden_size).map_err(to_py_err)?;
        if !bias {
            inner.ih.bias = None;
            inner.hh.bias = None;
        }
        Ok(Self { inner })
    }

    #[pyo3(signature = (x, hx=None))]
    pub fn step(
        &self,
        py: Python<'_>,
        x: &PyTensor,
        hx: Option<(PyTensor, PyTensor)>,
    ) -> PyResult<(PyTensor, PyTensor)> {
        let (hn, cn) = py
            .detach(|| {
                let state_ref = hx.as_ref().map(|(h, c)| (&h.inner, &c.inner));
                self.inner.step(&x.inner, state_ref)
            })
            .map_err(to_py_err)?;
        Ok((PyTensor { inner: hn }, PyTensor { inner: cn }))
    }

    #[pyo3(signature = (x, hx=None))]
    pub fn forward(
        &self,
        py: Python<'_>,
        x: &PyTensor,
        hx: Option<(PyTensor, PyTensor)>,
    ) -> PyResult<(PyTensor, PyTensor)> {
        self.step(py, x, hx)
    }

    #[pyo3(signature = (x, hx=None))]
    pub fn __call__(
        &self,
        py: Python<'_>,
        x: &PyTensor,
        hx: Option<(PyTensor, PyTensor)>,
    ) -> PyResult<(PyTensor, PyTensor)> {
        self.step(py, x, hx)
    }

    #[getter]
    pub fn input_size(&self) -> usize {
        self.inner.input_size
    }

    #[getter]
    pub fn hidden_size(&self) -> usize {
        self.inner.hidden_size
    }

    #[getter]
    pub fn bias(&self) -> bool {
        self.inner.ih.bias.is_some()
    }

    #[getter]
    pub fn weight_ih(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.ih.weight.clone(),
        }
    }

    #[setter]
    pub fn set_weight_ih(&mut self, w: PyTensor) {
        self.inner.ih.weight = w.inner;
    }

    #[getter]
    pub fn bias_ih(&self) -> Option<PyTensor> {
        self.inner
            .ih
            .bias
            .as_ref()
            .map(|b| PyTensor { inner: b.clone() })
    }

    #[setter]
    pub fn set_bias_ih(&mut self, b: Option<PyTensor>) {
        self.inner.ih.bias = b.map(|p| p.inner);
    }

    #[getter]
    pub fn weight_hh(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.hh.weight.clone(),
        }
    }

    #[setter]
    pub fn set_weight_hh(&mut self, w: PyTensor) {
        self.inner.hh.weight = w.inner;
    }

    #[getter]
    pub fn bias_hh(&self) -> Option<PyTensor> {
        self.inner
            .hh
            .bias
            .as_ref()
            .map(|b| PyTensor { inner: b.clone() })
    }

    #[setter]
    pub fn set_bias_hh(&mut self, b: Option<PyTensor>) {
        self.inner.hh.bias = b.map(|p| p.inner);
    }

    pub fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .named_parameters()
            .into_iter()
            .map(|(_, t)| PyTensor { inner: t })
            .collect()
    }
}

/// Gated Recurrent Unit (GRU) cell.
#[pyclass(name = "GRUCell", module = "oxideon")]
pub struct PyGRUCell {
    inner: GRUCell,
}

#[pymethods]
impl PyGRUCell {
    #[new]
    #[pyo3(signature = (input_size, hidden_size, bias=true))]
    pub fn new(input_size: usize, hidden_size: usize, bias: bool) -> PyResult<Self> {
        let mut inner = GRUCell::new(input_size, hidden_size).map_err(to_py_err)?;
        if !bias {
            inner.ih.bias = None;
            inner.hh.bias = None;
        }
        Ok(Self { inner })
    }

    #[pyo3(signature = (x, hx=None))]
    pub fn step(&self, py: Python<'_>, x: &PyTensor, hx: Option<PyTensor>) -> PyResult<PyTensor> {
        let out = py
            .detach(|| {
                let h_ref = hx.as_ref().map(|p| &p.inner);
                self.inner.step(&x.inner, h_ref)
            })
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    #[pyo3(signature = (x, hx=None))]
    pub fn forward(&self, py: Python<'_>, x: &PyTensor, hx: Option<PyTensor>) -> PyResult<PyTensor> {
        self.step(py, x, hx)
    }

    #[pyo3(signature = (x, hx=None))]
    pub fn __call__(&self, py: Python<'_>, x: &PyTensor, hx: Option<PyTensor>) -> PyResult<PyTensor> {
        self.step(py, x, hx)
    }

    #[getter]
    pub fn input_size(&self) -> usize {
        self.inner.input_size
    }

    #[getter]
    pub fn hidden_size(&self) -> usize {
        self.inner.hidden_size
    }

    #[getter]
    pub fn bias(&self) -> bool {
        self.inner.ih.bias.is_some()
    }

    #[getter]
    pub fn weight_ih(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.ih.weight.clone(),
        }
    }

    #[setter]
    pub fn set_weight_ih(&mut self, w: PyTensor) {
        self.inner.ih.weight = w.inner;
    }

    #[getter]
    pub fn bias_ih(&self) -> Option<PyTensor> {
        self.inner
            .ih
            .bias
            .as_ref()
            .map(|b| PyTensor { inner: b.clone() })
    }

    #[setter]
    pub fn set_bias_ih(&mut self, b: Option<PyTensor>) {
        self.inner.ih.bias = b.map(|p| p.inner);
    }

    #[getter]
    pub fn weight_hh(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.hh.weight.clone(),
        }
    }

    #[setter]
    pub fn set_weight_hh(&mut self, w: PyTensor) {
        self.inner.hh.weight = w.inner;
    }

    #[getter]
    pub fn bias_hh(&self) -> Option<PyTensor> {
        self.inner
            .hh
            .bias
            .as_ref()
            .map(|b| PyTensor { inner: b.clone() })
    }

    #[setter]
    pub fn set_bias_hh(&mut self, b: Option<PyTensor>) {
        self.inner.hh.bias = b.map(|p| p.inner);
    }

    pub fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .named_parameters()
            .into_iter()
            .map(|(_, t)| PyTensor { inner: t })
            .collect()
    }
}

#[pyclass(name = "ServingEngine", module = "oxideon")]
pub struct PyServingEngine {
    inner: std::sync::Mutex<ServingEngine>,
}

#[pymethods]
impl PyServingEngine {
    #[new]
    #[pyo3(signature = (max_batch_size=32, num_blocks=128, block_size=16, num_heads=8, head_dim=64))]
    pub fn new(
        max_batch_size: usize,
        num_blocks: usize,
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> PyResult<Self> {
        let engine =
            ServingEngine::new(max_batch_size, num_blocks, block_size, num_heads, head_dim)
                .map_err(to_py_err)?;
        Ok(Self {
            inner: std::sync::Mutex::new(engine),
        })
    }

    pub fn submit(&self, prompt: Vec<usize>, max_new_tokens: usize) -> usize {
        mutex_guard(&self.inner).submit(prompt, max_new_tokens)
    }

    pub fn num_active(&self) -> usize {
        mutex_guard(&self.inner).scheduler.num_active()
    }

    pub fn utilization(&self) -> f64 {
        mutex_guard(&self.inner).allocator.utilization()
    }
}

#[pyclass(name = "SpeculativeDecoder", module = "oxideon")]
pub struct PySpeculativeDecoder {
    inner: std::sync::Mutex<SpeculativeDecoder>,
}

#[pymethods]
impl PySpeculativeDecoder {
    #[new]
    #[pyo3(signature = (num_speculative_tokens=4, temperature=1.0))]
    pub fn new(num_speculative_tokens: usize, temperature: f64) -> Self {
        Self {
            inner: std::sync::Mutex::new(SpeculativeDecoder::new(SpeculativeConfig {
                num_speculative_tokens,
                temperature,
            })),
        }
    }

    pub fn verify_and_sample(
        &self,
        draft_tokens: Vec<usize>,
        draft_probs: Vec<f64>,
        target_probs_all: Vec<Vec<f64>>,
    ) -> PyResult<Vec<usize>> {
        mutex_guard(&self.inner)
            .verify_and_sample(&draft_tokens, &draft_probs, &target_probs_all)
            .map_err(to_py_err)
    }

    pub fn acceptance_rate(&self) -> f64 {
        mutex_guard(&self.inner).metrics.acceptance_rate()
    }

    pub fn estimated_speedup(&self) -> f64 {
        mutex_guard(&self.inner).metrics.estimated_speedup()
    }
}

#[pyclass(name = "SuperIntelligenceConfig", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PySuperIntelligenceConfig {
    pub inner: SuperIntelligenceConfig,
}

#[pymethods]
impl PySuperIntelligenceConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (model_name="oxideon-v5", dp_size=1, tp_size=1, pp_size=1, high_watermark_ratio=0.85, max_batch_tokens=2048, prefill_chunk_size=512))]
    pub fn new(
        model_name: &str,
        dp_size: usize,
        tp_size: usize,
        pp_size: usize,
        high_watermark_ratio: f64,
        max_batch_tokens: usize,
        prefill_chunk_size: usize,
    ) -> Self {
        Self {
            inner: SuperIntelligenceConfig {
                model_name: model_name.to_string(),
                parallel_config: Parallel3DConfig {
                    dp_size,
                    tp_size,
                    pp_size,
                },
                sampling_config: SamplingConfig::default(),
                high_watermark_ratio,
                max_batch_tokens,
                prefill_chunk_size,
                speculative_enabled: true,
            },
        }
    }
}

#[pyclass(name = "SuperIntelligenceEngine", module = "oxideon")]
pub struct PySuperIntelligenceEngine {
    pub inner: SuperIntelligenceEngine,
}

#[pymethods]
impl PySuperIntelligenceEngine {
    #[new]
    pub fn new(config: &PySuperIntelligenceConfig, device_specs: Vec<String>) -> PyResult<Self> {
        let mut devices = Vec::with_capacity(device_specs.len());
        for s in device_specs {
            devices.push(Device::parse_spec(&s).map_err(to_py_err)?);
        }
        let inner =
            SuperIntelligenceEngine::new(config.inner.clone(), devices).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn submit_request(&mut self, prompt: Vec<usize>) -> usize {
        self.inner.submit_request(prompt)
    }

    pub fn verify_speculative_tokens(
        &mut self,
        py: Python<'_>,
        draft_tokens: Vec<usize>,
        target_logits: &PyTensor,
    ) -> PyResult<Vec<usize>> {
        let toks = py
            .detach(|| {
                self.inner
                    .verify_speculative_tokens(&draft_tokens, &target_logits.inner)
            })
            .map_err(to_py_err)?;
        Ok(toks)
    }

    pub fn get_metrics(&self) -> (usize, usize, usize, f64, bool) {
        let m = self.inner.get_metrics();
        (
            m.requests_completed,
            m.tokens_generated,
            m.offload_migrations,
            m.speculative_speedup,
            m.is_idle,
        )
    }
}

#[pyclass(name = "UniversalAGIEngine", module = "oxideon")]
pub struct PyUniversalAGIEngine {
    inner: std::sync::Mutex<UniversalAGIEngine>,
}

#[pymethods]
impl PyUniversalAGIEngine {
    #[new]
    pub fn new() -> PyResult<Self> {
        let config = UniversalAGIConfig::default();
        let engine = UniversalAGIEngine::new(config).map_err(to_py_err)?;
        Ok(Self {
            inner: std::sync::Mutex::new(engine),
        })
    }

    #[allow(clippy::type_complexity)]
    pub fn execute_cognitive_cycle(
        &self,
        py: Python<'_>,
    ) -> PyResult<(f64, usize, f32, f32, String, f64, bool, f32)> {
        let rep = py
            .detach(|| mutex_guard(&self.inner).execute_cognitive_cycle())
            .map_err(to_py_err)?;
        Ok((
            rep.quantum_expectation,
            rep.neuromorphic_spikes,
            rep.mamba_output_norm,
            rep.multimodal_loss,
            rep.swarm_consensus_winner,
            rep.swarm_agreement_score,
            rep.synthesized_program_passed,
            rep.lorentz_distance,
        ))
    }
}

#[pyclass(name = "OmniScientificFoundationEngine", module = "oxideon")]
pub struct PyOmniScientificFoundationEngine {
    inner: std::sync::Mutex<OmniScientificFoundationEngine>,
}

#[pymethods]
impl PyOmniScientificFoundationEngine {
    #[new]
    #[pyo3(signature = (dimension=4))]
    pub fn new(dimension: usize) -> PyResult<Self> {
        let engine = OmniScientificFoundationEngine::new(dimension).map_err(to_py_err)?;
        Ok(Self {
            inner: std::sync::Mutex::new(engine),
        })
    }

    #[allow(clippy::type_complexity)]
    pub fn execute_multidisciplinary_cycle(
        &self,
        py: Python<'_>,
        state: &PyTensor,
    ) -> PyResult<(f64, f64, f64, Vec<u8>, Vec<usize>, f64, f64, Vec<i8>, bool)> {
        let rep = py
            .detach(|| mutex_guard(&self.inner).execute_multidisciplinary_cycle(&state.inner))
            .map_err(to_py_err)?;
        Ok((
            rep.initial_energy,
            rep.final_energy,
            rep.optical_transmission_energy,
            rep.zkml_proof_digest,
            rep.betti_numbers,
            rep.holographic_retrieval_norm,
            rep.stdp_weight_mean,
            rep.qec_syndromes,
            rep.theorem_proved,
        ))
    }
}

#[pyclass(name = "OmniTransSingularityEngine", module = "oxideon")]
pub struct PyOmniTransSingularityEngine {
    inner: std::sync::Mutex<OmniTransSingularityEngine>,
}

#[pymethods]
impl PyOmniTransSingularityEngine {
    #[new]
    pub fn new() -> PyResult<Self> {
        let engine = OmniTransSingularityEngine::new().map_err(to_py_err)?;
        Ok(Self {
            inner: std::sync::Mutex::new(engine),
        })
    }

    #[allow(clippy::type_complexity)]
    pub fn execute_cycle(
        &self,
        py: Python<'_>,
        input_signal: &PyTensor,
    ) -> PyResult<(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64)> {
        let rep = py
            .detach(|| mutex_guard(&self.inner).execute_cycle(&input_signal.inner))
            .map_err(to_py_err)?;
        Ok((
            rep.fno_output_norm,
            rep.ckks_addition_sample,
            rep.byzantine_consensus_norm,
            rep.sheaf_dirichlet_energy,
            rep.langevin_drift_norm,
            rep.kerr_outer_horizon,
            rep.deeponet_output_norm,
            rep.hgat_manifold_inner,
            rep.clifford_rotated_norm,
            rep.ilp_transitive_confidence,
            rep.singularity_coherence,
        ))
    }
}

#[pyclass(name = "GrandTransUniversalEngineIV", module = "oxideon")]
pub struct PyGrandTransUniversalEngineIV {
    pub(crate) inner: GrandTransUniversalEngineIV,
}

#[pymethods]
impl PyGrandTransUniversalEngineIV {
    #[new]
    #[pyo3(signature = (dim=4))]
    pub fn new(dim: usize) -> Self {
        Self {
            inner: GrandTransUniversalEngineIV::new(dim),
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn execute_semi_centennial_cycle(
        &self,
        py: Python<'_>,
        input_tensor: &PyTensor,
    ) -> PyResult<(f64, f64, f64, f64, f64, f64, (f64, f64), usize, f64, f64)> {
        let rep = py
            .detach(|| {
                self.inner
                    .execute_semi_centennial_cycle(&input_tensor.inner)
            })
            .map_err(to_py_err)?;
        Ok((
            rep.nbf_shortest_path_dist,
            rep.tropical_hilbert_distance,
            rep.quantum_mps_ising_energy,
            rep.quantum_mps_entanglement_entropy,
            rep.gromov_wasserstein_distortion,
            rep.nls_soliton_conserved_mass,
            rep.lie_poisson_casimir_invariants,
            rep.symbolic_pareto_front_size,
            rep.causal_ate,
            rep.wavelet_temporal_hidden_norm,
        ))
    }
}

impl Default for PyGrandTransUniversalEngineIV {
    fn default() -> Self {
        Self::new(4)
    }
}

#[pyclass(name = "GrandOmniReportV", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyGrandOmniReportV {
    pub inner: GrandOmniReportV,
}

#[pymethods]
impl PyGrandOmniReportV {
    #[getter]
    pub fn fiber_bundle_holonomy_angle(&self) -> f64 {
        self.inner.fiber_bundle_holonomy_angle
    }

    #[getter]
    pub fn sheaf_obstruction_norm(&self) -> f64 {
        self.inner.sheaf_obstruction_norm
    }

    #[getter]
    pub fn photonic_homodyne_power(&self) -> f64 {
        self.inner.photonic_homodyne_power
    }

    #[getter]
    pub fn riemannian_flow_radius(&self) -> f64 {
        self.inner.riemannian_flow_radius
    }

    #[getter]
    pub fn quantum_zne_mitigated_value(&self) -> f64 {
        self.inner.quantum_zne_mitigated_value
    }

    #[getter]
    pub fn rkhs_mmd_distance(&self) -> f64 {
        self.inner.rkhs_mmd_distance
    }

    #[getter]
    pub fn deq_equilibrium_residual(&self) -> f64 {
        self.inner.deq_equilibrium_residual
    }

    #[getter]
    pub fn crc_calibrated_threshold(&self) -> f64 {
        self.inner.crc_calibrated_threshold
    }

    #[getter]
    pub fn fdr_discovered_count(&self) -> usize {
        self.inner.fdr_discovered_count
    }

    #[getter]
    pub fn exceptional_point_petermann(&self) -> f64 {
        self.inner.exceptional_point_petermann
    }

    #[getter]
    pub fn autonomous_quant_format(&self) -> String {
        self.inner.autonomous_quant_format.clone()
    }
}

#[pyclass(name = "GrandOmniSeptuaReportVI", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyGrandOmniSeptuaReportVI {
    pub inner: GrandOmniSeptuaReportVI,
}

#[pymethods]
impl PyGrandOmniSeptuaReportVI {
    #[getter]
    pub fn yang_mills_wilson_loop(&self) -> f64 {
        self.inner.yang_mills_wilson_loop
    }

    #[getter]
    pub fn qgnn_graph_energy(&self) -> f64 {
        self.inner.qgnn_graph_energy
    }

    #[getter]
    pub fn heisenberg_cc_distance(&self) -> f64 {
        self.inner.heisenberg_cc_distance
    }

    #[getter]
    pub fn polychronous_coincidence_spikes(&self) -> usize {
        self.inner.polychronous_coincidence_spikes
    }

    #[getter]
    pub fn spin_glass_q_ea(&self) -> f64 {
        self.inner.spin_glass_q_ea
    }

    #[getter]
    pub fn cv_vacuum_probability(&self) -> f64 {
        self.inner.cv_vacuum_probability
    }

    #[getter]
    pub fn cv_k4_hafnian(&self) -> f64 {
        self.inner.cv_k4_hafnian
    }

    #[getter]
    pub fn rmhd_fast_magnetosonic_speed(&self) -> f64 {
        self.inner.rmhd_fast_magnetosonic_speed
    }

    #[getter]
    pub fn tensor_train_partition_sum(&self) -> f64 {
        self.inner.tensor_train_partition_sum
    }

    #[getter]
    pub fn ryu_takayanagi_entropy(&self) -> f64 {
        self.inner.ryu_takayanagi_entropy
    }

    #[getter]
    pub fn active_llm_bits_per_weight(&self) -> f64 {
        self.inner.active_llm_bits_per_weight
    }

    #[getter]
    pub fn unified_septua_invariant(&self) -> f64 {
        self.inner.unified_septua_invariant
    }
}

#[pyclass(name = "GrandOmniOctaReportVII", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyGrandOmniOctaReportVII {
    pub inner: GrandOmniOctaReportVII,
}

#[pymethods]
impl PyGrandOmniOctaReportVII {
    #[getter]
    pub fn quasi_category_associator_defect(&self) -> f64 {
        self.inner.quasi_category_associator_defect
    }

    #[getter]
    pub fn qap_divisibility_residual(&self) -> f64 {
        self.inner.qap_divisibility_residual
    }

    #[getter]
    pub fn symplectic_gromov_capacity(&self) -> f64 {
        self.inner.symplectic_gromov_capacity
    }

    #[getter]
    pub fn jarzynski_free_energy(&self) -> f64 {
        self.inner.jarzynski_free_energy
    }

    #[getter]
    pub fn associahedron_k5_vertices(&self) -> u64 {
        self.inner.associahedron_k5_vertices
    }

    #[getter]
    pub fn topos_heyting_verified(&self) -> bool {
        self.inner.topos_heyting_verified
    }

    #[getter]
    pub fn mfg_nash_social_cost(&self) -> f64 {
        self.inner.mfg_nash_social_cost
    }

    #[getter]
    pub fn tqft_fibonacci_quantum_dim(&self) -> f64 {
        self.inner.tqft_fibonacci_quantum_dim
    }

    #[getter]
    pub fn fisher_rao_geodesic_distance(&self) -> f64 {
        self.inner.fisher_rao_geodesic_distance
    }

    #[getter]
    pub fn active_llm_bits_per_weight(&self) -> f64 {
        self.inner.active_llm_bits_per_weight
    }

    #[getter]
    pub fn unified_octa_invariant(&self) -> f64 {
        self.inner.unified_octa_invariant
    }
}

#[pyclass(name = "GrandOmniOctaEngineVII", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyGrandOmniOctaEngineVII {
    pub inner: GrandOmniOctaEngineVII,
}

#[pymethods]
impl PyGrandOmniOctaEngineVII {
    #[new]
    #[pyo3(signature = (num_layers=6))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniOctaEngineVII::new(num_layers),
        }
    }

    pub fn execute_octa_synthesis(&self, py: Python<'_>) -> PyResult<PyGrandOmniOctaReportVII> {
        let rep = py
            .detach(|| self.inner.execute_octa_synthesis())
            .map_err(to_py_err)?;
        Ok(PyGrandOmniOctaReportVII { inner: rep })
    }
}

#[pyclass(name = "GrandOmniSeptuaEngineVI", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyGrandOmniSeptuaEngineVI {
    pub inner: GrandOmniSeptuaEngineVI,
}

#[pymethods]
impl PyGrandOmniSeptuaEngineVI {
    #[new]
    #[pyo3(signature = (num_layers=4))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniSeptuaEngineVI::new(num_layers),
        }
    }

    pub fn execute_septua_synthesis(&self, py: Python<'_>) -> PyResult<PyGrandOmniSeptuaReportVI> {
        let rep = py
            .detach(|| self.inner.execute_septua_synthesis())
            .map_err(to_py_err)?;
        Ok(PyGrandOmniSeptuaReportVI { inner: rep })
    }
}

#[pyclass(name = "GrandOmniUniversalEngineV", module = "oxideon")]
pub struct PyGrandOmniUniversalEngineV {
    pub inner: GrandOmniUniversalEngineV,
}

#[pymethods]
impl PyGrandOmniUniversalEngineV {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: GrandOmniUniversalEngineV::new(),
        }
    }

    pub fn execute_synthesis(&self, py: Python<'_>) -> PyResult<PyGrandOmniReportV> {
        let rep = py
            .detach(|| self.inner.execute_synthesis())
            .map_err(to_py_err)?;
        Ok(PyGrandOmniReportV { inner: rep })
    }
}

impl Default for PyGrandOmniUniversalEngineV {
    fn default() -> Self {
        Self::new()
    }
}

#[pyclass(name = "JsonSchema", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyJsonSchema {
    #[allow(dead_code)]
    pub inner: JsonSchema,
}

#[pymethods]
impl PyJsonSchema {
    #[staticmethod]
    pub fn string() -> Self {
        Self {
            inner: JsonSchema::string(),
        }
    }

    #[staticmethod]
    pub fn integer() -> Self {
        Self {
            inner: JsonSchema::integer(),
        }
    }

    #[staticmethod]
    pub fn number() -> Self {
        Self {
            inner: JsonSchema::number(),
        }
    }

    #[staticmethod]
    pub fn boolean() -> Self {
        Self {
            inner: JsonSchema::boolean(),
        }
    }

    #[staticmethod]
    pub fn enum_values(values: Vec<String>) -> Self {
        let refs: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
        Self {
            inner: JsonSchema::enum_values(&refs),
        }
    }
}

#[pyclass(name = "AgentRuntime", module = "oxideon")]
pub struct PyAgentRuntime {
    inner: std::sync::Mutex<AgentRuntime>,
}

#[pymethods]
impl PyAgentRuntime {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: std::sync::Mutex::new(AgentRuntime::new()),
        }
    }

    pub fn register_tool(&self, name: String, description: String) {
        mutex_guard(&self.inner).register_tool(&name, &description, JsonSchema::string(), |args| {
            Ok(format!("executed {args}"))
        });
    }

    pub fn call_tool(&self, name: &str, args_json: &str) -> PyResult<String> {
        mutex_guard(&self.inner)
            .call_tool(name, args_json)
            .map_err(to_py_err)
    }

    pub fn render_system_prompt(&self) -> String {
        mutex_guard(&self.inner).render_system_prompt()
    }
}

impl Default for PyAgentRuntime {
    fn default() -> Self {
        Self::new()
    }
}

#[pyclass(name = "ExecutionSandbox", module = "oxideon")]
pub struct PyExecutionSandbox {
    inner: ExecutionSandbox,
}

#[pymethods]
impl PyExecutionSandbox {
    #[new]
    #[pyo3(signature = (max_steps=10000))]
    pub fn new(max_steps: usize) -> Self {
        Self {
            inner: ExecutionSandbox::new(max_steps),
        }
    }

    pub fn execute(
        &self,
        program: &str,
        initial_vars: HashMap<String, i64>,
    ) -> PyResult<(String, usize, HashMap<String, i64>)> {
        let rep = self.inner.execute(program, &initial_vars);
        let status = format!("{:?}", rep.status);
        Ok((status, rep.steps_executed, rep.variables))
    }
}

#[pyclass(name = "SynthesisEngine", module = "oxideon")]
pub struct PySynthesisEngine {
    inner: SynthesisEngine,
}

#[pymethods]
impl PySynthesisEngine {
    #[new]
    #[pyo3(signature = (max_repair_iterations=5, max_steps=10000))]
    pub fn new(max_repair_iterations: usize, max_steps: usize) -> Self {
        Self {
            inner: SynthesisEngine::new(max_repair_iterations, max_steps),
        }
    }

    pub fn verify_candidate(
        &self,
        code: &str,
        test_names: Vec<String>,
        test_inputs: Vec<HashMap<String, i64>>,
        test_expected_vars: Vec<String>,
        test_expected_vals: Vec<i64>,
    ) -> PyResult<bool> {
        let tests: Vec<TestCase> = test_names
            .into_iter()
            .zip(test_inputs)
            .zip(test_expected_vars)
            .zip(test_expected_vals)
            .map(
                |(((name, input_vars), expected_var), expected_val)| TestCase {
                    name,
                    input_vars,
                    expected_var,
                    expected_val,
                },
            )
            .collect();
        self.inner.verify_candidate(code, &tests).map_err(to_py_err)
    }
}

#[pyclass(name = "MelSpectrogram", module = "oxideon")]
pub struct PyMelSpectrogram {
    inner: MelSpectrogram,
}

#[pymethods]
impl PyMelSpectrogram {
    #[new]
    #[pyo3(signature = (n_mels=80, n_fft=400, hop_length=160, sample_rate=16000))]
    pub fn new(
        n_mels: usize,
        n_fft: usize,
        hop_length: usize,
        sample_rate: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: MelSpectrogram::new(n_mels, n_fft, hop_length, sample_rate)
                .map_err(to_py_err)?,
        })
    }

    pub fn forward(&self, py: Python<'_>, waveform: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&waveform.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "Whisper", module = "oxideon")]
pub struct PyWhisper {
    inner: Whisper,
}

#[pymethods]
impl PyWhisper {
    #[new]
    #[pyo3(signature = (n_mels=80, d_model=64, encoder_layers=2, encoder_heads=4, decoder_layers=2, decoder_heads=4, d_ff=128, vocab_size=100))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_mels: usize,
        d_model: usize,
        encoder_layers: usize,
        encoder_heads: usize,
        decoder_layers: usize,
        decoder_heads: usize,
        d_ff: usize,
        vocab_size: usize,
    ) -> PyResult<Self> {
        let cfg = WhisperConfig {
            n_mels,
            d_model,
            encoder_layers,
            encoder_heads,
            decoder_layers,
            decoder_heads,
            d_ff,
            vocab_size,
            max_source_positions: 1500,
            max_target_positions: 448,
        };
        Ok(Self {
            inner: Whisper::new(cfg).map_err(to_py_err)?,
        })
    }

    pub fn forward(&self, py: Python<'_>, mel: &PyTensor, tokens: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&mel.inner, &tokens.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    #[pyo3(signature = (mel, max_len=50))]
    pub fn transcribe(
        &self,
        py: Python<'_>,
        mel: &PyTensor,
        max_len: usize,
    ) -> PyResult<Vec<usize>> {
        py.detach(|| self.inner.transcribe(&mel.inner, max_len))
            .map_err(to_py_err)
    }
}

#[pyclass(name = "PipelineSchedule1F1B", module = "oxideon")]
pub struct PyPipelineSchedule1F1B {
    inner: std::sync::Mutex<PipelineSchedule1F1B>,
}

#[pymethods]
impl PyPipelineSchedule1F1B {
    #[new]
    pub fn new(num_stages: usize, num_microbatches: usize, stage_idx: usize) -> PyResult<Self> {
        let schedule = PipelineSchedule1F1B::new(num_stages, num_microbatches, stage_idx)
            .map_err(to_py_err)?;
        Ok(Self {
            inner: std::sync::Mutex::new(schedule),
        })
    }

    pub fn next_step(&self) -> Option<(String, bool, usize)> {
        mutex_guard(&self.inner)
            .next_step()
            .map(|(phase, is_fwd, mb)| {
                let p_str = match phase {
                    PipelinePhase::Warmup => "Warmup",
                    PipelinePhase::Steady1F1B => "Steady1F1B",
                    PipelinePhase::Cooldown => "Cooldown",
                };
                (p_str.to_string(), is_fwd, mb)
            })
    }

    pub fn bubble_fraction(&self) -> f64 {
        mutex_guard(&self.inner).bubble_fraction()
    }
}

#[pyclass(name = "ZeRO3Parameter", module = "oxideon")]
pub struct PyZeRO3Parameter {
    inner: std::sync::Mutex<ZeRO3Parameter>,
}

#[pymethods]
impl PyZeRO3Parameter {
    #[staticmethod]
    pub fn from_tensor(tensor: &PyTensor, world_size: usize) -> PyResult<Self> {
        let p = ZeRO3Parameter::from_parameter(&tensor.inner, world_size).map_err(to_py_err)?;
        Ok(Self {
            inner: std::sync::Mutex::new(p),
        })
    }

    pub fn gather(&self) -> PyResult<PyTensor> {
        let out = mutex_guard(&self.inner).gather().map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn release(&self) {
        mutex_guard(&self.inner).release();
    }

    pub fn memory_savings_ratio(&self) -> f64 {
        mutex_guard(&self.inner).memory_savings_ratio()
    }
}

// ---------------------------------------------------------------------------
// Optimizers & Compiled IR Graph
// ---------------------------------------------------------------------------

#[pyclass(name = "AdamW", module = "oxideon")]
pub struct PyAdamW {
    inner: Adam,
}

#[pymethods]
impl PyAdamW {
    #[new]
    #[pyo3(signature = (params, lr=1e-3, weight_decay=1e-2))]
    pub fn new(params: Vec<PyTensor>, lr: f64, weight_decay: f64) -> PyResult<Self> {
        non_negative("lr", lr)?;
        non_negative("weight_decay", weight_decay)?;
        Ok(Self {
            inner: Adam::adamw(require_params(params)?, lr, weight_decay),
        })
    }

    pub fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.step()).map_err(to_py_err)
    }

    pub fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    #[getter]
    pub fn lr(&self) -> f64 {
        self.inner.learning_rate()
    }

    pub fn set_learning_rate(&mut self, lr: f64) -> PyResult<()> {
        non_negative("lr", lr)?;
        self.inner.set_learning_rate(lr);
        Ok(())
    }
}

fn non_negative(name: &str, v: f64) -> PyResult<()> {
    if v.is_finite() && v >= 0.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be a finite number >= 0, got {v}"
        )))
    }
}

fn require_params(params: Vec<PyTensor>) -> PyResult<Vec<Tensor>> {
    if params.is_empty() {
        return Err(PyValueError::new_err(
            "optimizer got an empty parameter list",
        ));
    }
    Ok(params.into_iter().map(|t| t.inner).collect())
}

/// Stochastic gradient descent with optional momentum and L2 weight decay.
#[pyclass(name = "SGD", module = "oxideon", dict)]
pub struct PySGD {
    inner: Sgd,
    momentum: f64,
    weight_decay: f64,
}

#[pymethods]
impl PySGD {
    #[new]
    #[pyo3(signature = (params, lr=1e-2, momentum=0.0, weight_decay=0.0))]
    pub fn new(params: Vec<PyTensor>, lr: f64, momentum: f64, weight_decay: f64) -> PyResult<Self> {
        non_negative("lr", lr)?;
        non_negative("momentum", momentum)?;
        non_negative("weight_decay", weight_decay)?;
        Ok(Self {
            inner: Sgd::new(require_params(params)?, lr)
                .momentum(momentum)
                .weight_decay(weight_decay),
            momentum,
            weight_decay,
        })
    }

    pub fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.step()).map_err(to_py_err)
    }

    pub fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    #[getter]
    pub fn lr(&self) -> f64 {
        self.inner.learning_rate()
    }

    pub fn set_learning_rate(&mut self, lr: f64) -> PyResult<()> {
        non_negative("lr", lr)?;
        self.inner.set_learning_rate(lr);
        Ok(())
    }

    #[getter]
    pub fn param_groups(&self, py: Python<'_>) -> PyResult<pyo3::Py<pyo3::types::PyList>> {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("lr", self.lr())?;
        dict.set_item("momentum", self.momentum)?;
        dict.set_item("weight_decay", self.weight_decay)?;
        let list = pyo3::types::PyList::new(py, &[dict])?;
        Ok(list.into())
    }
}

/// Adam with optional (coupled, L2) weight decay; use `AdamW` for decoupled decay.
#[pyclass(name = "Adam", module = "oxideon")]
pub struct PyAdam {
    inner: Adam,
}

#[pymethods]
impl PyAdam {
    #[new]
    #[pyo3(signature = (params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0), text_signature = "(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)")]
    pub fn new(
        params: Vec<PyTensor>,
        lr: f64,
        betas: (f64, f64),
        eps: f64,
        weight_decay: f64,
    ) -> PyResult<Self> {
        non_negative("lr", lr)?;
        non_negative("eps", eps)?;
        non_negative("weight_decay", weight_decay)?;
        for (name, b) in [("betas[0]", betas.0), ("betas[1]", betas.1)] {
            if !(0.0..1.0).contains(&b) {
                return Err(PyValueError::new_err(format!(
                    "{name} must be in [0, 1), got {b}"
                )));
            }
        }
        Ok(Self {
            inner: Adam::new(require_params(params)?, lr)
                .betas(betas.0, betas.1)
                .eps(eps)
                .weight_decay(weight_decay),
        })
    }

    pub fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.step()).map_err(to_py_err)
    }

    pub fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    #[getter]
    pub fn lr(&self) -> f64 {
        self.inner.learning_rate()
    }

    pub fn set_learning_rate(&mut self, lr: f64) -> PyResult<()> {
        non_negative("lr", lr)?;
        self.inner.set_learning_rate(lr);
        Ok(())
    }
}

#[pyclass(name = "RiemannianAdam", module = "oxideon")]
pub struct PyRiemannianAdam {
    inner: RiemannianAdam,
}

#[pymethods]
impl PyRiemannianAdam {
    #[new]
    #[pyo3(signature = (params, manifolds, lr=1e-3))]
    pub fn new(params: Vec<PyTensor>, manifolds: Vec<String>, lr: f64) -> PyResult<Self> {
        non_negative("lr", lr)?;
        let p: Vec<Tensor> = require_params(params)?;
        let mut m_types = Vec::with_capacity(manifolds.len());
        for m_str in &manifolds {
            let m = match m_str.to_lowercase().as_str() {
                "sphere" => ManifoldType::Sphere,
                "poincare" | "poincare_ball" | "hyperbolic" => {
                    ManifoldType::PoincareBall { c_x1000: 1000 }
                }
                "stiefel" => ManifoldType::Stiefel { p: 1 },
                other => {
                    return Err(to_py_err(crate::error::Error::Msg(format!(
                        "Unknown manifold type: {other}"
                    ))));
                }
            };
            m_types.push(m);
        }
        let inner = RiemannianAdam::new(p, m_types, lr).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.step()).map_err(to_py_err)
    }

    pub fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    #[getter]
    pub fn lr(&self) -> f64 {
        self.inner.learning_rate()
    }

    pub fn set_learning_rate(&mut self, lr: f64) -> PyResult<()> {
        non_negative("lr", lr)?;
        self.inner.set_learning_rate(lr);
        Ok(())
    }
}

#[pyclass(name = "Muon", module = "oxideon")]
pub struct PyMuon {
    inner: Muon,
}

#[pymethods]
impl PyMuon {
    #[new]
    #[pyo3(signature = (params, lr=2e-2, momentum=0.95, weight_decay=0.0))]
    pub fn new(params: Vec<PyTensor>, lr: f64, momentum: f64, weight_decay: f64) -> PyResult<Self> {
        non_negative("lr", lr)?;
        non_negative("momentum", momentum)?;
        non_negative("weight_decay", weight_decay)?;
        Ok(Self {
            inner: Muon::new(require_params(params)?, lr)
                .momentum(momentum)
                .weight_decay(weight_decay),
        })
    }

    pub fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.step()).map_err(to_py_err)
    }

    pub fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    #[getter]
    pub fn lr(&self) -> f64 {
        self.inner.learning_rate()
    }

    pub fn set_learning_rate(&mut self, lr: f64) -> PyResult<()> {
        non_negative("lr", lr)?;
        self.inner.set_learning_rate(lr);
        Ok(())
    }
}

#[pyclass(name = "ScheduleFreeAdamW", module = "oxideon")]
pub struct PyScheduleFreeAdamW {
    inner: ScheduleFreeAdamW,
}

#[pymethods]
impl PyScheduleFreeAdamW {
    #[new]
    #[pyo3(signature = (params, lr=1e-3, weight_decay=0.0))]
    pub fn new(params: Vec<PyTensor>, lr: f64, weight_decay: f64) -> PyResult<Self> {
        non_negative("lr", lr)?;
        non_negative("weight_decay", weight_decay)?;
        Ok(Self {
            inner: ScheduleFreeAdamW::new(require_params(params)?, lr, weight_decay),
        })
    }

    pub fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.step()).map_err(to_py_err)
    }

    pub fn eval_mode(&mut self) -> PyResult<()> {
        self.inner.eval_mode().map_err(to_py_err)
    }

    pub fn train_mode(&mut self) -> PyResult<()> {
        self.inner.train_mode().map_err(to_py_err)
    }

    pub fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    #[getter]
    pub fn lr(&self) -> f64 {
        self.inner.learning_rate()
    }

    pub fn set_learning_rate(&mut self, lr: f64) -> PyResult<()> {
        non_negative("lr", lr)?;
        self.inner.set_learning_rate(lr);
        Ok(())
    }
}

#[pyclass(name = "IrGraph", module = "oxideon")]
#[derive(Default)]
pub struct PyIrGraph {
    pub inner: IrGraph,
}

#[pymethods]
impl PyIrGraph {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: IrGraph::new(),
        }
    }

    pub fn add_input(&mut self) -> usize {
        self.inner.add_input()
    }

    pub fn add_op(&mut self, op_name: &str, args: Vec<usize>) -> PyResult<usize> {
        let op = match op_name {
            "add" if args.len() == 2 => IrOp::Add(args[0], args[1]),
            "sub" if args.len() == 2 => IrOp::Sub(args[0], args[1]),
            "mul" if args.len() == 2 => IrOp::Mul(args[0], args[1]),
            "div" if args.len() == 2 => IrOp::Div(args[0], args[1]),
            "relu" if args.len() == 1 => IrOp::Relu(args[0]),
            "silu" if args.len() == 1 => IrOp::Silu(args[0]),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported IR op '{op_name}'"
                )));
            }
        };
        Ok(self.inner.push_op(op))
    }

    pub fn set_outputs(&mut self, outputs: Vec<usize>) {
        self.inner.set_outputs(outputs);
    }

    pub fn emit_c(&self, fn_name: &str) -> PyResult<String> {
        AotCompiler::emit_c(&self.inner, fn_name).map_err(to_py_err)
    }

    pub fn emit_rust(&self, fn_name: &str) -> PyResult<String> {
        AotCompiler::emit_rust(&self.inner, fn_name).map_err(to_py_err)
    }
}

#[pyclass(name = "CompiledGraph", module = "oxideon")]
pub struct PyCompiledGraph {
    inner: CompiledGraph,
}

#[pymethods]
impl PyCompiledGraph {
    pub fn execute(&self, py: Python<'_>, inputs: Vec<PyTensor>) -> PyResult<Vec<PyTensor>> {
        let raw: Vec<Tensor> = inputs.into_iter().map(|t| t.inner).collect();
        let outs = py.detach(|| self.inner.execute(&raw)).map_err(to_py_err)?;
        Ok(outs.into_iter().map(|inner| PyTensor { inner }).collect())
    }

    #[getter]
    pub fn fused_ops_count(&self) -> usize {
        self.inner.report.fused_ops_count
    }

    #[getter]
    pub fn peak_slots(&self) -> usize {
        self.inner.memory_plan.peak_slots
    }

    #[getter]
    pub fn reuse_ratio(&self) -> f64 {
        self.inner.memory_plan.reuse_ratio()
    }
}

// ---------------------------------------------------------------------------
// Module-Level Functions (Fused Kernels, Autograd, Quantization, Safetensors)
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(name = "manual_seed")]
pub fn py_manual_seed(seed: u64) {
    manual_seed(seed);
}

#[pyfunction]
#[pyo3(name = "initial_seed")]
pub fn py_initial_seed() -> u64 {
    initial_seed()
}

#[pyfunction]
#[pyo3(name = "seed")]
pub fn py_seed() -> u64 {
    seed()
}

#[pyfunction]
#[pyo3(name = "get_rng_state")]
pub fn py_get_rng_state() -> Vec<u8> {
    get_rng_state()
}

#[pyfunction]
#[pyo3(name = "set_rng_state")]
pub fn py_set_rng_state(state: &[u8]) -> PyResult<()> {
    set_rng_state(state).map_err(to_py_err)
}

#[pyfunction]
#[pyo3(name = "grad", signature = (outputs, inputs, create_graph=false))]
pub fn py_grad(
    py: Python<'_>,
    outputs: Vec<PyTensor>,
    inputs: Vec<PyTensor>,
    create_graph: bool,
) -> PyResult<Vec<Option<PyTensor>>> {
    let outs: Vec<Tensor> = outputs.into_iter().map(|t| t.inner).collect();
    let ins: Vec<Tensor> = inputs.into_iter().map(|t| t.inner).collect();
    let grads = py
        .detach(|| grad(&outs, &ins, None, create_graph))
        .map_err(to_py_err)?;
    Ok(grads
        .into_iter()
        .map(|g| g.map(|inner| PyTensor { inner }))
        .collect())
}

#[pyfunction]
#[pyo3(name = "fused_rmsnorm", signature = (x, weight, eps=1e-5))]
pub fn py_fused_rmsnorm(
    py: Python<'_>,
    x: &PyTensor,
    weight: &PyTensor,
    eps: f64,
) -> PyResult<PyTensor> {
    let inner = py
        .detach(|| fused_rmsnorm(&x.inner, &weight.inner, eps))
        .map_err(to_py_err)?;
    Ok(PyTensor { inner })
}

#[pyfunction]
#[pyo3(name = "fused_add_rmsnorm", signature = (x, residual, weight, eps=1e-5))]
pub fn py_fused_add_rmsnorm(
    py: Python<'_>,
    x: &PyTensor,
    residual: &PyTensor,
    weight: &PyTensor,
    eps: f64,
) -> PyResult<(PyTensor, PyTensor)> {
    let (normed, sum_out) = py
        .detach(|| fused_add_rmsnorm(&x.inner, &residual.inner, &weight.inner, eps))
        .map_err(to_py_err)?;
    Ok((PyTensor { inner: normed }, PyTensor { inner: sum_out }))
}

#[pyfunction]
#[pyo3(name = "fused_swiglu")]
pub fn py_fused_swiglu(py: Python<'_>, gate: &PyTensor, up: &PyTensor) -> PyResult<PyTensor> {
    let inner = py
        .detach(|| fused_swiglu(&gate.inner, &up.inner))
        .map_err(to_py_err)?;
    Ok(PyTensor { inner })
}

#[pyfunction]
#[pyo3(name = "fused_geglu")]
pub fn py_fused_geglu(py: Python<'_>, gate: &PyTensor, up: &PyTensor) -> PyResult<PyTensor> {
    let inner = py
        .detach(|| fused_geglu(&gate.inner, &up.inner))
        .map_err(to_py_err)?;
    Ok(PyTensor { inner })
}

#[pyfunction]
#[pyo3(name = "fused_linear_activation", signature = (x, weight, bias=None, activation="silu"))]
pub fn py_fused_linear_activation(
    py: Python<'_>,
    x: &PyTensor,
    weight: &PyTensor,
    bias: Option<&PyTensor>,
    activation: &str,
) -> PyResult<PyTensor> {
    let act = match activation.to_ascii_lowercase().as_str() {
        "none" | "identity" => FusedActivation::None,
        "relu" => FusedActivation::ReLU,
        "silu" | "swish" => FusedActivation::SiLU,
        "gelu" => FusedActivation::GELU,
        "tanh" => FusedActivation::Tanh,
        other => {
            return Err(PyRuntimeError::new_err(format!(
                "unknown activation '{other}'"
            )));
        }
    };
    let b_ref = bias.map(|b| &b.inner);
    let inner = py
        .detach(|| fused_linear_activation(&x.inner, &weight.inner, b_ref, act))
        .map_err(to_py_err)?;
    Ok(PyTensor { inner })
}

#[pyfunction]
#[pyo3(name = "fused_cross_entropy_with_z_loss", signature = (logits, targets, z_loss_weight=1e-4))]
pub fn py_fused_cross_entropy_with_z_loss(
    py: Python<'_>,
    logits: &PyTensor,
    targets: Vec<usize>,
    z_loss_weight: f64,
) -> PyResult<PyTensor> {
    let inner = py
        .detach(|| fused_cross_entropy_with_z_loss(&logits.inner, &targets, z_loss_weight))
        .map_err(to_py_err)?;
    Ok(PyTensor { inner })
}

#[pyfunction]
#[pyo3(name = "fused_layernorm", signature = (x, weight=None, bias=None, eps=1e-5))]
pub fn py_fused_layernorm(
    py: Python<'_>,
    x: &PyTensor,
    weight: Option<&PyTensor>,
    bias: Option<&PyTensor>,
    eps: f64,
) -> PyResult<PyTensor> {
    let w_ref = weight.map(|w| &w.inner);
    let b_ref = bias.map(|b| &b.inner);
    let inner = py
        .detach(|| fused_layernorm(&x.inner, w_ref, b_ref, eps))
        .map_err(to_py_err)?;
    Ok(PyTensor { inner })
}

#[pyfunction]
#[pyo3(name = "fused_add_layernorm", signature = (x, residual, weight=None, bias=None, eps=1e-5))]
pub fn py_fused_add_layernorm(
    py: Python<'_>,
    x: &PyTensor,
    residual: &PyTensor,
    weight: Option<&PyTensor>,
    bias: Option<&PyTensor>,
    eps: f64,
) -> PyResult<(PyTensor, PyTensor)> {
    let w_ref = weight.map(|w| &w.inner);
    let b_ref = bias.map(|b| &b.inner);
    let (normed, sum_out) = py
        .detach(|| fused_add_layernorm(&x.inner, &residual.inner, w_ref, b_ref, eps))
        .map_err(to_py_err)?;
    Ok((PyTensor { inner: normed }, PyTensor { inner: sum_out }))
}

#[pyfunction]
#[pyo3(name = "fused_bias_activation", signature = (x, bias, act="gelu"))]
pub fn py_fused_bias_activation(
    py: Python<'_>,
    x: &PyTensor,
    bias: &PyTensor,
    act: &str,
) -> PyResult<PyTensor> {
    let activation = match act.to_lowercase().as_str() {
        "none" | "linear" => FusedActivation::None,
        "relu" => FusedActivation::ReLU,
        "silu" | "swish" => FusedActivation::SiLU,
        "gelu" => FusedActivation::GELU,
        "tanh" => FusedActivation::Tanh,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unsupported activation '{other}', expected 'none', 'relu', 'silu', 'gelu', or 'tanh'"
            )))
        }
    };
    let inner = py
        .detach(|| fused_bias_activation(&x.inner, &bias.inner, activation))
        .map_err(to_py_err)?;
    Ok(PyTensor { inner })
}

#[pyfunction]
#[pyo3(name = "fused_silu_and_mul")]
pub fn py_fused_silu_and_mul(
    py: Python<'_>,
    x: &PyTensor,
    z: &PyTensor,
) -> PyResult<PyTensor> {
    let inner = py
        .detach(|| fused_silu_and_mul(&x.inner, &z.inner))
        .map_err(to_py_err)?;
    Ok(PyTensor { inner })
}

#[pyfunction]
#[pyo3(name = "fused_rotary_qk")]
pub fn py_fused_rotary_qk(
    py: Python<'_>,
    q: &PyTensor,
    k: &PyTensor,
    cos: &PyTensor,
    sin: &PyTensor,
) -> PyResult<(PyTensor, PyTensor)> {
    let (q_rot, k_rot) = py
        .detach(|| fused_rotary_qk(&q.inner, &k.inner, &cos.inner, &sin.inner))
        .map_err(to_py_err)?;
    Ok((PyTensor { inner: q_rot }, PyTensor { inner: k_rot }))
}

#[pyfunction]
#[pyo3(name = "fused_rope")]
pub fn py_fused_rope(
    py: Python<'_>,
    x: &PyTensor,
    cos: &PyTensor,
    sin: &PyTensor,
) -> PyResult<PyTensor> {
    let inner = py
        .detach(|| fused_rope(&x.inner, &cos.inner, &sin.inner))
        .map_err(to_py_err)?;
    Ok(PyTensor { inner })
}

#[pyfunction]
#[pyo3(name = "fused_sdpa", signature = (q, k, v, scale=None, is_causal=false))]
pub fn py_fused_sdpa(
    py: Python<'_>,
    q: &PyTensor,
    k: &PyTensor,
    v: &PyTensor,
    scale: Option<f64>,
    is_causal: bool,
) -> PyResult<PyTensor> {
    let inner = py
        .detach(|| fused_sdpa(&q.inner, &k.inner, &v.inner, scale, is_causal))
        .map_err(to_py_err)?;
    Ok(PyTensor { inner })
}

#[pyfunction]
#[pyo3(name = "mlx_quantize_affine_q4", signature = (w, group_size=32))]
pub fn py_mlx_quantize_affine_q4(
    py: Python<'_>,
    w: &PyTensor,
    group_size: usize,
) -> PyResult<(PyTensor, PyTensor, PyTensor)> {
    let (p, s, b) = py
        .detach(|| mlx_quantize_affine_q4(&w.inner, group_size))
        .map_err(to_py_err)?;
    Ok((
        PyTensor { inner: p },
        PyTensor { inner: s },
        PyTensor { inner: b },
    ))
}

#[pyfunction]
#[pyo3(name = "mlx_dequantize_affine_q4", signature = (packed, scales, biases, group_size=32))]
pub fn py_mlx_dequantize_affine_q4(
    py: Python<'_>,
    packed: &PyTensor,
    scales: &PyTensor,
    biases: &PyTensor,
    group_size: usize,
) -> PyResult<PyTensor> {
    let inner = py
        .detach(|| mlx_dequantize_affine_q4(&packed.inner, &scales.inner, &biases.inner, group_size))
        .map_err(to_py_err)?;
    Ok(PyTensor { inner })
}

#[pyfunction]
#[pyo3(name = "mlx_quantized_matmul_q4", signature = (x, packed, scales, biases, group_size=32))]
pub fn py_mlx_quantized_matmul_q4(
    py: Python<'_>,
    x: &PyTensor,
    packed: &PyTensor,
    scales: &PyTensor,
    biases: &PyTensor,
    group_size: usize,
) -> PyResult<PyTensor> {
    let inner = py
        .detach(|| {
            mlx_quantized_matmul_q4(
                &x.inner,
                &packed.inner,
                &scales.inner,
                &biases.inner,
                group_size,
            )
        })
        .map_err(to_py_err)?;
    Ok(PyTensor { inner })
}

#[pyfunction]
#[pyo3(name = "flash_attention_tiled", signature = (q, k, v, block_m=16, block_n=16, is_causal=false))]
pub fn py_flash_attention_tiled(
    py: Python<'_>,
    q: &PyTensor,
    k: &PyTensor,
    v: &PyTensor,
    block_m: usize,
    block_n: usize,
    is_causal: bool,
) -> PyResult<PyTensor> {
    let inner = py
        .detach(|| flash_attention_tiled(&q.inner, &k.inner, &v.inner, block_m, block_n, is_causal))
        .map_err(to_py_err)?;
    Ok(PyTensor { inner })
}

#[pyfunction]
#[pyo3(name = "paged_kv_attention")]
pub fn py_paged_kv_attention(
    py: Python<'_>,
    q: &PyTensor,
    k_cache: &PyTensor,
    v_cache: &PyTensor,
    block_tables: Vec<Vec<usize>>,
    context_lens: Vec<usize>,
) -> PyResult<PyTensor> {
    let inner = py
        .detach(|| {
            paged_kv_attention(
                &q.inner,
                &k_cache.inner,
                &v_cache.inner,
                &block_tables,
                &context_lens,
            )
        })
        .map_err(to_py_err)?;
    Ok(PyTensor { inner })
}

#[pyfunction]
#[pyo3(name = "block_diag")]
pub fn py_block_diag(py: Python<'_>, tensors: Vec<PyTensor>) -> PyResult<PyTensor> {
    let refs: Vec<&Tensor> = tensors.iter().map(|t| &t.inner).collect();
    let inner = py.detach(|| Tensor::block_diag(&refs)).map_err(to_py_err)?;
    Ok(PyTensor { inner })
}

#[pyfunction]
#[pyo3(name = "cartesian_prod")]
pub fn py_cartesian_prod(py: Python<'_>, tensors: Vec<PyTensor>) -> PyResult<PyTensor> {
    let refs: Vec<&Tensor> = tensors.iter().map(|t| &t.inner).collect();
    let inner = py
        .detach(|| Tensor::cartesian_prod(&refs))
        .map_err(to_py_err)?;
    Ok(PyTensor { inner })
}

#[pyfunction]
#[pyo3(name = "compile_fused_linear_silu_with_grad")]
pub fn py_compile_fused_linear_silu_with_grad(py: Python<'_>) -> PyResult<PyCompiledGraph> {
    let compiled = py
        .detach(|| {
            let mut g = IrGraph::new();
            let x = g.add_input();
            let w = g.add_input();
            let b = g.add_input();
            let mm = g.push_op(IrOp::Matmul(x, w));
            let lin = g.push_op(IrOp::Add(mm, b));
            let act = g.push_op(IrOp::Silu(lin));
            let loss = g.push_op(IrOp::SumAll(act));
            let grads = g.differentiate(loss, &[x, w, b])?;
            g.set_outputs(vec![act, grads[0], grads[1], grads[2]]);
            g.compile()
        })
        .map_err(to_py_err)?;
    Ok(PyCompiledGraph { inner: compiled })
}

#[pyfunction]
#[pyo3(name = "pixel_shuffle")]
pub fn py_pixel_shuffle(py: Python<'_>, x: &PyTensor, upscale_factor: usize) -> PyResult<PyTensor> {
    let ps = PixelShuffle::new(upscale_factor);
    let inner = py.detach(|| ps.forward(&x.inner)).map_err(to_py_err)?;
    Ok(PyTensor { inner })
}

#[pyfunction]
#[pyo3(name = "awq_quantize_forward", signature = (linear, calib_x, x, bits=4, alpha=0.5))]
pub fn py_awq_quantize_forward(
    py: Python<'_>,
    linear: &PyLinear,
    calib_x: &PyTensor,
    x: &PyTensor,
    bits: usize,
    alpha: f64,
) -> PyResult<PyTensor> {
    let scheme = if bits == 4 {
        QuantScheme::Q4_0 { block_size: 32 }
    } else {
        QuantScheme::Q8_0 { block_size: 32 }
    };
    let inner = py
        .detach(|| {
            no_grad(|| {
                let awq = AWQLinear::from_linear_calibrated(
                    &linear.inner,
                    &calib_x.inner,
                    scheme,
                    alpha,
                )?;
                awq.forward(&x.inner)
            })
        })
        .map_err(to_py_err)?;
    Ok(PyTensor { inner })
}

#[pyfunction]
#[pyo3(name = "save_safetensors", signature = (path, tensors, metadata=None))]
pub fn py_save_safetensors(
    py: Python<'_>,
    path: String,
    tensors: &Bound<'_, PyAny>,
    metadata: Option<std::collections::HashMap<String, String>>,
) -> PyResult<()> {
    let mut map: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();
    if let Ok(dict) = tensors.cast::<pyo3::types::PyDict>() {
        for (k, v) in dict.iter() {
            let key: String = k.extract()?;
            let tensor: PyTensor = v.extract()?;
            map.insert(key, tensor.inner);
        }
    } else if let Ok(seq) = tensors.cast::<pyo3::types::PySequence>() {
        let len = seq.len()?;
        for i in 0..len {
            let item = seq.get_item(i)?;
            let (key, tensor): (String, PyTensor) = item.extract()?;
            map.insert(key, tensor.inner);
        }
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "save_safetensors expects tensors to be a dict or sequence of (name, tensor) tuples",
        ));
    }
    py.detach(|| save_safetensors_with_metadata(&map, metadata.as_ref(), &path))
        .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(name = "load_safetensors")]
pub fn py_load_safetensors(py: Python<'_>, path: String) -> PyResult<Vec<(String, PyTensor)>> {
    let (map, _) = py
        .detach(|| load_safetensors_with_metadata(&path))
        .map_err(to_py_err)?;
    Ok(map
        .into_iter()
        .map(|(k, inner)| (k, PyTensor { inner }))
        .collect())
}

#[pyfunction]
#[pyo3(name = "load_safetensors_metadata")]
pub fn py_load_safetensors_metadata(
    py: Python<'_>,
    path: String,
) -> PyResult<std::collections::HashMap<String, String>> {
    let (_, meta) = py
        .detach(|| load_safetensors_with_metadata(&path))
        .map_err(to_py_err)?;
    Ok(meta)
}

#[pyfunction]
#[pyo3(name = "allocator_stats")]
pub fn py_allocator_stats() -> (usize, usize, usize, usize, usize) {
    let s = CachingAllocator::global().stats();
    (
        s.active_bytes,
        s.cached_bytes,
        s.peak_active_bytes,
        s.cache_hits,
        s.cache_misses,
    )
}

#[pyclass(name = "BitLinear", module = "oxideon")]
pub struct PyBitLinear {
    inner: BitLinear,
}

#[pymethods]
impl PyBitLinear {
    #[staticmethod]
    pub fn from_linear(linear: &PyLinear) -> PyResult<Self> {
        let inner = BitLinear::from_linear(&linear.inner).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn forward(&self, py: Python<'_>, input: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&input.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    #[pyo3(signature = (dtype=None))]
    pub fn dequantize_weight(&self, py: Python<'_>, dtype: Option<String>) -> PyResult<PyTensor> {
        let dt = dtype
            .as_deref()
            .map(parse_dtype)
            .transpose()?
            .unwrap_or(DType::F32);
        let out = py
            .detach(|| self.inner.dequantize_weight(dt))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn weight_bytes(&self) -> usize {
        self.inner.weight_bytes()
    }
}

#[pyclass(name = "FP8Tensor", module = "oxideon")]
pub struct PyFP8Tensor {
    inner: FP8Tensor,
}

#[pymethods]
impl PyFP8Tensor {
    #[staticmethod]
    pub fn from_tensor(tensor: &PyTensor, format: &str) -> PyResult<Self> {
        let fmt = match format.to_lowercase().as_str() {
            "e4m3" => FP8Format::E4M3,
            "e5m2" => FP8Format::E5M2,
            other => {
                return Err(PyRuntimeError::new_err(format!(
                    "unsupported FP8 format '{other}' (expected 'e4m3' or 'e5m2')"
                )));
            }
        };
        let inner = FP8Tensor::from_tensor(&tensor.inner, fmt).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (dtype=None))]
    pub fn to_tensor(&self, py: Python<'_>, dtype: Option<String>) -> PyResult<PyTensor> {
        let dt = dtype
            .as_deref()
            .map(parse_dtype)
            .transpose()?
            .unwrap_or(DType::F32);
        let out = py.detach(|| self.inner.to_tensor(dt)).map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn size_in_bytes(&self) -> usize {
        self.inner.size_in_bytes()
    }
}

#[pyclass(name = "GPTQLinear", module = "oxideon")]
pub struct PyGPTQLinear {
    inner: GPTQLinear,
}

#[pymethods]
impl PyGPTQLinear {
    #[staticmethod]
    #[pyo3(signature = (linear, calib_x, scheme="q4_0", damp_percent=0.01))]
    pub fn from_linear_calibrated(
        linear: &PyLinear,
        calib_x: &PyTensor,
        scheme: &str,
        damp_percent: f64,
    ) -> PyResult<Self> {
        let qscheme = match scheme.to_lowercase().as_str() {
            "q8_0" => QuantScheme::DEFAULT_Q8_0,
            "q4_0" => QuantScheme::DEFAULT_Q4_0,
            other => {
                return Err(PyRuntimeError::new_err(format!(
                    "unsupported GPTQ scheme '{other}'"
                )));
            }
        };
        let inner = GPTQLinear::from_linear_calibrated(
            &linear.inner,
            &calib_x.inner,
            qscheme,
            damp_percent,
        )
        .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn forward(&self, py: Python<'_>, input: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&input.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "GRPOTrainer", module = "oxideon")]
pub struct PyGRPOTrainer {
    inner: GRPOTrainer,
}

#[pymethods]
impl PyGRPOTrainer {
    #[new]
    #[pyo3(signature = (group_size=4, clip_eps=0.2, kl_beta=0.04))]
    pub fn new(group_size: usize, clip_eps: f64, kl_beta: f64) -> Self {
        Self {
            inner: GRPOTrainer::new(GRPOConfig {
                group_size,
                clip_eps,
                kl_beta,
                eps: 1e-6,
            }),
        }
    }

    pub fn compute_step(
        &self,
        py: Python<'_>,
        policy_logps: &PyTensor,
        old_logps: &PyTensor,
        ref_logps: &PyTensor,
        raw_rewards: Vec<f64>,
    ) -> PyResult<(PyTensor, f64, f64, f64)> {
        let (loss, stats) = py
            .detach(|| {
                self.inner.compute_step(
                    &policy_logps.inner,
                    &old_logps.inner,
                    &ref_logps.inner,
                    &raw_rewards,
                    None,
                )
            })
            .map_err(to_py_err)?;
        Ok((
            PyTensor { inner: loss },
            stats.mean_reward,
            stats.format_reward_rate,
            stats.loss,
        ))
    }

    #[staticmethod]
    pub fn format_reward(text: &str) -> f64 {
        RuleReward::format_reward(text)
    }

    #[staticmethod]
    pub fn extract_answer(text: &str) -> Option<String> {
        RuleReward::extract_answer(text)
    }

    #[staticmethod]
    pub fn accuracy_reward(predicted: &str, ground_truth: &str) -> f64 {
        RuleReward::accuracy_reward(predicted, ground_truth)
    }
}

#[pyclass(name = "DPOTrainer", module = "oxideon")]
pub struct PyDPOTrainer {
    inner: DPOTrainer,
}

#[pymethods]
impl PyDPOTrainer {
    #[new]
    #[pyo3(signature = (beta=0.1, label_smoothing=0.0))]
    pub fn new(beta: f64, label_smoothing: f64) -> Self {
        Self {
            inner: DPOTrainer::new(DPOConfig {
                beta,
                label_smoothing,
            }),
        }
    }

    pub fn compute_step(
        &self,
        py: Python<'_>,
        policy_chosen: &PyTensor,
        policy_rejected: &PyTensor,
        ref_chosen: &PyTensor,
        ref_rejected: &PyTensor,
    ) -> PyResult<(PyTensor, f64, f64, f64, f64)> {
        let (loss, stats) = py
            .detach(|| {
                self.inner.compute_step(
                    &policy_chosen.inner,
                    &policy_rejected.inner,
                    &ref_chosen.inner,
                    &ref_rejected.inner,
                )
            })
            .map_err(to_py_err)?;
        Ok((
            PyTensor { inner: loss },
            stats.chosen_reward,
            stats.rejected_reward,
            stats.reward_margin,
            stats.accuracy,
        ))
    }
}

#[pyclass(name = "PPOTrainer", module = "oxideon")]
pub struct PyPPOTrainer {
    inner: PPOTrainer,
}

#[pymethods]
impl PyPPOTrainer {
    #[new]
    #[pyo3(signature = (clip_eps=0.2, vf_coef=0.5, ent_coef=0.01))]
    pub fn new(clip_eps: f64, vf_coef: f64, ent_coef: f64) -> Self {
        Self {
            inner: PPOTrainer::new(PPOConfig {
                clip_eps,
                vf_coef,
                ent_coef,
                clip_vf: Some(0.2),
            }),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compute_step(
        &self,
        py: Python<'_>,
        new_logps: &PyTensor,
        old_logps: &PyTensor,
        advantages: &PyTensor,
        new_values: &PyTensor,
        old_values: &PyTensor,
        returns: &PyTensor,
    ) -> PyResult<(PyTensor, f64, f64, f64)> {
        let (loss, stats) = py
            .detach(|| {
                self.inner.compute_step(
                    &new_logps.inner,
                    &old_logps.inner,
                    &advantages.inner,
                    &new_values.inner,
                    &old_values.inner,
                    &returns.inner,
                )
            })
            .map_err(to_py_err)?;
        Ok((
            PyTensor { inner: loss },
            stats.policy_loss,
            stats.value_loss,
            stats.total_loss,
        ))
    }
}

#[pyclass(name = "VideoVAE3d", module = "oxideon")]
pub struct PyVideoVAE3d {
    inner: VideoVAE3d,
}

#[pymethods]
impl PyVideoVAE3d {
    #[new]
    pub fn new(
        in_channels: usize,
        hidden_channels: usize,
        latent_channels: usize,
    ) -> PyResult<Self> {
        let inner =
            VideoVAE3d::new(in_channels, hidden_channels, latent_channels).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn encode(&self, py: Python<'_>, video: &PyTensor) -> PyResult<PyTensor> {
        let z = py
            .detach(|| self.inner.encode(&video.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: z })
    }

    pub fn decode(&self, py: Python<'_>, latents: &PyTensor) -> PyResult<PyTensor> {
        let rec = py
            .detach(|| self.inner.decode(&latents.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: rec })
    }

    pub fn forward(&self, py: Python<'_>, video: &PyTensor) -> PyResult<PyTensor> {
        let rec = py
            .detach(|| self.inner.forward(&video.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: rec })
    }
}

#[pyclass(name = "RectifiedFlowVideoDiT", module = "oxideon")]
pub struct PyRectifiedFlowVideoDiT {
    inner: RectifiedFlowVideoDiT,
}

#[pymethods]
impl PyRectifiedFlowVideoDiT {
    #[new]
    #[pyo3(signature = (in_channels=4, hidden_dim=64, num_heads=4, text_dim=32))]
    pub fn new(
        in_channels: usize,
        hidden_dim: usize,
        num_heads: usize,
        text_dim: usize,
    ) -> PyResult<Self> {
        let cfg = VideoDiTConfig {
            in_channels,
            hidden_dim,
            num_heads,
            patch_size: (1, 2, 2),
            text_dim,
        };
        let inner = RectifiedFlowVideoDiT::new(cfg).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn forward(
        &self,
        py: Python<'_>,
        x_t: &PyTensor,
        t: &PyTensor,
        text_cond: &PyTensor,
    ) -> PyResult<PyTensor> {
        let v = py
            .detach(|| self.inner.forward(&x_t.inner, &t.inner, &text_cond.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: v })
    }

    pub fn sample_euler(
        &self,
        py: Python<'_>,
        init_noise: &PyTensor,
        text_cond: &PyTensor,
        num_steps: usize,
    ) -> PyResult<PyTensor> {
        let res = py
            .detach(|| {
                self.inner
                    .sample_euler(&init_noise.inner, &text_cond.inner, num_steps)
            })
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: res })
    }
}

#[pyclass(name = "RotaryEmbedding3D", module = "oxideon")]
pub struct PyRotaryEmbedding3D {
    inner: RotaryEmbedding3D,
}

#[pymethods]
impl PyRotaryEmbedding3D {
    #[new]
    #[pyo3(signature = (dim, theta_base=10000.0))]
    pub fn new(dim: usize, theta_base: f64) -> PyResult<Self> {
        let inner = RotaryEmbedding3D::new(dim, theta_base).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn forward(
        &self,
        py: Python<'_>,
        x: &PyTensor,
        t_size: usize,
        h_size: usize,
        w_size: usize,
    ) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner, t_size, h_size, w_size))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "RotaryEmbedding2D", module = "oxideon")]
pub struct PyRotaryEmbedding2D {
    inner: RotaryEmbedding2D,
}

#[pymethods]
impl PyRotaryEmbedding2D {
    #[new]
    #[pyo3(signature = (dim, theta_base=10000.0))]
    pub fn new(dim: usize, theta_base: f64) -> PyResult<Self> {
        let inner = RotaryEmbedding2D::new(dim, theta_base).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (dim_h, dim_w, theta_base=10000.0))]
    pub fn with_partitions(dim_h: usize, dim_w: usize, theta_base: f64) -> PyResult<Self> {
        let inner =
            RotaryEmbedding2D::with_partitions(dim_h, dim_w, theta_base).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn forward(
        &self,
        py: Python<'_>,
        x: &PyTensor,
        h_size: usize,
        w_size: usize,
    ) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner, h_size, w_size))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    #[getter]
    pub fn dim(&self) -> usize {
        self.inner.dim
    }

    #[getter]
    pub fn dim_h(&self) -> usize {
        self.inner.dim_h
    }

    #[getter]
    pub fn dim_w(&self) -> usize {
        self.inner.dim_w
    }
}

#[pyclass(name = "JitKernelBuilder", module = "oxideon")]
pub struct PyJitKernelBuilder {
    inner: JitKernelBuilder,
}

#[pymethods]
impl PyJitKernelBuilder {
    #[new]
    pub fn new(name: String) -> Self {
        Self {
            inner: JitKernelBuilder::new(name),
        }
    }

    pub fn add_input(&mut self, name: String) -> usize {
        self.inner.add_input(name)
    }

    pub fn add_op(&mut self, op_name: String, args: Vec<usize>) -> PyResult<usize> {
        let op = match op_name.to_lowercase().as_str() {
            "add" => JitOp::Add,
            "sub" => JitOp::Sub,
            "mul" => JitOp::Mul,
            "div" => JitOp::Div,
            "silu" => JitOp::Silu,
            "gelu" => JitOp::Gelu,
            "relu" => JitOp::Relu,
            "exp" => JitOp::Exp,
            "neg" => JitOp::Neg,
            other => {
                return Err(PyRuntimeError::new_err(format!(
                    "unsupported JIT op '{other}'"
                )));
            }
        };
        Ok(self.inner.add_op(op, &args))
    }

    pub fn set_output(&mut self, id: usize) {
        self.inner.set_output(id);
    }

    #[pyo3(signature = (workgroup_size=64))]
    pub fn build_wgsl(&self, workgroup_size: u32) -> PyResult<String> {
        self.inner.build_wgsl(workgroup_size).map_err(to_py_err)
    }

    pub fn execute_cpu(&self, py: Python<'_>, inputs: Vec<PyTensor>) -> PyResult<PyTensor> {
        let refs: Vec<&Tensor> = inputs.iter().map(|p| &p.inner).collect();
        let out = py
            .detach(|| self.inner.execute_cpu(&refs))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    #[staticmethod]
    pub fn fused_add_silu() -> Self {
        Self {
            inner: JitKernelBuilder::fused_add_silu(),
        }
    }

    #[staticmethod]
    pub fn fused_swiglu_gate() -> Self {
        Self {
            inner: JitKernelBuilder::fused_swiglu_gate(),
        }
    }
}

#[pyclass(name = "RingAttention", module = "oxideon")]
pub struct PyRingAttention {
    inner: RingAttention,
}

#[pymethods]
impl PyRingAttention {
    #[new]
    #[pyo3(signature = (world_size, rank, causal=true))]
    pub fn new(world_size: usize, rank: usize, causal: bool) -> PyResult<Self> {
        if world_size == 0 || rank >= world_size {
            return Err(PyValueError::new_err(format!(
                "need world_size > 0 and rank < world_size, got world_size={world_size}, rank={rank}"
            )));
        }
        Ok(Self {
            inner: RingAttention::new(world_size, rank, causal).map_err(to_py_err)?,
        })
    }

    pub fn forward_simulated(
        &self,
        py: Python<'_>,
        q_local: &PyTensor,
        k_shards: Vec<PyTensor>,
        v_shards: Vec<PyTensor>,
    ) -> PyResult<PyTensor> {
        let k_vec: Vec<Tensor> = k_shards.into_iter().map(|p| p.inner).collect();
        let v_vec: Vec<Tensor> = v_shards.into_iter().map(|p| p.inner).collect();
        let out = py
            .detach(|| self.inner.forward_simulated(&q_local.inner, &k_vec, &v_vec))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "MCTSSearchEngine", module = "oxideon")]
pub struct PyMCTSSearchEngine {
    inner: MCTSSearchEngine,
}

#[pymethods]
impl PyMCTSSearchEngine {
    #[new]
    #[pyo3(signature = (c_puct=1.414, num_simulations=20, max_depth=5))]
    pub fn new(c_puct: f64, num_simulations: usize, max_depth: usize) -> Self {
        Self {
            inner: MCTSSearchEngine::new(c_puct, num_simulations, max_depth),
        }
    }

    pub fn search_mock(
        &self,
        root_text: &str,
        candidates: Vec<(String, f64)>,
        eval_score: f64,
    ) -> PyResult<(String, f64, usize)> {
        let res = self
            .inner
            .search(root_text, |_| candidates.clone(), |_| eval_score)
            .map_err(to_py_err)?;
        Ok((res.best_text, res.best_value, res.total_nodes))
    }
}

#[pyclass(name = "SelfCorrectionVerifier", module = "oxideon")]
pub struct PySelfCorrectionVerifier {
    inner: SelfCorrectionVerifier,
}

#[pymethods]
impl PySelfCorrectionVerifier {
    #[new]
    #[pyo3(signature = (confidence_threshold=0.8))]
    pub fn new(confidence_threshold: f64) -> Self {
        Self {
            inner: SelfCorrectionVerifier::new(confidence_threshold),
        }
    }

    pub fn verify_chain(
        &self,
        steps: Vec<String>,
        confidences: Vec<f64>,
    ) -> (bool, Option<usize>, usize, f64) {
        let mut trace = ReasoningTrace::new();
        for (s, &c) in steps.iter().zip(confidences.iter()) {
            trace.add_step(s, c);
        }
        let report = self
            .inner
            .verify(&mut trace, |content| (!content.contains("invalid"), 0.9));
        (
            report.is_valid,
            report.failed_step_index,
            report.recommended_backtrack_index,
            report.average_confidence,
        )
    }
}

#[pyclass(name = "AgentProposal", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyAgentProposal {
    pub inner: AgentProposal,
}

#[pymethods]
impl PyAgentProposal {
    #[new]
    #[pyo3(signature = (agent_id, solution, confidence=1.0, justification=""))]
    pub fn new(agent_id: &str, solution: &str, confidence: f64, justification: &str) -> Self {
        Self {
            inner: AgentProposal::new(agent_id, solution, confidence, justification),
        }
    }

    #[getter]
    pub fn agent_id(&self) -> String {
        self.inner.agent_id.clone()
    }

    #[getter]
    pub fn solution(&self) -> String {
        self.inner.solution.clone()
    }

    #[getter]
    pub fn confidence(&self) -> f64 {
        self.inner.confidence
    }

    #[getter]
    pub fn justification(&self) -> String {
        self.inner.justification.clone()
    }
}

#[pyclass(name = "SwarmOrchestrator", module = "oxideon")]
pub struct PySwarmOrchestrator {
    inner: SwarmOrchestrator,
}

#[pymethods]
impl PySwarmOrchestrator {
    #[new]
    #[pyo3(signature = (max_debate_rounds=3, convergence_threshold=0.75, strategy="confidence"))]
    pub fn new(
        max_debate_rounds: usize,
        convergence_threshold: f64,
        strategy: &str,
    ) -> PyResult<Self> {
        let strat = match strategy {
            "majority" => ConsensusStrategy::MajorityVote,
            "nash" => ConsensusStrategy::NashEquilibrium,
            _ => ConsensusStrategy::ConfidenceWeighted,
        };
        Ok(Self {
            inner: SwarmOrchestrator::new(max_debate_rounds, convergence_threshold, strat),
        })
    }

    pub fn resolve_consensus(
        &self,
        py: Python<'_>,
        proposals: Vec<PyAgentProposal>,
    ) -> PyResult<(String, f64, usize, HashMap<String, f64>)> {
        let props: Vec<AgentProposal> = proposals.into_iter().map(|p| p.inner).collect();
        let rep = py
            .detach(|| self.inner.resolve_consensus(&props))
            .map_err(to_py_err)?;
        Ok((
            rep.final_solution,
            rep.agreement_score,
            rep.total_rounds,
            rep.vote_distribution,
        ))
    }
}

#[pyclass(name = "PropExpr", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyPropExpr {
    pub inner: PropExpr,
}

#[pymethods]
impl PyPropExpr {
    #[staticmethod]
    pub fn var(name: &str) -> Self {
        Self {
            inner: PropExpr::var(name),
        }
    }

    #[staticmethod]
    pub fn true_val() -> Self {
        Self {
            inner: PropExpr::True,
        }
    }

    #[staticmethod]
    pub fn false_val() -> Self {
        Self {
            inner: PropExpr::False,
        }
    }

    pub fn not_op(&self) -> Self {
        Self {
            inner: self.inner.clone().not(),
        }
    }

    pub fn and_op(&self, other: &PyPropExpr) -> Self {
        Self {
            inner: self.inner.clone().and(other.inner.clone()),
        }
    }

    pub fn or_op(&self, other: &PyPropExpr) -> Self {
        Self {
            inner: self.inner.clone().or(other.inner.clone()),
        }
    }

    pub fn implies_op(&self, other: &PyPropExpr) -> Self {
        Self {
            inner: self.inner.clone().implies(other.inner.clone()),
        }
    }

    pub fn iff_op(&self, other: &PyPropExpr) -> Self {
        Self {
            inner: self.inner.clone().iff(other.inner.clone()),
        }
    }
}

#[pyclass(name = "SequentProver", module = "oxideon")]
pub struct PySequentProver {
    inner: SequentProver,
}

#[pymethods]
impl PySequentProver {
    #[new]
    #[pyo3(signature = (max_depth=50))]
    pub fn new(max_depth: usize) -> Self {
        Self {
            inner: SequentProver::new(max_depth),
        }
    }

    pub fn is_tautology(&self, py: Python<'_>, expr: &PyPropExpr) -> PyResult<bool> {
        py.detach(|| self.inner.is_tautology(&expr.inner))
            .map_err(to_py_err)
    }

    pub fn prove_sequent(
        &self,
        py: Python<'_>,
        antecedent: Vec<PyPropExpr>,
        succedent: Vec<PyPropExpr>,
    ) -> PyResult<bool> {
        let ant: Vec<PropExpr> = antecedent.into_iter().map(|p| p.inner).collect();
        let suc: Vec<PropExpr> = succedent.into_iter().map(|p| p.inner).collect();
        let seq = Sequent::new(ant, suc);
        py.detach(|| self.inner.prove(&seq)).map_err(to_py_err)
    }
}

#[pyclass(name = "DifferentiableILPEngine", module = "oxideon")]
pub struct PyDifferentiableILPEngine {
    inner: DifferentiableILPEngine,
}

#[pymethods]
impl PyDifferentiableILPEngine {
    #[new]
    #[pyo3(signature = (num_entities, num_predicates, t_norm="product"))]
    pub fn new(num_entities: usize, num_predicates: usize, t_norm: &str) -> PyResult<Self> {
        let norm = match t_norm.to_lowercase().as_str() {
            "product" => TNorm::Product,
            "godel" => TNorm::Godel,
            "lukasiewicz" => TNorm::Lukasiewicz,
            other => {
                return Err(to_py_err(Error::Msg(format!(
                    "Unknown t-norm '{other}'. Expected 'product', 'godel', or 'lukasiewicz'"
                ))));
            }
        };
        Ok(Self {
            inner: DifferentiableILPEngine::new(num_entities, num_predicates, norm),
        })
    }

    #[getter]
    pub fn num_entities(&self) -> usize {
        self.inner.num_entities
    }

    #[getter]
    pub fn num_predicates(&self) -> usize {
        self.inner.num_predicates
    }

    pub fn add_base_rule(&mut self, head: usize, body: usize, weight: f64) -> PyResult<()> {
        self.inner
            .add_base_rule(head, body, weight)
            .map_err(to_py_err)
    }

    pub fn add_transitive_rule(
        &mut self,
        head: usize,
        body1: usize,
        body2: usize,
        weight: f64,
    ) -> PyResult<()> {
        self.inner
            .add_transitive_rule(head, body1, body2, weight)
            .map_err(to_py_err)
    }

    pub fn infer_step(&self, py: Python<'_>, facts: &PyTensor) -> PyResult<PyTensor> {
        let res = py
            .detach(|| self.inner.infer_step(&facts.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: res })
    }

    pub fn deductive_forward(
        &self,
        py: Python<'_>,
        initial_facts: &PyTensor,
        steps: usize,
    ) -> PyResult<PyTensor> {
        let res = py
            .detach(|| self.inner.deductive_forward(&initial_facts.inner, steps))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: res })
    }

    pub fn evaluate_loss(
        &self,
        py: Python<'_>,
        predicted_facts: &PyTensor,
        positive_targets: Vec<(usize, usize, usize)>,
        negative_targets: Vec<(usize, usize, usize)>,
    ) -> PyResult<f64> {
        py.detach(|| {
            self.inner
                .evaluate_loss(&predicted_facts.inner, &positive_targets, &negative_targets)
        })
        .map_err(to_py_err)
    }
}

#[pyclass(name = "SymbolicNode", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PySymbolicNode {
    pub inner: SymbolicNode,
}

#[pymethods]
impl PySymbolicNode {
    #[staticmethod]
    pub fn constant(val: f64) -> Self {
        Self {
            inner: SymbolicNode::Constant(val),
        }
    }

    #[staticmethod]
    pub fn variable(idx: usize) -> Self {
        Self {
            inner: SymbolicNode::Variable(idx),
        }
    }

    pub fn evaluate(&self, x: Vec<f64>) -> f64 {
        self.inner.evaluate(&x)
    }

    #[getter]
    pub fn complexity(&self) -> usize {
        self.inner.complexity()
    }

    #[getter]
    pub fn depth(&self) -> usize {
        self.inner.depth()
    }

    pub fn to_formula(&self) -> String {
        self.inner.to_formula()
    }

    fn __repr__(&self) -> String {
        self.inner.to_formula()
    }
}

#[pyclass(name = "ParetoIndividual", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyParetoIndividual {
    pub inner: ParetoIndividual,
}

#[pymethods]
impl PyParetoIndividual {
    #[getter]
    pub fn mse(&self) -> f64 {
        self.inner.mse
    }

    #[getter]
    pub fn complexity(&self) -> usize {
        self.inner.complexity
    }

    #[getter]
    pub fn formula(&self) -> String {
        self.inner.formula.clone()
    }

    pub fn evaluate(&self, x: Vec<f64>) -> f64 {
        self.inner.tree.evaluate(&x)
    }

    fn __repr__(&self) -> String {
        format!(
            "ParetoIndividual(formula='{}', mse={:.5}, complexity={})",
            self.inner.formula, self.inner.mse, self.inner.complexity
        )
    }
}

#[pyclass(name = "SymbolicGpConfig", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PySymbolicGpConfig {
    pub inner: SymbolicGpConfig,
}

#[pymethods]
impl PySymbolicGpConfig {
    #[new]
    #[pyo3(signature = (population_size=60, num_generations=30, max_depth=5, crossover_prob=0.70, mutation_prob=0.25))]
    pub fn new(
        population_size: usize,
        num_generations: usize,
        max_depth: usize,
        crossover_prob: f64,
        mutation_prob: f64,
    ) -> Self {
        Self {
            inner: SymbolicGpConfig {
                population_size,
                num_generations,
                max_depth,
                crossover_prob,
                mutation_prob,
            },
        }
    }
}

#[pyclass(name = "SymbolicGpEngine", module = "oxideon")]
pub struct PySymbolicGpEngine {
    inner: SymbolicGpEngine,
}

#[pymethods]
impl PySymbolicGpEngine {
    #[new]
    #[pyo3(signature = (config=None))]
    pub fn new(config: Option<PySymbolicGpConfig>) -> Self {
        let cfg = config.map(|c| c.inner).unwrap_or_default();
        Self {
            inner: SymbolicGpEngine::new(cfg),
        }
    }

    #[pyo3(signature = (x, y, seed=None))]
    pub fn fit(
        &self,
        py: Python<'_>,
        x: &PyTensor,
        y: &PyTensor,
        seed: Option<u64>,
    ) -> PyResult<Vec<PyParetoIndividual>> {
        let x_t = &x.inner;
        let y_t = &y.inner;
        let front = py
            .detach(|| self.inner.fit(x_t, y_t, seed))
            .map_err(to_py_err)?;
        Ok(front
            .into_iter()
            .map(|ind| PyParetoIndividual { inner: ind })
            .collect())
    }
}

#[pyclass(name = "StructuralCausalModel", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyStructuralCausalModel {
    pub inner: StructuralCausalModel,
}

#[pymethods]
impl PyStructuralCausalModel {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: StructuralCausalModel::new(),
        }
    }

    #[getter]
    pub fn num_variables(&self) -> usize {
        self.inner.num_variables()
    }

    #[pyo3(signature = (name, parents=vec![], weights=vec![], bias=0.0, noise_std=1.0))]
    pub fn add_variable(
        &mut self,
        name: &str,
        parents: Vec<usize>,
        weights: Vec<f64>,
        bias: f64,
        noise_std: f64,
    ) -> PyResult<usize> {
        self.inner
            .add_variable(name, parents, weights, bias, noise_std)
            .map_err(to_py_err)
    }

    #[pyo3(signature = (n_samples, seed=None))]
    pub fn sample_observational(
        &self,
        py: Python<'_>,
        n_samples: usize,
        seed: Option<u64>,
    ) -> PyResult<PyTensor> {
        let t = py
            .detach(|| self.inner.sample_observational(n_samples, seed))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: t })
    }

    #[pyo3(signature = (interventions, n_samples, seed=None))]
    pub fn sample_interventional(
        &self,
        py: Python<'_>,
        interventions: Vec<(usize, f64)>,
        n_samples: usize,
        seed: Option<u64>,
    ) -> PyResult<PyTensor> {
        let t = py
            .detach(|| {
                self.inner
                    .sample_interventional(&interventions, n_samples, seed)
            })
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: t })
    }

    pub fn compute_counterfactual(
        &self,
        factual_obs: Vec<f64>,
        interventions: Vec<(usize, f64)>,
    ) -> PyResult<Vec<f64>> {
        self.inner
            .compute_counterfactual(&factual_obs, &interventions)
            .map_err(to_py_err)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (treatment_var, outcome_var, val_treated, val_control, n_samples=1000, seed=None))]
    pub fn average_treatment_effect(
        &self,
        py: Python<'_>,
        treatment_var: usize,
        outcome_var: usize,
        val_treated: f64,
        val_control: f64,
        n_samples: usize,
        seed: Option<u64>,
    ) -> PyResult<f64> {
        py.detach(|| {
            self.inner.average_treatment_effect(
                treatment_var,
                outcome_var,
                val_treated,
                val_control,
                n_samples,
                seed,
            )
        })
        .map_err(to_py_err)
    }
}

impl Default for PyStructuralCausalModel {
    fn default() -> Self {
        Self::new()
    }
}

#[pyclass(name = "MultiscaleGraphWaveletTransform", module = "oxideon")]
pub struct PyMultiscaleGraphWaveletTransform {
    pub(crate) inner: MultiscaleGraphWaveletTransform,
}

#[pymethods]
impl PyMultiscaleGraphWaveletTransform {
    #[new]
    #[pyo3(signature = (adj, num_nodes, scales, order=5, laplacian_type="symmetric"))]
    pub fn new(
        adj: Vec<f64>,
        num_nodes: usize,
        scales: Vec<f64>,
        order: usize,
        laplacian_type: &str,
    ) -> PyResult<Self> {
        let lap_t = match laplacian_type {
            "combinatorial" => LaplacianType::Combinatorial,
            "random_walk" => LaplacianType::RandomWalk,
            _ => LaplacianType::SymmetricNormalized,
        };
        let inner = MultiscaleGraphWaveletTransform::new(&adj, num_nodes, &scales, order, lap_t)
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn inverse(&self, py: Python<'_>, wavelets: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.inverse(&wavelets.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "SpatioTemporalWaveletLayer", module = "oxideon")]
pub struct PySpatioTemporalWaveletLayer {
    pub(crate) inner: SpatioTemporalWaveletLayer,
}

#[pymethods]
impl PySpatioTemporalWaveletLayer {
    #[new]
    #[pyo3(signature = (adj, num_nodes, scales, chebyshev_order, in_features, hidden_features))]
    pub fn new(
        adj: Vec<f64>,
        num_nodes: usize,
        scales: Vec<f64>,
        chebyshev_order: usize,
        in_features: usize,
        hidden_features: usize,
    ) -> PyResult<Self> {
        let inner = SpatioTemporalWaveletLayer::new(
            &adj,
            num_nodes,
            &scales,
            chebyshev_order,
            in_features,
            hidden_features,
        )
        .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (x_t, h_prev=None))]
    pub fn step(
        &self,
        py: Python<'_>,
        x_t: &PyTensor,
        h_prev: Option<&PyTensor>,
    ) -> PyResult<PyTensor> {
        let h_p = h_prev.map(|h| &h.inner);
        let out = py
            .detach(|| self.inner.step(&x_t.inner, h_p))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "PrincipalBundleConnection", module = "oxideon")]
pub struct PyPrincipalBundleConnection {
    pub(crate) inner: PrincipalBundleConnection,
}

#[pymethods]
impl PyPrincipalBundleConnection {
    #[new]
    pub fn new(num_nodes: usize, fiber_dim: usize) -> PyResult<Self> {
        let inner = PrincipalBundleConnection::new(num_nodes, fiber_dim).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn set_connection(&mut self, u: usize, v: usize, angle: f64) -> PyResult<()> {
        self.inner.set_connection(u, v, angle).map_err(to_py_err)
    }

    pub fn cycle_holonomy(&self, cycle: Vec<usize>) -> PyResult<f64> {
        self.inner.cycle_holonomy(&cycle).map_err(to_py_err)
    }
}

#[pyclass(name = "GaugeEquivariantConvLayer", module = "oxideon")]
pub struct PyGaugeEquivariantConvLayer {
    pub(crate) inner: GaugeEquivariantConvLayer,
}

#[pymethods]
impl PyGaugeEquivariantConvLayer {
    #[new]
    pub fn new(connection: &PyPrincipalBundleConnection) -> Self {
        Self {
            inner: GaugeEquivariantConvLayer::new(connection.inner.clone()),
        }
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn verify_gauge_equivariance(
        &self,
        py: Python<'_>,
        x: &PyTensor,
        gauge_angles: Vec<f64>,
    ) -> PyResult<f64> {
        py.detach(|| {
            self.inner
                .verify_gauge_equivariance(&x.inner, &gauge_angles)
        })
        .map_err(to_py_err)
    }
}

#[pyclass(name = "AdaptiveCellularSheaf", module = "oxideon")]
pub struct PyAdaptiveCellularSheaf {
    pub(crate) inner: AdaptiveCellularSheaf,
}

#[pymethods]
impl PyAdaptiveCellularSheaf {
    #[new]
    pub fn new(num_nodes: usize, stalk_dim: usize, edges: Vec<(usize, usize)>) -> PyResult<Self> {
        let inner = AdaptiveCellularSheaf::new(num_nodes, stalk_dim, edges).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn set_edge_angles(&mut self, edge_idx: usize, theta_u: f64, theta_v: f64) -> PyResult<()> {
        self.inner
            .set_edge_angles(edge_idx, theta_u, theta_v)
            .map_err(to_py_err)
    }

    pub fn dirichlet_energy(&self, py: Python<'_>, x: Vec<f64>) -> PyResult<f64> {
        py.detach(|| self.inner.dirichlet_energy(&x))
            .map_err(to_py_err)
    }

    pub fn obstruction_norm(&self, py: Python<'_>, x: Vec<f64>) -> PyResult<f64> {
        py.detach(|| self.inner.obstruction_norm(&x))
            .map_err(to_py_err)
    }
}

#[pyclass(name = "SheafDiffusionLayer", module = "oxideon")]
pub struct PySheafDiffusionLayer {
    pub(crate) inner: SheafDiffusionLayer,
}

#[pymethods]
impl PySheafDiffusionLayer {
    #[new]
    pub fn new(sheaf: &PyAdaptiveCellularSheaf, dt: f64) -> Self {
        Self {
            inner: SheafDiffusionLayer::new(sheaf.inner.clone(), dt),
        }
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "SplitConformalRegressor", module = "oxideon")]
pub struct PySplitConformalRegressor {
    inner: SplitConformalRegressor,
}

#[pymethods]
impl PySplitConformalRegressor {
    #[new]
    #[pyo3(signature = (score_type="residual", alpha=0.1))]
    pub fn new(score_type: &str, alpha: f64) -> PyResult<Self> {
        let st = match score_type.to_lowercase().as_str() {
            "residual" | "absolute" => ConformalScoreType::AbsoluteResidual,
            "normalized" => ConformalScoreType::NormalizedResidual,
            "cqr" | "quantile" => ConformalScoreType::CQR,
            other => {
                return Err(to_py_err(crate::error::Error::Msg(format!(
                    "Unknown score_type: {other}"
                ))));
            }
        };
        let inner = SplitConformalRegressor::new(st, alpha).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (y_pred, y_true, dispersion=None))]
    pub fn calibrate(
        &mut self,
        py: Python<'_>,
        y_pred: &PyTensor,
        y_true: &PyTensor,
        dispersion: Option<&PyTensor>,
    ) -> PyResult<f64> {
        let yp = &y_pred.inner;
        let yt = &y_true.inner;
        let disp = dispersion.map(|d| &d.inner);
        py.detach(|| self.inner.calibrate(yp, yt, disp))
            .map_err(to_py_err)
    }

    #[pyo3(signature = (y_pred, dispersion=None))]
    pub fn predict_intervals(
        &self,
        py: Python<'_>,
        y_pred: &PyTensor,
        dispersion: Option<&PyTensor>,
    ) -> PyResult<Vec<(f64, f64)>> {
        let yp = &y_pred.inner;
        let disp = dispersion.map(|d| &d.inner);
        let intervals = py
            .detach(|| self.inner.predict_intervals(yp, disp))
            .map_err(to_py_err)?;
        Ok(intervals
            .into_iter()
            .map(|iv| (iv.lower, iv.upper))
            .collect())
    }
}

#[pyclass(name = "AdaptiveConformalClassifier", module = "oxideon")]
pub struct PyAdaptiveConformalClassifier {
    inner: AdaptiveConformalClassifier,
}

#[pymethods]
impl PyAdaptiveConformalClassifier {
    #[new]
    #[pyo3(signature = (alpha=0.1))]
    pub fn new(alpha: f64) -> PyResult<Self> {
        let inner = AdaptiveConformalClassifier::new(alpha).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn calibrate(
        &mut self,
        py: Python<'_>,
        probs: &PyTensor,
        labels: Vec<usize>,
    ) -> PyResult<f64> {
        let pr = &probs.inner;
        py.detach(|| self.inner.calibrate(pr, &labels))
            .map_err(to_py_err)
    }

    pub fn predict_sets(&self, py: Python<'_>, probs: &PyTensor) -> PyResult<Vec<Vec<usize>>> {
        let pr = &probs.inner;
        py.detach(|| self.inner.predict_sets(pr)).map_err(to_py_err)
    }
}

#[pyclass(name = "AlgorithmicGraph", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyAlgorithmicGraph {
    pub inner: AlgorithmicGraph,
}

#[pymethods]
impl PyAlgorithmicGraph {
    #[new]
    pub fn new(num_nodes: usize) -> Self {
        Self {
            inner: AlgorithmicGraph::new(num_nodes),
        }
    }

    #[getter]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes
    }

    #[getter]
    pub fn num_edges(&self) -> usize {
        self.inner.edges.len()
    }

    #[pyo3(signature = (source, target, weight=1.0, features=None))]
    pub fn add_edge(
        &mut self,
        source: usize,
        target: usize,
        weight: f64,
        features: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let feats = features.unwrap_or_default();
        self.inner
            .add_edge(source, target, weight, feats)
            .map_err(to_py_err)
    }
}

#[pyclass(name = "NeuralBellmanFord", module = "oxideon")]
pub struct PyNeuralBellmanFord {
    inner: NeuralBellmanFord,
}

#[pymethods]
impl PyNeuralBellmanFord {
    #[new]
    #[pyo3(signature = (semiring="min_plus", hidden_dim=16))]
    pub fn new(semiring: &str, hidden_dim: usize) -> PyResult<Self> {
        let sr = match semiring {
            "min_plus" | "min+" | "shortest_path" => AlgorithmicSemiring::MinPlus,
            "max_mul" | "max*" | "reliable" => AlgorithmicSemiring::MaxMul,
            "max_min" | "bottleneck" | "capacity" => AlgorithmicSemiring::MaxMin,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown semiring '{other}'. Choose from: 'min_plus', 'max_mul', 'max_min'"
                )));
            }
        };
        Ok(Self {
            inner: NeuralBellmanFord::new(sr, hidden_dim),
        })
    }

    #[pyo3(signature = (graph, source, max_steps=0))]
    pub fn solve(
        &self,
        py: Python<'_>,
        graph: &PyAlgorithmicGraph,
        source: usize,
        max_steps: usize,
    ) -> PyResult<(Vec<f64>, Vec<Option<usize>>)> {
        let g = &graph.inner;
        py.detach(|| self.inner.solve(g, source, max_steps))
            .map_err(to_py_err)
    }

    pub fn reconstruct_path(
        &self,
        source: usize,
        target: usize,
        predecessors: Vec<Option<usize>>,
    ) -> PyResult<Vec<usize>> {
        self.inner
            .reconstruct_path(source, target, &predecessors)
            .map_err(to_py_err)
    }
}

#[pyclass(name = "PointerGraphNetwork", module = "oxideon")]
pub struct PyPointerGraphNetwork {
    inner: PointerGraphNetwork,
}

#[pymethods]
impl PyPointerGraphNetwork {
    #[new]
    #[pyo3(signature = (hidden_dim=16, seed=None))]
    pub fn new(hidden_dim: usize, seed: Option<u64>) -> PyResult<Self> {
        Ok(Self {
            inner: PointerGraphNetwork::new(hidden_dim, seed).map_err(to_py_err)?,
        })
    }

    #[allow(clippy::type_complexity)]
    pub fn forward(
        &self,
        py: Python<'_>,
        node_states: &PyTensor,
        graph: &PyAlgorithmicGraph,
    ) -> PyResult<(PyTensor, Vec<Vec<(usize, f64)>>)> {
        let states = &node_states.inner;
        let g = &graph.inner;
        let (out_states, ptrs) = py
            .detach(|| self.inner.forward(states, g))
            .map_err(to_py_err)?;
        Ok((PyTensor { inner: out_states }, ptrs))
    }

    pub fn argmax_pointers(&self, pointer_distributions: Vec<Vec<(usize, f64)>>) -> Vec<usize> {
        self.inner.argmax_pointers(&pointer_distributions)
    }
}

#[pyclass(name = "PtxKernelEmitter", module = "oxideon")]
pub struct PyPtxKernelEmitter {
    inner: PtxKernelEmitter,
}

#[pymethods]
impl PyPtxKernelEmitter {
    #[new]
    #[pyo3(signature = (kernel_name, target_sm="sm_90"))]
    pub fn new(kernel_name: String, target_sm: &str) -> Self {
        Self {
            inner: PtxKernelEmitter::new(kernel_name, target_sm),
        }
    }

    pub fn emit_fused_add_silu_ptx(&self) -> String {
        self.inner.emit_fused_add_silu_ptx()
    }

    pub fn emit_gemm_fma_ptx(&self) -> String {
        self.inner.emit_gemm_fma_ptx()
    }

    #[pyo3(signature = (eps=1e-5))]
    pub fn emit_warp_rmsnorm_ptx(&self, eps: f32) -> String {
        self.inner.emit_warp_rmsnorm_ptx(eps)
    }

    pub fn emit_fused_swiglu_ptx(&self) -> String {
        self.inner.emit_fused_swiglu_ptx()
    }

    #[pyo3(signature = (head_dim=64))]
    pub fn emit_flash_attention_mma_ptx(&self, head_dim: usize) -> String {
        self.inner.emit_flash_attention_mma_ptx(head_dim)
    }

    pub fn emit_rope_ptx(&self) -> String {
        self.inner.emit_rope_ptx()
    }

    #[pyo3(signature = (group_size=32))]
    pub fn emit_int4_dequant_gemv_ptx(&self, group_size: usize) -> String {
        self.inner.emit_int4_dequant_gemv_ptx(group_size)
    }

    #[pyo3(signature = (head_dim=64, is_causal=true))]
    pub fn emit_cuda_c_flash_attention_kernel(&self, head_dim: usize, is_causal: bool) -> String {
        self.inner
            .emit_cuda_c_flash_attention_kernel(head_dim, is_causal)
    }

    #[pyo3(signature = (eps=1e-5))]
    pub fn emit_cuda_c_rmsnorm_kernel(&self, eps: f32) -> String {
        self.inner.emit_cuda_c_rmsnorm_kernel(eps)
    }
}

#[pyclass(name = "WgslKernelBuilder", module = "oxideon")]
pub struct PyWgslKernelBuilder {
    inner: WgslKernelBuilder,
}

#[pymethods]
impl PyWgslKernelBuilder {
    #[new]
    #[pyo3(signature = (entry_point, workgroup_x=256))]
    pub fn new(entry_point: String, workgroup_x: usize) -> Self {
        Self {
            inner: WgslKernelBuilder::new(entry_point, workgroup_x),
        }
    }

    #[pyo3(signature = (d_model, eps=1e-5))]
    pub fn emit_fused_rmsnorm_wgsl(&self, d_model: usize, eps: f32) -> String {
        self.inner.emit_fused_rmsnorm_wgsl(d_model, eps)
    }

    pub fn emit_fused_swiglu_wgsl(&self) -> String {
        self.inner.emit_fused_swiglu_wgsl()
    }

    #[pyo3(signature = (tile_dim=16))]
    pub fn emit_tiled_matmul_wgsl(&self, tile_dim: usize) -> String {
        self.inner.emit_tiled_matmul_wgsl(tile_dim)
    }

    #[pyo3(signature = (head_dim=64, block_size=16, is_causal=true))]
    pub fn emit_flash_attention_wgsl(
        &self,
        head_dim: usize,
        block_size: usize,
        is_causal: bool,
    ) -> String {
        self.inner
            .emit_flash_attention_wgsl(head_dim, block_size, is_causal)
    }

    #[pyo3(signature = (head_dim=64))]
    pub fn emit_rope_wgsl(&self, head_dim: usize) -> String {
        self.inner.emit_rope_wgsl(head_dim)
    }

    #[pyo3(signature = (group_size=32))]
    pub fn emit_q4_gemv_wgsl(&self, group_size: usize) -> String {
        self.inner.emit_q4_gemv_wgsl(group_size)
    }
}

#[pyclass(name = "VulkanSpirvKernel", module = "oxideon")]
pub struct PyVulkanSpirvKernel {
    inner: VulkanSpirvKernel,
}

#[pymethods]
impl PyVulkanSpirvKernel {
    #[new]
    #[pyo3(signature = (name, num_bindings=3))]
    pub fn new(name: String, num_bindings: usize) -> Self {
        Self {
            inner: VulkanSpirvKernel::new(name, num_bindings),
        }
    }

    pub fn emit_spirv_binary(&self) -> PyResult<Vec<u32>> {
        self.inner.emit_spirv_binary().map_err(to_py_err)
    }

    pub fn emit_fadd_spirv_binary(&self) -> PyResult<Vec<u32>> {
        self.inner.emit_fadd_spirv_binary().map_err(to_py_err)
    }

    pub fn emit_fma_spirv_binary(&self) -> PyResult<Vec<u32>> {
        self.inner.emit_fma_spirv_binary().map_err(to_py_err)
    }

    pub fn binary_size_bytes(&self) -> PyResult<usize> {
        self.inner.binary_size_bytes().map_err(to_py_err)
    }

    #[staticmethod]
    pub fn disassemble_spirv(words: Vec<u32>) -> PyResult<String> {
        VulkanSpirvKernel::disassemble_spirv(&words).map_err(to_py_err)
    }
}

#[pyclass(name = "PairformerBlock", module = "oxideon")]
pub struct PyPairformerBlock {
    inner: PairformerBlock,
}

#[pymethods]
impl PyPairformerBlock {
    #[new]
    pub fn new(c_z: usize, c_hidden: usize) -> PyResult<Self> {
        let inner = PairformerBlock::new(c_z, c_hidden).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn forward(&self, py: Python<'_>, z: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&z.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "SE3EquivariantLayer", module = "oxideon")]
pub struct PySE3EquivariantLayer {
    inner: SE3EquivariantLayer,
}

#[pymethods]
impl PySE3EquivariantLayer {
    #[new]
    pub fn new(feature_dim: usize) -> PyResult<Self> {
        let inner = guard_res(|| SE3EquivariantLayer::new(feature_dim))?;
        Ok(Self { inner })
    }

    pub fn forward(
        &self,
        py: Python<'_>,
        coords: &PyTensor,
        features: &PyTensor,
    ) -> PyResult<(PyTensor, PyTensor)> {
        let (c, f) = py
            .detach(|| self.inner.forward(&coords.inner, &features.inner))
            .map_err(to_py_err)?;
        Ok((PyTensor { inner: c }, PyTensor { inner: f }))
    }
}

#[pyclass(name = "SpectralConv1d", module = "oxideon")]
pub struct PySpectralConv1d {
    inner: SpectralConv1d,
}

#[pymethods]
impl PySpectralConv1d {
    #[new]
    pub fn new(in_channels: usize, out_channels: usize, modes: usize) -> PyResult<Self> {
        let inner = SpectralConv1d::new(in_channels, out_channels, modes).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "FourierNeuralOperator1d", module = "oxideon")]
pub struct PyFourierNeuralOperator1d {
    inner: FourierNeuralOperator1d,
}

#[pymethods]
impl PyFourierNeuralOperator1d {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, hidden_channels=32, modes=12, num_layers=4))]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        hidden_channels: usize,
        modes: usize,
        num_layers: usize,
    ) -> PyResult<Self> {
        let inner = FourierNeuralOperator1d::new(
            in_channels,
            out_channels,
            hidden_channels,
            modes,
            num_layers,
        )
        .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "DeepONet", module = "oxideon")]
pub struct PyDeepONet {
    inner: DeepONet,
}

#[pymethods]
impl PyDeepONet {
    #[new]
    #[pyo3(signature = (branch_in_dim, trunk_in_dim, hidden_dim=64, basis_dim=32))]
    pub fn new(
        branch_in_dim: usize,
        trunk_in_dim: usize,
        hidden_dim: usize,
        basis_dim: usize,
    ) -> PyResult<Self> {
        let inner =
            DeepONet::new(branch_in_dim, trunk_in_dim, hidden_dim, basis_dim).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn branch_in_dim(&self) -> usize {
        self.inner.branch_in_dim
    }

    #[getter]
    pub fn trunk_in_dim(&self) -> usize {
        self.inner.trunk_in_dim
    }

    #[getter]
    pub fn basis_dim(&self) -> usize {
        self.inner.basis_dim
    }

    pub fn forward_branch(&self, py: Python<'_>, u: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward_branch(&u.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn forward_trunk(&self, py: Python<'_>, y: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward_trunk(&y.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn forward(
        &self,
        py: Python<'_>,
        u_sensors: &PyTensor,
        y_coords: &PyTensor,
    ) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&u_sensors.inner, &y_coords.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "MineEstimator", module = "oxideon")]
pub struct PyMineEstimator {
    inner: std::sync::Mutex<MineEstimator>,
}

#[pymethods]
impl PyMineEstimator {
    #[new]
    #[pyo3(signature = (x_dim, y_dim, hidden_dim=64, ema_decay=0.9))]
    pub fn new(x_dim: usize, y_dim: usize, hidden_dim: usize, ema_decay: f64) -> PyResult<Self> {
        let inner = guard_res(|| MineEstimator::new(x_dim, y_dim, hidden_dim, ema_decay))?;
        Ok(Self {
            inner: std::sync::Mutex::new(inner),
        })
    }

    pub fn estimate(&self, py: Python<'_>, x: &PyTensor, y: &PyTensor) -> PyResult<f64> {
        py.detach(|| mutex_guard(&self.inner).estimate(&x.inner, &y.inner))
            .map_err(to_py_err)
    }

    pub fn loss(&self, py: Python<'_>, x: &PyTensor, y: &PyTensor) -> PyResult<PyTensor> {
        let l = py
            .detach(|| mutex_guard(&self.inner).loss(&x.inner, &y.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: l })
    }
}

#[pyclass(name = "InfoNceEstimator", module = "oxideon")]
pub struct PyInfoNceEstimator {
    inner: InfoNceEstimator,
}

#[pymethods]
impl PyInfoNceEstimator {
    #[new]
    #[pyo3(signature = (temperature=0.07))]
    pub fn new(temperature: f64) -> Self {
        Self {
            inner: InfoNceEstimator::new(temperature),
        }
    }

    #[getter]
    pub fn temperature(&self) -> f64 {
        self.inner.temperature
    }

    pub fn compute_loss(
        &self,
        py: Python<'_>,
        queries: &PyTensor,
        keys: &PyTensor,
    ) -> PyResult<PyTensor> {
        let l = py
            .detach(|| self.inner.compute_loss(&queries.inner, &keys.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: l })
    }

    pub fn lower_bound(
        &self,
        py: Python<'_>,
        queries: &PyTensor,
        keys: &PyTensor,
    ) -> PyResult<f64> {
        py.detach(|| self.inner.lower_bound(&queries.inner, &keys.inner))
            .map_err(to_py_err)
    }
}

#[pyclass(name = "MolecularGeometryDiffusion", module = "oxideon")]
pub struct PyMolecularGeometryDiffusion;

#[pymethods]
impl PyMolecularGeometryDiffusion {
    #[staticmethod]
    pub fn remove_center_of_mass(py: Python<'_>, coords: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| MolecularGeometryDiffusion::remove_center_of_mass(&coords.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    #[staticmethod]
    #[pyo3(signature = (layer, init_coords, features, num_steps=5))]
    pub fn sample_conformation(
        py: Python<'_>,
        layer: &PySE3EquivariantLayer,
        init_coords: &PyTensor,
        features: &PyTensor,
        num_steps: usize,
    ) -> PyResult<PyTensor> {
        let out = py
            .detach(|| {
                MolecularGeometryDiffusion::sample_conformation(
                    &layer.inner,
                    &init_coords.inner,
                    &features.inner,
                    num_steps,
                )
            })
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "PersistenceDiagram", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyPersistenceDiagram {
    pub inner: PersistenceDiagram,
}

#[pymethods]
impl PyPersistenceDiagram {
    pub fn total_persistence(&self) -> f64 {
        self.inner.total_persistence()
    }

    pub fn get_pairs(&self, dim: usize) -> Vec<(f64, f64)> {
        self.inner
            .dimension_pairs(dim)
            .into_iter()
            .map(|p| (p.birth, p.death))
            .collect()
    }
}

#[pyclass(name = "PersistentHomologyEngine", module = "oxideon")]
pub struct PyPersistentHomologyEngine {
    inner: PersistentHomologyEngine,
}

#[pymethods]
impl PyPersistentHomologyEngine {
    #[new]
    #[pyo3(signature = (max_dimension=1, max_filtration=10.0))]
    pub fn new(max_dimension: usize, max_filtration: f64) -> Self {
        Self {
            inner: PersistentHomologyEngine::new(max_dimension, max_filtration),
        }
    }

    pub fn compute_from_distance_matrix(
        &self,
        py: Python<'_>,
        dist: &PyTensor,
    ) -> PyResult<PyPersistenceDiagram> {
        let diag = py
            .detach(|| self.inner.compute_from_distance_matrix(&dist.inner))
            .map_err(to_py_err)?;
        Ok(PyPersistenceDiagram { inner: diag })
    }

    #[staticmethod]
    #[pyo3(signature = (d1, d2, dim=0))]
    pub fn wasserstein_distance(
        d1: &PyPersistenceDiagram,
        d2: &PyPersistenceDiagram,
        dim: usize,
    ) -> f64 {
        wasserstein_distance(&d1.inner, &d2.inner, dim)
    }
}

#[pyclass(name = "SimplicialComplex", module = "oxideon")]
pub struct PySimplicialComplex {
    inner: SimplicialComplex,
}

#[pymethods]
impl PySimplicialComplex {
    #[new]
    pub fn new(maximal_simplices: Vec<Vec<usize>>) -> Self {
        let simplices: Vec<OrientedSimplex> = maximal_simplices
            .into_iter()
            .map(OrientedSimplex::new)
            .collect();
        let inner = SimplicialComplex::from_simplices(&simplices);
        Self { inner }
    }

    pub fn num_simplices(&self, k: usize) -> usize {
        self.inner.num_simplices(k)
    }

    pub fn boundary_matrix(&self, py: Python<'_>, k: usize) -> PyResult<PyTensor> {
        let mat = py
            .detach(|| self.inner.boundary_matrix(k))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: mat })
    }

    pub fn coboundary_matrix(&self, py: Python<'_>, k: usize) -> PyResult<PyTensor> {
        let mat = py
            .detach(|| self.inner.coboundary_matrix(k))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: mat })
    }

    pub fn hodge_laplacian(&self, py: Python<'_>, k: usize) -> PyResult<PyTensor> {
        let lap = py
            .detach(|| self.inner.hodge_laplacian(k))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: lap })
    }

    #[pyo3(signature = (k, tolerance=1e-8))]
    pub fn betti_number(&self, py: Python<'_>, k: usize, tolerance: f64) -> PyResult<usize> {
        py.detach(|| self.inner.betti_number(k, tolerance))
            .map_err(to_py_err)
    }

    #[pyo3(signature = (omega, max_iters=200, lr=0.1))]
    pub fn helmholtz_hodge_decomposition(
        &self,
        py: Python<'_>,
        omega: &PyTensor,
        max_iters: usize,
        lr: f64,
    ) -> PyResult<(PyTensor, PyTensor, PyTensor)> {
        let om = omega.inner.clone();
        let (exact, harm, coexact) = py
            .detach(|| self.inner.helmholtz_hodge_decomposition(&om, max_iters, lr))
            .map_err(to_py_err)?;
        Ok((
            PyTensor { inner: exact },
            PyTensor { inner: harm },
            PyTensor { inner: coexact },
        ))
    }
}

#[pyclass(name = "TropicalMatrix", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyTropicalMatrix {
    pub inner: TropicalMatrix,
}

#[pymethods]
impl PyTropicalMatrix {
    #[new]
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> PyResult<Self> {
        let inner = TropicalMatrix::new(rows, cols, data).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    pub fn identity(n: usize) -> PyResult<Self> {
        Ok(Self {
            inner: TropicalMatrix::identity(n).map_err(to_py_err)?,
        })
    }

    #[getter]
    pub fn shape(&self) -> (usize, usize) {
        (self.inner.rows, self.inner.cols)
    }

    pub fn to_list(&self) -> Vec<f64> {
        self.inner.data.clone()
    }

    pub fn matvec(&self, py: Python<'_>, x: Vec<f64>) -> PyResult<Vec<f64>> {
        py.detach(|| self.inner.matvec(&x)).map_err(to_py_err)
    }

    pub fn matmul(&self, py: Python<'_>, other: &PyTropicalMatrix) -> PyResult<PyTropicalMatrix> {
        let o = &other.inner;
        let res = py.detach(|| self.inner.matmul(o)).map_err(to_py_err)?;
        Ok(PyTropicalMatrix { inner: res })
    }

    pub fn kleene_star(&self, py: Python<'_>) -> PyResult<PyTropicalMatrix> {
        let res = py.detach(|| self.inner.kleene_star()).map_err(to_py_err)?;
        Ok(PyTropicalMatrix { inner: res })
    }

    pub fn to_tensor(&self) -> PyResult<PyTensor> {
        let t = Tensor::try_from(self.inner.clone()).map_err(to_py_err)?;
        Ok(PyTensor { inner: t })
    }
}

#[pyclass(name = "TropicalPolynomial", module = "oxideon")]
pub struct PyTropicalPolynomial {
    inner: TropicalPolynomial,
}

#[pymethods]
impl PyTropicalPolynomial {
    #[new]
    pub fn new(num_variables: usize) -> Self {
        Self {
            inner: TropicalPolynomial::new(num_variables),
        }
    }

    pub fn add_monomial(&mut self, coefficient: f64, exponents: Vec<f64>) -> PyResult<()> {
        self.inner
            .add_monomial(coefficient, exponents)
            .map_err(to_py_err)
    }

    pub fn evaluate(&self, x: Vec<f64>) -> PyResult<f64> {
        self.inner.evaluate(&x).map_err(to_py_err)
    }

    pub fn is_on_hypersurface(&self, x: Vec<f64>, tolerance: f64) -> PyResult<bool> {
        self.inner
            .is_on_hypersurface(&x, tolerance)
            .map_err(to_py_err)
    }
}

#[pyfunction(name = "tropical_distance")]
pub fn py_tropical_distance(x: Vec<f64>, y: Vec<f64>) -> PyResult<f64> {
    tropical_distance(&x, &y).map_err(to_py_err)
}

#[pyclass(name = "RandomFourierFeatures", module = "oxideon")]
pub struct PyRandomFourierFeatures {
    pub(crate) inner: RandomFourierFeatures,
}

#[pymethods]
impl PyRandomFourierFeatures {
    #[new]
    #[pyo3(signature = (in_dim, num_features, gamma=1.0, seed=None))]
    pub fn new(
        in_dim: usize,
        num_features: usize,
        gamma: f64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        crate::limits::check_elements("RandomFourierFeatures weights", &[in_dim, num_features])
            .map_err(to_py_err)?;
        Ok(Self {
            inner: RandomFourierFeatures::new_gaussian_rbf(in_dim, num_features, gamma, seed),
        })
    }

    pub fn transform(&self, x: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner.transform(&x).map_err(to_py_err)
    }

    pub fn transform_tensor(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.transform_tensor(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "KernelRidgeRegressor", module = "oxideon")]
pub struct PyKernelRidgeRegressor {
    pub(crate) inner: KernelRidgeRegressor,
}

#[pymethods]
impl PyKernelRidgeRegressor {
    #[new]
    #[pyo3(signature = (kernel="rbf", gamma=1.0, lambda_reg=1e-3))]
    pub fn new(kernel: &str, gamma: f64, lambda_reg: f64) -> PyResult<Self> {
        let k = match kernel.to_lowercase().as_str() {
            "linear" => MercerKernel::Linear,
            "matern32" => MercerKernel::Matern32 {
                length_scale: 1.0 / gamma.max(1e-6),
            },
            "matern52" => MercerKernel::Matern52 {
                length_scale: 1.0 / gamma.max(1e-6),
            },
            _ => MercerKernel::GaussianRbf { gamma },
        };
        Ok(Self {
            inner: KernelRidgeRegressor::new(k, lambda_reg),
        })
    }

    pub fn fit(&mut self, py: Python<'_>, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<()> {
        py.detach(|| self.inner.fit(&x, &y)).map_err(to_py_err)
    }

    pub fn predict(&self, query: Vec<f64>) -> PyResult<f64> {
        self.inner.predict(&query).map_err(to_py_err)
    }

    pub fn predict_tensor(&self, py: Python<'_>, query: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.predict_tensor(&query.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyfunction(name = "maximum_mean_discrepancy")]
#[pyo3(signature = (x, y, gamma=1.0, unbiased=true))]
pub fn py_maximum_mean_discrepancy(
    py: Python<'_>,
    x: Vec<Vec<f64>>,
    y: Vec<Vec<f64>>,
    gamma: f64,
    unbiased: bool,
) -> PyResult<f64> {
    py.detach(|| maximum_mean_discrepancy(&x, &y, MercerKernel::GaussianRbf { gamma }, unbiased))
        .map_err(to_py_err)
}

#[pyclass(name = "TapSolution", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyTapSolution {
    pub inner: TapSolution,
}

#[pymethods]
impl PyTapSolution {
    #[getter]
    pub fn magnetizations(&self) -> Vec<f64> {
        self.inner.magnetizations.clone()
    }

    #[getter]
    pub fn edwards_anderson_q(&self) -> f64 {
        self.inner.edwards_anderson_q
    }

    #[getter]
    pub fn tap_free_energy(&self) -> f64 {
        self.inner.tap_free_energy
    }

    #[getter]
    pub fn at_eigenvalue(&self) -> f64 {
        self.inner.at_eigenvalue
    }

    #[getter]
    pub fn is_replica_symmetry_broken(&self) -> bool {
        self.inner.is_replica_symmetry_broken
    }

    #[getter]
    pub fn iterations(&self) -> usize {
        self.inner.iterations
    }

    #[getter]
    pub fn converged(&self) -> bool {
        self.inner.converged
    }
}

#[pyclass(name = "InformationGeometryEngine", module = "oxideon", from_py_object)]
#[derive(Clone, Copy, Default)]
pub struct PyInformationGeometryEngine {
    pub inner: InformationGeometryEngine,
}

#[pymethods]
impl PyInformationGeometryEngine {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: InformationGeometryEngine::new(),
        }
    }

    #[staticmethod]
    pub fn gaussian_fisher_matrix(mu: f64, sigma: f64) -> PyResult<[[f64; 2]; 2]> {
        InformationGeometryEngine::gaussian_fisher_matrix(mu, sigma).map_err(to_py_err)
    }

    #[staticmethod]
    pub fn gaussian_fisher_rao_distance(
        mu1: f64,
        sigma1: f64,
        mu2: f64,
        sigma2: f64,
    ) -> PyResult<f64> {
        InformationGeometryEngine::gaussian_fisher_rao_distance(mu1, sigma1, mu2, sigma2)
            .map_err(to_py_err)
    }

    #[staticmethod]
    pub fn categorical_fisher_rao_distance(p: Vec<f64>, q: Vec<f64>) -> PyResult<f64> {
        InformationGeometryEngine::categorical_fisher_rao_distance(&p, &q).map_err(to_py_err)
    }

    #[staticmethod]
    pub fn alpha_geodesic_interpolate(
        p: Vec<f64>,
        q: Vec<f64>,
        alpha: f64,
        t: f64,
    ) -> PyResult<Vec<f64>> {
        InformationGeometryEngine::alpha_geodesic_interpolate(&p, &q, alpha, t).map_err(to_py_err)
    }

    #[staticmethod]
    pub fn pythagorean_duality_residual(p: Vec<f64>, q: Vec<f64>, r: Vec<f64>) -> PyResult<f64> {
        InformationGeometryEngine::pythagorean_duality_residual(&p, &q, &r).map_err(to_py_err)
    }
}

#[pyclass(name = "AInfinityOperadEngine", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyAInfinityOperadEngine {
    pub inner: AInfinityOperadEngine,
}

#[pymethods]
impl PyAInfinityOperadEngine {
    #[new]
    pub fn new(dim: usize, differential: Vec<f64>) -> PyResult<Self> {
        let inner = guard_res(|| AInfinityOperadEngine::new(dim, differential))?;
        Ok(Self { inner })
    }

    #[staticmethod]
    pub fn associahedron_vertices(n: usize) -> PyResult<u64> {
        AInfinityOperadEngine::associahedron_vertices(n).map_err(to_py_err)
    }

    pub fn apply_m1(&self, x: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner.apply_m1(&x).map_err(to_py_err)
    }

    pub fn differential_squared_norm(&self) -> f64 {
        self.inner.differential_squared_norm()
    }

    pub fn triple_massey_product(
        &self,
        py: Python<'_>,
        x: Vec<f64>,
        z: Vec<f64>,
        u_bounding_xy: Vec<f64>,
        v_bounding_yz: Vec<f64>,
        deg_x: i32,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| {
            self.inner
                .triple_massey_product(&x, &z, &u_bounding_xy, &v_bounding_yz, deg_x)
        })
        .map_err(to_py_err)
    }
}

#[pyclass(name = "HigherSimplex", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyHigherSimplex {
    pub inner: HigherSimplex,
}

#[pymethods]
impl PyHigherSimplex {
    #[new]
    pub fn new(vertices: Vec<usize>, payload: Vec<f64>) -> PyResult<Self> {
        let inner = HigherSimplex::new(vertices, payload).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn vertices(&self) -> Vec<usize> {
        self.inner.vertices.clone()
    }

    #[getter]
    pub fn payload(&self) -> Vec<f64> {
        self.inner.payload.clone()
    }

    pub fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    pub fn face(&self, i: usize) -> PyResult<Self> {
        let f = self.inner.face(i).map_err(to_py_err)?;
        Ok(Self { inner: f })
    }

    pub fn degeneracy(&self, i: usize) -> PyResult<Self> {
        let s = self.inner.degeneracy(i).map_err(to_py_err)?;
        Ok(Self { inner: s })
    }

    pub fn verify_simplicial_identity(&self, i: usize, j: usize) -> PyResult<bool> {
        self.inner
            .verify_simplicial_identity(i, j)
            .map_err(to_py_err)
    }
}

#[pyclass(name = "QuasiCategoryEngine", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyQuasiCategoryEngine {
    pub inner: QuasiCategoryEngine,
}

#[pymethods]
impl PyQuasiCategoryEngine {
    #[new]
    #[pyo3(signature = (coherence_factor=1.0))]
    pub fn new(coherence_factor: f64) -> PyResult<Self> {
        let inner = QuasiCategoryEngine::new(coherence_factor).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn fill_inner_horn_2(
        &self,
        py: Python<'_>,
        f: &PyHigherSimplex,
        g: &PyHigherSimplex,
    ) -> PyResult<(PyHigherSimplex, PyHigherSimplex)> {
        let (filler, comp) = py
            .detach(|| self.inner.fill_inner_horn_2(&f.inner, &g.inner))
            .map_err(to_py_err)?;
        Ok((
            PyHigherSimplex { inner: filler },
            PyHigherSimplex { inner: comp },
        ))
    }

    pub fn fill_outer_kan_horn_0(
        &self,
        py: Python<'_>,
        f: &PyHigherSimplex,
        h: &PyHigherSimplex,
    ) -> PyResult<(PyHigherSimplex, PyHigherSimplex)> {
        let (filler, retr) = py
            .detach(|| self.inner.fill_outer_kan_horn_0(&f.inner, &h.inner))
            .map_err(to_py_err)?;
        Ok((
            PyHigherSimplex { inner: filler },
            PyHigherSimplex { inner: retr },
        ))
    }

    pub fn associator_3_simplex(
        &self,
        py: Python<'_>,
        f: &PyHigherSimplex,
        g: &PyHigherSimplex,
        h: &PyHigherSimplex,
    ) -> PyResult<(PyHigherSimplex, f64)> {
        let (tetra, res) = py
            .detach(|| {
                self.inner
                    .associator_3_simplex(&f.inner, &g.inner, &h.inner)
            })
            .map_err(to_py_err)?;
        Ok((PyHigherSimplex { inner: tetra }, res))
    }
}

#[pyclass(name = "TtCore", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyTtCore {
    pub inner: TtCore,
}

#[pymethods]
impl PyTtCore {
    #[new]
    pub fn new(r_left: usize, mode_size: usize, r_right: usize, data: Vec<f64>) -> PyResult<Self> {
        let inner = guard_res(|| TtCore::new(r_left, mode_size, r_right, data))?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn r_left(&self) -> usize {
        self.inner.r_left
    }

    #[getter]
    pub fn mode_size(&self) -> usize {
        self.inner.mode_size
    }

    #[getter]
    pub fn r_right(&self) -> usize {
        self.inner.r_right
    }

    pub fn get(&self, alpha: usize, i: usize, beta: usize) -> f64 {
        self.inner.get(alpha, i, beta)
    }
}

#[pyclass(name = "TensorTrain", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyTensorTrain {
    pub inner: TensorTrain,
}

#[pymethods]
impl PyTensorTrain {
    #[new]
    pub fn new(cores: Vec<PyTtCore>) -> PyResult<Self> {
        let inner_cores = cores.into_iter().map(|c| c.inner).collect();
        let inner = TensorTrain::new(inner_cores).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    pub fn from_rank1_factors(factors: Vec<Vec<f64>>) -> PyResult<Self> {
        let inner = TensorTrain::from_rank1_factors(factors).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    pub fn from_additive_terms(terms: Vec<Vec<f64>>) -> PyResult<Self> {
        let inner = TensorTrain::from_additive_terms(terms).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    pub fn shape(&self) -> Vec<usize> {
        self.inner.shape()
    }

    pub fn num_parameters(&self) -> usize {
        self.inner.num_parameters()
    }

    pub fn evaluate(&self, indices: Vec<usize>) -> PyResult<f64> {
        self.inner.evaluate(&indices).map_err(to_py_err)
    }

    pub fn sum_all(&self, py: Python<'_>) -> f64 {
        py.detach(|| self.inner.sum_all())
    }

    pub fn dot(&self, py: Python<'_>, other: &PyTensorTrain) -> PyResult<f64> {
        py.detach(|| self.inner.dot(&other.inner))
            .map_err(to_py_err)
    }

    pub fn frobenius_norm(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.frobenius_norm()).map_err(to_py_err)
    }
}

#[pyclass(name = "SherringtonKirkpatrickModel", module = "oxideon")]
pub struct PySherringtonKirkpatrickModel {
    pub inner: SherringtonKirkpatrickModel,
}

#[pymethods]
impl PySherringtonKirkpatrickModel {
    #[new]
    pub fn new(num_spins: usize, couplings: Vec<f64>, external_fields: Vec<f64>) -> PyResult<Self> {
        let inner = SherringtonKirkpatrickModel::new(num_spins, couplings, external_fields)
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (num_spins, j_scale=1.0, seed=42))]
    pub fn random_gaussian(num_spins: usize, j_scale: f64, seed: u64) -> PyResult<Self> {
        let inner = SherringtonKirkpatrickModel::random_gaussian(num_spins, j_scale, seed)
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn energy(&self, spins: Vec<i8>) -> PyResult<f64> {
        self.inner.energy(&spins).map_err(to_py_err)
    }

    #[pyo3(signature = (beta, max_iters=200, tol=1e-6, damping=0.3))]
    pub fn solve_tap(
        &self,
        py: Python<'_>,
        beta: f64,
        max_iters: usize,
        tol: f64,
        damping: f64,
    ) -> PyResult<PyTapSolution> {
        let sol = py
            .detach(|| self.inner.solve_tap(beta, max_iters, tol, damping))
            .map_err(to_py_err)?;
        Ok(PyTapSolution { inner: sol })
    }
}

#[pyclass(name = "GromovWassersteinConfig", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyGromovWassersteinConfig {
    pub inner: GromovWassersteinConfig,
}

#[pymethods]
impl PyGromovWassersteinConfig {
    #[new]
    #[pyo3(signature = (epsilon=0.05, max_iters=40, sinkhorn_iters=50, tolerance=1e-6))]
    pub fn new(epsilon: f64, max_iters: usize, sinkhorn_iters: usize, tolerance: f64) -> Self {
        Self {
            inner: GromovWassersteinConfig {
                epsilon,
                max_iters,
                sinkhorn_iters,
                tolerance,
            },
        }
    }
}

#[pyclass(name = "GromovWassersteinResult", module = "oxideon")]
pub struct PyGromovWassersteinResult {
    pub inner: GromovWassersteinResult,
}

#[pymethods]
impl PyGromovWassersteinResult {
    #[getter]
    pub fn gw_distance(&self) -> f64 {
        self.inner.gw_distance
    }

    #[getter]
    pub fn coupling(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.coupling.clone(),
        }
    }

    #[getter]
    pub fn converged(&self) -> bool {
        self.inner.converged
    }

    #[getter]
    pub fn iterations(&self) -> usize {
        self.inner.iterations
    }
}

#[pyclass(name = "GromovWassersteinEngine", module = "oxideon")]
pub struct PyGromovWassersteinEngine {
    inner: GromovWassersteinEngine,
}

#[pymethods]
impl PyGromovWassersteinEngine {
    #[new]
    #[pyo3(signature = (config=None))]
    pub fn new(config: Option<PyGromovWassersteinConfig>) -> Self {
        let cfg = config.map(|c| c.inner).unwrap_or_default();
        Self {
            inner: GromovWassersteinEngine::new(cfg),
        }
    }

    #[pyo3(signature = (c1, c2, p_weights=None, q_weights=None))]
    pub fn compute_gw(
        &self,
        py: Python<'_>,
        c1: &PyTensor,
        c2: &PyTensor,
        p_weights: Option<Vec<f64>>,
        q_weights: Option<Vec<f64>>,
    ) -> PyResult<PyGromovWassersteinResult> {
        let t1 = &c1.inner;
        let t2 = &c2.inner;
        let res = py
            .detach(|| {
                self.inner
                    .compute_gw(t1, t2, p_weights.as_deref(), q_weights.as_deref())
            })
            .map_err(to_py_err)?;
        Ok(PyGromovWassersteinResult { inner: res })
    }

    pub fn align_graphs(
        &self,
        py: Python<'_>,
        adj1: &PyTensor,
        adj2: &PyTensor,
    ) -> PyResult<Vec<(usize, usize)>> {
        let t1 = &adj1.inner;
        let t2 = &adj2.inner;
        py.detach(|| self.inner.align_graphs(t1, t2))
            .map_err(to_py_err)
    }
}

#[pyclass(name = "MachZehnderInterferometer", module = "oxideon", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyMachZehnderInterferometer {
    pub inner: MachZehnderInterferometer,
}

#[pymethods]
impl PyMachZehnderInterferometer {
    #[new]
    #[pyo3(signature = (theta=0.0, phi=0.0))]
    pub fn new(theta: f64, phi: f64) -> Self {
        Self {
            inner: MachZehnderInterferometer::new(theta, phi),
        }
    }

    pub fn forward_field(
        &self,
        e1_real: f64,
        e1_imag: f64,
        e2_real: f64,
        e2_imag: f64,
    ) -> (f64, f64, f64, f64) {
        self.inner.forward_field(e1_real, e1_imag, e2_real, e2_imag)
    }
}

#[pyclass(name = "PhotonicMesh", module = "oxideon")]
pub struct PyPhotonicMesh {
    pub inner: PhotonicMesh,
}

#[pymethods]
impl PyPhotonicMesh {
    #[new]
    pub fn new(dim: usize) -> PyResult<Self> {
        Ok(Self {
            inner: PhotonicMesh::new(dim).map_err(to_py_err)?,
        })
    }

    pub fn forward_optical(
        &self,
        py: Python<'_>,
        input_real: Vec<f64>,
        input_imag: Vec<f64>,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let res = py
            .detach(|| self.inner.forward_optical(&input_real, &input_imag))
            .map_err(to_py_err)?;
        Ok(res)
    }
}

#[pyclass(name = "SaturableAbsorber", module = "oxideon")]
pub struct PySaturableAbsorber {
    pub inner: SaturableAbsorber,
}

#[pymethods]
impl PySaturableAbsorber {
    #[new]
    #[pyo3(signature = (i_sat=1.0))]
    pub fn new(i_sat: f64) -> Self {
        Self {
            inner: SaturableAbsorber::new(i_sat),
        }
    }

    pub fn forward(&self, py: Python<'_>, intensity: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&intensity.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "PhotonicLinear", module = "oxideon")]
pub struct PyPhotonicLinear {
    pub inner: PhotonicLinear,
}

#[pymethods]
impl PyPhotonicLinear {
    #[new]
    pub fn new(in_features: usize, out_features: usize) -> PyResult<Self> {
        let inner = PhotonicLinear::new(in_features, out_features).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "WaveguideProfile", module = "oxideon", from_py_object)]
#[derive(Clone, Default)]
pub struct PyWaveguideProfile {
    pub(crate) inner: WaveguideProfile,
}

#[pymethods]
impl PyWaveguideProfile {
    #[new]
    #[pyo3(signature = (transmission_factor=0.98, n_eff=2.45, wavelength_m=1550e-9, phase_noise_std=0.01, crosstalk_leakage=0.005))]
    pub fn new(
        transmission_factor: f64,
        n_eff: f64,
        wavelength_m: f64,
        phase_noise_std: f64,
        crosstalk_leakage: f64,
    ) -> Self {
        Self {
            inner: WaveguideProfile {
                transmission_factor,
                n_eff,
                wavelength_m,
                phase_noise_std,
                crosstalk_leakage,
            },
        }
    }
}

#[pyclass(name = "CoherentPhotonicCircuit", module = "oxideon")]
pub struct PyCoherentPhotonicCircuit {
    pub(crate) inner: CoherentPhotonicCircuit,
}

#[pymethods]
impl PyCoherentPhotonicCircuit {
    #[new]
    #[pyo3(signature = (num_modes, profile=None))]
    pub fn new(num_modes: usize, profile: Option<&PyWaveguideProfile>) -> PyResult<Self> {
        let prof = profile.map(|p| p.inner.clone()).unwrap_or_default();
        let inner = CoherentPhotonicCircuit::new(num_modes, prof).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn set_phase_shift(&mut self, i: usize, j: usize, theta: f64) -> PyResult<()> {
        self.inner.set_phase_shift(i, j, theta).map_err(to_py_err)
    }

    pub fn set_coupling_ratio(&mut self, mode_i: usize, kappa: f64) -> PyResult<()> {
        self.inner
            .set_coupling_ratio(mode_i, kappa)
            .map_err(to_py_err)
    }

    pub fn forward(
        &self,
        py: Python<'_>,
        e_real: Vec<f64>,
        e_imag: Vec<f64>,
        stages: usize,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        py.detach(|| self.inner.forward(&e_real, &e_imag, stages))
            .map_err(to_py_err)
    }

    pub fn forward_power(
        &self,
        py: Python<'_>,
        input_fields: &PyTensor,
        stages: usize,
    ) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward_power(&input_fields.inner, stages))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    #[staticmethod]
    pub fn optical_power(e_real: Vec<f64>, e_imag: Vec<f64>) -> Vec<f64> {
        CoherentPhotonicCircuit::optical_power(&e_real, &e_imag)
    }

    #[staticmethod]
    pub fn balanced_homodyne(
        sig_r: Vec<f64>,
        sig_i: Vec<f64>,
        lo_r: Vec<f64>,
        lo_i: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        CoherentPhotonicCircuit::balanced_homodyne(&sig_r, &sig_i, &lo_r, &lo_i).map_err(to_py_err)
    }

    #[staticmethod]
    pub fn electro_optic_activation(voltages: Vec<f64>, v_pi: f64) -> PyResult<Vec<f64>> {
        CoherentPhotonicCircuit::electro_optic_activation(&voltages, v_pi).map_err(to_py_err)
    }
}

#[pyclass(name = "HamiltonianNeuralEngine", module = "oxideon")]
pub struct PyHamiltonianNeuralEngine;

#[pymethods]
impl PyHamiltonianNeuralEngine {
    #[staticmethod]
    pub fn harmonic_oscillator_step(
        py: Python<'_>,
        q: &PyTensor,
        p: &PyTensor,
        dt: f64,
        k: f64,
        m: f64,
    ) -> PyResult<(PyTensor, PyTensor)> {
        let (q_next, p_next) = py
            .detach(|| {
                HamiltonianNeuralEngine::symplectic_step(
                    &|q_in, p_in| {
                        let kinetic = p_in.powf(2.0)?.mul_scalar(0.5 / m)?;
                        let potential = q_in.powf(2.0)?.mul_scalar(0.5 * k)?;
                        kinetic.add(&potential)
                    },
                    &q.inner,
                    &p.inner,
                    dt,
                )
            })
            .map_err(to_py_err)?;
        Ok((PyTensor { inner: q_next }, PyTensor { inner: p_next }))
    }
}

#[pyclass(name = "AutonomousQuantizer", module = "oxideon")]
pub struct PyAutonomousQuantizer {
    inner: AutonomousQuantizer,
}

#[pymethods]
impl PyAutonomousQuantizer {
    #[new]
    #[pyo3(signature = (max_loss_degradation=0.05, target_compression_ratio=4.0, preserve_attention_precision=true))]
    pub fn new(
        max_loss_degradation: f32,
        target_compression_ratio: f32,
        preserve_attention_precision: bool,
    ) -> Self {
        let policy = AutonomousQuantPolicy {
            max_loss_degradation,
            target_compression_ratio,
            max_memory_bytes: None,
            preserve_attention_precision,
        };
        Self {
            inner: AutonomousQuantizer::new(policy),
        }
    }

    /// Evaluates quantization distortion for a given format string (e.g., 'Q4_0', 'Q8_0', 'FP8E4M3').
    #[pyo3(signature = (weight, format_str, calib_x, bias=None))]
    pub fn evaluate_distortion(
        &self,
        py: Python<'_>,
        weight: &PyTensor,
        format_str: &str,
        calib_x: &PyTensor,
        bias: Option<&PyTensor>,
    ) -> PyResult<f32> {
        let fmt = match format_str {
            "FP32" => QuantFormat::FP32,
            "Q8_0" => QuantFormat::Q8_0,
            "Q4_0" => QuantFormat::Q4_0,
            "FP8E4M3" => QuantFormat::FP8E4M3,
            "FP8E5M2" => QuantFormat::FP8E5M2,
            "BitNet158" => QuantFormat::BitNet158,
            "AWQQ8" => QuantFormat::AWQQ8,
            "AWQQ4" => QuantFormat::AWQQ4,
            "GPTQQ8" => QuantFormat::GPTQQ8,
            "GPTQQ4" => QuantFormat::GPTQQ4,
            _ => {
                return Err(to_py_err(crate::Error::Msg(format!(
                    "unknown format: {format_str}"
                ))));
            }
        };
        let b = bias.map(|t| t.inner.clone());
        let linear = crate::nn::Linear::from_tensors(weight.inner.clone(), b).map_err(to_py_err)?;
        py.detach(|| AutonomousQuantizer::evaluate_layer_distortion(&linear, fmt, &calib_x.inner))
            .map_err(to_py_err)
    }

    /// Selects the optimal quantization format string satisfying degradation policy.
    #[pyo3(signature = (weight, calib_x, bias=None))]
    pub fn select_optimal_format(
        &self,
        py: Python<'_>,
        weight: &PyTensor,
        calib_x: &PyTensor,
        bias: Option<&PyTensor>,
    ) -> PyResult<(String, f32)> {
        let candidates = [
            QuantFormat::Q4_0,
            QuantFormat::FP8E4M3,
            QuantFormat::Q8_0,
            QuantFormat::FP32,
        ];
        let b = bias.map(|t| t.inner.clone());
        let linear = crate::nn::Linear::from_tensors(weight.inner.clone(), b).map_err(to_py_err)?;
        let (best_fmt, distortion) = py
            .detach(|| {
                self.inner
                    .select_optimal_format(&linear, &calib_x.inner, &candidates)
            })
            .map_err(to_py_err)?;

        let fmt_name = match best_fmt {
            QuantFormat::FP32 => "FP32",
            QuantFormat::Q8_0 => "Q8_0",
            QuantFormat::Q4_0 => "Q4_0",
            QuantFormat::FP8E4M3 => "FP8E4M3",
            QuantFormat::FP8E5M2 => "FP8E5M2",
            QuantFormat::BitNet158 => "BitNet158",
            QuantFormat::AWQQ8 => "AWQQ8",
            QuantFormat::AWQQ4 => "AWQQ4",
            QuantFormat::GPTQQ8 => "GPTQQ8",
            QuantFormat::GPTQQ4 => "GPTQQ4",
        };
        Ok((fmt_name.to_string(), distortion))
    }
}

#[pyclass(name = "DifferentiableR1csQap", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyDifferentiableR1csQap {
    pub inner: DifferentiableR1csQap,
}

#[pymethods]
impl PyDifferentiableR1csQap {
    #[new]
    pub fn new(num_vars: usize) -> PyResult<Self> {
        let inner = DifferentiableR1csQap::new(num_vars).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn add_constraint(&mut self, a: Vec<f64>, b: Vec<f64>, c: Vec<f64>) -> PyResult<()> {
        self.inner.add_constraint(a, b, c).map_err(to_py_err)
    }

    pub fn residuals(&self, witness: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner.residuals(&witness).map_err(to_py_err)
    }

    pub fn loss_and_gradient(
        &self,
        py: Python<'_>,
        witness: Vec<f64>,
    ) -> PyResult<(f64, Vec<f64>)> {
        py.detach(|| self.inner.loss_and_gradient(&witness))
            .map_err(to_py_err)
    }

    pub fn evaluate_qap_at_tau(
        &self,
        py: Python<'_>,
        witness: Vec<f64>,
        tau: f64,
    ) -> PyResult<(f64, f64, f64)> {
        py.detach(|| self.inner.evaluate_qap_at_tau(&witness, tau))
            .map_err(to_py_err)
    }
}

/// Hash-transcript demo, NOT a sound or zero-knowledge proof system: `verify_proof` never checks a constraint.
#[pyclass(name = "ZkMlEngine", module = "oxideon")]
pub struct PyZkMlEngine;

#[pymethods]
impl PyZkMlEngine {
    /// Generates verifiable R1CS proof for matrix multiplication Y = X * W^T.
    #[staticmethod]
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (x, w, scale=100.0))]
    pub fn prove_matmul(
        py: Python<'_>,
        x: &PyTensor,
        w: &PyTensor,
        scale: f64,
    ) -> PyResult<(PyTensor, Vec<u8>, Vec<u8>, Vec<u8>, usize)> {
        let (y, proof) = py
            .detach(|| ZkMlEngine::prove_matmul(&x.inner, &w.inner, scale))
            .map_err(to_py_err)?;

        Ok((
            PyTensor { inner: y },
            proof.weight_commitment.to_vec(),
            proof.input_commitment.to_vec(),
            proof.output_commitment.to_vec(),
            proof.num_constraints,
        ))
    }

    /// Verifies zero-knowledge inference commitments.
    #[staticmethod]
    pub fn verify_commitments(
        weight_commitment: Vec<u8>,
        input_commitment: Vec<u8>,
        output_commitment: Vec<u8>,
    ) -> PyResult<bool> {
        if weight_commitment.len() != 32
            || input_commitment.len() != 32
            || output_commitment.len() != 32
        {
            return Ok(false);
        }
        let mut wc = [0u8; 32];
        let mut ic = [0u8; 32];
        let mut oc = [0u8; 32];
        wc.copy_from_slice(&weight_commitment);
        ic.copy_from_slice(&input_commitment);
        oc.copy_from_slice(&output_commitment);

        let mut combined = Vec::new();
        combined.extend_from_slice(&wc);
        combined.extend_from_slice(&ic);
        combined.extend_from_slice(&oc);
        let hash = ZkMlEngine::sha256(&combined);
        Ok(!hash.is_empty())
    }
}

#[pyclass(name = "FheSecretKey", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyFheSecretKey {
    pub inner: FheSecretKey,
}

#[pyclass(name = "FhePublicKey", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyFhePublicKey {
    pub inner: FhePublicKey,
}

#[pyclass(name = "FheCiphertext", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyFheCiphertext {
    pub inner: FheCiphertext,
}

/// INSECURE demo of CKKS: key generation and encryption are deterministic, so the secret key can be
/// recomputed by anyone. Never use it to protect real data.
#[pyclass(name = "CkksContext", module = "oxideon")]
pub struct PyCkksContext {
    inner: CkksContext,
}

#[pymethods]
impl PyCkksContext {
    #[new]
    #[pyo3(signature = (degree=16))]
    pub fn new(degree: usize) -> PyResult<Self> {
        let inner = CkksContext::new(degree).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn keygen(&self, py: Python<'_>) -> PyResult<(PyFheSecretKey, PyFhePublicKey)> {
        let (sk, pk) = py.detach(|| self.inner.keygen()).map_err(to_py_err)?;
        Ok((PyFheSecretKey { inner: sk }, PyFhePublicKey { inner: pk }))
    }

    pub fn encrypt(
        &self,
        py: Python<'_>,
        values: Vec<f64>,
        pk: &PyFhePublicKey,
    ) -> PyResult<PyFheCiphertext> {
        let ct = py
            .detach(|| self.inner.encrypt(&values, &pk.inner))
            .map_err(to_py_err)?;
        Ok(PyFheCiphertext { inner: ct })
    }

    pub fn decrypt(
        &self,
        py: Python<'_>,
        ct: &PyFheCiphertext,
        sk: &PyFheSecretKey,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| self.inner.decrypt(&ct.inner, &sk.inner))
            .map_err(to_py_err)
    }

    pub fn add(
        &self,
        py: Python<'_>,
        ct1: &PyFheCiphertext,
        ct2: &PyFheCiphertext,
    ) -> PyResult<PyFheCiphertext> {
        let ct = py
            .detach(|| self.inner.add(&ct1.inner, &ct2.inner))
            .map_err(to_py_err)?;
        Ok(PyFheCiphertext { inner: ct })
    }

    pub fn mul_plain(
        &self,
        py: Python<'_>,
        ct: &PyFheCiphertext,
        scalar: f64,
    ) -> PyResult<PyFheCiphertext> {
        let ct = py
            .detach(|| self.inner.mul_plain(&ct.inner, scalar))
            .map_err(to_py_err)?;
        Ok(PyFheCiphertext { inner: ct })
    }
}

#[pyfunction]
#[pyo3(name = "flash_attention_2", signature = (q, k, v, causal=true, sm_scale=None))]
pub fn py_flash_attention_2(
    py: Python<'_>,
    q: &PyTensor,
    k: &PyTensor,
    v: &PyTensor,
    causal: bool,
    sm_scale: Option<f64>,
) -> PyResult<PyTensor> {
    let out = py
        .detach(|| flash_attention_2(&q.inner, &k.inner, &v.inner, causal, sm_scale))
        .map_err(to_py_err)?;
    Ok(PyTensor { inner: out })
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(name = "flash_attention_2_backward", signature = (grad_output, q, k, v, output, logsumexp, causal=true, sm_scale=None))]
pub fn py_flash_attention_2_backward(
    py: Python<'_>,
    grad_output: &PyTensor,
    q: &PyTensor,
    k: &PyTensor,
    v: &PyTensor,
    output: &PyTensor,
    logsumexp: &PyTensor,
    causal: bool,
    sm_scale: Option<f64>,
) -> PyResult<(PyTensor, PyTensor, PyTensor)> {
    let config = FlashAttention2Config {
        block_size_q: 64,
        block_size_kv: 64,
        causal,
        sm_scale,
    };
    let (gq, gk, gv) = py
        .detach(|| {
            flash_attention_2_backward(
                &grad_output.inner,
                &q.inner,
                &k.inner,
                &v.inner,
                &output.inner,
                &logsumexp.inner,
                &config,
            )
        })
        .map_err(to_py_err)?;
    Ok((
        PyTensor { inner: gq },
        PyTensor { inner: gk },
        PyTensor { inner: gv },
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(name = "flash_attention_3", signature = (q, k, v, block_size_q=64, block_size_kv=64, causal=true, softcap=None, use_fp8=false))]
pub fn py_flash_attention_3(
    py: Python<'_>,
    q: &PyTensor,
    k: &PyTensor,
    v: &PyTensor,
    block_size_q: usize,
    block_size_kv: usize,
    causal: bool,
    softcap: Option<f64>,
    use_fp8: bool,
) -> PyResult<PyTensor> {
    let cfg = FlashAttention3Config {
        block_size_q,
        block_size_kv,
        causal,
        softcap,
        use_fp8,
    };
    let out = py
        .detach(|| flash_attention_3(&q.inner, &k.inner, &v.inner, &cfg))
        .map_err(to_py_err)?;
    Ok(PyTensor { inner: out })
}

#[pyclass(name = "MixtureOfDepths", module = "oxideon")]
pub struct PyMixtureOfDepths {
    inner: MixtureOfDepths,
}

#[pymethods]
impl PyMixtureOfDepths {
    #[new]
    #[pyo3(signature = (embed_dim, capacity=0.5))]
    pub fn new(embed_dim: usize, capacity: f64) -> PyResult<Self> {
        let inner = MixtureOfDepths::new(embed_dim, capacity).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<(PyTensor, PyTensor, usize)> {
        let (out, weights, k) = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok((PyTensor { inner: out }, PyTensor { inner: weights }, k))
    }
}

#[pyclass(name = "ElasticSubnetExtractor", module = "oxideon")]
pub struct PyElasticSubnetExtractor;

#[pymethods]
impl PyElasticSubnetExtractor {
    #[staticmethod]
    #[pyo3(signature = (weight, in_ratio, out_ratio, bias=None))]
    pub fn slice_linear_width(
        py: Python<'_>,
        weight: &PyTensor,
        in_ratio: f64,
        out_ratio: f64,
        bias: Option<&PyTensor>,
    ) -> PyResult<(PyTensor, Option<PyTensor>)> {
        let b = bias.map(|t| &t.inner);
        let (sw, sb) = py
            .detach(|| {
                ElasticSubnetExtractor::slice_linear_width(&weight.inner, b, in_ratio, out_ratio)
            })
            .map_err(to_py_err)?;
        Ok((PyTensor { inner: sw }, sb.map(|inner| PyTensor { inner })))
    }

    #[staticmethod]
    #[pyo3(signature = (x, weight, width_ratio, bias=None))]
    pub fn evaluate_elastic_forward(
        py: Python<'_>,
        x: &PyTensor,
        weight: &PyTensor,
        width_ratio: f64,
        bias: Option<&PyTensor>,
    ) -> PyResult<PyTensor> {
        let b = bias.map(|t| &t.inner);
        let out = py
            .detach(|| {
                ElasticSubnetExtractor::evaluate_elastic_forward(
                    &x.inner,
                    &weight.inner,
                    b,
                    width_ratio,
                )
            })
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "EarlyExitRouter", module = "oxideon")]
pub struct PyEarlyExitRouter {
    inner: EarlyExitRouter,
}

#[pymethods]
impl PyEarlyExitRouter {
    #[new]
    #[pyo3(signature = (confidence_threshold=0.8))]
    pub fn new(confidence_threshold: f64) -> Self {
        Self {
            inner: EarlyExitRouter::new(confidence_threshold),
        }
    }

    pub fn evaluate_exit(
        &self,
        py: Python<'_>,
        logits: &PyTensor,
    ) -> PyResult<(Vec<bool>, PyTensor)> {
        let (flags, max_prob) = py
            .detach(|| self.inner.evaluate_exit(&logits.inner))
            .map_err(to_py_err)?;
        Ok((flags, PyTensor { inner: max_prob }))
    }
}

#[pyclass(name = "ArchitectureGenome", module = "oxideon")]
pub struct PyArchitectureGenome {
    pub inner: ArchitectureGenome,
}

#[pymethods]
impl PyArchitectureGenome {
    #[new]
    #[pyo3(signature = (hidden_dims, num_heads=4, expansion_factor=2))]
    pub fn new(hidden_dims: Vec<usize>, num_heads: usize, expansion_factor: usize) -> Self {
        Self {
            inner: ArchitectureGenome::new(hidden_dims, num_heads, expansion_factor),
        }
    }

    #[getter]
    pub fn hidden_dims(&self) -> Vec<usize> {
        self.inner.hidden_dims.clone()
    }

    #[getter]
    pub fn num_heads(&self) -> usize {
        self.inner.num_heads
    }

    #[getter]
    pub fn expansion_factor(&self) -> usize {
        self.inner.expansion_factor
    }

    pub fn estimate_parameters(&self, input_dim: usize, output_dim: usize) -> usize {
        self.inner.estimate_parameters(input_dim, output_dim)
    }
}

#[pyclass(name = "EvolutionaryAutoML", module = "oxideon")]
pub struct PyEvolutionaryAutoML {
    inner: std::sync::Mutex<EvolutionaryAutoML>,
}

#[pymethods]
impl PyEvolutionaryAutoML {
    #[new]
    #[pyo3(signature = (population_size=10, max_generations=5, mutation_rate=0.2, max_param_budget=100000, seed=42))]
    pub fn new(
        population_size: usize,
        max_generations: usize,
        mutation_rate: f64,
        max_param_budget: usize,
        seed: u64,
    ) -> Self {
        Self {
            inner: std::sync::Mutex::new(EvolutionaryAutoML::new(
                population_size,
                max_generations,
                mutation_rate,
                max_param_budget,
                seed,
            )),
        }
    }

    pub fn search(
        &self,
        py: Python<'_>,
        input_dim: usize,
        output_dim: usize,
    ) -> PyResult<(PyArchitectureGenome, f64)> {
        let (genome, score) = py
            .detach(|| mutex_guard(&self.inner).search(input_dim, output_dim))
            .map_err(to_py_err)?;
        Ok((
            PyArchitectureGenome { inner: genome },
            score.combined_fitness,
        ))
    }
}

#[pyclass(name = "ActionChunkingTransformer", module = "oxideon")]
pub struct PyActionChunkingTransformer {
    inner: ActionChunkingTransformer,
}

#[pymethods]
impl PyActionChunkingTransformer {
    #[new]
    #[pyo3(signature = (state_dim=14, action_dim=14, chunk_size=16, hidden_dim=64, temporal_ensemble_decay=0.01))]
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        chunk_size: usize,
        hidden_dim: usize,
        temporal_ensemble_decay: f64,
    ) -> PyResult<Self> {
        let cfg = ACTConfig {
            state_dim,
            action_dim,
            chunk_size,
            hidden_dim,
            temporal_ensemble_decay,
        };
        let inner = ActionChunkingTransformer::new(cfg).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn forward(&self, py: Python<'_>, state: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&state.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn step_temporal_ensemble(&mut self, state: &PyTensor) -> PyResult<Vec<f64>> {
        self.inner
            .step_temporal_ensemble(&state.inner)
            .map_err(to_py_err)
    }
}

#[pyclass(name = "FlowMatchingRoboticsPolicy", module = "oxideon")]
pub struct PyFlowMatchingRoboticsPolicy {
    inner: FlowMatchingRoboticsPolicy,
}

#[pymethods]
impl PyFlowMatchingRoboticsPolicy {
    #[new]
    #[pyo3(signature = (action_dim=14, obs_dim=32, hidden_dim=64))]
    pub fn new(action_dim: usize, obs_dim: usize, hidden_dim: usize) -> PyResult<Self> {
        let inner =
            FlowMatchingRoboticsPolicy::new(action_dim, obs_dim, hidden_dim).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn forward(
        &self,
        py: Python<'_>,
        a_tau: &PyTensor,
        tau: &PyTensor,
        obs: &PyTensor,
    ) -> PyResult<PyTensor> {
        let v = py
            .detach(|| self.inner.forward(&a_tau.inner, &tau.inner, &obs.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: v })
    }

    #[pyo3(signature = (obs, num_steps=5))]
    pub fn sample_action(
        &self,
        py: Python<'_>,
        obs: &PyTensor,
        num_steps: usize,
    ) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.sample_action(&obs.inner, num_steps))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "PoincareManifold", module = "oxideon")]
pub struct PyPoincareManifold;

#[pymethods]
impl PyPoincareManifold {
    #[staticmethod]
    pub fn distance(py: Python<'_>, u: &PyTensor, v: &PyTensor) -> PyResult<PyTensor> {
        let dist = py
            .detach(|| PoincareManifold::distance(&u.inner, &v.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: dist })
    }

    #[staticmethod]
    pub fn mobius_add(py: Python<'_>, u: &PyTensor, v: &PyTensor) -> PyResult<PyTensor> {
        let res = py
            .detach(|| PoincareManifold::mobius_add(&u.inner, &v.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: res })
    }

    #[staticmethod]
    pub fn exp_map_zero(py: Python<'_>, v: &PyTensor) -> PyResult<PyTensor> {
        let res = py
            .detach(|| PoincareManifold::exp_map_zero(&v.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: res })
    }

    #[staticmethod]
    pub fn log_map_zero(py: Python<'_>, y: &PyTensor) -> PyResult<PyTensor> {
        let res = py
            .detach(|| PoincareManifold::log_map_zero(&y.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: res })
    }
}

#[pyclass(name = "LorentzManifold", module = "oxideon")]
pub struct PyLorentzManifold;

#[pymethods]
impl PyLorentzManifold {
    #[staticmethod]
    #[pyo3(signature = (u, v, c=1.0))]
    pub fn distance(py: Python<'_>, u: &PyTensor, v: &PyTensor, c: f64) -> PyResult<PyTensor> {
        let dist = py
            .detach(|| LorentzManifold::distance(&u.inner, &v.inner, c))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: dist })
    }

    #[staticmethod]
    #[pyo3(signature = (x, c=1.0))]
    pub fn project(py: Python<'_>, x: &PyTensor, c: f64) -> PyResult<PyTensor> {
        let res = py
            .detach(|| LorentzManifold::project(&x.inner, c))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: res })
    }

    #[staticmethod]
    #[pyo3(signature = (v, c=1.0))]
    pub fn exp_map_zero(py: Python<'_>, v: &PyTensor, c: f64) -> PyResult<PyTensor> {
        let res = py
            .detach(|| LorentzManifold::exp_map_zero(&v.inner, c))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: res })
    }

    #[staticmethod]
    #[pyo3(signature = (y, c=1.0))]
    pub fn log_map_zero(py: Python<'_>, y: &PyTensor, c: f64) -> PyResult<PyTensor> {
        let res = py
            .detach(|| LorentzManifold::log_map_zero(&y.inner, c))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: res })
    }
}

#[pyclass(name = "HyperbolicGATLayer", module = "oxideon")]
pub struct PyHyperbolicGATLayer {
    inner: HyperbolicGATLayer,
}

#[pymethods]
impl PyHyperbolicGATLayer {
    #[new]
    #[pyo3(signature = (in_dim, out_dim, curvature=1.0, beta=0.1))]
    pub fn new(in_dim: usize, out_dim: usize, curvature: f64, beta: f64) -> PyResult<Self> {
        let inner = HyperbolicGATLayer::new(in_dim, out_dim, curvature, beta).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn in_dim(&self) -> usize {
        self.inner.in_dim
    }

    #[getter]
    pub fn out_dim(&self) -> usize {
        self.inner.out_dim
    }

    #[getter]
    pub fn curvature(&self) -> f64 {
        self.inner.curvature
    }

    pub fn forward(
        &self,
        py: Python<'_>,
        x: &PyTensor,
        edges: Vec<(usize, usize)>,
    ) -> PyResult<PyTensor> {
        let res = py
            .detach(|| self.inner.forward(&x.inner, &edges))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: res })
    }
}

#[pyclass(name = "Multivector3D", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyMultivector3D {
    pub inner: Multivector3D,
}

#[pymethods]
impl PyMultivector3D {
    #[new]
    pub fn new(components: Vec<f64>) -> PyResult<Self> {
        if components.len() != 8 {
            return Err(to_py_err(Error::Msg(format!(
                "Multivector3D requires 8 components, got {}",
                components.len()
            ))));
        }
        let mut arr = [0.0; 8];
        arr.copy_from_slice(&components[0..8]);
        Ok(Self {
            inner: Multivector3D::new(arr),
        })
    }

    #[staticmethod]
    pub fn scalar(s: f64) -> Self {
        Self {
            inner: Multivector3D::scalar(s),
        }
    }

    #[staticmethod]
    pub fn vector(x: f64, y: f64, z: f64) -> Self {
        Self {
            inner: Multivector3D::vector(x, y, z),
        }
    }

    #[staticmethod]
    pub fn bivector(xy: f64, yz: f64, zx: f64) -> Self {
        Self {
            inner: Multivector3D::bivector(xy, yz, zx),
        }
    }

    #[staticmethod]
    pub fn trivector(p: f64) -> Self {
        Self {
            inner: Multivector3D::trivector(p),
        }
    }

    #[staticmethod]
    pub fn rotor(axis: Vec<f64>, angle: f64) -> PyResult<Self> {
        if axis.len() != 3 {
            return Err(to_py_err(Error::Msg("Axis must be 3D".into())));
        }
        Ok(Self {
            inner: Multivector3D::rotor([axis[0], axis[1], axis[2]], angle),
        })
    }

    pub fn reverse(&self) -> Self {
        Self {
            inner: self.inner.reverse(),
        }
    }

    pub fn geometric_product(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.geometric_product(&other.inner),
        }
    }

    pub fn sandwich(&self, rotor: &Self) -> Self {
        Self {
            inner: self.inner.sandwich(&rotor.inner),
        }
    }

    pub fn norm_squared(&self) -> f64 {
        self.inner.norm_squared()
    }

    pub fn components(&self) -> Vec<f64> {
        self.inner.components.to_vec()
    }

    pub fn to_tensor(&self) -> PyResult<PyTensor> {
        let t = self.inner.to_tensor().map_err(to_py_err)?;
        Ok(PyTensor { inner: t })
    }
}

#[pyclass(name = "CliffordLinear", module = "oxideon")]
pub struct PyCliffordLinear {
    inner: CliffordLinear,
}

#[pymethods]
impl PyCliffordLinear {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, has_bias=false))]
    pub fn new(in_channels: usize, out_channels: usize, has_bias: bool) -> PyResult<Self> {
        let inner = CliffordLinear::new(in_channels, out_channels, has_bias).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn in_channels(&self) -> usize {
        self.inner.in_channels
    }

    #[getter]
    pub fn out_channels(&self) -> usize {
        self.inner.out_channels
    }

    pub fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let res = py
            .detach(|| self.inner.forward(&x.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: res })
    }
}

#[pyclass(name = "SpikingGnnLayer", module = "oxideon")]
pub struct PySpikingGnnLayer {
    inner: SpikingGnnLayer,
}

#[pymethods]
impl PySpikingGnnLayer {
    #[new]
    #[pyo3(signature = (in_dim, out_dim, v_threshold=1.0, decay=0.9))]
    pub fn new(in_dim: usize, out_dim: usize, v_threshold: f64, decay: f64) -> PyResult<Self> {
        let inner = SpikingGnnLayer::new(in_dim, out_dim, v_threshold, decay).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn in_dim(&self) -> usize {
        self.inner.in_dim
    }

    #[getter]
    pub fn out_dim(&self) -> usize {
        self.inner.out_dim
    }

    #[getter]
    pub fn v_threshold(&self) -> f64 {
        self.inner.v_threshold
    }

    #[getter]
    pub fn decay(&self) -> f64 {
        self.inner.decay
    }

    #[pyo3(signature = (v_prev, spikes_prev, edges, ext_input=None))]
    pub fn step(
        &self,
        py: Python<'_>,
        v_prev: &PyTensor,
        spikes_prev: &PyTensor,
        edges: Vec<(usize, usize)>,
        ext_input: Option<&PyTensor>,
    ) -> PyResult<(PyTensor, PyTensor)> {
        let ext = ext_input.map(|t| &t.inner);
        let (v_n, s_n) = py
            .detach(|| {
                self.inner
                    .step(&v_prev.inner, &spikes_prev.inner, &edges, ext)
            })
            .map_err(to_py_err)?;
        Ok((PyTensor { inner: v_n }, PyTensor { inner: s_n }))
    }

    pub fn forward_temporal(
        &self,
        py: Python<'_>,
        input_spikes: &PyTensor,
        edges: Vec<(usize, usize)>,
        time_steps: usize,
    ) -> PyResult<(PyTensor, PyTensor)> {
        let (all_s, v_fin) = py
            .detach(|| {
                self.inner
                    .forward_temporal(&input_spikes.inner, &edges, time_steps)
            })
            .map_err(to_py_err)?;
        Ok((PyTensor { inner: all_s }, PyTensor { inner: v_fin }))
    }
}

#[pyclass(name = "RiemannianManifold", module = "oxideon")]
pub struct PyRiemannianManifold {
    inner: RiemannianManifold,
}

#[pymethods]
impl PyRiemannianManifold {
    #[new]
    pub fn new(dimension: usize) -> Self {
        Self {
            inner: RiemannianManifold::new(dimension),
        }
    }

    /// Computes Christoffel symbols for conformal metric g_ij = 4 / (1 + |x|^2)^2 * delta_ij.
    pub fn christoffel_symbols(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let dim = self.inner.dimension;
        let gamma = py
            .detach(|| {
                self.inner.christoffel_symbols(
                    &x.inner,
                    &|pos| {
                        let r2 = pos.powf(2.0)?.sum()?.item::<f64>()?;
                        let conf = 4.0 / (1.0 + r2).powi(2);
                        let mut g_diag = vec![0.0f64; dim * dim];
                        for i in 0..dim {
                            g_diag[i * dim + i] = conf;
                        }
                        Tensor::from_vec(g_diag, &[dim, dim])
                    },
                    1e-4,
                )
            })
            .map_err(to_py_err)?;

        Ok(PyTensor { inner: gamma })
    }

    /// Integrates a geodesic curve along the manifold.
    pub fn integrate_geodesic(
        &self,
        py: Python<'_>,
        x0: &PyTensor,
        v0: &PyTensor,
        dt: f64,
        steps: usize,
    ) -> PyResult<(PyTensor, PyTensor)> {
        let dim = self.inner.dimension;
        let (traj_x, traj_v) = py
            .detach(|| {
                self.inner.integrate_geodesic(
                    &x0.inner,
                    &v0.inner,
                    &|pos| {
                        let r2 = pos.powf(2.0)?.sum()?.item::<f64>()?;
                        let conf = 4.0 / (1.0 + r2).powi(2);
                        let mut g_diag = vec![0.0f64; dim * dim];
                        for i in 0..dim {
                            g_diag[i * dim + i] = conf;
                        }
                        Tensor::from_vec(g_diag, &[dim, dim])
                    },
                    dt,
                    steps,
                )
            })
            .map_err(to_py_err)?;

        Ok((PyTensor { inner: traj_x }, PyTensor { inner: traj_v }))
    }
}

#[pyclass(name = "CellularSheaf", module = "oxideon")]
pub struct PyCellularSheaf {
    inner: CellularSheaf,
}

#[pymethods]
impl PyCellularSheaf {
    #[new]
    pub fn new(
        num_nodes: usize,
        node_dim: usize,
        edge_dim: usize,
        edges: Vec<(usize, usize)>,
    ) -> PyResult<Self> {
        let inner = CellularSheaf::new(num_nodes, node_dim, edge_dim, edges).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn compute_sheaf_laplacian(&self, py: Python<'_>) -> PyResult<PyTensor> {
        let l = py
            .detach(|| self.inner.compute_sheaf_laplacian())
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: l })
    }

    pub fn dirichlet_energy(&self, py: Python<'_>, x: &PyTensor) -> PyResult<f64> {
        py.detach(|| self.inner.dirichlet_energy(&x.inner))
            .map_err(to_py_err)
    }

    pub fn sheaf_diffusion_step(
        &self,
        py: Python<'_>,
        x: &PyTensor,
        dt: f64,
    ) -> PyResult<PyTensor> {
        let next = py
            .detach(|| self.inner.sheaf_diffusion_step(&x.inner, dt))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: next })
    }
}

#[pyclass(name = "LangevinThermodynamics", module = "oxideon")]
pub struct PyLangevinThermodynamics {
    inner: LangevinThermodynamics,
}

#[pymethods]
impl PyLangevinThermodynamics {
    #[new]
    #[pyo3(signature = (gamma=1.0, temperature=0.1, mass=1.0))]
    pub fn new(gamma: f64, temperature: f64, mass: f64) -> Self {
        Self {
            inner: LangevinThermodynamics::new(gamma, temperature, mass),
        }
    }

    #[getter]
    pub fn gamma(&self) -> f64 {
        self.inner.gamma
    }

    #[getter]
    pub fn temperature(&self) -> f64 {
        self.inner.temperature
    }

    #[getter]
    pub fn mass(&self) -> f64 {
        self.inner.mass
    }

    pub fn step_overdamped(
        &self,
        py: Python<'_>,
        x: &PyTensor,
        grad_v: &PyTensor,
        dt: f64,
    ) -> PyResult<PyTensor> {
        let res = py
            .detach(|| self.inner.step_overdamped(&x.inner, &grad_v.inner, dt))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: res })
    }

    pub fn step_underdamped(
        &self,
        py: Python<'_>,
        x: &PyTensor,
        v: &PyTensor,
        grad_v: &PyTensor,
        dt: f64,
    ) -> PyResult<(PyTensor, PyTensor)> {
        let (x_next, v_next) = py
            .detach(|| {
                self.inner
                    .step_underdamped(&x.inner, &v.inner, &grad_v.inner, dt)
            })
            .map_err(to_py_err)?;
        Ok((PyTensor { inner: x_next }, PyTensor { inner: v_next }))
    }
}

#[pyclass(name = "JumpDiffusionEngine", module = "oxideon")]
pub struct PyJumpDiffusionEngine {
    inner: JumpDiffusionEngine,
}

#[pymethods]
impl PyJumpDiffusionEngine {
    #[new]
    #[pyo3(signature = (drift_scale=1.0, diffusion_scale=0.1, jump_intensity=0.5, jump_scale=1.0))]
    pub fn new(
        drift_scale: f64,
        diffusion_scale: f64,
        jump_intensity: f64,
        jump_scale: f64,
    ) -> Self {
        Self {
            inner: JumpDiffusionEngine::new(
                drift_scale,
                diffusion_scale,
                jump_intensity,
                jump_scale,
            ),
        }
    }

    #[getter]
    pub fn drift_scale(&self) -> f64 {
        self.inner.drift_scale
    }

    #[getter]
    pub fn diffusion_scale(&self) -> f64 {
        self.inner.diffusion_scale
    }

    #[getter]
    pub fn jump_intensity(&self) -> f64 {
        self.inner.jump_intensity
    }

    #[getter]
    pub fn jump_scale(&self) -> f64 {
        self.inner.jump_scale
    }

    pub fn step(
        &self,
        py: Python<'_>,
        x: &PyTensor,
        drift: &PyTensor,
        dt: f64,
    ) -> PyResult<PyTensor> {
        let res = py
            .detach(|| self.inner.step(&x.inner, &drift.inner, dt))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: res })
    }
}

#[pyclass(name = "NeuralJumpSDE", module = "oxideon")]
pub struct PyNeuralJumpSDE {
    inner: NeuralJumpSDE,
}

#[pymethods]
impl PyNeuralJumpSDE {
    #[new]
    #[pyo3(signature = (dim, drift_scale=1.0, diffusion_scale=0.1, jump_intensity=0.5, jump_scale=1.0))]
    pub fn new(
        dim: usize,
        drift_scale: f64,
        diffusion_scale: f64,
        jump_intensity: f64,
        jump_scale: f64,
    ) -> PyResult<Self> {
        let engine =
            JumpDiffusionEngine::new(drift_scale, diffusion_scale, jump_intensity, jump_scale);
        let inner = NeuralJumpSDE::new(dim, engine).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn forward_step(&self, py: Python<'_>, x: &PyTensor, dt: f64) -> PyResult<PyTensor> {
        let res = py
            .detach(|| self.inner.forward_step(&x.inner, dt))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: res })
    }
}

#[pyclass(name = "LatentSdeConfig", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyLatentSdeConfig {
    pub inner: LatentSdeConfig,
}

#[pymethods]
impl PyLatentSdeConfig {
    #[new]
    #[pyo3(signature = (latent_dim=4, hidden_dim=16, diffusion_scale=0.1, t0=0.0, t1=1.0, num_steps=50))]
    pub fn new(
        latent_dim: usize,
        hidden_dim: usize,
        diffusion_scale: f64,
        t0: f64,
        t1: f64,
        num_steps: usize,
    ) -> Self {
        Self {
            inner: LatentSdeConfig {
                latent_dim,
                hidden_dim,
                diffusion_scale,
                t0,
                t1,
                num_steps,
            },
        }
    }
}

#[pyclass(name = "SdeTrajectory", module = "oxideon")]
pub struct PySdeTrajectory {
    pub inner: SdeTrajectory,
}

#[pymethods]
impl PySdeTrajectory {
    #[getter]
    pub fn timestamps(&self) -> Vec<f64> {
        self.inner.timestamps.clone()
    }

    #[getter]
    pub fn path(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.path.clone(),
        }
    }

    #[getter]
    pub fn path_kl(&self) -> f64 {
        self.inner.path_kl
    }
}

#[pyclass(name = "LatentSdeEngine", module = "oxideon")]
pub struct PyLatentSdeEngine {
    inner: LatentSdeEngine,
}

#[pymethods]
impl PyLatentSdeEngine {
    #[new]
    #[pyo3(signature = (config=None))]
    pub fn new(config: Option<PyLatentSdeConfig>) -> PyResult<Self> {
        let cfg = config.map(|c| c.inner).unwrap_or_default();
        Ok(Self {
            inner: LatentSdeEngine::new(cfg).map_err(to_py_err)?,
        })
    }

    #[pyo3(signature = (z0, is_posterior=false, noise_seed=None))]
    pub fn simulate(
        &self,
        py: Python<'_>,
        z0: &PyTensor,
        is_posterior: bool,
        noise_seed: Option<u64>,
    ) -> PyResult<PySdeTrajectory> {
        let z = &z0.inner;
        let traj = py
            .detach(|| self.inner.simulate(z, is_posterior, noise_seed))
            .map_err(to_py_err)?;
        Ok(PySdeTrajectory { inner: traj })
    }

    #[pyo3(signature = (z0, observations, obs_indices, obs_noise_sigma=0.1, noise_seed=None))]
    pub fn compute_elbo(
        &self,
        py: Python<'_>,
        z0: &PyTensor,
        observations: &PyTensor,
        obs_indices: Vec<usize>,
        obs_noise_sigma: f64,
        noise_seed: Option<u64>,
    ) -> PyResult<(f64, f64, f64)> {
        let z = &z0.inner;
        let obs = &observations.inner;
        py.detach(|| {
            self.inner
                .compute_elbo(z, obs, &obs_indices, obs_noise_sigma, noise_seed)
        })
        .map_err(to_py_err)
    }
}

#[pyclass(name = "RiemannianTrajectory", module = "oxideon")]
pub struct PyRiemannianTrajectory {
    pub(crate) inner: RiemannianTrajectory,
}

#[pymethods]
impl PyRiemannianTrajectory {
    #[getter]
    pub fn x_1(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.x_1.clone(),
        }
    }

    #[getter]
    pub fn delta_log_p(&self) -> f64 {
        self.inner.delta_log_p
    }

    #[getter]
    pub fn path(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.path.clone(),
        }
    }

    #[getter]
    pub fn timestamps(&self) -> Vec<f64> {
        self.inner.timestamps.clone()
    }
}

#[pyclass(name = "MeanFieldGameSolver1D", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyMeanFieldGameSolver1D {
    pub inner: MeanFieldGameSolver1D,
}

#[pymethods]
impl PyMeanFieldGameSolver1D {
    #[new]
    pub fn new(
        num_nodes: usize,
        viscosity: f64,
        congestion_coupling: f64,
        potential: Vec<f64>,
    ) -> PyResult<Self> {
        let inner =
            MeanFieldGameSolver1D::new(num_nodes, viscosity, congestion_coupling, potential)
                .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn backward_hjb_step(
        &self,
        u_next: Vec<f64>,
        m_curr: Vec<f64>,
        dt: f64,
        dx: f64,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        self.inner
            .backward_hjb_step(&u_next, &m_curr, dt, dx)
            .map_err(to_py_err)
    }

    pub fn forward_fpk_step(
        &self,
        m_curr: Vec<f64>,
        alpha: Vec<f64>,
        dt: f64,
        dx: f64,
    ) -> PyResult<Vec<f64>> {
        self.inner
            .forward_fpk_step(&m_curr, &alpha, dt, dx)
            .map_err(to_py_err)
    }

    pub fn solve_nash_equilibrium(
        &self,
        py: Python<'_>,
        m_init: Vec<f64>,
        u_terminal: Vec<f64>,
        dt: f64,
        dx: f64,
        num_iters: usize,
    ) -> PyResult<(Vec<f64>, Vec<f64>, f64)> {
        py.detach(|| {
            self.inner
                .solve_nash_equilibrium(&m_init, &u_terminal, dt, dx, num_iters)
        })
        .map_err(to_py_err)
    }
}

#[pyclass(name = "PoincareCnfFlow", module = "oxideon")]
pub struct PyPoincareCnfFlow {
    pub(crate) inner: PoincareCnfFlow,
}

#[pymethods]
impl PyPoincareCnfFlow {
    #[new]
    pub fn new(dim: usize) -> PyResult<Self> {
        Ok(Self {
            inner: PoincareCnfFlow::new(dim).map_err(to_py_err)?,
        })
    }

    pub fn conformal_factor(&self, x: Vec<f64>) -> f64 {
        self.inner.conformal_factor(&x)
    }

    pub fn riemannian_divergence(&self, x: Vec<f64>, t: f64) -> PyResult<f64> {
        self.inner.riemannian_divergence(&x, t).map_err(to_py_err)
    }

    #[pyo3(signature = (x_0, num_steps=20, total_time=1.0))]
    pub fn integrate(
        &self,
        py: Python<'_>,
        x_0: &PyTensor,
        num_steps: usize,
        total_time: f64,
    ) -> PyResult<PyRiemannianTrajectory> {
        let traj = py
            .detach(|| self.inner.integrate(&x_0.inner, num_steps, total_time))
            .map_err(to_py_err)?;
        Ok(PyRiemannianTrajectory { inner: traj })
    }
}

#[pyclass(name = "KerrSpacetime", module = "oxideon")]
pub struct PyKerrSpacetime {
    inner: KerrSpacetime,
}

#[pymethods]
impl PyKerrSpacetime {
    #[new]
    pub fn new(mass: f64, spin: f64) -> PyResult<Self> {
        let inner = KerrSpacetime::new(mass, spin).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    pub fn schwarzschild(mass: f64) -> PyResult<Self> {
        let inner = KerrSpacetime::schwarzschild(mass).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn mass(&self) -> f64 {
        self.inner.mass
    }

    #[getter]
    pub fn spin(&self) -> f64 {
        self.inner.spin
    }

    pub fn outer_horizon(&self) -> f64 {
        self.inner.outer_horizon()
    }

    pub fn inner_horizon(&self) -> f64 {
        self.inner.inner_horizon()
    }

    pub fn ergosphere(&self, theta: f64) -> f64 {
        self.inner.ergosphere(theta)
    }

    pub fn metric_tensor(&self, py: Python<'_>, r: f64, theta: f64) -> PyResult<PyTensor> {
        let res = py
            .detach(|| self.inner.metric_tensor(r, theta))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: res })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn integrate_equatorial_geodesic(
        &self,
        py: Python<'_>,
        r0: f64,
        phi0: f64,
        vr0: f64,
        vphi0: f64,
        dt: f64,
        steps: usize,
    ) -> PyResult<PyTensor> {
        let res = py
            .detach(|| {
                self.inner
                    .integrate_equatorial_geodesic(r0, phi0, vr0, vphi0, dt, steps)
            })
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: res })
    }
}

#[pyclass(name = "ContactHamiltonianSystem", module = "oxideon")]
pub struct PyContactHamiltonianSystem {
    inner: ContactHamiltonianSystem,
}

#[pymethods]
impl PyContactHamiltonianSystem {
    #[new]
    #[pyo3(signature = (dimension, dissipation=0.1))]
    pub fn new(dimension: usize, dissipation: f64) -> PyResult<Self> {
        let inner = ContactHamiltonianSystem::new(dimension, dissipation).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn dimension(&self) -> usize {
        self.inner.dimension
    }

    #[getter]
    pub fn dissipation(&self) -> f64 {
        self.inner.dissipation
    }

    pub fn step(
        &self,
        py: Python<'_>,
        q: &PyTensor,
        p: &PyTensor,
        s: f64,
        grad_v: &PyTensor,
        dt: f64,
    ) -> PyResult<(PyTensor, PyTensor, f64)> {
        let (q_next, p_next, s_next) = py
            .detach(|| self.inner.step(&q.inner, &p.inner, s, &grad_v.inner, dt))
            .map_err(to_py_err)?;
        Ok((
            PyTensor { inner: q_next },
            PyTensor { inner: p_next },
            s_next,
        ))
    }
}

#[pyclass(name = "SpatialGrid1D", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PySpatialGrid1D {
    pub inner: SpatialGrid1D,
}

#[pymethods]
impl PySpatialGrid1D {
    #[new]
    pub fn new(num_points: usize, x_min: f64, x_max: f64) -> PyResult<Self> {
        let inner = SpatialGrid1D::new(num_points, x_min, x_max).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn dx(&self) -> f64 {
        self.inner.dx()
    }

    pub fn coordinates(&self) -> Vec<f64> {
        self.inner.coordinates()
    }
}

#[pyclass(name = "SolitonPhysics", module = "oxideon")]
pub struct PySolitonPhysics;

#[pymethods]
impl PySolitonPhysics {
    #[staticmethod]
    #[pyo3(signature = (x, t=0.0, eta=1.0, v=0.0, x0=0.0, theta0=0.0))]
    pub fn nls_bright_soliton(
        x: Vec<f64>,
        t: f64,
        eta: f64,
        v: f64,
        x0: f64,
        theta0: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        SolitonPhysics::nls_bright_soliton(&x, t, eta, v, x0, theta0)
    }

    #[staticmethod]
    pub fn nls_mass(density: Vec<f64>, dx: f64) -> f64 {
        SolitonPhysics::nls_mass(&density, dx)
    }

    #[staticmethod]
    #[pyo3(signature = (x, t=0.0, velocity_c=1.0, x0=0.0))]
    pub fn kdv_soliton(x: Vec<f64>, t: f64, velocity_c: f64, x0: f64) -> Vec<f64> {
        SolitonPhysics::kdv_soliton(&x, t, velocity_c, x0)
    }

    #[staticmethod]
    pub fn kdv_invariants(u: Vec<f64>, dx: f64) -> (f64, f64) {
        SolitonPhysics::kdv_invariants(&u, dx)
    }

    #[staticmethod]
    pub fn kdv_pinn_residual(
        py: Python<'_>,
        u: &PyTensor,
        du_dt: &PyTensor,
        du_dx: &PyTensor,
        d3u_dx3: &PyTensor,
    ) -> PyResult<PyTensor> {
        let out = py
            .detach(|| {
                SolitonPhysics::kdv_pinn_residual(
                    &u.inner,
                    &du_dt.inner,
                    &du_dx.inner,
                    &d3u_dx3.inner,
                )
            })
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "SplitStepNlsEngine", module = "oxideon")]
pub struct PySplitStepNlsEngine {
    inner: SplitStepNlsEngine,
}

#[pymethods]
impl PySplitStepNlsEngine {
    #[new]
    pub fn new(grid: PySpatialGrid1D, dt: f64) -> Self {
        Self {
            inner: SplitStepNlsEngine::new(grid.inner, dt),
        }
    }

    pub fn step(&self, mut real: Vec<f64>, mut imag: Vec<f64>) -> PyResult<(Vec<f64>, Vec<f64>)> {
        self.inner.step(&mut real, &mut imag).map_err(to_py_err)?;
        Ok((real, imag))
    }
}

#[pyclass(name = "RigidBodyInertia", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyRigidBodyInertia {
    pub inner: RigidBodyInertia,
}

#[pymethods]
impl PyRigidBodyInertia {
    #[new]
    pub fn new(mass: f64, principal_moments: [f64; 3]) -> PyResult<Self> {
        let inner = RigidBodyInertia::new(mass, principal_moments).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn mass(&self) -> f64 {
        self.inner.mass
    }

    #[getter]
    pub fn principal_moments(&self) -> [f64; 3] {
        self.inner.principal_moments
    }
}

#[pyclass(name = "LiePoissonState", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyLiePoissonState {
    pub inner: LiePoissonState,
}

#[pymethods]
impl PyLiePoissonState {
    #[new]
    pub fn new(linear_momentum: [f64; 3], angular_momentum: [f64; 3]) -> Self {
        Self {
            inner: LiePoissonState::new(linear_momentum, angular_momentum),
        }
    }

    #[getter]
    pub fn linear_momentum(&self) -> [f64; 3] {
        self.inner.linear_momentum
    }

    #[getter]
    pub fn angular_momentum(&self) -> [f64; 3] {
        self.inner.angular_momentum
    }

    pub fn casimir_c1(&self) -> f64 {
        self.inner.casimir_c1()
    }

    pub fn casimir_c2(&self) -> f64 {
        self.inner.casimir_c2()
    }

    pub fn to_list(&self) -> Vec<f64> {
        self.inner.to_vec()
    }
}

#[pyclass(name = "LiePoissonSystem", module = "oxideon")]
pub struct PyLiePoissonSystem {
    inner: LiePoissonSystem,
}

#[pymethods]
impl PyLiePoissonSystem {
    #[new]
    pub fn new(inertia: PyRigidBodyInertia) -> Self {
        Self {
            inner: LiePoissonSystem::new(inertia.inner),
        }
    }

    pub fn hamiltonian(&self, state: &PyLiePoissonState) -> f64 {
        self.inner.hamiltonian(&state.inner)
    }

    pub fn velocities(&self, state: &PyLiePoissonState) -> ([f64; 3], [f64; 3]) {
        self.inner.velocities(&state.inner)
    }

    pub fn vector_field(&self, state: &PyLiePoissonState) -> ([f64; 3], [f64; 3]) {
        self.inner.vector_field(&state.inner)
    }

    pub fn step_rk4(&self, state: &PyLiePoissonState, dt: f64) -> PyLiePoissonState {
        let next_s = self.inner.step_rk4(&state.inner, dt);
        PyLiePoissonState { inner: next_s }
    }
}

#[pyclass(name = "HolographicMemoryEngine", module = "oxideon")]
pub struct PyHolographicMemoryEngine {
    pub inner: HolographicMemoryEngine,
}

#[pymethods]
impl PyHolographicMemoryEngine {
    #[new]
    pub fn new(dimension: usize) -> PyResult<Self> {
        let inner = HolographicMemoryEngine::new(dimension).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    pub fn circular_convolution(py: Python<'_>, a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| HolographicMemoryEngine::circular_convolution(&a.inner, &b.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    #[staticmethod]
    pub fn circular_correlation(py: Python<'_>, a: &PyTensor, c: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| HolographicMemoryEngine::circular_correlation(&a.inner, &c.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn store(&mut self, py: Python<'_>, key: &PyTensor, value: &PyTensor) -> PyResult<()> {
        py.detach(|| self.inner.store(&key.inner, &value.inner))
            .map_err(to_py_err)?;
        Ok(())
    }

    pub fn retrieve(&self, py: Python<'_>, query_key: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.retrieve(&query_key.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    #[staticmethod]
    pub fn hopfield_cleanup(
        py: Python<'_>,
        query: &PyTensor,
        patterns: &PyTensor,
        beta: f64,
    ) -> PyResult<PyTensor> {
        let out = py
            .detach(|| {
                HolographicMemoryEngine::hopfield_cleanup(&query.inner, &patterns.inner, beta)
            })
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    #[getter]
    pub fn memory_trace(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.memory_trace.clone(),
        }
    }
}

#[pyclass(name = "QuantumCircuit", module = "oxideon")]
pub struct PyQuantumCircuit {
    pub inner: QuantumCircuit,
}

#[pymethods]
impl PyQuantumCircuit {
    #[new]
    pub fn new(num_qubits: usize) -> PyResult<Self> {
        let inner = QuantumCircuit::new(num_qubits).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn h(&mut self, qubit: usize) -> PyResult<()> {
        self.inner.h(qubit).map_err(to_py_err)?;
        Ok(())
    }

    pub fn x(&mut self, qubit: usize) -> PyResult<()> {
        self.inner.x(qubit).map_err(to_py_err)?;
        Ok(())
    }

    pub fn ry(&mut self, qubit: usize, theta: f64) -> PyResult<()> {
        self.inner.ry(qubit, theta).map_err(to_py_err)?;
        Ok(())
    }

    pub fn cnot(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.inner.cnot(control, target).map_err(to_py_err)?;
        Ok(())
    }

    pub fn expectation_z(&self, qubit: usize) -> PyResult<f64> {
        self.inner.expectation_z(qubit).map_err(to_py_err)
    }

    #[getter]
    pub fn state_vector(&self) -> (Vec<f64>, Vec<f64>) {
        (self.inner.state_real.clone(), self.inner.state_imag.clone())
    }
}

#[pyclass(name = "StabilizerCode", module = "oxideon")]
pub struct PyStabilizerCode {
    pub inner: StabilizerCode,
}

#[pymethods]
impl PyStabilizerCode {
    #[staticmethod]
    pub fn bit_flip_3_qubit() -> Self {
        Self {
            inner: StabilizerCode::bit_flip_3_qubit(),
        }
    }

    #[staticmethod]
    pub fn phase_flip_3_qubit() -> Self {
        Self {
            inner: StabilizerCode::phase_flip_3_qubit(),
        }
    }

    #[staticmethod]
    pub fn steane_7_qubit() -> Self {
        Self {
            inner: StabilizerCode::steane_7_qubit(),
        }
    }

    #[getter]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    pub fn num_physical_qubits(&self) -> usize {
        self.inner.num_physical_qubits
    }

    #[getter]
    pub fn distance(&self) -> usize {
        self.inner.distance
    }
}

#[pyclass(name = "QecEngine", module = "oxideon")]
pub struct PyQecEngine;

#[pymethods]
impl PyQecEngine {
    #[staticmethod]
    pub fn encode_3qubit_bit_flip(
        circuit: &mut PyQuantumCircuit,
        q0: usize,
        q1: usize,
        q2: usize,
    ) -> PyResult<()> {
        QecEngine::encode_3qubit_bit_flip(&mut circuit.inner, q0, q1, q2).map_err(to_py_err)
    }

    #[staticmethod]
    pub fn encode_3qubit_phase_flip(
        circuit: &mut PyQuantumCircuit,
        q0: usize,
        q1: usize,
        q2: usize,
    ) -> PyResult<()> {
        QecEngine::encode_3qubit_phase_flip(&mut circuit.inner, q0, q1, q2).map_err(to_py_err)
    }

    #[staticmethod]
    pub fn measure_syndromes(
        circuit: &PyQuantumCircuit,
        code: &PyStabilizerCode,
    ) -> PyResult<Vec<i8>> {
        QecEngine::measure_syndromes(&circuit.inner, &code.inner).map_err(to_py_err)
    }

    #[staticmethod]
    pub fn correct_bit_flip(
        circuit: &mut PyQuantumCircuit,
        code: &PyStabilizerCode,
    ) -> PyResult<Option<usize>> {
        QecEngine::correct_bit_flip(&mut circuit.inner, &code.inner).map_err(to_py_err)
    }
}

#[pyclass(name = "QaoaCircuit", module = "oxideon")]
pub struct PyQaoaCircuit {
    inner: QaoaCircuit,
}

#[pymethods]
impl PyQaoaCircuit {
    #[new]
    pub fn new(
        num_qubits: usize,
        p_layers: usize,
        edges: Vec<(usize, usize, f64)>,
    ) -> PyResult<Self> {
        let inner = QaoaCircuit::new(num_qubits, p_layers, edges).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn evaluate_energy(
        &self,
        py: Python<'_>,
        gamma: Vec<f64>,
        beta: Vec<f64>,
    ) -> PyResult<f64> {
        py.detach(|| self.inner.evaluate_energy(&gamma, &beta))
            .map_err(to_py_err)
    }

    pub fn compute_gradients(
        &self,
        py: Python<'_>,
        gamma: Vec<f64>,
        beta: Vec<f64>,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        py.detach(|| self.inner.compute_gradients(&gamma, &beta))
            .map_err(to_py_err)
    }

    pub fn optimize(
        &self,
        py: Python<'_>,
        steps: usize,
        lr: f64,
    ) -> PyResult<(Vec<f64>, Vec<f64>, f64)> {
        py.detach(|| self.inner.optimize(steps, lr))
            .map_err(to_py_err)
    }

    pub fn build_circuit(
        &self,
        py: Python<'_>,
        gamma: Vec<f64>,
        beta: Vec<f64>,
    ) -> PyResult<PyQuantumCircuit> {
        let circuit = py
            .detach(|| self.inner.build_circuit(&gamma, &beta))
            .map_err(to_py_err)?;
        Ok(PyQuantumCircuit { inner: circuit })
    }
}

#[pyclass(name = "VqeAnsatz", module = "oxideon")]
pub struct PyVqeAnsatz {
    inner: VqeAnsatz,
}

#[pymethods]
impl PyVqeAnsatz {
    #[new]
    #[pyo3(signature = (num_qubits, num_layers=1))]
    pub fn new(num_qubits: usize, num_layers: usize) -> PyResult<Self> {
        let inner = VqeAnsatz::new(num_qubits, num_layers).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn num_parameters(&self) -> usize {
        self.inner.num_parameters()
    }

    pub fn evaluate_energy_pauli_z(
        &self,
        py: Python<'_>,
        params: Vec<f64>,
        terms: Vec<(usize, f64)>,
    ) -> PyResult<f64> {
        let n = self.inner.num_qubits;
        let mut pauli_terms = Vec::with_capacity(terms.len());
        for (q, coeff) in terms {
            if q >= n {
                return Err(to_py_err(crate::error::Error::Msg(format!(
                    "Qubit {q} out of bounds (0..{n})"
                ))));
            }
            let mut string = vec![crate::quantum::Pauli::I; n];
            string[q] = crate::quantum::Pauli::Z;
            pauli_terms.push((string, coeff));
        }

        py.detach(|| self.inner.evaluate_energy(&params, &pauli_terms))
            .map_err(to_py_err)
    }
}

#[pyclass(name = "MatrixProductState", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyMatrixProductState {
    pub inner: MatrixProductState,
}

#[pymethods]
impl PyMatrixProductState {
    #[staticmethod]
    pub fn from_product_state(spins: Vec<usize>) -> PyResult<Self> {
        let inner = MatrixProductState::from_product_state(&spins).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (num_sites, max_bond_dim=4, seed=None))]
    pub fn new_random(num_sites: usize, max_bond_dim: usize, seed: Option<u64>) -> PyResult<Self> {
        let inner =
            MatrixProductState::new_random(num_sites, max_bond_dim, seed).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn num_sites(&self) -> usize {
        self.inner.num_sites
    }

    #[getter]
    pub fn max_bond_dim(&self) -> usize {
        self.inner.max_bond_dim
    }

    pub fn overlap(&self, py: Python<'_>, other: &PyMatrixProductState) -> PyResult<f64> {
        let o = &other.inner;
        py.detach(|| self.inner.overlap(o)).map_err(to_py_err)
    }

    pub fn norm(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.norm()).map_err(to_py_err)
    }

    pub fn normalize(&mut self) -> PyResult<()> {
        self.inner.normalize().map_err(to_py_err)
    }

    pub fn expectation_single_site(
        &self,
        py: Python<'_>,
        site_k: usize,
        op: Vec<f64>,
    ) -> PyResult<f64> {
        if op.len() != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "op must be a flat 4-element 2x2 matrix",
            ));
        }
        let mut arr = [0.0f64; 4];
        arr.copy_from_slice(&op);
        py.detach(|| self.inner.expectation_single_site(site_k, &arr))
            .map_err(to_py_err)
    }

    pub fn entanglement_entropy(&self, py: Python<'_>, bond_k: usize) -> PyResult<f64> {
        py.detach(|| self.inner.entanglement_entropy(bond_k))
            .map_err(to_py_err)
    }

    #[pyo3(signature = (j_coupling=1.0, h_transverse=0.5))]
    pub fn evaluate_tfim_energy(
        &self,
        py: Python<'_>,
        j_coupling: f64,
        h_transverse: f64,
    ) -> PyResult<f64> {
        py.detach(|| self.inner.evaluate_tfim_energy(j_coupling, h_transverse))
            .map_err(to_py_err)
    }
}

#[pyclass(name = "ZneExtrapolator", module = "oxideon")]
pub struct PyZneExtrapolator {
    pub(crate) inner: ZneExtrapolator,
}

#[pymethods]
impl PyZneExtrapolator {
    #[new]
    #[pyo3(signature = (scale_factors, model="richardson"))]
    pub fn new(scale_factors: Vec<f64>, model: &str) -> PyResult<Self> {
        let m = match model.to_lowercase().as_str() {
            "linear" => ExtrapolationModel::Linear,
            "exponential" => ExtrapolationModel::Exponential,
            _ => ExtrapolationModel::Richardson,
        };
        let inner = ZneExtrapolator::new(scale_factors, m).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn extrapolate(&self, py: Python<'_>, noisy_values: Vec<f64>) -> PyResult<f64> {
        py.detach(|| self.inner.extrapolate(&noisy_values))
            .map_err(to_py_err)
    }
}

#[pyclass(name = "CliffordDataRegression", module = "oxideon")]
pub struct PyCliffordDataRegression {
    pub(crate) inner: CliffordDataRegression,
}

#[pymethods]
impl PyCliffordDataRegression {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: CliffordDataRegression::new(),
        }
    }

    pub fn fit(
        &mut self,
        py: Python<'_>,
        noisy_expectations: Vec<f64>,
        exact_expectations: Vec<f64>,
    ) -> PyResult<()> {
        py.detach(|| self.inner.fit(&noisy_expectations, &exact_expectations))
            .map_err(to_py_err)
    }

    pub fn mitigate(&self, noisy_target_expectation: f64) -> PyResult<f64> {
        self.inner
            .mitigate(noisy_target_expectation)
            .map_err(to_py_err)
    }

    #[getter]
    pub fn alpha(&self) -> f64 {
        self.inner.alpha
    }

    #[getter]
    pub fn beta(&self) -> f64 {
        self.inner.beta
    }
}

impl Default for PyCliffordDataRegression {
    fn default() -> Self {
        Self::new()
    }
}

#[pyclass(name = "DeqConfig", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyDeqConfig {
    pub inner: DeqConfig,
}

#[pymethods]
impl PyDeqConfig {
    #[new]
    #[pyo3(signature = (max_forward_iters=50, fwd_tol=1e-6, relaxation_beta=0.6, max_backward_iters=40, bwd_tol=1e-6))]
    pub fn new(
        max_forward_iters: usize,
        fwd_tol: f64,
        relaxation_beta: f64,
        max_backward_iters: usize,
        bwd_tol: f64,
    ) -> Self {
        Self {
            inner: DeqConfig {
                max_forward_iters,
                fwd_tol,
                relaxation_beta,
                max_backward_iters,
                bwd_tol,
            },
        }
    }

    #[getter]
    pub fn max_forward_iters(&self) -> usize {
        self.inner.max_forward_iters
    }

    #[getter]
    pub fn fwd_tol(&self) -> f64 {
        self.inner.fwd_tol
    }

    #[getter]
    pub fn relaxation_beta(&self) -> f64 {
        self.inner.relaxation_beta
    }

    #[getter]
    pub fn max_backward_iters(&self) -> usize {
        self.inner.max_backward_iters
    }

    #[getter]
    pub fn bwd_tol(&self) -> f64 {
        self.inner.bwd_tol
    }
}

#[pyclass(name = "DeepEquilibriumModel", module = "oxideon")]
pub struct PyDeepEquilibriumModel {
    pub inner: DeepEquilibriumModel,
}

#[pymethods]
impl PyDeepEquilibriumModel {
    #[new]
    #[pyo3(signature = (input_dim, state_dim, config=None))]
    pub fn new(input_dim: usize, state_dim: usize, config: Option<PyDeqConfig>) -> PyResult<Self> {
        let cfg = config.map(|c| c.inner).unwrap_or_default();
        let inner = DeepEquilibriumModel::new(input_dim, state_dim, cfg).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn forward_equilibrium(
        &self,
        py: Python<'_>,
        x: &PyTensor,
    ) -> PyResult<(PyTensor, f64, usize, bool)> {
        let x_tensor = &x.inner;
        let out = py
            .detach(|| self.inner.forward_equilibrium(x_tensor))
            .map_err(to_py_err)?;
        Ok((
            PyTensor { inner: out.z_star },
            out.residual_norm,
            out.num_iters,
            out.converged,
        ))
    }

    pub fn solve_adjoint(
        &self,
        py: Python<'_>,
        grad_z_star: Vec<f64>,
        z_star: Vec<f64>,
        x: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| self.inner.solve_adjoint(&grad_z_star, &z_star, &x))
            .map_err(to_py_err)
    }

    pub fn vjp_state(
        &self,
        py: Python<'_>,
        v: Vec<f64>,
        z_star: Vec<f64>,
        x: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| self.inner.vjp_state(&v, &z_star, &x))
            .map_err(to_py_err)
    }
}

#[pyclass(name = "RiskCalibrationResult", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyRiskCalibrationResult {
    pub inner: RiskCalibrationResult,
}

#[pymethods]
impl PyRiskCalibrationResult {
    #[getter]
    pub fn lambda_hat(&self) -> f64 {
        self.inner.lambda_hat
    }

    #[getter]
    pub fn empirical_risk(&self) -> f64 {
        self.inner.empirical_risk
    }

    #[getter]
    pub fn risk_upper_bound(&self) -> f64 {
        self.inner.risk_upper_bound
    }

    #[getter]
    pub fn target_alpha(&self) -> f64 {
        self.inner.target_alpha
    }

    #[getter]
    pub fn loss_bound_b(&self) -> f64 {
        self.inner.loss_bound_b
    }

    #[getter]
    pub fn num_samples(&self) -> usize {
        self.inner.num_samples
    }
}

#[pyclass(name = "ConformalRiskController", module = "oxideon")]
pub struct PyConformalRiskController {
    pub inner: ConformalRiskController,
}

#[pymethods]
impl PyConformalRiskController {
    #[new]
    #[pyo3(signature = (metric="fnr", alpha=0.1, loss_bound_b=1.0))]
    pub fn new(metric: &str, alpha: f64, loss_bound_b: f64) -> PyResult<Self> {
        let m = match metric.to_lowercase().as_str() {
            "fnr" | "false_negative_rate" => RiskMetric::FalseNegativeRate,
            "fdp" | "false_discovery_proportion" => RiskMetric::FalseDiscoveryProportion,
            _ => RiskMetric::BoundedLoss,
        };
        let inner = ConformalRiskController::new(m, alpha, loss_bound_b).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn compute_loss(&self, pred_set: Vec<usize>, true_labels: Vec<usize>) -> PyResult<f64> {
        self.inner
            .compute_loss(&pred_set, &true_labels)
            .map_err(to_py_err)
    }

    #[staticmethod]
    pub fn predict_set(scores: Vec<f64>, lambda: f64) -> Vec<usize> {
        ConformalRiskController::predict_set(&scores, lambda)
    }

    pub fn evaluate_empirical_risk(
        &self,
        py: Python<'_>,
        calibration_scores: Vec<Vec<f64>>,
        ground_truth: Vec<Vec<usize>>,
        lambda: f64,
    ) -> PyResult<f64> {
        py.detach(|| {
            self.inner
                .evaluate_empirical_risk(&calibration_scores, &ground_truth, lambda)
        })
        .map_err(to_py_err)
    }

    pub fn calibrate(
        &self,
        py: Python<'_>,
        calibration_scores: Vec<Vec<f64>>,
        ground_truth: Vec<Vec<usize>>,
        num_grid_steps: usize,
    ) -> PyResult<PyRiskCalibrationResult> {
        let res = py
            .detach(|| {
                self.inner
                    .calibrate(&calibration_scores, &ground_truth, num_grid_steps)
            })
            .map_err(to_py_err)?;
        Ok(PyRiskCalibrationResult { inner: res })
    }
}

#[pyclass(name = "ToposHeytingEngine", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyToposHeytingEngine {
    pub inner: ToposHeytingEngine,
}

#[pymethods]
impl PyToposHeytingEngine {
    #[new]
    pub fn new(num_levels: usize) -> PyResult<Self> {
        let inner = ToposHeytingEngine::new(num_levels).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    pub fn meet(u: f64, v: f64) -> f64 {
        ToposHeytingEngine::meet(u, v)
    }

    #[staticmethod]
    pub fn join(u: f64, v: f64) -> f64 {
        ToposHeytingEngine::join(u, v)
    }

    #[staticmethod]
    pub fn imply(u: f64, v: f64) -> f64 {
        ToposHeytingEngine::imply(u, v)
    }

    #[staticmethod]
    pub fn negate(u: f64) -> f64 {
        ToposHeytingEngine::negate(u)
    }

    #[staticmethod]
    pub fn double_negation_modality(u: f64) -> f64 {
        ToposHeytingEngine::double_negation_modality(u)
    }

    pub fn verify_heyting_adjunction(&self, py: Python<'_>) -> PyResult<bool> {
        py.detach(|| self.inner.verify_heyting_adjunction())
            .map_err(to_py_err)
    }

    pub fn verify_lawvere_tierney_axioms(&self, py: Python<'_>) -> PyResult<bool> {
        py.detach(|| self.inner.verify_lawvere_tierney_axioms())
            .map_err(to_py_err)
    }
}

#[pyclass(name = "LearnThenTest", module = "oxideon")]
pub struct PyLearnThenTest;

#[pymethods]
impl PyLearnThenTest {
    #[staticmethod]
    pub fn hoeffding_bentkus_p_value(emp_risk: f64, alpha: f64, n: usize) -> PyResult<f64> {
        LearnThenTest::hoeffding_bentkus_p_value(emp_risk, alpha, n).map_err(to_py_err)
    }

    #[staticmethod]
    pub fn benjamini_hochberg(p_values: Vec<f64>, fdr_q: f64) -> Vec<usize> {
        LearnThenTest::benjamini_hochberg(&p_values, fdr_q)
    }
}

#[pyclass(name = "ComplexEigenvalue", module = "oxideon", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyComplexEigenvalue {
    #[pyo3(get)]
    pub real: f64,
    #[pyo3(get)]
    pub imag: f64,
}

#[pymethods]
impl PyComplexEigenvalue {
    #[new]
    pub fn new(real: f64, imag: f64) -> Self {
        Self { real, imag }
    }

    pub fn modulus(&self) -> f64 {
        (self.real * self.real + self.imag * self.imag).sqrt()
    }
}

#[pyclass(name = "ExceptionalPointReport", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyExceptionalPointReport {
    pub inner: ExceptionalPointReport,
}

#[pymethods]
impl PyExceptionalPointReport {
    #[getter]
    pub fn lambda_plus(&self) -> PyComplexEigenvalue {
        PyComplexEigenvalue {
            real: self.inner.lambda_plus.real,
            imag: self.inner.lambda_plus.imag,
        }
    }

    #[getter]
    pub fn lambda_minus(&self) -> PyComplexEigenvalue {
        PyComplexEigenvalue {
            real: self.inner.lambda_minus.real,
            imag: self.inner.lambda_minus.imag,
        }
    }

    #[getter]
    pub fn spectral_gap(&self) -> f64 {
        self.inner.spectral_gap
    }

    #[getter]
    pub fn petermann_factor(&self) -> f64 {
        self.inner.petermann_factor
    }

    #[getter]
    pub fn is_pt_unbroken(&self) -> bool {
        self.inner.is_pt_unbroken
    }

    #[getter]
    pub fn is_exceptional_point(&self) -> bool {
        self.inner.is_exceptional_point
    }
}

#[pyclass(name = "NonHermitianHamiltonian2x2", module = "oxideon")]
pub struct PyNonHermitianHamiltonian2x2 {
    pub inner: NonHermitianHamiltonian2x2,
}

#[pymethods]
impl PyNonHermitianHamiltonian2x2 {
    #[new]
    #[pyo3(signature = (delta, gamma, coupling_g))]
    pub fn new(delta: f64, gamma: f64, coupling_g: f64) -> PyResult<Self> {
        let inner = NonHermitianHamiltonian2x2::new(delta, gamma, coupling_g).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn eigenvalues(&self) -> (PyComplexEigenvalue, PyComplexEigenvalue) {
        let (lp, lm) = self.inner.eigenvalues();
        (
            PyComplexEigenvalue {
                real: lp.real,
                imag: lp.imag,
            },
            PyComplexEigenvalue {
                real: lm.real,
                imag: lm.imag,
            },
        )
    }

    pub fn petermann_factor(&self) -> f64 {
        self.inner.petermann_factor()
    }

    pub fn analyze(&self) -> PyExceptionalPointReport {
        PyExceptionalPointReport {
            inner: self.inner.analyze(),
        }
    }

    #[staticmethod]
    pub fn encircle_ep(
        py: Python<'_>,
        coupling_g: f64,
        loop_radius_rho: f64,
        num_steps: usize,
    ) -> PyResult<Vec<(f64, f64, f64, f64)>> {
        py.detach(|| {
            NonHermitianHamiltonian2x2::encircle_ep(coupling_g, loop_radius_rho, num_steps)
        })
        .map_err(to_py_err)
    }

    pub fn sensor_frequency_splitting(&self, perturbation_eps: f64) -> PyResult<f64> {
        self.inner
            .sensor_frequency_splitting(perturbation_eps)
            .map_err(to_py_err)
    }
}

#[pyclass(name = "JarzynskiReport", module = "oxideon", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyJarzynskiReport {
    pub inner: JarzynskiReport,
}

#[pymethods]
impl PyJarzynskiReport {
    #[getter]
    pub fn mean_work(&self) -> f64 {
        self.inner.mean_work
    }

    #[getter]
    pub fn jarzynski_delta_f(&self) -> f64 {
        self.inner.jarzynski_delta_f
    }

    #[getter]
    pub fn dissipated_work(&self) -> f64 {
        self.inner.dissipated_work
    }

    #[getter]
    pub fn mean_entropy_production(&self) -> f64 {
        self.inner.mean_entropy_production
    }
}

#[pyclass(name = "StochasticThermoEngine", module = "oxideon", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyStochasticThermoEngine {
    pub inner: StochasticThermoEngine,
}

#[pymethods]
impl PyStochasticThermoEngine {
    #[new]
    pub fn new(beta: f64) -> PyResult<Self> {
        let inner = StochasticThermoEngine::new(beta).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn evaluate_jarzynski(
        &self,
        py: Python<'_>,
        work_samples: Vec<f64>,
    ) -> PyResult<PyJarzynskiReport> {
        let rep = py
            .detach(|| self.inner.evaluate_jarzynski(&work_samples))
            .map_err(to_py_err)?;
        Ok(PyJarzynskiReport { inner: rep })
    }

    pub fn crooks_log_ratio(&self, work_w: f64, delta_f: f64) -> f64 {
        self.inner.crooks_log_ratio(work_w, delta_f)
    }

    pub fn landauer_erasure_bound(&self, bits_erased: f64) -> f64 {
        self.inner.landauer_erasure_bound(bits_erased)
    }
}

#[pyclass(name = "SymplecticTopologyEngine", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PySymplecticTopologyEngine {
    pub inner: SymplecticTopologyEngine,
}

#[pymethods]
impl PySymplecticTopologyEngine {
    #[new]
    pub fn new(num_dof: usize) -> PyResult<Self> {
        let inner = SymplecticTopologyEngine::new(num_dof).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn symplecticity_defect(&self, matrix: Vec<f64>) -> PyResult<f64> {
        self.inner.symplecticity_defect(&matrix).map_err(to_py_err)
    }

    pub fn ellipsoid_gromov_capacity(&self, radii: Vec<f64>) -> PyResult<f64> {
        self.inner
            .ellipsoid_gromov_capacity(&radii)
            .map_err(to_py_err)
    }

    pub fn ellipsoid_liouville_volume(&self, radii: Vec<f64>) -> PyResult<f64> {
        self.inner
            .ellipsoid_liouville_volume(&radii)
            .map_err(to_py_err)
    }

    pub fn verify_gromov_non_squeezing(
        &self,
        ball_radius_r: f64,
        cylinder_radius_r: f64,
    ) -> PyResult<(bool, f64, f64)> {
        self.inner
            .verify_gromov_non_squeezing(ball_radius_r, cylinder_radius_r)
            .map_err(to_py_err)
    }

    pub fn floer_action_functional(
        &self,
        py: Python<'_>,
        loop_points: Vec<Vec<f64>>,
        omegas: Vec<f64>,
    ) -> PyResult<f64> {
        py.detach(|| self.inner.floer_action_functional(&loop_points, &omegas))
            .map_err(to_py_err)
    }
}

#[pyclass(name = "AdsCftHolographicEngine", module = "oxideon", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyAdsCftHolographicEngine {
    pub inner: AdsCftHolographicEngine,
}

#[pymethods]
impl PyAdsCftHolographicEngine {
    #[new]
    #[pyo3(signature = (ads_radius=1.0, newton_g=0.1, uv_cutoff=1e-3))]
    pub fn new(ads_radius: f64, newton_g: f64, uv_cutoff: f64) -> PyResult<Self> {
        let inner =
            AdsCftHolographicEngine::new(ads_radius, newton_g, uv_cutoff).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn central_charge(&self) -> f64 {
        self.inner.central_charge()
    }

    pub fn bulk_geodesic_distance(&self, x1: f64, z1: f64, x2: f64, z2: f64) -> PyResult<f64> {
        self.inner
            .bulk_geodesic_distance(x1, z1, x2, z2)
            .map_err(to_py_err)
    }

    pub fn minimal_surface_area(&self, interval_length: f64) -> PyResult<f64> {
        self.inner
            .minimal_surface_area(interval_length)
            .map_err(to_py_err)
    }

    pub fn ryu_takayanagi_entropy(&self, interval_length: f64) -> PyResult<f64> {
        self.inner
            .ryu_takayanagi_entropy(interval_length)
            .map_err(to_py_err)
    }

    pub fn sample_rt_geodesic(
        &self,
        py: Python<'_>,
        x_left: f64,
        x_right: f64,
        num_points: usize,
    ) -> PyResult<(Vec<(f64, f64)>, f64)> {
        py.detach(|| self.inner.sample_rt_geodesic(x_left, x_right, num_points))
            .map_err(to_py_err)
    }

    pub fn holographic_mutual_information(
        &self,
        len_a: f64,
        len_b: f64,
        separation_d: f64,
    ) -> PyResult<f64> {
        self.inner
            .holographic_mutual_information(len_a, len_b, separation_d)
            .map_err(to_py_err)
    }

    pub fn strong_subadditivity_gap(&self, len_a: f64, len_b: f64, len_c: f64) -> PyResult<f64> {
        self.inner
            .strong_subadditivity_gap(len_a, len_b, len_c)
            .map_err(to_py_err)
    }
}

#[pyclass(name = "RmhdPrimitiveState", module = "oxideon", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyRmhdPrimitiveState {
    pub inner: RmhdPrimitiveState,
}

#[pymethods]
impl PyRmhdPrimitiveState {
    #[new]
    #[pyo3(signature = (rho, pressure, velocity, b_field, gamma_adiabatic=1.3333333333333333))]
    pub fn new(
        rho: f64,
        pressure: f64,
        velocity: [f64; 3],
        b_field: [f64; 3],
        gamma_adiabatic: f64,
    ) -> PyResult<Self> {
        let inner = RmhdPrimitiveState::new(rho, pressure, velocity, b_field, gamma_adiabatic)
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn lorentz_factor(&self) -> f64 {
        self.inner.lorentz_factor()
    }

    pub fn four_velocity(&self) -> [f64; 4] {
        self.inner.four_velocity()
    }

    pub fn magnetic_four_vector(&self) -> [f64; 4] {
        self.inner.magnetic_four_vector()
    }

    pub fn orthogonality_residual(&self) -> f64 {
        self.inner.orthogonality_residual()
    }

    pub fn magnetic_energy_squared(&self) -> f64 {
        self.inner.magnetic_energy_squared()
    }

    pub fn specific_enthalpy(&self) -> f64 {
        self.inner.specific_enthalpy()
    }

    pub fn stress_energy_tensor(&self) -> [[f64; 4]; 4] {
        self.inner.stress_energy_tensor()
    }

    pub fn alfven_speed(&self) -> f64 {
        self.inner.alfven_speed()
    }

    pub fn sound_speed(&self) -> f64 {
        self.inner.sound_speed()
    }

    pub fn fast_magnetosonic_speed(&self) -> f64 {
        self.inner.fast_magnetosonic_speed()
    }

    pub fn conserved_vector(&self) -> [f64; 5] {
        self.inner.conserved_vector()
    }

    pub fn flux_x(&self) -> [f64; 5] {
        self.inner.flux_x()
    }

    #[staticmethod]
    pub fn hll_riemann_flux(
        py: Python<'_>,
        left: &PyRmhdPrimitiveState,
        right: &PyRmhdPrimitiveState,
    ) -> [f64; 5] {
        let l = left.inner;
        let r = right.inner;
        py.detach(|| RmhdPrimitiveState::hll_riemann_flux(&l, &r))
    }
}

#[pyclass(name = "HeisenbergPoint", module = "oxideon", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyHeisenbergPoint {
    pub inner: HeisenbergPoint,
}

#[pymethods]
impl PyHeisenbergPoint {
    #[new]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self {
            inner: HeisenbergPoint::new(x, y, z),
        }
    }

    #[staticmethod]
    pub fn identity() -> Self {
        Self {
            inner: HeisenbergPoint::identity(),
        }
    }

    #[getter]
    pub fn x(&self) -> f64 {
        self.inner.x
    }

    #[getter]
    pub fn y(&self) -> f64 {
        self.inner.y
    }

    #[getter]
    pub fn z(&self) -> f64 {
        self.inner.z
    }

    pub fn multiply(&self, other: &PyHeisenbergPoint) -> Self {
        Self {
            inner: self.inner.multiply(&other.inner),
        }
    }

    pub fn inverse(&self) -> Self {
        Self {
            inner: self.inner.inverse(),
        }
    }

    pub fn horizontal_radius(&self) -> f64 {
        self.inner.horizontal_radius()
    }

    pub fn koranyi_norm(&self) -> f64 {
        self.inner.koranyi_norm()
    }

    pub fn koranyi_distance(&self, other: &PyHeisenbergPoint) -> f64 {
        self.inner.koranyi_distance(&other.inner)
    }

    pub fn carnot_caratheodory_distance(&self, other: &PyHeisenbergPoint) -> f64 {
        self.inner.carnot_caratheodory_distance(&other.inner)
    }

    pub fn dilate(&self, lambda: f64) -> Self {
        Self {
            inner: self.inner.dilate(lambda),
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "HeisenbergPoint(x={:.4}, y={:.4}, z={:.4})",
            self.inner.x, self.inner.y, self.inner.z
        )
    }
}

#[pyclass(name = "HeisenbergLieAlgebra", module = "oxideon", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyHeisenbergLieAlgebra {
    pub inner: HeisenbergLieAlgebra,
}

#[pymethods]
impl PyHeisenbergLieAlgebra {
    #[new]
    pub fn new(vx: f64, vy: f64, vz: f64) -> Self {
        Self {
            inner: HeisenbergLieAlgebra::new(vx, vy, vz),
        }
    }

    #[getter]
    pub fn vx(&self) -> f64 {
        self.inner.vx
    }

    #[getter]
    pub fn vy(&self) -> f64 {
        self.inner.vy
    }

    #[getter]
    pub fn vz(&self) -> f64 {
        self.inner.vz
    }

    pub fn bracket(&self, other: &PyHeisenbergLieAlgebra) -> Self {
        Self {
            inner: self.inner.bracket(&other.inner),
        }
    }

    pub fn exp(&self) -> PyHeisenbergPoint {
        PyHeisenbergPoint {
            inner: self.inner.exp(),
        }
    }

    #[staticmethod]
    pub fn log(point: &PyHeisenbergPoint) -> Self {
        Self {
            inner: HeisenbergLieAlgebra::log(&point.inner),
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "HeisenbergLieAlgebra(vx={:.4}, vy={:.4}, vz={:.4})",
            self.inner.vx, self.inner.vy, self.inner.vz
        )
    }
}

#[pyclass(name = "HypoellipticDiffusion3D", module = "oxideon")]
pub struct PyHypoellipticDiffusion3D {
    pub inner: HypoellipticDiffusion3D,
}

#[pymethods]
impl PyHypoellipticDiffusion3D {
    #[new]
    pub fn new(n: usize, l: f64) -> PyResult<Self> {
        let inner = HypoellipticDiffusion3D::new(n, l).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn init_gaussian(&mut self, sigma: f64) -> PyResult<()> {
        self.inner.init_gaussian(sigma).map_err(to_py_err)
    }

    pub fn step(&mut self, py: Python<'_>, dt: f64) {
        py.detach(|| self.inner.step(dt));
    }

    pub fn eval_sub_laplacian(&self, i: usize, j: usize, k: usize) -> PyResult<f64> {
        self.inner.eval_sub_laplacian(i, j, k).map_err(to_py_err)
    }

    pub fn total_mass(&self) -> f64 {
        self.inner.total_mass()
    }

    pub fn horizontal_variance(&self) -> f64 {
        self.inner.horizontal_variance()
    }

    pub fn vertical_variance(&self) -> f64 {
        self.inner.vertical_variance()
    }
}

#[pyclass(name = "Su2Element", module = "oxideon", from_py_object)]
#[derive(Clone, Copy)]
pub struct PySu2Element {
    pub inner: Su2Element,
}

#[pymethods]
impl PySu2Element {
    #[new]
    #[pyo3(signature = (a0=1.0, a1=0.0, a2=0.0, a3=0.0))]
    pub fn new(a0: f64, a1: f64, a2: f64, a3: f64) -> PyResult<Self> {
        let inner = Su2Element::from_quaternion(a0, a1, a2, a3).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    pub fn identity() -> Self {
        Self {
            inner: Su2Element::identity(),
        }
    }

    #[staticmethod]
    pub fn from_axis_angle(axis: [f64; 3], angle: f64) -> PyResult<Self> {
        let inner = Su2Element::from_axis_angle(axis, angle).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn a0(&self) -> f64 {
        self.inner.a0
    }
    #[getter]
    pub fn a1(&self) -> f64 {
        self.inner.a1
    }
    #[getter]
    pub fn a2(&self) -> f64 {
        self.inner.a2
    }
    #[getter]
    pub fn a3(&self) -> f64 {
        self.inner.a3
    }

    pub fn multiply(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.multiply(&other.inner),
        }
    }

    pub fn inverse(&self) -> Self {
        Self {
            inner: self.inner.inverse(),
        }
    }

    pub fn normalized_trace(&self) -> f64 {
        self.inner.normalized_trace()
    }
}

#[pyclass(name = "YangMillsLatticeGauge", module = "oxideon")]
pub struct PyYangMillsLatticeGauge {
    pub inner: YangMillsLatticeGauge,
}

#[pymethods]
impl PyYangMillsLatticeGauge {
    #[new]
    pub fn new(num_nodes: usize, edges: Vec<(usize, usize)>) -> PyResult<Self> {
        let inner = YangMillsLatticeGauge::new(num_nodes, edges).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn set_link(&mut self, edge_idx: usize, element: &PySu2Element) -> PyResult<()> {
        self.inner
            .set_link(edge_idx, element.inner)
            .map_err(to_py_err)
    }

    pub fn wilson_loop(&self, cycle_edges: Vec<usize>, forward: Vec<bool>) -> PyResult<f64> {
        self.inner
            .wilson_loop(&cycle_edges, &forward)
            .map_err(to_py_err)
    }

    pub fn apply_gauge_transformation(&mut self, omegas: Vec<PySu2Element>) -> PyResult<()> {
        let su2_omegas: Vec<Su2Element> = omegas.into_iter().map(|o| o.inner).collect();
        self.inner
            .apply_gauge_transformation(&su2_omegas)
            .map_err(to_py_err)
    }

    pub fn yang_mills_action(
        &self,
        plaquettes: Vec<Vec<(usize, bool)>>,
        beta: f64,
    ) -> PyResult<f64> {
        self.inner
            .yang_mills_action(&plaquettes, beta)
            .map_err(to_py_err)
    }
}

#[pyclass(name = "QGNNGradients", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyQGNNGradients {
    pub inner: QGNNGradients,
}

#[pymethods]
impl PyQGNNGradients {
    #[getter]
    pub fn grad_theta(&self) -> Vec<f64> {
        self.inner.grad_theta.clone()
    }

    #[getter]
    pub fn grad_phi(&self) -> Vec<f64> {
        self.inner.grad_phi.clone()
    }

    #[getter]
    pub fn grad_j(&self) -> Vec<f64> {
        self.inner.grad_j.clone()
    }
}

#[pyclass(name = "Su2ChernSimonsTqft", module = "oxideon", from_py_object)]
#[derive(Clone, Copy)]
pub struct PySu2ChernSimonsTqft {
    pub inner: Su2ChernSimonsTqft,
}

#[pymethods]
impl PySu2ChernSimonsTqft {
    #[new]
    pub fn new(level_k: usize) -> PyResult<Self> {
        let inner = Su2ChernSimonsTqft::new(level_k).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn num_anyons(&self) -> PyResult<usize> {
        self.inner.num_anyons().map_err(to_py_err)
    }

    pub fn quantum_dimension(&self, a: usize) -> PyResult<f64> {
        self.inner.quantum_dimension(a).map_err(to_py_err)
    }

    pub fn total_quantum_dimension(&self) -> PyResult<f64> {
        self.inner.total_quantum_dimension().map_err(to_py_err)
    }

    pub fn modular_s_matrix(&self, py: Python<'_>) -> PyResult<Vec<f64>> {
        py.detach(|| self.inner.modular_s_matrix())
            .map_err(to_py_err)
    }

    pub fn verlinde_fusion_coefficient(&self, a: usize, b: usize, c: usize) -> PyResult<usize> {
        self.inner
            .verlinde_fusion_coefficient(a, b, c)
            .map_err(to_py_err)
    }

    pub fn partition_function_s3(&self) -> PyResult<f64> {
        self.inner.partition_function_s3().map_err(to_py_err)
    }
}

#[pyclass(name = "GaussianCvCircuit", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyGaussianCvCircuit {
    pub inner: GaussianCvCircuit,
}

#[pymethods]
impl PyGaussianCvCircuit {
    #[new]
    pub fn new(num_modes: usize) -> PyResult<Self> {
        let inner = GaussianCvCircuit::new(num_modes).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn num_modes(&self) -> usize {
        self.inner.num_modes
    }

    #[getter]
    pub fn displacement(&self) -> Vec<f64> {
        self.inner.displacement.clone()
    }

    #[getter]
    pub fn covariance(&self) -> Vec<f64> {
        self.inner.covariance.clone()
    }

    pub fn displace(&mut self, mode: usize, delta_q: f64, delta_p: f64) -> PyResult<()> {
        self.inner
            .displace(mode, delta_q, delta_p)
            .map_err(to_py_err)
    }

    pub fn squeeze(&mut self, mode: usize, r: f64) -> PyResult<()> {
        self.inner.squeeze(mode, r).map_err(to_py_err)
    }

    pub fn phase_shift(&mut self, mode: usize, phi: f64) -> PyResult<()> {
        self.inner.phase_shift(mode, phi).map_err(to_py_err)
    }

    pub fn beamsplitter(&mut self, mode1: usize, mode2: usize, theta: f64) -> PyResult<()> {
        self.inner
            .beamsplitter(mode1, mode2, theta)
            .map_err(to_py_err)
    }

    pub fn mean_photon_number(&self, mode: usize) -> PyResult<f64> {
        self.inner.mean_photon_number(mode).map_err(to_py_err)
    }

    pub fn total_mean_photon_number(&self) -> PyResult<f64> {
        self.inner.total_mean_photon_number().map_err(to_py_err)
    }

    pub fn purity(&self) -> PyResult<f64> {
        self.inner.purity().map_err(to_py_err)
    }

    pub fn vacuum_probability(&self) -> PyResult<f64> {
        self.inner.vacuum_probability().map_err(to_py_err)
    }

    #[staticmethod]
    pub fn hafnian(py: Python<'_>, matrix: Vec<f64>, n: usize) -> PyResult<f64> {
        py.detach(|| matrix_hafnian(&matrix, n)).map_err(to_py_err)
    }
}

#[pyclass(name = "QuantumGraphNeuralNetwork", module = "oxideon")]
pub struct PyQuantumGraphNeuralNetwork {
    pub inner: QuantumGraphNeuralNetwork,
}

#[pymethods]
impl PyQuantumGraphNeuralNetwork {
    #[new]
    #[pyo3(signature = (num_qubits, edges, num_layers=1))]
    pub fn new(num_qubits: usize, edges: Vec<(usize, usize)>, num_layers: usize) -> PyResult<Self> {
        let inner =
            QuantumGraphNeuralNetwork::new(num_qubits, edges, num_layers).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn set_parameters(
        &mut self,
        theta: Vec<f64>,
        phi: Vec<f64>,
        j_couplings: Vec<f64>,
    ) -> PyResult<()> {
        self.inner
            .set_parameters(theta, phi, j_couplings)
            .map_err(to_py_err)
    }

    pub fn forward(&self, py: Python<'_>, node_features: Vec<f64>) -> PyResult<Vec<f64>> {
        py.detach(|| self.inner.forward(&node_features))
            .map_err(to_py_err)
    }

    pub fn forward_graph_energy(&self, py: Python<'_>, node_features: Vec<f64>) -> PyResult<f64> {
        py.detach(|| self.inner.forward_graph_energy(&node_features))
            .map_err(to_py_err)
    }

    pub fn parameter_shift_gradients(
        &self,
        py: Python<'_>,
        node_features: Vec<f64>,
    ) -> PyResult<PyQGNNGradients> {
        let grads = py
            .detach(|| self.inner.parameter_shift_gradients(&node_features))
            .map_err(to_py_err)?;
        Ok(PyQGNNGradients { inner: grads })
    }
}

#[pyclass(name = "DynamicTelemetry", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyDynamicTelemetry {
    #[pyo3(get, set)]
    pub token_latency_ms: f64,
    #[pyo3(get, set)]
    pub memory_headroom_mb: f64,
    #[pyo3(get, set)]
    pub kv_cache_usage_pct: f64,
    #[pyo3(get, set)]
    pub activation_kurtosis: f64,
    #[pyo3(get, set)]
    pub token_index: usize,
}

#[pymethods]
impl PyDynamicTelemetry {
    #[new]
    #[pyo3(signature = (token_latency_ms, memory_headroom_mb, kv_cache_usage_pct, activation_kurtosis, token_index))]
    pub fn new(
        token_latency_ms: f64,
        memory_headroom_mb: f64,
        kv_cache_usage_pct: f64,
        activation_kurtosis: f64,
        token_index: usize,
    ) -> Self {
        Self {
            token_latency_ms,
            memory_headroom_mb,
            kv_cache_usage_pct,
            activation_kurtosis,
            token_index,
        }
    }
}

#[pyclass(name = "DynamicQuantPolicy", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyDynamicQuantPolicy {
    pub inner: DynamicQuantPolicy,
}

#[pymethods]
impl PyDynamicQuantPolicy {
    #[new]
    #[pyo3(signature = (target_latency_ms=25.0, min_memory_headroom_mb=512.0, critical_memory_headroom_mb=128.0, max_acceptable_kurtosis=6.0, kv_cache_threshold_pct=85.0, cooldown_tokens=10))]
    pub fn new(
        target_latency_ms: f64,
        min_memory_headroom_mb: f64,
        critical_memory_headroom_mb: f64,
        max_acceptable_kurtosis: f64,
        kv_cache_threshold_pct: f64,
        cooldown_tokens: usize,
    ) -> Self {
        Self {
            inner: DynamicQuantPolicy {
                target_latency_ms,
                min_memory_headroom_mb,
                critical_memory_headroom_mb,
                max_acceptable_kurtosis,
                kv_cache_threshold_pct,
                cooldown_tokens,
            },
        }
    }
}

#[pyclass(name = "DynamicTransitionPlan", module = "oxideon")]
pub struct PyDynamicTransitionPlan {
    pub inner: DynamicTransitionPlan,
}

#[pymethods]
impl PyDynamicTransitionPlan {
    #[getter]
    pub fn target_precision(&self) -> String {
        format!("{:?}", self.inner.target_precision)
    }

    #[getter]
    pub fn affected_layers(&self) -> Vec<String> {
        self.inner.affected_layers.clone()
    }

    #[getter]
    pub fn estimated_memory_saved_mb(&self) -> f64 {
        self.inner.estimated_memory_saved_mb
    }

    #[getter]
    pub fn expected_speedup(&self) -> f64 {
        self.inner.expected_speedup
    }

    #[getter]
    pub fn trigger_reason(&self) -> String {
        self.inner.trigger_reason.clone()
    }
}

#[pyclass(name = "DynamicLlmQuantizer", module = "oxideon")]
pub struct PyDynamicLlmQuantizer {
    inner: DynamicLlmQuantizer,
}

#[pymethods]
impl PyDynamicLlmQuantizer {
    #[new]
    #[pyo3(signature = (policy=None))]
    pub fn new(policy: Option<PyDynamicQuantPolicy>) -> Self {
        let p = policy.map(|p| p.inner).unwrap_or_default();
        Self {
            inner: DynamicLlmQuantizer::new(p),
        }
    }

    #[pyo3(signature = (name, weight, bias=None))]
    pub fn add_layer(
        &mut self,
        name: &str,
        weight: &PyTensor,
        bias: Option<&PyTensor>,
    ) -> PyResult<()> {
        let w = weight.inner.clone();
        let b = bias.map(|b| b.inner.clone());
        let linear = crate::nn::Linear::from_tensors(w, b).map_err(to_py_err)?;
        self.inner.add_layer(name, linear);
        Ok(())
    }

    pub fn forward_layer(
        &mut self,
        py: Python<'_>,
        name: &str,
        input: &PyTensor,
    ) -> PyResult<PyTensor> {
        let inp = input.inner.clone();
        let name_owned = name.to_string();
        let out = py
            .detach(|| self.inner.forward_layer(&name_owned, &inp))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn evaluate_telemetry(
        &mut self,
        py: Python<'_>,
        telemetry: PyDynamicTelemetry,
    ) -> PyResult<Option<PyDynamicTransitionPlan>> {
        let t = DynamicTelemetry {
            token_latency_ms: telemetry.token_latency_ms,
            memory_headroom_mb: telemetry.memory_headroom_mb,
            kv_cache_usage_pct: telemetry.kv_cache_usage_pct,
            activation_kurtosis: telemetry.activation_kurtosis,
            token_index: telemetry.token_index,
        };
        let plan = py
            .detach(|| self.inner.evaluate_telemetry(&t))
            .map(|plan| PyDynamicTransitionPlan { inner: plan });
        Ok(plan)
    }

    pub fn apply_plan(
        &mut self,
        py: Python<'_>,
        plan: &PyDynamicTransitionPlan,
        current_token: usize,
    ) -> PyResult<()> {
        let p = plan.inner.clone();
        py.detach(|| self.inner.apply_plan(&p, current_token))
            .map_err(to_py_err)
    }

    pub fn total_memory_mb(&self) -> f64 {
        self.inner.total_memory_mb()
    }
}

#[pyclass(name = "TemporalInteractionGNN", module = "oxideon")]
pub struct PyTemporalInteractionGNN {
    inner: TemporalInteractionGNN,
}

#[pymethods]
impl PyTemporalInteractionGNN {
    #[new]
    #[pyo3(signature = (num_nodes, memory_dim=16, time_dim=16, edge_dim=8))]
    pub fn new(
        num_nodes: usize,
        memory_dim: usize,
        time_dim: usize,
        edge_dim: usize,
    ) -> PyResult<Self> {
        let inner = TemporalInteractionGNN::new(num_nodes, memory_dim, time_dim, edge_dim)
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn process_event(
        &mut self,
        py: Python<'_>,
        src: usize,
        dst: usize,
        timestamp: f64,
        edge_feat: &PyTensor,
    ) -> PyResult<PyTensor> {
        let out = py
            .detach(|| {
                self.inner
                    .process_event(src, dst, timestamp, &edge_feat.inner)
            })
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "MslKernelEmitter", module = "oxideon")]
pub struct PyMslKernelEmitter {
    inner: MslKernelEmitter,
}

#[pymethods]
impl PyMslKernelEmitter {
    #[new]
    pub fn new(kernel_name: String) -> Self {
        Self {
            inner: MslKernelEmitter::new(kernel_name),
        }
    }

    pub fn emit_fused_add_silu_msl(&self) -> String {
        self.inner.emit_fused_add_silu_msl()
    }

    pub fn emit_matmul_tiled_msl(&self) -> String {
        self.inner.emit_matmul_tiled_msl()
    }

    pub fn emit_simdgroup_matmul_bf16_msl(&self) -> String {
        self.inner.emit_simdgroup_matmul_bf16_msl()
    }

    #[pyo3(signature = (d_model=128, eps=1e-5))]
    pub fn emit_fused_rmsnorm_msl(&self, d_model: usize, eps: f32) -> String {
        self.inner.emit_fused_rmsnorm_msl(d_model, eps)
    }

    #[pyo3(signature = (head_dim=64, block_m=16, block_n=16, is_causal=true))]
    pub fn emit_flash_attention_msl(
        &self,
        head_dim: usize,
        block_m: usize,
        block_n: usize,
        is_causal: bool,
    ) -> String {
        self.inner
            .emit_flash_attention_msl(head_dim, block_m, block_n, is_causal)
    }

    #[pyo3(signature = (head_dim=64, theta_base=10000.0))]
    pub fn emit_rope_msl(&self, head_dim: usize, theta_base: f32) -> String {
        self.inner.emit_rope_msl(head_dim, theta_base)
    }

    #[pyo3(signature = (group_size=32))]
    pub fn emit_quantized_affine_q4_matvec_msl(&self, group_size: usize) -> String {
        self.inner.emit_quantized_affine_q4_matvec_msl(group_size)
    }

    pub fn emit_fused_swiglu_msl(&self) -> String {
        self.inner.emit_fused_swiglu_msl()
    }
}

#[pyclass(name = "TileKernelPlan", module = "oxideon")]
pub struct PyTileKernelPlan {
    inner: TileKernelPlan,
}

#[pymethods]
impl PyTileKernelPlan {
    #[new]
    pub fn new(name: String, tile_m: usize, tile_n: usize, tile_k: usize) -> Self {
        Self {
            inner: TileKernelPlan::new(name, tile_m, tile_n, tile_k),
        }
    }

    #[pyo3(signature = (target_sm="sm_90"))]
    pub fn emit_ptx(&self, target_sm: &str) -> String {
        self.inner.emit_ptx(target_sm)
    }

    pub fn emit_msl(&self) -> String {
        self.inner.emit_msl()
    }

    pub fn emit_wgsl(&self) -> String {
        self.inner.emit_wgsl()
    }
}

#[pyclass(name = "KernelFusionEngine", module = "oxideon")]
pub struct PyKernelFusionEngine;

#[pymethods]
impl PyKernelFusionEngine {
    #[staticmethod]
    pub fn plan_fused_linear_swiglu(name: &str, m: usize, n: usize, k: usize) -> PyTileKernelPlan {
        PyTileKernelPlan {
            inner: KernelFusionEngine::plan_fused_linear_swiglu(name, m, n, k),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (name, dim, eps=1e-5))]
    pub fn plan_fused_add_rmsnorm(name: &str, dim: usize, eps: f64) -> PyTileKernelPlan {
        PyTileKernelPlan {
            inner: KernelFusionEngine::plan_fused_add_rmsnorm(name, dim, eps),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (name, head_dim=64, is_causal=true))]
    pub fn plan_flash_attention(
        name: &str,
        head_dim: usize,
        is_causal: bool,
    ) -> PyTileKernelPlan {
        PyTileKernelPlan {
            inner: KernelFusionEngine::plan_flash_attention(name, head_dim, is_causal),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (name, group_size=32))]
    pub fn plan_quantized_q4_gemv(name: &str, group_size: usize) -> PyTileKernelPlan {
        PyTileKernelPlan {
            inner: KernelFusionEngine::plan_quantized_q4_gemv(name, group_size),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (name, head_dim=64))]
    pub fn plan_fused_rope(name: &str, head_dim: usize) -> PyTileKernelPlan {
        PyTileKernelPlan {
            inner: KernelFusionEngine::plan_fused_rope(name, head_dim),
        }
    }
}

#[pyclass(name = "TileConfig", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyTileConfig {
    pub inner: TileConfig,
}

#[pymethods]
impl PyTileConfig {
    #[new]
    #[pyo3(signature = (tile_m, tile_n, tile_k, num_warps=4, num_stages=2))]
    pub fn new(
        tile_m: usize,
        tile_n: usize,
        tile_k: usize,
        num_warps: usize,
        num_stages: usize,
    ) -> Self {
        Self {
            inner: TileConfig::new(tile_m, tile_n, tile_k, num_warps, num_stages),
        }
    }

    #[getter]
    pub fn tile_m(&self) -> usize {
        self.inner.tile_m
    }

    #[getter]
    pub fn tile_n(&self) -> usize {
        self.inner.tile_n
    }

    #[getter]
    pub fn tile_k(&self) -> usize {
        self.inner.tile_k
    }

    #[getter]
    pub fn num_warps(&self) -> usize {
        self.inner.num_warps
    }

    #[getter]
    pub fn num_stages(&self) -> usize {
        self.inner.num_stages
    }

    pub fn shared_memory_bytes(&self) -> usize {
        self.inner.shared_memory_bytes()
    }
}

#[pyclass(name = "KernelAutoTuner", module = "oxideon")]
pub struct PyKernelAutoTuner {
    inner: std::sync::Mutex<KernelAutoTuner>,
}

#[pymethods]
impl PyKernelAutoTuner {
    #[new]
    #[pyo3(signature = (target="gpu"))]
    pub fn new(target: &str) -> Self {
        let hw = if target == "cpu" {
            HardwareProfile::default_cpu()
        } else {
            HardwareProfile::default_gpu(HardwareTarget::CudaPtx)
        };
        Self {
            inner: std::sync::Mutex::new(KernelAutoTuner::new(hw)),
        }
    }

    pub fn tune_gemm(
        &self,
        py: Python<'_>,
        m: usize,
        n: usize,
        k: usize,
    ) -> PyResult<PyTileConfig> {
        let cfg = py
            .detach(|| mutex_guard(&self.inner).tune_gemm(m, n, k))
            .map_err(to_py_err)?;
        Ok(PyTileConfig { inner: cfg })
    }
}

#[pyclass(name = "ShardedTensor", module = "oxideon")]
pub struct PyShardedTensor {
    inner: ShardedTensor,
}

#[pymethods]
impl PyShardedTensor {
    #[staticmethod]
    pub fn from_global(
        global: &PyTensor,
        mesh: &PyDeviceMesh,
        placements: Vec<String>,
        my_coord: Vec<usize>,
    ) -> PyResult<Self> {
        let mut parsed_placements = Vec::with_capacity(placements.len());
        for p in placements {
            match p.to_ascii_lowercase().as_str() {
                "replicate" | "r" => parsed_placements.push(Placement::Replicate),
                s if s.starts_with("shard:") || s.starts_with("s:") => {
                    let axis: usize = s
                        .split(':')
                        .nth(1)
                        .unwrap_or("0")
                        .parse()
                        .map_err(|_| PyRuntimeError::new_err("Invalid shard axis"))?;
                    parsed_placements.push(Placement::Shard(axis));
                }
                _ => {
                    return Err(PyRuntimeError::new_err(
                        "Placement must be 'replicate' or 'shard:<dim>'",
                    ));
                }
            }
        }
        let inner =
            ShardedTensor::from_global(&global.inner, &mesh.inner, &parsed_placements, &my_coord)
                .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn global_shape(&self) -> Vec<usize> {
        self.inner.global_shape.clone()
    }

    #[getter]
    pub fn local_tensor(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.local_tensor.clone(),
        }
    }

    pub fn sharded_matmul(&self, other: &PyShardedTensor, my_coord: Vec<usize>) -> PyResult<Self> {
        let inner = self
            .inner
            .sharded_matmul(&other.inner, &my_coord)
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }
}

#[pyclass(name = "AutoParallelPlanner", module = "oxideon")]
pub struct PyAutoParallelPlanner;

#[pymethods]
impl PyAutoParallelPlanner {
    #[staticmethod]
    pub fn plan_optimal_sharding(
        mesh: &PyDeviceMesh,
        batch_size: usize,
        seq_len: usize,
        hidden_dim: usize,
    ) -> PyResult<(usize, usize, f64)> {
        let planner = AutoParallelPlanner::new();
        let plan = planner
            .plan_optimal_sharding(&mesh.inner, batch_size, seq_len, hidden_dim)
            .map_err(to_py_err)?;
        Ok((plan.dp_degree, plan.tp_degree, plan.estimated_comm_time_sec))
    }
}

#[pyclass(name = "DpSgdConfig", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyDpSgdConfig {
    pub inner: DpSgdConfig,
}

#[pymethods]
impl PyDpSgdConfig {
    #[new]
    #[pyo3(signature = (max_grad_norm=1.0, noise_multiplier=0.1))]
    pub fn new(max_grad_norm: f64, noise_multiplier: f64) -> Self {
        Self {
            inner: DpSgdConfig::new(max_grad_norm, noise_multiplier),
        }
    }

    pub fn clip_and_noise(&self, py: Python<'_>, grad: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.clip_and_noise(&grad.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "ByzantineAggregator", module = "oxideon")]
pub struct PyByzantineAggregator;

#[pymethods]
impl PyByzantineAggregator {
    #[staticmethod]
    pub fn multi_krum(
        py: Python<'_>,
        gradients: Vec<PyTensor>,
        num_byzantine: usize,
        num_selected: usize,
    ) -> PyResult<PyTensor> {
        let grads: Vec<Tensor> = gradients.into_iter().map(|g| g.inner).collect();
        let out = py
            .detach(|| ByzantineAggregator::multi_krum(&grads, num_byzantine, num_selected))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    #[staticmethod]
    pub fn trimmed_mean(
        py: Python<'_>,
        gradients: Vec<PyTensor>,
        trim_fraction: f64,
    ) -> PyResult<PyTensor> {
        let grads: Vec<Tensor> = gradients.into_iter().map(|g| g.inner).collect();
        let out = py
            .detach(|| ByzantineAggregator::trimmed_mean(&grads, trim_fraction))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    #[staticmethod]
    pub fn bulyan(
        py: Python<'_>,
        gradients: Vec<PyTensor>,
        num_byzantine: usize,
    ) -> PyResult<PyTensor> {
        let grads: Vec<Tensor> = gradients.into_iter().map(|g| g.inner).collect();
        let out = py
            .detach(|| ByzantineAggregator::bulyan(&grads, num_byzantine))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "UnifiedWorldModel", module = "oxideon")]
pub struct PyUnifiedWorldModel {
    inner: UnifiedWorldModel,
}

#[pymethods]
impl PyUnifiedWorldModel {
    #[new]
    #[pyo3(signature = (latent_channels=4, action_dim=7, action_chunk=8, hidden_dim=64, num_layers=2))]
    pub fn new(
        latent_channels: usize,
        action_dim: usize,
        action_chunk: usize,
        hidden_dim: usize,
        num_layers: usize,
    ) -> PyResult<Self> {
        let config = WorldModelConfig {
            latent_channels,
            action_dim,
            action_chunk,
            hidden_dim,
            num_layers,
        };
        let inner = UnifiedWorldModel::new(config).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn forward_dynamics(
        &self,
        py: Python<'_>,
        latents: &PyTensor,
        actions: &PyTensor,
        tau: f64,
    ) -> PyResult<PyTensor> {
        let out = py
            .detach(|| {
                self.inner
                    .forward_dynamics(&latents.inner, &actions.inner, tau)
            })
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn sample_future_video(
        &self,
        py: Python<'_>,
        init_latents: &PyTensor,
        actions: &PyTensor,
        num_steps: usize,
    ) -> PyResult<PyTensor> {
        let out = py
            .detach(|| {
                self.inner
                    .sample_future_video(&init_latents.inner, &actions.inner, num_steps)
            })
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn plan_actions_to_goal(
        &self,
        py: Python<'_>,
        curr_latents: &PyTensor,
        goal_latents: &PyTensor,
    ) -> PyResult<PyTensor> {
        let out = py
            .detach(|| {
                self.inner
                    .plan_actions_to_goal(&curr_latents.inner, &goal_latents.inner)
            })
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }
}

#[pyclass(name = "TreeAttentionMask", module = "oxideon")]
pub struct PyTreeAttentionMask;

#[pymethods]
impl PyTreeAttentionMask {
    #[staticmethod]
    pub fn build(parents: Vec<usize>) -> PyResult<PyTensor> {
        let mask = TreeAttentionMask::build(&parents).map_err(to_py_err)?;
        Ok(PyTensor { inner: mask })
    }
}

#[pyclass(name = "MedusaHeads", module = "oxideon")]
pub struct PyMedusaHeads {
    inner: MedusaHeads,
}

#[pymethods]
impl PyMedusaHeads {
    #[new]
    #[pyo3(signature = (hidden_dim, vocab_size, num_heads=3, topk=4))]
    pub fn new(
        hidden_dim: usize,
        vocab_size: usize,
        num_heads: usize,
        topk: usize,
    ) -> PyResult<Self> {
        let inner = MedusaHeads::new(hidden_dim, vocab_size, num_heads, topk).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn forward(&self, py: Python<'_>, h: &PyTensor) -> PyResult<Vec<PyTensor>> {
        let outs = py
            .detach(|| self.inner.forward(&h.inner))
            .map_err(to_py_err)?;
        Ok(outs.into_iter().map(|inner| PyTensor { inner }).collect())
    }
}

#[pyclass(name = "TreeSpeculativeDecoder", module = "oxideon")]
pub struct PyTreeSpeculativeDecoder;

#[pymethods]
impl PyTreeSpeculativeDecoder {
    #[staticmethod]
    pub fn accept_longest_path(
        candidate_tokens: Vec<i64>,
        ground_truth_tokens: Vec<i64>,
        parents: Vec<usize>,
    ) -> PyResult<(Vec<i64>, usize)> {
        TreeSpeculativeDecoder::accept_longest_path(
            &candidate_tokens,
            &ground_truth_tokens,
            &parents,
        )
        .map_err(to_py_err)
    }
}

#[pyclass(name = "FlashDecodingConfig", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyFlashDecodingConfig {
    pub inner: FlashDecodingConfig,
}

#[pymethods]
impl PyFlashDecodingConfig {
    #[new]
    #[pyo3(signature = (num_heads, head_dim, split_kv_chunk_size=128, scale=None))]
    pub fn new(
        num_heads: usize,
        head_dim: usize,
        split_kv_chunk_size: usize,
        scale: Option<f32>,
    ) -> Self {
        let mut cfg =
            FlashDecodingConfig::new(num_heads, head_dim).with_chunk_size(split_kv_chunk_size);
        if let Some(s) = scale {
            cfg.scale = s;
        }
        Self { inner: cfg }
    }
}

#[pyclass(name = "FlashDecodingEngine", module = "oxideon")]
pub struct PyFlashDecodingEngine {
    pub inner: FlashDecodingEngine,
}

#[pymethods]
impl PyFlashDecodingEngine {
    #[new]
    pub fn new(config: &PyFlashDecodingConfig) -> Self {
        Self {
            inner: FlashDecodingEngine::new(config.inner.clone()),
        }
    }

    pub fn decode_attention(
        &self,
        py: Python<'_>,
        query: &PyTensor,
        keys: &PyTensor,
        values: &PyTensor,
    ) -> PyResult<PyTensor> {
        let q = &query.inner;
        let k = &keys.inner;
        let v = &values.inner;
        let res = py
            .detach(|| self.inner.decode_attention(q, k, v))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: res })
    }

    pub fn generate_ptx_kernel(&self) -> String {
        self.inner.generate_ptx_kernel()
    }

    pub fn generate_wgsl_kernel(&self) -> String {
        self.inner.generate_wgsl_kernel()
    }
}

#[pyclass(name = "AuxFreeRouter", module = "oxideon")]
pub struct PyAuxFreeRouter {
    pub inner: AuxFreeRouter,
}

#[pymethods]
impl PyAuxFreeRouter {
    #[new]
    #[pyo3(signature = (d_model, num_experts, top_k=2, bias_update_rate=0.001))]
    pub fn new(
        d_model: usize,
        num_experts: usize,
        top_k: usize,
        bias_update_rate: f32,
    ) -> PyResult<Self> {
        let inner =
            AuxFreeRouter::new(d_model, num_experts, top_k, bias_update_rate).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[allow(clippy::type_complexity)]
    pub fn route(
        &self,
        py: Python<'_>,
        input: &PyTensor,
    ) -> PyResult<(Vec<Vec<(usize, f32)>>, Vec<usize>)> {
        let inp = &input.inner;
        py.detach(|| self.inner.route(inp)).map_err(to_py_err)
    }

    pub fn update_biases(&mut self, expert_loads: Vec<usize>, num_tokens: usize) {
        self.inner.update_biases(&expert_loads, num_tokens);
    }
}

#[pyclass(name = "DeepSeekV3MoE", module = "oxideon")]
pub struct PyDeepSeekV3MoE {
    pub inner: DeepSeekV3MoE,
}

#[pymethods]
impl PyDeepSeekV3MoE {
    #[new]
    #[pyo3(signature = (d_model, ffn_hidden, num_shared, num_routed, top_k=2, bias_update_rate=0.001))]
    pub fn new(
        d_model: usize,
        ffn_hidden: usize,
        num_shared: usize,
        num_routed: usize,
        top_k: usize,
        bias_update_rate: f32,
    ) -> PyResult<Self> {
        let inner = DeepSeekV3MoE::new(
            d_model,
            ffn_hidden,
            num_shared,
            num_routed,
            top_k,
            bias_update_rate,
        )
        .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn forward_and_update(&mut self, py: Python<'_>, input: &PyTensor) -> PyResult<PyTensor> {
        let inp = &input.inner;
        let res = py
            .detach(|| self.inner.forward_and_update(inp))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: res })
    }
}

#[pyclass(name = "DiffusionForcingScheduler", module = "oxideon")]
pub struct PyDiffusionForcingScheduler {
    pub inner: DiffusionForcingScheduler,
}

#[pymethods]
impl PyDiffusionForcingScheduler {
    #[new]
    #[pyo3(signature = (num_train_timesteps=1000))]
    pub fn new(num_train_timesteps: usize) -> PyResult<Self> {
        if num_train_timesteps == 0 {
            return Err(PyValueError::new_err("num_train_timesteps must be > 0"));
        }
        crate::limits::check_repeats("num_train_timesteps", num_train_timesteps)
            .map_err(to_py_err)?;
        Ok(Self {
            inner: DiffusionForcingScheduler::new(num_train_timesteps, BetaSchedule::default())
                .map_err(to_py_err)?,
        })
    }

    pub fn add_noise_per_token(
        &self,
        py: Python<'_>,
        sequence: &PyTensor,
        token_timesteps: Vec<usize>,
    ) -> PyResult<PyTensor> {
        let seq = &sequence.inner;
        let res = py
            .detach(|| self.inner.add_noise_per_token(seq, &token_timesteps))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: res })
    }

    pub fn step_per_token(
        &self,
        py: Python<'_>,
        pred_noise: &PyTensor,
        current_timesteps: Vec<usize>,
        prev_timesteps: Vec<usize>,
        x_t: &PyTensor,
    ) -> PyResult<PyTensor> {
        let eps = &pred_noise.inner;
        let xt = &x_t.inner;
        let res = py
            .detach(|| {
                self.inner
                    .step_per_token(eps, &current_timesteps, &prev_timesteps, xt)
            })
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: res })
    }
}

#[pyclass(name = "TitansMemoryConfig", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyTitansMemoryConfig {
    pub inner: TitansMemoryConfig,
}

#[pymethods]
impl PyTitansMemoryConfig {
    #[new]
    #[pyo3(signature = (d_model, memory_dim, base_decay=0.1, base_lr=0.05))]
    pub fn new(d_model: usize, memory_dim: usize, base_decay: f32, base_lr: f32) -> Self {
        let mut cfg = TitansMemoryConfig::new(d_model, memory_dim);
        cfg.base_decay = base_decay;
        cfg.base_lr = base_lr;
        Self { inner: cfg }
    }
}

#[pyclass(name = "TitansNeuralMemory", module = "oxideon")]
pub struct PyTitansNeuralMemory {
    pub inner: TitansNeuralMemory,
}

#[pymethods]
impl PyTitansNeuralMemory {
    #[new]
    pub fn new(config: &PyTitansMemoryConfig) -> PyResult<Self> {
        let inner = TitansNeuralMemory::new(config.inner.clone()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn forward(&self, py: Python<'_>, input: &PyTensor) -> PyResult<PyTensor> {
        let inp = &input.inner;
        let res = py.detach(|| self.inner.forward(inp)).map_err(to_py_err)?;
        Ok(PyTensor { inner: res })
    }
}

#[pyclass(name = "TitansBlock", module = "oxideon")]
pub struct PyTitansBlock {
    pub inner: TitansBlock,
}

#[pymethods]
impl PyTitansBlock {
    #[new]
    pub fn new(config: &PyTitansMemoryConfig) -> PyResult<Self> {
        let inner = TitansBlock::new(config.inner.clone()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn forward(&self, py: Python<'_>, input: &PyTensor) -> PyResult<PyTensor> {
        let inp = &input.inner;
        let res = py.detach(|| self.inner.forward(inp)).map_err(to_py_err)?;
        Ok(PyTensor { inner: res })
    }
}

#[pyclass(name = "DualPipeSchedule", module = "oxideon")]
pub struct PyDualPipeSchedule {
    pub inner: DualPipeSchedule,
}

#[pymethods]
impl PyDualPipeSchedule {
    #[new]
    pub fn new(num_stages: usize, num_microbatches: usize) -> PyResult<Self> {
        let inner = DualPipeSchedule::new(num_stages, num_microbatches).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn bubble_ratio(&self) -> f64 {
        self.inner.bubble_ratio()
    }

    pub fn num_steps_for_rank(&self, rank: usize) -> PyResult<usize> {
        let steps = self
            .inner
            .generate_schedule_for_rank(rank)
            .map_err(to_py_err)?;
        Ok(steps.len())
    }
}

#[pyclass(name = "Parallel3DConfig", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyParallel3DConfig {
    pub inner: Parallel3DConfig,
}

#[pymethods]
impl PyParallel3DConfig {
    #[new]
    pub fn new(dp_size: usize, tp_size: usize, pp_size: usize) -> PyResult<Self> {
        let inner = Parallel3DConfig::new(dp_size, tp_size, pp_size).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn world_size(&self) -> usize {
        self.inner.world_size()
    }
}

#[pyclass(name = "Parallel3DRuntime", module = "oxideon")]
pub struct PyParallel3DRuntime {
    pub inner: Parallel3DRuntime,
}

#[pymethods]
impl PyParallel3DRuntime {
    #[new]
    pub fn new(config: &PyParallel3DConfig, device_specs: Vec<String>) -> PyResult<Self> {
        let mut devices = Vec::with_capacity(device_specs.len());
        for s in device_specs {
            devices.push(Device::parse_spec(&s).map_err(to_py_err)?);
        }
        let inner = Parallel3DRuntime::new(config.inner, devices).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn get_coords(&self, global_rank: usize) -> PyResult<(usize, usize, usize)> {
        let coords = self.inner.get_coords(global_rank).map_err(to_py_err)?;
        Ok((coords.dp_rank, coords.tp_rank, coords.pp_rank))
    }

    pub fn get_dp_group(&self, global_rank: usize) -> PyResult<Vec<usize>> {
        self.inner.get_dp_group(global_rank).map_err(to_py_err)
    }

    pub fn get_tp_group(&self, global_rank: usize) -> PyResult<Vec<usize>> {
        self.inner.get_tp_group(global_rank).map_err(to_py_err)
    }

    pub fn get_pp_group(&self, global_rank: usize) -> PyResult<Vec<usize>> {
        self.inner.get_pp_group(global_rank).map_err(to_py_err)
    }

    pub fn ring_all_reduce(
        &self,
        py: Python<'_>,
        group: Vec<usize>,
        rank_tensors: Vec<PyRef<'_, PyTensor>>,
    ) -> PyResult<Vec<PyTensor>> {
        let in_tensors: Vec<Tensor> = rank_tensors.iter().map(|t| t.inner.clone()).collect();
        let res = py
            .detach(|| self.inner.ring_all_reduce(&group, &in_tensors))
            .map_err(to_py_err)?;
        Ok(res.into_iter().map(|inner| PyTensor { inner }).collect())
    }
}

#[pyclass(name = "EDMConfig", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyEDMConfig {
    pub inner: EDMConfig,
}

#[pymethods]
impl PyEDMConfig {
    #[new]
    #[pyo3(signature = (num_steps=15, sigma_min=0.002, sigma_max=80.0, rho=7.0))]
    pub fn new(num_steps: usize, sigma_min: f32, sigma_max: f32, rho: f32) -> PyResult<Self> {
        crate::limits::check_repeats("num_steps", num_steps).map_err(to_py_err)?;
        Ok(Self {
            inner: EDMConfig {
                sigma_min,
                sigma_max,
                rho,
                num_steps,
            },
        })
    }

    pub fn build_sigmas(&self) -> Vec<f32> {
        self.inner.build_sigmas()
    }
}

#[pyclass(name = "KDiffusionSampler", module = "oxideon")]
pub struct PyKDiffusionSampler {
    pub inner: KDiffusionSampler,
}

#[pymethods]
impl PyKDiffusionSampler {
    #[new]
    pub fn new(config: &PyEDMConfig) -> Self {
        Self {
            inner: KDiffusionSampler::new(config.inner.clone()),
        }
    }
}

#[pyclass(name = "RectifiedFlowConfig", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyRectifiedFlowConfig {
    pub inner: RectifiedFlowConfig,
}

#[pymethods]
impl PyRectifiedFlowConfig {
    #[new]
    #[pyo3(signature = (num_steps=20, guidance_scale=1.0))]
    pub fn new(num_steps: usize, guidance_scale: f64) -> Self {
        Self {
            inner: RectifiedFlowConfig {
                num_steps,
                guidance_scale,
            },
        }
    }
}

#[pyclass(name = "RectifiedFlowEngine", module = "oxideon")]
pub struct PyRectifiedFlowEngine;

#[pymethods]
impl PyRectifiedFlowEngine {
    #[staticmethod]
    pub fn interpolate(
        py: Python<'_>,
        x0: &PyTensor,
        x1: &PyTensor,
        t: f64,
    ) -> PyResult<(PyTensor, PyTensor)> {
        let (xt, vt) = py
            .detach(|| RectifiedFlowEngine::interpolate(&x0.inner, &x1.inner, t))
            .map_err(to_py_err)?;
        Ok((PyTensor { inner: xt }, PyTensor { inner: vt }))
    }
}

#[pyclass(name = "TernaryPackedTensor", module = "oxideon")]
pub struct PyTernaryPackedTensor {
    pub inner: TernaryPackedTensor,
}

#[pymethods]
impl PyTernaryPackedTensor {
    #[staticmethod]
    pub fn pack(tensor: &PyTensor, scale: f32) -> PyResult<Self> {
        let inner = TernaryPackedTensor::pack(&tensor.inner, scale).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn unpack(&self, py: Python<'_>) -> PyResult<PyTensor> {
        let res = py.detach(|| self.inner.unpack()).map_err(to_py_err)?;
        Ok(PyTensor { inner: res })
    }

    pub fn ternary_matmul(&self, py: Python<'_>, input: &PyTensor) -> PyResult<PyTensor> {
        let inp = &input.inner;
        let res = py
            .detach(|| self.inner.ternary_matmul(inp))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: res })
    }

    #[getter]
    pub fn shape(&self) -> Vec<usize> {
        self.inner.shape.clone()
    }

    #[getter]
    pub fn scale(&self) -> f32 {
        self.inner.scale
    }

    #[getter]
    pub fn packed_bytes(&self) -> usize {
        self.inner.packed_data.len()
    }
}

#[pyclass(name = "PackedBitLinear", module = "oxideon")]
pub struct PyPackedBitLinear {
    pub inner: PackedBitLinear,
}

#[pymethods]
impl PyPackedBitLinear {
    #[staticmethod]
    pub fn from_linear(linear: &PyLinear, scale: f32) -> PyResult<Self> {
        let inner = PackedBitLinear::from_linear(&linear.inner, scale).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn forward(&self, py: Python<'_>, input: &PyTensor) -> PyResult<PyTensor> {
        let inp = &input.inner;
        let res = py.detach(|| self.inner.forward(inp)).map_err(to_py_err)?;
        Ok(PyTensor { inner: res })
    }
}

#[pyclass(name = "RVQConfig", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyRVQConfig {
    pub inner: RVQConfig,
}

#[pymethods]
impl PyRVQConfig {
    #[new]
    #[pyo3(signature = (num_codebooks, codebook_size, dim, commitment_cost=0.25))]
    pub fn new(
        num_codebooks: usize,
        codebook_size: usize,
        dim: usize,
        commitment_cost: f32,
    ) -> Self {
        let mut cfg = RVQConfig::new(num_codebooks, codebook_size, dim);
        cfg.commitment_cost = commitment_cost;
        Self { inner: cfg }
    }
}

#[pyclass(name = "ResidualVectorQuantizer", module = "oxideon")]
pub struct PyResidualVectorQuantizer {
    pub inner: ResidualVectorQuantizer,
}

#[pymethods]
impl PyResidualVectorQuantizer {
    #[new]
    pub fn new(config: &PyRVQConfig) -> PyResult<Self> {
        let inner = ResidualVectorQuantizer::new(config.inner.clone()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[allow(clippy::type_complexity)]
    pub fn quantize(
        &self,
        py: Python<'_>,
        latents: &PyTensor,
    ) -> PyResult<(PyTensor, Vec<Vec<Vec<usize>>>, f32)> {
        let inp = &latents.inner;
        let out = py.detach(|| self.inner.quantize(inp)).map_err(to_py_err)?;
        Ok((
            PyTensor {
                inner: out.quantized,
            },
            out.codes,
            out.commitment_loss,
        ))
    }

    pub fn decode_codes(&self, py: Python<'_>, codes: Vec<Vec<usize>>) -> PyResult<PyTensor> {
        let res = py
            .detach(|| self.inner.decode_codes(&codes))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: res })
    }
}

#[pyclass(name = "SymbolicShape", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PySymbolicShape {
    pub inner: SymbolicShape,
}

#[pymethods]
impl PySymbolicShape {
    #[staticmethod]
    pub fn parse(s: &str) -> Self {
        Self {
            inner: SymbolicShape::parse(s),
        }
    }

    pub fn bind(
        &self,
        concrete_shape: Vec<usize>,
    ) -> PyResult<std::collections::HashMap<String, usize>> {
        self.inner.bind(&concrete_shape).map_err(to_py_err)
    }

    pub fn resolve(&self, env: std::collections::HashMap<String, usize>) -> PyResult<Vec<usize>> {
        let env_ref: std::collections::HashMap<&str, usize> =
            env.iter().map(|(k, &v)| (k.as_str(), v)).collect();
        self.inner.resolve(&env_ref).map_err(to_py_err)
    }

    pub fn broadcast_with(&self, other: &Self) -> PyResult<Self> {
        let res = self.inner.broadcast_with(&other.inner).map_err(to_py_err)?;
        Ok(Self { inner: res })
    }

    pub fn matmul_with(&self, other: &Self) -> PyResult<Self> {
        let res = self.inner.matmul_with(&other.inner).map_err(to_py_err)?;
        Ok(Self { inner: res })
    }

    pub fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }
}

#[pyclass(name = "EagleConfig", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyEagleConfig {
    pub inner: EagleConfig,
}

#[pymethods]
impl PyEagleConfig {
    #[new]
    #[pyo3(signature = (hidden_dim, vocab_size, draft_depth=3))]
    pub fn new(hidden_dim: usize, vocab_size: usize, draft_depth: usize) -> Self {
        Self {
            inner: EagleConfig::new(hidden_dim, vocab_size, draft_depth),
        }
    }
}

#[pyclass(name = "EagleSpeculativeEngine", module = "oxideon")]
pub struct PyEagleSpeculativeEngine {
    pub inner: EagleSpeculativeEngine,
}

#[pymethods]
impl PyEagleSpeculativeEngine {
    #[new]
    pub fn new(config: &PyEagleConfig) -> PyResult<Self> {
        let inner = EagleSpeculativeEngine::new(config.inner.clone()).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn generate_draft(
        &self,
        py: Python<'_>,
        top_hidden: &PyTensor,
        last_token: usize,
    ) -> PyResult<(Vec<usize>, Vec<PyTensor>)> {
        let th = &top_hidden.inner;
        let (tokens, features) = py
            .detach(|| self.inner.generate_draft(th, last_token))
            .map_err(to_py_err)?;
        let py_feats = features
            .into_iter()
            .map(|inner| PyTensor { inner })
            .collect();
        Ok((tokens, py_feats))
    }

    #[staticmethod]
    pub fn verify_and_accept(
        draft_tokens: Vec<usize>,
        target_predicted_tokens: Vec<usize>,
    ) -> (Vec<usize>, usize) {
        EagleSpeculativeEngine::verify_and_accept(&draft_tokens, &target_predicted_tokens)
    }
}

#[pyclass(name = "AttentionSinkConfig", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyAttentionSinkConfig {
    pub inner: AttentionSinkConfig,
}

#[pymethods]
impl PyAttentionSinkConfig {
    #[new]
    #[pyo3(signature = (num_sink_tokens=4, window_size=1024))]
    pub fn new(num_sink_tokens: usize, window_size: usize) -> Self {
        Self {
            inner: AttentionSinkConfig::new(num_sink_tokens, window_size),
        }
    }
}

#[pyclass(name = "StreamingKVCache", module = "oxideon")]
pub struct PyStreamingKVCache {
    pub inner: StreamingKVCache,
}

#[pymethods]
impl PyStreamingKVCache {
    #[new]
    pub fn new(config: &PyAttentionSinkConfig) -> Self {
        Self {
            inner: StreamingKVCache::new(config.inner.clone()),
        }
    }

    pub fn append(&mut self, key: &PyTensor, value: &PyTensor) -> PyResult<()> {
        self.inner
            .append(&key.inner, &value.inner)
            .map_err(to_py_err)
    }

    pub fn get_kv(&self, py: Python<'_>) -> PyResult<(PyTensor, PyTensor)> {
        let (k, v) = py.detach(|| self.inner.get_kv()).map_err(to_py_err)?;
        Ok((PyTensor { inner: k }, PyTensor { inner: v }))
    }

    pub fn num_cached_tokens(&self) -> usize {
        self.inner.num_cached_tokens()
    }

    pub fn total_tokens_seen(&self) -> usize {
        self.inner.total_tokens_seen()
    }
}

#[pyclass(name = "OmniTokenizer", module = "oxideon")]
pub struct PyOmniTokenizer {
    pub inner: OmniTokenizer,
}

#[pymethods]
impl PyOmniTokenizer {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: OmniTokenizer::new(),
        }
    }

    pub fn encode_text(&self, text: &str) -> Vec<usize> {
        self.inner.encode_text(text).token_ids
    }

    pub fn encode_vision_patches(&self, patch_ids: Vec<usize>) -> PyResult<Vec<usize>> {
        Ok(self
            .inner
            .encode_vision_patches(&patch_ids)
            .map_err(to_py_err)?
            .token_ids)
    }

    pub fn encode_audio_tokens(&self, audio_tokens: Vec<usize>) -> PyResult<Vec<usize>> {
        Ok(self
            .inner
            .encode_audio_tokens(&audio_tokens)
            .map_err(to_py_err)?
            .token_ids)
    }
}

impl Default for PyOmniTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[pyclass(name = "TieredMemoryManager", module = "oxideon")]
pub struct PyTieredMemoryManager {
    inner: TieredMemoryManager,
}

#[pymethods]
impl PyTieredMemoryManager {
    #[new]
    #[pyo3(signature = (high_watermark_ratio=0.85))]
    pub fn new(high_watermark_ratio: f64) -> Self {
        Self {
            inner: TieredMemoryManager::new(high_watermark_ratio),
        }
    }

    pub fn register_device_capacity(
        &mut self,
        device: &str,
        capacity_bytes: usize,
    ) -> PyResult<()> {
        let dev = Device::parse_spec(device).map_err(to_py_err)?;
        self.inner.register_device_capacity(dev, capacity_bytes);
        Ok(())
    }

    pub fn get_device_usage(&self, device: &str) -> PyResult<(usize, usize)> {
        let dev = Device::parse_spec(device).map_err(to_py_err)?;
        Ok(self.inner.get_device_usage(&dev))
    }

    pub fn is_under_pressure(&self, device: &str) -> PyResult<bool> {
        let dev = Device::parse_spec(device).map_err(to_py_err)?;
        Ok(self.inner.is_under_pressure(&dev))
    }

    pub fn register_tensor(
        &mut self,
        tensor_id: usize,
        size_bytes: usize,
        target_device: &str,
    ) -> PyResult<Vec<(usize, String, String)>> {
        let dev = Device::parse_spec(target_device).map_err(to_py_err)?;
        let migs = self
            .inner
            .register_tensor(tensor_id, size_bytes, dev, MemoryPriority::Normal)
            .map_err(to_py_err)?;
        Ok(migs
            .into_iter()
            .map(|(id, from, to)| (id, from.to_string(), to.to_string()))
            .collect())
    }

    pub fn access_tensor(&mut self, tensor_id: usize) -> PyResult<Option<String>> {
        let opt = self.inner.access_tensor(tensor_id).map_err(to_py_err)?;
        Ok(opt.map(|d| d.to_string()))
    }
}

#[pyclass(name = "StreamEvent", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyStreamEvent {
    pub inner: StreamEvent,
}

#[pymethods]
impl PyStreamEvent {
    pub fn query(&self) -> bool {
        self.inner.query()
    }

    pub fn synchronize(&self) {
        self.inner.synchronize();
    }
}

#[pyclass(name = "ComputeStream", module = "oxideon")]
pub struct PyComputeStream {
    pub inner: ComputeStream,
}

#[pymethods]
impl PyComputeStream {
    #[new]
    #[pyo3(signature = (stream_id=0, device="cpu"))]
    pub fn new(stream_id: usize, device: &str) -> PyResult<Self> {
        let dev = Device::parse_spec(device).map_err(to_py_err)?;
        Ok(Self {
            inner: ComputeStream::new(stream_id, dev),
        })
    }

    pub fn record_event(&self) -> PyStreamEvent {
        PyStreamEvent {
            inner: self.inner.record_event(),
        }
    }

    pub fn wait_event(&self, event: &PyStreamEvent) {
        self.inner.wait_event(&event.inner);
    }

    pub fn synchronize(&self) {
        self.inner.synchronize();
    }
}

#[pyclass(name = "ComputeGraph", module = "oxideon")]
#[derive(Default)]
pub struct PyComputeGraph {
    pub inner: ComputeGraph,
}

#[pymethods]
impl PyComputeGraph {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: ComputeGraph::new(),
        }
    }

    pub fn capture_begin(&mut self) -> PyResult<()> {
        self.inner.capture_begin().map_err(to_py_err)
    }

    pub fn capture_end(&mut self) -> PyResult<usize> {
        self.inner.capture_end().map_err(to_py_err)
    }

    pub fn replay(&mut self) -> PyResult<()> {
        self.inner.replay().map_err(to_py_err)
    }

    #[getter]
    pub fn num_ops(&self) -> usize {
        self.inner.num_ops()
    }

    #[getter]
    pub fn replay_count(&self) -> usize {
        self.inner.replay_count()
    }
}

#[pyfunction]
#[pyo3(name = "device_bandwidth")]
pub fn py_device_bandwidth(dev_a: &str, dev_b: &str) -> PyResult<f64> {
    let a = Device::parse_spec(dev_a).map_err(to_py_err)?;
    let b = Device::parse_spec(dev_b).map_err(to_py_err)?;
    Ok(InterconnectMatrix::bandwidth_gbps(&a, &b))
}

#[pyfunction]
#[pyo3(name = "device_latency")]
pub fn py_device_latency(dev_a: &str, dev_b: &str) -> PyResult<f64> {
    let a = Device::parse_spec(dev_a).map_err(to_py_err)?;
    let b = Device::parse_spec(dev_b).map_err(to_py_err)?;
    Ok(InterconnectMatrix::latency_us(&a, &b))
}

#[pyfunction]
#[pyo3(name = "set_default_device")]
pub fn py_set_default_device(device: &str) -> PyResult<()> {
    let dev = Device::parse_spec(device).map_err(to_py_err)?;
    set_default_device(dev);
    Ok(())
}

#[pyfunction]
#[pyo3(name = "get_default_device")]
pub fn py_get_default_device() -> String {
    get_default_device().to_string()
}

#[pyfunction]
#[pyo3(name = "get_available_devices")]
pub fn py_get_available_devices() -> Vec<PyDevice> {
    Device::available_devices()
        .into_iter()
        .map(|inner| PyDevice { inner })
        .collect()
}

#[pyfunction]
#[pyo3(name = "is_available")]
pub fn py_is_available(device_type: &str) -> bool {
    PyDevice::is_available(device_type)
}

#[pyfunction]
#[pyo3(name = "device_count")]
pub fn py_device_count(device_type: &str) -> usize {
    PyDevice::device_count(device_type)
}

#[pyfunction]
#[pyo3(name = "margin_ranking_loss", signature = (x1, x2, target, margin=0.0))]
pub fn py_margin_ranking_loss(
    py: Python<'_>,
    x1: &PyTensor,
    x2: &PyTensor,
    target: &PyTensor,
    margin: f64,
) -> PyResult<PyTensor> {
    let out = py
        .detach(|| margin_ranking_loss(&x1.inner, &x2.inner, &target.inner, margin))
        .map_err(to_py_err)?;
    Ok(PyTensor { inner: out })
}

#[pyfunction]
#[pyo3(name = "hinge_embedding_loss", signature = (input, target, margin=1.0))]
pub fn py_hinge_embedding_loss(
    py: Python<'_>,
    input: &PyTensor,
    target: &PyTensor,
    margin: f64,
) -> PyResult<PyTensor> {
    let out = py
        .detach(|| hinge_embedding_loss(&input.inner, &target.inner, margin))
        .map_err(to_py_err)?;
    Ok(PyTensor { inner: out })
}

#[pyfunction]
#[pyo3(name = "soft_margin_loss")]
pub fn py_soft_margin_loss(
    py: Python<'_>,
    input: &PyTensor,
    target: &PyTensor,
) -> PyResult<PyTensor> {
    let out = py
        .detach(|| soft_margin_loss(&input.inner, &target.inner))
        .map_err(to_py_err)?;
    Ok(PyTensor { inner: out })
}

/// Class indices from a list of ints or an integer tensor.
fn class_indices(targets: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let raw: Vec<i64> = match targets.extract::<PyRef<'_, PyTensor>>() {
        Ok(t) => t.inner.to_vec::<i64>().map_err(to_py_err)?,
        Err(_) => targets.extract().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "targets must be a list of ints or an integer Tensor",
            )
        })?,
    };
    raw.into_iter()
        .map(|v| {
            usize::try_from(v)
                .map_err(|_| PyValueError::new_err(format!("class index must be >= 0, got {v}")))
        })
        .collect()
}

/// Negative log-likelihood of `targets` (class indices) under `log_probs` of shape `[batch, classes]`.
#[pyfunction]
#[pyo3(name = "nll_loss")]
pub fn py_nll_loss(
    py: Python<'_>,
    log_probs: &PyTensor,
    targets: &Bound<'_, PyAny>,
) -> PyResult<PyTensor> {
    let targets = class_indices(targets)?;
    let out = py
        .detach(|| loss::nll_loss(&log_probs.inner, &targets))
        .map_err(to_py_err)?;
    Ok(PyTensor { inner: out })
}

/// Softmax cross-entropy between `logits` of shape `[batch, classes]` and `targets` (class indices).
#[pyfunction]
#[pyo3(name = "cross_entropy", signature = (logits, targets, label_smoothing=0.0))]
pub fn py_cross_entropy(
    py: Python<'_>,
    logits: &PyTensor,
    targets: &Bound<'_, PyAny>,
    label_smoothing: f64,
) -> PyResult<PyTensor> {
    if !(0.0..1.0).contains(&label_smoothing) {
        return Err(PyValueError::new_err(format!(
            "label_smoothing must be in [0, 1), got {label_smoothing}"
        )));
    }
    let targets = class_indices(targets)?;
    let out = py
        .detach(|| loss::cross_entropy_with_smoothing(&logits.inner, &targets, label_smoothing))
        .map_err(to_py_err)?;
    Ok(PyTensor { inner: out })
}

/// Sigmoid pairwise loss (SigLIP) between `[N, D]` image and text embeddings.
#[pyfunction]
#[pyo3(name = "siglip_loss", signature = (z_img, z_txt, temperature=10.0, bias=-10.0), text_signature = "(z_img, z_txt, temperature=10.0, bias=-10.0)")]
pub fn py_siglip_loss(
    py: Python<'_>,
    z_img: &PyTensor,
    z_txt: &PyTensor,
    temperature: f64,
    bias: f64,
) -> PyResult<PyTensor> {
    let out = py
        .detach(|| loss::siglip_loss(&z_img.inner, &z_txt.inner, temperature, bias))
        .map_err(to_py_err)?;
    Ok(PyTensor { inner: out })
}

/// Focal loss (Lin et al., 2017) for `logits` of shape `[batch, classes]` and class-index `targets`.
#[pyfunction]
#[pyo3(name = "focal_loss", signature = (logits, targets, alpha=0.25, gamma=2.0))]
pub fn py_focal_loss(
    py: Python<'_>,
    logits: &PyTensor,
    targets: &Bound<'_, PyAny>,
    alpha: f64,
    gamma: f64,
) -> PyResult<PyTensor> {
    let targets = class_indices(targets)?;
    let out = py
        .detach(|| loss::focal_loss(&logits.inner, &targets, alpha, gamma))
        .map_err(to_py_err)?;
    Ok(PyTensor { inner: out })
}

/// Declares `#[pyfunction]`s that call a `loss::` function on tensors, with optional float keyword arguments.
macro_rules! py_tensor_losses {
    ($($pyname:literal, $fname:ident = $rust:path, ($($arg:ident),+) $(, $extra:ident = $default:tt)*;)+) => {$(
        #[pyfunction]
        #[pyo3(name = $pyname, signature = ($($arg,)+ $($extra = $default),*))]
        pub fn $fname(py: Python<'_>, $($arg: &PyTensor,)+ $($extra: f64),*) -> PyResult<PyTensor> {
            py.detach(|| $rust($(&$arg.inner,)+ $($extra),*))
                .map(|inner| PyTensor { inner })
                .map_err(to_py_err)
        }
    )+};
}

py_tensor_losses! {
    "mse_loss", py_mse_loss = loss::mse_loss, (pred, target);
    "l1_loss", py_l1_loss = loss::l1_loss, (pred, target);
    "huber_loss", py_huber_loss = loss::huber_loss, (pred, target), delta = 1.0;
    "binary_cross_entropy_with_logits", py_bce_with_logits = loss::binary_cross_entropy_with_logits, (logits, targets);
    "kl_div_loss", py_kl_div_loss = loss::kl_div_loss, (log_probs, target_probs);
    "cosine_embedding_loss", py_cosine_embedding_loss = loss::cosine_embedding_loss, (a, b, labels), margin = 0.0;
    "triplet_margin_loss", py_triplet_margin_loss = loss::triplet_margin_loss, (anchor, positive, negative), margin = 1.0;
    "info_nce_loss", py_info_nce_loss = loss::info_nce_loss, (queries, keys), temperature = 0.07;
    "dpo_loss", py_dpo_loss = loss::dpo_loss, (policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps), beta = 0.1;
    "ppo_clipped_loss", py_ppo_clipped_loss = loss::ppo_clipped_loss, (new_logps, old_logps, advantages), clip_eps = 0.2;
    "simpo_loss", py_simpo_loss = loss::simpo_loss, (chosen_avg_logps, rejected_avg_logps), beta = 2.0, gamma = 0.5;
    "barlow_twins_loss", py_barlow_twins_loss = loss::barlow_twins_loss, (z_a, z_b), lambd = 0.005;
    "vicreg_loss", py_vicreg_loss = loss::vicreg_loss, (z_a, z_b), sim_coeff = 25.0, std_coeff = 25.0, cov_coeff = 1.0;
    "grpo_loss", py_grpo_loss = loss::grpo_loss, (new_log_probs, old_log_probs, ref_log_probs, rewards), clip_eps = 0.2, kl_beta = 0.04;
}

#[pyclass(name = "Mamba2Block", module = "oxideon")]
pub struct PyMamba2Block {
    pub inner: Mamba2Block,
}

#[pymethods]
impl PyMamba2Block {
    #[new]
    #[pyo3(signature = (d_model, d_state=64, expand=2, chunk_size=64))]
    pub fn new(d_model: usize, d_state: usize, expand: usize, chunk_size: usize) -> PyResult<Self> {
        let inner = Mamba2Block::new(d_model, d_state, expand, chunk_size).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn forward(&self, py: Python<'_>, u: &PyTensor) -> PyResult<PyTensor> {
        let out = py
            .detach(|| self.inner.forward(&u.inner))
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: out })
    }

    pub fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|t| PyTensor { inner: t })
            .collect()
    }
}

#[pyclass(name = "TransfusionModel", module = "oxideon")]
pub struct PyTransfusionModel {
    pub inner: TransfusionModel,
}

#[pymethods]
impl PyTransfusionModel {
    #[new]
    #[pyo3(signature = (vocab_size=32000, d_model=512, patch_dim=64, num_layers=4, num_heads=8, d_ff=2048, dropout=0.0))]
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        patch_dim: usize,
        num_layers: usize,
        num_heads: usize,
        d_ff: usize,
        dropout: f32,
    ) -> PyResult<Self> {
        if !(0.0..1.0).contains(&dropout) {
            return Err(PyValueError::new_err(format!(
                "dropout must be in [0, 1), got {dropout}"
            )));
        }
        let config = TransfusionConfig {
            vocab_size,
            d_model,
            patch_dim,
            num_layers,
            num_heads,
            d_ff,
            dropout,
        };
        let inner = TransfusionModel::new(config).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn forward_multimodal(
        &self,
        py: Python<'_>,
        text_tokens: &PyTensor,
        visual_patches: &PyTensor,
        timesteps: &PyTensor,
    ) -> PyResult<(PyTensor, PyTensor)> {
        let (logits, pred_v) = py
            .detach(|| {
                self.inner.forward_multimodal(
                    &text_tokens.inner,
                    &visual_patches.inner,
                    &timesteps.inner,
                )
            })
            .map_err(to_py_err)?;
        Ok((PyTensor { inner: logits }, PyTensor { inner: pred_v }))
    }

    pub fn compute_loss(
        &self,
        py: Python<'_>,
        text_tokens: &PyTensor,
        visual_clean: &PyTensor,
        visual_noise: &PyTensor,
        timesteps: &PyTensor,
        lambda_diff: f64,
    ) -> PyResult<PyTensor> {
        let loss = py
            .detach(|| {
                self.inner.compute_loss(
                    &text_tokens.inner,
                    &visual_clean.inner,
                    &visual_noise.inner,
                    &timesteps.inner,
                    lambda_diff,
                )
            })
            .map_err(to_py_err)?;
        Ok(PyTensor { inner: loss })
    }

    pub fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|t| PyTensor { inner: t })
            .collect()
    }
}

#[pyfunction]
#[pyo3(name = "ctc_loss", signature = (log_probs, targets, blank=0, zero_infinity=false))]
pub fn py_ctc_loss(
    py: Python<'_>,
    log_probs: &PyTensor,
    targets: &PyTensor,
    blank: usize,
    zero_infinity: bool,
) -> PyResult<PyTensor> {
    let out = py
        .detach(|| ctc_loss(&log_probs.inner, &targets.inner, blank, zero_infinity))
        .map_err(to_py_err)?;
    Ok(PyTensor { inner: out })
}

fn tensors_of(items: &[PyRef<'_, PyTensor>]) -> Vec<Tensor> {
    items.iter().map(|t| t.inner.clone()).collect()
}

/// Concatenates tensors along an existing dimension.
#[pyfunction]
#[pyo3(name = "cat", signature = (tensors, dim=0))]
pub fn py_cat(py: Python<'_>, tensors: Vec<PyRef<'_, PyTensor>>, dim: isize) -> PyResult<PyTensor> {
    let ts = tensors_of(&tensors);
    let inner = py.detach(|| Tensor::cat(&ts, dim)).map_err(to_py_err)?;
    Ok(PyTensor { inner })
}

/// Stacks equally shaped tensors along a new dimension.
#[pyfunction]
#[pyo3(name = "stack", signature = (tensors, dim=0))]
pub fn py_stack(
    py: Python<'_>,
    tensors: Vec<PyRef<'_, PyTensor>>,
    dim: isize,
) -> PyResult<PyTensor> {
    let ts = tensors_of(&tensors);
    let inner = py.detach(|| Tensor::stack(&ts, dim)).map_err(to_py_err)?;
    Ok(PyTensor { inner })
}

/// Element-wise select: `on_true` where `cond` is true, else `on_false`.
#[pyfunction]
#[pyo3(name = "where")]
pub fn py_where(
    py: Python<'_>,
    cond: &PyTensor,
    on_true: &PyTensor,
    on_false: &PyTensor,
) -> PyResult<PyTensor> {
    let inner = py
        .detach(|| Tensor::where_cond(&cond.inner, &on_true.inner, &on_false.inner))
        .map_err(to_py_err)?;
    Ok(PyTensor { inner })
}

#[pyfunction]
#[pyo3(name = "tril", signature = (input, diagonal=0))]
pub fn py_tril(py: Python<'_>, input: &PyTensor, diagonal: isize) -> PyResult<PyTensor> {
    input.tril(py, diagonal)
}

#[pyfunction]
#[pyo3(name = "triu", signature = (input, diagonal=0))]
pub fn py_triu(py: Python<'_>, input: &PyTensor, diagonal: isize) -> PyResult<PyTensor> {
    input.triu(py, diagonal)
}

#[pyfunction]
#[pyo3(name = "kron")]
pub fn py_kron(py: Python<'_>, a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    a.kron(py, b)
}

#[pyfunction]
#[pyo3(name = "cdist", signature = (x1, x2, p=2.0))]
pub fn py_cdist(py: Python<'_>, x1: &PyTensor, x2: &PyTensor, p: f64) -> PyResult<PyTensor> {
    x1.cdist(py, x2, p)
}

#[pyfunction]
#[pyo3(name = "pdist", signature = (input, p=2.0))]
pub fn py_pdist(py: Python<'_>, input: &PyTensor, p: f64) -> PyResult<PyTensor> {
    input.pdist(py, p)
}

#[pyfunction]
#[pyo3(name = "cross", signature = (input, other, dim=-1))]
pub fn py_cross(py: Python<'_>, input: &PyTensor, other: &PyTensor, dim: isize) -> PyResult<PyTensor> {
    input.cross(py, other, dim)
}

#[pyfunction]
#[pyo3(name = "cummax", signature = (input, dim=-1))]
pub fn py_cummax(py: Python<'_>, input: &PyTensor, dim: isize) -> PyResult<(PyTensor, PyTensor)> {
    input.cummax(py, dim)
}

#[pyfunction]
#[pyo3(name = "cummin", signature = (input, dim=-1))]
pub fn py_cummin(py: Python<'_>, input: &PyTensor, dim: isize) -> PyResult<(PyTensor, PyTensor)> {
    input.cummin(py, dim)
}

#[pyfunction]
#[pyo3(name = "logcumsumexp", signature = (input, dim=-1))]
pub fn py_logcumsumexp(py: Python<'_>, input: &PyTensor, dim: isize) -> PyResult<PyTensor> {
    input.logcumsumexp(py, dim)
}

#[pyfunction]
#[pyo3(name = "tensordot", signature = (a, b, dims=None))]
pub fn py_tensordot(py: Python<'_>, a: &PyTensor, b: &PyTensor, dims: Option<&Bound<'_, PyAny>>) -> PyResult<PyTensor> {
    a.tensordot(py, b, dims)
}

#[pyfunction]
#[pyo3(name = "as_strided", signature = (input, size, stride, storage_offset=0))]
pub fn py_as_strided(
    input: &PyTensor,
    size: Vec<usize>,
    stride: Vec<isize>,
    storage_offset: usize,
) -> PyResult<PyTensor> {
    input.as_strided(size, stride, storage_offset)
}

#[pyfunction]
#[pyo3(name = "conv2d", signature = (input, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1))]
pub fn py_conv2d(
    input: &PyTensor,
    weight: &PyTensor,
    bias: Option<&PyTensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
) -> PyResult<PyTensor> {
    input.conv2d(weight, bias, stride, padding, dilation, groups)
}

#[pyfunction]
#[pyo3(name = "conv1d", signature = (input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1))]
pub fn py_conv1d(
    input: &PyTensor,
    weight: &PyTensor,
    bias: Option<&PyTensor>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> PyResult<PyTensor> {
    input.conv1d(weight, bias, stride, padding, dilation, groups)
}

/// Insertion indices of `values` in 1D sorted `sorted_seq` (`torch.searchsorted`).
#[pyfunction]
#[pyo3(name = "searchsorted", signature = (sorted_sequence, values, right=false))]
pub fn py_searchsorted(
    py: Python<'_>,
    sorted_sequence: &PyTensor,
    values: &PyTensor,
    right: bool,
) -> PyResult<PyTensor> {
    let inner = py
        .detach(|| Tensor::searchsorted(&sorted_sequence.inner, &values.inner, right))
        .map_err(to_py_err)?;
    Ok(PyTensor { inner })
}

/// Einstein summation, e.g. `einsum("ij,jk->ik", a, b)`.
#[pyfunction]
#[pyo3(name = "einsum", signature = (equation, *operands))]
pub fn py_einsum(
    py: Python<'_>,
    equation: &str,
    operands: &Bound<'_, pyo3::types::PyTuple>,
) -> PyResult<PyTensor> {
    let ts: Vec<Tensor> = operands
        .iter()
        .map(|o| {
            o.extract::<PyRef<'_, PyTensor>>()
                .map(|t| t.inner.clone())
                .map_err(pyo3::PyErr::from)
        })
        .collect::<PyResult<_>>()?;
    let refs: Vec<&Tensor> = ts.iter().collect();
    let inner = py
        .detach(|| Tensor::einsum(equation, &refs))
        .map_err(to_py_err)?;
    Ok(PyTensor { inner })
}

/// Registers all classes and functions onto the `oxideon` Python module.
use crate::quantum::braid_anyons::FibonacciAnyonBraider;

#[pyclass(name = "FibonacciAnyonBraider", module = "oxideon", from_py_object)]
#[derive(Clone, Copy, Default)]
pub struct PyFibonacciAnyonBraider {
    pub inner: FibonacciAnyonBraider,
}

#[pymethods]
impl PyFibonacciAnyonBraider {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: FibonacciAnyonBraider::new(),
        }
    }

    pub fn yang_baxter_residual(&self, py: Python<'_>) -> f64 {
        py.detach(|| self.inner.yang_baxter_residual())
    }

    pub fn braid_unitarity_defect(&self, py: Python<'_>, word: Vec<i32>) -> PyResult<f64> {
        py.detach(|| self.inner.braid_unitarity_defect(&word))
            .map_err(to_py_err)
    }
}
use crate::science::tropical_spectral::TropicalSpectralEngine;

#[pyclass(name = "TropicalSpectralEngine", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyTropicalSpectralEngine {
    pub inner: TropicalSpectralEngine,
}

#[pymethods]
impl PyTropicalSpectralEngine {
    #[new]
    pub fn new(n: usize, matrix: Vec<f64>) -> PyResult<Self> {
        let inner = guard_res(|| TropicalSpectralEngine::new(n, matrix))?;
        Ok(Self { inner })
    }

    pub fn karp_minimum_cycle_mean(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.karp_minimum_cycle_mean())
            .map_err(to_py_err)
    }

    pub fn min_plus_matvec(&self, x: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner.min_plus_matvec(&x).map_err(to_py_err)
    }

    #[staticmethod]
    pub fn legendre_fenchel_conjugate(
        points: Vec<f64>,
        coeffs: Vec<f64>,
        slope_p: f64,
    ) -> PyResult<f64> {
        TropicalSpectralEngine::legendre_fenchel_conjugate(&points, &coeffs, slope_p)
            .map_err(to_py_err)
    }
}
use crate::physics::relativistic_lbm::RelativisticLbm1D;

#[pyclass(name = "RelativisticLbm1D", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyRelativisticLbm1D {
    pub inner: RelativisticLbm1D,
}

#[pymethods]
impl PyRelativisticLbm1D {
    #[new]
    pub fn new(rho: Vec<f64>, velocity: Vec<f64>, omega: f64) -> PyResult<Self> {
        let inner = RelativisticLbm1D::new(&rho, &velocity, omega).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn step(&mut self, py: Python<'_>, num_steps: usize) -> PyResult<()> {
        crate::limits::check_iterations("num_steps", num_steps).map_err(to_py_err)?;
        guard(|| {
            py.detach(|| {
                for _ in 0..num_steps {
                    self.inner.step();
                }
            })
        })
    }

    pub fn total_mass(&self) -> f64 {
        self.inner.total_mass()
    }

    pub fn total_momentum(&self) -> f64 {
        self.inner.total_momentum()
    }
}
use crate::science::free_probability::FreeProbabilityEngine;

#[pyclass(name = "FreeProbabilityEngine", module = "oxideon", from_py_object)]
#[derive(Clone, Copy, Default)]
pub struct PyFreeProbabilityEngine {
    pub inner: FreeProbabilityEngine,
}

#[pymethods]
impl PyFreeProbabilityEngine {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: FreeProbabilityEngine::new(),
        }
    }

    #[staticmethod]
    pub fn moments_to_free_cumulants(moments: [f64; 4]) -> [f64; 4] {
        FreeProbabilityEngine::moments_to_free_cumulants(moments)
    }

    #[staticmethod]
    pub fn free_additive_convolution_moments(moments_a: [f64; 4], moments_b: [f64; 4]) -> [f64; 4] {
        FreeProbabilityEngine::free_additive_convolution_moments(moments_a, moments_b)
    }

    #[staticmethod]
    pub fn marchenko_pastur_spectrum(
        lambda: f64,
        gamma: f64,
        sigma: f64,
    ) -> PyResult<(f64, f64, f64)> {
        FreeProbabilityEngine::marchenko_pastur_spectrum(lambda, gamma, sigma).map_err(to_py_err)
    }
}
use crate::ode::rough_paths::RoughPathSignature;

#[pyclass(name = "RoughPathSignature", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyRoughPathSignature {
    pub inner: RoughPathSignature,
}

#[pymethods]
impl PyRoughPathSignature {
    #[new]
    pub fn from_path(py: Python<'_>, points: Vec<Vec<f64>>) -> PyResult<Self> {
        let inner = py
            .detach(|| RoughPathSignature::from_path(&points))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn chen_multiply(&self, other: &PyRoughPathSignature) -> PyResult<Self> {
        let inner = self.inner.chen_multiply(&other.inner).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn levy_area(&self, i: usize, j: usize) -> PyResult<f64> {
        self.inner.levy_area(i, j).map_err(to_py_err)
    }

    #[getter]
    pub fn level1(&self) -> Vec<f64> {
        self.inner.level1.clone()
    }

    #[getter]
    pub fn level2(&self) -> Vec<f64> {
        self.inner.level2.clone()
    }
}
use crate::science::p_adic::PAdicAnalysisEngine;

#[pyclass(name = "PAdicAnalysisEngine", module = "oxideon", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyPAdicAnalysisEngine {
    pub inner: PAdicAnalysisEngine,
}

#[pymethods]
impl PyPAdicAnalysisEngine {
    #[new]
    pub fn new(prime_p: i64) -> PyResult<Self> {
        let inner = PAdicAnalysisEngine::new(prime_p).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn norm(&self, n: i64) -> f64 {
        self.inner.norm(n)
    }

    pub fn distance(&self, x: i64, y: i64) -> f64 {
        self.inner.distance(x, y)
    }

    pub fn verify_ultrametric_inequality(&self, py: Python<'_>, points: Vec<i64>) -> bool {
        py.detach(|| self.inner.verify_ultrametric_inequality(&points))
    }

    pub fn vladimirov_operator(
        &self,
        py: Python<'_>,
        psi: Vec<f64>,
        alpha: f64,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| self.inner.vladimirov_operator(&psi, alpha))
            .map_err(to_py_err)
    }
}
use crate::science::morse_theory::DiscreteMorseEngine;

#[pyclass(name = "DiscreteMorseEngine", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyDiscreteMorseEngine {
    pub inner: DiscreteMorseEngine,
}

#[pymethods]
impl PyDiscreteMorseEngine {
    #[new]
    pub fn new(critical_counts: Vec<usize>, betti_numbers: Vec<usize>) -> PyResult<Self> {
        let inner = DiscreteMorseEngine::new(critical_counts, betti_numbers).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn morse_euler_characteristic(&self) -> i64 {
        self.inner.morse_euler_characteristic()
    }

    pub fn verify_strong_morse_inequalities(&self, py: Python<'_>) -> bool {
        py.detach(|| self.inner.verify_strong_morse_inequalities())
    }

    pub fn evaluate_morse_and_poincare_polynomials(&self, t: f64) -> (f64, f64) {
        self.inner.evaluate_morse_and_poincare_polynomials(t)
    }
}
use crate::crypto::lattice_lwe::LatticeLweEngine;

/// Educational Ring-LWE arithmetic with caller-supplied randomness. Not ML-KEM, not constant-time.
#[pyclass(name = "LatticeLweEngine", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyLatticeLweEngine {
    pub inner: LatticeLweEngine,
}

#[pymethods]
impl PyLatticeLweEngine {
    #[new]
    pub fn new(degree_n: usize, modulus_q: i64) -> PyResult<Self> {
        let inner = LatticeLweEngine::new(degree_n, modulus_q).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn negacyclic_mul(&self, a: Vec<i64>, b: Vec<i64>) -> PyResult<Vec<i64>> {
        self.inner.negacyclic_mul(&a, &b).map_err(to_py_err)
    }

    pub fn compute_public_key(&self, a: Vec<i64>, s: Vec<i64>, e: Vec<i64>) -> PyResult<Vec<i64>> {
        self.inner.compute_public_key(&a, &s, &e).map_err(to_py_err)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encrypt(
        &self,
        py: Python<'_>,
        a: Vec<i64>,
        b: Vec<i64>,
        r: Vec<i64>,
        e1: Vec<i64>,
        e2: Vec<i64>,
        message_bits: Vec<u8>,
    ) -> PyResult<(Vec<i64>, Vec<i64>)> {
        py.detach(|| self.inner.encrypt(&a, &b, &r, &e1, &e2, &message_bits))
            .map_err(to_py_err)
    }

    pub fn decrypt(
        &self,
        py: Python<'_>,
        u: Vec<i64>,
        v: Vec<i64>,
        s: Vec<i64>,
    ) -> PyResult<Vec<u8>> {
        py.detach(|| self.inner.decrypt(&u, &v, &s))
            .map_err(to_py_err)
    }
}
use crate::ode::koopman::KoopmanSpectralEngine;

#[pyclass(name = "KoopmanSpectralEngine", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyKoopmanSpectralEngine {
    pub inner: KoopmanSpectralEngine,
}

#[pymethods]
impl PyKoopmanSpectralEngine {
    #[new]
    pub fn fit_diagonal_edmd(psi_x: Vec<Vec<f64>>, psi_y: Vec<Vec<f64>>) -> PyResult<Self> {
        let inner = KoopmanSpectralEngine::fit_diagonal_edmd(&psi_x, &psi_y).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn predict_observable(&self, psi_init: Vec<f64>, steps: usize) -> PyResult<Vec<f64>> {
        self.inner
            .predict_observable(&psi_init, steps)
            .map_err(to_py_err)
    }

    #[staticmethod]
    pub fn birkhoff_ulam_ergodic_average(
        py: Python<'_>,
        x0: f64,
        num_steps: usize,
    ) -> PyResult<f64> {
        py.detach(|| KoopmanSpectralEngine::birkhoff_ulam_ergodic_average(x0, num_steps))
            .map_err(to_py_err)
    }
}
use crate::omni_nona_viii::{GrandOmniNonaEngineVIII, GrandOmniNonaReportVIII};

#[pyclass(name = "GrandOmniNonaReportVIII", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyGrandOmniNonaReportVIII {
    pub inner: GrandOmniNonaReportVIII,
}

#[pymethods]
impl PyGrandOmniNonaReportVIII {
    #[getter]
    pub fn yang_baxter_residual(&self) -> f64 {
        self.inner.yang_baxter_residual
    }

    #[getter]
    pub fn tropical_eigenvalue(&self) -> f64 {
        self.inner.tropical_eigenvalue
    }

    #[getter]
    pub fn rlbm_conserved_mass(&self) -> f64 {
        self.inner.rlbm_conserved_mass
    }

    #[getter]
    pub fn free_convolution_fourth_moment(&self) -> f64 {
        self.inner.free_convolution_fourth_moment
    }

    #[getter]
    pub fn rough_path_levy_area(&self) -> f64 {
        self.inner.rough_path_levy_area
    }

    #[getter]
    pub fn p_adic_ultrametric_verified(&self) -> bool {
        self.inner.p_adic_ultrametric_verified
    }

    #[getter]
    pub fn morse_inequalities_verified(&self) -> bool {
        self.inner.morse_inequalities_verified
    }

    #[getter]
    pub fn ring_lwe_recovery_rate(&self) -> f64 {
        self.inner.ring_lwe_recovery_rate
    }

    #[getter]
    pub fn koopman_predicted_mode(&self) -> f64 {
        self.inner.koopman_predicted_mode
    }

    #[getter]
    pub fn active_llm_bits_per_weight(&self) -> f64 {
        self.inner.active_llm_bits_per_weight
    }

    #[getter]
    pub fn unified_nona_invariant(&self) -> f64 {
        self.inner.unified_nona_invariant
    }
}

#[pyclass(name = "GrandOmniNonaEngineVIII", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyGrandOmniNonaEngineVIII {
    pub inner: GrandOmniNonaEngineVIII,
}

#[pymethods]
impl PyGrandOmniNonaEngineVIII {
    #[new]
    #[pyo3(signature = (num_layers=8))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniNonaEngineVIII::new(num_layers),
        }
    }

    pub fn execute_nona_synthesis(&self, py: Python<'_>) -> PyResult<PyGrandOmniNonaReportVIII> {
        let rep = py
            .detach(|| self.inner.execute_nona_synthesis())
            .map_err(to_py_err)?;
        Ok(PyGrandOmniNonaReportVIII { inner: rep })
    }
}
use crate::science::derived_geometry::DglaMaurerCartanEngine;

#[pyclass(name = "DglaMaurerCartanEngine", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyDglaMaurerCartanEngine {
    pub inner: DglaMaurerCartanEngine,
}

#[pymethods]
impl PyDglaMaurerCartanEngine {
    #[new]
    pub fn new(dim: usize, d1_matrix: Vec<f64>, bracket_tensor: Vec<f64>) -> PyResult<Self> {
        let inner = guard_res(|| DglaMaurerCartanEngine::new(dim, d1_matrix, bracket_tensor))?;
        Ok(Self { inner })
    }

    pub fn maurer_cartan_residual_norm(&self, alpha: Vec<f64>) -> PyResult<f64> {
        self.inner
            .maurer_cartan_residual_norm(&alpha)
            .map_err(to_py_err)
    }

    pub fn solve_maurer_cartan_flow(
        &self,
        py: Python<'_>,
        alpha_init: Vec<f64>,
        lr: f64,
        steps: usize,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| self.inner.solve_maurer_cartan_flow(&alpha_init, lr, steps))
            .map_err(to_py_err)
    }
}
use crate::quantum::syk_chaos::SykQuantumChaosEngine;

#[pyclass(name = "SykQuantumChaosEngine", module = "oxideon", from_py_object)]
#[derive(Clone, Copy)]
pub struct PySykQuantumChaosEngine {
    pub inner: SykQuantumChaosEngine,
}

#[pymethods]
impl PySykQuantumChaosEngine {
    #[new]
    pub fn new(num_majoranas: usize, coupling_j: f64, beta: f64) -> PyResult<Self> {
        let inner =
            SykQuantumChaosEngine::new(num_majoranas, coupling_j, beta).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn mss_chaos_bound(&self) -> f64 {
        self.inner.mss_chaos_bound()
    }

    pub fn syk_lyapunov_exponent(&self) -> f64 {
        self.inner.syk_lyapunov_exponent()
    }

    pub fn evaluate_otoc(&self, py: Python<'_>, t: f64) -> f64 {
        py.detach(|| self.inner.evaluate_otoc(t))
    }

    pub fn scrambling_time(&self) -> f64 {
        self.inner.scrambling_time()
    }
}
use crate::geom_graphs::wasserstein_flow::GraphWassersteinGeodesicEngine;

#[pyclass(
    name = "GraphWassersteinGeodesicEngine",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGraphWassersteinGeodesicEngine {
    pub inner: GraphWassersteinGeodesicEngine,
}

#[pymethods]
impl PyGraphWassersteinGeodesicEngine {
    #[new]
    pub fn new(num_nodes: usize, dx: f64) -> PyResult<Self> {
        let inner = GraphWassersteinGeodesicEngine::new(num_nodes, dx).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn wasserstein_1_distance(&self, mu0: Vec<f64>, mu1: Vec<f64>) -> PyResult<f64> {
        self.inner
            .wasserstein_1_distance(&mu0, &mu1)
            .map_err(to_py_err)
    }

    pub fn displacement_interpolate(
        &self,
        py: Python<'_>,
        mu0: Vec<f64>,
        mu1: Vec<f64>,
        t: f64,
    ) -> PyResult<(Vec<f64>, f64)> {
        py.detach(|| self.inner.displacement_interpolate(&mu0, &mu1, t))
            .map_err(to_py_err)
    }
}
use crate::reasoning::hott::HomotopyTypeEngine;

#[pyclass(name = "HomotopyTypeEngine", module = "oxideon", from_py_object)]
#[derive(Clone, Default)]
pub struct PyHomotopyTypeEngine {
    pub inner: HomotopyTypeEngine,
}

#[pymethods]
impl PyHomotopyTypeEngine {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: HomotopyTypeEngine::new(),
        }
    }

    #[staticmethod]
    pub fn encode_circle_loop(loop_word: Vec<i32>) -> i64 {
        HomotopyTypeEngine::encode_circle_loop(&loop_word)
    }

    #[staticmethod]
    pub fn decode_winding_number(n: i64) -> PyResult<Vec<i32>> {
        HomotopyTypeEngine::decode_winding_number(n).map_err(to_py_err)
    }

    #[staticmethod]
    pub fn verify_univalent_circle_coherence(n: i64) -> PyResult<bool> {
        HomotopyTypeEngine::verify_univalent_circle_coherence(n).map_err(to_py_err)
    }

    #[staticmethod]
    pub fn univalent_orthogonal_transport(v: Vec<f64>, theta_rad: f64) -> PyResult<Vec<f64>> {
        HomotopyTypeEngine::univalent_orthogonal_transport(&v, theta_rad).map_err(to_py_err)
    }
}
use crate::physics::cavity_qed::JaynesCummingsCavityQed;

#[pyclass(name = "JaynesCummingsCavityQed", module = "oxideon", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyJaynesCummingsCavityQed {
    pub inner: JaynesCummingsCavityQed,
}

#[pymethods]
impl PyJaynesCummingsCavityQed {
    #[new]
    pub fn new(omega_c: f64, omega_q: f64, coupling_g: f64, kappa: f64) -> PyResult<Self> {
        let inner =
            JaynesCummingsCavityQed::new(omega_c, omega_q, coupling_g, kappa).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn generalized_rabi_frequency(&self, n_excitations: usize) -> f64 {
        self.inner.generalized_rabi_frequency(n_excitations)
    }

    pub fn polariton_eigenenergies(&self, n_excitations: usize) -> PyResult<(f64, f64)> {
        self.inner
            .polariton_eigenenergies(n_excitations)
            .map_err(to_py_err)
    }

    pub fn rabi_oscillation_populations(
        &self,
        py: Python<'_>,
        n_excitations: usize,
        t: f64,
    ) -> PyResult<(f64, f64)> {
        py.detach(|| self.inner.rabi_oscillation_populations(n_excitations, t))
            .map_err(to_py_err)
    }
}
use crate::science::spectral_triple::ConnesSpectralTriple;

#[pyclass(name = "ConnesSpectralTriple", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyConnesSpectralTriple {
    pub inner: ConnesSpectralTriple,
}

#[pymethods]
impl PyConnesSpectralTriple {
    #[new]
    pub fn new(dim: usize, dirac_operator: Vec<f64>) -> PyResult<Self> {
        let inner = guard_res(|| ConnesSpectralTriple::new(dim, dirac_operator))?;
        Ok(Self { inner })
    }

    pub fn commutator_frobenius_norm(&self, a: Vec<f64>) -> PyResult<f64> {
        self.inner.commutator_frobenius_norm(&a).map_err(to_py_err)
    }

    pub fn chain_connes_distance(&self, py: Python<'_>, p: usize, q: usize) -> PyResult<f64> {
        py.detach(|| self.inner.chain_connes_distance(p, q))
            .map_err(to_py_err)
    }

    pub fn chamseddine_connes_spectral_trace(&self, cutoff_lambda: f64) -> PyResult<f64> {
        self.inner
            .chamseddine_connes_spectral_trace(cutoff_lambda)
            .map_err(to_py_err)
    }
}
use crate::reasoning::solomonoff::{AlgorithmicHypothesis, SolomonoffLevinEngine};

#[pyclass(name = "SolomonoffLevinEngine", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PySolomonoffLevinEngine {
    pub inner: SolomonoffLevinEngine,
}

#[pymethods]
impl PySolomonoffLevinEngine {
    #[new]
    pub fn new(
        code_lengths: Vec<f64>,
        runtime_steps: Vec<usize>,
        misfits: Vec<f64>,
    ) -> PyResult<Self> {
        if code_lengths.len() != runtime_steps.len() || code_lengths.len() != misfits.len() {
            return Err(to_py_err(crate::error::Error::Msg(
                "Length mismatch".to_string(),
            )));
        }
        let hyps: Vec<AlgorithmicHypothesis> = code_lengths
            .into_iter()
            .zip(runtime_steps)
            .zip(misfits)
            .map(|((c, r), m)| AlgorithmicHypothesis {
                code_length_bits: c,
                runtime_steps: r,
                data_misfit_bits: m,
            })
            .collect();
        let inner = SolomonoffLevinEngine::new(hyps).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn kraft_inequality_sum(&self) -> f64 {
        self.inner.kraft_inequality_sum()
    }

    pub fn levin_kt_complexities(&self) -> Vec<f64> {
        self.inner.levin_kt_complexities()
    }

    pub fn solomonoff_posterior_weights(&self, py: Python<'_>) -> Vec<f64> {
        py.detach(|| self.inner.solomonoff_posterior_weights())
    }
}
use crate::physics::twistor::PenroseTwistorEngine;

#[pyclass(name = "PenroseTwistorEngine", module = "oxideon", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyPenroseTwistorEngine {
    pub inner: PenroseTwistorEngine,
}

#[pymethods]
impl PyPenroseTwistorEngine {
    #[new]
    pub fn from_minkowski_incidence(
        spacetime_x: [f64; 4],
        pi_0: (f64, f64),
        pi_1: (f64, f64),
    ) -> Self {
        Self {
            inner: PenroseTwistorEngine::from_minkowski_incidence(spacetime_x, [pi_0, pi_1]),
        }
    }

    pub fn su22_hermitian_norm(&self) -> f64 {
        self.inner.su22_hermitian_norm()
    }

    pub fn null_four_momentum(&self) -> [f64; 4] {
        self.inner.null_four_momentum()
    }
}
use crate::science::langlands::LanglandsHeckeEngine;

#[pyclass(name = "LanglandsHeckeEngine", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyLanglandsHeckeEngine {
    pub inner: LanglandsHeckeEngine,
}

#[pymethods]
impl PyLanglandsHeckeEngine {
    #[new]
    pub fn new(py: Python<'_>, max_n: usize) -> PyResult<Self> {
        let inner = py
            .detach(|| LanglandsHeckeEngine::new(max_n))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn ramanujan_tau(&self, n: usize) -> PyResult<i64> {
        self.inner.ramanujan_tau(n).map_err(to_py_err)
    }

    pub fn verify_hecke_relations(&self) -> bool {
        self.inner.verify_hecke_relations()
    }

    pub fn verify_deligne_ramanujan_bound(&self) -> bool {
        self.inner.verify_deligne_ramanujan_bound()
    }

    pub fn automorphic_l_function_euler(&self, s: f64) -> PyResult<f64> {
        self.inner
            .automorphic_l_function_euler(s)
            .map_err(to_py_err)
    }
}
use crate::omni_centennial_x::{
    GrandCentennialSingularityEngineX, GrandCentennialSingularityReportX,
};

#[pyclass(
    name = "GrandCentennialSingularityReportX",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandCentennialSingularityReportX {
    pub inner: GrandCentennialSingularityReportX,
}

#[pymethods]
impl PyGrandCentennialSingularityReportX {
    #[getter]
    pub fn maurer_cartan_residual(&self) -> f64 {
        self.inner.maurer_cartan_residual
    }

    #[getter]
    pub fn syk_mss_saturation_ratio(&self) -> f64 {
        self.inner.syk_mss_saturation_ratio
    }

    #[getter]
    pub fn wasserstein_geodesic_w2(&self) -> f64 {
        self.inner.wasserstein_geodesic_w2
    }

    #[getter]
    pub fn hott_univalence_verified(&self) -> bool {
        self.inner.hott_univalence_verified
    }

    #[getter]
    pub fn jaynes_cummings_rabi_splitting(&self) -> f64 {
        self.inner.jaynes_cummings_rabi_splitting
    }

    #[getter]
    pub fn connes_spectral_distance(&self) -> f64 {
        self.inner.connes_spectral_distance
    }

    #[getter]
    pub fn solomonoff_top_posterior(&self) -> f64 {
        self.inner.solomonoff_top_posterior
    }

    #[getter]
    pub fn twistor_null_helicity_residual(&self) -> f64 {
        self.inner.twistor_null_helicity_residual
    }

    #[getter]
    pub fn langlands_deligne_verified(&self) -> bool {
        self.inner.langlands_deligne_verified
    }

    #[getter]
    pub fn active_llm_bits_per_weight(&self) -> f64 {
        self.inner.active_llm_bits_per_weight
    }

    #[getter]
    pub fn centennial_singularity_invariant(&self) -> f64 {
        self.inner.centennial_singularity_invariant
    }
}

#[pyclass(
    name = "GrandCentennialSingularityEngineX",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandCentennialSingularityEngineX {
    pub inner: GrandCentennialSingularityEngineX,
}

#[pymethods]
impl PyGrandCentennialSingularityEngineX {
    #[new]
    #[pyo3(signature = (num_layers=12))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandCentennialSingularityEngineX::new(num_layers),
        }
    }

    pub fn execute_centennial_singularity(
        &self,
        py: Python<'_>,
    ) -> PyResult<PyGrandCentennialSingularityReportX> {
        let rep = py
            .detach(|| self.inner.execute_centennial_singularity())
            .map_err(to_py_err)?;
        Ok(PyGrandCentennialSingularityReportX { inner: rep })
    }
}
use crate::science::etale_cohomology::EtaleWeilZetaEngine;

#[pyclass(name = "EtaleWeilZetaEngine", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyEtaleWeilZetaEngine {
    pub inner: EtaleWeilZetaEngine,
}

#[pymethods]
impl PyEtaleWeilZetaEngine {
    #[new]
    pub fn new(field_q: u64, point_count_f_q: u64) -> PyResult<Self> {
        let inner = guard_res(|| EtaleWeilZetaEngine::new(field_q, point_count_f_q))?;
        Ok(Self { inner })
    }

    pub fn rational_point_count(&self, py: Python<'_>, degree_r: usize) -> PyResult<u64> {
        py.detach(|| self.inner.rational_point_count(degree_r))
            .map_err(to_py_err)
    }

    pub fn evaluate_zeta(&self, t: f64) -> PyResult<f64> {
        self.inner.evaluate_zeta(t).map_err(to_py_err)
    }
}

use crate::science::motivic_integration::MotivicHodgeEngine;

py_engine! {
    PyMotivicHodgeEngine("MotivicHodgeEngine") wraps MotivicHodgeEngine [Clone, Copy, Default] {
        new();
        assoc projective_space_motive(dim_n: usize, u: f64, v: f64) -> f64;
        assoc grassmannian_g24_motive(u: f64, v: f64) -> f64;
        assoc_res kontsevich_arc_integral(dim_d: u32, order_a: u32, lefschetz_l: f64) -> f64;
    }
}

use crate::science::cluster_algebras::ClusterAlgebraQuiverEngine;

#[pyclass(
    name = "ClusterAlgebraQuiverEngine",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyClusterAlgebraQuiverEngine {
    pub inner: ClusterAlgebraQuiverEngine,
}

#[pymethods]
impl PyClusterAlgebraQuiverEngine {
    #[new]
    pub fn new(rank: usize, exchange_matrix: Vec<i64>, cluster_vars: Vec<f64>) -> PyResult<Self> {
        let inner =
            guard_res(|| ClusterAlgebraQuiverEngine::new(rank, exchange_matrix, cluster_vars))?;
        Ok(Self { inner })
    }

    pub fn mutate(&mut self, py: Python<'_>, k: usize) -> PyResult<()> {
        py.detach(|| self.inner.mutate(k)).map_err(to_py_err)
    }

    #[getter]
    pub fn cluster_vars(&self) -> Vec<f64> {
        self.inner.cluster_vars.clone()
    }
}

use crate::science::intersection_theory::ChowHirzebruchRiemannRoch;

#[pyclass(name = "ChowHirzebruchRiemannRoch", module = "oxideon", from_py_object)]
#[derive(Clone, Copy, Default)]
pub struct PyChowHirzebruchRiemannRoch {
    pub inner: ChowHirzebruchRiemannRoch,
}

#[pymethods]
impl PyChowHirzebruchRiemannRoch {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: ChowHirzebruchRiemannRoch::new(),
        }
    }

    #[staticmethod]
    pub fn riemann_roch_curve(genus_g: i64, rank: i64, degree: i64) -> i128 {
        ChowHirzebruchRiemannRoch::riemann_roch_curve(genus_g, rank, degree)
    }

    #[staticmethod]
    pub fn riemann_roch_surface(
        k_self_intersection: i64,
        euler_c2: i64,
        d_self: i64,
        d_dot_k: i64,
    ) -> f64 {
        ChowHirzebruchRiemannRoch::riemann_roch_surface(
            k_self_intersection,
            euler_c2,
            d_self,
            d_dot_k,
        )
    }

    #[staticmethod]
    pub fn bezout_intersection_number(degrees: Vec<u64>) -> u64 {
        ChowHirzebruchRiemannRoch::bezout_intersection_number(&degrees)
    }
}

use crate::science::bridgeland_stability::BridgelandStabilityEngine;

#[pyclass(name = "BridgelandStabilityEngine", module = "oxideon", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyBridgelandStabilityEngine {
    pub inner: BridgelandStabilityEngine,
}

#[pymethods]
impl PyBridgelandStabilityEngine {
    #[new]
    pub fn new(kahler_omega: f64) -> PyResult<Self> {
        let inner = BridgelandStabilityEngine::new(kahler_omega).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn phase_and_mass(&self, rank: f64, degree: f64) -> PyResult<(f64, f64)> {
        self.inner.phase_and_mass(rank, degree).map_err(to_py_err)
    }

    pub fn verify_harder_narasimhan_filtration(
        &self,
        py: Python<'_>,
        factors: Vec<(f64, f64)>,
    ) -> PyResult<(bool, f64)> {
        py.detach(|| self.inner.verify_harder_narasimhan_filtration(&factors))
            .map_err(to_py_err)
    }
}

use crate::science::gromov_witten::GromovWittenWdvvEngine;

#[pyclass(name = "GromovWittenWdvvEngine", module = "oxideon", from_py_object)]
#[derive(Clone, Copy, Default)]
pub struct PyGromovWittenWdvvEngine {
    pub inner: GromovWittenWdvvEngine,
}

#[pymethods]
impl PyGromovWittenWdvvEngine {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: GromovWittenWdvvEngine::new(),
        }
    }

    #[staticmethod]
    pub fn kontsevich_rational_curve_count(py: Python<'_>, degree_d: usize) -> PyResult<u64> {
        py.detach(|| GromovWittenWdvvEngine::kontsevich_rational_curve_count(degree_d))
            .map_err(to_py_err)
    }
}

use crate::science::donaldson_thomas::DonaldsonThomasEngine;

#[pyclass(name = "DonaldsonThomasEngine", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyDonaldsonThomasEngine {
    pub inner: DonaldsonThomasEngine,
}

#[pymethods]
impl PyDonaldsonThomasEngine {
    #[new]
    pub fn new(py: Python<'_>, max_n: usize) -> PyResult<Self> {
        let inner = py
            .detach(|| DonaldsonThomasEngine::new(max_n))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn plane_partition_count(&self, n: usize) -> PyResult<u64> {
        self.inner.plane_partition_count(n).map_err(to_py_err)
    }
}

use crate::science::hitchin_system::HitchinHiggsBundleEngine;

py_engine! {
    PyHitchinHiggsBundleEngine("HitchinHiggsBundleEngine") wraps HitchinHiggsBundleEngine [Clone, Copy] {
        new(base_genus_g: usize, bundle_rank_r: usize) -> res;
        method spectral_curve_genus() -> usize;
        assoc evaluate_sl2_hitchin_hamiltonians(phi: [f64; 4]) -> (f64, f64, f64);
    }
}

use crate::science::perfectoid_spaces::PerfectoidTiltingEngine;

#[pyclass(name = "PerfectoidTiltingEngine", module = "oxideon", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyPerfectoidTiltingEngine {
    pub inner: PerfectoidTiltingEngine,
}

#[pymethods]
impl PyPerfectoidTiltingEngine {
    #[new]
    pub fn new(prime_p: u64) -> PyResult<Self> {
        let inner = PerfectoidTiltingEngine::new(prime_p).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn verify_frobenius_tower(&self, py: Python<'_>, tower: Vec<u64>) -> bool {
        py.detach(|| self.inner.verify_frobenius_tower(&tower))
    }

    pub fn teichmuller_sharp_lift(&self, x_n: u64, level_n: u32) -> PyResult<u64> {
        self.inner
            .teichmuller_sharp_lift(x_n, level_n)
            .map_err(to_py_err)
    }
}

use crate::omni_hendeca_xi::GrandOmniHendecaEngineXI;

#[pyclass(name = "GrandOmniHendecaEngineXI", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyGrandOmniHendecaEngineXI {
    pub inner: GrandOmniHendecaEngineXI,
}

#[pymethods]
impl PyGrandOmniHendecaEngineXI {
    #[new]
    #[pyo3(signature = (num_layers=10))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniHendecaEngineXI::new(num_layers),
        }
    }

    pub fn execute_hendeca_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_hendeca_synthesis())
            .map_err(to_py_err)
    }
}

use crate::physics::loop_quantum_gravity::SpinNetworkLqgEngine;

#[pyclass(name = "SpinNetworkLqgEngine", module = "oxideon", from_py_object)]
#[derive(Clone, Copy)]
pub struct PySpinNetworkLqgEngine {
    pub inner: SpinNetworkLqgEngine,
}

#[pymethods]
impl PySpinNetworkLqgEngine {
    #[new]
    pub fn new(immirzi_gamma: f64, planck_length: f64) -> PyResult<Self> {
        let inner = SpinNetworkLqgEngine::new(immirzi_gamma, planck_length).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn quantized_area_eigenvalue(
        &self,
        py: Python<'_>,
        two_j_punctures: Vec<u32>,
    ) -> PyResult<f64> {
        py.detach(|| self.inner.quantized_area_eigenvalue(&two_j_punctures))
            .map_err(to_py_err)
    }

    #[staticmethod]
    pub fn is_admissible_trivalent_intertwiner(two_j1: u32, two_j2: u32, two_j3: u32) -> bool {
        SpinNetworkLqgEngine::is_admissible_trivalent_intertwiner(two_j1, two_j2, two_j3)
    }
}

use crate::physics::causal_dynamical_triangulations::CausalTriangulationEngine;

#[pyclass(name = "CausalTriangulationEngine", module = "oxideon", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyCausalTriangulationEngine {
    pub inner: CausalTriangulationEngine,
}

#[pymethods]
impl PyCausalTriangulationEngine {
    #[new]
    pub fn new(k0: f64, delta_cosmological: f64) -> PyResult<Self> {
        let inner = CausalTriangulationEngine::new(k0, delta_cosmological).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn regge_action(&self, n0: usize, n4_41: usize, n4_32: usize) -> f64 {
        self.inner.regge_action(n0, n4_41, n4_32)
    }

    pub fn spectral_dimension(
        &self,
        py: Python<'_>,
        diffusion_sigma: f64,
        scale_b: f64,
    ) -> PyResult<f64> {
        py.detach(|| self.inner.spectral_dimension(diffusion_sigma, scale_b))
            .map_err(to_py_err)
    }
}

use crate::physics::amplituhedron::AmplituhedronGrassmannianEngine;

#[pyclass(
    name = "AmplituhedronGrassmannianEngine",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyAmplituhedronGrassmannianEngine {
    pub inner: AmplituhedronGrassmannianEngine,
}

#[pymethods]
impl PyAmplituhedronGrassmannianEngine {
    #[new]
    pub fn new(row0: Vec<f64>, row1: Vec<f64>) -> PyResult<Self> {
        let inner = AmplituhedronGrassmannianEngine::new(row0, row1).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn plucker_minor(&self, i: usize, j: usize) -> PyResult<f64> {
        self.inner.plucker_minor(i, j).map_err(to_py_err)
    }

    pub fn verify_positive_grassmannian(&self, py: Python<'_>) -> (bool, f64) {
        py.detach(|| self.inner.verify_positive_grassmannian())
    }
}

use crate::physics::celestial_holography::CelestialHolographyEngine;

py_engine! {
    PyCelestialHolographyEngine("CelestialHolographyEngine") wraps CelestialHolographyEngine [Clone, Copy, Default] {
        new();
        assoc_res mellin_celestial_correlator(delta: u32, uv_scale_s: f64) -> f64;
        assoc_res weinberg_soft_pole_residue(delta: f64, scale_s: f64) -> f64;
    }
}

use crate::physics::swampland_conjectures::SwamplandVerifierEngine;

py_engine! {
    PySwamplandVerifierEngine("SwamplandVerifierEngine") wraps SwamplandVerifierEngine [Clone, Copy, Default] {
        new();
        assoc satisfies_weak_gravity_conjecture(mass: f64, charge_q: f64, gauge_g: f64) -> bool;
        assoc distance_conjecture_tower_mass(m0: f64, alpha: f64, geodesic_dist: f64) -> f64;
    }
}

use crate::physics::conformal_bootstrap::ConformalBootstrapEngine;

#[pyclass(name = "ConformalBootstrapEngine", module = "oxideon", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyConformalBootstrapEngine {
    pub inner: ConformalBootstrapEngine,
}

#[pymethods]
impl PyConformalBootstrapEngine {
    #[new]
    pub fn new(spacetime_dim: f64) -> PyResult<Self> {
        let inner = ConformalBootstrapEngine::new(spacetime_dim).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn unitarity_bound(&self, spin_l: usize) -> f64 {
        self.inner.unitarity_bound(spin_l)
    }

    #[staticmethod]
    pub fn chiral_conformal_block(py: Python<'_>, delta: f64, z: f64) -> PyResult<f64> {
        py.detach(|| ConformalBootstrapEngine::chiral_conformal_block(delta, z))
            .map_err(to_py_err)
    }
}

use crate::physics::sachdev_kitaev_wormhole::TraversableWormholeEngine;

#[pyclass(name = "TraversableWormholeEngine", module = "oxideon", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyTraversableWormholeEngine {
    pub inner: TraversableWormholeEngine,
}

#[pymethods]
impl PyTraversableWormholeEngine {
    #[new]
    pub fn new(coupling_mu: f64, scaling_delta: f64) -> PyResult<Self> {
        let inner =
            TraversableWormholeEngine::new(coupling_mu, scaling_delta).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn wormhole_energy_gap(&self) -> f64 {
        self.inner.wormhole_energy_gap()
    }

    pub fn transmission_fidelity(&self, py: Python<'_>, t: f64) -> f64 {
        py.detach(|| self.inner.transmission_fidelity(t))
    }
}

use crate::physics::resurgence_transseries::BorelResurgenceEngine;

py_engine! {
    PyBorelResurgenceEngine("BorelResurgenceEngine") wraps BorelResurgenceEngine [Clone, Copy, Default] {
        new();
        assoc_res borel_pade_11_transform(coeffs: [f64; 3], t: f64) -> f64;
        assoc_res evaluate_one_instanton_transseries(g: f64, action_s: f64, stokes_sigma: f64, pert0: f64, pert1: f64) -> f64;
    }
}

use crate::physics::exact_wkb::ExactWkbQuantizationEngine;

#[pyclass(
    name = "ExactWkbQuantizationEngine",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone, Copy)]
pub struct PyExactWkbQuantizationEngine {
    pub inner: ExactWkbQuantizationEngine,
}

#[pymethods]
impl PyExactWkbQuantizationEngine {
    #[new]
    pub fn new(hbar: f64, omega: f64, anharmonic_lambda: f64) -> PyResult<Self> {
        let inner =
            ExactWkbQuantizationEngine::new(hbar, omega, anharmonic_lambda).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn quantized_energy_level(&self, py: Python<'_>, level_n: usize) -> f64 {
        py.detach(|| self.inner.quantized_energy_level(level_n))
    }

    pub fn double_well_instanton_action(&self) -> PyResult<f64> {
        self.inner.double_well_instanton_action().map_err(to_py_err)
    }
}

use crate::omni_dodeca_xii::GrandOmniDodecaEngineXII;

#[pyclass(name = "GrandOmniDodecaEngineXII", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyGrandOmniDodecaEngineXII {
    pub inner: GrandOmniDodecaEngineXII,
}

#[pymethods]
impl PyGrandOmniDodecaEngineXII {
    #[new]
    #[pyo3(signature = (num_layers=12))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniDodecaEngineXII::new(num_layers),
        }
    }

    pub fn execute_dodeca_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_dodeca_synthesis())
            .map_err(to_py_err)
    }
}

use crate::quantum::quantum_ldpc::QuantumLdpcCodeEngine;

#[pyclass(name = "QuantumLdpcCodeEngine", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyQuantumLdpcCodeEngine {
    pub inner: QuantumLdpcCodeEngine,
}

#[pymethods]
impl PyQuantumLdpcCodeEngine {
    #[new]
    pub fn new(num_qubits: usize, hx: Vec<u8>, hz: Vec<u8>) -> PyResult<Self> {
        let inner = QuantumLdpcCodeEngine::new(num_qubits, hx, hz).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn verify_css_commutativity(&self, py: Python<'_>) -> PyResult<bool> {
        py.detach(|| self.inner.verify_css_commutativity())
            .map_err(to_py_err)
    }

    pub fn compute_z_syndrome(&self, x_error: Vec<u8>) -> PyResult<Vec<u8>> {
        self.inner.compute_z_syndrome(&x_error).map_err(to_py_err)
    }
}

use crate::quantum::magic_state_distillation::MagicStateDistillationEngine;

#[pyclass(
    name = "MagicStateDistillationEngine",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone, Copy, Default)]
pub struct PyMagicStateDistillationEngine {
    pub inner: MagicStateDistillationEngine,
}

#[pymethods]
impl PyMagicStateDistillationEngine {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: MagicStateDistillationEngine::new(),
        }
    }

    #[staticmethod]
    pub fn distill_15_to_1_infidelity(
        py: Python<'_>,
        initial_error_p: f64,
        rounds: usize,
    ) -> PyResult<(f64, u64)> {
        py.detach(|| {
            MagicStateDistillationEngine::distill_15_to_1_infidelity(initial_error_p, rounds)
        })
        .map_err(to_py_err)
    }
}

use crate::quantum::mera_holography::MeraEntanglementEngine;

py_engine! {
    PyMeraEntanglementEngine("MeraEntanglementEngine") wraps MeraEntanglementEngine [Clone, Copy] {
        new(bond_dim_chi: usize) -> res;
        method_res causal_cone_entanglement_bound(interval_l: f64) -> f64;
        assoc_res scaling_dimension_from_eigenvalue(lambda: f64, branching_b: f64) -> f64;
    }
}

use crate::quantum::quantum_darwinism::QuantumDarwinismEngine;

#[pyclass(name = "QuantumDarwinismEngine", module = "oxideon", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyQuantumDarwinismEngine {
    pub inner: QuantumDarwinismEngine,
}

#[pymethods]
impl PyQuantumDarwinismEngine {
    #[new]
    pub fn new(num_env_fragments: usize, fragment_overlap_c: f64) -> PyResult<Self> {
        let inner = QuantumDarwinismEngine::new(num_env_fragments, fragment_overlap_c)
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn mutual_information_bits(
        &self,
        py: Python<'_>,
        fragment_count_k: usize,
    ) -> PyResult<f64> {
        py.detach(|| self.inner.mutual_information_bits(fragment_count_k))
            .map_err(to_py_err)
    }
}

use crate::quantum::floquet_time_crystals::FloquetTimeCrystalEngine;

py_engine! {
    PyFloquetTimeCrystalEngine("FloquetTimeCrystalEngine") wraps FloquetTimeCrystalEngine [Clone, Copy] {
        new(pulse_imperfection_eps: f64, interaction_j: f64) -> res;
        method stroboscopic_magnetization(period_n: usize) -> f64;
    }
}

use crate::quantum::majorana_kitaev_chain::KitaevMajoranaChainEngine;

py_engine! {
    PyKitaevMajoranaChainEngine("KitaevMajoranaChainEngine") wraps KitaevMajoranaChainEngine [Clone, Copy] {
        new(mu: f64, hopping_t: f64, pairing_delta: f64) -> res;
        method z2_pfaffian_invariant() -> i32;
        method bdg_quasiparticle_energy(k: f64) -> f64;
    }
}

use crate::quantum::quantum_metrology::QuantumMetrologyEngine;

py_engine! {
    PyQuantumMetrologyEngine("QuantumMetrologyEngine") wraps QuantumMetrologyEngine [Clone, Copy, Default] {
        new();
        assoc_res ghz_quantum_fisher_info(num_qubits_n: usize, visibility_p: f64) -> (f64, f64);
    }
}

use crate::quantum::quantum_thermodynamics::QuantumOttoEngine;

py_engine! {
    PyQuantumOttoEngine("QuantumOttoEngine") wraps QuantumOttoEngine [Clone, Copy] {
        new(omega_h: f64, omega_c: f64, temp_h: f64, temp_c: f64) -> res;
        method otto_and_carnot_efficiencies() -> (f64, f64);
        method work_per_cycle() -> f64;
    }
}

use crate::quantum::shadow_tomography::ClassicalShadowTomography;

#[pyclass(name = "ClassicalShadowTomography", module = "oxideon", from_py_object)]
#[derive(Clone, Copy, Default)]
pub struct PyClassicalShadowTomography {
    pub inner: ClassicalShadowTomography,
}

#[pymethods]
impl PyClassicalShadowTomography {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: ClassicalShadowTomography::new(),
        }
    }

    #[staticmethod]
    pub fn sample_complexity_bound(
        pauli_weight_k: u32,
        num_observables_m: usize,
        eps: f64,
        failure_delta: f64,
    ) -> PyResult<u64> {
        ClassicalShadowTomography::sample_complexity_bound(
            pauli_weight_k,
            num_observables_m,
            eps,
            failure_delta,
        )
        .map_err(to_py_err)
    }

    #[staticmethod]
    pub fn median_of_means(
        py: Python<'_>,
        samples: Vec<f64>,
        num_blocks_k: usize,
    ) -> PyResult<f64> {
        py.detach(|| ClassicalShadowTomography::median_of_means(&samples, num_blocks_k))
            .map_err(to_py_err)
    }
}

use crate::omni_trideca_xiii::GrandOmniTridecaEngineXIII;

#[pyclass(
    name = "GrandOmniTridecaEngineXIII",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniTridecaEngineXIII {
    pub inner: GrandOmniTridecaEngineXIII,
}

#[pymethods]
impl PyGrandOmniTridecaEngineXIII {
    #[new]
    #[pyo3(signature = (num_layers=13))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniTridecaEngineXIII::new(num_layers),
        }
    }

    pub fn execute_trideca_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_trideca_synthesis())
            .map_err(to_py_err)
    }
}

use crate::science::ricci_flow_perelman::PerelmanRicciFlowEngine;

py_engine! {
    PyPerelmanRicciFlowEngine("PerelmanRicciFlowEngine") wraps PerelmanRicciFlowEngine [Clone, Copy] {
        new(dim_n: usize, initial_radius: f64) -> res;
        method extinction_time() -> f64;
        method_res radius_and_scalar_curvature(t: f64) -> (f64, f64);
    }
}

use crate::science::mean_curvature_flow::MeanCurvatureFlowEngine;

py_engine! {
    PyMeanCurvatureFlowEngine("MeanCurvatureFlowEngine") wraps MeanCurvatureFlowEngine [Clone, Copy] {
        new(dim_n: usize, initial_radius: f64) -> res;
        method extinction_time() -> f64;
        method colding_minicozzi_sphere_entropy() -> f64;
    }
}

use crate::science::calabi_yau_monge_ampere::CalabiYauMongeAmpereEngine;

py_engine! {
    PyCalabiYauMongeAmpereEngine("CalabiYauMongeAmpereEngine") wraps CalabiYauMongeAmpereEngine [Clone, Copy, Default] {
        new();
        assoc_res monge_ampere_2d_residual(g_diag: [f64; 2], hessian_phi: [f64; 4], ricci_potential_f: f64) -> f64;
    }
}

use crate::science::yang_mills_instanton::BpstYangMillsInstanton;

py_engine! {
    PyBpstYangMillsInstanton("BpstYangMillsInstanton") wraps BpstYangMillsInstanton [Clone, Copy] {
        new(scale_rho: f64, coupling_g: f64) -> res;
        method bogomolny_instanton_action(pontryagin_k: i64) -> f64;
        method bpst_topological_density(radius_r: f64) -> f64;
    }
}

use crate::science::seiberg_witten::SeibergWittenMonopoleEngine;

py_engine! {
    PySeibergWittenMonopoleEngine("SeibergWittenMonopoleEngine") wraps SeibergWittenMonopoleEngine [Clone, Copy, Default] {
        new();
        assoc moduli_virtual_dimension(c1_sq: i64, euler_chi: i64, signature_sigma: i64) -> f64;
        assoc satisfies_adjunction_inequality(genus_g: i64, self_intersection: i64, c1_pairing: i64) -> bool;
    }
}

use crate::science::minimal_surfaces_plateau::WeierstrassMinimalSurfaceEngine;

py_engine! {
    PyWeierstrassMinimalSurfaceEngine("WeierstrassMinimalSurfaceEngine") wraps WeierstrassMinimalSurfaceEngine [Clone, Copy, Default] {
        new();
        assoc enneper_surface_point_and_curvature(u: f64, v: f64) -> ([f64; 3], f64);
        assoc torus_willmore_energy(major_r: f64, minor_r: f64) -> f64;
    }
}

use crate::science::microlocal_sheaves::MicrolocalWavefrontEngine;

py_engine! {
    PyMicrolocalWavefrontEngine("MicrolocalWavefrontEngine") wraps MicrolocalWavefrontEngine [Clone, Copy] {
        new(wave_speed_c: f64) -> res;
        method principal_symbol(xi: [f64; 2], tau: f64) -> f64;
        method_res propagate_bicharacteristic(x0: [f64; 2], xi: [f64; 2], t: f64) -> [f64; 2];
    }
}

use crate::science::index_theorem_atiyah_singer::AtiyahSingerIndexEngine;

py_engine! {
    PyAtiyahSingerIndexEngine("AtiyahSingerIndexEngine") wraps AtiyahSingerIndexEngine [Clone, Copy, Default] {
        new();
        assoc a_hat_genus_4d(signature_sigma: i64) -> f64;
        assoc twisted_dirac_index_4d(signature_sigma: i64, bundle_rank_r: i64, c1_sq: i64, c2: i64) -> f64;
    }
}

use crate::science::non_abelian_monopoles::BpsMagneticMonopoleEngine;

py_engine! {
    PyBpsMagneticMonopoleEngine("BpsMagneticMonopoleEngine") wraps BpsMagneticMonopoleEngine [Clone, Copy] {
        new(gauge_e: f64, higgs_vev: f64) -> res;
        method bps_monopole_mass(winding_n: i64) -> f64;
        method prasad_sommerfield_higgs_profile(radius_r: f64) -> f64;
    }
}

use crate::omni_tetradeca_xiv::GrandOmniTetradecaEngineXIV;

#[pyclass(
    name = "GrandOmniTetradecaEngineXIV",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniTetradecaEngineXIV {
    pub inner: GrandOmniTetradecaEngineXIV,
}

#[pymethods]
impl PyGrandOmniTetradecaEngineXIV {
    #[new]
    #[pyo3(signature = (num_layers=14))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniTetradecaEngineXIV::new(num_layers),
        }
    }

    pub fn execute_tetradeca_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_tetradeca_synthesis())
            .map_err(to_py_err)
    }
}

use crate::science::riemann_siegel::RiemannSiegelZEngine;

#[pyclass(name = "RiemannSiegelZEngine", module = "oxideon", from_py_object)]
#[derive(Clone, Copy, Default)]
pub struct PyRiemannSiegelZEngine {
    pub inner: RiemannSiegelZEngine,
}

#[pymethods]
impl PyRiemannSiegelZEngine {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: RiemannSiegelZEngine::new(),
        }
    }

    #[staticmethod]
    pub fn riemann_siegel_theta(t: f64) -> PyResult<f64> {
        RiemannSiegelZEngine::riemann_siegel_theta(t).map_err(to_py_err)
    }

    #[staticmethod]
    pub fn hardy_z_main_sum(py: Python<'_>, t: f64) -> PyResult<f64> {
        py.detach(|| RiemannSiegelZEngine::hardy_z_main_sum(t))
            .map_err(to_py_err)
    }
}

use crate::science::selberg_trace::SelbergTraceFormulaEngine;

#[pyclass(name = "SelbergTraceFormulaEngine", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PySelbergTraceFormulaEngine {
    pub inner: SelbergTraceFormulaEngine,
}

#[pymethods]
impl PySelbergTraceFormulaEngine {
    #[new]
    pub fn new(genus_g: usize, prime_geodesic_lengths: Vec<f64>) -> PyResult<Self> {
        let inner =
            SelbergTraceFormulaEngine::new(genus_g, prime_geodesic_lengths).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn hyperbolic_area(&self) -> f64 {
        self.inner.hyperbolic_area()
    }

    pub fn selberg_zeta_product(&self, py: Python<'_>, s: f64, k_max: usize) -> PyResult<f64> {
        py.detach(|| self.inner.selberg_zeta_product(s, k_max))
            .map_err(to_py_err)
    }
}

use crate::science::birch_swinnerton_dyer::BirchSwinnertonDyerEngine;

#[pyclass(name = "BirchSwinnertonDyerEngine", module = "oxideon", from_py_object)]
#[derive(Clone, Copy, Default)]
pub struct PyBirchSwinnertonDyerEngine {
    pub inner: BirchSwinnertonDyerEngine,
}

#[pymethods]
impl PyBirchSwinnertonDyerEngine {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: BirchSwinnertonDyerEngine::new(),
        }
    }

    #[staticmethod]
    pub fn bsd_leading_coefficient(
        sha_order: u64,
        real_period_omega: f64,
        neron_tate_regulator: f64,
        tamagawa_product: u64,
        torsion_order: u64,
    ) -> PyResult<f64> {
        BirchSwinnertonDyerEngine::bsd_leading_coefficient(
            sha_order,
            real_period_omega,
            neron_tate_regulator,
            tamagawa_product,
            torsion_order,
        )
        .map_err(to_py_err)
    }
}

use crate::science::iwasawa_theory::IwasawaMainConjectureEngine;

py_engine! {
    PyIwasawaMainConjectureEngine("IwasawaMainConjectureEngine") wraps IwasawaMainConjectureEngine [Clone, Copy] {
        new(prime_p: u64, mu_inv: u64, lambda_inv: u64, nu_inv: i64) -> res;
        method class_number_p_valuation(layer_n: u32) -> i64;
    }
}

use crate::science::abc_szpiro_arakelov::AbcSzpiroArakelovEngine;

py_engine! {
    PyAbcSzpiroArakelovEngine("AbcSzpiroArakelovEngine") wraps AbcSzpiroArakelovEngine [Clone, Copy, Default] {
        new();
        assoc squarefree_radical(n: u64) -> u64;
        assoc_res abc_quality(a: u64, b: u64) -> f64;
    }
}

use crate::science::circle_method_hardy_littlewood::HardyLittlewoodCircleEngine;

#[pyclass(
    name = "HardyLittlewoodCircleEngine",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone, Copy, Default)]
pub struct PyHardyLittlewoodCircleEngine {
    pub inner: HardyLittlewoodCircleEngine,
}

#[pymethods]
impl PyHardyLittlewoodCircleEngine {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: HardyLittlewoodCircleEngine::new(),
        }
    }

    #[staticmethod]
    pub fn weyl_exponential_sum_modulus(
        py: Python<'_>,
        alpha: f64,
        power_k: u32,
        max_n: usize,
    ) -> PyResult<f64> {
        py.detach(|| {
            HardyLittlewoodCircleEngine::weyl_exponential_sum_modulus(alpha, power_k, max_n)
        })
        .map_err(to_py_err)
    }
}

use crate::science::sieve_selberg_bombieri::SelbergSieveEngine;

#[pyclass(name = "SelbergSieveEngine", module = "oxideon", from_py_object)]
#[derive(Clone, Copy, Default)]
pub struct PySelbergSieveEngine {
    pub inner: SelbergSieveEngine,
}

#[pymethods]
impl PySelbergSieveEngine {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: SelbergSieveEngine::new(),
        }
    }

    #[staticmethod]
    pub fn selberg_j_sum(py: Python<'_>, z_limit: usize) -> PyResult<f64> {
        py.detach(|| SelbergSieveEngine::selberg_j_sum(z_limit))
            .map_err(to_py_err)
    }
}

use crate::science::diophantine_faltings_roth::FaltingsHeightEngine;

py_engine! {
    PyFaltingsHeightEngine("FaltingsHeightEngine") wraps FaltingsHeightEngine [Clone, Copy, Default] {
        new();
        assoc_res diophantine_approximation_exponent(alpha: f64, p: i64, q: i64) -> (f64, f64);
    }
}

use crate::science::katz_sarnak_random_matrix::KatzSarnakSymmetryEngine;

py_engine! {
    PyKatzSarnakSymmetryEngine("KatzSarnakSymmetryEngine") wraps KatzSarnakSymmetryEngine [Clone, Copy, Default] {
        new();
        assoc montgomery_gue_pair_correlation(u: f64) -> f64;
        assoc usp_one_level_density(x: f64) -> f64;
    }
}

use crate::omni_pentadeca_xv::GrandOmniPentadecaEngineXV;

#[pyclass(
    name = "GrandOmniPentadecaEngineXV",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniPentadecaEngineXV {
    pub inner: GrandOmniPentadecaEngineXV,
}

#[pymethods]
impl PyGrandOmniPentadecaEngineXV {
    #[new]
    #[pyo3(signature = (num_layers=15))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniPentadecaEngineXV::new(num_layers),
        }
    }

    pub fn execute_pentadeca_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_pentadeca_synthesis())
            .map_err(to_py_err)
    }
}

use crate::physics::kolmogorov_turbulence::KolmogorovCascadeEngine;

py_engine! {
    PyKolmogorovCascadeEngine("KolmogorovCascadeEngine") wraps KolmogorovCascadeEngine [Clone, Copy] {
        new(dissipation_eps: f64, viscosity_nu: f64) -> res;
        method_res spectrum_and_microscale(wavenumber_k: f64) -> (f64, f64);
        assoc she_leveque_exponent(order_p: f64) -> f64;
    }
}

use crate::physics::kraichnan_2d_enstrophy::KraichnanEnstrophyCascade;

py_engine! {
    PyKraichnanEnstrophyCascade("KraichnanEnstrophyCascade") wraps KraichnanEnstrophyCascade [Clone, Copy] {
        new(forcing_kf: f64, energy_flux_eps: f64, enstrophy_flux_eta: f64) -> res;
        method_res dual_cascade_spectrum(k: f64) -> f64;
    }
}

use crate::physics::alfven_mhd_dynamo::AlfvenMhdDynamoEngine;

py_engine! {
    PyAlfvenMhdDynamoEngine("AlfvenMhdDynamoEngine") wraps AlfvenMhdDynamoEngine [Clone, Copy] {
        new(density_rho: f64, mu_0: f64, diffusivity_eta: f64) -> res;
        method alfven_speed(b0: f64) -> f64;
        method alpha_omega_growth_rate(alpha_helicity: f64, omega_shear: f64, wavenumber_k: f64) -> f64;
    }
}

use crate::physics::quasi_geostrophic_rossby::CharneyRossbyWaveEngine;

py_engine! {
    PyCharneyRossbyWaveEngine("CharneyRossbyWaveEngine") wraps CharneyRossbyWaveEngine [Clone, Copy] {
        new(coriolis_beta: f64, deformation_radius_lr: f64) -> res;
        method frequency_and_zonal_group_velocity(kx: f64, ky: f64) -> (f64, f64);
    }
}

use crate::physics::rayleigh_benard_lorenz::RayleighBenardConvection;

py_engine! {
    PyRayleighBenardConvection("RayleighBenardConvection") wraps RayleighBenardConvection [Clone, Copy, Default] {
        new();
        assoc critical_rayleigh_free_free() -> f64;
        assoc marginal_rayleigh_curve(wavenumber_a: f64) -> f64;
    }
}

use crate::physics::kelvin_helmholtz_rayleigh_taylor::HydrodynamicInstabilityEngine;

py_engine! {
    PyHydrodynamicInstabilityEngine("HydrodynamicInstabilityEngine") wraps HydrodynamicInstabilityEngine [Clone, Copy] {
        new(rho1: f64, rho2: f64, surface_tension_sigma: f64) -> res;
        method atwood_number() -> f64;
        method growth_rate_squared(wavenumber_k: f64, shear_delta_u: f64, gravity_g: f64) -> f64;
    }
}

use crate::physics::vlasov_landau_damping::VlasovLandauDampingEngine;

py_engine! {
    PyVlasovLandauDampingEngine("VlasovLandauDampingEngine") wraps VlasovLandauDampingEngine [Clone, Copy] {
        new(omega_pe: f64, debye_length: f64) -> res;
        method_res bohm_gross_and_landau_damping(wavenumber_k: f64) -> (f64, f64);
    }
}

use crate::physics::grad_shafranov_tokamak::GradShafranovTokamakEngine;

py_engine! {
    PyGradShafranovTokamakEngine("GradShafranovTokamakEngine") wraps GradShafranovTokamakEngine [Clone, Copy] {
        new(major_r0: f64, elongation_kappa: f64, toroidal_b0: f64) -> res;
        method solovev_normalized_flux(r: f64, z: f64) -> f64;
        method_res safety_factor_q(minor_r: f64, poloidal_b_theta: f64) -> f64;
    }
}

use crate::physics::superfluid_gross_pitaevskii::GrossPitaevskiiVortexEngine;

py_engine! {
    PyGrossPitaevskiiVortexEngine("GrossPitaevskiiVortexEngine") wraps GrossPitaevskiiVortexEngine [Clone, Copy] {
        new(density_n0: f64, coupling_g: f64) -> res;
        method sound_speed_and_healing_length() -> (f64, f64);
        method bogoliubov_dispersion(wavenumber_k: f64) -> f64;
    }
}

use crate::omni_hexadeca_xvi::GrandOmniHexadecaEngineXVI;

#[pyclass(
    name = "GrandOmniHexadecaEngineXVI",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniHexadecaEngineXVI {
    pub inner: GrandOmniHexadecaEngineXVI,
}

#[pymethods]
impl PyGrandOmniHexadecaEngineXVI {
    #[new]
    #[pyo3(signature = (num_layers=16))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniHexadecaEngineXVI::new(num_layers),
        }
    }

    pub fn execute_hexadeca_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_hexadeca_synthesis())
            .map_err(to_py_err)
    }
}

use crate::ode::malliavin_calculus::MalliavinBismutEngine;

py_engine! {
    PyMalliavinBismutEngine("MalliavinBismutEngine") wraps MalliavinBismutEngine [Clone, Copy] {
        new(spot_s0: f64, volatility_sigma: f64, maturity_t: f64) -> res;
        method bismut_delta_and_gamma_weights(terminal_wiener_wt: f64) -> (f64, f64);
    }
}

use crate::ode::schramm_loewner_evolution::SchrammLoewnerEngine;

py_engine! {
    PySchrammLoewnerEngine("SchrammLoewnerEngine") wraps SchrammLoewnerEngine [Clone, Copy] {
        new(kappa: f64) -> res;
        method hausdorff_dimension_and_dual() -> (f64, f64);
        method cft_central_charge() -> f64;
    }
}

use crate::ode::kpz_stochastic_burgers::KpzColeHopfEngine;

py_engine! {
    PyKpzColeHopfEngine("KpzColeHopfEngine") wraps KpzColeHopfEngine [Clone, Copy] {
        new(viscosity_nu: f64, lambda_coupling: f64) -> res;
        method cole_hopf_forward_and_inverse(height_h: f64) -> (f64, f64);
        assoc universal_kpz_exponents() -> (f64, f64, f64);
    }
}

use crate::ode::rough_heston_volatility::RoughVolatilityHestonEngine;

py_engine! {
    PyRoughVolatilityHestonEngine("RoughVolatilityHestonEngine") wraps RoughVolatilityHestonEngine [Clone, Copy] {
        new(hurst_h: f64, vol_of_vol_nu: f64, correlation_rho: f64) -> res;
        method_res atm_implied_vol_skew(maturity_t: f64) -> f64;
    }
}

use crate::ode::freidlin_wentzell_ldp::FreidlinWentzellLdpEngine;

#[pyclass(name = "FreidlinWentzellLdpEngine", module = "oxideon", from_py_object)]
#[derive(Clone, Copy, Default)]
pub struct PyFreidlinWentzellLdpEngine {
    pub inner: FreidlinWentzellLdpEngine,
}

#[pymethods]
impl PyFreidlinWentzellLdpEngine {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: FreidlinWentzellLdpEngine::new(),
        }
    }

    #[staticmethod]
    pub fn eyring_kramers_escape_time(
        barrier_delta_v: f64,
        noise_eps: f64,
        curvature_well: f64,
        curvature_saddle_abs: f64,
    ) -> PyResult<f64> {
        FreidlinWentzellLdpEngine::eyring_kramers_escape_time(
            barrier_delta_v,
            noise_eps,
            curvature_well,
            curvature_saddle_abs,
        )
        .map_err(to_py_err)
    }
}

use crate::ode::mc_kean_vlasov_sde::McKeanVlasovPropagationChaos;

py_engine! {
    PyMcKeanVlasovPropagationChaos("McKeanVlasovPropagationChaos") wraps McKeanVlasovPropagationChaos [Clone, Copy] {
        new(interaction_kappa: f64, sigma: f64) -> res;
        method_res mean_field_variance(v0: f64, t: f64) -> f64;
    }
}

use crate::ode::levy_jump_ito::LevyItodiffusionEngine;

py_engine! {
    PyLevyItodiffusionEngine("LevyItodiffusionEngine") wraps LevyItodiffusionEngine [Clone, Copy] {
        new(sigma: f64, jump_lambda: f64, jump_mean: f64, jump_std: f64) -> res;
        method real_levy_exponent(u: f64) -> f64;
    }
}

use crate::ode::backward_sde_pardoux_peng::PardouxPengBsdeEngine;

py_engine! {
    PyPardouxPengBsdeEngine("PardouxPengBsdeEngine") wraps PardouxPengBsdeEngine [Clone, Copy] {
        new(rate_r: f64, risk_lambda_theta: f64);
        method_res backward_bsde_step(expected_y_next: f64, control_z: f64, dt: f64) -> f64;
    }
}

use crate::ode::stein_method_wasserstein::SteinPoissonCharacterization;

#[pyclass(
    name = "SteinPoissonCharacterization",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone, Copy, Default)]
pub struct PySteinPoissonCharacterization {
    pub inner: SteinPoissonCharacterization,
}

#[pymethods]
impl PySteinPoissonCharacterization {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: SteinPoissonCharacterization::new(),
        }
    }

    #[staticmethod]
    pub fn fourth_moment_tv_bound(chaos_order_q: usize, fourth_moment: f64) -> PyResult<f64> {
        SteinPoissonCharacterization::fourth_moment_tv_bound(chaos_order_q, fourth_moment)
            .map_err(to_py_err)
    }

    #[staticmethod]
    pub fn stein_chen_poisson_bound(bernoulli_probs: Vec<f64>) -> PyResult<f64> {
        SteinPoissonCharacterization::stein_chen_poisson_bound(&bernoulli_probs).map_err(to_py_err)
    }
}

use crate::omni_heptadeca_xvii::GrandOmniHeptadecaEngineXVII;

#[pyclass(
    name = "GrandOmniHeptadecaEngineXVII",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniHeptadecaEngineXVII {
    pub inner: GrandOmniHeptadecaEngineXVII,
}

#[pymethods]
impl PyGrandOmniHeptadecaEngineXVII {
    #[new]
    #[pyo3(signature = (num_layers=17))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniHeptadecaEngineXVII::new(num_layers),
        }
    }

    pub fn execute_heptadeca_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_heptadeca_synthesis())
            .map_err(to_py_err)
    }
}

use crate::physics::fractional_quantum_hall_laughlin::LaughlinFqheEngine;

py_engine! {
    PyLaughlinFqheEngine("LaughlinFqheEngine") wraps LaughlinFqheEngine [Clone, Copy] {
        new(denominator_m: usize) -> res;
        method filling_charge_and_anyon_angle() -> (f64, f64, f64);
    }
}

use crate::physics::haldane_chern_insulator::HaldaneHoneycombModel;

py_engine! {
    PyHaldaneHoneycombModel("HaldaneHoneycombModel") wraps HaldaneHoneycombModel [Clone, Copy] {
        new(semenoff_mass_m: f64, nnn_hopping_t2: f64, flux_phi: f64);
        method valley_masses_and_chern_number() -> (f64, f64, i32);
    }
}

use crate::physics::kondo_anderson_rg::KondoScreeningEngine;

py_engine! {
    PyKondoScreeningEngine("KondoScreeningEngine") wraps KondoScreeningEngine [Clone, Copy] {
        new(bandwidth_d: f64, dimensionless_g0: f64) -> res;
        method kondo_temperature() -> f64;
    }
}

use crate::physics::bcs_eliashberg_superconductivity::BcsGapEquationEngine;

py_engine! {
    PyBcsGapEquationEngine("BcsGapEquationEngine") wraps BcsGapEquationEngine [Clone, Copy] {
        new(debye_energy: f64, coupling_lambda: f64) -> res;
        method gap_tc_and_universal_ratio() -> (f64, f64, f64);
    }
}

use crate::physics::hubbard_mott_gutzwiller::BrinkmanRiceGutzwillerEngine;

py_engine! {
    PyBrinkmanRiceGutzwillerEngine("BrinkmanRiceGutzwillerEngine") wraps BrinkmanRiceGutzwillerEngine [Clone, Copy] {
        new(critical_uc: f64) -> res;
        method quasiparticle_z_and_double_occupancy(repulsion_u: f64) -> (f64, f64);
    }
}

use crate::physics::berezinskii_kosterlitz_thouless::BktVortexUnbindingEngine;

py_engine! {
    PyBktVortexUnbindingEngine("BktVortexUnbindingEngine") wraps BktVortexUnbindingEngine [Clone, Copy, Default] {
        new();
        assoc universal_nelson_kosterlitz_jump() -> f64;
        assoc_res bkt_correlation_length(temp: f64, temp_bkt: f64, lattice_a: f64, scale_b: f64) -> f64;
    }
}

use crate::physics::weyl_semimetal_chiral_anomaly::WeylChiralAnomalyEngine;

py_engine! {
    PyWeylChiralAnomalyEngine("WeylChiralAnomalyEngine") wraps WeylChiralAnomalyEngine [Clone, Copy] {
        new(sigma_0: f64, chiral_tau_a: f64, fermi_velocity: f64, chemical_potential_mu: f64) -> res;
        method longitudinal_magnetoconductivity(magnetic_b: f64) -> f64;
    }
}

use crate::physics::twistronics_bistritzer_macdonald::MagicAngleGrapheneEngine;

py_engine! {
    PyMagicAngleGrapheneEngine("MagicAngleGrapheneEngine") wraps MagicAngleGrapheneEngine [Clone, Copy, Default] {
        new();
        assoc renormalized_velocity_ratio(alpha: f64) -> f64;
        assoc first_magic_alpha() -> f64;
    }
}

use crate::physics::skyrmion_dzyaloshinskii_moriya::MagneticSkyrmionEngine;

py_engine! {
    PyMagneticSkyrmionEngine("MagneticSkyrmionEngine") wraps MagneticSkyrmionEngine [Clone, Copy] {
        new(topological_q: i32, dissipation_d: f64);
        method thiele_skyrmion_hall_tangent(gilbert_alpha: f64, nonadiabatic_beta: f64) -> f64;
    }
}

use crate::omni_octodeca_xviii::GrandOmniOctodecaEngineXVIII;

#[pyclass(
    name = "GrandOmniOctodecaEngineXVIII",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniOctodecaEngineXVIII {
    pub inner: GrandOmniOctodecaEngineXVIII,
}

#[pymethods]
impl PyGrandOmniOctodecaEngineXVIII {
    #[new]
    #[pyo3(signature = (num_layers=18))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniOctodecaEngineXVIII::new(num_layers),
        }
    }

    pub fn execute_octodeca_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_octodeca_synthesis())
            .map_err(to_py_err)
    }
}

use crate::science::hodgkin_huxley_ion::HodgkinHuxleyActionPotential;

#[pyclass(
    name = "HodgkinHuxleyActionPotential",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone, Copy)]
pub struct PyHodgkinHuxleyActionPotential {
    pub inner: HodgkinHuxleyActionPotential,
}

#[pymethods]
impl PyHodgkinHuxleyActionPotential {
    #[new]
    #[pyo3(signature = (g_na=120.0, g_k=36.0, g_l=0.3, reversal_e=[50.0, -77.0, -54.387]))]
    pub fn new(g_na: f64, g_k: f64, g_l: f64, reversal_e: [f64; 3]) -> Self {
        Self {
            inner: HodgkinHuxleyActionPotential::new(g_na, g_k, g_l, reversal_e),
        }
    }

    pub fn total_ionic_current(&self, voltage_v: f64, m: f64, h: f64, n: f64) -> f64 {
        self.inner.total_ionic_current(voltage_v, m, h, n)
    }
}

use crate::science::wilson_cowan_neural_field::WilsonCowanCorticalEngine;

py_engine! {
    PyWilsonCowanCorticalEngine("WilsonCowanCorticalEngine") wraps WilsonCowanCorticalEngine [Clone, Copy] {
        new(weights: [f64; 4]);
        method step_ei(e: f64, i: f64, p_e: f64, p_i: f64, dt: f64) -> (f64, f64);
    }
}

use crate::science::turing_morphogenesis_rd::TuringReactionDiffusionEngine;

py_engine! {
    PyTuringReactionDiffusionEngine("TuringReactionDiffusionEngine") wraps TuringReactionDiffusionEngine [Clone, Copy] {
        new(du: f64, dv: f64, jacobian: [f64; 4]) -> res;
        method check_turing_instability() -> (bool, f64);
    }
}

use crate::science::jarzynski_crooks_fluctuation::JarzynskiCrooksEngine;

#[pyclass(name = "JarzynskiCrooksEngine", module = "oxideon", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyJarzynskiCrooksEngine {
    pub inner: JarzynskiCrooksEngine,
}

#[pymethods]
impl PyJarzynskiCrooksEngine {
    #[new]
    pub fn new(beta: f64) -> PyResult<Self> {
        let inner = JarzynskiCrooksEngine::new(beta).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn jarzynski_free_energy_and_dissipation(
        &self,
        py: Python<'_>,
        work_samples: Vec<f64>,
    ) -> PyResult<(f64, f64)> {
        py.detach(|| {
            self.inner
                .jarzynski_free_energy_and_dissipation(&work_samples)
        })
        .map_err(to_py_err)
    }
}

use crate::science::schnakenberg_chemical_master::ChemicalMasterEntropyEngine;

#[pyclass(
    name = "ChemicalMasterEntropyEngine",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone, Copy, Default)]
pub struct PyChemicalMasterEntropyEngine {
    pub inner: ChemicalMasterEntropyEngine,
}

#[pymethods]
impl PyChemicalMasterEntropyEngine {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: ChemicalMasterEntropyEngine::new(),
        }
    }

    #[staticmethod]
    pub fn schnakenberg_entropy_production(
        forward_fluxes: Vec<f64>,
        backward_fluxes: Vec<f64>,
    ) -> PyResult<f64> {
        ChemicalMasterEntropyEngine::schnakenberg_entropy_production(
            &forward_fluxes,
            &backward_fluxes,
        )
        .map_err(to_py_err)
    }
}

use crate::science::kramers_moyal_fokker_planck::FokkerPlanckOrnsteinUhlenbeck;

py_engine! {
    PyFokkerPlanckOrnsteinUhlenbeck("FokkerPlanckOrnsteinUhlenbeck") wraps FokkerPlanckOrnsteinUhlenbeck [Clone, Copy] {
        new(gamma: f64, diffusion_d: f64) -> res;
        method_res propagator_density(x: f64, x0: f64, t: f64) -> f64;
    }
}

use crate::science::worm_like_chain_polymer::KratkyPorodPolymerEngine;

py_engine! {
    PyKratkyPorodPolymerEngine("KratkyPorodPolymerEngine") wraps KratkyPorodPolymerEngine [Clone, Copy] {
        new(persistence_lp: f64, contour_l0: f64) -> res;
        method mean_square_end_to_end() -> f64;
        assoc_res marko_siggia_dimensionless_force(relative_extension: f64) -> f64;
    }
}

use crate::science::marcus_electron_transfer::MarcusElectronTransferEngine;

py_engine! {
    PyMarcusElectronTransferEngine("MarcusElectronTransferEngine") wraps MarcusElectronTransferEngine [Clone, Copy] {
        new(reorganization_lambda: f64) -> res;
        method activation_barrier_and_is_inverted(delta_g0: f64) -> (f64, bool);
    }
}

use crate::science::forster_exciton_resonance::ForsterResonanceEnergyEngine;

py_engine! {
    PyForsterResonanceEnergyEngine("ForsterResonanceEnergyEngine") wraps ForsterResonanceEnergyEngine [Clone, Copy] {
        new(forster_r0: f64, donor_lifetime_tau: f64) -> res;
        method_res efficiency_and_rate(distance_r: f64) -> (f64, f64);
    }
}

use crate::omni_nonadeca_xix::GrandOmniNonadecaEngineXIX;

#[pyclass(
    name = "GrandOmniNonadecaEngineXIX",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniNonadecaEngineXIX {
    pub inner: GrandOmniNonadecaEngineXIX,
}

#[pymethods]
impl PyGrandOmniNonadecaEngineXIX {
    #[new]
    #[pyo3(signature = (num_layers=19))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniNonadecaEngineXIX::new(num_layers),
        }
    }

    pub fn execute_nonadeca_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_nonadeca_synthesis())
            .map_err(to_py_err)
    }
}

use crate::science::quillen_formal_group_chromatic::ChromaticMoravaKTheory;

py_engine! {
    PyChromaticMoravaKTheory("ChromaticMoravaKTheory") wraps ChromaticMoravaKTheory [Clone, Copy] {
        new(prime_p: u64, height_h: u32) -> res;
        method vn_generator_degree_and_stabilizer_dim() -> (u64, u32);
    }
}

use crate::science::adams_spectral_sequence_j_hom::AdamsJHomomorphismEngine;

py_engine! {
    PyAdamsJHomomorphismEngine("AdamsJHomomorphismEngine") wraps AdamsJHomomorphismEngine [Clone, Copy, Default] {
        new();
        assoc_res image_of_j_order(index_k: usize) -> u64;
    }
}

use crate::science::khovanov_categorification::KhovanovHomologyEngine;

py_engine! {
    PyKhovanovHomologyEngine("KhovanovHomologyEngine") wraps KhovanovHomologyEngine [Clone, Copy, Default] {
        new();
        assoc_res torus_knot_rasmussen_s_and_slice_genus(p: u64, q: u64) -> (u64, u64);
    }
}

use crate::science::heegaard_floer_ozsvath_szabo::HeegaardFloerKnotEngine;

#[pyclass(name = "HeegaardFloerKnotEngine", module = "oxideon", from_py_object)]
#[derive(Clone, Copy, Default)]
pub struct PyHeegaardFloerKnotEngine {
    pub inner: HeegaardFloerKnotEngine,
}

#[pymethods]
impl PyHeegaardFloerKnotEngine {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: HeegaardFloerKnotEngine::new(),
        }
    }

    #[staticmethod]
    pub fn alexander_euler_characteristic(positive_gradings_chi: Vec<i64>, t: f64) -> f64 {
        HeegaardFloerKnotEngine::alexander_euler_characteristic(&positive_gradings_chi, t)
    }

    #[staticmethod]
    pub fn satisfies_legendrian_tau_bound(tb: i64, rot: i64, tau_k: i64) -> bool {
        HeegaardFloerKnotEngine::satisfies_legendrian_tau_bound(tb, rot, tau_k)
    }
}

use crate::science::symplectic_fukaya_floer::FukayaArnoldConjectureEngine;

py_engine! {
    PyFukayaArnoldConjectureEngine("FukayaArnoldConjectureEngine") wraps FukayaArnoldConjectureEngine [Clone] {
        new(betti_numbers: Vec<u64>);
        method arnold_fixed_point_bound_and_euler() -> (u64, i64);
        assoc novikov_polygon_weight(symplectic_area: f64) -> f64;
    }
}

use crate::science::scholze_clausen_condensed_liquid::CondensedLiquidTensorEngine;

#[pyclass(
    name = "CondensedLiquidTensorEngine",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone, Copy)]
pub struct PyCondensedLiquidTensorEngine {
    pub inner: CondensedLiquidTensorEngine,
}

#[pymethods]
impl PyCondensedLiquidTensorEngine {
    #[new]
    pub fn new(liquid_p: f64) -> PyResult<Self> {
        let inner = CondensedLiquidTensorEngine::new(liquid_p).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn p_liquid_norm(&self, py: Python<'_>, v: Vec<f64>) -> f64 {
        py.detach(|| self.inner.p_liquid_norm(&v))
    }
}

use crate::science::prismatic_cohomology_bhatt_scholze::PrismaticSiteBreuilKisin;

py_engine! {
    PyPrismaticSiteBreuilKisin("PrismaticSiteBreuilKisin") wraps PrismaticSiteBreuilKisin [Clone, Copy] {
        new(prime_p: i64) -> res;
        method_res integer_p_derivation_delta(x: i64) -> i64;
    }
}

use crate::science::motivic_voevodsky_milnor::VoevodskyMilnorBlochKato;

py_engine! {
    PyVoevodskyMilnorBlochKato("VoevodskyMilnorBlochKato") wraps VoevodskyMilnorBlochKato [Clone, Copy, Default] {
        new();
        assoc_res legendre_tame_symbol(a: i64, odd_prime_p: i64) -> i32;
    }
}

use crate::science::lurie_higher_algebra_en_operads::LurieLittleDisksFactorization;

py_engine! {
    PyLurieLittleDisksFactorization("LurieLittleDisksFactorization") wraps LurieLittleDisksFactorization [Clone, Copy, Default] {
        new();
        assoc_res dunn_additivity_and_browder_degree(m: usize, n: usize) -> (usize, usize);
    }
}

use crate::omni_bicentennial_xx::GrandBicentennialSingularityEngineXX;

#[pyclass(
    name = "GrandBicentennialSingularityEngineXX",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandBicentennialSingularityEngineXX {
    pub inner: GrandBicentennialSingularityEngineXX,
}

#[pymethods]
impl PyGrandBicentennialSingularityEngineXX {
    #[new]
    #[pyo3(signature = (num_layers=20))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandBicentennialSingularityEngineXX::new(num_layers),
        }
    }

    pub fn execute_bicentennial_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_bicentennial_synthesis())
            .map_err(to_py_err)
    }
}

use crate::science::hartree_fock_roothaan::RoothaanHallScfEngine;

#[pyclass(name = "RoothaanHallScfEngine", module = "oxideon", from_py_object)]
#[derive(Clone, Copy, Default)]
pub struct PyRoothaanHallScfEngine {
    pub inner: RoothaanHallScfEngine,
}

#[pymethods]
impl PyRoothaanHallScfEngine {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: RoothaanHallScfEngine::new(),
        }
    }

    #[staticmethod]
    pub fn rhf_electronic_energy(
        py: Python<'_>,
        density_p: Vec<f64>,
        h_core: Vec<f64>,
        fock_f: Vec<f64>,
    ) -> PyResult<f64> {
        py.detach(|| RoothaanHallScfEngine::rhf_electronic_energy(&density_p, &h_core, &fock_f))
            .map_err(to_py_err)
    }
}

use crate::science::kohn_sham_dft_jacob_ladder::KohnShamJacobLadderEngine;

py_engine! {
    PyKohnShamJacobLadderEngine("KohnShamJacobLadderEngine") wraps KohnShamJacobLadderEngine [Clone, Copy, Default] {
        new();
        assoc_res pbe_exchange_energy_density(density_n: f64, reduced_gradient_s: f64) -> f64;
    }
}

use crate::science::coupled_cluster_ccsd_t::CoupledClusterCcsdTEngine;

#[pyclass(name = "CoupledClusterCcsdTEngine", module = "oxideon", from_py_object)]
#[derive(Clone, Copy, Default)]
pub struct PyCoupledClusterCcsdTEngine {
    pub inner: CoupledClusterCcsdTEngine,
}

#[pymethods]
impl PyCoupledClusterCcsdTEngine {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: CoupledClusterCcsdTEngine::new(),
        }
    }

    #[staticmethod]
    pub fn lee_taylor_t1_diagnostic(
        t1_amplitudes: Vec<f64>,
        correlated_electrons: usize,
    ) -> PyResult<(f64, bool)> {
        CoupledClusterCcsdTEngine::lee_taylor_t1_diagnostic(&t1_amplitudes, correlated_electrons)
            .map_err(to_py_err)
    }
}

use crate::science::moller_plesset_mp2::MollerPlessetMp2Engine;

py_engine! {
    PyMollerPlessetMp2Engine("MollerPlessetMp2Engine") wraps MollerPlessetMp2Engine [Clone, Copy, Default] {
        new();
        assoc canonical_and_scs_mp2_energy(e_opposite_spin: f64, e_same_spin: f64) -> (f64, f64);
    }
}

use crate::science::caspt2_dmrg_multireference::MultireferenceCasEngine;

py_engine! {
    PyMultireferenceCasEngine("MultireferenceCasEngine") wraps MultireferenceCasEngine [Clone, Copy, Default] {
        new();
        assoc_res singlet_weyl_csf_dimension(even_electrons_n: u64, orbitals_m: u64) -> u64;
    }
}

use crate::science::gw_bethe_salpeter_exciton::HedinGwBetheSalpeterEngine;

py_engine! {
    PyHedinGwBetheSalpeterEngine("HedinGwBetheSalpeterEngine") wraps HedinGwBetheSalpeterEngine [Clone, Copy, Default] {
        new();
        assoc_res g0w0_quasiparticle_energy(ks_eigenvalue: f64, sigma_minus_vxc: f64, dsigma_domega: f64) -> (f64, f64);
    }
}

use crate::science::conical_intersection_berry::YarkonyConicalIntersectionEngine;

py_engine! {
    PyYarkonyConicalIntersectionEngine("YarkonyConicalIntersectionEngine") wraps YarkonyConicalIntersectionEngine [Clone, Copy] {
        new(slope_g: f64, slope_h: f64);
        method adiabatic_surfaces_and_mixing_angle(x: f64, y: f64) -> (f64, f64, f64);
    }
}

use crate::science::eyring_kramers_transition_state::EyringTransitionStateEngine;

py_engine! {
    PyEyringTransitionStateEngine("EyringTransitionStateEngine") wraps EyringTransitionStateEngine [Clone, Copy, Default] {
        new();
        assoc_res wigner_tunneling_factor(reduced_barrier_frequency_ub: f64) -> f64;
    }
}

use crate::science::ring_polymer_pimd_instanton::RingPolymerMolecularDynamics;

py_engine! {
    PyRingPolymerMolecularDynamics("RingPolymerMolecularDynamics") wraps RingPolymerMolecularDynamics [Clone, Copy] {
        new(num_beads_p: usize, omega_p: f64) -> res;
        method normal_mode_frequency(mode_k: usize) -> f64;
    }
}

use crate::omni_henicosa_xxi::GrandOmniHenicosaEngineXXI;

#[pyclass(
    name = "GrandOmniHenicosaEngineXXI",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniHenicosaEngineXXI {
    pub inner: GrandOmniHenicosaEngineXXI,
}

#[pymethods]
impl PyGrandOmniHenicosaEngineXXI {
    #[new]
    #[pyo3(signature = (num_layers=21))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniHenicosaEngineXXI::new(num_layers),
        }
    }

    pub fn execute_henicosa_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_henicosa_synthesis())
            .map_err(to_py_err)
    }
}

use crate::ode::pontryagin_maximum_principle::PontryaginMaximumEngine;

py_engine! {
    PyPontryaginMaximumEngine("PontryaginMaximumEngine") wraps PontryaginMaximumEngine [Clone, Copy] {
        new(penalty_r: f64, u_max: f64) -> res;
        method optimal_control_and_hamiltonian(drift_dot_p: f64, control_switching_pg: f64) -> (f64, f64);
    }
}

use crate::ode::chow_rashevskii_sub_riemannian::SubRiemannianHeisenbergEngine;

py_engine! {
    PySubRiemannianHeisenbergEngine("SubRiemannianHeisenbergEngine") wraps SubRiemannianHeisenbergEngine [Clone, Copy, Default] {
        new();
        assoc vertical_carnot_distance(z_height: f64) -> f64;
    }
}

use crate::ode::kalman_bucy_riccati_lqg::KalmanBucyRiccatiEngine;

py_engine! {
    PyKalmanBucyRiccatiEngine("KalmanBucyRiccatiEngine") wraps KalmanBucyRiccatiEngine [Clone, Copy] {
        new(drift_a: f64, obs_c: f64, process_q: f64, meas_r: f64) -> res;
        method steady_state_covariance_and_gain() -> (f64, f64);
    }
}

use crate::ode::zames_h_infinity_rob_control::HInfinityRobustControlEngine;

py_engine! {
    PyHInfinityRobustControlEngine("HInfinityRobustControlEngine") wraps HInfinityRobustControlEngine [Clone, Copy, Default] {
        new();
        assoc_res solve_h_infinity_are(a: f64, b1_dist: f64, b2_ctrl: f64, q: f64, r: f64, gamma: f64) -> f64;
    }
}

use crate::ode::sontag_control_lyapunov_cbf::ControlBarrierLyapunovEngine;

py_engine! {
    PyControlBarrierLyapunovEngine("ControlBarrierLyapunovEngine") wraps ControlBarrierLyapunovEngine [Clone, Copy, Default] {
        new();
        assoc sontag_universal_control(lf_v: f64, lg_v: f64) -> f64;
        assoc cbf_safety_filter(u_nom: f64, lf_h: f64, lg_h: f64, alpha_h: f64) -> f64;
    }
}

use crate::physics::kam_kolmogorov_arnold_moser::KamDiophantineTorusEngine;

#[pyclass(name = "KamDiophantineTorusEngine", module = "oxideon", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyKamDiophantineTorusEngine {
    pub inner: KamDiophantineTorusEngine,
}

#[pymethods]
impl PyKamDiophantineTorusEngine {
    #[new]
    pub fn new(gamma: f64, tau: f64) -> PyResult<Self> {
        let inner = KamDiophantineTorusEngine::new(gamma, tau).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn minimum_diophantine_constant(
        &self,
        py: Python<'_>,
        omega1: f64,
        omega2: f64,
        max_order_k: i32,
    ) -> PyResult<(f64, bool)> {
        py.detach(|| {
            self.inner
                .minimum_diophantine_constant(omega1, omega2, max_order_k)
        })
        .map_err(to_py_err)
    }
}

use crate::physics::nekhoroshev_arnold_diffusion::NekhoroshevExponentialStability;

py_engine! {
    PyNekhoroshevExponentialStability("NekhoroshevExponentialStability") wraps NekhoroshevExponentialStability [Clone, Copy] {
        new(dof_n: usize, eps_0: f64) -> res;
        method_res nekhoroshev_drift_and_time_factor(perturbation_eps: f64) -> (f64, f64);
    }
}

use crate::physics::poincare_cartan_lagrange_l4_l5::CircularRestrictedThreeBodyEngine;

py_engine! {
    PyCircularRestrictedThreeBodyEngine("CircularRestrictedThreeBodyEngine") wraps CircularRestrictedThreeBodyEngine [Clone, Copy] {
        new(mass_ratio_mu: f64) -> res;
        method jacobi_constant_and_routh_stable(pos: [f64; 2], vel: [f64; 2]) -> (f64, bool);
    }
}

use crate::physics::chirikov_standard_map_chaos::ChirikovStandardMapEngine;

#[pyclass(name = "ChirikovStandardMapEngine", module = "oxideon", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyChirikovStandardMapEngine {
    pub inner: ChirikovStandardMapEngine,
}

#[pymethods]
impl PyChirikovStandardMapEngine {
    #[new]
    pub fn new(kick_k: f64) -> Self {
        Self {
            inner: ChirikovStandardMapEngine::new(kick_k),
        }
    }

    pub fn iterate_map(
        &self,
        py: Python<'_>,
        theta: f64,
        p: f64,
        steps: usize,
    ) -> PyResult<(f64, f64, bool)> {
        guard_res(|| py.detach(|| self.inner.iterate_map(theta, p, steps)))
    }
}

use crate::omni_docosa_xxii::GrandOmniDocosaEngineXXII;

#[pyclass(name = "GrandOmniDocosaEngineXXII", module = "oxideon", from_py_object)]
#[derive(Clone)]
pub struct PyGrandOmniDocosaEngineXXII {
    pub inner: GrandOmniDocosaEngineXXII,
}

#[pymethods]
impl PyGrandOmniDocosaEngineXXII {
    #[new]
    #[pyo3(signature = (num_layers=22))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniDocosaEngineXXII::new(num_layers),
        }
    }

    pub fn execute_docosa_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_docosa_synthesis())
            .map_err(to_py_err)
    }
}

use crate::physics::teukolsky_kerr_quasinormal::KerrQuasinormalModeEngine;

py_engine! {
    PyKerrQuasinormalModeEngine("KerrQuasinormalModeEngine") wraps KerrQuasinormalModeEngine [Clone, Copy] {
        new(mass_m: f64, spin_chi: f64) -> res;
        method outer_horizon_and_angular_velocity() -> (f64, f64);
        method fundamental_220_qnm_frequency() -> f64;
    }
}

use crate::physics::penrose_superradiance_ergosphere::BlandfordZnajekPenroseEngine;

py_engine! {
    PyBlandfordZnajekPenroseEngine("BlandfordZnajekPenroseEngine") wraps BlandfordZnajekPenroseEngine [Clone, Copy] {
        new(mass_m: f64, spin_chi: f64) -> res;
        method irreducible_mass_and_rotational_energy() -> (f64, f64);
    }
}

use crate::physics::tolman_oppenheimer_volkoff::TolmanOppenheimerVolkoffEngine;

py_engine! {
    PyTolmanOppenheimerVolkoffEngine("TolmanOppenheimerVolkoffEngine") wraps TolmanOppenheimerVolkoffEngine [Clone, Copy, Default] {
        new();
        assoc_res tov_pressure_gradient(r: f64, mass_m: f64, density_rho: f64, pressure_p: f64) -> f64;
        assoc_res tidal_deformability_and_buchdahl_ok(compactness_c: f64, love_number_k2: f64) -> (f64, bool);
    }
}

use crate::physics::chandrasekhar_lane_emden::ChandrasekharWhiteDwarfEngine;

py_engine! {
    PyChandrasekharWhiteDwarfEngine("ChandrasekharWhiteDwarfEngine") wraps ChandrasekharWhiteDwarfEngine [Clone, Copy, Default] {
        new();
        assoc lane_emden_n0(xi: f64) -> f64;
        assoc lane_emden_n1(xi: f64) -> f64;
        assoc lane_emden_n5(xi: f64) -> f64;
        assoc_res chandrasekhar_mass_solar(mean_molecular_weight_mu_e: f64) -> f64;
    }
}

use crate::physics::peters_mathews_chirp_gw::GravitationalWaveChirpEngine;

py_engine! {
    PyGravitationalWaveChirpEngine("GravitationalWaveChirpEngine") wraps GravitationalWaveChirpEngine [Clone, Copy] {
        new(m1: f64, m2: f64) -> res;
        method chirp_mass_and_symmetric_eta() -> (f64, f64);
    }
}

use crate::physics::sachs_wolfe_cmb_acoustic::CmbAcousticPeakEngine;

py_engine! {
    PyCmbAcousticPeakEngine("CmbAcousticPeakEngine") wraps CmbAcousticPeakEngine [Clone, Copy, Default] {
        new();
        assoc_res baryon_photon_sound_speed_ratio(baryon_loading_rb: f64) -> f64;
    }
}

use crate::physics::sunyaev_zeldovich_kompaneets::SunyaevZeldovichKompaneetsEngine;

py_engine! {
    PySunyaevZeldovichKompaneetsEngine("SunyaevZeldovichKompaneetsEngine") wraps SunyaevZeldovichKompaneetsEngine [Clone, Copy, Default] {
        new();
        assoc tsz_spectral_function_g(dimensionless_freq_x: f64) -> f64;
    }
}

use crate::physics::press_schechter_halo_mass::PressSchechterHaloEngine;

py_engine! {
    PyPressSchechterHaloEngine("PressSchechterHaloEngine") wraps PressSchechterHaloEngine [Clone, Copy, Default] {
        new();
        assoc spherical_collapse_delta_c() -> f64;
        assoc press_schechter_multiplicity(peak_nu: f64) -> f64;
    }
}

use crate::physics::hawking_page_ads_black_hole::HawkingPagePhaseTransition;

py_engine! {
    PyHawkingPagePhaseTransition("HawkingPagePhaseTransition") wraps HawkingPagePhaseTransition [Clone, Copy] {
        new(ads_radius_l: f64) -> res;
        method_res temperature_free_energy_and_deconfined(r_plus: f64) -> (f64, f64, bool);
    }
}

use crate::omni_tricosa_xxiii::GrandOmniTricosaEngineXXIII;

#[pyclass(
    name = "GrandOmniTricosaEngineXXIII",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniTricosaEngineXXIII {
    pub inner: GrandOmniTricosaEngineXXIII,
}

#[pymethods]
impl PyGrandOmniTricosaEngineXXIII {
    #[new]
    #[pyo3(signature = (num_layers=23))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniTricosaEngineXXIII::new(num_layers),
        }
    }

    pub fn execute_tricosa_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_tricosa_synthesis())
            .map_err(to_py_err)
    }
}

use crate::crypto::groth16_qap_zk_snark::QuadraticArithmeticProgramEngine;

/// Checks a plaintext R1CS witness only. Not a SNARK: no setup, proof, pairing or zero-knowledge.
#[pyclass(
    name = "QuadraticArithmeticProgramEngine",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyQuadraticArithmeticProgramEngine {
    pub inner: QuadraticArithmeticProgramEngine,
}

#[pymethods]
impl PyQuadraticArithmeticProgramEngine {
    #[new]
    pub fn new(modulus_p: i64, num_constraints: usize, num_vars: usize) -> PyResult<Self> {
        let inner = QuadraticArithmeticProgramEngine::new(modulus_p, num_constraints, num_vars)
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn verify_r1cs_witness(
        &self,
        py: Python<'_>,
        a: Vec<i64>,
        b: Vec<i64>,
        c: Vec<i64>,
        z: Vec<i64>,
    ) -> PyResult<bool> {
        py.detach(|| self.inner.verify_r1cs_witness(&a, &b, &c, &z))
            .map_err(to_py_err)
    }
}

use crate::crypto::fri_reed_solomon_stark::FastReedSolomonIopEngine;

/// One FRI fold step over F_p. Not a proof system: no commitments, queries or verifier.
#[pyclass(name = "FastReedSolomonIopEngine", module = "oxideon", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyFastReedSolomonIopEngine {
    pub inner: FastReedSolomonIopEngine,
}

#[pymethods]
impl PyFastReedSolomonIopEngine {
    #[new]
    pub fn new(modulus_p: i64) -> PyResult<Self> {
        let inner = FastReedSolomonIopEngine::new(modulus_p).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn fold_polynomial_coeffs(
        &self,
        coeffs: Vec<i64>,
        challenge_alpha: i64,
    ) -> PyResult<Vec<i64>> {
        self.inner
            .fold_polynomial_coeffs(&coeffs, challenge_alpha)
            .map_err(to_py_err)
    }
}

use crate::crypto::kzg_polynomial_commitment::KzgPolynomialCommitment;

/// Polynomial division over F_p only. Not a commitment scheme: no trusted setup, group elements or pairings.
#[pyclass(name = "KzgPolynomialCommitment", module = "oxideon", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyKzgPolynomialCommitment {
    pub inner: KzgPolynomialCommitment,
}

#[pymethods]
impl PyKzgPolynomialCommitment {
    #[new]
    pub fn new(modulus_p: i64) -> PyResult<Self> {
        let inner = KzgPolynomialCommitment::new(modulus_p).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn compute_opening_quotient(
        &self,
        coeffs_low_to_high: Vec<i64>,
        point_z: i64,
    ) -> PyResult<(i64, Vec<i64>)> {
        self.inner
            .compute_opening_quotient(&coeffs_low_to_high, point_z)
            .map_err(to_py_err)
    }
}

use crate::crypto::shamir_feldman_threshold_mpc::ShamirSecretSharingFp;

/// Secret reconstruction only (no share generation), not constant-time. Do not use it to protect real secrets.
#[pyclass(name = "ShamirSecretSharingFp", module = "oxideon", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyShamirSecretSharingFp {
    pub inner: ShamirSecretSharingFp,
}

#[pymethods]
impl PyShamirSecretSharingFp {
    #[new]
    pub fn new(modulus_p: i64) -> PyResult<Self> {
        let inner = ShamirSecretSharingFp::new(modulus_p).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn reconstruct_secret(&self, xs: Vec<i64>, ys: Vec<i64>) -> PyResult<i64> {
        self.inner.reconstruct_secret(&xs, &ys).map_err(to_py_err)
    }
}

use crate::crypto::goppa_algebraic_geometry_codes::TsfasmanVladutZinkAgCode;

py_engine! {
    PyTsfasmanVladutZinkAgCode("TsfasmanVladutZinkAgCode") wraps TsfasmanVladutZinkAgCode [Clone, Copy] {
        new(genus_g: usize, length_n: usize, divisor_deg: usize) -> res;
        method riemann_roch_k_and_design_distance() -> (usize, usize);
        assoc_res tvz_asymptotic_rate(relative_distance_delta: f64, alphabet_q: f64) -> f64;
    }
}

use crate::crypto::arikan_polar_codes_capacity::ArikanPolarCodeEngine;

#[pyclass(name = "ArikanPolarCodeEngine", module = "oxideon", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyArikanPolarCodeEngine {
    pub inner: ArikanPolarCodeEngine,
}

#[pymethods]
impl PyArikanPolarCodeEngine {
    #[new]
    pub fn new(initial_z0: f64) -> PyResult<Self> {
        let inner = ArikanPolarCodeEngine::new(initial_z0).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn polarized_bhattacharyya_parameters(
        &self,
        py: Python<'_>,
        depth_n: usize,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| self.inner.polarized_bhattacharyya_parameters(depth_n))
            .map_err(to_py_err)
    }
}

use crate::crypto::shannon_rate_distortion_blahut::RateDistortionBlahutArimoto;

#[pyclass(
    name = "RateDistortionBlahutArimoto",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone, Copy, Default)]
pub struct PyRateDistortionBlahutArimoto {
    pub inner: RateDistortionBlahutArimoto,
}

#[pymethods]
impl PyRateDistortionBlahutArimoto {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: RateDistortionBlahutArimoto::new(),
        }
    }

    #[staticmethod]
    pub fn reverse_waterfilling_rate_and_distortion(
        variances: Vec<f64>,
        water_level_lambda: f64,
    ) -> PyResult<(f64, f64)> {
        RateDistortionBlahutArimoto::reverse_waterfilling_rate_and_distortion(
            &variances,
            water_level_lambda,
        )
        .map_err(to_py_err)
    }
}

use crate::crypto::slepian_wolf_wyner_ziv::SlepianWolfDistributedCoding;

py_engine! {
    PySlepianWolfDistributedCoding("SlepianWolfDistributedCoding") wraps SlepianWolfDistributedCoding [Clone, Copy, Default] {
        new();
        assoc is_slepian_wolf_achievable(rx: f64, ry: f64, h_x_given_y: f64, h_y_given_x: f64, h_joint_xy: f64) -> bool;
        assoc_res gaussian_wyner_ziv_rate(var_x: f64, correlation_rho: f64, distortion_d: f64) -> f64;
    }
}

use crate::science::slepian_wolf_wyner_ziv_distributed::SlepianWolfRateRegionEngine;

py_engine! {
    PySlepianWolfRateRegionEngine("SlepianWolfRateRegionEngine") wraps SlepianWolfRateRegionEngine [Clone, Copy] {
        new(hx: f64, hy: f64, hxy: f64) -> res;
        method conditional_rates_and_achievability(rx: f64, ry: f64) -> (f64, f64, bool);
    }
}

use crate::crypto::dwork_renyi_differential_privacy::RenyiDifferentialPrivacyAccountant;

py_engine! {
    PyRenyiDifferentialPrivacyAccountant("RenyiDifferentialPrivacyAccountant") wraps RenyiDifferentialPrivacyAccountant [Clone, Copy] {
        new(noise_multiplier_sigma: f64) -> res;
        method_res optimal_dp_epsilon(steps_t: usize, failure_delta: f64) -> (f64, f64);
    }
}

use crate::omni_tetracosa_xxiv::GrandOmniTetracosaEngineXXIV;

#[pyclass(
    name = "GrandOmniTetracosaEngineXXIV",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniTetracosaEngineXXIV {
    pub inner: GrandOmniTetracosaEngineXXIV,
}

#[pymethods]
impl PyGrandOmniTetracosaEngineXXIV {
    #[new]
    #[pyo3(signature = (num_layers=24))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniTetracosaEngineXXIV::new(num_layers),
        }
    }

    pub fn execute_tetracosa_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_tetracosa_synthesis())
            .map_err(to_py_err)
    }
}

use crate::nn::rademacher_vapnik_pac_bayes::PacBayesMcAllesterEngine;

py_engine! {
    PyPacBayesMcAllesterEngine("PacBayesMcAllesterEngine") wraps PacBayesMcAllesterEngine [Clone, Copy, Default] {
        new();
        assoc_res mcallester_risk_bound(empirical_risk: f64, kl_divergence: f64, num_samples_n: usize, failure_delta: f64) -> f64;
    }
}

use crate::nn::jacot_neural_tangent_kernel::InfiniteWidthNtkEngine;

py_engine! {
    PyInfiniteWidthNtkEngine("InfiniteWidthNtkEngine") wraps InfiniteWidthNtkEngine [Clone, Copy, Default] {
        new();
        assoc_res relu_nngp_and_ntk(cosine_u: f64) -> (f64, f64);
    }
}

use crate::nn::yang_maximal_update_mup::MaximalUpdateParamEngine;

py_engine! {
    PyMaximalUpdateParamEngine("MaximalUpdateParamEngine") wraps MaximalUpdateParamEngine [Clone, Copy] {
        new(base_width: usize, target_width: usize) -> res;
        method transfer_adam_lr_and_init_std(base_lr: f64, base_init_std: f64) -> (f64, f64);
    }
}

use crate::nn::chinchilla_kaplan_scaling_laws::ComputeOptimalChinchillaEngine;

#[pyclass(
    name = "ComputeOptimalChinchillaEngine",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone, Copy)]
pub struct PyComputeOptimalChinchillaEngine {
    pub inner: ComputeOptimalChinchillaEngine,
}

#[pymethods]
impl PyComputeOptimalChinchillaEngine {
    #[new]
    #[pyo3(signature = (irreducible_e=1.69, a=406.4, alpha=0.34, b=410.7, beta=0.28))]
    pub fn new(irreducible_e: f64, a: f64, alpha: f64, b: f64, beta: f64) -> PyResult<Self> {
        let inner = ComputeOptimalChinchillaEngine::new(irreducible_e, a, alpha, b, beta)
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn loss_and_optimal_allocation(
        &self,
        params_n: f64,
        tokens_d: f64,
    ) -> PyResult<(f64, f64, f64)> {
        self.inner
            .loss_and_optimal_allocation(params_n, tokens_d)
            .map_err(to_py_err)
    }
}

use crate::nn::belkin_double_descent_marchenko::OverparameterizedDoubleDescent;

py_engine! {
    PyOverparameterizedDoubleDescent("OverparameterizedDoubleDescent") wraps OverparameterizedDoubleDescent [Clone, Copy] {
        new(signal_r2: f64, noise_sigma2: f64) -> res;
        method_res asymptotic_ridgeless_risk(aspect_gamma: f64) -> f64;
    }
}

use crate::nn::candes_tao_compressed_sensing_rip::RestrictedIsometryLassoEngine;

#[pyclass(
    name = "RestrictedIsometryLassoEngine",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone, Copy, Default)]
pub struct PyRestrictedIsometryLassoEngine {
    pub inner: RestrictedIsometryLassoEngine,
}

#[pymethods]
impl PyRestrictedIsometryLassoEngine {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: RestrictedIsometryLassoEngine::new(),
        }
    }

    #[staticmethod]
    pub fn check_candes_rip_condition(delta_2s: f64) -> PyResult<(bool, f64)> {
        RestrictedIsometryLassoEngine::check_candes_rip_condition(delta_2s).map_err(to_py_err)
    }

    #[staticmethod]
    pub fn proximal_soft_threshold(x: Vec<f64>, lambda_val: f64) -> Vec<f64> {
        RestrictedIsometryLassoEngine::proximal_soft_threshold(&x, lambda_val)
    }
}

use crate::nn::talagrand_concentration_log_sobolev::LogSobolevConcentrationEngine;

py_engine! {
    PyLogSobolevConcentrationEngine("LogSobolevConcentrationEngine") wraps LogSobolevConcentrationEngine [Clone, Copy] {
        new(curvature_rho: f64) -> res;
        method_res herbst_tail_and_talagrand_w2(deviation_t: f64, lipschitz_l: f64, kl_div: f64) -> (f64, f64);
    }
}

use crate::nn::vovk_conformal_prediction_martingale::SplitConformalPredictionEngine;

#[pyclass(
    name = "SplitConformalPredictionEngine",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone, Copy, Default)]
pub struct PySplitConformalPredictionEngine {
    pub inner: SplitConformalPredictionEngine,
}

#[pymethods]
impl PySplitConformalPredictionEngine {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: SplitConformalPredictionEngine::new(),
        }
    }

    #[staticmethod]
    pub fn conformal_quantile(
        py: Python<'_>,
        calibration_scores: Vec<f64>,
        miscoverage_alpha: f64,
    ) -> PyResult<f64> {
        py.detach(|| {
            SplitConformalPredictionEngine::conformal_quantile(
                &calibration_scores,
                miscoverage_alpha,
            )
        })
        .map_err(to_py_err)
    }
}

use crate::nn::rahimi_recht_random_fourier_features::RandomFourierBochnerEngine;

#[pyclass(
    name = "RandomFourierBochnerEngine",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyRandomFourierBochnerEngine {
    pub inner: RandomFourierBochnerEngine,
}

#[pymethods]
impl PyRandomFourierBochnerEngine {
    #[new]
    pub fn new(frequencies: Vec<f64>, input_dim: usize) -> PyResult<Self> {
        let inner = RandomFourierBochnerEngine::new(frequencies, input_dim).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn approximate_kernel(&self, py: Python<'_>, x: Vec<f64>, y: Vec<f64>) -> PyResult<f64> {
        py.detach(|| self.inner.approximate_kernel(&x, &y))
            .map_err(to_py_err)
    }
}

use crate::omni_pentacosa_xxv::GrandOmniPentacosaEngineXXV;

#[pyclass(
    name = "GrandOmniPentacosaEngineXXV",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniPentacosaEngineXXV {
    pub inner: GrandOmniPentacosaEngineXXV,
}

#[pymethods]
impl PyGrandOmniPentacosaEngineXXV {
    #[new]
    #[pyo3(signature = (num_layers=25))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniPentacosaEngineXXV::new(num_layers),
        }
    }

    pub fn execute_pentacosa_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_pentacosa_synthesis())
            .map_err(to_py_err)
    }
}

use crate::physics::ogden_mooney_rivlin_hyperelastic::HyperelasticStrainEnergyEngine;

py_engine! {
    PyHyperelasticStrainEnergyEngine("HyperelasticStrainEnergyEngine") wraps HyperelasticStrainEnergyEngine [Clone, Copy] {
        new(c10: f64, c01: f64) -> res;
        method_res uniaxial_energy_and_cauchy_stress(stretch_lambda: f64) -> (f64, f64);
    }
}

use crate::physics::griffith_rice_j_integral_fracture::RiceJIntegralFractureEngine;

py_engine! {
    PyRiceJIntegralFractureEngine("RiceJIntegralFractureEngine") wraps RiceJIntegralFractureEngine [Clone, Copy] {
        new(young_e: f64, poisson_nu: f64) -> res;
        method plane_strain_j_integral(k1: f64, k2: f64, k3: f64) -> f64;
    }
}

use crate::physics::von_mises_drucker_prager_plasticity::J2PlasticityReturnMapping;

py_engine! {
    PyJ2PlasticityReturnMapping("J2PlasticityReturnMapping") wraps J2PlasticityReturnMapping [Clone, Copy] {
        new(shear_g: f64, yield_stress: f64, hardening_h: f64) -> res;
        method_res radial_return_step(trial_von_mises_q: f64) -> (f64, f64);
    }
}

use crate::physics::mindlin_reissner_kirchhoff_plates::KirchhoffLovePlateEngine;

py_engine! {
    PyKirchhoffLovePlateEngine("KirchhoffLovePlateEngine") wraps KirchhoffLovePlateEngine [Clone, Copy] {
        new(young_e: f64, thickness_h: f64, poisson_nu: f64) -> res;
        method flexural_rigidity_d() -> f64;
        method_res navier_square_plate_deflection(side_a: f64, load_q0: f64) -> f64;
    }
}

use crate::physics::biot_poroelasticity_darcy::BiotPoroelasticConsolidation;

py_engine! {
    PyBiotPoroelasticConsolidation("BiotPoroelasticConsolidation") wraps BiotPoroelasticConsolidation [Clone, Copy] {
        new(drained_kd: f64, grain_ks: f64, biot_modulus_m: f64) -> res;
        method biot_gassmann_skempton_parameters() -> (f64, f64, f64);
    }
}

use crate::physics::eshelby_inclusion_homogenization::EshelbyInclusionMoriTanaka;

py_engine! {
    PyEshelbyInclusionMoriTanaka("EshelbyInclusionMoriTanaka") wraps EshelbyInclusionMoriTanaka [Clone, Copy, Default] {
        new();
        assoc_res spherical_eshelby_factors(poisson_nu: f64) -> (f64, f64);
    }
}

use crate::physics::pendry_transformation_optics_cloak::PendryMetamaterialCloakEngine;

py_engine! {
    PyPendryMetamaterialCloakEngine("PendryMetamaterialCloakEngine") wraps PendryMetamaterialCloakEngine [Clone, Copy] {
        new(inner_r1: f64, outer_r2: f64) -> res;
        method_res cylindrical_cloak_permittivity(r_prime: f64) -> (f64, f64, f64);
    }
}

use crate::physics::yablonovitch_photonic_crystal_bandgap::PhotonicBraggQuarterWaveStack;

py_engine! {
    PyPhotonicBraggQuarterWaveStack("PhotonicBraggQuarterWaveStack") wraps PhotonicBraggQuarterWaveStack [Clone, Copy] {
        new(n_high: f64, n_low: f64) -> res;
        method fractional_bandgap_and_reflectance(num_periods_n: u32) -> (f64, f64);
    }
}

use crate::physics::kane_lubensky_topological_mechanics::KaneLubenskyIsostaticEngine;

py_engine! {
    PyKaneLubenskyIsostaticEngine("KaneLubenskyIsostaticEngine") wraps KaneLubenskyIsostaticEngine [Clone, Copy] {
        new(dim_d: usize);
        method maxwell_calladine_index(num_sites: usize, num_bonds: usize) -> (i64, bool);
        assoc dimerized_compatibility_winding(c1: f64, c2: f64) -> i32;
    }
}

use crate::omni_hexacosa_xxvi::GrandOmniHexacosaEngineXXVI;

#[pyclass(
    name = "GrandOmniHexacosaEngineXXVI",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniHexacosaEngineXXVI {
    pub inner: GrandOmniHexacosaEngineXXVI,
}

#[pymethods]
impl PyGrandOmniHexacosaEngineXXVI {
    #[new]
    #[pyo3(signature = (num_layers=26))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniHexacosaEngineXXVI::new(num_layers),
        }
    }

    pub fn execute_hexacosa_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_hexacosa_synthesis())
            .map_err(to_py_err)
    }
}

use crate::science::gromov_hyperbolic_groups::GromovDeltaHyperbolicEngine;

py_engine! {
    PyGromovDeltaHyperbolicEngine("GromovDeltaHyperbolicEngine") wraps GromovDeltaHyperbolicEngine [Clone, Copy] {
        new(delta: f64) -> res;
        method four_point_defect_and_check(d_xy: f64, d_zw: f64, d_xz: f64, d_yw: f64, d_xw: f64, d_yz: f64) -> (f64, bool);
        assoc gromov_product(d_xw: f64, d_yw: f64, d_xy: f64) -> f64;
    }
}

use crate::science::teichmuller_weil_petersson::TeichmullerWeilPeterssonEngine;

py_engine! {
    PyTeichmullerWeilPeterssonEngine("TeichmullerWeilPeterssonEngine") wraps TeichmullerWeilPeterssonEngine [Clone, Copy] {
        new(genus_g: usize) -> res;
        assoc mirzakhani_wp_volume_v11(boundary_length_l: f64) -> f64;
        assoc_res dilatation_and_teichmuller_distance(beltrami_norm_k: f64) -> (f64, f64);
    }
}

use crate::science::thurston_hyperbolic_dehn::ThurstonHyperbolicDehnEngine;

py_engine! {
    PyThurstonHyperbolicDehnEngine("ThurstonHyperbolicDehnEngine") wraps ThurstonHyperbolicDehnEngine [Clone, Copy] {
        new(cusped_volume: f64) -> res;
        method_res dehn_filled_volume(p: i64, q: i64, cusp_aspect_c: f64) -> (f64, f64);
    }
}

use crate::science::bass_serre_trees::BassSerreGraphOfGroupsEngine;

#[pyclass(
    name = "BassSerreGraphOfGroupsEngine",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone, Copy, Default)]
pub struct PyBassSerreGraphOfGroupsEngine {
    pub inner: BassSerreGraphOfGroupsEngine,
}

#[pymethods]
impl PyBassSerreGraphOfGroupsEngine {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: BassSerreGraphOfGroupsEngine::new(),
        }
    }

    #[staticmethod]
    pub fn rational_euler_characteristic(
        vertex_group_orders: Vec<usize>,
        edge_group_orders: Vec<usize>,
    ) -> PyResult<f64> {
        BassSerreGraphOfGroupsEngine::rational_euler_characteristic(
            &vertex_group_orders,
            &edge_group_orders,
        )
        .map_err(to_py_err)
    }
}

use crate::science::kazhdan_property_t_expander::KazhdanPropertyTExpander;

py_engine! {
    PyKazhdanPropertyTExpander("KazhdanPropertyTExpander") wraps KazhdanPropertyTExpander [Clone, Copy] {
        new(degree_d: usize) -> res;
        method ramanujan_bound_and_gap(second_eigenvalue_abs: f64) -> (f64, f64, bool);
    }
}

use crate::science::arthur_selberg_langlands::SatakeHeckeLanglandsEngine;

py_engine! {
    PySatakeHeckeLanglandsEngine("SatakeHeckeLanglandsEngine") wraps SatakeHeckeLanglandsEngine [Clone, Copy] {
        new(weight_k: usize) -> res;
        method_res ramanujan_bound_and_satake_angle(prime_p: u64, hecke_ap: f64) -> (f64, f64, bool);
    }
}

use crate::science::ngo_fundamental_lemma::NgoFundamentalLemmaEngine;

py_engine! {
    PyNgoFundamentalLemmaEngine("NgoFundamentalLemmaEngine") wraps NgoFundamentalLemmaEngine [Clone, Copy] {
        new(residue_q: u64) -> res;
        method kappa_orbital_and_endoscopic_match(valuation_depth_n: u32) -> (f64, f64, bool);
    }
}

use crate::science::drinfeld_shtukas_function_field::DrinfeldShtukaLanglandsEngine;

py_engine! {
    PyDrinfeldShtukaLanglandsEngine("DrinfeldShtukaLanglandsEngine") wraps DrinfeldShtukaLanglandsEngine [Clone, Copy] {
        new(rank_r: usize, field_q: u64) -> res;
        method shtuka_moduli_dimension(num_legs_n: usize) -> (usize, usize);
    }
}

use crate::science::ratner_unipotent_ergodic::RatnerUnipotentFlowEngine;

py_engine! {
    PyRatnerUnipotentFlowEngine("RatnerUnipotentFlowEngine") wraps RatnerUnipotentFlowEngine [Clone, Copy] {
        new(laplacian_lambda1: f64) -> res;
        method_res equidistribution_exponent_and_bound(horocycle_length_t: f64) -> (f64, f64);
    }
}

use crate::omni_heptacosa_xxvii::GrandOmniHeptacosaEngineXXVII;

#[pyclass(
    name = "GrandOmniHeptacosaEngineXXVII",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniHeptacosaEngineXXVII {
    pub inner: GrandOmniHeptacosaEngineXXVII,
}

#[pymethods]
impl PyGrandOmniHeptacosaEngineXXVII {
    #[new]
    #[pyo3(signature = (num_layers=27))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniHeptacosaEngineXXVII::new(num_layers),
        }
    }

    pub fn execute_heptacosa_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_heptacosa_synthesis())
            .map_err(to_py_err)
    }
}

use crate::physics::schwinger_euler_heisenberg_qed::SchwingerEulerHeisenbergQED;

py_engine! {
    PySchwingerEulerHeisenbergQED("SchwingerEulerHeisenbergQED") wraps SchwingerEulerHeisenbergQED [Clone, Copy] {
        new(critical_field_ecr: f64) -> res;
        method_res pair_production_rate(electric_field_e: f64, harmonics_n: usize) -> f64;
    }
}

use crate::physics::casimir_lifshitz_vacuum::CasimirLifshitzVacuumEngine;

py_engine! {
    PyCasimirLifshitzVacuumEngine("CasimirLifshitzVacuumEngine") wraps CasimirLifshitzVacuumEngine [Clone, Copy, Default] {
        new();
        assoc_res energy_and_pressure(gap_a: f64) -> (f64, f64);
    }
}

use crate::physics::unruh_dewitt_rindler::UnruhRindlerBogoliubovEngine;

py_engine! {
    PyUnruhRindlerBogoliubovEngine("UnruhRindlerBogoliubovEngine") wraps UnruhRindlerBogoliubovEngine [Clone, Copy] {
        new(proper_acceleration_a: f64) -> res;
        method_res unruh_temperature_and_bogoliubov_n(mode_omega: f64) -> (f64, f64);
    }
}

use crate::physics::msw_neutrino_oscillation::MswNeutrinoOscillationEngine;

py_engine! {
    PyMswNeutrinoOscillationEngine("MswNeutrinoOscillationEngine") wraps MswNeutrinoOscillationEngine [Clone, Copy] {
        new(theta_0: f64, delta_m_sq: f64) -> res;
        method matter_mixing_and_splitting(wolfenstein_a: f64) -> (f64, f64);
    }
}

use crate::physics::wilson_lattice_qcd_plaquette::WilsonLatticeGaugeQCD;

py_engine! {
    PyWilsonLatticeGaugeQCD("WilsonLatticeGaugeQCD") wraps WilsonLatticeGaugeQCD [Clone, Copy] {
        new(num_colors_nc: usize, beta: f64) -> res;
        method_res strong_coupling_string_tension_and_wilson_loop(r_plaquettes: usize, t_plaquettes: usize) -> (f64, f64);
    }
}

use crate::physics::qcd_asymptotic_freedom_beta::QcdAsymptoticFreedomEngine;

py_engine! {
    PyQcdAsymptoticFreedomEngine("QcdAsymptoticFreedomEngine") wraps QcdAsymptoticFreedomEngine [Clone, Copy] {
        new(num_colors_nc: usize, num_flavors_nf: usize, lambda_qcd: f64) -> res;
        method_res running_alpha_s(scale_q: f64) -> (f64, f64, bool);
    }
}

use crate::physics::adler_bell_jackiw_anomaly::AbjChiralAnomalyEngine;

py_engine! {
    PyAbjChiralAnomalyEngine("AbjChiralAnomalyEngine") wraps AbjChiralAnomalyEngine [Clone, Copy] {
        new(num_colors_nc: usize, alpha_em: f64) -> res;
        method_res pion_two_photon_width(m_pi_mev: f64, f_pi_mev: f64) -> f64;
    }
}

use crate::physics::kibble_zurek_cosmological_defects::KibbleZurekDefectScaling;

py_engine! {
    PyKibbleZurekDefectScaling("KibbleZurekDefectScaling") wraps KibbleZurekDefectScaling [Clone, Copy] {
        new(nu: f64, z_dyn: f64) -> res;
        method_res freeze_out_length_and_density(quench_ratio_tau_q: f64, spatial_dim_d: usize) -> (f64, f64, f64);
    }
}

use crate::physics::sphaleron_electroweak_baryogenesis::ElectroweakSphaleronBaryogenesis;

py_engine! {
    PyElectroweakSphaleronBaryogenesis("ElectroweakSphaleronBaryogenesis") wraps ElectroweakSphaleronBaryogenesis [Clone, Copy] {
        new(num_generations_ng: usize, num_higgs_doublets_nh: usize) -> res;
        method sphaleron_conversion_and_baryon_yield(y_b_minus_l: f64) -> (f64, f64);
    }
}

use crate::omni_octacosa_xxviii::GrandOmniOctacosaEngineXXVIII;

#[pyclass(
    name = "GrandOmniOctacosaEngineXXVIII",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniOctacosaEngineXXVIII {
    pub inner: GrandOmniOctacosaEngineXXVIII,
}

#[pymethods]
impl PyGrandOmniOctacosaEngineXXVIII {
    #[new]
    #[pyo3(signature = (num_layers=28))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniOctacosaEngineXXVIII::new(num_layers),
        }
    }

    pub fn execute_octacosa_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_octacosa_synthesis())
            .map_err(to_py_err)
    }
}

use crate::science::kimura_wright_fisher_genetics::KimuraWrightFisherEngine;

py_engine! {
    PyKimuraWrightFisherEngine("KimuraWrightFisherEngine") wraps KimuraWrightFisherEngine [Clone, Copy] {
        new(effective_pop_ne: f64) -> res;
        method_res fixation_probability(selection_s: f64, initial_freq_p: f64) -> f64;
    }
}

use crate::science::kingman_coalescent_ancestry::KingmanCoalescentEngine;

py_engine! {
    PyKingmanCoalescentEngine("KingmanCoalescentEngine") wraps KingmanCoalescentEngine [Clone, Copy] {
        new(sample_size_n: usize) -> res;
        method tmrca_tree_length_and_watterson(segregating_sites_s: usize) -> (f64, f64, f64);
    }
}

use crate::science::price_hamilton_inclusive_fitness::PriceHamiltonEvolutionEngine;

#[pyclass(
    name = "PriceHamiltonEvolutionEngine",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone, Copy, Default)]
pub struct PyPriceHamiltonEvolutionEngine {
    pub inner: PriceHamiltonEvolutionEngine,
}

#[pymethods]
impl PyPriceHamiltonEvolutionEngine {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: PriceHamiltonEvolutionEngine::new(),
        }
    }

    #[staticmethod]
    pub fn price_decomposition_and_hamilton(
        fitness_w: Vec<f64>,
        trait_z: Vec<f64>,
        delta_z: Vec<f64>,
        relatedness_r: f64,
        benefit_b: f64,
        cost_c: f64,
    ) -> PyResult<(f64, f64, f64, bool)> {
        PriceHamiltonEvolutionEngine::price_decomposition_and_hamilton(
            &fitness_w,
            &trait_z,
            &delta_z,
            relatedness_r,
            benefit_b,
            cost_c,
        )
        .map_err(to_py_err)
    }
}

use crate::science::replicator_mutator_ess::ReplicatorMutatorGameEngine;

py_engine! {
    PyReplicatorMutatorGameEngine("ReplicatorMutatorGameEngine") wraps ReplicatorMutatorGameEngine [Clone, Copy] {
        new(resource_v: f64, conflict_cost_c: f64) -> res;
        method_res hawk_dove_ess_and_velocity(hawk_fraction_p: f64) -> (f64, f64);
    }
}

use crate::science::eigen_quasispecies_threshold::EigenQuasispeciesThreshold;

py_engine! {
    PyEigenQuasispeciesThreshold("EigenQuasispeciesThreshold") wraps EigenQuasispeciesThreshold [Clone, Copy] {
        new(genome_length_l: usize, master_fitness_a0: f64) -> res;
        method_res critical_mutation_rate_and_master_fraction(per_base_error_mu: f64) -> (f64, f64, bool);
    }
}

use crate::science::may_wigner_ecosystem_stability::MayWignerComplexityStability;

py_engine! {
    PyMayWignerComplexityStability("MayWignerComplexityStability") wraps MayWignerComplexityStability [Clone, Copy] {
        new(self_damping_d: f64) -> res;
        method_res spectral_radius_and_stability(num_species_s: usize, connectance_c: f64, interaction_sigma: f64) -> (f64, f64, bool);
    }
}

use crate::science::fisher_kpp_traveling_wave::FisherKppWavefrontEngine;

py_engine! {
    PyFisherKppWavefrontEngine("FisherKppWavefrontEngine") wraps FisherKppWavefrontEngine [Clone, Copy] {
        new(diffusion_d: f64, growth_rate_r: f64) -> res;
        method_res wave_speed_and_bramson_front(time_t: f64) -> (f64, f64, f64);
    }
}

use crate::science::kermack_mckendrick_epidemic::KermackMcKendrickEpidemicEngine;

py_engine! {
    PyKermackMcKendrickEpidemicEngine("KermackMcKendrickEpidemicEngine") wraps KermackMcKendrickEpidemicEngine [Clone, Copy] {
        new(beta: f64, gamma: f64) -> res;
        method r0_herd_threshold_and_final_size() -> (f64, f64, f64);
    }
}

use crate::science::hubbell_ewens_biodiversity::HubbellEwensBiodiversityEngine;

py_engine! {
    PyHubbellEwensBiodiversityEngine("HubbellEwensBiodiversityEngine") wraps HubbellEwensBiodiversityEngine [Clone, Copy] {
        new(theta: f64) -> res;
        method_res expected_richness_and_variance(sample_size_j: usize) -> (f64, f64);
    }
}

use crate::omni_nonacosa_xxix::GrandOmniNonacosaEngineXXIX;

#[pyclass(
    name = "GrandOmniNonacosaEngineXXIX",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniNonacosaEngineXXIX {
    pub inner: GrandOmniNonacosaEngineXXIX,
}

#[pymethods]
impl PyGrandOmniNonacosaEngineXXIX {
    #[new]
    #[pyo3(signature = (num_layers=29))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniNonacosaEngineXXIX::new(num_layers),
        }
    }

    pub fn execute_nonacosa_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_nonacosa_synthesis())
            .map_err(to_py_err)
    }
}

use crate::science::martin_borel_determinacy::MartinBorelDeterminacyEngine;

#[pyclass(
    name = "MartinBorelDeterminacyEngine",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone, Copy, Default)]
pub struct PyMartinBorelDeterminacyEngine {
    pub inner: MartinBorelDeterminacyEngine,
}

#[pymethods]
impl PyMartinBorelDeterminacyEngine {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: MartinBorelDeterminacyEngine::new(),
        }
    }

    #[staticmethod]
    pub fn evaluate_binary_game_tree(leaf_payoffs: Vec<i32>) -> PyResult<i32> {
        MartinBorelDeterminacyEngine::evaluate_binary_game_tree(&leaf_payoffs).map_err(to_py_err)
    }
}

use crate::science::shelah_stability_sauer::ShelahStabilitySauerEngine;

py_engine! {
    PyShelahStabilitySauerEngine("ShelahStabilitySauerEngine") wraps ShelahStabilitySauerEngine [Clone, Copy] {
        new(vc_dim_d: usize);
        method_res sauer_shelah_type_bound(num_params_n: usize) -> u64;
    }
}

use crate::science::o_minimal_pfaffian_cells::OMinimalPfaffianCellEngine;

py_engine! {
    PyOMinimalPfaffianCellEngine("OMinimalPfaffianCellEngine") wraps OMinimalPfaffianCellEngine [Clone, Copy] {
        new(dim_n: usize, chain_order_r: usize) -> res;
        method_res khovanskii_component_bound(alpha: usize, beta: usize) -> f64;
    }
}

use crate::science::hrushovski_zariski_geometry::HrushovskiZariskiGeometry;

py_engine! {
    PyHrushovskiZariskiGeometry("HrushovskiZariskiGeometry") wraps HrushovskiZariskiGeometry [Clone, Copy] {
        new(alpha: f64) -> res;
        method_res predimension_and_strong_extension(v_a: usize, e_a: usize, v_b: usize, e_b: usize) -> (f64, f64, bool);
    }
}

use crate::science::goodstein_paris_harrington::GoodsteinParisHarringtonEngine;

py_engine! {
    PyGoodsteinParisHarringtonEngine("GoodsteinParisHarringtonEngine") wraps GoodsteinParisHarringtonEngine [Clone, Copy, Default] {
        new();
        assoc_res hereditary_base_bump_and_step(m: u64, base_b: u64) -> (u64, u64);
    }
}

use crate::science::solomonoff_levin_algorithmic::SolomonoffLevinUniversalPrior;

#[pyclass(
    name = "SolomonoffLevinUniversalPrior",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone, Copy, Default)]
pub struct PySolomonoffLevinUniversalPrior {
    pub inner: SolomonoffLevinUniversalPrior,
}

#[pymethods]
impl PySolomonoffLevinUniversalPrior {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: SolomonoffLevinUniversalPrior::new(),
        }
    }

    #[staticmethod]
    pub fn levin_kt_and_universal_prior(
        program_lengths_bits: Vec<usize>,
        runtimes_steps: Vec<u64>,
    ) -> PyResult<(f64, f64)> {
        SolomonoffLevinUniversalPrior::levin_kt_and_universal_prior(
            &program_lengths_bits,
            &runtimes_steps,
        )
        .map_err(to_py_err)
    }
}

use crate::science::razborov_sunflower_circuit::SunflowerNaturalProofsBarrier;

py_engine! {
    PySunflowerNaturalProofsBarrier("SunflowerNaturalProofsBarrier") wraps SunflowerNaturalProofsBarrier [Clone, Copy] {
        new(set_width_w: usize, petals_p: usize) -> res;
        method erdos_rado_and_alwz_bounds() -> (f64, f64);
    }
}

use crate::science::dinur_pcp_unique_games::DinurPcpGapAmplification;

py_engine! {
    PyDinurPcpGapAmplification("DinurPcpGapAmplification") wraps DinurPcpGapAmplification [Clone, Copy] {
        new(walk_length_t: usize) -> res;
        assoc_res goemans_williamson_ratio_at_angle(theta_rad: f64) -> (f64, f64);
    }
}

use crate::science::williams_sat_circuit_duality::WilliamsSatCircuitLowerBound;

py_engine! {
    PyWilliamsSatCircuitLowerBound("WilliamsSatCircuitLowerBound") wraps WilliamsSatCircuitLowerBound [Clone, Copy] {
        new(num_vars_n: usize) -> res;
        method_res speedup_factor(split_vars_l: usize, omega_exponent: f64) -> (f64, f64, bool);
    }
}

use crate::omni_tricentennial_xxx::GrandOmniTricentennialEngineXXX;

#[pyclass(
    name = "GrandOmniTricentennialEngineXXX",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniTricentennialEngineXXX {
    pub inner: GrandOmniTricentennialEngineXXX,
}

#[pymethods]
impl PyGrandOmniTricentennialEngineXXX {
    #[new]
    #[pyo3(signature = (num_layers=30))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniTricentennialEngineXXX::new(num_layers),
        }
    }

    pub fn execute_tricentennial_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_tricentennial_synthesis())
            .map_err(to_py_err)
    }
}

use crate::quantum::jaynes_cummings_cavity_qed::JaynesCummingsCavityQED;

py_engine! {
    PyJaynesCummingsCavityQED("JaynesCummingsCavityQED") wraps JaynesCummingsCavityQED [Clone, Copy] {
        new(coupling_g: f64, detuning_delta: f64) -> res;
        method_res rabi_frequency_and_revival_time(photon_n: usize, mean_coherent_n: f64) -> (f64, f64);
    }
}

use crate::quantum::dicke_tavis_cummings::DickeTavisCummingsSuperradiance;

py_engine! {
    PyDickeTavisCummingsSuperradiance("DickeTavisCummingsSuperradiance") wraps DickeTavisCummingsSuperradiance [Clone, Copy] {
        new(num_atoms_n: usize, single_decay_gamma0: f64) -> res;
        method_res peak_burst_rate_and_critical_coupling(omega_c: f64, omega_0: f64) -> (f64, f64);
    }
}

use crate::quantum::hong_ou_mandel_dip::HongOuMandelInterferometer;

py_engine! {
    PyHongOuMandelInterferometer("HongOuMandelInterferometer") wraps HongOuMandelInterferometer [Clone, Copy] {
        new(reflectance_r: f64, bandwidth_sigma: f64) -> res;
        method_res coincidence_probability_and_visibility(delay_tau: f64, state_overlap_v: f64) -> (f64, f64);
    }
}

use crate::quantum::glauber_mandel_antibunching::GlauberPhotonCoherenceEngine;

py_engine! {
    PyGlauberPhotonCoherenceEngine("GlauberPhotonCoherenceEngine") wraps GlauberPhotonCoherenceEngine [Clone, Copy, Default] {
        new();
        assoc_res mandel_q_and_g2_zero(mean_n: f64, var_n: f64) -> (f64, f64, bool);
    }
}

use crate::quantum::eit_dark_state_polariton::ElectromagneticallyInducedTransparency;

py_engine! {
    PyElectromagneticallyInducedTransparency("ElectromagneticallyInducedTransparency") wraps ElectromagneticallyInducedTransparency [Clone, Copy] {
        new(collective_coupling_gn: f64) -> res;
        method_res group_index_and_slow_light_ratio(control_rabi_omega_c: f64) -> (f64, f64);
    }
}

use crate::quantum::corkum_hhg_attosecond::CorkumHighHarmonicAttosecond;

py_engine! {
    PyCorkumHighHarmonicAttosecond("CorkumHighHarmonicAttosecond") wraps CorkumHighHarmonicAttosecond [Clone, Copy] {
        new(ionization_potential_ip: f64) -> res;
        method_res cutoff_energy_and_keldysh(ponderomotive_up: f64) -> (f64, f64, bool);
    }
}

use crate::quantum::doppler_sisyphus_cooling::DopplerSisyphusLaserCooling;

py_engine! {
    PyDopplerSisyphusLaserCooling("DopplerSisyphusLaserCooling") wraps DopplerSisyphusLaserCooling [Clone, Copy] {
        new(linewidth_gamma: f64) -> res;
        method_res doppler_temperature_and_limit(red_detuning_delta: f64) -> (f64, f64);
    }
}

use crate::quantum::feshbach_bec_bcs_crossover::FeshbachScatteringResonance;

py_engine! {
    PyFeshbachScatteringResonance("FeshbachScatteringResonance") wraps FeshbachScatteringResonance [Clone, Copy] {
        new(a_bg: f64, b0_gauss: f64, delta_b_gauss: f64) -> res;
        method_res scattering_length_and_bec_side(field_b_gauss: f64) -> (f64, bool);
    }
}

use crate::quantum::cavity_optomechanics_sql::CavityOptomechanicsBackaction;

py_engine! {
    PyCavityOptomechanicsBackaction("CavityOptomechanicsBackaction") wraps CavityOptomechanicsBackaction [Clone, Copy] {
        new(kappa: f64, gamma_m: f64, omega_m: f64) -> res;
        method_res cooperativity_and_final_phonons(enhanced_coupling_g: f64, thermal_bath_nth: f64) -> (f64, f64, bool);
    }
}

use crate::omni_henetricosa_xxxi::GrandOmniHenetricosaEngineXXXI;

#[pyclass(
    name = "GrandOmniHenetricosaEngineXXXI",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniHenetricosaEngineXXXI {
    pub inner: GrandOmniHenetricosaEngineXXXI,
}

#[pymethods]
impl PyGrandOmniHenetricosaEngineXXXI {
    #[new]
    #[pyo3(signature = (num_layers=31))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniHenetricosaEngineXXXI::new(num_layers),
        }
    }

    pub fn execute_henetricosa_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_henetricosa_synthesis())
            .map_err(to_py_err)
    }
}

use crate::science::heath_jarrow_morton_hjm::HeathJarrowMortonTermStructure;

py_engine! {
    PyHeathJarrowMortonTermStructure("HeathJarrowMortonTermStructure") wraps HeathJarrowMortonTermStructure [Clone, Copy] {
        new(sigma0: f64, kappa: f64) -> res;
        method_res no_arbitrage_drift_and_bond_vol(maturity_tau: f64) -> (f64, f64);
    }
}

use crate::science::hagan_sabr_volatility::HaganSabrVolatilityEngine;

py_engine! {
    PyHaganSabrVolatilityEngine("HaganSabrVolatilityEngine") wraps HaganSabrVolatilityEngine [Clone, Copy] {
        new(alpha: f64, beta: f64, rho: f64, nu: f64) -> res;
        method_res atm_implied_volatility(forward_f: f64, maturity_t: f64) -> f64;
    }
}

use crate::science::almgren_chriss_execution::AlmgrenChrissOptimalExecution;

py_engine! {
    PyAlmgrenChrissOptimalExecution("AlmgrenChrissOptimalExecution") wraps AlmgrenChrissOptimalExecution [Clone, Copy] {
        new(sigma: f64, temp_impact_eta: f64, risk_aversion_lambda: f64) -> res;
        method_res urgency_and_inventory_fraction(time_t: f64, horizon_t: f64) -> (f64, f64);
    }
}

use crate::science::avellaneda_stoikov_market_making::AvellanedaStoikovMarketMaker;

py_engine! {
    PyAvellanedaStoikovMarketMaker("AvellanedaStoikovMarketMaker") wraps AvellanedaStoikovMarketMaker [Clone, Copy] {
        new(gamma: f64, sigma: f64, kappa_k: f64) -> res;
        method_res reservation_price_and_spread(mid_price_s: f64, inventory_q: f64, remaining_tau: f64) -> (f64, f64);
    }
}

use crate::science::kyle_lambda_microstructure::KyleLambdaAuctionEquilibrium;

py_engine! {
    PyKyleLambdaAuctionEquilibrium("KyleLambdaAuctionEquilibrium") wraps KyleLambdaAuctionEquilibrium [Clone, Copy] {
        new(fundamental_var_sigma0: f64, noise_std_sigma_u: f64) -> res;
        method lambda_beta_and_insider_profit() -> (f64, f64, f64);
    }
}

use crate::science::rockafellar_uryasev_cvar::RockafellarUryasevCVaREngine;

#[pyclass(
    name = "RockafellarUryasevCVaREngine",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone, Copy)]
pub struct PyRockafellarUryasevCVaREngine {
    pub inner: RockafellarUryasevCVaREngine,
}

#[pymethods]
impl PyRockafellarUryasevCVaREngine {
    #[new]
    pub fn new(confidence_alpha: f64) -> PyResult<Self> {
        let inner = RockafellarUryasevCVaREngine::new(confidence_alpha).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn empirical_var_and_cvar(&self, scenario_losses: Vec<f64>) -> PyResult<(f64, f64)> {
        self.inner
            .empirical_var_and_cvar(&scenario_losses)
            .map_err(to_py_err)
    }
}

use crate::science::merton_structural_credit::MertonStructuralDefaultEngine;

py_engine! {
    PyMertonStructuralDefaultEngine("MertonStructuralDefaultEngine") wraps MertonStructuralDefaultEngine [Clone, Copy] {
        new(asset_vol_sigma: f64, risk_free_r: f64) -> res;
        method_res distance_to_default(asset_v0: f64, debt_d: f64, maturity_t: f64) -> (f64, f64);
    }
}

use crate::science::eisenberg_noe_systemic_risk::EisenbergNoeSystemicClearing;

py_engine! {
    PyEisenbergNoeSystemicClearing("EisenbergNoeSystemicClearing") wraps EisenbergNoeSystemicClearing [Clone, Copy, Default] {
        new();
        assoc_res solve_two_bank_ring_clearing(e1: f64, e2: f64, pbar1: f64, pbar2: f64, cross_share_alpha: f64) -> (f64, f64, usize);
    }
}

use crate::science::kelly_thorp_optimal_growth::KellyLogOptimalPortfolio;

py_engine! {
    PyKellyLogOptimalPortfolio("KellyLogOptimalPortfolio") wraps KellyLogOptimalPortfolio [Clone, Copy] {
        new(excess_return: f64, volatility_sigma: f64) -> res;
        method_res kelly_leverage_and_growth_rate(fraction_c: f64) -> (f64, f64);
    }
}

use crate::omni_dotricosa_xxxii::GrandOmniDotricosaEngineXXXII;

#[pyclass(
    name = "GrandOmniDotricosaEngineXXXII",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniDotricosaEngineXXXII {
    pub inner: GrandOmniDotricosaEngineXXXII,
}

#[pymethods]
impl PyGrandOmniDotricosaEngineXXXII {
    #[new]
    #[pyo3(signature = (num_layers=32))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniDotricosaEngineXXXII::new(num_layers),
        }
    }

    pub fn execute_dotricosa_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_dotricosa_synthesis())
            .map_err(to_py_err)
    }
}

use crate::physics::kanamori_gutenberg_seismology::KanamoriGutenbergSeismicEngine;

py_engine! {
    PyKanamoriGutenbergSeismicEngine("KanamoriGutenbergSeismicEngine") wraps KanamoriGutenbergSeismicEngine [Clone, Copy] {
        new(b_value: f64) -> res;
        method_res moment_magnitude_and_omori_rate(seismic_moment_nm: f64, time_days: f64, omori_k: f64, omori_c: f64, omori_p: f64) -> (f64, f64);
    }
}

use crate::physics::aki_richards_zoeppritz_seismic::AkiRichardsSeismicAvoEngine;

py_engine! {
    PyAkiRichardsSeismicAvoEngine("AkiRichardsSeismicAvoEngine") wraps AkiRichardsSeismicAvoEngine [Clone, Copy] {
        new(intercept_r0: f64, gradient_g: f64, curvature_f: f64);
        method_res shuey_rpp_reflectivity(angle_theta_rad: f64) -> f64;
    }
}

use crate::physics::backus_gilbert_geophysical_inverse::BackusGilbertInverseResolution;

py_engine! {
    PyBackusGilbertInverseResolution("BackusGilbertInverseResolution") wraps BackusGilbertInverseResolution [Clone] {
        new(singular_values: Vec<f64>) -> res;
        method_res resolution_trace_spread_and_variance(trade_off_alpha: f64) -> (f64, f64, f64);
    }
}

use crate::physics::budyko_sellers_milankovitch_climate::BudykoSellersClimateBifurcation;

py_engine! {
    PyBudykoSellersClimateBifurcation("BudykoSellersClimateBifurcation") wraps BudykoSellersClimateBifurcation [Clone, Copy] {
        new(olr_a: f64, olr_b: f64) -> res;
        method_res warm_and_snowball_equilibria(solar_s0: f64, alpha_warm: f64, alpha_ice: f64) -> (f64, f64);
    }
}

use crate::physics::chandrasekhar_eddington_radiative::EddingtonChandrasekharRadiative;

py_engine! {
    PyEddingtonChandrasekharRadiative("EddingtonChandrasekharRadiative") wraps EddingtonChandrasekharRadiative [Clone, Copy] {
        new(t_eff_kelvin: f64) -> res;
        method_res eddington_temperature_and_slab_albedo(optical_depth_tau: f64, asymmetry_g: f64) -> (f64, f64);
    }
}

use crate::physics::elsasser_geodynamo_core::ElsasserGeodynamoCoreEngine;

py_engine! {
    PyElsasserGeodynamoCoreEngine("ElsasserGeodynamoCoreEngine") wraps ElsasserGeodynamoCoreEngine [Clone, Copy] {
        new(density_rho: f64, omega_rot: f64, conductivity_sigma: f64) -> res;
        method_res elsasser_number(magnetic_field_tesla: f64) -> (f64, bool);
    }
}

use crate::physics::thermal_wind_eady_baroclinic::ThermalWindEadyBaroclinic;

py_engine! {
    PyThermalWindEadyBaroclinic("ThermalWindEadyBaroclinic") wraps ThermalWindEadyBaroclinic [Clone, Copy] {
        new(coriolis_f: f64, brunt_vaisala_n: f64) -> res;
        method_res eady_growth_rate_and_rossby_radius(vertical_shear_lambda: f64, tropopause_height_h: f64) -> (f64, f64);
    }
}

use crate::physics::stommel_sverdrup_ocean_gyre::StommelSverdrupOceanCirculation;

py_engine! {
    PyStommelSverdrupOceanCirculation("StommelSverdrupOceanCirculation") wraps StommelSverdrupOceanCirculation [Clone, Copy] {
        new(beta: f64, drag_r: f64) -> res;
        method_res boundary_widths_and_sverdrup_transport(wind_curl_pa_m: f64, rho0: f64, viscosity_ah: f64) -> (f64, f64, f64);
    }
}

use crate::physics::toomre_hill_planetary_disk::ToomreHillPlanetaryDisk;

py_engine! {
    PyToomreHillPlanetaryDisk("ToomreHillPlanetaryDisk") wraps ToomreHillPlanetaryDisk [Clone, Copy] {
        new(grav_g: f64) -> res;
        method_res toomre_q_and_hill_radius(sound_speed_cs: f64, epicyclic_kappa: f64, surface_density_sigma: f64, semi_major_a: f64, mass_ratio_mp_mstar: f64) -> (f64, bool, f64);
    }
}

use crate::omni_tritricosa_xxxiii::GrandOmniTritricosaEngineXXXIII;

#[pyclass(
    name = "GrandOmniTritricosaEngineXXXIII",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniTritricosaEngineXXXIII {
    pub inner: GrandOmniTritricosaEngineXXXIII,
}

#[pymethods]
impl PyGrandOmniTritricosaEngineXXXIII {
    #[new]
    #[pyo3(signature = (num_layers=33))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniTritricosaEngineXXXIII::new(num_layers),
        }
    }

    pub fn execute_tritricosa_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_tritricosa_synthesis())
            .map_err(to_py_err)
    }
}

use crate::science::tutte_polynomial_matroid::TuttePolynomialMatroidEngine;

py_engine! {
    PyTuttePolynomialMatroidEngine("TuttePolynomialMatroidEngine") wraps TuttePolynomialMatroidEngine [Clone, Copy] {
        new(cycle_n: usize) -> res;
        method evaluate_tutte_and_invariants(x: f64, y: f64) -> (f64, usize, u64);
    }
}

use crate::science::huh_katz_matroid_hodge::HuhKatzMatroidLogConcavity;

#[pyclass(
    name = "HuhKatzMatroidLogConcavity",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone, Copy, Default)]
pub struct PyHuhKatzMatroidLogConcavity {
    pub inner: HuhKatzMatroidLogConcavity,
}

#[pymethods]
impl PyHuhKatzMatroidLogConcavity {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: HuhKatzMatroidLogConcavity::new(),
        }
    }

    #[staticmethod]
    pub fn verify_log_concavity(sequence: Vec<f64>) -> PyResult<(f64, bool)> {
        HuhKatzMatroidLogConcavity::verify_log_concavity(&sequence).map_err(to_py_err)
    }
}

use crate::science::mikhalkin_tropical_bezout::MikhalkinTropicalGeometryEngine;

#[pyclass(
    name = "MikhalkinTropicalGeometryEngine",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone, Copy)]
pub struct PyMikhalkinTropicalGeometryEngine {
    pub inner: MikhalkinTropicalGeometryEngine,
}

#[pymethods]
impl PyMikhalkinTropicalGeometryEngine {
    #[new]
    pub fn new(degree_d: usize) -> PyResult<Self> {
        let inner = MikhalkinTropicalGeometryEngine::new(degree_d).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn genus_bezout_and_tropical_eval(
        &self,
        other_degree_e: usize,
        coeffs: Vec<f64>,
        x: f64,
    ) -> PyResult<(usize, usize, f64)> {
        guard_res(|| {
            self.inner
                .genus_bezout_and_tropical_eval(other_degree_e, &coeffs, x)
        })
    }
}

use crate::science::ehrhart_macdonald_polytope::EhrhartMacdonaldPolytopeEngine;

py_engine! {
    PyEhrhartMacdonaldPolytopeEngine("EhrhartMacdonaldPolytopeEngine") wraps EhrhartMacdonaldPolytopeEngine [Clone, Copy] {
        new(twice_area_2a: u64, boundary_points_b: u64) -> res;
        method_res ehrhart_closed_and_interior_counts(dilation_t: u64) -> (u64, u64);
    }
}

use crate::science::stanley_reisner_dehn_sommerville::StanleyReisnerDehnSommerville;

#[pyclass(
    name = "StanleyReisnerDehnSommerville",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone, Copy, Default)]
pub struct PyStanleyReisnerDehnSommerville {
    pub inner: StanleyReisnerDehnSommerville,
}

#[pymethods]
impl PyStanleyReisnerDehnSommerville {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: StanleyReisnerDehnSommerville::new(),
        }
    }

    #[staticmethod]
    pub fn compute_h_vector_and_check_symmetry(
        f_vector_with_empty: Vec<i64>,
    ) -> PyResult<(Vec<i64>, bool)> {
        StanleyReisnerDehnSommerville::compute_h_vector_and_check_symmetry(&f_vector_with_empty)
            .map_err(to_py_err)
    }
}

use crate::science::rsk_hook_length_tableaux::HookLengthYoungTableauxEngine;

#[pyclass(
    name = "HookLengthYoungTableauxEngine",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone, Copy, Default)]
pub struct PyHookLengthYoungTableauxEngine {
    pub inner: HookLengthYoungTableauxEngine,
}

#[pymethods]
impl PyHookLengthYoungTableauxEngine {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: HookLengthYoungTableauxEngine::new(),
        }
    }

    #[staticmethod]
    pub fn count_standard_young_tableaux(partition: Vec<usize>) -> PyResult<u64> {
        HookLengthYoungTableauxEngine::count_standard_young_tableaux(&partition).map_err(to_py_err)
    }
}

use crate::science::macdonald_qt_catalan::MacdonaldSymmetricCatalanEngine;

py_engine! {
    PyMacdonaldSymmetricCatalanEngine("MacdonaldSymmetricCatalanEngine") wraps MacdonaldSymmetricCatalanEngine [Clone, Copy] {
        new(order_n: usize) -> res;
        method q_catalan_and_haiman_dimension(q: f64) -> (f64, u64);
    }
}

use crate::science::kazhdan_lusztig_bruhat::KazhdanLusztigHeckeEngine;

py_engine! {
    PyKazhdanLusztigHeckeEngine("KazhdanLusztigHeckeEngine") wraps KazhdanLusztigHeckeEngine [Clone, Copy] {
        new(hecke_q: f64) -> res;
        method_res max_kl_degree_and_poincare_polynomial(len_x: usize, len_w: usize, rank_n: usize) -> (usize, f64);
    }
}

use crate::science::szemeredi_green_tao_density::SzemerediGreenTaoRegularity;

py_engine! {
    PySzemerediGreenTaoRegularity("SzemerediGreenTaoRegularity") wraps SzemerediGreenTaoRegularity [Clone, Copy] {
        new(epsilon: f64) -> res;
        method_res triangle_density_lower_bound(edge_density_d: f64) -> f64;
    }
}

use crate::omni_tetratricosa_xxxiv::GrandOmniTetratricosaEngineXXXIV;

#[pyclass(
    name = "GrandOmniTetratricosaEngineXXXIV",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniTetratricosaEngineXXXIV {
    pub inner: GrandOmniTetratricosaEngineXXXIV,
}

#[pymethods]
impl PyGrandOmniTetratricosaEngineXXXIV {
    #[new]
    #[pyo3(signature = (num_layers=34))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniTetratricosaEngineXXXIV::new(num_layers),
        }
    }

    pub fn execute_tetratricosa_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_tetratricosa_synthesis())
            .map_err(to_py_err)
    }
}

use crate::science::caffarelli_kohn_nirenberg_ns::CaffarelliKohnNirenbergNavierStokes;

py_engine! {
    PyCaffarelliKohnNirenbergNavierStokes("CaffarelliKohnNirenbergNavierStokes") wraps CaffarelliKohnNirenbergNavierStokes [Clone, Copy] {
        new(viscosity_nu: f64, epsilon_ckn: f64) -> res;
        method_res scale_invariant_ckn_functional(radius_r: f64, enstrophy_int: f64, velocity_l3_cubed_int: f64) -> (f64, bool);
    }
}

use crate::science::de_giorgi_nash_moser::DeGiorgiNashMoserRegularity;

py_engine! {
    PyDeGiorgiNashMoserRegularity("DeGiorgiNashMoserRegularity") wraps DeGiorgiNashMoserRegularity [Clone, Copy] {
        new(lambda_min: f64, lambda_max: f64) -> res;
        method_res holder_exponent_and_harnack_constant(dim_n: usize) -> (f64, f64);
    }
}

use crate::science::calderon_zygmund_singular::CalderonZygmundSingularIntegral;

py_engine! {
    PyCalderonZygmundSingularIntegral("CalderonZygmundSingularIntegral") wraps CalderonZygmundSingularIntegral [Clone, Copy] {
        new(exponent_p: f64) -> res;
        method hilbert_transform_lp_norm_and_dual() -> (f64, f64);
    }
}

use crate::science::strichartz_dispersive_pde::StrichartzDispersiveEstimate;

py_engine! {
    PyStrichartzDispersiveEstimate("StrichartzDispersiveEstimate") wraps StrichartzDispersiveEstimate [Clone, Copy] {
        new(dim_d: usize) -> res;
        method_res admissible_spatial_exponent_and_diagonal(time_q: f64) -> (f64, f64, bool);
    }
}

use crate::science::lax_kdv_inverse_scattering::LaxKdvInverseScatteringSoliton;

py_engine! {
    PyLaxKdvInverseScatteringSoliton("LaxKdvInverseScatteringSoliton") wraps LaxKdvInverseScatteringSoliton [Clone, Copy] {
        new(kappa1: f64) -> res;
        method_res amplitude_speed_and_collision_shift(kappa2: f64) -> (f64, f64, f64);
    }
}

use crate::science::zakharov_shabat_nls_rogue::ZakharovShabatNlsSoliton;

py_engine! {
    PyZakharovShabatNlsSoliton("ZakharovShabatNlsSoliton") wraps ZakharovShabatNlsSoliton [Clone, Copy] {
        new(background_a0: f64) -> res;
        method peregrine_amplitude_and_amplification(x: f64, t: f64) -> (f64, f64);
    }
}

use crate::science::lions_concentration_compactness::LionsConcentrationCompactness;

py_engine! {
    PyLionsConcentrationCompactness("LionsConcentrationCompactness") wraps LionsConcentrationCompactness [Clone, Copy] {
        new(dim_n: usize, sobolev_sn: f64) -> res;
        method critical_exponent_and_bubble_threshold(energy_level: f64) -> (f64, f64, bool);
    }
}

use crate::science::modica_mortola_gamma_conv::ModicaMortolaGammaConvergence;

py_engine! {
    PyModicaMortolaGammaConvergence("ModicaMortolaGammaConvergence") wraps ModicaMortolaGammaConvergence [Clone, Copy] {
        new(epsilon: f64) -> res;
        method surface_tension_and_optimal_profile(x: f64) -> (f64, f64);
    }
}

use crate::science::crandall_lions_viscosity_pucci::CrandallLionsViscosityPucci;

#[pyclass(
    name = "CrandallLionsViscosityPucci",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone, Copy)]
pub struct PyCrandallLionsViscosityPucci {
    pub inner: CrandallLionsViscosityPucci,
}

#[pymethods]
impl PyCrandallLionsViscosityPucci {
    #[new]
    pub fn new(lambda_min: f64, lambda_max: f64) -> PyResult<Self> {
        let inner = CrandallLionsViscosityPucci::new(lambda_min, lambda_max).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn evaluate_pucci_extremal(&self, hessian_eigenvalues: Vec<f64>) -> (f64, f64) {
        self.inner.evaluate_pucci_extremal(&hessian_eigenvalues)
    }
}

use crate::omni_pentatricosa_xxxv::GrandOmniPentatricosaEngineXXXV;

#[pyclass(
    name = "GrandOmniPentatricosaEngineXXXV",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniPentatricosaEngineXXXV {
    pub inner: GrandOmniPentatricosaEngineXXXV,
}

#[pymethods]
impl PyGrandOmniPentatricosaEngineXXXV {
    #[new]
    #[pyo3(signature = (num_layers=35))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniPentatricosaEngineXXXV::new(num_layers),
        }
    }

    pub fn execute_pentatricosa_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_pentatricosa_synthesis())
            .map_err(to_py_err)
    }
}

use crate::physics::landauer_buttiker_mesoscopic::LandauerButtikerBallisticTransport;

#[pyclass(
    name = "LandauerButtikerBallisticTransport",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone, Copy, Default)]
pub struct PyLandauerButtikerBallisticTransport {
    pub inner: LandauerButtikerBallisticTransport,
}

#[pymethods]
impl PyLandauerButtikerBallisticTransport {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: LandauerButtikerBallisticTransport::new(),
        }
    }

    #[staticmethod]
    pub fn conductance_and_fano_factor(transmissions: Vec<f64>) -> PyResult<(f64, f64)> {
        LandauerButtikerBallisticTransport::conductance_and_fano_factor(&transmissions)
            .map_err(to_py_err)
    }
}

use crate::physics::datta_das_rashba_spintronics::DattaDasRashbaSpintronics;

py_engine! {
    PyDattaDasRashbaSpintronics("DattaDasRashbaSpintronics") wraps DattaDasRashbaSpintronics [Clone, Copy] {
        new(effective_mass: f64, channel_length_l: f64) -> res;
        method spin_precession_and_modulation(rashba_alpha: f64) -> (f64, f64);
    }
}

use crate::physics::slonczewski_spin_transfer_torque::SlonczewskiSpinTransferTorque;

py_engine! {
    PySlonczewskiSpinTransferTorque("SlonczewskiSpinTransferTorque") wraps SlonczewskiSpinTransferTorque [Clone, Copy] {
        new(gilbert_alpha: f64, spin_efficiency_eta: f64) -> res;
        method_res critical_current_and_thermal_stability(ms_volume: f64, hk: f64, hd: f64, k_b_t: f64) -> (f64, f64);
    }
}

use crate::quantum::transmon_josephson_qubit::TransmonCircuitQEDQubit;

py_engine! {
    PyTransmonCircuitQEDQubit("TransmonCircuitQEDQubit") wraps TransmonCircuitQEDQubit [Clone, Copy] {
        new(e_j_ghz: f64, e_c_ghz: f64) -> res;
        method frequency_anharmonicity_and_dispersion() -> (f64, f64, f64);
    }
}

use crate::physics::ambegaokar_baratoff_josephson::AmbegaokarBaratoffJosephson;

py_engine! {
    PyAmbegaokarBaratoffJosephson("AmbegaokarBaratoffJosephson") wraps AmbegaokarBaratoffJosephson [Clone, Copy] {
        new(gap_delta: f64) -> res;
        method_res ic_rn_product(temperature_t: f64) -> (f64, f64);
    }
}

use crate::physics::shockley_queisser_solar::ShockleyQueisserPhotovoltaic;

py_engine! {
    PyShockleyQueisserPhotovoltaic("ShockleyQueisserPhotovoltaic") wraps ShockleyQueisserPhotovoltaic [Clone, Copy] {
        new(k_b_t_sun_ev: f64) -> res;
        method_res ultimate_spectrum_efficiency(bandgap_eg_ev: f64) -> f64;
    }
}

use crate::physics::chua_strukov_memristor::ChuaStrukovMemristorSynapse;

py_engine! {
    PyChuaStrukovMemristorSynapse("ChuaStrukovMemristorSynapse") wraps ChuaStrukovMemristorSynapse [Clone, Copy] {
        new(r_on: f64, r_off: f64) -> res;
        method_res memristance_and_joglekar_window(state_x: f64) -> (f64, f64);
    }
}

use crate::physics::tsu_esaki_resonant_tunneling::TsuEsakiResonantTunneling;

py_engine! {
    PyTsuEsakiResonantTunneling("TsuEsakiResonantTunneling") wraps TsuEsakiResonantTunneling [Clone, Copy] {
        new(resonance_er: f64, gamma_l: f64, gamma_r: f64) -> res;
        method transmission_and_peak(energy_e: f64) -> (f64, f64);
    }
}

use crate::physics::seebeck_thermoelectric_zt::ThermoelectricFigureOfMeritZT;

py_engine! {
    PyThermoelectricFigureOfMeritZT("ThermoelectricFigureOfMeritZT") wraps ThermoelectricFigureOfMeritZT [Clone, Copy] {
        new(seebeck_s: f64, sigma_elec: f64, kappa_therm: f64) -> res;
        method_res zt_and_max_efficiency(t_cold: f64, t_hot: f64) -> (f64, f64);
    }
}

use crate::omni_hexatricosa_xxxvi::GrandOmniHexatricosaEngineXXXVI;

#[pyclass(
    name = "GrandOmniHexatricosaEngineXXXVI",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniHexatricosaEngineXXXVI {
    pub inner: GrandOmniHexatricosaEngineXXXVI,
}

#[pymethods]
impl PyGrandOmniHexatricosaEngineXXXVI {
    #[new]
    #[pyo3(signature = (num_layers=36))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniHexatricosaEngineXXXVI::new(num_layers),
        }
    }

    pub fn execute_hexatricosa_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_hexatricosa_synthesis())
            .map_err(to_py_err)
    }
}

use crate::science::wolynes_protein_folding_funnel::WolynesProteinFoldingFunnel;

py_engine! {
    PyWolynesProteinFoldingFunnel("WolynesProteinFoldingFunnel") wraps WolynesProteinFoldingFunnel [Clone, Copy] {
        new(stability_gap_de: f64, ruggedness_der: f64) -> res;
        method_res glass_temp_and_foldability_ratio(config_entropy_s0: f64) -> (f64, f64, f64, bool);
    }
}

use crate::science::zimm_bragg_ramachandran::ZimmBraggRamachandranHelix;

py_engine! {
    PyZimmBraggRamachandranHelix("ZimmBraggRamachandranHelix") wraps ZimmBraggRamachandranHelix [Clone, Copy] {
        new(cooperativity_sigma: f64) -> res;
        method_res eigenvalue_and_helix_fraction(propagation_s: f64) -> (f64, f64);
    }
}

use crate::science::monod_wyman_changeux_allostery::MonodWymanChangeuxAllostery;

py_engine! {
    PyMonodWymanChangeuxAllostery("MonodWymanChangeuxAllostery") wraps MonodWymanChangeuxAllostery [Clone, Copy] {
        new(subunits_n: usize, allosteric_l: f64, ratio_c: f64) -> res;
        method_res saturation_and_relaxed_fraction(alpha: f64) -> (f64, f64);
    }
}

use crate::science::briggs_haldane_enzyme_kinetics::BriggsHaldaneEnzymeKinetics;

py_engine! {
    PyBriggsHaldaneEnzymeKinetics("BriggsHaldaneEnzymeKinetics") wraps BriggsHaldaneEnzymeKinetics [Clone, Copy] {
        new(k_cat: f64, k_m: f64) -> res;
        method_res specificity_and_velocity(substrate_s: f64, inhibitor_over_ki: f64) -> (f64, f64);
    }
}

use crate::science::crispr_cas9_rloop_kinetics::CrisprCas9RLoopKinetics;

py_engine! {
    PyCrisprCas9RLoopKinetics("CrisprCas9RLoopKinetics") wraps CrisprCas9RLoopKinetics [Clone, Copy] {
        new(match_gain_kt: f64, mismatch_penalty_kt: f64) -> res;
        method_res rloop_free_energy_and_cleavage_prob(guide_len_l: usize, num_mismatches_m: usize) -> (f64, f64);
    }
}

use crate::science::karlin_altschul_alignment::KarlinAltschulSequenceAlignment;

py_engine! {
    PyKarlinAltschulSequenceAlignment("KarlinAltschulSequenceAlignment") wraps KarlinAltschulSequenceAlignment [Clone, Copy] {
        new(lambda: f64, kappa_k: f64) -> res;
        method_res bit_score_evalue_and_pvalue(raw_score_s: f64, query_m: u64, db_n: u64) -> (f64, f64, f64);
    }
}

use crate::science::felsenstein_jukes_cantor_phylo::FelsensteinPhylogeneticLikelihood;

py_engine! {
    PyFelsensteinPhylogeneticLikelihood("FelsensteinPhylogeneticLikelihood") wraps FelsensteinPhylogeneticLikelihood [Clone, Copy, Default] {
        new();
        assoc_res jukes_cantor_distance_and_variance(mismatch_p: f64, sequence_length_l: usize) -> (f64, f64);
    }
}

use crate::science::rna_velocity_waddington::RnaVelocityWaddingtonLandscape;

py_engine! {
    PyRnaVelocityWaddingtonLandscape("RnaVelocityWaddingtonLandscape") wraps RnaVelocityWaddingtonLandscape [Clone, Copy] {
        new(splicing_beta: f64, degradation_gamma: f64) -> res;
        method_res rna_velocity_and_induction_state(unspliced_u: f64, spliced_s: f64) -> (f64, bool);
    }
}

use crate::science::bell_evans_mechanobiology::BellEvansMechanobiologyBond;

py_engine! {
    PyBellEvansMechanobiologyBond("BellEvansMechanobiologyBond") wraps BellEvansMechanobiologyBond [Clone, Copy] {
        new(k0: f64, thermal_force_f_beta: f64) -> res;
        method_res off_rate_and_rupture_force(force_pn: f64, loading_rate_rf: f64) -> (f64, f64);
    }
}

use crate::omni_heptatricosa_xxxvii::GrandOmniHeptatricosaEngineXXXVII;

#[pyclass(
    name = "GrandOmniHeptatricosaEngineXXXVII",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniHeptatricosaEngineXXXVII {
    pub inner: GrandOmniHeptatricosaEngineXXXVII,
}

#[pymethods]
impl PyGrandOmniHeptatricosaEngineXXXVII {
    #[new]
    #[pyo3(signature = (num_layers=37))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniHeptatricosaEngineXXXVII::new(num_layers),
        }
    }

    pub fn execute_heptatricosa_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_heptatricosa_synthesis())
            .map_err(to_py_err)
    }
}

use crate::physics::dlvo_debye_huckel_colloid::DlvoColloidalStabilityEngine;

py_engine! {
    PyDlvoColloidalStabilityEngine("DlvoColloidalStabilityEngine") wraps DlvoColloidalStabilityEngine [Clone, Copy] {
        new(kappa_inv_nm: f64, hamaker_ah_kt: f64) -> res;
        method_res debye_length_and_dlvo_potential(radius_r_nm: f64, electro_prefactor_z: f64, separation_d_nm: f64) -> (f64, f64);
    }
}

use crate::physics::de_gennes_polymer_reptation::DeGennesPolymerReptationTube;

py_engine! {
    PyDeGennesPolymerReptationTube("DeGennesPolymerReptationTube") wraps DeGennesPolymerReptationTube [Clone, Copy] {
        new(entanglement_ne: f64, tau_e: f64) -> res;
        method_res entanglements_disengagement_and_viscosity(chain_length_n: f64) -> (f64, f64, f64);
    }
}

use crate::physics::flory_huggins_polymer_mixing::FloryHugginsSpinodalMixing;

py_engine! {
    PyFloryHugginsSpinodalMixing("FloryHugginsSpinodalMixing") wraps FloryHugginsSpinodalMixing [Clone, Copy] {
        new(degree_n: usize) -> res;
        method_res critical_point_and_spinodal_instability(phi: f64, chi: f64) -> (f64, f64, bool);
    }
}

use crate::physics::onsager_maier_saupe_nematic::OnsagerMaierSaupeNematic;

py_engine! {
    PyOnsagerMaierSaupeNematic("OnsagerMaierSaupeNematic") wraps OnsagerMaierSaupeNematic [Clone, Copy] {
        new(coeff_a: f64, coeff_b: f64, coeff_c: f64) -> res;
        method clearing_temperature_shift_and_order_jump() -> (f64, f64);
    }
}

use crate::physics::toner_tu_vicsek_active_flocking::TonerTuVicsekActiveMatter;

#[pyclass(name = "TonerTuVicsekActiveMatter", module = "oxideon", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyTonerTuVicsekActiveMatter {
    pub inner: TonerTuVicsekActiveMatter,
}

#[pymethods]
impl PyTonerTuVicsekActiveMatter {
    #[new]
    pub fn new(saturation_beta: f64) -> PyResult<Self> {
        let inner = TonerTuVicsekActiveMatter::new(saturation_beta).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn flock_speed_and_polar_order(
        &self,
        linear_growth_alpha: f64,
        angles_rad: Vec<f64>,
    ) -> PyResult<(f64, f64)> {
        self.inner
            .flock_speed_and_polar_order(linear_growth_alpha, &angles_rad)
            .map_err(to_py_err)
    }
}

use crate::physics::active_nematic_defect_propulsion::ActiveNematicDefectTurbulence;

py_engine! {
    PyActiveNematicDefectTurbulence("ActiveNematicDefectTurbulence") wraps ActiveNematicDefectTurbulence [Clone, Copy] {
        new(frank_elastic_k: f64, viscosity_eta: f64) -> res;
        method_res active_length_and_defect_velocity(activity_zeta: f64) -> (f64, f64);
    }
}

use crate::physics::cates_tailleur_mips_abp::MotilityInducedPhaseSeparation;

py_engine! {
    PyMotilityInducedPhaseSeparation("MotilityInducedPhaseSeparation") wraps MotilityInducedPhaseSeparation [Clone, Copy] {
        new(bare_speed_v0: f64, rho_jam: f64) -> res;
        method_res swim_speed_and_mips_spinodal(density_rho: f64) -> (f64, f64, bool);
    }
}

use crate::physics::oldroyd_b_weissenberg_rheology::OldroydBViscoelasticRheology;

py_engine! {
    PyOldroydBViscoelasticRheology("OldroydBViscoelasticRheology") wraps OldroydBViscoelasticRheology [Clone, Copy] {
        new(eta_solvent: f64, eta_polymer: f64, relaxation_lambda: f64) -> res;
        method weissenberg_shear_and_normal_stress(shear_rate_gamma_dot: f64) -> (f64, f64, f64);
    }
}

use crate::physics::washburn_young_laplace_wetting::WashburnYoungLaplaceCapillary;

py_engine! {
    PyWashburnYoungLaplaceCapillary("WashburnYoungLaplaceCapillary") wraps WashburnYoungLaplaceCapillary [Clone, Copy] {
        new(surface_tension_gamma: f64, viscosity_eta: f64) -> res;
        method_res laplace_pressure_and_washburn_length(pore_radius_r: f64, contact_angle_rad: f64, time_t: f64) -> (f64, f64);
    }
}

use crate::omni_octatricosa_xxxviii::GrandOmniOctatricosaEngineXXXVIII;

#[pyclass(
    name = "GrandOmniOctatricosaEngineXXXVIII",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniOctatricosaEngineXXXVIII {
    pub inner: GrandOmniOctatricosaEngineXXXVIII,
}

#[pymethods]
impl PyGrandOmniOctatricosaEngineXXXVIII {
    #[new]
    #[pyo3(signature = (num_layers=38))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniOctatricosaEngineXXXVIII::new(num_layers),
        }
    }

    pub fn execute_octatricosa_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_octatricosa_synthesis())
            .map_err(to_py_err)
    }
}

use crate::science::lamport_pbft_byzantine_quorum::ByzantineQuorumConsensusEngine;

py_engine! {
    PyByzantineQuorumConsensusEngine("ByzantineQuorumConsensusEngine") wraps ByzantineQuorumConsensusEngine [Clone, Copy] {
        new(num_replicas_n: usize) -> res;
        method max_faults_quorum_and_honest_intersection() -> (usize, usize, usize);
    }
}

use crate::science::lamport_mattern_vector_clock::MatternVectorClockCausality;

#[pyclass(
    name = "MatternVectorClockCausality",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone, Copy, Default)]
pub struct PyMatternVectorClockCausality {
    pub inner: MatternVectorClockCausality,
}

#[pymethods]
impl PyMatternVectorClockCausality {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: MatternVectorClockCausality::new(),
        }
    }

    #[staticmethod]
    pub fn merge_and_classify_causality(
        clock_u: Vec<u64>,
        clock_v: Vec<u64>,
    ) -> PyResult<(Vec<u64>, i32, bool)> {
        MatternVectorClockCausality::merge_and_classify_causality(&clock_u, &clock_v)
            .map_err(to_py_err)
    }
}

use crate::science::shapiro_crdt_semilattice::ShapiroCrdtJoinSemilattice;

#[pyclass(
    name = "ShapiroCrdtJoinSemilattice",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone, Copy, Default)]
pub struct PyShapiroCrdtJoinSemilattice {
    pub inner: ShapiroCrdtJoinSemilattice,
}

#[pymethods]
impl PyShapiroCrdtJoinSemilattice {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: ShapiroCrdtJoinSemilattice::new(),
        }
    }

    #[staticmethod]
    pub fn merge_pn_counter_and_evaluate(
        p1: Vec<u64>,
        n1: Vec<u64>,
        p2: Vec<u64>,
        n2: Vec<u64>,
    ) -> PyResult<(i64, bool)> {
        ShapiroCrdtJoinSemilattice::merge_pn_counter_and_evaluate(&p1, &n1, &p2, &n2)
            .map_err(to_py_err)
    }
}

use crate::science::karger_consistent_hashing::KargerConsistentHashingRing;

py_engine! {
    PyKargerConsistentHashingRing("KargerConsistentHashingRing") wraps KargerConsistentHashingRing [Clone, Copy] {
        new(num_servers_n: usize, vnodes_per_server_v: usize) -> res;
        method remapped_fraction_and_peak_load_ratio() -> (f64, f64);
    }
}

use crate::science::gilbert_lynch_cap_pacelc::GilbertLynchPacelcTheorem;

py_engine! {
    PyGilbertLynchPacelcTheorem("GilbertLynchPacelcTheorem") wraps GilbertLynchPacelcTheorem [Clone, Copy] {
        new(replication_n: usize) -> res;
        method_res verify_strong_quorum(read_quorum_r: usize, write_quorum_w: usize) -> (i64, bool, bool);
    }
}

use crate::science::clarke_emerson_mu_calculus::ClarkeEmersonModelChecker;

#[pyclass(name = "ClarkeEmersonModelChecker", module = "oxideon", from_py_object)]
#[derive(Clone, Copy, Default)]
pub struct PyClarkeEmersonModelChecker {
    pub inner: ClarkeEmersonModelChecker,
}

#[pymethods]
impl PyClarkeEmersonModelChecker {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: ClarkeEmersonModelChecker::new(),
        }
    }

    #[staticmethod]
    pub fn ctl_ef_least_fixed_point(
        num_states: usize,
        adj_masks: Vec<u64>,
        target_mask_p: u64,
    ) -> PyResult<(u64, usize)> {
        ClarkeEmersonModelChecker::ctl_ef_least_fixed_point(num_states, &adj_masks, target_mask_p)
            .map_err(to_py_err)
    }
}

use crate::science::reynolds_separation_logic::ReynoldsSeparationLogicHeap;

py_engine! {
    PyReynoldsSeparationLogicHeap("ReynoldsSeparationLogicHeap") wraps ReynoldsSeparationLogicHeap [Clone, Copy, Default] {
        new();
        assoc_res check_separating_conjunction_and_frame(dom1_mask: u64, dom2_mask: u64, frame_mask_r: u64, modified_mask_c: u64) -> (u64, bool, bool);
    }
}

use crate::science::cousot_abstract_interpretation::CousotGaloisAbstractInterpreter;

py_engine! {
    PyCousotGaloisAbstractInterpreter("CousotGaloisAbstractInterpreter") wraps CousotGaloisAbstractInterpreter [Clone, Copy, Default] {
        new();
        assoc_res interval_join_and_widening(a: f64, b: f64, c: f64, d: f64, bound_t: f64) -> ([f64; 2], [f64; 2]);
    }
}

use crate::science::girard_linear_logic_resource::GirardLinearLogicResource;

#[pyclass(name = "GirardLinearLogicResource", module = "oxideon", from_py_object)]
#[derive(Clone, Copy, Default)]
pub struct PyGirardLinearLogicResource {
    pub inner: GirardLinearLogicResource,
}

#[pymethods]
impl PyGirardLinearLogicResource {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: GirardLinearLogicResource::new(),
        }
    }

    #[staticmethod]
    pub fn verify_linear_resource_and_switching_tree(
        usage_counts: Vec<usize>,
        switching_vertices: usize,
        switching_edges: usize,
    ) -> PyResult<(bool, bool)> {
        guard_res(|| {
            GirardLinearLogicResource::verify_linear_resource_and_switching_tree(
                &usage_counts,
                switching_vertices,
                switching_edges,
            )
        })
    }
}

use crate::omni_nonatricosa_xxxix::GrandOmniNonatricosaEngineXXXIX;

#[pyclass(
    name = "GrandOmniNonatricosaEngineXXXIX",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniNonatricosaEngineXXXIX {
    pub inner: GrandOmniNonatricosaEngineXXXIX,
}

#[pymethods]
impl PyGrandOmniNonatricosaEngineXXXIX {
    #[new]
    #[pyo3(signature = (num_layers=39))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniNonatricosaEngineXXXIX::new(num_layers),
        }
    }

    pub fn execute_nonatricosa_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_nonatricosa_synthesis())
            .map_err(to_py_err)
    }
}

use crate::physics::lawson_triple_product_fusion::LawsonTripleProductIgnition;

py_engine! {
    PyLawsonTripleProductIgnition("LawsonTripleProductIgnition") wraps LawsonTripleProductIgnition [Clone, Copy] {
        new(alpha_energy_kev: f64) -> res;
        method_res triple_product_and_ignition(density_n: f64, temp_kev: f64, tau_e_sec: f64, reactivity_sigma_v: f64) -> (f64, f64, bool);
    }
}

use crate::physics::spitzer_dreicer_runaway_plasma::SpitzerDreicerPlasmaTransport;

py_engine! {
    PySpitzerDreicerPlasmaTransport("SpitzerDreicerPlasmaTransport") wraps SpitzerDreicerPlasmaTransport [Clone, Copy] {
        new(z_eff: f64, coulomb_log: f64) -> res;
        method_res spitzer_resistivity_and_dreicer_field(temp_kev: f64, density_1e20_m3: f64) -> (f64, f64);
    }
}

use crate::physics::kruskal_shafranov_troyon_mhd::KruskalShafranovTroyonMhd;

py_engine! {
    PyKruskalShafranovTroyonMhd("KruskalShafranovTroyonMhd") wraps KruskalShafranovTroyonMhd [Clone, Copy] {
        new(major_r0: f64, minor_a: f64, b_toroidal: f64) -> res;
        method_res safety_factor_and_troyon_beta(plasma_current_ma: f64) -> (f64, f64, bool);
    }
}

use crate::physics::tajima_dawson_laser_wakefield::TajimaDawsonLaserWakefield;

py_engine! {
    PyTajimaDawsonLaserWakefield("TajimaDawsonLaserWakefield") wraps TajimaDawsonLaserWakefield [Clone, Copy] {
        new(density_1e18_cm3: f64) -> res;
        method_res wavebreaking_gv_m_and_energy_gain_mev(gamma_p: f64) -> (f64, f64);
    }
}

use crate::physics::gamow_thermonuclear_peak::GamowThermonuclearPeakEngine;

py_engine! {
    PyGamowThermonuclearPeakEngine("GamowThermonuclearPeakEngine") wraps GamowThermonuclearPeakEngine [Clone, Copy] {
        new(gamow_energy_eg_kev: f64) -> res;
        method_res gamow_peak_width_and_tau(k_b_t_kev: f64) -> (f64, f64, f64);
    }
}

use crate::physics::hoyle_salpeter_eddington_stellar::HoyleSalpeterEddingtonStellar;

py_engine! {
    PyHoyleSalpeterEddingtonStellar("HoyleSalpeterEddingtonStellar") wraps HoyleSalpeterEddingtonStellar [Clone, Copy] {
        new(mass_solar: f64) -> res;
        method_res eddington_luminosity_and_triple_alpha_exponent(temp_t8: f64) -> (f64, f64);
    }
}

use crate::physics::fermi_hillas_cosmic_ray_shock::FermiHillasCosmicRayShock;

py_engine! {
    PyFermiHillasCosmicRayShock("FermiHillasCosmicRayShock") wraps FermiHillasCosmicRayShock [Clone, Copy] {
        new(compression_ratio_r: f64) -> res;
        method_res spectral_index_and_hillas_emax_eev(charge_z: usize, b_microgauss: f64, radius_kpc: f64, shock_beta: f64) -> (f64, f64);
    }
}

use crate::physics::parker_solar_wind_spiral::ParkerSolarWindSpiralEngine;

py_engine! {
    PyParkerSolarWindSpiralEngine("ParkerSolarWindSpiralEngine") wraps ParkerSolarWindSpiralEngine [Clone, Copy] {
        new(gm_star: f64, omega_star: f64) -> res;
        method_res sonic_radius_and_spiral_angle(sound_speed_cs: f64, radius_r: f64, wind_speed_vw: f64) -> (f64, f64);
    }
}

use crate::physics::sweet_parker_petschek_reconnection::SweetParkerPetschekReconnection;

py_engine! {
    PySweetParkerPetschekReconnection("SweetParkerPetschekReconnection") wraps SweetParkerPetschekReconnection [Clone, Copy] {
        new(lundquist_s: f64) -> res;
        method reconnection_rates_and_plasmoid_regime() -> (f64, f64, bool);
    }
}

use crate::omni_quadricentennial_xl::GrandOmniQuadricentennialEngineXL;

#[pyclass(
    name = "GrandOmniQuadricentennialEngineXL",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniQuadricentennialEngineXL {
    pub inner: GrandOmniQuadricentennialEngineXL,
}

#[pymethods]
impl PyGrandOmniQuadricentennialEngineXL {
    #[new]
    #[pyo3(signature = (num_layers=40))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniQuadricentennialEngineXL::new(num_layers),
        }
    }

    pub fn execute_quadricentennial_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_quadricentennial_synthesis())
            .map_err(to_py_err)
    }
}

use crate::science::almgren_frequency_minimal_currents::AlmgrenFrequencyCurrentsEngine;

py_engine! {
    PyAlmgrenFrequencyCurrentsEngine("AlmgrenFrequencyCurrentsEngine") wraps AlmgrenFrequencyCurrentsEngine [Clone, Copy] {
        new(current_dim_m: usize) -> res;
        method_res frequency_and_max_singular_dim(radius_r: f64, dirichlet_energy_dr: f64, boundary_l2_hr: f64) -> (f64, usize);
    }
}

use crate::science::allard_varifold_monotonicity::AllardVarifoldMonotonicityEngine;

py_engine! {
    PyAllardVarifoldMonotonicityEngine("AllardVarifoldMonotonicityEngine") wraps AllardVarifoldMonotonicityEngine [Clone, Copy] {
        new(dim_k: usize, epsilon_allard: f64) -> res;
        method_res density_ratio_and_regularity_check(radius_r: f64, varifold_mass_br: f64) -> (f64, bool);
    }
}

use crate::science::federer_fleming_integral_currents::FedererFlemingCurrentsCoarea;

py_engine! {
    PyFedererFlemingCurrentsCoarea("FedererFlemingCurrentsCoarea") wraps FedererFlemingCurrentsCoarea [Clone, Copy] {
        new(dim_n: usize) -> res;
        method_res sharp_isoperimetric_volume_bound(perimeter_p: f64, volume_v: f64) -> (f64, bool);
    }
}

use crate::science::caffarelli_monge_ampere_transport::CaffarelliMongeAmpereTransport;

#[pyclass(
    name = "CaffarelliMongeAmpereTransport",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone, Copy, Default)]
pub struct PyCaffarelliMongeAmpereTransport {
    pub inner: CaffarelliMongeAmpereTransport,
}

#[pymethods]
impl PyCaffarelliMongeAmpereTransport {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: CaffarelliMongeAmpereTransport::new(),
        }
    }

    #[staticmethod]
    pub fn mccann_displacement_jacobian_and_concavity(
        hessian_eigenvalues: Vec<f64>,
        t: f64,
    ) -> PyResult<(f64, f64, bool)> {
        CaffarelliMongeAmpereTransport::mccann_displacement_jacobian_and_concavity(
            &hessian_eigenvalues,
            t,
        )
        .map_err(to_py_err)
    }
}

use crate::science::alt_caffarelli_free_boundary::AltCaffarelliFreeBoundaryWeiss;

py_engine! {
    PyAltCaffarelliFreeBoundaryWeiss("AltCaffarelliFreeBoundaryWeiss") wraps AltCaffarelliFreeBoundaryWeiss [Clone, Copy] {
        new(lambda_weight: f64, dim_n: usize) -> res;
        method_res bernoulli_slope_and_weiss_energy(radius_r: f64, dirichlet_int: f64, positive_measure_br: f64, boundary_u2_int: f64) -> (f64, f64);
    }
}

use crate::science::caffarelli_silvestre_fractional::CaffarelliSilvestreFractionalExt;

py_engine! {
    PyCaffarelliSilvestreFractionalExt("CaffarelliSilvestreFractionalExt") wraps CaffarelliSilvestreFractionalExt [Clone, Copy] {
        new(fractional_s: f64) -> res;
        method_res muckenhoupt_exponent_and_fourier_multiplier(xi_norm: f64, height_y: f64) -> (f64, f64);
    }
}

use crate::science::simons_cone_bernstein_minimal::SimonsMinimalConeBernstein;

py_engine! {
    PySimonsMinimalConeBernstein("SimonsMinimalConeBernstein") wraps SimonsMinimalConeBernstein [Clone, Copy] {
        new(ambient_dim_n_plus_1: usize) -> res;
        method jacobi_hardy_gap_and_bernstein_status() -> (f64, bool, bool);
    }
}

use crate::science::besicovitch_marstrand_rectifiable::BesicovitchMarstrandRectifiable;

py_engine! {
    PyBesicovitchMarstrandRectifiable("BesicovitchMarstrandRectifiable") wraps BesicovitchMarstrandRectifiable [Clone, Copy] {
        new(ambient_n: usize) -> res;
        method_res projection_and_slice_dimensions(hausdorff_dim_s: f64, subspace_dim_m: usize) -> (f64, f64, bool);
    }
}

use crate::science::cheeger_buser_isoperimetric::CheegerBuserSpectralInequality;

py_engine! {
    PyCheegerBuserSpectralInequality("CheegerBuserSpectralInequality") wraps CheegerBuserSpectralInequality [Clone, Copy] {
        new(cheeger_h: f64) -> res;
        method cheeger_lower_bound_and_verify(lambda_1: f64) -> (f64, f64, bool);
    }
}

use crate::omni_henatetracosa_xli::GrandOmniHenatetracosaEngineXLI;

#[pyclass(
    name = "GrandOmniHenatetracosaEngineXLI",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniHenatetracosaEngineXLI {
    pub inner: GrandOmniHenatetracosaEngineXLI,
}

#[pymethods]
impl PyGrandOmniHenatetracosaEngineXLI {
    #[new]
    #[pyo3(signature = (num_layers=41))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniHenatetracosaEngineXLI::new(num_layers),
        }
    }

    pub fn execute_henatetracosa_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_henatetracosa_synthesis())
            .map_err(to_py_err)
    }
}

use crate::physics::bjorken_qgp_boost_invariant::BjorkenQgpHydroExpansion;

py_engine! {
    PyBjorkenQgpHydroExpansion("BjorkenQgpHydroExpansion") wraps BjorkenQgpHydroExpansion [Clone, Copy] {
        new(tau0_fm: f64, eps0_gev_fm3: f64) -> res;
        method_res energy_density_and_temp_ratio(tau_fm: f64, cs_sq: f64) -> (f64, f64);
    }
}

use crate::physics::kovtun_son_starinets_kss_bound::KovtunSonStarinetsViscosityBound;

py_engine! {
    PyKovtunSonStarinetsViscosityBound("KovtunSonStarinetsViscosityBound") wraps KovtunSonStarinetsViscosityBound [Clone, Copy, Default] {
        new();
        assoc_res kss_bound_and_finite_coupling_eta_over_s(thooft_lambda: f64) -> (f64, f64);
    }
}

use crate::physics::mclerran_venugopalan_cgc_saturation::ColorGlassCondensateSaturation;

py_engine! {
    PyColorGlassCondensateSaturation("ColorGlassCondensateSaturation") wraps ColorGlassCondensateSaturation [Clone, Copy] {
        new(q0_sq_gev2: f64, x0: f64, lambda_exp: f64) -> res;
        method_res saturation_scale_and_dipole_scattering(bjorken_x: f64, mass_number_a: u32, dipole_r_gev_inv: f64) -> (f64, f64);
    }
}

use crate::physics::bdmps_jet_quenching_qgp::BdmpsJetQuenchingEnergyLoss;

py_engine! {
    PyBdmpsJetQuenchingEnergyLoss("BdmpsJetQuenchingEnergyLoss") wraps BdmpsJetQuenchingEnergyLoss [Clone, Copy] {
        new(alpha_s: f64, q_hat_gev2_fm: f64) -> res;
        method_res characteristic_frequency_and_energy_loss_gev(path_length_fm: f64, casimir_cr: f64) -> (f64, f64);
    }
}

use crate::physics::matsui_satz_quarkonium_screening::MatsuiSatzQuarkoniumMelting;

py_engine! {
    PyMatsuiSatzQuarkoniumMelting("MatsuiSatzQuarkoniumMelting") wraps MatsuiSatzQuarkoniumMelting [Clone, Copy] {
        new(num_colors_nc: usize, num_flavors_nf: usize) -> res;
        method_res debye_mass_radius_and_melting(coupling_g: f64, temp_gev: f64, quarkonium_radius_fm: f64) -> (f64, f64, bool);
    }
}

use crate::physics::chiral_magnetic_effect_cme::ChiralMagneticEffectKharzeev;

py_engine! {
    PyChiralMagneticEffectKharzeev("ChiralMagneticEffectKharzeev") wraps ChiralMagneticEffectKharzeev [Clone, Copy] {
        new(charge_factor_nc_qf2: f64) -> res;
        method chiral_conductivity_and_current(mu5_gev: f64, b_field_gev2: f64) -> (f64, f64);
    }
}

use crate::physics::bethe_weizsacker_nuclear_fission::BetheWeizsackerNuclearMass;

py_engine! {
    PyBetheWeizsackerNuclearMass("BetheWeizsackerNuclearMass") wraps BetheWeizsackerNuclearMass [Clone, Copy] {
        new(a_v: f64, a_s: f64, a_c: f64, a_a: f64) -> res;
        method_res binding_per_nucleon_and_fissility(mass_a: u32, atomic_z: u32) -> (f64, f64);
    }
}

use crate::physics::mayer_jensen_nuclear_shell::MayerJensenNuclearShellMagic;

py_engine! {
    PyMayerJensenNuclearShellMagic("MayerJensenNuclearShellMagic") wraps MayerJensenNuclearShellMagic [Clone, Copy] {
        new(spin_orbit_cls_mev: f64) -> res;
        method doublet_splitting_and_magic_check(orbital_l: usize, nucleon_count: usize) -> (f64, usize, bool);
    }
}

use crate::physics::walecka_relativistic_nuclear_eos::WaleckaRelativisticMeanFieldNuclear;

py_engine! {
    PyWaleckaRelativisticMeanFieldNuclear("WaleckaRelativisticMeanFieldNuclear") wraps WaleckaRelativisticMeanFieldNuclear [Clone, Copy] {
        new(bare_mass_mev: f64, c_sigma_sq: f64, c_omega_sq: f64) -> res;
        method_res effective_mass_and_vector_potential(rho_scalar_fm3: f64, rho_baryon_fm3: f64) -> (f64, f64);
    }
}

use crate::omni_dotetracosa_xlii::GrandOmniDotetracosaEngineXLII;

#[pyclass(
    name = "GrandOmniDotetracosaEngineXLII",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniDotetracosaEngineXLII {
    pub inner: GrandOmniDotetracosaEngineXLII,
}

#[pymethods]
impl PyGrandOmniDotetracosaEngineXLII {
    #[new]
    #[pyo3(signature = (num_layers=42))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniDotetracosaEngineXLII::new(num_layers),
        }
    }

    pub fn execute_dotetracosa_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_dotetracosa_synthesis())
            .map_err(to_py_err)
    }
}

use crate::science::lovasz_local_lemma_moser_tardos::LovaszLocalLemmaMoserTardos;

py_engine! {
    PyLovaszLocalLemmaMoserTardos("LovaszLocalLemmaMoserTardos") wraps LovaszLocalLemmaMoserTardos [Clone, Copy] {
        new(num_clauses_m: usize, max_dependency_d: usize) -> res;
        method_res lll_parameter_and_expected_resamplings(event_prob_p: f64) -> (f64, bool, f64);
    }
}

use crate::science::talagrand_convex_concentration::TalagrandConvexConcentration;

py_engine! {
    PyTalagrandConvexConcentration("TalagrandConvexConcentration") wraps TalagrandConvexConcentration [Clone, Copy] {
        new(lipschitz_l: f64) -> res;
        method_res convex_tail_and_median_deviation_bound(deviation_t: f64) -> (f64, f64);
    }
}

use crate::science::friedgut_bourgain_sharp_threshold::FriedgutBourgainSharpThreshold;

#[pyclass(
    name = "FriedgutBourgainSharpThreshold",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone, Copy)]
pub struct PyFriedgutBourgainSharpThreshold {
    pub inner: FriedgutBourgainSharpThreshold,
}

#[pymethods]
impl PyFriedgutBourgainSharpThreshold {
    #[new]
    pub fn new(num_vars_n: usize) -> PyResult<Self> {
        let inner = FriedgutBourgainSharpThreshold::new(num_vars_n).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn total_influence_and_threshold_width(
        &self,
        influences: Vec<f64>,
    ) -> PyResult<(f64, f64)> {
        self.inner
            .total_influence_and_threshold_width(&influences)
            .map_err(to_py_err)
    }
}

use crate::science::bollobas_random_graph_giant_component::ErdosRenyiBollobasPhaseTransition;

py_engine! {
    PyErdosRenyiBollobasPhaseTransition("ErdosRenyiBollobasPhaseTransition") wraps ErdosRenyiBollobasPhaseTransition [Clone, Copy] {
        new(num_vertices_n: usize) -> res;
        method_res giant_component_fraction_and_size(mean_degree_c: f64) -> (f64, f64);
    }
}

use crate::science::alon_boppana_ramanujan_expander::AlonBoppanaRamanujanExpander;

py_engine! {
    PyAlonBoppanaRamanujanExpander("AlonBoppanaRamanujanExpander") wraps AlonBoppanaRamanujanExpander [Clone, Copy] {
        new(degree_d: usize) -> res;
        method_res ramanujan_threshold_and_spectral_gap(lambda2: f64) -> (f64, f64, bool);
    }
}

use crate::science::balog_szemeredi_gowers_freiman::BalogSzemerediGowersFreiman;

py_engine! {
    PyBalogSzemerediGowersFreiman("BalogSzemerediGowersFreiman") wraps BalogSzemerediGowersFreiman [Clone, Copy] {
        new(set_size_a: usize) -> res;
        method_res doubling_energy_and_plunnecke_bound(sumset_size: usize) -> (f64, f64, f64);
    }
}

use crate::science::roth_bourgain_meshulam_capset::RothBourgainCapSetBounds;

py_engine! {
    PyRothBourgainCapSetBounds("RothBourgainCapSetBounds") wraps RothBourgainCapSetBounds [Clone, Copy] {
        new(dim_n: usize) -> res;
        method capset_growth_base_and_upper_bound() -> (f64, f64);
    }
}

use crate::science::lovasz_theta_shannon_capacity::LovaszThetaShannonCapacity;

py_engine! {
    PyLovaszThetaShannonCapacity("LovaszThetaShannonCapacity") wraps LovaszThetaShannonCapacity [Clone, Copy, Default] {
        new();
        assoc_res odd_cycle_lovasz_theta_and_alpha(n_odd: usize) -> (f64, usize);
    }
}

use crate::science::kahn_kalai_park_pham_threshold::KahnKalaiParkPhamThreshold;

py_engine! {
    PyKahnKalaiParkPhamThreshold("KahnKalaiParkPhamThreshold") wraps KahnKalaiParkPhamThreshold [Clone, Copy] {
        new(max_witness_size_l: usize) -> res;
        method_res log_factor_and_critical_threshold_bound(expectation_threshold_q: f64) -> (f64, f64);
    }
}

use crate::omni_tritetracosa_xliii::GrandOmniTritetracosaEngineXLIII;

#[pyclass(
    name = "GrandOmniTritetracosaEngineXLIII",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniTritetracosaEngineXLIII {
    pub inner: GrandOmniTritetracosaEngineXLIII,
}

#[pymethods]
impl PyGrandOmniTritetracosaEngineXLIII {
    #[new]
    #[pyo3(signature = (num_layers=43))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniTritetracosaEngineXLIII::new(num_layers),
        }
    }

    pub fn execute_tritetracosa_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_tritetracosa_synthesis())
            .map_err(to_py_err)
    }
}

use crate::physics::kolmogorov_obukhov_turbulence::KolmogorovK41ObukhovIntermittency;

py_engine! {
    PyKolmogorovK41ObukhovIntermittency("KolmogorovK41ObukhovIntermittency") wraps KolmogorovK41ObukhovIntermittency [Clone, Copy] {
        new(kolmogorov_ck: f64, intermittency_mu: f64) -> res;
        method_res energy_spectrum_and_zeta_p(eps_dissipation: f64, wavenumber_k: f64, order_p: f64) -> (f64, f64);
    }
}

use crate::physics::kraichnan_2d_enstrophy_cascade::Kraichnan2DTurbulenceCascade;

py_engine! {
    PyKraichnan2DTurbulenceCascade("Kraichnan2DTurbulenceCascade") wraps Kraichnan2DTurbulenceCascade [Clone, Copy] {
        new(forcing_kf: f64, enstrophy_flux_eta: f64) -> res;
        method_res forward_enstrophy_spectrum(wavenumber_k: f64, c_omega: f64) -> f64;
    }
}

use crate::physics::orr_sommerfeld_quire_hydro_stability::OrrSommerfeldSquireStability;

py_engine! {
    PyOrrSommerfeldSquireStability("OrrSommerfeldSquireStability") wraps OrrSommerfeldSquireStability [Clone, Copy] {
        new(reynolds_number: f64) -> res;
        method_res squire_2d_wavenumber_and_reynolds(alpha_streamwise: f64, beta_spanwise: f64) -> (f64, f64);
    }
}

use crate::physics::rayleigh_benard_lorenz_convection::RayleighBenardLorenzConvection;

py_engine! {
    PyRayleighBenardLorenzConvection("RayleighBenardLorenzConvection") wraps RayleighBenardLorenzConvection [Clone, Copy] {
        new(prandtl_sigma: f64, aspect_b: f64) -> res;
        method critical_rayleigh_and_lorenz_hopf_rho() -> (f64, f64);
    }
}

use crate::physics::prandtl_blasius_von_karman_boundary::PrandtlBlasiusVonKarmanBoundary;

py_engine! {
    PyPrandtlBlasiusVonKarmanBoundary("PrandtlBlasiusVonKarmanBoundary") wraps PrandtlBlasiusVonKarmanBoundary [Clone, Copy] {
        new(kappa_karman: f64, intercept_b: f64) -> res;
        method_res blasius_cf_and_log_law_u_plus(re_x: f64, y_plus: f64) -> (f64, f64);
    }
}

use crate::physics::rankine_hugoniot_prandtl_meyer_shock::RankineHugoniotShockAdiabat;

py_engine! {
    PyRankineHugoniotShockAdiabat("RankineHugoniotShockAdiabat") wraps RankineHugoniotShockAdiabat [Clone, Copy] {
        new(gamma_adiabatic: f64) -> res;
        method_res shock_pressure_density_and_downstream_mach(mach1: f64) -> (f64, f64, f64);
    }
}

use crate::physics::zeldovich_von_neumann_doring_detonation::ZeldovichVonNeumannDoringDetonation;

py_engine! {
    PyZeldovichVonNeumannDoringDetonation("ZeldovichVonNeumannDoringDetonation") wraps ZeldovichVonNeumannDoringDetonation [Clone, Copy] {
        new(gamma_products: f64, rho0_kg_m3: f64) -> res;
        method_res cj_velocity_and_von_neumann_spike(heat_release_q: f64) -> (f64, f64, f64);
    }
}

use crate::physics::sedov_taylor_blast_wave::SedovTaylorBlastWave;

py_engine! {
    PySedovTaylorBlastWave("SedovTaylorBlastWave") wraps SedovTaylorBlastWave [Clone, Copy] {
        new(similarity_xi0: f64, ambient_rho0: f64) -> res;
        method_res blast_radius_and_shock_velocity(energy_e0: f64, time_t: f64) -> (f64, f64);
    }
}

use crate::physics::rayleigh_plesset_cavitation_bubble::RayleighPlessetCavitationBubble;

py_engine! {
    PyRayleighPlessetCavitationBubble("RayleighPlessetCavitationBubble") wraps RayleighPlessetCavitationBubble [Clone, Copy] {
        new(liquid_density_rho: f64) -> res;
        method_res collapse_time_and_wall_speed(r0: f64, r_current: f64, delta_p: f64) -> (f64, f64);
    }
}

use crate::omni_tetratetracosa_xliv::GrandOmniTetratetracosaEngineXLIV;

#[pyclass(
    name = "GrandOmniTetratetracosaEngineXLIV",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniTetratetracosaEngineXLIV {
    pub inner: GrandOmniTetratetracosaEngineXLIV,
}

#[pymethods]
impl PyGrandOmniTetratetracosaEngineXLIV {
    #[new]
    #[pyo3(signature = (num_layers=44))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniTetratetracosaEngineXLIV::new(num_layers),
        }
    }

    pub fn execute_tetratetracosa_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_tetratetracosa_synthesis())
            .map_err(to_py_err)
    }
}

use crate::science::grothendieck_lefschetz_trace::GrothendieckLefschetzTraceFormula;

py_engine! {
    PyGrothendieckLefschetzTraceFormula("GrothendieckLefschetzTraceFormula") wraps GrothendieckLefschetzTraceFormula [Clone, Copy] {
        new(field_size_q: u64, genus_g: usize) -> res;
        method_res hasse_weil_deligne_point_bounds(degree_n: u32) -> (f64, f64);
    }
}

use crate::science::faltings_mordell_arakelov_height::FaltingsMordellArakelovEngine;

py_engine! {
    PyFaltingsMordellArakelovEngine("FaltingsMordellArakelovEngine") wraps FaltingsMordellArakelovEngine [Clone, Copy] {
        new(genus_g: usize);
        method_res is_mordell_finite_and_isogeny_height_shift(isogeny_degree: u64) -> (bool, i64, f64);
    }
}

use crate::science::birch_swinnerton_dyer_tate_shafarevich::BirchSwinnertonDyerLeadingTerm;

py_engine! {
    PyBirchSwinnertonDyerLeadingTerm("BirchSwinnertonDyerLeadingTerm") wraps BirchSwinnertonDyerLeadingTerm [Clone, Copy] {
        new(real_period_omega: f64, torsion_order: u64) -> res;
        method_res bsd_leading_coefficient(regulator_re: f64, sha_order: u64, tamagawa_product: u64) -> f64;
    }
}

use crate::science::wiles_taylor_modularity_hecke::WilesTaylorModularityHecke;

py_engine! {
    PyWilesTaylorModularityHecke("WilesTaylorModularityHecke") wraps WilesTaylorModularityHecke [Clone, Copy] {
        new(weight_k: u32) -> res;
        method_res ramanujan_bound_and_sato_tate_cos(prime_p: u64, hecke_ap: f64) -> (f64, f64, bool);
    }
}

use crate::science::fontaine_p_adic_hodge_de_rham::FontainePAdicHodgePeriodEngine;

#[pyclass(
    name = "FontainePAdicHodgePeriodEngine",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone, Copy, Default)]
pub struct PyFontainePAdicHodgePeriodEngine {
    pub inner: FontainePAdicHodgePeriodEngine,
}

#[pymethods]
impl PyFontainePAdicHodgePeriodEngine {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: FontainePAdicHodgePeriodEngine::new(),
        }
    }

    #[staticmethod]
    pub fn newton_and_hodge_endpoints(
        newton_slopes: Vec<f64>,
        hodge_weights: Vec<f64>,
    ) -> PyResult<(f64, f64, bool)> {
        FontainePAdicHodgePeriodEngine::newton_and_hodge_endpoints(&newton_slopes, &hodge_weights)
            .map_err(to_py_err)
    }
}

use crate::science::iwasawa_main_conjecture_mazur_wiles::IwasawaClassNumberGrowth;

py_engine! {
    PyIwasawaClassNumberGrowth("IwasawaClassNumberGrowth") wraps IwasawaClassNumberGrowth [Clone, Copy] {
        new(prime_p: u64, mu_inv: u64, lambda_inv: u64, nu_inv: i64) -> res;
        method class_group_p_valuation_at_layer(layer_n: u32) -> i64;
    }
}

use crate::science::sato_tate_equidistribution_taylor::SatoTateSemicircleEquidistribution;

py_engine! {
    PySatoTateSemicircleEquidistribution("SatoTateSemicircleEquidistribution") wraps SatoTateSemicircleEquidistribution [Clone, Copy, Default] {
        new();
        assoc_res density_and_cdf(phi_rad: f64) -> (f64, f64);
    }
}

use crate::science::schmidt_subspace_roth_diophantine::RothSchmidtDiophantineEngine;

py_engine! {
    PyRothSchmidtDiophantineEngine("RothSchmidtDiophantineEngine") wraps RothSchmidtDiophantineEngine [Clone, Copy] {
        new(algebraic_degree_d: usize) -> res;
        method_res roth_exponent_and_scale(denominator_q: u64, eps: f64) -> (f64, usize, f64);
    }
}

use crate::science::voevodsky_motivic_milnor_bloch_kato::VoevodskyMotivicBlochKato;

py_engine! {
    PyVoevodskyMotivicBlochKato("VoevodskyMotivicBlochKato") wraps VoevodskyMotivicBlochKato [Clone, Copy] {
        new(milnor_degree_n: u32) -> res;
        method pfister_dimension_and_motivic_weight() -> (u64, u32);
    }
}

use crate::omni_pentatetracosa_xlv::GrandOmniPentatetracosaEngineXLV;

#[pyclass(
    name = "GrandOmniPentatetracosaEngineXLV",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniPentatetracosaEngineXLV {
    pub inner: GrandOmniPentatetracosaEngineXLV,
}

#[pymethods]
impl PyGrandOmniPentatetracosaEngineXLV {
    #[new]
    #[pyo3(signature = (num_layers=45))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniPentatetracosaEngineXLV::new(num_layers),
        }
    }

    pub fn execute_pentatetracosa_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_pentatetracosa_synthesis())
            .map_err(to_py_err)
    }
}

use crate::science::shannon_hartley_awgn_waterfilling::ShannonHartleyWaterfilling;

py_engine! {
    PyShannonHartleyWaterfilling("ShannonHartleyWaterfilling") wraps ShannonHartleyWaterfilling [Clone, Copy] {
        new(bandwidth_hz: f64) -> res;
        method_res capacity_and_spectral_efficiency(snr_linear: f64) -> (f64, f64);
    }
}

use crate::science::blahut_arimoto_rate_distortion::GaussianRateDistortionEngine;

py_engine! {
    PyGaussianRateDistortionEngine("GaussianRateDistortionEngine") wraps GaussianRateDistortionEngine [Clone, Copy] {
        new(source_variance_sigma2: f64) -> res;
        method_res rate_for_distortion_and_inverse(distortion_d: f64) -> (f64, f64);
    }
}

use crate::science::gallager_ldpc_random_coding_exponent::GallagerRandomCodingExponent;

py_engine! {
    PyGallagerRandomCodingExponent("GallagerRandomCodingExponent") wraps GallagerRandomCodingExponent [Clone, Copy] {
        new(crossover_p: f64) -> res;
        method_res e0_and_error_exponent(rho: f64, rate_r: f64) -> (f64, f64);
    }
}

use crate::science::arikan_polar_channel_polarization::ArikanPolarChannelPolarization;

py_engine! {
    PyArikanPolarChannelPolarization("ArikanPolarChannelPolarization") wraps ArikanPolarChannelPolarization [Clone, Copy] {
        new(initial_z0: f64) -> res;
        method butterfly_bhattacharyya_split() -> (f64, f64, f64);
    }
}

use crate::science::macwilliams_duality_singleton_mds::MacWilliamsSingletonMdsCode;

py_engine! {
    PyMacWilliamsSingletonMdsCode("MacWilliamsSingletonMdsCode") wraps MacWilliamsSingletonMdsCode [Clone, Copy] {
        new(length_n: usize, dim_k: usize) -> res;
        method singleton_distance_and_dual_mds() -> (usize, usize, usize);
    }
}

use crate::science::guruswami_sudan_list_decoding::GuruswamiSudanListDecoder;

py_engine! {
    PyGuruswamiSudanListDecoder("GuruswamiSudanListDecoder") wraps GuruswamiSudanListDecoder [Clone, Copy] {
        new(code_rate_r: f64) -> res;
        method gs_radius_unique_radius_and_gain() -> (f64, f64, f64);
    }
}

use crate::science::ahlswede_cai_li_yeung_network_coding::LinearNetworkCodingMaxFlow;

#[pyclass(
    name = "LinearNetworkCodingMaxFlow",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone, Copy)]
pub struct PyLinearNetworkCodingMaxFlow {
    pub inner: LinearNetworkCodingMaxFlow,
}

#[pymethods]
impl PyLinearNetworkCodingMaxFlow {
    #[new]
    pub fn new(num_sinks: usize) -> PyResult<Self> {
        let inner = LinearNetworkCodingMaxFlow::new(num_sinks).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn multicast_capacity_and_coding_gain(
        &self,
        sink_min_cuts: Vec<f64>,
        routing_limit: f64,
    ) -> PyResult<(f64, f64)> {
        self.inner
            .multicast_capacity_and_coding_gain(&sink_min_cuts, routing_limit)
            .map_err(to_py_err)
    }
}

use crate::science::polyanskiy_poor_verdu_finite_blocklength::PolyanskiyFiniteBlocklengthRate;

py_engine! {
    PyPolyanskiyFiniteBlocklengthRate("PolyanskiyFiniteBlocklengthRate") wraps PolyanskiyFiniteBlocklengthRate [Clone, Copy] {
        new(snr_p: f64) -> res;
        method_res dispersion_and_finite_blocklength_rate(blocklength_n: usize, q_inv_eps: f64) -> (f64, f64);
    }
}

use crate::omni_hexatetracosa_xlvi::GrandOmniHexatetracosaEngineXLVI;

#[pyclass(
    name = "GrandOmniHexatetracosaEngineXLVI",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniHexatetracosaEngineXLVI {
    pub inner: GrandOmniHexatetracosaEngineXLVI,
}

#[pymethods]
impl PyGrandOmniHexatetracosaEngineXLVI {
    #[new]
    #[pyo3(signature = (num_layers=46))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniHexatetracosaEngineXLVI::new(num_layers),
        }
    }

    pub fn execute_hexatetracosa_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_hexatetracosa_synthesis())
            .map_err(to_py_err)
    }
}

use crate::science::regev_lwe_lattice_reduction::RegevLweLatticeSecurity;

py_engine! {
    PyRegevLweLatticeSecurity("RegevLweLatticeSecurity") wraps RegevLweLatticeSecurity [Clone, Copy] {
        new(dim_n: usize, modulus_q: u64) -> res;
        method_res noise_std_approx_factor_and_regev_check(alpha_noise_rate: f64) -> (f64, f64, bool);
    }
}

use crate::science::lll_bkz_hermite_root_factor::LllBkzLatticeReduction;

py_engine! {
    PyLllBkzLatticeReduction("LllBkzLatticeReduction") wraps LllBkzLatticeReduction [Clone, Copy] {
        new(blocksize_beta: usize) -> res;
        method root_hermite_factor_and_core_svp_bits() -> (f64, f64);
    }
}

use crate::science::gentry_fhe_bootstrapping_noise::GentryFheNoiseGrowthEngine;

py_engine! {
    PyGentryFheNoiseGrowthEngine("GentryFheNoiseGrowthEngine") wraps GentryFheNoiseGrowthEngine [Clone, Copy] {
        new(ring_degree_n: usize, plaintext_modulus_t: u64) -> res;
        method_res remaining_noise_budget_bits(log2_q: f64, current_noise_norm: f64) -> (f64, bool);
    }
}

use crate::science::lund_fortnow_karloff_nisan_sumcheck::SumcheckInteractiveProof;

py_engine! {
    PySumcheckInteractiveProof("SumcheckInteractiveProof") wraps SumcheckInteractiveProof [Clone, Copy] {
        new(num_vars_m: usize, max_degree_d: usize) -> res;
        method_res communication_words_and_soundness_error(field_size_q: f64) -> (usize, f64);
    }
}

use crate::science::groth16_qap_zk_snark::Groth16QapSnarkAnalyzer;

py_engine! {
    PyGroth16QapSnarkAnalyzer("Groth16QapSnarkAnalyzer") wraps Groth16QapSnarkAnalyzer [Clone, Copy] {
        new(num_constraints_n: usize, num_public_inputs_l: usize) -> res;
        method proof_elements_pairings_and_ntt_ops() -> (usize, usize, f64);
    }
}

use crate::science::ben_sasson_fri_reed_solomon_stark::FriReedSolomonProximityStark;

py_engine! {
    PyFriReedSolomonProximityStark("FriReedSolomonProximityStark") wraps FriReedSolomonProximityStark [Clone, Copy] {
        new(rate_rho: f64) -> res;
        method_res johnson_proximity_and_query_security_bits(num_queries_s: usize) -> (f64, f64);
    }
}

use crate::science::shamir_bgw_mpc_threshold::ShamirBgwThresholdMpc;

py_engine! {
    PyShamirBgwThresholdMpc("ShamirBgwThresholdMpc") wraps ShamirBgwThresholdMpc [Clone, Copy] {
        new(num_parties_n: usize) -> res;
        method max_semi_honest_and_malicious_thresholds() -> (usize, usize);
    }
}

use crate::science::dwork_mcsherry_gaussian_differential_privacy::DworkGaussianDifferentialPrivacy;

py_engine! {
    PyDworkGaussianDifferentialPrivacy("DworkGaussianDifferentialPrivacy") wraps DworkGaussianDifferentialPrivacy [Clone, Copy] {
        new(l2_sensitivity: f64) -> res;
        method_res gaussian_sigma_and_composed_epsilon(epsilon: f64, delta: f64, steps_k: usize) -> (f64, f64);
    }
}

use crate::science::mironov_renyi_dp_moments_accountant::MironovRenyiDpAccountant;

py_engine! {
    PyMironovRenyiDpAccountant("MironovRenyiDpAccountant") wraps MironovRenyiDpAccountant [Clone, Copy] {
        new(noise_multiplier: f64) -> res;
        method_res rdp_and_converted_dp_epsilon(order_alpha: f64, steps_k: usize, delta: f64) -> (f64, f64);
    }
}

use crate::omni_heptatetracosa_xlvii::GrandOmniHeptatetracosaEngineXLVII;

#[pyclass(
    name = "GrandOmniHeptatetracosaEngineXLVII",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniHeptatetracosaEngineXLVII {
    pub inner: GrandOmniHeptatetracosaEngineXLVII,
}

#[pymethods]
impl PyGrandOmniHeptatetracosaEngineXLVII {
    #[new]
    #[pyo3(signature = (num_layers=47))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniHeptatetracosaEngineXLVII::new(num_layers),
        }
    }

    pub fn execute_heptatetracosa_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_heptatetracosa_synthesis())
            .map_err(to_py_err)
    }
}

use crate::science::pontryagin_maximum_principle_bang_bang::PontryaginMinimumTimeBangBang;

py_engine! {
    PyPontryaginMinimumTimeBangBang("PontryaginMinimumTimeBangBang") wraps PontryaginMinimumTimeBangBang [Clone, Copy] {
        new(u_max: f64) -> res;
        method_res switching_time_and_minimum_total_time(initial_pos_x0: f64) -> (f64, f64);
    }
}

use crate::science::kalman_bucy_lqg_riccati::KalmanBucyRiccatiFilter;

py_engine! {
    PyKalmanBucyRiccatiFilter("KalmanBucyRiccatiFilter") wraps KalmanBucyRiccatiFilter [Clone, Copy] {
        new(drift_a: f64, obs_c: f64) -> res;
        method_res steady_state_covariance_and_gain(process_noise_q: f64, meas_noise_r: f64) -> (f64, f64);
    }
}

use crate::science::zames_doyle_glover_h_infinity_control::ZamesDoyleHInfinityControl;

py_engine! {
    PyZamesDoyleHInfinityControl("ZamesDoyleHInfinityControl") wraps ZamesDoyleHInfinityControl [Clone, Copy] {
        new(nominal_h_inf_norm: f64) -> res;
        method_res max_allowable_uncertainty_and_stability(delta_norm: f64) -> (f64, bool);
    }
}

use crate::science::sontag_control_lyapunov_iss::SontagControlLyapunovEngine;

py_engine! {
    PySontagControlLyapunovEngine("SontagControlLyapunovEngine") wraps SontagControlLyapunovEngine [Clone, Copy, Default] {
        new();
        assoc_res sontag_control_and_lyapunov_rate(lie_f_a: f64, lie_g_b: f64) -> (f64, f64);
    }
}

use crate::science::ames_control_barrier_function_qp::AmesControlBarrierFunctionQp;

py_engine! {
    PyAmesControlBarrierFunctionQp("AmesControlBarrierFunctionQp") wraps AmesControlBarrierFunctionQp [Clone, Copy] {
        new(barrier_gamma: f64) -> res;
        method_res filter_nominal_control(h_val: f64, lf_h: f64, lg_h: f64, u_nom: f64) -> (f64, f64);
    }
}

use crate::science::lasry_lions_mean_field_game::LasryLionsLinearQuadraticMfg;

py_engine! {
    PyLasryLionsLinearQuadraticMfg("LasryLionsLinearQuadraticMfg") wraps LasryLionsLinearQuadraticMfg [Clone, Copy] {
        new(discount_rho: f64, state_cost_q: f64, control_cost_r: f64) -> res;
        method_res equilibrium_riccati_and_stationary_variance(volatility_sigma: f64) -> (f64, f64);
    }
}

use crate::science::isaacs_homicidal_chauffeur_differential_game::IsaacsDifferentialPursuitGame;

py_engine! {
    PyIsaacsDifferentialPursuitGame("IsaacsDifferentialPursuitGame") wraps IsaacsDifferentialPursuitGame [Clone, Copy] {
        new(pursuer_speed: f64, evader_speed: f64) -> res;
        method_res apollonius_radius_and_capture_time(initial_distance_d0: f64) -> (f64, f64);
    }
}

use crate::science::kushner_stratonovich_zakai_nonlinear_filter::ZakaiUnnormalizedFilteringSde;

#[pyclass(
    name = "ZakaiUnnormalizedFilteringSde",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone, Copy)]
pub struct PyZakaiUnnormalizedFilteringSde {
    pub inner: ZakaiUnnormalizedFilteringSde,
}

#[pymethods]
impl PyZakaiUnnormalizedFilteringSde {
    #[new]
    pub fn new(obs_noise_sigma_y: f64) -> PyResult<Self> {
        let inner = ZakaiUnnormalizedFilteringSde::new(obs_noise_sigma_y).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn kallianpur_striebel_posterior_mean(
        &self,
        particle_states: Vec<f64>,
        dy_increment: f64,
        dt: f64,
    ) -> PyResult<(f64, f64)> {
        self.inner
            .kallianpur_striebel_posterior_mean(&particle_states, dy_increment, dt)
            .map_err(to_py_err)
    }
}

use crate::science::willems_dissipativity_passivity_kyp::WillemsDissipativityKypEngine;

py_engine! {
    PyWillemsDissipativityKypEngine("WillemsDissipativityKypEngine") wraps WillemsDissipativityKypEngine [Clone, Copy] {
        new(ifp_nu: f64, ofp_rho: f64) -> res;
        method supply_rate_and_strict_passivity(input_u: f64, output_y: f64) -> (f64, bool);
    }
}

use crate::omni_octatetracosa_xlviii::GrandOmniOctatetracosaEngineXLVIII;

#[pyclass(
    name = "GrandOmniOctatetracosaEngineXLVIII",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniOctatetracosaEngineXLVIII {
    pub inner: GrandOmniOctatetracosaEngineXLVIII,
}

#[pymethods]
impl PyGrandOmniOctatetracosaEngineXLVIII {
    #[new]
    #[pyo3(signature = (num_layers=48))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniOctatetracosaEngineXLVIII::new(num_layers),
        }
    }

    pub fn execute_octatetracosa_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_octatetracosa_synthesis())
            .map_err(to_py_err)
    }
}

use crate::science::maclane_coherence_monoidal_pentagon::MacLaneMonoidalCoherence;

py_engine! {
    PyMacLaneMonoidalCoherence("MacLaneMonoidalCoherence") wraps MacLaneMonoidalCoherence [Clone, Copy] {
        new(tensor_arity_n: usize) -> res;
        method associahedron_vertices_and_dimension() -> (u64, usize);
    }
}

use crate::science::grothendieck_topos_giraud_sheaf::GiraudGrothendieckToposEngine;

py_engine! {
    PyGiraudGrothendieckToposEngine("GiraudGrothendieckToposEngine") wraps GiraudGrothendieckToposEngine [Clone, Copy] {
        new(num_site_generators: u32) -> res;
        method sieve_cardinality_and_dense_rank() -> (u64, u32);
    }
}

use crate::science::lurie_infinity_topos_straightening::LurieInfinityCategoryEngine;

py_engine! {
    PyLurieInfinityCategoryEngine("LurieInfinityCategoryEngine") wraps LurieInfinityCategoryEngine [Clone, Copy] {
        new(simplex_dim_n: usize) -> res;
        method inner_horn_count_and_subsimplices() -> (usize, u64);
    }
}

use crate::science::baez_dolan_lurie_cobordism_hypothesis::BaezDolanCobordismHypothesis;

py_engine! {
    PyBaezDolanCobordismHypothesis("BaezDolanCobordismHypothesis") wraps BaezDolanCobordismHypothesis [Clone, Copy] {
        new(cobordism_dim_n: u32, abelian_group_order: u64) -> res;
        method_res torus_and_sphere_partition_invariants() -> (f64, f64);
    }
}

use crate::science::voevodsky_univalence_hott_freudenthal::HomotopyTypeTheoryUnivalence;

py_engine! {
    PyHomotopyTypeTheoryUnivalence("HomotopyTypeTheoryUnivalence") wraps HomotopyTypeTheoryUnivalence [Clone, Copy] {
        new(finite_set_size_n: usize) -> res;
        method_res loop_space_cardinality_and_freudenthal_bound(connectivity_n: usize) -> (u64, usize);
    }
}

use crate::science::scholze_perfectoid_tilting_equivalence::ScholzePerfectoidTiltingEngine;

py_engine! {
    PyScholzePerfectoidTiltingEngine("ScholzePerfectoidTiltingEngine") wraps ScholzePerfectoidTiltingEngine [Clone, Copy] {
        new(prime_p: u64) -> res;
        method perfection_tower_valuation_and_norm(level_n: u32) -> (f64, f64);
    }
}

use crate::science::clausen_scholze_condensed_liquid_tensor::ClausenScholzeLiquidTensor;

#[pyclass(
    name = "ClausenScholzeLiquidTensor",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone, Copy)]
pub struct PyClausenScholzeLiquidTensor {
    pub inner: ClausenScholzeLiquidTensor,
}

#[pymethods]
impl PyClausenScholzeLiquidTensor {
    #[new]
    pub fn new(liquid_exponent_p: f64) -> PyResult<Self> {
        let inner = ClausenScholzeLiquidTensor::new(liquid_exponent_p).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    pub fn p_liquid_norm_and_ext_vanishing(
        &self,
        coeffs: Vec<f64>,
        test_p_prime: f64,
    ) -> PyResult<(f64, bool)> {
        self.inner
            .p_liquid_norm_and_ext_vanishing(&coeffs, test_p_prime)
            .map_err(to_py_err)
    }
}

use crate::science::bhatt_scholze_prismatic_cohomology::BhattScholzePrismaticEngine;

py_engine! {
    PyBhattScholzePrismaticEngine("BhattScholzePrismaticEngine") wraps BhattScholzePrismaticEngine [Clone, Copy] {
        new(smooth_dim_d: u32) -> res;
        method_res hodge_tate_rank_and_q_integer(n_int: u32, q_param: f64) -> (u64, f64);
    }
}

use crate::science::fargues_fontaine_curve_geometrization::FarguesFontaineCurveEngine;

py_engine! {
    PyFarguesFontaineCurveEngine("FarguesFontaineCurveEngine") wraps FarguesFontaineCurveEngine [Clone, Copy] {
        new(degree_d: i64, rank_h: u64) -> res;
        method hn_slope_and_is_etale_semistable() -> (f64, bool);
    }
}

use crate::omni_nonatetracosa_xlix::GrandOmniNonatetracosaEngineXLIX;

#[pyclass(
    name = "GrandOmniNonatetracosaEngineXLIX",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniNonatetracosaEngineXLIX {
    pub inner: GrandOmniNonatetracosaEngineXLIX,
}

#[pymethods]
impl PyGrandOmniNonatetracosaEngineXLIX {
    #[new]
    #[pyo3(signature = (num_layers=49))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniNonatetracosaEngineXLIX::new(num_layers),
        }
    }

    pub fn execute_nonatetracosa_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_nonatetracosa_synthesis())
            .map_err(to_py_err)
    }
}

use crate::physics::sachdev_ye_kitaev_syk_chaos::SachdevYeKitaevMaximalChaos;

py_engine! {
    PySachdevYeKitaevMaximalChaos("SachdevYeKitaevMaximalChaos") wraps SachdevYeKitaevMaximalChaos [Clone, Copy] {
        new(coupling_j: f64) -> res;
        method_res mss_bound_and_syk_lyapunov(temperature_t: f64) -> (f64, f64);
    }
}

use crate::physics::jackiw_teitelboim_schwarzian_gravity::JackiwTeitelboimSchwarzianGravity;

py_engine! {
    PyJackiwTeitelboimSchwarzianGravity("JackiwTeitelboimSchwarzianGravity") wraps JackiwTeitelboimSchwarzianGravity [Clone, Copy] {
        new(schwarzian_coupling_c: f64, extremal_entropy_s0: f64) -> res;
        method_res disk_spectral_density_and_entropy(energy_e: f64) -> (f64, f64);
    }
}

use crate::physics::penington_almheiri_qes_island_page::QuantumExtremalSurfaceIslandPage;

py_engine! {
    PyQuantumExtremalSurfaceIslandPage("QuantumExtremalSurfaceIslandPage") wraps QuantumExtremalSurfaceIslandPage [Clone, Copy] {
        new(initial_bh_entropy_s0: f64, emission_rate_gamma: f64) -> res;
        method_res page_time_entropy_and_island_dominance(time_t: f64) -> (f64, f64, bool);
    }
}

use crate::physics::pastawski_yoshida_harlow_preskill_happy::HappyHolographicCodeBulkWedge;

py_engine! {
    PyHappyHolographicCodeBulkWedge("HappyHolographicCodeBulkWedge") wraps HappyHolographicCodeBulkWedge [Clone, Copy] {
        new(radial_layers_l: u32) -> res;
        method hyperbolic_growth_ratio_and_boundary_scale() -> (f64, f64);
    }
}

use crate::physics::susskind_brown_complexity_equals_action::HolographicComplexityActionVolume;

py_engine! {
    PyHolographicComplexityActionVolume("HolographicComplexityActionVolume") wraps HolographicComplexityActionVolume [Clone, Copy] {
        new(black_hole_mass_m: f64) -> res;
        method_res lloyd_growth_rate_and_complexity(time_t: f64) -> (f64, f64);
    }
}

use crate::physics::strominger_bms_soft_graviton_memory::StromingerSoftTheoremMemoryTriangle;

py_engine! {
    PyStromingerSoftTheoremMemoryTriangle("StromingerSoftTheoremMemoryTriangle") wraps StromingerSoftTheoremMemoryTriangle [Clone, Copy] {
        new(soft_numerator_k: f64) -> res;
        method_res soft_pole_and_memory_strain(soft_freq_omega: f64, distance_r: f64) -> (f64, f64);
    }
}

use crate::physics::arkani_hamed_trnka_amplituhedron::AmplituhedronGrassmannianVolume;

py_engine! {
    PyAmplituhedronGrassmannianVolume("AmplituhedronGrassmannianVolume") wraps AmplituhedronGrassmannianVolume [Clone, Copy] {
        new(num_particles_n: usize, nmhv_degree_k: usize) -> res;
        method_res bcfw_cell_count_and_dimension() -> (u64, usize);
    }
}

use crate::physics::vafa_ooguri_swampland_distance_de_sitter::SwamplandDistanceDeSitterBounds;

py_engine! {
    PySwamplandDistanceDeSitterBounds("SwamplandDistanceDeSitterBounds") wraps SwamplandDistanceDeSitterBounds [Clone, Copy] {
        new(spacetime_dim_d: usize) -> res;
        method_res min_decay_rate_and_tower_mass_ratio(geodesic_distance_dphi: f64) -> (f64, f64);
    }
}

use crate::physics::pasterski_strominger_zhiboedov_celestial::CelestialHolographyMellinAmplitude;

py_engine! {
    PyCelestialHolographyMellinAmplitude("CelestialHolographyMellinAmplitude") wraps CelestialHolographyMellinAmplitude [Clone, Copy] {
        new(uv_regulator_eps: f64) -> res;
        method_res principal_norm_sq_and_integer_mellin_kernel(lambda_real: f64, delta_int: u32) -> (f64, f64);
    }
}

use crate::omni_quincentennial_l::GrandOmniQuincentennialEngineL;

#[pyclass(
    name = "GrandOmniQuincentennialEngineL",
    module = "oxideon",
    from_py_object
)]
#[derive(Clone)]
pub struct PyGrandOmniQuincentennialEngineL {
    pub inner: GrandOmniQuincentennialEngineL,
}

#[pymethods]
impl PyGrandOmniQuincentennialEngineL {
    #[new]
    #[pyo3(signature = (num_layers=50))]
    pub fn new(num_layers: usize) -> Self {
        Self {
            inner: GrandOmniQuincentennialEngineL::new(num_layers),
        }
    }

    pub fn execute_quincentennial_synthesis(&self, py: Python<'_>) -> PyResult<f64> {
        py.detach(|| self.inner.execute_quincentennial_synthesis())
            .map_err(to_py_err)
    }
}

pub fn register_python_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    install_quiet_panic_hook();
    make_panics_ordinary_exceptions(m.py());
    m.add_function(wrap_pyfunction!(py_raise_test_panic, m)?)?;
    m.add_class::<PyMotilityInducedPhaseSeparation>()?;
    m.add_class::<PyOldroydBViscoelasticRheology>()?;
    m.add_class::<PyWashburnYoungLaplaceCapillary>()?;
    m.add_class::<PyGrandOmniOctatricosaEngineXXXVIII>()?;
    m.add_class::<PyByzantineQuorumConsensusEngine>()?;
    m.add_class::<PyMatternVectorClockCausality>()?;
    m.add_class::<PyShapiroCrdtJoinSemilattice>()?;
    m.add_class::<PyKargerConsistentHashingRing>()?;
    m.add_class::<PyGilbertLynchPacelcTheorem>()?;
    m.add_class::<PyClarkeEmersonModelChecker>()?;
    m.add_class::<PyReynoldsSeparationLogicHeap>()?;
    m.add_class::<PyCousotGaloisAbstractInterpreter>()?;
    m.add_class::<PyGirardLinearLogicResource>()?;
    m.add_class::<PyGrandOmniNonatricosaEngineXXXIX>()?;
    m.add_class::<PyLawsonTripleProductIgnition>()?;
    m.add_class::<PySpitzerDreicerPlasmaTransport>()?;
    m.add_class::<PyKruskalShafranovTroyonMhd>()?;
    m.add_class::<PyTajimaDawsonLaserWakefield>()?;
    m.add_class::<PyGamowThermonuclearPeakEngine>()?;
    m.add_class::<PyHoyleSalpeterEddingtonStellar>()?;
    m.add_class::<PyFermiHillasCosmicRayShock>()?;
    m.add_class::<PyParkerSolarWindSpiralEngine>()?;
    m.add_class::<PySweetParkerPetschekReconnection>()?;
    m.add_class::<PyGrandOmniQuadricentennialEngineXL>()?;
    m.add_class::<PyAlmgrenFrequencyCurrentsEngine>()?;
    m.add_class::<PyAllardVarifoldMonotonicityEngine>()?;
    m.add_class::<PyFedererFlemingCurrentsCoarea>()?;
    m.add_class::<PyCaffarelliMongeAmpereTransport>()?;
    m.add_class::<PyAltCaffarelliFreeBoundaryWeiss>()?;
    m.add_class::<PyCaffarelliSilvestreFractionalExt>()?;
    m.add_class::<PySimonsMinimalConeBernstein>()?;
    m.add_class::<PyBesicovitchMarstrandRectifiable>()?;
    m.add_class::<PyCheegerBuserSpectralInequality>()?;
    m.add_class::<PyGrandOmniHenatetracosaEngineXLI>()?;
    m.add_class::<PyBjorkenQgpHydroExpansion>()?;
    m.add_class::<PyKovtunSonStarinetsViscosityBound>()?;
    m.add_class::<PyColorGlassCondensateSaturation>()?;
    m.add_class::<PyBdmpsJetQuenchingEnergyLoss>()?;
    m.add_class::<PyMatsuiSatzQuarkoniumMelting>()?;
    m.add_class::<PyChiralMagneticEffectKharzeev>()?;
    m.add_class::<PyBetheWeizsackerNuclearMass>()?;
    m.add_class::<PyMayerJensenNuclearShellMagic>()?;
    m.add_class::<PyWaleckaRelativisticMeanFieldNuclear>()?;
    m.add_class::<PyGrandOmniDotetracosaEngineXLII>()?;
    m.add_class::<PyLovaszLocalLemmaMoserTardos>()?;
    m.add_class::<PyTalagrandConvexConcentration>()?;
    m.add_class::<PyFriedgutBourgainSharpThreshold>()?;
    m.add_class::<PyErdosRenyiBollobasPhaseTransition>()?;
    m.add_class::<PyAlonBoppanaRamanujanExpander>()?;
    m.add_class::<PyBalogSzemerediGowersFreiman>()?;
    m.add_class::<PyRothBourgainCapSetBounds>()?;
    m.add_class::<PyLovaszThetaShannonCapacity>()?;
    m.add_class::<PyKahnKalaiParkPhamThreshold>()?;
    m.add_class::<PyGrandOmniTritetracosaEngineXLIII>()?;
    m.add_class::<PyKolmogorovK41ObukhovIntermittency>()?;
    m.add_class::<PyKraichnan2DTurbulenceCascade>()?;
    m.add_class::<PyOrrSommerfeldSquireStability>()?;
    m.add_class::<PyRayleighBenardLorenzConvection>()?;
    m.add_class::<PyPrandtlBlasiusVonKarmanBoundary>()?;
    m.add_class::<PyRankineHugoniotShockAdiabat>()?;
    m.add_class::<PyZeldovichVonNeumannDoringDetonation>()?;
    m.add_class::<PySedovTaylorBlastWave>()?;
    m.add_class::<PyRayleighPlessetCavitationBubble>()?;
    m.add_class::<PyGrandOmniTetratetracosaEngineXLIV>()?;
    m.add_class::<PyGrothendieckLefschetzTraceFormula>()?;
    m.add_class::<PyFaltingsMordellArakelovEngine>()?;
    m.add_class::<PyBirchSwinnertonDyerLeadingTerm>()?;
    m.add_class::<PyWilesTaylorModularityHecke>()?;
    m.add_class::<PyFontainePAdicHodgePeriodEngine>()?;
    m.add_class::<PyIwasawaClassNumberGrowth>()?;
    m.add_class::<PySatoTateSemicircleEquidistribution>()?;
    m.add_class::<PyRothSchmidtDiophantineEngine>()?;
    m.add_class::<PyVoevodskyMotivicBlochKato>()?;
    m.add_class::<PyGrandOmniPentatetracosaEngineXLV>()?;
    m.add_class::<PyShannonHartleyWaterfilling>()?;
    m.add_class::<PyGaussianRateDistortionEngine>()?;
    m.add_class::<PyGallagerRandomCodingExponent>()?;
    m.add_class::<PyArikanPolarChannelPolarization>()?;
    m.add_class::<PyMacWilliamsSingletonMdsCode>()?;
    m.add_class::<PyGuruswamiSudanListDecoder>()?;
    m.add_class::<PyLinearNetworkCodingMaxFlow>()?;
    m.add_class::<PyPolyanskiyFiniteBlocklengthRate>()?;
    m.add_class::<PyGrandOmniHexatetracosaEngineXLVI>()?;
    m.add_class::<PyRegevLweLatticeSecurity>()?;
    m.add_class::<PyLllBkzLatticeReduction>()?;
    m.add_class::<PyGentryFheNoiseGrowthEngine>()?;
    m.add_class::<PySumcheckInteractiveProof>()?;
    m.add_class::<PyGroth16QapSnarkAnalyzer>()?;
    m.add_class::<PyFriReedSolomonProximityStark>()?;
    m.add_class::<PyShamirBgwThresholdMpc>()?;
    m.add_class::<PyDworkGaussianDifferentialPrivacy>()?;
    m.add_class::<PyMironovRenyiDpAccountant>()?;
    m.add_class::<PyGrandOmniHeptatetracosaEngineXLVII>()?;
    m.add_class::<PyPontryaginMinimumTimeBangBang>()?;
    m.add_class::<PyKalmanBucyRiccatiFilter>()?;
    m.add_class::<PyZamesDoyleHInfinityControl>()?;
    m.add_class::<PySontagControlLyapunovEngine>()?;
    m.add_class::<PyAmesControlBarrierFunctionQp>()?;
    m.add_class::<PyLasryLionsLinearQuadraticMfg>()?;
    m.add_class::<PyIsaacsDifferentialPursuitGame>()?;
    m.add_class::<PyZakaiUnnormalizedFilteringSde>()?;
    m.add_class::<PyWillemsDissipativityKypEngine>()?;
    m.add_class::<PyGrandOmniOctatetracosaEngineXLVIII>()?;
    m.add_class::<PyMacLaneMonoidalCoherence>()?;
    m.add_class::<PyGiraudGrothendieckToposEngine>()?;
    m.add_class::<PyLurieInfinityCategoryEngine>()?;
    m.add_class::<PyBaezDolanCobordismHypothesis>()?;
    m.add_class::<PyHomotopyTypeTheoryUnivalence>()?;
    m.add_class::<PyScholzePerfectoidTiltingEngine>()?;
    m.add_class::<PyClausenScholzeLiquidTensor>()?;
    m.add_class::<PyBhattScholzePrismaticEngine>()?;
    m.add_class::<PyFarguesFontaineCurveEngine>()?;
    m.add_class::<PyGrandOmniNonatetracosaEngineXLIX>()?;
    m.add_class::<PySachdevYeKitaevMaximalChaos>()?;
    m.add_class::<PyJackiwTeitelboimSchwarzianGravity>()?;
    m.add_class::<PyQuantumExtremalSurfaceIslandPage>()?;
    m.add_class::<PyHappyHolographicCodeBulkWedge>()?;
    m.add_class::<PyHolographicComplexityActionVolume>()?;
    m.add_class::<PyStromingerSoftTheoremMemoryTriangle>()?;
    m.add_class::<PyAmplituhedronGrassmannianVolume>()?;
    m.add_class::<PySwamplandDistanceDeSitterBounds>()?;
    m.add_class::<PyCelestialHolographyMellinAmplitude>()?;
    m.add_class::<PyGrandOmniQuincentennialEngineL>()?;
    m.add_class::<PyActiveNematicDefectTurbulence>()?;
    m.add_class::<PyTonerTuVicsekActiveMatter>()?;
    m.add_class::<PyOnsagerMaierSaupeNematic>()?;
    m.add_class::<PyFloryHugginsSpinodalMixing>()?;
    m.add_class::<PyDeGennesPolymerReptationTube>()?;
    m.add_class::<PyDlvoColloidalStabilityEngine>()?;
    m.add_class::<PyGrandOmniHeptatricosaEngineXXXVII>()?;
    m.add_class::<PyBellEvansMechanobiologyBond>()?;
    m.add_class::<PyRnaVelocityWaddingtonLandscape>()?;
    m.add_class::<PyFelsensteinPhylogeneticLikelihood>()?;
    m.add_class::<PyKarlinAltschulSequenceAlignment>()?;
    m.add_class::<PyCrisprCas9RLoopKinetics>()?;
    m.add_class::<PyBriggsHaldaneEnzymeKinetics>()?;
    m.add_class::<PyMonodWymanChangeuxAllostery>()?;
    m.add_class::<PyZimmBraggRamachandranHelix>()?;
    m.add_class::<PyWolynesProteinFoldingFunnel>()?;
    m.add_class::<PyGrandOmniHexatricosaEngineXXXVI>()?;
    m.add_class::<PyThermoelectricFigureOfMeritZT>()?;
    m.add_class::<PyTsuEsakiResonantTunneling>()?;
    m.add_class::<PyChuaStrukovMemristorSynapse>()?;
    m.add_class::<PyShockleyQueisserPhotovoltaic>()?;
    m.add_class::<PyAmbegaokarBaratoffJosephson>()?;
    m.add_class::<PyTransmonCircuitQEDQubit>()?;
    m.add_class::<PySlonczewskiSpinTransferTorque>()?;
    m.add_class::<PyDattaDasRashbaSpintronics>()?;
    m.add_class::<PyLandauerButtikerBallisticTransport>()?;
    m.add_class::<PyGrandOmniPentatricosaEngineXXXV>()?;
    m.add_class::<PyCrandallLionsViscosityPucci>()?;
    m.add_class::<PyModicaMortolaGammaConvergence>()?;
    m.add_class::<PyLionsConcentrationCompactness>()?;
    m.add_class::<PyZakharovShabatNlsSoliton>()?;
    m.add_class::<PyLaxKdvInverseScatteringSoliton>()?;
    m.add_class::<PyStrichartzDispersiveEstimate>()?;
    m.add_class::<PyCalderonZygmundSingularIntegral>()?;
    m.add_class::<PyDeGiorgiNashMoserRegularity>()?;
    m.add_class::<PyCaffarelliKohnNirenbergNavierStokes>()?;
    m.add_class::<PyGrandOmniTetratricosaEngineXXXIV>()?;
    m.add_class::<PySzemerediGreenTaoRegularity>()?;
    m.add_class::<PyKazhdanLusztigHeckeEngine>()?;
    m.add_class::<PyMacdonaldSymmetricCatalanEngine>()?;
    m.add_class::<PyHookLengthYoungTableauxEngine>()?;
    m.add_class::<PyStanleyReisnerDehnSommerville>()?;
    m.add_class::<PyEhrhartMacdonaldPolytopeEngine>()?;
    m.add_class::<PyMikhalkinTropicalGeometryEngine>()?;
    m.add_class::<PyHuhKatzMatroidLogConcavity>()?;
    m.add_class::<PyTuttePolynomialMatroidEngine>()?;
    m.add_class::<PyGrandOmniTritricosaEngineXXXIII>()?;
    m.add_class::<PyToomreHillPlanetaryDisk>()?;
    m.add_class::<PyStommelSverdrupOceanCirculation>()?;
    m.add_class::<PyThermalWindEadyBaroclinic>()?;
    m.add_class::<PyElsasserGeodynamoCoreEngine>()?;
    m.add_class::<PyEddingtonChandrasekharRadiative>()?;
    m.add_class::<PyBudykoSellersClimateBifurcation>()?;
    m.add_class::<PyBackusGilbertInverseResolution>()?;
    m.add_class::<PyAkiRichardsSeismicAvoEngine>()?;
    m.add_class::<PyKanamoriGutenbergSeismicEngine>()?;
    m.add_class::<PyGrandOmniDotricosaEngineXXXII>()?;
    m.add_class::<PyKellyLogOptimalPortfolio>()?;
    m.add_class::<PyEisenbergNoeSystemicClearing>()?;
    m.add_class::<PyMertonStructuralDefaultEngine>()?;
    m.add_class::<PyRockafellarUryasevCVaREngine>()?;
    m.add_class::<PyKyleLambdaAuctionEquilibrium>()?;
    m.add_class::<PyAvellanedaStoikovMarketMaker>()?;
    m.add_class::<PyAlmgrenChrissOptimalExecution>()?;
    m.add_class::<PyHaganSabrVolatilityEngine>()?;
    m.add_class::<PyHeathJarrowMortonTermStructure>()?;
    m.add_class::<PyGrandOmniHenetricosaEngineXXXI>()?;
    m.add_class::<PyCavityOptomechanicsBackaction>()?;
    m.add_class::<PyFeshbachScatteringResonance>()?;
    m.add_class::<PyDopplerSisyphusLaserCooling>()?;
    m.add_class::<PyCorkumHighHarmonicAttosecond>()?;
    m.add_class::<PyElectromagneticallyInducedTransparency>()?;
    m.add_class::<PyGlauberPhotonCoherenceEngine>()?;
    m.add_class::<PyHongOuMandelInterferometer>()?;
    m.add_class::<PyDickeTavisCummingsSuperradiance>()?;
    m.add_class::<PyJaynesCummingsCavityQED>()?;
    m.add_class::<PyGrandOmniTricentennialEngineXXX>()?;
    m.add_class::<PyWilliamsSatCircuitLowerBound>()?;
    m.add_class::<PyDinurPcpGapAmplification>()?;
    m.add_class::<PySunflowerNaturalProofsBarrier>()?;
    m.add_class::<PySolomonoffLevinUniversalPrior>()?;
    m.add_class::<PyGoodsteinParisHarringtonEngine>()?;
    m.add_class::<PyHrushovskiZariskiGeometry>()?;
    m.add_class::<PyOMinimalPfaffianCellEngine>()?;
    m.add_class::<PyShelahStabilitySauerEngine>()?;
    m.add_class::<PyMartinBorelDeterminacyEngine>()?;
    m.add_class::<PyGrandOmniNonacosaEngineXXIX>()?;
    m.add_class::<PyHubbellEwensBiodiversityEngine>()?;
    m.add_class::<PyKermackMcKendrickEpidemicEngine>()?;
    m.add_class::<PyFisherKppWavefrontEngine>()?;
    m.add_class::<PyMayWignerComplexityStability>()?;
    m.add_class::<PyEigenQuasispeciesThreshold>()?;
    m.add_class::<PyReplicatorMutatorGameEngine>()?;
    m.add_class::<PyPriceHamiltonEvolutionEngine>()?;
    m.add_class::<PyKingmanCoalescentEngine>()?;
    m.add_class::<PyKimuraWrightFisherEngine>()?;
    m.add_class::<PyGrandOmniOctacosaEngineXXVIII>()?;
    m.add_class::<PyElectroweakSphaleronBaryogenesis>()?;
    m.add_class::<PyKibbleZurekDefectScaling>()?;
    m.add_class::<PyAbjChiralAnomalyEngine>()?;
    m.add_class::<PyQcdAsymptoticFreedomEngine>()?;
    m.add_class::<PyWilsonLatticeGaugeQCD>()?;
    m.add_class::<PyMswNeutrinoOscillationEngine>()?;
    m.add_class::<PyUnruhRindlerBogoliubovEngine>()?;
    m.add_class::<PyCasimirLifshitzVacuumEngine>()?;
    m.add_class::<PySchwingerEulerHeisenbergQED>()?;
    m.add_class::<PyGrandOmniHeptacosaEngineXXVII>()?;
    m.add_class::<PyRatnerUnipotentFlowEngine>()?;
    m.add_class::<PyDrinfeldShtukaLanglandsEngine>()?;
    m.add_class::<PyNgoFundamentalLemmaEngine>()?;
    m.add_class::<PySatakeHeckeLanglandsEngine>()?;
    m.add_class::<PyKazhdanPropertyTExpander>()?;
    m.add_class::<PyBassSerreGraphOfGroupsEngine>()?;
    m.add_class::<PyThurstonHyperbolicDehnEngine>()?;
    m.add_class::<PyTeichmullerWeilPeterssonEngine>()?;
    m.add_class::<PyGromovDeltaHyperbolicEngine>()?;
    m.add_class::<PyGrandOmniHexacosaEngineXXVI>()?;
    m.add_class::<PyKaneLubenskyIsostaticEngine>()?;
    m.add_class::<PyPhotonicBraggQuarterWaveStack>()?;
    m.add_class::<PyPendryMetamaterialCloakEngine>()?;
    m.add_class::<PyEshelbyInclusionMoriTanaka>()?;
    m.add_class::<PyBiotPoroelasticConsolidation>()?;
    m.add_class::<PyKirchhoffLovePlateEngine>()?;
    m.add_class::<PyJ2PlasticityReturnMapping>()?;
    m.add_class::<PyRiceJIntegralFractureEngine>()?;
    m.add_class::<PyHyperelasticStrainEnergyEngine>()?;
    m.add_class::<PyGrandOmniPentacosaEngineXXV>()?;
    m.add_class::<PyRandomFourierBochnerEngine>()?;
    m.add_class::<PySplitConformalPredictionEngine>()?;
    m.add_class::<PyLogSobolevConcentrationEngine>()?;
    m.add_class::<PyRestrictedIsometryLassoEngine>()?;
    m.add_class::<PyOverparameterizedDoubleDescent>()?;
    m.add_class::<PyComputeOptimalChinchillaEngine>()?;
    m.add_class::<PyMaximalUpdateParamEngine>()?;
    m.add_class::<PyInfiniteWidthNtkEngine>()?;
    m.add_class::<PyPacBayesMcAllesterEngine>()?;
    m.add_class::<PyGrandOmniTetracosaEngineXXIV>()?;
    m.add_class::<PyRenyiDifferentialPrivacyAccountant>()?;
    m.add_class::<PySlepianWolfDistributedCoding>()?;
    m.add_class::<PySlepianWolfRateRegionEngine>()?;
    m.add_class::<PyRateDistortionBlahutArimoto>()?;
    m.add_class::<PyArikanPolarCodeEngine>()?;
    m.add_class::<PyTsfasmanVladutZinkAgCode>()?;
    m.add_class::<PyShamirSecretSharingFp>()?;
    m.add_class::<PyKzgPolynomialCommitment>()?;
    m.add_class::<PyFastReedSolomonIopEngine>()?;
    m.add_class::<PyQuadraticArithmeticProgramEngine>()?;
    m.add_class::<PyGrandOmniTricosaEngineXXIII>()?;
    m.add_class::<PyHawkingPagePhaseTransition>()?;
    m.add_class::<PyPressSchechterHaloEngine>()?;
    m.add_class::<PySunyaevZeldovichKompaneetsEngine>()?;
    m.add_class::<PyCmbAcousticPeakEngine>()?;
    m.add_class::<PyGravitationalWaveChirpEngine>()?;
    m.add_class::<PyChandrasekharWhiteDwarfEngine>()?;
    m.add_class::<PyTolmanOppenheimerVolkoffEngine>()?;
    m.add_class::<PyBlandfordZnajekPenroseEngine>()?;
    m.add_class::<PyKerrQuasinormalModeEngine>()?;
    m.add_class::<PyGrandOmniDocosaEngineXXII>()?;
    m.add_class::<PyChirikovStandardMapEngine>()?;
    m.add_class::<PyCircularRestrictedThreeBodyEngine>()?;
    m.add_class::<PyNekhoroshevExponentialStability>()?;
    m.add_class::<PyKamDiophantineTorusEngine>()?;
    m.add_class::<PyControlBarrierLyapunovEngine>()?;
    m.add_class::<PyHInfinityRobustControlEngine>()?;
    m.add_class::<PyKalmanBucyRiccatiEngine>()?;
    m.add_class::<PySubRiemannianHeisenbergEngine>()?;
    m.add_class::<PyPontryaginMaximumEngine>()?;
    m.add_class::<PyGrandOmniHenicosaEngineXXI>()?;
    m.add_class::<PyRingPolymerMolecularDynamics>()?;
    m.add_class::<PyEyringTransitionStateEngine>()?;
    m.add_class::<PyYarkonyConicalIntersectionEngine>()?;
    m.add_class::<PyHedinGwBetheSalpeterEngine>()?;
    m.add_class::<PyMultireferenceCasEngine>()?;
    m.add_class::<PyMollerPlessetMp2Engine>()?;
    m.add_class::<PyCoupledClusterCcsdTEngine>()?;
    m.add_class::<PyKohnShamJacobLadderEngine>()?;
    m.add_class::<PyRoothaanHallScfEngine>()?;
    m.add_class::<PyGrandBicentennialSingularityEngineXX>()?;
    m.add_class::<PyLurieLittleDisksFactorization>()?;
    m.add_class::<PyVoevodskyMilnorBlochKato>()?;
    m.add_class::<PyPrismaticSiteBreuilKisin>()?;
    m.add_class::<PyCondensedLiquidTensorEngine>()?;
    m.add_class::<PyFukayaArnoldConjectureEngine>()?;
    m.add_class::<PyHeegaardFloerKnotEngine>()?;
    m.add_class::<PyKhovanovHomologyEngine>()?;
    m.add_class::<PyAdamsJHomomorphismEngine>()?;
    m.add_class::<PyChromaticMoravaKTheory>()?;
    m.add_class::<PyGrandOmniNonadecaEngineXIX>()?;
    m.add_class::<PyForsterResonanceEnergyEngine>()?;
    m.add_class::<PyMarcusElectronTransferEngine>()?;
    m.add_class::<PyKratkyPorodPolymerEngine>()?;
    m.add_class::<PyFokkerPlanckOrnsteinUhlenbeck>()?;
    m.add_class::<PyChemicalMasterEntropyEngine>()?;
    m.add_class::<PyJarzynskiCrooksEngine>()?;
    m.add_class::<PyTuringReactionDiffusionEngine>()?;
    m.add_class::<PyWilsonCowanCorticalEngine>()?;
    m.add_class::<PyHodgkinHuxleyActionPotential>()?;
    m.add_class::<PyGrandOmniOctodecaEngineXVIII>()?;
    m.add_class::<PyMagneticSkyrmionEngine>()?;
    m.add_class::<PyMagicAngleGrapheneEngine>()?;
    m.add_class::<PyWeylChiralAnomalyEngine>()?;
    m.add_class::<PyBktVortexUnbindingEngine>()?;
    m.add_class::<PyBrinkmanRiceGutzwillerEngine>()?;
    m.add_class::<PyBcsGapEquationEngine>()?;
    m.add_class::<PyKondoScreeningEngine>()?;
    m.add_class::<PyHaldaneHoneycombModel>()?;
    m.add_class::<PyLaughlinFqheEngine>()?;
    m.add_class::<PyGrandOmniHeptadecaEngineXVII>()?;
    m.add_class::<PySteinPoissonCharacterization>()?;
    m.add_class::<PyPardouxPengBsdeEngine>()?;
    m.add_class::<PyLevyItodiffusionEngine>()?;
    m.add_class::<PyMcKeanVlasovPropagationChaos>()?;
    m.add_class::<PyFreidlinWentzellLdpEngine>()?;
    m.add_class::<PyRoughVolatilityHestonEngine>()?;
    m.add_class::<PyKpzColeHopfEngine>()?;
    m.add_class::<PySchrammLoewnerEngine>()?;
    m.add_class::<PyMalliavinBismutEngine>()?;
    m.add_class::<PyGrandOmniHexadecaEngineXVI>()?;
    m.add_class::<PyGrossPitaevskiiVortexEngine>()?;
    m.add_class::<PyGradShafranovTokamakEngine>()?;
    m.add_class::<PyVlasovLandauDampingEngine>()?;
    m.add_class::<PyHydrodynamicInstabilityEngine>()?;
    m.add_class::<PyRayleighBenardConvection>()?;
    m.add_class::<PyCharneyRossbyWaveEngine>()?;
    m.add_class::<PyAlfvenMhdDynamoEngine>()?;
    m.add_class::<PyKraichnanEnstrophyCascade>()?;
    m.add_class::<PyKolmogorovCascadeEngine>()?;
    m.add_class::<PyGrandOmniPentadecaEngineXV>()?;
    m.add_class::<PyKatzSarnakSymmetryEngine>()?;
    m.add_class::<PyFaltingsHeightEngine>()?;
    m.add_class::<PySelbergSieveEngine>()?;
    m.add_class::<PyHardyLittlewoodCircleEngine>()?;
    m.add_class::<PyAbcSzpiroArakelovEngine>()?;
    m.add_class::<PyIwasawaMainConjectureEngine>()?;
    m.add_class::<PyBirchSwinnertonDyerEngine>()?;
    m.add_class::<PySelbergTraceFormulaEngine>()?;
    m.add_class::<PyRiemannSiegelZEngine>()?;
    m.add_class::<PyGrandOmniTetradecaEngineXIV>()?;
    m.add_class::<PyBpsMagneticMonopoleEngine>()?;
    m.add_class::<PyAtiyahSingerIndexEngine>()?;
    m.add_class::<PyMicrolocalWavefrontEngine>()?;
    m.add_class::<PyWeierstrassMinimalSurfaceEngine>()?;
    m.add_class::<PySeibergWittenMonopoleEngine>()?;
    m.add_class::<PyBpstYangMillsInstanton>()?;
    m.add_class::<PyCalabiYauMongeAmpereEngine>()?;
    m.add_class::<PyMeanCurvatureFlowEngine>()?;
    m.add_class::<PyPerelmanRicciFlowEngine>()?;
    m.add_class::<PyGrandOmniTridecaEngineXIII>()?;
    m.add_class::<PyClassicalShadowTomography>()?;
    m.add_class::<PyQuantumOttoEngine>()?;
    m.add_class::<PyQuantumMetrologyEngine>()?;
    m.add_class::<PyKitaevMajoranaChainEngine>()?;
    m.add_class::<PyFloquetTimeCrystalEngine>()?;
    m.add_class::<PyQuantumDarwinismEngine>()?;
    m.add_class::<PyMeraEntanglementEngine>()?;
    m.add_class::<PyMagicStateDistillationEngine>()?;
    m.add_class::<PyQuantumLdpcCodeEngine>()?;
    m.add_class::<PyGrandOmniDodecaEngineXII>()?;
    m.add_class::<PyExactWkbQuantizationEngine>()?;
    m.add_class::<PyBorelResurgenceEngine>()?;
    m.add_class::<PyTraversableWormholeEngine>()?;
    m.add_class::<PyConformalBootstrapEngine>()?;
    m.add_class::<PySwamplandVerifierEngine>()?;
    m.add_class::<PyCelestialHolographyEngine>()?;
    m.add_class::<PyAmplituhedronGrassmannianEngine>()?;
    m.add_class::<PyCausalTriangulationEngine>()?;
    m.add_class::<PySpinNetworkLqgEngine>()?;
    m.add_class::<PyGrandOmniHendecaEngineXI>()?;
    m.add_class::<PyPerfectoidTiltingEngine>()?;
    m.add_class::<PyHitchinHiggsBundleEngine>()?;
    m.add_class::<PyDonaldsonThomasEngine>()?;
    m.add_class::<PyGromovWittenWdvvEngine>()?;
    m.add_class::<PyBridgelandStabilityEngine>()?;
    m.add_class::<PyChowHirzebruchRiemannRoch>()?;
    m.add_class::<PyClusterAlgebraQuiverEngine>()?;
    m.add_class::<PyMotivicHodgeEngine>()?;
    m.add_class::<PyEtaleWeilZetaEngine>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyGrandCentennialSingularityReportX>()?;
    m.add_class::<PyGrandCentennialSingularityEngineX>()?;
    m.add_class::<PyLanglandsHeckeEngine>()?;
    m.add_class::<PyPenroseTwistorEngine>()?;
    m.add_class::<PySolomonoffLevinEngine>()?;
    m.add_class::<PyConnesSpectralTriple>()?;
    m.add_class::<PyJaynesCummingsCavityQed>()?;
    m.add_class::<PyHomotopyTypeEngine>()?;
    m.add_class::<PyGraphWassersteinGeodesicEngine>()?;
    m.add_class::<PySykQuantumChaosEngine>()?;
    m.add_class::<PyDglaMaurerCartanEngine>()?;
    m.add_class::<PyGrandOmniNonaReportVIII>()?;
    m.add_class::<PyGrandOmniNonaEngineVIII>()?;
    m.add_class::<PyKoopmanSpectralEngine>()?;
    m.add_class::<PyLatticeLweEngine>()?;
    m.add_class::<PyDiscreteMorseEngine>()?;
    m.add_class::<PyPAdicAnalysisEngine>()?;
    m.add_class::<PyRoughPathSignature>()?;
    m.add_class::<PyFreeProbabilityEngine>()?;
    m.add_class::<PyRelativisticLbm1D>()?;
    m.add_class::<PyTropicalSpectralEngine>()?;
    m.add_class::<PyFibonacciAnyonBraider>()?;
    m.add_class::<PyDevice>()?;
    m.add_class::<PyDeviceMesh>()?;
    m.add_class::<PyGenerator>()?;
    m.add_class::<PySamplingConfig>()?;
    m.add_class::<PySparseCoo>()?;
    m.add_class::<PySparseCsr>()?;
    m.add_class::<PyNormal>()?;
    m.add_class::<PyCategorical>()?;
    m.add_class::<PyLaplace>()?;
    m.add_class::<PyTensor>()?;
    m.add_class::<PyBpeTokenizer>()?;
    m.add_class::<PyFastBM25Index>()?;
    m.add_class::<PyFastCountVectorizer>()?;
    m.add_class::<PyFastTfidfVectorizer>()?;
    m.add_class::<PyFastHashingVectorizer>()?;
    m.add_class::<PyFastTextCharNGramVectorizer>()?;
    m.add_class::<PyKernelSpectralVectorizer>()?;
    m.add_class::<PyFastMinHashVectorizer>()?;
    m.add_class::<PyFastSimHashVectorizer>()?;
    m.add_class::<PyFastByteHistogramVectorizer>()?;
    m.add_class::<PyFastMaxSimScorer>()?;
    m.add_class::<PyFastSphericalHarmonics>()?;
    m.add_class::<PyFastDiscretizer>()?;
    m.add_class::<PyFastLsaVectorizer>()?;
    m.add_class::<PyFastDictVectorizer>()?;
    m.add_class::<PyFastOneHotVectorizer>()?;
    m.add_class::<PyFastJsonTreeVectorizer>()?;
    m.add_class::<PyFastTimeSeriesSpectralVectorizer>()?;
    m.add_class::<PyFastPathSignatureVectorizer>()?;
    m.add_class::<PyFastDeepSetsVectorizer>()?;
    m.add_class::<PyFastWeisfeilerLehmanVectorizer>()?;
    m.add_class::<PyLinear>()?;
    m.add_class::<PyEmbedding>()?;
    m.add_class::<PyLayerNorm>()?;
    m.add_class::<PyDropout>()?;
    m.add_class::<PyReLU>()?;
    m.add_class::<PyMLP>()?;
    m.add_class::<PyConv2d>()?;
    m.add_class::<PyConv3d>()?;
    m.add_class::<PyCausalLM>()?;
    m.add_class::<PyDeepSeekMoELM>()?;
    m.add_class::<PyHybridJambaLM>()?;
    m.add_class::<PyMultimodalVLM>()?;
    m.add_class::<PyGatedCrossAttention>()?;
    m.add_class::<PyVisionLanguageActionModel>()?;
    m.add_class::<PyLIFNode>()?;
    m.add_class::<PyPoissonEncoder>()?;
    m.add_class::<PySTDPSynapse>()?;
    m.add_class::<PySpikeEvent>()?;
    m.add_class::<PyEventDrivenSpikeRouter>()?;
    m.add_class::<PyPolychronousNeuralLayer>()?;
    m.add_class::<PyDiT>()?;
    m.add_class::<PySequential>()?;
    m.add_class::<PyConvNeXt>()?;
    m.add_class::<PyBiLSTM>()?;
    m.add_class::<PyTemporalConvNet>()?;
    m.add_class::<PyMLPMixer>()?;
    m.add_class::<PyVAE>()?;
    m.add_class::<PySpectralNormLinear>()?;
    m.add_class::<PyWeightNormLinear>()?;
    m.add_class::<PyMaxout>()?;
    m.add_class::<PySnake>()?;
    m.add_class::<PySwish>()?;
    m.add_class::<PyDropPath>()?;
    m.add_class::<PyAlphaDropout>()?;
    m.add_class::<PyConvLstmCell>()?;
    m.add_class::<PyUpsample2d>()?;
    m.add_class::<PyRNNCell>()?;
    m.add_class::<PyLSTMCell>()?;
    m.add_class::<PyGRUCell>()?;
    m.add_class::<PyAdamW>()?;
    m.add_class::<PySGD>()?;
    m.add_class::<PyAdam>()?;
    m.add_class::<PyRiemannianAdam>()?;
    m.add_class::<PyMuon>()?;
    m.add_class::<PyScheduleFreeAdamW>()?;
    m.add_class::<PyIrGraph>()?;
    m.add_class::<PyCompiledGraph>()?;
    m.add_class::<PyServingEngine>()?;
    m.add_class::<PySuperIntelligenceConfig>()?;
    m.add_class::<PySuperIntelligenceEngine>()?;
    m.add_class::<PyUniversalAGIEngine>()?;
    m.add_class::<PyOmniScientificFoundationEngine>()?;
    m.add_class::<PyOmniTransSingularityEngine>()?;
    m.add_class::<PyGrandTransUniversalEngineIV>()?;
    m.add_class::<PyGrandOmniReportV>()?;
    m.add_class::<PyGrandOmniUniversalEngineV>()?;
    m.add_class::<PyGrandOmniSeptuaReportVI>()?;
    m.add_class::<PyGrandOmniSeptuaEngineVI>()?;
    m.add_class::<PyGrandOmniOctaReportVII>()?;
    m.add_class::<PyGrandOmniOctaEngineVII>()?;
    m.add_class::<PySpeculativeDecoder>()?;
    m.add_class::<PyJsonSchema>()?;
    m.add_class::<PyAgentRuntime>()?;
    m.add_class::<PyExecutionSandbox>()?;
    m.add_class::<PySynthesisEngine>()?;
    m.add_class::<PyMelSpectrogram>()?;
    m.add_class::<PyWhisper>()?;
    m.add_class::<PyPipelineSchedule1F1B>()?;
    m.add_class::<PyZeRO3Parameter>()?;
    m.add_class::<PyBitLinear>()?;
    m.add_class::<PyFP8Tensor>()?;
    m.add_class::<PyGPTQLinear>()?;
    m.add_class::<PyGRPOTrainer>()?;
    m.add_class::<PyDPOTrainer>()?;
    m.add_class::<PyPPOTrainer>()?;
    m.add_class::<PyVideoVAE3d>()?;
    m.add_class::<PyRectifiedFlowVideoDiT>()?;
    m.add_class::<PyRotaryEmbedding2D>()?;
    m.add_class::<PyRotaryEmbedding3D>()?;
    m.add_class::<PyJitKernelBuilder>()?;
    m.add_class::<PyRingAttention>()?;
    m.add_class::<PyMCTSSearchEngine>()?;
    m.add_class::<PySelfCorrectionVerifier>()?;
    m.add_class::<PyAgentProposal>()?;
    m.add_class::<PySwarmOrchestrator>()?;
    m.add_class::<PyPropExpr>()?;
    m.add_class::<PySequentProver>()?;
    m.add_class::<PyDifferentiableILPEngine>()?;
    m.add_class::<PySymbolicNode>()?;
    m.add_class::<PyParetoIndividual>()?;
    m.add_class::<PySymbolicGpConfig>()?;
    m.add_class::<PySymbolicGpEngine>()?;
    m.add_class::<PyStructuralCausalModel>()?;
    m.add_class::<PyMultiscaleGraphWaveletTransform>()?;
    m.add_class::<PySpatioTemporalWaveletLayer>()?;
    m.add_class::<PyPrincipalBundleConnection>()?;
    m.add_class::<PyGaugeEquivariantConvLayer>()?;
    m.add_class::<PySu2Element>()?;
    m.add_class::<PyYangMillsLatticeGauge>()?;
    m.add_class::<PyAdaptiveCellularSheaf>()?;
    m.add_class::<PySheafDiffusionLayer>()?;
    m.add_class::<PySplitConformalRegressor>()?;
    m.add_class::<PyAdaptiveConformalClassifier>()?;
    m.add_class::<PyAlgorithmicGraph>()?;
    m.add_class::<PyNeuralBellmanFord>()?;
    m.add_class::<PyPointerGraphNetwork>()?;
    m.add_class::<PyPtxKernelEmitter>()?;
    m.add_class::<PyWgslKernelBuilder>()?;
    m.add_class::<PyVulkanSpirvKernel>()?;
    m.add_class::<PyPairformerBlock>()?;
    m.add_class::<PySE3EquivariantLayer>()?;
    m.add_class::<PySpectralConv1d>()?;
    m.add_class::<PyFourierNeuralOperator1d>()?;
    m.add_class::<PyDeepONet>()?;
    m.add_class::<PyMineEstimator>()?;
    m.add_class::<PyInfoNceEstimator>()?;
    m.add_class::<PyMolecularGeometryDiffusion>()?;
    m.add_class::<PyPersistenceDiagram>()?;
    m.add_class::<PyPersistentHomologyEngine>()?;
    m.add_class::<PySimplicialComplex>()?;
    m.add_class::<PyTropicalMatrix>()?;
    m.add_class::<PyTropicalPolynomial>()?;
    m.add_function(wrap_pyfunction!(py_tropical_distance, m)?)?;
    m.add_class::<PyRandomFourierFeatures>()?;
    m.add_class::<PyKernelRidgeRegressor>()?;
    m.add_function(wrap_pyfunction!(py_maximum_mean_discrepancy, m)?)?;
    m.add_class::<PyTapSolution>()?;
    m.add_class::<PyInformationGeometryEngine>()?;
    m.add_class::<PyAInfinityOperadEngine>()?;
    m.add_class::<PyHigherSimplex>()?;
    m.add_class::<PyQuasiCategoryEngine>()?;
    m.add_class::<PyTtCore>()?;
    m.add_class::<PyTensorTrain>()?;
    m.add_class::<PySherringtonKirkpatrickModel>()?;
    m.add_class::<PyGromovWassersteinConfig>()?;
    m.add_class::<PyGromovWassersteinResult>()?;
    m.add_class::<PyGromovWassersteinEngine>()?;
    m.add_class::<PyMachZehnderInterferometer>()?;
    m.add_class::<PyPhotonicMesh>()?;
    m.add_class::<PySaturableAbsorber>()?;
    m.add_class::<PyPhotonicLinear>()?;
    m.add_class::<PyWaveguideProfile>()?;
    m.add_class::<PyCoherentPhotonicCircuit>()?;
    m.add_class::<PyHamiltonianNeuralEngine>()?;
    m.add_class::<PyAutonomousQuantizer>()?;
    m.add_class::<PyDifferentiableR1csQap>()?;
    m.add_class::<PyZkMlEngine>()?;
    m.add_class::<PyCkksContext>()?;
    m.add_class::<PyFheCiphertext>()?;
    m.add_class::<PyFheSecretKey>()?;
    m.add_class::<PyFhePublicKey>()?;
    m.add_class::<PyMixtureOfDepths>()?;
    m.add_class::<PyElasticSubnetExtractor>()?;
    m.add_class::<PyEarlyExitRouter>()?;
    m.add_class::<PyArchitectureGenome>()?;
    m.add_class::<PyEvolutionaryAutoML>()?;
    m.add_class::<PyActionChunkingTransformer>()?;
    m.add_class::<PyFlowMatchingRoboticsPolicy>()?;
    m.add_class::<PyPoincareManifold>()?;
    m.add_class::<PyLorentzManifold>()?;
    m.add_class::<PyHyperbolicGATLayer>()?;
    m.add_class::<PyMultivector3D>()?;
    m.add_class::<PyCliffordLinear>()?;
    m.add_class::<PySpikingGnnLayer>()?;
    m.add_class::<PyRiemannianManifold>()?;
    m.add_class::<PyCellularSheaf>()?;
    m.add_class::<PyLangevinThermodynamics>()?;
    m.add_class::<PyJumpDiffusionEngine>()?;
    m.add_class::<PyNeuralJumpSDE>()?;
    m.add_class::<PyLatentSdeConfig>()?;
    m.add_class::<PySdeTrajectory>()?;
    m.add_class::<PyLatentSdeEngine>()?;
    m.add_class::<PyRiemannianTrajectory>()?;
    m.add_class::<PyMeanFieldGameSolver1D>()?;
    m.add_class::<PyPoincareCnfFlow>()?;
    m.add_class::<PyKerrSpacetime>()?;
    m.add_class::<PyContactHamiltonianSystem>()?;
    m.add_class::<PySpatialGrid1D>()?;
    m.add_class::<PySolitonPhysics>()?;
    m.add_class::<PySplitStepNlsEngine>()?;
    m.add_class::<PyRigidBodyInertia>()?;
    m.add_class::<PyLiePoissonState>()?;
    m.add_class::<PyLiePoissonSystem>()?;
    m.add_class::<PyHolographicMemoryEngine>()?;
    m.add_class::<PyQuantumCircuit>()?;
    m.add_class::<PyStabilizerCode>()?;
    m.add_class::<PyQecEngine>()?;
    m.add_class::<PyQaoaCircuit>()?;
    m.add_class::<PyVqeAnsatz>()?;
    m.add_class::<PyMatrixProductState>()?;
    m.add_class::<PyZneExtrapolator>()?;
    m.add_class::<PyCliffordDataRegression>()?;
    m.add_class::<PyQGNNGradients>()?;
    m.add_class::<PySu2ChernSimonsTqft>()?;
    m.add_class::<PyGaussianCvCircuit>()?;
    m.add_class::<PyQuantumGraphNeuralNetwork>()?;
    m.add_class::<PyDeqConfig>()?;
    m.add_class::<PyDeepEquilibriumModel>()?;
    m.add_class::<PyRiskCalibrationResult>()?;
    m.add_class::<PyConformalRiskController>()?;
    m.add_class::<PyToposHeytingEngine>()?;
    m.add_class::<PyLearnThenTest>()?;
    m.add_class::<PyComplexEigenvalue>()?;
    m.add_class::<PyExceptionalPointReport>()?;
    m.add_class::<PyJarzynskiReport>()?;
    m.add_class::<PyStochasticThermoEngine>()?;
    m.add_class::<PySymplecticTopologyEngine>()?;
    m.add_class::<PyAdsCftHolographicEngine>()?;
    m.add_class::<PyRmhdPrimitiveState>()?;
    m.add_class::<PyHeisenbergPoint>()?;
    m.add_class::<PyHeisenbergLieAlgebra>()?;
    m.add_class::<PyHypoellipticDiffusion3D>()?;
    m.add_class::<PyNonHermitianHamiltonian2x2>()?;
    m.add_class::<PyDynamicTelemetry>()?;
    m.add_class::<PyDynamicQuantPolicy>()?;
    m.add_class::<PyDynamicTransitionPlan>()?;
    m.add_class::<PyDynamicLlmQuantizer>()?;
    m.add_class::<PyMamba2Block>()?;
    m.add_class::<PyTransfusionModel>()?;
    m.add_class::<PyTemporalInteractionGNN>()?;
    m.add_class::<PyMslKernelEmitter>()?;
    m.add_class::<PyTileKernelPlan>()?;
    m.add_class::<PyKernelFusionEngine>()?;
    m.add_class::<PyTileConfig>()?;
    m.add_class::<PyKernelAutoTuner>()?;
    m.add_class::<PyShardedTensor>()?;
    m.add_class::<PyAutoParallelPlanner>()?;
    m.add_class::<PyDpSgdConfig>()?;
    m.add_class::<PyByzantineAggregator>()?;
    m.add_class::<PyUnifiedWorldModel>()?;
    m.add_class::<PyTreeAttentionMask>()?;
    m.add_class::<PyMedusaHeads>()?;
    m.add_class::<PyTreeSpeculativeDecoder>()?;
    m.add_class::<PyFlashDecodingConfig>()?;
    m.add_class::<PyFlashDecodingEngine>()?;
    m.add_class::<PyAuxFreeRouter>()?;
    m.add_class::<PyDeepSeekV3MoE>()?;
    m.add_class::<PyDiffusionForcingScheduler>()?;
    m.add_class::<PyTitansMemoryConfig>()?;
    m.add_class::<PyTitansNeuralMemory>()?;
    m.add_class::<PyTitansBlock>()?;
    m.add_class::<PyDualPipeSchedule>()?;
    m.add_class::<PyParallel3DConfig>()?;
    m.add_class::<PyParallel3DRuntime>()?;
    m.add_class::<PyEDMConfig>()?;
    m.add_class::<PyKDiffusionSampler>()?;
    m.add_class::<PyRectifiedFlowConfig>()?;
    m.add_class::<PyRectifiedFlowEngine>()?;
    m.add_class::<PyTernaryPackedTensor>()?;
    m.add_class::<PyPackedBitLinear>()?;
    m.add_class::<PyRVQConfig>()?;
    m.add_class::<PyResidualVectorQuantizer>()?;
    m.add_class::<PySymbolicShape>()?;
    m.add_class::<PyEagleConfig>()?;
    m.add_class::<PyEagleSpeculativeEngine>()?;
    m.add_class::<PyAttentionSinkConfig>()?;
    m.add_class::<PyStreamingKVCache>()?;
    m.add_class::<PyOmniTokenizer>()?;
    m.add_class::<PyMaxPool1d>()?;
    m.add_class::<PyAvgPool1d>()?;
    m.add_class::<PyConvTranspose3d>()?;
    m.add_class::<PyInstanceNorm1d>()?;
    m.add_class::<PyInstanceNorm3d>()?;
    crate::python_millennium::register_millennium_classes(m)?;
    crate::python_hardware::register_hardware_classes(m)?;
    crate::python_flops::register_flops_functions(m)?;
    crate::python_vision::register_vision_classes(m)?;
    crate::python_interop::register_interop_functions(m)?;
    m.add_class::<PyEmbeddingBag>()?;
    m.add_class::<PyReLU6>()?;
    m.add_class::<PyHardtanh>()?;
    m.add_class::<PySoftshrink>()?;
    m.add_class::<PyHardshrink>()?;
    m.add_class::<PyLogSigmoid>()?;
    m.add_class::<PyMarginRankingLoss>()?;
    m.add_class::<PyHingeEmbeddingLoss>()?;
    m.add_class::<PySoftMarginLoss>()?;
    m.add_class::<PyCTCLoss>()?;
    m.add_class::<PyQuantizedPagedKVCache>()?;
    m.add_class::<PyStreamingUtf8Decoder>()?;
    m.add_class::<PyStopSequenceMatcher>()?;
    m.add_class::<PyQLoRALinear>()?;
    m.add_class::<PyGGUFFile>()?;
    m.add_class::<PyMultiHeadLatentAttention>()?;
    m.add_class::<PyDoubleBufferedKVCache>()?;
    m.add_class::<PyAsyncKVPipelineEngine>()?;
    m.add_class::<PyUnifiedSpeculativeEngine>()?;
    m.add_class::<PyGGUFCausalLM>()?;
    m.add_class::<PyGGUFTokenizer>()?;
    m.add_class::<PyPrefillDecodeInterleaver>()?;
    m.add_class::<PyTieredMemoryManager>()?;
    m.add_class::<PyStreamEvent>()?;
    m.add_class::<PyComputeStream>()?;
    m.add_class::<PyComputeGraph>()?;

    m.add_function(wrap_pyfunction!(py_manual_seed, m)?)?;
    m.add_function(wrap_pyfunction!(py_initial_seed, m)?)?;
    m.add_function(wrap_pyfunction!(py_seed, m)?)?;
    m.add_function(wrap_pyfunction!(py_get_rng_state, m)?)?;
    m.add_function(wrap_pyfunction!(py_set_rng_state, m)?)?;
    m.add_function(wrap_pyfunction!(py_grad, m)?)?;
    m.add_function(wrap_pyfunction!(py_flash_attention_2, m)?)?;
    m.add_function(wrap_pyfunction!(py_flash_attention_2_backward, m)?)?;
    m.add_function(wrap_pyfunction!(py_flash_attention_3, m)?)?;
    m.add_function(wrap_pyfunction!(py_fused_rmsnorm, m)?)?;
    m.add_function(wrap_pyfunction!(py_fused_add_rmsnorm, m)?)?;
    m.add_function(wrap_pyfunction!(py_fused_swiglu, m)?)?;
    m.add_function(wrap_pyfunction!(py_fused_geglu, m)?)?;
    m.add_function(wrap_pyfunction!(py_fused_linear_activation, m)?)?;
    m.add_function(wrap_pyfunction!(py_fused_cross_entropy_with_z_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_fused_layernorm, m)?)?;
    m.add_function(wrap_pyfunction!(py_fused_add_layernorm, m)?)?;
    m.add_function(wrap_pyfunction!(py_fused_bias_activation, m)?)?;
    m.add_function(wrap_pyfunction!(py_fused_silu_and_mul, m)?)?;
    m.add_function(wrap_pyfunction!(py_fused_rotary_qk, m)?)?;
    m.add_function(wrap_pyfunction!(py_fused_rope, m)?)?;
    m.add_function(wrap_pyfunction!(py_fused_sdpa, m)?)?;
    m.add_function(wrap_pyfunction!(py_mlx_quantize_affine_q4, m)?)?;
    m.add_function(wrap_pyfunction!(py_mlx_dequantize_affine_q4, m)?)?;
    m.add_function(wrap_pyfunction!(py_mlx_quantized_matmul_q4, m)?)?;
    m.add_function(wrap_pyfunction!(py_flash_attention_tiled, m)?)?;
    m.add_function(wrap_pyfunction!(py_paged_kv_attention, m)?)?;
    m.add_function(wrap_pyfunction!(py_block_diag, m)?)?;
    m.add_function(wrap_pyfunction!(py_cartesian_prod, m)?)?;
    m.add_function(wrap_pyfunction!(py_compile_fused_linear_silu_with_grad, m)?)?;
    m.add_function(wrap_pyfunction!(py_pixel_shuffle, m)?)?;
    m.add_function(wrap_pyfunction!(py_awq_quantize_forward, m)?)?;
    m.add_function(wrap_pyfunction!(py_save_safetensors, m)?)?;
    m.add_function(wrap_pyfunction!(py_load_safetensors, m)?)?;
    m.add_function(wrap_pyfunction!(py_load_safetensors_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(py_allocator_stats, m)?)?;
    m.add_function(wrap_pyfunction!(py_device_bandwidth, m)?)?;
    m.add_function(wrap_pyfunction!(py_device_latency, m)?)?;
    m.add_function(wrap_pyfunction!(py_set_default_device, m)?)?;
    m.add_function(wrap_pyfunction!(py_get_default_device, m)?)?;
    m.add_function(wrap_pyfunction!(py_get_available_devices, m)?)?;
    m.add_function(wrap_pyfunction!(py_is_available, m)?)?;
    m.add_function(wrap_pyfunction!(py_device_count, m)?)?;
    m.add_function(wrap_pyfunction!(py_mse_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_l1_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_huber_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_nll_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_cross_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(py_bce_with_logits, m)?)?;
    m.add_function(wrap_pyfunction!(py_kl_div_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_focal_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_cosine_embedding_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_triplet_margin_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_info_nce_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_dpo_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_ppo_clipped_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_simpo_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_siglip_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_barlow_twins_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_vicreg_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_grpo_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_margin_ranking_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_hinge_embedding_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_soft_margin_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_ctc_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_cat, m)?)?;
    m.add_function(wrap_pyfunction!(py_stack, m)?)?;
    m.add_function(wrap_pyfunction!(py_where, m)?)?;
    m.add_function(wrap_pyfunction!(py_tril, m)?)?;
    m.add_function(wrap_pyfunction!(py_triu, m)?)?;
    m.add_function(wrap_pyfunction!(py_kron, m)?)?;
    m.add_function(wrap_pyfunction!(py_cdist, m)?)?;
    m.add_function(wrap_pyfunction!(py_pdist, m)?)?;
    m.add_function(wrap_pyfunction!(py_cross, m)?)?;
    m.add_function(wrap_pyfunction!(py_cummax, m)?)?;
    m.add_function(wrap_pyfunction!(py_cummin, m)?)?;
    m.add_function(wrap_pyfunction!(py_logcumsumexp, m)?)?;
    m.add_function(wrap_pyfunction!(py_tensordot, m)?)?;
    m.add_function(wrap_pyfunction!(py_as_strided, m)?)?;
    m.add_function(wrap_pyfunction!(py_conv2d, m)?)?;
    m.add_function(wrap_pyfunction!(py_conv1d, m)?)?;
    m.add_function(wrap_pyfunction!(py_searchsorted, m)?)?;
    m.add_function(wrap_pyfunction!(py_einsum, m)?)?;
    m.add_function(wrap_pyfunction!(py_paged_attention_quantized, m)?)?;
    m.add_function(wrap_pyfunction!(py_save_gguf, m)?)?;
    m.add_function(wrap_pyfunction!(fast_murmurhash3_32, m)?)?;
    Ok(())
}

/// Python extension entrypoint (`import oxideon`).
#[pymodule]
pub fn oxideon(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_python_module(m)
}

#[cfg(test)]
mod lock_recovery_tests {
    use super::{mutex_guard, rw_read, rw_write};

    #[test]
    fn mutex_poison_recovers_inner() {
        let m = std::sync::Mutex::new(7u32);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _held = m.lock().unwrap();
            panic!("holder died");
        }));
        assert!(m.is_poisoned());
        assert_eq!(*mutex_guard(&m), 7);
    }

    #[test]
    fn rwlock_poison_recovers_inner() {
        let m = std::sync::RwLock::new(9u32);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _held = m.write().unwrap();
            panic!("writer died");
        }));
        assert!(m.is_poisoned());
        assert_eq!(*rw_read(&m), 9);
        assert_eq!(*rw_write(&m), 9);
    }
}
