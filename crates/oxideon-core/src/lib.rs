//! Oxideon core: multi-dtype strided-view tensors, higher-order autograd, neural-network layers,
//! optimizers, quantization, backends and serving.
//!
//! The science / physics / crypto-demo engine catalog lives in `oxideon-engines`; the `oxideon` crate
//! re-exports both and hosts the Python bindings.

pub mod adaptive;
pub mod allocator;
pub mod amp;
pub mod audio;
pub mod autograd_utils;
pub mod backend;
pub mod compiler;
pub mod opgraph;
pub mod complex;
pub mod compression;
pub mod data;
pub mod data_ext;
pub mod data_transforms;
pub mod device;
pub mod diffusion;
pub mod comm;
pub mod distributed;
pub mod distributions;
pub mod driver;
pub mod distributions_ext;
pub mod distributions_transforms;
pub mod dtype;
pub mod einops;
pub mod embedded_nostd;
pub mod c_api;
pub mod error;
pub mod image_codec;
pub mod string_tensor;
pub mod explain;
pub mod fft;
pub mod functional;
pub mod functional_ext;
pub mod image_aug;
pub mod metrics_more;
pub mod classical;
pub mod classical_linear;
pub mod classical_more;
pub mod manifold;
pub mod timeseries;
pub mod preprocessing;
pub mod model_selection;
pub mod inspect;
pub mod checkpointing;
pub mod tuning_ext;
pub mod gp;
pub mod scientific;
pub mod optimization;
pub mod graph_algos;
pub mod stats_tests;
pub mod survival;
pub mod inference;
pub mod text_ir;
pub mod spatial;
pub mod numtheory;
pub mod dsp;
pub mod ml_extras;
pub mod decomp;
pub mod dataio;
pub mod anomaly;
pub mod recsys;
pub mod glm;
pub mod causal;
pub mod sketches;
pub mod iterative;
pub mod frame;
pub mod numerics;
pub mod evolutionary;
pub mod naive_bayes;
pub mod prob_models;
pub mod algebra;
pub mod qmc;
pub mod imageproc;
pub mod geometry3d;
pub mod finance;
pub mod operations;
pub mod fused_kernels;
pub mod gemm;
pub mod geom_graphs;
pub mod grad_checkpoint;
pub mod grammar;
pub mod graph;
pub mod indexing;
pub mod inplace;
pub mod jit;
pub mod layout;
pub mod limits;
pub mod linalg;
pub mod lapack;
pub mod numeric_kernels;
pub mod optim_kernels;
pub mod linalg_extra;
pub mod long_context;
pub mod metrics;
pub mod metrics_ext;
pub mod nn;
pub mod numeric_ext;
pub mod ode;
pub mod ops;
pub mod trace;
pub mod ops_extra;
pub mod optim;
pub mod optim_extra;
pub mod quant;
pub mod quant_aware;
pub mod quantum;
pub mod reasoning;
pub mod rl;
pub mod rng;
pub mod robotics;
pub mod sampling;
pub mod special;
pub mod special_ext;
pub mod serialize;
pub mod serving;
pub mod signal;
pub mod voice;
pub mod sparse;
pub mod structured;
pub mod checkpoint;
pub mod dlpack;
pub mod hub;
pub mod loader;
pub mod memory;
pub mod train_loop;
pub mod pool;
pub mod profiler;
pub mod summary_writer;
pub mod transforms;
pub mod sparse_ext;
pub mod ops_stats;
pub mod samplers_ext;
pub mod privacy;
pub mod flops;
pub mod super_intelligence;
pub mod tensor;
pub mod tensor_misc;
pub mod tile_ir;
pub mod tokenizer;
pub mod trainer;
pub mod training_utils;
pub mod tuning;
pub mod vectorize;
pub mod video;
pub mod vision;
pub mod vision_ops;
pub mod wasm;
pub mod world_model;
pub use adaptive::{
    ArchitectureGenome, AutonomousQuantPolicy, AutonomousQuantizer, BlockQuantProfile,
    EarlyExitRouter, ElasticSubnetExtractor, EvolutionaryAutoML, MixtureOfDepths, QuantFormat,
    QuantizedCausalLM, QuantizedLMBlock, QuantizedLayer, ZeroCostScore, compute_synflow,
};
pub use allocator::{AllocatorStats, CachingAllocator};
pub use amp::{EMA, GradScaler, MasterParams, autocast, autocast_cast, autocast_dtype, is_autocast_enabled};
pub use audio::{
    AudioEncoder, MelSpectrogram, RVQConfig, RVQOutput, ResidualVectorQuantizer, VQCodebookStage,
    Whisper, WhisperConfig, mel_filterbank_matrix,
};
pub use backend::{ComputeGraph, ComputeStream, CpuStorage, Storage, StreamEvent, StreamId};
pub use compiler::{
    AotCompiler, AotTarget, CompiledGraph, IrGraph, IrNode, IrOp, MemoryPlan, OptimizationReport,
    SymDim, SymbolicShape, ValueId, jit_compile, trace_to_ir,
};
pub use compression::{
    PruningMask, distillation_loss, prune_magnitude_unstructured, prune_structured_l2_rows,
};
pub use device::{
    AutoDevicePlacer, CompositeDevice, Device, DeviceMesh, DeviceProperties, DeviceType,
    InterconnectMatrix, InterconnectType, LayerPlacement, LayerPlacementSpec, MemoryPriority,
    TieredMemoryManager, TieredTensorEntry, get_default_device, set_default_device, with_device,
};
pub use diffusion::{
    BetaSchedule, DDPMScheduler, DiffusionForcingScheduler, EDMConfig, KDiffusionSampler,
    RectifiedFlowConfig, RectifiedFlowEngine, timestep_embedding,
};
pub use distributed::{
    AutoParallelPlan, AutoParallelPlanner, ByzantineAggregator, ColumnParallelLinear, DpSgdConfig,
    DualPipeDirection, DualPipeOpKind, DualPipeSchedule, DualPipeStep, Parallel3DConfig,
    Parallel3DRuntime, PipelinePhase, PipelineSchedule1F1B, Placement, RankCoords3D, ReduceOp,
    RowParallelLinear, ShardedTensor, ZeRO3Parameter, ZeROAdamW, all_gather_tensors,
    all_reduce_tensors, reduce_scatter_tensors, sync_gradients_ddp,
};
pub use distributions::{Categorical, Laplace, Normal};
pub use driver::{MslKernelEmitter, PtxKernelEmitter, VulkanSpirvKernel, WgslKernelBuilder};
pub use dtype::{DType, Element, bf16, f16};
pub use error::{Error, Result};
pub use gemm::{PrecisionMode, precision_mode, set_precision_mode};
pub use explain::{grad_cam_2d, integrated_gradients, saliency_map, smooth_grad};
pub use functional::{hessian, hvp, jacobian, jvp, ntk_matrix, vjp, vmap, vmap2};
pub use fused_kernels::{
    FusedActivation, fused_add_rmsnorm, fused_cross_entropy_with_z_loss, fused_geglu,
    flash_attention_tiled, fused_layernorm, fused_linear_activation, fused_rmsnorm, fused_rope,
    fused_sdpa, fused_swiglu, mlx_dequantize_affine_q4, mlx_quantize_affine_q4,
    mlx_quantized_matmul_q4, paged_kv_attention,
};
pub use geom_graphs::GraphWassersteinGeodesicEngine;
pub use geom_graphs::{
    CellularSheaf, ChebyshevWaveletFilter, CliffordLinear, GaugeEquivariantConvLayer,
    GraphLaplacian, HyperbolicGATLayer, LaplacianType, LorentzManifold,
    MultiscaleGraphWaveletTransform, Multivector3D, PoincareManifold, PrincipalBundleConnection,
    RiemannianManifold, SpatioTemporalWaveletLayer, SpikingGnnLayer, Su2Element,
    TemporalInteractionGNN, YangMillsLatticeGauge,
};
pub use grammar::{
    AgentRuntime, ExecutionReport, ExecutionSandbox, GrammarLogitProcessor, JsonParserState,
    JsonSchema, SandboxStatus, SchemaType, SynthesisEngine, SynthesisReport, TestCase, ToolDef,
};
pub use graph::{
    ExecutionProfile, GraphStats, analyze_graph, export_graphviz_dot, profile_forward_backward,
};
pub use indexing::{IndexArg, SliceElem};
pub use jit::{JitKernelBuilder, JitNode, JitOp};
pub use layout::{DimMap, Layout, ViewSpec};
pub use long_context::{
    ChunkedPrefillEngine, FlashAttention2Config, FlashAttention2ForwardOutput,
    FlashAttention3Config, FlashDecodingConfig, FlashDecodingEngine, HolographicMemoryEngine,
    RingAttention, flash_attention_2, flash_attention_2_backward, flash_attention_2_forward,
    flash_attention_3,
};
pub use nn::ComputeOptimalChinchillaEngine;
pub use nn::InfiniteWidthNtkEngine;
pub use nn::LogSobolevConcentrationEngine;
pub use nn::MaximalUpdateParamEngine;
pub use nn::OverparameterizedDoubleDescent;
pub use nn::PacBayesMcAllesterEngine;
pub use nn::RandomFourierBochnerEngine;
pub use nn::RestrictedIsometryLassoEngine;
pub use nn::SplitConformalPredictionEngine;
pub use ode::ControlBarrierLyapunovEngine;
pub use ode::FreidlinWentzellLdpEngine;
pub use ode::HInfinityRobustControlEngine;
pub use ode::KalmanBucyRiccatiEngine;
pub use ode::KoopmanSpectralEngine;
pub use ode::KpzColeHopfEngine;
pub use ode::LevyItodiffusionEngine;
pub use ode::MalliavinBismutEngine;
pub use ode::McKeanVlasovPropagationChaos;
pub use ode::PardouxPengBsdeEngine;
pub use ode::PontryaginMaximumEngine;
pub use ode::RoughPathSignature;
pub use ode::RoughVolatilityHestonEngine;
pub use ode::SchrammLoewnerEngine;
pub use ode::SteinPoissonCharacterization;
pub use ode::SubRiemannianHeisenbergEngine;
pub use ode::{
    FlowMatcher, JumpDiffusionEngine, LatentSdeConfig, LatentSdeEngine, MeanFieldGameSolver1D,
    NeuralJumpSDE, NeuralODE, OdeMethod, PoincareCnfFlow, RiemannianTrajectory, SdeTrajectory,
    odeint, odeint_trajectory,
};
pub use optim::{ManifoldType, RiemannianAdam};
pub use quant::{
    AWQLinear, AdaptiveQuantLinear, BitLinear, DynamicLlmQuantizer, DynamicPrecisionLevel,
    DynamicQuantPolicy, DynamicTelemetry, DynamicTransitionPlan, FP8Format, FP8Linear, FP8Tensor,
    GPTQLinear, PackedBitLinear, QuantScheme, QuantizedLinear, QuantizedTensor,
    TernaryPackedTensor, fp8_e4m3, fp8_e5m2,
};
pub use quantum::CavityOptomechanicsBackaction;
pub use quantum::ClassicalShadowTomography;
pub use quantum::CorkumHighHarmonicAttosecond;
pub use quantum::DickeTavisCummingsSuperradiance;
pub use quantum::DopplerSisyphusLaserCooling;
pub use quantum::ElectromagneticallyInducedTransparency;
pub use quantum::FeshbachScatteringResonance;
pub use quantum::FibonacciAnyonBraider;
pub use quantum::FloquetTimeCrystalEngine;
pub use quantum::GlauberPhotonCoherenceEngine;
pub use quantum::HongOuMandelInterferometer;
pub use quantum::JaynesCummingsCavityQED;
pub use quantum::KitaevMajoranaChainEngine;
pub use quantum::MagicStateDistillationEngine;
pub use quantum::MeraEntanglementEngine;
pub use quantum::QuantumDarwinismEngine;
pub use quantum::QuantumLdpcCodeEngine;
pub use quantum::QuantumMetrologyEngine;
pub use quantum::QuantumOttoEngine;
pub use quantum::SykQuantumChaosEngine;
pub use quantum::TransmonCircuitQEDQubit;
pub use quantum::{
    CliffordDataRegression, ExtrapolationModel, GaussianCvCircuit, MatrixProductState,
    MpsSiteTensor, Pauli, QGNNGradients, QaoaCircuit, QecEngine, QuantumCircuit, QuantumGate,
    QuantumGraphNeuralNetwork, Stabilizer, StabilizerCode, Su2ChernSimonsTqft, VqeAnsatz,
    ZneExtrapolator, matrix_hafnian,
};
pub use reasoning::HomotopyTypeEngine;
pub use reasoning::{
    AdaptiveConformalClassifier, AgentProposal, AlgorithmicEdge, AlgorithmicGraph,
    AlgorithmicSemiring, CausalVariable, ConformalEvaluator, ConformalRiskController,
    ConformalScoreType, ConsensusReport, ConsensusStrategy, DifferentiableILPEngine, LearnThenTest,
    MCTSNode, MCTSSearchEngine, NeuralBellmanFord, OpType, ParetoIndividual, PointerGraphNetwork,
    PredictionInterval, PropExpr, ReasoningStep, ReasoningTrace, RiskCalibrationResult, RiskMetric,
    SearchResult, SelfCorrectionVerifier, Sequent, SequentProver, SplitConformalRegressor,
    StructuralCausalModel, SwarmOrchestrator, SymbolicGpConfig, SymbolicGpEngine, SymbolicNode,
    TNorm, ToposHeytingEngine, VerificationReport,
};
pub use reasoning::{AlgorithmicHypothesis, SolomonoffLevinEngine};
pub use rl::{
    DPOConfig, DPOStepStats, DPOTrainer, GRPOConfig, GRPOStepStats, GRPOTrainer, PPOConfig,
    PPOStepStats, PPOTrainer, RuleReward, compute_group_advantages, grpo_loss,
};
pub use rng::{
    Generator, fork_rng, get_rng_state, initial_seed, manual_seed, seed, set_rng_state, with_rng,
};
pub use robotics::{ACTConfig, ActionChunkingTransformer, FlowMatchingRoboticsPolicy};
pub use sampling::SamplingConfig;
pub use serialize::{
    GGMLType, GGUFFile, GGUFTensorInfo, GGUFValue, deserialize_safetensors, load_safetensors,
    save_gguf, save_safetensors, serialize_gguf, serialize_safetensors,
};
pub use serving::{
    AsyncKVPipelineEngine, AttentionSinkConfig, BatchRequest, BlockAllocator, BlockTable,
    ContinuousBatchScheduler, DecodeRequest, DoubleBufferedKVCache, EagleConfig, EagleDraftHead,
    EagleSpeculativeEngine, InterleavedBatch, MedusaHeads, PagedKVCache, PhysicalBlockId,
    PrefillChunk, PrefillDecodeInterleaver, PrefillRequest, QuantKVDType, QuantizedPagedKVCache,
    ServingEngine, SpeculativeConfig, SpeculativeDecoder, SpeculativeMetrics, SpeculativeTree,
    SpeculativeTreeNode, StopSequenceMatcher, StreamingKVCache, StreamingUtf8Decoder,
    TransferDirection, TransferEvent, TreeAttentionMask, TreeSpeculativeDecoder,
    UnifiedSpeculativeEngine, VerificationOutcome, paged_attention, paged_attention_quantized,
};
pub use sparse::{SparseCoo, SparseCsr};
pub use super_intelligence::{
    SuperIntelligenceConfig, SuperIntelligenceEngine, SuperIntelligenceMetrics,
};
pub use tensor::{
    CustomOp, Tensor, checkpoint, enable_grad, grad, is_grad_enabled, no_grad, set_grad_enabled,
};
pub use tile_ir::{
    HardwareProfile, HardwareTarget, KernelAutoTuner, KernelFusionEngine, MemoryScope, TileConfig,
    TileKernelPlan, TileOp, TuneMetric,
};
pub use tokenizer::{
    BOS_TOKEN_ID, BpeTokenizer, EOS_TOKEN_ID, ModalityKind, OmniSequence, OmniTokenSegment,
    OmniTokenizer, PAD_TOKEN_ID, UNK_TOKEN_ID,
};
pub use trainer::{EarlyStopping, EpochMetrics, Trainer, TrainerConfig, TrainingHistory};
pub use tuning::{ParamDomain, ParamSpace, ParamValue, Study, TrialConfig, TrialResult};
pub use vectorize::{
    DictVal, FastBM25Index, FastByteHistogramVectorizer, FastCountVectorizer, FastDeepSetsVectorizer,
    FastDictVectorizer, FastDiscretizer, FastHashingVectorizer, FastJsonTreeVectorizer,
    FastLsaVectorizer, FastMaxSimScorer, FastMinHashVectorizer, FastOneHotVectorizer,
    FastPathSignatureVectorizer, FastSimHashVectorizer, FastSphericalHarmonics,
    FastTextCharNGramVectorizer, FastTfidfVectorizer, FastTimeSeriesSpectralVectorizer,
    FastWeisfeilerLehmanVectorizer, KernelSpectralVectorizer, extract_char_ngrams, extract_ngrams,
    murmurhash3_32, tokenize_words,
};
pub use video::{
    CausalConv3d, RectifiedFlowVideoDiT, RotaryEmbedding3D, VideoDiTBlock, VideoDiTConfig,
    VideoResBlock3d, VideoVAE3d,
};
pub use vision::RotaryEmbedding2D;
pub use vision_ops::{affine_grid, box_iou, grid_sample, nms, roi_align};
pub use wasm::{WasmInferenceSession, WasmTensor};
pub use world_model::{UnifiedWorldModel, WorldModelConfig};

/// Convenient glob import: `use oxideon_core::prelude::*;`
pub mod prelude {
    pub use crate::adaptive::{
        ArchitectureGenome, AutonomousQuantPolicy, AutonomousQuantizer, BlockQuantProfile,
        EarlyExitRouter, ElasticSubnetExtractor, EvolutionaryAutoML, MixtureOfDepths, QuantFormat,
        QuantizedCausalLM, QuantizedLMBlock, QuantizedLayer, ZeroCostScore, compute_synflow,
    };
    pub use crate::allocator::{AllocatorStats, CachingAllocator};
    pub use crate::amp::{
        EMA, GradScaler, MasterParams, autocast, autocast_cast, autocast_dtype, is_autocast_enabled,
    };
    pub use crate::audio::{
        AudioEncoder, MelSpectrogram, RVQConfig, RVQOutput, ResidualVectorQuantizer, Whisper,
        WhisperConfig, mel_filterbank_matrix,
    };
    pub use crate::backend::{ComputeGraph, ComputeStream, CpuStorage, StreamEvent, StreamId};
    pub use crate::compiler::{
        AotCompiler, AotTarget, CompiledGraph, IrGraph, IrNode, IrOp, MemoryPlan,
        OptimizationReport, SymDim, SymbolicShape, ValueId, jit_compile, trace_to_ir,
    };
    pub use crate::compression::{
        PruningMask, distillation_loss, prune_magnitude_unstructured, prune_structured_l2_rows,
    };
    pub use crate::data::{
        AddGaussianNoise, Compose, DataLoader, Dataset, Normalize, TensorDataset, Transform,
        batch_indices, k_fold_split, make_blobs, make_circles, make_moons, train_test_split,
    };
    pub use crate::device::{
        AutoDevicePlacer, CompositeDevice, Device, DeviceMesh, DeviceProperties, DeviceType,
        InterconnectMatrix, InterconnectType, LayerPlacement, LayerPlacementSpec, MemoryPriority,
        TieredMemoryManager, TieredTensorEntry, get_default_device, set_default_device,
        with_device,
    };
    pub use crate::diffusion::{
        BetaSchedule, DDPMScheduler, DiffusionForcingScheduler, EDMConfig, KDiffusionSampler,
        RectifiedFlowConfig, RectifiedFlowEngine, timestep_embedding,
    };
    pub use crate::distributed::{
        AutoParallelPlan, AutoParallelPlanner, ColumnParallelLinear, DualPipeDirection,
        DualPipeOpKind, DualPipeSchedule, DualPipeStep, Parallel3DConfig, Parallel3DRuntime,
        PipelinePhase, PipelineSchedule1F1B, Placement, RankCoords3D, ReduceOp, RowParallelLinear,
        ShardedTensor, ZeRO3Parameter, ZeROAdamW, all_gather_tensors, all_reduce_tensors,
        reduce_scatter_tensors, sync_gradients_ddp,
    };
    pub use crate::distributions::{Categorical, Laplace, Normal};
    pub use crate::driver::{
        MslKernelEmitter, PtxKernelEmitter, VulkanSpirvKernel, WgslKernelBuilder,
    };
    pub use crate::dtype::{DType, Element, bf16, f16};
    pub use crate::error::{Error, Result};
    pub use crate::explain::{grad_cam_2d, integrated_gradients, saliency_map, smooth_grad};
    pub use crate::functional::{hessian, hvp, jacobian, jvp, ntk_matrix, vjp, vmap, vmap2};
    pub use crate::fused_kernels::{
        FusedActivation, fused_add_rmsnorm, fused_cross_entropy_with_z_loss, fused_geglu,
        flash_attention_tiled, fused_layernorm, fused_linear_activation, fused_rmsnorm, fused_rope,
        fused_sdpa, fused_swiglu, mlx_dequantize_affine_q4, mlx_quantize_affine_q4,
        mlx_quantized_matmul_q4, paged_kv_attention,
    };
    pub use crate::geom_graphs::{
        CliffordLinear, HyperbolicGATLayer, LorentzManifold, Multivector3D, PoincareManifold,
        RiemannianManifold, SpikingGnnLayer, TemporalInteractionGNN,
    };
    pub use crate::grammar::{
        AgentRuntime, ExecutionReport, ExecutionSandbox, GrammarLogitProcessor, JsonParserState,
        JsonSchema, SandboxStatus, SchemaType, SynthesisEngine, SynthesisReport, TestCase, ToolDef,
    };
    pub use crate::graph::{
        ExecutionProfile, GraphStats, analyze_graph, export_graphviz_dot, profile_forward_backward,
    };
    pub use crate::indexing::{IndexArg, SliceElem};
    pub use crate::jit::{JitKernelBuilder, JitNode, JitOp};
    pub use crate::long_context::{
        ChunkedPrefillEngine, FlashAttention2Config, FlashAttention2ForwardOutput,
        FlashAttention3Config, FlashDecodingConfig, FlashDecodingEngine, RingAttention,
        flash_attention_2, flash_attention_2_backward, flash_attention_2_forward,
        flash_attention_3,
    };
    pub use crate::metrics;
    pub use crate::nn::{
        self, ActivationKind, AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveMaxPool1d,
        AdaptiveMaxPool2d, AlphaDropout, AuxFreeRouter, AvgPool1d, AvgPool2d, AvgPool3d,
        BatchNorm1d, BatchNorm2d, BatchNorm3d, BiLSTM, Bilinear, CELU, CTCLoss, CausalLM,
        CausalLMBlock, Conv1d, Conv2d, Conv3d, ConvLstmCell, ConvNeXt, ConvNeXtBlock, ConvTranspose1d,
        ConvTranspose2d, ConvTranspose3d, CosineLinear, DeepEquilibriumModel, DeepSeekBlock,
        DeepSeekMoELM, DeepSeekV3MoE, DeqConfig, DeqForwardOutput, DiT, DiTBlock, DiffAttention,
        DoubleConv2d, DropPath, Dropout, Dropout2d, ELU, Embedding, EmbeddingBag, EmbeddingBagMode,
        EventDrivenSpikeRouter, FeedForward, Flatten, GATConv, GCNConv, GELU, GGUFCausalLM,
        GGUFDecoderBlock, GGUFModelConfig, GLU, GRU, GRUCell, GatedCrossAttention, GroupNorm,
        GroupedQueryAttention, Hardshrink, Hardsigmoid, Hardswish, Hardtanh, HingeEmbeddingLoss,
        HybridChannelMixer, HybridJambaLM, HybridSequenceMixer, HyenaOperator, Identity,
        InstanceNorm1d, InstanceNorm2d, InstanceNorm3d, KAN, KANLinear, KVCache, LIFNode, LPPool1d,
        LPPool2d, LSTM, LSTMCell, LambdaModule, LayerKVCache, LayerNorm, LeakyReLU, Linear,
        LoRALinear, LogSigmoid, MLP, MLPMixer, MLPMixerBlock, Mamba2Block, MambaBlock,
        MarginRankingLoss, MaxPool1d, MaxPool2d, MaxPool3d, Maxout, Mish, Module, ModuleDict, ModuleList,
        MultiHeadAttention, MultiHeadLatentAttention, MultiTokenPredictionModule, MultimodalVLM,
        PReLU, ParallelAdd, ParallelCat, PatchEmbedding, PixelShuffle, PixelUnshuffle,
        PoissonEncoder, PolychronousNeuralLayer, QLoRALinear, RMSNorm, RNN, RNNCell, ReLU, ReLU6,
        ResetMode, Residual, ResidualBlock, RewardModel, RotaryEmbedding, SELU, STDPConfig,
        STDPSynapse, SelectiveSSM, Sequential, SharedRoutedMoE, SiLU, Sigmoid,
        SinusoidalPositionalEncoding, SlidingWindowAttention, Snake, SoftMarginLoss, Softplus, Softshrink,
        Softsign, SparseMoE, SpectralNormLinear, SpikeEvent, SqueezeExcitation2d, SurrogateKind,
        SwiGLU, TCNBlock, Tanh, TemporalConvNet, TitansBlock, TitansMemoryConfig,
        TitansNeuralMemory, TransformerDecoder, TransformerDecoderLayer, TransformerEncoder,
        TransformerEncoderLayer, TransfusionConfig, TransfusionModel, UNet2d, Upsample2d,
        UpsampleMode, VariationalAutoencoder, VisionLanguageActionModel, VisionTransformer,
        WeightNormLinear, alibi_bias, bradley_terry_reward_loss, ctc_loss, flash_attention,
        hinge_embedding_loss, margin_ranking_loss, multilabel_margin_loss, normalize_adjacency,
        soft_margin_loss, vae_kl_loss, vae_reparameterize,
    };
    pub use crate::ode::{
        FlowMatcher, JumpDiffusionEngine, LatentSdeConfig, LatentSdeEngine, NeuralJumpSDE,
        NeuralODE, OdeMethod, SdeTrajectory, odeint, odeint_trajectory,
    };
    pub use crate::optim::{
        Adagrad, Adam, CosineAnnealingLr, CosineAnnealingWarmRestarts, ExponentialLr, LBFGS,
        LinearWarmupCosineLr, Lion, Lookahead, ManifoldType, Muon, OneCycleLR, Optimizer,
        ReduceLROnPlateau, RiemannianAdam, RmsProp, ScheduleFreeAdamW, Sgd, StepLr, clip_grad_norm,
        clip_grad_value, newton_schulz_orthogonalize,
    };
    pub use crate::quant::{
        AWQLinear, AdaptiveQuantLinear, BitLinear, DynamicLlmQuantizer, DynamicPrecisionLevel,
        DynamicQuantPolicy, DynamicTelemetry, DynamicTransitionPlan, FP8Format, FP8Linear,
        FP8Tensor, GPTQLinear, PackedBitLinear, QuantScheme, QuantizedLinear, QuantizedTensor,
        TernaryPackedTensor, fp8_e4m3, fp8_e5m2,
    };
    pub use crate::quantum::{
        MatrixProductState, MpsSiteTensor, QaoaCircuit, QuantumCircuit, QuantumGate, VqeAnsatz,
    };
    pub use crate::reasoning::{
        AdaptiveConformalClassifier, AgentProposal, AlgorithmicEdge, AlgorithmicGraph,
        AlgorithmicSemiring, CausalVariable, ConformalEvaluator, ConsensusReport,
        ConsensusStrategy, DifferentiableILPEngine, MCTSNode, MCTSSearchEngine, NeuralBellmanFord,
        PointerGraphNetwork, PredictionInterval, ReasoningStep, ReasoningTrace, SearchResult,
        SelfCorrectionVerifier, SplitConformalRegressor, StructuralCausalModel, SwarmOrchestrator,
        SymbolicGpConfig, SymbolicGpEngine, SymbolicNode, TNorm, VerificationReport,
    };
    pub use crate::rl::{
        DPOConfig, DPOStepStats, DPOTrainer, GRPOConfig, GRPOStepStats, GRPOTrainer, PPOConfig,
        PPOStepStats, PPOTrainer, RuleReward, compute_group_advantages, grpo_loss,
    };
    pub use crate::robotics::{ACTConfig, ActionChunkingTransformer, FlowMatchingRoboticsPolicy};
    pub use crate::sampling::SamplingConfig;
    pub use crate::serialize::{
        GGMLType, GGUFFile, GGUFTensorInfo, GGUFValue, deserialize_safetensors, load_safetensors,
        save_gguf, save_safetensors, serialize_gguf, serialize_safetensors,
    };
    pub use crate::serving::{
        AsyncKVPipelineEngine, AttentionSinkConfig, BatchRequest, BlockAllocator, BlockTable,
        ContinuousBatchScheduler, DecodeRequest, DoubleBufferedKVCache, EagleConfig,
        EagleDraftHead, EagleSpeculativeEngine, InterleavedBatch, MedusaHeads, PagedKVCache,
        PhysicalBlockId, PrefillChunk, PrefillDecodeInterleaver, PrefillRequest, QuantKVDType,
        QuantizedPagedKVCache, ServingEngine, SpeculativeConfig, SpeculativeDecoder,
        SpeculativeMetrics, SpeculativeTree, SpeculativeTreeNode, StopSequenceMatcher,
        StreamingKVCache, StreamingUtf8Decoder, TreeAttentionMask, TreeSpeculativeDecoder,
        UnifiedSpeculativeEngine, VerificationOutcome, paged_attention, paged_attention_quantized,
    };
    pub use crate::sparse::{SparseCoo, SparseCsr};
    pub use crate::super_intelligence::{
        SuperIntelligenceConfig, SuperIntelligenceEngine, SuperIntelligenceMetrics,
    };
    pub use crate::tile_ir::{
        HardwareProfile, HardwareTarget, KernelAutoTuner, KernelFusionEngine, MemoryScope,
        TileConfig, TileKernelPlan, TileOp, TuneMetric,
    };
    pub use crate::tokenizer::{
        BOS_TOKEN_ID, BpeTokenizer, EOS_TOKEN_ID, ModalityKind, OmniSequence, OmniTokenSegment,
        OmniTokenizer, PAD_TOKEN_ID, UNK_TOKEN_ID,
    };
    pub use crate::trainer::{
        EarlyStopping, EpochMetrics, Trainer, TrainerConfig, TrainingHistory,
    };
    pub use crate::tuning::{ParamDomain, ParamSpace, ParamValue, Study, TrialConfig, TrialResult};
    pub use crate::vectorize::{
        DictVal, FastBM25Index, FastByteHistogramVectorizer, FastCountVectorizer,
        FastDeepSetsVectorizer, FastDictVectorizer, FastDiscretizer, FastHashingVectorizer,
        FastJsonTreeVectorizer, FastLsaVectorizer, FastMaxSimScorer, FastMinHashVectorizer,
        FastOneHotVectorizer, FastPathSignatureVectorizer, FastSimHashVectorizer,
        FastSphericalHarmonics, FastTextCharNGramVectorizer, FastTfidfVectorizer,
        FastTimeSeriesSpectralVectorizer, FastWeisfeilerLehmanVectorizer, KernelSpectralVectorizer,
        extract_char_ngrams, extract_ngrams, murmurhash3_32, tokenize_words,
    };
    pub use crate::video::{
        CausalConv3d, RectifiedFlowVideoDiT, RotaryEmbedding3D, VideoDiTBlock, VideoDiTConfig,
        VideoResBlock3d, VideoVAE3d,
    };
    pub use crate::vision::RotaryEmbedding2D;
    pub use crate::vision_ops::{affine_grid, box_iou, grid_sample, nms, roi_align};
    pub use crate::wasm::{WasmInferenceSession, WasmTensor};
    pub use crate::world_model::{UnifiedWorldModel, WorldModelConfig};
    pub use crate::{
        CustomOp, Generator, Tensor, checkpoint, enable_grad, fork_rng, get_rng_state, grad,
        initial_seed, is_grad_enabled, manual_seed, no_grad, s, seed, set_rng_state,
    };
}
