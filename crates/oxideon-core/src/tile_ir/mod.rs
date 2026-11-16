//! Embedded Tile-IR and Dynamic Kernel Fusion JIT Emitter.

pub mod autotune;
pub mod fusion;

pub use autotune::{HardwareProfile, HardwareTarget, KernelAutoTuner, TileConfig, TuneMetric};
pub use fusion::{KernelFusionEngine, MemoryScope, TileKernelPlan, TileOp};
