//! Bare-Metal Compute Drivers & Hardware Assemblers.
//!
//! Includes:
//! - Apple Metal MSL Kernel Emitter ([`MslKernelEmitter`]) for Apple Silicon
//! - NVIDIA PTX Assembler Emitter ([`PtxKernelEmitter`])
//! - Headless Vulkan SPIR-V Driver ([`VulkanSpirvKernel`])

pub mod metal;
pub mod ptx;
pub mod vulkan;
pub mod wgsl;

pub use metal::MslKernelEmitter;
pub use ptx::PtxKernelEmitter;
pub use vulkan::VulkanSpirvKernel;
pub use wgsl::WgslKernelBuilder;
