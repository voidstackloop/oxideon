//! Hardware Accelerators, Optical Computing & Photonic Neural Architectures.

pub mod coherent_photonic;
pub mod photonic;

pub use coherent_photonic::{CoherentPhotonicCircuit, WaveguideProfile};
pub use photonic::{MachZehnderInterferometer, PhotonicLinear, PhotonicMesh, SaturableAbsorber};
