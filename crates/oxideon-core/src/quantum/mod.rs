//! Quantum-Classical Hybrid Neural Computing & Differentiable Simulation.
//!
//! Provides full $N$-qubit state vector simulation, universal gate operations,
//! expectation values, and analytical parameter-shift gradient computation.

pub mod circuit;
pub mod continuous_variable;
pub mod mitigation;
pub mod mps;
pub mod qaoa;
pub mod qec;
pub mod qgnn;
pub mod tqft;

pub use circuit::{QuantumCircuit, QuantumGate};
pub use continuous_variable::{GaussianCvCircuit, matrix_hafnian};
pub use mitigation::{CliffordDataRegression, ExtrapolationModel, ZneExtrapolator};
pub use mps::{MatrixProductState, MpsSiteTensor};
pub use qaoa::{QaoaCircuit, VqeAnsatz};
pub use qec::{Pauli, QecEngine, Stabilizer, StabilizerCode};
pub use qgnn::{QGNNGradients, QuantumGraphNeuralNetwork};
pub use tqft::Su2ChernSimonsTqft;

pub mod braid_anyons;
pub use braid_anyons::FibonacciAnyonBraider;

pub mod syk_chaos;
pub use syk_chaos::SykQuantumChaosEngine;

pub mod quantum_ldpc;
pub use quantum_ldpc::QuantumLdpcCodeEngine;

pub mod magic_state_distillation;
pub use magic_state_distillation::MagicStateDistillationEngine;

pub mod mera_holography;
pub use mera_holography::MeraEntanglementEngine;

pub mod quantum_darwinism;
pub use quantum_darwinism::QuantumDarwinismEngine;

pub mod floquet_time_crystals;
pub use floquet_time_crystals::FloquetTimeCrystalEngine;

pub mod majorana_kitaev_chain;
pub use majorana_kitaev_chain::KitaevMajoranaChainEngine;

pub mod quantum_metrology;
pub use quantum_metrology::QuantumMetrologyEngine;

pub mod quantum_thermodynamics;
pub use quantum_thermodynamics::QuantumOttoEngine;

pub mod shadow_tomography;
pub use shadow_tomography::ClassicalShadowTomography;

pub mod jaynes_cummings_cavity_qed;
pub use jaynes_cummings_cavity_qed::JaynesCummingsCavityQED;

pub mod dicke_tavis_cummings;
pub use dicke_tavis_cummings::DickeTavisCummingsSuperradiance;

pub mod hong_ou_mandel_dip;
pub use hong_ou_mandel_dip::HongOuMandelInterferometer;

pub mod glauber_mandel_antibunching;
pub use glauber_mandel_antibunching::GlauberPhotonCoherenceEngine;

pub mod eit_dark_state_polariton;
pub use eit_dark_state_polariton::ElectromagneticallyInducedTransparency;

pub mod corkum_hhg_attosecond;
pub use corkum_hhg_attosecond::CorkumHighHarmonicAttosecond;

pub mod doppler_sisyphus_cooling;
pub use doppler_sisyphus_cooling::DopplerSisyphusLaserCooling;

pub mod feshbach_bec_bcs_crossover;
pub use feshbach_bec_bcs_crossover::FeshbachScatteringResonance;

pub mod cavity_optomechanics_sql;
pub use cavity_optomechanics_sql::CavityOptomechanicsBackaction;

pub mod transmon_josephson_qubit;
pub use transmon_josephson_qubit::TransmonCircuitQEDQubit;
