//! Cryptographic and Verifiable Machine Learning Module.
//!
//! Includes Zero-Knowledge Machine Learning ([`zkml`]) for verifiable inference proofs,
//! arithmetic circuits over finite fields, and cryptographic commitments.
//!
//! # Security warning: educational and research use only
//!
//! Nothing in this module is a vetted cryptographic implementation. Do not use it to protect real data.
//!
//! - [`fhe`]: `keygen` and `encrypt` are fully deterministic (fixed-pattern secret key, public polynomial,
//!   ephemeral key and noise), so every key pair is identical and anyone can recompute the secret key.
//!   The parameters (`N` of a few dozen, `q` about 2^40) are also far below any security level.
//! - [`zkml`]: a "proof" is an unsalted SHA-256 hash of the quantised inputs plus a Fiat-Shamir challenge
//!   derived from those public hashes. `verify_proof` only recomputes that hash and never checks a constraint,
//!   so the proof is neither sound nor zero-knowledge.
//! - [`lattice_lwe`] and [`shamir_feldman_threshold_mpc`] take caller-supplied randomness and use plain `i64`
//!   arithmetic: not constant-time, and products overflow for large moduli.
//! - [`kzg_polynomial_commitment`], [`fri_reed_solomon_stark`] and [`groth16_qap_zk_snark`] are polynomial and
//!   R1CS helpers (opening quotient, folding, witness check). They contain no commitments, group operations
//!   or pairings, so they are not the schemes their names suggest.
//! - The crate RNG (`crate::rng`) is SplitMix64, which is explicitly not cryptographically secure.

pub mod fhe;
pub mod r1cs_qap;
pub mod zkml;

pub use fhe::{
    CKKS_DEFAULT_MODULUS, CKKS_DEFAULT_SCALE, CkksContext, FheCiphertext, FhePoly, FhePublicKey,
    FheSecretKey,
};
pub use r1cs_qap::{DifferentiableR1csQap, R1csConstraint};
pub use zkml::{FieldElement, LinearCombination, R1CSConstraint, ZkMlEngine, ZkMlProof};

pub mod lattice_lwe;
pub use lattice_lwe::LatticeLweEngine;

pub mod groth16_qap_zk_snark;
pub use groth16_qap_zk_snark::QuadraticArithmeticProgramEngine;

pub mod fri_reed_solomon_stark;
pub use fri_reed_solomon_stark::FastReedSolomonIopEngine;

pub mod kzg_polynomial_commitment;
pub use kzg_polynomial_commitment::KzgPolynomialCommitment;

pub mod shamir_feldman_threshold_mpc;
pub use shamir_feldman_threshold_mpc::ShamirSecretSharingFp;

pub mod goppa_algebraic_geometry_codes;
pub use goppa_algebraic_geometry_codes::TsfasmanVladutZinkAgCode;

pub mod arikan_polar_codes_capacity;
pub use arikan_polar_codes_capacity::ArikanPolarCodeEngine;

pub mod shannon_rate_distortion_blahut;
pub use shannon_rate_distortion_blahut::RateDistortionBlahutArimoto;

pub mod slepian_wolf_wyner_ziv;
pub use slepian_wolf_wyner_ziv::SlepianWolfDistributedCoding;

pub mod dwork_renyi_differential_privacy;
pub use dwork_renyi_differential_privacy::RenyiDifferentialPrivacyAccountant;
