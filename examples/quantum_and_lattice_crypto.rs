//! Quantum Computing, Lattice Cryptography & Zero-Knowledge Proofs Showcase:
//! Demonstrates Regev LWE post-quantum lattice security, BKZ root Hermite factor,
//! and Groth16 QAP zero-knowledge pairing verification in Oxideon.

use oxideon::prelude::*;
use oxideon::science::{Groth16QapSnarkAnalyzer, LllBkzLatticeReduction, RegevLweLatticeSecurity};

fn main() -> Result<()> {
    println!("================================================================================");
    println!("       OXIDEON v500.0.0 — QUANTUM, LATTICES & ZERO-KNOWLEDGE PROOFS");
    println!("================================================================================");

    // 1. Post-Quantum Lattice Security with Regev Learning With Errors (LWE)
    println!("\n[1] Regev Learning With Errors (LWE) Lattice Security Analysis:");
    let lwe = RegevLweLatticeSecurity::new(512, 3329)?; // Kyber-512 parameter regime
    let alpha_noise = 0.005;
    let (sigma, gamma_approx, valid) = lwe.noise_std_approx_factor_and_regev_check(alpha_noise)?;
    println!(
        "    Lattice Dimension n: {}, Modulus q: {}",
        lwe.dim_n, lwe.modulus_q
    );
    println!("    Discrete Gaussian Noise Width sigma: {:.4}", sigma);
    println!(
        "    Worst-Case GapSVP Approximation Factor: {:.1e}",
        gamma_approx
    );
    println!(
        "    Quantum Hardness Reduction Condition (sigma > 2*sqrt(n)): {}",
        if valid { "SATISFIED" } else { "VIOLATED" }
    );

    // 2. Lenstra-Lenstra-Lovász (LLL) and Block Korkine-Zolotarev (BKZ) Reduction
    println!("\n[2] BKZ-beta Lattice Basis Reduction & Root Hermite Factor:");
    let bkz = LllBkzLatticeReduction::new(60)?;
    let (delta0, core_svp_bits) = bkz.root_hermite_factor_and_core_svp_bits();
    println!("    BKZ Blocksize beta: {}", bkz.blocksize_beta);
    println!("    Root Hermite Factor delta_0(beta): {:.6}", delta0);
    println!(
        "    Classical Core-SVP Hardness: {:.1} bits of security",
        core_svp_bits
    );

    // 3. Groth16 QAP Zero-Knowledge SNARK Pairing Verification
    println!("\n[3] Groth16 Quadratic Arithmetic Program (QAP) zk-SNARK Analyzer:");
    let snark = Groth16QapSnarkAnalyzer::new(131072, 8)?; // 128k constraints, 8 public inputs
    let (elements, pairings, ntt_ops) = snark.proof_elements_pairings_and_ntt_ops();
    println!(
        "    Circuit Constraints: {}, Public Inputs: {}",
        snark.num_constraints_n, snark.num_public_inputs_l
    );
    println!(
        "    Proof Size: {} group elements (constant G1/G2 pairing size)",
        elements
    );
    println!(
        "    Verification Complexity: {} bilinear pairings",
        pairings
    );
    println!("    Prover FFT/NTT Workload: {:.2e} operations", ntt_ops);

    println!("\n[OK] Quantum, Lattice & ZK-Proofs Showcase completed successfully.");
    Ok(())
}
