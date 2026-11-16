#!/usr/bin/env python3
"""Oxideon v500.0.0 — Python Physics, Quantum & Cryptographic Science Showcase.

Demonstrates:
1. Sedov-Taylor self-similar point explosion blast waves
2. Post-quantum Regev Learning With Errors (LWE) lattice security
3. BKZ-beta lattice reduction and root Hermite factor analysis
4. Groth16 QAP zero-knowledge pairing verification complexity
5. Lawson triple product thermonuclear fusion criterion
"""

import oxideon as ox


def main():
    print("=" * 80)
    print("    OXIDEON v500.0.0 — PYTHON PHYSICS, QUANTUM & LATTICE CRYPTO SHOWCASE")
    print("=" * 80)

    # 1. Sedov-Taylor Relativistic / High-Energy Blast Waves
    print("\n[1] Sedov-Taylor Self-Similar Point Explosion Dynamics:")
    blast = ox.SedovTaylorBlastWave(1.15, 1.225)
    energy_joules = 1.0e14  # 100 Terajoules
    time_sec = 0.05         # 50 ms
    radius, shock_vel = blast.blast_radius_and_shock_velocity(energy_joules, time_sec)
    print(f"    Energy Release:  {energy_joules:.1e} J over {time_sec * 1000.0:.1f} ms")
    print(f"    Shock Radius:    {radius:.2f} m")
    print(f"    Shock Velocity:  {shock_vel:.2f} m/s (Mach {shock_vel / 343.0:.1f})")

    # 2. Regev Learning With Errors (LWE) Lattice Security
    print("\n[2] Regev Learning With Errors (LWE) Post-Quantum Cryptography:")
    lwe = ox.RegevLweLatticeSecurity(512, 3329)  # Kyber-512 lattice regime
    alpha_noise = 0.005
    sigma, gamma_approx, valid = lwe.noise_std_approx_factor_and_regev_check(alpha_noise)
    print(f"    Lattice Dimension: 512, Modulus: 3329")
    print(f"    Gaussian Noise Std Dev sigma: {sigma:.4f}")
    print(f"    GapSVP Approximation Factor:  {gamma_approx:.1e}")
    print(f"    Quantum Hardness Guarantee (sigma > 2*sqrt(n)): {valid}")

    # 3. LLL and BKZ Lattice Reduction
    print("\n[3] Block Korkine-Zolotarev (BKZ-beta) Lattice Reduction:")
    bkz = ox.LllBkzLatticeReduction(60)
    delta0, core_svp_bits = bkz.root_hermite_factor_and_core_svp_bits()
    print(f"    BKZ Blocksize beta: 60")
    print(f"    Root Hermite Factor delta_0(beta): {delta0:.6f}")
    print(f"    Core-SVP Hardness: {core_svp_bits:.1f} bits of security")

    # 4. Groth16 Quadratic Arithmetic Program (QAP) Zero-Knowledge SNARK
    print("\n[4] Groth16 zk-SNARK Pairing Verification Complexity:")
    snark = ox.Groth16QapSnarkAnalyzer(131072, 8)
    proof_elems, pairings, ntt_ops = snark.proof_elements_pairings_and_ntt_ops()
    print(f"    R1CS Constraints: 131,072, Public Inputs: 8")
    print(f"    Proof Size:       {proof_elems} group elements (constant O(1))")
    print(f"    Pairing Checks:   {pairings} bilinear pairings")
    print(f"    Prover NTT Ops:   {ntt_ops:.2e}")

    # 5. Lawson Triple Product Thermonuclear Fusion Criterion
    print("\n[5] Lawson Fusion Triple Product & Alpha Heating Ignition:")
    fusion = ox.LawsonTripleProductIgnition(3500.0)  # 3.5 MeV alpha particle heating
    density_n = 2.5e20      # m^-3
    temp_kev = 15.0         # keV (~170 MK)
    tau_e = 3.2             # seconds
    reactivity = 3.0e-22    # m^3/s
    achieved, threshold, ignited = fusion.triple_product_and_ignition(density_n, temp_kev, tau_e, reactivity)
    print(f"    Achieved Triple Product: {achieved:.2e} keV s m^-3")
    print(f"    Ignition Threshold:      {threshold:.2e} keV s m^-3")
    print(f"    Thermonuclear Ignition:  {'YES (Self-Sustaining Q -> inf)' if ignited else 'NO'}")

    print("\n[OK] Python Physics, Quantum & Science Showcase completed successfully.")


if __name__ == "__main__":
    main()
