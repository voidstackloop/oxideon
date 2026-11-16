//! Physics & High-Energy PDE Solvers Showcase:
//! Demonstrates relativistic hydrodynamics, self-similar blast waves,
//! and plasma magnetic reconnection in Oxideon.

use oxideon::physics::{
    BjorkenQgpHydroExpansion, LawsonTripleProductIgnition, SedovTaylorBlastWave,
    SweetParkerPetschekReconnection,
};
use oxideon::prelude::*;

fn main() -> Result<()> {
    println!("================================================================================");
    println!("       OXIDEON v500.0.0 — MULTI-PHYSICS & PDE RELATIVISTIC DYNAMICS");
    println!("================================================================================");

    // 1. Sedov-Taylor Blast Wave
    println!("\n[1] Sedov-Taylor Self-Similar Point-Explosion Blast Wave:");
    let blast = SedovTaylorBlastWave::default();
    let energy_joules = 1.0e14; // 100 Terajoules
    let time_sec = 0.05; // 50 milliseconds
    let (radius, shock_vel) = blast.blast_radius_and_shock_velocity(energy_joules, time_sec)?;
    println!(
        "    Energy Input: {:.1e} J, Time: {:.3} s",
        energy_joules, time_sec
    );
    println!("    Shock Front Radius R(t): {:.2} m", radius);
    println!(
        "    Shock Front Velocity Us(t): {:.2} m/s (Mach {:.1})",
        shock_vel,
        shock_vel / 343.0
    );

    // 2. Relativistic Quark-Gluon Plasma (QGP) Bjorken Hydrodynamics
    println!("\n[2] Bjorken Boost-Invariant Relativistic QGP Expansion:");
    let tau_0 = 0.6; // fm/c
    let eps_0 = 30.0; // GeV/fm^3
    let qgp = BjorkenQgpHydroExpansion::new(tau_0, eps_0)?;
    let tau_f = 6.0; // fm/c
    let cs_sq = 1.0 / 3.0; // conformal EOS
    let (eps_f, temp_ratio) = qgp.energy_density_and_temp_ratio(tau_f, cs_sq)?;
    println!(
        "    Initial State at tau = {:.1} fm/c: eps = {:.1} GeV/fm^3",
        tau_0, eps_0
    );
    println!(
        "    Expanded State at tau = {:.1} fm/c: eps = {:.3} GeV/fm^3",
        tau_f, eps_f
    );
    println!(
        "    Temperature Cooling Ratio T(tau_f)/T0: {:.4}",
        temp_ratio
    );

    // 3. Sweet-Parker vs. Petschek Magnetic Reconnection
    println!("\n[3] Magnetohydrodynamic Plasma Reconnection Rates:");
    let lundquist_s = 1.0e8; // High Lundquist number astrophysical plasma
    let plasma = SweetParkerPetschekReconnection::new(lundquist_s)?;
    let (m_sp, m_petschek, plasmoid_unstable) = plasma.reconnection_rates_and_plasmoid_regime();
    println!("    Lundquist Number S: {:.1e}", lundquist_s);
    println!("    Sweet-Parker Reconnection Rate M_SP: {:.2e}", m_sp);
    println!(
        "    Petschek Fast Reconnection Rate M_P:  {:.4}",
        m_petschek
    );
    println!(
        "    Plasmoid Tearing Instability Active: {}",
        plasmoid_unstable
    );

    // 4. Lawson Triple Product Criterion for Nuclear Fusion
    println!("\n[4] Lawson Triple Product Fusion Ignition Criterion:");
    let fusion = LawsonTripleProductIgnition::default();
    let density_n = 2.5e20; // m^-3
    let temp_kev = 15.0; // keV (~170 million Kelvin)
    let tau_e = 3.2; // seconds energy confinement
    let reactivity = 3.0e-22; // m^3 / s at 15 keV
    let (triple_prod, threshold, ignited) =
        fusion.triple_product_and_ignition(density_n, temp_kev, tau_e, reactivity)?;
    println!(
        "    Achieved Triple Product n * T * tau_E: {:.2e} keV s m^-3",
        triple_prod
    );
    println!("    Ignition Threshold: {:.2e} keV s m^-3", threshold);
    println!(
        "    Thermonuclear Ignition Achieved: {}",
        if ignited {
            "YES (Self-Sustaining Q -> inf)"
        } else {
            "NO"
        }
    );

    println!("\n[OK] Physics & PDE Solvers Showcase completed successfully.");
    Ok(())
}
