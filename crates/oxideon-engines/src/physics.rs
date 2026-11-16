//! Differentiable Physics & Partial Differential Equation (PDE) primitives for
//! Physics-Informed Neural Networks (**PINNs**, Raissi et al. 2019):
//! - **Differential Operators**: [`spatial_gradient`] ($\nabla_x u$), [`divergence`] ($\nabla \cdot v$), and [`laplacian`] ($\Delta u = \sum_i \frac{\partial^2 u}{\partial x_i^2}$) via higher-order autograd (`create_graph = true`)
//! - **PDE Residuals & PINN Losses**: [`helmholtz_pinn_residual`] and [`poisson_pinn_loss`]

use crate::error::{Error, Result};
use crate::tensor::{Tensor, grad};

pub mod ads_cft;
pub mod contact_geometry;
pub mod hamiltonian;
pub mod heisenberg;
pub mod lie_poisson;
pub mod non_hermitian;
pub mod rmhd;
pub mod soliton;
pub mod stochastic_thermo;
pub mod symplectic_topology;
pub mod thermo_relativity;

pub use ads_cft::AdsCftHolographicEngine;
pub use contact_geometry::ContactHamiltonianSystem;
pub use hamiltonian::HamiltonianNeuralEngine;
pub use heisenberg::{HeisenbergLieAlgebra, HeisenbergPoint, HypoellipticDiffusion3D};
pub use lie_poisson::{LiePoissonState, LiePoissonSystem, RigidBodyInertia};
pub use non_hermitian::{ComplexEigenvalue, ExceptionalPointReport, NonHermitianHamiltonian2x2};
pub use rmhd::RmhdPrimitiveState;
pub use soliton::{SolitonPhysics, SpatialGrid1D, SplitStepNlsEngine};
pub use stochastic_thermo::{JarzynskiReport, StochasticThermoEngine};
pub use symplectic_topology::SymplecticTopologyEngine;
pub use thermo_relativity::{KerrSpacetime, LangevinThermodynamics};

/// Computes the exact batched spatial gradient $\nabla_x u(x)$ of shape `[B, D]` for a scalar
/// field $u: \mathbb{R}^{B \times D} \to \mathbb{R}^{B \times 1}$ (or `[B]`) with `create_graph = true`
/// so second-order PDE operators (such as [`laplacian`]) can differentiate through the result.
pub fn spatial_gradient<F>(u_fn: F, x: &Tensor) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    if x.ndim() != 2 {
        return Err(Error::Msg(format!(
            "spatial_gradient expects 2D coordinate tensor [B, D], got {:?}",
            x.shape()
        )));
    }
    let x_var = if x.requires_grad() {
        x.clone()
    } else {
        x.clone().requires_grad_(true)
    };
    let u = u_fn(&x_var)?;
    let u_sum = u.sum()?;
    let du_dx = grad(&[u_sum], std::slice::from_ref(&x_var), None, true)?
        .pop()
        .flatten()
        .ok_or_else(|| {
            Error::Autograd("spatial_gradient: field output did not depend on x".into())
        })?;
    Ok(du_dx)
}

/// Computes the exact batched divergence $\nabla \cdot v(x) = \sum_{d=0}^{D-1} \frac{\partial v_d(x)}{\partial x_d}$
/// of shape `[B, 1]` for a vector field $v: \mathbb{R}^{B \times D} \to \mathbb{R}^{B \times D}$
/// with `create_graph = true`.
pub fn divergence<F>(v_fn: F, x: &Tensor) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    if x.ndim() != 2 {
        return Err(Error::Msg(format!(
            "divergence expects 2D coordinate tensor [B, D], got {:?}",
            x.shape()
        )));
    }
    let b = x.shape()[0];
    let d = x.shape()[1];
    let x_var = if x.requires_grad() {
        x.clone()
    } else {
        x.clone().requires_grad_(true)
    };
    let v = v_fn(&x_var)?;
    if v.shape() != [b, d] {
        return Err(Error::ShapeMismatch {
            op: "divergence",
            expected: vec![b, d],
            got: v.shape().to_vec(),
        });
    }

    let mut div_acc: Option<Tensor> = None;
    for dim in 0..d {
        let v_d = v.narrow(1, dim, 1)?;
        let v_d_sum = v_d.sum()?;
        let grad_x = grad(&[v_d_sum], std::slice::from_ref(&x_var), None, true)?
            .pop()
            .flatten()
            .ok_or_else(|| {
                Error::Autograd("divergence: vector field did not depend on x".into())
            })?;
        let dv_d_dxd = grad_x.narrow(1, dim, 1)?;
        div_acc = Some(match div_acc {
            Some(acc) => acc.add(&dv_d_dxd)?,
            None => dv_d_dxd,
        });
    }
    div_acc.ok_or_else(|| Error::Msg("divergence requires D >= 1".into()))
}

/// Computes the exact batched Laplacian $\Delta u(x) = \sum_{d=0}^{D-1} \frac{\partial^2 u(x)}{\partial x_d^2}$
/// of shape `[B, 1]` for a scalar field $u: \mathbb{R}^{B \times D} \to \mathbb{R}^{B \times 1}$ (or `[B]`)
/// using second-order reverse-mode automatic differentiation (`create_graph = true`).
pub fn laplacian<F>(u_fn: F, x: &Tensor) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    if x.ndim() != 2 {
        return Err(Error::Msg(format!(
            "laplacian expects 2D coordinate tensor [B, D], got {:?}",
            x.shape()
        )));
    }
    let x_var = if x.requires_grad() {
        x.clone()
    } else {
        x.clone().requires_grad_(true)
    };
    let u = u_fn(&x_var)?;
    laplacian_of(&u, &x_var)
}

/// Laplacian of an already-evaluated field `u` (shape `[B, *]`) w.r.t. 2D
/// coordinates `x_var` (`[B, D]`), sharing one `u_fn` evaluation with callers
/// like [`helmholtz_pinn_residual`] that also need `u` itself.
pub fn laplacian_of(u: &Tensor, x_var: &Tensor) -> Result<Tensor> {
    if x_var.ndim() != 2 {
        return Err(Error::Msg(format!(
            "laplacian expects 2D coordinate tensor [B, D], got {:?}",
            x_var.shape()
        )));
    }
    let d = x_var.shape()[1];
    let u_sum = u.sum()?;
    let du = grad(&[u_sum], std::slice::from_ref(x_var), None, true)?
        .pop()
        .flatten()
        .ok_or_else(|| Error::Autograd("laplacian: field output did not depend on x".into()))?;

    let mut lap_acc: Option<Tensor> = None;
    for dim in 0..d {
        let du_d = du.narrow(1, dim, 1)?;
        let du_d_sum = du_d.sum()?;
        let d2u_dx = grad(&[du_d_sum], std::slice::from_ref(x_var), None, true)?
            .pop()
            .flatten()
            .ok_or_else(|| {
                Error::Autograd("laplacian: first derivative did not depend on x".into())
            })?;
        let d2u_dxd2 = d2u_dx.narrow(1, dim, 1)?;
        lap_acc = Some(match lap_acc {
            Some(acc) => acc.add(&d2u_dxd2)?,
            None => d2u_dxd2,
        });
    }
    lap_acc.ok_or_else(|| Error::Msg("laplacian requires D >= 1".into()))
}

/// Computes the Helmholtz PDE residual $r(x) = \Delta u(x) + k^2 u(x)$ of shape `[B, 1]`
/// on collocation points `x` (`[B, D]`).
pub fn helmholtz_pinn_residual<F>(u_fn: F, x: &Tensor, wavenumber_k: f64) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    if x.ndim() != 2 {
        return Err(Error::Msg(format!(
            "helmholtz_pinn_residual: x must be [B, D], got {:?}",
            x.shape()
        )));
    }
    let x_var = if x.requires_grad() {
        x.clone()
    } else {
        x.clone().requires_grad_(true)
    };
    let u = u_fn(&x_var)?.reshape(&[x_var.shape()[0], 1])?;
    let lap = laplacian_of(&u, &x_var)?;
    lap.add(&u.mul_scalar(wavenumber_k * wavenumber_k)?)
}

/// Computes the composite Physics-Informed Neural Network (`PINN`) objective for the Poisson
/// equation $-\Delta u(x) = f(x)$ on `interior_x` (`[B_int, D]`) with Dirichlet boundary
/// condition $u(x) = g(x)$ on `boundary_x` (`[B_bc, D]`):
/// $$\mathcal{L}_{\text{PINN}} = \|\Delta u(x_{\text{int}}) + f(x_{\text{int}})\|_2^2 + \lambda_{\text{bc}} \|u(x_{\text{bc}}) - u_{\text{bc}}\|_2^2$$
pub fn poisson_pinn_loss<F>(
    u_fn: F,
    interior_x: &Tensor,
    interior_rhs_f: &Tensor,
    boundary_x: &Tensor,
    boundary_u: &Tensor,
    bc_weight: f64,
) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    if interior_x.ndim() != 2 || boundary_x.ndim() != 2 {
        return Err(Error::Msg(format!(
            "poisson_pinn_loss: interior_x/boundary_x must be [B, D], got {:?}/{:?}",
            interior_x.shape(),
            boundary_x.shape()
        )));
    }
    let b_int = interior_x.shape()[0];
    let b_bc = boundary_x.shape()[0];
    let lap = laplacian(&u_fn, interior_x)?;
    let rhs = interior_rhs_f.reshape(&[b_int, 1])?;
    let pde_res = lap.add(&rhs)?.square()?.mean()?;

    let u_bc_pred = u_fn(boundary_x)?.reshape(&[b_bc, 1])?;
    let u_bc_targ = boundary_u.reshape(&[b_bc, 1])?;
    let bc_loss = u_bc_pred.sub(&u_bc_targ)?.square()?.mean()?;

    pde_res.add(&bc_loss.mul_scalar(bc_weight)?)
}

pub mod relativistic_lbm;
pub use relativistic_lbm::RelativisticLbm1D;

pub mod cavity_qed;
pub use cavity_qed::JaynesCummingsCavityQed;

pub mod twistor;
pub use twistor::PenroseTwistorEngine;

pub mod loop_quantum_gravity;
pub use loop_quantum_gravity::SpinNetworkLqgEngine;

pub mod causal_dynamical_triangulations;
pub use causal_dynamical_triangulations::CausalTriangulationEngine;

pub mod amplituhedron;
pub use amplituhedron::AmplituhedronGrassmannianEngine;

pub mod celestial_holography;
pub use celestial_holography::CelestialHolographyEngine;

pub mod swampland_conjectures;
pub use swampland_conjectures::SwamplandVerifierEngine;

pub mod conformal_bootstrap;
pub use conformal_bootstrap::ConformalBootstrapEngine;

pub mod sachdev_kitaev_wormhole;
pub use sachdev_kitaev_wormhole::TraversableWormholeEngine;

pub mod resurgence_transseries;
pub use resurgence_transseries::BorelResurgenceEngine;

pub mod exact_wkb;
pub use exact_wkb::ExactWkbQuantizationEngine;

pub mod kolmogorov_turbulence;
pub use kolmogorov_turbulence::KolmogorovCascadeEngine;

pub mod kraichnan_2d_enstrophy;
pub use kraichnan_2d_enstrophy::KraichnanEnstrophyCascade;

pub mod alfven_mhd_dynamo;
pub use alfven_mhd_dynamo::AlfvenMhdDynamoEngine;

pub mod quasi_geostrophic_rossby;
pub use quasi_geostrophic_rossby::CharneyRossbyWaveEngine;

pub mod rayleigh_benard_lorenz;
pub use rayleigh_benard_lorenz::RayleighBenardConvection;

pub mod kelvin_helmholtz_rayleigh_taylor;
pub use kelvin_helmholtz_rayleigh_taylor::HydrodynamicInstabilityEngine;

pub mod vlasov_landau_damping;
pub use vlasov_landau_damping::VlasovLandauDampingEngine;

pub mod grad_shafranov_tokamak;
pub use grad_shafranov_tokamak::GradShafranovTokamakEngine;

pub mod superfluid_gross_pitaevskii;
pub use superfluid_gross_pitaevskii::GrossPitaevskiiVortexEngine;

pub mod fractional_quantum_hall_laughlin;
pub use fractional_quantum_hall_laughlin::LaughlinFqheEngine;

pub mod haldane_chern_insulator;
pub use haldane_chern_insulator::HaldaneHoneycombModel;

pub mod kondo_anderson_rg;
pub use kondo_anderson_rg::KondoScreeningEngine;

pub mod bcs_eliashberg_superconductivity;
pub use bcs_eliashberg_superconductivity::BcsGapEquationEngine;

pub mod hubbard_mott_gutzwiller;
pub use hubbard_mott_gutzwiller::BrinkmanRiceGutzwillerEngine;

pub mod berezinskii_kosterlitz_thouless;
pub use berezinskii_kosterlitz_thouless::BktVortexUnbindingEngine;

pub mod weyl_semimetal_chiral_anomaly;
pub use weyl_semimetal_chiral_anomaly::WeylChiralAnomalyEngine;

pub mod twistronics_bistritzer_macdonald;
pub use twistronics_bistritzer_macdonald::MagicAngleGrapheneEngine;

pub mod skyrmion_dzyaloshinskii_moriya;
pub use skyrmion_dzyaloshinskii_moriya::MagneticSkyrmionEngine;

pub mod kam_kolmogorov_arnold_moser;
pub use kam_kolmogorov_arnold_moser::KamDiophantineTorusEngine;

pub mod nekhoroshev_arnold_diffusion;
pub use nekhoroshev_arnold_diffusion::NekhoroshevExponentialStability;

pub mod poincare_cartan_lagrange_l4_l5;
pub use poincare_cartan_lagrange_l4_l5::CircularRestrictedThreeBodyEngine;

pub mod chirikov_standard_map_chaos;
pub use chirikov_standard_map_chaos::ChirikovStandardMapEngine;

pub mod teukolsky_kerr_quasinormal;
pub use teukolsky_kerr_quasinormal::KerrQuasinormalModeEngine;

pub mod penrose_superradiance_ergosphere;
pub use penrose_superradiance_ergosphere::BlandfordZnajekPenroseEngine;

pub mod tolman_oppenheimer_volkoff;
pub use tolman_oppenheimer_volkoff::TolmanOppenheimerVolkoffEngine;

pub mod chandrasekhar_lane_emden;
pub use chandrasekhar_lane_emden::ChandrasekharWhiteDwarfEngine;

pub mod peters_mathews_chirp_gw;
pub use peters_mathews_chirp_gw::GravitationalWaveChirpEngine;

pub mod sachs_wolfe_cmb_acoustic;
pub use sachs_wolfe_cmb_acoustic::CmbAcousticPeakEngine;

pub mod sunyaev_zeldovich_kompaneets;
pub use sunyaev_zeldovich_kompaneets::SunyaevZeldovichKompaneetsEngine;

pub mod press_schechter_halo_mass;
pub use press_schechter_halo_mass::PressSchechterHaloEngine;

pub mod hawking_page_ads_black_hole;
pub use hawking_page_ads_black_hole::HawkingPagePhaseTransition;

pub mod ogden_mooney_rivlin_hyperelastic;
pub use ogden_mooney_rivlin_hyperelastic::HyperelasticStrainEnergyEngine;

pub mod griffith_rice_j_integral_fracture;
pub use griffith_rice_j_integral_fracture::RiceJIntegralFractureEngine;

pub mod von_mises_drucker_prager_plasticity;
pub use von_mises_drucker_prager_plasticity::J2PlasticityReturnMapping;

pub mod mindlin_reissner_kirchhoff_plates;
pub use mindlin_reissner_kirchhoff_plates::KirchhoffLovePlateEngine;

pub mod biot_poroelasticity_darcy;
pub use biot_poroelasticity_darcy::BiotPoroelasticConsolidation;

pub mod eshelby_inclusion_homogenization;
pub use eshelby_inclusion_homogenization::EshelbyInclusionMoriTanaka;

pub mod pendry_transformation_optics_cloak;
pub use pendry_transformation_optics_cloak::PendryMetamaterialCloakEngine;

pub mod yablonovitch_photonic_crystal_bandgap;
pub use yablonovitch_photonic_crystal_bandgap::PhotonicBraggQuarterWaveStack;

pub mod kane_lubensky_topological_mechanics;
pub use kane_lubensky_topological_mechanics::KaneLubenskyIsostaticEngine;

pub mod schwinger_euler_heisenberg_qed;
pub use schwinger_euler_heisenberg_qed::SchwingerEulerHeisenbergQED;

pub mod casimir_lifshitz_vacuum;
pub use casimir_lifshitz_vacuum::CasimirLifshitzVacuumEngine;

pub mod unruh_dewitt_rindler;
pub use unruh_dewitt_rindler::UnruhRindlerBogoliubovEngine;

pub mod msw_neutrino_oscillation;
pub use msw_neutrino_oscillation::MswNeutrinoOscillationEngine;

pub mod wilson_lattice_qcd_plaquette;
pub use wilson_lattice_qcd_plaquette::WilsonLatticeGaugeQCD;

pub mod qcd_asymptotic_freedom_beta;
pub use qcd_asymptotic_freedom_beta::QcdAsymptoticFreedomEngine;

pub mod adler_bell_jackiw_anomaly;
pub use adler_bell_jackiw_anomaly::AbjChiralAnomalyEngine;

pub mod kibble_zurek_cosmological_defects;
pub use kibble_zurek_cosmological_defects::KibbleZurekDefectScaling;

pub mod sphaleron_electroweak_baryogenesis;
pub use sphaleron_electroweak_baryogenesis::ElectroweakSphaleronBaryogenesis;

pub mod kanamori_gutenberg_seismology;
pub use kanamori_gutenberg_seismology::KanamoriGutenbergSeismicEngine;

pub mod aki_richards_zoeppritz_seismic;
pub use aki_richards_zoeppritz_seismic::AkiRichardsSeismicAvoEngine;

pub mod backus_gilbert_geophysical_inverse;
pub use backus_gilbert_geophysical_inverse::BackusGilbertInverseResolution;

pub mod budyko_sellers_milankovitch_climate;
pub use budyko_sellers_milankovitch_climate::BudykoSellersClimateBifurcation;

pub mod chandrasekhar_eddington_radiative;
pub use chandrasekhar_eddington_radiative::EddingtonChandrasekharRadiative;

pub mod elsasser_geodynamo_core;
pub use elsasser_geodynamo_core::ElsasserGeodynamoCoreEngine;

pub mod thermal_wind_eady_baroclinic;
pub use thermal_wind_eady_baroclinic::ThermalWindEadyBaroclinic;

pub mod stommel_sverdrup_ocean_gyre;
pub use stommel_sverdrup_ocean_gyre::StommelSverdrupOceanCirculation;

pub mod toomre_hill_planetary_disk;
pub use toomre_hill_planetary_disk::ToomreHillPlanetaryDisk;

pub mod landauer_buttiker_mesoscopic;
pub use landauer_buttiker_mesoscopic::LandauerButtikerBallisticTransport;

pub mod datta_das_rashba_spintronics;
pub use datta_das_rashba_spintronics::DattaDasRashbaSpintronics;

pub mod slonczewski_spin_transfer_torque;
pub use slonczewski_spin_transfer_torque::SlonczewskiSpinTransferTorque;

pub mod ambegaokar_baratoff_josephson;
pub use ambegaokar_baratoff_josephson::AmbegaokarBaratoffJosephson;

pub mod shockley_queisser_solar;
pub use shockley_queisser_solar::ShockleyQueisserPhotovoltaic;

pub mod chua_strukov_memristor;
pub use chua_strukov_memristor::ChuaStrukovMemristorSynapse;

pub mod tsu_esaki_resonant_tunneling;
pub use tsu_esaki_resonant_tunneling::TsuEsakiResonantTunneling;

pub mod seebeck_thermoelectric_zt;
pub use seebeck_thermoelectric_zt::ThermoelectricFigureOfMeritZT;

pub mod dlvo_debye_huckel_colloid;
pub use dlvo_debye_huckel_colloid::DlvoColloidalStabilityEngine;

pub mod de_gennes_polymer_reptation;
pub use de_gennes_polymer_reptation::DeGennesPolymerReptationTube;

pub mod flory_huggins_polymer_mixing;
pub use flory_huggins_polymer_mixing::FloryHugginsSpinodalMixing;

pub mod onsager_maier_saupe_nematic;
pub use onsager_maier_saupe_nematic::OnsagerMaierSaupeNematic;

pub mod toner_tu_vicsek_active_flocking;
pub use toner_tu_vicsek_active_flocking::TonerTuVicsekActiveMatter;

pub mod active_nematic_defect_propulsion;
pub use active_nematic_defect_propulsion::ActiveNematicDefectTurbulence;

pub mod cates_tailleur_mips_abp;
pub use cates_tailleur_mips_abp::MotilityInducedPhaseSeparation;

pub mod oldroyd_b_weissenberg_rheology;
pub use oldroyd_b_weissenberg_rheology::OldroydBViscoelasticRheology;

pub mod washburn_young_laplace_wetting;
pub use washburn_young_laplace_wetting::WashburnYoungLaplaceCapillary;

pub mod lawson_triple_product_fusion;
pub use lawson_triple_product_fusion::LawsonTripleProductIgnition;

pub mod spitzer_dreicer_runaway_plasma;
pub use spitzer_dreicer_runaway_plasma::SpitzerDreicerPlasmaTransport;

pub mod kruskal_shafranov_troyon_mhd;
pub use kruskal_shafranov_troyon_mhd::KruskalShafranovTroyonMhd;

pub mod tajima_dawson_laser_wakefield;
pub use tajima_dawson_laser_wakefield::TajimaDawsonLaserWakefield;

pub mod gamow_thermonuclear_peak;
pub use gamow_thermonuclear_peak::GamowThermonuclearPeakEngine;

pub mod hoyle_salpeter_eddington_stellar;
pub use hoyle_salpeter_eddington_stellar::HoyleSalpeterEddingtonStellar;

pub mod fermi_hillas_cosmic_ray_shock;
pub use fermi_hillas_cosmic_ray_shock::FermiHillasCosmicRayShock;

pub mod parker_solar_wind_spiral;
pub use parker_solar_wind_spiral::ParkerSolarWindSpiralEngine;

pub mod sweet_parker_petschek_reconnection;
pub use sweet_parker_petschek_reconnection::SweetParkerPetschekReconnection;

pub mod bjorken_qgp_boost_invariant;
pub use bjorken_qgp_boost_invariant::BjorkenQgpHydroExpansion;

pub mod kovtun_son_starinets_kss_bound;
pub use kovtun_son_starinets_kss_bound::KovtunSonStarinetsViscosityBound;

pub mod mclerran_venugopalan_cgc_saturation;
pub use mclerran_venugopalan_cgc_saturation::ColorGlassCondensateSaturation;

pub mod bdmps_jet_quenching_qgp;
pub use bdmps_jet_quenching_qgp::BdmpsJetQuenchingEnergyLoss;

pub mod matsui_satz_quarkonium_screening;
pub use matsui_satz_quarkonium_screening::MatsuiSatzQuarkoniumMelting;

pub mod chiral_magnetic_effect_cme;
pub use chiral_magnetic_effect_cme::ChiralMagneticEffectKharzeev;

pub mod bethe_weizsacker_nuclear_fission;
pub use bethe_weizsacker_nuclear_fission::BetheWeizsackerNuclearMass;

pub mod mayer_jensen_nuclear_shell;
pub use mayer_jensen_nuclear_shell::MayerJensenNuclearShellMagic;

pub mod walecka_relativistic_nuclear_eos;
pub use walecka_relativistic_nuclear_eos::WaleckaRelativisticMeanFieldNuclear;

pub mod kolmogorov_obukhov_turbulence;
pub use kolmogorov_obukhov_turbulence::KolmogorovK41ObukhovIntermittency;

pub mod kraichnan_2d_enstrophy_cascade;
pub use kraichnan_2d_enstrophy_cascade::Kraichnan2DTurbulenceCascade;

pub mod orr_sommerfeld_quire_hydro_stability;
pub use orr_sommerfeld_quire_hydro_stability::OrrSommerfeldSquireStability;

pub mod rayleigh_benard_lorenz_convection;
pub use rayleigh_benard_lorenz_convection::RayleighBenardLorenzConvection;

pub mod prandtl_blasius_von_karman_boundary;
pub use prandtl_blasius_von_karman_boundary::PrandtlBlasiusVonKarmanBoundary;

pub mod rankine_hugoniot_prandtl_meyer_shock;
pub use rankine_hugoniot_prandtl_meyer_shock::RankineHugoniotShockAdiabat;

pub mod zeldovich_von_neumann_doring_detonation;
pub use zeldovich_von_neumann_doring_detonation::ZeldovichVonNeumannDoringDetonation;

pub mod sedov_taylor_blast_wave;
pub use sedov_taylor_blast_wave::SedovTaylorBlastWave;

pub mod rayleigh_plesset_cavitation_bubble;
pub use rayleigh_plesset_cavitation_bubble::RayleighPlessetCavitationBubble;

pub mod sachdev_ye_kitaev_syk_chaos;
pub use sachdev_ye_kitaev_syk_chaos::SachdevYeKitaevMaximalChaos;

pub mod jackiw_teitelboim_schwarzian_gravity;
pub use jackiw_teitelboim_schwarzian_gravity::JackiwTeitelboimSchwarzianGravity;

pub mod penington_almheiri_qes_island_page;
pub use penington_almheiri_qes_island_page::QuantumExtremalSurfaceIslandPage;

pub mod pastawski_yoshida_harlow_preskill_happy;
pub use pastawski_yoshida_harlow_preskill_happy::HappyHolographicCodeBulkWedge;

pub mod susskind_brown_complexity_equals_action;
pub use susskind_brown_complexity_equals_action::HolographicComplexityActionVolume;

pub mod strominger_bms_soft_graviton_memory;
pub use strominger_bms_soft_graviton_memory::StromingerSoftTheoremMemoryTriangle;

pub mod arkani_hamed_trnka_amplituhedron;
pub use arkani_hamed_trnka_amplituhedron::AmplituhedronGrassmannianVolume;

pub mod vafa_ooguri_swampland_distance_de_sitter;
pub use vafa_ooguri_swampland_distance_de_sitter::SwamplandDistanceDeSitterBounds;

pub mod pasterski_strominger_zhiboedov_celestial;
pub use pasterski_strominger_zhiboedov_celestial::CelestialHolographyMellinAmplitude;
