//! Scientific & Bio-Molecular Foundation Suite (AlphaFold3 & Equivariant AI).
//!
//! Includes:
//! - AlphaFold3 Pairformer ([`PairformerBlock`], [`TriangularMultiplicativeUpdate`])
//! - $SE(3)$-Equivariant Geometric Layers ([`SE3EquivariantLayer`])
//! - 3D Molecular Geometry Diffusion ([`MolecularGeometryDiffusion`])

/// $\ln\Gamma(x)$ for $x \ge 1/2$ (Lanczos approximation, $g = 7$, relative error about $10^{-15}$).
#[allow(clippy::excessive_precision)]
fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    let z = x - 1.0;
    let series = C
        .iter()
        .enumerate()
        .skip(1)
        .fold(C[0], |acc, (i, c)| acc + c / (z + i as f64));
    let t = z + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + series.ln()
}

/// $\ln$ of the surface area of the unit sphere $S^n \subset \mathbb{R}^{n+1}$, i.e. $\ln\big(2\pi^{(n+1)/2} / \Gamma((n+1)/2)\big)$.
/// Closed form in log space, so it is $O(1)$ and finite for every `n` (the area itself underflows beyond $n \approx 500$).
pub(crate) fn ln_unit_sphere_area(n: usize) -> f64 {
    let x = (n as f64 + 1.0) / 2.0;
    std::f64::consts::LN_2 + x * std::f64::consts::PI.ln() - ln_gamma(x)
}

/// Surface area of the unit sphere $S^n \subset \mathbb{R}^{n+1}$.
pub(crate) fn unit_sphere_area(n: usize) -> f64 {
    ln_unit_sphere_area(n).exp()
}

/// Volume of the unit ball in $\mathbb{R}^k$, i.e. $\pi^{k/2} / \Gamma(k/2 + 1) = A_{k-1} / k$ (1 for $k = 0$).
pub(crate) fn unit_ball_volume(k: usize) -> f64 {
    if k == 0 {
        1.0
    } else {
        unit_sphere_area(k - 1) / k as f64
    }
}

pub mod cohomology;
pub mod deeponet;
pub mod geometry_diffusion;
pub mod gromov_wasserstein;
pub mod infinity_category;
pub mod information_geometry;
pub mod mutual_info;
pub mod neural_operator;
pub mod operad;
pub mod pairformer;
pub mod rkhs;
pub mod se3;
pub mod sheaf_diffusion;
pub mod spin_glass;
pub mod tda;
pub mod tensor_train;
pub mod tropical;

pub use cohomology::{OrientedSimplex, SimplicialComplex};
pub use deeponet::DeepONet;
pub use geometry_diffusion::MolecularGeometryDiffusion;
pub use gromov_wasserstein::{
    GromovWassersteinConfig, GromovWassersteinEngine, GromovWassersteinResult,
};
pub use infinity_category::{HigherSimplex, QuasiCategoryEngine};
pub use information_geometry::InformationGeometryEngine;
pub use mutual_info::{InfoNceEstimator, MineEstimator};
pub use neural_operator::{FourierNeuralOperator1d, SpectralConv1d};
pub use operad::AInfinityOperadEngine;
pub use pairformer::{PairformerBlock, TriangularMultiplicativeUpdate};
pub use rkhs::{
    KernelRidgeRegressor, MercerKernel, RandomFourierFeatures, maximum_mean_discrepancy,
};
pub use se3::SE3EquivariantLayer;
pub use sheaf_diffusion::{AdaptiveCellularSheaf, SheafDiffusionLayer};
pub use spin_glass::{SherringtonKirkpatrickModel, TapSolution};
pub use tda::{
    PersistenceDiagram, PersistencePair, PersistentHomologyEngine, Simplex, wasserstein_distance,
};
pub use tensor_train::{TensorTrain, TtCore};
pub use tropical::{TropicalMatrix, TropicalMonomial, TropicalPolynomial, tropical_distance};

pub mod tropical_spectral;
pub use tropical_spectral::TropicalSpectralEngine;

pub mod free_probability;
pub use free_probability::FreeProbabilityEngine;

pub mod p_adic;
pub use p_adic::PAdicAnalysisEngine;

pub mod morse_theory;
pub use morse_theory::DiscreteMorseEngine;

pub mod derived_geometry;
pub use derived_geometry::DglaMaurerCartanEngine;

pub mod spectral_triple;
pub use spectral_triple::ConnesSpectralTriple;

pub mod langlands;
pub use langlands::LanglandsHeckeEngine;

pub mod etale_cohomology;
pub use etale_cohomology::EtaleWeilZetaEngine;

pub mod motivic_integration;
pub use motivic_integration::MotivicHodgeEngine;

pub mod cluster_algebras;
pub use cluster_algebras::ClusterAlgebraQuiverEngine;

pub mod intersection_theory;
pub use intersection_theory::ChowHirzebruchRiemannRoch;

pub mod bridgeland_stability;
pub use bridgeland_stability::BridgelandStabilityEngine;

pub mod gromov_witten;
pub use gromov_witten::GromovWittenWdvvEngine;

pub mod donaldson_thomas;
pub use donaldson_thomas::DonaldsonThomasEngine;

pub mod hitchin_system;
pub use hitchin_system::HitchinHiggsBundleEngine;

pub mod perfectoid_spaces;
pub use perfectoid_spaces::PerfectoidTiltingEngine;

pub mod ricci_flow_perelman;
pub use ricci_flow_perelman::PerelmanRicciFlowEngine;

pub mod mean_curvature_flow;
pub use mean_curvature_flow::MeanCurvatureFlowEngine;

pub mod calabi_yau_monge_ampere;
pub use calabi_yau_monge_ampere::CalabiYauMongeAmpereEngine;

pub mod yang_mills_instanton;
pub use yang_mills_instanton::BpstYangMillsInstanton;

pub mod seiberg_witten;
pub use seiberg_witten::SeibergWittenMonopoleEngine;

pub mod minimal_surfaces_plateau;
pub use minimal_surfaces_plateau::WeierstrassMinimalSurfaceEngine;

pub mod microlocal_sheaves;
pub use microlocal_sheaves::MicrolocalWavefrontEngine;

pub mod index_theorem_atiyah_singer;
pub use index_theorem_atiyah_singer::AtiyahSingerIndexEngine;

pub mod non_abelian_monopoles;
pub use non_abelian_monopoles::BpsMagneticMonopoleEngine;

pub mod riemann_siegel;
pub use riemann_siegel::RiemannSiegelZEngine;

pub mod selberg_trace;
pub use selberg_trace::SelbergTraceFormulaEngine;

pub mod birch_swinnerton_dyer;
pub use birch_swinnerton_dyer::BirchSwinnertonDyerEngine;

pub mod iwasawa_theory;
pub use iwasawa_theory::IwasawaMainConjectureEngine;

pub mod abc_szpiro_arakelov;
pub use abc_szpiro_arakelov::AbcSzpiroArakelovEngine;

pub mod circle_method_hardy_littlewood;
pub use circle_method_hardy_littlewood::HardyLittlewoodCircleEngine;

pub mod sieve_selberg_bombieri;
pub use sieve_selberg_bombieri::SelbergSieveEngine;

pub mod diophantine_faltings_roth;
pub use diophantine_faltings_roth::FaltingsHeightEngine;

pub mod katz_sarnak_random_matrix;
pub use katz_sarnak_random_matrix::KatzSarnakSymmetryEngine;

pub mod hodgkin_huxley_ion;
pub use hodgkin_huxley_ion::HodgkinHuxleyActionPotential;

pub mod wilson_cowan_neural_field;
pub use wilson_cowan_neural_field::WilsonCowanCorticalEngine;

pub mod turing_morphogenesis_rd;
pub use turing_morphogenesis_rd::TuringReactionDiffusionEngine;

pub mod jarzynski_crooks_fluctuation;
pub use jarzynski_crooks_fluctuation::JarzynskiCrooksEngine;

pub mod schnakenberg_chemical_master;
pub use schnakenberg_chemical_master::ChemicalMasterEntropyEngine;

pub mod kramers_moyal_fokker_planck;
pub use kramers_moyal_fokker_planck::FokkerPlanckOrnsteinUhlenbeck;

pub mod worm_like_chain_polymer;
pub use worm_like_chain_polymer::KratkyPorodPolymerEngine;

pub mod marcus_electron_transfer;
pub use marcus_electron_transfer::MarcusElectronTransferEngine;

pub mod forster_exciton_resonance;
pub use forster_exciton_resonance::ForsterResonanceEnergyEngine;

pub mod quillen_formal_group_chromatic;
pub use quillen_formal_group_chromatic::ChromaticMoravaKTheory;

pub mod adams_spectral_sequence_j_hom;
pub use adams_spectral_sequence_j_hom::AdamsJHomomorphismEngine;

pub mod khovanov_categorification;
pub use khovanov_categorification::KhovanovHomologyEngine;

pub mod heegaard_floer_ozsvath_szabo;
pub use heegaard_floer_ozsvath_szabo::HeegaardFloerKnotEngine;

pub mod symplectic_fukaya_floer;
pub use symplectic_fukaya_floer::FukayaArnoldConjectureEngine;

pub mod scholze_clausen_condensed_liquid;
pub use scholze_clausen_condensed_liquid::CondensedLiquidTensorEngine;

pub mod prismatic_cohomology_bhatt_scholze;
pub use prismatic_cohomology_bhatt_scholze::PrismaticSiteBreuilKisin;

pub mod motivic_voevodsky_milnor;
pub use motivic_voevodsky_milnor::VoevodskyMilnorBlochKato;

pub mod lurie_higher_algebra_en_operads;
pub use lurie_higher_algebra_en_operads::LurieLittleDisksFactorization;

pub mod hartree_fock_roothaan;
pub use hartree_fock_roothaan::RoothaanHallScfEngine;

pub mod kohn_sham_dft_jacob_ladder;
pub use kohn_sham_dft_jacob_ladder::KohnShamJacobLadderEngine;

pub mod coupled_cluster_ccsd_t;
pub use coupled_cluster_ccsd_t::CoupledClusterCcsdTEngine;

pub mod moller_plesset_mp2;
pub use moller_plesset_mp2::MollerPlessetMp2Engine;

pub mod caspt2_dmrg_multireference;
pub use caspt2_dmrg_multireference::MultireferenceCasEngine;

pub mod gw_bethe_salpeter_exciton;
pub use gw_bethe_salpeter_exciton::HedinGwBetheSalpeterEngine;

pub mod conical_intersection_berry;
pub use conical_intersection_berry::YarkonyConicalIntersectionEngine;

pub mod eyring_kramers_transition_state;
pub use eyring_kramers_transition_state::EyringTransitionStateEngine;

pub mod ring_polymer_pimd_instanton;
pub use ring_polymer_pimd_instanton::RingPolymerMolecularDynamics;

pub mod gromov_hyperbolic_groups;
pub use gromov_hyperbolic_groups::GromovDeltaHyperbolicEngine;

pub mod teichmuller_weil_petersson;
pub use teichmuller_weil_petersson::TeichmullerWeilPeterssonEngine;

pub mod thurston_hyperbolic_dehn;
pub use thurston_hyperbolic_dehn::ThurstonHyperbolicDehnEngine;

pub mod bass_serre_trees;
pub use bass_serre_trees::BassSerreGraphOfGroupsEngine;

pub mod kazhdan_property_t_expander;
pub use kazhdan_property_t_expander::KazhdanPropertyTExpander;

pub mod arthur_selberg_langlands;
pub use arthur_selberg_langlands::SatakeHeckeLanglandsEngine;

pub mod ngo_fundamental_lemma;
pub use ngo_fundamental_lemma::NgoFundamentalLemmaEngine;

pub mod drinfeld_shtukas_function_field;
pub use drinfeld_shtukas_function_field::DrinfeldShtukaLanglandsEngine;

pub mod ratner_unipotent_ergodic;
pub use ratner_unipotent_ergodic::RatnerUnipotentFlowEngine;

pub mod kimura_wright_fisher_genetics;
pub use kimura_wright_fisher_genetics::KimuraWrightFisherEngine;

pub mod kingman_coalescent_ancestry;
pub use kingman_coalescent_ancestry::KingmanCoalescentEngine;

pub mod price_hamilton_inclusive_fitness;
pub use price_hamilton_inclusive_fitness::PriceHamiltonEvolutionEngine;

pub mod replicator_mutator_ess;
pub use replicator_mutator_ess::ReplicatorMutatorGameEngine;

pub mod eigen_quasispecies_threshold;
pub use eigen_quasispecies_threshold::EigenQuasispeciesThreshold;

pub mod may_wigner_ecosystem_stability;
pub use may_wigner_ecosystem_stability::MayWignerComplexityStability;

pub mod fisher_kpp_traveling_wave;
pub use fisher_kpp_traveling_wave::FisherKppWavefrontEngine;

pub mod kermack_mckendrick_epidemic;
pub use kermack_mckendrick_epidemic::KermackMcKendrickEpidemicEngine;

pub mod hubbell_ewens_biodiversity;
pub use hubbell_ewens_biodiversity::HubbellEwensBiodiversityEngine;

pub mod martin_borel_determinacy;
pub use martin_borel_determinacy::MartinBorelDeterminacyEngine;

pub mod shelah_stability_sauer;
pub use shelah_stability_sauer::ShelahStabilitySauerEngine;

pub mod o_minimal_pfaffian_cells;
pub use o_minimal_pfaffian_cells::OMinimalPfaffianCellEngine;

pub mod hrushovski_zariski_geometry;
pub use hrushovski_zariski_geometry::HrushovskiZariskiGeometry;

pub mod goodstein_paris_harrington;
pub use goodstein_paris_harrington::GoodsteinParisHarringtonEngine;

pub mod solomonoff_levin_algorithmic;
pub use solomonoff_levin_algorithmic::SolomonoffLevinUniversalPrior;

pub mod razborov_sunflower_circuit;
pub use razborov_sunflower_circuit::SunflowerNaturalProofsBarrier;

pub mod dinur_pcp_unique_games;
pub use dinur_pcp_unique_games::DinurPcpGapAmplification;

pub mod williams_sat_circuit_duality;
pub use williams_sat_circuit_duality::WilliamsSatCircuitLowerBound;

pub mod heath_jarrow_morton_hjm;
pub use heath_jarrow_morton_hjm::HeathJarrowMortonTermStructure;

pub mod hagan_sabr_volatility;
pub use hagan_sabr_volatility::HaganSabrVolatilityEngine;

pub mod almgren_chriss_execution;
pub use almgren_chriss_execution::AlmgrenChrissOptimalExecution;

pub mod avellaneda_stoikov_market_making;
pub use avellaneda_stoikov_market_making::AvellanedaStoikovMarketMaker;

pub mod kyle_lambda_microstructure;
pub use kyle_lambda_microstructure::KyleLambdaAuctionEquilibrium;

pub mod rockafellar_uryasev_cvar;
pub use rockafellar_uryasev_cvar::RockafellarUryasevCVaREngine;

pub mod merton_structural_credit;
pub use merton_structural_credit::MertonStructuralDefaultEngine;

pub mod eisenberg_noe_systemic_risk;
pub use eisenberg_noe_systemic_risk::EisenbergNoeSystemicClearing;

pub mod kelly_thorp_optimal_growth;
pub use kelly_thorp_optimal_growth::KellyLogOptimalPortfolio;

pub mod tutte_polynomial_matroid;
pub use tutte_polynomial_matroid::TuttePolynomialMatroidEngine;

pub mod huh_katz_matroid_hodge;
pub use huh_katz_matroid_hodge::HuhKatzMatroidLogConcavity;

pub mod mikhalkin_tropical_bezout;
pub use mikhalkin_tropical_bezout::MikhalkinTropicalGeometryEngine;

pub mod ehrhart_macdonald_polytope;
pub use ehrhart_macdonald_polytope::EhrhartMacdonaldPolytopeEngine;

pub mod stanley_reisner_dehn_sommerville;
pub use stanley_reisner_dehn_sommerville::StanleyReisnerDehnSommerville;

pub mod rsk_hook_length_tableaux;
pub use rsk_hook_length_tableaux::HookLengthYoungTableauxEngine;

pub mod macdonald_qt_catalan;
pub use macdonald_qt_catalan::MacdonaldSymmetricCatalanEngine;

pub mod kazhdan_lusztig_bruhat;
pub use kazhdan_lusztig_bruhat::KazhdanLusztigHeckeEngine;

pub mod szemeredi_green_tao_density;
pub use szemeredi_green_tao_density::SzemerediGreenTaoRegularity;

pub mod caffarelli_kohn_nirenberg_ns;
pub use caffarelli_kohn_nirenberg_ns::CaffarelliKohnNirenbergNavierStokes;

pub mod de_giorgi_nash_moser;
pub use de_giorgi_nash_moser::DeGiorgiNashMoserRegularity;

pub mod calderon_zygmund_singular;
pub use calderon_zygmund_singular::CalderonZygmundSingularIntegral;

pub mod strichartz_dispersive_pde;
pub use strichartz_dispersive_pde::StrichartzDispersiveEstimate;

pub mod lax_kdv_inverse_scattering;
pub use lax_kdv_inverse_scattering::LaxKdvInverseScatteringSoliton;

pub mod zakharov_shabat_nls_rogue;
pub use zakharov_shabat_nls_rogue::ZakharovShabatNlsSoliton;

pub mod lions_concentration_compactness;
pub use lions_concentration_compactness::LionsConcentrationCompactness;

pub mod modica_mortola_gamma_conv;
pub use modica_mortola_gamma_conv::ModicaMortolaGammaConvergence;

pub mod crandall_lions_viscosity_pucci;
pub use crandall_lions_viscosity_pucci::CrandallLionsViscosityPucci;

pub mod wolynes_protein_folding_funnel;
pub use wolynes_protein_folding_funnel::WolynesProteinFoldingFunnel;

pub mod zimm_bragg_ramachandran;
pub use zimm_bragg_ramachandran::ZimmBraggRamachandranHelix;

pub mod monod_wyman_changeux_allostery;
pub use monod_wyman_changeux_allostery::MonodWymanChangeuxAllostery;

pub mod briggs_haldane_enzyme_kinetics;
pub use briggs_haldane_enzyme_kinetics::BriggsHaldaneEnzymeKinetics;

pub mod crispr_cas9_rloop_kinetics;
pub use crispr_cas9_rloop_kinetics::CrisprCas9RLoopKinetics;

pub mod karlin_altschul_alignment;
pub use karlin_altschul_alignment::KarlinAltschulSequenceAlignment;

pub mod felsenstein_jukes_cantor_phylo;
pub use felsenstein_jukes_cantor_phylo::FelsensteinPhylogeneticLikelihood;

pub mod rna_velocity_waddington;
pub use rna_velocity_waddington::RnaVelocityWaddingtonLandscape;

pub mod bell_evans_mechanobiology;
pub use bell_evans_mechanobiology::BellEvansMechanobiologyBond;

pub mod lamport_pbft_byzantine_quorum;
pub use lamport_pbft_byzantine_quorum::ByzantineQuorumConsensusEngine;

pub mod lamport_mattern_vector_clock;
pub use lamport_mattern_vector_clock::MatternVectorClockCausality;

pub mod shapiro_crdt_semilattice;
pub use shapiro_crdt_semilattice::ShapiroCrdtJoinSemilattice;

pub mod karger_consistent_hashing;
pub use karger_consistent_hashing::KargerConsistentHashingRing;

pub mod gilbert_lynch_cap_pacelc;
pub use gilbert_lynch_cap_pacelc::GilbertLynchPacelcTheorem;

pub mod clarke_emerson_mu_calculus;
pub use clarke_emerson_mu_calculus::ClarkeEmersonModelChecker;

pub mod reynolds_separation_logic;
pub use reynolds_separation_logic::ReynoldsSeparationLogicHeap;

pub mod cousot_abstract_interpretation;
pub use cousot_abstract_interpretation::CousotGaloisAbstractInterpreter;

pub mod girard_linear_logic_resource;
pub use girard_linear_logic_resource::GirardLinearLogicResource;

pub mod almgren_frequency_minimal_currents;
pub use almgren_frequency_minimal_currents::AlmgrenFrequencyCurrentsEngine;

pub mod allard_varifold_monotonicity;
pub use allard_varifold_monotonicity::AllardVarifoldMonotonicityEngine;

pub mod federer_fleming_integral_currents;
pub use federer_fleming_integral_currents::FedererFlemingCurrentsCoarea;

pub mod caffarelli_monge_ampere_transport;
pub use caffarelli_monge_ampere_transport::CaffarelliMongeAmpereTransport;

pub mod alt_caffarelli_free_boundary;
pub use alt_caffarelli_free_boundary::AltCaffarelliFreeBoundaryWeiss;

pub mod caffarelli_silvestre_fractional;
pub use caffarelli_silvestre_fractional::CaffarelliSilvestreFractionalExt;

pub mod simons_cone_bernstein_minimal;
pub use simons_cone_bernstein_minimal::SimonsMinimalConeBernstein;

pub mod besicovitch_marstrand_rectifiable;
pub use besicovitch_marstrand_rectifiable::BesicovitchMarstrandRectifiable;

pub mod cheeger_buser_isoperimetric;
pub use cheeger_buser_isoperimetric::CheegerBuserSpectralInequality;

pub mod lovasz_local_lemma_moser_tardos;
pub use lovasz_local_lemma_moser_tardos::LovaszLocalLemmaMoserTardos;

pub mod talagrand_convex_concentration;
pub use talagrand_convex_concentration::TalagrandConvexConcentration;

pub mod friedgut_bourgain_sharp_threshold;
pub use friedgut_bourgain_sharp_threshold::FriedgutBourgainSharpThreshold;

pub mod bollobas_random_graph_giant_component;
pub use bollobas_random_graph_giant_component::ErdosRenyiBollobasPhaseTransition;

pub mod alon_boppana_ramanujan_expander;
pub use alon_boppana_ramanujan_expander::AlonBoppanaRamanujanExpander;

pub mod balog_szemeredi_gowers_freiman;
pub use balog_szemeredi_gowers_freiman::BalogSzemerediGowersFreiman;

pub mod roth_bourgain_meshulam_capset;
pub use roth_bourgain_meshulam_capset::RothBourgainCapSetBounds;

pub mod lovasz_theta_shannon_capacity;
pub use lovasz_theta_shannon_capacity::LovaszThetaShannonCapacity;

pub mod kahn_kalai_park_pham_threshold;
pub use kahn_kalai_park_pham_threshold::KahnKalaiParkPhamThreshold;

pub mod grothendieck_lefschetz_trace;
pub use grothendieck_lefschetz_trace::GrothendieckLefschetzTraceFormula;

pub mod faltings_mordell_arakelov_height;
pub use faltings_mordell_arakelov_height::FaltingsMordellArakelovEngine;

pub mod birch_swinnerton_dyer_tate_shafarevich;
pub use birch_swinnerton_dyer_tate_shafarevich::BirchSwinnertonDyerLeadingTerm;

pub mod wiles_taylor_modularity_hecke;
pub use wiles_taylor_modularity_hecke::WilesTaylorModularityHecke;

pub mod fontaine_p_adic_hodge_de_rham;
pub use fontaine_p_adic_hodge_de_rham::FontainePAdicHodgePeriodEngine;

pub mod iwasawa_main_conjecture_mazur_wiles;
pub use iwasawa_main_conjecture_mazur_wiles::IwasawaClassNumberGrowth;

pub mod sato_tate_equidistribution_taylor;
pub use sato_tate_equidistribution_taylor::SatoTateSemicircleEquidistribution;

pub mod schmidt_subspace_roth_diophantine;
pub use schmidt_subspace_roth_diophantine::RothSchmidtDiophantineEngine;

pub mod voevodsky_motivic_milnor_bloch_kato;
pub use voevodsky_motivic_milnor_bloch_kato::VoevodskyMotivicBlochKato;

pub mod shannon_hartley_awgn_waterfilling;
pub use shannon_hartley_awgn_waterfilling::ShannonHartleyWaterfilling;

pub mod slepian_wolf_wyner_ziv_distributed;
pub use slepian_wolf_wyner_ziv_distributed::SlepianWolfRateRegionEngine;

pub mod blahut_arimoto_rate_distortion;
pub use blahut_arimoto_rate_distortion::GaussianRateDistortionEngine;

pub mod gallager_ldpc_random_coding_exponent;
pub use gallager_ldpc_random_coding_exponent::GallagerRandomCodingExponent;

pub mod arikan_polar_channel_polarization;
pub use arikan_polar_channel_polarization::ArikanPolarChannelPolarization;

pub mod macwilliams_duality_singleton_mds;
pub use macwilliams_duality_singleton_mds::MacWilliamsSingletonMdsCode;

pub mod guruswami_sudan_list_decoding;
pub use guruswami_sudan_list_decoding::GuruswamiSudanListDecoder;

pub mod ahlswede_cai_li_yeung_network_coding;
pub use ahlswede_cai_li_yeung_network_coding::LinearNetworkCodingMaxFlow;

pub mod polyanskiy_poor_verdu_finite_blocklength;
pub use polyanskiy_poor_verdu_finite_blocklength::PolyanskiyFiniteBlocklengthRate;

pub mod regev_lwe_lattice_reduction;
pub use regev_lwe_lattice_reduction::RegevLweLatticeSecurity;

pub mod lll_bkz_hermite_root_factor;
pub use lll_bkz_hermite_root_factor::LllBkzLatticeReduction;

pub mod gentry_fhe_bootstrapping_noise;
pub use gentry_fhe_bootstrapping_noise::GentryFheNoiseGrowthEngine;

pub mod lund_fortnow_karloff_nisan_sumcheck;
pub use lund_fortnow_karloff_nisan_sumcheck::SumcheckInteractiveProof;

pub mod groth16_qap_zk_snark;
pub use groth16_qap_zk_snark::Groth16QapSnarkAnalyzer;

pub mod ben_sasson_fri_reed_solomon_stark;
pub use ben_sasson_fri_reed_solomon_stark::FriReedSolomonProximityStark;

pub mod shamir_bgw_mpc_threshold;
pub use shamir_bgw_mpc_threshold::ShamirBgwThresholdMpc;

pub mod dwork_mcsherry_gaussian_differential_privacy;
pub use dwork_mcsherry_gaussian_differential_privacy::DworkGaussianDifferentialPrivacy;

pub mod mironov_renyi_dp_moments_accountant;
pub use mironov_renyi_dp_moments_accountant::MironovRenyiDpAccountant;

pub mod pontryagin_maximum_principle_bang_bang;
pub use pontryagin_maximum_principle_bang_bang::PontryaginMinimumTimeBangBang;

pub mod kalman_bucy_lqg_riccati;
pub use kalman_bucy_lqg_riccati::KalmanBucyRiccatiFilter;

pub mod zames_doyle_glover_h_infinity_control;
pub use zames_doyle_glover_h_infinity_control::ZamesDoyleHInfinityControl;

pub mod sontag_control_lyapunov_iss;
pub use sontag_control_lyapunov_iss::SontagControlLyapunovEngine;

pub mod ames_control_barrier_function_qp;
pub use ames_control_barrier_function_qp::AmesControlBarrierFunctionQp;

pub mod lasry_lions_mean_field_game;
pub use lasry_lions_mean_field_game::LasryLionsLinearQuadraticMfg;

pub mod isaacs_homicidal_chauffeur_differential_game;
pub use isaacs_homicidal_chauffeur_differential_game::IsaacsDifferentialPursuitGame;

pub mod kushner_stratonovich_zakai_nonlinear_filter;
pub use kushner_stratonovich_zakai_nonlinear_filter::ZakaiUnnormalizedFilteringSde;

pub mod willems_dissipativity_passivity_kyp;
pub use willems_dissipativity_passivity_kyp::WillemsDissipativityKypEngine;

pub mod maclane_coherence_monoidal_pentagon;
pub use maclane_coherence_monoidal_pentagon::MacLaneMonoidalCoherence;

pub mod grothendieck_topos_giraud_sheaf;
pub use grothendieck_topos_giraud_sheaf::GiraudGrothendieckToposEngine;

pub mod lurie_infinity_topos_straightening;
pub use lurie_infinity_topos_straightening::LurieInfinityCategoryEngine;

pub mod baez_dolan_lurie_cobordism_hypothesis;
pub use baez_dolan_lurie_cobordism_hypothesis::BaezDolanCobordismHypothesis;

pub mod voevodsky_univalence_hott_freudenthal;
pub use voevodsky_univalence_hott_freudenthal::HomotopyTypeTheoryUnivalence;

pub mod scholze_perfectoid_tilting_equivalence;
pub use scholze_perfectoid_tilting_equivalence::ScholzePerfectoidTiltingEngine;

pub mod clausen_scholze_condensed_liquid_tensor;
pub use clausen_scholze_condensed_liquid_tensor::ClausenScholzeLiquidTensor;

pub mod bhatt_scholze_prismatic_cohomology;
pub use bhatt_scholze_prismatic_cohomology::BhattScholzePrismaticEngine;

pub mod fargues_fontaine_curve_geometrization;
pub use fargues_fontaine_curve_geometrization::FarguesFontaineCurveEngine;

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Closed forms: A_n = 2 pi^((n+1)/2) / Gamma((n+1)/2) for S^0..S^5, omega_k = pi^(k/2) / Gamma(k/2 + 1) for B^0..B^5.
    #[test]
    fn sphere_area_and_ball_volume_match_closed_forms() {
        let areas = [
            2.0,
            2.0 * PI,
            4.0 * PI,
            2.0 * PI * PI,
            8.0 * PI * PI / 3.0,
            PI.powi(3),
        ];
        for (n, a) in areas.iter().enumerate() {
            assert!((unit_sphere_area(n) - a).abs() < 1e-12 * a, "S^{n}");
        }
        let volumes = [
            1.0,
            2.0,
            PI,
            4.0 * PI / 3.0,
            PI * PI / 2.0,
            8.0 * PI * PI / 15.0,
        ];
        for (k, v) in volumes.iter().enumerate() {
            assert!((unit_ball_volume(k) - v).abs() < 1e-12 * v, "B^{k}");
        }
    }

    /// The closed form is O(1): huge dimensions return at once and stay finite in log space.
    #[test]
    fn huge_dimensions_are_constant_time() {
        for n in [1_000_000usize, 1_000_000_000_000, usize::MAX / 4] {
            assert!(
                ln_unit_sphere_area(n).is_finite() && unit_sphere_area(n) >= 0.0,
                "n={n}"
            );
        }
    }
}
