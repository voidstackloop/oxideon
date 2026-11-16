//! Cross-validation batch 40: closed-form / small-algorithm engines in `ode` vs SciPy / NumPy / quadrature
//! (tools/crossval/check_batch40.py).
use oxideon_core::ode::*;
use oxideon_core::ode::chow_rashevskii_sub_riemannian::SubRiemannianHeisenbergEngine;
use oxideon_core::ode::kalman_bucy_riccati_lqg::KalmanBucyRiccatiEngine;
use oxideon_core::ode::pontryagin_maximum_principle::PontryaginMaximumEngine;
use oxideon_core::ode::sontag_control_lyapunov_cbf::ControlBarrierLyapunovEngine;
use oxideon_core::ode::zames_h_infinity_rob_control::HInfinityRobustControlEngine;

fn p(name: &str, v: &[f64]) {
    println!("{name} {}", v.iter().map(|a| format!("{a:e}")).collect::<Vec<_>>().join(" "));
}

#[test]
#[ignore]
fn emit_ode() {
    // Kalman-Bucy steady state
    let mut v = vec![];
    for (a, c, q, r) in [(-0.5, 1.0, 2.0, 0.7), (0.8, 2.0, 0.3, 1.5), (0.0, 1.0, 1.0, 1.0), (3.0, 0.5, 4.0, 0.2)] {
        let (pp, k) = KalmanBucyRiccatiEngine::new(a, c, q, r).unwrap().steady_state_covariance_and_gain();
        v.extend([pp, k]);
    }
    p("kb", &v);
    // H-infinity scalar ARE
    p("hinf", &[
        HInfinityRobustControlEngine::solve_h_infinity_are(-0.5, 0.4, 1.0, 2.0, 0.7, 2.0).unwrap(),
        HInfinityRobustControlEngine::solve_h_infinity_are(0.8, 0.2, 1.5, 1.0, 1.0, 3.0).unwrap(),
    ]);
    // Malliavin weights at several W_T
    let m = MalliavinBismutEngine::new(100.0, 0.3, 1.5).unwrap();
    let mut v = vec![];
    for i in 0..=6000 { let w = -12.0 + 24.0 * i as f64 / 6000.0; let (d, g) = m.bismut_delta_and_gamma_weights(w); v.extend([d, g]); }
    p("malliavin", &v);
    // McKean-Vlasov variance
    let mk = McKeanVlasovPropagationChaos::new(0.8, 0.5).unwrap();
    p("mckean", &[0.0, 0.3, 1.0, 4.0].iter().map(|&t| mk.mean_field_variance(2.0, t).unwrap()).collect::<Vec<_>>());
    // Levy exponent
    let l = LevyItodiffusionEngine::new(0.4, 1.3, 0.2, 0.6).unwrap();
    p("levy", &[0.0, 0.5, 1.0, 2.5, 6.0].iter().map(|&u| l.real_levy_exponent(u)).collect::<Vec<_>>());
    // Signature of a 3-d path
    let path: Vec<Vec<f64>> = (0..7).map(|i| { let t = i as f64; vec![(0.9 * t).sin(), (0.5 * t).cos() * t * 0.3, (1.7 * t).sin() * 0.5] }).collect();
    let s = RoughPathSignature::from_path(&path).unwrap();
    p("sig1", &s.level1); p("sig2", &s.level2);
    p("levy_area", &[s.levy_area(0, 1).unwrap(), s.levy_area(1, 2).unwrap(), s.levy_area(2, 0).unwrap()]);
    // Sontag control: V_dot = a + b u
    let mut v = vec![];
    for (a, b) in [(1.0, 2.0), (-1.0, 0.5), (3.0, -1.5), (0.0, 1.0), (2.0, 0.0)] { v.push(ControlBarrierLyapunovEngine::sontag_universal_control(a, b)); }
    p("sontag", &v);
    let mut v = vec![];
    for (u, f, g, al) in [(1.0, 0.5, 1.0, 0.2), (-2.0, 0.5, 1.0, 0.2), (0.3, -1.0, -2.0, 0.5), (5.0, 1.0, 2.0, 0.1)] { v.push(ControlBarrierLyapunovEngine::cbf_safety_filter(u, f, g, al)); }
    p("cbf", &v);
    // Pontryagin
    let pm = PontryaginMaximumEngine::new(0.8, 1.0).unwrap();
    let mut v = vec![];
    for (d, g) in [(0.5, 0.3), (0.5, 2.0), (-1.0, -3.0)] { let (u, h) = pm.optimal_control_and_hamiltonian(d, g); v.extend([u, h]); }
    p("pontryagin", &v);
    // Stein-Chen bound and 4th-moment bound
    p("steinchen", &[SteinPoissonCharacterization::stein_chen_poisson_bound(&[0.1, 0.2, 0.05, 0.3, 0.15]).unwrap(),
                     SteinPoissonCharacterization::stein_chen_poisson_bound(&[0.5, 0.5]).unwrap()]);
    p("fourth", &[SteinPoissonCharacterization::fourth_moment_tv_bound(2, 3.5).unwrap(), SteinPoissonCharacterization::fourth_moment_tv_bound(4, 4.2).unwrap()]);
    // Koopman diagonal EDMD
    let px: Vec<Vec<f64>> = (0..20).map(|i| vec![(0.3 * i as f64).sin(), (0.7 * i as f64).cos() + 0.2, 0.1 * i as f64]).collect();
    let py: Vec<Vec<f64>> = px.iter().map(|r| vec![0.9 * r[0] + 0.01, -0.4 * r[1] + 0.02, 1.1 * r[2]]).collect();
    let k = KoopmanSpectralEngine::fit_diagonal_edmd(&px, &py).unwrap();
    p("koopman_k", &k.koopman_matrix);
    p("koopman_pred", &k.predict_observable(&[1.0, 2.0, -1.5], 5).unwrap());
    // Birkhoff average of logistic map
    p("birkhoff", &[KoopmanSpectralEngine::birkhoff_ulam_ergodic_average(0.3, 200000).unwrap()]);
    // Heisenberg vertical distance
    p("carnot", &[SubRiemannianHeisenbergEngine::vertical_carnot_distance(2.5), SubRiemannianHeisenbergEngine::vertical_carnot_distance(-0.7)]);
    // Eyring-Kramers
    p("ek", &[FreidlinWentzellLdpEngine::eyring_kramers_escape_time(1.0, 0.1, 2.0, 3.0).unwrap()]);
    // Mean-field game: FPK mass conservation and one HJB step
    let n = 16;
    let pot: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos()).collect();
    let mfg = MeanFieldGameSolver1D::new(n, 0.1, 0.5, pot).unwrap();
    let m0: Vec<f64> = (0..n).map(|i| 1.0 + 0.3 * (i as f64 * 0.8).sin()).collect();
    let m0s: f64 = m0.iter().sum(); let m0: Vec<f64> = m0.iter().map(|x| x / m0s).collect();
    let u0: Vec<f64> = (0..n).map(|i| (i as f64 * 0.5).cos()).collect();
    let (u1, alpha) = mfg.backward_hjb_step(&u0, &m0, 0.01, 0.2).unwrap();
    let m1 = mfg.forward_fpk_step(&m0, &alpha, 0.01, 0.2).unwrap();
    p("mfg_u1", &u1); p("mfg_m1", &m1);
}
