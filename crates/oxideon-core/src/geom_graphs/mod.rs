//! Hyperbolic Manifolds & Continuous Temporal Graph Networks.
//!
//! Includes:
//! - Poincaré Ball Model ([`PoincareManifold`]) for hierarchical knowledge embeddings
//! - Continuous Temporal Dynamic Interaction GNN ([`TemporalInteractionGNN`])

pub mod clifford;
pub mod fiber_bundle;
pub mod hyperbolic_gat;
pub mod lorentz;
pub mod poincare;
pub mod riemannian;
pub mod sheaf;
pub mod spiking_gnn;
pub mod temporal_gnn;
pub mod wavelet;
pub mod yang_mills;

pub use clifford::{CliffordLinear, Multivector3D};
pub use fiber_bundle::{GaugeEquivariantConvLayer, PrincipalBundleConnection};
pub use hyperbolic_gat::HyperbolicGATLayer;
pub use lorentz::LorentzManifold;
pub use poincare::PoincareManifold;
pub use riemannian::RiemannianManifold;
pub use sheaf::CellularSheaf;
pub use spiking_gnn::SpikingGnnLayer;
pub use temporal_gnn::TemporalInteractionGNN;
pub use wavelet::{
    ChebyshevWaveletFilter, GraphLaplacian, LaplacianType, MultiscaleGraphWaveletTransform,
    SpatioTemporalWaveletLayer,
};
pub use yang_mills::{Su2Element, YangMillsLatticeGauge};

pub mod wasserstein_flow;
pub use wasserstein_flow::GraphWassersteinGeodesicEngine;
