//! Change detector for the synthesis chain.
//!
//! Every engine in the `omni_*` / `millennium` chain adds the previous engine's result to its own pillars, so a golden
//! value at the top (`GrandOmniMillennialEngineC`), the middle (`...L`) and the base (`...XI`) moves if any pillar's
//! formula changes. The sum has no physical meaning and the `inv > N` thresholds elsewhere cannot tell a right formula
//! from a wrong one; this test only makes silent changes visible.
//!
//! After an intended formula fix: run this test, check that the reported change is the one you meant, then update the
//! constants below. Each pillar's correctness is covered by its own closed-form or identity test, not by this one.

use oxideon::millennium::GrandOmniMillennialEngineC;
use oxideon::omni_hendeca_xi::GrandOmniHendecaEngineXI;
use oxideon::omni_quincentennial_l::GrandOmniQuincentennialEngineL;

const REL_TOL: f64 = 1e-9;

fn check(name: &str, got: f64, golden: f64) {
    assert!(
        (got - golden).abs() <= REL_TOL * golden.abs(),
        "{name}: got {got:.12}, golden {golden:.12}. If a pillar formula was changed on purpose, update the golden value."
    );
}

#[test]
fn synthesis_chain_matches_golden_values() {
    check(
        "XI(10)",
        GrandOmniHendecaEngineXI::new(10)
            .execute_hendeca_synthesis()
            .unwrap(),
        697.0,
    );
    check(
        "L(50)",
        GrandOmniQuincentennialEngineL::new(50)
            .execute_quincentennial_synthesis()
            .unwrap(),
        8199.797857448008,
    );
    check(
        "C(100)",
        GrandOmniMillennialEngineC::new(100)
            .execute_millennial_synthesis()
            .unwrap(),
        9473.297204325914,
    );
}
