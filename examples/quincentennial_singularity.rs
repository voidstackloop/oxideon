//! Quincentennial Singularity Showcase:
//! Full hierarchical synthesis through v500.0.0 via [`GrandOmniQuincentennialEngineL`].
//! Demonstrates the cascading evaluation across 50 decadal milestone synthesis engines,
//! coupled with online autonomous LLM quantization and higher-order autograd.

use oxideon::omni_quincentennial_l::GrandOmniQuincentennialEngineL;
use oxideon::prelude::*;

fn main() -> Result<()> {
    manual_seed(500);
    println!("================================================================================");
    println!("       OXIDEON v500.0.0 — QUINCENTENNIAL SINGULARITY SYNTHESIS ENGINE");
    println!("================================================================================");

    // 1. Initialize the Grand Omni-Universal Quincentennial Foundation Engine L (Milestone v500)
    let decadal_layers = 50;
    println!(
        "\n[1] Initializing GrandOmniQuincentennialEngineL (50 Decadal Milestone Synthesis)..."
    );
    let quincentennial_engine = GrandOmniQuincentennialEngineL::new(decadal_layers);
    println!(
        "    Engine created: num_layers = {}",
        quincentennial_engine.num_layers
    );

    // 2. Execute Hierarchical Synthesis Cascade (v10 -> v500)
    println!("\n[2] Executing Full Quincentennial Synthesis Cascade across 500 Subsystems...");
    let start_time = std::time::Instant::now();
    let synthesis_invariant = quincentennial_engine.execute_quincentennial_synthesis()?;
    let elapsed = start_time.elapsed();
    println!(
        "    Quincentennial Invariant Computed: {:.10}",
        synthesis_invariant
    );
    println!("    Execution Time: {:?}", elapsed);
    assert!(
        synthesis_invariant > 0.0,
        "Invariant must be strictly positive"
    );

    // 3. Higher-Order Autograd Coupled to Quincentennial Scale
    println!("\n[3] Autograd Coupling with Invariant-Scaled Tensor Dynamics...");
    let t = Tensor::from_slice(&[synthesis_invariant as f32], &[1])?.requires_grad_(true);
    let y = t.powf(2.0)?.sin()?;
    let grad_1 = grad(
        std::slice::from_ref(&y),
        std::slice::from_ref(&t),
        None,
        true,
    )?
    .pop()
    .unwrap()
    .unwrap();
    let grad_2 = grad(
        std::slice::from_ref(&grad_1),
        std::slice::from_ref(&t),
        None,
        false,
    )?
    .pop()
    .unwrap()
    .unwrap();
    println!("    f(x)  = sin(x^2)");
    println!("    f'(x)  = {:.6}", grad_1.item::<f32>()?);
    println!("    f''(x) = {:.6}", grad_2.item::<f32>()?);

    println!("\n[OK] Quincentennial Singularity Showcase completed successfully.");
    Ok(())
}
