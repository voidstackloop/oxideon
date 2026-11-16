//! Autonomous LLM Quantization Showcase:
//! Demonstrates online dynamic precision adaptation (`DynamicLlmQuantizer`)
//! across FP32, FP16, Q8_0, Q4_0, and BitLinear 1.58-bit based on live inference telemetry
//! (memory headroom, KV-cache pressure, latency, and activation kurtosis).

use oxideon::prelude::*;
use oxideon::quant::{DynamicLlmQuantizer, DynamicQuantPolicy, DynamicTelemetry};

fn main() -> Result<()> {
    manual_seed(42);
    println!("================================================================================");
    println!("       OXIDEON v500.0.0 — AUTONOMOUS LLM DYNAMIC QUANTIZATION");
    println!("================================================================================");

    // 1. Initialize Autonomous Quantization Policy
    println!("\n[1] Configuring Autonomous Quantization Policy...");
    let policy = DynamicQuantPolicy {
        cooldown_tokens: 5,
        critical_memory_headroom_mb: 512.0,
        max_acceptable_kurtosis: 25.0,
        ..Default::default()
    };

    let mut quantizer = DynamicLlmQuantizer::new(policy);

    // 2. Register Multi-Layer Transformer Projections
    println!("\n[2] Registering Transformer Projection Layers (Q, K, V, O, FFN)...");
    let d_model = 64;
    let d_ffn = 128;
    quantizer.add_layer(
        "model.layers.0.self_attn.q_proj",
        Linear::new(d_model, d_model)?,
    );
    quantizer.add_layer(
        "model.layers.0.self_attn.k_proj",
        Linear::new(d_model, d_model)?,
    );
    quantizer.add_layer(
        "model.layers.0.self_attn.v_proj",
        Linear::new(d_model, d_model)?,
    );
    quantizer.add_layer(
        "model.layers.0.self_attn.o_proj",
        Linear::new(d_model, d_model)?,
    );
    quantizer.add_layer(
        "model.layers.0.mlp.gate_up_proj",
        Linear::new(d_model, d_ffn)?,
    );
    quantizer.add_layer("model.layers.0.mlp.down_proj", Linear::new(d_ffn, d_model)?);

    let initial_mem = quantizer.total_memory_mb();
    println!("    Initial Total Weights Memory: {:.4} MB", initial_mem);

    // 3. Forward pass at baseline precision
    let input = Tensor::randn(&[1, d_model])?;
    let q_out = quantizer.forward_layer("model.layers.0.self_attn.q_proj", &input)?;
    println!(
        "    Forward pass successful: output shape = {:?}",
        q_out.shape()
    );

    // 4. Simulate Token Stream & Telemetry-driven Adaptation
    println!("\n[3] Simulating Token Generation Stream & Telemetry-Driven Adaptation...");

    // Scenario A: Memory pressure spike -> Downgrade to BitLinear 1.58
    println!("\n    --> [Event A] Memory headroom drops to 256MB (Critical Memory Pressure)");
    let telemetry_pressure = DynamicTelemetry {
        token_index: 10,
        token_latency_ms: 12.5,
        memory_headroom_mb: 256.0,
        kv_cache_usage_pct: 96.0,
        activation_kurtosis: 4.2,
    };
    if let Some(plan) = quantizer.evaluate_telemetry(&telemetry_pressure) {
        println!("        Transition Triggered: {}", plan.trigger_reason);
        println!("        Target Precision: {:?}", plan.target_precision);
        let _ = quantizer.apply_plan(&plan, 10);
        let compressed_mem = quantizer.total_memory_mb();
        println!(
            "        Memory Reduced: {:.4} MB -> {:.4} MB ({:.1}% savings)",
            initial_mem,
            compressed_mem,
            100.0 * (1.0 - compressed_mem / initial_mem)
        );
    }

    // Forward pass after compression
    let q_compressed = quantizer.forward_layer("model.layers.0.self_attn.q_proj", &input)?;
    assert_eq!(q_compressed.shape(), &[1, d_model]);
    println!("        Quantized Layer Inference: OK");

    // Scenario B: Kurtosis spike -> Outlier Protection
    println!("\n    --> [Event B] Outlier Kurtosis Spikes to 35.0 (Precision Uplift Protection)");
    let telemetry_kurtosis = DynamicTelemetry {
        token_index: 20,
        token_latency_ms: 14.0,
        memory_headroom_mb: 2048.0,
        kv_cache_usage_pct: 40.0,
        activation_kurtosis: 35.0,
    };
    if let Some(plan) = quantizer.evaluate_telemetry(&telemetry_kurtosis) {
        println!("        Transition Triggered: {}", plan.trigger_reason);
        println!("        Target Precision: {:?}", plan.target_precision);
        let _ = quantizer.apply_plan(&plan, 20);
        let restored_mem = quantizer.total_memory_mb();
        println!(
            "        Precision elevated to {:?}: {:.4} MB",
            plan.target_precision, restored_mem
        );
    }

    println!("\n[OK] Autonomous LLM Quantization Showcase completed successfully.");
    Ok(())
}
