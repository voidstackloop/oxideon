#!/usr/bin/env python3
"""Oxideon v500.0.0 — Python Autonomous LLM Quantization Showcase.

Demonstrates:
1. Online adaptive precision orchestration via `DynamicLlmQuantizer`
2. Telemetry ingestion (token latency, memory headroom, KV-cache utilization, activation kurtosis)
3. Emergency dynamic downgrade to 1.58-bit BitLinear under memory pressure
4. High-fidelity precision uplift to FP16 under outlier kurtosis spikes
"""

import oxideon as ox


def main():
    print("=" * 80)
    print("       OXIDEON v500.0.0 — PYTHON AUTONOMOUS LLM QUANTIZATION SHOWCASE")
    print("=" * 80)

    # 1. Configure Dynamic Quantization Policy
    print("\n[1] Configuring Autonomous Quantization Policy...")
    policy = ox.DynamicQuantPolicy(
        target_latency_ms=20.0,
        min_memory_headroom_mb=512.0,
        critical_memory_headroom_mb=128.0,
        max_acceptable_kurtosis=6.0,
        kv_cache_threshold_pct=85.0,
        cooldown_tokens=5,
    )
    quantizer = ox.DynamicLlmQuantizer(policy)

    # 2. Register Transformer Weights
    print("\n[2] Registering Transformer Projection Layers...")
    d_model = 128
    weights = {
        "model.layers.0.self_attn.q_proj": ox.Tensor.randn([d_model, d_model]),
        "model.layers.0.self_attn.k_proj": ox.Tensor.randn([d_model, d_model]),
        "model.layers.0.self_attn.v_proj": ox.Tensor.randn([d_model, d_model]),
        "model.layers.0.self_attn.o_proj": ox.Tensor.randn([d_model, d_model]),
        "model.layers.0.mlp.gate_up_proj": ox.Tensor.randn([d_model * 2, d_model]),
        "model.layers.0.mlp.down_proj": ox.Tensor.randn([d_model, d_model * 2]),
    }

    for name, w in weights.items():
        quantizer.add_layer(name, w)

    initial_mem_mb = quantizer.total_memory_mb()
    print(f"    Baseline Memory (FP32 weights): {initial_mem_mb:.4f} MB")

    # 3. Forward Pass at Baseline Precision
    inp = ox.Tensor.randn([1, d_model])
    out = quantizer.forward_layer("model.layers.0.self_attn.q_proj", inp)
    print(f"    Baseline Forward Pass OK: output shape = {out.shape}")

    # 4. Event A: Memory Headroom Collapse (KV-Cache Full)
    print("\n[3] Ingesting Live Telemetry — Critical Memory Pressure Event:")
    telemetry_pressure = ox.DynamicTelemetry(
        token_latency_ms=18.5,
        memory_headroom_mb=64.0,       # Below 128MB critical limit
        kv_cache_usage_pct=96.5,       # Above 95% threshold
        activation_kurtosis=3.5,
        token_index=10,
    )

    plan = quantizer.evaluate_telemetry(telemetry_pressure)
    if plan:
        print(f"    --> Transition Triggered: {plan.trigger_reason}")
        print(f"    --> Target Precision:    {plan.target_precision}")
        print(f"    --> Expected Speedup:    {plan.expected_speedup:.2f}x")
        quantizer.apply_plan(plan, 10)
        compressed_mem_mb = quantizer.total_memory_mb()
        savings_pct = 100.0 * (1.0 - compressed_mem_mb / initial_mem_mb)
        print(f"    --> Memory Reduced:       {initial_mem_mb:.4f} MB -> {compressed_mem_mb:.4f} MB ({savings_pct:.1f}% savings)")

    # Forward pass after quantization
    out_quant = quantizer.forward_layer("model.layers.0.self_attn.q_proj", inp)
    assert out_quant.shape == [1, d_model]
    print(f"    Quantized Forward Pass (1.58-bit): OK")

    # 5. Event B: Outlier Kurtosis Recovery
    print("\n[4] Ingesting Live Telemetry — Outlier Kurtosis Precision Uplift Event:")
    telemetry_kurtosis = ox.DynamicTelemetry(
        token_latency_ms=12.0,
        memory_headroom_mb=2048.0,
        kv_cache_usage_pct=30.0,
        activation_kurtosis=28.0,      # Outlier spike
        token_index=20,
    )

    uplift_plan = quantizer.evaluate_telemetry(telemetry_kurtosis)
    if uplift_plan:
        print(f"    --> Transition Triggered: {uplift_plan.trigger_reason}")
        print(f"    --> Target Precision:    {uplift_plan.target_precision}")
        quantizer.apply_plan(uplift_plan, 20)
        restored_mem_mb = quantizer.total_memory_mb()
        print(f"    --> Precision Elevated:   {restored_mem_mb:.4f} MB")

    print("\n[OK] Python Autonomous LLM Quantization Showcase completed successfully.")


if __name__ == "__main__":
    main()
