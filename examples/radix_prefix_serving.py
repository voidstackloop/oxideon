#!/usr/bin/env python3
"""Radix-Tree KV-Cache Prefix Sharing & Low-Latency Serving Showcase.

Demonstrates vLLM / SGLang-style Radix Tree prefix caching for LLM serving:
1. Shared System Prompt Pre-Warming: Zero-latency reuse of KV blocks across user sessions.
2. Radix Tree Prefix Matching: O(depth) longest-prefix lookup for dynamic conversation trees.
3. Dynamic Branch Splitting: Dynamic tree branching when queries share identical prompt prefixes.
4. Multi-Turn Context Reuse: 100% KV block reuse for prior conversation turns.
5. LRU Memory Management: Eviction of least recently accessed leaf nodes under memory pressure.
6. Latency & Memory Metrics: Quantifying prefill latency reduction and KV memory savings.
"""

import time
import numpy as np
import oxideon as ox


def simulate_prefill(prompt_tokens: list, cached_token_count: int, ms_per_token: float = 0.45) -> float:
    """Simulates prefill time in ms, computing only the uncached novel tokens."""
    novel_tokens = len(prompt_tokens) - cached_token_count
    return max(novel_tokens, 0) * ms_per_token


def main():
    print("=" * 80)
    print("🌲 Oxideon Radix-Tree Prefix Caching & High-Throughput LLM Serving Showcase")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Initialize RadixPrefixCache
    # -------------------------------------------------------------------------
    block_size = 16  # Each physical KV block stores 16 tokens
    print(f"\n[Step 1] Initializing RadixPrefixCache with block_size={block_size}...")
    cache = ox.prefix_cache.RadixPrefixCache(block_size=block_size)
    print(f"  ✓ RadixPrefixCache initialized successfully (block_size={block_size}).")

    # -------------------------------------------------------------------------
    # 2. Define System Prompt and User Conversations
    # -------------------------------------------------------------------------
    print("\n[Step 2] Defining System Prompt and Multi-Turn Conversations...")
    # Simulated token IDs for common system instructions and conversation turns
    system_prompt = [
        101, 2054, 2003, 1037, 4372, 2814, 1999, 4013,  # "You are a helpful assistant in scientific"
        2829, 1010, 2031, 2017, 2156, 1996, 4242, 1012   # "computing, please explain the theory."
    ]  # 16 tokens = 1 block

    conv1_turn1 = system_prompt + [
        3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008,  # "User: What is quantum superposition?"
        4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008   # "Assistant: It is a fundamental principle..."
    ]  # 32 tokens total = 2 blocks

    conv2_turn1 = system_prompt + [
        5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008,  # "User: What is quantum entanglement?"
        6001, 6002, 6003, 6004, 6005, 6006, 6007, 6008   # "Assistant: Particles remain connected..."
    ]  # 32 tokens total = 2 blocks

    print(f"  System prompt tokens : {len(system_prompt)} tokens ({len(system_prompt)//block_size} block)")
    print(f"  Conversation 1 length: {len(conv1_turn1)} tokens ({len(conv1_turn1)//block_size} blocks)")
    print(f"  Conversation 2 length: {len(conv2_turn1)} tokens ({len(conv2_turn1)//block_size} blocks)")

    # -------------------------------------------------------------------------
    # 3. Pre-Warming System Prompt in Prefix Cache
    # -------------------------------------------------------------------------
    print("\n[Step 3] Pre-warming Shared System Prompt into Radix Prefix Cache...")
    sys_blocks = [10]
    cache.insert(system_prompt, sys_blocks)
    print(f"  Pre-warmed system prompt with physical block: {sys_blocks}")
    print(f"  Current total cached tokens: {cache.total_cached_tokens}")

    # -------------------------------------------------------------------------
    # 4. Processing Request 1 (Prompt Matching & Branch Insertion)
    # -------------------------------------------------------------------------
    print("\n[Step 4] Processing Request 1 (System Prompt Hit & Turn 1 Prefill)...")
    matched_tokens, matched_blocks = cache.match_prefix(conv1_turn1)
    print(f"  Prefix match: {len(matched_tokens)} tokens matched! Reused block table: {matched_blocks}")
    assert matched_tokens == system_prompt, "System prompt prefix should match exactly"

    latency_cold = simulate_prefill(conv1_turn1, cached_token_count=0)
    latency_warm1 = simulate_prefill(conv1_turn1, cached_token_count=len(matched_tokens))
    print(f"  Uncached prefill latency: {latency_cold:.2f} ms")
    print(f"  Cached prefill latency  : {latency_warm1:.2f} ms ({latency_cold / latency_warm1:.2f}x faster)")

    # Allocate physical block 11 for newly generated turn 1 tokens
    conv1_blocks = matched_blocks + [11]
    cache.insert(conv1_turn1, conv1_blocks)
    print(f"  Inserted Request 1 branch into tree with blocks: {conv1_blocks}")

    # -------------------------------------------------------------------------
    # 5. Processing Request 2 (Concurrent Session with Shared System Prompt)
    # -------------------------------------------------------------------------
    print("\n[Step 5] Processing Request 2 (Concurrent Session Reusing System Prompt)...")
    matched_tokens2, matched_blocks2 = cache.match_prefix(conv2_turn1)
    print(f"  Prefix match: {len(matched_tokens2)} tokens matched! Reused block table: {matched_blocks2}")
    assert matched_tokens2 == system_prompt, "System prompt prefix should match"

    latency_warm2 = simulate_prefill(conv2_turn1, cached_token_count=len(matched_tokens2))
    print(f"  Cached prefill latency  : {latency_warm2:.2f} ms")

    # Allocate physical block 12 for Request 2 novel tokens
    conv2_blocks = matched_blocks2 + [12]
    cache.insert(conv2_turn1, conv2_blocks)
    print(f"  Inserted Request 2 branch into tree with blocks: {conv2_blocks}")

    # -------------------------------------------------------------------------
    # 6. Multi-Turn Follow-Up: Request 1 Turn 2 (Deep Prefix Hit)
    # -------------------------------------------------------------------------
    print("\n[Step 6] Processing Request 1 Turn 2 (Deep Context Reuse)...")
    conv1_turn2 = conv1_turn1 + [
        7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008,  # "User: Can you show a mathematical example?"
        8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008   # "Assistant: Consider |psi> = 1/sqrt(2)..."
    ]
    matched_tokens_turn2, matched_blocks_turn2 = cache.match_prefix(conv1_turn2)
    print(f"  Matched prefix length: {len(matched_tokens_turn2)} tokens ({len(matched_tokens_turn2)//block_size} blocks)")
    print(f"  Reused blocks from Turn 1: {matched_blocks_turn2}")
    assert matched_blocks_turn2 == conv1_blocks, "All of Turn 1 should be reused"

    latency_cold_turn2 = simulate_prefill(conv1_turn2, cached_token_count=0)
    latency_warm_turn2 = simulate_prefill(conv1_turn2, cached_token_count=len(matched_tokens_turn2))
    speedup_turn2 = latency_cold_turn2 / latency_warm_turn2 if latency_warm_turn2 > 0 else 1.0
    print(f"  Turn 2 prefill time: {latency_warm_turn2:.2f} ms vs {latency_cold_turn2:.2f} ms cold ({speedup_turn2:.2f}x speedup)")

    conv1_turn2_blocks = matched_blocks_turn2 + [13]
    cache.insert(conv1_turn2, conv1_turn2_blocks)

    # -------------------------------------------------------------------------
    # 7. Demonstrating Least-Recently-Used (LRU) Eviction
    # -------------------------------------------------------------------------
    print("\n[Step 7] Simulating Server Memory Pressure & LRU Eviction...")
    print(f"  Total tokens before eviction: {cache.total_cached_tokens}")
    freed = cache.evict_lru(num_blocks=1)
    print(f"  LRU Evicted physical block(s): {freed}")
    print(f"  Total tokens after eviction : {cache.total_cached_tokens}")
    assert len(freed) >= 1, "At least 1 block should be evicted"

    # Confirm that system prompt remains intact as root
    rem_tokens, rem_blocks = cache.match_prefix(system_prompt)
    print(f"  System prompt prefix intact : {len(rem_tokens)} tokens, block: {rem_blocks}")
    assert rem_blocks == sys_blocks, "System prompt should stay preserved as hot prefix"

    # -------------------------------------------------------------------------
    # Summary Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"{'Serving Phase':<30} | {'Context (Tokens)':<18} | {'Compute Saved':<15} | {'Speedup':<12}")
    print("-" * 80)
    print(f"{'Request 1 Turn 1':<30} | {f'{len(conv1_turn1)} tokens':<18} | {'16 tokens (50%)':<15} | {f'{latency_cold/latency_warm1:.2f}x':<12}")
    print(f"{'Request 2 Turn 1':<30} | {f'{len(conv2_turn1)} tokens':<18} | {'16 tokens (50%)':<15} | {f'{latency_cold/latency_warm2:.2f}x':<12}")
    print(f"{'Request 1 Turn 2':<30} | {f'{len(conv1_turn2)} tokens':<18} | {'32 tokens (67%)':<15} | {f'{speedup_turn2:.2f}x':<12}")
    print("=" * 80)
    print("✅ Radix-Tree Prefix Caching Serving showcase executed successfully!\n")


if __name__ == "__main__":
    main()
