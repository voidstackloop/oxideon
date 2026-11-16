#!/usr/bin/env python3
"""Vision-Language Model (VLM) Architecture and Multimodal Fusion Showcase.

Demonstrates modern multi-modal vision-language architectures in Oxideon:
  1. LLaVA-style Multi-Layer Perceptron (MLP) visual patch projection
  2. Flamingo-style Perceiver Resampler with learnable latent queries
  3. Arevalo-style Gated Multimodal Fusion (GMU) for adaptive feature gating
  4. Multi-Head Cross-Attention Fusion across heterogeneous modalities
  5. End-to-end multi-modal contrastive training using SigLIP pairwise loss

Usage:
  python examples/multimodal_vlm_and_fusion.py
"""

import sys
import time
import numpy as np

import oxideon as ox
from oxideon import Tensor, from_numpy
from oxideon.multimodal_vlm import (
    VLMProjection,
    PerceiverResampler,
    VisionLanguageConnector,
)
from oxideon.multimodal_fusion import (
    CrossAttentionFusion,
    GatedMultimodalFusion,
)
from oxideon.optim import AdamW


def banner(title: str) -> None:
    print("\n" + "=" * 76)
    print(f"👁️‍🗨️  {title}")
    print("=" * 76)


def main():
    ox.manual_seed(42)
    np.random.seed(42)

    banner("Oxideon Multimodal Vision-Language (VLM) & Fusion Architecture")

    batch_size = 4
    num_patches = 49      # 7x7 visual patches from a Vision Transformer (ViT)
    visual_dim = 128      # Visual encoder embedding dimension
    text_dim = 64         # Language model (LLM) embedding dimension
    text_seq_len = 16     # User prompt token count
    num_latents = 12      # Fixed compressed latents for Perceiver Resampler

    print(f"Configurations:")
    print(f"  • Batch size: {batch_size}")
    print(f"  • Visual patches per image: {num_patches} (dim = {visual_dim})")
    print(f"  • Text prompt length: {text_seq_len} tokens (dim = {text_dim})")
    print(f"  • Compressed visual latents: {num_latents}")

    # Synthesize simulated ViT visual patch features and LLM prompt tokens
    visual_patches = ox.randn([batch_size, num_patches, visual_dim])
    text_tokens = ox.randn([batch_size, text_seq_len, text_dim])

    # -----------------------------------------------------------------------
    # Step 1: LLaVA-style MLP Vision-Language Connector
    # -----------------------------------------------------------------------
    print("\n[Step 1] Constructing LLaVA-style MLP Connector...")
    llava_connector = VisionLanguageConnector(
        visual_dim=visual_dim,
        text_dim=text_dim,
        mode="mlp",
    )
    t0 = time.perf_counter()
    llava_prefix = llava_connector(visual_patches, text_tokens)
    t1 = time.perf_counter()

    expected_llava_seq = num_patches + text_seq_len
    print(f"  • LLaVA sequence shape: {llava_prefix.shape} (patches {num_patches} + text {text_seq_len} = {expected_llava_seq})")
    print(f"  • Projection latency: {(t1 - t0) * 1000.0:.2f} ms")
    assert llava_prefix.shape == (batch_size, expected_llava_seq, text_dim)

    # -----------------------------------------------------------------------
    # Step 2: Flamingo-style Perceiver Resampler Connector
    # -----------------------------------------------------------------------
    print("\n[Step 2] Constructing Flamingo-style Perceiver Resampler...")
    flamingo_connector = VisionLanguageConnector(
        visual_dim=visual_dim,
        text_dim=text_dim,
        mode="resampler",
        num_latents=num_latents,
    )
    t0 = time.perf_counter()
    flamingo_prefix = flamingo_connector(visual_patches, text_tokens)
    t1 = time.perf_counter()

    expected_flamingo_seq = num_latents + text_seq_len
    print(f"  • Flamingo sequence shape: {flamingo_prefix.shape} (latents {num_latents} + text {text_seq_len} = {expected_flamingo_seq})")
    print(f"  • Token compression ratio: {num_patches / num_latents:.1f}x fewer visual tokens passed to LLM")
    print(f"  • Resampling latency: {(t1 - t0) * 1000.0:.2f} ms")
    assert flamingo_prefix.shape == (batch_size, expected_flamingo_seq, text_dim)

    # -----------------------------------------------------------------------
    # Step 3: Gated Multimodal Fusion (GMU)
    # -----------------------------------------------------------------------
    print("\n[Step 3] Evaluating Gated Multimodal Fusion (GMU)...")
    # Pool sequences to representation vectors [B, D]
    img_repr = visual_patches.mean(dim=1)  # [B, visual_dim]
    txt_repr = text_tokens.mean(dim=1)     # [B, text_dim]

    gated_fusion = GatedMultimodalFusion(dim_a=visual_dim, dim_b=text_dim, out_dim=96)
    fused_repr = gated_fusion(img_repr, txt_repr)
    print(f"  • Image input: {img_repr.shape}, Text input: {txt_repr.shape}")
    print(f"  • Dynamically gated joint representation: {fused_repr.shape}")
    assert fused_repr.shape == (batch_size, 96)

    # -----------------------------------------------------------------------
    # Step 4: Multi-Head Cross-Attention Fusion
    # -----------------------------------------------------------------------
    print("\n[Step 4] Evaluating Multi-Head Cross-Attention Fusion...")
    cross_attn = CrossAttentionFusion(d_model=text_dim, num_heads=4)
    # Text attends to projected visual tokens
    projected_visual = llava_connector.projector(visual_patches)  # [B, num_patches, text_dim]
    attended_text = cross_attn(text_tokens, projected_visual)
    print(f"  • Text query {text_tokens.shape} cross-attended to Visual KV {projected_visual.shape}")
    print(f"  • Fused text output shape: {attended_text.shape}")
    assert attended_text.shape == text_tokens.shape

    # -----------------------------------------------------------------------
    # Step 5: Multi-Modal Contrastive Alignment via SigLIP
    # -----------------------------------------------------------------------
    print("\n[Step 5] Training Multimodal Projection with SigLIP Loss & Autograd...")
    proj_img = ox.nn.Linear(visual_dim, 32)
    proj_txt = ox.nn.Linear(text_dim, 32)
    optimizer = AdamW(list(proj_img.parameters()) + list(proj_txt.parameters()), lr=1e-2)

    initial_loss = 0.0
    final_loss = 0.0

    print("  Iter | SigLIP Pairwise Loss | Loss Reduction")
    print("  -----|----------------------|---------------")

    for step in range(1, 11):
        optimizer.zero_grad()
        # Embeddings normalized along feature dimension
        z_i = proj_img(img_repr)
        z_t = proj_txt(txt_repr)

        # SigLIP loss between vision and text embeddings
        loss = ox.siglip_loss(z_i, z_t, temperature=10.0, bias=-10.0)
        curr_loss = float(loss.to_numpy().flat[0])

        if step == 1:
            initial_loss = curr_loss
        final_loss = curr_loss

        loss.backward()
        optimizer.step()

        reduction_pct = (1.0 - curr_loss / initial_loss) * 100.0 if initial_loss > 0 else 0.0
        if step % 2 == 0 or step == 1:
            print(f"  {step:4d} | {curr_loss:20.5f} | {reduction_pct:12.1f}%")

    print(f"\n  ✓ Loss successfully converged from {initial_loss:.4f} down to {final_loss:.4f}.")
    assert final_loss < initial_loss, "Expected training loss to decrease"

    banner("Multimodal VLM & Fusion Showcase Completed Successfully! ✅")


if __name__ == "__main__":
    main()
