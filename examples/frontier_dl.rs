//! Showcase of Oxideon's frontier deep learning subsystems:
//! 1. Parameter-Efficient Fine-Tuning (PEFT) with [`LoRALinear`] + weight merging
//! 2. Block-wise Post-Training Quantization ([`QuantScheme::DEFAULT_Q8_0`] & [`QuantScheme::DEFAULT_Q4_0`])
//! 3. [`GroupedQueryAttention`] (`GQA`) & incremental [`KVCache`] decoding on [`CausalLM`]
//! 4. Differentiable Spectral Ops (`fft`, `rfft`, `stft`) & Matrix Decompositions (`eigh`, `svd`)
//! 5. [`VisionTransformer`] (`ViT`) and [`UNet2d`] + [`GradScaler`] & [`EMA`]

use oxideon::prelude::*;

fn main() -> Result<()> {
    manual_seed(42);
    println!("=== Oxideon Frontier Deep Learning Showcase ===");

    // 1. LoRA Parameter-Efficient Fine-Tuning (PEFT)
    let base = Linear::new(128, 64)?;
    let lora = LoRALinear::from_linear(&base, 4, 8.0, 0.0)?;
    println!(
        "\n[1] LoRALinear: frozen base params = {}, trainable adapter params = {} ({:.1}% of base)",
        lora.num_frozen_parameters(),
        lora.num_parameters(),
        100.0 * (lora.num_parameters() as f64) / (lora.num_frozen_parameters() as f64)
    );
    let x = Tensor::randn(&[8, 128])?;
    let target = Tensor::randn(&[8, 64])?;
    let mut opt = Adam::new(lora.parameters(), 0.02);
    let initial_loss = nn::loss::mse_loss(&lora.forward(&x)?, &target)?.item::<f32>()?;
    for _ in 0..25 {
        let pred = lora.forward(&x)?;
        let loss = nn::loss::mse_loss(&pred, &target)?;
        opt.zero_grad();
        loss.backward()?;
        opt.step()?;
    }
    let final_loss = nn::loss::mse_loss(&lora.forward(&x)?, &target)?.item::<f32>()?;
    lora.merge()?;
    println!("    LoRA MSE loss: {initial_loss:.4} -> {final_loss:.4} (merged into base weight)");

    // 2. Block-wise Weight Quantization (Q8_0 and Q4_0)
    let merged_linear = lora.to_merged_linear()?;
    let q8 = QuantizedLinear::from_linear(&merged_linear, QuantScheme::DEFAULT_Q8_0)?;
    let q4 = QuantizedLinear::from_linear(&merged_linear, QuantScheme::DEFAULT_Q4_0)?;
    let ref_out = merged_linear.forward(&x)?;
    let q8_mae = ref_out
        .sub(&q8.forward(&x)?)?
        .abs()?
        .mean()?
        .item::<f32>()?;
    let q4_mae = ref_out
        .sub(&q4.forward(&x)?)?
        .abs()?
        .mean()?
        .item::<f32>()?;
    println!(
        "\n[2] Weight Quantization:\n    Q8_0 compression = {:.2}x (MAE = {:.5})\n    Q4_0 compression = {:.2}x (MAE = {:.5})",
        q8.qweight.compression_ratio_f32(),
        q8_mae,
        q4.qweight.compression_ratio_f32(),
        q4_mae
    );

    // 3. CausalLM with Stateful KVCache Incremental Decoding
    let lm = CausalLM::new(32, 64, 2, 4, 128, 64)?;
    let prompt = Tensor::from_slice(&[1i64, 5, 9, 12, 3, 7], &[1, 6])?;
    let full_logits = lm.forward(&prompt)?;
    let mut cache = KVCache::new(lm.blocks.len())?;
    let _prefill = lm.forward_cached(&prompt.narrow(1, 0, 5)?, &mut cache)?;
    let last_step_logits = lm.forward_cached(&prompt.narrow(1, 5, 1)?, &mut cache)?;
    let full_last = full_logits.narrow(1, 5, 1)?;
    let cache_diff = full_last
        .sub(&last_step_logits)?
        .abs()?
        .mean()?
        .item::<f32>()?;
    println!(
        "\n[3] CausalLM + KVCache: cached seq_len = {}, last-token logit diff vs full pass = {:.2e}",
        cache.seq_len(),
        cache_diff
    );

    // 4. Differentiable FFT/STFT & SVD/Eigh
    let signal = Tensor::randn(&[2, 64])?.requires_grad_(true);
    let spec = signal.stft(16, 8, true)?;
    let spec_energy = spec.complex_abs()?.square()?.mean()?;
    spec_energy.backward()?;
    let mat = Tensor::randn(&[6, 4])?.requires_grad_(true);
    let (_u, s, _vt) = mat.svd()?;
    let nuclear_norm = s.sum()?;
    nuclear_norm.backward()?;
    println!(
        "\n[4] Spectral & Linalg Autograd:\n    STFT shape = {:?}, nuclear norm ||A||_* = {:.4}",
        spec.shape(),
        nuclear_norm.item::<f32>()?
    );

    // 5. VisionTransformer (ViT) & UNet2d
    let vit = VisionTransformer::new(16, 4, 3, 10, 32, 2, 4, 64, 0.0)?;
    let unet = UNet2d::new(3, 2, 8)?;
    let images = Tensor::randn(&[2, 3, 16, 16])?;
    let cls_logits = vit.forward(&images)?;
    let seg_map = unet.forward(&images)?;
    println!(
        "\n[5] Vision Architectures:\n    ViT output shape = {:?} ({} params)\n    UNet2d output shape = {:?} ({} params)",
        cls_logits.shape(),
        vit.num_parameters(),
        seg_map.shape(),
        unet.num_parameters()
    );

    // 6. FlashAttention-2, MambaBlock (Selective SSM) & SparseMoE
    let q = Tensor::randn(&[2, 4, 16, 16])?;
    let k = Tensor::randn(&[2, 4, 16, 16])?;
    let v = Tensor::randn(&[2, 4, 16, 16])?;
    let fa_out = flash_attention(&q, &k, &v, 4, 4, true)?;
    let mamba = MambaBlock::new(32, 2, 8)?;
    let moe = SparseMoE::new(32, 64, 4, 2)?;
    let seq_in = Tensor::randn(&[2, 8, 32])?;
    let mamba_out = mamba.forward(&seq_in)?;
    let (moe_out, aux_loss) = moe.forward_with_aux_loss(&mamba_out)?;
    println!(
        "\n[6] FlashAttention-2, Mamba SSM & SparseMoE:\n    FlashAttention shape = {:?}, Mamba->MoE shape = {:?}, MoE load-balance loss = {:.4}",
        fa_out.shape(),
        moe_out.shape(),
        aux_loss.item::<f32>()?
    );

    // 7. Functional Calculus (Jacobian, Hessian, NTK), DDPM/DDIM Diffusion, BPE & GNNs
    let x_vec = Tensor::from_slice(&[1.2f32, -0.7, 0.5], &[3])?;
    let h_mat = hessian(|z| z.powf(4.0)?.sum(), &x_vec)?;
    let ddpm = DDPMScheduler::new(100, BetaSchedule::Cosine { s: 0.008 })?;
    let x0 = Tensor::randn(&[2, 3, 8, 8])?;
    let noise = Tensor::randn(&[2, 3, 8, 8])?;
    let xt = ddpm.add_noise(&x0, &noise, &[40, 40])?;
    let x0_ddim = ddpm.step_ddim(&noise, 40, None, &xt, 0.0)?;
    let ddim_err = x0.sub(&x0_ddim)?.abs()?.mean()?.item::<f32>()?;

    let bpe = BpeTokenizer::train(
        "oxideon deep learning rust tensor autograd oxideon rust",
        280,
    );
    let sample_txt = "oxideon rust tensor";
    let tok_ids = bpe.encode(sample_txt);
    assert_eq!(bpe.decode(&tok_ids), sample_txt);

    let adj = Tensor::from_slice(&[0.0f32, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0], &[3, 3])?;
    let gat = GATConv::new(8, 4, 0.2)?;
    let gat_out = gat.forward_graph(&Tensor::randn(&[3, 8])?, &adj)?;
    println!(
        "\n[7] Functional Calculus, Diffusion, BPE & GNN:\n    Hessian diag = {:?}\n    DDIM exact-noise reconstruction MAE = {:.2e}\n    BPE '{}' ({} bytes) -> {} tokens {:?}\n    GATConv output shape = {:?}",
        h_mat.diagonal(0, 0, 1)?.to_vec::<f32>()?,
        ddim_err,
        sample_txt,
        sample_txt.len(),
        tok_ids.len(),
        tok_ids,
        gat_out.shape()
    );

    // 8. Graph Profiler, Muon & L-BFGS, Neural ODEs / Flow Matching, Speculative Decoding, KAN & Trainer
    let kan = KAN::new(&[8, 16, 4], 5)?;
    let kan_in = Tensor::randn(&[4, 8])?.requires_grad_(true);
    let prof = profile_forward_backward(|| kan.forward(&kan_in))?;
    let y_rk4 = odeint(
        |_t, y| y.mul_scalar(-1.0),
        &Tensor::ones_f32(&[3])?,
        0.0,
        1.0,
        10,
        OdeMethod::RK4,
    )?;
    let draft_lm = CausalLM::new(32, 32, 1, 2, 64, 64)?;
    let (spec_tokens, accepted, drafted) = lm.speculative_generate(&draft_lm, &[1, 5, 9], 6, 3)?;
    println!(
        "\n[8] Graph Profiler, KAN, Neural ODE & Speculative Decoding:\n    KAN graph nodes = {} (ops = {}, est FLOPs = {})\n    RK4 odeint y(1) = {:.6} (exact e^-1 = {:.6})\n    Speculative decoding generated {:?} (accepted {}/{} draft tokens)",
        prof.graph_stats.total_nodes,
        prof.graph_stats.op_nodes,
        prof.graph_stats.estimated_flops,
        y_rk4.to_vec::<f32>()?[0],
        (-1.0f32).exp(),
        spec_tokens,
        accepted,
        drafted
    );

    // 9. Symbolic IR Compiler & Fusion, Sparse CSR SpMM, Distributions/Gumbel-Softmax, vmap & Compression
    let mut ir = IrGraph::new();
    let ix = ir.add_input();
    let iw = ir.add_input();
    let ib = ir.add_input();
    let iz = ir.push_op(IrOp::ZeroLike { src: ix });
    let x_plus_0 = ir.push_op(IrOp::Add(ix, iz));
    let mm_a = ir.push_op(IrOp::Matmul(x_plus_0, iw));
    let mm_b = ir.push_op(IrOp::Matmul(x_plus_0, iw)); // duplicate -> CSE
    let lin = ir.push_op(IrOp::Add(mm_a, ib));
    let act = ir.push_op(IrOp::Relu(lin)); // Matmul + Add + Relu -> FusedLinearRelu
    let _dead_node = ir.push_op(IrOp::Exp(mm_b)); // dead -> DCE
    let fma_mul = ir.push_op(IrOp::Mul(act, act));
    let ir_out = ir.push_op(IrOp::Add(fma_mul, act)); // Mul + Add -> FusedFma
    ir.set_outputs(vec![ir_out]);
    let ir_rep = ir.optimize()?;

    let sp_dense = Tensor::from_slice(&[2.0f32, 0.0, -1.0, 0.0, 0.0, 3.5, 1.0, 0.0], &[2, 4])?;
    let csr = SparseCsr::from_dense(&sp_dense, 1e-6)?;
    let sp_out = csr.spmm(&Tensor::ones_f32(&[4, 3])?.requires_grad_(true))?;

    let cat = Categorical::from_logits(Tensor::from_slice(&[0.5f32, 2.0, -1.0], &[1, 3])?);
    let gumbel_oh = cat.gumbel_softmax(0.5, true)?;

    let vm_out = vmap(
        |row| row.norm_p(2.0, &[0], false),
        &Tensor::randn(&[5, 4])?,
        0,
        0,
    )?;
    let mask = prune_magnitude_unstructured(&merged_linear, 0.5)?;
    println!(
        "\n[9] IR Compiler, Sparse SpMM, Distributions, vmap & Compression:\n    IR optimize: {} -> {} nodes (fused_ops = {})\n    Sparse CSR: nnz={} (sparsity={:.1}%), spmm shape={:?}\n    Straight-Through Gumbel-Softmax one-hot={:?}, vmap row-L2 shape={:?}, pruned sparsity={:.1}%",
        ir_rep.nodes_before,
        ir_rep.nodes_after,
        ir_rep.fused_ops_count,
        csr.nnz(),
        csr.sparsity() * 100.0,
        sp_out.shape(),
        gumbel_oh.to_vec::<f32>()?,
        vm_out.shape(),
        mask.overall_sparsity()? * 100.0
    );

    // 10. Symbolic IR Autodiff + Memory Planner, Tensor Parallelism, Spatial Vision, PINNs, MLA/DiffAttn/Hyena & XAI
    let mut ad_ir = IrGraph::new();
    let sym_x = ad_ir.add_input();
    let sym_w = ad_ir.add_input();
    let sym_mm = ad_ir.push_op(IrOp::Matmul(sym_x, sym_w));
    let sym_act = ad_ir.push_op(IrOp::Tanh(sym_mm));
    let sym_loss = ad_ir.push_op(IrOp::SumAll(sym_act));
    let sym_grads = ad_ir.differentiate(sym_loss, &[sym_x, sym_w])?;
    ad_ir.set_outputs(vec![sym_loss, sym_grads[0], sym_grads[1]]);
    let compiled = ad_ir.compile()?;

    let mla = MultiHeadLatentAttention::new(64, 8, 16, 32)?;
    let diff_attn = DiffAttention::new(64, 8, 1)?;
    let hyena = HyenaOperator::new(64, 2, 3)?;
    let seq64 = Tensor::randn(&[2, 8, 64])?;
    let mla_out = mla.forward(&seq64)?;
    let da_out = diff_attn.forward(&mla_out)?;
    let hy_out = hyena.forward(&da_out)?;

    let theta = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0], &[1, 2, 3])?;
    let stn_grid = affine_grid(&theta, [1, 3, 8, 8], true)?;
    let stn_out = grid_sample(&Tensor::randn(&[1, 3, 8, 8])?, &stn_grid, true)?;

    let pinn_xy = Tensor::from_slice(&[1.0f32, 2.0, -1.0, 0.5], &[2, 2])?.requires_grad_(true);
    let lap = laplacian(|xy| xy.square()?.sum_axis(1, true), &pinn_xy)?;

    let ig = integrated_gradients(
        |z| z.powf(3.0)?.sum(),
        &Tensor::from_slice(&[2.0f32, -1.0], &[1, 2])?,
        None,
        32,
    )?;
    println!(
        "\n[10] Symbolic IR Autodiff + Memory Planner, MLA/DiffAttn/Hyena, STN, PINNs & XAI:\n    Compiled IR fwd+bwd: {} live values -> {} peak buffer slots ({} reused, {:.1}% reuse)\n    MLA KV-cache compression = {:.1}x, MLA->DiffAttn->Hyena shape = {:?}\n    STN grid_sample shape = {:?}, PINN Laplacian(x^2 + y^2) = {:?}\n    Integrated Gradients sum = {:.4} (exact f(x)-f(0) = 7.0000)",
        compiled.memory_plan.total_live_values,
        compiled.memory_plan.peak_slots,
        compiled.memory_plan.reused_allocations,
        compiled.memory_plan.reuse_ratio() * 100.0,
        mla.kv_cache_compression_ratio(),
        hy_out.shape(),
        stn_out.shape(),
        lap.to_vec::<f32>()?,
        ig.sum()?.item::<f32>()?
    );

    // 11. Complete Frontier Foundation Models: DeepSeek-V3 (MLA+MoE+MTP), HybridJambaLM, MultimodalVLM & DiT
    let deepseek = DeepSeekMoELM::new(64, 32, 2, 4, 8, 16, 64, 1, 4, 2, 2)?;
    let ds_toks = Tensor::from_slice(&[1i64, 5, 9, 12, 3, 7, 2, 8, 4, 6, 11, 15], &[2, 6])?;
    let (ds_main, ds_mtp, ds_aux) = deepseek.forward_with_mtp(&ds_toks)?;

    let jamba = HybridJambaLM::new(64, 32, 4, 4, 2, 8, 64, 4, 2, 2, 2)?;
    let jamba_out = jamba.forward(&ds_toks)?;

    let vlm = MultimodalVLM::new(16, 4, 3, 32, 1, 4, 64, 32, 2, 4, 2, 64, 64)?;
    let vlm_img = Tensor::randn(&[1, 3, 16, 16])?;
    let vlm_gen = vlm.generate_multimodal(&vlm_img, &[1, 5, 9], 4)?;

    let dit = DiT::new(16, 4, 3, 32, 2, 4, 10)?;
    let dit_out = dit.forward_dit(
        &Tensor::randn(&[2, 3, 16, 16])?,
        &[20, 80],
        Some(&Tensor::from_slice(&[3i64, 7], &[2])?),
    )?;
    println!(
        "\n[11] Frontier Foundation Model Suite (DeepSeek-V3, HybridJamba, MultimodalVLM, DiT):\n    DeepSeekMoELM ({} params): main={:?}, MTP depths=[{:?}, {:?}], MoE aux={:.4}\n    HybridJambaLM ({} params): Mamba+GQA+MoE output={:?}\n    MultimodalVLM ({} params): image+text generated tokens={:?}\n    DiT AdaLN-Zero ({} params): conditioned velocity field={:?}",
        deepseek.num_parameters(),
        ds_main.shape(),
        ds_mtp[0].shape(),
        ds_mtp[1].shape(),
        ds_aux.item::<f32>()?,
        jamba.num_parameters(),
        jamba_out.shape(),
        vlm.num_parameters(),
        vlm_gen,
        dit.num_parameters(),
        dit_out.shape()
    );

    // 12. Fused SIMD Kernels, Dynamic JIT (`jit_compile`), Conv3d/PixelShuffle, AWQ/FP8 & CachingAllocator
    let x_f = Tensor::randn(&[4, 64])?.requires_grad_(true);
    let r_f = Tensor::randn(&[4, 64])?.requires_grad_(true);
    let w_rms = Tensor::ones(&[64], DType::F32, &Device::Cpu)?.requires_grad_(true);
    let (normed_f, _res_f) = fused_add_rmsnorm(&x_f, &r_f, &w_rms, 1e-5)?;
    let swi_f = fused_swiglu(&normed_f, &r_f)?;
    swi_f.sum()?.backward()?;

    let jit_w = Tensor::randn(&[64, 32])?;
    let jit_b = Tensor::randn(&[4, 32])?;
    let jit_fn = jit_compile(
        |ins| ins[0].matmul(&ins[1])?.add(&ins[2])?.silu(),
        &[x_f.detach(), jit_w.clone(), jit_b.clone()],
    )?;
    let jit_res = jit_fn.execute(&[x_f.detach(), jit_w, jit_b])?;

    let c3d = Conv3d::new(
        2,
        4,
        (3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        true,
        DType::F32,
        Device::Cpu,
    )?;
    let v3d = c3d.forward(&Tensor::randn(&[1, 2, 4, 8, 8])?)?;
    let ps_hr = PixelShuffle::new(2).forward(&Tensor::randn(&[1, 12, 8, 8])?)?;

    let awq_layer = AWQLinear::from_linear_calibrated(
        &Linear::new(64, 32)?,
        &x_f.detach(),
        QuantScheme::Q4_0 { block_size: 32 },
        0.5,
    )?;
    let awq_out = awq_layer.forward(&x_f.detach())?;

    let pool = CachingAllocator::global();
    let b1 = pool.alloc_f32(4096)?;
    pool.recycle_f32(b1);
    let b2 = pool.alloc_f32(4096)?;
    pool.recycle_f32(b2);
    let p_stats = pool.stats();
    println!(
        "\n[12] Fused Single-Pass Kernels, Dynamic JIT (`jit_compile`), Conv3d, AWQ & CachingAllocator:\n    fused_add_rmsnorm + fused_swiglu grad norm = {:.4}, jit_compile fused ops = {} (out = {:?})\n    Conv3d output = {:?}, PixelShuffle 2x SR = {:?}, AWQ Q4_0 output = {:?}\n    CachingAllocator hits = {}, misses = {}, peak_active_bytes = {}",
        x_f.grad().unwrap().norm()?.item::<f32>()?,
        jit_fn.report.fused_ops_count,
        jit_res[0].shape(),
        v3d.shape(),
        ps_hr.shape(),
        awq_out.shape(),
        p_stats.cache_hits,
        p_stats.cache_misses,
        p_stats.peak_active_bytes
    );

    Ok(())
}
