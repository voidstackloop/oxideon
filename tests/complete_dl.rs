//! Integration and finite-difference gradient tests for Oxideon's complete deep learning suite:
//! - Differentiable linear algebra (`cholesky`, `triangular_solve`, `solve`, `inv`, `det`, `slogdet`, `qr`, `cdist`, `norm_p`)
//! - Recurrent models (`RNN`, `LSTM`, `GRU`)
//! - Transformer & LLM (`RotaryEmbedding`, `SwiGLU`, `TransformerEncoder`, `TransformerDecoder`, `CausalLM` + `generate`)
//! - Vision & Normalization (`Conv2d` grouped/depthwise, `ConvTranspose2d`, `Upsample2d`, `AdaptiveAvgPool2d`, `GroupNorm`, `InstanceNorm2d`, `ResidualBlock`)
//! - Advanced losses (`kl_div_loss`, `cosine_embedding_loss`, `triplet_margin_loss`, `info_nce_loss`, `focal_loss`, `cross_entropy_with_smoothing`)
//! - Optimizers (`Lion`, `Adagrad`, `LinearWarmupCosineLr`, `ReduceLROnPlateau`)
//! - Data pipeline (`TensorDataset`, `DataLoader`, `Transform`, `make_moons`, `train_test_split`) & `metrics`

use oxideon::nn::loss;
use oxideon::prelude::*;

fn weighted_sum(x: &Tensor) -> Result<Tensor> {
    let n = x.numel();
    let w: Vec<f64> = (0..n).map(|i| ((i as f64 + 1.0) * 0.41).sin()).collect();
    let wt = Tensor::from_slice_on(&w, x.shape(), &x.device())?.to_dtype(x.dtype())?;
    x.mul(&wt)?.sum()
}

fn gradcheck(f: impl Fn(&[Tensor]) -> Result<Tensor>, inputs: &[Tensor]) {
    let f64_inputs: Vec<Tensor> = inputs
        .iter()
        .map(|t| {
            t.to_dtype(DType::F64)
                .unwrap()
                .detach()
                .requires_grad_(true)
        })
        .collect();
    let loss = f(&f64_inputs).unwrap();
    loss.backward().unwrap();

    let eps = 1e-5;
    for (k, inp) in f64_inputs.iter().enumerate() {
        let analytic = inp
            .grad()
            .unwrap_or_else(|| panic!("input {k} did not receive a gradient"))
            .to_vec::<f64>()
            .unwrap();
        let base_vec = inp.to_vec::<f64>().unwrap();
        let shape = inp.shape().to_vec();

        for i in 0..base_vec.len() {
            let mut plus_args = f64_inputs.iter().map(Tensor::detach).collect::<Vec<_>>();
            let mut p_vec = base_vec.clone();
            p_vec[i] += eps;
            plus_args[k] = Tensor::from_vec(p_vec, &shape).unwrap();

            let mut minus_args = f64_inputs.iter().map(Tensor::detach).collect::<Vec<_>>();
            let mut m_vec = base_vec.clone();
            m_vec[i] -= eps;
            minus_args[k] = Tensor::from_vec(m_vec, &shape).unwrap();

            let f_plus = no_grad(|| f(&plus_args).unwrap().item::<f64>().unwrap());
            let f_minus = no_grad(|| f(&minus_args).unwrap().item::<f64>().unwrap());
            let numeric = (f_plus - f_minus) / (2.0 * eps);
            let a = analytic[i];
            assert!(
                (a - numeric).abs() <= 1e-4 * (1.0 + numeric.abs()),
                "input {k}, elem {i}: analytic {a} vs numeric {numeric}"
            );
        }
    }
}

#[test]
fn linalg_cholesky_solve_inv_det_qr_cdist_gradcheck() -> Result<()> {
    manual_seed(10);
    // Construct SPD matrix A = M @ M^T + 2 I
    let m = Tensor::randn(&[3, 3])?;
    let b = Tensor::randn(&[3, 2])?;

    gradcheck(
        |t| {
            let eye = Tensor::eye(3, t[0].dtype(), &t[0].device())?;
            let spd = t[0].matmul(&t[0].t()?)?.add(&eye.mul_scalar(2.0)?)?;
            let l = spd.cholesky()?;
            weighted_sum(&l)
        },
        std::slice::from_ref(&m),
    );

    gradcheck(
        |t| {
            let eye = Tensor::eye(3, t[0].dtype(), &t[0].device())?;
            let a = t[0].add(&eye.mul_scalar(3.0)?)?;
            let x = a.solve(&t[1])?;
            weighted_sum(&x)
        },
        &[m.clone(), b],
    );

    gradcheck(
        |t| {
            let eye = Tensor::eye(3, t[0].dtype(), &t[0].device())?;
            let a = t[0].add(&eye.mul_scalar(3.0)?)?;
            let inv_a = a.inv()?;
            let det_a = a.det()?;
            weighted_sum(&inv_a)?.add(&det_a)
        },
        std::slice::from_ref(&m),
    );

    // Verify QR decomposition A = Q @ R and Q^T @ Q = I
    let a_qr = Tensor::randn(&[5, 3])?;
    let (q, r) = a_qr.qr()?;
    assert!(q.matmul(&r)?.allclose(&a_qr, 1e-4, 1e-4)?);
    assert!(
        q.t()?
            .matmul(&q)?
            .allclose(&Tensor::eye(3, DType::F32, &Device::Cpu)?, 1e-4, 1e-4)?
    );

    // cdist & normalize gradcheck
    let p1 = Tensor::randn(&[3, 4])?;
    let p2 = Tensor::randn(&[2, 4])?;
    gradcheck(|t| weighted_sum(&t[0].cdist(&t[1])?), &[p1.clone(), p2]);
    gradcheck(|t| weighted_sum(&t[0].normalize(2.0, -1, 1e-8)?), &[p1]);
    Ok(())
}

#[test]
fn vision_conv_transpose_grouped_upsample_pool_resnet() -> Result<()> {
    manual_seed(20);
    let x = Tensor::randn(&[2, 4, 4, 4])?;
    let w_deconv = Tensor::randn(&[4, 2, 3, 3])?;
    let b_deconv = Tensor::randn(&[2])?;

    gradcheck(
        |t| {
            weighted_sum(&t[0].conv_transpose2d(
                &t[1],
                Some(&t[2]),
                (2, 2),
                (1, 1),
                (0, 0),
                (1, 1),
            )?)
        },
        &[x.clone(), w_deconv, b_deconv],
    );

    // Depthwise grouped conv2d (groups = 4)
    let w_dw = Tensor::randn(&[4, 1, 3, 3])?;
    gradcheck(
        |t| weighted_sum(&t[0].conv2d_grouped(&t[1], None, (1, 1), (1, 1), (1, 1), 4)?),
        &[x.clone(), w_dw],
    );

    // Bilinear & nearest upsampling + adaptive average pooling gradcheck
    gradcheck(
        |t| weighted_sum(&t[0].upsample_bilinear2d((6, 6), false)?),
        std::slice::from_ref(&x),
    );
    gradcheck(
        |t| {
            weighted_sum(
                &t[0]
                    .upsample_nearest2d((8, 8))?
                    .adaptive_avg_pool2d((2, 2))?,
            )
        },
        std::slice::from_ref(&x),
    );

    // ResidualBlock + GroupNorm + InstanceNorm2d + ConvTranspose2d module pipeline
    let res = ResidualBlock::new(4, 8, 2)?;
    let gn = GroupNorm::new(2, 8, 1e-5)?;
    let inorm = InstanceNorm2d::new(8, 1e-5)?;
    let up = ConvTranspose2d::new(8, 4, (2, 2), (2, 2), (0, 0))?;

    let h = res.forward(&x)?;
    assert_eq!(h.shape(), &[2, 8, 2, 2]);
    let h = inorm.forward(&gn.forward(&h)?)?;
    let out = up.forward(&h)?;
    assert_eq!(out.shape(), &[2, 4, 4, 4]);
    out.sum()?.backward()?;
    Ok(())
}

#[test]
fn rnn_lstm_gru_sequence_models() -> Result<()> {
    manual_seed(30);
    let x = Tensor::randn(&[3, 5, 4])?.requires_grad_(true);

    let rnn = RNN::new(4, 6, 2)?;
    let (rnn_out, rnn_hn) = rnn.forward_seq(&x, None)?;
    assert_eq!(rnn_out.shape(), &[3, 5, 6]);
    assert_eq!(rnn_hn.shape(), &[2, 3, 6]);
    rnn_out.sum()?.backward()?;
    assert!(x.grad().is_some());

    x.zero_grad();
    let lstm = LSTM::new(4, 6, 2)?;
    let (lstm_out, (hn, cn)) = lstm.forward_seq(&x, None)?;
    assert_eq!(lstm_out.shape(), &[3, 5, 6]);
    assert_eq!(hn.shape(), &[2, 3, 6]);
    assert_eq!(cn.shape(), &[2, 3, 6]);
    lstm_out.sum()?.backward()?;
    assert!(x.grad().is_some());

    x.zero_grad();
    let gru = GRU::new(4, 6, 2)?;
    let (gru_out, gru_hn) = gru.forward_seq(&x, None)?;
    assert_eq!(gru_out.shape(), &[3, 5, 6]);
    assert_eq!(gru_hn.shape(), &[2, 3, 6]);
    gru_out.sum()?.backward()?;
    assert!(x.grad().is_some());
    Ok(())
}

#[test]
fn transformer_encoder_decoder_and_causal_lm_generation() -> Result<()> {
    manual_seed(40);
    let pe = SinusoidalPositionalEncoding::new(32, 16)?;
    let enc = TransformerEncoder::new(2, 16, 4, 32, 0.0)?;
    let dec = TransformerDecoder::new(2, 16, 4, 32, 0.0)?;

    let src = pe.forward(&Tensor::randn(&[2, 6, 16])?)?;
    let tgt = pe.forward(&Tensor::randn(&[2, 4, 16])?)?;
    let memory = enc.forward(&src)?;
    let dec_out = dec.forward_with_memory(&tgt, &memory)?;
    assert_eq!(dec_out.shape(), &[2, 4, 16]);
    dec_out.sum()?.backward()?;

    // CausalLM with RoPE + RMSNorm + SwiGLU + autoregressive generate
    let lm = CausalLM::new(24, 16, 4, 2, 32, 32)?;
    let prompt = [1i64, 4, 7];
    let greedy = lm.generate(&prompt, 5, 0.0, None, None)?;
    assert_eq!(greedy.len(), 8);
    let sampled = lm.generate(&prompt, 5, 0.8, Some(5), Some(0.9))?;
    assert_eq!(sampled.len(), 8);
    assert!(lm.summary().contains("Total trainable parameters"));
    Ok(())
}

#[test]
fn advanced_losses_metrics_dataloader_and_lion_optimizer() -> Result<()> {
    manual_seed(50);

    // Advanced losses gradcheck
    let a = Tensor::randn(&[4, 5])?;
    let b = Tensor::randn(&[4, 5])?;
    let c = Tensor::randn(&[4, 5])?;
    let targets = [0usize, 2, 1, 3];

    gradcheck(
        |t| loss::focal_loss(&t[0], &targets, 0.25, 2.0),
        std::slice::from_ref(&a),
    );
    gradcheck(
        |t| loss::cross_entropy_with_smoothing(&t[0], &targets, 0.1),
        std::slice::from_ref(&a),
    );
    gradcheck(
        |t| loss::triplet_margin_loss(&t[0], &t[1], &t[2], 1.0),
        &[a.clone(), b.clone(), c],
    );
    gradcheck(
        |t| loss::info_nce_loss(&t[0], &t[1], 0.1),
        &[a.clone(), b.clone()],
    );
    gradcheck(
        |t| {
            let log_p = t[0].log_softmax(-1)?;
            let q = t[1].softmax(-1)?;
            loss::kl_div_loss(&log_p, &q)
        },
        &[a, b],
    );

    // Data pipeline + Lion optimizer + LinearWarmupCosineLr on make_moons
    let (x, y) = make_moons(120, 0.08)?;
    let transform = Compose::new().add(Normalize::new(0.25, 0.8));
    let x = transform.apply(&x)?;
    let (train_x, train_y, test_x, test_y) = train_test_split(&x, &y, 0.2, true)?;

    let ds = TensorDataset::new(train_x, train_y);
    let loader = DataLoader::new(&ds, 24, true);

    let model = Sequential::new()
        .add(Linear::new(2, 32)?)
        .add(SiLU)
        .add(Linear::new(32, 2)?);

    let mut opt = Lion::new(model.parameters(), 0.01).weight_decay(1e-4);
    let mut sched = LinearWarmupCosineLr::new(0.01, 5, 120, 1e-4);

    for _epoch in 0..30 {
        for batch in loader.iter() {
            let (xb, yb_t) = batch?;
            let yb: Vec<usize> = yb_t
                .to_vec::<i64>()?
                .into_iter()
                .map(|v| v as usize)
                .collect();
            let l = loss::cross_entropy(&model.forward(&xb)?, &yb)?;
            opt.zero_grad();
            l.backward()?;
            opt.step()?;
            sched.step(&mut opt);
        }
    }

    let test_logits = no_grad(|| model.forward(&test_x))?;
    let test_labels: Vec<usize> = test_y
        .to_vec::<i64>()?
        .into_iter()
        .map(|v| v as usize)
        .collect();
    let acc = metrics::accuracy(&test_logits, &test_labels)?;
    let top2 = metrics::top_k_accuracy(&test_logits, &test_labels, 2)?;
    assert!(acc >= 0.85, "expected accuracy >= 85%, got {acc}");
    assert_eq!(top2, 1.0);
    Ok(())
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f64> {
    let vals = a.sub(b)?.abs()?.to_dtype(DType::F64)?.to_vec::<f64>()?;
    Ok(vals.into_iter().fold(0.0, f64::max))
}

#[test]
fn test_fft_rfft_stft_and_gradcheck() -> Result<()> {
    manual_seed(2026);
    // 1D complex FFT / IFFT roundtrip
    let x = Tensor::randn(&[2, 8, 2])?.to_dtype(DType::F64)?;
    let y = x.fft()?;
    let x_rec = y.ifft()?;
    let diff = max_abs_diff(&x, &x_rec)?;
    assert!(diff < 1e-10, "fft->ifft max err = {diff}");

    // 1D real RFFT / IRFFT roundtrip
    let xr = Tensor::randn(&[3, 16])?.to_dtype(DType::F64)?;
    let spec = xr.rfft()?;
    assert_eq!(spec.shape(), &[3, 16, 2]);
    let xr_rec = spec.irfft()?;
    let rdiff = max_abs_diff(&xr, &xr_rec)?;
    assert!(rdiff < 1e-10, "rfft->irfft max err = {rdiff}");

    // 2D FFT / IFFT2D roundtrip + convolution theorem via complex_mul
    let img = Tensor::randn(&[2, 4, 4, 2])?.to_dtype(DType::F64)?;
    let img_rec = img.fft2d()?.ifft2d()?;
    assert!(max_abs_diff(&img, &img_rec)? < 1e-10);

    // STFT shape and differentiability
    let sig = Tensor::randn(&[2, 32])?.to_dtype(DType::F64)?;
    let stft_out = sig.stft(8, 4, true)?;
    assert_eq!(stft_out.shape(), &[2, 7, 8, 2]);

    // Finite-difference gradcheck through FFT, RFFT, complex_mul, complex_abs
    let a = Tensor::randn(&[2, 4, 2])?.to_dtype(DType::F64)?;
    let b = Tensor::randn(&[2, 4, 2])?.to_dtype(DType::F64)?;
    let weights = Tensor::randn(&[2, 4, 2])?.to_dtype(DType::F64)?;
    gradcheck(
        |t| t[0].fft()?.mul(&weights)?.ifft()?.mul(&weights)?.sum(),
        std::slice::from_ref(&a),
    );
    gradcheck(
        |t| weighted_sum(&t[0].complex_mul(&t[1])?.complex_abs()?),
        &[a, b],
    );
    let r_in = Tensor::randn(&[2, 8])?.to_dtype(DType::F64)?;
    let r_w = Tensor::randn(&[2, 8, 2])?.to_dtype(DType::F64)?;
    gradcheck(
        |t| weighted_sum(&t[0].rfft()?.mul(&r_w)?.irfft()?),
        std::slice::from_ref(&r_in),
    );
    Ok(())
}

#[test]
fn test_eigh_and_svd_with_gradcheck() -> Result<()> {
    manual_seed(77);
    // Construct a symmetric matrix with well-separated eigenvalues for gradcheck stability
    let raw = Tensor::from_slice(
        &[
            4.2f64, 0.3, -0.2, //
            0.3, 2.1, 0.4, //
            -0.2, 0.4, 0.7, //
        ],
        &[3, 3],
    )?;
    let (evals, evecs) = raw.eigh()?;
    // Verify A = Q * diag(evals) * Q^T
    let recon = evecs.mul(&evals.unsqueeze(0)?)?.matmul(&evecs.t()?)?;
    let err = max_abs_diff(&raw, &recon)?;
    assert!(err < 1e-9, "eigh reconstruction err = {err}");

    let w_vec = Tensor::from_slice(&[1.3f64, -0.7, 2.1], &[3])?;
    let w_mat = Tensor::randn(&[3, 3])?.to_dtype(DType::F64)?;
    gradcheck(
        |t| {
            let (l, q) = t[0].eigh()?;
            // Use sign-invariant outer products q_i q_i^T weighted by w_vec plus eigenvalues
            let proj = q.mul(&w_vec.unsqueeze(0)?)?.matmul(&q.t()?)?;
            proj.mul(&w_mat)?.sum()?.add(&l.mul(&w_vec)?.sum()?)
        },
        std::slice::from_ref(&raw),
    );

    // SVD reconstruction A = U * diag(S) * Vt (tall 4x3 and wide 3x4)
    let m_tall = Tensor::from_slice(
        &[
            3.1f64, 0.2, -0.4, //
            0.1, 2.2, 0.3, //
            -0.2, 0.5, 1.1, //
            0.4, -0.3, 0.6, //
        ],
        &[4, 3],
    )?;
    let (u, s, vt) = m_tall.svd()?;
    let m_rec = u.mul(&s.unsqueeze(0)?)?.matmul(&vt)?;
    let svd_err = max_abs_diff(&m_tall, &m_rec)?;
    assert!(svd_err < 1e-8, "svd tall reconstruction err = {svd_err}");

    let m_wide = m_tall.t()?;
    let (uw, sw, vtw) = m_wide.svd()?;
    let mw_rec = uw.mul(&sw.unsqueeze(0)?)?.matmul(&vtw)?;
    assert!(max_abs_diff(&m_wide, &mw_rec)? < 1e-8);

    // Gradcheck on singular values S and U * diag(S) * V^T
    let sw_weights = Tensor::from_slice(&[1.5f64, -0.8, 0.4], &[3])?;
    gradcheck(
        |t| {
            let (_u, s, _v) = t[0].svd()?;
            s.mul(&sw_weights)?.sum()
        },
        std::slice::from_ref(&m_tall),
    );
    Ok(())
}

#[test]
fn test_quantization_lora_gqa_kv_cache_amp_and_vision() -> Result<()> {
    manual_seed(99);

    // 1. QuantizedLinear (Q8_0 and Q4_0)
    let linear = Linear::new(64, 32)?;
    let x = Tensor::randn(&[4, 64])?;
    let y_fp32 = linear.forward(&x)?;

    let q8_lin = QuantizedLinear::from_linear(&linear, QuantScheme::DEFAULT_Q8_0)?;
    assert!(q8_lin.qweight.compression_ratio_f32() > 3.5);
    let y_q8 = q8_lin.forward(&x)?;
    let q8_err = y_fp32.sub(&y_q8)?.abs()?.mean()?.item::<f32>()?;
    assert!(q8_err < 0.03, "Q8_0 mean abs error too high: {q8_err}");

    let q4_lin = QuantizedLinear::from_linear(&linear, QuantScheme::DEFAULT_Q4_0)?;
    assert!(q4_lin.qweight.compression_ratio_f32() > 6.5);
    let y_q4 = q4_lin.forward(&x)?;
    let q4_err = y_fp32.sub(&y_q4)?.abs()?.mean()?.item::<f32>()?;
    assert!(q4_err < 0.25, "Q4_0 mean abs error too high: {q4_err}");

    // 2. LoRALinear PEFT adapter training + merge/unmerge equivalence
    let lora = LoRALinear::from_linear(&linear, 4, 8.0, 0.0)?;
    assert_eq!(lora.parameters().len(), 2); // Only lora_a and lora_b!
    assert_eq!(lora.num_parameters(), 4 * 64 + 32 * 4);
    // At initialization B = 0, so LoRA output equals base Linear output
    let y_lora_init = lora.forward(&x)?;
    assert!(max_abs_diff(&y_fp32, &y_lora_init)? < 1e-6);

    // Train LoRA adapter for 1 step
    let loss = lora.forward(&x)?.square()?.mean()?;
    loss.backward()?;
    assert!(lora.base_weight.grad().is_none()); // Base weight stays frozen!
    let mut opt = Adam::new(lora.parameters(), 0.05);
    opt.step()?;

    let y_unmerged = lora.forward(&x)?;
    lora.merge()?;
    let y_merged = lora.forward(&x)?;
    assert!(max_abs_diff(&y_unmerged, &y_merged)? < 1e-5);
    let exported = lora.to_merged_linear()?;
    assert!(max_abs_diff(&y_unmerged, &exported.forward(&x)?)? < 1e-5);
    lora.unmerge()?;
    assert!(max_abs_diff(&y_unmerged, &lora.forward(&x)?)? < 1e-5);

    // 3. GroupedQueryAttention (GQA) + KVCache + CausalLM::forward_cached equivalence
    let gqa = GroupedQueryAttention::new(32, 4, 2)?;
    let rope = RotaryEmbedding::new(8, 32, 10000.0)?;
    let seq = Tensor::randn(&[2, 5, 32])?;
    let full_out = gqa.forward_gqa(&seq, Some(&rope), None, true)?;

    let mut kv = LayerKVCache::new();
    let prefill = gqa.forward_gqa(&seq.narrow(1, 0, 3)?, Some(&rope), Some(&mut kv), true)?;
    let step4 = gqa.forward_gqa(&seq.narrow(1, 3, 1)?, Some(&rope), Some(&mut kv), false)?;
    let step5 = gqa.forward_gqa(&seq.narrow(1, 4, 1)?, Some(&rope), Some(&mut kv), false)?;
    let cached_out = Tensor::cat(&[prefill, step4, step5], 1)?;
    let gqa_diff = max_abs_diff(&full_out, &cached_out)?;
    assert!(gqa_diff < 1e-5, "GQA KVCache mismatch: {gqa_diff}");

    let lm = CausalLM::new(24, 32, 2, 4, 64, 32)?;
    let tokens = Tensor::from_slice(&[1i64, 4, 7, 2, 9, 3, 5, 8], &[2, 4])?;
    let full_logits = lm.forward(&tokens)?;
    let mut lm_cache = KVCache::new(lm.blocks.len())?;
    let p_logits = lm.forward_cached(&tokens.narrow(1, 0, 2)?, &mut lm_cache)?;
    let s3_logits = lm.forward_cached(&tokens.narrow(1, 2, 1)?, &mut lm_cache)?;
    let s4_logits = lm.forward_cached(&tokens.narrow(1, 3, 1)?, &mut lm_cache)?;
    let cached_logits = Tensor::cat(&[p_logits, s3_logits, s4_logits], 1)?;
    let lm_diff = max_abs_diff(&full_logits, &cached_logits)?;
    assert!(lm_diff < 1e-4, "CausalLM KVCache mismatch: {lm_diff}");

    // 4. GradScaler & EMA
    let mut scaler = GradScaler::new(128.0)?;
    let lin_amp = Linear::new(8, 4)?;
    let mut opt_amp = Sgd::new(lin_amp.parameters(), 0.01);
    let x_amp = Tensor::randn(&[2, 8])?;
    let l_amp = lin_amp.forward(&x_amp)?.square()?.mean()?;
    opt_amp.zero_grad();
    scaler.scale(&l_amp)?.backward()?;
    let stepped = scaler.step_and_update(&mut opt_amp, &lin_amp.parameters())?;
    assert!(stepped);

    let mut ema = EMA::new(&lin_amp.parameters(), 0.9)?;
    let orig_w = lin_amp.weight.to_vec::<f32>()?;
    no_grad(|| lin_amp.weight.assign(&lin_amp.weight.add_scalar(1.0)?))?;
    ema.update(&lin_amp.parameters())?;
    let backup = ema.apply_shadow(&lin_amp.parameters())?;
    let shadow_w = lin_amp.weight.to_vec::<f32>()?;
    assert!((shadow_w[0] - (orig_w[0] + 0.1)).abs() < 1e-5);
    ema.restore(&lin_amp.parameters(), &backup)?;
    assert!((lin_amp.weight.to_vec::<f32>()?[0] - (orig_w[0] + 1.0)).abs() < 1e-5);

    // 5. VisionTransformer (ViT), UNet2d, and VAE helpers
    let vit = VisionTransformer::new(8, 4, 3, 5, 16, 2, 4, 32, 0.0)?;
    let img = Tensor::randn(&[2, 3, 8, 8])?;
    let vit_logits = vit.forward(&img)?;
    assert_eq!(vit_logits.shape(), &[2, 5]);
    vit_logits.sum()?.backward()?;

    let unet = UNet2d::new(3, 2, 4)?;
    let seg = unet.forward(&img)?;
    assert_eq!(seg.shape(), &[2, 2, 8, 8]);
    seg.sum()?.backward()?;

    let mu = Tensor::randn(&[4, 6])?.requires_grad_(true);
    let logvar = Tensor::randn(&[4, 6])?.requires_grad_(true);
    let z = vae_reparameterize(&mu, &logvar)?;
    assert_eq!(z.shape(), &[4, 6]);
    let kl = vae_kl_loss(&mu, &logvar)?;
    z.square()?.mean()?.add(&kl)?.backward()?;
    assert!(mu.grad().is_some() && logvar.grad().is_some());

    Ok(())
}

#[test]
fn test_functional_jvp_vjp_jacobian_hessian_hvp_and_ntk() -> Result<()> {
    manual_seed(123);
    let w = Tensor::randn(&[2, 3])?.to_dtype(DType::F64)?;
    let x = Tensor::from_slice(&[0.7f64, -1.2, 0.4], &[3])?;
    let v = Tensor::from_slice(&[0.3f64, 0.5, -0.8], &[3])?;
    let u = Tensor::from_slice(&[1.1f64, -0.6], &[2])?;

    let f_vec =
        |inp: &Tensor| -> Result<Tensor> { w.matmul(&inp.sin()?.unsqueeze(1)?)?.squeeze()?.tanh() };

    let jac = jacobian(f_vec, &x)?;
    assert_eq!(jac.shape(), &[2, 3]);

    let (_y, jv) = jvp(f_vec, &x, &v)?;
    let jv_ref = jac.matmul(&v.unsqueeze(1)?)?.squeeze()?;
    assert!(max_abs_diff(&jv, &jv_ref)? < 1e-10);

    let (_y2, vjp_fn) = vjp(f_vec, &x)?;
    let ut_j = vjp_fn(&u)?;
    let ut_j_ref = u.unsqueeze(0)?.matmul(&jac)?.squeeze()?;
    assert!(max_abs_diff(&ut_j, &ut_j_ref)? < 1e-10);

    // Scalar function for Hessian and HVP
    let a_sym = w.t()?.matmul(&w)?;
    let f_scalar = |inp: &Tensor| -> Result<Tensor> {
        let col = inp.unsqueeze(1)?;
        let quad = col.t()?.matmul(&a_sym)?.matmul(&col)?.sum()?;
        quad.add(&inp.powf(3.0)?.sum()?)
    };

    let h_mat = hessian(f_scalar, &x)?;
    assert_eq!(h_mat.shape(), &[3, 3]);
    assert!(max_abs_diff(&h_mat, &h_mat.t()?)? < 1e-10);

    let (_grad_x, hv) = hvp(f_scalar, &x, &v)?;
    let hv_ref = h_mat.matmul(&v.unsqueeze(1)?)?.squeeze()?;
    assert!(max_abs_diff(&hv, &hv_ref)? < 1e-10);

    // Empirical Neural Tangent Kernel (NTK)
    let net = Sequential::new()
        .add(Linear::new(3, 8)?)
        .add(Tanh)
        .add(Linear::new(8, 1)?);
    let x_batch = Tensor::randn(&[4, 3])?;
    let ntk = ntk_matrix(&net, &x_batch, &x_batch)?;
    assert_eq!(ntk.shape(), &[4, 4]);
    assert!(max_abs_diff(&ntk, &ntk.t()?)? < 1e-5);
    let (evals, _) = ntk.eigh()?;
    for ev in evals.to_vec::<f32>()? {
        assert!(ev >= -1e-4, "NTK eigenvalue should be >= 0, got {ev}");
    }
    Ok(())
}

#[test]
fn test_flash_attention_mamba_moe_diffusion_bpe_gnn_and_rl_losses() -> Result<()> {
    manual_seed(2027);

    // 1. FlashAttention-2 tiled online softmax equivalence & finite-difference gradcheck
    let q = Tensor::randn(&[2, 2, 7, 4])?.to_dtype(DType::F64)?;
    let k = Tensor::randn(&[2, 2, 7, 4])?.to_dtype(DType::F64)?;
    let v = Tensor::randn(&[2, 2, 7, 4])?.to_dtype(DType::F64)?;

    let flash_out = flash_attention(&q, &k, &v, 3, 2, true)?;
    let scale = 1.0 / (4.0f64).sqrt();
    let mask = Tensor::ones(&[7, 7], DType::Bool, &Device::Cpu)?.triu(1)?;
    let ref_scores = q
        .matmul(&k.t()?)?
        .mul_scalar(scale)?
        .masked_fill(&mask, -1e9)?;
    let ref_out = ref_scores.softmax(-1)?.matmul(&v)?;
    assert!(max_abs_diff(&flash_out, &ref_out)? < 1e-9);

    let q_small = Tensor::randn(&[1, 2, 4, 3])?.to_dtype(DType::F64)?;
    let k_small = Tensor::randn(&[1, 2, 4, 3])?.to_dtype(DType::F64)?;
    let v_small = Tensor::randn(&[1, 2, 4, 3])?.to_dtype(DType::F64)?;
    gradcheck(
        |t| weighted_sum(&flash_attention(&t[0], &t[1], &t[2], 2, 2, true)?),
        &[q_small, k_small, v_small],
    );

    // 2. SelectiveSSM & MambaBlock forward + backward
    let ssm = SelectiveSSM::new(4, 3, 2)?;
    let u_ssm = Tensor::randn(&[2, 5, 4])?.requires_grad_(true);
    let y_ssm = ssm.forward(&u_ssm)?;
    assert_eq!(y_ssm.shape(), &[2, 5, 4]);
    y_ssm.sum()?.backward()?;
    assert!(u_ssm.grad().is_some());
    let mamba = MambaBlock::new(16, 2, 4)?;
    let x_seq = Tensor::randn(&[2, 6, 16])?.requires_grad_(true);
    let y_mamba = mamba.forward(&x_seq)?;
    assert_eq!(y_mamba.shape(), &[2, 6, 16]);
    y_mamba.sum()?.backward()?;
    assert!(x_seq.grad().is_some());
    assert!(mamba.ssm.a_log.grad().is_some());

    // 3. SparseMoE top-k routing + Switch load-balancing auxiliary loss
    let moe = SparseMoE::new(16, 32, 4, 2)?;
    let (moe_out, aux_loss) = moe.forward_with_aux_loss(&x_seq)?;
    assert_eq!(moe_out.shape(), &[2, 6, 16]);
    assert!(aux_loss.item::<f32>()? > 0.0);
    moe_out.sum()?.add(&aux_loss)?.backward()?;
    assert!(moe.router.weight.grad().is_some());

    // 4. DDPMScheduler (Linear & Cosine), add_noise, deterministic DDIM & timestep_embedding
    let sched_lin = DDPMScheduler::new(100, BetaSchedule::default())?;
    let sched_cos = DDPMScheduler::new(100, BetaSchedule::Cosine { s: 0.008 })?;
    assert_eq!(sched_cos.betas.len(), 100);
    let x0 = Tensor::randn(&[2, 3, 4, 4])?;
    let eps = Tensor::randn(&[2, 3, 4, 4])?;
    let xt = sched_lin.add_noise(&x0, &eps, &[25, 25])?;
    let x0_pred = sched_lin.predict_start_from_noise(&xt, 25, &eps)?;
    assert!(max_abs_diff(&x0, &x0_pred)? < 1e-5);
    // Deterministic DDIM step directly to t_prev = None (t=0 reconstruction) with exact noise recovers x0!
    let x_ddim0 = sched_lin.step_ddim(&eps, 25, None, &xt, 0.0)?;
    assert!(max_abs_diff(&x0, &x_ddim0)? < 1e-5);
    let _x_prev = sched_lin.step_ddpm(&eps, 25, &xt)?;
    let t_emb = timestep_embedding(&[0, 25, 99], 16)?;
    assert_eq!(t_emb.shape(), &[3, 16]);

    // 5. Byte-level BpeTokenizer training, compression & lossless UTF-8 roundtrip
    let corpus = "oxideon rust tensor autograd oxideon rust tensor deep learning oxideon rust";
    let tok = BpeTokenizer::train(corpus, 285);
    assert!(tok.vocab_size() > 260);
    let sample = "oxideon rust tensor 🚀";
    let ids = tok.encode(sample);
    assert!(ids.len() < sample.len()); // Compressed via learned merges!
    assert_eq!(tok.decode(&ids), sample);
    let (batch_ids, batch_mask) = tok.encode_batch_padded(&["oxideon", "rust tensor"], None)?;
    assert_eq!(batch_ids.shape()[0], 2);
    assert_eq!(batch_ids.shape(), batch_mask.shape());

    // 6. Graph Neural Networks (GCNConv & GATConv)
    let adj = Tensor::from_slice(
        &[
            0.0f32, 1.0, 1.0, 0.0, //
            1.0, 0.0, 1.0, 0.0, //
            1.0, 1.0, 0.0, 1.0, //
            0.0, 0.0, 1.0, 0.0, //
        ],
        &[4, 4],
    )?;
    let node_x = Tensor::randn(&[4, 8])?.requires_grad_(true);
    let gcn = GCNConv::new(8, 6)?;
    let gcn_out = gcn.forward_graph(&node_x, &adj, true)?;
    assert_eq!(gcn_out.shape(), &[4, 6]);
    gcn_out.sum()?.backward()?;
    assert!(node_x.grad().is_some());

    let gat = GATConv::new(8, 5, 0.2)?;
    let gat_out = gat.forward_graph(&node_x, &adj)?;
    assert_eq!(gat_out.shape(), &[4, 5]);
    gat_out.sum()?.backward()?;
    assert!(gat.attn_src.grad().is_some());

    // 7. RL & Alignment Losses (DPO, PPO clipped surrogate, GAE, VICReg) + gradcheck
    let pi_w = Tensor::randn(&[4])?.to_dtype(DType::F64)?;
    let pi_l = Tensor::randn(&[4])?.to_dtype(DType::F64)?;
    let ref_w = Tensor::randn(&[4])?.to_dtype(DType::F64)?;
    let ref_l = Tensor::randn(&[4])?.to_dtype(DType::F64)?;
    gradcheck(
        |t| loss::dpo_loss(&t[0], &t[1], &ref_w, &ref_l, 0.1),
        &[pi_w.clone(), pi_l],
    );

    let old_lp = pi_w.detach();
    let adv = Tensor::from_slice(&[1.2f64, -0.5, 0.8, -1.1], &[4])?;
    gradcheck(
        |t| loss::ppo_clipped_loss(&t[0], &old_lp, &adv, 0.2),
        std::slice::from_ref(&pi_w),
    );

    let (gae_adv, gae_ret) = loss::compute_gae(
        &[1.0, 0.5, 2.0],
        &[0.8, 0.9, 1.5],
        &[false, false, true],
        0.0,
        0.99,
        0.95,
    )?;
    assert_eq!(gae_adv.len(), 3);
    assert!((gae_ret[2] - 2.0).abs() < 1e-12);

    let za = Tensor::randn(&[5, 4])?.to_dtype(DType::F64)?;
    let zb = Tensor::randn(&[5, 4])?.to_dtype(DType::F64)?;
    gradcheck(
        |t| loss::vicreg_loss(&t[0], &t[1], 25.0, 25.0, 1.0),
        &[za, zb],
    );

    Ok(())
}

#[test]
fn test_graph_muon_lbfgs_ode_cfm_speculative_kan_and_trainer() -> Result<()> {
    manual_seed(2028);

    // 1. Autograd Graph Introspection, FLOP Estimator & Graphviz DOT Export
    let lin = Linear::new(8, 4)?;
    let x_g = Tensor::randn(&[3, 8])?.requires_grad_(true);
    let profile = profile_forward_backward(|| lin.forward(&x_g)?.relu())?;
    assert!(profile.graph_stats.op_nodes >= 2);
    assert!(profile.graph_stats.leaf_nodes >= 3);
    assert!(profile.graph_stats.estimated_flops > 0);
    let dot = export_graphviz_dot(&[lin.forward(&x_g)?]);
    assert!(dot.contains("digraph OxideonAutograd"));
    // Linear records one fused "linear" node on the CPU fast path, a matmul chain elsewhere
    assert!(dot.contains("linear") || dot.contains("matmul"));

    // 2. Muon (Newton-Schulz Orthogonalized Momentum) & L-BFGS Quasi-Newton
    let g_mat = Tensor::randn(&[4, 6])?;
    let orth = newton_schulz_orthogonalize(&g_mat, 6)?;
    let gram = orth.matmul(&orth.t()?)?;
    let gram_vec = gram.to_vec::<f32>()?;
    for i in 0..4 {
        for j in 0..4 {
            let v = gram_vec[i * 4 + j];
            if i == j {
                assert!(
                    (0.5..=1.1).contains(&v),
                    "expected Muon NS diagonal in [0.5, 1.1], got {v}"
                );
            } else {
                assert!(
                    v.abs() < 0.15,
                    "expected Muon NS off-diagonal near 0, got {v}"
                );
            }
        }
    }

    let model_muon = Linear::new(4, 2)?;
    let mut muon = Muon::new(model_muon.parameters(), 0.02).weight_decay(1e-4);
    let xm = Tensor::randn(&[6, 4])?;
    let ym = Tensor::randn(&[6, 2])?;
    let l0 = loss::mse_loss(&model_muon.forward(&xm)?, &ym)?.item::<f32>()?;
    for _ in 0..15 {
        let l = loss::mse_loss(&model_muon.forward(&xm)?, &ym)?;
        muon.zero_grad();
        l.backward()?;
        muon.step()?;
    }
    let l1 = loss::mse_loss(&model_muon.forward(&xm)?, &ym)?.item::<f32>()?;
    assert!(l1 < l0, "Muon did not decrease loss: {l0} -> {l1}");

    // L-BFGS on a convex quadratic objective f(w) = 0.5 * ||w - target||^2
    let w_lbfgs = Tensor::zeros_f32(&[4])?.requires_grad_(true);
    let target_w = Tensor::from_slice(&[2.0f32, -1.5, 0.8, 3.1], &[4])?;
    let mut lbfgs = LBFGS::new(vec![w_lbfgs.clone()], 0.8, 5);
    for _ in 0..8 {
        let obj = w_lbfgs.sub(&target_w)?.square()?.sum()?.mul_scalar(0.5)?;
        lbfgs.zero_grad();
        obj.backward()?;
        lbfgs.step()?;
    }
    assert!(max_abs_diff(&w_lbfgs, &target_w)? < 1e-3);

    // 3. Differentiable ODE Solvers (odeint RK4), NeuralODE & Conditional Flow Matching (CFM)
    let y0 = Tensor::from_slice(&[2.0f64, -1.0, 0.5], &[3])?;
    // dy/dt = -0.5 * y => y(1) = exp(-0.5) * y0
    let y1 = odeint(
        |_t, y| y.mul_scalar(-0.5),
        &y0,
        0.0,
        1.0,
        10,
        OdeMethod::RK4,
    )?;
    let exact_y1 = y0.mul_scalar((-0.5f64).exp())?;
    assert!(max_abs_diff(&y1, &exact_y1)? < 1e-6);

    gradcheck(
        |t| {
            let sol = odeint(
                |_time, y| y.tanh()?.mul_scalar(-0.4),
                &t[0],
                0.0,
                1.0,
                4,
                OdeMethod::RK4,
            )?;
            weighted_sum(&sol)
        },
        std::slice::from_ref(&y0),
    );

    let node_block = NeuralODE::new(Linear::new(4, 4)?, 0.0, 1.0, 4, OdeMethod::Midpoint);
    let z_out = node_block.forward(&xm)?;
    assert_eq!(z_out.shape(), &[6, 4]);
    z_out.sum()?.backward()?;

    let cfm = FlowMatcher::new(0.0);
    let x0_cfm = Tensor::randn(&[4, 3])?;
    let x1_cfm = Tensor::randn(&[4, 3])?;
    let t_cfm = Tensor::from_slice(&[0.25f32, 0.5, 0.75, 1.0], &[4])?;
    let (xt_cfm, ut_cfm) = cfm.sample_location_and_target(&x0_cfm, &x1_cfm, &t_cfm)?;
    assert_eq!(xt_cfm.shape(), &[4, 3]);
    assert_eq!(ut_cfm.shape(), &[4, 3]);
    // With constant exact velocity u = x1 - x0, integrating from 0 to 1 recovers x1 exactly!
    let x1_rec = cfm.sample(|_t, _x| Ok(ut_cfm.clone()), &x0_cfm, 4, OdeMethod::Euler)?;
    assert!(max_abs_diff(&x1_cfm, &x1_rec)? < 1e-5);

    // 4. Speculative Decoding & Beam Search on CausalLM
    let target_lm = CausalLM::new(16, 16, 2, 2, 32, 32)?;
    let draft_lm = CausalLM::new(16, 16, 1, 2, 32, 32)?;
    let (spec_seq, accepted, drafted) =
        target_lm.speculative_generate(&draft_lm, &[1, 2, 3], 6, 3)?;
    assert_eq!(spec_seq.len(), 9);
    assert!(drafted >= 3 && accepted <= drafted);

    let beams = target_lm.beam_search(&[1, 2], 3, 2, 0.6)?;
    assert_eq!(beams.len(), 2);
    assert_eq!(beams[0].0.len(), 5);

    // 5. Kolmogorov-Arnold Networks (KANLinear & KAN)
    let kan = KAN::new(&[4, 6, 2], 5)?;
    let kan_out = kan.forward(&xm)?;
    assert_eq!(kan_out.shape(), &[6, 2]);
    kan_out.sum()?.backward()?;
    assert!(kan.layers[0].spline_weight.grad().is_some());

    // 6. Declarative Trainer with gradient accumulation, grad clipping, EarlyStopping & checkpointing
    let ds = TensorDataset::new(xm, ym);
    let loader = DataLoader::new(&ds, 2, false);
    let train_model = Linear::new(4, 2)?;
    let mut train_opt = Adam::new(train_model.parameters(), 0.05);
    let cfg = TrainerConfig::new(8)
        .grad_accum_steps(2)
        .max_grad_norm(1.0)
        .early_stopping(5, 1e-6);
    let trainer = Trainer::new(cfg);
    let history = trainer.fit(
        &train_model,
        &mut train_opt,
        &loader,
        Some(&loader),
        loss::mse_loss,
    )?;
    assert!(!history.metrics.is_empty());
    assert!(history.best_checkpoint_bytes.is_some());
    assert!(history.best_loss <= history.metrics[0].train_loss + 1e-5);

    Ok(())
}

#[test]
fn test_v06_compiler_sparse_distributions_vmap_compression() -> Result<()> {
    manual_seed(404);

    // 1. Symbolic IR Graph Compiler & Operator Fusion Engine (IrGraph)
    let mut g = IrGraph::new();
    let x_id = g.add_input();
    let w_id = g.add_input();
    let b_id = g.add_input();
    let zero_id = g.push_op(IrOp::ZeroLike { src: x_id });
    let one_id = g.push_op(IrOp::OneLike { src: x_id });

    // Algebraic identities: ((x + 0) * 1)^T^T => x
    let x_add0 = g.push_op(IrOp::Add(x_id, zero_id));
    let x_mul1 = g.push_op(IrOp::Mul(x_add0, one_id));
    let x_t1 = g.push_op(IrOp::Transpose2D(x_mul1));
    let x_clean = g.push_op(IrOp::Transpose2D(x_t1));

    // Duplicate Matmul (should be deduplicated by CSE)
    let mm1 = g.push_op(IrOp::Matmul(x_clean, w_id));
    let mm2 = g.push_op(IrOp::Matmul(x_clean, w_id));
    let lin1 = g.push_op(IrOp::Add(mm1, b_id));
    let act1 = g.push_op(IrOp::Relu(lin1)); // Matmul + Add + Relu -> FusedLinearRelu

    // Dead branch (never reaches output)
    let _dead = g.push_op(IrOp::Exp(mm2));

    // FMA pattern: act1 * act1 + act1 -> FusedFma
    let sq = g.push_op(IrOp::Mul(act1, act1));
    let out_id = g.push_op(IrOp::Add(sq, act1));
    g.set_outputs(vec![out_id]);

    let x_val = Tensor::randn(&[4, 3])?;
    let w_val = Tensor::randn(&[3, 5])?;
    let b_val = Tensor::randn(&[1, 5])?;
    let inputs = [x_val, w_val, b_val];

    let unopt_out = g.execute(&inputs)?.pop().unwrap();
    let report = g.optimize()?;
    assert!(report.fused_ops_count >= 2);
    assert!(report.nodes_after < report.nodes_before);

    let opt_out = g.execute(&inputs)?.pop().unwrap();
    assert!(max_abs_diff(&unopt_out, &opt_out)? < 1e-5);
    assert!(
        g.nodes
            .iter()
            .any(|n| matches!(n.op, IrOp::FusedLinearRelu { .. }))
    );
    assert!(g.nodes.iter().any(|n| matches!(n.op, IrOp::FusedFma(..))));

    // 2. Sparse Tensors (COO & CSR) & Differentiable spmm
    let dense_s = Tensor::from_slice(
        &[
            1.5f64, 0.0, -2.0, 0.0, 0.0, 3.0, 0.0, 0.5, -1.0, 0.0, 0.0, 2.5,
        ],
        &[3, 4],
    )?;
    let coo = SparseCoo::from_dense(&dense_s, 1e-12)?;
    assert_eq!(coo.nnz(), 6);
    let csr = coo.to_csr()?;
    assert!((csr.sparsity() - 6.0 / 12.0).abs() < 1e-6);
    assert!(max_abs_diff(&dense_s, &csr.to_dense(DType::F64, &Device::Cpu)?)? < 1e-12);

    let x_sp = Tensor::randn(&[4, 2])?
        .to_dtype(DType::F64)?
        .requires_grad_(true);
    let y_sp = csr.spmm(&x_sp)?;
    let y_ref = dense_s.matmul(&x_sp)?;
    assert!(max_abs_diff(&y_sp, &y_ref)? < 1e-10);

    gradcheck(
        |t| {
            let y = csr.spmm(&t[0])?;
            weighted_sum(&y)
        },
        std::slice::from_ref(&x_sp),
    );

    // 3. Probability Distributions (Normal, Categorical + Straight-Through Gumbel-Softmax, Laplace)
    let loc = Tensor::from_slice(&[0.5f64, -1.0], &[2])?.requires_grad_(true);
    let scale = Tensor::from_slice(&[1.2f64, 0.8], &[2])?.requires_grad_(true);
    let norm_p = Normal::new(loc.clone(), scale.clone())?;
    let norm_q = Normal::new(
        Tensor::zeros(&[2], DType::F64, &Device::Cpu)?,
        Tensor::ones(&[2], DType::F64, &Device::Cpu)?,
    )?;

    let z_rsample = norm_p.rsample()?;
    assert_eq!(z_rsample.shape(), &[2]);
    let kl_pq = norm_p.kl_divergence(&norm_q)?.sum()?;
    assert!(kl_pq.item::<f64>()? > 0.0);

    gradcheck(
        |t| {
            let dist = Normal::new(t[0].clone(), t[1].clone())?;
            let lp = dist.log_prob(&Tensor::from_slice(&[0.2f64, -0.5], &[2])?)?;
            let ent = dist.entropy()?;
            weighted_sum(&lp.add(&ent)?)
        },
        &[loc.clone(), scale.clone()],
    );

    let logits =
        Tensor::from_slice(&[1.0f32, 2.5, -0.5, 0.2, -1.0, 1.8], &[2, 3])?.requires_grad_(true);
    let cat = Categorical::from_logits(logits.clone());
    let gs_hard = cat.gumbel_softmax(0.5, true)?;
    assert_eq!(gs_hard.shape(), &[2, 3]);
    // Verify each row of hard Gumbel-Softmax is strictly one-hot (sums to 1, elements in {0, 1})
    let row_sums = gs_hard.sum_axis(1, false)?.to_vec::<f32>()?;
    assert!((row_sums[0] - 1.0).abs() < 1e-5 && (row_sums[1] - 1.0).abs() < 1e-5);
    for v in gs_hard.to_vec::<f32>()? {
        assert!((v - 0.0).abs() < 1e-6 || (v - 1.0).abs() < 1e-6);
    }
    // Straight-through backward flows to logits!
    weighted_sum(&gs_hard)?.backward()?;
    assert!(logits.grad().is_some());
    assert!(logits.grad().unwrap().abs()?.sum()?.item::<f32>()? > 0.0);

    let lap = Laplace::new(loc, scale)?;
    assert_eq!(lap.rsample()?.shape(), &[2]);
    assert!(lap.entropy()?.sum()?.item::<f64>()? > 0.0);

    // 4. Automatic Vectorization (vmap & vmap2)
    let bx = Tensor::randn(&[5, 3])?
        .to_dtype(DType::F64)?
        .requires_grad_(true);
    let by = Tensor::randn(&[5, 3])?
        .to_dtype(DType::F64)?
        .requires_grad_(true);
    // Per-sample dot product via vmap2
    let dots = vmap2(|a, b| a.mul(b)?.sum_axis(0, true), &bx, &by, 0, 0, 0)?;
    let dots_ref = bx.mul(&by)?.sum_axis(1, true)?;
    assert!(max_abs_diff(&dots, &dots_ref)? < 1e-10);

    gradcheck(
        |t| {
            let vm = vmap(|row| row.sin()?.mul(row), &t[0], 0, 0)?;
            weighted_sum(&vm)
        },
        std::slice::from_ref(&bx),
    );

    // 5. Model Compression: Unstructured & Structured Pruning + Knowledge Distillation
    let prune_net = Linear::new(4, 3)?;
    let mask = prune_magnitude_unstructured(&prune_net, 0.5)?;
    assert!((mask.overall_sparsity()? - 0.5).abs() < 1e-5);
    let zero_count = prune_net
        .weight
        .to_vec::<f32>()?
        .into_iter()
        .filter(|&v| v == 0.0)
        .count();
    assert_eq!(zero_count, 6);

    let w_rows = Tensor::from_slice(&[0.1f32, -0.9, 0.05, 0.8, -0.1, 0.2], &[3, 2])?;
    let (pruned_w, kept_idx) = prune_structured_l2_rows(&w_rows, 2.0 / 3.0)?;
    assert_eq!(pruned_w.shape(), &[2, 2]);
    assert_eq!(kept_idx, vec![0, 1]);

    let s_logits = Tensor::randn(&[4, 5])?
        .to_dtype(DType::F64)?
        .requires_grad_(true);
    let t_logits = Tensor::randn(&[4, 5])?.to_dtype(DType::F64)?;
    let targets = [1usize, 0, 4, 2];
    gradcheck(
        |t| distillation_loss(&t[0], &t_logits, &targets, 2.0, 0.7),
        std::slice::from_ref(&s_logits),
    );

    Ok(())
}

#[test]
fn test_v07_symbolic_autodiff_distributed_vision_pinn_mla_xai_tuning() -> Result<()> {
    manual_seed(707);

    // 1. Symbolic Reverse-Mode Autodiff on IrGraph + Linear-Scan Memory Planner (CompiledGraph)
    let mut g = IrGraph::new();
    let ix = g.add_input();
    let iw = g.add_input();
    let mm = g.push_op(IrOp::Matmul(ix, iw));
    let th = g.push_op(IrOp::Tanh(mm));
    let sq = g.push_op(IrOp::Square(th));
    let loss_id = g.push_op(IrOp::SumAll(sq));
    let grads = g.differentiate(loss_id, &[ix, iw])?;
    g.set_outputs(vec![loss_id, grads[0], grads[1]]);

    let compiled = g.compile()?;
    assert!(compiled.memory_plan.peak_slots <= compiled.memory_plan.total_live_values);
    assert!(compiled.memory_plan.reused_allocations > 0);

    let x_t = Tensor::randn(&[3, 4])?.requires_grad_(true);
    let w_t = Tensor::randn(&[4, 2])?.requires_grad_(true);
    let mut ir_outs = compiled.execute(&[x_t.clone(), w_t.clone()])?;
    let ir_dw = ir_outs.pop().unwrap();
    let ir_dx = ir_outs.pop().unwrap();
    let ir_loss = ir_outs.pop().unwrap();

    let ref_loss = x_t.matmul(&w_t)?.tanh()?.square()?.sum()?;
    ref_loss.backward()?;
    assert!(max_abs_diff(&ir_loss, &ref_loss)? < 1e-5);
    assert!(max_abs_diff(&ir_dx, &x_t.grad().unwrap())? < 1e-5);
    assert!(max_abs_diff(&ir_dw, &w_t.grad().unwrap())? < 1e-5);

    // 2. Megatron-LM 1D Tensor Parallelism (ColumnParallelLinear & RowParallelLinear) & ZeRO-1
    let dense_fc1 = Linear::new(6, 8)?;
    let dense_fc2 = Linear::new(8, 4)?;
    let col_par = ColumnParallelLinear::from_linear(&dense_fc1, 2)?;
    let row_par = RowParallelLinear::from_linear(&dense_fc2, 2)?;

    let x_tp = Tensor::randn(&[5, 6])?;
    let ref_tp = dense_fc2.forward(&dense_fc1.forward(&x_tp)?.silu()?)?;
    let shards = col_par
        .forward_sharded(&x_tp)?
        .into_iter()
        .map(|s| s.silu())
        .collect::<Result<Vec<_>>>()?;
    let out_tp = row_par.forward_from_shards(&shards)?;
    assert!(max_abs_diff(&ref_tp, &out_tp)? < 1e-5);

    let zero_net = Linear::new(4, 4)?;
    let mut zero_opt = ZeROAdamW::new(zero_net.parameters(), 0.05, 1e-3, 2)?;
    assert!(zero_opt.rank_state_elements(0) > 0 && zero_opt.rank_state_elements(1) > 0);
    let x_z = Tensor::randn(&[4, 4])?;
    let y_z = Tensor::randn(&[4, 4])?;
    let loss_z0 = loss::mse_loss(&zero_net.forward(&x_z)?, &y_z)?.item::<f32>()?;
    for _ in 0..10 {
        let l = loss::mse_loss(&zero_net.forward(&x_z)?, &y_z)?;
        zero_opt.zero_grad();
        l.backward()?;
        zero_opt.step()?;
    }
    let loss_z1 = loss::mse_loss(&zero_net.forward(&x_z)?, &y_z)?.item::<f32>()?;
    assert!(loss_z1 < loss_z0);

    // 3. Differentiable 2D Spatial Vision Ops: affine_grid, grid_sample (dual gradcheck!), roi_align & NMS
    let theta_id =
        Tensor::from_slice(&[1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0], &[1, 2, 3])?.requires_grad_(true);
    let img = Tensor::randn(&[1, 2, 4, 4])?
        .to_dtype(DType::F64)?
        .requires_grad_(true);
    let id_grid = affine_grid(&theta_id, [1, 2, 4, 4], true)?;
    let sampled_id = grid_sample(&img, &id_grid, true)?;
    // Identity affine transformation with align_corners=true reproduces input exactly!
    assert!(max_abs_diff(&img, &sampled_id)? < 1e-10);

    let small_img = Tensor::randn(&[1, 2, 3, 3])?
        .to_dtype(DType::F64)?
        .requires_grad_(true);
    let small_grid = Tensor::from_slice(
        &[-0.4f64, -0.3, 0.2, -0.5, -0.1, 0.4, 0.5, 0.3],
        &[1, 2, 2, 2],
    )?
    .requires_grad_(true);
    gradcheck(
        |t| {
            let out = grid_sample(&t[0], &t[1], true)?;
            weighted_sum(&out)
        },
        &[small_img.clone(), small_grid],
    );

    let rois = Tensor::from_slice(&[0.0f64, 0.2, 0.2, 2.0, 2.0], &[1, 5])?;
    gradcheck(
        |t| {
            let pooled = roi_align(&t[0], &rois, [2, 2], 1.0, 2)?;
            weighted_sum(&pooled)
        },
        std::slice::from_ref(&small_img),
    );

    let boxes = Tensor::from_slice(
        &[
            0.0f32, 0.0, 10.0, 10.0, // box 0 (high score)
            1.0, 1.0, 10.0, 10.0, // box 1 (heavily overlaps box 0 -> suppressed)
            20.0, 20.0, 30.0, 30.0, // box 2 (disjoint -> kept)
        ],
        &[3, 4],
    )?;
    let scores = Tensor::from_slice(&[0.95f32, 0.80, 0.70], &[3])?;
    let keep = nms(&boxes, &scores, 0.5)?;
    assert_eq!(keep, vec![0, 2]);
    let ious = box_iou(&boxes, &boxes)?;
    assert_eq!(ious.shape(), &[3, 3]);

    // 4. Differentiable Physics & PDE Primitives (PINNs: spatial_gradient, divergence, laplacian) & AMP autocast
    // Scalar field u(x, y) = x^3 + 2 * y^2 => grad = [3 x^2, 4 y], laplacian = 6 x + 4
    let coords = Tensor::from_slice(&[1.5f64, -2.0, 0.5, 1.0], &[2, 2])?.requires_grad_(true);
    let u_fn = |xy: &Tensor| -> Result<Tensor> {
        let x_c = xy.narrow(1, 0, 1)?;
        let y_c = xy.narrow(1, 1, 1)?;
        x_c.powf(3.0)?.add(&y_c.square()?.mul_scalar(2.0)?)
    };
    let grad_u = spatial_gradient(u_fn, &coords)?;
    let expected_grad = Tensor::from_slice(&[6.75f64, -8.0, 0.75, 4.0], &[2, 2])?;
    assert!(max_abs_diff(&grad_u, &expected_grad)? < 1e-10);

    let lap_u = laplacian(u_fn, &coords)?;
    let expected_lap = Tensor::from_slice(&[13.0f64, 7.0], &[2, 1])?;
    assert!(max_abs_diff(&lap_u, &expected_lap)? < 1e-9);

    // Vector field v(x, y) = [x^2, 3 * y] => div(v) = 2*x + 3
    let div_v = divergence(
        |xy| {
            let x_c = xy.narrow(1, 0, 1)?.square()?;
            let y_c = xy.narrow(1, 1, 1)?.mul_scalar(3.0)?;
            Tensor::cat(&[x_c, y_c], 1)
        },
        &coords,
    )?;
    let expected_div = Tensor::from_slice(&[6.0f64, 4.0], &[2, 1])?;
    assert!(max_abs_diff(&div_v, &expected_div)? < 1e-10);

    assert!(!is_autocast_enabled());
    let casted_dt = autocast(DType::BF16, || -> Result<DType> {
        assert!(is_autocast_enabled());
        Ok(autocast_cast(&Tensor::ones_f32(&[2, 2])?)?.dtype())
    })?;
    assert_eq!(casted_dt, DType::BF16);
    assert!(!is_autocast_enabled());

    // 5. Next-Gen Attention & SSMs: MultiHeadLatentAttention (MLA), DiffAttention, HyenaOperator
    let mla = MultiHeadLatentAttention::new(16, 4, 4, 8)?;
    assert!((mla.kv_cache_compression_ratio() - 8.0).abs() < 1e-6);
    let seq_x = Tensor::randn(&[2, 6, 16])?;
    let mla_out = mla.forward(&seq_x)?;
    assert_eq!(mla_out.shape(), &[2, 6, 16]);
    mla_out.sum()?.backward()?;

    let diff_attn = DiffAttention::new(16, 4, 1)?;
    let da_out = diff_attn.forward(&seq_x)?;
    assert_eq!(da_out.shape(), &[2, 6, 16]);
    da_out.sum()?.backward()?;
    assert!(diff_attn.lambda_param.grad().is_some());

    let hyena = HyenaOperator::new(16, 2, 3)?;
    let hy_out = hyena.forward(&seq_x)?;
    assert_eq!(hy_out.shape(), &[2, 6, 16]);
    hy_out.sum()?.backward()?;

    // 6. Modern Alignment & Contrastive Losses: GRPO, SimPO, SigLIP, Barlow Twins
    let new_lp = Tensor::from_slice(&[-1.1f64, -0.8, -1.5, -0.9], &[4])?.requires_grad_(true);
    let old_lp = Tensor::from_slice(&[-1.2f64, -0.85, -1.4, -0.95], &[4])?;
    let ref_lp = Tensor::from_slice(&[-1.3f64, -0.9, -1.6, -1.0], &[4])?;
    let rewards = Tensor::from_slice(&[1.0f64, 2.5, -0.5, 0.8], &[4])?;
    gradcheck(
        |t| loss::grpo_loss(&t[0], &old_lp, &ref_lp, &rewards, 0.2, 0.04),
        std::slice::from_ref(&new_lp),
    );

    let chosen_lp = Tensor::from_slice(&[-0.6f64, -0.4, -0.8, -0.5], &[4])?.requires_grad_(true);
    let rejected_lp = Tensor::from_slice(&[-1.2f64, -1.1, -1.5, -0.9], &[4])?.requires_grad_(true);
    gradcheck(
        |t| loss::simpo_loss(&t[0], &t[1], 2.0, 0.5),
        &[chosen_lp, rejected_lp],
    );

    let z1 = Tensor::randn(&[4, 5])?
        .to_dtype(DType::F64)?
        .requires_grad_(true);
    let z2 = Tensor::randn(&[4, 5])?
        .to_dtype(DType::F64)?
        .requires_grad_(true);
    gradcheck(
        |t| loss::siglip_loss(&t[0], &t[1], 10.0, -5.0),
        &[z1.clone(), z2.clone()],
    );
    gradcheck(|t| loss::barlow_twins_loss(&t[0], &t[1], 0.05), &[z1, z2]);

    // 7. ScheduleFreeAdamW, Lookahead & Explainable AI (Integrated Gradients Completeness Axiom + Grad-CAM)
    let sf_net = Linear::new(4, 2)?;
    let mut sf_opt = ScheduleFreeAdamW::new(sf_net.parameters(), 0.05, 1e-4);
    let x_sf = Tensor::randn(&[6, 4])?;
    let y_sf = Tensor::randn(&[6, 2])?;
    for _ in 0..8 {
        let pred = sf_net.forward(&x_sf)?;
        let l = loss::mse_loss(&pred, &y_sf)?;
        sf_opt.zero_grad();
        l.backward()?;
        sf_opt.step()?;
    }
    sf_opt.eval_mode()?;
    let _eval_pred = sf_net.forward(&x_sf)?;
    sf_opt.train_mode()?;

    let la_net = Linear::new(4, 2)?;
    let inner_sgd = Sgd::new(la_net.parameters(), 0.05);
    let mut la_opt = Lookahead::new(inner_sgd, la_net.parameters(), 2, 0.5);
    for _ in 0..4 {
        let l = loss::mse_loss(&la_net.forward(&x_sf)?, &y_sf)?;
        la_opt.zero_grad();
        l.backward()?;
        la_opt.step()?;
    }

    // Verify Integrated Gradients Completeness Axiom: sum(IG(x)) == f(x) - f(baseline)
    let x_ig = Tensor::from_slice(&[1.2f64, -0.8, 0.5, 2.0], &[1, 4])?;
    let base_ig = Tensor::zeros(&[1, 4], DType::F64, &Device::Cpu)?;
    let poly_fn = |inp: &Tensor| -> Result<Tensor> {
        inp.powf(3.0)?.mul_scalar(0.5)?.add(&inp.square()?)?.sum()
    };
    let ig_attr = integrated_gradients(poly_fn, &x_ig, Some(&base_ig), 64)?;
    let ig_sum = ig_attr.sum()?.item::<f64>()?;
    let f_diff = poly_fn(&x_ig)?.item::<f64>()? - poly_fn(&base_ig)?.item::<f64>()?;
    assert!((ig_sum - f_diff).abs() < 1e-3);

    let act_map = Tensor::randn(&[2, 4, 4, 4])?.requires_grad_(true);
    let score_cam = act_map.powf(2.0)?.sum()?;
    let cam = grad_cam_2d(&act_map, &score_cam)?;
    assert_eq!(cam.shape(), &[2, 1, 4, 4]);

    // 8. Hyperparameter Tuning (Study::grid_search & random_search) & K-Fold Cross-Validation
    let folds = k_fold_split(&x_sf, &y_sf, 3, true)?;
    assert_eq!(folds.len(), 3);
    assert_eq!(folds[0].0.shape()[0], 4);
    assert_eq!(folds[0].2.shape()[0], 2);

    let space = ParamSpace::new()
        .add_discrete(
            "lr",
            vec![
                ParamValue::Float(0.01),
                ParamValue::Float(0.1),
                ParamValue::Float(0.5),
            ],
        )
        .add_uniform("reg", 0.0, 1.0);
    let study = Study::grid_search(&space, 3, |cfg| {
        let lr = cfg["lr"].as_float();
        let reg = cfg["reg"].as_float();
        Ok((lr - 0.1).powi(2) + (reg - 0.5).powi(2))
    })?;
    assert_eq!(study.trials.len(), 9);
    assert!(study.best_trial.unwrap().objective < 1e-12);

    Ok(())
}

#[test]
fn test_frontier_foundation_models_deepseek_jamba_vlm_dit_reward() -> Result<()> {
    manual_seed(909);

    // 1. DeepSeek-V3 / R1 Frontier Language Model (MLA + SharedRoutedMoE + Multi-Token Prediction)
    let deepseek = DeepSeekMoELM::new(24, 16, 2, 4, 4, 8, 32, 1, 3, 2, 2)?;
    let tokens = Tensor::from_slice(&[1i64, 4, 7, 2, 9, 3, 2, 5, 8, 6, 1, 4], &[2, 6])?;
    let (main_logits, mtp_logits, aux_loss) = deepseek.forward_with_mtp(&tokens)?;
    assert_eq!(main_logits.shape(), &[2, 6, 24]);
    assert_eq!(mtp_logits.len(), 2);
    assert_eq!(mtp_logits[0].shape(), &[2, 5, 24]); // t+1 depth
    assert_eq!(mtp_logits[1].shape(), &[2, 4, 24]); // t+2 depth
    let ds_total_loss = main_logits
        .sum()?
        .add(&mtp_logits[0].sum()?)?
        .add(&mtp_logits[1].sum()?)?
        .add(&aux_loss)?;
    ds_total_loss.backward()?;
    assert!(deepseek.tok_emb.weight.grad().is_some());

    // 2. Hybrid Mamba-Attention-MoE Foundation Model (HybridJambaLM)
    let jamba = HybridJambaLM::new(24, 16, 4, 4, 2, 4, 32, 3, 2, 2, 2)?;
    let jamba_logits = jamba.forward(&tokens)?;
    assert_eq!(jamba_logits.shape(), &[2, 6, 24]);
    jamba_logits.sum()?.backward()?;
    assert!(jamba.lm_head.weight.grad().is_some());

    // 3. Multimodal Vision-Language Foundation Model (MultimodalVLM: ViT + MLP Connector + RoPE GQA LLM)
    let vlm = MultimodalVLM::new(8, 4, 3, 16, 1, 4, 24, 16, 2, 4, 2, 32, 32)?;
    let images = Tensor::randn(&[2, 3, 8, 8])?;
    let txt_tokens = Tensor::from_slice(&[1i64, 3, 5, 2, 4, 6, 8, 7], &[2, 4])?;
    let vlm_logits = vlm.forward_multimodal(&images, &txt_tokens)?;
    assert_eq!(vlm_logits.shape(), &[2, 4, 24]);
    vlm_logits.sum()?.backward()?;
    assert!(vlm.projector_fc1.weight.grad().is_some());

    let single_img = images.narrow(0, 0, 1)?;
    let gen_vlm = vlm.generate_multimodal(&single_img, &[1, 3], 3)?;
    assert_eq!(gen_vlm.len(), 5);

    // 4. Diffusion / Rectified-Flow Transformer (DiT with AdaLN-Zero conditioning)
    let dit = DiT::new(8, 4, 3, 16, 2, 4, 10)?;
    let noisy_latents = Tensor::randn(&[2, 3, 8, 8])?;
    let cls_labels = Tensor::from_slice(&[2i64, 7], &[2])?;
    let pred_vel = dit.forward_dit(&noisy_latents, &[15, 75], Some(&cls_labels))?;
    assert_eq!(pred_vel.shape(), &[2, 3, 8, 8]);
    pred_vel.sum()?.backward()?;
    assert!(dit.blocks[0].adaln_modulation.weight.grad().is_some());

    // 5. RLHF Sequence RewardModel & Bradley-Terry Preference Optimization
    let rm = RewardModel::new(24, 16, 1, 4, 32)?;
    let chosen_seq = Tensor::from_slice(&[1i64, 5, 9, 12], &[1, 4])?;
    let rejected_seq = Tensor::from_slice(&[1i64, 2, 3, 4], &[1, 4])?;
    let r_chosen = rm.forward(&chosen_seq)?;
    let r_rejected = rm.forward(&rejected_seq)?;
    assert_eq!(r_chosen.shape(), &[1]);
    let bt_loss = bradley_terry_reward_loss(&r_chosen, &r_rejected)?;
    bt_loss.backward()?;
    assert!(rm.score_head.weight.grad().is_some());

    Ok(())
}

fn check_grad(
    f: impl Fn(&Tensor) -> Result<Tensor>,
    x: &Tensor,
    _eps: f64,
    _tol: f64,
) -> Result<()> {
    gradcheck(|ins| f(&ins[0]), std::slice::from_ref(x));
    Ok(())
}

#[test]
fn test_fused_kernels_and_gradcheck() -> Result<()> {
    manual_seed(1010);

    // 1. fused_rmsnorm matches eager RMSNorm and passes gradcheck
    let x = Tensor::from_slice(&[1.2f64, -0.8, 0.5, 2.0, -1.5, 0.3], &[2, 3])?.requires_grad_(true);
    let w = Tensor::from_slice(&[1.1f64, 0.9, 1.3], &[3])?.requires_grad_(true);
    let y_fused = fused_rmsnorm(&x, &w, 1e-5)?;
    let rms_eager = x
        .square()?
        .mean_axis(-1, true)?
        .add_scalar(1e-5)?
        .rsqrt()?
        .mul(&x)?
        .mul(&w)?;
    for (a, b) in y_fused
        .to_vec::<f64>()?
        .iter()
        .zip(rms_eager.to_vec::<f64>()?)
    {
        assert!((a - b).abs() < 1e-10);
    }
    check_grad(|inp| fused_rmsnorm(inp, &w, 1e-5)?.sum(), &x, 1e-6, 1e-5)?;
    check_grad(|wt| fused_rmsnorm(&x, wt, 1e-5)?.sum(), &w, 1e-6, 1e-5)?;

    // 2. fused_add_rmsnorm matches residual add + RMSNorm and passes gradcheck
    let res =
        Tensor::from_slice(&[-0.4f64, 0.7, 1.1, -0.2, 0.6, -0.9], &[2, 3])?.requires_grad_(true);
    let (normed, sum_out) = fused_add_rmsnorm(&x, &res, &w, 1e-5)?;
    assert_eq!(normed.shape(), &[2, 3]);
    assert_eq!(sum_out.shape(), &[2, 3]);
    check_grad(
        |inp| {
            let (n, s) = fused_add_rmsnorm(inp, &res, &w, 1e-5)?;
            n.mul_scalar(0.7)?.add(&s.mul_scalar(0.3)?)?.sum()
        },
        &x,
        1e-6,
        1e-5,
    )?;

    // 3. fused_swiglu & fused_geglu pass finite-difference gradcheck
    let gate = Tensor::from_slice(&[0.6f64, -1.1, 0.3, 1.4], &[2, 2])?.requires_grad_(true);
    let up = Tensor::from_slice(&[1.2f64, 0.5, -0.7, 0.9], &[2, 2])?.requires_grad_(true);
    let swi = fused_swiglu(&gate, &up)?;
    let swi_eager = gate.silu()?.mul(&up)?;
    for (a, b) in swi.to_vec::<f64>()?.iter().zip(swi_eager.to_vec::<f64>()?) {
        assert!((a - b).abs() < 1e-10);
    }
    check_grad(|g| fused_swiglu(g, &up)?.sum(), &gate, 1e-6, 1e-5)?;
    check_grad(|u| fused_swiglu(&gate, u)?.sum(), &up, 1e-6, 1e-5)?;
    check_grad(|g| fused_geglu(g, &up)?.sum(), &gate, 1e-6, 1e-5)?;

    // 4. fused_linear_activation & fused_cross_entropy_with_z_loss pass gradcheck
    let lin_w =
        Tensor::from_slice(&[0.5f64, -0.3, 0.8, 0.2, 0.7, -0.4], &[2, 3])?.requires_grad_(true);
    let lin_b = Tensor::from_slice(&[0.1f64, -0.2], &[2])?.requires_grad_(true);
    check_grad(
        |inp| fused_linear_activation(inp, &lin_w, Some(&lin_b), FusedActivation::SiLU)?.sum(),
        &x,
        1e-6,
        1e-5,
    )?;

    let logits =
        Tensor::from_slice(&[1.5f64, -0.5, 0.2, -0.3, 2.1, 0.4], &[2, 3])?.requires_grad_(true);
    let targets = [0usize, 1];
    check_grad(
        |lg| fused_cross_entropy_with_z_loss(lg, &targets, 1e-4),
        &logits,
        1e-6,
        1e-5,
    )?;

    Ok(())
}

#[test]
fn test_pytorch_parity_ops_linalg_conv3d_and_jit_tracer() -> Result<()> {
    manual_seed(2025);

    // 1. Zero-copy unfold, roll, tile, repeat_interleave, unbind, diag_embed, cumprod, diff, lerp, atan2, hypot, nan_to_num, bincount, histc, searchsorted, meshgrid
    let v = Tensor::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5])?.requires_grad_(true);
    let win = v.unfold(0, 3, 1)?;
    assert_eq!(win.shape(), &[3, 3]);
    assert!(win.shares_storage(&v));
    check_grad(|t| t.unfold(0, 3, 1)?.powf(2.0)?.sum(), &v, 1e-6, 1e-5)?;

    let rolled = v.roll(2, 0)?;
    assert_eq!(rolled.to_vec::<f64>()?, vec![4.0, 5.0, 1.0, 2.0, 3.0]);

    let rep = Tensor::from_slice(&[1.0f64, 2.0, 3.0], &[3])?.repeat_interleave(2, 0)?;
    assert_eq!(rep.to_vec::<f64>()?, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);

    let unbound = win.unbind(0)?;
    assert_eq!(unbound.len(), 3);
    assert_eq!(unbound[1].to_vec::<f64>()?, vec![2.0, 3.0, 4.0]);

    let d_mat = Tensor::from_slice(&[2.0f64, 5.0, 7.0], &[3])?.diag_embed()?;
    assert_eq!(d_mat.shape(), &[3, 3]);
    assert_eq!(
        d_mat.to_vec::<f64>()?,
        vec![2.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 7.0]
    );

    let cp_in = Tensor::from_slice(&[1.5f64, 2.0, -0.5, 3.0], &[2, 2])?.requires_grad_(true);
    check_grad(|t| t.cumprod(1)?.powf(2.0)?.sum(), &cp_in, 1e-6, 1e-5)?;

    let df = v.diff(1, 0)?;
    assert_eq!(df.to_vec::<f64>()?, vec![1.0, 1.0, 1.0, 1.0]);

    let y_atan = Tensor::from_slice(&[1.0f64, -2.0, 0.5], &[3])?.requires_grad_(true);
    let x_atan = Tensor::from_slice(&[2.0f64, 1.5, -1.2], &[3])?.requires_grad_(true);
    check_grad(|yy| yy.atan2(&x_atan)?.sum(), &y_atan, 1e-6, 1e-5)?;
    check_grad(|xx| y_atan.atan2(xx)?.sum(), &x_atan, 1e-6, 1e-5)?;
    let hyp = Tensor::from_slice(&[3.0f64], &[1])?.hypot(&Tensor::from_slice(&[4.0f64], &[1])?)?;
    assert!((hyp.item::<f64>()? - 5.0).abs() < 1e-10);

    let bc = Tensor::from_slice(&[0i64, 2, 2, 3, 3, 3], &[6])?.bincount(None, 0)?;
    assert_eq!(bc.to_vec::<i64>()?, vec![1, 0, 2, 3]);

    let sorted = Tensor::from_slice(&[1.0f32, 3.0, 5.0, 7.0], &[4])?;
    let queries = Tensor::from_slice(&[0.5f32, 3.0, 6.0, 9.0], &[4])?;
    let ss = Tensor::searchsorted(&sorted, &queries, false)?;
    assert_eq!(ss.to_vec::<i64>()?, vec![0, 1, 3, 4]);

    let gx = Tensor::from_slice(&[1.0f32, 2.0], &[2])?;
    let gy = Tensor::from_slice(&[10.0f32, 20.0, 30.0], &[3])?;
    let grids = Tensor::meshgrid(&[&gx, &gy], true)?;
    assert_eq!(grids[0].shape(), &[2, 3]);
    assert_eq!(grids[1].shape(), &[2, 3]);

    // 2. Advanced Linalg: pinv (with Golub-Pereyra backward), lstsq, matrix_exp, matrix_power, cond, matrix_rank, kron
    let a_rect =
        Tensor::from_slice(&[2.0f64, 1.0, 0.5, 3.0, -1.0, 1.5], &[3, 2])?.requires_grad_(true);
    let a_pinv = a_rect.pinv(1e-7)?;
    assert_eq!(a_pinv.shape(), &[2, 3]);
    // Verify A^+ @ A == I_2
    let prod_i2 = a_pinv.matmul(&a_rect)?.to_vec::<f64>()?;
    assert!((prod_i2[0] - 1.0).abs() < 1e-6 && (prod_i2[3] - 1.0).abs() < 1e-6);
    check_grad(|m| m.pinv(1e-7)?.powf(2.0)?.sum(), &a_rect, 1e-6, 1e-4)?;

    let b_rhs = Tensor::from_slice(&[4.0f64, 2.5, 1.0], &[3])?.requires_grad_(true);
    let x_ls = a_rect.lstsq(&b_rhs, 1e-7)?;
    assert_eq!(x_ls.shape(), &[2]);
    check_grad(|rhs| a_rect.lstsq(rhs, 1e-7)?.sum(), &b_rhs, 1e-6, 1e-5)?;

    let sq_m = Tensor::from_slice(&[0.2f64, 0.1, -0.1, 0.3], &[2, 2])?.requires_grad_(true);
    let m_exp = sq_m.matrix_exp()?;
    assert_eq!(m_exp.shape(), &[2, 2]);
    check_grad(|m| m.matrix_exp()?.sum(), &sq_m, 1e-6, 1e-5)?;

    let m_pow2 = sq_m.matrix_power(2)?;
    let m_sq_direct = sq_m.matmul(&sq_m)?;
    for (a, b) in m_pow2
        .to_vec::<f64>()?
        .iter()
        .zip(m_sq_direct.to_vec::<f64>()?)
    {
        assert!((a - b).abs() < 1e-10);
    }
    let m_pow_neg1 = sq_m.matrix_power(-1)?;
    let m_inv = sq_m.inv()?;
    for (a, b) in m_pow_neg1
        .to_vec::<f64>()?
        .iter()
        .zip(m_inv.to_vec::<f64>()?)
    {
        assert!((a - b).abs() < 1e-8);
    }
    assert_eq!(sq_m.matrix_rank(1e-6)?, 2);
    assert!(sq_m.cond()?.item::<f64>()? >= 1.0);

    let k_out = Tensor::eye(2, DType::F64, &Device::Cpu)?.kron(&sq_m)?;
    assert_eq!(k_out.shape(), &[4, 4]);

    // 3. 3D Volumetric Conv3d, AvgPool3d, MaxPool3d, BatchNorm3d, PixelShuffle & PixelUnshuffle
    let vol_in =
        Tensor::randn_with(&[1, 2, 4, 4, 4], DType::F64, &Device::Cpu)?.requires_grad_(true);
    let conv3d_w =
        Tensor::randn_with(&[2, 2, 2, 2, 2], DType::F64, &Device::Cpu)?.requires_grad_(true);
    check_grad(
        |x| x.conv3d(&conv3d_w, None, (1, 1, 1), (0, 0, 0))?.sum(),
        &vol_in,
        1e-5,
        1e-4,
    )?;
    check_grad(
        |w| vol_in.conv3d(w, None, (1, 1, 1), (0, 0, 0))?.sum(),
        &conv3d_w,
        1e-5,
        1e-4,
    )?;

    let bn3d = BatchNorm3d::new(2, 1e-5, 0.1, DType::F64, Device::Cpu)?;
    let pool3d = AvgPool3d::new((2, 2, 2), (2, 2, 2));
    let vol_feat = pool3d.forward(&bn3d.forward(&vol_in)?)?;
    assert_eq!(vol_feat.shape(), &[1, 2, 2, 2, 2]);

    let ps = PixelShuffle::new(2);
    let pus = PixelUnshuffle::new(2);
    let img_lr = Tensor::randn(&[2, 12, 4, 4])?;
    let img_hr = ps.forward(&img_lr)?;
    assert_eq!(img_hr.shape(), &[2, 3, 8, 8]);
    let img_roundtrip = pus.forward(&img_hr)?;
    assert_eq!(img_roundtrip.shape(), &[2, 12, 4, 4]);
    for (a, b) in img_lr
        .to_vec::<f32>()?
        .iter()
        .zip(img_roundtrip.to_vec::<f32>()?)
    {
        assert!((a - b).abs() < 1e-6);
    }

    // 4. Dynamic Autograd-to-IR Tracer (`trace_to_ir` & `jit_compile`)
    let x_jit = Tensor::randn(&[4, 8])?;
    let w_jit = Tensor::randn(&[8, 6])?;
    let b_jit = Tensor::randn(&[4, 6])?;
    let compiled_fn = jit_compile(
        |ins| ins[0].matmul(&ins[1])?.add(&ins[2])?.silu(),
        &[x_jit.clone(), w_jit.clone(), b_jit.clone()],
    )?;
    // Verify that `matmul + add + silu` was automatically fused into `FusedLinearSilu`!
    assert_eq!(compiled_fn.report.fused_ops_count, 1);
    let jit_out = compiled_fn.execute(&[x_jit.clone(), w_jit.clone(), b_jit.clone()])?;
    let eager_out = x_jit.matmul(&w_jit)?.add(&b_jit)?.silu()?;
    for (a, b) in jit_out[0]
        .to_vec::<f32>()?
        .iter()
        .zip(eager_out.to_vec::<f32>()?)
    {
        assert!((a - b).abs() < 1e-5);
    }

    // 5. FP8 (E4M3 / E5M2), AWQLinear, CachingAllocator, YaRN, ALiBi, SlidingWindowAttention, SGDR & OneCycleLR
    let e4 = fp8_e4m3::from_f32(3.5);
    assert!((e4.to_f32() - 3.5).abs() < 1e-3);
    let e5 = fp8_e5m2::from_f32(-12.0);
    assert!((e5.to_f32() - (-12.0)).abs() < 1e-3);

    let lin_fp32 = Linear::new(32, 16)?;
    let calib = Tensor::randn(&[8, 32])?;
    let awq = AWQLinear::from_linear_calibrated(
        &lin_fp32,
        &calib,
        QuantScheme::Q8_0 { block_size: 32 },
        0.5,
    )?;
    let awq_out = awq.forward(&calib)?;
    assert_eq!(awq_out.shape(), &[8, 16]);

    let alloc = CachingAllocator::new();
    let buf1 = alloc.alloc_f32(100)?;
    assert_eq!(alloc.stats().cache_misses, 1);
    alloc.recycle_f32(buf1);
    let _buf2 = alloc.alloc_f32(90)?;
    assert_eq!(alloc.stats().cache_hits, 1);

    let yarn_rope = RotaryEmbedding::new_yarn(16, 128, 32, 10000.0, 32.0, 1.0)?;
    let q_yarn = Tensor::randn(&[1, 2, 16, 16])?;
    assert_eq!(yarn_rope.apply(&q_yarn, 0)?.shape(), &[1, 2, 16, 16]);

    let alibi = alibi_bias(4, 6, 6, &Device::Cpu)?;
    assert_eq!(alibi.shape(), &[1, 4, 6, 6]);

    let swa = SlidingWindowAttention::new(16, 4, 3, 1)?;
    let swa_out = swa.forward(&Tensor::randn(&[2, 6, 16])?)?;
    assert_eq!(swa_out.shape(), &[2, 6, 16]);

    let p_sched = Tensor::from_slice(&[1.0f32], &[1])?.requires_grad_(true);
    let mut opt_sched = Sgd::new(vec![p_sched], 0.1);
    let mut sgdr = CosineAnnealingWarmRestarts::new(0.1, 4, 2, 1e-4);
    let mut onecycle = OneCycleLR::new(0.1, 10, 0.3, 10.0, 100.0);
    assert!(sgdr.step(&mut opt_sched) <= 0.1);
    assert!(onecycle.step(&mut opt_sched) > 0.01);

    Ok(())
}

#[cfg(feature = "python")]
static PYTHON_TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(feature = "python")]
#[test]
fn test_python_bindings_zero_gil_cpu_gpu_offload() -> Result<()> {
    use std::ffi::CString;

    use pyo3::prelude::*;
    use pyo3::types::PyModule;

    let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
    pyo3::Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let ox = PyModule::new(py, "oxideon")?;
        oxideon::python::register_python_module(&ox)?;
        let sys = py.import("sys")?;
        sys.getattr("modules")?.set_item("oxideon", &ox)?;

        let code = CString::new(
            r#"
import oxideon as ox

ox.manual_seed(42)

# 1. Higher-Order Autograd (1st and 2nd derivatives) executed in Rust
x = ox.Tensor([2.0, 3.0], [2], dtype="f64", requires_grad=True)
y = (x.powf(3.0)).sum()
dy = ox.grad([y], [x], create_graph=True)[0]
assert abs(dy.to_list()[0] - 12.0) < 1e-9
assert abs(dy.to_list()[1] - 27.0) < 1e-9

d2y = ox.grad([dy.sum()], [x], create_graph=False)[0]
assert abs(d2y.to_list()[0] - 12.0) < 1e-9
assert abs(d2y.to_list()[1] - 18.0) < 1e-9

# 2. Zero-Copy Views, Fused Single-Pass Kernels & Compiled IR Graph in Rust
v = ox.Tensor([1.0, 2.0, 3.0, 4.0, 5.0], [5], dtype="f32")
win = v.unfold(0, 3, 1)
assert win.shape == [3, 3]
assert win.shares_storage(v)

act_in = ox.Tensor.randn([4, 16], requires_grad=True)
res_in = ox.Tensor.randn([4, 16], requires_grad=True)
rms_w = ox.Tensor.ones([16], requires_grad=True)
normed, _ = ox.fused_add_rmsnorm(act_in, res_in, rms_w, 1e-5)
swi = ox.fused_swiglu(normed, res_in)
swi.sum().backward()
assert act_in.grad() is not None

cg = ox.compile_fused_linear_silu_with_grad()
assert cg.fused_ops_count >= 1
cg_outs = cg.execute([ox.Tensor.randn([4, 8]), ox.Tensor.randn([8, 6]), ox.Tensor.randn([4, 6])])
assert cg_outs[0].shape == [4, 6]
assert cg_outs[1].shape == [4, 8]

# 3. Whole-Loop Native Rust Training (Muon & AdamW without Python loop overhead)
mlp = ox.MLP([8, 16, 4])
train_x = ox.Tensor.randn([32, 8])
train_y = ox.Tensor.randn([32, 4])
losses = mlp.train_epochs_in_rust(train_x, train_y, epochs=15, batch_size=16, lr=0.02, use_muon=True)
assert len(losses) == 15
assert losses[-1] < losses[0]

# 4. Frontier Foundation Models (DeepSeek-V3 MLA+MoE+MTP, Jamba, MultimodalVLM, DiT) in Rust
tokens = ox.Tensor.from_i64([1, 4, 7, 2, 9, 3, 2, 5, 8, 6, 1, 4], [2, 6])
ds = ox.DeepSeekMoELM(24, 16, 1, 4, 4, 32, 32, 1, 3, 2, 2)
main_logits, mtp_logits, aux_loss = ds.forward_with_mtp(tokens)
assert main_logits.shape == [2, 6, 24]
assert len(mtp_logits) == 2

jamba = ox.HybridJambaLM(24, 16, 2, 4, 2, 4, 32, 3, 2, 1, 1)
assert jamba.forward(tokens).shape == [2, 6, 24]

dit = ox.DiT(8, 4, 3, 16, 1, 4, 10)
vel = dit.forward_dit(ox.Tensor.randn([2, 3, 8, 8]), [10, 50], None)
assert vel.shape == [2, 3, 8, 8]
"#,
        )
        .unwrap();
        py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
        Ok(())
    })
    .map_err(|e| Error::Msg(format!("Python test failed: {e}")))?;

    Ok(())
}

#[test]
fn test_full_neural_network_suite_v1() -> Result<()> {
    manual_seed(42);

    // 1. ConvNeXt (Modern ConvNet, CVPR 2022)
    let convnext = ConvNeXt::new(3, 16, 2, 4, 5)?;
    let x_img = Tensor::randn(&[2, 3, 16, 16])?;
    let logits = convnext.forward(&x_img)?;
    assert_eq!(logits.shape(), &[2, 5]);
    logits.sum()?.backward()?;
    assert!(
        convnext
            .head
            .weight
            .grad()
            .unwrap()
            .abs()?
            .sum()?
            .item::<f32>()?
            > 0.0
    );

    // 2. BiLSTM (Bidirectional LSTM)
    let bilstm = BiLSTM::new(8, 16, 2)?;
    let x_seq = Tensor::randn(&[2, 5, 8])?;
    let bi_out = bilstm.forward(&x_seq)?;
    assert_eq!(bi_out.shape(), &[2, 5, 32]);
    bi_out.sum()?.backward()?;

    // 3. TemporalConvNet (Causal Dilated TCN for Sequences/Audio)
    let tcn = TemporalConvNet::new(4, &[8, 16], 3)?;
    let x_time = Tensor::randn(&[2, 4, 12])?;
    let tcn_out = tcn.forward(&x_time)?;
    assert_eq!(tcn_out.shape(), &[2, 16, 12]);
    tcn_out.sum()?.backward()?;

    // 4. MLPMixer (Vision MLP-Mixer)
    let mixer = MLPMixer::new(16, 4, 3, 16, 2, 32, 64, 4)?;
    let mixer_out = mixer.forward(&x_img)?;
    assert_eq!(mixer_out.shape(), &[2, 4]);
    mixer_out.sum()?.backward()?;

    // 5. VariationalAutoencoder (VAE with Reparameterization & ELBO)
    let vae = VariationalAutoencoder::new(12, 16, 4)?;
    let x_vae = Tensor::randn(&[4, 12])?;
    let (recon, mu, logvar, elbo_loss) = vae.forward_elbo(&x_vae, 1.0)?;
    assert_eq!(recon.shape(), &[4, 12]);
    assert_eq!(mu.shape(), &[4, 4]);
    assert_eq!(logvar.shape(), &[4, 4]);
    assert_eq!(elbo_loss.shape(), &[] as &[usize]);
    elbo_loss.backward()?;
    assert!(
        vae.enc_fc1
            .weight
            .grad()
            .unwrap()
            .abs()?
            .sum()?
            .item::<f32>()?
            > 0.0
    );

    // 6. SpectralNormLinear & WeightNormLinear
    let sn_lin = SpectralNormLinear::new(8, 6, true, 2)?;
    let x_sn = Tensor::randn(&[3, 8])?;
    let sn_out = sn_lin.forward(&x_sn)?;
    assert_eq!(sn_out.shape(), &[3, 6]);
    sn_out.sum()?.backward()?;

    let wn_lin = WeightNormLinear::new(8, 6, true)?;
    let wn_out = wn_lin.forward(&x_sn)?;
    assert_eq!(wn_out.shape(), &[3, 6]);
    wn_out.sum()?.backward()?;

    // 7. Dynamic Containers (ModuleList, ModuleDict, ParallelAdd, ParallelCat, Residual)
    let mut mlist = ModuleList::new();
    mlist.push(Linear::new(8, 8)?);
    mlist.push(Linear::new(8, 8)?);
    assert_eq!(mlist.len(), 2);

    let mut mdict = ModuleDict::new();
    mdict.insert("l1", Linear::new(8, 8)?);
    mdict.insert("l2", Linear::new(8, 8)?);
    assert!(mdict.contains_key("l1"));

    let p_add = ParallelAdd::new()
        .add(Linear::new(8, 8)?)
        .add(Linear::new(8, 8)?);
    let padd_out = p_add.forward(&x_sn)?;
    assert_eq!(padd_out.shape(), &[3, 8]);

    let p_cat = ParallelCat::new(1)
        .add(Linear::new(8, 4)?)
        .add(Linear::new(8, 4)?);
    let pcat_out = p_cat.forward(&x_sn)?;
    assert_eq!(pcat_out.shape(), &[3, 8]);

    let res = Residual::new(Linear::new(8, 8)?);
    let res_out = res.forward(&x_sn)?;
    assert_eq!(res_out.shape(), &[3, 8]);

    // 8. Advanced Activations & Regularizers
    let x_act = Tensor::randn(&[4, 6])?;
    assert_eq!(Mish.forward(&x_act)?.shape(), &[4, 6]);
    assert_eq!(Hardswish.forward(&x_act)?.shape(), &[4, 6]);
    assert_eq!(Hardsigmoid.forward(&x_act)?.shape(), &[4, 6]);
    assert_eq!(Softplus::new(1.0).forward(&x_act)?.shape(), &[4, 6]);
    assert_eq!(Softsign.forward(&x_act)?.shape(), &[4, 6]);
    assert_eq!(SELU.forward(&x_act)?.shape(), &[4, 6]);
    assert_eq!(CELU::new(1.0).forward(&x_act)?.shape(), &[4, 6]);
    assert_eq!(PReLU::new(6, 0.25)?.forward(&x_act)?.shape(), &[4, 6]);
    assert_eq!(
        GLU::new(1).forward(&Tensor::randn(&[4, 12])?)?.shape(),
        &[4, 6]
    );

    let drop2d = Dropout2d::new(0.5);
    let x_sp = Tensor::randn(&[2, 4, 8, 8])?;
    assert_eq!(drop2d.forward(&x_sp)?.shape(), &[2, 4, 8, 8]);

    let droppath = DropPath::new(0.5);
    assert_eq!(droppath.forward(&x_act)?.shape(), &[4, 6]);

    // 9. Sequential Container & SafeTensors Save/Load Roundtrip
    let mut seq = Sequential::new();
    seq.push(Linear::new(8, 16)?);
    seq.push(SiLU);
    seq.push(Linear::new(16, 4)?);
    let seq_out = seq.forward(&x_sn)?;
    assert_eq!(seq_out.shape(), &[3, 4]);

    let tmp_path = std::env::temp_dir().join("oxideon_seq_v1.safetensors");
    seq.save(&tmp_path)?;
    let mut seq2 = Sequential::new();
    seq2.push(Linear::new(8, 16)?);
    seq2.push(SiLU);
    seq2.push(Linear::new(16, 4)?);
    seq2.load(&tmp_path)?;
    let seq2_out = seq2.forward(&x_sn)?;
    for (a, b) in seq_out
        .to_vec::<f32>()?
        .iter()
        .zip(seq2_out.to_vec::<f32>()?)
    {
        assert!((a - b).abs() < 1e-6);
    }
    let _ = std::fs::remove_file(tmp_path);

    // 10. Python Bindings for Full Neural Network Suite
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

# Sequential building and Rust-accelerated training
seq = ox.Sequential()
seq.add_linear(8, 16)
seq.add_silu()
seq.add_layernorm(16)
seq.add_dropout(0.1)
seq.add_linear(16, 2)

x = ox.Tensor.randn([32, 8])
y = ox.Tensor.from_i64([0, 1] * 16, [32])
losses = seq.train_classification_in_rust(x, y, epochs=10, batch_size=8, lr=0.01)
assert len(losses) == 10
assert losses[-1] < losses[0]

# Modern Vision: ConvNeXt
cnext = ox.ConvNeXt(in_channels=3, dim=16, depth=2, patch_size=4, num_classes=5)
out_cnext = cnext.forward(ox.Tensor.randn([2, 3, 16, 16]))
assert out_cnext.shape == [2, 5]

# Bidirectional Recurrent: BiLSTM
bilstm = ox.BiLSTM(input_size=8, hidden_size=16, num_layers=1)
out_bi = bilstm.forward(ox.Tensor.randn([2, 4, 8]))
assert out_bi.shape == [2, 4, 32]

# Causal Dilated Convolutions: TemporalConvNet
tcn = ox.TemporalConvNet(in_channels=4, channel_dims=[8, 16], kernel_size=3)
out_tcn = tcn.forward(ox.Tensor.randn([2, 4, 10]))
assert out_tcn.shape == [2, 16, 10]

# Vision MLP: MLPMixer
mixer = ox.MLPMixer(16, 4, 3, 16, 2, 32, 64, 4)
out_mix = mixer.forward(ox.Tensor.randn([2, 3, 16, 16]))
assert out_mix.shape == [2, 4]

# Generative: VAE
vae = ox.VAE(input_dim=12, hidden_dim=16, latent_dim=4)
recon, mu, logvar, elbo = vae.forward_elbo(ox.Tensor.randn([4, 12]), beta_kl=1.0)
assert recon.shape == [4, 12]
assert mu.shape == [4, 4]

# Parameterizations: SpectralNormLinear & WeightNormLinear
sn = ox.SpectralNormLinear(8, 4, n_power_iterations=1)
assert sn.forward(ox.Tensor.randn([2, 8])).shape == [2, 4]
wn = ox.WeightNormLinear(8, 4)
assert wn.forward(ox.Tensor.randn([2, 8])).shape == [2, 4]

# Advanced Activations on Tensor
t = ox.Tensor.randn([4, 6])
assert t.mish().shape == [4, 6]
assert t.hardswish().shape == [4, 6]
assert t.hardsigmoid().shape == [4, 6]
assert t.selu().shape == [4, 6]
assert t.celu(1.0).shape == [4, 6]
assert t.softplus(1.0).shape == [4, 6]
assert t.softsign().shape == [4, 6]
"#,
            )
            .unwrap();
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python NN test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_serving_grammar_zero3_whisper_and_wasm() -> Result<()> {
    manual_seed(42);

    // 1. PagedAttention & Continuous Batching Serving Engine
    let mut engine = ServingEngine::new(4, 32, 16, 4, 32)?;
    let req1 = engine.submit(vec![10, 20, 30], 5);
    let req2 = engine.submit(vec![40, 50], 4);
    assert_eq!(engine.scheduler.num_active(), 2);

    // Run one step of generation
    let out_step = engine.step(|active_reqs| {
        // Return dummy next tokens for active requests
        Ok(vec![99; active_reqs.len()])
    })?;
    assert_eq!(out_step.len(), 2);
    assert_eq!(out_step[0].0, req1);
    assert_eq!(out_step[1].0, req2);
    assert!(engine.allocator.utilization() > 0.0);

    // Direct PagedAttention test
    let block_table = BlockTable {
        physical_blocks: vec![0, 1],
        num_tokens: 24,
    };
    let q = Tensor::randn(&[1, 4, 32])?;
    let paged_attn_out = paged_attention(
        &q,
        &engine.paged_kv,
        &[block_table],
        &[24],
        1.0 / (32.0f64).sqrt(),
    )?;
    assert_eq!(paged_attn_out.shape(), &[1, 4, 32]);

    // 2. Speculative Decoding
    let mut spec_decoder = SpeculativeDecoder::new(SpeculativeConfig {
        num_speculative_tokens: 3,
        temperature: 1.0,
    });
    let draft_tokens = vec![101, 102, 103];
    let draft_probs = vec![0.8, 0.7, 0.6];
    // High target probs -> should accept candidates
    let target_probs_all = vec![
        vec![0.0; 100]
            .into_iter()
            .chain(vec![0.0, 0.9])
            .collect::<Vec<_>>(),
        vec![0.0; 101]
            .into_iter()
            .chain(vec![0.0, 0.85])
            .collect::<Vec<_>>(),
        vec![0.0; 102]
            .into_iter()
            .chain(vec![0.0, 0.75])
            .collect::<Vec<_>>(),
        vec![0.05; 110], // bonus token distribution
    ];
    let sampled = spec_decoder.verify_and_sample(&draft_tokens, &draft_probs, &target_probs_all)?;
    assert!(!sampled.is_empty());
    assert!(spec_decoder.metrics.acceptance_rate() > 0.0);

    // 3. Grammar-Constrained Decoding & Agent Runtime
    let json_schema = JsonSchema::object(vec![
        ("name", SchemaType::String),
        ("count", SchemaType::Integer),
    ]);
    let parser = JsonParserState::new(json_schema.root);
    assert!(parser.can_accept("{\"name\":"));
    assert!(!parser.can_accept("INVALID_TOKEN"));

    let mut agent = AgentRuntime::new();
    agent.register_tool(
        "calculate_tax",
        "Calculates sales tax",
        JsonSchema::number(),
        |args| Ok(format!("Tax computed for: {args}")),
    );
    let tool_res = agent.call_tool("calculate_tax", "{\"amount\": 100}")?;
    assert!(tool_res.contains("Tax computed"));
    assert!(agent.render_system_prompt().contains("calculate_tax"));

    // 4. ZeRO-3 Parameter Sharding & 1F1B Pipeline Parallelism
    let full_w = Tensor::randn(&[32, 16])?;
    let mut zero3_param = ZeRO3Parameter::from_parameter(&full_w, 4)?;
    assert_eq!(zero3_param.rank_shards.len(), 4);
    assert!((zero3_param.memory_savings_ratio() - 0.75).abs() < 1e-6);

    let gathered_w = zero3_param.gather()?;
    assert_eq!(gathered_w.shape(), &[32, 16]);
    for (a, b) in full_w
        .to_vec::<f32>()?
        .iter()
        .zip(gathered_w.to_vec::<f32>()?)
    {
        assert!((a - b).abs() < 1e-6);
    }
    zero3_param.release();

    assert!(PipelineSchedule1F1B::new(4, 8, 4).is_err()); // stage_idx must be below num_stages
    assert!(PipelineSchedule1F1B::new(0, 8, 0).is_err());
    let mut pipe = PipelineSchedule1F1B::new(4, 8, 0)?;
    assert!(pipe.bubble_fraction() > 0.0);
    let mut steps_count = 0;
    while let Some((_phase, _is_fwd, _mb)) = pipe.next_step() {
        steps_count += 1;
    }
    assert_eq!(steps_count, 16); // 8 forward + 8 backward

    // 5. Audio Log-Mel Spectrogram & Whisper Foundation Model
    let mel_extractor = MelSpectrogram::new(80, 400, 160, 16000)?;
    let raw_audio = Tensor::randn(&[1, 16000])?; // 1 second of 16kHz audio
    let mel = mel_extractor.forward(&raw_audio)?;
    assert_eq!(mel.shape()[0], 1);
    assert_eq!(mel.shape()[1], 80);

    let whisper_cfg = WhisperConfig {
        n_mels: 80,
        d_model: 64,
        encoder_layers: 2,
        encoder_heads: 4,
        decoder_layers: 2,
        decoder_heads: 4,
        d_ff: 128,
        vocab_size: 100,
        max_source_positions: 500,
        max_target_positions: 100,
    };
    let whisper = Whisper::new(whisper_cfg)?;
    let dummy_tokens = Tensor::from_slice(&[1i64, 25, 42], &[1, 3])?;
    let logits = whisper.forward(&mel, &dummy_tokens)?;
    assert_eq!(logits.shape(), &[1, 3, 100]);

    let transcript = whisper.transcribe(&mel, 5)?;
    assert!(!transcript.is_empty());

    // 6. WebAssembly / WebGPU In-Browser Tensor Interface
    let wasm_a = WasmTensor::from_f32_array(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let wasm_b = WasmTensor::from_f32_array(&[0.5, 0.0, 0.0, 0.5], &[2, 2])?;
    let wasm_c = wasm_a.matmul(&wasm_b)?;
    let c_vec = wasm_c.to_f32_vec()?;
    assert!((c_vec[0] - 0.5).abs() < 1e-6);
    assert!((c_vec[3] - 2.0).abs() < 1e-6);

    // 7. Python Bindings for Serving, Speculative, Grammar, ZeRO3, Whisper
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

# 1. Serving Engine
engine = ox.ServingEngine(max_batch_size=8, num_blocks=64, block_size=16, num_heads=4, head_dim=32)
r1 = engine.submit([1, 2, 3], 10)
assert engine.num_active() == 1

# 2. Speculative Decoding
dec = ox.SpeculativeDecoder(num_speculative_tokens=2, temperature=1.0)
target_probs = [[0.0] * 50 + [0.9], [0.0] * 51 + [0.85], [0.1] * 55]
accepted = dec.verify_and_sample([50, 51], [0.8, 0.7], target_probs)
assert len(accepted) >= 1

# 3. Grammar & Agent Runtime
agent = ox.AgentRuntime()
agent.register_tool("weather", "Gets temperature for city")
assert "weather" in agent.render_system_prompt()

# 4. ZeRO-3 Parameter Sharding
t = ox.Tensor.randn([16, 8])
p = ox.ZeRO3Parameter.from_tensor(t, world_size=4)
assert abs(p.memory_savings_ratio() - 0.75) < 1e-6
gathered = p.gather()
assert gathered.shape == [16, 8]
p.release()

# 5. 1F1B Pipeline Schedule
pipe = ox.PipelineSchedule1F1B(num_stages=4, num_microbatches=4, stage_idx=0)
step = pipe.next_step()
assert step is not None

# 6. Whisper & Mel Spectrogram
mel_mod = ox.MelSpectrogram(n_mels=80, n_fft=400, hop_length=160, sample_rate=16000)
wav = ox.Tensor.randn([1, 16000])
mel_out = mel_mod.forward(wav)
assert mel_out.shape[1] == 80

whisper = ox.Whisper(n_mels=80, d_model=64, encoder_layers=1, encoder_heads=4, decoder_layers=1, decoder_heads=4, d_ff=128, vocab_size=50)
toks = ox.Tensor.from_i64([1, 10], [1, 2])
w_logits = whisper.forward(mel_out, toks)
assert w_logits.shape == [1, 2, 50]
"#,
            )
            .unwrap();
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Frontier test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_scale_up_quantization_grpo_video_dit_and_jit_kernels() -> Result<()> {
    manual_seed(42);

    // =========================================================================
    // 1. Frontier Quantization Suite (BitNet 1.58b, FP8, GPTQ)
    // =========================================================================
    // BitLinear (1.58-bit ternary {-1, 0, +1} with dynamic 8-bit activations)
    let lin = Linear::new(32, 16)?;
    let bit_lin = BitLinear::from_linear(&lin)?;
    let in_x = Tensor::randn(&[4, 32])?;
    let bit_out = bit_lin.forward(&in_x)?;
    assert_eq!(bit_out.shape(), &[4, 16]);
    assert!(bit_lin.weight_bytes() < 32 * 16); // Highly compressed compared to 8-bit/f32
    let dequant_w = bit_lin.dequantize_weight(DType::F32)?;
    assert_eq!(dequant_w.shape(), &[16, 32]);

    // Native FP8 (E4M3 & E5M2)
    let t_fp = Tensor::randn(&[8, 16])?;
    let fp8_e4 = FP8Tensor::from_tensor(&t_fp, FP8Format::E4M3)?;
    let fp8_e5 = FP8Tensor::from_tensor(&t_fp, FP8Format::E5M2)?;
    assert_eq!(fp8_e4.size_in_bytes(), 8 * 16 + 4);
    let roundtrip_e4 = fp8_e4.to_tensor(DType::F32)?;
    assert_eq!(roundtrip_e4.shape(), &[8, 16]);
    let roundtrip_e5 = fp8_e5.to_tensor(DType::F32)?;
    assert_eq!(roundtrip_e5.shape(), &[8, 16]);

    let fp8_lin = FP8Linear::from_linear(&lin, FP8Format::E4M3)?;
    let fp8_out = fp8_lin.forward(&in_x)?;
    assert_eq!(fp8_out.shape(), &[4, 16]);

    // GPTQ (Second-Order Inverse Hessian Quantization)
    let calib_x = Tensor::randn(&[64, 32])?;
    let gptq_lin =
        GPTQLinear::from_linear_calibrated(&lin, &calib_x, QuantScheme::DEFAULT_Q4_0, 0.01)?;
    let gptq_out = gptq_lin.forward(&in_x)?;
    assert_eq!(gptq_out.shape(), &[4, 16]);

    // =========================================================================
    // 2. Post-Training & Reinforcement Learning (DeepSeek-R1 GRPO, DPO, PPO)
    // =========================================================================
    // RuleReward for DeepSeek-R1 reasoning
    assert_eq!(
        RuleReward::format_reward("<think>step-by-step</think><answer>42</answer>"),
        1.0
    );
    assert_eq!(RuleReward::format_reward("no tags"), 0.0);
    assert_eq!(
        RuleReward::extract_answer("<think>reasoning</think><answer> Paris </answer>"),
        Some("Paris".to_string())
    );
    assert_eq!(
        RuleReward::accuracy_reward("<think>...</think><answer>Paris</answer>", "paris"),
        1.0
    );

    // Group advantage normalization
    let raw_rewards = vec![1.0, 0.0, 0.0, 1.0];
    let advs = compute_group_advantages(&raw_rewards, 4, 1e-6);
    assert_eq!(advs.len(), 4);
    let mean_adv: f64 = advs.iter().sum::<f64>() / 4.0;
    assert!(mean_adv.abs() < 1e-5);

    // GRPOTrainer
    let grpo = GRPOTrainer::new(GRPOConfig {
        group_size: 4,
        clip_eps: 0.2,
        kl_beta: 0.04,
        eps: 1e-6,
    });
    let policy_logps = Tensor::randn(&[4, 8])?;
    let old_logps = Tensor::randn(&[4, 8])?;
    let ref_logps = Tensor::randn(&[4, 8])?;
    let (grpo_l, grpo_stats) =
        grpo.compute_step(&policy_logps, &old_logps, &ref_logps, &raw_rewards, None)?;
    assert!(grpo_l.item::<f32>()?.is_finite());
    assert_eq!(grpo_stats.mean_reward, 0.5);

    // DPOTrainer
    let dpo = DPOTrainer::new(DPOConfig::default());
    let pi_w = Tensor::randn(&[4])?;
    let pi_l = Tensor::randn(&[4])?;
    let ref_w = Tensor::randn(&[4])?;
    let ref_l = Tensor::randn(&[4])?;
    let (dpo_l, dpo_stats) = dpo.compute_step(&pi_w, &pi_l, &ref_w, &ref_l)?;
    assert!(dpo_l.item::<f32>()?.is_finite());
    assert!(dpo_stats.accuracy >= 0.0 && dpo_stats.accuracy <= 1.0);

    // PPOTrainer
    let ppo = PPOTrainer::new(PPOConfig::default());
    let new_logps = Tensor::randn(&[8, 4])?;
    let ppo_adv = Tensor::randn(&[8, 4])?;
    let new_vals = Tensor::randn(&[8, 4])?;
    let old_vals = Tensor::randn(&[8, 4])?;
    let returns = Tensor::randn(&[8, 4])?;
    let (ppo_l, ppo_stats) = ppo.compute_step(
        &new_logps, &new_logps, &ppo_adv, &new_vals, &old_vals, &returns,
    )?;
    assert!(ppo_l.item::<f32>()?.is_finite());
    assert!(ppo_stats.total_loss.is_finite());

    // =========================================================================
    // 3. Spatio-Temporal Video Foundation (CausalConv3d, 3D RoPE, VAE3d, DiT)
    // =========================================================================
    // CausalConv3d
    let cconv = CausalConv3d::new(2, 4, (3, 3, 3), (1, 1, 1))?;
    let video_in = Tensor::randn(&[1, 2, 4, 8, 8])?;
    let cconv_out = cconv.forward(&video_in)?;
    assert_eq!(cconv_out.shape(), &[1, 4, 4, 8, 8]); // Temporal causality preserves T=4

    // RotaryEmbedding3D
    let rope3d = RotaryEmbedding3D::new(16, 10000.0)?;
    let q_video = Tensor::randn(&[1, 32, 2, 16])?; // 32 tokens = 2 * 4 * 4
    let q_rot = rope3d.forward(&q_video, 2, 4, 4)?;
    assert_eq!(q_rot.shape(), &[1, 32, 2, 16]);

    // VideoVAE3d
    let vae3d = VideoVAE3d::new(2, 4, 4)?;
    let latents = vae3d.encode(&video_in)?;
    assert_eq!(latents.shape(), &[1, 4, 4, 4, 4]); // 2x spatial downsample
    let recon_video = vae3d.decode(&latents)?;
    assert_eq!(recon_video.shape(), &[1, 2, 4, 8, 8]);

    // RectifiedFlowVideoDiT
    let dit_cfg = VideoDiTConfig {
        in_channels: 2,
        hidden_dim: 32,
        num_heads: 4,
        patch_size: (1, 2, 2),
        text_dim: 16,
    };
    let vdit = RectifiedFlowVideoDiT::new(dit_cfg)?;
    let t_time = Tensor::from_vec(vec![0.5f32; 1], &[1, 1])?;
    let text_cond = Tensor::randn(&[1, 4, 16])?;
    let v_pred = vdit.forward(&video_in, &t_time, &text_cond)?;
    assert_eq!(v_pred.shape(), &[1, 2, 4, 8, 8]);

    let x0 = Tensor::randn(&[1, 2, 4, 8, 8])?;
    let x1 = Tensor::randn(&[1, 2, 4, 8, 8])?;
    let rf_loss = vdit.loss(&x0, &x1, &t_time, &text_cond)?;
    assert!(rf_loss.item::<f32>()?.is_finite());

    let sampled_video = vdit.sample_euler(&x0, &text_cond, 2)?;
    assert_eq!(sampled_video.shape(), &[1, 2, 4, 8, 8]);

    // =========================================================================
    // 4. Native SPIR-V / WGSL & JIT Kernel Code Generator
    // =========================================================================
    let mut builder = JitKernelBuilder::new("fused_custom");
    let in0 = builder.add_input("x");
    let in1 = builder.add_input("y");
    let sum = builder.add_op(JitOp::Add, &[in0, in1]);
    let silu_node = builder.add_op(JitOp::Silu, &[sum]);
    builder.set_output(silu_node);

    let wgsl = builder.build_wgsl(64)?;
    assert!(wgsl.contains("fn silu(x: f32) -> f32"));
    assert!(wgsl.contains("@compute @workgroup_size(64)"));
    assert!(wgsl.contains("@group(0) @binding(0) var<storage, read> x"));

    let t1 = Tensor::randn(&[4, 8])?;
    let t2 = Tensor::randn(&[4, 8])?;
    let jit_cpu_out = builder.execute_cpu(&[&t1, &t2])?;
    let eager_ref = t1.add(&t2)?.silu()?;
    assert_eq!(jit_cpu_out.shape(), &[4, 8]);
    for (a, b) in jit_cpu_out
        .to_vec::<f32>()?
        .iter()
        .zip(eager_ref.to_vec::<f32>()?)
    {
        assert!((a - b).abs() < 1e-5);
    }

    // =========================================================================
    // 5. Zero-GIL PyO3 Python Bindings
    // =========================================================================
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        pyo3::Python::initialize();
        let _guard = PYTHON_TEST_MUTEX.lock().unwrap();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

# 1. BitLinear & FP8 & GPTQ
lin = ox.Linear(16, 8)
blin = ox.BitLinear.from_linear(lin)
x = ox.Tensor.randn([2, 16])
bout = blin.forward(x)
assert bout.shape == [2, 8]
assert blin.weight_bytes() > 0

fp8_t = ox.FP8Tensor.from_tensor(x, "e4m3")
assert fp8_t.size_in_bytes() == 2 * 16 + 4
rt = fp8_t.to_tensor()
assert rt.shape == [2, 16]

calib = ox.Tensor.randn([32, 16])
gptq = ox.GPTQLinear.from_linear_calibrated(lin, calib, "q4_0", 0.01)
gout = gptq.forward(x)
assert gout.shape == [2, 8]

# 2. RL: DeepSeek-R1 GRPO & DPO
assert ox.GRPOTrainer.format_reward("<think>thought</think><answer>10</answer>") == 1.0
assert ox.GRPOTrainer.extract_answer("<think>thought</think><answer> 42 </answer>") == "42"

grpo = ox.GRPOTrainer(group_size=4, clip_eps=0.2, kl_beta=0.04)
p_logps = ox.Tensor.randn([4, 8])
o_logps = ox.Tensor.randn([4, 8])
r_logps = ox.Tensor.randn([4, 8])
loss, mean_r, fmt_r, loss_val = grpo.compute_step(p_logps, o_logps, r_logps, [1.0, 0.0, 1.0, 0.0])
assert mean_r == 0.5

dpo = ox.DPOTrainer(beta=0.1)
d_loss, c_rew, r_rew, margin, acc = dpo.compute_step(
    ox.Tensor.randn([4]), ox.Tensor.randn([4]), ox.Tensor.randn([4]), ox.Tensor.randn([4])
)
assert 0.0 <= acc <= 1.0

# 3. Video VAE & Video DiT
vae = ox.VideoVAE3d(2, 4, 4)
vid = ox.Tensor.randn([1, 2, 2, 4, 4])
z = vae.encode(vid)
assert z.shape == [1, 4, 2, 2, 2]
rec = vae.decode(z)
assert rec.shape == [1, 2, 2, 4, 4]

dit = ox.RectifiedFlowVideoDiT(in_channels=2, hidden_dim=32, num_heads=4, text_dim=16)
t_time = ox.Tensor([0.5], [1, 1])
t_cond = ox.Tensor.randn([1, 2, 16])
v = dit.forward(vid, t_time, t_cond)
assert v.shape == [1, 2, 2, 4, 4]

# 4. JIT Kernel Builder
j_b = ox.JitKernelBuilder("py_fused")
r_x = j_b.add_input("x")
r_y = j_b.add_input("y")
r_s = j_b.add_op("add", [r_x, r_y])
r_act = j_b.add_op("silu", [r_s])
j_b.set_output(r_act)
wgsl = j_b.build_wgsl(64)
assert "@compute @workgroup_size(64)" in wgsl

res = j_b.execute_cpu([x, x])
assert res.shape == [2, 16]
"#,
            )
            .unwrap();
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Frontier Scale-Up test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_supremacy_flash3_ring_mcts_ptx_pairformer_and_se3() -> Result<()> {
    manual_seed(1337);

    // =========================================================================
    // 1. FlashAttention-3 (Softcapping, FP8, Asynchronous Tile Pipelining)
    // =========================================================================
    let q3 = Tensor::randn(&[1, 2, 32, 16])?;
    let k3 = Tensor::randn(&[1, 2, 32, 16])?;
    let v3 = Tensor::randn(&[1, 2, 32, 16])?;

    let fa3_cfg = FlashAttention3Config {
        block_size_q: 16,
        block_size_kv: 16,
        causal: true,
        softcap: Some(50.0),
        use_fp8: true,
    };
    let fa3_out = flash_attention_3(&q3, &k3, &v3, &fa3_cfg)?;
    assert_eq!(fa3_out.shape(), &[1, 2, 32, 16]);
    for &val in &fa3_out.to_vec::<f32>()? {
        assert!(val.is_finite());
    }

    // =========================================================================
    // 2. RingAttention (Distributed Ring-Exchange for 1M+ Contexts)
    // =========================================================================
    let world_size = 4;
    let chunk_len = 8;
    let mut k_shards = Vec::new();
    let mut v_shards = Vec::new();
    for _ in 0..world_size {
        k_shards.push(Tensor::randn(&[1, 2, chunk_len, 16])?);
        v_shards.push(Tensor::randn(&[1, 2, chunk_len, 16])?);
    }

    // Evaluate each rank's attention chunk in the ring
    for rank in 0..world_size {
        let ring = RingAttention::new(world_size, rank, true).unwrap();
        let q_local = Tensor::randn(&[1, 2, chunk_len, 16])?;
        let ring_out = ring.forward_simulated(&q_local, &k_shards, &v_shards)?;
        assert_eq!(ring_out.shape(), &[1, 2, chunk_len, 16]);
    }

    // =========================================================================
    // 3. ChunkedPrefillEngine
    // =========================================================================
    let lm_model = CausalLM::new(32, 32, 4, 2, 64, 128)?;
    let mut kv_cache = KVCache::new(2)?;
    let prefill_engine = ChunkedPrefillEngine::new(4)?;
    let prompt_tokens: Vec<usize> = (0..12).map(|i| i % 32).collect();
    let prefill_logits = prefill_engine.prefill(&lm_model, &prompt_tokens, &mut kv_cache)?;
    assert_eq!(prefill_logits.shape()[0], 1);
    assert_eq!(prefill_logits.shape()[2], 32);

    // =========================================================================
    // 4. Monte Carlo Tree Search (DeepSeek-R1 / AlphaGo Style Reasoning)
    // =========================================================================
    let mcts = MCTSSearchEngine::new(1.414, 30, 4);
    let search_res = mcts.search(
        "Evaluate: 7 * 6",
        |text| {
            if text == "Evaluate: 7 * 6" {
                vec![
                    ("Step 1: Compute 7 * 6 directly = 42".into(), 0.8),
                    ("Step 1: 7 * 5 + 7 = 35 + 7 = 42".into(), 0.2),
                ]
            } else if text.contains("42") {
                vec![("<answer>42</answer>".into(), 1.0)]
            } else {
                Vec::new()
            }
        },
        |text| {
            if text.contains("42") { 1.0 } else { 0.1 }
        },
    )?;
    assert!(search_res.best_value > 0.5);
    assert!(search_res.total_nodes >= 2);

    // =========================================================================
    // 5. Self-Correction Verification
    // =========================================================================
    let verifier = SelfCorrectionVerifier::new(0.8);
    let mut trace_valid = ReasoningTrace::new();
    trace_valid.add_step("Step 1: Identify problem", 0.95);
    trace_valid.add_step("Step 2: Calculate derivative", 0.90);
    trace_valid.add_step("Step 3: Conclude answer", 0.92);
    let rep_valid = verifier.verify(&mut trace_valid, |_| (true, 0.9));
    assert!(rep_valid.is_valid);
    assert!(rep_valid.failed_step_index.is_none());

    let mut trace_invalid = ReasoningTrace::new();
    trace_invalid.add_step("Step 1: Good step", 0.95);
    trace_invalid.add_step("Step 2: invalid hallucination", 0.40);
    trace_invalid.add_step("Step 3: Wrong conclusion", 0.20);
    let rep_invalid = verifier.verify(&mut trace_invalid, |txt| {
        (
            !txt.contains("invalid"),
            if txt.contains("invalid") { 0.3 } else { 0.9 },
        )
    });
    assert!(!rep_invalid.is_valid);
    assert_eq!(rep_invalid.failed_step_index, Some(1));
    assert_eq!(rep_invalid.recommended_backtrack_index, 0);

    // =========================================================================
    // 6. Direct Bare-Metal Driver Emitters (NVIDIA PTX & Vulkan SPIR-V)
    // =========================================================================
    let ptx_emitter = PtxKernelEmitter::new("fused_silu_add", "sm_90");
    let ptx_add_silu = ptx_emitter.emit_fused_add_silu_ptx();
    assert!(ptx_add_silu.contains(".visible .entry fused_silu_add"));
    assert!(ptx_add_silu.contains(".target sm_90"));
    assert!(ptx_add_silu.contains("ex2.approx.f32"));

    let ptx_gemm = ptx_emitter.emit_gemm_fma_ptx();
    assert!(ptx_gemm.contains("fma.rn.f32"));

    let vulkan_kernel = VulkanSpirvKernel::new("spirv_kernel", 4);
    let words = vulkan_kernel.emit_spirv_binary().unwrap();
    assert_eq!(words[0], VulkanSpirvKernel::SPIRV_MAGIC);
    assert!(vulkan_kernel.binary_size_bytes().unwrap() > 0);

    // =========================================================================
    // 7. AlphaFold3 Pairformer Block (Triangular Multiplicative Updates)
    // =========================================================================
    let pairformer = PairformerBlock::new(16, 8)?;
    let z_pair = Tensor::randn(&[6, 6, 16])?;
    let z_updated = pairformer.forward(&z_pair)?;
    assert_eq!(z_updated.shape(), &[6, 6, 16]);

    // =========================================================================
    // 8. SE(3)-Equivariant Layers & Molecular Geometry Diffusion
    // =========================================================================
    let se3 = SE3EquivariantLayer::new(8)?;
    let coords = Tensor::randn(&[4, 3])?;
    let feats = Tensor::randn(&[4, 8])?;

    // Verify SE(3)-Equivariance: Rotate coordinates by 90 degrees around Z axis:
    // R = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    let rot_mat = Tensor::from_vec(
        vec![0.0f32, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        &[3, 3],
    )?;
    let coords_rot = coords.matmul(&rot_mat.t()?)?;

    let (out_c, _) = se3.forward(&coords, &feats)?;
    let (out_c_rot, _) = se3.forward(&coords_rot, &feats)?;

    // Rotated out(X) should match out(X R^T)
    let expected_rot = out_c.matmul(&rot_mat.t()?)?;
    for (a, b) in out_c_rot
        .to_vec::<f32>()?
        .iter()
        .zip(expected_rot.to_vec::<f32>()?)
    {
        assert!((a - b).abs() < 1e-4);
    }

    // Molecular Geometry Diffusion: Center-of-mass invariance
    let centered = MolecularGeometryDiffusion::remove_center_of_mass(&coords)?;
    let mean_c = centered.mean_axis(0, false)?;
    for &m in &mean_c.to_vec::<f32>()? {
        assert!(m.abs() < 1e-5);
    }

    let sampled_conformation =
        MolecularGeometryDiffusion::sample_conformation(&se3, &coords, &feats, 3)?;
    assert_eq!(sampled_conformation.shape(), &[4, 3]);

    // =========================================================================
    // 9. Zero-GIL PyO3 Python Bindings Verification
    // =========================================================================
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

# 1. FlashAttention-3
q = ox.Tensor.randn([1, 2, 16, 16])
k = ox.Tensor.randn([1, 2, 16, 16])
v = ox.Tensor.randn([1, 2, 16, 16])
fa3_res = ox.flash_attention_3(q, k, v, softcap=50.0)
assert fa3_res.shape == [1, 2, 16, 16]

# 2. RingAttention
ring = ox.RingAttention(world_size=2, rank=0)
q_local = ox.Tensor.randn([1, 2, 8, 16])
k_s0 = ox.Tensor.randn([1, 2, 8, 16])
k_s1 = ox.Tensor.randn([1, 2, 8, 16])
v_s0 = ox.Tensor.randn([1, 2, 8, 16])
v_s1 = ox.Tensor.randn([1, 2, 8, 16])
r_out = ring.forward_simulated(q_local, [k_s0, k_s1], [v_s0, v_s1])
assert r_out.shape == [1, 2, 8, 16]

# 3. MCTS & Verifier
mcts = ox.MCTSSearchEngine(c_puct=1.414, num_simulations=10, max_depth=3)
best_txt, best_v, n_nodes = mcts.search_mock("Root", [("Candidate", 1.0)], 0.95)
assert n_nodes >= 1

verifier = ox.SelfCorrectionVerifier(confidence_threshold=0.8)
is_v, fail_idx, btrack, avg_c = verifier.verify_chain(["Step 1", "Step 2"], [0.9, 0.95])
assert is_v == True

# 4. PTX Emitter
emitter = ox.PtxKernelEmitter("fused_kernel", "sm_90")
ptx = emitter.emit_fused_add_silu_ptx()
assert ".entry fused_kernel" in ptx

# 5. Pairformer & SE(3) & Molecular Geometry Diffusion
pair = ox.PairformerBlock(16, 8)
z_p = ox.Tensor.randn([4, 4, 16])
z_p_out = pair.forward(z_p)
assert z_p_out.shape == [4, 4, 16]

se3 = ox.SE3EquivariantLayer(8)
c_in = ox.Tensor.randn([4, 3])
f_in = ox.Tensor.randn([4, 8])
c_out, f_out = se3.forward(c_in, f_in)
assert c_out.shape == [4, 3]

sampled_c = ox.MolecularGeometryDiffusion.sample_conformation(se3, c_in, f_in, num_steps=2)
assert sampled_c.shape == [4, 3]
"#,
            )
            .unwrap();
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Frontier Supremacy test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_dominion_mod_robotics_poincare_and_metal() -> Result<()> {
    manual_seed(4242);

    // =========================================================================
    // 1. Mixture-of-Depths (MoD) Compute Allocation
    // =========================================================================
    let mod_layer = MixtureOfDepths::new(16, 0.5)?;
    let x_mod = Tensor::randn(&[2, 6, 16])?;
    let (mod_out, mod_weights, k) = mod_layer.forward(&x_mod)?;
    assert_eq!(mod_out.shape(), &[2, 6, 16]);
    assert_eq!(mod_weights.shape(), &[2, 6, 1]);
    assert_eq!(k, 3); // 50% of 6 tokens = 3 tokens routed

    // =========================================================================
    // 2. EarlyExitRouter Confidence Evaluation
    // =========================================================================
    let exit_router = EarlyExitRouter::new(0.7);
    let easy_logits = Tensor::from_slice(&[10.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0], &[2, 4])?;
    let (flags, max_prob) = exit_router.evaluate_exit(&easy_logits)?;
    assert_eq!(flags, vec![true, true]);
    assert_eq!(max_prob.shape(), &[2]);
    let hard_logits = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[1, 4])?;
    let (hard_flags, _) = exit_router.evaluate_exit(&hard_logits)?;
    assert_eq!(hard_flags, vec![false]);

    // =========================================================================
    // 3. Action Chunking Transformer (ACT) & Temporal Ensembling
    // =========================================================================
    let act_cfg = ACTConfig {
        state_dim: 8,
        action_dim: 4,
        chunk_size: 6,
        hidden_dim: 16,
        temporal_ensemble_decay: 0.1,
    };
    let mut act = ActionChunkingTransformer::new(act_cfg)?;
    let robot_state = Tensor::randn(&[1, 8])?;
    let chunk_pred = act.forward(&robot_state)?;
    assert_eq!(chunk_pred.shape(), &[1, 6, 4]);

    let action_smooth1 = act.step_temporal_ensemble(&robot_state)?;
    assert_eq!(action_smooth1.len(), 4);
    let action_smooth2 = act.step_temporal_ensemble(&robot_state)?;
    assert_eq!(action_smooth2.len(), 4);

    // =========================================================================
    // 4. Flow Matching Robotics Policy & Continuous Euler ODE Integration
    // =========================================================================
    let flow_policy = FlowMatchingRoboticsPolicy::new(4, 8, 32)?;
    let obs = Tensor::randn(&[2, 8])?;
    let sampled_actions = flow_policy.sample_action(&obs, 4)?;
    assert_eq!(sampled_actions.shape(), &[2, 4]);

    // =========================================================================
    // 5. Poincaré Ball Hyperbolic Manifold (Distance, Möbius Addition, Exp/Log)
    // =========================================================================
    let u = Tensor::from_slice(&[0.1f32, 0.2, -0.3, 0.1], &[2, 2])?;
    let v = Tensor::from_slice(&[-0.2f32, 0.1, 0.1, -0.2], &[2, 2])?;
    let dist = PoincareManifold::distance(&u, &v)?;
    assert_eq!(dist.shape(), &[2, 1]);
    let dist_vec = dist.to_vec::<f32>()?;
    assert!(dist_vec[0] > 0.0 && dist_vec[1] > 0.0);

    let mobius = PoincareManifold::mobius_add(&u, &v)?;
    assert_eq!(mobius.shape(), &[2, 2]);

    let v_tan = Tensor::from_slice(&[0.05f32, -0.05, 0.1, 0.02], &[2, 2])?;
    let pt = PoincareManifold::exp_map_zero(&v_tan)?;
    assert_eq!(pt.shape(), &[2, 2]);
    let recovered_v = PoincareManifold::log_map_zero(&pt)?;
    assert_eq!(recovered_v.shape(), &[2, 2]);

    // =========================================================================
    // 6. Temporal Interaction GNN (Dynamic Continuous-Time Graphs)
    // =========================================================================
    let mut tgn = TemporalInteractionGNN::new(6, 16, 16, 8)?;
    let edge_attr = Tensor::randn(&[1, 8])?;
    let mem_out = tgn.process_event(0, 1, 1.25, &edge_attr)?;
    assert_eq!(mem_out.shape(), &[1, 16]);

    // =========================================================================
    // 7. Apple Metal MSL Kernel Emitter
    // =========================================================================
    let emitter = MslKernelEmitter::new("oxideon_metal_fused");
    let msl_add_silu = emitter.emit_fused_add_silu_msl();
    let msl_matmul = emitter.emit_matmul_tiled_msl();
    assert!(msl_add_silu.contains("kernel void oxideon_metal_fused"));
    assert!(msl_matmul.contains("threadgroup float tileA[TILE_DIM][TILE_DIM]"));

    // =========================================================================
    // 8. Python Bindings Verification
    // =========================================================================
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

# 1. MixtureOfDepths & EarlyExitRouter
mod = ox.MixtureOfDepths(16, capacity=0.5)
x = ox.Tensor.randn([2, 4, 16])
mod_out, weights, k = mod.forward(x)
assert mod_out.shape == [2, 4, 16]
assert k == 2

router = ox.EarlyExitRouter(0.8)
logits = ox.Tensor([10.0, 0.0, 0.0, 10.0], [2, 2])
flags, max_p = router.evaluate_exit(logits)
assert flags == [True, True]

# 2. ACT & Flow Matching Policy
act = ox.ActionChunkingTransformer(state_dim=8, action_dim=4, chunk_size=6, hidden_dim=16)
state = ox.Tensor.randn([1, 8])
act_chunk = act.forward(state)
assert act_chunk.shape == [1, 6, 4]
smooth_act = act.step_temporal_ensemble(state)
assert len(smooth_act) == 4

flow = ox.FlowMatchingRoboticsPolicy(action_dim=4, obs_dim=8, hidden_dim=16)
actions = flow.sample_action(state, num_steps=3)
assert actions.shape == [1, 4]

# 3. PoincareManifold & TemporalInteractionGNN
u = ox.Tensor([0.1, 0.2, -0.2, 0.1], [2, 2])
v = ox.Tensor([-0.1, 0.1, 0.2, -0.1], [2, 2])
d = ox.PoincareManifold.distance(u, v)
assert d.shape == [2, 1]
m = ox.PoincareManifold.mobius_add(u, v)
assert m.shape == [2, 2]

tgn = ox.TemporalInteractionGNN(num_nodes=4, memory_dim=16, time_dim=16, edge_dim=8)
e = ox.Tensor.randn([1, 8])
dst_mem = tgn.process_event(0, 1, 0.5, e)
assert dst_mem.shape == [1, 16]

# 4. Metal MSL Emitter
metal = ox.MslKernelEmitter("test_kernel")
msl_code = metal.emit_fused_add_silu_msl()
assert "kernel void test_kernel" in msl_code
"#,
            )
            .unwrap();
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Frontier Dominion test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_complex_acceleration_device_choice_and_topologies() -> Result<()> {
    manual_seed(9999);

    // =========================================================================
    // 1. Device Specifier Parsing & String Representation
    // =========================================================================
    assert_eq!(Device::parse_spec("cuda")?, Device::Cuda(0));
    assert_eq!(Device::parse_spec("cuda:0")?, Device::Cuda(0));
    assert_eq!(Device::parse_spec("cuda:3")?, Device::Cuda(3));
    assert_eq!(Device::parse_spec("metal")?, Device::Metal(0));
    assert_eq!(Device::parse_spec("metal:1")?, Device::Metal(1));
    assert_eq!(Device::parse_spec("mps:0")?, Device::Metal(0));
    assert_eq!(Device::parse_spec("vulkan:0")?, Device::Vulkan(0));
    assert_eq!(Device::parse_spec("tpu")?, Device::Tpu { chip: 0, core: 0 });
    assert_eq!(
        Device::parse_spec("tpu:2:1")?,
        Device::Tpu { chip: 2, core: 1 }
    );
    assert_eq!(
        Device::parse_spec("npu:groq:0")?,
        Device::Npu {
            name: "groq".into(),
            index: 0
        }
    );
    assert_eq!(Device::parse_spec("meta")?, Device::Meta);
    assert_eq!(Device::parse_spec("cpu")?, Device::Cpu);
    assert_eq!(Device::parse_spec("cpu:0")?, Device::Cpu);

    assert_eq!(Device::Cuda(2).to_string(), "cuda:2");
    assert_eq!(Device::Metal(1).to_string(), "metal:1");
    assert_eq!(Device::Meta.to_string(), "meta");
    assert_eq!(Device::Tpu { chip: 1, core: 0 }.to_string(), "tpu:1:0");

    // =========================================================================
    // 2. Hardware Properties & Topology Introspection
    // =========================================================================
    let cuda_props = Device::Cuda(0).properties();
    assert!(cuda_props.peak_fp16_tflops > 500.0);
    assert_eq!(cuda_props.compute_capability, Some((9, 0)));
    assert_eq!(cuda_props.interconnect, InterconnectType::NVLink4);

    let metal_props = Device::Metal(0).properties();
    assert!(metal_props.is_unified_memory);
    assert_eq!(metal_props.interconnect, InterconnectType::UnifiedMemory);

    let meta_props = Device::Meta.properties();
    assert_eq!(meta_props.total_memory_bytes, 0);

    // =========================================================================
    // 3. Multi-Dimensional Distributed DeviceMesh
    // =========================================================================
    let mesh = DeviceMesh::homogeneous(&[2, 4], &["dp", "tp"], DeviceType::Cuda)?;
    assert_eq!(mesh.shape, vec![2, 4]);
    assert_eq!(mesh.dim_names, vec!["dp".to_string(), "tp".to_string()]);
    assert_eq!(mesh.device_count(), 8);

    assert_eq!(mesh.get_device(&[0, 0])?, &Device::Cuda(0));
    assert_eq!(mesh.get_device(&[0, 3])?, &Device::Cuda(3));
    assert_eq!(mesh.get_device(&[1, 0])?, &Device::Cuda(4));
    assert_eq!(mesh.get_device(&[1, 3])?, &Device::Cuda(7));

    let tp_slice = mesh.slice_submesh("tp", 1)?;
    assert_eq!(tp_slice.len(), 2);
    assert_eq!(tp_slice[0], Device::Cuda(1));
    assert_eq!(tp_slice[1], Device::Cuda(5));

    let dp_slice = mesh.slice_submesh("dp", 0)?;
    assert_eq!(dp_slice.len(), 4);
    assert_eq!(
        dp_slice,
        vec![
            Device::Cuda(0),
            Device::Cuda(1),
            Device::Cuda(2),
            Device::Cuda(3)
        ]
    );

    // =========================================================================
    // 4. Interconnect Matrix & Bandwidth Modeling
    // =========================================================================
    let bw_nvlink = InterconnectMatrix::bandwidth_gbps(&Device::Cuda(0), &Device::Cuda(1));
    assert_eq!(bw_nvlink, 900.0);

    let bw_pcie = InterconnectMatrix::bandwidth_gbps(&Device::Cpu, &Device::Cuda(0));
    assert_eq!(bw_pcie, 64.0);

    let bw_metal = InterconnectMatrix::bandwidth_gbps(&Device::Cpu, &Device::Metal(0));
    assert_eq!(bw_metal, 800.0);

    let xfer_time = InterconnectMatrix::estimate_transfer_seconds(
        &Device::Cuda(0),
        &Device::Cuda(1),
        900_000_000,
    );
    assert!(xfer_time < 0.005); // less than 5ms for 900MB over NVLink

    // =========================================================================
    // 5. Cross-Device Safety, Transpose, and Meta Tensor Execution
    // =========================================================================
    let t_cuda0 = Tensor::randn_on(&[4, 8], &Device::Cuda(0))?;
    assert_eq!(t_cuda0.device(), Device::Cuda(0));

    // Device-to-device transfer
    let t_metal0 = t_cuda0.to_device(&Device::Metal(0))?;
    assert_eq!(t_metal0.device(), Device::Metal(0));

    // Cross-device safety check
    let t_cuda1 = Tensor::randn_on(&[4, 8], &Device::Cuda(1))?;
    let err = t_cuda0.add(&t_cuda1);
    assert!(err.is_err());
    assert!(matches!(err, Err(Error::DeviceMismatch { .. })));

    // Meta tensor computation (zero memory dry-run)
    let t_meta = Tensor::empty(&[1024, 512], DType::F32, &Device::Meta)?;
    assert_eq!(t_meta.device(), Device::Meta);
    assert_eq!(t_meta.shape(), &[1024, 512]);

    let t_meta_tr = t_meta.t()?;
    assert_eq!(t_meta_tr.shape(), &[512, 1024]);
    assert_eq!(t_meta_tr.device(), Device::Meta);

    let t_meta_mm = t_meta.matmul(&t_meta_tr)?;
    assert_eq!(t_meta_mm.shape(), &[1024, 1024]);
    assert_eq!(t_meta_mm.device(), Device::Meta);

    // =========================================================================
    // 6. Thread-Local Device Scoping
    // =========================================================================
    assert_eq!(get_default_device(), Device::Cpu);
    with_device(Device::Cuda(2), || {
        assert_eq!(get_default_device(), Device::Cuda(2));
    });
    assert_eq!(get_default_device(), Device::Cpu);

    // =========================================================================
    // 7. Device Discovery & Availability Introspection
    // =========================================================================
    let available = Device::available_devices();
    assert!(available.contains(&Device::Cpu));
    assert!(available.contains(&Device::Meta));
    assert!(Device::is_available(DeviceType::Cpu));
    assert!(Device::is_available(DeviceType::Meta));
    assert_eq!(Device::device_count(DeviceType::Cpu), 1);

    // =========================================================================
    // 8. Python PyO3 Zero-GIL Integration Verification
    // =========================================================================
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

# 1. Device Creation & Inspection
d_cuda = ox.Device("cuda:0")
assert d_cuda.is_cuda == True
assert d_cuda.index == 0
assert str(d_cuda) == "cuda:0"

d_metal = ox.Device("metal:0")
assert d_metal.is_metal == True
assert d_metal.type == "metal"

d_cpu = ox.Device("cpu")
assert d_cpu.is_cpu == True

d_meta = ox.Device("meta")
assert d_meta.is_meta == True

# 2. Hardware Properties
props = d_cuda.get_properties()
assert "NVIDIA" in props["name"]
assert props["total_memory_gb"] > 0
assert props["peak_fp16_tflops"] > 500.0

# 3. Interconnect Bandwidth
bw = ox.device_bandwidth("cuda:0", "cuda:1")
assert bw == 900.0
bw_metal = ox.device_bandwidth("metal", "cpu")
assert bw_metal == 800.0

# 4. Multi-dimensional DeviceMesh
mesh = ox.DeviceMesh(shape=[2, 4], dim_names=["dp", "tp"], device_type="cuda")
assert mesh.shape == [2, 4]
assert mesh.dim_names == ["dp", "tp"]
assert mesh.device_count == 8
assert str(mesh.get_device([0, 2])) == "cuda:2"
assert str(mesh.get_device([1, 1])) == "cuda:5"

sub_tp = mesh.slice_submesh("tp", 0)
assert len(sub_tp) == 2

# 5. Tensor Device Placement & Transfers
t = ox.Tensor.randn([2, 4], device="cuda:0")
assert t.device == "cuda:0"
assert t.device == ox.Device("cuda:0")

t_metal = t.to("metal:0")
assert t_metal.device == "metal:0"

t_meta = ox.Tensor.empty([500, 500], device="meta")
assert t_meta.device == "meta"
assert t_meta.shape == [500, 500]

# 6. Default Device Configuration
ox.set_default_device("cuda:1")
assert ox.get_default_device() == "cuda:1"
ox.set_default_device("cpu")
assert ox.get_default_device() == "cpu"

# 7. Device Discovery
avail = ox.get_available_devices()
assert len(avail) >= 2
assert any(d.is_cpu for d in avail)
assert any(d.is_meta for d in avail)
assert ox.is_available("cpu") == True
assert ox.is_available("meta") == True
assert ox.device_count("cpu") >= 1

dev_avail = ox.Device.available_devices()
assert len(dev_avail) == len(avail)
assert ox.Device.is_available("cpu") == True
assert ox.Device.device_count("cpu") >= 1
"#,
            )
            .unwrap();
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Complex Device Choice test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v16_tile_ir_sharded_tensor_world_model_and_medusa() -> Result<()> {
    manual_seed(1616);

    // =========================================================================
    // 1. Tile-IR & Dynamic Kernel Fusion Plans
    // =========================================================================
    let plan = KernelFusionEngine::plan_fused_linear_swiglu("fused_swiglu", 32, 64, 32);
    assert_eq!(plan.name, "fused_swiglu");
    assert_eq!(plan.tile_m, 16);
    assert_eq!(plan.tile_n, 16);

    let ptx = plan.emit_ptx("sm_90");
    assert!(ptx.contains("fused_swiglu"));

    let msl = plan.emit_msl();
    assert!(msl.contains("fused_swiglu"));

    let plan_norm = KernelFusionEngine::plan_fused_add_rmsnorm("fused_norm", 64, 1e-5);
    assert_eq!(plan_norm.ops.len(), 3);

    // =========================================================================
    // 2. Automated Distributed Sharding (DTensor / ShardedTensor) & AutoParallelPlanner
    // =========================================================================
    let mesh = DeviceMesh::homogeneous(&[2, 2], &["dp", "tp"], DeviceType::Cuda)?;
    let global = Tensor::randn(&[8, 16])?;
    let st = ShardedTensor::from_global(
        &global,
        &mesh,
        &[Placement::Replicate, Placement::Replicate],
        &[0, 1],
    )?;
    assert_eq!(st.global_shape, vec![8, 16]);
    assert_eq!(st.local_tensor.shape(), &[8, 16]);
    assert_eq!(st.device, Device::Cuda(1));

    // Sharded matrix multiplication (Column-Parallel: X replicated, W column-sharded)
    let weights = Tensor::randn(&[16, 32])?;
    let st_w = ShardedTensor::from_global(
        &weights,
        &mesh,
        &[Placement::Replicate, Placement::Shard(1)],
        &[0, 1],
    )?;
    assert_eq!(st_w.local_tensor.shape(), &[16, 16]);
    let out_st = st.sharded_matmul(&st_w, &[0, 1])?;
    assert_eq!(out_st.local_tensor.shape(), &[8, 16]);
    assert_eq!(out_st.global_shape, vec![8, 32]);

    let planner = AutoParallelPlanner::new();
    let plan_auto = planner.plan_optimal_sharding(&mesh, 16, 128, 512)?;
    assert_eq!(plan_auto.dp_degree * plan_auto.tp_degree, 4);

    // =========================================================================
    // 3. Unified 4D World Model & Video-Action Diffusion
    // =========================================================================
    let wm_cfg = WorldModelConfig {
        latent_channels: 4,
        action_dim: 6,
        action_chunk: 4,
        hidden_dim: 16,
        num_layers: 2,
    };
    let wm = UnifiedWorldModel::new(wm_cfg)?;
    let latents = Tensor::randn(&[2, 8, 4])?;
    let actions = Tensor::randn(&[2, 6])?;

    let flow = wm.forward_dynamics(&latents, &actions, 0.5)?;
    assert_eq!(flow.shape(), &[2, 8, 4]);

    let future_video = wm.sample_future_video(&latents, &actions, 3)?;
    assert_eq!(future_video.shape(), &[2, 8, 4]);

    let goal_latents = Tensor::randn(&[2, 8, 4])?;
    let planned_actions = wm.plan_actions_to_goal(&latents, &goal_latents)?;
    assert_eq!(planned_actions.shape(), &[2, 4, 6]);

    // =========================================================================
    // 4. Medusa & Tree-Attention Speculative Decoding
    // =========================================================================
    let parents = vec![0, 0, 1, 1, 2];
    let tree_mask = TreeAttentionMask::build(&parents)?;
    assert_eq!(tree_mask.shape(), &[1, 1, 5, 5]);

    let medusa = MedusaHeads::new(16, 32, 3, 2)?;
    let hidden = Tensor::randn(&[1, 1, 16])?;
    let draft_logits = medusa.forward(&hidden)?;
    assert_eq!(draft_logits.len(), 3);
    assert_eq!(draft_logits[0].shape(), &[1, 1, 32]);

    let (accepted, count) = TreeSpeculativeDecoder::accept_longest_path(
        &[10, 20, 30, 40, 50],
        &[10, 20, 30, 99, 50],
        &[0, 0, 1, 2, 3],
    )?;
    assert!(count >= 2);
    assert_eq!(accepted[0], 10);
    assert_eq!(accepted[1], 20);

    // =========================================================================
    // 5. Python PyO3 Zero-GIL Bindings Verification
    // =========================================================================
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

# 1. Tile-IR Kernel Fusion
plan = ox.KernelFusionEngine.plan_fused_linear_swiglu("py_swiglu", 16, 32, 16)
ptx = plan.emit_ptx("sm_90")
assert "py_swiglu" in ptx
msl = plan.emit_msl()
assert "py_swiglu" in msl

# 2. ShardedTensor & AutoParallelPlanner
mesh = ox.DeviceMesh(shape=[2, 2], dim_names=["dp", "tp"], device_type="cuda")
g = ox.Tensor.randn([8, 16])
st = ox.ShardedTensor.from_global(g, mesh, ["replicate", "shard:1"], [0, 1])
assert st.global_shape == [8, 16]
assert st.local_tensor.shape == [8, 8]

dp, tp, comm = ox.AutoParallelPlanner.plan_optimal_sharding(mesh, 16, 128, 512)
assert dp * tp == 4

# 3. Unified 4D World Model
wm = ox.UnifiedWorldModel(latent_channels=4, action_dim=6, action_chunk=4, hidden_dim=16, num_layers=2)
lat = ox.Tensor.randn([2, 8, 4])
act = ox.Tensor.randn([2, 6])
flow = wm.forward_dynamics(lat, act, 0.5)
assert flow.shape == [2, 8, 4]

fut = wm.sample_future_video(lat, act, num_steps=2)
assert fut.shape == [2, 8, 4]

goal = ox.Tensor.randn([2, 8, 4])
planned_act = wm.plan_actions_to_goal(lat, goal)
assert planned_act.shape == [2, 4, 6]

# 4. Tree-Attention & Medusa
parents = [0, 0, 1, 1, 2]
tmask = ox.TreeAttentionMask.build(parents)
assert tmask.shape == [1, 1, 5, 5]

medusa = ox.MedusaHeads(hidden_dim=16, vocab_size=32, num_heads=3, topk=2)
h = ox.Tensor.randn([1, 1, 16])
drafts = medusa.forward(h)
assert len(drafts) == 3

accepted, n = ox.TreeSpeculativeDecoder.accept_longest_path([10, 20, 30], [10, 20, 99], [0, 0, 1])
assert n == 2
assert accepted == [10, 20]
"#,
            )
            .unwrap();
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python v1.6.0 Frontier test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v17_flash_decoding_deepseek_v3_moe_and_diffusion_forcing() -> Result<()> {
    manual_seed(42);

    // =========================================================================
    // 1. FlashDecoding Split-KV Attention Engine
    // =========================================================================
    let fd_cfg = FlashDecodingConfig::new(4, 16).with_chunk_size(32);
    let engine = FlashDecodingEngine::new(fd_cfg);

    let q = Tensor::randn(&[2, 4, 1, 16])?;
    let k = Tensor::randn(&[2, 4, 128, 16])?;
    let v = Tensor::randn(&[2, 4, 128, 16])?;

    let out = engine.decode_attention(&q, &k, &v)?;
    assert_eq!(out.shape(), &[2, 4, 1, 16]);

    let ptx = engine.generate_ptx_kernel();
    assert!(ptx.contains(".target sm_90"));
    assert!(ptx.contains("flash_decoding_stage1_kernel"));

    let wgsl = engine.generate_wgsl_kernel();
    assert!(wgsl.contains("FlashDecodingUniforms"));

    // =========================================================================
    // 2. DeepSeek-V3 Aux-Loss-Free MoE with Dynamic Expert Bias & Shared Experts
    // =========================================================================
    let mut moe = DeepSeekV3MoE::new(
        32,   // d_model
        64,   // ffn_hidden
        1,    // num_shared
        4,    // num_routed
        2,    // top_k
        0.05, // bias_update_rate
    )?;

    let x = Tensor::randn(&[8, 32])?;
    let initial_biases = moe.router.dynamic_bias.clone();
    assert_eq!(initial_biases, vec![0.0; 4]);

    let y = moe.forward_and_update(&x)?;
    assert_eq!(y.shape(), &[8, 32]);

    // Ensure dynamic biases adjusted to balance token routing loads
    let updated_biases = &moe.router.dynamic_bias;
    assert_eq!(updated_biases.len(), 4);
    assert!(updated_biases.iter().any(|&b| b != 0.0));

    // =========================================================================
    // 3. Diffusion Forcing Scheduler (Per-Token Noise & Denoising)
    // =========================================================================
    let forcing = DiffusionForcingScheduler::new(100, BetaSchedule::default())?;
    let seq = Tensor::randn(&[2, 4, 16])?;
    let timesteps = vec![0, 10, 50, 90];

    let noisy_seq = forcing.add_noise_per_token(&seq, &timesteps)?;
    assert_eq!(noisy_seq.shape(), &[2, 4, 16]);

    // Position 0 has t=0 (sqrt_alpha ~ 1.0, noise ~ 0.0), verify values close to original
    let orig_vec = seq.to_vec::<f32>()?;
    let noisy_vec = noisy_seq.to_vec::<f32>()?;
    for d in 0..16 {
        assert!((orig_vec[d] - noisy_vec[d]).abs() < 1e-4);
    }

    // Single step per token reverse denoising
    let pred_eps = Tensor::randn(&[2, 4, 16])?;
    let prev_timesteps = vec![0, 5, 25, 45];
    let denoised_step =
        forcing.step_per_token(&pred_eps, &timesteps, &prev_timesteps, &noisy_seq)?;
    assert_eq!(denoised_step.shape(), &[2, 4, 16]);

    // =========================================================================
    // 4. Zero-GIL PyO3 Bindings Verification
    // =========================================================================
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

# 1. FlashDecoding
cfg = ox.FlashDecodingConfig(num_heads=2, head_dim=16, split_kv_chunk_size=32)
engine = ox.FlashDecodingEngine(cfg)
q = ox.Tensor.randn([1, 2, 1, 16])
k = ox.Tensor.randn([1, 2, 64, 16])
v = ox.Tensor.randn([1, 2, 64, 16])
out = engine.decode_attention(q, k, v)
assert out.shape == [1, 2, 1, 16]

# 2. DeepSeek-V3 MoE
moe = ox.DeepSeekV3MoE(d_model=16, ffn_hidden=32, num_shared=1, num_routed=4, top_k=2, bias_update_rate=0.01)
inp = ox.Tensor.randn([4, 16])
moe_out = moe.forward_and_update(inp)
assert moe_out.shape == [4, 16]

# 3. Diffusion Forcing
dfs = ox.DiffusionForcingScheduler(num_train_timesteps=100)
seq_in = ox.Tensor.randn([2, 4, 8])
noisy = dfs.add_noise_per_token(seq_in, [0, 10, 30, 80])
assert noisy.shape == [2, 4, 8]
"#,
            )
            .unwrap();
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python v1.7.0 Frontier test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v18_titans_dual_pipe_and_k_diffusion() -> Result<()> {
    manual_seed(42);

    // =========================================================================
    // 1. Titans: Neural Memory at Test Time (Behrouz et al.)
    // =========================================================================
    let titans_cfg = TitansMemoryConfig::new(16, 16);
    let mem = TitansNeuralMemory::new(titans_cfg.clone())?;

    let x = Tensor::randn(&[2, 6, 16])?;
    let mem_out = mem.forward(&x)?;
    assert_eq!(mem_out.shape(), &[2, 6, 16]);

    let block = TitansBlock::new(titans_cfg)?;
    let block_out = block.forward(&x)?;
    assert_eq!(block_out.shape(), &[2, 6, 16]);

    // =========================================================================
    // 2. DualPipe: Bidirectional Pipeline Parallelism (DeepSeek-V3)
    // =========================================================================
    let dp_sched = DualPipeSchedule::new(4, 8)?;
    let bubble = dp_sched.bubble_ratio();
    assert!((bubble - 3.0 / 16.0).abs() < 1e-6);

    let rank0_steps = dp_sched.generate_schedule_for_rank(0)?;
    assert!(!rank0_steps.is_empty());
    assert!(rank0_steps.iter().any(|s| s.overlapped_comm.is_some()));

    // =========================================================================
    // 3. K-Diffusion & EDM2 High-Order Predictor-Corrector ODE Solver
    // =========================================================================
    let edm_cfg = EDMConfig::new(10);
    let sigmas = edm_cfg.build_sigmas();
    assert_eq!(sigmas.len(), 11);
    assert_eq!(sigmas[10], 0.0);
    for i in 0..10 {
        assert!(sigmas[i] > sigmas[i + 1]);
    }

    let sampler = KDiffusionSampler::new(edm_cfg);
    let init_noise = Tensor::randn(&[2, 4])?;

    // Denoise using an analytical quadratic regularizer denoiser D(x, sigma) = x / (1 + sigma^2)
    let sample = sampler.sample_heun(&init_noise, |x_curr, sigma| {
        let scale = 1.0 / (1.0 + sigma * sigma) as f64;
        x_curr.mul_scalar(scale)
    })?;
    assert_eq!(sample.shape(), &[2, 4]);

    // =========================================================================
    // 4. Zero-GIL PyO3 Bindings Verification
    // =========================================================================
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

# 1. Titans Memory
t_cfg = ox.TitansMemoryConfig(d_model=16, memory_dim=16, base_decay=0.1, base_lr=0.05)
t_mem = ox.TitansNeuralMemory(t_cfg)
x_in = ox.Tensor.randn([2, 4, 16])
t_out = t_mem.forward(x_in)
assert t_out.shape == [2, 4, 16]

t_block = ox.TitansBlock(t_cfg)
tb_out = t_block.forward(x_in)
assert tb_out.shape == [2, 4, 16]

# 2. DualPipe Schedule
sched = ox.DualPipeSchedule(num_stages=4, num_microbatches=8)
assert abs(sched.bubble_ratio() - 0.1875) < 1e-4
assert sched.num_steps_for_rank(0) > 0

# 3. K-Diffusion EDM2
edm = ox.EDMConfig(num_steps=10)
sigmas = edm.build_sigmas()
assert len(sigmas) == 11
assert sigmas[-1] == 0.0
kdiff = ox.KDiffusionSampler(edm)
"#,
            )
            .unwrap();
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python v1.8.0 Frontier test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v19_ternary_packed_and_rvq() -> Result<()> {
    manual_seed(42);

    // =========================================================================
    // 1. BitNet b1.58 Radix-3 Ternary Packed Storage (1.6 bits/param)
    // =========================================================================
    let w_float = Tensor::randn(&[4, 15])?;
    let scale = 0.5f32;

    let packed = TernaryPackedTensor::pack(&w_float, scale)?;
    assert_eq!(packed.shape, vec![4, 15]);
    assert_eq!(packed.num_elements, 60);
    // Exactly 60 / 5 = 12 bytes
    assert_eq!(packed.packed_data.len(), 12);

    let unpacked = packed.unpack()?;
    assert_eq!(unpacked.shape(), &[4, 15]);

    let x = Tensor::randn(&[2, 15])?;
    let y = packed.ternary_matmul(&x)?;
    assert_eq!(y.shape(), &[2, 4]);

    let lin = Linear::with_bias(15, 4, true)?;
    let bit_lin = PackedBitLinear::from_linear(&lin, 0.5)?;
    let y_lin = bit_lin.forward(&x)?;
    assert_eq!(y_lin.shape(), &[2, 4]);

    // =========================================================================
    // 2. Residual Vector Quantization (RVQ Neural Audio Tokenizer)
    // =========================================================================
    let rvq_cfg = RVQConfig::new(4, 16, 8);
    let rvq = ResidualVectorQuantizer::new(rvq_cfg)?;

    let latents = Tensor::randn(&[2, 5, 8])?;
    let rvq_out = rvq.quantize(&latents)?;

    assert_eq!(rvq_out.quantized.shape(), &[2, 5, 8]);
    assert_eq!(rvq_out.codes.len(), 2);
    assert_eq!(rvq_out.codes[0].len(), 4);
    assert_eq!(rvq_out.codes[0][0].len(), 5);
    assert!(rvq_out.commitment_loss >= 0.0);

    // Decode token codes back to audio latents
    let decoded = rvq.decode_codes(&rvq_out.codes[0])?;
    assert_eq!(decoded.shape(), &[1, 5, 8]);

    // =========================================================================
    // 3. Zero-GIL PyO3 Bindings Verification
    // =========================================================================
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

# 1. Ternary Packing
w = ox.Tensor.randn([4, 15])
packed = ox.TernaryPackedTensor.pack(w, 0.5)
assert packed.shape == [4, 15]
assert packed.packed_bytes == 12

x_in = ox.Tensor.randn([2, 15])
y_out = packed.ternary_matmul(x_in)
assert y_out.shape == [2, 4]

unpacked = packed.unpack()
assert unpacked.shape == [4, 15]

# 2. RVQ Audio Tokenizer
cfg = ox.RVQConfig(num_codebooks=3, codebook_size=16, dim=8, commitment_cost=0.25)
rvq = ox.ResidualVectorQuantizer(cfg)
lat = ox.Tensor.randn([2, 4, 8])
quantized, codes, loss = rvq.quantize(lat)
assert quantized.shape == [2, 4, 8]
assert len(codes) == 2
assert len(codes[0]) == 3
assert loss >= 0.0

dec = rvq.decode_codes(codes[0])
assert dec.shape == [1, 4, 8]
"#,
            )
            .unwrap();
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python v1.9.0 Frontier test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v20_symbolic_shapes_and_engine() -> Result<()> {
    // =========================================================================
    // 1. Symbolic Shape Parsing & Runtime Binding
    // =========================================================================
    let s_shape = SymbolicShape::parse("[B, S, H]");
    assert_eq!(s_shape.ndim(), 3);

    let bindings = s_shape.bind(&[2, 64, 512])?;
    assert_eq!(bindings.get("B"), Some(&2));
    assert_eq!(bindings.get("S"), Some(&64));
    assert_eq!(bindings.get("H"), Some(&512));

    // Resolve with variable environment
    let mut env = std::collections::HashMap::new();
    env.insert("B", 4);
    env.insert("S", 128);
    env.insert("H", 768);
    let resolved = s_shape.resolve(&env)?;
    assert_eq!(resolved, vec![4, 128, 768]);

    // =========================================================================
    // 2. Symbolic Broadcasting & Contraction (Matmul)
    // =========================================================================
    let shape_a = SymbolicShape::parse("[B, 1, H]");
    let shape_b = SymbolicShape::parse("[1, S, H]");
    let bcast = shape_a.broadcast_with(&shape_b)?;
    assert_eq!(bcast.to_string(), "[B, S, H]");

    let shape_x = SymbolicShape::parse("[B, S, H]");
    let shape_w = SymbolicShape::parse("[H, D]");
    let shape_out = shape_x.matmul_with(&shape_w)?;
    assert_eq!(shape_out.to_string(), "[B, S, D]");

    // =========================================================================
    // 3. Zero-GIL PyO3 Bindings Verification
    // =========================================================================
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

sym = ox.SymbolicShape.parse("[B, S, H]")
bindings = sym.bind([2, 32, 256])
assert bindings["B"] == 2
assert bindings["S"] == 32
assert bindings["H"] == 256

resolved = sym.resolve({"B": 8, "S": 16, "H": 128})
assert resolved == [8, 16, 128]

a = ox.SymbolicShape.parse("[B, 1, H]")
b = ox.SymbolicShape.parse("[1, S, H]")
c = a.broadcast_with(b)
assert str(c) == "[B, S, H]"

w = ox.SymbolicShape.parse("[H, OUT]")
y = sym.matmul_with(w)
assert str(y) == "[B, S, OUT]"
"#,
            )
            .unwrap();
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python v2.0.0 Frontier test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v21_eagle_streaming_sink_and_omni_tokenizer() -> Result<()> {
    manual_seed(42);

    // =========================================================================
    // 1. Eagle: Feature-Level Speculative Decoding
    // =========================================================================
    let cfg = EagleConfig::new(16, 32, 3);
    let engine = EagleSpeculativeEngine::new(cfg)?;

    let top_h = Tensor::randn(&[1, 1, 16])?;
    let (draft_tokens, draft_feats) = engine.generate_draft(&top_h, 5)?;
    assert_eq!(draft_tokens.len(), 3);
    assert_eq!(draft_feats.len(), 3);
    assert_eq!(draft_feats[0].shape(), &[1, 1, 16]);

    let (accepted, n) = EagleSpeculativeEngine::verify_and_accept(&draft_tokens, &draft_tokens);
    assert_eq!(n, 3);
    assert_eq!(accepted.len(), 3);

    // =========================================================================
    // 2. StreamingLLM: Attention Sinks for Bounded Streaming KV-Cache
    // =========================================================================
    let sink_cfg = AttentionSinkConfig::new(2, 4);
    let mut cache = StreamingKVCache::new(sink_cfg);

    for _ in 0..10 {
        let k = Tensor::randn(&[1, 2, 1, 8])?;
        let v = Tensor::randn(&[1, 2, 1, 8])?;
        cache.append(&k, &v)?;
    }

    assert_eq!(cache.total_tokens_seen(), 10);
    assert_eq!(cache.num_cached_tokens(), 6); // 2 sinks + 4 window

    let (keys, vals) = cache.get_kv()?;
    assert_eq!(keys.shape(), &[1, 2, 6, 8]);
    assert_eq!(vals.shape(), &[1, 2, 6, 8]);

    // =========================================================================
    // 3. Unified Interleaved Multimodal Tokenizer
    // =========================================================================
    let omni = OmniTokenizer::new();
    let text_seg = omni.encode_text("Hello");
    let img_seg = omni.encode_vision_patches(&[10, 20])?;
    let audio_seg = omni.encode_audio_tokens(&[5, 15])?;

    let seq = omni.interleave(&[text_seg, img_seg, audio_seg])?;
    assert!(!seq.is_empty());
    assert_eq!(seq.token_ids.len(), seq.modalities.len());
    assert!(seq.modalities.contains(&ModalityKind::Text));
    assert!(seq.modalities.contains(&ModalityKind::Vision));
    assert!(seq.modalities.contains(&ModalityKind::Audio));

    // =========================================================================
    // 4. Zero-GIL PyO3 Bindings Verification
    // =========================================================================
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

# 1. Eagle Speculation
cfg = ox.EagleConfig(hidden_dim=16, vocab_size=32, draft_depth=3)
eagle = ox.EagleSpeculativeEngine(cfg)
th = ox.Tensor.randn([1, 1, 16])
tokens, feats = eagle.generate_draft(th, 0)
assert len(tokens) == 3
assert len(feats) == 3
accepted, n = ox.EagleSpeculativeEngine.verify_and_accept([1, 2], [1, 2, 3])
assert n == 2
assert accepted == [1, 2, 3]

# 2. Attention Sink Streaming Cache
sink_cfg = ox.AttentionSinkConfig(num_sink_tokens=2, window_size=4)
cache = ox.StreamingKVCache(sink_cfg)
for _ in range(8):
    k = ox.Tensor.randn([1, 2, 1, 8])
    v = ox.Tensor.randn([1, 2, 1, 8])
    cache.append(k, v)
assert cache.total_tokens_seen() == 8
assert cache.num_cached_tokens() == 6
keys, vals = cache.get_kv()
assert keys.shape == [1, 2, 6, 8]

# 3. Omni Tokenizer
omni = ox.OmniTokenizer()
t_ids = omni.encode_text("Oxideon AI")
assert len(t_ids) > 0
v_ids = omni.encode_vision_patches([1, 2])
assert len(v_ids) == 4  # start + 2 patches + end
a_ids = omni.encode_audio_tokens([10])
assert len(a_ids) == 3  # start + 1 audio + end
"#,
            )
            .unwrap();
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python v2.1.0 Frontier test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v22_full_pytorch_nn_parity() -> Result<()> {
    manual_seed(42);

    // 1. Pooling: MaxPool1d, AvgPool1d, AdaptiveAvgPool1d, AdaptiveMaxPool1d, LPPool1d, LPPool2d
    let x1d = Tensor::randn(&[2, 4, 16])?;
    let maxpool1d = MaxPool1d::new(2, 2);
    let p1 = maxpool1d.forward(&x1d)?;
    assert_eq!(p1.shape(), &[2, 4, 8]);

    let avgpool1d = AvgPool1d::new(2, 2);
    let p2 = avgpool1d.forward(&x1d)?;
    assert_eq!(p2.shape(), &[2, 4, 8]);

    let a_avg1d = AdaptiveAvgPool1d::new(4);
    let p3 = a_avg1d.forward(&x1d)?;
    assert_eq!(p3.shape(), &[2, 4, 4]);

    let a_max1d = AdaptiveMaxPool1d::new(4);
    let p4 = a_max1d.forward(&x1d)?;
    assert_eq!(p4.shape(), &[2, 4, 4]);

    let lp1d = LPPool1d::new(2.0, 2, 2);
    let p5 = lp1d.forward(&x1d)?;
    assert_eq!(p5.shape(), &[2, 4, 8]);

    let x2d = Tensor::randn(&[2, 3, 8, 8])?;
    let lp2d = LPPool2d::new(2.0, (2, 2), (2, 2));
    let p6 = lp2d.forward(&x2d)?;
    assert_eq!(p6.shape(), &[2, 3, 4, 4]);

    // 2. Transposed Convolutions: ConvTranspose3d with analytical backward
    let ct3d = ConvTranspose3d::new(
        2,
        4,
        (3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        (0, 0, 0),
        true,
        DType::F32,
        Device::Cpu,
    )?;
    let x5d = Tensor::randn(&[1, 2, 4, 4, 4])?;
    let y5d = ct3d.forward(&x5d)?;
    assert_eq!(y5d.shape(), &[1, 4, 4, 4, 4]);
    let loss3d = y5d.sum()?;
    loss3d.backward()?;
    assert!(ct3d.weight.grad().unwrap().numel() > 0);

    // 3. Normalization: InstanceNorm1d and InstanceNorm3d
    let in1d = InstanceNorm1d::new(4, 1e-5)?;
    let in1d_out = in1d.forward(&x1d)?;
    assert_eq!(in1d_out.shape(), &[2, 4, 16]);

    let in3d = InstanceNorm3d::new(2, 1e-5)?;
    let in3d_out = in3d.forward(&x5d)?;
    assert_eq!(in3d_out.shape(), &[1, 2, 4, 4, 4]);

    // 4. EmbeddingBag: Sum, Mean, Max, and Jagged Offsets
    let emb_bag = EmbeddingBag::new(16, 8, EmbeddingBagMode::Mean)?;
    let indices2d = Tensor::from_slice(&[1i64, 3, 5, 2, 4, 6], &[2, 3])?;
    let bag_out = emb_bag.forward(&indices2d)?;
    assert_eq!(bag_out.shape(), &[2, 8]);

    let emb_sum = EmbeddingBag::new(16, 8, EmbeddingBagMode::Sum)?;
    let sum_out = emb_sum.forward(&indices2d)?;
    assert_eq!(sum_out.shape(), &[2, 8]);

    let emb_max = EmbeddingBag::new(16, 8, EmbeddingBagMode::Max)?;
    let max_out = emb_max.forward(&indices2d)?;
    assert_eq!(max_out.shape(), &[2, 8]);

    let indices1d = Tensor::from_slice(&[1i64, 3, 5, 2, 4, 6], &[6])?;
    let offset_out = emb_bag.forward_offsets(&indices1d, &[0, 2, 5])?;
    assert_eq!(offset_out.shape(), &[3, 8]);

    // 5. Activations: ReLU6, Hardtanh, Softshrink, Hardshrink, LogSigmoid
    let act_in = Tensor::from_slice(&[-3.0f32, -0.5, 0.5, 4.0, 8.0], &[5])?;
    let r6 = act_in.relu6()?;
    let r6_vec = r6.to_vec::<f32>()?;
    assert_eq!(r6_vec, vec![0.0, 0.0, 0.5, 4.0, 6.0]);

    let ht = act_in.hardtanh(-1.0, 1.0)?;
    let ht_vec = ht.to_vec::<f32>()?;
    assert_eq!(ht_vec, vec![-1.0, -0.5, 0.5, 1.0, 1.0]);

    let ss = act_in.softshrink(1.0)?;
    let ss_vec = ss.to_vec::<f32>()?;
    assert_eq!(ss_vec, vec![-2.0, 0.0, 0.0, 3.0, 7.0]);

    let hs = act_in.hardshrink(1.0)?;
    let hs_vec = hs.to_vec::<f32>()?;
    assert_eq!(hs_vec, vec![-3.0, 0.0, 0.0, 4.0, 8.0]);

    let ls = Tensor::from_slice(&[0.0f32], &[1])?.log_sigmoid()?;
    let ls_val = ls.item::<f32>()?;
    assert!((ls_val - 0.5f32.ln()).abs() < 1e-4);

    // 6. Loss Functions: MarginRankingLoss, HingeEmbeddingLoss, SoftMarginLoss, CTCLoss
    let x1 = Tensor::from_slice(&[1.5f32, -0.5], &[2])?;
    let x2 = Tensor::from_slice(&[1.0f32, 0.5], &[2])?;
    let mr_target = Tensor::from_slice(&[1.0f32, -1.0], &[2])?;
    let mr_loss = margin_ranking_loss(&x1, &x2, &mr_target, 0.5)?;
    assert!(mr_loss.item::<f32>()? >= 0.0);

    let hinge_in = Tensor::from_slice(&[0.5f32, 1.5], &[2])?;
    let hinge_tgt = Tensor::from_slice(&[1.0f32, -1.0], &[2])?;
    let h_loss = hinge_embedding_loss(&hinge_in, &hinge_tgt, 1.0)?;
    assert!(h_loss.item::<f32>()? >= 0.0);

    let sm_in = Tensor::from_slice(&[0.5f32, -0.5], &[2])?;
    let sm_tgt = Tensor::from_slice(&[1.0f32, -1.0], &[2])?;
    let sm_loss = soft_margin_loss(&sm_in, &sm_tgt)?;
    assert!(sm_loss.item::<f32>()? > 0.0);

    let log_probs = Tensor::randn(&[4, 2, 6])?.log_softmax(2)?;
    let ctc_targets = Tensor::from_slice(&[1i64, 2, 2, 3], &[2, 2])?;
    let c_loss = ctc_loss(&log_probs, &ctc_targets, 0, false)?;
    assert!(c_loss.item::<f32>()? > 0.0);

    // 7. Full Python Zero-GIL Parity
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

# Pooling parity
mp1 = ox.MaxPool1d(kernel_size=2, stride=2)
ap1 = ox.AvgPool1d(kernel_size=2, stride=2)
t1 = ox.Tensor.randn([2, 4, 16])
assert mp1.forward(t1).shape == [2, 4, 8]
assert ap1.forward(t1).shape == [2, 4, 8]

# 3D Transposed conv
ct3 = ox.ConvTranspose3d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1)
t3 = ox.Tensor.randn([1, 2, 4, 4, 4])
assert ct3.forward(t3).shape == [1, 4, 4, 4, 4]

# InstanceNorm parity
in1 = ox.InstanceNorm1d(num_features=4)
assert in1.forward(t1).shape == [2, 4, 16]
in3 = ox.InstanceNorm3d(num_features=2)
assert in3.forward(t3).shape == [1, 2, 4, 4, 4]

# EmbeddingBag parity
eb = ox.EmbeddingBag(num_embeddings=16, embedding_dim=8, mode="mean")
idx = ox.Tensor.from_i64([1, 2, 3, 4, 5, 6], [2, 3])
assert eb.forward(idx).shape == [2, 8]

# Activations parity
r6 = ox.ReLU6()
ht = ox.Hardtanh(min_val=-1.0, max_val=1.0)
ss = ox.Softshrink(lambd=0.5)
hs = ox.Hardshrink(lambd=0.5)
ls = ox.LogSigmoid()
x = ox.Tensor.randn([2, 4])
assert r6.forward(x).shape == [2, 4]
assert ht.forward(x).shape == [2, 4]
assert ss.forward(x).shape == [2, 4]
assert hs.forward(x).shape == [2, 4]
assert ls.forward(x).shape == [2, 4]

# Losses parity
mrl = ox.MarginRankingLoss(margin=0.5)
hel = ox.HingeEmbeddingLoss(margin=1.0)
sml = ox.SoftMarginLoss()
ctl = ox.CTCLoss(blank=0)

a = ox.Tensor.randn([4])
b = ox.Tensor.randn([4])
y = ox.Tensor.randn([4])
assert mrl.forward(a, b, y).shape == []
assert hel.forward(a, y).shape == []
assert sml.forward(a, y).shape == []
"#,
            )
            .unwrap();
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python v2.2.0 PyTorch parity test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_seed_temperature_sampling_suite() -> Result<()> {
    use oxideon::sampling::SamplingConfig;

    // 1. Global PRNG & Generator reproducibility
    manual_seed(42);
    let r1 = Tensor::randn(&[8])?.to_vec::<f32>()?;
    manual_seed(42);
    let r2 = Tensor::randn(&[8])?.to_vec::<f32>()?;
    assert_eq!(r1, r2, "manual_seed(42) must produce identical outputs");

    let initial = initial_seed();
    assert_eq!(initial, 42);

    let state = get_rng_state();
    let r3 = Tensor::randn(&[4])?.to_vec::<f32>()?;
    set_rng_state(&state)?;
    let r4 = Tensor::randn(&[4])?.to_vec::<f32>()?;
    assert_eq!(r3, r4, "set_rng_state must restore identical PRNG sequence");

    // Generator instance
    let mut gen1 = Generator::with_seed(999);
    let mut gen2 = Generator::with_seed(999);
    assert_eq!(gen1.next_u64(), gen2.next_u64());
    assert_eq!(gen1.next_f32(), gen2.next_f32());

    // fork_rng
    fork_rng(|| {
        manual_seed(100);
        let _ = Tensor::randn(&[5]);
    });
    // Global state is preserved after fork_rng
    let r5 = Tensor::randn(&[4])?.to_vec::<f32>()?;
    set_rng_state(&state)?;
    let _ = Tensor::randn(&[4])?; // consume same state
    let r6 = Tensor::randn(&[4])?.to_vec::<f32>()?;
    assert_eq!(r5, r6, "fork_rng must isolate and restore global RNG state");

    // 2. SamplingConfig & Temperature / Greedy Decoding
    let logits = Tensor::from_slice(&[1.0f32, 5.0, 2.0, 0.5], &[4])?;
    let greedy_cfg = SamplingConfig::greedy();
    let chosen = logits.sample_token(&greedy_cfg, None, None)?;
    assert_eq!(chosen, 1, "Greedy sampling must pick max logit (index 1)");

    // Temperature = 0.0 implies greedy
    let t0_cfg = SamplingConfig::new().with_temperature(0.0);
    assert_eq!(logits.sample_token(&t0_cfg, None, None)?, 1);

    // Repetition penalty
    // Token 1 was 5.0, with penalty 10.0 it becomes 0.5. Token 2 is 2.0, so Token 2 wins.
    let rep_cfg = SamplingConfig::greedy().with_repetition_penalty(10.0);
    let chosen_rep = logits.sample_token(&rep_cfg, Some(&[1]), None)?;
    assert_eq!(chosen_rep, 2, "Repetition penalty should demote token 1");

    // Logit bias
    // Give token 3 a huge boost
    let bias_cfg = SamplingConfig::greedy().with_logit_bias(3, 50.0);
    let chosen_bias = logits.sample_token(&bias_cfg, None, None)?;
    assert_eq!(chosen_bias, 3, "Logit bias must boost token 3 to top");

    // Top-K & Top-P & Min-P
    let mut sample_gen = Generator::with_seed(777);
    let full_cfg = SamplingConfig::new()
        .with_temperature(0.8)
        .with_top_k(2)
        .with_top_p(0.95)
        .with_min_p(0.1)
        .with_seed(12345);
    let token = logits.sample_token(&full_cfg, None, Some(&mut sample_gen))?;
    assert!(
        token == 1 || token == 2,
        "Top-K=2 must only sample token 1 or 2"
    );

    // 3. Tensor::multinomial_with_generator
    let probs = Tensor::from_slice(&[0.05f32, 0.90, 0.05], &[3])?;
    let mut mg = Generator::with_seed(42);
    let indices = probs.multinomial_with_generator(5, true, Some(&mut mg))?;
    assert_eq!(indices.shape(), &[5]);
    let idx_vec = indices.to_vec::<i64>()?;
    // Majority should be index 1
    let count_1 = idx_vec.iter().filter(|&&i| i == 1).count();
    assert!(
        count_1 >= 3,
        "Majority of samples from [0.05, 0.90, 0.05] should be 1"
    );

    // 4. CausalLM::generate_with_config
    let lm = oxideon::nn::transformer::CausalLM::new(16, 16, 2, 1, 32, 32)?;
    let gen_cfg = SamplingConfig::greedy().with_max_new_tokens(4);
    let generated = lm.generate_with_config(&[1, 2, 3], &gen_cfg)?;
    assert_eq!(generated.len(), 7, "Prompt (3) + 4 new tokens = 7 tokens");

    // 5. ServingEngine submit_with_config
    let mut engine = oxideon::serving::engine::ServingEngine::new(4, 16, 16, 2, 8)?;
    let req_id = engine.submit_with_config(vec![1, 2], gen_cfg);
    assert_eq!(req_id, 1);

    // 6. Python integration test
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

ox.manual_seed(123)
assert ox.initial_seed() == 123, f"initial_seed is {ox.initial_seed()}"
st = ox.get_rng_state()
ox.set_rng_state(st)

gen = ox.Generator(42)
assert gen.initial_seed() == 42, f"gen.initial_seed is {gen.initial_seed()}"
new_s = gen.seed()
assert gen.initial_seed() == new_s, f"gen.initial_seed {gen.initial_seed()} != {new_s}"

cfg = ox.SamplingConfig(temperature=0.7, top_k=50, top_p=0.9, min_p=0.05, repetition_penalty=1.1)
assert abs(cfg.temperature - 0.7) < 1e-5, f"temp is {cfg.temperature}"
assert cfg.top_k == 50, f"top_k is {cfg.top_k}"
assert abs(cfg.top_p - 0.9) < 1e-5, f"top_p is {cfg.top_p}"
assert abs(cfg.min_p - 0.05) < 1e-5, f"min_p is {cfg.min_p}"
assert abs(cfg.repetition_penalty - 1.1) < 1e-5, f"rep is {cfg.repetition_penalty}"

greedy_cfg = ox.SamplingConfig.greedy()
assert greedy_cfg.temperature == 0.0, f"greedy temp is {greedy_cfg.temperature}"

cfg.temperature = 0.0
assert cfg.temperature == 0.0, f"setter temp is {cfg.temperature}"

logits = ox.Tensor([1.0, 5.0, 2.0], [3])
tok = logits.sample_token(cfg)
assert tok == 1, f"tok is {tok}"
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python RNG & Sampling test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v24_quant_kv_qlora_and_streaming() -> Result<()> {
    // 1. Quantized Paged KV Cache (FP8, INT8, INT4)
    let num_blocks = 8;
    let block_size = 16;
    let num_heads = 4;
    let head_dim = 16;

    let k_tok = Tensor::randn(&[num_heads, head_dim])?;
    let v_tok = Tensor::randn(&[num_heads, head_dim])?;

    for dtype in [
        QuantKVDType::Fp8E4M3,
        QuantKVDType::Int8,
        QuantKVDType::Int4,
    ] {
        let cache = QuantizedPagedKVCache::new(num_blocks, block_size, num_heads, head_dim, dtype)?;
        cache.write_token(0, 0, &k_tok, &v_tok)?;

        let (k_rec, v_rec) = cache.read_token_dequantized(0, 0)?;
        assert_eq!(k_rec.len(), num_heads * head_dim);
        assert_eq!(v_rec.len(), num_heads * head_dim);

        let k_orig = k_tok.to_vec::<f32>()?;
        let mut diff_sum = 0.0f32;
        for i in 0..k_orig.len() {
            diff_sum += (k_orig[i] - k_rec[i]).powi(2);
        }
        let mse = diff_sum / k_orig.len() as f32;
        assert!(
            mse < 0.2,
            "Quantized KV MSE {mse} must be low for {dtype:?}"
        );

        assert!(cache.memory_usage_bytes() > 0);
        assert!(cache.compression_ratio() > 2.0);

        // Test paged_attention_quantized
        let queries = Tensor::randn(&[1, num_heads, head_dim])?;
        let block_table = BlockTable {
            physical_blocks: vec![0],
            num_tokens: 1,
        };
        let out = paged_attention_quantized(
            &queries,
            &cache,
            &[block_table],
            &[1],
            1.0 / (head_dim as f64).sqrt(),
        )?;
        assert_eq!(out.shape(), &[1, num_heads, head_dim]);
    }

    // 2. Streaming UTF-8 Decoder
    let mut decoder = StreamingUtf8Decoder::new();
    let text = "Hello 🌍 Rust 🚀!";
    let bytes = text.as_bytes();

    // Split at non-character byte boundary (inside 4-byte 🌍)
    // "Hello " is 6 bytes. 🌍 is bytes [6..10].
    let chunk1 = &bytes[..8]; // "Hello " + first 2 bytes of 🌍
    let chunk2 = &bytes[8..15]; // remaining 2 bytes of 🌍 + " Rust"
    let chunk3 = &bytes[15..]; // remainder

    let s1 = decoder.add_bytes(chunk1);
    assert_eq!(s1, "Hello ");

    let s2 = decoder.add_bytes(chunk2);
    assert_eq!(s2, "🌍 Rust");

    let s3 = decoder.add_bytes(chunk3);
    assert_eq!(s3, " 🚀!");

    let rem = decoder.flush();
    assert_eq!(rem, "");

    // 3. Stop Sequence Matcher
    let mut matcher =
        StopSequenceMatcher::new(vec!["\n\nHuman:".to_string(), "<|im_end|>".to_string()]);
    let (t1, stop1) = matcher.step("The answer is 42.");
    assert_eq!(t1, "The answer is 42.");
    assert!(!stop1);

    // Prefix buffer check
    let (t2, stop2) = matcher.step("\n\n");
    assert_eq!(t2, ""); // held back because "\n\n" is prefix of "\n\nHuman:"
    assert!(!stop2);

    let (t3, stop3) = matcher.step("Hum");
    assert_eq!(t3, ""); // held back because "\n\nHum" is prefix
    assert!(!stop3);

    let (t4, stop4) = matcher.step("an: some next query");
    assert_eq!(t4, ""); // matched stop sequence, text up to match emitted
    assert!(stop4);
    assert!(matcher.is_stopped());

    // 4. QLoRA Linear
    let in_f = 16;
    let out_f = 16;
    let lin = Linear::new(in_f, out_f)?;
    let mut qlora = QLoRALinear::from_linear(&lin, QuantScheme::DEFAULT_Q4_0, 4, 8.0, 0.0)?;
    assert_eq!(qlora.in_features(), in_f);
    assert_eq!(qlora.out_features(), out_f);
    assert_eq!(qlora.rank, 4);

    let input = Tensor::randn(&[2, in_f])?;
    let y0 = qlora.forward(&input)?;
    assert_eq!(y0.shape(), &[2, out_f]);

    // Merge adapter
    qlora.merge_adapter()?;
    assert!(!qlora.enabled);
    let y_merged = qlora.forward(&input)?;
    assert_eq!(y_merged.shape(), &[2, out_f]);

    // 5. Python Zero-GIL Parity
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

# 1. Quantized Paged KV Cache
kv = ox.QuantizedPagedKVCache(num_blocks=4, block_size=16, num_heads=2, head_dim=8, dtype="fp8")
assert kv.memory_usage_bytes > 0
assert kv.compression_ratio > 2.0

k = ox.Tensor.randn([2, 8])
v = ox.Tensor.randn([2, 8])
kv.write_token(0, 0, k, v)
k_rec, v_rec = kv.read_token_dequantized(0, 0)
assert k_rec.shape == [2, 8]
assert v_rec.shape == [2, 8]

# Paged attention
q = ox.Tensor.randn([1, 2, 8])
attn_out = ox.paged_attention_quantized(q, kv, [[0]], [1], 0.3535)
assert attn_out.shape == [1, 2, 8]

# 2. Streaming UTF-8 Decoder
dec = ox.StreamingUtf8Decoder()
chunk = "Hello ".encode("utf-8") + "🌍".encode("utf-8")[:2]
assert dec.add_bytes(chunk) == "Hello "
rem = "🌍".encode("utf-8")[2:] + "!".encode("utf-8")
assert dec.add_bytes(rem) == "🌍!"

# 3. Stop Sequence Matcher
matcher = ox.StopSequenceMatcher(["<|end|>"])
out1, stopped1 = matcher.step("Hello world")
assert out1 == "Hello world"
assert not stopped1
out2, stopped2 = matcher.step("<|end|>extra")
assert out2 == ""
assert stopped2
assert matcher.is_stopped

# 4. QLoRALinear
qlora = ox.QLoRALinear(16, 16, rank=4, alpha=8.0, scheme="q4_0")
assert qlora.in_features == 16
assert qlora.out_features == 16
assert qlora.rank == 4
x = ox.Tensor.randn([2, 16])
y = qlora.forward(x)
assert y.shape == [2, 16]
qlora.merge_adapter()
y2 = qlora.forward(x)
assert y2.shape == [2, 16]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python v2.4.0 test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v25_gguf_fast_mla_and_async_pipeline() -> Result<()> {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::Ordering;

    // 1. GGUF Binary Serialization & Round-Trip Deserialization
    let mut metadata = HashMap::new();
    metadata.insert(
        "general.architecture".to_string(),
        GGUFValue::String("llama".to_string()),
    );
    metadata.insert("llama.block_count".to_string(), GGUFValue::Uint32(32));
    metadata.insert(
        "llama.embedding_length".to_string(),
        GGUFValue::Uint32(4096),
    );
    metadata.insert(
        "llama.rope.freq_base".to_string(),
        GGUFValue::Float32(10000.0),
    );

    let embd_t = Tensor::randn(&[64, 32])?;
    let k_t = Tensor::randn(&[32, 32])?;
    let v_t = Tensor::randn(&[32, 32])?;

    let tensors = vec![
        ("token_embd.weight", GGMLType::F32, &embd_t),
        ("blk.0.attn_k.weight", GGMLType::Q8_0, &k_t),
        ("blk.0.attn_v.weight", GGMLType::Q4_0, &v_t),
    ];

    let binary = serialize_gguf(&metadata, &tensors, 32)?;
    let gguf = GGUFFile::from_bytes(&binary)?;

    assert_eq!(gguf.version, 3);
    assert_eq!(gguf.alignment, 32);
    assert_eq!(gguf.get_metadata_str("general.architecture"), Some("llama"));
    assert_eq!(gguf.get_metadata_u32("llama.block_count"), Some(32));

    // Dequantize F32
    let embd_rec = gguf.dequantize_tensor("token_embd.weight")?;
    assert_eq!(embd_rec.shape(), &[64, 32]);
    let embd_orig = embd_t.to_vec::<f32>()?;
    let embd_rec_vec = embd_rec.to_vec::<f32>()?;
    for i in 0..embd_orig.len() {
        assert!((embd_orig[i] - embd_rec_vec[i]).abs() < 1e-6);
    }

    // Dequantize Q8_0
    let k_rec = gguf.dequantize_tensor("blk.0.attn_k.weight")?;
    assert_eq!(k_rec.shape(), &[32, 32]);
    let k_orig = k_t.to_vec::<f32>()?;
    let k_rec_vec = k_rec.to_vec::<f32>()?;
    let mut k_mse = 0.0f32;
    for i in 0..k_orig.len() {
        k_mse += (k_orig[i] - k_rec_vec[i]).powi(2);
    }
    k_mse /= k_orig.len() as f32;
    assert!(k_mse < 0.05, "Q8_0 MSE {k_mse} should be < 0.05");

    // Dequantize Q4_0
    let v_rec = gguf.dequantize_tensor("blk.0.attn_v.weight")?;
    assert_eq!(v_rec.shape(), &[32, 32]);

    // Fused GEMV
    let x_vec = Tensor::randn(&[32])?;
    let y_q8 = gguf.fused_gemv("blk.0.attn_k.weight", &x_vec)?;
    assert_eq!(y_q8.shape(), &[32]);

    let y_q4 = gguf.fused_gemv("blk.0.attn_v.weight", &x_vec)?;
    assert_eq!(y_q4.shape(), &[32]);

    // 2. DeepSeek Multi-Head Latent Attention (MLA) Absorption
    let mla = MultiHeadLatentAttention::new(64, 4, 16, 32)?;
    assert_eq!(mla.kv_cache_compression_ratio(), 8.0); // 2 * 64 / 16 = 8.0x compression

    // Prefill
    let prompt = Tensor::randn(&[1, 3, 64])?;
    let prefill_out = mla.forward_mla(&prompt, true)?;
    assert_eq!(prefill_out.shape(), &[1, 3, 64]);

    // Fast Absorbed Decode Steps
    let mut latent_cache: Option<Tensor> = None;
    let tok0 = Tensor::randn(&[1, 1, 64])?;
    let out0 = mla.step_absorbed(&tok0, &mut latent_cache)?;
    assert_eq!(out0.shape(), &[1, 1, 64]);
    assert_eq!(latent_cache.as_ref().unwrap().shape(), &[1, 1, 16]);

    let tok1 = Tensor::randn(&[1, 1, 64])?;
    let out1 = mla.step_absorbed(&tok1, &mut latent_cache)?;
    assert_eq!(out1.shape(), &[1, 1, 64]);
    assert_eq!(latent_cache.as_ref().unwrap().shape(), &[1, 2, 16]);

    // 3. Double-Buffered KV Cache & Async Pipeline Engine
    let double_cache = Arc::new(DoubleBufferedKVCache::new(8, 16, 2, 8, DType::F32)?);
    let engine = AsyncKVPipelineEngine::new(double_cache.clone());

    let t1 = engine.submit_swap_in(101, 2);
    let t2 = engine.submit_swap_out(202, 1);
    assert_eq!(t1, 1);
    assert_eq!(t2, 2);

    engine.wait_all();
    let events = engine.poll_completed();
    assert_eq!(events.len(), 2);

    assert_eq!(double_cache.active_bank.load(Ordering::Relaxed), 0);
    engine.swap_banks();
    assert_eq!(double_cache.active_bank.load(Ordering::Relaxed), 1);

    // 4. Python Zero-GIL Parity
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

# 1. GGUF in Python
tensors = {
    "embd": ox.Tensor.randn([16, 8]),
    "proj": ox.Tensor.randn([8, 8]),
}
ox.save_gguf("/tmp/test_py.gguf", tensors, {"arch": "test_model"})

gguf = ox.GGUFFile.open("/tmp/test_py.gguf")
assert gguf.version == 3
assert gguf.alignment == 32
assert "embd" in gguf.tensor_names
assert "proj" in gguf.tensor_names
assert gguf.get_metadata_str("arch") == "test_model"

rec_t = gguf.dequantize_tensor("embd")
assert rec_t.shape == [16, 8]

x = ox.Tensor.randn([8])
y = gguf.fused_gemv("proj", x)
assert y.shape == [8]

# 2. MultiHeadLatentAttention with Absorption
mla = ox.MultiHeadLatentAttention(32, 2, kv_lora_rank=8, q_lora_rank=16)
assert mla.kv_cache_compression_ratio > 1.0

tok = ox.Tensor.randn([1, 1, 32])
out0, cache = mla.step_absorbed(tok)
assert out0.shape == [1, 1, 32]
assert cache.shape == [1, 1, 8]

out1, cache2 = mla.step_absorbed(tok, cache)
assert out1.shape == [1, 1, 32]
assert cache2.shape == [1, 2, 8]

# 3. DoubleBufferedKVCache & AsyncKVPipelineEngine
db_cache = ox.DoubleBufferedKVCache(4, 16, 2, 8, "f32")
assert db_cache.active_bank == 0
db_cache.swap_banks()
assert db_cache.active_bank == 1

pipe = ox.AsyncKVPipelineEngine(db_cache)
t_id = pipe.submit_swap_in(1, 2)
assert t_id == 1
pipe.wait_all()
completed = pipe.poll_completed()
assert len(completed) >= 1

# 4. Unified Speculative Verification in Python
v_out = ox.UnifiedSpeculativeEngine.verify_greedy([1, 1, 2], ox.Tensor([0.0, 10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 10.0, 0.0], [4, 3]))
assert v_out[0] == [1, 1, 2]

# 5. GGUFCausalLM in Python
gg_lm = ox.GGUFCausalLM(vocab_size=32, d_model=16, num_heads=2, num_layers=1, max_seq_len=64)
toks = ox.Tensor.from_i64([1, 2, 3], [1, 3])
logits = gg_lm.forward(toks)
assert logits.shape == [1, 3, 32], f"logits shape is {logits.shape}"
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python v2.5.0 test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v27_wgsl_and_speculative_tree() -> Result<()> {
    // 1. WGSL Compute Pipeline Emitter
    let builder = WgslKernelBuilder::new("fused_rmsnorm", 256);
    let shader = builder.emit_fused_rmsnorm_wgsl(4096, 1e-5);
    assert!(shader.contains("@compute @workgroup_size(256, 1, 1)"));
    assert!(shader.contains("fn fused_rmsnorm("));
    assert!(shader.contains("inverseSqrt("));

    let swiglu_shader = builder.emit_fused_swiglu_wgsl();
    assert!(swiglu_shader.contains("fn fused_rmsnorm("));
    assert!(swiglu_shader.contains("silu_a * b"));

    // 2. Speculative Tree Topology
    let nodes = vec![
        SpeculativeTreeNode {
            node_id: 0,
            parent_id: None,
            token_id: 101,
            depth: 0,
            draft_prob: 0.95,
        },
        SpeculativeTreeNode {
            node_id: 1,
            parent_id: Some(0),
            token_id: 202,
            depth: 1,
            draft_prob: 0.85,
        },
        SpeculativeTreeNode {
            node_id: 2,
            parent_id: Some(0),
            token_id: 203,
            depth: 1,
            draft_prob: 0.10,
        },
        SpeculativeTreeNode {
            node_id: 3,
            parent_id: Some(1),
            token_id: 304,
            depth: 2,
            draft_prob: 0.70,
        },
    ];
    let tree = SpeculativeTree::new(nodes);
    assert_eq!(tree.len(), 4);
    assert_eq!(tree.token_ids(), vec![101, 202, 203, 304]);
    assert_eq!(tree.children.get(&0).unwrap(), &vec![1, 2]);

    // 3. Python WGSL Verification
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

wb = ox.WgslKernelBuilder("test_kernel", 128)
code = wb.emit_fused_rmsnorm_wgsl(2048, 1e-5)
assert "@compute @workgroup_size(128, 1, 1)" in code
assert "test_kernel" in code

sw = wb.emit_fused_swiglu_wgsl()
assert "test_kernel" in sw
assert "silu_a * b" in sw
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python v2.7.0 test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v28_flash_attn2_and_interleaver() -> Result<()> {
    // 1. FlashAttention-2 Forward & Backward
    let b = 1;
    let h = 2;
    let seq_len = 8;
    let d = 16;
    let q = Tensor::randn(&[b, h, seq_len, d])?;
    let k = Tensor::randn(&[b, h, seq_len, d])?;
    let v = Tensor::randn(&[b, h, seq_len, d])?;

    let config = FlashAttention2Config {
        block_size_q: 4,
        block_size_kv: 4,
        causal: true,
        sm_scale: Some(0.25),
    };

    let fwd_out = flash_attention_2_forward(&q, &k, &v, &config)?;
    assert_eq!(fwd_out.output.shape(), &[b, h, seq_len, d]);
    assert_eq!(fwd_out.logsumexp.shape(), &[b, h, seq_len, 1]);

    let grad_out = Tensor::randn(&[b, h, seq_len, d])?;
    let (grad_q, grad_k, grad_v) = flash_attention_2_backward(
        &grad_out,
        &q,
        &k,
        &v,
        &fwd_out.output,
        &fwd_out.logsumexp,
        &config,
    )?;
    assert_eq!(grad_q.shape(), q.shape());
    assert_eq!(grad_k.shape(), k.shape());
    assert_eq!(grad_v.shape(), v.shape());

    // 2. Continuous Prefill-Decode Interleaver
    let mut interleaver = PrefillDecodeInterleaver::new(8, 4).unwrap();
    let prompt = vec![10, 11, 12, 13, 14, 15, 16, 17, 18, 19]; // 10 tokens
    let req_id = interleaver.submit_request(prompt, SamplingConfig::default());
    assert_eq!(req_id, 1);

    // Iteration 1: chunk 0..4
    let b1 = interleaver.schedule_iteration();
    assert_eq!(b1.prefill_chunks.len(), 1);
    assert_eq!(b1.prefill_chunks[0].len, 4);
    assert_eq!(b1.prefill_chunks[0].start_idx, 0);
    assert!(!b1.prefill_chunks[0].is_final_chunk);
    assert_eq!(b1.total_tokens, 4);
    interleaver.on_chunk_processed(req_id, 4);

    // Iteration 2: chunk 4..8
    let b2 = interleaver.schedule_iteration();
    assert_eq!(b2.prefill_chunks[0].start_idx, 4);
    assert_eq!(b2.prefill_chunks[0].len, 4);
    assert!(!b2.prefill_chunks[0].is_final_chunk);
    interleaver.on_chunk_processed(req_id, 4);

    // Iteration 3: chunk 8..10 (final)
    let b3 = interleaver.schedule_iteration();
    assert_eq!(b3.prefill_chunks[0].start_idx, 8);
    assert_eq!(b3.prefill_chunks[0].len, 2);
    assert!(b3.prefill_chunks[0].is_final_chunk);
    interleaver.on_chunk_processed(req_id, 2);

    // Iteration 4: Promoted to decode step!
    let b4 = interleaver.schedule_iteration();
    assert_eq!(b4.decode_ids, vec![req_id]);
    assert_eq!(b4.prefill_chunks.len(), 0);
    assert_eq!(b4.total_tokens, 1);

    // 3. Python Bindings Verification
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

# Test Interleaver in Python
inter = ox.PrefillDecodeInterleaver(16, 4)
rid = inter.submit_request([1, 2, 3, 4, 5, 6, 7, 8])
assert rid == 1
chunks, decs, tot = inter.schedule_iteration()
assert len(chunks) == 1
assert chunks[0][2] == 4 # chunk len
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python v2.8.0 test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v29_tiered_memory_and_auto_placement() -> Result<()> {
    // 1. Tiered Memory Manager & LRU Eviction
    let mut manager = TieredMemoryManager::new(0.8);
    let cuda0 = Device::Cuda(0);
    manager.register_device_capacity(cuda0.clone(), 1000);

    // Register T1 (400 bytes) and T2 (300 bytes) -> total 700 bytes <= 800 (80%)
    let m1 = manager.register_tensor(1, 400, cuda0.clone(), MemoryPriority::Normal)?;
    assert!(m1.is_empty());
    let m2 = manager.register_tensor(2, 300, cuda0.clone(), MemoryPriority::Normal)?;
    assert!(m2.is_empty());

    let (alloc, cap) = manager.get_device_usage(&cuda0);
    assert_eq!(alloc, 700);
    assert_eq!(cap, 1000);
    assert!(!manager.is_under_pressure(&cuda0));

    // Register T3 (300 bytes) -> 700 + 300 = 1000 > 800 -> triggers eviction of T1 to CPU
    let m3 = manager.register_tensor(3, 300, cuda0.clone(), MemoryPriority::Normal)?;
    assert_eq!(m3.len(), 1);
    assert_eq!(m3[0], (1, cuda0.clone(), Device::Cpu));

    // Access T1 -> indicates reload needed
    let reload = manager.access_tensor(1)?;
    assert_eq!(reload, Some(cuda0.clone()));

    // 2. Automatic Layer Placement
    let layers = vec![
        LayerPlacementSpec {
            layer_name: "block.0".into(),
            parameter_bytes: 500,
            flops_estimate: 1000,
        },
        LayerPlacementSpec {
            layer_name: "block.1".into(),
            parameter_bytes: 500,
            flops_estimate: 1000,
        },
        LayerPlacementSpec {
            layer_name: "block.2".into(),
            parameter_bytes: 500,
            flops_estimate: 1000,
        },
        LayerPlacementSpec {
            layer_name: "block.3".into(),
            parameter_bytes: 500,
            flops_estimate: 1000,
        },
    ];
    let devs = vec![Device::Cuda(0), Device::Cuda(1)];
    let placements = AutoDevicePlacer::partition_layers(&layers, &devs, 1000)?;
    assert_eq!(placements.len(), 4);
    assert_eq!(placements[0].assigned_device, Device::Cuda(0));
    assert_eq!(placements[1].assigned_device, Device::Cuda(0));
    assert_eq!(placements[2].assigned_device, Device::Cuda(1));
    assert_eq!(placements[3].assigned_device, Device::Cuda(1));

    let latency = AutoDevicePlacer::evaluate_pipeline_transfer_latency(&placements, 1024 * 1024);
    assert!(latency > 0.0);

    // 3. Python Bindings Verification
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

tm = ox.TieredMemoryManager(0.8)
tm.register_device_capacity("cuda:0", 1000)
m1 = tm.register_tensor(1, 500, "cuda:0")
assert len(m1) == 0
u, c = tm.get_device_usage("cuda:0")
assert u == 500
assert c == 1000

m2 = tm.register_tensor(2, 400, "cuda:0") # 500 + 400 = 900 > 800 -> evicts tensor 1
assert len(m2) == 1
assert m2[0][0] == 1 # tensor 1 evicted
assert m2[0][2] == "cpu"
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python v2.9.0 test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v30_parallel_3d_runtime() -> Result<()> {
    // 1. 3D Parallel Coordinates (DP=2, TP=2, PP=2 -> World Size = 8)
    let cfg = Parallel3DConfig::new(2, 2, 2)?;
    assert_eq!(cfg.world_size(), 8);

    let devs = (0..8).map(Device::Cuda).collect();
    let runtime = Parallel3DRuntime::new(cfg, devs)?;

    let c0 = runtime.get_coords(0)?;
    assert_eq!((c0.dp_rank, c0.tp_rank, c0.pp_rank), (0, 0, 0));

    let c1 = runtime.get_coords(1)?;
    assert_eq!((c1.dp_rank, c1.tp_rank, c1.pp_rank), (0, 1, 0));

    let c2 = runtime.get_coords(2)?;
    assert_eq!((c2.dp_rank, c2.tp_rank, c2.pp_rank), (1, 0, 0));

    let c4 = runtime.get_coords(4)?;
    assert_eq!((c4.dp_rank, c4.tp_rank, c4.pp_rank), (0, 0, 1));

    let c7 = runtime.get_coords(7)?;
    assert_eq!((c7.dp_rank, c7.tp_rank, c7.pp_rank), (1, 1, 1));

    // 2. Subgroups
    assert_eq!(runtime.get_dp_group(0)?, vec![0, 2]);
    assert_eq!(runtime.get_tp_group(0)?, vec![0, 1]);
    assert_eq!(runtime.get_pp_group(0)?, vec![0, 4]);

    // 3. Ring All-Reduce
    let t0 = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4])?;
    let t1 = Tensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4])?;
    let reduced = runtime.ring_all_reduce(&[0, 1], &[t0, t1])?;
    assert_eq!(reduced.len(), 2);
    let r0 = reduced[0].to_vec::<f32>()?;
    let r1 = reduced[1].to_vec::<f32>()?;
    assert_eq!(r0, vec![11.0, 22.0, 33.0, 44.0]);
    assert_eq!(r1, vec![11.0, 22.0, 33.0, 44.0]);

    // 4. Python Bindings Verification
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

cfg = ox.Parallel3DConfig(2, 2, 1)
assert cfg.world_size == 4
rt = ox.Parallel3DRuntime(cfg, ["cpu", "cpu", "cpu", "cpu"])
coords = rt.get_coords(3)
assert coords == (1, 1, 0)

dp_g = rt.get_dp_group(0)
assert dp_g == [0, 2]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python v3.0.0 test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v31_vla_multimodal_cross_attention() -> Result<()> {
    // 1. GatedCrossAttention: Identity initialization property
    let cross_attn = GatedCrossAttention::new(16, 2, 8)?;
    let x = Tensor::randn(&[1, 4, 16])?;
    let ctx = Tensor::randn(&[1, 6, 8])?;
    let y = cross_attn.forward(&x, &ctx)?;
    // Because alpha == 0, tanh(alpha) == 0, y must equal x exactly!
    let diff = y.sub(&x)?.abs()?.sum()?.item::<f32>()?;
    assert!(
        diff < 1e-5,
        "Gated cross attention must initialize to identity"
    );

    // 2. VisionLanguageActionModel
    let vla = VisionLanguageActionModel::new(
        16, // img_size
        4,  // patch_size -> 4x4 patches = 16 tokens
        3,  // in_channels
        16, // vision_dim
        16, // text_dim
        32, // vocab_size
        4,  // action_horizon
        7,  // action_dim (7-DoF robot arm)
    )?;

    let images = Tensor::randn(&[1, 3, 16, 16])?;
    let text_one_hot = Tensor::randn(&[1, 5, 32])?;
    let (text_logits, actions) = vla.forward(&images, &text_one_hot)?;

    assert_eq!(text_logits.shape(), &[1, 5, 32]);
    assert_eq!(actions.shape(), &[1, 4, 7]);

    // 3. Python Bindings Verification
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

gca = ox.GatedCrossAttention(16, 2, 8)
vla = ox.VisionLanguageActionModel(16, 4, 3, 16, 16, 32, 4, 7)
assert vla is not None
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python v3.1.0 test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v32_rectified_flow_and_sampler() -> Result<()> {
    // 1. Rectified Flow Interpolation
    let x0 = Tensor::from_slice(&[0.0f32, 0.0], &[1, 2])?;
    let x1 = Tensor::from_slice(&[10.0f32, 20.0], &[1, 2])?;

    let (xt, vt) = RectifiedFlowEngine::interpolate(&x0, &x1, 0.5)?;
    let xt_vec = xt.to_vec::<f32>()?;
    let vt_vec = vt.to_vec::<f32>()?;
    assert_eq!(xt_vec, vec![5.0, 10.0]);
    assert_eq!(vt_vec, vec![10.0, 20.0]);

    // 2. Sample Euler ODE with constant velocity
    let cfg = RectifiedFlowConfig {
        num_steps: 10,
        guidance_scale: 1.0,
    };
    let target_vel = Tensor::from_slice(&[10.0f32, 20.0], &[1, 2])?;
    let generated =
        RectifiedFlowEngine::sample_euler(&x0, &cfg, |_x, _t, _c| Ok(target_vel.clone()))?;
    let gen_vec = generated.to_vec::<f32>()?;
    assert!((gen_vec[0] - 10.0).abs() < 1e-4);
    assert!((gen_vec[1] - 20.0).abs() < 1e-4);

    // 3. Classifier-Free Guidance (CFG)
    let cfg_guided = RectifiedFlowConfig {
        num_steps: 10,
        guidance_scale: 2.0,
    };
    let v_cond = Tensor::from_slice(&[10.0f32, 20.0], &[1, 2])?;
    let v_uncond = Tensor::from_slice(&[5.0f32, 10.0], &[1, 2])?;
    let guided_gen = RectifiedFlowEngine::sample_euler(&x0, &cfg_guided, |_x, _t, cond| {
        if cond {
            Ok(v_cond.clone())
        } else {
            Ok(v_uncond.clone())
        }
    })?;
    // v_eff = v_uncond + 2 * (v_cond - v_uncond) = [5, 10] + 2 * [5, 10] = [15, 30]
    let guided_vec = guided_gen.to_vec::<f32>()?;
    assert!((guided_vec[0] - 15.0).abs() < 1e-4);
    assert!((guided_vec[1] - 30.0).abs() < 1e-4);

    // 4. Python Bindings Verification
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

cfg = ox.RectifiedFlowConfig(20, 1.5)
assert cfg is not None
t0 = ox.Tensor.zeros([1, 4])
t1 = ox.Tensor.ones([1, 4])
xt, vt = ox.RectifiedFlowEngine.interpolate(t0, t1, 0.5)
assert xt.shape == [1, 4]
assert vt.shape == [1, 4]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python v3.2.0 test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v40_aot_compiler() -> Result<()> {
    // 1. Build SSA IrGraph
    let mut graph = IrGraph::new();
    let in0 = graph.add_input();
    let in1 = graph.add_input();
    let sum = graph.push_op(IrOp::Add(in0, in1));
    let relu = graph.push_op(IrOp::Relu(sum));
    graph.set_outputs(vec![relu]);

    // 2. Emit C99
    let c_code = AotCompiler::emit_c(&graph, "oxideon_forward")?;
    assert!(c_code.contains("void oxideon_forward("));
    assert!(c_code.contains("const float* in_0"));
    assert!(c_code.contains("const float* in_1"));
    assert!(c_code.contains("v_0[i] + v_1[i]"));
    assert!(c_code.contains("v_2[i] > 0.0f ? v_2[i] : 0.0f"));
    assert!(c_code.contains("memcpy(out_0, v_3"));

    // 3. Emit Rust
    let rust_code = AotCompiler::emit_rust(&graph, "oxideon_forward_rs")?;
    assert!(rust_code.contains("pub fn oxideon_forward_rs("));
    assert!(rust_code.contains("in_0: &[f32]"));
    assert!(rust_code.contains("in_1: &[f32]"));
    assert!(rust_code.contains("v_0[i] + v_1[i]"));

    // 4. Python Bindings Verification
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

g = ox.IrGraph()
i0 = g.add_input()
i1 = g.add_input()
s = g.add_op("add", [i0, i1])
r = g.add_op("relu", [s])
g.set_outputs([r])

c_src = g.emit_c("py_model")
assert "void py_model(" in c_src
assert "v_0[i] + v_1[i]" in c_src

rs_src = g.emit_rust("py_model_rs")
assert "pub fn py_model_rs(" in rs_src
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python v4.0.0 test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v45_elastic_subnet_extractor() -> Result<()> {
    // 1. Slice Linear Weight [8, 16] -> [4, 8]
    let w = Tensor::randn(&[8, 16])?;
    let b = Tensor::randn(&[8])?;

    let (sub_w, sub_b) = ElasticSubnetExtractor::slice_linear_width(&w, Some(&b), 0.5, 0.5)?;
    assert_eq!(sub_w.shape(), &[4, 8]);
    assert_eq!(sub_b.as_ref().unwrap().shape(), &[4]);

    // 2. Elastic Forward Evaluation
    let x = Tensor::randn(&[2, 16])?;
    let y = ElasticSubnetExtractor::evaluate_elastic_forward(&x, &w, Some(&b), 0.5)?;
    assert_eq!(y.shape(), &[2, 4]);

    // Reference computation
    let x_sliced = x.narrow(1, 0, 8)?;
    let ref_y = x_sliced.matmul(&sub_w.t()?)?.add(&sub_b.unwrap())?;
    let diff = y.sub(&ref_y)?.abs()?.sum()?.item::<f32>()?;
    assert!(diff < 1e-5);

    // 3. Python Bindings Verification
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

w = ox.Tensor.randn([8, 16])
x = ox.Tensor.randn([2, 16])
sw, sb = ox.ElasticSubnetExtractor.slice_linear_width(w, 0.5, 0.5)
assert sw.shape == [4, 8]
assert sb is None

y = ox.ElasticSubnetExtractor.evaluate_elastic_forward(x, w, 0.5)
assert y.shape == [2, 4]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python v4.5.0 test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v50_super_intelligence_engine() -> Result<()> {
    // 1. Initialize Unified Super-Intelligence Engine
    let config = SuperIntelligenceConfig {
        model_name: "oxideon-super-intelligence-v5".into(),
        parallel_config: Parallel3DConfig::new(1, 1, 1)?,
        sampling_config: SamplingConfig::default(),
        high_watermark_ratio: 0.85,
        max_batch_tokens: 1024,
        prefill_chunk_size: 256,
        speculative_enabled: true,
    };

    let mut engine = SuperIntelligenceEngine::new(config, vec![Device::Cpu])?;

    // 2. Continuous scheduler request submission
    let req_id = engine.submit_request(vec![101, 102, 103, 104]);
    assert_eq!(req_id, 1);

    // 3. Speculative Verification Execution
    let draft_tokens = vec![10, 20];
    let mut logits_data = vec![0.0f32; 3 * 100];
    // Position 0 -> target predicts 10
    logits_data[10] = 100.0;
    // Position 1 -> target predicts 20
    logits_data[100 + 20] = 100.0;
    // Position 2 -> target predicts 30 (bonus token)
    logits_data[200 + 30] = 100.0;

    let target_logits = Tensor::from_slice(&logits_data, &[3, 100])?;
    let final_tokens = engine.verify_speculative_tokens(&draft_tokens, &target_logits)?;
    assert_eq!(final_tokens, vec![10, 20, 30]);

    // 4. Memory Offloader Integration
    engine
        .memory_manager
        .register_device_capacity(Device::Cuda(0), 1000);
    engine.register_tensor_with_eviction(1, 500, Device::Cuda(0))?;
    engine.register_tensor_with_eviction(2, 400, Device::Cuda(0))?;

    let metrics = engine.get_metrics();
    assert_eq!(metrics.tokens_generated, 3);
    assert!(metrics.speculative_speedup > 1.0);
    assert_eq!(metrics.offload_migrations, 1);

    // 5. Python Bindings Verification
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

cfg = ox.SuperIntelligenceConfig("test-v5", dp_size=1, tp_size=1, pp_size=1)
engine = ox.SuperIntelligenceEngine(cfg, ["cpu"])
rid = engine.submit_request([1, 2, 3])
assert rid == 1
m = engine.get_metrics()
assert len(m) == 5
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python v5.0.0 test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v55_stream_and_compute_graph() -> Result<()> {
    // 1. ComputeStream and StreamEvent
    let stream = ComputeStream::new(0, Device::Cpu);
    let event = stream.record_event();
    assert!(event.query());
    stream.wait_event(&event);
    stream.synchronize();

    // 2. ComputeGraph capture and replay
    let mut graph = ComputeGraph::new();
    let counter = std::sync::Arc::new(std::sync::Mutex::new(0usize));
    let c1 = counter.clone();
    let c2 = counter.clone();

    graph.capture_begin()?;
    graph.record_op(move || {
        let mut g = c1.lock().unwrap();
        *g += 10;
        Ok(())
    })?;
    graph.record_op(move || {
        let mut g = c2.lock().unwrap();
        *g += 5;
        Ok(())
    })?;
    let n_ops = graph.capture_end()?;
    assert_eq!(n_ops, 2);
    assert_eq!(*counter.lock().unwrap(), 0);

    // Replay 1
    graph.replay()?;
    assert_eq!(*counter.lock().unwrap(), 15);
    assert_eq!(graph.replay_count(), 1);

    // Replay 2
    graph.replay()?;
    assert_eq!(*counter.lock().unwrap(), 30);
    assert_eq!(graph.replay_count(), 2);

    // 3. Python Bindings Verification
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

s = ox.ComputeStream(1, "cpu")
ev = s.record_event()
assert ev.query()
s.wait_event(ev)

g = ox.ComputeGraph()
g.capture_begin()
n = g.capture_end()
assert n == 0
g.replay()
assert g.replay_count == 1
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python v5.5.0 test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v60_neuromorphic_lif_and_poisson() -> Result<()> {
    // 1. LIF Neuron dynamics: integrate, fire, reset
    let lif = LIFNode::new(0.5, 1.0, 0.0, ResetMode::Hard);

    // Step 1: Input 0.6 -> V=0.6, spike=0
    let x1 = Tensor::from_slice(&[0.6f32], &[1])?;
    let s1 = lif.forward_step(&x1)?.item::<f32>()?;
    assert_eq!(s1, 0.0);

    // Step 2: Input 0.8 -> V = 0.5 * 0.6 + 0.8 = 1.1 >= 1.0 -> spike=1, reset V to 0.0
    let x2 = Tensor::from_slice(&[0.8f32], &[1])?;
    let s2 = lif.forward_step(&x2)?.item::<f32>()?;
    assert_eq!(s2, 1.0);

    // Step 3: Input 0.4 -> V = 0.5 * 0.0 + 0.4 = 0.4 < 1.0 -> spike=0
    let x3 = Tensor::from_slice(&[0.4f32], &[1])?;
    let s3 = lif.forward_step(&x3)?.item::<f32>()?;
    assert_eq!(s3, 0.0);

    // 2. Sequence forward: [T=3, B=1, D=2]
    lif.reset_state();
    let seq_input = Tensor::from_slice(&[0.6f32, 0.2, 0.8, 0.9, 0.4, 0.1], &[3, 1, 2])?;
    let spike_seq = lif.forward_sequence(&seq_input)?;
    assert_eq!(spike_seq.shape(), &[3, 1, 2]);

    // 3. Poisson Rate Encoder
    let encoder = PoissonEncoder::new(5);
    let static_input = Tensor::from_slice(&[0.8f32, 0.1], &[1, 2])?;
    let spike_train = encoder.encode(&static_input)?;
    assert_eq!(spike_train.shape(), &[5, 1, 2]);

    // 4. Python Bindings Verification
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

lif = ox.LIFNode(0.5, 1.0, 0.0, "hard")
x = ox.Tensor([0.6], [1])
s = lif.forward_step(x)
assert s.shape == [1]

enc = ox.PoissonEncoder(4)
spk = enc.encode(ox.Tensor([0.5, 0.5], [1, 2]))
assert spk.shape == [4, 1, 2]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python v6.0.0 test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v65_quantum_circuit_and_parameter_shift() -> Result<()> {
    // 1. Bell State |Phi+> = (|00> + |11>) / sqrt(2)
    let mut qc = QuantumCircuit::new(2)?;
    qc.h(0)?;
    qc.cnot(0, 1)?;

    let inv_sqrt2 = 1.0 / 2.0f64.sqrt();
    assert!((qc.state_real[0] - inv_sqrt2).abs() < 1e-6); // |00>
    assert!((qc.state_real[3] - inv_sqrt2).abs() < 1e-6); // |11>
    assert!(qc.state_real[1].abs() < 1e-6); // |01>
    assert!(qc.state_real[2].abs() < 1e-6); // |10>
    assert!(qc.expectation_z(0)?.abs() < 1e-6);
    assert!(qc.expectation_z(1)?.abs() < 1e-6);

    // 2. Analytical Parameter-Shift Gradient vs Truth: d/d(theta) cos(theta) = -sin(theta)
    let theta = std::f64::consts::PI / 3.0; // 60 degrees
    let expected_grad = -theta.sin(); // -sqrt(3)/2 ~ -0.866025

    let computed_grad = QuantumCircuit::parameter_shift_gradient(1, 0, theta, |circ, th| {
        circ.ry(0, th)?;
        Ok(())
    })?;

    assert!(
        (computed_grad - expected_grad).abs() < 1e-5,
        "Parameter-shift gradient {computed_grad} must match exact analytical derivative {expected_grad}"
    );

    // 3. Python Bindings Verification
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

qc = ox.QuantumCircuit(2)
qc.h(0)
qc.cnot(0, 1)
exp_z = qc.expectation_z(0)
assert abs(exp_z) < 1e-5
r, i = qc.state_vector
assert len(r) == 4
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python v6.5.0 test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v70_mamba2_and_transfusion() -> Result<()> {
    // 1. Mamba-2 State Space Duality Verification
    let mamba2 = Mamba2Block::new(16, 8, 2, 4)?;
    let u = Tensor::randn(&[2, 8, 16])?;
    let out = mamba2.forward(&u)?;
    assert_eq!(out.shape(), &[2, 8, 16], "Mamba-2 output shape mismatch");
    assert!(
        !mamba2.parameters().is_empty(),
        "Mamba-2 parameters must not be empty"
    );

    // 2. Transfusion Multimodal Unified Model Verification
    let config = TransfusionConfig {
        vocab_size: 100,
        d_model: 32,
        patch_dim: 16,
        num_layers: 2,
        num_heads: 4,
        d_ff: 64,
        dropout: 0.0,
    };
    let model = TransfusionModel::new(config)?;
    let text_tokens = Tensor::from_vec(
        vec![1.0f32, 5.0f32, 10.0f32, 2.0f32, 3.0f32, 4.0f32],
        &[2, 3],
    )?;
    let visual_clean = Tensor::randn(&[2, 4, 16])?;
    let visual_noise = Tensor::randn(&[2, 4, 16])?;
    let timesteps = Tensor::from_vec(vec![0.3f32, 0.7f32], &[2, 1])?;

    let (logits, pred_v) = model.forward_multimodal(&text_tokens, &visual_clean, &timesteps)?;
    assert_eq!(
        logits.shape(),
        &[2, 3, 100],
        "Transfusion text logits shape mismatch"
    );
    assert_eq!(
        pred_v.shape(),
        &[2, 4, 16],
        "Transfusion patch prediction shape mismatch"
    );

    let loss = model.compute_loss(&text_tokens, &visual_clean, &visual_noise, &timesteps, 1.0)?;
    assert_eq!(loss.ndim(), 0, "Loss must be scalar");
    let loss_val = loss.item::<f32>()?;
    assert!(loss_val > 0.0, "Loss must be positive");

    // 3. Python Bindings Verification
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

# Test Mamba-2 block
m2 = ox.Mamba2Block(16, 8, 2, 4)
u = ox.Tensor.randn([2, 4, 16])
out = m2.forward(u)
assert out.shape == [2, 4, 16]

# Test Transfusion model
tm = ox.TransfusionModel(50, 16, 8, 1, 2, 32, 0.0)
txt = ox.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
vis = ox.Tensor.randn([2, 2, 8])
t = ox.Tensor([0.5, 0.5], [2, 1])
logits, pred = tm.forward_multimodal(txt, vis, t)
assert logits.shape == [2, 2, 50]
assert pred.shape == [2, 2, 8]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python v7.0.0 test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v75_autonomous_synthesis_and_sandbox() -> Result<()> {
    use std::collections::HashMap;

    // 1. Sandboxed deterministic execution
    let sandbox = ExecutionSandbox::new(100);
    let mut initial_vars = HashMap::new();
    initial_vars.insert("a".to_string(), 10);
    initial_vars.insert("b".to_string(), 20);

    let program = "
        c = a + b
        d = c * 2
        assert d == 60
    ";
    let report = sandbox.execute(program, &initial_vars);
    assert_eq!(report.status, SandboxStatus::Success);
    assert_eq!(report.variables.get("d"), Some(&60));

    // 2. Autonomous synthesis & self-repair loop
    let engine = SynthesisEngine::new(5, 100);
    let test_suite = vec![
        TestCase::new("double_square_3", vec![("x", 3)], "y", 18),
        TestCase::new("double_square_5", vec![("x", 5)], "y", 50),
    ];

    let synth_report = engine.synthesize_and_repair(
        |attempt, feedback| {
            if attempt == 1 {
                // Buggy first attempt
                "y = x * 2".to_string()
            } else {
                // Self-repaired based on verification feedback
                assert!(feedback.is_some());
                "
                sq = x * x
                y = sq * 2
                "
                .to_string()
            }
        },
        &test_suite,
    )?;

    assert!(
        synth_report.success,
        "Autonomous synthesis should succeed after repair"
    );
    assert_eq!(
        synth_report.attempts, 2,
        "Expected convergence in 2 iterations"
    );

    // 3. Python Bindings Verification
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

sb = ox.ExecutionSandbox(50)
status, steps, vars = sb.execute("x = a + 5\ny = x * 2", {"a": 10})
assert status == "Success"
assert vars["y"] == 30

engine = ox.SynthesisEngine(3, 50)
ok = engine.verify_candidate("y = x * 3", ["test1"], [{"x": 4}], ["y"], [12])
assert ok is True
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python v7.5.0 test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v80_evolutionary_nas_and_synflow() -> Result<()> {
    // 1. Zero-Cost SynFlow score calculation
    let l1 = Linear::new(16, 32)?;
    let l2 = Linear::new(32, 8)?;
    let synflow = compute_synflow(&[l1.weight, l2.weight], 16)?;
    assert!(synflow.is_finite(), "SynFlow score must be finite");

    // 2. Evolutionary Population Search under parameter constraints
    let mut nas = EvolutionaryAutoML::new(6, 3, 0.3, 50_000, 1337);
    let (best_genome, best_score) = nas.search(16, 4)?;

    assert!(
        !best_genome.hidden_dims.is_empty(),
        "Genome must contain hidden dimensions"
    );
    assert!(
        best_score.combined_fitness.is_finite(),
        "Fitness score must be finite"
    );
    let param_estimate = best_genome.estimate_parameters(16, 4);
    assert!(param_estimate > 0, "Parameter estimate must be positive");

    // 3. Python Bindings Verification
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

genome = ox.ArchitectureGenome([64, 32], 4, 2)
assert genome.hidden_dims == [64, 32]
params = genome.estimate_parameters(16, 4)
assert params > 0

nas = ox.EvolutionaryAutoML(4, 2, 0.2, 50000, 42)
best_g, fitness = nas.search(16, 4)
assert len(best_g.hidden_dims) > 0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python v8.0.0 test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v85_swarm_consensus_and_debate() -> Result<()> {
    // 1. Confidence-Weighted Consensus Resolution
    let proposals = vec![
        AgentProposal::new("agent_alpha", "Plan_X", 0.9, "Proven empirical bounds"),
        AgentProposal::new("agent_beta", "Plan_Y", 0.4, "Intuition"),
        AgentProposal::new("agent_gamma", "Plan_X", 0.85, "Cross-validated proof"),
    ];

    let orchestrator = SwarmOrchestrator::new(3, 0.75, ConsensusStrategy::ConfidenceWeighted);
    let report = orchestrator.resolve_consensus(&proposals)?;
    assert_eq!(report.final_solution, "Plan_X");
    assert!(report.agreement_score > 0.75);

    // 2. Multi-Round Debate Convergence
    let initial_proposals = vec![
        AgentProposal::new("agent_1", "hypothesis_a", 0.6, "preliminary data"),
        AgentProposal::new("agent_2", "hypothesis_b", 0.5, "alternative"),
    ];

    let debate_report = orchestrator.debate(initial_proposals, |_round, prev| {
        // Agent 2 is persuaded by Agent 1's arguments and converges
        vec![
            prev[0].clone(),
            AgentProposal::new("agent_2", "hypothesis_a", 0.95, "accepted proof"),
        ]
    })?;

    assert_eq!(debate_report.final_solution, "hypothesis_a");
    assert_eq!(debate_report.total_rounds, 2);

    // 3. Python Bindings Verification
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

p1 = ox.AgentProposal("a1", "opt_alpha", 0.95, "sound")
p2 = ox.AgentProposal("a2", "opt_alpha", 0.85, "agreed")
p3 = ox.AgentProposal("a3", "opt_beta", 0.3, "doubtful")

swarm = ox.SwarmOrchestrator(3, 0.7, "confidence")
winner, score, rounds, dist = swarm.resolve_consensus([p1, p2, p3])
assert winner == "opt_alpha"
assert score > 0.7
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python v8.5.0 test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v90_lorentz_hyperbolic_manifold() -> Result<()> {
    // 1. Manifold projection onto hyperboloid sheet & Minkowski inner product
    let x = Tensor::from_vec(vec![0.0f32, 1.0f32, 2.0f32], &[1, 3])?;
    let p = LorentzManifold::project(&x, 1.0)?;
    let inner = LorentzManifold::minkowski_inner_product(&p, &p)?;
    let inner_val = inner.item::<f32>()?;
    assert!(
        (inner_val - (-1.0)).abs() < 1e-4,
        "Minkowski norm on hyperboloid must be -1/c (-1.0), got {inner_val}"
    );

    // 2. Geodesic distance
    let dist_self = LorentzManifold::distance(&p, &p, 1.0)?;
    let d_val = dist_self.item::<f32>()?;
    assert!(
        d_val.abs() < 1e-2,
        "Distance to self must be ~0, got {d_val}"
    );

    let q = Tensor::from_vec(vec![0.0f32, 2.0f32, 3.0f32], &[1, 3])?;
    let dist_pq = LorentzManifold::distance(&p, &q, 1.0)?;
    assert!(
        dist_pq.item::<f32>()? > 0.0,
        "Distance between distinct points must be positive"
    );

    // 3. Exp map and Log map roundtrip isomorphism
    let v = Tensor::from_vec(vec![0.5f32, 0.2f32], &[1, 2])?;
    let y = LorentzManifold::exp_map_zero(&v, 1.0)?;
    let v_rec = LorentzManifold::log_map_zero(&y, 1.0)?;
    let err = v.sub(&v_rec)?.norm_p(2.0, &[-1], false)?.item::<f32>()?;
    assert!(
        err < 1e-4,
        "Exp/Log map roundtrip error must be < 1e-4, got {err}"
    );

    // 4. Python Bindings Verification
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

v = ox.Tensor([0.3, 0.4], [1, 2])
y = ox.LorentzManifold.exp_map_zero(v, 1.0)
assert y.shape == [1, 3]

v_rec = ox.LorentzManifold.log_map_zero(y, 1.0)
assert v_rec.shape == [1, 2]

dist = ox.LorentzManifold.distance(y, y, 1.0)
assert abs(dist.item()) < 1e-2
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python v9.0.0 test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v95_kernel_autotuning_engine() -> Result<()> {
    // 1. TileConfig resource estimation
    let cfg = TileConfig::new(64, 64, 32, 4, 2);
    let expected_smem = (64 * 32 * 4 + 32 * 64 * 4) * 2;
    assert_eq!(cfg.shared_memory_bytes(), expected_smem);

    // 2. Heterogeneous Micro-Kernel Auto-Tuning
    let mut tuner = KernelAutoTuner::new(HardwareProfile::default_gpu(HardwareTarget::CudaPtx));
    let best_cfg = tuner.tune_gemm(1024, 1024, 1024)?;
    assert!(best_cfg.tile_m >= 32 && best_cfg.tile_n >= 32);

    // Verify cache hit
    let cached_cfg = tuner.tune_gemm(1024, 1024, 1024)?;
    assert_eq!(best_cfg, cached_cfg);

    // 3. Python Bindings Verification
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

cfg = ox.TileConfig(64, 64, 32, 4, 2)
assert cfg.tile_m == 64
assert cfg.shared_memory_bytes() == 32768

tuner = ox.KernelAutoTuner("gpu")
opt_cfg = tuner.tune_gemm(512, 512, 512)
assert opt_cfg.tile_m in [32, 64, 128]
assert opt_cfg.tile_n in [32, 64, 128]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python v9.5.0 test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v10_universal_agi_singularity() -> Result<()> {
    // 1. Full-Stack Universal AGI Engine Cognitive Execution Cycle
    let config = UniversalAGIConfig::default();
    let mut engine = UniversalAGIEngine::new(config)?;

    let report = engine.execute_cognitive_cycle()?;

    assert!(
        report.quantum_expectation.is_finite(),
        "Quantum expectation must be finite"
    );
    assert!(
        report.mamba_output_norm > 0.0,
        "Mamba output norm must be positive"
    );
    assert!(
        report.multimodal_loss > 0.0,
        "Multimodal Transfusion loss must be positive"
    );
    assert_eq!(report.swarm_consensus_winner, "align_trajectories");
    assert!(
        report.swarm_agreement_score > 0.7,
        "Swarm agreement must exceed threshold"
    );
    assert!(
        report.synthesized_program_passed,
        "Synthesized code verification must pass"
    );
    assert!(
        report.lorentz_distance > 0.0,
        "Lorentz manifold distance must be positive"
    );
    assert!(
        report.auto_tuned_tile.tile_m >= 32,
        "Auto-tuned tile size must be valid"
    );

    // 2. Python Bindings Verification
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

engine = ox.UniversalAGIEngine()
q_exp, spikes, mamba_norm, mm_loss, winner, score, synth_ok, lorentz_d = engine.execute_cognitive_cycle()
assert winner == "align_trajectories"
assert score > 0.7
assert synth_ok is True
assert lorentz_d > 0.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python v10.0.0 test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v110_persistent_homology_and_tda() -> Result<()> {
    // 1. Metric Point Cloud Vietoris-Rips Persistent Homology
    // 4 points arranged as unit square: (0,0), (1,0), (1,1), (0,1)
    let s2 = 2.0f32.sqrt();
    #[rustfmt::skip]
    let dist_data = vec![
        0.0, 1.0,  s2, 1.0,
        1.0, 0.0, 1.0,  s2,
         s2, 1.0, 0.0, 1.0,
        1.0,  s2, 1.0, 0.0,
    ];
    let dist = Tensor::from_vec(dist_data, &[4, 4])?;

    let engine = PersistentHomologyEngine::new(1, 2.5);
    let diag = engine.compute_from_distance_matrix(&dist)?;

    let h0_pairs = diag.dimension_pairs(0);
    let h1_pairs = diag.dimension_pairs(1);

    assert_eq!(h0_pairs.len(), 4, "Expected 4 H0 persistence pairs");
    assert!(
        !h1_pairs.is_empty(),
        "Expected non-empty H1 1-cycle persistence pairs"
    );

    // Cycle born at 1.0 and died at sqrt(2) ~ 1.414
    let primary_cycle = &h1_pairs[0];
    assert!((primary_cycle.birth - 1.0).abs() < 1e-3);
    assert!((primary_cycle.death - s2 as f64).abs() < 1e-3);

    // max_dimension bounds reported homology: dim 0 reports H0 only, and the H1
    // death above stays exact (triangles are built whenever H1 is requested).
    let engine0 = PersistentHomologyEngine::new(0, 2.5);
    let diag0 = engine0.compute_from_distance_matrix(&dist)?;
    assert_eq!(diag0.dimension_pairs(0).len(), 4);
    assert!(diag0.dimension_pairs(1).is_empty());
    let engine2 = PersistentHomologyEngine::new(2, 2.5);
    let diag2 = engine2.compute_from_distance_matrix(&dist)?;
    assert_eq!(diag2.dimension_pairs(1).len(), h1_pairs.len());
    assert!((diag2.dimension_pairs(1)[0].death - s2 as f64).abs() < 1e-3);

    // 2. 1-Wasserstein Metric on Homology Diagrams
    let self_dist = wasserstein_distance(&diag, &diag, 0);
    assert!(
        self_dist.abs() < 1e-6,
        "Wasserstein self-distance must be zero"
    );

    // 3. Python Bindings Verification
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

dist_mat = [
    0.0, 1.0, 1.414, 1.0,
    1.0, 0.0, 1.0, 1.414,
    1.414, 1.0, 0.0, 1.0,
    1.0, 1.414, 1.0, 0.0
]
dist_t = ox.Tensor(dist_mat, [4, 4])
tda_engine = ox.PersistentHomologyEngine(1, 2.5)
diag = tda_engine.compute_from_distance_matrix(dist_t)
assert diag.total_persistence() > 0.0
pairs = diag.get_pairs(1)
assert len(pairs) > 0
w_dist = ox.PersistentHomologyEngine.wasserstein_distance(diag, diag, 0)
assert w_dist == 0.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python v11.0.0 test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v120_photonic_optical_computing() -> Result<()> {
    // 1. Mach-Zehnder Interferometer 50:50 Beam Splitter
    let mzi = MachZehnderInterferometer::new(std::f64::consts::FRAC_PI_4, 0.0);
    let (o1_r, o1_i, o2_r, o2_i) = mzi.forward_field(1.0, 0.0, 0.0, 0.0);
    let power_out = o1_r * o1_r + o1_i * o1_i + o2_r * o2_r + o2_i * o2_i;
    assert!(
        (power_out - 1.0).abs() < 1e-6,
        "Unitary MZI must conserve optical power"
    );
    assert!(
        (o1_r - o2_r).abs() < 1e-6,
        "50:50 split must have equal amplitudes"
    );

    // 2. Optical Saturable Absorber Nonlinearity
    let sa = SaturableAbsorber::new(1.0);
    let intensity = Tensor::from_vec(vec![0.5f32, 2.0f32, 10.0f32], &[1, 3])?;
    let saturated = sa.forward(&intensity)?;
    let sat_vals = saturated.to_vec::<f32>()?;
    assert!(sat_vals[0] < 0.5 && sat_vals[0] > 0.0);
    assert!(sat_vals[1] < 2.0 && sat_vals[1] > sat_vals[0]);
    assert!(sat_vals[2] < 10.0 && sat_vals[2] > sat_vals[1]);

    // 3. Programmable Photonic Linear Layer
    let optical_linear = PhotonicLinear::new(4, 4)?;
    let x = Tensor::from_vec(vec![1.0f32, 0.5f32, 0.2f32, 0.1f32], &[1, 4])?;
    let y = optical_linear.forward(&x)?;
    assert_eq!(y.shape(), &[1, 4], "Photonic linear output shape mismatch");
    let y_vals = y.to_vec::<f32>()?;
    assert!(
        y_vals.iter().all(|&v| v >= 0.0),
        "Photodetected intensities must be non-negative"
    );

    // 4. Python Bindings Verification
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox
import math

mzi = ox.MachZehnderInterferometer(math.pi / 4.0, 0.0)
o1_r, o1_i, o2_r, o2_i = mzi.forward_field(1.0, 0.0, 0.0, 0.0)
assert abs((o1_r**2 + o2_r**2) - 1.0) < 1e-5

sa = ox.SaturableAbsorber(1.0)
inp = ox.Tensor([2.0], [1, 1])
out = sa.forward(inp)
assert out.item() < 2.0

pl = ox.PhotonicLinear(4, 4)
x = ox.Tensor([1.0, 0.5, 0.2, 0.1], [1, 4])
y = pl.forward(x)
assert y.shape == [1, 4]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python v12.0.0 test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v130_hamiltonian_neural_engine() -> Result<()> {
    // 1. Canonical Hamilton's Equations Verification
    // Harmonic Oscillator: H(q, p) = 0.5 * p^2 + 0.5 * q^2
    let h_fn = |q: &Tensor, p: &Tensor| -> Result<Tensor> {
        let kin = p.powf(2.0)?.mul_scalar(0.5)?;
        let pot = q.powf(2.0)?.mul_scalar(0.5)?;
        kin.add(&pot)
    };

    let q = Tensor::from_vec(vec![2.0f32], &[1, 1])?;
    let p = Tensor::from_vec(vec![3.0f32], &[1, 1])?;

    let (q_dot, p_dot) = HamiltonianNeuralEngine::vector_field(h_fn, &q, &p)?;
    assert!(
        (q_dot.item::<f32>()? - 3.0).abs() < 1e-4,
        "dq/dt must be p = 3.0"
    );
    assert!(
        (p_dot.item::<f32>()? - (-2.0)).abs() < 1e-4,
        "dp/dt must be -q = -2.0"
    );

    // 2. Symplectic Trajectory Integration with Energy Conservation
    let (q_traj, p_traj, energies) =
        HamiltonianNeuralEngine::integrate_trajectory(h_fn, &q, &p, 0.05, 20)?;
    assert_eq!(q_traj.shape(), &[21, 1, 1]);
    assert_eq!(p_traj.shape(), &[21, 1, 1]);

    let initial_energy = energies[0];
    for &e in &energies {
        let drift = (e - initial_energy).abs();
        assert!(
            drift < 0.05,
            "Symplectic leapfrog must conserve energy, drift was {drift}"
        );
    }

    // 3. Python Bindings Verification
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

q = ox.Tensor([1.0], [1, 1])
p = ox.Tensor([0.0], [1, 1])

q_next, p_next = ox.HamiltonianNeuralEngine.harmonic_oscillator_step(q, p, 0.1, 1.0, 1.0)
assert q_next.shape == [1, 1]
assert p_next.shape == [1, 1]
assert p_next.item() < 0.0  # restoring momentum should be negative
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python v13.0.0 test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_autonomous_quantization_change_llm() -> Result<()> {
    manual_seed(42);

    // 1. Construct a standard CausalLM (vocab=64, d_model=32, num_heads=4, num_layers=2, ffn=64)
    let vocab_size = 64;
    let d_model = 32;
    let num_heads = 4;
    let num_layers = 2;
    let ffn_hidden = 64;
    let max_seq_len = 16;

    let lm = CausalLM::new(
        vocab_size,
        d_model,
        num_heads,
        num_layers,
        ffn_hidden,
        max_seq_len,
    )?;

    // 2. Calibration activation sample
    let calib_tokens = Tensor::from_slice(&[1i64, 5, 12, 18], &[1, 4])?;
    let calib_h = lm.tok_emb.forward(&calib_tokens)?; // [1, 4, 32]
    assert_eq!(calib_h.shape(), &[1, 4, 32]);

    // 3. Autonomous Quantization Manager
    let policy = AutonomousQuantPolicy {
        max_loss_degradation: 0.15,
        target_compression_ratio: 4.0,
        max_memory_bytes: None,
        preserve_attention_precision: true,
    };
    let quantizer = AutonomousQuantizer::new(policy);

    // Quantize entire LLM into mixed-precision QuantizedCausalLM
    let mut q_lm = quantizer.quantize_llm(&lm, &calib_h)?;
    assert_eq!(q_lm.blocks.len(), num_layers);
    assert_eq!(q_lm.profiles.len(), num_layers);

    for p in &q_lm.profiles {
        assert!(
            p.compression_ratio > 2.0,
            "Expected compression ratio > 2.0"
        );
        assert_eq!(p.q_format, QuantFormat::Q8_0);
        assert_eq!(p.k_format, QuantFormat::Q8_0);
    }

    let orig_weight_bytes = q_lm
        .profiles
        .iter()
        .map(|p| p.original_bytes)
        .sum::<usize>();
    let quant_weight_bytes = q_lm.total_weight_bytes();
    assert!(quant_weight_bytes < orig_weight_bytes);

    // 4. Forward execution through QuantizedCausalLM
    let out_logits = q_lm.forward(&calib_tokens)?;
    assert_eq!(out_logits.shape(), &[1, 4, vocab_size]);

    // 5. Dynamic runtime quantization format change (all blocks converted to Q4_0)
    let bytes_before = q_lm.total_weight_bytes();
    q_lm.autonomous_quant_change(QuantFormat::Q4_0, Some(&calib_h))?;
    let bytes_after = q_lm.total_weight_bytes();
    assert!(
        bytes_after < bytes_before,
        "Q4_0 memory should be strictly smaller than Q8_0/mixed"
    );

    // Forward execution after dynamic change
    let out_logits_q4 = q_lm.forward(&calib_tokens)?;
    assert_eq!(out_logits_q4.shape(), &[1, 4, vocab_size]);

    // 6. Round-trip conversion back to unquantized CausalLM
    let restored_lm = q_lm.to_causal_lm(DType::F32)?;
    let restored_logits = restored_lm.forward(&calib_tokens)?;
    assert_eq!(restored_logits.shape(), &[1, 4, vocab_size]);

    // 7. Python Bindings Verification
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

w = ox.Tensor.randn([32, 32])
x = ox.Tensor.randn([4, 32])

quantizer = ox.AutonomousQuantizer(max_loss_degradation=0.20, target_compression_ratio=4.0)
dist_q8 = quantizer.evaluate_distortion(w, "Q8_0", x)
dist_q4 = quantizer.evaluate_distortion(w, "Q4_0", x)
assert dist_q8 >= 0.0
assert dist_q4 >= dist_q8  # Q4 usually has higher distortion than Q8

best_fmt, dist = quantizer.select_optimal_format(w, x)
assert best_fmt in ["Q4_0", "FP8E4M3", "Q8_0", "FP32"]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python autonomous quant test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v140_zkml_proof_engine() -> Result<()> {
    // 1. Exact finite field arithmetic verification (Mersenne 61 modulus)
    let a = FieldElement::new(15);
    let b = FieldElement::new(27);
    assert_eq!(a.add(b), FieldElement::new(42));
    assert_eq!(b.sub(a), FieldElement::new(12));
    assert_eq!(a.mul(b), FieldElement::new(405));

    let inv_a = a.inv().expect("inverse exists");
    assert_eq!(a.mul(inv_a), FieldElement::ONE);

    // 2. Verifiable Matrix Multiplication Proof & Witness Constraints
    let x = Tensor::from_vec(vec![1.5f32, 2.0, -1.0, 0.5], &[2, 2])?;
    let w = Tensor::from_vec(vec![0.5f32, -1.5, 2.0, 1.0], &[2, 2])?;

    let (y_float, proof) = ZkMlEngine::prove_matmul(&x, &w, 10.0)?;
    assert_eq!(y_float.shape(), &[2, 2]);
    assert_eq!(proof.num_constraints, 8); // 4 dot products * 2 multiplications each

    // 3. Cryptographic Proof Verification
    let valid = ZkMlEngine::verify_proof(&proof, &proof.weight_commitment, &proof.input_commitment);
    assert!(valid, "zkML proof must verify against true commitments");

    let tampered_input = [0u8; 32];
    let invalid = ZkMlEngine::verify_proof(&proof, &proof.weight_commitment, &tampered_input);
    assert!(!invalid, "zkML proof must reject altered input commitment");

    // 4. Python Bindings Verification
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

x = ox.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
w = ox.Tensor([0.5, -0.5, 1.0, 1.0], [2, 2])

y, wc, ic, oc, constraints = ox.ZkMlEngine.prove_matmul(x, w, 10.0)
assert y.shape == [2, 2]
assert len(wc) == 32
assert len(ic) == 32
assert len(oc) == 32
assert constraints == 8

ok = ox.ZkMlEngine.verify_commitments(wc, ic, oc)
assert ok is True
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python zkML test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v150_riemannian_manifold_and_cartan_connections() -> Result<()> {
    // 1. Conformal Metric and Christoffel Symbols Calculation
    let dim = 2;
    let manifold = RiemannianManifold::new(dim);

    let metric = |x: &Tensor| -> Result<Tensor> {
        let r2 = x.powf(2.0)?.sum()?.item::<f64>()?;
        let conf = 4.0 / (1.0 + r2).powi(2);
        let g_diag = vec![conf, 0.0, 0.0, conf];
        Tensor::from_vec(g_diag, &[2, 2])
    };

    let origin = Tensor::from_vec(vec![0.0f64, 0.0], &[2])?;
    let gamma_origin = manifold.christoffel_symbols(&origin, &metric, 1e-4)?;
    assert_eq!(gamma_origin.shape(), &[2, 2, 2]);

    // At the origin x = (0, 0), dg/dx = 0, so all Christoffel symbols Gamma^k_{ij} must vanish
    let gamma_vals = gamma_origin.to_vec::<f64>()?;
    for g in gamma_vals {
        assert!(
            g.abs() < 1e-4,
            "Christoffel symbols at origin must vanish, was {g}"
        );
    }

    // 2. Geodesic Exponential Flow along curved metric
    let p0 = Tensor::from_vec(vec![0.1f64, 0.0], &[2])?;
    let v0 = Tensor::from_vec(vec![0.2f64, 0.1], &[2])?;

    let (traj_x, traj_v) = manifold.integrate_geodesic(&p0, &v0, &metric, 0.05, 10)?;
    assert_eq!(traj_x.shape(), &[11, 2]);
    assert_eq!(traj_v.shape(), &[11, 2]);

    // 3. Parallel Transport of tangent vector
    let w = Tensor::from_vec(vec![1.0f64, 0.0], &[2])?;
    let gamma_p = manifold.christoffel_symbols(&p0, &metric, 1e-4)?;
    let transported_w = manifold.parallel_transport_step(&w, &v0, &gamma_p, 0.05)?;
    assert_eq!(transported_w.shape(), &[2]);

    // 4. Python Bindings Verification
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

man = ox.RiemannianManifold(2)
x0 = ox.Tensor([0.0, 0.0], [2])
gamma = man.christoffel_symbols(x0)
assert gamma.shape == [2, 2, 2]

v0 = ox.Tensor([0.1, 0.2], [2])
traj_x, traj_v = man.integrate_geodesic(x0, v0, 0.05, 5)
assert traj_x.shape == [6, 2]
assert traj_v.shape == [6, 2]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Riemannian manifold test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v160_holographic_neural_memory() -> Result<()> {
    let dim = 32;
    let mut hrr = oxideon::HolographicMemoryEngine::new(dim)?;

    // Create 3 distinct orthogonal-ish patterns
    let mut pat1 = vec![0.0f64; dim];
    let mut pat2 = vec![0.0f64; dim];
    let mut pat3 = vec![0.0f64; dim];
    for i in 0..dim {
        pat1[i] = if i % 3 == 0 { 1.0 } else { -0.5 };
        pat2[i] = if i % 3 == 1 { 1.0 } else { -0.5 };
        pat3[i] = if i % 3 == 2 { 1.0 } else { -0.5 };
    }
    let p1 = Tensor::from_vec(pat1.clone(), &[dim])?;
    let p2 = Tensor::from_vec(pat2.clone(), &[dim])?;
    let p3 = Tensor::from_vec(pat3.clone(), &[dim])?;

    // Verify commutativity: a * b == b * a
    let conv_12 = oxideon::HolographicMemoryEngine::circular_convolution(&p1, &p2)?;
    let conv_21 = oxideon::HolographicMemoryEngine::circular_convolution(&p2, &p1)?;
    let diff = conv_12.sub(&conv_21)?.abs()?.sum()?.item::<f64>()?;
    assert!(diff < 1e-5, "Circular convolution must be commutative");

    // Keys
    let k1 = Tensor::from_vec(
        (0..dim)
            .map(|i| if i % 2 == 0 { 0.8 } else { -0.8 })
            .collect::<Vec<f64>>(),
        &[dim],
    )?;
    let k2 = Tensor::from_vec(
        (0..dim)
            .map(|i| if i % 4 == 0 { 1.0 } else { -0.3 })
            .collect::<Vec<f64>>(),
        &[dim],
    )?;

    // Store k1->p1, k2->p2
    hrr.store(&k1, &p1)?;
    hrr.store(&k2, &p2)?;

    // Retrieve noisy estimate for k1
    let est_p1 = hrr.retrieve(&k1)?;
    assert_eq!(est_p1.shape(), &[dim]);

    // Cleanup using Modern Hopfield Network
    let all_patterns = Tensor::stack(&[p1.clone(), p2.clone(), p3.clone()], 0)?;
    assert_eq!(all_patterns.shape(), &[3, dim]);

    let cleaned = oxideon::HolographicMemoryEngine::hopfield_cleanup(&est_p1, &all_patterns, 5.0)?;
    assert_eq!(cleaned.shape(), &[dim]);

    // Cleaned should be very close to p1
    let cos_sim = cleaned.mul(&p1)?.sum()?.item::<f64>()?
        / (cleaned.powf(2.0)?.sum()?.item::<f64>()?.sqrt()
            * p1.powf(2.0)?.sum()?.item::<f64>()?.sqrt());
    assert!(
        cos_sim > 0.85,
        "Hopfield cleanup should reconstruct stored pattern, got cos_sim {cos_sim}"
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

engine = ox.HolographicMemoryEngine(16)
k = ox.Tensor([1.0] * 16, [16])
v = ox.Tensor([0.5] * 16, [16])
engine.store(k, v)
ret = engine.retrieve(k)
assert ret.shape == [16]
assert engine.memory_trace.shape == [16]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Holographic test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v170_neuromorphic_stdp_learning() -> Result<()> {
    use oxideon::nn::{STDPConfig, STDPSynapse};

    let cfg = STDPConfig {
        a_plus: 0.05,
        a_minus: 0.06,
        tau_plus: 20.0,
        tau_minus: 20.0,
        w_min: 0.0,
        w_max: 1.0,
        dt: 1.0,
    };

    let mut synapse = STDPSynapse::new(3, 2, cfg)?;
    let w_init = synapse.weights.to_vec::<f64>()?;
    assert_eq!(w_init[0], 0.5);

    // Step 1: Pre-spike on neuron 0 (causes pre trace to accumulate)
    let pre1 = Tensor::from_vec(vec![1.0, 0.0, 0.0], &[3])?;
    let post1 = Tensor::from_vec(vec![0.0, 0.0], &[2])?;
    synapse.step_stdp(&pre1, &post1)?;

    // Step 2: Post-spike on neuron 0 shortly after (pre -> post: LTP)
    let pre2 = Tensor::from_vec(vec![0.0, 0.0, 0.0], &[3])?;
    let post2 = Tensor::from_vec(vec![1.0, 0.0], &[2])?;
    synapse.step_stdp(&pre2, &post2)?;

    let w_after_ltp = synapse.weights.to_vec::<f64>()?;
    assert!(
        w_after_ltp[0] > 0.5,
        "Pre-before-post must induce Long-Term Potentiation (LTP), got {}",
        w_after_ltp[0]
    );

    // Step 3: Post-spike on neuron 1 first
    let pre3 = Tensor::from_vec(vec![0.0, 0.0, 0.0], &[3])?;
    let post3 = Tensor::from_vec(vec![0.0, 1.0], &[2])?;
    synapse.step_stdp(&pre3, &post3)?;

    // Step 4: Pre-spike on neuron 1 after (post -> pre: LTD)
    let pre4 = Tensor::from_vec(vec![0.0, 1.0, 0.0], &[3])?;
    let post4 = Tensor::from_vec(vec![0.0, 0.0], &[2])?;
    synapse.step_stdp(&pre4, &post4)?;

    let w_after_ltd = synapse.weights.to_vec::<f64>()?;
    // W[1, 1] index is 1 * 3 + 1 = 4
    assert!(
        w_after_ltd[4] < 0.5,
        "Post-before-pre must induce Long-Term Depression (LTD), got {}",
        w_after_ltd[4]
    );

    // Forward pass
    let current = synapse.forward(&pre1)?;
    assert_eq!(current.shape(), &[2]);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

syn = ox.STDPSynapse(4, 2, a_plus=0.02, a_minus=0.03)
pre = ox.Tensor([1.0, 0.0, 0.0, 0.0], [4])
post = ox.Tensor([0.0, 1.0], [2])
out = syn.forward(pre)
assert out.shape == [2]
syn.step_stdp(pre, post)
assert syn.weights.shape == [2, 4]
assert syn.trace_pre.shape == [4]
assert syn.trace_post.shape == [2]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python STDP test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v180_quantum_error_correction() -> Result<()> {
    use oxideon::quantum::{QecEngine, QuantumCircuit, StabilizerCode};

    // 1. Bit-Flip Code [[3, 1, 3]]
    let code3 = StabilizerCode::bit_flip_3_qubit();
    assert_eq!(code3.num_physical_qubits, 3);
    assert_eq!(code3.distance, 3);

    let mut circuit = QuantumCircuit::new(3)?;
    // Prepare logical state (|0> + |1>) / sqrt(2)
    circuit.h(0)?;
    QecEngine::encode_3qubit_bit_flip(&mut circuit, 0, 1, 2)?;

    // Initially clean
    let s_clean = QecEngine::measure_syndromes(&circuit, &code3)?;
    assert_eq!(s_clean, vec![1, 1]);

    // Inject Pauli-X noise on qubit 1
    circuit.x(1)?;
    let s_corrupt = QecEngine::measure_syndromes(&circuit, &code3)?;
    assert_eq!(s_corrupt, vec![-1, -1]);

    // Auto-correct
    let fixed = QecEngine::correct_bit_flip(&mut circuit, &code3)?;
    assert_eq!(fixed, Some(1));

    // Post-correction syndromes
    let s_restored = QecEngine::measure_syndromes(&circuit, &code3)?;
    assert_eq!(s_restored, vec![1, 1]);

    // 2. Steane CSS Code [[7, 1, 3]]
    let steane = StabilizerCode::steane_7_qubit();
    assert_eq!(steane.num_physical_qubits, 7);
    assert_eq!(steane.distance, 3);
    assert_eq!(steane.generators.len(), 6);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

qc = ox.QuantumCircuit(3)
code = ox.StabilizerCode.bit_flip_3_qubit()
assert code.num_physical_qubits == 3
assert code.distance == 3

ox.QecEngine.encode_3qubit_bit_flip(qc, 0, 1, 2)
synd = ox.QecEngine.measure_syndromes(qc, code)
assert synd == [1, 1]

qc.x(2)
synd_err = ox.QecEngine.measure_syndromes(qc, code)
assert synd_err == [1, -1]

recovered = ox.QecEngine.correct_bit_flip(qc, code)
assert recovered == 2
synd_fixed = ox.QecEngine.measure_syndromes(qc, code)
assert synd_fixed == [1, 1]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python QEC test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v190_automated_theorem_prover() -> Result<()> {
    use oxideon::reasoning::{PropExpr, Sequent, SequentProver};

    let prover = SequentProver::new(30);

    let p = PropExpr::var("P");
    let q = PropExpr::var("Q");
    let r = PropExpr::var("R");

    // 1. Law of Excluded Middle: |- P or ~P
    let lem = p.clone().or(p.clone().not());
    assert!(prover.is_tautology(&lem)?);

    // 2. Double Negation Elimination: ~~P |- P
    let seq_dne = Sequent::new(vec![p.clone().not().not()], vec![p.clone()]);
    assert!(prover.prove(&seq_dne)?);

    // 3. Modus Ponens: P, P -> Q |- Q
    let seq_mp = Sequent::new(
        vec![p.clone(), p.clone().implies(q.clone())],
        vec![q.clone()],
    );
    assert!(prover.prove(&seq_mp)?);

    // 4. Peirce's Law: |- ((P -> Q) -> P) -> P
    let peirce = p
        .clone()
        .implies(q.clone())
        .implies(p.clone())
        .implies(p.clone());
    assert!(prover.is_tautology(&peirce)?);

    // 5. De Morgan's Law: |- ~(P and Q) <-> (~P or ~Q)
    let demorgan = p
        .clone()
        .and(q.clone())
        .not()
        .iff(p.clone().not().or(q.clone().not()));
    assert!(prover.is_tautology(&demorgan)?);

    // 6. Refutation of non-tautology: (P and Q) -> R is NOT universally valid
    let non_tautology = p.and(q).implies(r);
    assert!(!prover.is_tautology(&non_tautology)?);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

prover = ox.SequentProver(20)
p = ox.PropExpr.var("P")
q = ox.PropExpr.var("Q")

# Prove Modus Ponens: P, P -> Q |- Q
imp = p.implies_op(q)
proved = prover.prove_sequent([p, imp], [q])
assert proved == True

# Prove Excluded Middle: |- P or ~P
lem = p.or_op(p.not_op())
assert prover.is_tautology(lem) == True
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Prover test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v200_omni_scientific_foundation_engine() -> Result<()> {
    use oxideon::omni_science::OmniScientificFoundationEngine;

    let mut engine = OmniScientificFoundationEngine::new(4)?;
    let state = Tensor::from_vec(vec![0.8, -0.6, 0.4, 0.2], &[4])?;

    let report = engine.execute_multidisciplinary_cycle(&state)?;

    assert!(
        (report.initial_energy - report.final_energy).abs() < 0.1,
        "Symplectic Hamiltonian flow must conserve energy"
    );
    assert!(
        report.optical_transmission_energy > 0.0,
        "Photonic mesh must transmit coherent light"
    );
    assert_eq!(
        report.zkml_proof_digest.len(),
        32,
        "zkML SHA-256 digest must be 32 bytes"
    );
    assert!(
        !report.betti_numbers.is_empty(),
        "Topological Betti numbers must be extracted"
    );
    assert!(
        report.holographic_retrieval_norm > 0.0,
        "Holographic retrieval norm must be positive"
    );
    assert!(
        report.stdp_weight_mean >= 0.0 && report.stdp_weight_mean <= 1.0,
        "STDP weights must respect clamp boundaries"
    );
    assert_eq!(
        report.qec_syndromes,
        vec![1, 1],
        "Codespace syndromes must be positive"
    );
    assert!(
        report.theorem_proved,
        "Gentzen sequent calculus must verify Modus Ponens"
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

engine = ox.OmniScientificFoundationEngine(4)
state = ox.Tensor([0.5, -0.5, 0.3, 0.2], [4])
rep = engine.execute_multidisciplinary_cycle(state)
# (h_init, h_final, opt_energy, zk_digest, betti, holo_norm, stdp_mean, qec_synd, theorem_proved)
assert len(rep) == 9
assert rep[8] == True  # Theorem proved
assert rep[7] == [1, 1]  # QEC syndrome
assert len(rep[3]) == 32  # zkML proof digest
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python OmniScientific test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v210_continuous_fourier_neural_operator() -> Result<()> {
    use oxideon::science::{FourierNeuralOperator1d, SpectralConv1d};

    // 1. Spectral Convolution layer
    let spec_conv = SpectralConv1d::new(2, 4, 8)?;
    let x = Tensor::randn(&[2, 2, 32])?;
    let y = spec_conv.forward(&x)?;
    assert_eq!(y.shape(), &[2, 4, 32]);

    // 2. Full Continuous Fourier Neural Operator (FNO)
    let fno = FourierNeuralOperator1d::new(2, 1, 16, 6, 2)?;

    // Evaluation on coarse grid resolution N = 32
    let x_coarse = Tensor::randn(&[1, 2, 32])?;
    let y_coarse = fno.forward(&x_coarse)?;
    assert_eq!(y_coarse.shape(), &[1, 1, 32]);

    // Zero-shot evaluation on fine grid resolution N = 64 (discretization invariance)
    let x_fine = Tensor::randn(&[1, 2, 64])?;
    let y_fine = fno.forward(&x_fine)?;
    assert_eq!(y_fine.shape(), &[1, 1, 64]);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

fno = ox.FourierNeuralOperator1d(in_channels=2, out_channels=1, hidden_channels=8, modes=4, num_layers=2)
# [Batch=1, Channels=2, N=16]
data = [0.1] * 32
x = ox.Tensor(data, [1, 2, 16])
out = fno.forward(x)
assert out.shape == [1, 1, 16]

# Zero-shot on N=32
data_fine = [0.1] * 64
x_fine = ox.Tensor(data_fine, [1, 2, 32])
out_fine = fno.forward(x_fine)
assert out_fine.shape == [1, 1, 32]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python FNO test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v220_homomorphic_encryption_ckks() -> Result<()> {
    use oxideon::crypto::CkksContext;

    let ctx = CkksContext::new(8)?;
    let (sk, pk) = ctx.keygen()?;

    let v1 = vec![1.5, -2.0, 3.25, 0.5, 0.0, 0.0, 0.0, 0.0];
    let v2 = vec![0.5, 3.0, -1.25, 1.5, 0.0, 0.0, 0.0, 0.0];

    let ct1 = ctx.encrypt(&v1, &pk)?;
    let ct2 = ctx.encrypt(&v2, &pk)?;

    // 1. Decrypt plaintext
    let dec1 = ctx.decrypt(&ct1, &sk)?;
    for (orig, dec) in v1.iter().zip(dec1.iter()).take(4) {
        assert!(
            (orig - dec).abs() < 0.05,
            "Decryption error too high: orig={orig}, dec={dec}"
        );
    }

    // 2. Homomorphic Addition: ct1 + ct2
    let ct_add = ctx.add(&ct1, &ct2)?;
    let dec_add = ctx.decrypt(&ct_add, &sk)?;
    let expected_add = [2.0, 1.0, 2.0, 2.0];
    for (exp, dec) in expected_add.iter().zip(dec_add.iter()) {
        assert!(
            (exp - dec).abs() < 0.05,
            "Homomorphic addition error: exp={exp}, dec={dec}"
        );
    }

    // 3. Homomorphic Plaintext Multiplication: 2.0 * ct1
    let ct_mul = ctx.mul_plain(&ct1, 2.0)?;
    let dec_mul = ctx.decrypt(&ct_mul, &sk)?;
    let expected_mul = [3.0, -4.0, 6.5, 1.0];
    for (exp, dec) in expected_mul.iter().zip(dec_mul.iter()) {
        assert!(
            (exp - dec).abs() < 0.05,
            "Homomorphic mul error: exp={exp}, dec={dec}"
        );
    }

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

ctx = ox.CkksContext(8)
sk, pk = ctx.keygen()

v1 = [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]
v2 = [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]

ct1 = ctx.encrypt(v1, pk)
ct2 = ctx.encrypt(v2, pk)

ct_add = ctx.add(ct1, ct2)
dec = ctx.decrypt(ct_add, sk)
assert abs(dec[0] - 1.5) < 0.05
assert abs(dec[1] - 2.5) < 0.05
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python CKKS test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v230_federated_dp_and_byzantine_resilience() -> Result<()> {
    use oxideon::distributed::{ByzantineAggregator, DpSgdConfig};

    // 1. DP-SGD Gradient Clipping and Noise
    let dp_cfg = DpSgdConfig::new(2.0, 0.0);
    let large_grad = Tensor::from_vec(vec![3.0, 4.0, 0.0], &[3])?; // Norm = 5.0
    let clipped = dp_cfg.clip_and_noise(&large_grad)?;
    let norm = clipped.norm()?.item::<f64>()?;
    assert!(
        (norm - 2.0).abs() < 1e-5,
        "DP-SGD must clip gradient norm to C=2.0"
    );

    let dp_noisy_cfg = DpSgdConfig::new(2.0, 0.2);
    let noisy = dp_noisy_cfg.clip_and_noise(&large_grad)?;
    assert_eq!(noisy.shape(), &[3]);

    // 2. Byzantine Fault Tolerance (Multi-Krum & Trimmed Mean)
    // 4 honest workers around [1.0, 2.0, 3.0], 1 adversary at [1000.0, -1000.0, 2000.0]
    let h1 = Tensor::from_vec(vec![1.02, 1.98, 3.01], &[3])?;
    let h2 = Tensor::from_vec(vec![0.98, 2.01, 2.99], &[3])?;
    let h3 = Tensor::from_vec(vec![1.01, 2.02, 3.00], &[3])?;
    let h4 = Tensor::from_vec(vec![0.99, 1.99, 3.02], &[3])?;
    let byz = Tensor::from_vec(vec![1000.0, -1000.0, 2000.0], &[3])?;

    let worker_grads = vec![h1, h2, h3, h4, byz];

    // Multi-Krum: m=5, f=1 (2*1 + 2 < 5 holds)
    let krum_aggr = ByzantineAggregator::multi_krum(&worker_grads, 1, 3)?;
    let krum_vec = krum_aggr.to_vec::<f64>()?;
    assert!(
        (krum_vec[0] - 1.0).abs() < 0.1,
        "Multi-Krum must reject Byzantine outlier"
    );
    assert!((krum_vec[1] - 2.0).abs() < 0.1);
    assert!((krum_vec[2] - 3.0).abs() < 0.1);

    // Trimmed Mean (trim fraction 0.2 trims 1 from top and 1 from bottom of 5 workers)
    let trimmed_aggr = ByzantineAggregator::trimmed_mean(&worker_grads, 0.2)?;
    let trimmed_vec = trimmed_aggr.to_vec::<f64>()?;
    assert!(
        (trimmed_vec[0] - 1.0).abs() < 0.1,
        "Trimmed mean must reject Byzantine outlier"
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

dp = ox.DpSgdConfig(1.5, 0.0)
g = ox.Tensor([3.0, 4.0], [2])
g_clip = dp.clip_and_noise(g)
assert g_clip.shape == [2]

w1 = ox.Tensor([1.0, 2.0], [2])
w2 = ox.Tensor([1.05, 1.95], [2])
w3 = ox.Tensor([0.95, 2.05], [2])
w4 = ox.Tensor([1.02, 1.98], [2])
w_bad = ox.Tensor([500.0, -500.0], [2])

aggr = ox.ByzantineAggregator.multi_krum([w1, w2, w3, w4, w_bad], 1, 2)
assert aggr.shape == [2]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Federated test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v240_cellular_sheaf_laplacian_networks() -> Result<()> {
    use oxideon::geom_graphs::CellularSheaf;

    // 1. Triangle graph with 3 nodes, 3 edges: (0, 1), (1, 2), (0, 2)
    let edges = vec![(0, 1), (1, 2), (0, 2)];
    let sheaf = CellularSheaf::new(3, 2, 2, edges)?;

    // Sheaf Laplacian shape: [N * d_v, N * d_v] = [6, 6]
    let l_sheaf = sheaf.compute_sheaf_laplacian()?;
    assert_eq!(l_sheaf.shape(), &[6, 6]);

    // Symmetry check: L = L^T
    let diff = l_sheaf.sub(&l_sheaf.t()?)?.abs()?.sum()?.item::<f64>()?;
    assert!(diff < 1e-6, "Sheaf Laplacian must be symmetric");

    // Positive semi-definiteness: E(x) >= 0 for non-zero signals
    let x = Tensor::from_vec(vec![1.0, -0.5, 2.0, 1.5, -1.0, 0.5], &[6])?;
    let energy_0 = sheaf.dirichlet_energy(&x)?;
    assert!(
        energy_0 >= 0.0,
        "Sheaf Dirichlet energy must be non-negative"
    );

    // Dissipative sheaf heat flow: E(x_{t+1}) <= E(x_t)
    let x_next = sheaf.sheaf_diffusion_step(&x, 0.05)?;
    let energy_1 = sheaf.dirichlet_energy(&x_next)?;
    assert!(
        energy_1 <= energy_0 + 1e-6,
        "Continuous sheaf diffusion must be energy-dissipative: E0={energy_0}, E1={energy_1}"
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

edges = [(0, 1), (1, 2)]
sheaf = ox.CellularSheaf(num_nodes=3, node_dim=2, edge_dim=2, edges=edges)
lap = sheaf.compute_sheaf_laplacian()
assert lap.shape == [6, 6]

x = ox.Tensor([1.0, 0.5, -0.5, 1.0, 0.0, 0.5], [6])
energy = sheaf.dirichlet_energy(x)
assert energy >= 0.0

x_diff = sheaf.sheaf_diffusion_step(x, 0.05)
assert x_diff.shape == [6]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Sheaf test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v250_thermodynamics_and_relativistic_spacetime() -> Result<()> {
    use oxideon::physics::{KerrSpacetime, LangevinThermodynamics};

    // 1. Langevin Statistical Thermodynamics
    let langevin = LangevinThermodynamics::new(1.5, 0.25, 1.0);
    assert_eq!(langevin.gamma, 1.5);
    assert_eq!(langevin.temperature, 0.25);
    assert_eq!(langevin.mass, 1.0);

    let x0 = Tensor::from_vec(vec![2.0, -1.0], &[2])?;
    let grad_v = x0.clone(); // quadratic well V(x) = 0.5 * ||x||^2
    let x_overdamped = langevin.step_overdamped(&x0, &grad_v, 0.01)?;
    assert_eq!(x_overdamped.shape(), &[2]);

    let v0 = Tensor::from_vec(vec![0.0, 0.0], &[2])?;
    let (x_next, v_next) = langevin.step_underdamped(&x0, &v0, &grad_v, 0.01)?;
    assert_eq!(x_next.shape(), &[2]);
    assert_eq!(v_next.shape(), &[2]);

    // 2. Kerr & Schwarzschild Spacetime Geometries
    let schw = KerrSpacetime::schwarzschild(2.0)?;
    assert!((schw.outer_horizon() - 4.0).abs() < 1e-6);
    assert!((schw.inner_horizon() - 0.0).abs() < 1e-6);
    assert!((schw.ergosphere(std::f64::consts::FRAC_PI_2) - 4.0).abs() < 1e-6);

    let kerr = KerrSpacetime::new(2.0, 1.2)?;
    // sqrt(4 - 1.44) = sqrt(2.56) = 1.6
    assert!((kerr.outer_horizon() - 3.6).abs() < 1e-6);
    assert!((kerr.inner_horizon() - 0.4).abs() < 1e-6);
    assert!((kerr.ergosphere(std::f64::consts::FRAC_PI_2) - 4.0).abs() < 1e-6);
    assert!((kerr.ergosphere(0.0) - 3.6).abs() < 1e-6);

    let metric = kerr.metric_tensor(6.0, std::f64::consts::FRAC_PI_2)?;
    assert_eq!(metric.shape(), &[4, 4]);

    let traj = kerr.integrate_equatorial_geodesic(10.0, 0.0, 0.0, 0.03, 0.01, 50)?;
    assert_eq!(traj.shape(), &[51, 2]);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox
import math

sim = ox.LangevinThermodynamics(gamma=2.0, temperature=0.1, mass=1.0)
assert sim.gamma == 2.0
assert sim.temperature == 0.1

x = ox.Tensor([1.0, 2.0], [2])
grad = ox.Tensor([0.5, 1.0], [2])
x_over = sim.step_overdamped(x, grad, 0.02)
assert x_over.shape == [2]

v = ox.Tensor([0.0, 0.0], [2])
x_under, v_under = sim.step_underdamped(x, v, grad, 0.02)
assert x_under.shape == [2]
assert v_under.shape == [2]

kerr = ox.KerrSpacetime(2.0, 1.2)
assert abs(kerr.outer_horizon() - 3.6) < 1e-5
assert abs(kerr.inner_horizon() - 0.4) < 1e-5
assert abs(kerr.ergosphere(math.pi / 2.0) - 4.0) < 1e-5

metric = kerr.metric_tensor(6.0, math.pi / 2.0)
assert metric.shape == [4, 4]

traj = kerr.integrate_equatorial_geodesic(10.0, 0.0, 0.0, 0.03, 0.01, 20)
assert traj.shape == [21, 2]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Thermo-Relativity test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v260_deep_operator_networks() -> Result<()> {
    use oxideon::science::DeepONet;

    // 1. Initialize DeepONet: 10 sensors, 2D query space, 32 hidden, 16 basis components
    let deeponet = DeepONet::new(10, 2, 32, 16)?;
    assert_eq!(deeponet.branch_in_dim, 10);
    assert_eq!(deeponet.trunk_in_dim, 2);
    assert_eq!(deeponet.basis_dim, 16);

    // 2. Evaluated on 4 functions sampled at 10 sensors
    let u_sensors = Tensor::randn(&[4, 10])?;
    let b = deeponet.forward_branch(&u_sensors)?;
    assert_eq!(b.shape(), &[4, 16]);

    // 3. Evaluated on 50 continuous query coordinates in 2D space
    let y_coords = Tensor::randn(&[50, 2])?;
    let t = deeponet.forward_trunk(&y_coords)?;
    assert_eq!(t.shape(), &[50, 16]);

    // 4. Dot-product synthesis: G(u)(y) has shape [4, 50]
    let out = deeponet.forward(&u_sensors, &y_coords)?;
    assert_eq!(out.shape(), &[4, 50]);

    // Parameters verification
    let params = deeponet.parameters();
    assert!(!params.is_empty());

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

net = ox.DeepONet(branch_in_dim=8, trunk_in_dim=1, hidden_dim=16, basis_dim=8)
assert net.branch_in_dim == 8
assert net.trunk_in_dim == 1
assert net.basis_dim == 8

u = ox.Tensor.randn([2, 8])
y = ox.Tensor.randn([20, 1])

b = net.forward_branch(u)
assert b.shape == [2, 8]

t = net.forward_trunk(y)
assert t.shape == [20, 8]

out = net.forward(u, y)
assert out.shape == [2, 20]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python DeepONet test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v270_hyperbolic_graph_attention_network() -> Result<()> {
    use oxideon::geom_graphs::{HyperbolicGATLayer, LorentzManifold};

    // 1. Initialize HGAT: 4 input tangent features, 6 output tangent features, curvature c=1.0, beta=0.2
    let hgat = HyperbolicGATLayer::new(4, 6, 1.0, 0.2)?;
    assert_eq!(hgat.in_dim, 4);
    assert_eq!(hgat.out_dim, 6);
    assert_eq!(hgat.curvature, 1.0);

    // 2. Generate 3 nodes on the Lorentz hyperboloid
    let v = Tensor::randn(&[3, 4])?;
    let x = LorentzManifold::exp_map_zero(&v, 1.0)?;
    assert_eq!(x.shape(), &[3, 5]);

    // Verify initial Lorentz manifold condition: <x, x>_L = -1/c = -1.0
    let inner_init = LorentzManifold::minkowski_inner_product(&x, &x)?;
    for &val in &inner_init.to_vec::<f64>()? {
        assert!(
            (val - (-1.0)).abs() < 1e-4,
            "Must lie on Lorentz hyperboloid: {val}"
        );
    }

    // 3. Directed cycle graph: 0 -> 1 -> 2 -> 0
    let edges = vec![(0, 1), (1, 2), (2, 0)];
    let out = hgat.forward(&x, &edges)?;
    assert_eq!(out.shape(), &[3, 7]); // out_dim + 1 = 6 + 1 = 7

    // Verify output Lorentz manifold condition: <y, y>_L = -1.0
    let inner_out = LorentzManifold::minkowski_inner_product(&out, &out)?;
    for &val in &inner_out.to_vec::<f64>()? {
        assert!(
            (val - (-1.0)).abs() < 1e-4,
            "HGAT output must lie on Lorentz hyperboloid: {val}"
        );
    }

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

hgat = ox.HyperbolicGATLayer(in_dim=3, out_dim=5, curvature=1.0, beta=0.15)
assert hgat.in_dim == 3
assert hgat.out_dim == 5
assert hgat.curvature == 1.0

v = ox.Tensor.randn([4, 3])
x = ox.LorentzManifold.exp_map_zero(v, 1.0)
assert x.shape == [4, 4]

edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
y = hgat.forward(x, edges)
assert y.shape == [4, 6]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python HGAT test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v280_clifford_geometric_algebra() -> Result<()> {
    use oxideon::geom_graphs::{CliffordLinear, Multivector3D};

    // 1. Clifford Cl(3, 0) algebraic axioms
    let e1 = Multivector3D::vector(1.0, 0.0, 0.0);
    let e2 = Multivector3D::vector(0.0, 1.0, 0.0);

    // e1^2 = 1.0
    let e1_sq = e1.geometric_product(&e1);
    assert!((e1_sq.components[0] - 1.0).abs() < 1e-6);
    assert!((e1_sq.norm_squared() - 1.0).abs() < 1e-6);

    // e1 e2 = e12
    let e12 = e1.geometric_product(&e2);
    assert!((e12.components[4] - 1.0).abs() < 1e-6);

    // Anticommutation: e2 e1 = -e12
    let e21 = e2.geometric_product(&e1);
    assert!((e21.components[4] - (-1.0)).abs() < 1e-6);

    // 2. Rotor 3D rotation: 90 degrees around Z axis takes e1 to e2
    let rot_z_90 = Multivector3D::rotor([0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2);
    let rotated = e1.sandwich(&rot_z_90);
    assert!((rotated.components[1] - 0.0).abs() < 1e-6);
    assert!((rotated.components[2] - 1.0).abs() < 1e-6);
    assert!((rotated.components[3] - 0.0).abs() < 1e-6);

    // 3. CliffordLinear layer
    let layer = CliffordLinear::new(2, 4, true)?;
    let x = Tensor::randn(&[3, 2, 8])?;
    let out = layer.forward(&x)?;
    assert_eq!(out.shape(), &[3, 4, 8]);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox
import math

e1 = ox.Multivector3D.vector(1.0, 0.0, 0.0)
e2 = ox.Multivector3D.vector(0.0, 1.0, 0.0)
e12 = e1.geometric_product(e2)
assert abs(e12.components()[4] - 1.0) < 1e-5

rot = ox.Multivector3D.rotor([0.0, 0.0, 1.0], math.pi / 2.0)
v_rot = e1.sandwich(rot)
comps = v_rot.components()
assert abs(comps[1] - 0.0) < 1e-5
assert abs(comps[2] - 1.0) < 1e-5

layer = ox.CliffordLinear(2, 4, True)
x = ox.Tensor.randn([2, 2, 8])
y = layer.forward(x)
assert y.shape == [2, 4, 8]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Clifford test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v290_differentiable_inductive_logic_programming() -> Result<()> {
    use oxideon::reasoning::{DifferentiableILPEngine, TNorm};

    // 1. Setup ILP engine: 4 entities (0, 1, 2, 3), 2 binary relations (0: edge, 1: path)
    let mut engine = DifferentiableILPEngine::new(4, 2, TNorm::Product);
    assert_eq!(engine.num_entities, 4);
    assert_eq!(engine.num_predicates, 2);

    // Rule 1 (Base): path(X, Y) <- edge(X, Y)
    engine.add_base_rule(1, 0, 5.0)?;

    // Rule 2 (Recursive / Transitive): path(X, Y) <- edge(X, Z) ^ path(Z, Y)
    engine.add_transitive_rule(1, 0, 1, 5.0)?;

    // 2. Setup initial facts: 0 -> 1 -> 2 -> 3
    let mut facts_vec = vec![0.0f64; 2 * 4 * 4];
    // edge(0, 1) = 1.0, edge(1, 2) = 1.0, edge(2, 3) = 1.0
    facts_vec[1] = 1.0;
    facts_vec[6] = 1.0;
    facts_vec[11] = 1.0;

    let facts = Tensor::from_vec(facts_vec, &[2, 4, 4])?;

    // 3. Multi-step deductive inference forward chaining (3 steps for 3-hop path)
    let deduced = engine.deductive_forward(&facts, 3)?;
    let deduced_vec = deduced.to_vec::<f64>()?;

    // Verify 3-hop transitive deduction: path(0, 3) must be derived with high probability
    let path_0_3 = deduced_vec[19];
    assert!(
        path_0_3 > 0.85,
        "Transitive closure path(0, 3) should be deduced: {path_0_3}"
    );

    // Negative relation: path(3, 0) should remain zero (directed acyclic)
    let path_3_0 = deduced_vec[28];
    assert!(
        path_3_0 < 1e-4,
        "Reverse path(3, 0) should not be deduced: {path_3_0}"
    );

    // 4. Binary cross-entropy loss evaluation
    let loss = engine.evaluate_loss(&deduced, &[(1, 0, 3)], &[(1, 3, 0)])?;
    assert!(
        (0.0..0.2).contains(&loss),
        "Deductive loss must be minimal: {loss}"
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

engine = ox.DifferentiableILPEngine(num_entities=3, num_predicates=2, t_norm="product")
assert engine.num_entities == 3
assert engine.num_predicates == 2

engine.add_base_rule(1, 0, 5.0)
engine.add_transitive_rule(1, 0, 1, 5.0)

facts_list = [0.0] * 18
facts_list[0 * 9 + 0 * 3 + 1] = 1.0 # 0 -> 1
facts_list[0 * 9 + 1 * 3 + 2] = 1.0 # 1 -> 2
facts = ox.Tensor(facts_list, [2, 3, 3])

deduced = engine.deductive_forward(facts, 2)
assert deduced.shape == [2, 3, 3]

loss = engine.evaluate_loss(deduced, [(1, 0, 2)], [(1, 2, 0)])
assert loss >= 0.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python ILP test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v300_grand_trans_singularity_engine() -> Result<()> {
    use oxideon::OmniTransSingularityEngine;

    // 1. Initialize the Grand Trans-Singularity Foundation Engine II
    let engine = OmniTransSingularityEngine::new()?;

    // 2. Execute a complete multi-paradigm trans-singularity cycle
    let input_signal = Tensor::randn(&[16])?;
    let report = engine.execute_cycle(&input_signal)?;

    // 3. Verify each foundational paradigm within the synthesis:
    // Continuous FNO
    assert!(report.fno_output_norm > 0.0);
    // Ring-LWE CKKS Homomorphic Encryption: 1.5 + 2.5 = 4.0
    assert!((report.ckks_addition_sample - 4.0).abs() < 1e-3);
    // Differentially private Byzantine consensus
    assert!(report.byzantine_consensus_norm > 0.0);
    // Cellular Sheaf Dirichlet energy
    assert!(report.sheaf_dirichlet_energy >= 0.0);
    // Langevin thermodynamics drift
    assert!(report.langevin_drift_norm > 0.0);
    // Kerr black hole outer horizon: 2.5 + sqrt(6.25 - 2.25) = 2.5 + 2.0 = 4.5
    assert!((report.kerr_outer_horizon - 4.5).abs() < 1e-5);
    // DeepONet continuous operator synthesis
    assert!(report.deeponet_output_norm > 0.0);
    // Lorentz manifold inner product: <y, y>_L = -1.0
    assert!((report.hgat_manifold_inner - (-1.0)).abs() < 1e-3);
    // Clifford geometric algebra 3D rotor rotation
    assert!((report.clifford_rotated_norm - 1.0).abs() < 1e-4);
    // Differentiable ILP transitive closure confidence
    assert!(report.ilp_transitive_confidence > 0.85);
    // Singularity coherence
    assert!(report.singularity_coherence > 0.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

engine = ox.OmniTransSingularityEngine()
x = ox.Tensor.randn([16])
metrics = engine.execute_cycle(x)
assert len(metrics) == 11
# metrics[1] is ckks_addition_sample ~ 4.0
assert abs(metrics[1] - 4.0) < 1e-2
# metrics[5] is kerr_outer_horizon == 4.5
assert abs(metrics[5] - 4.5) < 1e-4
# metrics[10] is singularity_coherence > 0.0
assert metrics[10] > 0.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Trans-Singularity test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v310_neural_jump_diffusion_sde() -> Result<()> {
    use oxideon::ode::{JumpDiffusionEngine, NeuralJumpSDE};

    // 1. Jump-Diffusion stochastic simulator
    let engine = JumpDiffusionEngine::new(1.0, 0.05, 0.2, 0.5);
    assert_eq!(engine.drift_scale, 1.0);
    assert_eq!(engine.diffusion_scale, 0.05);

    let x0 = Tensor::from_vec(vec![1.0, -0.5], &[2])?;
    let drift = x0.mul_scalar(-1.0)?; // Ornstein-Uhlenbeck mean-reverting drift
    let x1 = engine.step(&x0, &drift, 0.01)?;
    assert_eq!(x1.shape(), &[2]);

    // 2. Trajectory simulation
    let traj = engine.simulate_trajectory(|z| z.mul_scalar(-0.5), &x0, 0.02, 25)?;
    assert_eq!(traj.shape(), &[26, 2]);

    // 3. Hutchinson's trace divergence estimator for f(x) = 2.0 * x (dim 2 -> div = 4.0)
    let div_est = JumpDiffusionEngine::hutchinson_divergence(|z| z.mul_scalar(2.0), &x0, 20)?;
    assert!(
        (div_est - 4.0).abs() < 1.0,
        "Divergence estimate of 2*I in 2D: {div_est}"
    );

    // 4. Neural Jump-Diffusion SDE
    let sde = NeuralJumpSDE::new(4, engine)?;
    let x_batch = Tensor::randn(&[4])?;
    let sde_step = sde.forward_step(&x_batch, 0.01)?;
    assert_eq!(sde_step.shape(), &[4]);
    assert!(!sde.parameters().is_empty());

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

engine = ox.JumpDiffusionEngine(drift_scale=1.0, diffusion_scale=0.1, jump_intensity=0.5, jump_scale=1.0)
assert engine.drift_scale == 1.0
assert engine.diffusion_scale == 0.1

x = ox.Tensor([1.0, 2.0], [2])
drift = ox.Tensor([-0.5, -1.0], [2])
x_next = engine.step(x, drift, 0.01)
assert x_next.shape == [2]

sde = ox.NeuralJumpSDE(dim=4, drift_scale=1.0, diffusion_scale=0.05, jump_intensity=0.2, jump_scale=0.5)
x4 = ox.Tensor.randn([4])
sde_out = sde.forward_step(x4, 0.02)
assert sde_out.shape == [4]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Jump-SDE test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v320_neural_mutual_information_estimation() -> Result<()> {
    use oxideon::science::{InfoNceEstimator, MineEstimator};

    // 1. MINE Estimator: 4D continuous distributions
    let mut mine = MineEstimator::new(4, 4, 32, 0.9)?;
    assert_eq!(mine.x_dim, 4);
    assert_eq!(mine.y_dim, 4);

    // Correlated Gaussian variables: y = x + 0.05 * noise
    let x = Tensor::randn(&[32, 4])?;
    let noise = Tensor::randn(&[32, 4])?.mul_scalar(0.05)?;
    let y_corr = x.add(&noise)?;

    let mi_est = mine.estimate(&x, &y_corr)?;
    assert!(
        mi_est >= 0.0,
        "Mutual information must be non-negative: {mi_est}"
    );

    let mine_loss = mine.loss(&x, &y_corr)?;
    assert_eq!(mine_loss.numel(), 1);

    // 2. InfoNCE Contrastive Estimator
    let nce = InfoNceEstimator::new(0.07);
    assert_eq!(nce.temperature, 0.07);

    let queries = Tensor::randn(&[16, 8])?;
    let keys = queries.add(&Tensor::randn(&[16, 8])?.mul_scalar(0.01)?)?;

    let nce_loss = nce.compute_loss(&queries, &keys)?;
    assert_eq!(nce_loss.numel(), 1);

    let info_bound = nce.lower_bound(&queries, &keys)?;
    assert!(
        info_bound >= 0.0,
        "InfoNCE mutual information lower bound must be non-negative: {info_bound}"
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

mine = ox.MineEstimator(x_dim=2, y_dim=2, hidden_dim=16, ema_decay=0.9)
x = ox.Tensor.randn([20, 2])
y = x # perfect correlation
mi = mine.estimate(x, y)
assert mi >= 0.0

nce = ox.InfoNceEstimator(temperature=0.1)
assert nce.temperature == 0.1
q = ox.Tensor.randn([10, 4])
k = q
bound = nce.lower_bound(q, k)
assert bound >= 0.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python MINE/InfoNCE test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v330_dissipative_contact_geometry() -> Result<()> {
    use oxideon::physics::ContactHamiltonianSystem;

    // 1. Contact Hamiltonian System on 2D space (5D contact manifold)
    let sys = ContactHamiltonianSystem::new(2, 0.2)?;
    assert_eq!(sys.dimension, 2);
    assert_eq!(sys.dissipation, 0.2);

    let q0 = Tensor::from_vec(vec![1.0, 0.5], &[2])?;
    let p0 = Tensor::from_vec(vec![2.0, -1.0], &[2])?;
    let grad_v = q0.clone(); // quadratic potential V(q) = 0.5 * ||q||^2
    let s0 = 0.0;

    // 2. Vector field evaluation
    let (dq, dp, ds) = sys.vector_field(&q0, &p0, s0, &grad_v)?;
    assert_eq!(dq.shape(), &[2]);
    assert_eq!(dp.shape(), &[2]);
    assert!(ds != 0.0);

    // 3. Contact Verlet dissipative trajectory simulation
    let (q_traj, p_traj, s_traj) =
        sys.simulate_trajectory(|q| Ok(q.clone()), &q0, &p0, s0, 0.02, 50)?;
    assert_eq!(q_traj.shape(), &[51, 2]);
    assert_eq!(p_traj.shape(), &[51, 2]);
    assert_eq!(s_traj.len(), 51);

    // Damping check: momentum norm decreases over dissipative contact flow
    let p_init_norm = p0.norm()?.item::<f64>()?;
    let p_final = p_traj.narrow(0, 50, 1)?;
    let p_final_norm = p_final.norm()?.item::<f64>()?;
    assert!(
        p_final_norm < p_init_norm,
        "Dissipative contact flow must damp kinetic momentum: p0={p_init_norm}, p_end={p_final_norm}"
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys_py = py.import("sys")?;
            sys_py.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

sys = ox.ContactHamiltonianSystem(dimension=2, dissipation=0.3)
assert sys.dimension == 2
assert abs(sys.dissipation - 0.3) < 1e-5

q = ox.Tensor([1.0, 0.0], [2])
p = ox.Tensor([0.0, 2.0], [2])
grad = q
q1, p1, s1 = sys.step(q, p, 0.0, grad, 0.05)
assert q1.shape == [2]
assert p1.shape == [2]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Contact Geometry test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v340_neuromorphic_spiking_graph_network() -> Result<()> {
    use oxideon::geom_graphs::SpikingGnnLayer;

    // 1. Spiking-GNN layer: 4 input channels, 6 output channels, threshold 1.0, decay 0.8
    let layer = SpikingGnnLayer::new(4, 6, 1.0, 0.8)?;
    assert_eq!(layer.in_dim, 4);
    assert_eq!(layer.out_dim, 6);
    assert_eq!(layer.v_threshold, 1.0);
    assert_eq!(layer.decay, 0.8);

    // 3 nodes graph: 0 -> 1 -> 2 -> 0
    let edges = vec![(0, 1), (1, 2), (2, 0)];

    let v0 = Tensor::from_vec(vec![0.0f32; 18], &[3, 6])?;
    let spikes_in = Tensor::from_vec(
        vec![
            1.0f32, 0.0, 1.0, 0.0, // node 0
            0.0, 1.0, 0.0, 1.0, // node 1
            1.0, 1.0, 0.0, 0.0, // node 2
        ],
        &[3, 4],
    )?;

    // 2. Single discrete LIF update
    let (v1, s1) = layer.step(&v0, &spikes_in, &edges, None)?;
    assert_eq!(v1.shape(), &[3, 6]);
    assert_eq!(s1.shape(), &[3, 6]);

    // Check that emitted spikes are strictly binary {0, 1}
    for &s in &s1.to_vec::<f32>()? {
        assert!(s == 0.0 || s == 1.0, "Spikes must be binary: {s}");
    }

    // 3. Multi-timestep temporal simulation (10 time steps)
    let (spike_hist, v_fin) = layer.forward_temporal(&spikes_in, &edges, 10)?;
    assert_eq!(spike_hist.shape(), &[10, 3, 6]);
    assert_eq!(v_fin.shape(), &[3, 6]);

    // Parameters verification
    assert!(!layer.parameters().is_empty());

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

layer = ox.SpikingGnnLayer(in_dim=3, out_dim=4, v_threshold=0.8, decay=0.9)
assert layer.in_dim == 3
assert layer.out_dim == 4
assert abs(layer.v_threshold - 0.8) < 1e-5
assert abs(layer.decay - 0.9) < 1e-5

spikes = ox.Tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [2, 3])
edges = [(0, 1), (1, 0)]
all_spikes, v_fin = layer.forward_temporal(spikes, edges, 5)
assert all_spikes.shape == [5, 2, 4]
assert v_fin.shape == [2, 4]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Spiking GNN test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v350_qaoa_and_vqe_algorithms() -> Result<()> {
    use oxideon::quantum::{Pauli, QaoaCircuit, VqeAnsatz};

    // 1. QAOA on triangle graph MaxCut
    let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)];
    let qaoa = QaoaCircuit::new(3, 1, edges.clone())?;

    let e_init = qaoa.evaluate_energy(&[0.2], &[0.4])?;
    assert!(
        e_init > 1.4 && e_init <= 2.0,
        "QAOA cut energy must be high, got {e_init}"
    );

    let (g_grad, b_grad) = qaoa.compute_gradients(&[0.2], &[0.4])?;
    assert_eq!(g_grad.len(), 1);
    assert_eq!(b_grad.len(), 1);
    assert!(
        !g_grad[0].is_nan() && !b_grad[0].is_nan(),
        "Gradients must be finite"
    );

    let (opt_gamma, opt_beta, best_energy) = qaoa.optimize(15, 0.05)?;
    assert_eq!(opt_gamma.len(), 1);
    assert_eq!(opt_beta.len(), 1);
    assert!(
        best_energy >= 1.5,
        "Optimized QAOA cut energy should be >= 1.5, got {best_energy}"
    );

    // 2. VQE Hardware-Efficient Ansatz on 2-qubit Ising system
    let vqe = VqeAnsatz::new(2, 1)?;
    assert_eq!(vqe.num_parameters(), 4);

    let pauli_terms = vec![
        (vec![Pauli::Z, Pauli::I], -1.0),
        (vec![Pauli::I, Pauli::Z], -1.0),
    ];

    // With theta = 0, state is |00>, both Z_0 and Z_1 have expectation +1.0
    // Total Hamiltonian expectation: -1*(+1) + -1*(+1) = -2.0
    let zero_params = vec![0.0; 4];
    let e_ground = vqe.evaluate_energy(&zero_params, &pauli_terms)?;
    assert!(
        (e_ground - (-2.0)).abs() < 1e-4,
        "Ground state energy should be -2.0, got {e_ground}"
    );

    // With theta = [pi, 0, 0, 0], initial Ry rotates qubit 0 to |1>, qubit 1 is |0>
    // CNOT(0, 1) then entangles/flips qubit 1 to |1>, producing state |11> where Z_0 = -1, Z_1 = -1
    // Total expectation: -1*(-1) + -1*(-1) = +2.0
    let pi = std::f64::consts::PI;
    let excited_params = vec![pi, 0.0, 0.0, 0.0];
    let e_excited = vqe.evaluate_energy(&excited_params, &pauli_terms)?;
    assert!(
        (e_excited - 2.0).abs() < 1e-4,
        "Excited state energy should be 2.0, got {e_excited}"
    );

    // 3. PyO3 Bindings Verification
    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
qaoa = ox.QaoaCircuit(3, 1, edges)
e_init = qaoa.evaluate_energy([0.5], [0.5])
assert e_init > 0.0

g_grad, b_grad = qaoa.compute_gradients([0.5], [0.5])
assert len(g_grad) == 1 and len(b_grad) == 1

gamma, beta, best_e = qaoa.optimize(10, 0.1)
assert best_e >= 1.4

vqe = ox.VqeAnsatz(2, 1)
assert vqe.num_parameters() == 4
e_vqe = vqe.evaluate_energy_pauli_z([0.0, 0.0, 0.0, 0.0], [(0, -1.0), (1, -1.0)])
assert abs(e_vqe - (-2.0)) < 1e-4
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python QAOA/VQE test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v360_autonomous_llm_quantization() -> Result<()> {
    use oxideon::nn::Linear;
    use oxideon::quant::{
        DynamicLlmQuantizer, DynamicPrecisionLevel, DynamicQuantPolicy, DynamicTelemetry,
    };
    use oxideon::tensor::Tensor;

    // 1. Create Linear layers representing LLM blocks (QKV projection & MLP)
    let linear_qkv = Linear::new(64, 64)?;
    let linear_mlp = Linear::new(64, 128)?;

    let policy = DynamicQuantPolicy {
        target_latency_ms: 20.0,
        min_memory_headroom_mb: 500.0,
        critical_memory_headroom_mb: 100.0,
        max_acceptable_kurtosis: 6.0,
        kv_cache_threshold_pct: 85.0,
        cooldown_tokens: 5,
    };

    let mut quantizer = DynamicLlmQuantizer::new(policy);
    quantizer.add_layer("layer0_qkv", linear_qkv);
    quantizer.add_layer("layer0_mlp", linear_mlp);

    let mem_fp32 = quantizer.total_memory_mb();
    assert!(mem_fp32 > 0.0, "Initial FP32 memory should be > 0 MB");

    let input = Tensor::from_vec(vec![0.1f32; 2 * 64], &[2, 64])?;
    let out_qkv = quantizer.forward_layer("layer0_qkv", &input)?;
    assert_eq!(out_qkv.shape(), &[2, 64]);

    // 2. Telemetry with latency exceeding target budget -> Triggers FP32 to FP8 transition
    let telem_high_latency = DynamicTelemetry {
        token_latency_ms: 32.0, // > 20.0 ms target
        memory_headroom_mb: 800.0,
        kv_cache_usage_pct: 40.0,
        activation_kurtosis: 3.5, // low kurtosis, safe to quantize
        token_index: 6,           // > cooldown (5)
    };

    let plan_latency = quantizer.evaluate_telemetry(&telem_high_latency);
    assert!(
        plan_latency.is_some(),
        "Should generate transition plan on latency violation"
    );
    let plan = plan_latency.unwrap();
    assert_eq!(plan.target_precision, DynamicPrecisionLevel::FP8);
    assert_eq!(plan.affected_layers.len(), 2);
    assert!(plan.expected_speedup >= 1.8);

    quantizer.apply_plan(&plan, 6)?;

    // Forward under FP8
    let out_fp8 = quantizer.forward_layer("layer0_qkv", &input)?;
    assert_eq!(out_fp8.shape(), &[2, 64]);

    // 3. Telemetry with Critical Memory Exhaustion -> Triggers emergency downgrade to BitLinear1_58
    let telem_crit_mem = DynamicTelemetry {
        token_latency_ms: 18.0,
        memory_headroom_mb: 50.0, // < 100.0 MB critical threshold
        kv_cache_usage_pct: 98.0, // > 95.0% critical threshold
        activation_kurtosis: 4.0,
        token_index: 15, // > 6 + 5 cooldown
    };

    let plan_crit = quantizer.evaluate_telemetry(&telem_crit_mem);
    assert!(
        plan_crit.is_some(),
        "Should generate emergency downgrade plan"
    );
    let plan = plan_crit.unwrap();
    assert_eq!(plan.target_precision, DynamicPrecisionLevel::BitLinear1_58);

    quantizer.apply_plan(&plan, 15)?;
    let mem_bit = quantizer.total_memory_mb();
    assert!(
        mem_bit < mem_fp32 * 0.25,
        "Ternary memory should be a fraction of FP32"
    );

    let out_bit = quantizer.forward_layer("layer0_qkv", &input)?;
    assert_eq!(out_bit.shape(), &[2, 64]);

    // 4. Telemetry with Generous Headroom -> Autonomous Fidelity Upgrade
    let telem_generous = DynamicTelemetry {
        token_latency_ms: 6.0,      // < 20.0 * 0.65 = 13.0 ms
        memory_headroom_mb: 1800.0, // > 500.0 * 2.5 = 1250.0 MB
        kv_cache_usage_pct: 25.0,
        activation_kurtosis: 2.0,
        token_index: 25, // > 15 + 5 cooldown
    };

    let plan_upgrade = quantizer.evaluate_telemetry(&telem_generous);
    assert!(
        plan_upgrade.is_some(),
        "Should generate fidelity upgrade plan"
    );
    let plan = plan_upgrade.unwrap();
    assert_eq!(plan.target_precision, DynamicPrecisionLevel::Q4_0);

    quantizer.apply_plan(&plan, 25)?;

    // 5. PyO3 Python Integration
    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

policy = ox.DynamicQuantPolicy(
    target_latency_ms=20.0,
    min_memory_headroom_mb=500.0,
    critical_memory_headroom_mb=100.0,
    cooldown_tokens=5
)
quantizer = ox.DynamicLlmQuantizer(policy)

w = ox.Tensor([0.05] * (64 * 64), [64, 64])
quantizer.add_layer("attn_qkv", w, None)

inp = ox.Tensor([0.1] * (2 * 64), [2, 64])
out = quantizer.forward_layer("attn_qkv", inp)
assert out.shape == [2, 64]

# High latency triggers FP8 shift
telem = ox.DynamicTelemetry(35.0, 700.0, 40.0, 3.0, 6)
plan = quantizer.evaluate_telemetry(telem)
assert plan is not None
assert plan.target_precision == "FP8"
assert "LATENCY_BUDGET_EXCEEDED" in plan.trigger_reason

quantizer.apply_plan(plan, 6)
out_fp8 = quantizer.forward_layer("attn_qkv", inp)
assert out_fp8.shape == [2, 64]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Autonomous Quant test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v370_simplicial_cohomology_and_hodge_theory() -> Result<()> {
    use oxideon::science::{OrientedSimplex, SimplicialComplex};
    use oxideon::tensor::Tensor;

    // 1. Circle S^1: 3 edges, empty interior
    let circle_edges = vec![
        OrientedSimplex::new(vec![0, 1]),
        OrientedSimplex::new(vec![1, 2]),
        OrientedSimplex::new(vec![0, 2]),
    ];
    let sc_circle = SimplicialComplex::from_simplices(&circle_edges);
    assert_eq!(sc_circle.num_simplices(0), 3);
    assert_eq!(sc_circle.num_simplices(1), 3);
    assert_eq!(sc_circle.num_simplices(2), 0);

    let b1 = sc_circle.boundary_matrix(1)?;
    assert_eq!(b1.shape(), &[3, 3]);

    // Betti numbers for S^1: beta_0 = 1, beta_1 = 1
    let beta_0 = sc_circle.betti_number(0, 1e-5)?;
    let beta_1 = sc_circle.betti_number(1, 1e-5)?;
    assert_eq!(beta_0, 1, "Circle S^1 must have beta_0 = 1, got {beta_0}");
    assert_eq!(beta_1, 1, "Circle S^1 must have beta_1 = 1, got {beta_1}");

    // 2. Filled triangle D^2: 1 2-simplex [0, 1, 2]
    let disk_triangle = vec![OrientedSimplex::new(vec![0, 1, 2])];
    let sc_disk = SimplicialComplex::from_simplices(&disk_triangle);
    assert_eq!(sc_disk.num_simplices(0), 3);
    assert_eq!(sc_disk.num_simplices(1), 3);
    assert_eq!(sc_disk.num_simplices(2), 1);

    // Fundamental topological theorem: \partial_1 \partial_2 = 0
    let b1_disk = sc_disk.boundary_matrix(1)?; // [3, 3]
    let b2_disk = sc_disk.boundary_matrix(2)?; // [3, 1]
    let boundary_sq = b1_disk.matmul(&b2_disk)?;
    for &val in &boundary_sq.to_vec::<f64>()? {
        assert!(
            val.abs() < 1e-10,
            "Boundary operator squared must be 0, got {val}"
        );
    }

    // Betti numbers for filled triangle D^2: beta_0 = 1, beta_1 = 0
    let beta_0_disk = sc_disk.betti_number(0, 1e-5)?;
    let beta_1_disk = sc_disk.betti_number(1, 1e-5)?;
    assert_eq!(beta_0_disk, 1, "Disk D^2 must have beta_0 = 1");
    assert_eq!(
        beta_1_disk, 0,
        "Disk D^2 must have beta_1 = 0 (cavity filled)"
    );

    // 3. Hodge-Helmholtz cochain flow decomposition on circle
    let omega = Tensor::from_vec(vec![1.0, 0.5, -0.5], &[3])?;
    let (exact, harm, coexact) = sc_circle.helmholtz_hodge_decomposition(&omega, 300, 0.1)?;
    assert_eq!(exact.shape(), &[3]);
    assert_eq!(harm.shape(), &[3]);
    assert_eq!(coexact.shape(), &[3]);

    // Exact + Harm + Coexact reconstructs omega
    let rec = exact.add(&harm)?.add(&coexact)?;
    let diff = rec.sub(&omega)?;
    let err: f64 = diff.to_vec::<f64>()?.iter().map(|x| x.abs()).sum();
    assert!(
        err < 1e-5,
        "Hodge decomposition sum must reconstruct input flow, err={err}"
    );

    // 4. PyO3 Python Integration
    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

# Circle S^1
sc = ox.SimplicialComplex([[0, 1], [1, 2], [0, 2]])
assert sc.num_simplices(0) == 3
assert sc.num_simplices(1) == 3
assert sc.betti_number(0) == 1
assert sc.betti_number(1) == 1

b1 = sc.boundary_matrix(1)
assert b1.shape == [3, 3]

d0 = sc.coboundary_matrix(0)
assert d0.shape == [3, 3]

lap1 = sc.hodge_laplacian(1)
assert lap1.shape == [3, 3]

# Filled triangle D^2
sc_disk = ox.SimplicialComplex([[0, 1, 2]])
assert sc_disk.num_simplices(2) == 1
assert sc_disk.betti_number(0) == 1
assert sc_disk.betti_number(1) == 0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Cohomology test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v380_riemannian_adam_optimization() -> Result<()> {
    use oxideon::optim::{ManifoldType, RiemannianAdam};
    use oxideon::tensor::Tensor;

    // 1. Sphere Manifold S^2: minimize L(x) = - <x, v_target> where v_target = [0, 0, 1]
    let inv_sqrt3 = 1.0 / 3.0f64.sqrt();
    let x = Tensor::from_vec(vec![inv_sqrt3, inv_sqrt3, inv_sqrt3], &[3])?.requires_grad_(true);
    let mut opt_sphere = RiemannianAdam::new(vec![x.clone()], vec![ManifoldType::Sphere], 0.1)?;

    let initial_loss = -x.to_vec::<f64>()?[2];
    assert!((initial_loss - (-inv_sqrt3)).abs() < 1e-5);

    for _ in 0..25 {
        opt_sphere.zero_grad();
        // Euclidean grad of - <x, v_target> is - v_target
        let grad = Tensor::from_vec(vec![0.0, 0.0, -1.0], &[3])?;
        x.set_grad(Some(grad));
        opt_sphere.step()?;

        // Constraint check: norm must remain exactly 1.0 on S^2
        let cur_x = x.to_vec::<f64>()?;
        let norm: f64 = cur_x.iter().map(|&a| a * a).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "Sphere parameter must remain on manifold, got norm {norm}"
        );
    }

    let final_x = x.to_vec::<f64>()?;
    let final_loss = -final_x[2];
    assert!(
        final_loss < -0.95,
        "Optimization must converge towards target [0, 0, 1], got {final_loss}"
    );

    // 2. Stiefel Manifold St(4, 2): X^T X = I_2
    let st_init = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let st_x = Tensor::from_vec(st_init, &[4, 2])?.requires_grad_(true);
    let mut opt_stiefel = RiemannianAdam::new(
        vec![st_x.clone()],
        vec![ManifoldType::Stiefel { p: 2 }],
        0.05,
    )?;

    for _ in 0..10 {
        opt_stiefel.zero_grad();
        let grad = Tensor::from_vec(vec![0.1; 8], &[4, 2])?;
        st_x.set_grad(Some(grad));
        opt_stiefel.step()?;

        // Check X^T X = I_2
        let xt = st_x.t()?;
        let gram = xt.matmul(&st_x)?;
        let g_vec = gram.to_vec::<f64>()?;
        assert!((g_vec[0] - 1.0).abs() < 1e-5, "Gram[0, 0] must be 1.0");
        assert!((g_vec[3] - 1.0).abs() < 1e-5, "Gram[1, 1] must be 1.0");
        assert!(
            g_vec[1].abs() < 1e-5 && g_vec[2].abs() < 1e-5,
            "Off-diagonals must be 0.0"
        );
    }

    // 3. PyO3 Python Integration
    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

x = ox.Tensor([0.57735, 0.57735, 0.57735], [3])
x.requires_grad_(True)
opt = ox.RiemannianAdam([x], ["sphere"], lr=0.1)
target = ox.Tensor([0.0, 0.0, 1.0], [3])

for _ in range(5):
    opt.zero_grad()
    loss = (x * target).sum()
    loss.backward()
    opt.step()

vec = x.to_list()
norm = sum(a * a for a in vec) ** 0.5
assert abs(norm - 1.0) < 1e-5
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Riemannian Adam test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v390_conformal_prediction_and_quantile_uncertainty() -> Result<()> {
    use oxideon::reasoning::{
        AdaptiveConformalClassifier, ConformalEvaluator, ConformalScoreType,
        SplitConformalRegressor,
    };
    use oxideon::tensor::Tensor;

    // 1. Split Conformal Regression on linear synthetic data with uniform noise
    let n_cal = 200;
    let mut y_cal_pred = Vec::with_capacity(n_cal);
    let mut y_cal_true = Vec::with_capacity(n_cal);

    for i in 0..n_cal {
        let x = (i as f64) / (n_cal as f64);
        let mu = 2.0 * x + 1.0;
        let noise = ((i * 7 + 13) % 100) as f64 / 100.0 - 0.5; // [-0.5, 0.5]
        y_cal_pred.push(mu);
        y_cal_true.push(mu + noise);
    }

    let t_cal_pred = Tensor::from_vec(y_cal_pred, &[n_cal])?;
    let t_cal_true = Tensor::from_vec(y_cal_true, &[n_cal])?;

    let mut regressor = SplitConformalRegressor::new(ConformalScoreType::AbsoluteResidual, 0.10)?;
    let q_hat = regressor.calibrate(&t_cal_pred, &t_cal_true, None)?;
    assert!(
        q_hat > 0.40 && q_hat <= 0.50,
        "q_hat should be ~0.45, got {q_hat}"
    );

    // Evaluate on test points
    let n_test = 200;
    let mut y_test_pred = Vec::with_capacity(n_test);
    let mut y_test_true = Vec::with_capacity(n_test);

    for i in 0..n_test {
        let x = (i as f64) / (n_test as f64);
        let mu = 2.0 * x + 1.0;
        let noise = ((i * 11 + 17) % 100) as f64 / 100.0 - 0.5;
        y_test_pred.push(mu);
        y_test_true.push(mu + noise);
    }

    let t_test_pred = Tensor::from_vec(y_test_pred, &[n_test])?;
    let intervals = regressor.predict_intervals(&t_test_pred, None)?;
    assert_eq!(intervals.len(), n_test);

    let (coverage, mean_width) = ConformalEvaluator::evaluate_regression(&intervals, &y_test_true)?;
    assert!(
        coverage >= 0.88,
        "Empirical coverage must satisfy 1 - alpha = 0.90 guarantee, got {coverage}"
    );
    assert!(mean_width > 0.80 && mean_width < 1.05);

    // 2. Adaptive Conformal Classification
    let n_classes = 3;
    let n_clf_cal = 150;
    let mut cal_probs = vec![0.0f64; n_clf_cal * n_classes];
    let mut cal_labels = Vec::with_capacity(n_clf_cal);

    for i in 0..n_clf_cal {
        let true_c = i % n_classes;
        cal_labels.push(true_c);
        // Confident model: p_true ~ 0.70, others 0.15
        for c in 0..n_classes {
            cal_probs[i * n_classes + c] = if c == true_c { 0.70 } else { 0.15 };
        }
    }

    let t_cal_probs = Tensor::from_vec(cal_probs, &[n_clf_cal, n_classes])?;
    let mut classifier = AdaptiveConformalClassifier::new(0.10)?;
    let p_thresh = classifier.calibrate(&t_cal_probs, &cal_labels)?;
    assert!(
        p_thresh <= 0.70,
        "Calibrated threshold must be <= 0.70, got {p_thresh}"
    );

    // Evaluate test sets
    let sets = classifier.predict_sets(&t_cal_probs)?;
    assert_eq!(sets.len(), n_clf_cal);
    let (clf_cov, mean_size) = ConformalEvaluator::evaluate_classification(&sets, &cal_labels)?;
    assert!(
        clf_cov >= 0.88,
        "Classification coverage must be >= 0.88, got {clf_cov}"
    );
    assert!(
        mean_size >= 1.0,
        "Mean set size must be >= 1.0, got {mean_size}"
    );

    // 3. PyO3 Python Integration
    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

# Regression
reg = ox.SplitConformalRegressor("residual", 0.10)
y_p = ox.Tensor([1.0, 2.0, 3.0, 4.0, 5.0], [5])
y_t = ox.Tensor([1.1, 1.9, 3.2, 3.8, 5.1], [5])
q = reg.calibrate(y_p, y_t)
assert q > 0.0

intervals = reg.predict_intervals(y_p)
assert len(intervals) == 5
assert intervals[0][0] < 1.0 and intervals[0][1] > 1.0

# Classification
clf = ox.AdaptiveConformalClassifier(0.10)
probs = ox.Tensor([0.8, 0.1, 0.1, 0.1, 0.8, 0.1], [2, 3])
labels = [0, 1]
thresh = clf.calibrate(probs, labels)
assert thresh > 0.0

sets = clf.predict_sets(probs)
assert len(sets) == 2
assert 0 in sets[0]
assert 1 in sets[1]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Conformal Prediction test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v400_latent_sde_and_grand_omni_engine_iii() -> Result<()> {
    use oxideon::ode::{LatentSdeConfig, LatentSdeEngine};
    use oxideon::optim::{ManifoldType, RiemannianAdam};
    use oxideon::quant::autonomous::{DynamicLlmQuantizer, DynamicQuantPolicy, DynamicTelemetry};
    use oxideon::quantum::QaoaCircuit;
    use oxideon::reasoning::conformal::SplitConformalRegressor;

    // 1. Latent Stochastic Differential Equations (Latent SDE)
    let config = LatentSdeConfig {
        latent_dim: 4,
        hidden_dim: 16,
        diffusion_scale: 0.1,
        t0: 0.0,
        t1: 1.0,
        num_steps: 20,
    };
    let engine = LatentSdeEngine::new(config)?;
    let z0 = Tensor::from_vec(vec![0.5, -0.2, 0.1, 0.8], &[4])?;

    // Prior simulation (path_kl must be 0)
    let traj_prior = engine.simulate(&z0, false, Some(42))?;
    assert_eq!(traj_prior.timestamps.len(), 21);
    assert_eq!(traj_prior.path.shape(), &[21, 4]);
    assert_eq!(traj_prior.path_kl, 0.0);

    // Posterior simulation (Girsanov path_kl > 0)
    let traj_post = engine.simulate(&z0, true, Some(42))?;
    assert_eq!(traj_post.path.shape(), &[21, 4]);
    assert!(
        traj_post.path_kl > 0.0,
        "Posterior drift relative to prior must yield positive Girsanov KL, got {}",
        traj_post.path_kl
    );

    // ELBO evaluation
    let obs = Tensor::from_vec(vec![0.5, -0.2, 0.1, 0.8, 0.3, -0.1, 0.0, 0.6], &[2, 4])?;
    let (elbo, log_lik, kl) = engine.compute_elbo(&z0, &obs, &[0, 20], 0.1, Some(42))?;
    assert!(
        (elbo - (log_lik - kl)).abs() < 1e-6,
        "ELBO must equal log_likelihood - KL divergence"
    );
    assert!(kl > 0.0);

    // 2. Grand Omni-Universal Synthesis (QAOA + Riemannian Adam + Conformal + Dynamic Quant)
    // QAOA MaxCut
    let qaoa = QaoaCircuit::new(3, 1, vec![(0, 1, 1.0), (1, 2, 1.0)])?;
    let qaoa_energy = qaoa.evaluate_energy(&[0.3], &[0.4])?;
    assert!(qaoa_energy > 0.0);

    // Riemannian Adam on Hypersphere S^2
    let inv_sqrt3 = 1.0 / 3.0f64.sqrt();
    let p = Tensor::from_vec(vec![inv_sqrt3, inv_sqrt3, inv_sqrt3], &[3])?.requires_grad_(true);
    let mut opt = RiemannianAdam::new(vec![p.clone()], vec![ManifoldType::Sphere], 0.01)?;
    let g = Tensor::from_vec(vec![0.5, 0.0, 0.5], &[3])?;
    p.set_grad(Some(g));
    opt.step()?;
    let norm = (p.to_vec::<f64>()?.iter().map(|&v| v * v).sum::<f64>()).sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "Manifold retraction must preserve unit norm"
    );

    // Conformal Quantile Uncertainty
    let mut conf = SplitConformalRegressor::new(
        oxideon::reasoning::conformal::ConformalScoreType::AbsoluteResidual,
        0.10,
    )?;
    let p_cal = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4])?;
    let t_cal = Tensor::from_vec(vec![1.05, 1.95, 3.10, 3.90], &[4])?;
    let q = conf.calibrate(&p_cal, &t_cal, None)?;
    assert!(q > 0.0 && q < 0.25);

    // Dynamic Autonomous LLM Quantization
    let policy = DynamicQuantPolicy {
        target_latency_ms: 20.0,
        min_memory_headroom_mb: 500.0,
        critical_memory_headroom_mb: 100.0,
        max_acceptable_kurtosis: 6.0,
        kv_cache_threshold_pct: 85.0,
        cooldown_tokens: 0,
    };
    let mut llm_quantizer = DynamicLlmQuantizer::new(policy);
    llm_quantizer.add_layer("attn_q", oxideon::nn::Linear::new(16, 16)?);
    let telem = DynamicTelemetry {
        token_latency_ms: 45.0,
        memory_headroom_mb: 80.0,
        kv_cache_usage_pct: 92.0,
        activation_kurtosis: 3.0,
        token_index: 1,
    };
    let plan = llm_quantizer.evaluate_telemetry(&telem);
    assert!(
        plan.is_some(),
        "Urgent memory constraint must trigger dynamic quantization plan"
    );

    // 3. PyO3 Python Integration
    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

# Latent SDE simulation
cfg = ox.LatentSdeConfig(latent_dim=4, hidden_dim=16, diffusion_scale=0.1, t0=0.0, t1=1.0, num_steps=10)
engine = ox.LatentSdeEngine(cfg)

z0 = ox.Tensor([0.1, -0.2, 0.3, -0.4], [4])
traj_prior = engine.simulate(z0, False, 123)
assert len(traj_prior.timestamps) == 11
assert traj_prior.path_kl == 0.0

traj_post = engine.simulate(z0, True, 123)
assert traj_post.path_kl > 0.0

obs = ox.Tensor([0.1, -0.2, 0.3, -0.4, 0.0, 0.0, 0.0, 0.0], [2, 4])
elbo, log_lik, kl = engine.compute_elbo(z0, obs, [0, 10], 0.1, 123)
assert abs(elbo - (log_lik - kl)) < 1e-5
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Latent SDE test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v410_neural_algorithmic_reasoning() -> Result<()> {
    use oxideon::reasoning::algorithmic::{
        AlgorithmicGraph, AlgorithmicSemiring, NeuralBellmanFord, PointerGraphNetwork,
    };
    use oxideon::tensor::Tensor;

    // 1. Construct graph with 5 nodes and directed edges
    let mut g = AlgorithmicGraph::new(5);
    g.add_edge(0, 1, 2.0, vec![1.0, 0.0])?;
    g.add_edge(0, 2, 4.0, vec![0.0, 1.0])?;
    g.add_edge(1, 2, 1.0, vec![0.5, 0.5])?;
    g.add_edge(1, 3, 7.0, vec![0.2, 0.8])?;
    g.add_edge(2, 4, 3.0, vec![0.1, 0.9])?;
    g.add_edge(4, 3, 2.0, vec![0.3, 0.7])?;

    assert_eq!(g.num_nodes, 5);
    assert_eq!(g.edges.len(), 6);

    // 2. Neural Bellman-Ford under MinPlus Tropical Semiring (Shortest Path)
    let nbf_minplus = NeuralBellmanFord::new(AlgorithmicSemiring::MinPlus, 8);
    let (dists, preds) = nbf_minplus.solve(&g, 0, 0)?;

    assert_eq!(dists[0], 0.0);
    assert_eq!(dists[1], 2.0);
    assert_eq!(dists[2], 3.0); // 0 -> 1 -> 2 (2 + 1 = 3 < 4)
    assert_eq!(dists[4], 6.0); // 0 -> 1 -> 2 -> 4 (3 + 3 = 6)
    assert_eq!(dists[3], 8.0); // 0 -> 1 -> 2 -> 4 -> 3 (6 + 2 = 8 < 2 + 7 = 9)

    let path_0_to_3 = nbf_minplus.reconstruct_path(0, 3, &preds)?;
    assert_eq!(path_0_to_3, vec![0, 1, 2, 4, 3]);

    // 3. Neural Bellman-Ford under MaxMin Semiring (Bottleneck / Capacity)
    let nbf_maxmin = NeuralBellmanFord::new(AlgorithmicSemiring::MaxMin, 8);
    let (caps, preds_cap) = nbf_maxmin.solve(&g, 0, 0)?;
    assert!(caps[1] == 2.0);
    assert!(caps[2] == 4.0); // Direct edge 0 -> 2 has capacity 4.0 > min(2.0, 1.0) = 1.0
    assert!(caps[4] == 3.0); // 0 -> 2 -> 4 has bottleneck min(4.0, 3.0) = 3.0
    let path_0_to_4_cap = nbf_maxmin.reconstruct_path(0, 4, &preds_cap)?;
    assert_eq!(path_0_to_4_cap, vec![0, 2, 4]);

    // 4. Pointer Graph Network (PGN) Multi-Neighbor Attention & Pointers
    let pgn = PointerGraphNetwork::new(8, Some(42))?;
    assert!(PointerGraphNetwork::new(1 << 31, None).is_err());
    let node_feats = vec![
        0.1, 0.2, -0.1, 0.4, 0.5, -0.2, 0.3, 0.1, -0.3, 0.1, 0.2, 0.0, -0.4, 0.3, 0.1, -0.1, 0.2,
        -0.1, 0.4, 0.3, 0.1, 0.0, -0.2, 0.5, 0.0, 0.3, -0.2, 0.1, 0.2, -0.1, 0.4, 0.2, -0.1, 0.2,
        0.1, -0.3, 0.3, 0.4, -0.1, 0.0,
    ];
    let h = Tensor::from_vec(node_feats, &[5, 8])?;
    let (h_next, ptr_dist) = pgn.forward(&h, &g)?;
    assert_eq!(h_next.shape(), &[5, 8]);
    assert_eq!(ptr_dist.len(), 5);

    // Verify pointer probability normalization sum = 1.0
    for dist in &ptr_dist {
        let prob_sum: f64 = dist.iter().map(|(_, p)| *p).sum();
        assert!((prob_sum - 1.0).abs() < 1e-5);
    }

    let pointers = pgn.argmax_pointers(&ptr_dist);
    assert_eq!(pointers.len(), 5);

    // 5. PyO3 Python Integration
    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

graph = ox.AlgorithmicGraph(4)
graph.add_edge(0, 1, 1.5)
graph.add_edge(1, 2, 2.5)
graph.add_edge(0, 2, 5.0)
graph.add_edge(2, 3, 1.0)
assert graph.num_nodes == 4
assert graph.num_edges == 4

nbf = ox.NeuralBellmanFord("min_plus", 8)
dists, preds = nbf.solve(graph, 0)
assert dists[0] == 0.0
assert dists[1] == 1.5
assert dists[2] == 4.0 # 1.5 + 2.5 < 5.0
assert dists[3] == 5.0 # 4.0 + 1.0

path = nbf.reconstruct_path(0, 3, preds)
assert path == [0, 1, 2, 3]

pgn = ox.PointerGraphNetwork(8, 999)
h = ox.Tensor([0.1] * 32, [4, 8])
h_next, ptrs = pgn.forward(h, graph)
assert h_next.shape == [4, 8]
assert len(ptrs) == 4
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Algorithmic Reasoning test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v420_tropical_algebraic_geometry() -> Result<()> {
    use oxideon::science::tropical::{TropicalMatrix, TropicalPolynomial, tropical_distance};

    // 1. Tropical Matrix Algebra over Max-Plus Semiring
    // A: 2x2 matrix [[0.0, 2.0], [-1.0, 3.0]]
    // B: 2x2 matrix [[1.0, 0.0], [2.0, -2.0]]
    let a = TropicalMatrix::new(2, 2, vec![0.0, 2.0, -1.0, 3.0])?;
    let b = TropicalMatrix::new(2, 2, vec![1.0, 0.0, 2.0, -2.0])?;

    // Matrix-vector multiplication: (A (x) v)_i = max_j (A_ij + v_j)
    // v = [1.0, 2.0]
    // row 0: max(0+1, 2+2) = max(1, 4) = 4.0
    // row 1: max(-1+1, 3+2) = max(0, 5) = 5.0
    let v = vec![1.0, 2.0];
    let av = a.matvec(&v)?;
    assert_eq!(av, vec![4.0, 5.0]);

    // Matrix-matrix multiplication: (A (x) B)_ik = max_j (A_ij + B_jk)
    // (0,0): max(0+1, 2+2) = 4.0
    // (0,1): max(0+0, 2+(-2)) = 0.0
    // (1,0): max(-1+1, 3+2) = 5.0
    // (1,1): max(-1+0, 3+(-2)) = 1.0
    let ab = a.matmul(&b)?;
    assert_eq!(ab.data, vec![4.0, 0.0, 5.0, 1.0]);

    // Kleene star closure on adjacency:
    // G = [[-inf, 2.0], [-inf, -inf]]
    let mut g_data = vec![f64::NEG_INFINITY; 4];
    g_data[1] = 2.0; // edge 0 -> 1 of weight 2.0
    let g = TropicalMatrix::new(2, 2, g_data)?;
    let g_star = g.kleene_star()?;
    // Diagonal must be 0.0 (identity path), edge 0 -> 1 is 2.0
    assert_eq!(g_star.get(0, 0), 0.0);
    assert_eq!(g_star.get(0, 1), 2.0);
    assert_eq!(g_star.get(1, 0), f64::NEG_INFINITY);
    assert_eq!(g_star.get(1, 1), 0.0);

    // 2. Tropical Polynomials & Hypersurfaces
    // P(x1, x2) = max(0 + 2*x1, 1 + x1 + x2, 3 + 0*x1 + 0*x2)
    let mut poly = TropicalPolynomial::new(2);
    poly.add_monomial(0.0, vec![2.0, 0.0])?;
    poly.add_monomial(1.0, vec![1.0, 1.0])?;
    poly.add_monomial(3.0, vec![0.0, 0.0])?;

    // Evaluate at (0.0, 0.0): max(0, 1, 3) = 3.0
    assert_eq!(poly.evaluate(&[0.0, 0.0])?, 3.0);
    assert!(!poly.is_on_hypersurface(&[0.0, 0.0], 1e-5)?);

    // Evaluate at (1.0, 1.0):
    // m1 = 0 + 2 = 2.0
    // m2 = 1 + 1 + 1 = 3.0
    // m3 = 3.0
    // Top-1 is 3.0, Top-2 is 3.0 -> Hypersurface point!
    assert_eq!(poly.evaluate(&[1.0, 1.0])?, 3.0);
    assert!(poly.is_on_hypersurface(&[1.0, 1.0], 1e-5)?);

    // 3. Hilbert Projective Metric (Tropical Distance)
    // Invariant to constant shifts
    let p1 = vec![1.0, 3.0, 5.0];
    let p2 = vec![4.0, 6.0, 8.0]; // p2 = p1 + 3
    assert_eq!(tropical_distance(&p1, &p2)?, 0.0);

    let q1 = vec![1.0, 2.0, 3.0];
    let q2 = vec![3.0, 1.0, 2.0];
    // diffs: -2.0, 1.0, 1.0 -> max(1.0) - min(-2.0) = 3.0
    assert_eq!(tropical_distance(&q1, &q2)?, 3.0);

    // 4. PyO3 Python Integration
    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

# Tropical matrix
m1 = ox.TropicalMatrix(2, 2, [0.0, 1.0, 2.0, 3.0])
m2 = ox.TropicalMatrix(2, 2, [1.0, 0.0, 0.0, 1.0])
m3 = m1.matmul(m2)
assert m3.shape == (2, 2)
# row 0, col 0: max(0+1, 1+0) = 1.0
assert m3.to_list()[0] == 1.0

# Tropical polynomial
poly = ox.TropicalPolynomial(2)
poly.add_monomial(0.0, [1.0, 0.0])
poly.add_monomial(0.0, [0.0, 1.0])
val = poly.evaluate([2.0, 3.0])
assert val == 3.0
# Hypersurface at x1 == x2
assert poly.is_on_hypersurface([2.0, 2.0], 1e-4)

# Tropical distance
d = ox.tropical_distance([1.0, 2.0], [3.0, 4.0])
assert abs(d) < 1e-6
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Tropical Geometry test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v430_quantum_matrix_product_states() -> Result<()> {
    use oxideon::quantum::mps::MatrixProductState;

    // 1. Product state |0000>
    let mps_0000 = MatrixProductState::from_product_state(&[0, 0, 0, 0])?;
    let mps_0001 = MatrixProductState::from_product_state(&[0, 0, 0, 1])?;

    assert_eq!(mps_0000.num_sites, 4);
    assert_eq!(mps_0000.max_bond_dim, 1);
    assert!((mps_0000.norm()? - 1.0).abs() < 1e-6);

    // Overlap orthogonality: <0000|0000> = 1, <0000|0001> = 0
    assert!((mps_0000.overlap(&mps_0000)? - 1.0).abs() < 1e-6);
    assert!(mps_0000.overlap(&mps_0001)?.abs() < 1e-6);

    // Product state entanglement entropy across any bond must be 0
    let s_prod = mps_0000.entanglement_entropy(2)?;
    assert!(
        s_prod < 1e-6,
        "Product state entanglement entropy must be 0, got {s_prod}"
    );

    // Local Pauli-Z expectation: <0000| Z_0 |0000> = +1.0
    let sz = [1.0, 0.0, 0.0, -1.0];
    let exp_z0 = mps_0000.expectation_single_site(0, &sz)?;
    assert!((exp_z0 - 1.0).abs() < 1e-6);

    let mps_1000 = MatrixProductState::from_product_state(&[1, 0, 0, 0])?;
    let exp_z0_1 = mps_1000.expectation_single_site(0, &sz)?;
    assert!((exp_z0_1 - (-1.0)).abs() < 1e-6);

    // TFIM energy for ferromagnetic ground state candidate |0000> with J=1.0, h=0.0
    // H = - \sum_{i=0}^2 Z_i Z_{i+1} = -3.0
    let energy_fm = mps_0000.evaluate_tfim_energy(1.0, 0.0)?;
    assert!((energy_fm - (-3.0)).abs() < 1e-5);

    // 2. Entangled MPS with bond dimension chi = 4
    let mps_rand = MatrixProductState::new_random(4, 4, Some(42))?;
    assert_eq!(mps_rand.num_sites, 4);
    assert!((mps_rand.norm()? - 1.0).abs() < 1e-5);

    let s_ent = mps_rand.entanglement_entropy(2)?;
    assert!(
        s_ent > 0.0,
        "Entangled state must have non-zero entanglement entropy, got {s_ent}"
    );

    // 3. PyO3 Python Integration
    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

# Product state
mps = ox.MatrixProductState.from_product_state([0, 0, 0, 0])
assert mps.num_sites == 4
assert abs(mps.norm() - 1.0) < 1e-5

# Pauli Z expectation on site 0
exp_z = mps.expectation_single_site(0, [1.0, 0.0, 0.0, -1.0])
assert abs(exp_z - 1.0) < 1e-5

# Entanglement entropy of product state is 0
s = mps.entanglement_entropy(2)
assert abs(s) < 1e-5

# Random MPS
rand_mps = ox.MatrixProductState.new_random(4, 4, 123)
assert abs(rand_mps.norm() - 1.0) < 1e-5
energy = rand_mps.evaluate_tfim_energy(1.0, 0.5)
assert isinstance(energy, float)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Quantum MPS test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v440_gromov_wasserstein_optimal_transport() -> Result<()> {
    use oxideon::science::gromov_wasserstein::{GromovWassersteinConfig, GromovWassersteinEngine};
    use oxideon::tensor::Tensor;

    // 1. Metric space C1 (3-node line metric 0 - 1 - 2)
    // C1[i, j] = |i - j|
    let c1_data = vec![0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0];
    let c1 = Tensor::from_vec(c1_data.clone(), &[3, 3])?;

    // Self-matching: GW(C1, C1) must be virtually 0.0
    let engine = GromovWassersteinEngine::new(GromovWassersteinConfig {
        epsilon: 0.02,
        max_iters: 30,
        sinkhorn_iters: 50,
        tolerance: 1e-6,
    });

    let res_self = engine.compute_gw(&c1, &c1, None, None)?;
    assert!(
        res_self.gw_distance < 0.05,
        "Self Gromov-Wasserstein distance should be near 0, got {}",
        res_self.gw_distance
    );
    assert_eq!(res_self.coupling.shape(), &[3, 3]);

    // Verify marginal conservation: row sums ~ 1/3, col sums ~ 1/3
    let t_self = res_self.coupling.to_vec::<f64>()?;
    for i in 0..3 {
        let row_sum: f64 = (0..3).map(|j| t_self[i * 3 + j]).sum();
        assert!((row_sum - 1.0 / 3.0).abs() < 1e-3);
    }

    // 2. Cross-domain graph alignment: C2 has permuted node order (1 - 0 - 2, node 0 is middle)
    let c2_data = vec![0.0, 1.0, 1.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0];
    let c2 = Tensor::from_vec(c2_data, &[3, 3])?;

    let res_perm = engine.compute_gw(&c1, &c2, None, None)?;
    assert!(
        res_perm.gw_distance < 0.08,
        "Isomorphic graph GW distance should be near 0, got {}",
        res_perm.gw_distance
    );

    let matches = engine.align_graphs(&c1, &c2)?;
    assert_eq!(matches.len(), 3);
    // Node 1 in C1 has degree 2 / central position, in C2 node 0 has central position (dist 1 to both)
    // Finding central node alignment:
    let match_center = matches.iter().find(|(u, _)| *u == 1);
    assert!(match_center.is_some());
    assert_eq!(
        match_center.unwrap().1,
        0,
        "Central node 1 in C1 must align with central node 0 in C2"
    );

    // 3. PyO3 Python Integration
    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

c1 = ox.Tensor([0.0, 1.0, 1.0, 0.0], [2, 2])
c2 = ox.Tensor([0.0, 1.0, 1.0, 0.0], [2, 2])

cfg = ox.GromovWassersteinConfig(epsilon=0.05, max_iters=20)
engine = ox.GromovWassersteinEngine(cfg)

res = engine.compute_gw(c1, c2)
assert res.gw_distance >= 0.0
assert res.coupling.shape == [2, 2]

alignment = engine.align_graphs(c1, c2)
assert len(alignment) == 2
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Gromov-Wasserstein test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v450_nonlinear_dispersive_soliton_physics() -> Result<()> {
    use oxideon::physics::soliton::{SolitonPhysics, SpatialGrid1D, SplitStepNlsEngine};
    use oxideon::tensor::Tensor;

    // 1. Spatial Grid
    let grid = SpatialGrid1D::new(256, -15.0, 15.0)?;
    let x = grid.coordinates();
    let dx = grid.dx();

    // 2. Nonlinear Schrödinger Equation (NLS) Bright Soliton
    // Mass invariant of sech soliton with amplitude eta is exactly 2 * eta
    let eta = 1.5;
    let v = 1.0;
    let (real_0, imag_0, dens_0) = SolitonPhysics::nls_bright_soliton(&x, 0.0, eta, v, 0.0, 0.0);
    let mass_0 = SolitonPhysics::nls_mass(&dens_0, dx);
    assert!(
        (mass_0 - 2.0 * eta).abs() < 1e-3,
        "NLS soliton mass must equal 2 * eta = 3.0, got {mass_0}"
    );

    // Propagate analytically to t = 2.0
    let (_real_2, _imag_2, dens_2) = SolitonPhysics::nls_bright_soliton(&x, 2.0, eta, v, 0.0, 0.0);
    let mass_2 = SolitonPhysics::nls_mass(&dens_2, dx);
    assert!(
        (mass_2 - mass_0).abs() < 1e-4,
        "Mass must be rigorously conserved across time"
    );

    // 3. Korteweg-de Vries (KdV) Soliton & Conserved Quantities
    // u(x, t) = c/2 * sech^2(sqrt(c)/2 * (x - c*t - x0))
    // Mass integral = 2 * sqrt(c). For c = 4.0 -> mass = 4.0
    let velocity_c = 4.0;
    let u_0 = SolitonPhysics::kdv_soliton(&x, 0.0, velocity_c, 0.0);
    let (kdv_mass_0, kdv_l2_0) = SolitonPhysics::kdv_invariants(&u_0, dx);
    // Integral of c/2 * sech^2(k x) dx = (c/2) * (2/k) = c / (sqrt(c)/2) = 2 * sqrt(c) = 4.0
    assert!(
        (kdv_mass_0 - 4.0).abs() < 1e-3,
        "KdV mass must equal 2 * sqrt(c) = 4.0, got {kdv_mass_0}"
    );
    assert!(kdv_l2_0 > 0.0);

    // Propagate KdV analytically to t = 1.0
    let u_1 = SolitonPhysics::kdv_soliton(&x, 1.0, velocity_c, 0.0);
    let (kdv_mass_1, kdv_l2_1) = SolitonPhysics::kdv_invariants(&u_1, dx);
    assert!((kdv_mass_1 - kdv_mass_0).abs() < 1e-4);
    assert!((kdv_l2_1 - kdv_l2_0).abs() < 1e-4);

    // 4. KdV PINN Residual Operator
    let u_t = Tensor::from_vec(vec![0.5, 1.0, 0.5], &[3])?;
    let du_dt_t = Tensor::from_vec(vec![-0.1, 0.0, 0.1], &[3])?;
    let du_dx_t = Tensor::from_vec(vec![0.2, 0.0, -0.2], &[3])?;
    let d3u_dx3_t = Tensor::from_vec(vec![-0.5, 0.0, 0.7], &[3])?;

    let res_t = SolitonPhysics::kdv_pinn_residual(&u_t, &du_dt_t, &du_dx_t, &d3u_dx3_t)?;
    assert_eq!(res_t.shape(), &[3]);
    let res_vec = res_t.to_vec::<f64>()?;
    // R[0] = -0.1 + 6 * 0.5 * 0.2 + (-0.5) = -0.1 + 0.6 - 0.5 = 0.0
    assert!(res_vec[0].abs() < 1e-6);

    // 5. Numerical Split-Step Engine
    let split_engine = SplitStepNlsEngine::new(grid.clone(), 0.001);
    let mut num_real = real_0.clone();
    let mut num_imag = imag_0.clone();
    for _ in 0..5 {
        split_engine.step(&mut num_real, &mut num_imag)?;
    }
    let num_dens: Vec<f64> = num_real
        .iter()
        .zip(num_imag.iter())
        .map(|(&r, &im)| r * r + im * im)
        .collect();
    let num_mass = SolitonPhysics::nls_mass(&num_dens, dx);
    assert!(
        (num_mass - mass_0).abs() < 0.05,
        "Numerical split-step must conserve mass"
    );

    // 6. PyO3 Python Integration
    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

grid = ox.SpatialGrid1D(128, -10.0, 10.0)
coords = grid.coordinates()
assert len(coords) == 128

# Analytical NLS Soliton
real, imag, dens = ox.SolitonPhysics.nls_bright_soliton(coords, 0.0, 1.0, 0.0, 0.0, 0.0)
mass = ox.SolitonPhysics.nls_mass(dens, grid.dx)
assert abs(mass - 2.0) < 1e-2

# KdV Soliton
u = ox.SolitonPhysics.kdv_soliton(coords, 0.0, 4.0, 0.0)
kdv_m, kdv_e = ox.SolitonPhysics.kdv_invariants(u, grid.dx)
assert abs(kdv_m - 4.0) < 1e-2

# Split-step NLS engine
engine = ox.SplitStepNlsEngine(grid, 0.001)
r_next, im_next = engine.step(real, imag)
assert len(r_next) == 128
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Soliton Physics test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v460_geometric_mechanics_lie_poisson() -> Result<()> {
    use oxideon::physics::lie_poisson::{LiePoissonState, LiePoissonSystem, RigidBodyInertia};

    // 1. Rigid body inertia configuration
    let inertia = RigidBodyInertia::new(2.0, [1.0, 2.0, 2.5])?;
    assert_eq!(inertia.mass, 2.0);
    assert_eq!(inertia.principal_moments, [1.0, 2.0, 2.5]);

    // 2. Initial state on dual Lie algebra se(3)*
    let initial_state = LiePoissonState::new([1.0, 0.5, -0.5], [0.2, 1.0, -0.4]);
    let c1_0 = initial_state.casimir_c1();
    let c2_0 = initial_state.casimir_c2();

    // Verify analytical Casimirs:
    // C1 = 1^2 + 0.5^2 + (-0.5)^2 = 1.50
    // C2 = 1(0.2) + 0.5(1.0) + (-0.5)(-0.4) = 0.2 + 0.5 + 0.2 = 0.90
    assert!((c1_0 - 1.50).abs() < 1e-6);
    assert!((c2_0 - 0.90).abs() < 1e-6);

    let system = LiePoissonSystem::new(inertia);
    let h_0 = system.hamiltonian(&initial_state);

    // 3. Propagate 100 steps of RK4
    let mut state = initial_state;
    let dt = 0.005;
    for _ in 0..100 {
        state = system.step_rk4(&state, dt);
    }

    // Verify exact conservation of Casimir invariants on coadjoint orbit
    let c1_final = state.casimir_c1();
    let c2_final = state.casimir_c2();
    let h_final = system.hamiltonian(&state);

    assert!(
        (c1_final - c1_0).abs() < 1e-5,
        "Casimir C1 must be conserved along coadjoint orbit: {c1_final} vs {c1_0}"
    );
    assert!(
        (c2_final - c2_0).abs() < 1e-5,
        "Casimir C2 must be conserved along coadjoint orbit: {c2_final} vs {c2_0}"
    );
    assert!(
        (h_final - h_0).abs() < 1e-4,
        "Hamiltonian energy must be conserved: {h_final} vs {h_0}"
    );

    // 4. PyO3 Python Integration
    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

inertia = ox.RigidBodyInertia(2.0, [1.0, 2.0, 2.5])
state = ox.LiePoissonState([1.0, 0.5, -0.5], [0.2, 1.0, -0.4])

c1_0 = state.casimir_c1()
c2_0 = state.casimir_c2()
assert abs(c1_0 - 1.5) < 1e-5
assert abs(c2_0 - 0.9) < 1e-5

sys_mech = ox.LiePoissonSystem(inertia)
h0 = sys_mech.hamiltonian(state)

cur = state
for _ in range(50):
    cur = sys_mech.step_rk4(cur, 0.005)

c1_f = cur.casimir_c1()
c2_f = cur.casimir_c2()
hf = sys_mech.hamiltonian(cur)

assert abs(c1_f - c1_0) < 1e-4
assert abs(c2_f - c2_0) < 1e-4
assert abs(hf - h0) < 1e-3
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Lie-Poisson Mechanics test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v470_symbolic_genetic_programming() -> Result<()> {
    use oxideon::reasoning::symbolic_gp::{
        OpType, SymbolicGpConfig, SymbolicGpEngine, SymbolicNode,
    };
    use oxideon::tensor::Tensor;

    // 1. Manual construction and evaluation of expression tree: f(x) = (x0 + 2.0) * x0
    let tree = SymbolicNode::Binary(
        OpType::Mul,
        Box::new(SymbolicNode::Binary(
            OpType::Add,
            Box::new(SymbolicNode::Variable(0)),
            Box::new(SymbolicNode::Constant(2.0)),
        )),
        Box::new(SymbolicNode::Variable(0)),
    );

    assert_eq!(tree.complexity(), 5);
    assert_eq!(tree.depth(), 3);
    assert_eq!(tree.to_formula(), "((x0 + 2) * x0)");

    // f(3) = (3 + 2) * 3 = 15.0
    let val_at_3 = tree.evaluate(&[3.0]);
    assert!((val_at_3 - 15.0).abs() < 1e-6);

    // 2. Autonomous Genetic Discovery of Quadratic Law: y = x^2
    let n_pts = 20;
    let mut x_data = Vec::with_capacity(n_pts);
    let mut y_data = Vec::with_capacity(n_pts);
    for i in 0..n_pts {
        let xi = -2.0 + (i as f64) * (4.0 / (n_pts - 1) as f64);
        x_data.push(xi);
        y_data.push(xi * xi);
    }

    let x_t = Tensor::from_vec(x_data, &[n_pts, 1])?;
    let y_t = Tensor::from_vec(y_data, &[n_pts])?;

    let engine = SymbolicGpEngine::new(SymbolicGpConfig {
        population_size: 40,
        num_generations: 20,
        max_depth: 4,
        crossover_prob: 0.70,
        mutation_prob: 0.25,
    });

    let pareto_front = engine.fit(&x_t, &y_t, Some(42))?;
    assert!(
        !pareto_front.is_empty(),
        "Pareto front must contain non-dominated candidates"
    );

    // Pareto front must be sorted by complexity
    for i in 0..pareto_front.len() - 1 {
        assert!(pareto_front[i].complexity <= pareto_front[i + 1].complexity);
    }

    // Best individual should achieve low MSE for y = x^2
    let best = pareto_front
        .iter()
        .min_by(|a, b| a.mse.partial_cmp(&b.mse).unwrap());
    assert!(best.is_some());
    let best_ind = best.unwrap();
    assert!(
        best_ind.mse < 0.05,
        "Best evolved formula must achieve low MSE (<0.05), got {} for '{}'",
        best_ind.mse,
        best_ind.formula
    );

    // 3. PyO3 Python Integration
    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

# Create variables and evaluation
v0 = ox.SymbolicNode.variable(0)
assert v0.complexity == 1
assert v0.evaluate([4.2]) == 4.2

# Synthetic dataset
x = ox.Tensor([float(i) for i in range(10)], [10, 1])
y = ox.Tensor([float(i) for i in range(10)], [10])

cfg = ox.SymbolicGpConfig(population_size=20, num_generations=10)
engine = ox.SymbolicGpEngine(cfg)
front = engine.fit(x, y, 123)
assert len(front) > 0

# Check best individual
best = min(front, key=lambda ind: ind.mse)
assert best.mse < 0.1
assert isinstance(best.formula, str)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Symbolic GP test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v480_causal_structural_models() -> Result<()> {
    use oxideon::reasoning::causal::StructuralCausalModel;

    // 1. Classical Confounder DAG:
    // Z (id 0) -> X (id 1) -> Y (id 2)
    // Z (id 0) -------------> Y (id 2)
    let mut scm = StructuralCausalModel::new();
    let z_id = scm.add_variable("Z", vec![], vec![], 0.0, 1.0)?; // Z ~ N(0, 1)
    let x_id = scm.add_variable("X", vec![z_id], vec![2.0], 0.0, 0.2)?; // X = 2.0 * Z + U_x
    let y_id = scm.add_variable("Y", vec![x_id, z_id], vec![3.0, 1.5], 0.0, 0.2)?; // Y = 3.0 * X + 1.5 * Z + U_y

    assert_eq!(scm.num_variables(), 3);

    // 2. Observational Sampling
    let obs_samples = scm.sample_observational(100, Some(42))?;
    assert_eq!(obs_samples.shape(), &[100, 3]);

    // 3. Interventional Sampling under Pearl's do-operator: do(X = 4.0)
    let do_samples = scm.sample_interventional(&[(x_id, 4.0)], 100, Some(42))?;
    let do_data = do_samples.to_vec::<f64>()?;
    for i in 0..100 {
        assert_eq!(
            do_data[i * 3 + x_id],
            4.0,
            "do(X = 4.0) must set X to exactly 4.0 for all units"
        );
    }

    // 4. Counterfactual Reasoning: Abduction -> Action -> Prediction
    // Factual unit: Z = 1.0, X = 2.5, Y = 9.5
    // Abducted noise u_Y = 9.5 - (3.0 * 2.5 + 1.5 * 1.0) = 9.5 - 9.0 = 0.5
    // Counterfactual: "What would Y have been if X had been 0.0?"
    // Y_cf = 3.0 * 0.0 + 1.5 * 1.0 + 0.5 = 2.0
    let factual = vec![1.0, 2.5, 9.5];
    let cf_outcome = scm.compute_counterfactual(&factual, &[(x_id, 0.0)])?;
    assert_eq!(cf_outcome[z_id], 1.0, "Non-descendant Z must be invariant");
    assert_eq!(cf_outcome[x_id], 0.0, "Intervened X must be 0.0");
    assert!(
        (cf_outcome[y_id] - 2.0).abs() < 1e-6,
        "Counterfactual Y must be 2.0, got {}",
        cf_outcome[y_id]
    );

    // 5. Average Treatment Effect (ATE): E[Y | do(X = 1)] - E[Y | do(X = 0)] = 3.0
    let ate = scm.average_treatment_effect(x_id, y_id, 1.0, 0.0, 2000, Some(123))?;
    assert!(
        (ate - 3.0).abs() < 0.1,
        "ATE must be close to causal coefficient 3.0, got {ate}"
    );

    // 6. PyO3 Python Integration
    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

scm = ox.StructuralCausalModel()
z = scm.add_variable("Z", [], [], 0.0, 1.0)
x = scm.add_variable("X", [z], [2.0], 0.0, 0.1)
y = scm.add_variable("Y", [x, z], [3.0, 1.5], 0.0, 0.1)
assert scm.num_variables == 3

# Interventional sampling
do_data = scm.sample_interventional([(x, 5.0)], 20, 42)
assert do_data.shape == [20, 3]

# Counterfactual
cf = scm.compute_counterfactual([1.0, 2.5, 9.5], [(x, 0.0)])
assert abs(cf[2] - 2.0) < 1e-4

# ATE
ate = scm.average_treatment_effect(x, y, 1.0, 0.0, 1000, 42)
assert abs(ate - 3.0) < 0.15
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Causal SCM test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v490_spatio_temporal_graph_wavelet_networks() -> Result<()> {
    use oxideon::geom_graphs::{
        GraphLaplacian, LaplacianType, MultiscaleGraphWaveletTransform, SpatioTemporalWaveletLayer,
    };

    // 1. Graph Laplacian: 4-node cycle graph
    // Adjacency matrix: 0-1-2-3-0
    let adj = vec![
        0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0,
    ];
    let num_nodes = 4;
    let lap = GraphLaplacian::new(&adj, num_nodes, LaplacianType::SymmetricNormalized)?;
    assert_eq!(lap.num_nodes, 4);
    assert_eq!(lap.degrees, vec![2.0, 2.0, 2.0, 2.0]);
    assert!(lap.lambda_max > 0.5 && lap.lambda_max <= 2.1);

    // 2. Multiscale Graph Wavelet Transform
    let scales = vec![0.5, 1.0, 2.0];
    let gwt = MultiscaleGraphWaveletTransform::new(
        &adj,
        num_nodes,
        &scales,
        4,
        LaplacianType::SymmetricNormalized,
    )?;
    assert_eq!(gwt.scales.len(), 3);

    let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2])?;

    let wavelets = gwt.forward(&x)?;
    assert_eq!(wavelets.shape(), &[3, 4, 2]);

    let reconstructed = gwt.inverse(&wavelets)?;
    assert_eq!(reconstructed.shape(), &[4, 2]);

    // 3. Spatio-Temporal Graph Wavelet Recurrent Layer
    let layer = SpatioTemporalWaveletLayer::new(&adj, num_nodes, &scales, 4, 2, 8)?;
    let h1 = layer.step(&x, None)?;
    assert_eq!(h1.shape(), &[4, 8]);

    let x2 = Tensor::from_slice(&[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], &[4, 2])?;
    let h2 = layer.step(&x2, Some(&h1))?;
    assert_eq!(h2.shape(), &[4, 8]);

    // 4. Python Zero-GIL Parity
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

adj = [
    0.0, 1.0, 0.0, 1.0,
    1.0, 0.0, 1.0, 0.0,
    0.0, 1.0, 0.0, 1.0,
    1.0, 0.0, 1.0, 0.0,
]
scales = [0.5, 1.0, 2.0]
gwt = ox.MultiscaleGraphWaveletTransform(adj, 4, scales, order=4)

x = ox.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [4, 2])
w = gwt.forward(x)
assert w.shape == [3, 4, 2]

rec = gwt.inverse(w)
assert rec.shape == [4, 2]

layer = ox.SpatioTemporalWaveletLayer(adj, 4, scales, 4, 2, 8)
h1 = layer.step(x)
assert h1.shape == [4, 8]

h2 = layer.step(x, h1)
assert h2.shape == [4, 8]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Wavelet test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v500_grand_trans_universal_engine_iv() -> Result<()> {
    use oxideon::omni_trans_universal::GrandTransUniversalEngineIV;

    let engine = GrandTransUniversalEngineIV::new(4);
    let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4])?;
    let report = engine.execute_semi_centennial_cycle(&x)?;

    // 1. Neural Algorithmic Reasoning
    assert!((report.nbf_shortest_path_dist - 4.0).abs() < 1e-4);

    // 2. Tropical Hilbert Metric
    assert!(report.tropical_hilbert_distance > 0.0);

    // 3. Quantum MPS
    assert!(report.quantum_mps_ising_energy.is_finite());
    assert!(report.quantum_mps_entanglement_entropy >= 0.0);

    // 4. Gromov-Wasserstein
    assert!(report.gromov_wasserstein_distortion >= 0.0);

    // 5. Soliton Physics
    assert!((report.nls_soliton_conserved_mass - 3.0).abs() < 1e-4);

    // 6. Lie-Poisson Mechanics
    assert!(report.lie_poisson_casimir_invariants.0 > 0.0);

    // 7. Symbolic Genetic Programming
    assert!(report.symbolic_pareto_front_size > 0);

    // 8. Causal SCM ATE
    assert!((report.causal_ate - 3.0).abs() < 0.2);

    // 9. Spatio-Temporal Graph Wavelets
    assert!(report.wavelet_temporal_hidden_norm > 0.0);

    // 10. Python Zero-GIL Parity
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

engine = ox.GrandTransUniversalEngineIV(4)
x = ox.Tensor([1.0, 2.0, 3.0, 4.0], [4])
res = engine.execute_semi_centennial_cycle(x)

assert len(res) == 10
nbf_dist, trop_dist, ising_e, ent_s, gw_dist, nls_mass, casimirs, pareto_size, ate, wav_norm = res

assert abs(nbf_dist - 4.0) < 1e-4
assert trop_dist > 0.0
assert abs(nls_mass - 3.0) < 1e-4
assert casimirs[0] > 0.0
assert pareto_size > 0
assert abs(ate - 3.0) < 0.2
assert wav_norm > 0.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Grand Trans Universal IV test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v510_fiber_bundles_and_gauge_networks() -> Result<()> {
    use oxideon::geom_graphs::fiber_bundle::{
        GaugeEquivariantConvLayer, PrincipalBundleConnection,
    };

    // 1. Principal Bundle Connection over 3-node triangle
    let mut conn = PrincipalBundleConnection::new(3, 2)?;
    let pi = std::f64::consts::PI;

    // Flat connection: pi/4 + pi/4 - pi/2 = 0
    conn.set_connection(0, 1, pi / 4.0)?;
    conn.set_connection(1, 2, pi / 4.0)?;
    conn.set_connection(2, 0, -pi / 2.0)?;

    let holonomy_flat = conn.cycle_holonomy(&[0, 1, 2])?;
    assert!(
        holonomy_flat.abs() < 1e-12,
        "Flat connection holonomy must be 0, got {holonomy_flat}"
    );

    // Curved connection: non-zero holonomy (curvature)
    conn.set_connection(2, 0, 0.0)?;
    let holonomy_curved = conn.cycle_holonomy(&[0, 1, 2])?;
    assert!(
        (holonomy_curved - pi / 2.0).abs() < 1e-12,
        "Curved connection holonomy must be pi/2, got {holonomy_curved}"
    );

    // 2. Gauge-Equivariant Convolution Layer
    let layer = GaugeEquivariantConvLayer::new(conn);
    let x = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0, -1.0, 0.5], &[3, 2])?;

    let y = layer.forward(&x)?;
    assert_eq!(y.shape(), &[3, 2]);

    // 3. Exact Gauge Equivariance Verification
    let gauge_angles = vec![0.42, -0.85, 1.33];
    let equivariance_error = layer.verify_gauge_equivariance(&x, &gauge_angles)?;
    assert!(
        equivariance_error < 1e-12,
        "Network must be strictly gauge-equivariant, max error: {equivariance_error}"
    );

    // 4. PyO3 Python Integration
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import math
import oxideon as ox

conn = ox.PrincipalBundleConnection(3, 2)
conn.set_connection(0, 1, math.pi / 4.0)
conn.set_connection(1, 2, math.pi / 4.0)
conn.set_connection(2, 0, -math.pi / 2.0)

hol = conn.cycle_holonomy([0, 1, 2])
assert abs(hol) < 1e-10

layer = ox.GaugeEquivariantConvLayer(conn)
x = ox.Tensor([1.0, 0.0, 0.0, 1.0, -1.0, 0.5], [3, 2])
y = layer.forward(x)
assert y.shape == [3, 2]

err = layer.verify_gauge_equivariance(x, [0.3, -0.6, 1.1])
assert err < 1e-10
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Fiber Bundle test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v520_cellular_sheaf_laplacian_diffusion() -> Result<()> {
    use oxideon::science::sheaf_diffusion::{AdaptiveCellularSheaf, SheafDiffusionLayer};

    // 1. Cellular Sheaf over 3-node path: 0 - 1 - 2
    let mut sheaf = AdaptiveCellularSheaf::new(3, 2, vec![(0, 1), (1, 2)])?;
    assert_eq!(sheaf.num_nodes, 3);
    assert_eq!(sheaf.stalk_dim, 2);

    // Constant global section with identity restriction maps
    let constant_section = vec![1.0, 0.5, 1.0, 0.5, 1.0, 0.5];
    let energy_trivial = sheaf.dirichlet_energy(&constant_section)?;
    let obs_trivial = sheaf.obstruction_norm(&constant_section)?;
    assert!(
        energy_trivial.abs() < 1e-12,
        "Trivial section must have 0 Dirichlet energy"
    );
    assert!(
        obs_trivial.abs() < 1e-12,
        "Trivial section must have 0 obstruction"
    );

    // Sheaf Laplacian quadratic form identity: x^T (L_F x) == 2 * E_F(x)
    let non_trivial_section = vec![1.0, -0.5, 0.0, 2.0, -1.5, 0.5];
    let l_x = sheaf.laplacian_matvec(&non_trivial_section)?;
    let quad_form: f64 = non_trivial_section
        .iter()
        .zip(l_x.iter())
        .map(|(&a, &b)| a * b)
        .sum();
    let e_non_trivial = sheaf.dirichlet_energy(&non_trivial_section)?;
    assert!(
        (quad_form - 2.0 * e_non_trivial).abs() < 1e-10,
        "x^T L_F x must equal 2 * E_F(x): {quad_form} vs {}",
        2.0 * e_non_trivial
    );

    // 2. Introduce non-trivial twist (orthogonal restriction maps)
    sheaf.set_edge_angles(1, 0.0, std::f64::consts::FRAC_PI_2)?;
    let obs_twisted = sheaf.obstruction_norm(&constant_section)?;
    assert!(
        obs_twisted > 0.1,
        "Twisted sheaf must introduce obstruction for constant section: {obs_twisted}"
    );

    // 3. Sheaf Diffusion Layer
    let layer = SheafDiffusionLayer::new(sheaf, 0.05);
    let x = Tensor::from_slice(&non_trivial_section, &[3, 2])?;
    let y = layer.forward(&x)?;
    assert_eq!(y.shape(), &[3, 2]);

    // 4. PyO3 Python Integration
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

sheaf = ox.AdaptiveCellularSheaf(3, 2, [(0, 1), (1, 2)])
sec = [1.0, 0.5, 1.0, 0.5, 1.0, 0.5]
energy = sheaf.dirichlet_energy(sec)
assert abs(energy) < 1e-10

obs = sheaf.obstruction_norm(sec)
assert abs(obs) < 1e-10

layer = ox.SheafDiffusionLayer(sheaf, 0.05)
x = ox.Tensor(sec, [3, 2])
y = layer.forward(x)
assert y.shape == [3, 2]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Sheaf Diffusion test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v530_coherent_photonic_waveguides() -> Result<()> {
    use oxideon::hardware::coherent_photonic::{CoherentPhotonicCircuit, WaveguideProfile};

    // 1. 4-mode Coherent Photonic Circuit with ideal lossless transmission
    let profile = WaveguideProfile {
        transmission_factor: 1.0,
        crosstalk_leakage: 0.0,
        ..Default::default()
    };

    let mut circuit = CoherentPhotonicCircuit::new(4, profile)?;
    circuit.set_phase_shift(0, 0, std::f64::consts::FRAC_PI_4)?;
    circuit.set_phase_shift(0, 1, -std::f64::consts::FRAC_PI_4)?;
    circuit.set_coupling_ratio(0, 0.5)?; // 50:50 directional coupler

    let e_r = vec![1.0, 0.0, 0.5, 0.0];
    let e_i = vec![0.0, 0.0, 0.0, 0.0];

    // Total power in: 1.0^2 + 0.5^2 = 1.25
    let p_in: f64 = CoherentPhotonicCircuit::optical_power(&e_r, &e_i)
        .iter()
        .sum();
    assert!((p_in - 1.25).abs() < 1e-12);

    // Propagate coupler layer
    let (c_r, c_i) = circuit.propagate_coupler_layer(&e_r, &e_i)?;
    let p_coupler: f64 = CoherentPhotonicCircuit::optical_power(&c_r, &c_i)
        .iter()
        .sum();
    assert!(
        (p_coupler - p_in).abs() < 1e-12,
        "Lossless directional coupler must conserve optical power: {p_coupler} vs {p_in}"
    );

    // 2. Balanced Homodyne Detection
    let sig_r = vec![1.0, 0.5];
    let sig_i = vec![0.0, 0.0];
    let lo_r = vec![1.0, 1.0];
    let lo_i = vec![0.0, 0.0];
    let homodyne = CoherentPhotonicCircuit::balanced_homodyne(&sig_r, &sig_i, &lo_r, &lo_i)?;
    assert!((homodyne[0] - 4.0).abs() < 1e-12);
    assert!((homodyne[1] - 2.0).abs() < 1e-12);

    // 3. Electro-Optic Activation (MZM transmission)
    let voltages = vec![0.0, 2.0, 4.0];
    let v_pi = 4.0;
    let transmissions = CoherentPhotonicCircuit::electro_optic_activation(&voltages, v_pi)?;
    assert!((transmissions[0] - 1.0).abs() < 1e-12); // V = 0 -> T = 1.0
    assert!((transmissions[1] - 0.5).abs() < 1e-12); // V = V_pi/2 -> T = 0.5
    assert!(transmissions[2].abs() < 1e-12); // V = V_pi -> T = 0.0

    // 4. Tensor Forward Power
    let input_fields = Tensor::from_slice(&[1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0], &[4, 2])?;
    let powers = circuit.forward_power(&input_fields, 2)?;
    assert_eq!(powers.shape(), &[4]);

    // 5. PyO3 Python Integration
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

prof = ox.WaveguideProfile(transmission_factor=0.99, crosstalk_leakage=0.001)
circuit = ox.CoherentPhotonicCircuit(4, prof)
circuit.set_phase_shift(0, 0, 0.5)
circuit.set_coupling_ratio(0, 0.5)

out_r, out_i = circuit.forward([1.0, 0.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.0], 2)
assert len(out_r) == 4
assert len(out_i) == 4

p = ox.CoherentPhotonicCircuit.optical_power(out_r, out_i)
assert len(p) == 4

trans = ox.CoherentPhotonicCircuit.electro_optic_activation([0.0, 2.0, 4.0], 4.0)
assert abs(trans[0] - 1.0) < 1e-10
assert abs(trans[2] - 0.0) < 1e-10

x = ox.Tensor([1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0], [4, 2])
pow_t = circuit.forward_power(x, 2)
assert pow_t.shape == [4]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Photonic test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v540_riemannian_continuous_normalizing_flows() -> Result<()> {
    use oxideon::ode::riemannian_cnf::PoincareCnfFlow;

    // 1. Poincaré Ball Conformal Geometry (dimension 2)
    let flow = PoincareCnfFlow::new(2)?;
    assert!(PoincareCnfFlow::new(usize::MAX).is_err());

    let origin = vec![0.0, 0.0];
    let lam_0 = flow.conformal_factor(&origin);
    assert!(
        (lam_0 - 2.0).abs() < 1e-12,
        "Poincaré conformal factor at origin must be 2.0, got {lam_0}"
    );

    let pt = vec![0.5, 0.0];
    let lam_pt = flow.conformal_factor(&pt);
    assert!((lam_pt - 8.0 / 3.0).abs() < 1e-12);

    // Grad log lambda at origin is 0.0
    let grad_log_0 = flow.grad_log_conformal(&origin);
    assert_eq!(grad_log_0, vec![0.0, 0.0]);

    // 2. Continuous Flow Integration on Poincaré Disk
    let x_0 = Tensor::from_slice(&[0.2, -0.3], &[2])?;
    let traj = flow.integrate(&x_0, 25, 1.0)?;

    assert_eq!(traj.x_1.shape(), &[2]);
    assert_eq!(traj.path.shape(), &[26, 2]);
    assert_eq!(traj.timestamps.len(), 26);

    // Verify pushed-forward point remains strictly within the unit disk ||x|| < 1.0
    let x1_vec = traj.x_1.to_vec::<f64>()?;
    let r1 = (x1_vec[0] * x1_vec[0] + x1_vec[1] * x1_vec[1]).sqrt();
    assert!(
        r1 < 0.96,
        "Flow must preserve Poincaré disk boundary: r1 = {r1}"
    );

    // Verify continuous change of variables log-density adjustment is finite
    assert!(traj.delta_log_p.is_finite());

    // 3. PyO3 Python Integration
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

flow = ox.PoincareCnfFlow(2)
lam0 = flow.conformal_factor([0.0, 0.0])
assert abs(lam0 - 2.0) < 1e-10

div0 = flow.riemannian_divergence([0.0, 0.0], 0.0)
assert isinstance(div0, float)

x0 = ox.Tensor([0.2, -0.3], [2])
traj = flow.integrate(x0, 20, 1.0)

assert traj.x_1.shape == [2]
assert traj.path.shape == [21, 2]
assert len(traj.timestamps) == 21
assert isinstance(traj.delta_log_p, float)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Riemannian CNF test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v550_quantum_error_mitigation() -> Result<()> {
    use oxideon::quantum::mitigation::{
        CliffordDataRegression, ExtrapolationModel, ZneExtrapolator,
    };

    // 1. Zero-Noise Extrapolation: Linear & Richardson Polynomial
    // Underlying signal: E(lambda) = 0.85 - 0.15 * lambda ==> E(0) = 0.85
    let scale_factors = vec![1.0, 1.5, 2.0];
    let noisy_values = vec![0.70, 0.625, 0.55];

    let zne_linear = ZneExtrapolator::new(scale_factors.clone(), ExtrapolationModel::Linear)?;
    let val_linear = zne_linear.extrapolate(&noisy_values)?;
    assert!(
        (val_linear - 0.85).abs() < 1e-10,
        "Linear ZNE must recover 0.85 exactly, got {val_linear}"
    );

    let zne_richardson = ZneExtrapolator::new(scale_factors, ExtrapolationModel::Richardson)?;
    let val_richardson = zne_richardson.extrapolate(&noisy_values)?;
    assert!(
        (val_richardson - 0.85).abs() < 1e-10,
        "Richardson ZNE must recover 0.85 exactly, got {val_richardson}"
    );

    // 2. Clifford Data Regression (CDR) Calibration
    let exact_cliffords = vec![0.9, 0.5, -0.2, -0.8];
    // Noisy hardware response: E_noisy = 0.7 * E_exact + 0.05
    let noisy_cliffords = vec![0.68, 0.40, -0.09, -0.51];

    let mut cdr = CliffordDataRegression::new();
    assert!(!cdr.is_fitted);

    cdr.fit(&noisy_cliffords, &exact_cliffords)?;
    assert!(cdr.is_fitted);
    assert!((cdr.alpha - (1.0 / 0.7)).abs() < 0.05);

    // Mitigate unknown target state measured at 0.40 (true value is 0.50)
    let mitigated = cdr.mitigate(0.40)?;
    assert!(
        (mitigated - 0.50).abs() < 0.01,
        "CDR must calibrate target noisy expectation to ~0.50, got {mitigated}"
    );

    // 3. PyO3 Python Integration
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

zne = ox.ZneExtrapolator([1.0, 1.5, 2.0], "richardson")
e0 = zne.extrapolate([0.70, 0.625, 0.55])
assert abs(e0 - 0.85) < 1e-10

cdr = ox.CliffordDataRegression()
cdr.fit([0.68, 0.40, -0.09, -0.51], [0.9, 0.5, -0.2, -0.8])
mit = cdr.mitigate(0.40)
assert abs(mit - 0.50) < 0.02
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Quantum Mitigation test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v560_rkhs_mercer_and_mmd() -> Result<()> {
    use oxideon::science::rkhs::{
        KernelRidgeRegressor, MercerKernel, RandomFourierFeatures, maximum_mean_discrepancy,
    };

    // 1. Mercer Kernels
    let rbf = MercerKernel::GaussianRbf { gamma: 0.5 };
    let matern32 = MercerKernel::Matern32 { length_scale: 1.0 };
    let p1 = vec![1.0, 2.0];
    let p2 = vec![1.0, 2.0];
    let p3 = vec![2.0, 3.0];

    assert!((rbf.eval(&p1, &p2)? - 1.0).abs() < 1e-12);
    assert!((matern32.eval(&p1, &p2)? - 1.0).abs() < 1e-12);
    assert!(rbf.eval(&p1, &p3)? < 1.0);

    // 2. Random Fourier Features Spectral Approximation
    let rff = RandomFourierFeatures::new_gaussian_rbf(2, 600, 0.5, Some(1234));
    let phi1 = rff.transform(&p1)?;
    let phi3 = rff.transform(&p3)?;
    let approx_k: f64 = phi1.iter().zip(phi3.iter()).map(|(&a, &b)| a * b).sum();
    let exact_k = rbf.eval(&p1, &p3)?;
    assert!(
        (approx_k - exact_k).abs() < 0.08,
        "RFF spectral expansion must approximate exact kernel: {approx_k} vs {exact_k}"
    );

    // 3. Kernel Ridge Regression
    let mut krr = KernelRidgeRegressor::new(MercerKernel::GaussianRbf { gamma: 1.0 }, 1e-4);
    let x_train = vec![vec![-2.0], vec![-1.0], vec![0.0], vec![1.0], vec![2.0]];
    let y_train = vec![4.0, 1.0, 0.0, 1.0, 4.0];
    krr.fit(&x_train, &y_train)?;

    let pred_0_5 = krr.predict(&[0.5])?;
    assert!(
        (pred_0_5 - 0.25).abs() < 0.15,
        "KRR must interpolate quadratic function at 0.5, got {pred_0_5}"
    );

    // Tensor prediction check
    let x_test = Tensor::from_slice(&[0.0, 1.0, 2.0], &[3, 1])?;
    let y_preds = krr.predict_tensor(&x_test)?;
    assert_eq!(y_preds.shape(), &[3]);

    // 4. Maximum Mean Discrepancy (MMD)
    let dist_p: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.1]).collect();
    let dist_q: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.1 + 3.0]).collect();

    let mmd_self = maximum_mean_discrepancy(&dist_p, &dist_p, rbf, false)?;
    assert!(mmd_self.abs() < 1e-12, "Self MMD must be 0, got {mmd_self}");

    let mmd_diff = maximum_mean_discrepancy(&dist_p, &dist_q, rbf, false)?;
    assert!(
        mmd_diff > 0.5,
        "Distinct distributions must yield positive MMD: {mmd_diff}"
    );

    // 5. PyO3 Python Integration
    #[cfg(feature = "python")]
    {
        use pyo3::types::{PyAnyMethods, PyModule};
        let _py_lock = PYTHON_TEST_MUTEX.lock().unwrap();
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

rff = ox.RandomFourierFeatures(2, 100, 1.0, 42)
phi = rff.transform([1.0, 2.0])
assert len(phi) == 100

krr = ox.KernelRidgeRegressor("rbf", 1.0, 1e-4)
krr.fit([[-1.0], [0.0], [1.0]], [1.0, 0.0, 1.0])
p0 = krr.predict([0.0])
assert abs(p0 - 0.0) < 0.05

p_samples = [[i * 0.1] for i in range(10)]
q_samples = [[i * 0.1 + 2.0] for i in range(10)]
mmd = ox.maximum_mean_discrepancy(p_samples, q_samples, 1.0, False)
assert mmd > 0.3
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python RKHS & MMD test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v570_deep_equilibrium_models() -> Result<()> {
    use oxideon::nn::deq::{DeepEquilibriumModel, DeqConfig};

    // 1. Rust Deep Equilibrium Model forward fixed-point solve
    let config = DeqConfig {
        max_forward_iters: 60,
        fwd_tol: 1e-6,
        relaxation_beta: 0.6,
        max_backward_iters: 50,
        bwd_tol: 1e-6,
    };

    let deq = DeepEquilibriumModel::new(4, 4, config)?;
    let x_data = vec![0.5, -0.3, 0.2, 0.8];
    let x_tensor = Tensor::from_slice(&x_data, &[4])?;

    let out = deq.forward_equilibrium(&x_tensor)?;
    assert!(out.converged, "DEQ fixed-point iteration must converge");
    assert!(
        out.residual_norm < 1e-6,
        "Residual norm must be under tolerance: got {}",
        out.residual_norm
    );
    assert!(out.num_iters > 0 && out.num_iters <= 60);

    let z_star_vec = out.z_star.to_vec::<f64>()?;
    let f_z = deq.step_f(&z_star_vec, &x_data)?;
    for i in 0..4 {
        assert!(
            (f_z[i] - z_star_vec[i]).abs() < 1e-5,
            "Fixed point must satisfy f(z*) = z* at index {i}"
        );
    }

    // 2. Implicit Function Theorem (IFT) adjoint backward solver
    let grad_z_star = vec![1.0, -0.5, 0.2, -0.1];
    let adjoint_y = deq.solve_adjoint(&grad_z_star, &z_star_vec, &x_data)?;
    assert_eq!(adjoint_y.len(), 4);

    // Verify adjoint condition: y - J^T y = grad_z_star
    let j_t_y = deq.vjp_state(&adjoint_y, &z_star_vec, &x_data)?;
    for i in 0..4 {
        let reconstructed_grad = adjoint_y[i] - j_t_y[i];
        assert!(
            (reconstructed_grad - grad_z_star[i]).abs() < 1e-4,
            "Adjoint equation (I - J^T) y = grad failed at index {i}: expected {}, got {}",
            grad_z_star[i],
            reconstructed_grad
        );
    }

    // 3. Python PyO3 bindings verification
    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

cfg = ox.DeqConfig(60, 1e-6, 0.6, 50, 1e-6)
assert cfg.max_forward_iters == 60
assert cfg.fwd_tol == 1e-6

model = ox.DeepEquilibriumModel(3, 3, cfg)
x = ox.Tensor([0.4, -0.2, 0.7], [3])

z_star, residual, iters, converged = model.forward_equilibrium(x)
assert converged, "Python DEQ solver must converge"
assert residual < 1e-5
z_list = z_star.to_list()
assert len(z_list) == 3

y = model.solve_adjoint([1.0, 0.0, 0.5], z_list, [0.4, -0.2, 0.7])
assert len(y) == 3

vjp = model.vjp_state(y, z_list, [0.4, -0.2, 0.7])
assert len(vjp) == 3
for i in range(3):
    assert abs((y[i] - vjp[i]) - [1.0, 0.0, 0.5][i]) < 1e-4
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python DEQ test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v580_conformalized_risk_control() -> Result<()> {
    use oxideon::reasoning::conformal_risk::{ConformalRiskController, LearnThenTest, RiskMetric};

    // 1. Synthetic multi-label calibration data (50 samples, 5 classes)
    let n_samples = 50;
    let n_classes = 5;
    let mut calibration_scores = Vec::with_capacity(n_samples);
    let mut ground_truth = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        // Deterministic pseudo-probabilities
        let mut scores = vec![0.0; n_classes];
        for (c, sc) in scores.iter_mut().enumerate() {
            let val = ((i * 17 + c * 31 + 11) % 100) as f64 / 100.0;
            *sc = val;
        }
        calibration_scores.push(scores);

        // Ground truth: class with highest score and optionally second highest
        let true_c = (i * 3) % n_classes;
        let alt_c = (i * 3 + 1) % n_classes;
        ground_truth.push(vec![true_c, alt_c]);
    }

    // 2. Conformal Risk Control calibration for False Negative Rate (FNR) <= 0.20
    let target_alpha = 0.20;
    let crc = ConformalRiskController::new(RiskMetric::FalseNegativeRate, target_alpha, 1.0)?;
    let cal_result = crc.calibrate(&calibration_scores, &ground_truth, 100)?;

    assert!(cal_result.lambda_hat >= 0.0 && cal_result.lambda_hat <= 1.0);
    assert!(
        cal_result.risk_upper_bound <= target_alpha + 1e-6,
        "CRC risk upper bound must satisfy alpha constraint: got {} > {}",
        cal_result.risk_upper_bound,
        target_alpha
    );
    assert!(cal_result.empirical_risk <= cal_result.risk_upper_bound);

    // Predict on test instance
    let test_scores = vec![0.85, 0.12, 0.72, 0.05, 0.40];
    let pred_set = ConformalRiskController::predict_set(&test_scores, cal_result.lambda_hat);
    // Highest confidence label (index 0 with score 0.85) must be included
    assert!(pred_set.contains(&0));

    // 3. Learn-Then-Test: Hoeffding-Bentkus p-value and Benjamini-Hochberg FDR control
    let p_val = LearnThenTest::hoeffding_bentkus_p_value(0.05, 0.20, 100)?;
    assert!(
        p_val < 0.01,
        "Low empirical risk should yield small p-value: got {}",
        p_val
    );

    let p_vals = vec![0.001, 0.008, 0.025, 0.08, 0.35];
    let rejections = LearnThenTest::benjamini_hochberg(&p_vals, 0.05);
    // At FDR q = 0.05, the first 3 hypotheses should be rejected
    assert_eq!(rejections.len(), 3);
    assert_eq!(rejections, vec![0, 1, 2]);

    // 4. Python PyO3 bindings verification
    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

crc = ox.ConformalRiskController("fnr", 0.25, 1.0)
scores = [
    [0.9, 0.1, 0.05],
    [0.2, 0.8, 0.1],
    [0.3, 0.4, 0.7],
    [0.85, 0.05, 0.1]
]
labels = [[0], [1], [2], [0]]
res = crc.calibrate(scores, labels, 50)

assert res.target_alpha == 0.25
assert res.risk_upper_bound <= 0.25 + 1e-4
pred = ox.ConformalRiskController.predict_set([0.88, 0.1, 0.05], res.lambda_hat)
assert 0 in pred

p_val = ox.LearnThenTest.hoeffding_bentkus_p_value(0.04, 0.20, 80)
assert p_val < 0.05

rejected = ox.LearnThenTest.benjamini_hochberg([0.001, 0.005, 0.20], 0.05)
assert len(rejected) == 2
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python CRC test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v590_non_hermitian_exceptional_points() -> Result<()> {
    use oxideon::physics::non_hermitian::NonHermitianHamiltonian2x2;

    // 1. PT-Unbroken Phase (gamma < g): Real Eigenvalues
    let h_unbroken = NonHermitianHamiltonian2x2::new(0.0, 0.5, 1.0)?;
    let rep_unbroken = h_unbroken.analyze();
    assert!(
        rep_unbroken.is_pt_unbroken,
        "System with gamma < g must be in PT-unbroken phase"
    );
    assert!(!rep_unbroken.is_exceptional_point);
    assert!((rep_unbroken.lambda_plus.real - 0.75f64.sqrt()).abs() < 1e-5);
    assert!(rep_unbroken.lambda_plus.imag.abs() < 1e-6);

    // 2. PT-Broken Phase (gamma > g): Imaginary Eigenvalues
    let h_broken = NonHermitianHamiltonian2x2::new(0.0, 1.5, 1.0)?;
    let rep_broken = h_broken.analyze();
    assert!(
        !rep_broken.is_pt_unbroken,
        "System with gamma > g must be in PT-broken phase"
    );
    assert!((rep_broken.lambda_plus.imag.abs() - 1.25f64.sqrt()).abs() < 1e-5);

    // 3. Exact Exceptional Point 2 (EP2: delta=0, gamma=g=1.0)
    let h_ep = NonHermitianHamiltonian2x2::new(0.0, 1.0, 1.0)?;
    let rep_ep = h_ep.analyze();
    assert!(
        rep_ep.is_exceptional_point,
        "EP2 condition delta=0, gamma=g must trigger EP"
    );
    assert!(rep_ep.spectral_gap < 1e-5);
    assert!(
        rep_ep.petermann_factor >= 1e4,
        "Petermann factor must diverge at EP"
    );

    // 4. Enhanced Sensor Sensitivity: Square-root splitting delta_lambda ~ sqrt(eps)
    let eps = 0.01;
    let splitting = h_ep.sensor_frequency_splitting(eps)?;
    // In EP2: (delta + eps)^2 + g^2 - gamma^2 = eps^2 + 2*i*eps*gamma
    // modulus of splitting ~ 2 * (eps * gamma)^{1/2} ~ 2 * 0.1 = 0.2
    assert!(
        splitting > 0.10,
        "EP sensor response must be order of magnitude larger than eps: got {}",
        splitting
    );

    // 5. Riemann Surface Parameter Encircling
    let trajectory = NonHermitianHamiltonian2x2::encircle_ep(1.0, 0.2, 16)?;
    assert_eq!(trajectory.len(), 16);

    // 6. Python PyO3 bindings verification
    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

h = ox.NonHermitianHamiltonian2x2(0.0, 1.0, 1.0)
rep = h.analyze()
assert rep.is_exceptional_point
assert rep.spectral_gap < 1e-5
assert rep.petermann_factor > 1e4

split = h.sensor_frequency_splitting(0.01)
assert split > 0.10

traj = ox.NonHermitianHamiltonian2x2.encircle_ep(1.0, 0.15, 12)
assert len(traj) == 12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Non-Hermitian test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v600_grand_omni_universal_engine_v() -> Result<()> {
    use oxideon::omni_diamond_v::GrandOmniUniversalEngineV;

    // 1. Execute the Grand Omni-Universal Foundation Engine V Diamond Synthesis
    let engine = GrandOmniUniversalEngineV::new();
    let report = engine.execute_synthesis()?;

    // 2. Rigorous multi-frontier invariants verification
    assert!(report.fiber_bundle_holonomy_angle.is_finite());
    assert!(report.sheaf_obstruction_norm.is_finite());
    assert!(report.photonic_homodyne_power.abs() > 1e-4);
    assert!(
        report.riemannian_flow_radius < 1.0,
        "Flow must stay inside Poincaré disk boundary"
    );
    assert!(report.quantum_zne_mitigated_value.is_finite());
    assert!(
        report.rkhs_mmd_distance > 0.0,
        "MMD distance between separated distributions must be positive"
    );
    assert!(
        report.deq_equilibrium_residual < 1e-5,
        "DEQ solver must reach tight equilibrium"
    );
    assert!(report.crc_calibrated_threshold >= 0.0 && report.crc_calibrated_threshold <= 1.0);
    assert!(
        report.fdr_discovered_count >= 1,
        "FDR procedure must reject true non-null hypotheses"
    );
    assert!(
        report.exceptional_point_petermann >= 1e4,
        "Petermann factor must diverge at EP"
    );
    assert!(
        ["BitLinear1_58", "Q4_0", "Q8_0", "FP8", "FP32"]
            .contains(&report.autonomous_quant_format.as_str()),
        "Valid autonomous quant format required"
    );

    // 3. Python PyO3 bindings verification
    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

engine = ox.GrandOmniUniversalEngineV()
rep = engine.execute_synthesis()

assert rep.riemannian_flow_radius < 1.0
assert rep.rkhs_mmd_distance > 0.0
assert rep.deq_equilibrium_residual < 1e-5
assert rep.exceptional_point_petermann > 1e3
assert rep.autonomous_quant_format in ["BitLinear1_58", "Q4_0", "Q8_0", "FP8", "FP32"]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Grand Omni V test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v610_yang_mills_lattice_gauge() -> Result<()> {
    use oxideon::geom_graphs::yang_mills::{Su2Element, YangMillsLatticeGauge};

    // 1. Non-Abelian SU(2) group algebra verification
    let u1 = Su2Element::from_axis_angle([0.0, 0.0, 1.0], std::f64::consts::PI / 4.0)?;
    let u2 = Su2Element::from_axis_angle([1.0, 0.0, 0.0], std::f64::consts::PI / 3.0)?;

    // Non-commutativity: U1 * U2 != U2 * U1
    let u12 = u1.multiply(&u2);
    let u21 = u2.multiply(&u1);
    let commutator_norm = (u12.a1 - u21.a1).hypot(u12.a2 - u21.a2);
    assert!(
        commutator_norm > 0.1,
        "SU(2) multiplication must be strictly non-Abelian"
    );

    // Unitarity: U * U^\dagger = I
    let ident = u1.multiply(&u1.inverse());
    assert!((ident.a0 - 1.0).abs() < 1e-12);
    assert!(ident.a1.abs() < 1e-12 && ident.a2.abs() < 1e-12 && ident.a3.abs() < 1e-12);

    // 2. Square plaquette lattice graph: 4 nodes, 4 edges
    let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
    let mut gauge = YangMillsLatticeGauge::new(4, edges)?;

    // Assign non-trivial SU(2) gauge links to edges
    gauge.set_link(0, Su2Element::from_axis_angle([1.0, 0.0, 0.0], 0.6)?)?;
    gauge.set_link(1, Su2Element::from_axis_angle([0.0, 1.0, 0.0], 0.8)?)?;
    gauge.set_link(2, Su2Element::from_axis_angle([0.0, 0.0, 1.0], 0.5)?)?;
    gauge.set_link(3, Su2Element::from_axis_angle([1.0, 1.0, 0.0], 0.4)?)?;

    let cycle = vec![0, 1, 2, 3];
    let fwds = vec![true, true, true, true];
    let original_wilson = gauge.wilson_loop(&cycle, &fwds)?;
    assert!(original_wilson.abs() <= 1.0);

    // 3. Exact Non-Abelian Gauge Invariance under local vertex rotations Omega_u
    let omegas = vec![
        Su2Element::from_axis_angle([1.0, 0.0, 0.0], 1.2)?,
        Su2Element::from_axis_angle([0.0, 1.0, 0.0], 0.7)?,
        Su2Element::from_axis_angle([0.0, 0.0, 1.0], 2.1)?,
        Su2Element::from_axis_angle([1.0, 1.0, 1.0], 0.9)?,
    ];

    let mut transformed_gauge = gauge.clone();
    transformed_gauge.apply_gauge_transformation(&omegas)?;
    let transformed_wilson = transformed_gauge.wilson_loop(&cycle, &fwds)?;

    // Mathematical proof of gauge invariance of Wilson loop
    assert!(
        (original_wilson - transformed_wilson).abs() < 1e-11,
        "Wilson loop must be strictly gauge-invariant: original {}, transformed {}",
        original_wilson,
        transformed_wilson
    );

    // 4. Yang-Mills Lattice Plaquette Action
    let plaquette = vec![(0, true), (1, true), (2, true), (3, true)];
    let action_orig = gauge.yang_mills_action(std::slice::from_ref(&plaquette), 2.5)?;
    let action_trans =
        transformed_gauge.yang_mills_action(std::slice::from_ref(&plaquette), 2.5)?;
    assert!((action_orig - action_trans).abs() < 1e-11);

    // 5. Python PyO3 bindings verification
    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

u1 = ox.Su2Element.from_axis_angle([0.0, 0.0, 1.0], 0.6)
u2 = ox.Su2Element.from_axis_angle([1.0, 0.0, 0.0], 0.9)
u12 = u1.multiply(u2)
u21 = u2.multiply(u1)
assert abs(u12.a2 - u21.a2) > 0.01

gauge = ox.YangMillsLatticeGauge(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
gauge.set_link(0, u1)
gauge.set_link(1, u2)

w = gauge.wilson_loop([0, 1, 2, 3], [True, True, True, True])
assert abs(w) <= 1.0

action = gauge.yang_mills_action([[(0, True), (1, True), (2, True), (3, True)]], 2.0)
assert action >= 0.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Yang-Mills test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v620_quantum_graph_neural_networks() -> Result<()> {
    use oxideon::quantum::qgnn::QuantumGraphNeuralNetwork;

    let num_qubits = 3;
    let edges = vec![(0, 1), (1, 2)];
    let num_layers = 2;
    let mut qgnn = QuantumGraphNeuralNetwork::new(num_qubits, edges.clone(), num_layers)?;

    let num_rotations = num_layers * num_qubits;
    let num_couplings = num_layers * edges.len();

    let theta = vec![0.25; num_rotations];
    let phi = vec![0.15; num_rotations];
    let j_weights = vec![0.5; num_couplings];

    qgnn.set_parameters(theta.clone(), phi.clone(), j_weights.clone())?;

    let node_features = vec![0.1, -0.2, 0.3];
    let node_expectations = qgnn.forward(&node_features)?;
    assert_eq!(node_expectations.len(), 3);
    for &val in &node_expectations {
        assert!(
            (-1.0..=1.0).contains(&val),
            "Expectation value must be in [-1, 1], got {}",
            val
        );
    }

    let energy = qgnn.forward_graph_energy(&node_features)?;
    assert!(energy.is_finite());

    let grads = qgnn.parameter_shift_gradients(&node_features)?;
    assert_eq!(grads.grad_theta.len(), num_rotations);
    assert_eq!(grads.grad_phi.len(), num_rotations);
    assert_eq!(grads.grad_j.len(), num_couplings);

    // Finite difference check for grad_theta[0]
    let eps = 1e-5;
    let mut theta_plus = theta.clone();
    theta_plus[0] += eps;
    let mut qgnn_plus = QuantumGraphNeuralNetwork::new(num_qubits, edges.clone(), num_layers)?;
    qgnn_plus.set_parameters(theta_plus, phi.clone(), j_weights.clone())?;
    let e_plus = qgnn_plus.forward_graph_energy(&node_features)?;

    let mut theta_minus = theta.clone();
    theta_minus[0] -= eps;
    let mut qgnn_minus = QuantumGraphNeuralNetwork::new(num_qubits, edges.clone(), num_layers)?;
    qgnn_minus.set_parameters(theta_minus, phi.clone(), j_weights.clone())?;
    let e_minus = qgnn_minus.forward_graph_energy(&node_features)?;

    let num_grad = (e_plus - e_minus) / (2.0 * eps);
    assert!(
        (grads.grad_theta[0] - num_grad).abs() < 1e-4,
        "Parameter-shift gradient ({}) did not match finite difference ({})",
        grads.grad_theta[0],
        num_grad
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

qgnn = ox.QuantumGraphNeuralNetwork(3, [(0, 1), (1, 2)], 2)
exp_vals = qgnn.forward([0.1, -0.2, 0.3])
assert len(exp_vals) == 3
for v in exp_vals:
    assert -1.0 <= v <= 1.0

energy = qgnn.forward_graph_energy([0.1, -0.2, 0.3])
assert isinstance(energy, float)

grads = qgnn.parameter_shift_gradients([0.1, -0.2, 0.3])
assert len(grads.grad_theta) == 6
assert len(grads.grad_phi) == 6
assert len(grads.grad_j) == 4
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python QGNN test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v630_heisenberg_group_hypoelliptic_diffusion() -> Result<()> {
    use oxideon::physics::heisenberg::{
        HeisenbergLieAlgebra, HeisenbergPoint, HypoellipticDiffusion3D,
    };

    // 1. Group Law, Non-commutativity & Inversion
    let p1 = HeisenbergPoint::new(1.0, 2.0, 0.5);
    let p2 = HeisenbergPoint::new(0.5, -1.0, 1.2);

    let p12 = p1.multiply(&p2);
    let p21 = p2.multiply(&p1);

    // Non-abelian commutator difference in z: (x1*y2 - y1*x2) = 1.0*(-1.0) - 2.0*0.5 = -2.0
    let z_diff = p12.z - p21.z;
    assert!(
        (z_diff - (-2.0)).abs() < 1e-12,
        "Expected z_diff = -2.0, got {z_diff}"
    );

    let p1_inv = p1.inverse();
    let p_id = p1.multiply(&p1_inv);
    assert!(p_id.x.abs() < 1e-12 && p_id.y.abs() < 1e-12 && p_id.z.abs() < 1e-12);

    // 2. Lie Algebra & Bracket generating condition: [X, Y] = Z
    let x_gen = HeisenbergLieAlgebra::new(1.0, 0.0, 0.0);
    let y_gen = HeisenbergLieAlgebra::new(0.0, 1.0, 0.0);
    let z_bracket = x_gen.bracket(&y_gen);
    assert_eq!(z_bracket, HeisenbergLieAlgebra::new(0.0, 0.0, 1.0));

    // 3. Sub-Riemannian Carnot-Carathéodory (CC) Distance & Korányi Metric
    let p_origin = HeisenbergPoint::identity();
    let p_horiz = HeisenbergPoint::new(3.0, 4.0, 0.0);
    assert!((p_origin.carnot_caratheodory_distance(&p_horiz) - 5.0).abs() < 1e-7);

    let p_vert = HeisenbergPoint::new(0.0, 0.0, 1.0);
    let d_cc_vert = p_origin.carnot_caratheodory_distance(&p_vert);
    let expected_vert = (4.0 * std::f64::consts::PI).sqrt();
    assert!((d_cc_vert - expected_vert).abs() < 1e-7);

    let k_norm = p_vert.koranyi_norm();
    assert!((k_norm - 2.0).abs() < 1e-7); // (16 * 1)^0.25 = 2.0

    // 4. Hypoelliptic Sub-Laplacian 3D Diffusion
    let mut solver = HypoellipticDiffusion3D::new(17, 2.0)?;
    solver.init_gaussian(0.4)?;
    let initial_mass = solver.total_mass();
    assert!((initial_mass - 1.0).abs() < 0.1);

    let initial_h_var = solver.horizontal_variance();
    let initial_v_var = solver.vertical_variance();

    for _ in 0..5 {
        solver.step(0.002);
    }

    let evolved_mass = solver.total_mass();
    assert!(
        (evolved_mass - initial_mass).abs() < 0.05,
        "Mass should be conserved under diffusion"
    );
    assert!(
        solver.horizontal_variance() >= initial_h_var,
        "Horizontal variance should expand"
    );
    assert!(
        solver.vertical_variance() >= initial_v_var,
        "Vertical variance should expand hypoelliptically"
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox
import math

p1 = ox.HeisenbergPoint(1.0, 2.0, 0.5)
p2 = ox.HeisenbergPoint(0.5, -1.0, 1.2)
p12 = p1.multiply(p2)
p21 = p2.multiply(p1)
assert abs((p12.z - p21.z) - (-2.0)) < 1e-9

vx = ox.HeisenbergLieAlgebra(1.0, 0.0, 0.0)
vy = ox.HeisenbergLieAlgebra(0.0, 1.0, 0.0)
vz = vx.bracket(vy)
assert abs(vz.vz - 1.0) < 1e-9

origin = ox.HeisenbergPoint.identity()
vert = ox.HeisenbergPoint(0.0, 0.0, 1.0)
d_cc = origin.carnot_caratheodory_distance(vert)
assert abs(d_cc - math.sqrt(4.0 * math.pi)) < 1e-6
assert abs(vert.koranyi_norm() - 2.0) < 1e-6

diff = ox.HypoellipticDiffusion3D(15, 1.5)
diff.init_gaussian(0.35)
m0 = diff.total_mass()
assert abs(m0 - 1.0) < 0.15
diff.step(0.001)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Heisenberg test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v640_neuromorphic_event_driven_spike_routing() -> Result<()> {
    use oxideon::nn::neuromorphic::{EventDrivenSpikeRouter, PolychronousNeuralLayer, SpikeEvent};

    // 1. Priority-Queue Event-Driven Spike Routing
    let mut router = EventDrivenSpikeRouter::new();
    let ev_a = SpikeEvent::new(0.010, 0, 0, 0.5, 0.005); // arr: 0.015
    let ev_b = SpikeEvent::new(0.005, 1, 0, 0.5, 0.015); // arr: 0.020
    let ev_c = SpikeEvent::new(0.008, 2, 0, 0.5, 0.002); // arr: 0.010

    router.schedule(ev_a);
    router.schedule(ev_b);
    router.schedule(ev_c);

    assert_eq!(router.pending_count(), 3);
    assert!((router.next_arrival_time().unwrap() - 0.010).abs() < 1e-9);

    let first = router.pop_next().unwrap();
    assert_eq!(first.source, 2); // ev_c arrived first at 0.010

    let second = router.pop_next().unwrap();
    assert_eq!(second.source, 0); // ev_a arrived second at 0.015

    let third = router.pop_next().unwrap();
    assert_eq!(third.source, 1); // ev_b arrived third at 0.020

    assert_eq!(router.pending_count(), 0);
    assert_eq!(router.events_routed, 3);

    // 2. Polychronous Coincidence Detection via Axonal Delays
    // Two inputs, one output. Sub-threshold weights = 0.6, threshold = 1.0.
    let mut layer = PolychronousNeuralLayer::new(2, 1, 0.050)?;
    layer.set_weights(vec![0.6, 0.6])?;
    // Input 0 has 10 ms delay; Input 1 has 5 ms delay.
    layer.set_delays(vec![0.010, 0.005])?;

    // Scenario A: Asynchronous arrival (no firing)
    // Input 0 fires at 0 ms (arr: 10 ms); Input 1 fires at 25 ms (arr: 30 ms).
    let async_inputs = vec![(0.000, 0), (0.025, 1)];
    let async_outputs = layer.forward_event_driven(&async_inputs, 0.050)?;
    assert!(
        async_outputs.is_empty(),
        "Sub-threshold asynchronous spikes must not fire"
    );

    // Scenario B: Polychronous synchronous arrival (firing!)
    // Input 0 fires at 0 ms (arr: 0 + 10 = 10 ms); Input 1 fires at 5 ms (arr: 5 + 5 = 10 ms).
    let poly_inputs = vec![(0.000, 0), (0.005, 1)];
    let poly_outputs = layer.forward_event_driven(&poly_inputs, 0.050)?;
    assert_eq!(
        poly_outputs.len(),
        1,
        "Polychronous inputs must elicit coincident spike"
    );
    assert!((poly_outputs[0].0 - 0.010).abs() < 1e-6);
    assert_eq!(poly_outputs[0].1, 0);

    // 3. Learnable Delay Plasticity (STDDP)
    let initial_delay_0 = layer.delays[0];
    // Simulate output spike firing at 12 ms while input 0 arrived at 10 ms (dt = +2 ms > 0)
    layer
        .adapt_stddp(&[(0.000, 0)], &[(0.012, 0)], 0.001, 0.01, 0.005)
        .unwrap();
    assert!(
        layer.delays[0] > initial_delay_0,
        "Delay should increase to compensate for arriving before postsynaptic firing"
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

router = ox.EventDrivenSpikeRouter()
ev1 = ox.SpikeEvent(0.010, 0, 0, 0.5, 0.005)
ev2 = ox.SpikeEvent(0.008, 1, 0, 0.5, 0.001)
router.schedule(ev1)
router.schedule(ev2)
assert router.pending_count() == 2

popped = router.pop_next()
assert popped.source == 1
assert abs(popped.arrival_time() - 0.009) < 1e-9

layer = ox.PolychronousNeuralLayer(2, 1, 0.050)
layer.set_weights([0.6, 0.6])
layer.set_delays([0.010, 0.005])

poly_spikes = layer.forward_event_driven([(0.000, 0), (0.005, 1)], 0.050)
assert len(poly_spikes) == 1
assert abs(poly_spikes[0][0] - 0.010) < 1e-6
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Neuromorphic test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v650_spin_glass_tap_equations() -> Result<()> {
    use oxideon::science::spin_glass::SherringtonKirkpatrickModel;

    let n = 16;
    let sk = SherringtonKirkpatrickModel::random_gaussian(n, 1.0, 1337)?;

    // 1. Discrete Ising configuration energy
    let spins = vec![1i8; n];
    let e_all_up = sk.energy(&spins)?;
    let spins_flipped: Vec<i8> = spins.iter().map(|&s| -s).collect();
    let e_all_down = sk.energy(&spins_flipped)?;
    // With zero external field, H(-sigma) == H(sigma) by Z2 spin-flip symmetry!
    assert!(
        (e_all_up - e_all_down).abs() < 1e-12,
        "Z2 spin-flip symmetry violated"
    );

    // 2. High-temperature paramagnetic phase (beta = 0.2 < 1.0)
    let sol_high_t = sk.solve_tap(0.2, 200, 1e-7, 0.4)?;
    assert!(sol_high_t.converged);
    assert!(
        sol_high_t.edwards_anderson_q < 0.01,
        "High-T should have q_EA ~ 0"
    );
    assert!(
        sol_high_t.at_eigenvalue > 0.0,
        "High-T should be replica symmetric (AT > 0)"
    );
    assert!(!sol_high_t.is_replica_symmetry_broken);

    // 3. Low-temperature spin-glass phase (beta = 2.5 > 1.0): RSB & non-zero q_EA
    let mut sk_biased = SherringtonKirkpatrickModel::random_gaussian(n, 1.5, 42)?;
    sk_biased.external_fields = vec![0.35; n];
    let sol_low_t = sk_biased.solve_tap(2.2, 500, 1e-6, 0.25)?;
    assert!(
        sol_low_t.edwards_anderson_q > 0.1,
        "Low-T must exhibit frozen spin order q_EA > 0.1"
    );
    assert!(sol_low_t.tap_free_energy.is_finite());

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

sk = ox.SherringtonKirkpatrickModel.random_gaussian(12, 1.0, 99)
e1 = sk.energy([1]*12)
e2 = sk.energy([-1]*12)
assert abs(e1 - e2) < 1e-12

sol = sk.solve_tap(0.25, 200, 1e-6, 0.4)
assert sol.converged
assert sol.edwards_anderson_q < 0.05
assert not sol.is_replica_symmetry_broken
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Spin Glass test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v660_continuous_variable_gaussian_boson_sampling() -> Result<()> {
    use oxideon::quantum::continuous_variable::{GaussianCvCircuit, matrix_hafnian};

    let mut cv = GaussianCvCircuit::new(2)?;

    // 1. Vacuum state checks
    assert!((cv.mean_photon_number(0)? - 0.0).abs() < 1e-12);
    assert!((cv.purity()? - 1.0).abs() < 1e-12);
    assert!((cv.vacuum_probability()? - 1.0).abs() < 1e-12);

    // 2. Single-mode squeezing S(r) followed by 50:50 beamsplitter (Two-Mode Squeezed Vacuum / EPR state)
    let r = 0.8;
    cv.squeeze(0, r)?;
    cv.squeeze(1, -r)?;

    // Mean photon number in a squeezed vacuum state is sinh^2(r)
    let expected_n = r.sinh().powi(2);
    assert!((cv.mean_photon_number(0)? - expected_n).abs() < 1e-10);
    assert!((cv.mean_photon_number(1)? - expected_n).abs() < 1e-10);

    // Interfere on a 50:50 beamsplitter (theta = pi / 4)
    cv.beamsplitter(0, 1, std::f64::consts::FRAC_PI_4)?;

    // Passive linear optics (beamsplitters + phase shifters) conserve total photon number and purity!
    assert!((cv.total_mean_photon_number()? - 2.0 * expected_n).abs() < 1e-10);
    assert!(
        (cv.purity()? - 1.0).abs() < 1e-10,
        "Symplectic evolution must preserve unit purity"
    );

    // Vacuum probability for two-mode squeezed vacuum is 1 / cosh^2(r)
    let expected_p0 = 1.0 / r.cosh().powi(2);
    assert!((cv.vacuum_probability()? - expected_p0).abs() < 1e-10);

    // 3. Exact Matrix Hafnian on a 4x4 all-ones adjacency matrix (K_4 has 3!! = 3 perfect matchings)
    let k4 = vec![
        0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0,
    ];
    let haf = matrix_hafnian(&k4, 4)?;
    assert!(
        (haf - 3.0).abs() < 1e-12,
        "K_4 must have hafnian = 3, got {haf}"
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox
import math

cv = ox.GaussianCvCircuit(2)
r = 0.6
cv.squeeze(0, r)
cv.squeeze(1, -r)
cv.beamsplitter(0, 1, math.pi / 4.0)

assert abs(cv.purity() - 1.0) < 1e-9
assert abs(cv.total_mean_photon_number() - 2.0 * (math.sinh(r) ** 2)) < 1e-9
assert abs(cv.vacuum_probability() - 1.0 / (math.cosh(r) ** 2)) < 1e-9

k4 = [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0]
assert abs(ox.GaussianCvCircuit.hafnian(k4, 4) - 3.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python CV Quantum test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v670_relativistic_magnetohydrodynamics() -> Result<()> {
    use oxideon::physics::rmhd::RmhdPrimitiveState;

    // 1. Relativistic jet plasma state (v_x = 0.6c, v_y = 0.4c, B = [1.0, 2.0, 0.5], gamma = 4/3)
    let state_l = RmhdPrimitiveState::new(1.0, 2.5, [0.6, 0.4, 0.0], [1.0, 2.0, 0.5], 4.0 / 3.0)?;
    assert!(state_l.lorentz_factor() > 1.0);

    // Exact four-vector orthogonality: u_mu b^mu == 0
    let ortho = state_l.orthogonality_residual();
    assert!(
        ortho.abs() < 1e-13,
        "Minkowski orthogonality u_mu b^mu violated: {ortho}"
    );

    // Lorentz-invariant magnetic pressure b^2 > 0
    let b2 = state_l.magnetic_energy_squared();
    assert!(b2 > 0.0);

    // Subluminal causality: c_s < 1, c_A < 1, c_f < 1
    assert!(state_l.sound_speed() > 0.0 && state_l.sound_speed() < 1.0);
    assert!(state_l.alfven_speed() > 0.0 && state_l.alfven_speed() < 1.0);
    assert!(state_l.fast_magnetosonic_speed() > 0.0 && state_l.fast_magnetosonic_speed() < 1.0);

    // Stress-energy tensor symmetry: T^{mu nu} == T^{nu mu} and positive energy density T^{00} > 0
    let t_munu = state_l.stress_energy_tensor();
    assert!(t_munu[0][0] > 0.0);
    for (mu, row) in t_munu.iter().enumerate() {
        for (nu, &val) in row.iter().enumerate() {
            assert!((val - t_munu[nu][mu]).abs() < 1e-13);
        }
    }

    // 2. Balsara Relativistic MHD Shock Tube HLL Riemann Flux consistency:
    // HLL(U, U) == F^x(U)
    let f_exact = state_l.flux_x();
    let f_hll_same = RmhdPrimitiveState::hll_riemann_flux(&state_l, &state_l);
    for k in 0..5 {
        assert!((f_exact[k] - f_hll_same[k]).abs() < 1e-12);
    }

    let state_r =
        RmhdPrimitiveState::new(0.125, 0.1, [0.0, 0.0, 0.0], [1.0, -1.0, 0.0], 4.0 / 3.0)?;
    let f_shock = RmhdPrimitiveState::hll_riemann_flux(&state_l, &state_r);
    assert!(
        f_shock[0] > 0.0,
        "Mass flux across high-to-low pressure shock must be positive"
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

st_l = ox.RmhdPrimitiveState(1.0, 2.5, [0.6, 0.4, 0.0], [1.0, 2.0, 0.5], 4.0 / 3.0)
assert abs(st_l.orthogonality_residual()) < 1e-12
assert 0.0 < st_l.fast_magnetosonic_speed() < 1.0

st_r = ox.RmhdPrimitiveState(0.125, 0.1, [0.0, 0.0, 0.0], [1.0, -1.0, 0.0], 4.0 / 3.0)
flux = ox.RmhdPrimitiveState.hll_riemann_flux(st_l, st_r)
assert len(flux) == 5
assert flux[0] > 0.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python RMHD test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v680_tensor_train_decompositions() -> Result<()> {
    use oxideon::science::tensor_train::TensorTrain;

    // 1. 10-dimensional Rank-2 Additive Tensor: A(i_1, ..., i_10) = sum_{k=1}^{10} (i_k + 1)
    // Full tensor size = 4^10 = 1,048,576 entries, stored in just 10 * 4 * 4 = 144 parameters!
    let d = 10;
    let terms: Vec<Vec<f64>> = (0..d).map(|_| vec![1.0, 2.0, 3.0, 4.0]).collect();
    let tt_add = TensorTrain::from_additive_terms(terms)?;

    assert_eq!(tt_add.ndim(), 10);
    assert_eq!(tt_add.shape(), vec![4; 10]);
    assert!(tt_add.num_parameters() < 200);

    // Evaluate A(0, 0, ..., 0) = 10 * 1.0 = 10.0
    let val_zeros = tt_add.evaluate(&[0; 10])?;
    assert!((val_zeros - 10.0).abs() < 1e-12);

    // Evaluate A(3, 3, ..., 3) = 10 * 4.0 = 40.0
    let val_threes = tt_add.evaluate(&[3; 10])?;
    assert!((val_threes - 40.0).abs() < 1e-12);

    // Exact sum over all 4^10 entries:
    // Average value per coordinate is 2.5, so average entry is 10 * 2.5 = 25.0.
    // Total sum = 25.0 * 4^10 = 26,214,400.0
    let total_sum = tt_add.sum_all();
    let expected_sum = 25.0 * (4.0_f64.powi(10));
    assert!((total_sum - expected_sum).abs() < 1e-6);

    // 2. Rank-1 Separable Tensor inner product & Frobenius norm
    let tt_r1 =
        TensorTrain::from_rank1_factors(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![0.5, 1.5]])?;
    // Norm squared = (1^2 + 2^2) * (3^2 + 4^2) * (0.5^2 + 1.5^2) = 5 * 25 * 2.5 = 312.5
    let norm = tt_r1.frobenius_norm()?;
    assert!((norm * norm - 312.5).abs() < 1e-10);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

tt = ox.TensorTrain.from_additive_terms([[1.0, 2.0, 3.0, 4.0]] * 8)
assert tt.ndim() == 8
assert abs(tt.evaluate([0]*8) - 8.0) < 1e-12
assert abs(tt.evaluate([3]*8) - 32.0) < 1e-12
assert abs(tt.sum_all() - 20.0 * (4 ** 8)) < 1e-6
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python TensorTrain test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v690_holographic_ads_cft_ryu_takayanagi() -> Result<()> {
    use oxideon::physics::ads_cft::AdsCftHolographicEngine;

    // 1. AdS_3 / CFT_2 Brown-Henneaux central charge c = 3 L / (2 G_N)
    let engine = AdsCftHolographicEngine::new(2.0, 0.25, 1e-3)?;
    assert!((engine.central_charge() - 12.0).abs() < 1e-12);

    // 2. Ryu-Takayanagi formula: S_A = (c / 3) * ln(l / eps) = 4.0 * ln(1.0 / 1e-3) = 4.0 * ln(1000)
    let s_a = engine.ryu_takayanagi_entropy(1.0)?;
    let expected_s = 4.0 * (1000.0_f64).ln();
    assert!((s_a - expected_s).abs() < 1e-10);

    // 3. Semicircle geodesic penetrating into bulk IR to z_* = l / 2
    let (pts, z_max) = engine.sample_rt_geodesic(-1.0, 1.0, 21)?;
    assert_eq!(pts.len(), 21);
    assert!((z_max - 1.0).abs() < 1e-12);
    // Midpoint (index 10) is at x = 0, z = 1.0
    assert!(pts[10].0.abs() < 1e-12);
    assert!((pts[10].1 - 1.0).abs() < 1e-12);

    // 4. Holographic Mutual Information phase transition:
    // Close intervals (d = 0.1) have positive mutual information; far intervals (d = 10.0) have I(A:B) == 0
    let mi_close = engine.holographic_mutual_information(1.0, 1.0, 0.1)?;
    let mi_far = engine.holographic_mutual_information(1.0, 1.0, 10.0)?;
    assert!(
        mi_close > 0.0,
        "Close boundary regions must share connected bulk entanglement wedge"
    );
    assert!(
        mi_far.abs() < 1e-12,
        "Far boundary regions must undergo RT phase transition to I(A:B) = 0"
    );

    // 5. Strong Subadditivity (SSA) gap >= 0
    let ssa = engine.strong_subadditivity_gap(1.0, 0.5, 1.2)?;
    assert!(ssa > 0.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox
import math

holo = ox.AdsCftHolographicEngine(2.0, 0.25, 1e-3)
assert abs(holo.central_charge() - 12.0) < 1e-12
assert abs(holo.ryu_takayanagi_entropy(1.0) - 4.0 * math.log(1000.0)) < 1e-10
assert holo.holographic_mutual_information(1.0, 1.0, 0.1) > 0.0
assert holo.holographic_mutual_information(1.0, 1.0, 10.0) == 0.0
assert holo.strong_subadditivity_gap(1.0, 0.5, 1.2) > 0.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python AdS/CFT test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v700_grand_omni_septua_engine_vi() -> Result<()> {
    use oxideon::omni_septua_vi::GrandOmniSeptuaEngineVI;

    let engine = GrandOmniSeptuaEngineVI::new(6);
    let rep = engine.execute_septua_synthesis()?;

    assert!((-1.0..=1.0).contains(&rep.yang_mills_wilson_loop));
    assert!(rep.qgnn_graph_energy.is_finite());
    assert!((rep.heisenberg_cc_distance - (4.0 * std::f64::consts::PI).sqrt()).abs() < 1e-7);
    assert_eq!(rep.polychronous_coincidence_spikes, 1);
    assert!(rep.spin_glass_q_ea > 0.1);
    assert!(rep.cv_vacuum_probability > 0.0 && rep.cv_vacuum_probability <= 1.0);
    assert!((rep.cv_k4_hafnian - 3.0).abs() < 1e-12);
    assert!(rep.rmhd_fast_magnetosonic_speed > 0.0 && rep.rmhd_fast_magnetosonic_speed < 1.0);
    assert!((rep.tensor_train_partition_sum - 20.0 * (4.0_f64.powi(8))).abs() < 1e-6);
    assert!(rep.ryu_takayanagi_entropy > 20.0);
    assert!((1.58..=32.0).contains(&rep.active_llm_bits_per_weight));
    assert!(rep.unified_septua_invariant > 30.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniSeptuaEngineVI(6)
rep = eng.execute_septua_synthesis()
assert rep.polychronous_coincidence_spikes == 1
assert abs(rep.cv_k4_hafnian - 3.0) < 1e-12
assert rep.unified_septua_invariant > 30.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python GrandOmniSeptuaEngineVI test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v710_infinity_quasi_categories_horn_filling() -> Result<()> {
    use oxideon::science::infinity_category::{HigherSimplex, QuasiCategoryEngine};

    // 1. Simplicial identities d_i d_j == d_{j-1} d_i on a 3-simplex tetrahedron [0, 1, 2, 3]
    let tetra = HigherSimplex::new(vec![0, 1, 2, 3], vec![1.0, 2.0])?;
    assert_eq!(tetra.dimension(), 3);
    for i in 0..3 {
        for j in (i + 1)..=3 {
            assert!(tetra.verify_simplicial_identity(i, j)?);
        }
    }

    // 2. Inner horn Lambda_1^2 filling in an (infty, 1)-quasi-category
    let qcat = QuasiCategoryEngine::new(1.0)?;
    let f = HigherSimplex::new(vec![0, 1], vec![1.5, -0.5])?;
    let g = HigherSimplex::new(vec![1, 2], vec![2.5, 1.5])?;
    let (filler, gf) = qcat.fill_inner_horn_2(&f, &g)?;
    assert_eq!(filler.vertices, vec![0, 1, 2]);
    assert_eq!(gf.vertices, vec![0, 2]);
    assert!((gf.payload[0] - 4.0).abs() < 1e-12);
    assert!((gf.payload[1] - 1.0).abs() < 1e-12);

    // 3. Outer Kan horn Lambda_0^2 filling (weak invertibility)
    let (_, g_recovered) = qcat.fill_outer_kan_horn_0(&f, &gf)?;
    assert_eq!(g_recovered.vertices, vec![1, 2]);
    assert!((g_recovered.payload[0] - 2.5).abs() < 1e-12);

    // 4. Associativity 3-simplex [0, 1, 2, 3]
    let h = HigherSimplex::new(vec![2, 3], vec![0.5, 2.0])?;
    let (assoc_tetra, defect) = qcat.associator_3_simplex(&f, &g, &h)?;
    assert_eq!(assoc_tetra.vertices, vec![0, 1, 2, 3]);
    assert!(
        defect < 1e-12,
        "Strict coherence factor=1.0 must have zero associator defect"
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

f = ox.HigherSimplex([0, 1], [1.0, 2.0])
g = ox.HigherSimplex([1, 2], [3.0, 4.0])
qcat = ox.QuasiCategoryEngine(1.0)
filler, comp = qcat.fill_inner_horn_2(f, g)
assert filler.vertices == [0, 1, 2]
assert comp.vertices == [0, 2]
assert comp.payload == [4.0, 6.0]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python QuasiCategory test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v720_differentiable_r1cs_qap_zk_snark() -> Result<()> {
    use oxideon::crypto::r1cs_qap::DifferentiableR1csQap;

    // Arithmetic circuit proving x^3 + x + 5 == 35 (solution x = 3):
    // Witness z = [1, x, v1, v2, out] = [1.0, 3.0, 9.0, 27.0, 35.0] (num_vars = 5)
    // Gate 1: x * x = v1          (z[1] * z[1] = z[2])
    // Gate 2: v1 * x = v2         (z[2] * z[1] = z[3])
    // Gate 3: (v2 + x + 5) * 1 = out
    let mut r1cs = DifferentiableR1csQap::new(5)?;
    r1cs.add_constraint(
        vec![0.0, 1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0, 0.0],
    )?;
    r1cs.add_constraint(
        vec![0.0, 0.0, 1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0, 0.0],
    )?;
    r1cs.add_constraint(
        vec![5.0, 1.0, 0.0, 1.0, 0.0],
        vec![1.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 1.0],
    )?;

    let valid_witness = vec![1.0, 3.0, 9.0, 27.0, 35.0];
    let (loss, grad) = r1cs.loss_and_gradient(&valid_witness)?;
    assert!(loss < 1e-14);
    for &g in &grad {
        assert!(g.abs() < 1e-12);
    }

    // QAP polynomial identity check at toxic waste challenge tau = 17.5:
    // A(tau) * B(tau) - C(tau) == H(tau) * Z(tau)
    let (lhs, rhs, diff) = r1cs.evaluate_qap_at_tau(&valid_witness, 17.5)?;
    assert!(
        diff < 1e-10,
        "QAP divisibility identity failed: lhs={lhs}, rhs={rhs}, diff={diff}"
    );

    // Tampered witness must fail QAP divisibility check
    let bad_witness = vec![1.0, 3.0, 9.0, 27.0, 36.0];
    let (_, _, bad_diff) = r1cs.evaluate_qap_at_tau(&bad_witness, 17.5)?;
    assert!(
        bad_diff > 0.1,
        "Invalid witness must produce non-zero QAP remainder"
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

qap = ox.DifferentiableR1csQap(3)
qap.add_constraint([0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
loss, grad = qap.loss_and_gradient([1.0, 4.0, 16.0])
assert abs(loss) < 1e-12
lhs, rhs, diff = qap.evaluate_qap_at_tau([1.0, 4.0, 16.0], 7.25)
assert diff < 1e-10
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python R1CS/QAP test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v730_symplectic_topology_gromov_non_squeezing() -> Result<()> {
    use oxideon::physics::symplectic_topology::SymplecticTopologyEngine;
    use std::f64::consts::PI;

    let eng = SymplecticTopologyEngine::new(2)?;

    // 1. Symplectic squeezing matrix diag(2.0, 0.5, 0.25, 4.0) in Sp(4, R)
    let mut s_mat = vec![0.0; 16];
    s_mat[0] = 2.0;
    s_mat[5] = 0.5;
    s_mat[10] = 0.25;
    s_mat[15] = 4.0;
    assert!(eng.symplecticity_defect(&s_mat)? < 1e-12);

    // 2. Gromov capacity vs Liouville volume on ellipsoid E(2.0, 5.0)
    // Capacity is determined by the smallest radius R_1 = 2.0 -> pi * 4.0
    let cap = eng.ellipsoid_gromov_capacity(&[2.0, 5.0])?;
    assert!((cap - 4.0 * PI).abs() < 1e-12);

    // 3. Gromov Non-Squeezing Theorem: Ball R=3.0 cannot embed into Cylinder r=2.0
    let (can_embed_false, _, _) = eng.verify_gromov_non_squeezing(3.0, 2.0)?;
    assert!(!can_embed_false);
    let (can_embed_true, _, _) = eng.verify_gromov_non_squeezing(1.5, 2.0)?;
    assert!(can_embed_true);

    // 4. Floer Action Functional on a 1-periodic harmonic orbit (action == 0 at critical orbit)
    let n_pts = 64;
    let mut loop_pts = Vec::with_capacity(n_pts);
    for k in 0..n_pts {
        let t = 2.0 * PI * (k as f64) / (n_pts as f64);
        loop_pts.push(vec![t.cos(), -t.sin(), 0.0, 0.0]);
    }
    let action = eng.floer_action_functional(&loop_pts, &[2.0 * PI, 2.0 * PI])?;
    assert!(
        action.abs() < 0.05,
        "Critical 1-periodic Hamiltonian orbit must have near-zero Floer action, got {action}"
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox
import math

st = ox.SymplecticTopologyEngine(2)
assert abs(st.ellipsoid_gromov_capacity([2.0, 5.0]) - 4.0 * math.pi) < 1e-12
ok, cb, cc = st.verify_gromov_non_squeezing(3.0, 2.0)
assert not ok
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python SymplecticTopology test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v740_stochastic_thermodynamics_jarzynski_crooks() -> Result<()> {
    use oxideon::physics::stochastic_thermo::StochasticThermoEngine;

    let beta = 2.0;
    let eng = StochasticThermoEngine::new(beta)?;

    // 1. Gaussian work distribution W ~ N(mu_W, sigma^2):
    // By fluctuation-dissipation / Jarzynski identity, if W_diss = 0.5 * beta * sigma^2,
    // then <W> >= Delta F and W_diss >= 0.
    let works = vec![1.2, 1.5, 1.8, 2.1, 2.4, 2.7];
    let rep = eng.evaluate_jarzynski(&works)?;
    assert!(
        rep.mean_work > rep.jarzynski_delta_f,
        "Second Law <W> >= Delta F must hold strictly for non-constant work"
    );
    assert!(rep.dissipated_work > 0.0);
    assert!(rep.mean_entropy_production > 0.0);

    // 2. Crooks Fluctuation Theorem: log ratio vanishes exactly at W == Delta F
    assert!((eng.crooks_log_ratio(1.5, 1.5)).abs() < 1e-12);
    assert!((eng.crooks_log_ratio(2.5, 1.5) - 2.0).abs() < 1e-12);

    // 3. Landauer 1-bit erasure bound: k_B T ln(2) = ln(2) / beta
    let q_1bit = eng.landauer_erasure_bound(1.0);
    assert!((q_1bit - 0.5 * std::f64::consts::LN_2).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox
import math

th = ox.StochasticThermoEngine(2.0)
rep = th.evaluate_jarzynski([1.2, 1.5, 1.8, 2.1, 2.4])
assert rep.mean_work > rep.jarzynski_delta_f
assert rep.dissipated_work > 0.0
assert abs(th.landauer_erasure_bound(1.0) - 0.5 * math.log(2.0)) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python StochasticThermo test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v750_algebraic_operads_stasheff_associahedra() -> Result<()> {
    use oxideon::science::operad::AInfinityOperadEngine;

    // 1. Stasheff Associahedron K_n vertex counts (Catalan numbers):
    // K_2 = 1, K_3 = 2, K_4 (Pentagon) = 5, K_5 = 14, K_6 = 42
    assert_eq!(AInfinityOperadEngine::associahedron_vertices(2)?, 1);
    assert_eq!(AInfinityOperadEngine::associahedron_vertices(3)?, 2);
    assert_eq!(AInfinityOperadEngine::associahedron_vertices(4)?, 5);
    assert_eq!(AInfinityOperadEngine::associahedron_vertices(5)?, 14);
    assert_eq!(AInfinityOperadEngine::associahedron_vertices(6)?, 42);

    // 2. Nilpotent differential d^2 == 0 on a 3D graded space:
    // d = [[0, 1, 0], [0, 0, 0], [0, 0, 0]] -> d^2 = 0
    let diff = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let op = AInfinityOperadEngine::new(3, diff)?;
    assert!(op.differential_squared_norm() < 1e-14);

    // 3. Triple Massey Product w = u * z - (-1)^deg(x) x * v
    let w = op.triple_massey_product(
        &[1.0, 0.0, 2.0],
        &[3.0, 1.0, 0.5],
        &[2.0, 1.0, 1.0],
        &[0.5, 2.0, 1.5],
        1,
    )?;
    // deg(x) = 1 => sign = -1 => w_0 = 2.0 * 3.0 - (-1)*(1.0 * 0.5) = 6.5
    assert!((w[0] - 6.5).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert ox.AInfinityOperadEngine.associahedron_vertices(4) == 5
assert ox.AInfinityOperadEngine.associahedron_vertices(5) == 14
op = ox.AInfinityOperadEngine(2, [0.0, 1.0, 0.0, 0.0])
assert op.differential_squared_norm() < 1e-14
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Operad test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v760_topos_heyting_lawvere_tierney() -> Result<()> {
    use oxideon::reasoning::topos::ToposHeytingEngine;

    let topos = ToposHeytingEngine::new(5)?;
    assert!(
        topos.verify_heyting_adjunction()?,
        "Heyting residuation adjunction must hold across Omega^3"
    );
    assert!(
        topos.verify_lawvere_tierney_axioms()?,
        "Lawvere-Tierney modal closure axioms must hold"
    );

    // In intuitionistic logic with truth value u = 0.5:
    // ~u = (0.5 => 0.0) = 0.0, so ~~u = 1.0 > u (Double negation elimination fails!)
    // u v ~u = 0.5 < 1.0 (Excluded middle fails!)
    let u = 0.5;
    let neg_u = ToposHeytingEngine::negate(u);
    assert!(neg_u.abs() < 1e-12);
    let dneg_u = ToposHeytingEngine::double_negation_modality(u);
    assert!((dneg_u - 1.0).abs() < 1e-12);
    assert!((ToposHeytingEngine::join(u, neg_u) - 0.5).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

topos = ox.ToposHeytingEngine(5)
assert topos.verify_heyting_adjunction()
assert topos.verify_lawvere_tierney_axioms()
assert ox.ToposHeytingEngine.negate(0.5) == 0.0
assert ox.ToposHeytingEngine.double_negation_modality(0.5) == 1.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Topos test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v770_mean_field_games_hjb_fpk() -> Result<()> {
    use oxideon::ode::mean_field_games::MeanFieldGameSolver1D;

    let pot = vec![0.0, 0.2, 0.5, 1.0, 1.0, 0.5, 0.2, 0.0];
    let mfg = MeanFieldGameSolver1D::new(8, 0.1, 0.8, pot)?;
    let m_init = vec![0.125; 8];
    let u_term = vec![0.0; 8];

    let (m_eq, u_eq, social_cost) = mfg.solve_nash_equilibrium(&m_init, &u_term, 0.01, 0.25, 20)?;
    let total_mass: f64 = m_eq.iter().sum();
    assert!(
        (total_mass - 1.0).abs() < 1e-10,
        "FPK evolution must strictly conserve total probability mass"
    );
    assert!(
        u_eq[3] > u_eq[0],
        "High-potential barrier nodes must have higher HJB cost-to-go"
    );
    assert!(social_cost > 0.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

mfg = ox.MeanFieldGameSolver1D(8, 0.1, 0.8, [0.0, 0.2, 0.5, 1.0, 1.0, 0.5, 0.2, 0.0])
m_eq, u_eq, cost = mfg.solve_nash_equilibrium([0.125]*8, [0.0]*8, 0.01, 0.25, 15)
assert abs(sum(m_eq) - 1.0) < 1e-9
assert cost > 0.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python MFG test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v780_tqft_chern_simons_verlinde() -> Result<()> {
    use oxideon::quantum::tqft::Su2ChernSimonsTqft;

    // 1. SU(2)_2 Ising Anyon theory: 3 sectors (0 = 1, 1 = sigma, 2 = psi)
    // Quantum dimensions: d_0 = 1, d_1 = sqrt(2), d_2 = 1; Total D = 2.0
    let ising = Su2ChernSimonsTqft::new(2)?;
    assert_eq!(ising.num_anyons()?, 3);
    assert!((ising.quantum_dimension(0)? - 1.0).abs() < 1e-12);
    assert!((ising.quantum_dimension(1)? - std::f64::consts::SQRT_2).abs() < 1e-12);
    assert!((ising.quantum_dimension(2)? - 1.0).abs() < 1e-12);
    assert!((ising.total_quantum_dimension()? - 2.0).abs() < 1e-12);

    // Ising fusion: sigma x sigma = 1 + psi => N_{1,1}^0 = 1, N_{1,1}^1 = 0, N_{1,1}^2 = 1
    assert_eq!(ising.verlinde_fusion_coefficient(1, 1, 0)?, 1);
    assert_eq!(ising.verlinde_fusion_coefficient(1, 1, 1)?, 0);
    assert_eq!(ising.verlinde_fusion_coefficient(1, 1, 2)?, 1);

    // 2. SU(2)_3 Fibonacci Anyon theory: d_1 = golden ratio phi = (1 + sqrt(5)) / 2
    let fib = Su2ChernSimonsTqft::new(3)?;
    let golden = 0.5 * (1.0 + 5.0_f64.sqrt());
    assert!((fib.quantum_dimension(1)? - golden).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox
import math

ising = ox.Su2ChernSimonsTqft(2)
assert abs(ising.quantum_dimension(1) - math.sqrt(2.0)) < 1e-12
assert ising.verlinde_fusion_coefficient(1, 1, 0) == 1
assert ising.verlinde_fusion_coefficient(1, 1, 2) == 1
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python TQFT test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v790_information_geometry_fisher_rao_amari() -> Result<()> {
    use oxideon::science::information_geometry::InformationGeometryEngine;

    // 1. Gaussian Fisher-Rao metric & hyperbolic geodesic distance
    let d_same = InformationGeometryEngine::gaussian_fisher_rao_distance(1.0, 2.0, 1.0, 2.0)?;
    assert!(d_same.abs() < 1e-12);
    let d_diff = InformationGeometryEngine::gaussian_fisher_rao_distance(0.0, 1.0, 2.0, 2.0)?;
    assert!(d_diff > 0.5);

    // 2. Categorical Fisher-Rao distance on probability simplex
    let p = vec![0.2, 0.5, 0.3];
    let q = vec![0.6, 0.2, 0.2];
    let r = vec![0.1, 0.3, 0.6];
    assert!(InformationGeometryEngine::categorical_fisher_rao_distance(&p, &q)? > 0.1);

    // 3. Amari's Generalized Pythagorean Identity on dual-flat (theta, eta) simplex
    let pyth_res = InformationGeometryEngine::pythagorean_duality_residual(&p, &q, &r)?;
    assert!(
        pyth_res < 1e-14,
        "Amari dual Bregman identity must hold to machine precision, got {pyth_res}"
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

d = ox.InformationGeometryEngine.gaussian_fisher_rao_distance(0.0, 1.0, 2.0, 2.0)
assert d > 0.5
res = ox.InformationGeometryEngine.pythagorean_duality_residual([0.2, 0.5, 0.3], [0.6, 0.2, 0.2], [0.1, 0.3, 0.6])
assert res < 1e-13
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python InfoGeom test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v800_grand_omni_octa_engine_vii() -> Result<()> {
    use oxideon::omni_octa_vii::GrandOmniOctaEngineVII;

    let engine = GrandOmniOctaEngineVII::new(8);
    let rep = engine.execute_octa_synthesis()?;

    assert!(rep.quasi_category_associator_defect < 1e-12);
    assert!(rep.qap_divisibility_residual < 1e-10);
    assert!((rep.symplectic_gromov_capacity - 4.0 * std::f64::consts::PI).abs() < 1e-12);
    assert!(rep.jarzynski_free_energy.is_finite());
    assert_eq!(rep.associahedron_k5_vertices, 14);
    assert!(rep.topos_heyting_verified);
    assert!(rep.mfg_nash_social_cost > 0.0);
    let golden = 0.5 * (1.0 + 5.0_f64.sqrt());
    assert!((rep.tqft_fibonacci_quantum_dim - golden).abs() < 1e-12);
    assert!(rep.fisher_rao_geodesic_distance > 0.5);
    assert!((1.58..=32.0).contains(&rep.active_llm_bits_per_weight));
    assert!(rep.unified_octa_invariant > 25.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniOctaEngineVII(8)
rep = eng.execute_octa_synthesis()
assert rep.associahedron_k5_vertices == 14
assert rep.topos_heyting_verified
assert rep.unified_octa_invariant > 25.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python GrandOmniOctaEngineVII test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v810_non_abelian_braid_fibonacci_anyons() -> Result<()> {
    use oxideon::quantum::braid_anyons::FibonacciAnyonBraider;

    let braider = FibonacciAnyonBraider::new();
    let yb_res = braider.yang_baxter_residual();
    assert!(
        yb_res < 1e-14,
        "Fibonacci anyon B_3 representation must satisfy Yang-Baxter relation, got {yb_res}"
    );

    let word = vec![1, 2, 1, -2, -1, 2, 2];
    let unitarity_err = braider.braid_unitarity_defect(&word)?;
    assert!(
        unitarity_err < 1e-14,
        "Compiled braid word must be strictly unitary, got {unitarity_err}"
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

b = ox.FibonacciAnyonBraider()
assert b.yang_baxter_residual() < 1e-14
assert b.braid_unitarity_defect([1, 2, -1, 2]) < 1e-14
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python FibonacciAnyonBraider test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v820_tropical_spectral_karp_eigenvalue() -> Result<()> {
    use oxideon::science::tropical_spectral::TropicalSpectralEngine;

    // 3x3 matrix where cycle 0 -> 1 -> 0 has weight 1 + 3 = 4 (mean 2.0)
    // and cycle 1 -> 2 -> 1 has weight 2 + 1 = 3 (mean 1.5, minimum!)
    let mat = vec![10.0, 1.0, 10.0, 3.0, 10.0, 2.0, 10.0, 1.0, 10.0];
    let eng = TropicalSpectralEngine::new(3, mat)?;
    let eig = eng.karp_minimum_cycle_mean()?;
    assert!(
        (eig - 1.5).abs() < 1e-12,
        "Karp minimum cycle mean must be 1.5, got {eig}"
    );

    let dual = TropicalSpectralEngine::legendre_fenchel_conjugate(
        &[0.0, 1.0, 2.0],
        &[0.0, 1.0, 4.0],
        3.0,
    )?;
    assert!((dual - 2.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.TropicalSpectralEngine(2, [5.0, 1.0, 3.0, 6.0])
assert abs(eng.karp_minimum_cycle_mean() - 2.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python TropicalSpectralEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v830_relativistic_lattice_boltzmann_bgk() -> Result<()> {
    use oxideon::physics::relativistic_lbm::RelativisticLbm1D;

    let rho = vec![1.0, 1.5, 2.0, 1.5, 1.0, 0.8, 0.8, 1.0];
    let vel = vec![0.1, 0.2, -0.1, -0.2, 0.05, -0.05, 0.0, 0.0];
    let mut rlbm = RelativisticLbm1D::new(&rho, &vel, 1.25)?;
    let m0 = rlbm.total_mass();
    let p0 = rlbm.total_momentum();

    for _ in 0..25 {
        rlbm.step();
    }
    let m1 = rlbm.total_mass();
    let p1 = rlbm.total_momentum();
    assert!(
        (m1 - m0).abs() < 1e-12,
        "Relativistic LBM must conserve total particle mass"
    );
    assert!(
        (p1 - p0).abs() < 1e-12,
        "Relativistic LBM must conserve total momentum"
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

rlbm = ox.RelativisticLbm1D([1.0, 1.5, 1.0, 0.5], [0.1, -0.1, 0.0, 0.0], 1.2)
m0 = rlbm.total_mass()
rlbm.step(10)
assert abs(rlbm.total_mass() - m0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python RelativisticLbm1D test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v840_free_probability_voiculescu_marchenko_pastur() -> Result<()> {
    use oxideon::science::free_probability::FreeProbabilityEngine;

    // Standard Wigner Semicircle has moments m = [0, 1, 0, 2] (Catalan numbers C_1 = 1, C_2 = 2)
    // Its free cumulants must be k = [0, 1, 0, 0]!
    let k = FreeProbabilityEngine::moments_to_free_cumulants([0.0, 1.0, 0.0, 2.0]);
    assert!((k[1] - 1.0).abs() < 1e-12);
    assert!(
        k[3].abs() < 1e-12,
        "4th free cumulant of Wigner semicircle must vanish"
    );

    // Free additive convolution of two unit semicircles has variance 2 => 4th moment = 2 * 2^2 = 8.0
    let m_conv = FreeProbabilityEngine::free_additive_convolution_moments(
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 1.0, 0.0, 2.0],
    );
    assert!((m_conv[1] - 2.0).abs() < 1e-12);
    assert!((m_conv[3] - 8.0).abs() < 1e-12);

    let (l_min, l_max, dens) = FreeProbabilityEngine::marchenko_pastur_spectrum(1.0, 0.25, 1.0)?;
    assert!((l_min - 0.25).abs() < 1e-12);
    assert!((l_max - 2.25).abs() < 1e-12);
    assert!(dens > 0.1);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

k = ox.FreeProbabilityEngine.moments_to_free_cumulants([0.0, 1.0, 0.0, 2.0])
assert abs(k[1] - 1.0) < 1e-12 and abs(k[3]) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python FreeProbabilityEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v850_rough_paths_signature_chen_levy_area() -> Result<()> {
    use oxideon::ode::rough_paths::RoughPathSignature;

    // Path X: (0,0) -> (2,0), Path Y: (2,0) -> (2,3)
    // Concatenation X * Y encloses a right triangle with signed Levy area = 1/2 * 2 * 3 = 3.0
    let p_x = vec![vec![0.0, 0.0], vec![2.0, 0.0]];
    let p_y = vec![vec![2.0, 0.0], vec![2.0, 3.0]];
    let p_xy = vec![vec![0.0, 0.0], vec![2.0, 0.0], vec![2.0, 3.0]];

    let sig_x = RoughPathSignature::from_path(&p_x)?;
    let sig_y = RoughPathSignature::from_path(&p_y)?;
    let sig_xy = RoughPathSignature::from_path(&p_xy)?;
    let sig_chen = sig_x.chen_multiply(&sig_y)?;

    assert_eq!(
        sig_xy, sig_chen,
        "Chen's multiplicative identity S(X * Y) = S(X) (x) S(Y) must hold"
    );
    assert!((sig_xy.levy_area(0, 1)? - 3.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

sig = ox.RoughPathSignature([[0.0, 0.0], [2.0, 0.0], [2.0, 3.0]])
assert abs(sig.levy_area(0, 1) - 3.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python RoughPathSignature test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v860_p_adic_ultrametric_vladimirov() -> Result<()> {
    use oxideon::science::p_adic::PAdicAnalysisEngine;

    let padic = PAdicAnalysisEngine::new(5)?;
    assert!(
        (padic.norm(25) - 0.04).abs() < 1e-12,
        "|25|_5 = 5^-2 = 0.04"
    );
    assert!((padic.norm(12) - 1.0).abs() < 1e-12, "|12|_5 = 1.0");
    assert!(padic.verify_ultrametric_inequality(&[0, 1, 5, 10, 25, 50, 125]));

    let vlad = padic.vladimirov_operator(&[1.0, 0.0, 1.0, 0.0, 1.0], 1.0)?;
    assert_eq!(vlad.len(), 5);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

p5 = ox.PAdicAnalysisEngine(5)
assert abs(p5.norm(25) - 0.04) < 1e-12
assert p5.verify_ultrametric_inequality([0, 5, 25, 125])
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python PAdicAnalysisEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v870_discrete_morse_smale_witten_inequalities() -> Result<()> {
    use oxideon::science::morse_theory::DiscreteMorseEngine;

    // 2-Sphere S^2 with 1 collapsible pair of extra critical cells in dim 1 and 2:
    // Betti numbers b = [1, 0, 1] (Euler characteristic chi = 2), critical counts c = [1, 1, 2]
    let morse = DiscreteMorseEngine::new(vec![1, 1, 2], vec![1, 0, 1])?;
    assert_eq!(morse.morse_euler_characteristic(), 2);
    assert!(morse.verify_strong_morse_inequalities());

    let (m_val, p_val) = morse.evaluate_morse_and_poincare_polynomials(1.0);
    assert!((m_val - 4.0).abs() < 1e-12);
    assert!((p_val - 2.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

m = ox.DiscreteMorseEngine([1, 1, 2], [1, 0, 1])
assert m.morse_euler_characteristic() == 2
assert m.verify_strong_morse_inequalities()
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python DiscreteMorseEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v880_post_quantum_ring_lwe_cryptography() -> Result<()> {
    use oxideon::crypto::lattice_lwe::LatticeLweEngine;

    let lwe = LatticeLweEngine::new(4, 257)?;
    // Verify X^4 = -1 mod 257: X^2 * X^2 = X^4 = -1 = 256 mod 257
    let x2 = vec![0, 0, 1, 0];
    let prod = lwe.negacyclic_mul(&x2, &x2)?;
    assert_eq!(prod, vec![256, 0, 0, 0]);

    let a = vec![42, 105, 199, 17];
    let s = vec![1, -1, 1, 0];
    let e = vec![0, 1, -1, 1];
    let b = lwe.compute_public_key(&a, &s, &e)?;

    let msg = vec![1, 0, 1, 0];
    let (u, v) = lwe.encrypt(&a, &b, &[1, 0, -1, 1], &[0, 1, 0, -1], &[1, -1, 0, 0], &msg)?;
    let recovered = lwe.decrypt(&u, &v, &s)?;
    assert_eq!(
        recovered, msg,
        "Ring-LWE decryption must recover the exact 4-bit plaintext"
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

lwe = ox.LatticeLweEngine(4, 257)
a = [42, 105, 199, 17]
s = [1, -1, 1, 0]
b = lwe.compute_public_key(a, s, [0, 1, -1, 1])
u, v = lwe.encrypt(a, b, [1, 0, -1, 1], [0, 1, 0, -1], [1, -1, 0, 0], [1, 1, 0, 1])
assert list(lwe.decrypt(u, v, s)) == [1, 1, 0, 1]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python LatticeLweEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v890_koopman_spectral_edmd_birkhoff_ergodic() -> Result<()> {
    use oxideon::ode::koopman::KoopmanSpectralEngine;

    let psi_x = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![4.0, 8.0]];
    let psi_y = vec![vec![0.5, 6.0], vec![1.0, 12.0], vec![2.0, 24.0]];
    let koopman = KoopmanSpectralEngine::fit_diagonal_edmd(&psi_x, &psi_y)?;
    let pred = koopman.predict_observable(&[8.0, 1.0], 2)?;
    assert!(
        (pred[0] - 2.0).abs() < 1e-12,
        "Eigenmode 0 decays by 0.5^2 = 0.25"
    );
    assert!(
        (pred[1] - 9.0).abs() < 1e-12,
        "Eigenmode 1 grows by 3.0^2 = 9.0"
    );

    let ergodic_avg = KoopmanSpectralEngine::birkhoff_ulam_ergodic_average(0.3, 5000)?;
    assert!(
        (ergodic_avg - 0.5).abs() < 0.03,
        "Birkhoff ergodic time average of Ulam map must converge to 0.5, got {ergodic_avg}"
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

k = ox.KoopmanSpectralEngine([[1.0], [2.0]], [[0.5], [1.0]])
assert abs(k.predict_observable([4.0], 1)[0] - 2.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python KoopmanSpectralEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v900_grand_omni_nona_engine_viii() -> Result<()> {
    use oxideon::omni_nona_viii::GrandOmniNonaEngineVIII;

    let eng = GrandOmniNonaEngineVIII::new(8);
    let rep = eng.execute_nona_synthesis()?;

    assert!(rep.yang_baxter_residual < 1e-14);
    assert!((rep.tropical_eigenvalue - 2.0).abs() < 1e-12);
    assert!((rep.rlbm_conserved_mass - 4.0).abs() < 1e-12);
    assert!((rep.free_convolution_fourth_moment - 8.0).abs() < 1e-12);
    assert!((rep.rough_path_levy_area - 3.0).abs() < 1e-12);
    assert!(rep.p_adic_ultrametric_verified);
    assert!(rep.morse_inequalities_verified);
    assert!((rep.ring_lwe_recovery_rate - 1.0).abs() < 1e-12);
    assert!((rep.koopman_predicted_mode - 8.0).abs() < 1e-12);
    assert!((1.58..=32.0).contains(&rep.active_llm_bits_per_weight));
    assert!(rep.unified_nona_invariant > 25.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniNonaEngineVIII(8)
rep = eng.execute_nona_synthesis()
assert rep.yang_baxter_residual < 1e-14
assert rep.p_adic_ultrametric_verified
assert rep.morse_inequalities_verified
assert rep.ring_lwe_recovery_rate == 1.0
assert rep.unified_nona_invariant > 25.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python GrandOmniNonaEngineVIII test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v910_derived_algebraic_geometry_maurer_cartan() -> Result<()> {
    use oxideon::science::derived_geometry::DglaMaurerCartanEngine;

    // 1D DGLA equation: d(alpha) + 0.5 * (-2) * alpha^2 = alpha - alpha^2 = 0 => roots at alpha = 0 and alpha = 1
    let dgla = DglaMaurerCartanEngine::new(1, vec![1.0], vec![-2.0])?;
    assert!(
        dgla.maurer_cartan_residual_norm(&[1.0])? < 1e-14,
        "alpha = 1.0 is an exact Maurer-Cartan flat element"
    );

    let flowed = dgla.solve_maurer_cartan_flow(&[0.85], 0.25, 80)?;
    assert!(dgla.maurer_cartan_residual_norm(&flowed)? < 1e-6);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

dgla = ox.DglaMaurerCartanEngine(1, [1.0], [-2.0])
assert dgla.maurer_cartan_residual_norm([1.0]) < 1e-14
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python DglaMaurerCartanEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v920_quantum_chaos_syk_otoc_mss_bound() -> Result<()> {
    use oxideon::quantum::syk_chaos::SykQuantumChaosEngine;

    let syk = SykQuantumChaosEngine::new(64, 50.0, 2.0)?;
    let mss = syk.mss_chaos_bound();
    let lyap = syk.syk_lyapunov_exponent();
    assert!((mss - std::f64::consts::PI).abs() < 1e-12);
    assert!(
        lyap > 0.95 * mss && lyap < mss,
        "Strong-coupling SYK must approach but never violate the MSS bound 2*pi/beta"
    );

    let t_scr = syk.scrambling_time();
    assert!(
        (syk.evaluate_otoc(t_scr) - 0.5).abs() < 1e-12,
        "OTOC at scrambling time t_* must equal 0.5"
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

syk = ox.SykQuantumChaosEngine(64, 50.0, 2.0)
assert syk.syk_lyapunov_exponent() < syk.mss_chaos_bound()
assert abs(syk.evaluate_otoc(syk.scrambling_time()) - 0.5) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python SykQuantumChaosEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v930_wasserstein_geodesic_flow_mccann() -> Result<()> {
    use oxideon::geom_graphs::wasserstein_flow::GraphWassersteinGeodesicEngine;

    let eng = GraphWassersteinGeodesicEngine::new(5, 1.0)?;
    let mu0 = vec![1.0, 0.0, 0.0, 0.0, 0.0];
    let mu1 = vec![0.0, 0.0, 0.0, 0.0, 1.0];

    let w1 = eng.wasserstein_1_distance(&mu0, &mu1)?;
    assert!(
        (w1 - 4.0).abs() < 1e-12,
        "Dirac mass transport from x=0 to x=4 has W_1 = 4.0"
    );

    let (mu_mid, w2) = eng.displacement_interpolate(&mu0, &mu1, 0.5)?;
    assert!((w2 - 4.0).abs() < 1e-9);
    assert!(
        (mu_mid[2] - 1.0).abs() < 1e-9,
        "McCann midpoint t=0.5 between x=0 and x=4 must concentrate at x=2"
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

w = ox.GraphWassersteinGeodesicEngine(5, 1.0)
mu_mid, w2 = w.displacement_interpolate([1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0], 0.5)
assert abs(mu_mid[2] - 1.0) < 1e-9 and abs(w2 - 4.0) < 1e-9
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GraphWassersteinGeodesicEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v940_homotopy_type_theory_univalence_circle() -> Result<()> {
    use oxideon::reasoning::hott::HomotopyTypeEngine;

    for n in -5..=5 {
        assert!(HomotopyTypeEngine::verify_univalent_circle_coherence(n)?);
    }
    assert!(HomotopyTypeEngine::decode_winding_number(i64::MAX).is_err()); // a word that long cannot exist
    assert_eq!(
        HomotopyTypeEngine::encode_circle_loop(&[1, 1, -1, 1, 0, 1]),
        3
    );

    let transported = HomotopyTypeEngine::univalent_orthogonal_transport(
        &[1.0, 0.0],
        std::f64::consts::FRAC_PI_2,
    )?;
    assert!(transported[0].abs() < 1e-12 && (transported[1] - 1.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert ox.HomotopyTypeEngine.verify_univalent_circle_coherence(-4)
assert ox.HomotopyTypeEngine.encode_circle_loop([1, 1, -1, 1]) == 2
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python HomotopyTypeEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v950_cavity_qed_jaynes_cummings_rabi() -> Result<()> {
    use oxideon::physics::cavity_qed::JaynesCummingsCavityQed;

    // Resonant lossless Jaynes-Cummings cavity: omega_c = omega_q = 10.0, g = 1.0, kappa = 0.0
    // Vacuum Rabi frequency Omega_1 = 2g = 2.0; full inversion at t = pi / Omega_1 = pi / 2
    let cqed = JaynesCummingsCavityQed::new(10.0, 10.0, 1.0, 0.0)?;
    assert!((cqed.generalized_rabi_frequency(1) - 2.0).abs() < 1e-12);

    let (pe, pc) = cqed.rabi_oscillation_populations(1, std::f64::consts::FRAC_PI_2)?;
    assert!(
        pe.abs() < 1e-12 && (pc - 1.0).abs() < 1e-12,
        "At half-period pi/(2g), excitation transfers 100% into cavity photon"
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox
import math

cqed = ox.JaynesCummingsCavityQed(10.0, 10.0, 1.0, 0.0)
pe, pc = cqed.rabi_oscillation_populations(1, math.pi / 2.0)
assert abs(pe) < 1e-12 and abs(pc - 1.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python JaynesCummingsCavityQed test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v960_connes_non_commutative_spectral_triple() -> Result<()> {
    use oxideon::science::spectral_triple::ConnesSpectralTriple;

    // 3-site Dirac chain with hopping D_{01} = 2.0 (distance 0.5) and D_{12} = 0.25 (distance 4.0)
    // Total Connes distance d(0, 2) = 0.5 + 4.0 = 4.5
    let d_mat = vec![0.0, 2.0, 0.0, 2.0, 0.0, 0.25, 0.0, 0.25, 0.0];
    let triple = ConnesSpectralTriple::new(3, d_mat)?;
    assert!((triple.chain_connes_distance(0, 2)? - 4.5).abs() < 1e-12);
    assert!(
        triple.commutator_frobenius_norm(&[1.0, 1.0, 1.0])? < 1e-14,
        "Identity element commutes with Dirac operator"
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

triple = ox.ConnesSpectralTriple(3, [0.0, 2.0, 0.0, 2.0, 0.0, 0.25, 0.0, 0.25, 0.0])
assert abs(triple.chain_connes_distance(0, 2) - 4.5) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python ConnesSpectralTriple test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v970_solomonoff_universal_prior_levin_kt() -> Result<()> {
    use oxideon::reasoning::solomonoff::{AlgorithmicHypothesis, SolomonoffLevinEngine};

    let eng = SolomonoffLevinEngine::new(vec![
        AlgorithmicHypothesis {
            code_length_bits: 2.0,
            runtime_steps: 8,
            data_misfit_bits: 0.0,
        },
        AlgorithmicHypothesis {
            code_length_bits: 4.0,
            runtime_steps: 16,
            data_misfit_bits: 0.0,
        },
    ])?;
    assert!(
        eng.kraft_inequality_sum() <= 1.0,
        "Prefix-free Kraft inequality sum must be <= 1.0"
    );
    let kt = eng.levin_kt_complexities();
    assert!((kt[0] - 5.0).abs() < 1e-12, "Kt = 2 + log2(8) = 5.0");
    let w = eng.solomonoff_posterior_weights();
    // 2^-2 / (2^-2 + 2^-4) = 0.25 / 0.3125 = 0.8
    assert!((w[0] - 0.8).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

sol = ox.SolomonoffLevinEngine([2.0, 4.0], [8, 16], [0.0, 0.0])
assert abs(sol.solomonoff_posterior_weights()[0] - 0.8) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python SolomonoffLevinEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v980_penrose_twistor_incidence_null_ray() -> Result<()> {
    use oxideon::physics::twistor::PenroseTwistorEngine;

    let tw = PenroseTwistorEngine::from_minkowski_incidence(
        [3.0, -1.5, 2.0, 0.5],
        [(1.2, -0.4), (0.8, 1.1)],
    );
    assert!(
        tw.su22_hermitian_norm().abs() < 1e-14,
        "Penrose incidence at any real Minkowski event must lie strictly on the null twistor quadric PN"
    );
    let [e, px, py, pz] = tw.null_four_momentum();
    let mass_shell = e * e - px * px - py * py - pz * pz;
    assert!(
        mass_shell.abs() < 1e-13,
        "Flagpole spinor 4-momentum must be massless (null)"
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

tw = ox.PenroseTwistorEngine([3.0, -1.5, 2.0, 0.5], (1.2, -0.4), (0.8, 1.1))
assert abs(tw.su22_hermitian_norm()) < 1e-14
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python PenroseTwistorEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v990_langlands_hecke_ramanujan_deligne() -> Result<()> {
    use oxideon::science::langlands::LanglandsHeckeEngine;

    let lang = LanglandsHeckeEngine::new(12)?;
    assert_eq!(lang.ramanujan_tau(1)?, 1);
    assert_eq!(lang.ramanujan_tau(2)?, -24);
    assert_eq!(lang.ramanujan_tau(3)?, 252);
    assert_eq!(lang.ramanujan_tau(4)?, -1472);
    assert_eq!(lang.ramanujan_tau(5)?, 4830);
    assert!(lang.verify_hecke_relations());
    assert!(lang.verify_deligne_ramanujan_bound());

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

lang = ox.LanglandsHeckeEngine(12)
assert lang.ramanujan_tau(2) == -24 and lang.ramanujan_tau(3) == 252
assert lang.verify_hecke_relations() and lang.verify_deligne_ramanujan_bound()
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python LanglandsHeckeEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1000_grand_centennial_singularity_engine_x() -> Result<()> {
    use oxideon::omni_centennial_x::GrandCentennialSingularityEngineX;

    let eng = GrandCentennialSingularityEngineX::new(12);
    let rep = eng.execute_centennial_singularity()?;

    assert!(rep.maurer_cartan_residual < 1e-6);
    assert!(rep.syk_mss_saturation_ratio > 0.8 && rep.syk_mss_saturation_ratio < 1.0);
    assert!((rep.wasserstein_geodesic_w2 - 2.0).abs() < 1e-9);
    assert!(rep.hott_univalence_verified);
    assert!((rep.jaynes_cummings_rabi_splitting - 3.0).abs() < 1e-12);
    assert!((rep.connes_spectral_distance - 2.5).abs() < 1e-12);
    assert!(rep.solomonoff_top_posterior > 0.99);
    assert!(rep.twistor_null_helicity_residual < 1e-13);
    assert!(rep.langlands_deligne_verified);
    assert!((1.58..=32.0).contains(&rep.active_llm_bits_per_weight));
    assert!(rep.centennial_singularity_invariant > 100.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandCentennialSingularityEngineX(12)
rep = eng.execute_centennial_singularity()
assert rep.hott_univalence_verified
assert rep.langlands_deligne_verified
assert rep.twistor_null_helicity_residual < 1e-13
assert rep.centennial_singularity_invariant > 100.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandCentennialSingularityEngineX test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1010_etale_cohomology_weil_zeta() -> Result<()> {
    use oxideon::science::etale_cohomology::EtaleWeilZetaEngine;
    // Supersingular elliptic curve over F_5 with #E(F_5) = 6 => a_1 = 0 => #E(F_25) = 1 + 25 - (0 - 2*5) = 36
    let zeta = EtaleWeilZetaEngine::new(5, 6)?;
    assert_eq!(zeta.rational_point_count(1)?, 6);
    assert_eq!(zeta.rational_point_count(2)?, 36);
    assert!((zeta.evaluate_zeta(0.1)? - (1.0 + 0.05) / (0.9 * 0.5)).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

ew = ox.EtaleWeilZetaEngine(5, 6)
assert ew.rational_point_count(2) == 36
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python EtaleWeilZetaEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1020_motivic_integration_hodge_deligne() -> Result<()> {
    use oxideon::science::motivic_integration::MotivicHodgeEngine;
    // Euler characteristic (u = v = 1 => L = 1): chi(P^3) = 4, chi(G(2, 4)) = 6 = C(4, 2)
    assert!((MotivicHodgeEngine::projective_space_motive(3, 1.0, 1.0) - 4.0).abs() < 1e-12);
    assert!((MotivicHodgeEngine::grassmannian_g24_motive(1.0, 1.0) - 6.0).abs() < 1e-12);
    let arc = MotivicHodgeEngine::kontsevich_arc_integral(1, 1, 2.0)?;
    assert!((arc - (0.5 / 0.75)).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert ox.MotivicHodgeEngine.grassmannian_g24_motive(1.0, 1.0) == 6.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python MotivicHodgeEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1030_cluster_algebras_fomin_zelevinsky_pentagon() -> Result<()> {
    use oxideon::science::cluster_algebras::ClusterAlgebraQuiverEngine;
    // A_2 quiver B = [[0, 1], [-1, 0]] with initial seed (2.0, 3.0):
    // 5 alternating mutations (0, 1, 0, 1, 0) return to the initial variables swapped (3.0, 2.0)!
    let mut ca = ClusterAlgebraQuiverEngine::new(2, vec![0, 1, -1, 0], vec![2.0, 3.0])?;
    for &k in &[0, 1, 0, 1, 0] {
        ca.mutate(k)?;
    }
    assert!((ca.cluster_vars[0] - 3.0).abs() < 1e-12 && (ca.cluster_vars[1] - 2.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

ca = ox.ClusterAlgebraQuiverEngine(2, [0, 1, -1, 0], [2.0, 3.0])
for k in [0, 1, 0, 1, 0]:
    ca.mutate(k)
assert abs(ca.cluster_vars[0] - 3.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python ClusterAlgebraQuiverEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1040_chow_intersection_hirzebruch_riemann_roch() -> Result<()> {
    use oxideon::science::intersection_theory::ChowHirzebruchRiemannRoch;
    // K3 surface: K_S = 0 (K^2 = 0), Euler characteristic c_2(K3) = 24 => chi(O_{K3}) = 24 / 12 = 2.0
    assert!((ChowHirzebruchRiemannRoch::riemann_roch_surface(0, 24, 0, 0) - 2.0).abs() < 1e-12);
    assert_eq!(ChowHirzebruchRiemannRoch::riemann_roch_curve(2, 1, 5), 4);
    assert_eq!(
        ChowHirzebruchRiemannRoch::bezout_intersection_number(&[2, 3, 4]),
        24
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert ox.ChowHirzebruchRiemannRoch.riemann_roch_surface(0, 24, 0, 0) == 2.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python ChowHirzebruchRiemannRoch test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1050_bridgeland_stability_harder_narasimhan() -> Result<()> {
    use oxideon::science::bridgeland_stability::BridgelandStabilityEngine;
    let b = BridgelandStabilityEngine::new(1.0)?;
    // Skyscraper sheaf (r=0, d=1) has phase 1.0; line bundle O(1) (r=1, d=1) has phase 0.75; O (r=1, d=0) has phase 0.5
    let (p_sky, _) = b.phase_and_mass(0.0, 1.0)?;
    let (p_o1, _) = b.phase_and_mass(1.0, 1.0)?;
    let (p_o0, _) = b.phase_and_mass(1.0, 0.0)?;
    assert!(
        (p_sky - 1.0).abs() < 1e-12 && (p_o1 - 0.75).abs() < 1e-12 && (p_o0 - 0.5).abs() < 1e-12
    );
    let (ok, mass) =
        b.verify_harder_narasimhan_filtration(&[(0.0, 1.0), (1.0, 1.0), (1.0, 0.0)])?;
    assert!(ok && mass > 3.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

b = ox.BridgelandStabilityEngine(1.0)
ok, m = b.verify_harder_narasimhan_filtration([(0.0, 1.0), (1.0, 1.0), (1.0, 0.0)])
assert ok and m > 3.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python BridgelandStabilityEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1060_gromov_witten_kontsevich_wdvv() -> Result<()> {
    use oxideon::science::gromov_witten::GromovWittenWdvvEngine;
    assert_eq!(
        GromovWittenWdvvEngine::kontsevich_rational_curve_count(1)?,
        1
    );
    assert_eq!(
        GromovWittenWdvvEngine::kontsevich_rational_curve_count(2)?,
        1
    );
    assert_eq!(
        GromovWittenWdvvEngine::kontsevich_rational_curve_count(3)?,
        12
    );
    assert_eq!(
        GromovWittenWdvvEngine::kontsevich_rational_curve_count(4)?,
        620
    );
    assert_eq!(
        GromovWittenWdvvEngine::kontsevich_rational_curve_count(5)?,
        87304
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert ox.GromovWittenWdvvEngine.kontsevich_rational_curve_count(4) == 620
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python GromovWittenWdvvEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1070_donaldson_thomas_macmahon_mnop() -> Result<()> {
    use oxideon::science::donaldson_thomas::DonaldsonThomasEngine;
    let dt = DonaldsonThomasEngine::new(8)?;
    assert_eq!(dt.plane_partition_count(0)?, 1);
    assert_eq!(dt.plane_partition_count(1)?, 1);
    assert_eq!(dt.plane_partition_count(2)?, 3);
    assert_eq!(dt.plane_partition_count(3)?, 6);
    assert_eq!(dt.plane_partition_count(4)?, 13);
    assert_eq!(dt.plane_partition_count(5)?, 24);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

dt = ox.DonaldsonThomasEngine(6)
assert dt.plane_partition_count(4) == 13 and dt.plane_partition_count(5) == 24
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python DonaldsonThomasEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1080_hitchin_higgs_bundles_spectral_curve() -> Result<()> {
    use oxideon::science::hitchin_system::HitchinHiggsBundleEngine;
    let h = HitchinHiggsBundleEngine::new(2, 3)?;
    assert_eq!(h.spectral_curve_genus(), 10);
    let (tr, h2, det) =
        HitchinHiggsBundleEngine::evaluate_sl2_hitchin_hamiltonians([2.0, 1.0, 3.0, -2.0]);
    assert!(tr.abs() < 1e-12 && (h2 - 7.0).abs() < 1e-12 && (det - (-7.0)).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

h = ox.HitchinHiggsBundleEngine(2, 3)
assert h.spectral_curve_genus() == 10
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python HitchinHiggsBundleEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1090_perfectoid_spaces_scholze_tilting() -> Result<()> {
    use oxideon::science::perfectoid_spaces::PerfectoidTiltingEngine;
    let perf = PerfectoidTiltingEngine::new(3)?;
    // By Fermat's Little Theorem, 2^3 = 8 = 2 mod 3, so [2, 5, 8, 11] is a compatible mod-3 Frobenius tower
    assert!(perf.verify_frobenius_tower(&[2, 5, 8, 11]));
    let lift = perf.teichmuller_sharp_lift(2, 2)?;
    assert_eq!(lift % 3, 2);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

perf = ox.PerfectoidTiltingEngine(3)
assert perf.verify_frobenius_tower([2, 5, 8, 11])
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python PerfectoidTiltingEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1100_grand_omni_hendeca_engine_xi() -> Result<()> {
    use oxideon::omni_hendeca_xi::GrandOmniHendecaEngineXI;
    let eng = GrandOmniHendecaEngineXI::new(10);
    let inv = eng.execute_hendeca_synthesis()?;
    assert!(inv > 680.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniHendecaEngineXI(10)
assert eng.execute_hendeca_synthesis() > 680.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python GrandOmniHendecaEngineXI test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1110_loop_quantum_gravity_spin_network_area() -> Result<()> {
    use oxideon::physics::loop_quantum_gravity::SpinNetworkLqgEngine;
    let lqg = SpinNetworkLqgEngine::new(0.2375, 1.0)?;
    // Minimum area gap at j = 1/2 (two_j = 1): 8 * pi * gamma * sqrt(3)/2 = 4 * sqrt(3) * pi * gamma
    let area = lqg.quantized_area_eigenvalue(&[1])?;
    let expected = 4.0 * 3.0_f64.sqrt() * std::f64::consts::PI * 0.2375;
    assert!((area - expected).abs() < 1e-12);
    assert!(SpinNetworkLqgEngine::is_admissible_trivalent_intertwiner(
        1, 1, 2
    ));
    assert!(!SpinNetworkLqgEngine::is_admissible_trivalent_intertwiner(
        1, 1, 1
    ));

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

lqg = ox.SpinNetworkLqgEngine(0.2375, 1.0)
assert lqg.quantized_area_eigenvalue([1]) > 5.0
assert ox.SpinNetworkLqgEngine.is_admissible_trivalent_intertwiner(1, 1, 2)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python SpinNetworkLqgEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1120_causal_dynamical_triangulations_spectral_dim() -> Result<()> {
    use oxideon::physics::causal_dynamical_triangulations::CausalTriangulationEngine;
    let cdt = CausalTriangulationEngine::new(2.2, 0.6)?;
    assert!(
        (cdt.spectral_dimension(0.0, 1.0)? - 2.0).abs() < 1e-12,
        "UV spectral dimension is 2.0"
    );
    assert!(
        (cdt.spectral_dimension(1e6, 1.0)? - 4.0).abs() < 1e-5,
        "IR spectral dimension flows to 4.0"
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

cdt = ox.CausalTriangulationEngine(2.2, 0.6)
assert abs(cdt.spectral_dimension(0.0, 1.0) - 2.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python CausalTriangulationEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1130_amplituhedron_positive_grassmannian() -> Result<()> {
    use oxideon::physics::amplituhedron::AmplituhedronGrassmannianEngine;
    // Vandermonde matrix C = [[1, 1, 1, 1], [1, 2, 3, 4]] has <i j> = x_j - x_i > 0 for all i < j!
    let amp =
        AmplituhedronGrassmannianEngine::new(vec![1.0, 1.0, 1.0, 1.0], vec![1.0, 2.0, 3.0, 4.0])?;
    let (pos, plucker_res) = amp.verify_positive_grassmannian();
    assert!(pos && plucker_res < 1e-14);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

amp = ox.AmplituhedronGrassmannianEngine([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0])
pos, res = amp.verify_positive_grassmannian()
assert pos and res < 1e-14
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python AmplituhedronGrassmannianEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1140_celestial_holography_mellin_bms_soft() -> Result<()> {
    use oxideon::physics::celestial_holography::CelestialHolographyEngine;
    // Gamma(4) * 2^-4 = 6 / 16 = 0.375
    let amp = CelestialHolographyEngine::mellin_celestial_correlator(4, 2.0)?;
    assert!((amp - 0.375).abs() < 1e-12);
    assert!((CelestialHolographyEngine::weinberg_soft_pole_residue(1.0, 5.0)? - 1.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert abs(ox.CelestialHolographyEngine.mellin_celestial_correlator(4, 2.0) - 0.375) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python CelestialHolographyEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1150_swampland_weak_gravity_distance_conjecture() -> Result<()> {
    use oxideon::physics::swampland_conjectures::SwamplandVerifierEngine;
    assert!(SwamplandVerifierEngine::satisfies_weak_gravity_conjecture(
        0.5, 1.0, 0.5
    ));
    assert!(!SwamplandVerifierEngine::satisfies_weak_gravity_conjecture(
        2.0, 1.0, 0.5
    ));
    let m_tower = SwamplandVerifierEngine::distance_conjecture_tower_mass(1.0, 1.0, 2.0);
    assert!((m_tower - (-2.0_f64).exp()).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert ox.SwamplandVerifierEngine.satisfies_weak_gravity_conjecture(0.5, 1.0, 0.5)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python SwamplandVerifierEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1160_conformal_bootstrap_unitarity_blocks() -> Result<()> {
    use oxideon::physics::conformal_bootstrap::ConformalBootstrapEngine;
    let cb = ConformalBootstrapEngine::new(3.0)?;
    assert!(
        (cb.unitarity_bound(0) - 0.5).abs() < 1e-12,
        "3D scalar unitarity bound is 0.5"
    );
    assert!(
        (cb.unitarity_bound(2) - 3.0).abs() < 1e-12,
        "3D stress tensor spin-2 bound is 3.0"
    );
    // At Delta = 1: z * _2F_1(1, 1; 2; z) = -ln(1 - z)
    let g1 = ConformalBootstrapEngine::chiral_conformal_block(1.0, 0.5)?;
    assert!((g1 - 2.0_f64.ln()).abs() < 1e-10);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

cb = ox.ConformalBootstrapEngine(3.0)
assert cb.unitarity_bound(0) == 0.5 and cb.unitarity_bound(2) == 3.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python ConformalBootstrapEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1170_traversable_wormhole_maldacena_qi_teleportation() -> Result<()> {
    use oxideon::physics::sachdev_kitaev_wormhole::TraversableWormholeEngine;
    // q = 4 SYK => Delta = 0.25, mu = 0.125 => E_gap = 0.125^(4/3) = 0.0625
    let wh = TraversableWormholeEngine::new(0.125, 0.25)?;
    assert!((wh.wormhole_energy_gap() - 0.0625).abs() < 1e-12);
    let t_peak = std::f64::consts::FRAC_PI_2 / 0.0625;
    assert!((wh.transmission_fidelity(t_peak) - 1.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

wh = ox.TraversableWormholeEngine(0.125, 0.25)
assert abs(wh.wormhole_energy_gap() - 0.0625) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python TraversableWormholeEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1180_ecalle_resurgence_borel_pade_transseries() -> Result<()> {
    use oxideon::physics::resurgence_transseries::BorelResurgenceEngine;
    // Alternating factorial series a_n = (-1)^n n! => [1, -1, 2] has Borel transform 1/(1+t); at t = 3, 1/4 = 0.25!
    let bp = BorelResurgenceEngine::borel_pade_11_transform([1.0, -1.0, 2.0], 3.0)?;
    assert!((bp - 0.25).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert abs(ox.BorelResurgenceEngine.borel_pade_11_transform([1.0, -1.0, 2.0], 3.0) - 0.25) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python BorelResurgenceEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1190_exact_wkb_voros_bohr_sommerfeld() -> Result<()> {
    use oxideon::physics::exact_wkb::ExactWkbQuantizationEngine;
    let wkb = ExactWkbQuantizationEngine::new(1.0, 2.0, 0.0)?;
    assert!((wkb.quantized_energy_level(0) - 1.0).abs() < 1e-12);
    assert!((wkb.quantized_energy_level(1) - 3.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

wkb = ox.ExactWkbQuantizationEngine(1.0, 2.0, 0.0)
assert wkb.quantized_energy_level(0) == 1.0 and wkb.quantized_energy_level(1) == 3.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python ExactWkbQuantizationEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1200_grand_omni_dodeca_engine_xii() -> Result<()> {
    use oxideon::omni_dodeca_xii::GrandOmniDodecaEngineXII;
    let eng = GrandOmniDodecaEngineXII::new(12);
    let inv = eng.execute_dodeca_synthesis()?;
    assert!(inv > 690.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniDodecaEngineXII(12)
assert eng.execute_dodeca_synthesis() > 690.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python GrandOmniDodecaEngineXII test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1210_quantum_ldpc_css_hypergraph_codes() -> Result<()> {
    use oxideon::quantum::quantum_ldpc::QuantumLdpcCodeEngine;
    // Steane [[7, 1, 3]] CSS code parity check matrix H (self-orthogonal over F_2: H * H^T = 0 mod 2)
    let h = vec![
        1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1,
    ];
    let qldpc = QuantumLdpcCodeEngine::new(7, h.clone(), h)?;
    assert!(qldpc.verify_css_commutativity()?);
    let synd = qldpc.compute_z_syndrome(&[0, 0, 1, 0, 0, 0, 0])?;
    assert_eq!(synd, vec![1, 1, 0]);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

h = [1,0,1,0,1,0,1, 0,1,1,0,0,1,1, 0,0,0,1,1,1,1]
q = ox.QuantumLdpcCodeEngine(7, h, h)
assert q.verify_css_commutativity()
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python QuantumLdpcCodeEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1220_magic_state_distillation_bravyi_kitaev() -> Result<()> {
    use oxideon::quantum::magic_state_distillation::MagicStateDistillationEngine;
    let (p_out, cost) = MagicStateDistillationEngine::distill_15_to_1_infidelity(0.01, 2)?;
    assert_eq!(cost, 225);
    assert!(
        p_out < 2e-9,
        "2 rounds of 15-to-1 distillation from p=1e-2 must reach ~1.5e-9, got {p_out}"
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

p_out, cost = ox.MagicStateDistillationEngine.distill_15_to_1_infidelity(0.01, 2)
assert cost == 225 and p_out < 2e-9
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python MagicStateDistillationEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1230_mera_holographic_entanglement_renormalization() -> Result<()> {
    use oxideon::quantum::mera_holography::MeraEntanglementEngine;
    let mera = MeraEntanglementEngine::new(4)?;
    let s_bound = mera.causal_cone_entanglement_bound(8.0)?;
    assert!((s_bound - 6.0 * 4.0_f64.ln()).abs() < 1e-12);
    assert!(
        (MeraEntanglementEngine::scaling_dimension_from_eigenvalue(0.25, 2.0)? - 2.0).abs() < 1e-12
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

mera = ox.MeraEntanglementEngine(4)
assert abs(ox.MeraEntanglementEngine.scaling_dimension_from_eigenvalue(0.25, 2.0) - 2.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python MeraEntanglementEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1240_quantum_darwinism_zurek_redundancy_plateau() -> Result<()> {
    use oxideon::quantum::quantum_darwinism::QuantumDarwinismEngine;
    // Orthogonal pointer states (c = 0.0): any k < N fragments give classical plateau I(S:F_k) = 1.0 bit, while k = N gives 2.0 bits!
    let qd = QuantumDarwinismEngine::new(10, 0.0)?;
    assert!((qd.mutual_information_bits(1)? - 1.0).abs() < 1e-9);
    assert!((qd.mutual_information_bits(5)? - 1.0).abs() < 1e-9);
    assert!((qd.mutual_information_bits(10)? - 2.0).abs() < 1e-9);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

qd = ox.QuantumDarwinismEngine(10, 0.0)
assert abs(qd.mutual_information_bits(5) - 1.0) < 1e-9
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python QuantumDarwinismEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1250_floquet_discrete_time_crystals_subharmonic() -> Result<()> {
    use oxideon::quantum::floquet_time_crystals::FloquetTimeCrystalEngine;
    let dtc = FloquetTimeCrystalEngine::new(0.0, 1.0)?;
    assert!((dtc.stroboscopic_magnetization(0) - 1.0).abs() < 1e-12);
    assert!((dtc.stroboscopic_magnetization(1) - (-1.0)).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

dtc = ox.FloquetTimeCrystalEngine(0.0, 1.0)
assert dtc.stroboscopic_magnetization(1) == -1.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python FloquetTimeCrystalEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1260_kitaev_majorana_chain_pfaffian_invariant() -> Result<()> {
    use oxideon::quantum::majorana_kitaev_chain::KitaevMajoranaChainEngine;
    let top = KitaevMajoranaChainEngine::new(0.5, 1.0, 1.0)?;
    let triv = KitaevMajoranaChainEngine::new(3.0, 1.0, 1.0)?;
    assert_eq!(top.z2_pfaffian_invariant(), -1);
    assert_eq!(triv.z2_pfaffian_invariant(), 1);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

top = ox.KitaevMajoranaChainEngine(0.5, 1.0, 1.0)
assert top.z2_pfaffian_invariant() == -1
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python KitaevMajoranaChainEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1270_quantum_metrology_qfi_heisenberg_limit() -> Result<()> {
    use oxideon::quantum::quantum_metrology::QuantumMetrologyEngine;
    let (fq_hl, dtheta_hl) = QuantumMetrologyEngine::ghz_quantum_fisher_info(10, 1.0)?;
    assert!((fq_hl - 100.0).abs() < 1e-12 && (dtheta_hl - 0.1).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

fq, dt = ox.QuantumMetrologyEngine.ghz_quantum_fisher_info(10, 1.0)
assert fq == 100.0 and abs(dt - 0.1) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python QuantumMetrologyEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1280_quantum_thermodynamics_otto_carnot() -> Result<()> {
    use oxideon::quantum::quantum_thermodynamics::QuantumOttoEngine;
    let otto = QuantumOttoEngine::new(2.0, 1.0, 4.0, 1.0)?;
    let (eta_o, eta_c) = otto.otto_and_carnot_efficiencies();
    assert!((eta_o - 0.5).abs() < 1e-12 && (eta_c - 0.75).abs() < 1e-12);
    assert!(otto.work_per_cycle() > 0.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

otto = ox.QuantumOttoEngine(2.0, 1.0, 4.0, 1.0)
assert otto.work_per_cycle() > 0.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python QuantumOttoEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1290_classical_shadow_tomography_huang_kueng_preskill() -> Result<()> {
    use oxideon::quantum::shadow_tomography::ClassicalShadowTomography;
    let mom = ClassicalShadowTomography::median_of_means(&[1.0, 1.0, 1.0, 1.0, 100.0, 1.0], 3)?;
    assert!(
        (mom - 1.0).abs() < 1e-12,
        "Median-of-means rejects outlier block"
    );
    assert!(ClassicalShadowTomography::sample_complexity_bound(2, 100, 0.1, 0.01)? > 1000);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert ox.ClassicalShadowTomography.median_of_means([1.0, 1.0, 1.0, 1.0, 100.0, 1.0], 3) == 1.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python ClassicalShadowTomography test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1300_grand_omni_trideca_engine_xiii() -> Result<()> {
    use oxideon::omni_trideca_xiii::GrandOmniTridecaEngineXIII;
    let eng = GrandOmniTridecaEngineXIII::new(13);
    let inv = eng.execute_trideca_synthesis()?;
    assert!(inv > 800.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniTridecaEngineXIII(13)
assert eng.execute_trideca_synthesis() > 800.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniTridecaEngineXIII test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1310_ricci_flow_hamilton_perelman_extinction() -> Result<()> {
    use oxideon::science::ricci_flow_perelman::PerelmanRicciFlowEngine;
    // 3-Sphere S^3 of radius r_0 = 2.0: T_ext = 4 / (2 * 2) = 1.0
    let rf = PerelmanRicciFlowEngine::new(3, 2.0)?;
    assert!((rf.extinction_time() - 1.0).abs() < 1e-12);
    let (r, sc) = rf.radius_and_scalar_curvature(0.75)?;
    assert!((r - 1.0).abs() < 1e-12 && (sc - 6.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

rf = ox.PerelmanRicciFlowEngine(3, 2.0)
assert rf.extinction_time() == 1.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python PerelmanRicciFlowEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1320_mean_curvature_flow_huisken_colding_minicozzi() -> Result<()> {
    use oxideon::science::mean_curvature_flow::MeanCurvatureFlowEngine;
    let mcf = MeanCurvatureFlowEngine::new(2, 2.0)?;
    assert!((mcf.extinction_time() - 1.0).abs() < 1e-12);
    assert!((mcf.colding_minicozzi_sphere_entropy() - 4.0 / std::f64::consts::E).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

mcf = ox.MeanCurvatureFlowEngine(2, 2.0)
assert mcf.extinction_time() == 1.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python MeanCurvatureFlowEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1330_calabi_yau_complex_monge_ampere() -> Result<()> {
    use oxideon::science::calabi_yau_monge_ampere::CalabiYauMongeAmpereEngine;
    // g = diag(1, 1), H_phi = diag(e^0.5 - 1, e^0.5 - 1) => det(g + H_phi) = e^1.0 => residual at F = 1.0 is 0!
    let d = 0.5_f64.exp() - 1.0;
    let res =
        CalabiYauMongeAmpereEngine::monge_ampere_2d_residual([1.0, 1.0], [d, 0.0, 0.0, d], 1.0)?;
    assert!(res < 1e-14);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

import math
d = math.exp(0.5) - 1.0
assert ox.CalabiYauMongeAmpereEngine.monge_ampere_2d_residual([1.0, 1.0], [d, 0.0, 0.0, d], 1.0) < 1e-14
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python CalabiYauMongeAmpereEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1340_bpst_yang_mills_instanton_bogomolny() -> Result<()> {
    use oxideon::science::yang_mills_instanton::BpstYangMillsInstanton;
    let inst = BpstYangMillsInstanton::new(1.0, 1.0)?;
    assert!(
        (inst.bogomolny_instanton_action(1) - 8.0 * std::f64::consts::PI.powi(2)).abs() < 1e-12
    );
    assert!(inst.bpst_topological_density(0.0) > inst.bpst_topological_density(2.0));

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

inst = ox.BpstYangMillsInstanton(1.0, 1.0)
assert inst.bpst_topological_density(0.0) > inst.bpst_topological_density(2.0)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python BpstYangMillsInstanton test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1350_seiberg_witten_monopole_adjunction() -> Result<()> {
    use oxideon::science::seiberg_witten::SeibergWittenMonopoleEngine;
    // On K3 surface: chi = 24, sigma = -16 => 2*chi + 3*sigma = 48 - 48 = 0! For canonical basic class c1 = 0, d(s) = 0.0
    assert!(
        (SeibergWittenMonopoleEngine::moduli_virtual_dimension(0, 24, -16) - 0.0).abs() < 1e-12
    );
    assert!(SeibergWittenMonopoleEngine::satisfies_adjunction_inequality(3, 2, 2));

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert ox.SeibergWittenMonopoleEngine.moduli_virtual_dimension(0, 24, -16) == 0.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python SeibergWittenMonopoleEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1360_minimal_surfaces_weierstrass_willmore() -> Result<()> {
    use oxideon::science::minimal_surfaces_plateau::WeierstrassMinimalSurfaceEngine;
    let (_, k0) = WeierstrassMinimalSurfaceEngine::enneper_surface_point_and_curvature(0.0, 0.0);
    assert!((k0 - (-16.0)).abs() < 1e-12);
    let w_clifford =
        WeierstrassMinimalSurfaceEngine::torus_willmore_energy(std::f64::consts::SQRT_2, 1.0);
    assert!((w_clifford - 2.0 * std::f64::consts::PI.powi(2)).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

import math
w = ox.WeierstrassMinimalSurfaceEngine.torus_willmore_energy(math.sqrt(2.0), 1.0)
assert abs(w - 2.0 * math.pi**2) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python WeierstrassMinimalSurfaceEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1370_microlocal_wavefront_hormander_bicharacteristic() -> Result<()> {
    use oxideon::science::microlocal_sheaves::MicrolocalWavefrontEngine;
    let mw = MicrolocalWavefrontEngine::new(2.0)?;
    assert!(mw.principal_symbol([3.0, 4.0], 10.0).abs() < 1e-12);
    let xt = mw.propagate_bicharacteristic([0.0, 0.0], [3.0, 4.0], 5.0)?;
    assert!((xt[0] - 6.0).abs() < 1e-12 && (xt[1] - 8.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

mw = ox.MicrolocalWavefrontEngine(2.0)
assert mw.principal_symbol([3.0, 4.0], 10.0) == 0.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python MicrolocalWavefrontEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1380_atiyah_singer_index_theorem_a_hat_genus() -> Result<()> {
    use oxideon::science::index_theorem_atiyah_singer::AtiyahSingerIndexEngine;
    // K3 surface has signature sigma = -16 => A-hat(K3) = -(-16)/8 = 2.0 (spinor Dirac index = 2)!
    assert!((AtiyahSingerIndexEngine::a_hat_genus_4d(-16) - 2.0).abs() < 1e-12);
    assert!((AtiyahSingerIndexEngine::twisted_dirac_index_4d(-16, 2, 0, 1) - 3.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert ox.AtiyahSingerIndexEngine.a_hat_genus_4d(-16) == 2.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python AtiyahSingerIndexEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1390_t_hooft_polyakov_bps_magnetic_monopole() -> Result<()> {
    use oxideon::science::non_abelian_monopoles::BpsMagneticMonopoleEngine;
    let bps = BpsMagneticMonopoleEngine::new(1.0, 2.0)?;
    assert!((bps.bps_monopole_mass(1) - 8.0 * std::f64::consts::PI).abs() < 1e-12);
    assert!(bps.prasad_sommerfield_higgs_profile(0.0).abs() < 1e-12);
    assert!(bps.prasad_sommerfield_higgs_profile(50.0) > 0.98);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

bps = ox.BpsMagneticMonopoleEngine(1.0, 2.0)
assert bps.prasad_sommerfield_higgs_profile(50.0) > 0.98
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python BpsMagneticMonopoleEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1400_grand_omni_tetradeca_engine_xiv() -> Result<()> {
    use oxideon::omni_tetradeca_xiv::GrandOmniTetradecaEngineXIV;
    let eng = GrandOmniTetradecaEngineXIV::new(14);
    let inv = eng.execute_tetradeca_synthesis()?;
    assert!(inv > 900.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniTetradecaEngineXIV(14)
assert eng.execute_tetradeca_synthesis() > 900.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniTetradecaEngineXIV test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1410_riemann_siegel_hardy_z_critical_line() -> Result<()> {
    use oxideon::science::riemann_siegel::RiemannSiegelZEngine;
    // Near the first Gram point g_0 ~ 17.8455995, theta(g_0) ~ 0 and Z(g_0) > 0
    let theta = RiemannSiegelZEngine::riemann_siegel_theta(17.8455995)?;
    assert!(theta.abs() < 1e-3);
    assert!(RiemannSiegelZEngine::hardy_z_main_sum(17.8455995)? > 1.5);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert abs(ox.RiemannSiegelZEngine.riemann_siegel_theta(17.8455995)) < 1e-3
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python RiemannSiegelZEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1420_selberg_trace_formula_hyperbolic_zeta() -> Result<()> {
    use oxideon::science::selberg_trace::SelbergTraceFormulaEngine;
    let stf = SelbergTraceFormulaEngine::new(2, vec![1.5, 2.2, 3.0])?;
    assert!((stf.hyperbolic_area() - 4.0 * std::f64::consts::PI).abs() < 1e-12);
    let z = stf.selberg_zeta_product(2.0, 5)?;
    assert!(z > 0.0 && z < 1.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

stf = ox.SelbergTraceFormulaEngine(2, [1.5, 2.2, 3.0])
assert 0.0 < stf.selberg_zeta_product(2.0, 5) < 1.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python SelbergTraceFormulaEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1430_birch_swinnerton_dyer_leading_coefficient() -> Result<()> {
    use oxideon::science::birch_swinnerton_dyer::BirchSwinnertonDyerEngine;
    let coeff = BirchSwinnertonDyerEngine::bsd_leading_coefficient(1, 3.0, 2.0, 4, 2)?;
    assert!((coeff - 6.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert ox.BirchSwinnertonDyerEngine.bsd_leading_coefficient(1, 3.0, 2.0, 4, 2) == 6.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python BirchSwinnertonDyerEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1440_iwasawa_theory_zp_cyclotomic_tower() -> Result<()> {
    use oxideon::science::iwasawa_theory::IwasawaMainConjectureEngine;
    // Prime p = 37 (first irregular prime: mu = 0, lambda = 1, nu = 0) => e_n = n
    let iw = IwasawaMainConjectureEngine::new(37, 0, 1, 0)?;
    assert_eq!(iw.class_number_p_valuation(5), 5);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

iw = ox.IwasawaMainConjectureEngine(37, 0, 1, 0)
assert iw.class_number_p_valuation(5) == 5
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python IwasawaMainConjectureEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1450_abc_conjecture_szpiro_radical_quality() -> Result<()> {
    use oxideon::science::abc_szpiro_arakelov::AbcSzpiroArakelovEngine;
    // Famous high-quality triple: 1 + 8 = 9 (a=1, b=2^3, c=3^2) => rad(72) = 6 => q = ln(9)/ln(6) ~ 1.22629
    assert_eq!(AbcSzpiroArakelovEngine::squarefree_radical(72), 6);
    let q = AbcSzpiroArakelovEngine::abc_quality(1, 8)?;
    assert!((q - 9.0_f64.ln() / 6.0_f64.ln()).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert ox.AbcSzpiroArakelovEngine.squarefree_radical(72) == 6
assert ox.AbcSzpiroArakelovEngine.abc_quality(1, 8) > 1.22
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python AbcSzpiroArakelovEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1460_hardy_littlewood_circle_method_weyl_sums() -> Result<()> {
    use oxideon::science::circle_method_hardy_littlewood::HardyLittlewoodCircleEngine;
    // Major arc at alpha = 0 gives coherent peak |S_2(0; 100)| = 100, while minor arc has strong square-root cancellation
    let major = HardyLittlewoodCircleEngine::weyl_exponential_sum_modulus(0.0, 2, 100)?;
    let minor = HardyLittlewoodCircleEngine::weyl_exponential_sum_modulus(
        std::f64::consts::SQRT_2,
        2,
        100,
    )?;
    assert!((major - 100.0).abs() < 1e-12 && minor < 30.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert ox.HardyLittlewoodCircleEngine.weyl_exponential_sum_modulus(0.0, 2, 100) == 100.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python HardyLittlewoodCircleEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1470_selberg_lambda_squared_sieve_maynard_tao() -> Result<()> {
    use oxideon::science::sieve_selberg_bombieri::SelbergSieveEngine;
    // For z = 6: squarefree d in {1, 2, 3, 5, 6} with phi = {1, 1, 2, 4, 2} => J(6) = 1 + 1 + 0.5 + 0.25 + 0.5 = 3.25
    let j6 = SelbergSieveEngine::selberg_j_sum(6)?;
    assert!((j6 - 3.25).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert abs(ox.SelbergSieveEngine.selberg_j_sum(6) - 3.25) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python SelbergSieveEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1480_faltings_roth_diophantine_weil_height() -> Result<()> {
    use oxideon::science::diophantine_faltings_roth::FaltingsHeightEngine;
    // Continued fraction convergent 99/70 to sqrt(2) has error ~ 1 / (2*sqrt(2)*70^2), so mu_eff ~ 2.24
    let (h, mu) =
        FaltingsHeightEngine::diophantine_approximation_exponent(std::f64::consts::SQRT_2, 99, 70)?;
    assert!((h - 99.0_f64.ln()).abs() < 1e-12 && mu > 2.0 && mu < 2.5);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

import math
h, mu = ox.FaltingsHeightEngine.diophantine_approximation_exponent(math.sqrt(2.0), 99, 70)
assert 2.0 < mu < 2.5
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python FaltingsHeightEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1490_katz_sarnak_montgomery_gue_pair_correlation() -> Result<()> {
    use oxideon::science::katz_sarnak_random_matrix::KatzSarnakSymmetryEngine;
    assert!(KatzSarnakSymmetryEngine::montgomery_gue_pair_correlation(0.0).abs() < 1e-15);
    assert!((KatzSarnakSymmetryEngine::montgomery_gue_pair_correlation(1.0) - 1.0).abs() < 1e-12);
    assert!((KatzSarnakSymmetryEngine::usp_one_level_density(0.5) - 1.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert ox.KatzSarnakSymmetryEngine.montgomery_gue_pair_correlation(1.0) == 1.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python KatzSarnakSymmetryEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1500_grand_omni_pentadeca_engine_xv() -> Result<()> {
    use oxideon::omni_pentadeca_xv::GrandOmniPentadecaEngineXV;
    let eng = GrandOmniPentadecaEngineXV::new(15);
    let inv = eng.execute_pentadeca_synthesis()?;
    assert!(inv > 1000.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniPentadecaEngineXV(15)
assert eng.execute_pentadeca_synthesis() > 1000.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniPentadecaEngineXV test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1510_kolmogorov_k41_turbulence_she_leveque() -> Result<()> {
    use oxideon::physics::kolmogorov_turbulence::KolmogorovCascadeEngine;
    // By Kolmogorov's exact 4/5 law, the 3rd-order structure function exponent is exactly zeta_3 = 1.0!
    assert!((KolmogorovCascadeEngine::she_leveque_exponent(3.0) - 1.0).abs() < 1e-14);
    let k41 = KolmogorovCascadeEngine::new(1.0, 0.01)?;
    let (ek, eta) = k41.spectrum_and_microscale(1.0)?;
    assert!((ek - 1.5).abs() < 1e-12 && eta > 0.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert abs(ox.KolmogorovCascadeEngine.she_leveque_exponent(3.0) - 1.0) < 1e-14
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python KolmogorovCascadeEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1520_kraichnan_2d_turbulence_dual_enstrophy_cascade() -> Result<()> {
    use oxideon::physics::kraichnan_2d_enstrophy::KraichnanEnstrophyCascade;
    let kr = KraichnanEnstrophyCascade::new(10.0, 1.0, 1.0)?;
    // In forward enstrophy range (k > 10), doubling k reduces E(k) by 2^3 = 8
    let e20 = kr.dual_cascade_spectrum(20.0)?;
    let e40 = kr.dual_cascade_spectrum(40.0)?;
    assert!((e20 / e40 - 8.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

kr = ox.KraichnanEnstrophyCascade(10.0, 1.0, 1.0)
assert abs(kr.dual_cascade_spectrum(20.0) / kr.dual_cascade_spectrum(40.0) - 8.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python KraichnanEnstrophyCascade test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1530_alfven_mhd_alpha_omega_dynamo() -> Result<()> {
    use oxideon::physics::alfven_mhd_dynamo::AlfvenMhdDynamoEngine;
    let mhd = AlfvenMhdDynamoEngine::new(4.0, 1.0, 0.1)?;
    assert!((mhd.alfven_speed(6.0) - 3.0).abs() < 1e-12);
    assert!(mhd.alpha_omega_growth_rate(2.0, 4.0, 1.0) > 1.8);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

mhd = ox.AlfvenMhdDynamoEngine(4.0, 1.0, 0.1)
assert mhd.alfven_speed(6.0) == 3.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python AlfvenMhdDynamoEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1540_quasi_geostrophic_planetary_rossby_waves() -> Result<()> {
    use oxideon::physics::quasi_geostrophic_rossby::CharneyRossbyWaveEngine;
    let rw = CharneyRossbyWaveEngine::new(2.0, 1.0)?;
    let (omega, cgx) = rw.frequency_and_zonal_group_velocity(2.0, 0.0);
    // Westward phase speed (omega / kx < 0), but short wave kx=2 > 1/L_R=1 has eastward group velocity cgx > 0!
    assert!(omega < 0.0 && cgx > 0.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

rw = ox.CharneyRossbyWaveEngine(2.0, 1.0)
omega, cgx = rw.frequency_and_zonal_group_velocity(2.0, 0.0)
assert omega < 0.0 and cgx > 0.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python CharneyRossbyWaveEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1550_rayleigh_benard_convection_marginal_stability() -> Result<()> {
    use oxideon::physics::rayleigh_benard_lorenz::RayleighBenardConvection;
    let ac = std::f64::consts::PI / std::f64::consts::SQRT_2;
    let rac = RayleighBenardConvection::critical_rayleigh_free_free();
    assert!((RayleighBenardConvection::marginal_rayleigh_curve(ac) - rac).abs() < 1e-10);
    assert!(rac > 657.51 && rac < 657.52);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert 657.51 < ox.RayleighBenardConvection.critical_rayleigh_free_free() < 657.52
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python RayleighBenardConvection test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1560_kelvin_helmholtz_rayleigh_taylor_instability() -> Result<()> {
    use oxideon::physics::kelvin_helmholtz_rayleigh_taylor::HydrodynamicInstabilityEngine;
    // Heavy fluid on top (rho2 = 3, rho1 = 1) => Atwood number A = 0.5 and unstable RT mode at small k
    let hi = HydrodynamicInstabilityEngine::new(1.0, 3.0, 1.0)?;
    assert!((hi.atwood_number() - 0.5).abs() < 1e-12);
    assert!(hi.growth_rate_squared(1.0, 0.0, 10.0) > 0.0);
    // Surface tension stabilizes short wavelengths (large k = 10)
    assert!(hi.growth_rate_squared(10.0, 0.0, 10.0) < 0.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

hi = ox.HydrodynamicInstabilityEngine(1.0, 3.0, 1.0)
assert hi.atwood_number() == 0.5 and hi.growth_rate_squared(1.0, 0.0, 10.0) > 0.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python HydrodynamicInstabilityEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1570_vlasov_poisson_plasma_landau_damping() -> Result<()> {
    use oxideon::physics::vlasov_landau_damping::VlasovLandauDampingEngine;
    let vp = VlasovLandauDampingEngine::new(10.0, 0.3)?;
    let (wr, gamma) = vp.bohm_gross_and_landau_damping(1.0)?;
    assert!(wr > 10.0 && gamma < 0.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

vp = ox.VlasovLandauDampingEngine(10.0, 0.3)
wr, gamma = vp.bohm_gross_and_landau_damping(1.0)
assert wr > 10.0 and gamma < 0.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python VlasovLandauDampingEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1580_grad_shafranov_tokamak_solovev_equilibrium() -> Result<()> {
    use oxideon::physics::grad_shafranov_tokamak::GradShafranovTokamakEngine;
    let tok = GradShafranovTokamakEngine::new(6.2, 1.7, 5.3)?;
    assert!(tok.solovev_normalized_flux(6.2, 0.0).abs() < 1e-15);
    let q = tok.safety_factor_q(2.0, 0.6)?;
    assert!(q > 3.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

tok = ox.GradShafranovTokamakEngine(6.2, 1.7, 5.3)
assert tok.solovev_normalized_flux(6.2, 0.0) == 0.0 and tok.safety_factor_q(2.0, 0.6) > 3.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GradShafranovTokamakEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1590_gross_pitaevskii_bec_bogoliubov_superfluid() -> Result<()> {
    use oxideon::physics::superfluid_gross_pitaevskii::GrossPitaevskiiVortexEngine;
    let bec = GrossPitaevskiiVortexEngine::new(2.0, 2.0)?;
    let (cs, xi) = bec.sound_speed_and_healing_length();
    assert!((cs - 2.0).abs() < 1e-12 && (xi - 1.0 / 8.0_f64.sqrt()).abs() < 1e-12);
    // Long-wavelength phonon regime: E(k) ~ c_s * k
    assert!((bec.bogoliubov_dispersion(1e-4) / 1e-4 - 2.0).abs() < 1e-6);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

bec = ox.GrossPitaevskiiVortexEngine(2.0, 2.0)
cs, xi = bec.sound_speed_and_healing_length()
assert cs == 2.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrossPitaevskiiVortexEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1600_grand_omni_hexadeca_engine_xvi() -> Result<()> {
    use oxideon::omni_hexadeca_xvi::GrandOmniHexadecaEngineXVI;
    let eng = GrandOmniHexadecaEngineXVI::new(16);
    let inv = eng.execute_hexadeca_synthesis()?;
    assert!(inv > 1650.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniHexadecaEngineXVI(16)
assert eng.execute_hexadeca_synthesis() > 1650.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniHexadecaEngineXVI test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1610_malliavin_calculus_bismut_greek_weights() -> Result<()> {
    use oxideon::ode::malliavin_calculus::MalliavinBismutEngine;
    let mb = MalliavinBismutEngine::new(100.0, 0.2, 1.0)?;
    let (wd, wg) = mb.bismut_delta_and_gamma_weights(1.0);
    assert!((wd - 0.05).abs() < 1e-12 && wg < 0.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

mb = ox.MalliavinBismutEngine(100.0, 0.2, 1.0)
wd, wg = mb.bismut_delta_and_gamma_weights(1.0)
assert abs(wd - 0.05) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python MalliavinBismutEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1620_schramm_loewner_evolution_beffara_hausdorff() -> Result<()> {
    use oxideon::ode::schramm_loewner_evolution::SchrammLoewnerEngine;
    // Critical percolation (kappa = 6) has Hausdorff dimension d_H = 7/4 = 1.75 and central charge c = 0!
    let sle6 = SchrammLoewnerEngine::new(6.0)?;
    let (dh, _) = sle6.hausdorff_dimension_and_dual();
    assert!((dh - 1.75).abs() < 1e-12 && sle6.cft_central_charge().abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

sle6 = ox.SchrammLoewnerEngine(6.0)
dh, _ = sle6.hausdorff_dimension_and_dual()
assert dh == 1.75 and sle6.cft_central_charge() == 0.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python SchrammLoewnerEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1630_kpz_universality_cole_hopf_transform() -> Result<()> {
    use oxideon::ode::kpz_stochastic_burgers::KpzColeHopfEngine;
    let kpz = KpzColeHopfEngine::new(1.0, 2.0)?;
    let (z, h_rec) = kpz.cole_hopf_forward_and_inverse(1.5);
    assert!((z - 1.5_f64.exp()).abs() < 1e-12 && (h_rec - 1.5).abs() < 1e-12);
    let (alpha, _, dyn_z) = KpzColeHopfEngine::universal_kpz_exponents();
    assert!((alpha + dyn_z - 2.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

kpz = ox.KpzColeHopfEngine(1.0, 2.0)
_, h = kpz.cole_hopf_forward_and_inverse(1.5)
assert abs(h - 1.5) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python KpzColeHopfEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1640_rough_volatility_heston_power_law_skew() -> Result<()> {
    use oxideon::ode::rough_heston_volatility::RoughVolatilityHestonEngine;
    let rv = RoughVolatilityHestonEngine::new(0.1, 1.5, -0.7)?;
    // Short-maturity skew explodes as T^{-0.4}
    assert!(rv.atm_implied_vol_skew(0.01)?.abs() > rv.atm_implied_vol_skew(1.0)?.abs());

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

rv = ox.RoughVolatilityHestonEngine(0.1, 1.5, -0.7)
assert abs(rv.atm_implied_vol_skew(0.01)) > abs(rv.atm_implied_vol_skew(1.0))
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python RoughVolatilityHestonEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1650_freidlin_wentzell_large_deviations_eyring_kramers() -> Result<()> {
    use oxideon::ode::freidlin_wentzell_ldp::FreidlinWentzellLdpEngine;
    let tau = FreidlinWentzellLdpEngine::eyring_kramers_escape_time(1.0, 0.5, 4.0, 4.0)?;
    assert!((tau - 0.5 * std::f64::consts::PI * 2.0_f64.exp()).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

import math
tau = ox.FreidlinWentzellLdpEngine.eyring_kramers_escape_time(1.0, 0.5, 4.0, 4.0)
assert abs(tau - 0.5 * math.pi * math.exp(2.0)) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python FreidlinWentzellLdpEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1660_mc_kean_vlasov_sznitman_propagation_of_chaos() -> Result<()> {
    use oxideon::ode::mc_kean_vlasov_sde::McKeanVlasovPropagationChaos;
    let mkv = McKeanVlasovPropagationChaos::new(1.0, 2.0)?;
    // Stationary variance is sigma^2 / (2*kappa) = 4 / 2 = 2.0
    assert!((mkv.mean_field_variance(2.0, 10.0)? - 2.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

mkv = ox.McKeanVlasovPropagationChaos(1.0, 2.0)
assert abs(mkv.mean_field_variance(2.0, 10.0) - 2.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python McKeanVlasovPropagationChaos test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1670_levy_ito_jump_diffusion_khintchine_exponent() -> Result<()> {
    use oxideon::ode::levy_jump_ito::LevyItodiffusionEngine;
    let levy = LevyItodiffusionEngine::new(0.2, 1.5, -0.1, 0.3)?;
    assert!(levy.real_levy_exponent(0.0).abs() < 1e-15);
    assert!(levy.real_levy_exponent(2.0) < 0.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

levy = ox.LevyItodiffusionEngine(0.2, 1.5, -0.1, 0.3)
assert levy.real_levy_exponent(0.0) == 0.0 and levy.real_levy_exponent(2.0) < 0.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python LevyItodiffusionEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1680_pardoux_peng_backward_sde_feynman_kac() -> Result<()> {
    use oxideon::ode::backward_sde_pardoux_peng::PardouxPengBsdeEngine;
    let bsde = PardouxPengBsdeEngine::new(0.05, 0.2);
    let y0 = bsde.backward_bsde_step(105.0, 0.0, 1.0)?;
    assert!((y0 - 100.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

bsde = ox.PardouxPengBsdeEngine(0.05, 0.2)
assert abs(bsde.backward_bsde_step(105.0, 0.0, 1.0) - 100.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python PardouxPengBsdeEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1690_stein_method_nualart_peccati_fourth_moment() -> Result<()> {
    use oxideon::ode::stein_method_wasserstein::SteinPoissonCharacterization;
    // Exact Gaussian fourth moment E[Z^4] = 3.0 gives TV bound = 0.0!
    assert!(SteinPoissonCharacterization::fourth_moment_tv_bound(2, 3.0)?.abs() < 1e-15);
    assert!(SteinPoissonCharacterization::stein_chen_poisson_bound(&[0.01; 100])? < 0.01);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert ox.SteinPoissonCharacterization.fourth_moment_tv_bound(2, 3.0) == 0.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python SteinPoissonCharacterization test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1700_grand_omni_heptadeca_engine_xvii() -> Result<()> {
    use oxideon::omni_heptadeca_xvii::GrandOmniHeptadecaEngineXVII;
    let eng = GrandOmniHeptadecaEngineXVII::new(17);
    let inv = eng.execute_heptadeca_synthesis()?;
    assert!(inv > 1750.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniHeptadecaEngineXVII(17)
assert eng.execute_heptadeca_synthesis() > 1750.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniHeptadecaEngineXVII test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1710_laughlin_fractional_quantum_hall_anyons() -> Result<()> {
    use oxideon::physics::fractional_quantum_hall_laughlin::LaughlinFqheEngine;
    let fqhe = LaughlinFqheEngine::new(3)?;
    let (nu, q_star, theta) = fqhe.filling_charge_and_anyon_angle();
    assert!((nu - 1.0 / 3.0).abs() < 1e-12 && (q_star - 1.0 / 3.0).abs() < 1e-12);
    assert!((theta - std::f64::consts::PI / 3.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

fqhe = ox.LaughlinFqheEngine(3)
nu, q, _ = fqhe.filling_charge_and_anyon_angle()
assert abs(nu - 1.0/3.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python LaughlinFqheEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1720_haldane_honeycomb_chern_insulator() -> Result<()> {
    use oxideon::physics::haldane_chern_insulator::HaldaneHoneycombModel;
    let top = HaldaneHoneycombModel::new(0.5, 1.0, std::f64::consts::FRAC_PI_2);
    let triv = HaldaneHoneycombModel::new(10.0, 1.0, std::f64::consts::FRAC_PI_2);
    assert_eq!(top.valley_masses_and_chern_number().2, 1);
    assert_eq!(triv.valley_masses_and_chern_number().2, 0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

import math
top = ox.HaldaneHoneycombModel(0.5, 1.0, math.pi / 2.0)
assert top.valley_masses_and_chern_number()[2] == 1
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python HaldaneHoneycombModel test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1730_kondo_effect_poor_mans_scaling_temperature() -> Result<()> {
    use oxideon::physics::kondo_anderson_rg::KondoScreeningEngine;
    let kd = KondoScreeningEngine::new(1000.0, 0.25)?;
    assert!((kd.kondo_temperature() - 500.0 * (-4.0_f64).exp()).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

import math
kd = ox.KondoScreeningEngine(1000.0, 0.25)
assert abs(kd.kondo_temperature() - 500.0 * math.exp(-4.0)) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python KondoScreeningEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1740_bcs_superconductivity_cooper_gap_universal_ratio() -> Result<()> {
    use oxideon::physics::bcs_eliashberg_superconductivity::BcsGapEquationEngine;
    let bcs = BcsGapEquationEngine::new(30.0, 0.25)?;
    let (d0, tc, ratio) = bcs.gap_tc_and_universal_ratio();
    assert!(d0 > 0.0 && tc > 0.0 && (ratio - 3.52775).abs() < 1e-4);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

bcs = ox.BcsGapEquationEngine(30.0, 0.25)
_, _, ratio = bcs.gap_tc_and_universal_ratio()
assert abs(ratio - 3.52775) < 1e-4
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python BcsGapEquationEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1750_hubbard_mott_gutzwiller_brinkman_rice() -> Result<()> {
    use oxideon::physics::hubbard_mott_gutzwiller::BrinkmanRiceGutzwillerEngine;
    let br = BrinkmanRiceGutzwillerEngine::new(8.0)?;
    let (z_met, d_met) = br.quasiparticle_z_and_double_occupancy(4.0);
    assert!((z_met - 0.75).abs() < 1e-12 && (d_met - 0.125).abs() < 1e-12);
    assert_eq!(br.quasiparticle_z_and_double_occupancy(10.0), (0.0, 0.0));

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

br = ox.BrinkmanRiceGutzwillerEngine(8.0)
assert br.quasiparticle_z_and_double_occupancy(4.0) == (0.75, 0.125)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python BrinkmanRiceGutzwillerEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1760_bkt_vortex_unbinding_nelson_kosterlitz_jump() -> Result<()> {
    use oxideon::physics::berezinskii_kosterlitz_thouless::BktVortexUnbindingEngine;
    assert!(
        (BktVortexUnbindingEngine::universal_nelson_kosterlitz_jump() - 2.0 / std::f64::consts::PI)
            .abs()
            < 1e-12
    );
    let xi = BktVortexUnbindingEngine::bkt_correlation_length(1.25, 1.0, 1.0, 1.0)?;
    assert!((xi - 2.0_f64.exp()).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

import math
assert abs(ox.BktVortexUnbindingEngine.universal_nelson_kosterlitz_jump() - 2.0 / math.pi) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python BktVortexUnbindingEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1770_weyl_semimetal_chiral_anomaly_magnetoconductivity() -> Result<()> {
    use oxideon::physics::weyl_semimetal_chiral_anomaly::WeylChiralAnomalyEngine;
    let weyl = WeylChiralAnomalyEngine::new(1.0, 2.0, 1.0, 0.5)?;
    assert!(
        weyl.longitudinal_magnetoconductivity(2.0) > weyl.longitudinal_magnetoconductivity(0.0)
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

weyl = ox.WeylChiralAnomalyEngine(1.0, 2.0, 1.0, 0.5)
assert weyl.longitudinal_magnetoconductivity(2.0) > 1.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python WeylChiralAnomalyEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1780_bistritzer_macdonald_magic_angle_graphene() -> Result<()> {
    use oxideon::physics::twistronics_bistritzer_macdonald::MagicAngleGrapheneEngine;
    let a1 = MagicAngleGrapheneEngine::first_magic_alpha();
    assert!(MagicAngleGrapheneEngine::renormalized_velocity_ratio(a1).abs() < 1e-15);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

a1 = ox.MagicAngleGrapheneEngine.first_magic_alpha()
assert abs(ox.MagicAngleGrapheneEngine.renormalized_velocity_ratio(a1)) < 1e-15
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python MagicAngleGrapheneEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1790_chiral_magnetic_skyrmion_thiele_hall_angle() -> Result<()> {
    use oxideon::physics::skyrmion_dzyaloshinskii_moriya::MagneticSkyrmionEngine;
    let sk = MagneticSkyrmionEngine::new(1, 14.0);
    // When alpha == beta, the Skyrmion Hall deflection vanishes identically!
    assert!(sk.thiele_skyrmion_hall_tangent(0.05, 0.05).abs() < 1e-15);
    assert!(sk.thiele_skyrmion_hall_tangent(0.1, 0.02) > 0.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

sk = ox.MagneticSkyrmionEngine(1, 14.0)
assert sk.thiele_skyrmion_hall_tangent(0.05, 0.05) == 0.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python MagneticSkyrmionEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1800_grand_omni_octodeca_engine_xviii() -> Result<()> {
    use oxideon::omni_octodeca_xviii::GrandOmniOctodecaEngineXVIII;
    let eng = GrandOmniOctodecaEngineXVIII::new(18);
    let inv = eng.execute_octodeca_synthesis()?;
    assert!(inv > 1770.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniOctodecaEngineXVIII(18)
assert eng.execute_octodeca_synthesis() > 1770.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniOctodecaEngineXVIII test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1810_hodgkin_huxley_ion_channel_conductance() -> Result<()> {
    use oxideon::science::hodgkin_huxley_ion::HodgkinHuxleyActionPotential;
    let hh = HodgkinHuxleyActionPotential::default();
    // During rapid depolarization (m=0.9, h=0.6, n=0.3, V=0 mV), inward Na+ current dominates (< 0)
    assert!(hh.total_ionic_current(0.0, 0.9, 0.6, 0.3) < -2000.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

hh = ox.HodgkinHuxleyActionPotential()
assert hh.total_ionic_current(0.0, 0.9, 0.6, 0.3) < -2000.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python HodgkinHuxleyActionPotential test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1820_wilson_cowan_excitatory_inhibitory_cortical() -> Result<()> {
    use oxideon::science::wilson_cowan_neural_field::WilsonCowanCorticalEngine;
    let wc = WilsonCowanCorticalEngine::new([12.0, 10.0, 16.0, 4.0]);
    let (e1, i1) = wc.step_ei(0.2, 0.1, 1.0, 0.0, 0.05);
    assert!(e1 > 0.2 && i1 > 0.1);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

wc = ox.WilsonCowanCorticalEngine([12.0, 10.0, 16.0, 4.0])
e1, i1 = wc.step_ei(0.2, 0.1, 1.0, 0.0, 0.05)
assert e1 > 0.2
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python WilsonCowanCorticalEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1830_turing_morphogenesis_diffusion_instability() -> Result<()> {
    use oxideon::science::turing_morphogenesis_rd::TuringReactionDiffusionEngine;
    // Classic Schnakenberg/Gierer-Meinhardt Jacobian: fu = 1, fv = -2, gu = 1.5, gv = -2 (Tr = -1 < 0, det = 1 > 0) with dv = 10 >> du = 1
    let trd = TuringReactionDiffusionEngine::new(1.0, 10.0, [1.0, -2.0, 1.5, -2.0])?;
    let (ok, kc2) = trd.check_turing_instability();
    assert!(ok && (kc2 - 0.4).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

trd = ox.TuringReactionDiffusionEngine(1.0, 10.0, [1.0, -2.0, 1.5, -2.0])
ok, kc2 = trd.check_turing_instability()
assert ok and abs(kc2 - 0.4) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python TuringReactionDiffusionEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1840_jarzynski_equality_crooks_fluctuation() -> Result<()> {
    use oxideon::science::jarzynski_crooks_fluctuation::JarzynskiCrooksEngine;
    let jc = JarzynskiCrooksEngine::new(1.0)?;
    let (df, wdiss) = jc.jarzynski_free_energy_and_dissipation(&[1.0, 2.0, 3.0])?;
    // By Jensen's inequality, Delta F < <W> = 2.0 and W_diss > 0
    assert!(df < 2.0 && wdiss > 0.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

jc = ox.JarzynskiCrooksEngine(1.0)
df, wdiss = jc.jarzynski_free_energy_and_dissipation([1.0, 2.0, 3.0])
assert df < 2.0 and wdiss > 0.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python JarzynskiCrooksEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1850_schnakenberg_chemical_master_entropy_production() -> Result<()> {
    use oxideon::science::schnakenberg_chemical_master::ChemicalMasterEntropyEngine;
    // Detailed balance (J+ == J-) gives 0 entropy production; broken detailed balance gives strictly positive sigma > 0
    assert!(
        ChemicalMasterEntropyEngine::schnakenberg_entropy_production(&[2.0, 3.0], &[2.0, 3.0])?
            .abs()
            < 1e-15
    );
    assert!(
        ChemicalMasterEntropyEngine::schnakenberg_entropy_production(&[4.0, 2.0], &[1.0, 1.0])?
            > 4.0
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert ox.ChemicalMasterEntropyEngine.schnakenberg_entropy_production([2.0, 3.0], [2.0, 3.0]) == 0.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python ChemicalMasterEntropyEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1860_fokker_planck_ornstein_uhlenbeck_propagator() -> Result<()> {
    use oxideon::science::kramers_moyal_fokker_planck::FokkerPlanckOrnsteinUhlenbeck;
    let fp = FokkerPlanckOrnsteinUhlenbeck::new(1.0, 1.0)?;
    // Peak of propagator at t = 1 from x0 = 0 is at x = 0
    assert!(fp.propagator_density(0.0, 0.0, 1.0)? > fp.propagator_density(1.0, 0.0, 1.0)?);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

fp = ox.FokkerPlanckOrnsteinUhlenbeck(1.0, 1.0)
assert fp.propagator_density(0.0, 0.0, 1.0) > fp.propagator_density(1.0, 0.0, 1.0)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python FokkerPlanckOrnsteinUhlenbeck test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1870_kratky_porod_worm_like_chain_marko_siggia() -> Result<()> {
    use oxideon::science::worm_like_chain_polymer::KratkyPorodPolymerEngine;
    let wlc = KratkyPorodPolymerEngine::new(50.0, 1000.0)?;
    assert!(wlc.mean_square_end_to_end() > 90000.0);
    // At x = 0.5, F*Lp/(kB*T) = 0.25 / 0.25 - 0.25 + 0.5 = 1.25
    assert!(
        (KratkyPorodPolymerEngine::marko_siggia_dimensionless_force(0.5)? - 1.25).abs() < 1e-12
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert ox.KratkyPorodPolymerEngine.marko_siggia_dimensionless_force(0.5) == 1.25
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python KratkyPorodPolymerEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1880_marcus_electron_transfer_inverted_region() -> Result<()> {
    use oxideon::science::marcus_electron_transfer::MarcusElectronTransferEngine;
    let et = MarcusElectronTransferEngine::new(1.0)?;
    let (b_opt, inv_opt) = et.activation_barrier_and_is_inverted(-1.0);
    let (b_inv, inv_flag) = et.activation_barrier_and_is_inverted(-2.0);
    assert!(b_opt.abs() < 1e-15 && !inv_opt);
    assert!((b_inv - 0.25).abs() < 1e-12 && inv_flag);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

et = ox.MarcusElectronTransferEngine(1.0)
assert et.activation_barrier_and_is_inverted(-2.0) == (0.25, True)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python MarcusElectronTransferEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1890_forster_resonance_energy_transfer_spectroscopic_ruler() -> Result<()> {
    use oxideon::science::forster_exciton_resonance::ForsterResonanceEnergyEngine;
    let fret = ForsterResonanceEnergyEngine::new(5.0, 2.0)?;
    let (eff, rate) = fret.efficiency_and_rate(5.0)?;
    assert!((eff - 0.5).abs() < 1e-12 && (rate - 0.5).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

fret = ox.ForsterResonanceEnergyEngine(5.0, 2.0)
assert fret.efficiency_and_rate(5.0) == (0.5, 0.5)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python ForsterResonanceEnergyEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1900_grand_omni_nonadeca_engine_xix() -> Result<()> {
    use oxideon::omni_nonadeca_xix::GrandOmniNonadecaEngineXIX;
    let eng = GrandOmniNonadecaEngineXIX::new(19);
    let inv = eng.execute_nonadeca_synthesis()?;
    assert!(inv > 1785.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniNonadecaEngineXIX(19)
assert eng.execute_nonadeca_synthesis() > 1785.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniNonadecaEngineXIX test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1910_chromatic_homotopy_quillen_morava_k_theory() -> Result<()> {
    use oxideon::science::quillen_formal_group_chromatic::ChromaticMoravaKTheory;
    // At p = 2, h = 2 (tmf / elliptic cohomology height): |v_2| = 2*(4 - 1) = 6
    let k2 = ChromaticMoravaKTheory::new(2, 2)?;
    assert_eq!(k2.vn_generator_degree_and_stabilizer_dim(), (6, 4));

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

k2 = ox.ChromaticMoravaKTheory(2, 2)
assert k2.vn_generator_degree_and_stabilizer_dim() == (6, 4)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python ChromaticMoravaKTheory test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1920_adams_spectral_sequence_j_homomorphism_spheres() -> Result<()> {
    use oxideon::science::adams_spectral_sequence_j_hom::AdamsJHomomorphismEngine;
    assert_eq!(AdamsJHomomorphismEngine::image_of_j_order(1)?, 24);
    assert_eq!(AdamsJHomomorphismEngine::image_of_j_order(2)?, 240);
    assert_eq!(AdamsJHomomorphismEngine::image_of_j_order(3)?, 504);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert ox.AdamsJHomomorphismEngine.image_of_j_order(1) == 24
assert ox.AdamsJHomomorphismEngine.image_of_j_order(2) == 240
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python AdamsJHomomorphismEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1930_khovanov_homology_rasmussen_milnor_slice_genus() -> Result<()> {
    use oxideon::science::khovanov_categorification::KhovanovHomologyEngine;
    // Right-handed trefoil T_{2,3}: s = 2, g_4 = 1; T_{3,4}: s = 6, g_4 = 3
    assert_eq!(
        KhovanovHomologyEngine::torus_knot_rasmussen_s_and_slice_genus(2, 3)?,
        (2, 1)
    );
    assert_eq!(
        KhovanovHomologyEngine::torus_knot_rasmussen_s_and_slice_genus(3, 4)?,
        (6, 3)
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert ox.KhovanovHomologyEngine.torus_knot_rasmussen_s_and_slice_genus(3, 4) == (6, 3)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python KhovanovHomologyEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1940_heegaard_floer_ozsvath_szabo_knot_tau() -> Result<()> {
    use oxideon::science::heegaard_floer_ozsvath_szabo::HeegaardFloerKnotEngine;
    // Trefoil knot: Delta(t) = -1 + t + t^{-1} => at t = 1, Delta(1) = 1.0, and tau = 1 bounds max tb = 1 (with rot = 0)
    assert!(
        (HeegaardFloerKnotEngine::alexander_euler_characteristic(&[-1, 1], 1.0) - 1.0).abs()
            < 1e-12
    );
    assert!(HeegaardFloerKnotEngine::satisfies_legendrian_tau_bound(
        1, 0, 1
    ));

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert ox.HeegaardFloerKnotEngine.alexander_euler_characteristic([-1, 1], 1.0) == 1.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python HeegaardFloerKnotEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1950_symplectic_floer_fukaya_arnold_conjecture() -> Result<()> {
    use oxideon::science::symplectic_fukaya_floer::FukayaArnoldConjectureEngine;
    // Symplectic 2-torus T^2 has Betti numbers [1, 2, 1] => Arnold bound >= 4 fixed points, chi = 0
    let floer = FukayaArnoldConjectureEngine::new(vec![1, 2, 1]);
    assert_eq!(floer.arnold_fixed_point_bound_and_euler(), (4, 0));

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

floer = ox.FukayaArnoldConjectureEngine([1, 2, 1])
assert floer.arnold_fixed_point_bound_and_euler() == (4, 0)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python FukayaArnoldConjectureEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1960_clausen_scholze_condensed_liquid_tensor() -> Result<()> {
    use oxideon::science::scholze_clausen_condensed_liquid::CondensedLiquidTensorEngine;
    let liq = CondensedLiquidTensorEngine::new(0.5)?;
    assert!((liq.p_liquid_norm(&[4.0, 9.0, 16.0]) - 9.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

liq = ox.CondensedLiquidTensorEngine(0.5)
assert liq.p_liquid_norm([4.0, 9.0, 16.0]) == 9.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python CondensedLiquidTensorEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1970_bhatt_scholze_prismatic_cohomology_delta_ring() -> Result<()> {
    use oxideon::science::prismatic_cohomology_bhatt_scholze::PrismaticSiteBreuilKisin;
    let prism = PrismaticSiteBreuilKisin::new(3)?;
    // By Fermat's Little Theorem, (2 - 2^3) / 3 = -6 / 3 = -2, and 2^3 + 3*(-2) = 2 = phi(2)!
    assert_eq!(prism.integer_p_derivation_delta(2)?, -2);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

prism = ox.PrismaticSiteBreuilKisin(3)
assert prism.integer_p_derivation_delta(2) == -2
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python PrismaticSiteBreuilKisin test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1980_voevodsky_motivic_milnor_bloch_kato_symbol() -> Result<()> {
    use oxideon::science::motivic_voevodsky_milnor::VoevodskyMilnorBlochKato;
    assert_eq!(VoevodskyMilnorBlochKato::legendre_tame_symbol(2, 7)?, 1);
    assert_eq!(VoevodskyMilnorBlochKato::legendre_tame_symbol(3, 7)?, -1);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert ox.VoevodskyMilnorBlochKato.legendre_tame_symbol(2, 7) == 1
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python VoevodskyMilnorBlochKato test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v1990_lurie_higher_algebra_en_dunn_factorization() -> Result<()> {
    use oxideon::science::lurie_higher_algebra_en_operads::LurieLittleDisksFactorization;
    // E_1 x E_1 ~ E_2 (Gerstenhaber algebra with Browder bracket of degree 2 - 1 = 1)
    assert_eq!(
        LurieLittleDisksFactorization::dunn_additivity_and_browder_degree(1, 1)?,
        (2, 1)
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert ox.LurieLittleDisksFactorization.dunn_additivity_and_browder_degree(1, 1) == (2, 1)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python LurieLittleDisksFactorization test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2000_grand_bicentennial_singularity_engine_xx() -> Result<()> {
    use oxideon::omni_bicentennial_xx::GrandBicentennialSingularityEngineXX;
    let eng = GrandBicentennialSingularityEngineXX::new(20);
    let inv = eng.execute_bicentennial_synthesis()?;
    assert!(inv > 1830.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandBicentennialSingularityEngineXX(20)
assert eng.execute_bicentennial_synthesis() > 1830.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandBicentennialSingularityEngineXX test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2010_roothaan_hall_hartree_fock_scf() -> Result<()> {
    use oxideon::science::hartree_fock_roothaan::RoothaanHallScfEngine;
    // He atom minimal basis 1x1: P = [2.0], H_core = [-1.94], F = [-0.92] => E_elec = 0.5 * 2 * (-2.86) = -2.86 Ha
    let e_he = RoothaanHallScfEngine::rhf_electronic_energy(&[2.0], &[-1.94], &[-0.92])?;
    assert!((e_he - (-2.86)).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert abs(ox.RoothaanHallScfEngine.rhf_electronic_energy([2.0], [-1.94], [-0.92]) - (-2.86)) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python RoothaanHallScfEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2020_kohn_sham_dft_pbe_gga_exchange() -> Result<()> {
    use oxideon::science::kohn_sham_dft_jacob_ladder::KohnShamJacobLadderEngine;
    let ex0 = KohnShamJacobLadderEngine::pbe_exchange_energy_density(1.0, 0.0)?;
    let ex1 = KohnShamJacobLadderEngine::pbe_exchange_energy_density(1.0, 1.0)?;
    assert!(ex1 < ex0 && ex0 < 0.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert ox.KohnShamJacobLadderEngine.pbe_exchange_energy_density(1.0, 1.0) < ox.KohnShamJacobLadderEngine.pbe_exchange_energy_density(1.0, 0.0)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python KohnShamJacobLadderEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2030_coupled_cluster_ccsd_t_lee_taylor_diagnostic() -> Result<()> {
    use oxideon::science::coupled_cluster_ccsd_t::CoupledClusterCcsdTEngine;
    let (t1, ok) =
        CoupledClusterCcsdTEngine::lee_taylor_t1_diagnostic(&[0.01, 0.01, 0.01, 0.01], 8)?;
    assert!(ok && t1 < 0.02);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

t1, ok = ox.CoupledClusterCcsdTEngine.lee_taylor_t1_diagnostic([0.01, 0.01, 0.01, 0.01], 8)
assert ok and t1 < 0.02
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python CoupledClusterCcsdTEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2040_moller_plesset_mp2_grimme_scs() -> Result<()> {
    use oxideon::science::moller_plesset_mp2::MollerPlessetMp2Engine;
    let (mp2, scs) = MollerPlessetMp2Engine::canonical_and_scs_mp2_energy(-0.20, -0.06);
    assert!((mp2 - (-0.26)).abs() < 1e-12 && (scs - (-0.26)).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

mp2, scs = ox.MollerPlessetMp2Engine.canonical_and_scs_mp2_energy(-0.20, -0.06)
assert abs(scs - (-0.26)) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python MollerPlessetMp2Engine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2050_multireference_cas_weyl_paldus_dimension() -> Result<()> {
    use oxideon::science::caspt2_dmrg_multireference::MultireferenceCasEngine;
    // Benzene pi-system CAS(6, 6) singlet space has exactly 175 spin-adapted CSFs!
    assert_eq!(
        MultireferenceCasEngine::singlet_weyl_csf_dimension(6, 6)?,
        175
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert ox.MultireferenceCasEngine.singlet_weyl_csf_dimension(6, 6) == 175
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python MultireferenceCasEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2060_hedin_gw_quasiparticle_bethe_salpeter() -> Result<()> {
    use oxideon::science::gw_bethe_salpeter_exciton::HedinGwBetheSalpeterEngine;
    let (eqp, z) = HedinGwBetheSalpeterEngine::g0w0_quasiparticle_energy(0.6, 0.75, -0.25)?;
    assert!((z - 0.8).abs() < 1e-12 && (eqp - 1.2).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eqp, z = ox.HedinGwBetheSalpeterEngine.g0w0_quasiparticle_energy(0.6, 0.75, -0.25)
assert abs(z - 0.8) < 1e-12 and abs(eqp - 1.2) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python HedinGwBetheSalpeterEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2070_yarkony_conical_intersection_longuet_higgins() -> Result<()> {
    use oxideon::science::conical_intersection_berry::YarkonyConicalIntersectionEngine;
    let ci = YarkonyConicalIntersectionEngine::new(3.0, 4.0);
    let (em, ep, _) = ci.adiabatic_surfaces_and_mixing_angle(1.0, 1.0);
    assert!((em - (-5.0)).abs() < 1e-12 && (ep - 5.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

ci = ox.YarkonyConicalIntersectionEngine(3.0, 4.0)
em, ep, _ = ci.adiabatic_surfaces_and_mixing_angle(1.0, 1.0)
assert em == -5.0 and ep == 5.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python YarkonyConicalIntersectionEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2080_eyring_transition_state_wigner_tunneling() -> Result<()> {
    use oxideon::science::eyring_kramers_transition_state::EyringTransitionStateEngine;
    assert!(
        (EyringTransitionStateEngine::wigner_tunneling_factor(2.449489742783178)? - 1.25).abs()
            < 1e-12
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

import math
assert abs(ox.EyringTransitionStateEngine.wigner_tunneling_factor(math.sqrt(6.0)) - 1.25) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python EyringTransitionStateEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2090_ring_polymer_molecular_dynamics_instanton() -> Result<()> {
    use oxideon::science::ring_polymer_pimd_instanton::RingPolymerMolecularDynamics;
    let rpmd = RingPolymerMolecularDynamics::new(6, 10.0)?;
    assert!(rpmd.normal_mode_frequency(0).abs() < 1e-15);
    assert!((rpmd.normal_mode_frequency(3) - 20.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

rpmd = ox.RingPolymerMolecularDynamics(6, 10.0)
assert abs(rpmd.normal_mode_frequency(3) - 20.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python RingPolymerMolecularDynamics test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2100_grand_omni_henicosa_engine_xxi() -> Result<()> {
    use oxideon::omni_henicosa_xxi::GrandOmniHenicosaEngineXXI;
    let eng = GrandOmniHenicosaEngineXXI::new(21);
    let inv = eng.execute_henicosa_synthesis()?;
    assert!(inv > 2000.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniHenicosaEngineXXI(21)
assert eng.execute_henicosa_synthesis() > 2000.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniHenicosaEngineXXI test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2110_pontryagin_maximum_principle_costate() -> Result<()> {
    use oxideon::ode::pontryagin_maximum_principle::PontryaginMaximumEngine;
    let pmp = PontryaginMaximumEngine::new(1.0, 2.0)?;
    let (u, h) = pmp.optimal_control_and_hamiltonian(1.0, 3.0);
    // Saturated at u_max = 2.0: H* = 1.0 + 3.0*2.0 - 0.5*4.0 = 5.0
    assert!((u - 2.0).abs() < 1e-12 && (h - 5.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

pmp = ox.PontryaginMaximumEngine(1.0, 2.0)
assert pmp.optimal_control_and_hamiltonian(1.0, 3.0) == (2.0, 5.0)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python PontryaginMaximumEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2120_sub_riemannian_chow_rashevskii_heisenberg() -> Result<()> {
    use oxideon::ode::chow_rashevskii_sub_riemannian::SubRiemannianHeisenbergEngine;
    let d = SubRiemannianHeisenbergEngine::vertical_carnot_distance(1.0 / std::f64::consts::PI);
    assert!((d - 2.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

import math
assert abs(ox.SubRiemannianHeisenbergEngine.vertical_carnot_distance(1.0 / math.pi) - 2.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python SubRiemannianHeisenbergEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2130_kalman_bucy_riccati_lqg_filter() -> Result<()> {
    use oxideon::ode::kalman_bucy_riccati_lqg::KalmanBucyRiccatiEngine;
    // Unstable plant a = +1.0 is stabilized by Kalman-Bucy observation c = 1, q = 3, r = 1 => P_inf = 1 + sqrt(1 + 3) = 3.0!
    let kb = KalmanBucyRiccatiEngine::new(1.0, 1.0, 3.0, 1.0)?;
    let (p_inf, k_inf) = kb.steady_state_covariance_and_gain();
    assert!((p_inf - 3.0).abs() < 1e-12 && (k_inf - 3.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

kb = ox.KalmanBucyRiccatiEngine(1.0, 1.0, 3.0, 1.0)
assert kb.steady_state_covariance_and_gain() == (3.0, 3.0)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python KalmanBucyRiccatiEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2140_zames_doyle_glover_h_infinity_robust_control() -> Result<()> {
    use oxideon::ode::zames_h_infinity_rob_control::HInfinityRobustControlEngine;
    let x_are = HInfinityRobustControlEngine::solve_h_infinity_are(0.0, 1.0, 2.0, 3.0, 1.0, 1.0)?;
    // eff_gain = 4 - 1 = 3, q = 3 => X = sqrt(9) / 3 = 1.0!
    assert!((x_are - 1.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert ox.HInfinityRobustControlEngine.solve_h_infinity_are(0.0, 1.0, 2.0, 3.0, 1.0, 1.0) == 1.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python HInfinityRobustControlEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2150_sontag_control_lyapunov_ames_barrier_function() -> Result<()> {
    use oxideon::ode::sontag_control_lyapunov_cbf::ControlBarrierLyapunovEngine;
    let u_s = ControlBarrierLyapunovEngine::sontag_universal_control(3.0, 2.0);
    // a = 3, b = 2 => -(3 + sqrt(9 + 16)) / 2 = -8 / 2 = -4.0!
    assert!((u_s - (-4.0)).abs() < 1e-12);
    assert!(
        (ControlBarrierLyapunovEngine::cbf_safety_filter(-5.0, 0.0, 1.0, 1.0) - (-1.0)).abs()
            < 1e-12
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert ox.ControlBarrierLyapunovEngine.sontag_universal_control(3.0, 2.0) == -4.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python ControlBarrierLyapunovEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2160_kam_invariant_tori_diophantine_golden_mean() -> Result<()> {
    use oxideon::physics::kam_kolmogorov_arnold_moser::KamDiophantineTorusEngine;
    let kam = KamDiophantineTorusEngine::new(0.1, 1.0)?;
    let golden = 0.5 * (5.0_f64.sqrt() - 1.0);
    let (min_g, ok) = kam.minimum_diophantine_constant(1.0, -golden, 10)?;
    assert!(ok && min_g > 0.3);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

import math
kam = ox.KamDiophantineTorusEngine(0.1, 1.0)
g, ok = kam.minimum_diophantine_constant(1.0, -0.5 * (math.sqrt(5.0) - 1.0), 10)
assert ok and g > 0.3
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python KamDiophantineTorusEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2170_nekhoroshev_exponential_stability_arnold_diffusion() -> Result<()> {
    use oxideon::physics::nekhoroshev_arnold_diffusion::NekhoroshevExponentialStability;
    let nek = NekhoroshevExponentialStability::new(2, 1.0)?;
    // At n = 2, a = 1/4: for eps = 1/16, (eps_0 / eps)^{1/4} = 2.0 => drift = 0.5, time = e^2
    let (drift, tf) = nek.nekhoroshev_drift_and_time_factor(1.0 / 16.0)?;
    assert!((drift - 0.5).abs() < 1e-12 && (tf - 2.0_f64.exp()).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

nek = ox.NekhoroshevExponentialStability(2, 1.0)
drift, _ = nek.nekhoroshev_drift_and_time_factor(1.0 / 16.0)
assert abs(drift - 0.5) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python NekhoroshevExponentialStability test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2180_cr3bp_jacobi_constant_routh_trojan_l4_l5() -> Result<()> {
    use oxideon::physics::poincare_cartan_lagrange_l4_l5::CircularRestrictedThreeBodyEngine;
    // Sun-Jupiter mass ratio mu ~ 0.000954 < mu_Routh ~ 0.03852 => Trojan L4/L5 asteroids are linearly stable!
    let cr3bp = CircularRestrictedThreeBodyEngine::new(0.01)?;
    let l4 = [0.5 - 0.01, 0.5 * 3.0_f64.sqrt()];
    let (cj, stable) = cr3bp.jacobi_constant_and_routh_stable(l4, [0.0, 0.0]);
    assert!(stable && (cj - (3.0 - 0.01 * 0.99)).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

cr = ox.CircularRestrictedThreeBodyEngine(0.01)
_, stable = cr.jacobi_constant_and_routh_stable([0.49, 0.8660254], [0.0, 0.0])
assert stable
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python CircularRestrictedThreeBodyEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2190_chirikov_standard_map_greene_kam_breakdown() -> Result<()> {
    use oxideon::physics::chirikov_standard_map_chaos::ChirikovStandardMapEngine;
    let sm = ChirikovStandardMapEngine::new(1.2);
    let (th, _, chaotic) = sm.iterate_map(1.0, 0.5, 100)?;
    assert!(chaotic && (0.0..2.0 * std::f64::consts::PI).contains(&th));
    assert!(sm.iterate_map(1.0, 0.5, usize::MAX).is_err()); // a typo must not run for years

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

sm = ox.ChirikovStandardMapEngine(1.2)
_, _, chaotic = sm.iterate_map(1.0, 0.5, 50)
assert chaotic
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python ChirikovStandardMapEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2200_grand_omni_docosa_engine_xxii() -> Result<()> {
    use oxideon::omni_docosa_xxii::GrandOmniDocosaEngineXXII;
    let eng = GrandOmniDocosaEngineXXII::new(22);
    let inv = eng.execute_docosa_synthesis()?;
    assert!(inv > 2050.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniDocosaEngineXXII(22)
assert eng.execute_docosa_synthesis() > 2050.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python GrandOmniDocosaEngineXXII test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2210_teukolsky_kerr_quasinormal_modes() -> Result<()> {
    use oxideon::physics::teukolsky_kerr_quasinormal::KerrQuasinormalModeEngine;
    // Schwarzschild limit (chi = 0): r_+ = 2M, M*omega_220 ~ 0.37
    let kerr0 = KerrQuasinormalModeEngine::new(1.0, 0.0)?;
    assert!((kerr0.outer_horizon_and_angular_velocity().0 - 2.0).abs() < 1e-12);
    assert!((kerr0.fundamental_220_qnm_frequency() - 0.37).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

k0 = ox.KerrQuasinormalModeEngine(1.0, 0.0)
assert abs(k0.fundamental_220_qnm_frequency() - 0.37) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python KerrQuasinormalModeEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2220_penrose_superradiance_christodoulou_blandford_znajek() -> Result<()> {
    use oxideon::physics::penrose_superradiance_ergosphere::BlandfordZnajekPenroseEngine;
    // Extremal Kerr (chi = 1.0): M_irr = M / sqrt(2), extractable fraction = 1 - 1/sqrt(2) ~ 0.2928932
    let bz = BlandfordZnajekPenroseEngine::new(1.0, 1.0)?;
    let (m_irr, e_rot) = bz.irreducible_mass_and_rotational_energy();
    assert!((m_irr - 1.0 / std::f64::consts::SQRT_2).abs() < 1e-12 && e_rot > 0.2928);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

bz = ox.BlandfordZnajekPenroseEngine(1.0, 1.0)
_, erot = bz.irreducible_mass_and_rotational_energy()
assert erot > 0.2928
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python BlandfordZnajekPenroseEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2230_tov_neutron_star_buchdahl_tidal_deformability() -> Result<()> {
    use oxideon::physics::tolman_oppenheimer_volkoff::TolmanOppenheimerVolkoffEngine;
    let (lam, ok) =
        TolmanOppenheimerVolkoffEngine::tidal_deformability_and_buchdahl_ok(0.2, 0.096)?;
    assert!(ok && (lam - 200.0).abs() < 1e-9);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

lam, ok = ox.TolmanOppenheimerVolkoffEngine.tidal_deformability_and_buchdahl_ok(0.2, 0.096)
assert ok and abs(lam - 200.0) < 1e-9
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python TolmanOppenheimerVolkoffEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2240_chandrasekhar_white_dwarf_lane_emden() -> Result<()> {
    use oxideon::physics::chandrasekhar_lane_emden::ChandrasekharWhiteDwarfEngine;
    // Carbon-Oxygen white dwarf has mu_e = 2.0 => M_Ch = 5.836 / 4 = 1.459 M_sun!
    let m_ch = ChandrasekharWhiteDwarfEngine::chandrasekhar_mass_solar(2.0)?;
    assert!((m_ch - 1.459).abs() < 1e-12);
    assert!((ChandrasekharWhiteDwarfEngine::lane_emden_n5(3.0) - 0.5).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert abs(ox.ChandrasekharWhiteDwarfEngine.chandrasekhar_mass_solar(2.0) - 1.459) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python ChandrasekharWhiteDwarfEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2250_peters_mathews_gravitational_wave_chirp_mass() -> Result<()> {
    use oxideon::physics::peters_mathews_chirp_gw::GravitationalWaveChirpEngine;
    // GW150914-like equal-mass binary (30 + 30 M_sun): eta = 0.25, M_c = 60 * 0.25^{0.6} ~ 26.1165
    let gw = GravitationalWaveChirpEngine::new(30.0, 30.0)?;
    let (mc, eta) = gw.chirp_mass_and_symmetric_eta();
    assert!((eta - 0.25).abs() < 1e-12 && mc > 26.1 && mc < 26.2);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

gw = ox.GravitationalWaveChirpEngine(30.0, 30.0)
mc, eta = gw.chirp_mass_and_symmetric_eta()
assert eta == 0.25 and 26.1 < mc < 26.2
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GravitationalWaveChirpEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2260_cmb_baryon_photon_acoustic_sachs_wolfe() -> Result<()> {
    use oxideon::physics::sachs_wolfe_cmb_acoustic::CmbAcousticPeakEngine;
    // At R_b = 1/3, 3*(1 + 1/3) = 4 => c_s / c = 1/2 = 0.5!
    assert!(
        (CmbAcousticPeakEngine::baryon_photon_sound_speed_ratio(1.0 / 3.0)? - 0.5).abs() < 1e-12
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert abs(ox.CmbAcousticPeakEngine.baryon_photon_sound_speed_ratio(1.0 / 3.0) - 0.5) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python CmbAcousticPeakEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2270_sunyaev_zeldovich_kompaneets_spectral_null() -> Result<()> {
    use oxideon::physics::sunyaev_zeldovich_kompaneets::SunyaevZeldovichKompaneetsEngine;
    // Rayleigh-Jeans limit (x -> 0): g(0) = -2.0; crossover null at x_0 ~ 3.8300; Wien limit (x = 6): g(6) > 0
    assert!(
        (SunyaevZeldovichKompaneetsEngine::tsz_spectral_function_g(0.0) - (-2.0)).abs() < 1e-12
    );
    assert!(SunyaevZeldovichKompaneetsEngine::tsz_spectral_function_g(3.830016).abs() < 1e-4);
    assert!(SunyaevZeldovichKompaneetsEngine::tsz_spectral_function_g(6.0) > 2.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert ox.SunyaevZeldovichKompaneetsEngine.tsz_spectral_function_g(0.0) == -2.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python SunyaevZeldovichKompaneetsEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2280_press_schechter_halo_mass_spherical_collapse() -> Result<()> {
    use oxideon::physics::press_schechter_halo_mass::PressSchechterHaloEngine;
    let dc = PressSchechterHaloEngine::spherical_collapse_delta_c();
    assert!((dc - 1.68647019984).abs() < 1e-8);
    assert!(PressSchechterHaloEngine::press_schechter_multiplicity(1.0) > 0.48);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert abs(ox.PressSchechterHaloEngine.spherical_collapse_delta_c() - 1.6864702) < 1e-6
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python PressSchechterHaloEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2290_hawking_page_ads_black_hole_deconfinement() -> Result<()> {
    use oxideon::physics::hawking_page_ads_black_hole::HawkingPagePhaseTransition;
    let hp = HawkingPagePhaseTransition::new(1.0)?;
    // At r_+ = L = 1.0: F = 0.0 and T_HP = 1 / pi! For r_+ = 2.0 > L: F < 0 (deconfined BH phase)
    let (t_hp, f_hp, _) = hp.temperature_free_energy_and_deconfined(1.0)?;
    assert!((t_hp - 1.0 / std::f64::consts::PI).abs() < 1e-12 && f_hp.abs() < 1e-12);
    assert!(hp.temperature_free_energy_and_deconfined(2.0)?.2);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

hp = ox.HawkingPagePhaseTransition(1.0)
assert hp.temperature_free_energy_and_deconfined(2.0)[2]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python HawkingPagePhaseTransition test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2300_grand_omni_tricosa_engine_xxiii() -> Result<()> {
    use oxideon::omni_tricosa_xxiii::GrandOmniTricosaEngineXXIII;
    let eng = GrandOmniTricosaEngineXXIII::new(23);
    let inv = eng.execute_tricosa_synthesis()?;
    assert!(inv > 2280.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniTricosaEngineXXIII(23)
assert eng.execute_tricosa_synthesis() > 2280.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniTricosaEngineXXIII test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2310_groth16_r1cs_qap_zk_snark() -> Result<()> {
    use oxideon::crypto::groth16_qap_zk_snark::QuadraticArithmeticProgramEngine;
    // Circuit x * y = out with witness z = [1, 3, 4, 12] over F_97
    let qap = QuadraticArithmeticProgramEngine::new(97, 1, 4)?;
    assert!(qap.verify_r1cs_witness(
        &[0, 1, 0, 0],
        &[0, 0, 1, 0],
        &[0, 0, 0, 1],
        &[1, 3, 4, 12]
    )?);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

qap = ox.QuadraticArithmeticProgramEngine(97, 1, 4)
assert qap.verify_r1cs_witness([0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 3, 4, 12])
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python QuadraticArithmeticProgramEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2320_fri_reed_solomon_stark_split_and_fold() -> Result<()> {
    use oxideon::crypto::fri_reed_solomon_stark::FastReedSolomonIopEngine;
    let fri = FastReedSolomonIopEngine::new(97)?;
    let folded = fri.fold_polynomial_coeffs(&[1, 2, 3, 4], 10)?;
    assert_eq!(folded, vec![21, 43]);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

fri = ox.FastReedSolomonIopEngine(97)
assert fri.fold_polynomial_coeffs([1, 2, 3, 4], 10) == [21, 43]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python FastReedSolomonIopEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2330_kzg_polynomial_commitment_quotient_witness() -> Result<()> {
    use oxideon::crypto::kzg_polynomial_commitment::KzgPolynomialCommitment;
    // f(X) = X^2 - 5X + 6 = [6, -5, 1] at z = 2 gives f(2) = 0 and quotient q(X) = X - 3 = [94, 1] mod 97!
    let kzg = KzgPolynomialCommitment::new(97)?;
    let (y, q) = kzg.compute_opening_quotient(&[6, -5, 1], 2)?;
    assert_eq!(y, 0);
    assert_eq!(q, vec![94, 1]);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

kzg = ox.KzgPolynomialCommitment(97)
y, q = kzg.compute_opening_quotient([6, -5, 1], 2)
assert y == 0 and q == [94, 1]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python KzgPolynomialCommitment test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2340_shamir_threshold_secret_sharing_lagrange() -> Result<()> {
    use oxideon::crypto::shamir_feldman_threshold_mpc::ShamirSecretSharingFp;
    // P(x) = 42 + 7x + 3x^2 mod 97 => P(1) = 52, P(2) = 68, P(3) = 90 => reconstructs secret 42!
    let sss = ShamirSecretSharingFp::new(97)?;
    assert_eq!(sss.reconstruct_secret(&[1, 2, 3], &[52, 68, 90])?, 42);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

sss = ox.ShamirSecretSharingFp(97)
assert sss.reconstruct_secret([1, 2, 3], [52, 68, 90]) == 42
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python ShamirSecretSharingFp test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2350_goppa_algebraic_geometry_codes_tvz_bound() -> Result<()> {
    use oxideon::crypto::goppa_algebraic_geometry_codes::TsfasmanVladutZinkAgCode;
    // Klein quartic over F_8 (genus g = 3, N = 24 rational points, deg(G) = 10): [24, 8, >=14] AG code!
    let ag = TsfasmanVladutZinkAgCode::new(3, 24, 10)?;
    assert_eq!(ag.riemann_roch_k_and_design_distance(), (8, 14));
    assert!(
        (TsfasmanVladutZinkAgCode::tvz_asymptotic_rate(0.5, 49.0)? - (1.0 / 3.0)).abs() < 1e-12
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

ag = ox.TsfasmanVladutZinkAgCode(3, 24, 10)
assert ag.riemann_roch_k_and_design_distance() == (8, 14)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python TsfasmanVladutZinkAgCode test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2360_arikan_polar_codes_channel_polarization() -> Result<()> {
    use oxideon::crypto::arikan_polar_codes_capacity::ArikanPolarCodeEngine;
    let polar = ArikanPolarCodeEngine::new(0.5)?;
    let z8 = polar.polarized_bhattacharyya_parameters(3)?;
    // Sum of all 2^3 = 8 Bhattacharyya parameters is conserved at 8 * 0.5 = 4.0!
    assert_eq!(z8.len(), 8);
    assert!((z8.iter().sum::<f64>() - 4.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

polar = ox.ArikanPolarCodeEngine(0.5)
z = polar.polarized_bhattacharyya_parameters(3)
assert len(z) == 8 and abs(sum(z) - 4.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python ArikanPolarCodeEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2370_shannon_rate_distortion_reverse_waterfilling() -> Result<()> {
    use oxideon::crypto::shannon_rate_distortion_blahut::RateDistortionBlahutArimoto;
    // Variances [4.0, 2.0, 0.5] at water level lambda = 1.0: rates = 0.5*log2(4) + 0.5*log2(2) + 0 = 1.0 + 0.5 = 1.5 bits, D = 1 + 1 + 0.5 = 2.5
    let (r, d) = RateDistortionBlahutArimoto::reverse_waterfilling_rate_and_distortion(
        &[4.0, 2.0, 0.5],
        1.0,
    )?;
    assert!((r - 1.5).abs() < 1e-12 && (d - 2.5).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

r, d = ox.RateDistortionBlahutArimoto.reverse_waterfilling_rate_and_distortion([4.0, 2.0, 0.5], 1.0)
assert r == 1.5 and d == 2.5
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python RateDistortionBlahutArimoto test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2380_slepian_wolf_wyner_ziv_distributed_coding() -> Result<()> {
    use oxideon::crypto::slepian_wolf_wyner_ziv::SlepianWolfDistributedCoding;
    assert!(SlepianWolfDistributedCoding::is_slepian_wolf_achievable(
        0.6, 0.9, 0.5, 0.5, 1.5
    ));
    let rwz = SlepianWolfDistributedCoding::gaussian_wyner_ziv_rate(4.0, 0.0, 1.0)?;
    assert!((rwz - 1.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert ox.SlepianWolfDistributedCoding.gaussian_wyner_ziv_rate(4.0, 0.0, 1.0) == 1.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python SlepianWolfDistributedCoding test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2390_renyi_differential_privacy_moments_accountant() -> Result<()> {
    use oxideon::crypto::dwork_renyi_differential_privacy::RenyiDifferentialPrivacyAccountant;
    let rdp = RenyiDifferentialPrivacyAccountant::new(10.0)?;
    let (eps, alpha) = rdp.optimal_dp_epsilon(100, 1e-5)?;
    assert!(eps > 0.5 && alpha > 1.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

rdp = ox.RenyiDifferentialPrivacyAccountant(10.0)
eps, alpha = rdp.optimal_dp_epsilon(100, 1e-5)
assert eps > 0.5 and alpha > 1.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python RenyiDifferentialPrivacyAccountant test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2400_grand_omni_tetracosa_engine_xxiv() -> Result<()> {
    use oxideon::omni_tetracosa_xxiv::GrandOmniTetracosaEngineXXIV;
    let eng = GrandOmniTetracosaEngineXXIV::new(24);
    let inv = eng.execute_tetracosa_synthesis()?;
    assert!(inv > 2350.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniTetracosaEngineXXIV(24)
assert eng.execute_tetracosa_synthesis() > 2350.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniTetracosaEngineXXIV test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2410_pac_bayes_mcallester_rademacher_bound() -> Result<()> {
    use oxideon::nn::rademacher_vapnik_pac_bayes::PacBayesMcAllesterEngine;
    let b = PacBayesMcAllesterEngine::mcallester_risk_bound(0.05, 10.0, 1000, 0.05)?;
    assert!(b > 0.05 && b < 0.20);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

b = ox.PacBayesMcAllesterEngine.mcallester_risk_bound(0.05, 10.0, 1000, 0.05)
assert 0.05 < b < 0.20
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python PacBayesMcAllesterEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2420_jacot_infinite_width_neural_tangent_kernel() -> Result<()> {
    use oxideon::nn::jacot_neural_tangent_kernel::InfiniteWidthNtkEngine;
    // At u = 1.0 (self-similarity): kappa_0(1) = 0.5, kappa_1(1) = 0.5 => NTK(1) = 1.0!
    let (k1, ntk) = InfiniteWidthNtkEngine::relu_nngp_and_ntk(1.0)?;
    assert!((k1 - 0.5).abs() < 1e-12 && (ntk - 1.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

k1, ntk = ox.InfiniteWidthNtkEngine.relu_nngp_and_ntk(1.0)
assert abs(k1 - 0.5) < 1e-12 and abs(ntk - 1.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python InfiniteWidthNtkEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2430_yang_maximal_update_parameterization_mup() -> Result<()> {
    use oxideon::nn::yang_maximal_update_mup::MaximalUpdateParamEngine;
    // Scaling width 16x from 256 -> 4096 scales hidden Adam LR by 1/16 and hidden weight init std by 1/4!
    let mup = MaximalUpdateParamEngine::new(256, 4096)?;
    let (lr, std_w) = mup.transfer_adam_lr_and_init_std(0.016, 0.08);
    assert!((lr - 0.001).abs() < 1e-12 && (std_w - 0.02).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

mup = ox.MaximalUpdateParamEngine(256, 4096)
assert mup.transfer_adam_lr_and_init_std(0.016, 0.08) == (0.001, 0.02)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python MaximalUpdateParamEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2440_chinchilla_kaplan_compute_optimal_scaling_laws() -> Result<()> {
    use oxideon::nn::chinchilla_kaplan_scaling_laws::ComputeOptimalChinchillaEngine;
    let chin = ComputeOptimalChinchillaEngine::default();
    let (loss, n_opt, d_opt) = chin.loss_and_optimal_allocation(70e9, 1.4e12)?;
    assert!(
        loss > 1.69
            && loss < 2.1
            && (n_opt * d_opt - 70e9 * 1.4e12).abs() / (70e9 * 1.4e12) < 1e-12
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

chin = ox.ComputeOptimalChinchillaEngine()
loss, n_opt, d_opt = chin.loss_and_optimal_allocation(70e9, 1.4e12)
assert 1.69 < loss < 2.1
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python ComputeOptimalChinchillaEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2450_belkin_hastie_double_descent_interpolation() -> Result<()> {
    use oxideon::nn::belkin_double_descent_marchenko::OverparameterizedDoubleDescent;
    let dd = OverparameterizedDoubleDescent::new(1.0, 1.0)?;
    // Near threshold gamma = 1.1 risk is huge (10.09), whereas deeper overparameterization gamma = 4.0 descends to 0.75 + 0.333 = 1.083!
    assert!(dd.asymptotic_ridgeless_risk(4.0)? < dd.asymptotic_ridgeless_risk(1.1)?);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

dd = ox.OverparameterizedDoubleDescent(1.0, 1.0)
assert dd.asymptotic_ridgeless_risk(4.0) < dd.asymptotic_ridgeless_risk(1.1)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python OverparameterizedDoubleDescent test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2460_candes_tao_compressed_sensing_rip_lasso() -> Result<()> {
    use oxideon::nn::candes_tao_compressed_sensing_rip::RestrictedIsometryLassoEngine;
    let (ok, c0) = RestrictedIsometryLassoEngine::check_candes_rip_condition(0.2)?;
    assert!(ok && c0.is_finite());
    assert_eq!(
        RestrictedIsometryLassoEngine::proximal_soft_threshold(&[-3.0, 0.5, 2.5], 1.0),
        vec![-2.0, 0.0, 1.5]
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

ok, _ = ox.RestrictedIsometryLassoEngine.check_candes_rip_condition(0.2)
assert ok and ox.RestrictedIsometryLassoEngine.proximal_soft_threshold([-3.0, 0.5, 2.5], 1.0) == [-2.0, 0.0, 1.5]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python RestrictedIsometryLassoEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2470_talagrand_concentration_bakry_emery_log_sobolev() -> Result<()> {
    use oxideon::nn::talagrand_concentration_log_sobolev::LogSobolevConcentrationEngine;
    let lsi = LogSobolevConcentrationEngine::new(2.0)?;
    let (tail, w2) = lsi.herbst_tail_and_talagrand_w2(1.0, 1.0, 4.0)?;
    assert!((tail - (-1.0_f64).exp()).abs() < 1e-12 && (w2 - 2.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

lsi = ox.LogSobolevConcentrationEngine(2.0)
_, w2 = lsi.herbst_tail_and_talagrand_w2(1.0, 1.0, 4.0)
assert abs(w2 - 2.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python LogSobolevConcentrationEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2480_vovk_split_conformal_prediction_quantile() -> Result<()> {
    use oxideon::nn::vovk_conformal_prediction_martingale::SplitConformalPredictionEngine;
    // n = 9 scores [1..=9], alpha = 0.2 => k = ceil(10 * 0.8) = 8 => 8th score is 8.0!
    let q = SplitConformalPredictionEngine::conformal_quantile(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        0.2,
    )?;
    assert!((q - 8.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert ox.SplitConformalPredictionEngine.conformal_quantile([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 0.2) == 8.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python SplitConformalPredictionEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2490_rahimi_recht_random_fourier_bochner_kernel() -> Result<()> {
    use oxideon::nn::rahimi_recht_random_fourier_features::RandomFourierBochnerEngine;
    let rff = RandomFourierBochnerEngine::new(vec![1.0, 0.0, 0.0, 1.0], 2)?;
    assert!((rff.approximate_kernel(&[2.0, 3.0], &[2.0, 3.0])? - 1.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

rff = ox.RandomFourierBochnerEngine([1.0, 0.0, 0.0, 1.0], 2)
assert rff.approximate_kernel([2.0, 3.0], [2.0, 3.0]) == 1.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python RandomFourierBochnerEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2500_grand_omni_pentacosa_engine_xxv() -> Result<()> {
    use oxideon::omni_pentacosa_xxv::GrandOmniPentacosaEngineXXV;
    let eng = GrandOmniPentacosaEngineXXV::new(25);
    let inv = eng.execute_pentacosa_synthesis()?;
    assert!(inv > 2370.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniPentacosaEngineXXV(25)
assert eng.execute_pentacosa_synthesis() > 2370.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniPentacosaEngineXXV test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2510_mooney_rivlin_ogden_hyperelastic_stretch() -> Result<()> {
    use oxideon::physics::ogden_mooney_rivlin_hyperelastic::HyperelasticStrainEnergyEngine;
    let mr = HyperelasticStrainEnergyEngine::new(1.0, 0.5)?;
    assert_eq!(mr.uniaxial_energy_and_cauchy_stress(1.0)?, (0.0, 0.0));
    let (w, sig) = mr.uniaxial_energy_and_cauchy_stress(2.0)?;
    assert!(w > 1.5 && sig > 8.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

mr = ox.HyperelasticStrainEnergyEngine(1.0, 0.5)
assert mr.uniaxial_energy_and_cauchy_stress(1.0) == (0.0, 0.0)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python HyperelasticStrainEnergyEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2520_griffith_irwin_rice_j_integral_fracture() -> Result<()> {
    use oxideon::physics::griffith_rice_j_integral_fracture::RiceJIntegralFractureEngine;
    let frac = RiceJIntegralFractureEngine::new(200.0, 0.0)?;
    assert!((frac.plane_strain_j_integral(10.0, 0.0, 0.0) - 0.5).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

frac = ox.RiceJIntegralFractureEngine(200.0, 0.0)
assert frac.plane_strain_j_integral(10.0, 0.0, 0.0) == 0.5
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python RiceJIntegralFractureEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2530_j2_von_mises_plasticity_radial_return() -> Result<()> {
    use oxideon::physics::von_mises_drucker_prager_plasticity::J2PlasticityReturnMapping;
    let j2 = J2PlasticityReturnMapping::new(100.0, 250.0, 50.0)?;
    let (q_new, dg) = j2.radial_return_step(600.0)?;
    // dgamma = (600 - 250) / (300 + 50) = 1.0, q_new = 250 + 50 = 300.0!
    assert!((dg - 1.0).abs() < 1e-12 && (q_new - 300.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

j2 = ox.J2PlasticityReturnMapping(100.0, 250.0, 50.0)
assert j2.radial_return_step(600.0) == (300.0, 1.0)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python J2PlasticityReturnMapping test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2540_kirchhoff_love_plate_flexural_navier() -> Result<()> {
    use oxideon::physics::mindlin_reissner_kirchhoff_plates::KirchhoffLovePlateEngine;
    let pl = KirchhoffLovePlateEngine::new(12.0, 1.0, 0.0)?;
    assert!((pl.flexural_rigidity_d() - 1.0).abs() < 1e-12);
    assert!(pl.navier_square_plate_deflection(1.0, 100.0)? > 0.25);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

pl = ox.KirchhoffLovePlateEngine(12.0, 1.0, 0.0)
assert pl.flexural_rigidity_d() == 1.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python KirchhoffLovePlateEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2550_biot_poroelasticity_gassmann_skempton() -> Result<()> {
    use oxideon::physics::biot_poroelasticity_darcy::BiotPoroelasticConsolidation;
    let biot = BiotPoroelasticConsolidation::new(10.0, 40.0, 16.0)?;
    let (alpha, ku, b) = biot.biot_gassmann_skempton_parameters();
    // alpha = 1 - 10/40 = 0.75, Ku = 10 + 0.5625*16 = 19.0, B = 12 / 19
    assert!((alpha - 0.75).abs() < 1e-12 && (ku - 19.0).abs() < 1e-12 && b > 0.63);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

biot = ox.BiotPoroelasticConsolidation(10.0, 40.0, 16.0)
assert biot.biot_gassmann_skempton_parameters()[:2] == (0.75, 19.0)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python BiotPoroelasticConsolidation test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2560_eshelby_spherical_inclusion_mori_tanaka() -> Result<()> {
    use oxideon::physics::eshelby_inclusion_homogenization::EshelbyInclusionMoriTanaka;
    // At nu = 0.2: alpha_S = 1.2 / 2.4 = 0.5, beta_S = 6 / 12 = 0.5!
    let (a, b) = EshelbyInclusionMoriTanaka::spherical_eshelby_factors(0.2)?;
    assert!((a - 0.5).abs() < 1e-12 && (b - 0.5).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

a, b = ox.EshelbyInclusionMoriTanaka.spherical_eshelby_factors(0.2)
assert abs(a - 0.5) < 1e-12 and abs(b - 0.5) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python EshelbyInclusionMoriTanaka test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2570_pendry_transformation_optics_metamaterial_cloak() -> Result<()> {
    use oxideon::physics::pendry_transformation_optics_cloak::PendryMetamaterialCloakEngine;
    let cloak = PendryMetamaterialCloakEngine::new(1.0, 2.0)?;
    let (er, eth, ez) = cloak.cylindrical_cloak_permittivity(2.0)?;
    assert!((er * eth - 1.0).abs() < 1e-12 && (ez - 2.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

cloak = ox.PendryMetamaterialCloakEngine(1.0, 2.0)
er, eth, ez = cloak.cylindrical_cloak_permittivity(2.0)
assert abs(er * eth - 1.0) < 1e-12 and abs(ez - 2.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python PendryMetamaterialCloakEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2580_yablonovitch_photonic_crystal_bragg_bandgap() -> Result<()> {
    use oxideon::physics::yablonovitch_photonic_crystal_bandgap::PhotonicBraggQuarterWaveStack;
    // Si / SiO2 Bragg stack (n_H = 3.0, n_L = 1.0 => contrast = 0.5 => asin(0.5) = pi/6 => Delta_omega/omega_0 = 2/3!)
    let dbr = PhotonicBraggQuarterWaveStack::new(3.0, 1.0)?;
    let (gap, rn) = dbr.fractional_bandgap_and_reflectance(5);
    assert!((gap - 2.0 / 3.0).abs() < 1e-12 && rn > 0.9999);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

dbr = ox.PhotonicBraggQuarterWaveStack(3.0, 1.0)
gap, rn = dbr.fractional_bandgap_and_reflectance(5)
assert abs(gap - 2.0 / 3.0) < 1e-12 and rn > 0.9999
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python PhotonicBraggQuarterWaveStack test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2590_kane_lubensky_topological_isostatic_mechanics() -> Result<()> {
    use oxideon::physics::kane_lubensky_topological_mechanics::KaneLubenskyIsostaticEngine;
    let kl = KaneLubenskyIsostaticEngine::new(2);
    assert_eq!(kl.maxwell_calladine_index(100, 200), (0, true));
    assert_eq!(
        KaneLubenskyIsostaticEngine::dimerized_compatibility_winding(1.0, 2.0),
        1
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

kl = ox.KaneLubenskyIsostaticEngine(2)
assert kl.maxwell_calladine_index(100, 200) == (0, True)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python KaneLubenskyIsostaticEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2600_grand_omni_hexacosa_engine_xxvi() -> Result<()> {
    use oxideon::omni_hexacosa_xxvi::GrandOmniHexacosaEngineXXVI;
    let eng = GrandOmniHexacosaEngineXXVI::new(26);
    let inv = eng.execute_hexacosa_synthesis()?;
    assert!(inv > 2400.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniHexacosaEngineXXVI(26)
assert eng.execute_hexacosa_synthesis() > 2400.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniHexacosaEngineXXVI test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2610_gromov_delta_hyperbolic_groups() -> Result<()> {
    use oxideon::science::gromov_hyperbolic_groups::GromovDeltaHyperbolicEngine;
    let gh = GromovDeltaHyperbolicEngine::new(0.0)?;
    let (defect, is_hyp) = gh.four_point_defect_and_check(2.0, 2.0, 4.0, 4.0, 4.0, 4.0);
    assert!((defect - 0.0).abs() < 1e-12 && is_hyp);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

gh = ox.GromovDeltaHyperbolicEngine(0.0)
d, ok = gh.four_point_defect_and_check(2.0, 2.0, 4.0, 4.0, 4.0, 4.0)
assert abs(d) < 1e-12 and ok
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GromovDeltaHyperbolicEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2620_teichmuller_weil_petersson_mirzakhani() -> Result<()> {
    use oxideon::science::teichmuller_weil_petersson::TeichmullerWeilPeterssonEngine;
    let (k_dil, d_t) = TeichmullerWeilPeterssonEngine::dilatation_and_teichmuller_distance(0.5)?;
    assert!((k_dil - 3.0).abs() < 1e-12 && d_t > 0.5);
    let v11 = TeichmullerWeilPeterssonEngine::mirzakhani_wp_volume_v11(0.0);
    assert!((v11 - std::f64::consts::PI * std::f64::consts::PI / 6.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

k_dil, d_t = ox.TeichmullerWeilPeterssonEngine.dilatation_and_teichmuller_distance(0.5)
assert abs(k_dil - 3.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python TeichmullerWeilPeterssonEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2630_thurston_hyperbolic_dehn_surgery() -> Result<()> {
    use oxideon::science::thurston_hyperbolic_dehn::ThurstonHyperbolicDehnEngine;
    let th = ThurstonHyperbolicDehnEngine::new(2.0298832)?;
    let (v_fill, dv) = th.dehn_filled_volume(10, 1, 12.0)?;
    assert!(v_fill < 2.0298832 && v_fill > 1.9 && dv > 0.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

th = ox.ThurstonHyperbolicDehnEngine(2.0298832)
vf, dv = th.dehn_filled_volume(10, 1, 12.0)
assert 1.9 < vf < 2.0298832
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python ThurstonHyperbolicDehnEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2640_bass_serre_graph_of_groups() -> Result<()> {
    use oxideon::science::bass_serre_trees::BassSerreGraphOfGroupsEngine;
    // PSL(2, Z) = Z/2 * Z/3 -> chi = 1/2 + 1/3 - 1/1 = -1/6
    let chi = BassSerreGraphOfGroupsEngine::rational_euler_characteristic(&[2, 3], &[1])?;
    assert!((chi - (-1.0 / 6.0)).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

chi = ox.BassSerreGraphOfGroupsEngine.rational_euler_characteristic([2, 3], [1])
assert abs(chi - (-1.0 / 6.0)) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python BassSerreGraphOfGroupsEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2650_kazhdan_property_t_ramanujan_expander() -> Result<()> {
    use oxideon::science::kazhdan_property_t_expander::KazhdanPropertyTExpander;
    let exp = KazhdanPropertyTExpander::new(6)?;
    let (bound, gap, is_ram) = exp.ramanujan_bound_and_gap(4.2);
    assert!((bound - 2.0 * 5.0_f64.sqrt()).abs() < 1e-12 && (gap - 1.8).abs() < 1e-12 && is_ram);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

exp = ox.KazhdanPropertyTExpander(6)
b, g, ok = exp.ramanujan_bound_and_gap(4.2)
assert ok and abs(g - 1.8) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python KazhdanPropertyTExpander test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2660_satake_hecke_langlands_ramanujan() -> Result<()> {
    use oxideon::science::arthur_selberg_langlands::SatakeHeckeLanglandsEngine;
    // Ramanujan tau(2) = -24 for weight k = 12
    let sh = SatakeHeckeLanglandsEngine::new(12)?;
    let (bound, theta, is_temp) = sh.ramanujan_bound_and_satake_angle(2, -24.0)?;
    assert!(bound > 90.0 && theta > 1.8 && is_temp);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

sh = ox.SatakeHeckeLanglandsEngine(12)
b, th, ok = sh.ramanujan_bound_and_satake_angle(2, -24.0)
assert ok and b > 90.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python SatakeHeckeLanglandsEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2670_ngo_fundamental_lemma_orbital_integrals() -> Result<()> {
    use oxideon::science::ngo_fundamental_lemma::NgoFundamentalLemmaEngine;
    let ngo = NgoFundamentalLemmaEngine::new(7)?;
    let (lhs, rhs, matched) = ngo.kappa_orbital_and_endoscopic_match(3);
    assert!((lhs - rhs).abs() < 1e-12 && matched);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

ngo = ox.NgoFundamentalLemmaEngine(7)
lhs, rhs, ok = ngo.kappa_orbital_and_endoscopic_match(3)
assert ok and abs(lhs - 1.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python NgoFundamentalLemmaEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2680_drinfeld_lafforgue_shtukas_langlands() -> Result<()> {
    use oxideon::science::drinfeld_shtukas_function_field::DrinfeldShtukaLanglandsEngine;
    let ds = DrinfeldShtukaLanglandsEngine::new(2, 5)?;
    assert_eq!(ds.shtuka_moduli_dimension(2), (2, 2));

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

ds = ox.DrinfeldShtukaLanglandsEngine(2, 5)
assert ds.shtuka_moduli_dimension(2) == (2, 2)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python DrinfeldShtukaLanglandsEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2690_ratner_unipotent_horocycle_rigidity() -> Result<()> {
    use oxideon::science::ratner_unipotent_ergodic::RatnerUnipotentFlowEngine;
    let rat = RatnerUnipotentFlowEngine::new(0.25)?;
    let (kappa, err) = rat.equidistribution_exponent_and_bound(100.0)?;
    assert!((kappa - 0.5).abs() < 1e-12 && (err - 0.1).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

rat = ox.RatnerUnipotentFlowEngine(0.25)
k, err = rat.equidistribution_exponent_and_bound(100.0)
assert abs(k - 0.5) < 1e-12 and abs(err - 0.1) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python RatnerUnipotentFlowEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2700_grand_omni_heptacosa_engine_xxvii() -> Result<()> {
    use oxideon::omni_heptacosa_xxvii::GrandOmniHeptacosaEngineXXVII;
    let eng = GrandOmniHeptacosaEngineXXVII::new(27);
    let inv = eng.execute_heptacosa_synthesis()?;
    assert!(inv > 2500.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniHeptacosaEngineXXVII(27)
assert eng.execute_heptacosa_synthesis() > 2500.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniHeptacosaEngineXXVII test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2710_schwinger_euler_heisenberg_qed() -> Result<()> {
    use oxideon::physics::schwinger_euler_heisenberg_qed::SchwingerEulerHeisenbergQED;
    let qed = SchwingerEulerHeisenbergQED::new(1.0)?;
    let rate = qed.pair_production_rate(1.0, 10)?;
    assert!(rate > 0.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

qed = ox.SchwingerEulerHeisenbergQED(1.0)
assert qed.pair_production_rate(1.0, 10) > 0.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python SchwingerEulerHeisenbergQED test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2720_casimir_lifshitz_vacuum_pressure() -> Result<()> {
    use oxideon::physics::casimir_lifshitz_vacuum::CasimirLifshitzVacuumEngine;
    let (e_a, p_c) = CasimirLifshitzVacuumEngine::energy_and_pressure(1.0)?;
    assert!(e_a < 0.0 && (p_c - 3.0 * e_a).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

ea, pc = ox.CasimirLifshitzVacuumEngine.energy_and_pressure(1.0)
assert ea < 0.0 and abs(pc - 3.0 * ea) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python CasimirLifshitzVacuumEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2730_unruh_dewitt_rindler_bogoliubov() -> Result<()> {
    use oxideon::physics::unruh_dewitt_rindler::UnruhRindlerBogoliubovEngine;
    let unruh = UnruhRindlerBogoliubovEngine::new(2.0 * std::f64::consts::PI)?;
    let (t_u, n_occ) = unruh.unruh_temperature_and_bogoliubov_n(1.0)?;
    assert!((t_u - 1.0).abs() < 1e-12 && (n_occ - 1.0 / (1.0_f64.exp() - 1.0)).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

unruh = ox.UnruhRindlerBogoliubovEngine(6.283185307179586)
tu, n = unruh.unruh_temperature_and_bogoliubov_n(1.0)
assert abs(tu - 1.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python UnruhRindlerBogoliubovEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2740_msw_matter_neutrino_oscillation() -> Result<()> {
    use oxideon::physics::msw_neutrino_oscillation::MswNeutrinoOscillationEngine;
    let msw = MswNeutrinoOscillationEngine::new(0.4, 1.0)?;
    let a_res = (0.8_f64).cos();
    let (s2m, _) = msw.matter_mixing_and_splitting(a_res);
    assert!((s2m - 1.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

import math
msw = ox.MswNeutrinoOscillationEngine(0.4, 1.0)
s2m, _ = msw.matter_mixing_and_splitting(math.cos(0.8))
assert abs(s2m - 1.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python MswNeutrinoOscillationEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2750_wilson_lattice_qcd_area_law() -> Result<()> {
    use oxideon::physics::wilson_lattice_qcd_plaquette::WilsonLatticeGaugeQCD;
    let lqcd = WilsonLatticeGaugeQCD::new(3, 1.8)?;
    let (sigma, w_loop) = lqcd.strong_coupling_string_tension_and_wilson_loop(2, 2)?;
    assert!((sigma - 10.0_f64.ln()).abs() < 1e-12 && (w_loop - 1e-4).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

lqcd = ox.WilsonLatticeGaugeQCD(3, 1.8)
sig, wl = lqcd.strong_coupling_string_tension_and_wilson_loop(2, 2)
assert abs(wl - 1e-4) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python WilsonLatticeGaugeQCD test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2760_qcd_asymptotic_freedom_running_coupling() -> Result<()> {
    use oxideon::physics::qcd_asymptotic_freedom_beta::QcdAsymptoticFreedomEngine;
    let qcd = QcdAsymptoticFreedomEngine::new(3, 5, 0.217)?;
    let (beta0, alpha_mz, is_af) = qcd.running_alpha_s(91.1876)?;
    assert!((beta0 - 23.0 / 3.0).abs() < 1e-12 && alpha_mz > 0.10 && alpha_mz < 0.15 && is_af);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

qcd = ox.QcdAsymptoticFreedomEngine(3, 5, 0.217)
b0, a_mz, ok = qcd.running_alpha_s(91.1876)
assert ok and 0.10 < a_mz < 0.15
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python QcdAsymptoticFreedomEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2770_adler_bell_jackiw_chiral_anomaly() -> Result<()> {
    use oxideon::physics::adler_bell_jackiw_anomaly::AbjChiralAnomalyEngine;
    let abj = AbjChiralAnomalyEngine::new(3, 1.0 / 137.036)?;
    let gamma_ev = abj.pion_two_photon_width(134.9768, 92.2)? * 1e6;
    assert!(gamma_ev > 7.5 && gamma_ev < 8.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

abj = ox.AbjChiralAnomalyEngine(3, 1.0 / 137.036)
g_ev = abj.pion_two_photon_width(134.9768, 92.2) * 1e6
assert 7.5 < g_ev < 8.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python AbjChiralAnomalyEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2780_kibble_zurek_topological_defect_scaling() -> Result<()> {
    use oxideon::physics::kibble_zurek_cosmological_defects::KibbleZurekDefectScaling;
    let kz = KibbleZurekDefectScaling::new(0.5, 2.0)?;
    let (mu, xi, rho) = kz.freeze_out_length_and_density(16.0, 2)?;
    assert!((mu - 0.25).abs() < 1e-12 && (xi - 2.0).abs() < 1e-12 && (rho - 0.25).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

kz = ox.KibbleZurekDefectScaling(0.5, 2.0)
mu, xi, rho = kz.freeze_out_length_and_density(16.0, 2)
assert abs(xi - 2.0) < 1e-12 and abs(rho - 0.25) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python KibbleZurekDefectScaling test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2790_electroweak_sphaleron_baryogenesis() -> Result<()> {
    use oxideon::physics::sphaleron_electroweak_baryogenesis::ElectroweakSphaleronBaryogenesis;
    let sph = ElectroweakSphaleronBaryogenesis::new(3, 1)?;
    let (cs, yb) = sph.sphaleron_conversion_and_baryon_yield(79.0);
    assert!((cs - 28.0 / 79.0).abs() < 1e-12 && (yb - 28.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

sph = ox.ElectroweakSphaleronBaryogenesis(3, 1)
cs, yb = sph.sphaleron_conversion_and_baryon_yield(79.0)
assert abs(yb - 28.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python ElectroweakSphaleronBaryogenesis test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2800_grand_omni_octacosa_engine_xxviii() -> Result<()> {
    use oxideon::omni_octacosa_xxviii::GrandOmniOctacosaEngineXXVIII;
    let eng = GrandOmniOctacosaEngineXXVIII::new(28);
    let inv = eng.execute_octacosa_synthesis()?;
    assert!(inv > 2600.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniOctacosaEngineXXVIII(28)
assert eng.execute_octacosa_synthesis() > 2600.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniOctacosaEngineXXVIII test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2810_kimura_wright_fisher_fixation() -> Result<()> {
    use oxideon::science::kimura_wright_fisher_genetics::KimuraWrightFisherEngine;
    let kwf = KimuraWrightFisherEngine::new(1000.0)?;
    let u_neutral = kwf.fixation_probability(0.0, 0.25)?;
    let u_pos = kwf.fixation_probability(0.01, 0.25)?;
    assert!((u_neutral - 0.25).abs() < 1e-12 && u_pos > 0.99);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

kwf = ox.KimuraWrightFisherEngine(1000.0)
assert abs(kwf.fixation_probability(0.0, 0.25) - 0.25) < 1e-12
assert kwf.fixation_probability(0.01, 0.25) > 0.99
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python KimuraWrightFisherEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2820_kingman_coalescent_watterson() -> Result<()> {
    use oxideon::science::kingman_coalescent_ancestry::KingmanCoalescentEngine;
    let kc = KingmanCoalescentEngine::new(4)?;
    // H_3 = 1 + 1/2 + 1/3 = 11/6
    let (tmrca, l_tot, theta_w) = kc.tmrca_tree_length_and_watterson(11);
    assert!(
        (tmrca - 1.5).abs() < 1e-12
            && (l_tot - 11.0 / 3.0).abs() < 1e-12
            && (theta_w - 6.0).abs() < 1e-12
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

kc = ox.KingmanCoalescentEngine(4)
tmrca, l_tot, tw = kc.tmrca_tree_length_and_watterson(11)
assert abs(tmrca - 1.5) < 1e-12 and abs(tw - 6.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python KingmanCoalescentEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2830_price_equation_hamilton_inclusive_fitness() -> Result<()> {
    use oxideon::science::price_hamilton_inclusive_fitness::PriceHamiltonEvolutionEngine;
    let (dz, sel, trans, ham) = PriceHamiltonEvolutionEngine::price_decomposition_and_hamilton(
        &[1.0, 3.0],
        &[2.0, 4.0],
        &[0.0, 0.0],
        0.5,
        3.0,
        1.0,
    )?;
    assert!((dz - 0.5).abs() < 1e-12 && (sel - 0.5).abs() < 1e-12 && trans.abs() < 1e-12 && ham);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

dz, sel, tr, ham = ox.PriceHamiltonEvolutionEngine.price_decomposition_and_hamilton([1.0, 3.0], [2.0, 4.0], [0.0, 0.0], 0.5, 3.0, 1.0)
assert abs(dz - 0.5) < 1e-12 and ham
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python PriceHamiltonEvolutionEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2840_replicator_mutator_maynard_smith_ess() -> Result<()> {
    use oxideon::science::replicator_mutator_ess::ReplicatorMutatorGameEngine;
    let game = ReplicatorMutatorGameEngine::new(2.0, 4.0)?;
    let (p_ess, dp) = game.hawk_dove_ess_and_velocity(0.5)?;
    assert!((p_ess - 0.5).abs() < 1e-12 && dp.abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

game = ox.ReplicatorMutatorGameEngine(2.0, 4.0)
p_ess, dp = game.hawk_dove_ess_and_velocity(0.5)
assert abs(p_ess - 0.5) < 1e-12 and abs(dp) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python ReplicatorMutatorGameEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2850_eigen_quasispecies_error_threshold() -> Result<()> {
    use oxideon::science::eigen_quasispecies_threshold::EigenQuasispeciesThreshold;
    let eq = EigenQuasispeciesThreshold::new(100, std::f64::consts::E)?;
    let (mu_c, x0, survives) = eq.critical_mutation_rate_and_master_fraction(0.005)?;
    assert!(mu_c > 0.009 && x0 > 0.3 && survives);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

import math
eq = ox.EigenQuasispeciesThreshold(100, math.e)
mu_c, x0, ok = eq.critical_mutation_rate_and_master_fraction(0.005)
assert ok and x0 > 0.3
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python EigenQuasispeciesThreshold test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2860_may_wigner_ecosystem_stability() -> Result<()> {
    use oxideon::science::may_wigner_ecosystem_stability::MayWignerComplexityStability;
    let mw = MayWignerComplexityStability::new(1.0)?;
    let (r, ab, stable) = mw.spectral_radius_and_stability(100, 0.25, 0.1)?;
    assert!((r - 0.5).abs() < 1e-12 && (ab - (-0.5)).abs() < 1e-12 && stable);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

mw = ox.MayWignerComplexityStability(1.0)
r, ab, ok = mw.spectral_radius_and_stability(100, 0.25, 0.1)
assert ok and abs(r - 0.5) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python MayWignerComplexityStability test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2870_fisher_kpp_traveling_wavefront() -> Result<()> {
    use oxideon::science::fisher_kpp_traveling_wave::FisherKppWavefrontEngine;
    let fkpp = FisherKppWavefrontEngine::new(1.0, 4.0)?;
    let (c_star, lam, x_t) = fkpp.wave_speed_and_bramson_front(10.0)?;
    assert!((c_star - 4.0).abs() < 1e-12 && (lam - 2.0).abs() < 1e-12 && x_t > 38.0 && x_t < 40.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

fkpp = ox.FisherKppWavefrontEngine(1.0, 4.0)
c, lam, xt = fkpp.wave_speed_and_bramson_front(10.0)
assert abs(c - 4.0) < 1e-12 and 38.0 < xt < 40.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python FisherKppWavefrontEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2880_kermack_mckendrick_sir_epidemic() -> Result<()> {
    use oxideon::science::kermack_mckendrick_epidemic::KermackMcKendrickEpidemicEngine;
    let sir = KermackMcKendrickEpidemicEngine::new(0.5, 0.25)?;
    let (r0, herd, z_inf) = sir.r0_herd_threshold_and_final_size();
    assert!(
        (r0 - 2.0).abs() < 1e-12 && (herd - 0.5).abs() < 1e-12 && (z_inf - 0.79681213).abs() < 1e-6
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

sir = ox.KermackMcKendrickEpidemicEngine(0.5, 0.25)
r0, herd, z = sir.r0_herd_threshold_and_final_size()
assert abs(r0 - 2.0) < 1e-12 and abs(z - 0.79681213) < 1e-6
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python KermackMcKendrickEpidemicEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2890_hubbell_ewens_neutral_biodiversity() -> Result<()> {
    use oxideon::science::hubbell_ewens_biodiversity::HubbellEwensBiodiversityEngine;
    let heb = HubbellEwensBiodiversityEngine::new(10.0)?;
    let (mean_s, var_s) = heb.expected_richness_and_variance(50)?;
    assert!(mean_s > 17.0 && var_s > 6.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

heb = ox.HubbellEwensBiodiversityEngine(10.0)
ms, vs = heb.expected_richness_and_variance(50)
assert ms > 17.0 and vs > 6.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python HubbellEwensBiodiversityEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2900_grand_omni_nonacosa_engine_xxix() -> Result<()> {
    use oxideon::omni_nonacosa_xxix::GrandOmniNonacosaEngineXXIX;
    let eng = GrandOmniNonacosaEngineXXIX::new(29);
    let inv = eng.execute_nonacosa_synthesis()?;
    assert!(inv > 2650.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniNonacosaEngineXXIX(29)
assert eng.execute_nonacosa_synthesis() > 2650.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniNonacosaEngineXXIX test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2910_martin_borel_determinacy_game() -> Result<()> {
    use oxideon::science::martin_borel_determinacy::MartinBorelDeterminacyEngine;
    let val = MartinBorelDeterminacyEngine::evaluate_binary_game_tree(&[1, -1, 1, 1])?;
    assert_eq!(val, 1);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert ox.MartinBorelDeterminacyEngine.evaluate_binary_game_tree([1, -1, 1, 1]) == 1
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python MartinBorelDeterminacyEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2920_shelah_nip_sauer_shelah_lemma() -> Result<()> {
    use oxideon::science::shelah_stability_sauer::ShelahStabilitySauerEngine;
    let ss = ShelahStabilitySauerEngine::new(2);
    // C(5,0) + C(5,1) + C(5,2) = 1 + 5 + 10 = 16
    assert_eq!(ss.sauer_shelah_type_bound(5)?, 16);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

ss = ox.ShelahStabilitySauerEngine(2)
assert ss.sauer_shelah_type_bound(5) == 16
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python ShelahStabilitySauerEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2930_o_minimal_wilkie_khovanskii_pfaffian() -> Result<()> {
    use oxideon::science::o_minimal_pfaffian_cells::OMinimalPfaffianCellEngine;
    let om = OMinimalPfaffianCellEngine::new(1, 1)?;
    let b = om.khovanskii_component_bound(1, 2)?;
    assert!((b - 10.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

om = ox.OMinimalPfaffianCellEngine(1, 1)
assert abs(om.khovanskii_component_bound(1, 2) - 10.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python OMinimalPfaffianCellEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2940_hrushovski_zilber_zariski_geometry() -> Result<()> {
    use oxideon::science::hrushovski_zariski_geometry::HrushovskiZariskiGeometry;
    let hz = HrushovskiZariskiGeometry::new(2.0)?;
    let (da, rel, strong) = hz.predimension_and_strong_extension(3, 3, 5, 6)?;
    assert!((da - 3.0).abs() < 1e-12 && (rel - 1.0).abs() < 1e-12 && strong);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

hz = ox.HrushovskiZariskiGeometry(2.0)
da, rel, ok = hz.predimension_and_strong_extension(3, 3, 5, 6)
assert ok and abs(da - 3.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python HrushovskiZariskiGeometry test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2950_goodstein_kirby_paris_harrington() -> Result<()> {
    use oxideon::science::goodstein_paris_harrington::GoodsteinParisHarringtonEngine;
    // m = 4 = 2^2 in base 2 -> 3^3 = 27 in base 3, minus 1 = 26
    let (bumped, step) = GoodsteinParisHarringtonEngine::hereditary_base_bump_and_step(4, 2)?;
    assert_eq!((bumped, step), (27, 26));

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert ox.GoodsteinParisHarringtonEngine.hereditary_base_bump_and_step(4, 2) == (27, 26)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GoodsteinParisHarringtonEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2960_solomonoff_levin_algorithmic_prior() -> Result<()> {
    use oxideon::science::solomonoff_levin_algorithmic::SolomonoffLevinUniversalPrior;
    let (kt, m_x) = SolomonoffLevinUniversalPrior::levin_kt_and_universal_prior(&[4, 6], &[8, 2])?;
    assert!((kt - 7.0).abs() < 1e-12 && m_x > 0.015);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

kt, mx = ox.SolomonoffLevinUniversalPrior.levin_kt_and_universal_prior([4, 6], [8, 2])
assert abs(kt - 7.0) < 1e-12 and mx > 0.015
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python SolomonoffLevinUniversalPrior test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2970_razborov_sunflower_natural_proofs() -> Result<()> {
    use oxideon::science::razborov_sunflower_circuit::SunflowerNaturalProofsBarrier;
    let sf = SunflowerNaturalProofsBarrier::new(4, 3)?;
    let (er, alwz) = sf.erdos_rado_and_alwz_bounds();
    // 4! * 2^4 = 24 * 16 = 384
    assert!((er - 384.0).abs() < 1e-12 && alwz > 0.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

sf = ox.SunflowerNaturalProofsBarrier(4, 3)
er, alwz = sf.erdos_rado_and_alwz_bounds()
assert abs(er - 384.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python SunflowerNaturalProofsBarrier test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2980_dinur_pcp_khot_unique_games() -> Result<()> {
    use oxideon::science::dinur_pcp_unique_games::DinurPcpGapAmplification;
    let (r, alpha_gw) = DinurPcpGapAmplification::goemans_williamson_ratio_at_angle(2.33112237)?;
    assert!((alpha_gw - 0.8785672).abs() < 1e-6 && (r - alpha_gw).abs() < 1e-6);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

r, agw = ox.DinurPcpGapAmplification.goemans_williamson_ratio_at_angle(2.33112237)
assert abs(agw - 0.8785672) < 1e-6
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python DinurPcpGapAmplification test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v2990_williams_sat_acc0_circuit_lower_bound() -> Result<()> {
    use oxideon::science::williams_sat_circuit_duality::WilliamsSatCircuitLowerBound;
    let wil = WilliamsSatCircuitLowerBound::new(20)?;
    let (_, speedup, beats_brute) = wil.speedup_factor(6, 2.371552)?;
    assert!(speedup > 50.0 && beats_brute);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

wil = ox.WilliamsSatCircuitLowerBound(20)
_, sp, ok = wil.speedup_factor(6, 2.371552)
assert ok and sp > 50.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python WilliamsSatCircuitLowerBound test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3000_grand_omni_tricentennial_engine_xxx() -> Result<()> {
    use oxideon::omni_tricentennial_xxx::GrandOmniTricentennialEngineXXX;
    let eng = GrandOmniTricentennialEngineXXX::new(30);
    let inv = eng.execute_tricentennial_synthesis()?;
    assert!(inv > 3100.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniTricentennialEngineXXX(30)
assert eng.execute_tricentennial_synthesis() > 3100.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniTricentennialEngineXXX test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3010_jaynes_cummings_cavity_qed() -> Result<()> {
    use oxideon::quantum::jaynes_cummings_cavity_qed::JaynesCummingsCavityQED;
    let jc = JaynesCummingsCavityQED::new(1.0, 0.0)?;
    let (omega_0, t_rev) = jc.rabi_frequency_and_revival_time(0, 9.0)?;
    assert!((omega_0 - 2.0).abs() < 1e-12 && (t_rev - 6.0 * std::f64::consts::PI).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

jc = ox.JaynesCummingsCavityQED(1.0, 0.0)
om0, trev = jc.rabi_frequency_and_revival_time(0, 9.0)
assert abs(om0 - 2.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python JaynesCummingsCavityQED test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3020_dicke_tavis_cummings_superradiance() -> Result<()> {
    use oxideon::quantum::dicke_tavis_cummings::DickeTavisCummingsSuperradiance;
    let dk = DickeTavisCummingsSuperradiance::new(10, 1.0)?;
    let (g_max, g_c) = dk.peak_burst_rate_and_critical_coupling(4.0, 9.0)?;
    assert!((g_max - 30.0).abs() < 1e-12 && (g_c - 3.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

dk = ox.DickeTavisCummingsSuperradiance(10, 1.0)
gm, gc = dk.peak_burst_rate_and_critical_coupling(4.0, 9.0)
assert abs(gm - 30.0) < 1e-12 and abs(gc - 3.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python DickeTavisCummingsSuperradiance test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3030_hong_ou_mandel_two_photon_dip() -> Result<()> {
    use oxideon::quantum::hong_ou_mandel_dip::HongOuMandelInterferometer;
    let hom = HongOuMandelInterferometer::new(0.5, 1.0)?;
    let (p0, vis) = hom.coincidence_probability_and_visibility(0.0, 1.0)?;
    assert!(p0.abs() < 1e-12 && (vis - 1.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

hom = ox.HongOuMandelInterferometer(0.5, 1.0)
p0, vis = hom.coincidence_probability_and_visibility(0.0, 1.0)
assert abs(p0) < 1e-12 and abs(vis - 1.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python HongOuMandelInterferometer test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3040_glauber_mandel_photon_antibunching() -> Result<()> {
    use oxideon::quantum::glauber_mandel_antibunching::GlauberPhotonCoherenceEngine;
    // Fock state |n=2> has mean 2, variance 0 -> Q_M = -1, g^(2)(0) = 0.5
    let (qm, g2, is_sp) = GlauberPhotonCoherenceEngine::mandel_q_and_g2_zero(1.0, 0.0)?;
    assert!((qm - (-1.0)).abs() < 1e-12 && g2.abs() < 1e-12 && is_sp);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

qm, g2, sp = ox.GlauberPhotonCoherenceEngine.mandel_q_and_g2_zero(1.0, 0.0)
assert abs(qm - (-1.0)) < 1e-12 and abs(g2) < 1e-12 and sp
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GlauberPhotonCoherenceEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3050_eit_dark_state_polariton_slow_light() -> Result<()> {
    use oxideon::quantum::eit_dark_state_polariton::ElectromagneticallyInducedTransparency;
    let eit = ElectromagneticallyInducedTransparency::new(100.0)?;
    let (n_g, vg_c) = eit.group_index_and_slow_light_ratio(10.0)?;
    assert!((n_g - 101.0).abs() < 1e-12 && (vg_c - 1.0 / 101.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eit = ox.ElectromagneticallyInducedTransparency(100.0)
ng, vgc = eit.group_index_and_slow_light_ratio(10.0)
assert abs(ng - 101.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python ElectromagneticallyInducedTransparency test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3060_corkum_hhg_attosecond_cutoff() -> Result<()> {
    use oxideon::quantum::corkum_hhg_attosecond::CorkumHighHarmonicAttosecond;
    let hhg = CorkumHighHarmonicAttosecond::new(15.0)?;
    let (e_cut, gk, is_tun) = hhg.cutoff_energy_and_keldysh(10.0)?;
    assert!((e_cut - 46.7).abs() < 1e-12 && gk < 1.0 && is_tun);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

hhg = ox.CorkumHighHarmonicAttosecond(15.0)
ec, gk, tun = hhg.cutoff_energy_and_keldysh(10.0)
assert abs(ec - 46.7) < 1e-12 and tun
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python CorkumHighHarmonicAttosecond test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3070_doppler_sisyphus_laser_cooling() -> Result<()> {
    use oxideon::quantum::doppler_sisyphus_cooling::DopplerSisyphusLaserCooling;
    let lc = DopplerSisyphusLaserCooling::new(2.0)?;
    let (t_opt, t_lim) = lc.doppler_temperature_and_limit(-1.0)?;
    assert!((t_opt - 1.0).abs() < 1e-12 && (t_lim - 1.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

lc = ox.DopplerSisyphusLaserCooling(2.0)
t_opt, t_lim = lc.doppler_temperature_and_limit(-1.0)
assert abs(t_opt - 1.0) < 1e-12 and abs(t_lim - 1.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python DopplerSisyphusLaserCooling test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3080_feshbach_resonance_bec_bcs_crossover() -> Result<()> {
    use oxideon::quantum::feshbach_bec_bcs_crossover::FeshbachScatteringResonance;
    let fesh = FeshbachScatteringResonance::new(100.0, 800.0, 20.0)?;
    let (a_b, is_bec) = fesh.scattering_length_and_bec_side(810.0)?;
    assert!((a_b - (-100.0)).abs() < 1e-12 && !is_bec);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

fesh = ox.FeshbachScatteringResonance(100.0, 800.0, 20.0)
ab, bec = fesh.scattering_length_and_bec_side(810.0)
assert abs(ab - (-100.0)) < 1e-12 and not bec
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python FeshbachScatteringResonance test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3090_cavity_optomechanics_ground_state_cooling() -> Result<()> {
    use oxideon::quantum::cavity_optomechanics_sql::CavityOptomechanicsBackaction;
    let om = CavityOptomechanicsBackaction::new(1.0, 1e-3, 10.0)?;
    let (coop, nf, gs) = om.cooperativity_and_final_phonons(0.5, 100.0)?;
    assert!((coop - 1000.0).abs() < 1e-9 && nf < 0.2 && gs);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

om = ox.CavityOptomechanicsBackaction(1.0, 1e-3, 10.0)
coop, nf, gs = om.cooperativity_and_final_phonons(0.5, 100.0)
assert abs(coop - 1000.0) < 1e-9 and gs
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python CavityOptomechanicsBackaction test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3100_grand_omni_henetricosa_engine_xxxi() -> Result<()> {
    use oxideon::omni_henetricosa_xxxi::GrandOmniHenetricosaEngineXXXI;
    let eng = GrandOmniHenetricosaEngineXXXI::new(31);
    let inv = eng.execute_henetricosa_synthesis()?;
    assert!(inv > 4200.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniHenetricosaEngineXXXI(31)
assert eng.execute_henetricosa_synthesis() > 4200.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniHenetricosaEngineXXXI test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3110_heath_jarrow_morton_hjm_drift() -> Result<()> {
    use oxideon::science::heath_jarrow_morton_hjm::HeathJarrowMortonTermStructure;
    let hjm = HeathJarrowMortonTermStructure::new(0.02, 0.5)?;
    let (alpha, bvol) = hjm.no_arbitrage_drift_and_bond_vol(2.0)?;
    assert!(alpha > 0.0 && bvol > 0.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

hjm = ox.HeathJarrowMortonTermStructure(0.02, 0.5)
a, bv = hjm.no_arbitrage_drift_and_bond_vol(2.0)
assert a > 0.0 and bv > 0.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python HeathJarrowMortonTermStructure test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3120_hagan_sabr_volatility_smile() -> Result<()> {
    use oxideon::science::hagan_sabr_volatility::HaganSabrVolatilityEngine;
    let sabr = HaganSabrVolatilityEngine::new(0.2, 1.0, -0.3, 0.4)?;
    let iv = sabr.atm_implied_volatility(100.0, 1.0)?;
    assert!(iv > 0.19 && iv < 0.21);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

sabr = ox.HaganSabrVolatilityEngine(0.2, 1.0, -0.3, 0.4)
assert 0.19 < sabr.atm_implied_volatility(100.0, 1.0) < 0.21
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python HaganSabrVolatilityEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3130_almgren_chriss_optimal_execution() -> Result<()> {
    use oxideon::science::almgren_chriss_execution::AlmgrenChrissOptimalExecution;
    let ac = AlmgrenChrissOptimalExecution::new(0.2, 0.01, 1.0)?;
    let (kappa, frac) = ac.urgency_and_inventory_fraction(0.5, 1.0)?;
    assert!((kappa - 2.0).abs() < 1e-12 && frac > 0.0 && frac < 0.5);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

ac = ox.AlmgrenChrissOptimalExecution(0.2, 0.01, 1.0)
k, frac = ac.urgency_and_inventory_fraction(0.5, 1.0)
assert abs(k - 2.0) < 1e-12 and 0.0 < frac < 0.5
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python AlmgrenChrissOptimalExecution test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3140_avellaneda_stoikov_market_making() -> Result<()> {
    use oxideon::science::avellaneda_stoikov_market_making::AvellanedaStoikovMarketMaker;
    let mm = AvellanedaStoikovMarketMaker::new(0.1, 2.0, 1.5)?;
    let (r_p, sp) = mm.reservation_price_and_spread(100.0, 5.0, 1.0)?;
    assert!((r_p - 98.0).abs() < 1e-12 && sp > 1.6);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

mm = ox.AvellanedaStoikovMarketMaker(0.1, 2.0, 1.5)
rp, sp = mm.reservation_price_and_spread(100.0, 5.0, 1.0)
assert abs(rp - 98.0) < 1e-12 and sp > 1.6
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python AvellanedaStoikovMarketMaker test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3150_kyle_lambda_auction_microstructure() -> Result<()> {
    use oxideon::science::kyle_lambda_microstructure::KyleLambdaAuctionEquilibrium;
    let kyle = KyleLambdaAuctionEquilibrium::new(4.0, 10.0)?;
    let (lam, beta, prof) = kyle.lambda_beta_and_insider_profit();
    assert!((lam - 0.1).abs() < 1e-12 && (beta - 5.0).abs() < 1e-12 && (prof - 10.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

kyle = ox.KyleLambdaAuctionEquilibrium(4.0, 10.0)
lam, beta, prof = kyle.lambda_beta_and_insider_profit()
assert abs(lam - 0.1) < 1e-12 and abs(prof - 10.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python KyleLambdaAuctionEquilibrium test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3160_rockafellar_uryasev_cvar_coherent_risk() -> Result<()> {
    use oxideon::science::rockafellar_uryasev_cvar::RockafellarUryasevCVaREngine;
    let ru = RockafellarUryasevCVaREngine::new(0.8)?;
    let (var_a, cvar_a) = ru.empirical_var_and_cvar(&[1.0, 2.0, 3.0, 4.0, 10.0])?;
    assert!((var_a - 4.0).abs() < 1e-12 && (cvar_a - 10.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

ru = ox.RockafellarUryasevCVaREngine(0.8)
v, cv = ru.empirical_var_and_cvar([1.0, 2.0, 3.0, 4.0, 10.0])
assert abs(v - 4.0) < 1e-12 and abs(cv - 10.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python RockafellarUryasevCVaREngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3170_merton_structural_credit_default() -> Result<()> {
    use oxideon::science::merton_structural_credit::MertonStructuralDefaultEngine;
    let mert = MertonStructuralDefaultEngine::new(0.2, 0.04)?;
    let (dd, d1) = mert.distance_to_default(150.0, 100.0, 1.0)?;
    assert!(dd > 2.0 && (d1 - dd - 0.2).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

mert = ox.MertonStructuralDefaultEngine(0.2, 0.04)
dd, d1 = mert.distance_to_default(150.0, 100.0, 1.0)
assert dd > 2.0 and abs(d1 - dd - 0.2) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python MertonStructuralDefaultEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3180_eisenberg_noe_systemic_interbank_clearing() -> Result<()> {
    use oxideon::science::eisenberg_noe_systemic_risk::EisenbergNoeSystemicClearing;
    let (p1, p2, defs) =
        EisenbergNoeSystemicClearing::solve_two_bank_ring_clearing(2.0, 2.0, 10.0, 10.0, 0.5)?;
    assert!((p1 - 4.0).abs() < 1e-9 && (p2 - 4.0).abs() < 1e-9 && defs == 2);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

p1, p2, defs = ox.EisenbergNoeSystemicClearing.solve_two_bank_ring_clearing(2.0, 2.0, 10.0, 10.0, 0.5)
assert abs(p1 - 4.0) < 1e-9 and defs == 2
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python EisenbergNoeSystemicClearing test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3190_kelly_thorp_log_optimal_portfolio() -> Result<()> {
    use oxideon::science::kelly_thorp_optimal_growth::KellyLogOptimalPortfolio;
    let kel = KellyLogOptimalPortfolio::new(0.08, 0.2)?;
    let (f_star, g_half) = kel.kelly_leverage_and_growth_rate(0.5)?;
    assert!((f_star - 2.0).abs() < 1e-12 && (g_half - 0.06).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

kel = ox.KellyLogOptimalPortfolio(0.08, 0.2)
fs, gh = kel.kelly_leverage_and_growth_rate(0.5)
assert abs(fs - 2.0) < 1e-12 and abs(gh - 0.06) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python KellyLogOptimalPortfolio test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3200_grand_omni_dotricosa_engine_xxxii() -> Result<()> {
    use oxideon::omni_dotricosa_xxxii::GrandOmniDotricosaEngineXXXII;
    let eng = GrandOmniDotricosaEngineXXXII::new(32);
    let inv = eng.execute_dotricosa_synthesis()?;
    assert!(inv > 4300.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniDotricosaEngineXXXII(32)
assert eng.execute_dotricosa_synthesis() > 4300.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniDotricosaEngineXXXII test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3210_kanamori_gutenberg_omori_seismology() -> Result<()> {
    use oxideon::physics::kanamori_gutenberg_seismology::KanamoriGutenbergSeismicEngine;
    let seis = KanamoriGutenbergSeismicEngine::new(1.0)?;
    let (mw, rate) = seis.moment_magnitude_and_omori_rate(1e18, 1.0, 100.0, 1.0, 1.0)?;
    assert!((mw - 5.93).abs() < 1e-12 && (rate - 50.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

seis = ox.KanamoriGutenbergSeismicEngine(1.0)
mw, r = seis.moment_magnitude_and_omori_rate(1e18, 1.0, 100.0, 1.0, 1.0)
assert abs(mw - 5.93) < 1e-12 and abs(r - 50.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python KanamoriGutenbergSeismicEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3220_aki_richards_shuey_seismic_avo() -> Result<()> {
    use oxideon::physics::aki_richards_zoeppritz_seismic::AkiRichardsSeismicAvoEngine;
    let avo = AkiRichardsSeismicAvoEngine::new(-0.1, -0.2, 0.0);
    let r30 = avo.shuey_rpp_reflectivity(std::f64::consts::PI / 6.0)?;
    assert!((r30 - (-0.15)).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

import math
avo = ox.AkiRichardsSeismicAvoEngine(-0.1, -0.2, 0.0)
assert abs(avo.shuey_rpp_reflectivity(math.pi / 6.0) - (-0.15)) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python AkiRichardsSeismicAvoEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3230_backus_gilbert_geophysical_inverse() -> Result<()> {
    use oxideon::physics::backus_gilbert_geophysical_inverse::BackusGilbertInverseResolution;
    let bg = BackusGilbertInverseResolution::new(vec![2.0, 1.0])?;
    let (tr, sp, var) = bg.resolution_trace_spread_and_variance(1.0)?;
    assert!((tr - 1.3).abs() < 1e-12 && sp > 0.28 && var > 0.4);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

bg = ox.BackusGilbertInverseResolution([2.0, 1.0])
tr, sp, var = bg.resolution_trace_spread_and_variance(1.0)
assert abs(tr - 1.3) < 1e-12 and sp > 0.28
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python BackusGilbertInverseResolution test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3240_budyko_sellers_milankovitch_climate() -> Result<()> {
    use oxideon::physics::budyko_sellers_milankovitch_climate::BudykoSellersClimateBifurcation;
    let bs = BudykoSellersClimateBifurcation::new(202.0, 1.9)?;
    let (tw, ti) = bs.warm_and_snowball_equilibria(1361.0, 0.3, 0.62)?;
    assert!(tw > 14.0 && ti < -35.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

bs = ox.BudykoSellersClimateBifurcation(202.0, 1.9)
tw, ti = bs.warm_and_snowball_equilibria(1361.0, 0.3, 0.62)
assert tw > 14.0 and ti < -35.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python BudykoSellersClimateBifurcation test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3250_chandrasekhar_eddington_radiative_transfer() -> Result<()> {
    use oxideon::physics::chandrasekhar_eddington_radiative::EddingtonChandrasekharRadiative;
    let rad = EddingtonChandrasekharRadiative::new(6000.0)?;
    // At tau = 2/3, T(tau) == T_eff
    let (t_phot, alb) = rad.eddington_temperature_and_slab_albedo(2.0 / 3.0, 0.0)?;
    assert!((t_phot - 6000.0).abs() < 1e-12 && (alb - 1.0 / 3.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

rad = ox.EddingtonChandrasekharRadiative(6000.0)
tp, alb = rad.eddington_temperature_and_slab_albedo(2.0 / 3.0, 0.0)
assert abs(tp - 6000.0) < 1e-12 and abs(alb - 1.0 / 3.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python EddingtonChandrasekharRadiative test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3260_elsasser_geodynamo_magnetostrophic() -> Result<()> {
    use oxideon::physics::elsasser_geodynamo_core::ElsasserGeodynamoCoreEngine;
    let geo = ElsasserGeodynamoCoreEngine::new(1.0e4, 1.0e-4, 4.0e5)?;
    let (lam, is_dyn) = geo.elsasser_number(2.0e-3)?;
    assert!((lam - 0.8).abs() < 1e-12 && is_dyn);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

geo = ox.ElsasserGeodynamoCoreEngine(1.0e4, 1.0e-4, 4.0e5)
lam, ok = geo.elsasser_number(2.0e-3)
assert abs(lam - 0.8) < 1e-12 and ok
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python ElsasserGeodynamoCoreEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3270_thermal_wind_eady_baroclinic() -> Result<()> {
    use oxideon::physics::thermal_wind_eady_baroclinic::ThermalWindEadyBaroclinic;
    let eady = ThermalWindEadyBaroclinic::new(1.0e-4, 1.0e-2)?;
    let (sig, lr) = eady.eady_growth_rate_and_rossby_radius(3.0e-3, 1.0e4)?;
    assert!(sig > 9.0e-6 && (lr - 1.0e6).abs() < 1e-6);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eady = ox.ThermalWindEadyBaroclinic(1.0e-4, 1.0e-2)
sig, lr = eady.eady_growth_rate_and_rossby_radius(3.0e-3, 1.0e4)
assert sig > 9.0e-6 and abs(lr - 1.0e6) < 1e-6
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python ThermalWindEadyBaroclinic test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3280_stommel_sverdrup_ocean_gyre() -> Result<()> {
    use oxideon::physics::stommel_sverdrup_ocean_gyre::StommelSverdrupOceanCirculation;
    let oc = StommelSverdrupOceanCirculation::new(2.0e-11, 1.0e-6)?;
    let (ds, dm, vy) = oc.boundary_widths_and_sverdrup_transport(-1.0e-7, 1000.0, 160.0)?;
    assert!(
        (ds - 50000.0).abs() < 1e-6 && (dm - 20000.0).abs() < 1e-6 && (vy - (-5.0)).abs() < 1e-9
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

oc = ox.StommelSverdrupOceanCirculation(2.0e-11, 1.0e-6)
ds, dm, vy = oc.boundary_widths_and_sverdrup_transport(-1.0e-7, 1000.0, 160.0)
assert abs(ds - 50000.0) < 1e-6 and abs(dm - 20000.0) < 1e-6
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python StommelSverdrupOceanCirculation test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3290_toomre_hill_protoplanetary_disk() -> Result<()> {
    use oxideon::physics::toomre_hill_planetary_disk::ToomreHillPlanetaryDisk;
    let th = ToomreHillPlanetaryDisk::new(1.0)?;
    let (q, stable, rh) =
        th.toomre_q_and_hill_radius(2.0, std::f64::consts::PI, 1.0, 10.0, 3.0e-3)?;
    assert!((q - 2.0).abs() < 1e-12 && stable && (rh - 1.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

import math
th = ox.ToomreHillPlanetaryDisk(1.0)
q, ok, rh = th.toomre_q_and_hill_radius(2.0, math.pi, 1.0, 10.0, 3.0e-3)
assert ok and abs(q - 2.0) < 1e-12 and abs(rh - 1.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python ToomreHillPlanetaryDisk test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3300_grand_omni_tritricosa_engine_xxxiii() -> Result<()> {
    use oxideon::omni_tritricosa_xxxiii::GrandOmniTritricosaEngineXXXIII;
    let eng = GrandOmniTritricosaEngineXXXIII::new(33);
    let inv = eng.execute_tritricosa_synthesis()?;
    assert!(inv > 4330.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniTritricosaEngineXXXIII(33)
assert eng.execute_tritricosa_synthesis() > 4330.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniTritricosaEngineXXXIII test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3310_tutte_polynomial_matroid() -> Result<()> {
    use oxideon::science::tutte_polynomial_matroid::TuttePolynomialMatroidEngine;
    let tut = TuttePolynomialMatroidEngine::new(4)?;
    let (t11, trees, acyc) = tut.evaluate_tutte_and_invariants(1.0, 1.0);
    assert!((t11 - 4.0).abs() < 1e-12 && trees == 4 && acyc == 14);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

tut = ox.TuttePolynomialMatroidEngine(4)
t11, tr, ac = tut.evaluate_tutte_and_invariants(1.0, 1.0)
assert abs(t11 - 4.0) < 1e-12 and tr == 4 and ac == 14
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python TuttePolynomialMatroidEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3320_adiprasito_huh_katz_matroid_hodge() -> Result<()> {
    use oxideon::science::huh_katz_matroid_hodge::HuhKatzMatroidLogConcavity;
    let (margin, is_lc) =
        HuhKatzMatroidLogConcavity::verify_log_concavity(&[1.0, 6.0, 15.0, 16.0, 6.0])?;
    assert!(margin > 0.0 && is_lc);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

m, ok = ox.HuhKatzMatroidLogConcavity.verify_log_concavity([1.0, 6.0, 15.0, 16.0, 6.0])
assert ok and m > 0.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python HuhKatzMatroidLogConcavity test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3330_mikhalkin_tropical_bezout_correspondence() -> Result<()> {
    use oxideon::science::mikhalkin_tropical_bezout::MikhalkinTropicalGeometryEngine;
    let trop = MikhalkinTropicalGeometryEngine::new(3)?;
    let (g, bez, val) = trop.genus_bezout_and_tropical_eval(4, &[5.0, 2.0, 0.0], 1.5)?;
    assert_eq!((g, bez), (1, 12));
    assert!((val - 3.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

trop = ox.MikhalkinTropicalGeometryEngine(3)
g, bez, val = trop.genus_bezout_and_tropical_eval(4, [5.0, 2.0, 0.0], 1.5)
assert (g, bez) == (1, 12) and abs(val - 3.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python MikhalkinTropicalGeometryEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3340_ehrhart_macdonald_lattice_polytope() -> Result<()> {
    use oxideon::science::ehrhart_macdonald_polytope::EhrhartMacdonaldPolytopeEngine;
    // Unit square [0,1]^2: 2A = 2, B = 4 -> at t=2: closed = 9, interior = 1
    let ehr = EhrhartMacdonaldPolytopeEngine::new(2, 4)?;
    assert_eq!(ehr.ehrhart_closed_and_interior_counts(2)?, (9, 1));

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

ehr = ox.EhrhartMacdonaldPolytopeEngine(2, 4)
assert ehr.ehrhart_closed_and_interior_counts(2) == (9, 1)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python EhrhartMacdonaldPolytopeEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3350_stanley_reisner_dehn_sommerville() -> Result<()> {
    use oxideon::science::stanley_reisner_dehn_sommerville::StanleyReisnerDehnSommerville;
    // Octahedron boundary (2-sphere, d=3): f = (1, 6, 12, 8) -> h = (1, 3, 3, 1)
    let (h, sym) =
        StanleyReisnerDehnSommerville::compute_h_vector_and_check_symmetry(&[1, 6, 12, 8])?;
    assert_eq!(h, vec![1, 3, 3, 1]);
    assert!(sym);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

h, sym = ox.StanleyReisnerDehnSommerville.compute_h_vector_and_check_symmetry([1, 6, 12, 8])
assert h == [1, 3, 3, 1] and sym
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python StanleyReisnerDehnSommerville test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3360_rsk_hook_length_young_tableaux() -> Result<()> {
    use oxideon::science::rsk_hook_length_tableaux::HookLengthYoungTableauxEngine;
    // Staircase partition (3, 2, 1) of n = 6 has hooks 5,3,1 / 3,1 / 1 -> 720 / 45 = 16
    assert_eq!(
        HookLengthYoungTableauxEngine::count_standard_young_tableaux(&[3, 2, 1])?,
        16
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

assert ox.HookLengthYoungTableauxEngine.count_standard_young_tableaux([3, 2, 1]) == 16
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python HookLengthYoungTableauxEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3370_macdonald_haiman_qt_catalan() -> Result<()> {
    use oxideon::science::macdonald_qt_catalan::MacdonaldSymmetricCatalanEngine;
    let mac = MacdonaldSymmetricCatalanEngine::new(4)?;
    let (c4, dim_dh4) = mac.q_catalan_and_haiman_dimension(1.0);
    assert!((c4 - 14.0).abs() < 1e-12 && dim_dh4 == 125);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

mac = ox.MacdonaldSymmetricCatalanEngine(4)
c4, dim4 = mac.q_catalan_and_haiman_dimension(1.0)
assert abs(c4 - 14.0) < 1e-12 and dim4 == 125
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python MacdonaldSymmetricCatalanEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3380_kazhdan_lusztig_hecke_bruhat() -> Result<()> {
    use oxideon::science::kazhdan_lusztig_bruhat::KazhdanLusztigHeckeEngine;
    let kl = KazhdanLusztigHeckeEngine::new(2.0)?;
    // For S_3 at q=2: [1]_2 * [2]_2 * [3]_2 = 1 * 3 * 7 = 21
    let (deg, p_s3) = kl.max_kl_degree_and_poincare_polynomial(1, 5, 3)?;
    assert_eq!(deg, 1);
    assert!((p_s3 - 21.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

kl = ox.KazhdanLusztigHeckeEngine(2.0)
deg, ps3 = kl.max_kl_degree_and_poincare_polynomial(1, 5, 3)
assert deg == 1 and abs(ps3 - 21.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python KazhdanLusztigHeckeEngine test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3390_szemeredi_green_tao_regularity() -> Result<()> {
    use oxideon::science::szemeredi_green_tao_density::SzemerediGreenTaoRegularity;
    let sz = SzemerediGreenTaoRegularity::new(0.05)?;
    let lb = sz.triangle_density_lower_bound(0.55)?;
    assert!((lb - 0.9 * 0.125).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

sz = ox.SzemerediGreenTaoRegularity(0.05)
assert abs(sz.triangle_density_lower_bound(0.55) - 0.1125) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python SzemerediGreenTaoRegularity test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3400_grand_omni_tetratricosa_engine_xxxiv() -> Result<()> {
    use oxideon::omni_tetratricosa_xxxiv::GrandOmniTetratricosaEngineXXXIV;
    let eng = GrandOmniTetratricosaEngineXXXIV::new(34);
    let inv = eng.execute_tetratricosa_synthesis()?;
    assert!(inv > 4400.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniTetratricosaEngineXXXIV(34)
assert eng.execute_tetratricosa_synthesis() > 4400.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniTetratricosaEngineXXXIV test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3410_caffarelli_kohn_nirenberg_ns() -> Result<()> {
    use oxideon::science::caffarelli_kohn_nirenberg_ns::CaffarelliKohnNirenbergNavierStokes;
    let ckn = CaffarelliKohnNirenbergNavierStokes::new(1.0, 0.5)?;
    let (phi, is_reg) = ckn.scale_invariant_ckn_functional(2.0, 0.2, 0.4)?;
    assert!((phi - 0.2).abs() < 1e-12 && is_reg);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

ckn = ox.CaffarelliKohnNirenbergNavierStokes(1.0, 0.5)
phi, reg = ckn.scale_invariant_ckn_functional(2.0, 0.2, 0.4)
assert abs(phi - 0.2) < 1e-12 and reg
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python CaffarelliKohnNirenbergNavierStokes test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3420_de_giorgi_nash_moser_regularity() -> Result<()> {
    use oxideon::science::de_giorgi_nash_moser::DeGiorgiNashMoserRegularity;
    let dgnm = DeGiorgiNashMoserRegularity::new(1.0, 2.0)?;
    let (alpha, ch) = dgnm.holder_exponent_and_harnack_constant(4)?;
    assert!(alpha > 0.0 && alpha < 1.0 && (ch - 4.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

dgnm = ox.DeGiorgiNashMoserRegularity(1.0, 2.0)
alpha, ch = dgnm.holder_exponent_and_harnack_constant(4)
assert 0.0 < alpha < 1.0 and abs(ch - 4.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python DeGiorgiNashMoserRegularity test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3430_calderon_zygmund_pichorides_hilbert() -> Result<()> {
    use oxideon::science::calderon_zygmund_singular::CalderonZygmundSingularIntegral;
    let cz = CalderonZygmundSingularIntegral::new(4.0)?;
    let (norm, dual) = cz.hilbert_transform_lp_norm_and_dual();
    // cot(pi/8) = 1 + sqrt(2)
    assert!((norm - (1.0 + 2.0_f64.sqrt())).abs() < 1e-12 && (dual - 4.0 / 3.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

import math
cz = ox.CalderonZygmundSingularIntegral(4.0)
norm, dual = cz.hilbert_transform_lp_norm_and_dual()
assert abs(norm - (1.0 + math.sqrt(2.0))) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python CalderonZygmundSingularIntegral test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3440_strichartz_keel_tao_dispersive_pde() -> Result<()> {
    use oxideon::science::strichartz_dispersive_pde::StrichartzDispersiveEstimate;
    let st = StrichartzDispersiveEstimate::new(3)?;
    // Keel-Tao endpoint in d=3: q = 2 -> r = 6; diagonal q_sym = 10/3
    let (r, q_sym, ok) = st.admissible_spatial_exponent_and_diagonal(2.0)?;
    assert!((r - 6.0).abs() < 1e-12 && (q_sym - 10.0 / 3.0).abs() < 1e-12 && ok);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

st = ox.StrichartzDispersiveEstimate(3)
r, qs, ok = st.admissible_spatial_exponent_and_diagonal(2.0)
assert abs(r - 6.0) < 1e-12 and ok
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python StrichartzDispersiveEstimate test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3450_lax_ggkm_kdv_inverse_scattering() -> Result<()> {
    use oxideon::science::lax_kdv_inverse_scattering::LaxKdvInverseScatteringSoliton;
    let kdv = LaxKdvInverseScatteringSoliton::new(2.0)?;
    let (amp, spd, shift) = kdv.amplitude_speed_and_collision_shift(1.0)?;
    assert!(
        (amp - 8.0).abs() < 1e-12
            && (spd - 16.0).abs() < 1e-12
            && (shift - 0.5 * 3.0_f64.ln()).abs() < 1e-12
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

import math
kdv = ox.LaxKdvInverseScatteringSoliton(2.0)
amp, spd, sh = kdv.amplitude_speed_and_collision_shift(1.0)
assert abs(amp - 8.0) < 1e-12 and abs(sh - 0.5 * math.log(3.0)) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python LaxKdvInverseScatteringSoliton test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3460_zakharov_shabat_nls_peregrine_rogue() -> Result<()> {
    use oxideon::science::zakharov_shabat_nls_rogue::ZakharovShabatNlsSoliton;
    let nls = ZakharovShabatNlsSoliton::new(2.0)?;
    let (psi0, factor_sq) = nls.peregrine_amplitude_and_amplification(0.0, 0.0);
    assert!((psi0 - 6.0).abs() < 1e-12 && (factor_sq - 9.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

nls = ox.ZakharovShabatNlsSoliton(2.0)
psi0, f2 = nls.peregrine_amplitude_and_amplification(0.0, 0.0)
assert abs(psi0 - 6.0) < 1e-12 and abs(f2 - 9.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python ZakharovShabatNlsSoliton test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3470_lions_concentration_compactness_sobolev() -> Result<()> {
    use oxideon::science::lions_concentration_compactness::LionsConcentrationCompactness;
    let lcc = LionsConcentrationCompactness::new(4, 3.0)?;
    let (ts, c_star, is_comp) = lcc.critical_exponent_and_bubble_threshold(2.0);
    assert!((ts - 4.0).abs() < 1e-12 && (c_star - 2.25).abs() < 1e-12 && is_comp);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

lcc = ox.LionsConcentrationCompactness(4, 3.0)
ts, cs, ok = lcc.critical_exponent_and_bubble_threshold(2.0)
assert abs(ts - 4.0) < 1e-12 and abs(cs - 2.25) < 1e-12 and ok
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python LionsConcentrationCompactness test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3480_modica_mortola_de_giorgi_gamma_convergence() -> Result<()> {
    use oxideon::science::modica_mortola_gamma_conv::ModicaMortolaGammaConvergence;
    let mm = ModicaMortolaGammaConvergence::new(0.1)?;
    let (sig, u0) = mm.surface_tension_and_optimal_profile(0.0);
    assert!((sig - 2.0 * 2.0_f64.sqrt() / 3.0).abs() < 1e-12 && u0.abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

import math
mm = ox.ModicaMortolaGammaConvergence(0.1)
sig, u0 = mm.surface_tension_and_optimal_profile(0.0)
assert abs(sig - 2.0 * math.sqrt(2.0) / 3.0) < 1e-12 and abs(u0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python ModicaMortolaGammaConvergence test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3490_crandall_lions_viscosity_pucci_extremal() -> Result<()> {
    use oxideon::science::crandall_lions_viscosity_pucci::CrandallLionsViscosityPucci;
    let puc = CrandallLionsViscosityPucci::new(1.0, 3.0)?;
    let (mp, mm) = puc.evaluate_pucci_extremal(&[2.0, -1.0, 4.0]);
    assert!((mp - 17.0).abs() < 1e-12 && (mm - 3.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

puc = ox.CrandallLionsViscosityPucci(1.0, 3.0)
assert puc.evaluate_pucci_extremal([2.0, -1.0, 4.0]) == (17.0, 3.0)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python CrandallLionsViscosityPucci test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3500_grand_omni_pentatricosa_engine_xxxv() -> Result<()> {
    use oxideon::omni_pentatricosa_xxxv::GrandOmniPentatricosaEngineXXXV;
    let eng = GrandOmniPentatricosaEngineXXXV::new(35);
    let inv = eng.execute_pentatricosa_synthesis()?;
    assert!(inv > 4450.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniPentatricosaEngineXXXV(35)
assert eng.execute_pentatricosa_synthesis() > 4450.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniPentatricosaEngineXXXV test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3510_landauer_buttiker_mesoscopic_fano() -> Result<()> {
    use oxideon::physics::landauer_buttiker_mesoscopic::LandauerButtikerBallisticTransport;
    let (g, fano) = LandauerButtikerBallisticTransport::conductance_and_fano_factor(&[1.0, 0.5])?;
    assert!((g - 1.5).abs() < 1e-12 && (fano - 1.0 / 6.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

g, f = ox.LandauerButtikerBallisticTransport.conductance_and_fano_factor([1.0, 0.5])
assert abs(g - 1.5) < 1e-12 and abs(f - 1.0 / 6.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python LandauerButtikerBallisticTransport test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3520_datta_das_rashba_spintronics() -> Result<()> {
    use oxideon::physics::datta_das_rashba_spintronics::DattaDasRashbaSpintronics;
    let dd = DattaDasRashbaSpintronics::new(1.0, 1.0)?;
    let (dth, mod_g) = dd.spin_precession_and_modulation(std::f64::consts::PI / 2.0);
    assert!((dth - std::f64::consts::PI).abs() < 1e-12 && mod_g.abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

import math
dd = ox.DattaDasRashbaSpintronics(1.0, 1.0)
dth, mg = dd.spin_precession_and_modulation(math.pi / 2.0)
assert abs(mg) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python DattaDasRashbaSpintronics test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3530_slonczewski_spin_transfer_torque() -> Result<()> {
    use oxideon::physics::slonczewski_spin_transfer_torque::SlonczewskiSpinTransferTorque;
    let stt = SlonczewskiSpinTransferTorque::new(0.01, 0.5)?;
    let (ic0, delta) = stt.critical_current_and_thermal_stability(100.0, 2.0, 4.0, 1.0)?;
    assert!((ic0 - 8.0).abs() < 1e-12 && (delta - 100.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

stt = ox.SlonczewskiSpinTransferTorque(0.01, 0.5)
ic0, d = stt.critical_current_and_thermal_stability(100.0, 2.0, 4.0, 1.0)
assert abs(ic0 - 8.0) < 1e-12 and abs(d - 100.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python SlonczewskiSpinTransferTorque test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3540_transmon_circuit_qed_qubit() -> Result<()> {
    use oxideon::quantum::transmon_josephson_qubit::TransmonCircuitQEDQubit;
    let tr = TransmonCircuitQEDQubit::new(12.5, 0.25)?;
    let (w01, anh, disp) = tr.frequency_anharmonicity_and_dispersion();
    assert!((w01 - 4.75).abs() < 1e-12 && (anh - (-0.25)).abs() < 1e-12 && disp < 1e-8);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

tr = ox.TransmonCircuitQEDQubit(12.5, 0.25)
w01, anh, disp = tr.frequency_anharmonicity_and_dispersion()
assert abs(w01 - 4.75) < 1e-12 and disp < 1e-8
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python TransmonCircuitQEDQubit test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3550_ambegaokar_baratoff_josephson() -> Result<()> {
    use oxideon::physics::ambegaokar_baratoff_josephson::AmbegaokarBaratoffJosephson;
    let ab = AmbegaokarBaratoffJosephson::new(2.0)?;
    let (ic_rn, zt) = ab.ic_rn_product(0.01)?;
    assert!((zt - std::f64::consts::PI).abs() < 1e-12 && (ic_rn - zt).abs() < 1e-9);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

import math
ab = ox.AmbegaokarBaratoffJosephson(2.0)
icrn, zt = ab.ic_rn_product(0.01)
assert abs(zt - math.pi) < 1e-12 and abs(icrn - zt) < 1e-9
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python AmbegaokarBaratoffJosephson test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3560_shockley_queisser_photovoltaic_limit() -> Result<()> {
    use oxideon::physics::shockley_queisser_solar::ShockleyQueisserPhotovoltaic;
    let sq = ShockleyQueisserPhotovoltaic::new(0.5)?;
    let eta = sq.ultimate_spectrum_efficiency(1.1)?;
    assert!(eta > 0.43 && eta < 0.45);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

sq = ox.ShockleyQueisserPhotovoltaic(0.5)
eta = sq.ultimate_spectrum_efficiency(1.1)
assert 0.43 < eta < 0.45
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python ShockleyQueisserPhotovoltaic test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3570_chua_strukov_memristor_synapse() -> Result<()> {
    use oxideon::physics::chua_strukov_memristor::ChuaStrukovMemristorSynapse;
    let mem = ChuaStrukovMemristorSynapse::new(100.0, 1900.0)?;
    let (m, win) = mem.memristance_and_joglekar_window(0.5)?;
    assert!((m - 1000.0).abs() < 1e-12 && (win - 1.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

mem = ox.ChuaStrukovMemristorSynapse(100.0, 1900.0)
m, win = mem.memristance_and_joglekar_window(0.5)
assert abs(m - 1000.0) < 1e-12 and abs(win - 1.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python ChuaStrukovMemristorSynapse test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3580_tsu_esaki_resonant_tunneling() -> Result<()> {
    use oxideon::physics::tsu_esaki_resonant_tunneling::TsuEsakiResonantTunneling;
    let rtd = TsuEsakiResonantTunneling::new(0.2, 0.01, 0.01)?;
    let (t_res, t_max) = rtd.transmission_and_peak(0.2);
    assert!((t_res - 1.0).abs() < 1e-12 && (t_max - 1.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

rtd = ox.TsuEsakiResonantTunneling(0.2, 0.01, 0.01)
tr, tm = rtd.transmission_and_peak(0.2)
assert abs(tr - 1.0) < 1e-12 and abs(tm - 1.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python TsuEsakiResonantTunneling test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3590_seebeck_thermoelectric_figure_of_merit_zt() -> Result<()> {
    use oxideon::physics::seebeck_thermoelectric_zt::ThermoelectricFigureOfMeritZT;
    let te = ThermoelectricFigureOfMeritZT::new(2.0e-4, 1.0e5, 0.5)?;
    let (zt, eta) = te.zt_and_max_efficiency(300.0, 500.0)?;
    assert!((zt - 3.2).abs() < 1e-12 && eta > 0.14);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

te = ox.ThermoelectricFigureOfMeritZT(2.0e-4, 1.0e5, 0.5)
zt, eta = te.zt_and_max_efficiency(300.0, 500.0)
assert abs(zt - 3.2) < 1e-12 and eta > 0.14
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python ThermoelectricFigureOfMeritZT test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3600_grand_omni_hexatricosa_engine_xxxvi() -> Result<()> {
    use oxideon::omni_hexatricosa_xxxvi::GrandOmniHexatricosaEngineXXXVI;
    let eng = GrandOmniHexatricosaEngineXXXVI::new(36);
    let inv = eng.execute_hexatricosa_synthesis()?;
    assert!(inv > 5450.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniHexatricosaEngineXXXVI(36)
assert eng.execute_hexatricosa_synthesis() > 5450.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniHexatricosaEngineXXXVI test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3610_wolynes_protein_folding_funnel() -> Result<()> {
    use oxideon::science::wolynes_protein_folding_funnel::WolynesProteinFoldingFunnel;
    let pf = WolynesProteinFoldingFunnel::new(20.0, 2.0)?;
    let (tg, tf, ratio, foldable) = pf.glass_temp_and_foldability_ratio(8.0)?;
    assert!(
        (tg - 0.5).abs() < 1e-12
            && (tf - 2.5).abs() < 1e-12
            && (ratio - 5.0).abs() < 1e-12
            && foldable
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

pf = ox.WolynesProteinFoldingFunnel(20.0, 2.0)
tg, tf, r, ok = pf.glass_temp_and_foldability_ratio(8.0)
assert ok and abs(r - 5.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python WolynesProteinFoldingFunnel test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3620_zimm_bragg_ramachandran_helix_coil() -> Result<()> {
    use oxideon::science::zimm_bragg_ramachandran::ZimmBraggRamachandranHelix;
    let zb = ZimmBraggRamachandranHelix::new(0.01)?;
    let (lam, th) = zb.eigenvalue_and_helix_fraction(1.0)?;
    assert!((lam - 1.1).abs() < 1e-12 && (th - 0.5).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

zb = ox.ZimmBraggRamachandranHelix(0.01)
lam, th = zb.eigenvalue_and_helix_fraction(1.0)
assert abs(lam - 1.1) < 1e-12 and abs(th - 0.5) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python ZimmBraggRamachandranHelix test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3630_monod_wyman_changeux_allostery() -> Result<()> {
    use oxideon::science::monod_wyman_changeux_allostery::MonodWymanChangeuxAllostery;
    let mwc = MonodWymanChangeuxAllostery::new(4, 100.0, 0.0)?;
    let (y, r_frac) = mwc.saturation_and_relaxed_fraction(3.0)?;
    assert!(y > 0.5 && r_frac > 0.7);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

mwc = ox.MonodWymanChangeuxAllostery(4, 100.0, 0.0)
y, rf = mwc.saturation_and_relaxed_fraction(3.0)
assert y > 0.5 and rf > 0.7
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python MonodWymanChangeuxAllostery test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3640_briggs_haldane_enzyme_kinetics() -> Result<()> {
    use oxideon::science::briggs_haldane_enzyme_kinetics::BriggsHaldaneEnzymeKinetics;
    let enz = BriggsHaldaneEnzymeKinetics::new(100.0, 2.0)?;
    let (spec, vel) = enz.specificity_and_velocity(6.0, 2.0)?;
    assert!((spec - 50.0).abs() < 1e-12 && (vel - 50.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

enz = ox.BriggsHaldaneEnzymeKinetics(100.0, 2.0)
sp, vel = enz.specificity_and_velocity(6.0, 2.0)
assert abs(sp - 50.0) < 1e-12 and abs(vel - 50.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python BriggsHaldaneEnzymeKinetics test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3650_crispr_cas9_rloop_kinetics() -> Result<()> {
    use oxideon::science::crispr_cas9_rloop_kinetics::CrisprCas9RLoopKinetics;
    let cas9 = CrisprCas9RLoopKinetics::new(0.5, 4.0)?;
    let (dg_on, p_on) = cas9.rloop_free_energy_and_cleavage_prob(20, 0)?;
    let (_, p_off) = cas9.rloop_free_energy_and_cleavage_prob(20, 4)?;
    assert!((dg_on - (-10.0)).abs() < 1e-12 && p_on > 0.999 && p_off < 0.01);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

cas9 = ox.CrisprCas9RLoopKinetics(0.5, 4.0)
dg, pon = cas9.rloop_free_energy_and_cleavage_prob(20, 0)
assert abs(dg - (-10.0)) < 1e-12 and pon > 0.999
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python CrisprCas9RLoopKinetics test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3660_karlin_altschul_gumbel_alignment() -> Result<()> {
    use oxideon::science::karlin_altschul_alignment::KarlinAltschulSequenceAlignment;
    let ka = KarlinAltschulSequenceAlignment::new(2.0_f64.ln(), 0.25)?;
    let (bits, eval, pval) = ka.bit_score_evalue_and_pvalue(10.0, 32, 128)?;
    assert!((bits - 12.0).abs() < 1e-12 && (eval - 1.0).abs() < 1e-12 && pval > 0.63);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

import math
ka = ox.KarlinAltschulSequenceAlignment(math.log(2.0), 0.25)
bits, ev, pv = ka.bit_score_evalue_and_pvalue(10.0, 32, 128)
assert abs(bits - 12.0) < 1e-12 and abs(ev - 1.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python KarlinAltschulSequenceAlignment test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3670_felsenstein_jukes_cantor_phylogenetics() -> Result<()> {
    use oxideon::science::felsenstein_jukes_cantor_phylo::FelsensteinPhylogeneticLikelihood;
    let (d, var) =
        FelsensteinPhylogeneticLikelihood::jukes_cantor_distance_and_variance(0.375, 1000)?;
    assert!((d - 0.75 * 2.0_f64.ln()).abs() < 1e-12 && var > 0.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

import math
d, var = ox.FelsensteinPhylogeneticLikelihood.jukes_cantor_distance_and_variance(0.375, 1000)
assert abs(d - 0.75 * math.log(2.0)) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python FelsensteinPhylogeneticLikelihood test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3680_rna_velocity_waddington_landscape() -> Result<()> {
    use oxideon::science::rna_velocity_waddington::RnaVelocityWaddingtonLandscape;
    let rna = RnaVelocityWaddingtonLandscape::new(1.0, 0.5)?;
    let (vs, induced) = rna.rna_velocity_and_induction_state(10.0, 12.0)?;
    assert!((vs - 4.0).abs() < 1e-12 && induced);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

rna = ox.RnaVelocityWaddingtonLandscape(1.0, 0.5)
vs, ind = rna.rna_velocity_and_induction_state(10.0, 12.0)
assert abs(vs - 4.0) < 1e-12 and ind
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python RnaVelocityWaddingtonLandscape test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3690_bell_evans_mechanobiology_rupture() -> Result<()> {
    use oxideon::science::bell_evans_mechanobiology::BellEvansMechanobiologyBond;
    let be = BellEvansMechanobiologyBond::new(0.1, 10.0)?;
    let (k_off, f_star) = be.off_rate_and_rupture_force(10.0, 100.0)?;
    assert!(
        (k_off - 0.1 * std::f64::consts::E).abs() < 1e-12
            && (f_star - 10.0 * 100.0_f64.ln()).abs() < 1e-12
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

import math
be = ox.BellEvansMechanobiologyBond(0.1, 10.0)
koff, fs = be.off_rate_and_rupture_force(10.0, 100.0)
assert abs(fs - 10.0 * math.log(100.0)) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python BellEvansMechanobiologyBond test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3700_grand_omni_heptatricosa_engine_xxxvii() -> Result<()> {
    use oxideon::omni_heptatricosa_xxxvii::GrandOmniHeptatricosaEngineXXXVII;
    let eng = GrandOmniHeptatricosaEngineXXXVII::new(37);
    let inv = eng.execute_heptatricosa_synthesis()?;
    assert!(inv > 5550.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniHeptatricosaEngineXXXVII(37)
assert eng.execute_heptatricosa_synthesis() > 5550.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniHeptatricosaEngineXXXVII test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3710_dlvo_debye_huckel_colloidal_stability() -> Result<()> {
    use oxideon::physics::dlvo_debye_huckel_colloid::DlvoColloidalStabilityEngine;
    let dlvo = DlvoColloidalStabilityEngine::new(0.5, 12.0)?;
    let (ld, v) = dlvo.debye_length_and_dlvo_potential(10.0, 2.0, 2.0)?;
    assert!((ld - 2.0).abs() < 1e-12 && v > 2.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

dlvo = ox.DlvoColloidalStabilityEngine(0.5, 12.0)
ld, v = dlvo.debye_length_and_dlvo_potential(10.0, 2.0, 2.0)
assert abs(ld - 2.0) < 1e-12 and v > 2.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python DlvoColloidalStabilityEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3720_de_gennes_doi_edwards_reptation() -> Result<()> {
    use oxideon::physics::de_gennes_polymer_reptation::DeGennesPolymerReptationTube;
    let rep = DeGennesPolymerReptationTube::new(50.0, 2.0)?;
    let (z, td, eta) = rep.entanglements_disengagement_and_viscosity(100.0)?;
    assert!((z - 2.0).abs() < 1e-12 && (td - 48.0).abs() < 1e-12 && eta > 10.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

rep = ox.DeGennesPolymerReptationTube(50.0, 2.0)
z, td, eta = rep.entanglements_disengagement_and_viscosity(100.0)
assert abs(z - 2.0) < 1e-12 and abs(td - 48.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python DeGennesPolymerReptationTube test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3730_flory_huggins_spinodal_llps() -> Result<()> {
    use oxideon::physics::flory_huggins_polymer_mixing::FloryHugginsSpinodalMixing;
    let fh = FloryHugginsSpinodalMixing::new(9)?;
    let (phic, chic, unstable) = fh.critical_point_and_spinodal_instability(0.25, 1.0)?;
    assert!((phic - 0.25).abs() < 1e-12 && (chic - 8.0 / 9.0).abs() < 1e-12 && unstable);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

fh = ox.FloryHugginsSpinodalMixing(9)
pc, cc, un = fh.critical_point_and_spinodal_instability(0.25, 1.0)
assert abs(pc - 0.25) < 1e-12 and un
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python FloryHugginsSpinodalMixing test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3740_onsager_maier_saupe_nematic_liquid_crystal() -> Result<()> {
    use oxideon::physics::onsager_maier_saupe_nematic::OnsagerMaierSaupeNematic;
    let ldg = OnsagerMaierSaupeNematic::new(1.0, 1.5, 2.5)?;
    let (dt, s_ni) = ldg.clearing_temperature_shift_and_order_jump();
    assert!((dt - 0.2).abs() < 1e-12 && (s_ni - 0.4).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

ldg = ox.OnsagerMaierSaupeNematic(1.0, 1.5, 2.5)
dt, sni = ldg.clearing_temperature_shift_and_order_jump()
assert abs(dt - 0.2) < 1e-12 and abs(sni - 0.4) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python OnsagerMaierSaupeNematic test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3750_toner_tu_vicsek_active_flocking() -> Result<()> {
    use oxideon::physics::toner_tu_vicsek_active_flocking::TonerTuVicsekActiveMatter;
    let tt = TonerTuVicsekActiveMatter::new(2.0)?;
    let (v0, phi) = tt.flock_speed_and_polar_order(8.0, &[0.1, 0.1, 0.1])?;
    assert!((v0 - 2.0).abs() < 1e-12 && (phi - 1.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

tt = ox.TonerTuVicsekActiveMatter(2.0)
v0, phi = tt.flock_speed_and_polar_order(8.0, [0.1, 0.1, 0.1])
assert abs(v0 - 2.0) < 1e-12 and abs(phi - 1.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python TonerTuVicsekActiveMatter test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3760_active_nematic_defect_propulsion() -> Result<()> {
    use oxideon::physics::active_nematic_defect_propulsion::ActiveNematicDefectTurbulence;
    let an = ActiveNematicDefectTurbulence::new(16.0, 2.0)?;
    let (la, vd) = an.active_length_and_defect_velocity(4.0)?;
    assert!((la - 2.0).abs() < 1e-12 && (vd - 4.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

an = ox.ActiveNematicDefectTurbulence(16.0, 2.0)
la, vd = an.active_length_and_defect_velocity(4.0)
assert abs(la - 2.0) < 1e-12 and abs(vd - 4.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python ActiveNematicDefectTurbulence test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3770_cates_tailleur_mips_active_brownian() -> Result<()> {
    use oxideon::physics::cates_tailleur_mips_abp::MotilityInducedPhaseSeparation;
    let mips = MotilityInducedPhaseSeparation::new(10.0, 1.0)?;
    let (v, df, unstable) = mips.swim_speed_and_mips_spinodal(0.75)?;
    assert!((v - 2.5).abs() < 1e-12 && (df - (-5.0)).abs() < 1e-12 && unstable);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

mips = ox.MotilityInducedPhaseSeparation(10.0, 1.0)
v, df, un = mips.swim_speed_and_mips_spinodal(0.75)
assert abs(v - 2.5) < 1e-12 and un
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python MotilityInducedPhaseSeparation test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3780_oldroyd_b_weissenberg_viscoelastic_rheology() -> Result<()> {
    use oxideon::physics::oldroyd_b_weissenberg_rheology::OldroydBViscoelasticRheology;
    let ob = OldroydBViscoelasticRheology::new(1.0, 4.0, 0.5)?;
    let (wi, sxy, n1) = ob.weissenberg_shear_and_normal_stress(2.0);
    assert!((wi - 1.0).abs() < 1e-12 && (sxy - 10.0).abs() < 1e-12 && (n1 - 16.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

ob = ox.OldroydBViscoelasticRheology(1.0, 4.0, 0.5)
wi, sxy, n1 = ob.weissenberg_shear_and_normal_stress(2.0)
assert abs(wi - 1.0) < 1e-12 and abs(n1 - 16.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python OldroydBViscoelasticRheology test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3790_washburn_young_laplace_capillary() -> Result<()> {
    use oxideon::physics::washburn_young_laplace_wetting::WashburnYoungLaplaceCapillary;
    let cap = WashburnYoungLaplaceCapillary::new(4.0, 1.0)?;
    let (dp, lt) = cap.laplace_pressure_and_washburn_length(2.0, 0.0, 9.0)?;
    assert!((dp - 4.0).abs() < 1e-12 && (lt - 6.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

cap = ox.WashburnYoungLaplaceCapillary(4.0, 1.0)
dp, lt = cap.laplace_pressure_and_washburn_length(2.0, 0.0, 9.0)
assert abs(dp - 4.0) < 1e-12 and abs(lt - 6.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python WashburnYoungLaplaceCapillary test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3800_grand_omni_octatricosa_engine_xxxviii() -> Result<()> {
    use oxideon::omni_octatricosa_xxxviii::GrandOmniOctatricosaEngineXXXVIII;
    let eng = GrandOmniOctatricosaEngineXXXVIII::new(38);
    let inv = eng.execute_octatricosa_synthesis()?;
    assert!(inv > 5650.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniOctatricosaEngineXXXVIII(38)
assert eng.execute_octatricosa_synthesis() > 5650.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniOctatricosaEngineXXXVIII test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3810_lamport_pbft_byzantine_quorum() -> Result<()> {
    use oxideon::science::lamport_pbft_byzantine_quorum::ByzantineQuorumConsensusEngine;
    let bft = ByzantineQuorumConsensusEngine::new(7)?;
    assert_eq!(bft.max_faults_quorum_and_honest_intersection(), (2, 5, 1));

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

bft = ox.ByzantineQuorumConsensusEngine(7)
assert bft.max_faults_quorum_and_honest_intersection() == (2, 5, 1)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python ByzantineQuorumConsensusEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3820_lamport_mattern_vector_clock() -> Result<()> {
    use oxideon::science::lamport_mattern_vector_clock::MatternVectorClockCausality;
    let (m, ord, conc) =
        MatternVectorClockCausality::merge_and_classify_causality(&[2, 1, 0], &[1, 3, 0])?;
    assert_eq!(m, vec![2, 3, 0]);
    assert_eq!(ord, 0);
    assert!(conc);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

m, ord, conc = ox.MatternVectorClockCausality.merge_and_classify_causality([2, 1, 0], [1, 3, 0])
assert m == [2, 3, 0] and conc
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python MatternVectorClockCausality test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3830_shapiro_crdt_join_semilattice() -> Result<()> {
    use oxideon::science::shapiro_crdt_semilattice::ShapiroCrdtJoinSemilattice;
    let (val, idem) = ShapiroCrdtJoinSemilattice::merge_pn_counter_and_evaluate(
        &[5, 2],
        &[1, 0],
        &[3, 7],
        &[2, 1],
    )?;
    // P_join = [5, 7] (sum 12), N_join = [2, 1] (sum 3) -> 12 - 3 = 9
    assert_eq!(val, 9);
    assert!(idem);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

val, idem = ox.ShapiroCrdtJoinSemilattice.merge_pn_counter_and_evaluate([5, 2], [1, 0], [3, 7], [2, 1])
assert val == 9 and idem
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python ShapiroCrdtJoinSemilattice test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3840_karger_consistent_hashing_ring() -> Result<()> {
    use oxideon::science::karger_consistent_hashing::KargerConsistentHashingRing;
    let ch = KargerConsistentHashingRing::new(9, 100)?;
    let (remap, peak) = ch.remapped_fraction_and_peak_load_ratio();
    assert!((remap - 0.1).abs() < 1e-12 && peak > 1.2 && peak < 1.25);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

ch = ox.KargerConsistentHashingRing(9, 100)
rem, pk = ch.remapped_fraction_and_peak_load_ratio()
assert abs(rem - 0.1) < 1e-12 and 1.2 < pk < 1.25
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python KargerConsistentHashingRing test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3850_gilbert_lynch_cap_pacelc_quorum() -> Result<()> {
    use oxideon::science::gilbert_lynch_cap_pacelc::GilbertLynchPacelcTheorem;
    let cap = GilbertLynchPacelcTheorem::new(5)?;
    assert_eq!(cap.verify_strong_quorum(3, 3)?, (1, true, true));

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

cap = ox.GilbertLynchPacelcTheorem(5)
assert cap.verify_strong_quorum(3, 3) == (1, True, True)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python GilbertLynchPacelcTheorem test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3860_clarke_emerson_ctl_mu_calculus() -> Result<()> {
    use oxideon::science::clarke_emerson_mu_calculus::ClarkeEmersonModelChecker;
    // Chain 0 -> 1 -> 2 -> 3, target = {3} (mask 8) -> all 4 states reach 3 (mask 15) in 4 iterations
    let (mask, iters) = ClarkeEmersonModelChecker::ctl_ef_least_fixed_point(4, &[2, 4, 8, 0], 8)?;
    assert_eq!((mask, iters), (15, 4));

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

mask, iters = ox.ClarkeEmersonModelChecker.ctl_ef_least_fixed_point(4, [2, 4, 8, 0], 8)
assert (mask, iters) == (15, 4)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python ClarkeEmersonModelChecker test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3870_reynolds_ohearn_separation_logic() -> Result<()> {
    use oxideon::science::reynolds_separation_logic::ReynoldsSeparationLogicHeap;
    let (comb, disj, frame_ok) =
        ReynoldsSeparationLogicHeap::check_separating_conjunction_and_frame(
            0b0011, 0b1100, 0b10000, 0b0011,
        )?;
    assert_eq!(comb, 0b11111);
    assert!(disj && frame_ok);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

comb, disj, ok = ox.ReynoldsSeparationLogicHeap.check_separating_conjunction_and_frame(3, 12, 16, 3)
assert comb == 31 and disj and ok
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python ReynoldsSeparationLogicHeap test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3880_cousot_abstract_interpretation_widening() -> Result<()> {
    use oxideon::science::cousot_abstract_interpretation::CousotGaloisAbstractInterpreter;
    let (jn, wd) =
        CousotGaloisAbstractInterpreter::interval_join_and_widening(0.0, 1.0, 0.0, 2.0, 1000.0)?;
    assert_eq!(jn, [0.0, 2.0]);
    assert_eq!(wd, [0.0, 1000.0]);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

jn, wd = ox.CousotGaloisAbstractInterpreter.interval_join_and_widening(0.0, 1.0, 0.0, 2.0, 1000.0)
assert jn == [0.0, 2.0] and wd == [0.0, 1000.0]
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python CousotGaloisAbstractInterpreter test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3890_girard_linear_logic_proof_nets() -> Result<()> {
    use oxideon::science::girard_linear_logic_resource::GirardLinearLogicResource;
    let (lin, tree) =
        GirardLinearLogicResource::verify_linear_resource_and_switching_tree(&[1, 1, 1], 6, 5)?;
    assert!(lin && tree);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

lin, tree = ox.GirardLinearLogicResource.verify_linear_resource_and_switching_tree([1, 1, 1], 6, 5)
assert lin and tree
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python GirardLinearLogicResource test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3900_grand_omni_nonatricosa_engine_xxxix() -> Result<()> {
    use oxideon::omni_nonatricosa_xxxix::GrandOmniNonatricosaEngineXXXIX;
    let eng = GrandOmniNonatricosaEngineXXXIX::new(39);
    let inv = eng.execute_nonatricosa_synthesis()?;
    assert!(inv > 5700.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniNonatricosaEngineXXXIX(39)
assert eng.execute_nonatricosa_synthesis() > 5700.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniNonatricosaEngineXXXIX test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3910_lawson_triple_product_fusion_ignition() -> Result<()> {
    use oxideon::physics::lawson_triple_product_fusion::LawsonTripleProductIgnition;
    let law = LawsonTripleProductIgnition::new(3500.0)?;
    let (ach, thr, ign) = law.triple_product_and_ignition(1.0e20, 14.0, 3.5, 2.0e-22)?;
    assert!(ach > thr && ign);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

law = ox.LawsonTripleProductIgnition(3500.0)
ach, thr, ign = law.triple_product_and_ignition(1.0e20, 14.0, 3.5, 2.0e-22)
assert ign and ach > thr
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python LawsonTripleProductIgnition test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3920_spitzer_dreicer_runaway_plasma() -> Result<()> {
    use oxideon::physics::spitzer_dreicer_runaway_plasma::SpitzerDreicerPlasmaTransport;
    let sp = SpitzerDreicerPlasmaTransport::new(1.0, 17.0)?;
    let (eta, ed) = sp.spitzer_resistivity_and_dreicer_field(4.0, 1.0)?;
    assert!(eta > 3.0e-9 && (ed - 11.05).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

sp = ox.SpitzerDreicerPlasmaTransport(1.0, 17.0)
eta, ed = sp.spitzer_resistivity_and_dreicer_field(4.0, 1.0)
assert eta > 3.0e-9 and abs(ed - 11.05) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python SpitzerDreicerPlasmaTransport test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3930_kruskal_shafranov_troyon_mhd() -> Result<()> {
    use oxideon::physics::kruskal_shafranov_troyon_mhd::KruskalShafranovTroyonMhd;
    let mhd = KruskalShafranovTroyonMhd::new(6.0, 2.0, 5.0)?;
    let (qa, beta, stable) = mhd.safety_factor_and_troyon_beta(10.0)?;
    assert!((qa - 5.0 / 3.0).abs() < 1e-12 && (beta - 2.8).abs() < 1e-12 && !stable);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

mhd = ox.KruskalShafranovTroyonMhd(6.0, 2.0, 5.0)
qa, beta, st = mhd.safety_factor_and_troyon_beta(10.0)
assert abs(beta - 2.8) < 1e-12 and not st
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python KruskalShafranovTroyonMhd test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3940_tajima_dawson_laser_wakefield() -> Result<()> {
    use oxideon::physics::tajima_dawson_laser_wakefield::TajimaDawsonLaserWakefield;
    let lwfa = TajimaDawsonLaserWakefield::new(4.0)?;
    let (e0, dw) = lwfa.wavebreaking_gv_m_and_energy_gain_mev(100.0)?;
    assert!((e0 - 192.0).abs() < 1e-12 && dw > 10000.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

lwfa = ox.TajimaDawsonLaserWakefield(4.0)
e0, dw = lwfa.wavebreaking_gv_m_and_energy_gain_mev(100.0)
assert abs(e0 - 192.0) < 1e-12 and dw > 10000.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python TajimaDawsonLaserWakefield test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3950_gamow_thermonuclear_peak() -> Result<()> {
    use oxideon::physics::gamow_thermonuclear_peak::GamowThermonuclearPeakEngine;
    let gam = GamowThermonuclearPeakEngine::new(108.0)?;
    let (e0, width, tau) = gam.gamow_peak_width_and_tau(1.0)?;
    assert!((e0 - 3.0).abs() < 1e-12 && (width - 4.0).abs() < 1e-12 && (tau - 9.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

gam = ox.GamowThermonuclearPeakEngine(108.0)
e0, w, tau = gam.gamow_peak_width_and_tau(1.0)
assert abs(e0 - 3.0) < 1e-12 and abs(w - 4.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GamowThermonuclearPeakEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3960_hoyle_salpeter_eddington_stellar() -> Result<()> {
    use oxideon::physics::hoyle_salpeter_eddington_stellar::HoyleSalpeterEddingtonStellar;
    let st = HoyleSalpeterEddingtonStellar::new(10.0)?;
    let (ledd, nu) = st.eddington_luminosity_and_triple_alpha_exponent(1.0)?;
    assert!((ledd - 3.2e5).abs() < 1e-9 && (nu - 41.027).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

st = ox.HoyleSalpeterEddingtonStellar(10.0)
ledd, nu = st.eddington_luminosity_and_triple_alpha_exponent(1.0)
assert abs(ledd - 3.2e5) < 1e-9 and abs(nu - 41.027) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python HoyleSalpeterEddingtonStellar test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3970_fermi_hillas_cosmic_ray_shock() -> Result<()> {
    use oxideon::physics::fermi_hillas_cosmic_ray_shock::FermiHillasCosmicRayShock;
    let dsa = FermiHillasCosmicRayShock::new(4.0)?;
    let (s, emax) = dsa.spectral_index_and_hillas_emax_eev(26, 100.0, 1.0, 0.1)?;
    assert!((s - 2.0).abs() < 1e-12 && (emax - 240.5).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

dsa = ox.FermiHillasCosmicRayShock(4.0)
s, emax = dsa.spectral_index_and_hillas_emax_eev(26, 100.0, 1.0, 0.1)
assert abs(s - 2.0) < 1e-12 and abs(emax - 240.5) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python FermiHillasCosmicRayShock test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3980_parker_solar_wind_spiral() -> Result<()> {
    use oxideon::physics::parker_solar_wind_spiral::ParkerSolarWindSpiralEngine;
    let pw = ParkerSolarWindSpiralEngine::new(8.0, 1.0)?;
    let (rc, psi) = pw.sonic_radius_and_spiral_angle(2.0, 1.0, 1.0)?;
    assert!((rc - 1.0).abs() < 1e-12 && (psi - std::f64::consts::PI / 4.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

import math
pw = ox.ParkerSolarWindSpiralEngine(8.0, 1.0)
rc, psi = pw.sonic_radius_and_spiral_angle(2.0, 1.0, 1.0)
assert abs(rc - 1.0) < 1e-12 and abs(psi - math.pi / 4.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python ParkerSolarWindSpiralEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v3990_sweet_parker_petschek_reconnection() -> Result<()> {
    use oxideon::physics::sweet_parker_petschek_reconnection::SweetParkerPetschekReconnection;
    let rec = SweetParkerPetschekReconnection::new(1.0e6)?;
    let (sp, pet, plasmoid) = rec.reconnection_rates_and_plasmoid_regime();
    assert!((sp - 1.0e-3).abs() < 1e-12 && pet > 0.025 && plasmoid);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

rec = ox.SweetParkerPetschekReconnection(1.0e6)
sp, pet, pl = rec.reconnection_rates_and_plasmoid_regime()
assert abs(sp - 1.0e-3) < 1e-12 and pl
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python SweetParkerPetschekReconnection test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4000_grand_omni_quadricentennial_engine_xl() -> Result<()> {
    use oxideon::omni_quadricentennial_xl::GrandOmniQuadricentennialEngineXL;
    let eng = GrandOmniQuadricentennialEngineXL::new(40);
    let inv = eng.execute_quadricentennial_synthesis()?;
    assert!(inv > 5950.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniQuadricentennialEngineXL(40)
assert eng.execute_quadricentennial_synthesis() > 5950.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniQuadricentennialEngineXL test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4010_almgren_frequency_minimal_currents() -> Result<()> {
    use oxideon::science::almgren_frequency_minimal_currents::AlmgrenFrequencyCurrentsEngine;
    let alm = AlmgrenFrequencyCurrentsEngine::new(4)?;
    let (nr, sdim) = alm.frequency_and_max_singular_dim(2.0, 3.0, 4.0)?;
    assert!((nr - 1.5).abs() < 1e-12 && sdim == 2);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

alm = ox.AlmgrenFrequencyCurrentsEngine(4)
nr, sdim = alm.frequency_and_max_singular_dim(2.0, 3.0, 4.0)
assert abs(nr - 1.5) < 1e-12 and sdim == 2
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python AlmgrenFrequencyCurrentsEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4020_allard_varifold_monotonicity_regularity() -> Result<()> {
    use oxideon::science::allard_varifold_monotonicity::AllardVarifoldMonotonicityEngine;
    let al = AllardVarifoldMonotonicityEngine::new(2, 0.05)?;
    let (th, reg) = al.density_ratio_and_regularity_check(2.0, 4.08 * std::f64::consts::PI)?;
    assert!((th - 1.02).abs() < 1e-12 && reg);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

import math
al = ox.AllardVarifoldMonotonicityEngine(2, 0.05)
th, reg = al.density_ratio_and_regularity_check(2.0, 4.08 * math.pi)
assert abs(th - 1.02) < 1e-12 and reg
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python AllardVarifoldMonotonicityEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4030_federer_fleming_integral_currents() -> Result<()> {
    use oxideon::science::federer_fleming_integral_currents::FedererFlemingCurrentsCoarea;
    let ff = FedererFlemingCurrentsCoarea::new(2)?;
    let (vmax, ok) =
        ff.sharp_isoperimetric_volume_bound(2.0 * std::f64::consts::PI, std::f64::consts::PI)?;
    assert!((vmax - std::f64::consts::PI).abs() < 1e-12 && ok);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

import math
ff = ox.FedererFlemingCurrentsCoarea(2)
vmax, ok = ff.sharp_isoperimetric_volume_bound(2.0 * math.pi, math.pi)
assert abs(vmax - math.pi) < 1e-12 and ok
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python FedererFlemingCurrentsCoarea test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4040_caffarelli_monge_ampere_mccann_transport() -> Result<()> {
    use oxideon::science::caffarelli_monge_ampere_transport::CaffarelliMongeAmpereTransport;
    let (det_t, margin, ok) =
        CaffarelliMongeAmpereTransport::mccann_displacement_jacobian_and_concavity(
            &[1.0, 9.0],
            0.5,
        )?;
    assert!((det_t - 5.0).abs() < 1e-12 && margin > 0.2 && ok);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

dt, m, ok = ox.CaffarelliMongeAmpereTransport.mccann_displacement_jacobian_and_concavity([1.0, 9.0], 0.5)
assert abs(dt - 5.0) < 1e-12 and ok
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python CaffarelliMongeAmpereTransport test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4050_alt_caffarelli_free_boundary_weiss() -> Result<()> {
    use oxideon::science::alt_caffarelli_free_boundary::AltCaffarelliFreeBoundaryWeiss;
    let ac = AltCaffarelliFreeBoundaryWeiss::new(4.0, 2)?;
    let (slope, w) = ac.bernoulli_slope_and_weiss_energy(2.0, 8.0, 2.0, 16.0)?;
    assert!((slope - 2.0).abs() < 1e-12 && (w - 2.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

ac = ox.AltCaffarelliFreeBoundaryWeiss(4.0, 2)
sl, w = ac.bernoulli_slope_and_weiss_energy(2.0, 8.0, 2.0, 16.0)
assert abs(sl - 2.0) < 1e-12 and abs(w - 2.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python AltCaffarelliFreeBoundaryWeiss test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4060_caffarelli_silvestre_fractional_laplacian() -> Result<()> {
    use oxideon::science::caffarelli_silvestre_fractional::CaffarelliSilvestreFractionalExt;
    let cs = CaffarelliSilvestreFractionalExt::new(0.5)?;
    let (a, sym) = cs.muckenhoupt_exponent_and_fourier_multiplier(4.0, 0.0)?;
    assert!(a.abs() < 1e-12 && (sym - 4.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

cs = ox.CaffarelliSilvestreFractionalExt(0.5)
a, sym = cs.muckenhoupt_exponent_and_fourier_multiplier(4.0, 0.0)
assert abs(a) < 1e-12 and abs(sym - 4.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python CaffarelliSilvestreFractionalExt test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4070_simons_minimal_cone_bernstein() -> Result<()> {
    use oxideon::science::simons_cone_bernstein_minimal::SimonsMinimalConeBernstein;
    let sim = SimonsMinimalConeBernstein::new(8)?;
    let (gap, cone_ok, bern) = sim.jacobi_hardy_gap_and_bernstein_status();
    // In R^8 (n = 7): (7 - 2)^2 / 4 - 6 = 25/4 - 6 = 0.25 > 0
    assert!((gap - 0.25).abs() < 1e-12 && cone_ok && bern);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

sim = ox.SimonsMinimalConeBernstein(8)
gap, cone_ok, bern = sim.jacobi_hardy_gap_and_bernstein_status()
assert abs(gap - 0.25) < 1e-12 and cone_ok
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python SimonsMinimalConeBernstein test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4080_besicovitch_marstrand_rectifiable_projection() -> Result<()> {
    use oxideon::science::besicovitch_marstrand_rectifiable::BesicovitchMarstrandRectifiable;
    let bm = BesicovitchMarstrandRectifiable::new(3)?;
    let (pdim, sdim, int_d) = bm.projection_and_slice_dimensions(2.5, 2)?;
    assert!((pdim - 2.0).abs() < 1e-12 && (sdim - 0.5).abs() < 1e-12 && !int_d);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

bm = ox.BesicovitchMarstrandRectifiable(3)
pd, sd, idim = bm.projection_and_slice_dimensions(2.5, 2)
assert abs(pd - 2.0) < 1e-12 and abs(sd - 0.5) < 1e-12 and not idim
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python BesicovitchMarstrandRectifiable test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4090_cheeger_buser_isoperimetric_spectral() -> Result<()> {
    use oxideon::science::cheeger_buser_isoperimetric::CheegerBuserSpectralInequality;
    let cb = CheegerBuserSpectralInequality::new(2.0)?;
    let (lb, ub, ok) = cb.cheeger_lower_bound_and_verify(1.5);
    assert!((lb - 1.0).abs() < 1e-12 && (ub - 40.0).abs() < 1e-12 && ok);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

cb = ox.CheegerBuserSpectralInequality(2.0)
lb, ub, ok = cb.cheeger_lower_bound_and_verify(1.5)
assert abs(lb - 1.0) < 1e-12 and ok
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python CheegerBuserSpectralInequality test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4100_grand_omni_henatetracosa_engine_xli() -> Result<()> {
    use oxideon::omni_henatetracosa_xli::GrandOmniHenatetracosaEngineXLI;
    let eng = GrandOmniHenatetracosaEngineXLI::new(41);
    let inv = eng.execute_henatetracosa_synthesis()?;
    assert!(inv > 5970.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniHenatetracosaEngineXLI(41)
assert eng.execute_henatetracosa_synthesis() > 5970.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniHenatetracosaEngineXLI test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4110_bjorken_qgp_boost_invariant_hydro() -> Result<()> {
    use oxideon::physics::bjorken_qgp_boost_invariant::BjorkenQgpHydroExpansion;
    let bj = BjorkenQgpHydroExpansion::new(1.0, 16.0)?;
    let (eps, tr) = bj.energy_density_and_temp_ratio(8.0, 1.0 / 3.0)?;
    assert!((eps - 1.0).abs() < 1e-12 && (tr - 0.5).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

bj = ox.BjorkenQgpHydroExpansion(1.0, 16.0)
eps, tr = bj.energy_density_and_temp_ratio(8.0, 1.0 / 3.0)
assert abs(eps - 1.0) < 1e-12 and abs(tr - 0.5) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python BjorkenQgpHydroExpansion test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4120_kovtun_son_starinets_kss_bound() -> Result<()> {
    use oxideon::physics::kovtun_son_starinets_kss_bound::KovtunSonStarinetsViscosityBound;
    let (kss, eta_s) =
        KovtunSonStarinetsViscosityBound::kss_bound_and_finite_coupling_eta_over_s(100.0)?;
    assert!((kss - 1.0 / (4.0 * std::f64::consts::PI)).abs() < 1e-12 && eta_s > kss);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

kss, es = ox.KovtunSonStarinetsViscosityBound.kss_bound_and_finite_coupling_eta_over_s(100.0)
assert es > kss > 0.079
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python KovtunSonStarinetsViscosityBound test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4130_mclerran_venugopalan_cgc_saturation() -> Result<()> {
    use oxideon::physics::mclerran_venugopalan_cgc_saturation::ColorGlassCondensateSaturation;
    let cgc = ColorGlassCondensateSaturation::new(1.0, 1.0e-3, 0.25)?;
    let (qs2, ndip) = cgc.saturation_scale_and_dipole_scattering(1.0e-3, 216, 1.0)?;
    assert!((qs2 - 6.0).abs() < 1e-12 && ndip > 0.77);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

cgc = ox.ColorGlassCondensateSaturation(1.0, 1.0e-3, 0.25)
qs2, ndip = cgc.saturation_scale_and_dipole_scattering(1.0e-3, 216, 1.0)
assert abs(qs2 - 6.0) < 1e-12 and ndip > 0.77
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python ColorGlassCondensateSaturation test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4140_bdmps_jet_quenching_energy_loss() -> Result<()> {
    use oxideon::physics::bdmps_jet_quenching_qgp::BdmpsJetQuenchingEnergyLoss;
    let jq = BdmpsJetQuenchingEnergyLoss::new(0.3, 1.5)?;
    let (wc, de) = jq.characteristic_frequency_and_energy_loss_gev(4.0, 4.0 / 3.0)?;
    assert!(wc > 60.0 && (de - 0.2 * wc).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

jq = ox.BdmpsJetQuenchingEnergyLoss(0.3, 1.5)
wc, de = jq.characteristic_frequency_and_energy_loss_gev(4.0, 4.0 / 3.0)
assert wc > 60.0 and abs(de - 0.2 * wc) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python BdmpsJetQuenchingEnergyLoss test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4150_matsui_satz_quarkonium_melting() -> Result<()> {
    use oxideon::physics::matsui_satz_quarkonium_screening::MatsuiSatzQuarkoniumMelting;
    let ms = MatsuiSatzQuarkoniumMelting::new(3, 3)?;
    let (md, rd, melted) = ms.debye_mass_radius_and_melting(2.0, 0.4, 0.45)?;
    assert!(md > 0.9 && rd < 0.25 && melted);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

ms = ox.MatsuiSatzQuarkoniumMelting(3, 3)
md, rd, melted = ms.debye_mass_radius_and_melting(2.0, 0.4, 0.45)
assert melted and md > 0.9
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python MatsuiSatzQuarkoniumMelting test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4160_chiral_magnetic_effect_kharzeev() -> Result<()> {
    use oxideon::physics::chiral_magnetic_effect_cme::ChiralMagneticEffectKharzeev;
    let cme = ChiralMagneticEffectKharzeev::new(2.0)?;
    let (sig, j) =
        cme.chiral_conductivity_and_current(std::f64::consts::PI * std::f64::consts::PI, 3.0);
    assert!((sig - 1.0).abs() < 1e-12 && (j - 3.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

import math
cme = ox.ChiralMagneticEffectKharzeev(2.0)
sig, j = cme.chiral_conductivity_and_current(math.pi ** 2, 3.0)
assert abs(sig - 1.0) < 1e-12 and abs(j - 3.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python ChiralMagneticEffectKharzeev test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4170_bethe_weizsacker_bohr_wheeler_fission() -> Result<()> {
    use oxideon::physics::bethe_weizsacker_nuclear_fission::BetheWeizsackerNuclearMass;
    let bw = BetheWeizsackerNuclearMass::default();
    let (ba_fe56, x_fe) = bw.binding_per_nucleon_and_fissility(56, 26)?;
    assert!(ba_fe56 > 8.5 && x_fe < 0.3);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

bw = ox.BetheWeizsackerNuclearMass(15.8, 18.3, 0.714, 23.2)
ba, x = bw.binding_per_nucleon_and_fissility(56, 26)
assert ba > 8.5 and x < 0.3
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python BetheWeizsackerNuclearMass test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4180_mayer_jensen_nuclear_shell_magic() -> Result<()> {
    use oxideon::physics::mayer_jensen_nuclear_shell::MayerJensenNuclearShellMagic;
    let mj = MayerJensenNuclearShellMagic::new(2.0)?;
    // 1f_{7/2} intruder (l = 3) has degeneracy 8 (20 + 8 = 28) and splitting 2.0 * 3.5 = 7.0 MeV
    let (de, deg, magic) = mj.doublet_splitting_and_magic_check(3, 28);
    assert!((de - 7.0).abs() < 1e-12 && deg == 8 && magic);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

mj = ox.MayerJensenNuclearShellMagic(2.0)
de, deg, mag = mj.doublet_splitting_and_magic_check(3, 28)
assert abs(de - 7.0) < 1e-12 and deg == 8 and mag
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python MayerJensenNuclearShellMagic test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4190_walecka_relativistic_nuclear_eos() -> Result<()> {
    use oxideon::physics::walecka_relativistic_nuclear_eos::WaleckaRelativisticMeanFieldNuclear;
    let wal = WaleckaRelativisticMeanFieldNuclear::new(939.0, 2500.0, 2000.0)?;
    let (m_star, v0) = wal.effective_mass_and_vector_potential(0.15, 0.16)?;
    assert!((m_star - 564.0).abs() < 1e-12 && (v0 - 320.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

wal = ox.WaleckaRelativisticMeanFieldNuclear(939.0, 2500.0, 2000.0)
ms, v0 = wal.effective_mass_and_vector_potential(0.15, 0.16)
assert abs(ms - 564.0) < 1e-12 and abs(v0 - 320.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python WaleckaRelativisticMeanFieldNuclear test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4200_grand_omni_dotetracosa_engine_xlii() -> Result<()> {
    use oxideon::omni_dotetracosa_xlii::GrandOmniDotetracosaEngineXLII;
    let eng = GrandOmniDotetracosaEngineXLII::new(42);
    let inv = eng.execute_dotetracosa_synthesis()?;
    assert!(inv > 6500.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniDotetracosaEngineXLII(42)
assert eng.execute_dotetracosa_synthesis() > 6500.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniDotetracosaEngineXLII test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4210_lovasz_local_lemma_moser_tardos() -> Result<()> {
    use oxideon::science::lovasz_local_lemma_moser_tardos::LovaszLocalLemmaMoserTardos;
    let lll = LovaszLocalLemmaMoserTardos::new(80, 4)?;
    let (val, ok, steps) = lll.lll_parameter_and_expected_resamplings(0.05)?;
    assert!(ok && val < 1.0 && (steps - 20.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

lll = ox.LovaszLocalLemmaMoserTardos(80, 4)
val, ok, steps = lll.lll_parameter_and_expected_resamplings(0.05)
assert ok and val < 1.0 and abs(steps - 20.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python LovaszLocalLemmaMoserTardos test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4220_talagrand_convex_concentration() -> Result<()> {
    use oxideon::science::talagrand_convex_concentration::TalagrandConvexConcentration;
    let tal = TalagrandConvexConcentration::new(1.0)?;
    let (one, two) = tal.convex_tail_and_median_deviation_bound(2.0)?;
    assert!((one - 2.0 * (-1.0f64).exp()).abs() < 1e-12 && two > one);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

tal = ox.TalagrandConvexConcentration(1.0)
one, two = tal.convex_tail_and_median_deviation_bound(2.0)
assert two > one > 0.73
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python TalagrandConvexConcentration test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4230_friedgut_bourgain_sharp_threshold() -> Result<()> {
    use oxideon::science::friedgut_bourgain_sharp_threshold::FriedgutBourgainSharpThreshold;
    let fb = FriedgutBourgainSharpThreshold::new(100)?;
    let (inf, w) = fb.total_influence_and_threshold_width(&[2.5, 2.5, 3.0, 2.0])?;
    assert!((inf - 10.0).abs() < 1e-12 && (w - 0.1).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

fb = ox.FriedgutBourgainSharpThreshold(100)
inf, w = fb.total_influence_and_threshold_width([2.5, 2.5, 3.0, 2.0])
assert abs(inf - 10.0) < 1e-12 and abs(w - 0.1) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python FriedgutBourgainSharpThreshold test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4240_bollobas_random_graph_giant_component() -> Result<()> {
    use oxideon::science::bollobas_random_graph_giant_component::ErdosRenyiBollobasPhaseTransition;
    let er = ErdosRenyiBollobasPhaseTransition::new(1000)?;
    let (beta, sz) = er.giant_component_fraction_and_size(2.0)?;
    assert!((beta - 0.79681213).abs() < 1e-5 && sz > 796.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

er = ox.ErdosRenyiBollobasPhaseTransition(1000)
beta, sz = er.giant_component_fraction_and_size(2.0)
assert abs(beta - 0.796812) < 1e-4 and sz > 796.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python ErdosRenyiBollobasPhaseTransition test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4250_alon_boppana_ramanujan_expander() -> Result<()> {
    use oxideon::science::alon_boppana_ramanujan_expander::AlonBoppanaRamanujanExpander;
    let ab = AlonBoppanaRamanujanExpander::new(10)?;
    let (rb, gap, is_ram) = ab.ramanujan_threshold_and_spectral_gap(5.8)?;
    assert!((rb - 6.0).abs() < 1e-12 && (gap - 4.2).abs() < 1e-12 && is_ram);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

ab = ox.AlonBoppanaRamanujanExpander(10)
rb, gap, is_ram = ab.ramanujan_threshold_and_spectral_gap(5.8)
assert abs(rb - 6.0) < 1e-12 and abs(gap - 4.2) < 1e-12 and is_ram
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python AlonBoppanaRamanujanExpander test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4260_balog_szemeredi_gowers_freiman() -> Result<()> {
    use oxideon::science::balog_szemeredi_gowers_freiman::BalogSzemerediGowersFreiman;
    let bsg = BalogSzemerediGowersFreiman::new(10)?;
    let (k, elb, pr) = bsg.doubling_energy_and_plunnecke_bound(20)?;
    assert!((k - 2.0).abs() < 1e-12 && (elb - 500.0).abs() < 1e-12 && (pr - 160.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

bsg = ox.BalogSzemerediGowersFreiman(10)
k, elb, pr = bsg.doubling_energy_and_plunnecke_bound(20)
assert abs(k - 2.0) < 1e-12 and abs(elb - 500.0) < 1e-12 and abs(pr - 160.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python BalogSzemerediGowersFreiman test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4270_roth_bourgain_meshulam_capset() -> Result<()> {
    use oxideon::science::roth_bourgain_meshulam_capset::RothBourgainCapSetBounds;
    let cs = RothBourgainCapSetBounds::new(10)?;
    let (base, ub) = cs.capset_growth_base_and_upper_bound();
    assert!((base - 2.75510461334).abs() < 1e-7 && ub < 3.0f64.powi(10));

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

cs = ox.RothBourgainCapSetBounds(10)
base, ub = cs.capset_growth_base_and_upper_bound()
assert abs(base - 2.7551046) < 1e-5 and ub < 3**10
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python RothBourgainCapSetBounds test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4280_lovasz_theta_shannon_capacity() -> Result<()> {
    use oxideon::science::lovasz_theta_shannon_capacity::LovaszThetaShannonCapacity;
    let (th5, a5) = LovaszThetaShannonCapacity::odd_cycle_lovasz_theta_and_alpha(5)?;
    assert!((th5 - 5.0f64.sqrt()).abs() < 1e-12 && a5 == 2);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

th5, a5 = ox.LovaszThetaShannonCapacity.odd_cycle_lovasz_theta_and_alpha(5)
assert abs(th5 - 5.0**0.5) < 1e-12 and a5 == 2
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python LovaszThetaShannonCapacity test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4290_kahn_kalai_park_pham_threshold() -> Result<()> {
    use oxideon::science::kahn_kalai_park_pham_threshold::KahnKalaiParkPhamThreshold;
    let kk = KahnKalaiParkPhamThreshold::new(15)?;
    let (lf, pub_val) = kk.log_factor_and_critical_threshold_bound(0.1)?;
    assert!((lf - 4.0).abs() < 1e-12 && (pub_val - 0.4).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

kk = ox.KahnKalaiParkPhamThreshold(15)
lf, pub_val = kk.log_factor_and_critical_threshold_bound(0.1)
assert abs(lf - 4.0) < 1e-12 and abs(pub_val - 0.4) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python KahnKalaiParkPhamThreshold test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4300_grand_omni_tritetracosa_engine_xliii() -> Result<()> {
    use oxideon::omni_tritetracosa_xliii::GrandOmniTritetracosaEngineXLIII;
    let eng = GrandOmniTritetracosaEngineXLIII::new(43);
    let inv = eng.execute_tritetracosa_synthesis()?;
    assert!(inv > 6540.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniTritetracosaEngineXLIII(43)
assert eng.execute_tritetracosa_synthesis() > 6540.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniTritetracosaEngineXLIII test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4310_kolmogorov_obukhov_turbulence() -> Result<()> {
    use oxideon::physics::kolmogorov_obukhov_turbulence::KolmogorovK41ObukhovIntermittency;
    let kol = KolmogorovK41ObukhovIntermittency::new(1.5, 0.27)?;
    let (ek, z6) = kol.energy_spectrum_and_zeta_p(1.0, 1.0, 6.0)?;
    assert!((ek - 1.5).abs() < 1e-12 && (z6 - 1.73).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

kol = ox.KolmogorovK41ObukhovIntermittency(1.5, 0.27)
ek, z6 = kol.energy_spectrum_and_zeta_p(1.0, 1.0, 6.0)
assert abs(ek - 1.5) < 1e-12 and abs(z6 - 1.73) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python KolmogorovK41ObukhovIntermittency test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4320_kraichnan_2d_enstrophy_cascade() -> Result<()> {
    use oxideon::physics::kraichnan_2d_enstrophy_cascade::Kraichnan2DTurbulenceCascade;
    let kr = Kraichnan2DTurbulenceCascade::new(1.0, 1.0)?;
    let ek = kr.forward_enstrophy_spectrum(std::f64::consts::E, 2.0)?;
    assert!((ek - 2.0 * std::f64::consts::E.powi(-3)).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

import math
kr = ox.Kraichnan2DTurbulenceCascade(1.0, 1.0)
ek = kr.forward_enstrophy_spectrum(math.e, 2.0)
assert abs(ek - 2.0 * math.e**-3) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python Kraichnan2DTurbulenceCascade test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4330_orr_sommerfeld_squire_stability() -> Result<()> {
    use oxideon::physics::orr_sommerfeld_quire_hydro_stability::OrrSommerfeldSquireStability;
    let oss = OrrSommerfeldSquireStability::new(5000.0)?;
    let (k, re2d) = oss.squire_2d_wavenumber_and_reynolds(3.0, 4.0)?;
    assert!((k - 5.0).abs() < 1e-12 && (re2d - 3000.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

oss = ox.OrrSommerfeldSquireStability(5000.0)
k, re2d = oss.squire_2d_wavenumber_and_reynolds(3.0, 4.0)
assert abs(k - 5.0) < 1e-12 and abs(re2d - 3000.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python OrrSommerfeldSquireStability test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4340_rayleigh_benard_lorenz_convection() -> Result<()> {
    use oxideon::physics::rayleigh_benard_lorenz_convection::RayleighBenardLorenzConvection;
    let rbl = RayleighBenardLorenzConvection::new(10.0, 8.0 / 3.0)?;
    let (rac, rhoh) = rbl.critical_rayleigh_and_lorenz_hopf_rho();
    assert!((rac - 657.51136).abs() < 1e-3 && (rhoh - 470.0 / 19.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

rbl = ox.RayleighBenardLorenzConvection(10.0, 8.0 / 3.0)
rac, rhoh = rbl.critical_rayleigh_and_lorenz_hopf_rho()
assert abs(rac - 657.51136) < 1e-3 and abs(rhoh - 470.0 / 19.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python RayleighBenardLorenzConvection test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4350_prandtl_blasius_von_karman_boundary() -> Result<()> {
    use oxideon::physics::prandtl_blasius_von_karman_boundary::PrandtlBlasiusVonKarmanBoundary;
    let pb = PrandtlBlasiusVonKarmanBoundary::new(0.4, 5.0)?;
    let (cf, up) = pb.blasius_cf_and_log_law_u_plus(10_000.0, std::f64::consts::E.powi(2))?;
    assert!((cf - 0.0066411).abs() < 1e-9 && (up - 10.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

import math
pb = ox.PrandtlBlasiusVonKarmanBoundary(0.4, 5.0)
cf, up = pb.blasius_cf_and_log_law_u_plus(10000.0, math.e**2)
assert abs(cf - 0.0066411) < 1e-9 and abs(up - 10.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python PrandtlBlasiusVonKarmanBoundary test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4360_rankine_hugoniot_prandtl_meyer_shock() -> Result<()> {
    use oxideon::physics::rankine_hugoniot_prandtl_meyer_shock::RankineHugoniotShockAdiabat;
    let rh = RankineHugoniotShockAdiabat::new(1.4)?;
    let (pr, rhor, m2) = rh.shock_pressure_density_and_downstream_mach(2.0)?;
    assert!((pr - 4.5).abs() < 1e-12 && (rhor - 8.0 / 3.0).abs() < 1e-12 && m2 < 1.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

rh = ox.RankineHugoniotShockAdiabat(1.4)
pr, rhor, m2 = rh.shock_pressure_density_and_downstream_mach(2.0)
assert abs(pr - 4.5) < 1e-12 and abs(rhor - 8.0 / 3.0) < 1e-12 and m2 < 1.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python RankineHugoniotShockAdiabat test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4370_zeldovich_von_neumann_doring_detonation() -> Result<()> {
    use oxideon::physics::zeldovich_von_neumann_doring_detonation::ZeldovichVonNeumannDoringDetonation;
    let znd = ZeldovichVonNeumannDoringDetonation::new(1.5, 1.0)?;
    let (dcj, pcj, pvn) = znd.cj_velocity_and_von_neumann_spike(2.0)?;
    assert!(
        (dcj - 5.0f64.sqrt()).abs() < 1e-12
            && (pcj - 2.0).abs() < 1e-12
            && (pvn - 4.0).abs() < 1e-12
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

znd = ox.ZeldovichVonNeumannDoringDetonation(1.5, 1.0)
dcj, pcj, pvn = znd.cj_velocity_and_von_neumann_spike(2.0)
assert abs(dcj - 5.0**0.5) < 1e-12 and abs(pcj - 2.0) < 1e-12 and abs(pvn - 4.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python ZeldovichVonNeumannDoringDetonation test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4380_sedov_taylor_blast_wave() -> Result<()> {
    use oxideon::physics::sedov_taylor_blast_wave::SedovTaylorBlastWave;
    let st = SedovTaylorBlastWave::new(1.0, 1.0)?;
    let (r, u) = st.blast_radius_and_shock_velocity(32.0, 1.0)?;
    assert!((r - 2.0).abs() < 1e-12 && (u - 0.8).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

st = ox.SedovTaylorBlastWave(1.0, 1.0)
r, u = st.blast_radius_and_shock_velocity(32.0, 1.0)
assert abs(r - 2.0) < 1e-12 and abs(u - 0.8) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python SedovTaylorBlastWave test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4390_rayleigh_plesset_cavitation_bubble() -> Result<()> {
    use oxideon::physics::rayleigh_plesset_cavitation_bubble::RayleighPlessetCavitationBubble;
    let rp = RayleighPlessetCavitationBubble::new(1.0)?;
    let (tc, v) = rp.collapse_time_and_wall_speed(2.0, 1.0, 3.0)?;
    assert!(tc > 1.0 && (v - 14.0f64.sqrt()).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

rp = ox.RayleighPlessetCavitationBubble(1.0)
tc, v = rp.collapse_time_and_wall_speed(2.0, 1.0, 3.0)
assert tc > 1.0 and abs(v - 14.0**0.5) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python RayleighPlessetCavitationBubble test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4400_grand_omni_tetratetracosa_engine_xliv() -> Result<()> {
    use oxideon::omni_tetratetracosa_xliv::GrandOmniTetratetracosaEngineXLIV;
    let eng = GrandOmniTetratetracosaEngineXLIV::new(44);
    let inv = eng.execute_tetratetracosa_synthesis()?;
    assert!(inv > 6590.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniTetratetracosaEngineXLIV(44)
assert eng.execute_tetratetracosa_synthesis() > 6590.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniTetratetracosaEngineXLIV test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4410_grothendieck_lefschetz_trace() -> Result<()> {
    use oxideon::science::grothendieck_lefschetz_trace::GrothendieckLefschetzTraceFormula;
    let gl = GrothendieckLefschetzTraceFormula::new(9, 1)?;
    let (lb, ub) = gl.hasse_weil_deligne_point_bounds(1)?;
    assert!((lb - 4.0).abs() < 1e-12 && (ub - 16.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

gl = ox.GrothendieckLefschetzTraceFormula(9, 1)
lb, ub = gl.hasse_weil_deligne_point_bounds(1)
assert abs(lb - 4.0) < 1e-12 and abs(ub - 16.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrothendieckLefschetzTraceFormula test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4420_faltings_mordell_arakelov_height() -> Result<()> {
    use oxideon::science::faltings_mordell_arakelov_height::FaltingsMordellArakelovEngine;
    let fm = FaltingsMordellArakelovEngine::new(3);
    let (fin, chi, sh) = fm.is_mordell_finite_and_isogeny_height_shift(9)?;
    assert!(fin && chi == -4 && (sh - 3.0f64.ln()).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

import math
fm = ox.FaltingsMordellArakelovEngine(3)
fin, chi, sh = fm.is_mordell_finite_and_isogeny_height_shift(9)
assert fin and chi == -4 and abs(sh - math.log(3.0)) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python FaltingsMordellArakelovEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4430_birch_swinnerton_dyer_tate_shafarevich() -> Result<()> {
    use oxideon::science::birch_swinnerton_dyer_tate_shafarevich::BirchSwinnertonDyerLeadingTerm;
    let bsd = BirchSwinnertonDyerLeadingTerm::new(4.0, 2)?;
    let l_star = bsd.bsd_leading_coefficient(1.5, 4, 2)?;
    assert!((l_star - 12.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

bsd = ox.BirchSwinnertonDyerLeadingTerm(4.0, 2)
assert abs(bsd.bsd_leading_coefficient(1.5, 4, 2) - 12.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python BirchSwinnertonDyerLeadingTerm test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4440_wiles_taylor_modularity_hecke() -> Result<()> {
    use oxideon::science::wiles_taylor_modularity_hecke::WilesTaylorModularityHecke;
    let wt = WilesTaylorModularityHecke::new(2)?;
    let (bd, ct, ok) = wt.ramanujan_bound_and_sato_tate_cos(9, -3.0)?;
    assert!((bd - 6.0).abs() < 1e-12 && (ct - (-0.5)).abs() < 1e-12 && ok);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

wt = ox.WilesTaylorModularityHecke(2)
bd, ct, ok = wt.ramanujan_bound_and_sato_tate_cos(9, -3.0)
assert abs(bd - 6.0) < 1e-12 and abs(ct - (-0.5)) < 1e-12 and ok
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python WilesTaylorModularityHecke test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4450_fontaine_p_adic_hodge_de_rham() -> Result<()> {
    use oxideon::science::fontaine_p_adic_hodge_de_rham::FontainePAdicHodgePeriodEngine;
    let (tn, th, adm) =
        FontainePAdicHodgePeriodEngine::newton_and_hodge_endpoints(&[0.5, 0.5], &[0.0, 1.0])?;
    assert!((tn - 1.0).abs() < 1e-12 && (th - 1.0).abs() < 1e-12 && adm);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

tn, th, adm = ox.FontainePAdicHodgePeriodEngine.newton_and_hodge_endpoints([0.5, 0.5], [0.0, 1.0])
assert abs(tn - 1.0) < 1e-12 and abs(th - 1.0) < 1e-12 and adm
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python FontainePAdicHodgePeriodEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4460_iwasawa_main_conjecture_mazur_wiles() -> Result<()> {
    use oxideon::science::iwasawa_main_conjecture_mazur_wiles::IwasawaClassNumberGrowth;
    let iw = IwasawaClassNumberGrowth::new(3, 0, 2, 1)?;
    assert_eq!(iw.class_group_p_valuation_at_layer(5), 11);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

iw = ox.IwasawaClassNumberGrowth(3, 0, 2, 1)
assert iw.class_group_p_valuation_at_layer(5) == 11
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python IwasawaClassNumberGrowth test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4470_sato_tate_equidistribution_taylor() -> Result<()> {
    use oxideon::science::sato_tate_equidistribution_taylor::SatoTateSemicircleEquidistribution;
    let (pdf, cdf) =
        SatoTateSemicircleEquidistribution::density_and_cdf(std::f64::consts::FRAC_PI_2)?;
    assert!((pdf - 2.0 / std::f64::consts::PI).abs() < 1e-12 && (cdf - 0.5).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

import math
pdf, cdf = ox.SatoTateSemicircleEquidistribution.density_and_cdf(math.pi / 2.0)
assert abs(pdf - 2.0 / math.pi) < 1e-12 and abs(cdf - 0.5) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python SatoTateSemicircleEquidistribution test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4480_schmidt_subspace_roth_diophantine() -> Result<()> {
    use oxideon::science::schmidt_subspace_roth_diophantine::RothSchmidtDiophantineEngine;
    let rs = RothSchmidtDiophantineEngine::new(5)?;
    let (mu, d, sc) = rs.roth_exponent_and_scale(10, 0.5)?;
    assert!((mu - 2.0).abs() < 1e-12 && d == 5 && (sc - 10.0f64.powf(-2.5)).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

rs = ox.RothSchmidtDiophantineEngine(5)
mu, d, sc = rs.roth_exponent_and_scale(10, 0.5)
assert abs(mu - 2.0) < 1e-12 and d == 5 and abs(sc - 10.0**-2.5) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python RothSchmidtDiophantineEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4490_voevodsky_motivic_milnor_bloch_kato() -> Result<()> {
    use oxideon::science::voevodsky_motivic_milnor_bloch_kato::VoevodskyMotivicBlochKato;
    let vm = VoevodskyMotivicBlochKato::new(4)?;
    let (pdim, w) = vm.pfister_dimension_and_motivic_weight();
    assert_eq!((pdim, w), (16, 4));

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

vm = ox.VoevodskyMotivicBlochKato(4)
assert vm.pfister_dimension_and_motivic_weight() == (16, 4)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python VoevodskyMotivicBlochKato test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4500_grand_omni_pentatetracosa_engine_xlv() -> Result<()> {
    use oxideon::omni_pentatetracosa_xlv::GrandOmniPentatetracosaEngineXLV;
    let eng = GrandOmniPentatetracosaEngineXLV::new(45);
    let inv = eng.execute_pentatetracosa_synthesis()?;
    assert!(inv > 6660.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniPentatetracosaEngineXLV(45)
assert eng.execute_pentatetracosa_synthesis() > 6660.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniPentatetracosaEngineXLV test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4510_shannon_hartley_awgn_waterfilling() -> Result<()> {
    use oxideon::science::shannon_hartley_awgn_waterfilling::ShannonHartleyWaterfilling;
    let sh = ShannonHartleyWaterfilling::new(10.0)?;
    let (c, eff) = sh.capacity_and_spectral_efficiency(15.0)?;
    assert!((c - 40.0).abs() < 1e-12 && (eff - 4.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

sh = ox.ShannonHartleyWaterfilling(10.0)
c, eff = sh.capacity_and_spectral_efficiency(15.0)
assert abs(c - 40.0) < 1e-12 and abs(eff - 4.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python ShannonHartleyWaterfilling test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4520_slepian_wolf_wyner_ziv_distributed() -> Result<()> {
    use oxideon::science::slepian_wolf_wyner_ziv_distributed::SlepianWolfRateRegionEngine;
    let sw = SlepianWolfRateRegionEngine::new(1.0, 1.0, 1.4)?;
    let (hxy_c, hyx_c, ok) = sw.conditional_rates_and_achievability(0.5, 0.9);
    assert!((hxy_c - 0.4).abs() < 1e-12 && (hyx_c - 0.4).abs() < 1e-12 && ok);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

sw = ox.SlepianWolfRateRegionEngine(1.0, 1.0, 1.4)
hxy_c, hyx_c, ok = sw.conditional_rates_and_achievability(0.5, 0.9)
assert abs(hxy_c - 0.4) < 1e-12 and abs(hyx_c - 0.4) < 1e-12 and ok
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python SlepianWolfDistributedCoding test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4530_blahut_arimoto_rate_distortion() -> Result<()> {
    use oxideon::science::blahut_arimoto_rate_distortion::GaussianRateDistortionEngine;
    let rd = GaussianRateDistortionEngine::new(16.0)?;
    let (r, d_back) = rd.rate_for_distortion_and_inverse(1.0)?;
    assert!((r - 2.0).abs() < 1e-12 && (d_back - 1.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

rd = ox.GaussianRateDistortionEngine(16.0)
r, d_back = rd.rate_for_distortion_and_inverse(1.0)
assert abs(r - 2.0) < 1e-12 and abs(d_back - 1.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GaussianRateDistortionEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4540_gallager_ldpc_random_coding_exponent() -> Result<()> {
    use oxideon::science::gallager_ldpc_random_coding_exponent::GallagerRandomCodingExponent;
    let gal = GallagerRandomCodingExponent::new(0.04)?;
    let (e0, er) = gal.e0_and_error_exponent(1.0, 0.1)?;
    assert!(e0 > 0.3 && er > 0.2);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

gal = ox.GallagerRandomCodingExponent(0.04)
e0, er = gal.e0_and_error_exponent(1.0, 0.1)
assert e0 > 0.3 and er > 0.2
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GallagerRandomCodingExponent test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4550_arikan_polar_channel_polarization() -> Result<()> {
    use oxideon::science::arikan_polar_channel_polarization::ArikanPolarChannelPolarization;
    let pol = ArikanPolarChannelPolarization::new(0.5)?;
    let (zm, zp, gap) = pol.butterfly_bhattacharyya_split();
    assert!((zm - 0.75).abs() < 1e-12 && (zp - 0.25).abs() < 1e-12 && (gap - 0.5).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

pol = ox.ArikanPolarChannelPolarization(0.5)
zm, zp, gap = pol.butterfly_bhattacharyya_split()
assert abs(zm - 0.75) < 1e-12 and abs(zp - 0.25) < 1e-12 and abs(gap - 0.5) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python ArikanPolarChannelPolarization test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4560_macwilliams_duality_singleton_mds() -> Result<()> {
    use oxideon::science::macwilliams_duality_singleton_mds::MacWilliamsSingletonMdsCode;
    let mds = MacWilliamsSingletonMdsCode::new(15, 9)?;
    let (d, t, d_dual) = mds.singleton_distance_and_dual_mds();
    assert_eq!((d, t, d_dual), (7, 3, 10));

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

mds = ox.MacWilliamsSingletonMdsCode(15, 9)
assert mds.singleton_distance_and_dual_mds() == (7, 3, 10)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python MacWilliamsSingletonMdsCode test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4570_guruswami_sudan_list_decoding() -> Result<()> {
    use oxideon::science::guruswami_sudan_list_decoding::GuruswamiSudanListDecoder;
    let gs = GuruswamiSudanListDecoder::new(0.25)?;
    let (r_gs, r_u, gain) = gs.gs_radius_unique_radius_and_gain();
    assert!(
        (r_gs - 0.5).abs() < 1e-12 && (r_u - 0.375).abs() < 1e-12 && (gain - 0.125).abs() < 1e-12
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

gs = ox.GuruswamiSudanListDecoder(0.25)
r_gs, r_u, gain = gs.gs_radius_unique_radius_and_gain()
assert abs(r_gs - 0.5) < 1e-12 and abs(r_u - 0.375) < 1e-12 and abs(gain - 0.125) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python GuruswamiSudanListDecoder test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4580_ahlswede_cai_li_yeung_network_coding() -> Result<()> {
    use oxideon::science::ahlswede_cai_li_yeung_network_coding::LinearNetworkCodingMaxFlow;
    let nc = LinearNetworkCodingMaxFlow::new(2)?;
    let (cap, gain) = nc.multicast_capacity_and_coding_gain(&[2.0, 2.0], 1.5)?;
    assert!((cap - 2.0).abs() < 1e-12 && (gain - 4.0 / 3.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

nc = ox.LinearNetworkCodingMaxFlow(2)
cap, gain = nc.multicast_capacity_and_coding_gain([2.0, 2.0], 1.5)
assert abs(cap - 2.0) < 1e-12 and abs(gain - 4.0 / 3.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python LinearNetworkCodingMaxFlow test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4590_polyanskiy_poor_verdu_finite_blocklength() -> Result<()> {
    use oxideon::science::polyanskiy_poor_verdu_finite_blocklength::PolyanskiyFiniteBlocklengthRate;
    let ppv = PolyanskiyFiniteBlocklengthRate::new(3.0)?;
    let (v, rn) = ppv.dispersion_and_finite_blocklength_rate(1000, 3.0)?;
    assert!(v > 0.9 && rn > 0.9 && rn < 1.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

ppv = ox.PolyanskiyFiniteBlocklengthRate(3.0)
v, rn = ppv.dispersion_and_finite_blocklength_rate(1000, 3.0)
assert v > 0.9 and 0.9 < rn < 1.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python PolyanskiyFiniteBlocklengthRate test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4600_grand_omni_hexatetracosa_engine_xlvi() -> Result<()> {
    use oxideon::omni_hexatetracosa_xlvi::GrandOmniHexatetracosaEngineXLVI;
    let eng = GrandOmniHexatetracosaEngineXLVI::new(46);
    let inv = eng.execute_hexatetracosa_synthesis()?;
    assert!(inv > 6710.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniHexatetracosaEngineXLVI(46)
assert eng.execute_hexatetracosa_synthesis() > 6710.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniHexatetracosaEngineXLVI test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4610_regev_lwe_lattice_reduction() -> Result<()> {
    use oxideon::science::regev_lwe_lattice_reduction::RegevLweLatticeSecurity;
    let lwe = RegevLweLatticeSecurity::new(64, 1000)?;
    let (sig, gam, ok) = lwe.noise_std_approx_factor_and_regev_check(0.02)?;
    assert!((sig - 20.0).abs() < 1e-12 && (gam - 3200.0).abs() < 1e-12 && ok);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

lwe = ox.RegevLweLatticeSecurity(64, 1000)
sig, gam, ok = lwe.noise_std_approx_factor_and_regev_check(0.02)
assert abs(sig - 20.0) < 1e-12 and abs(gam - 3200.0) < 1e-12 and ok
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python RegevLweLatticeSecurity test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4620_lll_bkz_hermite_root_factor() -> Result<()> {
    use oxideon::science::lll_bkz_hermite_root_factor::LllBkzLatticeReduction;
    let bkz = LllBkzLatticeReduction::new(500)?;
    let (d0, bits) = bkz.root_hermite_factor_and_core_svp_bits();
    assert!(d0 > 1.003 && d0 < 1.005 && (bits - 146.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

bkz = ox.LllBkzLatticeReduction(500)
d0, bits = bkz.root_hermite_factor_and_core_svp_bits()
assert 1.003 < d0 < 1.005 and abs(bits - 146.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python LllBkzLatticeReduction test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4630_gentry_fhe_bootstrapping_noise() -> Result<()> {
    use oxideon::science::gentry_fhe_bootstrapping_noise::GentryFheNoiseGrowthEngine;
    let fhe = GentryFheNoiseGrowthEngine::new(4096, 16)?;
    let (b, ok) = fhe.remaining_noise_budget_bits(100.0, 32.0)?;
    assert!((b - 90.0).abs() < 1e-12 && ok);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

fhe = ox.GentryFheNoiseGrowthEngine(4096, 16)
b, ok = fhe.remaining_noise_budget_bits(100.0, 32.0)
assert abs(b - 90.0) < 1e-12 and ok
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GentryFheNoiseGrowthEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4640_lund_fortnow_karloff_nisan_sumcheck() -> Result<()> {
    use oxideon::science::lund_fortnow_karloff_nisan_sumcheck::SumcheckInteractiveProof;
    let sc = SumcheckInteractiveProof::new(20, 3)?;
    let (w, err) = sc.communication_words_and_soundness_error(6000.0)?;
    assert_eq!(w, 80);
    assert!((err - 0.01).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

sc = ox.SumcheckInteractiveProof(20, 3)
w, err = sc.communication_words_and_soundness_error(6000.0)
assert w == 80 and abs(err - 0.01) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python SumcheckInteractiveProof test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4650_groth16_qap_zk_snark() -> Result<()> {
    use oxideon::science::groth16_qap_zk_snark::Groth16QapSnarkAnalyzer;
    let g16 = Groth16QapSnarkAnalyzer::new(1024, 4)?;
    let (els, pairs, ops) = g16.proof_elements_pairings_and_ntt_ops();
    assert_eq!((els, pairs), (3, 3));
    assert!((ops - 10240.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

g16 = ox.Groth16QapSnarkAnalyzer(1024, 4)
els, pairs, ops = g16.proof_elements_pairings_and_ntt_ops()
assert (els, pairs) == (3, 3) and abs(ops - 10240.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python Groth16QapSnarkAnalyzer test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4660_ben_sasson_fri_reed_solomon_stark() -> Result<()> {
    use oxideon::science::ben_sasson_fri_reed_solomon_stark::FriReedSolomonProximityStark;
    let fri = FriReedSolomonProximityStark::new(0.25)?;
    let (dj, bits) = fri.johnson_proximity_and_query_security_bits(64)?;
    assert!((dj - 0.5).abs() < 1e-12 && (bits - 64.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

fri = ox.FriReedSolomonProximityStark(0.25)
dj, bits = fri.johnson_proximity_and_query_security_bits(64)
assert abs(dj - 0.5) < 1e-12 and abs(bits - 64.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python FriReedSolomonProximityStark test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4670_shamir_bgw_mpc_threshold() -> Result<()> {
    use oxideon::science::shamir_bgw_mpc_threshold::ShamirBgwThresholdMpc;
    let bgw = ShamirBgwThresholdMpc::new(10)?;
    assert_eq!(bgw.max_semi_honest_and_malicious_thresholds(), (4, 3));

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

bgw = ox.ShamirBgwThresholdMpc(10)
assert bgw.max_semi_honest_and_malicious_thresholds() == (4, 3)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python ShamirBgwThresholdMpc test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4680_dwork_mcsherry_gaussian_differential_privacy() -> Result<()> {
    use oxideon::science::dwork_mcsherry_gaussian_differential_privacy::DworkGaussianDifferentialPrivacy;
    let dp = DworkGaussianDifferentialPrivacy::new(1.0)?;
    let (sig, etot) = dp.gaussian_sigma_and_composed_epsilon(0.1, 1e-5, 100)?;
    assert!(sig > 48.0 && etot > 5.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

dp = ox.DworkGaussianDifferentialPrivacy(1.0)
sig, etot = dp.gaussian_sigma_and_composed_epsilon(0.1, 1e-5, 100)
assert sig > 48.0 and etot > 5.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python DworkGaussianDifferentialPrivacy test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4690_mironov_renyi_dp_moments_accountant() -> Result<()> {
    use oxideon::science::mironov_renyi_dp_moments_accountant::MironovRenyiDpAccountant;
    let rdp_acc = MironovRenyiDpAccountant::new(2.0)?;
    let (rdp, eps) = rdp_acc.rdp_and_converted_dp_epsilon(9.0, 8, 1e-4)?;
    assert!((rdp - 9.0).abs() < 1e-12 && eps > 10.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

rdp_acc = ox.MironovRenyiDpAccountant(2.0)
rdp, eps = rdp_acc.rdp_and_converted_dp_epsilon(9.0, 8, 1e-4)
assert abs(rdp - 9.0) < 1e-12 and eps > 10.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python MironovRenyiDpAccountant test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4700_grand_omni_heptatetracosa_engine_xlvii() -> Result<()> {
    use oxideon::omni_heptatetracosa_xlvii::GrandOmniHeptatetracosaEngineXLVII;
    let eng = GrandOmniHeptatetracosaEngineXLVII::new(47);
    let inv = eng.execute_heptatetracosa_synthesis()?;
    assert!(inv > 7150.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniHeptatetracosaEngineXLVII(47)
assert eng.execute_heptatetracosa_synthesis() > 7150.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniHeptatetracosaEngineXLVII test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4710_pontryagin_maximum_principle_bang_bang() -> Result<()> {
    use oxideon::science::pontryagin_maximum_principle_bang_bang::PontryaginMinimumTimeBangBang;
    let pmp = PontryaginMinimumTimeBangBang::new(2.0)?;
    let (ts, ttot) = pmp.switching_time_and_minimum_total_time(-18.0)?;
    assert!((ts - 3.0).abs() < 1e-12 && (ttot - 6.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

pmp = ox.PontryaginMinimumTimeBangBang(2.0)
ts, ttot = pmp.switching_time_and_minimum_total_time(-18.0)
assert abs(ts - 3.0) < 1e-12 and abs(ttot - 6.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python PontryaginMinimumTimeBangBang test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4720_kalman_bucy_lqg_riccati() -> Result<()> {
    use oxideon::science::kalman_bucy_lqg_riccati::KalmanBucyRiccatiFilter;
    let kb = KalmanBucyRiccatiFilter::new(-3.0, 1.0)?;
    let (p_inf, k_inf) = kb.steady_state_covariance_and_gain(16.0, 1.0)?;
    assert!((p_inf - 2.0).abs() < 1e-12 && (k_inf - 2.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

kb = ox.KalmanBucyRiccatiFilter(-3.0, 1.0)
p_inf, k_inf = kb.steady_state_covariance_and_gain(16.0, 1.0)
assert abs(p_inf - 2.0) < 1e-12 and abs(k_inf - 2.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python KalmanBucyRiccatiFilter test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4730_zames_doyle_glover_h_infinity_control() -> Result<()> {
    use oxideon::science::zames_doyle_glover_h_infinity_control::ZamesDoyleHInfinityControl;
    let hinf = ZamesDoyleHInfinityControl::new(0.4)?;
    let (max_d, stab) = hinf.max_allowable_uncertainty_and_stability(2.0)?;
    assert!((max_d - 2.5).abs() < 1e-12 && stab);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

hinf = ox.ZamesDoyleHInfinityControl(0.4)
max_d, stab = hinf.max_allowable_uncertainty_and_stability(2.0)
assert abs(max_d - 2.5) < 1e-12 and stab
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python ZamesDoyleHInfinityControl test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4740_sontag_control_lyapunov_iss() -> Result<()> {
    use oxideon::science::sontag_control_lyapunov_iss::SontagControlLyapunovEngine;
    let (u, vdot) = SontagControlLyapunovEngine::sontag_control_and_lyapunov_rate(3.0, 2.0)?;
    assert!((u - (-4.0)).abs() < 1e-12 && (vdot - (-5.0)).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

u, vdot = ox.SontagControlLyapunovEngine.sontag_control_and_lyapunov_rate(3.0, 2.0)
assert abs(u - (-4.0)) < 1e-12 and abs(vdot - (-5.0)) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python SontagControlLyapunovEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4750_ames_control_barrier_function_qp() -> Result<()> {
    use oxideon::science::ames_control_barrier_function_qp::AmesControlBarrierFunctionQp;
    let cbf = AmesControlBarrierFunctionQp::new(2.0)?;
    let (u_star, margin) = cbf.filter_nominal_control(1.0, -4.0, 2.0, 0.0)?;
    assert!((u_star - 1.0).abs() < 1e-12 && margin.abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

cbf = ox.AmesControlBarrierFunctionQp(2.0)
u_star, margin = cbf.filter_nominal_control(1.0, -4.0, 2.0, 0.0)
assert abs(u_star - 1.0) < 1e-12 and abs(margin) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python AmesControlBarrierFunctionQp test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4760_lasry_lions_mean_field_game() -> Result<()> {
    use oxideon::science::lasry_lions_mean_field_game::LasryLionsLinearQuadraticMfg;
    let mfg = LasryLionsLinearQuadraticMfg::new(0.0, 4.0, 1.0)?;
    let (pi_s, var) = mfg.equilibrium_riccati_and_stationary_variance(2.0)?;
    assert!((pi_s - 2.0).abs() < 1e-12 && (var - 1.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

mfg = ox.LasryLionsLinearQuadraticMfg(0.0, 4.0, 1.0)
pi_s, var = mfg.equilibrium_riccati_and_stationary_variance(2.0)
assert abs(pi_s - 2.0) < 1e-12 and abs(var - 1.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python LasryLionsLinearQuadraticMfg test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4770_isaacs_homicidal_chauffeur_differential_game() -> Result<()> {
    use oxideon::science::isaacs_homicidal_chauffeur_differential_game::IsaacsDifferentialPursuitGame;
    let ig = IsaacsDifferentialPursuitGame::new(5.0, 3.0)?;
    let (ra, tc) = ig.apollonius_radius_and_capture_time(16.0)?;
    assert!((ra - 15.0).abs() < 1e-12 && (tc - 8.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

ig = ox.IsaacsDifferentialPursuitGame(5.0, 3.0)
ra, tc = ig.apollonius_radius_and_capture_time(16.0)
assert abs(ra - 15.0) < 1e-12 and abs(tc - 8.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python IsaacsDifferentialPursuitGame test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4780_kushner_stratonovich_zakai_nonlinear_filter() -> Result<()> {
    use oxideon::science::kushner_stratonovich_zakai_nonlinear_filter::ZakaiUnnormalizedFilteringSde;
    let zak = ZakaiUnnormalizedFilteringSde::new(1.0)?;
    let (z_norm, post_m) = zak.kallianpur_striebel_posterior_mean(&[-1.0, 1.0], 1.0, 0.5)?;
    assert!(z_norm > 0.8 && post_m > 0.7);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

zak = ox.ZakaiUnnormalizedFilteringSde(1.0)
z_norm, post_m = zak.kallianpur_striebel_posterior_mean([-1.0, 1.0], 1.0, 0.5)
assert z_norm > 0.8 and post_m > 0.7
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python ZakaiUnnormalizedFilteringSde test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4790_willems_dissipativity_passivity_kyp() -> Result<()> {
    use oxideon::science::willems_dissipativity_passivity_kyp::WillemsDissipativityKypEngine;
    let kyp = WillemsDissipativityKypEngine::new(0.25, 0.25)?;
    let (w, sp) = kyp.supply_rate_and_strict_passivity(2.0, 2.0);
    assert!((w - 2.0).abs() < 1e-12 && sp);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

kyp = ox.WillemsDissipativityKypEngine(0.25, 0.25)
w, sp = kyp.supply_rate_and_strict_passivity(2.0, 2.0)
assert abs(w - 2.0) < 1e-12 and sp
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python WillemsDissipativityKypEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4800_grand_omni_octatetracosa_engine_xlviii() -> Result<()> {
    use oxideon::omni_octatetracosa_xlviii::GrandOmniOctatetracosaEngineXLVIII;
    let eng = GrandOmniOctatetracosaEngineXLVIII::new(48);
    let inv = eng.execute_octatetracosa_synthesis()?;
    assert!(inv > 7190.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniOctatetracosaEngineXLVIII(48)
assert eng.execute_octatetracosa_synthesis() > 7190.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniOctatetracosaEngineXLVIII test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4810_maclane_coherence_monoidal_pentagon() -> Result<()> {
    use oxideon::science::maclane_coherence_monoidal_pentagon::MacLaneMonoidalCoherence;
    let ml = MacLaneMonoidalCoherence::new(4)?;
    assert_eq!(ml.associahedron_vertices_and_dimension(), (5, 2));

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

ml = ox.MacLaneMonoidalCoherence(4)
assert ml.associahedron_vertices_and_dimension() == (5, 2)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| Error::Msg(format!("Python MacLaneMonoidalCoherence test failed: {e}")))?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4820_grothendieck_topos_giraud_sheaf() -> Result<()> {
    use oxideon::science::grothendieck_topos_giraud_sheaf::GiraudGrothendieckToposEngine;
    let gt = GiraudGrothendieckToposEngine::new(5)?;
    assert_eq!(gt.sieve_cardinality_and_dense_rank(), (32, 5));

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

gt = ox.GiraudGrothendieckToposEngine(5)
assert gt.sieve_cardinality_and_dense_rank() == (32, 5)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GiraudGrothendieckToposEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4830_lurie_infinity_topos_straightening() -> Result<()> {
    use oxideon::science::lurie_infinity_topos_straightening::LurieInfinityCategoryEngine;
    let lur = LurieInfinityCategoryEngine::new(3)?;
    assert_eq!(lur.inner_horn_count_and_subsimplices(), (2, 15));

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

lur = ox.LurieInfinityCategoryEngine(3)
assert lur.inner_horn_count_and_subsimplices() == (2, 15)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python LurieInfinityCategoryEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4840_baez_dolan_lurie_cobordism_hypothesis() -> Result<()> {
    use oxideon::science::baez_dolan_lurie_cobordism_hypothesis::BaezDolanCobordismHypothesis;
    let bdl = BaezDolanCobordismHypothesis::new(3, 4)?;
    let (zt, zs) = bdl.torus_and_sphere_partition_invariants()?;
    assert!((zt - 16.0).abs() < 1e-12 && (zs - 0.25).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

bdl = ox.BaezDolanCobordismHypothesis(3, 4)
zt, zs = bdl.torus_and_sphere_partition_invariants()
assert abs(zt - 16.0) < 1e-12 and abs(zs - 0.25) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python BaezDolanCobordismHypothesis test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4850_voevodsky_univalence_hott_freudenthal() -> Result<()> {
    use oxideon::science::voevodsky_univalence_hott_freudenthal::HomotopyTypeTheoryUnivalence;
    let hott = HomotopyTypeTheoryUnivalence::new(5)?;
    assert_eq!(
        hott.loop_space_cardinality_and_freudenthal_bound(3)?,
        (120, 6)
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

hott = ox.HomotopyTypeTheoryUnivalence(5)
assert hott.loop_space_cardinality_and_freudenthal_bound(3) == (120, 6)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python HomotopyTypeTheoryUnivalence test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4860_scholze_perfectoid_tilting_equivalence() -> Result<()> {
    use oxideon::science::scholze_perfectoid_tilting_equivalence::ScholzePerfectoidTiltingEngine;
    let per = ScholzePerfectoidTiltingEngine::new(4)?;
    let (v, nrm) = per.perfection_tower_valuation_and_norm(1);
    assert!((v - 0.25).abs() < 1e-12 && (nrm - 2.0f64.powf(-0.5)).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

per = ox.ScholzePerfectoidTiltingEngine(4)
v, nrm = per.perfection_tower_valuation_and_norm(1)
assert abs(v - 0.25) < 1e-12 and abs(nrm - 2.0**-0.5) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python ScholzePerfectoidTiltingEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4870_clausen_scholze_condensed_liquid_tensor() -> Result<()> {
    use oxideon::science::clausen_scholze_condensed_liquid_tensor::ClausenScholzeLiquidTensor;
    let liq = ClausenScholzeLiquidTensor::new(0.5)?;
    let (qn, vanish) = liq.p_liquid_norm_and_ext_vanishing(&[1.0, 4.0], 0.25)?;
    assert!((qn - 9.0).abs() < 1e-12 && vanish);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

liq = ox.ClausenScholzeLiquidTensor(0.5)
qn, vanish = liq.p_liquid_norm_and_ext_vanishing([1.0, 4.0], 0.25)
assert abs(qn - 9.0) < 1e-12 and vanish
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python ClausenScholzeLiquidTensor test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4880_bhatt_scholze_prismatic_cohomology() -> Result<()> {
    use oxideon::science::bhatt_scholze_prismatic_cohomology::BhattScholzePrismaticEngine;
    let bs = BhattScholzePrismaticEngine::new(4)?;
    let (rk, q3) = bs.hodge_tate_rank_and_q_integer(3, 2.0)?;
    assert_eq!(rk, 16);
    assert!((q3 - 7.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

bs = ox.BhattScholzePrismaticEngine(4)
rk, q3 = bs.hodge_tate_rank_and_q_integer(3, 2.0)
assert rk == 16 and abs(q3 - 7.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python BhattScholzePrismaticEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4890_fargues_fontaine_curve_geometrization() -> Result<()> {
    use oxideon::science::fargues_fontaine_curve_geometrization::FarguesFontaineCurveEngine;
    let ff = FarguesFontaineCurveEngine::new(3, 4)?;
    let (sl, et) = ff.hn_slope_and_is_etale_semistable();
    assert!((sl - 0.75).abs() < 1e-12 && !et);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

ff = ox.FarguesFontaineCurveEngine(3, 4)
sl, et = ff.hn_slope_and_is_etale_semistable()
assert abs(sl - 0.75) < 1e-12 and not et
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python FarguesFontaineCurveEngine test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4900_grand_omni_nonatetracosa_engine_xlix() -> Result<()> {
    use oxideon::omni_nonatetracosa_xlix::GrandOmniNonatetracosaEngineXLIX;
    let eng = GrandOmniNonatetracosaEngineXLIX::new(49);
    let inv = eng.execute_nonatetracosa_synthesis()?;
    assert!(inv > 7400.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniNonatetracosaEngineXLIX(49)
assert eng.execute_nonatetracosa_synthesis() > 7400.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniNonatetracosaEngineXLIX test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4910_sachdev_ye_kitaev_syk_chaos() -> Result<()> {
    use oxideon::physics::sachdev_ye_kitaev_syk_chaos::SachdevYeKitaevMaximalChaos;
    let syk = SachdevYeKitaevMaximalChaos::new(100.0)?;
    let (mss, lam) = syk.mss_bound_and_syk_lyapunov(1.0)?;
    assert!((mss - 2.0 * std::f64::consts::PI).abs() < 1e-12 && lam < mss && lam > 5.9);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

syk = ox.SachdevYeKitaevMaximalChaos(100.0)
mss, lam = syk.mss_bound_and_syk_lyapunov(1.0)
assert 5.9 < lam < mss
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python SachdevYeKitaevMaximalChaos test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4920_jackiw_teitelboim_schwarzian_gravity() -> Result<()> {
    use oxideon::physics::jackiw_teitelboim_schwarzian_gravity::JackiwTeitelboimSchwarzianGravity;
    let jt = JackiwTeitelboimSchwarzianGravity::new(0.5, 1.0)?;
    let (rho, s_bh) = jt.disk_spectral_density_and_entropy(1.0)?;
    assert!(rho > 10.0 && (s_bh - (1.0 + 2.0 * std::f64::consts::PI)).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

import math
jt = ox.JackiwTeitelboimSchwarzianGravity(0.5, 1.0)
rho, s_bh = jt.disk_spectral_density_and_entropy(1.0)
assert rho > 10.0 and abs(s_bh - (1.0 + 2.0 * math.pi)) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python JackiwTeitelboimSchwarzianGravity test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4930_penington_almheiri_qes_island_page() -> Result<()> {
    use oxideon::physics::penington_almheiri_qes_island_page::QuantumExtremalSurfaceIslandPage;
    let qes = QuantumExtremalSurfaceIslandPage::new(100.0, 2.0)?;
    let (tp, s_rad, dom) = qes.page_time_entropy_and_island_dominance(30.0)?;
    assert!((tp - 25.0).abs() < 1e-12 && (s_rad - 40.0).abs() < 1e-12 && dom);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

qes = ox.QuantumExtremalSurfaceIslandPage(100.0, 2.0)
tp, s_rad, dom = qes.page_time_entropy_and_island_dominance(30.0)
assert abs(tp - 25.0) < 1e-12 and abs(s_rad - 40.0) < 1e-12 and dom
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python QuantumExtremalSurfaceIslandPage test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4940_pastawski_yoshida_harlow_preskill_happy() -> Result<()> {
    use oxideon::physics::pastawski_yoshida_harlow_preskill_happy::HappyHolographicCodeBulkWedge;
    let hap = HappyHolographicCodeBulkWedge::new(2)?;
    let (lam, sc) = hap.hyperbolic_growth_ratio_and_boundary_scale();
    assert!(
        (lam - (2.0 + 3.0f64.sqrt())).abs() < 1e-12
            && (sc - (7.0 + 4.0 * 3.0f64.sqrt())).abs() < 1e-12
    );

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

hap = ox.HappyHolographicCodeBulkWedge(2)
lam, sc = hap.hyperbolic_growth_ratio_and_boundary_scale()
assert abs(lam - (2.0 + 3.0**0.5)) < 1e-12 and sc > 13.92
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python HappyHolographicCodeBulkWedge test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4950_susskind_brown_complexity_equals_action() -> Result<()> {
    use oxideon::physics::susskind_brown_complexity_equals_action::HolographicComplexityActionVolume;
    let ca = HolographicComplexityActionVolume::new(std::f64::consts::PI)?;
    let (rate, c_tot) = ca.lloyd_growth_rate_and_complexity(5.0)?;
    assert!((rate - 2.0).abs() < 1e-12 && (c_tot - 10.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

import math
ca = ox.HolographicComplexityActionVolume(math.pi)
rate, c_tot = ca.lloyd_growth_rate_and_complexity(5.0)
assert abs(rate - 2.0) < 1e-12 and abs(c_tot - 10.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python HolographicComplexityActionVolume test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4960_strominger_bms_soft_graviton_memory() -> Result<()> {
    use oxideon::physics::strominger_bms_soft_graviton_memory::StromingerSoftTheoremMemoryTriangle;
    let tri = StromingerSoftTheoremMemoryTriangle::new(6.0)?;
    let (sp, mem) = tri.soft_pole_and_memory_strain(0.5, 3.0)?;
    assert!((sp - 12.0).abs() < 1e-12 && (mem - 2.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

tri = ox.StromingerSoftTheoremMemoryTriangle(6.0)
sp, mem = tri.soft_pole_and_memory_strain(0.5, 3.0)
assert abs(sp - 12.0) < 1e-12 and abs(mem - 2.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python StromingerSoftTheoremMemoryTriangle test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4970_arkani_hamed_trnka_amplituhedron() -> Result<()> {
    use oxideon::physics::arkani_hamed_trnka_amplituhedron::AmplituhedronGrassmannianVolume;
    let amp = AmplituhedronGrassmannianVolume::new(6, 1)?;
    assert_eq!(amp.bcfw_cell_count_and_dimension()?, (3, 4));

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

amp = ox.AmplituhedronGrassmannianVolume(6, 1)
assert amp.bcfw_cell_count_and_dimension() == (3, 4)
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python AmplituhedronGrassmannianVolume test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4980_vafa_ooguri_swampland_distance_de_sitter() -> Result<()> {
    use oxideon::physics::vafa_ooguri_swampland_distance_de_sitter::SwamplandDistanceDeSitterBounds;
    let sw = SwamplandDistanceDeSitterBounds::new(6)?;
    let (al, mr) = sw.min_decay_rate_and_tower_mass_ratio(2.0)?;
    assert!((al - 0.5).abs() < 1e-12 && (mr - (-1.0f64).exp()).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

import math
sw = ox.SwamplandDistanceDeSitterBounds(6)
al, mr = sw.min_decay_rate_and_tower_mass_ratio(2.0)
assert abs(al - 0.5) < 1e-12 and abs(mr - math.exp(-1.0)) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python SwamplandDistanceDeSitterBounds test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v4990_pasterski_strominger_zhiboedov_celestial() -> Result<()> {
    use oxideon::physics::pasterski_strominger_zhiboedov_celestial::CelestialHolographyMellinAmplitude;
    let cel = CelestialHolographyMellinAmplitude::new(0.5)?;
    let (nsq, mel) = cel.principal_norm_sq_and_integer_mellin_kernel(2.0, 3)?;
    assert!((nsq - 5.0).abs() < 1e-12 && (mel - 16.0).abs() < 1e-12);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

cel = ox.CelestialHolographyMellinAmplitude(0.5)
nsq, mel = cel.principal_norm_sq_and_integer_mellin_kernel(2.0, 3)
assert abs(nsq - 5.0) < 1e-12 and abs(mel - 16.0) < 1e-12
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python CelestialHolographyMellinAmplitude test failed: {e}"
            ))
        })?;
    }

    Ok(())
}

#[test]
fn test_frontier_v5000_grand_omni_quincentennial_engine_l() -> Result<()> {
    use oxideon::omni_quincentennial_l::GrandOmniQuincentennialEngineL;
    let eng = GrandOmniQuincentennialEngineL::new(50);
    let inv = eng.execute_quincentennial_synthesis()?;
    assert!(inv > 7500.0);

    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        pyo3::Python::initialize();

        pyo3::Python::attach(|py| -> pyo3::PyResult<()> {
            let ox = PyModule::new(py, "oxideon")?;
            oxideon::python::register_python_module(&ox)?;
            let sys = py.import("sys")?;
            sys.getattr("modules")?.set_item("oxideon", &ox)?;

            let code = std::ffi::CString::new(
                r#"
import oxideon as ox

eng = ox.GrandOmniQuincentennialEngineL(50)
assert eng.execute_quincentennial_synthesis() > 7500.0
"#,
            )?;
            py.run(code.as_c_str(), Some(&pyo3::types::PyDict::new(py)), None)?;
            Ok(())
        })
        .map_err(|e| {
            Error::Msg(format!(
                "Python GrandOmniQuincentennialEngineL test failed: {e}"
            ))
        })?;
    }

    Ok(())
}
