//! Logit parity of `GGUFCausalLM` against llama.cpp itself, on the tiny random-weight GGUF files that
//! llama.cpp's own test suite writes (`llama`, Mixtral-style MoE, `qwen2`, `qwen2moe`).
//!
//! Needs two directories (the test is skipped when they are not set):
//! * `OXIDEON_LLAMA_TEST_MODELS` - holds `<name>.gguf` (`$LLAMA/build*/tests/test-models`)
//! * `OXIDEON_LLAMA_LOGITS`      - holds `<name>.logits` from `tools/ggml_reference/llama_logits.c`
//!   run with the tokens `1 2 ... 48`.

use oxideon::Tensor;
use oxideon::nn::GGUFCausalLM;
use oxideon::serialize::GGUFFile;
use std::path::PathBuf;

/// Long enough to exercise 32-token sliding windows (Gemma 2/3 reference models).
const N_TOKENS: usize = 48;

/// llama.cpp's CPU GELU is an f16 lookup table; reproduce it so its noise does not compound over the layers.
fn ggml_numerics() {
    oxideon::nn::gguf_model::emulate_ggml_gelu_table(true);
}

fn parity(name: &str) {
    ggml_numerics();
    let (Some(models), Some(logits)) = (
        std::env::var_os("OXIDEON_LLAMA_TEST_MODELS"),
        std::env::var_os("OXIDEON_LLAMA_LOGITS"),
    ) else {
        eprintln!("skipping {name}: OXIDEON_LLAMA_TEST_MODELS / OXIDEON_LLAMA_LOGITS not set");
        return;
    };
    let file = GGUFFile::open(PathBuf::from(models).join(format!("{name}.gguf"))).unwrap();
    let model = GGUFCausalLM::from_gguf(&file).unwrap();

    let reference =
        std::fs::read_to_string(PathBuf::from(logits).join(format!("{name}.logits"))).unwrap();
    let mut lines = reference.lines();
    let head: Vec<&str> = lines.next().unwrap().split(' ').collect();
    let vocab: usize = head[1].parse().unwrap();
    let want: Vec<Vec<f32>> = lines
        .map(|l| l.split(' ').map(|v| v.parse().unwrap()).collect())
        .collect();
    assert_eq!(want.len(), N_TOKENS);
    assert_eq!(model.config.vocab_size, vocab);

    let tokens: Vec<i64> = (1..=N_TOKENS as i64).collect();
    let got = model
        .forward(&Tensor::from_slice(&tokens, &[1, N_TOKENS]).unwrap())
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let mut worst = 0.0f32;
    for (pos, row) in want.iter().enumerate() {
        let scale = row.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
        for (j, w) in row.iter().enumerate() {
            worst = worst.max((got[pos * vocab + j] - w).abs() / scale);
        }
    }
    if std::env::var_os("OXIDEON_PARITY_VERBOSE").is_some() {
        for (pos, row) in want.iter().enumerate() {
            let scale = row.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
            let e = row.iter().enumerate().fold(0.0f32, |m, (j, w)| {
                m.max((got[pos * vocab + j] - w).abs() / scale)
            });
            eprintln!("  {name} pos {pos}: {e:.2e}");
        }
    }
    if std::env::var_os("OXIDEON_PARITY_DUMP").is_some() {
        let r: Vec<String> = (0..8)
            .map(|j| format!("{:.4}/{:.4}", got[j], want[0][j]))
            .collect();
        eprintln!("  {name} pos0 got/want: {}", r.join(" "));
        eprintln!(
            "  {name} scales: emb {:?} res {:?} logit {:?} mul {:?} attn_scale {:?}",
            model.config.embedding_scale,
            model.config.residual_scale,
            model.config.logit_scale,
            model.config.logit_mul,
            model.blocks[0].attn_scale
        );
    }
    eprintln!("{name}: worst error relative to the logit scale = {worst:.3e}");
    assert!(worst < 2e-3, "{name} disagrees with llama.cpp: {worst}");
}

#[test]
fn llama_dense_matches_llama_cpp() {
    parity("llama-dense");
}

#[test]
fn llama_moe_matches_llama_cpp() {
    parity("llama-moe");
}

#[test]
fn qwen2_dense_matches_llama_cpp() {
    parity("qwen2-dense");
}

#[test]
fn qwen2moe_matches_llama_cpp() {
    parity("qwen2moe-moe");
}

#[test]
fn qwen3_dense_matches_llama_cpp() {
    parity("qwen3-dense");
}

#[test]
fn qwen3moe_moe_matches_llama_cpp() {
    parity("qwen3moe-moe");
}

#[test]
fn gemma_dense_matches_llama_cpp() {
    parity("gemma-dense");
}

#[test]
fn gemma2_dense_matches_llama_cpp() {
    parity("gemma2-dense");
}

#[test]
fn gemma3_dense_matches_llama_cpp() {
    parity("gemma3-dense");
}

#[test]
fn mistral3_dense_matches_llama_cpp() {
    parity("mistral3-dense");
}

#[test]
fn internlm2_dense_matches_llama_cpp() {
    parity("internlm2-dense");
}

#[test]
fn exaone_dense_matches_llama_cpp() {
    parity("exaone-dense");
}

#[test]
fn smollm3_dense_matches_llama_cpp() {
    parity("smollm3-dense");
}

#[test]
fn granite_dense_matches_llama_cpp() {
    parity("granite-dense");
}

#[test]
fn granitemoe_dense_matches_llama_cpp() {
    parity("granitemoe-dense");
}

#[test]
fn granite_moe_matches_llama_cpp() {
    parity("granite-moe");
}

#[test]
fn olmo2_dense_matches_llama_cpp() {
    parity("olmo2-dense");
}

#[test]
fn granitemoe_moe_matches_llama_cpp() {
    parity("granitemoe-moe");
}

#[test]
fn phi3_dense_matches_llama_cpp() {
    parity("phi3-dense");
}

#[test]
fn olmoe_moe_matches_llama_cpp() {
    parity("olmoe-moe");
}

#[test]
fn deepseek2_moe_matches_llama_cpp() {
    parity("deepseek2-moe");
}

/// Incremental decoding (prefill, then one token at a time with the KV cache) must reproduce the full forward
/// on the architectures with the trickiest caches: MLA latent caches, sliding windows, MoE, NoPE layers.
#[test]
fn cached_decoding_matches_full_forward_on_reference_models() {
    let Some(models) = std::env::var_os("OXIDEON_LLAMA_TEST_MODELS") else {
        eprintln!("skipping: OXIDEON_LLAMA_TEST_MODELS not set");
        return;
    };
    for name in [
        "deepseek2-moe",
        "gemma2-dense",
        "gemma3-dense",
        "phi3-dense",
        "qwen3moe-moe",
        "smollm3-dense",
        "granite-moe",
        "gpt2-dense",
        "mimo2-moe",
        "afmoe-moe",
        "qwen3vl-dense",
        "flint-moe",
        "mamba-dense",
        "mamba2-dense",
        "nemotron_h-dense",
        "granitehybrid-dense",
        "falcon-h1-dense",
        "jamba-dense",
        "lfm2-dense",
        "qwen35-dense",
        "qwen3next-moe",
        "beam-moe",
        "talkie-dense",
        "hrm_text-dense",
        "k2-horizon-moe",
        "muse-spark-moe",
        "plamo2-dense",
        "kimi-linear-moe",
        "bailingmoe3-moe",
        "kimi-k3-moe",
        "dots3note-moe",
        "minimax-01-moe",
        "minimax-m3-moe",
        "hy_v4-moe",
        "glm5-next-moe",
        "qwen4exp-moe",
        "deepseek4-moe",
    ] {
        cached_matches_full(&PathBuf::from(&models).join(format!("{name}.gguf")), name);
    }
}

/// The same check on the tiny models of `tools/ggml_reference/tiny_models.py` (recurrent RWKV state, Gemma 3n's shared
/// K/V and AltUp streams, ...), which live in `OXIDEON_LLAMA_EXTRA_MODELS`.
#[test]
fn cached_decoding_matches_full_forward_on_tiny_models() {
    let Some(models) = std::env::var_os("OXIDEON_LLAMA_EXTRA_MODELS") else {
        eprintln!("skipping: OXIDEON_LLAMA_EXTRA_MODELS not set");
        return;
    };
    for name in [
        "apertus",
        "bitnet",
        "plm",
        "chameleon",
        "gemma3n",
        "rwkv6",
        "rwkv7",
        "arwkv7",
        "rwkv6qwen2",
        "step35",
        "gemma4",
        "gemma4-moe",
    ] {
        let path = PathBuf::from(&models).join(format!("{name}.gguf"));
        if path.exists() {
            cached_matches_full(&path, name);
        }
    }
}

fn cached_matches_full(path: &std::path::Path, name: &str) {
    {
        let file = GGUFFile::open(path).unwrap();
        let model = GGUFCausalLM::from_gguf(&file).unwrap();
        let vocab = model.config.vocab_size;
        let tokens: Vec<i64> = (1..=40).collect();
        let full = model
            .forward(&Tensor::from_slice(&tokens, &[1, 40]).unwrap())
            .unwrap()
            .to_vec::<f32>()
            .unwrap();

        let mut cache = model.new_cache();
        let mut got = model
            .forward_cached(
                &Tensor::from_slice(&tokens[..25], &[1, 25]).unwrap(),
                &mut cache,
            )
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        for &t in &tokens[25..] {
            got.extend(
                model
                    .forward_cached(&Tensor::from_slice(&[t], &[1, 1]).unwrap(), &mut cache)
                    .unwrap()
                    .to_vec::<f32>()
                    .unwrap(),
            );
        }
        assert_eq!(full.len(), got.len(), "{name}");
        // Suppressed tokens (Chameleon's image ids come out as -f32::MAX) stay out of the scale.
        let scale = full
            .iter()
            .filter(|v| v.abs() < 1e30)
            .fold(0.0f32, |m, v| m.max(v.abs()))
            .max(1e-6);
        let worst = full
            .iter()
            .zip(&got)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs() / scale));
        eprintln!("{name}: cached vs full worst relative error {worst:.3e} (vocab {vocab})");
        assert!(
            worst < 1e-3,
            "{name}: cached decoding diverges from the full forward: {worst}"
        );
    }
}

#[test]
fn stablelm_dense_matches_llama_cpp() {
    parity("stablelm-dense");
}

#[test]
fn phi2_dense_matches_llama_cpp() {
    parity("phi2-dense");
}

#[test]
fn starcoder2_dense_matches_llama_cpp() {
    parity("starcoder2-dense");
}

#[test]
fn command_r_dense_matches_llama_cpp() {
    parity("command-r-dense");
}

#[test]
fn olmo_dense_matches_llama_cpp() {
    parity("olmo-dense");
}

#[test]
fn xverse_dense_matches_llama_cpp() {
    parity("xverse-dense");
}

#[test]
fn baichuan_dense_matches_llama_cpp() {
    parity("baichuan-dense");
}

#[test]
fn nanbeige_dense_matches_llama_cpp() {
    parity("nanbeige-dense");
}

#[test]
fn arcee_dense_matches_llama_cpp() {
    parity("arcee-dense");
}

#[test]
fn maincoder_dense_matches_llama_cpp() {
    parity("maincoder-dense");
}

#[test]
fn seed_oss_dense_matches_llama_cpp() {
    parity("seed_oss-dense");
}

#[test]
fn exaone4_dense_matches_llama_cpp() {
    parity("exaone4-dense");
}

#[test]
fn nemotron_dense_matches_llama_cpp() {
    parity("nemotron-dense");
}

#[test]
fn jais2_dense_matches_llama_cpp() {
    parity("jais2-dense");
}

#[test]
fn orion_dense_matches_llama_cpp() {
    parity("orion-dense");
}

#[test]
fn gptneox_dense_matches_llama_cpp() {
    parity("gptneox-dense");
}

#[test]
fn falcon_dense_matches_llama_cpp() {
    parity("falcon-dense");
}

#[test]
fn codeshell_dense_matches_llama_cpp() {
    parity("codeshell-dense");
}

#[test]
fn minicpm_dense_matches_llama_cpp() {
    parity("minicpm-dense");
}

#[test]
fn phimoe_moe_matches_llama_cpp() {
    parity("phimoe-moe");
}

#[test]
fn pangu_embedded_dense_matches_llama_cpp() {
    parity("pangu-embedded-dense");
}

#[test]
fn hunyuan_dense_dense_matches_llama_cpp() {
    parity("hunyuan-dense-dense");
}

#[test]
fn mistral3_moe_matches_llama_cpp() {
    parity("mistral3-moe");
}

#[test]
fn hunyuan_moe_moe_matches_llama_cpp() {
    parity("hunyuan-moe-moe");
}

#[test]
fn dbrx_moe_matches_llama_cpp() {
    parity("dbrx-moe");
}

#[test]
fn ernie4_5_moe_moe_matches_llama_cpp() {
    parity("ernie4_5-moe-moe");
}

#[test]
fn bailingmoe_moe_matches_llama_cpp() {
    parity("bailingmoe-moe");
}

#[test]
fn dots1_moe_matches_llama_cpp() {
    parity("dots1-moe");
}

#[test]
fn bailingmoe2_moe_matches_llama_cpp() {
    parity("bailingmoe2-moe");
}

/// llama.cpp's reference file for the dense `ernie4_5` architecture is (confusingly) named `ernie4_5-moe`.
#[test]
fn ernie4_5_dense_matches_llama_cpp() {
    parity("ernie4_5-moe");
}

#[test]
fn gpt_oss_moe_matches_llama_cpp() {
    parity("gpt-oss-moe");
}

#[test]
fn cohere2_dense_matches_llama_cpp() {
    parity("cohere2-dense");
}

#[test]
fn granite_swa_dense_matches_llama_cpp() {
    parity("granite_swa-dense");
}

#[test]
fn deci_dense_matches_llama_cpp() {
    parity("deci-dense");
}

#[test]
fn openelm_dense_matches_llama_cpp() {
    parity("openelm-dense");
}

#[test]
fn glm4_dense_matches_llama_cpp() {
    parity("glm4-dense");
}

#[test]
fn chatglm_dense_matches_llama_cpp() {
    parity("chatglm-dense");
}

#[test]
fn glm4moe_moe_matches_llama_cpp() {
    parity("glm4moe-moe");
}

#[test]
fn qwen_dense_matches_llama_cpp() {
    parity("qwen-dense");
}

#[test]
fn deepseek_moe_matches_llama_cpp() {
    parity("deepseek-moe");
}

#[test]
fn plamo_dense_matches_llama_cpp() {
    parity("plamo-dense");
}

#[test]
fn mellum_moe_matches_llama_cpp() {
    parity("mellum-moe");
}

#[test]
fn maple_moe_matches_llama_cpp() {
    parity("maple-moe");
}

#[test]
fn minimax_m2_moe_matches_llama_cpp() {
    parity("minimax-m2-moe");
}

#[test]
fn exaone_moe_moe_matches_llama_cpp() {
    parity("exaone-moe-moe");
}

#[test]
fn afmoe_moe_matches_llama_cpp() {
    parity("afmoe-moe");
}

#[test]
fn spark2_5_dense_matches_llama_cpp() {
    parity("spark2_5-dense");
}

#[test]
fn mimo2_moe_matches_llama_cpp() {
    parity("mimo2-moe");
}

#[test]
fn laguna_moe_matches_llama_cpp() {
    parity("laguna-moe");
}

#[test]
fn kolibri1_moe_matches_llama_cpp() {
    parity("kolibri1-moe");
}

#[test]
fn flint_moe_matches_llama_cpp() {
    parity("flint-moe");
}

#[test]
fn minicpm_moe_matches_llama_cpp() {
    parity("minicpm-moe");
}

#[test]
fn cohere2moe_moe_matches_llama_cpp() {
    parity("cohere2moe-moe");
}

#[test]
fn grok_moe_matches_llama_cpp() {
    parity("grok-moe");
}

#[test]
fn arctic_moe_matches_llama_cpp() {
    parity("arctic-moe");
}

#[test]
fn qwen2vl_dense_matches_llama_cpp() {
    parity("qwen2vl-dense");
}

#[test]
fn qwen3vl_dense_matches_llama_cpp() {
    parity("qwen3vl-dense");
}

#[test]
fn qwen3vlmoe_moe_matches_llama_cpp() {
    parity("qwen3vlmoe-moe");
}

#[test]
fn paddleocr_moe_matches_llama_cpp() {
    parity("paddleocr-moe");
}

#[test]
fn hunyuan_vl_dense_matches_llama_cpp() {
    parity("hunyuan_vl-dense");
}

#[test]
fn gpt2_dense_matches_llama_cpp() {
    parity("gpt2-dense");
}

#[test]
fn starcoder_dense_matches_llama_cpp() {
    parity("starcoder-dense");
}

#[test]
fn bloom_dense_matches_llama_cpp() {
    parity("bloom-dense");
}

#[test]
fn mpt_dense_matches_llama_cpp() {
    parity("mpt-dense");
}

#[test]
fn refact_dense_matches_llama_cpp() {
    parity("refact-dense");
}

#[test]
fn jais_dense_matches_llama_cpp() {
    parity("jais-dense");
}

#[test]
fn mistral4_moe_matches_llama_cpp() {
    parity("mistral4-moe");
}

#[test]
fn llama4_moe_matches_llama_cpp() {
    parity("llama4-moe");
}

#[test]
fn hy_v3_moe_matches_llama_cpp() {
    parity("hy_v3-moe");
}

#[test]
fn plamo3_dense_matches_llama_cpp() {
    parity("plamo3-dense");
}

#[test]
fn smallthinker_moe_matches_llama_cpp() {
    parity("smallthinker-moe");
}

#[test]
fn grovemoe_moe_matches_llama_cpp() {
    parity("grovemoe-moe");
}

#[test]
fn minicpm3_dense_matches_llama_cpp() {
    parity("minicpm3-dense");
}

#[test]
fn maion_coder_moe_matches_llama_cpp() {
    parity("maion-coder-moe");
}

/// A real, quantized checkpoint instead of a tiny F32 test model. llama.cpp's CPU kernels quantize the
/// activations to Q8_0 before a Q4_0 dot product while this crate dequantizes the weights and multiplies in
/// F32, so the two differ by quantization noise: the check is a loose relative error plus top-1 agreement.
///
/// `OXIDEON_LLAMA_REAL_MODELS` holds `<name>.gguf`; `OXIDEON_LLAMA_LOGITS` holds `<name>.logits`.
fn real_model_parity(name: &str, max_error: f32, min_top1: f32) {
    let (Some(models), Some(logits)) = (
        std::env::var_os("OXIDEON_LLAMA_REAL_MODELS"),
        std::env::var_os("OXIDEON_LLAMA_LOGITS"),
    ) else {
        eprintln!("skipping {name}: OXIDEON_LLAMA_REAL_MODELS / OXIDEON_LLAMA_LOGITS not set");
        return;
    };
    let file = GGUFFile::open(PathBuf::from(models).join(format!("{name}.gguf"))).unwrap();
    let model = GGUFCausalLM::from_gguf(&file).unwrap();
    let reference =
        std::fs::read_to_string(PathBuf::from(logits).join(format!("{name}.logits"))).unwrap();
    let mut lines = reference.lines();
    let vocab: usize = lines
        .next()
        .unwrap()
        .split(' ')
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    let want: Vec<Vec<f32>> = lines
        .map(|l| l.split(' ').map(|v| v.parse().unwrap()).collect())
        .collect();
    assert_eq!(want.len(), N_TOKENS);
    assert_eq!(model.config.vocab_size, vocab);

    let tokens: Vec<i64> = (1..=N_TOKENS as i64).collect();
    let got = model
        .forward(&Tensor::from_slice(&tokens, &[1, N_TOKENS]).unwrap())
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let (mut worst, mut top1) = (0.0f32, 0usize);
    for (pos, row) in want.iter().enumerate() {
        let scale = row.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
        let mut e = 0.0f32;
        let (mut best_w, mut best_g) = (0usize, 0usize);
        for (j, w) in row.iter().enumerate() {
            let g = got[pos * vocab + j];
            e = e.max((g - w).abs() / scale);
            if *w > row[best_w] {
                best_w = j;
            }
            if g > got[pos * vocab + best_g] {
                best_g = j;
            }
        }
        worst = worst.max(e);
        top1 += usize::from(best_w == best_g);
    }
    let agree = top1 as f32 / N_TOKENS as f32;
    eprintln!("{name}: worst relative error {worst:.3e}, top-1 agreement {top1}/{N_TOKENS}");
    assert!(worst < max_error, "{name}: worst relative error {worst}");
    assert!(agree >= min_top1, "{name}: top-1 agreement {agree}");
}

#[test]
fn qwen25_0_5b_q4_0_real_model_matches_llama_cpp() {
    real_model_parity("qwen25-0_5b-q4_0", 0.2, 0.95);
}

#[test]
fn mamba_dense_matches_llama_cpp() {
    parity("mamba-dense");
}

#[test]
fn mamba2_dense_matches_llama_cpp() {
    parity("mamba2-dense");
}

#[test]
fn nemotron_h_dense_matches_llama_cpp() {
    parity("nemotron_h-dense");
}

#[test]
fn nemotron_h_moe_dense_matches_llama_cpp() {
    parity("nemotron_h_moe-dense");
}

#[test]
fn granitehybrid_dense_matches_llama_cpp() {
    parity("granitehybrid-dense");
}

#[test]
fn falcon_h1_dense_matches_llama_cpp() {
    parity("falcon-h1-dense");
}

#[test]
fn jamba_dense_matches_llama_cpp() {
    parity("jamba-dense");
}

#[test]
fn lfm2_dense_matches_llama_cpp() {
    parity("lfm2-dense");
}

#[test]
fn lfm2moe_moe_matches_llama_cpp() {
    parity("lfm2moe-moe");
}

#[test]
fn qwen35_dense_matches_llama_cpp() {
    parity("qwen35-dense");
}

#[test]
fn qwen35moe_moe_matches_llama_cpp() {
    parity("qwen35moe-moe");
}

#[test]
fn qwen3next_moe_matches_llama_cpp() {
    parity("qwen3next-moe");
}

#[test]
fn deepseek32_moe_matches_llama_cpp() {
    parity("deepseek32-moe");
}

#[test]
fn glm_dsa_moe_matches_llama_cpp() {
    parity("glm-dsa-moe");
}

#[test]
fn muse_glimmer_dense_matches_llama_cpp() {
    parity("muse-glimmer-dense");
}

#[test]
fn plamo2_dense_matches_llama_cpp() {
    parity("plamo2-dense");
}

#[test]
fn kimi_linear_moe_matches_llama_cpp() {
    parity("kimi-linear-moe");
}

#[test]
fn dream_dense_matches_llama_cpp() {
    parity("dream-dense");
}

#[test]
fn llada_dense_matches_llama_cpp() {
    parity("llada-dense");
}

#[test]
fn llada_moe_matches_llama_cpp() {
    parity("llada-moe-moe");
}

#[test]
fn rnd1_moe_matches_llama_cpp() {
    parity("rnd1-moe");
}

#[test]
fn cogvlm_dense_matches_llama_cpp() {
    parity("cogvlm-dense");
}

#[test]
fn qwen3tts_dense_matches_llama_cpp() {
    parity("qwen3tts-dense");
}

#[test]
fn pockettts_dense_matches_llama_cpp() {
    parity("pockettts-dense");
}

#[test]
fn muse_spark_moe_matches_llama_cpp() {
    parity("muse-spark-moe");
}

#[test]
fn k2_horizon_dense_matches_llama_cpp() {
    parity("k2-horizon-dense");
}

#[test]
fn k2_horizon_moe_matches_llama_cpp() {
    parity("k2-horizon-moe");
}

#[test]
fn hrm_text_dense_matches_llama_cpp() {
    parity("hrm_text-dense");
}

#[test]
fn talkie_dense_matches_llama_cpp() {
    parity("talkie-dense");
}

#[test]
fn beam_moe_matches_llama_cpp() {
    parity("beam-moe");
}

/// Every sparse-attention model now runs its indexer, so a context beyond the indexer's budget is no longer refused
/// (the refusal only remains for a model whose layers carry no indexer tensors).
#[test]
fn dsa_models_run_contexts_beyond_the_indexer_top_k() {
    let Some(models) = std::env::var_os("OXIDEON_LLAMA_TEST_MODELS") else {
        eprintln!("skipping: OXIDEON_LLAMA_TEST_MODELS not set");
        return;
    };
    for name in ["minimax-m3-moe", "qwen4exp-moe", "glm5-next-moe"] {
        let file = GGUFFile::open(PathBuf::from(&models).join(format!("{name}.gguf"))).unwrap();
        let mut model = GGUFCausalLM::from_gguf(&file).unwrap();
        assert!(model.config.dsa_top_k.is_some(), "{name}");
        model.config.dsa_top_k = Some(4);
        let ids: Vec<i64> = (1..=8).collect();
        model
            .forward(&Tensor::from_slice(&ids, &[1, 8]).unwrap())
            .unwrap_or_else(|e| panic!("{name} must run beyond its budget: {e}"));
    }
}

#[test]
fn bailingmoe3_moe_matches_llama_cpp() {
    parity("bailingmoe3-moe");
}

#[test]
fn kimi_k3_moe_matches_llama_cpp() {
    parity("kimi-k3-moe");
}

#[test]
fn dots3note_moe_matches_llama_cpp() {
    parity("dots3note-moe");
}

#[test]
fn minimax_01_moe_matches_llama_cpp() {
    parity("minimax-01-moe");
}

#[test]
fn minimax_m3_moe_matches_llama_cpp() {
    parity("minimax-m3-moe");
}

#[test]
fn hy_v4_moe_matches_llama_cpp() {
    parity("hy_v4-moe");
}

#[test]
fn glm5_next_moe_matches_llama_cpp() {
    parity("glm5-next-moe");
}

#[test]
fn qwen4exp_moe_matches_llama_cpp() {
    parity("qwen4exp-moe");
}

#[test]
fn deepseek4_moe_matches_llama_cpp() {
    parity("deepseek4-moe");
}

/// Parity on a custom GGUF (`<stem>.gguf` in the models directory, or in `OXIDEON_LLAMA_EXTRA_MODELS`) and prompt: the
/// reference logits are `<stem>.logits` in the logits directory, made with the same tokens.
fn parity_tokens(stem: &str, tokens: &[i64]) {
    ggml_numerics();
    let (Some(logits), Some(models)) = (
        std::env::var_os("OXIDEON_LLAMA_LOGITS"),
        std::env::var_os("OXIDEON_LLAMA_EXTRA_MODELS")
            .or_else(|| std::env::var_os("OXIDEON_LLAMA_TEST_MODELS")),
    ) else {
        eprintln!("skipping {stem}: OXIDEON_LLAMA_* not set");
        return;
    };
    let path = PathBuf::from(models).join(format!("{stem}.gguf"));
    if !path.exists() {
        eprintln!("skipping {stem}: {} not found", path.display());
        return;
    }
    let model = GGUFCausalLM::from_gguf(&GGUFFile::open(path).unwrap()).unwrap();
    let reference =
        std::fs::read_to_string(PathBuf::from(logits).join(format!("{stem}.logits"))).unwrap();
    let mut lines = reference.lines();
    let vocab: usize = lines
        .next()
        .unwrap()
        .split(' ')
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    let want: Vec<Vec<f32>> = lines
        .map(|l| l.split(' ').map(|v| v.parse().unwrap()).collect())
        .collect();
    assert_eq!(want.len(), tokens.len());
    let got = model
        .forward(&Tensor::from_slice(tokens, &[1, tokens.len()]).unwrap())
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    // A reference value of -f32::MAX marks a suppressed token (Chameleon's image ids): it must come out just as
    // hopeless and stays out of the scale and the error.
    const SENTINEL: f32 = 1e30;
    let row_error = |pos: usize, row: &Vec<f32>| {
        let scale = row
            .iter()
            .filter(|v| v.abs() < SENTINEL)
            .fold(0.0f32, |m, v| m.max(v.abs()))
            .max(1e-6);
        row.iter().enumerate().fold(0.0f32, |m, (j, w)| {
            let g = got[pos * vocab + j];
            if w.abs() >= SENTINEL {
                assert!(g <= -SENTINEL, "{stem}: suppressed token {j} at {pos} came out as {g}");
                m
            } else if g <= -SENTINEL && stem.starts_with("chameleon") && (4..8196).contains(&j) {
                // llama.cpp hides the image ids with a flat `ggml_set_1d`, which only reaches the first output row (the
                // only row when generating); this crate hides them in every row.
                m
            } else {
                m.max((g - w).abs() / scale)
            }
        })
    };
    let worst = want
        .iter()
        .enumerate()
        .fold(0.0f32, |m, (pos, row)| m.max(row_error(pos, row)));
    if std::env::var_os("OXIDEON_PARITY_VERBOSE").is_some() {
        for (pos, row) in want.iter().enumerate() {
            eprintln!("  {stem} pos {pos}: {:.2e}", row_error(pos, row));
        }
    }
    eprintln!("{stem}: worst error relative to the logit scale = {worst:.3e}");
    assert!(worst < 2e-3, "{stem} disagrees with llama.cpp: {worst}");
}

/// Tokens `1..=127` repeated, so a prompt can exceed the 48-token default and the 127-token vocabulary.
fn long_prompt(n: usize) -> Vec<i64> {
    (0..n as i64).map(|i| i % 127 + 1).collect()
}

/// 200 tokens: long enough for DeepSeek-V4's ratio-128 layer to see a complete compressed block.
#[test]
fn deepseek4_long_prompt_matches_llama_cpp() {
    parity_tokens("deepseek4-long", &long_prompt(200));
}

/// The same model with `indexer.top_k = 3`, so the compressed-block indexer really selects among 50 blocks.
#[test]
fn deepseek4_indexer_top_k_matches_llama_cpp() {
    parity_tokens("deepseek4-topk", &long_prompt(200));
}

/// Layer 0 routes by token id (`hash_layer_count = 1`) and each token uses a single expert, so the table matters.
#[test]
fn deepseek4_hash_routing_matches_llama_cpp() {
    parity_tokens("deepseek4-hash", &long_prompt(200));
}

/// YaRN scaling (factor 4) on the compressing layers' rotary schedule.
#[test]
fn deepseek4_yarn_matches_llama_cpp() {
    parity_tokens("deepseek4-yarn", &long_prompt(200));
}

/// The sparse-attention MLA models with `indexer.top_k = 4` (`tools/ggml_reference/topk_variants.py`) on a 100-token
/// prompt, so the lightning indexer selects among far more tokens than it keeps.
#[test]
fn deepseek32_indexer_top_k_matches_llama_cpp() {
    parity_tokens("deepseek32-topk", &long_prompt(100));
}

#[test]
fn glm_dsa_indexer_top_k_matches_llama_cpp() {
    parity_tokens("glm-dsa-topk", &long_prompt(100));
}

#[test]
fn dots3note_indexer_top_k_matches_llama_cpp() {
    parity_tokens("dots3note-topk", &long_prompt(100));
}

#[test]
fn hy_v4_indexer_top_k_matches_llama_cpp() {
    parity_tokens("hy_v4-topk", &long_prompt(100));
}

/// GLM-5-Next with `indexer.top_k = 8` and `kpool = 4`: two pooled-key blocks plus the incomplete tail per token.
#[test]
fn glm5_next_kpool_indexer_matches_llama_cpp() {
    parity_tokens("glm5-next-topk", &long_prompt(100));
}

/// MiniMax-M3 with `indexer.top_k = 3` blocks of 4 tokens (one forced local block): 25 blocks on a 100-token prompt.
/// The variant scales the attention projections up (see `OXIDEON_TENSOR_GAIN` in `topk_variants.py`): in the stock tiny
/// model the attention output barely reaches the logits, so a wrong selection would go unnoticed.
#[test]
fn minimax_m3_block_indexer_matches_llama_cpp() {
    parity_tokens("minimax-m3-topk", &long_prompt(100));
}

/// Qwen4-Exp with `indexer.top_k = 8` and `kpool = 4`: two pooled blocks plus the incomplete tail per token.
#[test]
fn qwen4exp_qsa_indexer_matches_llama_cpp() {
    parity_tokens("qwen4exp-topk", &long_prompt(100));
}

/// Architectures llama.cpp's test tool cannot save: `tools/ggml_reference/tiny_models.py` writes tiny random models
/// for them (into `OXIDEON_LLAMA_EXTRA_MODELS`), and the reference logits come from `llama_logits` on those files.
/// Apertus: Q/K-normed attention and an ungated xIELU feed-forward with per-layer parameters.
#[test]
fn apertus_matches_llama_cpp() {
    parity_tokens("apertus", &long_prompt(48));
}

/// Chameleon: per-head LayerNorm Q/K norms and the image-token logits forced to -f32::MAX (vocabulary of 8200).
#[test]
fn chameleon_matches_llama_cpp() {
    parity_tokens("chameleon", &long_prompt(48));
}

/// Chameleon's `swin_norm` layout (the norms follow the attention / FFN outputs). llama.cpp feeds the FFN the normed
/// attention output instead of the residual sum that the reference implementation uses, so this crate deliberately
/// differs there; the variant silences the FFN (`ffn_down` = 0) and compares the attention half only.
#[test]
fn chameleon_swin_norm_attention_matches_llama_cpp() {
    parity_tokens("chameleon-swin", &long_prompt(48));
}

/// Gemma 3n: four AltUp streams, LAuReL, per-layer inputs, activation sparsity (first ten layers) and shared K/V
/// (layers 20 and up, so the model has 24 layers) on a 48-token prompt.
#[test]
fn gemma3n_matches_llama_cpp() {
    parity_tokens("gemma3n", &long_prompt(48));
}

/// RWKV-6: data-dependent token shift, per-head WKV recurrence with a matrix state, group norm, channel mix, and a
/// halved residual every second layer.
#[test]
fn rwkv6_matches_llama_cpp() {
    parity_tokens("rwkv6", &long_prompt(48));
}

/// RWKV-7: delta-rule WKV state, low-rank decay / learning-rate / value-residual / gate paths, layer 0's values blended
/// into the later layers.
#[test]
fn rwkv7_matches_llama_cpp() {
    parity_tokens("rwkv7", &long_prompt(48));
}

/// ARWKV7: the RWKV-7 time mix behind RMS norms with a SwiGLU feed-forward (one token-shift slot, no group norm).
#[test]
fn arwkv7_matches_llama_cpp() {
    parity_tokens("arwkv7", &long_prompt(48));
}

/// QRWKV6: gated linear attention with grouped key / value heads, biases, RMS norms and a SwiGLU feed-forward.
#[test]
fn rwkv6qwen2_matches_llama_cpp() {
    parity_tokens("rwkv6qwen2", &long_prompt(48));
}

/// Step 3.5: per-layer head counts and rotary widths, a head-wise attention gate on some layers, sigmoid-routed experts
/// with a shared expert and per-layer SwiGLU clamps.
#[test]
fn step35_matches_llama_cpp() {
    parity_tokens("step35", &long_prompt(48));
}

/// Gemma 4 (dense): per-layer head and rotary widths, values taken from the keys, shared K/V on the last two layers,
/// per-layer input embeddings and output scalars.
#[test]
fn gemma4_matches_llama_cpp() {
    parity_tokens("gemma4", &long_prompt(48));
}

/// Gemma 4 with MoE layers: a dense MLP and softmax-routed GELU experts (per-expert scales) side by side, the router
/// reading the normed residual, each branch behind its own post norm.
#[test]
fn gemma4_moe_matches_llama_cpp() {
    parity_tokens("gemma4-moe", &long_prompt(48));
}

/// PLM: MLA with a single Q projection and the rotary dims at the end of each head, ReLU^2 FFN, tied head.
#[test]
fn plm_matches_llama_cpp() {
    parity_tokens("plm", &long_prompt(48));
}

/// BitNet: RMS "sub norms" over the attention output and over the FFN's inner activation, tied output head.
#[test]
fn bitnet_matches_llama_cpp() {
    parity_tokens("bitnet", &long_prompt(48));
}

/// BERT encoder (post-LayerNorm blocks, learned positions, token-type row 0) against `llama_embed`'s output, for each
/// pooling type: mean, CLS, and none (a vector per token). 40 tokens.
fn bert_embeddings(stem: &str) {
    ggml_numerics();
    let (Some(refs), Some(models)) = (
        std::env::var_os("OXIDEON_LLAMA_LOGITS"),
        std::env::var_os("OXIDEON_LLAMA_EXTRA_MODELS"),
    ) else {
        eprintln!("skipping {stem}: OXIDEON_LLAMA_* not set");
        return;
    };
    let path = PathBuf::from(models).join(format!("{stem}.gguf"));
    if !path.exists() {
        eprintln!("skipping {stem}: {} not found", path.display());
        return;
    }
    let enc = oxideon::nn::gguf_bert::BertEncoder::from_gguf(&GGUFFile::open(path).unwrap()).unwrap();
    let text = std::fs::read_to_string(PathBuf::from(refs).join(format!("{stem}.embed"))).unwrap();
    let mut lines = text.lines();
    let head: Vec<usize> = lines
        .next()
        .unwrap()
        .split(' ')
        .skip(1)
        .step_by(2)
        .map(|v| v.parse().unwrap())
        .collect();
    let want: Vec<f32> = lines
        .flat_map(|l| l.split(' ').map(|v| v.parse::<f32>().unwrap()))
        .collect();
    assert_eq!(want.len(), head[0] * head[1]);
    let tokens = long_prompt(40);
    let got = enc.embed(&tokens).unwrap().to_vec::<f32>().unwrap();
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let worst = got
        .iter()
        .zip(&want)
        .fold(0.0f32, |m, (g, w)| m.max((g - w).abs() / scale));
    eprintln!("{stem}: worst relative embedding error {worst:.3e}");
    assert!(worst < 5e-4, "{stem}: {worst:e}");
}

#[test]
fn bert_mean_pooled_matches_llama_cpp() {
    bert_embeddings("bert");
}

#[test]
fn bert_cls_pooled_matches_llama_cpp() {
    bert_embeddings("bert-cls");
}

#[test]
fn bert_per_token_matches_llama_cpp() {
    bert_embeddings("bert-none");
}

/// nomic-bert: RoPE, fused QKV and a SwiGLU FFN in the same post-LayerNorm encoder.
#[test]
fn nomic_bert_matches_llama_cpp() {
    bert_embeddings("nomic-bert");
}

/// jina-bert-v2: ALiBi attention, per-width Q / K norms, a second post-attention norm and a fused GEGLU FFN.
#[test]
fn jina_bert_v2_matches_llama_cpp() {
    bert_embeddings("jina-bert-v2");
}

/// jina-bert-v2 with a separate FFN gate tensor.
#[test]
fn jina_bert_v2_gate_matches_llama_cpp() {
    bert_embeddings("jina-bert-v2-gate");
}

/// Reranker head: CLS-pooled `cls` + tanh + `cls.output` (3 labels).
#[test]
fn bert_rank_cls_matches_llama_cpp() {
    bert_embeddings("bert-rank");
}

/// Reranker head with mean pooling before the head (`classifier.pooling_type = mean`) and a GELU activation.
#[test]
fn bert_rank_mean_matches_llama_cpp() {
    bert_embeddings("bert-rank-mean");
}
