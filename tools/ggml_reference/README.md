# GGML reference data

Both files need a checkout of llama.cpp (path as `$LLAMA`).

* `gen_tables.py $LLAMA [out.rs]` regenerates `crates/oxideon-core/src/serialize/gguf_iq_tables.rs`
  (the IQ codebooks) from `ggml/src/ggml-common.h`.
* `harness.c` links GGML's own `dequantize_row_*` and prints random blocks plus a hash of the exact
  f32 output, i.e. `crates/oxideon-core/src/serialize/testdata/ggml_reference_blocks.txt`:

```sh
gcc -O1 -I $LLAMA/ggml/include -I $LLAMA/ggml/src tools/ggml_reference/harness.c \
    $LLAMA/ggml/src/ggml-quants.c -lm -o harness && ./harness > \
    crates/oxideon-core/src/serialize/testdata/ggml_reference_blocks.txt
```
The four GGML symbols `ggml-quants.c` needs at link time are stubbed in the harness.

## Logit parity with llama.cpp

`llama_logits.c` prints llama.cpp's logits for a token sequence; `tests/gguf_llama_cpp_parity.rs` compares
them with `GGUFCausalLM` on the tiny random-weight models that llama.cpp's own test suite writes
(`llama-dense`, `llama-moe`, `qwen2-dense`, `qwen2moe-moe`):

```sh
gcc -O1 -I $LLAMA/include -I $LLAMA/ggml/include tools/ggml_reference/llama_logits.c \
    -L $LLAMA/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$LLAMA/build/bin -o llama_logits
M=$LLAMA/build/tests/test-models; mkdir -p logits
for m in llama-dense llama-moe qwen2-dense qwen2moe-moe; do ./llama_logits $M/$m.gguf $(seq 1 48) > logits/$m.logits; done
OXIDEON_LLAMA_TEST_MODELS=$M OXIDEON_LLAMA_LOGITS=$PWD/logits cargo test --test gguf_llama_cpp_parity
```
Without the two environment variables the parity tests skip themselves.

### Models llama.cpp's test tool cannot write

* `topk_variants.py` copies the sparse-attention test models with a small `indexer.top_k` (and optional tensor gains,
  `OXIDEON_TENSOR_GAIN`) so the indexers really select on a 100-token prompt.
* `tiny_models.py` writes tiny random-weight GGUFs for architectures the tool cannot save (apertus, bitnet, plm,
  chameleon): it reuses a test model's key/value block under the new architecture and writes exactly the tensors that
  architecture's loader creates. Put them in `OXIDEON_LLAMA_EXTRA_MODELS`.
* `llama_embed.c` is the encoder counterpart: it prints the pooled (or per-token) embeddings of `llama_encode`, with flash
  attention off. `tiny_models.py` writes `bert`, `bert-cls`, `bert-none` (mean / CLS / no pooling) and `nomic-bert` for `nn::gguf_bert`.
* Their reference logits are made with `LLAMA_LOGITS_KV_F32=1 LLAMA_LOGITS_NO_FA=1 llama_logits ...`: llama.cpp's
  default f16 K/V cache and flash-attention kernel round the attention inputs, which swamps the comparison for models
  whose attention output is large against the residual stream (the stock test models hide this behind tiny scales).
