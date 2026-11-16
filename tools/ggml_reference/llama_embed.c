// Prints the embeddings llama.cpp computes for a token sequence: the pooled sequence vector when the model has a
// pooling type, otherwise one vector per token. Build like llama_logits.c (see README.md).
//   usage: llama_embed model.gguf tok...
#include "llama.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.gguf tok...\n", argv[0]); return 1; }
    const int n = argc - 2;

    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    struct llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 2; }
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256; cp.n_batch = 256; cp.n_ubatch = 256; cp.n_threads = 4;
    cp.embeddings = true;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; // flash attention rounds K/V through f16
    struct llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 3; }

    struct llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        b.token[i] = atoi(argv[i + 2]); b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
    }
    b.n_tokens = n;
    if (llama_encode(ctx, b) != 0) { fprintf(stderr, "encode failed\n"); return 4; }

    const int d = llama_model_n_embd(model);
    const int pooled = llama_pooling_type(ctx) != LLAMA_POOLING_TYPE_NONE;
    const int rank = llama_pooling_type(ctx) == LLAMA_POOLING_TYPE_RANK;
    const int dout = rank ? (int) llama_model_n_cls_out(model) : d;
    printf("dim %d rows %d\n", dout, pooled ? 1 : n);
    for (int i = 0; i < (pooled ? 1 : n); i++) {
        const float * e = pooled ? llama_get_embeddings_seq(ctx, 0) : llama_get_embeddings_ith(ctx, i);
        for (int j = 0; j < dout; j++) printf(j ? " %.9g" : "%.9g", e[j]);
        printf("\n");
    }
    llama_batch_free(b); llama_free(ctx); llama_model_free(model);
    return 0;
}
