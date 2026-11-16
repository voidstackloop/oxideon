// Prints llama.cpp's logits for every position of a token sequence, for parity tests against
// oxideon's GGUFCausalLM (tests/gguf_llama_cpp_parity.rs).
//
//   gcc -O1 -I $LLAMA/include -I $LLAMA/ggml/include tools/ggml_reference/llama_logits.c \
//       -L $LLAMA/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$LLAMA/build/bin -o llama_logits
//   ./llama_logits model.gguf 1 2 3 4 5 > model.logits
//
// Output: "vocab <V> tokens <N>" then N lines of V floats (%.9g).
#include "llama.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// LLAMA_LOGITS_DUMP=name,name,... prints the first values (token 0) of the named graph tensors to stderr, e.g.
// "attn_laurel-0" - a debugging aid for finding the first layer where a re-implementation diverges.
static const char * dump_names = NULL;

static bool dump_cb(struct ggml_tensor * t, bool ask, void * ud) {
    (void) ud;
    const char * hit = NULL;
    for (const char * p = dump_names; p && *p;) {
        const char * end = strchr(p, ',');
        size_t len = end ? (size_t) (end - p) : strlen(p);
        if (strlen(t->name) == len && strncmp(t->name, p, len) == 0) { hit = p; break; }
        p = end ? end + 1 : NULL;
    }
    if (!hit) return ask ? false : true;
    if (ask) return true;
    if (t->type != GGML_TYPE_F32) return true;
    float v[8];
    int m = t->ne[0] < 8 ? (int) t->ne[0] : 8;
    // LLAMA_LOGITS_DUMP_TOKEN=k prints token k instead of token 0 (the tensors are [width, tokens, ...])
    const char * tok = getenv("LLAMA_LOGITS_DUMP_TOKEN");
    size_t off = tok ? (size_t) atoi(tok) * t->nb[1] : 0;
    ggml_backend_tensor_get(t, v, off, m * sizeof(float));
    fprintf(stderr, "DUMP %s:", t->name);
    for (int i = 0; i < m; i++) fprintf(stderr, " %.6g", v[i]);
    fprintf(stderr, "\n");
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.gguf tok...\n", argv[0]); return 1; }
    const int n = argc - 2;
    llama_token * toks = malloc(sizeof(llama_token) * n);
    for (int i = 0; i < n; i++) toks[i] = atoi(argv[i + 2]);

    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    struct llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 2; }
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256; cp.n_batch = 256; cp.n_ubatch = 256; cp.n_threads = 4;
    // LLAMA_LOGITS_KV_F32=1 keeps the K/V cache in f32, LLAMA_LOGITS_NO_FA=1 turns flash attention off (whose CPU kernel
    // rounds K/V to f16): both only matter for models whose attention output is large against the residual stream
    if (getenv("LLAMA_LOGITS_KV_F32")) { cp.type_k = GGML_TYPE_F32; cp.type_v = GGML_TYPE_F32; }
    if (getenv("LLAMA_LOGITS_NO_FA")) { cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; }
    dump_names = getenv("LLAMA_LOGITS_DUMP");
    if (dump_names) { cp.cb_eval = dump_cb; cp.cb_eval_user_data = NULL; }
    struct llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 3; }

    struct llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        b.token[i] = toks[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
    }
    b.n_tokens = n;
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); return 4; }

    const int v = llama_vocab_n_tokens(llama_model_get_vocab(model));
    printf("vocab %d tokens %d\n", v, n);
    for (int i = 0; i < n; i++) {
        const float * l = llama_get_logits_ith(ctx, i);
        for (int j = 0; j < v; j++) printf(j ? " %.9g" : "%.9g", l[j]);
        printf("\n");
    }
    llama_batch_free(b); llama_free(ctx); llama_model_free(model);
    return 0;
}
