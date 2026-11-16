// Dumps llama.cpp's own dequantization of random blocks as a fixture for the Rust decoders.
#include "ggml.h"
#include "ggml-quants.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void ggml_abort(const char * file, int line, const char * fmt, ...) { (void)fmt; fprintf(stderr, "abort %s:%d\n", file, line); exit(2); }
size_t ggml_row_size(enum ggml_type t, int64_t n) { (void)t; (void)n; exit(3); }
const char * ggml_type_name(enum ggml_type t) { (void)t; return "?"; }
size_t ggml_type_size(enum ggml_type t) { (void)t; exit(4); }

typedef void (*deq_fn)(const void *, float *, int64_t);
// d_off / m_off: byte offsets of the f16 scale and min (or -1) so they can be kept finite and moderate.
typedef struct { int id; const char * name; size_t bytes; int64_t n; deq_fn fn; int d_off; int m_off; } entry;

#define E(id, nm, bytes, n, d, m) { id, #nm, bytes, n, (deq_fn)dequantize_row_##nm, d, m }
static const entry entries[] = {
    E(2, q4_0, 18, 32, 0, -1), E(3, q4_1, 20, 32, 0, 2), E(6, q5_0, 22, 32, 0, -1), E(7, q5_1, 24, 32, 0, 2),
    E(8, q8_0, 34, 32, 0, -1),
    E(10, q2_K, 84, 256, 80, 82), E(11, q3_K, 110, 256, 108, -1), E(12, q4_K, 144, 256, 0, 2),
    E(13, q5_K, 176, 256, 0, 2), E(14, q6_K, 210, 256, 208, -1),
    E(16, iq2_xxs, 66, 256, 0, -1), E(17, iq2_xs, 74, 256, 0, -1), E(18, iq3_xxs, 98, 256, 0, -1),
    E(19, iq1_s, 50, 256, 0, -1), E(20, iq4_nl, 18, 32, 0, -1), E(21, iq3_s, 110, 256, 0, -1),
    E(22, iq2_s, 82, 256, 0, -1), E(23, iq4_xs, 136, 256, 0, -1), E(29, iq1_m, 56, 256, -1, -1),
    E(34, tq1_0, 54, 256, 52, -1), E(35, tq2_0, 66, 256, 64, -1), E(39, mxfp4, 17, 32, -1, -1),
    E(40, nvfp4, 36, 64, -1, -1), E(41, q1_0, 18, 128, 0, -1), E(42, q2_0, 18, 64, 0, -1),
};

static uint64_t s = 0x9E3779B97F4A7C15ULL;
static uint8_t rnd(void) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint8_t)(s >> 24); }

static void moderate_f16(uint8_t * p) { p[1] = 0x30 | (p[1] & 0x07); } // finite half around 0.1..1

int main(void) {
    for (size_t e = 0; e < sizeof(entries) / sizeof(entries[0]); e++) {
        const entry * en = &entries[e];
        for (int blk = 0; blk < 3; blk++) {
            uint8_t buf[256]; float out[256];
            for (;;) {
                for (size_t i = 0; i < en->bytes; i++) buf[i] = rnd();
                if (en->d_off >= 0) moderate_f16(buf + en->d_off);
                if (en->m_off >= 0) moderate_f16(buf + en->m_off);
                if (en->id == 12 || en->id == 13) { /* dmin already handled via m_off */ }
                if (en->id == 29) { // iq1_m: the f16 scale is spread over the top nibbles of 4 u16 words
                    uint16_t h = (uint16_t)(0x3000 | ((rnd() << 8 | rnd()) & 0x07ff));
                    uint16_t * sc = (uint16_t *)(buf + 48);
                    sc[0] = (sc[0] & 0x0fff) | ((h & 0x000f) << 12);
                    sc[1] = (sc[1] & 0x0fff) | ((h & 0x00f0) << 8);
                    sc[2] = (sc[2] & 0x0fff) | ((h & 0x0f00) << 4);
                    sc[3] = (sc[3] & 0x0fff) | (h & 0xf000);
                }
                en->fn(buf, out, en->n);
                int ok = 1;
                for (int64_t i = 0; i < en->n; i++) if (!isfinite(out[i])) ok = 0;
                if (ok) break;
            }
            uint64_t h = 14695981039346656037ULL;
            uint32_t bits[256];
            for (int64_t i = 0; i < en->n; i++) {
                memcpy(&bits[i], &out[i], 4);
                for (int b = 0; b < 4; b++) { h ^= (bits[i] >> (8 * b)) & 0xff; h *= 1099511628211ULL; }
            }
            printf("%d ", en->id);
            for (size_t i = 0; i < en->bytes; i++) printf("%02x", buf[i]);
            printf(" %08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x %016llx\n", bits[0], bits[1], bits[2], bits[3],
                   bits[4], bits[5], bits[6], bits[7], (unsigned long long)h);
        }
    }
    return 0;
}
