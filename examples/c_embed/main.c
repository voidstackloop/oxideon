/* Minimal C program using the Oxideon C API: a tiny "conv -> layer_norm -> GELU -> softmax" forward pass. */
#include <stdio.h>
#include <stdlib.h>

#include "oxideon.h"

static int check(const void* ptr, const char* what) {
    if (ptr != NULL) return 1;
    const char* msg = oxideon_last_error_message();
    fprintf(stderr, "%s failed: %s\n", what, msg ? msg : "unknown error");
    return 0;
}

int main(void) {
    /* A 1x1x4x4 "image" and a 2x1x3x3 convolution kernel (output channel 0: 3x3 box blur, channel 1: center tap only). */
    float img[16];
    for (int i = 0; i < 16; ++i) img[i] = (float)((i * i) % 7) / 4.0f;
    float kernel[18] = {0};
    for (int i = 0; i < 9; ++i) kernel[i] = 1.0f / 9.0f;
    kernel[9 + 4] = 1.0f;

    const size_t img_shape[4] = {1, 1, 4, 4};
    const size_t ker_shape[4] = {2, 1, 3, 3};
    OxideonTensor* x = oxideon_tensor_from_f32(img, 16, img_shape, 4);
    OxideonTensor* w = oxideon_tensor_from_f32(kernel, 18, ker_shape, 4);
    if (!check(x, "tensor_from_f32(img)") || !check(w, "tensor_from_f32(kernel)")) return 1;

    /* conv2d -> [1, 2, 2, 2] (stride 1, no padding, no bias) */
    OxideonTensor* conv = oxideon_tensor_conv2d(x, w, NULL, 1, 1, 0, 0, 1, 1);
    if (!check(conv, "conv2d")) return 1;

    /* flatten to [2, 4], normalize each row, then GELU and softmax along the last axis */
    const size_t flat_shape[2] = {2, 4};
    OxideonTensor* flat = oxideon_tensor_reshape(conv, flat_shape, 2);
    const size_t norm_shape[1] = {4};
    OxideonTensor* normed = flat ? oxideon_tensor_layer_norm(flat, norm_shape, 1, NULL, NULL, 1e-5) : NULL;
    OxideonTensor* act = normed ? oxideon_tensor_gelu(normed) : NULL;
    OxideonTensor* probs = act ? oxideon_tensor_softmax(act, -1) : NULL;
    if (!check(probs, "reshape/layer_norm/gelu/softmax")) return 1;

    float out[8];
    if (oxideon_tensor_copy_f32(probs, out, 8) != 0) {
        fprintf(stderr, "copy failed: %s\n", oxideon_last_error_message());
        return 1;
    }
    for (int row = 0; row < 2; ++row) {
        printf("row %d:", row);
        float sum = 0.0f;
        for (int col = 0; col < 4; ++col) {
            printf(" %.4f", out[row * 4 + col]);
            sum += out[row * 4 + col];
        }
        printf("  (sum = %.4f)\n", sum);
    }

    oxideon_tensor_free(probs);
    oxideon_tensor_free(act);
    oxideon_tensor_free(normed);
    oxideon_tensor_free(flat);
    oxideon_tensor_free(conv);
    oxideon_tensor_free(w);
    oxideon_tensor_free(x);
    return 0;
}
