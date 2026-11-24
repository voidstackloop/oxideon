#ifndef OXIDEON_H
#define OXIDEON_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OxideonTensor OxideonTensor;

const char* oxideon_last_error_message(void);

OxideonTensor* oxideon_tensor_from_f32(
    const float* data,
    size_t len,
    const size_t* shape,
    size_t ndim
);

void oxideon_tensor_free(OxideonTensor* t);

size_t oxideon_tensor_numel(const OxideonTensor* t);
size_t oxideon_tensor_ndim(const OxideonTensor* t);

int32_t oxideon_tensor_copy_f32(
    const OxideonTensor* t,
    float* out,
    size_t cap
);

/* Writes the shape into out[cap]; returns 0, or -1 if cap < ndim. */
int32_t oxideon_tensor_shape(
    const OxideonTensor* t,
    size_t* out,
    size_t cap
);

OxideonTensor* oxideon_tensor_add(
    const OxideonTensor* a,
    const OxideonTensor* b
);

OxideonTensor* oxideon_tensor_matmul(
    const OxideonTensor* a,
    const OxideonTensor* b
);

OxideonTensor* oxideon_tensor_relu(const OxideonTensor* x);

/* Elementwise activations. */
OxideonTensor* oxideon_tensor_gelu(const OxideonTensor* x);
OxideonTensor* oxideon_tensor_silu(const OxideonTensor* x);

/* Softmax along `axis` (negative axes count from the last dimension). */
OxideonTensor* oxideon_tensor_softmax(const OxideonTensor* x, int64_t axis);

/* Reshape to shape[ndim]; the element count must not change. */
OxideonTensor* oxideon_tensor_reshape(
    const OxideonTensor* x,
    const size_t* shape,
    size_t ndim
);

/* Swap two dimensions (negative dims count from the last dimension). */
OxideonTensor* oxideon_tensor_transpose(
    const OxideonTensor* x,
    int64_t dim0,
    int64_t dim1
);

/* Layer norm over the trailing `normalized_ndim` dims (sizes in normalized_shape).
 * `weight` and `bias` may be NULL. */
OxideonTensor* oxideon_tensor_layer_norm(
    const OxideonTensor* x,
    const size_t* normalized_shape,
    size_t normalized_ndim,
    const OxideonTensor* weight,
    const OxideonTensor* bias,
    double eps
);

/* 2D convolution: input [N, C_in, H, W], weight [C_out, C_in, kH, kW], optional bias [C_out] (may be NULL). */
OxideonTensor* oxideon_tensor_conv2d(
    const OxideonTensor* input,
    const OxideonTensor* weight,
    const OxideonTensor* bias,
    size_t stride_h,
    size_t stride_w,
    size_t pad_h,
    size_t pad_w,
    size_t dilation_h,
    size_t dilation_w
);

#ifdef __cplusplus
}
#endif

#endif /* OXIDEON_H */
