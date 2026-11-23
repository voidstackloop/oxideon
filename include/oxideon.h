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

#ifdef __cplusplus
}
#endif

#endif /* OXIDEON_H */
