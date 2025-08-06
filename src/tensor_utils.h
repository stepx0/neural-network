
#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

#include <math.h>


#define SQRTF_2_OVER_PI 0.79788456f // ≈ sqrt(2 / π)
#define SQRTF_2 1.4142136f          // sqrt(2)
#define M_PIF 3.1415927f            // π


void linear_to_multi_index(size_t linear_idx, const size_t* dims, size_t ndim, size_t axis, size_t* out_indices);
size_t calc_offset(const size_t* indices, size_t ndim, size_t axis, size_t axis_index, const size_t* strides);
void compute_strides(const size_t* dims, size_t ndim, size_t* strides);
void softmax_vector(const float* input, float* output, size_t length);

/* 
 * Clamp value to range [epsilon, 1 - epsilon].
 * epsilon is the minimum float positive const close to zero
 */
static inline float clamp_prob(float p) {
    const float epsilon = 1e-15f;
    if (p < epsilon) return epsilon;
    if (p > 1.f - epsilon) return 1.f - epsilon;
    return p;
}

#endif