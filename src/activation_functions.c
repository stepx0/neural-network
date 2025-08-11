#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include "activation_functions.h"
#include "tensor.h"

#define SQRTF_2_OVER_PI 0.79788456f // ≈ sqrt(2 / π)
#define SQRTF_2 1.4142136f          // sqrt(2)
#define M_PIF 3.1415927f            // π

/*------------------------------
 * Scalar activation functions 
 *------------------------------*/

float sigmoid(float x, float alpha) {
    (void)alpha; //unused
    return 1.f / (1.f + expf(-x));
}

/* 
 * 'output' is expected to already be the result of the sigmoid function,
 * so we can avoid redundant computation and ensure better performance.
 */
float sigmoid_derivative(float x, float y, float alpha) {
    (void)x; // unused, derivative depends only on output
    (void)alpha; //unused
    return y * (1.f - y);
}

float relu(float x, float alpha) {
    (void)alpha; //unused
    return (x > 0) ? x : 0.f;
}

float relu_derivative(float x, float y, float alpha) {
    (void)y; // unused, derivative depends only on input sign
    (void)alpha; //unused
    return (x > 0) ? 1.f : 0.f;
}

/*
 * Custom implementation of tanh, for educational purposes.
 * For better performance and precision, consider using tanh() from <math.h>.
 *
 * To improve efficiency, 'e_x_negative' can be replaced with (1.f / e_x), to avoid two calls to expf().
 * This lighter version proposed slightly reduces precision.
 */
float tanh_custom(float x, float alpha) {
    (void)alpha; //unused
    float e_x = expf(x);
    float e_x_negative = expf(-x);

    return ((e_x - e_x_negative)/(e_x + e_x_negative));
}

/* 
 * 'output' is expected to already be the result of the tanh function,
 * so we can avoid redundant computation and ensure better performance.
 */
float tanh_derivative(float x, float y, float alpha) {
    (void)x; // unused
    (void)alpha; //unused
    return 1.f - y * y;
}

float leaky_relu(float x, float alpha) {
    return (x > 0) ? x : alpha * x;
}

float leaky_relu_derivative(float x, float y, float alpha) {
    (void)y; // unused
    return (x > 0) ? 1.f : alpha;
}

float elu(float x, float alpha) {
    return (x >= 0.f) ? x : (alpha * (expf(x) - 1.f));
}

float elu_derivative(float x, float y, float alpha) {
    (void)y; // unused
    return (x > 0.f) ? 1.f : (alpha * expf(x));
}

float swish(float x, float alpha) {
    (void)alpha; //unused
    return x * sigmoid(x, 0.f); // passed a useless 'alpha' value, not needed
}

float swish_derivative(float x, float y, float alpha) {
    (void)y; // unused
    (void)alpha; // unused
    float sigm_x = sigmoid(x, 0.f);
    return sigm_x + (x * sigm_x * (1 - sigm_x));
}

float gelu(float x, float alpha) {
    (void)alpha; //unused
    return x * 0.5f * (1+erff(x / SQRTF_2));
}

/*
 * Derivative formula:
 * dx/d​[x ⋅ Φ(x)] = Φ(x) + x ⋅ ϕ(x)
 * 
 * Φ(x) = 0.5 ⋅ (1 + erf(x / √(2)))
 * ϕ(x) = (1/(√(2*π)) ⋅ expf(-0.5 ⋅ x^2))
 */
float gelu_derivative(float x, float y, float alpha) {
    (void)y; // unused
    (void)alpha; //unused
    float Phi = 0.5f * (1.f + erff(x / SQRTF_2));
    float phi = expf(-0.5f * x * x) / (SQRTF_2 * M_PIF);

    return Phi + x * phi;
}

float gelu_approx(float x, float alpha) {
    (void)alpha; //unused
    float x3 = x * x * x;
    float inner = SQRTF_2_OVER_PI * (x + GELU_COEFF * x3);
    return 0.5f * x * (1.f + tanhf(inner));
}

float gelu_approx_derivative(float x, float y, float alpha) {
    (void)y; // unused
    (void)alpha; //unused
    float x2 = x * x;
    float x3 = x * x * x;
    float tan_u = tanhf(SQRTF_2_OVER_PI * (x + GELU_COEFF * x3));
    float sech2_u = 1.f - tan_u * tan_u; // tan_u derivative

    float u_prime = SQRTF_2_OVER_PI * (1.f + 3.f * GELU_COEFF * x2);

    return 0.5f * (1 + tan_u) + 0.5f * x * sech2_u * u_prime;
}

/*------------------------------
 * Vector activation functions 
 *------------------------------*/

/* Small helper for numerically stable softmax over a 1D array. */
static void softmax_vector(const float* input, float* output, size_t length) {
    if (length == 0) return;

    float max_val = input[0];
    for (size_t i = 1; i < length; i++)
        if (input[i] > max_val)
            max_val = input[i];

    float sum = 0.f;
    for (size_t i = 0; i < length; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    for (size_t i = 0; i < length; i++)
        output[i] /= sum;
}

/* Convert a flat slice index to multi-dimensional indices (excluding 'axis'). */
static void linear_to_multi_index(size_t linear_idx, const Tensor *t, size_t axis, size_t *out_indices) {
    for (int i = (int)t->ndim - 1, j = (int)t->ndim - 2; i >= 0; i--) {
        if ((size_t)i == axis) continue;
        out_indices[j] = linear_idx % t->dims[i];
        linear_idx /= t->dims[i];
        j--;
    }
}

/* Calculate flat offset given all indices (excluding axis, plus axis_index). */
static size_t calc_offset(const size_t* indices, const Tensor *t, size_t axis, size_t axis_index) {
    size_t offset = 0;
    for (size_t i = 0; i < t->ndim; i++) {
        size_t idx;
        if (i == axis) {
            idx = axis_index;
        } else {
            size_t pos = (i < axis) ? i : i - 1;
            idx = indices[pos];
        }
        offset += idx * t->strides[i];
    }
    return offset;
}

/* 
 * Params:
 * - input: tensor flattened into a 1D array
 * - output: it's also a flattened array
 * - t: Tensor descriptor
 * - axis: axis where to apply the softmax function
 */
void softmax_tensor(const Tensor *t, const float* input, float* output, size_t axis) {
    size_t total = tensor_numel(t);
    size_t axis_dim = t->dims[axis];
    size_t slice_count = total / axis_dim;

    size_t* indices = (size_t*)malloc((t->ndim - 1) * sizeof(size_t));
    float* input_slice = (float*)malloc(axis_dim * sizeof(float));
    float* output_slice = (float*)malloc(axis_dim * sizeof(float));

    for (size_t slice = 0; slice < slice_count; slice++) {
        linear_to_multi_index(slice, t, axis, indices);

        for (size_t i = 0; i < axis_dim; i++) {
            size_t offset = calc_offset(indices, t, axis, i);
            input_slice[i] = input[offset];
        }

        softmax_vector(input_slice, output_slice, axis_dim);

        for (size_t i = 0; i < axis_dim; i++) {
            size_t offset = calc_offset(indices, t, axis, i);
            output[offset] = output_slice[i];
        }
    }

    free(indices);
    free(input_slice);
    free(output_slice);
}

/*
 * Computes the derivative of the softmax function for a given softmax output vector.
 *
 * Note: This implementation assumes you're applying the derivative for a softmax
 * followed by a cross-entropy loss function, which simplifies the derivative to:
 * 
 *     dL/dz = softmax_output - target
 *
 * However, if you're computing the Jacobian (full matrix of derivatives), that would
 * require a 2D output, which is not teh case here because of Jacobian computational complexity:
 * (O(n^2) for input size n).
 *
 * Parameters:
 * - softmax_output: The result of the softmax function (input vector).
 * - output_derivative: Array where the derivative will be stored (same size as softmax_output).
 * - length: Number of elements in the softmax vector.
 *
 * This simplified version assumes you're computing dsoftmax_i/dz_i = s_i * (1 - s_i)
 * for use cases like standalone gradient inspection or compatibility with other loss functions.
 */
void softmax_derivative(const float* softmax_output, float* output_derivative, size_t length) {
    for (size_t i = 0; i < length; i++) {
        float s = softmax_output[i];
        output_derivative[i] = s * (1.f - s);
    }
}

/*
 * Computes the Jacobian matrix of the softmax function for a given input vector.
 *
 * Parameters:
 * - softmax_output: Pointer to the softmax output vector.
 * - jacobian: Pointer to a preallocated 2D array (flattened as 1D) of size length*length
 *             where the computed Jacobian matrix will be stored.
 * - length: The size of the softmax output vector.
 *
 * Note:
 * The Jacobian is stored in a row-major order vector:
 *   jacobian[i * length + j] = d softmax[i] / d input[j]
 */
void softmax_jacobian_derivative(const float* softmax_output, float* jacobian, size_t length) {
    for (size_t i = 0; i < length; i++) {
        for (size_t j = 0; j < length; j++) {
            if (i == j) {
                jacobian[i * length + j] = softmax_output[i] * (1.f - softmax_output[i]);
            } else {
                jacobian[i * length + j] = -softmax_output[i] * softmax_output[j];
            }
        }
    }
}
