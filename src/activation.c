#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include "activation.h"
#include "tensor_utils.h"

float sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}

/* 
 * 'output' is expected to already be the result of the sigmoid function,
 * so we can avoid redundant computation and ensure better performance.
 */
float sigmoid_derivative(float output) {
    return output * (1.f - output);
}

float relu(float x) {
    return (x > 0) ? x : 0.f;
}

float relu_derivative(float x) {
    return (x > 0) ? 1.f : 0.f;
}

/*
 * Custom implementation of tanh, for educational purposes.
 * For better performance and precision, consider using tanh() from <math.h>.
 *
 * To improve efficiency, 'e_x_negative' can be replaced with (1.f / e_x), to avoid two calls to expf().
 * This lighter version proposed slightly reduces precision.
 */
float tanh_custom(float x) {
    float e_x = expf(x);
    float e_x_negative = expf(-x);

    return ((e_x - e_x_negative)/(e_x + e_x_negative));
}

/* 
 * 'output' is expected to already be the result of the tanh function,
 * so we can avoid redundant computation and ensure better performance.
 */
float tanh_derivative(float output) {
    return 1.f - output * output;
}

float leaky_relu(float x) {
    return (x > 0) ? x : 0.01f;
}

float leaky_relu_derivative(float x) {
    return (x > 0) ? 1.f : 0.01f;
}

/* 
 * Params:
 * - input: tensor flattened into a 1D array
 * - output: it's also a flattened array
 * - dims: array that contains input dimension for each axis
 * - ndim: dims size
 * - axis: axis where to apply the softmax function
 */
void softmax(const float* input, float* output, const size_t* dims, size_t ndim, size_t axis) {
    
    size_t total = 1; // total input items
    for (size_t i = 0; i < ndim; i++) total *= dims[i];

    size_t axis_dim = dims[axis]; // dimension of the axis
    
    size_t slice_count = total / axis_dim; // num of slices along selected axis to cycle

    size_t* indices = malloc((ndim - 1) * sizeof(size_t)); // used to calculate the offset in the tensor

    // Allocate and compute strides once before the loop
    size_t* strides = malloc(ndim * sizeof(size_t));
    compute_strides(dims, ndim, strides);

    // Allocate temporary input and output slices
    float* input_slice = malloc(axis_dim * sizeof(float));
    float* output_slice = malloc(axis_dim * sizeof(float));

    for (size_t slice = 0; slice < slice_count; slice++) {
        linear_to_multi_index(slice, dims, ndim, axis, indices);

        // Extract input slice values
        for (size_t i = 0; i < axis_dim; i++) {
            size_t offset = calc_offset(indices, ndim, axis, i, strides);
            input_slice[i] = input[offset];
        }

        // Apply softmax to the slice
        softmax_vector(input_slice, output_slice, axis_dim);

        // Write results back to output tensor
        for (size_t i = 0; i < axis_dim; i++) {
            size_t offset = calc_offset(indices, ndim, axis, i, strides);
            output[offset] = output_slice[i];
        }
    }

    free(indices);
    free(strides);
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

float elu(float x, float alpha) {
    return (x >= 0.f) ? x : (alpha * (expf(x) - 1.f));
}


float elu_derivative(float x, float alpha) {
    return (x > 0.f) ? 1.f : (alpha * expf(x));
}

float swish(float x) {
    return x * sigmoid(x);
}

float swish_derivative(float x) {
    float sigm_x = sigmoid(x);

    return sigm_x + (x * sigm_x * (1 - sigm_x));
}

float gelu(float x) {
    return x * 0.5f * (1+erff(x / SQRTF_2));
}

/*
 * Derivative formula:
 * dx/d​[x ⋅ Φ(x)] = Φ(x) + x ⋅ ϕ(x)
 * 
 * Φ(x) = 0.5 ⋅ (1 + erf(x / √(2)))
 * ϕ(x) = (1/(√(2*π)) ⋅ expf(-0.5 ⋅ x^2))
 */
float gelu_derivative(float x) {
    float Phi = 0.5f * (1.f + erff(x / SQRTF_2));
    float phi = expf(-0.5f * x * x) / (SQRTF_2 * M_PIF);

    return Phi + x * phi;
}

float gelu_approx(float x) {
    float x3 = x * x * x;
    float inner = SQRTF_2_OVER_PI * (x + GELU_COEFF * x3);
    return 0.5f * x * (1.f + tanhf(inner));
}

float gelu_approx_derivative(float x) {
    float x2 = x * x;
    float x3 = x * x * x;
    float tan_u = tanhf(SQRTF_2_OVER_PI * (x + GELU_COEFF * x3));
    float sech2_u = 1.f - tan_u * tan_u; // tan_u derivative

    float u_prime = SQRTF_2_OVER_PI * (1.f + 3.f * GELU_COEFF * x2);

    return 0.5f * (1 + tan_u) + 0.5f * x * sech2_u * u_prime;
}

