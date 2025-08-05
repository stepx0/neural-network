#include <stddef.h>
#include "activation.h"

float sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}

/* The output parameter is expected to already be the result of the sigmoid function,
 * so we can avoid redundant computation and ensure better performance. */
float sigmoid_derivative(float output) {
    return output * (1.f - output);
}

float relu(float x) {
    return x > 0 ? x : 0.f;
}

float relu_derivative(float x) {
    return x > 0 ? 1.f : 0.f;
}

float tanh(float x) {
    return ((expf(x) - expf(-x))/(expf(x) + expf(-x)));
}

/* The output parameter is expected to already be the result of the tanh function,
 * so we can avoid redundant computation and ensure better performance. */
float tanh_derivative(float output) {
    return 1.f - output * output;
}

float leaky_relu(float x) {
    return x > 0 ? x : 0.01f;
}

float leaky_relu_derivative(float x) {
    return (x > 0) ? 1.f : 0.01f;
}

/* input: tensor flattened into a 1D array
 * output: it's also a flattened array
 * dims: array that contains input dimension for each axis
 * ndim: dims size
 * axis: axis where to apply the softmax function */
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
            size_t offset = calc_offset(indices, dims, ndim, axis, i, strides);
            input_slice[i] = input[offset];
        }

        // Apply softmax to the slice
        softmax_vector(input_slice, output_slice, axis_dim);

        // Write results back to output tensor
        for (size_t i = 0; i < axis_dim; i++) {
            size_t offset = calc_offset(indices, dims, ndim, axis, i, strides);
            output[offset] = output_slice[i];
        }
    }

    free(indices);
    free(strides);
    free(input_slice);
    free(output_slice);
}

/* 
 * Convert a flat linear index (i.e., slice index) into a set of multidimensional indices 
 * for a tensor, excluding the axis along which softmax is applied.
 *
 * Parameters:
 * - linear_idx: Current slice index (from 0 to slice_count - 1)
 * - dims: Array containing the size of each dimension of the input tensor
 * - ndim: Number of dimensions (i.e., length of dims[])
 * - axis: The axis along which softmax is applied (this axis is excluded in out_indices)
 * - out_indices: Output array of size (ndim - 1) that stores the computed indices 
 *                for all dimensions except the softmax axis
 */
void linear_to_multi_index(size_t linear_idx, const size_t* dims, size_t ndim, size_t axis, size_t* out_indices) {
    for (int i = ndim - 1, j = ndim - 2; i >= 0; i--) {
        if ((size_t)i == axis) continue;
        out_indices[j] = linear_idx % dims[i];
        linear_idx /= dims[i];
        j--;
    }
}

/*
 * Calculates the flat (1D) offset in the input/output array corresponding to the given multi-dimensional indices.
 *
 * Parameters:
 * - indices: Array containing the multi-dimensional indices of the current slice (excluding the axis dimension).
 * - dims: Array containing the size of each dimension of the input tensor.
 * - ndim: Number of dimensions (length of dims).
 * - axis: The axis along which softmax is applied (this axis is excluded from indices).
 * - axis_index: The index along the 'axis' dimension.
 *
 * Returns:
 * - The computed flat index (offset) in the 1D input/output array.
 */
size_t calc_offset(const size_t* indices, const size_t* dims, size_t ndim, size_t axis, size_t axis_index, const size_t* strides) {
    size_t offset = 0;
    for (int i = 0; i < ndim; i++) {
        size_t idx;
        if ((size_t)i == axis) {
            idx = axis_index;  // index varies along the 'axis'
        } else {
            // take index from 'indices' array, skipping the 'axis' dimension
            size_t pos = i < (int)axis ? i : i - 1;
            idx = indices[pos];
        }
        offset += idx * strides[i];
    }
    return offset;
}

/*
 * Computes the strides for each dimension of a tensor.
 *
 * Parameters:
 * - dims: Array containing the size of each dimension of the input tensor.
 * - ndim: Number of dimensions (length of dims).
 * - strides: Output array where the computed strides for each dimension will be stored.
 *            Must be preallocated with at least ndim elements.
 *
 * Explanation:
 * The stride for a dimension defines how many elements in the flattened (1D) array
 * you need to skip to move by one step along that dimension in the original tensor.
 * 
 * The stride of the last dimension is always 1, since elements along this dimension
 * are simply stored contiguously in memory by design.
 * 
 * For dimension i (from right to left), the stride is calculated as:
 *   strides[i] = strides[i+1] * dims[i+1]
 * 
 * This means to move one step along dimension i, you skip strides[i] elements in the 1D array.
 */
void compute_strides(const size_t* dims, size_t ndim, size_t* strides) {
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
}

/**
 * Applies the softmax function to a 1D input vector.
 *
 * This function performs a numerically stable softmax computation by subtracting
 * the maximum input value before exponentiation to avoid overflow.
 *
 * Parameters:
 * - input: Pointer to the input array of floats.
 * - output: Pointer to the array where the output will be written.
 * - length: Number of elements in the input and output arrays.
 */
void softmax_vector(const float* input, float* output, size_t length) {
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

/**
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
 * This simplified version assumes you're computing ∂softmax_i/∂z_i = s_i * (1 - s_i)
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
 * The softmax Jacobian is an n x n matrix where each element J[i][j] represents
 * the partial derivative of the i-th softmax output with respect to the j-th input:
 *
 *   J[i][j] = s_i * (delta_ij - s_j)
 *
 * where s_i and s_j are the softmax outputs, and delta_ij is the Kronecker delta
 * (1 if i == j, 0 otherwise).
 *
 * This matrix is symmetric and captures the interaction between all output components.
 * It is mainly used in theoretical analyses and custom backpropagation implementations,
 * but rarely computed explicitly in typical neural network training due to its
 * computational complexity (O(n^2) for input size n).
 *
 * Parameters:
 * - softmax_output: Pointer to the softmax output vector.
 * - jacobian: Pointer to a preallocated 2D array (flattened as 1D) of size length*length
 *             where the computed Jacobian matrix will be stored.
 * - length: The size of the softmax output vector.
 *
 * Note:
 * The Jacobian is stored in row-major order:
 *   jacobian[i * length + j] = d softmax[i] / d input[j]
 */
void softmax_jacobian_derivative(const float* softmax_output, float* jacobian, size_t length) {
    for (size_t i = 0; i < length; i++) {
        for (size_t j = 0; j < length; j++) {
            if (i == j) {
                jacobian[i * length + j] = softmax_output[i] * (1.0f - softmax_output[i]);
            } else {
                jacobian[i * length + j] = -softmax_output[i] * softmax_output[j];
            }
        }
    }
}