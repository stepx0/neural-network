#include <stddef.h>
#include "tensor_utils.h"

/* 
 * Initialize the TensorShape struct:
 * - Copies the dimensions array
 * - Computes the strides array based on the dimensions
 */
void tensor_shape_init(TensorShape *shape, const size_t *dims, size_t ndim) {
    shape->ndim = ndim;
    shape->dims = (size_t*)malloc(ndim * sizeof(size_t));
    shape->strides = (size_t*)malloc(ndim * sizeof(size_t));

    if (!shape->dims || !shape->strides) {
        // Handle memory allocation failure (consider more robust error handling in production)
        free(shape->dims);
        free(shape->strides);
        shape->dims = NULL;
        shape->strides = NULL;
        shape->ndim = 0;
        return;
    }

    for (size_t i = 0; i < ndim; i++) {
    shape->dims[i] = dims[i];
    }

    compute_strides(shape->dims, ndim, shape->strides);
}

/*
 * Free allocated arrays in TensorShape
 */
void tensor_shape_free(TensorShape *shape) {
    if (shape->dims) free(shape->dims);
    if (shape->strides) free(shape->strides);
    shape->dims = NULL;
    shape->strides = NULL;
    shape->ndim = 0;
}

/*
 *Compute total number of elements in the tensor
 */
size_t tensor_shape_volume(const TensorShape *shape) {
    size_t volume = 1;

    for (size_t i = 0; i < shape->ndim; i++) {
        volume *= shape->dims[i];
    }

    return volume;
}

/*
 * Compute offset (flat index) given multi-dimensional indices
 */
 size_t tensor_shape_offset(const TensorShape *shape, const size_t *indices) {
    size_t offset = 0;

    for (size_t i = 0; i < shape->ndim; i++) {
        offset += indices[i] * shape->strides[i];
    }
    
    return offset;
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
 * - shape: to get number of dims and strides.
 * - axis: The axis along which softmax is applied (this axis is excluded from indices).
 * - axis_index: The index along the 'axis' dimension.
 *
 * Returns:
 * - The computed flat index (offset) in the 1D input/output array.
 */
size_t calc_offset(const size_t* indices, const TensorShape *shape, size_t axis, size_t axis_index) {
    size_t offset = 0;
    for (size_t i = 0; i < shape->ndim; i++) {
        size_t idx;
        if (i == axis) {
            idx = axis_index;  // vary along softmax axis
        } else {
            size_t pos = i < axis ? i : i - 1;
            idx = indices[pos];
        }
        offset += idx * shape->strides[i];
    }
    return offset;
}

/*
 * Computes the strides for each dimension of a tensor.
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
void compute_strides(const TensorShape *shape) {
    size_t ndim = shape->ndim;
    size_t *strides = shape->strides;
    const size_t *dims = shape->dims;

    strides[ndim - 1] = 1;
    for (int i = (int)ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
}
/*
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