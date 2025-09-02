#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

/* Central N-D tensor value type */
typedef struct Tensor {
    float *data;        // pointer to data
    size_t offset;      // index of elements offset from base (data[0]) (used for views)
    size_t *dims;       // array of length ndim, describes sizes of each axis
    size_t *strides;    // array of length ndim, how far (in elements) you move in data when incrementing that axis.
    size_t ndim;        // number of axes (dimensions)
    int owns_data;      // value 1 => free(data) on destroy
    int owns_shape;     // value 1 => free(dims/strides) on destroy (Option A)
} Tensor;

/* --- Minimal tensor helpers (we can expand later) --- */

/* Create an owning, contiguous tensor (uninitialized).
   dims length = ndim. Returns 0 on success, -1 on error. */
int tensor_create(Tensor *t, const size_t *dims, size_t ndim);

/* Calls 'tensor_create' and populates it with rand values in [-scale, +scale].
   Returns 0 on success, -1 on error. */
int tensor_create_random(Tensor *t, const size_t *dims, size_t ndim, float scale);

/* Create a non-owning DATA view that OWNS ITS SHAPE (Option A).
   Copies dims/strides into small heap blocks; sets owns_shape=1, owns_data=0.
   Returns 0 on success, -1 on error. */
int tensor_view(Tensor *view, const Tensor *base,
                const size_t *dims, const size_t *strides,
                size_t ndim, size_t offset);

/* Free owned buffers (data if owns_data, shape if owns_shape); zeroes the struct. */
void tensor_destroy(Tensor *t);

/* Compute total number of elements. Returns:
   - element count on success
   - 0 if t is NULL or ndim==0
   - SIZE_MAX if overflow would occur (error sentinel) */
size_t tensor_numel(const Tensor *t);

/* Return 1 if memory layout is contiguous row-major for THIS view’s shape,
   regardless of offset. Returns 0 otherwise. */
int tensor_is_contiguous(const Tensor *t);

#endif /* TENSOR_H */

/*
 * AI CONSIDERATIONS:
 * 
 * ⚠️ Minor things to keep in mind:
 * tensor_numel returns SIZE_MAX as an overflow sentinel, but I have to check if callers check it (TODO). That’s fine if you plan to handle it later.
 * Views disallow zero-sized dimensions (if (dims[i] == 0) return 0; in view_fits_base). That’s a deliberate choice, but it means you cannot represent empty slices like NumPy can.
 * tensor_create_random uses rand(), which is fine, but not thread-safe or great for reproducibility. Not wrong, just worth noting.
 * If a user calls tensor_destroy twice on the same tensor, it’s safe (because you zeroed it), which is excellent. 
 */