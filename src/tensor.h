#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

// Central N-D tensor value type. Shapes live here; TensorShape becomes unnecessary.
typedef struct Tensor {
    float  *data;     // base pointer to storage
    size_t  offset;   // element offset from base (for views)
    size_t *dims;     // length = ndim
    size_t *strides;  // length = ndim (row-major by default)
    size_t  ndim;     // number of dimensions
    int     owns_data;// 1 => free(data) on destroy
} Tensor;

// --- Minimal tensor helpers (we can expand later) ---
// Create an owning, contiguous tensor not intialized. dims length = ndim. Returns 0 on success.
int tensor_create(Tensor *t, const size_t *dims, size_t ndim);
// Calls 'tensor create' and populates it with rand values between [-1, 1]. Returns 0 on success.
int tensor_create_random(Tensor *t, const size_t *dims, size_t ndim, float scale);
// Create a non-owning view into an existing tensor (no allocation, just metadata).
void tensor_view(Tensor *view, const Tensor *base,
                 const size_t *dims, const size_t *strides,
                 size_t ndim, size_t offset);
// Free internal buffers if t->owns_data; zero the struct.
void tensor_destroy(Tensor *t);
// Compute total number of elements.
size_t tensor_numel(const Tensor *t);
// Return 1 if memory is contiguous in row-major layout.
int tensor_is_contiguous(const Tensor *t);

#endif // TENSOR_H