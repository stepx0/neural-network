#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

/* Central N-D tensor value type */
typedef struct Tensor {
    float *data;
    size_t offset;      /* element offset from base (for views) */
    size_t *dims;       /* array, sizes of each axis */
    size_t *strides;    /* row-major strides per axis */
    size_t ndim;        /* number of axes */
    int owns_data;      /* 1 => free(data) on destroy */
    int owns_shape;     /* 1 => free(dims/strides) on destroy (Option A) */
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

/* Free owned buffers (data if owns_data, shape if owns_shape); zero the struct. */
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
