#include "tensor.h"
#include <stdlib.h>
#include <string.h>

/* Compute row-major strides from dims (length = ndim). */
static void compute_row_major_strides(size_t *strides, const size_t *dims, size_t ndim) {
    if (ndim == 0) return;
    strides[ndim - 1] = 1;
    for (size_t i = (ndim - 2); i > 0; i--) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
}

/* Zero out a Tensor struct. */
static void tensor_zero(Tensor *t) {
    if (!t) return;
    t->data = NULL;
    t->offset = 0;
    t->dims = NULL;
    t->strides = NULL;
    t->ndim = 0;
    t->owns_data = 0;
}

int tensor_create_random(Tensor *t, const size_t *dims, size_t ndim, float scale) {
    if (tensor_create(t, dims, ndim) != 0) return -1;
    for (size_t i = 0; i < tensor_numel(t); i++) {
        t->data[i] = ((float)rand() / RAND_MAX) * 2.f - 1.f; // [-1, 1]
        t->data[i] *= scale; // optional scaling
    }
    return 0;
}

int tensor_create(Tensor *t, const size_t *dims, size_t ndim) {
    if (!t || !dims || ndim == 0) return -1;

    tensor_zero(t);

    // Allocate shape arrays
    size_t *dims_buf = (size_t*)malloc(ndim * sizeof(size_t));
    size_t *strides_buf = (size_t*)malloc(ndim * sizeof(size_t));
    if (!dims_buf || !strides_buf) {
        free(dims_buf);
        free(strides_buf);
        return -1;
    }

    // Copy dims and compute strides
    memcpy(dims_buf, dims, ndim * sizeof(size_t));
    compute_row_major_strides(strides_buf, dims_buf, ndim);

    // Compute number of elements
    size_t numel = 1;
    for (size_t i = 0; i < ndim; ++i) {
        numel *= dims_buf[i];
    }

    // Allocate data (contiguous row-major)
    float *data_buf = (float*)malloc(numel * sizeof(float));
    if (!data_buf) {
        free(dims_buf);
        free(strides_buf);
        return -1;
    }

    // Initialize tensor
    t->data = data_buf;
    t->offset = 0;
    t->dims = dims_buf;
    t->strides = strides_buf;
    t->ndim = ndim;
    t->owns_data = 1;

    return 0;
}

void tensor_view(Tensor *view, const Tensor *base,
                 const size_t *dims, const size_t *strides,
                 size_t ndim, size_t offset) {
    if (!view || !base || !dims || !strides || ndim == 0) return;

    // Create a non-owning view into an existing tensor (no allocation, just metadata).
    view->data = base->data;
    view->offset = offset;
    /* We intentionally do not copy dims/strides for views—follow caller pointers.
       The caller usually passes base->dims/base->strides. */
    view->dims = (size_t*)dims;         /* discard const to fit API; do not modify */
    view->strides = (size_t*)strides;   /* discard const to fit API; do not modify */
    view->ndim = ndim;
    view->owns_data = 0;
}

void tensor_destroy(Tensor *t) {
    if (!t) return;

    /* Free internal buffers if t->owns_data; zero the struct. */
    if (t->owns_data) {
        free(t->data);
        free(t->dims);
        free(t->strides);
    }

    tensor_zero(t);
}

size_t tensor_numel(const Tensor *t) {
    if (!t || t->ndim == 0 || !t->dims) return 0;
    size_t n = 1;
    for (size_t i = 0; i < t->ndim; ++i) n *= t->dims[i];
    return n;
}

int tensor_is_contiguous(const Tensor *t) {
    if (!t || t->ndim == 0 || !t->dims || !t->strides) return 0;

    /* A contiguous view must have row-major strides and zero offset. */
    if (t->offset != 0) return 0;

    // Expected row-major strides
    size_t expect = 1;
    for (size_t i = t->ndim; i-- > 0; ) {
        if (t->strides[i] != expect) return 0;
        expect *= t->dims[i];
    }
    return 1;
}
