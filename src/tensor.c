#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <limits.h>


// ------------ internal helpers ------------ //

/* Compute row-major strides from dims (length = ndim). */
static void compute_row_major_strides(size_t *strides, const size_t *dims, size_t ndim) {
    if (ndim == 0) return;
    strides[ndim - 1] = 1;
    for (size_t i = ndim - 1; i > 0; i--) {
        strides[i - 1] = strides[i] * dims[i];
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
    t->owns_shape = 0;
}

/* Avoids multiplication overflow */
static int safe_mul_size(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a) return -1;
    *out = a * b; return 0;
}

/* When building a view, verify that the furthest reachable linear index fits base. */
static int view_fits_base(const Tensor *base, const size_t *dims, const size_t *strides, size_t ndim, size_t offset) {
    if (!base || !dims || !strides || ndim == 0) return 0;

    /* base size must be known and not overflowed */
    size_t base_numel = 0;
    if (!base->dims || base->ndim == 0) return 0;
    /* compute base numel with overflow detection */
    {
        size_t n = 1, tmp;
        for (size_t i = 0; i < base->ndim; ++i) {
            if (safe_mul_size(n, base->dims[i], &tmp) != 0) return 0;
            n = tmp;
        }
        base_numel = n;
    }

    /* Compute the addressable range of the base: [base->offset, base->offset + base_numel - 1],
       guarding additions against overflow. */
    if (base_numel == 0) return 0; /* disallow views over empty base for now */

    size_t max_allowed;
    /* base->offset + (base_numel - 1) with overflow checks */
    {
        size_t last = base_numel - 1;
        if (base->offset > SIZE_MAX - last) return 0;
        max_allowed = base->offset + last;
    }
    const size_t min_allowed = base->offset;

    /* Starting linear index of the view within base */
    size_t start_index;
    if (base->offset > SIZE_MAX - offset) return 0;           /* overflow in start computation */
    start_index = base->offset + offset;

    /* max linear index accessed by the view = start_index + sum_i (strides[i] * (dims[i]-1)) */
    size_t max_index = start_index;
    for (size_t i = 0; i < ndim; ++i) {
        if (dims[i] == 0) return 0; /* empty dims not allowed for views here */
        size_t add = 0;
        if (safe_mul_size(strides[i], dims[i] - 1, &add) != 0) return 0;
        if (SIZE_MAX - max_index < add) return 0; /* overflow in addition */
        max_index += add;
    }

    /* Containment: [start_index, max_index] must lie within [min_allowed, max_allowed] */
    if (start_index < min_allowed) return 0;
    if (max_index > max_allowed) return 0;

    return 1;
}

// ------------ public API ------------ //

int tensor_create_random(Tensor *t, const size_t *dims, size_t ndim, float scale) {
    if (tensor_create(t, dims, ndim) != 0) return -1;

    size_t n = tensor_numel(t);          // <-- cache once
    if (n == SIZE_MAX) {                 // <-- overflow sentinel
        tensor_destroy(t);
        return -1;
    }

    for (size_t i = 0; i < n; ++i) {
        float r = (float)rand() / (float)RAND_MAX;  // [0,1]
        t->data[i] = (r * 2.f - 1.f) * scale;       // [0, 2] -> [-1, 1] -> [-scale, +scale]
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
    size_t numel = 1, tmp;
    for (size_t i = 0; i < ndim; ++i) {
        if (safe_mul_size(numel, dims_buf[i], &tmp) != 0) {  // overflow
            free(dims_buf); free(strides_buf); return -1;
        }
        numel = tmp;
    }

    // Safely calculate byte size
    size_t nbytes;
    if (safe_mul_size(numel, sizeof(float), &nbytes) != 0) { // overflow
        free(dims_buf); free(strides_buf); return -1;
    }

    // Allocate data (contiguous row-major)
    float *data_buf = (float*)malloc(nbytes);
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
    t->owns_shape = 1;

    return 0;
}

/* View copies shape (dims/strides) into its own small heap blocks */
int tensor_view(Tensor *view, const Tensor *base,
                const size_t *dims, const size_t *strides,
                size_t ndim, size_t offset) {
    if (!view || !base || !dims || !strides || ndim == 0) return -1;

    /* Validate the view fits in base */
    if (!view_fits_base(base, dims, strides, ndim, offset)) return -1;

    tensor_zero(view);

    view->data = base->data;  /* share data */
    view->offset = base->offset + offset;  /* accumulate base offset */
    view->ndim = ndim;
    view->owns_data  = 0;
    view->owns_shape = 1;

    view->dims = (size_t*)malloc(ndim * sizeof(size_t));
    view->strides = (size_t*)malloc(ndim * sizeof(size_t));
    if (!view->dims || !view->strides) {
        free(view->dims); free(view->strides);
        tensor_zero(view);
        return -1;
    }
    memcpy(view->dims, dims, ndim * sizeof(size_t));
    memcpy(view->strides, strides, ndim * sizeof(size_t));

    return 0;
}

void tensor_destroy(Tensor *t) {
    if (!t) return;

    if (t->owns_data) {
        free(t->data);
    }
    if (t->owns_shape) {
        free(t->dims);
        free(t->strides);
    }
    tensor_zero(t);
}


//TODO: handle the SIZE_MAX numel all over the project...
size_t tensor_numel(const Tensor *t) {
    if (!t || t->ndim == 0 || !t->dims) return 0;

    size_t n = 1;
    size_t tmp;
    for (size_t i = 0; i < t->ndim; i++) {
        if (safe_mul_size(n, t->dims[i], &tmp) != 0) return SIZE_MAX; // overflow -> signaling error with SIZE_MAX
        n = tmp;
    }
    return n;
}

int tensor_is_contiguous(const Tensor *t) {
    if (!t || t->ndim == 0 || !t->dims || !t->strides) return 0;

    // Expected row-major strides
    size_t expect = 1;
    for (size_t i = t->ndim; i-- > 0; ) {
        if (t->strides[i] != expect) return 0;
        expect *= t->dims[i];
    }
    return 1;
}
