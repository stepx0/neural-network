#include "layers.h"
#include <stdlib.h>
#include <string.h>

/* helper to do products */
static size_t product(const size_t *a, size_t n) {
    size_t p = 1;

    for(size_t i=0; i<n; i++) 
        p *= a[i];
    

    return p;
}

/* Activation pipeline */
/* Activation pipeline
 *
 * Applies a sequence of scalar/vector activations over the whole tensor.
 * - Scalar steps: elementwise; can be done in-place when safe and original values
 *   aren't needed anymore.
 * - Vector steps (e.g., softmax): require read-all-then-write; we ping-pong buffers.
 *
 * Notes:
 * - If the pipeline contains vector steps and 'in' == 'out' (same storage + offset),
 *   we allocate a temporary scratch buffer once and reuse it for ping-pong.
 * - For simplicity, this version handles contiguous fast-paths; add a strided
 *   iterator if you want non-contiguous views support in-place.
 */
void act_pipeline_forward(const ActivationPipeline *pipe, const Tensor *in, Tensor *out) {
    // No pipes? copying input in output as is 
    if (pipe == NULL || pipe->count == 0) {
        if (out != in) {
            size_t n = tensor_numel(in);
            const float *src = in->data + in->offset;
            float *dst = out->data + out->offset;
            if (tensor_is_contiguous(in) && tensor_is_contiguous(out)) {
                memcpy(dst, src, n * sizeof(float));
            } else {
                // TODO: strided copy for non-contiguous views 
                for (size_t i = 0; i < n; i++) dst[i] = src[i];
            }
        }
        return;
    }

    // Check if we have any vector activation steps 
    int has_vector = 0;
    for (size_t i = 0; i < pipe->count; i++) {
        if (pipe->steps[i].kind == ACT_VECTOR) { has_vector = 1; break; }
    }

    // Scratch for ping-pong when needed (only if in == out and we have vector ops) 
    float *scratch = NULL;
    Tensor tmp; // scratch tensor view; same shape as 'out' but data = scratch 
    int need_scratch = 0;

    const int same_storage = (in->data == out->data);
    const int same_view = same_storage && (in->offset == out->offset);

    // allocationg scratch buffer only if needed
    if (has_vector && same_view) {
        size_t n = tensor_numel(out);
        scratch = (float*)malloc(n * sizeof(float));
        if (!scratch) {
            // Fallback: if scratch alloc fails, force out-of-place by early copying 'in' to 'out'
            need_scratch = 0;
        } else {
            // Build a view over scratch (contiguous) with same dims/strides as 'out'
            tensor_view(&tmp, out, out->dims, out->strides, out->ndim, 0);
            tmp.data = scratch;
            tmp.offset = 0;
            need_scratch = 1;
        }
    }

    /* Set up ping-pong: we’ll write alternately into 'out' and 'tmp' if needed.
       Start reading from 'cur_in', writing to 'cur_out'. */
    const Tensor *cur_in = in;
    Tensor *cur_out = out;

    for (size_t s = 0; s < pipe->count; ++s) {
        const ActStep *st = &pipe->steps[s];

        if (st->kind == ACT_SCALAR) {
            /* Scalar elementwise step */
            size_t n = tensor_numel(cur_in);

            if (tensor_is_contiguous(cur_in) && tensor_is_contiguous(cur_out)) {
                const float *src = cur_in->data + cur_in->offset;
                float *dst       = cur_out->data + cur_out->offset;

                int same_stor = (cur_in->data == cur_out->data);
                int same_vw   = same_stor && (cur_in->offset == cur_out->offset);
                size_t bytes  = n * sizeof(float);
                const char *sb = (const char*)src;
                char *db       = (char*)dst;
                int overlap = same_stor && (db < sb + bytes) && (sb < db + bytes);

                if (same_vw) {
                    /* true in-place */
                    for (size_t i = 0; i < n; i++)
                        dst[i] = st->fwd.s(dst[i], st->params.v.alpha);
                } else if (!overlap) {
                    /* out-of-place (no overlap) */
                    for (size_t i = 0; i < n; i++)
                        dst[i] = st->fwd.s(src[i], st->params.v.alpha);
                } else {
                    /* overlapping: choose direction like memmove */
                    if (dst > src) {
                        for (size_t i = n; i-- > 0; )
                            dst[i] = st->fwd.s(src[i], st->params.v.alpha);
                    } else {
                        for (size_t i = 0; i < n; i++)
                            dst[i] = st->fwd.s(src[i], st->params.v.alpha);
                    }
                }
            } else {
                /* TODO: non-contiguous path — implement a strided iterator.
                 * For now, fall back to flat loop assuming compatible layout. */
                const float *src = cur_in->data + cur_in->offset;
                float *dst       = cur_out->data + cur_out->offset;
                for (size_t i = 0; i < n; i++)
                    dst[i] = st->fwd.s(src[i], st->params.v.alpha);
            }

            /* After scalar step, next reads from what we wrote */
            cur_in = cur_out;
        } else {
            /* Vector step (e.g., softmax). Must not overwrite input mid-compute.
               Ensure cur_in != cur_out; if equal and we have scratch, write into tmp. */
            Tensor *target = cur_out;

            if ((cur_in->data == cur_out->data) && (cur_in->offset == cur_out->offset)) {
                if (need_scratch) {
                    target = &tmp;  /* write to scratch */
                } else {
                    /* No scratch: if out != in, redirect write to 'out', else this is unsafe.
                       In practice we should ensure caller gave out!=in for vector pipelines. */
                    target = out;
                }
            }

            st->fwd.v(cur_in, target, &st->params);

            /* Ping-pong: next step reads from where we just wrote;
               flip target between 'out' and 'tmp' if scratch exists. */
            cur_in  = target;
            if (need_scratch) {
                cur_out = (target == out) ? &tmp : out;
            } else {
                cur_out = out; /* keep writing to out if we have no scratch */
            }
        }
    }

    /* Final result must end up in 'out' */
    if (cur_in != out) {
        size_t n = tensor_numel(out);
        const float *src = cur_in->data + cur_in->offset;
        float *dst       = out->data + out->offset;
        if (tensor_is_contiguous(cur_in) && tensor_is_contiguous(out)) {
            memcpy(dst, src, n * sizeof(float));
        } else {
            /* TODO: strided copy for non-contiguous views */
            for (size_t i = 0; i < n; i++) dst[i] = src[i];
        }
    }

    if (scratch) free(scratch);
}

void act_pipeline_backward(const ActivationPipeline *pipe, const Tensor *y, Tensor *dy) {
    if (pipe==NULL || pipe->count==0) return; // dy already holds upstream grad

    // Apply steps in reverse. For scalar steps we do elementwise: dy *= f'(y).
    for (size_t r=pipe->count; r>0; ) {
        const ActStep *st = &pipe->steps[--r];
        if (st->kind == ACT_SCALAR) {
            size_t n = tensor_numel(y);
            const float *yp = y->data + y->offset;
            float *dyp = dy->data + dy->offset;
            for (size_t i = 0; i < n; i++) {
                float g = st->bwd.s(yp[i], st->params.v.alpha); // derivative w.r.t. y
                dyp[i] *= g;
            }
        } else {
            // Vector backward provided by the op (e.g., softmax jacobian-vector product)
            st->bwd.v(y, dy, &st->params);
        }
    }
}

// ----------- Network impl -----------
int nn_create(Network *net, float learning_rate, size_t initial_capacity) {
    if (!net) return -1;
    net->count = 0;
    net->capacity = (initial_capacity? initial_capacity : 4);
    net->learning_rate = learning_rate;
    net->layers = (Layer*)calloc(net->capacity, sizeof(Layer));
    return net->layers ? 0 : -1;
}

static int nn_grow(Network *net) {
    size_t newcap = net->capacity ? net->capacity*2 : 4;
    Layer *nl = (Layer*)realloc(net->layers, newcap*sizeof(Layer));
    if (!nl) return -1;
    // zero new region
    memset(nl + net->capacity, 0, (newcap - net->capacity) * sizeof(Layer));
    net->layers = nl;
    net->capacity = newcap;
    return 0;
}

int nn_add_layer(Network *net, const Layer *layer) {
    if (!net || !layer) return -1;
    if (net->count == net->capacity && nn_grow(net)!=0) return -1;
    net->layers[net->count++] = *layer; // shallow copy; params owned by creator
    return 0;
}

void nn_forward(Network *net, const Tensor *x, Tensor *y) {
    const Tensor *cur_x = x;
    Tensor *cur_y = y;
    for (size_t i=0;i<net->count; ++i) {
        Layer *L = &net->layers[i];
        L->ops.forward(L, cur_x, cur_y);
        // Next layer input is current output
        cur_x = &L->out; // layers are expected to write their result into L->out (or assign a view to it)
    }
    // Final output: if the last layer didn't write to *y directly, copy/view L->out into *y
    if (net->count>0 && cur_x != y) {
        *y = net->layers[net->count-1].out; // shallow assign; caller owns destination policy
    }
}

void nn_backward(Network *net, const Tensor *x, const Tensor *y,
                 const Tensor *dy, Tensor *dx) {
    const Tensor *cur_x = x;   (void)cur_x; // not used in this minimal skeleton
    const Tensor *cur_y = y;   (void)cur_y;

    // Upstream grad starts at dy. dx will hold grad wrt input of the first layer.
    Tensor *cur_dx = dx;
    Tensor upstream = *dy; // copy metadata; data points to user's grad

    for (size_t idx = net->count; idx>0; ) {
        Layer *L = &net->layers[--idx];
        // We pass: x (if cached), y (L->out), dy (upstream), and write dx
        L->ops.backward(L, NULL, &L->out, &upstream, cur_dx);
        // Next iteration: upstream becomes dx
        upstream = *cur_dx;
    }
}

void nn_update(Network *net) {
    for (size_t i=0;i<net->count; ++i) {
        Layer *L = &net->layers[i];
        if (L->ops.update) L->ops.update(L, net->learning_rate);
    }
}

void nn_destroy(Network *net) {
    if (!net) return;
    for (size_t i=0;i<net->count; ++i) {
        Layer *L = &net->layers[i];
        if (L->ops.destroy) L->ops.destroy(L);
    }
    free(net->layers);
    net->layers = NULL; net->count = net->capacity = 0; net->learning_rate = 0.f;
}
