#ifndef LAYERS_H
#define LAYERS_H

#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>

/* Activation pipeline (scalar + vector) */
typedef enum { ACT_SCALAR = 0, ACT_VECTOR = 1 } ActKind;
typedef enum { PARAM_NONE = 0, PARAM_ALPHA = 1, PARAM_VECTOR = 2 } ParamKind;

/* Activation Function params */
typedef struct {
    ParamKind kind;
    union {
        float alpha; // for scalar activations
        struct { size_t axis; } vec; // vector params kept minimal; tensor carries dims/strides
    } v;
} ActParams;

/* Scalar activations: forward(x, alpha) and backward(y, alpha) 
 * y = f(x) */
typedef float (*ActScalarFwd)(float x, float alpha);
typedef float (*ActScalarBwd)(float x, float y, float alpha);

/* Vector activations that operate over whole tensors (e.g., softmax along axis) */
typedef void (*ActVectorFwd)(const Tensor *in, Tensor *out, const ActParams *p);
typedef void (*ActVectorBwd)(const Tensor *y, Tensor *dy, const ActParams *p);

typedef struct {
    ActKind kind;
    ActParams params;  // tagged union of params, no heap
    union { ActScalarFwd s; ActVectorFwd v; } fwd;
    union { ActScalarBwd s; ActVectorBwd v; } bwd;
} ActStep;

#ifndef ACT_PIPE_MAX
#define ACT_PIPE_MAX 4 // small-vector optimization: most chains are short

#endif

/* Activation pipeline is a vector of ActStep,
 * more sofisticated data structures are not needed in this simplified network */
typedef struct {
    ActStep steps[ACT_PIPE_MAX];
    size_t count;
} ActivationPipeline;

/* Applies the pipeline in-place or out-of-place depending on contiguity; safe if in!=out. */
void act_pipeline_forward(const ActivationPipeline *pipe, const Tensor *in, Tensor *out);
void act_pipeline_backward(const ActivationPipeline *pipe, const Tensor *y, Tensor *dy);

/* Generic layer interface (vtable)
 *
 * The idea here is to have a virtual table and function pointers that define
 * the operations a layer must implement (forward, backward, update, destroy).
 * This allows the Network to store and manage different types of layers
 * (Dense, Conv, etc.) in a single generic array and call their functions,
 * without knowing the specific implementation details at compile time. */
struct Layer;

typedef void (*LayerForward)(struct Layer *layer, const Tensor *x, Tensor *y);
typedef void (*LayerBackward)(struct Layer *layer,const Tensor *x,  const Tensor *y, const Tensor *dy, Tensor *dx);
typedef void (*LayerUpdate)(struct Layer *layer, float lr);
typedef void (*LayerFree)(struct Layer *layer);

typedef struct {
    LayerForward forward;
    LayerBackward backward;
    LayerUpdate update;
    LayerFree destroy;
} LayerOps;

/* 'params' is kept generic (void*) so that different layer types (Dense, Conv, etc.)
 * can store their own parameters without changing the Layer definition. */
typedef struct Layer {
    LayerOps ops;     // function table
    void   *params;   // layer-specific params (weights, caches, etc.)
    Tensor  out;      // optional cached output buffer/view for reuse
} Layer;

/* Network container */
typedef struct {
    Layer *layers;
    size_t count;
    size_t capacity;
    float  learning_rate;
} Network;

/* Network lifecycle */
int  nn_create(Network *net, float learning_rate, size_t initial_capacity);
void nn_destroy(Network *net);
int  nn_add_layer(Network *net, const Layer *layer); // copies the Layer struct (shallow copy of params pointer)

// Execution
void nn_forward(Network *net, const Tensor *x, Tensor *y);
void nn_backward(Network *net, const Tensor *x, const Tensor *y, const Tensor *dy, Tensor *dx);
void nn_update(Network *net);

#endif //LAYERS_H


