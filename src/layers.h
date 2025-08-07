#ifndef LAYERS_H
#define LAYERS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensor_utils.h"

// Forward declarations
typedef struct Layer Layer;
typedef struct Network Network;

// Activation function types
typedef float (*ActivationFunc)(float x, float alpha);                              
typedef void (*VectorActivationFunc)(const float *inputs, float *outputs, size_t length, void *params);

/*
 * Not fully convinced about this handling of vector activation function and derivatives params
 * I'll search for an alternative later... 
 * 
 * It implies logics with the 'params' variable in the 'ActivationStep' struct below
 * and of course all the consecuent implementation logics.
 * 
 */
typedef struct {
    const size_t *dims;
    size_t ndim;
    size_t axis;

    size_t length;
} VectorParams;

/*
 * To properly handle multiple chained scalar/vector activations.
 * Each activation step in the chain has this structure.
 * It can only be scalar or vector, so using union we avoid allocating unneded functions
 */
typedef struct ActivationStep {
    char is_vector;  // 0 = scalar, 1 = vector
    
    /*
     * This field is optional optional, can either be used to store:
     * - a const 'alpha' (for certain scalar activation functions)
     * - a VectorParams instance (for vector activatio functions)
     */ 
    void *params;   

    union {
        ActivationFunc scalar_func;
        VectorActivationFunc vector_func;
    } func;
    union {
        ActivationFunc scalar_deriv;
        VectorActivationFunc vector_deriv;
    } derivative;
} ActivationStep;

/*
 * This is the pipeline to handle multiple chained scalar/vector activations.
 * 'intermediate_buffer' is used to ping-pong intermediate activation results with the outputs vector.
 * Further details of this logic in the implementation of the pipeline.
 */
typedef struct ActivationPipeline {
    ActivationStep *steps;
    size_t num_steps;
    size_t capacity;
    float *intermediate_buffer; // Buffer for intermediate activation results
} ActivationPipeline;

/*
 * Layer structure - represents a single layer in the neural network
*/
typedef struct Layer {

    TensorShape input_shape;  // Input tensor shape: dimensions and strides
    TensorShape output_shape;  // Output tensor shape: dimensions and strides
    
    float *weights;          // Weight n-dimensional vector  //TODO: check if implementation in layers.c is n-dimensional prone
    float *biases;           // Bias vector [output_size]
    
    // Forward pass cache variables (REQUIRED for backprop)
    float *inputs;           // Cache of last inputs [input_size]
    float *outputs;          // Cache of last outputs [output_size]
    float *net_inputs;       // Cache of pre-activation values [output_size]
    
    ActivationPipeline *activation_pipeline;  // Chain of activation functions
    
    // Only keep delta for backprop - gradients computed and applied immediately
    float *delta;            // Error terms [output_size]
} Layer;

/*
 * Network structure - optimized for memory and cache efficiency.
 *
 * Layers are stored directly as a contiguous dynamic array,
 * which improves data locality and reduces pointer indirection compared to a linked list.
 * This layout is more memory-efficient, improves CPU cache usage during forward/backward passes,
 * and simplifies memory management and iteration.
 *
 * The array can be resized dynamically using the `capacity` field.
 */
typedef struct Network {
    Layer *layers;
    size_t num_layers;
    size_t capacity;       // Maximum capacity (for dynamic resizing)
    float learning_rate;   // Learning rate for training
} Network;



/*-------------------------------------------
 * Layer creation and destruction functions
 *-------------------------------------------*/
Layer* create_layer(
    const TensorShape *input_shape,
    const TensorShape *output_shape,
    ActivationFunc activation,
    ActivationFunc activation_derivative);

int add_vector_layer_to_network(
    Network *network,
    const TensorShape *input_shape,
    const TensorShape *output_shape,
    VectorActivationFunc activation,
    VectorActivationFunc activation_deriv);

void destroy_layer(Layer *layer);


/*--------------------------------------------------------
 * Network creation and management (dynamic array based)
 *--------------------------------------------------------*/
Network* create_network(float learning_rate);
Network* create_network_with_capacity(float learning_rate, int initial_capacity);
void destroy_network(Network *network);


/*------------------------------------
 * Layer/Network operation functions
 *------------------------------------*/
void forward_pass_layer(Layer *layer, float *inputs);
float* get_layer_outputs(Layer *layer);
float* forward_pass_network(Network *network, float *inputs);
void backward_pass_network(Network *network, float *expected_outputs, float (*loss_derivative)(float predicted, float actual));


/*----------------------------------
 * Weight initialization functions 
 *----------------------------------*/
void init_weights_random(Layer *layer, float min_val, float max_val);
void init_weights_xavier(Layer *layer);
void init_weights_he(Layer *layer);


/*--------------------
 * Utility functions 
 *--------------------*/
void print_layer_info(Layer *layer, int layer_index);
void print_network_info(Network *network);
size_t get_layer_memory_usage(Layer *layer);
size_t get_network_memory_usage(Network *network);

#endif