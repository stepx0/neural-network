#include "layers.h"
#include <math.h>
#include <time.h>

#define INITIAL_NETWORK_CAPACITY 8
#define INITIAL_PIPELINE_CAPACITY 4

// TODO: Implementation review, consistency checks and tests
// - Committed as a baseline to avoid losing work; should be cleaned up before production use.
// - added to the repo with a lot of compiler errors.
// - n-dimensionality not handled (as stated in file layers.h)
// - This file contains partial/AI-generated code blocks and needs a full implementation audit.
// - Check for API incongruities, memory-safety issues, and edge-case handling (NULL params, sizes = 0, overflow).
// - Add unit tests for all activation functions (scalar + vector), activation pipeline permutations, and layer/network operations.

// Create a new activation pipeline
ActivationPipeline* create_activation_pipeline(size_t output_size) {
    ActivationPipeline *pipeline = (ActivationPipeline*)malloc(sizeof(ActivationPipeline));
    if (!pipeline) return NULL;
    
    pipeline->steps = (ActivationStep*)malloc(INITIAL_PIPELINE_CAPACITY * sizeof(ActivationStep));
    if (!pipeline->steps) {
        free(pipeline);
        return NULL;
    }
    
    pipeline->intermediate_buffer = (float*)malloc(output_size * sizeof(float));
    if (!pipeline->intermediate_buffer) {
        free(pipeline->steps);
        free(pipeline);
        return NULL;
    }
    
    pipeline->num_steps = 0;
    pipeline->capacity = INITIAL_PIPELINE_CAPACITY;
    
    return pipeline;
}

// Add scalar activation to pipeline
void add_scalar_activation(ActivationPipeline *pipeline, ActivationFunc func, ActivationFunc deriv) {
    if (!pipeline) return;
    
    // Resize if needed
    if (pipeline->num_steps >= pipeline->capacity) {
        int new_capacity = pipeline->capacity * 2;
        ActivationStep *new_steps = (ActivationStep*)realloc(pipeline->steps, 
                                                            new_capacity * sizeof(ActivationStep));
        if (!new_steps) return;
        pipeline->steps = new_steps;
        pipeline->capacity = new_capacity;
    }
    
    ActivationStep *step = &pipeline->steps[pipeline->num_steps];
    step->is_vector = 0;
    step->func.scalar_func = func;
    step->derivative.scalar_deriv = deriv;
    
    pipeline->num_steps++;
}

// Add vector activation to pipeline
void add_vector_activation(ActivationPipeline *pipeline, VectorActivationFunc func, VectorActivationFunc deriv) {
    if (!pipeline) return;
    
    // Resize if needed
    if (pipeline->num_steps >= pipeline->capacity) {
        int new_capacity = pipeline->capacity * 2;
        ActivationStep *new_steps = (ActivationStep*)realloc(pipeline->steps, 
                                                            new_capacity * sizeof(ActivationStep));
        if (!new_steps) return;
        pipeline->steps = new_steps;
        pipeline->capacity = new_capacity;
    }
    
    ActivationStep *step = &pipeline->steps[pipeline->num_steps];
    step->is_vector = 1;
    step->func.vector_func = func;
    step->derivative.vector_deriv = deriv;
    
    pipeline->num_steps++;
}

// Destroy activation pipeline
void destroy_activation_pipeline(ActivationPipeline *pipeline) {
    if (!pipeline) return;
    
    free(pipeline->steps);
    free(pipeline->intermediate_buffer);
    free(pipeline);
}

// Apply activation pipeline
void apply_activation_pipeline(ActivationPipeline *pipeline, float *inputs, float *outputs, int size) {
    if (!pipeline || pipeline->num_steps == 0) {
        // No activations - copy inputs to outputs (linear)
        memcpy(outputs, inputs, size * sizeof(float));
        return;
    }
    
    float *current_input = inputs;
    float *current_output = (pipeline->num_steps == 1) ? outputs : pipeline->intermediate_buffer;
    
    // Apply each activation step in sequence
    for (int step_idx = 0; step_idx < pipeline->num_steps; step_idx++) {
        ActivationStep *step = &pipeline->steps[step_idx];
        
        if (step->is_vector) {
            // Vector activation
            step->func.vector_func(current_input, current_output, size, (VectorParams *)step->params);
        } else {
                float alpha = step->params ? *(float *)step->params : 0.f;
            // Scalar activation - apply to each element
            for (int i = 0; i < size; i++) {
                current_output[i] = step->func.scalar_func(current_input[i], alpha);
            }
        }
        
        // Setup for next step
        current_input = current_output;
        
        // Determine output buffer for next step
        if (step_idx == pipeline->num_steps - 2) {
            // Next step is the last one - output to final buffer
            current_output = outputs;
        } else if (step_idx < pipeline->num_steps - 2) {
            // More steps remaining - alternate between intermediate buffer and outputs
            current_output = (current_output == outputs) ? pipeline->intermediate_buffer : outputs;
        }
    }
}

// Create layer with activation pipeline
Layer* create_layer_with_pipeline(size_t input_size, size_t output_size) {
    if (input_size <= 0 || output_size <= 0) {
        fprintf(stderr, "Error: Invalid layer dimensions\n");
        return NULL;
    }
    
    Layer *layer = (Layer*)malloc(sizeof(Layer));
    if (!layer) {
        fprintf(stderr, "Error: Memory allocation failed for layer\n");
        return NULL;
    }
    
    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->activation_pipeline = create_activation_pipeline(output_size);
    
    if (!layer->activation_pipeline) {
        free(layer);
        return NULL;
    }

    
    // Allocate memory same as before
    layer->weights = (float**)malloc(output_size * sizeof(float*));
    if (!layer->weights) {
        destroy_activation_pipeline(layer->activation_pipeline);
        free(layer);
        return NULL;
    }
    
    for (int i = 0; i < output_size; i++) {
        layer->weights[i] = (float*)calloc(input_size, sizeof(float));
        if (!layer->weights[i]) {
            for (int j = 0; j < i; j++) {
                free(layer->weights[j]);
            }
            free(layer->weights);
            destroy_activation_pipeline(layer->activation_pipeline);
            free(layer);
            return NULL;
        }
    }
    
    layer->biases = (float*)calloc(output_size, sizeof(float));
    layer->inputs = (float*)malloc(input_size * sizeof(float));
    layer->outputs = (float*)malloc(output_size * sizeof(float));
    layer->net_inputs = (float*)malloc(output_size * sizeof(float));
    layer->delta = (float*)malloc(output_size * sizeof(float));
    
    if (!layer->biases || !layer->inputs || !layer->outputs || 
        !layer->net_inputs || !layer->delta) {
        destroy_layer(layer);
        return NULL;
    }
    
    init_weights_xavier(layer);
    return layer;
}
// Create a new layer (backward compatibility)
Layer* create_layer(size_t input_size, size_t output_size, ActivationFunc activation, ActivationFunc activation_deriv) {
    Layer *layer = create_layer_with_pipeline(input_size, output_size);
    if (!layer) return NULL;
    
    // Add single scalar activation
    if (activation) {
        add_scalar_activation(layer->activation_pipeline, activation, activation_deriv);
    }
    
    return layer;
}

// Create layer with vector activation (backward compatibility)
Layer* create_layer_vector(size_t input_size, size_t output_size, VectorActivationFunc activation, VectorActivationFunc activation_deriv) {
    Layer *layer = create_layer_with_pipeline(input_size, output_size);
    if (!layer) return NULL;
    
    // Add single vector activation
    if (activation) {
        add_vector_activation(layer->activation_pipeline, activation, activation_deriv);
    }
    
    return layer;
}
void destroy_layer(Layer *layer) {
    if (!layer) return;
    
    // Free weight matrix
    if (layer->weights) {
        for (int i = 0; i < layer->output_size; i++) {
            free(layer->weights[i]);
        }
        free(layer->weights);
    }
    
    // Free vectors
    free(layer->biases);
    free(layer->inputs);
    free(layer->outputs);
    free(layer->net_inputs);
    free(layer->delta);
    
    free(layer);
}

// Perform forward pass through a single layer
void forward_pass_layer(Layer *layer, float *inputs) {
    if (!layer || !inputs) return;
    
    // Cache inputs for backpropagation (REQUIRED)
    memcpy(layer->inputs, inputs, layer->input_size * sizeof(float));
    
    // Compute net inputs (weighted sum + bias) and apply activation
    for (int i = 0; i < layer->output_size; i++) {
        float net_input = layer->biases[i];
        
        // Compute weighted sum
        for (int j = 0; j < layer->input_size; j++) {
            net_input += layer->weights[i][j] * inputs[j];
        }
        
        layer->net_inputs[i] = net_input;
    }
    
    // Apply activation pipeline (handles multiple activations in sequence)
    apply_activation_pipeline(layer->activation_pipeline, layer->net_inputs, layer->outputs, layer->output_size);
}

// Get the outputs of a layer
float* get_layer_outputs(Layer *layer) {
    return layer ? layer->outputs : NULL;
}

// Initialize weights with random values
void init_weights_random(Layer *layer, float min_val, float max_val) {
    if (!layer) return;
    
    static int seeded = 0;
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = 1;
    }
    
    for (int i = 0; i < layer->output_size; i++) {
        for (int j = 0; j < layer->input_size; j++) {
            float random = (float)rand() / RAND_MAX;
            layer->weights[i][j] = min_val + random * (max_val - min_val);
        }
    }
}

// Initialize weights using Xavier initialization
void init_weights_xavier(Layer *layer) {
    if (!layer) return;
    
    float limit = sqrtf(6.0f / (layer->input_size + layer->output_size));
    init_weights_random(layer, -limit, limit);
}

// Initialize weights using He initialization (good for ReLU)
void init_weights_he(Layer *layer) {
    if (!layer) return;
    
    float std_dev = sqrtf(2.0f / layer->input_size);
    init_weights_random(layer, -std_dev, std_dev);
}

// Create a new network with default capacity
Network* create_network(float learning_rate) {
    return create_network_with_capacity(learning_rate, INITIAL_NETWORK_CAPACITY);
}

// Create a new network with specified capacity
Network* create_network_with_capacity(float learning_rate, int initial_capacity) {
    if (initial_capacity <= 0) {
        fprintf(stderr, "Error: Invalid network capacity\n");
        return NULL;
    }
    
    Network *network = (Network*)malloc(sizeof(Network));
    if (!network) return NULL;
    
    // Allocate contiguous array of Layer structs (not pointers!)
    network->layers = (Layer*)malloc(initial_capacity * sizeof(Layer));
    if (!network->layers) {
        free(network);
        return NULL;
    }
    
    network->num_layers = 0;
    network->capacity = initial_capacity;
    network->learning_rate = learning_rate;
    
    return network;
}

// Add layer with activation pipeline to network
int add_pipeline_layer_to_network(Network *network, size_t input_size, size_t output_size, 
                                 ActivationPipeline *pipeline) {
    if (!network || !pipeline) return 0;
    
    // Check if we need to resize
    if (network->num_layers >= network->capacity) {
        int new_capacity = network->capacity * 2;
        Layer *new_layers = (Layer*)realloc(network->layers, new_capacity * sizeof(Layer));
        if (!new_layers) {
            fprintf(stderr, "Error: Failed to resize network\n");
            return 0;
        }
        network->layers = new_layers;
        network->capacity = new_capacity;
    }
    
    // Create layer with custom pipeline
    Layer *new_layer = create_layer_with_pipeline(input_size, output_size);
    if (!new_layer) return 0;
    
    // Replace default pipeline with custom one
    destroy_activation_pipeline(new_layer->activation_pipeline);
    new_layer->activation_pipeline = pipeline;
    
    // Copy layer data into network array
    network->layers[network->num_layers] = *new_layer;
    free(new_layer);  // Free struct but not its contents
    
    network->num_layers++;
    return 1;
} 
int add_vector_layer_to_network(Network *network, size_t input_size, size_t output_size,
                               VectorActivationFunc activation, VectorActivationFunc activation_deriv) {
    if (!network) return 0;
    
    // Check if we need to resize
    if (network->num_layers >= network->capacity) {
        int new_capacity = network->capacity * 2;
        Layer *new_layers = (Layer*)realloc(network->layers, new_capacity * sizeof(Layer));
        if (!new_layers) {
            fprintf(stderr, "Error: Failed to resize network\n");
            return 0;
        }
        network->layers = new_layers;
        network->capacity = new_capacity;
    }
    
    // Create vector layer directly in the array
    Layer *new_layer = create_layer_vector(input_size, output_size, activation, activation_deriv);
    if (!new_layer) return 0;
    
    // Copy layer data into network array
    network->layers[network->num_layers] = *new_layer;
    
    // Free the temporary layer struct 
    free(new_layer);
    
    network->num_layers++;
    return 1;
}
// Add a scalar activation layer to the network (creates layer internally for better memory management)
int add_layer_to_network(Network *network, size_t input_size, size_t output_size, 
                        ActivationFunc activation, ActivationFunc activation_deriv) {
    if (!network) return 0;
    
    // Check if we need to resize
    if (network->num_layers >= network->capacity) {
        int new_capacity = network->capacity * 2;
        Layer *new_layers = (Layer*)realloc(network->layers, new_capacity * sizeof(Layer));
        if (!new_layers) {
            fprintf(stderr, "Error: Failed to resize network\n");
            return 0;
        }
        network->layers = new_layers;
        network->capacity = new_capacity;
    }
    
    // Create layer directly in the array (no separate allocation)
    Layer *new_layer = create_layer(input_size, output_size, activation, activation_deriv);
    if (!new_layer) return 0;
    
    // Copy layer data into network array
    network->layers[network->num_layers] = *new_layer;
    
    // Free the temporary layer struct (but not its data, which was copied)
    free(new_layer);
    
    network->num_layers++;
    return 1;
}

// Destroy network and all its layers
void destroy_network(Network *network) {
    if (!network) return;
    
    // Destroy each layer's allocated memory
    for (int i = 0; i < network->num_layers; i++) {
        Layer *layer = &network->layers[i];
        
        // Free weight matrix
        if (layer->weights) {
            for (int j = 0; j < layer->output_size; j++) {
                free(layer->weights[j]);
            }
            free(layer->weights);
        }
        
        // Free vectors
        free(layer->biases);
        free(layer->inputs);
        free(layer->outputs);
        free(layer->net_inputs);
        free(layer->delta);
    }
    
    // Free the contiguous layers array
    free(network->layers);
    free(network);
}

// Forward pass through entire network
float* forward_pass_network(Network *network, float *inputs) {
    if (!network || !inputs || network->num_layers == 0) return NULL;
    
    float *current_inputs = inputs;
    
    // Process each layer in sequence (array access, no pointer chasing!)
    for (int i = 0; i < network->num_layers; i++) {
        forward_pass_layer(&network->layers[i], current_inputs);
        current_inputs = network->layers[i].outputs;
    }
    
    // Return output of last layer
    return network->layers[network->num_layers - 1].outputs;
}

// Streaming gradient computation and weight update (memory efficient)
void compute_and_apply_gradients_layer(Layer *layer, float learning_rate) {
    if (!layer) return;
    
    // Update biases using streaming gradients (no storage needed)
    for (int i = 0; i < layer->output_size; i++) {
        float bias_gradient = layer->delta[i];
        layer->biases[i] -= learning_rate * bias_gradient;
        
        // Update weights using streaming gradients (no storage needed)
        for (int j = 0; j < layer->input_size; j++) {
            float weight_gradient = layer->delta[i] * layer->inputs[j];
            layer->weights[i][j] -= learning_rate * weight_gradient;
        }
    }
}

// Backward pass through entire network (backpropagation with streaming updates)
void backward_pass_network(Network *network, float *expected_outputs, 
                          float (*loss_derivative)(float predicted, float actual)) {
    if (!network || !expected_outputs || network->num_layers == 0) return;
    
    // Start with output layer delta computation
    Layer *output_layer = &network->layers[network->num_layers - 1];
    for (int i = 0; i < output_layer->output_size; i++) {
        float error = loss_derivative(output_layer->outputs[i], expected_outputs[i]);
        float activation_deriv;
        
        if (output_layer->use_vector_activation) {
            // For softmax with cross-entropy, derivative simplifies to: predicted - actual
            activation_deriv = 1.0f;  // The error already contains the derivative
        } else {
            activation_deriv = output_layer->activation_deriv ? 
                              output_layer->activation_deriv(output_layer->net_inputs[i]) : 1.0f;
        }
        output_layer->delta[i] = error * activation_deriv;
    }
    
    // Apply gradients to output layer immediately
    compute_and_apply_gradients_layer(output_layer, network->learning_rate);
    
    // Backpropagate through hidden layers (working backwards)
    for (int layer_idx = network->num_layers - 2; layer_idx >= 0; layer_idx--) {
        Layer *current_layer = &network->layers[layer_idx];
        Layer *next_layer = &network->layers[layer_idx + 1];
        
        // Compute delta for current layer
        for (int i = 0; i < current_layer->output_size; i++) {
            float error = 0.0f;
            
            // Sum weighted deltas from next layer
            for (int j = 0; j < next_layer->output_size; j++) {
                error += next_layer->delta[j] * next_layer->weights[j][i];
            }
            
            float activation_deriv;
            if (current_layer->use_vector_activation && current_layer->vector_activation_deriv) {
                // For vector activations, we need special handling of derivatives
                // This is a simplified approach - real softmax derivative is more complex
                activation_deriv = current_layer->outputs[i] * (1.0f - current_layer->outputs[i]);
            } else {
                activation_deriv = current_layer->activation_deriv ? 
                                  current_layer->activation_deriv(current_layer->net_inputs[i]) : 1.0f;
            }
            current_layer->delta[i] = error * activation_deriv;
        }
        
        // Apply gradients immediately (streaming updates)
        compute_and_apply_gradients_layer(current_layer, network->learning_rate);
    }
}

// Print layer information
void print_layer_info(Layer *layer, int layer_index) {
    if (!layer) return;
    
    printf("Layer %d Info:\n", layer_index);
    printf("  Input size: %d\n", layer->input_size);
    printf("  Output size: %d\n", layer->output_size);
    printf("  Memory usage: %.2f KB\n", get_layer_memory_usage(layer) / 1024.0f);
    
    printf("  First few weights:\n");
    int max_show = (layer->output_size < 3) ? layer->output_size : 3;
    for (int i = 0; i < max_show; i++) {
        printf("    Neuron %d: ", i);
        int max_weights = (layer->input_size < 5) ? layer->input_size : 5;
        for (int j = 0; j < max_weights; j++) {
            printf("%.3f ", layer->weights[i][j]);
        }
        if (layer->input_size > 5) printf("...");
        printf("(bias: %.3f)\n", layer->biases[i]);
    }
    if (layer->output_size > 3) printf("  ...\n");
}

// Print network information
void print_network_info(Network *network) {
    if (!network) return;
    
    printf("Network Info:\n");
    printf("  Number of layers: %d\n", network->num_layers);
    printf("  Capacity: %d\n", network->capacity);
    printf("  Learning rate: %.6f\n", network->learning_rate);
    printf("  Total memory usage: %.2f KB\n", get_network_memory_usage(network) / 1024.0f);
    
    for (int i = 0; i < network->num_layers; i++) {
        printf("\n  Layer %d: %d → %d neurons\n", 
               i, network->layers[i].input_size, network->layers[i].output_size);
    }
}

// Calculate memory usage of a layer
size_t get_layer_memory_usage(Layer *layer) {
    if (!layer) return 0;
    
    size_t total = sizeof(Layer);
    
    // Weight matrix
    total += layer->output_size * sizeof(float*);
    total += layer->output_size * layer->input_size * sizeof(float);
    
    // Vectors
    total += layer->output_size * sizeof(float); // biases
    total += layer->input_size * sizeof(float);  // inputs
    total += layer->output_size * sizeof(float); // outputs
    total += layer->output_size * sizeof(float); // net_inputs
    total += layer->output_size * sizeof(float); // delta
    
    return total;
}

// Calculate memory usage of entire network
size_t get_network_memory_usage(Network *network) {
    if (!network) return 0;
    
    size_t total = sizeof(Network);
    total += network->capacity * sizeof(Layer); // Layer array
    
    for (int i = 0; i < network->num_layers; i++) {
        total += get_layer_memory_usage(&network->layers[i]);
        total -= sizeof(Layer); // Don't double-count the Layer struct
    }
    
    return total;
}