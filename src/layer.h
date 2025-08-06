#ifndef LAYER_H
#define LAYER_H

#include <stddef.h>

typedef float (*scalar_activation)(float);

typedef struct {
    float weight;
    float bias;
    scalar_activation activation;
} ScalarNeuron;

void neuron_init(ScalarNeuron* neuron, float weight, float bias, scalar_activation act);

float neuron_forward(const ScalarNeuron* neuron, float input);

#endif
