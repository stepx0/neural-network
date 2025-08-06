#include "layer.h"

void neuron_init(ScalarNeuron* neuron, float weight, float bias, scalar_activation act) {
    if (neuron == NULL) return;
    neuron->weight = weight;
    neuron->bias = bias;
    neuron->activation = act;
}

float neuron_forward(const ScalarNeuron* neuron, float input) {
    if (neuron == NULL || neuron->activation == NULL) return 0.f;

    float z = neuron->weight * input + neuron->bias;
    return neuron->activation(z);
}