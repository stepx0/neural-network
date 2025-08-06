#include <stdio.h>
#include "layer.h"
#include "activation.h"

int main(void) {
    ScalarNeuron neuron;

    // Initialize neuron with weight=0.5, bias=0.1, activation=sigmoid
    neuron_init(&neuron, 0.5f, 0.1f, sigmoid);

    float input = 1.f;
    float output = neuron_forward(&neuron, input);

    printf("Input: %f\n", input);
    printf("Output: %f\n", output);

    return 0;
}
