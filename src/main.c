#include <stdio.h>
#include <stdlib.h>
#include "layers.h"
#include "tensor.h"
#include "activation_functions.h"

// ---- A minimal mock layer that just doubles each input ----
void mock_forward(struct Layer *layer, const Tensor *x, Tensor *y) {
    size_t n = tensor_numel(x);
    const float *src = x->data + x->offset;
    float *dst = y->data + y->offset;
    for (size_t i = 0; i < n; i++) {
        dst[i] = src[i] * 2.0f; // simple "layer" logic
    }
}

void mock_backward(struct Layer *layer, const Tensor *x, const Tensor *y,
                   const Tensor *dy, Tensor *dx) {
    size_t n = tensor_numel(y);
    const float *src = dy->data + dy->offset;
    float *dst = dx->data + dx->offset;
    for (size_t i = 0; i < n; i++) {
        dst[i] = src[i] * 2.0f; // just mirror the forward scaling
    }
}

void mock_update(struct Layer *layer, float lr) {
    // No parameters to update in this mock
}

void mock_destroy(struct Layer *layer) {
    // Nothing allocated in this mock
}

int main(void) {
    Network net;
    nn_create(&net, 0.1f, 1);

    // Prepare mock layer
    Layer mock = {0};
    mock.ops.forward = mock_forward;
    mock.ops.backward = mock_backward;
    mock.ops.update = mock_update;
    mock.ops.destroy = mock_destroy;

    // Allocate output tensor for the layer
    size_t dims[2] = {2, 2};
    tensor_create(&mock.out, dims, 2);

    // Add layer to network
    nn_add_layer(&net, &mock);

    // Create input & output tensors
    Tensor input, output;
    tensor_create(&input, dims, 2);
    tensor_create(&output, dims, 2);

    // Fill input with test data
    for (size_t i = 0; i < tensor_numel(&input); i++) {
        input.data[i] = (float)(i + 1); // 1, 2, 3, 4
    }

    // Run forward pass
    nn_forward(&net, &input, &output);

    // Print output
    printf("Output tensor:\n");
    for (size_t i = 0; i < tensor_numel(&output); i++) {
        printf("%.2f ", output.data[i]);
    }
    printf("\n");

    // Cleanup
    tensor_destroy(&input);
    tensor_destroy(&output);
    nn_destroy(&net);

    return 0;
}