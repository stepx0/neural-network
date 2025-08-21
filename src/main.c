#include "layers.h"
#include "tensor.h"
#include "activation_functions.h"
#include <stdio.h>
#include <stdlib.h>

// ---- A minimal mock layer that just doubles each input ----
void mock_forward(struct Layer *layer, const Tensor *x, Tensor *y) {
    // Write into layer->out (per nn_forward contract)
    size_t n = tensor_numel(x);
    const float *src = x->data + x->offset;
    float *dst = layer->out.data + layer->out.offset;
    for (size_t i = 0; i < n; i++) {
        dst[i] = src[i] * 2.0f; // simple "layer" logic
    }

    // (Optional) if a non-NULL y is provided and you want to mirror:
    // *y = layer->out;  // shallow assign
}

void mock_backward(struct Layer *layer, const Tensor *x, const Tensor *y, const Tensor *dy, Tensor *dx) {
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
    // Nothing allocated in this mock beyond layer->out, which nn_destroy will free
}

int main(void) {
    Network net;
    if (nn_create(&net, 0.1f, 1) != 0) {
        fprintf(stderr, "nn_create failed\n");
        return 1;
    }

    // Prepare mock layer
    Layer mock = (Layer){0};
    mock.ops.forward = mock_forward;
    mock.ops.backward = mock_backward;
    mock.ops.update = mock_update;
    mock.ops.destroy = mock_destroy;

    // Allocate output tensor for the layer (where mock_forward writes)
    size_t dims[2] = {2, 2};
    if (tensor_create(&mock.out, dims, 2) != 0) {
        fprintf(stderr, "tensor_create(mock.out) failed\n");
        nn_destroy(&net);
        return 1;
    }

    // Add layer to network
    if (nn_add_layer(&net, &mock) != 0) {
        fprintf(stderr, "nn_add_layer failed\n");
        tensor_destroy(&mock.out);
        nn_destroy(&net);
        return 1;
    }

    // Create input tensor and fill with test data
    Tensor input;
    if (tensor_create(&input, dims, 2) != 0) {
        fprintf(stderr, "tensor_create(input) failed\n");
        nn_destroy(&net);
        return 1;
    }
    for (size_t i = 0; i < tensor_numel(&input); i++) {
        input.data[i] = (float)(i + 1); // 1, 2, 3, 4
    }

    // Let nn_forward return a shallow view of the last layer's out in 'output'
    Tensor output = {0};  // do NOT pre-allocate; nn_forward assigns it
    nn_forward(&net, &input, &output);

    // Print output (should be doubled values: 2, 4, 6, 8)
    printf("Output tensor:\n");
    for (size_t i = 0; i < tensor_numel(&output); i++) {
        printf("%.2f ", output.data[i]);
    }
    printf("\n");

    // Cleanup: destroy only what you own
    tensor_destroy(&input);
    nn_destroy(&net);   // frees layer->out; 'output' is just a shallow view, don't destroy it

    return 0;
}
