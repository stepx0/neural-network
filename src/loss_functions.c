#include <stddef.h>
#include <math.h>
#include "loss_functions.h"
#include "tensor_utils.h"

float mse_loss(const float* predicted, const float* target, size_t size) {
        float sum = 0.f;
    for (size_t i = 0; i < size; i++) {
        float diff = predicted[i] - target[i];
        sum += diff * diff;
    }
    return sum / size;
}

float mse_derivative(float predicted, float target, size_t size) {
    return 2.f * (predicted - target) / (float)size;
}

float binary_crossentropy_loss(const float* predicted, const float* target, size_t size) {
    float loss = 0.f;
    for (size_t i = 0; i < size; i++) {
        
        float p = clamp_prob(predicted[i]);

        loss += - (target[i] * logf(p) + (1.f - target[i]) * logf(1.f - p));
    }
    return loss / (float)size;
}

float binary_crossentropy_derivative(float predicted, float target) {
    predicted = clamp_prob(predicted);

    return (predicted - target) / (predicted * (1.f - predicted));
}

float categorical_crossentropy_loss(const float* predicted, const float* target, size_t size) {
    float loss = 0.f;
    for (size_t i = 0; i < size; i++) {
        float p = clamp_prob(predicted[i]);

        loss += - target[i] * logf(p);
    }
    return loss;
}

/*
 * 'output_grad' must be preallocated with 'size' elements
 */
void categorical_crossentropy_derivative(const float* predicted, const float* target, float* output_grad, size_t size) {
    for (size_t i = 0; i < size; i++) {
        float p = clamp_prob(predicted[i]);

        output_grad[i] = - target[i] / p;
    }
}