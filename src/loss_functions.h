#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

#include <stddef.h>

// TODO: Think about logic to implement loss functions along a specific axis

float mse_loss(const float* predicted, const float* target, size_t size);
float mse_derivative(float predicted, float target, size_t size);

float binary_crossentropy_loss(const float* predicted, const float* target, size_t size);
float binary_crossentropy_derivative(float predicted, float target);

float categorical_crossentropy_loss(const float* predicted, const float* target, size_t size);
void categorical_crossentropy_derivative(const float* predicted, const float* target, float* output_grad, size_t size);
float softmax_crossentropy_with_logits(const float* logits, const float* target, float* dlogits, size_t size);

#endif
