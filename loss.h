#ifndef LOSS_H
#define LOSS_H

#include <stddef.h> // per size_t

// ===== Mean Squared Error (MSE) =====
float mse_loss(const float* predicted, const float* target, size_t size);
float mse_derivative(float predicted, float target);

// ===== Binary Cross-Entropy =====
float binary_crossentropy_loss(const float* predicted, const float* target, size_t size);
float binary_crossentropy_derivative(float predicted, float target);

// ===== Categorical Cross-Entropy (con softmax output) =====
float categorical_crossentropy_loss(const float* predicted, const float* target, size_t size);
void categorical_crossentropy_derivative(const float* predicted, const float* target, float* output_grad, size_t size);

#endif // LOSS_FUNCTIONS_H
