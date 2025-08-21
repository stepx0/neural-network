#include "loss_functions.h"
#include <stddef.h>
#include <math.h>

/* 
 * Clamp value to range [epsilon, 1 - epsilon].
 * epsilon is the minimum float positive const close to zero
 */
static inline float clamp_prob(float p) {
    const float epsilon = 1e-15f;
    if (p < epsilon) return epsilon;
    if (p > 1.f - epsilon) return 1.f - epsilon;
    return p;
}

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
    return loss / (float)size;
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


/* Fused: cross-entropy with softmax logits.
 * 'logits' -> softmax(logits) internally using the log-sum-exp trick
 * Returns mean loss; 'dlogits' gets dL/dz = (y_hat - y) / size.
 * 'dlogits' may be NULL if you only want the loss.
 */
float softmax_crossentropy_with_logits(const float* logits, const float* target, float* dlogits, size_t size) {
    // 1) log-sum-exp for stability
    float max_z = logits[0];
    for (size_t i = 1; i < size; ++i) if (logits[i] > max_z) max_z = logits[i];

    float sum_exp = 0.f;
    for (size_t i = 0; i < size; ++i) sum_exp += expf(logits[i] - max_z);

    float log_sum_exp = logf(sum_exp) + max_z;

    // 2) loss = -sum y_i * log softmax_i = -sum y_i*(z_i - log_sum_exp) = (log_sum_exp) - sum y_i*z_i
    float yz = 0.f;
    for (size_t i = 0; i < size; ++i) yz += target[i] * logits[i];
    float loss = log_sum_exp - yz;

    // 3) gradient: dL/dz = softmax(z) - y
    if (dlogits) {
        for (size_t i = 0; i < size; ++i) {
            float yhat_i = expf(logits[i] - log_sum_exp);
            dlogits[i] = (yhat_i - target[i]) / (float)size; // mean
        }
    }

    return loss / (float)size; // mean
}
