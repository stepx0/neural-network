#ifndef ACTIVATION_H
#define ACTIVATION_H

float sigmoid(float x);
float sigmoid_derivative(float output);

float relu(float x);
float relu_derivative(float x);

float tanh(float x);
float tanh_derivative(float x);

float leaky_relu(float x);
float leaky_relu_derivative(float x);

void softmax(const float* input, float* output, const size_t* dims, size_t ndim, size_t axis);
void softmax_derivative(const float* softmax_output, float* output_derivative, size_t length);
void softmax_jacobian_derivative(const float* softmax_output, float* jacobian, size_t length);

#endif
