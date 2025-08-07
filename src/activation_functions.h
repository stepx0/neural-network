#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H


#define GELU_COEFF 0.044715f

/*------------------------------
 * Scalar activation functions 
 *------------------------------*/
/*
 * The constant 'alpha' is passed to all activation functions, but is only actually used by ELU and its derivative.
 * It is passed universally due to the abstraction layer introduced by the activation pipeline system.
 * Functions that do not require 'alpha' simply ignore it.
 */
float sigmoid(float x, float alpha);
float sigmoid_derivative(float output, float alpha);

float relu(float x, float alpha);
float relu_derivative(float x, float alpha);

float tanh_custom(float x, float alpha);
float tanh_derivative(float x, float alpha);

float leaky_relu(float x, float alpha);
float leaky_relu_derivative(float x, float alpha);

float elu(float x, float alpha);
float elu_derivative(float x, float alpha);

float swish(float x, float alpha);
float swish_derivative(float x, float alpha);

/*
 * Gaussian Error Linear Unit (GELU) activation exists in two main forms:
 *
 * 1. gelu(x) — the exact formulation based on the Gaussian cumulative distribution function (CDF):
 *    gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
 *    This version is mathematically precise but computationally expensive due to the use of the 'erf' function.
 *
 * 2. gelu_approx(x) — an approximate but much faster version, used in many deep learning frameworks:
 *        gelu_approx(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
 *    This approximation, proposed by Hendrycks & Gimpel, closely matches the exact GELU and is optimized for speed.
 *
 * Note:
 * - Use 'gelu' for research or precision-critical applications.
 * - Use 'gelu_approx' in performance-oriented inference or training scenarios.
 *
 * Both versions are differentiable, and their respective derivatives are provided below.
 */
float gelu(float x);
float gelu_derivative(float x);

float gelu_approx(float x);
float gelu_approx_derivative(float x);



/*------------------------------
 * Vector activation functions 
 *------------------------------*/

/*
 * This library provides two forms of softmax derivative computations,
 * depending on the intended use:
 *
 * 1. softmax_derivative(...) — simplified gradient form, used when
 *    the softmax output is immediately followed by a cross-entropy loss.
 *    In that common case, the derivative simplifies significantly to:
 *        dL/dz = softmax_output - target
 *    This function provides a per-element gradient vector for efficient backpropagation.
 *
 * 2. softmax_jacobian_derivative(...) — full Jacobian matrix, which computes:
 *   J[i][j] = s_i * (delta_ij - s_j)
 * 
 * where s_i and s_j are elements of the softmax output vector, and delta_ij is the Kronecker delta
 * (1 if i == j, 0 otherwise).
 * It's an NxN matrix where each element J[i][j] represents the partial derivative
 * of the i-th softmax output with respect to the j-th input.
 * 
 *    This form is useful for:
 *      - Understanding internal mechanics during research or debugging
 *      - Theoretical analyses and custom backpropagation implementations
 *      - More complex loss functions or network structures where the simplified form doesn't apply
 *    However, it is more computationally expensive: (O(n²) vs. O(n)).
 *
 * Use 'softmax_derivative' for typical training setups (softmax + cross-entropy).
 * Use 'softmax_jacobian_derivative' only when the full structure of the derivative is explicitly needed.
 */
void softmax(const float* input, float* output, const TensorShape* shape, size_t axis);
void softmax_derivative(const float* softmax_output, float* output_derivative, size_t length);
void softmax_jacobian_derivative(const float* softmax_output, float* jacobian, size_t length);


#endif
