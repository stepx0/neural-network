# Neural Network in C (Work in Progress)

This project is a **work in progress** aimed at building a fully functional **neural network implementation in C**, primarily for **educational purposes**.

---

## 📘 Overview

The goal of this project is to:

- Learn and understand the inner workings of neural networks by implementing them from scratch.
- Explore concepts such as forward propagation, backpropagation, activation functions, loss functions, and training loops in a low-level language.
- Gain deeper insight into how deep learning frameworks operate under the hood.

This is not intended to compete with optimized libraries like TensorFlow, PyTorch, or ONNX, but rather to **demystify** what's going on behind those libraries.

---

## ⚠️ Status: Work In Progress

> 🛠️ This project is under active development and **not yet complete**.  
> You may find missing components, experimental code, and areas subject to change or refactoring.

Currently implemented or in progress:

- [x] Common activation functions (ReLU, Sigmoid, Tanh, GELU, Softmax, etc.)
- [x] Softmax with full Jacobian and simplified derivative
- [x] Stride-based tensor indexing for N-dimensional data
- [x] Loss functions (e.g. MSE, Cross-Entropy)
- [ ] Dense layers
- [ ] Backpropagation and gradient computation
- [ ] Model training loop

---

## 🛠️ Why C?

Using **C** forces us to manage memory, computation, and architecture manually, offering a deeper understanding of:

- Tensor representations
- Memory layout and indexing
- Efficiency trade-offs
- Numerical stability (e.g., softmax implementation)

---

## 🚀 Getting Started

To compile and run the project, follow these steps:

### 🛠 Requirements

- GCC or compatible C compiler
- `make` utility
- (Optional) Linux, macOS, or WSL — tested on Unix-like systems

### 📦 Build Instructions

1. **Clone the repository** (if you haven't already):

   ```
   git clone https://github.com/stepx0/neural-network
   cd path-to-project/neural-networl
   ```

2. **Build the project**:

   ```
   make
   ```
  
3. **Run the program**:
   ```
   ./train
   ```

---

## 🧪 Educational Use
This code is designed to be read/modified or even extended.
Ideal for:

Students learning neural networks

Developers exploring lower-level ML concepts

Hobbyists who want to build a neural net without magic black boxes

---

## 📄 License
This project is released under the MIT License.
You are free to use, modify, and share it for any purpose.

---

## 🙏 Contributions
This is a solo educational effort for now.
If you're also learning or interested in contributing (clean code, tests, comments, fixes), feel free to fork the repo or open an issue.

---

## 💬 Disclaimer
This is not production-grade software.
It’s not optimized for production use and focuses on clarity and understanding rather than speed or scalability.
Use it at your own risk — especially if you plan to build rockets or self-driving cars with it. 😉
