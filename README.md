# NOMA (Neural-Oriented Machine Architecture)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-green.svg)]()
[![Stage](https://img.shields.io/badge/Stage-Pre--Alpha-orange)]()
[![Milestone 1](https://img.shields.io/badge/Milestone%201-Complete-brightgreen)]()

> **The "C" of the Brain.**
> The first systems programming language with native, compile-time differentiation.

## Overview

NOMA is a statically-typed, compiled, bare-metal language designed to build the AGI architectures of tomorrow. It eliminates the traditional separation between code (logic) and weights (intelligence) by making learning a language primitive.

Unlike existing frameworks that act as libraries on top of Python, NOMA is a standalone language that compiles directly to machine code (ASM/PTX), enabling differentiation at the hardware level without the overhead of an interpreter or dynamic graph construction.

## The Problem: Stack Overhead

Modern AI development relies on an inefficient abstraction stack that separates the researcher from the hardware:

```text
Python (Script) -> PyTorch (Dynamic Graph) -> C++ (Dispatcher) -> CUDA (Kernel) -> GPU

```

This abstraction layer results in:

1. **Latency & Overhead:** Significant cycles are wasted in language interoperability and dynamic dispatch.
2. **Static Topologies:** Creating or destroying neurons dynamically during execution is prohibitively expensive.
3. **Deployment Bloat:** Running a model requires a heavy OS and gigabytes of dependencies (Python, Torch, Drivers).

**NOMA changes the paradigm: The language IS the neural network.**

## Core Philosophy

### 1. First-Class Gradients (`learn` keyword)

In C or Rust, a variable is a fixed memory location. In NOMA, a variable declared with `learn` possesses a dual state: a scalar value and a gradient. The compiler automatically tracks dependency graphs and manages gradient accumulation.

### 2. Static Graph Compilation

NOMA does not use a runtime engine to calculate gradients. It analyzes the Control Flow Graph (CFG) during compilation and injects the necessary backward-pass instructions directly into the binary. This results in "Zero-Cost Abstractions" for neural training.

### 3. Tensor-First Memory Management

NOMA bypasses generic heap allocators (`malloc`). It utilizes a structured memory model designed for Tensor Cores and TPUs, allowing for deterministic memory usage and efficient dynamic allocation (`alloc<tensor>`) for self-modifying network architectures.

## Syntax Preview

NOMA combines the rigorous memory safety of Rust with mathematical primitives.

### Neural Kernel Definition

```rust
// Extension: .noma

// A GPU-compatible struct representing a perceptron layer
struct Perceptron {
    // 'learn' indicates parameters subject to optimization
    learn weights: tensor<2x1>;
    learn bias: tensor<1>;

    // 'diff fn' instructs the compiler to generate a backward pass variant
    diff fn forward(input: tensor<1x2>) -> tensor<1> {
        let x = dot(input, self.weights) + self.bias;
        return sigmoid(x);
    }
}

```

### Native Optimization Loop

```rust
fn main() {
    let inputs: tensor<4x2> = [[0,0], [0,1], [1,0], [1,1]];
    let targets: tensor<4x1> = [[0], [1], [1], [0]];
    
    // Direct allocation on VRAM (or RAM for CPU target)
    let mut model = Perceptron::new();

    // 'optimize' primitive: 
    // The compiler unrolls this loop and injects gradient updates (SGD)
    optimize(model) until loss < 0.01 {
        let pred = model.forward(inputs);
        let loss = mse(pred, targets);
        
        // Triggers the backpropagation routine
        minimize loss; 
    }

    print("Training complete. Final weights: ", model.weights);
}

```

## Technical Architecture

NOMA operates as a modern compiler pipeline based on LLVM and MLIR infrastructure:

1. **Frontend (Rust):** Lexical analysis and parsing of `.noma` source files into an Abstract Syntax Tree (AST).
2. **Semantic Analysis:** Type checking and identifying differentiable variables (`learn`).
3. **Graph Lowering:** Transformation of AST into a Directed Acyclic Graph (DAG) for data flow analysis.
4. **Autodiff Pass:** Automatic generation of gradient nodes using Chain Rule application on the DAG.
5. **Backend:** Code generation via LLVM (for CPU) and eventually NVPTX (for NVIDIA GPUs).

## Development Status & Roadmap

We are currently in the **Bootstrap Phase**. The immediate goal is not full GPU support, but achieving the "Tipping Point": a minimal compiler capable of differentiating a scalar function on the CPU.

### Milestone 1: The Skeleton ✅ COMPLETED

*Objective: Syntax recognition and AST construction.*

* [x] **Lexer:** Tokenization of keywords `learn`, `optimize`, `tensor`.
* [x] **Parser:** Basic recursive descent parser for tokenization.
* [x] **CLI:** Functional `noma build` command with `--tokens` and `--ast` flags.

**Status:** Lexical analysis fully operational. 22 tokens successfully generated from test files.

### Milestone 2: The Graph Engine

*Objective: Internal Representation (IR).*

* [ ] **Lowering:** Converting AST to a computational graph.
* [ ] **State Management:** Implementing the `{value, gradient}` memory structure.
* [ ] **Forward Pass:** Executing simple arithmetic operations.

### Milestone 3: The Tipping Point (MVP)

*Objective: Proof of Concept - CPU Autodiff.*

* [ ] **Backward Pass:** Implementing reverse-mode automatic differentiation for scalars.
* [ ] **Optimization Loop:** Generating code that updates variables based on gradients.
* [ ] **Demo:** A script solving `y = x^2` (finding x=0 via gradient descent).

### Milestone 4: The Metal

*Objective: Performance and Hardware.*

* [ ] LLVM IR generation.
* [ ] PTX/CUDA backend integration.

## Contributing

NOMA is an experimental project aiming to redefine the software stack for Artificial General Intelligence. We are looking for contributors interested in compiler design and low-level systems programming.

**How to start:**

1. Clone the repository.
2. Review the `src/lexer.rs` file (currently under development).
3. Check the Issues tab for "Good First Issue" tags regarding syntax implementation.

## License

This project is licensed under the MIT License.
