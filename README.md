# NOMA (Neural-Oriented Machine Architecture)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-green.svg)]()
[![Stage](https://img.shields.io/badge/Stage-Pre--Alpha-orange)]()
[![Milestone 4](https://img.shields.io/badge/Milestone%204-In%20Progress-blue)]()

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

**Current Milestone:** Milestone 4 - The Metal (in progress)

### Milestone 1: The Skeleton - COMPLETED

*Objective: Syntax recognition and AST construction.*

* [x] **Lexer:** Tokenization of keywords `learn`, `optimize`, `tensor`.
* [x] **Parser:** Basic recursive descent parser for tokenization.
* [x] **CLI:** Functional `noma build` command with `--tokens` and `--ast` flags.

**Status:** Lexical analysis fully operational. 22 tokens successfully generated from test files.

### Milestone 2: The Graph Engine - COMPLETED

*Objective: Internal Representation (IR).*

* [x] **Lowering:** Converting AST to a computational graph.
* [x] **State Management:** Implementing the `{value, gradient}` memory structure.
* [x] **Forward Pass:** Executing simple arithmetic operations.

**Status:** Parser fully operational. AST construction working. Computational graph IR implemented with forward pass evaluation. All 12 unit tests passing.

### Milestone 3: The Tipping Point - COMPLETED

*Objective: Proof of Concept - CPU Autodiff.*

* [x] **Backward Pass:** Implementing reverse-mode automatic differentiation for scalars.
* [x] **Optimization Loop:** Generating code that updates variables based on gradients.
* [x] **Demo:** A script solving `y = x^2` (finding x=0 via gradient descent).

**Status:** Complete. CPU autodiff working end-to-end with demonstrated convergence.

#### Recent additions (post-M3)

- `optimize … until` now executes gradient steps (SGD) until the condition flips non-zero.
- New operators `%`, `^/**`, `&&`, `||` parsed, evaluated, and emitted in LLVM/PTX.
- Graph forward/backward uses deterministic topological order; function-call gradients for `sigmoid`/`relu`.
- LLVM codegen initializes learnables/vars safely; PTX emits logical ops with predicates.
- Extended regression tests across lexer, parser, graph, LLVM/PTX codegen.

### Milestone 4: The Metal (in progress)

*Objective: Performance and Hardware.*

* [x] LLVM IR generation for scalar graphs (CPU).
* [x] Minimal PTX backend for arithmetic, pow, sigmoid/relu, logical ops.
* [x] Deterministic graph execution order (topological) for forward/backward.
* [x] Native execution of `optimize … until` loops with SGD.
* [x] Language support for `%`, `^/**`, `&&`, `||` with codegen on both backends.
* [x] Rich tensor types and kernels (phased):
    - CPU: statically-shaped tensors (1D/2D), elementwise ops (+, -, *, /, pow, relu/sigmoid), broadcasting minimal.
    - Autodiff: gradients for elementwise ops; shape checks.
    - GPU: PTX kernels for elementwise ops; host launch stub (demo); full host wiring pending.
* [x] Full NVPTX lowering and host stubs (phased):
    - Emit NVPTX module with kernel entrypoints and parameter ABI (PTX backend emits `.entry compute`, params incl. `n_elems` for elementwise).
    - Host stubs for loading modules, allocating device buffers, launching kernels (feature-gated `cuda` runtime; CLI `run-ptx`).
    - Data movement helpers (host↔device) and synchronization (provided in `nvptx_host.rs`).
* [x] Optimized LLVM passes per target (phased):
    - CPU: opt pipeline selection (`-O1`/`-O2`/`-O3`) and fast-math toggles (`--fast-math` flag; emits `fadd fast`, etc.).
    - GPU: NVPTX-specific opts via llc with `-march=nvptx64`; fast-math enables FMA fusion and approx div/sqrt.

### Milestone 5: The Ecosystem (Future)

*Objective: Making NOMA usable for real-world tasks.*

* [ ] **Standard Library (`std`):**
    - File I/O (loading datasets from disk).
    - Random Number Generation (Xavier/He initialization).
    - Networking (basic sockets for distributed training).
* [ ] **Interop:** FFI (Foreign Function Interface) to call C/Rust functions.
* [ ] **Tooling:** Language Server (LSP) for syntax highlighting in VS Code.

## Contributing

NOMA is an experimental project aiming to redefine the software stack for Artificial General Intelligence. We are looking for contributors interested in compiler design and low-level systems programming.

**How to start:**

1. Clone the repository.
2. Review the `src/lexer.rs` file (currently under development).
3. Check the Issues tab for "Good First Issue" tags regarding syntax implementation.

## License

This project is licensed under the MIT License.
