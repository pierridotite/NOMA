# NOMA

**Neural-Oriented Machine Architecture**  
*The first systems programming language with native, compile-time automatic differentiation*

[![Stage](https://img.shields.io/badge/stage-alpha-green)](https://github.com/pierridotite/Noma)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Built with Rust](https://img.shields.io/badge/language-Rust-red)](https://www.rust-lang.org/)
[![Docs](https://img.shields.io/badge/docs-language_guide-orange)](LANGUAGE_GUIDE.md)

**[Quick Start](QUICKSTART.md) | [Language Guide](LANGUAGE_GUIDE.md) | [Contributing](CONTRIBUTING.md)**

```
┌─────────────────────────────────────────────────────────────┐
│  NOMA Code          →  LLVM IR  →  Native Binary (16KB)     │
│  + Autodiff Compiler →  No Runtime  →  15x Faster           │
└─────────────────────────────────────────────────────────────┘
```

---

## What Makes NOMA Different?

![Architecture Comparison](docs/architecture_comparison.svg)

> **TL;DR:** PyTorch computes gradients at runtime (slow). NOMA computes them at compile-time (fast).

**Gradients are computed by the compiler, not a library.**

Most ML frameworks (PyTorch, TensorFlow) implement autodiff as a *runtime library*. NOMA implements it as a *compiler pass* — just like type checking or optimization. Your gradients are native code, not interpreter overhead.

```noma
fn main() {
    learn x = 5.0;
    
    optimize(x) until loss < 0.0001 {
        let loss = x * x;
        minimize loss;
    }
    
    return x;  // Compiler automatically computed gradients
}
```

**Result:** Standalone 16KB binary with zero dependencies. No Python. No PyTorch. No CUDA toolkit required for CPU.

---

## The "Wow" Feature: Dynamic Topology Growth

![Dynamic Topology Growth](docs/dynamic_topology.svg)

Networks can grow *during training* without restarting:

```noma
fn main() {
    learn W = tensor [[0.1], [0.2]];  // Start with 2 neurons
    
    optimize(W) until loss < 0.01 {
        let pred = matmul(X, W);
        let loss = mean((pred - Y) * (pred - Y));
        
        // Network struggling? Grow it instantly
        if loss > 0.5 {
            realloc W = [10, 1];  // Now 10 neurons, training continues
        }
        
        minimize loss;
    }
    
    return W;  // Final shape determined at runtime
}
```

Try doing that in PyTorch without stopping training and rebuilding the optimizer.

---

## Quick Start

```bash
# Clone and build
git clone https://github.com/pierridotite/Noma.git
cd Noma
cargo build --release

# Run immediately (no compilation needed)
cargo run -- run examples/03_gradient_descent.noma

# Or compile to standalone binary
cargo run -- build-exe examples/12_linear_regression.noma -o model
./model
```

---

## Benchmark: NOMA vs Python

Solving `5w = 25` via gradient descent:

| Metric | NOMA | Python + Manual Gradients |
|--------|------|---------------------------|
| **Execution Time** | 0.001s | 0.016s |
| **Speedup** | **16x** | baseline |
| **Binary Size** | 16 KB | ~100 MB runtime |
| **Dependencies** | 0 | numpy, interpreter |
| **Gradients** | Automatic (compiler) | Manual (error-prone) |

---

## How It Works

![Compilation Flow](docs/compilation_flow.svg)

NOMA performs automatic differentiation during compilation, not at runtime. When you write `minimize loss;`, the compiler:

1. **Builds a computational graph** of your forward pass
2. **Applies the chain rule** to generate backward pass code
3. **Lowers to LLVM IR** with gradients as native instructions
4. **Produces a standalone binary** with zero dependencies

The result: gradients are **native machine code**, not library calls.

---

## Key Features

### Native Autodiff
- Reverse-mode automatic differentiation as a compiler pass
- Gradients computed at compile-time, executed as native code
- Chain rule applied during LLVM IR generation

### Zero Dependencies
- No Python runtime
- No PyTorch/TensorFlow
- No CUDA toolkit for CPU execution
- Standalone binaries: 16-50 KB typical size

### Multiple Optimizers
- **SGD**: Classic gradient descent
- **Adam**: Adaptive moments (beta1, beta2, epsilon)
- **RMSprop**: Root mean square propagation

### Production Features
- Dynamic memory allocation (`alloc`, `realloc`, `free`)
- Batch processing for mini-batch SGD
- File I/O (CSV, Safetensors format)
- User-defined functions with full autodiff support
- Random initialization (Xavier, He methods)

### Systems-Level Performance
- Compiles to LLVM IR → native code
- Deterministic memory model (no GC)
- Optional fast-math optimizations
- Experimental GPU support (PTX/CUDA)

---

## Language Highlights

```noma
// Variables
let x = 5.0;        // Immutable constant
learn w = 0.1;      // Learnable (gradients computed automatically)

// User functions work with autodiff
fn mse(pred, target) {
    let error = pred - target;
    return mean(error * error);
}

// Tensors and linear algebra
let X = tensor [[1.0, 2.0], [3.0, 4.0]];
let W = tensor [[0.5], [0.3]];
let Y = matmul(X, W);

// Batch processing
batch x_batch in dataset with 32.0 {
    let pred = matmul(x_batch, W);
}

// File I/O
load_csv data = "train.csv";
save_safetensors { model: W }, "trained.safetensors";

// Dynamic allocation
alloc buffer = [rows, cols];
realloc buffer = [new_rows, cols];  // Resize during training
free buffer;
```

**[→ Full Language Guide](LANGUAGE_GUIDE.md)**

---

## Examples

| Example | Description |
|---------|-------------|
| [01_hello.noma](examples/01_hello.noma) | Basic arithmetic |
| [03_gradient_descent.noma](examples/03_gradient_descent.noma) | Minimize x² |
| [06_neural_network.noma](examples/06_neural_network.noma) | 2-layer perceptron |
| [12_linear_regression.noma](examples/12_linear_regression.noma) | Full ML pipeline |
| [20_growing_network.noma](examples/20_growing_network.noma) | Dynamic topology growth |
| [22_adam_optimizer.noma](examples/22_adam_optimizer.noma) | Adam optimizer |
| [28_batch_training.noma](examples/28_batch_training.noma) | Mini-batch SGD with file I/O |

**[→ All Examples](examples/)**

---

## Compiler Commands

```bash
# Interpret and run (no compilation)
cargo run -- run <file.noma>

# Build standalone executable
cargo run -- build-exe <file.noma> -o output

# Compile to LLVM IR
cargo run -- compile <file.noma> -o output.ll

# With optimizations
cargo run -- build-exe <file.noma> -o output -O 3 --fast-math
```

---

## Architecture

```
Source Code               Compilation Pipeline              Output
────────────              ──────────────────────            ──────

  .noma file                                              Native Binary
      │                                                      (16 KB)
      ├──> Lexer ──> Parser ──> AST                            │
      │                            │                           │
      │                            ▼                           │
      │                    Computational Graph                 │
      │                            │                           │
      │                            ├──> Autodiff Pass          │
      │                            │    (Chain Rule)           │
      │                            │                           │
      │                            ▼                           │
      └──────────────────> LLVM IR Generation ───────> Optimization
                                   │                           │
                                   └─────> clang ──────────────┘
                                          Linker
```

**Key Insight:** Autodiff happens *during compilation*, not at runtime. Your gradients are baked into the binary as native instructions.

---

## Current Status: Alpha

### Implemented
- Lexer, parser, AST
- Computational graph with autodiff
- LLVM IR codegen + native compilation
- Optimization loops (SGD, Adam, RMSprop)
- Tensor operations with broadcasting
- User-defined functions
- Dynamic memory (`alloc`/`realloc`/`free`)
- Batch processing
- File I/O (CSV, Safetensors)
- Random initialization (Xavier, He)
- Interpreter mode for rapid testing

### Known Limitations
- **Single data type**: Only `f64` (no int, bool, string)
- **No module system**: Single-file programs only
- **Control flow**: Compile-time evaluation (while loops unroll graph)
- **No recursion**: Functions are inlined
- **No debugging**: No breakpoints or source maps yet

### Roadmap
- Multi-file projects & imports
- Additional data types (int, bool, string)
- Runtime control flow (dynamic branching)
- Debugging support
- Extended GPU support

---

## Why "NOMA"?

**N**eural-**O**riented **M**achine **A**rchitecture

The name reflects the philosophy: neural network training should be a *first-class language feature*, not a library bolted onto a general-purpose language. NOMA treats gradients like any other compiler concept — types, memory, optimization passes.

---

## Contributing

Contributions welcome! Please open an issue or PR. Areas particularly valuable:
- Additional optimizers (L-BFGS, AdaGrad)
- More built-in functions (convolutions, pooling)
- Improved error messages
- BLAS/LAPACK integration
- Extended GPU support

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Learn More

- **[Language Guide](LANGUAGE_GUIDE.md)** — Complete language reference
- **[Examples](examples/)** — 28+ code samples
- **[VS Code Extension](noma-vscode/)** — Syntax highlighting

---

<p align="center">
  <em>Built for ML engineers who want native performance without the runtime overhead.</em>
</p>
