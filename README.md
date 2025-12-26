# NOMA

<p align="center">
  <strong>Neural-Oriented Machine Architecture</strong><br>
  <em>The first systems programming language with native, compile-time automatic differentiation.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/stage-pre--alpha-orange" alt="Stage: Pre-Alpha">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License: MIT">
  <img src="https://img.shields.io/badge/language-Rust-red" alt="Built with Rust">
</p>

---

## What is NOMA?

NOMA is a compiled programming language designed for machine learning at the hardware level. Unlike Python/PyTorch which interpret code at runtime, NOMA compiles directly to native machine code with **automatic differentiation built into the compiler**.

```noma
fn main() {
    learn x = 5.0;
    
    optimize(x) until loss < 0.0001 {
        let loss = x * x;
        minimize loss;
    }
    
    return x;  // Returns ~0.0
}
```

### Why NOMA?

| Aspect | Python + PyTorch | NOMA |
|--------|------------------|------|
| Gradients | Library-based (runtime) | **Compiler-native (compile-time)** |
| Execution | Interpreted | **Compiled to native binary** |
| Binary size | ~100MB+ runtime | **~16KB standalone** |
| Dependencies | numpy, torch, cuda... | **None** |
| Memory model | Garbage collected | **Deterministic** |
| Speed | Baseline | **10-20x faster** |

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/pierridotite/Noma.git
cd Noma

# Build the compiler
cargo build --release
```

### Run Your First Program

```bash
# Interpret and run directly (no external toolchain required)
cargo run -- run examples/03_gradient_descent.noma

# Or compile to a standalone executable
cargo run -- build-exe examples/04_linear_solve.noma -o solver
./solver
```

---

## Language Guide

### Variables

```noma
let x = 5.0;        // Immutable constant
learn w = 0.1;      // Learnable parameter (tracked for gradients)
```

### Optimization Loop

The core of NOMA: define what to optimize and let the compiler handle gradients.

```noma
learn x = 5.0;

optimize(x) until loss < 0.0001 {
    let loss = x * x;
    minimize loss;
}
```

### Hyperparameters

Control training with special variable names:

```noma
let learning_rate = 0.01;    // or: let lr = 0.01;
let max_iterations = 10000;  // or: let max_iter = 10000;

learn w = 0.0;
optimize(w) until loss < 0.001 {
    let loss = (w - 5.0) * (w - 5.0);
    minimize loss;
}
```

### Built-in Functions

```noma
sigmoid(x)    // 1 / (1 + e^(-x))
relu(x)       // max(0, x)
sum(tensor)   // Sum all elements → scalar
mean(tensor)  // Average of all elements → scalar
print(x)      // Print value (passes through for chaining)
```

### Tensors

```noma
// Creation
let v = tensor [1.0, 2.0, 3.0];              // 1D: shape [3]
let m = tensor [[1.0, 2.0], [3.0, 4.0]];     // 2D: shape [2, 2]

// Elementwise operations (with broadcasting)
let a = m + 1.0;       // Add scalar to all elements
let b = m * m;         // Elementwise multiply
let c = sigmoid(m);    // Apply function elementwise

// Reductions
let s = sum(m);        // Sum → scalar
let u = mean(m);       // Mean → scalar

// Indexing
let x = m[0][1];       // Access element (row-major)

// Linear algebra
let d = dot(v1, v2);           // Dot product (1D vectors) → scalar
let p = matmul(A, B);          // Matrix multiply (2D) → 2D
let y = matmul(X, W);          // (n×k) @ (k×m) → (n×m)
```

### Negative Numbers in Tensors

```noma
let data = tensor [[-0.5], [1.0], [-2.5]];   // Negative literals supported
```

---

## Examples

### Basic Examples

| Example | Description | Concept |
|---------|-------------|---------|
| `01_hello.noma` | Basic arithmetic | Expressions |
| `02_sigmoid.noma` | Neural activation | Built-in functions |
| `03_gradient_descent.noma` | Minimize x² | Optimization basics |
| `04_linear_solve.noma` | Solve 5w = 25 | Linear equations |
| `05_quadratic_min.noma` | Find minimum of (x-3)² | Quadratic optimization |

### Neural Networks

| Example | Description | Concept |
|---------|-------------|---------|
| `06_neural_network.noma` | 2-layer perceptron | Multi-layer networks |
| `07_rosenbrock.noma` | Rosenbrock function | Non-convex optimization |
| `08_system_equations.noma` | Nonlinear system | Multi-variable optimization |

### Tensor Operations

| Example | Description | Concept |
|---------|-------------|---------|
| `09_hyperparams.noma` | Custom learning rate | Hyperparameter control |
| `10_tensor_ops.noma` | Tensor basics | Creation, indexing, reductions |
| `11_matmul.noma` | Matrix multiplication | Linear algebra |
| `12_linear_regression.noma` | Simple regression | ML pipeline |
| `13_broadcast.noma` | Broadcasting | Bias addition |
| `14_synth_regression.noma` | 20-sample regression | Full training loop |

### Linear Regression Example

```noma
fn main() {
    // 20 samples, 2 features
    let X = tensor [
        [1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [3.0, 2.0],
        [2.0, 3.0], [3.0, 1.0], [4.0, 2.0], [5.0, 3.0],
        [6.0, 2.0], [7.0, 3.0], [8.0, 4.0], [9.0, 5.0],
        [10.0, 6.0], [2.5, 1.5], [3.5, 2.5], [4.5, 3.5],
        [5.5, 3.5], [6.5, 4.5], [7.5, 5.5], [8.5, 6.5]
    ];

    // Targets: Y = 1.5*X1 + 2.0*X2
    let T = tensor [
        [3.5], [5.5], [5.0], [8.5],
        [9.0], [6.5], [10.0], [13.5],
        [13.0], [16.5], [20.0], [23.5],
        [27.0], [6.75], [10.25], [13.75],
        [15.25], [18.75], [22.25], [25.75]
    ];

    learn W = tensor [[0.0], [0.0]];

    let learning_rate = 0.001;
    let max_iterations = 100000;

    optimize(W) until loss < 0.001 {
        let Y = matmul(X, W);
        let E = Y - T;
        let loss = mean(E * E);
        minimize loss;
    }

    print(loss);
    return W;
}
```

---

## Compiler Commands

```bash
# Interpret and run (no compilation needed)
cargo run -- run <file.noma>

# Syntax check only
cargo run -- build <file.noma>

# Compile to LLVM IR
cargo run -- compile <file.noma> -o output.ll

# Build standalone executable
cargo run -- build-exe <file.noma> -o output

# Compile to PTX (GPU, experimental)
cargo run -- compile-ptx <file.noma> -o output.ptx
```

### Build Options

```bash
# Optimization level (0-3)
cargo run -- build-exe <file.noma> -o output -O 3

# Fast math optimizations
cargo run -- build-exe <file.noma> -o output --fast-math

# Debug output
cargo run -- build <file.noma> --ast --tokens --graph
```

---

## Architecture

```
┌─────────────┐    ┌────────┐    ┌─────┐    ┌───────┐    ┌─────────┐
│ NOMA Source │───▶│ Lexer  │───▶│ AST │───▶│ Graph │───▶│ LLVM IR │
└─────────────┘    └────────┘    └─────┘    └───────┘    └─────────┘
                                               │              │
                                         ┌─────┴─────┐        ▼
                                         │  Autodiff │   ┌─────────┐
                                         │  (Chain   │   │ Native  │
                                         │   Rule)   │   │ Binary  │
                                         └───────────┘   └─────────┘
```

**Compilation Pipeline:**

1. **Lexer** — Tokenizes source into keywords, operators, literals
2. **Parser** — Builds Abstract Syntax Tree (AST)
3. **Graph Builder** — Lowers AST to computational graph
4. **Autodiff Pass** — Applies reverse-mode automatic differentiation
5. **LLVM Codegen** — Generates optimized LLVM IR
6. **Native Compilation** — Produces standalone executable via `clang`

---

## Benchmark

Solving `5 * w = 25` via gradient descent:

```bash
# NOMA (compiled)
cargo run -q -- build-exe examples/04_linear_solve.noma -o /tmp/solver
time /tmp/solver
# → 4.995215 in ~0.001s

# Python (interpreted)
time python3 -c "
w = 0.1
for _ in range(1000):
    pred = 5.0 * w
    error = pred - 25.0
    loss = error * error
    if loss < 0.001: break
    grad = 2 * error * 5.0
    w = w - 0.01 * grad
print(f'{w:.6f}')
"
# → 4.995215 in ~0.016s
```

| Metric | NOMA | Python |
|--------|------|--------|
| Execution time | **0.001s** | 0.016s |
| Speedup | **16x** | baseline |
| Binary size | **16 KB** | ~100 MB runtime |
| Gradients | Automatic | Manual |

---

## VS Code Extension

Syntax highlighting is available for `.noma` files. See the [`noma-vscode`](./noma-vscode) extension folder.

---

## Status

**Stage: Pre-Alpha**

### Implemented

- ✅ Lexer and parser
- ✅ Computational graph with autodiff
- ✅ Reverse-mode automatic differentiation
- ✅ LLVM IR code generation
- ✅ Standalone binary compilation
- ✅ Optimization loops (SGD)
- ✅ Tensor literals and operations
- ✅ Broadcasting (numpy-like N-D)
- ✅ Reductions (sum, mean)
- ✅ Indexing with gradient support
- ✅ Linear algebra (dot, matmul)
- ✅ Interpreter mode (`run` command)
- ✅ Variable hyperparameters

### Planned

- 🔲 GPU execution (PTX/CUDA)
- 🔲 Standard library
- 🔲 Control flow (if/else, loops)
- 🔲 User-defined functions
- 🔲 Adam/RMSprop optimizers
- 🔲 Batch processing
- 🔲 Model serialization

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
