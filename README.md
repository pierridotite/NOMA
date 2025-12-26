# NOMA

> The first systems programming language with native, compile-time differentiation.

## What is NOMA?

NOMA is a compiled language designed for machine learning at the hardware level. Unlike Python/PyTorch which interpret code at runtime, NOMA compiles directly to machine code with automatic differentiation built into the compiler.

```rust
// Find x that minimizes x^2
fn main() {
    learn x = 5.0;
    
    optimize(x) until loss < 0.0001 {
        let loss = x * x;
        minimize loss;
    }
    
    return x;  // Returns ~0.0
}
```

## Quick Start

```bash
# Build
cargo build --release

# Compile a NOMA program to executable
cargo run -- build-exe examples/04_linear_solve.noma -o solver

# Run
./solver
# Output: 4.995215
```

## Examples

| File | Description | Output | Notes |
|------|-------------|--------|-------|
| `01_hello.noma` | Basic computation | 25.0 | |
| `02_sigmoid.noma` | Neural activation | 0.999 | |
| `03_gradient_descent.noma` | Minimize x^2 | ~0.01 | |
| `04_linear_solve.noma` | Solve 5w = 25 | ~5.0 | |
| `05_quadratic_min.noma` | Minimize (x-3)^2 | 3.0 | |
| `06_neural_network.noma` | 2-layer perceptron | ~0.89 | Target: 0.9 |
| `07_rosenbrock.noma` | Rosenbrock function | ~0.99 | Target: 1.0 |
| `08_system_equations.noma` | Nonlinear system | ~4.6 | Local minimum |

## Python Comparison

### The Problem

```python
# Python: Manual gradients required
x = 5.0
for _ in range(1000):
    y = x * x
    grad = 2 * x  # YOU compute this
    x = x - 0.01 * grad
```

### The Solution

```rust
// NOMA: Automatic differentiation
fn main() {
    learn x = 5.0;
    optimize(x) until y < 0.0001 {
        let y = x * x;
        minimize y;  // Compiler computes gradients
    }
    return x;
}
```

### Benchmark

Same computation: solve `5 * w = 25` via gradient descent.

```bash
# NOMA: Compile and time execution
cargo run --quiet -- build-exe examples/04_linear_solve.noma -o /tmp/solver
time /tmp/solver
# Output: 4.995215 in ~0.001s

# Python: Same computation
time python3 -c "
w = 0.1
for _ in range(1000):
    pred = 5.0 * w
    error = pred - 25.0
    loss = error * error
    if loss < 0.001: break
    grad = 2 * error * 5.0  # Manual gradient!
    w = w - 0.01 * grad
print(f'{w:.6f}')
"
# Output: 4.995215 in ~0.016s
```

| | NOMA | Python |
|---|------|--------|
| Execution | 0.001s | 0.016s |
| Speedup | **16x faster** | baseline |
| Binary size | 16 KB | ~100 MB runtime |
| Gradients | Automatic | Manual |

### Key Differences

| Aspect | Python + PyTorch | NOMA |
|--------|------------------|------|
| Gradients | Manual or library | Automatic (compiler) |
| Execution | Interpreted | Compiled to native |
| Binary size | ~100MB+ runtime | ~16KB standalone |
| Dependencies | numpy, torch, cuda | None |
| Memory | GC, dynamic | Deterministic |

## Language Features

### Variables

```rust
let x = 5.0;        // Immutable constant
learn w = 0.1;      // Learnable parameter (has gradient)
```

### Functions

```rust
sigmoid(x)          // 1 / (1 + e^-x)
relu(x)             // max(0, x)
```

### Optimization

```rust
optimize(variable) until condition {
    // Define loss
    minimize loss;
}
```

## Architecture

```
NOMA Source -> Lexer -> Parser -> AST -> Graph -> LLVM IR -> Native Binary
                                           |
                                    Autodiff Pass
                                    (Chain Rule)
```

The compiler:
1. Parses NOMA syntax to AST
2. Lowers to computational graph
3. Applies reverse-mode autodiff
4. Generates LLVM IR
5. Compiles to native executable

## Status

**Stage: Pre-Alpha**

- [x] Lexer and parser
- [x] Computational graph
- [x] Reverse-mode autodiff
- [x] LLVM IR generation
- [x] Standalone binary compilation
- [x] Optimization loops (SGD)
- [ ] Tensor operations
- [ ] GPU (PTX) execution
- [ ] Standard library

## License

MIT
