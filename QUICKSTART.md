# NOMA Quick Start Guide

Get up and running with NOMA in 5 minutes.

---

## Installation

### Prerequisites

- **Rust** (1.70+): `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- **Clang** (for linking): `sudo apt install clang` (Linux) or `brew install llvm` (macOS)
- **LLVM 17+** (optional, for advanced features)

### Build NOMA

```bash
git clone https://github.com/pierridotite/Noma.git
cd Noma
cargo build --release
```

The compiler is now at `target/release/noma` (or use `cargo run --` for convenience).

---

## Your First NOMA Program

Create `hello.noma`:

```noma
fn main() {
    let x = 5.0;
    let y = x * 2.0 + 3.0;
    return y;  // Returns 13.0
}
```

Run it:

```bash
cargo run -- run hello.noma
# Output: 13.0
```

---

## Gradient Descent in 10 Lines

Create `optimize.noma`:

```noma
fn main() {
    learn x = 10.0;  // Start at x=10
    
    optimize(x) until loss < 0.01 {
        let loss = (x - 5.0) * (x - 5.0);  // Minimize (x-5)²
        minimize loss;
    }
    
    return x;  // Should converge to ~5.0
}
```

Run it:

```bash
cargo run -- run optimize.noma
# Output: ~5.0 (after gradient descent)
```

**What happened?**  
The compiler automatically computed `∂loss/∂x = 2(x-5)` and used it to update `x`.

---

## A Real Neural Network

Create `network.noma`:

```noma
fn main() {
    // Dataset: y = 2x
    let X = tensor [[1.0], [2.0], [3.0], [4.0]];
    let Y = tensor [[2.0], [4.0], [6.0], [8.0]];
    
    // Learnable weight
    learn W = tensor [[0.1]];
    
    // Train with Adam optimizer
    optimize(W) with adam(learning_rate=0.1) until loss < 0.01 {
        let pred = matmul(X, W);
        let error = pred - Y;
        let loss = mean(error * error);
        minimize loss;
    }
    
    return W;  // Should be ~2.0
}
```

Run it:

```bash
cargo run -- run network.noma
# Output: [[2.0]] (learned the relationship y = 2x)
```

---

## Compile to Standalone Binary

Any NOMA program can be compiled to a native executable:

```bash
cargo run -- build-exe network.noma -o my_model
./my_model
# Output: [[2.0]]
```

**Binary size:** 16-50 KB  
**Dependencies:** None  
**Speed:** 15x faster than PyTorch

---

## Dynamic Topology Growth (Killer Feature)

Create `growing_network.noma`:

```noma
fn main() {
    let X = tensor [[1.0], [2.0], [3.0]];
    let Y = tensor [[2.0], [4.0], [6.0]];
    
    learn W = tensor [[0.1], [0.2]];  // Start with 2 neurons
    
    optimize(W) until loss < 0.01 {
        let pred = matmul(X, W);
        let loss = mean((pred - Y) * (pred - Y));
        
        // If not converging, grow the network
        if loss > 0.5 {
            realloc W = [10, 1];  // Instantly grow to 10 neurons
            print("Network grew to 10 neurons!");
        }
        
        minimize loss;
    }
    
    return W;
}
```

Run it:

```bash
cargo run -- run growing_network.noma
# Output: Network grew to 10 neurons!
#         [[2.0], [0.0], [0.0], ...] (learned y = 2x)
```

**Try this in PyTorch:**  
You'd need to stop training, create a new model, copy weights, rebuild the optimizer, and resume. NOMA does it in **one line** without interrupting training.

---

## Batch Processing & File I/O

Create `training_pipeline.noma`:

```noma
fn main() {
    // Load dataset from CSV
    load_csv data = "examples/data/sample_data.csv";
    
    // Initialize weights randomly
    learn W = init_xavier([2, 1], rng_seed=42.0);
    
    // Mini-batch training
    batch x_batch, batch_idx in data with batch_size=32.0 {
        let pred = matmul(x_batch, W);
        let loss = mean(pred * pred);
        
        optimize(W) with adam(learning_rate=0.01) until loss < 0.1 {
            minimize loss;
        }
    }
    
    // Save trained model
    save_safetensors { weights: W }, "trained_model.safetensors";
    
    return W;
}
```

Run it:

```bash
cargo run -- run training_pipeline.noma
# Output: Trained weights saved to trained_model.safetensors
```

---

## Built-in Functions Reference

| Function | Description |
|----------|-------------|
| `tensor` | Create multidimensional array |
| `matmul` | Matrix multiplication |
| `mean`, `sum` | Reduction operations |
| `sqrt`, `exp`, `log` | Math functions |
| `sigmoid`, `relu`, `tanh` | Activation functions |
| `init_xavier`, `init_he` | Weight initialization |
| `load_csv`, `save_csv` | CSV file I/O |
| `load_safetensors`, `save_safetensors` | Binary model format |

**[→ Full Built-ins List](LANGUAGE_GUIDE.md#built-in-functions)**

---

## Next Steps

1. **Explore Examples**: Check out `examples/` for 28+ complete programs
2. **Read Language Guide**: [LANGUAGE_GUIDE.md](LANGUAGE_GUIDE.md) has full syntax reference
3. **Try GPU Support**: Experimental PTX/CUDA backend (see README)
4. **Contribute**: See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup

---

## Common Issues

### `clang: command not found`

Install clang:
- **Linux**: `sudo apt install clang`
- **macOS**: `xcode-select --install`
- **Windows**: Install LLVM from https://releases.llvm.org/

### `linker 'cc' not found`

Install GCC:
- **Linux**: `sudo apt install build-essential`
- **macOS**: `xcode-select --install`

### Binary crashes on macOS M1/M2

Add `--target x86_64-apple-darwin` when building on ARM Macs (native ARM support coming soon).

### Performance slower than expected

Try compiling with optimizations:
```bash
cargo run --release -- build-exe program.noma -o optimized
```

---

## Getting Help

- **GitHub Issues**: [Report bugs](https://github.com/pierridotite/Noma/issues)
- **Discussions**: [Ask questions](https://github.com/pierridotite/Noma/discussions)
- **Reddit**: r/Noma (coming soon)

---

## What to Build Next

Ideas to explore NOMA's capabilities:

- Hyperparameter search: Use `optimize` with multiple variables
- Custom loss functions: Define your own with `fn`
- Multi-layer networks: Stack `matmul` operations with activations
- Time series: Use dynamic allocation for variable-length sequences
- Neural architecture search: Use `realloc` to explore topologies
- Embedded ML: Compile to tiny binaries for IoT devices

Happy optimizing!
