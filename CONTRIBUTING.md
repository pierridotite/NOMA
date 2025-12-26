# Contributing to NOMA

Thank you for your interest in contributing to NOMA! This document provides guidelines for contributing to the project.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Areas for Contribution](#areas-for-contribution)

---

## Code of Conduct

Be respectful, constructive, and collaborative. We welcome contributors of all skill levels.

---

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork**: `git clone https://github.com/YOUR_USERNAME/Noma.git`
3. **Create a branch**: `git checkout -b feature/your-feature-name`
4. **Make your changes** (see development setup below)
5. **Test your changes**: `cargo test && cargo run -- run examples/01_hello.noma`
6. **Commit**: `git commit -m "Add feature: your description"`
7. **Push**: `git push origin feature/your-feature-name`
8. **Open a Pull Request** on GitHub

---

## Development Setup

### Prerequisites

- **Rust** 1.70+: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- **Clang** for linking: `sudo apt install clang` (Linux) or `brew install llvm` (macOS)
- **LLVM 17+** (optional for advanced features)

### Build the Project

```bash
cd Noma
cargo build --release
```

### Run Tests

```bash
cargo test
```

### Run Examples

```bash
cargo run -- run examples/03_gradient_descent.noma
```

### Debug Mode

For faster iteration during development:

```bash
cargo run -- run your_test.noma
```

---

## Project Structure

```
Noma/
├── src/
│   ├── main.rs              # CLI entry point
│   ├── lexer.rs             # Tokenization
│   ├── token.rs             # Token definitions
│   ├── parser.rs            # AST construction
│   ├── ast.rs               # AST node definitions
│   ├── graph.rs             # Computational graph & autodiff
│   ├── llvm_codegen.rs      # LLVM IR generation
│   ├── ptx_codegen.rs       # GPU/PTX code generation
│   ├── nvptx_host.rs        # CUDA host interface
│   ├── error.rs             # Error handling
│   └── lib.rs               # Public API
├── examples/                # Example programs (.noma files)
├── docs/                    # Documentation and diagrams
├── noma-vscode/             # VS Code extension
├── LANGUAGE_GUIDE.md        # Complete language reference
├── QUICKSTART.md            # User quickstart guide
└── README.md                # Project overview
```

### Key Modules

- **Lexer** (`src/lexer.rs`): Converts source text into tokens
- **Parser** (`src/parser.rs`): Builds Abstract Syntax Tree (AST) from tokens
- **AST** (`src/ast.rs`): Node definitions for expressions, statements, functions
- **Graph** (`src/graph.rs`): Computational graph for autodiff pass
- **LLVM Codegen** (`src/llvm_codegen.rs`): Generates LLVM IR from graph
- **Main** (`src/main.rs`): CLI commands (run, build-exe, build-llvm, etc.)

---

## How to Contribute

### Reporting Bugs

Open an issue with:
- **Description**: What went wrong?
- **Reproduction**: Minimal NOMA code that triggers the bug
- **Expected behavior**: What should happen?
- **Actual behavior**: What actually happens?
- **Environment**: OS, Rust version, LLVM version

Example:
```
Title: "Crash when using realloc with negative dimensions"

Code:
```noma
fn main() {
    alloc W = [5, -2];  // Negative dimension
}
```

Expected: Compile-time error
Actual: Segfault during execution
Environment: Ubuntu 22.04, Rust 1.75, LLVM 17
```

### Suggesting Features

Open an issue with:
- **Use case**: What problem does this solve?
- **Proposed syntax**: How would it look in NOMA?
- **Alternatives considered**: Other ways to achieve the same goal

Example:
```
Title: "Add softmax activation function"

Use Case: Neural networks for classification
Proposed Syntax: `let probs = softmax(logits);`
Alternatives: Users can implement manually with exp/sum, but built-in would be faster
```

### Writing Code

See [Areas for Contribution](#areas-for-contribution) below for ideas.

---

## Coding Standards

### Rust Style

Follow standard Rust conventions:
- Use `cargo fmt` to format code
- Use `cargo clippy` to catch common mistakes
- Write descriptive variable names (`gradient_accumulator`, not `ga`)
- Add comments for non-obvious logic

### NOMA Example Style

When adding examples to `examples/`:
- Use clear variable names (`weights`, not `w`)
- Add comments explaining what the code does
- Keep examples focused on one concept
- Test that examples run successfully

Example:

```noma
// 29_custom_activation.noma
// Demonstrates user-defined activation functions with autodiff

fn my_relu(x) {
    // ReLU: max(0, x)
    return x * (x > 0.0);
}

fn main() {
    learn x = -2.0;
    
    optimize(x) until loss < 0.01 {
        let activated = my_relu(x);
        let loss = activated * activated;
        minimize loss;
    }
    
    return x;  // Should converge to 0
}
```

---

## Testing

### Running Tests

```bash
cargo test
```

### Adding Tests

Add unit tests in the same file as the code:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_shape() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(tensor.shape, vec![3]);
    }
}
```

### Integration Tests

Add `.noma` files to `examples/` and verify they run:

```bash
cargo run -- run examples/your_new_example.noma
```

---

## Pull Request Process

1. **Update documentation** if you change syntax or add features
2. **Add tests** for new functionality
3. **Run `cargo fmt`** and `cargo clippy`
4. **Verify examples still work**: `cargo run -- run examples/*.noma`
5. **Write a clear PR description**:
   - What does this change?
   - Why is it needed?
   - How was it tested?

### PR Checklist

- [ ] Code compiles without warnings
- [ ] All tests pass
- [ ] New features have examples
- [ ] Documentation updated (README.md, LANGUAGE_GUIDE.md)
- [ ] Commit messages are clear

---

## Areas for Contribution

### High Priority

#### 1. Error Messages
**What**: Improve compiler error messages with helpful suggestions  
**Difficulty**: Medium (2/5)  
**Files**: `src/error.rs`, `src/parser.rs`, `src/graph.rs`

Example:
```
Current: "Unexpected token: RBrace"
Better:  "Unexpected '}'. Did you forget a semicolon on the previous line?"
```

#### 2. Standard Library Functions
**What**: Add more built-in functions (softmax, dropout, batch_norm, etc.)  
**Difficulty**: Medium (2/5)  
**Files**: `src/graph.rs` (add to `eval_builtin_function`)

Example:
```rust
"softmax" => {
    // Implement: softmax(x) = exp(x) / sum(exp(x))
    // ...
}
```

#### 3. Optimizer Improvements
**What**: Add more optimizers (AdaGrad, Lion, etc.)  
**Difficulty**: Hard (3/5)  
**Files**: `src/graph.rs`, `src/ast.rs`

#### 4. GPU Support
**What**: Improve PTX/CUDA backend for production use  
**Difficulty**: Expert (4/5)  
**Files**: `src/ptx_codegen.rs`, `src/nvptx_host.rs`

### Medium Priority

#### 5. Documentation Examples
**What**: Add more example programs to `examples/`  
**Difficulty**: Easy (1/5)  
**Ideas**:
- Convolutional operations
- Recurrent networks
- Reinforcement learning (Q-learning)
- GAN training
- Transfer learning

#### 6. VS Code Extension
**What**: Improve syntax highlighting, add snippets, error highlighting  
**Difficulty**: Medium (2/5)  
**Files**: `noma-vscode/`

#### 7. Benchmarks
**What**: Add comprehensive benchmarks vs PyTorch/TensorFlow  
**Difficulty**: Medium (2/5)  
**What to measure**:
- Training time (various network sizes)
- Inference latency
- Memory usage
- Binary size

#### 8. Type System
**What**: Add static shape checking at compile time  
**Difficulty**: Expert (4/5)  
**Files**: `src/parser.rs`, `src/ast.rs`, new `src/typecheck.rs`

### Low Priority (Future Work)

- Distributed training across multiple machines
- Quantization (INT8, FP16)
- Model compression
- Automatic mixed precision
- JIT compilation mode
- WASM backend for web deployment

---

## Code Review Process

All PRs are reviewed by maintainers. Expect:
- Constructive feedback on code quality
- Suggestions for improvements
- Questions about design decisions

Reviews usually take 2-7 days. Feel free to ping maintainers if it's been longer.

---

## Questions?

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For general questions
- **Email**: [maintainer@noma-lang.org] (if applicable)

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License (same as the project).

---

Thank you for making NOMA better!
