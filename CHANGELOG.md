# Changelog

All notable changes to the **NOMA** project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Changed
- Revised README with Table of Contents, News section, and improved clarity
- Updated citation format in documentation
- Streamlined CONTRIBUTING.md

### Removed
- Removed QUICKSTART.md and related links
- Removed architecture comparison documentation and SVG diagrams

---

## [0.1.0] - 2025-12-27

### Added

#### Core Language Features
- **Control flow statements** — `if-else` conditionals and `while` loops
- **User-defined functions** with inlining support in the computational graph
- **Dynamic memory management** — `alloc`, `free`, and `realloc` for heap tensors
- **Core math functions** — `exp`, `log`, `sqrt`, `pow`, `sin`, `cos`, `tanh`, `abs`, `min`, `max`

#### Optimizers & Training
- **Adam optimizer** with momentum and adaptive learning rates
- **RMSprop optimizer** for gradient-based optimization
- **RNG functions** for weight initialization (`rand_uniform`, `rand_normal`)

#### I/O & Serialization
- **CSV file support** — read and write datasets
- **Safetensors support** — save and load model weights

#### Code Generation
- **LLVM backend** for native CPU execution
- **PTX backend** for NVIDIA GPU execution
- **NVPTX host support** with elementwise kernel execution
- **Fast-math optimizations** for both LLVM and PTX backends

#### Tooling & IDE
- **VS Code extension** with syntax highlighting, snippets, and commands

#### Examples & Demos
- Self-growing XOR demo with Python benchmarking and visualization utilities
- 28 example programs covering tensors, neural networks, optimizers, and I/O

### Changed
- Refactored code structure for improved readability and maintainability
- Enhanced dynamic network growth example for clarity

### Fixed
- Formatting issues in README and loss curve visualizations

---

## [0.0.1] - 2025-12-25

### Added

#### Compiler Foundation
- **Lexer** — tokenization of NOMA source files
- **Parser** — AST construction from token stream
- **Graph Engine** — computational graph representation
- **Autodiff** — automatic differentiation with backward pass
- **Gradient-based optimization** — core training loop support

### Documentation
- Initial README with project overview
- Language guide and contributing guidelines

---

## Development Milestones

| Date       | Milestone                                        |
|------------|--------------------------------------------------|
| 2025-12-25 | Milestone 1: Lexer complete                      |
| 2025-12-25 | Milestone 2: Graph Engine complete               |
| 2025-12-25 | Milestone 3: Autodiff & Optimization complete    |
| 2025-12-26 | LLVM & PTX code generation backends              |
| 2025-12-26 | Control flow, functions, dynamic memory          |
| 2025-12-27 | Adam/RMSprop optimizers, CSV/Safetensors I/O     |

---

[Unreleased]: https://github.com/pierridotite/NOMA/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/pierridotite/NOMA/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/pierridotite/NOMA/releases/tag/v0.0.1
