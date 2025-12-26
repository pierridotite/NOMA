# NOMA Architecture Diagram

## Comparison: Traditional ML Frameworks vs NOMA

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRADITIONAL ML FRAMEWORKS (PyTorch, TF)                  │
└─────────────────────────────────────────────────────────────────────────────┘

    Python Source Code
         │
         ├──> Python Interpreter (Runtime)
         │         │
         │         ├──> PyTorch Library
         │         │         │
         │         │         ├──> Autograd Engine (Runtime Graph Building)
         │         │         │
         │         │         ├──> Backward Pass (Runtime Gradient Computation)
         │         │         │
         │         │         └──> CUDA/CPU Kernels
         │         │
         │         └──> NumPy/BLAS
         │
         └──> Result

    Dependencies: Python + PyTorch + CUDA + NumPy (~100 MB runtime)
    Execution: Interpreted, Runtime Overhead
    Binary: No standalone binary (requires full Python environment)


┌─────────────────────────────────────────────────────────────────────────────┐
│                              NOMA APPROACH                                   │
└─────────────────────────────────────────────────────────────────────────────┘

    NOMA Source Code (.noma)
         │
         ├──> NOMA Compiler (Compile-Time)
         │         │
         │         ├──> Lexer → Parser → AST
         │         │
         │         ├──> Computational Graph Builder
         │         │
         │         ├──> AUTODIFF PASS (Chain Rule Applied at Compile-Time)
         │         │         │
         │         │         └──> Gradient Code Generated as Native Instructions
         │         │
         │         ├──> LLVM IR Generation
         │         │         │
         │         │         └──> Optimized Machine Code
         │         │
         │         └──> clang Linker
         │
         └──> Standalone Native Binary (16-50 KB)
                   │
                   └──> Runs directly on CPU (no dependencies)

    Dependencies: NONE (standalone binary)
    Execution: Native machine code (15x faster)
    Binary: Single executable, 16-50 KB


┌─────────────────────────────────────────────────────────────────────────────┐
│                         KEY DIFFERENCE: WHERE AUTODIFF HAPPENS               │
└─────────────────────────────────────────────────────────────────────────────┘

    PyTorch/TensorFlow:
        ┌──────────────────┐
        │  Runtime Library │  ←── Autodiff computed during execution
        │  (Interpreter)   │      (overhead on every run)
        └──────────────────┘

    NOMA:
        ┌──────────────────┐
        │ Compiler Pass    │  ←── Autodiff computed once at compile-time
        │ (LLVM IR Gen)    │      (baked into binary as native code)
        └──────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                    EXECUTION FLOW: NOMA vs PYTORCH                          │
└─────────────────────────────────────────────────────────────────────────────┘

PyTorch (Runtime):
    ┌─────┐    ┌──────────┐    ┌─────────┐    ┌──────────┐
    │ Run │───▶│ Interpret│───▶│ Build   │───▶│ Compute  │
    │ Code│    │ Python   │    │ Graph   │    │ Gradients│
    └─────┘    └──────────┘    └─────────┘    └──────────┘
                    ↓                ↓              ↓
               Overhead          Overhead       Overhead

NOMA (Compile Once, Run Fast):
    ┌─────────┐              ┌───────────────┐
    │ Compile │─────────────▶│ Run Native    │
    │ Once    │  (Autodiff   │ Binary        │
    │         │   Done Here) │ (No Overhead) │
    └─────────┘              └───────────────┘
         ↑                           │
         │                           └──▶ 15x Faster
    One-time cost                    Every execution


┌─────────────────────────────────────────────────────────────────────────────┐
│                  FEATURE COMPARISON TABLE                                    │
└─────────────────────────────────────────────────────────────────────────────┘

Feature                 │ PyTorch/TF           │ NOMA
────────────────────────┼──────────────────────┼──────────────────────────
Autodiff                │ Runtime library      │ Compiler pass
Execution               │ Interpreted          │ Native binary
Binary size             │ ~100 MB+ (runtime)   │ 16-50 KB
Dependencies            │ Many (Python, libs)  │ None (standalone)
Startup time            │ Slow (interpreter)   │ Instant (native)
Dynamic topology        │ Restart required     │ realloc during training
Memory model            │ Garbage collected    │ Deterministic
Speed                   │ Baseline             │ 15x faster
Deployment              │ Complex (env setup)  │ Copy single binary


┌─────────────────────────────────────────────────────────────────────────────┐
│                  EXAMPLE: DYNAMIC TOPOLOGY GROWTH                            │
└─────────────────────────────────────────────────────────────────────────────┘

PyTorch:
    1. Stop training
    2. Create new model with more neurons
    3. Copy old weights to new model
    4. Rebuild optimizer state
    5. Resume training
    
    Result: Complex, error-prone, breaks training loop


NOMA:
    optimize(W) until loss < 0.01 {
        if loss > 0.5 {
            realloc W = [10, 1];  // Grow network instantly
        }
        minimize loss;
    }
    
    Result: Single line, training continues seamlessly
