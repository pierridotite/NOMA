# NOMA Language Guide

Complete reference for the NOMA programming language.

## Table of Contents
- [Variables](#variables)
- [Optimization Loop](#optimization-loop)
- [Hyperparameters](#hyperparameters)
- [Optimizers](#optimizers)
- [User-Defined Functions](#user-defined-functions)
- [Built-in Functions](#built-in-functions)
- [Random Number Generation](#random-number-generation)
- [Tensors](#tensors)
- [Dynamic Memory Allocation](#dynamic-memory-allocation)
- [File I/O](#file-io)
- [Batch Processing](#batch-processing)
- [Control Flow](#control-flow)

---

## Variables

```noma
let x = 5.0;        // Immutable constant
learn w = 0.1;      // Learnable parameter (tracked for gradients)
```

The `learn` keyword marks a variable as learnable, automatically including it in gradient computations during optimization.

---

## Optimization Loop

The core of NOMA: define what to optimize and let the compiler handle gradients automatically.

```noma
learn x = 5.0;

optimize(x) until loss < 0.0001 {
    let loss = x * x;
    minimize loss;
}
```

The `optimize` loop runs until the condition is met. The compiler automatically computes gradients and updates learnable parameters.

---

## Hyperparameters

Control training behavior with special variable names:

```noma
let learning_rate = 0.01;    // or: let lr = 0.01;
let max_iterations = 10000;  // or: let max_iter = 10000;

learn w = 0.0;
optimize(w) until loss < 0.001 {
    let loss = (w - 5.0) * (w - 5.0);
    minimize loss;
}
```

**Recognized hyperparameters:**
- `learning_rate` / `lr`: Step size for gradient updates (default: 0.01)
- `max_iterations` / `max_iter`: Maximum optimization iterations (default: 1000)

---

## Optimizers

NOMA supports three optimizers: **SGD**, **Adam**, and **RMSprop**.

### SGD (Stochastic Gradient Descent)

```noma
let learning_rate = 0.01;

learn w = 0.0;
optimize(w) until loss < 0.0001 {
    let loss = (w - 5.0) * (w - 5.0);
    minimize loss;
}
```

### Adam (Adaptive Moment Estimation)

```noma
let optimizer = 2.0;         // Select Adam
let learning_rate = 0.001;
let beta1 = 0.9;             // Momentum decay
let beta2 = 0.999;           // Squared gradient decay
let epsilon = 0.00000001;    // Numerical stability (1e-8)

learn w = 0.0;
optimize(w) until loss < 0.0001 {
    let loss = (w - 5.0) * (w - 5.0);
    minimize loss;
}
```

### RMSprop

```noma
let optimizer = 3.0;         // Select RMSprop
let learning_rate = 0.001;
let beta2 = 0.9;             // Squared gradient decay
let epsilon = 0.00000001;    // Numerical stability

learn w = 0.0;
optimize(w) until loss < 0.0001 {
    let loss = (w - 5.0) * (w - 5.0);
    minimize loss;
}
```

**Alternative selection syntax:**
```noma
let use_adam = 1.0;      // Non-zero enables Adam
let use_rmsprop = 1.0;   // Non-zero enables RMSprop
```

| Optimizer | Best For | Key Parameters |
|-----------|----------|----------------|
| **SGD** | Simple problems, fine-tuning | `learning_rate` |
| **Adam** | Most deep learning tasks | `learning_rate`, `beta1`, `beta2`, `epsilon` |
| **RMSprop** | RNNs, non-stationary objectives | `learning_rate`, `beta2`, `epsilon` |

---

## User-Defined Functions

Define reusable functions with automatic differentiation support:

```noma
// Basic function
fn square(x) {
    return x * x;
}

// Multi-parameter function
fn mse(pred, target) {
    let error = pred - target;
    return error * error;
}

// Functions can call other functions
fn polynomial(a, b, c, x) {
    return a * square(x) + b * x + c;
}

fn main() {
    let result = square(5.0);          // Returns 25.0
    let loss = mse(10.0, 8.0);         // Returns 4.0
    let y = polynomial(2.0, 3.0, 1.0, 2.0);  // Returns 15.0
    return y;
}
```

### Functions with Optimization

User functions work seamlessly with autodiff:

```noma
fn loss_fn(pred, target) {
    let err = pred - target;
    return err * err;
}

fn main() {
    learn w = 0.0;
    let target = 5.0;
    
    optimize(w) until loss < 0.0001 {
        let loss = loss_fn(w, target);
        minimize loss;
    }
    
    return w;  // Converges to 5.0
}
```

**Note:** Functions are inlined at compile-time, so recursion will cause infinite compilation loops.

---

## Built-in Functions

### Activation Functions
```noma
sigmoid(x)    // 1 / (1 + e^(-x))
relu(x)       // max(0, x)
tanh(x)       // hyperbolic tangent
```

### Math Functions
```noma
sin(x)        // sine
cos(x)        // cosine
exp(x)        // e^x
log(x)        // natural logarithm
sqrt(x)       // square root
abs(x)        // absolute value
floor(x)      // floor (no autodiff)
ceil(x)       // ceil (no autodiff)
```

### Tensor Operations
```noma
sum(tensor)   // Sum all elements → scalar
mean(tensor)  // Average of all elements → scalar
dot(a, b)     // Dot product (1D) → scalar
matmul(A, B)  // Matrix multiplication → tensor
```

### Utilities
```noma
print(x)      // Print value (returns input for chaining)
```

---

## Random Number Generation

### Basic RNG
```noma
rand()                    // Random float in [0, 1)
rand_uniform(min, max)    // Random float in [min, max)
rand_normal(mean, std)    // Random from N(mean, std)
```

### Tensor RNG
```noma
rand_tensor(d1, d2, ...)              // Tensor with uniform [0, 1)
rand_normal_tensor(mean, std, d1, d2, ...)  // Tensor with N(mean, std)
```

### Weight Initialization
```noma
xavier_init(fan_in, fan_out, d1, d2, ...)   // Xavier/Glorot (tanh/sigmoid)
he_init(fan_in, d1, d2, ...)                // He/Kaiming (ReLU)
```

**Example: Neural network layer initialization**
```noma
// Layer: 64 inputs → 32 outputs with ReLU
let W = he_init(64.0, 64.0, 32.0);
let b = rand_tensor(32.0);
```

---

## Tensors

### Creation
```noma
let v = tensor [1.0, 2.0, 3.0];              // 1D: shape [3]
let m = tensor [[1.0, 2.0], [3.0, 4.0]];     // 2D: shape [2, 2]
let neg = tensor [[-0.5], [1.0], [-2.5]];    // Negative values supported
```

### Elementwise Operations
```noma
let a = m + 1.0;       // Broadcasting: scalar added to all elements
let b = m * m;         // Elementwise multiply
let c = sigmoid(m);    // Apply function elementwise
```

### Reductions
```noma
let s = sum(m);        // Sum all elements → scalar
let u = mean(m);       // Average → scalar
```

### Indexing
```noma
let x = m[0][1];       // Access element (row-major order)
```

### Linear Algebra
```noma
let d = dot(v1, v2);           // Dot product (1D) → scalar
let p = matmul(A, B);          // Matrix multiply (2D) → 2D
let y = matmul(X, W);          // (n×k) @ (k×m) → (n×m)
```

---

## Dynamic Memory Allocation

Allocate tensors with runtime-determined shapes:

```noma
// Allocate heap tensor (initialized to zeros)
alloc buffer = [3, 3];

// Use computed dimensions
let rows = 4.0;
let cols = 8.0;
alloc workspace = [rows, cols];

// Access like any tensor
let element = buffer[1][2];

// Resize during execution (preserves data)
realloc buffer = [5, 5];

// Free when done
free buffer;
```

**Use cases:**
- Networks that grow during training
- Dynamic batch sizes
- Temporary workspace allocation
- Memory-efficient computation

---

## File I/O

### CSV Files

Load and save numeric data:

```noma
// Load CSV (numeric data, row-major)
load_csv data = "dataset.csv";

// Process
let scaled = data * 2.0;
let result = scaled + 1.0;

// Save to CSV
save_csv result, "output.csv";
```

**CSV Format:**
- Comma-separated numeric values
- One row per line
- Supports 1D and 2D tensors
- Comments start with `#`

### Safetensors Format

Binary format for efficient model storage:

```noma
// Save multiple tensors
save_safetensors {
    weights: W,
    bias: b,
    layer2: W2
}, "model.safetensors";

// Load tensors
load_safetensors model = "model.safetensors";

// Use loaded data
let output = matmul(X, model);
```

**Safetensors:**
- Efficient binary format
- Supports F32 and F64 dtypes
- Multiple tensors per file
- Industry-standard format

---

## Batch Processing

Process data in batches for efficient training:

### Basic Batch Loop
```noma
batch item in data with batch_size {
    let pred = matmul(item, W);
    print(pred);
}
```

### Batch Loop with Index
```noma
batch item, idx in data with batch_size {
    print(idx);        // Batch number: 0, 1, 2, ...
    print(item);       // Current batch data
    
    let pred = matmul(item, W);
}
```

### Full Training Example
```noma
fn main() {
    let X = tensor [/* training data */];
    let Y = tensor [/* labels */];
    learn W = tensor [[0.0], [0.0]];
    
    let batch_size = 32.0;
    let learning_rate = 0.01;
    
    optimize(W) until loss < 0.01 {
        // Process all batches
        batch x_batch in X with batch_size {
            let pred = matmul(x_batch, W);
        }
        
        // Compute overall loss
        let Y_pred = matmul(X, W);
        let loss = mean((Y_pred - Y) * (Y_pred - Y));
        minimize loss;
    }
    
    // Save trained model
    save_safetensors { weights: W }, "model.safetensors";
    return W;
}
```

---

## Control Flow

### If-Else
```noma
if condition {
    // then branch
} else {
    // else branch
}
```

### While Loop
```noma
while condition {
    // loop body
}
```

**Important:** Control flow is evaluated at compile-time during graph lowering. This means:
- Non-taken branches are not compiled
- While loops unroll the graph (use small iteration counts)
- Dynamic runtime branching is not yet supported

---

## Complete Example: Linear Regression

```noma
fn main() {
    // Dataset: 20 samples, 2 features
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
    
    // Initialize weights
    learn W = tensor [[0.0], [0.0]];
    
    let learning_rate = 0.001;
    let max_iterations = 100000;
    
    // Training
    optimize(W) until loss < 0.001 {
        let Y = matmul(X, W);
        let E = Y - T;
        let loss = mean(E * E);
        minimize loss;
    }
    
    print(loss);
    return W;  // Converges to [[1.5], [2.0]]
}
```

---

## Limitations

Current limitations to be aware of:

- **Single data type**: Only `f64` supported (no integers, strings, or booleans as first-class types)
- **No recursion**: User functions are inlined; recursive calls cause infinite compilation
- **Control flow**: Evaluated at compile-time; while loops unroll the graph
- **No autodiff through**: `floor`, `ceil`, or external C calls
- **No module system**: Single-file programs only
- **No debugging support**: No breakpoints or source maps yet
- **Comparison operators**: Return `0.0`/`1.0` instead of true booleans

---

## Next Steps

- See [README.md](README.md) for installation and quick start
- Browse [examples/](examples/) for more code samples
- Check the [architecture documentation](README.md#architecture) for compiler internals
