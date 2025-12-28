<p align="center">
  <img src="NOMA_logo.png" alt="NOMA Logo" width="200"/>
</p>

# NOMA

**Neural-Oriented Machine Architecture**

A systems programming language where automatic differentiation is a compiler pass and model parameters are explicit, growable memory.

[![Stage](https://img.shields.io/badge/stage-alpha-green)](https://github.com/pierridotite/Noma)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Built with Rust](https://img.shields.io/badge/language-Rust-red)](https://www.rust-lang.org/)
[![Docs](https://img.shields.io/badge/docs-language_guide-orange)](LANGUAGE_GUIDE.md)
[![Discord](https://img.shields.io/badge/Discord-Join%20us-7289DA?logo=discord&logoColor=white)](https://discord.gg/GCYvkJWsPf)

**[Quick Start](QUICKSTART.md) | [Language Guide](LANGUAGE_GUIDE.md) | [Contributing](CONTRIBUTING.md) | [Discord](https://discord.gg/GCYvkJWsPf)**

---

## TL;DR

NOMA is an experimental systems language for machine learning where:

- Reverse-mode autodiff is implemented as a compiler pass (LLVM IR)
- Training loops are first-class language constructs
- Model parameters are explicit memory buffers that can grow during training
- Compiled programs produce small, standalone native binaries

The project explores whether treating training and topology changes as language semantics (instead of library behavior) enables new guarantees and workflows.

---

## Motivation

In mainstream ML frameworks, model structure and training state live inside framework-managed objects. Changing topology during training typically requires rebuilding graphs, copying parameters, and resetting optimizer state.

NOMA explores an alternative design: treating learnable parameters as managed memory, and optimization as a language-level construct. This makes topology changes (growth, pruning, resizing) explicit and mechanically well-defined.

---

## What is Different

NOMA is not "faster Python" and does not compete directly with PyTorch or JAX. Its focus is on semantics and guarantees.

[![](https://mermaid.ink/img/pako:eNqdVW1vmzAQ_iuWp3xzqvIWEqZNggSqSm03pdU-rMkHF5xgDTAyZk1W5b_vzEtC2lXTZiUI-5577vz4zrzgWCQMe3g0euEFVx56WRUIrbBKWc5W2IPXJ1rBGxmsf6OS06eMVRrQOGjTRhQqojnP9q3fdaGYJKjm44oW1bhikm8IqvaVYvm45gT5wJJ1zD3BPf_VhTXscjcwlpLnVO7nIhOyBXzYNOMt5oHt1BBnGMbUdN_iAiETJofIhbFwFsEAmfGCDQGTwDWnlwMAbFHxV3lFs8iPgEVjDp1um0w8xymVqpEMFuJa_jwJzEFKdFgVh8NotCpWxRGObpaaYDRCn_48OuuDpAlXXBQ0Q5GkOXsW8kf1d9eqftpKWqZDgscVfocOr9t9J1yyWJu79BDyDfB6vF_O1-jrXqVgmUNhgQMajz8j39TW5cP6RIaWdaG4rrGO0zdbqKWh18s1umrymouiUrJuop2wVou1NdZfrHsy5NdKJFwXRY-0W6SjkVdLjb3Se2OFAuq8rBUdMLMi0eq_Ea1buvty6yO_LKWgcfoO7CioBkPQc5_3BQxOAjYu96KW8VHAoBEwCtdN0jxjEpSEdtEJ95xBK2Bg9aL0YqCvtDqdXdBqF9i9zjc3327R9fIEaCULGsmC6ztICDT6yVDAC6j0P0h1r_bQKFs9izOItWAbFFOZoA3PMu9D1AwCxyh-MK9rsm46fuaJSj2j3BG584xLIvf6Get28rrO_XhGnPJtmsFfBfQYIAzN0DwGCCMbxqsAF85_hLgSoo-xmEfz0D3GMM2544T_GuMYBfqF-CbxLeI7JDBIYJLAJoHTyPZxALPP9juwwDGepdmxV3AU7Ow26CRq7qSTRE7ohu-cgd0kb5_Ymnr8fxpM8FbyBHvQxozgnMmc6il--6lJ2IbWmdJX5wHcSlp8FyLvPaWotyn2NjSrYFaXCVVswSm0W35clVCX-k6H-wBDdLchwd4L3mHPtS5mM9ecTOypYVnu1CF4D6DZ5cXEMBzbmsDPmdrmgeBfTdjLi-lsatiOaznmxJqaUFSYgbBC3rZfzebjefgNxNY79w?type=png)](https://mermaid.live/edit#pako:eNqdVW1vmzAQ_iuWp3xzqvIWEqZNggSqSm03pdU-rMkHF5xgDTAyZk1W5b_vzEtC2lXTZiUI-5577vz4zrzgWCQMe3g0euEFVx56WRUIrbBKWc5W2IPXJ1rBGxmsf6OS06eMVRrQOGjTRhQqojnP9q3fdaGYJKjm44oW1bhikm8IqvaVYvm45gT5wJJ1zD3BPf_VhTXscjcwlpLnVO7nIhOyBXzYNOMt5oHt1BBnGMbUdN_iAiETJofIhbFwFsEAmfGCDQGTwDWnlwMAbFHxV3lFs8iPgEVjDp1um0w8xymVqpEMFuJa_jwJzEFKdFgVh8NotCpWxRGObpaaYDRCn_48OuuDpAlXXBQ0Q5GkOXsW8kf1d9eqftpKWqZDgscVfocOr9t9J1yyWJu79BDyDfB6vF_O1-jrXqVgmUNhgQMajz8j39TW5cP6RIaWdaG4rrGO0zdbqKWh18s1umrymouiUrJuop2wVou1NdZfrHsy5NdKJFwXRY-0W6SjkVdLjb3Se2OFAuq8rBUdMLMi0eq_Ea1buvty6yO_LKWgcfoO7CioBkPQc5_3BQxOAjYu96KW8VHAoBEwCtdN0jxjEpSEdtEJ95xBK2Bg9aL0YqCvtDqdXdBqF9i9zjc3327R9fIEaCULGsmC6ztICDT6yVDAC6j0P0h1r_bQKFs9izOItWAbFFOZoA3PMu9D1AwCxyh-MK9rsm46fuaJSj2j3BG584xLIvf6Get28rrO_XhGnPJtmsFfBfQYIAzN0DwGCCMbxqsAF85_hLgSoo-xmEfz0D3GMM2544T_GuMYBfqF-CbxLeI7JDBIYJLAJoHTyPZxALPP9juwwDGepdmxV3AU7Ow26CRq7qSTRE7ohu-cgd0kb5_Ymnr8fxpM8FbyBHvQxozgnMmc6il--6lJ2IbWmdJX5wHcSlp8FyLvPaWotyn2NjSrYFaXCVVswSm0W35clVCX-k6H-wBDdLchwd4L3mHPtS5mM9ecTOypYVnu1CF4D6DZ5cXEMBzbmsDPmdrmgeBfTdjLi-lsatiOaznmxJqaUFSYgbBC3rZfzebjefgNxNY79w)

**Key differences:**

| Aspect | Traditional Frameworks | NOMA |
|--------|----------------------|------|
| Autodiff | Runtime library | Compiler pass |
| Training loops | Library API | Language construct (`optimize { }`) |
| Parameters | Framework objects | Explicit memory buffers (`alloc`/`realloc`/`free`) |
| Optimizer state | Hidden in optimizer | Tracked, preserved across topology changes |
| Output | Requires runtime | Standalone binary |

---

## Dynamic Topology Growth

NOMA allows parameter buffers to be resized during training using `realloc`. Existing values are preserved, new slots are initialized, and optimizer state for existing parameters is retained.

This enables experiments where model capacity adapts online, without restarting training or reconstructing optimizer state.

```noma
fn main() {
    learn W = tensor [[0.1], [0.2]];  // Start small
    
    optimize(W) until loss < 0.01 {
        let pred = matmul(X, W);
        let loss = mean((pred - Y) * (pred - Y));
        
        if loss > 0.5 {
            realloc W = [10, 1];  // Grow network, training continues
        }
        
        minimize loss;
    }
    
    return W;
}
```

[![](https://mermaid.ink/img/pako:eNqlVO1u2jAUfRXLFf8Cw_kieFslGEGqtnXVWg1phB8mccBqEke200Ipz7EH2ovN-YBmRe2mLULEH-eee-71iXcw5BGFGHY6O5YxhcEuyAAIoFrTlAYQ6-GSSD0yWuvfiGBkmVBZAqqAcivmmZqSlCXbOu4iU1QYoGBdSTLZlVSw2AByKxVNuwUzwEizJA3zgeCaPTRpkZ1vWpu5YCkR2w884aIGnMXVc4q5oRvVxiGEPHNwihtzEVHRRk7QxJmMW8iEZbQNcMcD0-u3ALpExZ7pmg6no6lmKTH7pm9xwu_DNRGqapleCAtx99RgplsJ9kG233c6QRZkRzj49LUkkMVyJUi-BmMac0HnAawHQFCSJDwM4KJWFDFBQ8V41gQCMEMaPZ8twIyy1VpJDObmzx9o8W4p3pzPv1zdLMAoIimQiiiKQWqAu4aNZlGppZV9FOt6NV31_pvc5klu1H9K_tH3rxbA3zCpWLbCIBdUu-SORvX2xeWFFndJ74FMeBlcOlRbRlskekG9PnmtmkbPK2i61e2ePwaw0Y3BDLwvBRloEcDHurga3umAa7XVh78qZ2FCpJzQGIRERCBmSYLPptVjSCX4LcWNcZpp955Fao1RvjHEBmt-sS3_w9IiuHHj29-I15RoJx6oK_scqX3HH_gvUNsVtf0aNalPpKb2_anZUu1arjtFz6h7zh90H-m1tYyZWXWlyil1y-ih1f9ayxNRbbL_4dEHeHtdcfXBazWb-UYXAA24EiyCWImCGjClIiXlFJ5eiRGNSZGo8hPf67CcZN85Tw-RgherNcQxSaSeFXmkrTlhRH9C6XFVlDbVd0-RKYiRa1UkEO_gBuKB1RsOB6br2h6yrIHnGHCrQcN-z0XIsS1X_xzPNvcGfKjS9nve0EO2M7Ac07U807YNSCOmuPhc3-7VJb__BYHQ4i0?type=png)](https://mermaid.live/edit#pako:eNqlVO1u2jAUfRXLFf8Cw_kieFslGEGqtnXVWg1phB8mccBqEke200Ipz7EH2ovN-YBmRe2mLULEH-eee-71iXcw5BGFGHY6O5YxhcEuyAAIoFrTlAYQ6-GSSD0yWuvfiGBkmVBZAqqAcivmmZqSlCXbOu4iU1QYoGBdSTLZlVSw2AByKxVNuwUzwEizJA3zgeCaPTRpkZ1vWpu5YCkR2w884aIGnMXVc4q5oRvVxiGEPHNwihtzEVHRRk7QxJmMW8iEZbQNcMcD0-u3ALpExZ7pmg6no6lmKTH7pm9xwu_DNRGqapleCAtx99RgplsJ9kG233c6QRZkRzj49LUkkMVyJUi-BmMac0HnAawHQFCSJDwM4KJWFDFBQ8V41gQCMEMaPZ8twIyy1VpJDObmzx9o8W4p3pzPv1zdLMAoIimQiiiKQWqAu4aNZlGppZV9FOt6NV31_pvc5klu1H9K_tH3rxbA3zCpWLbCIBdUu-SORvX2xeWFFndJ74FMeBlcOlRbRlskekG9PnmtmkbPK2i61e2ePwaw0Y3BDLwvBRloEcDHurga3umAa7XVh78qZ2FCpJzQGIRERCBmSYLPptVjSCX4LcWNcZpp955Fao1RvjHEBmt-sS3_w9IiuHHj29-I15RoJx6oK_scqX3HH_gvUNsVtf0aNalPpKb2_anZUu1arjtFz6h7zh90H-m1tYyZWXWlyil1y-ih1f9ayxNRbbL_4dEHeHtdcfXBazWb-UYXAA24EiyCWImCGjClIiXlFJ5eiRGNSZGo8hPf67CcZN85Tw-RgherNcQxSaSeFXmkrTlhRH9C6XFVlDbVd0-RKYiRa1UkEO_gBuKB1RsOB6br2h6yrIHnGHCrQcN-z0XIsS1X_xzPNvcGfKjS9nve0EO2M7Ac07U807YNSCOmuPhc3-7VJb__BYHQ4i0)

**What happens during `realloc`:**
1. Existing weights are preserved
2. New neurons are initialized (Xavier/He)
3. Optimizer momentum (Adam/RMSprop) is retained for existing parameters

---

## Project Status

**NOMA is alpha software.**

### What Works Today

- Lexer, parser, AST construction
- Computational graph with reverse-mode autodiff
- LLVM IR codegen with tensor support (matmul, sigmoid, relu, tanh, sum, mean)
- Native compilation via `build-exe`
- Multiple optimizers (SGD, Adam, RMSprop)
- Dynamic memory allocation (`alloc`/`realloc`/`free`)
- User-defined functions with autodiff support
- Batch processing, file I/O (CSV, Safetensors)

### Known Limitations

| Limitation | Description |
|------------|-------------|
| Single data type | Only `f64` (no int, bool, string) |
| No module system | Single-file programs only |
| Control flow | Compile-time evaluation (loops unroll the graph) |
| No recursion | Functions are inlined |
| No debugging | No breakpoints or source maps |
| Training timing | Training occurs during compilation; final weights are embedded |

---

## Experimental Results

We include a small, fully reproducible benchmark on a self-growing XOR network. This is a toy problem intended to validate semantics, not to claim real-world performance.

### Setup

- Task: XOR classification with dynamic network growth
- Architecture: 2→hidden→1, hidden layer grows from 4 to 8 neurons at step 200
- Optimizer: Adam (lr=0.1, β1=0.9, β2=0.999)
- Training: 200 iterations before growth + 120 after

### Results

| Implementation | Execution Time | Final Loss |
|----------------|----------------|------------|
| NOMA (compiled) | 0.8 ms | 0.0004 |
| C++ (manual gradients) | 0.8 ms | 0.0020 |
| Python + NumPy | 29 ms | 0.0020 |
| NOMA (interpreted) | 99 ms | 0.0007 |

### Observations

1. Compiled NOMA matches hand-written C++ execution time
2. NOMA achieves lower final loss because optimizer state is preserved across `realloc`
3. Baselines reset optimizer state after growth, requiring more iterations to reconverge

| Mode | Final Loss | Note |
|------|------------|------|
| NOMA (preserve state) | 0.0007 | Momentum preserved across growth |
| NOMA (reset state) | 0.0014 | State explicitly reset |
| NumPy / C++ | 0.0020 | State reset by reconstruction |

### Loss Curve

<p align="center">
  <img src="demo_self_growing_xor/assets/loss.png" alt="Loss curve comparison" width="600"/>
</p>

The vertical dashed line marks the `realloc` point (step 200). NOMA with preserved optimizer state shows faster reconvergence after growth compared to implementations that reset state.

**Reproducing:** See [demo_self_growing_xor/](demo_self_growing_xor/) for full benchmark code and scripts.

---

## Architecture Overview

[![](https://mermaid.ink/img/pako:eNqtVmtv2zYU_SsECwMbRruWbPmhbgVsy8qMOg84QVGs9gdGomwiEmVQVGsn8H77LinJj-bVbRGQRJc8PPfce6QbPeAgDRl2ca32wAVXLnqYC4TmWK1YwubYhdtbmsEdOVr_TCWntzHLNMAc0FtRKpRPEx5vi3MToZgkKOf1jIqsnjHJI4KybaZYUs85QQNgiUvmiuCa35dprfZ6c7S5ljyhcjtK41QWgHeRuR5jbthGHeMsy-rZ3ce4YSpDJo-RnuU53vAIGXPBjgGdYdfuNY8AUKLiP-jy-_7ABxaN2ZV9i-L0e7CiUpmWwUKQy2-HBnNoJdrNxW5Xq83FXOzhaDrTBLUa-qO6yngi1rl6ai_Lb5eSrlcF4qs2QiPxotAccskCxVNRUiN0PRsB6iv8WaCGSBOKsjSXASuPMBFqSU-K8CV4ZgAv6KhAkGSPf17NdPxFq_HHCzRlGwZNXaB6_SO6Gsyq9Ssqs8PG4PpGb0xmC337kuhK9TkPw5jVn9B9KrzAAfnRgeeFn80GV39WSkZpAk2nGkBjdKbp9no9DRp4IDdXacijCArKst9v5fuPv4xWlAs0y2P260-0f0iDu9e6X2IgZ4V-offTz-da3FA3H-7RZIbOmGDSFFIVcHl1cwK6XCue8PsTzHRy8akCjWIqlug3NOXiztj2WlmXuXrt2S4gkKHEPl_TcHJhhEwuFugCNH5jaMgFvLJFw_-2OujT8Klm-_AOIhrINMtQpugSxp0wL0tR4PiLDuGJM6Ex3yx4xTY0Roe6D2YB8u-Zr9UWBstSR0EM1nssQgGVIYp4HLvvfHORTMn0jrnlUCrD-nceqpVrrTdEblyrSeRW_w70-HHLSffhhNhIr5jNZNozj51xd_wMc9swt19iXvHlKoYfdZamlXZv5I_G3X0G2x45zviHDA3n59Xvc4DPZYpxy7d975DC6nf81r9NsU-iDSVgJoH5QsBNYpwk2j8CDzoxBmpzDrq0xye1H22BzSeay1QZOM6KgfyfrTjwVFP0DaiKyfYGROVweQOm4o3-H0SY4KXkIXaVzBnBCZMJ1SF-_G0TsojmsdL_q3dwbE3FX2maVCdlmi9X2I1onEGUr0OqmMcpTKBkvyqhaP0RkQuF3ZZtGRLsPuANhE6j3--2mm3L7jn9vtMjeItdu9lsdCzL6bS6vbbd7vV3BN-brM1Gr9-z2k635didVs9uOwSzkKtUnhdfaeZjbfcPyrQA_w?type=png)](https://mermaid.live/edit#pako:eNqtVmtv2zYU_SsECwMbRruWbPmhbgVsy8qMOg84QVGs9gdGomwiEmVQVGsn8H77LinJj-bVbRGQRJc8PPfce6QbPeAgDRl2ca32wAVXLnqYC4TmWK1YwubYhdtbmsEdOVr_TCWntzHLNMAc0FtRKpRPEx5vi3MToZgkKOf1jIqsnjHJI4KybaZYUs85QQNgiUvmiuCa35dprfZ6c7S5ljyhcjtK41QWgHeRuR5jbthGHeMsy-rZ3ce4YSpDJo-RnuU53vAIGXPBjgGdYdfuNY8AUKLiP-jy-_7ABxaN2ZV9i-L0e7CiUpmWwUKQy2-HBnNoJdrNxW5Xq83FXOzhaDrTBLUa-qO6yngi1rl6ai_Lb5eSrlcF4qs2QiPxotAccskCxVNRUiN0PRsB6iv8WaCGSBOKsjSXASuPMBFqSU-K8CV4ZgAv6KhAkGSPf17NdPxFq_HHCzRlGwZNXaB6_SO6Gsyq9Ssqs8PG4PpGb0xmC337kuhK9TkPw5jVn9B9KrzAAfnRgeeFn80GV39WSkZpAk2nGkBjdKbp9no9DRp4IDdXacijCArKst9v5fuPv4xWlAs0y2P260-0f0iDu9e6X2IgZ4V-offTz-da3FA3H-7RZIbOmGDSFFIVcHl1cwK6XCue8PsTzHRy8akCjWIqlug3NOXiztj2WlmXuXrt2S4gkKHEPl_TcHJhhEwuFugCNH5jaMgFvLJFw_-2OujT8Klm-_AOIhrINMtQpugSxp0wL0tR4PiLDuGJM6Ex3yx4xTY0Roe6D2YB8u-Zr9UWBstSR0EM1nssQgGVIYp4HLvvfHORTMn0jrnlUCrD-nceqpVrrTdEblyrSeRW_w70-HHLSffhhNhIr5jNZNozj51xd_wMc9swt19iXvHlKoYfdZamlXZv5I_G3X0G2x45zviHDA3n59Xvc4DPZYpxy7d975DC6nf81r9NsU-iDSVgJoH5QsBNYpwk2j8CDzoxBmpzDrq0xye1H22BzSeay1QZOM6KgfyfrTjwVFP0DaiKyfYGROVweQOm4o3-H0SY4KXkIXaVzBnBCZMJ1SF-_G0TsojmsdL_q3dwbE3FX2maVCdlmi9X2I1onEGUr0OqmMcpTKBkvyqhaP0RkQuF3ZZtGRLsPuANhE6j3--2mm3L7jn9vtMjeItdu9lsdCzL6bS6vbbd7vV3BN-brM1Gr9-z2k635didVs9uOwSzkKtUnhdfaeZjbfcPyrQA_w)

**Key insight:** Autodiff happens during compilation. Gradients are native machine instructions, not runtime library calls.

---

## Quick Start

```bash
# Clone and build
git clone https://github.com/pierridotite/Noma.git
cd Noma
cargo build --release

# Run in interpreter mode
cargo run -- run examples/03_gradient_descent.noma

# Compile to standalone binary
cargo run -- build-exe examples/12_linear_regression.noma -o model
./model
```

### Execution Modes

| Mode | Command | Use Case |
|------|---------|----------|
| Interpreter | `run` | Development, debugging |
| JIT | `fast-run` | Quick testing |
| Compile | `build-exe` | Production deployment |

---

## Language Overview

```noma
// Constants and learnable parameters
let x = 5.0;
learn w = 0.1;  // Gradients computed automatically

// User functions (autodiff-compatible)
fn mse(pred, target) {
    let error = pred - target;
    return mean(error * error);
}

// Tensors
let X = tensor [[1.0, 2.0], [3.0, 4.0]];
let W = tensor [[0.5], [0.3]];
let Y = matmul(X, W);

// Training loop
optimize(W) with adam(0.01) until loss < 0.001 {
    let pred = matmul(X, W);
    let loss = mse(pred, target);
    minimize loss;
}

// Dynamic allocation
alloc buffer = [rows, cols];
realloc buffer = [new_rows, cols];  // Preserves optimizer state
free buffer;
reset_optimizer();  // Explicitly clear momentum if needed
```

**[→ Full Language Guide](LANGUAGE_GUIDE.md)**

---

## Examples

| Example | Description |
|---------|-------------|
| [03_gradient_descent.noma](examples/03_gradient_descent.noma) | Minimize x² |
| [06_neural_network.noma](examples/06_neural_network.noma) | 2-layer perceptron |
| [12_linear_regression.noma](examples/12_linear_regression.noma) | Full ML pipeline |
| [20_growing_network.noma](examples/20_growing_network.noma) | Dynamic topology growth |
| [22_adam_optimizer.noma](examples/22_adam_optimizer.noma) | Adam optimizer |
| [28_batch_training.noma](examples/28_batch_training.noma) | Mini-batch SGD |

**[→ All Examples](examples/)**

---

## Roadmap

- [ ] True runtime control flow (dynamic branching)
- [ ] Multi-file projects and imports
- [ ] Additional data types (int, bool, string)
- [ ] Debugging support (source maps, breakpoints)
- [ ] Extended GPU backend (CUDA/PTX)
- [ ] Additional optimizers (L-BFGS, AdaGrad)

---

## Related Work

NOMA builds on ideas from two active research areas: **compiler-level automatic differentiation** and **dynamic neural architectures**.

### Autodiff as Program Transformation

| Paper | Relevance to NOMA |
|-------|-------------------|
| [Innes (2018) - *Don't Unroll Adjoint: Differentiating SSA-Form Programs*](https://arxiv.org/abs/1810.07951) | Reverse-mode AD on SSA/IR, foundational for compile-time autodiff |
| [van Merriënboer et al. (2018) - *Automatic differentiation in ML*](https://papers.neurips.cc/paper/8092-automatic-differentiation-in-ml-where-we-are-and-where-we-should-be-going.pdf) | Survey positioning AD approaches, runtime vs compile-time tradeoffs |
| [Abadi & Plotkin (2020) - *A Simple Differentiable Programming Language*](https://arxiv.org/abs/1911.04523) | Formal semantics for differentiable languages |
| [Moses & Churavy - *Enzyme*](https://enzyme.mit.edu/) | AD on LLVM IR; NOMA differs by owning the full language semantics |

### Differentiable Programming Languages

| Paper | Relevance to NOMA |
|-------|-------------------|
| [Hu et al. (2019) - *DiffTaichi*](https://arxiv.org/abs/1910.00935) | Domain-specific language with integrated AD |
| [Saeta et al. (2021) - *Swift for TensorFlow*](https://proceedings.mlsys.org/paper_files/paper/2021/file/5fd0b37cd7dbbb00f97ba6ce92bf5add-Paper.pdf) | First-class AD in a general-purpose language |
| [Bradbury et al. (2018) - *JAX*](https://github.com/google/jax) | Composable transformations (grad, jit, vmap) |

### Dynamic Architectures (Grow/Shrink)

| Paper | Relevance to NOMA |
|-------|-------------------|
| [Chen et al. (2016) - *Net2Net*](https://arxiv.org/abs/1511.05641) | Widening/deepening networks while preserving learned representations |
| [Karras et al. (2018) - *Progressive Growing of GANs*](https://openreview.net/forum?id=Hk99zCeAb) | Adding layers during training |
| [Cortes et al. (2017) - *AdaNet*](https://arxiv.org/abs/1607.01097) | Adaptive structure learning with theoretical guarantees |
| [Stanley & Miikkulainen (2002) - *NEAT*](https://dl.acm.org/doi/10.1162/106365602320169811) | Topology augmentation in neuroevolution |

---

## Research & Contributions

NOMA is intended as a research vehicle for exploring language-level semantics around differentiable programming, optimization state, and dynamic topology.

### Areas of Interest

- Optimizer implementations (L-BFGS, AdaGrad)
- Built-in operations (convolutions, pooling)
- Error messages and diagnostics
- BLAS/LAPACK integration
- GPU backend improvements

Feedback, issues, and experimental contributions are welcome.

**[→ Contributing Guide](CONTRIBUTING.md)**

---

## Why "NOMA"?

**N**eural-**O**riented **M**achine **A**rchitecture

The name reflects the design philosophy: neural network training as a first-class language feature, not a library. Gradients are treated like any other compiler concept-types, memory, optimization passes.

---

## Resources

- **[Language Guide](LANGUAGE_GUIDE.md)** - Complete language reference
- **[Quick Start](QUICKSTART.md)** - Installation and first steps
- **[Examples](examples/)** - 28 code samples
- **[VS Code Extension](noma-vscode/)** - Syntax highlighting
- **[Architecture Comparison](docs/ARCHITECTURE_COMPARISON.md)** - Detailed design document
