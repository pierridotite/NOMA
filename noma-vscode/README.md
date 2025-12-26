# NOMA Language Extension for VS Code

Syntax highlighting, snippets, and language support for the NOMA programming language.

## Features

- **Syntax Highlighting** — Full TextMate grammar for `.noma` files
- **Snippets** — Quick templates for common patterns
- **Auto-closing** — Brackets, braces, parentheses, and strings
- **Indentation** — Smart indent/outdent for blocks

## Installation

### From Source (Development)

1. Clone the repository:
   ```bash
   cd noma-vscode
   npm install
   ```

2. Open VS Code and press `F5` to launch the Extension Development Host

### Package as VSIX

```bash
npm install
npm run package
```

Then install the `.vsix` file via VS Code: Extensions → ... → Install from VSIX

## Snippets

| Prefix | Description |
|--------|-------------|
| `fn main` | Main function template |
| `fn` | Function template |
| `let` | Immutable variable |
| `learn` | Learnable parameter |
| `optimize` | Optimization loop |
| `tensor1d` | 1D tensor |
| `tensor2d` | 2D tensor (matrix) |
| `matmul` | Matrix multiplication |
| `mse` | Mean squared error |
| `hyper` | Hyperparameters |
| `linreg` | Full linear regression template |
| `sigmoid` | Sigmoid activation |
| `relu` | ReLU activation |
| `print` | Print value |
| `sum` | Sum reduction |
| `mean` | Mean reduction |
| `dot` | Dot product |

## Syntax Highlighting

The extension provides highlighting for:

- **Keywords** — `fn`, `let`, `learn`, `return`, `optimize`, `until`, `minimize`, `if`, `else`, `while`, `for`
- **Types** — `tensor`
- **Built-ins** — `sigmoid`, `relu`, `sum`, `mean`, `dot`, `matmul`, `print`, `exp`, `log`, `sqrt`, `abs`, `sin`, `cos`, `tan`
- **Operators** — Arithmetic, comparison, logical, assignment
- **Numbers** — Integers and floats (including scientific notation)
- **Comments** — Line (`//`) and block (`/* */`)
- **Strings** — Double-quoted with escape sequences

## Example

```noma
fn main() {
    learn x = 5.0;
    
    let learning_rate = 0.01;
    let max_iterations = 1000;
    
    optimize(x) until loss < 0.0001 {
        let loss = x * x;
        minimize loss;
    }
    
    print(x);
    return x;
}
```

## License

MIT
