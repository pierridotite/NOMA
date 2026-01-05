use std::fmt;

/// Represents expressions in the NOMA language
#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    /// Numeric literal (e.g., 5.0, 42)
    Number(f64),
    /// String literal (e.g., "hello.csv")
    StringLiteral(String),
    /// Tensor literal with flat data and shape (row-major)
    TensorLiteral {
        data: Vec<f64>,
        shape: Vec<usize>,
    },
    /// Identifier (e.g., x, y, main)
    Identifier(String),
    /// Binary operation (e.g., x + y, x * y)
    BinaryOp {
        left: Box<Expression>,
        op: BinaryOperator,
        right: Box<Expression>,
    },
    /// Unary operation (e.g., -x)
    UnaryOp {
        op: UnaryOperator,
        expr: Box<Expression>,
    },
    /// Function call (e.g., sigmoid(x), dot(a, b))
    Call {
        name: String,
        args: Vec<Expression>,
    },
    /// Indexing into tensors: a[i], a[i][j]
    Index {
        target: Box<Expression>,
        indices: Vec<Expression>,
    },
    /// Type cast (e.g., x as f64)
    Cast {
        expr: Box<Expression>,
        target_type: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperator {
    Add,       // +
    Sub,       // -
    Mul,       // *
    Div,       // /
    Mod,       // %
    Pow,       // ^ or **
    Equal,     // ==
    NotEqual,  // !=
    Less,      // <
    Greater,   // >
    LessEq,    // <=
    GreaterEq, // >=
    And,       // &&
    Or,        // ||
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOperator {
    Neg, // -
    Not, // !
}

/// Represents statements in the NOMA language
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    /// Variable declaration with 'learn' keyword
    LearnDeclaration {
        name: String,
        value: Expression,
    },
    /// Variable declaration with 'let' keyword
    LetDeclaration {
        name: String,
        value: Expression,
    },
    /// Assignment (e.g., x = 5.0)
    Assignment {
        name: String,
        value: Expression,
    },
    /// Minimize statement (e.g., minimize loss)
    Minimize(Expression),
    /// Optimize loop (e.g., optimize(model) until loss < 0.01 { ... })
    OptimizeLoop {
        target: String,
        condition: Expression,
        body: Vec<Statement>,
    },
    /// Expression statement (e.g., print(...))
    Expression(Expression),
    /// Return statement
    Return(Option<Expression>),
    /// If-else control flow
    If {
        condition: Expression,
        then_branch: Vec<Statement>,
        else_branch: Vec<Statement>,
    },
    /// While loop
    While {
        condition: Expression,
        body: Vec<Statement>,
    },
    /// Block of statements
    Block(Vec<Statement>),
    /// Dynamic allocation: alloc name = tensor_shape;
    /// Creates a heap-allocated tensor with the given shape
    Alloc {
        name: String,
        shape: Vec<Expression>,
    },
    /// Deallocation: free name;
    /// Frees a heap-allocated tensor
    Free {
        name: String,
    },
    /// Reallocation: realloc name = [new_shape];
    /// Resizes a heap-allocated tensor (preserves data where possible)
    Realloc {
        name: String,
        shape: Vec<Expression>,
    },
    /// Reset optimizer state: reset_optimizer();
    /// Clears Adam m/v/t state to restart optimization from scratch
    ResetOptimizer,
    /// Load tensor from CSV file: let data = load_csv("path.csv");
    LoadCsv {
        name: String,
        path: String,
    },
    /// Save tensor to CSV file: save_csv(tensor, "path.csv");
    SaveCsv {
        tensor: Expression,
        path: String,
    },
    /// Load tensors from Safetensors file: let model = load_safetensors("model.safetensors");
    LoadSafetensors {
        name: String,
        path: String,
    },
    /// Load a specific named tensor from Safetensors file: load_safetensors_named x = "file.safetensors" : "tensor_name";
    LoadSafetensorsNamed {
        name: String,
        path: String,
        tensor_name: String,
    },
    /// Save tensors to Safetensors file: save_safetensors(tensors, "model.safetensors");
    SaveSafetensors {
        tensors: Vec<(String, Expression)>,
        path: String,
    },
    /// Batch loop: batch item, index in data with batch_size { body }
    /// Iterates over data in batches
    BatchLoop {
        item_name: String,
        index_name: Option<String>,
        data: Expression,
        batch_size: Expression,
        body: Vec<Statement>,
    },
    /// Epoch training loop with synchronized mini-batch gradient descent
    /// epoch N batch X, Y with batch_size as x_batch, y_batch { body with minimize }
    /// Runs N epochs, each epoch processes all batches with weight updates per batch
    EpochLoop {
        epochs: Expression,
        x_data: Expression,
        y_data: Expression,
        batch_size: Expression,
        x_batch_name: String,
        y_batch_name: String,
        body: Vec<Statement>,
    },
    /// Streaming TTA loop with causal prediction-then-adapt pattern
    /// streaming_adapt X, Y with batch_size -> x_batch, y_batch
    ///     predict { prediction_body } -> pred_var
    ///     adapt { adaptation_body with minimize }
    /// For each batch:
    ///   1. Execute predict body and store predictions (BEFORE any updates)
    ///   2. Execute adapt body with minimize (updates only learn variables)
    ///   3. Move to next batch
    /// This ensures causal predictions where each batch is predicted with
    /// the model state from BEFORE seeing that batch.
    StreamingAdaptLoop {
        x_data: Expression,
        y_data: Expression,
        batch_size: Expression,
        x_batch_name: String,
        y_batch_name: String,
        predict_body: Vec<Statement>,
        prediction_output: String,      // Variable name to accumulate predictions
        adapt_body: Vec<Statement>,
    },
}

/// Function definition
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDef {
    pub name: String,
    pub params: Vec<String>,
    pub body: Vec<Statement>,
}

/// Struct definition
#[derive(Debug, Clone, PartialEq)]
pub struct StructDef {
    pub name: String,
    pub fields: Vec<(String, String)>, // (name, type)
}

/// Top-level item in a NOMA program
#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    Function(FunctionDef),
    Struct(StructDef),
}

/// The root of the AST - represents a complete NOMA program
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub items: Vec<Item>,
}

impl Program {
    pub fn new() -> Self {
        Program { items: Vec::new() }
    }

    pub fn add_function(&mut self, name: String, params: Vec<String>, body: Vec<Statement>) {
        self.items.push(Item::Function(FunctionDef { name, params, body }));
    }

    pub fn add_struct(&mut self, name: String, fields: Vec<(String, String)>) {
        self.items.push(Item::Struct(StructDef { name, fields }));
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Expression::Number(n) => write!(f, "{}", n),
            Expression::StringLiteral(s) => write!(f, "\"{}\"", s),
            Expression::TensorLiteral { data, shape } => {
                write!(f, "tensor[shape={:?}, data={:?}]", shape, data)
            }
            Expression::Identifier(name) => write!(f, "{}", name),
            Expression::BinaryOp { left, op, right } => {
                write!(f, "({} {} {})", left, op, right)
            }
            Expression::UnaryOp { op, expr } => {
                write!(f, "({}{})", op, expr)
            }
            Expression::Call { name, args } => {
                let args_str = args.iter()
                    .map(|a| a.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "{}({})", name, args_str)
            }
            Expression::Index { target, indices } => {
                let idx_str = indices.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", ");
                write!(f, "{}[{}]", target, idx_str)
            }
            Expression::Cast { expr, target_type } => {
                write!(f, "({} as {})", expr, target_type)
            }
        }
    }
}

impl fmt::Display for BinaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BinaryOperator::Add => write!(f, "+"),
            BinaryOperator::Sub => write!(f, "-"),
            BinaryOperator::Mul => write!(f, "*"),
            BinaryOperator::Div => write!(f, "/"),
            BinaryOperator::Mod => write!(f, "%"),
            BinaryOperator::Pow => write!(f, "^"),
            BinaryOperator::Equal => write!(f, "=="),
            BinaryOperator::NotEqual => write!(f, "!="),
            BinaryOperator::Less => write!(f, "<"),
            BinaryOperator::Greater => write!(f, ">"),
            BinaryOperator::LessEq => write!(f, "<="),
            BinaryOperator::GreaterEq => write!(f, ">="),
            BinaryOperator::And => write!(f, "&&"),
            BinaryOperator::Or => write!(f, "||"),
        }
    }
}

impl fmt::Display for UnaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            UnaryOperator::Neg => write!(f, "-"),
            UnaryOperator::Not => write!(f, "!"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expression_display() {
        let expr = Expression::BinaryOp {
            left: Box::new(Expression::Identifier("x".to_string())),
            op: BinaryOperator::Add,
            right: Box::new(Expression::Number(5.0)),
        };
        assert_eq!(expr.to_string(), "(x + 5)");
    }

    #[test]
    fn test_function_def() {
        let func = FunctionDef {
            name: "main".to_string(),
            params: vec![],
            body: vec![],
        };
        assert_eq!(func.name, "main");
    }

    #[test]
    fn test_program() {
        let mut program = Program::new();
        program.add_function("main".to_string(), vec![], vec![]);
        assert_eq!(program.items.len(), 1);
    }
}
