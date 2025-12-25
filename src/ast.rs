use std::fmt;

/// Represents expressions in the NOMA language
#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    /// Numeric literal (e.g., 5.0, 42)
    Number(f64),
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperator {
    Add,       // +
    Sub,       // -
    Mul,       // *
    Div,       // /
    Pow,       // ^ or **
    Equal,     // ==
    NotEqual,  // !=
    Less,      // <
    Greater,   // >
    LessEq,    // <=
    GreaterEq, // >=
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
    /// Block of statements
    Block(Vec<Statement>),
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
            BinaryOperator::Pow => write!(f, "^"),
            BinaryOperator::Equal => write!(f, "=="),
            BinaryOperator::NotEqual => write!(f, "!="),
            BinaryOperator::Less => write!(f, "<"),
            BinaryOperator::Greater => write!(f, ">"),
            BinaryOperator::LessEq => write!(f, "<="),
            BinaryOperator::GreaterEq => write!(f, ">="),
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
