use std::fmt;

/// Token types recognized by the NOMA lexer
#[derive(Debug, Clone, PartialEq)]
pub enum TokenType {
    // Keywords
    Learn,       // learn
    Diff,        // diff
    Fn,          // fn
    Let,         // let
    Mut,         // mut
    Struct,      // struct
    Return,      // return
    If,          // if
    Else,        // else
    While,       // while
    Optimize,    // optimize
    Until,       // until
    Minimize,    // minimize
    GpuStruct,   // gpu_struct
    Alloc,       // alloc
    Free,        // free
    Realloc,     // realloc
    ResetOptimizer, // reset_optimizer
    
    // File I/O
    LoadCsv,     // load_csv
    SaveCsv,     // save_csv
    LoadSafetensors,  // load_safetensors
    LoadSafetensorsNamed,  // load_safetensors_named
    SaveSafetensors,  // save_safetensors
    
    // Batch processing
    Batch,       // batch
    In,          // in
    Epoch,       // epoch
    StreamingAdapt, // streaming_adapt
    Predict,     // predict
    Adapt,       // adapt
    
    // Types
    Tensor,      // tensor
    
    // String Literals
    StringLiteral(String),
    
    // Identifiers and Literals
    Identifier(String),
    Number(f64),
    
    // Operators
    Plus,        // +
    Minus,       // -
    Star,        // *
    Slash,       // /
    Percent,     // %
    Power,       // ** or ^
    Assign,      // =
    Lt,          // <
    Gt,          // >
    LtEq,        // <=
    GtEq,        // >=
    Equal,       // ==
    NotEq,       // !=
    And,         // &&
    Or,          // ||
    Not,         // !
    Dot,         // .
    As,          // as (for type casting)
    
    // Delimiters
    LParen,      // (
    RParen,      // )
    LBrace,      // {
    RBrace,      // }
    LBracket,    // [
    RBracket,    // ]
    Comma,       // ,
    Semicolon,   // ;
    Colon,       // :
    Arrow,       // ->
    
    // Special
    Eof,
    Newline,
}

/// A token with its type and position information
#[derive(Debug, Clone)]
pub struct Token {
    pub token_type: TokenType,
    pub line: usize,
    pub column: usize,
}

impl Token {
    pub fn new(token_type: TokenType, line: usize, column: usize) -> Self {
        Self {
            token_type,
            line,
            column,
        }
    }
}

impl fmt::Display for TokenType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TokenType::Learn => write!(f, "learn"),
            TokenType::Diff => write!(f, "diff"),
            TokenType::Fn => write!(f, "fn"),
            TokenType::Let => write!(f, "let"),
            TokenType::Mut => write!(f, "mut"),
            TokenType::Struct => write!(f, "struct"),
            TokenType::Return => write!(f, "return"),
            TokenType::Optimize => write!(f, "optimize"),
            TokenType::Until => write!(f, "until"),
            TokenType::Minimize => write!(f, "minimize"),
            TokenType::GpuStruct => write!(f, "gpu_struct"),
            TokenType::Alloc => write!(f, "alloc"),
            TokenType::Free => write!(f, "free"),
            TokenType::Realloc => write!(f, "realloc"),
            TokenType::ResetOptimizer => write!(f, "reset_optimizer"),
            TokenType::LoadCsv => write!(f, "load_csv"),
            TokenType::SaveCsv => write!(f, "save_csv"),
            TokenType::LoadSafetensors => write!(f, "load_safetensors"),
            TokenType::LoadSafetensorsNamed => write!(f, "load_safetensors_named"),
            TokenType::SaveSafetensors => write!(f, "save_safetensors"),
            TokenType::Batch => write!(f, "batch"),
            TokenType::In => write!(f, "in"),
            TokenType::Epoch => write!(f, "epoch"),
            TokenType::StreamingAdapt => write!(f, "streaming_adapt"),
            TokenType::Predict => write!(f, "predict"),
            TokenType::Adapt => write!(f, "adapt"),
            TokenType::Tensor => write!(f, "tensor"),
            TokenType::Identifier(name) => write!(f, "identifier({})", name),
            TokenType::Number(n) => write!(f, "number({})", n),
            TokenType::StringLiteral(s) => write!(f, "string(\"{}\")", s),
            _ => write!(f, "{:?}", self),
        }
    }
}
