// NOMA Compiler Library
// The Neural-Oriented Machine Architecture Compiler

pub mod lexer;
pub mod token;
pub mod error;
pub mod ast;
pub mod parser;
pub mod graph;
pub mod llvm_codegen;
pub mod ptx_codegen;
pub mod nvptx_host;

pub use lexer::Lexer;
pub use token::{Token, TokenType};
pub use error::NomaError;
pub use ast::{Expression, Statement, Program, BinaryOperator, UnaryOperator, Item, FunctionDef};
pub use parser::Parser;
pub use graph::{ComputationalGraph, NodeId, NodeType, Tensor, Value, FunctionRegistry, UserFunction, OptimizerType, OptimizerConfig, OptimizerState, load_csv_file, save_csv_file, load_safetensors_file, save_safetensors_file};
pub use llvm_codegen::LLVMCodegen;
pub use ptx_codegen::PTXCodegen;
pub use nvptx_host::run_elementwise_kernel;
