use clap::{Parser, Subcommand};
use noma_compiler::{Lexer, Parser as NomaParser, ComputationalGraph};
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "noma")]
#[command(about = "NOMA Compiler - Neural-Oriented Machine Architecture", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build a NOMA source file
    Build {
        /// Input .noma file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Print the Abstract Syntax Tree
        #[arg(short, long)]
        ast: bool,

        /// Print tokens from lexer
        #[arg(short, long)]
        tokens: bool,

        /// Print the computational graph
        #[arg(short, long)]
        graph: bool,
    },

    /// Check syntax without building
    Check {
        /// Input .noma file
        #[arg(value_name = "FILE")]
        file: PathBuf,
    },

    /// Display compiler version and build info
    Version,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Build { file, ast: print_ast, tokens: print_tokens, graph: print_graph } => {
            build_file(file, print_ast, print_tokens, print_graph)?;
        }
        Commands::Check { file } => {
            check_file(file)?;
        }
        Commands::Version => {
            println!("NOMA Compiler v{}", env!("CARGO_PKG_VERSION"));
            println!("The Neural-Oriented Machine Architecture");
            println!("Status: Pre-Alpha (Milestone 2 - The Graph Engine)");
        }
    }

    Ok(())
}

fn build_file(file: PathBuf, print_ast: bool, print_tokens: bool, print_graph: bool) -> anyhow::Result<()> {
    println!("Building: {}", file.display());

    // Read source file
    let source = fs::read_to_string(&file)?;

    // Lexical analysis
    let mut lexer = Lexer::new(&source);
    let tokens = lexer.tokenize()?;

    if print_tokens {
        println!("\n=== TOKENS ===");
        for (idx, token) in tokens.iter().enumerate() {
            println!("{:3}: {:?}", idx, token);
        }
    }

    // Parsing
    let mut parser = NomaParser::new(tokens.clone());
    let program = match parser.parse() {
        Ok(prog) => prog,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            return Err(e.into());
        }
    };

    if print_ast {
        println!("\n=== AST ===");
        println!("{:#?}", program);
    }

    // Build computational graph
    let mut _graph = ComputationalGraph::new();
    
    if print_graph {
        println!("\n=== COMPUTATIONAL GRAPH ===");
        _graph.print_structure();
    }

    println!("\nCompilation: OK");
    println!("Total tokens: {}", tokens.len());
    println!("Items: {}", program.items.len());

    Ok(())
}

fn check_file(file: PathBuf) -> anyhow::Result<()> {
    println!("Checking: {}", file.display());

    let source = fs::read_to_string(&file)?;
    let mut lexer = Lexer::new(&source);
    
    match lexer.tokenize() {
        Ok(tokens) => {
            let mut parser = NomaParser::new(tokens);
            match parser.parse() {
                Ok(_) => {
                    println!("Syntax check: OK");
                    Ok(())
                }
                Err(e) => {
                    eprintln!("Parse error: {}", e);
                    Err(e.into())
                }
            }
        }
        Err(e) => {
            eprintln!("Lexical error: {}", e);
            Err(e.into())
        }
    }
}
