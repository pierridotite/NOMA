use clap::{Parser, Subcommand};
use noma_compiler::Lexer;
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
        Commands::Build { file, ast, tokens } => {
            build_file(file, ast, tokens)?;
        }
        Commands::Check { file } => {
            check_file(file)?;
        }
        Commands::Version => {
            println!("NOMA Compiler v{}", env!("CARGO_PKG_VERSION"));
            println!("The Neural-Oriented Machine Architecture");
            println!("Status: Pre-Alpha (Milestone 1 - The Skeleton)");
        }
    }

    Ok(())
}

fn build_file(file: PathBuf, print_ast: bool, print_tokens: bool) -> anyhow::Result<()> {
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

    if print_ast {
        println!("\n=== AST ===");
        println!("(Not yet implemented - Milestone 2)");
    }

    println!("\nLexical analysis: OK");
    println!("Total tokens: {}", tokens.len());

    Ok(())
}

fn check_file(file: PathBuf) -> anyhow::Result<()> {
    println!("Checking: {}", file.display());

    let source = fs::read_to_string(&file)?;
    let mut lexer = Lexer::new(&source);
    
    match lexer.tokenize() {
        Ok(tokens) => {
            println!("✓ Syntax check passed ({} tokens)", tokens.len());
            Ok(())
        }
        Err(e) => {
            eprintln!("✗ Syntax error: {}", e);
            Err(e.into())
        }
    }
}
