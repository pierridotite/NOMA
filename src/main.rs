use clap::{Parser, Subcommand};
use noma_compiler::{Lexer, Parser as NomaParser, ComputationalGraph, LLVMCodegen, PTXCodegen};
use std::fs;
use std::path::PathBuf;
use std::collections::HashMap;
use std::process::Command;
use std::env;

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

    /// Compile NOMA to LLVM IR
    Compile {
        /// Input .noma file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Output LLVM IR file (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Optimize the LLVM IR (uses opt)
        #[arg(short = 'O', long)]
        optimize: bool,

        /// Optimization level (1,2,3). Defaults to 3 when optimize is enabled.
        #[arg(long = "opt-level", value_parser = clap::value_parser!(u8).range(1..=3))]
        opt_level: Option<u8>,

        /// Emit native assembly via llc (if available)
        #[arg(long = "emit-asm")]
        emit_asm: bool,

        /// Emit native object via llc (if available)
        #[arg(long = "emit-obj")]
        emit_obj: bool,

        /// Enable fast-math optimizations (unsafe FP transforms)
        #[arg(long = "fast-math")]
        fast_math: bool,
    },

    /// Build a standalone native executable
    BuildExe {
        /// Input .noma file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Output executable path
        #[arg(short, long, value_name = "OUTPUT")]
        output: PathBuf,

        /// Optimization level (0,1,2,3). Defaults to 2.
        #[arg(short = 'O', long = "opt-level", value_parser = clap::value_parser!(u8).range(0..=3))]
        opt_level: Option<u8>,

        /// Enable fast-math optimizations
        #[arg(long = "fast-math")]
        fast_math: bool,
    },

    /// Compile NOMA to PTX (placeholder backend)
    CompilePtx {
        /// Input .noma file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Output PTX file (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Number of elements for elementwise kernels (host stub)
        #[arg(long = "n-elems")] 
        n_elems: Option<u32>,

        /// Print a pseudo host launch stub for the PTX kernel
        #[arg(long = "host-stub")]
        host_stub: bool,

        /// Optimize PTX via opt with NVPTX backend
        #[arg(short = 'O', long)]
        optimize: bool,

        /// Enable fast-math for PTX (ftz, approx_div, etc.)
        #[arg(long = "fast-math")]
        fast_math: bool,
    },

    /// Check syntax without building
    Check {
        /// Input .noma file
        #[arg(value_name = "FILE")]
        file: PathBuf,
    },

    /// Run autodiff demo: minimize y = x^2
    Demo,

    /// Display compiler version and build info
    Version,

    /// Load PTX and attempt an elementwise kernel launch (feature: cuda)
    RunPtx {
        /// PTX input file to load
        #[arg(value_name = "PTX_FILE")]
        ptx_file: PathBuf,

        /// Number of elements for elementwise kernel
        #[arg(long = "n-elems")]
        n_elems: u32,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Build { file, ast: print_ast, tokens: print_tokens, graph: print_graph } => {
            build_file(file, print_ast, print_tokens, print_graph)?;
        }
            Commands::Compile { file, output, optimize, opt_level, emit_asm, emit_obj, fast_math } => {
                compile_to_llvm(file, output, optimize, opt_level, emit_asm, emit_obj, fast_math)?;
        }
        Commands::BuildExe { file, output, opt_level, fast_math } => {
            build_executable(file, output, opt_level, fast_math)?;
        }
        Commands::CompilePtx { file, output, n_elems, host_stub, optimize, fast_math } => {
            compile_to_ptx(file, output, n_elems, host_stub, optimize, fast_math)?;
        }
        Commands::Check { file } => {
            check_file(file)?;
        }
        Commands::Demo => {
            run_demo()?;
        }
        Commands::Version => {
            println!("NOMA Compiler v{}", env!("CARGO_PKG_VERSION"));
            println!("The Neural-Oriented Machine Architecture");
            println!("Status: Pre-Alpha (Milestone 4 - The Metal)");
        }
        Commands::RunPtx { ptx_file, n_elems } => {
            let ptx = fs::read_to_string(&ptx_file)?;
            match noma_compiler::run_elementwise_kernel(&ptx, "compute", n_elems) {
                Ok(out) => {
                    println!("Kernel executed. First 8 outputs: {:?}", &out[..out.len().min(8)]);
                }
                Err(e) => {
                    println!("[info] PTX host launch unavailable: {}", e);
                    println!("Hint: build with --features cuda and ensure NVIDIA drivers are installed.");
                }
            }
        }
    }

    Ok(())
}

fn run_optimize_loop(
    graph: &mut ComputationalGraph,
    variables: &HashMap<String, noma_compiler::NodeId>,
    cond_id: noma_compiler::NodeId,
    objective_id: noma_compiler::NodeId,
    target: &str,
    learning_rate: f64,
    max_iter: usize,
) -> Result<(), String> {
    if !variables.contains_key(target) {
        return Err(format!("Optimize target '{}' not defined", target));
    }

    for _ in 0..max_iter {
        graph.forward_pass()?;
        let cond_val = graph.get_node(cond_id).and_then(|n| n.value.clone()).and_then(|v| match v { noma_compiler::Value::Scalar(s) => Some(s), _ => None }).unwrap_or(0.0);
        if cond_val != 0.0 {
            return Ok(());
        }

        graph.backward_pass(objective_id)?;
        graph.optimize_step(learning_rate)?;
        graph.reset_gradients();
    }

    Err("Optimize loop reached max iterations without satisfying condition".to_string())
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

fn run_demo() -> anyhow::Result<()> {
    println!("NOMA Autodiff Demo: Minimize y = x^2");
    println!("======================================\n");

    let mut graph = ComputationalGraph::new();
    
    // Create: y = x^2 (x * x)
    let x = graph.add_learnable("x".to_string(), 5.0);
    let y = graph.add_binary_op("mul", x, x);

    println!("Initial state: x = 5.0");
    println!("Goal: Minimize y = x^2 (find x ≈ 0)\n");
    
    println!("Iteration | x value | y value | gradient |");
    println!("-----------|---------|---------|----------|");
    
    let learning_rate = 0.1;
    let max_iterations = 50;

    // Training loop
    for iteration in 0..=max_iterations {
        // Forward pass
        graph.forward_pass().map_err(|e| anyhow::anyhow!(e))?;
        
        let x_val = graph.get_node(x).and_then(|n| n.value.clone()).and_then(|v| match v { noma_compiler::Value::Scalar(s) => Some(s), _ => None }).unwrap_or(0.0);
        let y_val = graph.get_node(y).and_then(|n| n.value.clone()).and_then(|v| match v { noma_compiler::Value::Scalar(s) => Some(s), _ => None }).unwrap_or(0.0);

        // Print progress
        if iteration == 0 || iteration % 5 == 0 || iteration == max_iterations {
            println!("{:^9} | {:7.4} | {:7.4} | -        |", iteration, x_val, y_val);
        }

        // Backward pass
        graph.backward_pass(y).map_err(|e| anyhow::anyhow!(e))?;

        // Optimization step
        graph.optimize_step(learning_rate).map_err(|e| anyhow::anyhow!(e))?;
        graph.reset_gradients();

        // Early stopping if converged
        if y_val.abs() < 1e-6 {
            println!("\nConverged at iteration {}!", iteration);
            break;
        }
    }

    println!();
    let final_x = graph.get_node(x).and_then(|n| n.value.clone()).and_then(|v| match v { noma_compiler::Value::Scalar(s) => Some(s), _ => None }).unwrap_or(0.0);
    let final_y = graph.get_node(y).and_then(|n| n.value.clone()).and_then(|v| match v { noma_compiler::Value::Scalar(s) => Some(s), _ => None }).unwrap_or(0.0);

    println!("Final result:");
    println!("  x = {:.6}", final_x);
    println!("  y = {:.6}", final_y);
    println!("\nSuccess! Gradient descent found a minimum near x = 0");

    Ok(())
}

fn compile_to_llvm(file: PathBuf, output: Option<PathBuf>, optimize: bool, opt_level: Option<u8>, emit_asm: bool, emit_obj: bool, fast_math: bool) -> anyhow::Result<()> {
    // Read source file
    let source = fs::read_to_string(&file)?;

    // Tokenize and parse
    let mut lexer = Lexer::new(&source);
    let tokens = lexer.tokenize().map_err(|e| anyhow::anyhow!("{:?}", e))?;
    
    let mut parser = NomaParser::new(tokens);
    let ast = parser.parse().map_err(|e| anyhow::anyhow!("{:?}", e))?;

    // Lower the first function (main) to a computational graph
    let mut graph = ComputationalGraph::new();
    let mut variables: HashMap<String, noma_compiler::NodeId> = HashMap::new();
    let mut last_node: Option<noma_compiler::NodeId> = None;

    // Find a function to compile (prefers main)
    let maybe_func = ast.items.iter().find_map(|item| {
        if let noma_compiler::Item::Function(func) = item {
            if func.name == "main" { return Some(func); }
        }
        None
    }).or_else(|| ast.items.iter().find_map(|item| {
        if let noma_compiler::Item::Function(func) = item { Some(func) } else { None }
    }));

    let func = maybe_func.ok_or_else(|| anyhow::anyhow!("No function found to compile"))?;

    fn lower_statements(
        graph: &mut ComputationalGraph,
        variables: &mut HashMap<String, noma_compiler::NodeId>,
        stmts: &[noma_compiler::Statement],
        last_node: &mut Option<noma_compiler::NodeId>,
    ) -> Result<(), String> {
        for stmt in stmts {
            match stmt {
                noma_compiler::Statement::LearnDeclaration { name, value } => {
                    let init_val = if let noma_compiler::Expression::Number(n) = value { *n } else { 0.0 };
                    let node_id = graph.add_learnable(name.clone(), init_val);
                    variables.insert(name.clone(), node_id);
                    *last_node = Some(node_id);
                }
                noma_compiler::Statement::LetDeclaration { name, value } => {
                    let val_id = graph.build_from_expression(value, variables)?;
                    variables.insert(name.clone(), val_id);
                    *last_node = Some(val_id);
                }
                noma_compiler::Statement::Assignment { name, value } => {
                    let val_id = graph.build_from_expression(value, variables)?;
                    variables.insert(name.clone(), val_id);
                    *last_node = Some(val_id);
                }
                noma_compiler::Statement::Minimize(expr) => {
                    let id = graph.build_from_expression(expr, variables)?;
                    *last_node = Some(id);
                }
                noma_compiler::Statement::Expression(expr) => {
                    let id = graph.build_from_expression(expr, variables)?;
                    *last_node = Some(id);
                }
                noma_compiler::Statement::Return(opt_expr) => {
                    if let Some(expr) = opt_expr {
                        let id = graph.build_from_expression(expr, variables)?;
                        *last_node = Some(id);
                    }
                }
                noma_compiler::Statement::Block(inner) => {
                    lower_statements(graph, variables, inner, last_node)?;
                }
                noma_compiler::Statement::OptimizeLoop { target, condition, body, .. } => {
                    let cond_id = graph.build_from_expression(condition, variables)?;

                    let mut loop_last: Option<noma_compiler::NodeId> = None;
                    lower_statements(graph, variables, body, &mut loop_last)?;
                    let objective = loop_last.or(*last_node).ok_or_else(|| "Optimize loop body produced no expressions".to_string())?;

                    run_optimize_loop(graph, variables, cond_id, objective, target, 0.1, 1000)?;
                    *last_node = Some(objective);
                }
            }
        }
        Ok(())
    }

    lower_statements(&mut graph, &mut variables, &func.body, &mut last_node)
        .map_err(|e| anyhow::anyhow!(e))?;

    // Ensure we have something to return
    let _output_node = last_node.ok_or_else(|| anyhow::anyhow!("No expressions to compile"))?;

    // Perform forward pass to compute values (best-effort; allows constants/learnables)
    let _ = graph.forward_pass();

    // Generate LLVM IR
    let mut codegen = LLVMCodegen::new().with_fast_math(fast_math);
    let mut ir = codegen.generate(&graph).map_err(|e| anyhow::anyhow!(e))?;

    let mut run_opt = optimize || opt_level.is_some();
    if run_opt {
        let level = opt_level.unwrap_or(3).clamp(1, 3);
        let opt_flag = format!("-O{}", level);

        // Check opt availability
        match Command::new("opt").arg("--version").output() {
            Ok(_) => {}
            Err(_) => {
                println!("[warn] --optimize requested but 'opt' not found in PATH; emitting unoptimized IR");
                run_opt = false;
            }
        }

        if run_opt {
            let tmp_dir = env::temp_dir();
            let input_path = tmp_dir.join("noma_ir.ll");
            let output_path = tmp_dir.join("noma_ir_opt.ll");
            fs::write(&input_path, &ir)?;

            let status = Command::new("opt")
                .arg("-S")
                .arg(&opt_flag)
                .arg(&input_path)
                .arg("-o")
                .arg(&output_path)
                .status();

            match status {
                Ok(s) if s.success() => {
                    ir = fs::read_to_string(&output_path)?;
                    println!("[info] --optimize applied via opt {}", opt_flag);
                }
                _ => {
                    println!("[warn] --optimize requested but opt failed; emitting unoptimized IR");
                }
            }
        }
    }

    // Optionally emit assembly/object via llc
    if emit_asm || emit_obj {
        match Command::new("llc").arg("--version").output() {
            Ok(_) => {}
            Err(_) => {
                println!("[warn] --emit-asm/--emit-obj requested but 'llc' not found in PATH; skipping");
                // If neither output requested now possible, fall through with IR only
                return output_ir(output, ir);
            }
        }

        let tmp_dir = env::temp_dir();
        let ir_path = tmp_dir.join("noma_ir.ll");
        fs::write(&ir_path, &ir)?;

        if emit_asm {
            let asm_path = tmp_dir.join("noma.s");
            let status = Command::new("llc")
                .arg("-filetype=asm")
                .arg(&ir_path)
                .arg("-o")
                .arg(&asm_path)
                .status();
            if let Ok(s) = status {
                if s.success() {
                    let asm = fs::read_to_string(&asm_path)?;
                    println!("=== ASM (llc) ===\n{}", asm);
                } else {
                    println!("[warn] llc asm emission failed (status {})", s);
                }
            }
        }

        if emit_obj {
            let obj_path = tmp_dir.join("noma.o");
            let status = Command::new("llc")
                .arg("-filetype=obj")
                .arg(&ir_path)
                .arg("-o")
                .arg(&obj_path)
                .status();
            if let Ok(s) = status {
                if s.success() {
                    println!("[info] Object emitted at {}", obj_path.display());
                } else {
                    println!("[warn] llc object emission failed (status {})", s);
                }
            }
        }
    }

    output_ir(output, ir)
}

fn output_ir(output: Option<PathBuf>, ir: String) -> anyhow::Result<()> {
    match output {
        Some(out_file) => {
            fs::write(out_file.clone(), ir.clone())?;
            println!("Generated LLVM IR to: {}", out_file.display());
        }
        None => {
            println!("{}", ir);
        }
    }
    Ok(())
}

fn compile_to_ptx(file: PathBuf, output: Option<PathBuf>, n_elems: Option<u32>, host_stub: bool, optimize: bool, fast_math: bool) -> anyhow::Result<()> {
    // Read source file
    let source = fs::read_to_string(&file)?;

    // Tokenize and parse
    let mut lexer = Lexer::new(&source);
    let tokens = lexer.tokenize().map_err(|e| anyhow::anyhow!("{:?}", e))?;
    
    let mut parser = NomaParser::new(tokens);
    let ast = parser.parse().map_err(|e| anyhow::anyhow!("{:?}", e))?;

    // Lower first function to graph (reuse compile_to_llvm lowering helper)
    let mut graph = ComputationalGraph::new();
    let mut variables: HashMap<String, noma_compiler::NodeId> = HashMap::new();
    let mut last_node: Option<noma_compiler::NodeId> = None;

    let maybe_func = ast.items.iter().find_map(|item| {
        if let noma_compiler::Item::Function(func) = item {
            if func.name == "main" { return Some(func); }
        }
        None
    }).or_else(|| ast.items.iter().find_map(|item| {
        if let noma_compiler::Item::Function(func) = item { Some(func) } else { None }
    }));

    let func = maybe_func.ok_or_else(|| anyhow::anyhow!("No function found to compile"))?;

    fn lower_statements(
        graph: &mut ComputationalGraph,
        variables: &mut HashMap<String, noma_compiler::NodeId>,
        stmts: &[noma_compiler::Statement],
        last_node: &mut Option<noma_compiler::NodeId>,
    ) -> Result<(), String> {
        for stmt in stmts {
            match stmt {
                noma_compiler::Statement::LearnDeclaration { name, value } => {
                    let init_val = if let noma_compiler::Expression::Number(n) = value { *n } else { 0.0 };
                    let node_id = graph.add_learnable(name.clone(), init_val);
                    variables.insert(name.clone(), node_id);
                    *last_node = Some(node_id);
                }
                noma_compiler::Statement::LetDeclaration { name, value } => {
                    let val_id = graph.build_from_expression(value, variables)?;
                    variables.insert(name.clone(), val_id);
                    *last_node = Some(val_id);
                }
                noma_compiler::Statement::Assignment { name, value } => {
                    let val_id = graph.build_from_expression(value, variables)?;
                    variables.insert(name.clone(), val_id);
                    *last_node = Some(val_id);
                }
                noma_compiler::Statement::Minimize(expr) => {
                    let id = graph.build_from_expression(expr, variables)?;
                    *last_node = Some(id);
                }
                noma_compiler::Statement::Expression(expr) => {
                    let id = graph.build_from_expression(expr, variables)?;
                    *last_node = Some(id);
                }
                noma_compiler::Statement::Return(opt_expr) => {
                    if let Some(expr) = opt_expr {
                        let id = graph.build_from_expression(expr, variables)?;
                        *last_node = Some(id);
                    }
                }
                noma_compiler::Statement::Block(inner) => {
                    lower_statements(graph, variables, inner, last_node)?;
                }
                noma_compiler::Statement::OptimizeLoop { target, condition, body, .. } => {
                    let cond_id = graph.build_from_expression(condition, variables)?;
                    let mut loop_last: Option<noma_compiler::NodeId> = None;
                    lower_statements(graph, variables, body, &mut loop_last)?;
                    let objective = loop_last.or(*last_node).ok_or_else(|| "Optimize loop body produced no expressions".to_string())?;

                    run_optimize_loop(graph, variables, cond_id, objective, target, 0.1, 1000)?;
                    *last_node = Some(objective);
                }
            }
        }
        Ok(())
    }

    lower_statements(&mut graph, &mut variables, &func.body, &mut last_node)
        .map_err(|e| anyhow::anyhow!(e))?;

    let mut codegen = PTXCodegen::new();
    let ptx = codegen.generate(&graph).map_err(|e| anyhow::anyhow!(e))?;

    // Log NVPTX-specific optimization availability
    if optimize || fast_math {
        println!("[info] NVPTX optimizations requested (--optimize or --fast-math)");
        if fast_math {
            println!("[info] Fast-math: FMA fusion, approx div/sqrt when compiling PTX to device code");
        }
    }

    match output {
        Some(out_file) => {
            fs::write(out_file.clone(), ptx.clone())?;
            println!("Generated PTX (placeholder) to: {}", out_file.display());
        }
        None => {
            println!("{}", ptx);
        }
    }

    if host_stub {
        let n = n_elems.unwrap_or(0);
        println!("\n=== PTX Host Launch Stub (pseudo) ===");
        println!("Kernel: compute");
        println!("Params: .param .u64 in_ptr, .param .u64 out_ptr, .param .u32 n_elems");
        println!("n_elems: {}", n);
        println!("block_dim: 128");
        println!("grid_dim: (n_elems + 127) / 128");
        println!("Bind params: [in_ptr], [out_ptr], [n_elems]");
        println!("Thread index: %tid.x, byte offset: %rd_idx = tid * 8");
        println!("Load element: add.u64 %rd2, in_ptr, base + %rd_idx");
        println!("Store element: add.u64 %rd3, out_ptr, %rd_idx");
        println!("======================================\n");
    }

    Ok(())
}
// Add before the final closing brace of main.rs

fn build_executable(file: PathBuf, output: PathBuf, opt_level: Option<u8>, fast_math: bool) -> anyhow::Result<()> {
    println!("Building executable: {} -> {}", file.display(), output.display());
    
    // Read source file
    let source = fs::read_to_string(&file)?;

    // Tokenize and parse
    let mut lexer = Lexer::new(&source);
    let tokens = lexer.tokenize().map_err(|e| anyhow::anyhow!("{:?}", e))?;
    
    let mut parser = NomaParser::new(tokens);
    let ast = parser.parse().map_err(|e| anyhow::anyhow!("{:?}", e))?;

    // Lower the first function to a computational graph
    let mut graph = ComputationalGraph::new();
    let mut variables: HashMap<String, noma_compiler::NodeId> = HashMap::new();
    let mut last_node: Option<noma_compiler::NodeId> = None;

    let maybe_func = ast.items.iter().find_map(|item| {
        if let noma_compiler::Item::Function(func) = item {
            if func.name == "main" { return Some(func); }
        }
        None
    }).or_else(|| ast.items.iter().find_map(|item| {
        if let noma_compiler::Item::Function(func) = item { Some(func) } else { None }
    }));

    let func = maybe_func.ok_or_else(|| anyhow::anyhow!("No function found to compile"))?;

    fn lower_statements(
        graph: &mut ComputationalGraph,
        variables: &mut HashMap<String, noma_compiler::NodeId>,
        stmts: &[noma_compiler::Statement],
        last_node: &mut Option<noma_compiler::NodeId>,
    ) -> Result<(), String> {
        for stmt in stmts {
            match stmt {
                noma_compiler::Statement::LearnDeclaration { name, value } => {
                    let init_val = if let noma_compiler::Expression::Number(n) = value { *n } else { 0.0 };
                    let node_id = graph.add_learnable(name.clone(), init_val);
                    variables.insert(name.clone(), node_id);
                    *last_node = Some(node_id);
                }
                noma_compiler::Statement::LetDeclaration { name, value } => {
                    let val_id = graph.build_from_expression(value, variables)?;
                    variables.insert(name.clone(), val_id);
                    *last_node = Some(val_id);
                }
                noma_compiler::Statement::Assignment { name, value } => {
                    let val_id = graph.build_from_expression(value, variables)?;
                    variables.insert(name.clone(), val_id);
                    *last_node = Some(val_id);
                }
                noma_compiler::Statement::Minimize(expr) => {
                    let id = graph.build_from_expression(expr, variables)?;
                    *last_node = Some(id);
                }
                noma_compiler::Statement::Expression(expr) => {
                    let id = graph.build_from_expression(expr, variables)?;
                    *last_node = Some(id);
                }
                noma_compiler::Statement::Return(opt_expr) => {
                    if let Some(expr) = opt_expr {
                        let id = graph.build_from_expression(expr, variables)?;
                        *last_node = Some(id);
                    }
                }
                noma_compiler::Statement::Block(inner) => {
                    lower_statements(graph, variables, inner, last_node)?;
                }
                noma_compiler::Statement::OptimizeLoop { target, condition, body, .. } => {
                    // Lower the body first - this creates the nodes
                    lower_statements(graph, variables, body, last_node)?;
                    
                    // The last node after lowering body should be the loss (from minimize statement)
                    let objective_id = last_node.ok_or("No objective in optimize loop")?;
                    
                    // Build condition expression ONCE (using variables that now exist)
                    let cond_id = graph.build_from_expression(condition, variables)?;
                    
                    // Run optimization at compile-time
                    let lr = 0.001;  // Small learning rate for stability on hard problems
                    let max_iter = 100000;  // More iterations to compensate for small lr
                    for _ in 0..max_iter {
                        graph.forward_pass()?;
                        let cond_val = graph.get_node(cond_id)
                            .and_then(|n| n.value.clone())
                            .and_then(|v| match v { noma_compiler::Value::Scalar(s) => Some(s), _ => None })
                            .unwrap_or(0.0);
                        if cond_val != 0.0 {
                            break;
                        }
                        graph.backward_pass(objective_id)?;
                        graph.optimize_step(lr)?;
                        graph.reset_gradients();
                    }
                    
                    // After optimization, set last_node to target variable's value
                    if let Some(&target_id) = variables.get(target) {
                        *last_node = Some(target_id);
                    }
                }
            }
        }
        Ok(())
    }

    lower_statements(&mut graph, &mut variables, &func.body, &mut last_node)
        .map_err(|e| anyhow::anyhow!(e))?;

    // Perform forward pass (values are already computed after optimization)
    let _ = graph.forward_pass();

    // Generate LLVM IR with a main() wrapper, returning the specific node from the last statement
    let mut codegen = LLVMCodegen::new().with_fast_math(fast_math);
    let compute_ir = codegen.generate_with_return(&graph, last_node).map_err(|e| anyhow::anyhow!(e))?;

    // Wrap in a main() that calls compute() and prints result
    let wrapped_ir = format!(
        "{}\n\ndefine i32 @main() {{\nentry:\n  %result = call double @compute()\n  %resultptr = alloca double\n  store double %result, double* %resultptr\n  %fmt = getelementptr [4 x i8], [4 x i8]* @.str, i32 0, i32 0\n  call i32 (i8*, ...) @printf(i8* %fmt, double %result)\n  ret i32 0\n}}\n",
        compute_ir
    );

    // Write IR to temporary file
    let tmp_dir = env::temp_dir();
    let ir_path = tmp_dir.join("noma_build.ll");
    fs::write(&ir_path, &wrapped_ir)?;

    // Optimize with opt if requested
    let opt_level_val = opt_level.unwrap_or(2);
    if opt_level_val > 0 {
        match Command::new("opt").arg("--version").output() {
            Ok(_) => {
                let opt_output = tmp_dir.join("noma_build_opt.ll");
                let opt_flag = format!("-O{}", opt_level_val);
                let status = Command::new("opt")
                    .arg("-S")
                    .arg(&opt_flag)
                    .arg(&ir_path)
                    .arg("-o")
                    .arg(&opt_output)
                    .status();
                
                match status {
                    Ok(s) if s.success() => {
                        println!("[info] Optimized IR with opt {}", opt_flag);
                        fs::copy(&opt_output, &ir_path)?;
                    }
                    _ => println!("[warn] opt failed; using unoptimized IR"),
                }
            }
            Err(_) => println!("[warn] opt not found; skipping optimization"),
        }
    }

    // Compile IR to object file with llc
    let obj_path = tmp_dir.join("noma_build.o");
    match Command::new("llc").arg("--version").output() {
        Ok(_) => {
            let status = Command::new("llc")
                .arg("-filetype=obj")
                .arg(&ir_path)
                .arg("-o")
                .arg(&obj_path)
                .status();
            
            match status {
                Ok(s) if s.success() => println!("[info] Compiled IR to object file"),
                _ => return Err(anyhow::anyhow!("llc failed to compile IR")),
            }
        }
        Err(_) => return Err(anyhow::anyhow!("llc not found; cannot compile to native code")),
    }

    // Link object file to executable with gcc or clang
    let linkers = vec!["gcc", "clang"];
    let mut link_success = false;

    for linker in &linkers {
        match Command::new(linker)
            .arg(&obj_path)
            .arg("-lm")  // Link math library
            .arg("-no-pie")  // Avoid PIE relocation issues
            .arg("-o")
            .arg(&output)
            .status()
        {
            Ok(s) if s.success() => {
                println!("[info] Linked executable with {}", linker);
                println!("✅ Built standalone executable: {}", output.display());
                link_success = true;
                break;
            }
            _ => continue,
        }
    }

    if !link_success {
        return Err(anyhow::anyhow!("Failed to link executable (tried gcc, clang)"));
    }

    Ok(())
}
