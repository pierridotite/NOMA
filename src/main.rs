use clap::{Parser, Subcommand};
use noma_compiler::{Lexer, Parser as NomaParser, ComputationalGraph, LLVMCodegen, PTXCodegen, FunctionRegistry, OptimizerType, OptimizerConfig, OptimizerState};
use std::fs;
use std::path::PathBuf;
use std::collections::HashMap;
use std::process::Command;
use std::env;

/// Helper function to collect user-defined functions from AST into a registry
fn collect_functions(ast: &noma_compiler::Program) -> (FunctionRegistry, Option<noma_compiler::FunctionDef>) {
    let mut func_registry = FunctionRegistry::new();
    let mut main_func = None;

    for item in &ast.items {
        if let noma_compiler::Item::Function(func) = item {
            if func.name == "main" {
                main_func = Some(func.clone());
            } else {
                // Register non-main functions for inlining
                func_registry.register(func.name.clone(), func.params.clone(), func.body.clone());
            }
        }
    }

    // If no main, use the first function as main
    let main_func = main_func.or_else(|| {
        ast.items.iter().find_map(|item| {
            if let noma_compiler::Item::Function(f) = item { Some(f.clone()) } else { None }
        })
    });

    (func_registry, main_func)
}

/// Shared function to lower statements into the computational graph with user function support
fn lower_statements_shared(
    graph: &mut ComputationalGraph,
    variables: &mut HashMap<String, noma_compiler::NodeId>,
    stmts: &[noma_compiler::Statement],
    last_node: &mut Option<noma_compiler::NodeId>,
    func_registry: &FunctionRegistry,
    optimizer_state: &mut OptimizerState,
) -> Result<(), String> {
    for stmt in stmts {
        match stmt {
            noma_compiler::Statement::LearnDeclaration { name, value } => {
                match value {
                    noma_compiler::Expression::Number(n) => {
                        let node_id = graph.add_learnable(name.clone(), *n);
                        variables.insert(name.clone(), node_id);
                        *last_node = Some(node_id);
                    }
                    noma_compiler::Expression::TensorLiteral { data, shape } => {
                        let node_id = graph.add_learnable_tensor(name.clone(), data.clone(), shape.clone())
                            .map_err(|e| e.to_string())?;
                        variables.insert(name.clone(), node_id);
                        *last_node = Some(node_id);
                    }
                    _ => {
                        // Evaluate expression to get an initial value (scalar or tensor)
                        let val_id = graph.build_from_expression_with_functions(value, variables, func_registry)?;
                        graph.forward_pass()?;
                        let init_val = graph.get_node(val_id)
                            .and_then(|n| n.value.clone())
                            .ok_or_else(|| format!("Failed to evaluate initializer for '{}'", name))?;

                        let node_id = match init_val {
                            noma_compiler::Value::Scalar(s) => graph.add_learnable(name.clone(), s),
                            noma_compiler::Value::Tensor(t) => graph.add_learnable_tensor(name.clone(), t.data, t.shape)
                                .map_err(|e| e.to_string())?,
                        };

                        variables.insert(name.clone(), node_id);
                        *last_node = Some(node_id);
                    }
                }
            }
            noma_compiler::Statement::LetDeclaration { name, value } => {
                let val_id = graph.build_from_expression_with_functions(value, variables, func_registry)?;
                variables.insert(name.clone(), val_id);
                *last_node = Some(val_id);
            }
            noma_compiler::Statement::Assignment { name, value } => {
                let val_id = graph.build_from_expression_with_functions(value, variables, func_registry)?;
                variables.insert(name.clone(), val_id);
                *last_node = Some(val_id);
            }
            noma_compiler::Statement::Minimize(expr) => {
                let id = graph.build_from_expression_with_functions(expr, variables, func_registry)?;
                *last_node = Some(id);
            }
            noma_compiler::Statement::Expression(expr) => {
                let id = graph.build_from_expression_with_functions(expr, variables, func_registry)?;
                *last_node = Some(id);
            }
            noma_compiler::Statement::Return(opt_expr) => {
                if let Some(expr) = opt_expr {
                    let id = graph.build_from_expression_with_functions(expr, variables, func_registry)?;
                    *last_node = Some(id);
                }
            }
            noma_compiler::Statement::Block(inner) => {
                lower_statements_shared(graph, variables, inner, last_node, func_registry, optimizer_state)?;
            }
            noma_compiler::Statement::If { condition, then_branch, else_branch } => {
                let cond_id = graph.build_from_expression_with_functions(condition, variables, func_registry)?;
                let _ = graph.forward_pass();
                let cond_val = graph.get_node(cond_id)
                    .and_then(|n| n.value.clone())
                    .and_then(|v| match v { noma_compiler::Value::Scalar(s) => Some(s), _ => None })
                    .unwrap_or(0.0);
                if cond_val != 0.0 {
                    lower_statements_shared(graph, variables, then_branch, last_node, func_registry, optimizer_state)?;
                } else {
                    lower_statements_shared(graph, variables, else_branch, last_node, func_registry, optimizer_state)?;
                }
            }
            noma_compiler::Statement::While { condition, body } => {
                for _ in 0..1_000_000usize {
                    let cond_id = graph.build_from_expression_with_functions(condition, variables, func_registry)?;
                    let _ = graph.forward_pass();
                    let cond_val = graph.get_node(cond_id)
                        .and_then(|n| n.value.clone())
                        .and_then(|v| match v { noma_compiler::Value::Scalar(s) => Some(s), _ => None })
                        .unwrap_or(0.0);
                    if cond_val == 0.0 { break; }
                    lower_statements_shared(graph, variables, body, last_node, func_registry, optimizer_state)?;
                }
            }
            noma_compiler::Statement::OptimizeLoop { target, condition, body, .. } => {
                // Lower body first so condition can reference values like `loss`
                let mut loop_last: Option<noma_compiler::NodeId> = None;
                lower_statements_shared(graph, variables, body, &mut loop_last, func_registry, optimizer_state)?;
                let objective = loop_last.or(*last_node).ok_or_else(|| "Optimize loop body produced no expressions".to_string())?;
                let cond_id = graph.build_from_expression_with_functions(condition, variables, func_registry)?;

                let (config, iters) = pick_hyperparams(graph, variables, 0.1, 1000);
                // Use shared optimizer state to preserve momentum across optimize loops
                run_optimize_loop(graph, variables, cond_id, objective, target, config, iters, optimizer_state)?;
                *last_node = Some(objective);
            }
            noma_compiler::Statement::Alloc { name, shape } => {
                // Evaluate shape dimensions at lowering time
                let mut dims = Vec::new();
                for dim_expr in shape {
                    let dim_id = graph.build_from_expression_with_functions(dim_expr, variables, func_registry)?;
                    graph.forward_pass()?;
                    let dim_val = graph.get_node(dim_id)
                        .and_then(|n| n.value.clone())
                        .and_then(|v| match v { noma_compiler::Value::Scalar(s) => Some(s as usize), _ => None })
                        .ok_or_else(|| format!("Alloc dimension must be a scalar for '{}'", name))?;
                    dims.push(dim_val);
                }
                let node_id = graph.add_heap_tensor(name.clone(), dims)?;
                variables.insert(name.clone(), node_id);
                *last_node = Some(node_id);
            }
            noma_compiler::Statement::Free { name } => {
                graph.free_heap_tensor(name)?;
                variables.remove(name);
                // free doesn't change last_node
            }
            noma_compiler::Statement::Realloc { name, shape } => {
                // Evaluate shape dimensions; avoid forward passes during realloc to prevent transient mismatches
                let mut dims = Vec::new();
                for dim_expr in shape {
                    if let noma_compiler::Expression::Number(n) = dim_expr {
                        dims.push(*n as usize);
                    } else {
                        let dim_id = graph.build_from_expression_with_functions(dim_expr, variables, func_registry)?;
                        // Fallback: single forward pass for non-literal dims
                        graph.forward_pass()?;
                        let dim_val = graph.get_node(dim_id)
                            .and_then(|n| n.value.clone())
                            .and_then(|v| match v { noma_compiler::Value::Scalar(s) => Some(s as usize), _ => None })
                            .ok_or_else(|| format!("Realloc dimension must be a scalar for '{}'", name))?;
                        dims.push(dim_val);
                    }
                }
                // Prefer reallocating learnable tensors; fallback to heap tensors
                if let Some(existing_id) = variables.get(name) {
                    if let Some(node) = graph.get_node(*existing_id) {
                        match &node.node_type {
                            noma_compiler::NodeType::Learnable(_) => {
                                let node_id = graph.realloc_learnable_tensor_by_id(*existing_id, dims.clone())?;
                                variables.insert(name.clone(), node_id);
                                *last_node = Some(node_id);
                            }
                            _ => {
                                let node_id = graph.realloc_heap_tensor(name, dims)?;
                                variables.insert(name.clone(), node_id);
                                *last_node = Some(node_id);
                            }
                        }
                    } else {
                        let node_id = graph.realloc_heap_tensor(name, dims)?;
                        variables.insert(name.clone(), node_id);
                        *last_node = Some(node_id);
                    }
                } else {
                    let node_id = graph.realloc_heap_tensor(name, dims)?;
                    variables.insert(name.clone(), node_id);
                    *last_node = Some(node_id);
                }
            }
            noma_compiler::Statement::LoadCsv { name, path } => {
                // Load CSV file and create tensor
                let tensor_data = noma_compiler::load_csv_file(path)?;
                let node_id = graph.add_constant_tensor(tensor_data.0, tensor_data.1)?;
                variables.insert(name.clone(), node_id);
                *last_node = Some(node_id);
            }
            noma_compiler::Statement::SaveCsv { tensor, path } => {
                // Evaluate tensor and save to CSV
                let tensor_id = graph.build_from_expression_with_functions(tensor, variables, func_registry)?;
                graph.forward_pass()?;
                let tensor_val = graph.get_node(tensor_id)
                    .and_then(|n| n.value.clone())
                    .ok_or_else(|| "Cannot evaluate tensor for save_csv".to_string())?;
                noma_compiler::save_csv_file(&tensor_val, path)?;
                *last_node = Some(tensor_id);
            }
            noma_compiler::Statement::LoadSafetensors { name, path } => {
                // Load Safetensors file and create a tensor
                let tensors = noma_compiler::load_safetensors_file(path)?;
                if let Some((_, (data, shape))) = tensors.into_iter().next() {
                    let node_id = graph.add_constant_tensor(data, shape)?;
                    variables.insert(name.clone(), node_id);
                    *last_node = Some(node_id);
                } else {
                    return Err(format!("No tensors found in safetensors file: {}", path));
                }
            }
            noma_compiler::Statement::LoadSafetensorsNamed { name, path, tensor_name } => {
                // Load a specific named tensor from Safetensors file
                let tensors = noma_compiler::load_safetensors_file(path)?;
                let mut found = false;
                for (tname, (data, shape)) in tensors {
                    if tname == *tensor_name {
                        let node_id = graph.add_constant_tensor(data, shape)?;
                        variables.insert(name.clone(), node_id);
                        *last_node = Some(node_id);
                        found = true;
                        break;
                    }
                }
                if !found {
                    return Err(format!("Tensor '{}' not found in safetensors file: {}", tensor_name, path));
                }
            }
            noma_compiler::Statement::SaveSafetensors { tensors, path } => {
                // Evaluate all tensors and save to Safetensors format
                let mut tensor_map = Vec::new();
                for (tensor_name, tensor_expr) in tensors {
                    let tensor_id = graph.build_from_expression_with_functions(tensor_expr, variables, func_registry)?;
                    graph.forward_pass()?;
                    let tensor_val = graph.get_node(tensor_id)
                        .and_then(|n| n.value.clone())
                        .ok_or_else(|| format!("Cannot evaluate tensor '{}' for save_safetensors", tensor_name))?;
                    tensor_map.push((tensor_name.clone(), tensor_val));
                }
                noma_compiler::save_safetensors_file(&tensor_map, path)?;
            }
            noma_compiler::Statement::BatchLoop { item_name, index_name, data, batch_size, body } => {
                // Evaluate data tensor and batch size
                let data_id = graph.build_from_expression_with_functions(data, variables, func_registry)?;
                graph.forward_pass()?;
                let data_val = graph.get_node(data_id)
                    .and_then(|n| n.value.clone())
                    .ok_or_else(|| "Cannot evaluate data for batch loop".to_string())?;
                
                let batch_size_id = graph.build_from_expression_with_functions(batch_size, variables, func_registry)?;
                graph.forward_pass()?;
                let batch_size_val = graph.get_node(batch_size_id)
                    .and_then(|n| n.value.clone())
                    .and_then(|v| v.as_scalar())
                    .ok_or_else(|| "Batch size must be a scalar".to_string())? as usize;
                
                if batch_size_val == 0 {
                    return Err("Batch size cannot be zero".to_string());
                }
                
                // Get tensor data
                let (tensor_data, tensor_shape) = match &data_val {
                    noma_compiler::Value::Tensor(t) => (t.data.clone(), t.shape.clone()),
                    noma_compiler::Value::Scalar(s) => (vec![*s], vec![1]),
                };
                
                let num_samples = if tensor_shape.is_empty() { 1 } else { tensor_shape[0] };
                let num_batches = (num_samples + batch_size_val - 1) / batch_size_val;
                
                // Iterate over batches
                for batch_idx in 0..num_batches {
                    let start = batch_idx * batch_size_val;
                    let end = (start + batch_size_val).min(num_samples);
                    
                    // Extract batch data
                    let batch_data = if tensor_shape.len() == 1 {
                        tensor_data[start..end].to_vec()
                    } else {
                        let row_size: usize = tensor_shape[1..].iter().product();
                        tensor_data[start * row_size..end * row_size].to_vec()
                    };
                    
                    let batch_shape = if tensor_shape.len() == 1 {
                        vec![end - start]
                    } else {
                        let mut shape = vec![end - start];
                        shape.extend(&tensor_shape[1..]);
                        shape
                    };
                    
                    // Create batch tensor node
                    let batch_node_id = graph.add_constant_tensor(batch_data, batch_shape)?;
                    variables.insert(item_name.clone(), batch_node_id);
                    
                    // Optionally set index variable
                    if let Some(idx_name) = index_name {
                        let idx_node_id = graph.add_constant(batch_idx as f64);
                        variables.insert(idx_name.clone(), idx_node_id);
                    }
                    
                    // Execute batch body
                    lower_statements_shared(graph, variables, body, last_node, func_registry, optimizer_state)?;
                }
            }
            noma_compiler::Statement::EpochLoop { epochs, x_data, y_data, batch_size, x_batch_name, y_batch_name, body } => {
                // Evaluate epochs count
                let epochs_id = graph.build_from_expression_with_functions(epochs, variables, func_registry)?;
                graph.forward_pass()?;
                let num_epochs = graph.get_node(epochs_id)
                    .and_then(|n| n.value.clone())
                    .and_then(|v| v.as_scalar())
                    .ok_or_else(|| "Epoch count must be a scalar".to_string())? as usize;
                
                // Evaluate X and Y data tensors
                let x_data_id = graph.build_from_expression_with_functions(x_data, variables, func_registry)?;
                let y_data_id = graph.build_from_expression_with_functions(y_data, variables, func_registry)?;
                graph.forward_pass()?;
                
                let x_val = graph.get_node(x_data_id)
                    .and_then(|n| n.value.clone())
                    .ok_or_else(|| "Cannot evaluate X data for epoch loop".to_string())?;
                let y_val = graph.get_node(y_data_id)
                    .and_then(|n| n.value.clone())
                    .ok_or_else(|| "Cannot evaluate Y data for epoch loop".to_string())?;
                
                // Evaluate batch size
                let batch_size_id = graph.build_from_expression_with_functions(batch_size, variables, func_registry)?;
                graph.forward_pass()?;
                let batch_size_val = graph.get_node(batch_size_id)
                    .and_then(|n| n.value.clone())
                    .and_then(|v| v.as_scalar())
                    .ok_or_else(|| "Batch size must be a scalar".to_string())? as usize;
                
                if batch_size_val == 0 {
                    return Err("Batch size cannot be zero".to_string());
                }
                
                // Extract tensor data
                let (x_tensor_data, x_tensor_shape) = match &x_val {
                    noma_compiler::Value::Tensor(t) => (t.data.clone(), t.shape.clone()),
                    noma_compiler::Value::Scalar(s) => (vec![*s], vec![1]),
                };
                let (y_tensor_data, y_tensor_shape) = match &y_val {
                    noma_compiler::Value::Tensor(t) => (t.data.clone(), t.shape.clone()),
                    noma_compiler::Value::Scalar(s) => (vec![*s], vec![1]),
                };
                
                let num_samples = if x_tensor_shape.is_empty() { 1 } else { x_tensor_shape[0] };
                let num_batches = (num_samples + batch_size_val - 1) / batch_size_val;
                
                // Get optimizer config
                let (config, _) = pick_hyperparams(graph, variables, 0.01, 1);
                
                println!("Starting epoch training: {} epochs, {} samples, batch_size={}, {} batches/epoch", 
                         num_epochs, num_samples, batch_size_val, num_batches);
                
                // ============================================================
                // OPTIMIZATION: Build graph ONCE, reuse for all batches
                // ============================================================
                
                // Create placeholder batch nodes with first batch data
                let first_batch_size = batch_size_val.min(num_samples);
                let x_row_size: usize = if x_tensor_shape.len() > 1 { x_tensor_shape[1..].iter().product() } else { 1 };
                let y_row_size: usize = if y_tensor_shape.len() > 1 { y_tensor_shape[1..].iter().product() } else { 1 };
                
                let x_first_batch: Vec<f64> = x_tensor_data[0..first_batch_size * x_row_size].to_vec();
                let y_first_batch: Vec<f64> = y_tensor_data[0..first_batch_size * y_row_size].to_vec();
                
                let x_batch_shape_template = if x_tensor_shape.len() == 1 {
                    vec![first_batch_size]
                } else {
                    let mut shape = vec![first_batch_size];
                    shape.extend(&x_tensor_shape[1..]);
                    shape
                };
                let y_batch_shape_template = if y_tensor_shape.len() == 1 {
                    vec![first_batch_size]
                } else {
                    let mut shape = vec![first_batch_size];
                    shape.extend(&y_tensor_shape[1..]);
                    shape
                };
                
                // Create the batch nodes once
                let x_batch_node = graph.add_constant_tensor(x_first_batch, x_batch_shape_template.clone())?;
                let y_batch_node = graph.add_constant_tensor(y_first_batch, y_batch_shape_template.clone())?;
                variables.insert(x_batch_name.clone(), x_batch_node);
                variables.insert(y_batch_name.clone(), y_batch_node);
                
                // Build the graph ONCE by processing the body
                let mut body_last: Option<noma_compiler::NodeId> = None;
                let mut loss_node_id: Option<noma_compiler::NodeId> = None;
                
                for stmt in body {
                    match stmt {
                        noma_compiler::Statement::Minimize(expr) => {
                            let loss_id = graph.build_from_expression_with_functions(expr, variables, func_registry)?;
                            loss_node_id = Some(loss_id);
                            body_last = Some(loss_id);
                        }
                        _ => {
                            lower_statements_shared(graph, variables, &[stmt.clone()], &mut body_last, func_registry, optimizer_state)?;
                        }
                    }
                }
                
                let loss_id = loss_node_id.ok_or_else(|| "EpochLoop body must contain a minimize statement".to_string())?;
                
                // Epoch loop - now REUSING the graph
                for epoch in 0..num_epochs {
                    let mut epoch_loss = 0.0;
                    let mut batch_count = 0;
                    
                    // Batch loop
                    for batch_idx in 0..num_batches {
                        let start = batch_idx * batch_size_val;
                        let end = (start + batch_size_val).min(num_samples);
                        let actual_batch_size = end - start;
                        
                        // Skip last batch if size differs (to keep graph shape consistent)
                        if actual_batch_size != first_batch_size {
                            continue;
                        }
                        
                        // Extract X batch data
                        let x_batch_data: Vec<f64> = x_tensor_data[start * x_row_size..end * x_row_size].to_vec();
                        let y_batch_data: Vec<f64> = y_tensor_data[start * y_row_size..end * y_row_size].to_vec();
                        
                        // UPDATE node values in-place (no graph rebuild!)
                        graph.update_node_value(x_batch_node, x_batch_data, x_batch_shape_template.clone())?;
                        graph.update_node_value(y_batch_node, y_batch_data, y_batch_shape_template.clone())?;
                        
                        // Forward pass (recomputes all values)
                        graph.forward_pass()?;
                        
                        // Get loss value for logging
                        if let Some(node) = graph.get_node(loss_id) {
                            if let Some(noma_compiler::Value::Scalar(loss)) = &node.value {
                                epoch_loss += loss;
                                batch_count += 1;
                                
                                // Progress update every 10 batches
                                if batch_count % 10 == 0 {
                                    let progress = (batch_idx + 1) as f64 / num_batches as f64 * 100.0;
                                    let avg_loss_so_far = epoch_loss / batch_count as f64;
                                    eprintln!("  [{:>3.0}%] Epoch {}/{} batch {}/{} | loss: {:.6}", 
                                           progress, epoch + 1, num_epochs, batch_idx + 1, num_batches, avg_loss_so_far);
                                }
                            }
                        }
                        
                        // Backward pass
                        graph.backward_pass(loss_id)?;
                        
                        // Update weights using optimizer and reset gradients
                        graph.optimize_step_with_config(optimizer_state, &config)?;
                        graph.reset_gradients();
                    }
                    
                    // Print epoch summary
                    let avg_loss = if batch_count > 0 { epoch_loss / batch_count as f64 } else { 0.0 };
                    println!("\n✓ Epoch {}/{} complete: avg_loss = {:.6}", epoch + 1, num_epochs, avg_loss);
                }
                
                println!("Epoch training complete!");
            }
            noma_compiler::Statement::StreamingAdaptLoop { 
                x_data, y_data, batch_size, x_batch_name, y_batch_name, 
                predict_body, prediction_output, adapt_body 
            } => {
                // ============================================================
                // STREAMING TTA: Causal predict-then-adapt loop
                // For each batch:
                //   1. PREDICT with current model state (no gradients)
                //   2. Store predictions
                //   3. ADAPT using labels (compute gradients, update weights)
                // This ensures each prediction uses model state BEFORE seeing that batch
                // ============================================================
                
                // Evaluate X and Y data tensors
                let x_data_id = graph.build_from_expression_with_functions(x_data, variables, func_registry)?;
                let y_data_id = graph.build_from_expression_with_functions(y_data, variables, func_registry)?;
                graph.forward_pass()?;
                
                let x_val = graph.get_node(x_data_id)
                    .and_then(|n| n.value.clone())
                    .ok_or_else(|| "Cannot evaluate X data for streaming_adapt loop".to_string())?;
                let y_val = graph.get_node(y_data_id)
                    .and_then(|n| n.value.clone())
                    .ok_or_else(|| "Cannot evaluate Y data for streaming_adapt loop".to_string())?;
                
                // Evaluate batch size
                let batch_size_id = graph.build_from_expression_with_functions(batch_size, variables, func_registry)?;
                graph.forward_pass()?;
                let batch_size_val = graph.get_node(batch_size_id)
                    .and_then(|n| n.value.clone())
                    .and_then(|v| v.as_scalar())
                    .ok_or_else(|| "Batch size must be a scalar".to_string())? as usize;
                
                if batch_size_val == 0 {
                    return Err("Batch size cannot be zero".to_string());
                }
                
                // Extract tensor data
                let (x_tensor_data, x_tensor_shape) = match &x_val {
                    noma_compiler::Value::Tensor(t) => (t.data.clone(), t.shape.clone()),
                    noma_compiler::Value::Scalar(s) => (vec![*s], vec![1]),
                };
                let (y_tensor_data, y_tensor_shape) = match &y_val {
                    noma_compiler::Value::Tensor(t) => (t.data.clone(), t.shape.clone()),
                    noma_compiler::Value::Scalar(s) => (vec![*s], vec![1]),
                };
                
                let num_samples = if x_tensor_shape.is_empty() { 1 } else { x_tensor_shape[0] };
                let num_batches = (num_samples + batch_size_val - 1) / batch_size_val;
                
                // Get optimizer config
                let (config, _) = pick_hyperparams(graph, variables, 0.01, 1);
                
                // Compute row sizes
                let x_row_size: usize = if x_tensor_shape.len() > 1 { x_tensor_shape[1..].iter().product() } else { 1 };
                let y_row_size: usize = if y_tensor_shape.len() > 1 { y_tensor_shape[1..].iter().product() } else { 1 };
                
                println!("Starting streaming TTA: {} samples, batch_size={}, {} batches", 
                         num_samples, batch_size_val, num_batches);
                println!("Mode: CAUSAL (predict BEFORE adapt on each batch)");
                
                // First batch to set up graph structure
                let first_batch_size = batch_size_val.min(num_samples);
                let x_first_batch: Vec<f64> = x_tensor_data[0..first_batch_size * x_row_size].to_vec();
                let y_first_batch: Vec<f64> = y_tensor_data[0..first_batch_size * y_row_size].to_vec();
                
                let x_batch_shape_template = if x_tensor_shape.len() == 1 {
                    vec![first_batch_size]
                } else {
                    let mut shape = vec![first_batch_size];
                    shape.extend(&x_tensor_shape[1..]);
                    shape
                };
                let y_batch_shape_template = if y_tensor_shape.len() == 1 {
                    vec![first_batch_size]
                } else {
                    let mut shape = vec![first_batch_size];
                    shape.extend(&y_tensor_shape[1..]);
                    shape
                };
                
                // Create batch nodes
                let x_batch_node = graph.add_constant_tensor(x_first_batch, x_batch_shape_template.clone())?;
                let y_batch_node = graph.add_constant_tensor(y_first_batch, y_batch_shape_template.clone())?;
                variables.insert(x_batch_name.clone(), x_batch_node);
                variables.insert(y_batch_name.clone(), y_batch_node);
                
                // ============================================================
                // Build PREDICT graph (just forward pass, store predictions)
                // ============================================================
                let mut predict_last: Option<noma_compiler::NodeId> = None;
                let mut predict_output_node: Option<noma_compiler::NodeId> = None;
                
                for stmt in predict_body {
                    match stmt {
                        noma_compiler::Statement::LetDeclaration { name, value } => {
                            let node_id = graph.build_from_expression_with_functions(value, variables, func_registry)?;
                            variables.insert(name.clone(), node_id);
                            predict_last = Some(node_id);
                            predict_output_node = Some(node_id);
                        }
                        _ => {
                            lower_statements_shared(graph, variables, &[stmt.clone()], &mut predict_last, func_registry, optimizer_state)?;
                        }
                    }
                }
                
                let pred_node = predict_output_node.ok_or_else(|| 
                    "streaming_adapt predict block must contain at least one let statement for output".to_string())?;
                
                // ============================================================
                // Build ADAPT graph (forward + backward + optimize)
                // ============================================================
                let mut adapt_last: Option<noma_compiler::NodeId> = None;
                let mut loss_node_id: Option<noma_compiler::NodeId> = None;
                
                for stmt in adapt_body {
                    match stmt {
                        noma_compiler::Statement::Minimize(expr) => {
                            let loss_id = graph.build_from_expression_with_functions(expr, variables, func_registry)?;
                            loss_node_id = Some(loss_id);
                            adapt_last = Some(loss_id);
                        }
                        _ => {
                            lower_statements_shared(graph, variables, &[stmt.clone()], &mut adapt_last, func_registry, optimizer_state)?;
                        }
                    }
                }
                
                let loss_id = loss_node_id.ok_or_else(|| 
                    "streaming_adapt adapt block must contain a minimize statement".to_string())?;
                
                // Storage for all predictions
                let mut all_predictions: Vec<f64> = Vec::new();
                let mut total_loss = 0.0;
                let mut batch_count = 0;
                
                // Suppress print output during the streaming loop
                graph.set_suppress_print(true);
                
                // ============================================================
                // Main streaming loop: PREDICT then ADAPT for each batch
                // ============================================================
                for batch_idx in 0..num_batches {
                    let start = batch_idx * batch_size_val;
                    let end = (start + batch_size_val).min(num_samples);
                    let actual_batch_size = end - start;
                    
                    // Skip if batch size differs
                    if actual_batch_size != first_batch_size {
                        continue;
                    }
                    
                    // Extract batch data
                    let x_batch_data: Vec<f64> = x_tensor_data[start * x_row_size..end * x_row_size].to_vec();
                    let y_batch_data: Vec<f64> = y_tensor_data[start * y_row_size..end * y_row_size].to_vec();
                    
                    // Update batch values
                    graph.update_node_value(x_batch_node, x_batch_data, x_batch_shape_template.clone())?;
                    graph.update_node_value(y_batch_node, y_batch_data, y_batch_shape_template.clone())?;
                    
                    // ============================================================
                    // STEP 1: PREDICT (forward only, store predictions)
                    // ============================================================
                    graph.forward_pass()?;
                    
                    // Store predictions from this batch
                    if let Some(node) = graph.get_node(pred_node) {
                        if let Some(val) = &node.value {
                            match val {
                                noma_compiler::Value::Tensor(t) => {
                                    all_predictions.extend(&t.data);
                                }
                                noma_compiler::Value::Scalar(s) => {
                                    all_predictions.push(*s);
                                }
                            }
                        }
                    }
                    
                    // ============================================================
                    // STEP 2: ADAPT (backward + update weights)
                    // ============================================================
                    // Re-run forward to ensure loss is computed
                    graph.forward_pass()?;
                    
                    // Get loss for logging
                    if let Some(node) = graph.get_node(loss_id) {
                        if let Some(noma_compiler::Value::Scalar(loss)) = &node.value {
                            total_loss += loss;
                            batch_count += 1;
                        }
                    }
                    
                    // Backward and update
                    graph.backward_pass(loss_id)?;
                    graph.optimize_step_with_config(optimizer_state, &config)?;
                    graph.reset_gradients();
                    
                    // Progress update
                    if batch_idx % 10 == 0 {
                        let progress = (batch_idx + 1) as f64 / num_batches as f64 * 100.0;
                        eprintln!("  [{:>5.1}%] batch {}/{} | avg_loss: {:.6}", 
                               progress, batch_idx + 1, num_batches, 
                               if batch_count > 0 { total_loss / batch_count as f64 } else { 0.0 });
                    }
                }
                
                // Re-enable print output after loop
                graph.set_suppress_print(false);
                
                // Store accumulated predictions as output tensor
                let pred_shape = if y_tensor_shape.len() > 1 {
                    let mut shape = vec![batch_count * first_batch_size];
                    shape.extend(&y_tensor_shape[1..]);
                    shape
                } else {
                    vec![all_predictions.len()]
                };
                
                let output_node = graph.add_constant_tensor(all_predictions.clone(), pred_shape)?;
                variables.insert(prediction_output.clone(), output_node);
                
                let avg_loss = if batch_count > 0 { total_loss / batch_count as f64 } else { 0.0 };
                println!("\n✓ Streaming TTA complete: {} batches processed, avg_loss = {:.6}", batch_count, avg_loss);
                println!("  Predictions stored in '{}'", prediction_output);
            }
            noma_compiler::Statement::ResetOptimizer => {
                // Clear all optimizer state (m, v, t) to restart from scratch
                optimizer_state.reset();
            }
        }
    }
    Ok(())
}

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

        /// Additional libraries to link (passed as -l<name>)
        #[arg(long = "link-lib", value_name = "LIB", num_args = 1.., action = clap::ArgAction::Append)]
        link_libs: Vec<String>,

        /// Additional library search paths (passed as -L<path>)
        #[arg(long = "link-path", value_name = "PATH", num_args = 1.., action = clap::ArgAction::Append)]
        link_paths: Vec<String>,
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

    /// Run a NOMA source file by interpreting the graph and printing the return value
    Run {
        /// Input .noma file
        #[arg(value_name = "FILE")]
        file: PathBuf,
    },

    /// Compile and run a NOMA source file (faster execution, no training support)
    FastRun {
        /// Input .noma file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Optimization level (0,1,2,3). Defaults to 2.
        #[arg(short = 'O', long = "opt-level", value_parser = clap::value_parser!(u8).range(0..=3))]
        opt_level: Option<u8>,

        /// Enable fast-math optimizations
        #[arg(long = "fast-math")]
        fast_math: bool,
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
        Commands::BuildExe { file, output, opt_level, fast_math, link_libs, link_paths } => {
            build_executable(file, output, opt_level, fast_math, link_libs, link_paths)?;
        }
        Commands::CompilePtx { file, output, n_elems, host_stub, optimize, fast_math } => {
            compile_to_ptx(file, output, n_elems, host_stub, optimize, fast_math)?;
        }
        Commands::Check { file } => {
            check_file(file)?;
        }
        Commands::Run { file } => {
            run_noma(file)?;
        }
        Commands::FastRun { file, opt_level, fast_math } => {
            fast_run_noma(file, opt_level, fast_math)?;
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
    config: OptimizerConfig,
    max_iter: usize,
    optimizer_state: &mut OptimizerState,
) -> Result<(), String> {
    if !variables.contains_key(target) {
        return Err(format!("Optimize target '{}' not defined", target));
    }

    // Use shared optimizer state to preserve momentum across optimize loops
    // This enables faster convergence when architecture changes (realloc)

    for _ in 0..max_iter {
        graph.forward_pass()?;
        let cond_val = graph.get_node(cond_id).and_then(|n| n.value.clone()).and_then(|v| match v { noma_compiler::Value::Scalar(s) => Some(s), _ => None }).unwrap_or(0.0);
        if cond_val != 0.0 {
            return Ok(());
        }

        graph.backward_pass(objective_id)?;
        graph.optimize_step_with_config(optimizer_state, &config)?;
        graph.reset_gradients();
    }

    // Log warning but continue (non-convergence is allowed; program continues)
    eprintln!("Warning: Optimize loop reached max iterations without satisfying condition");
    Ok(())
}

/// Pick hyperparameters and optimizer configuration from NOMA variables if present.
/// Recognized names:
///  - optimizer: "optimizer" (values: "sgd", "adam", "rmsprop")
///  - learning rate: "learning_rate", "lr"
///  - max iterations: "max_iterations", "max_iter", "iterations"
///  - Adam/RMSprop beta1: "beta1" (default 0.9)
///  - Adam/RMSprop beta2: "beta2" (default 0.999 for Adam, 0.9 for RMSprop)
///  - epsilon: "epsilon", "eps" (default 1e-8)
/// If not found or invalid, fall back to provided defaults.
fn pick_hyperparams(
    graph: &mut ComputationalGraph,
    variables: &HashMap<String, noma_compiler::NodeId>,
    default_lr: f64,
    default_iters: usize,
) -> (OptimizerConfig, usize) {
    // Try a forward pass so constants/expressions have values; ignore errors.
    let _ = graph.forward_pass();

    // Helper to read a scalar value from a variable name
    let read_scalar = |name: &str| -> Option<f64> {
        variables.get(name)
            .and_then(|nid| graph.get_node(*nid))
            .and_then(|n| n.value.clone())
            .and_then(|v| match v { noma_compiler::Value::Scalar(s) => Some(s), _ => None })
    };

    // Determine optimizer type
    // We use a convention: optimizer = 1.0 for SGD, 2.0 for Adam, 3.0 for RMSprop
    // Or check variable names: use_adam, use_rmsprop as flags
    let optimizer_type = if let Some(opt_val) = read_scalar("optimizer") {
        match opt_val as i32 {
            2 => OptimizerType::Adam,
            3 => OptimizerType::RMSprop,
            _ => OptimizerType::SGD,
        }
    } else if read_scalar("use_adam").map(|v| v != 0.0).unwrap_or(false) {
        OptimizerType::Adam
    } else if read_scalar("use_rmsprop").map(|v| v != 0.0).unwrap_or(false) {
        OptimizerType::RMSprop
    } else {
        OptimizerType::SGD
    };

    // learning rate
    let lr = read_scalar("learning_rate")
        .or_else(|| read_scalar("lr"))
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(default_lr);

    // Adam/RMSprop hyperparameters
    let beta1 = read_scalar("beta1")
        .filter(|v| v.is_finite() && *v >= 0.0 && *v < 1.0)
        .unwrap_or(0.9);

    let default_beta2 = match optimizer_type {
        OptimizerType::Adam => 0.999,
        OptimizerType::RMSprop => 0.9,
        OptimizerType::SGD => 0.999,
    };
    let beta2 = read_scalar("beta2")
        .filter(|v| v.is_finite() && *v >= 0.0 && *v < 1.0)
        .unwrap_or(default_beta2);

    let epsilon = read_scalar("epsilon")
        .or_else(|| read_scalar("eps"))
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(1e-8);

    // iterations
    let iters_f = read_scalar("max_iterations")
        .or_else(|| read_scalar("max_iter"))
        .or_else(|| read_scalar("iterations"))
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(default_iters as f64);

    let mut iters = iters_f.round() as isize;
    if iters < 1 { iters = 1; }
    // Cap very large values to avoid runaway compile-time loops
    let iters = (iters as usize).min(10_000_000);

    let config = OptimizerConfig {
        optimizer_type,
        learning_rate: lr,
        beta1,
        beta2,
        epsilon,
    };

    (config, iters)
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

    // Collect user-defined functions
    let (func_registry, main_func) = collect_functions(&ast);
    let func = main_func.ok_or_else(|| anyhow::anyhow!("No function found to compile"))?;

    // Lower the first function (main) to a computational graph
    let mut graph = ComputationalGraph::new();
    let mut variables: HashMap<String, noma_compiler::NodeId> = HashMap::new();
    let mut last_node: Option<noma_compiler::NodeId> = None;
    let mut optimizer_state = OptimizerState::new();

    lower_statements_shared(&mut graph, &mut variables, &func.body, &mut last_node, &func_registry, &mut optimizer_state)
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

fn run_noma(file: PathBuf) -> anyhow::Result<()> {
    println!("Running: {}", file.display());

    let source = std::fs::read_to_string(&file)?;
    let mut lexer = Lexer::new(&source);
    let tokens = lexer.tokenize().map_err(|e| anyhow::anyhow!("{:?}", e))?;

    let mut parser = NomaParser::new(tokens);
    let ast = parser.parse().map_err(|e| anyhow::anyhow!("{:?}", e))?;

    // Collect all user-defined functions into a registry
    let (func_registry, main_func) = collect_functions(&ast);
    let func = main_func.ok_or_else(|| anyhow::anyhow!("No function found to run"))?;

    // Lower first function to graph
    let mut graph = ComputationalGraph::new();
    let mut variables: HashMap<String, noma_compiler::NodeId> = HashMap::new();
    let mut last_node: Option<noma_compiler::NodeId> = None;
    let mut optimizer_state = OptimizerState::new();

    lower_statements_shared(&mut graph, &mut variables, &func.body, &mut last_node, &func_registry, &mut optimizer_state)
        .map_err(|e| anyhow::anyhow!(e))?;

    graph.forward_pass().map_err(|e| anyhow::anyhow!(e))?;
    let out_node = last_node.ok_or_else(|| anyhow::anyhow!("No value to return"))?;
    let val = graph.get_node(out_node).and_then(|n| n.value.clone()).ok_or_else(|| anyhow::anyhow!("No value computed"))?;
    match val {
        noma_compiler::Value::Scalar(s) => println!("Result: {}", s),
        noma_compiler::Value::Tensor(t) => println!("Result tensor {:?}: {:?}", t.shape, t.data),
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

    // Collect user-defined functions
    let (func_registry, main_func) = collect_functions(&ast);
    let func = main_func.ok_or_else(|| anyhow::anyhow!("No function found to compile"))?;

    // Lower first function to graph
    let mut graph = ComputationalGraph::new();
    let mut variables: HashMap<String, noma_compiler::NodeId> = HashMap::new();
    let mut last_node: Option<noma_compiler::NodeId> = None;
    let mut optimizer_state = OptimizerState::new();

    lower_statements_shared(&mut graph, &mut variables, &func.body, &mut last_node, &func_registry, &mut optimizer_state)
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

/// Fast-run: compile to native and execute (for programs without training loops)
fn fast_run_noma(file: PathBuf, opt_level: Option<u8>, fast_math: bool) -> anyhow::Result<()> {
    use std::time::Instant;
    
    let start = Instant::now();
    
    // Create temporary executable
    let tmp_dir = env::temp_dir();
    let exe_path = tmp_dir.join("noma_fast_run");
    
    // Build the executable (quietly)
    let source = fs::read_to_string(&file)?;
    let mut lexer = Lexer::new(&source);
    let tokens = lexer.tokenize().map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let mut parser = NomaParser::new(tokens);
    let ast = parser.parse().map_err(|e| anyhow::anyhow!("{:?}", e))?;
    
    let (func_registry, main_func) = collect_functions(&ast);
    let func = main_func.ok_or_else(|| anyhow::anyhow!("No function found to compile"))?;
    
    let mut graph = ComputationalGraph::new();
    let mut variables: HashMap<String, noma_compiler::NodeId> = HashMap::new();
    let mut last_node: Option<noma_compiler::NodeId> = None;
    let mut optimizer_state = OptimizerState::new();
    
    lower_statements_shared(&mut graph, &mut variables, &func.body, &mut last_node, &func_registry, &mut optimizer_state)
        .map_err(|e| anyhow::anyhow!(e))?;
    
    let _ = graph.forward_pass();
    
    let mut codegen = LLVMCodegen::new().with_fast_math(fast_math);
    let compute_ir = codegen.generate_with_return(&graph, last_node).map_err(|e| anyhow::anyhow!(e))?;
    
    let wrapped_ir = format!(
        "{}\n\ndefine i32 @main() {{\nentry:\n  %result = call double @compute()\n  %resultptr = alloca double\n  store double %result, double* %resultptr\n  %fmt = getelementptr [4 x i8], [4 x i8]* @.str, i32 0, i32 0\n  call i32 (i8*, ...) @printf(i8* %fmt, double %result)\n  ret i32 0\n}}\n",
        compute_ir
    );
    
    let ir_path = tmp_dir.join("noma_fast_run.ll");
    fs::write(&ir_path, &wrapped_ir)?;
    
    // Optimize
    let opt_level_val = opt_level.unwrap_or(2);
    if opt_level_val > 0 {
        if let Ok(_) = Command::new("opt").arg("--version").output() {
            let opt_output = tmp_dir.join("noma_fast_run_opt.ll");
            let opt_flag = format!("-O{}", opt_level_val);
            if let Ok(s) = Command::new("opt")
                .arg("-S").arg(&opt_flag).arg(&ir_path).arg("-o").arg(&opt_output)
                .status() {
                if s.success() {
                    fs::copy(&opt_output, &ir_path)?;
                }
            }
        }
    }
    
    // Compile to object
    let obj_path = tmp_dir.join("noma_fast_run.o");
    if let Ok(_) = Command::new("llc").arg("--version").output() {
        let status = Command::new("llc")
            .arg("-filetype=obj").arg(&ir_path).arg("-o").arg(&obj_path)
            .status();
        if status.is_err() || !status.unwrap().success() {
            return Err(anyhow::anyhow!("llc failed"));
        }
    } else {
        return Err(anyhow::anyhow!("llc not found"));
    }
    
    // Link
    let link_ok = Command::new("gcc")
        .arg(&obj_path).arg("-lm").arg("-no-pie").arg("-o").arg(&exe_path)
        .status().map(|s| s.success()).unwrap_or(false)
        || Command::new("clang")
        .arg(&obj_path).arg("-lm").arg("-no-pie").arg("-o").arg(&exe_path)
        .status().map(|s| s.success()).unwrap_or(false);
    
    if !link_ok {
        return Err(anyhow::anyhow!("Linking failed"));
    }
    
    let compile_time = start.elapsed();
    
    // Execute
    let exec_start = Instant::now();
    let output = Command::new(&exe_path).output()?;
    let exec_time = exec_start.elapsed();
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    print!("{}", stdout);
    
    // Cleanup
    let _ = fs::remove_file(&ir_path);
    let _ = fs::remove_file(&obj_path);
    let _ = fs::remove_file(&exe_path);
    
    eprintln!("[fast-run] Compiled in {:?}, executed in {:?}", compile_time, exec_time);
    
    Ok(())
}

fn build_executable(file: PathBuf, output: PathBuf, opt_level: Option<u8>, fast_math: bool, link_libs: Vec<String>, link_paths: Vec<String>) -> anyhow::Result<()> {
    println!("Building executable: {} -> {}", file.display(), output.display());
    
    // Read source file
    let source = fs::read_to_string(&file)?;

    // Tokenize and parse
    let mut lexer = Lexer::new(&source);
    let tokens = lexer.tokenize().map_err(|e| anyhow::anyhow!("{:?}", e))?;
    
    let mut parser = NomaParser::new(tokens);
    let ast = parser.parse().map_err(|e| anyhow::anyhow!("{:?}", e))?;

    // Collect user-defined functions
    let (func_registry, main_func) = collect_functions(&ast);
    let func = main_func.ok_or_else(|| anyhow::anyhow!("No function found to compile"))?;

    // Lower the first function to a computational graph
    let mut graph = ComputationalGraph::new();
    let mut variables: HashMap<String, noma_compiler::NodeId> = HashMap::new();
    let mut last_node: Option<noma_compiler::NodeId> = None;
    let mut optimizer_state = OptimizerState::new();

    lower_statements_shared(&mut graph, &mut variables, &func.body, &mut last_node, &func_registry, &mut optimizer_state)
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
        let mut cmd = Command::new(linker);
        cmd.arg(&obj_path);

        // Library search paths
        for p in &link_paths {
            cmd.arg(format!("-L{}", p));
        }

        // Math library by default
        cmd.arg("-lm");

        // User-specified libraries
        for lib in &link_libs {
            cmd.arg(format!("-l{}", lib));
        }

        cmd.arg("-no-pie");
        cmd.arg("-o");
        cmd.arg(&output);

        match cmd.status() {
            Ok(s) if s.success() => {
                println!("[info] Linked executable with {}", linker);
                println!("Built standalone executable: {}", output.display());
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
