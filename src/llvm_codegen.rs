use crate::graph::{ComputationalGraph, NodeId, NodeType, Value};
use std::collections::HashMap;

/// LLVM IR code generator
/// Converts a computational graph to LLVM Intermediate Representation
pub struct LLVMCodegen {
    counter: usize,
    fast_math: bool,
}

impl LLVMCodegen {
    pub fn new() -> Self {
        Self { counter: 0, fast_math: false }
    }

    pub fn with_fast_math(mut self, enabled: bool) -> Self {
        self.fast_math = enabled;
        self
    }

    fn fmt_f64(&self, v: f64) -> String {
        format!("{:.16e}", v)
    }

    fn fresh_var(&mut self) -> String {
        let var = format!("%{}", self.counter);
        self.counter += 1;
        var
    }

    /// Generate LLVM IR for a computational graph, returning the value of a specific node
    pub fn generate_with_return(&mut self, graph: &ComputationalGraph, return_node: Option<NodeId>) -> Result<String, String> {
        self.generate_internal(graph, return_node)
    }

    /// Generate LLVM IR for a computational graph
    pub fn generate(&mut self, graph: &ComputationalGraph) -> Result<String, String> {
        self.generate_internal(graph, None)
    }
    
    fn generate_internal(&mut self, graph: &ComputationalGraph, return_node: Option<NodeId>) -> Result<String, String> {
        let mut ir = String::new();

        // LLVM module header
        ir.push_str("; NOMA LLVM IR Generated Code\n");
        ir.push_str("source_filename = \"noma_generated\"\n");
        ir.push_str("target datalayout = \"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128\"\n");
        ir.push_str("target triple = \"x86_64-unknown-linux-gnu\"\n\n");

        // Declare external functions (printf for debugging)
        ir.push_str("declare i32 @printf(i8*, ...)\n");
        ir.push_str("@.str = private unnamed_addr constant [4 x i8] c\"%f\\0A\\00\", align 1\n\n");

        // Generate compute function
        ir.push_str("define double @compute() {\nentry:\n");

        let mut var_map = HashMap::new();
        let nodes = graph.nodes();

        // Sort nodes by their numeric id for deterministic order
        let mut node_ids: Vec<NodeId> = nodes.keys().copied().collect();
        node_ids.sort_by_key(|id| id.index());

        let mut last_var: Option<String> = None;

        // Process nodes in deterministic order
        for node_id in node_ids {
            let node = &nodes[&node_id];
            
            // For operations that create temporaries, we'll allocate var later
            let needs_deferred_var = match &node.node_type {
                NodeType::BinaryOp(op) => matches!(op.as_str(), "eq" | "ne" | "lt" | "gt" | "le" | "ge" | "and" | "or"),
                NodeType::FunctionCall(f) => f == "sigmoid",
                _ => false,
            };
            
            let var = if needs_deferred_var {
                String::new() // Placeholder, will be set later
            } else {
                let v = self.fresh_var();
                var_map.insert(node_id, v.clone());
                last_var = Some(v.clone());
                v
            };

            match &node.node_type {
                NodeType::Constant(Value::Scalar(val)) => {
                    ir.push_str(&format!("  {} = fadd double {}, 0.0\n", var, self.fmt_f64(*val)));
                }
                NodeType::Constant(Value::Tensor(_)) => {
                    return Err("Tensor values are not yet supported in LLVM codegen".to_string());
                }
                NodeType::Learnable(_) => {
                    let val = match node.value.clone() {
                        Some(Value::Scalar(v)) => v,
                        Some(Value::Tensor(_)) => return Err("Tensor values are not yet supported in LLVM codegen".to_string()),
                        None => 0.0,
                    };
                    ir.push_str(&format!("  {} = fadd double {}, 0.0\n", var, self.fmt_f64(val)));
                }
                NodeType::Variable(_) => {
                    let val = match node.value.clone() {
                        Some(Value::Scalar(v)) => v,
                        Some(Value::Tensor(_)) => return Err("Tensor values are not yet supported in LLVM codegen".to_string()),
                        None => 0.0,
                    };
                    ir.push_str(&format!("  {} = fadd double {}, 0.0\n", var, self.fmt_f64(val)));
                }
                NodeType::BinaryOp(op_str) => {
                    if node.inputs.len() != 2 {
                        return Err("Binary operation requires 2 inputs".to_string());
                    }
                    let left_var = var_map.get(&node.inputs[0]).ok_or("Left operand not found")?;
                    let right_var = var_map.get(&node.inputs[1]).ok_or("Right operand not found")?;

                    let fmf = if self.fast_math { " fast" } else { "" };
                    match op_str.as_str() {
                        "add" => {
                            ir.push_str(&format!("  {} = fadd{} double {}, {}\n", var, fmf, left_var, right_var));
                        }
                        "sub" => {
                            ir.push_str(&format!("  {} = fsub{} double {}, {}\n", var, fmf, left_var, right_var));
                        }
                        "mul" => {
                            ir.push_str(&format!("  {} = fmul{} double {}, {}\n", var, fmf, left_var, right_var));
                        }
                        "div" => {
                            ir.push_str(&format!("  {} = fdiv{} double {}, {}\n", var, fmf, left_var, right_var));
                        }
                        "mod" => {
                            ir.push_str(&format!("  {} = frem double {}, {}\n", var, left_var, right_var));
                        }
                        "pow" => {
                            ir.push_str(&format!(
                                "  {} = call double @llvm.pow.f64(double {}, double {})\n",
                                var, left_var, right_var
                            ));
                        }
                        "eq" | "ne" | "lt" | "gt" | "le" | "ge" => {
                            let pred = match op_str.as_str() {
                                "eq" => "oeq",
                                "ne" => "one",
                                "lt" => "olt",
                                "gt" => "ogt",
                                "le" => "ole",
                                "ge" => "oge",
                                _ => unreachable!(),
                            };
                            let cmp = self.fresh_var();
                            ir.push_str(&format!("  {} = fcmp {} double {}, {}\n", cmp, pred, left_var, right_var));
                            let result_var = self.fresh_var();
                            ir.push_str(&format!("  {} = uitofp i1 {} to double\n", result_var, cmp));
                            var_map.insert(node_id, result_var.clone());
                            last_var = Some(result_var);
                        }
                        "and" => {
                            let l0 = self.fresh_var();
                            ir.push_str(&format!("  {} = fcmp one double {}, 0.0\n", l0, left_var));
                            let r0 = self.fresh_var();
                            ir.push_str(&format!("  {} = fcmp one double {}, 0.0\n", r0, right_var));
                            let pred = self.fresh_var();
                            ir.push_str(&format!("  {} = and i1 {}, {}\n", pred, l0, r0));
                            let result_var = self.fresh_var();
                            ir.push_str(&format!("  {} = uitofp i1 {} to double\n", result_var, pred));
                            var_map.insert(node_id, result_var.clone());
                            last_var = Some(result_var);
                        }
                        "or" => {
                            let l0 = self.fresh_var();
                            ir.push_str(&format!("  {} = fcmp one double {}, 0.0\n", l0, left_var));
                            let r0 = self.fresh_var();
                            ir.push_str(&format!("  {} = fcmp one double {}, 0.0\n", r0, right_var));
                            let pred = self.fresh_var();
                            ir.push_str(&format!("  {} = or i1 {}, {}\n", pred, l0, r0));
                            let result_var = self.fresh_var();
                            ir.push_str(&format!("  {} = uitofp i1 {} to double\n", result_var, pred));
                            var_map.insert(node_id, result_var.clone());
                            last_var = Some(result_var);
                        }
                        _ => {
                            return Err(format!("Unsupported binary operator: {}", op_str));
                        }
                    }
                }
                NodeType::UnaryOp(op_str) => {
                    if node.inputs.len() != 1 {
                        return Err("Unary operation requires 1 input".to_string());
                    }
                    let arg_var = var_map.get(&node.inputs[0]).ok_or("Argument not found")?;

                    match op_str.as_str() {
                        "neg" => {
                            ir.push_str(&format!("  {} = fsub double 0.0, {}\n", var, arg_var));
                        }
                        "not" => {
                            return Err("NOT operator not supported in numeric code generation".to_string());
                        }
                        _ => {
                            return Err(format!("Unsupported unary operator: {}", op_str));
                        }
                    }
                }
                NodeType::FunctionCall(func_name) => {
                    match func_name.as_str() {
                        "sigmoid" => {
                            if node.inputs.len() != 1 {
                                return Err("sigmoid expects 1 argument".to_string());
                            }
                            let arg_var = var_map.get(&node.inputs[0]).ok_or("Argument not found")?;
                            let neg_var = self.fresh_var();
                            ir.push_str(&format!("  {} = fsub double 0.0, {}\n", neg_var, arg_var));
                            let exp_var = self.fresh_var();
                            ir.push_str(&format!("  {} = call double @llvm.exp.f64(double {})\n", exp_var, neg_var));
                            let one_add = self.fresh_var();
                            ir.push_str(&format!("  {} = fadd double 1.0, {}\n", one_add, exp_var));
                            // Use a fresh var for the result (allocated after temporaries)
                            let result_var = self.fresh_var();
                            ir.push_str(&format!("  {} = fdiv double 1.0, {}\n", result_var, one_add));
                            // Update var_map with the actual result variable
                            var_map.insert(node_id, result_var.clone());
                            last_var = Some(result_var);
                        }
                        "relu" => {
                            if node.inputs.len() != 1 {
                                return Err("relu expects 1 argument".to_string());
                            }
                            let arg_var = var_map.get(&node.inputs[0]).ok_or("Argument not found")?;
                            ir.push_str(&format!("  {} = call double @llvm.maxnum.f64(double {}, double 0.0)\n", var, arg_var));
                        }
                        _ => {
                            return Err(format!("Unsupported function: {}", func_name));
                        }
                    }
                }
            }
        }

        // Return the specified node, or the last computed value if not specified
        let ret_var = if let Some(ret_node) = return_node {
            var_map.get(&ret_node).cloned()
        } else {
            last_var
        };
        
        if let Some(ret_var) = ret_var {
            ir.push_str(&format!("  ret double {}\n", ret_var));
        } else {
            ir.push_str("  ret double 0.0\n");
        }

        ir.push_str("}\n\n");

        // Helper: declare LLVM intrinsics
        ir.push_str("declare double @llvm.pow.f64(double, double)\n");
        ir.push_str("declare double @llvm.exp.f64(double)\n");
        ir.push_str("declare double @llvm.maxnum.f64(double, double)\n");

        Ok(ir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llvm_constant() {
        let graph = ComputationalGraph::new();
        let mut codegen = LLVMCodegen::new();
        let ir = codegen.generate(&graph).expect("IR generation failed");
        
        assert!(ir.contains("source_filename = \"noma_generated\""));
        assert!(ir.contains("define double @compute()"));
        assert!(ir.contains("ret double"));
    }

    #[test]
    fn test_llvm_mod_and_or() {
        let mut graph = ComputationalGraph::new();
        let a = graph.add_constant(5.0);
        let b = graph.add_constant(2.0);
        let _m = graph.add_binary_op("mod", a, b);
        let _o = graph.add_binary_op("or", a, b);
        let _a = graph.add_binary_op("and", a, b);

        let mut codegen = LLVMCodegen::new();
        let ir = codegen.generate(&graph).expect("IR generation failed");

        assert!(ir.contains("frem double"));
        assert!(ir.contains("and i1"));
        assert!(ir.contains("or i1"));
    }
}

