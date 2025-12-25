use std::collections::HashMap;
use crate::ast::{BinaryOperator, Expression, UnaryOperator};

/// A unique identifier for a node in the computational graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

impl NodeId {
    pub fn new(id: usize) -> Self {
        NodeId(id)
    }
}

/// Represents a node in the computational graph
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub id: NodeId,
    pub node_type: NodeType,
    pub inputs: Vec<NodeId>, // Input dependencies
    pub value: Option<f64>,  // Current value
    pub gradient: Option<f64>, // Gradient for backprop
}

/// Represents an operation/node in the computational graph
#[derive(Debug, Clone, PartialEq)]
pub enum NodeType {
    /// Constant value
    Constant(f64),
    /// Learnable variable (declared with 'learn')
    Learnable(String),
    /// Regular variable (declared with 'let')
    Variable(String),
    /// Binary operation: add, sub, mul, div, pow, eq, ne, lt, gt, le, ge
    BinaryOp(String),
    /// Unary operation: neg, not
    UnaryOp(String),
    /// Function call: sigmoid, relu, dot, mse, etc.
    FunctionCall(String),
}

/// The computational graph - directed acyclic graph (DAG)
#[derive(Debug, Clone)]
pub struct ComputationalGraph {
    nodes: HashMap<NodeId, Node>,
    next_id: usize,
    learnables: Vec<String>, // Track learnable variables for gradients
}

impl ComputationalGraph {
    pub fn new() -> Self {
        ComputationalGraph {
            nodes: HashMap::new(),
            next_id: 0,
            learnables: Vec::new(),
        }
    }

    /// Add a constant node
    pub fn add_constant(&mut self, value: f64) -> NodeId {
        let id = NodeId::new(self.next_id);
        self.next_id += 1;

        let node = Node {
            id,
            node_type: NodeType::Constant(value),
            inputs: Vec::new(),
            value: Some(value),
            gradient: None,
        };

        self.nodes.insert(id, node);
        id
    }

    /// Add a learnable variable node
    pub fn add_learnable(&mut self, name: String, initial_value: f64) -> NodeId {
        let id = NodeId::new(self.next_id);
        self.next_id += 1;
        self.learnables.push(name.clone());

        let node = Node {
            id,
            node_type: NodeType::Learnable(name),
            inputs: Vec::new(),
            value: Some(initial_value),
            gradient: Some(0.0), // Gradients start at zero
        };

        self.nodes.insert(id, node);
        id
    }

    /// Add a regular variable node
    pub fn add_variable(&mut self, name: String, input: NodeId) -> NodeId {
        let id = NodeId::new(self.next_id);
        self.next_id += 1;

        let node = Node {
            id,
            node_type: NodeType::Variable(name),
            inputs: vec![input],
            value: None,
            gradient: None,
        };

        self.nodes.insert(id, node);
        id
    }

    /// Add a binary operation node
    pub fn add_binary_op(&mut self, op: &str, left: NodeId, right: NodeId) -> NodeId {
        let id = NodeId::new(self.next_id);
        self.next_id += 1;

        let node = Node {
            id,
            node_type: NodeType::BinaryOp(op.to_string()),
            inputs: vec![left, right],
            value: None,
            gradient: None,
        };

        self.nodes.insert(id, node);
        id
    }

    /// Add a unary operation node
    pub fn add_unary_op(&mut self, op: &str, operand: NodeId) -> NodeId {
        let id = NodeId::new(self.next_id);
        self.next_id += 1;

        let node = Node {
            id,
            node_type: NodeType::UnaryOp(op.to_string()),
            inputs: vec![operand],
            value: None,
            gradient: None,
        };

        self.nodes.insert(id, node);
        id
    }

    /// Add a function call node
    pub fn add_function_call(&mut self, name: String, args: Vec<NodeId>) -> NodeId {
        let id = NodeId::new(self.next_id);
        self.next_id += 1;

        let node = Node {
            id,
            node_type: NodeType::FunctionCall(name),
            inputs: args,
            value: None,
            gradient: None,
        };

        self.nodes.insert(id, node);
        id
    }

    /// Convert an AST expression into a computational graph
    pub fn build_from_expression(&mut self, expr: &Expression, variables: &HashMap<String, NodeId>) -> Result<NodeId, String> {
        match expr {
            Expression::Number(n) => Ok(self.add_constant(*n)),
            Expression::Identifier(name) => {
                variables.get(name)
                    .copied()
                    .ok_or_else(|| format!("Undefined variable: {}", name))
            }
            Expression::BinaryOp { left, op, right } => {
                let left_id = self.build_from_expression(left, variables)?;
                let right_id = self.build_from_expression(right, variables)?;
                
                let op_str = match op {
                    BinaryOperator::Add => "add",
                    BinaryOperator::Sub => "sub",
                    BinaryOperator::Mul => "mul",
                    BinaryOperator::Div => "div",
                    BinaryOperator::Pow => "pow",
                    BinaryOperator::Equal => "eq",
                    BinaryOperator::NotEqual => "ne",
                    BinaryOperator::Less => "lt",
                    BinaryOperator::Greater => "gt",
                    BinaryOperator::LessEq => "le",
                    BinaryOperator::GreaterEq => "ge",
                };
                
                Ok(self.add_binary_op(op_str, left_id, right_id))
            }
            Expression::UnaryOp { op, expr } => {
                let expr_id = self.build_from_expression(expr, variables)?;
                let op_str = match op {
                    UnaryOperator::Neg => "neg",
                    UnaryOperator::Not => "not",
                };
                Ok(self.add_unary_op(op_str, expr_id))
            }
            Expression::Call { name, args } => {
                let mut arg_ids = Vec::new();
                for arg in args {
                    arg_ids.push(self.build_from_expression(arg, variables)?);
                }
                Ok(self.add_function_call(name.clone(), arg_ids))
            }
        }
    }

    /// Get a node by ID
    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    /// Get a mutable node by ID
    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(&id)
    }

    /// Get all nodes
    pub fn nodes(&self) -> &HashMap<NodeId, Node> {
        &self.nodes
    }

    /// Get learnable variables
    pub fn learnables(&self) -> &[String] {
        &self.learnables
    }

    /// Evaluate the forward pass
    pub fn forward_pass(&mut self) -> Result<(), String> {
        // Collect node data to iterate over
        let node_ids: Vec<NodeId> = self.nodes.keys().copied().collect();
        
        for node_id in node_ids {
            let node_type = self.nodes.get(&node_id).map(|n| &n.node_type).cloned();
            let inputs = self.nodes.get(&node_id).map(|n| n.inputs.clone()).unwrap_or_default();
            
            if let Some(node_type) = node_type {
                match node_type {
                    NodeType::Constant(v) => {
                        if let Some(node) = self.nodes.get_mut(&node_id) {
                            node.value = Some(v);
                        }
                    }
                    NodeType::Learnable(_, ) => {
                        // Already initialized
                    }
                    NodeType::Variable(_, ) => {
                        if inputs.len() == 1 {
                            if let Some(input_val) = self.nodes.get(&inputs[0]).and_then(|n| n.value) {
                                if let Some(node) = self.nodes.get_mut(&node_id) {
                                    node.value = Some(input_val);
                                }
                            }
                        }
                    }
                    NodeType::BinaryOp(op) => {
                        if inputs.len() == 2 {
                            let left_val = self.nodes.get(&inputs[0])
                                .and_then(|n| n.value)
                                .ok_or("Missing left operand")?;
                            let right_val = self.nodes.get(&inputs[1])
                                .and_then(|n| n.value)
                                .ok_or("Missing right operand")?;

                            let result = match op.as_str() {
                                "add" => left_val + right_val,
                                "sub" => left_val - right_val,
                                "mul" => left_val * right_val,
                                "div" => left_val / right_val,
                                "pow" => left_val.powf(right_val),
                                _ => return Err(format!("Unknown binary op: {}", op)),
                            };
                            
                            if let Some(node) = self.nodes.get_mut(&node_id) {
                                node.value = Some(result);
                            }
                        }
                    }
                    NodeType::UnaryOp(op) => {
                        if inputs.len() == 1 {
                            let val = self.nodes.get(&inputs[0])
                                .and_then(|n| n.value)
                                .ok_or("Missing operand")?;

                            let result = match op.as_str() {
                                "neg" => -val,
                                "not" => if val != 0.0 { 0.0 } else { 1.0 },
                                _ => return Err(format!("Unknown unary op: {}", op)),
                            };
                            
                            if let Some(node) = self.nodes.get_mut(&node_id) {
                                node.value = Some(result);
                            }
                        }
                    }
                    NodeType::FunctionCall(name) => {
                        // Placeholder for function calls
                        match name.as_str() {
                            "sigmoid" => {
                                if inputs.len() == 1 {
                                    let val = self.nodes.get(&inputs[0])
                                        .and_then(|n| n.value)
                                        .ok_or("Missing argument")?;
                                    let result = 1.0 / (1.0 + (-val).exp());
                                    if let Some(node) = self.nodes.get_mut(&node_id) {
                                        node.value = Some(result);
                                    }
                                }
                            }
                            "relu" => {
                                if inputs.len() == 1 {
                                    let val = self.nodes.get(&inputs[0])
                                        .and_then(|n| n.value)
                                        .ok_or("Missing argument")?;
                                    let result = if val > 0.0 { val } else { 0.0 };
                                    if let Some(node) = self.nodes.get_mut(&node_id) {
                                        node.value = Some(result);
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Print the graph structure for debugging
    pub fn print_structure(&self) {
        println!("=== Computational Graph ===");
        for (id, node) in &self.nodes {
            println!(
                "Node {:?}: {:?}, value: {:?}, inputs: {:?}",
                id, node.node_type, node.value, node.inputs
            );
        }
    }
}

impl Default for ComputationalGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_constant() {
        let mut graph = ComputationalGraph::new();
        let id = graph.add_constant(5.0);
        assert_eq!(graph.get_node(id).map(|n| &n.node_type), Some(&NodeType::Constant(5.0)));
    }

    #[test]
    fn test_add_learnable() {
        let mut graph = ComputationalGraph::new();
        let id = graph.add_learnable("x".to_string(), 2.0);
        assert!(matches!(graph.get_node(id).map(|n| &n.node_type), Some(NodeType::Learnable(_))));
    }

    #[test]
    fn test_binary_operation() {
        let mut graph = ComputationalGraph::new();
        let a = graph.add_constant(3.0);
        let b = graph.add_constant(2.0);
        let result = graph.add_binary_op("add", a, b);
        assert_eq!(graph.get_node(result).map(|n| n.inputs.len()), Some(2));
    }

    #[test]
    fn test_forward_pass() {
        let mut graph = ComputationalGraph::new();
        let a = graph.add_constant(3.0);
        let b = graph.add_constant(2.0);
        let _sum = graph.add_binary_op("add", a, b);

        graph.forward_pass().unwrap();

        assert_eq!(graph.get_node(a).and_then(|n| n.value), Some(3.0));
        assert_eq!(graph.get_node(b).and_then(|n| n.value), Some(2.0));
    }
}
