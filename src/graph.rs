use std::collections::HashMap;
use crate::ast::{BinaryOperator, Expression, Statement, UnaryOperator};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

/// Represents a user-defined function for inlining
#[derive(Debug, Clone)]
pub struct UserFunction {
    pub name: String,
    pub params: Vec<String>,
    pub body: Vec<Statement>,
}

/// Registry of user-defined functions
#[derive(Debug, Clone, Default)]
pub struct FunctionRegistry {
    functions: HashMap<String, UserFunction>,
}

impl FunctionRegistry {
    pub fn new() -> Self {
        FunctionRegistry {
            functions: HashMap::new(),
        }
    }

    pub fn register(&mut self, name: String, params: Vec<String>, body: Vec<Statement>) {
        self.functions.insert(name.clone(), UserFunction { name, params, body });
    }

    pub fn get(&self, name: &str) -> Option<&UserFunction> {
        self.functions.get(name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }
}

impl NodeId {
    pub fn new(id: usize) -> Self {
        NodeId(id)
    }

    pub fn index(&self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Result<Self, String> {
        let expected: usize = shape.iter().product();
        if expected != data.len() {
            return Err(format!("Tensor data/shape mismatch: expected {}, got {}", expected, data.len()));
        }
        Ok(Tensor { data, shape })
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Tensor { data: vec![0.0; size], shape }
    }

    pub fn zeros_like(&self) -> Self {
        Tensor::zeros(self.shape.clone())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Scalar(f64),
    Tensor(Tensor),
}

impl Value {
    pub fn as_scalar(&self) -> Option<f64> {
        match self {
            Value::Scalar(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_tensor(&self) -> Option<&Tensor> {
        match self {
            Value::Tensor(t) => Some(t),
            _ => None,
        }
    }

    pub fn zeros_like(&self) -> Value {
        match self {
            Value::Scalar(_) => Value::Scalar(0.0),
            Value::Tensor(t) => Value::Tensor(t.zeros_like()),
        }
    }

    pub fn ones_like(&self) -> Value {
        match self {
            Value::Scalar(_) => Value::Scalar(1.0),
            Value::Tensor(t) => Value::Tensor(Tensor { data: vec![1.0; t.data.len()], shape: t.shape.clone() }),
        }
    }

    pub fn map_unary<F>(&self, f: F) -> Result<Value, String>
    where
        F: Fn(f64) -> f64,
    {
        match self {
            Value::Scalar(v) => Ok(Value::Scalar(f(*v))),
            Value::Tensor(t) => Ok(Value::Tensor(Tensor { data: t.data.iter().map(|x| f(*x)).collect(), shape: t.shape.clone() })),
        }
    }

    pub fn map2<F>(&self, other: &Value, f: F) -> Result<Value, String>
    where
        F: Fn(f64, f64) -> f64,
    {
        match (self, other) {
            (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(f(*a, *b))),
            (Value::Tensor(t1), Value::Tensor(t2)) => {
                if t1.shape != t2.shape {
                    return Err("Tensor shape mismatch".to_string());
                }
                let data = t1.data.iter().zip(t2.data.iter()).map(|(a, b)| f(*a, *b)).collect();
                Ok(Value::Tensor(Tensor { data, shape: t1.shape.clone() }))
            }
            (Value::Scalar(s), Value::Tensor(t)) => {
                let data = t.data.iter().map(|b| f(*s, *b)).collect();
                Ok(Value::Tensor(Tensor { data, shape: t.shape.clone() }))
            }
            (Value::Tensor(t), Value::Scalar(s)) => {
                let data = t.data.iter().map(|a| f(*a, *s)).collect();
                Ok(Value::Tensor(Tensor { data, shape: t.shape.clone() }))
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub id: NodeId,
    pub node_type: NodeType,
    pub inputs: Vec<NodeId>,
    pub value: Option<Value>,
    pub gradient: Option<Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NodeType {
    Constant(Value),
    Learnable(String),
    Variable(String),
    BinaryOp(String),
    UnaryOp(String),
    FunctionCall(String),
    /// Heap-allocated tensor with dynamic shape
    HeapTensor(String),
    /// Reference to a freed tensor (for tracking)
    FreedTensor(String),
}

#[derive(Debug, Clone)]
pub struct ComputationalGraph {
    nodes: HashMap<NodeId, Node>,
    next_id: usize,
    learnables: Vec<String>,
    /// Track heap-allocated tensors for memory management
    heap_allocations: HashMap<String, NodeId>,
}

impl ComputationalGraph {
    pub fn new() -> Self {
        ComputationalGraph {
            nodes: HashMap::new(),
            next_id: 0,
            learnables: Vec::new(),
            heap_allocations: HashMap::new(),
        }
    }

    pub fn add_constant(&mut self, value: f64) -> NodeId {
        let id = NodeId::new(self.next_id);
        self.next_id += 1;

        let node = Node {
            id,
            node_type: NodeType::Constant(Value::Scalar(value)),
            inputs: Vec::new(),
            value: Some(Value::Scalar(value)),
            gradient: None,
        };

        self.nodes.insert(id, node);
        id
    }

    pub fn add_constant_tensor(&mut self, data: Vec<f64>, shape: Vec<usize>) -> Result<NodeId, String> {
        let tensor = Tensor::new(data, shape)?;
        let id = NodeId::new(self.next_id);
        self.next_id += 1;

        let node = Node {
            id,
            node_type: NodeType::Constant(Value::Tensor(tensor.clone())),
            inputs: Vec::new(),
            value: Some(Value::Tensor(tensor)),
            gradient: None,
        };

        self.nodes.insert(id, node);
        Ok(id)
    }

    pub fn add_learnable(&mut self, name: String, initial_value: f64) -> NodeId {
        let id = NodeId::new(self.next_id);
        self.next_id += 1;
        self.learnables.push(name.clone());

        let node = Node {
            id,
            node_type: NodeType::Learnable(name),
            inputs: Vec::new(),
            value: Some(Value::Scalar(initial_value)),
            gradient: Some(Value::Scalar(0.0)),
        };

        self.nodes.insert(id, node);
        id
    }

    pub fn add_learnable_tensor(&mut self, name: String, data: Vec<f64>, shape: Vec<usize>) -> Result<NodeId, String> {
        let tensor = Tensor::new(data, shape)?;
        let id = NodeId::new(self.next_id);
        self.next_id += 1;
        self.learnables.push(name.clone());

        let grad = Value::Tensor(Tensor::zeros(tensor.shape.clone()));

        let node = Node {
            id,
            node_type: NodeType::Learnable(name),
            inputs: Vec::new(),
            value: Some(Value::Tensor(tensor)),
            gradient: Some(grad),
        };

        self.nodes.insert(id, node);
        Ok(id)
    }

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

    /// Allocate a heap tensor with the given shape (dimensions as NodeIds)
    pub fn add_heap_tensor(&mut self, name: String, shape: Vec<usize>) -> Result<NodeId, String> {
        let id = NodeId::new(self.next_id);
        self.next_id += 1;

        // Create tensor filled with zeros
        let size: usize = shape.iter().product();
        let tensor = Tensor::new(vec![0.0; size], shape)?;
        let grad = Value::Tensor(Tensor::zeros(tensor.shape.clone()));

        let node = Node {
            id,
            node_type: NodeType::HeapTensor(name.clone()),
            inputs: Vec::new(),
            value: Some(Value::Tensor(tensor)),
            gradient: Some(grad),
        };

        self.nodes.insert(id, node);
        self.heap_allocations.insert(name, id);
        Ok(id)
    }

    /// Free a heap-allocated tensor
    pub fn free_heap_tensor(&mut self, name: &str) -> Result<(), String> {
        if let Some(node_id) = self.heap_allocations.remove(name) {
            // Mark the node as freed (we keep it for graph integrity but clear the data)
            if let Some(node) = self.nodes.get_mut(&node_id) {
                node.node_type = NodeType::FreedTensor(name.to_string());
                node.value = None;
                node.gradient = None;
            }
            Ok(())
        } else {
            Err(format!("Cannot free '{}': not a heap-allocated tensor", name))
        }
    }

    /// Check if a tensor is still allocated (not freed)
    pub fn is_heap_allocated(&self, name: &str) -> bool {
        self.heap_allocations.contains_key(name)
    }

    /// Get the node ID for a heap-allocated tensor
    pub fn get_heap_tensor(&self, name: &str) -> Option<NodeId> {
        self.heap_allocations.get(name).copied()
    }

    /// Reallocate a heap tensor with a new shape (preserves data where possible)
    pub fn realloc_heap_tensor(&mut self, name: &str, new_shape: Vec<usize>) -> Result<NodeId, String> {
        // Get the old node and its data
        let old_node_id = self.heap_allocations.get(name)
            .ok_or_else(|| format!("Cannot realloc '{}': not a heap-allocated tensor", name))?
            .clone();
        
        // Get old data
        let old_data = self.nodes.get(&old_node_id)
            .and_then(|n| n.value.clone())
            .and_then(|v| match v {
                Value::Tensor(t) => Some(t.data),
                _ => None,
            })
            .unwrap_or_default();
        
        // Mark old node as freed
        if let Some(node) = self.nodes.get_mut(&old_node_id) {
            node.node_type = NodeType::FreedTensor(name.to_string());
            node.value = None;
            node.gradient = None;
        }
        
        // Calculate new size
        let new_size: usize = new_shape.iter().product();
        
        // Create new data, copying from old where possible
        let mut new_data = vec![0.0; new_size];
        let copy_len = old_data.len().min(new_size);
        new_data[..copy_len].copy_from_slice(&old_data[..copy_len]);
        
        // Create new node
        let new_id = NodeId::new(self.next_id);
        self.next_id += 1;
        
        let tensor = Tensor::new(new_data, new_shape.clone())?;
        let grad = Value::Tensor(Tensor::zeros(new_shape));
        
        let node = Node {
            id: new_id,
            node_type: NodeType::HeapTensor(name.to_string()),
            inputs: Vec::new(),
            value: Some(Value::Tensor(tensor)),
            gradient: Some(grad),
        };
        
        self.nodes.insert(new_id, node);
        self.heap_allocations.insert(name.to_string(), new_id);
        
        Ok(new_id)
    }

    pub fn build_from_expression(&mut self, expr: &Expression, variables: &HashMap<String, NodeId>) -> Result<NodeId, String> {
        self.build_from_expression_with_functions(expr, variables, &FunctionRegistry::new())
    }

    /// Build expression with support for user-defined function inlining
    pub fn build_from_expression_with_functions(
        &mut self,
        expr: &Expression,
        variables: &HashMap<String, NodeId>,
        functions: &FunctionRegistry,
    ) -> Result<NodeId, String> {
        match expr {
            Expression::Number(n) => Ok(self.add_constant(*n)),
            Expression::TensorLiteral { data, shape } => {
                self.add_constant_tensor(data.clone(), shape.clone())
            }
            Expression::Identifier(name) => variables.get(name).copied().ok_or_else(|| format!("Undefined variable: {}", name)),
            Expression::Index { target, indices } => {
                // Lower as a function call: index(target, i, j, ...)
                let t_id = self.build_from_expression_with_functions(target, variables, functions)?;
                let mut args = vec![t_id];
                for idx in indices {
                    args.push(self.build_from_expression_with_functions(idx, variables, functions)?);
                }
                Ok(self.add_function_call("index".to_string(), args))
            }
            Expression::BinaryOp { left, op, right } => {
                let left_id = self.build_from_expression_with_functions(left, variables, functions)?;
                let right_id = self.build_from_expression_with_functions(right, variables, functions)?;

                let op_str = match op {
                    BinaryOperator::Add => "add",
                    BinaryOperator::Sub => "sub",
                    BinaryOperator::Mul => "mul",
                    BinaryOperator::Div => "div",
                    BinaryOperator::Mod => "mod",
                    BinaryOperator::Pow => "pow",
                    BinaryOperator::Equal => "eq",
                    BinaryOperator::NotEqual => "ne",
                    BinaryOperator::Less => "lt",
                    BinaryOperator::Greater => "gt",
                    BinaryOperator::LessEq => "le",
                    BinaryOperator::GreaterEq => "ge",
                    BinaryOperator::And => "and",
                    BinaryOperator::Or => "or",
                };

                Ok(self.add_binary_op(op_str, left_id, right_id))
            }
            Expression::UnaryOp { op, expr } => {
                let expr_id = self.build_from_expression_with_functions(expr, variables, functions)?;
                let op_str = match op {
                    UnaryOperator::Neg => "neg",
                    UnaryOperator::Not => "not",
                };
                Ok(self.add_unary_op(op_str, expr_id))
            }
            Expression::Call { name, args } => {
                // Check if this is a user-defined function
                if let Some(user_fn) = functions.get(name) {
                    // Inline the user function
                    if args.len() != user_fn.params.len() {
                        return Err(format!(
                            "Function '{}' expects {} arguments, got {}",
                            name, user_fn.params.len(), args.len()
                        ));
                    }

                    // Evaluate arguments and bind to parameters
                    let mut local_vars = variables.clone();
                    for (param, arg_expr) in user_fn.params.iter().zip(args.iter()) {
                        let arg_id = self.build_from_expression_with_functions(arg_expr, variables, functions)?;
                        local_vars.insert(param.clone(), arg_id);
                    }

                    // Execute function body and get return value
                    let body = user_fn.body.clone();
                    let result = self.inline_function_body(&body, &mut local_vars, functions)?;
                    Ok(result)
                } else {
                    // Built-in function or external call
                    let mut arg_ids = Vec::new();
                    for arg in args {
                        arg_ids.push(self.build_from_expression_with_functions(arg, variables, functions)?);
                    }
                    Ok(self.add_function_call(name.clone(), arg_ids))
                }
            }
        }
    }

    /// Inline a function body and return the result node
    fn inline_function_body(
        &mut self,
        body: &[Statement],
        variables: &mut HashMap<String, NodeId>,
        functions: &FunctionRegistry,
    ) -> Result<NodeId, String> {
        let mut last_node: Option<NodeId> = None;

        for stmt in body {
            match stmt {
                Statement::LetDeclaration { name, value } => {
                    let val_id = self.build_from_expression_with_functions(value, variables, functions)?;
                    variables.insert(name.clone(), val_id);
                    last_node = Some(val_id);
                }
                Statement::LearnDeclaration { name, value } => {
                    // In function context, learn declarations become let declarations
                    match value {
                        Expression::Number(n) => {
                            let node_id = self.add_learnable(name.clone(), *n);
                            variables.insert(name.clone(), node_id);
                            last_node = Some(node_id);
                        }
                        Expression::TensorLiteral { data, shape } => {
                            let node_id = self.add_learnable_tensor(name.clone(), data.clone(), shape.clone())?;
                            variables.insert(name.clone(), node_id);
                            last_node = Some(node_id);
                        }
                        _ => {
                            let val_id = self.build_from_expression_with_functions(value, variables, functions)?;
                            variables.insert(name.clone(), val_id);
                            last_node = Some(val_id);
                        }
                    }
                }
                Statement::Assignment { name, value } => {
                    let val_id = self.build_from_expression_with_functions(value, variables, functions)?;
                    variables.insert(name.clone(), val_id);
                    last_node = Some(val_id);
                }
                Statement::Return(Some(expr)) => {
                    let id = self.build_from_expression_with_functions(expr, variables, functions)?;
                    return Ok(id);
                }
                Statement::Return(None) => {
                    // Return last computed value
                    if let Some(id) = last_node {
                        return Ok(id);
                    }
                    return Err("Function has empty return".to_string());
                }
                Statement::Expression(expr) => {
                    let id = self.build_from_expression_with_functions(expr, variables, functions)?;
                    last_node = Some(id);
                }
                Statement::Block(inner) => {
                    let result = self.inline_function_body(inner, variables, functions)?;
                    last_node = Some(result);
                }
                Statement::If { condition, then_branch, else_branch } => {
                    // For now, evaluate condition at lowering time
                    let cond_id = self.build_from_expression_with_functions(condition, variables, functions)?;
                    self.forward_pass()?;
                    let cond_val = self.get_node(cond_id)
                        .and_then(|n| n.value.clone())
                        .and_then(|v| match v { Value::Scalar(s) => Some(s), _ => None })
                        .unwrap_or(0.0);
                    if cond_val != 0.0 {
                        let result = self.inline_function_body(then_branch, variables, functions)?;
                        last_node = Some(result);
                    } else if !else_branch.is_empty() {
                        let result = self.inline_function_body(else_branch, variables, functions)?;
                        last_node = Some(result);
                    }
                }
                Statement::While { condition, body: loop_body } => {
                    for _ in 0..1_000_000usize {
                        let cond_id = self.build_from_expression_with_functions(condition, variables, functions)?;
                        self.forward_pass()?;
                        let cond_val = self.get_node(cond_id)
                            .and_then(|n| n.value.clone())
                            .and_then(|v| match v { Value::Scalar(s) => Some(s), _ => None })
                            .unwrap_or(0.0);
                        if cond_val == 0.0 { break; }
                        let result = self.inline_function_body(loop_body, variables, functions)?;
                        last_node = Some(result);
                    }
                }
                Statement::Minimize(expr) => {
                    let id = self.build_from_expression_with_functions(expr, variables, functions)?;
                    last_node = Some(id);
                }
                Statement::OptimizeLoop { .. } => {
                    return Err("OptimizeLoop not supported inside user functions".to_string());
                }
                Statement::Alloc { name, shape } => {
                    // Evaluate shape dimensions at lowering time
                    let mut dims = Vec::new();
                    for dim_expr in shape {
                        let dim_id = self.build_from_expression_with_functions(dim_expr, variables, functions)?;
                        self.forward_pass()?;
                        let dim_val = self.get_node(dim_id)
                            .and_then(|n| n.value.clone())
                            .and_then(|v| match v { Value::Scalar(s) => Some(s as usize), _ => None })
                            .ok_or_else(|| format!("Alloc dimension must be a scalar for '{}'", name))?;
                        dims.push(dim_val);
                    }
                    let node_id = self.add_heap_tensor(name.clone(), dims)?;
                    variables.insert(name.clone(), node_id);
                    last_node = Some(node_id);
                }
                Statement::Free { name } => {
                    self.free_heap_tensor(name)?;
                    variables.remove(name);
                    // free doesn't produce a value, but we keep last_node unchanged
                }
                Statement::Realloc { name, shape } => {
                    // Evaluate shape dimensions at lowering time
                    let mut dims = Vec::new();
                    for dim_expr in shape {
                        let dim_id = self.build_from_expression_with_functions(dim_expr, variables, functions)?;
                        self.forward_pass()?;
                        let dim_val = self.get_node(dim_id)
                            .and_then(|n| n.value.clone())
                            .and_then(|v| match v { Value::Scalar(s) => Some(s as usize), _ => None })
                            .ok_or_else(|| format!("Realloc dimension must be a scalar for '{}'", name))?;
                        dims.push(dim_val);
                    }
                    let node_id = self.realloc_heap_tensor(name, dims)?;
                    variables.insert(name.clone(), node_id);
                    last_node = Some(node_id);
                }
            }
        }

        last_node.ok_or_else(|| "Function body produced no value".to_string())
    }

    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(&id)
    }

    pub fn nodes(&self) -> &HashMap<NodeId, Node> {
        &self.nodes
    }

    pub fn learnables(&self) -> &[String] {
        &self.learnables
    }

    fn topological_order(&self) -> Result<Vec<NodeId>, String> {
        let mut indegree: HashMap<NodeId, usize> = HashMap::new();
        let mut adj: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        for (&id, node) in &self.nodes {
            indegree.insert(id, node.inputs.len());
            for input in &node.inputs {
                adj.entry(*input).or_default().push(id);
            }
        }

        let mut zero_indegree: Vec<NodeId> = indegree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();
        zero_indegree.sort_by_key(|n| n.index());

        let mut order = Vec::with_capacity(self.nodes.len());

        while let Some(id) = zero_indegree.first().cloned() {
            zero_indegree.remove(0);
            order.push(id);

            if let Some(children) = adj.get(&id) {
                for child in children {
                    if let Some(entry) = indegree.get_mut(child) {
                        *entry -= 1;
                        if *entry == 0 {
                            zero_indegree.push(*child);
                        }
                    }
                }
                zero_indegree.sort_by_key(|n| n.index());
            }
        }

        if order.len() != self.nodes.len() {
            return Err("Graph contains a cycle or disconnected nodes".to_string());
        }

        Ok(order)
    }

    pub fn forward_pass(&mut self) -> Result<(), String> {
        let node_ids = self.topological_order()?;

        for node_id in node_ids {
            let node_type = self.nodes.get(&node_id).map(|n| &n.node_type).cloned();
            let inputs = self.nodes.get(&node_id).map(|n| n.inputs.clone()).unwrap_or_default();

            if let Some(node_type) = node_type {
                match node_type {
                    NodeType::Constant(v) => {
                        if let Some(node) = self.nodes.get_mut(&node_id) {
                            node.value = Some(v.clone());
                        }
                    }
                    NodeType::Learnable(_) => {}
                    NodeType::Variable(_) => {
                        if inputs.len() == 1 {
                            if let Some(input_val) = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone()) {
                                if let Some(node) = self.nodes.get_mut(&node_id) {
                                    node.value = Some(input_val);
                                }
                            }
                        }
                    }
                    NodeType::BinaryOp(op) => {
                        if inputs.len() == 2 {
                            let left_val = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone()).ok_or("Missing left operand")?;
                            let right_val = self.nodes.get(&inputs[1]).and_then(|n| n.value.clone()).ok_or("Missing right operand")?;

                            let result = match op.as_str() {
                                "add" | "sub" | "mul" | "div" => broadcast_binary(&left_val, &right_val, op.as_str())?,
                                "mod" => left_val.map2(&right_val, |a, b| a % b)?,
                                "pow" => left_val.map2(&right_val, |a, b| a.powf(b))?,
                                "eq" => Value::Scalar(if (left_val.as_scalar().unwrap_or(f64::NAN) - right_val.as_scalar().unwrap_or(f64::NAN)).abs() < f64::EPSILON { 1.0 } else { 0.0 }),
                                "ne" => Value::Scalar(if (left_val.as_scalar().unwrap_or(f64::NAN) - right_val.as_scalar().unwrap_or(f64::NAN)).abs() >= f64::EPSILON { 1.0 } else { 0.0 }),
                                "lt" => Value::Scalar(if left_val.as_scalar().ok_or("Non-scalar left")? < right_val.as_scalar().ok_or("Non-scalar right")? { 1.0 } else { 0.0 }),
                                "gt" => Value::Scalar(if left_val.as_scalar().ok_or("Non-scalar left")? > right_val.as_scalar().ok_or("Non-scalar right")? { 1.0 } else { 0.0 }),
                                "le" => Value::Scalar(if left_val.as_scalar().ok_or("Non-scalar left")? <= right_val.as_scalar().ok_or("Non-scalar right")? { 1.0 } else { 0.0 }),
                                "ge" => Value::Scalar(if left_val.as_scalar().ok_or("Non-scalar left")? >= right_val.as_scalar().ok_or("Non-scalar right")? { 1.0 } else { 0.0 }),
                                "and" => Value::Scalar(if left_val.as_scalar().ok_or("Non-scalar left")? != 0.0 && right_val.as_scalar().ok_or("Non-scalar right")? != 0.0 { 1.0 } else { 0.0 }),
                                "or" => Value::Scalar(if left_val.as_scalar().ok_or("Non-scalar left")? != 0.0 || right_val.as_scalar().ok_or("Non-scalar right")? != 0.0 { 1.0 } else { 0.0 }),
                                _ => return Err(format!("Unknown binary op: {}", op)),
                            };

                            if let Some(node) = self.nodes.get_mut(&node_id) {
                                node.value = Some(result);
                            }
                        }
                    }
                    NodeType::UnaryOp(op) => {
                        if inputs.len() == 1 {
                            let val = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone()).ok_or("Missing operand")?;

                            let result = match op.as_str() {
                                "neg" => val.map_unary(|v| -v)?,
                                "not" => {
                                    let s = val.as_scalar().ok_or("Non-scalar for not")?;
                                    Value::Scalar(if s != 0.0 { 0.0 } else { 1.0 })
                                }
                                _ => return Err(format!("Unknown unary op: {}", op)),
                            };

                            if let Some(node) = self.nodes.get_mut(&node_id) {
                                node.value = Some(result);
                            }
                        }
                    }
                    NodeType::FunctionCall(name) => match name.as_str() {
                        "sigmoid" => {
                            if inputs.len() == 1 {
                                let val = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone()).ok_or("Missing argument")?;
                                let result = val.map_unary(|v| 1.0 / (1.0 + (-v).exp()))?;
                                if let Some(node) = self.nodes.get_mut(&node_id) {
                                    node.value = Some(result);
                                }
                            }
                        }
                        "relu" => {
                            if inputs.len() == 1 {
                                let val = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone()).ok_or("Missing argument")?;
                                let result = val.map_unary(|v| if v > 0.0 { v } else { 0.0 })?;
                                if let Some(node) = self.nodes.get_mut(&node_id) {
                                    node.value = Some(result);
                                }
                            }
                        }
                        "print" => {
                            if inputs.len() != 1 { return Err("print expects 1 argument".to_string()); }
                            let val = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone()).ok_or("Missing argument")?;
                            match &val {
                                Value::Scalar(s) => println!("[print] {}", s),
                                Value::Tensor(t) => println!("[print] tensor {:?}: {:?}", t.shape, if t.data.len() > 16 { &t.data[..16] } else { &t.data[..] }),
                            }
                            if let Some(node) = self.nodes.get_mut(&node_id) { node.value = Some(val); }
                        }
                        "dot" => {
                            if inputs.len() != 2 { return Err("dot expects 2 arguments".to_string()); }
                            let a = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone()).ok_or("Missing arg a")?;
                            let b = self.nodes.get(&inputs[1]).and_then(|n| n.value.clone()).ok_or("Missing arg b")?;
                            let result = match (a, b) {
                                (Value::Tensor(ta), Value::Tensor(tb)) => {
                                    if ta.shape.len() != 1 || tb.shape.len() != 1 { return Err("dot expects rank-1 tensors".to_string()); }
                                    if ta.shape[0] != tb.shape[0] { return Err("dot vector sizes must match".to_string()); }
                                    let s = ta.data.iter().zip(tb.data.iter()).map(|(x,y)| x*y).sum();
                                    Value::Scalar(s)
                                }
                                _ => return Err("dot expects tensors".to_string()),
                            };
                            if let Some(node) = self.nodes.get_mut(&node_id) { node.value = Some(result); }
                        }
                        "matmul" => {
                            if inputs.len() != 2 { return Err("matmul expects 2 arguments".to_string()); }
                            let a = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone()).ok_or("Missing arg a")?;
                            let b = self.nodes.get(&inputs[1]).and_then(|n| n.value.clone()).ok_or("Missing arg b")?;
                            let result = match (a, b) {
                                (Value::Tensor(ta), Value::Tensor(tb)) => {
                                    if ta.shape.len() != 2 || tb.shape.len() != 2 { return Err("matmul expects rank-2 tensors".to_string()); }
                                    let (m,k) = (ta.shape[0], ta.shape[1]);
                                    let (k2,n) = (tb.shape[0], tb.shape[1]);
                                    if k != k2 { return Err("matmul inner dimensions must match".to_string()); }
                                    let out = matmul_tensors(&ta, &tb)?;
                                    Value::Tensor(out)
                                }
                                _ => return Err("matmul expects tensors".to_string()),
                            };
                            if let Some(node) = self.nodes.get_mut(&node_id) { node.value = Some(result); }
                        }
                        "matvec" => {
                            if inputs.len() != 2 { return Err("matvec expects 2 arguments".to_string()); }
                            let a = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone()).ok_or("Missing arg A")?;
                            let x = self.nodes.get(&inputs[1]).and_then(|n| n.value.clone()).ok_or("Missing arg x")?;
                            let result = match (a, x) {
                                (Value::Tensor(ta), Value::Tensor(tx)) => {
                                    if ta.shape.len() != 2 || tx.shape.len() != 1 { return Err("matvec expects A rank-2 and x rank-1".to_string()); }
                                    let (m,k) = (ta.shape[0], ta.shape[1]);
                                    if tx.shape[0] != k { return Err("matvec inner dimension mismatch".to_string()); }
                                    let mut out = vec![0.0; m];
                                    for i in 0..m {
                                        let mut s = 0.0;
                                        for j in 0..k { s += ta.data[i*k + j] * tx.data[j]; }
                                        out[i] = s;
                                    }
                                    Value::Tensor(Tensor { data: out, shape: vec![m] })
                                }
                                _ => return Err("matvec expects tensors".to_string()),
                            };
                            if let Some(node) = self.nodes.get_mut(&node_id) { node.value = Some(result); }
                        }
                        "vecmat" => {
                            if inputs.len() != 2 { return Err("vecmat expects 2 arguments".to_string()); }
                            let x = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone()).ok_or("Missing arg x")?;
                            let b = self.nodes.get(&inputs[1]).and_then(|n| n.value.clone()).ok_or("Missing arg B")?;
                            let result = match (x, b) {
                                (Value::Tensor(tx), Value::Tensor(tb)) => {
                                    if tx.shape.len() != 1 || tb.shape.len() != 2 { return Err("vecmat expects x rank-1 and B rank-2".to_string()); }
                                    let (m,n) = (tb.shape[0], tb.shape[1]);
                                    if tx.shape[0] != m { return Err("vecmat inner dimension mismatch".to_string()); }
                                    let mut out = vec![0.0; n];
                                    for j in 0..n {
                                        let mut s = 0.0;
                                        for i in 0..m { s += tx.data[i] * tb.data[i*n + j]; }
                                        out[j] = s;
                                    }
                                    Value::Tensor(Tensor { data: out, shape: vec![n] })
                                }
                                _ => return Err("vecmat expects tensors".to_string()),
                            };
                            if let Some(node) = self.nodes.get_mut(&node_id) { node.value = Some(result); }
                        }
                        "sum" => {
                            if inputs.len() != 1 { return Err("sum expects 1 argument".to_string()); }
                            let val = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone()).ok_or("Missing argument")?;
                            let result = match val {
                                Value::Scalar(s) => Value::Scalar(s),
                                Value::Tensor(t) => Value::Scalar(t.data.iter().copied().sum()),
                            };
                            if let Some(node) = self.nodes.get_mut(&node_id) { node.value = Some(result); }
                        }
                        "mean" => {
                            if inputs.len() != 1 { return Err("mean expects 1 argument".to_string()); }
                            let val = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone()).ok_or("Missing argument")?;
                            let result = match val {
                                Value::Scalar(s) => Value::Scalar(s),
                                Value::Tensor(t) => {
                                    let n = t.data.len() as f64;
                                    Value::Scalar(t.data.iter().copied().sum::<f64>() / n)
                                }
                            };
                            if let Some(node) = self.nodes.get_mut(&node_id) { node.value = Some(result); }
                        }
                        "sin" => {
                            if inputs.len() != 1 { return Err("sin expects 1 argument".to_string()); }
                            let v = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone()).ok_or("Missing argument")?;
                            let res = v.map_unary(|x| x.sin())?;
                            if let Some(node) = self.nodes.get_mut(&node_id) { node.value = Some(res); }
                        }
                        "cos" => {
                            if inputs.len() != 1 { return Err("cos expects 1 argument".to_string()); }
                            let v = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone()).ok_or("Missing argument")?;
                            let res = v.map_unary(|x| x.cos())?;
                            if let Some(node) = self.nodes.get_mut(&node_id) { node.value = Some(res); }
                        }
                        "tanh" => {
                            if inputs.len() != 1 { return Err("tanh expects 1 argument".to_string()); }
                            let v = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone()).ok_or("Missing argument")?;
                            let res = v.map_unary(|x| x.tanh())?;
                            if let Some(node) = self.nodes.get_mut(&node_id) { node.value = Some(res); }
                        }
                        "exp" => {
                            if inputs.len() != 1 { return Err("exp expects 1 argument".to_string()); }
                            let v = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone()).ok_or("Missing argument")?;
                            let res = v.map_unary(|x| x.exp())?;
                            if let Some(node) = self.nodes.get_mut(&node_id) { node.value = Some(res); }
                        }
                        "log" => {
                            if inputs.len() != 1 { return Err("log expects 1 argument".to_string()); }
                            let v = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone()).ok_or("Missing argument")?;
                            let res = v.map_unary(|x| x.ln())?;
                            if let Some(node) = self.nodes.get_mut(&node_id) { node.value = Some(res); }
                        }
                        "sqrt" => {
                            if inputs.len() != 1 { return Err("sqrt expects 1 argument".to_string()); }
                            let v = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone()).ok_or("Missing argument")?;
                            let res = v.map_unary(|x| x.sqrt())?;
                            if let Some(node) = self.nodes.get_mut(&node_id) { node.value = Some(res); }
                        }
                        "abs" => {
                            if inputs.len() != 1 { return Err("abs expects 1 argument".to_string()); }
                            let v = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone()).ok_or("Missing argument")?;
                            let res = v.map_unary(|x| x.abs())?;
                            if let Some(node) = self.nodes.get_mut(&node_id) { node.value = Some(res); }
                        }
                        "floor" => {
                            if inputs.len() != 1 { return Err("floor expects 1 argument".to_string()); }
                            let v = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone()).ok_or("Missing argument")?;
                            let res = v.map_unary(|x| x.floor())?;
                            if let Some(node) = self.nodes.get_mut(&node_id) { node.value = Some(res); }
                        }
                        "ceil" => {
                            if inputs.len() != 1 { return Err("ceil expects 1 argument".to_string()); }
                            let v = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone()).ok_or("Missing argument")?;
                            let res = v.map_unary(|x| x.ceil())?;
                            if let Some(node) = self.nodes.get_mut(&node_id) { node.value = Some(res); }
                        }
                        "index" => {
                            if inputs.len() < 2 { return Err("index expects at least target and one index".to_string()); }
                            let target_val = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone()).ok_or("Missing target")?;
                            let mut idx_vals: Vec<usize> = Vec::new();
                            for &iid in &inputs[1..] {
                                let s = self.nodes.get(&iid).and_then(|n| n.value.clone()).and_then(|v| v.as_scalar());
                                let s = s.ok_or("Index must be scalar")?;
                                if !s.is_finite() { return Err("Index not finite".to_string()); }
                                let u = if s >= 0.0 { s as usize } else { return Err("Negative index".to_string()); };
                                idx_vals.push(u);
                            }
                            let result = match target_val {
                                Value::Tensor(t) => {
                                    if idx_vals.len() != t.shape.len() { return Err("Index rank must match tensor rank".to_string()); }
                                    // Compute linear index row-major
                                    let mut stride = 1usize;
                                    let mut strides = vec![0usize; t.shape.len()];
                                    for (i, dim) in t.shape.iter().enumerate().rev() {
                                        strides[i] = stride;
                                        stride *= *dim;
                                    }
                                    let mut linear = 0usize;
                                    for (i, &idx) in idx_vals.iter().enumerate() {
                                        if idx >= t.shape[i] { return Err("Index out of bounds".to_string()); }
                                        linear += idx * strides[i];
                                    }
                                    let v = t.data[linear];
                                    Value::Scalar(v)
                                }
                                Value::Scalar(_) => return Err("Cannot index into scalar".to_string()),
                            };
                            if let Some(node) = self.nodes.get_mut(&node_id) { node.value = Some(result); }
                        }
                        _ => {}
                    },
                    NodeType::HeapTensor(_) => {
                        // HeapTensor already has its value set during allocation
                        // Nothing to do in forward pass
                    }
                    NodeType::FreedTensor(_) => {
                        // FreedTensor nodes are skipped in forward pass
                        // They have no value - just continue to next node
                    }
                }
            }
        }
        Ok(())
    }

    pub fn print_structure(&self) {
        println!("=== Computational Graph ===");
        for (id, node) in &self.nodes {
            println!(
                "Node {:?}: {:?}, value: {:?}, gradients: {:?}, inputs: {:?}",
                id, node.node_type, node.value, node.gradient, node.inputs
            );
        }
    }

    pub fn backward_pass(&mut self, output_id: NodeId) -> Result<(), String> {
        if let Some(node) = self.nodes.get_mut(&output_id) {
            node.gradient = Some(node.value.clone().map(|v| v.ones_like()).unwrap_or(Value::Scalar(1.0)));
        }

        let mut node_ids = self.topological_order()?;
        node_ids.reverse();

        for node_id in node_ids {
            let grad_opt = self.get_node(node_id).and_then(|n| n.gradient.clone());
            let Some(gradient) = grad_opt else { continue };
            match &gradient {
                Value::Scalar(g) if *g == 0.0 => continue,
                Value::Tensor(t) if t.data.iter().all(|v| *v == 0.0) => continue,
                _ => {}
            }

            if let Some(node) = self.get_node(node_id) {
                let node_type = node.node_type.clone();
                let inputs = node.inputs.clone();

                match node_type {
                    NodeType::Constant(_) => {}
                    NodeType::Learnable(_) => {}
                    NodeType::Variable(_) => {
                        if inputs.len() == 1 {
                            if let Some(input_node) = self.nodes.get_mut(&inputs[0]) {
                                let updated = add_grad(input_node.gradient.clone(), gradient.clone())?;
                                input_node.gradient = Some(updated);
                            }
                        }
                    }
                    NodeType::BinaryOp(ref op) => {
                        if inputs.len() == 2 {
                            let left_val = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone());
                            let right_val = self.nodes.get(&inputs[1]).and_then(|n| n.value.clone());

                            match op.as_str() {
                                "add" => {
                                    let (la, lb) = (left_val.clone().unwrap(), right_val.clone().unwrap());
                                    let ga = reduce_grad_for_input(gradient.clone(), &la, &lb, "add", true)?;
                                    let gb = reduce_grad_for_input(gradient.clone(), &lb, &la, "add", false)?;
                                    if let Some(left_node) = self.nodes.get_mut(&inputs[0]) { left_node.gradient = Some(add_grad(left_node.gradient.clone(), ga)?); }
                                    if let Some(right_node) = self.nodes.get_mut(&inputs[1]) { right_node.gradient = Some(add_grad(right_node.gradient.clone(), gb)?); }
                                }
                                "sub" => {
                                    let (la, lb) = (left_val.clone().unwrap(), right_val.clone().unwrap());
                                    let ga = reduce_grad_for_input(gradient.clone(), &la, &lb, "sub", true)?;
                                    let gb = reduce_grad_for_input(gradient.clone(), &lb, &la, "sub", false)?;
                                    if let Some(left_node) = self.nodes.get_mut(&inputs[0]) { left_node.gradient = Some(add_grad(left_node.gradient.clone(), ga)?); }
                                    if let Some(right_node) = self.nodes.get_mut(&inputs[1]) { right_node.gradient = Some(add_grad(right_node.gradient.clone(), gb)?); }
                                }
                                "mul" => {
                                    let (la, lb) = (left_val.clone().unwrap(), right_val.clone().unwrap());
                                    let ga = reduce_grad_for_input(gradient.clone(), &la, &lb, "mul", true)?;
                                    let gb = reduce_grad_for_input(gradient.clone(), &lb, &la, "mul", false)?;
                                    if let Some(left_node) = self.nodes.get_mut(&inputs[0]) { left_node.gradient = Some(add_grad(left_node.gradient.clone(), ga)?); }
                                    if let Some(right_node) = self.nodes.get_mut(&inputs[1]) { right_node.gradient = Some(add_grad(right_node.gradient.clone(), gb)?); }
                                }
                                "div" => {
                                    let (la, lb) = (left_val.clone().unwrap(), right_val.clone().unwrap());
                                    let ga = reduce_grad_for_input(gradient.clone(), &la, &lb, "div", true)?;
                                    let gb = reduce_grad_for_input(gradient.clone(), &lb, &la, "div", false)?;
                                    if let Some(left_node) = self.nodes.get_mut(&inputs[0]) { left_node.gradient = Some(add_grad(left_node.gradient.clone(), ga)?); }
                                    if let Some(right_node) = self.nodes.get_mut(&inputs[1]) { right_node.gradient = Some(add_grad(right_node.gradient.clone(), gb)?); }
                                }
                                "pow" => {
                                    if let Some(left_node) = self.nodes.get_mut(&inputs[0]) {
                                        if let (Some(a), Some(b)) = (left_val.clone(), right_val.clone()) {
                                            let b_minus_one = add_const(&b, -1.0)?;
                                            let a_pow = pow_value(&a, &b_minus_one)?;
                                            let local = mul_grad(gradient.clone(), mul_grad(b.clone(), a_pow)?)?;
                                            left_node.gradient = Some(add_grad(left_node.gradient.clone(), local)?);
                                        }
                                    }
                                    if let Some(right_node) = self.nodes.get_mut(&inputs[1]) {
                                        if let (Some(a), Some(b)) = (left_val.clone(), right_val.clone()) {
                                            let pow_ab = pow_value(&a, &b)?;
                                            let ln_a = ln_value(&a)?;
                                            let local = mul_grad(gradient.clone(), mul_grad(pow_ab, ln_a)?)?;
                                            right_node.gradient = Some(add_grad(right_node.gradient.clone(), local)?);
                                        }
                                    }
                                }
                                "mod" => {}
                                "and" | "or" => {}
                                _ => {}
                            }
                        }
                    }
                    NodeType::UnaryOp(ref op) => {
                        if inputs.len() == 1 {
                            match op.as_str() {
                                "neg" => {
                                    if let Some(node) = self.nodes.get_mut(&inputs[0]) {
                                        node.gradient = Some(add_grad(node.gradient.clone(), negate_value(gradient.clone()))?);
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    NodeType::FunctionCall(name) => {
                        if inputs.len() == 1 {
                            let val = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone());
                            match name.as_str() {
                                "sigmoid" => {
                                    if let Some(Value::Scalar(v)) = val {
                                        let s = 1.0 / (1.0 + (-v).exp());
                                        let g = gradient.as_scalar().ok_or("Expected scalar gradient for scalar sigmoid")?;
                                        let local = Value::Scalar(g * s * (1.0 - s));
                                        if let Some(node) = self.nodes.get_mut(&inputs[0]) {
                                            node.gradient = Some(add_grad(node.gradient.clone(), local)?);
                                        }
                                    } else if let Some(Value::Tensor(t)) = val {
                                        let deriv: Vec<f64> = t.data.iter().map(|x| {
                                            let s = 1.0 / (1.0 + (-x).exp());
                                            s * (1.0 - s)
                                        }).collect();
                                        let local = match gradient.clone() {
                                            Value::Tensor(g) => {
                                                if g.shape != t.shape {
                                                    return Err("Sigmoid tensor gradient shape mismatch".to_string());
                                                }
                                                Value::Tensor(Tensor { data: g.data.iter().zip(deriv.iter()).map(|(g, d)| g * d).collect(), shape: g.shape })
                                            }
                                            Value::Scalar(s) => Value::Tensor(Tensor { data: deriv.iter().map(|d| s * d).collect(), shape: t.shape.clone() }),
                                        };
                                        if let Some(node) = self.nodes.get_mut(&inputs[0]) {
                                            node.gradient = Some(add_grad(node.gradient.clone(), local)?);
                                        }
                                    }
                                }
                                "relu" => {
                                    if let Some(Value::Scalar(v)) = val {
                                        let local = if v > 0.0 { gradient.clone() } else { Value::Scalar(0.0) };
                                        if let Some(node) = self.nodes.get_mut(&inputs[0]) {
                                            node.gradient = Some(add_grad(node.gradient.clone(), local)?);
                                        }
                                    } else if let Some(Value::Tensor(t)) = val {
                                        let mask: Vec<f64> = t.data.iter().map(|x| if *x > 0.0 { 1.0 } else { 0.0 }).collect();
                                        let local = match gradient.clone() {
                                            Value::Tensor(g) => {
                                                if g.shape != t.shape {
                                                    return Err("ReLU tensor gradient shape mismatch".to_string());
                                                }
                                                Value::Tensor(Tensor { data: g.data.iter().zip(mask.iter()).map(|(g, m)| g * m).collect(), shape: g.shape })
                                            }
                                            Value::Scalar(s) => Value::Tensor(Tensor { data: mask.iter().map(|m| s * m).collect(), shape: t.shape.clone() }),
                                        };
                                        if let Some(node) = self.nodes.get_mut(&inputs[0]) {
                                            node.gradient = Some(add_grad(node.gradient.clone(), local)?);
                                        }
                                    }
                                }
                                "sum" => {
                                    if let Some(v) = val {
                                        match v {
                                            Value::Scalar(_) => {
                                                if let Some(node) = self.nodes.get_mut(&inputs[0]) {
                                                    node.gradient = Some(add_grad(node.gradient.clone(), gradient.clone())?);
                                                }
                                            }
                                            Value::Tensor(t) => {
                                                let g = gradient.as_scalar().ok_or("Expected scalar gradient for sum")?;
                                                let data = vec![g; t.data.len()];
                                                let local = Value::Tensor(Tensor { data, shape: t.shape.clone() });
                                                if let Some(node) = self.nodes.get_mut(&inputs[0]) {
                                                    node.gradient = Some(add_grad(node.gradient.clone(), local)?);
                                                }
                                            }
                                        }
                                    }
                                }
                                "mean" => {
                                    if let Some(v) = val {
                                        match v {
                                            Value::Scalar(_) => {
                                                if let Some(node) = self.nodes.get_mut(&inputs[0]) {
                                                    node.gradient = Some(add_grad(node.gradient.clone(), gradient.clone())?);
                                                }
                                            }
                                            Value::Tensor(t) => {
                                                let g = gradient.as_scalar().ok_or("Expected scalar gradient for mean")?;
                                                let n = t.data.len() as f64;
                                                let each = g / n;
                                                let data = vec![each; t.data.len()];
                                                let local = Value::Tensor(Tensor { data, shape: t.shape.clone() });
                                                if let Some(node) = self.nodes.get_mut(&inputs[0]) {
                                                    node.gradient = Some(add_grad(node.gradient.clone(), local)?);
                                                }
                                            }
                                        }
                                    }
                                }
                                "sin" => {
                                    if let Some(v) = val {
                                        let local = match v {
                                            Value::Scalar(x) => Value::Scalar(gradient.as_scalar().ok_or("Expected scalar gradient for sin")? * x.cos()),
                                            Value::Tensor(t) => match gradient.clone() {
                                                Value::Tensor(g) => {
                                                    if g.shape != t.shape { return Err("sin tensor gradient shape mismatch".to_string()); }
                                                    Value::Tensor(Tensor { data: g.data.iter().zip(t.data.iter()).map(|(g,x)| g * x.cos()).collect(), shape: t.shape })
                                                }
                                                Value::Scalar(s) => Value::Tensor(Tensor { data: t.data.iter().map(|x| s * x.cos()).collect(), shape: t.shape }),
                                            },
                                        };
                                        if let Some(node) = self.nodes.get_mut(&inputs[0]) { node.gradient = Some(add_grad(node.gradient.clone(), local)?); }
                                    }
                                }
                                "cos" => {
                                    if let Some(v) = val {
                                        let local = match v {
                                            Value::Scalar(x) => Value::Scalar(gradient.as_scalar().ok_or("Expected scalar gradient for cos")? * -x.sin()),
                                            Value::Tensor(t) => match gradient.clone() {
                                                Value::Tensor(g) => {
                                                    if g.shape != t.shape { return Err("cos tensor gradient shape mismatch".to_string()); }
                                                    Value::Tensor(Tensor { data: g.data.iter().zip(t.data.iter()).map(|(g,x)| g * -x.sin()).collect(), shape: t.shape })
                                                }
                                                Value::Scalar(s) => Value::Tensor(Tensor { data: t.data.iter().map(|x| s * -x.sin()).collect(), shape: t.shape }),
                                            },
                                        };
                                        if let Some(node) = self.nodes.get_mut(&inputs[0]) { node.gradient = Some(add_grad(node.gradient.clone(), local)?); }
                                    }
                                }
                                "tanh" => {
                                    if let Some(v) = val {
                                        let local = match v {
                                            Value::Scalar(x) => {
                                                let th = x.tanh();
                                                Value::Scalar(gradient.as_scalar().ok_or("Expected scalar gradient for tanh")? * (1.0 - th * th))
                                            }
                                            Value::Tensor(t) => match gradient.clone() {
                                                Value::Tensor(g) => {
                                                    if g.shape != t.shape { return Err("tanh tensor gradient shape mismatch".to_string()); }
                                                    Value::Tensor(Tensor { data: g.data.iter().zip(t.data.iter()).map(|(g,x)| {
                                                        let th = x.tanh();
                                                        g * (1.0 - th * th)
                                                    }).collect(), shape: t.shape })
                                                }
                                                Value::Scalar(s) => Value::Tensor(Tensor { data: t.data.iter().map(|x| {
                                                    let th = x.tanh();
                                                    s * (1.0 - th * th)
                                                }).collect(), shape: t.shape }),
                                            },
                                        };
                                        if let Some(node) = self.nodes.get_mut(&inputs[0]) { node.gradient = Some(add_grad(node.gradient.clone(), local)?); }
                                    }
                                }
                                "exp" => {
                                    if let Some(v) = val {
                                        let local = match v {
                                            Value::Scalar(x) => Value::Scalar(gradient.as_scalar().ok_or("Expected scalar gradient for exp")? * x.exp()),
                                            Value::Tensor(t) => match gradient.clone() {
                                                Value::Tensor(g) => {
                                                    if g.shape != t.shape { return Err("exp tensor gradient shape mismatch".to_string()); }
                                                    Value::Tensor(Tensor { data: g.data.iter().zip(t.data.iter()).map(|(g,x)| g * x.exp()).collect(), shape: t.shape })
                                                }
                                                Value::Scalar(s) => Value::Tensor(Tensor { data: t.data.iter().map(|x| s * x.exp()).collect(), shape: t.shape }),
                                            },
                                        };
                                        if let Some(node) = self.nodes.get_mut(&inputs[0]) { node.gradient = Some(add_grad(node.gradient.clone(), local)?); }
                                    }
                                }
                                "log" => {
                                    if let Some(v) = val {
                                        let local = match v {
                                            Value::Scalar(x) => Value::Scalar(gradient.as_scalar().ok_or("Expected scalar gradient for log")? * (1.0 / x)),
                                            Value::Tensor(t) => match gradient.clone() {
                                                Value::Tensor(g) => {
                                                    if g.shape != t.shape { return Err("log tensor gradient shape mismatch".to_string()); }
                                                    Value::Tensor(Tensor { data: g.data.iter().zip(t.data.iter()).map(|(g,x)| g * (1.0 / x)).collect(), shape: t.shape })
                                                }
                                                Value::Scalar(s) => Value::Tensor(Tensor { data: t.data.iter().map(|x| s * (1.0 / x)).collect(), shape: t.shape }),
                                            },
                                        };
                                        if let Some(node) = self.nodes.get_mut(&inputs[0]) { node.gradient = Some(add_grad(node.gradient.clone(), local)?); }
                                    }
                                }
                                "sqrt" => {
                                    if let Some(v) = val {
                                        let local = match v {
                                            Value::Scalar(x) => Value::Scalar(gradient.as_scalar().ok_or("Expected scalar gradient for sqrt")? * (0.5 / x.sqrt())),
                                            Value::Tensor(t) => match gradient.clone() {
                                                Value::Tensor(g) => {
                                                    if g.shape != t.shape { return Err("sqrt tensor gradient shape mismatch".to_string()); }
                                                    Value::Tensor(Tensor { data: g.data.iter().zip(t.data.iter()).map(|(g,x)| g * (0.5 / x.sqrt())).collect(), shape: t.shape })
                                                }
                                                Value::Scalar(s) => Value::Tensor(Tensor { data: t.data.iter().map(|x| s * (0.5 / x.sqrt())).collect(), shape: t.shape }),
                                            },
                                        };
                                        if let Some(node) = self.nodes.get_mut(&inputs[0]) { node.gradient = Some(add_grad(node.gradient.clone(), local)?); }
                                    }
                                }
                                "abs" => {
                                    if let Some(v) = val {
                                        let local = match v {
                                            Value::Scalar(x) => {
                                                let sign = if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 };
                                                Value::Scalar(gradient.as_scalar().ok_or("Expected scalar gradient for abs")? * sign)
                                            }
                                            Value::Tensor(t) => match gradient.clone() {
                                                Value::Tensor(g) => {
                                                    if g.shape != t.shape { return Err("abs tensor gradient shape mismatch".to_string()); }
                                                    Value::Tensor(Tensor { data: g.data.iter().zip(t.data.iter()).map(|(g,x)| {
                                                        let sign = if *x > 0.0 { 1.0 } else if *x < 0.0 { -1.0 } else { 0.0 };
                                                        g * sign
                                                    }).collect(), shape: t.shape })
                                                }
                                                Value::Scalar(s) => Value::Tensor(Tensor { data: t.data.iter().map(|x| {
                                                    let sign = if *x > 0.0 { 1.0 } else if *x < 0.0 { -1.0 } else { 0.0 };
                                                    s * sign
                                                }).collect(), shape: t.shape }),
                                            },
                                        };
                                        if let Some(node) = self.nodes.get_mut(&inputs[0]) { node.gradient = Some(add_grad(node.gradient.clone(), local)?); }
                                    }
                                }
                                "floor" | "ceil" => {
                                    let zero = match val {
                                        Some(Value::Scalar(_)) => Value::Scalar(0.0),
                                        Some(Value::Tensor(t)) => Value::Tensor(Tensor::zeros(t.shape.clone())),
                                        _ => gradient.clone().zeros_like(),
                                    };
                                    if let Some(node) = self.nodes.get_mut(&inputs[0]) {
                                        node.gradient = Some(add_grad(node.gradient.clone(), zero)?);
                                    }
                                }
                                "print" => {
                                    // Pass-through gradient to the printed value
                                    if let Some(node) = self.nodes.get_mut(&inputs[0]) {
                                        node.gradient = Some(add_grad(node.gradient.clone(), gradient.clone())?);
                                    }
                                }
                                _ => {}
                            }
                        } else {
                            // 2+ argument function calls
                            match name.as_str() {
                                "index" => {
                                    // Gradient w.r.t. target tensor: scatter upstream scalar into the chosen index
                                    let target_val = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone());
                                    if let Some(Value::Tensor(t)) = target_val {
                                        // Recompute linear index from inputs[1..]
                                        let mut idx_vals: Vec<usize> = Vec::new();
                                        for &iid in &inputs[1..] {
                                            let s = self.nodes.get(&iid).and_then(|n| n.value.clone()).and_then(|v| v.as_scalar());
                                            let s = s.ok_or("Index must be scalar")?;
                                            if !s.is_finite() { return Err("Index not finite".to_string()); }
                                            let u = if s >= 0.0 { s as usize } else { return Err("Negative index".to_string()); };
                                            idx_vals.push(u);
                                        }
                                        if idx_vals.len() != t.shape.len() { return Err("Index rank must match tensor rank".to_string()); }
                                        let strides = compute_strides(&t.shape);
                                        let mut linear = 0usize;
                                        for (i, &idx) in idx_vals.iter().enumerate() {
                                            if idx >= t.shape[i] { return Err("Index out of bounds".to_string()); }
                                            linear += idx * strides[i];
                                        }
                                        let g_up = gradient.as_scalar().ok_or("Expected scalar gradient for index result")?;
                                        let mut data = vec![0.0; t.data.len()];
                                        data[linear] = g_up;
                                        let local = Value::Tensor(Tensor { data, shape: t.shape.clone() });
                                        if let Some(node) = self.nodes.get_mut(&inputs[0]) {
                                            node.gradient = Some(add_grad(node.gradient.clone(), local)?);
                                        }
                                    }
                                }
                                "dot" => {
                                    // y = dot(a,b) -> scalar; dy/da = g * b; dy/db = g * a
                                    let a = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone());
                                    let b = self.nodes.get(&inputs[1]).and_then(|n| n.value.clone());
                                    if let (Some(Value::Tensor(ta)), Some(Value::Tensor(tb))) = (a, b) {
                                        if ta.shape.len() != 1 || tb.shape.len() != 1 || ta.shape[0] != tb.shape[0] {
                                            return Err("dot backward shape mismatch".to_string());
                                        }
                                        let g = gradient.as_scalar().ok_or("Expected scalar gradient for dot")?;
                                        let ga = Value::Tensor(Tensor { data: tb.data.iter().map(|&v| g * v).collect(), shape: ta.shape.clone() });
                                        let gb = Value::Tensor(Tensor { data: ta.data.iter().map(|&v| g * v).collect(), shape: tb.shape.clone() });
                                        if let Some(node) = self.nodes.get_mut(&inputs[0]) {
                                            node.gradient = Some(add_grad(node.gradient.clone(), ga)?);
                                        }
                                        if let Some(node) = self.nodes.get_mut(&inputs[1]) {
                                            node.gradient = Some(add_grad(node.gradient.clone(), gb)?);
                                        }
                                    }
                                }
                                "matmul" => {
                                    // Y = A(m,k) @ B(k,n); dA = dY @ B^T ; dB = A^T @ dY
                                    let a = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone());
                                    let b = self.nodes.get(&inputs[1]).and_then(|n| n.value.clone());
                                    if let (Some(Value::Tensor(ta)), Some(Value::Tensor(tb))) = (a, b) {
                                        if ta.shape.len() != 2 || tb.shape.len() != 2 { return Err("matmul backward expects rank-2 tensors".to_string()); }
                                        let (m,k) = (ta.shape[0], ta.shape[1]);
                                        let (k2,n) = (tb.shape[0], tb.shape[1]);
                                        if k != k2 { return Err("matmul backward inner dims mismatch".to_string()); }
                                        let gy = match &gradient {
                                            Value::Tensor(t) => t.clone(),
                                            Value::Scalar(_) => return Err("matmul upstream gradient must be tensor".to_string()),
                                        };
                                        if gy.shape != vec![m,n] { return Err("matmul upstream gradient shape mismatch".to_string()); }
                                        // dA = gy (m,n) @ B^T (n,k) = (m,k)
                                        let bt = transpose_tensor(&tb);
                                        let dA = matmul_tensors_raw(&gy, &bt)?;
                                        // dB = A^T (k,m) @ gy (m,n) = (k,n)
                                        let at = transpose_tensor(&ta);
                                        let dB = matmul_tensors_raw(&at, &gy)?;
                                        if let Some(node) = self.nodes.get_mut(&inputs[0]) {
                                            node.gradient = Some(add_grad(node.gradient.clone(), Value::Tensor(dA))?);
                                        }
                                        if let Some(node) = self.nodes.get_mut(&inputs[1]) {
                                            node.gradient = Some(add_grad(node.gradient.clone(), Value::Tensor(dB))?);
                                        }
                                    }
                                }
                                "matvec" => {
                                    // y = A(m,k) @ x(k); dA_ij = g_i * x_j ; dx_j = sum_i g_i * A_ij
                                    let a = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone());
                                    let x = self.nodes.get(&inputs[1]).and_then(|n| n.value.clone());
                                    if let (Some(Value::Tensor(ta)), Some(Value::Tensor(tx))) = (a, x) {
                                        if ta.shape.len() != 2 || tx.shape.len() != 1 { return Err("matvec backward rank mismatch".to_string()); }
                                        let (m,k) = (ta.shape[0], ta.shape[1]);
                                        if tx.shape[0] != k { return Err("matvec backward inner mismatch".to_string()); }
                                        let gy = match &gradient { Value::Tensor(t) => t.clone(), _ => return Err("matvec upstream grad must be tensor".to_string()) };
                                        if gy.shape != vec![m] { return Err("matvec upstream grad shape mismatch".to_string()); }
                                        // dA
                                        let mut dA = vec![0.0; m*k];
                                        for i in 0..m { for j in 0..k { dA[i*k + j] = gy.data[i] * tx.data[j]; } }
                                        // dx
                                        let mut dx = vec![0.0; k];
                                        for j in 0..k { let mut s = 0.0; for i in 0..m { s += gy.data[i] * ta.data[i*k + j]; } dx[j] = s; }
                                        if let Some(node) = self.nodes.get_mut(&inputs[0]) { node.gradient = Some(add_grad(node.gradient.clone(), Value::Tensor(Tensor { data: dA, shape: ta.shape.clone() }))?); }
                                        if let Some(node) = self.nodes.get_mut(&inputs[1]) { node.gradient = Some(add_grad(node.gradient.clone(), Value::Tensor(Tensor { data: dx, shape: tx.shape.clone() }))?); }
                                    }
                                }
                                "vecmat" => {
                                    // y = x(m) @ B(m,n); dB_ij = x_i * g_j ; dx_i = sum_j g_j * B_ij
                                    let x = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone());
                                    let b = self.nodes.get(&inputs[1]).and_then(|n| n.value.clone());
                                    if let (Some(Value::Tensor(tx)), Some(Value::Tensor(tb))) = (x, b) {
                                        if tx.shape.len() != 1 || tb.shape.len() != 2 { return Err("vecmat backward rank mismatch".to_string()); }
                                        let (m,n) = (tb.shape[0], tb.shape[1]);
                                        if tx.shape[0] != m { return Err("vecmat backward inner mismatch".to_string()); }
                                        let gy = match &gradient { Value::Tensor(t) => t.clone(), _ => return Err("vecmat upstream grad must be tensor".to_string()) };
                                        if gy.shape != vec![n] { return Err("vecmat upstream grad shape mismatch".to_string()); }
                                        // dB
                                        let mut dB = vec![0.0; m*n];
                                        for i in 0..m { for j in 0..n { dB[i*n + j] = tx.data[i] * gy.data[j]; } }
                                        // dx
                                        let mut dx = vec![0.0; m];
                                        for i in 0..m { let mut s = 0.0; for j in 0..n { s += gy.data[j] * tb.data[i*n + j]; } dx[i] = s; }
                                        if let Some(node) = self.nodes.get_mut(&inputs[0]) { node.gradient = Some(add_grad(node.gradient.clone(), Value::Tensor(Tensor { data: dx, shape: tx.shape.clone() }))?); }
                                        if let Some(node) = self.nodes.get_mut(&inputs[1]) { node.gradient = Some(add_grad(node.gradient.clone(), Value::Tensor(Tensor { data: dB, shape: tb.shape.clone() }))?); }
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    NodeType::HeapTensor(_) => {
                        // HeapTensors accumulate gradients like Learnables
                        // Nothing special needed here - gradient is already set
                    }
                    NodeType::FreedTensor(name) => {
                        return Err(format!("Cannot compute gradient for freed tensor '{}'", name));
                    }
                }
            }
        }

        Ok(())
    }

    pub fn optimize_step(&mut self, learning_rate: f64) -> Result<(), String> {
        let node_ids: Vec<NodeId> = self.nodes.keys().copied().collect();

        for node_id in node_ids {
            if let Some(node) = self.nodes.get(&node_id) {
                if let NodeType::Learnable(_) = &node.node_type {
                    if let (Some(value), Some(gradient)) = (node.value.clone(), node.gradient.clone()) {
                        let updated = match (value, gradient) {
                            (Value::Scalar(v), Value::Scalar(g)) => Value::Scalar(v - learning_rate * g),
                            (Value::Tensor(v), Value::Tensor(g)) => {
                                if v.shape != g.shape {
                                    return Err("Gradient/value tensor shape mismatch".to_string());
                                }
                                let data = v.data.iter().zip(g.data.iter()).map(|(v, g)| v - learning_rate * g).collect();
                                Value::Tensor(Tensor { data, shape: v.shape })
                            }
                            _ => return Err("Mixed scalar/tensor optimization not supported".to_string()),
                        };

                        if let Some(node) = self.nodes.get_mut(&node_id) {
                            let zero = updated.zeros_like();
                            node.value = Some(updated);
                            node.gradient = Some(zero);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    pub fn reset_gradients(&mut self) {
        for node in self.nodes.values_mut() {
            if let Some(val) = node.value.clone() {
                node.gradient = Some(val.zeros_like());
            }
        }
    }
}

impl Default for ComputationalGraph {
    fn default() -> Self {
        Self::new()
    }
}

fn add_grad(current: Option<Value>, delta: Value) -> Result<Value, String> {
    match (current, delta) {
        (None, d) => Ok(d),
        (Some(Value::Scalar(a)), Value::Scalar(b)) => Ok(Value::Scalar(a + b)),
        (Some(Value::Tensor(t1)), Value::Tensor(t2)) => {
            if t1.shape != t2.shape {
                return Err("Gradient tensor shape mismatch".to_string());
            }
            let data = t1.data.iter().zip(t2.data.iter()).map(|(a, b)| a + b).collect();
            Ok(Value::Tensor(Tensor { data, shape: t1.shape }))
        }
        _ => Err("Mixed scalar/tensor gradients not supported".to_string()),
    }
}

fn mul_grad(a: Value, b: Value) -> Result<Value, String> {
    a.map2(&b, |x, y| x * y)
}

fn div_value(a: Value, b: Value) -> Result<Value, String> {
    a.map2(&b, |x, y| x / y)
}

fn negate_value(v: Value) -> Value {
    match v {
        Value::Scalar(s) => Value::Scalar(-s),
        Value::Tensor(t) => Value::Tensor(Tensor { data: t.data.iter().map(|x| -x).collect(), shape: t.shape }),
    }
}

fn inv_value(v: &Value) -> Result<Value, String> {
    match v {
        Value::Scalar(s) => Ok(Value::Scalar(1.0 / s)),
        Value::Tensor(t) => Ok(Value::Tensor(Tensor { data: t.data.iter().map(|x| 1.0 / x).collect(), shape: t.shape.clone() })),
    }
}

fn pow_value(a: &Value, b: &Value) -> Result<Value, String> {
    a.map2(b, |x, y| x.powf(y))
}

fn add_const(v: &Value, c: f64) -> Result<Value, String> {
    match v {
        Value::Scalar(s) => Ok(Value::Scalar(s + c)),
        Value::Tensor(t) => Ok(Value::Tensor(Tensor { data: t.data.iter().map(|x| x + c).collect(), shape: t.shape.clone() })),
    }
}

fn ln_value(v: &Value) -> Result<Value, String> {
    match v {
        Value::Scalar(s) => Ok(Value::Scalar(s.ln())),
        Value::Tensor(t) => Ok(Value::Tensor(Tensor { data: t.data.iter().map(|x| x.ln()).collect(), shape: t.shape.clone() })),
    }
}

fn shape_product(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    let ra = a.len();
    let rb = b.len();
    let r = ra.max(rb);
    let mut out = vec![1usize; r];
    for i in 0..r {
        let da = if i < r - ra { 1 } else { a[i - (r - ra)] };
        let db = if i < r - rb { 1 } else { b[i - (r - rb)] };
        if da == db || da == 1 || db == 1 {
            out[i] = da.max(db);
        } else {
            return Err("Broadcast shapes not compatible".to_string());
        }
    }
    Ok(out)
}

fn indices_from_linear(mut idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut indices = vec![0usize; shape.len()];
    for i in (0..shape.len()).rev() {
        let dim = shape[i];
        indices[i] = idx % dim;
        idx /= dim;
    }
    indices
}

fn offset_from_indices(indices: &[usize], strides: &[usize]) -> usize {
    indices.iter().zip(strides.iter()).map(|(i,s)| i * s).sum()
}

fn broadcast_to(t: &Tensor, out_shape: &[usize]) -> Result<Tensor, String> {
    let in_shape = &t.shape;
    let ra = in_shape.len();
    let rb = out_shape.len();
    let r = rb;
    // Check compatibility (rely on broadcast_shapes)
    let _ = broadcast_shapes(in_shape, out_shape)?;

    let in_strides = compute_strides(in_shape);
    let mut data = vec![0.0; shape_product(out_shape)];
    for lin in 0..data.len() {
        let out_idx = indices_from_linear(lin, out_shape);
        // Align ranks from right
        let mut in_idx = vec![0usize; ra];
        for i in 0..r {
            let out_axis = i;
            let in_axis = if i < r - ra { None } else { Some(i - (r - ra)) };
            if let Some(ia) = in_axis {
                let dim_in = in_shape[ia];
                in_idx[ia] = if dim_in == 1 { 0 } else { out_idx[out_axis] };
            }
        }
        let in_off = offset_from_indices(&in_idx, &in_strides);
        data[lin] = t.data[in_off];
    }
    Ok(Tensor { data, shape: out_shape.to_vec() })
}

fn reduce_to_shape(t: &Tensor, target_shape: &[usize]) -> Result<Tensor, String> {
    if &t.shape == target_shape { return Ok(t.clone()); }
    let out = target_shape;
    // Determine axes to reduce: where target_dim == 1 and t_dim > 1, or target rank < t rank
    let rt = t.shape.len();
    let ro = out.len();
    // Map output axes to t axes (align from right)
    let mut reduce_axes = vec![false; rt];
    for i in 0..rt {
        let ta = t.shape[rt - 1 - i];
        let oa = if i < ro { out[ro - 1 - i] } else { 1 };
        if oa == 1 && ta > 1 { reduce_axes[rt - 1 - i] = true; }
        if i >= ro { reduce_axes[rt - 1 - i] = true; }
    }
    // Sum over reduce_axes
    let out_size = shape_product(out);
    let mut out_data = vec![0.0; out_size];
    let out_strides = compute_strides(out);
    let t_strides = compute_strides(&t.shape);
    for lin in 0..shape_product(&t.shape) {
        let idx = indices_from_linear(lin, &t.shape);
        // Build output index by zeroing reduced axes (since target dim is 1), else keep
        let mut out_idx = vec![0usize; ro];
        for i in 0..ro {
            // Corresponding t axis
            let t_axis = rt - 1 - i;
            let dim_o = out[ro - 1 - i];
            let v = if dim_o == 1 { 0 } else { idx[t_axis] };
            out_idx[ro - 1 - i] = v;
        }
        let o_off = offset_from_indices(&out_idx, &out_strides);
        out_data[o_off] += t.data[lin];
    }
    Ok(Tensor { data: out_data, shape: out.to_vec() })
}

fn broadcast_binary(a: &Value, b: &Value, op: &str) -> Result<Value, String> {
    match (a, b) {
        (Value::Scalar(x), Value::Scalar(y)) => Ok(Value::Scalar(match op { "add"=>x+y, "sub"=>x-y, "mul"=>x*y, "div"=>x/y, _=>unreachable!() })),
        (Value::Scalar(x), Value::Tensor(tb)) => {
            let data = tb.data.iter().map(|&y| match op { "add"=>x + y, "sub"=>x - y, "mul"=>x * y, "div"=>x / y, _=>unreachable!() }).collect();
            Ok(Value::Tensor(Tensor { data, shape: tb.shape.clone() }))
        }
        (Value::Tensor(ta), Value::Scalar(y)) => {
            let data = ta.data.iter().map(|&x| match op { "add"=>x + y, "sub"=>x - y, "mul"=>x * y, "div"=>x / y, _=>unreachable!() }).collect();
            Ok(Value::Tensor(Tensor { data, shape: ta.shape.clone() }))
        }
        (Value::Tensor(ta), Value::Tensor(tb)) => {
            let out_shape = broadcast_shapes(&ta.shape, &tb.shape)?;
            if ta.shape == out_shape && tb.shape == out_shape {
                let data = ta.data.iter().zip(tb.data.iter()).map(|(&x,&y)| match op { "add"=>x+y, "sub"=>x-y, "mul"=>x*y, "div"=>x/y, _=>unreachable!() }).collect();
                Ok(Value::Tensor(Tensor { data, shape: out_shape }))
            } else {
                let a_exp = broadcast_to(ta, &out_shape)?;
                let b_exp = broadcast_to(tb, &out_shape)?;
                let data = a_exp.data.iter().zip(b_exp.data.iter()).map(|(&x,&y)| match op { "add"=>x+y, "sub"=>x-y, "mul"=>x*y, "div"=>x/y, _=>unreachable!() }).collect();
                Ok(Value::Tensor(Tensor { data, shape: out_shape }))
            }
        }
    }
}

fn reduce_grad_for_input(upstream: Value, input: &Value, other: &Value, op: &str, left_side: bool) -> Result<Value, String> {
    // Compute raw grad contribution in output shape, then reduce to input shape
    match (upstream, input, other) {
        (Value::Scalar(g), Value::Scalar(_), Value::Scalar(o)) => {
            let gg = match op {
                "add" => g,
                "sub" => if left_side { g } else { -g },
                "mul" => g * *o,
                "div" => if left_side { g / *o } else { -g / (o * o) },
                _ => g,
            };
            Ok(Value::Scalar(gg))
        }
        (Value::Tensor(gy), Value::Scalar(_), Value::Tensor(to)) => {
            // reduce sum of all elements after computing per-op factor
            let out_shape = &gy.shape;
            let o_exp = broadcast_to(to, out_shape)?;
            let data: Vec<f64> = match op {
                "add" => gy.data.clone(),
                "sub" => if left_side { gy.data.clone() } else { gy.data.iter().map(|&v| -v).collect() },
                "mul" => gy.data.iter().zip(o_exp.data.iter()).map(|(&g,&o)| g * o).collect(),
                "div" => if left_side { gy.data.iter().zip(o_exp.data.iter()).map(|(&g,&o)| g / o).collect() } else { gy.data.iter().zip(o_exp.data.iter()).map(|(&g,&o)| -g * 1.0 / (o*o)).collect() },
                _ => gy.data.clone(),
            };
            let sum: f64 = data.iter().sum();
            Ok(Value::Scalar(sum))
        }
        (Value::Tensor(gy), Value::Tensor(ti), Value::Scalar(o)) => {
            // No broadcast on input; just elementwise factor
            let data: Vec<f64> = match op {
                "add" => gy.data.clone(),
                "sub" => if left_side { gy.data.clone() } else { gy.data.iter().map(|&v| -v).collect() },
                "mul" => gy.data.iter().map(|&g| g * *o).collect(),
                "div" => if left_side { gy.data.iter().map(|&g| g / *o).collect() } else { gy.data.iter().map(|&g| -g * 1.0 / (o*o)).collect() },
                _ => gy.data.clone(),
            };
            let red = reduce_to_shape(&Tensor { data, shape: gy.shape.clone() }, &ti.shape)?;
            Ok(Value::Tensor(red))
        }
        (Value::Tensor(gy), Value::Tensor(ti), Value::Tensor(to)) => {
            let out_shape = &gy.shape;
            let o_exp = broadcast_to(to, out_shape)?;
            let data: Vec<f64> = match op {
                "add" => gy.data.clone(),
                "sub" => if left_side { gy.data.clone() } else { gy.data.iter().map(|&v| -v).collect() },
                "mul" => gy.data.iter().zip(o_exp.data.iter()).map(|(&g,&o)| g * o).collect(),
                "div" => if left_side { gy.data.iter().zip(o_exp.data.iter()).map(|(&g,&o)| g / o).collect() } else { gy.data.iter().zip(o_exp.data.iter()).map(|(&g,&o)| -g * 1.0 / (o*o)).collect() },
                _ => gy.data.clone(),
            };
            let red = reduce_to_shape(&Tensor { data, shape: out_shape.clone() }, &ti.shape)?;
            Ok(Value::Tensor(red))
        }
        (Value::Scalar(g), Value::Tensor(ti), Value::Tensor(to)) => {
            // Upstream scalar: first create gy filled with g with output shape determined by broadcasting of inputs
            let out_shape = broadcast_shapes(&ti.shape, &to.shape)?;
            let gy = Tensor { data: vec![g; shape_product(&out_shape)], shape: out_shape.clone() };
            let o_exp = broadcast_to(to, &out_shape)?;
            let data: Vec<f64> = match op {
                "add" => gy.data.clone(),
                "sub" => if left_side { gy.data.clone() } else { gy.data.iter().map(|&v| -v).collect() },
                "mul" => gy.data.iter().zip(o_exp.data.iter()).map(|(&g,&o)| g * o).collect(),
                "div" => if left_side { gy.data.iter().zip(o_exp.data.iter()).map(|(&g,&o)| g / o).collect() } else { gy.data.iter().zip(o_exp.data.iter()).map(|(&g,&o)| -g * 1.0 / (o*o)).collect() },
                _ => gy.data.clone(),
            };
            let red = reduce_to_shape(&Tensor { data, shape: out_shape }, &ti.shape)?;
            Ok(Value::Tensor(red))
        }
        _ => Err("Unsupported gradient configuration".to_string()),
    }
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut stride = 1usize;
    let mut strides = vec![0usize; shape.len()];
    for (i, dim) in shape.iter().enumerate().rev() {
        strides[i] = stride;
        stride *= *dim;
    }
    strides
}

fn transpose_tensor(t: &Tensor) -> Tensor {
    if t.shape.len() != 2 { return t.clone(); }
    let (m,n) = (t.shape[0], t.shape[1]);
    let mut out = vec![0.0; m*n];
    for i in 0..m { for j in 0..n { out[j*m + i] = t.data[i*n + j]; } }
    Tensor { data: out, shape: vec![n, m] }
}

fn matmul_tensors(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if a.shape.len() != 2 || b.shape.len() != 2 { return Err("matmul expects rank-2 tensors".to_string()); }
    let (m,k) = (a.shape[0], a.shape[1]);
    let (k2,n) = (b.shape[0], b.shape[1]);
    if k != k2 { return Err("matmul inner dimensions must match".to_string()); }
    let mut out = vec![0.0; m*n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for p in 0..k {
                s += a.data[i*k + p] * b.data[p*n + j];
            }
            out[i*n + j] = s;
        }
    }
    Ok(Tensor { data: out, shape: vec![m, n] })
}

fn matmul_tensors_raw(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    matmul_tensors(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_constant() {
        let mut graph = ComputationalGraph::new();
        let id = graph.add_constant(5.0);
        assert_eq!(graph.get_node(id).map(|n| &n.node_type), Some(&NodeType::Constant(Value::Scalar(5.0))));
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

        assert_eq!(graph.get_node(a).and_then(|n| n.value.clone()), Some(Value::Scalar(3.0)));
        assert_eq!(graph.get_node(b).and_then(|n| n.value.clone()), Some(Value::Scalar(2.0)));
    }

    #[test]
    fn test_backward_pass_simple() {
        let mut graph = ComputationalGraph::new();

        let x = graph.add_learnable("x".to_string(), 3.0);
        let y = graph.add_binary_op("mul", x, x);

        graph.forward_pass().unwrap();
        assert_eq!(graph.get_node(y).and_then(|n| n.value.clone()), Some(Value::Scalar(9.0)));

        graph.backward_pass(y).unwrap();

        let x_gradient = graph.get_node(x).and_then(|n| n.gradient.clone());
        assert!(x_gradient.is_some());
        match x_gradient.unwrap() {
            Value::Scalar(g) => assert!((g - 6.0).abs() < 0.0001),
            _ => panic!("expected scalar gradient"),
        }
    }

    #[test]
    fn test_gradient_descent() {
        let mut graph = ComputationalGraph::new();

        let x = graph.add_learnable("x".to_string(), 5.0);
        let y = graph.add_binary_op("mul", x, x);

        for _ in 0..10 {
            graph.forward_pass().unwrap();
            graph.backward_pass(y).unwrap();
            graph.optimize_step(0.1).unwrap();
            graph.reset_gradients();
        }

        let final_x = graph.get_node(x).and_then(|n| n.value.clone());
        assert!(final_x.is_some());
        if let Value::Scalar(v) = final_x.unwrap() {
            assert!(v.abs() < 5.0);
        } else {
            panic!("expected scalar value");
        }
    }

    #[test]
    fn test_mod_and_or_forward() {
        let mut graph = ComputationalGraph::new();
        let a = graph.add_constant(5.0);
        let b = graph.add_constant(2.0);
        let m = graph.add_binary_op("mod", a, b);
        let o = graph.add_binary_op("or", a, b);
        let z = graph.add_constant(0.0);
        let a2 = graph.add_binary_op("and", a, z);

        graph.forward_pass().unwrap();
        assert_eq!(graph.get_node(m).and_then(|n| n.value.clone()), Some(Value::Scalar(1.0)));
        assert_eq!(graph.get_node(o).and_then(|n| n.value.clone()), Some(Value::Scalar(1.0)));
        assert_eq!(graph.get_node(a2).and_then(|n| n.value.clone()), Some(Value::Scalar(0.0)));
    }

    #[test]
    fn test_tensor_forward_backward() {
        let mut graph = ComputationalGraph::new();
        let t = graph.add_constant_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let l = graph.add_learnable_tensor("w".to_string(), vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]).unwrap();
        let m = graph.add_binary_op("mul", t, l);

        graph.forward_pass().unwrap();
        if let Some(Value::Tensor(val)) = graph.get_node(m).and_then(|n| n.value.clone()) {
            assert_eq!(val.data, vec![1.0, 2.0, 3.0, 4.0]);
        } else {
            panic!("expected tensor value");
        }

        graph.backward_pass(m).unwrap();
        if let Some(Value::Tensor(g)) = graph.get_node(l).and_then(|n| n.gradient.clone()) {
            assert_eq!(g.data, vec![1.0, 2.0, 3.0, 4.0]);
        } else {
            panic!("expected tensor gradient");
        }
    }

    #[test]
    fn test_tensor_broadcast_add() {
        let mut graph = ComputationalGraph::new();
        let t = graph.add_constant_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let c = graph.add_constant(1.5);
        let s = graph.add_binary_op("add", t, c);

        graph.forward_pass().unwrap();
        if let Some(Value::Tensor(val)) = graph.get_node(s).and_then(|n| n.value.clone()) {
            assert_eq!(val.data, vec![2.5, 3.5, 4.5, 5.5]);
            assert_eq!(val.shape, vec![2, 2]);
        } else {
            panic!("expected tensor value");
        }
    }

    #[test]
    fn test_tensor_sigmoid_backward() {
        let mut graph = ComputationalGraph::new();
        let x = graph.add_learnable_tensor("x".to_string(), vec![0.0, 1.0, -1.0, 2.0], vec![2, 2]).unwrap();
        let y = graph.add_function_call("sigmoid".to_string(), vec![x]);

        graph.forward_pass().unwrap();
        graph.backward_pass(y).unwrap();

        // gradient at x should be sigmoid(x) * (1 - sigmoid(x))
        if let Some(Value::Tensor(g)) = graph.get_node(x).and_then(|n| n.gradient.clone()) {
            let xs: Vec<f64> = vec![0.0_f64, 1.0_f64, -1.0_f64, 2.0_f64];
            let s: Vec<f64> = xs.iter().map(|&v| 1.0_f64 / (1.0_f64 + (-v).exp())).collect();
            let expected: Vec<f64> = s.iter().map(|&sv| sv * (1.0 - sv)).collect();
            for (a, b) in g.data.iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-9);
            }
        } else {
            panic!("expected tensor gradient");
        }
    }

    #[test]
    fn test_backward_linear_affine() {
        // y = (x * 2) + 5 => dy/dx = 2
        let mut graph = ComputationalGraph::new();
        let x = graph.add_learnable("x".to_string(), 3.0);
        let c2 = graph.add_constant(2.0);
        let c5 = graph.add_constant(5.0);
        let mul = graph.add_binary_op("mul", x, c2);
        let y = graph.add_binary_op("add", mul, c5);

        graph.forward_pass().unwrap();
        graph.backward_pass(y).unwrap();

        let x_gradient = graph.get_node(x).and_then(|n| n.gradient.clone());
        assert!(x_gradient.is_some());
        match x_gradient.unwrap() {
            Value::Scalar(g) => assert!((g - 2.0).abs() < 1e-9),
            _ => panic!("expected scalar gradient"),
        }
    }
}
