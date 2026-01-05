use crate::ast::*;
use crate::error::NomaError;
use crate::token::{Token, TokenType};

/// Parser for the NOMA language
/// Converts a stream of tokens into an Abstract Syntax Tree
pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_optimize_loop() {
        let tokens = vec![
            Token::new(TokenType::Fn, 1, 1),
            Token::new(TokenType::Identifier("main".into()), 1, 4),
            Token::new(TokenType::LParen, 1, 8),
            Token::new(TokenType::RParen, 1, 9),
            Token::new(TokenType::LBrace, 1, 11),
            Token::new(TokenType::Optimize, 2, 1),
            Token::new(TokenType::Identifier("x".into()), 2, 11),
            Token::new(TokenType::Until, 2, 13),
            Token::new(TokenType::Identifier("done".into()), 2, 19),
            Token::new(TokenType::LBrace, 2, 24),
            Token::new(TokenType::Identifier("x".into()), 3, 5),
            Token::new(TokenType::Assign, 3, 7),
            Token::new(TokenType::Number(1.0), 3, 9),
            Token::new(TokenType::Semicolon, 3, 10),
            Token::new(TokenType::RBrace, 4, 1),
            Token::new(TokenType::RBrace, 5, 1),
            Token::new(TokenType::Eof, 5, 2),
        ];

        let mut parser = Parser::new(tokens);
        let program = parser.parse().expect("should parse optimize loop");
        assert_eq!(program.items.len(), 1);
        let func = match &program.items[0] {
            Item::Function(f) => f,
            _ => panic!("expected function"),
        };
        assert!(matches!(func.body[0], Statement::OptimizeLoop { .. }));
    }

    #[test]
    fn parse_power_precedence() {
        // x ^ 2 * 3 should parse as (x ^ 2) * 3
        let tokens = vec![
            Token::new(TokenType::Fn, 1, 1),
            Token::new(TokenType::Identifier("main".into()), 1, 4),
            Token::new(TokenType::LParen, 1, 8),
            Token::new(TokenType::RParen, 1, 9),
            Token::new(TokenType::LBrace, 1, 11),
            Token::new(TokenType::Return, 2, 3),
            Token::new(TokenType::Identifier("x".into()), 2, 10),
            Token::new(TokenType::Power, 2, 12),
            Token::new(TokenType::Number(2.0), 2, 13),
            Token::new(TokenType::Star, 2, 15),
            Token::new(TokenType::Number(3.0), 2, 17),
            Token::new(TokenType::Semicolon, 2, 18),
            Token::new(TokenType::RBrace, 3, 1),
            Token::new(TokenType::Eof, 3, 2),
        ];

        let mut parser = Parser::new(tokens);
        let program = parser.parse().expect("should parse power");
        let func = match &program.items[0] {
            Item::Function(f) => f,
            _ => panic!("expected function"),
        };

        if let Statement::Return(Some(Expression::BinaryOp { left, op, right })) = &func.body[0] {
            assert_eq!(*op, BinaryOperator::Mul);
            if let Expression::BinaryOp { op: pow_op, .. } = &**left {
                assert_eq!(*pow_op, BinaryOperator::Pow);
            } else {
                panic!("expected power on left side");
            }
            assert_eq!(**right, Expression::Number(3.0));
        } else {
            panic!("unexpected return expression shape");
        }
    }
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, current: 0 }
    }

    /// Parse a complete program
    pub fn parse(&mut self) -> Result<Program, NomaError> {
        let mut program = Program::new();

        while !self.is_at_end() {
            let item = self.parse_item()?;
            program.items.push(item);
        }

        Ok(program)
    }

    /// Parse a top-level item (function or struct)
    fn parse_item(&mut self) -> Result<Item, NomaError> {
        match self.peek().token_type {
            TokenType::Fn => self.parse_function(),
            TokenType::Struct => self.parse_struct(),
            _ => {
                // Provide a helpful error message suggesting wrapping in fn main()
                let msg = if !matches!(self.peek().token_type, TokenType::Eof) {
                    format!(
                        "Expected 'fn' or 'struct' at top level, but found {:?}. \
                         All code must be inside a function. Try wrapping your code in: fn main() {{ ... }}",
                        self.peek().token_type
                    )
                } else {
                    "Expected 'fn' or 'struct'".to_string()
                };
                Err(NomaError::ParseError {
                    message: msg,
                    line: self.peek().line,
                    column: self.peek().column,
                })
            }
        }
    }

    /// Parse a function definition
    fn parse_function(&mut self) -> Result<Item, NomaError> {
        self.consume(TokenType::Fn, "Expected 'fn'")?;
        let name = self.parse_identifier("Expected function name")?;
        self.consume(TokenType::LParen, "Expected '('")?;

        let mut params = Vec::new();
        if !matches!(self.peek().token_type, TokenType::RParen) {
            loop {
                params.push(self.parse_identifier("Expected parameter name")?);
                if !matches!(self.peek().token_type, TokenType::Comma) {
                    break;
                }
                self.advance();
            }
        }
        self.consume(TokenType::RParen, "Expected ')'")?;
        self.consume(TokenType::LBrace, "Expected '{'")?;

        let body = self.parse_block()?;

        Ok(Item::Function(FunctionDef {
            name,
            params,
            body,
        }))
    }

    /// Parse a struct definition
    fn parse_struct(&mut self) -> Result<Item, NomaError> {
        self.consume(TokenType::Struct, "Expected 'struct'")?;
        let name = self.parse_identifier("Expected struct name")?;
        self.consume(TokenType::LBrace, "Expected '{'")?;

        let mut fields = Vec::new();
        while !matches!(self.peek().token_type, TokenType::RBrace) && !self.is_at_end() {
            let field_name = self.parse_identifier("Expected field name")?;
            self.consume(TokenType::Colon, "Expected ':'")?;
            let field_type = self.parse_identifier("Expected field type")?;
            fields.push((field_name, field_type));

            if matches!(self.peek().token_type, TokenType::Comma) {
                self.advance();
            }
        }

        self.consume(TokenType::RBrace, "Expected '}'")?;

        Ok(Item::Struct(StructDef { name, fields }))
    }

    /// Parse a block of statements
    fn parse_block(&mut self) -> Result<Vec<Statement>, NomaError> {
        let mut statements = Vec::new();

        while !matches!(self.peek().token_type, TokenType::RBrace) && !self.is_at_end() {
            statements.push(self.parse_statement()?);
        }

        self.consume(TokenType::RBrace, "Expected '}'")?;
        Ok(statements)
    }

    /// Parse a single statement
    fn parse_statement(&mut self) -> Result<Statement, NomaError> {
        match self.peek().token_type {
            TokenType::LBrace => {
                self.advance();
                let block = self.parse_block()?;
                Ok(Statement::Block(block))
            }
            TokenType::Learn => self.parse_learn_declaration(),
            TokenType::Let => self.parse_let_declaration(),
            TokenType::Optimize => self.parse_optimize_loop(),
            TokenType::Minimize => self.parse_minimize(),
            TokenType::If => self.parse_if_statement(),
            TokenType::While => self.parse_while_statement(),
            TokenType::Return => self.parse_return(),
            TokenType::Alloc => self.parse_alloc(),
            TokenType::Free => self.parse_free(),
            TokenType::Realloc => self.parse_realloc(),
            TokenType::ResetOptimizer => self.parse_reset_optimizer(),
            TokenType::LoadCsv => self.parse_load_csv(),
            TokenType::SaveCsv => self.parse_save_csv(),
            TokenType::LoadSafetensors => self.parse_load_safetensors(),
            TokenType::LoadSafetensorsNamed => self.parse_load_safetensors_named(),
            TokenType::SaveSafetensors => self.parse_save_safetensors(),
            TokenType::Batch => self.parse_batch_loop(),
            TokenType::Epoch => self.parse_epoch_loop(),
            TokenType::StreamingAdapt => self.parse_streaming_adapt_loop(),
            _ => {
                // Handle assignment: identifier '=' expr;
                if matches!(self.peek().token_type, TokenType::Identifier(_)) && matches!(self.peek_next().map(|t| &t.token_type), Some(TokenType::Assign)) {
                    self.parse_assignment()
                } else {
                    let expr = self.parse_expression()?;
                    self.consume(TokenType::Semicolon, "Expected ';'")?;
                    Ok(Statement::Expression(expr))
                }
            }
        }
    }

    /// Parse an if/else statement
    fn parse_if_statement(&mut self) -> Result<Statement, NomaError> {
        self.consume(TokenType::If, "Expected 'if'")?;
        let condition = self.parse_expression()?;
        self.consume(TokenType::LBrace, "Expected '{' after if condition")?;
        let then_branch = self.parse_block()?;

        let mut else_branch = Vec::new();
        if matches!(self.peek().token_type, TokenType::Else) {
            self.advance();
            self.consume(TokenType::LBrace, "Expected '{' after else")?;
            else_branch = self.parse_block()?;
        }

        Ok(Statement::If { condition, then_branch, else_branch })
    }

    /// Parse a while loop: while <cond> { body }
    fn parse_while_statement(&mut self) -> Result<Statement, NomaError> {
        self.consume(TokenType::While, "Expected 'while'")?;
        let condition = self.parse_expression()?;
        self.consume(TokenType::LBrace, "Expected '{' after while condition")?;
        let body = self.parse_block()?;
        Ok(Statement::While { condition, body })
    }

    /// Parse 'learn' declaration
    fn parse_learn_declaration(&mut self) -> Result<Statement, NomaError> {
        self.consume(TokenType::Learn, "Expected 'learn'")?;
        let name = self.parse_identifier("Expected variable name")?;
        self.consume(TokenType::Assign, "Expected '='")?;
        let value = self.parse_expression()?;
        self.consume(TokenType::Semicolon, "Expected ';'")?;

        Ok(Statement::LearnDeclaration { name, value })
    }

    /// Parse 'let' declaration
    fn parse_let_declaration(&mut self) -> Result<Statement, NomaError> {
        self.consume(TokenType::Let, "Expected 'let'")?;
        let name = self.parse_identifier("Expected variable name")?;
        self.consume(TokenType::Assign, "Expected '='")?;
        let value = self.parse_expression()?;
        self.consume(TokenType::Semicolon, "Expected ';'")?;

        Ok(Statement::LetDeclaration { name, value })
    }

    /// Parse assignment statement
    fn parse_assignment(&mut self) -> Result<Statement, NomaError> {
        let name = self.parse_identifier("Expected variable name")?;
        self.consume(TokenType::Assign, "Expected '='")?;
        let value = self.parse_expression()?;
        self.consume(TokenType::Semicolon, "Expected ';'")?;

        Ok(Statement::Assignment { name, value })
    }

    /// Parse optimize loop: optimize <target> until <condition> { body }
    fn parse_optimize_loop(&mut self) -> Result<Statement, NomaError> {
        self.consume(TokenType::Optimize, "Expected 'optimize'")?;
        // Support both: optimize target until ... AND optimize(target) until ...
        let target = if matches!(self.peek().token_type, TokenType::LParen) {
            self.advance(); // consume '('
            let t = self.parse_identifier("Expected target to optimize")?;
            self.consume(TokenType::RParen, "Expected ')'")?;
            t
        } else {
            self.parse_identifier("Expected target to optimize")?
        };
        self.consume(TokenType::Until, "Expected 'until'")?;
        let condition = self.parse_expression()?;
        self.consume(TokenType::LBrace, "Expected '{'")?;
        let body = self.parse_block()?;
        Ok(Statement::OptimizeLoop { target, condition, body })
    }

    /// Parse 'minimize' statement
    fn parse_minimize(&mut self) -> Result<Statement, NomaError> {
        self.consume(TokenType::Minimize, "Expected 'minimize'")?;
        let expr = self.parse_expression()?;
        self.consume(TokenType::Semicolon, "Expected ';'")?;

        Ok(Statement::Minimize(expr))
    }

    /// Parse 'return' statement
    fn parse_return(&mut self) -> Result<Statement, NomaError> {
        self.consume(TokenType::Return, "Expected 'return'")?;
        let value = if matches!(self.peek().token_type, TokenType::Semicolon) {
            None
        } else {
            Some(self.parse_expression()?)
        };
        self.consume(TokenType::Semicolon, "Expected ';'")?;

        Ok(Statement::Return(value))
    }

    /// Parse 'alloc' statement: alloc name = [dim1, dim2, ...];
    fn parse_alloc(&mut self) -> Result<Statement, NomaError> {
        self.consume(TokenType::Alloc, "Expected 'alloc'")?;
        let name = self.parse_identifier("Expected variable name")?;
        self.consume(TokenType::Assign, "Expected '='")?;
        self.consume(TokenType::LBracket, "Expected '[' for shape dimensions")?;
        
        let mut shape = Vec::new();
        if !matches!(self.peek().token_type, TokenType::RBracket) {
            loop {
                shape.push(self.parse_expression()?);
                if !matches!(self.peek().token_type, TokenType::Comma) {
                    break;
                }
                self.advance(); // consume comma
            }
        }
        
        self.consume(TokenType::RBracket, "Expected ']'")?;
        self.consume(TokenType::Semicolon, "Expected ';'")?;
        
        Ok(Statement::Alloc { name, shape })
    }

    /// Parse 'free' statement: free name;
    fn parse_free(&mut self) -> Result<Statement, NomaError> {
        self.consume(TokenType::Free, "Expected 'free'")?;
        let name = self.parse_identifier("Expected variable name")?;
        self.consume(TokenType::Semicolon, "Expected ';'")?;
        
        Ok(Statement::Free { name })
    }

    /// Parse 'realloc' statement: realloc name = [dim1, dim2, ...];
    fn parse_realloc(&mut self) -> Result<Statement, NomaError> {
        self.consume(TokenType::Realloc, "Expected 'realloc'")?;
        let name = self.parse_identifier("Expected variable name")?;
        self.consume(TokenType::Assign, "Expected '='")?;
        self.consume(TokenType::LBracket, "Expected '[' for shape dimensions")?;
        
        let mut shape = Vec::new();
        if !matches!(self.peek().token_type, TokenType::RBracket) {
            loop {
                shape.push(self.parse_expression()?);
                if !matches!(self.peek().token_type, TokenType::Comma) {
                    break;
                }
                self.advance(); // consume comma
            }
        }
        
        self.consume(TokenType::RBracket, "Expected ']'")?;
        self.consume(TokenType::Semicolon, "Expected ';'")?;
        
        Ok(Statement::Realloc { name, shape })
    }

    /// Parse 'reset_optimizer' statement: reset_optimizer();
    fn parse_reset_optimizer(&mut self) -> Result<Statement, NomaError> {
        self.consume(TokenType::ResetOptimizer, "Expected 'reset_optimizer'")?;
        self.consume(TokenType::LParen, "Expected '(' after reset_optimizer")?;
        self.consume(TokenType::RParen, "Expected ')'")?;
        self.consume(TokenType::Semicolon, "Expected ';'")?;
        
        Ok(Statement::ResetOptimizer)
    }

    /// Parse 'load_csv' statement: let name = load_csv("path.csv");
    fn parse_load_csv(&mut self) -> Result<Statement, NomaError> {
        self.consume(TokenType::LoadCsv, "Expected 'load_csv'")?;
        let name = self.parse_identifier("Expected variable name")?;
        self.consume(TokenType::Assign, "Expected '='")?;

        let path = self.parse_string_literal("Expected file path string")?;
        self.consume(TokenType::Semicolon, "Expected ';'")?;
        
        Ok(Statement::LoadCsv { name, path })
    }

    /// Parse 'save_csv' statement: save_csv tensor, "path.csv";
    fn parse_save_csv(&mut self) -> Result<Statement, NomaError> {
        self.consume(TokenType::SaveCsv, "Expected 'save_csv'")?;
        let tensor = self.parse_expression()?;
        self.consume(TokenType::Comma, "Expected ',' after tensor expression")?;
        let path = self.parse_string_literal("Expected file path string")?;
        self.consume(TokenType::Semicolon, "Expected ';'")?;
        
        Ok(Statement::SaveCsv { tensor, path })
    }

    /// Parse 'load_safetensors' statement: load_safetensors name = "path.safetensors";
    fn parse_load_safetensors(&mut self) -> Result<Statement, NomaError> {
        self.consume(TokenType::LoadSafetensors, "Expected 'load_safetensors'")?;
        let name = self.parse_identifier("Expected variable name")?;
        self.consume(TokenType::Assign, "Expected '='")?;

        let path = self.parse_string_literal("Expected file path string")?;
        self.consume(TokenType::Semicolon, "Expected ';'")?;
        
        Ok(Statement::LoadSafetensors { name, path })
    }

    /// Parse 'load_safetensors_named' statement: load_safetensors_named name = "path.safetensors" : "tensor_name";
    fn parse_load_safetensors_named(&mut self) -> Result<Statement, NomaError> {
        self.consume(TokenType::LoadSafetensorsNamed, "Expected 'load_safetensors_named'")?;
        let name = self.parse_identifier("Expected variable name")?;
        self.consume(TokenType::Assign, "Expected '='")?;

        let path = self.parse_string_literal("Expected file path string")?;
        self.consume(TokenType::Colon, "Expected ':' before tensor name")?;
        let tensor_name = self.parse_string_literal("Expected tensor name string")?;
        self.consume(TokenType::Semicolon, "Expected ';'")?;
        
        Ok(Statement::LoadSafetensorsNamed { name, path, tensor_name })
    }

    /// Parse 'save_safetensors' statement: save_safetensors { name1: tensor1, name2: tensor2 }, "path.safetensors";
    fn parse_save_safetensors(&mut self) -> Result<Statement, NomaError> {
        self.consume(TokenType::SaveSafetensors, "Expected 'save_safetensors'")?;
        self.consume(TokenType::LBrace, "Expected '{' for tensor dictionary")?;
        
        let mut tensors = Vec::new();
        if !matches!(self.peek().token_type, TokenType::RBrace) {
            loop {
                let tensor_name = self.parse_identifier("Expected tensor name")?;
                self.consume(TokenType::Colon, "Expected ':' after tensor name")?;
                let tensor_expr = self.parse_expression()?;
                tensors.push((tensor_name, tensor_expr));
                
                if !matches!(self.peek().token_type, TokenType::Comma) {
                    break;
                }
                self.advance(); // consume comma
            }
        }
        
        self.consume(TokenType::RBrace, "Expected '}'")?;
        self.consume(TokenType::Comma, "Expected ',' before file path")?;
        let path = self.parse_string_literal("Expected file path string")?;
        self.consume(TokenType::Semicolon, "Expected ';'")?;
        
        Ok(Statement::SaveSafetensors { tensors, path })
    }

    /// Parse 'batch' loop: batch item, index in data with batch_size { body }
    /// or: batch item in data with batch_size { body }
    fn parse_batch_loop(&mut self) -> Result<Statement, NomaError> {
        self.consume(TokenType::Batch, "Expected 'batch'")?;
        
        // Parse item name
        let item_name = self.parse_identifier("Expected batch item name")?;
        
        // Check for optional index: batch item, index in ...
        let index_name = if matches!(self.peek().token_type, TokenType::Comma) {
            self.advance(); // consume comma
            Some(self.parse_identifier("Expected batch index name")?)
        } else {
            None
        };
        
        self.consume(TokenType::In, "Expected 'in' after batch variable(s)")?;
        let data = self.parse_expression()?;
        
        // Parse 'with batch_size'
        if !matches!(self.peek().token_type, TokenType::Identifier(ref s) if s == "with") {
            return Err(NomaError::ParseError {
                message: "Expected 'with' for batch size".to_string(),
                line: self.peek().line,
                column: self.peek().column,
            });
        }
        self.advance(); // consume 'with'
        let batch_size = self.parse_expression()?;
        
        self.consume(TokenType::LBrace, "Expected '{' after batch header")?;
        let body = self.parse_block()?;
        
        Ok(Statement::BatchLoop {
            item_name,
            index_name,
            data,
            batch_size,
            body,
        })
    }

    /// Parse epoch training loop: epoch N batch X, Y with batch_size -> x_batch, y_batch { body }
    /// This runs N epochs where each epoch processes all data in mini-batches with weight updates per batch
    fn parse_epoch_loop(&mut self) -> Result<Statement, NomaError> {
        self.consume(TokenType::Epoch, "Expected 'epoch'")?;
        
        // Parse number of epochs
        let epochs = self.parse_expression()?;
        
        // Expect 'batch' keyword
        self.consume(TokenType::Batch, "Expected 'batch' after epoch count")?;
        
        // Parse X data expression
        let x_data = self.parse_expression()?;
        
        // Expect comma
        self.consume(TokenType::Comma, "Expected ',' between X and Y data")?;
        
        // Parse Y data expression
        let y_data = self.parse_expression()?;
        
        // Parse 'with batch_size'
        if !matches!(self.peek().token_type, TokenType::Identifier(ref s) if s == "with") {
            return Err(NomaError::ParseError {
                message: "Expected 'with' for batch size".to_string(),
                line: self.peek().line,
                column: self.peek().column,
            });
        }
        self.advance(); // consume 'with'
        
        // Parse batch_size as primary expression only (number or identifier)
        let batch_size = self.parse_primary()?;
        
        // Expect '->' arrow for batch variable names
        self.consume(TokenType::Arrow, "Expected '->' before batch variable names")?;
        
        // Parse x_batch name
        let x_batch_name = self.parse_identifier("Expected x_batch variable name")?;
        
        // Expect comma
        self.consume(TokenType::Comma, "Expected ',' between batch variable names")?;
        
        // Parse y_batch name
        let y_batch_name = self.parse_identifier("Expected y_batch variable name")?;
        
        // Parse body
        self.consume(TokenType::LBrace, "Expected '{' after epoch header")?;
        let body = self.parse_block()?;
        
        Ok(Statement::EpochLoop {
            epochs,
            x_data,
            y_data,
            batch_size,
            x_batch_name,
            y_batch_name,
            body,
        })
    }

    /// Parse streaming_adapt loop for causal test-time adaptation
    /// Syntax:
    ///   streaming_adapt X, Y with batch_size -> x_batch, y_batch
    ///       predict { body } -> prediction_var
    ///       adapt { body with minimize }
    fn parse_streaming_adapt_loop(&mut self) -> Result<Statement, NomaError> {
        self.consume(TokenType::StreamingAdapt, "Expected 'streaming_adapt'")?;
        
        // Parse X data expression
        let x_data = self.parse_expression()?;
        
        // Expect comma
        self.consume(TokenType::Comma, "Expected ',' between X and Y data")?;
        
        // Parse Y data expression
        let y_data = self.parse_expression()?;
        
        // Parse 'with batch_size'
        if !matches!(self.peek().token_type, TokenType::Identifier(ref s) if s == "with") {
            return Err(NomaError::ParseError {
                message: "Expected 'with' for batch size".to_string(),
                line: self.peek().line,
                column: self.peek().column,
            });
        }
        self.advance(); // consume 'with'
        
        // Parse batch_size as primary expression only (number or identifier)
        let batch_size = self.parse_primary()?;
        
        // Expect '->' arrow for batch variable names
        self.consume(TokenType::Arrow, "Expected '->' before batch variable names")?;
        
        // Parse x_batch name
        let x_batch_name = self.parse_identifier("Expected x_batch variable name")?;
        
        // Expect comma
        self.consume(TokenType::Comma, "Expected ',' between batch variable names")?;
        
        // Parse y_batch name
        let y_batch_name = self.parse_identifier("Expected y_batch variable name")?;
        
        // Parse 'predict' block
        self.consume(TokenType::Predict, "Expected 'predict' keyword")?;
        self.consume(TokenType::LBrace, "Expected '{' after predict")?;
        let predict_body = self.parse_block()?;
        
        // Parse '->' and prediction output variable name
        self.consume(TokenType::Arrow, "Expected '->' after predict block")?;
        let prediction_output = self.parse_identifier("Expected prediction output variable name")?;
        
        // Parse 'adapt' block
        self.consume(TokenType::Adapt, "Expected 'adapt' keyword")?;
        self.consume(TokenType::LBrace, "Expected '{' after adapt")?;
        let adapt_body = self.parse_block()?;
        
        Ok(Statement::StreamingAdaptLoop {
            x_data,
            y_data,
            batch_size,
            x_batch_name,
            y_batch_name,
            predict_body,
            prediction_output,
            adapt_body,
        })
    }

    /// Helper to parse a string literal
    fn parse_string_literal(&mut self, error_msg: &str) -> Result<String, NomaError> {
        match &self.peek().token_type {
            TokenType::StringLiteral(s) => {
                let s = s.clone();
                self.advance();
                Ok(s)
            }
            _ => Err(NomaError::ParseError {
                message: error_msg.to_string(),
                line: self.peek().line,
                column: self.peek().column,
            }),
        }
    }

    /// Parse an expression with operator precedence
    fn parse_expression(&mut self) -> Result<Expression, NomaError> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Expression, NomaError> {
        let mut expr = self.parse_and()?;
        while matches!(self.peek().token_type, TokenType::Or) {
            self.advance();
            let right = self.parse_and()?;
            expr = Expression::BinaryOp {
                left: Box::new(expr),
                op: BinaryOperator::Or,
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    fn parse_and(&mut self) -> Result<Expression, NomaError> {
        let mut expr = self.parse_equality()?;
        while matches!(self.peek().token_type, TokenType::And) {
            self.advance();
            let right = self.parse_equality()?;
            expr = Expression::BinaryOp {
                left: Box::new(expr),
                op: BinaryOperator::And,
                right: Box::new(right),
            };
        }
        Ok(expr)
    }

    fn parse_equality(&mut self) -> Result<Expression, NomaError> {
        let mut expr = self.parse_comparison()?;

        while let Some(op) = self.match_equality() {
            let right = self.parse_comparison()?;
            expr = Expression::BinaryOp {
                left: Box::new(expr),
                op,
                right: Box::new(right),
            };
        }

        Ok(expr)
    }

    fn parse_comparison(&mut self) -> Result<Expression, NomaError> {
        let mut expr = self.parse_term()?;

        while let Some(op) = self.match_comparison() {
            let right = self.parse_term()?;
            expr = Expression::BinaryOp {
                left: Box::new(expr),
                op,
                right: Box::new(right),
            };
        }

        Ok(expr)
    }

    fn parse_term(&mut self) -> Result<Expression, NomaError> {
        let mut expr = self.parse_factor()?;

        while let Some(op) = self.match_term() {
            let right = self.parse_factor()?;
            expr = Expression::BinaryOp {
                left: Box::new(expr),
                op,
                right: Box::new(right),
            };
        }

        Ok(expr)
    }

    fn parse_factor(&mut self) -> Result<Expression, NomaError> {
        let mut expr = self.parse_power()?;

        while let Some(op) = self.match_factor() {
            let right = self.parse_power()?;
            expr = Expression::BinaryOp {
                left: Box::new(expr),
                op,
                right: Box::new(right),
            };
        }

        Ok(expr)
    }

    fn parse_power(&mut self) -> Result<Expression, NomaError> {
        let mut expr = self.parse_unary()?;

        while let Some(op) = self.match_power() {
            let right = self.parse_unary()?;
            expr = Expression::BinaryOp {
                left: Box::new(expr),
                op,
                right: Box::new(right),
            };
        }

        Ok(expr)
    }

    fn parse_unary(&mut self) -> Result<Expression, NomaError> {
        match self.peek().token_type {
            TokenType::Minus => {
                self.advance();
                let expr = self.parse_unary()?;
                Ok(Expression::UnaryOp {
                    op: UnaryOperator::Neg,
                    expr: Box::new(expr),
                })
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> Result<Expression, NomaError> {
        let mut expr = self.parse_primary()?;

        loop {
            match self.peek().token_type {
                TokenType::LParen => {
                    self.advance();
                    let mut args = Vec::new();
                    if !matches!(self.peek().token_type, TokenType::RParen) {
                        loop {
                            args.push(self.parse_expression()?);
                            if !matches!(self.peek().token_type, TokenType::Comma) {
                                break;
                            }
                            self.advance();
                        }
                    }
                    self.consume(TokenType::RParen, "Expected ')'")?;

                    if let Expression::Identifier(name) = expr {
                        expr = Expression::Call { name, args };
                    } else {
                        return Err(NomaError::ParseError {
                            message: "Can only call identifiers".to_string(),
                            line: self.peek().line,
                            column: self.peek().column,
                        });
                    }
                }
                TokenType::LBracket => {
                    // Indexing: expr[ index ] ; allow chaining: expr[a][b]
                    self.advance(); // consume '['
                    let index_expr = self.parse_expression()?;
                    self.consume(TokenType::RBracket, "Expected ']' after index expression")?;

                    // If existing expr is already an Index, append; else create new
                    expr = match expr {
                        Expression::Index { target, mut indices } => {
                            indices.push(index_expr);
                            Expression::Index { target, indices }
                        }
                        other => Expression::Index { target: Box::new(other), indices: vec![index_expr] },
                    };
                }
                TokenType::As => {
                    // Type cast: expr as type_name
                    self.advance(); // consume 'as'
                    let target_type = self.parse_identifier("Expected type name after 'as'")?;
                    expr = Expression::Cast {
                        expr: Box::new(expr),
                        target_type,
                    };
                }
                _ => break,
            }
        }

        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expression, NomaError> {
        match self.peek().token_type {
            TokenType::Tensor => {
                self.advance();
                self.parse_tensor_literal()
            }
            TokenType::Number(n) => {
                self.advance();
                Ok(Expression::Number(n))
            }
            TokenType::StringLiteral(ref s) => {
                let s = s.clone();
                self.advance();
                Ok(Expression::StringLiteral(s))
            }
            TokenType::Identifier(ref name) => {
                let name = name.clone();
                self.advance();
                Ok(Expression::Identifier(name))
            }
            TokenType::LParen => {
                self.advance();
                let expr = self.parse_expression()?;
                self.consume(TokenType::RParen, "Expected ')'")?;
                Ok(expr)
            }
            _ => Err(NomaError::ParseError {
                message: format!("Unexpected token: {:?}", self.peek().token_type),
                line: self.peek().line,
                column: self.peek().column,
            }),
        }
    }

    /// Parse a tensor literal: tensor [ 1, 2, 3 ] or tensor [ [1,2], [3,4] ]
    fn parse_tensor_literal(&mut self) -> Result<Expression, NomaError> {
        self.consume(TokenType::LBracket, "Expected '[' after 'tensor'")?;
        let (data, shape) = self.parse_tensor_list()?;
        self.consume(TokenType::RBracket, "Expected ']' to close tensor literal")?;
        Ok(Expression::TensorLiteral { data, shape })
    }

    /// Parses either a flat list of numbers or a nested list of equal-shaped lists.
    /// Assumes opening '[' has already been consumed by caller of parse_tensor_literal.
    fn parse_tensor_list(&mut self) -> Result<(Vec<f64>, Vec<usize>), NomaError> {
        // Detect nested vs flat by peeking: if next is '[' then nested
        let mut elements: Vec<(Vec<f64>, Vec<usize>)> = Vec::new();

        if matches!(self.peek().token_type, TokenType::RBracket) {
            return Err(NomaError::ParseError { message: "Empty tensor literal not allowed".into(), line: self.peek().line, column: self.peek().column });
        }

        loop {
            match self.peek().token_type {
                TokenType::LBracket => {
                    // Nested list
                    self.advance(); // consume '['
                    let (sub_data, sub_shape) = self.parse_tensor_list()?;
                    self.consume(TokenType::RBracket, "Expected ']' in nested tensor list")?;
                    elements.push((sub_data, sub_shape));
                }
                TokenType::Number(_) | TokenType::Minus => {
                    // Flat number, parse a sequence of numbers separated by commas
                    let mut data = Vec::new();
                    // We are inside the outermost '[', so collect until ']' or ', [ ... ]' would have been handled above
                    loop {
                        let v = self.parse_number_literal()?;
                        data.push(v);

                        if matches!(self.peek().token_type, TokenType::Comma) {
                            self.advance();
                            if matches!(self.peek().token_type, TokenType::RBracket) {
                                return Err(NomaError::ParseError { message: "Trailing comma in tensor literal".into(), line: self.peek().line, column: self.peek().column });
                            }
                            continue;
                        }
                        break;
                    }
                    let len = data.len();
                    return Ok((data, vec![len]));
                }
                _ => {
                    return Err(NomaError::ParseError { message: "Expected '[' or number in tensor literal".into(), line: self.peek().line, column: self.peek().column });
                }
            }

            // After each element, expect either ',' to continue or ']' handled by caller
            if matches!(self.peek().token_type, TokenType::Comma) {
                self.advance();
                if matches!(self.peek().token_type, TokenType::RBracket) {
                    return Err(NomaError::ParseError { message: "Trailing comma in tensor literal".into(), line: self.peek().line, column: self.peek().column });
                }
            } else {
                // No comma: end of list (next should be RBracket, validated by caller)
                break;
            }
        }

        // Validate rectangular nested lists and compute shape
        if elements.is_empty() {
            return Err(NomaError::ParseError { message: "Empty tensor literal not allowed".into(), line: self.peek().line, column: self.peek().column });
        }

        let first_shape = elements[0].1.clone();
        for (_, sh) in &elements {
            if *sh != first_shape {
                return Err(NomaError::ParseError { message: "Tensor literal must be rectangular".into(), line: self.peek().line, column: self.peek().column });
            }
        }

        let mut data: Vec<f64> = Vec::new();
        for (sub_data, _) in elements {
            data.extend(sub_data);
        }

        let mut shape = Vec::with_capacity(1 + first_shape.len());
        shape.push(data.len() / first_shape.iter().product::<usize>().max(1));
        shape.extend(first_shape);

        Ok((data, shape))
    }

    fn parse_number_literal(&mut self) -> Result<f64, NomaError> {
        let neg = matches!(self.peek().token_type, TokenType::Minus);
        if neg { self.advance(); }
        match self.peek().token_type {
            TokenType::Number(n) => { self.advance(); Ok(if neg { -n } else { n }) }
            _ => Err(NomaError::ParseError { message: "Expected number literal".into(), line: self.peek().line, column: self.peek().column }),
        }
    }

    // Helper methods

    fn match_equality(&mut self) -> Option<BinaryOperator> {
        match self.peek().token_type {
            TokenType::Equal => {
                self.advance();
                Some(BinaryOperator::Equal)
            }
            TokenType::NotEq => {
                self.advance();
                Some(BinaryOperator::NotEqual)
            }
            _ => None,
        }
    }

    fn match_comparison(&mut self) -> Option<BinaryOperator> {
        match self.peek().token_type {
            TokenType::Lt => {
                self.advance();
                Some(BinaryOperator::Less)
            }
            TokenType::Gt => {
                self.advance();
                Some(BinaryOperator::Greater)
            }
            TokenType::LtEq => {
                self.advance();
                Some(BinaryOperator::LessEq)
            }
            TokenType::GtEq => {
                self.advance();
                Some(BinaryOperator::GreaterEq)
            }
            _ => None,
        }
    }

    fn match_term(&mut self) -> Option<BinaryOperator> {
        match self.peek().token_type {
            TokenType::Plus => {
                self.advance();
                Some(BinaryOperator::Add)
            }
            TokenType::Minus => {
                self.advance();
                Some(BinaryOperator::Sub)
            }
            _ => None,
        }
    }

    fn match_factor(&mut self) -> Option<BinaryOperator> {
        match self.peek().token_type {
            TokenType::Star => {
                self.advance();
                Some(BinaryOperator::Mul)
            }
            TokenType::Slash => {
                self.advance();
                Some(BinaryOperator::Div)
            }
            TokenType::Percent => {
                self.advance();
                Some(BinaryOperator::Mod)
            }
            _ => None,
        }
    }

    fn match_power(&mut self) -> Option<BinaryOperator> {
        match self.peek().token_type {
            TokenType::Power => {
                self.advance();
                Some(BinaryOperator::Pow)
            }
            _ => None,
        }
    }

    fn parse_identifier(&mut self, message: &str) -> Result<String, NomaError> {
        match self.peek().token_type {
            TokenType::Identifier(ref name) => {
                let name = name.clone();
                self.advance();
                Ok(name)
            }
            _ => Err(NomaError::ParseError {
                message: message.to_string(),
                line: self.peek().line,
                column: self.peek().column,
            }),
        }
    }

    fn peek_next(&self) -> Option<&Token> {
        self.tokens.get(self.current + 1)
    }

    fn consume(&mut self, token_type: TokenType, message: &str) -> Result<(), NomaError> {
        if std::mem::discriminant(&self.peek().token_type) == std::mem::discriminant(&token_type) {
            self.advance();
            Ok(())
        } else {
            Err(NomaError::ParseError {
                message: message.to_string(),
                line: self.peek().line,
                column: self.peek().column,
            })
        }
    }

    fn peek(&self) -> Token {
        self.tokens.get(self.current)
            .cloned()
            .unwrap_or_else(|| Token::new(TokenType::Eof, 0, 0))
    }

    fn advance(&mut self) {
        if !self.is_at_end() {
            self.current += 1;
        }
    }

    fn is_at_end(&self) -> bool {
        matches!(self.peek().token_type, TokenType::Eof)
    }
}
