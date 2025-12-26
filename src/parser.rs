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
            _ => Err(NomaError::ParseError {
                message: "Expected 'fn' or 'struct'".to_string(),
                line: self.peek().line,
                column: self.peek().column,
            }),
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
            TokenType::Return => self.parse_return(),
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
                _ => break,
            }
        }

        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expression, NomaError> {
        match self.peek().token_type {
            TokenType::Number(n) => {
                self.advance();
                Ok(Expression::Number(n))
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
