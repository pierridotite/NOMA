use crate::error::NomaError;
use crate::token::{Token, TokenType};

/// The NOMA Lexer
/// Converts source code into a stream of tokens
pub struct Lexer {
    source: Vec<char>,
    current: usize,
    line: usize,
    column: usize,
}

impl Lexer {
    pub fn new(source: &str) -> Self {
        Self {
            source: source.chars().collect(),
            current: 0,
            line: 1,
            column: 1,
        }
    }

    pub fn tokenize(&mut self) -> Result<Vec<Token>, NomaError> {
        let mut tokens = Vec::new();

        while !self.is_at_end() {
            self.skip_whitespace();
            if self.is_at_end() {
                break;
            }

            let token = self.next_token()?;
            
            // Skip newlines for now (might be significant later for certain syntaxes)
            if !matches!(token.token_type, TokenType::Newline) {
                tokens.push(token);
            }
        }

        tokens.push(Token::new(TokenType::Eof, self.line, self.column));
        Ok(tokens)
    }

    fn next_token(&mut self) -> Result<Token, NomaError> {
        let start_line = self.line;
        let start_column = self.column;

        let ch = self.advance();

        let token_type = match ch {
            '\n' => {
                self.line += 1;
                self.column = 1;
                TokenType::Newline
            }
            '+' => TokenType::Plus,
            '*' => {
                if self.peek() == '*' {
                    self.advance();
                    TokenType::Power
                } else {
                    TokenType::Star
                }
            }
            '/' => {
                if self.peek() == '/' {
                    // Single-line comment
                    while self.peek() != '\n' && !self.is_at_end() {
                        self.advance();
                    }
                    return self.next_token(); // Skip comment and get next token
                }
                TokenType::Slash
            }
            '%' => TokenType::Percent,
            '(' => TokenType::LParen,
            ')' => TokenType::RParen,
            '{' => TokenType::LBrace,
            '}' => TokenType::RBrace,
            '[' => TokenType::LBracket,
            ']' => TokenType::RBracket,
            ',' => TokenType::Comma,
            ';' => TokenType::Semicolon,
            ':' => TokenType::Colon,
            '.' => TokenType::Dot,
            '-' => {
                if self.peek() == '>' {
                    self.advance();
                    TokenType::Arrow
                } else {
                    TokenType::Minus
                }
            }
            '=' => {
                if self.peek() == '=' {
                    self.advance();
                    TokenType::Equal
                } else {
                    TokenType::Assign
                }
            }
            '<' => {
                if self.peek() == '=' {
                    self.advance();
                    TokenType::LtEq
                } else {
                    TokenType::Lt
                }
            }
            '>' => {
                if self.peek() == '=' {
                    self.advance();
                    TokenType::GtEq
                } else {
                    TokenType::Gt
                }
            }
            '!' => {
                if self.peek() == '=' {
                    self.advance();
                    TokenType::NotEq
                } else {
                    return Err(NomaError::unexpected_char(ch, start_line, start_column));
                }
            }
            '&' => {
                if self.peek() == '&' {
                    self.advance();
                    TokenType::And
                } else {
                    return Err(NomaError::unexpected_char(ch, start_line, start_column));
                }
            }
            '|' => {
                if self.peek() == '|' {
                    self.advance();
                    TokenType::Or
                } else {
                    return Err(NomaError::unexpected_char(ch, start_line, start_column));
                }
            }
            '^' => TokenType::Power,
            '"' => {
                let string = self.read_string()?;
                TokenType::StringLiteral(string)
            }
            _ if ch.is_alphabetic() || ch == '_' => {
                let identifier = self.read_identifier(ch);
                self.keyword_or_identifier(identifier)
            }
            _ if ch.is_ascii_digit() => {
                let number = self.read_number(ch)?;
                TokenType::Number(number)
            }
            _ => {
                return Err(NomaError::unexpected_char(ch, start_line, start_column));
            }
        };

        Ok(Token::new(token_type, start_line, start_column))
    }

    fn read_identifier(&mut self, first: char) -> String {
        let mut identifier = String::from(first);
        
        while !self.is_at_end() {
            let ch = self.peek();
            if ch.is_alphanumeric() || ch == '_' {
                identifier.push(self.advance());
            } else {
                break;
            }
        }

        identifier
    }

    fn read_number(&mut self, first: char) -> Result<f64, NomaError> {
        let mut number = String::from(first);
        let start_line = self.line;
        let start_column = self.column - 1;

        // Read integer part
        while !self.is_at_end() && self.peek().is_ascii_digit() {
            number.push(self.advance());
        }

        // Read decimal part if exists
        if !self.is_at_end() && self.peek() == '.' {
            let next_pos = self.current + 1;
            if next_pos < self.source.len() && self.source[next_pos].is_ascii_digit() {
                number.push(self.advance()); // consume '.'
                while !self.is_at_end() && self.peek().is_ascii_digit() {
                    number.push(self.advance());
                }
            }
        }

        number.parse::<f64>().map_err(|_| NomaError::InvalidNumber {
            line: start_line,
            column: start_column,
        })
    }

    fn read_string(&mut self) -> Result<String, NomaError> {
        let start_line = self.line;
        let start_column = self.column - 1;
        let mut string = String::new();

        while !self.is_at_end() && self.peek() != '"' {
            let ch = self.advance();
            if ch == '\\' && !self.is_at_end() {
                // Handle escape sequences
                let escaped = self.advance();
                match escaped {
                    'n' => string.push('\n'),
                    't' => string.push('\t'),
                    'r' => string.push('\r'),
                    '\\' => string.push('\\'),
                    '"' => string.push('"'),
                    _ => {
                        string.push('\\');
                        string.push(escaped);
                    }
                }
            } else if ch == '\n' {
                self.line += 1;
                self.column = 1;
                string.push(ch);
            } else {
                string.push(ch);
            }
        }

        if self.is_at_end() {
            return Err(NomaError::ParseError {
                message: "Unterminated string literal".to_string(),
                line: start_line,
                column: start_column,
            });
        }

        self.advance(); // consume closing quote
        Ok(string)
    }

    fn keyword_or_identifier(&self, word: String) -> TokenType {
        match word.as_str() {
            "learn" => TokenType::Learn,
            "diff" => TokenType::Diff,
            "fn" => TokenType::Fn,
            "let" => TokenType::Let,
            "mut" => TokenType::Mut,
            "struct" => TokenType::Struct,
            "return" => TokenType::Return,
            "if" => TokenType::If,
            "else" => TokenType::Else,
            "while" => TokenType::While,
            "optimize" => TokenType::Optimize,
            "until" => TokenType::Until,
            "minimize" => TokenType::Minimize,
            "gpu_struct" => TokenType::GpuStruct,
            "tensor" => TokenType::Tensor,
            "alloc" => TokenType::Alloc,
            "free" => TokenType::Free,
            "realloc" => TokenType::Realloc,
            "reset_optimizer" => TokenType::ResetOptimizer,
            "load_csv" => TokenType::LoadCsv,
            "save_csv" => TokenType::SaveCsv,
            "load_safetensors" => TokenType::LoadSafetensors,
            "load_safetensors_named" => TokenType::LoadSafetensorsNamed,
            "save_safetensors" => TokenType::SaveSafetensors,
            "batch" => TokenType::Batch,
            "epoch" => TokenType::Epoch,
            "streaming_adapt" => TokenType::StreamingAdapt,
            "predict" => TokenType::Predict,
            "adapt" => TokenType::Adapt,
            "in" => TokenType::In,
            "as" => TokenType::As,
            _ => TokenType::Identifier(word),
        }
    }

    fn skip_whitespace(&mut self) {
        while !self.is_at_end() {
            let ch = self.peek();
            if ch == ' ' || ch == '\t' || ch == '\r' {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn advance(&mut self) -> char {
        let ch = self.source[self.current];
        self.current += 1;
        if ch != '\n' {
            self.column += 1;
        }
        ch
    }

    fn peek(&self) -> char {
        if self.is_at_end() {
            '\0'
        } else {
            self.source[self.current]
        }
    }

    fn is_at_end(&self) -> bool {
        self.current >= self.source.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keywords() {
        let source = "learn fn optimize tensor";
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize().unwrap();

        assert_eq!(tokens[0].token_type, TokenType::Learn);
        assert_eq!(tokens[1].token_type, TokenType::Fn);
        assert_eq!(tokens[2].token_type, TokenType::Optimize);
        assert_eq!(tokens[3].token_type, TokenType::Tensor);
    }

    #[test]
    fn test_operators() {
        let source = "+ - * / % ** = == < > <= >= != -> && || ^";
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize().unwrap();

        assert_eq!(tokens[0].token_type, TokenType::Plus);
        assert_eq!(tokens[1].token_type, TokenType::Minus);
        assert_eq!(tokens[2].token_type, TokenType::Star);
        assert_eq!(tokens[3].token_type, TokenType::Slash);
        assert_eq!(tokens[4].token_type, TokenType::Percent);
        assert_eq!(tokens[5].token_type, TokenType::Power);
        assert_eq!(tokens[6].token_type, TokenType::Assign);
        assert_eq!(tokens[7].token_type, TokenType::Equal);
        assert_eq!(tokens[12].token_type, TokenType::NotEq);
        assert_eq!(tokens[13].token_type, TokenType::Arrow);
        assert_eq!(tokens[14].token_type, TokenType::And);
        assert_eq!(tokens[15].token_type, TokenType::Or);
        assert_eq!(tokens[16].token_type, TokenType::Power);
    }

    #[test]
    fn test_numbers() {
        let source = "42 3.14 0.001";
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize().unwrap();

        assert_eq!(tokens[0].token_type, TokenType::Number(42.0));
        assert_eq!(tokens[1].token_type, TokenType::Number(3.14));
        assert_eq!(tokens[2].token_type, TokenType::Number(0.001));
    }

    #[test]
    fn test_identifiers() {
        let source = "x my_var weight_matrix";
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize().unwrap();

        assert_eq!(tokens[0].token_type, TokenType::Identifier("x".to_string()));
        assert_eq!(tokens[1].token_type, TokenType::Identifier("my_var".to_string()));
        assert_eq!(tokens[2].token_type, TokenType::Identifier("weight_matrix".to_string()));
    }

    #[test]
    fn test_comments() {
        let source = "learn x // this is a comment\nlet y";
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize().unwrap();

        assert_eq!(tokens[0].token_type, TokenType::Learn);
        assert_eq!(tokens[1].token_type, TokenType::Identifier("x".to_string()));
        assert_eq!(tokens[2].token_type, TokenType::Let);
        assert_eq!(tokens[3].token_type, TokenType::Identifier("y".to_string()));
    }
}
