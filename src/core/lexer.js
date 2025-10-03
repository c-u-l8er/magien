/**
 * Zapp Lexer - Tokenizes source code
 * Elixir-inspired syntax with GPU considerations
 */

const TokenType = {
  // Keywords
  DEF: 'DEF',
  DEFMACRO: 'DEFMACRO',
  DEFACTOR: 'DEFACTOR',
  DEFPLANET: 'DEFPLANET',
  DEFSTAR: 'DEFSTAR',
  DO: 'DO',
  END: 'END',
  CASE: 'CASE',
  IF: 'IF',
  WHEN: 'WHEN',
  THEN: 'THEN',
  ELSE: 'ELSE',
  QUOTE: 'QUOTE',
  UNQUOTE: 'UNQUOTE',
  GPU_KERNEL: 'GPU_KERNEL',
  GPU_COMPUTE: 'GPU_COMPUTE',
  
  // Logical operators (as keywords)
  AND: 'AND',
  OR: 'OR',
  NOT: 'NOT',
  
  // Literals
  ATOM: 'ATOM',
  INTEGER: 'INTEGER',
  FLOAT: 'FLOAT',
  STRING: 'STRING',
  BOOLEAN: 'BOOLEAN',
  
  // Identifiers
  IDENTIFIER: 'IDENTIFIER',
  MODULE_IDENTIFIER: 'MODULE_IDENTIFIER',
  
  // Operators
  ARROW: 'ARROW',           // ->
  DOUBLE_COLON: 'DOUBLE_COLON', // ::
  PIPE: 'PIPE',            // |>
  MATCH: 'MATCH',          // =
  PLUS: 'PLUS',
  MINUS: 'MINUS',
  STAR: 'STAR',
  SLASH: 'SLASH',
  
  // Comparison operators
  LT: 'LT',                // <
  GT: 'GT',                // >
  LTE: 'LTE',              // <=
  GTE: 'GTE',              // >=
  EQ: 'EQ',                // ==
  NEQ: 'NEQ',              // !=
  
  // Delimiters
  LPAREN: 'LPAREN',
  RPAREN: 'RPAREN',
  LBRACE: 'LBRACE',
  RBRACE: 'RBRACE',
  LBRACKET: 'LBRACKET',
  RBRACKET: 'RBRACKET',
  COMMA: 'COMMA',
  DOT: 'DOT',
  
  // Special
  NEWLINE: 'NEWLINE',
  EOF: 'EOF',
  ANNOTATION: 'ANNOTATION'  // @gpu_kernel, etc.
};

class Token {
  constructor(type, value, line, column) {
    this.type = type;
    this.value = value;
    this.line = line;
    this.column = column;
  }
}

class Lexer {
  constructor(source) {
    this.source = source;
    this.position = 0;
    this.line = 1;
    this.column = 1;
    this.tokens = [];
  }

  tokenize() {
    while (this.position < this.source.length) {
      this.skipWhitespace();
      
      if (this.position >= this.source.length) break;

      const char = this.currentChar();

      // Comments
      if (char === '#') {
        this.skipComment();
        continue;
      }

      // Annotations
      if (char === '@') {
        this.tokens.push(this.readAnnotation());
        continue;
      }

      // Strings
      if (char === '"' || char === "'") {
        this.tokens.push(this.readString(char));
        continue;
      }

      // Numbers
      if (this.isDigit(char)) {
        this.tokens.push(this.readNumber());
        continue;
      }

      // Atoms
      if (char === ':') {
        this.tokens.push(this.readAtom());
        continue;
      }

      // Identifiers and keywords
      if (this.isAlpha(char) || char === '_') {
        this.tokens.push(this.readIdentifierOrKeyword());
        continue;
      }

      // Module identifiers
      if (this.isUpperCase(char)) {
        this.tokens.push(this.readModuleIdentifier());
        continue;
      }

      // Operators and delimiters
      this.tokens.push(this.readOperatorOrDelimiter());
    }

    this.tokens.push(new Token(TokenType.EOF, null, this.line, this.column));
    return this.tokens;
  }

  currentChar() {
    return this.source[this.position];
  }

  peek(offset = 1) {
    const pos = this.position + offset;
    return pos < this.source.length ? this.source[pos] : null;
  }

  advance() {
    const char = this.currentChar();
    this.position++;
    if (char === '\n') {
      this.line++;
      this.column = 1;
    } else {
      this.column++;
    }
    return char;
  }

  skipWhitespace() {
    while (this.position < this.source.length) {
      const char = this.currentChar();
      if (char === ' ' || char === '\t' || char === '\r') {
        this.advance();
      } else if (char === '\n') {
        this.advance();
        // Newlines are significant in Zapp for line-based syntax
      } else {
        break;
      }
    }
  }

  skipComment() {
    while (this.currentChar() !== '\n' && this.position < this.source.length) {
      this.advance();
    }
  }

  readAnnotation() {
    const startLine = this.line;
    const startColumn = this.column;
    this.advance(); // skip @
    
    let value = '';
    while (this.isAlphaNumeric(this.currentChar()) || this.currentChar() === '_') {
      value += this.advance();
    }

    return new Token(TokenType.ANNOTATION, value, startLine, startColumn);
  }

  readString(quote) {
    const startLine = this.line;
    const startColumn = this.column;
    this.advance(); // skip opening quote
    
    let value = '';
    let escaped = false;

    while (this.position < this.source.length) {
      const char = this.currentChar();
      
      if (escaped) {
        // Handle escape sequences
        const escapeMap = {
          'n': '\n', 't': '\t', 'r': '\r', '\\': '\\',
          '"': '"', "'": "'"
        };
        value += escapeMap[char] || char;
        escaped = false;
        this.advance();
      } else if (char === '\\') {
        escaped = true;
        this.advance();
      } else if (char === quote) {
        this.advance(); // skip closing quote
        break;
      } else {
        value += this.advance();
      }
    }

    return new Token(TokenType.STRING, value, startLine, startColumn);
  }

  readNumber() {
    const startLine = this.line;
    const startColumn = this.column;
    let value = '';
    let isFloat = false;

    while (this.isDigit(this.currentChar()) || this.currentChar() === '.') {
      if (this.currentChar() === '.') {
        if (isFloat) break; // Second dot, stop
        isFloat = true;
      }
      value += this.advance();
    }

    // Scientific notation
    if (this.currentChar() === 'e' || this.currentChar() === 'E') {
      isFloat = true;
      value += this.advance();
      if (this.currentChar() === '+' || this.currentChar() === '-') {
        value += this.advance();
      }
      while (this.isDigit(this.currentChar())) {
        value += this.advance();
      }
    }

    const type = isFloat ? TokenType.FLOAT : TokenType.INTEGER;
    const numValue = isFloat ? parseFloat(value) : parseInt(value, 10);
    return new Token(type, numValue, startLine, startColumn);
  }

  readAtom() {
    const startLine = this.line;
    const startColumn = this.column;
    this.advance(); // skip :
    
    let value = '';
    while (this.isAlphaNumeric(this.currentChar()) || this.currentChar() === '_') {
      value += this.advance();
    }

    return new Token(TokenType.ATOM, value, startLine, startColumn);
  }

  readIdentifierOrKeyword() {
    const startLine = this.line;
    const startColumn = this.column;
    let value = '';

    while (this.isAlphaNumeric(this.currentChar()) || this.currentChar() === '_') {
      value += this.advance();
    }

    // Check for boolean literals
    if (value === 'true' || value === 'false') {
      return new Token(TokenType.BOOLEAN, value === 'true', startLine, startColumn);
    }

    // Check for keywords including logical operators
    const keywords = {
      'def': TokenType.DEF,
      'defmacro': TokenType.DEFMACRO,
      'defactor': TokenType.DEFACTOR,
      'defplanet': TokenType.DEFPLANET,
      'defstar': TokenType.DEFSTAR,
      'do': TokenType.DO,
      'end': TokenType.END,
      'case': TokenType.CASE,
      'if': TokenType.IF,
      'when': TokenType.WHEN,
      'then': TokenType.THEN,
      'else': TokenType.ELSE,
      'quote': TokenType.QUOTE,
      'unquote': TokenType.UNQUOTE,
      'and': TokenType.AND,
      'or': TokenType.OR,
      'not': TokenType.NOT
    };

    const type = keywords[value] || TokenType.IDENTIFIER;
    return new Token(type, value, startLine, startColumn);
  }

  readModuleIdentifier() {
    const startLine = this.line;
    const startColumn = this.column;
    let value = '';

    while (this.isAlphaNumeric(this.currentChar()) || this.currentChar() === '_') {
      value += this.advance();
    }

    return new Token(TokenType.MODULE_IDENTIFIER, value, startLine, startColumn);
  }

  readOperatorOrDelimiter() {
    const startLine = this.line;
    const startColumn = this.column;
    const char = this.currentChar();
    const next = this.peek();

    // Two-character operators
    if (char === '-' && next === '>') {
      this.advance();
      this.advance();
      return new Token(TokenType.ARROW, '->', startLine, startColumn);
    }

    if (char === ':' && next === ':') {
      this.advance();
      this.advance();
      return new Token(TokenType.DOUBLE_COLON, '::', startLine, startColumn);
    }

    if (char === '|' && next === '>') {
      this.advance();
      this.advance();
      return new Token(TokenType.PIPE, '|>', startLine, startColumn);
    }

    if (char === '<' && next === '=') {
      this.advance();
      this.advance();
      return new Token(TokenType.LTE, '<=', startLine, startColumn);
    }

    if (char === '>' && next === '=') {
      this.advance();
      this.advance();
      return new Token(TokenType.GTE, '>=', startLine, startColumn);
    }

    if (char === '=' && next === '=') {
      this.advance();
      this.advance();
      return new Token(TokenType.EQ, '==', startLine, startColumn);
    }

    if (char === '!' && next === '=') {
      this.advance();
      this.advance();
      return new Token(TokenType.NEQ, '!=', startLine, startColumn);
    }

    // Single-character tokens
    const singleChar = {
      '(': TokenType.LPAREN,
      ')': TokenType.RPAREN,
      '{': TokenType.LBRACE,
      '}': TokenType.RBRACE,
      '[': TokenType.LBRACKET,
      ']': TokenType.RBRACKET,
      ',': TokenType.COMMA,
      '.': TokenType.DOT,
      '+': TokenType.PLUS,
      '-': TokenType.MINUS,
      '*': TokenType.STAR,
      '/': TokenType.SLASH,
      '=': TokenType.MATCH,
      '<': TokenType.LT,
      '>': TokenType.GT
    };

    const type = singleChar[char];
    if (type) {
      this.advance();
      return new Token(type, char, startLine, startColumn);
    }

    throw new Error(`Unexpected character: ${char} at ${startLine}:${startColumn}`);
  }

  isDigit(char) {
    return char >= '0' && char <= '9';
  }

  isAlpha(char) {
    return (char >= 'a' && char <= 'z') || (char >= 'A' && char <= 'Z');
  }

  isUpperCase(char) {
    return char >= 'A' && char <= 'Z';
  }

  isAlphaNumeric(char) {
    return this.isAlpha(char) || this.isDigit(char);
  }
}

// Export for both Node.js and ES6 modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { Lexer, Token, TokenType };
}

// ES6 module exports for browser
export { Lexer, Token, TokenType };