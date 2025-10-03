# Zapp Language: Complete Implementation Specification

This document series provides comprehensive specifications for building the Zapp Language from scratch. Each phase is designed for autonomous implementation by generative AI systems.

---

# LAYER 1: Zapp Language Core (JavaScript + WebGPU)

## Document 1.1: Language Fundamentals & Runtime Architecture

**Target**: Generative AI coding from scratch
**Language**: Pure JavaScript (ES2022+) with WebGPU API
**Deliverable**: Self-hosting compiler core that can parse and execute basic Zapp programs

### 1.1.1 Core Architecture Overview

```javascript
// File: src/core/runtime.js
/**
 * Zapp Runtime - Core execution environment
 * Manages WebGPU device, memory, and execution contexts
 */
class ZappRuntime {
  constructor() {
    this.device = null;           // WebGPU device
    this.adapter = null;          // WebGPU adapter
    this.bufferPool = new Map();  // Reusable GPU buffers
    this.modules = new Map();     // Loaded Zapp modules
    this.actors = new Map();      // Active actor instances
  }

  async initialize() {
    // Initialize WebGPU
    if (!navigator.gpu) {
      throw new Error('WebGPU not supported');
    }

    this.adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance'
    });

    this.device = await this.adapter.requestDevice({
      requiredFeatures: ['shader-f16'],
      requiredLimits: {
        maxStorageBufferBindingSize: 1024 * 1024 * 1024, // 1GB
        maxComputeWorkgroupStorageSize: 16384,
        maxComputeInvocationsPerWorkgroup: 256,
        maxComputeWorkgroupsPerDimension: 65535
      }
    });

    // Handle device loss
    this.device.lost.then((info) => {
      console.error('WebGPU device lost:', info);
      this.reinitialize();
    });

    return this;
  }

  createBuffer(size, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST) {
    return this.device.createBuffer({
      size: Math.ceil(size / 4) * 4, // Align to 4 bytes
      usage,
      mappedAtCreation: false
    });
  }

  createComputePipeline(shaderCode, entryPoint = 'main') {
    const shaderModule = this.device.createShaderModule({
      code: shaderCode,
      label: `Zapp Compute Shader: ${entryPoint}`
    });

    return this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint
      }
    });
  }

  async executeCompute(pipeline, bindGroups, workgroupCounts) {
    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    
    passEncoder.setPipeline(pipeline);
    bindGroups.forEach((bg, index) => {
      passEncoder.setBindGroup(index, bg);
    });
    
    passEncoder.dispatchWorkgroups(...workgroupCounts);
    passEncoder.end();

    this.device.queue.submit([commandEncoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
  }
}
```

### 1.1.2 Lexer Implementation

```javascript
// File: src/core/lexer.js
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
  QUOTE: 'QUOTE',
  UNQUOTE: 'UNQUOTE',
  GPU_KERNEL: 'GPU_KERNEL',
  GPU_COMPUTE: 'GPU_COMPUTE',
  
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

    // Check for keywords
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
      'quote': TokenType.QUOTE,
      'unquote': TokenType.UNQUOTE
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
      '=': TokenType.MATCH
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

// Export
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { Lexer, Token, TokenType };
}
```

### 1.1.3 AST Definitions

```javascript
// File: src/core/ast.js
/**
 * Abstract Syntax Tree node definitions
 * All nodes have source location for error reporting
 */

class ASTNode {
  constructor(type, location) {
    this.type = type;
    this.location = location; // { line, column }
  }
}

// Module and definitions
class Module extends ASTNode {
  constructor(name, body, location) {
    super('Module', location);
    this.name = name;
    this.body = body; // Array of definitions
  }
}

class FunctionDef extends ASTNode {
  constructor(name, params, guards, body, annotations, location) {
    super('FunctionDef', location);
    this.name = name;
    this.params = params; // Array of patterns
    this.guards = guards; // Optional guard expressions
    this.body = body;
    this.annotations = annotations; // @gpu_kernel, etc.
  }
}

class MacroDef extends ASTNode {
  constructor(name, params, body, location) {
    super('MacroDef', location);
    this.name = name;
    this.params = params;
    this.body = body; // AST to be quoted/unquoted
  }
}

class ActorDef extends ASTNode {
  constructor(name, state, handlers, annotations, location) {
    super('ActorDef', location);
    this.name = name;
    this.state = state; // State schema
    this.handlers = handlers; // Message handlers
    this.annotations = annotations;
  }
}

class PlanetDef extends ASTNode {
  constructor(name, orbitals, annotations, location) {
    super('PlanetDef', location);
    this.name = name;
    this.orbitals = orbitals; // Field definitions
    this.annotations = annotations;
  }
}

class StarDef extends ASTNode {
  constructor(name, layers, annotations, location) {
    super('StarDef', location);
    this.name = name;
    this.layers = layers; // Variant definitions
    this.annotations = annotations;
  }
}

// Expressions
class Literal extends ASTNode {
  constructor(value, literalType, location) {
    super('Literal', location);
    this.value = value;
    this.literalType = literalType; // 'integer', 'float', 'string', 'atom', 'boolean'
  }
}

class Identifier extends ASTNode {
  constructor(name, location) {
    super('Identifier', location);
    this.name = name;
  }
}

class BinaryOp extends ASTNode {
  constructor(operator, left, right, location) {
    super('BinaryOp', location);
    this.operator = operator; // '+', '-', '*', '/', etc.
    this.left = left;
    this.right = right;
  }
}

class UnaryOp extends ASTNode {
  constructor(operator, operand, location) {
    super('UnaryOp', location);
    this.operator = operator; // '-', 'not', etc.
    this.operand = operand;
  }
}

class FunctionCall extends ASTNode {
  constructor(func, args, location) {
    super('FunctionCall', location);
    this.func = func; // Can be Identifier or MemberAccess
    this.args = args;
  }
}

class CaseExpr extends ASTNode {
  constructor(scrutinee, clauses, location) {
    super('CaseExpr', location);
    this.scrutinee = scrutinee; // Expression to match
    this.clauses = clauses; // Array of { pattern, guard, body }
  }
}

class IfExpr extends ASTNode {
  constructor(condition, thenBranch, elseBranch, location) {
    super('IfExpr', location);
    this.condition = condition;
    this.thenBranch = thenBranch;
    this.elseBranch = elseBranch;
  }
}

class BlockExpr extends ASTNode {
  constructor(expressions, location) {
    super('BlockExpr', location);
    this.expressions = expressions;
  }
}

class ListExpr extends ASTNode {
  constructor(elements, location) {
    super('ListExpr', location);
    this.elements = elements;
  }
}

class TupleExpr extends ASTNode {
  constructor(elements, location) {
    super('TupleExpr', location);
    this.elements = elements;
  }
}

class MapExpr extends ASTNode {
  constructor(pairs, location) {
    super('MapExpr', location);
    this.pairs = pairs; // Array of { key, value }
  }
}

// Pattern matching
class Pattern extends ASTNode {
  constructor(patternType, value, location) {
    super('Pattern', location);
    this.patternType = patternType; // 'literal', 'variable', 'constructor', 'list', 'tuple'
    this.value = value;
  }
}

// Quote/Unquote for metaprogramming
class QuoteExpr extends ASTNode {
  constructor(expr, location) {
    super('QuoteExpr', location);
    this.expr = expr; // AST node to be quoted
  }
}

class UnquoteExpr extends ASTNode {
  constructor(expr, location) {
    super('UnquoteExpr', location);
    this.expr = expr; // Expression to be evaluated and spliced
  }
}

// GPU-specific nodes
class GPUKernel extends ASTNode {
  constructor(workgroupSize, body, location) {
    super('GPUKernel', location);
    this.workgroupSize = workgroupSize;
    this.body = body;
  }
}

class GPUBufferAlloc extends ASTNode {
  constructor(size, usage, location) {
    super('GPUBufferAlloc', location);
    this.size = size;
    this.usage = usage;
  }
}

// Type annotations
class TypeAnnotation extends ASTNode {
  constructor(baseType, constraints, location) {
    super('TypeAnnotation', location);
    this.baseType = baseType; // 'u32', 'f32', 'Point2D', etc.
    this.constraints = constraints;
  }
}

// Export all node types
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    ASTNode,
    Module,
    FunctionDef,
    MacroDef,
    ActorDef,
    PlanetDef,
    StarDef,
    Literal,
    Identifier,
    BinaryOp,
    UnaryOp,
    FunctionCall,
    CaseExpr,
    IfExpr,
    BlockExpr,
    ListExpr,
    TupleExpr,
    MapExpr,
    Pattern,
    QuoteExpr,
    UnquoteExpr,
    GPUKernel,
    GPUBufferAlloc,
    TypeAnnotation
  };
}
```

### 1.1.4 Parser Implementation

```javascript
// File: src/core/parser.js
/**
 * Recursive descent parser for Zapp language
 * Produces AST from token stream
 */

const { TokenType } = require('./lexer');
const AST = require('./ast');

class ParseError extends Error {
  constructor(message, token) {
    super(`Parse error at ${token.line}:${token.column}: ${message}`);
    this.token = token;
  }
}

class Parser {
  constructor(tokens) {
    this.tokens = tokens;
    this.position = 0;
  }

  parse() {
    const body = [];
    
    while (!this.isAtEnd()) {
      if (this.match(TokenType.EOF)) break;
      body.push(this.parseTopLevel());
    }

    return new AST.Module(null, body, { line: 1, column: 1 });
  }

  parseTopLevel() {
    const annotations = this.parseAnnotations();

    if (this.match(TokenType.DEF)) {
      return this.parseFunctionDef(annotations);
    }

    if (this.match(TokenType.DEFMACRO)) {
      return this.parseMacroDef();
    }

    if (this.match(TokenType.DEFACTOR)) {
      return this.parseActorDef(annotations);
    }

    if (this.match(TokenType.DEFPLANET)) {
      return this.parsePlanetDef(annotations);
    }

    if (this.match(TokenType.DEFSTAR)) {
      return this.parseStarDef(annotations);
    }

    // Standalone expression
    return this.parseExpression();
  }

  parseAnnotations() {
    const annotations = [];
    while (this.check(TokenType.ANNOTATION)) {
      annotations.push(this.advance().value);
    }
    return annotations;
  }

  parseFunctionDef(annotations) {
    const location = this.previous().location;
    const name = this.consume(TokenType.IDENTIFIER, 'Expected function name').value;

    this.consume(TokenType.LPAREN, 'Expected (');
    const params = this.parseParameterList();
    this.consume(TokenType.RPAREN, 'Expected )');

    // Optional guard
    let guards = null;
    if (this.match(TokenType.WHEN)) {
      guards = this.parseExpression();
    }

    this.consume(TokenType.DO, 'Expected do');
    const body = this.parseBlock();
    this.consume(TokenType.END, 'Expected end');

    return new AST.FunctionDef(name, params, guards, body, annotations, location);
  }

  parseMacroDef() {
    const location = this.previous().location;
    const name = this.consume(TokenType.IDENTIFIER, 'Expected macro name').value;

    this.consume(TokenType.LPAREN, 'Expected (');
    const params = this.parseParameterList();
    this.consume(TokenType.RPAREN, 'Expected )');

    this.consume(TokenType.DO, 'Expected do');
    const body = this.parseBlock();
    this.consume(TokenType.END, 'Expected end');

    return new AST.MacroDef(name, params, body, location);
  }

  parseActorDef(annotations) {
    const location = this.previous().location;
    const name = this.consume(TokenType.MODULE_IDENTIFIER, 'Expected actor name').value;

    this.consume(TokenType.DO, 'Expected do');

    // Parse state block
    let state = null;
    if (this.matchIdentifier('state')) {
      this.consume(TokenType.DO, 'Expected do after state');
      state = this.parseStateFields();
      this.consume(TokenType.END, 'Expected end');
    }

    // Parse handlers
    const handlers = [];
    while (this.matchIdentifier('def')) {
      handlers.push(this.parseFunctionDef([]));
    }

    this.consume(TokenType.END, 'Expected end');

    return new AST.ActorDef(name, state, handlers, annotations, location);
  }

  parsePlanetDef(annotations) {
    const location = this.previous().location;
    const name = this.consume(TokenType.MODULE_IDENTIFIER, 'Expected planet name').value;

    this.consume(TokenType.DO, 'Expected do');

    // Parse orbitals block
    this.matchIdentifier('orbitals');
    this.consume(TokenType.DO, 'Expected do');
    const orbitals = this.parseOrbitalFields();
    this.consume(TokenType.END, 'Expected end');

    this.consume(TokenType.END, 'Expected end');

    return new AST.PlanetDef(name, orbitals, annotations, location);
  }

  parseStarDef(annotations) {
    const location = this.previous().location;
    const name = this.consume(TokenType.MODULE_IDENTIFIER, 'Expected star name').value;

    this.consume(TokenType.DO, 'Expected do');

    // Parse layers block
    this.matchIdentifier('layers');
    this.consume(TokenType.DO, 'Expected do');
    const layers = this.parseLayerVariants();
    this.consume(TokenType.END, 'Expected end');

    this.consume(TokenType.END, 'Expected end');

    return new AST.StarDef(name, layers, annotations, location);
  }

  parseParameterList() {
    const params = [];
    
    if (this.check(TokenType.RPAREN)) {
      return params;
    }

    do {
      params.push(this.parsePattern());
    } while (this.match(TokenType.COMMA));

    return params;
  }

  parsePattern() {
    const location = this.current().location;

    // Variable pattern
    if (this.check(TokenType.IDENTIFIER)) {
      const name = this.advance().value;
      
      // Type annotation
      if (this.match(TokenType.DOUBLE_COLON)) {
        const typeAnnotation = this.parseTypeAnnotation();
        return new AST.Pattern('variable', { name, type: typeAnnotation }, location);
      }

      return new AST.Pattern('variable', { name }, location);
    }

    // Literal pattern
    if (this.check(TokenType.INTEGER) || this.check(TokenType.FLOAT) || 
        this.check(TokenType.STRING) || this.check(TokenType.ATOM) ||
        this.check(TokenType.BOOLEAN)) {
      const token = this.advance();
      return new AST.Pattern('literal', token.value, location);
    }

    // List pattern
    if (this.match(TokenType.LBRACKET)) {
      const elements = [];
      
      if (!this.check(TokenType.RBRACKET)) {
        do {
          elements.push(this.parsePattern());
        } while (this.match(TokenType.COMMA));
      }

      this.consume(TokenType.RBRACKET, 'Expected ]');
      return new AST.Pattern('list', elements, location);
    }

    // Tuple pattern
    if (this.match(TokenType.LBRACE)) {
      const elements = [];
      
      do {
        elements.push(this.parsePattern());
      } while (this.match(TokenType.COMMA));

      this.consume(TokenType.RBRACE, 'Expected }');
      return new AST.Pattern('tuple', elements, location);
    }

    // Constructor pattern (for sum types)
    if (this.check(TokenType.MODULE_IDENTIFIER)) {
      const constructor = this.advance().value;
      const args = [];

      if (this.match(TokenType.LPAREN)) {
        if (!this.check(TokenType.RPAREN)) {
          do {
            args.push(this.parsePattern());
          } while (this.match(TokenType.COMMA));
        }
        this.consume(TokenType.RPAREN, 'Expected )');
      }

      return new AST.Pattern('constructor', { constructor, args }, location);
    }

    throw new ParseError('Expected pattern', this.current());
  }

  parseTypeAnnotation() {
    const location = this.current().location;
    
    // Basic types: u32, f32, i32, bool, Point2D, etc.
    if (this.check(TokenType.IDENTIFIER) || this.check(TokenType.MODULE_IDENTIFIER)) {
      const baseType = this.advance().value;
      
      // Generic type parameters
      if (this.match(TokenType.LPAREN)) {
        const params = [];
        do {
          params.push(this.parseTypeAnnotation());
        } while (this.match(TokenType.COMMA));
        this.consume(TokenType.RPAREN, 'Expected )');
        
        return new AST.TypeAnnotation(baseType, { params }, location);
      }

      return new AST.TypeAnnotation(baseType, {}, location);
    }

    throw new ParseError('Expected type annotation', this.current());
  }

  parseStateFields() {
    const fields = [];

    while (!this.check(TokenType.END)) {
      const name = this.consume(TokenType.IDENTIFIER, 'Expected field name').value;
      this.consume(TokenType.DOUBLE_COLON, 'Expected ::');
      const type = this.parseTypeAnnotation();
      
      fields.push({ name, type });
    }

    return fields;
  }

  parseOrbitalFields() {
    const orbitals = [];

    while (!this.check(TokenType.END)) {
      // moon keyword for fields
      this.matchIdentifier('moon');
      
      const name = this.consume(TokenType.IDENTIFIER, 'Expected orbital name').value;
      this.consume(TokenType.DOUBLE_COLON, 'Expected ::');
      const type = this.parseTypeAnnotation();
      
      // Optional field attributes
      const attributes = {};
      while (this.match(TokenType.COMMA)) {
        const attr = this.consume(TokenType.IDENTIFIER, 'Expected attribute name').value;
        this.consume(TokenType.DOUBLE_COLON, 'Expected ::');
        attributes[attr] = this.parsePrimary();
      }

      orbitals.push({ name, type, attributes });
    }

    return orbitals;
  }

  parseLayerVariants() {
    const layers = [];

    while (!this.check(TokenType.END)) {
      // core keyword for variants
      this.matchIdentifier('core');
      
      const constructor = this.consume(TokenType.MODULE_IDENTIFIER, 'Expected variant name').value;
      const fields = [];

      // Optional fields for variant
      if (this.match(TokenType.COMMA)) {
        do {
          const fieldName = this.consume(TokenType.IDENTIFIER, 'Expected field name').value;
          this.consume(TokenType.DOUBLE_COLON, 'Expected ::');
          const fieldType = this.parseTypeAnnotation();
          fields.push({ name: fieldName, type: fieldType });
        } while (this.match(TokenType.COMMA));
      }

      layers.push({ constructor, fields });
    }

    return layers;
  }

  parseBlock() {
    const expressions = [];

    while (!this.check(TokenType.END) && !this.isAtEnd()) {
      expressions.push(this.parseExpression());
    }

    return new AST.BlockExpr(expressions, this.current().location);
  }

  parseExpression() {
    return this.parsePipe();
  }

  parsePipe() {
    let expr = this.parseLogicalOr();

    while (this.match(TokenType.PIPE)) {
      const operator = this.previous();
      const right = this.parseLogicalOr();
      expr = new AST.BinaryOp('|>', expr, right, operator.location);
    }

    return expr;
  }

  parseLogicalOr() {
    let expr = this.parseLogicalAnd();

    while (this.matchIdentifier('or')) {
      const right = this.parseLogicalAnd();
      expr = new AST.BinaryOp('or', expr, right, this.previous().location);
    }

    return expr;
  }

  parseLogicalAnd() {
    let expr = this.parseEquality();

    while (this.matchIdentifier('and')) {
      const right = this.parseEquality();
      expr = new AST.BinaryOp('and', expr, right, this.previous().location);
    }

    return expr;
  }

  parseEquality() {
    let expr = this.parseComparison();

    while (this.matchIdentifier('==') || this.matchIdentifier('!=')) {
      const operator = this.previous().value;
      const right = this.parseComparison();
      expr = new AST.BinaryOp(operator, expr, right, this.previous().location);
    }

    return expr;
  }

  parseComparison() {
    let expr = this.parseAdditive();

    while (this.matchIdentifier('<') || this.matchIdentifier('>') || 
           this.matchIdentifier('<=') || this.matchIdentifier('>=')) {
      const operator = this.previous().value;
      const right = this.parseAdditive();
      expr = new AST.BinaryOp(operator, expr, right, this.previous().location);
    }

    return expr;
  }

  parseAdditive() {
    let expr = this.parseMultiplicative();

    while (this.match(TokenType.PLUS, TokenType.MINUS)) {
      const operator = this.previous().value;
      const right = this.parseMultiplicative();
      expr = new AST.BinaryOp(operator, expr, right, this.previous().location);
    }

    return expr;
  }

  parseMultiplicative() {
    let expr = this.parseUnary();

    while (this.match(TokenType.STAR, TokenType.SLASH)) {
      const operator = this.previous().value;
      const right = this.parseUnary();
      expr = new AST.BinaryOp(operator, expr, right, this.previous().location);
    }

    return expr;
  }

  parseUnary() {
    if (this.match(TokenType.MINUS) || this.matchIdentifier('not')) {
      const operator = this.previous().value;
      const operand = this.parseUnary();
      return new AST.UnaryOp(operator, operand, this.previous().location);
    }

    return this.parsePostfix();
  }

  parsePostfix() {
    let expr = this.parsePrimary();

    while (true) {
      // Function call
      if (this.match(TokenType.LPAREN)) {
        const args = this.parseArgumentList();
        this.consume(TokenType.RPAREN, 'Expected )');
        expr = new AST.FunctionCall(expr, args, this.previous().location);
      }
      // Member access
      else if (this.match(TokenType.DOT)) {
        const member = this.consume(TokenType.IDENTIFIER, 'Expected member name').value;
        expr = new AST.BinaryOp('.', expr, new AST.Identifier(member, this.previous().location), this.previous().location);
      }
      // List/Map access
      else if (this.match(TokenType.LBRACKET)) {
        const index = this.parseExpression();
        this.consume(TokenType.RBRACKET, 'Expected ]');
        expr = new AST.BinaryOp('[]', expr, index, this.previous().location);
      }
      else {
        break;
      }
    }

    return expr;
  }

  parseArgumentList() {
    const args = [];

    if (this.check(TokenType.RPAREN)) {
      return args;
    }

    do {
      args.push(this.parseExpression());
    } while (this.match(TokenType.COMMA));

    return args;
  }

  parsePrimary() {
    const location = this.current().location;

    // Literals
    if (this.check(TokenType.INTEGER)) {
      const value = this.advance().value;
      return new AST.Literal(value, 'integer', location);
    }

    if (this.check(TokenType.FLOAT)) {
      const value = this.advance().value;
      return new AST.Literal(value, 'float', location);
    }

    if (this.check(TokenType.STRING)) {
      const value = this.advance().value;
      return new AST.Literal(value, 'string', location);
    }

    if (this.check(TokenType.ATOM)) {
      const value = this.advance().value;
      return new AST.Literal(value, 'atom', location);
    }

    if (this.check(TokenType.BOOLEAN)) {
      const value = this.advance().value;
      return new AST.Literal(value, 'boolean', location);
    }

    // Identifiers
    if (this.check(TokenType.IDENTIFIER)) {
      const name = this.advance().value;
      return new AST.Identifier(name, location);
    }

    if (this.check(TokenType.MODULE_IDENTIFIER)) {
      const name = this.advance().value;
      return new AST.Identifier(name, location);
    }

    // Parenthesized expression
    if (this.match(TokenType.LPAREN)) {
      const expr = this.parseExpression();
      this.consume(TokenType.RPAREN, 'Expected )');
      return expr;
    }

    // List literal
    if (this.match(TokenType.LBRACKET)) {
      const elements = [];
      
      if (!this.check(TokenType.RBRACKET)) {
        do {
          elements.push(this.parseExpression());
        } while (this.match(TokenType.COMMA));
      }

      this.consume(TokenType.RBRACKET, 'Expected ]');
      return new AST.ListExpr(elements, location);
    }

    // Tuple/Map literal
    if (this.match(TokenType.LBRACE)) {
      // Check if it's a map (has key: value pairs)
      const checkpoint = this.position;
      let isMap = false;

      if (!this.check(TokenType.RBRACE)) {
        this.parseExpression();
        if (this.match(TokenType.DOUBLE_COLON) || this.match(TokenType.ARROW)) {
          isMap = true;
        }
      }

      // Reset to checkpoint
      this.position = checkpoint;

      if (isMap) {
        return this.parseMapLiteral(location);
      } else {
        return this.parseTupleLiteral(location);
      }
    }

    // Case expression
    if (this.match(TokenType.CASE)) {
      return this.parseCaseExpression(location);
    }

    // If expression
    if (this.match(TokenType.IF)) {
      return this.parseIfExpression(location);
    }

    // Quote expression
    if (this.match(TokenType.QUOTE)) {
      this.consume(TokenType.DO, 'Expected do');
      const expr = this.parseExpression();
      this.consume(TokenType.END, 'Expected end');
      return new AST.QuoteExpr(expr, location);
    }

    // Unquote expression
    if (this.match(TokenType.UNQUOTE)) {
      this.consume(TokenType.LPAREN, 'Expected (');
      const expr = this.parseExpression();
      this.consume(TokenType.RPAREN, 'Expected )');
      return new AST.UnquoteExpr(expr, location);
    }

    throw new ParseError('Expected expression', this.current());
  }

  parseMapLiteral(location) {
    const pairs = [];

    if (!this.check(TokenType.RBRACE)) {
      do {
        const key = this.parseExpression();
        this.match(TokenType.DOUBLE_COLON) || this.match(TokenType.ARROW);
        const value = this.parseExpression();
        pairs.push({ key, value });
      } while (this.match(TokenType.COMMA));
    }

    this.consume(TokenType.RBRACE, 'Expected }');
    return new AST.MapExpr(pairs, location);
  }

  parseTupleLiteral(location) {
    const elements = [];

    if (!this.check(TokenType.RBRACE)) {
      do {
        elements.push(this.parseExpression());
      } while (this.match(TokenType.COMMA));
    }

    this.consume(TokenType.RBRACE, 'Expected }');
    return new AST.TupleExpr(elements, location);
  }

  parseCaseExpression(location) {
    const scrutinee = this.parseExpression();
    this.consume(TokenType.DO, 'Expected do');

    const clauses = [];

    while (!this.check(TokenType.END)) {
      const pattern = this.parsePattern();
      
      // Optional guard
      let guard = null;
      if (this.match(TokenType.WHEN)) {
        guard = this.parseExpression();
      }

      this.consume(TokenType.ARROW, 'Expected ->');
      const body = this.parseExpression();

      clauses.push({ pattern, guard, body });
    }

    this.consume(TokenType.END, 'Expected end');
    return new AST.CaseExpr(scrutinee, clauses, location);
  }

  parseIfExpression(location) {
    const condition = this.parseExpression();
    this.consume(TokenType.DO, 'Expected do');
    const thenBranch = this.parseExpression();

    let elseBranch = null;
    if (this.matchIdentifier('else')) {
      elseBranch = this.parseExpression();
    }

    this.consume(TokenType.END, 'Expected end');
    return new AST.IfExpr(condition, thenBranch, elseBranch, location);
  }

  // Helper methods
  match(...types) {
    for (const type of types) {
      if (this.check(type)) {
        this.advance();
        return true;
      }
    }
    return false;
  }

  matchIdentifier(value) {
    if (this.check(TokenType.IDENTIFIER) && this.current().value === value) {
      this.advance();
      return true;
    }
    return false;
  }

  check(type) {
    if (this.isAtEnd()) return false;
    return this.current().type === type;
  }

  advance() {
    if (!this.isAtEnd()) this.position++;
    return this.previous();
  }

  isAtEnd() {
    return this.position >= this.tokens.length || this.current().type === TokenType.EOF;
  }

  current() {
    return this.tokens[this.position];
  }

  previous() {
    return this.tokens[this.position - 1];
  }

  consume(type, message) {
    if (this.check(type)) return this.advance();
    throw new ParseError(message, this.current());
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { Parser, ParseError };
}
```

### 1.1.5 GPU Code Generator (WGSL)

```javascript
// File: src/core/gpu_codegen.js
/**
 * Generates WebGPU Shading Language (WGSL) from Zapp AST
 * This is the critical bridge between high-level Zapp and GPU execution
 */

class WGSLGenerator {
  constructor() {
    this.bufferBindings = [];
    this.structDefinitions = new Map();
    this.currentIndent = 0;
  }

  generate(ast) {
    const wgsl = [];

    // Generate struct definitions first
    for (const [name, def] of this.structDefinitions) {
      wgsl.push(this.generateStruct(name, def));
    }

    // Generate buffer bindings
    wgsl.push(this.generateBindings());

    // Generate compute shader entry point
    wgsl.push(this.generateComputeShader(ast));

    return wgsl.join('\n\n');
  }

  generateStruct(name, fields) {
    const lines = [`struct ${name} {`];
    
    for (const field of fields) {
      const alignment = field.alignment || this.getDefaultAlignment(field.type);
      lines.push(`  @align(${alignment}) ${field.name}: ${this.mapType(field.type)},`);
    }

    lines.push('}');
    return lines.join('\n');
  }

  generateBindings() {
    const lines = [];

    for (let i = 0; i < this.bufferBindings.length; i++) {
      const binding = this.bufferBindings[i];
      const access = binding.readOnly ? 'read' : 'read_write';
      
      lines.push(
        `@group(0) @binding(${i}) var<storage, ${access}> ${binding.name}: array<${binding.type}>;`
      );
    }

    return lines.join('\n');
  }

  generateComputeShader(functionDef) {
    if (functionDef.type !== 'FunctionDef') {
      throw new Error('GPU kernel must be a function definition');
    }

    const workgroupSize = this.extractWorkgroupSize(functionDef.annotations);
    const lines = [
      `@compute @workgroup_size(${workgroupSize[0]}, ${workgroupSize[1]}, ${workgroupSize[2]})`,
      `fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {`
    ];

    this.currentIndent = 1;
    
    // Generate function body
    const body = this.generateStatement(functionDef.body);
    lines.push(this.indent(body));

    lines.push('}');
    return lines.join('\n');
  }

  generateStatement(node) {
    switch (node.type) {
      case 'BlockExpr':
        return node.expressions.map(expr => this.generateStatement(expr)).join('\n');

      case 'FunctionCall':
        return this.generateFunctionCall(node) + ';';

      case 'BinaryOp':
        return this.generateBinaryOp(node);

      case 'IfExpr':
        return this.generateIf(node);

      case 'CaseExpr':
        return this.generateCase(node);

      case 'Identifier':
        return node.name;

      case 'Literal':
        return this.generateLiteral(node);

      default:
        throw new Error(`Unsupported node type for GPU: ${node.type}`);
    }
  }

  generateBinaryOp(node) {
    const left = this.generateStatement(node.left);
    const right = this.generateStatement(node.right);

    // Map operators
    const opMap = {
      '+': '+',
      '-': '-',
      '*': '*',
      '/': '/',
      '==': '==',
      '!=': '!=',
      '<': '<',
      '>': '>',
      '<=': '<=',
      '>=': '>=',
      'and': '&&',
      'or': '||'
    };

    const op = opMap[node.operator] || node.operator;
    return `(${left} ${op} ${right})`;
  }

  generateIf(node) {
    const lines = [];
    const condition = this.generateStatement(node.condition);
    
    lines.push(`if (${condition}) {`);
    this.currentIndent++;
    lines.push(this.indent(this.generateStatement(node.thenBranch)));
    this.currentIndent--;
    lines.push(this.indent('}'));

    if (node.elseBranch) {
      lines.push(this.indent('else {'));
      this.currentIndent++;
      lines.push(this.indent(this.generateStatement(node.elseBranch)));
      this.currentIndent--;
      lines.push(this.indent('}'));
    }

    return lines.join('\n');
  }

  generateCase(node) {
    // WGSL doesn't have pattern matching, so we generate if-else chains
    const scrutinee = this.generateStatement(node.scrutinee);
    const lines = [];

    for (let i = 0; i < node.clauses.length; i++) {
      const clause = node.clauses[i];
      const condition = this.generatePatternCondition(scrutinee, clause.pattern);

      if (i === 0) {
        lines.push(`if (${condition}) {`);
      } else {
        lines.push(this.indent(`} else if (${condition}) {`));
      }

      this.currentIndent++;
      lines.push(this.indent(this.generateStatement(clause.body)));
      this.currentIndent--;
    }

    lines.push(this.indent('}'));
    return lines.join('\n');
  }

  generatePatternCondition(scrutinee, pattern) {
    switch (pattern.patternType) {
      case 'literal':
        return `${scrutinee} == ${this.generateLiteral({ value: pattern.value, literalType: typeof pattern.value })}`;

      case 'variable':
        // Always matches, bind variable
        return 'true';

      case 'constructor':
        // For sum types, check tag field
        return `${scrutinee}.tag == ${pattern.value.constructor}`;

      default:
        throw new Error(`Unsupported pattern type: ${pattern.patternType}`);
    }
  }

  generateFunctionCall(node) {
    const func = this.generateStatement(node.func);
    const args = node.args.map(arg => this.generateStatement(arg)).join(', ');

    // Map Zapp built-ins to WGSL built-ins
    const builtinMap = {
      'sqrt': 'sqrt',
      'abs': 'abs',
      'min': 'min',
      'max': 'max',
      'dot': 'dot',
      'cross': 'cross',
      'length': 'length',
      'normalize': 'normalize',
      'distance': 'distance'
    };

    const mappedFunc = builtinMap[func] || func;
    return `${mappedFunc}(${args})`;
  }

  generateLiteral(node) {
    switch (node.literalType) {
      case 'integer':
        return `${node.value}`;
      case 'float':
        return `${node.value}f`;
      case 'boolean':
        return node.value ? 'true' : 'false';
      case 'string':
        throw new Error('Strings not supported in GPU kernels');
      default:
        throw new Error(`Unknown literal type: ${node.literalType}`);
    }
  }

  mapType(zappType) {
    const typeMap = {
      'u32': 'u32',
      'i32': 'i32',
      'f32': 'f32',
      'f16': 'f16',
      'bool': 'bool',
      'Point2D': 'vec2<f32>',
      'Point3D': 'vec3<f32>',
      'Vector2D': 'vec2<f32>',
      'Vector3D': 'vec3<f32>',
      'Matrix4x4': 'mat4x4<f32>'
    };

    return typeMap[zappType] || zappType;
  }

  getDefaultAlignment(type) {
    const alignmentMap = {
      'u32': 4,
      'i32': 4,
      'f32': 4,
      'f16': 2,
      'bool': 4,
      'vec2<f32>': 8,
      'vec3<f32>': 16,
      'vec4<f32>': 16,
      'mat4x4<f32>': 16
    };

    return alignmentMap[this.mapType(type)] || 4;
  }

  extractWorkgroupSize(annotations) {
    for (const annotation of annotations) {
      if (annotation.startsWith('gpu_kernel')) {
        // Parse workgroup_size from annotation
        // @gpu_kernel(workgroup_size: {256, 1, 1})
        const match = annotation.match(/workgroup_size:\s*\{(\d+),\s*(\d+),\s*(\d+)\}/);
        if (match) {
          return [parseInt(match[1]), parseInt(match[2]), parseInt(match[3])];
        }
      }
    }

    // Default workgroup size
    return [256, 1, 1];
  }

  indent(text) {
    const spaces = '  '.repeat(this.currentIndent);
    return text.split('\n').map(line => spaces + line).join('\n');
  }

  // Public API for adding buffer bindings
  addBuffer(name, type, readOnly = false) {
    this.bufferBindings.push({ name, type, readOnly });
  }

  addStruct(name, fields) {
    this.structDefinitions.set(name, fields);
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { WGSLGenerator };
}
```

### 1.1.6 Interpreter & Executor

```javascript
// File: src/core/interpreter.js
/**
 * Zapp interpreter for CPU execution
 * Also orchestrates GPU execution for @gpu_kernel functions
 */

const { WGSLGenerator } = require('./gpu_codegen');

class Environment {
  constructor(parent = null) {
    this.parent = parent;
    this.bindings = new Map();
  }

  define(name, value) {
    this.bindings.set(name, value);
  }

  get(name) {
    if (this.bindings.has(name)) {
      return this.bindings.get(name);
    }
    if (this.parent) {
      return this.parent.get(name);
    }
    throw new Error(`Undefined variable: ${name}`);
  }

  set(name, value) {
    if (this.bindings.has(name)) {
      this.bindings.set(name, value);
      return;
    }
    if (this.parent) {
      this.parent.set(name, value);
      return;
    }
    throw new Error(`Undefined variable: ${name}`);
  }
}

class ZappInterpreter {
  constructor(runtime) {
    this.runtime = runtime;
    this.globalEnv = new Environment();
    this.actors = new Map();
    
    // Register built-in functions
    this.registerBuiltins();
  }

  registerBuiltins() {
    // Math functions
    this.globalEnv.define('sqrt', Math.sqrt);
    this.globalEnv.define('abs', Math.abs);
    this.globalEnv.define('min', Math.min);
    this.globalEnv.define('max', Math.max);
    this.globalEnv.define('floor', Math.floor);
    this.globalEnv.define('ceil', Math.ceil);
    this.globalEnv.define('round', Math.round);

    // Actor primitives
    this.globalEnv.define('spawn', (actorDef, initialState) => this.spawnActor(actorDef, initialState));
    this.globalEnv.define('send', (pid, message) => this.sendMessage(pid, message));

    // GPU primitives
    this.globalEnv.define('gpu_alloc', (size) => this.runtime.createBuffer(size));
  }

  async evaluate(ast) {
    return this.evaluateNode(ast, this.globalEnv);
  }

  async evaluateNode(node, env) {
    switch (node.type) {
      case 'Module':
        return this.evaluateModule(node, env);

      case 'FunctionDef':
        return this.evaluateFunctionDef(node, env);

      case 'ActorDef':
        return this.evaluateActorDef(node, env);

      case 'PlanetDef':
        return this.evaluatePlanetDef(node, env);

      case 'StarDef':
        return this.evaluateStarDef(node, env);

      case 'Literal':
        return node.value;

      case 'Identifier':
        return env.get(node.name);

      case 'BinaryOp':
        return this.evaluateBinaryOp(node, env);

      case 'UnaryOp':
        return this.evaluateUnaryOp(node, env);

      case 'FunctionCall':
        return await this.evaluateFunctionCall(node, env);

      case 'CaseExpr':
        return this.evaluateCase(node, env);

      case 'IfExpr':
        return this.evaluateIf(node, env);

      case 'BlockExpr':
        return this.evaluateBlock(node, env);

      case 'ListExpr':
        return await Promise.all(node.elements.map(e => this.evaluateNode(e, env)));

      case 'TupleExpr':
        return await Promise.all(node.elements.map(e => this.evaluateNode(e, env)));

      case 'MapExpr':
        return this.evaluateMap(node, env);

      case 'QuoteExpr':
        return node.expr; // Return AST node itself

      case 'UnquoteExpr':
        return this.evaluateNode(node.expr, env);

      default:
        throw new Error(`Unsupported node type: ${node.type}`);
    }
  }

  evaluateModule(node, env) {
    let result = null;
    for (const statement of node.body) {
      result = this.evaluateNode(statement, env);
    }
    return result;
  }

  evaluateFunctionDef(node, env) {
    // Check if this is a GPU kernel
    const isGPUKernel = node.annotations.includes('gpu_kernel') || 
                        node.annotations.includes('gpu_compute');

    if (isGPUKernel) {
      // Compile to GPU
      const gpuFunction = this.compileToGPU(node);
      env.define(node.name, gpuFunction);
      return gpuFunction;
    } else {
      // Regular CPU function
      const func = async (...args) => {
        const funcEnv = new Environment(env);
        
        // Bind parameters
        for (let i = 0; i < node.params.length; i++) {
          const param = node.params[i];
          if (param.patternType === 'variable') {
            funcEnv.define(param.value.name, args[i]);
          }
        }

        // Evaluate body
        return this.evaluateNode(node.body, funcEnv);
      };

      env.define(node.name, func);
      return func;
    }
  }

  compileToGPU(functionDef) {
    const generator = new WGSLGenerator();
    
    // Analyze function to determine buffer requirements
    this.analyzeGPUBuffers(functionDef, generator);

    // Generate WGSL
    const wgslCode = generator.generate(functionDef);
    
    // Create compute pipeline
    const pipeline = this.runtime.createComputePipeline(wgslCode);

    // Return GPU-executable function
    return async (...args) => {
      // Marshal arguments to GPU buffers
      const buffers = await this.marshalToGPU(args, generator.bufferBindings);

      // Create bind group
      const bindGroup = this.runtime.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: buffers.map((buffer, index) => ({
          binding: index,
          resource: { buffer }
        }))
      });

      // Execute
      const workgroupCounts = this.calculateWorkgroupCounts(args);
      await this.runtime.executeCompute(pipeline, [bindGroup], workgroupCounts);

      // Read back results
      return this.unmarshalFromGPU(buffers[buffers.length - 1]);
    };
  }

  analyzeGPUBuffers(functionDef, generator) {
    // Simplified analysis - in production, would do full data flow analysis
    for (const param of functionDef.params) {
      if (param.patternType === 'variable' && param.value.type) {
        const type = param.value.type.baseType;
        generator.addBuffer(param.value.name, type, true); // Input buffers
      }
    }

    // Always add result buffer
    generator.addBuffer('result', 'f32', false); // Output buffer
  }

  async marshalToGPU(args, bindings) {
    const buffers = [];

    for (let i = 0; i < args.length; i++) {
      const arg = args[i];
      const binding = bindings[i];

      if (Array.isArray(arg)) {
        // Array data
        const typedArray = this.convertToTypedArray(arg, binding.type);
        const buffer = this.runtime.createBuffer(typedArray.byteLength);
        this.runtime.device.queue.writeBuffer(buffer, 0, typedArray);
        buffers.push(buffer);
      } else {
        // Scalar value
        const typedArray = new Float32Array([arg]);
        const buffer = this.runtime.createBuffer(typedArray.byteLength);
        this.runtime.device.queue.writeBuffer(buffer, 0, typedArray);
        buffers.push(buffer);
      }
    }

    // Add result buffer
    const resultBuffer = this.runtime.createBuffer(1024); // 1KB result buffer
    buffers.push(resultBuffer);

    return buffers;
  }

  async unmarshalFromGPU(buffer) {
    const stagingBuffer = this.runtime.device.createBuffer({
      size: buffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    const commandEncoder = this.runtime.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, buffer.size);
    this.runtime.device.queue.submit([commandEncoder.finish()]);

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(stagingBuffer.getMappedRange());
    const result = Array.from(data);
    stagingBuffer.unmap();

    return result;
  }

  convertToTypedArray(array, type) {
    const typeMap = {
      'u32': Uint32Array,
      'i32': Int32Array,
      'f32': Float32Array,
      'f16': Uint16Array // f16 requires special handling
    };

    const ArrayType = typeMap[type] || Float32Array;
    return new ArrayType(array);
  }

  calculateWorkgroupCounts(args) {
    // Simple heuristic: divide data size by 256
    const dataSize = Array.isArray(args[0]) ? args[0].length : 1;
    const workgroupSize = 256;
    const workgroups = Math.ceil(dataSize / workgroupSize);
    return [workgroups, 1, 1];
  }

  evaluateActorDef(node, env) {
    // Store actor definition for spawning
    const actorDef = {
      name: node.name,
      state: node.state,
      handlers: node.handlers,
      annotations: node.annotations
    };

    env.define(node.name, actorDef);
    return actorDef;
  }

  evaluatePlanetDef(node, env) {
    // Create constructor function for planet (struct)
    const constructor = (fields) => {
      const instance = { __planet__: node.name };
      
      for (const orbital of node.orbitals) {
        instance[orbital.name] = fields[orbital.name];
      }

      return instance;
    };

    env.define(node.name, constructor);
    return constructor;
  }

  evaluateStarDef(node, env) {
    // Create constructor functions for each variant
    const variants = {};

    for (const layer of node.layers) {
      variants[layer.constructor] = (fields = {}) => {
        return {
          __star__: node.name,
          __variant__: layer.constructor,
          ...fields
        };
      };
    }

    env.define(node.name, variants);
    return variants;
  }

  evaluateBinaryOp(node, env) {
    const left = this.evaluateNode(node.left, env);
    const right = this.evaluateNode(node.right, env);

    const operators = {
      '+': (l, r) => l + r,
      '-': (l, r) => l - r,
      '*': (l, r) => l * r,
      '/': (l, r) => l / r,
      '==': (l, r) => l === r,
      '!=': (l, r) => l !== r,
      '<': (l, r) => l < r,
      '>': (l, r) => l > r,
      '<=': (l, r) => l <= r,
      '>=': (l, r) => l >= r,
      'and': (l, r) => l && r,
      'or': (l, r) => l || r,
      '|>': (l, r) => r(l) // Pipe operator
    };

    const op = operators[node.operator];
    if (!op) {
      throw new Error(`Unknown operator: ${node.operator}`);
    }

    return op(left, right);
  }

  evaluateUnaryOp(node, env) {
    const operand = this.evaluateNode(node.operand, env);

    const operators = {
      '-': (x) => -x,
      'not': (x) => !x
    };

    const op = operators[node.operator];
    if (!op) {
      throw new Error(`Unknown unary operator: ${node.operator}`);
    }

    return op(operand);
  }

  async evaluateFunctionCall(node, env) {
    const func = await this.evaluateNode(node.func, env);
    const args = await Promise.all(node.args.map(arg => this.evaluateNode(arg, env)));

    if (typeof func !== 'function') {
      throw new Error(`${node.func} is not a function`);
    }

    return func(...args);
  }

  evaluateCase(node, env) {
    const scrutinee = this.evaluateNode(node.scrutinee, env);

    for (const clause of node.clauses) {
      const match = this.matchPattern(scrutinee, clause.pattern, env);
      
      if (match) {
        // Check guard if present
        if (clause.guard) {
          const guardResult = this.evaluateNode(clause.guard, match.env);
          if (!guardResult) continue;
        }

        return this.evaluateNode(clause.body, match.env);
      }
    }

    throw new Error('No matching case clause');
  }

  matchPattern(value, pattern, env) {
    const matchEnv = new Environment(env);

    switch (pattern.patternType) {
      case 'literal':
        return value === pattern.value ? { env: matchEnv } : null;

      case 'variable':
        matchEnv.define(pattern.value.name, value);
        return { env: matchEnv };

      case 'constructor':
        if (value.__variant__ === pattern.value.constructor) {
          // Match constructor fields
          for (let i = 0; i < pattern.value.args.length; i++) {
            const fieldPattern = pattern.value.args[i];
            const fieldValue = value[Object.keys(value)[i + 2]]; // Skip __star__ and __variant__
            
            const fieldMatch = this.matchPattern(fieldValue, fieldPattern, matchEnv);
            if (!fieldMatch) return null;
          }
          return { env: matchEnv };
        }
        return null;

      case 'list':
        if (!Array.isArray(value)) return null;
        if (value.length !== pattern.value.length) return null;

        for (let i = 0; i < pattern.value.length; i++) {
          const elemMatch = this.matchPattern(value[i], pattern.value[i], matchEnv);
          if (!elemMatch) return null;
        }
        return { env: matchEnv };

      case 'tuple':
        if (!Array.isArray(value)) return null;
        if (value.length !== pattern.value.length) return null;

        for (let i = 0; i < pattern.value.length; i++) {
          const elemMatch = this.matchPattern(value[i], pattern.value[i], matchEnv);
          if (!elemMatch) return null;
        }
        return { env: matchEnv };

      default:
        throw new Error(`Unknown pattern type: ${pattern.patternType}`);
    }
  }

  evaluateIf(node, env) {
    const condition = this.evaluateNode(node.condition, env);

    if (condition) {
      return this.evaluateNode(node.thenBranch, env);
    } else if (node.elseBranch) {
      return this.evaluateNode(node.elseBranch, env);
    }

    return null;
  }

  evaluateBlock(node, env) {
    let result = null;
    for (const expr of node.expressions) {
      result = this.evaluateNode(expr, env);
    }
    return result;
  }

  async evaluateMap(node, env) {
    const map = new Map();

    for (const pair of node.pairs) {
      const key = await this.evaluateNode(pair.key, env);
      const value = await this.evaluateNode(pair.value, env);
      map.set(key, value);
    }

    return map;
  }

  // Actor system
  spawnActor(actorDef, initialState) {
    const pid = crypto.randomUUID();
    const isGPUActor = actorDef.annotations.includes('gpu_compute');

    const actor = {
      pid,
      actorDef,
      state: initialState,
      mailbox: [],
      isGPUActor,
      worker: null
    };

    if (isGPUActor) {
      // Spawn as GPU-backed actor (uses Web Worker + GPU)
      actor.worker = new Worker('zapp_gpu_actor_worker.js');
      actor.worker.postMessage({
        type: 'init',
        actorDef: actorDef,
        initialState: initialState
      });
    }

    this.actors.set(pid, actor);
    return pid;
  }

  sendMessage(pid, message) {
    const actor = this.actors.get(pid);
    if (!actor) {
      throw new Error(`Actor not found: ${pid}`);
    }

    if (actor.isGPUActor) {
      // Send to GPU actor worker
      actor.worker.postMessage({
        type: 'message',
        message: message
      });
    } else {
      // CPU actor - process immediately
      actor.mailbox.push(message);
      this.processActorMessage(actor);
    }
  }

  async processActorMessage(actor) {
    if (actor.mailbox.length === 0) return;

    const message = actor.mailbox.shift();

    // Find matching handler
    for (const handler of actor.actorDef.handlers) {
      const match = this.matchPattern(message, handler.params[0], this.globalEnv);
      
      if (match) {
        const handlerEnv = new Environment(this.globalEnv);
        handlerEnv.define('state', actor.state);

        const result = await this.evaluateNode(handler.body, match.env);

        // Update actor state
        if (result && result.state) {
          actor.state = result.state;
        }

        break;
      }
    }
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { ZappInterpreter, Environment };
}
```

---

# LAYER 2: Zapp Language Macros (Written in Zapp Core)

## Document 2.1: Macro System Bootstrap

**Target**: Generative AI implementing macros IN Zapp
**Prerequisites**: Layer 1 complete (JavaScript runtime)
**Language**: Zapp Core (using quote/unquote)

### 2.1.1 Core Macro Infrastructure

```elixir
# File: stdlib/macros/core.zapp
# This file is written in Zapp and implements the macro expansion system

defmacro defmacro(name, args, body) do
  quote do
    def unquote(name)(unquote(args)) do
      # Macro body gets AST manipulation powers
      unquote(body)
    end
  end
end

# Unless macro - classic example
defmacro unless(condition, do: block) do
  quote do
    if not unquote(condition) do
      unquote(block)
    end
  end
end

# Assert macro with location tracking
defmacro assert(condition, message) do
  quote do
    case unquote(condition) do
      true -> :ok
      false -> 
        error("Assertion failed: #{unquote(message)} at #{__location__}")
    end
  end
end

# Pipe operator as macro (could be built-in, but shown as example)
defmacro pipe(left, right) do
  quote do
    unquote(right)(unquote(left))
  end
end
```

### 2.1.2 Stellarmorphism Core Macros

```elixir
# File: stdlib/macros/stellarmorphism.zapp
# Implementing defplanet and defstar as macros

defmacro defplanet(name, do: block) do
  # Extract orbitals from block
  orbitals = extract_orbitals(block)
  
  quote do
    defmodule unquote(name) do
      # Generate struct definition
      @orbitals unquote(orbitals)
      
      # Constructor
      def new(fields) do
        struct = %{__planet__: unquote(name)}
        
        # Validate and assign fields
        for orbital <- @orbitals do
          value = Map.get(fields, orbital.name)
          
          # Type checking
          unless valid_type?(value, orbital.type) do
            error("Type mismatch for #{orbital.name}")
          end
          
          struct = Map.put(struct, orbital.name, value)
        end
        
        struct
      end
      
      # Getters for each orbital
      for orbital <- @orbitals do
        def unquote(orbital.name)(planet) do
          Map.get(planet, unquote(orbital.name))
        end
      end
    end
  end
end

defmacro defstar(name, do: block) do
  # Extract layers (variants) from block
  layers = extract_layers(block)
  
  quote do
    defmodule unquote(name) do
      @layers unquote(layers)
      
      # Generate constructor for each variant
      for layer <- @layers do
        def unquote(layer.constructor)(fields) do
          %{
            __star__: unquote(name),
            __variant__: unquote(layer.constructor),
            __fields__: fields
          }
        end
      end
      
      # Pattern matching helper
      defmacro match_star(value, clauses) do
        quote do
          case unquote(value).__variant__ do
            unquote_splicing(
              for clause <- clauses do
                quote do
                  unquote(clause.pattern) -> unquote(clause.body)
                end
              end
            )
          end
        end
      end
    end
  end
end

# Core helper for defstar
defmacro core(constructor, fields) do
  quote do
    %{
      __variant__: unquote(constructor),
      unquote_splicing(fields)
    }
  end
end

# Fission macro - pattern matching on stars
defmacro fission(star_type, value, do: clauses) do
  quote do
    case unquote(value).__variant__ do
      unquote_splicing(
        for clause <- clauses do
          pattern = clause.pattern
          body = clause.body
          
          quote do
            unquote(pattern.__variant__) ->
              # Bind fields from pattern
              unquote_splicing(bind_fields(pattern, value))
              unquote(body)
          end
        end
      )
    end
  end
end

# Fusion macro - construction with pattern matching
defmacro fusion(star_type, value, do: clauses) do
  quote do
    case unquote(value) do
      unquote_splicing(
        for clause <- clauses do
          quote do
            unquote(clause.pattern) -> unquote(clause.body)
          end
        end
      )
    end
  end
end
```

### 2.1.3 GPU Kernel Macros

```elixir
# File: stdlib/macros/gpu.zapp
# Macros for GPU computation

defmacro defgpu(name, params, do: body) do
  # Analyze body for GPU compatibility
  ensure_gpu_compatible!(body)
  
  quote do
    @gpu_kernel(workgroup_size: {256, 1, 1})
    def unquote(name)(unquote_splicing(params)) do
      unquote(body)
    end
  end
end

# Parallel comprehension - compiles to GPU kernel
defmacro gpu_for(generator, do: body) do
  {var, collection} = generator
  
  quote do
    @gpu_kernel
    def __parallel_map__(input_buffer, output_buffer) do
      gid = builtin_global_id()
      
      if gid < length(input_buffer) do
        unquote(var) = input_buffer[gid]
        result = unquote(body)
        output_buffer[gid] = result
      end
    end
    
    # Call the generated kernel
    __parallel_map__(unquote(collection), gpu_alloc(length(unquote(collection))))
  end
end

# GPU reduction pattern
defmacro gpu_reduce(collection, acc, fun) do
  quote do
    @gpu_kernel(workgroup_size: {256, 1, 1})
    def __parallel_reduce__(input, output, acc_init) do
      # Shared memory for reduction
      shared_mem = workgroup_array(256, type: f32)
      
      gid = builtin_global_id()
      lid = builtin_local_id()
      
      # Load data into shared memory
      if gid < length(input) do
        shared_mem[lid] = input[gid]
      else
        shared_mem[lid] = acc_init
      end
      
      workgroup_barrier()
      
      # Tree reduction within workgroup
      stride = 128
      while stride > 0 do
        if lid < stride do
          shared_mem[lid] = unquote(fun)(shared_mem[lid], shared_mem[lid + stride])
        end
        workgroup_barrier()
        stride = stride / 2
      end
      
      # Write workgroup result
      if lid == 0 do
        output[builtin_workgroup_id()] = shared_mem[0]
      end
    end
    
    # Launch reduction
    __parallel_reduce__(unquote(collection), gpu_alloc(256), unquote(acc))
  end
end
```

### 2.1.4 Actor System Macros

```elixir
# File: stdlib/macros/actors.zapp
# Actor model implementation as macros

defmacro defactor(name, do: block) do
  # Extract state schema and handlers
  {state_schema, handlers} = parse_actor_block(block)
  
  is_gpu = has_annotation?(block, :gpu_compute)
  
  quote do
    defmodule unquote(name) do
      @state_schema unquote(state_schema)
      @handlers unquote(handlers)
      @is_gpu unquote(is_gpu)
      
      def start_link(initial_state) do
        pid = spawn(__MODULE__, :init, [initial_state])
        {:ok, pid}
      end
      
      def init(state) do
        if @is_gpu do
          # Initialize GPU resources
          gpu_state = allocate_gpu_state(state)
          loop(gpu_state)
        else
          loop(state)
        end
      end
      
      def loop(state) do
        receive do
          message ->
            # Pattern match against handlers
            new_state = handle_message(message, state)
            loop(new_state)
        end
      end
      
      # Generate handler functions
      unquote_splicing(
        for handler <- handlers do
          quote do
            def handle_message(unquote(handler.pattern), state) do
              unquote(handler.body)
            end
          end
        end
      )
    end
  end
end

# Send macro with type checking
defmacro send(pid, message) do
  quote do
    # Validate message format at compile time if possible
    runtime_send(unquote(pid), unquote(message))
  end
end

# Receive macro with timeout
defmacro receive(timeout: timeout_ms, do: clauses) do
  quote do
    start_time = now()
    
    loop do
      if mailbox_empty?() and (now() - start_time) > unquote(timeout_ms) do
        :timeout
      else
        message = dequeue_message()
        
        case message do
          unquote_splicing(
            for clause <- clauses do
              quote do
                unquote(clause.pattern) -> unquote(clause.body)
              end
            end
          )
          _ -> loop() # No match, continue
        end
      end
    end
  end
end
```

---

# LAYER 3+: Advanced Features (Written in Zapp Macros)

## Document 3.1: GIS & Spatial Types

**Target**: Fleet management system
**Prerequisites**: Layers 1-2 complete
**Language**: Zapp with macros

### 3.1.1 Spatial Type Definitions

```elixir
# File: stdlib/gis/types.zapp
# PostGIS-compatible spatial types for GPU

defplanet Point2D do
  orbitals do
    moon x :: f32, @gpu_align(4)
    moon y :: f32, @gpu_align(4)
  end
  
  @gpu_compute
  def distance(p1, p2) do
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    sqrt(dx * dx + dy * dy)
  end
end

defplanet Polygon do
  orbitals do
    moon points :: [Point2D], @gpu_buffer
    moon num_points :: u32
  end
  
  @gpu_kernel(workgroup_size: {256, 1, 1})
  def contains_point?(polygon, point) do
    # Ray casting algorithm on GPU
    gid = builtin_global_id()
    
    if gid == 0 do
      inside = false
      j = polygon.num_points - 1
      
      for i <- 0..polygon.num_points-1 do
        pi = polygon.points[i]
        pj = polygon.points[j]
        
        if ((pi.y > point.y) != (pj.y > point.y)) and
           (point.x < (pj.x - pi.x) * (point.y - pi.y) / (pj.y - pi.y) + pi.x) do
          inside = not inside
        end
        
        j = i
      end
      
      result_buffer[0] = if inside then 1 else 0 end
    end
  end
end

defstar GeofenceType do
  layers do
    core InclusionZone, radius :: f32
    core ExclusionZone, radius :: f32
    core SpeedZone, speed_limit :: f32
  end
end

defplanet Geofence do
  orbitals do
    moon id :: u32, @gpu_align(4)
    moon boundary :: Polygon, @gpu_buffer
    moon fence_type :: GeofenceType
  end
end
```

### 3.1.2 Fleet Management Actors

```elixir
# File: apps/geofleet/fleet_manager.zapp
# GPU-accelerated fleet management

defactor FleetManager do
  @gpu_compute
  
  state do
    vehicles :: [Vehicle], @gpu_buffer
    geofences :: [Geofence], @gpu_buffer
    breach_events :: [BreachEvent], @gpu_buffer
  end
  
  @gpu_kernel(workgroup_size: {256, 1, 1})
  def handle_location_update({:update, vehicle_id, new_location}) do
    gid = builtin_global_id()
    
    # Each workgroup processes one vehicle against all geofences
    if gid < length(state.vehicles) do
      vehicle = state.vehicles[gid]
      
      if vehicle.id == vehicle_id do
        # Update location
        vehicle.location = new_location
        
        # Check all geofences in parallel
        for geofence <- state.geofences do
          is_inside = Polygon.contains_point?(geofence.boundary, new_location)
          was_inside = Polygon.contains_point?(geofence.boundary, vehicle.previous_location)
          
          # Detect breach
          if is_inside != was_inside do
            breach = %BreachEvent{
              vehicle_id: vehicle_id,
              geofence_id: geofence.id,
              breach_type: if is_inside then :entry else :exit end,
              timestamp: now()
            }
            
            # Atomic append to breach buffer
            atomic_append(state.breach_events, breach)
          end
        end
        
        vehicle.previous_location = new_location
      end
    end
    
    {:noreply, state}
  end
  
  def handle_query({:get_vehicles_in_geofence, geofence_id}) do
    # Launch GPU kernel to find all vehicles in geofence
    result = gpu_for vehicle <- state.vehicles do
      geofence = find_geofence(state.geofences, geofence_id)
      
      if Polygon.contains_point?(geofence.boundary, vehicle.location) do
        vehicle.id
      else
        nil
      end
    end
    
    vehicles = Enum.filter(result, fn x -> x != nil end)
    {:reply, vehicles, state}
  end
end
```

### 3.1.3 Web3 Integration

```elixir
# File: stdlib/web3/ethereum.zapp
# Web3 protocol integration

defmacro defcontract(name, abi_path) do
  abi = load_abi(abi_path)
  
  quote do
    defmodule unquote(name) do
      @abi unquote(abi)
      
      # Generate function wrappers for each ABI method
      unquote_splicing(
        for method <- abi.methods do
          quote do
            def unquote(method.name)(unquote_splicing(method.params)) do
              # Encode call data
              call_data = encode_function_call(
                unquote(method.signature),
                [unquote_splicing(method.params)]
              )
              
              # Send transaction
              tx = %Transaction{
                to: @contract_address,
                data: call_data,
                gas: estimate_gas(call_data)
              }
              
              send_transaction(tx)
            end
          end
        end
      )
    end
  end
end

# Fleet registry smart contract
defcontract FleetRegistry, "contracts/FleetRegistry.json"

defactor BlockchainSync do
  state do
    last_block :: u64
    fleet_manager_pid :: pid
  end
  
  def init(fleet_manager_pid) do
    # Subscribe to blockchain events
    subscribe_events(FleetRegistry.address(), ["VehicleRegistered", "GeofenceCreated"])
    {:ok, %{last_block: current_block(), fleet_manager_pid: fleet_manager_pid}}
  end
  
  def handle_event({:VehicleRegistered, vehicle_id, owner, block_number}) do
    # Sync vehicle to local GPU state
    send(state.fleet_manager_pid, {:register_vehicle, vehicle_id, owner})
    {:noreply, %{state | last_block: block_number}}
  end
  
  def handle_event({:GeofenceCreated, geofence_id, boundary_points, block_number}) do
    # Convert blockchain data to GPU-friendly format
    polygon = Polygon.new(%{
      points: parse_points(boundary_points),
      num_points: length(boundary_points)
    })
    
    send(state.fleet_manager_pid, {:add_geofence, geofence_id, polygon})
    {:noreply, %{state | last_block: block_number}}
  end
end
```

### 3.1.4 Browser Runtime Integration

```elixir
# File: stdlib/browser/runtime.zapp
# Browser API compatibility layer

defmodule ZappBrowser do
  # Initialize Zapp runtime in browser
  def start() do
    # Initialize WebGPU
    {:ok, gpu_runtime} = ZappRuntime.initialize()
    
    # Start actor supervisor
    {:ok, actor_sup} = Supervisor.start_link([], strategy: :one_for_one)
    
    # Expose global API
    window.Zapp = %{
      spawn: fn actor, state -> spawn_actor(actor, state) end,
      send: fn pid, msg -> send_message(pid, msg) end,
      gpu: gpu_runtime
    }
    
    {:ok, gpu_runtime}
  end
  
  # Web Worker integration for actors
  defactor WorkerActor do
    def init(actor_module, initial_state) do
      # This actor runs in a Web Worker
      worker = Worker.new("zapp_worker.js")
      
      worker.on_message(fn message ->
        handle_message(message, state)
      end)
      
      {:ok, %{worker: worker, state: initial_state}}
    end
  end
end

# WebSocket actor for real-time sync
defactor WebSocketClient do
  state do
    socket :: WebSocket
    handlers :: map
  end
  
  def connect(url) do
    socket = WebSocket.new(url)
    
    socket.on("open", fn ->
      send(self(), :connected)
    end)
    
    socket.on("message", fn data ->
      send(self(), {:message, decode(data)})
    end)
    
    {:ok, %{socket: socket, handlers: %{}}}
  end
  
  def handle_cast({:send, data}) do
    WebSocket.send(state.socket, encode(data))
    {:noreply, state}
  end
  
  def handle_info({:message, data}) do
    # Route to appropriate handler
    case data.type do
      "location_update" -> 
        broadcast(:fleet_updates, data)
      "geofence_breach" ->
        broadcast(:alerts, data)
    end
    
    {:noreply, state}
  end
end
```

This comprehensive specification provides everything needed to build Zapp from scratch across three layers:

1. **Layer 1** (JavaScript): Lexer, parser, AST, GPU codegen, interpreter
2. **Layer 2** (Zapp Core): Macro system, Stellarmorphism, GPU kernels, actors
3. **Layer 3+** (Zapp Macros): GIS types, fleet management, Web3, browser integration

The key innovation is that Layer 1 provides just enough functionality to bootstrap the macro system, then everything else is built using those macros, creating a self-hosting, GPU-accelerated language perfect for your fleet management use case.