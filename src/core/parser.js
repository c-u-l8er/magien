/**
 * Z Parser - Recursive descent parser for Zapp language
 * Produces AST from token stream
 */

import { TokenType } from './lexer.js';
import * as AST from './ast.js';

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
    
    while (!this.check(TokenType.END) && !this.isAtEnd()) {
      const name = this.consume(TokenType.IDENTIFIER, 'Expected field name').value;
      this.consume(TokenType.DOUBLE_COLON, 'Expected ::');
      const type = this.parseTypeAnnotation();
      
      fields.push({ name, type });
    }
    
    return fields;
  }

  parseOrbitalFields() {
    const fields = [];
    
    while (!this.check(TokenType.END) && !this.isAtEnd()) {
      this.matchIdentifier('moon'); // optional moon keyword
      const name = this.consume(TokenType.IDENTIFIER, 'Expected orbital name').value;
      this.consume(TokenType.DOUBLE_COLON, 'Expected ::');
      const type = this.parseTypeAnnotation();
      
      fields.push({ name, type });
    }
    
    return fields;
  }

  parseLayerVariants() {
    const variants = [];
    
    while (!this.check(TokenType.END) && !this.isAtEnd()) {
      this.matchIdentifier('core'); // optional core keyword
      const constructor = this.consume(TokenType.MODULE_IDENTIFIER, 'Expected constructor name').value;
      
      const fields = [];
      if (this.match(TokenType.LPAREN)) {
        if (!this.check(TokenType.RPAREN)) {
          do {
            const fieldName = this.consume(TokenType.IDENTIFIER, 'Expected field name').value;
            this.consume(TokenType.DOUBLE_COLON, 'Expected ::');
            const fieldType = this.parseTypeAnnotation();
            fields.push({ name: fieldName, type: fieldType });
          } while (this.match(TokenType.COMMA));
        }
        this.consume(TokenType.RPAREN, 'Expected )');
      }
      
      variants.push({ constructor, fields });
    }
    
    return variants;
  }

  parseExpression() {
    return this.parseBinaryOp();
  }

  parseBinaryOp() {
    let left = this.parseUnaryOp();

    while (this.matchEquality() || this.matchComparison() || 
           this.match(TokenType.PLUS) || this.match(TokenType.MINUS) ||
           this.match(TokenType.AND) || this.match(TokenType.OR)) {
      const operator = this.previous().value;
      const right = this.parseUnaryOp();
      left = new AST.BinaryOp(operator, left, right, this.previous().location);
    }

    return left;
  }

  parseUnaryOp() {
    if (this.match(TokenType.MINUS) || this.match(TokenType.NOT)) {
      const operator = this.previous().value;
      const operand = this.parseUnaryOp();
      return new AST.UnaryOp(operator, operand, this.previous().location);
    }

    return this.parseCall();
  }

  parseCall() {
    let expr = this.parsePrimary();

    while (this.match(TokenType.LPAREN)) {
      const args = this.parseArgumentList();
      this.consume(TokenType.RPAREN, 'Expected )');
      expr = new AST.FunctionCall(expr, args, this.previous().location);
    }

    return expr;
  }

  parsePrimary() {
    if (this.match(TokenType.INTEGER)) {
      return new AST.Literal(this.previous().value, 'integer', this.previous().location);
    }

    if (this.match(TokenType.FLOAT)) {
      return new AST.Literal(this.previous().value, 'float', this.previous().location);
    }

    if (this.match(TokenType.STRING)) {
      return new AST.Literal(this.previous().value, 'string', this.previous().location);
    }

    if (this.match(TokenType.BOOLEAN)) {
      return new AST.Literal(this.previous().value, 'boolean', this.previous().location);
    }

    if (this.match(TokenType.ATOM)) {
      return new AST.Literal(this.previous().value, 'atom', this.previous().location);
    }

    if (this.match(TokenType.IF)) {
      return this.parseIfExpr();
    }

    if (this.match(TokenType.CASE)) {
      return this.parseCaseExpr();
    }

    if (this.match(TokenType.DO)) {
      return this.parseBlock();
    }

    if (this.match(TokenType.LBRACKET)) {
      return this.parseListExpr();
    }

    if (this.match(TokenType.LBRACE)) {
      return this.parseTupleExpr();
    }

    if (this.check(TokenType.IDENTIFIER)) {
      const name = this.advance().value;
      return new AST.Identifier(name, this.previous().location);
    }

    throw new ParseError('Expected expression', this.current());
  }

  parseIfExpr() {
    const location = this.previous().location;
    const condition = this.parseExpression();
    
    this.consume(TokenType.THEN, 'Expected then');
    const thenBranch = this.parseExpression();
    
    let elseBranch = null;
    if (this.match(TokenType.ELSE)) {
      elseBranch = this.parseExpression();
    }

    return new AST.IfExpr(condition, thenBranch, elseBranch, location);
  }

  parseCaseExpr() {
    const location = this.previous().location;
    const scrutinee = this.parseExpression();
    
    this.consume(TokenType.DO, 'Expected do');
    
    const clauses = [];
    while (!this.check(TokenType.END) && !this.isAtEnd()) {
      const pattern = this.parsePattern();
      
      let guard = null;
      if (this.match(TokenType.WHEN)) {
        guard = this.parseExpression();
      }
      
      this.consume(TokenType.MATCH, 'Expected =>');
      const body = this.parseExpression();
      
      clauses.push({ pattern, guard, body });
    }
    
    this.consume(TokenType.END, 'Expected end');
    
    return new AST.CaseExpr(scrutinee, clauses, location);
  }

  parseBlock() {
    const location = this.previous().location;
    const expressions = [];
    
    while (!this.check(TokenType.END) && !this.isAtEnd()) {
      expressions.push(this.parseExpression());
    }
    
    return new AST.BlockExpr(expressions, location);
  }

  parseListExpr() {
    const location = this.previous().location;
    const elements = [];
    
    if (!this.check(TokenType.RBRACKET)) {
      do {
        elements.push(this.parseExpression());
      } while (this.match(TokenType.COMMA));
    }
    
    this.consume(TokenType.RBRACKET, 'Expected ]');
    return new AST.ListExpr(elements, location);
  }

  parseTupleExpr() {
    const location = this.previous().location;
    const elements = [];
    
    if (!this.check(TokenType.RBRACE)) {
      do {
        elements.push(this.parseExpression());
      } while (this.match(TokenType.COMMA));
    }
    
    this.consume(TokenType.RBRACE, 'Expected }');
    return new AST.TupleExpr(elements, location);
  }

  parseArgumentList() {
    const args = [];
    
    if (!this.check(TokenType.RPAREN)) {
      do {
        args.push(this.parseExpression());
      } while (this.match(TokenType.COMMA));
    }
    
    return args;
  }

  // Helper methods
  match(tokenType) {
    if (this.check(tokenType)) {
      this.advance();
      return true;
    }
    return false;
  }

  matchIdentifier(name) {
    if (this.check(TokenType.IDENTIFIER) && this.current().value === name) {
      this.advance();
      return true;
    }
    return false;
  }

  matchEquality() {
    return this.match(TokenType.EQ) || this.match(TokenType.NEQ);
  }

  matchComparison() {
    return this.match(TokenType.LT) || this.match(TokenType.GT) || 
           this.match(TokenType.LTE) || this.match(TokenType.GTE);
  }

  check(tokenType) {
    if (this.isAtEnd()) return false;
    return this.current().type === tokenType;
  }

  advance() {
    if (!this.isAtEnd()) this.position++;
    return this.previous();
  }

  isAtEnd() {
    return this.peek().type === TokenType.EOF;
  }

  current() {
    return this.tokens[this.position];
  }

  previous() {
    return this.tokens[this.position - 1];
  }

  peek() {
    return this.tokens[this.position];
  }

  consume(tokenType, message) {
    if (this.check(tokenType)) return this.advance();
    
    throw new ParseError(message, this.current());
  }
}

// Export for both Node.js and ES6 modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { Parser, ParseError };
}

// ES6 module exports for browser
export { Parser, ParseError };