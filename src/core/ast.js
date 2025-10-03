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

// Export all node types for both Node.js and ES6 modules
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

// ES6 module exports for browser
export {
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