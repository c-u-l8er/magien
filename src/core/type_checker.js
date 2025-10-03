/**
 * Zapp Type System - Type checking and inference
 * Provides static type analysis for GPU compatibility and safety
 */

class Type {
  constructor(name, params = []) {
    this.name = name;
    this.params = params;
  }

  toString() {
    if (this.params.length === 0) {
      return this.name;
    }
    return `${this.name}<${this.params.map(p => p.toString()).join(', ')}>`;
  }

  equals(other) {
    if (this.name !== other.name) return false;
    if (this.params.length !== other.params.length) return false;
    return this.params.every((param, i) => param.equals(other.params[i]));
  }

  isGeneric() {
    return this.name.startsWith('$');
  }

  isGPUCompatible() {
    const gpuTypes = new Set([
      'u32', 'i32', 'f32', 'f16', 'bool', 'vec2', 'vec3', 'vec4',
      'mat2', 'mat3', 'mat4', 'array', 'struct'
    ]);
    return gpuTypes.has(this.name);
  }
}

// Predefined types
const Types = {
  // Primitive types
  U32: new Type('u32'),
  I32: new Type('i32'),
  F32: new Type('f32'),
  F16: new Type('f16'),
  BOOL: new Type('bool'),
  STRING: new Type('string'),
  ATOM: new Type('atom'),
  
  // Generic types
  ANY: new Type('$any'),
  UNKNOWN: new Type('$unknown'),
  
  // Function type
  FUNCTION: (params, returnType) => new Type('function', [...params, returnType]),
  
  // Array type
  ARRAY: (elementType) => new Type('array', [elementType]),
  
  // Tuple type
  TUPLE: (elementTypes) => new Type('tuple', elementTypes),
  
  // Map type
  MAP: (keyType, valueType) => new Type('map', [keyType, valueType]),
  
  // Vector types (GPU)
  VEC2: new Type('vec2'),
  VEC3: new Type('vec3'),
  VEC4: new Type('vec4'),
  
  // Matrix types (GPU)
  MAT2: new Type('mat2'),
  MAT3: new Type('mat3'),
  MAT4: new Type('mat4'),
  
  // Custom type constructor
  CUSTOM: (name, params = []) => new Type(name, params)
};

class TypeEnvironment {
  constructor(parent = null) {
    this.parent = parent;
    this.types = new Map();
    this.variables = new Map();
  }

  defineType(name, type) {
    this.types.set(name, type);
  }

  defineVariable(name, type) {
    this.variables.set(name, type);
  }

  lookupType(name) {
    if (this.types.has(name)) {
      return this.types.get(name);
    }
    if (this.parent) {
      return this.parent.lookupType(name);
    }
    return null;
  }

  lookupVariable(name) {
    if (this.variables.has(name)) {
      return this.variables.get(name);
    }
    if (this.parent) {
      return this.parent.lookupVariable(name);
    }
    return null;
  }

  extend() {
    return new TypeEnvironment(this);
  }
}

class TypeConstraint {
  constructor(type, constraints = {}) {
    this.type = type;
    this.constraints = constraints;
  }
}

class TypeError extends Error {
  constructor(message, node) {
    super(message);
    this.node = node;
    this.name = 'TypeError';
  }
}

class TypeChecker {
  constructor() {
    this.env = new TypeEnvironment();
    this.gpuCompatibleTypes = new Set([
      'u32', 'i32', 'f32', 'f16', 'bool', 'vec2', 'vec3', 'vec4',
      'mat2', 'mat3', 'mat4'
    ]);
    
    // Initialize built-in types
    this.initializeBuiltinTypes();
  }

  initializeBuiltinTypes() {
    // Primitive types
    this.env.defineType('u32', Types.U32);
    this.env.defineType('i32', Types.I32);
    this.env.defineType('f32', Types.F32);
    this.env.defineType('f16', Types.F16);
    this.env.defineType('bool', Types.BOOL);
    this.env.defineType('string', Types.STRING);
    this.env.defineType('atom', Types.ATOM);
    
    // GPU vector types
    this.env.defineType('vec2', Types.VEC2);
    this.env.defineType('vec3', Types.VEC3);
    this.env.defineType('vec4', Types.VEC4);
    
    // GPU matrix types
    this.env.defineType('mat2', Types.MAT2);
    this.env.defineType('mat3', Types.MAT3);
    this.env.defineType('mat4', Types.MAT4);
  }

  check(ast) {
    if (!ast) {
      throw new TypeError('Empty AST cannot be type-checked');
    }

    switch (ast.type) {
      case 'Module':
        return this.checkModule(ast);
      case 'FunctionDef':
        return this.checkFunctionDef(ast);
      case 'MacroDef':
        return this.checkMacroDef(ast);
      case 'ActorDef':
        return this.checkActorDef(ast);
      case 'PlanetDef':
        return this.checkPlanetDef(ast);
      case 'StarDef':
        return this.checkStarDef(ast);
      case 'Literal':
        return this.checkLiteral(ast);
      case 'Identifier':
        return this.checkIdentifier(ast);
      case 'BinaryOp':
        return this.checkBinaryOp(ast);
      case 'UnaryOp':
        return this.checkUnaryOp(ast);
      case 'FunctionCall':
        return this.checkFunctionCall(ast);
      case 'CaseExpr':
        return this.checkCaseExpr(ast);
      case 'IfExpr':
        return this.checkIfExpr(ast);
      case 'BlockExpr':
        return this.checkBlockExpr(ast);
      case 'ListExpr':
        return this.checkListExpr(ast);
      case 'TupleExpr':
        return this.checkTupleExpr(ast);
      case 'MapExpr':
        return this.checkMapExpr(ast);
      case 'Pattern':
        return this.checkPattern(ast);
      default:
        throw new TypeError(`Unknown AST node type: ${ast.type}`, ast);
    }
  }

  checkModule(module) {
    const moduleEnv = this.env.extend();
    const oldEnv = this.env;
    this.env = moduleEnv;

    try {
      for (const def of module.body) {
        this.check(def);
      }
      return Types.ANY;
    } finally {
      this.env = oldEnv;
    }
  }

  checkFunctionDef(funcDef) {
    const funcEnv = this.env.extend();
    const oldEnv = this.env;
    this.env = funcEnv;

    try {
      const paramTypes = [];
      
      // Check parameters and their type annotations
      for (const param of funcDef.params) {
        if (param.patternType === 'variable') {
          let paramType = Types.UNKNOWN;
          
          if (param.value.type) {
            paramType = this.resolveTypeAnnotation(param.value.type);
          }
          
          this.env.defineVariable(param.value.name, paramType);
          paramTypes.push(paramType);
        } else {
          const inferredType = this.inferPatternType(param);
          paramTypes.push(inferredType);
        }
      }

      // Check function body
      const returnType = this.check(funcDef.body);

      // If GPU kernel, validate all types are GPU-compatible
      if (funcDef.annotations.includes('gpu_kernel')) {
        this.validateGPUTypes(funcDef, paramTypes, returnType);
      }

      // Create function type
      const functionType = Types.FUNCTION(paramTypes, returnType);
      
      // Define function in outer environment
      this.env.defineVariable(funcDef.name, functionType);
      
      return functionType;
    } finally {
      this.env = oldEnv;
    }
  }

  checkMacroDef(macroDef) {
    // Macros are checked at expansion time, not definition time
    // For now, just ensure the macro name is defined
    this.env.defineVariable(macroDef.name, Types.ANY);
    return Types.ANY;
  }

  checkActorDef(actorDef) {
    const actorEnv = this.env.extend();
    const oldEnv = this.env;
    this.env = actorEnv;

    try {
      // Check state fields
      if (actorDef.state) {
        for (const field of actorDef.state) {
          const fieldType = this.resolveTypeAnnotation(field.type);
          this.env.defineVariable(field.name, fieldType);
        }
      }

      // Check message handlers
      for (const handler of actorDef.handlers) {
        this.check(handler);
      }

      return Types.CUSTOM('Actor', [actorDef.name]);
    } finally {
      this.env = oldEnv;
    }
  }

  checkPlanetDef(planetDef) {
    // Create a product type
    const fieldTypes = {};
    
    for (const orbital of planetDef.orbitals) {
      const fieldType = this.resolveTypeAnnotation(orbital.type);
      fieldTypes[orbital.name] = fieldType;
    }

    const planetType = Types.CUSTOM(planetDef.name, Object.values(fieldTypes));
    this.env.defineType(planetDef.name, planetType);
    
    return planetType;
  }

  checkStarDef(starDef) {
    // Create a sum type
    const variantTypes = [];
    
    for (const layer of starDef.layers) {
      const fieldTypes = layer.fields.map(f => this.resolveTypeAnnotation(f.type));
      const variantType = Types.CUSTOM(layer.constructor, fieldTypes);
      variantTypes.push(variantType);
    }

    const starType = Types.CUSTOM(starDef.name, variantTypes);
    this.env.defineType(starDef.name, starType);
    
    return starType;
  }

  checkLiteral(literal) {
    switch (literal.literalType) {
      case 'integer':
        return Types.I32;
      case 'float':
        return Types.F32;
      case 'string':
        return Types.STRING;
      case 'atom':
        return Types.ATOM;
      case 'boolean':
        return Types.BOOL;
      default:
        return Types.UNKNOWN;
    }
  }

  checkIdentifier(identifier) {
    const varType = this.env.lookupVariable(identifier.name);
    if (varType) {
      return varType;
    }
    
    const typeType = this.env.lookupType(identifier.name);
    if (typeType) {
      return typeType;
    }
    
    // For demo purposes, assume unknown identifiers are numeric
    // In a real implementation, this would be an error
    this.env.defineVariable(identifier.name, Types.F32);
    return Types.F32;
  }

  checkBinaryOp(binaryOp) {
    const leftType = this.check(binaryOp.left);
    const rightType = this.check(binaryOp.right);

    switch (binaryOp.operator) {
      case '+':
      case '-':
      case '*':
      case '/':
        return this.checkArithmeticOp(leftType, rightType, binaryOp);
      
      case '==':
      case '!=':
        return this.checkEqualityOp(leftType, rightType, binaryOp);
      
      case '<':
      case '>':
      case '<=':
      case '>=':
        return this.checkComparisonOp(leftType, rightType, binaryOp);
      
      case 'and':
      case 'or':
        return this.checkLogicalOp(leftType, rightType, binaryOp);
      
      default:
        throw new TypeError(`Unknown binary operator: ${binaryOp.operator}`, binaryOp);
    }
  }

  checkUnaryOp(unaryOp) {
    const operandType = this.check(unaryOp.operand);

    switch (unaryOp.operator) {
      case '-':
        if (!this.isNumericType(operandType)) {
          throw new TypeError(`Cannot apply unary minus to non-numeric type: ${operandType}`, unaryOp);
        }
        return operandType;
      
      case 'not':
        if (!operandType.equals(Types.BOOL)) {
          throw new TypeError(`Cannot apply logical not to non-boolean type: ${operandType}`, unaryOp);
        }
        return Types.BOOL;
      
      default:
        throw new TypeError(`Unknown unary operator: ${unaryOp.operator}`, unaryOp);
    }
  }

  checkFunctionCall(call) {
    const funcType = this.check(call.func);
    
    if (funcType.name !== 'function') {
      throw new TypeError(`Cannot call non-function type: ${funcType}`, call);
    }

    const argTypes = call.args.map(arg => this.check(arg));
    const paramTypes = funcType.params.slice(0, -1);
    const returnType = funcType.params[funcType.params.length - 1];

    if (argTypes.length !== paramTypes.length) {
      throw new TypeError(
        `Function expects ${paramTypes.length} arguments but got ${argTypes.length}`,
        call
      );
    }

    for (let i = 0; i < argTypes.length; i++) {
      if (!this.isTypeCompatible(argTypes[i], paramTypes[i])) {
        throw new TypeError(
          `Argument ${i + 1} type mismatch: expected ${paramTypes[i]} but got ${argTypes[i]}`,
          call.args[i]
        );
      }
    }

    return returnType;
  }

  checkCaseExpr(caseExpr) {
    const scrutineeType = this.check(caseExpr.scrutinee);
    
    for (const clause of caseExpr.clauses) {
      const patternEnv = this.env.extend();
      const oldEnv = this.env;
      this.env = patternEnv;

      try {
        // Check pattern and bind variables
        this.checkPattern(clause.pattern, scrutineeType);
        
        // Check guard if present
        if (clause.guard) {
          const guardType = this.check(clause.guard);
          if (!guardType.equals(Types.BOOL)) {
            throw new TypeError(`Case guard must be boolean, got ${guardType}`, clause.guard);
          }
        }
        
        // Check body
        this.check(clause.body);
      } finally {
        this.env = oldEnv;
      }
    }

    return Types.UNKNOWN; // Case expressions can have different return types
  }

  checkIfExpr(ifExpr) {
    const conditionType = this.check(ifExpr.condition);
    if (!conditionType.equals(Types.BOOL)) {
      throw new TypeError(`If condition must be boolean, got ${conditionType}`, ifExpr.condition);
    }

    const thenType = this.check(ifExpr.thenBranch);
    const elseType = ifExpr.elseBranch ? this.check(ifExpr.elseBranch) : Types.UNKNOWN;
    
    // Return the common type or unknown
    return this.findCommonType(thenType, elseType);
  }

  checkBlockExpr(blockExpr) {
    let lastType = Types.UNKNOWN;
    
    for (const expr of blockExpr.expressions) {
      lastType = this.check(expr);
    }
    
    return lastType;
  }

  checkListExpr(listExpr) {
    const elementTypes = listExpr.elements.map(elem => this.check(elem));
    const commonType = this.findCommonType(...elementTypes);
    return Types.ARRAY(commonType);
  }

  checkTupleExpr(tupleExpr) {
    const elementTypes = tupleExpr.elements.map(elem => this.check(elem));
    return Types.TUPLE(elementTypes);
  }

  checkMapExpr(mapExpr) {
    const keyTypes = [];
    const valueTypes = [];
    
    for (const pair of mapExpr.pairs) {
      keyTypes.push(this.check(pair.key));
      valueTypes.push(this.check(pair.value));
    }
    
    const commonKeyType = this.findCommonType(...keyTypes);
    const commonValueType = this.findCommonType(...valueTypes);
    
    return Types.MAP(commonKeyType, commonValueType);
  }

  checkPattern(pattern, expectedType = null) {
    switch (pattern.patternType) {
      case 'literal':
        return this.checkLiteral(pattern.value);
      
      case 'variable':
        let varType = Types.UNKNOWN;
        if (pattern.value.type) {
          varType = this.resolveTypeAnnotation(pattern.value.type);
        } else if (expectedType) {
          varType = expectedType;
        }
        this.env.defineVariable(pattern.value.name, varType);
        return varType;
      
      case 'list':
        const elementTypes = pattern.value.map(elem => this.checkPattern(elem));
        const commonElementType = this.findCommonType(...elementTypes);
        return Types.ARRAY(commonElementType);
      
      case 'tuple':
        const tupleElementTypes = pattern.value.map(elem => this.checkPattern(elem));
        return Types.TUPLE(tupleElementTypes);
      
      case 'constructor':
        const constructorType = this.env.lookupType(pattern.value.constructor);
        if (!constructorType) {
          throw new TypeError(`Unknown constructor: ${pattern.value.constructor}`, pattern);
        }
        return constructorType;
      
      default:
        return Types.UNKNOWN;
    }
  }

  resolveTypeAnnotation(typeAnnotation) {
    if (!typeAnnotation) {
      return Types.UNKNOWN;
    }

    const baseType = this.env.lookupType(typeAnnotation.baseType);
    if (!baseType) {
      throw new TypeError(`Unknown type: ${typeAnnotation.baseType}`, typeAnnotation);
    }

    if (typeAnnotation.constraints.params) {
      const paramTypes = typeAnnotation.constraints.params.map(param => 
        this.resolveTypeAnnotation(param)
      );
      return Types.CUSTOM(baseType.name, paramTypes);
    }

    return baseType;
  }

  validateGPUTypes(funcDef, paramTypes, returnType) {
    // Check parameter types
    for (let i = 0; i < paramTypes.length; i++) {
      if (!this.isGPUCompatible(paramTypes[i])) {
        throw new TypeError(
          `GPU kernel parameter ${i + 1} has non-GPU-compatible type: ${paramTypes[i]}`,
          funcDef.params[i]
        );
      }
    }

    // Check return type
    if (!this.isGPUCompatible(returnType)) {
      throw new TypeError(
        `GPU kernel return type is not GPU-compatible: ${returnType}`,
        funcDef
      );
    }

    // Additional validation for GPU kernels
    this.validateGPUKernelBody(funcDef.body);
  }

  validateGPUKernelBody(body) {
    // Recursively check that all operations in the body are GPU-compatible
    // This is a simplified version - a full implementation would need to
    // check for unsupported operations like recursion, dynamic allocation, etc.
    
    if (body.type === 'FunctionCall') {
      // Check if the called function is also a GPU kernel
      const funcType = this.env.lookupVariable(body.func.name);
      if (funcType && !funcType.annotations?.includes('gpu_kernel')) {
        throw new TypeError(
          `GPU kernel cannot call non-GPU function: ${body.func.name}`,
          body
        );
      }
    }
  }

  isGPUCompatible(type) {
    if (type.isGeneric()) {
      return true; // Generic types will be checked during instantiation
    }
    
    if (type.name === 'array') {
      return this.isGPUCompatible(type.params[0]);
    }
    
    if (type.name === 'tuple') {
      return type.params.every(param => this.isGPUCompatible(param));
    }
    
    return this.gpuCompatibleTypes.has(type.name);
  }

  isNumericType(type) {
    return type.equals(Types.I32) || type.equals(Types.F32) || 
           type.equals(Types.F16) || type.equals(Types.U32);
  }

  isTypeCompatible(source, target) {
    if (target.equals(Types.ANY) || target.equals(Types.UNKNOWN)) {
      return true;
    }
    
    if (source.equals(target)) {
      return true;
    }
    
    // Numeric type promotion
    if (this.isNumericType(source) && this.isNumericType(target)) {
      return true;
    }
    
    return false;
  }

  checkArithmeticOp(leftType, rightType, node) {
    // Handle unknown types gracefully for demo purposes
    if (leftType.equals(Types.UNKNOWN) || rightType.equals(Types.UNKNOWN)) {
      return Types.F32; // Assume numeric for unknown types in demo
    }
    
    if (!this.isNumericType(leftType) || !this.isNumericType(rightType)) {
      throw new TypeError(
        `Arithmetic operator requires numeric types, got ${leftType} and ${rightType}`,
        node
      );
    }
    
    // Return the "larger" type for promotion
    if (leftType.equals(Types.F32) || rightType.equals(Types.F32)) {
      return Types.F32;
    }
    
    return Types.I32;
  }

  checkEqualityOp(leftType, rightType, node) {
    if (!this.isTypeCompatible(leftType, rightType) && !this.isTypeCompatible(rightType, leftType)) {
      throw new TypeError(
        `Equality operator requires compatible types, got ${leftType} and ${rightType}`,
        node
      );
    }
    
    return Types.BOOL;
  }

  checkComparisonOp(leftType, rightType, node) {
    // Handle unknown types gracefully for demo purposes
    if (leftType.equals(Types.UNKNOWN) || rightType.equals(Types.UNKNOWN)) {
      return Types.BOOL; // Assume comparison works for unknown types
    }
    
    if (!this.isNumericType(leftType) || !this.isNumericType(rightType)) {
      throw new TypeError(
        `Comparison operator requires numeric types, got ${leftType} and ${rightType}`,
        node
      );
    }
    
    return Types.BOOL;
  }

  checkLogicalOp(leftType, rightType, node) {
    // Handle unknown types gracefully for demo purposes
    if (leftType.equals(Types.UNKNOWN) || rightType.equals(Types.UNKNOWN)) {
      return Types.BOOL; // Assume logical works for unknown types
    }
    
    if (!leftType.equals(Types.BOOL) || !rightType.equals(Types.BOOL)) {
      throw new TypeError(
        `Logical operator requires boolean types, got ${leftType} and ${rightType}`,
        node
      );
    }
    
    return Types.BOOL;
  }

  findCommonType(...types) {
    if (types.length === 0) return Types.UNKNOWN;
    if (types.length === 1) return types[0];
    
    // If all types are the same, return that type
    const firstType = types[0];
    if (types.every(type => type.equals(firstType))) {
      return firstType;
    }
    
    // For numeric types, find the common promotion type
    const numericTypes = types.filter(type => this.isNumericType(type));
    if (numericTypes.length === types.length) {
      if (numericTypes.some(type => type.equals(Types.F32))) {
        return Types.F32;
      }
      return Types.I32;
    }
    
    // Otherwise, return unknown
    return Types.UNKNOWN;
  }

  inferPatternType(pattern) {
    // This is a simplified version - full pattern type inference is more complex
    switch (pattern.patternType) {
      case 'literal':
        return this.checkLiteral(pattern.value);
      case 'variable':
        return pattern.value.type ? this.resolveTypeAnnotation(pattern.value.type) : Types.UNKNOWN;
      case 'list':
        return Types.ARRAY(Types.UNKNOWN);
      case 'tuple':
        return Types.TUPLE(pattern.value.map(() => Types.UNKNOWN));
      default:
        return Types.UNKNOWN;
    }
  }
}

// Export for both Node.js and ES6 modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { TypeChecker, Type, Types, TypeEnvironment, TypeError };
}

// ES6 module exports for browser
export { TypeChecker, Type, Types, TypeEnvironment, TypeError };