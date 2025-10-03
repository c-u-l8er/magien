/**
 * GPU Code Generator
 * Generates WGSL (WebGPU Shading Language) code from analyzed Zapp AST
 */

class CodeGenerationError extends Error {
  constructor(message, node) {
    super(message);
    this.node = node;
    this.name = 'CodeGenerationError';
  }
}

class GPUCodeGenerator {
  constructor(typeChecker) {
    this.typeChecker = typeChecker;
    this.generatedFunctions = new Map();
    this.bufferBindings = new Map();
    this.nextBindingIndex = 0;
  }

  generateFunction(funcDef, analysis) {
    if (!analysis.isParallelizable) {
      throw new CodeGenerationError(
        `Function ${funcDef.name} is not parallelizable and cannot be generated for GPU`,
        funcDef
      );
    }

    const functionName = funcDef.name;
    const params = funcDef.params;
    const body = funcDef.body;
    const workgroupSize = analysis.workgroupSize;

    // Generate parameter declarations
    const paramDecls = params.map((param, index) => {
      const paramName = param.patternType === 'variable' ? param.value.name : `param${index}`;
      const paramType = this._typeToWGSL(this.typeChecker.inferPatternType(param));
      return `  ${paramName}: ${paramType}`;
    }).join(',\n');

    // Generate function body
    const bodyCode = this._generateNode(body, { 
      functionName,
      isEntryPoint: true 
    });

    // Generate buffer bindings if needed
    const bufferBindings = this._generateBufferBindings(functionName);

    // Generate the complete WGSL function
    const wgslCode = `
// Generated from Zapp function: ${functionName}
@compute @workgroup_size(${workgroupSize.x}, ${workgroupSize.y}, ${workgroupSize.z})
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>${paramDecls ? ',\n' + paramDecls : ''}
) -> ${this._typeToWGSL(this.typeChecker.check(body))} {
${bodyCode}
}`;

    this.generatedFunctions.set(functionName, {
      wgsl: wgslCode,
      bindings: bufferBindings,
      workgroupSize
    });

    return wgslCode;
  }

  _generateNode(node, context) {
    if (!node) {
      return '    // Empty node';
    }

    switch (node.type) {
      case 'Literal':
        return this._generateLiteral(node);
      
      case 'Identifier':
        return this._generateIdentifier(node);
      
      case 'BinaryOp':
        return this._generateBinaryOp(node, context);
      
      case 'UnaryOp':
        return this._generateUnaryOp(node, context);
      
      case 'FunctionCall':
        return this._generateFunctionCall(node, context);
      
      case 'IfExpr':
        return this._generateIfExpr(node, context);
      
      case 'BlockExpr':
        return this._generateBlockExpr(node, context);
      
      case 'ListExpr':
        return this._generateListExpr(node, context);
      
      default:
        throw new CodeGenerationError(
          `Unsupported node type for GPU code generation: ${node.type}`,
          node
        );
    }
  }

  _generateLiteral(literal) {
    switch (literal.literalType) {
      case 'integer':
        return `    ${literal.value}i32`;
      case 'float':
        return `    ${literal.value}f32`;
      case 'boolean':
        return `    ${literal.value}`;
      case 'string':
        return `    // String literal: "${literal.value}"`;
      case 'atom':
        return `    // Atom: :${literal.value}`;
      default:
        return `    // Unknown literal type: ${literal.literalType}`;
    }
  }

  _generateIdentifier(identifier) {
    // Check if this is a built-in GPU function
    const builtinFunctions = new Map([
      ['builtin_global_id', 'global_id'],
      ['builtin_local_id', 'local_id'],
      ['builtin_workgroup_id', 'workgroup_id']
    ]);

    if (builtinFunctions.has(identifier.name)) {
      return `    ${builtinFunctions.get(identifier.name)}`;
    }

    return `    ${identifier.name}`;
  }

  _generateBinaryOp(binaryOp, context) {
    const leftCode = this._generateNode(binaryOp.left, context);
    const rightCode = this._generateNode(binaryOp.right, context);
    
    // Map Zapp operators to WGSL operators
    const operatorMap = {
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

    const wgslOperator = operatorMap[binaryOp.operator];
    if (!wgslOperator) {
      throw new CodeGenerationError(
        `Unsupported binary operator: ${binaryOp.operator}`,
        binaryOp
      );
    }

    // Handle type promotion for mixed operations
    const leftType = this.typeChecker.check(binaryOp.left);
    const rightType = this.typeChecker.check(binaryOp.right);
    const resultType = this.typeChecker.check(binaryOp);

    // Add type casts if needed
    let leftExpr = leftCode;
    let rightExpr = rightCode;

    if (!leftType.equals(rightType)) {
      if (this._isNumericType(leftType) && this._isNumericType(rightType)) {
        // Promote to common type
        if (resultType.equals(this.typeChecker.Types.F32)) {
          if (leftType.equals(this.typeChecker.Types.I32)) {
            leftExpr = `f32(${leftCode})`;
          }
          if (rightType.equals(this.typeChecker.Types.I32)) {
            rightExpr = `f32(${rightCode})`;
          }
        }
      }
    }

    return `    (${leftExpr} ${wgslOperator} ${rightExpr})`;
  }

  _generateUnaryOp(unaryOp, context) {
    const operandCode = this._generateNode(unaryOp.operand, context);
    
    switch (unaryOp.operator) {
      case '-':
        return `    (-${operandCode})`;
      case 'not':
        return `    (!${operandCode})`;
      default:
        throw new CodeGenerationError(
          `Unsupported unary operator: ${unaryOp.operator}`,
          unaryOp
        );
    }
  }

  _generateFunctionCall(call, context) {
    const argsCode = call.args.map(arg => 
      this._generateNode(arg, context)
    ).join(', ');

    // Handle built-in functions
    const builtinFunctions = new Map([
      ['abs', 'abs'],
      ['min', 'min'],
      ['max', 'max'],
      ['sqrt', 'sqrt'],
      ['sin', 'sin'],
      ['cos', 'cos'],
      ['tan', 'tan'],
      ['floor', 'floor'],
      ['ceil', 'ceil'],
      ['round', 'round']
    ]);

    if (builtinFunctions.has(call.func.name)) {
      return `    ${builtinFunctions.get(call.func.name)}(${argsCode})`;
    }

    // Check if this is a user-defined GPU function
    if (this.generatedFunctions.has(call.func.name)) {
      return `    ${call.func.name}(${argsCode})`;
    }

    throw new CodeGenerationError(
      `Unknown function: ${call.func.name}`,
      call
    );
  }

  _generateIfExpr(ifExpr, context) {
    const conditionCode = this._generateNode(ifExpr.condition, context);
    const thenCode = this._generateNode(ifExpr.thenBranch, context);
    const elseCode = ifExpr.elseBranch ? 
      this._generateNode(ifExpr.elseBranch, context) : '    0';

    return `    select(${elseCode}, ${thenCode}, ${conditionCode})`;
  }

  _generateBlockExpr(blockExpr, context) {
    const expressions = blockExpr.expressions.map(expr => 
      this._generateNode(expr, context)
    );

    // Return the last expression
    if (expressions.length > 0) {
      return expressions[expressions.length - 1];
    }

    return '    0';
  }

  _generateListExpr(listExpr, context) {
    // Generate array literal
    const elements = listExpr.elements.map(elem => 
      this._generateNode(elem, context)
    ).join(', ');

    return `    array(${elements})`;
  }

  _typeToWGSL(type) {
    if (!type) {
      return 'f32'; // Default type
    }

    switch (type.name) {
      case 'u32':
        return 'u32';
      case 'i32':
        return 'i32';
      case 'f32':
        return 'f32';
      case 'f16':
        return 'f16';
      case 'bool':
        return 'bool';
      case 'string':
        return 'array<u32>'; // Strings as UTF-32 arrays
      case 'array':
        const elementType = this._typeToWGSL(type.params[0]);
        return `array<${elementType}>`;
      case 'tuple':
        const elementTypes = type.params.map(param => this._typeToWGSL(param));
        return `vec${elementTypes.length}<${elementTypes.join(', ')}>`;
      case 'vec2':
        return 'vec2<f32>';
      case 'vec3':
        return 'vec3<f32>';
      case 'vec4':
        return 'vec4<f32>';
      case 'mat2':
        return 'mat2x2<f32>';
      case 'mat3':
        return 'mat3x3<f32>';
      case 'mat4':
        return 'mat4x4<f32>';
      default:
        return 'f32'; // Default to f32 for unknown types
    }
  }

  _isNumericType(type) {
    return type.equals(this.typeChecker.Types.I32) ||
           type.equals(this.typeChecker.Types.F32) ||
           type.equals(this.typeChecker.Types.F16) ||
           type.equals(this.typeChecker.Types.U32);
  }

  _generateBufferBindings(functionName) {
    const bindings = [];
    
    // Generate buffer bindings for function parameters that need them
    // This is a simplified version - a full implementation would analyze
    // memory access patterns and generate appropriate buffer layouts
    
    return bindings;
  }

  generateCompleteShader(functions) {
    const shaderParts = [];
    
    // Add type definitions and utility functions
    shaderParts.push(`
// Zapp GPU Runtime - Generated WGSL Shader
// Generated at: ${new Date().toISOString()}

// Utility functions
fn select<T>(cond: bool, true_value: T, false_value: T) -> T {
  if (cond) {
    return true_value;
  } else {
    return false_value;
  }
}

`);

    // Add all generated functions
    for (const [name, funcInfo] of this.generatedFunctions) {
      shaderParts.push(funcInfo.wgsl);
      shaderParts.push('\n');
    }

    // Add buffer layout definitions
    if (this.bufferBindings.size > 0) {
      shaderParts.push('\n// Buffer Layouts\n');
      for (const [name, binding] of this.bufferBindings) {
        shaderParts.push(binding);
        shaderParts.push('\n');
      }
    }

    return shaderParts.join('');
  }

  validateGeneratedCode(wgslCode) {
    // Basic validation of generated WGSL code
    const errors = [];
    
    // Check for balanced braces
    let braceCount = 0;
    let parenCount = 0;
    
    for (let i = 0; i < wgslCode.length; i++) {
      const char = wgslCode[i];
      if (char === '{') braceCount++;
      if (char === '}') braceCount--;
      if (char === '(') parenCount++;
      if (char === ')') parenCount--;
      
      if (braceCount < 0) {
        errors.push(`Unmatched closing brace at position ${i}`);
      }
      if (parenCount < 0) {
        errors.push(`Unmatched closing parenthesis at position ${i}`);
      }
    }
    
    if (braceCount !== 0) {
      errors.push(`Unmatched braces: ${braceCount} extra opening braces`);
    }
    
    if (parenCount !== 0) {
      errors.push(`Unmatched parentheses: ${parenCount} extra opening parentheses`);
    }
    
    // Check for required functions
    if (!wgslCode.includes('fn main(')) {
      errors.push('Generated shader missing main function');
    }
    
    return errors;
  }
}

// Export for both Node.js and ES6 modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { GPUCodeGenerator, CodeGenerationError };
}

// ES6 module exports for browser
export { GPUCodeGenerator, CodeGenerationError };