/**
 * GPU Parallelization Analyzer
 * Analyzes Zapp code for GPU compatibility and parallelization potential
 */

class ParallelizationError extends Error {
  constructor(message, node) {
    super(message);
    this.node = node;
    this.name = 'ParallelizationError';
  }
}

class GPUAnalyzer {
  constructor(typeChecker) {
    this.typeChecker = typeChecker;
    this.gpuCompatibleOps = new Set([
      '+', '-', '*', '/', '<', '>', '<=', '>=', '==', '!=', 'and', 'or'
    ]);
    this.requiredFunctions = new Set();
    this.analyzedFunctions = new Map();
  }

  analyzeFunction(funcDef) {
    if (this.analyzedFunctions.has(funcDef.name)) {
      return this.analyzedFunctions.get(funcDef.name);
    }

    const analysis = {
      isParallelizable: false,
      requiresSequential: false,
      hasRecursion: false,
      hasDynamicAllocation: false,
      hasControlFlow: false,
      workgroupSize: { x: 256, y: 1, z: 1 },
      memoryRequirements: {},
      dependencies: [],
      errors: []
    };

    try {
      // Check if function has GPU kernel annotation
      if (!funcDef.annotations.includes('gpu_kernel')) {
        analysis.errors.push({
          type: 'missing_annotation',
          message: `Function ${funcDef.name} lacks @gpu_kernel annotation`,
          node: funcDef
        });
        this.analyzedFunctions.set(funcDef.name, analysis);
        return analysis;
      }

      // Analyze function body for parallelizability
      this._analyzeNode(funcDef.body, analysis);

      // Determine if function is parallelizable
      analysis.isParallelizable = !analysis.requiresSequential && 
                                !analysis.hasRecursion && 
                                !analysis.hasDynamicAllocation &&
                                analysis.errors.length === 0;

      // Calculate optimal workgroup size based on memory requirements
      this._calculateWorkgroupSize(analysis);

    } catch (error) {
      analysis.errors.push({
        type: 'analysis_error',
        message: error.message,
        node: error.node || funcDef
      });
    }

    this.analyzedFunctions.set(funcDef.name, analysis);
    return analysis;
  }

  _analyzeNode(node, analysis, context = {}) {
    if (!node) return;

    switch (node.type) {
      case 'FunctionCall':
        this._analyzeFunctionCall(node, analysis, context);
        break;
      
      case 'BinaryOp':
        this._analyzeBinaryOp(node, analysis, context);
        break;
      
      case 'UnaryOp':
        this._analyzeUnaryOp(node, analysis, context);
        break;
      
      case 'IfExpr':
        this._analyzeIfExpr(node, analysis, context);
        break;
      
      case 'CaseExpr':
        this._analyzeCaseExpr(node, analysis, context);
        break;
      
      case 'BlockExpr':
        this._analyzeBlockExpr(node, analysis, context);
        break;
      
      case 'ListExpr':
        this._analyzeListExpr(node, analysis, context);
        break;
      
      case 'Literal':
      case 'Identifier':
        // Literals and identifiers are always safe
        break;
      
      default:
        analysis.errors.push({
          type: 'unsupported_construct',
          message: `Unsupported construct for GPU: ${node.type}`,
          node: node
        });
    }
  }

  _analyzeFunctionCall(call, analysis, context) {
    // Check if this is a recursive call
    if (call.func.name === context.currentFunction) {
      analysis.hasRecursion = true;
      analysis.errors.push({
        type: 'recursion',
        message: `Recursive function ${call.func.name} cannot be executed on GPU`,
        node: call
      });
      return;
    }

    // Check if the called function is also a GPU kernel
    const funcAnalysis = this.analyzedFunctions.get(call.func.name);
    if (funcAnalysis && !funcAnalysis.isParallelizable) {
      analysis.requiresSequential = true;
      analysis.errors.push({
        type: 'sequential_dependency',
        message: `Function ${call.func.name} is not GPU-compatible`,
        node: call
      });
    }

    // Check for functions that require sequential execution
    const sequentialFunctions = new Set([
      'spawn', 'send', 'receive', 'spawn_actor', 'self'
    ]);
    
    if (sequentialFunctions.has(call.func.name)) {
      analysis.requiresSequential = true;
      analysis.errors.push({
        type: 'sequential_operation',
        message: `Function ${call.func.name} requires sequential execution`,
        node: call
      });
    }

    // Check for dynamic allocation functions
    const allocationFunctions = new Set([
      'alloc', 'malloc', 'new', 'create_buffer'
    ]);
    
    if (allocationFunctions.has(call.func.name)) {
      analysis.hasDynamicAllocation = true;
      analysis.errors.push({
        type: 'dynamic_allocation',
        message: `Function ${call.func.name} performs dynamic allocation`,
        node: call
      });
    }

    // Analyze arguments
    call.args.forEach(arg => this._analyzeNode(arg, analysis, context));
  }

  _analyzeBinaryOp(op, analysis, context) {
    if (!this.gpuCompatibleOps.has(op.operator)) {
      analysis.errors.push({
        type: 'unsupported_operator',
        message: `Operator ${op.operator} is not supported on GPU`,
        node: op
      });
    }

    this._analyzeNode(op.left, analysis, context);
    this._analyzeNode(op.right, analysis, context);
  }

  _analyzeUnaryOp(op, analysis, context) {
    if (op.operator !== '-' && op.operator !== 'not') {
      analysis.errors.push({
        type: 'unsupported_operator',
        message: `Unary operator ${op.operator} is not supported on GPU`,
        node: op
      });
    }

    this._analyzeNode(op.operand, analysis, context);
  }

  _analyzeIfExpr(ifExpr, analysis, context) {
    analysis.hasControlFlow = true;
    
    // Check if condition is based on global ID (parallelizable)
    const isParallelCondition = this._isParallelCondition(ifExpr.condition);
    
    if (!isParallelCondition) {
      analysis.requiresSequential = true;
      analysis.errors.push({
        type: 'sequential_control_flow',
        message: 'If condition is not based on global ID, causing divergence',
        node: ifExpr
      });
    }

    this._analyzeNode(ifExpr.condition, analysis, context);
    this._analyzeNode(ifExpr.thenBranch, analysis, context);
    
    if (ifExpr.elseBranch) {
      this._analyzeNode(ifExpr.elseBranch, analysis, context);
    }
  }

  _analyzeCaseExpr(caseExpr, analysis, context) {
    analysis.hasControlFlow = true;
    analysis.requiresSequential = true;
    
    analysis.errors.push({
      type: 'sequential_control_flow',
      message: 'Case expressions cause thread divergence and cannot be parallelized',
      node: caseExpr
    });

    this._analyzeNode(caseExpr.scrutinee, analysis, context);
    
    caseExpr.clauses.forEach(clause => {
      this._analyzeNode(clause.guard, analysis, context);
      this._analyzeNode(clause.body, analysis, context);
    });
  }

  _analyzeBlockExpr(blockExpr, analysis, context) {
    blockExpr.expressions.forEach(expr => {
      this._analyzeNode(expr, analysis, context);
    });
  }

  _analyzeListExpr(listExpr, analysis, context) {
    listExpr.elements.forEach(element => {
      this._analyzeNode(element, analysis, context);
    });
  }

  _isParallelCondition(condition) {
    // Check if condition is based on global ID or other parallel-safe constructs
    if (condition.type === 'FunctionCall') {
      const parallelFunctions = new Set([
        'builtin_global_id', 'builtin_local_id', 'builtin_workgroup_id'
      ]);
      return parallelFunctions.has(condition.func.name);
    }
    
    if (condition.type === 'BinaryOp') {
      // Check if both sides of comparison are based on global ID
      return this._isParallelCondition(condition.left) && 
             this._isParallelCondition(condition.right);
    }
    
    return false;
  }

  _calculateWorkgroupSize(analysis) {
    // Calculate optimal workgroup size based on memory requirements and complexity
    const memoryPerThread = this._estimateMemoryUsage(analysis);
    const maxSharedMemory = 32768; // 32KB typical limit
    
    // Adjust workgroup size based on memory usage
    if (memoryPerThread > 0) {
      const maxThreadsByMemory = Math.floor(maxSharedMemory / memoryPerThread);
      analysis.workgroupSize.x = Math.min(256, maxThreadsByMemory);
    }
    
    // Ensure workgroup size is a power of 2 and within limits
    analysis.workgroupSize.x = Math.pow(2, Math.floor(Math.log2(analysis.workgroupSize.x)));
    analysis.workgroupSize.x = Math.max(1, Math.min(256, analysis.workgroupSize.x));
  }

  _estimateMemoryUsage(analysis) {
    // Estimate memory usage per thread in bytes
    let memoryUsage = 0;
    
    // Base memory for registers
    memoryUsage += 64;
    
    // Additional memory for complex operations
    if (analysis.hasControlFlow) {
      memoryUsage += 32;
    }
    
    // Memory for function call stack (simplified)
    memoryUsage += analysis.dependencies.length * 16;
    
    return memoryUsage;
  }

  generateParallelizationReport(funcDef) {
    const analysis = this.analyzeFunction(funcDef);
    
    return {
      functionName: funcDef.name,
      isParallelizable: analysis.isParallelizable,
      workgroupSize: analysis.workgroupSize,
      issues: analysis.errors,
      recommendations: this._generateRecommendations(analysis),
      memoryEstimate: this._estimateMemoryUsage(analysis),
      performanceScore: this._calculatePerformanceScore(analysis)
    };
  }

  _generateRecommendations(analysis) {
    const recommendations = [];
    
    if (analysis.hasRecursion) {
      recommendations.push({
        type: 'refactor',
        message: 'Convert recursive function to iterative form for GPU execution'
      });
    }
    
    if (analysis.hasControlFlow) {
      recommendations.push({
        type: 'optimize',
        message: 'Consider using arithmetic operations instead of control flow to avoid thread divergence'
      });
    }
    
    if (analysis.hasDynamicAllocation) {
      recommendations.push({
        type: 'refactor',
        message: 'Replace dynamic allocation with pre-allocated buffers'
      });
    }
    
    if (analysis.requiresSequential) {
      recommendations.push({
        type: 'restructure',
        message: 'Restructure algorithm to use data-parallel patterns'
      });
    }
    
    return recommendations;
  }

  _calculatePerformanceScore(analysis) {
    let score = 100;
    
    if (analysis.hasRecursion) score -= 50;
    if (analysis.hasControlFlow) score -= 30;
    if (analysis.hasDynamicAllocation) score -= 40;
    if (analysis.requiresSequential) score -= 60;
    
    // Deduct points for errors
    score -= analysis.errors.length * 10;
    
    return Math.max(0, score);
  }
}

// Export for both Node.js and ES6 modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { GPUAnalyzer, ParallelizationError };
}

// ES6 module exports for browser
export { GPUAnalyzer, ParallelizationError };