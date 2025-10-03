/**
 * Zapp Language Test Suite
 * Tests for lexer, type checker, GPU analyzer, and actor system
 */

// Import modules (would use proper module system in production)
const { Lexer, Token, TokenType } = require('../src/core/lexer');
const { TypeChecker, Type, Types } = require('../src/core/type_checker');
const { GPUAnalyzer } = require('../src/core/gpu_analyzer');
const { GPUCodeGenerator } = require('../src/core/gpu_codegen');
const { ActorRuntime } = require('../src/core/actor_system');

class TestSuite {
  constructor() {
    this.tests = [];
    this.passed = 0;
    this.failed = 0;
  }

  test(name, testFn) {
    this.tests.push({ name, testFn });
  }

  async run() {
    console.log('🧪 Running Zapp Test Suite...\n');
    
    for (const test of this.tests) {
      try {
        await test.testFn();
        console.log(`✅ ${test.name}`);
        this.passed++;
      } catch (error) {
        console.log(`❌ ${test.name}`);
        console.log(`   Error: ${error.message}`);
        this.failed++;
      }
    }
    
    console.log(`\n📊 Results: ${this.passed} passed, ${this.failed} failed`);
    return this.failed === 0;
  }

  assert(condition, message) {
    if (!condition) {
      throw new Error(message || 'Assertion failed');
    }
  }

  assertEqual(actual, expected, message) {
    if (actual !== expected) {
      throw new Error(message || `Expected ${expected}, got ${actual}`);
    }
  }

  assertThrows(fn, message) {
    try {
      fn();
      throw new Error(message || 'Expected function to throw');
    } catch (error) {
      // Expected behavior
    }
  }
}

// Lexer Tests
function createLexerTests() {
  const suite = new TestSuite();

  suite.test('Lexer should tokenize basic arithmetic', () => {
    const lexer = new Lexer('1 + 2 * 3');
    const tokens = lexer.tokenize();
    
    suite.assertEqual(tokens[0].type, TokenType.INTEGER);
    suite.assertEqual(tokens[0].value, 1);
    suite.assertEqual(tokens[1].type, TokenType.PLUS);
    suite.assertEqual(tokens[2].type, TokenType.INTEGER);
    suite.assertEqual(tokens[2].value, 2);
    suite.assertEqual(tokens[3].type, TokenType.STAR);
    suite.assertEqual(tokens[4].type, TokenType.INTEGER);
    suite.assertEqual(tokens[4].value, 3);
  });

  suite.test('Lexer should tokenize comparison operators', () => {
    const lexer = new Lexer('a <= b >= c == d != e');
    const tokens = lexer.tokenize();
    
    suite.assertEqual(tokens[1].type, TokenType.LTE);
    suite.assertEqual(tokens[3].type, TokenType.GTE);
    suite.assertEqual(tokens[5].type, TokenType.EQ);
    suite.assertEqual(tokens[7].type, TokenType.NEQ);
  });

  suite.test('Lexer should tokenize logical operators', () => {
    const lexer = new Lexer('true and false or not x');
    const tokens = lexer.tokenize();
    
    suite.assertEqual(tokens[0].type, TokenType.BOOLEAN);
    suite.assertEqual(tokens[0].value, true);
    suite.assertEqual(tokens[1].type, TokenType.AND);
    suite.assertEqual(tokens[2].type, TokenType.BOOLEAN);
    suite.assertEqual(tokens[2].value, false);
    suite.assertEqual(tokens[3].type, TokenType.OR);
    suite.assertEqual(tokens[4].type, TokenType.NOT);
  });

  suite.test('Lexer should tokenize function definitions', () => {
    const lexer = new Lexer('def add(x, y) do x + y end');
    const tokens = lexer.tokenize();
    
    suite.assertEqual(tokens[0].type, TokenType.DEF);
    suite.assertEqual(tokens[1].type, TokenType.IDENTIFIER);
    suite.assertEqual(tokens[1].value, 'add');
    suite.assertEqual(tokens[2].type, TokenType.LPAREN);
    suite.assertEqual(tokens[3].type, TokenType.IDENTIFIER);
    suite.assertEqual(tokens[3].value, 'x');
    suite.assertEqual(tokens[4].type, TokenType.COMMA);
    suite.assertEqual(tokens[5].type, TokenType.IDENTIFIER);
    suite.assertEqual(tokens[5].value, 'y');
    suite.assertEqual(tokens[6].type, TokenType.RPAREN);
    suite.assertEqual(tokens[7].type, TokenType.DO);
    suite.assertEqual(tokens[8].type, TokenType.END);
  });

  return suite;
}

// Type Checker Tests
function createTypeCheckerTests() {
  const suite = new TestSuite();

  suite.test('Type checker should infer basic types', () => {
    const typeChecker = new TypeChecker();
    
    // Test literal types
    const intLiteral = { type: 'Literal', literalType: 'integer', value: 42 };
    const intType = typeChecker.check(intLiteral);
    suite.assert(intType.equals(Types.I32));
    
    const floatLiteral = { type: 'Literal', literalType: 'float', value: 3.14 };
    const floatType = typeChecker.check(floatLiteral);
    suite.assert(floatType.equals(Types.F32));
    
    const boolLiteral = { type: 'Literal', literalType: 'boolean', value: true };
    const boolType = typeChecker.check(boolLiteral);
    suite.assert(boolType.equals(Types.BOOL));
  });

  suite.test('Type checker should validate binary operations', () => {
    const typeChecker = new TypeChecker();
    
    const intAdd = {
      type: 'BinaryOp',
      operator: '+',
      left: { type: 'Literal', literalType: 'integer', value: 1 },
      right: { type: 'Literal', literalType: 'integer', value: 2 }
    };
    
    const resultType = typeChecker.check(intAdd);
    suite.assert(resultType.equals(Types.I32));
  });

  suite.test('Type checker should reject invalid operations', () => {
    const typeChecker = new TypeChecker();
    
    const invalidOp = {
      type: 'BinaryOp',
      operator: '+',
      left: { type: 'Literal', literalType: 'string', value: 'hello' },
      right: { type: 'Literal', literalType: 'integer', value: 42 }
    };
    
    suite.assertThrows(() => typeChecker.check(invalidOp));
  });

  return suite;
}

// GPU Analyzer Tests
function createGPUAnalyzerTests() {
  const suite = new TestSuite();

  suite.test('GPU analyzer should accept simple parallel functions', () => {
    const typeChecker = new TypeChecker();
    const analyzer = new GPUAnalyzer(typeChecker);
    
    const parallelFunc = {
      type: 'FunctionDef',
      name: 'parallel_add',
      params: [
        { patternType: 'variable', value: { name: 'x' } },
        { patternType: 'variable', value: { name: 'y' } }
      ],
      annotations: ['gpu_kernel'],
      body: {
        type: 'BinaryOp',
        operator: '+',
        left: { type: 'Identifier', name: 'x' },
        right: { type: 'Identifier', name: 'y' }
      }
    };
    
    const analysis = analyzer.analyzeFunction(parallelFunc);
    suite.assert(analysis.isParallelizable);
    suite.assertEqual(analysis.errors.length, 0);
  });

  suite.test('GPU analyzer should reject recursive functions', () => {
    const typeChecker = new TypeChecker();
    const analyzer = new GPUAnalyzer(typeChecker);
    
    const recursiveFunc = {
      type: 'FunctionDef',
      name: 'factorial',
      params: [
        { patternType: 'variable', value: { name: 'n' } }
      ],
      annotations: ['gpu_kernel'],
      body: {
        type: 'IfExpr',
        condition: {
          type: 'BinaryOp',
          operator: '==',
          left: { type: 'Identifier', name: 'n' },
          right: { type: 'Literal', literalType: 'integer', value: 0 }
        },
        thenBranch: { type: 'Literal', literalType: 'integer', value: 1 },
        elseBranch: {
          type: 'BinaryOp',
          operator: '*',
          left: { type: 'Identifier', name: 'n' },
          right: {
            type: 'FunctionCall',
            func: { name: 'factorial' },
            args: [{
              type: 'BinaryOp',
              operator: '-',
              left: { type: 'Identifier', name: 'n' },
              right: { type: 'Literal', literalType: 'integer', value: 1 }
            }]
          }
        }
      }
    };
    
    const analysis = analyzer.analyzeFunction(recursiveFunc);
    suite.assert(!analysis.isParallelizable);
    suite.assert(analysis.hasRecursion);
    suite.assert(analysis.errors.length > 0);
  });

  return suite;
}

// GPU Code Generator Tests
function createGPUCodeGeneratorTests() {
  const suite = new TestSuite();

  suite.test('GPU code generator should produce valid WGSL', () => {
    const typeChecker = new TypeChecker();
    const codegen = new GPUCodeGenerator(typeChecker);
    const analyzer = new GPUAnalyzer(typeChecker);
    
    const simpleFunc = {
      type: 'FunctionDef',
      name: 'add_vectors',
      params: [
        { patternType: 'variable', value: { name: 'a' } },
        { patternType: 'variable', value: { name: 'b' } }
      ],
      annotations: ['gpu_kernel'],
      body: {
        type: 'BinaryOp',
        operator: '+',
        left: { type: 'Identifier', name: 'a' },
        right: { type: 'Identifier', name: 'b' }
      }
    };
    
    const analysis = analyzer.analyzeFunction(simpleFunc);
    const wgsl = codegen.generateFunction(simpleFunc, analysis);
    
    suite.assert(wgsl.includes('@compute'));
    suite.assert(wgsl.includes('@workgroup_size'));
    suite.assert(wgsl.includes('fn main('));
    suite.assert(wgsl.includes('global_id'));
  });

  suite.test('GPU code generator should handle type promotion', () => {
    const typeChecker = new TypeChecker();
    const codegen = new GPUCodeGenerator(typeChecker);
    const analyzer = new GPUAnalyzer(typeChecker);
    
    const mixedTypeFunc = {
      type: 'FunctionDef',
      name: 'mixed_add',
      params: [
        { patternType: 'variable', value: { name: 'x' } },
        { patternType: 'variable', value: { name: 'y' } }
      ],
      annotations: ['gpu_kernel'],
      body: {
        type: 'BinaryOp',
        operator: '+',
        left: { type: 'Literal', literalType: 'integer', value: 1 },
        right: { type: 'Literal', literalType: 'float', value: 2.5 }
      }
    };
    
    const analysis = analyzer.analyzeFunction(mixedTypeFunc);
    const wgsl = codegen.generateFunction(mixedTypeFunc, analysis);
    
    suite.assert(wgsl.includes('f32(')); // Should include type cast
  });

  return suite;
}

// Actor System Tests
function createActorSystemTests() {
  const suite = new TestSuite();

  suite.test('Actor system should spawn actors', async () => {
    const runtime = new ActorRuntime();
    await runtime.initialize();
    
    const actorDef = {
      name: 'TestActor',
      handlers: [
        {
          name: 'ping',
          params: [],
          body: { type: 'Literal', literalType: 'atom', value: 'pong' }
        }
      ]
    };
    
    const actorRef = await runtime.spawnActor(actorDef, { count: 0 });
    suite.assert(actorRef.pid);
    suite.assert(actorRef.mailbox);
    
    await runtime.shutdown();
  });

  suite.test('Actor system should handle messages', async () => {
    const runtime = new ActorRuntime();
    await runtime.initialize();
    
    const actorDef = {
      name: 'EchoActor',
      handlers: [
        {
          name: 'echo',
          params: ['message'],
          body: { type: 'Identifier', name: 'message' }
        }
      ]
    };
    
    const actorRef = await runtime.spawnActor(actorDef);
    
    const message = { type: 'echo', data: { text: 'hello' } };
    await actorRef.send(message);
    
    // Give some time for message processing
    await new Promise(resolve => setTimeout(resolve, 100));
    
    const status = runtime.getActorStatus(actorRef.pid);
    suite.assertEqual(status.state.lastMessage, 'echo');
    
    await runtime.shutdown();
  });

  return suite;
}

// Integration Tests
function createIntegrationTests() {
  const suite = new TestSuite();

  suite.test('End-to-end GPU compilation pipeline', async () => {
    const typeChecker = new TypeChecker();
    const analyzer = new GPUAnalyzer(typeChecker);
    const codegen = new GPUCodeGenerator(typeChecker);
    
    // Define a simple parallel function
    const funcDef = {
      type: 'FunctionDef',
      name: 'parallel_sum',
      params: [
        { patternType: 'variable', value: { name: 'array' } },
        { patternType: 'variable', value: { name: 'index' } }
      ],
      annotations: ['gpu_kernel'],
      body: {
        type: 'BinaryOp',
        operator: '+',
        left: { type: 'Identifier', name: 'array' },
        right: { type: 'Identifier', name: 'index' }
      }
    };
    
    // Type check
    const funcType = typeChecker.check(funcDef);
    suite.assert(funcType);
    
    // Analyze for parallelization
    const analysis = analyzer.analyzeFunction(funcDef);
    suite.assert(analysis.isParallelizable);
    
    // Generate WGSL
    const wgsl = codegen.generateFunction(funcDef, analysis);
    suite.assert(wgsl.length > 0);
    
    // Validate generated code
    const errors = codegen.validateGeneratedCode(wgsl);
    suite.assertEqual(errors.length, 0);
  });

  return suite;
}

// Run all tests
async function runAllTests() {
  console.log('🚀 Starting Zapp Language Test Suite\n');
  
  const testSuites = [
    createLexerTests(),
    createTypeCheckerTests(),
    createGPUAnalyzerTests(),
    createGPUCodeGeneratorTests(),
    createActorSystemTests(),
    createIntegrationTests()
  ];
  
  let totalPassed = 0;
  let totalFailed = 0;
  
  for (const suite of testSuites) {
    const success = await suite.run();
    totalPassed += suite.passed;
    totalFailed += suite.failed;
    console.log('');
  }
  
  console.log(`🏁 Final Results: ${totalPassed} tests passed, ${totalFailed} tests failed`);
  
  if (totalFailed === 0) {
    console.log('✅ All tests passed!');
  } else {
    console.log('❌ Some tests failed. Please review the implementation.');
  }
  
  return totalFailed === 0;
}

// Export for use in Node.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { runAllTests };
}

// Run tests if this file is executed directly
if (typeof window === 'undefined' && require.main === module) {
  runAllTests();
}