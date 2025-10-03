/**
 * Comprehensive Test Suite for Interaction Net System
 * Tests core interaction net functionality, reduction rules, and GPU integration
 */

import { InteractionNet, Agent, Port, AgentType, PortType } from '../src/core/interaction_net.js';
import { NetParser, parseZappToNet, tokenizeZapp } from '../src/core/net_parser.js';

class InteractionNetTests {
  constructor() {
    this.tests = [];
    this.passed = 0;
    this.failed = 0;
  }

  assert(condition, message) {
    if (condition) {
      this.passed++;
      return true;
    } else {
      this.failed++;
      console.error(`❌ FAILED: ${message}`);
      return false;
    }
  }

  assertEqual(actual, expected, message) {
    return this.assert(actual === expected, `${message} - Expected: ${expected}, Got: ${actual}`);
  }

  assertNotEqual(actual, expected, message) {
    return this.assert(actual !== expected, `${message} - Should not equal: ${expected}, Got: ${actual}`);
  }

  // Test basic agent creation and port management
  testAgentCreation() {
    console.log('🧪 Testing Agent Creation...');
    
    const net = new InteractionNet();
    const agent = net.createAgent(AgentType.LAM, 1, 'x');
    
    this.assert(agent !== null, 'Agent should be created');
    this.assertEqual(agent.type, AgentType.LAM, 'Agent type should be LAM');
    this.assertEqual(agent.arity, 1, 'Agent arity should be 1');
    this.assertEqual(agent.data, 'x', 'Agent data should be x');
    this.assertEqual(agent.auxiliaryPorts.length, 1, 'Should have 1 auxiliary port');
    this.assert(agent.principalPort !== null, 'Should have principal port');
    
    console.log('✅ Agent creation tests passed');
  }

  // Test port connections and disconnections
  testPortConnections() {
    console.log('🧪 Testing Port Connections...');
    
    const net = new InteractionNet();
    const agent1 = net.createAgent(AgentType.CON, 0);
    const agent2 = net.createAgent(AgentType.CON, 0);
    
    // Test connection
    agent1.principalPort.connect(agent2.principalPort);
    this.assert(agent1.principalPort.isConnected(), 'Port should be connected');
    this.assert(agent2.principalPort.isConnected(), 'Other port should be connected');
    this.assertEqual(agent1.principalPort.connectedTo, agent2.principalPort, 'Ports should reference each other');
    
    // Test disconnection
    agent1.principalPort.disconnect();
    this.assert(!agent1.principalPort.isConnected(), 'Port should be disconnected');
    this.assert(!agent2.principalPort.isConnected(), 'Other port should be disconnected');
    this.assertEqual(agent1.principalPort.connectedTo, null, 'Port reference should be null');
    
    console.log('✅ Port connection tests passed');
  }

  // Test active pair detection
  testActivePairDetection() {
    console.log('🧪 Testing Active Pair Detection...');
    
    const net = new InteractionNet();
    const agent1 = net.createAgent(AgentType.LAM, 1);
    const agent2 = net.createAgent(AgentType.APP, 1);
    
    // Connect principal ports to create active pair
    agent1.principalPort.connect(agent2.principalPort);
    
    const activePairs = net.findActivePairs();
    this.assertEqual(activePairs.length, 1, 'Should find 1 active pair');
    this.assertEqual(activePairs[0].agent1, agent1, 'First agent should be lambda');
    this.assertEqual(activePairs[0].agent2, agent2, 'Second agent should be app');
    
    console.log('✅ Active pair detection tests passed');
  }

  // Test lambda-beta reduction
  testLambdaBetaReduction() {
    console.log('🧪 Testing Lambda-Beta Reduction...');
    
    const net = new InteractionNet();
    
    // Create (λx.x) y structure
    const lambda = net.createAgent(AgentType.LAM, 1, 'x');
    const app = net.createAgent(AgentType.APP, 1);
    const arg = net.createAgent(AgentType.CON, 0, 'y');
    
    // Connect: (λx.x) principal-to-principal with APP
    lambda.principalPort.connect(app.principalPort);
    // Connect argument to APP's auxiliary port
    app.auxiliaryPorts[0].connect(arg.principalPort);
    
    // Perform reduction
    const result = net.reduceToNormalForm(10);
    
    this.assert(result.normalForm, 'Should reach normal form');
    this.assert(result.steps > 0, 'Should perform reduction steps');
    
    console.log('✅ Lambda-beta reduction tests passed');
  }

  // Test arithmetic operations
  testArithmeticReduction() {
    console.log('🧪 Testing Arithmetic Reduction...');
    
    const net = new InteractionNet();
    
    // Create 2 + 3 structure
    const op2 = net.createAgent(AgentType.OP2, 2, 1); // Addition operator
    const num1 = net.createAgent(AgentType.NUM, 0, 2);
    const num2 = net.createAgent(AgentType.NUM, 0, 3);
    
    // Connect operation to numbers
    op2.auxiliaryPorts[0].connect(num1.principalPort);
    op2.auxiliaryPorts[1].connect(num2.principalPort);
    
    // Perform reduction
    const result = net.reduceToNormalForm(10);
    
    this.assert(result.steps > 0, 'Should perform reduction steps');
    
    console.log('✅ Arithmetic reduction tests passed');
  }

  // Test GPU buffer conversion
  testGPUBufferConversion() {
    console.log('🧪 Testing GPU Buffer Conversion...');
    
    const net = new InteractionNet();
    
    // Create a simple net
    const agent1 = net.createAgent(AgentType.LAM, 1, 'x');
    const agent2 = net.createAgent(AgentType.APP, 1);
    agent1.principalPort.connect(agent2.principalPort);
    
    // Convert to GPU buffer
    const buffer = net.toGPUBuffer();
    
    this.assert(buffer !== null, 'GPU buffer should be created');
    this.assertEqual(buffer.agents.length, 2, 'Should have 2 agents');
    this.assertEqual(buffer.ports.length, 4, 'Should have 4 ports');
    this.assert(buffer.metadata.agentCount > 0, 'Should have metadata');
    
    // Test round-trip conversion
    const restoredNet = InteractionNet.fromGPUBuffer(buffer);
    this.assertEqual(restoredNet.agents.size, 2, 'Restored net should have 2 agents');
    
    console.log('✅ GPU buffer conversion tests passed');
  }

  // Test Zapp parsing
  testZappParsing() {
    console.log('🧪 Testing Zapp Parsing...');
    
    // Test simple expression
    const sourceCode = '2 + 3';
    const tokens = tokenizeZapp(sourceCode);
    
    this.assert(tokens.length > 0, 'Should generate tokens');
    this.assertEqual(tokens[0].value, '2', 'First token should be 2');
    this.assertEqual(tokens[1].value, '+', 'Second token should be +');
    this.assertEqual(tokens[2].value, '3', 'Third token should be 3');
    
    // Test parsing to net
    const net = parseZappToNet(sourceCode);
    this.assert(net !== null, 'Should parse to net');
    this.assert(net.agents.size > 0, 'Should create agents');
    
    console.log('✅ Zapp parsing tests passed');
  }

  // Test complex expressions
  testComplexExpressions() {
    console.log('🧪 Testing Complex Expressions...');
    
    const expressions = [
      '(λx.x + 2) 3',
      '2 + 3 * 4',
      'λx.λy.x + y',
      '(λf.λx.f (f x)) (λy.y + 1)'
    ];
    
    for (const expr of expressions) {
      try {
        const net = parseZappToNet(expr);
        this.assert(net !== null, `Should parse: ${expr}`);
        this.assert(net.agents.size > 0, `Should create agents for: ${expr}`);
        
        const stats = net.getStats();
        console.log(`  📊 ${expr}: ${stats.agentCount} agents, ${stats.activePairs} active pairs`);
        
      } catch (error) {
        console.error(`  ❌ Failed to parse: ${expr} - ${error.message}`);
        this.failed++;
      }
    }
    
    console.log('✅ Complex expression tests passed');
  }

  // Test reduction limits and termination
  testReductionLimits() {
    console.log('🧪 Testing Reduction Limits...');
    
    const net = new InteractionNet();
    
    // Create a simple net that should terminate
    const agent1 = net.createAgent(AgentType.LAM, 1);
    const agent2 = net.createAgent(AgentType.APP, 1);
    agent1.principalPort.connect(agent2.principalPort);
    
    // Test with step limit
    const result = net.reduceToNormalForm(5);
    this.assert(result.steps <= 5, 'Should respect step limit');
    
    // Test normal form detection
    const finalStats = net.getStats();
    this.assertEqual(finalStats.activePairs, 0, 'Should have no active pairs in normal form');
    
    console.log('✅ Reduction limit tests passed');
  }

  // Test net statistics and debugging
  testNetStatistics() {
    console.log('🧪 Testing Net Statistics...');
    
    const net = new InteractionNet();
    
    // Create various agent types
    net.createAgent(AgentType.LAM, 1);
    net.createAgent(AgentType.APP, 1);
    net.createAgent(AgentType.NUM, 0, 42);
    net.createAgent(AgentType.OP2, 2, 1);
    
    const stats = net.getStats();
    
    this.assertEqual(stats.agentCount, 4, 'Should count 4 agents');
    this.assert(stats.typeDistribution.LAM === 1, 'Should count 1 LAM agent');
    this.assert(stats.typeDistribution.APP === 1, 'Should count 1 APP agent');
    this.assert(stats.typeDistribution.NUM === 1, 'Should count 1 NUM agent');
    this.assert(stats.typeDistribution.OP2 === 1, 'Should count 1 OP2 agent');
    
    // Test DOT generation
    const dotFormat = net.toDot();
    this.assert(dotFormat.includes('digraph'), 'Should generate valid DOT format');
    this.assert(dotFormat.includes('LAM'), 'Should include LAM in DOT');
    
    console.log('✅ Net statistics tests passed');
  }

  // Test error handling and edge cases
  testErrorHandling() {
    console.log('🧪 Testing Error Handling...');
    
    // Test empty net
    const emptyNet = new InteractionNet();
    const emptyResult = emptyNet.reduceToNormalForm(10);
    this.assertEqual(emptyResult.steps, 0, 'Empty net should not reduce');
    this.assert(emptyResult.normalForm, 'Empty net should be in normal form');
    
    // Test invalid connections
    const net = new InteractionNet();
    const agent = net.createAgent(AgentType.CON, 0);
    
    // Test self-connection (should be allowed but handled gracefully)
    agent.principalPort.connect(agent.principalPort);
    this.assert(agent.principalPort.isConnected(), 'Self-connection should be handled');
    
    // Test multiple connections
    const agent2 = net.createAgent(AgentType.CON, 0);
    agent.principalPort.disconnect();
    agent.principalPort.connect(agent2.principalPort);
    this.assert(agent.principalPort.isConnected(), 'Reconnection should work');
    
    console.log('✅ Error handling tests passed');
  }

  // Run all tests
  async runAllTests() {
    console.log('🚀 Starting Interaction Net Test Suite...\n');
    
    const startTime = performance.now();
    
    // Run individual test methods
    this.testAgentCreation();
    this.testPortConnections();
    this.testActivePairDetection();
    this.testLambdaBetaReduction();
    this.testArithmeticReduction();
    this.testGPUBufferConversion();
    this.testZappParsing();
    this.testComplexExpressions();
    this.testReductionLimits();
    this.testNetStatistics();
    this.testErrorHandling();
    
    const endTime = performance.now();
    const duration = endTime - startTime;
    
    // Print summary
    console.log('\n📊 Test Results:');
    console.log('═══════════════════════════');
    console.log(`✅ Passed: ${this.passed}`);
    console.log(`❌ Failed: ${this.failed}`);
    console.log(`⏱️  Duration: ${duration.toFixed(2)}ms`);
    console.log(`📈 Success Rate: ${((this.passed / (this.passed + this.failed)) * 100).toFixed(1)}%`);
    
    if (this.failed === 0) {
      console.log('\n🎉 All tests passed! Interaction net system is working correctly.');
    } else {
      console.log('\n⚠️  Some tests failed. Please review the implementation.');
    }
    
    return {
      passed: this.passed,
      failed: this.failed,
      duration,
      successRate: (this.passed / (this.passed + this.failed)) * 100
    };
  }
}

// Export for both Node.js and ES6 modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { InteractionNetTests };
}

// ES6 module exports for browser
export { InteractionNetTests };

// Auto-run tests if this file is executed directly
if (typeof window !== 'undefined') {
  // Browser environment - tests will be run manually
  console.log('Interaction Net Tests loaded. Call runTests() to execute.');
} else if (typeof global !== 'undefined') {
  // Node.js environment - run tests automatically
  const tests = new InteractionNetTests();
  tests.runAllTests().catch(console.error);
}