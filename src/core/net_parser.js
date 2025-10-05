/**
 * Zapp to Interaction Net Parser
 * Converts Zapp syntax into interaction net representation
 */

import { InteractionNet, Agent, AgentType } from './interaction_net.js';

class NetParser {
  constructor() {
    this.net = new InteractionNet();
    this.variableMap = new Map(); // variable -> agent
    this.lambdaStack = []; // Stack for lambda abstractions
  }

  parse(tokens) {
    this.net = new InteractionNet();
    this.variableMap.clear();
    this.lambdaStack = [];

    // Parse tokens into net
    this.parseTokens(tokens);

    // Create root agent if needed
    if (this.net.agents.size > 0) {
      this.createRootAgent();
    }

    return this.net;
  }

  parseTokens(tokens) {
    let i = 0;
    
    while (i < tokens.length) {
      const token = tokens[i];
      
      switch (token.type) {
        case 'DEF':
          i = this.parseFunction(tokens, i);
          break;
          
        case 'LAMBDA':
          i = this.parseLambda(tokens, i);
          break;
          
        case 'IDENTIFIER':
          i = this.parseExpression(tokens, i);
          break;
          
        case 'INTEGER':
        case 'FLOAT':
          // If this is part of an arithmetic expression, parse the whole expression
          if (this.isArithmeticExpression(tokens, i)) {
            i = this.parseExpressionWithPrecedence(tokens, i);
          } else {
            i = this.parseLiteral(tokens, i);
          }
          break;
          
        case '(':
          // Check if this is a lambda application: (λx.body) arg
          if (i + 1 < tokens.length && tokens[i + 1].type === 'LAMBDA') {
            i = this.parseLambdaApplication(tokens, i);
          } else {
            i = this.parseParenthesized(tokens, i);
          }
          break;
          
        default:
          i++;
      }
    }
  }
  
  isArithmeticExpression(tokens, startIndex) {
    // Check if this token is followed by an operator
    if (startIndex + 1 < tokens.length) {
      const nextToken = tokens[startIndex + 1];
      return this.isBinaryOperator(tokens, startIndex + 1);
    }
    return false;
  }

  parseFunction(tokens, startIndex) {
    let i = startIndex + 1; // Skip DEF
    
    if (i >= tokens.length || tokens[i].type !== 'IDENTIFIER') {
      return i;
    }
    
    const functionName = tokens[i].value;
    i++;
    
    // Skip parameters for now (simplified)
    while (i < tokens.length && tokens[i].type !== '=' && tokens[i].type !== 'DO') {
      i++;
    }
    
    // Skip the '=' or 'DO' token
    if (i < tokens.length && (tokens[i].type === '=' || tokens[i].type === 'DO')) {
      i++;
      
      // Parse function body - look for the next function definition or end
      const bodyEnd = this.findFunctionBodyEnd(tokens, i);
      const bodyTokens = tokens.slice(i, bodyEnd);
      
      // Parse body as expression
      this.parseTokens(bodyTokens);
      
      i = bodyEnd;
    }
    
    return i;
  }

  parseExpression(tokens, startIndex) {
    // Check if this is a lambda expression
    if (startIndex < tokens.length && tokens[startIndex].type === 'LAMBDA') {
      return this.parseLambda(tokens, startIndex);
    }
    
    // Check if this is an arithmetic expression
    if (this.isArithmeticExpression(tokens, startIndex)) {
      return this.parseExpressionWithPrecedence(tokens, startIndex);
    }
    
    // Handle simple identifiers
    if (startIndex < tokens.length && tokens[startIndex].type === 'IDENTIFIER') {
      const varName = tokens[startIndex].value;
      this.createVariableReference(varName);
      return startIndex + 1;
    }
    
    return startIndex;
  }

  parseBinaryOperation(tokens, startIndex) {
    // Parse with operator precedence using a simple shunting-yard approach
    return this.parseExpressionWithPrecedence(tokens, startIndex);
  }
  
  parseExpressionWithPrecedence(tokens, startIndex) {
    // Check if this is a lambda expression - if so, handle it separately
    if (startIndex < tokens.length && tokens[startIndex].type === 'LAMBDA') {
      return this.parseLambda(tokens, startIndex);
    }
    
    // For now, implement simple precedence: * and / before + and -
    // This is a simplified version - a full implementation would use shunting-yard algorithm
    
    let i = startIndex;
    const operators = [];
    const operands = [];
    
    // Parse first operand
    const firstOperandEnd = this.findOperandEnd(tokens, i);
    const firstOperandTokens = tokens.slice(i, firstOperandEnd);
    
    // Store the current net state to track agents created by this operand
    const beforeOperandCount = this.net.agents.size;
    
    // Check if this is a complex expression (like parentheses or lambda)
    if (firstOperandTokens.length === 1 &&
        (firstOperandTokens[0].type === 'INTEGER' || firstOperandTokens[0].type === 'FLOAT')) {
      // Simple literal - parse directly
      this.parseLiteral(firstOperandTokens, 0);
    } else if (firstOperandTokens.length > 0) {
      // Complex expression - parse but avoid infinite recursion
      this.parseTokensSimple(firstOperandTokens);
    }
    
    const afterOperandCount = this.net.agents.size;
    
    // Get the agents created by this operand
    const agents = Array.from(this.net.agents.values());
    const operandAgents = agents.slice(beforeOperandCount);
    operands.push(...operandAgents);
    
    i = firstOperandEnd;
    
    // Parse remaining operators and operands
    while (i < tokens.length && this.isBinaryOperator(tokens, i)) {
      const opToken = tokens[i];
      i++;
      
      // Parse next operand
      const nextOperandEnd = this.findOperandEnd(tokens, i);
      const nextOperandTokens = tokens.slice(i, nextOperandEnd);
      
      const beforeNextOperandCount = this.net.agents.size;
      
      // Check if this is a complex expression
      if (nextOperandTokens.length === 1 &&
          (nextOperandTokens[0].type === 'INTEGER' || nextOperandTokens[0].type === 'FLOAT')) {
        // Simple literal - parse directly
        this.parseLiteral(nextOperandTokens, 0);
      } else if (nextOperandTokens.length > 0) {
        // Complex expression - parse but avoid infinite recursion
        this.parseTokensSimple(nextOperandTokens);
      }
      
      const afterNextOperandCount = this.net.agents.size;
      
      const nextAgents = Array.from(this.net.agents.values());
      const nextOperandAgents = nextAgents.slice(beforeNextOperandCount);
      operands.push(...nextOperandAgents);
      
      operators.push(opToken.value);
      i = nextOperandEnd;
    }
    
    // Now build the expression tree with precedence
    this.buildExpressionTree(operators, operands);
    
    return i;
  }
  
  // Simplified token parsing that avoids infinite recursion
  parseTokensSimple(tokens) {
    let i = 0;
    
    while (i < tokens.length) {
      const token = tokens[i];
      
      switch (token.type) {
        case 'INTEGER':
        case 'FLOAT':
          i = this.parseLiteral(tokens, i);
          break;
          
        case 'IDENTIFIER':
          // For simple parsing, just create a variable reference
          this.createVariableReference(token.value);
          i++;
          break;
          
        case 'LAMBDA':
          // Handle lambda expressions directly
          i = this.parseLambda(tokens, i);
          break;
          
        case 'LPAREN':
          i = this.parseParenthesized(tokens, i);
          break;
          
        default:
          i++;
      }
    }
  }
  
  buildExpressionTree(operators, operands) {
    if (operators.length === 0) return;
    
    // First, handle * and / (higher precedence)
    let i = 0;
    while (i < operators.length) {
      const op = operators[i];
      if (op === '*' || op === '/') {
        // Create binary operation for this operator
        const leftAgent = operands[i];
        const rightAgent = operands[i + 1];
        
        const opAgent = this.net.createAgent(AgentType.OP2, 2, this.getOperatorCode(op));
        this.net.connectPorts(opAgent.auxiliaryPorts[0], leftAgent.principalPort);
        this.net.connectPorts(opAgent.auxiliaryPorts[1], rightAgent.principalPort);
        
        // Replace the two operands with the operation result
        operands.splice(i, 2, opAgent);
        operators.splice(i, 1);
        
        // Don't increment i since we removed an operator
      } else {
        i++;
      }
    }
    
    // Then, handle + and - (lower precedence)
    i = 0;
    while (i < operators.length) {
      const op = operators[i];
      
      // Create binary operation for this operator
      const leftAgent = operands[i];
      const rightAgent = operands[i + 1];
      
      const opAgent = this.net.createAgent(AgentType.OP2, 2, this.getOperatorCode(op));
      this.net.connectPorts(opAgent.auxiliaryPorts[0], leftAgent.principalPort);
      this.net.connectPorts(opAgent.auxiliaryPorts[1], rightAgent.principalPort);
      
      // Replace the two operands with the operation result
      operands.splice(i, 2, opAgent);
      operators.splice(i, 1);
      
      // Don't increment i since we removed an operator
    }
  }

  parseLambda(tokens, startIndex) {
    let i = startIndex + 1; // Skip λ symbol
    
    if (i >= tokens.length || tokens[i].type !== 'IDENTIFIER') {
      return i;
    }
    
    const paramName = tokens[i].value;
    i++;
    
    // Skip '.' or '->'
    if (i < tokens.length && (tokens[i].value === '.' || tokens[i].value === '->')) {
      i++;
    }
    
    // Parse lambda body (until end of expression or parent)
    const bodyEnd = this.findLambdaBodyEnd(tokens, i);
    const bodyTokens = tokens.slice(i, bodyEnd);
    
    // Create lambda agent
    const lambdaAgent = this.net.createAgent(AgentType.LAM, 1, paramName);
    
    // Push to lambda stack for variable binding
    this.lambdaStack.push({
      lambda: lambdaAgent,
      paramName: paramName
    });
    
    // Parse body
    this.parseTokens(bodyTokens);
    
    // Pop from stack
    this.lambdaStack.pop();
    
    // Check if this lambda is being applied to arguments (like (λx.x) 3)
    if (bodyEnd < tokens.length && tokens[bodyEnd].type === 'INTEGER' ||
        (bodyEnd < tokens.length && tokens[bodyEnd].type === 'IDENTIFIER' &&
         tokens[bodyEnd].value !== 'if' && tokens[bodyEnd].value !== 'then' && tokens[bodyEnd].value !== 'else')) {
      // This is a lambda application, create APP agent
      const appAgent = this.net.createAgent(AgentType.APP, 1);
      
      // Connect LAM and APP principal ports to create active pair
      this.net.connectPorts(lambdaAgent.principalPort, appAgent.principalPort);
      
      // Parse the argument
      const argEnd = this.findArgumentEnd(tokens, bodyEnd);
      const argTokens = tokens.slice(bodyEnd, argEnd);
      
      // Parse argument
      this.parseTokens(argTokens);
      
      // Connect argument to APP's auxiliary port
      const agents = Array.from(this.net.agents.values());
      if (agents.length > 0) {
        const lastAgent = agents[agents.length - 1];
        this.net.connectPorts(appAgent.auxiliaryPorts[0], lastAgent.principalPort);
      }
      
      return argEnd;
    }
    
    return bodyEnd;
  }

  parseFunctionCall(tokens, startIndex) {
    let i = startIndex;
    
    const funcName = tokens[i].value;
    i += 2; // Skip function name and '('
    
    // Parse arguments
    const args = [];
    while (i < tokens.length && tokens[i].type !== 'RPAREN') {
      const argEnd = this.findArgumentEnd(tokens, i);
      const argTokens = tokens.slice(i, argEnd);
      
      // Parse argument
      this.parseTokens(argTokens);
      args.push(argTokens);
      
      i = argEnd;
      
      if (tokens[i] && tokens[i].type === 'COMMA') {
        i++;
      }
    }
    
    i++; // Skip ')'
    
    // Create application agent
    this.createApplication(funcName, args.length);
    
    return i;
  }

  parseLiteral(tokens, startIndex) {
    const token = tokens[startIndex];
    let value;
    
    if (token.type === 'INTEGER') {
      value = parseInt(token.value);
    } else if (token.type === 'FLOAT') {
      value = parseFloat(token.value);
    }
    
    // Create number agent
    this.net.createAgent(AgentType.NUM, 0, value);
    
    return startIndex + 1;
  }

  parseLambdaApplication(tokens, startIndex) {
    let i = startIndex + 1; // Skip '('
    
    // Find matching ')'
    let depth = 1;
    while (i < tokens.length && depth > 0) {
      if (tokens[i].type === '(') {
        depth++;
      } else if (tokens[i].type === ')') {
        depth--;
      }
      i++;
    }
    
    const parenEnd = i; // Position of the closing ')'
    
    // Parse the lambda inside parentheses
    const insideTokens = tokens.slice(startIndex + 1, parenEnd - 1);
    const beforeLambdaCount = this.net.agents.size;
    
    // Parse lambda expression
    this.parseTokens(insideTokens);
    const afterLambdaCount = this.net.agents.size;
    
    // Get the lambda agent (should be the first one created)
    const agents = Array.from(this.net.agents.values());
    const lambdaAgent = agents[beforeLambdaCount];
    
    // Check if there are arguments after the parentheses
    if (parenEnd < tokens.length) {
      // Create APP agent
      const appAgent = this.net.createAgent(AgentType.APP, 1);
      
      // Connect LAM and APP principal ports to create active pair
      lambdaAgent.principalPort.connect(appAgent.principalPort);
      
      // Parse arguments
      const argStart = parenEnd;
      const argEnd = this.findArgumentEnd(tokens, argStart);
      const argTokens = tokens.slice(argStart, argEnd);
      
      this.parseTokens(argTokens);
      
      // Connect arguments to APP's auxiliary ports
      const afterArgCount = this.net.agents.size;
      
      for (let j = 0; j < Math.min(1, afterArgCount - afterLambdaCount - 1); j++) {
        const argAgent = Array.from(this.net.agents.values())[afterLambdaCount + 1 + j];
        appAgent.auxiliaryPorts[j].connect(argAgent.principalPort);
      }
      
      return argEnd;
    }
    
    return parenEnd;
  }

  parseParenthesized(tokens, startIndex) {
    let i = startIndex + 1; // Skip '('
    
    // Find matching ')'
    let depth = 1;
    while (i < tokens.length && depth > 0) {
      if (tokens[i].type === '(') {
        depth++;
      } else if (tokens[i].type === ')') {
        depth--;
      }
      i++;
    }
    
    // Parse content inside parentheses
    const contentTokens = tokens.slice(startIndex + 1, i - 1);
    this.parseTokens(contentTokens);
    
    return i;
  }

  createVariableReference(varName) {
    // Check if variable is bound by a lambda
    const lambdaBinding = this.lambdaStack.find(binding => binding.paramName === varName);
    
    if (lambdaBinding) {
      // Create reference to lambda parameter
      const refAgent = this.net.createAgent(AgentType.CON, 0, varName);
      
      // Connect to lambda's auxiliary port
      this.net.connectPorts(
        refAgent.principalPort,
        lambdaBinding.lambda.auxiliaryPorts[0]
      );
      
      return refAgent;
    }
    
    // Check if we already have this variable
    if (this.variableMap.has(varName)) {
      return this.variableMap.get(varName);
    }
    
    // Create new variable agent
    const varAgent = this.net.createAgent(AgentType.CON, 0, varName);
    this.variableMap.set(varName, varAgent);
    
    return varAgent;
  }

  createBinaryOperation(operator) {
    const opAgent = this.net.createAgent(AgentType.OP2, 2, this.getOperatorCode(operator));
    
    // Connect to recent agents (simplified - real implementation would need proper tracking)
    const agents = Array.from(this.net.agents.values());
    if (agents.length >= 2) {
      const leftAgent = agents[agents.length - 2];
      const rightAgent = agents[agents.length - 1];
      
      // Connect operands to OP2's auxiliary ports for arithmetic reduction
      this.net.connectPorts(opAgent.auxiliaryPorts[0], leftAgent.principalPort);
      this.net.connectPorts(opAgent.auxiliaryPorts[1], rightAgent.principalPort);
    }
    
    return opAgent;
  }

  createApplication(funcName, arity) {
    const appAgent = this.net.createAgent(AgentType.APP, arity + 1);
    
    // Connect to function and arguments (simplified)
    const agents = Array.from(this.net.agents.values());
    const neededAgents = arity + 1;
    
    if (agents.length >= neededAgents) {
      // Create active pair with the function (first agent)
      const funcAgent = agents[agents.length - neededAgents];
      this.net.connectPorts(appAgent.principalPort, funcAgent.principalPort);
      
      // Connect arguments to auxiliary ports
      for (let i = 0; i < arity; i++) {
        const argAgent = agents[agents.length - neededAgents + 1 + i];
        this.net.connectPorts(appAgent.auxiliaryPorts[i], argAgent.principalPort);
      }
    }
    
    return appAgent;
  }

  createRootAgent() {
    const rootAgent = this.net.createAgent(AgentType.ROOT, 1);
    
    // Connect to the last created agent
    const agents = Array.from(this.net.agents.values());
    if (agents.length > 1) {
      const lastAgent = agents[agents.length - 1];
      this.net.connectPorts(rootAgent.auxiliaryPorts[0], lastAgent.principalPort);
    }
    
    return rootAgent;
  }

  getOperatorCode(operator) {
    const opCodes = {
      '+': 1,
      '-': 2,
      '*': 3,
      '/': 4,
      '==': 5,
      '!=': 6,
      '<': 7,
      '>': 8,
      '<=': 9,
      '>=': 10,
      'and': 11,
      'or': 12
    };
    
    return opCodes[operator] || 0;
  }

  isBinaryOperator(tokens, index) {
    if (index >= tokens.length) return false;
    
    const token = tokens[index];
    return ['+', '-', '*', '/', '==', '!=', '<', '>', '<=', '>=', 'and', 'or'].includes(token.value);
  }

  findOperandEnd(tokens, startIndex) {
    let i = startIndex;
    let depth = 0;
    let foundLambda = false;
    
    while (i < tokens.length) {
      const token = tokens[i];
      
      if (token.type === 'LPAREN') {
        depth++;
      } else if (token.type === 'RPAREN') {
        depth--;
        if (depth < 0) break;
      } else if (token.type === 'LAMBDA') {
        foundLambda = true;
      } else if (depth === 0 && !foundLambda && this.isBinaryOperator(tokens, i)) {
        break;
      } else if (depth === 0 && foundLambda && (token.type === 'IDENTIFIER' || token.type === 'INTEGER' || token.type === 'FLOAT')) {
        // For lambda expressions, the operand ends after the first identifier/literal after the lambda body
        i++;
        break;
      }
      
      i++;
    }
    
    return i;
  }

  findArgumentEnd(tokens, startIndex) {
    let i = startIndex;
    let depth = 0;
    
    while (i < tokens.length) {
      const token = tokens[i];
      
      if (token.type === 'LPAREN') {
        depth++;
      } else if (token.type === 'RPAREN') {
        depth--;
        if (depth < 0) break;
      } else if (depth === 0 && (token.type === 'COMMA' || token.type === 'RPAREN')) {
        break;
      }
      
      i++;
    }
    
    return i;
  }

  findMatchingEnd(tokens, startIndex) {
    let i = startIndex;
    let depth = 0;
    
    while (i < tokens.length) {
      const token = tokens[i];
      
      if (token.type === 'DO') {
        depth++;
      } else if (token.type === 'END') {
        depth--;
        if (depth === 0) {
          return i;
        }
      }
      
      i++;
    }
    
    return tokens.length;
  }
  
  findFunctionBodyEnd(tokens, startIndex) {
    let i = startIndex;
    
    // Find the end of the function body (next DEF or end of tokens)
    while (i < tokens.length) {
      if (tokens[i].type === 'DEF') {
        // Found next function definition
        return i;
      }
      i++;
    }
    
    return tokens.length;
  }

  findLambdaBodyEnd(tokens, startIndex) {
    let i = startIndex;
    let depth = 0;
    
    while (i < tokens.length) {
      const token = tokens[i];
      
      if (token.type === 'DO') {
        depth++;
      } else if (token.type === 'END') {
        depth--;
        if (depth < 0) {
          return i;
        }
      } else if (token.type === 'RPAREN') {
        if (depth === 0) {
          return i;
        }
      } else if (token.type === 'LAMBDA') {
        depth++;
      }
      
      i++;
    }
    
    return tokens.length;
  }
}

// Helper function to parse Zapp code directly
function parseZappToNet(sourceCode) {
  // Import lexer (would need to be adapted for ES6 modules)
  // For now, simplified tokenization
  const tokens = tokenizeZapp(sourceCode);
  const parser = new NetParser();
  return parser.parse(tokens);
}

// Simplified tokenizer for demonstration
function tokenizeZapp(sourceCode) {
  const tokens = [];
  let i = 0;
  
  while (i < sourceCode.length) {
    // Skip whitespace
    if (/\s/.test(sourceCode[i])) {
      i++;
      continue;
    }
    
    // Lambda symbol (λ)
    if (sourceCode[i] === 'λ') {
      tokens.push({ type: 'LAMBDA', value: 'λ' });
      i++;
      continue;
    }
    
    // Numbers
    if (/\d/.test(sourceCode[i])) {
      let num = '';
      while (i < sourceCode.length && /[\d.]/.test(sourceCode[i])) {
        num += sourceCode[i++];
      }
      tokens.push({
        type: num.includes('.') ? 'FLOAT' : 'INTEGER',
        value: num
      });
      continue;
    }
    
    // Identifiers and keywords
    if (/[a-zA-Z_]/.test(sourceCode[i])) {
      let ident = '';
      while (i < sourceCode.length && /[a-zA-Z0-9_]/.test(sourceCode[i])) {
        ident += sourceCode[i++];
      }
      
      // Check for keywords
      const keywords = ['def', 'do', 'end', 'if', 'then', 'else', 'and', 'or'];
      tokens.push({
        type: keywords.includes(ident) ? ident.toUpperCase() : 'IDENTIFIER',
        value: ident
      });
      continue;
    }
    
    // Operators and delimiters
    const char = sourceCode[i];
    if ('+-*/=<>().'.includes(char)) {
      // Check for two-character operators
      if (i + 1 < sourceCode.length) {
        const twoChar = sourceCode.substr(i, 2);
        if (['==', '!=', '<=', '>=', '->'].includes(twoChar)) {
          tokens.push({ type: twoChar, value: twoChar });
          i += 2;
          continue;
        }
      }
      
      // Single character operators and delimiters
      if (char === '=') {
        tokens.push({ type: '=', value: '=' });
      } else if (char === '.') {
        tokens.push({ type: '.', value: '.' });
      } else {
        tokens.push({ type: char, value: char });
      }
      i++;
      continue;
    }
    
    i++;
  }
  
  return tokens;
}

// Export for both Node.js and ES6 modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { NetParser, parseZappToNet, tokenizeZapp };
}

// ES6 module exports for browser
export { NetParser, parseZappToNet, tokenizeZapp };