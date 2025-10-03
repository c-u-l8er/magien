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
          i = this.parseLiteral(tokens, i);
          break;
          
        case 'LPAREN':
          i = this.parseParenthesized(tokens, i);
          break;
          
        default:
          i++;
      }
    }
  }

  parseFunction(tokens, startIndex) {
    let i = startIndex + 1; // Skip DEF
    
    if (i >= tokens.length || tokens[i].type !== 'IDENTIFIER') {
      return i;
    }
    
    const functionName = tokens[i].value;
    i++;
    
    // Skip parameters for now (simplified)
    while (i < tokens.length && tokens[i].type !== 'DO') {
      i++;
    }
    
    if (i < tokens.length && tokens[i].type === 'DO') {
      i++;
      
      // Parse function body
      const bodyEnd = this.findMatchingEnd(tokens, i);
      const bodyTokens = tokens.slice(i, bodyEnd);
      
      // Parse body as expression
      this.parseTokens(bodyTokens);
      
      i = bodyEnd + 1;
    }
    
    return i;
  }

  parseExpression(tokens, startIndex) {
    let i = startIndex;
    
    if (i >= tokens.length) return i;
    
    const token = tokens[i];
    
    // Handle binary operations
    if (this.isBinaryOperator(tokens, i)) {
      return this.parseBinaryOperation(tokens, i);
    }
    
    // Handle function calls
    if (i + 1 < tokens.length && tokens[i + 1].type === 'LPAREN') {
      return this.parseFunctionCall(tokens, i);
    }
    
    // Handle identifiers
    if (token.type === 'IDENTIFIER') {
      const varName = token.value;
      
      // Check if this is a lambda abstraction
      if (i + 1 < tokens.length && tokens[i + 1].value === '->') {
        return this.parseLambda(tokens, i);
      }
      
      // Create variable reference
      this.createVariableReference(varName);
      i++;
    }
    
    return i;
  }

  parseBinaryOperation(tokens, startIndex) {
    let i = startIndex;
    
    // Parse left operand
    const leftEnd = this.findOperandEnd(tokens, i);
    const leftTokens = tokens.slice(i, leftEnd);
    this.parseTokens(leftTokens);
    
    i = leftEnd;
    
    if (i >= tokens.length) return i;
    
    const opToken = tokens[i];
    i++;
    
    // Parse right operand
    const rightEnd = this.findOperandEnd(tokens, i);
    const rightTokens = tokens.slice(i, rightEnd);
    this.parseTokens(rightTokens);
    
    i = rightEnd;
    
    // Create binary operation agent
    this.createBinaryOperation(opToken.value);
    
    return i;
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

  parseParenthesized(tokens, startIndex) {
    let i = startIndex + 1; // Skip '('
    
    // Find matching ')'
    let depth = 1;
    while (i < tokens.length && depth > 0) {
      if (tokens[i].type === 'LPAREN') {
        depth++;
      } else if (tokens[i].type === 'RPAREN') {
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
      
      // Create active pairs by connecting principal ports
      this.net.connectPorts(opAgent.principalPort, leftAgent.principalPort);
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
    
    while (i < tokens.length) {
      const token = tokens[i];
      
      if (token.type === 'LPAREN') {
        depth++;
      } else if (token.type === 'RPAREN') {
        depth--;
        if (depth < 0) break;
      } else if (depth === 0 && this.isBinaryOperator(tokens, i)) {
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
    if ('+-*/=<>()'.includes(char)) {
      // Check for two-character operators
      if (i + 1 < sourceCode.length) {
        const twoChar = sourceCode.substr(i, 2);
        if (['==', '!=', '<=', '>=', '->'].includes(twoChar)) {
          tokens.push({ type: twoChar, value: twoChar });
          i += 2;
          continue;
        }
      }
      
      tokens.push({ type: char, value: char });
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