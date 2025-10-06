/**
 * Zapp to Interaction Net Parser
 * Converts Zapp syntax into interaction net representation
 */

import { InteractionNet, Agent, AgentType } from './interaction_net.js';

function getAgentTypeName(agentType) {
  for (const [key, value] of Object.entries(AgentType)) {
    if (typeof value === 'number' && value === agentType) {
      return key;
    }
  }
  return 'UNKNOWN';
}

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

    // Create root agent if needed (but not for lambda applications)
    if (this.net.agents.size > 0 && !this.hasLambdaApplication(tokens)) {
      this.createRootAgent();
    }

    return this.net;
  }
  
  hasLambdaApplication(tokens) {
    // Check if the tokens contain a lambda application
    for (let i = 0; i < tokens.length; i++) {
      if (tokens[i].type === '(' && i + 1 < tokens.length && tokens[i + 1].type === 'LAMBDA') {
        // Check if there are arguments after the closing parenthesis
        let depth = 1;
        let j = i + 2;
        while (j < tokens.length && depth > 0) {
          if (tokens[j].type === '(') {
            depth++;
          } else if (tokens[j].type === ')') {
            depth--;
          }
          j++;
        }
        
        // If there are tokens after the closing parenthesis, it's a lambda application
        if (j < tokens.length) {
          return true;
        }
      }
    }
    return false;
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
    console.log(`=== parseExpressionWithPrecedence ===`);
    console.log(`Tokens:`, tokens.map(t => `${t.type}:${t.value}`).join(', '));
    console.log(`Start index: ${startIndex}`);
    
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
    console.log(`First operand tokens:`, firstOperandTokens.map(t => `${t.type}:${t.value}`).join(', '));
    
    // Store the current net state to track agents created by this operand
    const beforeOperandCount = this.net.agents.size;
    console.log(`Agent count before first operand: ${beforeOperandCount}`);
    
    // Check if this is a complex expression (like parentheses or lambda)
    if (firstOperandTokens.length === 1 &&
        (firstOperandTokens[0].type === 'INTEGER' || firstOperandTokens[0].type === 'FLOAT')) {
      // Simple literal - parse directly
      console.log(`Parsing simple literal: ${firstOperandTokens[0].value}`);
      this.parseLiteral(firstOperandTokens, 0);
    } else if (firstOperandTokens.length > 0) {
      // Complex expression - parse but avoid infinite recursion
      console.log(`Parsing complex expression`);
      this.parseTokensSimple(firstOperandTokens);
    }
    
    const afterOperandCount = this.net.agents.size;
    console.log(`Agent count after first operand: ${afterOperandCount}`);
    
    // Get the agents created by this operand
    const agents = Array.from(this.net.agents.values());
    const operandAgents = agents.slice(beforeOperandCount);
    console.log(`First operand agents:`, operandAgents.map(a => `${a.id}:${getAgentTypeName(a.type)}`));
    operands.push(...operandAgents);
    
    i = firstOperandEnd;
    
    // Parse remaining operators and operands
    while (i < tokens.length && this.isBinaryOperator(tokens, i)) {
      const opToken = tokens[i];
      console.log(`Found operator: ${opToken.value}`);
      i++;
      
      // Parse next operand
      const nextOperandEnd = this.findOperandEnd(tokens, i);
      const nextOperandTokens = tokens.slice(i, nextOperandEnd);
      console.log(`Next operand tokens:`, nextOperandTokens.map(t => `${t.type}:${t.value}`).join(', '));
      
      const beforeNextOperandCount = this.net.agents.size;
      console.log(`Agent count before next operand: ${beforeNextOperandCount}`);
      
      // Check if this is a complex expression
      if (nextOperandTokens.length === 1 &&
          (nextOperandTokens[0].type === 'INTEGER' || nextOperandTokens[0].type === 'FLOAT')) {
        // Simple literal - parse directly
        console.log(`Parsing simple literal: ${nextOperandTokens[0].value}`);
        this.parseLiteral(nextOperandTokens, 0);
      } else if (nextOperandTokens.length > 0) {
        // Complex expression - parse but avoid infinite recursion
        console.log(`Parsing complex expression`);
        this.parseTokensSimple(nextOperandTokens);
      }
      
      const afterNextOperandCount = this.net.agents.size;
      console.log(`Agent count after next operand: ${afterNextOperandCount}`);
      
      const nextAgents = Array.from(this.net.agents.values());
      const nextOperandAgents = nextAgents.slice(beforeNextOperandCount);
      console.log(`Next operand agents:`, nextOperandAgents.map(a => `${a.id}:${getAgentTypeName(a.type)}`));
      operands.push(...nextOperandAgents);
      
      operators.push(opToken.value);
      i = nextOperandEnd;
    }
    
    console.log(`Operators:`, operators);
    console.log(`Total operands:`, operands.map(a => `${a.id}:${getAgentTypeName(a.type)}`));
    
    // Now build the expression tree with precedence
    this.buildExpressionTree(operators, operands);
    
    console.log(`=== Finished parseExpressionWithPrecedence ===`);
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
    console.log(`=== buildExpressionTree ===`);
    console.log(`Operators:`, operators);
    console.log(`Operands:`, operands.map(a => `${a.id}:${getAgentTypeName(a.type)}${a.data !== null ? '(' + a.data + ')' : ''}`));
    
    if (operators.length === 0) {
      console.log('No operators to process');
      return;
    }
    
    // First, handle * and / (higher precedence)
    let i = 0;
    while (i < operators.length) {
      const op = operators[i];
      if (op === '*' || op === '/') {
        console.log(`Processing high-precedence operator: ${op}`);
        // Create binary operation for this operator
        const leftAgent = operands[i];
        const rightAgent = operands[i + 1];
        
        console.log(`Left operand: ${leftAgent.id} (${getAgentTypeName(leftAgent.type)}), Right operand: ${rightAgent.id} (${getAgentTypeName(rightAgent.type)})`);
        
        const opAgent = this.net.createAgent(AgentType.OP2, 2, this.getOperatorCode(op));
        console.log(`Created OP2 agent ${opAgent.id} for operator ${op}`);
        
        this.net.connectPorts(opAgent.auxiliaryPorts[0], leftAgent.principalPort);
        this.net.connectPorts(opAgent.auxiliaryPorts[1], rightAgent.principalPort);
        console.log(`Connected operands to OP2 agent`);
        
        // Replace the two operands with the operation result
        operands.splice(i, 2, opAgent);
        operators.splice(i, 1);
        
        console.log(`After processing ${op}: operands=${operands.length}, operators=${operators.length}`);
        
        // Don't increment i since we removed an operator
      } else {
        i++;
      }
    }
    
    // Then, handle + and - (lower precedence)
    i = 0;
    while (i < operators.length) {
      const op = operators[i];
      console.log(`Processing low-precedence operator: ${op}`);
      
      // Create binary operation for this operator
      const leftAgent = operands[i];
      const rightAgent = operands[i + 1];
      
      console.log(`Left operand: ${leftAgent.id} (${getAgentTypeName(leftAgent.type)}), Right operand: ${rightAgent.id} (${getAgentTypeName(rightAgent.type)})`);
      
      const opAgent = this.net.createAgent(AgentType.OP2, 2, this.getOperatorCode(op));
      console.log(`Created OP2 agent ${opAgent.id} for operator ${op}`);
      
      this.net.connectPorts(opAgent.auxiliaryPorts[0], leftAgent.principalPort);
      this.net.connectPorts(opAgent.auxiliaryPorts[1], rightAgent.principalPort);
      console.log(`Connected operands to OP2 agent`);
      
      // Replace the two operands with the operation result
      operands.splice(i, 2, opAgent);
      operators.splice(i, 1);
      
      console.log(`After processing ${op}: operands=${operands.length}, operators=${operators.length}`);
      
      // Don't increment i since we removed an operator
    }
    
    console.log(`=== Finished buildExpressionTree ===`);
    console.log(`Final operands:`, operands.map(a => `${a.id}:${getAgentTypeName(a.type)}${a.data !== null ? '(' + a.data + ')' : ''}`));
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
    const beforeBodyCount = this.net.agents.size;
    this.parseTokens(bodyTokens);
    const newAgents = Array.from(this.net.agents.values()).slice(beforeBodyCount);
    
    // Find the root agent of the body (the one with no incoming connections from other body agents)
    let bodyRoot = null;
    for (const agent of newAgents) {
      let hasIncoming = false;
      for (const other of newAgents) {
        if (other !== agent) {
          other.getAllPorts().forEach(port => {
            if (port.isConnected() && port.getConnectedAgent() === agent) {
              hasIncoming = true;
            }
          });
        }
      }
      if (!hasIncoming) {
        bodyRoot = agent;
        break;
      }
    }
    
    // If no root found, use the last agent
    if (!bodyRoot && newAgents.length > 0) {
      bodyRoot = newAgents[newAgents.length - 1];
    }
    
    // Connect the body root to the lambda's auxiliary port
    if (bodyRoot) {
      this.net.connectPorts(lambdaAgent.auxiliaryPorts[0], bodyRoot.principalPort);
    }
    
    // Pop from stack
    this.lambdaStack.pop();
    
    // Check if this lambda is being applied to arguments (like (λx.x) 3)
    if (bodyEnd < tokens.length &&
        (tokens[bodyEnd].type === 'INTEGER' || tokens[bodyEnd].type === 'FLOAT' ||
         (tokens[bodyEnd].type === 'IDENTIFIER' &&
          tokens[bodyEnd].value !== 'if' && tokens[bodyEnd].value !== 'then' && tokens[bodyEnd].value !== 'else') ||
         tokens[bodyEnd].type === '(')) {
      
      // This is a lambda application, create APP agent
      const appAgent = this.net.createAgent(AgentType.APP, 1);
      
      // Connect LAM and APP principal ports to create active pair
      this.net.connectPorts(lambdaAgent.principalPort, appAgent.principalPort);
      
      // Parse all arguments
      let argStart = bodyEnd;
      let argCount = 0;
      
      while (argStart < tokens.length &&
             ![')', ';', '\n'].includes(tokens[argStart].value) &&
             tokens[argStart].type !== 'RPAREN') {
        
        const argEnd = this.findArgumentEnd(tokens, argStart);
        const argTokens = tokens.slice(argStart, argEnd);
        
        // Parse argument
        this.parseTokens(argTokens);
        
        // Connect argument to APP's auxiliary port
        const agents = Array.from(this.net.agents.values());
        if (agents.length > 0 && argCount < appAgent.auxiliaryPorts.length) {
          const lastAgent = agents[agents.length - 1];
          this.net.connectPorts(appAgent.auxiliaryPorts[argCount], lastAgent.principalPort);
        }
        
        argCount++;
        argStart = argEnd;
        
        // Skip commas between arguments
        if (argStart < tokens.length && tokens[argStart].type === 'COMMA') {
          argStart++;
        }
      }
      
      return argStart;
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
    console.log(`=== parseLambdaApplication starting at index ${startIndex} ===`);
    console.log(`Tokens:`, tokens.slice(startIndex, Math.min(startIndex + 10, tokens.length)).map(t => `${t.type}:${t.value}`).join(', '));
    
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
    console.log(`Found closing parenthesis at index ${parenEnd}`);
    
    // Parse the lambda inside parentheses
    const insideTokens = tokens.slice(startIndex + 1, parenEnd - 1);
    console.log(`Inside tokens:`, insideTokens.map(t => `${t.type}:${t.value}`).join(', '));
    const beforeLambdaCount = this.net.agents.size;
    console.log(`Agent count before parsing lambda: ${beforeLambdaCount}`);
    
    // Parse lambda expression
    this.parseTokens(insideTokens);
    const afterLambdaCount = this.net.agents.size;
    console.log(`Agent count after parsing lambda: ${afterLambdaCount}`);
    
    // Get the lambda agent - it should be the first LAM agent created
    const agents = Array.from(this.net.agents.values());
    const newAgents = agents.slice(beforeLambdaCount);
    const lambdaAgent = newAgents.find(agent => agent.type === AgentType.LAM);
    console.log(`Lambda agent: ${lambdaAgent ? lambdaAgent.id : 'not found'} (${lambdaAgent ? getAgentTypeName(lambdaAgent.type) : 'unknown'})`);
    
    // Check if there are arguments after the parentheses
    if (parenEnd < tokens.length) {
      // Parse all arguments
      const args = [];
      let argStart = parenEnd;
      
      // Keep parsing arguments until we run out or hit a delimiter
      while (argStart < tokens.length &&
             ![')', ';', '\n'].includes(tokens[argStart].value) &&
             tokens[argStart].type !== 'RPAREN') {
        
        const argEnd = this.findArgumentEnd(tokens, argStart);
        const argTokens = tokens.slice(argStart, argEnd);
        
        console.log(`Found argument: ${argTokens.map(t => t.value).join(' ')}`);
        
        // Store argument tokens for later parsing
        args.push(argTokens);
        argStart = argEnd;
        
        // Skip whitespace between arguments
        while (argStart < tokens.length && /\s/.test(tokens[argStart].value)) {
          argStart++;
        }
        
        // Skip commas between arguments
        if (argStart < tokens.length && tokens[argStart].type === 'COMMA') {
          argStart++;
        }
      }
      
      if (args.length > 0) {
        console.log(`=== Processing ${args.length} arguments ===`);
        console.log(`Arguments:`, args.map(arg => arg.map(t => `${t.type}:${t.value}`).join(' ')));
        
        // For multiple arguments, we need to create a chain of applications
        // This implements currying: ((λx.λy.body) arg1) arg2
        
        // Parse all arguments first
        const argAgents = [];
        for (const argTokens of args) {
          const beforeArgCount = this.net.agents.size;
          console.log(`\n--- Parsing argument: ${argTokens.map(t => t.value).join(' ')} ---`);
          
          // Check if this is a parenthesized expression
          const isParenthesized = argTokens.length > 0 && argTokens[0].type === '(';
          
          if (isParenthesized) {
            // This is a parenthesized expression, we need to parse it as a unit
            // Remove the outer parentheses for parsing
            let newAgents;
            const innerTokens = argTokens.slice(1, argTokens.length - 1); // Exclude closing )
            console.log('Parsing parenthesized argument:', innerTokens.map(t => t.value).join(' '));
            console.log('Is lambda branch?', innerTokens.length > 0 && innerTokens[0].type === 'LAMBDA');
            
            // Check if this is a lambda expression
            if (innerTokens.length > 0 && innerTokens[0].type === 'LAMBDA') {
              console.log('Entering lambda branch');
              // Parse the lambda expression
              this.parseTokens(innerTokens);
              
              // Get the lambda agent created
              newAgents = Array.from(this.net.agents.values()).slice(beforeArgCount);
              console.log('After parsing lambda, newAgents length:', newAgents.length);
              console.log('New agents:', newAgents.map(a => `${a.id}:${getAgentTypeName(a.type)}${a.data ? '(' + a.data + ')' : ''}`).join(', '));
              const lamAgent = newAgents.find(agent => agent.type === AgentType.LAM);
              
              if (lamAgent) {
                console.log('Found lambda agent:', lamAgent.id);
                argAgents.push(lamAgent);
              } else {
                console.log('No lambda agent found in newAgents');
              }
            } else {
              console.log('Entering non-lambda branch (arithmetic expression)');
              // This is an arithmetic expression or other complex expression
              // Parse it with precedence
              console.log('Before parsing arithmetic, agent count:', this.net.agents.size);
              this.parseExpressionWithPrecedence(innerTokens, 0);
              console.log('After parsing arithmetic, agent count:', this.net.agents.size);
              
              // Get all agents created by this argument
              newAgents = Array.from(this.net.agents.values()).slice(beforeArgCount);
              console.log('After parsing non-lambda, newAgents length:', newAgents.length);
              console.log('newAgents details:', newAgents.map(a => `${a.id}: ${getAgentTypeName(a.type)} ${a.data ? '(' + a.data + ')' : ''}`).join(', '));
              
              // For arithmetic expressions, we need to find the root agent
              // The root agent should be the one that represents the entire expression
              let rootAgent = null;
              
              // First, try to find an OP2 agent (binary operation)
              for (const agent of newAgents) {
                if (agent.type === AgentType.OP2) {
                  rootAgent = agent;
                  console.log('Found OP2 root:', rootAgent.id);
                  break;
                }
              }
              
              // If no OP2 agent found, look for a LAM agent
              if (!rootAgent) {
                for (const agent of newAgents) {
                  if (agent.type === AgentType.LAM) {
                    rootAgent = agent;
                    console.log('Found LAM root:', rootAgent.id);
                    break;
                  }
                }
              }
              
              // If still no root agent found, use the last agent created
              if (!rootAgent && newAgents.length > 0) {
                rootAgent = newAgents[newAgents.length - 1];
                console.log('Using last agent as root:', rootAgent.id, getAgentTypeName(rootAgent.type));
              }
              
              if (rootAgent) {
                console.log('Pushing rootAgent to argAgents:', rootAgent.id, getAgentTypeName(rootAgent.type));
                argAgents.push(rootAgent);
              } else {
                console.log('No root agent found for this argument');
              }
            }
            console.log('After parsing argument, new agents:', newAgents ? newAgents.map(a => `${a.id}:${a.type}`).join(', ') : 'undefined');
          } else {
            // Simple argument, parse normally
            this.parseTokens(argTokens);
            
            // Get the argument agent created
            const newAgents = Array.from(this.net.agents.values()).slice(beforeArgCount);
            if (newAgents.length > 0) {
              argAgents.push(newAgents[newAgents.length - 1]);
            }
          }
        }
        
        console.log(`\n=== Argument parsing complete ===`);
        console.log(`Argument agents:`, argAgents.map(a => `${a.id}(${getAgentTypeName(a.type)})`));
        
        if (argAgents.length === 1) {
          // Single argument - simple case
          console.log(`Creating single APP agent for argument ${argAgents[0].id}`);
          const appAgent = this.net.createAgent(AgentType.APP, 1);
          this.net.connectPorts(lambdaAgent.principalPort, appAgent.principalPort);
          this.net.connectPorts(appAgent.auxiliaryPorts[0], argAgents[0].principalPort);
        } else {
          // Multiple arguments - create currying chain
          console.log(`Creating currying chain for ${argAgents.length} arguments`);
          // The correct structure for ((λx.λy.body) arg1) arg2 is:
          // LAM0 - APP1 (active pair for first reduction)
          // APP1 has arg1
          // After reduction: new LAM (from body) - APP2 (active pair for second reduction)
          // APP2 has arg2
          
          // Create all APP agents first
          const appAgents = [];
          for (let i = 0; i < argAgents.length; i++) {
            const appAgent = this.net.createAgent(AgentType.APP, 1);
            appAgents.push(appAgent);
            console.log(`Created APP agent ${appAgent.id} for argument ${i}`);
          }
          
          // Connect LAM to first APP (this creates the active pair for first reduction)
          console.log(`Connecting lambda ${lambdaAgent.id} to first APP ${appAgents[0].id}`);
          this.net.connectPorts(lambdaAgent.principalPort, appAgents[0].principalPort);
          
          // Connect arguments to their respective APP agents
          for (let i = 0; i < argAgents.length; i++) {
            console.log(`Connecting argument ${argAgents[i].id} to APP ${appAgents[i].id}`);
            this.net.connectPorts(appAgents[i].auxiliaryPorts[0], argAgents[i].principalPort);
          }
          
          // For currying chain, we need to set up the structure for subsequent reductions
          // The key insight is that we DON'T connect APP agents directly
          // Instead, we rely on the reduction process to create the proper connections
          // After the first reduction (LAM-APP), the result will be a new LAM agent
          // This new LAM agent should then be connected to the next APP agent
          
          // To achieve this, we need to modify the reduction process to handle currying
          // For now, let's create a special marker that indicates this is a currying chain
          if (appAgents.length > 1) {
            // Create a chain of APP agents for currying
            // Mark each APP agent with information about the next one
            for (let i = 0; i < appAgents.length - 1; i++) {
              appAgents[i].data = {
                isCurryingChain: true,
                nextAppId: appAgents[i + 1].id,
                chainLength: appAgents.length - i
              };
              console.log(`Set currying data for APP ${appAgents[i].id}: next=${appAgents[i + 1].id}, length=${appAgents.length - i}`);
            }
          }
        }
        
        return argStart;
      }
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
    
    // Connect to the last created agent that's not the ROOT itself
    const agents = Array.from(this.net.agents.values());
    const nonRootAgents = agents.filter(agent => agent.type !== AgentType.ROOT);
    
    if (nonRootAgents.length > 0) {
      const lastAgent = nonRootAgents[nonRootAgents.length - 1];
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
      
      if (token.type === '(' || token.type === 'LPAREN') {
        depth++;
        i++;
      } else if (token.type === ')' || token.type === 'RPAREN') {
        depth--;
        i++;
        if (depth < 0) break;
        // If we've closed all parentheses and we're back at depth 0, this argument ends here
        if (depth === 0) break;
      } else if (depth === 0 && (token.type === 'COMMA' || token.type === 'RPAREN')) {
        break;
      } else if (depth === 0) {
        // At depth 0, check if this is a simple token that should be its own argument
        if (token.type === 'INTEGER' || token.type === 'FLOAT' ||
            (token.type === 'IDENTIFIER' && token.value !== 'if' &&
             token.value !== 'then' && token.value !== 'else')) {
          i++;
          break;
        } else {
          i++;
        }
      } else {
        // Inside parentheses, just advance
        i++;
      }
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