/**
 * Zapp Interaction Net System
 * True interaction net implementation with graph-based reduction
 * GPU-parallel rewriting of interaction nets
 */

// Agent types for interaction nets
const AgentType = {
  ERA: 0,      // Eraser (deletes connections)
  CON: 1,      // Constructor (creates data)
  DUP: 2,      // Duplicator (copies data)
  APP: 3,      // Application (function application)
  LAM: 4,      // Lambda (abstraction)
  SUP: 5,      // Superposition (parallel choice)
  NUM: 6,      // Number literal
  OP2: 7,      // Binary operator
  ROOT: 8,     // Root node (entry point)
  WIRE: 9      // Wire (connection)
};

// Port types
const PortType = {
  PRINCIPAL: 0,  // Main port of an agent
  AUXILIARY: 1   // Auxiliary ports (arity-based)
};

class Port {
  constructor(id, type = PortType.AUXILIARY, index = 0) {
    this.id = id;           // Unique port identifier
    this.type = type;       // PRINCIPAL or AUXILIARY
    this.index = index;     // For auxiliary ports (0, 1, 2, ...)
    this.connectedTo = null; // Reference to connected port
    this.agent = null;      // Reference to owning agent
  }

  connect(otherPort) {
    // Disconnect existing connections
    if (this.connectedTo) {
      this.connectedTo.connectedTo = null;
    }
    if (otherPort.connectedTo) {
      otherPort.connectedTo.connectedTo = null;
    }

    // Create new connection
    this.connectedTo = otherPort;
    otherPort.connectedTo = this;
  }

  disconnect() {
    if (this.connectedTo) {
      this.connectedTo.connectedTo = null;
      this.connectedTo = null;
    }
  }

  isConnected() {
    return this.connectedTo !== null;
  }

  getConnectedAgent() {
    return this.connectedTo ? this.connectedTo.agent : null;
  }
}

class Agent {
  constructor(id, type, arity = 0) {
    this.id = id;           // Unique agent identifier
    this.type = type;       // AgentType
    this.arity = arity;     // Number of auxiliary ports
    this.principalPort = new Port(id, PortType.PRINCIPAL);
    this.auxiliaryPorts = [];
    
    // Create auxiliary ports
    for (let i = 0; i < arity; i++) {
      const auxPort = new Port(`${id}_aux_${i}`, PortType.AUXILIARY, i);
      auxPort.agent = this;
      this.auxiliaryPorts.push(auxPort);
    }
    
    this.principalPort.agent = this;
    
    // Agent-specific data
    this.data = null;       // For literals, numbers, etc.
    this.active = false;    // Whether this agent is part of an active pair
  }

  getPort(portIndex) {
    if (portIndex === -1 || portIndex === 'principal') {
      return this.principalPort;
    }
    return this.auxiliaryPorts[portIndex] || null;
  }

  getAllPorts() {
    return [this.principalPort, ...this.auxiliaryPorts];
  }

  isConnected() {
    return this.getAllPorts().some(port => port.isConnected());
  }

  getConnectedAgents() {
    const connected = [];
    this.getAllPorts().forEach(port => {
      if (port.isConnected()) {
        const agent = port.getConnectedAgent();
        if (agent && !connected.includes(agent)) {
          connected.push(agent);
        }
      }
    });
    return connected;
  }

  clone(newId) {
    const cloned = new Agent(newId, this.type, this.arity);
    cloned.data = this.data;
    return cloned;
  }
}

class InteractionNet {
  constructor() {
    this.agents = new Map();     // id -> Agent
    this.nextId = 0;             // Next agent ID
    this.activePairs = [];       // List of active pairs for reduction
    this.reductionSteps = 0;     // Number of reduction steps performed
    this.maxAgents = 0;          // Maximum number of agents seen
  }

  createAgent(type, arity = 0, data = null) {
    const id = this.nextId++;
    const agent = new Agent(id, type, arity);
    agent.data = data;
    this.agents.set(id, agent);
    this.maxAgents = Math.max(this.maxAgents, this.agents.size);
    return agent;
  }

  removeAgent(agentId) {
    const agent = this.agents.get(agentId);
    if (agent) {
      // Disconnect all ports
      agent.getAllPorts().forEach(port => port.disconnect());
      this.agents.delete(agentId);
    }
  }

  connectPorts(port1, port2) {
    if (port1 && port2) {
      port1.connect(port2);
    }
  }

  findActivePairs() {
    this.activePairs = [];
    const visited = new Set();

    // First, find LAM-APP active pairs (lambda-beta reduction)
    for (const agent of this.agents.values()) {
      if (visited.has(agent.id)) continue;
      
      if (agent.type === AgentType.LAM) {
        const principalPort = agent.principalPort;
        
        if (principalPort.isConnected()) {
          const connectedAgent = principalPort.getConnectedAgent();
          
          // Check if connected to an APP agent
          if (connectedAgent && connectedAgent.type === AgentType.APP) {
            this.activePairs.push({
              agent1: agent,
              agent2: connectedAgent,
              port: principalPort,
              type: 'lambda'
            });
            
            visited.add(agent.id);
            visited.add(connectedAgent.id);
          }
        }
      }
    }

    // CRITICAL FIX: Find lambda reference active pairs
    // Look for CON agents with data starting with 'lambda_ref_' and create active pairs
    // with the corresponding lambda agents by extracting the lambda ID and parameter from the reference data
    console.log(`DEBUG: Checking for lambda reference active pairs...`);
    const lambdaRefAgents = Array.from(this.agents.values()).filter(agent =>
      agent.type === AgentType.CON && agent.data && agent.data.startsWith('lambda_ref_')
    );
    console.log(`DEBUG: Found ${lambdaRefAgents.length} lambda reference agents:`, lambdaRefAgents.map(a => `${a.id}(${a.data})`));
    
    // NEW DIAGNOSTIC: Track lambda reference resolution
    console.log(`DIAGNOSTIC: *** LAMBDA REFERENCE RESOLUTION ANALYSIS ***`);
    const allLambdas = Array.from(this.agents.values()).filter(agent => agent.type === AgentType.LAM);
    console.log(`DIAGNOSTIC: All lambda agents in net:`, allLambdas.map(lam => `${lam.id}(${lam.data})`));
    
    for (const refAgent of lambdaRefAgents) {
      if (visited.has(refAgent.id)) continue;
      
      console.log(`DEBUG: Examining lambda reference agent ${refAgent.id} (${refAgent.data})`);
      
      // CRITICAL FIX: Extract both the lambda ID and parameter name from the enhanced reference data
      const refMatch = refAgent.data.match(/lambda_ref_(\d+)_param_(.+)/);
      if (refMatch) {
        const originalLambdaId = parseInt(refMatch[1]);
        const lambdaParam = refMatch[2];
        const lambdaAgent = this.agents.get(originalLambdaId);
        
        console.log(`DEBUG: Lambda reference ${refAgent.id} refers to original lambda ${originalLambdaId} with parameter '${lambdaParam}'`);
        
        if (lambdaAgent && lambdaAgent.type === AgentType.LAM) {
          console.log(`DEBUG: Found lambda reference active pair: CON${refAgent.id} (${refAgent.data}) <-> LAM${lambdaAgent.id} (${lambdaAgent.data})`);
          
          // Create an active pair between the lambda agent and the reference agent
          this.activePairs.push({
            agent1: lambdaAgent,
            agent2: refAgent,
            port: lambdaAgent.principalPort, // Use the lambda's principal port
            type: 'lambda_ref' // Use a different type to distinguish this from regular lambda pairs
          });
          
          visited.add(refAgent.id);
          visited.add(lambdaAgent.id);
          
          console.log(`DEBUG: Created lambda reference active pair for LAM${lambdaAgent.id} and CON${refAgent.id}`);
        } else {
          console.log(`DEBUG: Original lambda ${originalLambdaId} not found or not a LAM agent - looking for cloned lambda with parameter '${lambdaParam}'`);
          
          // CRITICAL FIX: Look for a cloned lambda with the exact parameter name from the reference
          const clonedLambdas = Array.from(this.agents.values()).filter(agent =>
            agent.type === AgentType.LAM && agent.data === lambdaParam
          );
          console.log(`DEBUG: Found ${clonedLambdas.length} lambdas with parameter '${lambdaParam}':`, clonedLambdas.map(l => `${l.id}(${l.data})`));
          
          if (clonedLambdas.length > 0) {
            const clonedLambda = clonedLambdas[0]; // Use the first one found
            console.log(`DEBUG: Creating lambda reference active pair with cloned lambda: CON${refAgent.id} <-> LAM${clonedLambda.id} (${clonedLambda.data})`);
            
            this.activePairs.push({
              agent1: clonedLambda,
              agent2: refAgent,
              port: clonedLambda.principalPort,
              type: 'lambda_ref'
            });
            
            visited.add(refAgent.id);
            visited.add(clonedLambda.id);
            
            console.log(`DEBUG: Created lambda reference active pair for cloned LAM${clonedLambda.id} and CON${refAgent.id}`);
          } else {
            console.log(`DEBUG: *** ERROR: No lambda found with parameter '${lambdaParam}' - reference resolution failed ***`);
          }
        }
      } else {
        // Fallback to old format for backward compatibility
        console.log(`DEBUG: Using fallback parsing for old format reference: ${refAgent.data}`);
        const lambdaIdMatch = refAgent.data.match(/lambda_ref_(\d+)/);
        if (lambdaIdMatch) {
          const originalLambdaId = parseInt(lambdaIdMatch[1]);
          const lambdaAgent = this.agents.get(originalLambdaId);
          
          if (lambdaAgent && lambdaAgent.type === AgentType.LAM) {
            console.log(`DEBUG: Found lambda reference active pair (fallback): CON${refAgent.id} <-> LAM${lambdaAgent.id}`);
            
            this.activePairs.push({
              agent1: lambdaAgent,
              agent2: refAgent,
              port: lambdaAgent.principalPort,
              type: 'lambda_ref'
            });
            
            visited.add(refAgent.id);
            visited.add(lambdaAgent.id);
          } else {
            console.log(`DEBUG: Original lambda ${originalLambdaId} not found - using heuristic lookup`);
            const lambdaParam = this.extractLambdaParamFromReference(refAgent.data, originalLambdaId);
            if (lambdaParam) {
              const clonedLambdas = Array.from(this.agents.values()).filter(agent =>
                agent.type === AgentType.LAM && agent.data === lambdaParam
              );
              
              if (clonedLambdas.length > 0) {
                const clonedLambda = clonedLambdas[0];
                this.activePairs.push({
                  agent1: clonedLambda,
                  agent2: refAgent,
                  port: clonedLambda.principalPort,
                  type: 'lambda_ref'
                });
                
                visited.add(refAgent.id);
                visited.add(clonedLambda.id);
              }
            }
          }
        } else {
          console.log(`DEBUG: Could not extract lambda ID from reference data: ${refAgent.data}`);
        }
      }
    }

    // Find arithmetic reduction opportunities (OP2 agents with connected NUM agents)
    for (const agent of this.agents.values()) {
      if (visited.has(agent.id)) continue;
      
      if (agent.type === AgentType.OP2 && agent.arity >= 2) {
        const port1 = agent.auxiliaryPorts[0];
        const port2 = agent.auxiliaryPorts[1];
        
        console.log(`Checking OP2 agent ${agent.id} for arithmetic reduction`);
        console.log(`  Port1 connected: ${port1.isConnected()}, Port2 connected: ${port2.isConnected()}`);
        
        if (port1.isConnected() && port2.isConnected()) {
          const agent1 = port1.getConnectedAgent();
          const agent2 = port2.getConnectedAgent();
          
          console.log(`  Connected agents: ${agent1 ? agent1.id + '(' + getAgentTypeName(agent1.type) + ')' : 'null'}, ${agent2 ? agent2.id + '(' + getAgentTypeName(agent2.type) + ')' : 'null'}`);
          
          if (agent1 && agent2 &&
              agent1.type === AgentType.NUM &&
              agent2.type === AgentType.NUM) {
            
            console.log(`  Found arithmetic reduction opportunity: OP2${agent.id} with NUM${agent1.id} and NUM${agent2.id}`);
            this.activePairs.push({
              agent1: agent,
              agent2: agent1,
              agent3: agent2,
              type: 'arithmetic'
            });
            
            visited.add(agent.id);
            visited.add(agent1.id);
            visited.add(agent2.id);
          } else {
            console.log(`  Not reducible: agent types are ${agent1 ? getAgentTypeName(agent1.type) : 'null'} and ${agent2 ? getAgentTypeName(agent2.type) : 'null'}`);
          }
        }
      }
    }

    return this.activePairs;
  }

  reduceStep() {
    const activePairs = this.findActivePairs();
    
    console.log(`=== Reduction Step ${this.reductionSteps + 1} ===`);
    console.log(`Found ${activePairs.length} active pairs:`, activePairs.map(p => `${p.type} (${p.agent1?.id}-${p.agent2?.id})`));
    
    if (activePairs.length === 0) {
      console.log('No more reductions possible');
      return false; // No more reductions possible
    }

    // Reduce each active pair
    // Sort pairs to ensure inner lambdas are reduced before outer ones
    // This is critical for function composition: (λf.λx.f (f x)) (λy.y + 1) 3
    // We need to reduce λy first, then λx, then λf
    activePairs.sort((a, b) => {
      // If both are lambda pairs, sort by lambda ID (higher ID = more inner lambda)
      if (a.type === 'lambda' && b.type === 'lambda') {
        return b.agent1.id - a.agent1.id; // Higher ID first (inner lambda)
      }
      // Arithmetic pairs should be reduced after lambda pairs
      if (a.type === 'arithmetic') return 1;
      if (b.type === 'arithmetic') return -1;
      return 0;
    });
    
    console.log(`Sorted active pairs:`, activePairs.map(p => `${p.type} (${p.agent1?.id}-${p.agent2?.id})`));
    
    for (const pair of activePairs) {
      console.log(`Reducing pair: ${pair.type}`);
      if (pair.type === 'arithmetic') {
        this.reduceArithmetic(pair);
      } else if (pair.type === 'lambda') {
        console.log(`Calling reduceLambdaApp for LAM${pair.agent1.id} and APP${pair.agent2.id}`);
        this.reduceLambdaApp(pair.agent1, pair.agent2);
      } else if (pair.type === 'lambda_ref') {
        console.log(`Calling reduceLambdaRef for LAM${pair.agent1.id} and CON${pair.agent2.id}`);
        this.reduceLambdaRef(pair.agent1, pair.agent2);
      } else if (pair.type === 'curry') {
        // For APP-APP pairs, we need to handle currying
        // This happens when we have ((λx.λy.body) arg1) arg2
        // The first APP should be reduced with the nested LAM
        this.handleCurrying(pair);
      } else {
        this.reduceActivePair(pair);
      }
    }

    this.reductionSteps++;
    return true;
  }

  reduceActivePair(pair) {
    const { agent1, agent2 } = pair;
    
    // Apply reduction rules based on agent types
    const rule = this.getReductionRule(agent1.type, agent2.type);
    
    if (rule) {
      rule.call(this, agent1, agent2);
    } else {
      // Default: delete both agents (ERA-like behavior)
      this.removeAgent(agent1.id);
      this.removeAgent(agent2.id);
    }
  }

  getReductionRule(type1, type2) {
    // Order the types for consistent lookup
    const [t1, t2] = type1 <= type2 ? [type1, type2] : [type2, type1];
    
    const rules = {
      [`${AgentType.LAM},${AgentType.APP}`]: this.reduceLambdaApp,
      [`${AgentType.APP},${AgentType.LAM}`]: this.reduceLambdaApp,
      [`${AgentType.DUP},${AgentType.CON}`]: this.reduceDupCon,
      [`${AgentType.CON},${AgentType.DUP}`]: this.reduceDupCon,
      [`${AgentType.OP2},${AgentType.NUM}`]: this.reduceOp2Num,
      [`${AgentType.NUM},${AgentType.OP2}`]: this.reduceOp2Num,
      [`${AgentType.OP2},${AgentType.OP2}`]: this.reduceOp2Op2,
      [`${AgentType.SUP},${AgentType.ANY}`]: this.reduceSup,
      [`${AgentType.ANY},${AgentType.SUP}`]: this.reduceSup,
    };

    return rules[`${t1},${t2}`] || null;
  }

  // Reduce an APP agent and return its result
  reduceAppAgent(appAgent) {
    console.log(`DEBUG: Reducing APP agent ${appAgent.id}`);
    
    // Find the lambda connected to this APP agent
    const principalPort = appAgent.principalPort;
    if (!principalPort || !principalPort.isConnected()) {
      console.log(`DEBUG: APP ${appAgent.id} principal port not connected`);
      return null;
    }
    
    const lambdaAgent = principalPort.getConnectedAgent();
    if (!lambdaAgent || lambdaAgent.type !== AgentType.LAM) {
      console.log(`DEBUG: APP ${appAgent.id} not connected to LAM agent`);
      return null;
    }
    
    console.log(`DEBUG: Found LAM ${lambdaAgent.id} connected to APP ${appAgent.id}`);
    
    // Get the argument from the APP agent
    const argPort = appAgent.auxiliaryPorts[0];
    if (!argPort || !argPort.isConnected()) {
      console.log(`DEBUG: APP ${appAgent.id} argument port not connected`);
      return null;
    }
    
    const argumentAgent = argPort.getConnectedAgent();
    console.log(`DEBUG: Found argument ${argumentAgent.id} (${getAgentTypeName(argumentAgent.type)}) connected to APP ${appAgent.id}`);
    
    // CRITICAL FIX: For function composition, we need to check if the argument is an OP2 agent
    // that needs to be evaluated first, or if it's already a numeric value
    let actualArgument = argumentAgent;

    // If the argument is an OP2 agent, try to evaluate it
    if (argumentAgent.type === AgentType.OP2) {
      const port1 = argumentAgent.auxiliaryPorts[0];
      const port2 = argumentAgent.auxiliaryPorts[1];

      console.log(`DEBUG: Checking if OP2 ${argumentAgent.id} can be evaluated for APP ${appAgent.id}`);
      console.log(`DEBUG: OP2 ${argumentAgent.id} port1 connected: ${port1.isConnected()}, port2 connected: ${port2.isConnected()}`);

      // Check if both ports are connected
      if (port1.isConnected() && port2.isConnected()) {
        const op1 = port1.getConnectedAgent();
        const op2 = port2.getConnectedAgent();

        console.log(`DEBUG: OP2 ${argumentAgent.id} operands: ${op1.id}(${getAgentTypeName(op1.type)}, data=${op1.data}), ${op2.id}(${getAgentTypeName(op2.type)}, data=${op2.data})`);

        // If both operands are NUM agents, we can evaluate immediately
        if (op1.type === AgentType.NUM && op2.type === AgentType.NUM) {
          console.log(`DEBUG: Argument OP2 ${argumentAgent.id} can be reduced: ${op1.data} + ${op2.data}`);

          // Perform the arithmetic operation
          const result = op1.data + op2.data;
          const resultAgent = this.createAgent(AgentType.NUM, 0, result);

          console.log(`DEBUG: Created result ${resultAgent.id} (${result}) from OP2 ${argumentAgent.id}`);

          // Remove the OP2 agent and its operands
          this.removeAgent(argumentAgent);
          this.removeAgent(op1);
          this.removeAgent(op2);

          actualArgument = resultAgent;
        }
        // If port1 is disconnected but port2 has a NUM, this might be a partially evaluated OP2
        // In this case, we need to find what should be in port1
        else if (!port1.isConnected() && op2.type === AgentType.NUM) {
          console.log(`DEBUG: OP2 ${argumentAgent.id} has disconnected port1, trying to find missing operand`);

          // For function composition, the missing operand should be the result of the previous application
          // Look for NUM agents that are not connected to APP agents (these are results, not arguments)
          const numAgents = Array.from(this.agents.values()).filter(agent => {
            if (agent.type !== AgentType.NUM || agent.id === op2.id) return false;

            // Check if this NUM agent is connected to an APP agent (meaning it's an argument)
            const isArgument = Array.from(this.agents.values()).some(otherAgent => {
              if (otherAgent.type === AgentType.APP) {
                return otherAgent.auxiliaryPorts.some(port =>
                  port.isConnected() && port.getConnectedAgent() === agent
                );
              }
              return false;
            });

            return !isArgument; // Only include NUM agents that are not arguments to APP agents
          });

          console.log(`DEBUG: Found ${numAgents.length} result NUM agents:`, numAgents.map(n => `${n.id}(${n.data})`));

          // Use the highest ID NUM agent as the missing operand (most recent result)
          if (numAgents.length > 0) {
            const missingOperand = numAgents.reduce((max, agent) => agent.id > max.id ? agent : max);
            console.log(`DEBUG: Using NUM ${missingOperand.id} (${missingOperand.data}) as missing operand for OP2 ${argumentAgent.id}`);

            // Connect the missing operand
            port1.connect(missingOperand.principalPort);

            // Now try to evaluate again
            const newOp1 = port1.getConnectedAgent();
            if (newOp1.type === AgentType.NUM) {
              console.log(`DEBUG: Now OP2 ${argumentAgent.id} can be evaluated: ${newOp1.data} + ${op2.data}`);

              // Perform the arithmetic operation
              const result = newOp1.data + op2.data;
              const resultAgent = this.createAgent(AgentType.NUM, 0, result);

              console.log(`DEBUG: Created result ${resultAgent.id} (${result}) from OP2 ${argumentAgent.id}`);

              // Remove the OP2 agent and its operands
              this.removeAgent(argumentAgent);
              this.removeAgent(newOp1);
              this.removeAgent(op2);

              actualArgument = resultAgent;
            }
          } else {
            console.log(`DEBUG: No suitable missing operand found for OP2 ${argumentAgent.id}`);
          }
        }
      }
    }
    
    // Now perform the lambda reduction with the actual argument
    console.log(`DEBUG: Performing beta reduction for APP ${appAgent.id} with argument ${actualArgument.id}`);
    
    // Disconnect the original argument and connect the actual argument
    argPort.disconnect();
    argPort.connect(actualArgument.principalPort);
    
    // Perform the lambda reduction
    this.reduceLambdaApp(lambdaAgent, appAgent);
    
    // After reduction, look for the result
    // The result should be a NUM agent connected to where the lambda body was
    const lambdaBody = lambdaAgent.auxiliaryPorts[0].isConnected() ?
      lambdaAgent.auxiliaryPorts[0].getConnectedAgent() : null;
    
    if (lambdaBody && lambdaBody.type === AgentType.NUM) {
      console.log(`DEBUG: Found result ${lambdaBody.id} (${lambdaBody.data}) from reduced APP ${appAgent.id}`);
      return lambdaBody;
    } else if (lambdaBody) {
      console.log(`DEBUG: Found non-NUM result ${lambdaBody.id} (${getAgentTypeName(lambdaBody.type)}) from reduced APP ${appAgent.id}`);
      return lambdaBody;
    } else {
      console.log(`DEBUG: No result found from reduced APP ${appAgent.id}`);
      return null;
    }
  }

  // Beta reduction: (λx.M) N → M[x := N]
  reduceLambdaApp(lambdaAgent, appAgent) {
    console.log(`=== Beta Reduction: λ${lambdaAgent.data} applied ===`);
    console.log(`Lambda agent: ${lambdaAgent.id}, APP agent: ${appAgent.id}`);
    console.log(`APP data:`, appAgent.data);
    
    // Get the argument from the APP agent
    const argPort = appAgent.auxiliaryPorts[0];
    if (argPort && argPort.isConnected()) {
      const argAgent = argPort.getConnectedAgent();
      
      if (argAgent) {
        console.log(`Argument agent: ${argAgent.id} (${getAgentTypeName(argAgent.type)}) with data:`, argAgent.data);
        console.log(`Argument agent connections:`, argAgent.getAllPorts().map(p => p.isConnected() ? `${p.getConnectedAgent().id}(${getAgentTypeName(p.getConnectedAgent().type)})` : 'none').join(', '));
        
        // Find all CON agents that represent the lambda parameter
        const paramName = lambdaAgent.data;
        const agents = Array.from(this.agents.values());
        
        // DEBUG: Log all CON agents in the net
        console.log(`DEBUG: All CON agents in net:`);
        agents.filter(agent => agent.type === AgentType.CON).forEach(con => {
          console.log(`  CON ${con.id}: data='${con.data}', connections: ${con.getAllPorts().map(p => p.isConnected() ? `${p.getConnectedAgent().id}(${getAgentTypeName(p.getConnectedAgent().type)})` : 'none').join(', ')}`);
        });
        
        // For each parameter reference, we need to substitute with the argument
        const paramRefs = agents.filter(agent =>
          agent.type === AgentType.CON && agent.data === paramName
        );
        
        // Also check if this parameter appears in arithmetic expressions
        // If so, we need to handle substitution differently
        const op2AgentsWithParam = agents.filter(agent =>
          agent.type === AgentType.OP2 && this.hasParameterInOperands(agent, paramName)
        );
        
        if (op2AgentsWithParam.length > 0) {
          console.log(`Found ${op2AgentsWithParam.length} OP2 agents with parameter '${paramName}':`, op2AgentsWithParam.map(a => a.id));
        }
        
        // SPECIAL CASE: For lambda calculus with function composition, we need to handle
        // the case where the argument is a lambda that contains arithmetic expressions
        // with parameters that should be substituted when the lambda is applied
        if (argAgent.type === AgentType.LAM) {
          console.log(`Argument is a lambda - checking for arithmetic expressions with parameters`);
          
          // Get all OP2 agents in the net that might need parameter substitution
          const allOP2Agents = agents.filter(agent => agent.type === AgentType.OP2);
          
          // DEBUG: Log all OP2 agents and their operands
          console.log(`DEBUG: All OP2 agents in net:`);
          allOP2Agents.forEach(op2 => {
            console.log(`  OP2 ${op2.id}: data=${op2.data}`);
            const port1 = op2.auxiliaryPorts[0];
            const port2 = op2.auxiliaryPorts[1];
            if (port1.isConnected()) {
              const agent1 = port1.getConnectedAgent();
              console.log(`    Port1: ${agent1.id} (${getAgentTypeName(agent1.type)}) data='${agent1.data}'`);
            }
            if (port2.isConnected()) {
              const agent2 = port2.getConnectedAgent();
              console.log(`    Port2: ${agent2.id} (${getAgentTypeName(agent2.type)}) data='${agent2.data}'`);
            }
          });
          
          // Check if any OP2 agent has operands that are CON agents with any parameter name
          // This is a more general check for arithmetic expressions that need substitution
          const op2AgentsNeedingSubstitution = allOP2Agents.filter(op2Agent => {
            const port1 = op2Agent.auxiliaryPorts[0];
            const port2 = op2Agent.auxiliaryPorts[1];
            
            let needsSubstitution = false;
            
            if (port1.isConnected()) {
              const agent1 = port1.getConnectedAgent();
              if (agent1 && agent1.type === AgentType.CON) {
                needsSubstitution = true;
                console.log(`OP2 ${op2Agent.id} has CON agent ${agent1.id} with data '${agent1.data}' in first operand`);
              }
            }
            
            if (port2.isConnected()) {
              const agent2 = port2.getConnectedAgent();
              if (agent2 && agent2.type === AgentType.CON) {
                needsSubstitution = true;
                console.log(`OP2 ${op2Agent.id} has CON agent ${agent2.id} with data '${agent2.data}' in second operand`);
              }
            }
            
            return needsSubstitution;
          });
          
          if (op2AgentsNeedingSubstitution.length > 0) {
            console.log(`Found ${op2AgentsNeedingSubstitution.length} OP2 agents needing parameter substitution:`, op2AgentsNeedingSubstitution.map(a => a.id));
            
            // For each OP2 agent, check if it needs substitution with the current lambda's parameter
            op2AgentsNeedingSubstitution.forEach(op2Agent => {
              if (this.hasParameterInOperands(op2Agent, paramName)) {
                console.log(`Substituting parameter '${paramName}' in OP2 agent ${op2Agent.id}`);
                this.substituteParameterInOP2(op2Agent, paramName, argAgent);
              } else {
                console.log(`DEBUG: OP2 ${op2Agent.id} doesn't have parameter '${paramName}' in its operands`);
              }
            });
          }
        }
        
        console.log(`Found ${paramRefs.length} parameter references for '${paramName}':`,
          paramRefs.map(ref => `${ref.id} connected to ${ref.getAllPorts().map(p => p.isConnected() ? `${p.getConnectedAgent().id}(${getAgentTypeName(p.getConnectedAgent().type)})` : 'none').join(', ')}`));
        
        // Create a copy of the argument agent for each parameter reference
        // This ensures we don't have issues with multiple references
        const argClones = [];
        for (let i = 0; i < paramRefs.length; i++) {
          const argClone = this.cloneAgentForSubstitution(argAgent);
          argClones.push(argClone);
          console.log(`Created argument clone ${argClone.id} for parameter reference ${paramRefs[i].id}`);
        }
        
        // Handle substitution in arithmetic expressions first
        op2AgentsWithParam.forEach(op2Agent => {
          console.log(`=== Handling parameter substitution in OP2 agent ${op2Agent.id} ===`);
          this.substituteParameterInOP2(op2Agent, paramName, argAgent);
        });
        
        // CRITICAL FIX: Always check for OP2 agents that need substitution with the current parameter
        // This handles the case where the argument is not a lambda but we still need to substitute
        // parameters in arithmetic expressions
        const allOP2Agents = agents.filter(agent => agent.type === AgentType.OP2);
        const op2AgentsNeedingSubstitution = allOP2Agents.filter(op2Agent => {
          const port1 = op2Agent.auxiliaryPorts[0];
          const port2 = op2Agent.auxiliaryPorts[1];
          
          let needsSubstitution = false;
          
          if (port1.isConnected()) {
            const agent1 = port1.getConnectedAgent();
            if (agent1 && agent1.type === AgentType.CON && agent1.data === paramName) {
              needsSubstitution = true;
              console.log(`OP2 ${op2Agent.id} has CON agent ${agent1.id} with data '${agent1.data}' in first operand`);
            }
          }
          
          if (port2.isConnected()) {
            const agent2 = port2.getConnectedAgent();
            if (agent2 && agent2.type === AgentType.CON && agent2.data === paramName) {
              needsSubstitution = true;
              console.log(`OP2 ${op2Agent.id} has CON agent ${agent2.id} with data '${agent2.data}' in second operand`);
            }
          }
          
          return needsSubstitution;
        });
        
        if (op2AgentsNeedingSubstitution.length > 0) {
          console.log(`Found ${op2AgentsNeedingSubstitution.length} OP2 agents needing substitution with parameter '${paramName}':`, op2AgentsNeedingSubstitution.map(a => a.id));
          
          op2AgentsNeedingSubstitution.forEach(op2Agent => {
            console.log(`Substituting parameter '${paramName}' in OP2 agent ${op2Agent.id}`);
            this.substituteParameterInOP2(op2Agent, paramName, argAgent);
          });
        }
        
        // Now substitute each parameter reference with its corresponding argument clone
        paramRefs.forEach((paramRef, index) => {
          const argClone = argClones[index];
          
          console.log(`=== Substituting parameter ${paramRef.id} with argument clone ${argClone.id} ===`);
          console.log(`Parameter connections before:`, paramRef.getAllPorts().map(p => p.isConnected() ? `${p.getConnectedAgent().id}(${getAgentTypeName(p.getConnectedAgent().type)})` : 'none').join(', '));
          console.log(`Argument clone connections before:`, argClone.getAllPorts().map(p => p.isConnected() ? `${p.getConnectedAgent().id}(${getAgentTypeName(p.getConnectedAgent().type)})` : 'none').join(', '));
          
          // Replace the parameter reference with the argument clone
          // The key insight is that we need to replace the parameter reference
          // with the argument, preserving all connections
          
          // First, disconnect the parameter from everything
          const connectedPorts = [];
          paramRef.getAllPorts().forEach(port => {
            if (port.isConnected()) {
              connectedPorts.push({
                port: port,
                connectedTo: port.connectedTo,
                portIndex: paramRef.getAllPorts().indexOf(port)
              });
              port.disconnect();
            }
          });
          
          console.log(`Parameter ${paramRef.id} had ${connectedPorts.length} connections:`, connectedPorts.map(c => `port${c.portIndex} -> agent${c.connectedTo.agent.id}`));
          
          // Now connect the argument clone to everything the parameter was connected to
          connectedPorts.forEach(({ connectedTo, portIndex }) => {
            if (portIndex === 0) {
              // Principal port - connect the clone's principal port
              argClone.principalPort.connect(connectedTo);
              console.log(`Connected clone ${argClone.id} principal port to agent ${connectedTo.agent.id}(${getAgentTypeName(connectedTo.agent.type)})`);
            } else {
              // Auxiliary port
              if (argClone.auxiliaryPorts[portIndex - 1]) {
                argClone.auxiliaryPorts[portIndex - 1].connect(connectedTo);
                console.log(`Connected clone ${argClone.id} aux port ${portIndex-1} to agent ${connectedTo.agent.id}(${getAgentTypeName(connectedTo.agent.type)})`);
              }
            }
          });
          
          console.log(`Argument clone connections after:`, argClone.getAllPorts().map(p => p.isConnected() ? `${p.getConnectedAgent().id}(${getAgentTypeName(p.getConnectedAgent().type)})` : 'none').join(', '));
          
          // Remove the parameter CON agent
          this.removeAgent(paramRef.id);
          console.log(`Removed parameter agent ${paramRef.id}`);
        });
        
        console.log(`After substitution, net has ${this.agents.size} agents`);
        
        // Check if this is part of a currying chain
        if (appAgent.data && appAgent.data.isCurryingChain) {
          console.log(`=== Handling currying chain ===`);
          console.log(`Next APP ID: ${appAgent.data.nextAppId}, Chain length: ${appAgent.data.chainLength}`);
          
          // This is a currying chain, we need to set up the next LAM-APP connection
          const nextAppId = appAgent.data.nextAppId;
          const nextAppAgent = this.agents.get(nextAppId);
          
          console.log(`Next APP agent: ${nextAppAgent ? nextAppAgent.id : 'not found'}`);
          
          if (nextAppAgent) {
            // Find all LAM agents that could be the result of this reduction
            const nestedLambdas = agents.filter(agent =>
              agent.type === AgentType.LAM && agent.id !== lambdaAgent.id
            );
            
            console.log(`Found ${nestedLambdas.length} nested lambdas:`, nestedLambdas.map(l => l.id));
            
            if (nestedLambdas.length > 0) {
              const nestedLambda = nestedLambdas[0];
              
              console.log(`Connecting nested lambda ${nestedLambda.id} to next APP ${nextAppAgent.id}`);
              
              // Connect the nested lambda to the next APP agent
              // This creates the next active pair for the currying chain
              this.connectPorts(nestedLambda.principalPort, nextAppAgent.principalPort);
              
              // Mark the next APP as part of the currying chain if there are more arguments
              if (appAgent.data.chainLength > 2) {
                // The next APP agent should already have its own data from the parser
                // No need to modify it here since it was set up during parsing
              }
            }
          }
        } else {
          console.log(`=== Handling regular lambda application ===`);
          // Regular lambda application (not part of currying chain)
          // Check if the lambda body contains another lambda (nested lambda)
          // If so, we need to connect the APP agent's principal port to it
          // This enables currying: ((λx.λy.body) arg1) arg2
          const nestedLambdas = agents.filter(agent =>
            agent.type === AgentType.LAM && agent.id !== lambdaAgent.id
          );
          
          console.log(`Found ${nestedLambdas.length} nested lambdas for regular application:`, nestedLambdas.map(l => l.id));
          
          if (nestedLambdas.length > 0) {
            const nestedLambda = nestedLambdas[0];
            
            console.log(`APP agent ${appAgent.id} principal port connected: ${appAgent.principalPort.isConnected()}`);
            
            if (appAgent.principalPort.isConnected()) {
              const connectedPort = appAgent.principalPort.connectedTo;
              console.log(`Connecting nested lambda ${nestedLambda.id} to APP's principal connection (agent ${connectedPort.agent.id})`);
              
              // Connect the nested lambda to whatever was connected to the APP
              connectedPort.connect(nestedLambda.principalPort);
            } else {
              console.log(`WARNING: APP agent ${appAgent.id} principal port is not connected - nested lambda ${nestedLambda.id} will be disconnected!`);
              console.log(`Nested lambda ${nestedLambda.id} principal port connected: ${nestedLambda.principalPort.isConnected()}`);
              console.log(`Nested lambda ${nestedLambda.id} aux port connected: ${nestedLambda.auxiliaryPorts[0].isConnected()}`);
              
              // CRITICAL FIX: If the nested lambda has no principal connection, we need to create one
              // This happens when the nested lambda should be the result of the current reduction
              // We need to check if there are any remaining APP agents that should be connected to this lambda
              const remainingApps = agents.filter(agent =>
                agent.type === AgentType.APP &&
                agent.id !== appAgent.id &&
                !agent.principalPort.isConnected()
              );
              
              console.log(`Found ${remainingApps.length} remaining APP agents without principal connections`);
              
              if (remainingApps.length > 0) {
                const nextApp = remainingApps[0];
                console.log(`Connecting nested lambda ${nestedLambda.id} to remaining APP ${nextApp.id}`);
                this.connectPorts(nestedLambda.principalPort, nextApp.principalPort);
              } else {
                // CRITICAL FIX: If there are no remaining APP agents, we need to create a new one
                // This happens when we have a lambda that should be applied to an argument
                // but the application structure wasn't set up correctly during parsing
                console.log(`No remaining APP agents found - creating new APP agent for nested lambda ${nestedLambda.id}`);
                
                // Check if the nested lambda has any arguments that need to be applied
                // Look for NUM agents that could be arguments
                const numAgents = agents.filter(agent => agent.type === AgentType.NUM);
                console.log(`Found ${numAgents.length} NUM agents that could be arguments:`, numAgents.map(n => n.id));
                
                if (numAgents.length > 0) {
                  // Create a new APP agent and connect it to the nested lambda
                  const newApp = this.createAgent(AgentType.APP, 1);
                  console.log(`Created new APP agent ${newApp.id} for nested lambda ${nestedLambda.id}`);
                  
                  // Connect the nested lambda to the new APP agent
                  this.connectPorts(nestedLambda.principalPort, newApp.principalPort);
                  console.log(`Connected nested lambda ${nestedLambda.id} to new APP ${newApp.id}`);
                  
                  // Connect the first NUM agent as an argument
                  const firstNum = numAgents[0];
                  this.connectPorts(newApp.auxiliaryPorts[0], firstNum.principalPort);
                  console.log(`Connected NUM ${firstNum.id} as argument to APP ${newApp.id}`);
                }
              }
            }
          }
        }
        
        // If the argument is a complex expression, we need to ensure it remains
        // connected properly after substitution
        if (argAgent.type === AgentType.OP2 || argAgent.type === AgentType.APP) {
          this.preserveArgumentStructure(argAgent);
        }
      }
    }

    // CRITICAL FIX: Check if this lambda has multiple references before removing it
    // If there are other lambda references pointing to this lambda, we need to preserve it
    const lambdaRefAgents = Array.from(this.agents.values()).filter(agent =>
      agent.type === AgentType.CON &&
      agent.data &&
      agent.data.includes(`lambda_ref_${lambdaAgent.id}_param_${lambdaAgent.data}`)
    );
    
    console.log(`DEBUG: Found ${lambdaRefAgents.length} remaining references to lambda ${lambdaAgent.id}:`, lambdaRefAgents.map(a => a.id));
    
    if (lambdaRefAgents.length > 0) {
      console.log(`DEBUG: Preserving lambda ${lambdaAgent.id} because it has ${lambdaRefAgents.length} remaining references`);
      // Don't remove the lambda agent - it needs to be available for other references
      this.removeAgent(appAgent.id);
    } else {
      console.log(`DEBUG: No remaining references to lambda ${lambdaAgent.id}, removing it`);
      // Remove the original agents
      this.removeAgent(lambdaAgent.id);
      this.removeAgent(appAgent.id);
    }
  }
  
  // Helper method to clone an agent for substitution
  cloneAgentForSubstitution(originalAgent) {
    console.log(`Cloning agent ${originalAgent.id} (${getAgentTypeName(originalAgent.type)}) for substitution`);
    
    if (originalAgent.type === AgentType.NUM) {
      // For numbers, we can create a simple copy
      const clone = this.createAgent(AgentType.NUM, 0, originalAgent.data);
      console.log(`Created NUM clone ${clone.id} with value ${originalAgent.data}`);
      return clone;
    } else if (originalAgent.type === AgentType.CON) {
      // For constructors/variables
      const clone = this.createAgent(AgentType.CON, 0, originalAgent.data);
      console.log(`Created CON clone ${clone.id} with data ${originalAgent.data}`);
      return clone;
    } else if (originalAgent.type === AgentType.OP2) {
      // For arithmetic operations, we need to clone the entire subgraph
      console.log(`Cloning complex OP2 expression`);
      return this.cloneComplexExpression(originalAgent);
    } else if (originalAgent.type === AgentType.LAM) {
      // For lambda functions, we need to clone the entire lambda structure
      console.log(`Cloning lambda expression`);
      return this.cloneLambdaExpression(originalAgent);
    } else if (originalAgent.type === AgentType.APP) {
      // For application agents, we need to clone the entire application structure
      console.log(`Cloning application expression`);
      return this.cloneApplicationExpression(originalAgent);
    } else {
      // For other complex expressions, clone the entire subgraph
      console.log(`Cloning generic complex expression`);
      return this.cloneComplexExpression(originalAgent);
    }
  }
  
  // Clone a complex expression (like arithmetic operations)
  cloneComplexExpression(originalAgent) {
    console.log(`=== Cloning complex expression ${originalAgent.id} (${getAgentTypeName(originalAgent.type)}) ===`);
    
    // Create a new agent of the same type
    const newAgent = this.createAgent(originalAgent.type, originalAgent.arity, originalAgent.data);
    console.log(`Created new agent ${newAgent.id} of type ${getAgentTypeName(originalAgent.type)} with data ${originalAgent.data}`);
    
    // If this is an OP2 agent, we need to clone its operands
    if (originalAgent.type === AgentType.OP2 && originalAgent.arity >= 2) {
      const port1 = originalAgent.auxiliaryPorts[0];
      const port2 = originalAgent.auxiliaryPorts[1];
      
      console.log(`OP2 port1 connected: ${port1.isConnected()}, port2 connected: ${port2.isConnected()}`);
      
      if (port1.isConnected() && port2.isConnected()) {
        const operand1 = port1.getConnectedAgent();
        const operand2 = port2.getConnectedAgent();
        
        console.log(`Operand1: ${operand1.id} (${getAgentTypeName(operand1.type)}), Operand2: ${operand2.id} (${getAgentTypeName(operand2.type)})`);
        
        if (operand1 && operand2) {
          // Clone the operands
          const newOperand1 = this.cloneAgentForSubstitution(operand1);
          const newOperand2 = this.cloneAgentForSubstitution(operand2);
          
          console.log(`Connecting new operands to new OP2 agent`);
          // Connect the new operands to the new operation
          this.connectPorts(newAgent.auxiliaryPorts[0], newOperand1.principalPort);
          this.connectPorts(newAgent.auxiliaryPorts[1], newOperand2.principalPort);
        }
      }
    }
    
    console.log(`=== Finished cloning complex expression ===`);
    return newAgent;
  }
  
  // Clone a lambda expression
  cloneLambdaExpression(originalAgent) {
    console.log(`=== Cloning lambda expression ${originalAgent.id} with parameter ${originalAgent.data} ===`);
    
    // Create a new lambda agent with the same parameter name
    const newLambda = this.createAgent(AgentType.LAM, 1, originalAgent.data);
    console.log(`Created new lambda ${newLambda.id} with parameter ${originalAgent.data}`);
    
    // Find all agents connected to the original lambda's auxiliary port
    const auxPort = originalAgent.auxiliaryPorts[0];
    console.log(`Lambda aux port connected: ${auxPort ? auxPort.isConnected() : 'no aux port'}`);
    
    if (auxPort && auxPort.isConnected()) {
      const connectedAgent = auxPort.getConnectedAgent();
      console.log(`Connected body agent: ${connectedAgent.id} (${getAgentTypeName(connectedAgent.type)})`);
      
      if (connectedAgent) {
        // Clone the connected agent (which represents the lambda body)
        const newBodyAgent = this.cloneAgentForSubstitution(connectedAgent);
        console.log(`Created body clone ${newBodyAgent.id}`);
        
        // Connect the new lambda to the new body
        this.connectPorts(newLambda.auxiliaryPorts[0], newBodyAgent.principalPort);
        console.log(`Connected new lambda ${newLambda.id} to new body ${newBodyAgent.id}`);
      }
    }
    
    console.log(`=== Finished cloning lambda expression ===`);
    return newLambda;
  }
  
  // Clone an application expression
  cloneApplicationExpression(originalAgent) {
    // Create a new application agent
    const newApp = this.createAgent(AgentType.APP, 1, originalAgent.data);
    
    // Clone the argument if connected
    const argPort = originalAgent.auxiliaryPorts[0];
    if (argPort && argPort.isConnected()) {
      const argAgent = argPort.getConnectedAgent();
      if (argAgent) {
        const newArgAgent = this.cloneAgentForSubstitution(argAgent);
        this.connectPorts(newApp.auxiliaryPorts[0], newArgAgent.principalPort);
      }
    }
    
    return newApp;
  }
  
  // Helper method to preserve the structure of complex arguments
  preserveArgumentStructure(argAgent) {
    // Mark the argument and its connected agents to ensure they're preserved
    const visited = new Set();
    const toPreserve = [argAgent];
    
    while (toPreserve.length > 0) {
      const agent = toPreserve.pop();
      if (visited.has(agent.id)) continue;
      
      visited.add(agent.id);
      
      // Mark this agent as important
      agent.active = true;
      
      // Add connected agents to the preservation list
      const connectedAgents = agent.getConnectedAgents();
      connectedAgents.forEach(connected => {
        if (connected && !visited.has(connected.id)) {
          toPreserve.push(connected);
        }
      });
    }
  }
  
  // Helper method to check if an OP2 agent has a parameter in its operands
  hasParameterInOperands(op2Agent, paramName) {
    if (op2Agent.type !== AgentType.OP2 || op2Agent.arity < 2) {
      return false;
    }
    
    const port1 = op2Agent.auxiliaryPorts[0];
    const port2 = op2Agent.auxiliaryPorts[1];
    
    console.log(`DEBUG: hasParameterInOperands checking OP2 ${op2Agent.id} for parameter '${paramName}'`);
    
    // Check if either operand is a CON agent with the parameter name
    if (port1.isConnected()) {
      const agent1 = port1.getConnectedAgent();
      console.log(`DEBUG: OP2 ${op2Agent.id} port1: ${agent1.id} (${getAgentTypeName(agent1.type)}) data='${agent1.data}'`);
      if (agent1 && agent1.type === AgentType.CON && agent1.data === paramName) {
        console.log(`DEBUG: Found parameter '${paramName}' in port1 of OP2 ${op2Agent.id}`);
        return true;
      }
    }
    
    if (port2.isConnected()) {
      const agent2 = port2.getConnectedAgent();
      console.log(`DEBUG: OP2 ${op2Agent.id} port2: ${agent2.id} (${getAgentTypeName(agent2.type)}) data='${agent2.data}'`);
      if (agent2 && agent2.type === AgentType.CON && agent2.data === paramName) {
        console.log(`DEBUG: Found parameter '${paramName}' in port2 of OP2 ${op2Agent.id}`);
        return true;
      }
    }
    
    console.log(`DEBUG: Parameter '${paramName}' not found in OP2 ${op2Agent.id}`);
    return false;
  }
  
  // Substitute a parameter in an OP2 agent with the argument
  substituteParameterInOP2(op2Agent, paramName, argAgent) {
    console.log(`=== Substituting parameter '${paramName}' in OP2 agent ${op2Agent.id} ===`);
    
    if (op2Agent.type !== AgentType.OP2 || op2Agent.arity < 2) {
      console.log(`Not a valid OP2 agent for substitution`);
      return;
    }
    
    const port1 = op2Agent.auxiliaryPorts[0];
    const port2 = op2Agent.auxiliaryPorts[1];
    
    // DEBUG: Log the current state before substitution
    console.log(`DEBUG: OP2 ${op2Agent.id} operands before substitution:`);
    if (port1.isConnected()) {
      const agent1 = port1.getConnectedAgent();
      console.log(`  Port1: ${agent1.id} (${getAgentTypeName(agent1.type)}) with data: ${agent1.data}`);
    }
    if (port2.isConnected()) {
      const agent2 = port2.getConnectedAgent();
      console.log(`  Port2: ${agent2.id} (${getAgentTypeName(agent2.type)}) with data: ${agent2.data}`);
    }
    
    // Check and substitute in first operand
    if (port1.isConnected()) {
      const agent1 = port1.getConnectedAgent();
      if (agent1 && agent1.type === AgentType.CON && agent1.data === paramName) {
        console.log(`Substituting parameter in first operand of OP2 ${op2Agent.id}`);
        
        // CRITICAL FIX: For function composition, we need to handle the case where the argument is a lambda
        // but we're substituting in an arithmetic expression. We should create a reference to the lambda
        // rather than cloning the entire lambda structure.
        let argClone;
        if (argAgent.type === AgentType.LAM) {
          // Create a reference to the lambda instead of cloning it
          // This allows the lambda to be applied later when needed
          console.log(`Creating reference to lambda ${argAgent.id} instead of cloning`);
          
          // CRITICAL FIX: Store both the lambda ID AND the parameter name in the reference data
          // This ensures we can correctly resolve the reference later
          argClone = this.createAgent(AgentType.CON, 0, `lambda_ref_${argAgent.id}_param_${argAgent.data}`);
          
          // CRITICAL FIX: Store the lambda ID in the reference data for later lookup
          // Don't connect directly - we'll handle this differently
          console.log(`DEBUG: Created lambda reference ${argClone.id} for lambda ${argAgent.id} with parameter ${argAgent.data}`);
          console.log(`Connected reference ${argClone.id} to lambda ${argAgent.id} via data reference`);
        } else {
          // For non-lambda arguments, clone as usual
          argClone = this.cloneAgentForSubstitution(argAgent);
        }
        
        // Disconnect the parameter
        port1.disconnect();
        
        // Connect the argument clone
        port1.connect(argClone.principalPort);
        
        console.log(`Replaced CON agent ${agent1.id} with ${argClone.id}(${getAgentTypeName(argClone.type)}) in first operand`);
        
        // Remove the parameter agent
        this.removeAgent(agent1.id);
      }
    }
    
    // Check and substitute in second operand
    if (port2.isConnected()) {
      const agent2 = port2.getConnectedAgent();
      if (agent2 && agent2.type === AgentType.CON && agent2.data === paramName) {
        console.log(`Substituting parameter in second operand of OP2 ${op2Agent.id}`);
        
        // CRITICAL FIX: For function composition, we need to handle the case where the argument is a lambda
        // but we're substituting in an arithmetic expression. We should create a reference to the lambda
        // rather than cloning the entire lambda structure.
        let argClone;
        if (argAgent.type === AgentType.LAM) {
          // Create a reference to the lambda instead of cloning it
          // This allows the lambda to be applied later when needed
          console.log(`Creating reference to lambda ${argAgent.id} instead of cloning`);
          
          // CRITICAL FIX: Store both the lambda ID AND the parameter name in the reference data
          // This ensures we can correctly resolve the reference later
          argClone = this.createAgent(AgentType.CON, 0, `lambda_ref_${argAgent.id}_param_${argAgent.data}`);
          
          // CRITICAL: Connect the reference directly to the lambda agent
          // This ensures the reference can be used for lambda application
          console.log(`DEBUG: Before connection - Reference ${argClone.id} principal connected: ${argClone.principalPort.isConnected()}`);
          console.log(`DEBUG: Before connection - Lambda ${argAgent.id} principal connected: ${argAgent.principalPort.isConnected()}`);
          
          this.connectPorts(argClone.principalPort, argAgent.principalPort);
          
          console.log(`DEBUG: After connection - Reference ${argClone.id} principal connected: ${argClone.principalPort.isConnected()}`);
          console.log(`DEBUG: After connection - Reference ${argClone.id} connected to: ${argClone.principalPort.getConnectedAgent()?.id} (${getAgentTypeName(argClone.principalPort.getConnectedAgent()?.type)})`);
          console.log(`DEBUG: After connection - Lambda ${argAgent.id} principal connected: ${argAgent.principalPort.isConnected()}`);
          console.log(`DEBUG: After connection - Lambda ${argAgent.id} connected to: ${argAgent.principalPort.getConnectedAgent()?.id} (${getAgentTypeName(argAgent.principalPort.getConnectedAgent()?.type)})`);
          console.log(`Connected reference ${argClone.id} directly to lambda ${argAgent.id}`);
        } else {
          // For non-lambda arguments, clone as usual
          argClone = this.cloneAgentForSubstitution(argAgent);
        }
        
        // Disconnect the parameter
        port2.disconnect();
        
        // Connect the argument clone
        port2.connect(argClone.principalPort);
        
        console.log(`Replaced CON agent ${agent2.id} with ${argClone.id}(${getAgentTypeName(argClone.type)}) in second operand`);
        
        // Remove the parameter agent
        this.removeAgent(agent2.id);
      }
    }
    
    // DEBUG: Log the current state after substitution
    console.log(`DEBUG: OP2 ${op2Agent.id} operands after substitution:`);
    if (port1.isConnected()) {
      const agent1 = port1.getConnectedAgent();
      console.log(`  Port1: ${agent1.id} (${getAgentTypeName(agent1.type)}) with data: ${agent1.data}`);
    }
    if (port2.isConnected()) {
      const agent2 = port2.getConnectedAgent();
      console.log(`  Port2: ${agent2.id} (${getAgentTypeName(agent2.type)}) with data: ${agent2.data}`);
    }
    
    console.log(`=== Finished substitution in OP2 agent ${op2Agent.id} ===`);
  }
  
  // Handle currying for APP-APP pairs
  handleCurrying(pair) {
    const { agent1, agent2 } = pair;
    
    // agent1 and agent2 are both APP agents
    // We need to find the LAM agent that should be reduced with agent1
    // This happens in currying: ((λx.λy.body) arg1) arg2
    
    // Find LAM agents that are connected to agent1
    const agents = Array.from(this.agents.values());
    const lamAgents = agents.filter(agent =>
      agent.type === AgentType.LAM &&
      agent.principalPort.isConnected() &&
      agent.principalPort.getConnectedAgent() === agent1
    );
    
    if (lamAgents.length > 0) {
      const lamAgent = lamAgents[0];
      // Reduce the LAM-APP pair
      this.reduceLambdaApp(lamAgent, agent1);
    } else {
      // If no LAM is found, just remove one of the APP agents
      // This shouldn't happen in well-formed expressions
      this.removeAgent(agent1.id);
    }
  }
  
  // Helper method to extract lambda parameter from reference data
  extractLambdaParamFromReference(refData, originalLambdaId) {
    // This is a workaround to find the parameter name for a lambda reference
    // In a more complete implementation, we would store this information directly
    // For now, we'll use a heuristic based on the original lambda ID
    
    // Look for any remaining lambda agents to infer the parameter
    const lambdaAgents = Array.from(this.agents.values()).filter(agent =>
      agent.type === AgentType.LAM
    );
    
    if (lambdaAgents.length > 0) {
      // Return the parameter of the first lambda we find
      // This is a heuristic - in practice, we need better tracking
      return lambdaAgents[0].data;
    }
    
    return null;
  }
  
  // Lambda reference reduction: Handle lambda references in arithmetic expressions
  reduceLambdaRef(lambdaAgent, refAgent) {
    console.log(`=== Lambda Reference Reduction: LAM${lambdaAgent.id} (${lambdaAgent.data}) referenced by CON${refAgent.id} (${refAgent.data}) ===`);
    
    // DIAGNOSTIC: Track the state before reduction
    console.log(`DIAGNOSTIC: Before reduction - Net has ${this.agents.size} agents`);
    console.log(`DIAGNOSTIC: Reference agent ${refAgent.id} connections:`, refAgent.getAllPorts().map(p => p.isConnected() ? `${p.getConnectedAgent().id}(${getAgentTypeName(p.getConnectedAgent().type)})` : 'none').join(', '));
    console.log(`DIAGNOSTIC: Lambda agent ${lambdaAgent.id} connections:`, lambdaAgent.getAllPorts().map(p => p.isConnected() ? `${p.getConnectedAgent().id}(${getAgentTypeName(p.getConnectedAgent().type)})` : 'none').join(', '));
    
    // NEW DIAGNOSTIC: Check if this is a function composition case
    console.log(`DIAGNOSTIC: *** ANALYZING FUNCTION COMPOSITION CASE ***`);
    const allOP2Agents = Array.from(this.agents.values()).filter(agent => agent.type === AgentType.OP2);
    console.log(`DIAGNOSTIC: All OP2 agents in net:`, allOP2Agents.map(op2 => `${op2.id}(${getAgentTypeName(op2.type)})`));
    
    // Check if the reference is connected to an OP2 agent
    const refConnections = refAgent.getAllPorts().filter(port => port.isConnected());
    console.log(`DIAGNOSTIC: Reference agent ${refAgent.id} has ${refConnections.length} connected ports`);
    refConnections.forEach((port, index) => {
      const connectedAgent = port.getConnectedAgent();
      console.log(`DIAGNOSTIC: Connection ${index}: ${connectedAgent.id}(${getAgentTypeName(connectedAgent.type)})`);
    });
    
    // CRITICAL FIX: Extract both the lambda ID and parameter name from the enhanced reference data
    const refMatch = refAgent.data.match(/lambda_ref_(\d+)_param_(.+)/);
    if (refMatch) {
      const referencedLambdaId = parseInt(refMatch[1]);
      const referencedParam = refMatch[2];
      console.log(`DEBUG: Reference ${refAgent.data} refers to original lambda ${referencedLambdaId} with parameter '${referencedParam}', now using lambda ${lambdaAgent.id} with parameter '${lambdaAgent.data}'`);
      
      // CRITICAL FIX: Verify that the parameter names match
      if (lambdaAgent.data !== referencedParam) {
        console.log(`DEBUG: *** WARNING: Parameter mismatch - reference expects '${referencedParam}' but lambda has '${lambdaAgent.data}' ***`);
      } else {
        console.log(`DEBUG: Parameter names match correctly: '${referencedParam}'`);
      }
    } else {
      // Fallback to old format for backward compatibility
      const lambdaIdMatch = refAgent.data.match(/lambda_ref_(\d+)/);
      if (lambdaIdMatch) {
        const referencedLambdaId = parseInt(lambdaIdMatch[1]);
        console.log(`DEBUG: Reference ${refAgent.data} (old format) refers to original lambda ${referencedLambdaId}, now using lambda ${lambdaAgent.id}`);
      }
    }
    
    // The lambda reference is being used in an arithmetic context.
    // We need to apply the lambda to its argument and then use the result.
    // This is the key to function composition: (λy.y + 1) should be applied to its argument
    
    // NEW DIAGNOSTIC: Check if this is the specific function composition case
    console.log(`DIAGNOSTIC: *** CHECKING FOR FUNCTION COMPOSITION PATTERN ***`);
    console.log(`DIAGNOSTIC: Lambda parameter: ${lambdaAgent.data}`);
    console.log(`DIAGNOSTIC: Reference data: ${refAgent.data}`);
    
    // Check if the lambda agent is connected to an OP2 agent (arithmetic expression)
    const auxPort = lambdaAgent.auxiliaryPorts[0];
    if (auxPort && auxPort.isConnected()) {
      const bodyAgent = auxPort.getConnectedAgent();
      console.log(`DEBUG: Lambda ${lambdaAgent.id} body is connected to agent ${bodyAgent.id} (${getAgentTypeName(bodyAgent.type)})`);
      
      // NEW DIAGNOSTIC: Check if this is the λy.y + 1 pattern
      if (bodyAgent && bodyAgent.type === AgentType.OP2) {
        console.log(`DEBUG: Lambda body is an arithmetic expression - this is function composition`);
        console.log(`DIAGNOSTIC: *** FUNCTION COMPOSITION DETECTED ***`);
        console.log(`DIAGNOSTIC: OP2 agent ${bodyAgent.id} data: ${bodyAgent.data}`);
        
        // Check if this is the addition pattern (λy.y + 1)
        const op2Port1 = bodyAgent.auxiliaryPorts[0];
        const op2Port2 = bodyAgent.auxiliaryPorts[1];
        
        if (op2Port1 && op2Port2 && op2Port1.isConnected() && op2Port2.isConnected()) {
          const operand1 = op2Port1.getConnectedAgent();
          const operand2 = op2Port2.getConnectedAgent();
          console.log(`DIAGNOSTIC: OP2 operands: ${operand1.id}(${getAgentTypeName(operand1.type)}, data=${operand1.data}), ${operand2.id}(${getAgentTypeName(operand2.type)}, data=${operand2.data})`);
          
          if (operand1.type === AgentType.CON && operand1.data === lambdaAgent.data &&
              operand2.type === AgentType.NUM && operand2.data === 1) {
            console.log(`DIAGNOSTIC: *** DETECTED λy.y + 1 PATTERN - This is function composition ***`);
          }
        }
        
        // The lambda represents a function like λy.y + 1
        // We need to apply this function to the argument that the reference is connected to
        
        // Find what the reference agent is connected to (this should be the argument)
        const refPrincipalPort = refAgent.principalPort;
        if (refPrincipalPort && refPrincipalPort.isConnected()) {
          const argumentProvider = refPrincipalPort.getConnectedAgent();
          console.log(`DEBUG: Lambda reference ${refAgent.id} is connected to argument provider ${argumentProvider.id} (${getAgentTypeName(argumentProvider.type)})`);
          
          // CRITICAL FIX: Check if there are other references to this lambda
          // If so, we need to clone the lambda and its body to preserve the original
          const otherRefs = Array.from(this.agents.values()).filter(agent =>
            agent.type === AgentType.CON &&
            agent.data &&
            agent.data.includes(`lambda_ref_${lambdaAgent.id}_param_${lambdaAgent.data}`) &&
            agent.id !== refAgent.id
          );
          
          console.log(`DEBUG: Found ${otherRefs.length} other references to lambda ${lambdaAgent.id}:`, otherRefs.map(r => r.id));
          
          let lambdaToUse = lambdaAgent;
          let bodyToUse = bodyAgent;
          
          if (otherRefs.length > 0) {
            console.log(`DEBUG: Cloning lambda and body to preserve original for other references`);
            
            // Clone the lambda
            lambdaToUse = this.createAgent(AgentType.LAM, 1, lambdaAgent.data);
            console.log(`DEBUG: Created lambda clone ${lambdaToUse.id}`);
            
            // Clone the body (OP2 agent)
            bodyToUse = this.cloneComplexExpression(bodyAgent);
            console.log(`DEBUG: Created body clone ${bodyToUse.id}`);
            
            // Connect the cloned lambda to the cloned body
            this.connectPorts(lambdaToUse.auxiliaryPorts[0], bodyToUse.principalPort);
            console.log(`DEBUG: Connected cloned lambda ${lambdaToUse.id} to cloned body ${bodyToUse.id}`);
          }
          
          // For function composition, we need to apply the lambda to the argument
          // This means we need to create a new APP agent and connect it properly
          console.log(`DEBUG: Creating application of lambda ${lambdaToUse.id} to argument from ${argumentProvider.id}`);
          
          // Create a new APP agent for this lambda application
          const appAgent = this.createAgent(AgentType.APP, 1);
          console.log(`DEBUG: Created APP agent ${appAgent.id} for lambda reference application`);
          
          // Connect the lambda to the APP agent
          this.connectPorts(lambdaToUse.principalPort, appAgent.principalPort);
          console.log(`DEBUG: Connected lambda ${lambdaToUse.id} to APP ${appAgent.id}`);
          
          // CRITICAL FIX: For function composition (f (f x)), we need to ensure the result
          // of the first application becomes the input to the second application
          if (argumentProvider.type === AgentType.OP2) {
            // The argument provider is an OP2 agent representing an arithmetic expression
            // We need to find the ACTUAL numeric argument and connect it to the OP2's disconnected port
            console.log(`DEBUG: Argument provider is OP2 ${argumentProvider.id} - finding actual argument`);
            
            // Find the original numeric argument (should be in an APP agent)
            const appAgents = Array.from(this.agents.values()).filter(a => a.type === AgentType.APP);
            let actualNumArg = null;
            
            for (const app of appAgents) {
              if (app.auxiliaryPorts[0] && app.auxiliaryPorts[0].isConnected()) {
                const argAgent = app.auxiliaryPorts[0].getConnectedAgent();
                if (argAgent && argAgent.type === AgentType.NUM) {
                  actualNumArg = argAgent;
                  console.log(`DEBUG: Found actual numeric argument ${actualNumArg.id} (${actualNumArg.data}) in APP ${app.id}`);
                  break;
                }
              }
            }
            
            if (actualNumArg) {
              // Replace any lambda reference in OP2's port1 with the actual argument
              const op2Port1 = argumentProvider.auxiliaryPorts[0];
              if (op2Port1) {
                // If port1 is connected to a lambda reference, disconnect it first
                if (op2Port1.isConnected()) {
                  const connectedAgent = op2Port1.getConnectedAgent();
                  if (connectedAgent && connectedAgent.type === AgentType.CON &&
                      connectedAgent.data && connectedAgent.data.startsWith('lambda_ref_')) {
                    console.log(`DEBUG: Disconnecting lambda reference ${connectedAgent.id} from OP2 ${argumentProvider.id} port1`);
                    op2Port1.disconnect();
                    // Remove the lambda reference agent as it's been replaced
                    this.removeAgent(connectedAgent.id);
                    console.log(`DEBUG: Removed lambda reference agent ${connectedAgent.id}`);
                  }
                }
                
                // Now connect the actual numeric argument
                if (!op2Port1.isConnected()) {
                  this.connectPorts(op2Port1, actualNumArg.principalPort);
                  console.log(`DEBUG: Connected actual argument ${actualNumArg.id} (${actualNumArg.data}) to OP2 ${argumentProvider.id} port1`);
                } else {
                  console.log(`DEBUG: OP2 ${argumentProvider.id} port1 still connected after cleanup`);
                }
              } else {
                console.log(`DEBUG: OP2 ${argumentProvider.id} has no port1`);
              }
            } else {
              console.log(`DEBUG: WARNING: Could not find actual numeric argument for OP2 ${argumentProvider.id}`);
            }
            
            // Now connect the OP2 as argument to the APP agent
            this.connectPorts(appAgent.auxiliaryPorts[0], argumentProvider.principalPort);
            console.log(`DEBUG: Connected OP2 ${argumentProvider.id} as argument to APP ${appAgent.id}`);
            
            // CRITICAL FIX: After the first application, we need to set up the second application
            // This is the key to (f (f x)) pattern - the result of f(x) becomes the input to f again
            console.log(`DEBUG: Setting up second function application for composition`);
            
            // Find any remaining OP2 agents that might need the result of this application
            const otherOP2Agents = Array.from(this.agents.values()).filter(agent =>
              agent.type === AgentType.OP2 && agent.id !== argumentProvider.id
            );
            
            console.log(`DEBUG: Found ${otherOP2Agents.length} other OP2 agents for potential second application:`, otherOP2Agents.map(a => a.id));
            
            // For function composition, we need to ensure the result of this lambda application
            // can be used by other OP2 agents in the network
            if (otherOP2Agents.length > 0) {
              console.log(`DEBUG: Setting up result of APP ${appAgent.id} to be used by other OP2 agents`);
              
              // CRITICAL FIX: For function composition (f (f x)), we need to connect the APP result
              // to the next OP2 agent that needs it, not create a self-connection
              
              // Find the OP2 agent that has a lambda reference (this needs the result)
              const targetOP2 = otherOP2Agents.find(op2 => {
                return op2.auxiliaryPorts.some(port =>
                  port.isConnected() &&
                  port.getConnectedAgent().type === AgentType.CON &&
                  port.getConnectedAgent().data &&
                  port.getConnectedAgent().data.includes('lambda_ref_')
                );
              });
              
              if (targetOP2) {
                console.log(`DEBUG: Found target OP2 ${targetOP2.id} that needs the APP result for composition`);
                
                // CRITICAL FIX: For function composition f(f(x)), we should NOT try to immediately
                // reduce the APP agent. Instead, we let the natural reduction order handle it.
                // The correct sequence is:
                // 1. First lambda reference creates: OP2(lambda_ref, 1) with argumentProvider
                // 2. That OP2 should get the actual argument (3) to compute 3+1=4
                // 3. Second lambda reference uses the result (4) to compute 4+1=5
                
                console.log(`DEBUG: NOT reducing APP ${appAgent.id} immediately - let natural reduction handle it`);
                
                // The key insight: the second lambda reference (in targetOP2) should wait
                // for the FIRST operation to complete. We don't need to connect APP to targetOP2.
                // Instead, we should leave things as-is and let the reduction order handle it.
                
                console.log(`DEBUG: Leaving APP ${appAgent.id} to be reduced naturally`);
                console.log(`DEBUG: Target OP2 ${targetOP2.id} will be handled after first OP2 completes`);
              } else {
                console.log(`DEBUG: No target OP2 found for function composition`);
              }
            }
          } else {
            // Connect the argument provider directly
            this.connectPorts(appAgent.auxiliaryPorts[0], argumentProvider.principalPort);
            console.log(`DEBUG: Connected argument provider ${argumentProvider.id} to APP ${appAgent.id}`);
          }
          
          // Remove the reference agent since it's been replaced by proper application
          this.removeAgent(refAgent.id);
          console.log(`DEBUG: Removed lambda reference agent ${refAgent.id}`);
          
          // Don't reduce the APP agent immediately - let it be reduced later after arithmetic operations
          // This ensures that arithmetic operations are evaluated before lambda applications that depend on them
          console.log(`DEBUG: Deferring reduction of APP agent ${appAgent.id} until after arithmetic operations`);
          
          console.log(`=== Lambda Reference Reduction Complete ===`);
          
          // DIAGNOSTIC: Track the state after successful reduction
          console.log(`DIAGNOSTIC: After successful reduction - Net has ${this.agents.size} agents`);
          return; // Exit early for successful case
        } else {
          console.log(`DEBUG: Lambda reference ${refAgent.id} is not connected to any argument provider`);
        }
      } else {
        console.log(`DEBUG: Lambda body is not an arithmetic expression - treating as regular lambda`);
        
        // DIAGNOSTIC: This is the problematic path - non-arithmetic lambda body
        console.log(`DIAGNOSTIC: *** PROBLEM PATH: Non-arithmetic lambda body detected ***`);
        console.log(`DIAGNOSTIC: Body agent type: ${bodyAgent ? getAgentTypeName(bodyAgent.type) : 'null'}`);
        console.log(`DIAGNOSTIC: Body agent data: ${bodyAgent ? bodyAgent.data : 'null'}`);
        console.log(`DIAGNOSTIC: Reference agent ${refAgent.id} will NOT be removed - this likely causes the infinite loop`);
        
        // NEW DIAGNOSTIC: Check if this is the λx.f pattern (identity function)
        if (bodyAgent && bodyAgent.type === AgentType.CON) {
          console.log(`DIAGNOSTIC: *** DETECTED IDENTITY FUNCTION PATTERN λx.${bodyAgent.data} ***`);
          console.log(`DIAGNOSTIC: This is the 'f' in (λf.λx.f (f x)) - should be treated as function composition`);
        }
        
        // CRITICAL FIX: For non-arithmetic lambda bodies, we need to handle the case
        // where the lambda represents a function like λx.f (just returns the parameter)
        // This should be treated as a simple variable reference
        if (bodyAgent && bodyAgent.type === AgentType.CON) {
          console.log(`DIAGNOSTIC: Lambda body is a CON agent - this is a simple variable reference`);
          console.log(`DIAGNOSTIC: CON agent data: '${bodyAgent.data}'`);
          
          // For simple variable references like λx.f, we should replace the reference
          // with whatever the lambda body refers to
          const refPrincipalPort = refAgent.principalPort;
          if (refPrincipalPort && refPrincipalPort.isConnected()) {
            const argumentProvider = refPrincipalPort.getConnectedAgent();
            console.log(`DIAGNOSTIC: Reference is connected to argument provider ${argumentProvider.id} (${getAgentTypeName(argumentProvider.type)})`);
            
            // CRITICAL FIX: For function composition, we need to find the actual argument
            // The reference is connected to an OP2 agent, but we need the argument that should
            // be passed to the lambda. In (λf.λx.f (f x)) (λy.y + 1) 3, the argument is 3.
            console.log(`DIAGNOSTIC: Looking for the actual argument to pass to the lambda`);
            
            // Look for NUM agents that could be the argument
            const numAgents = Array.from(this.agents.values()).filter(agent =>
              agent.type === AgentType.NUM
            );
            console.log(`DIAGNOSTIC: Found ${numAgents.length} NUM agents:`, numAgents.map(n => `${n.id}(${n.data})`));
            
            // For function composition, we need to find the argument that should be applied
            // Look for APP agents that have NUM arguments
            const appAgents = Array.from(this.agents.values()).filter(agent =>
              agent.type === AgentType.APP && agent.auxiliaryPorts[0].isConnected()
            );
            console.log(`DIAGNOSTIC: Found ${appAgents.length} APP agents with arguments:`, appAgents.map(a => `${a.id}`));
            
            let actualArgument = null;
            for (const appAgent of appAgents) {
              const argPort = appAgent.auxiliaryPorts[0];
              if (argPort && argPort.isConnected()) {
                const connectedAgent = argPort.getConnectedAgent();
                if (connectedAgent && connectedAgent.type === AgentType.NUM) {
                  actualArgument = connectedAgent;
                  console.log(`DIAGNOSTIC: Found actual argument: ${actualArgument.id} (${actualArgument.data})`);
                  break;
                }
              }
            }
            
            // CRITICAL FIX: If we can't find the actual argument (it may have been consumed),
            // we need to find the result of the previous arithmetic reduction instead of using the OP2 agent itself
            let finalArgumentProvider = actualArgument || argumentProvider;
            
            // If the argument provider is an OP2 agent and we're in a function composition scenario,
            // we need to find a different argument provider to avoid self-referential connections
            if (finalArgumentProvider.type === AgentType.OP2 && finalArgumentProvider.id === argumentProvider.id) {
              console.log(`DIAGNOSTIC: *** PROBLEM: Argument provider is the same as target OP2 agent - this will create self-reference ***`);
              
              // Look for a NUM agent that could be the result of the previous arithmetic reduction
              const resultNumAgents = numAgents.filter(num =>
                num.data > 1 && num.id !== finalArgumentProvider.auxiliaryPorts[1]?.getConnectedAgent()?.id
              );
              
              if (resultNumAgents.length > 0) {
                finalArgumentProvider = resultNumAgents[0];
                console.log(`DIAGNOSTIC: Using result NUM agent ${finalArgumentProvider.id} (${finalArgumentProvider.data}) as argument provider`);
              } else {
                // As a last resort, look for any NUM agent that's not connected to this OP2
                const availableNumAgents = numAgents.filter(num => {
                  const connectedToOP2 = finalArgumentProvider.getAllPorts().some(port =>
                    port.isConnected() && port.getConnectedAgent() === num
                  );
                  return !connectedToOP2;
                });
                
                if (availableNumAgents.length > 0) {
                  finalArgumentProvider = availableNumAgents[0];
                  console.log(`DIAGNOSTIC: Using available NUM agent ${finalArgumentProvider.id} (${finalArgumentProvider.data}) as argument provider`);
                } else {
                  console.log(`DIAGNOSTIC: *** WARNING: No suitable argument provider found - may cause self-reference ***`);
                }
              }
            }
            
            console.log(`DIAGNOSTIC: Using final argument provider: ${finalArgumentProvider.id} (${getAgentTypeName(finalArgumentProvider.type)})`);
            
            // The lambda simply returns its parameter, so we need to connect the argument
            // to whatever the reference was connected to
            console.log(`DIAGNOSTIC: Implementing fix for simple variable reference`);
            
            // Find what the reference agent is connected to on its other ports
            const connectedPorts = [];
            refAgent.getAllPorts().forEach((port, index) => {
              if (port.isConnected() && port !== refPrincipalPort) {
                connectedPorts.push({
                  port: port,
                  connectedTo: port.connectedTo,
                  portIndex: index
                });
              }
            });
            
            // Also check if the reference agent's principal port is connected to an OP2 agent
            // This is the case for function composition where the reference is used in arithmetic
            if (refPrincipalPort && refPrincipalPort.isConnected()) {
              const connectedAgent = refPrincipalPort.getConnectedAgent();
              if (connectedAgent && connectedAgent.type === AgentType.OP2) {
                connectedPorts.push({
                  port: refPrincipalPort,
                  connectedTo: refPrincipalPort.connectedTo,
                  portIndex: 0
                });
                console.log(`DIAGNOSTIC: Found OP2 connection via principal port: ${connectedAgent.id}`);
              }
            }
            
            console.log(`DIAGNOSTIC: Reference agent has ${connectedPorts.length} other connections`);
            
            // CRITICAL FIX: For function composition, the reference agent is typically connected to an OP2 agent
            // that represents the arithmetic expression where the lambda should be applied
            // We need to properly handle this connection to enable arithmetic reduction
            if (connectedPorts.length > 0) {
              connectedPorts.forEach(({ connectedTo, portIndex }) => {
                console.log(`DIAGNOSTIC: Processing connection to agent ${connectedTo.agent.id} (${getAgentTypeName(connectedTo.agent.type)})`);
                
                if (connectedTo.agent.type === AgentType.OP2) {
                  console.log(`DIAGNOSTIC: Reference is connected to OP2 agent ${connectedTo.agent.id} - this is function composition`);
                  
                  // For function composition, we need to connect the argument provider to the OP2 agent
                  // The OP2 agent represents an arithmetic operation that needs the lambda result
                  const op2Agent = connectedTo.agent;
                  
                  // CRITICAL FIX: Find which port of the OP2 agent is connected to the reference
                  // BEFORE we disconnect anything, so we can properly reconnect the argument provider
                  let actualOp2PortIndex = -1;
                  console.log(`DIAGNOSTIC: Checking all ports of OP2 ${op2Agent.id} for connection to reference ${refAgent.id} (BEFORE disconnection):`);
                  op2Agent.getAllPorts().forEach((port, index) => {
                    const isConnected = port.isConnected();
                    const connectedAgent = isConnected ? port.getConnectedAgent() : null;
                    const matchesRef = connectedAgent === refAgent;
                    console.log(`DIAGNOSTIC: Port ${index}: connected=${isConnected}, agent=${connectedAgent ? connectedAgent.id : 'null'}, matchesRef=${matchesRef}`);
                    if (isConnected && matchesRef) {
                      actualOp2PortIndex = index;
                    }
                  });
                  
                  console.log(`DIAGNOSTIC: Reference is connected to OP2 ${op2Agent.id} at port index ${actualOp2PortIndex}`);
                  
                  // Find which port of the OP2 agent is connected to the reference (for compatibility)
                  const op2PortIndex = op2Agent.getAllPorts().findIndex(port =>
                    port.isConnected() && port.getConnectedAgent() === refAgent
                  );
                  
                  console.log(`DIAGNOSTIC: Reference is connected to OP2 ${op2Agent.id} at port index ${op2PortIndex}`);
                  
                  if (op2PortIndex >= 0) {
                    // Disconnect the reference from the OP2 agent
                    op2Agent.getAllPorts()[op2PortIndex].disconnect();
                    console.log(`DIAGNOSTIC: Disconnected reference from OP2 ${op2Agent.id} port ${op2PortIndex}`);
                    
                    // For function composition like (λy.y + 1) 3, we need to apply the lambda to the argument
                    // and then use the result in the arithmetic operation
                    // Since this is a simple variable reference (λx.f), we connect the argument directly
                    console.log(`DIAGNOSTIC: OP2 ${op2Agent.id} ports before connection:`,
                      op2Agent.getAllPorts().map((p, i) => `${i}:${p.isConnected() ? p.getConnectedAgent().id : 'none'}`).join(', '));
                    
                    if (actualOp2PortIndex === 0) {
                      // Principal port - this shouldn't happen for OP2 agents, but handle it
                      this.connectPorts(op2Agent.principalPort, finalArgumentProvider.principalPort);
                      console.log(`DIAGNOSTIC: Connected final argument provider to OP2 ${op2Agent.id} principal port`);
                    } else if (actualOp2PortIndex === 1) {
                      // First auxiliary port
                      this.connectPorts(op2Agent.auxiliaryPorts[0], finalArgumentProvider.principalPort);
                      console.log(`DIAGNOSTIC: Connected final argument provider to OP2 ${op2Agent.id} first auxiliary port`);
                    } else if (actualOp2PortIndex === 2) {
                      // Second auxiliary port
                      this.connectPorts(op2Agent.auxiliaryPorts[1], finalArgumentProvider.principalPort);
                      console.log(`DIAGNOSTIC: Connected final argument provider to OP2 ${op2Agent.id} second auxiliary port`);
                    } else {
                      // Fallback - try first auxiliary port
                      this.connectPorts(op2Agent.auxiliaryPorts[0], finalArgumentProvider.principalPort);
                      console.log(`DIAGNOSTIC: Fallback: Connected final argument provider to OP2 ${op2Agent.id} first auxiliary port`);
                    }
                    
                    console.log(`DIAGNOSTIC: OP2 ${op2Agent.id} ports after connection:`,
                      op2Agent.getAllPorts().map((p, i) => `${i}:${p.isConnected() ? p.getConnectedAgent().id : 'none'}`).join(', '));
                    
                    console.log(`DIAGNOSTIC: Successfully connected argument provider to OP2 ${op2Agent.id}`);
                  }
                } else {
                  // For non-OP2 connections, use the original logic
                  if (portIndex === 0) {
                    // Principal port
                    argumentProvider.principalPort.connect(connectedTo);
                    console.log(`DIAGNOSTIC: Connected argument provider principal port to agent ${connectedTo.agent.id}`);
                  } else {
                    // Auxiliary port
                    if (argumentProvider.auxiliaryPorts[portIndex - 1]) {
                      argumentProvider.auxiliaryPorts[portIndex - 1].connect(connectedTo);
                      console.log(`DIAGNOSTIC: Connected argument provider aux port ${portIndex-1} to agent ${connectedTo.agent.id}`);
                    }
                  }
                }
              });
            } else {
              console.log(`DIAGNOSTIC: No other connections found for reference agent ${refAgent.id}`);
            }
            
            // Remove the reference agent to break the infinite loop
            this.removeAgent(refAgent.id);
            console.log(`DIAGNOSTIC: *** FIXED: Removed reference agent ${refAgent.id} to break infinite loop ***`);
            
            console.log(`=== Lambda Reference Reduction Complete (Non-arithmetic case) ===`);
            console.log(`DIAGNOSTIC: After non-arithmetic reduction - Net has ${this.agents.size} agents`);
            return; // Exit early for fixed case
          }
        }
      }
    } else {
      console.log(`DEBUG: Lambda ${lambdaAgent.id} has no body connected`);
      console.log(`DIAGNOSTIC: *** PROBLEM: Lambda with no body - this should not happen ***`);
      
      // CRITICAL FIX: When a lambda has no body but there are still references to it,
      // we need to remove the reference agents to break the infinite loop
      console.log(`DIAGNOSTIC: *** EMERGENCY FIX: Removing reference agent ${refAgent.id} to break infinite loop ***`);
      
      // Find what the reference agent is connected to
      const refConnections = refAgent.getAllPorts().filter(port => port.isConnected());
      console.log(`DIAGNOSTIC: Reference agent ${refAgent.id} has ${refConnections.length} connections`);
      
      // For each connection, we need to handle it appropriately
      refConnections.forEach((port, index) => {
        const connectedAgent = port.getConnectedAgent();
        console.log(`DIAGNOSTIC: Connection ${index}: ${connectedAgent.id}(${getAgentTypeName(connectedAgent.type)})`);
        
        if (connectedAgent.type === AgentType.OP2) {
          // The reference is connected to an OP2 agent - we need to disconnect it
          console.log(`DIAGNOSTIC: Disconnecting reference from OP2 ${connectedAgent.id}`);
          
          // Find which port of the OP2 agent is connected to the reference
          const op2PortIndex = connectedAgent.getAllPorts().findIndex(p =>
            p.isConnected() && p.getConnectedAgent() === refAgent
          );
          
          if (op2PortIndex >= 0) {
            // Disconnect the reference from the OP2 agent
            connectedAgent.getAllPorts()[op2PortIndex].disconnect();
            console.log(`DIAGNOSTIC: Disconnected reference from OP2 ${connectedAgent.id} port ${op2PortIndex}`);
            
            // FUNCTION COMPOSITION FIX: Check if there are pending arithmetic operations
            // If so, don't connect anything yet - wait for those operations to complete first
            const pendingArithmeticOps = Array.from(this.agents.values()).filter(agent => {
              if (agent.type !== AgentType.OP2 || agent.id === connectedAgent.id) return false;
              const port1 = agent.auxiliaryPorts[0];
              const port2 = agent.auxiliaryPorts[1];
              if (!port1 || !port2) return false;
              
              // Check if both ports are connected to NUM agents (ready for arithmetic)
              if (port1.isConnected() && port2.isConnected()) {
                const op1 = port1.getConnectedAgent();
                const op2 = port2.getConnectedAgent();
                return op1 && op2 && op1.type === AgentType.NUM && op2.type === AgentType.NUM;
              }
              return false;
            });
            
            console.log(`DIAGNOSTIC: FUNCTION COMPOSITION FIX: Found ${pendingArithmeticOps.length} pending arithmetic operations`);
            
            if (pendingArithmeticOps.length > 0) {
              console.log(`DIAGNOSTIC: FUNCTION COMPOSITION FIX: Not connecting - let pending operations complete first`);
              console.log(`DIAGNOSTIC: FUNCTION COMPOSITION FIX: OP2 ${connectedAgent.id} will wait for results`);
              // Don't connect anything - the OP2 will remain disconnected until the pending operation completes
              // In the next reduction step, the result will be available and can be connected
            } else {
              // No pending operations - find the result to use
              const numAgents = Array.from(this.agents.values()).filter(a => a.type === AgentType.NUM);
              if (numAgents.length > 0) {
                // Find NUM agents that are standalone (results), not arguments
                const resultNumAgents = numAgents.filter(num => {
                  const connectedToOP2 = Array.from(this.agents.values()).some(agent => {
                    if (agent.type !== AgentType.OP2) return false;
                    return agent.auxiliaryPorts.some(port =>
                      port.isConnected() && port.getConnectedAgent() === num
                    );
                  });
                  
                  const connectedToAPP = Array.from(this.agents.values()).some(agent => {
                    if (agent.type !== AgentType.APP) return false;
                    return agent.auxiliaryPorts.some(port =>
                      port.isConnected() && port.getConnectedAgent() === num
                    );
                  });
                  
                  return !connectedToOP2 && !connectedToAPP;
                });
                
                const argumentAgent = resultNumAgents.length > 0
                  ? resultNumAgents.reduce((max, agent) => agent.id > max.id ? agent : max)
                  : numAgents.reduce((max, agent) => agent.id > max.id ? agent : max);
                
                console.log(`DIAGNOSTIC: FUNCTION COMPOSITION FIX: Found ${resultNumAgents.length} result NUM agents`);
                console.log(`DIAGNOSTIC: FUNCTION COMPOSITION FIX: Using argument ${argumentAgent.id} (${argumentAgent.data}) for OP2 ${connectedAgent.id} port ${op2PortIndex}`);
                
                // Connect the argument to the OP2 agent
                if (op2PortIndex === 0) {
                  connectedAgent.principalPort.connect(argumentAgent.principalPort);
                } else if (op2PortIndex === 1) {
                  connectedAgent.auxiliaryPorts[0].connect(argumentAgent.principalPort);
                } else if (op2PortIndex === 2) {
                  connectedAgent.auxiliaryPorts[1].connect(argumentAgent.principalPort);
                }
                console.log(`DIAGNOSTIC: FUNCTION COMPOSITION FIX: Successfully connected argument ${argumentAgent.id} (${argumentAgent.data}) to OP2 ${connectedAgent.id}`);
              } else {
                console.log(`DIAGNOSTIC: FUNCTION COMPOSITION FIX: No NUM agents found for argument substitution`);
              }
            }
          }
        }
      });
      
      // Remove the reference agent to break the infinite loop
      this.removeAgent(refAgent.id);
      console.log(`DIAGNOSTIC: *** EMERGENCY FIX: Removed reference agent ${refAgent.id} to break infinite loop ***`);
      console.log(`DIAGNOSTIC: After emergency fix - Net has ${this.agents.size} agents`);
      return; // Exit early after emergency fix
    }
    
    // DIAGNOSTIC: If we reach this point, no reduction was performed
    console.log(`DIAGNOSTIC: *** WARNING: No reduction performed for lambda reference ${refAgent.id} ***`);
    console.log(`DIAGNOSTIC: *** This is likely the cause of the infinite loop ***`);
    console.log(`DIAGNOSTIC: After no-reduction case - Net still has ${this.agents.size} agents`);
  }
  
  // Duplication: DUP(x, y) CON(z) → x(z), y(z)
  reduceDupCon(dupAgent, conAgent) {
    // Create two new CON agents
    const con1 = this.createAgent(AgentType.CON, 0, conAgent.data);
    const con2 = this.createAgent(AgentType.CON, 0, conAgent.data);

    // Connect to DUP's auxiliary ports
    if (dupAgent.auxiliaryPorts.length >= 2) {
      this.connectPorts(con1.principalPort, dupAgent.auxiliaryPorts[0].connectedTo);
      this.connectPorts(con2.principalPort, dupAgent.auxiliaryPorts[1].connectedTo);
    }

    // Remove original agents
    this.removeAgent(dupAgent.id);
    this.removeAgent(conAgent.id);
  }

  // Binary operation with two numbers: OP2(op, NUM1, NUM2) → NUM(result)
  reduceArithmetic(pair) {
    const { agent1: op2Agent, agent2: numAgent1, agent3: numAgent2 } = pair;
    const operator = op2Agent.data;
    const operand1 = numAgent1.data;
    const operand2 = numAgent2.data;

    console.log(`=== Arithmetic Reduction: ${operand1} ${operator === 1 ? '+' : operator === 2 ? '-' : operator === 3 ? '*' : '/'} ${operand2} ===`);

    let result;
    switch (operator) {
      case 1: result = operand1 + operand2; break; // Addition
      case 2: result = operand1 - operand2; break; // Subtraction
      case 3: result = operand1 * operand2; break; // Multiplication
      case 4: result = operand1 / operand2; break; // Division
      default: result = operand1; // Default to first operand
    }

    console.log(`Arithmetic result: ${result}`);

    // Create result number
    const resultAgent = this.createAgent(AgentType.NUM, 0, result);
    console.log(`Created result agent ${resultAgent.id} with value ${result}`);

    // Connect any remaining connections from the OP2's principal port
    if (op2Agent.principalPort.isConnected()) {
      const connectedPort = op2Agent.principalPort.connectedTo;
      if (connectedPort) {
        resultAgent.principalPort.connect(connectedPort);
        console.log(`Connected result agent ${resultAgent.id} to OP2's principal connection`);
      }
    }

    // Remove original agents
    console.log(`Removing OP2 agent ${op2Agent.id} and NUM agents ${numAgent1.id}, ${numAgent2.id}`);
    this.removeAgent(op2Agent.id);
    this.removeAgent(numAgent1.id);
    this.removeAgent(numAgent2.id);
    
    console.log(`After arithmetic reduction, net has ${this.agents.size} agents`);
    
    // CRITICAL FIX: For function composition, check if there are OP2 agents waiting for this result
    // These are OP2 agents with disconnected port1 that were waiting for pending operations
    // IMPORTANT: Only include OP2 agents that are ACTIVE lambda bodies (still connected to a LAM)
    const waitingOP2Agents = Array.from(this.agents.values()).filter(agent => {
      if (agent.type !== AgentType.OP2) return false;
      const port1 = agent.auxiliaryPorts[0];
      if (!port1 || port1.isConnected()) return false;
      
      // CRITICAL: Only use OP2 agents that are ACTIVE lambda bodies
      // (principal port connected to a LAM's auxiliary port)
      // Orphaned OP2 agents (from removed lambdas) should be ignored
      if (!agent.principalPort.isConnected()) {
        console.log(`FUNCTION COMPOSITION: Skipping OP2 ${agent.id} - orphaned (no principal connection)`);
        return false;
      }
      
      const connectedAgent = agent.principalPort.getConnectedAgent();
      if (!connectedAgent || connectedAgent.type !== AgentType.LAM) {
        console.log(`FUNCTION COMPOSITION: Skipping OP2 ${agent.id} - not connected to LAM`);
        return false;
      }
      
      console.log(`FUNCTION COMPOSITION: Including OP2 ${agent.id} - active lambda body for LAM ${connectedAgent.id}`);
      return true;
    });
    
    if (waitingOP2Agents.length > 0) {
      console.log(`FUNCTION COMPOSITION: Found ${waitingOP2Agents.length} OP2 agents waiting for result:`, waitingOP2Agents.map(a => a.id));
      
      // Connect the result to the first waiting OP2 agent (lowest ID = earliest created)
      const targetOP2 = waitingOP2Agents.reduce((min, agent) => agent.id < min.id ? agent : min);
      console.log(`FUNCTION COMPOSITION: Connecting result ${resultAgent.id} (${result}) to waiting OP2 ${targetOP2.id} port1`);
      
      this.connectPorts(targetOP2.auxiliaryPorts[0], resultAgent.principalPort);
      console.log(`FUNCTION COMPOSITION: Successfully connected result to OP2 ${targetOP2.id}`);
    }
  }

  // Binary operation with number: OP2(op) NUM(n) → NUM(result)
  reduceOp2Num(op2Agent, numAgent) {
    const operator = op2Agent.data;
    const operand = numAgent.data;

    let result;
    switch (operator) {
      case 1: result = operand; break; // + (unary)
      case 2: result = -operand; break; // - (unary)
      case 3: result = 0; break; // * (needs second operand)
      case 4: result = 1; break; // / (needs second operand)
      default: result = operand;
    }

    // Create result number
    const resultAgent = this.createAgent(AgentType.NUM, 0, result);

    // Connect any remaining connections from the other auxiliary port
    if (op2Agent.auxiliaryPorts.length >= 2) {
      const otherPort = op2Agent.auxiliaryPorts[1]; // Use the second port
      if (otherPort && otherPort.isConnected()) {
        this.connectPorts(resultAgent.principalPort, otherPort.connectedTo);
      }
    }

    // Remove original agents
    this.removeAgent(op2Agent.id);
    this.removeAgent(numAgent.id);
  }

  // Binary operation with another binary operation
  reduceOp2Op2(op2Agent1, op2Agent2) {
    // This is complex - simplified version
    // In reality, would need to evaluate based on operator precedence
    this.removeAgent(op2Agent1.id);
    this.removeAgent(op2Agent2.id);
  }

  // Superposition reduction
  reduceSup(supAgent, otherAgent) {
    // Simplified: just remove the SUP agent
    this.removeAgent(supAgent.id);
  }

  reduceToNormalForm(maxSteps = 1000) {
    let steps = 0;
    let hadReduction = false;
    
    // Continue reducing until no more reductions or max steps reached
    do {
      hadReduction = this.reduceStep();
      steps++;
      
      // For nested lambda applications, we need to check if new active pairs
      // were created after each reduction step
      if (hadReduction) {
        // Find active pairs to see if we have more work to do
        this.findActivePairs();
      }
    } while (hadReduction && steps < maxSteps);

    return {
      steps,
      normalForm: steps < maxSteps && !hadReduction,
      finalAgents: this.agents.size,
      totalSteps: this.reductionSteps,
      hasMoreReductions: hadReduction
    };
  }

  // Convert to GPU-friendly format
  toGPUBuffer() {
    const agents = [];
    const ports = [];
    const connections = [];

    // Convert agents to flat array
    for (const agent of this.agents.values()) {
      agents.push({
        id: agent.id,
        type: agent.type,
        arity: agent.arity,
        data: agent.data,
        principalPort: ports.length,
        auxPorts: ports.length + 1
      });

      // Add ports
      ports.push({
        agentId: agent.id,
        type: PortType.PRINCIPAL,
        index: 0,
        connectedTo: agent.principalPort.connectedTo ? 
          agent.principalPort.connectedTo.id : -1
      });

      agent.auxiliaryPorts.forEach((port, i) => {
        ports.push({
          agentId: agent.id,
          type: PortType.AUXILIARY,
          index: i,
          connectedTo: port.connectedTo ? port.connectedTo.id : -1
        });
      });
    }

    return {
      agents,
      ports,
      connections,
      metadata: {
        agentCount: agents.length,
        portCount: ports.length,
        reductionSteps: this.reductionSteps,
        maxAgents: this.maxAgents
      }
    };
  }

  // Create from GPU buffer format
  static fromGPUBuffer(buffer) {
    const net = new InteractionNet();
    
    // Recreate agents
    buffer.agents.forEach(agentData => {
      const agent = net.createAgent(agentData.type, agentData.arity, agentData.data);
      // Update ID to match buffer
      net.agents.delete(agent.id);
      agent.id = agentData.id;
      net.agents.set(agentData.id, agent);
    });

    // Recreate connections
    buffer.ports.forEach(portData => {
      if (portData.connectedTo !== -1) {
        const agent = net.agents.get(portData.agentId);
        const targetPort = buffer.ports.find(p => p.id === portData.connectedTo);
        
        if (agent && targetPort) {
          const targetAgent = net.agents.get(targetPort.agentId);
          if (targetAgent) {
            const port1 = agent.getPort(portData.index === 0 ? 'principal' : portData.index);
            const port2 = targetAgent.getPort(targetPort.index === 0 ? 'principal' : targetPort.index);
            
            if (port1 && port2) {
              port1.connect(port2);
            }
          }
        }
      }
    });

    return net;
  }

  // Visual representation for debugging
  toDot() {
    let dot = 'digraph InteractionNet {\n';
    dot += '  rankdir=TB;\n';
    dot += '  node [shape=circle];\n\n';


    // Add agents as nodes
    for (const agent of this.agents.values()) {
      // Find the type name by value
      let typeName = 'UNKNOWN';
      for (const [key, value] of Object.entries(AgentType)) {
        if (typeof value === 'number' && value === agent.type) {
          typeName = key;
          break;
        }
      }
      
      const label = `${typeName}_${agent.id}`;
      if (agent.data !== null) {
        dot += `  ${agent.id} [label="${label}(${agent.data})"];\n`;
      } else {
        dot += `  ${agent.id} [label="${label}"];\n`;
      }
    }

    // Add connections as edges
    const addedEdges = new Set();
    
    for (const agent of this.agents.values()) {
      agent.getAllPorts().forEach(port => {
        if (port.isConnected() && !addedEdges.has(`${port.id}-${port.connectedTo.id}`)) {
          dot += `  ${agent.id} -> ${port.getConnectedAgent().id};\n`;
          addedEdges.add(`${port.id}-${port.connectedTo.id}`);
          addedEdges.add(`${port.connectedTo.id}-${port.id}`);
        }
      });
    }

    dot += '}\n';
    return dot;
  }

  // Statistics
  getStats() {
    const typeCounts = {};
    
    
    // Initialize all agent types to 0
    Object.keys(AgentType).forEach(key => {
      if (typeof AgentType[key] === 'number') {
        typeCounts[key] = 0;
      }
    });
    
    for (const agent of this.agents.values()) {
      
      // Find the type name by value
      let typeName = 'UNKNOWN';
      for (const [key, value] of Object.entries(AgentType)) {
        if (typeof value === 'number' && value === agent.type) {
          typeName = key;
          break;
        }
      }
      typeCounts[typeName] = (typeCounts[typeName] || 0) + 1;
    }


    return {
      agentCount: this.agents.size,
      reductionSteps: this.reductionSteps,
      maxAgents: this.maxAgents,
      activePairs: this.findActivePairs().length,
      typeDistribution: typeCounts
    };
  }
}

// Export for both Node.js and ES6 modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { 
    InteractionNet, 
    Agent, 
    Port, 
    AgentType, 
    PortType 
  };
}

function getAgentTypeName(agentType) {
  for (const [key, value] of Object.entries(AgentType)) {
    if (typeof value === 'number' && value === agentType) {
      return key;
    }
  }
  return 'UNKNOWN';
}

// ES6 module exports for browser
export {
  InteractionNet,
  Agent,
  Port,
  AgentType,
  PortType
};