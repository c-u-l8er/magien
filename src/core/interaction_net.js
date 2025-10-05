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
    port1.connect(port2);
  }

  findActivePairs() {
    this.activePairs = [];
    const visited = new Set();

    // Find principal-to-principal active pairs (lambda-beta reduction)
    for (const agent of this.agents.values()) {
      if (visited.has(agent.id)) continue;

      const principalPort = agent.principalPort;
      
      if (principalPort.isConnected()) {
        const connectedAgent = principalPort.getConnectedAgent();
        
        // Check if this is an active pair (principal-to-principal connection)
        if (connectedAgent &&
            principalPort.type === PortType.PRINCIPAL &&
            principalPort.connectedTo &&
            principalPort.connectedTo.type === PortType.PRINCIPAL) {
          
          this.activePairs.push({
            agent1: agent,
            agent2: connectedAgent,
            port: principalPort,
            type: 'principal'
          });
          
          visited.add(agent.id);
          visited.add(connectedAgent.id);
        }
      }
    }

    // Find arithmetic reduction opportunities (OP2 agents with connected NUM agents)
    for (const agent of this.agents.values()) {
      if (visited.has(agent.id)) continue;
      
      if (agent.type === AgentType.OP2 && agent.arity >= 2) {
        const port1 = agent.auxiliaryPorts[0];
        const port2 = agent.auxiliaryPorts[1];
        
        if (port1.isConnected() && port2.isConnected()) {
          const agent1 = port1.getConnectedAgent();
          const agent2 = port2.getConnectedAgent();
          
          if (agent1 && agent2 &&
              agent1.type === AgentType.NUM &&
              agent2.type === AgentType.NUM) {
            
            this.activePairs.push({
              agent1: agent,
              agent2: agent1,
              agent3: agent2,
              type: 'arithmetic'
            });
            
            visited.add(agent.id);
            visited.add(agent1.id);
            visited.add(agent2.id);
          }
        }
      }
    }

    return this.activePairs;
  }

  reduceStep() {
    const activePairs = this.findActivePairs();
    
    if (activePairs.length === 0) {
      return false; // No more reductions possible
    }

    // Reduce each active pair
    for (const pair of activePairs) {
      if (pair.type === 'arithmetic') {
        this.reduceArithmetic(pair);
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

  // Beta reduction: (λx.M) N → M[x := N]
  reduceLambdaApp(lambdaAgent, appAgent) {
    // Get the argument from the APP agent
    const argPort = appAgent.auxiliaryPorts[0];
    if (argPort && argPort.isConnected()) {
      const argAgent = argPort.getConnectedAgent();
      
      if (argAgent) {
        // Find all CON agents that represent the lambda parameter
        const paramName = lambdaAgent.data;
        const agents = Array.from(this.agents.values());
        
        for (const agent of agents) {
          if (agent.type === AgentType.CON && agent.data === paramName) {
            // Replace the parameter reference with the argument
            // Connect whatever was connected to the parameter to the argument
            if (agent.principalPort.isConnected()) {
              const connectedPort = agent.principalPort.connectedTo;
              connectedPort.connect(argAgent.principalPort);
            }
            // Remove the parameter CON agent
            this.removeAgent(agent.id);
          }
        }
      }
    }

    // Remove the original agents
    this.removeAgent(lambdaAgent.id);
    this.removeAgent(appAgent.id);
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

    let result;
    switch (operator) {
      case 1: result = operand1 + operand2; break; // Addition
      case 2: result = operand1 - operand2; break; // Subtraction
      case 3: result = operand1 * operand2; break; // Multiplication
      case 4: result = operand1 / operand2; break; // Division
      default: result = operand1; // Default to first operand
    }

    // Create result number
    const resultAgent = this.createAgent(AgentType.NUM, 0, result);

    // Connect any remaining connections from the OP2's principal port
    if (op2Agent.principalPort.isConnected()) {
      const connectedPort = op2Agent.principalPort.connectedTo;
      if (connectedPort) {
        resultAgent.principalPort.connect(connectedPort);
      }
    }

    // Remove original agents
    this.removeAgent(op2Agent.id);
    this.removeAgent(numAgent1.id);
    this.removeAgent(numAgent2.id);
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
    
    while (this.reduceStep() && steps < maxSteps) {
      steps++;
    }

    return {
      steps,
      normalForm: steps < maxSteps,
      finalAgents: this.agents.size,
      totalSteps: this.reductionSteps
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

// ES6 module exports for browser
export { 
  InteractionNet, 
  Agent, 
  Port, 
  AgentType, 
  PortType 
};