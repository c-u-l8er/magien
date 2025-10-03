/**
 * GPU-Accelerated Interaction Net Reducer
 * Parallel reduction of interaction nets using WebGPU compute shaders
 */

class GPUNetReducer {
  constructor(device) {
    this.device = device;
    this.pipeline = null;
    this.bindGroupLayout = null;
    this.maxAgents = 65536; // Maximum agents per net
    this.maxPorts = this.maxAgents * 4; // Each agent can have up to 4 ports
    this.workgroupSize = 256;
  }

  async initialize() {
    // Create bind group layout
    this.bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'storage',
            hasDynamicOffset: false,
            minBindingSize: this.maxAgents * 32 // Agent struct
          }
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'storage',
            hasDynamicOffset: false,
            minBindingSize: this.maxPorts * 16 // Port struct
          }
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'storage',
            hasDynamicOffset: false,
            minBindingSize: this.maxPorts * 4 // Connection array
          }
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'storage',
            hasDynamicOffset: false,
            minBindingSize: 8 // Reduction state (step_count, active_pairs)
          }
        }
      ]
    });

    // Create compute pipeline
    const shaderModule = this.device.createShaderModule({
      code: this.getComputeShader(),
      label: 'Interaction Net Reducer'
    });

    this.pipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.bindGroupLayout]
      }),
      compute: {
        module: shaderModule,
        entryPoint: 'main'
      }
    });
  }

  createBuffers(netData) {
    const agentCount = netData.agents.length;
    const portCount = netData.ports.length;

    // Agent buffer
    const agentBuffer = this.device.createBuffer({
      size: Math.max(this.maxAgents * 32, agentCount * 32),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
      label: 'Agent Buffer'
    });

    const agentArray = new Int32Array(agentBuffer.getMappedRange());
    
    // Pack agent data: [id, type, arity, data, principal_port, aux_ports, flags, padding]
    netData.agents.forEach((agent, i) => {
      const offset = i * 8;
      agentArray[offset + 0] = agent.id;
      agentArray[offset + 1] = agent.type;
      agentArray[offset + 2] = agent.arity;
      agentArray[offset + 3] = agent.data || 0;
      agentArray[offset + 4] = agent.principalPort;
      agentArray[offset + 5] = agent.auxPorts;
      agentArray[offset + 6] = 0; // flags (active, deleted, etc.)
      agentArray[offset + 7] = 0; // padding
    });

    agentBuffer.unmap();

    // Port buffer
    const portBuffer = this.device.createBuffer({
      size: Math.max(this.maxPorts * 16, portCount * 16),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
      label: 'Port Buffer'
    });

    const portArray = new Int32Array(portBuffer.getMappedRange());
    
    // Pack port data: [agent_id, type, index, connected_to]
    netData.ports.forEach((port, i) => {
      const offset = i * 4;
      portArray[offset + 0] = port.agentId;
      portArray[offset + 1] = port.type;
      portArray[offset + 2] = port.index;
      portArray[offset + 3] = port.connectedTo;
    });

    portBuffer.unmap();

    // Connection buffer (for fast lookup)
    const connectionBuffer = this.device.createBuffer({
      size: Math.max(this.maxPorts * 4, portCount * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
      label: 'Connection Buffer'
    });

    const connectionArray = new Int32Array(connectionBuffer.getMappedRange());
    
    // Build connection lookup table
    netData.ports.forEach((port, i) => {
      connectionArray[i] = port.connectedTo;
    });

    connectionBuffer.unmap();

    // State buffer
    const stateBuffer = this.device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
      label: 'State Buffer'
    });

    const stateArray = new Int32Array(stateBuffer.getMappedRange());
    stateArray[0] = 0; // step_count
    stateArray[1] = 0; // active_pairs
    stateBuffer.unmap();

    return {
      agentBuffer,
      portBuffer,
      connectionBuffer,
      stateBuffer,
      agentCount,
      portCount
    };
  }

  async reduceNet(netData, maxSteps = 100) {
    if (!this.pipeline) {
      await this.initialize();
    }

    const buffers = this.createBuffers(netData);
    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: buffers.agentBuffer } },
        { binding: 1, resource: { buffer: buffers.portBuffer } },
        { binding: 2, resource: { buffer: buffers.connectionBuffer } },
        { binding: 3, resource: { buffer: buffers.stateBuffer } }
      ]
    });

    let step = 0;
    let hasActivePairs = true;

    while (hasActivePairs && step < maxSteps) {
      // Reset state
      const stateEncoder = this.device.createCommandEncoder();
      const statePass = stateEncoder.beginComputePass();
      
      // Reset step count and active pairs
      // This would normally be done with a separate reset shader
      // For now, we'll do it on CPU between passes
      
      statePass.end();
      this.device.queue.submit([stateEncoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();

      // Execute reduction step
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, bindGroup);
      
      const workgroups = Math.ceil(buffers.agentCount / this.workgroupSize);
      pass.dispatchWorkgroups(workgroups);
      
      pass.end();
      
      this.device.queue.submit([encoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();

      // Check if there are still active pairs
      const stateReadback = this.device.createBuffer({
        size: 8,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      });

      const copyEncoder = this.device.createCommandEncoder();
      copyEncoder.copyBufferToBuffer(buffers.stateBuffer, 0, stateReadback, 0, 8);
      this.device.queue.submit([copyEncoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();

      await stateReadback.mapAsync(GPUMapMode.READ);
      const stateData = new Int32Array(stateReadback.getMappedRange());
      const activePairs = stateData[1];
      stateReadback.unmap();

      hasActivePairs = activePairs > 0;
      step++;
    }

    // Read back results
    const result = await this.readResults(buffers);
    
    // Clean up buffers
    buffers.agentBuffer.destroy();
    buffers.portBuffer.destroy();
    buffers.connectionBuffer.destroy();
    buffers.stateBuffer.destroy();

    return {
      ...result,
      steps: step,
      completed: step < maxSteps
    };
  }

  async readResults(buffers) {
    // Read agent buffer
    const agentReadback = this.device.createBuffer({
      size: buffers.agentCount * 32,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    // Read port buffer
    const portReadback = this.device.createBuffer({
      size: buffers.portCount * 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffers.agentBuffer, 0, agentReadback, 0, buffers.agentCount * 32);
    encoder.copyBufferToBuffer(buffers.portBuffer, 0, portReadback, 0, buffers.portCount * 16);
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    await agentReadback.mapAsync(GPUMapMode.READ);
    await portReadback.mapAsync(GPUMapMode.READ);

    const agentData = new Int32Array(agentReadback.getMappedRange());
    const portData = new Int32Array(portReadback.getMappedRange());

    agentReadback.unmap();
    portReadback.unmap();

    // Convert back to net format
    const agents = [];
    const ports = [];

    for (let i = 0; i < buffers.agentCount; i++) {
      const offset = i * 8;
      const flags = agentData[offset + 6];
      
      // Skip deleted agents
      if (flags & 1) continue;
      
      agents.push({
        id: agentData[offset + 0],
        type: agentData[offset + 1],
        arity: agentData[offset + 2],
        data: agentData[offset + 3],
        principalPort: agentData[offset + 4],
        auxPorts: agentData[offset + 5]
      });
    }

    for (let i = 0; i < buffers.portCount; i++) {
      const offset = i * 4;
      ports.push({
        agentId: portData[offset + 0],
        type: portData[offset + 1],
        index: portData[offset + 2],
        connectedTo: portData[offset + 3]
      });
    }

    return {
      agents,
      ports,
      metadata: {
        agentCount: agents.length,
        portCount: ports.length
      }
    };
  }

  getComputeShader() {
    return `
    // Agent types
    const AGENT_ERA: u32 = 0u;
    const AGENT_CON: u32 = 1u;
    const AGENT_DUP: u32 = 2u;
    const AGENT_APP: u32 = 3u;
    const AGENT_LAM: u32 = 4u;
    const AGENT_SUP: u32 = 5u;
    const AGENT_NUM: u32 = 6u;
    const AGENT_OP2: u32 = 7u;
    const AGENT_ROOT: u32 = 8u;
    const AGENT_WIRE: u32 = 9u;

    // Port types
    const PORT_PRINCIPAL: u32 = 0u;
    const PORT_AUXILIARY: u32 = 1u;

    // Agent flags
    const FLAG_DELETED: u32 = 1u;
    const FLAG_ACTIVE: u32 = 2u;
    const FLAG_PROCESSED: u32 = 4u;

    struct Agent {
      id: u32,
      type: u32,
      arity: u32,
      data: i32,
      principal_port: u32,
      aux_ports: u32,
      flags: u32,
      padding: u32
    };

    struct Port {
      agent_id: u32,
      type: u32,
      index: u32,
      connected_to: i32
    };

    struct ReductionState {
      step_count: u32,
      active_pairs: u32
    };

    @group(0) @binding(0) var<storage, read_write> agents: array<Agent>;
    @group(0) @binding(1) var<storage, read_write> ports: array<Port>;
    @group(0) @binding(2) var<storage, read_write> connections: array<i32>;
    @group(0) @binding(3) var<storage, read_write> state: ReductionState;

    fn get_port(port_index: u32) -> Port {
      return ports[port_index];
    }

    fn is_connected(port: Port) -> bool {
      return port.connected_to >= 0;
    }

    fn get_connected_agent(port: Port) -> Agent {
      let connected_port = ports[u32(port.connected_to)];
      return agents[connected_port.agent_id];
    }

    fn mark_agent_deleted(agent_id: u32) {
      agents[agent_id].flags = agents[agent_id].flags | FLAG_DELETED;
    }

    fn create_result_agent(type: u32, data: i32, port_index: u32) -> u32 {
      // Find a free slot (simplified - in reality would use free list)
      for (var i = 0u; i < arrayLength(&agents); i = i + 1u) {
        if ((agents[i].flags & FLAG_DELETED) != 0u || agents[i].type == AGENT_ERA) {
          agents[i].type = type;
          agents[i].data = data;
          agents[i].flags = 0u;
          return i;
        }
      }
      return 0u; // Failed to create
    }

    fn reduce_lambda_app(lambda_idx: u32, app_idx: u32, lambda_port: u32, app_port: u32) {
      let lambda = agents[lambda_idx];
      let app = agents[app_idx];
      
      // Create result agent (simplified beta reduction)
      let result_idx = create_result_agent(AGENT_CON, lambda.data, lambda.principal_port);
      
      // Connect argument to result if available
      if (app.arity > 0u) {
        let app_aux_port = ports[app.aux_ports];
        if (is_connected(app_aux_port)) {
          ports[lambda.principal_port].connected_to = app_aux_port.connected_to;
        }
      }
      
      // Mark original agents as deleted
      mark_agent_deleted(lambda_idx);
      mark_agent_deleted(app_idx);
    }

    fn reduce_op2_num(op2_idx: u32, num_idx: u32, op2_port: u32, num_port: u32) {
      let op2 = agents[op2_idx];
      let num = agents[num_idx];
      
      var result: i32 = num.data;
      
      // Apply unary operations
      switch (op2.data) {
        case 1: { result = result; break; } // + (no effect)
        case 2: { result = -result; break; } // -
        case 3: { result = 0; break; } // * (needs second operand)
        case 4: { result = 1; break; } // / (needs second operand)
        default: { break; }
      }
      
      // Create result number
      let result_idx = create_result_agent(AGENT_NUM, result, op2.principal_port);
      
      // Connect remaining connections
      if (op2.arity > 1u) {
        let op2_aux = ports[op2.aux_ports + 1u];
        if (is_connected(op2_aux)) {
          ports[op2.principal_port].connected_to = op2_aux.connected_to;
        }
      }
      
      // Mark original agents as deleted
      mark_agent_deleted(op2_idx);
      mark_agent_deleted(num_idx);
    }

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      let idx = global_id.x;
      
      if (idx >= arrayLength(&agents)) {
        return;
      }
      
      let agent = agents[idx];
      
      // Skip deleted or already processed agents
      if ((agent.flags & (FLAG_DELETED | FLAG_PROCESSED)) != 0u) {
        return;
      }
      
      // Check if principal port is connected to another principal port
      let principal_port = ports[agent.principal_port];
      
      if (!is_connected(principal_port)) {
        return;
      }
      
      let connected_port = ports[u32(principal_port.connected_to)];
      
      // Only process principal-to-principal connections
      if (connected_port.type != PORT_PRINCIPAL) {
        return;
      }
      
      let connected_agent = agents[connected_port.agent_id];
      
      // Avoid double processing
      if ((connected_agent.flags & FLAG_PROCESSED) != 0u) {
        return;
      }
      
      // Mark both as processed
      agents[idx].flags = agents[idx].flags | FLAG_PROCESSED;
      agents[connected_port.agent_id].flags = agents[connected_port.agent_id].flags | FLAG_PROCESSED;
      
      // Apply reduction rules
      var reduced = false;
      
      // Lambda-Beta reduction
      if (agent.type == AGENT_LAM && connected_agent.type == AGENT_APP) {
        reduce_lambda_app(idx, connected_port.agent_id, agent.principal_port, connected_port.connected_to);
        reduced = true;
      } else if (agent.type == AGENT_APP && connected_agent.type == AGENT_LAM) {
        reduce_lambda_app(connected_port.agent_id, idx, connected_port.connected_to, agent.principal_port);
        reduced = true;
      }
      
      // Operation with number
      if (agent.type == AGENT_OP2 && connected_agent.type == AGENT_NUM) {
        reduce_op2_num(idx, connected_port.agent_id, agent.principal_port, connected_port.connected_to);
        reduced = true;
      } else if (agent.type == AGENT_NUM && connected_agent.type == AGENT_OP2) {
        reduce_op2_num(connected_port.agent_id, idx, connected_port.connected_to, agent.principal_port);
        reduced = true;
      }
      
      // Count reductions (atomic operation)
      if (reduced) {
        atomicAdd(&state.active_pairs, 1u);
      }
    }
    `;
  }
}

// Export for both Node.js and ES6 modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { GPUNetReducer };
}

// ES6 module exports for browser
export { GPUNetReducer };