/**
 * WebGPU Runtime - Fixed buffer management and synchronization
 * Addresses the issues identified in the assessment
 */

class GPUResourceManager {
  constructor(maxMemory = 512 * 1024 * 1024) { // 512MB default
    this.maxMemory = maxMemory;
    this.allocatedMemory = 0;
    this.buffers = new Map();
    this.textures = new Map();
    this.samplers = new Map();
  }
  
  createBuffer(device, size, usage, label = 'Buffer') {
    if (this.allocatedMemory + size > this.maxMemory) {
      throw new Error(`GPU memory limit exceeded: ${this.allocatedMemory + size} > ${this.maxMemory}`);
    }
    
    const buffer = device.createBuffer({ size, usage, label });
    this.buffers.set(buffer, { size, label, usage });
    this.allocatedMemory += size;
    
    return buffer;
  }
  
  destroyBuffer(buffer) {
    const info = this.buffers.get(buffer);
    if (info) {
      this.allocatedMemory -= info.size;
      this.buffers.delete(buffer);
      buffer.destroy();
    }
  }
  
  getMemoryUsage() {
    return {
      allocated: this.allocatedMemory,
      max: this.maxMemory,
      buffers: this.buffers.size,
      utilization: (this.allocatedMemory / this.maxMemory) * 100
    };
  }
}

class MappedBufferManager {
  constructor(device) {
    this.device = device;
    this.mappedBuffers = new Map();
  }
  
  createMappedBuffer(size, usage, label = 'MappedBuffer') {
    const buffer = this.device.createBuffer({
      size,
      usage,
      mappedAtCreation: true,
      label
    });
    
    const mapping = new Uint8Array(buffer.getMappedRange());
    const bufferInfo = {
      buffer,
      mapping,
      size,
      label,
      isMapped: true
    };
    
    this.mappedBuffers.set(buffer, bufferInfo);
    return bufferInfo;
  }
  
  unmapBuffer(buffer) {
    const bufferInfo = this.mappedBuffers.get(buffer);
    if (bufferInfo && bufferInfo.isMapped) {
      buffer.unmap();
      bufferInfo.isMapped = false;
      bufferInfo.mapping = null;
      return bufferInfo;
    }
    return null;
  }
  
  ensureUnmapped(buffer) {
    const bufferInfo = this.mappedBuffers.get(buffer);
    if (bufferInfo && bufferInfo.isMapped) {
      this.unmapBuffer(buffer);
    }
  }
  
  cleanup() {
    for (const [buffer, bufferInfo] of this.mappedBuffers) {
      if (bufferInfo.isMapped) {
        buffer.unmap();
      }
    }
    this.mappedBuffers.clear();
  }
}

class WebGPURuntime {
  constructor() {
    this.device = null;
    this.adapter = null;
    this.queue = null;
    this.resourceManager = new GPUResourceManager();
    this.mappedBufferManager = null;
    this.commandEncoder = null;
    this.computePasses = [];
  }

  async initialize() {
    if (!navigator.gpu) {
      throw new Error('WebGPU not supported in this browser');
    }

    try {
      this.adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance'
      });

      if (!this.adapter) {
        throw new Error('Failed to get GPU adapter');
      }

      this.device = await this.adapter.requestDevice({
        requiredFeatures: ['shader-f16'],
        requiredLimits: {
          maxStorageBufferBindingSize: 1024 * 1024 * 1024, // 1GB
          maxComputeWorkgroupStorageSize: 16384,
          maxComputeInvocationsPerWorkgroup: 256,
          maxComputeWorkgroupsPerDimension: 65535
        }
      });

      this.queue = this.device.queue;
      this.mappedBufferManager = new MappedBufferManager(this.device);

      // Handle device loss
      this.device.lost.then((info) => {
        console.error('WebGPU device lost:', info);
        this.cleanup();
      });

      return this;
    } catch (error) {
      throw new Error(`WebGPU initialization failed: ${error.message}`);
    }
  }

  createBuffer(size, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label = 'Buffer') {
    return this.resourceManager.createBuffer(this.device, size, usage, label);
  }

  createMappedBuffer(size, usage, label = 'MappedBuffer') {
    return this.mappedBufferManager.createMappedBuffer(size, usage, label);
  }

  createComputePipeline(shaderCode, entryPoint = 'main', label = 'ComputePipeline') {
    const shaderModule = this.device.createShaderModule({
      code: shaderCode,
      label: `${label} Shader`
    });

    // Check compilation errors
    const compilationInfo = shaderModule.getCompilationInfo();
    if (compilationInfo.messages.length > 0) {
      for (const msg of compilationInfo.messages) {
        if (msg.type === 'error') {
          throw new Error(`Shader compilation error at line ${msg.lineNum}: ${msg.message}`);
        }
      }
    }

    return this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint
      },
      label
    });
  }

  createBindGroup(layout, entries, label = 'BindGroup') {
    return this.device.createBindGroup({
      layout,
      entries,
      label
    });
  }

  beginComputePass(label = 'ComputePass') {
    if (this.commandEncoder) {
      const passEncoder = this.commandEncoder.beginComputePass({ label });
      this.computePasses.push(passEncoder);
      return passEncoder;
    }
    throw new Error('No command encoder active. Call beginCommandEncoder() first.');
  }

  beginCommandEncoder(label = 'CommandEncoder') {
    if (this.commandEncoder) {
      throw new Error('Command encoder already active. Call endCommandEncoder() first.');
    }
    this.commandEncoder = this.device.createCommandEncoder({ label });
    this.computePasses = [];
  }

  endCommandEncoder() {
    if (!this.commandEncoder) {
      throw new Error('No command encoder active');
    }

    // End all active compute passes
    for (const passEncoder of this.computePasses) {
      passEncoder.end();
    }
    this.computePasses = [];

    const commandBuffer = this.commandEncoder.finish();
    this.commandEncoder = null;
    return commandBuffer;
  }

  async executeComputePasses(pipelines, bindGroups, workgroupCounts) {
    if (!Array.isArray(pipelines)) {
      pipelines = [pipelines];
    }
    if (!Array.isArray(bindGroups)) {
      bindGroups = [bindGroups];
    }
    if (!Array.isArray(workgroupCounts)) {
      workgroupCounts = [workgroupCounts];
    }

    if (pipelines.length !== bindGroups.length || pipelines.length !== workgroupCounts.length) {
      throw new Error('Arrays must have the same length');
    }

    this.beginCommandEncoder('MultiPass Compute');

    // Create all compute passes upfront for better parallelism
    for (let i = 0; i < pipelines.length; i++) {
      const passEncoder = this.beginComputePass(`Compute Pass ${i}`);
      
      passEncoder.setPipeline(pipelines[i]);
      passEncoder.setBindGroup(0, bindGroups[i]);
      
      const workgroupCount = workgroupCounts[i];
      passEncoder.dispatchWorkgroups(
        workgroupCount.x || 1,
        workgroupCount.y || 1,
        workgroupCount.z || 1
      );
    }

    const commandBuffer = this.endCommandEncoder();
    
    // Submit all passes at once
    this.queue.submit([commandBuffer]);
    
    // Wait for completion
    await this.queue.onSubmittedWorkDone();
  }

  async executeMultiPassReduction(compiled, maxPasses = 50) {
    const commandEncoders = [];
    
    // Analyze reduction graph to determine dependencies
    const passes = this.analyzeReductionPasses(compiled);
    
    // Create all command buffers upfront
    for (let i = 0; i < Math.min(passes.length, maxPasses); i++) {
      this.beginCommandEncoder(`Reduction Pass ${i}`);
      
      const passEncoder = this.beginComputePass(`Compute Pass ${i}`);
      passEncoder.setPipeline(compiled.pipeline);
      passEncoder.setBindGroup(0, compiled.bindGroup);
      
      const workgroups = Math.ceil(compiled.nodeCount / 256);
      passEncoder.dispatchWorkgroups(workgroups, 1, 1);
      
      const commandBuffer = this.endCommandEncoder();
      commandEncoders.push(commandBuffer);
    }
    
    // Submit all at once - GPU scheduler handles dependencies
    this.queue.submit(commandEncoders);
    await this.queue.onSubmittedWorkDone();
    
    return commandEncoders.length;
  }

  analyzeReductionPasses(compiled) {
    // This is a simplified analysis - a full implementation would
    // analyze the dependency graph of the reduction
    const passes = [];
    
    for (let i = 0; i < compiled.nodeCount; i += 256) {
      passes.push({
        startNode: i,
        endNode: Math.min(i + 256, compiled.nodeCount),
        workgroups: 1
      });
    }
    
    return passes;
  }

  async readBuffer(buffer, size) {
    const stagingBuffer = this.createBuffer(
      size,
      GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      'Staging Buffer'
    );

    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, size);
    
    const commandBuffer = encoder.finish();
    this.queue.submit([commandBuffer]);
    await this.queue.onSubmittedWorkDone();

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const data = stagingBuffer.getMappedRange();
    
    // Create a copy of the data
    const result = new Uint8Array(data);
    
    stagingBuffer.unmap();
    this.resourceManager.destroyBuffer(stagingBuffer);
    
    return result;
  }

  async readBufferAsInt32(buffer, size) {
    const data = await this.readBuffer(buffer, size);
    return new Int32Array(data.buffer, data.byteOffset, data.byteLength / 4);
  }

  async readBufferAsFloat32(buffer, size) {
    const data = await this.readBuffer(buffer, size);
    return new Float32Array(data.buffer, data.byteOffset, data.byteLength / 4);
  }

  createStagingBuffer(size) {
    return this.createBuffer(
      size,
      GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      'Staging Buffer'
    );
  }

  async copyBufferToStaging(sourceBuffer, stagingBuffer, size) {
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(sourceBuffer, 0, stagingBuffer, 0, size);
    
    const commandBuffer = encoder.finish();
    this.queue.submit([commandBuffer]);
    await this.queue.onSubmittedWorkDone();
  }

  getMemoryUsage() {
    return this.resourceManager.getMemoryUsage();
  }

  cleanup() {
    // Clean up mapped buffers
    if (this.mappedBufferManager) {
      this.mappedBufferManager.cleanup();
    }
    
    // Clean up all resources
    this.resourceManager = new GPUResourceManager();
    
    // Clear references
    this.device = null;
    this.adapter = null;
    this.queue = null;
    this.commandEncoder = null;
    this.computePasses = [];
  }
}

// Export for both Node.js and ES6 modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    WebGPURuntime,
    GPUResourceManager,
    MappedBufferManager
  };
}

// ES6 module exports for browser
export {
  WebGPURuntime,
  GPUResourceManager,
  MappedBufferManager
};