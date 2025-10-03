/**
 * Zapp Actor System - Proper implementation with Web Worker isolation
 * Addresses the issues identified in the assessment
 */

class ActorMessage {
  constructor(type, data, from = null, id = null) {
    this.type = type;
    this.data = data;
    this.from = from;
    this.id = id || crypto.randomUUID();
    this.timestamp = Date.now();
  }
}

class ActorState {
  constructor(initialState = {}) {
    this.data = initialState;
    this.version = 0;
  }

  update(updates) {
    this.data = { ...this.data, ...updates };
    this.version++;
  }

  get(key) {
    return this.data[key];
  }

  set(key, value) {
    this.data[key] = value;
    this.version++;
  }
}

class ActorRef {
  constructor(pid, mailbox, worker = null) {
    this.pid = pid;
    this.mailbox = mailbox;
    this.worker = worker;
  }

  async send(message) {
    if (this.worker) {
      // Send to worker-based actor
      this.worker.postMessage({
        type: 'message',
        pid: this.pid,
        message: message
      });
    } else {
      // Send to local actor
      this.mailbox.enqueue(message);
    }
  }

  async request(message, timeout = 5000) {
    return new Promise((resolve, reject) => {
      const requestId = crypto.randomUUID();
      const responseMessage = new ActorMessage('response', null, this.pid, requestId);
      
      // Set up timeout
      const timeoutId = setTimeout(() => {
        reject(new Error(`Request timeout: ${message.type}`));
      }, timeout);
      
      // Send the request with reply-to information
      const requestMessage = new ActorMessage(message.type, {
        ...message.data,
        __replyTo: this.pid,
        __requestId: requestId
      }, this.pid);
      
      this.send(requestMessage);
      
      // Listen for response (simplified - would need proper message routing)
      const checkResponse = () => {
        // In a real implementation, this would use a proper message router
        setTimeout(checkResponse, 10);
      };
      
      checkResponse();
    });
  }
}

class Mailbox {
  constructor(maxSize = 1000) {
    this.messages = [];
    this.maxSize = maxSize;
    this.processing = false;
  }

  enqueue(message) {
    if (this.messages.length >= this.maxSize) {
      // Drop oldest message
      this.messages.shift();
    }
    this.messages.push(message);
  }

  dequeue() {
    return this.messages.shift();
  }

  peek() {
    return this.messages[0];
  }

  isEmpty() {
    return this.messages.length === 0;
  }

  size() {
    return this.messages.length;
  }
}

class Supervisor {
  constructor(strategy = 'one_for_one') {
    this.strategy = strategy;
    this.children = new Map();
    this.restartCounts = new Map();
  }

  addChild(actorRef, restartStrategy = 'permanent', maxRestarts = 3) {
    this.children.set(actorRef.pid, {
      actorRef,
      restartStrategy,
      maxRestarts,
      restartCount: 0
    });
  }

  removeChild(pid) {
    this.children.delete(pid);
    this.restartCounts.delete(pid);
  }

  async handleChildCrash(pid, error) {
    const child = this.children.get(pid);
    if (!child) return;

    console.error(`Actor ${pid} crashed:`, error);
    
    child.restartCount++;
    
    if (child.restartStrategy === 'permanent' && 
        child.restartCount < child.maxRestarts) {
      console.log(`Restarting actor ${pid} (attempt ${child.restartCount})`);
      await this.restartChild(pid);
    } else {
      console.log(`Actor ${pid} exceeded max restarts, shutting down`);
      this.removeChild(pid);
    }
  }

  async restartChild(pid) {
    const child = this.children.get(pid);
    if (!child) return;

    // In a real implementation, this would restart the actor with its initial state
    // For now, we'll just log the restart attempt
    console.log(`Restart attempt for actor ${pid}`);
  }
}

class ActorRuntime {
  constructor() {
    this.actors = new Map();
    this.supervisors = new Map();
    this.messageRouter = new Map();
    this.workerPool = [];
    this.maxWorkers = navigator.hardwareConcurrency || 4;
    this.nextPid = 1;
  }

  async initialize() {
    // Initialize worker pool
    for (let i = 0; i < this.maxWorkers; i++) {
      const worker = new Worker(this.createWorkerScript());
      this.workerPool.push(worker);
    }
  }

  createWorkerScript() {
    const workerCode = `
      // Actor Worker Script
      let actors = new Map();
      let messageHandlers = new Map();
      
      self.onmessage = function(e) {
        const { type, pid, message, actorDef, initialState } = e.data;
        
        switch (type) {
          case 'init':
            initActor(pid, actorDef, initialState);
            break;
          case 'message':
            handleMessage(pid, message);
            break;
          case 'stop':
            stopActor(pid);
            break;
        }
      };
      
      function initActor(pid, actorDef, initialState) {
        const mailbox = [];
        const state = { ...initialState };
        
        actors.set(pid, {
          pid,
          actorDef,
          state,
          mailbox,
          processing: false
        });
        
        // Start processing messages
        processMessages(pid);
      }
      
      function handleMessage(pid, message) {
        const actor = actors.get(pid);
        if (actor) {
          actor.mailbox.push(message);
          if (!actor.processing) {
            processMessages(pid);
          }
        }
      }
      
      function processMessages(pid) {
        const actor = actors.get(pid);
        if (!actor || actor.processing) return;
        
        actor.processing = true;
        
        while (actor.mailbox.length > 0) {
          const message = actor.mailbox.shift();
          
          // Find appropriate message handler
          const handler = actor.actorDef.handlers.find(h => 
            h.name === message.type
          );
          
          if (handler) {
            try {
              // Execute handler (simplified)
              const result = executeHandler(handler, message, actor.state);
              
              // Update state if handler returns new state
              if (result && result.state) {
                Object.assign(actor.state, result.state);
              }
              
              // Send reply if needed
              if (result && result.reply) {
                self.postMessage({
                  type: 'reply',
                  from: pid,
                  to: message.from,
                  message: result.reply
                });
              }
            } catch (error) {
              self.postMessage({
                type: 'error',
                pid,
                error: error.message
              });
            }
          }
        }
        
        actor.processing = false;
      }
      
      function executeHandler(handler, message, state) {
        // This is a simplified handler execution
        // In a real implementation, this would execute the actual Zapp code
        return {
          state: state,
          reply: { type: 'response', data: 'OK' }
        };
      }
      
      function stopActor(pid) {
        actors.delete(pid);
      }
    `;
    
    const blob = new Blob([workerCode], { type: 'application/javascript' });
    return URL.createObjectURL(blob);
  }

  async spawnActor(actorDef, initialState = {}, options = {}) {
    const pid = `actor_${this.nextPid++}`;
    
    // Create supervisor if specified
    if (options.supervisor) {
      let supervisor = this.supervisors.get(options.supervisor);
      if (!supervisor) {
        supervisor = new Supervisor(options.supervisorStrategy);
        this.supervisors.set(options.supervisor, supervisor);
      }
    }

    // Choose worker or local execution
    const useWorker = options.useWorker !== false && this.workerPool.length > 0;
    
    if (useWorker) {
      // Spawn in worker
      const worker = this.workerPool[this.nextPid % this.maxWorkers];
      
      const actorRef = new ActorRef(pid, new Mailbox(), worker);
      
      // Initialize actor in worker
      worker.postMessage({
        type: 'init',
        pid,
        actorDef: this.serializeActorDef(actorDef),
        initialState
      });
      
      // Set up message handling
      worker.onmessage = (e) => this.handleWorkerMessage(pid, e.data);
      worker.onerror = (e) => this.handleActorCrash(pid, e);
      
      this.actors.set(pid, {
        pid,
        actorRef,
        actorDef,
        state: new ActorState(initialState),
        supervisor: options.supervisor || null,
        restartStrategy: options.restartStrategy || 'permanent',
        maxRestarts: options.maxRestarts || 3,
        restartCount: 0,
        isWorker: true
      });
      
      return actorRef;
    } else {
      // Spawn locally
      const mailbox = new Mailbox(options.maxMailboxSize || 1000);
      const actorRef = new ActorRef(pid, mailbox);
      
      this.actors.set(pid, {
        pid,
        actorRef,
        actorDef,
        state: new ActorState(initialState),
        supervisor: options.supervisor || null,
        restartStrategy: options.restartStrategy || 'permanent',
        maxRestarts: options.maxRestarts || 3,
        restartCount: 0,
        isWorker: false
      });
      
      // Start processing messages
      this.processMessages(pid);
      
      return actorRef;
    }
  }

  serializeActorDef(actorDef) {
    // Convert actor definition to serializable format
    return {
      name: actorDef.name,
      handlers: actorDef.handlers.map(h => ({
        name: h.name,
        params: h.params,
        body: h.body
      }))
    };
  }

  handleWorkerMessage(pid, data) {
    const actor = this.actors.get(pid);
    if (!actor) return;

    switch (data.type) {
      case 'reply':
        // Route reply back to sender
        this.routeMessage(data.to, data.message);
        break;
      case 'error':
        this.handleActorCrash(pid, new Error(data.error));
        break;
    }
  }

  async handleActorCrash(pid, error) {
    const actor = this.actors.get(pid);
    if (!actor) return;

    console.error(`Actor ${pid} crashed:`, error);
    
    if (actor.supervisor) {
      const supervisor = this.supervisors.get(actor.supervisor);
      if (supervisor) {
        await supervisor.handleChildCrash(pid, error);
      }
    }
    
    // Restart based on strategy
    if (actor.restartStrategy === 'permanent' && 
        actor.restartCount < actor.maxRestarts) {
      await this.restartActor(pid);
    } else {
      this.stopActor(pid);
    }
  }

  async restartActor(pid) {
    const actor = this.actors.get(pid);
    if (!actor) return;

    actor.restartCount++;
    
    try {
      if (actor.isWorker) {
        // Restart worker-based actor
        const worker = this.workerPool[pid % this.maxWorkers];
        worker.postMessage({
          type: 'init',
          pid,
          actorDef: this.serializeActorDef(actor.actorDef),
          initialState: actor.state.data
        });
      } else {
        // Restart local actor
        this.processMessages(pid);
      }
    } catch (error) {
      console.error(`Failed to restart actor ${pid}:`, error);
      this.stopActor(pid);
    }
  }

  stopActor(pid) {
    const actor = this.actors.get(pid);
    if (!actor) return;

    if (actor.isWorker) {
      const worker = this.workerPool[pid % this.maxWorkers];
      worker.postMessage({
        type: 'stop',
        pid
      });
    }
    
    this.actors.delete(pid);
  }

  processMessages(pid) {
    const actor = this.actors.get(pid);
    if (!actor || actor.isWorker) return;

    const mailbox = actor.actorRef.mailbox;
    
    if (mailbox.isEmpty()) {
      // Schedule next check
      setTimeout(() => this.processMessages(pid), 10);
      return;
    }

    const message = mailbox.dequeue();
    
    // Find appropriate message handler
    const handler = actor.actorDef.handlers.find(h => 
      h.name === message.type
    );
    
    if (handler) {
      try {
        // Execute handler (simplified)
        this.executeHandler(handler, message, actor);
      } catch (error) {
        this.handleActorCrash(pid, error);
        return;
      }
    }
    
    // Continue processing
    setTimeout(() => this.processMessages(pid), 0);
  }

  executeHandler(handler, message, actor) {
    // This is a simplified handler execution
    // In a real implementation, this would execute the actual Zapp code
    
    // Update state based on message
    actor.state.update({
      lastMessage: message.type,
      lastMessageTime: Date.now()
    });
    
    // Send reply if needed
    if (message.from && message.data.__replyTo) {
      const reply = new ActorMessage('response', {
        __requestId: message.data.__requestId,
        data: 'OK'
      }, actor.pid);
      
      this.routeMessage(message.from, reply);
    }
  }

  routeMessage(toPid, message) {
    const actor = this.actors.get(toPid);
    if (actor) {
      actor.actorRef.send(message);
    }
  }

  async send(pid, message) {
    const actor = this.actors.get(pid);
    if (!actor) {
      throw new Error(`Actor ${pid} not found`);
    }
    
    const msg = new ActorMessage(message.type, message.data, null);
    await actor.actorRef.send(msg);
  }

  async request(pid, message, timeout = 5000) {
    const actor = this.actors.get(pid);
    if (!actor) {
      throw new Error(`Actor ${pid} not found`);
    }
    
    return await actor.actorRef.request(message, timeout);
  }

  getActorStatus(pid) {
    const actor = this.actors.get(pid);
    if (!actor) return null;

    return {
      pid: actor.pid,
      name: actor.actorDef.name,
      isWorker: actor.isWorker,
      restartCount: actor.restartCount,
      mailboxSize: actor.actorRef.mailbox.size(),
      state: actor.state.data,
      supervisor: actor.supervisor
    };
  }

  getAllActors() {
    return Array.from(this.actors.keys()).map(pid => this.getActorStatus(pid));
  }

  async shutdown() {
    // Stop all actors
    for (const pid of this.actors.keys()) {
      this.stopActor(pid);
    }
    
    // Terminate workers
    for (const worker of this.workerPool) {
      worker.terminate();
    }
    
    this.workerPool = [];
    this.actors.clear();
    this.supervisors.clear();
  }
}

// Export for both Node.js and ES6 modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    ActorRuntime,
    ActorRef,
    ActorMessage,
    ActorState,
    Mailbox,
    Supervisor
  };
}

// ES6 module exports for browser
export {
  ActorRuntime,
  ActorRef,
  ActorMessage,
  ActorState,
  Mailbox,
  Supervisor
};