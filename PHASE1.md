# Zapp Phase 1: Core Language Foundation

## Goal
Establish a stable lambda calculus interpreter with atoms, result types, arithmetic, and I/O. Everything designed for GPU parallelism from day one.

## Prerequisites
- Existing interaction net system (✓)
- Existing parser for lambda expressions (✓)
- Basic arithmetic operations (✓)

---

## Tasks

### 1.1 Fix Current Lambda Calculus Issues ⚠️ CRITICAL

**Priority: Fix the infinite loop bug first!**

Current issue: `(λf.λx.f (f x)) (λy.y + 1) 3` causes infinite loop

**What to fix:**
- Lambda reference resolution in arithmetic expressions
- Nested lambda applications
- Function composition
- Currying chains

**Acceptance Criteria:**
```zapp
(λx.x) 5                              # => 5
(λx.x + 1) 5                          # => 6
(λf.λx.f x) (λy.y + 1) 5             # => 6
(λf.λx.f (f x)) (λy.y + 1) 3         # => 5
```

---

### 1.2 Add Atom Type

Replace booleans with atoms for Elixir compatibility and GPU efficiency.

```javascript
const AgentType = {
  // ... existing ...
  ATOM: 20,     // Atom literal (interned symbol)
};

class AtomRegistry {
  constructor() {
    this.atoms = new Map();      // symbol -> id
    this.reverse = new Map();    // id -> symbol
    this.nextId = 0;
    
    // Pre-register common atoms
    this.intern('true');
    this.intern('false');
    this.intern('ok');
    this.intern('error');
    this.intern('nil');
  }
  
  intern(symbol) {
    if (!this.atoms.has(symbol)) {
      this.atoms.set(symbol, this.nextId);
      this.reverse.set(this.nextId, symbol);
      this.nextId++;
    }
    return this.atoms.get(symbol);
  }
  
  lookup(id) {
    return this.reverse.get(id);
  }
}

// In InteractionNet:
class InteractionNet {
  constructor() {
    // ... existing ...
    this.atomRegistry = new AtomRegistry();
  }
  
  createAtom(symbol) {
    const atomId = this.atomRegistry.intern(symbol);
    return this.createAgent(AgentType.ATOM, 0, atomId);
  }
}
```

**GPU Encoding:**
```c
// Atoms are just integers on GPU
struct Atom {
  uint32_t type;      // AgentType.ATOM
  uint32_t atom_id;   // Index into atom table (0=:true, 1=:false, etc.)
  uint32_t port_data;
};

// Atom table in constant memory (fast!)
__constant__ char atom_table[MAX_ATOMS][MAX_ATOM_LENGTH];
```

**Parser Support:**
```javascript
// In tokenizer:
if (sourceCode[i] === ':') {
  i++;
  let symbol = '';
  
  if (sourceCode[i] === '"') {
    // Quoted atom: :"multi word"
    i++;
    while (sourceCode[i] !== '"') {
      symbol += sourceCode[i++];
    }
    i++;
  } else {
    // Unquoted atom: :true
    while (/[a-zA-Z0-9_]/.test(sourceCode[i])) {
      symbol += sourceCode[i++];
    }
  }
  
  tokens.push({ type: 'ATOM', value: symbol });
}
```

**Syntax:**
```zapp
:true
:false
:ok
:error
:my_custom_atom
:"atoms can have spaces"
```

---

### 1.3 Add Comparison Operators

Comparisons return atoms (`:true` or `:false`), not booleans.

```javascript
// In InteractionNet:
reduceComparison(op2Agent, leftAgent, rightAgent) {
  const operator = op2Agent.data;
  let result;
  
  if (leftAgent.type === AgentType.ATOM && rightAgent.type === AgentType.ATOM) {
    // Atom comparison
    switch(operator) {
      case 5:  // ==
        result = leftAgent.data === rightAgent.data;
        break;
      case 6:  // !=
        result = leftAgent.data !== rightAgent.data;
        break;
      default:
        throw new Error(`Cannot compare atoms with operator ${operator}`);
    }
  } else if (leftAgent.type === AgentType.NUM && rightAgent.type === AgentType.NUM) {
    // Number comparison
    switch(operator) {
      case 5:  result = leftAgent.data === rightAgent.data; break;  // ==
      case 6:  result = leftAgent.data !== rightAgent.data; break;  // !=
      case 7:  result = leftAgent.data < rightAgent.data; break;    // <
      case 8:  result = leftAgent.data > rightAgent.data; break;    // >
      case 9:  result = leftAgent.data <= rightAgent.data; break;   // <=
      case 10: result = leftAgent.data >= rightAgent.data; break;   // >=
    }
  } else {
    throw new Error('Type mismatch in comparison');
  }
  
  // Return atom :true or :false
  const atomId = this.atomRegistry.intern(result ? 'true' : 'false');
  const boolAgent = this.createAgent(AgentType.ATOM, 0, atomId);
  
  // Connect to whatever op2 was connected to
  if (op2Agent.principalPort.isConnected()) {
    const target = op2Agent.principalPort.connectedTo;
    boolAgent.principalPort.connect(target);
  }
  
  // Cleanup
  this.removeAgent(op2Agent.id);
  this.removeAgent(leftAgent.id);
  this.removeAgent(rightAgent.id);
  
  return boolAgent;
}
```

**Operator codes:**
```javascript
const OperatorCodes = {
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
```

---

### 1.4 Add If-Then-Else

If expects an atom. Truthy atoms: everything except `:false` and `:nil`.

```javascript
const AgentType = {
  // ... existing ...
  IF: 25,       // Conditional
};

// In InteractionNet:
reduceIf(ifAgent, condAgent) {
  if (condAgent.type !== AgentType.ATOM) {
    throw new Error('If condition must be an atom');
  }
  
  const condSymbol = this.atomRegistry.lookup(condAgent.data);
  
  // Truthy: everything except :false and :nil
  const isTruthy = condSymbol !== 'false' && condSymbol !== 'nil';
  
  const thenBranch = ifAgent.auxiliaryPorts[0].getConnectedAgent();
  const elseBranch = ifAgent.auxiliaryPorts[1].getConnectedAgent();
  
  const chosen = isTruthy ? thenBranch : elseBranch;
  const discarded = isTruthy ? elseBranch : thenBranch;
  
  // Remove discarded branch (entire subgraph)
  this.removeSubgraph(discarded);
  
  // Connect chosen branch to if's principal port
  if (ifAgent.principalPort.isConnected()) {
    const target = ifAgent.principalPort.connectedTo;
    chosen.principalPort.connect(target);
  }
  
  // Cleanup
  this.removeAgent(ifAgent.id);
  this.removeAgent(condAgent.id);
  
  return chosen;
}

removeSubgraph(rootAgent) {
  if (!rootAgent) return;
  
  const visited = new Set();
  const toRemove = [rootAgent];
  
  while (toRemove.length > 0) {
    const agent = toRemove.pop();
    if (visited.has(agent.id)) continue;
    visited.add(agent.id);
    
    // Add all connected agents
    agent.getAllPorts().forEach(port => {
      if (port.isConnected()) {
        const connected = port.getConnectedAgent();
        if (connected && !visited.has(connected.id)) {
          toRemove.push(connected);
        }
      }
    });
    
    this.removeAgent(agent.id);
  }
}
```

**Parser:**
```javascript
// In NetParser:
parseIf(tokens, startIndex) {
  let i = startIndex + 1; // Skip 'if'
  
  // Parse condition
  const thenPos = this.findKeyword(tokens, i, 'then');
  const condTokens = tokens.slice(i, thenPos);
  
  const beforeCond = this.net.agents.size;
  this.parseTokens(condTokens);
  const condAgents = Array.from(this.net.agents.values()).slice(beforeCond);
  const condAgent = condAgents[condAgents.length - 1];
  
  i = thenPos + 1; // Skip 'then'
  
  // Parse then branch
  const elsePos = this.findKeyword(tokens, i, 'else');
  const thenTokens = tokens.slice(i, elsePos);
  
  const beforeThen = this.net.agents.size;
  this.parseTokens(thenTokens);
  const thenAgents = Array.from(this.net.agents.values()).slice(beforeThen);
  const thenAgent = thenAgents[thenAgents.length - 1];
  
  i = elsePos + 1; // Skip 'else'
  
  // Parse else branch
  const endPos = this.findKeyword(tokens, i, 'end');
  const elseTokens = tokens.slice(i, endPos);
  
  const beforeElse = this.net.agents.size;
  this.parseTokens(elseTokens);
  const elseAgents = Array.from(this.net.agents.values()).slice(beforeElse);
  const elseAgent = elseAgents[elseAgents.length - 1];
  
  // Create IF agent
  const ifAgent = this.net.createAgent(AgentType.IF, 3);
  
  // Connect: port 0 = condition, port 1 = then, port 2 = else
  this.net.connectPorts(ifAgent.auxiliaryPorts[0], condAgent.principalPort);
  this.net.connectPorts(ifAgent.auxiliaryPorts[1], thenAgent.principalPort);
  this.net.connectPorts(ifAgent.auxiliaryPorts[2], elseAgent.principalPort);
  
  return endPos + 1;
}
```

**Syntax:**
```zapp
if :true then
  "yes"
else
  "no"
end                           # => "yes"

if x > 5 then
  "big"
else
  "small"
end

# Nested
if x > 10 then
  "huge"
else
  if x > 5 then
    "big"
  else
    "small"
  end
end
```

---

### 1.5 Add Logical Operators

```javascript
const AgentType = {
  // ... existing ...
  AND: 26,      // Logical AND
  OR: 27,       // Logical OR
  NOT: 28,      // Logical NOT
};

reduceAnd(andAgent, leftAgent, rightAgent) {
  if (leftAgent.type !== AgentType.ATOM || rightAgent.type !== AgentType.ATOM) {
    throw new Error('AND requires atom operands');
  }
  
  const left = this.atomRegistry.lookup(leftAgent.data);
  const right = this.atomRegistry.lookup(rightAgent.data);
  
  const leftTruthy = left !== 'false' && left !== 'nil';
  const rightTruthy = right !== 'false' && right !== 'nil';
  
  const result = leftTruthy && rightTruthy;
  const atomId = this.atomRegistry.intern(result ? 'true' : 'false');
  
  const resultAgent = this.createAgent(AgentType.ATOM, 0, atomId);
  
  // Cleanup
  this.removeAgent(andAgent.id);
  this.removeAgent(leftAgent.id);
  this.removeAgent(rightAgent.id);
  
  return resultAgent;
}

reduceOr(orAgent, leftAgent, rightAgent) {
  if (leftAgent.type !== AgentType.ATOM || rightAgent.type !== AgentType.ATOM) {
    throw new Error('OR requires atom operands');
  }
  
  const left = this.atomRegistry.lookup(leftAgent.data);
  const right = this.atomRegistry.lookup(rightAgent.data);
  
  const leftTruthy = left !== 'false' && left !== 'nil';
  const rightTruthy = right !== 'false' && right !== 'nil';
  
  const result = leftTruthy || rightTruthy;
  const atomId = this.atomRegistry.intern(result ? 'true' : 'false');
  
  const resultAgent = this.createAgent(AgentType.ATOM, 0, atomId);
  
  // Cleanup
  this.removeAgent(orAgent.id);
  this.removeAgent(leftAgent.id);
  this.removeAgent(rightAgent.id);
  
  return resultAgent;
}

reduceNot(notAgent, atomAgent) {
  if (atomAgent.type !== AgentType.ATOM) {
    throw new Error('NOT requires atom operand');
  }
  
  const symbol = this.atomRegistry.lookup(atomAgent.data);
  const isTruthy = symbol !== 'false' && symbol !== 'nil';
  
  const result = !isTruthy;
  const atomId = this.atomRegistry.intern(result ? 'true' : 'false');
  
  const resultAgent = this.createAgent(AgentType.ATOM, 0, atomId);
  
  // Cleanup
  this.removeAgent(notAgent.id);
  this.removeAgent(atomAgent.id);
  
  return resultAgent;
}
```

**Syntax:**
```zapp
:true and :false          # => :false
:true or :false           # => :true
not :true                 # => :false
not :nil                  # => :true

x > 5 and y < 10
x == 0 or y == 0
not (x > 100)
```

---

### 1.6 Add Result Types

Result types are just tagged tuples with atoms.

```javascript
// Result types are trees!
// {:ok, value} = Node(Atom(:ok), value)
// {:error, reason} = Node(Atom(:error), reason)

// Helper methods in InteractionNet:
createOk(valueAgent) {
  const okAtom = this.createAtom('ok');
  const node = this.createAgent(AgentType.NODE, 2);
  this.connectPorts(node.auxiliaryPorts[0], okAtom.principalPort);
  this.connectPorts(node.auxiliaryPorts[1], valueAgent.principalPort);
  return node;
}

createError(reasonAgent) {
  const errorAtom = this.createAtom('error');
  const node = this.createAgent(AgentType.NODE, 2);
  this.connectPorts(node.auxiliaryPorts[0], errorAtom.principalPort);
  this.connectPorts(node.auxiliaryPorts[1], reasonAgent.principalPort);
  return node;
}

isOk(resultAgent) {
  if (resultAgent.type !== AgentType.NODE) return false;
  
  const tag = resultAgent.auxiliaryPorts[0].getConnectedAgent();
  if (tag.type !== AgentType.ATOM) return false;
  
  const symbol = this.atomRegistry.lookup(tag.data);
  return symbol === 'ok';
}

isError(resultAgent) {
  if (resultAgent.type !== AgentType.NODE) return false;
  
  const tag = resultAgent.auxiliaryPorts[0].getConnectedAgent();
  if (tag.type !== AgentType.ATOM) return false;
  
  const symbol = this.atomRegistry.lookup(tag.data);
  return symbol === 'error';
}
```

**Syntax:**
```zapp
{:ok, 42}
{:error, "something went wrong"}

# Functions return results
def divide(a, b) =
  if b == 0 then
    {:error, "division by zero"}
  else
    {:ok, a / b}
  end
```

---

### 1.7 Add Print (I/O)

```javascript
const AgentType = {
  // ... existing ...
  PRINT: 30,    // Output operation
  UNIT: 31,     // Unit/void type (like :ok with no value)
};

reducePrint(printAgent, valueAgent) {
  console.log(`=== Print ===`);
  
  // Evaluate value to printable form
  const value = this.evaluateToValue(valueAgent);
  
  // Perform side effect
  console.log(`OUTPUT: ${value}`);
  
  // Return unit (represented as :ok atom)
  const unitAgent = this.createAtom('ok');
  
  // Cleanup
  this.removeAgent(printAgent.id);
  this.removeAgent(valueAgent.id);
  
  return unitAgent;
}

evaluateToValue(agent) {
  if (agent.type === AgentType.NUM) {
    return agent.data;
  } else if (agent.type === AgentType.ATOM) {
    return ':' + this.atomRegistry.lookup(agent.data);
  } else if (agent.type === AgentType.STR) {
    return agent.data;
  } else if (agent.type === AgentType.NODE) {
    // Try to evaluate as result type
    const tag = agent.auxiliaryPorts[0].getConnectedAgent();
    const value = agent.auxiliaryPorts[1].getConnectedAgent();
    
    if (tag.type === AgentType.ATOM) {
      const tagSymbol = this.atomRegistry.lookup(tag.data);
      const valueStr = this.evaluateToValue(value);
      return `{:${tagSymbol}, ${valueStr}}`;
    }
  }
  
  return `<agent ${agent.id}>`;
}
```

**Syntax:**
```zapp
print("Hello, world!")
print(42)
print(:ok)
print({:ok, 5})
```

---

### 1.8 Add String Type

```javascript
const AgentType = {
  // ... existing ...
  STR: 32,      // String literal
};

// In tokenizer:
if (sourceCode[i] === '"') {
  i++;
  let str = '';
  while (i < sourceCode.length && sourceCode[i] !== '"') {
    if (sourceCode[i] === '\\') {
      // Escape sequences
      i++;
      switch(sourceCode[i]) {
        case 'n': str += '\n'; break;
        case 't': str += '\t'; break;
        case '\\': str += '\\'; break;
        case '"': str += '"'; break;
        default: str += sourceCode[i];
      }
      i++;
    } else {
      str += sourceCode[i++];
    }
  }
  i++; // Skip closing "
  
  tokens.push({ type: 'STRING', value: str });
}
```

**Syntax:**
```zapp
"hello"
"multi\nline"
"escaped \"quotes\""
```

---

### 1.9 Add Let Bindings

```javascript
const AgentType = {
  // ... existing ...
  LET: 33,      // Let binding
};

reduceLet(letAgent, valueAgent) {
  const varName = letAgent.data.name;
  const bodyAgent = letAgent.auxiliaryPorts[0].getConnectedAgent();
  
  console.log(`=== Let binding: ${varName} ===`);
  
  // Substitute varName with valueAgent in bodyAgent
  this.substituteInSubgraph(bodyAgent, varName, valueAgent);
  
  // Remove let agent
  this.removeAgent(letAgent.id);
  
  return bodyAgent;
}

substituteInSubgraph(rootAgent, varName, valueAgent) {
  const visited = new Set();
  const toVisit = [rootAgent];
  
  while (toVisit.length > 0) {
    const agent = toVisit.pop();
    if (visited.has(agent.id)) continue;
    visited.add(agent.id);
    
    // If this is a variable reference, replace it
    if (agent.type === AgentType.CON && agent.data === varName) {
      // Clone value and replace this reference
      const valueClone = this.cloneAgentForSubstitution(valueAgent);
      
      // Transfer connections
      agent.getAllPorts().forEach((port, i) => {
        if (port.isConnected()) {
          const target = port.connectedTo;
          port.disconnect();
          
          if (i === 0) {
            valueClone.principalPort.connect(target);
          } else if (valueClone.auxiliaryPorts[i - 1]) {
            valueClone.auxiliaryPorts[i - 1].connect(target);
          }
        }
      });
      
      this.removeAgent(agent.id);
    }
    
    // Visit connected agents
    agent.getAllPorts().forEach(port => {
      if (port.isConnected()) {
        const connected = port.getConnectedAgent();
        if (connected && !visited.has(connected.id)) {
          toVisit.push(connected);
        }
      }
    });
  }
}
```

**Parser:**
```javascript
parseLet(tokens, startIndex) {
  let i = startIndex + 1; // Skip 'let'
  
  // Parse variable name
  if (tokens[i].type !== 'IDENTIFIER') {
    throw new Error('Expected identifier after let');
  }
  const varName = tokens[i].value;
  i++;
  
  // Expect '='
  if (tokens[i].value !== '=') {
    throw new Error('Expected = after variable name');
  }
  i++;
  
  // Parse value expression
  const inPos = this.findKeyword(tokens, i, 'in');
  const valueTokens = tokens.slice(i, inPos);
  
  const beforeValue = this.net.agents.size;
  this.parseTokens(valueTokens);
  const valueAgents = Array.from(this.net.agents.values()).slice(beforeValue);
  const valueAgent = valueAgents[valueAgents.length - 1];
  
  i = inPos + 1; // Skip 'in'
  
  // Parse body expression
  const bodyEnd = this.findExpressionEnd(tokens, i);
  const bodyTokens = tokens.slice(i, bodyEnd);
  
  const beforeBody = this.net.agents.size;
  this.parseTokens(bodyTokens);
  const bodyAgents = Array.from(this.net.agents.values()).slice(beforeBody);
  const bodyAgent = bodyAgents[bodyAgents.length - 1];
  
  // Create LET agent
  const letAgent = this.net.createAgent(AgentType.LET, 2);
  letAgent.data = { name: varName };
  
  this.net.connectPorts(letAgent.auxiliaryPorts[0], bodyAgent.principalPort);
  this.net.connectPorts(letAgent.auxiliaryPorts[1], valueAgent.principalPort);
  
  return bodyEnd;
}
```

**Syntax:**
```zapp
let x = 5 in x + 3                    # => 8
let x = 5 in let y = 3 in x + y       # => 8
let f = λx.x * 2 in f(5)              # => 10
```

---

## Testing Strategy

### Test Suite 1.1: Lambda Calculus (CRITICAL)
```zapp
# Identity
(λx.x) 42                             # => 42

# Application
(λx.x + 1) 5                          # => 6

# Currying
(λx.λy.x + y) 3 4                     # => 7

# Function composition (the bug!)
(λf.λx.f (f x)) (λy.y * 2) 3          # => 12
(λf.λx.f (f x)) (λy.y + 1) 3          # => 5
```

### Test Suite 1.2: Atoms
```zapp
:true                                  # => :true
:false                                 # => :false
:ok                                    # => :ok
:error                                 # => :error
:my_atom                               # => :my_atom
:"multi word atom"                     # => :"multi word atom"
```

### Test Suite 1.3: Comparisons
```zapp
5 == 5                                 # => :true
5 == 6                                 # => :false
5 < 10                                 # => :true
10 > 5                                 # => :true
:ok == :ok                             # => :true
:ok == :error                          # => :false
```

### Test Suite 1.4: If-Then-Else
```zapp
if :true then 1 else 0 end             # => 1
if :false then 1 else 0 end            # => 0
if :ok then "yes" else "no" end        # => "yes"
if :nil then "yes" else "no" end       # => "no"

if 5 > 3 then 10 else 20 end           # => 10
if 2 > 3 then 10 else 20 end           # => 20
```

### Test Suite 1.5: Logical Operators
```zapp
:true and :true                        # => :true
:true and :false                       # => :false
:true or :false                        # => :true
:false or :false                       # => :false
not :true                              # => :false
not :false                             # => :true

5 > 3 and 10 < 20                      # => :true
5 > 10 or 3 < 2                        # => :false
```

### Test Suite 1.6: Result Types
```zapp
{:ok, 42}                              # => {:ok, 42}
{:error, "fail"}                       # => {:error, "fail"}

def safe_divide(a, b) =
  if b == 0 then
    {:error, "division by zero"}
  else
    {:ok, a / b}
  end

safe_divide(10, 2)                     # => {:ok, 5}
safe_divide(10, 0)                     # => {:error, "division by zero"}
```

### Test Suite 1.7: Print
```zapp
print("Hello")                         # OUTPUT: Hello
print(42)                              # OUTPUT: 42
print(:ok)                             # OUTPUT: :ok
print({:ok, 5})                        # OUTPUT: {:ok, 5}
```

### Test Suite 1.8: Let Bindings
```zapp
let x = 5 in x                         # => 5
let x = 5 in x + 3                     # => 8
let x = 5 in let y = 3 in x + y        # => 8
let f = λx.x * 2 in f(5)               # => 10
```

---

## GPU Encoding Summary

### Atoms on GPU
```c
__constant__ uint32_t atom_table[1024];  // Pre-interned atoms
// atom_table[0] = "true"
// atom_table[1] = "false"
// atom_table[2] = "ok"
// atom_table[3] = "error"

// Atom comparison = integer comparison
__device__ bool atom_eq(uint32_t a, uint32_t b) {
  return a == b;  // Single GPU instruction!
}
```

### If-Then-Else on GPU
```c
__device__ void reduce_if(Atom* cond, Agent* then_br, Agent* else_br) {
  // Fast integer check
  bool is_truthy = (cond->atom_id != 1) && (cond->atom_id != 4);
  // 1=:false, 4=:nil
  
  Agent* chosen = is_truthy ? then_br : else_br;
  // Continue with chosen branch
}
```

### Result Types on GPU
```c
// Result is just a tagged tuple
struct Result {
  uint32_t tag;    // Atom ID (0=:ok, 3=:error)
  uint32_t value;  // Payload
};

// Pattern match = switch
__device__ void handle_result(Result r) {
  switch (r.tag) {
    case 0: use_ok_value(r.value); break;
    case 3: handle_error(r.value); break;
  }
}
```

---

## Deliverables
- ✅ Fixed lambda calculus reduction
- ✅ Atom type with registry
- ✅ Comparison operators returning atoms
- ✅ If-then-else with atoms
- ✅ Logical operators (and/or/not)
- ✅ Result types ({:ok, _} / {:error, _})
- ✅ Print operation
- ✅ String type
- ✅ Let bindings
- ✅ GPU-compatible encodings
- ✅ Comprehensive test suite

## Success Criteria
All test suites pass, and you can write basic functional programs with:
- Lambda expressions
- Atoms instead of booleans
- Arithmetic
- Conditionals
- Result types for error handling
- Local bindings
- Output

## Estimated Time
2-3 weeks (most time on fixing lambda calculus bug)

---

**Next Phase:** Phase 2 - Tree-Based Data Structures (bend/fold)