# Zapp Phase 4: Meta-Programming Foundation

## Goal
Add quote/unquote/eval and AST manipulation to enable meta-programming. AST represented as trees for GPU compatibility.

## Prerequisites
- Phase 1-3 complete
- Tree-based data structures working
- Module system operational

## Philosophy
- AST is just another tree structure
- Quote converts code → data (tree)
- Unquote injects values into quoted code
- Eval converts data → code (executes tree)
- All GPU-compatible

---

## Tasks

### 4.1 AST as Tree Structure

AST nodes are trees with metadata.

```javascript
const AgentType = {
  // ... existing ...
  QUOTE: 60,    // Quotation wrapper
  UNQUOTE: 61,  // Unquote (inject value)
  SPLICE: 62,   // Unquote splicing (for lists)
  EVAL: 63,     // Evaluate quoted code
  AST_NODE: 64, // AST node wrapper
};

class ASTNode {
  constructor(net, nodeType, children = []) {
    this.nodeType = nodeType;  // 'lambda', 'app', 'if', 'op2', etc.
    this.children = children;  // Array of child AST nodes
    this.metadata = {
      sourceLocation: null,
      environment: null
    };
  }
  
  // Convert to tree representation
  toTree(net) {
    // AST node as tree:
    // Node(
    //   Leaf(nodeType as atom),
    //   children as tree
    // )
    
    const typeAtom = net.createAtom(this.nodeType);
    
    if (this.children.length === 0) {
      return net.createNode(typeAtom, net.createEmpty());
    }
    
    // Build children as balanced tree
    const childTrees = this.children.map(child => 
      child instanceof ASTNode ? child.toTree(net) : child
    );
    const childrenTree = net.buildBalancedTree(childTrees);
    
    return net.createNode(typeAtom, childrenTree);
  }
}
```

---

### 4.2 Quote Implementation

Quote converts code agents into AST tree.

```javascript
// In InteractionNet:
reduceQuote(quoteAgent, codeAgent) {
  console.log(`=== Quote: Converting code to AST ===`);
  
  // Reify agent graph into AST tree
  const astTree = this.reifyAgentGraph(codeAgent);
  
  // Remove quote agent
  this.removeAgent(quoteAgent.id);
  
  return astTree;
}

reifyAgentGraph(rootAgent) {
  const visited = new Set();
  
  const reify = (agent) => {
    if (visited.has(agent.id)) {
      // Circular reference - create ref node
      const refAtom = this.createAtom('ref');
      const idNum = this.createAgent(AgentType.NUM, 0, agent.id);
      return this.createNode(refAtom, idNum);
    }
    
    visited.add(agent.id);
    
    // Get agent type name
    const typeName = this.getAgentTypeName(agent.type);
    const typeAtom = this.createAtom(typeName);
    
    // Build metadata node
    const metadata = this.createNode(
      this.createAgent(AgentType.NUM, 0, agent.id),
      agent.data ? this.reifyData(agent.data) : this.createEmpty()
    );
    
    // Reify children (connected agents)
    const children = [];
    agent.getAllPorts().forEach(port => {
      if (port.isConnected()) {
        const connected = port.getConnectedAgent();
        if (connected && !visited.has(connected.id)) {
          children.push(reify(connected));
        }
      }
    });
    
    const childrenTree = children.length > 0 
      ? this.buildBalancedTree(children)
      : this.createEmpty();
    
    // AST node structure:
    // Node(
    //   type,
    //   Node(metadata, children)
    // )
    return this.createNode(
      typeAtom,
      this.createNode(metadata, childrenTree)
    );
  };
  
  return reify(rootAgent);
}

reifyData(data) {
  if (typeof data === 'number') {
    return this.createAgent(AgentType.NUM, 0, data);
  } else if (typeof data === 'string') {
    return this.createAgent(AgentType.STR, 0, data);
  } else {
    return this.createEmpty();
  }
}

getAgentTypeName(type) {
  for (const [name, value] of Object.entries(AgentType)) {
    if (value === type) {
      return name.toLowerCase();
    }
  }
  return 'unknown';
}
```

**Parser:**
```javascript
parseQuote(tokens, startIndex) {
  let i = startIndex + 1; // Skip 'quote'
  
  // Expect 'do'
  if (tokens[i].value !== 'do') {
    throw new Error('Expected "do" after quote');
  }
  i++;
  
  // Find matching 'end'
  const endPos = this.findKeyword(tokens, i, 'end');
  const bodyTokens = tokens.slice(i, endPos);
  
  // Create quote agent
  const quoteAgent = this.net.createAgent(AgentType.QUOTE, 1);
  
  // Parse body in quoted context
  this.quotedContext = true;
  const beforeCount = this.net.agents.size;
  this.parseTokens(bodyTokens);
  this.quotedContext = false;
  
  // Get parsed body
  const newAgents = Array.from(this.net.agents.values()).slice(beforeCount);
  const bodyAgent = newAgents[newAgents.length - 1];
  
  // Connect quote to body
  this.net.connectPorts(quoteAgent.auxiliaryPorts[0], bodyAgent.principalPort);
  
  return endPos + 1;
}
```

**Syntax:**
```zapp
quote do
  x + 1
end

# Returns AST tree:
# Node(
#   :op2,
#   Node(
#     metadata,
#     Node(
#       Node(:con, "x"),
#       Node(:num, 1)
#     )
#   )
# )
```

---

### 4.3 Unquote Implementation

Unquote injects evaluated values into quoted code.

```javascript
reduceUnquote(unquoteAgent, valueAgent) {
  console.log(`=== Unquote: Injecting value ===`);
  
  // If we're in quoted context, evaluate and inject
  if (this.quotedContext) {
    // Evaluate the value (escape quoted context)
    const wasQuoted = this.quotedContext;
    this.quotedContext = false;
    
    const evaluatedValue = this.evaluateToValue(valueAgent);
    
    this.quotedContext = wasQuoted;
    
    // Convert value to AST node
    const astNode = this.valueToAST(evaluatedValue);
    
    this.removeAgent(unquoteAgent.id);
    return astNode;
  }
  
  // Not in quoted context - error
  throw new Error('unquote outside of quote context');
}

valueToAST(value) {
  if (typeof value === 'number') {
    const numAtom = this.createAtom('num');
    const numAgent = this.createAgent(AgentType.NUM, 0, value);
    return this.createNode(numAtom, numAgent);
  } else if (typeof value === 'string') {
    const strAtom = this.createAtom('str');
    const strAgent = this.createAgent(AgentType.STR, 0, value);
    return this.createNode(strAtom, strAgent);
  } else if (value.type === AgentType.ATOM) {
    const atomAtom = this.createAtom('atom');
    return this.createNode(atomAtom, value);
  }
  
  // For complex values, convert to AST tree
  return this.reifyAgentGraph(value);
}
```

**Parser:**
```javascript
parseUnquote(tokens, startIndex) {
  let i = startIndex + 1; // Skip 'unquote'
  
  // Expect '('
  if (tokens[i].type !== '(') {
    throw new Error('Expected "(" after unquote');
  }
  i++;
  
  // Find matching ')'
  const endPos = this.findMatchingParen(tokens, i);
  const exprTokens = tokens.slice(i, endPos);
  
  // Create unquote agent
  const unquoteAgent = this.net.createAgent(AgentType.UNQUOTE, 1);
  
  // Parse expression (NOT in quoted context)
  const wasQuoted = this.quotedContext;
  this.quotedContext = false;
  
  const beforeCount = this.net.agents.size;
  this.parseTokens(exprTokens);
  
  this.quotedContext = wasQuoted;
  
  // Get expression
  const newAgents = Array.from(this.net.agents.values()).slice(beforeCount);
  const exprAgent = newAgents[newAgents.length - 1];
  
  // Connect unquote to expression
  this.net.connectPorts(unquoteAgent.auxiliaryPorts[0], exprAgent.principalPort);
  
  return endPos + 1;
}
```

**Syntax:**
```zapp
let x = 5 in
quote do
  unquote(x) + 1
end

# Returns AST for: 5 + 1
# (x is evaluated and injected)
```

---

### 4.4 Eval Implementation

Eval converts AST tree back into executable agents.

```javascript
reduceEval(evalAgent, astTreeAgent) {
  console.log(`=== Eval: Executing quoted code ===`);
  
  // Reflect AST tree back into agent graph
  const codeAgent = this.reflectAST(astTreeAgent);
  
  // Remove eval and AST agents
  this.removeAgent(evalAgent.id);
  this.removeAgent(astTreeAgent.id);
  
  return codeAgent;
}

reflectAST(astTree) {
  if (!astTree || astTree.type === AgentType.EMPTY) {
    return this.createEmpty();
  }
  
  if (astTree.type === AgentType.LEAF) {
    // Simple value
    return astTree;
  }
  
  // AST node: Node(type, Node(metadata, children))
  const typeAgent = astTree.auxiliaryPorts[0].getConnectedAgent();
  const restTree = astTree.auxiliaryPorts[1].getConnectedAgent();
  
  if (typeAgent.type !== AgentType.ATOM) {
    throw new Error('Invalid AST: expected atom for type');
  }
  
  const nodeType = this.atomRegistry.lookup(typeAgent.data);
  
  // Extract metadata and children
  const metadataTree = restTree.auxiliaryPorts[0].getConnectedAgent();
  const childrenTree = restTree.auxiliaryPorts[1].getConnectedAgent();
  
  // Extract data from metadata
  const dataAgent = metadataTree.auxiliaryPorts[1].getConnectedAgent();
  let data = null;
  if (dataAgent.type === AgentType.NUM) {
    data = dataAgent.data;
  } else if (dataAgent.type === AgentType.STR) {
    data = dataAgent.data;
  }
  
  // Create agent of appropriate type
  const agentType = AgentType[nodeType.toUpperCase()];
  if (agentType === undefined) {
    throw new Error(`Unknown agent type: ${nodeType}`);
  }
  
  // Reflect children
  const children = this.flattenTree(childrenTree).map(child => 
    this.reflectAST(child)
  );
  
  const agent = this.createAgent(agentType, children.length, data);
  
  // Connect children
  children.forEach((child, i) => {
    if (agent.auxiliaryPorts[i]) {
      this.connectPorts(agent.auxiliaryPorts[i], child.principalPort);
    }
  });
  
  return agent;
}
```

**Syntax:**
```zapp
let code = quote do x + 1 end in
let x = 5 in
eval(code)    # => 6

# Dynamic code generation
let make_adder = λn.quote do λx.x + unquote(n) end in
let add5_code = make_adder(5) in
let add5 = eval(add5_code) in
add5(3)       # => 8
```

---

### 4.5 Unquote Splicing

For injecting lists of values.

```javascript
reduceSplice(spliceAgent, listAgent) {
  console.log(`=== Unquote Splice: Injecting list ===`);
  
  if (!this.quotedContext) {
    throw new Error('unquote_splicing outside of quote context');
  }
  
  // Flatten tree to list of elements
  const elements = this.flattenTree(listAgent);
  
  // Convert each element to AST
  const astElements = elements.map(elem => this.valueToAST(elem));
  
  // Build as tree of AST nodes
  const astTree = this.buildBalancedTree(astElements);
  
  this.removeAgent(spliceAgent.id);
  return astTree;
}
```

**Syntax:**
```zapp
let values = [1, 2, 3] in
quote do
  [unquote_splicing(values)]
end

# Returns AST for: [1, 2, 3]
```

---

### 4.6 AST Inspection Functions

Standard library for working with AST trees.

```zapp
module AST do
  # Get node type
  def node_type(ast) =
    match ast with
      Node(type, _) -> type
      _ -> :unknown
    end
  
  # Get children
  def children(ast) =
    match ast with
      Node(_, Node(_, children)) -> children
      _ -> []
    end
  
  # Get metadata
  def metadata(ast) =
    match ast with
      Node(_, Node(meta, _)) -> meta
      _ -> Empty
    end
  
  # Check node type
  def is_lambda(ast) = node_type(ast) == :lam
  def is_app(ast) = node_type(ast) == :app
  def is_num(ast) = node_type(ast) == :num
  def is_atom(ast) = node_type(ast) == :atom
  def is_if(ast) = node_type(ast) == :if
  
  # Get data from leaf nodes
  def node_data(ast) =
    match metadata(ast) with
      Node(_, data) -> data
      _ -> :none
    end
end
```

---

### 4.7 AST Transformation Functions

```zapp
module AST do
  # Walk AST with visitor function
  def walk(ast, visitor) =
    match ast with
      Empty -> Empty
      Leaf(x) -> visitor(Leaf(x))
      Node(type, rest) ->
        let new_node = visitor(ast) in
        let children = children(new_node) in
        let new_children = map(λc.walk(c, visitor), children) in
        rebuild(node_type(new_node), new_children)
    end
  
  # Replace nodes matching predicate
  def replace(ast, pred, replacement) =
    walk(ast, λnode.
      if pred(node) then
        replacement
      else
        node
      end
    )
  
  # Find all nodes matching predicate
  def find_all(ast, pred) =
    if pred(ast) then
      [ast]
    else
      []
    end ++
    fold (children(ast)) λ(acc, child).
      acc ++ find_all(child, pred)
  
  # Count nodes
  def count_nodes(ast) =
    1 + fold (children(ast)) λ(acc, child).
      acc + count_nodes(child)
  
  # Find free variables
  def free_vars(ast) =
    match ast with
      Node(:con, Node(Node(_, Leaf(name)), _)) ->
        [name]
      Node(:lam, _) ->
        let param = node_data(ast) in
        let body = children(ast) in
        filter(λv.v != param, free_vars(body))
      _ ->
        fold (children(ast)) λ(acc, child).
          acc ++ free_vars(child)
    end
  
  # Find bound variables
  def bound_vars(ast) =
    match ast with
      Node(:lam, _) ->
        [node_data(ast)] ++ 
        fold (children(ast)) λ(acc, child).
          acc ++ bound_vars(child)
      _ ->
        fold (children(ast)) λ(acc, child).
          acc ++ bound_vars(child)
    end
end
```

---

### 4.8 Hygiene System (Basic)

Prevent variable capture in generated code.

```javascript
class HygieneContext {
  constructor() {
    this.gensymCounter = 0;
    this.renamedVars = new Map();
  }
  
  gensym(base = 'g') {
    return `${base}_${this.gensymCounter++}`;
  }
  
  renameVar(originalName) {
    if (!this.renamedVars.has(originalName)) {
      this.renamedVars.set(originalName, this.gensym(originalName));
    }
    return this.renamedVars.get(originalName);
  }
}

// In InteractionNet:
reifyAgentGraphHygienic(rootAgent) {
  const hygiene = new HygieneContext();
  
  const reify = (agent, boundVars = new Set()) => {
    if (agent.type === AgentType.LAM) {
      // Lambda introduces new binding
      const originalParam = agent.data;
      const newParam = hygiene.renameVar(originalParam);
      
      // Build AST with renamed parameter
      const lamAtom = this.createAtom('lam');
      const paramAtom = this.createAtom(newParam);
      
      // Recursively process body with new bound var
      const body = agent.auxiliaryPorts[0].getConnectedAgent();
      const newBoundVars = new Set([...boundVars, originalParam]);
      const bodyAST = reify(body, newBoundVars);
      
      return this.createNode(lamAtom, this.createNode(paramAtom, bodyAST));
    } else if (agent.type === AgentType.CON && boundVars.has(agent.data)) {
      // Bound variable reference - rename
      const renamedVar = hygiene.renamedVars.get(agent.data);
      const conAtom = this.createAtom('con');
      const varAtom = this.createAtom(renamedVar);
      return this.createNode(conAtom, varAtom);
    }
    
    // Default case
    return this.reifyAgentGraph(agent);
  };
  
  return reify(rootAgent);
}
```

---

### 4.9 Source Location Tracking

```javascript
class SourceLocation {
  constructor(file, line, column) {
    this.file = file;
    this.line = line;
    this.column = column;
  }
  
  toString() {
    return `${this.file}:${this.line}:${this.column}`;
  }
}

class Agent {
  constructor(id, type, arity = 0) {
    // ... existing ...
    this.sourceLocation = null;
  }
}

// During parsing:
parseExpression(tokens, startIndex) {
  const token = tokens[startIndex];
  const agent = this.net.createAgent(/* ... */);
  
  // Attach source location
  if (token.line && token.column) {
    agent.sourceLocation = new SourceLocation(
      this.currentFile,
      token.line,
      token.column
    );
  }
  
  return agent;
}

// Include in AST metadata
reifyAgentGraph(rootAgent) {
  // ...
  const metadata = this.createNode(
    this.createAgent(AgentType.NUM, 0, agent.id),
    agent.sourceLocation 
      ? this.createSourceLocationNode(agent.sourceLocation)
      : this.createEmpty()
  );
  // ...
}

createSourceLocationNode(loc) {
  const fileAtom = this.createAtom(loc.file);
  const lineNum = this.createAgent(AgentType.NUM, 0, loc.line);
  const colNum = this.createAgent(AgentType.NUM, 0, loc.column);
  
  return this.createNode(
    fileAtom,
    this.createNode(lineNum, colNum)
  );
}
```

---

## GPU Encoding

### AST Tree on GPU
```c
// AST is just a tree - same as any other tree!
struct ASTNode {
  uint32_t type;         // EMPTY, LEAF, NODE
  uint32_t node_type;    // Atom ID: :lam, :app, :num, etc.
  uint32_t left_id;      // Left child
  uint32_t right_id;     // Right child
  uint32_t data;         // Leaf data
};

// Quote/Unquote/Eval are just tree transformations
__device__ TreeNode* quote_agent(Agent* code) {
  // Convert agent to tree representation
  return reify_to_tree(code);
}

__device__ Agent* eval_tree(TreeNode* ast) {
  // Convert tree back to agent
  return reflect_from_tree(ast);
}
```

---

## Testing Strategy

### Test Suite 4.1: Basic Quote
```zapp
quote do 5 end                    # => AST for 5

quote do x + 1 end                # => AST for x + 1

quote do
  if x > 5 then x else 0 end
end                                # => AST for if expression
```

### Test Suite 4.2: Unquote
```zapp
let x = 5 in
quote do
  unquote(x) + 1
end                                # => AST for 5 + 1

let op = :+ in
quote do
  1 unquote(op) 2
end                                # => AST for 1 + 2
```

### Test Suite 4.3: Eval
```zapp
let code = quote do 5 + 3 end in
eval(code)                         # => 8

let x = 10 in
let code = quote do unquote(x) * 2 end in
eval(code)                         # => 20
```

### Test Suite 4.4: Unquote Splicing
```zapp
let values = [1, 2, 3] in
quote do
  [unquote_splicing(values)]
end                                # => AST for [1, 2, 3]
```

### Test Suite 4.5: AST Inspection
```zapp
import AST

let ast = quote do x + 1 end in
AST.node_type(ast)                 # => :op2
AST.is_num(ast)                    # => :false
AST.count_nodes(ast)               # => 3
```

### Test Suite 4.6: AST Transformation
```zapp
import AST

# Replace all numbers with 0
let ast = quote do 1 + 2 + 3 end in
let transformed = AST.replace(
  ast,
  λn.AST.is_num(n),
  quote do 0 end
) in
eval(transformed)                  # => 0
```

### Test Suite 4.7: Free Variables
```zapp
import AST

let ast1 = quote do x + y end in
AST.free_vars(ast1)                # => ["x", "y"]

let ast2 = quote do λx.x + y end in
AST.free_vars(ast2)                # => ["y"]
```

### Test Suite 4.8: Hygiene
```zapp
let x = 1 in
let make_incrementer = quote do
  λx.x + unquote(x)
end in
let inc = eval(make_incrementer) in
inc(5)                             # => 6 (not 10)
```

---

## Deliverables
- ✅ AST as tree structure
- ✅ Quote implementation (code → data)
- ✅ Unquote implementation (inject values)
- ✅ Eval implementation (data → code)
- ✅ Unquote splicing
- ✅ AST inspection functions
- ✅ AST transformation functions
- ✅ Basic hygiene system
- ✅ Source location tracking
- ✅ GPU-compatible encoding
- ✅ Comprehensive test suite

## Success Criteria
- Can quote code to get AST
- Can unquote values into quoted code
- Can eval quoted code
- AST inspection works
- AST transformation works
- Basic hygiene prevents variable capture
- All operations are tree-based (GPU-friendly)

## Estimated Time
3-4 weeks

---

**Key Insight:** AST is just another tree structure. This means:
1. Same GPU-parallel operations (map, fold, etc.)
2. O(log n) traversal and transformation
3. Consistent with rest of Zapp's design

**Next Phase:** Phase 5 - Macro System (build on quote/unquote)