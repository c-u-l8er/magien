# Zapp Phase 2: Tree-Based Data Structures

## Goal
Build ALL data structures on binary trees for native GPU parallelism. Lists, tuples, strings - everything is a tree underneath.

## Prerequisites
- Phase 1 complete (atoms, result types, lambda calculus working)

## Philosophy
**Everything is a tree.** This gives:
- Native GPU parallelism for all operations
- Consistent mental model
- Automatic parallel execution
- O(log n) operations instead of O(n)

---

## Tasks

### 2.1 Binary Tree Agent Types

```javascript
const AgentType = {
  // ... existing ...
  EMPTY: 50,    // Empty tree (like nil)
  LEAF: 51,     // Leaf node (holds single value)
  NODE: 52,     // Internal node (has 2 children)
};

// In InteractionNet:
createEmpty() {
  return this.createAgent(AgentType.EMPTY, 0);
}

createLeaf(value) {
  return this.createAgent(AgentType.LEAF, 0, value);
}

createNode(leftAgent, rightAgent) {
  const node = this.createAgent(AgentType.NODE, 2);
  this.connectPorts(node.auxiliaryPorts[0], leftAgent.principalPort);
  this.connectPorts(node.auxiliaryPorts[1], rightAgent.principalPort);
  return node;
}
```

**Tree structure:**
```
Empty                    # Empty tree
Leaf(42)                # Single value
Node(Leaf(1), Leaf(2))  # Binary node

# Larger tree:
      Node
      /  \
   Node  Node
   / \   / \
  1   2 3   4
```

---

### 2.2 List Syntax → Tree Encoding

Lists are syntactic sugar over balanced binary trees.

```javascript
// In NetParser:
parseList(tokens, startIndex) {
  let i = startIndex + 1; // Skip '['
  
  const elements = [];
  
  while (i < tokens.length && tokens[i].type !== ']') {
    const elemEnd = this.findListElementEnd(tokens, i);
    const elemTokens = tokens.slice(i, elemEnd);
    
    // Parse element
    const beforeCount = this.net.agents.size;
    this.parseTokens(elemTokens);
    const newAgents = Array.from(this.net.agents.values()).slice(beforeCount);
    
    if (newAgents.length > 0) {
      elements.push(newAgents[newAgents.length - 1]);
    }
    
    i = elemEnd;
    
    // Skip comma
    if (i < tokens.length && tokens[i].type === ',') {
      i++;
    }
  }
  
  i++; // Skip ']'
  
  // Build balanced binary tree
  const tree = this.buildBalancedTree(elements);
  
  return i;
}

buildBalancedTree(elements) {
  if (elements.length === 0) {
    return this.net.createEmpty();
  }
  
  if (elements.length === 1) {
    return this.net.createLeaf(elements[0]);
  }
  
  // Split in half for balance
  const mid = Math.floor(elements.length / 2);
  const leftElements = elements.slice(0, mid);
  const rightElements = elements.slice(mid);
  
  const leftTree = this.buildBalancedTree(leftElements);
  const rightTree = this.buildBalancedTree(rightElements);
  
  return this.net.createNode(leftTree, rightTree);
}

findListElementEnd(tokens, startIndex) {
  let i = startIndex;
  let depth = 0;
  
  while (i < tokens.length) {
    if (tokens[i].type === '[' || tokens[i].type === '(') {
      depth++;
    } else if (tokens[i].type === ']' || tokens[i].type === ')') {
      if (depth === 0) break;
      depth--;
    } else if (tokens[i].type === ',' && depth === 0) {
      break;
    }
    i++;
  }
  
  return i;
}
```

**Syntax:**
```zapp
# User writes
[]                        # Empty tree
[42]                      # Leaf(42)
[1, 2, 3, 4]             # Balanced tree

# Compiler creates:
#       Node
#       /  \
#     Node Node
#     / \  / \
#    1   2 3  4
```

---

### 2.3 Bend - Parallel Tree Construction

Generate trees in parallel using a range and function.

```javascript
const AgentType = {
  // ... existing ...
  BEND: 53,     // Parallel construction
};

reduceBend(bendAgent, startAgent, endAgent, bodyAgent) {
  const start = startAgent.data;
  const end = endAgent.data;
  
  console.log(`=== Bend: range ${start}..${end} ===`);
  
  // Base case: empty range
  if (end <= start) {
    return this.createEmpty();
  }
  
  // Base case: single element
  if (end - start === 1) {
    // Evaluate body with this index
    const valueAgent = this.evaluateBody(bodyAgent, start);
    return this.createLeaf(valueAgent);
  }
  
  // Recursive case: split in half
  const mid = Math.floor((start + end) / 2);
  
  // Create left subtree (can be parallel on GPU)
  const leftStart = this.createAgent(AgentType.NUM, 0, start);
  const leftEnd = this.createAgent(AgentType.NUM, 0, mid);
  const leftBodyClone = this.cloneAgent(bodyAgent);
  const leftTree = this.reduceBend(bendAgent, leftStart, leftEnd, leftBodyClone);
  
  // Create right subtree (can be parallel on GPU)
  const rightStart = this.createAgent(AgentType.NUM, 0, mid);
  const rightEnd = this.createAgent(AgentType.NUM, 0, end);
  const rightBodyClone = this.cloneAgent(bodyAgent);
  const rightTree = this.reduceBend(bendAgent, rightStart, rightEnd, rightBodyClone);
  
  // Combine into node
  return this.createNode(leftTree, rightTree);
}

evaluateBody(bodyAgent, index) {
  // Apply body function to index
  if (bodyAgent.type === AgentType.LAM) {
    // Create application
    const appAgent = this.createAgent(AgentType.APP, 1);
    const indexAgent = this.createAgent(AgentType.NUM, 0, index);
    
    this.connectPorts(appAgent.principalPort, bodyAgent.principalPort);
    this.connectPorts(appAgent.auxiliaryPorts[0], indexAgent.principalPort);
    
    // Reduce to get result
    this.reduceLambdaApp(bodyAgent, appAgent);
    
    // Return result (simplified - needs proper implementation)
    return indexAgent;
  }
  
  return this.createAgent(AgentType.NUM, 0, index);
}
```

**Parser:**
```javascript
parseBend(tokens, startIndex) {
  let i = startIndex + 1; // Skip 'bend'
  
  // Parse start
  const start = parseInt(tokens[i].value);
  i++;
  
  // Parse end
  const end = parseInt(tokens[i].value);
  i++;
  
  // Parse body (lambda or expression)
  const bodyEnd = this.findExpressionEnd(tokens, i);
  const bodyTokens = tokens.slice(i, bodyEnd);
  
  const beforeBody = this.net.agents.size;
  this.parseTokens(bodyTokens);
  const bodyAgents = Array.from(this.net.agents.values()).slice(beforeBody);
  const bodyAgent = bodyAgents[bodyAgents.length - 1];
  
  // Create BEND agent
  const startAgent = this.net.createAgent(AgentType.NUM, 0, start);
  const endAgent = this.net.createAgent(AgentType.NUM, 0, end);
  
  const bendAgent = this.net.createAgent(AgentType.BEND, 3);
  this.net.connectPorts(bendAgent.auxiliaryPorts[0], startAgent.principalPort);
  this.net.connectPorts(bendAgent.auxiliaryPorts[1], endAgent.principalPort);
  this.net.connectPorts(bendAgent.auxiliaryPorts[2], bodyAgent.principalPort);
  
  return bodyEnd;
}
```

**Syntax:**
```zapp
# Generate range [0..8)
bend 0 8 λn.n
# Creates tree: [0, 1, 2, 3, 4, 5, 6, 7]

# With transformation
bend 0 8 λn.n * n
# Creates: [0, 1, 4, 9, 16, 25, 36, 49]

# Nested
bend 0 4 λi.bend 0 3 λj.i + j
# Creates: [[0,1,2], [1,2,3], [2,3,4], [3,4,5]]
```

---

### 2.4 Fold - Parallel Tree Reduction

Reduce trees in parallel using a combining function.

```javascript
const AgentType = {
  // ... existing ...
  FOLD: 54,     // Parallel reduction
};

reduceFold(foldAgent, treeAgent, combineAgent, identityAgent) {
  console.log(`=== Fold: tree ${treeAgent.id} ===`);
  
  // Base case: empty tree
  if (treeAgent.type === AgentType.EMPTY) {
    return identityAgent;  // Return identity element
  }
  
  // Base case: leaf
  if (treeAgent.type === AgentType.LEAF) {
    return treeAgent;  // Leaf value is already reduced
  }
  
  // Recursive case: NODE
  if (treeAgent.type === AgentType.NODE) {
    const left = treeAgent.auxiliaryPorts[0].getConnectedAgent();
    const right = treeAgent.auxiliaryPorts[1].getConnectedAgent();
    
    // Fold both sides (PARALLEL on GPU!)
    const leftResult = this.reduceFold(
      foldAgent, 
      left, 
      this.cloneAgent(combineAgent),
      this.cloneAgent(identityAgent)
    );
    
    const rightResult = this.reduceFold(
      foldAgent,
      right,
      this.cloneAgent(combineAgent),
      this.cloneAgent(identityAgent)
    );
    
    // Combine results
    return this.applyCombine(combineAgent, leftResult, rightResult);
  }
  
  throw new Error(`Unknown tree type: ${treeAgent.type}`);
}

applyCombine(combineAgent, leftAgent, rightAgent) {
  // combineAgent is a lambda: λ(l, r).l + r
  if (combineAgent.type === AgentType.LAM) {
    // Apply to first argument
    const app1 = this.createAgent(AgentType.APP, 1);
    this.connectPorts(app1.principalPort, combineAgent.principalPort);
    this.connectPorts(app1.auxiliaryPorts[0], leftAgent.principalPort);
    
    this.reduceLambdaApp(combineAgent, app1);
    
    // Result is another lambda, apply to second argument
    const resultLambda = app1; // Simplified
    const app2 = this.createAgent(AgentType.APP, 1);
    this.connectPorts(app2.principalPort, resultLambda.principalPort);
    this.connectPorts(app2.auxiliaryPorts[0], rightAgent.principalPort);
    
    this.reduceLambdaApp(resultLambda, app2);
    
    return app2; // Final result
  }
  
  return combineAgent;
}
```

**Parser:**
```javascript
parseFold(tokens, startIndex) {
  let i = startIndex + 1; // Skip 'fold'
  
  // Parse tree expression
  const treeEnd = this.findComma(tokens, i);
  const treeTokens = tokens.slice(i, treeEnd);
  
  const beforeTree = this.net.agents.size;
  this.parseTokens(treeTokens);
  const treeAgents = Array.from(this.net.agents.values()).slice(beforeTree);
  const treeAgent = treeAgents[treeAgents.length - 1];
  
  i = treeEnd + 1; // Skip comma
  
  // Parse combine function
  const combineEnd = this.findExpressionEnd(tokens, i);
  const combineTokens = tokens.slice(i, combineEnd);
  
  const beforeCombine = this.net.agents.size;
  this.parseTokens(combineTokens);
  const combineAgents = Array.from(this.net.agents.values()).slice(beforeCombine);
  const combineAgent = combineAgents[combineAgents.length - 1];
  
  // Identity element (default 0 for numbers)
  const identityAgent = this.net.createAgent(AgentType.NUM, 0, 0);
  
  // Create FOLD agent
  const foldAgent = this.net.createAgent(AgentType.FOLD, 3);
  this.net.connectPorts(foldAgent.auxiliaryPorts[0], treeAgent.principalPort);
  this.net.connectPorts(foldAgent.auxiliaryPorts[1], combineAgent.principalPort);
  this.net.connectPorts(foldAgent.auxiliaryPorts[2], identityAgent.principalPort);
  
  return combineEnd;
}
```

**Syntax:**
```zapp
# Sum a list
fold [1, 2, 3, 4] λ(a, b).a + b
# Execution (parallel):
#    fold
#     /\
#   fold fold
#   /\   /\
#  1  2 3  4
# Step 1: 1+2=3, 3+4=7
# Step 2: 3+7=10

# Product
fold [1, 2, 3, 4] λ(a, b).a * b     # => 24

# Max
fold [5, 2, 8, 1] λ(a, b).if a > b then a else b
# => 8
```

---

### 2.5 Map - Parallel Transformation

```zapp
# Map as tree recursion
def map(f, tree) =
  match tree with
    Empty -> Empty
    Leaf(x) -> Leaf(f(x))
    Node(l, r) -> Node(map(f, l), map(f, r))
  end

# Usage
map(λx.x * 2, [1, 2, 3, 4])
# => [2, 4, 6, 8]

# On GPU: both branches parallel!
#      map
#      / \
#   map   map
#   / \   / \
#  1   2 3   4
# All 4 applications happen simultaneously
```

**Implementation:**
```javascript
// Map is just a special fold + rebuild
reduceMap(mapAgent, functionAgent, treeAgent) {
  if (treeAgent.type === AgentType.EMPTY) {
    return this.createEmpty();
  }
  
  if (treeAgent.type === AgentType.LEAF) {
    // Apply function to leaf value
    const result = this.applyFunction(functionAgent, treeAgent);
    return this.createLeaf(result);
  }
  
  if (treeAgent.type === AgentType.NODE) {
    const left = treeAgent.auxiliaryPorts[0].getConnectedAgent();
    const right = treeAgent.auxiliaryPorts[1].getConnectedAgent();
    
    // Map both sides (PARALLEL)
    const leftMapped = this.reduceMap(
      mapAgent,
      this.cloneAgent(functionAgent),
      left
    );
    
    const rightMapped = this.reduceMap(
      mapAgent,
      this.cloneAgent(functionAgent),
      right
    );
    
    return this.createNode(leftMapped, rightMapped);
  }
}
```

---

### 2.6 Filter - Parallel Filtering

```zapp
def filter(pred, tree) =
  match tree with
    Empty -> Empty
    Leaf(x) -> if pred(x) then Leaf(x) else Empty
    Node(l, r) ->
      let l' = filter(pred, l) in
      let r' = filter(pred, r) in
      merge(l', r')
  end

# Helper: merge trees
def merge(t1, t2) =
  match (t1, t2) with
    (Empty, t) -> t
    (t, Empty) -> t
    (t1, t2) -> Node(t1, t2)
  end

# Usage
filter(λx.x > 2, [1, 2, 3, 4])
# => [3, 4]
```

---

### 2.7 Standard Tree Operations

```zapp
# Size (count elements)
def size(tree) =
  match tree with
    Empty -> 0
    Leaf(_) -> 1
    Node(l, r) -> size(l) + size(r)
  end

# Or with fold:
def size(tree) =
  fold tree λ(l, r).l + r with
    Empty -> 0
    Leaf(_) -> 1
  end

# Nth element (tree indexing)
def nth(tree, n) =
  match tree with
    Empty -> {:error, "index out of bounds"}
    Leaf(x) -> 
      if n == 0 then {:ok, x} 
      else {:error, "index out of bounds"}
    Node(l, r) ->
      let left_size = size(l) in
      if n < left_size then
        nth(l, n)
      else
        nth(r, n - left_size)
      end
  end

# Flatten (for display)
def flatten(tree) =
  match tree with
    Empty -> []
    Leaf(x) -> [x]
    Node(l, r) -> concat(flatten(l), flatten(r))
  end

# Concat trees
def concat(t1, t2) =
  match (t1, t2) with
    (Empty, t) -> t
    (t, Empty) -> t
    (t1, t2) -> Node(t1, t2)
  end

# Reverse
def reverse(tree) =
  match tree with
    Empty -> Empty
    Leaf(x) -> Leaf(x)
    Node(l, r) -> Node(reverse(r), reverse(l))
  end

# Take first n elements
def take(n, tree) =
  if n <= 0 then
    Empty
  else
    match tree with
      Empty -> Empty
      Leaf(x) -> Leaf(x)
      Node(l, r) ->
        let left_size = size(l) in
        if n <= left_size then
          take(n, l)
        else
          Node(l, take(n - left_size, r))
        end
    end
  end

# Drop first n elements  
def drop(n, tree) =
  if n <= 0 then
    tree
  else
    match tree with
      Empty -> Empty
      Leaf(_) -> if n > 0 then Empty else Leaf(x)
      Node(l, r) ->
        let left_size = size(l) in
        if n < left_size then
          Node(drop(n, l), r)
        else
          drop(n - left_size, r)
        end
    end
  end
```

---

### 2.8 Tuples as Fixed Trees

```zapp
# Pair (2-tuple)
(1, 2) → Node(Leaf(1), Leaf(2))

# Triple (3-tuple)  
(1, 2, 3) → Node(
              Node(Leaf(1), Leaf(2)),
              Leaf(3)
            )

# Access helpers
def fst((x, y)) = x
def snd((x, y)) = y

# Pattern matching
match (1, 2, 3) with
  (a, b, c) -> a + b + c
end
```

**Parser:**
```javascript
parseTuple(tokens, startIndex) {
  let i = startIndex + 1; // Skip '('
  
  const elements = [];
  
  while (i < tokens.length && tokens[i].type !== ')') {
    const elemEnd = this.findTupleElementEnd(tokens, i);
    const elemTokens = tokens.slice(i, elemEnd);
    
    const beforeCount = this.net.agents.size;
    this.parseTokens(elemTokens);
    const newAgents = Array.from(this.net.agents.values()).slice(beforeCount);
    
    if (newAgents.length > 0) {
      elements.push(newAgents[newAgents.length - 1]);
    }
    
    i = elemEnd;
    
    if (i < tokens.length && tokens[i].type === ',') {
      i++;
    }
  }
  
  i++; // Skip ')'
  
  // Build tuple as tree
  if (elements.length === 0) {
    // Unit type
    return this.net.createAtom('unit');
  }
  
  if (elements.length === 1) {
    // Not a tuple, just parenthesized expression
    return elements[0];
  }
  
  // Build right-associative tree for tuple
  let tree = elements[elements.length - 1];
  for (let j = elements.length - 2; j >= 0; j--) {
    tree = this.net.createNode(elements[j], tree);
  }
  
  return tree;
}
```

---

### 2.9 Strings as Character Trees

```zapp
# Strings are trees of characters
"hello" → Node(
            Node(Leaf('h'), Leaf('e')),
            Node(
              Leaf('l'),
              Node(Leaf('l'), Leaf('o'))
            )
          )

# String operations are tree operations
def uppercase(str) =
  map(char_upper, str)

def concat_str(s1, s2) =
  concat(s1, s2)

# Length
def string_length(str) =
  size(str)
```

---

### 2.10 Pattern Matching on Trees

```javascript
const AgentType = {
  // ... existing ...
  MATCH: 55,    // Pattern match
  PATTERN: 56,  // Pattern definition
};

reduceMatch(matchAgent, valueAgent, patterns) {
  console.log(`=== Pattern Match ===`);
  
  // Try each pattern in order
  for (const pattern of patterns) {
    const bindings = this.matchPattern(pattern.pattern, valueAgent);
    
    if (bindings !== null) {
      // Pattern matched! Evaluate body with bindings
      return this.evaluateWithBindings(pattern.body, bindings);
    }
  }
  
  // No pattern matched
  throw new Error('Pattern match failed');
}

matchPattern(patternAgent, valueAgent) {
  // Empty pattern
  if (patternAgent.type === AgentType.EMPTY && valueAgent.type === AgentType.EMPTY) {
    return new Map();
  }
  
  // Leaf pattern
  if (patternAgent.type === AgentType.LEAF && valueAgent.type === AgentType.LEAF) {
    return new Map();
  }
  
  // Variable pattern (binds to anything)
  if (patternAgent.type === AgentType.CON) {
    const bindings = new Map();
    bindings.set(patternAgent.data, valueAgent);
    return bindings;
  }
  
  // Node pattern
  if (patternAgent.type === AgentType.NODE && valueAgent.type === AgentType.NODE) {
    const patternLeft = patternAgent.auxiliaryPorts[0].getConnectedAgent();
    const patternRight = patternAgent.auxiliaryPorts[1].getConnectedAgent();
    const valueLeft = valueAgent.auxiliaryPorts[0].getConnectedAgent();
    const valueRight = valueAgent.auxiliaryPorts[1].getConnectedAgent();
    
    const leftBindings = this.matchPattern(patternLeft, valueLeft);
    if (leftBindings === null) return null;
    
    const rightBindings = this.matchPattern(patternRight, valueRight);
    if (rightBindings === null) return null;
    
    // Merge bindings
    return new Map([...leftBindings, ...rightBindings]);
  }
  
  // No match
  return null;
}
```

**Syntax:**
```zapp
match [1, 2, 3] with
  [] -> 0
  [x] -> x
  [x, y] -> x + y
  [x, y, z] -> x + y + z
  _ -> -1
end

match {:ok, 42} with
  {:ok, value} -> value
  {:error, msg} -> 0
end
```

---

### 2.11 Tree Rebalancing

```javascript
// Keep trees balanced for O(log n) access
rebalanceTree(treeAgent) {
  // Flatten to array
  const elements = this.flattenTree(treeAgent);
  
  // Rebuild as balanced tree
  return this.buildBalancedTree(elements);
}

flattenTree(treeAgent) {
  if (treeAgent.type === AgentType.EMPTY) {
    return [];
  }
  
  if (treeAgent.type === AgentType.LEAF) {
    return [treeAgent];
  }
  
  if (treeAgent.type === AgentType.NODE) {
    const left = treeAgent.auxiliaryPorts[0].getConnectedAgent();
    const right = treeAgent.auxiliaryPorts[1].getConnectedAgent();
    
    return [
      ...this.flattenTree(left),
      ...this.flattenTree(right)
    ];
  }
  
  return [];
}

shouldRebalance(treeAgent) {
  const depth = this.treeDepth(treeAgent);
  const size = this.treeSize(treeAgent);
  const idealDepth = Math.ceil(Math.log2(size + 1));
  
  // Rebalance if depth > 1.5 * ideal
  return depth > idealDepth * 1.5;
}

treeDepth(treeAgent) {
  if (treeAgent.type === AgentType.EMPTY) return 0;
  if (treeAgent.type === AgentType.LEAF) return 1;
  
  const left = treeAgent.auxiliaryPorts[0].getConnectedAgent();
  const right = treeAgent.auxiliaryPorts[1].getConnectedAgent();
  
  return 1 + Math.max(
    this.treeDepth(left),
    this.treeDepth(right)
  );
}

treeSize(treeAgent) {
  if (treeAgent.type === AgentType.EMPTY) return 0;
  if (treeAgent.type === AgentType.LEAF) return 1;
  
  const left = treeAgent.auxiliaryPorts[0].getConnectedAgent();
  const right = treeAgent.auxiliaryPorts[1].getConnectedAgent();
  
  return this.treeSize(left) + this.treeSize(right);
}
```

---

## GPU Encoding

### Tree Structure on GPU
```c
struct TreeNode {
  uint32_t type;       // EMPTY, LEAF, NODE
  uint32_t left_id;    // Child index (for NODE)
  uint32_t right_id;   // Child index (for NODE)
  uint32_t data;       // Value (for LEAF)
};

// Tree array in global memory
__global__ TreeNode tree_nodes[MAX_NODES];
```

### Parallel Map on GPU
```c
__global__ void parallel_map(
  TreeNode* input,
  TreeNode* output,
  int n,
  Func f
) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (tid < n) {
    TreeNode node = input[tid];
    
    if (node.type == LEAF) {
      output[tid].type = LEAF;
      output[tid].data = f(node.data);
    } else if (node.type == NODE) {
      output[tid].type = NODE;
      output[tid].left_id = node.left_id;
      output[tid].right_id = node.right_id;
    } else {
      output[tid].type = EMPTY;
    }
  }
}
```

### Parallel Fold on GPU
```c
__global__ void parallel_fold_step(
  TreeNode* tree,
  uint32_t* results,
  int level,
  int n,
  Func combine
) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (tid < n) {
    TreeNode node = tree[tid];
    
    if (node.type == LEAF) {
      results[tid] = node.data;
    } else if (node.type == NODE && level_matches(node, level)) {
      // Wait for children results
      __syncthreads();
      
      uint32_t left = results[node.left_id];
      uint32_t right = results[node.right_id];
      results[tid] = combine(left, right);
    }
  }
}

// Launch kernel for each level (log n steps)
for (int level = 0; level < max_depth; level++) {
  parallel_fold_step<<<blocks, threads>>>(
    tree, results, level, n, combine_func
  );
  cudaDeviceSynchronize();
}
```

---

## Testing Strategy

### Test Suite 2.1: Basic Trees
```zapp
Empty                              # => Empty
Leaf(42)                          # => Leaf(42)
Node(Leaf(1), Leaf(2))            # => Node(Leaf(1), Leaf(2))
```

### Test Suite 2.2: Lists as Trees
```zapp
[]                                 # => Empty
[42]                               # => Leaf(42)
[1, 2, 3, 4]                      # => Balanced tree

size([1, 2, 3, 4])                # => 4
nth([1, 2, 3, 4], 0)              # => {:ok, 1}
nth([1, 2, 3, 4], 2)              # => {:ok, 3}
nth([1, 2], 5)                    # => {:error, "index out of bounds"}
```

### Test Suite 2.3: Bend
```zapp
bend 0 8 λn.n                     # => [0,1,2,3,4,5,6,7]
bend 0 4 λn.n * n                 # => [0,1,4,9]
bend 1 5 λn.n + 10                # => [11,12,13,14]
bend 0 0 λn.n                     # => []
```

### Test Suite 2.4: Fold
```zapp
fold [1,2,3,4] λ(a,b).a + b       # => 10
fold [1,2,3,4] λ(a,b).a * b       # => 24
fold [5,2,8,1] λ(a,b).if a > b then a else b  # => 8
fold [] λ(a,b).a + b              # => 0
```

### Test Suite 2.5: Map
```zapp
map(λx.x * 2, [1,2,3])            # => [2,4,6]
map(λx.x + 1, [])                 # => []
map(λx.x, [42])                   # => [42]
```

### Test Suite 2.6: Filter
```zapp
filter(λx.x > 2, [1,2,3,4])       # => [3,4]
filter(λx.x % 2 == 0, [1,2,3,4,5]) # => [2,4]
filter(λx.:true, [])              # => []
```

### Test Suite 2.7: Complex Operations
```zapp
# Map + fold
let lst = [1, 2, 3, 4] in
let doubled = map(λx.x * 2, lst) in
fold doubled λ(a,b).a + b         # => 20

# Nested
map(λx.fold x λ(a,b).a+b, [[1,2], [3,4]])  # => [3, 7]

# Filter + map
let lst = [1,2,3,4,5] in
let evens = filter(λx.x % 2 == 0, lst) in
map(λx.x * x, evens)              # => [4, 16]
```

### Test Suite 2.8: Pattern Matching
```zapp
match [] with
  [] -> :empty
  [x] -> :single
  _ -> :multiple
end                                # => :empty

match [1, 2, 3] with
  [] -> 0
  [x] -> x
  [x, y] -> x + y
  [x, y, z] -> x + y + z
end                                # => 6

match {:ok, 42} with
  {:ok, value} -> value
  {:error, _} -> 0
end                                # => 42
```

### Test Suite 2.9: Tuples
```zapp
(1, 2)                             # => (1, 2)
(1, 2, 3)                          # => (1, 2, 3)

fst((10, 20))                      # => 10
snd((10, 20))                      # => 20

match (1, 2, 3) with
  (a, b, c) -> a + b + c
end                                # => 6
```

### Test Suite 2.10: Strings
```zapp
"hello"                            # => "hello"
string_length("hello")             # => 5
concat_str("hello", " world")      # => "hello world"
uppercase("hello")                 # => "HELLO"
```

---

## Performance Characteristics

| Operation | Sequential List | Balanced Tree |
|-----------|----------------|---------------|
| Construction | O(n) | O(log n) parallel |
| Map | O(n) | O(log n) parallel |
| Fold | O(n) | O(log n) parallel |
| Filter | O(n) | O(log n) parallel |
| Nth access | O(n) | O(log n) |
| Size | O(n) | O(log n) parallel |
| Concat | O(n) | O(1) (lazy) |

**GPU Speedup Example:**
- 1M element list: 1,000,000 sequential steps
- 1M element tree: ~20 parallel steps (log₂ 1M ≈ 20)
- **50,000x faster on GPU!**

---

## Deliverables
- ✅ Binary tree agent types (EMPTY, LEAF, NODE)
- ✅ List syntax → balanced tree encoding
- ✅ Bend for parallel construction
- ✅ Fold for parallel reduction
- ✅ Map implementation (parallel)
- ✅ Filter implementation (parallel)
- ✅ Standard tree operations
- ✅ Tuples as trees
- ✅ Strings as trees
- ✅ Pattern matching on trees
- ✅ Tree rebalancing
- ✅ GPU encoding
- ✅ Comprehensive test suite

## Success Criteria
- All list operations work through tree encoding
- Operations are naturally parallel
- GPU can execute tree operations efficiently
- Users don't need to think about trees (transparent)
- Pattern matching works on all tree structures

## Estimated Time
3-4 weeks

---

**Key Insight:** Everything is a tree. This single decision gives us:
1. Automatic GPU parallelism
2. O(log n) operations everywhere
3. Consistent mental model
4. No special cases

**Next Phase:** Phase 3 - Module System (tree-based storage),b).a + b       # => 10
fold [1,2,3,4] λ(a