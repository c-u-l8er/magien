# Zapp Phase 3: Module System

## Goal
Module system built on tree foundation, with result types for error handling.

## Prerequisites
- Phase 1 complete (atoms, result types)
- Phase 2 complete (trees, bend/fold)

## Philosophy
- Module metadata stored as trees
- Import/export lists as trees
- Error handling uses `{:ok, _}` / `{:error, _}` (no exceptions)
- Parallel module lookup

---

## Tasks

### 3.1 Module Structure

Everything in a module is stored as trees.

```javascript
class Module {
  constructor(name, parent = null) {
    this.name = name;
    this.parent = parent;
    
    // Everything stored as balanced trees!
    this.exports = null;      // Tree of {name, definition} pairs
    this.private = null;      // Tree of private definitions
    this.imports = null;      // Tree of imported modules
    this.metadata = null;     // Tree of attributes
  }
  
  // Add export - maintain balanced tree
  addExport(name, definition) {
    const pair = this.net.createNode(
      this.net.createAtom(name),
      definition
    );
    
    if (this.exports === null) {
      this.exports = this.net.createLeaf(pair);
    } else {
      // Insert maintaining balance
      this.exports = this.insertIntoTree(this.exports, pair);
    }
  }
  
  // Lookup - O(log n) in balanced tree
  lookupExport(name) {
    return this.searchTree(this.exports, name);
  }
  
  insertIntoTree(tree, item) {
    // Simplified - should maintain balance
    if (tree.type === AgentType.EMPTY) {
      return this.net.createLeaf(item);
    }
    
    if (tree.type === AgentType.LEAF) {
      return this.net.createNode(tree, this.net.createLeaf(item));
    }
    
    // Insert into left or right based on comparison
    // Then rebalance if needed
  }
  
  searchTree(tree, name) {
    if (!tree || tree.type === AgentType.EMPTY) {
      return null;
    }
    
    if (tree.type === AgentType.LEAF) {
      const pair = tree.data;
      const pairName = pair.auxiliaryPorts[0].getConnectedAgent().data;
      if (pairName === name) {
        return pair.auxiliaryPorts[1].getConnectedAgent();
      }
      return null;
    }
    
    // Search both sides (can be parallel on GPU)
    const left = tree.auxiliaryPorts[0].getConnectedAgent();
    const right = tree.auxiliaryPorts[1].getConnectedAgent();
    
    const leftResult = this.searchTree(left, name);
    if (leftResult) return leftResult;
    
    return this.searchTree(right, name);
  }
}

class ModuleSystem {
  constructor() {
    this.modules = new Map();    // name -> Module
    this.currentModule = null;
    this.net = null;             // Reference to interaction net
  }
  
  defineModule(name, parent = null) {
    const module = new Module(name, parent);
    module.net = this.net;
    this.modules.set(name, module);
    return module;
  }
  
  getModule(name) {
    if (this.modules.has(name)) {
      return this.net.createOk(this.modules.get(name));
    }
    return this.net.createError(`Module ${name} not found`);
  }
}
```

---

### 3.2 Module Definition Syntax

```javascript
// In NetParser:
parseModule(tokens, startIndex) {
  let i = startIndex + 1; // Skip 'module'
  
  // Parse module name
  if (tokens[i].type !== 'IDENTIFIER') {
    throw new Error('Expected module name');
  }
  const moduleName = tokens[i].value;
  i++;
  
  // Expect 'do'
  if (tokens[i].value !== 'do') {
    throw new Error('Expected "do" after module name');
  }
  i++;
  
  // Create module
  const module = this.moduleSystem.defineModule(moduleName);
  const previousModule = this.moduleSystem.currentModule;
  this.moduleSystem.currentModule = module;
  
  // Parse module body
  const endPos = this.findKeyword(tokens, i, 'end');
  const bodyTokens = tokens.slice(i, endPos);
  
  // Parse definitions in module context
  this.parseModuleBody(bodyTokens, module);
  
  // Restore previous module
  this.moduleSystem.currentModule = previousModule;
  
  return endPos + 1;
}

parseModuleBody(tokens, module) {
  let i = 0;
  
  while (i < tokens.length) {
    const token = tokens[i];
    
    if (token.value === 'def') {
      i = this.parseFunctionDefinition(tokens, i, module);
    } else if (token.value === 'private') {
      i = this.parsePrivateDefinition(tokens, i, module);
    } else if (token.value === '@') {
      i = this.parseModuleAttribute(tokens, i, module);
    } else {
      i++;
    }
  }
}

parseFunctionDefinition(tokens, startIndex, module) {
  let i = startIndex + 1; // Skip 'def'
  
  // Parse function name
  const fnName = tokens[i].value;
  i++;
  
  // Parse parameters
  const params = [];
  if (tokens[i].type === '(') {
    i++;
    while (tokens[i].type !== ')') {
      if (tokens[i].type === 'IDENTIFIER') {
        params.push(tokens[i].value);
      }
      i++;
    }
    i++; // Skip ')'
  }
  
  // Parse '='
  if (tokens[i].value !== '=') {
    throw new Error('Expected = after function signature');
  }
  i++;
  
  // Parse function body
  const bodyEnd = this.findFunctionBodyEnd(tokens, i);
  const bodyTokens = tokens.slice(i, bodyEnd);
  
  // Create function as lambda
  const beforeCount = this.net.agents.size;
  let fnAgent;
  
  if (params.length === 0) {
    // No params - just evaluate body
    this.parseTokens(bodyTokens);
    const bodyAgents = Array.from(this.net.agents.values()).slice(beforeCount);
    fnAgent = bodyAgents[bodyAgents.length - 1];
  } else {
    // Build nested lambdas for currying
    fnAgent = this.buildCurriedFunction(params, bodyTokens);
  }
  
  // Add to module exports
  module.addExport(fnName, fnAgent);
  
  return bodyEnd;
}

buildCurriedFunction(params, bodyTokens) {
  // Build λp1.λp2.λp3.body
  const beforeCount = this.net.agents.size;
  this.parseTokens(bodyTokens);
  const bodyAgents = Array.from(this.net.agents.values()).slice(beforeCount);
  let body = bodyAgents[bodyAgents.length - 1];
  
  // Wrap in lambdas (right to left)
  for (let i = params.length - 1; i >= 0; i--) {
    const lambda = this.net.createAgent(AgentType.LAM, 1, params[i]);
    this.net.connectPorts(lambda.auxiliaryPorts[0], body.principalPort);
    body = lambda;
  }
  
  return body;
}
```

**Syntax:**
```zapp
module Math do
  # Public function
  def add(x, y) = x + y
  
  # Function with result type
  def divide(x, y) =
    if y == 0 then
      {:error, "division by zero"}
    else
      {:ok, x / y}
    end
  
  # Private helper
  private def validate(x) =
    if x < 0 then
      {:error, "negative number"}
    else
      {:ok, x}
    end
end
```

---

### 3.3 Import System

```javascript
parseImport(tokens, startIndex) {
  let i = startIndex + 1; // Skip 'import'
  
  // Parse module name
  const moduleName = tokens[i].value;
  i++;
  
  let alias = null;
  let selectiveImports = null;
  
  // Check for 'as' (alias)
  if (i < tokens.length && tokens[i].value === 'as') {
    i++;
    alias = tokens[i].value;
    i++;
  }
  
  // Check for selective imports [func1, func2]
  if (i < tokens.length && tokens[i].type === '[') {
    i++;
    selectiveImports = [];
    
    while (tokens[i].type !== ']') {
      if (tokens[i].type === 'IDENTIFIER') {
        selectiveImports.push(tokens[i].value);
      }
      i++;
    }
    i++; // Skip ']'
  }
  
  // Load module
  const module = this.moduleSystem.getModule(moduleName);
  
  // Add to current module's imports
  const currentModule = this.moduleSystem.currentModule;
  if (currentModule) {
    const importName = alias || moduleName;
    currentModule.addImport(importName, module, selectiveImports);
  }
  
  return i;
}
```

**Syntax:**
```zapp
# Import entire module
import Math

Math.add(5, 3)

# Import with alias
import Math as M

M.add(5, 3)

# Selective import
import Math [add, divide]

add(5, 3)
divide(10, 2)
```

---

### 3.4 Module Attributes

```javascript
parseModuleAttribute(tokens, startIndex) {
  let i = startIndex + 1; // Skip '@'
  
  // Parse attribute name
  const attrName = tokens[i].value;
  i++;
  
  // Parse attribute value (string or expression)
  let attrValue;
  if (tokens[i].type === 'STRING') {
    attrValue = this.net.createAgent(AgentType.STR, 0, tokens[i].value);
    i++;
  } else {
    // Parse expression
    const exprEnd = this.findExpressionEnd(tokens, i);
    const exprTokens = tokens.slice(i, exprEnd);
    
    const beforeCount = this.net.agents.size;
    this.parseTokens(exprTokens);
    const agents = Array.from(this.net.agents.values()).slice(beforeCount);
    attrValue = agents[agents.length - 1];
    
    i = exprEnd;
  }
  
  // Add to current module
  const module = this.moduleSystem.currentModule;
  if (module) {
    module.addAttribute(attrName, attrValue);
  }
  
  return i;
}
```

**Syntax:**
```zapp
module Math do
  @doc "Mathematical operations"
  @version "1.0.0"
  @author "Zapp Team"
  
  def add(x, y) = x + y
end

# Access attributes
Math.@doc         # => "Mathematical operations"
Math.@version     # => "1.0.0"
```

---

### 3.5 Nested Modules

```javascript
// Nested modules are just modules with parent references
parseModule(tokens, startIndex) {
  // ... (same as before, but check for parent)
  
  const parent = this.moduleSystem.currentModule;
  const module = this.moduleSystem.defineModule(moduleName, parent);
  
  // ... rest of parsing
}
```

**Syntax:**
```zapp
module Collections do
  module List do
    def map(f, lst) =
      match lst with
        [] -> []
        [h :: t] -> [f(h) :: map(f, t)]
      end
  end
  
  module Tree do
    def insert(x, tree) = ...
  end
end

# Usage
import Collections.List

List.map(λx.x * 2, [1, 2, 3])
```

---

### 3.6 Module Loading with Error Handling

```javascript
class ModuleLoader {
  constructor(moduleSystem, net) {
    this.moduleSystem = moduleSystem;
    this.net = net;
    this.searchPaths = ['./stdlib', './lib', './'];
  }
  
  load(moduleName) {
    // Try to resolve module path
    const pathResult = this.resolve(moduleName);
    
    // Check if resolution succeeded
    if (this.net.isError(pathResult)) {
      return pathResult;  // Return error
    }
    
    // Extract path from {:ok, path}
    const path = pathResult.auxiliaryPorts[1].getConnectedAgent().data;
    
    // Try to read file
    try {
      const source = this.readFile(path);
      
      // Parse and compile
      const module = this.parseAndCompile(source, moduleName);
      
      return this.net.createOk(module);
    } catch (e) {
      return this.net.createError(`Failed to load ${moduleName}: ${e.message}`);
    }
  }
  
  resolve(moduleName) {
    for (const searchPath of this.searchPaths) {
      const fullPath = `${searchPath}/${moduleName}.zapp`;
      
      if (this.fileExists(fullPath)) {
        return this.net.createOk(fullPath);
      }
    }
    
    return this.net.createError(`Module ${moduleName} not found`);
  }
  
  parseAndCompile(source, moduleName) {
    const parser = new NetParser();
    parser.moduleSystem = this.moduleSystem;
    parser.net = this.net;
    
    const tokens = tokenizeZapp(source);
    parser.parse(tokens);
    
    return this.moduleSystem.getModule(moduleName);
  }
}
```

**Usage:**
```zapp
# Module loading returns result type
match Module.load("Math") with
  {:ok, mod} -> print("Loaded!")
  {:error, msg} -> print("Error: " ++ msg)
end
```

---

### 3.7 Standard Library Organization

```zapp
# Prelude (auto-imported in every file)
module Prelude do
  # Result type helpers
  def ok(x) = {:ok, x}
  def error(msg) = {:error, msg}
  
  # Core operations (from Phase 2)
  def map(f, tree) = ...
  def fold(f, tree) = ...
  def filter(pred, tree) = ...
  def size(tree) = ...
end

# Tree operations
module Tree do
  def bend(start, end, f) = ...
  def fold(tree, combine) = ...
  def map(f, tree) = ...
  def filter(pred, tree) = ...
  def size(tree) = ...
  def nth(tree, n) = ...
  def concat(t1, t2) = ...
  def reverse(tree) = ...
  def take(n, tree) = ...
  def drop(n, tree) = ...
end

# Result handling
module Result do
  def is_ok(result) =
    match result with
      {:ok, _} -> :true
      _ -> :false
    end
  
  def is_error(result) =
    match result with
      {:error, _} -> :true
      _ -> :false
    end
  
  def unwrap_or(result, default) =
    match result with
      {:ok, value} -> value
      {:error, _} -> default
    end
  
  def unwrap(result) =
    match result with
      {:ok, value} -> value
      {:error, msg} -> error("unwrap failed: " ++ msg)
    end
  
  def map(result, f) =
    match result with
      {:ok, value} -> {:ok, f(value)}
      {:error, msg} -> {:error, msg}
    end
  
  def and_then(result, f) =
    match result with
      {:ok, value} -> f(value)
      {:error, msg} -> {:error, msg}
    end
end

# String operations
module String do
  def length(s) = size(s)
  def concat(s1, s2) = Tree.concat(s1, s2)
  def uppercase(s) = map(char_upper, s)
  def lowercase(s) = map(char_lower, s)
  def split(sep, s) = ...
  def trim(s) = ...
end

# I/O operations
module IO do
  def print(x) = ...
  def println(x) = print(x ++ "\n")
  def read_line() = ...
  def read_file(path) = ...
  def write_file(path, content) = ...
end

# Math operations
module Math do
  def abs(x) = if x < 0 then -x else x
  def max(a, b) = if a > b then a else b
  def min(a, b) = if a < b then a else b
  def pow(x, n) = ...
  def sqrt(x) = ...
end
```

---

### 3.8 Visibility Control

```javascript
class Module {
  // ... existing code ...
  
  addPrivate(name, definition) {
    const pair = this.net.createNode(
      this.net.createAtom(name),
      definition
    );
    
    if (this.private === null) {
      this.private = this.net.createLeaf(pair);
    } else {
      this.private = this.insertIntoTree(this.private, pair);
    }
  }
  
  canAccess(name, fromModule) {
    // Can always access own private members
    if (fromModule === this) {
      const privateResult = this.searchTree(this.private, name);
      if (privateResult) return true;
    }
    
    // Check exports
    const exportResult = this.searchTree(this.exports, name);
    return exportResult !== null;
  }
}
```

**Syntax:**
```zapp
module Example do
  # Public (default)
  def public_func(x) = x + 1
  
  # Explicitly private
  private def helper(x) = x * 2
  
  def use_helper(x) = helper(x)  # OK: same module
end

import Example

Example.public_func(5)    # => 6
Example.use_helper(5)     # => 10
Example.helper(5)         # Error: helper is private
```

---

## GPU Encoding

### Module Exports Tree on GPU
```c
// Module exports stored as balanced tree in constant memory
struct ModuleExport {
  uint32_t name_id;      // Atom ID for function name
  uint32_t func_id;      // Function agent ID
};

__constant__ TreeNode module_exports[MAX_MODULES][MAX_EXPORTS];

// Parallel lookup (O(log n) on GPU)
__device__ Result lookup_function(
  uint32_t module_id,
  uint32_t func_name_id
) {
  TreeNode* exports = module_exports[module_id];
  
  // Binary search in tree (parallel-friendly)
  TreeNode* node = search_tree_parallel(exports, func_name_id);
  
  if (node == NULL) {
    return Result{ERROR, 0};
  }
  
  return Result{OK, node->data};
}

__device__ TreeNode* search_tree_parallel(
  TreeNode* tree,
  uint32_t target_id
) {
  if (tree->type == EMPTY) return NULL;
  
  if (tree->type == LEAF) {
    ModuleExport* exp = (ModuleExport*)tree->data;
    return (exp->name_id == target_id) ? tree : NULL;
  }
  
  // Search both sides in parallel
  TreeNode* left_result;
  TreeNode* right_result;
  
  // Launch parallel searches
  #pragma omp parallel sections
  {
    #pragma omp section
    left_result = search_tree_parallel(tree->left, target_id);
    
    #pragma omp section  
    right_result = search_tree_parallel(tree->right, target_id);
  }
  
  return left_result ? left_result : right_result;
}
```

---

## Testing Strategy

### Test Suite 3.1: Basic Modules
```zapp
module Math do
  def add(x, y) = x + y
  def mult(x, y) = x * y
end

import Math

Math.add(5, 3)             # => 8
Math.mult(4, 5)            # => 20
```

### Test Suite 3.2: Result Types in Modules
```zapp
module SafeMath do
  def divide(x, y) =
    if y == 0 then
      {:error, "division by zero"}
    else
      {:ok, x / y}
    end
end

import SafeMath

match SafeMath.divide(10, 2) with
  {:ok, result} -> result     # => 5
  {:error, msg} -> 0
end

match SafeMath.divide(10, 0) with
  {:ok, result} -> result
  {:error, msg} -> print(msg)  # Prints "division by zero"
end
```

### Test Suite 3.3: Selective Imports
```zapp
module Utils do
  def func1(x) = x + 1
  def func2(x) = x * 2
  def func3(x) = x - 1
end

import Utils [func1, func2]

func1(5)                   # => 6
func2(5)                   # => 10
func3(5)                   # Error: func3 not imported
```

### Test Suite 3.4: Nested Modules
```zapp
module Collections do
  module List do
    def length(lst) = size(lst)
  end
  
  module Tree do
    def depth(tree) = ...
  end
end

import Collections.List

List.length([1, 2, 3])     # => 3
```

### Test Suite 3.5: Module Aliases
```zapp
module VeryLongModuleName do
  def func(x) = x + 1
end

import VeryLongModuleName as Short

Short.func(5)              # => 6
```

### Test Suite 3.6: Private Functions
```zapp
module Example do
  def public(x) = helper(x) * 2
  
  private def helper(x) = x + 1
end

import Example

Example.public(5)          # => 12
Example.helper(5)          # Error: helper is private
```

### Test Suite 3.7: Module Attributes
```zapp
module Math do
  @doc "Math operations"
  @version "1.0"
  
  def add(x, y) = x + y
end

Math.@doc                  # => "Math operations"
Math.@version              # => "1.0"
```

### Test Suite 3.8: Module Loading
```zapp
# Load module dynamically
match Module.load("Math") with
  {:ok, _} -> print("Loaded successfully")
  {:error, msg} -> print("Failed: " ++ msg)
end
```

---

## Deliverables
- ✅ Module definition syntax
- ✅ Tree-based storage for exports/imports
- ✅ Result types for error handling
- ✅ Nested modules
- ✅ Selective imports
- ✅ Module aliases
- ✅ Private/public visibility
- ✅ Module attributes
- ✅ Standard library organization
- ✅ Module loading with results
- ✅ GPU-compatible lookups
- ✅ Comprehensive test suite

## Success Criteria
- Modules can be defined and imported
- All module metadata stored as trees
- Error handling uses result types
- Module lookup is O(log n) parallel-friendly
- Private/public visibility enforced
- Standard library well-organized

## Estimated Time
2-3 weeks

---

**Key Changes from Original:**
1. Everything stored as trees (exports, imports, attributes)
2. Error handling uses result types, not exceptions
3. Parallel-friendly lookups (O(log n))
4. GPU-compatible design

**Next Phase:** Phase 4 - Meta-Programming (tree-based AST manipulation)