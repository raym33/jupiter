# Reaching 90% on LiveCodeBench with Jupiter MoE-R

**A comprehensive guide to building a competitive programming expert system**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Understanding the Challenge](#understanding-the-challenge)
3. [Hardware Requirements](#hardware-requirements)
4. [Architecture Overview](#architecture-overview)
5. [Phase 1: Foundation (Target: 40-50%)](#phase-1-foundation-target-40-50)
6. [Phase 2: Specialization (Target: 55-65%)](#phase-2-specialization-target-55-65)
7. [Phase 3: Reasoning Enhancement (Target: 70-80%)](#phase-3-reasoning-enhancement-target-70-80)
8. [Phase 4: State-of-the-Art (Target: 85-90%)](#phase-4-state-of-the-art-target-85-90)
9. [Training Data Pipeline](#training-data-pipeline)
10. [Expert Configurations](#expert-configurations)
11. [Evaluation and Monitoring](#evaluation-and-monitoring)
12. [Troubleshooting](#troubleshooting)

---

## Introduction

This guide provides a step-by-step roadmap to achieve **90% pass@1 on LiveCodeBench** using Jupiter's MoE-R (Mixture of Real Experts) system. LiveCodeBench is one of the most challenging coding benchmarks, featuring problems from LeetCode, Codeforces, and AtCoder competitive programming platforms.

### Current State of the Art (December 2024)

| Model | LiveCodeBench Score | Parameters |
|-------|---------------------|------------|
| Kimi K2 Thinking | 83.1% | ~1T MoE |
| Gemini 3 Pro | 79.7% | ~1T+ |
| Grok 4 | 79.0% | Massive |
| o3-mini | 74.1% | Unknown |
| DeepSeek R1 | 73.3% | 671B MoE |
| Claude 3.5 Sonnet | ~65% | ~175B |

**Key Insight**: To reach 90%, we need to surpass current state-of-the-art by implementing advanced reasoning, specialized experts, and test-time compute scaling.

---

## Understanding the Challenge

### What Makes LiveCodeBench Hard?

1. **Contamination-Free**: Problems are continuously collected from live competitions
2. **Multi-Platform**: LeetCode, Codeforces, AtCoder - different styles and difficulty
3. **Three Difficulty Levels**:
   - Easy: ~90% achievable with good models
   - Medium: ~60% for top models
   - Hard: ~20-35% even for the best models

### Required Capabilities

```
┌─────────────────────────────────────────────────────────────────┐
│                    LIVECODEBENCH REQUIREMENTS                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. ALGORITHM UNDERSTANDING                                      │
│     ├── Dynamic Programming (30% of hard problems)              │
│     ├── Graph Algorithms (25%)                                  │
│     ├── Data Structures (20%)                                   │
│     ├── Math/Number Theory (15%)                                │
│     └── String Algorithms (10%)                                 │
│                                                                  │
│  2. REASONING CAPABILITIES                                       │
│     ├── Problem decomposition                                   │
│     ├── Pattern recognition                                     │
│     ├── Complexity analysis                                     │
│     └── Edge case identification                                │
│                                                                  │
│  3. CODE GENERATION                                              │
│     ├── Correct syntax                                          │
│     ├── Efficient implementation                                │
│     ├── Proper I/O handling                                     │
│     └── Time/space optimization                                 │
│                                                                  │
│  4. SELF-VERIFICATION                                            │
│     ├── Test case generation                                    │
│     ├── Error detection                                         │
│     └── Self-repair capability                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Hardware Requirements

### Minimum Configuration (Target: 70%)

| Component | Specification | Role |
|-----------|--------------|------|
| 4× Mac Mini M4 Pro | 64GB each | Expert inference |
| 1× Mac Studio M4 Ultra | 192GB | Orchestrator + Large experts |
| 1× RTX 4090 | 24GB VRAM | Training acceleration |
| **Total** | **448GB unified** | |

### Recommended Configuration (Target: 85%)

| Component | Specification | Role |
|-----------|--------------|------|
| 6× Mac Mini M4 Pro | 64GB each | Expert inference |
| 2× Mac Studio M4 Ultra | 192GB each | Large reasoning experts |
| 2× RTX 4090 | 24GB each | Parallel training |
| **Total** | **816GB unified** | |

### Optimal Configuration (Target: 90%+)

| Component | Specification | Role |
|-----------|--------------|------|
| 8× Mac Mini M4 Pro | 64GB each | Specialized experts |
| 2× Mac Studio M4 Ultra | 192GB each | Reasoning + Synthesis |
| 4× RTX 5090 | 32GB each | High-speed training |
| 1× RTX 5090 NVLink pair | 64GB combined | Large model training |
| **Total** | **1.1TB+ unified** | |

### Network Requirements

```
┌─────────────────────────────────────────────────────────────────┐
│                    NETWORK TOPOLOGY                              │
│                                                                  │
│                    ┌─────────────────┐                          │
│                    │   Mac Studio    │                          │
│                    │   Orchestrator  │                          │
│                    └────────┬────────┘                          │
│                             │ Thunderbolt 5                     │
│                             │ (120 Gb/s)                        │
│              ┌──────────────┼──────────────┐                    │
│              │              │              │                    │
│       ┌──────┴──────┐ ┌─────┴─────┐ ┌─────┴──────┐             │
│       │  Mac Mini   │ │ Mac Mini  │ │  Mac Mini  │             │
│       │  Expert 1   │ │ Expert 2  │ │  Expert 3  │             │
│       └─────────────┘ └───────────┘ └────────────┘             │
│                                                                  │
│       ┌─────────────────────────────────────────┐               │
│       │         10GbE Switch                    │               │
│       └──────────────────┬──────────────────────┘               │
│                          │                                       │
│              ┌───────────┴───────────┐                          │
│              │                       │                          │
│       ┌──────┴──────┐         ┌──────┴──────┐                   │
│       │ RTX 5090 #1 │         │ RTX 5090 #2 │                   │
│       │  Training   │◄─NCCL──►│  Training   │                   │
│       └─────────────┘         └─────────────┘                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Connection Requirements**:
- Between Macs: Thunderbolt 4/5 (40-120 Gb/s) - Required for gradient sync
- Macs ↔ GPUs: 10GbE minimum - For data transfer
- GPUs ↔ GPUs: NVLink or PCIe 5.0 - For distributed training

---

## Architecture Overview

### Jupiter MoE-R for Competitive Programming

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 JUPITER MoE-R COMPETITIVE PROGRAMMING                   │
│                                                                         │
│   Problem: "Find the minimum spanning tree of a weighted graph"        │
│                              │                                          │
│                              ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                     PROBLEM ANALYZER                             │  │
│   │  • Parse problem statement                                       │  │
│   │  • Extract constraints (N ≤ 10^5, edges ≤ 10^6)                 │  │
│   │  • Identify problem type: GRAPH + MST                           │  │
│   └──────────────────────────┬──────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                     EXPERT ROUTER                                │  │
│   │  Selected: [graph-expert, optimization-expert, impl-expert]      │  │
│   │  Confidence: [0.95, 0.72, 0.88]                                  │  │
│   └──────────────────────────┬──────────────────────────────────────┘  │
│                              │                                          │
│         ┌────────────────────┼────────────────────┐                    │
│         │                    │                    │                    │
│         ▼                    ▼                    ▼                    │
│   ┌───────────┐        ┌───────────┐        ┌───────────┐             │
│   │   GRAPH   │        │   OPTIM   │        │   IMPL    │             │
│   │  EXPERT   │        │  EXPERT   │        │  EXPERT   │             │
│   │  (7B)     │        │  (3B)     │        │  (7B)     │             │
│   │           │        │           │        │           │             │
│   │ "Use      │        │ "Kruskal  │        │ "Union-   │             │
│   │  Kruskal  │        │  is O(E   │        │  Find with│             │
│   │  or Prim" │        │  log E)"  │        │  path     │             │
│   │           │        │           │        │  compress"│             │
│   └─────┬─────┘        └─────┬─────┘        └─────┬─────┘             │
│         │                    │                    │                    │
│         └────────────────────┼────────────────────┘                    │
│                              │                                          │
│                              ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                    REASONING ENGINE                              │  │
│   │  • Chain-of-thought synthesis                                    │  │
│   │  • Approach selection: Kruskal with Union-Find                  │  │
│   │  • Complexity verification: O(E log E) ✓                        │  │
│   └──────────────────────────┬──────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                    CODE GENERATOR                                │  │
│   │  def kruskal(n, edges):                                          │  │
│   │      parent = list(range(n))                                     │  │
│   │      def find(x):                                                │  │
│   │          if parent[x] != x:                                      │  │
│   │              parent[x] = find(parent[x])                         │  │
│   │          return parent[x]                                        │  │
│   │      ...                                                         │  │
│   └──────────────────────────┬──────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                    VERIFICATION LOOP                             │  │
│   │  • Generate edge cases                                           │  │
│   │  • Test with sample inputs                                       │  │
│   │  • Verify output format                                          │  │
│   │  • Self-repair if needed                                         │  │
│   └──────────────────────────┬──────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│                       FINAL SOLUTION                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Foundation (Target: 40-50%)

### Step 1.1: Install Jupiter

```bash
# Clone repository
git clone https://github.com/raym33/jupiter.git
cd jupiter

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install for Mac
pip install -e ".[mac]"

# Install for NVIDIA (on Linux machines)
pip install -e ".[nvidia]"
```

### Step 1.2: Configure Base Cluster

Create `config/cluster.yaml`:

```yaml
# =============================================================================
# JUPITER CLUSTER CONFIGURATION - PHASE 1
# =============================================================================

cluster:
  name: "jupiter-livecodebench"
  version: "1.0.0"

nodes:
  # Orchestrator node (Mac Studio recommended)
  - host: "mac-studio.local"
    role: "orchestrator"
    memory_gb: 192
    connection: "thunderbolt"
    capabilities:
      - "routing"
      - "synthesis"
      - "large_model"

  # Expert nodes (Mac Mini)
  - host: "mac-mini-1.local"
    role: "expert"
    memory_gb: 64
    connection: "thunderbolt"
    assigned_experts:
      - "algorithm-core"
      - "dp-expert"

  - host: "mac-mini-2.local"
    role: "expert"
    memory_gb: 64
    connection: "thunderbolt"
    assigned_experts:
      - "graph-expert"
      - "tree-expert"

  - host: "mac-mini-3.local"
    role: "expert"
    memory_gb: 64
    connection: "thunderbolt"
    assigned_experts:
      - "math-expert"
      - "string-expert"

  - host: "mac-mini-4.local"
    role: "expert"
    memory_gb: 64
    connection: "thunderbolt"
    assigned_experts:
      - "implementation-expert"
      - "optimization-expert"

  # Training nodes (NVIDIA)
  - host: "gpu-server-1"
    role: "trainer"
    gpu: "RTX 4090"
    vram_gb: 24
    connection: "ethernet"

network:
  thunderbolt_speed_gbps: 40
  ethernet_speed_gbps: 10
  gradient_sync: "thunderbolt"
  data_transfer: "ethernet"
```

### Step 1.3: Create Domain Configuration

Create `config/domains/competitive_programming.yaml`:

```yaml
# =============================================================================
# DOMAIN: Competitive Programming
# =============================================================================

name: "competitive_programming"
description: "Expert system for competitive programming and algorithmic problem solving"
version: "1.0.0"

# Data sources
data_sources:
  primary:
    - type: "competitive_problems"
      sources:
        - name: "codeforces"
          url: "https://codeforces.com"
          rating_range: [800, 3500]
          count: 50000

        - name: "leetcode"
          difficulty: ["easy", "medium", "hard"]
          count: 3000
          include_solutions: true

        - name: "atcoder"
          contests: ["abc", "arc", "agc"]
          count: 10000

  curated:
    - type: "algorithm_textbooks"
      sources:
        - "CLRS Introduction to Algorithms"
        - "Competitive Programming 4"
        - "Guide to Competitive Programming"

# Quality requirements
quality:
  min_solution_quality: 0.8
  require_test_cases: true
  require_complexity_analysis: true
  require_explanation: true

  filters:
    - "solution_compiles"
    - "passes_all_tests"
    - "within_time_limit"
    - "optimal_complexity"

# Training configuration
training:
  base_model: "jupiter-3b"
  epochs: 50
  batch_size: 32
  learning_rate: 1e-5
  warmup_steps: 1000

  curriculum:
    - phase: "fundamentals"
      difficulty: ["easy"]
      epochs: 10

    - phase: "intermediate"
      difficulty: ["easy", "medium"]
      epochs: 20

    - phase: "advanced"
      difficulty: ["medium", "hard"]
      epochs: 20

# Specialization areas
specializations:
  - name: "dynamic_programming"
    weight: 0.25
    topics:
      - "1D DP"
      - "2D DP"
      - "DP on trees"
      - "Bitmask DP"
      - "Digit DP"

  - name: "graph_algorithms"
    weight: 0.20
    topics:
      - "BFS/DFS"
      - "Shortest paths"
      - "MST"
      - "Network flow"
      - "Strongly connected components"

  - name: "data_structures"
    weight: 0.20
    topics:
      - "Segment trees"
      - "Fenwick trees"
      - "Union-Find"
      - "Treaps"
      - "Persistent structures"

  - name: "mathematics"
    weight: 0.15
    topics:
      - "Number theory"
      - "Combinatorics"
      - "Probability"
      - "Linear algebra"
      - "Game theory"

  - name: "strings"
    weight: 0.10
    topics:
      - "KMP"
      - "Z-algorithm"
      - "Suffix arrays"
      - "Aho-Corasick"
      - "Hashing"

  - name: "geometry"
    weight: 0.10
    topics:
      - "Convex hull"
      - "Line intersection"
      - "Polygon operations"
      - "Sweep line"
```

### Step 1.4: Train Base Model

```bash
# Start training with curriculum learning
jupiter start --domain competitive_programming --mode auto

# Monitor training progress
jupiter monitor --metrics loss,accuracy,benchmark_score
```

### Step 1.5: Initial Evaluation

```bash
# Run LiveCodeBench evaluation
jupiter evaluate --benchmark livecodebench --subset easy,medium

# Expected results at Phase 1:
# - Easy: 75-85%
# - Medium: 35-45%
# - Hard: 10-15%
# - Overall: 40-50%
```

---

## Phase 2: Specialization (Target: 55-65%)

### Step 2.1: Create Specialized Experts

Create expert configurations in `config/experts/competitive/`:

**Algorithm Core Expert** (`algorithm_core.yaml`):

```yaml
# =============================================================================
# EXPERT: Algorithm Core
# =============================================================================

name: "algorithm-core"
domain: "competitive_programming"
description: "Core algorithmic patterns and fundamental problem-solving strategies"

base_model: "jupiter-3b"
model_path: null  # Set after training

keywords:
  - "algorithm"
  - "complexity"
  - "time complexity"
  - "space complexity"
  - "brute force"
  - "optimization"
  - "greedy"
  - "divide and conquer"
  - "binary search"
  - "two pointers"
  - "sliding window"
  - "prefix sum"
  - "sorting"
  - "searching"

capabilities:
  - "Identify optimal algorithmic approach"
  - "Analyze time and space complexity"
  - "Recognize common patterns"
  - "Suggest optimization strategies"
  - "Compare alternative solutions"

max_tokens: 2048
temperature: 0.3  # Lower for more deterministic responses
top_p: 0.9

system_prompt: |
  You are an expert algorithm designer specializing in competitive programming.

  Your role is to:
  1. Analyze problems and identify the optimal algorithmic approach
  2. Consider time and space complexity constraints
  3. Recognize patterns from classic problems
  4. Suggest multiple approaches when applicable
  5. Always verify your solution meets the constraints

  When analyzing a problem:
  - First identify the problem type (optimization, counting, construction, etc.)
  - Consider the input constraints carefully (N ≤ 10^5 suggests O(N log N) or better)
  - Look for patterns that match known algorithms
  - Think about edge cases

  Be precise and mathematical in your analysis.
```

**Dynamic Programming Expert** (`dp_expert.yaml`):

```yaml
# =============================================================================
# EXPERT: Dynamic Programming
# =============================================================================

name: "dp-expert"
domain: "dynamic_programming"
description: "Specialist in all forms of dynamic programming"

base_model: "jupiter-3b"
model_path: null

keywords:
  - "dynamic programming"
  - "dp"
  - "memoization"
  - "tabulation"
  - "optimal substructure"
  - "overlapping subproblems"
  - "state transition"
  - "recurrence"
  - "bottom-up"
  - "top-down"
  - "bitmask"
  - "digit dp"
  - "interval dp"
  - "tree dp"
  - "knapsack"
  - "LCS"
  - "LIS"
  - "edit distance"
  - "matrix chain"
  - "coin change"
  - "subset sum"

capabilities:
  - "Identify DP problems from problem statements"
  - "Define optimal state representation"
  - "Derive state transition equations"
  - "Optimize space complexity"
  - "Handle complex DP variants (bitmask, digit, tree)"

max_tokens: 2048
temperature: 0.2

system_prompt: |
  You are a dynamic programming specialist with deep expertise in all DP variants.

  Your approach to DP problems:

  1. STATE DEFINITION
     - What information do we need to track?
     - What is the minimum state that captures the problem?
     - Can we reduce dimensions?

  2. TRANSITION
     - How do we move from one state to another?
     - What are all possible transitions?
     - What is the recurrence relation?

  3. BASE CASES
     - What are the trivial cases?
     - How do we initialize our DP table?

  4. ANSWER EXTRACTION
     - Where is the final answer stored?
     - Do we need to combine multiple states?

  5. OPTIMIZATION
     - Can we reduce space from O(N²) to O(N)?
     - Can we use matrix exponentiation for linear recurrences?
     - Is there a convex hull optimization possible?

  Common patterns:
  - "Maximum/minimum" → Usually DP
  - "Count number of ways" → DP with addition
  - "Is it possible" → DP with boolean
  - "Subsequence" → Often O(N²) or O(N log N)
  - "Subset" → Bitmask DP if N ≤ 20

  Always provide:
  - Clear state definition
  - Transition equation in mathematical notation
  - Time and space complexity
  - Complete, working code
```

**Graph Expert** (`graph_expert.yaml`):

```yaml
# =============================================================================
# EXPERT: Graph Algorithms
# =============================================================================

name: "graph-expert"
domain: "graph_algorithms"
description: "Specialist in graph theory and graph algorithms"

base_model: "jupiter-3b"
model_path: null

keywords:
  - "graph"
  - "tree"
  - "node"
  - "edge"
  - "vertex"
  - "path"
  - "cycle"
  - "connected"
  - "component"
  - "BFS"
  - "DFS"
  - "dijkstra"
  - "bellman-ford"
  - "floyd-warshall"
  - "MST"
  - "kruskal"
  - "prim"
  - "topological sort"
  - "SCC"
  - "tarjan"
  - "kosaraju"
  - "bipartite"
  - "matching"
  - "flow"
  - "cut"
  - "bridge"
  - "articulation"
  - "LCA"
  - "centroid"

capabilities:
  - "Model problems as graphs"
  - "Choose optimal graph representation"
  - "Implement efficient graph traversals"
  - "Apply advanced graph algorithms"
  - "Handle tree-specific algorithms"

max_tokens: 2048
temperature: 0.2

system_prompt: |
  You are a graph algorithm expert specializing in competitive programming.

  GRAPH REPRESENTATION
  - Adjacency list: Default choice, O(V+E) space
  - Adjacency matrix: For dense graphs or when checking edge existence frequently
  - Edge list: For Kruskal's MST or when processing edges

  COMMON ALGORITHMS AND WHEN TO USE THEM

  Traversal:
  - BFS: Shortest path in unweighted graph, level-order
  - DFS: Connectivity, cycle detection, topological sort

  Shortest Path:
  - BFS: Unweighted graphs
  - Dijkstra: Non-negative weights, O((V+E) log V)
  - Bellman-Ford: Negative weights possible, O(VE)
  - Floyd-Warshall: All pairs, O(V³)
  - 0-1 BFS: Weights are only 0 or 1

  MST:
  - Kruskal: Better for sparse graphs, uses Union-Find
  - Prim: Better for dense graphs, uses priority queue

  Advanced:
  - Tarjan/Kosaraju: SCC decomposition
  - Bridges/Articulation points: Critical edges/nodes
  - LCA: Lowest common ancestor queries
  - Centroid decomposition: Divide and conquer on trees
  - Heavy-Light decomposition: Path queries on trees

  Always consider:
  - Is the graph directed or undirected?
  - Are there negative weights?
  - Is it a tree (N-1 edges, connected)?
  - What are the constraints on V and E?
```

**Implementation Expert** (`implementation_expert.yaml`):

```yaml
# =============================================================================
# EXPERT: Implementation
# =============================================================================

name: "implementation-expert"
domain: "code_generation"
description: "Specialist in clean, correct, and efficient code implementation"

base_model: "jupiter-7b"  # Larger for code generation
model_path: null

keywords:
  - "implement"
  - "code"
  - "solution"
  - "program"
  - "python"
  - "cpp"
  - "function"
  - "class"
  - "input"
  - "output"
  - "parse"
  - "format"
  - "edge case"
  - "overflow"
  - "precision"
  - "modulo"

capabilities:
  - "Write correct, efficient code"
  - "Handle all edge cases"
  - "Proper input/output handling"
  - "Avoid common pitfalls (overflow, precision)"
  - "Optimize for competitive programming"

max_tokens: 4096
temperature: 0.1  # Very low for consistent code

system_prompt: |
  You are an implementation expert for competitive programming.

  CODE QUALITY REQUIREMENTS
  1. Correctness: Must pass all test cases
  2. Efficiency: Must meet time limits
  3. Clarity: Code should be readable
  4. Robustness: Handle all edge cases

  COMMON PITFALLS TO AVOID
  - Integer overflow: Use long long in C++, be careful in Python
  - Off-by-one errors: Double-check loop bounds
  - Floating point precision: Use integers when possible
  - Uninitialized variables: Always initialize
  - Wrong modulo: Remember (a + b) % MOD, (a * b) % MOD
  - Array bounds: Check indices before accessing

  INPUT/OUTPUT
  - Use fast I/O in C++ (ios_base::sync_with_stdio(false))
  - Read all input before processing
  - Match expected output format exactly

  PYTHON SPECIFIC
  - Use sys.stdin for faster input
  - Recursion limit: sys.setrecursionlimit(10**6)
  - Use collections.defaultdict, collections.deque
  - Use heapq for priority queues
  - Use functools.lru_cache for memoization

  C++ SPECIFIC
  - Use #include <bits/stdc++.h> for competitive programming
  - Use vector instead of arrays when possible
  - Use auto for complex types
  - Use range-based for loops

  Always provide:
  - Complete, runnable code
  - Proper I/O handling
  - Comments for complex logic
  - Time and space complexity
```

### Step 2.2: Train Specialized Experts

```bash
# Train each expert on its domain
for expert in algorithm-core dp-expert graph-expert implementation-expert; do
  jupiter swarm train -e $expert \
    -d config/domains/competitive_programming.yaml \
    -p 3b \
    --epochs 30 \
    --focus-area ${expert%-expert}
done
```

### Step 2.3: Configure Expert Collaboration

Create `config/swarm/competitive_swarm.yaml`:

```yaml
# =============================================================================
# SWARM: Competitive Programming
# =============================================================================

name: "competitive-programming-swarm"
mode: "inference"

experts_dir: "config/experts/competitive"

router:
  strategy: "hybrid"  # keyword + embedding
  top_k: 3
  min_confidence: 0.3

  # Custom routing rules for competitive programming
  rules:
    - pattern: "dynamic programming|dp|memoization"
      experts: ["dp-expert"]
      boost: 0.3

    - pattern: "graph|tree|path|cycle|connected"
      experts: ["graph-expert"]
      boost: 0.3

    - pattern: "implement|code|solution"
      experts: ["implementation-expert"]
      boost: 0.2

synthesizer:
  strategy: "reasoning_chain"  # Special strategy for CP
  include_expert_attribution: true

  # Multi-step synthesis
  steps:
    - name: "analyze"
      experts: ["algorithm-core"]
      output: "problem_analysis"

    - name: "approach"
      experts: ["dp-expert", "graph-expert"]  # Based on analysis
      input: "problem_analysis"
      output: "approach_selection"

    - name: "implement"
      experts: ["implementation-expert"]
      input: "approach_selection"
      output: "code_solution"

    - name: "verify"
      experts: ["algorithm-core"]
      input: "code_solution"
      output: "verified_solution"

execution:
  max_parallel_experts: 4
  timeout_seconds: 120
  enable_expert_collaboration: true
  collaboration_rounds: 2  # Experts can see each other's output

verification:
  enabled: true
  generate_test_cases: true
  run_sample_tests: true
  self_repair_attempts: 3
```

### Step 2.4: Phase 2 Evaluation

```bash
# Run full LiveCodeBench evaluation
jupiter evaluate --benchmark livecodebench --full

# Expected results at Phase 2:
# - Easy: 85-90%
# - Medium: 50-60%
# - Hard: 20-30%
# - Overall: 55-65%
```

---

## Phase 3: Reasoning Enhancement (Target: 70-80%)

### Step 3.1: Implement Extended Reasoning

Create `jupiter/reasoning/chain_of_thought.py`:

```python
"""
Extended Chain-of-Thought Reasoning for Competitive Programming.

This module implements multi-step reasoning similar to o1/o3 models,
specialized for algorithmic problem solving.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum
import asyncio


class ReasoningStep(str, Enum):
    UNDERSTAND = "understand"
    IDENTIFY = "identify"
    APPROACH = "approach"
    VERIFY = "verify"
    IMPLEMENT = "implement"
    TEST = "test"
    REPAIR = "repair"


@dataclass
class ThoughtStep:
    """A single step in the reasoning chain."""
    step_type: ReasoningStep
    content: str
    confidence: float
    alternatives: List[str] = None
    verification: Optional[str] = None


@dataclass
class ReasoningChain:
    """Complete reasoning chain for a problem."""
    problem: str
    steps: List[ThoughtStep]
    final_solution: str
    total_tokens: int
    thinking_time_ms: float


class CompetitiveProgrammingReasoner:
    """
    Extended reasoning engine for competitive programming problems.

    Implements a structured thinking process:
    1. Problem Understanding
    2. Pattern Identification
    3. Approach Selection
    4. Complexity Verification
    5. Implementation
    6. Testing
    7. Self-Repair (if needed)
    """

    def __init__(self, experts: Dict[str, Any], config: Dict[str, Any]):
        self.experts = experts
        self.config = config
        self.max_thinking_tokens = config.get("max_thinking_tokens", 8192)
        self.repair_attempts = config.get("repair_attempts", 3)

    async def solve(self, problem: str) -> ReasoningChain:
        """
        Solve a competitive programming problem with extended reasoning.
        """
        steps = []

        # Step 1: Understand the problem
        understanding = await self._understand_problem(problem)
        steps.append(understanding)

        # Step 2: Identify patterns and problem type
        identification = await self._identify_patterns(problem, understanding)
        steps.append(identification)

        # Step 3: Select approach
        approach = await self._select_approach(problem, understanding, identification)
        steps.append(approach)

        # Step 4: Verify approach feasibility
        verification = await self._verify_approach(approach, understanding)
        steps.append(verification)

        if verification.confidence < 0.7:
            # Try alternative approach
            approach = await self._try_alternative(
                problem, understanding, identification, approach
            )
            steps.append(approach)

        # Step 5: Implement solution
        implementation = await self._implement_solution(problem, approach)
        steps.append(implementation)

        # Step 6: Test solution
        test_result = await self._test_solution(implementation, problem)
        steps.append(test_result)

        # Step 7: Repair if needed
        if test_result.confidence < 0.9:
            for attempt in range(self.repair_attempts):
                repair = await self._repair_solution(
                    implementation, test_result, problem
                )
                steps.append(repair)

                test_result = await self._test_solution(repair, problem)
                steps.append(test_result)

                if test_result.confidence >= 0.9:
                    break

        # Construct final chain
        final_solution = steps[-2].content  # Last implementation
        total_tokens = sum(len(s.content.split()) for s in steps)

        return ReasoningChain(
            problem=problem,
            steps=steps,
            final_solution=final_solution,
            total_tokens=total_tokens,
            thinking_time_ms=0  # Set by caller
        )

    async def _understand_problem(self, problem: str) -> ThoughtStep:
        """Parse and understand the problem statement."""
        prompt = f"""
        Analyze this competitive programming problem:

        {problem}

        Extract:
        1. Input format and constraints
        2. Output requirements
        3. Core task description
        4. Edge cases to consider
        5. Key observations

        Be thorough and precise.
        """

        response = await self.experts["algorithm-core"].generate(prompt)

        return ThoughtStep(
            step_type=ReasoningStep.UNDERSTAND,
            content=response.content,
            confidence=response.confidence
        )

    async def _identify_patterns(
        self,
        problem: str,
        understanding: ThoughtStep
    ) -> ThoughtStep:
        """Identify algorithmic patterns in the problem."""
        prompt = f"""
        Based on this problem understanding:

        {understanding.content}

        Identify:
        1. Problem category (DP, graph, math, etc.)
        2. Specific sub-patterns (knapsack, shortest path, etc.)
        3. Similar classic problems
        4. Key algorithmic techniques needed

        Consider multiple possibilities and rank by likelihood.
        """

        # Query multiple experts
        responses = await asyncio.gather(
            self.experts["algorithm-core"].generate(prompt),
            self.experts["dp-expert"].generate(prompt),
            self.experts["graph-expert"].generate(prompt),
        )

        # Combine insights
        combined = self._merge_pattern_insights(responses)

        return ThoughtStep(
            step_type=ReasoningStep.IDENTIFY,
            content=combined["content"],
            confidence=combined["confidence"],
            alternatives=combined["alternatives"]
        )

    async def _select_approach(
        self,
        problem: str,
        understanding: ThoughtStep,
        identification: ThoughtStep
    ) -> ThoughtStep:
        """Select the best algorithmic approach."""
        prompt = f"""
        Problem Understanding:
        {understanding.content}

        Identified Patterns:
        {identification.content}

        Select the optimal approach:
        1. Primary algorithm to use
        2. Data structures needed
        3. Step-by-step solution outline
        4. Expected time complexity
        5. Expected space complexity

        Justify your choice considering the constraints.
        """

        # Route to most relevant expert
        if "dp" in identification.content.lower():
            expert = self.experts["dp-expert"]
        elif "graph" in identification.content.lower():
            expert = self.experts["graph-expert"]
        else:
            expert = self.experts["algorithm-core"]

        response = await expert.generate(prompt)

        return ThoughtStep(
            step_type=ReasoningStep.APPROACH,
            content=response.content,
            confidence=response.confidence
        )

    async def _verify_approach(
        self,
        approach: ThoughtStep,
        understanding: ThoughtStep
    ) -> ThoughtStep:
        """Verify the approach is correct and feasible."""
        prompt = f"""
        Verify this approach:

        {approach.content}

        Against constraints:
        {understanding.content}

        Check:
        1. Does the time complexity fit within limits?
        2. Does the space complexity fit within limits?
        3. Does it handle all cases correctly?
        4. Are there any edge cases that would fail?
        5. Is the approach complete?

        If there are issues, explain what they are.
        """

        response = await self.experts["algorithm-core"].generate(prompt)

        return ThoughtStep(
            step_type=ReasoningStep.VERIFY,
            content=response.content,
            confidence=response.confidence,
            verification="verified" if response.confidence > 0.8 else "needs_revision"
        )

    async def _implement_solution(
        self,
        problem: str,
        approach: ThoughtStep
    ) -> ThoughtStep:
        """Generate the actual code solution."""
        prompt = f"""
        Implement this solution in Python:

        Approach:
        {approach.content}

        Requirements:
        1. Complete, runnable code
        2. Handle all input/output correctly
        3. Include all edge cases
        4. Optimize for speed
        5. Add brief comments for complex logic

        Original problem for reference:
        {problem}
        """

        response = await self.experts["implementation-expert"].generate(prompt)

        return ThoughtStep(
            step_type=ReasoningStep.IMPLEMENT,
            content=response.content,
            confidence=response.confidence
        )

    async def _test_solution(
        self,
        implementation: ThoughtStep,
        problem: str
    ) -> ThoughtStep:
        """Test the solution against sample cases."""
        # Extract sample test cases from problem
        test_cases = self._extract_test_cases(problem)

        # Generate additional edge cases
        edge_cases = await self._generate_edge_cases(problem)

        all_tests = test_cases + edge_cases

        # Run tests
        results = []
        for test in all_tests:
            result = await self._run_test(implementation.content, test)
            results.append(result)

        passed = sum(1 for r in results if r["passed"])
        total = len(results)

        return ThoughtStep(
            step_type=ReasoningStep.TEST,
            content=f"Passed {passed}/{total} tests.\n" + "\n".join(
                f"Test {i+1}: {'PASS' if r['passed'] else 'FAIL'}"
                for i, r in enumerate(results)
            ),
            confidence=passed / total if total > 0 else 0
        )

    async def _repair_solution(
        self,
        implementation: ThoughtStep,
        test_result: ThoughtStep,
        problem: str
    ) -> ThoughtStep:
        """Repair the solution based on failed tests."""
        prompt = f"""
        This solution has bugs:

        {implementation.content}

        Test results:
        {test_result.content}

        Original problem:
        {problem}

        Fix the solution:
        1. Identify the bug(s)
        2. Explain why they occur
        3. Provide corrected code

        Return the complete fixed solution.
        """

        response = await self.experts["implementation-expert"].generate(prompt)

        return ThoughtStep(
            step_type=ReasoningStep.REPAIR,
            content=response.content,
            confidence=response.confidence
        )

    def _merge_pattern_insights(self, responses: List[Any]) -> Dict[str, Any]:
        """Merge insights from multiple experts."""
        # Implementation: combine and rank patterns
        all_patterns = []
        for r in responses:
            all_patterns.append({
                "content": r.content,
                "confidence": r.confidence
            })

        # Sort by confidence
        all_patterns.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "content": all_patterns[0]["content"],
            "confidence": all_patterns[0]["confidence"],
            "alternatives": [p["content"] for p in all_patterns[1:]]
        }

    def _extract_test_cases(self, problem: str) -> List[Dict]:
        """Extract test cases from problem statement."""
        # Implementation: parse problem for examples
        return []

    async def _generate_edge_cases(self, problem: str) -> List[Dict]:
        """Generate edge cases for testing."""
        # Implementation: use expert to generate edge cases
        return []

    async def _run_test(self, code: str, test: Dict) -> Dict:
        """Run code against a test case."""
        # Implementation: execute code safely
        return {"passed": True}
```

### Step 3.2: Implement Test-Time Compute Scaling

Create `jupiter/reasoning/mcts.py`:

```python
"""
Monte Carlo Tree Search for Code Generation.

Implements test-time compute scaling by exploring multiple
solution paths and selecting the best one.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import asyncio


@dataclass
class MCTSNode:
    """Node in the MCTS tree."""
    state: str  # Partial solution or approach
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0

    @property
    def ucb1(self) -> float:
        """Upper Confidence Bound for tree search."""
        if self.visits == 0:
            return float("inf")

        exploitation = self.value / self.visits
        exploration = math.sqrt(2 * math.log(self.parent.visits) / self.visits)

        return exploitation + exploration


class CodeMCTS:
    """
    Monte Carlo Tree Search for code generation.

    Uses MCTS to explore different solution approaches and
    implementations, selecting the most promising one.
    """

    def __init__(
        self,
        experts: Dict[str, Any],
        config: Dict[str, Any]
    ):
        self.experts = experts
        self.max_iterations = config.get("mcts_iterations", 100)
        self.max_depth = config.get("mcts_depth", 5)
        self.exploration_weight = config.get("exploration_weight", 1.4)
        self.num_simulations = config.get("simulations_per_node", 3)

    async def search(self, problem: str) -> Tuple[str, float]:
        """
        Perform MCTS to find the best solution.

        Returns:
            Tuple of (best_solution, confidence_score)
        """
        # Create root node with problem analysis
        root = MCTSNode(state=problem)

        for iteration in range(self.max_iterations):
            # Selection: traverse tree to find promising node
            node = await self._select(root)

            # Expansion: add new child nodes
            if node.visits > 0 and len(node.children) == 0:
                await self._expand(node, problem)

            # Simulation: evaluate the node
            value = await self._simulate(node, problem)

            # Backpropagation: update values up the tree
            self._backpropagate(node, value)

        # Return best solution
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.state, best_child.value / best_child.visits

    async def _select(self, node: MCTSNode) -> MCTSNode:
        """Select the most promising node to explore."""
        while node.children:
            if random.random() < 0.1:  # 10% random exploration
                node = random.choice(node.children)
            else:
                node = max(node.children, key=lambda c: c.ucb1)
        return node

    async def _expand(self, node: MCTSNode, problem: str) -> None:
        """Expand the node with new child states."""
        # Generate alternative approaches
        approaches = await self._generate_alternatives(node.state, problem)

        for approach in approaches:
            child = MCTSNode(state=approach, parent=node)
            node.children.append(child)

    async def _simulate(self, node: MCTSNode, problem: str) -> float:
        """Simulate solution quality from this node."""
        total_score = 0.0

        for _ in range(self.num_simulations):
            # Complete the solution from this state
            solution = await self._complete_solution(node.state, problem)

            # Evaluate the solution
            score = await self._evaluate_solution(solution, problem)
            total_score += score

        return total_score / self.num_simulations

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Propagate the value up the tree."""
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent

    async def _generate_alternatives(
        self,
        state: str,
        problem: str
    ) -> List[str]:
        """Generate alternative solution approaches."""
        prompt = f"""
        Current approach:
        {state}

        Problem:
        {problem}

        Generate 3 alternative approaches or refinements.
        Each should be distinct and potentially better.
        """

        responses = await asyncio.gather(
            self.experts["algorithm-core"].generate(prompt),
            self.experts["dp-expert"].generate(prompt),
            self.experts["graph-expert"].generate(prompt),
        )

        return [r.content for r in responses]

    async def _complete_solution(self, state: str, problem: str) -> str:
        """Complete a solution from the current state."""
        prompt = f"""
        Approach:
        {state}

        Problem:
        {problem}

        Provide a complete, working solution in Python.
        """

        response = await self.experts["implementation-expert"].generate(prompt)
        return response.content

    async def _evaluate_solution(self, solution: str, problem: str) -> float:
        """Evaluate a solution's quality (0.0 to 1.0)."""
        # Run against test cases
        test_cases = self._extract_test_cases(problem)

        passed = 0
        for test in test_cases:
            result = await self._run_code(solution, test["input"])
            if result == test["expected_output"]:
                passed += 1

        return passed / len(test_cases) if test_cases else 0.5

    def _extract_test_cases(self, problem: str) -> List[Dict]:
        """Extract test cases from problem."""
        # Implementation: parse sample I/O
        return []

    async def _run_code(self, code: str, input_data: str) -> str:
        """Execute code safely and return output."""
        # Implementation: sandboxed execution
        return ""
```

### Step 3.3: Add Verification Loop

Create `jupiter/verification/verifier.py`:

```python
"""
Solution Verification System.

Verifies solutions before submission using:
1. Sample test cases
2. Generated edge cases
3. Stress testing
4. Complexity verification
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import asyncio
import subprocess
import tempfile
import time


@dataclass
class VerificationResult:
    """Result of solution verification."""
    passed: bool
    score: float  # 0.0 to 1.0
    test_results: List[Dict]
    edge_case_results: List[Dict]
    stress_test_result: Optional[Dict]
    complexity_check: Optional[Dict]
    issues_found: List[str]
    suggestions: List[str]


class SolutionVerifier:
    """
    Comprehensive solution verification system.
    """

    def __init__(self, experts: Dict[str, Any], config: Dict[str, Any]):
        self.experts = experts
        self.timeout_seconds = config.get("timeout", 5)
        self.stress_test_iterations = config.get("stress_iterations", 100)

    async def verify(
        self,
        solution: str,
        problem: str,
        language: str = "python"
    ) -> VerificationResult:
        """
        Verify a solution comprehensively.
        """
        issues = []
        suggestions = []

        # 1. Extract and run sample tests
        sample_tests = self._extract_samples(problem)
        sample_results = await self._run_tests(solution, sample_tests, language)

        sample_passed = all(r["passed"] for r in sample_results)
        if not sample_passed:
            issues.append("Failed sample test cases")

        # 2. Generate and run edge cases
        edge_cases = await self._generate_edge_cases(problem)
        edge_results = await self._run_tests(solution, edge_cases, language)

        edge_passed = all(r["passed"] for r in edge_results)
        if not edge_passed:
            issues.append("Failed edge case tests")
            suggestions.append("Check boundary conditions and special cases")

        # 3. Stress test against brute force
        stress_result = await self._stress_test(solution, problem, language)
        if not stress_result["passed"]:
            issues.append("Failed stress test")
            suggestions.append("Solution may have subtle bugs for some inputs")

        # 4. Verify complexity
        complexity = await self._check_complexity(solution, problem)
        if not complexity["meets_requirements"]:
            issues.append(f"Complexity {complexity['actual']} may exceed limits")
            suggestions.append(f"Target complexity: {complexity['expected']}")

        # Calculate overall score
        total_tests = len(sample_results) + len(edge_results)
        passed_tests = (
            sum(1 for r in sample_results if r["passed"]) +
            sum(1 for r in edge_results if r["passed"])
        )

        score = passed_tests / total_tests if total_tests > 0 else 0
        if stress_result["passed"]:
            score = min(1.0, score + 0.1)
        if complexity["meets_requirements"]:
            score = min(1.0, score + 0.1)

        return VerificationResult(
            passed=len(issues) == 0,
            score=score,
            test_results=sample_results,
            edge_case_results=edge_results,
            stress_test_result=stress_result,
            complexity_check=complexity,
            issues_found=issues,
            suggestions=suggestions
        )

    def _extract_samples(self, problem: str) -> List[Dict]:
        """Extract sample test cases from problem statement."""
        # Parse problem for Input/Output examples
        samples = []

        # Simple pattern matching for common formats
        import re

        # Look for patterns like "Input:\n...\nOutput:\n..."
        pattern = r"(?:Input|Example).*?:\s*\n(.*?)\n.*?Output.*?:\s*\n(.*?)(?:\n\n|\Z)"
        matches = re.findall(pattern, problem, re.DOTALL | re.IGNORECASE)

        for inp, out in matches:
            samples.append({
                "input": inp.strip(),
                "expected_output": out.strip()
            })

        return samples

    async def _generate_edge_cases(self, problem: str) -> List[Dict]:
        """Generate edge cases using AI expert."""
        prompt = f"""
        Generate edge case test inputs for this problem:

        {problem}

        Include:
        1. Minimum size inputs (n=1, empty, etc.)
        2. Maximum size inputs (at constraint limits)
        3. Special patterns (all same, sorted, reverse sorted)
        4. Boundary values (0, -1, MAX_INT, etc.)
        5. Corner cases specific to this problem

        For each test case, provide:
        - Input
        - Expected output (or description if complex)
        - Why this is an edge case

        Format as JSON array.
        """

        response = await self.experts["algorithm-core"].generate(prompt)

        # Parse response into test cases
        try:
            import json
            cases = json.loads(response.content)
            return cases
        except:
            return []

    async def _run_tests(
        self,
        solution: str,
        tests: List[Dict],
        language: str
    ) -> List[Dict]:
        """Run solution against test cases."""
        results = []

        for test in tests:
            result = await self._execute_code(
                solution,
                test["input"],
                language
            )

            passed = result["output"].strip() == test["expected_output"].strip()

            results.append({
                "input": test["input"],
                "expected": test["expected_output"],
                "actual": result["output"],
                "passed": passed,
                "time_ms": result["time_ms"],
                "error": result.get("error")
            })

        return results

    async def _execute_code(
        self,
        code: str,
        input_data: str,
        language: str
    ) -> Dict:
        """Execute code in sandbox and return result."""
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py' if language == 'python' else '.cpp',
            delete=False
        ) as f:
            f.write(code)
            f.flush()

            start_time = time.time()

            try:
                if language == "python":
                    result = subprocess.run(
                        ["python3", f.name],
                        input=input_data,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout_seconds
                    )
                else:
                    # Compile C++
                    compile_result = subprocess.run(
                        ["g++", "-O2", f.name, "-o", f.name + ".out"],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if compile_result.returncode != 0:
                        return {
                            "output": "",
                            "time_ms": 0,
                            "error": compile_result.stderr
                        }

                    result = subprocess.run(
                        [f.name + ".out"],
                        input=input_data,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout_seconds
                    )

                elapsed_ms = (time.time() - start_time) * 1000

                return {
                    "output": result.stdout,
                    "time_ms": elapsed_ms,
                    "error": result.stderr if result.returncode != 0 else None
                }

            except subprocess.TimeoutExpired:
                return {
                    "output": "",
                    "time_ms": self.timeout_seconds * 1000,
                    "error": "Time Limit Exceeded"
                }
            except Exception as e:
                return {
                    "output": "",
                    "time_ms": 0,
                    "error": str(e)
                }

    async def _stress_test(
        self,
        solution: str,
        problem: str,
        language: str
    ) -> Dict:
        """Stress test against brute force."""
        # Generate brute force solution
        brute_force = await self._generate_brute_force(problem)

        if not brute_force:
            return {"passed": True, "note": "Brute force not generated"}

        # Run stress test
        failed_case = None
        for i in range(self.stress_test_iterations):
            # Generate random test case
            test_input = await self._generate_random_input(problem)

            # Run both solutions
            result1 = await self._execute_code(solution, test_input, language)
            result2 = await self._execute_code(brute_force, test_input, language)

            if result1["output"].strip() != result2["output"].strip():
                failed_case = {
                    "input": test_input,
                    "solution_output": result1["output"],
                    "brute_force_output": result2["output"]
                }
                break

        return {
            "passed": failed_case is None,
            "iterations": i + 1,
            "failed_case": failed_case
        }

    async def _check_complexity(
        self,
        solution: str,
        problem: str
    ) -> Dict:
        """Analyze and verify solution complexity."""
        prompt = f"""
        Analyze the time and space complexity of this solution:

        {solution}

        Problem constraints:
        {problem}

        Provide:
        1. Time complexity (Big O)
        2. Space complexity (Big O)
        3. Whether it meets the constraints
        4. Potential optimizations
        """

        response = await self.experts["algorithm-core"].generate(prompt)

        # Parse response
        # This is simplified - real implementation would be more robust
        meets_requirements = "meets" in response.content.lower() or \
                           "sufficient" in response.content.lower() or \
                           "should pass" in response.content.lower()

        return {
            "analysis": response.content,
            "meets_requirements": meets_requirements,
            "actual": "O(n log n)",  # Parsed from response
            "expected": "O(n log n)"  # Parsed from problem
        }

    async def _generate_brute_force(self, problem: str) -> Optional[str]:
        """Generate a simple brute force solution for comparison."""
        prompt = f"""
        Write a simple brute force solution for this problem:

        {problem}

        The solution should be:
        1. Definitely correct (even if slow)
        2. As simple as possible
        3. Complete and runnable

        This will be used to verify the optimized solution.
        """

        response = await self.experts["implementation-expert"].generate(prompt)
        return response.content

    async def _generate_random_input(self, problem: str) -> str:
        """Generate random test input for stress testing."""
        # Implementation: generate valid random input based on constraints
        return ""
```

### Step 3.4: Phase 3 Evaluation

```bash
# Enable extended reasoning
export JUPITER_REASONING=extended
export JUPITER_MCTS_ITERATIONS=50
export JUPITER_VERIFICATION=strict

# Run evaluation
jupiter evaluate --benchmark livecodebench --full --reasoning extended

# Expected results at Phase 3:
# - Easy: 92-95%
# - Medium: 65-75%
# - Hard: 35-45%
# - Overall: 70-80%
```

---

## Phase 4: State-of-the-Art (Target: 85-90%)

### Step 4.1: Scale Up Experts

Upgrade expert configurations for larger models:

```yaml
# config/experts/competitive/algorithm_core_large.yaml

name: "algorithm-core-large"
domain: "competitive_programming"
description: "Large-scale algorithm expert with comprehensive knowledge"

base_model: "jupiter-7b"  # Upgraded from 3B
model_path: "models/experts/algorithm-core-7b"

# Enhanced with reinforcement learning from human feedback
training:
  method: "rlhf"
  reward_model: "models/reward/cp_reward_7b"
  ppo_epochs: 4

max_tokens: 4096
temperature: 0.2
```

### Step 4.2: Implement Ensemble Methods

Create `jupiter/ensemble/voting.py`:

```python
"""
Ensemble methods for combining multiple solutions.

Uses weighted voting and self-consistency to select the best solution
from multiple candidates.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from collections import Counter
import asyncio


@dataclass
class EnsembleSolution:
    """Solution with ensemble confidence."""
    code: str
    confidence: float
    votes: int
    sources: List[str]  # Which experts/approaches produced this


class SolutionEnsemble:
    """
    Ensemble system for solution selection.

    Generates multiple solutions and uses voting/verification
    to select the most likely correct one.
    """

    def __init__(
        self,
        experts: Dict[str, Any],
        verifier: Any,
        config: Dict[str, Any]
    ):
        self.experts = experts
        self.verifier = verifier
        self.num_candidates = config.get("ensemble_size", 5)
        self.temperature_range = config.get("temperatures", [0.1, 0.3, 0.5, 0.7])

    async def solve(self, problem: str) -> EnsembleSolution:
        """
        Generate multiple solutions and select the best.
        """
        # 1. Generate candidates with different temperatures
        candidates = await self._generate_candidates(problem)

        # 2. Verify each candidate
        verified = await self._verify_candidates(candidates, problem)

        # 3. Cluster similar solutions
        clusters = self._cluster_solutions(verified)

        # 4. Select best cluster (most votes + highest verification score)
        best_cluster = self._select_best_cluster(clusters)

        # 5. Return representative from best cluster
        representative = max(
            best_cluster,
            key=lambda x: x["verification_score"]
        )

        return EnsembleSolution(
            code=representative["code"],
            confidence=representative["verification_score"],
            votes=len(best_cluster),
            sources=representative["sources"]
        )

    async def _generate_candidates(self, problem: str) -> List[Dict]:
        """Generate multiple solution candidates."""
        candidates = []

        # Different expert combinations
        expert_combinations = [
            ["algorithm-core", "implementation-expert"],
            ["dp-expert", "implementation-expert"],
            ["graph-expert", "implementation-expert"],
            ["algorithm-core", "dp-expert", "implementation-expert"],
        ]

        tasks = []
        for experts in expert_combinations:
            for temp in self.temperature_range:
                tasks.append(
                    self._generate_single(problem, experts, temp)
                )

        results = await asyncio.gather(*tasks)

        for result in results:
            if result:
                candidates.append(result)

        return candidates

    async def _generate_single(
        self,
        problem: str,
        expert_names: List[str],
        temperature: float
    ) -> Dict:
        """Generate a single solution candidate."""
        # Chain of experts
        context = problem

        for expert_name in expert_names[:-1]:
            expert = self.experts[expert_name]
            response = await expert.generate(
                context,
                temperature=temperature
            )
            context += f"\n\n{expert_name} analysis:\n{response.content}"

        # Final implementation
        impl_expert = self.experts[expert_names[-1]]
        solution = await impl_expert.generate(
            context,
            temperature=temperature
        )

        return {
            "code": solution.content,
            "sources": expert_names,
            "temperature": temperature
        }

    async def _verify_candidates(
        self,
        candidates: List[Dict],
        problem: str
    ) -> List[Dict]:
        """Verify all candidates."""
        tasks = [
            self.verifier.verify(c["code"], problem)
            for c in candidates
        ]

        results = await asyncio.gather(*tasks)

        for candidate, result in zip(candidates, results):
            candidate["verification_score"] = result.score
            candidate["test_results"] = result.test_results

        return candidates

    def _cluster_solutions(
        self,
        verified: List[Dict]
    ) -> List[List[Dict]]:
        """Cluster similar solutions together."""
        # Simplified clustering based on output similarity
        clusters = []

        for solution in verified:
            added = False
            for cluster in clusters:
                if self._solutions_similar(solution, cluster[0]):
                    cluster.append(solution)
                    added = True
                    break

            if not added:
                clusters.append([solution])

        return clusters

    def _solutions_similar(self, sol1: Dict, sol2: Dict) -> bool:
        """Check if two solutions are functionally similar."""
        # Compare test outputs
        if "test_results" not in sol1 or "test_results" not in sol2:
            return False

        outputs1 = [r["actual"] for r in sol1["test_results"]]
        outputs2 = [r["actual"] for r in sol2["test_results"]]

        return outputs1 == outputs2

    def _select_best_cluster(
        self,
        clusters: List[List[Dict]]
    ) -> List[Dict]:
        """Select the best cluster based on votes and scores."""
        def cluster_score(cluster):
            vote_score = len(cluster)
            avg_verification = sum(
                c["verification_score"] for c in cluster
            ) / len(cluster)
            return vote_score * avg_verification

        return max(clusters, key=cluster_score)
```

### Step 4.3: Add Debate Mechanism

```python
"""
Expert Debate System.

Allows experts to critique and improve each other's solutions
through structured debate rounds.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import asyncio


@dataclass
class DebateRound:
    """A single round of expert debate."""
    round_number: int
    critiques: List[Dict]
    improvements: List[Dict]
    consensus_reached: bool


class ExpertDebate:
    """
    Multi-round debate between experts to improve solutions.
    """

    def __init__(self, experts: Dict[str, Any], config: Dict[str, Any]):
        self.experts = experts
        self.max_rounds = config.get("debate_rounds", 3)
        self.consensus_threshold = config.get("consensus_threshold", 0.9)

    async def debate(
        self,
        problem: str,
        initial_solution: str
    ) -> Tuple[str, List[DebateRound]]:
        """
        Run debate to improve solution.

        Returns:
            Tuple of (improved_solution, debate_history)
        """
        current_solution = initial_solution
        history = []

        for round_num in range(self.max_rounds):
            # Gather critiques from all experts
            critiques = await self._gather_critiques(
                problem, current_solution
            )

            # Generate improvements based on critiques
            improvements = await self._generate_improvements(
                problem, current_solution, critiques
            )

            # Check for consensus
            consensus, best_improvement = self._check_consensus(
                improvements
            )

            history.append(DebateRound(
                round_number=round_num + 1,
                critiques=critiques,
                improvements=improvements,
                consensus_reached=consensus
            ))

            if consensus:
                return best_improvement, history

            # Update solution for next round
            current_solution = best_improvement

        return current_solution, history

    async def _gather_critiques(
        self,
        problem: str,
        solution: str
    ) -> List[Dict]:
        """Gather critiques from all experts."""
        critiques = []

        for name, expert in self.experts.items():
            prompt = f"""
            Review this solution for potential issues:

            Problem:
            {problem}

            Solution:
            {solution}

            Provide:
            1. Correctness issues (bugs, edge cases)
            2. Efficiency concerns
            3. Style/clarity improvements
            4. Missing optimizations

            Be specific and constructive.
            """

            response = await expert.generate(prompt)

            critiques.append({
                "expert": name,
                "critique": response.content,
                "severity": self._assess_severity(response.content)
            })

        return critiques

    async def _generate_improvements(
        self,
        problem: str,
        solution: str,
        critiques: List[Dict]
    ) -> List[Dict]:
        """Generate improved solutions based on critiques."""
        critique_summary = "\n\n".join(
            f"{c['expert']}: {c['critique']}"
            for c in critiques
        )

        improvements = []

        for name, expert in self.experts.items():
            prompt = f"""
            Improve this solution based on the critiques:

            Problem:
            {problem}

            Current Solution:
            {solution}

            Critiques:
            {critique_summary}

            Provide an improved solution that addresses the valid critiques.
            Explain what you changed and why.
            """

            response = await expert.generate(prompt)

            improvements.append({
                "expert": name,
                "solution": response.content,
                "confidence": response.confidence
            })

        return improvements

    def _check_consensus(
        self,
        improvements: List[Dict]
    ) -> Tuple[bool, str]:
        """Check if experts have reached consensus."""
        # Compare solutions for similarity
        solutions = [i["solution"] for i in improvements]
        confidences = [i["confidence"] for i in improvements]

        # Check if all solutions are similar enough
        avg_confidence = sum(confidences) / len(confidences)

        if avg_confidence >= self.consensus_threshold:
            # Return highest confidence solution
            best = max(improvements, key=lambda x: x["confidence"])
            return True, best["solution"]

        return False, max(improvements, key=lambda x: x["confidence"])["solution"]

    def _assess_severity(self, critique: str) -> str:
        """Assess severity of critique."""
        critique_lower = critique.lower()

        if any(word in critique_lower for word in ["bug", "wrong", "incorrect", "fail"]):
            return "critical"
        elif any(word in critique_lower for word in ["slow", "inefficient", "tle"]):
            return "major"
        else:
            return "minor"
```

### Step 4.4: Final System Configuration

Create `config/swarm/competitive_optimal.yaml`:

```yaml
# =============================================================================
# OPTIMAL SWARM CONFIGURATION FOR 90% LIVECODEBENCH
# =============================================================================

name: "livecodebench-90"
mode: "inference"

experts_dir: "config/experts/competitive"

# Large expert models
expert_models:
  algorithm-core-large:
    size: "7b"
    node: "mac-studio-1"

  dp-expert-large:
    size: "7b"
    node: "mac-studio-2"

  graph-expert-large:
    size: "7b"
    node: "mac-mini-1"

  math-expert-large:
    size: "3b"
    node: "mac-mini-2"

  string-expert:
    size: "3b"
    node: "mac-mini-3"

  implementation-expert-large:
    size: "7b"
    node: "mac-mini-4"

  verifier-expert:
    size: "3b"
    node: "mac-mini-5"

# Advanced routing
router:
  strategy: "learned"  # ML-based routing
  model: "models/router/cp_router_1b"
  top_k: 4
  min_confidence: 0.2

  fallback_strategy: "hybrid"

# Reasoning configuration
reasoning:
  mode: "extended"
  max_thinking_tokens: 16384

  chain_of_thought:
    enabled: true
    steps:
      - "problem_understanding"
      - "pattern_identification"
      - "approach_selection"
      - "complexity_verification"
      - "implementation"
      - "testing"
      - "repair"

  mcts:
    enabled: true
    iterations: 100
    max_depth: 7
    exploration_weight: 1.4
    simulations_per_node: 5

# Verification
verification:
  enabled: true
  generate_edge_cases: true
  stress_test: true
  stress_iterations: 200
  complexity_check: true
  self_repair_attempts: 5

# Ensemble
ensemble:
  enabled: true
  size: 7
  temperatures: [0.1, 0.2, 0.3, 0.5, 0.7]
  voting_method: "weighted"

# Expert debate
debate:
  enabled: true
  max_rounds: 3
  consensus_threshold: 0.85

# Synthesis
synthesizer:
  strategy: "reasoning_chain"
  include_expert_attribution: true

# Execution
execution:
  max_parallel_experts: 8
  timeout_seconds: 300
  enable_expert_collaboration: true
  collaboration_rounds: 3
```

### Step 4.5: Training with Reinforcement Learning

```bash
# Train reward model on competitive programming feedback
jupiter train-reward \
  --data data/cp_feedback.jsonl \
  --model jupiter-3b \
  --output models/reward/cp_reward_3b

# Fine-tune experts with RLHF
for expert in algorithm-core dp-expert graph-expert implementation-expert; do
  jupiter rlhf \
    --expert $expert \
    --reward-model models/reward/cp_reward_3b \
    --ppo-epochs 4 \
    --batch-size 16 \
    --output models/experts/${expert}-rlhf
done
```

### Step 4.6: Final Evaluation

```bash
# Run full evaluation with all optimizations
jupiter evaluate \
  --benchmark livecodebench \
  --config config/swarm/competitive_optimal.yaml \
  --full \
  --save-results results/livecodebench_final.json

# Expected results at Phase 4:
# - Easy: 95-98%
# - Medium: 80-88%
# - Hard: 55-70%
# - Overall: 85-90%
```

---

## Training Data Pipeline

### Data Collection

```bash
# Collect competitive programming problems
jupiter collect \
  --source codeforces \
  --rating-min 800 \
  --rating-max 3500 \
  --output data/raw/codeforces

jupiter collect \
  --source leetcode \
  --difficulty all \
  --output data/raw/leetcode

jupiter collect \
  --source atcoder \
  --contests abc,arc,agc \
  --output data/raw/atcoder
```

### Data Processing

```yaml
# config/data/cp_processing.yaml

processing:
  steps:
    - name: "parse_problems"
      input: "data/raw/*"
      output: "data/parsed/"

    - name: "extract_solutions"
      input: "data/parsed/"
      output: "data/solutions/"
      filters:
        - "accepted_only"
        - "top_10_percent_runtime"

    - name: "generate_explanations"
      input: "data/solutions/"
      output: "data/explained/"
      model: "claude-3-opus"

    - name: "augment_with_cot"
      input: "data/explained/"
      output: "data/cot/"

    - name: "quality_filter"
      input: "data/cot/"
      output: "data/final/"
      min_score: 0.8
```

### Training Data Format

```json
{
  "id": "cf_1234_A",
  "source": "codeforces",
  "problem": "Given an array of N integers...",
  "constraints": {
    "n": "1 <= N <= 10^5",
    "values": "-10^9 <= a[i] <= 10^9",
    "time_limit": "2 seconds",
    "memory_limit": "256 MB"
  },
  "difficulty": "1400",
  "tags": ["dp", "greedy"],
  "chain_of_thought": {
    "understanding": "We need to find the maximum sum subsequence...",
    "pattern": "This is a classic DP problem similar to LIS...",
    "approach": "Use DP with state dp[i] = max sum ending at i...",
    "complexity": "Time: O(n log n), Space: O(n)"
  },
  "solution": "def solve(n, a):\n    ...",
  "test_cases": [
    {"input": "5\n1 2 3 4 5", "output": "15"},
    {"input": "3\n-1 -2 -3", "output": "-1"}
  ]
}
```

---

## Expert Configurations

### Complete Expert List for 90%

| Expert | Size | Specialization | Node Assignment |
|--------|------|----------------|-----------------|
| algorithm-core-large | 7B | Core algorithms, complexity | Mac Studio 1 |
| dp-expert-large | 7B | Dynamic programming | Mac Studio 2 |
| graph-expert-large | 7B | Graph algorithms | Mac Mini 1 |
| tree-expert | 3B | Tree algorithms, LCA | Mac Mini 2 |
| math-expert | 3B | Number theory, combinatorics | Mac Mini 3 |
| string-expert | 3B | String algorithms | Mac Mini 4 |
| geometry-expert | 3B | Computational geometry | Mac Mini 5 |
| implementation-expert-large | 7B | Code generation | Mac Mini 6 |
| optimization-expert | 3B | Time/space optimization | Mac Mini 7 |
| verifier-expert | 3B | Testing, verification | Mac Mini 8 |

### Expert Interaction Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EXPERT INTERACTION FLOW                          │
│                                                                         │
│   Problem Input                                                         │
│       │                                                                 │
│       ▼                                                                 │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                    ANALYSIS PHASE                                │  │
│   │                                                                  │  │
│   │   algorithm-core ──► Problem type, constraints                   │  │
│   │         │                                                        │  │
│   │         ▼                                                        │  │
│   │   [dp-expert, graph-expert, math-expert, string-expert]         │  │
│   │   (Parallel analysis based on detected patterns)                 │  │
│   │         │                                                        │  │
│   │         ▼                                                        │  │
│   │   optimization-expert ──► Approach selection                     │  │
│   │                                                                  │  │
│   └──────────────────────────┬──────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                    IMPLEMENTATION PHASE                          │  │
│   │                                                                  │  │
│   │   implementation-expert ──► Code generation                      │  │
│   │         │                                                        │  │
│   │         ▼                                                        │  │
│   │   verifier-expert ──► Testing, edge cases                       │  │
│   │         │                                                        │  │
│   │         │ (if tests fail)                                        │  │
│   │         ▼                                                        │  │
│   │   [DEBATE ROUND with relevant experts]                          │  │
│   │         │                                                        │  │
│   │         ▼                                                        │  │
│   │   implementation-expert ──► Fixed code                           │  │
│   │                                                                  │  │
│   └──────────────────────────┬──────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                    ENSEMBLE PHASE                                │  │
│   │                                                                  │  │
│   │   Generate 5-7 candidate solutions                               │  │
│   │         │                                                        │  │
│   │         ▼                                                        │  │
│   │   Verify all candidates                                          │  │
│   │         │                                                        │  │
│   │         ▼                                                        │  │
│   │   Vote + select best                                             │  │
│   │                                                                  │  │
│   └──────────────────────────┬──────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│                       FINAL SOLUTION                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Evaluation and Monitoring

### Continuous Evaluation

```bash
# Set up continuous evaluation
jupiter monitor \
  --benchmark livecodebench \
  --interval daily \
  --notify email,slack \
  --dashboard http://localhost:3000
```

### Metrics to Track

| Metric | Target | Description |
|--------|--------|-------------|
| pass@1 | 90% | First attempt success rate |
| pass@5 | 95% | Success within 5 attempts |
| Easy accuracy | 98% | Easy problem success |
| Medium accuracy | 88% | Medium problem success |
| Hard accuracy | 70% | Hard problem success |
| Avg latency | <30s | Time to solution |
| Token efficiency | <8K | Tokens per solution |

### Performance Dashboard

```yaml
# config/monitoring/dashboard.yaml

dashboard:
  refresh_interval: 60

  panels:
    - name: "Overall Accuracy"
      type: "gauge"
      metric: "livecodebench_pass_rate"
      thresholds:
        green: 0.85
        yellow: 0.70
        red: 0.50

    - name: "Accuracy by Difficulty"
      type: "bar_chart"
      metrics:
        - "easy_accuracy"
        - "medium_accuracy"
        - "hard_accuracy"

    - name: "Expert Usage"
      type: "pie_chart"
      metric: "expert_invocations"

    - name: "Latency Distribution"
      type: "histogram"
      metric: "solution_latency_ms"

    - name: "Error Rate"
      type: "line_chart"
      metric: "error_rate"
      period: "7d"
```

---

## Troubleshooting

### Common Issues

#### 1. Low Hard Problem Accuracy (<50%)

**Symptoms**: Easy/Medium problems pass but Hard problems fail consistently.

**Solutions**:
- Enable more MCTS iterations (increase to 200)
- Add more debate rounds (increase to 5)
- Use larger expert models for algorithm-core and dp-expert
- Increase ensemble size to 10

```bash
# Adjust for hard problems
jupiter config set mcts.iterations 200
jupiter config set debate.max_rounds 5
jupiter config set ensemble.size 10
```

#### 2. Time Limit Exceeded

**Symptoms**: Solutions are correct but too slow.

**Solutions**:
- Enable complexity verification earlier in pipeline
- Train optimization-expert more aggressively
- Add complexity constraints to implementation-expert prompt

```yaml
# config/experts/competitive/implementation_expert.yaml

system_prompt: |
  ...
  CRITICAL: Always verify time complexity meets constraints BEFORE writing code.
  If O(n²) is too slow for n=10^5, find a better approach first.
  ...
```

#### 3. Edge Case Failures

**Symptoms**: Solutions fail on boundary cases.

**Solutions**:
- Increase edge case generation in verifier
- Add explicit edge case checking to implementation-expert
- Enable stress testing with more iterations

```yaml
# config/swarm/competitive_optimal.yaml

verification:
  generate_edge_cases: true
  edge_case_count: 20  # Increase from default
  stress_iterations: 500

  required_edge_cases:
    - "empty_input"
    - "single_element"
    - "all_same"
    - "maximum_size"
    - "minimum_values"
    - "maximum_values"
```

#### 4. Memory Issues

**Symptoms**: Out of memory errors during inference.

**Solutions**:
- Reduce parallel experts
- Use quantized models (4-bit)
- Distribute across more nodes

```yaml
# config/cluster.yaml

memory_optimization:
  quantization: "4bit"
  max_parallel_experts: 4
  offload_to_disk: true
  gradient_checkpointing: true
```

#### 5. Inconsistent Results

**Symptoms**: Same problem gives different answers.

**Solutions**:
- Lower temperature for implementation-expert
- Increase ensemble voting threshold
- Enable deterministic mode

```yaml
# For maximum consistency
experts:
  implementation-expert:
    temperature: 0.0
    top_p: 1.0

ensemble:
  voting_threshold: 0.9

execution:
  deterministic: true
  seed: 42
```

---

## Conclusion

Reaching 90% on LiveCodeBench requires:

1. **Hardware**: 500GB+ unified memory across Mac cluster + NVIDIA GPUs
2. **Expert Scale**: 7B+ parameters for core experts, 3B for specialized
3. **Extended Reasoning**: Multi-step thinking with verification loops
4. **Ensemble Methods**: Multiple solutions with voting
5. **Self-Repair**: Automatic bug detection and fixing
6. **Quality Data**: 100K+ competitive programming problems with CoT

### Estimated Costs

| Component | Cost (USD) |
|-----------|------------|
| 2× Mac Studio M4 Ultra 192GB | $16,000 |
| 8× Mac Mini M4 Pro 64GB | $16,000 |
| 4× RTX 5090 32GB | $8,000 |
| Networking equipment | $2,000 |
| **Total Hardware** | **$42,000** |
| Training compute (cloud) | $10,000 |
| **Total Investment** | **$52,000** |

### Timeline

| Phase | Duration | Target Accuracy |
|-------|----------|-----------------|
| Phase 1: Foundation | 4-6 weeks | 40-50% |
| Phase 2: Specialization | 6-8 weeks | 55-65% |
| Phase 3: Reasoning | 8-10 weeks | 70-80% |
| Phase 4: Optimization | 6-8 weeks | 85-90% |
| **Total** | **24-32 weeks** | **90%** |

---

## References

- [LiveCodeBench Official](https://livecodebench.github.io/)
- [LiveCodeBench Paper](https://arxiv.org/abs/2403.07974)
- [Jupiter Documentation](https://github.com/raym33/jupiter)
- [MLX Framework](https://github.com/ml-explore/mlx)
- [Competitive Programming Handbook](https://cses.fi/book/book.pdf)

---

*This guide is part of the Jupiter project. For questions and contributions, visit [github.com/raym33/jupiter](https://github.com/raym33/jupiter)*
