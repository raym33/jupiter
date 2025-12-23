# Jupiter Examples & Use Cases

Complete examples of how to use Jupiter for training domain-specific expert models and running collaborative inference with MoE-R.

---

## Table of Contents

1. [Training a Python Expert](#1-training-a-python-expert)
2. [Training an Unreal Engine Expert](#2-training-an-unreal-engine-expert)
3. [Training a Medical/Chemistry Expert](#3-training-a-medicalchemistry-expert)
4. [Running MoE-R for Full-Stack Development](#4-running-moe-r-for-full-stack-development)
5. [Building a Custom Code Assistant](#5-building-a-custom-code-assistant)
6. [Creating a Domain-Specific Chatbot](#6-creating-a-domain-specific-chatbot)
7. [Multi-Language Documentation Assistant](#7-multi-language-documentation-assistant)
8. [Game Development Assistant with MoE-R](#8-game-development-assistant-with-moe-r)

---

## 1. Training a Python Expert

**Goal**: Train a 1B parameter model that's an expert in Python programming.

### Step 1: Create the domain configuration

```bash
jupiter new-domain python_expert
```

Edit `config/domains/python_expert.yaml`:

```yaml
domain:
  name: "python_expert"
  description: "Expert in Python programming, libraries, and best practices"
  language: "en"

  keywords:
    - "python"
    - "pip"
    - "virtualenv"
    - "pandas"
    - "numpy"
    - "django"
    - "fastapi"
    - "flask"
    - "asyncio"
    - "typing"

data_sources:
  documentation:
    - "https://docs.python.org/3/"
    - "https://docs.djangoproject.com/"
    - "https://fastapi.tiangolo.com/"
    - "https://pandas.pydata.org/docs/"

  github:
    repos:
      - "python/cpython"
      - "tiangolo/fastapi"
      - "pandas-dev/pandas"
    file_types:
      - ".py"
      - ".md"
      - ".rst"

  websites:
    - "https://realpython.com"
    - "https://www.fullstackpython.com"

generation:
  templates:
    - type: "qa"
      system_prompt: |
        You are a Python expert with 15 years of experience.
        Always provide working code examples with type hints.
      prompt: |
        Generate a technical question about {topic} in Python
        and provide a detailed answer with code examples.
      topics:
        - "decorators"
        - "context managers"
        - "generators"
        - "asyncio"
        - "dataclasses"
        - "type hints"
        - "metaclasses"

    - type: "tutorial"
      prompt: |
        Create a step-by-step tutorial for {task}.
        Include complete, runnable code.
      tasks:
        - "building a REST API with FastAPI"
        - "data analysis with pandas"
        - "web scraping with BeautifulSoup"
        - "testing with pytest"

    - type: "debug"
      prompt: |
        Generate a common Python error related to {topic}.
        Show the error, explain the cause, and provide the fix.
      topics:
        - "ImportError"
        - "TypeError"
        - "AttributeError"
        - "async/await mistakes"

mix_ratio:
  real_docs: 0.30
  real_code: 0.25
  synthetic_qa: 0.25
  synthetic_tutorials: 0.15
  synthetic_debug: 0.05
```

### Step 2: Configure your cluster

```yaml
# config/cluster.yaml
backend: "mpi"

nodes:
  - host: "macbook.local"
    name: "MacBook Pro"
    device_type: "mac"
    memory_gb: 24
    chip: "M4"
    role: "generator"

  - host: "mac-mini-1.local"
    name: "Mac Mini 1"
    device_type: "mac"
    memory_gb: 16
    chip: "M4"
    role: "trainer"

  - host: "mac-mini-2.local"
    name: "Mac Mini 2"
    device_type: "mac"
    memory_gb: 16
    chip: "M4"
    role: "trainer"
```

### Step 3: Start training

```bash
# Check everything is configured correctly
jupiter check

# Start training in interactive mode
jupiter start --domain python_expert --interactive

# Or automatic mode (runs until convergence)
jupiter start --domain python_expert --auto --epochs 50
```

### Step 4: Use your trained model

```python
from jupiter.training.model.architecture import JupiterModel

# Load the trained model
model = JupiterModel.load("checkpoints/python_expert/best")

# Generate response
response = model.generate("How do I implement a decorator with arguments?")
print(response)
```

---

## 2. Training an Unreal Engine Expert

**Goal**: Create an AI assistant specialized in Unreal Engine 5 game development.

### Domain configuration

The project includes a ready-to-use UE5 domain: `config/domains/unreal_engine.yaml`

```bash
# Start training with the included config
jupiter start --domain unreal_engine --interactive
```

### What it learns

- C++ gameplay programming
- Blueprint visual scripting
- Gameplay Ability System (GAS)
- Niagara particle systems
- Networking and replication
- Animation blueprints
- Material editor
- Level design patterns

### Example queries it can answer

```
Q: How do I implement a combo system using GAS?
Q: Explain the Actor lifecycle in UE5
Q: How do I replicate inventory in multiplayer?
Q: Create a third-person character with wall running
```

---

## 3. Training a Medical/Chemistry Expert

**Goal**: Train a model to help students understand university-level chemistry.

### Domain configuration

```yaml
# config/domains/chemistry.yaml
domain:
  name: "chemistry"
  description: "Expert in university-level chemistry"
  language: "en"

  keywords:
    - "atom"
    - "molecule"
    - "reaction"
    - "equilibrium"
    - "thermodynamics"
    - "kinetics"
    - "organic"
    - "inorganic"

data_sources:
  documentation:
    - "https://chem.libretexts.org"
    - "https://www.chemguide.co.uk"

  websites:
    - "https://www.khanacademy.org/science/chemistry"

generation:
  templates:
    - type: "qa"
      system_prompt: |
        You are a chemistry professor. Explain concepts clearly
        with examples and analogies. Show calculations step by step.
      prompt: |
        Generate an exam-style question about {topic}
        with a detailed solution showing all steps.
      topics:
        - "stoichiometry"
        - "chemical equilibrium"
        - "acid-base reactions"
        - "electrochemistry"
        - "thermochemistry"
        - "molecular orbital theory"
        - "reaction mechanisms"

    - type: "explanation"
      prompt: |
        Explain {concept} as if teaching a university student.
        Include diagrams described in text, examples, and common misconceptions.
      topics:
        - "hybridization"
        - "resonance structures"
        - "Le Chatelier's principle"
        - "Gibbs free energy"
```

### Use cases

- **Students**: Get step-by-step explanations of complex problems
- **Teachers**: Generate practice problems and solutions
- **Researchers**: Quick reference for reactions and mechanisms

---

## 4. Running MoE-R for Full-Stack Development

**Goal**: Use multiple specialized experts collaborating to build full-stack applications.

### Setup experts

```bash
# List available experts
jupiter swarm list

# Output:
# ┌─────────────────┬──────────────┬─────────────────────────────┐
# │ Name            │ Domain       │ Description                 │
# ├─────────────────┼──────────────┼─────────────────────────────┤
# │ python-core     │ python       │ Python fundamentals         │
# │ python-backend  │ web_backend  │ FastAPI, Django, Flask      │
# │ react-frontend  │ frontend     │ React, TypeScript           │
# │ api-architect   │ api_design   │ REST, GraphQL, system design│
# └─────────────────┴──────────────┴─────────────────────────────┘
```

### Interactive development session

```bash
jupiter swarm chat --experts config/experts/fullstack
```

```
You: I need to build a todo app with React frontend and FastAPI backend

Swarm: [Routing to: react-frontend, python-backend, api-architect]

## API Design (api-architect)
I recommend a RESTful API with these endpoints:
- GET /todos - List all todos
- POST /todos - Create todo
- PUT /todos/{id} - Update todo
- DELETE /todos/{id} - Delete todo

## Backend (python-backend)
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Todo(BaseModel):
    id: int
    title: str
    completed: bool = False

@app.get("/todos")
async def get_todos():
    return todos

@app.post("/todos")
async def create_todo(todo: Todo):
    todos.append(todo)
    return todo
```

## Frontend (react-frontend)
```typescript
import { useState, useEffect } from 'react';

function TodoApp() {
  const [todos, setTodos] = useState<Todo[]>([]);

  useEffect(() => {
    fetch('/api/todos')
      .then(res => res.json())
      .then(setTodos);
  }, []);

  return (
    <ul>
      {todos.map(todo => (
        <TodoItem key={todo.id} todo={todo} />
      ))}
    </ul>
  );
}
```

[Experts: react-frontend, python-backend, api-architect]
```

### Single query mode

```bash
jupiter swarm query "How do I implement JWT authentication in FastAPI with React frontend?" -k 3
```

---

## 5. Building a Custom Code Assistant

**Goal**: Create a specialized coding assistant for your company's tech stack.

### Step 1: Create domain for your stack

```yaml
# config/domains/company_stack.yaml
domain:
  name: "company_stack"
  description: "Expert in our company's technology stack"

  keywords:
    - "internal_framework"
    - "company_api"
    - "our_database"

data_sources:
  # Your internal documentation
  documentation:
    - "https://docs.internal.company.com"

  # Your company's repositories
  github:
    repos:
      - "company/main-backend"
      - "company/frontend-app"
      - "company/shared-libs"
    file_types:
      - ".py"
      - ".ts"
      - ".tsx"

generation:
  templates:
    - type: "qa"
      system_prompt: |
        You are an expert in our company's codebase and conventions.
        Always follow our coding standards and patterns.
      topics:
        - "authentication flow"
        - "database models"
        - "API patterns"
        - "frontend components"
```

### Step 2: Train and deploy

```bash
jupiter start --domain company_stack --auto
```

### Step 3: Integrate with your workflow

```python
# integration_example.py
import asyncio
from jupiter.swarm import Swarm, SwarmConfig

async def code_review(code: str) -> str:
    """Use Jupiter to review code."""
    swarm = Swarm(SwarmConfig(experts_dir="config/experts/company"))
    await swarm.initialize()

    response = await swarm.query(f"""
    Review this code for our coding standards:

    ```python
    {code}
    ```

    Check for:
    1. Our naming conventions
    2. Error handling patterns
    3. Security issues
    4. Performance concerns
    """)

    await swarm.shutdown()
    return response.content

# Use in CI/CD
code = open("pull_request_diff.py").read()
review = asyncio.run(code_review(code))
print(review)
```

---

## 6. Creating a Domain-Specific Chatbot

**Goal**: Build a customer support chatbot for a SaaS product.

### Step 1: Create the domain

```yaml
# config/domains/saas_support.yaml
domain:
  name: "saas_support"
  description: "Customer support for our SaaS product"

data_sources:
  documentation:
    - "https://docs.yourproduct.com"
    - "https://help.yourproduct.com"

  # Include FAQs, tickets, etc.
  websites:
    - "https://community.yourproduct.com"

generation:
  templates:
    - type: "qa"
      system_prompt: |
        You are a helpful customer support agent for YourProduct.
        Be friendly, concise, and always try to solve the user's problem.
        If you don't know something, say so and offer to escalate.
      topics:
        - "billing questions"
        - "feature requests"
        - "bug reports"
        - "account setup"
        - "integrations"
```

### Step 2: Create expert configuration

```yaml
# config/experts/support/product_expert.yaml
name: "product-support"
domain: "customer_support"
description: "Expert in YourProduct features and troubleshooting"

keywords:
  - "help"
  - "issue"
  - "problem"
  - "how do I"
  - "error"
  - "not working"
  - "billing"
  - "subscription"

capabilities:
  - "Product feature explanation"
  - "Troubleshooting common issues"
  - "Billing and account questions"
  - "Integration setup help"

system_prompt: |
  You are a friendly customer support agent for YourProduct.

  Guidelines:
  - Be helpful and empathetic
  - Provide step-by-step instructions
  - Include relevant documentation links
  - Escalate complex issues appropriately
```

### Step 3: Deploy as API

```python
# api_server.py
from fastapi import FastAPI
from jupiter.swarm import Swarm, SwarmConfig

app = FastAPI()
swarm = None

@app.on_event("startup")
async def startup():
    global swarm
    swarm = Swarm(SwarmConfig(experts_dir="config/experts/support"))
    await swarm.initialize()
    await swarm.load_expert("product-support")

@app.post("/chat")
async def chat(message: str):
    response = await swarm.query(message)
    return {
        "response": response.content,
        "confidence": max(response.expert_contributions.values())
    }
```

---

## 7. Multi-Language Documentation Assistant

**Goal**: Create an assistant that can answer questions about docs in multiple languages.

### Setup experts for each language

```yaml
# config/experts/docs/english.yaml
name: "docs-english"
domain: "documentation"
keywords: ["documentation", "guide", "tutorial", "how to"]
system_prompt: "You answer questions about our documentation in English."

# config/experts/docs/spanish.yaml
name: "docs-spanish"
domain: "documentation"
keywords: ["documentación", "guía", "tutorial", "cómo"]
system_prompt: "Respondes preguntas sobre la documentación en español."

# config/experts/docs/french.yaml
name: "docs-french"
domain: "documentation"
keywords: ["documentation", "guide", "tutoriel", "comment"]
system_prompt: "Vous répondez aux questions sur la documentation en français."
```

### Query in any language

```bash
# English
jupiter swarm query "How do I configure authentication?"

# Spanish
jupiter swarm query "¿Cómo configuro la autenticación?"

# French
jupiter swarm query "Comment configurer l'authentification?"
```

---

## 8. Game Development Assistant with MoE-R

**Goal**: Build a comprehensive game dev assistant with experts for different aspects.

### Create specialized experts

```yaml
# config/experts/gamedev/gameplay.yaml
name: "gameplay-expert"
domain: "gameplay_programming"
keywords:
  - "character"
  - "movement"
  - "combat"
  - "inventory"
  - "quest"
  - "AI"
  - "behavior tree"
system_prompt: |
  You are an expert in gameplay programming.
  Focus on game mechanics, player controls, and game systems.

# config/experts/gamedev/graphics.yaml
name: "graphics-expert"
domain: "graphics"
keywords:
  - "shader"
  - "material"
  - "lighting"
  - "particle"
  - "VFX"
  - "rendering"
  - "post-process"
system_prompt: |
  You are an expert in game graphics and rendering.
  Focus on shaders, materials, VFX, and visual quality.

# config/experts/gamedev/optimization.yaml
name: "optimization-expert"
domain: "performance"
keywords:
  - "performance"
  - "FPS"
  - "memory"
  - "profiling"
  - "LOD"
  - "culling"
  - "optimization"
system_prompt: |
  You are an expert in game performance optimization.
  Focus on profiling, optimization techniques, and best practices.
```

### Interactive game dev session

```bash
jupiter swarm chat --experts config/experts/gamedev
```

```
You: I'm making an open world game and the framerate drops when there are many NPCs

Swarm: [Routing to: optimization-expert, gameplay-expert]

## Performance Analysis (optimization-expert)

For open-world NPC performance, consider:

1. **LOD System for NPCs**
   - Full AI + animations at <50m
   - Simplified AI at 50-100m
   - Static/disabled beyond 100m

2. **Spatial Partitioning**
   - Use octrees or grid-based culling
   - Only tick NPCs in player's vicinity

3. **Async Processing**
   - Move AI calculations to separate thread
   - Use behavior tree task parallelization

## Implementation (gameplay-expert)

```cpp
// NPC Manager with distance-based LOD
void ANPCManager::UpdateNPCs()
{
    FVector PlayerLoc = GetPlayerLocation();

    for (ANPC* NPC : AllNPCs)
    {
        float Distance = FVector::Dist(NPC->GetLocation(), PlayerLoc);

        if (Distance < NearDistance)
        {
            NPC->SetLODLevel(ELODLevel::Full);
            NPC->EnableAI(true);
        }
        else if (Distance < MidDistance)
        {
            NPC->SetLODLevel(ELODLevel::Medium);
            NPC->EnableAI(true);
        }
        else
        {
            NPC->SetLODLevel(ELODLevel::Minimal);
            NPC->EnableAI(false);
        }
    }
}
```

Profiling recommendation: Use Unreal Insights to identify the specific bottleneck.

[Experts: optimization-expert, gameplay-expert]
```

---

## Summary: What Jupiter Can Do

| Use Case | Mode | Description |
|----------|------|-------------|
| **Train Domain Expert** | `jupiter start` | Train a specialized model on any domain |
| **Multi-Expert Chat** | `jupiter swarm chat` | Multiple experts collaborating in real-time |
| **Code Assistant** | `jupiter swarm query` | Get help with programming questions |
| **Documentation Bot** | MoE-R + API | Answer questions about your docs |
| **Customer Support** | MoE-R + API | Automated support with domain knowledge |
| **Education Tool** | Training | Create tutors for any subject |
| **Game Dev Helper** | MoE-R | Specialized help for game development |

---

## Hardware Recommendations by Use Case

| Use Case | Minimum | Recommended |
|----------|---------|-------------|
| Single Expert (500M) | 1× Mac 16GB | 2× Mac 16GB |
| Single Expert (1B) | 2× Mac 16GB | 4× Mac 16GB |
| MoE-R (3 experts) | 3× Mac 16GB | 5× Mac 16-24GB |
| MoE-R (5+ experts) | 5× Mac 16GB | Cluster with 100GB+ total |

---

## Next Steps

- [Set up your cluster](mac_cluster_setup.md)
- [Create a new domain](new_domain.md)
- [Configure MoE-R experts](swarm_moer.md)
- [Understand self-improvement](self_improvement.md)
