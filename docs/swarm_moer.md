# MoE-R: Mixture of Real Experts

Jupiter's experimental **MoE-R** (Mixture of Real Experts) system enables multiple specialized small models to collaborate in real-time, providing better answers than any single model could produce alone.

## How It Works

Unlike traditional Mixture of Experts (MoE) which uses a single model with specialized sub-networks, MoE-R uses **completely separate models** that run on different machines in your cluster:

```
Traditional MoE:                    Jupiter MoE-R:
┌─────────────────┐                 ┌─────────────────┐
│   Single Model  │                 │   Mac 1         │
│  ┌───┐ ┌───┐   │                 │   Python Expert │
│  │E1 │ │E2 │   │                 └────────┬────────┘
│  └───┘ └───┘   │                          │
│  ┌───┐ ┌───┐   │                 ┌────────┴────────┐
│  │E3 │ │E4 │   │                 │                 │
│  └───┘ └───┘   │        ┌────────┴───────┐ ┌───────┴────────┐
└─────────────────┘        │   Mac 2        │ │   Mac 3        │
                           │   React Expert │ │   API Expert   │
                           └────────────────┘ └────────────────┘
```

### Advantages of MoE-R

1. **True Specialization**: Each expert is trained exclusively on its domain
2. **Parallel Execution**: Experts run simultaneously on different machines
3. **Collaborative Refinement**: Experts can see and build on each other's responses
4. **Scalable**: Add more experts by adding more Macs
5. **Flexible**: Mix and match experts for different tasks

## Architecture

### Components

#### 1. Expert
A specialized model trained on a specific domain (e.g., Python, React, databases).

```python
from jupiter.swarm import Expert, ExpertConfig

config = ExpertConfig(
    name="python-core",
    domain="python",
    keywords=["python", "def", "class", "import"],
    capabilities=["Python syntax", "Standard library", "Type hints"],
)

expert = Expert(config)
await expert.load()
response = await expert.generate("Explain decorators")
```

#### 2. Router
Decides which experts to activate for a given query.

Routing strategies:
- **Keyword**: Fast matching based on domain keywords
- **Embedding**: Semantic similarity (requires embedding model)
- **LLM**: Uses an LLM to classify the query
- **Hybrid**: Combines multiple strategies

```python
from jupiter.swarm import Router, RouterConfig

router = Router(RouterConfig(
    strategy="hybrid",
    top_k=3,  # Activate up to 3 experts
    min_confidence=0.3,
))

decision = await router.route("How do I create a FastAPI endpoint?")
# Selected: ["python-backend", "api-architect"]
```

#### 3. Synthesizer
Combines responses from multiple experts into a coherent answer.

Synthesis strategies:
- **Concatenate**: Simple combination with headers
- **Weighted**: Weight by confidence scores
- **LLM Merge**: Intelligent merging using an LLM
- **Best Only**: Use only the highest confidence response
- **Debate**: Multi-round refinement between experts

```python
from jupiter.swarm import Synthesizer, SynthesizerConfig

synthesizer = Synthesizer(SynthesizerConfig(
    strategy="llm_merge",
    include_expert_attribution=True,
))

final_response = await synthesizer.synthesize(query, expert_responses)
```

#### 4. Swarm
The main orchestrator that coordinates everything.

```python
from jupiter.swarm import Swarm, SwarmConfig

swarm = Swarm(SwarmConfig(
    name="python-experts",
    experts_dir="config/experts/python",
    enable_expert_collaboration=True,
))

await swarm.initialize()
response = await swarm.query("How do I use pandas for data analysis?")
print(response.content)
```

## Creating Experts

### 1. Create Expert Configuration

```bash
jupiter swarm create-expert pandas-expert --domain data_science
```

This creates `config/experts/pandas-expert.yaml`:

```yaml
name: "pandas-expert"
domain: "data_science"
description: "Expert in pandas data manipulation"

base_model: "jupiter-500m"
model_path: null  # Set after training

keywords:
  - "pandas"
  - "dataframe"
  - "series"
  - "groupby"
  - "merge"

capabilities:
  - "DataFrame operations"
  - "Data cleaning"
  - "Aggregations"

system_prompt: |
  You are a pandas expert. Provide efficient, vectorized solutions.
  Always include complete, runnable code examples.
```

### 2. Train the Expert (Coming Soon)

```bash
jupiter swarm train -e pandas-expert -d config/domains/data_science.yaml
```

### 3. Use Pre-trained Base Models

Until training is complete, experts can use pre-trained base models:

```yaml
base_model: "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
```

## Running the Swarm

### Interactive Chat

```bash
jupiter swarm chat --experts config/experts/python
```

### Single Query

```bash
jupiter swarm query "Explain React hooks and when to use them"
```

### Programmatic Usage

```python
import asyncio
from jupiter.swarm import Swarm, SwarmConfig

async def main():
    swarm = Swarm(SwarmConfig(experts_dir="config/experts"))
    await swarm.initialize()

    response = await swarm.query(
        "Create a FastAPI app with SQLAlchemy and React frontend"
    )

    print(f"Response: {response.content}")
    print(f"Experts used: {list(response.expert_contributions.keys())}")
    print(f"Tokens: {response.total_tokens}")

    await swarm.shutdown()

asyncio.run(main())
```

## Collaboration Modes

### Parallel (Default)
All experts generate responses simultaneously:

```
Query → [Expert1, Expert2, Expert3] → Synthesizer → Response
         (parallel execution)
```

### Collaborative
Experts can see and build on each other's responses:

```
Query → Expert1 → response1
      → Expert2 (sees response1) → response2
      → Expert3 (sees response1, response2) → response3
      → Synthesizer → Final Response
```

Enable with:
```python
SwarmConfig(
    enable_expert_collaboration=True,
    collaboration_rounds=1,
)
```

## Distributed Deployment

Each expert can run on a different Mac in your cluster:

```yaml
# config/experts/python/core.yaml
name: "python-core"
assigned_node: "mac-mini-1.local"  # Runs on Mac Mini 1

# config/experts/python/backend.yaml
name: "python-backend"
assigned_node: "mac-mini-2.local"  # Runs on Mac Mini 2
```

## Best Practices

1. **Specialize Narrowly**: Each expert should focus on one specific area
2. **Overlap Keywords Carefully**: Some overlap helps with routing, too much causes confusion
3. **Clear System Prompts**: Write detailed system prompts that define expertise boundaries
4. **Test Routing**: Use `jupiter swarm query` to verify correct experts are selected
5. **Monitor Confidence**: Low confidence scores indicate routing or specialization issues

## Troubleshooting

### "No experts available"
- Check `jupiter swarm list` to see available experts
- Verify expert YAML files are valid
- Check that base models are accessible

### Slow responses
- Reduce `top_k` to activate fewer experts
- Use smaller base models (500M instead of 1B)
- Enable parallel execution

### Poor response quality
- Check expert specialization (keywords, capabilities)
- Improve system prompts
- Consider using LLM merge strategy for synthesis

## Next Steps

- [Train Custom Experts](training_experts.md)
- [Configure Cluster](mac_cluster_setup.md)
- [Create Domain Configurations](new_domain.md)
