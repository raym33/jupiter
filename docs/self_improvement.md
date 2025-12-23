# Self-Improvement Cycle

Jupiter implements a cycle where the trained model can replace the generator model when it surpasses it.

## Concept

```
┌─────────────────────────────────────────────────────────────────┐
│                    SELF-IMPROVEMENT CYCLE                       │
│                                                                 │
│   ┌──────────────┐                                              │
│   │  GENERATOR   │ ◄─────────────────────────────────┐          │
│   │  (Llama 3B)  │                                   │          │
│   └──────┬───────┘                                   │          │
│          │                                           │          │
│          │ generates data                            │          │
│          ▼                                           │          │
│   ┌──────────────┐                                   │          │
│   │  SYNTHETIC   │                                   │          │
│   │    DATA      │                                   │          │
│   └──────┬───────┘                                   │          │
│          │                                           │          │
│          │ + real data                               │          │
│          ▼                                           │          │
│   ┌──────────────┐                                   │          │
│   │ DISTRIBUTED  │                                   │          │
│   │  TRAINING    │                                   │          │
│   └──────┬───────┘                                   │          │
│          │                                           │          │
│          ▼                                           │          │
│   ┌──────────────┐      ┌──────────────┐            │          │
│   │   TRAINED    │      │  EVALUATION  │            │          │
│   │    MODEL     │ ───► │              │            │          │
│   └──────────────┘      └──────┬───────┘            │          │
│                                │                     │          │
│                                │ surpasses           │          │
│                                │ generator?          │          │
│                                │                     │          │
│                         ┌──────┴──────┐              │          │
│                         │             │              │          │
│                    NO   ▼        YES  ▼              │          │
│                  continue       REPLACE ─────────────┘          │
│                  training       GENERATOR                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Why It Works

### Trained Model vs Generator

- **Generator (e.g., Llama 3B)**: General knowledge, can generate about any topic
- **Trained model (1B)**: Specialized in the specific domain

Although the trained model is smaller, it can surpass the generator **in the specific domain** because:

1. It has seen thousands of domain examples
2. It's optimized for that type of content
3. It has learned domain-specific patterns

### Avoiding Model Collapse

The risk of using a model to generate training data for itself is "model collapse": the model converges to a degenerate state.

Jupiter avoids this with:

1. **Real data anchor**: Always includes 30-40% real data (documentation, code)
2. **Quality filtering**: Discards low-quality generations
3. **Deduplication**: Prevents model from memorizing its own generations
4. **Rigorous evaluation**: Only replaces generator if there's real improvement

## Configuration

In `config/cluster.yaml` or in code:

```yaml
self_improvement:
  # Evaluate every N steps
  eval_every_steps: 5000

  # Improvement threshold to replace generator
  improvement_threshold: 0.05  # 5% better

  # Minimum steps before considering replacement
  min_steps_before_replace: 10000

  # Samples for benchmark
  benchmark_samples: 500

  # Safety
  keep_last_n_checkpoints: 3
  always_keep_base_generator: true  # Never delete original Llama
```

## Evaluation Metrics

Jupiter evaluates the model across several dimensions:

### 1. Accuracy (40%)
- Are the answers correct?
- Does the code compile/run?
- Are the facts accurate?

### 2. Coherence (30%)
- Do the answers make sense?
- Do they flow logically?
- Are they well structured?

### 3. Domain Knowledge (30%)
- Uses correct domain terminology?
- Demonstrates deep knowledge?
- Follows field best practices?

## The Upgrade Process

When the trained model surpasses the generator:

```
1. Save trained model checkpoint
2. Convert to generator format
3. Update generator reference
4. Save previous generator in history
5. Regenerate synthetic data with new generator
6. Continue training with new data
```

### Generator History

Jupiter maintains a history:

```
generators/
├── generator_v1/  (Original Llama 3B)
├── generator_v2/  (Your model after 10k steps)
├── generator_v3/  (Your model after 25k steps)
└── current -> generator_v3
```

### Rollback

If something goes wrong, you can revert to a previous generator:

```python
from jupiter.orchestrator import SelfImprover

improver = SelfImprover(config, trainer)
await improver.rollback_generator()
```

## Progress Example

```
Epoch 1-5:
  Generator: Llama 3B
  Eval score: 0.65

Epoch 6-10:
  Generator: Llama 3B
  Eval score: 0.72 (+0.07)
  → Doesn't exceed replacement threshold

Epoch 11-15:
  Generator: Llama 3B
  Eval score: 0.78 (+0.13)
  → UPGRADE! Trained model is now the generator

Epoch 16-20:
  Generator: Your model v1
  Eval score: 0.81 (+0.03)
  → Generated data is higher quality

Epoch 21-25:
  Generator: Your model v1
  Eval score: 0.85 (+0.07)
  → UPGRADE! Your model v2 is now the generator

...
```

## Visualizing Progress

Jupiter saves metrics you can visualize:

```python
import json
import matplotlib.pyplot as plt

with open("pipeline_state.json") as f:
    state = json.load(f)

# Score history (hypothetical)
epochs = range(1, state["current_epoch"] + 1)
scores = [...]  # Load from logs

plt.plot(epochs, scores)
plt.axhline(y=0.7, color='r', linestyle='--', label='Original generator threshold')
plt.xlabel("Epoch")
plt.ylabel("Eval Score")
plt.title("Self-Improvement Progress")
plt.legend()
plt.show()
```

## Limitations

1. **Not magic**: The model can't learn what's not in the data
2. **Ceiling effect**: You'll eventually hit a limit
3. **Specialization**: Model will be very good in domain but may forget general knowledge

## Recommendations

1. **Start with good generator**: Llama 3B or Mistral 7B
2. **Quality real data**: It's the anchor that prevents collapse
3. **Evaluate frequently**: Detect problems early
4. **Save checkpoints**: In case you need rollback
5. **Monitor diversity**: If generations become repetitive, there's a problem

## Next Steps

- [Set up cluster](mac_cluster_setup.md)
- [Create new domain](new_domain.md)
