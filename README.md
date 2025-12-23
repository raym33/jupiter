# Jupiter

**Distributed training framework for creating domain-specific expert language models.**

Train your own 1B parameter model using a cluster of Macs (M4/M5) and/or NVIDIA GPUs, with automatic synthetic data generation and self-improvement cycle.

**NEW: MoE-R (Mixture of Real Experts)** - Multiple specialized models collaborating in real-time!

---

## Features

- **Configurable domains**: Change a YAML file to train experts in any area (Unreal Engine, Python, Chemistry, etc.)
- **Distributed training**: Supports Mac clusters via Thunderbolt and NVIDIA GPUs via NCCL
- **Synthetic data generation**: Uses local models (Llama, Mistral) to generate training data
- **Self-improvement cycle**: The trained model can replace the generator when it surpasses it
- **Heterogeneous hardware**: Combine Macs with different RAM amounts and NVIDIA GPUs
- **MoE-R Swarm**: Multiple expert models collaborating to solve complex multi-domain tasks

---

## MoE-R: Mixture of Real Experts

Jupiter includes an experimental **MoE-R** system where multiple specialized small models (500M-1B each) work together:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    JUPITER MoE-R (Mixture of Real Experts)              │
│                                                                         │
│   Query: "Create a React + FastAPI app with pandas data processing"     │
│                              │                                          │
│                              ▼                                          │
│                    ┌─────────────────┐                                  │
│                    │     ROUTER      │  ← Selects relevant experts      │
│                    └────────┬────────┘                                  │
│                             │                                           │
│         ┌───────────────────┼───────────────────┐                       │
│         ▼                   ▼                   ▼                       │
│   ┌───────────┐       ┌───────────┐       ┌───────────┐                │
│   │  React    │       │  FastAPI  │       │  Pandas   │                │
│   │  Expert   │       │  Expert   │       │  Expert   │                │
│   │  (Mac 1)  │       │  (Mac 2)  │       │  (Mac 3)  │                │
│   └─────┬─────┘       └─────┬─────┘       └─────┬─────┘                │
│         │                   │                   │                       │
│         └───────────────────┼───────────────────┘                       │
│                             ▼                                           │
│                    ┌─────────────────┐                                  │
│                    │   SYNTHESIZER   │  ← Combines expert responses     │
│                    └────────┬────────┘                                  │
│                             │                                           │
│                             ▼                                           │
│                    Unified Response                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

### Quick Start with MoE-R

```bash
# List available experts
jupiter swarm list

# Chat with the expert swarm
jupiter swarm chat --experts config/experts/python

# Single query
jupiter swarm query "How do I create a FastAPI endpoint with Pydantic validation?"

# Create a new expert
jupiter swarm create-expert my-expert --domain python
```

### Included Expert Configurations

| Expert | Domain | Specialization |
|--------|--------|----------------|
| `python-core` | Python | Language fundamentals, stdlib, typing |
| `python-datascience` | Data Science | pandas, numpy, matplotlib |
| `python-backend` | Web Backend | FastAPI, Django, Flask, databases |
| `react-frontend` | Frontend | React, TypeScript, hooks, state |
| `api-architect` | API Design | REST, GraphQL, system design |

---

## Hardware Requirements

### Minimum Configuration (Macs Only)
| Device | RAM | Role |
|--------|-----|------|
| 1× Mac (M4/M5) | 16-24GB | Data generator |
| 1-4× Mac Mini (M4/M5) | 16GB+ each | Training nodes |

### Hybrid Configuration (Macs + NVIDIA)
| Device | VRAM/RAM | Role |
|--------|----------|------|
| Macs | 16GB+ | Generation + MLX Training |
| RTX 3090/4090 | 24GB | PyTorch Training |

### Connections
- **Between Macs**: Thunderbolt 4/5 (recommended) or 10GbE Ethernet
- **Macs ↔ NVIDIA**: Ethernet (data syncs, not gradients directly)

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/raym33/jupiter.git
cd jupiter
```

### 2. Install dependencies (Mac)
```bash
# Install Xcode CLI tools
xcode-select --install

# Install system dependencies
brew install mpich

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install Jupiter
pip install -e ".[mac]"
```

### 3. Install dependencies (NVIDIA Linux)
```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install Jupiter with CUDA support
pip install -e ".[nvidia]"
```

---

## Quick Start

### Option A: Train a Single Expert Model

#### Step 1: Configure your domain

```bash
jupiter new-domain python_expert
```

Edit `config/domains/python_expert.yaml` with your data sources.

#### Step 2: Configure the cluster

Edit `config/cluster.yaml`:
```yaml
nodes:
  - host: "macbook.local"
    role: "generator"
    memory_gb: 24

  - host: "mac-mini-1.local"
    role: "trainer"
    memory_gb: 16
```

#### Step 3: Start training

```bash
# Interactive mode (recommended to start)
jupiter start --domain python_expert --interactive

# Automatic mode (full cycle)
jupiter start --domain python_expert --auto
```

### Option B: Run MoE-R Expert Swarm

```bash
# Check system status
jupiter swarm status

# Start interactive chat with Python experts
jupiter swarm chat --experts config/experts/python

# Query with specific experts
jupiter swarm query "Explain decorators in Python" -l python-core
```

---

## Architecture

### Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                         JUPITER                                 │
│                                                                 │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐    │
│   │  COLLECTOR  │      │  SYNTHETIC  │      │   QUALITY   │    │
│   │  real data  │ ───► │  GENERATOR  │ ───► │   FILTER    │    │
│   │             │      │  (Llama)    │      │             │    │
│   └─────────────┘      └─────────────┘      └──────┬──────┘    │
│                                                    │           │
│                                              ┌─────▼─────┐     │
│                                              │   DATA    │     │
│                                              │   QUEUE   │     │
│                                              └─────┬─────┘     │
│                                                    │           │
│   ┌────────────────────────────────────────────────┼─────────┐ │
│   │              DISTRIBUTED TRAINING              │         │ │
│   │                                                ▼         │ │
│   │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │ │
│   │   │ Mac #1  │  │ Mac #2  │  │ Mac #3  │  │ RTX GPU │    │ │
│   │   │  MLX    │  │  MLX    │  │  MLX    │  │ PyTorch │    │ │
│   │   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘    │ │
│   │        └────────────┴─────┬──────┴────────────┘          │ │
│   │                           │                              │ │
│   │                    ┌──────▼──────┐                       │ │
│   │                    │  GRADIENT   │                       │ │
│   │                    │    SYNC     │                       │ │
│   │                    └─────────────┘                       │ │
│   └──────────────────────────────────────────────────────────┘ │
│                                                                │
│                        ┌──────────────┐                        │
│                        │ SELF-IMPROVE │                        │
│                        │    CYCLE     │ ──► Model replaces     │
│                        └──────────────┘     generator          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
jupiter/
├── config/
│   ├── cluster.yaml           # Your cluster configuration
│   ├── domains/
│   │   ├── _template.yaml     # Template for new domains
│   │   └── unreal_engine.yaml # Example: UE5 expert
│   └── experts/               # MoE-R expert configurations
│       ├── python/            # Python experts
│       ├── fullstack/         # Fullstack experts
│       └── gamedev/           # Game dev experts
│
├── jupiter/
│   ├── config/                # Configuration system
│   ├── data/
│   │   ├── collectors/        # Real data collectors
│   │   ├── generators/        # Synthetic data generators
│   │   ├── filters/           # Quality filters
│   │   └── queue/             # Data queue for training
│   ├── training/
│   │   ├── model/             # Model architecture (MLX)
│   │   ├── distributed/       # Distributed training
│   │   └── evaluation/        # Benchmarks and evaluation
│   ├── swarm/                 # MoE-R system
│   │   ├── expert.py          # Expert agent
│   │   ├── router.py          # Query router
│   │   ├── synthesizer.py     # Response combiner
│   │   └── swarm.py           # Main orchestrator
│   └── orchestrator/          # Training orchestration
│
├── docs/                      # Documentation
└── tests/                     # Tests
```

---

## CLI Commands

### Training Commands

```bash
# Start training pipeline
jupiter start --domain <name> [--auto|--interactive]

# Check configuration
jupiter check

# Create new domain
jupiter new-domain <name>

# Show model info
jupiter model-info --preset 1b
```

### Swarm Commands (MoE-R)

```bash
# List available experts
jupiter swarm list

# Check swarm status
jupiter swarm status

# Interactive chat
jupiter swarm chat [--experts <dir>] [-l <expert-name>]

# Single query
jupiter swarm query "<question>" [-l <expert-name>] [-k <top-k>]

# Create new expert config
jupiter swarm create-expert <name> --domain <domain>

# Train an expert (coming soon)
jupiter swarm train -e <name> -d <domain-config>
```

---

## Example Domains

| Domain | Description | Config |
|--------|-------------|--------|
| `unreal_engine` | Expert in game development with UE5 | [View](config/domains/unreal_engine.yaml) |
| `python` | Expert in Python programming | Coming soon |
| `chemistry` | Expert in university chemistry | Coming soon |

---

## Documentation

- [Examples & Use Cases](docs/examples.md) - Complete examples of what Jupiter can do
- [MoE-R Swarm Guide](docs/swarm_moer.md) - Detailed guide on expert collaboration
- [**Reaching 90% on LiveCodeBench**](docs/livecodebench_90_percent_guide.md) - Comprehensive guide to building a competitive programming expert system
- [Mac Cluster Setup](docs/mac_cluster_setup.md)
- [Adding NVIDIA GPUs](docs/nvidia_setup.md)
- [Creating a New Domain](docs/new_domain.md)
- [Understanding Self-Improvement](docs/self_improvement.md)
- [Spanish Documentation](docs/español/)

---

## Roadmap

- [x] Core training pipeline
- [x] MLX distributed training
- [x] Synthetic data generation
- [x] Self-improvement cycle
- [x] MoE-R swarm system
- [ ] PyTorch model implementation
- [ ] GGUF export for llama.cpp
- [ ] Web UI for monitoring
- [ ] Expert debate/refinement rounds

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT License - see [LICENSE](LICENSE)

---

## Credits

- [MLX](https://github.com/ml-explore/mlx) - ML framework for Apple Silicon
- [Exo](https://github.com/exo-explore/exo) - Inspiration for distributed networking
- [mlx-lm](https://github.com/ml-explore/mlx-lm) - Fine-tuning with MLX
