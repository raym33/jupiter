# Jupiter

**Distributed training framework for creating domain-specific expert language models.**

Train your own 1B parameter model using a cluster of Macs (M4/M5) and/or NVIDIA GPUs, with automatic synthetic data generation and self-improvement cycle.

---

## Features

- **Configurable domains**: Change a YAML file to train experts in any area (Unreal Engine, Python, Chemistry, etc.)
- **Distributed training**: Supports Mac clusters via Thunderbolt and NVIDIA GPUs via NCCL
- **Synthetic data generation**: Uses local models (Llama, Mistral) to generate training data
- **Self-improvement cycle**: The trained model can replace the generator when it surpasses it
- **Heterogeneous hardware**: Combine Macs with different RAM amounts and NVIDIA GPUs

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

### Step 1: Configure your domain

Copy the template and edit it:
```bash
cp config/domains/_template.yaml config/domains/my_domain.yaml
```

Edit `my_domain.yaml` with your domain's data sources and configuration.

### Step 2: Configure the cluster

Edit `config/cluster.yaml` with your devices:
```yaml
nodes:
  - host: "macbook.local"
    role: "generator"
    memory_gb: 24

  - host: "mac-mini-1.local"
    role: "trainer"
    memory_gb: 16

  # Add more nodes...
```

### Step 3: Start training

```bash
# Interactive mode (recommended to start)
jupiter start --domain my_domain --interactive

# Automatic mode (full cycle)
jupiter start --domain my_domain --auto
```

---

## Architecture

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
│   │        │            │            │            │          │ │
│   │        └────────────┴─────┬──────┴────────────┘          │ │
│   │                           │                              │ │
│   │                    ┌──────▼──────┐                       │ │
│   │                    │  GRADIENT   │                       │ │
│   │                    │    SYNC     │                       │ │
│   │                    └──────┬──────┘                       │ │
│   │                           │                              │ │
│   └───────────────────────────┼──────────────────────────────┘ │
│                               │                                │
│                        ┌──────▼──────┐                         │
│                        │ CHECKPOINT  │                         │
│                        │   + EVAL    │                         │
│                        └──────┬──────┘                         │
│                               │                                │
│                        ┌──────▼──────┐                         │
│                        │  SURPASSES  │                         │
│                        │ GENERATOR?  │                         │
│                        └──────┬──────┘                         │
│                               │ Yes                            │
│                        ┌──────▼──────┐                         │
│                        │  REPLACE    │                         │
│                        │  GENERATOR  │ ──► CYCLE CONTINUES     │
│                        └─────────────┘                         │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
jupiter/
├── config/
│   ├── cluster.yaml           # Your cluster configuration
│   └── domains/
│       ├── _template.yaml     # Template for new domains
│       └── unreal_engine.yaml # Example: UE5 expert
│
├── jupiter/
│   ├── config/                # Configuration system
│   ├── data/
│   │   ├── collectors/        # Real data collectors
│   │   ├── generators/        # Synthetic data generators
│   │   ├── filters/           # Quality filters
│   │   └── queue/             # Data queue for training
│   ├── training/
│   │   ├── model/             # Model architecture
│   │   ├── distributed/       # Distributed training (MLX + PyTorch)
│   │   └── evaluation/        # Benchmarks and evaluation
│   └── orchestrator/          # Full cycle orchestration
│
├── scripts/                   # Utility scripts
├── tests/                     # Tests
└── docs/                      # Extended documentation
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

- [Mac Cluster Setup](docs/mac_cluster_setup.md)
- [Adding NVIDIA GPUs](docs/nvidia_setup.md)
- [Creating a New Domain](docs/new_domain.md)
- [Understanding Self-Improvement](docs/self_improvement.md)
- [Documentación en Español](docs/español/)

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
