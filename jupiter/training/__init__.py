"""
Módulo de training de Jupiter.

Incluye:
- Arquitectura del modelo (transformers)
- Training distribuido (MLX + PyTorch)
- Evaluación y benchmarks
"""

from jupiter.training.model import JupiterModel, ModelConfig
from jupiter.training.distributed import DistributedTrainer

__all__ = [
    "JupiterModel",
    "ModelConfig",
    "DistributedTrainer",
]
