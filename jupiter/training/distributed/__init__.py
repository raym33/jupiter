"""
Training distribuido para Jupiter.

Soporta:
- Clusters de Macs via MLX Distributed
- GPUs NVIDIA via PyTorch DDP
- Hardware híbrido
"""

from jupiter.training.distributed.trainer import DistributedTrainer
from jupiter.training.distributed.coordinator import TrainingCoordinator
from jupiter.training.distributed.gradient_sync import GradientSynchronizer

__all__ = [
    "DistributedTrainer",
    "TrainingCoordinator",
    "GradientSynchronizer",
]
