"""
Jupiter - Framework de entrenamiento distribuido para modelos de lenguaje expertos

Entrena modelos de 1B parámetros en clusters de Macs y/o GPUs NVIDIA,
con generación automática de datos sintéticos y ciclo de auto-mejora.
"""

__version__ = "0.1.0"
__author__ = "Jupiter Contributors"

from jupiter.config import JupiterConfig, DomainConfig, ClusterConfig
from jupiter.orchestrator import Orchestrator

__all__ = [
    "JupiterConfig",
    "DomainConfig",
    "ClusterConfig",
    "Orchestrator",
    "__version__",
]
