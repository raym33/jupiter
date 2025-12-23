"""
Módulo de datos de Jupiter.

Incluye:
- Recolectores de datos reales (web, GitHub, docs)
- Generadores de datos sintéticos
- Filtros de calidad
- Cola de datos para training
"""

from jupiter.data.collectors import DataCollector
from jupiter.data.generators import SyntheticGenerator
from jupiter.data.filters import QualityFilter
from jupiter.data.queue import DataQueue

__all__ = [
    "DataCollector",
    "SyntheticGenerator",
    "QualityFilter",
    "DataQueue",
]
