"""
Orchestrador del pipeline de Jupiter.

Maneja el ciclo completo:
1. Recolección de datos
2. Generación sintética
3. Training distribuido
4. Evaluación
5. Auto-mejora (reemplazo del generador)
"""

from jupiter.orchestrator.pipeline import Orchestrator
from jupiter.orchestrator.self_improve import SelfImprover

__all__ = [
    "Orchestrator",
    "SelfImprover",
]
