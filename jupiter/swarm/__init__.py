"""
Jupiter Swarm - Mixture of Real Experts (MoE-R)

A system where multiple specialized small models collaborate in real-time
to solve complex tasks that span multiple domains.

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                      SWARM                               │
    │                                                          │
    │   Query → Router → [Expert₁, Expert₂, ...] → Synthesizer │
    │                                                          │
    └─────────────────────────────────────────────────────────┘

Components:
    - Expert: Specialized model trained on a specific domain
    - Router: Decides which experts to activate for a query
    - Synthesizer: Combines expert outputs into coherent response
    - Swarm: Orchestrates the entire collaboration
"""

from jupiter.swarm.expert import Expert, ExpertConfig
from jupiter.swarm.router import Router, RouterConfig
from jupiter.swarm.synthesizer import Synthesizer, SynthesizerConfig
from jupiter.swarm.swarm import Swarm, SwarmConfig

__all__ = [
    "Expert",
    "ExpertConfig",
    "Router",
    "RouterConfig",
    "Synthesizer",
    "SynthesizerConfig",
    "Swarm",
    "SwarmConfig",
]
