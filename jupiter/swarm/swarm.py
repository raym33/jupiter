"""
Swarm - Main orchestrator for the MoE-R system.

The Swarm manages the entire lifecycle:
1. Loading and distributing experts across cluster nodes
2. Routing queries to appropriate experts
3. Coordinating parallel execution
4. Synthesizing responses
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, AsyncIterator
from pathlib import Path
from enum import Enum
import asyncio
import json
import time

from jupiter.swarm.expert import Expert, ExpertConfig, ExpertResponse, ExpertStatus
from jupiter.swarm.router import Router, RouterConfig, RoutingDecision
from jupiter.swarm.synthesizer import Synthesizer, SynthesizerConfig, SynthesizedResponse


class SwarmMode(str, Enum):
    """Operating mode of the swarm."""
    INFERENCE = "inference"  # Only answer queries
    TRAINING = "training"  # Train experts on their domains
    HYBRID = "hybrid"  # Both training and inference


@dataclass
class SwarmConfig:
    """Configuration for the swarm system."""

    # Identity
    name: str = "jupiter-swarm"

    # Mode
    mode: SwarmMode = SwarmMode.INFERENCE

    # Expert configuration files directory
    experts_dir: str = "config/experts"

    # Component configs
    router: RouterConfig = field(default_factory=RouterConfig)
    synthesizer: SynthesizerConfig = field(default_factory=SynthesizerConfig)

    # Execution settings
    max_parallel_experts: int = 4
    timeout_seconds: float = 60.0

    # Collaboration settings
    enable_expert_collaboration: bool = True
    collaboration_rounds: int = 1  # How many rounds of inter-expert communication

    # Auto-scaling
    auto_load_experts: bool = True  # Load experts on demand
    unload_idle_experts: bool = True  # Unload after inactivity
    idle_timeout_seconds: float = 300.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SwarmConfig":
        """Create from dictionary."""
        config_data = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}

        if "router" in data:
            config_data["router"] = RouterConfig.from_dict(data["router"])
        if "synthesizer" in data:
            config_data["synthesizer"] = SynthesizerConfig.from_dict(data["synthesizer"])
        if "mode" in data and isinstance(data["mode"], str):
            config_data["mode"] = SwarmMode(data["mode"])

        return cls(**config_data)

    @classmethod
    def from_yaml(cls, path: str) -> "SwarmConfig":
        """Load from YAML file."""
        import yaml
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))


@dataclass
class SwarmStats:
    """Statistics about swarm operation."""

    total_queries: int = 0
    total_tokens_generated: int = 0
    total_latency_ms: float = 0.0
    expert_usage: Dict[str, int] = field(default_factory=dict)
    avg_experts_per_query: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "total_tokens_generated": self.total_tokens_generated,
            "avg_latency_ms": self.total_latency_ms / max(1, self.total_queries),
            "expert_usage": self.expert_usage,
            "avg_experts_per_query": self.avg_experts_per_query,
        }


class Swarm:
    """
    Main orchestrator for the Mixture of Real Experts system.

    The Swarm coordinates multiple specialized expert models running
    across a distributed cluster to collaboratively answer queries.

    Architecture:
        User Query
            │
            ▼
        ┌───────┐
        │ Router │ ── Selects relevant experts
        └───┬───┘
            │
            ▼
        ┌─────────────────────────────┐
        │   Parallel Expert Execution  │
        │  ┌─────┐ ┌─────┐ ┌─────┐   │
        │  │Exp 1│ │Exp 2│ │Exp 3│   │
        │  └─────┘ └─────┘ └─────┘   │
        └─────────────┬───────────────┘
                      │
                      ▼
              ┌─────────────┐
              │ Synthesizer │ ── Merges responses
              └──────┬──────┘
                     │
                     ▼
              Final Response
    """

    def __init__(self, config: SwarmConfig):
        self.config = config
        self.router = Router(config.router)
        self.synthesizer = Synthesizer(config.synthesizer)
        self._experts: Dict[str, Expert] = {}
        self._expert_configs: Dict[str, ExpertConfig] = {}
        self._stats = SwarmStats()
        self._running = False

    @property
    def experts(self) -> List[Expert]:
        """Get all registered experts."""
        return list(self._experts.values())

    @property
    def stats(self) -> SwarmStats:
        """Get swarm statistics."""
        return self._stats

    async def initialize(self) -> None:
        """Initialize the swarm system."""
        print(f"Initializing swarm: {self.config.name}")

        # Initialize components
        await self.router.initialize()
        await self.synthesizer.initialize()

        # Load expert configurations
        if self.config.auto_load_experts:
            await self._discover_experts()

        self._running = True
        print(f"Swarm ready with {len(self._expert_configs)} expert configurations")

    async def _discover_experts(self) -> None:
        """Discover expert configurations from directory."""
        experts_path = Path(self.config.experts_dir)

        if not experts_path.exists():
            print(f"Experts directory not found: {experts_path}")
            return

        for config_file in experts_path.glob("*.yaml"):
            try:
                config = ExpertConfig.from_yaml(str(config_file))
                self._expert_configs[config.name] = config
                print(f"  Discovered expert: {config.name} ({config.domain})")
            except Exception as e:
                print(f"  Error loading {config_file}: {e}")

    def register_expert_config(self, config: ExpertConfig) -> None:
        """Register an expert configuration (doesn't load the model yet)."""
        self._expert_configs[config.name] = config

    async def load_expert(self, name: str) -> Expert:
        """Load an expert by name."""
        if name in self._experts:
            return self._experts[name]

        if name not in self._expert_configs:
            raise ValueError(f"Unknown expert: {name}")

        config = self._expert_configs[name]
        expert = Expert(config)

        await expert.load()
        self._experts[name] = expert
        self.router.register_expert(expert)

        return expert

    async def unload_expert(self, name: str) -> None:
        """Unload an expert to free memory."""
        if name in self._experts:
            expert = self._experts[name]
            await expert.unload()
            self.router.unregister_expert(name)
            del self._experts[name]

    async def query(
        self,
        query: str,
        context: Optional[str] = None,
        stream: bool = False,
    ) -> SynthesizedResponse:
        """
        Process a query through the swarm.

        Args:
            query: User's question or task
            context: Optional additional context
            stream: Whether to stream the response (not yet implemented)

        Returns:
            SynthesizedResponse with the combined answer
        """
        start_time = time.time()

        # 1. Route the query
        routing = await self.router.route(query, context)
        print(f"Routing: {routing.selected_experts}")

        # 2. Ensure selected experts are loaded
        for expert_name in routing.selected_experts:
            if expert_name not in self._experts:
                try:
                    await self.load_expert(expert_name)
                except Exception as e:
                    print(f"Failed to load expert {expert_name}: {e}")
                    routing.selected_experts.remove(expert_name)

        if not routing.selected_experts:
            return SynthesizedResponse(
                content="No experts available to answer this query.",
                expert_contributions={},
                strategy_used=self.synthesizer.config.strategy,
                total_tokens=0,
                total_latency_ms=0,
            )

        # 3. Execute experts
        if self.config.enable_expert_collaboration and self.config.collaboration_rounds > 0:
            responses = await self._execute_collaborative(query, context, routing)
        else:
            responses = await self._execute_parallel(query, context, routing)

        # 4. Synthesize responses
        result = await self.synthesizer.synthesize(query, responses)

        # 5. Update stats
        self._update_stats(routing, result, time.time() - start_time)

        return result

    async def _execute_parallel(
        self,
        query: str,
        context: Optional[str],
        routing: RoutingDecision,
    ) -> List[ExpertResponse]:
        """Execute experts in parallel without collaboration."""
        tasks = []

        for expert_name in routing.selected_experts:
            expert = self._experts.get(expert_name)
            if expert and expert.status == ExpertStatus.READY:
                tasks.append(expert.generate(query, context))

        # Execute with timeout
        try:
            responses = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.timeout_seconds,
            )

            # Filter out exceptions
            return [r for r in responses if isinstance(r, ExpertResponse)]

        except asyncio.TimeoutError:
            print(f"Timeout waiting for experts")
            return []

    async def _execute_collaborative(
        self,
        query: str,
        context: Optional[str],
        routing: RoutingDecision,
    ) -> List[ExpertResponse]:
        """
        Execute experts with collaboration rounds.

        In each round, experts can see previous responses and refine their answers.
        """
        all_responses: List[ExpertResponse] = []

        for round_num in range(self.config.collaboration_rounds + 1):
            tasks = []
            previous_responses = all_responses if round_num > 0 else None

            for expert_name in routing.selected_experts:
                expert = self._experts.get(expert_name)
                if expert and expert.status == ExpertStatus.READY:
                    tasks.append(
                        expert.generate(query, context, previous_responses)
                    )

            try:
                responses = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.config.timeout_seconds,
                )

                all_responses = [r for r in responses if isinstance(r, ExpertResponse)]

            except asyncio.TimeoutError:
                print(f"Timeout in collaboration round {round_num}")
                break

        return all_responses

    def _update_stats(
        self,
        routing: RoutingDecision,
        result: SynthesizedResponse,
        elapsed: float,
    ) -> None:
        """Update swarm statistics."""
        self._stats.total_queries += 1
        self._stats.total_tokens_generated += result.total_tokens
        self._stats.total_latency_ms += elapsed * 1000

        for expert_name in routing.selected_experts:
            self._stats.expert_usage[expert_name] = (
                self._stats.expert_usage.get(expert_name, 0) + 1
            )

        total_experts_used = sum(self._stats.expert_usage.values())
        self._stats.avg_experts_per_query = total_experts_used / self._stats.total_queries

    async def chat(self, stream: bool = False) -> AsyncIterator[str]:
        """
        Interactive chat mode with the swarm.

        Yields response chunks if streaming, otherwise yields complete response.
        """
        print(f"\n{'='*60}")
        print(f"Jupiter Swarm - {self.config.name}")
        print(f"Experts available: {list(self._expert_configs.keys())}")
        print(f"Type 'quit' to exit, 'stats' to see statistics")
        print(f"{'='*60}\n")

        while self._running:
            try:
                query = input("You: ").strip()

                if not query:
                    continue
                if query.lower() == "quit":
                    break
                if query.lower() == "stats":
                    print(json.dumps(self._stats.to_dict(), indent=2))
                    continue

                print("\nSwarm: ", end="", flush=True)
                response = await self.query(query)
                print(response.content)
                print(f"\n[Experts: {list(response.expert_contributions.keys())}]\n")

                yield response.content

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\nError: {e}\n")

    async def shutdown(self) -> None:
        """Shutdown the swarm gracefully."""
        self._running = False

        # Unload all experts
        for name in list(self._experts.keys()):
            await self.unload_expert(name)

        print("Swarm shutdown complete")

    def __repr__(self) -> str:
        loaded = len(self._experts)
        available = len(self._expert_configs)
        return f"Swarm(name='{self.config.name}', experts={loaded}/{available} loaded)"


# =============================================================================
# Factory functions for creating pre-configured swarms
# =============================================================================

def create_python_swarm() -> Swarm:
    """Create a swarm specialized for Python development."""
    config = SwarmConfig(
        name="python-experts",
        experts_dir="config/experts/python",
    )
    return Swarm(config)


def create_fullstack_swarm() -> Swarm:
    """Create a swarm for full-stack web development."""
    config = SwarmConfig(
        name="fullstack-experts",
        experts_dir="config/experts/fullstack",
    )
    return Swarm(config)


def create_game_dev_swarm() -> Swarm:
    """Create a swarm for game development."""
    config = SwarmConfig(
        name="gamedev-experts",
        experts_dir="config/experts/gamedev",
    )
    return Swarm(config)
