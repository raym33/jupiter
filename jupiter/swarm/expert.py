"""
Expert - Specialized model agent in the MoE-R system.

Each expert is a small model (~500M-1B) trained on a specific domain.
Experts run on individual nodes in the cluster and communicate via network.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
from enum import Enum
import asyncio
import json


class ExpertStatus(str, Enum):
    """Expert operational status."""
    OFFLINE = "offline"
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"


@dataclass
class ExpertConfig:
    """Configuration for an expert agent."""

    # Identity
    name: str
    domain: str  # e.g., "python", "react", "api_design"
    description: str = ""

    # Model
    model_path: Optional[str] = None  # Path to trained model
    base_model: str = "jupiter-500m"  # Preset if no custom model

    # Specialization keywords (for routing)
    keywords: List[str] = field(default_factory=list)

    # Capabilities (what this expert can do)
    capabilities: List[str] = field(default_factory=list)

    # Node assignment
    assigned_node: Optional[str] = None  # Hostname or IP

    # Generation parameters
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9

    # System prompt for this expert
    system_prompt: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExpertConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_yaml(cls, path: str) -> "ExpertConfig":
        """Load from YAML file."""
        import yaml
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "domain": self.domain,
            "description": self.description,
            "model_path": self.model_path,
            "base_model": self.base_model,
            "keywords": self.keywords,
            "capabilities": self.capabilities,
            "assigned_node": self.assigned_node,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "system_prompt": self.system_prompt,
        }


@dataclass
class ExpertResponse:
    """Response from an expert."""

    expert_name: str
    domain: str
    content: str
    confidence: float  # 0.0 - 1.0
    reasoning: Optional[str] = None  # Why this expert thinks it's relevant
    tokens_used: int = 0
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expert_name": self.expert_name,
            "domain": self.domain,
            "content": self.content,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
        }


class Expert:
    """
    Specialized expert agent in the MoE-R system.

    Each expert:
    1. Receives queries routed to it
    2. Generates domain-specific responses
    3. Reports confidence in its response
    4. Can collaborate with other experts
    """

    def __init__(self, config: ExpertConfig):
        self.config = config
        self.status = ExpertStatus.OFFLINE
        self._model = None
        self._tokenizer = None

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def domain(self) -> str:
        return self.config.domain

    async def load(self) -> None:
        """Load the expert's model."""
        self.status = ExpertStatus.LOADING

        try:
            if self.config.model_path:
                # Load custom trained model
                await self._load_custom_model()
            else:
                # Load base model with expert system prompt
                await self._load_base_model()

            self.status = ExpertStatus.READY
            print(f"Expert '{self.name}' ({self.domain}) ready")

        except Exception as e:
            self.status = ExpertStatus.ERROR
            raise RuntimeError(f"Failed to load expert {self.name}: {e}")

    async def _load_custom_model(self) -> None:
        """Load a custom trained Jupiter model."""
        from jupiter.training.model.architecture import JupiterModel

        self._model = JupiterModel.load(self.config.model_path)

        # Load tokenizer
        from jupiter.training.model.tokenizer import JupiterTokenizer
        tokenizer_path = Path(self.config.model_path) / "tokenizer"
        if tokenizer_path.exists():
            self._tokenizer = JupiterTokenizer.load(str(tokenizer_path))
        else:
            self._tokenizer = JupiterTokenizer()  # Default

    async def _load_base_model(self) -> None:
        """Load a pre-trained base model (e.g., from mlx-community)."""
        try:
            from mlx_lm import load

            # Map base_model to actual model path
            model_map = {
                "jupiter-500m": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                "jupiter-1b": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                "jupiter-3b": "mlx-community/Qwen2.5-3B-Instruct-4bit",
            }

            model_id = model_map.get(self.config.base_model, self.config.base_model)
            self._model, self._tokenizer = load(model_id)

        except ImportError:
            raise RuntimeError("mlx-lm required for base models: pip install mlx-lm")

    async def generate(
        self,
        query: str,
        context: Optional[str] = None,
        other_expert_responses: Optional[List[ExpertResponse]] = None,
    ) -> ExpertResponse:
        """
        Generate a response for the given query.

        Args:
            query: The user's question or task
            context: Additional context for the query
            other_expert_responses: Responses from other experts (for collaboration)

        Returns:
            ExpertResponse with content and confidence
        """
        if self.status != ExpertStatus.READY:
            raise RuntimeError(f"Expert {self.name} not ready (status: {self.status})")

        self.status = ExpertStatus.BUSY

        try:
            import time
            start = time.time()

            # Build prompt
            prompt = self._build_prompt(query, context, other_expert_responses)

            # Generate
            content, tokens = await self._generate_response(prompt)

            # Estimate confidence based on domain keywords in query
            confidence = self._estimate_confidence(query)

            latency = (time.time() - start) * 1000

            return ExpertResponse(
                expert_name=self.name,
                domain=self.domain,
                content=content,
                confidence=confidence,
                tokens_used=tokens,
                latency_ms=latency,
            )

        finally:
            self.status = ExpertStatus.READY

    def _build_prompt(
        self,
        query: str,
        context: Optional[str],
        other_responses: Optional[List[ExpertResponse]],
    ) -> str:
        """Build the prompt for generation."""

        # System prompt
        system = self.config.system_prompt or self._default_system_prompt()

        # Build user message
        user_msg = query

        if context:
            user_msg = f"Context:\n{context}\n\nQuestion:\n{query}"

        # Include other expert responses if collaborating
        if other_responses:
            collab_context = "\n\nOther experts have provided:\n"
            for resp in other_responses:
                collab_context += f"\n[{resp.expert_name} - {resp.domain}]:\n{resp.content[:500]}...\n"
            user_msg += collab_context
            user_msg += "\n\nPlease provide your specialized perspective, building on or complementing the above."

        # Format for chat model
        prompt = f"""<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user_msg}<|im_end|>
<|im_start|>assistant
"""
        return prompt

    def _default_system_prompt(self) -> str:
        """Generate default system prompt based on config."""
        return f"""You are an expert in {self.domain}. {self.config.description}

Your specializations include: {', '.join(self.config.capabilities) if self.config.capabilities else self.domain}

Provide detailed, accurate, and practical responses in your area of expertise.
If a question is outside your domain, acknowledge this and provide what help you can.
Always include code examples when relevant."""

    async def _generate_response(self, prompt: str) -> tuple[str, int]:
        """Generate response using the loaded model."""

        if hasattr(self._model, '__call__'):
            # Jupiter model
            raise NotImplementedError("Jupiter model generation not yet implemented")
        else:
            # mlx-lm model
            from mlx_lm import generate

            response = generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=self.config.max_tokens,
                temp=self.config.temperature,
                top_p=self.config.top_p,
            )

            tokens = len(self._tokenizer.encode(response))
            return response, tokens

    def _estimate_confidence(self, query: str) -> float:
        """Estimate confidence based on keyword matching."""
        query_lower = query.lower()

        # Count keyword matches
        matches = sum(1 for kw in self.config.keywords if kw.lower() in query_lower)

        if not self.config.keywords:
            return 0.5  # Default confidence

        # Normalize to 0.3-1.0 range
        confidence = 0.3 + (matches / len(self.config.keywords)) * 0.7
        return min(1.0, confidence)

    def matches_query(self, query: str) -> float:
        """
        Score how well this expert matches a query.
        Used by the router to select experts.
        """
        return self._estimate_confidence(query)

    async def unload(self) -> None:
        """Unload the model to free memory."""
        self._model = None
        self._tokenizer = None
        self.status = ExpertStatus.OFFLINE

    def __repr__(self) -> str:
        return f"Expert(name='{self.name}', domain='{self.domain}', status={self.status.value})"
