"""
Synthesizer - Combines expert responses into coherent output.

The synthesizer takes responses from multiple experts and:
1. Resolves conflicts between expert opinions
2. Merges complementary information
3. Produces a unified, coherent response
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum

from jupiter.swarm.expert import ExpertResponse


class SynthesisStrategy(str, Enum):
    """Strategy for combining expert responses."""
    CONCATENATE = "concatenate"  # Simple concatenation with headers
    WEIGHTED = "weighted"  # Weight by confidence scores
    LLM_MERGE = "llm_merge"  # Use LLM to intelligently merge
    BEST_ONLY = "best_only"  # Only use highest confidence response
    DEBATE = "debate"  # Multi-round refinement between experts


@dataclass
class SynthesizerConfig:
    """Configuration for the synthesizer."""

    strategy: SynthesisStrategy = SynthesisStrategy.LLM_MERGE

    # For weighted strategy
    confidence_threshold: float = 0.5

    # For LLM merge
    merge_model: Optional[str] = None

    # For debate strategy
    debate_rounds: int = 2

    # Output format
    include_expert_attribution: bool = True
    max_output_length: int = 4096

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SynthesizerConfig":
        if "strategy" in data and isinstance(data["strategy"], str):
            data["strategy"] = SynthesisStrategy(data["strategy"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SynthesizedResponse:
    """Final synthesized response from the swarm."""

    content: str
    expert_contributions: Dict[str, float]  # Expert -> contribution weight
    strategy_used: SynthesisStrategy
    total_tokens: int
    total_latency_ms: float

    # Original expert responses (for debugging/analysis)
    expert_responses: List[ExpertResponse] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "expert_contributions": self.expert_contributions,
            "strategy": self.strategy_used.value,
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
            "experts_used": [r.expert_name for r in self.expert_responses],
        }


class Synthesizer:
    """
    Combines multiple expert responses into a coherent output.

    The synthesizer is the final stage of the MoE-R pipeline,
    responsible for producing a unified response that leverages
    the best of each expert's contribution.
    """

    def __init__(self, config: SynthesizerConfig):
        self.config = config
        self._merge_model = None
        self._merge_tokenizer = None

    async def initialize(self) -> None:
        """Initialize synthesis models if needed."""
        if self.config.strategy == SynthesisStrategy.LLM_MERGE:
            await self._load_merge_model()

    async def _load_merge_model(self) -> None:
        """Load the LLM for intelligent merging."""
        if self.config.merge_model:
            try:
                from mlx_lm import load
                self._merge_model, self._merge_tokenizer = load(self.config.merge_model)
            except Exception as e:
                print(f"Warning: Could not load merge model: {e}")

    async def synthesize(
        self,
        query: str,
        responses: List[ExpertResponse],
    ) -> SynthesizedResponse:
        """
        Synthesize expert responses into a unified output.

        Args:
            query: Original user query
            responses: List of expert responses

        Returns:
            SynthesizedResponse with merged content
        """
        if not responses:
            return SynthesizedResponse(
                content="No expert responses available.",
                expert_contributions={},
                strategy_used=self.config.strategy,
                total_tokens=0,
                total_latency_ms=0,
                expert_responses=[],
            )

        strategy = self.config.strategy

        if strategy == SynthesisStrategy.CONCATENATE:
            return await self._synthesize_concatenate(query, responses)
        elif strategy == SynthesisStrategy.WEIGHTED:
            return await self._synthesize_weighted(query, responses)
        elif strategy == SynthesisStrategy.LLM_MERGE:
            return await self._synthesize_llm_merge(query, responses)
        elif strategy == SynthesisStrategy.BEST_ONLY:
            return await self._synthesize_best_only(query, responses)
        elif strategy == SynthesisStrategy.DEBATE:
            return await self._synthesize_debate(query, responses)
        else:
            return await self._synthesize_concatenate(query, responses)

    async def _synthesize_concatenate(
        self, query: str, responses: List[ExpertResponse]
    ) -> SynthesizedResponse:
        """Simple concatenation with expert headers."""
        parts = []

        # Sort by confidence
        sorted_responses = sorted(responses, key=lambda r: r.confidence, reverse=True)

        for resp in sorted_responses:
            if self.config.include_expert_attribution:
                parts.append(f"## {resp.expert_name} ({resp.domain}) [confidence: {resp.confidence:.2f}]\n")
            parts.append(resp.content)
            parts.append("\n\n")

        content = "".join(parts).strip()

        # Calculate contributions (equal for concatenation)
        contributions = {r.expert_name: 1.0 / len(responses) for r in responses}

        return SynthesizedResponse(
            content=content,
            expert_contributions=contributions,
            strategy_used=SynthesisStrategy.CONCATENATE,
            total_tokens=sum(r.tokens_used for r in responses),
            total_latency_ms=max(r.latency_ms for r in responses),  # Parallel execution
            expert_responses=responses,
        )

    async def _synthesize_weighted(
        self, query: str, responses: List[ExpertResponse]
    ) -> SynthesizedResponse:
        """Weight responses by confidence, filter low confidence."""
        # Filter by confidence threshold
        filtered = [r for r in responses if r.confidence >= self.config.confidence_threshold]

        if not filtered:
            filtered = responses[:1]  # At least one response

        # Normalize confidence to weights
        total_conf = sum(r.confidence for r in filtered)
        contributions = {r.expert_name: r.confidence / total_conf for r in filtered}

        # Build weighted response (highest confidence first, longer sections)
        sorted_responses = sorted(filtered, key=lambda r: r.confidence, reverse=True)

        parts = []
        for resp in sorted_responses:
            weight = contributions[resp.expert_name]
            # More confident experts get more space
            max_chars = int(self.config.max_output_length * weight)
            excerpt = resp.content[:max_chars]

            if self.config.include_expert_attribution:
                parts.append(f"### {resp.expert_name} ({resp.domain})\n")
            parts.append(excerpt)
            if len(resp.content) > max_chars:
                parts.append("...")
            parts.append("\n\n")

        return SynthesizedResponse(
            content="".join(parts).strip(),
            expert_contributions=contributions,
            strategy_used=SynthesisStrategy.WEIGHTED,
            total_tokens=sum(r.tokens_used for r in filtered),
            total_latency_ms=max(r.latency_ms for r in filtered),
            expert_responses=filtered,
        )

    async def _synthesize_llm_merge(
        self, query: str, responses: List[ExpertResponse]
    ) -> SynthesizedResponse:
        """Use LLM to intelligently merge responses."""
        if self._merge_model is None:
            # Fall back to weighted if no model
            return await self._synthesize_weighted(query, responses)

        # Build merge prompt
        expert_responses_text = "\n\n".join([
            f"[{r.expert_name} - {r.domain} - confidence: {r.confidence:.2f}]\n{r.content}"
            for r in responses
        ])

        prompt = f"""<|im_start|>system
You are a synthesis AI. Your task is to combine multiple expert responses into a single, coherent, comprehensive answer.

Guidelines:
- Merge complementary information from different experts
- Resolve any conflicts by favoring higher-confidence experts
- Maintain code examples and technical accuracy
- Create a unified response that reads naturally
- Do not simply concatenate - truly synthesize the information<|im_end|>
<|im_start|>user
Original question: {query}

Expert responses to synthesize:
{expert_responses_text}

Please synthesize these into a single comprehensive response.<|im_end|>
<|im_start|>assistant
"""

        try:
            from mlx_lm import generate
            import time

            start = time.time()
            merged_content = generate(
                self._merge_model,
                self._merge_tokenizer,
                prompt=prompt,
                max_tokens=self.config.max_output_length,
                temp=0.3,  # Lower temp for consistent merging
            )
            merge_latency = (time.time() - start) * 1000

            # Calculate contributions based on confidence
            total_conf = sum(r.confidence for r in responses)
            contributions = {r.expert_name: r.confidence / total_conf for r in responses}

            return SynthesizedResponse(
                content=merged_content,
                expert_contributions=contributions,
                strategy_used=SynthesisStrategy.LLM_MERGE,
                total_tokens=sum(r.tokens_used for r in responses) + len(self._merge_tokenizer.encode(merged_content)),
                total_latency_ms=max(r.latency_ms for r in responses) + merge_latency,
                expert_responses=responses,
            )

        except Exception as e:
            print(f"LLM merge failed: {e}, falling back to weighted")
            return await self._synthesize_weighted(query, responses)

    async def _synthesize_best_only(
        self, query: str, responses: List[ExpertResponse]
    ) -> SynthesizedResponse:
        """Use only the highest confidence response."""
        best = max(responses, key=lambda r: r.confidence)

        return SynthesizedResponse(
            content=best.content,
            expert_contributions={best.expert_name: 1.0},
            strategy_used=SynthesisStrategy.BEST_ONLY,
            total_tokens=best.tokens_used,
            total_latency_ms=best.latency_ms,
            expert_responses=[best],
        )

    async def _synthesize_debate(
        self, query: str, responses: List[ExpertResponse]
    ) -> SynthesizedResponse:
        """
        Multi-round debate/refinement between experts.

        This is an advanced strategy where experts iteratively
        improve their responses based on feedback from others.
        """
        # For now, implement as weighted + note about debate potential
        result = await self._synthesize_weighted(query, responses)

        # TODO: Implement actual debate rounds
        # 1. Share initial responses with all experts
        # 2. Each expert refines based on others' input
        # 3. Repeat for N rounds
        # 4. Final synthesis

        return SynthesizedResponse(
            content=result.content,
            expert_contributions=result.expert_contributions,
            strategy_used=SynthesisStrategy.DEBATE,
            total_tokens=result.total_tokens,
            total_latency_ms=result.total_latency_ms,
            expert_responses=result.expert_responses,
        )
