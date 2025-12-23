"""
Router - Intelligent query routing for MoE-R system.

The router analyzes incoming queries and decides which experts to activate.
It can use different strategies:
1. Keyword matching (fast, no model needed)
2. Embedding similarity (requires embedding model)
3. LLM classification (most accurate, slowest)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import asyncio


class RoutingStrategy(str, Enum):
    """Strategy for routing queries to experts."""
    KEYWORD = "keyword"  # Fast keyword matching
    EMBEDDING = "embedding"  # Semantic similarity
    LLM = "llm"  # LLM-based classification
    HYBRID = "hybrid"  # Combine multiple strategies


@dataclass
class RouterConfig:
    """Configuration for the router."""

    strategy: RoutingStrategy = RoutingStrategy.HYBRID

    # How many experts to activate per query
    top_k: int = 3
    min_confidence: float = 0.3  # Minimum confidence to activate expert

    # For embedding strategy
    embedding_model: Optional[str] = None

    # For LLM strategy
    classifier_model: Optional[str] = None

    # Expert collaboration modes
    enable_collaboration: bool = True  # Experts can see each other's responses
    parallel_execution: bool = True  # Run experts in parallel vs sequential

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RouterConfig":
        if "strategy" in data and isinstance(data["strategy"], str):
            data["strategy"] = RoutingStrategy(data["strategy"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RoutingDecision:
    """Result of routing a query."""

    query: str
    selected_experts: List[str]  # Expert names
    confidence_scores: Dict[str, float]  # Expert -> confidence
    reasoning: str
    strategy_used: RoutingStrategy


class Router:
    """
    Routes queries to appropriate experts in the MoE-R system.

    The router is responsible for:
    1. Analyzing the query to understand domains involved
    2. Selecting the most relevant experts
    3. Deciding on collaboration strategy
    """

    def __init__(self, config: RouterConfig):
        self.config = config
        self._experts: Dict[str, Any] = {}  # name -> Expert
        self._embedding_model = None
        self._classifier_model = None

    def register_expert(self, expert: Any) -> None:
        """Register an expert with the router."""
        self._experts[expert.name] = expert

    def unregister_expert(self, name: str) -> None:
        """Remove an expert from routing."""
        self._experts.pop(name, None)

    @property
    def available_experts(self) -> List[str]:
        """Get names of all registered experts."""
        return list(self._experts.keys())

    async def initialize(self) -> None:
        """Initialize router models if needed."""
        if self.config.strategy in (RoutingStrategy.EMBEDDING, RoutingStrategy.HYBRID):
            await self._load_embedding_model()

        if self.config.strategy in (RoutingStrategy.LLM, RoutingStrategy.HYBRID):
            await self._load_classifier_model()

    async def _load_embedding_model(self) -> None:
        """Load embedding model for semantic routing."""
        try:
            # Use sentence-transformers or similar
            # For now, we'll use a simple approach
            pass
        except Exception as e:
            print(f"Warning: Could not load embedding model: {e}")

    async def _load_classifier_model(self) -> None:
        """Load LLM for classification-based routing."""
        try:
            if self.config.classifier_model:
                from mlx_lm import load
                self._classifier_model, self._classifier_tokenizer = load(
                    self.config.classifier_model
                )
        except Exception as e:
            print(f"Warning: Could not load classifier model: {e}")

    async def route(self, query: str, context: Optional[str] = None) -> RoutingDecision:
        """
        Route a query to appropriate experts.

        Args:
            query: The user's query
            context: Optional additional context

        Returns:
            RoutingDecision with selected experts and confidence scores
        """
        strategy = self.config.strategy

        if strategy == RoutingStrategy.KEYWORD:
            return await self._route_keyword(query)
        elif strategy == RoutingStrategy.EMBEDDING:
            return await self._route_embedding(query)
        elif strategy == RoutingStrategy.LLM:
            return await self._route_llm(query)
        else:  # HYBRID
            return await self._route_hybrid(query)

    async def _route_keyword(self, query: str) -> RoutingDecision:
        """Route based on keyword matching."""
        scores = {}

        for name, expert in self._experts.items():
            score = expert.matches_query(query)
            if score >= self.config.min_confidence:
                scores[name] = score

        # Sort by score and take top_k
        sorted_experts = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [name for name, _ in sorted_experts[:self.config.top_k]]

        return RoutingDecision(
            query=query,
            selected_experts=selected,
            confidence_scores=scores,
            reasoning="Selected based on keyword matching",
            strategy_used=RoutingStrategy.KEYWORD,
        )

    async def _route_embedding(self, query: str) -> RoutingDecision:
        """Route based on embedding similarity."""
        # For now, fall back to keyword if no embedding model
        if self._embedding_model is None:
            return await self._route_keyword(query)

        # TODO: Implement embedding-based routing
        # 1. Embed the query
        # 2. Compare with expert domain embeddings
        # 3. Select most similar

        return await self._route_keyword(query)

    async def _route_llm(self, query: str) -> RoutingDecision:
        """Route using LLM classification."""
        if self._classifier_model is None:
            return await self._route_keyword(query)

        # Build classification prompt
        expert_list = "\n".join([
            f"- {name}: {exp.config.domain} - {exp.config.description}"
            for name, exp in self._experts.items()
        ])

        prompt = f"""<|im_start|>system
You are a query router. Given a query, select the most relevant experts to handle it.
Return a JSON object with "experts" (list of expert names) and "reasoning" (why these were selected).

Available experts:
{expert_list}<|im_end|>
<|im_start|>user
Query: {query}

Which experts should handle this? Select up to {self.config.top_k} experts.<|im_end|>
<|im_start|>assistant
"""

        try:
            from mlx_lm import generate

            response = generate(
                self._classifier_model,
                self._classifier_tokenizer,
                prompt=prompt,
                max_tokens=200,
                temp=0.1,  # Low temperature for consistency
            )

            # Parse response
            import json
            try:
                result = json.loads(response)
                selected = result.get("experts", [])
                reasoning = result.get("reasoning", "LLM classification")
            except json.JSONDecodeError:
                # Fallback: extract expert names from response
                selected = [
                    name for name in self._experts.keys()
                    if name.lower() in response.lower()
                ][:self.config.top_k]
                reasoning = response

            # Calculate confidence scores
            scores = {name: 0.8 for name in selected}  # Default high confidence for LLM selection

            return RoutingDecision(
                query=query,
                selected_experts=selected,
                confidence_scores=scores,
                reasoning=reasoning,
                strategy_used=RoutingStrategy.LLM,
            )

        except Exception as e:
            print(f"LLM routing failed: {e}, falling back to keyword")
            return await self._route_keyword(query)

    async def _route_hybrid(self, query: str) -> RoutingDecision:
        """
        Hybrid routing combining multiple strategies.

        Strategy:
        1. Fast keyword matching to get candidates
        2. Refine with embedding similarity or LLM if available
        """
        # Start with keyword matching
        keyword_result = await self._route_keyword(query)

        # If we have many candidates, refine
        if len(keyword_result.selected_experts) > self.config.top_k:
            # Try LLM refinement
            if self._classifier_model:
                llm_result = await self._route_llm(query)
                # Combine scores
                combined_scores = {}
                for name in set(keyword_result.selected_experts + llm_result.selected_experts):
                    kw_score = keyword_result.confidence_scores.get(name, 0)
                    llm_score = llm_result.confidence_scores.get(name, 0)
                    combined_scores[name] = 0.4 * kw_score + 0.6 * llm_score

                sorted_experts = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
                selected = [name for name, _ in sorted_experts[:self.config.top_k]]

                return RoutingDecision(
                    query=query,
                    selected_experts=selected,
                    confidence_scores=combined_scores,
                    reasoning="Hybrid routing (keyword + LLM)",
                    strategy_used=RoutingStrategy.HYBRID,
                )

        return keyword_result

    def explain_routing(self, decision: RoutingDecision) -> str:
        """Generate human-readable explanation of routing decision."""
        lines = [
            f"Query: {decision.query[:100]}...",
            f"Strategy: {decision.strategy_used.value}",
            f"Selected {len(decision.selected_experts)} experts:",
        ]

        for name in decision.selected_experts:
            score = decision.confidence_scores.get(name, 0)
            expert = self._experts.get(name)
            domain = expert.domain if expert else "unknown"
            lines.append(f"  - {name} ({domain}): {score:.2f} confidence")

        lines.append(f"Reasoning: {decision.reasoning}")

        return "\n".join(lines)
