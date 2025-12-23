"""
Arquitectura del modelo Jupiter.

Implementación de un transformer decoder-only estilo Llama,
compatible con MLX y PyTorch.
"""

from typing import Optional, Tuple
import math

from jupiter.training.model.config import ModelConfig


# =============================================================================
# Intentar importar MLX primero, luego PyTorch
# =============================================================================

try:
    import mlx.core as mx
    import mlx.nn as nn

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None
    nn = None

try:
    import torch
    import torch.nn as torch_nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    torch_nn = None


# =============================================================================
# Implementación MLX
# =============================================================================

if HAS_MLX:

    class MLXRMSNorm(nn.Module):
        """RMSNorm para MLX."""

        def __init__(self, hidden_size: int, eps: float = 1e-5):
            super().__init__()
            self.weight = mx.ones((hidden_size,))
            self.eps = eps

        def __call__(self, x):
            variance = mx.mean(x * x, axis=-1, keepdims=True)
            x = x * mx.rsqrt(variance + self.eps)
            return self.weight * x

    class MLXRotaryEmbedding(nn.Module):
        """Rotary Position Embedding para MLX."""

        def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
            super().__init__()
            self.dim = dim
            self.max_position_embeddings = max_position_embeddings
            self.base = base

            inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
            self._inv_freq = inv_freq

        def __call__(self, x, seq_len: int):
            t = mx.arange(seq_len).astype(mx.float32)
            freqs = mx.outer(t, self._inv_freq)
            emb = mx.concatenate([freqs, freqs], axis=-1)
            return mx.cos(emb), mx.sin(emb)

    def mlx_rotate_half(x):
        """Rota la mitad de las dimensiones."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return mx.concatenate([-x2, x1], axis=-1)

    def mlx_apply_rotary_pos_emb(q, k, cos, sin):
        """Aplica rotary embeddings a q y k."""
        q_embed = (q * cos) + (mlx_rotate_half(q) * sin)
        k_embed = (k * cos) + (mlx_rotate_half(k) * sin)
        return q_embed, k_embed

    class MLXAttention(nn.Module):
        """Multi-head attention con RoPE para MLX."""

        def __init__(self, config: ModelConfig):
            super().__init__()
            self.hidden_size = config.hidden_size
            self.num_heads = config.num_attention_heads
            self.head_dim = self.hidden_size // self.num_heads
            self.num_kv_heads = config.num_key_value_heads
            self.num_kv_groups = self.num_heads // self.num_kv_heads

            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
            self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
            self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

            self.rotary_emb = MLXRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
            )

        def __call__(self, hidden_states, attention_mask=None):
            batch_size, seq_len, _ = hidden_states.shape

            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

            # Reshape para multi-head
            q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

            # Rotary embeddings
            cos, sin = self.rotary_emb(hidden_states, seq_len)
            q, k = mlx_apply_rotary_pos_emb(q, k, cos, sin)

            # GQA: repetir k, v si es necesario
            if self.num_kv_groups > 1:
                k = mx.repeat(k, self.num_kv_groups, axis=1)
                v = mx.repeat(v, self.num_kv_groups, axis=1)

            # Attention
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_weights = mx.matmul(q, k.transpose(0, 1, 3, 2)) * scale

            # Causal mask
            causal_mask = mx.triu(mx.full((seq_len, seq_len), float("-inf")), k=1)
            attn_weights = attn_weights + causal_mask

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = mx.softmax(attn_weights, axis=-1)
            attn_output = mx.matmul(attn_weights, v)

            # Reshape back
            attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
            return self.o_proj(attn_output)

    class MLXMLP(nn.Module):
        """MLP con gate (SwiGLU) para MLX."""

        def __init__(self, config: ModelConfig):
            super().__init__()
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
            self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
            self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

        def __call__(self, x):
            return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

    class MLXTransformerBlock(nn.Module):
        """Bloque de transformer para MLX."""

        def __init__(self, config: ModelConfig, layer_idx: int):
            super().__init__()
            self.layer_idx = layer_idx
            self.input_layernorm = MLXRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.self_attn = MLXAttention(config)
            self.post_attention_layernorm = MLXRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.mlp = MLXMLP(config)

        def __call__(self, hidden_states, attention_mask=None):
            # Self attention with residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.self_attn(hidden_states, attention_mask)
            hidden_states = residual + hidden_states

            # MLP with residual
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

            return hidden_states

    class MLXJupiterModel(nn.Module):
        """Modelo Jupiter completo para MLX."""

        def __init__(self, config: ModelConfig):
            super().__init__()
            self.config = config

            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
            self.layers = [
                MLXTransformerBlock(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
            self.norm = MLXRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

            # LM head (puede estar tied con embeddings)
            if config.tie_word_embeddings:
                self.lm_head = None
            else:
                self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        def __call__(self, input_ids, attention_mask=None):
            hidden_states = self.embed_tokens(input_ids)

            for layer in self.layers:
                hidden_states = layer(hidden_states, attention_mask)

            hidden_states = self.norm(hidden_states)

            # LM head
            if self.lm_head is not None:
                logits = self.lm_head(hidden_states)
            else:
                logits = hidden_states @ self.embed_tokens.weight.T

            return logits


# =============================================================================
# Wrapper unificado
# =============================================================================

class JupiterModel:
    """
    Wrapper unificado para el modelo Jupiter.

    Detecta automáticamente si usar MLX o PyTorch.
    """

    def __init__(self, config: ModelConfig, backend: str = "auto"):
        """
        Args:
            config: Configuración del modelo
            backend: "auto", "mlx", o "torch"
        """
        self.config = config

        if backend == "auto":
            backend = "mlx" if HAS_MLX else "torch"

        self.backend = backend

        if backend == "mlx":
            if not HAS_MLX:
                raise ImportError("MLX no está instalado")
            self._model = MLXJupiterModel(config)
        elif backend == "torch":
            if not HAS_TORCH:
                raise ImportError("PyTorch no está instalado")
            # TODO: Implementar versión PyTorch
            raise NotImplementedError("Backend PyTorch aún no implementado")
        else:
            raise ValueError(f"Backend no soportado: {backend}")

    def __call__(self, input_ids, attention_mask=None):
        """Forward pass."""
        return self._model(input_ids, attention_mask)

    @property
    def parameters(self):
        """Devuelve los parámetros del modelo."""
        if self.backend == "mlx":
            return self._model.parameters()
        else:
            return self._model.parameters()

    def save(self, path: str) -> None:
        """Save the model."""
        if self.backend == "mlx":
            import json
            from pathlib import Path

            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)

            # Save weights - flatten the nested parameter dict
            flat_weights = {name: param for name, param in nn.utils.tree_flatten(self._model.parameters())}
            mx.savez(str(path / "weights.npz"), **flat_weights)

            # Save config
            with open(path / "config.json", "w") as f:
                json.dump(self.config.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str, backend: str = "auto") -> "JupiterModel":
        """Carga un modelo guardado."""
        import json
        from pathlib import Path

        path = Path(path)

        # Cargar config
        with open(path / "config.json") as f:
            config = ModelConfig.from_dict(json.load(f))

        # Crear modelo
        model = cls(config, backend=backend)

        # Cargar pesos
        if backend == "mlx" or (backend == "auto" and HAS_MLX):
            weights = mx.load(str(path / "weights.npz"))
            model._model.load_weights(list(weights.items()))

        return model

    def __repr__(self) -> str:
        return f"JupiterModel(backend='{self.backend}', config={self.config})"
