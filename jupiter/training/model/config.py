"""
Configuración del modelo Jupiter.

Define las arquitecturas predefinidas y permite configuración custom.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """
    Configuración de la arquitectura del modelo.

    Arquitecturas predefinidas:
    - 125M: Para testing y desarrollo rápido
    - 350M: Modelo pequeño funcional
    - 500M: Buen balance para hardware limitado
    - 1B: Objetivo principal para clusters pequeños
    - 3B: Para clusters con más memoria
    """

    # Identificación
    name: str = "jupiter-1b"
    version: str = "0.1.0"

    # Arquitectura
    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 5632  # ~2.75x hidden_size (como Llama)
    num_hidden_layers: int = 22
    num_attention_heads: int = 16
    num_key_value_heads: int = 16  # Puede ser menor para GQA
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0

    # Normalization
    rms_norm_eps: float = 1e-5
    use_rms_norm: bool = True  # RMSNorm vs LayerNorm

    # Attention
    attention_dropout: float = 0.0
    attention_bias: bool = False
    use_flash_attention: bool = True

    # MLP
    hidden_act: str = "silu"  # SiLU/Swish activation
    mlp_bias: bool = False

    # Embeddings
    tie_word_embeddings: bool = True

    # Training
    initializer_range: float = 0.02
    use_cache: bool = True

    # Precisión
    dtype: str = "float16"  # "float16", "bfloat16", "float32"

    @classmethod
    def from_preset(cls, preset: str) -> "ModelConfig":
        """
        Crea configuración desde un preset predefinido.

        Presets disponibles:
        - "125m": 125M parámetros (testing)
        - "350m": 350M parámetros
        - "500m": 500M parámetros
        - "1b": 1B parámetros (default)
        - "3b": 3B parámetros
        """
        presets = {
            "125m": cls(
                name="jupiter-125m",
                hidden_size=768,
                intermediate_size=2048,
                num_hidden_layers=12,
                num_attention_heads=12,
                num_key_value_heads=12,
                max_position_embeddings=2048,
            ),
            "350m": cls(
                name="jupiter-350m",
                hidden_size=1024,
                intermediate_size=2816,
                num_hidden_layers=16,
                num_attention_heads=16,
                num_key_value_heads=16,
                max_position_embeddings=2048,
            ),
            "500m": cls(
                name="jupiter-500m",
                hidden_size=1280,
                intermediate_size=3584,
                num_hidden_layers=18,
                num_attention_heads=16,
                num_key_value_heads=8,  # GQA
                max_position_embeddings=2048,
            ),
            "1b": cls(
                name="jupiter-1b",
                hidden_size=2048,
                intermediate_size=5632,
                num_hidden_layers=22,
                num_attention_heads=16,
                num_key_value_heads=16,
                max_position_embeddings=2048,
            ),
            "3b": cls(
                name="jupiter-3b",
                hidden_size=2560,
                intermediate_size=7168,
                num_hidden_layers=32,
                num_attention_heads=20,
                num_key_value_heads=4,  # GQA agresivo
                max_position_embeddings=4096,
            ),
        }

        if preset not in presets:
            raise ValueError(
                f"Preset '{preset}' no encontrado. "
                f"Opciones: {list(presets.keys())}"
            )

        return presets[preset]

    @property
    def num_parameters(self) -> int:
        """Estima el número de parámetros del modelo."""
        # Embeddings
        embed_params = self.vocab_size * self.hidden_size

        # Por capa de transformer
        # Attention: Q, K, V, O proyecciones
        attn_params = (
            self.hidden_size * self.hidden_size  # Q
            + self.hidden_size * (self.hidden_size // self.num_attention_heads * self.num_key_value_heads)  # K
            + self.hidden_size * (self.hidden_size // self.num_attention_heads * self.num_key_value_heads)  # V
            + self.hidden_size * self.hidden_size  # O
        )

        # MLP: gate, up, down
        mlp_params = (
            self.hidden_size * self.intermediate_size  # gate
            + self.hidden_size * self.intermediate_size  # up
            + self.intermediate_size * self.hidden_size  # down
        )

        # Norms (2 por capa)
        norm_params = self.hidden_size * 2

        layer_params = attn_params + mlp_params + norm_params
        total_layer_params = layer_params * self.num_hidden_layers

        # LM head (tied con embeddings si tie_word_embeddings=True)
        lm_head_params = 0 if self.tie_word_embeddings else self.hidden_size * self.vocab_size

        # Final norm
        final_norm_params = self.hidden_size

        total = embed_params + total_layer_params + lm_head_params + final_norm_params

        return total

    @property
    def num_parameters_str(self) -> str:
        """Número de parámetros en formato legible."""
        params = self.num_parameters
        if params >= 1e9:
            return f"{params / 1e9:.1f}B"
        elif params >= 1e6:
            return f"{params / 1e6:.0f}M"
        else:
            return f"{params / 1e3:.0f}K"

    @property
    def estimated_memory_gb(self) -> float:
        """Estima la memoria necesaria en GB."""
        params = self.num_parameters

        # Bytes por parámetro según dtype
        bytes_per_param = {
            "float32": 4,
            "float16": 2,
            "bfloat16": 2,
        }

        param_memory = params * bytes_per_param.get(self.dtype, 2)

        # Para training, necesitamos ~4x más (gradientes, optimizer states, activations)
        training_memory = param_memory * 4

        return training_memory / (1024**3)

    def to_dict(self) -> dict:
        """Convierte a diccionario."""
        return {
            "name": self.name,
            "version": self.version,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "rms_norm_eps": self.rms_norm_eps,
            "use_rms_norm": self.use_rms_norm,
            "attention_dropout": self.attention_dropout,
            "attention_bias": self.attention_bias,
            "use_flash_attention": self.use_flash_attention,
            "hidden_act": self.hidden_act,
            "mlp_bias": self.mlp_bias,
            "tie_word_embeddings": self.tie_word_embeddings,
            "initializer_range": self.initializer_range,
            "use_cache": self.use_cache,
            "dtype": self.dtype,
            "num_parameters": self.num_parameters,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        """Crea desde diccionario."""
        # Filtrar keys que no son parámetros del constructor
        valid_keys = {
            "name", "version", "vocab_size", "hidden_size", "intermediate_size",
            "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
            "max_position_embeddings", "rope_theta", "rms_norm_eps", "use_rms_norm",
            "attention_dropout", "attention_bias", "use_flash_attention",
            "hidden_act", "mlp_bias", "tie_word_embeddings", "initializer_range",
            "use_cache", "dtype",
        }
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def __repr__(self) -> str:
        return (
            f"ModelConfig(name='{self.name}', "
            f"params={self.num_parameters_str}, "
            f"layers={self.num_hidden_layers}, "
            f"hidden={self.hidden_size}, "
            f"heads={self.num_attention_heads})"
        )
