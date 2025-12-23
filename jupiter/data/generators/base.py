"""
Clase base para generadores de datos sintéticos.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Optional
import hashlib
import json
import aiofiles


@dataclass
class GeneratedSample:
    """
    Muestra generada sintéticamente.
    """

    # Contenido
    content: str
    prompt_used: str

    # Metadata
    template_type: str  # "qa", "tutorial", "debug", etc.
    topic: Optional[str] = None
    generator_model: str = ""

    # Calidad
    quality_score: Optional[float] = None
    is_valid: bool = True

    # Tracking
    generated_at: datetime = field(default_factory=datetime.now)
    content_hash: str = ""

    def __post_init__(self):
        """Calcula el hash del contenido."""
        if not self.content_hash and self.content:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]

    @property
    def token_estimate(self) -> int:
        """Estimación de tokens."""
        return len(self.content) // 4

    def to_dict(self) -> dict:
        """Convierte a diccionario."""
        return {
            "content": self.content,
            "prompt_used": self.prompt_used,
            "template_type": self.template_type,
            "topic": self.topic,
            "generator_model": self.generator_model,
            "quality_score": self.quality_score,
            "is_valid": self.is_valid,
            "generated_at": self.generated_at.isoformat(),
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GeneratedSample":
        """Crea desde diccionario."""
        data = data.copy()
        if "generated_at" in data and isinstance(data["generated_at"], str):
            data["generated_at"] = datetime.fromisoformat(data["generated_at"])
        return cls(**data)

    async def save(self, path: Path) -> None:
        """Guarda la muestra a un archivo JSON."""
        async with aiofiles.open(path, "w") as f:
            await f.write(json.dumps(self.to_dict(), ensure_ascii=False, indent=2))

    @classmethod
    async def load(cls, path: Path) -> "GeneratedSample":
        """Carga una muestra desde archivo JSON."""
        async with aiofiles.open(path, "r") as f:
            data = json.loads(await f.read())
            return cls.from_dict(data)


class SyntheticGenerator(ABC):
    """
    Clase base abstracta para generadores de datos sintéticos.

    Los generadores usan modelos de lenguaje para crear datos de entrenamiento
    basados en templates específicos del dominio.
    """

    def __init__(
        self,
        output_dir: Path,
        model_name: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
        max_samples: int = 100000,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 2048,
    ):
        """
        Args:
            output_dir: Directorio donde guardar las muestras
            model_name: Nombre/path del modelo a usar
            max_samples: Máximo de muestras a generar
            temperature: Temperatura de sampling
            top_p: Top-p para nucleus sampling
            max_tokens: Máximo de tokens a generar
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self.max_samples = max_samples
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        self._generated_hashes: set[str] = set()
        self._load_existing_hashes()

    def _load_existing_hashes(self) -> None:
        """Carga hashes de muestras existentes."""
        for file in self.output_dir.glob("*.json"):
            try:
                with open(file) as f:
                    data = json.load(f)
                    if "content_hash" in data:
                        self._generated_hashes.add(data["content_hash"])
            except Exception:
                pass

    def is_duplicate(self, content: str) -> bool:
        """Verifica si el contenido ya fue generado."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return content_hash in self._generated_hashes

    async def save_sample(self, sample: GeneratedSample) -> bool:
        """
        Guarda una muestra si no es duplicada.

        Returns:
            True si se guardó, False si era duplicado
        """
        if sample.content_hash in self._generated_hashes:
            return False

        filename = f"{sample.template_type}_{sample.content_hash}.json"
        filepath = self.output_dir / filename

        await sample.save(filepath)
        self._generated_hashes.add(sample.content_hash)
        return True

    @abstractmethod
    async def generate(
        self,
        template_type: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> Optional[GeneratedSample]:
        """
        Genera una muestra sintética.

        Args:
            template_type: Tipo de template ("qa", "tutorial", etc.)
            prompt: Prompt para el modelo
            system_prompt: System prompt opcional
            topic: Tópico específico

        Returns:
            GeneratedSample o None si falla
        """
        pass

    @abstractmethod
    async def generate_batch(
        self,
        template_type: str,
        prompts: list[str],
        system_prompt: Optional[str] = None,
        topics: Optional[list[str]] = None,
    ) -> AsyncIterator[GeneratedSample]:
        """
        Genera múltiples muestras.

        Args:
            template_type: Tipo de template
            prompts: Lista de prompts
            system_prompt: System prompt común
            topics: Lista de tópicos correspondientes

        Yields:
            GeneratedSample para cada generación exitosa
        """
        pass

    @property
    def generated_count(self) -> int:
        """Número de muestras generadas."""
        return len(self._generated_hashes)

    @property
    def has_capacity(self) -> bool:
        """¿Podemos generar más muestras?"""
        return self.generated_count < self.max_samples

    @abstractmethod
    async def load_model(self) -> None:
        """Carga el modelo en memoria."""
        pass

    @abstractmethod
    async def unload_model(self) -> None:
        """Descarga el modelo de memoria."""
        pass
