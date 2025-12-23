"""
Clase base para recolectores de datos.
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
class CollectedDocument:
    """
    Documento recolectado de una fuente.
    """

    # Contenido
    content: str
    title: Optional[str] = None

    # Metadata
    source_type: str = ""  # "web", "github", "docs", "forum"
    source_url: Optional[str] = None
    language: str = "es"

    # Clasificación
    doc_type: str = "text"  # "text", "code", "qa", "tutorial"
    domain_relevance: float = 1.0

    # Tracking
    collected_at: datetime = field(default_factory=datetime.now)
    content_hash: str = ""

    def __post_init__(self):
        """Calcula el hash del contenido."""
        if not self.content_hash and self.content:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]

    @property
    def token_estimate(self) -> int:
        """Estimación de tokens (aproximada: 4 chars = 1 token)."""
        return len(self.content) // 4

    def to_dict(self) -> dict:
        """Convierte a diccionario para serialización."""
        return {
            "content": self.content,
            "title": self.title,
            "source_type": self.source_type,
            "source_url": self.source_url,
            "language": self.language,
            "doc_type": self.doc_type,
            "domain_relevance": self.domain_relevance,
            "collected_at": self.collected_at.isoformat(),
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CollectedDocument":
        """Crea desde diccionario."""
        data = data.copy()
        if "collected_at" in data and isinstance(data["collected_at"], str):
            data["collected_at"] = datetime.fromisoformat(data["collected_at"])
        return cls(**data)

    async def save(self, path: Path) -> None:
        """Guarda el documento a un archivo JSON."""
        async with aiofiles.open(path, "w") as f:
            await f.write(json.dumps(self.to_dict(), ensure_ascii=False, indent=2))

    @classmethod
    async def load(cls, path: Path) -> "CollectedDocument":
        """Carga un documento desde archivo JSON."""
        async with aiofiles.open(path, "r") as f:
            data = json.loads(await f.read())
            return cls.from_dict(data)


class DataCollector(ABC):
    """
    Clase base abstracta para recolectores de datos.

    Cada recolector implementa la lógica para obtener datos de una fuente
    específica (web, GitHub, documentación, etc.).
    """

    def __init__(
        self,
        output_dir: Path,
        domain_keywords: list[str] = None,
        negative_keywords: list[str] = None,
        max_documents: int = 10000,
        language: str = "es",
    ):
        """
        Args:
            output_dir: Directorio donde guardar los documentos
            domain_keywords: Palabras clave del dominio (para filtrado)
            negative_keywords: Palabras que indican contenido no deseado
            max_documents: Máximo de documentos a recolectar
            language: Idioma principal
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.domain_keywords = domain_keywords or []
        self.negative_keywords = negative_keywords or []
        self.max_documents = max_documents
        self.language = language

        self._collected_hashes: set[str] = set()
        self._load_existing_hashes()

    def _load_existing_hashes(self) -> None:
        """Carga hashes de documentos existentes para evitar duplicados."""
        for file in self.output_dir.glob("*.json"):
            try:
                with open(file) as f:
                    data = json.load(f)
                    if "content_hash" in data:
                        self._collected_hashes.add(data["content_hash"])
            except Exception:
                pass

    def is_duplicate(self, content: str) -> bool:
        """Verifica si el contenido ya fue recolectado."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return content_hash in self._collected_hashes

    def calculate_relevance(self, content: str) -> float:
        """
        Calcula qué tan relevante es el contenido para el dominio.

        Returns:
            Score entre 0 y 1
        """
        content_lower = content.lower()

        # Contar keywords positivas
        positive_count = sum(1 for kw in self.domain_keywords if kw.lower() in content_lower)

        # Contar keywords negativas
        negative_count = sum(1 for kw in self.negative_keywords if kw.lower() in content_lower)

        # Si hay muchas negativas, descartar
        if negative_count >= 2:
            return 0.0

        # Calcular score basado en densidad de keywords
        if not self.domain_keywords:
            return 0.5  # Sin keywords, asumir relevancia media

        relevance = min(positive_count / (len(self.domain_keywords) * 0.3), 1.0)
        return max(relevance - (negative_count * 0.2), 0.0)

    async def save_document(self, doc: CollectedDocument) -> bool:
        """
        Guarda un documento si no es duplicado.

        Returns:
            True si se guardó, False si era duplicado
        """
        if doc.content_hash in self._collected_hashes:
            return False

        # Generar nombre de archivo
        filename = f"{doc.source_type}_{doc.content_hash}.json"
        filepath = self.output_dir / filename

        await doc.save(filepath)
        self._collected_hashes.add(doc.content_hash)
        return True

    @abstractmethod
    async def collect(self) -> AsyncIterator[CollectedDocument]:
        """
        Recolecta documentos de la fuente.

        Yields:
            CollectedDocument para cada documento encontrado
        """
        pass

    @property
    def collected_count(self) -> int:
        """Número de documentos recolectados."""
        return len(self._collected_hashes)

    @property
    def has_capacity(self) -> bool:
        """¿Podemos recolectar más documentos?"""
        return self.collected_count < self.max_documents
