"""
Filtro de calidad para datos de entrenamiento.

Evalúa la calidad de documentos y muestras sintéticas usando
heurísticas y (opcionalmente) un modelo de evaluación.
"""

from dataclasses import dataclass
from typing import Optional
import re


@dataclass
class QualityScore:
    """
    Puntuación de calidad de un documento.
    """

    # Scores individuales (0-1)
    length_score: float = 0.0
    structure_score: float = 0.0
    language_score: float = 0.0
    domain_score: float = 0.0
    coherence_score: float = 0.0

    # Score final
    overall_score: float = 0.0

    # Flags
    is_valid: bool = True
    rejection_reason: Optional[str] = None

    def __post_init__(self):
        """Calcula el score general si no está definido."""
        if self.overall_score == 0.0:
            self.overall_score = self._calculate_overall()

    def _calculate_overall(self) -> float:
        """Calcula el score general ponderado."""
        weights = {
            "length": 0.15,
            "structure": 0.20,
            "language": 0.20,
            "domain": 0.25,
            "coherence": 0.20,
        }
        return (
            self.length_score * weights["length"]
            + self.structure_score * weights["structure"]
            + self.language_score * weights["language"]
            + self.domain_score * weights["domain"]
            + self.coherence_score * weights["coherence"]
        )


class QualityFilter:
    """
    Filtro de calidad para datos de entrenamiento.

    Evalúa documentos y muestras sintéticas usando heurísticas:
    - Longitud adecuada
    - Estructura coherente
    - Calidad de lenguaje
    - Relevancia al dominio
    - Coherencia general
    """

    def __init__(
        self,
        min_length: int = 100,
        max_length: int = 50000,
        min_score: float = 0.5,
        domain_keywords: list[str] = None,
        language: str = "es",
    ):
        """
        Args:
            min_length: Longitud mínima en caracteres
            max_length: Longitud máxima en caracteres
            min_score: Score mínimo para aceptar
            domain_keywords: Palabras clave del dominio
            language: Idioma esperado
        """
        self.min_length = min_length
        self.max_length = max_length
        self.min_score = min_score
        self.domain_keywords = domain_keywords or []
        self.language = language

        # Patrones de baja calidad
        self._low_quality_patterns = [
            r"lorem ipsum",
            r"click here",
            r"subscribe now",
            r"buy now",
            r"limited offer",
            r"asdf+",
            r"test+\s*test+",
            r"\b[A-Z]{20,}\b",  # Muchas mayúsculas seguidas
        ]

        # Caracteres de código/formato excesivos
        self._format_noise_patterns = [
            r"[<>]{5,}",  # HTML roto
            r"\[object Object\]",
            r"undefined",
            r"null",
            r"NaN",
        ]

    def _score_length(self, content: str) -> float:
        """Evalúa la longitud del contenido."""
        length = len(content)

        if length < self.min_length:
            return 0.0
        if length > self.max_length:
            return 0.5  # Muy largo, pero no descartar

        # Score óptimo entre 500 y 5000 caracteres
        if 500 <= length <= 5000:
            return 1.0
        elif length < 500:
            return length / 500
        else:
            return max(0.5, 1 - (length - 5000) / 45000)

    def _score_structure(self, content: str) -> float:
        """Evalúa la estructura del contenido."""
        score = 0.5  # Base

        # Presencia de párrafos
        paragraphs = content.split("\n\n")
        if len(paragraphs) > 1:
            score += 0.1

        # Presencia de encabezados
        if re.search(r"^#+\s", content, re.MULTILINE):
            score += 0.1

        # Presencia de listas
        if re.search(r"^[-*]\s", content, re.MULTILINE):
            score += 0.1

        # Presencia de código
        if "```" in content:
            score += 0.1

        # Buena proporción de oraciones
        sentences = re.split(r"[.!?]+", content)
        avg_sentence_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        if 10 <= avg_sentence_len <= 30:
            score += 0.1

        return min(score, 1.0)

    def _score_language(self, content: str) -> float:
        """Evalúa la calidad del lenguaje."""
        score = 1.0

        # Detectar patrones de baja calidad
        content_lower = content.lower()
        for pattern in self._low_quality_patterns:
            if re.search(pattern, content_lower):
                score -= 0.2

        # Detectar ruido de formato
        for pattern in self._format_noise_patterns:
            if re.search(pattern, content):
                score -= 0.15

        # Proporción de caracteres alfanuméricos
        alnum_chars = sum(c.isalnum() for c in content)
        total_chars = len(content)
        if total_chars > 0:
            alnum_ratio = alnum_chars / total_chars
            if alnum_ratio < 0.5:  # Demasiados símbolos
                score -= 0.2

        # Proporción de mayúsculas
        upper_chars = sum(c.isupper() for c in content)
        if alnum_chars > 0:
            upper_ratio = upper_chars / alnum_chars
            if upper_ratio > 0.5:  # Demasiadas mayúsculas
                score -= 0.2

        return max(score, 0.0)

    def _score_domain(self, content: str) -> float:
        """Evalúa la relevancia al dominio."""
        if not self.domain_keywords:
            return 0.7  # Sin keywords, asumir relevancia media

        content_lower = content.lower()

        # Contar keywords presentes
        present = sum(1 for kw in self.domain_keywords if kw.lower() in content_lower)
        ratio = present / len(self.domain_keywords)

        # Queremos al menos 10% de keywords presentes
        if ratio < 0.1:
            return 0.3
        elif ratio < 0.2:
            return 0.5
        elif ratio < 0.3:
            return 0.7
        else:
            return min(0.5 + ratio, 1.0)

    def _score_coherence(self, content: str) -> float:
        """Evalúa la coherencia general."""
        score = 0.7  # Base

        # Verificar que no haya repetición excesiva
        words = content.lower().split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:  # Muy repetitivo
                score -= 0.3
            elif unique_ratio > 0.7:  # Buena variedad
                score += 0.1

        # Verificar que las oraciones tengan sentido básico
        # (empiezan con mayúscula, terminan con puntuación)
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        well_formed = 0
        for line in lines:
            if line and (
                line[0].isupper() or line[0] in "#-*`"
            ):  # Empieza bien
                if line[-1] in ".!?:\"'`)" or line.startswith("#"):  # Termina bien
                    well_formed += 1

        if lines:
            formation_ratio = well_formed / len(lines)
            score = score * 0.5 + formation_ratio * 0.5

        return min(max(score, 0.0), 1.0)

    def evaluate(self, content: str) -> QualityScore:
        """
        Evalúa la calidad de un contenido.

        Args:
            content: Texto a evaluar

        Returns:
            QualityScore con la evaluación
        """
        # Verificar longitud mínima
        if len(content) < self.min_length:
            return QualityScore(
                is_valid=False,
                rejection_reason=f"Contenido muy corto ({len(content)} < {self.min_length})",
            )

        # Calcular scores individuales
        length_score = self._score_length(content)
        structure_score = self._score_structure(content)
        language_score = self._score_language(content)
        domain_score = self._score_domain(content)
        coherence_score = self._score_coherence(content)

        # Crear score
        score = QualityScore(
            length_score=length_score,
            structure_score=structure_score,
            language_score=language_score,
            domain_score=domain_score,
            coherence_score=coherence_score,
        )

        # Verificar score mínimo
        if score.overall_score < self.min_score:
            score.is_valid = False
            score.rejection_reason = f"Score muy bajo ({score.overall_score:.2f} < {self.min_score})"

        return score

    def filter(self, content: str) -> tuple[bool, QualityScore]:
        """
        Filtra un contenido.

        Args:
            content: Texto a filtrar

        Returns:
            (aceptado, score)
        """
        score = self.evaluate(content)
        return score.is_valid, score
