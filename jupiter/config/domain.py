"""
Configuración de dominios de especialización.

Un dominio define qué tipo de experto se va a entrenar:
- Fuentes de datos (documentación, código, foros, etc.)
- Templates de generación sintética
- Criterios de evaluación específicos
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class GenerationTemplateType(str, Enum):
    """Tipos de templates de generación."""

    QA = "qa"  # Pregunta y respuesta
    TUTORIAL = "tutorial"  # Tutorial paso a paso
    CODE = "code"  # Código con explicación
    DEBUG = "debug"  # Problema y solución
    EXPLANATION = "explanation"  # Explicación conceptual
    CONVERSATION = "conversation"  # Diálogo multi-turno


@dataclass
class GitHubSource:
    """Configuración de fuente de datos de GitHub."""

    repos: list[str] = field(default_factory=list)
    file_types: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)
    max_file_size_kb: int = 500


@dataclass
class DataSources:
    """Fuentes de datos para el dominio."""

    # URLs de documentación oficial
    documentation: list[str] = field(default_factory=list)

    # Repositorios de GitHub
    github: GitHubSource = field(default_factory=GitHubSource)

    # Sitios web para scraping
    websites: list[str] = field(default_factory=list)

    # Foros y comunidades
    forums: list[str] = field(default_factory=list)

    # Canales de YouTube (para transcripciones)
    youtube_channels: list[str] = field(default_factory=list)

    # Datasets de Hugging Face (como base, no como única fuente)
    huggingface_datasets: list[str] = field(default_factory=list)


@dataclass
class GenerationTemplate:
    """Template para generación de datos sintéticos."""

    type: GenerationTemplateType
    prompt: str
    topics: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    system_prompt: Optional[str] = None
    examples: list[dict] = field(default_factory=list)


@dataclass
class GenerationSettings:
    """Configuración de generación sintética para el dominio."""

    templates: list[GenerationTemplate] = field(default_factory=list)

    # Idioma de generación
    language: str = "es"

    # Proporciones de cada tipo de template
    template_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class EvaluationSettings:
    """Configuración de evaluación para el dominio."""

    # Benchmarks a usar
    benchmarks: list[str] = field(default_factory=list)

    # Archivo de preguntas de evaluación custom
    custom_eval_file: Optional[str] = None

    # Umbrales de calidad
    min_accuracy: float = 0.7
    min_coherence: float = 0.8


@dataclass
class DataMixRatio:
    """Proporciones de mezcla de datos."""

    real_docs: float = 0.30
    real_code: float = 0.20
    synthetic_qa: float = 0.25
    synthetic_tutorials: float = 0.15
    synthetic_debug: float = 0.10

    def validate(self) -> bool:
        """Verifica que las proporciones sumen 1.0."""
        total = (
            self.real_docs
            + self.real_code
            + self.synthetic_qa
            + self.synthetic_tutorials
            + self.synthetic_debug
        )
        return abs(total - 1.0) < 0.01


@dataclass
class DomainConfig:
    """Configuración completa de un dominio."""

    # Identificación
    name: str
    description: str
    language: str = "es"

    # Fuentes de datos
    data_sources: DataSources = field(default_factory=DataSources)

    # Generación sintética
    generation: GenerationSettings = field(default_factory=GenerationSettings)

    # Evaluación
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)

    # Mezcla de datos
    mix_ratio: DataMixRatio = field(default_factory=DataMixRatio)

    # Palabras clave del dominio (para filtrado)
    keywords: list[str] = field(default_factory=list)

    # Términos que indican baja calidad
    negative_keywords: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "DomainConfig":
        """Crea DomainConfig desde un diccionario (YAML parseado)."""

        domain_data = data.get("domain", data)

        # Parsear data_sources
        ds_data = data.get("data_sources", {})
        github_data = ds_data.get("github", {})
        data_sources = DataSources(
            documentation=ds_data.get("documentation", []),
            github=GitHubSource(
                repos=github_data.get("repos", []),
                file_types=github_data.get("file_types", []),
                exclude_patterns=github_data.get("exclude_patterns", []),
            ),
            websites=ds_data.get("websites", []),
            forums=ds_data.get("forums", []),
            youtube_channels=ds_data.get("youtube", {}).get("channels", []),
            huggingface_datasets=ds_data.get("huggingface_datasets", []),
        )

        # Parsear templates de generación
        gen_data = data.get("generation", {})
        templates = []
        for t in gen_data.get("templates", []):
            templates.append(
                GenerationTemplate(
                    type=GenerationTemplateType(t.get("type", "qa")),
                    prompt=t.get("prompt", ""),
                    topics=t.get("topics", []),
                    tasks=t.get("tasks", []),
                    system_prompt=t.get("system_prompt"),
                    examples=t.get("examples", []),
                )
            )

        generation = GenerationSettings(
            templates=templates,
            language=domain_data.get("language", "es"),
            template_weights=gen_data.get("template_weights", {}),
        )

        # Parsear evaluación
        eval_data = data.get("evaluation", {})
        evaluation = EvaluationSettings(
            benchmarks=eval_data.get("benchmarks", []),
            custom_eval_file=eval_data.get("custom_eval_file"),
            min_accuracy=eval_data.get("min_accuracy", 0.7),
            min_coherence=eval_data.get("min_coherence", 0.8),
        )

        # Parsear mix_ratio
        mix_data = data.get("mix_ratio", {})
        mix_ratio = DataMixRatio(
            real_docs=mix_data.get("real_docs", 0.30),
            real_code=mix_data.get("real_code", 0.20),
            synthetic_qa=mix_data.get("synthetic_qa", 0.25),
            synthetic_tutorials=mix_data.get("synthetic_tutorials", 0.15),
            synthetic_debug=mix_data.get("synthetic_debug", 0.10),
        )

        return cls(
            name=domain_data.get("name", "unknown"),
            description=domain_data.get("description", ""),
            language=domain_data.get("language", "es"),
            data_sources=data_sources,
            generation=generation,
            evaluation=evaluation,
            mix_ratio=mix_ratio,
            keywords=domain_data.get("keywords", []),
            negative_keywords=domain_data.get("negative_keywords", []),
        )
