"""
Configuración base de Jupiter.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml

from jupiter.config.domain import DomainConfig
from jupiter.config.cluster import ClusterConfig


@dataclass
class TrainingConfig:
    """Configuración de entrenamiento."""

    # Modelo
    model_size: str = "1B"  # "500M", "1B", "3B"
    vocab_size: int = 32000
    max_seq_length: int = 2048

    # Training
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    warmup_steps: int = 1000
    max_steps: int = 100000

    # Checkpointing
    save_every_steps: int = 1000
    eval_every_steps: int = 500

    # Mixed precision
    use_mixed_precision: bool = True

    # Data
    real_data_ratio: float = 0.4
    synthetic_data_ratio: float = 0.5
    fresh_data_ratio: float = 0.1


@dataclass
class GenerationConfig:
    """Configuración de generación de datos sintéticos."""

    # Modelo generador
    generator_model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"

    # Parámetros de generación
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 2048

    # Batch
    batch_size: int = 10
    queue_size: int = 10000

    # Calidad
    min_quality_score: float = 0.7
    dedup_threshold: float = 0.85


@dataclass
class SelfImprovementConfig:
    """Configuración del ciclo de auto-mejora."""

    # Cuándo evaluar
    eval_every_steps: int = 5000

    # Umbrales para reemplazo del generador
    improvement_threshold: float = 0.05  # 5% mejor
    min_steps_before_replace: int = 10000

    # Benchmarks
    benchmark_samples: int = 500

    # Seguridad
    keep_last_n_checkpoints: int = 3
    always_keep_base_generator: bool = True


@dataclass
class JupiterConfig:
    """Configuración completa de Jupiter."""

    # Paths
    project_dir: Path = field(default_factory=lambda: Path.cwd())
    data_dir: Path = field(default_factory=lambda: Path.cwd() / "data")
    checkpoints_dir: Path = field(default_factory=lambda: Path.cwd() / "checkpoints")
    logs_dir: Path = field(default_factory=lambda: Path.cwd() / "logs")

    # Sub-configuraciones
    domain: Optional[DomainConfig] = None
    cluster: Optional[ClusterConfig] = None
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    self_improvement: SelfImprovementConfig = field(default_factory=SelfImprovementConfig)

    @classmethod
    def from_yaml(cls, config_path: Path, domain_name: str) -> "JupiterConfig":
        """
        Carga configuración desde archivos YAML.

        Args:
            config_path: Path al directorio de configuración
            domain_name: Nombre del dominio a cargar

        Returns:
            JupiterConfig completo
        """
        config_path = Path(config_path)

        # Cargar configuración del cluster
        cluster_file = config_path / "cluster.yaml"
        cluster = None
        if cluster_file.exists():
            with open(cluster_file) as f:
                cluster_data = yaml.safe_load(f)
                cluster = ClusterConfig.from_dict(cluster_data)

        # Cargar configuración del dominio
        domain_file = config_path / "domains" / f"{domain_name}.yaml"
        if not domain_file.exists():
            raise FileNotFoundError(f"Dominio no encontrado: {domain_file}")

        with open(domain_file) as f:
            domain_data = yaml.safe_load(f)
            domain = DomainConfig.from_dict(domain_data)

        return cls(
            domain=domain,
            cluster=cluster,
        )

    def ensure_dirs(self) -> None:
        """Crea los directorios necesarios si no existen."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectorios de data
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        (self.data_dir / "synthetic").mkdir(exist_ok=True)
        (self.data_dir / "queue").mkdir(exist_ok=True)
