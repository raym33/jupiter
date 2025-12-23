"""
Orchestrador principal del pipeline de Jupiter.

Coordina todo el ciclo de vida:
1. Recolección de datos reales
2. Generación de datos sintéticos
3. Filtrado y preparación
4. Training distribuido
5. Evaluación
6. Auto-mejora
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

from jupiter.config import JupiterConfig, DomainConfig
from jupiter.data.collectors import WebCollector, GitHubCollector, DocsCollector
from jupiter.data.generators import LLMGenerator
from jupiter.data.filters import QualityFilter, Deduplicator
from jupiter.data.queue import DataQueue
from jupiter.training.distributed import DistributedTrainer
from jupiter.orchestrator.self_improve import SelfImprover


@dataclass
class PipelineState:
    """Estado del pipeline."""

    phase: str = "idle"  # idle, collecting, generating, training, evaluating
    started_at: Optional[datetime] = None
    current_epoch: int = 0
    total_documents: int = 0
    total_synthetic: int = 0
    best_eval_score: float = 0.0
    generator_version: int = 0


@dataclass
class PipelineStats:
    """Estadísticas del pipeline."""

    documents_collected: int = 0
    documents_filtered: int = 0
    samples_generated: int = 0
    samples_accepted: int = 0
    training_steps: int = 0
    evaluations_run: int = 0
    generator_upgrades: int = 0


class Orchestrator:
    """
    Orchestrador principal del pipeline de Jupiter.

    Maneja el ciclo completo de training de un modelo experto:
    1. Recolectar datos reales del dominio
    2. Generar datos sintéticos con LLM local
    3. Filtrar y preparar datos para training
    4. Entrenar modelo distribuido
    5. Evaluar modelo
    6. Si el modelo supera al generador, reemplazarlo
    7. Repetir
    """

    def __init__(self, config: JupiterConfig):
        """
        Args:
            config: Configuración de Jupiter
        """
        self.config = config
        self.state = PipelineState()
        self.stats = PipelineStats()

        # Componentes
        self.quality_filter = QualityFilter(
            min_length=100,
            max_length=50000,
            min_score=0.5,
            domain_keywords=config.domain.keywords if config.domain else [],
        )
        self.deduplicator = Deduplicator(similarity_threshold=0.85)
        self.data_queue: Optional[DataQueue] = None
        self.trainer: Optional[DistributedTrainer] = None
        self.self_improver: Optional[SelfImprover] = None

        # Paths
        self.data_dir = config.data_dir
        self.checkpoints_dir = config.checkpoints_dir

        # Asegurar directorios
        config.ensure_dirs()

    async def run(
        self,
        mode: str = "auto",
        max_epochs: int = 100,
        collect_data: bool = True,
        generate_data: bool = True,
    ) -> None:
        """
        Ejecuta el pipeline completo.

        Args:
            mode: "auto" (ciclo completo) o "interactive"
            max_epochs: Número máximo de épocas
            collect_data: Si recolectar datos reales
            generate_data: Si generar datos sintéticos
        """
        print("=" * 60)
        print("JUPITER - Pipeline de Training")
        print("=" * 60)
        print(f"Dominio: {self.config.domain.name}")
        print(f"Modo: {mode}")
        print()

        self.state.started_at = datetime.now()

        try:
            # Fase 1: Recolección de datos reales
            if collect_data:
                await self._collect_real_data()

            # Fase 2: Generación de datos sintéticos
            if generate_data:
                await self._generate_synthetic_data()

            # Fase 3: Preparar cola de datos
            await self._prepare_data_queue()

            # Fase 4: Ciclo de training
            for epoch in range(max_epochs):
                self.state.current_epoch = epoch
                print(f"\n{'=' * 60}")
                print(f"ÉPOCA {epoch + 1}/{max_epochs}")
                print("=" * 60)

                # Training
                await self._run_training_epoch()

                # Evaluación
                if (epoch + 1) % self.config.self_improvement.eval_every_steps == 0:
                    should_upgrade = await self._evaluate_and_maybe_upgrade()

                    if should_upgrade:
                        # Regenerar datos con el nuevo generador
                        await self._generate_synthetic_data()
                        await self._prepare_data_queue()

                # Guardar estado
                await self._save_pipeline_state()

        except KeyboardInterrupt:
            print("\nPipeline interrumpido por el usuario")
        except Exception as e:
            print(f"\nError en el pipeline: {e}")
            raise
        finally:
            self.state.phase = "idle"
            await self._save_pipeline_state()

        print("\nPipeline completado")
        self._print_final_stats()

    async def _collect_real_data(self) -> int:
        """
        Recolecta datos reales del dominio.

        Returns:
            Número de documentos recolectados
        """
        self.state.phase = "collecting"
        print("\n--- Fase 1: Recolección de Datos Reales ---")

        domain = self.config.domain
        sources = domain.data_sources

        total_collected = 0

        # Recolectar documentación
        if sources.documentation:
            print(f"Recolectando documentación de {len(sources.documentation)} fuentes...")
            collector = DocsCollector(
                urls=sources.documentation,
                output_dir=self.data_dir / "raw" / "docs",
                domain_keywords=domain.keywords,
                negative_keywords=domain.negative_keywords,
                max_documents=5000,
            )
            async for doc in collector.collect():
                total_collected += 1
                self.stats.documents_collected += 1

            print(f"  Documentación: {collector.collected_count} documentos")

        # Recolectar código de GitHub
        if sources.github.repos:
            print(f"Recolectando código de {len(sources.github.repos)} repos...")
            collector = GitHubCollector(
                repos=sources.github.repos,
                output_dir=self.data_dir / "raw" / "code",
                file_types=sources.github.file_types,
                exclude_patterns=sources.github.exclude_patterns,
                domain_keywords=domain.keywords,
                max_documents=10000,
            )
            async for doc in collector.collect():
                total_collected += 1
                self.stats.documents_collected += 1

            print(f"  Código: {collector.collected_count} archivos")

        # Recolectar websites
        if sources.websites:
            print(f"Recolectando de {len(sources.websites)} websites...")
            collector = WebCollector(
                urls=sources.websites,
                output_dir=self.data_dir / "raw" / "web",
                domain_keywords=domain.keywords,
                negative_keywords=domain.negative_keywords,
                max_documents=3000,
                max_depth=2,
            )
            async for doc in collector.collect():
                total_collected += 1
                self.stats.documents_collected += 1

            print(f"  Web: {collector.collected_count} páginas")

        self.state.total_documents = total_collected
        print(f"\nTotal recolectado: {total_collected} documentos")

        return total_collected

    async def _generate_synthetic_data(self) -> int:
        """
        Genera datos sintéticos usando el LLM local.

        Returns:
            Número de muestras generadas
        """
        self.state.phase = "generating"
        print("\n--- Fase 2: Generación de Datos Sintéticos ---")

        domain = self.config.domain

        # Crear generador
        generator = LLMGenerator(
            output_dir=self.data_dir / "synthetic",
            model_name=self.config.generation.generator_model,
            max_samples=self.config.generation.queue_size,
            temperature=self.config.generation.temperature,
            top_p=self.config.generation.top_p,
            max_tokens=self.config.generation.max_tokens,
        )

        # Cargar modelo
        await generator.load_model()

        total_generated = 0

        # Generar para cada template
        for template in domain.generation.templates:
            print(f"Generando {template.type.value}...")

            # Calcular cuántas muestras de este tipo
            weight = domain.generation.template_weights.get(template.type.value, 0.25)
            num_samples = int(self.config.generation.queue_size * weight)

            async for sample in generator.generate_from_template(template, num_samples):
                # Filtrar calidad
                is_valid, score = self.quality_filter.filter(sample.content)

                if is_valid:
                    # Verificar duplicado
                    result, _ = self.deduplicator.check_and_add(sample.content)

                    if not result.is_duplicate:
                        sample.quality_score = score.overall_score
                        total_generated += 1
                        self.stats.samples_accepted += 1

                self.stats.samples_generated += 1

            print(f"  {template.type.value}: {total_generated} aceptadas")

        # Descargar modelo
        await generator.unload_model()

        self.state.total_synthetic = total_generated
        print(f"\nTotal generado: {total_generated} muestras")

        return total_generated

    async def _prepare_data_queue(self) -> None:
        """Prepara la cola de datos para training."""
        print("\n--- Preparando Cola de Datos ---")

        self.data_queue = DataQueue(
            data_dir=self.data_dir / "queue",
            batch_size=self.config.training.batch_size,
            mix_ratios={
                "real": self.config.training.real_data_ratio,
                "synthetic": self.config.training.synthetic_data_ratio,
                "fresh": self.config.training.fresh_data_ratio,
            },
        )

        # Cargar datos
        await self.data_queue.load_from_directories(
            real_dir=self.data_dir / "raw",
            synthetic_dir=self.data_dir / "synthetic",
            fresh_dir=self.data_dir / "fresh",
        )

        print(f"Datos cargados: {self.data_queue.samples_per_source}")

    async def _run_training_epoch(self) -> None:
        """Ejecuta una época de training."""
        self.state.phase = "training"

        if self.trainer is None:
            self.trainer = DistributedTrainer(self.config)

        # Entrenar época
        metrics = self.trainer.train_epoch(self.data_queue)

        self.stats.training_steps = self.trainer.state.step

    async def _evaluate_and_maybe_upgrade(self) -> bool:
        """
        Evalúa el modelo y decide si reemplazar el generador.

        Returns:
            True si se hizo upgrade del generador
        """
        self.state.phase = "evaluating"
        print("\n--- Evaluación ---")

        if self.self_improver is None:
            self.self_improver = SelfImprover(
                config=self.config,
                trainer=self.trainer,
            )

        # Evaluar
        eval_score = await self.self_improver.evaluate()
        self.stats.evaluations_run += 1

        print(f"Score de evaluación: {eval_score:.4f}")
        print(f"Mejor score anterior: {self.state.best_eval_score:.4f}")

        # Verificar si debemos hacer upgrade
        improvement = eval_score - self.state.best_eval_score
        threshold = self.config.self_improvement.improvement_threshold

        if improvement > threshold:
            print(f"¡Mejora de {improvement:.4f}! Actualizando generador...")

            await self.self_improver.upgrade_generator()
            self.state.best_eval_score = eval_score
            self.state.generator_version += 1
            self.stats.generator_upgrades += 1

            return True

        return False

    async def _save_pipeline_state(self) -> None:
        """Guarda el estado del pipeline."""
        state_file = self.config.project_dir / "pipeline_state.json"

        state = {
            "phase": self.state.phase,
            "started_at": self.state.started_at.isoformat() if self.state.started_at else None,
            "current_epoch": self.state.current_epoch,
            "total_documents": self.state.total_documents,
            "total_synthetic": self.state.total_synthetic,
            "best_eval_score": self.state.best_eval_score,
            "generator_version": self.state.generator_version,
            "stats": {
                "documents_collected": self.stats.documents_collected,
                "documents_filtered": self.stats.documents_filtered,
                "samples_generated": self.stats.samples_generated,
                "samples_accepted": self.stats.samples_accepted,
                "training_steps": self.stats.training_steps,
                "evaluations_run": self.stats.evaluations_run,
                "generator_upgrades": self.stats.generator_upgrades,
            },
        }

        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _print_final_stats(self) -> None:
        """Imprime estadísticas finales."""
        print("\n" + "=" * 60)
        print("ESTADÍSTICAS FINALES")
        print("=" * 60)
        print(f"Documentos recolectados: {self.stats.documents_collected}")
        print(f"Muestras generadas: {self.stats.samples_generated}")
        print(f"Muestras aceptadas: {self.stats.samples_accepted}")
        print(f"Steps de training: {self.stats.training_steps}")
        print(f"Evaluaciones: {self.stats.evaluations_run}")
        print(f"Upgrades del generador: {self.stats.generator_upgrades}")
        print(f"Mejor score: {self.state.best_eval_score:.4f}")
        print("=" * 60)
