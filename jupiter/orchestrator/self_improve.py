"""
Sistema de auto-mejora para Jupiter.

Implementa el ciclo donde el modelo entrenado puede
reemplazar al modelo generador cuando lo supera.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import shutil

from jupiter.config import JupiterConfig


@dataclass
class EvaluationResult:
    """Resultado de una evaluación."""

    accuracy: float = 0.0
    coherence: float = 0.0
    domain_knowledge: float = 0.0
    overall_score: float = 0.0
    samples_evaluated: int = 0
    details: dict = None


class SelfImprover:
    """
    Sistema de auto-mejora.

    Evalúa si el modelo entrenado supera al generador actual
    y gestiona el reemplazo cuando corresponde.
    """

    def __init__(
        self,
        config: JupiterConfig,
        trainer: "DistributedTrainer" = None,  # noqa: F821
    ):
        """
        Args:
            config: Configuración de Jupiter
            trainer: Trainer con el modelo actual
        """
        self.config = config
        self.trainer = trainer

        self.eval_samples: list[dict] = []
        self.current_generator_path: Optional[Path] = None
        self.generator_history: list[Path] = []

    async def load_eval_samples(self, path: Optional[Path] = None) -> int:
        """
        Carga muestras de evaluación.

        Args:
            path: Path a archivo de evaluación (JSON lines)

        Returns:
            Número de muestras cargadas
        """
        if path is None:
            # Usar evaluación del dominio si existe
            domain_eval = self.config.domain.evaluation.custom_eval_file
            if domain_eval:
                path = Path(domain_eval)
            else:
                # Generar muestras de evaluación automáticamente
                return await self._generate_eval_samples()

        if not path.exists():
            print(f"Archivo de evaluación no encontrado: {path}")
            return 0

        self.eval_samples = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    self.eval_samples.append(json.loads(line))

        return len(self.eval_samples)

    async def _generate_eval_samples(self) -> int:
        """
        Genera muestras de evaluación automáticamente.

        Returns:
            Número de muestras generadas
        """
        # Usar una pequeña porción de los datos sintéticos como eval
        synthetic_dir = self.config.data_dir / "synthetic"

        if not synthetic_dir.exists():
            return 0

        # Tomar las primeras N muestras
        num_samples = self.config.self_improvement.benchmark_samples
        samples = []

        for file in sorted(synthetic_dir.glob("*.json"))[:num_samples]:
            with open(file) as f:
                data = json.load(f)
                samples.append({
                    "prompt": data.get("prompt_used", ""),
                    "expected": data.get("content", ""),
                    "type": data.get("template_type", "qa"),
                })

        self.eval_samples = samples
        return len(samples)

    async def evaluate(self) -> float:
        """
        Evalúa el modelo actual.

        Returns:
            Score de evaluación (0-1)
        """
        if not self.eval_samples:
            await self.load_eval_samples()

        if not self.eval_samples:
            print("No hay muestras de evaluación disponibles")
            return 0.0

        print(f"Evaluando con {len(self.eval_samples)} muestras...")

        # Evaluar modelo
        results = await self._run_evaluation()

        print(f"Resultados:")
        print(f"  Accuracy: {results.accuracy:.4f}")
        print(f"  Coherence: {results.coherence:.4f}")
        print(f"  Domain knowledge: {results.domain_knowledge:.4f}")
        print(f"  Overall: {results.overall_score:.4f}")

        return results.overall_score

    async def _run_evaluation(self) -> EvaluationResult:
        """
        Ejecuta la evaluación del modelo.

        Returns:
            Resultados de evaluación
        """
        if self.trainer is None:
            return EvaluationResult()

        # Evaluar cada muestra
        scores = {
            "accuracy": [],
            "coherence": [],
            "domain": [],
        }

        for sample in self.eval_samples:
            score = await self._evaluate_sample(sample)
            scores["accuracy"].append(score.get("accuracy", 0))
            scores["coherence"].append(score.get("coherence", 0))
            scores["domain"].append(score.get("domain", 0))

        # Promediar
        accuracy = sum(scores["accuracy"]) / len(scores["accuracy"]) if scores["accuracy"] else 0
        coherence = sum(scores["coherence"]) / len(scores["coherence"]) if scores["coherence"] else 0
        domain = sum(scores["domain"]) / len(scores["domain"]) if scores["domain"] else 0

        overall = (accuracy * 0.4) + (coherence * 0.3) + (domain * 0.3)

        return EvaluationResult(
            accuracy=accuracy,
            coherence=coherence,
            domain_knowledge=domain,
            overall_score=overall,
            samples_evaluated=len(self.eval_samples),
        )

    async def _evaluate_sample(self, sample: dict) -> dict:
        """
        Evalúa una muestra individual.

        Args:
            sample: Muestra con prompt y expected

        Returns:
            Scores para esta muestra
        """
        # Obtener respuesta del modelo
        prompt = sample.get("prompt", "")
        expected = sample.get("expected", "")

        if not prompt:
            return {"accuracy": 0, "coherence": 0, "domain": 0}

        # Generar respuesta
        # TODO: Implementar generación con el modelo entrenado
        # Por ahora, usar heurísticas simples

        # Placeholder: evaluar basado en similitud con expected
        # En implementación real, usar el modelo para generar y comparar

        return {
            "accuracy": 0.7,  # Placeholder
            "coherence": 0.8,
            "domain": 0.75,
        }

    async def upgrade_generator(self) -> bool:
        """
        Reemplaza el generador con el modelo actual.

        Returns:
            True si el upgrade fue exitoso
        """
        if self.trainer is None:
            print("No hay trainer disponible")
            return False

        print("Actualizando generador...")

        # Guardar generador actual en historial
        if self.current_generator_path:
            self.generator_history.append(self.current_generator_path)

        # Convertir modelo entrenado a formato de generador
        new_generator_path = await self._export_as_generator()

        if new_generator_path:
            self.current_generator_path = new_generator_path
            print(f"Nuevo generador: {new_generator_path}")
            return True

        return False

    async def _export_as_generator(self) -> Optional[Path]:
        """
        Exporta el modelo actual como generador.

        Returns:
            Path al nuevo generador
        """
        # Crear directorio para el nuevo generador
        version = len(self.generator_history) + 1
        generator_dir = self.config.checkpoints_dir / f"generator_v{version}"
        generator_dir.mkdir(parents=True, exist_ok=True)

        # Copiar el mejor checkpoint
        best_checkpoint = self.config.checkpoints_dir / "best"

        if best_checkpoint.exists():
            # Copiar modelo
            model_src = best_checkpoint / "model"
            if model_src.exists():
                shutil.copytree(model_src, generator_dir / "model", dirs_exist_ok=True)

            # Copiar tokenizer
            tokenizer_src = best_checkpoint / "tokenizer"
            if tokenizer_src.exists():
                shutil.copytree(tokenizer_src, generator_dir / "tokenizer", dirs_exist_ok=True)

            # Guardar metadata
            metadata = {
                "version": version,
                "source_checkpoint": str(best_checkpoint),
                "training_steps": self.trainer.state.step if self.trainer else 0,
            }
            with open(generator_dir / "generator_meta.json", "w") as f:
                json.dump(metadata, f, indent=2)

            return generator_dir

        return None

    async def rollback_generator(self) -> bool:
        """
        Vuelve al generador anterior.

        Returns:
            True si el rollback fue exitoso
        """
        if not self.generator_history:
            print("No hay historial de generadores")
            return False

        # Restaurar último generador
        self.current_generator_path = self.generator_history.pop()
        print(f"Rollback a: {self.current_generator_path}")

        return True

    def get_generator_info(self) -> dict:
        """
        Obtiene información del generador actual.

        Returns:
            Info del generador
        """
        if not self.current_generator_path:
            return {"status": "default", "version": 0}

        meta_file = self.current_generator_path / "generator_meta.json"
        if meta_file.exists():
            with open(meta_file) as f:
                return json.load(f)

        return {"status": "unknown", "path": str(self.current_generator_path)}
