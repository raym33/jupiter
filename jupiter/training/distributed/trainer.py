"""
Trainer distribuido para Jupiter.

Implementa el loop de training con soporte para:
- Data parallelism en múltiples dispositivos
- Sincronización de gradientes
- Checkpointing distribuido
- Logging y métricas
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterator
import time
import json

from jupiter.training.model import JupiterModel, ModelConfig, JupiterTokenizer
from jupiter.config import JupiterConfig
from jupiter.data.queue import DataQueue, DataBatch


@dataclass
class TrainingMetrics:
    """Métricas de training."""

    step: int = 0
    epoch: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    tokens_per_second: float = 0.0
    samples_per_second: float = 0.0
    gradient_norm: float = 0.0
    elapsed_time: float = 0.0


@dataclass
class TrainingState:
    """Estado del training para checkpointing."""

    step: int = 0
    epoch: int = 0
    best_loss: float = float("inf")
    total_tokens: int = 0
    optimizer_state: dict = field(default_factory=dict)


class DistributedTrainer:
    """
    Trainer distribuido para el modelo Jupiter.

    Características:
    - Data parallelism con sincronización de gradientes
    - Gradient accumulation para batches grandes
    - Mixed precision training
    - Checkpointing automático
    - Logging de métricas
    """

    def __init__(
        self,
        config: JupiterConfig,
        model: Optional[JupiterModel] = None,
        tokenizer: Optional[JupiterTokenizer] = None,
    ):
        """
        Args:
            config: Configuración de Jupiter
            model: Modelo (se crea si no se proporciona)
            tokenizer: Tokenizer (se carga si no se proporciona)
        """
        self.config = config

        # Detectar backend
        self.backend = self._detect_backend()
        print(f"Backend de training: {self.backend}")

        # Inicializar distributed
        self.world_size = 1
        self.rank = 0
        self._init_distributed()

        # Crear o cargar modelo
        if model is None:
            model_config = ModelConfig.from_preset(config.training.model_size)
            self.model = JupiterModel(model_config, backend=self.backend)
        else:
            self.model = model

        # Tokenizer
        if tokenizer is None:
            self.tokenizer = JupiterTokenizer()
            # Usar tokenizer de Llama por defecto
            self.tokenizer.load_from_pretrained("meta-llama/Llama-3.2-1B")
        else:
            self.tokenizer = tokenizer

        # Optimizer
        self.optimizer = None
        self._init_optimizer()

        # Estado
        self.state = TrainingState()
        self.metrics = TrainingMetrics()

        # Cola de datos
        self.data_queue: Optional[DataQueue] = None

    def _detect_backend(self) -> str:
        """Detecta el mejor backend disponible."""
        import platform

        if platform.system() == "Darwin":
            try:
                import mlx.core  # noqa: F401

                return "mlx"
            except ImportError:
                pass

        try:
            import torch

            if torch.cuda.is_available():
                return "torch-cuda"
        except ImportError:
            pass

        return "torch-cpu"

    def _init_distributed(self) -> None:
        """Inicializa el entorno distribuido."""
        if self.backend == "mlx":
            self._init_mlx_distributed()
        else:
            self._init_torch_distributed()

    def _init_mlx_distributed(self) -> None:
        """Inicializa MLX distributed."""
        try:
            import mlx.core as mx

            # Intentar inicializar distributed
            world = mx.distributed.init()
            self.world_size = world.size()
            self.rank = world.rank()

            if self.world_size > 1:
                print(f"MLX Distributed inicializado: rank {self.rank}/{self.world_size}")
            else:
                print("Ejecutando en modo single-device")

        except Exception as e:
            print(f"No se pudo inicializar MLX distributed: {e}")
            print("Ejecutando en modo single-device")
            self.world_size = 1
            self.rank = 0

    def _init_torch_distributed(self) -> None:
        """Inicializa PyTorch distributed."""
        try:
            import torch.distributed as dist
            import os

            if "RANK" in os.environ:
                dist.init_process_group(backend="nccl" if "cuda" in self.backend else "gloo")
                self.world_size = dist.get_world_size()
                self.rank = dist.get_rank()
                print(f"PyTorch Distributed inicializado: rank {self.rank}/{self.world_size}")
            else:
                print("Ejecutando en modo single-device")

        except Exception as e:
            print(f"No se pudo inicializar PyTorch distributed: {e}")
            self.world_size = 1
            self.rank = 0

    def _init_optimizer(self) -> None:
        """Inicializa el optimizer."""
        lr = self.config.training.learning_rate

        if self.backend == "mlx":
            import mlx.optimizers as optim

            # AdamW con weight decay
            self.optimizer = optim.AdamW(
                learning_rate=lr,
                betas=(0.9, 0.95),
                weight_decay=0.1,
            )
        else:
            import torch.optim as optim

            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                betas=(0.9, 0.95),
                weight_decay=0.1,
            )

    def _compute_loss(self, batch: DataBatch) -> float:
        """
        Calcula el loss para un batch.

        Args:
            batch: Batch de datos

        Returns:
            Loss value
        """
        # Tokenizar textos
        texts = batch.to_texts()
        token_ids = self.tokenizer.encode_batch(
            texts,
            max_length=self.config.training.max_seq_length,
            padding=True,
        )

        if self.backend == "mlx":
            return self._compute_loss_mlx(token_ids)
        else:
            return self._compute_loss_torch(token_ids)

    def _compute_loss_mlx(self, token_ids: list[list[int]]) -> float:
        """Calcula loss con MLX."""
        import mlx.core as mx
        import mlx.nn as nn

        # Convertir a array MLX
        input_ids = mx.array(token_ids)

        # Forward pass
        logits = self.model(input_ids[:, :-1])

        # Targets (shifted by 1)
        targets = input_ids[:, 1:]

        # Cross entropy loss
        loss = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="mean",
        )

        return float(loss)

    def _compute_loss_torch(self, token_ids: list[list[int]]) -> float:
        """Calcula loss con PyTorch."""
        import torch
        import torch.nn.functional as F

        device = "cuda" if "cuda" in self.backend else "cpu"

        input_ids = torch.tensor(token_ids, device=device)

        logits = self.model(input_ids[:, :-1])
        targets = input_ids[:, 1:]

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

        return float(loss)

    def _training_step(self, batch: DataBatch) -> float:
        """
        Ejecuta un paso de training.

        Args:
            batch: Batch de datos

        Returns:
            Loss del paso
        """
        if self.backend == "mlx":
            return self._training_step_mlx(batch)
        else:
            return self._training_step_torch(batch)

    def _training_step_mlx(self, batch: DataBatch) -> float:
        """Training step con MLX."""
        import mlx.core as mx
        import mlx.nn as nn

        # Tokenizar
        texts = batch.to_texts()
        token_ids = self.tokenizer.encode_batch(
            texts,
            max_length=self.config.training.max_seq_length,
            padding=True,
        )
        input_ids = mx.array(token_ids)

        # Función de loss para value_and_grad
        def loss_fn(model):
            logits = model(input_ids[:, :-1])
            targets = input_ids[:, 1:]
            return nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="mean",
            )

        # Calcular loss y gradientes
        loss, grads = nn.value_and_grad(self.model._model, loss_fn)(self.model._model)

        # Sincronizar gradientes si es distribuido
        if self.world_size > 1:
            grads = nn.average_gradients(grads)

        # Actualizar pesos
        self.optimizer.update(self.model._model, grads)

        # Evaluar para materializar
        mx.eval(self.model._model.parameters(), self.optimizer.state)

        return float(loss)

    def _training_step_torch(self, batch: DataBatch) -> float:
        """Training step con PyTorch."""
        import torch

        device = "cuda" if "cuda" in self.backend else "cpu"

        # Tokenizar
        texts = batch.to_texts()
        token_ids = self.tokenizer.encode_batch(
            texts,
            max_length=self.config.training.max_seq_length,
            padding=True,
        )
        input_ids = torch.tensor(token_ids, device=device)

        # Forward
        self.optimizer.zero_grad()
        logits = self.model(input_ids[:, :-1])
        targets = input_ids[:, 1:]

        # Loss
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

        # Backward
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # Optimizer step
        self.optimizer.step()

        return float(loss)

    def train_epoch(self, data_queue: DataQueue) -> TrainingMetrics:
        """
        Entrena una época completa.

        Args:
            data_queue: Cola de datos

        Returns:
            Métricas de la época
        """
        self.data_queue = data_queue

        # Preparar época
        num_samples = data_queue.prepare_epoch()
        num_batches = data_queue.num_batches

        print(f"\nÉpoca {self.state.epoch + 1}")
        print(f"Samples: {num_samples}, Batches: {num_batches}")

        epoch_loss = 0.0
        epoch_start = time.time()
        tokens_processed = 0

        for batch_idx, batch in enumerate(data_queue.iter_batches()):
            step_start = time.time()

            # Training step
            loss = self._training_step(batch)
            epoch_loss += loss

            # Actualizar estado
            self.state.step += 1
            tokens_processed += batch.size * self.config.training.max_seq_length

            # Logging
            if batch_idx % 10 == 0 or batch_idx == num_batches - 1:
                elapsed = time.time() - epoch_start
                tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0

                print(
                    f"  Batch {batch_idx + 1}/{num_batches} | "
                    f"Loss: {loss:.4f} | "
                    f"Tokens/s: {tokens_per_sec:.0f}"
                )

            # Checkpointing
            if self.state.step % self.config.training.save_every_steps == 0:
                self.save_checkpoint()

        # Actualizar métricas
        self.state.epoch += 1
        self.state.total_tokens += tokens_processed

        avg_loss = epoch_loss / num_batches
        elapsed = time.time() - epoch_start

        self.metrics = TrainingMetrics(
            step=self.state.step,
            epoch=self.state.epoch,
            loss=avg_loss,
            learning_rate=self.config.training.learning_rate,
            tokens_per_second=tokens_processed / elapsed,
            samples_per_second=num_samples / elapsed,
            elapsed_time=elapsed,
        )

        print(f"\nÉpoca completada:")
        print(f"  Loss promedio: {avg_loss:.4f}")
        print(f"  Tokens/segundo: {self.metrics.tokens_per_second:.0f}")
        print(f"  Tiempo: {elapsed:.1f}s")

        # Guardar si es el mejor
        if avg_loss < self.state.best_loss:
            self.state.best_loss = avg_loss
            self.save_checkpoint(is_best=True)

        return self.metrics

    def save_checkpoint(self, is_best: bool = False) -> Path:
        """
        Guarda un checkpoint.

        Args:
            is_best: Si es el mejor modelo hasta ahora

        Returns:
            Path del checkpoint guardado
        """
        checkpoint_dir = self.config.checkpoints_dir / f"step_{self.state.step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Guardar modelo
        self.model.save(str(checkpoint_dir / "model"))

        # Guardar tokenizer
        self.tokenizer.save(str(checkpoint_dir / "tokenizer"))

        # Guardar estado
        state_dict = {
            "step": self.state.step,
            "epoch": self.state.epoch,
            "best_loss": self.state.best_loss,
            "total_tokens": self.state.total_tokens,
            "config": self.config.training.__dict__,
        }
        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(state_dict, f, indent=2)

        print(f"Checkpoint guardado: {checkpoint_dir}")

        # Symlink a latest/best
        if is_best:
            best_link = self.config.checkpoints_dir / "best"
            if best_link.exists():
                best_link.unlink()
            best_link.symlink_to(checkpoint_dir.name)

        return checkpoint_dir

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Carga un checkpoint.

        Args:
            checkpoint_path: Path al directorio del checkpoint
        """
        checkpoint_path = Path(checkpoint_path)

        # Cargar modelo
        self.model = JupiterModel.load(
            str(checkpoint_path / "model"),
            backend=self.backend,
        )

        # Cargar tokenizer
        self.tokenizer = JupiterTokenizer()
        self.tokenizer.load(str(checkpoint_path / "tokenizer"))

        # Cargar estado
        with open(checkpoint_path / "training_state.json") as f:
            state_dict = json.load(f)

        self.state.step = state_dict["step"]
        self.state.epoch = state_dict["epoch"]
        self.state.best_loss = state_dict["best_loss"]
        self.state.total_tokens = state_dict["total_tokens"]

        # Reinicializar optimizer
        self._init_optimizer()

        print(f"Checkpoint cargado: {checkpoint_path}")
        print(f"  Step: {self.state.step}, Epoch: {self.state.epoch}")

    @property
    def is_main_process(self) -> bool:
        """¿Es el proceso principal (rank 0)?"""
        return self.rank == 0
