"""
Sincronización de gradientes para training distribuido.

Implementa operaciones de all-reduce para promediar gradientes
entre nodos MLX y opcionalmente NVIDIA.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import time


@dataclass
class SyncStats:
    """Estadísticas de sincronización."""

    num_syncs: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    last_sync_time_ms: float = 0.0
    gradients_synced: int = 0


class GradientSynchronizer:
    """
    Sincronizador de gradientes para training distribuido.

    Soporta:
    - MLX distributed (all_sum / average_gradients)
    - PyTorch DDP (all_reduce)
    - Sincronización híbrida MLX + PyTorch
    """

    def __init__(
        self,
        backend: str = "auto",
        compression: bool = False,
        sync_every_n_steps: int = 1,
    ):
        """
        Args:
            backend: "auto", "mlx", "torch", o "hybrid"
            compression: Usar compresión de gradientes
            sync_every_n_steps: Sincronizar cada N steps
        """
        self.backend = self._detect_backend() if backend == "auto" else backend
        self.compression = compression
        self.sync_every_n_steps = sync_every_n_steps

        self.stats = SyncStats()
        self._step = 0
        self._world_size = 1
        self._rank = 0

        self._init_backend()

    def _detect_backend(self) -> str:
        """Detecta el backend disponible."""
        try:
            import mlx.core as mx

            world = mx.distributed.init()
            if world.size() > 1:
                return "mlx"
        except Exception:
            pass

        try:
            import torch.distributed as dist

            if dist.is_initialized():
                return "torch"
        except Exception:
            pass

        return "none"

    def _init_backend(self) -> None:
        """Inicializa el backend de sincronización."""
        if self.backend == "mlx":
            self._init_mlx()
        elif self.backend == "torch":
            self._init_torch()
        elif self.backend == "hybrid":
            self._init_hybrid()
        else:
            print("Sincronización deshabilitada (single node)")

    def _init_mlx(self) -> None:
        """Inicializa MLX distributed."""
        import mlx.core as mx

        world = mx.distributed.init()
        self._world_size = world.size()
        self._rank = world.rank()

        print(f"GradientSync MLX: rank {self._rank}/{self._world_size}")

    def _init_torch(self) -> None:
        """Inicializa PyTorch distributed."""
        import torch.distributed as dist

        if not dist.is_initialized():
            print("WARNING: PyTorch distributed no inicializado")
            self.backend = "none"
            return

        self._world_size = dist.get_world_size()
        self._rank = dist.get_rank()

        print(f"GradientSync PyTorch: rank {self._rank}/{self._world_size}")

    def _init_hybrid(self) -> None:
        """Inicializa modo híbrido MLX + PyTorch."""
        # TODO: Implementar sincronización entre clusters MLX y PyTorch
        # Por ahora, solo usar MLX si está disponible
        try:
            self._init_mlx()
            self.backend = "mlx"
        except Exception:
            self._init_torch()
            self.backend = "torch"

    def should_sync(self) -> bool:
        """
        Verifica si debemos sincronizar en este step.

        Returns:
            True si debemos sincronizar
        """
        self._step += 1
        return (
            self._world_size > 1
            and self._step % self.sync_every_n_steps == 0
        )

    def sync_gradients(self, gradients: Any) -> Any:
        """
        Sincroniza gradientes entre nodos.

        Args:
            gradients: Gradientes a sincronizar (estructura de arrays)

        Returns:
            Gradientes promediados
        """
        if not self.should_sync():
            return gradients

        start_time = time.time()

        if self.backend == "mlx":
            synced = self._sync_mlx(gradients)
        elif self.backend == "torch":
            synced = self._sync_torch(gradients)
        else:
            synced = gradients

        # Actualizar stats
        elapsed_ms = (time.time() - start_time) * 1000
        self.stats.num_syncs += 1
        self.stats.total_time_ms += elapsed_ms
        self.stats.last_sync_time_ms = elapsed_ms
        self.stats.avg_time_ms = self.stats.total_time_ms / self.stats.num_syncs

        return synced

    def _sync_mlx(self, gradients: Any) -> Any:
        """Sincroniza gradientes con MLX."""
        import mlx.nn as nn

        # Usar la función optimizada de MLX
        return nn.average_gradients(gradients)

    def _sync_torch(self, gradients: Any) -> Any:
        """Sincroniza gradientes con PyTorch."""
        import torch
        import torch.distributed as dist

        # All-reduce cada tensor
        for key, grad in gradients.items():
            if isinstance(grad, torch.Tensor):
                dist.all_reduce(grad, op=dist.ReduceOp.AVG)

        return gradients

    def sync_model_params(self, model: Any) -> None:
        """
        Sincroniza parámetros del modelo entre nodos.

        Útil para broadcast inicial o recovery.

        Args:
            model: Modelo a sincronizar
        """
        if self._world_size <= 1:
            return

        if self.backend == "mlx":
            self._broadcast_mlx(model)
        elif self.backend == "torch":
            self._broadcast_torch(model)

    def _broadcast_mlx(self, model: Any) -> None:
        """Broadcast de parámetros con MLX."""
        import mlx.core as mx

        # Broadcast desde rank 0
        params = model.parameters()

        for name, param in params.items():
            # All-reduce para asegurar que todos tengan los mismos valores
            synced = mx.distributed.all_sum(param) / self._world_size
            # Actualizar in-place
            param[:] = synced

        mx.eval(model.parameters())

    def _broadcast_torch(self, model: Any) -> None:
        """Broadcast de parámetros con PyTorch."""
        import torch.distributed as dist

        for param in model.parameters():
            dist.broadcast(param.data, src=0)

    def barrier(self) -> None:
        """
        Barrera de sincronización.

        Bloquea hasta que todos los nodos lleguen a este punto.
        """
        if self._world_size <= 1:
            return

        if self.backend == "mlx":
            import mlx.core as mx
            # MLX no tiene barrera explícita, usar all_sum como barrera
            mx.distributed.all_sum(mx.zeros(1))
        elif self.backend == "torch":
            import torch.distributed as dist
            dist.barrier()

    def get_stats_summary(self) -> str:
        """
        Obtiene resumen de estadísticas.

        Returns:
            Resumen formateado
        """
        return (
            f"Sync Stats: {self.stats.num_syncs} syncs, "
            f"avg {self.stats.avg_time_ms:.2f}ms, "
            f"last {self.stats.last_sync_time_ms:.2f}ms"
        )

    @property
    def world_size(self) -> int:
        """Número de nodos en el cluster."""
        return self._world_size

    @property
    def rank(self) -> int:
        """Rank de este nodo."""
        return self._rank

    @property
    def is_main(self) -> bool:
        """¿Es el nodo principal (rank 0)?"""
        return self._rank == 0
