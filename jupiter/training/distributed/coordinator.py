"""
Coordinador de training distribuido.

Maneja:
- Descubrimiento de nodos
- Distribución de trabajo
- Sincronización entre Macs y NVIDIA
- Monitoreo del cluster
"""

import asyncio
import json
import socket
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import subprocess


@dataclass
class NodeStatus:
    """Estado de un nodo del cluster."""

    host: str
    name: str
    is_online: bool = False
    is_ready: bool = False
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    current_step: int = 0
    samples_processed: int = 0
    last_heartbeat: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class ClusterStatus:
    """Estado del cluster completo."""

    nodes: list[NodeStatus] = field(default_factory=list)
    is_healthy: bool = True
    total_memory_gb: float = 0.0
    active_nodes: int = 0
    current_step: int = 0
    start_time: Optional[datetime] = None


class TrainingCoordinator:
    """
    Coordinador central para training distribuido.

    Responsabilidades:
    - Verificar conectividad con todos los nodos
    - Sincronizar estado entre nodos
    - Distribuir datos y comandos
    - Monitorear progreso y salud
    """

    def __init__(self, cluster_config: "ClusterConfig"):  # noqa: F821
        """
        Args:
            cluster_config: Configuración del cluster
        """
        self.config = cluster_config
        self.nodes: dict[str, NodeStatus] = {}
        self.cluster_status = ClusterStatus()
        self._running = False

    async def initialize(self) -> bool:
        """
        Inicializa el coordinador y verifica el cluster.

        Returns:
            True si el cluster está listo
        """
        print("Inicializando coordinador de training...")

        # Crear status para cada nodo
        for node in self.config.nodes:
            self.nodes[node.host] = NodeStatus(
                host=node.host,
                name=node.display_name,
                memory_total_gb=node.memory_gb,
            )

        # Verificar conectividad
        online_count = await self._check_connectivity()
        print(f"Nodos online: {online_count}/{len(self.nodes)}")

        if online_count == 0:
            print("ERROR: No hay nodos disponibles")
            return False

        # Verificar que los nodos tengan las dependencias
        ready_count = await self._verify_dependencies()
        print(f"Nodos listos: {ready_count}/{online_count}")

        if ready_count == 0:
            print("ERROR: Ningún nodo tiene las dependencias instaladas")
            return False

        # Actualizar cluster status
        self.cluster_status.is_healthy = True
        self.cluster_status.active_nodes = ready_count
        self.cluster_status.total_memory_gb = sum(
            n.memory_total_gb for n in self.nodes.values() if n.is_ready
        )
        self.cluster_status.start_time = datetime.now()

        print(f"Cluster listo: {ready_count} nodos, {self.cluster_status.total_memory_gb}GB RAM")
        return True

    async def _check_connectivity(self) -> int:
        """
        Verifica conectividad SSH con todos los nodos.

        Returns:
            Número de nodos online
        """
        tasks = []
        for host, status in self.nodes.items():
            tasks.append(self._ping_node(host, status))

        await asyncio.gather(*tasks)

        return sum(1 for n in self.nodes.values() if n.is_online)

    async def _ping_node(self, host: str, status: NodeStatus) -> None:
        """Verifica si un nodo está online via SSH."""
        # Primero intentar ping simple
        try:
            # Resolver hostname
            socket.gethostbyname(host)
        except socket.gaierror:
            status.is_online = False
            status.error_message = "No se puede resolver hostname"
            return

        # Intentar SSH
        try:
            proc = await asyncio.create_subprocess_exec(
                "ssh",
                "-o", "ConnectTimeout=5",
                "-o", "BatchMode=yes",
                host,
                "echo ok",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)

            if proc.returncode == 0 and b"ok" in stdout:
                status.is_online = True
                status.last_heartbeat = datetime.now()
            else:
                status.is_online = False
                status.error_message = stderr.decode().strip()

        except asyncio.TimeoutError:
            status.is_online = False
            status.error_message = "Timeout de conexión"
        except Exception as e:
            status.is_online = False
            status.error_message = str(e)

    async def _verify_dependencies(self) -> int:
        """
        Verifica que los nodos tengan las dependencias necesarias.

        Returns:
            Número de nodos listos
        """
        tasks = []
        for host, status in self.nodes.items():
            if status.is_online:
                tasks.append(self._check_node_deps(host, status))

        if tasks:
            await asyncio.gather(*tasks)

        return sum(1 for n in self.nodes.values() if n.is_ready)

    async def _check_node_deps(self, host: str, status: NodeStatus) -> None:
        """Verifica dependencias en un nodo."""
        try:
            # Verificar Python y MLX
            cmd = 'python3 -c "import mlx.core; print(\'ok\')"'

            proc = await asyncio.create_subprocess_exec(
                "ssh",
                "-o", "ConnectTimeout=10",
                host,
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

            if proc.returncode == 0 and b"ok" in stdout:
                status.is_ready = True
            else:
                status.is_ready = False
                status.error_message = f"MLX no disponible: {stderr.decode().strip()}"

        except Exception as e:
            status.is_ready = False
            status.error_message = str(e)

    async def launch_training(
        self,
        script_path: Path,
        args: list[str] = None,
    ) -> bool:
        """
        Lanza el training en todos los nodos.

        Args:
            script_path: Path al script de training
            args: Argumentos adicionales

        Returns:
            True si se lanzó correctamente
        """
        if not self.cluster_status.is_healthy:
            print("ERROR: Cluster no está listo")
            return False

        ready_nodes = [n for n in self.nodes.values() if n.is_ready]
        num_nodes = len(ready_nodes)

        if num_nodes == 0:
            print("ERROR: No hay nodos listos")
            return False

        print(f"\nLanzando training en {num_nodes} nodos...")

        # Construir comando MPI
        hostfile = self._create_hostfile()

        mpi_cmd = [
            "mpirun",
            "-np", str(num_nodes),
            "-hostfile", str(hostfile),
            "--map-by", "node",
            "python3", str(script_path),
        ]

        if args:
            mpi_cmd.extend(args)

        print(f"Comando: {' '.join(mpi_cmd)}")

        # Lanzar
        try:
            process = await asyncio.create_subprocess_exec(
                *mpi_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            self._running = True

            # Leer output en tiempo real
            async for line in process.stdout:
                print(line.decode().rstrip())

            await process.wait()
            self._running = False

            return process.returncode == 0

        except Exception as e:
            print(f"ERROR lanzando training: {e}")
            self._running = False
            return False

    def _create_hostfile(self) -> Path:
        """
        Crea archivo hostfile para MPI.

        Returns:
            Path al hostfile
        """
        hostfile = Path("/tmp/jupiter_hostfile")

        with open(hostfile, "w") as f:
            for node in self.config.nodes:
                status = self.nodes.get(node.host)
                if status and status.is_ready:
                    # Formato: hostname slots=N
                    f.write(f"{node.host} slots=1\n")

        return hostfile

    async def monitor_training(self) -> None:
        """
        Monitorea el progreso del training.

        Ejecutar en paralelo con launch_training.
        """
        while self._running:
            # Actualizar status de nodos
            for host, status in self.nodes.items():
                if status.is_ready:
                    await self._update_node_status(host, status)

            # Calcular métricas globales
            total_samples = sum(n.samples_processed for n in self.nodes.values())
            avg_step = sum(n.current_step for n in self.nodes.values() if n.is_ready)
            if self.cluster_status.active_nodes > 0:
                avg_step //= self.cluster_status.active_nodes

            self.cluster_status.current_step = avg_step

            # Esperar antes de siguiente check
            await asyncio.sleep(5)

    async def _update_node_status(self, host: str, status: NodeStatus) -> None:
        """Actualiza el estado de un nodo."""
        try:
            # Leer archivo de status si existe
            cmd = "cat /tmp/jupiter_node_status.json 2>/dev/null || echo '{}'"

            proc = await asyncio.create_subprocess_exec(
                "ssh",
                "-o", "ConnectTimeout=5",
                host,
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)

            if proc.returncode == 0:
                data = json.loads(stdout.decode())
                status.current_step = data.get("step", 0)
                status.samples_processed = data.get("samples", 0)
                status.memory_used_gb = data.get("memory_used_gb", 0)
                status.last_heartbeat = datetime.now()

        except Exception:
            pass  # Ignorar errores de status update

    def get_cluster_summary(self) -> str:
        """
        Obtiene un resumen del estado del cluster.

        Returns:
            Resumen formateado
        """
        lines = [
            "=== Estado del Cluster ===",
            f"Nodos activos: {self.cluster_status.active_nodes}/{len(self.nodes)}",
            f"Memoria total: {self.cluster_status.total_memory_gb}GB",
            f"Step actual: {self.cluster_status.current_step}",
            "",
            "Nodos:",
        ]

        for host, status in self.nodes.items():
            state = "✓" if status.is_ready else ("○" if status.is_online else "✗")
            lines.append(
                f"  {state} {status.name} ({host}): "
                f"{status.memory_used_gb:.1f}/{status.memory_total_gb}GB, "
                f"step {status.current_step}"
            )
            if status.error_message:
                lines.append(f"      Error: {status.error_message}")

        return "\n".join(lines)

    async def shutdown(self) -> None:
        """Detiene el training y limpia recursos."""
        self._running = False

        # Enviar señal de stop a todos los nodos
        for host, status in self.nodes.items():
            if status.is_ready:
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "ssh", host, "pkill -f jupiter_train || true",
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    await proc.wait()
                except Exception:
                    pass

        print("Training detenido")
