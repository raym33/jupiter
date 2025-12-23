"""
Configuración del cluster de dispositivos.

Soporta:
- Macs con Apple Silicon (M1/M2/M3/M4/M5) via MLX
- GPUs NVIDIA via PyTorch/CUDA
- Topologías: Thunderbolt, Ethernet, mixtas
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class DeviceType(str, Enum):
    """Tipo de dispositivo."""

    MAC_APPLE_SILICON = "mac"
    NVIDIA_GPU = "nvidia"
    CPU_ONLY = "cpu"


class NodeRole(str, Enum):
    """Rol del nodo en el cluster."""

    GENERATOR = "generator"  # Genera datos sintéticos
    TRAINER = "trainer"  # Entrena el modelo
    HYBRID = "hybrid"  # Ambos roles
    EVALUATOR = "evaluator"  # Solo evaluación


class ConnectionType(str, Enum):
    """Tipo de conexión entre nodos."""

    THUNDERBOLT = "thunderbolt"
    ETHERNET = "ethernet"
    WIFI = "wifi"  # No recomendado para training


@dataclass
class NodeConfig:
    """Configuración de un nodo del cluster."""

    # Identificación
    host: str  # hostname o IP
    name: Optional[str] = None  # Nombre amigable

    # Hardware
    device_type: DeviceType = DeviceType.MAC_APPLE_SILICON
    memory_gb: int = 16  # RAM (Mac) o VRAM (NVIDIA)
    chip: Optional[str] = None  # "M4", "M4 Pro", "RTX 4090", etc.

    # Rol
    role: NodeRole = NodeRole.TRAINER

    # Conexión
    connection_type: ConnectionType = ConnectionType.THUNDERBOLT
    ssh_user: Optional[str] = None
    ssh_port: int = 22

    # Capacidades
    can_generate: bool = True
    can_train: bool = True

    @property
    def display_name(self) -> str:
        """Nombre para mostrar."""
        return self.name or self.host

    @property
    def is_mac(self) -> bool:
        """¿Es un Mac con Apple Silicon?"""
        return self.device_type == DeviceType.MAC_APPLE_SILICON

    @property
    def is_nvidia(self) -> bool:
        """¿Es una GPU NVIDIA?"""
        return self.device_type == DeviceType.NVIDIA_GPU


@dataclass
class ClusterConfig:
    """Configuración completa del cluster."""

    # Nodos
    nodes: list[NodeConfig] = field(default_factory=list)

    # Backend de comunicación
    backend: str = "mpi"  # "mpi", "ring", "jaccl" (solo TB5)

    # Red
    network_interface: Optional[str] = None  # e.g., "en0", "en5"
    use_thunderbolt_bridge: bool = True

    # Sincronización
    sync_every_n_steps: int = 1
    gradient_compression: bool = False

    # Tolerancia a fallos
    checkpoint_on_failure: bool = True
    auto_restart_failed_nodes: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "ClusterConfig":
        """Crea ClusterConfig desde un diccionario."""

        nodes = []
        for node_data in data.get("nodes", []):
            nodes.append(
                NodeConfig(
                    host=node_data.get("host", "localhost"),
                    name=node_data.get("name"),
                    device_type=DeviceType(node_data.get("device_type", "mac")),
                    memory_gb=node_data.get("memory_gb", 16),
                    chip=node_data.get("chip"),
                    role=NodeRole(node_data.get("role", "trainer")),
                    connection_type=ConnectionType(
                        node_data.get("connection_type", "thunderbolt")
                    ),
                    ssh_user=node_data.get("ssh_user"),
                    ssh_port=node_data.get("ssh_port", 22),
                    can_generate=node_data.get("can_generate", True),
                    can_train=node_data.get("can_train", True),
                )
            )

        return cls(
            nodes=nodes,
            backend=data.get("backend", "mpi"),
            network_interface=data.get("network_interface"),
            use_thunderbolt_bridge=data.get("use_thunderbolt_bridge", True),
            sync_every_n_steps=data.get("sync_every_n_steps", 1),
            gradient_compression=data.get("gradient_compression", False),
            checkpoint_on_failure=data.get("checkpoint_on_failure", True),
            auto_restart_failed_nodes=data.get("auto_restart_failed_nodes", True),
        )

    @property
    def generator_nodes(self) -> list[NodeConfig]:
        """Nodos que pueden generar datos."""
        return [n for n in self.nodes if n.role in (NodeRole.GENERATOR, NodeRole.HYBRID)]

    @property
    def trainer_nodes(self) -> list[NodeConfig]:
        """Nodos que pueden entrenar."""
        return [n for n in self.nodes if n.role in (NodeRole.TRAINER, NodeRole.HYBRID)]

    @property
    def mac_nodes(self) -> list[NodeConfig]:
        """Nodos Mac."""
        return [n for n in self.nodes if n.is_mac]

    @property
    def nvidia_nodes(self) -> list[NodeConfig]:
        """Nodos NVIDIA."""
        return [n for n in self.nodes if n.is_nvidia]

    @property
    def total_memory_gb(self) -> int:
        """Memoria total del cluster."""
        return sum(n.memory_gb for n in self.nodes)

    @property
    def total_trainer_memory_gb(self) -> int:
        """Memoria total de nodos de training."""
        return sum(n.memory_gb for n in self.trainer_nodes)

    def get_node(self, host: str) -> Optional[NodeConfig]:
        """Obtiene un nodo por su host."""
        for node in self.nodes:
            if node.host == host:
                return node
        return None

    def validate(self) -> list[str]:
        """Valida la configuración y retorna lista de errores."""
        errors = []

        if not self.nodes:
            errors.append("No hay nodos configurados")

        if not self.generator_nodes:
            errors.append("No hay nodos configurados como generadores")

        if not self.trainer_nodes:
            errors.append("No hay nodos configurados para training")

        # Verificar que haya suficiente memoria
        if self.total_trainer_memory_gb < 16:
            errors.append("Memoria total de training insuficiente (mínimo 16GB)")

        # Verificar hosts duplicados
        hosts = [n.host for n in self.nodes]
        if len(hosts) != len(set(hosts)):
            errors.append("Hay hosts duplicados en la configuración")

        return errors
