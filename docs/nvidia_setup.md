# Añadir GPUs NVIDIA al Cluster

Jupiter soporta clusters híbridos con Macs y GPUs NVIDIA trabajando juntos.

## Arquitectura Híbrida

```
┌─────────────────────────────────────────────────────────────┐
│                    CLUSTER HÍBRIDO                          │
│                                                             │
│   ┌─────────────────────┐       ┌─────────────────────┐    │
│   │   Cluster Mac       │       │   Nodos NVIDIA      │    │
│   │   (MLX)             │       │   (PyTorch)         │    │
│   │                     │       │                     │    │
│   │ ┌─────┐ ┌─────┐    │       │ ┌─────────────────┐ │    │
│   │ │Mac 1│ │Mac 2│    │       │ │  Linux + RTX    │ │    │
│   │ └─────┘ └─────┘    │       │ │  4090 / 3090    │ │    │
│   │ ┌─────┐ ┌─────┐    │       │ └─────────────────┘ │    │
│   │ │Mac 3│ │Mac 4│    │       │                     │    │
│   │ └─────┘ └─────┘    │       │                     │    │
│   └──────────┬──────────┘       └──────────┬──────────┘    │
│              │                             │               │
│              └──────────┬──────────────────┘               │
│                         │                                  │
│              ┌──────────▼──────────┐                       │
│              │   Sincronización    │                       │
│              │   via Ethernet      │                       │
│              └─────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

## Requisitos

### Nodo NVIDIA
- Linux (Ubuntu 22.04+ recomendado)
- GPU NVIDIA (RTX 3090, 4090, etc.)
- CUDA 12.0+
- 24GB+ VRAM para modelos 7B

### Conectividad
- Red Ethernet entre Macs y PC NVIDIA
- SSH sin contraseña desde el nodo líder

## Paso 1: Preparar el Nodo NVIDIA

### Instalar CUDA

```bash
# Ubuntu
sudo apt update
sudo apt install nvidia-driver-535 nvidia-cuda-toolkit
```

### Instalar Python y dependencias

```bash
# Python 3.11
sudo apt install python3.11 python3.11-venv

# Clonar Jupiter
git clone https://github.com/raym33/jupiter.git
cd jupiter

# Crear entorno
python3.11 -m venv venv
source venv/bin/activate

# Instalar con soporte NVIDIA
pip install -e ".[nvidia]"
```

### Verificar CUDA

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name()}')"
```

## Paso 2: Configurar SSH

Desde el Mac líder:

```bash
ssh-copy-id usuario@192.168.1.100  # IP del nodo NVIDIA
```

## Paso 3: Configurar cluster.yaml

```yaml
backend: "mpi"

nodes:
  # Macs (como antes)
  - host: "macbook.local"
    device_type: "mac"
    memory_gb: 24
    role: "generator"

  - host: "mac-mini-1.local"
    device_type: "mac"
    memory_gb: 16
    role: "trainer"

  # ... más Macs

  # Nodo NVIDIA
  - host: "192.168.1.100"
    name: "Workstation RTX 4090"
    device_type: "nvidia"
    memory_gb: 24  # VRAM
    chip: "RTX 4090"
    role: "trainer"
    connection_type: "ethernet"
    ssh_user: "usuario"
```

## Cómo Funciona

### Sincronización de Gradientes

Los clusters Mac y NVIDIA no comparten gradientes directamente (MLX vs PyTorch).
En su lugar:

1. Cada cluster entrena de forma semi-independiente
2. Los checkpoints se sincronizan periódicamente
3. Se usa "model averaging" para combinar los pesos

```
Época 1:
  Macs: entrenan steps 1-100
  NVIDIA: entrena steps 1-100
  → Promediar pesos

Época 2:
  Macs: entrenan steps 101-200 (con pesos promediados)
  NVIDIA: entrena steps 101-200 (con pesos promediados)
  → Promediar pesos

...
```

### Ventajas

1. **Más compute total**: Sumas la capacidad de ambos clusters
2. **Diversidad**: Diferentes inicializaciones pueden mejorar generalización
3. **Resiliencia**: Si un cluster falla, el otro continúa

### Limitaciones

1. **No es data parallelism puro**: Los gradientes no se promedian cada step
2. **Overhead de sincronización**: Copiar checkpoints toma tiempo
3. **Complejidad**: Más difícil de debuggear

## Modo Solo NVIDIA

Si solo tienes GPUs NVIDIA (sin Macs):

```yaml
backend: "torch"

nodes:
  - host: "gpu-server-1"
    device_type: "nvidia"
    memory_gb: 24
    chip: "RTX 4090"
    role: "hybrid"  # Genera y entrena

  - host: "gpu-server-2"
    device_type: "nvidia"
    memory_gb: 24
    chip: "RTX 4090"
    role: "trainer"
```

## Optimizaciones para NVIDIA

### Flash Attention

```bash
pip install flash-attn --no-build-isolation
```

En el código:
```python
config = ModelConfig.from_preset("1b")
config.use_flash_attention = True
```

### Mixed Precision

Habilitado por defecto para NVIDIA. Usa FP16 para forward/backward y FP32 para acumulación.

### Gradient Checkpointing

Para modelos grandes que no caben en VRAM:

```python
# En config de training
gradient_checkpointing = True
```

## Solución de Problemas

### "CUDA out of memory"

1. Reducir batch size
2. Habilitar gradient checkpointing
3. Usar modelo más pequeño (500M en vez de 1B)

### Sincronización lenta

1. Verificar velocidad de red (debería ser 1Gbps+)
2. Aumentar intervalo de sincronización
3. Considerar compresión de checkpoints

### "GPU not detected"

```bash
# Verificar driver
nvidia-smi

# Verificar CUDA
nvcc --version

# Verificar PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

## Próximos Pasos

- [Configurar cluster de Macs](mac_cluster_setup.md)
- [Crear un nuevo dominio](new_domain.md)
