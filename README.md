# Jupiter

**Framework de entrenamiento distribuido para crear modelos de lenguaje expertos en dominios específicos.**

Entrena tu propio modelo de 1B parámetros usando un cluster de Macs (M4/M5) y/o GPUs NVIDIA, con generación automática de datos sintéticos y ciclo de auto-mejora.

---

## Características

- **Dominio configurable**: Cambia un archivo YAML para entrenar expertos en cualquier área (Unreal Engine, Python, Química, etc.)
- **Training distribuido**: Soporta clusters de Macs via Thunderbolt y GPUs NVIDIA via NCCL
- **Generación de datos sintéticos**: Usa modelos locales (Llama, Mistral) para generar datos de entrenamiento
- **Ciclo auto-mejorante**: El modelo entrenado puede reemplazar al generador cuando lo supere
- **Hardware heterogéneo**: Combina Macs con diferentes cantidades de RAM y GPUs NVIDIA

---

## Requisitos de Hardware

### Configuración Mínima (Solo Macs)
| Dispositivo | RAM | Rol |
|-------------|-----|-----|
| 1× Mac (M4/M5) | 16-24GB | Generador de datos |
| 1-4× Mac Mini (M4/M5) | 16GB+ c/u | Nodos de training |

### Configuración Híbrida (Macs + NVIDIA)
| Dispositivo | VRAM/RAM | Rol |
|-------------|----------|-----|
| Macs | 16GB+ | Generación + Training MLX |
| RTX 3090/4090 | 24GB | Training PyTorch |

### Conexiones
- **Entre Macs**: Thunderbolt 4/5 (recomendado) o Ethernet 10GbE
- **Macs ↔ NVIDIA**: Ethernet (los datos se sincronizan, no los gradientes directamente)

---

## Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/raym33/jupiter.git
cd jupiter
```

### 2. Instalar dependencias (Mac)
```bash
# Instalar Xcode CLI tools
xcode-select --install

# Instalar dependencias del sistema
brew install mpich

# Crear entorno virtual
python3.11 -m venv venv
source venv/bin/activate

# Instalar Jupiter
pip install -e ".[mac]"
```

### 3. Instalar dependencias (NVIDIA Linux)
```bash
# Crear entorno virtual
python3.11 -m venv venv
source venv/bin/activate

# Instalar Jupiter con soporte CUDA
pip install -e ".[nvidia]"
```

---

## Inicio Rápido

### Paso 1: Configurar tu dominio

Copia el template y edítalo:
```bash
cp config/domains/_template.yaml config/domains/mi_dominio.yaml
```

Edita `mi_dominio.yaml` con las fuentes de datos y configuración de tu dominio.

### Paso 2: Configurar el cluster

Edita `config/cluster.yaml` con tus dispositivos:
```yaml
nodes:
  - host: "macbook.local"
    role: "generator"
    memory_gb: 24

  - host: "mac-mini-1.local"
    role: "trainer"
    memory_gb: 16

  # Añadir más nodos...
```

### Paso 3: Iniciar el entrenamiento

```bash
# Modo interactivo (recomendado para empezar)
jupiter start --domain mi_dominio --interactive

# Modo automático (ciclo completo)
jupiter start --domain mi_dominio --auto
```

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                         JUPITER                                 │
│                                                                 │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐    │
│   │  RECOLECTOR │      │  GENERADOR  │      │   FILTRO    │    │
│   │  de datos   │ ───► │  sintético  │ ───► │  calidad    │    │
│   │  reales     │      │  (Llama)    │      │             │    │
│   └─────────────┘      └─────────────┘      └──────┬──────┘    │
│                                                    │           │
│                                              ┌─────▼─────┐     │
│                                              │   COLA    │     │
│                                              │  de datos │     │
│                                              └─────┬─────┘     │
│                                                    │           │
│   ┌────────────────────────────────────────────────┼─────────┐ │
│   │                TRAINING DISTRIBUIDO            │         │ │
│   │                                                ▼         │ │
│   │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │ │
│   │   │ Mac #1  │  │ Mac #2  │  │ Mac #3  │  │ RTX GPU │    │ │
│   │   │  MLX    │  │  MLX    │  │  MLX    │  │ PyTorch │    │ │
│   │   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘    │ │
│   │        │            │            │            │          │ │
│   │        └────────────┴─────┬──────┴────────────┘          │ │
│   │                           │                              │ │
│   │                    ┌──────▼──────┐                       │ │
│   │                    │  GRADIENT   │                       │ │
│   │                    │   SYNC      │                       │ │
│   │                    └──────┬──────┘                       │ │
│   │                           │                              │ │
│   └───────────────────────────┼──────────────────────────────┘ │
│                               │                                │
│                        ┌──────▼──────┐                         │
│                        │ CHECKPOINT  │                         │
│                        │  + EVAL     │                         │
│                        └──────┬──────┘                         │
│                               │                                │
│                        ┌──────▼──────┐                         │
│                        │ ¿SUPERA AL  │                         │
│                        │ GENERADOR?  │                         │
│                        └──────┬──────┘                         │
│                               │ Sí                             │
│                        ┌──────▼──────┐                         │
│                        │ REEMPLAZAR  │                         │
│                        │ GENERADOR   │ ──► CICLO CONTINÚA      │
│                        └─────────────┘                         │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Estructura del Proyecto

```
jupiter/
├── config/
│   ├── cluster.yaml           # Configuración de tu cluster
│   └── domains/
│       ├── _template.yaml     # Template para nuevos dominios
│       └── unreal_engine.yaml # Ejemplo: experto en UE5
│
├── data/
│   ├── collectors/            # Recolectores de datos reales
│   ├── generators/            # Generadores de datos sintéticos
│   ├── filters/               # Filtros de calidad
│   └── queue/                 # Cola de datos para training
│
├── training/
│   ├── distributed/           # Training distribuido (MLX + PyTorch)
│   ├── model/                 # Arquitectura del modelo
│   └── evaluation/            # Benchmarks y evaluación
│
├── orchestrator/              # Orquestación del ciclo completo
│
├── scripts/                   # Scripts de utilidad
│
├── tests/                     # Tests
│
└── docs/                      # Documentación extendida
```

---

## Dominios de Ejemplo

| Dominio | Descripción | Config |
|---------|-------------|--------|
| `unreal_engine` | Experto en desarrollo de videojuegos con UE5 | [Ver](config/domains/unreal_engine.yaml) |
| `python` | Experto en programación Python | Próximamente |
| `chemistry` | Experto en química universitaria | Próximamente |

---

## Guías

- [Configurar un cluster de Macs](docs/mac_cluster_setup.md)
- [Añadir GPUs NVIDIA al cluster](docs/nvidia_setup.md)
- [Crear un nuevo dominio](docs/new_domain.md)
- [Entender el ciclo de auto-mejora](docs/self_improvement.md)

---

## Contribuir

¡Las contribuciones son bienvenidas! Ver [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Licencia

MIT License - ver [LICENSE](LICENSE)

---

## Créditos

- [MLX](https://github.com/ml-explore/mlx) - Framework de ML para Apple Silicon
- [Exo](https://github.com/exo-explore/exo) - Inspiración para networking distribuido
- [mlx-lm](https://github.com/ml-explore/mlx-lm) - Fine-tuning con MLX
