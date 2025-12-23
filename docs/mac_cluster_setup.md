# Configuración de un Cluster de Macs

Esta guía explica cómo configurar un cluster de Macs con Apple Silicon para training distribuido con Jupiter.

## Requisitos de Hardware

### Mínimo
- 2 Macs con Apple Silicon (M1/M2/M3/M4/M5)
- 16GB RAM por Mac
- Conexión Thunderbolt o Ethernet entre Macs

### Recomendado
- 4-5 Macs con M4 o superior
- 16-24GB RAM por Mac
- Conexión Thunderbolt 4/5 para máximo rendimiento
- 1 Mac dedicado como generador (mayor RAM)

## Paso 1: Preparar cada Mac

### 1.1 Instalar dependencias del sistema

En **cada Mac** del cluster:

```bash
# Instalar Xcode Command Line Tools
xcode-select --install

# Instalar Homebrew (si no está instalado)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Instalar MPI
brew install mpich
```

### 1.2 Instalar Jupiter

```bash
# Clonar el repositorio
git clone https://github.com/raym33/jupiter.git
cd jupiter

# Crear entorno virtual
python3.11 -m venv venv
source venv/bin/activate

# Instalar Jupiter con dependencias para Mac
pip install -e ".[mac]"
```

### 1.3 Verificar MLX

```bash
python -c "import mlx.core as mx; print(f'MLX version: {mx.__version__}')"
```

## Paso 2: Configurar Conectividad

### Opción A: Thunderbolt (Recomendado)

Thunderbolt ofrece 40Gb/s (TB4) o 80Gb/s (TB5), mucho más rápido que Ethernet.

#### Topología Daisy-Chain

```
MacBook ──TB──► Mac Mini 1 ──TB──► Mac Mini 2 ──TB──► Mac Mini 3 ──TB──► Mac Mini 4
```

1. Conectar Macs en cadena con cables Thunderbolt
2. macOS detectará automáticamente las conexiones
3. Verificar en Preferencias del Sistema > Red

#### Configurar Thunderbolt Bridge

En cada par de Macs conectados:

1. Abrir **Preferencias del Sistema > Red**
2. Seleccionar la conexión **Thunderbolt Bridge**
3. Configurar IP manualmente:
   - Mac principal: `169.254.1.1`
   - Mac Mini 1: `169.254.1.2`
   - Mac Mini 2: `169.254.1.3`
   - etc.

### Opción B: Ethernet

Si no puedes usar Thunderbolt:

1. Conectar todos los Macs al mismo switch Ethernet
2. Usar IPs estáticas en la misma subred:
   - `192.168.1.10`, `192.168.1.11`, etc.

## Paso 3: Configurar SSH

Para que MPI pueda lanzar procesos en los nodos remotos, necesitas SSH sin contraseña.

### En el Mac principal (nodo líder):

```bash
# Generar clave SSH si no existe
ssh-keygen -t ed25519

# Copiar clave a cada nodo
ssh-copy-id usuario@mac-mini-1.local
ssh-copy-id usuario@mac-mini-2.local
ssh-copy-id usuario@mac-mini-3.local
ssh-copy-id usuario@mac-mini-4.local
```

### Verificar conexión:

```bash
# Debería conectar sin pedir contraseña
ssh mac-mini-1.local "echo 'Conexión OK'"
```

## Paso 4: Configurar el Cluster en Jupiter

Editar `config/cluster.yaml`:

```yaml
backend: "mpi"
use_thunderbolt_bridge: true

nodes:
  - host: "macbook.local"
    name: "MacBook Principal"
    device_type: "mac"
    memory_gb: 24
    chip: "M4"
    role: "generator"
    connection_type: "thunderbolt"

  - host: "mac-mini-1.local"
    name: "Mac Mini 1"
    device_type: "mac"
    memory_gb: 16
    chip: "M4"
    role: "trainer"
    connection_type: "thunderbolt"

  # ... más nodos
```

## Paso 5: Verificar el Cluster

```bash
# En el Mac principal
cd jupiter
source venv/bin/activate

# Verificar configuración
jupiter check
```

Deberías ver algo como:

```
Archivos de Configuración
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Archivo          ┃ Estado                              ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ cluster.yaml     │ ✓ Encontrado                        │
│ Dominios         │ ✓ 1 encontrados: unreal_engine      │
└──────────────────┴─────────────────────────────────────┘

Verificando cluster...
✓ Configuración del cluster válida

Nodos del Cluster
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┓
┃ Nombre             ┃ Host                 ┃ Tipo  ┃ RAM   ┃ Rol       ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━━━━┩
│ MacBook Principal  │ macbook.local        │ mac   │ 24GB  │ generator │
│ Mac Mini 1         │ mac-mini-1.local     │ mac   │ 16GB  │ trainer   │
│ Mac Mini 2         │ mac-mini-2.local     │ mac   │ 16GB  │ trainer   │
│ Mac Mini 3         │ mac-mini-3.local     │ mac   │ 16GB  │ trainer   │
│ Mac Mini 4         │ mac-mini-4.local     │ mac   │ 16GB  │ trainer   │
└────────────────────┴──────────────────────┴───────┴───────┴───────────┘

Memoria total: 88GB
```

## Paso 6: Probar Training Distribuido

```bash
# Test rápido con modelo pequeño
jupiter start --domain unreal_engine --epochs 1
```

## Solución de Problemas

### "Connection refused" en SSH

```bash
# Habilitar Remote Login en el Mac destino
# Preferencias del Sistema > Compartir > Remote Login
```

### "MLX not found" en nodos remotos

Asegúrate de que el entorno virtual esté activado en todos los nodos:

```bash
# Añadir a ~/.bashrc o ~/.zshrc
export PATH="$HOME/jupiter/venv/bin:$PATH"
```

### Rendimiento lento

1. Verificar que estás usando Thunderbolt, no Ethernet
2. Verificar que no hay otras apps usando la red
3. Considerar aumentar `sync_every_n_steps` si la red es lenta

## Siguientes Pasos

- [Crear un nuevo dominio](new_domain.md)
- [Añadir GPUs NVIDIA](nvidia_setup.md)
- [Entender el ciclo de auto-mejora](self_improvement.md)
