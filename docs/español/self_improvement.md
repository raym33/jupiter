# Ciclo de Auto-Mejora

Jupiter implementa un ciclo donde el modelo entrenado puede reemplazar al modelo generador cuando lo supera.

## Concepto

```
┌─────────────────────────────────────────────────────────────────┐
│                    CICLO DE AUTO-MEJORA                         │
│                                                                 │
│   ┌──────────────┐                                              │
│   │  GENERADOR   │ ◄─────────────────────────────────┐          │
│   │  (Llama 3B)  │                                   │          │
│   └──────┬───────┘                                   │          │
│          │                                           │          │
│          │ genera datos                              │          │
│          ▼                                           │          │
│   ┌──────────────┐                                   │          │
│   │    DATOS     │                                   │          │
│   │  SINTÉTICOS  │                                   │          │
│   └──────┬───────┘                                   │          │
│          │                                           │          │
│          │ + datos reales                            │          │
│          ▼                                           │          │
│   ┌──────────────┐                                   │          │
│   │  TRAINING    │                                   │          │
│   │  DISTRIBUIDO │                                   │          │
│   └──────┬───────┘                                   │          │
│          │                                           │          │
│          ▼                                           │          │
│   ┌──────────────┐      ┌──────────────┐            │          │
│   │   MODELO     │      │  EVALUACIÓN  │            │          │
│   │  ENTRENADO   │ ───► │              │            │          │
│   └──────────────┘      └──────┬───────┘            │          │
│                                │                     │          │
│                                │ ¿supera al          │          │
│                                │ generador?          │          │
│                                │                     │          │
│                         ┌──────┴──────┐              │          │
│                         │             │              │          │
│                    NO   ▼        SI   ▼              │          │
│                  continuar      REEMPLAZAR ──────────┘          │
│                  training       GENERADOR                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## ¿Por qué funciona?

### El modelo entrenado vs el generador

- **Generador (ej: Llama 3B)**: Conocimiento general, puede generar sobre cualquier tema
- **Modelo entrenado (1B)**: Especializado en el dominio específico

Aunque el modelo entrenado es más pequeño, puede superar al generador **en el dominio específico** porque:

1. Ha visto miles de ejemplos del dominio
2. Está optimizado para ese tipo de contenido
3. Ha aprendido patrones específicos del dominio

### Evitando Model Collapse

El riesgo de usar un modelo para generar datos de entrenamiento para sí mismo es el "model collapse": el modelo converge a un estado degenerado.

Jupiter lo evita con:

1. **Ancla de datos reales**: Siempre incluye 30-40% de datos reales (documentación, código)
2. **Filtrado de calidad**: Descarta generaciones de baja calidad
3. **Deduplicación**: Evita que el modelo memorice sus propias generaciones
4. **Evaluación rigurosa**: Solo reemplaza el generador si hay mejora real

## Configuración

En `config/cluster.yaml` o en código:

```yaml
self_improvement:
  # Evaluar cada N steps
  eval_every_steps: 5000

  # Umbral de mejora para reemplazar generador
  improvement_threshold: 0.05  # 5% mejor

  # Mínimo de steps antes de considerar reemplazo
  min_steps_before_replace: 10000

  # Muestras para benchmark
  benchmark_samples: 500

  # Seguridad
  keep_last_n_checkpoints: 3
  always_keep_base_generator: true  # Nunca borrar el Llama original
```

## Métricas de Evaluación

Jupiter evalúa el modelo en varias dimensiones:

### 1. Accuracy (40%)
- ¿Las respuestas son correctas?
- ¿El código compila/ejecuta?
- ¿Los hechos son precisos?

### 2. Coherence (30%)
- ¿Las respuestas tienen sentido?
- ¿Fluyen lógicamente?
- ¿Están bien estructuradas?

### 3. Domain Knowledge (30%)
- ¿Usa terminología correcta del dominio?
- ¿Demuestra conocimiento profundo?
- ¿Sigue las mejores prácticas del campo?

## El Proceso de Upgrade

Cuando el modelo entrenado supera al generador:

```
1. Guardar checkpoint del modelo entrenado
2. Convertir a formato de generador
3. Actualizar referencia del generador
4. Guardar generador anterior en historial
5. Regenerar datos sintéticos con nuevo generador
6. Continuar training con nuevos datos
```

### Historial de Generadores

Jupiter mantiene un historial:

```
generators/
├── generator_v1/  (Llama 3B original)
├── generator_v2/  (Tu modelo después de 10k steps)
├── generator_v3/  (Tu modelo después de 25k steps)
└── current -> generator_v3
```

### Rollback

Si algo sale mal, puedes volver a un generador anterior:

```python
from jupiter.orchestrator import SelfImprover

improver = SelfImprover(config, trainer)
await improver.rollback_generator()
```

## Ejemplo de Progreso

```
Época 1-5:
  Generador: Llama 3B
  Eval score: 0.65

Época 6-10:
  Generador: Llama 3B
  Eval score: 0.72 (+0.07)
  → No supera umbral de reemplazo

Época 11-15:
  Generador: Llama 3B
  Eval score: 0.78 (+0.13)
  → ¡UPGRADE! Modelo entrenado es ahora el generador

Época 16-20:
  Generador: Tu modelo v1
  Eval score: 0.81 (+0.03)
  → Datos generados son de mejor calidad

Época 21-25:
  Generador: Tu modelo v1
  Eval score: 0.85 (+0.07)
  → ¡UPGRADE! Tu modelo v2 es ahora el generador

...
```

## Visualizando el Progreso

Jupiter guarda métricas que puedes visualizar:

```python
import json
import matplotlib.pyplot as plt

with open("pipeline_state.json") as f:
    state = json.load(f)

# Historial de scores (hipotético)
epochs = range(1, state["current_epoch"] + 1)
scores = [...]  # Cargar de logs

plt.plot(epochs, scores)
plt.axhline(y=0.7, color='r', linestyle='--', label='Umbral generador original')
plt.xlabel("Época")
plt.ylabel("Eval Score")
plt.title("Progreso de Auto-Mejora")
plt.legend()
plt.show()
```

## Limitaciones

1. **No es magia**: El modelo no puede aprender lo que no está en los datos
2. **Ceiling effect**: Eventualmente alcanzarás un límite
3. **Especialización**: El modelo será muy bueno en el dominio pero puede olvidar conocimiento general

## Recomendaciones

1. **Empieza con buen generador**: Llama 3B o Mistral 7B
2. **Datos reales de calidad**: Son el ancla que evita el collapse
3. **Evalúa frecuentemente**: Detecta problemas temprano
4. **Guarda checkpoints**: Por si necesitas rollback
5. **Monitorea diversidad**: Si las generaciones se vuelven repetitivas, hay problema

## Próximos Pasos

- [Configurar cluster](mac_cluster_setup.md)
- [Crear nuevo dominio](new_domain.md)
