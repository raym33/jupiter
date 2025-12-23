# Crear un Nuevo Dominio

Esta guía explica cómo configurar un nuevo dominio de especialización para Jupiter.

## ¿Qué es un Dominio?

Un dominio define **qué tipo de experto** quieres entrenar. Incluye:

- **Fuentes de datos**: Dónde obtener información (docs, código, web)
- **Templates de generación**: Cómo crear datos sintéticos
- **Criterios de evaluación**: Cómo medir la calidad del modelo

## Paso 1: Crear el Archivo de Dominio

```bash
jupiter new-domain mi_dominio
```

Esto crea `config/domains/mi_dominio.yaml` desde el template.

## Paso 2: Configurar el Dominio

Edita el archivo YAML:

### Información Básica

```yaml
domain:
  name: "python_expert"
  description: "Experto en programación Python"
  language: "es"  # o "en"

  # Palabras clave para filtrar contenido relevante
  keywords:
    - "python"
    - "pip"
    - "django"
    - "flask"
    - "pandas"
    - "numpy"

  # Contenido a evitar
  negative_keywords:
    - "javascript"  # Otro lenguaje
    - "java"
```

### Fuentes de Datos

```yaml
data_sources:
  # Documentación oficial
  documentation:
    - "https://docs.python.org/3/"
    - "https://docs.djangoproject.com/"

  # Código de GitHub
  github:
    repos:
      - "python/cpython"
      - "django/django"
      - "pallets/flask"
    file_types:
      - ".py"
      - ".md"
      - ".rst"
    exclude_patterns:
      - "test_*"
      - "*_test.py"
      - "docs/conf.py"

  # Sitios web
  websites:
    - "https://realpython.com"
    - "https://www.fullstackpython.com"

  # Foros
  forums:
    - "https://discuss.python.org"
```

### Templates de Generación

Los templates definen cómo el LLM generará datos sintéticos:

```yaml
generation:
  templates:
    # Preguntas y Respuestas
    - type: "qa"
      system_prompt: |
        Eres un experto en Python con 15 años de experiencia.
        Respondes de forma clara y con ejemplos de código.
      prompt: |
        Genera una pregunta técnica sobre {topic} en Python
        y su respuesta detallada.

        Incluye:
        1. Explicación del concepto
        2. Código de ejemplo
        3. Errores comunes
        4. Mejores prácticas
      topics:
        - "decoradores"
        - "generadores"
        - "context managers"
        - "asyncio"
        - "type hints"
        - "dataclasses"

    # Tutoriales
    - type: "tutorial"
      system_prompt: |
        Eres un instructor de Python.
      prompt: |
        Crea un tutorial paso a paso para {task}.
      tasks:
        - "crear una API REST con FastAPI"
        - "web scraping con BeautifulSoup"
        - "análisis de datos con pandas"

    # Debugging
    - type: "debug"
      prompt: |
        Genera un error común en Python relacionado con {topic}.
        Incluye el traceback y la solución.
      topics:
        - "IndentationError"
        - "ImportError"
        - "AttributeError"
        - "TypeError"

  # Proporción de cada tipo
  template_weights:
    qa: 0.40
    tutorial: 0.35
    debug: 0.25
```

### Evaluación

```yaml
evaluation:
  benchmarks:
    - "code_syntax"      # ¿El código es válido?
    - "best_practices"   # ¿Sigue PEP8?
    - "completeness"     # ¿La respuesta es completa?

  min_accuracy: 0.75
  min_coherence: 0.80
```

### Proporción de Datos

```yaml
mix_ratio:
  real_docs: 0.30        # 30% documentación real
  real_code: 0.25        # 25% código real
  synthetic_qa: 0.20     # 20% Q&A generadas
  synthetic_tutorials: 0.15
  synthetic_debug: 0.10
```

## Ejemplos de Dominios

### Química Universitaria

```yaml
domain:
  name: "chemistry"
  description: "Experto en química universitaria"
  language: "es"

  keywords:
    - "átomo"
    - "molécula"
    - "enlace"
    - "reacción"
    - "equilibrio"
    - "termodinámica"

data_sources:
  documentation:
    - "https://chem.libretexts.org"

  websites:
    - "https://www.khanacademy.org/science/chemistry"

generation:
  templates:
    - type: "qa"
      prompt: |
        Genera una pregunta de examen universitario sobre {topic}
        con su respuesta detallada y explicación paso a paso.
      topics:
        - "estequiometría"
        - "equilibrio químico"
        - "cinética"
        - "termodinámica"
        - "electroquímica"

    - type: "explanation"
      prompt: |
        Explica el concepto de {topic} como si fueras un profesor.
        Incluye ejemplos y analogías.
      topics:
        - "orbitales atómicos"
        - "hibridación"
        - "resonancia"
```

### Medicina (Radiología)

```yaml
domain:
  name: "radiology"
  description: "Experto en interpretación radiológica"
  language: "es"

  keywords:
    - "radiografía"
    - "tomografía"
    - "resonancia"
    - "ecografía"
    - "diagnóstico"

generation:
  templates:
    - type: "qa"
      prompt: |
        Describe los hallazgos radiológicos típicos de {condition}
        y el diagnóstico diferencial.
      topics:
        - "neumonía"
        - "fractura de cadera"
        - "tumor cerebral"
        - "derrame pleural"
```

## Paso 3: Probar el Dominio

```bash
# Verificar que el dominio carga correctamente
jupiter check

# Ejecutar un test rápido
jupiter start --domain mi_dominio --epochs 1 --no-collect
```

## Consejos

1. **Empieza simple**: Pocos topics, pocas fuentes
2. **Itera**: Añade más fuentes y templates gradualmente
3. **Revisa la calidad**: Los datos sintéticos deben ser útiles
4. **Balancea**: Mantén proporción de datos reales vs sintéticos

## Siguientes Pasos

- [Configurar cluster de Macs](mac_cluster_setup.md)
- [Entender auto-mejora](self_improvement.md)
