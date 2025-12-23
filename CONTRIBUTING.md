# Contribuir a Jupiter

¡Gracias por tu interés en contribuir a Jupiter!

## Cómo Contribuir

### Reportar Bugs

1. Verifica que el bug no haya sido reportado antes
2. Crea un issue con:
   - Descripción clara del problema
   - Pasos para reproducir
   - Comportamiento esperado vs actual
   - Información del sistema (macOS/Linux, chip, RAM)

### Proponer Features

1. Abre un issue describiendo la feature
2. Discute el diseño antes de implementar
3. Referencia el issue en tu PR

### Pull Requests

1. Fork el repositorio
2. Crea una branch: `git checkout -b feature/mi-feature`
3. Haz tus cambios
4. Añade tests si aplica
5. Asegúrate de que pasan los tests: `pytest`
6. Formatea el código: `black . && ruff check --fix .`
7. Commit: `git commit -m "feat: descripción"`
8. Push: `git push origin feature/mi-feature`
9. Abre un Pull Request

### Estilo de Código

- Usamos [Black](https://black.readthedocs.io/) para formateo
- Usamos [Ruff](https://docs.astral.sh/ruff/) para linting
- Líneas de máximo 100 caracteres
- Docstrings en español o inglés

### Commits

Usamos [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` nueva funcionalidad
- `fix:` corrección de bug
- `docs:` documentación
- `refactor:` refactorización
- `test:` tests
- `chore:` mantenimiento

## Áreas de Contribución

### Alta Prioridad

- [ ] Tests automatizados
- [ ] Implementación PyTorch del modelo
- [ ] Más dominios de ejemplo
- [ ] Optimizaciones de rendimiento

### Ideas

- Soporte para más backends de tokenizer
- Visualización de métricas en tiempo real
- Integración con Weights & Biases
- Exportación a GGUF para llama.cpp

## Desarrollo Local

```bash
# Clonar
git clone https://github.com/raym33/jupiter.git
cd jupiter

# Instalar en modo desarrollo
pip install -e ".[dev]"

# Ejecutar tests
pytest

# Formatear código
black .
ruff check --fix .
```

## Preguntas

Abre un issue con la etiqueta `question`.

¡Gracias por contribuir! 🚀
