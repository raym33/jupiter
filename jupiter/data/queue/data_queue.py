"""
Cola de datos para alimentar el training distribuido.

Características:
- Mezcla datos de diferentes fuentes según proporciones configuradas
- Buffer en disco para manejar grandes volúmenes
- Soporte para múltiples workers consumiendo datos
- Shuffling y batching
"""

import asyncio
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional
import aiofiles


@dataclass
class DataBatch:
    """
    Batch de datos para training.
    """

    samples: list[dict]
    batch_id: int
    source_mix: dict[str, int]  # Conteo por tipo de fuente

    @property
    def size(self) -> int:
        """Número de muestras en el batch."""
        return len(self.samples)

    def to_texts(self) -> list[str]:
        """Extrae solo los textos de las muestras."""
        return [s.get("content", s.get("text", "")) for s in self.samples]


@dataclass
class DataQueue:
    """
    Cola de datos para training distribuido.

    Maneja la mezcla de datos de diferentes fuentes (reales, sintéticos)
    y proporciona batches balanceados para el training.
    """

    data_dir: Path
    batch_size: int = 32
    shuffle_buffer_size: int = 10000
    mix_ratios: dict[str, float] = field(default_factory=dict)

    # Estado interno
    _samples: list[dict] = field(default_factory=list, repr=False)
    _batch_counter: int = 0
    _sources: dict[str, list[dict]] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Inicializa la cola."""
        self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Ratios por defecto
        if not self.mix_ratios:
            self.mix_ratios = {
                "real": 0.4,
                "synthetic": 0.5,
                "fresh": 0.1,
            }

    async def load_from_directories(
        self,
        real_dir: Optional[Path] = None,
        synthetic_dir: Optional[Path] = None,
        fresh_dir: Optional[Path] = None,
    ) -> int:
        """
        Carga datos desde directorios.

        Args:
            real_dir: Directorio con datos reales
            synthetic_dir: Directorio con datos sintéticos
            fresh_dir: Directorio con datos frescos (scraping reciente)

        Returns:
            Número total de muestras cargadas
        """
        total = 0

        if real_dir and real_dir.exists():
            count = await self._load_directory(real_dir, "real")
            total += count
            print(f"Cargados {count} documentos reales")

        if synthetic_dir and synthetic_dir.exists():
            count = await self._load_directory(synthetic_dir, "synthetic")
            total += count
            print(f"Cargados {count} documentos sintéticos")

        if fresh_dir and fresh_dir.exists():
            count = await self._load_directory(fresh_dir, "fresh")
            total += count
            print(f"Cargados {count} documentos frescos")

        return total

    async def _load_directory(self, directory: Path, source_type: str) -> int:
        """Carga todos los JSON de un directorio."""
        if source_type not in self._sources:
            self._sources[source_type] = []

        count = 0
        for file in directory.glob("*.json"):
            try:
                async with aiofiles.open(file, "r") as f:
                    data = json.loads(await f.read())

                # Añadir metadata de source
                data["_source_type"] = source_type
                data["_source_file"] = str(file)

                self._sources[source_type].append(data)
                count += 1

            except Exception as e:
                print(f"Error cargando {file}: {e}")

        return count

    def _mix_samples(self, count: int) -> list[dict]:
        """
        Mezcla muestras de diferentes fuentes según los ratios.

        Args:
            count: Número total de muestras a mezclar

        Returns:
            Lista de muestras mezcladas
        """
        mixed = []

        for source_type, ratio in self.mix_ratios.items():
            if source_type not in self._sources:
                continue

            source_samples = self._sources[source_type]
            if not source_samples:
                continue

            # Calcular cuántas muestras de esta fuente
            num_samples = int(count * ratio)

            # Samplear con reemplazo si no hay suficientes
            if num_samples > 0:
                if len(source_samples) >= num_samples:
                    selected = random.sample(source_samples, num_samples)
                else:
                    selected = random.choices(source_samples, k=num_samples)

                mixed.extend(selected)

        # Shuffle final
        random.shuffle(mixed)

        return mixed[:count]

    def prepare_epoch(self, num_samples: Optional[int] = None) -> int:
        """
        Prepara las muestras para una época de training.

        Args:
            num_samples: Número de muestras (None = todas las disponibles)

        Returns:
            Número de muestras preparadas
        """
        # Calcular total disponible
        total_available = sum(len(samples) for samples in self._sources.values())

        if num_samples is None:
            num_samples = total_available

        # Mezclar
        self._samples = self._mix_samples(num_samples)
        self._batch_counter = 0

        print(f"Preparadas {len(self._samples)} muestras para la época")
        return len(self._samples)

    def get_batch(self) -> Optional[DataBatch]:
        """
        Obtiene el siguiente batch.

        Returns:
            DataBatch o None si no hay más datos
        """
        start_idx = self._batch_counter * self.batch_size
        end_idx = start_idx + self.batch_size

        if start_idx >= len(self._samples):
            return None

        batch_samples = self._samples[start_idx:end_idx]

        # Contar fuentes en el batch
        source_mix = {}
        for sample in batch_samples:
            source = sample.get("_source_type", "unknown")
            source_mix[source] = source_mix.get(source, 0) + 1

        batch = DataBatch(
            samples=batch_samples,
            batch_id=self._batch_counter,
            source_mix=source_mix,
        )

        self._batch_counter += 1
        return batch

    def iter_batches(self) -> Iterator[DataBatch]:
        """
        Itera sobre todos los batches de la época.

        Yields:
            DataBatch para cada batch
        """
        while True:
            batch = self.get_batch()
            if batch is None:
                break
            yield batch

    @property
    def num_batches(self) -> int:
        """Número total de batches en la época."""
        return (len(self._samples) + self.batch_size - 1) // self.batch_size

    @property
    def total_samples(self) -> int:
        """Número total de muestras disponibles."""
        return sum(len(samples) for samples in self._sources.values())

    @property
    def samples_per_source(self) -> dict[str, int]:
        """Número de muestras por fuente."""
        return {source: len(samples) for source, samples in self._sources.items()}

    async def add_sample(self, sample: dict, source_type: str = "synthetic") -> None:
        """
        Añade una muestra a la cola.

        Args:
            sample: Muestra a añadir
            source_type: Tipo de fuente
        """
        sample["_source_type"] = source_type

        if source_type not in self._sources:
            self._sources[source_type] = []

        self._sources[source_type].append(sample)

    async def save_state(self, path: Path) -> None:
        """Guarda el estado de la cola."""
        state = {
            "batch_counter": self._batch_counter,
            "mix_ratios": self.mix_ratios,
            "samples_per_source": self.samples_per_source,
        }

        async with aiofiles.open(path, "w") as f:
            await f.write(json.dumps(state, indent=2))

    def clear(self) -> None:
        """Limpia la cola."""
        self._samples.clear()
        self._sources.clear()
        self._batch_counter = 0
