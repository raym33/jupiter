"""
Deduplicador de datos de entrenamiento.

Utiliza múltiples estrategias:
- Hash exacto
- MinHash para similitud aproximada
- N-gram overlap
"""

from dataclasses import dataclass
from typing import Optional
import hashlib
import re


@dataclass
class DedupResult:
    """Resultado de deduplicación."""

    is_duplicate: bool
    duplicate_of: Optional[str] = None  # Hash del duplicado
    similarity: float = 0.0
    method: str = ""  # "exact", "minhash", "ngram"


class Deduplicator:
    """
    Deduplicador de documentos usando múltiples estrategias.

    Estrategias:
    1. Hash exacto: detecta copias exactas
    2. MinHash: detecta documentos muy similares (~85%+)
    3. N-gram: detecta overlap significativo
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        num_hashes: int = 100,
        ngram_size: int = 5,
    ):
        """
        Args:
            similarity_threshold: Umbral de similitud para considerar duplicado
            num_hashes: Número de hashes para MinHash
            ngram_size: Tamaño de n-gramas
        """
        self.similarity_threshold = similarity_threshold
        self.num_hashes = num_hashes
        self.ngram_size = ngram_size

        # Almacenamiento de hashes
        self._exact_hashes: set[str] = set()
        self._minhash_signatures: dict[str, list[int]] = {}

        # Índice invertido para n-gramas (para búsqueda rápida)
        self._ngram_index: dict[str, set[str]] = {}

    def _compute_exact_hash(self, content: str) -> str:
        """Calcula hash exacto del contenido."""
        # Normalizar: lowercase, remover espacios extra
        normalized = re.sub(r"\s+", " ", content.lower().strip())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _get_ngrams(self, content: str) -> set[str]:
        """Extrae n-gramas del contenido."""
        # Normalizar
        normalized = re.sub(r"\s+", " ", content.lower().strip())
        words = normalized.split()

        if len(words) < self.ngram_size:
            return {normalized}

        ngrams = set()
        for i in range(len(words) - self.ngram_size + 1):
            ngram = " ".join(words[i : i + self.ngram_size])
            ngrams.add(ngram)

        return ngrams

    def _compute_minhash(self, ngrams: set[str]) -> list[int]:
        """
        Calcula la firma MinHash.

        Usa múltiples funciones hash para crear una firma que
        permite estimar similitud de Jaccard eficientemente.
        """
        if not ngrams:
            return [0] * self.num_hashes

        signature = []

        for seed in range(self.num_hashes):
            min_hash = float("inf")

            for ngram in ngrams:
                # Hash con seed diferente
                h = hashlib.md5(f"{seed}:{ngram}".encode()).hexdigest()
                hash_val = int(h[:8], 16)

                if hash_val < min_hash:
                    min_hash = hash_val

            signature.append(min_hash if min_hash != float("inf") else 0)

        return signature

    def _estimate_similarity(self, sig1: list[int], sig2: list[int]) -> float:
        """Estima similitud de Jaccard usando firmas MinHash."""
        if len(sig1) != len(sig2):
            return 0.0

        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)

    def _compute_ngram_overlap(self, ngrams1: set[str], ngrams2: set[str]) -> float:
        """Calcula overlap de n-gramas (Jaccard)."""
        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)

        return intersection / union if union > 0 else 0.0

    def check(self, content: str) -> DedupResult:
        """
        Verifica si un contenido es duplicado.

        Args:
            content: Texto a verificar

        Returns:
            DedupResult con el resultado
        """
        # 1. Check exacto
        exact_hash = self._compute_exact_hash(content)
        if exact_hash in self._exact_hashes:
            return DedupResult(
                is_duplicate=True,
                duplicate_of=exact_hash,
                similarity=1.0,
                method="exact",
            )

        # 2. Calcular n-gramas y MinHash
        ngrams = self._get_ngrams(content)
        minhash = self._compute_minhash(ngrams)

        # 3. Check MinHash contra existentes
        for stored_hash, stored_sig in self._minhash_signatures.items():
            similarity = self._estimate_similarity(minhash, stored_sig)

            if similarity >= self.similarity_threshold:
                return DedupResult(
                    is_duplicate=True,
                    duplicate_of=stored_hash,
                    similarity=similarity,
                    method="minhash",
                )

        # 4. Check n-gram overlap para candidatos cercanos
        # (usar índice invertido para eficiencia)
        candidate_hashes: dict[str, int] = {}

        for ngram in list(ngrams)[:50]:  # Limitar para eficiencia
            if ngram in self._ngram_index:
                for doc_hash in self._ngram_index[ngram]:
                    candidate_hashes[doc_hash] = candidate_hashes.get(doc_hash, 0) + 1

        # Verificar candidatos con muchos n-gramas en común
        for doc_hash, count in candidate_hashes.items():
            if count >= 5:  # Al menos 5 n-gramas en común
                # Obtener n-gramas del documento original (simplificado)
                stored_sig = self._minhash_signatures.get(doc_hash)
                if stored_sig:
                    similarity = self._estimate_similarity(minhash, stored_sig)
                    if similarity >= self.similarity_threshold:
                        return DedupResult(
                            is_duplicate=True,
                            duplicate_of=doc_hash,
                            similarity=similarity,
                            method="ngram",
                        )

        # No es duplicado
        return DedupResult(is_duplicate=False, similarity=0.0)

    def add(self, content: str) -> str:
        """
        Añade un documento al índice de deduplicación.

        Args:
            content: Texto a añadir

        Returns:
            Hash del documento añadido
        """
        exact_hash = self._compute_exact_hash(content)

        # Añadir hash exacto
        self._exact_hashes.add(exact_hash)

        # Calcular y almacenar MinHash
        ngrams = self._get_ngrams(content)
        minhash = self._compute_minhash(ngrams)
        self._minhash_signatures[exact_hash] = minhash

        # Actualizar índice invertido (solo primeros 100 n-gramas)
        for ngram in list(ngrams)[:100]:
            if ngram not in self._ngram_index:
                self._ngram_index[ngram] = set()
            self._ngram_index[ngram].add(exact_hash)

        return exact_hash

    def check_and_add(self, content: str) -> tuple[DedupResult, Optional[str]]:
        """
        Verifica y añade si no es duplicado.

        Args:
            content: Texto a verificar/añadir

        Returns:
            (resultado, hash si se añadió o None)
        """
        result = self.check(content)

        if not result.is_duplicate:
            doc_hash = self.add(content)
            return result, doc_hash

        return result, None

    @property
    def document_count(self) -> int:
        """Número de documentos en el índice."""
        return len(self._exact_hashes)

    def clear(self) -> None:
        """Limpia todos los índices."""
        self._exact_hashes.clear()
        self._minhash_signatures.clear()
        self._ngram_index.clear()
