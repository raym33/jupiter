"""
Tokenizer para Jupiter.

Usa SentencePiece/tiktoken como backend.
Puede entrenar un tokenizer custom o usar uno existente.
"""

from pathlib import Path
from typing import Optional, Union
import json


class JupiterTokenizer:
    """
    Tokenizer para el modelo Jupiter.

    Soporta:
    - Cargar tokenizer existente (Llama, GPT, etc.)
    - Entrenar tokenizer custom con SentencePiece
    - Tokenización y detokenización
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        model_path: Optional[str] = None,
    ):
        """
        Args:
            vocab_size: Tamaño del vocabulario
            model_path: Path a un tokenizer existente
        """
        self.vocab_size = vocab_size
        self._tokenizer = None
        self._backend = None

        if model_path:
            self.load(model_path)

    def load(self, path: str) -> None:
        """
        Carga un tokenizer desde un path.

        Detecta automáticamente el tipo de tokenizer.
        """
        path = Path(path)

        # Intentar cargar como tokenizer de HuggingFace
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(path))
            self._backend = "transformers"
            self.vocab_size = len(self._tokenizer)
            return
        except Exception:
            pass

        # Intentar cargar como SentencePiece
        try:
            import sentencepiece as spm

            sp = spm.SentencePieceProcessor()
            sp.load(str(path / "tokenizer.model") if path.is_dir() else str(path))
            self._tokenizer = sp
            self._backend = "sentencepiece"
            self.vocab_size = sp.get_piece_size()
            return
        except Exception:
            pass

        raise ValueError(f"No se pudo cargar el tokenizer desde {path}")

    def load_from_pretrained(self, model_name: str) -> None:
        """
        Carga un tokenizer pretrained de HuggingFace.

        Args:
            model_name: Nombre del modelo (ej: "meta-llama/Llama-2-7b-hf")
        """
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._backend = "transformers"
        self.vocab_size = len(self._tokenizer)

    def train(
        self,
        texts: list[str],
        vocab_size: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> None:
        """
        Entrena un tokenizer custom usando SentencePiece.

        Args:
            texts: Lista de textos para entrenar
            vocab_size: Tamaño del vocabulario
            output_path: Path donde guardar el modelo
        """
        import sentencepiece as spm
        import tempfile

        vocab_size = vocab_size or self.vocab_size

        # Escribir textos a archivo temporal
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for text in texts:
                f.write(text + "\n")
            input_file = f.name

        # Configurar output
        if output_path:
            model_prefix = str(Path(output_path) / "tokenizer")
        else:
            model_prefix = tempfile.mktemp()

        # Entrenar
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=0.9995,
            num_threads=8,
            split_digits=True,
            byte_fallback=True,
            # Tokens especiales
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
        )

        # Cargar modelo entrenado
        self._tokenizer = spm.SentencePieceProcessor()
        self._tokenizer.load(f"{model_prefix}.model")
        self._backend = "sentencepiece"
        self.vocab_size = self._tokenizer.get_piece_size()

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
    ) -> list[int]:
        """
        Tokeniza un texto.

        Args:
            text: Texto a tokenizar
            add_special_tokens: Añadir tokens especiales (BOS, EOS)
            max_length: Longitud máxima
            padding: Añadir padding hasta max_length

        Returns:
            Lista de token IDs
        """
        if self._tokenizer is None:
            raise ValueError("Tokenizer no cargado. Usa load() o train() primero.")

        if self._backend == "transformers":
            tokens = self._tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
            )
        elif self._backend == "sentencepiece":
            tokens = self._tokenizer.encode(text)
            if add_special_tokens:
                bos_id = self._tokenizer.bos_id()
                eos_id = self._tokenizer.eos_id()
                if bos_id >= 0:
                    tokens = [bos_id] + tokens
                if eos_id >= 0:
                    tokens = tokens + [eos_id]
        else:
            raise ValueError(f"Backend no soportado: {self._backend}")

        # Truncar si es necesario
        if max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]

        # Padding si es necesario
        if padding and max_length and len(tokens) < max_length:
            pad_id = self.pad_token_id
            tokens = tokens + [pad_id] * (max_length - len(tokens))

        return tokens

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decodifica tokens a texto.

        Args:
            token_ids: Lista de token IDs
            skip_special_tokens: Omitir tokens especiales

        Returns:
            Texto decodificado
        """
        if self._tokenizer is None:
            raise ValueError("Tokenizer no cargado.")

        if self._backend == "transformers":
            return self._tokenizer.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
            )
        elif self._backend == "sentencepiece":
            return self._tokenizer.decode(token_ids)
        else:
            raise ValueError(f"Backend no soportado: {self._backend}")

    def encode_batch(
        self,
        texts: list[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
    ) -> list[list[int]]:
        """
        Tokeniza múltiples textos.

        Args:
            texts: Lista de textos
            add_special_tokens: Añadir tokens especiales
            max_length: Longitud máxima
            padding: Añadir padding para igualar longitudes

        Returns:
            Lista de listas de token IDs
        """
        encoded = [
            self.encode(text, add_special_tokens, max_length, padding=False)
            for text in texts
        ]

        if padding:
            # Encontrar longitud máxima
            max_len = max(len(e) for e in encoded)
            if max_length:
                max_len = min(max_len, max_length)

            # Añadir padding
            pad_id = self.pad_token_id
            encoded = [
                e[:max_len] + [pad_id] * (max_len - len(e))
                for e in encoded
            ]

        return encoded

    @property
    def pad_token_id(self) -> int:
        """ID del token de padding."""
        if self._backend == "transformers":
            return self._tokenizer.pad_token_id or 0
        elif self._backend == "sentencepiece":
            return self._tokenizer.pad_id()
        return 0

    @property
    def bos_token_id(self) -> int:
        """ID del token de inicio."""
        if self._backend == "transformers":
            return self._tokenizer.bos_token_id or 1
        elif self._backend == "sentencepiece":
            return self._tokenizer.bos_id()
        return 1

    @property
    def eos_token_id(self) -> int:
        """ID del token de fin."""
        if self._backend == "transformers":
            return self._tokenizer.eos_token_id or 2
        elif self._backend == "sentencepiece":
            return self._tokenizer.eos_id()
        return 2

    def save(self, path: str) -> None:
        """
        Guarda el tokenizer.

        Args:
            path: Directorio donde guardar
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self._backend == "transformers":
            self._tokenizer.save_pretrained(str(path))
        elif self._backend == "sentencepiece":
            # Copiar archivos del modelo
            import shutil

            model_file = self._tokenizer.serialized_model_proto()
            with open(path / "tokenizer.model", "wb") as f:
                f.write(model_file)

        # Guardar metadata
        metadata = {
            "vocab_size": self.vocab_size,
            "backend": self._backend,
        }
        with open(path / "tokenizer_config.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def __len__(self) -> int:
        """Tamaño del vocabulario."""
        return self.vocab_size

    def __repr__(self) -> str:
        return f"JupiterTokenizer(vocab_size={self.vocab_size}, backend='{self._backend}')"
