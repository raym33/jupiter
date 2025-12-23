"""
Generador de datos sintéticos usando LLMs locales.

Soporta:
- MLX (Apple Silicon)
- PyTorch/Transformers (NVIDIA/CPU)
"""

import asyncio
from pathlib import Path
from typing import AsyncIterator, Optional
import platform

from jupiter.data.generators.base import SyntheticGenerator, GeneratedSample


class LLMGenerator(SyntheticGenerator):
    """
    Generador de datos sintéticos usando LLMs locales.

    Detecta automáticamente el backend disponible:
    - Mac con Apple Silicon: usa MLX
    - Linux/Windows con NVIDIA: usa PyTorch
    - Fallback: usa CPU con PyTorch
    """

    def __init__(
        self,
        output_dir: Path,
        model_name: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
        max_samples: int = 100000,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        backend: str = "auto",  # "auto", "mlx", "torch"
    ):
        super().__init__(
            output_dir=output_dir,
            model_name=model_name,
            max_samples=max_samples,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        self.backend = self._detect_backend() if backend == "auto" else backend
        self._model = None
        self._tokenizer = None

    def _detect_backend(self) -> str:
        """Detecta el mejor backend disponible."""
        system = platform.system()

        if system == "Darwin":
            # macOS - verificar si hay MLX disponible
            try:
                import mlx.core  # noqa: F401

                return "mlx"
            except ImportError:
                pass

        # Verificar CUDA
        try:
            import torch

            if torch.cuda.is_available():
                return "torch-cuda"
        except ImportError:
            pass

        # Fallback a CPU
        return "torch-cpu"

    async def load_model(self) -> None:
        """Carga el modelo según el backend detectado."""
        print(f"Cargando modelo {self.model_name} con backend {self.backend}...")

        if self.backend == "mlx":
            await self._load_mlx_model()
        else:
            await self._load_torch_model()

        print("Modelo cargado.")

    async def _load_mlx_model(self) -> None:
        """Carga el modelo usando MLX."""
        # Ejecutar en thread pool para no bloquear
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_mlx_model_sync)

    def _load_mlx_model_sync(self) -> None:
        """Carga sincrónica del modelo MLX."""
        from mlx_lm import load

        self._model, self._tokenizer = load(self.model_name)

    async def _load_torch_model(self) -> None:
        """Carga el modelo usando PyTorch/Transformers."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_torch_model_sync)

    def _load_torch_model_sync(self) -> None:
        """Carga sincrónica del modelo PyTorch."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        # Determinar device
        if self.backend == "torch-cuda":
            device = "cuda"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32

        # Para PyTorch, usar el modelo sin cuantizar de HF
        # Mapear nombre MLX a nombre HF si es necesario
        model_name = self.model_name
        if "mlx-community" in model_name:
            # Intentar convertir a nombre HF equivalente
            model_name = model_name.replace("mlx-community/", "")
            model_name = model_name.replace("-4bit", "")
            model_name = model_name.replace("-8bit", "")
            model_name = f"meta-llama/{model_name}"

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
        )

    async def unload_model(self) -> None:
        """Descarga el modelo de memoria."""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None

            # Liberar memoria
            if self.backend == "mlx":
                import mlx.core as mx
                mx.metal.clear_cache()
            elif "torch" in self.backend:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _format_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Formatea el prompt para el modelo.

        Usa formato de chat si el modelo lo soporta.
        """
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        # Usar el tokenizer para aplicar chat template si existe
        if hasattr(self._tokenizer, "apply_chat_template"):
            formatted = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return formatted
        else:
            # Fallback simple
            if system_prompt:
                return f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            return f"User: {prompt}\n\nAssistant:"

    async def generate(
        self,
        template_type: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> Optional[GeneratedSample]:
        """
        Genera una muestra sintética.
        """
        if self._model is None:
            await self.load_model()

        # Reemplazar {topic} en el prompt si existe
        if topic and "{topic}" in prompt:
            prompt = prompt.replace("{topic}", topic)

        formatted_prompt = self._format_prompt(prompt, system_prompt)

        try:
            if self.backend == "mlx":
                content = await self._generate_mlx(formatted_prompt)
            else:
                content = await self._generate_torch(formatted_prompt)

            if not content or len(content) < 50:
                return None

            # Verificar duplicado
            if self.is_duplicate(content):
                return None

            sample = GeneratedSample(
                content=content,
                prompt_used=prompt,
                template_type=template_type,
                topic=topic,
                generator_model=self.model_name,
            )

            return sample

        except Exception as e:
            print(f"Error generando: {e}")
            return None

    async def _generate_mlx(self, prompt: str) -> str:
        """Genera usando MLX."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate_mlx_sync, prompt)

    def _generate_mlx_sync(self, prompt: str) -> str:
        """Generación sincrónica con MLX."""
        from mlx_lm import generate

        response = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temp=self.temperature,
            top_p=self.top_p,
        )

        return response

    async def _generate_torch(self, prompt: str) -> str:
        """Genera usando PyTorch."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate_torch_sync, prompt)

    def _generate_torch_sync(self, prompt: str) -> str:
        """Generación sincrónica con PyTorch."""
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt")
        if "torch-cuda" in self.backend:
            inputs = inputs.to("cuda")

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decodificar solo los tokens nuevos
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response

    async def generate_batch(
        self,
        template_type: str,
        prompts: list[str],
        system_prompt: Optional[str] = None,
        topics: Optional[list[str]] = None,
    ) -> AsyncIterator[GeneratedSample]:
        """
        Genera múltiples muestras.
        """
        if self._model is None:
            await self.load_model()

        topics = topics or [None] * len(prompts)

        for prompt, topic in zip(prompts, topics):
            if not self.has_capacity:
                break

            sample = await self.generate(
                template_type=template_type,
                prompt=prompt,
                system_prompt=system_prompt,
                topic=topic,
            )

            if sample:
                if await self.save_sample(sample):
                    yield sample

    async def generate_from_template(
        self,
        template: "GenerationTemplate",  # noqa: F821
        num_samples: int = 10,
    ) -> AsyncIterator[GeneratedSample]:
        """
        Genera muestras a partir de un template de dominio.

        Args:
            template: Template de generación del dominio
            num_samples: Número de muestras a generar

        Yields:
            GeneratedSample para cada generación exitosa
        """
        # Determinar qué items iterar (topics o tasks)
        items = template.topics if template.topics else template.tasks

        if not items:
            items = ["general"]

        samples_per_item = max(1, num_samples // len(items))

        for item in items:
            for _ in range(samples_per_item):
                if not self.has_capacity:
                    return

                # Reemplazar placeholders en el prompt
                prompt = template.prompt
                if "{topic}" in prompt:
                    prompt = prompt.replace("{topic}", item)
                if "{task}" in prompt:
                    prompt = prompt.replace("{task}", item)

                sample = await self.generate(
                    template_type=template.type.value,
                    prompt=prompt,
                    system_prompt=template.system_prompt,
                    topic=item,
                )

                if sample:
                    if await self.save_sample(sample):
                        yield sample
