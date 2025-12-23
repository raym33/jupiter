"""
Recolector de datos de GitHub.
"""

import asyncio
import base64
from pathlib import Path
from typing import AsyncIterator, Optional
import re

import httpx

from jupiter.data.collectors.base import DataCollector, CollectedDocument


class GitHubCollector(DataCollector):
    """
    Recolector de código y documentación de GitHub.

    Características:
    - Descarga archivos de repositorios públicos
    - Filtra por tipos de archivo
    - Extrae código con contexto
    - Respeta rate limits de GitHub API
    """

    # Rate limits de GitHub API (sin auth)
    RATE_LIMIT_DELAY = 1.0  # segundos entre requests

    def __init__(
        self,
        repos: list[str],
        output_dir: Path,
        file_types: list[str] = None,
        exclude_patterns: list[str] = None,
        domain_keywords: list[str] = None,
        negative_keywords: list[str] = None,
        max_documents: int = 10000,
        max_file_size_kb: int = 500,
        github_token: Optional[str] = None,
        language: str = "es",
    ):
        """
        Args:
            repos: Lista de repos en formato "owner/repo"
            file_types: Extensiones de archivo a incluir (ej: [".py", ".cpp"])
            exclude_patterns: Patrones a excluir (ej: ["test_*", "*_test.py"])
            max_file_size_kb: Tamaño máximo de archivo en KB
            github_token: Token de GitHub (opcional, aumenta rate limit)
        """
        super().__init__(
            output_dir=output_dir,
            domain_keywords=domain_keywords,
            negative_keywords=negative_keywords,
            max_documents=max_documents,
            language=language,
        )

        self.repos = repos
        self.file_types = file_types or [".py", ".cpp", ".h", ".md", ".txt"]
        self.exclude_patterns = exclude_patterns or []
        self.max_file_size_kb = max_file_size_kb
        self.github_token = github_token

        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Obtiene o crea el cliente HTTP."""
        if self._client is None:
            headers = {
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Jupiter-DataCollector/1.0",
            }
            if self.github_token:
                headers["Authorization"] = f"token {self.github_token}"

            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers=headers,
            )
        return self._client

    async def _close_client(self) -> None:
        """Cierra el cliente HTTP."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _should_include_file(self, path: str) -> bool:
        """Verifica si un archivo debe incluirse."""
        # Verificar extensión
        if self.file_types:
            has_valid_ext = any(path.endswith(ext) for ext in self.file_types)
            if not has_valid_ext:
                return False

        # Verificar patrones de exclusión
        for pattern in self.exclude_patterns:
            # Convertir glob pattern a regex simple
            regex_pattern = pattern.replace("*", ".*").replace("?", ".")
            if re.search(regex_pattern, path):
                return False

        return True

    def _detect_doc_type(self, path: str, content: str) -> str:
        """Detecta el tipo de documento."""
        ext = Path(path).suffix.lower()

        if ext in (".md", ".rst", ".txt"):
            return "documentation"
        elif ext in (".py", ".cpp", ".c", ".h", ".hpp", ".js", ".ts"):
            return "code"
        elif ext in (".yaml", ".yml", ".json", ".toml"):
            return "config"
        else:
            return "text"

    async def _fetch_repo_tree(self, owner: str, repo: str) -> list[dict]:
        """
        Obtiene el árbol de archivos de un repositorio.

        Returns:
            Lista de objetos de archivo con path, type, size, url
        """
        client = await self._get_client()

        # Obtener branch por defecto
        repo_url = f"https://api.github.com/repos/{owner}/{repo}"
        response = await client.get(repo_url)

        if response.status_code != 200:
            print(f"Error fetching repo {owner}/{repo}: {response.status_code}")
            return []

        repo_data = response.json()
        default_branch = repo_data.get("default_branch", "main")

        # Obtener árbol completo
        await asyncio.sleep(self.RATE_LIMIT_DELAY)
        tree_url = f"{repo_url}/git/trees/{default_branch}?recursive=1"
        response = await client.get(tree_url)

        if response.status_code != 200:
            print(f"Error fetching tree for {owner}/{repo}: {response.status_code}")
            return []

        tree_data = response.json()
        return tree_data.get("tree", [])

    async def _fetch_file_content(self, owner: str, repo: str, path: str) -> Optional[str]:
        """
        Descarga el contenido de un archivo.

        Returns:
            Contenido del archivo o None si falla
        """
        client = await self._get_client()
        await asyncio.sleep(self.RATE_LIMIT_DELAY)

        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        response = await client.get(url)

        if response.status_code != 200:
            return None

        data = response.json()

        # El contenido viene en base64
        if "content" in data:
            try:
                content = base64.b64decode(data["content"]).decode("utf-8")
                return content
            except Exception:
                return None

        return None

    def _format_code_document(
        self, content: str, path: str, repo: str, doc_type: str
    ) -> str:
        """
        Formatea el contenido del código para entrenamiento.

        Añade contexto sobre el archivo.
        """
        ext = Path(path).suffix.lower()
        filename = Path(path).name

        # Detectar lenguaje para syntax highlighting
        lang_map = {
            ".py": "python",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "cpp",
            ".hpp": "cpp",
            ".js": "javascript",
            ".ts": "typescript",
            ".md": "markdown",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
        }
        lang = lang_map.get(ext, "")

        if doc_type == "code":
            formatted = f"""## Archivo: {filename}
**Repositorio:** {repo}
**Path:** {path}

```{lang}
{content}
```
"""
        else:
            formatted = f"""## {filename}
**Repositorio:** {repo}
**Path:** {path}

{content}
"""
        return formatted

    async def _collect_repo(self, repo: str) -> AsyncIterator[CollectedDocument]:
        """
        Recolecta documentos de un repositorio.
        """
        parts = repo.split("/")
        if len(parts) != 2:
            print(f"Formato de repo inválido: {repo} (debe ser owner/repo)")
            return

        owner, repo_name = parts

        print(f"Recolectando de {repo}...")

        # Obtener árbol de archivos
        tree = await self._fetch_repo_tree(owner, repo_name)

        for item in tree:
            if not self.has_capacity:
                break

            # Solo archivos (no directorios)
            if item.get("type") != "blob":
                continue

            path = item.get("path", "")
            size = item.get("size", 0)

            # Filtrar por tamaño
            if size > self.max_file_size_kb * 1024:
                continue

            # Filtrar por patrón
            if not self._should_include_file(path):
                continue

            # Descargar contenido
            content = await self._fetch_file_content(owner, repo_name, path)
            if not content:
                continue

            # Verificar contenido mínimo
            if len(content) < 50:
                continue

            # Detectar tipo
            doc_type = self._detect_doc_type(path, content)

            # Calcular relevancia
            relevance = self.calculate_relevance(content)

            # Formatear para training
            formatted_content = self._format_code_document(
                content, path, repo, doc_type
            )

            # Verificar duplicado
            if self.is_duplicate(formatted_content):
                continue

            doc = CollectedDocument(
                content=formatted_content,
                title=Path(path).name,
                source_type="github",
                source_url=f"https://github.com/{repo}/blob/main/{path}",
                language=self.language,
                doc_type=doc_type,
                domain_relevance=relevance,
            )

            if await self.save_document(doc):
                yield doc

    async def collect(self) -> AsyncIterator[CollectedDocument]:
        """
        Recolecta documentos de todos los repositorios configurados.
        """
        try:
            for repo in self.repos:
                if not self.has_capacity:
                    break

                async for doc in self._collect_repo(repo):
                    yield doc

        finally:
            await self._close_client()
