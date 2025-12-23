"""
Recolector de documentación oficial.

Especializado en parsear sitios de documentación con estructura conocida.
"""

import asyncio
from pathlib import Path
from typing import AsyncIterator, Optional
from urllib.parse import urljoin, urlparse
import re

import httpx
from bs4 import BeautifulSoup

from jupiter.data.collectors.base import DataCollector, CollectedDocument


class DocsCollector(DataCollector):
    """
    Recolector especializado en documentación técnica.

    Optimizado para:
    - Sitios de documentación (ReadTheDocs, Docusaurus, GitBook, etc.)
    - Extracción limpia de contenido técnico
    - Preservación de bloques de código
    - Estructura de secciones
    """

    # Selectores CSS para diferentes plataformas de docs
    CONTENT_SELECTORS = [
        # Docusaurus
        "article.markdown",
        ".markdown",
        # ReadTheDocs
        ".rst-content",
        ".document",
        # GitBook
        ".page-inner",
        ".markdown-section",
        # Sphinx
        ".body",
        ".section",
        # Generic
        "main article",
        "main .content",
        ".documentation",
        ".docs-content",
        "article",
        "main",
    ]

    def __init__(
        self,
        urls: list[str],
        output_dir: Path,
        domain_keywords: list[str] = None,
        negative_keywords: list[str] = None,
        max_documents: int = 10000,
        max_depth: int = 5,
        delay_seconds: float = 1.0,
        language: str = "es",
    ):
        super().__init__(
            output_dir=output_dir,
            domain_keywords=domain_keywords,
            negative_keywords=negative_keywords,
            max_documents=max_documents,
            language=language,
        )

        self.urls = urls
        self.max_depth = max_depth
        self.delay_seconds = delay_seconds

        self._visited_urls: set[str] = set()
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Obtiene o crea el cliente HTTP."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                headers={
                    "User-Agent": "Jupiter-DocsCollector/1.0 (Educational AI Training)",
                    "Accept": "text/html,application/xhtml+xml",
                },
            )
        return self._client

    async def _close_client(self) -> None:
        """Cierra el cliente HTTP."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _find_content(self, soup: BeautifulSoup) -> Optional[BeautifulSoup]:
        """Encuentra el contenedor de contenido principal."""
        for selector in self.CONTENT_SELECTORS:
            content = soup.select_one(selector)
            if content:
                return content
        return soup.find("body")

    def _extract_code_blocks(self, soup: BeautifulSoup) -> list[dict]:
        """
        Extrae bloques de código con su lenguaje.

        Returns:
            Lista de {"language": str, "code": str}
        """
        code_blocks = []

        for pre in soup.find_all("pre"):
            code = pre.find("code")
            if code:
                # Intentar detectar lenguaje de la clase
                classes = code.get("class", [])
                language = ""
                for cls in classes:
                    if cls.startswith("language-"):
                        language = cls.replace("language-", "")
                        break
                    elif cls.startswith("lang-"):
                        language = cls.replace("lang-", "")
                        break

                code_blocks.append({
                    "language": language,
                    "code": code.get_text(),
                })

        return code_blocks

    def _extract_content(self, html: str) -> tuple[str, str, list[str]]:
        """
        Extrae contenido de documentación.

        Returns:
            (title, formatted_content, links)
        """
        soup = BeautifulSoup(html, "html.parser")

        # Remover elementos no deseados
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()

        # También remover menús de navegación
        for nav in soup.select(".sidebar, .toc, .navigation, .menu, .nav"):
            nav.decompose()

        # Obtener título
        title = ""
        if soup.title:
            title = soup.title.string or ""
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)

        # Encontrar contenido principal
        content_elem = self._find_content(soup)
        if not content_elem:
            return title, "", []

        # Procesar el contenido preservando código
        formatted_parts = []

        for elem in content_elem.descendants:
            if elem.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                level = int(elem.name[1])
                text = elem.get_text(strip=True)
                formatted_parts.append(f"\n{'#' * level} {text}\n")

            elif elem.name == "p":
                text = elem.get_text(strip=True)
                if text:
                    formatted_parts.append(f"\n{text}\n")

            elif elem.name == "pre":
                code = elem.find("code")
                if code:
                    # Detectar lenguaje
                    classes = code.get("class", [])
                    lang = ""
                    for cls in classes:
                        if "language-" in cls or "lang-" in cls:
                            lang = cls.replace("language-", "").replace("lang-", "")
                            break

                    code_text = code.get_text()
                    formatted_parts.append(f"\n```{lang}\n{code_text}\n```\n")

            elif elem.name == "li":
                text = elem.get_text(strip=True)
                if text and not any(text in p for p in formatted_parts):
                    formatted_parts.append(f"- {text}")

            elif elem.name == "code" and elem.parent.name != "pre":
                # Código inline - no hacer nada, ya está en el texto del padre
                pass

        content = "\n".join(formatted_parts)

        # Limpiar espacios múltiples
        content = re.sub(r"\n{3,}", "\n\n", content)

        # Extraer links para seguir
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # Solo links a otras páginas de docs
            if not href.startswith(("#", "javascript:", "mailto:")):
                links.append(href)

        return title.strip(), content.strip(), links

    def _is_doc_url(self, url: str, base_domain: str) -> bool:
        """Verifica si una URL parece ser de documentación."""
        parsed = urlparse(url)

        # Solo mismo dominio
        if base_domain not in parsed.netloc:
            return False

        # Ignorar assets
        ignore_patterns = [
            r"\.(css|js|png|jpg|jpeg|gif|svg|ico|woff|ttf|eot)$",
            r"/api/",
            r"/auth/",
            r"/_next/",
            r"/_static/",
            r"/assets/",
        ]
        for pattern in ignore_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return False

        return True

    async def _crawl_docs(
        self, url: str, depth: int, base_domain: str
    ) -> AsyncIterator[CollectedDocument]:
        """
        Crawlea páginas de documentación.
        """
        # Normalizar URL
        url = url.rstrip("/")

        if url in self._visited_urls:
            return
        if not self.has_capacity:
            return

        self._visited_urls.add(url)

        # Delay
        await asyncio.sleep(self.delay_seconds)

        # Descargar
        try:
            client = await self._get_client()
            response = await client.get(url)

            if response.status_code != 200:
                return

            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type:
                return

        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return

        # Extraer contenido
        title, content, links = self._extract_content(response.text)

        # Verificar contenido mínimo
        if len(content) < 300:
            return

        # Calcular relevancia
        relevance = self.calculate_relevance(content)
        if relevance < 0.1:
            return

        # Verificar duplicado
        if not self.is_duplicate(content):
            doc = CollectedDocument(
                content=content,
                title=title,
                source_type="docs",
                source_url=url,
                language=self.language,
                doc_type="documentation",
                domain_relevance=relevance,
            )

            if await self.save_document(doc):
                yield doc

        # Seguir links
        if depth < self.max_depth:
            for link in links:
                full_url = urljoin(url, link)

                if self._is_doc_url(full_url, base_domain):
                    async for doc in self._crawl_docs(full_url, depth + 1, base_domain):
                        yield doc

    async def collect(self) -> AsyncIterator[CollectedDocument]:
        """
        Recolecta documentación de las URLs configuradas.
        """
        try:
            for seed_url in self.urls:
                if not self.has_capacity:
                    break

                # Extraer dominio base
                parsed = urlparse(seed_url)
                base_domain = ".".join(parsed.netloc.split(".")[-2:])

                print(f"Recolectando documentación de {parsed.netloc}...")

                async for doc in self._crawl_docs(seed_url, depth=0, base_domain=base_domain):
                    yield doc

        finally:
            await self._close_client()
