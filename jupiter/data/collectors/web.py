"""
Recolector de datos web (scraping).
"""

import asyncio
from pathlib import Path
from typing import AsyncIterator, Optional
from urllib.parse import urljoin, urlparse
import re

import httpx
from bs4 import BeautifulSoup

from jupiter.data.collectors.base import DataCollector, CollectedDocument


class WebCollector(DataCollector):
    """
    Recolector de contenido web.

    Características:
    - Scraping respetuoso (delays, user-agent apropiado)
    - Extracción de texto limpio de HTML
    - Seguimiento de links internos
    - Detección de contenido relevante
    """

    def __init__(
        self,
        urls: list[str],
        output_dir: Path,
        domain_keywords: list[str] = None,
        negative_keywords: list[str] = None,
        max_documents: int = 10000,
        max_depth: int = 3,
        delay_seconds: float = 1.0,
        follow_links: bool = True,
        language: str = "es",
    ):
        """
        Args:
            urls: Lista de URLs semilla para empezar el scraping
            output_dir: Directorio de salida
            max_depth: Profundidad máxima de seguimiento de links
            delay_seconds: Delay entre requests (respeto al servidor)
            follow_links: Si seguir links internos
        """
        super().__init__(
            output_dir=output_dir,
            domain_keywords=domain_keywords,
            negative_keywords=negative_keywords,
            max_documents=max_documents,
            language=language,
        )

        self.seed_urls = urls
        self.max_depth = max_depth
        self.delay_seconds = delay_seconds
        self.follow_links = follow_links

        self._visited_urls: set[str] = set()
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Obtiene o crea el cliente HTTP."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                headers={
                    "User-Agent": "Jupiter-DataCollector/1.0 (Educational AI Training)",
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "es,en;q=0.9",
                },
            )
        return self._client

    async def _close_client(self) -> None:
        """Cierra el cliente HTTP."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _normalize_url(self, url: str) -> str:
        """Normaliza una URL para comparación."""
        parsed = urlparse(url)
        # Remover fragmentos y trailing slashes
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        return normalized

    def _is_valid_url(self, url: str, base_domain: str) -> bool:
        """Verifica si una URL es válida para seguir."""
        try:
            parsed = urlparse(url)

            # Solo HTTP/HTTPS
            if parsed.scheme not in ("http", "https"):
                return False

            # Solo mismo dominio o subdominios
            if not parsed.netloc.endswith(base_domain):
                return False

            # Ignorar ciertos patrones
            ignore_patterns = [
                r"/api/",
                r"/auth/",
                r"/login",
                r"/logout",
                r"/admin/",
                r"\.(pdf|zip|tar|gz|exe|dmg|pkg)$",
                r"\.(jpg|jpeg|png|gif|svg|ico|webp)$",
                r"\.(mp4|mp3|wav|avi|mov)$",
            ]
            for pattern in ignore_patterns:
                if re.search(pattern, url, re.IGNORECASE):
                    return False

            return True
        except Exception:
            return False

    def _extract_text(self, html: str) -> tuple[str, str, list[str]]:
        """
        Extrae texto limpio del HTML.

        Returns:
            (title, content, links)
        """
        soup = BeautifulSoup(html, "html.parser")

        # Remover scripts, styles, etc.
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()

        # Obtener título
        title = ""
        if soup.title:
            title = soup.title.string or ""
        elif soup.h1:
            title = soup.h1.get_text(strip=True)

        # Obtener contenido principal
        # Intentar encontrar el contenido principal
        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find(class_=re.compile(r"content|post|entry|article", re.I))
            or soup.find("body")
        )

        if main_content:
            # Extraer texto
            content = main_content.get_text(separator="\n", strip=True)
        else:
            content = soup.get_text(separator="\n", strip=True)

        # Limpiar espacios múltiples
        content = re.sub(r"\n{3,}", "\n\n", content)
        content = re.sub(r" {2,}", " ", content)

        # Extraer links
        links = []
        for a in soup.find_all("a", href=True):
            links.append(a["href"])

        return title.strip(), content.strip(), links

    async def _fetch_page(self, url: str) -> Optional[tuple[str, str, list[str]]]:
        """
        Descarga y parsea una página.

        Returns:
            (title, content, links) o None si falla
        """
        try:
            client = await self._get_client()
            response = await client.get(url)

            if response.status_code != 200:
                return None

            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type:
                return None

            return self._extract_text(response.text)

        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    async def _crawl_url(
        self, url: str, depth: int, base_domain: str
    ) -> AsyncIterator[CollectedDocument]:
        """
        Crawlea una URL y sus links.

        Args:
            url: URL a crawlear
            depth: Profundidad actual
            base_domain: Dominio base para filtrar links
        """
        url = self._normalize_url(url)

        # Verificar si ya visitamos o llegamos al límite
        if url in self._visited_urls:
            return
        if not self.has_capacity:
            return

        self._visited_urls.add(url)

        # Delay para ser respetuoso
        await asyncio.sleep(self.delay_seconds)

        # Descargar página
        result = await self._fetch_page(url)
        if not result:
            return

        title, content, links = result

        # Verificar contenido mínimo
        if len(content) < 200:  # Muy poco contenido
            return

        # Verificar relevancia
        relevance = self.calculate_relevance(content)
        if relevance < 0.1:  # Muy poco relevante
            return

        # Verificar duplicado
        if not self.is_duplicate(content):
            doc = CollectedDocument(
                content=content,
                title=title,
                source_type="web",
                source_url=url,
                language=self.language,
                doc_type="text",
                domain_relevance=relevance,
            )

            if await self.save_document(doc):
                yield doc

        # Seguir links si corresponde
        if self.follow_links and depth < self.max_depth:
            for link in links:
                # Convertir a URL absoluta
                full_url = urljoin(url, link)

                if self._is_valid_url(full_url, base_domain):
                    async for doc in self._crawl_url(full_url, depth + 1, base_domain):
                        yield doc

    async def collect(self) -> AsyncIterator[CollectedDocument]:
        """
        Recolecta documentos de las URLs semilla.
        """
        try:
            for seed_url in self.seed_urls:
                if not self.has_capacity:
                    break

                # Extraer dominio base
                parsed = urlparse(seed_url)
                base_domain = parsed.netloc.split(".")[-2:]  # ej: ["example", "com"]
                base_domain = ".".join(base_domain)

                async for doc in self._crawl_url(seed_url, depth=0, base_domain=base_domain):
                    yield doc

        finally:
            await self._close_client()
