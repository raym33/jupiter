"""
Recolectores de datos de diversas fuentes.
"""

from jupiter.data.collectors.base import DataCollector, CollectedDocument
from jupiter.data.collectors.web import WebCollector
from jupiter.data.collectors.github import GitHubCollector
from jupiter.data.collectors.docs import DocsCollector

__all__ = [
    "DataCollector",
    "CollectedDocument",
    "WebCollector",
    "GitHubCollector",
    "DocsCollector",
]
