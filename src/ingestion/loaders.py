"""Simple text/markdown document loaders.

Keeps ingestion dependency-light: plain ``.txt`` and ``.md`` files are read
directly as ``langchain_core.documents.Document`` objects with source-path
metadata attached. This avoids pulling in heavyweight PDF/HTML parsers for a
project meant to demonstrate the retrieval pipeline rather than every
possible file format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".txt", ".md"}


def load_documents_from_dir(directory: str | Path) -> list[Document]:
    """Load all supported text files from ``directory`` (recursively).

    Args:
        directory: Path to a folder containing ``.txt`` / ``.md`` files.

    Returns:
        A list of :class:`~langchain_core.documents.Document` objects, one
        per file, with ``source`` metadata set to the file path.

    Raises:
        FileNotFoundError: If ``directory`` does not exist.
    """

    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    documents: list[Document] = []
    for path in _iter_supported_files(directory):
        text = path.read_text(encoding="utf-8", errors="ignore")
        documents.append(
            Document(
                page_content=text,
                metadata={"source": str(path), "filename": path.name},
            )
        )
    return documents


def _iter_supported_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path
