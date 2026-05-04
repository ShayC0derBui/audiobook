"""EPUB loading and spine traversal."""

from __future__ import annotations

import logging
from pathlib import Path

import ebooklib
from ebooklib import epub

logger = logging.getLogger(__name__)


class EpubBook:
    """Wrapper around ebooklib epub providing spine-ordered document access."""

    def __init__(self, book: epub.EpubBook, source_path: Path) -> None:
        self.book = book
        self.source_path = source_path
        self._spine_items: list[epub.EpubItem] | None = None

    @property
    def title(self) -> str:
        return self.book.get_metadata("DC", "title")[0][0] if self.book.get_metadata("DC", "title") else "Untitled"

    @property
    def spine_items(self) -> list[epub.EpubItem]:
        """Get documents in spine (reading) order."""
        if self._spine_items is None:
            self._spine_items = []
            for item_id, _ in self.book.spine:
                item = self.book.get_item_with_id(item_id)
                if item and item.get_type() == ebooklib.ITEM_DOCUMENT:
                    self._spine_items.append(item)
        return self._spine_items

    @property
    def toc_map(self) -> dict[str, str]:
        """Map file names to TOC titles for metadata enhancement."""
        toc_titles: dict[str, str] = {}
        self._walk_toc(self.book.toc, toc_titles)
        return toc_titles

    def _walk_toc(self, toc_entries: list, titles: dict[str, str]) -> None:
        """Recursively walk TOC tree and collect href -> title mappings."""
        for entry in toc_entries:
            if isinstance(entry, tuple):
                # Section with children
                section, children = entry
                if hasattr(section, "href") and hasattr(section, "title"):
                    href = section.href.split("#")[0]
                    titles[href] = section.title
                self._walk_toc(children, titles)
            elif hasattr(entry, "href") and hasattr(entry, "title"):
                href = entry.href.split("#")[0]
                titles[href] = entry.title


def load_epub(path: Path) -> EpubBook:
    """Load an EPUB file and return wrapped book object."""
    if not path.exists():
        raise FileNotFoundError(f"EPUB file not found: {path}")
    if not path.suffix.lower() == ".epub":
        raise ValueError(f"File is not an EPUB: {path}")

    logger.info(f"Loading EPUB: {path.name}")
    book = epub.read_epub(str(path), options={"ignore_ncx": False})
    return EpubBook(book, path)
