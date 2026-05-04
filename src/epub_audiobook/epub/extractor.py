"""Chapter XHTML cleanup and normalized text extraction."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, NavigableString, Tag

from epub_audiobook.epub.parser import EpubBook

logger = logging.getLogger(__name__)

# Elements that don't contribute to reading content
NON_READING_TAGS = {
    "script", "style", "nav", "header", "footer", "aside",
    "figure", "figcaption", "img", "svg", "math", "table",
}

# Block elements that imply paragraph boundaries
BLOCK_TAGS = {
    "p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
    "blockquote", "li", "br", "hr", "section", "article",
}


@dataclass
class Chapter:
    """Canonical chapter with stable ID for resume tracking."""

    id: str
    title: str
    text: str
    spine_index: int

    @property
    def char_count(self) -> int:
        return len(self.text)


def _generate_chapter_id(spine_index: int, file_name: str) -> str:
    """Generate a stable chapter ID from spine position and filename."""
    raw = f"{spine_index:04d}:{file_name}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _extract_title_from_html(soup: BeautifulSoup) -> str | None:
    """Try to extract chapter title from heading elements."""
    for tag_name in ("h1", "h2", "h3"):
        heading = soup.find(tag_name)
        if heading:
            text = heading.get_text(strip=True)
            if text and len(text) < 200:
                return text
    return None


def _clean_html_to_text(html_content: bytes | str) -> str:
    """Convert chapter HTML to clean reading text."""
    soup = BeautifulSoup(html_content, "lxml")

    # Remove non-reading elements
    for tag in soup.find_all(NON_READING_TAGS):
        tag.decompose()

    # Remove footnote markers but keep reference text inline
    for sup in soup.find_all("sup"):
        link = sup.find("a")
        if link:
            # Convert footnote reference to inline readable form
            ref_text = link.get_text(strip=True)
            if ref_text.isdigit():
                sup.replace_with(f" (note {ref_text}) ")
            else:
                sup.replace_with(f" ({ref_text}) ")

    # Extract text preserving paragraph boundaries
    paragraphs: list[str] = []
    body = soup.find("body") or soup

    for element in body.descendants:
        if isinstance(element, Tag) and element.name in BLOCK_TAGS:
            text = element.get_text(separator=" ", strip=True)
            if text:
                paragraphs.append(text)
        elif isinstance(element, NavigableString):
            parent = element.parent
            if parent and parent.name not in BLOCK_TAGS and parent.name not in NON_READING_TAGS:
                text = str(element).strip()
                if text and not paragraphs:
                    paragraphs.append(text)

    # Join paragraphs with double newline, normalize whitespace within
    result = "\n\n".join(paragraphs)
    result = re.sub(r"[ \t]+", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def extract_chapters(book: EpubBook) -> list[Chapter]:
    """Extract all chapters from an EPUB in spine order."""
    toc_map = book.toc_map
    chapters: list[Chapter] = []

    for idx, item in enumerate(book.spine_items):
        file_name = item.get_name()
        html_content = item.get_content()

        text = _clean_html_to_text(html_content)
        if not text or len(text) < 50:
            logger.debug(f"Skipping short/empty spine item: {file_name}")
            continue

        # Title resolution: TOC first, then heading extraction, then fallback
        title = toc_map.get(file_name)
        if not title:
            soup = BeautifulSoup(html_content, "lxml")
            title = _extract_title_from_html(soup)
        if not title:
            title = f"Chapter {len(chapters) + 1}"

        chapter_id = _generate_chapter_id(idx, file_name)
        chapters.append(Chapter(
            id=chapter_id,
            title=title,
            text=text,
            spine_index=idx,
        ))

    logger.info(f"Extracted {len(chapters)} chapters from '{book.title}'")
    return chapters
