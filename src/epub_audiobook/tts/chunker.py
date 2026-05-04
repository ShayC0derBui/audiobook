"""Sentence-aware text chunking for TTS synthesis."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class TextChunk:
    """A chunk of text ready for TTS synthesis."""

    id: str
    chapter_id: str
    index: int
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)


# Sentence boundary pattern: period/question/exclamation followed by space and capital
SENTENCE_BOUNDARY = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z"\u201c])|(?<=[.!?])\s*\n'
)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences preserving boundaries."""
    sentences = SENTENCE_BOUNDARY.split(text)
    # Filter empty and strip
    return [s.strip() for s in sentences if s and s.strip()]


def chunk_text(
    text: str,
    chapter_id: str,
    max_chars: int = 500,
    max_tokens: int = 250,
) -> list[TextChunk]:
    """Split chapter text into TTS-ready chunks.

    Uses sentence-aware splitting to avoid mid-sentence breaks.
    Respects both character and approximate token limits.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[TextChunk] = []
    current_sentences: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        approx_tokens = sentence_len // 4  # rough char-to-token ratio

        # If a single sentence exceeds limits, it becomes its own chunk
        if sentence_len > max_chars:
            # Flush current buffer first
            if current_sentences:
                chunk_text_str = " ".join(current_sentences)
                chunks.append(TextChunk(
                    id=f"{chapter_id}_c{len(chunks):04d}",
                    chapter_id=chapter_id,
                    index=len(chunks),
                    text=chunk_text_str,
                ))
                current_sentences = []
                current_len = 0

            # Force-split the long sentence at word boundaries
            words = sentence.split()
            word_buffer: list[str] = []
            buf_len = 0
            for word in words:
                if buf_len + len(word) + 1 > max_chars and word_buffer:
                    chunks.append(TextChunk(
                        id=f"{chapter_id}_c{len(chunks):04d}",
                        chapter_id=chapter_id,
                        index=len(chunks),
                        text=" ".join(word_buffer),
                    ))
                    word_buffer = []
                    buf_len = 0
                word_buffer.append(word)
                buf_len += len(word) + 1

            if word_buffer:
                chunks.append(TextChunk(
                    id=f"{chapter_id}_c{len(chunks):04d}",
                    chapter_id=chapter_id,
                    index=len(chunks),
                    text=" ".join(word_buffer),
                ))
            continue

        # Check if adding this sentence would exceed limits
        new_len = current_len + sentence_len + (1 if current_sentences else 0)
        new_approx_tokens = new_len // 4

        if new_len > max_chars or new_approx_tokens > max_tokens:
            # Flush current buffer
            if current_sentences:
                chunk_text_str = " ".join(current_sentences)
                chunks.append(TextChunk(
                    id=f"{chapter_id}_c{len(chunks):04d}",
                    chapter_id=chapter_id,
                    index=len(chunks),
                    text=chunk_text_str,
                ))
            current_sentences = [sentence]
            current_len = sentence_len
        else:
            current_sentences.append(sentence)
            current_len = new_len

    # Flush remaining
    if current_sentences:
        chunk_text_str = " ".join(current_sentences)
        chunks.append(TextChunk(
            id=f"{chapter_id}_c{len(chunks):04d}",
            chapter_id=chapter_id,
            index=len(chunks),
            text=chunk_text_str,
        ))

    return chunks
