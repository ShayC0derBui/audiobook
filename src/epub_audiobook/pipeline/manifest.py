"""Persistent run state manifest for resume support."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"


class Status(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ChunkState:
    """State for a single TTS chunk."""

    def __init__(self, chunk_id: str, chapter_id: str, index: int) -> None:
        self.chunk_id = chunk_id
        self.chapter_id = chapter_id
        self.index = index
        self.status = Status.PENDING
        self.output_path: str | None = None
        self.error: str | None = None
        self.attempts: int = 0
        self.last_attempt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "chapter_id": self.chapter_id,
            "index": self.index,
            "status": self.status.value,
            "output_path": self.output_path,
            "error": self.error,
            "attempts": self.attempts,
            "last_attempt": self.last_attempt,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkState":
        state = cls(data["chunk_id"], data["chapter_id"], data["index"])
        state.status = Status(data["status"])
        state.output_path = data.get("output_path")
        state.error = data.get("error")
        state.attempts = data.get("attempts", 0)
        state.last_attempt = data.get("last_attempt")
        return state


class ChapterState:
    """State for a chapter (aggregated from chunks)."""

    def __init__(self, chapter_id: str, title: str, spine_index: int) -> None:
        self.chapter_id = chapter_id
        self.title = title
        self.spine_index = spine_index
        self.parse_status = Status.PENDING
        self.chunk_status = Status.PENDING
        self.tts_status = Status.PENDING
        self.audio_status = Status.PENDING
        self.output_path: str | None = None
        self.chunks: list[ChunkState] = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "chapter_id": self.chapter_id,
            "title": self.title,
            "spine_index": self.spine_index,
            "parse_status": self.parse_status.value,
            "chunk_status": self.chunk_status.value,
            "tts_status": self.tts_status.value,
            "audio_status": self.audio_status.value,
            "output_path": self.output_path,
            "chunks": [c.to_dict() for c in self.chunks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChapterState":
        state = cls(data["chapter_id"], data["title"], data["spine_index"])
        state.parse_status = Status(data["parse_status"])
        state.chunk_status = Status(data["chunk_status"])
        state.tts_status = Status(data["tts_status"])
        state.audio_status = Status(data["audio_status"])
        state.output_path = data.get("output_path")
        state.chunks = [ChunkState.from_dict(c) for c in data.get("chunks", [])]
        return state


class Manifest:
    """Run manifest tracking the entire conversion pipeline state."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.manifest_path = output_dir / MANIFEST_FILENAME
        self.input_path: str = ""
        self.book_title: str = ""
        self.language: str = "en"
        self.voice_prompt: str = ""
        self.profile: str = ""
        self.created_at: str = ""
        self.updated_at: str = ""
        self.chapters: list[ChapterState] = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": self.input_path,
            "book_title": self.book_title,
            "language": self.language,
            "voice_prompt": self.voice_prompt,
            "profile": self.profile,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "chapters": [ch.to_dict() for ch in self.chapters],
        }

    def save(self) -> None:
        """Atomically save manifest to disk."""
        self.updated_at = datetime.now(timezone.utc).isoformat()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        tmp_path = self.manifest_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(self.to_dict(), indent=2))
        tmp_path.rename(self.manifest_path)

    @classmethod
    def load(cls, output_dir: Path) -> "Manifest":
        """Load manifest from disk."""
        manifest_path = output_dir / MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest found at: {manifest_path}")

        data = json.loads(manifest_path.read_text())
        manifest = cls(output_dir)
        manifest.input_path = data["input_path"]
        manifest.book_title = data["book_title"]
        manifest.language = data["language"]
        manifest.voice_prompt = data["voice_prompt"]
        manifest.profile = data["profile"]
        manifest.created_at = data["created_at"]
        manifest.updated_at = data["updated_at"]
        manifest.chapters = [ChapterState.from_dict(ch) for ch in data["chapters"]]
        return manifest

    @classmethod
    def exists(cls, output_dir: Path) -> bool:
        return (output_dir / MANIFEST_FILENAME).exists()

    def get_pending_chapters(self) -> list[ChapterState]:
        """Get chapters that still need TTS processing."""
        return [
            ch for ch in self.chapters
            if ch.tts_status != Status.COMPLETED
        ]

    def get_pending_chunks(self, chapter: ChapterState) -> list[ChunkState]:
        """Get chunks within a chapter that still need synthesis."""
        return [
            c for c in chapter.chunks
            if c.status not in (Status.COMPLETED, Status.SKIPPED)
        ]

    def mark_chunk_complete(self, chunk: ChunkState, output_path: str) -> None:
        """Mark a chunk as completed and save."""
        chunk.status = Status.COMPLETED
        chunk.output_path = output_path
        chunk.last_attempt = datetime.now(timezone.utc).isoformat()
        self.save()

    def mark_chunk_failed(self, chunk: ChunkState, error: str) -> None:
        """Mark a chunk as failed and save."""
        chunk.status = Status.FAILED
        chunk.error = error
        chunk.attempts += 1
        chunk.last_attempt = datetime.now(timezone.utc).isoformat()
        self.save()

    def mark_chapter_tts_complete(self, chapter: ChapterState) -> None:
        """Mark chapter TTS as complete."""
        chapter.tts_status = Status.COMPLETED
        self.save()

    def mark_chapter_audio_complete(self, chapter: ChapterState, output_path: str) -> None:
        """Mark chapter audio assembly as complete."""
        chapter.audio_status = Status.COMPLETED
        chapter.output_path = output_path
        self.save()

    def summary(self) -> dict[str, int]:
        """Return summary counts."""
        total_chapters = len(self.chapters)
        completed = sum(1 for ch in self.chapters if ch.audio_status == Status.COMPLETED)
        failed_chunks = sum(
            sum(1 for c in ch.chunks if c.status == Status.FAILED)
            for ch in self.chapters
        )
        return {
            "total_chapters": total_chapters,
            "completed_chapters": completed,
            "pending_chapters": total_chapters - completed,
            "failed_chunks": failed_chunks,
        }
