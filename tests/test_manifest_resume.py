"""Tests for manifest resume and idempotency."""

import json
import pytest
from pathlib import Path
from epub_audiobook.pipeline.manifest import Manifest, ChapterState, ChunkState, Status


@pytest.fixture
def tmp_output(tmp_path):
    return tmp_path / "output"


class TestManifest:
    def test_save_and_load(self, tmp_output):
        manifest = Manifest(tmp_output)
        manifest.input_path = "/path/to/book.epub"
        manifest.book_title = "Test Book"
        manifest.language = "en"
        manifest.voice_prompt = "Calm narrator"
        manifest.profile = "apple-silicon"
        manifest.created_at = "2024-01-01T00:00:00Z"

        ch = ChapterState("ch001", "Chapter 1", 0)
        ch.parse_status = Status.COMPLETED
        chunk = ChunkState("ch001_c0000", "ch001", 0)
        ch.chunks.append(chunk)
        manifest.chapters.append(ch)

        manifest.save()

        loaded = Manifest.load(tmp_output)
        assert loaded.book_title == "Test Book"
        assert len(loaded.chapters) == 1
        assert loaded.chapters[0].chunks[0].chunk_id == "ch001_c0000"

    def test_atomic_save(self, tmp_output):
        manifest = Manifest(tmp_output)
        manifest.input_path = "test.epub"
        manifest.book_title = "Book"
        manifest.language = "en"
        manifest.voice_prompt = ""
        manifest.profile = "colab"
        manifest.created_at = "2024-01-01T00:00:00Z"
        manifest.save()

        # No .tmp file should remain
        assert not (tmp_output / "manifest.tmp").exists()
        assert (tmp_output / "manifest.json").exists()

    def test_mark_chunk_complete(self, tmp_output):
        manifest = Manifest(tmp_output)
        manifest.input_path = "test.epub"
        manifest.book_title = "Book"
        manifest.language = "en"
        manifest.voice_prompt = ""
        manifest.profile = "colab"
        manifest.created_at = "2024-01-01T00:00:00Z"

        ch = ChapterState("ch001", "Chapter 1", 0)
        chunk = ChunkState("ch001_c0000", "ch001", 0)
        ch.chunks.append(chunk)
        manifest.chapters.append(ch)

        manifest.mark_chunk_complete(chunk, "/output/chunks/ch001_c0000.wav")

        assert chunk.status == Status.COMPLETED
        assert chunk.output_path == "/output/chunks/ch001_c0000.wav"

    def test_mark_chunk_failed(self, tmp_output):
        manifest = Manifest(tmp_output)
        manifest.input_path = "test.epub"
        manifest.book_title = "Book"
        manifest.language = "en"
        manifest.voice_prompt = ""
        manifest.profile = "colab"
        manifest.created_at = "2024-01-01T00:00:00Z"

        ch = ChapterState("ch001", "Chapter 1", 0)
        chunk = ChunkState("ch001_c0000", "ch001", 0)
        ch.chunks.append(chunk)
        manifest.chapters.append(ch)

        manifest.mark_chunk_failed(chunk, "TTS timeout")

        assert chunk.status == Status.FAILED
        assert chunk.attempts == 1
        assert chunk.error == "TTS timeout"

    def test_get_pending_chapters(self, tmp_output):
        manifest = Manifest(tmp_output)
        manifest.input_path = "test.epub"
        manifest.book_title = "Book"
        manifest.language = "en"
        manifest.voice_prompt = ""
        manifest.profile = "colab"
        manifest.created_at = "2024-01-01T00:00:00Z"

        ch1 = ChapterState("ch001", "Done Chapter", 0)
        ch1.tts_status = Status.COMPLETED
        ch2 = ChapterState("ch002", "Pending Chapter", 1)
        ch2.tts_status = Status.PENDING

        manifest.chapters = [ch1, ch2]
        pending = manifest.get_pending_chapters()

        assert len(pending) == 1
        assert pending[0].chapter_id == "ch002"

    def test_summary(self, tmp_output):
        manifest = Manifest(tmp_output)
        manifest.input_path = "test.epub"
        manifest.book_title = "Book"
        manifest.language = "en"
        manifest.voice_prompt = ""
        manifest.profile = "colab"
        manifest.created_at = "2024-01-01T00:00:00Z"

        ch1 = ChapterState("ch001", "Ch1", 0)
        ch1.audio_status = Status.COMPLETED
        ch1.chunks = [ChunkState("c1", "ch001", 0)]
        ch1.chunks[0].status = Status.COMPLETED

        ch2 = ChapterState("ch002", "Ch2", 1)
        ch2.chunks = [ChunkState("c2", "ch002", 0)]
        ch2.chunks[0].status = Status.FAILED

        manifest.chapters = [ch1, ch2]
        summary = manifest.summary()

        assert summary["total_chapters"] == 2
        assert summary["completed_chapters"] == 1
        assert summary["failed_chunks"] == 1
