"""Pipeline orchestrator wiring parse, chunk, TTS, and audio stages."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from epub_audiobook.config import AppConfig, RuntimeProfile, check_runtime_readiness
from epub_audiobook.epub.parser import load_epub
from epub_audiobook.epub.extractor import extract_chapters, Chapter
from epub_audiobook.tts.chunker import chunk_text, TextChunk
from epub_audiobook.tts.qwen_client import QwenTTSClient, TTSSynthesisError
from epub_audiobook.audio.processor import concatenate_chunks, concatenate_chapters
from epub_audiobook.pipeline.manifest import (
    Manifest, ChapterState, ChunkState, Status,
)

logger = logging.getLogger(__name__)
console = Console()


def _parse_chapter_range(chapters_str: str | None, total: int) -> tuple[int, int]:
    """Parse chapter range string like '1-5' or '3' into (start, end) 0-indexed."""
    if not chapters_str:
        return 0, total

    if "-" in chapters_str:
        parts = chapters_str.split("-", 1)
        start = max(0, int(parts[0]) - 1)
        end = min(total, int(parts[1]))
    else:
        idx = int(chapters_str) - 1
        start = max(0, idx)
        end = min(total, idx + 1)

    return start, end


def _retry_delay(attempt: int, config: AppConfig) -> float:
    """Calculate exponential backoff delay."""
    delay = config.retry.base_delay_seconds * (2 ** attempt)
    return min(delay, config.retry.max_delay_seconds)


def run_convert(config: AppConfig) -> None:
    """Run the full conversion pipeline."""
    # Runtime check
    issues = check_runtime_readiness(config)
    if issues:
        for issue in issues:
            console.print(f"[red]✗[/red] {issue}")
        raise SystemExit(1)

    console.print(f"[green]✓[/green] Runtime: {config.profile.value}")
    console.print(f"  Device: {config.get_torch_device()}")

    # Parse EPUB
    console.print(f"\n[bold]Loading:[/bold] {config.input_path.name}")
    book = load_epub(config.input_path)
    chapters = extract_chapters(book)

    if not chapters:
        console.print("[red]No chapters found in EPUB.[/red]")
        raise SystemExit(1)

    # Apply chapter range
    start, end = _parse_chapter_range(config.chapters, len(chapters))
    chapters = chapters[start:end]
    console.print(f"  Processing chapters {start + 1}–{end} ({len(chapters)} total)")

    # Initialize manifest
    manifest = _init_manifest(config, book.title, chapters)

    # Run TTS pipeline
    _run_tts_pipeline(config, manifest)

    # Assembly
    _run_audio_assembly(config, manifest)

    # Summary
    _print_summary(manifest)


def run_resume(output_dir: Path, profile: RuntimeProfile) -> None:
    """Resume an interrupted conversion."""
    if not Manifest.exists(output_dir):
        console.print(f"[red]No manifest found in: {output_dir}[/red]")
        raise SystemExit(1)

    manifest = Manifest.load(output_dir)
    console.print(f"[green]✓[/green] Resuming: {manifest.book_title}")

    summary = manifest.summary()
    console.print(
        f"  {summary['completed_chapters']}/{summary['total_chapters']} chapters done, "
        f"{summary['failed_chunks']} failed chunks"
    )

    # Rebuild config from manifest
    config = AppConfig(
        profile=profile,
        input_path=Path(manifest.input_path),
        output_dir=output_dir,
        model_path=None,
    )
    config.tts.language = manifest.language
    config.tts.voice_design_prompt = manifest.voice_prompt

    issues = check_runtime_readiness(config)
    if issues:
        for issue in issues:
            console.print(f"[red]✗[/red] {issue}")
        raise SystemExit(1)

    # Reset failed chunks to pending for retry
    for chapter in manifest.chapters:
        for chunk in chapter.chunks:
            if chunk.status == Status.FAILED and chunk.attempts < config.retry.max_retries:
                chunk.status = Status.PENDING
    manifest.save()

    _run_tts_pipeline(config, manifest)
    _run_audio_assembly(config, manifest)
    _print_summary(manifest)


def _init_manifest(config: AppConfig, book_title: str, chapters: list[Chapter]) -> Manifest:
    """Initialize or load existing manifest."""
    manifest = Manifest(config.output_dir)
    manifest.input_path = str(config.input_path)
    manifest.book_title = book_title
    manifest.language = config.tts.language
    manifest.voice_prompt = config.tts.voice_design_prompt
    manifest.profile = config.profile.value
    manifest.created_at = datetime.now(timezone.utc).isoformat()

    for chapter in chapters:
        ch_state = ChapterState(chapter.id, chapter.title, chapter.spine_index)
        ch_state.parse_status = Status.COMPLETED

        # Chunk the chapter text
        chunks = chunk_text(
            chapter.text,
            chapter.id,
            max_chars=config.tts.max_chunk_chars,
            max_tokens=config.tts.max_chunk_tokens,
        )
        ch_state.chunk_status = Status.COMPLETED

        for chunk in chunks:
            chunk_state = ChunkState(chunk.id, chunk.chapter_id, chunk.index)
            ch_state.chunks.append(chunk_state)

        manifest.chapters.append(ch_state)

    manifest.save()
    console.print(f"  Manifest saved: {manifest.manifest_path}")
    return manifest


def _run_tts_pipeline(config: AppConfig, manifest: Manifest) -> None:
    """Run TTS synthesis for all pending chunks."""
    client = QwenTTSClient(config)
    chunks_dir = config.output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    pending_chapters = manifest.get_pending_chapters()
    if not pending_chapters:
        console.print("\n[green]All chapters already synthesized.[/green]")
        return

    console.print(f"\n[bold]Synthesizing {len(pending_chapters)} chapters...[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        for ch_state in pending_chapters:
            pending_chunks = manifest.get_pending_chunks(ch_state)
            if not pending_chunks:
                ch_state.tts_status = Status.COMPLETED
                manifest.save()
                continue

            task = progress.add_task(
                f"  {ch_state.title[:40]}",
                total=len(pending_chunks),
            )
            ch_state.tts_status = Status.IN_PROGRESS
            manifest.save()

            # Re-chunk from manifest to get text (we need original text)
            # Load the chapter text from the EPUB again
            book = load_epub(Path(manifest.input_path))
            all_chapters = extract_chapters(book)
            chapter_text_map = {ch.id: ch.text for ch in all_chapters}

            chapter_text = chapter_text_map.get(ch_state.chapter_id, "")
            text_chunks = chunk_text(
                chapter_text,
                ch_state.chapter_id,
                max_chars=config.tts.max_chunk_chars,
                max_tokens=config.tts.max_chunk_tokens,
            )
            chunk_text_map = {c.id: c.text for c in text_chunks}

            for chunk_state in pending_chunks:
                text = chunk_text_map.get(chunk_state.chunk_id, "")
                if not text:
                    chunk_state.status = Status.SKIPPED
                    manifest.save()
                    progress.advance(task)
                    continue

                success = _synthesize_chunk_with_retry(
                    client, config, manifest, chunk_state, text, chunks_dir
                )
                progress.advance(task)

            # Check if all chunks done
            all_done = all(
                c.status in (Status.COMPLETED, Status.SKIPPED)
                for c in ch_state.chunks
            )
            if all_done:
                manifest.mark_chapter_tts_complete(ch_state)


def _synthesize_chunk_with_retry(
    client: QwenTTSClient,
    config: AppConfig,
    manifest: Manifest,
    chunk_state: ChunkState,
    text: str,
    chunks_dir: Path,
) -> bool:
    """Synthesize a single chunk with retry logic."""
    import soundfile as sf

    max_retries = config.retry.max_retries
    remaining = max_retries - chunk_state.attempts

    for attempt in range(remaining):
        try:
            audio = client.synthesize(text, config.tts, chunk_id=chunk_state.chunk_id)

            # Save chunk WAV
            output_path = chunks_dir / f"{chunk_state.chunk_id}.wav"
            sf.write(str(output_path), audio, config.tts.sample_rate)

            manifest.mark_chunk_complete(chunk_state, str(output_path))
            return True

        except TTSSynthesisError as e:
            logger.warning(f"Chunk {chunk_state.chunk_id} attempt {attempt + 1} failed: {e}")
            manifest.mark_chunk_failed(chunk_state, str(e))

            if attempt < remaining - 1:
                delay = _retry_delay(chunk_state.attempts, config)
                time.sleep(delay)
                chunk_state.status = Status.PENDING  # Reset for retry

    return False


def _run_audio_assembly(config: AppConfig, manifest: Manifest) -> None:
    """Assemble chapter WAVs from completed chunks."""
    chapters_dir = config.output_dir / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)

    completed_chapter_paths: list[Path] = []

    for ch_state in manifest.chapters:
        if ch_state.audio_status == Status.COMPLETED and ch_state.output_path:
            completed_chapter_paths.append(Path(ch_state.output_path))
            continue

        if ch_state.tts_status != Status.COMPLETED:
            continue

        # Gather chunk WAVs in order
        chunk_paths = []
        for chunk in sorted(ch_state.chunks, key=lambda c: c.index):
            if chunk.status == Status.COMPLETED and chunk.output_path:
                chunk_paths.append(Path(chunk.output_path))

        if not chunk_paths:
            continue

        # Concatenate into chapter WAV
        chapter_wav = chapters_dir / f"{ch_state.chapter_id}_{_safe_filename(ch_state.title)}.wav"
        try:
            concatenate_chunks(
                chunk_paths,
                chapter_wav,
                config.tts.sample_rate,
                config.audio.inter_chunk_silence_ms,
            )
            manifest.mark_chapter_audio_complete(ch_state, str(chapter_wav))
            completed_chapter_paths.append(chapter_wav)
        except Exception as e:
            logger.error(f"Failed to assemble chapter '{ch_state.title}': {e}")

    # Optional full-book merge
    if config.audio.merge_full_book and len(completed_chapter_paths) > 1:
        book_wav = config.output_dir / f"{_safe_filename(manifest.book_title)}_full.wav"
        try:
            concatenate_chapters(
                completed_chapter_paths,
                book_wav,
                config.tts.sample_rate,
                config.audio.inter_chapter_silence_ms,
            )
            console.print(f"\n[green]✓[/green] Full book: {book_wav}")
        except Exception as e:
            logger.error(f"Failed to merge full book: {e}")


def _safe_filename(text: str, max_len: int = 40) -> str:
    """Convert text to a safe filename."""
    import re
    safe = re.sub(r"[^\w\s-]", "", text)
    safe = re.sub(r"[\s]+", "_", safe)
    return safe[:max_len].strip("_").lower()


def _print_summary(manifest: Manifest) -> None:
    """Print conversion summary."""
    summary = manifest.summary()
    console.print("\n[bold]─── Summary ───[/bold]")
    console.print(f"  Chapters: {summary['completed_chapters']}/{summary['total_chapters']} completed")

    if summary["failed_chunks"] > 0:
        console.print(f"  [yellow]Failed chunks: {summary['failed_chunks']}[/yellow]")
        console.print(f"  Resume with: [cyan]epub-audiobook resume {manifest.output_dir}[/cyan]")
    else:
        console.print("  [green]All chunks synthesized successfully![/green]")

    console.print(f"  Output: {manifest.output_dir}")
