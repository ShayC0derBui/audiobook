"""WAV merge, silence insertion, and sample-rate consistency."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def generate_silence(duration_ms: int, sample_rate: int) -> np.ndarray:
    """Generate silence as a numpy array."""
    num_samples = int(sample_rate * duration_ms / 1000)
    return np.zeros(num_samples, dtype=np.float32)


def validate_audio_file(path: Path, expected_sr: int) -> tuple[np.ndarray, int]:
    """Load and validate an audio file. Returns (data, sample_rate)."""
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        # Convert to mono by averaging channels
        data = data.mean(axis=1)
    if sr != expected_sr:
        logger.warning(
            f"Sample rate mismatch in {path.name}: got {sr}, expected {expected_sr}. Resampling."
        )
        data = _resample(data, sr, expected_sr)
    return data, expected_sr


def _resample(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear interpolation resampling."""
    duration = len(data) / orig_sr
    target_len = int(duration * target_sr)
    indices = np.linspace(0, len(data) - 1, target_len)
    return np.interp(indices, np.arange(len(data)), data).astype(np.float32)


def concatenate_chunks(
    chunk_paths: list[Path],
    output_path: Path,
    sample_rate: int,
    inter_chunk_silence_ms: int = 300,
) -> Path:
    """Concatenate chunk WAV files into a single chapter WAV."""
    segments: list[np.ndarray] = []
    silence = generate_silence(inter_chunk_silence_ms, sample_rate)

    for i, chunk_path in enumerate(chunk_paths):
        data, _ = validate_audio_file(chunk_path, sample_rate)
        segments.append(data)
        if i < len(chunk_paths) - 1:
            segments.append(silence)

    if not segments:
        raise ValueError("No audio segments to concatenate")

    combined = np.concatenate(segments)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), combined, sample_rate)

    duration_sec = len(combined) / sample_rate
    logger.info(f"Chapter WAV: {output_path.name} ({duration_sec:.1f}s)")
    return output_path


def concatenate_chapters(
    chapter_paths: list[Path],
    output_path: Path,
    sample_rate: int,
    inter_chapter_silence_ms: int = 2000,
) -> Path:
    """Concatenate chapter WAV files into a full-book WAV."""
    segments: list[np.ndarray] = []
    silence = generate_silence(inter_chapter_silence_ms, sample_rate)

    for i, chapter_path in enumerate(chapter_paths):
        data, _ = validate_audio_file(chapter_path, sample_rate)
        segments.append(data)
        if i < len(chapter_paths) - 1:
            segments.append(silence)

    if not segments:
        raise ValueError("No chapter audio to concatenate")

    combined = np.concatenate(segments)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), combined, sample_rate)

    duration_sec = len(combined) / sample_rate
    logger.info(f"Full book WAV: {output_path.name} ({duration_sec:.1f}s)")
    return output_path
