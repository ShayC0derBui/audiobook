"""Typed configuration model and runtime profile handling."""

from __future__ import annotations

import platform
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class RuntimeProfile(str, Enum):
    """Supported execution environments."""

    COLAB = "colab"
    APPLE_SILICON = "apple-silicon"


class TTSConfig(BaseModel):
    """TTS generation settings."""

    voice_design_prompt: str = Field(
        default="A calm, clear adult narrator with natural pacing and warm tone.",
        description="VoiceDesign instruction for consistent narrator identity.",
    )
    language: str = Field(
        default="en",
        description="Fixed language code for the entire book.",
    )
    sample_rate: int = Field(default=24000, description="Output sample rate in Hz.")
    max_chunk_chars: int = Field(
        default=500, description="Maximum characters per TTS chunk."
    )
    max_chunk_tokens: int = Field(
        default=250, description="Maximum tokens per TTS chunk (soft ceiling)."
    )


class AudioConfig(BaseModel):
    """Audio assembly settings."""

    inter_chunk_silence_ms: int = Field(
        default=300, description="Silence between chunks in milliseconds."
    )
    inter_chapter_silence_ms: int = Field(
        default=2000, description="Silence between chapters in milliseconds."
    )
    merge_full_book: bool = Field(
        default=False, description="Whether to produce a full-book WAV."
    )


class RetryConfig(BaseModel):
    """Retry policy for failed chunks."""

    max_retries: int = Field(default=3, description="Maximum retry attempts per chunk.")
    base_delay_seconds: float = Field(
        default=2.0, description="Base delay for exponential backoff."
    )
    max_delay_seconds: float = Field(
        default=60.0, description="Maximum backoff delay."
    )


class AppConfig(BaseModel):
    """Top-level application configuration."""

    profile: RuntimeProfile = Field(
        default=RuntimeProfile.APPLE_SILICON,
        description="Runtime execution profile.",
    )
    input_path: Path = Field(description="Path to input EPUB file.")
    output_dir: Path = Field(
        default=Path("./output"), description="Directory for output files."
    )
    chapters: Optional[str] = Field(
        default=None,
        description="Chapter range to process (e.g., '1-5' or '3'). None means all.",
    )
    model_path: Optional[str] = Field(
        default=None,
        description="Path or HuggingFace ID for the Qwen3-TTS model.",
    )
    tts: TTSConfig = Field(default_factory=TTSConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)

    @classmethod
    def detect_profile(cls) -> RuntimeProfile:
        """Auto-detect the best runtime profile for current system."""
        machine = platform.machine().lower()
        system = platform.system().lower()
        if system == "darwin" and machine in ("arm64", "aarch64"):
            return RuntimeProfile.APPLE_SILICON
        # Default to colab for non-Apple-Silicon systems
        return RuntimeProfile.COLAB

    def get_torch_device(self) -> str:
        """Return the appropriate torch device string for the profile."""
        if self.profile == RuntimeProfile.APPLE_SILICON:
            return "mps"
        return "cuda"

    def get_torch_dtype(self) -> str:
        """Return the appropriate torch dtype for the profile."""
        if self.profile == RuntimeProfile.APPLE_SILICON:
            return "float32"
        return "float16"


def check_runtime_readiness(config: AppConfig) -> list[str]:
    """Check if the runtime environment is ready. Returns list of issues."""
    issues: list[str] = []

    try:
        import torch  # noqa: F401
    except ImportError:
        issues.append("PyTorch not installed. Install with profile extras.")
        return issues

    import torch

    device = config.get_torch_device()
    if device == "mps" and not torch.backends.mps.is_available():
        issues.append("MPS (Apple Silicon GPU) not available on this system.")
    elif device == "cuda" and not torch.cuda.is_available():
        issues.append("CUDA not available. Check GPU drivers and torch installation.")

    try:
        import transformers  # noqa: F401
    except ImportError:
        issues.append("transformers library not installed.")

    return issues
