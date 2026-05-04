"""Qwen3-TTS model initialization and VoiceDesign synthesis.

Uses the official `qwen-tts` package (pip install qwen-tts).
Supports VoiceDesign mode for narrator identity generation,
and CustomVoice mode (0.6B preset speakers) for faster inference.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from epub_audiobook.config import AppConfig, TTSConfig

logger = logging.getLogger(__name__)

# Model IDs on HuggingFace
VOICE_DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
CUSTOM_VOICE_MODEL_FAST = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"  # ~3x faster than 1.7B

# Default preset speaker for CustomVoice fast mode (English narrator)
DEFAULT_FAST_SPEAKER = "Ryan"

# Language code mapping from our short codes to Qwen3-TTS expected values
LANGUAGE_MAP = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
    "auto": "Auto",
}


class TTSSynthesisError(Exception):
    """Raised when TTS synthesis fails."""

    def __init__(self, message: str, chunk_id: str | None = None) -> None:
        self.chunk_id = chunk_id
        super().__init__(message)


class QwenTTSClient:
    """Client for Qwen3-TTS VoiceDesign synthesis using qwen-tts package."""

    def __init__(self, config: "AppConfig") -> None:
        self.config = config
        self.model = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy-load model on first use."""
        if self._loaded:
            return

        import torch
        from qwen_tts import Qwen3TTSModel

        model_id = self.config.model_path or VOICE_DESIGN_MODEL
        device = self.config.get_torch_device()
        dtype_str = self.config.get_torch_dtype()

        # Map dtype string to torch dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map.get(dtype_str, torch.float32)

        logger.info(f"Loading Qwen3-TTS model: {model_id}")
        logger.info(f"  Device: {device}, Dtype: {dtype}")
        logger.info("  (First run downloads ~3.8GB of model weights)")

        load_kwargs: dict = {"dtype": dtype}

        if device == "cuda":
            load_kwargs["device_map"] = "cuda:0"
            # Flash Attention 2 requires float16/bfloat16 and reduces VRAM + speeds up inference
            if dtype in (torch.float16, torch.bfloat16):
                try:
                    import flash_attn  # noqa: F401
                    load_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("  Flash Attention 2 enabled")
                except ImportError:
                    logger.info("  flash-attn not installed; run: pip install flash-attn --no-build-isolation")

        self.model = Qwen3TTSModel.from_pretrained(model_id, **load_kwargs)

        # torch.compile speeds up inference ~2x on CUDA (PyTorch 2.x)
        if device == "cuda":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("  torch.compile applied (reduce-overhead mode)")
            except Exception:
                pass  # Non-fatal: compile is a best-effort optimization

        self._loaded = True
        logger.info("Model loaded successfully")

    def _resolve_language(self, lang_code: str) -> str:
        """Convert short language code to Qwen3-TTS language name."""
        return LANGUAGE_MAP.get(lang_code.lower(), lang_code)

    def _is_fast_mode(self) -> bool:
        """True if using a CustomVoice model (preset speakers, faster inference)."""
        model_id = self.config.model_path or VOICE_DESIGN_MODEL
        return "CustomVoice" in model_id

    def synthesize(
        self,
        text: str,
        tts_config: "TTSConfig",
        chunk_id: str | None = None,
    ) -> np.ndarray:
        """Synthesize text to audio.

        Uses VoiceDesign (instruct-based) or CustomVoice (preset speaker)
        depending on the loaded model. Returns numpy array of audio samples.
        """
        self._ensure_loaded()

        language = self._resolve_language(tts_config.language)

        try:
            if self._is_fast_mode():
                speaker = getattr(tts_config, "speaker", DEFAULT_FAST_SPEAKER)
                wavs, sr = self.model.generate_custom_voice(
                    text=text,
                    language=language,
                    speaker=speaker,
                )
            else:
                wavs, sr = self.model.generate_voice_design(
                    text=text,
                    language=language,
                    instruct=tts_config.voice_design_prompt,
                )

            audio = wavs[0]

            if not isinstance(audio, np.ndarray):
                audio = np.array(audio, dtype=np.float32)

            if audio.ndim == 0 or len(audio) == 0:
                raise TTSSynthesisError(
                    f"Empty audio output for text: {text[:50]}...",
                    chunk_id=chunk_id,
                )

            self._last_sample_rate = sr
            return audio

        except TTSSynthesisError:
            raise
        except Exception as e:
            raise TTSSynthesisError(
                f"TTS synthesis failed: {e}",
                chunk_id=chunk_id,
            ) from e

    def synthesize_batch(
        self,
        texts: list[str],
        tts_config: "TTSConfig",
    ) -> list[np.ndarray]:
        """Batch synthesize multiple texts in a single model forward pass.

        More efficient than calling synthesize() in a loop.
        """
        self._ensure_loaded()

        language = self._resolve_language(tts_config.language)

        try:
            if self._is_fast_mode():
                speaker = getattr(tts_config, "speaker", DEFAULT_FAST_SPEAKER)
                wavs, sr = self.model.generate_custom_voice(
                    text=texts,
                    language=[language] * len(texts),
                    speaker=[speaker] * len(texts),
                )
            else:
                wavs, sr = self.model.generate_voice_design(
                    text=texts,
                    language=[language] * len(texts),
                    instruct=[tts_config.voice_design_prompt] * len(texts),
                )

            self._last_sample_rate = sr
            result = []
            for wav in wavs:
                audio = wav if isinstance(wav, np.ndarray) else np.array(wav, dtype=np.float32)
                result.append(audio)
            return result

        except Exception as e:
            raise TTSSynthesisError(f"Batch TTS synthesis failed: {e}") from e

    @property
    def sample_rate(self) -> int | None:
        """Return sample rate from last synthesis, or None if not yet synthesized."""
        return getattr(self, "_last_sample_rate", None)
