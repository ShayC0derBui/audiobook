"""Qwen3-TTS model initialization and VoiceDesign synthesis."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from epub_audiobook.config import AppConfig, TTSConfig

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS"


class TTSSynthesisError(Exception):
    """Raised when TTS synthesis fails."""

    def __init__(self, message: str, chunk_id: str | None = None) -> None:
        self.chunk_id = chunk_id
        super().__init__(message)


class QwenTTSClient:
    """Client for Qwen3-TTS VoiceDesign synthesis."""

    def __init__(self, config: "AppConfig") -> None:
        self.config = config
        self.model = None
        self.processor = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy-load model and processor on first use."""
        if self._loaded:
            return

        import torch
        from transformers import AutoProcessor, AutoModelForTextToWaveform

        model_id = self.config.model_path or DEFAULT_MODEL_ID
        device = self.config.get_torch_device()
        dtype_str = self.config.get_torch_dtype()
        dtype = torch.float32 if dtype_str == "float32" else torch.float16

        logger.info(f"Loading Qwen3-TTS model: {model_id}")
        logger.info(f"  Device: {device}, Dtype: {dtype}")

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForTextToWaveform.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)

        self._loaded = True
        logger.info("Model loaded successfully")

    def synthesize(
        self,
        text: str,
        tts_config: "TTSConfig",
        chunk_id: str | None = None,
    ) -> np.ndarray:
        """Synthesize text to audio using VoiceDesign mode.

        Returns numpy array of audio samples at the configured sample rate.
        """
        import torch

        self._ensure_loaded()

        device = self.config.get_torch_device()

        # Build VoiceDesign prompt following Qwen3-TTS format
        voice_prompt = tts_config.voice_design_prompt
        language = tts_config.language

        try:
            # Construct the input in Qwen3-TTS VoiceDesign format
            inputs = self.processor(
                text=text,
                voice_design_prompt=voice_prompt,
                language=language,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = self.model.generate(**inputs)

            # Extract audio waveform
            audio = outputs.cpu().numpy().squeeze()

            if audio.ndim == 0 or len(audio) == 0:
                raise TTSSynthesisError(
                    f"Empty audio output for text: {text[:50]}...",
                    chunk_id=chunk_id,
                )

            return audio

        except TTSSynthesisError:
            raise
        except Exception as e:
            raise TTSSynthesisError(
                f"TTS synthesis failed: {e}",
                chunk_id=chunk_id,
            ) from e
