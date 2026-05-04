"""CLI entrypoints for epub-audiobook."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

from epub_audiobook import __version__
from epub_audiobook.config import AppConfig, RuntimeProfile, check_runtime_readiness

app = typer.Typer(
    name="epub-audiobook",
    help="Convert EPUB books to audiobooks using Qwen3-TTS VoiceDesign.",
    no_args_is_help=True,
)
console = Console()


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """EPUB Audiobook CLI — convert books to speech with Qwen3-TTS."""
    _setup_logging(verbose)


@app.command()
def convert(
    input_path: Path = typer.Argument(..., help="Path to the EPUB file."),
    output_dir: Path = typer.Option(
        Path("./output"), "--output", "-o", help="Output directory."
    ),
    profile: Optional[str] = typer.Option(
        None, "--profile", "-p", help="Runtime profile: colab or apple-silicon."
    ),
    language: str = typer.Option("en", "--language", "-l", help="Book language code."),
    voice_prompt: Optional[str] = typer.Option(
        None, "--voice-prompt", help="VoiceDesign narrator prompt."
    ),
    fast: bool = typer.Option(
        False, "--fast", help="Use 0.6B CustomVoice model (~3x faster, preset speakers)."
    ),
    speaker: str = typer.Option(
        "Ryan", "--speaker", help="Preset speaker for --fast mode (e.g. Ryan, Aiden, Vivian)."
    ),
    chapters: Optional[str] = typer.Option(
        None, "--chapters", "-c", help="Chapter range (e.g., '1-5')."
    ),
    model_path: Optional[str] = typer.Option(
        None, "--model", "-m", help="Qwen3-TTS model path or HuggingFace ID."
    ),
    merge_book: bool = typer.Option(
        False, "--merge-book", help="Merge all chapters into a single WAV."
    ),
) -> None:
    """Convert an EPUB file to audiobook WAV files."""
    from epub_audiobook.pipeline import run_convert
    from epub_audiobook.tts.qwen_client import CUSTOM_VOICE_MODEL_FAST

    detected_profile = RuntimeProfile(profile) if profile else AppConfig.detect_profile()

    # --fast overrides model to the 0.6B CustomVoice model
    resolved_model = model_path
    if fast and not model_path:
        resolved_model = CUSTOM_VOICE_MODEL_FAST

    config = AppConfig(
        profile=detected_profile,
        input_path=input_path,
        output_dir=output_dir,
        chapters=chapters,
        model_path=resolved_model,
    )
    if voice_prompt:
        config.tts.voice_design_prompt = voice_prompt
    config.tts.language = language
    config.tts.speaker = speaker
    config.audio.merge_full_book = merge_book

    run_convert(config)


@app.command()
def resume(
    output_dir: Path = typer.Argument(
        ..., help="Output directory from a previous run."
    ),
    profile: Optional[str] = typer.Option(
        None, "--profile", "-p", help="Runtime profile override."
    ),
) -> None:
    """Resume an interrupted conversion from its manifest."""
    from epub_audiobook.pipeline import run_resume

    detected_profile = RuntimeProfile(profile) if profile else AppConfig.detect_profile()
    run_resume(output_dir, detected_profile)


@app.command()
def inspect(
    input_path: Path = typer.Argument(..., help="Path to the EPUB file."),
) -> None:
    """Inspect an EPUB and display chapter structure."""
    from epub_audiobook.epub.parser import load_epub
    from epub_audiobook.epub.extractor import extract_chapters

    book = load_epub(input_path)
    chapters = extract_chapters(book)

    console.print(f"\n[bold]EPUB:[/bold] {input_path.name}")
    console.print(f"[bold]Chapters found:[/bold] {len(chapters)}\n")

    for i, ch in enumerate(chapters, 1):
        preview = ch.text[:80] + "..." if len(ch.text) > 80 else ch.text
        console.print(f"  {i:3d}. [cyan]{ch.title}[/cyan] ({len(ch.text)} chars)")
        console.print(f"       {preview}\n")


@app.command(name="test-tts")
def test_tts(
    text: str = typer.Option(
        "Hello, this is a test of the text to speech system.",
        "--text",
        "-t",
        help="Text to synthesize.",
    ),
    output: Path = typer.Option(
        Path("./test_output.wav"), "--output", "-o", help="Output WAV path."
    ),
    profile: Optional[str] = typer.Option(
        None, "--profile", "-p", help="Runtime profile."
    ),
    voice_prompt: Optional[str] = typer.Option(
        None, "--voice-prompt", help="VoiceDesign prompt."
    ),
    fast: bool = typer.Option(
        False, "--fast", help="Use 0.6B CustomVoice model (~3x faster)."
    ),
    speaker: str = typer.Option(
        "Ryan", "--speaker", help="Preset speaker for --fast mode."
    ),
    model_path: Optional[str] = typer.Option(
        None, "--model", "-m", help="Qwen3-TTS model path."
    ),
) -> None:
    """Test TTS synthesis with a short text sample."""
    from epub_audiobook.config import TTSConfig
    from epub_audiobook.tts.qwen_client import QwenTTSClient, CUSTOM_VOICE_MODEL_FAST

    detected_profile = RuntimeProfile(profile) if profile else AppConfig.detect_profile()

    tts_config = TTSConfig()
    if voice_prompt:
        tts_config.voice_design_prompt = voice_prompt
    tts_config.speaker = speaker

    resolved_model = model_path
    if fast and not model_path:
        resolved_model = CUSTOM_VOICE_MODEL_FAST

    config = AppConfig(
        profile=detected_profile,
        input_path=Path("."),  # dummy
        model_path=resolved_model,
    )

    # Check runtime
    issues = check_runtime_readiness(config)
    if issues:
        for issue in issues:
            console.print(f"[red]✗[/red] {issue}")
        raise typer.Exit(1)

    mode = "CustomVoice/fast (0.6B)" if fast else "VoiceDesign (1.7B)"
    console.print("[green]✓[/green] Runtime ready")
    console.print(f"  Profile: {detected_profile.value}")
    console.print(f"  Device: {config.get_torch_device()}")
    console.print(f"  Dtype: {config.get_torch_dtype()}")
    console.print(f"  Mode: {mode}")

    client = QwenTTSClient(config)
    audio_data = client.synthesize(text, tts_config)

    sr = client.sample_rate or tts_config.sample_rate

    import soundfile as sf

    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), audio_data, sr)
    console.print(f"\n[green]✓[/green] Saved: {output} ({len(audio_data)/sr:.1f}s)")


if __name__ == "__main__":
    app()
