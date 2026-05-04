# EPUB Audiobook CLI

Convert EPUB books to audiobooks using **Qwen3-TTS VoiceDesign** — chapter-by-chapter with full resume support.

## Features

- **Spine-ordered EPUB parsing** with TOC-enhanced chapter titles
- **Sentence-aware chunking** respecting character/token limits
- **Qwen3-TTS VoiceDesign** for consistent narrator identity
- **Resume support** — interrupt and resume without re-processing completed chunks
- **Per-chapter WAV output** with optional full-book merge
- **Two runtime profiles**: Google Colab (CUDA) and Apple Silicon (MPS)

## Installation

```bash
# Base install
pip install -e .

# With Apple Silicon support
pip install -e ".[apple-silicon]"

# With Colab support
pip install -e ".[colab]"

# Development
pip install -e ".[dev]"
```

## Quick Start

```bash
# Test TTS setup
epub-audiobook test-tts --text "Hello world" --output test.wav

# Inspect an EPUB
epub-audiobook inspect path/to/book.epub

# Convert full book
epub-audiobook convert path/to/book.epub --output ./my_audiobook

# Convert specific chapters
epub-audiobook convert path/to/book.epub -c 1-5 --output ./my_audiobook

# Resume interrupted conversion
epub-audiobook resume ./my_audiobook

# Merge into single file
epub-audiobook convert path/to/book.epub --merge-book --output ./my_audiobook
```

## Commands

| Command | Description |
|---------|-------------|
| `convert` | Convert EPUB to audiobook WAV files |
| `resume` | Resume an interrupted conversion |
| `inspect` | Display EPUB chapter structure |
| `test-tts` | Test TTS with a short sample |

## Options

```
--profile, -p     Runtime profile: colab or apple-silicon (auto-detected)
--language, -l    Book language code (default: en)
--voice-prompt    VoiceDesign narrator description
--chapters, -c    Chapter range (e.g., "1-5" or "3")
--model, -m       Qwen3-TTS model path or HuggingFace ID
--merge-book      Merge all chapters into a single WAV
--output, -o      Output directory
--verbose, -v     Enable debug logging
```

## Runtime Profiles

### Apple Silicon (default on M1/M2/M3 Macs)
- Uses MPS backend
- Float32 precision
- See [docs/apple-silicon.md](docs/apple-silicon.md)

### Google Colab
- Uses CUDA backend
- Float16 precision
- See [docs/colab.md](docs/colab.md)

## Output Structure

```
output/
├── manifest.json          # Run state for resume
├── chunks/                # Per-chunk WAV files
│   ├── <chapter_id>_c0000.wav
│   └── ...
├── chapters/              # Assembled chapter WAVs
│   ├── <chapter_id>_chapter_title.wav
│   └── ...
└── book_title_full.wav    # Optional merged book
```

## VoiceDesign Prompts

Control narrator style with descriptive prompts:

```bash
# Calm narrator
epub-audiobook convert book.epub --voice-prompt "A calm, warm adult male narrator with measured pacing"

# Energetic narrator
epub-audiobook convert book.epub --voice-prompt "An energetic young female narrator with clear articulation"
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
