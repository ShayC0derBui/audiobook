# Apple Silicon Setup

Run epub-audiobook locally on Apple M1/M2/M3 Macs using MPS acceleration.

## Requirements

- macOS 12.3+ (Monterey or later)
- Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- ~8GB free RAM (model dependent)

## Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with Apple Silicon profile
pip install -e ".[apple-silicon]"
```

## Verify Setup

```bash
epub-audiobook test-tts --text "Testing Apple Silicon TTS" --output test.wav
```

Expected output:
```
✓ Runtime ready
  Profile: apple-silicon
  Device: mps
  Dtype: float32
✓ Saved: test.wav
```

## Usage

```bash
# The profile is auto-detected on Apple Silicon
epub-audiobook convert my_book.epub --output ./audiobook

# Explicit profile selection
epub-audiobook convert my_book.epub -p apple-silicon --output ./audiobook
```

## Performance Notes

- MPS backend uses float32 (float16 not fully supported on MPS)
- First chunk is slower due to model loading and MPS compilation
- Subsequent chunks benefit from MPS shader caching
- Memory usage scales with chunk size — keep max_chunk_chars ≤ 500

## Troubleshooting

**MPS not available**: Ensure macOS 12.3+ and PyTorch 2.1+
```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

**Out of memory**: Reduce chunk sizes or close other applications.
