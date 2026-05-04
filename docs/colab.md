# Google Colab Setup

Run epub-audiobook in Google Colab with GPU acceleration.

## Quick Start (One Click)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShayC0derBui/audiobook/blob/main/notebooks/test_tts_colab.ipynb)

## Manual Setup

```python
# Install epub-audiobook from GitHub
!pip install -q "git+https://github.com/ShayC0derBui/audiobook.git#egg=epub-audiobook[colab]"
```

## Verify GPU

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Test TTS

```python
!epub-audiobook test-tts \
    --text "Hello, this is a test." \
    --output /content/test.wav \
    --profile colab

from IPython.display import Audio
Audio('/content/test.wav')
```

## Upload & Convert EPUB

```python
from google.colab import files
uploaded = files.upload()  # Upload your .epub file

!epub-audiobook convert uploaded_book.epub --profile colab --output /content/audiobook -l en
```

## Resume (if runtime disconnects)

```python
!epub-audiobook resume /content/audiobook --profile colab
```

## Download Results

```python
import shutil
shutil.make_archive("/content/audiobook", "zip", "/content/audiobook/chapters")
files.download("/content/audiobook.zip")
```

## Using Colab MCP (Agent-Driven Testing)

You can also use [Google Colab MCP](https://github.com/googlecolab/colab-mcp) to let your local AI agent run code directly in a Colab session.

### Setup Colab MCP

Add to your VS Code `settings.json` or MCP configuration:

```json
{
  "mcpServers": {
    "colab-mcp": {
      "command": "uvx",
      "args": ["git+https://github.com/googlecolab/colab-mcp"],
      "timeout": 30000
    }
  }
}
```

Prerequisites:
- Install `uv`: `pip install uv`
- Have a Colab session open in your browser with GPU runtime

## Tips

- Use Colab Pro for longer runtimes and better GPUs
- Save output to Google Drive to persist across sessions:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  !epub-audiobook convert book.epub --output /content/drive/MyDrive/audiobook
  ```
- bfloat16 is used by default for faster inference on CUDA
- T4 GPU handles most books; A100 recommended for very long books
- Flash Attention 2 is auto-enabled when `flash-attn` is installed (included in `[colab]` extras)
