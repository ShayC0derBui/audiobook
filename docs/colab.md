# Google Colab Setup

Run epub-audiobook in Google Colab with GPU acceleration.

## Setup Cell

```python
# Install epub-audiobook in Colab
!pip install git+https://github.com/youruser/epub-audiobook.git#egg=epub-audiobook[colab]

# Or from local upload
!pip install -e "./epub-audiobook[colab]"
```

## Verify GPU

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Upload EPUB

```python
from google.colab import files
uploaded = files.upload()  # Upload your .epub file
```

## Convert

```python
!epub-audiobook convert uploaded_book.epub --profile colab --output ./audiobook -l en
```

## Resume (if runtime disconnects)

```python
# Re-mount drive or re-upload, then:
!epub-audiobook resume ./audiobook --profile colab
```

## Download Results

```python
import shutil
shutil.make_archive("audiobook", "zip", "./audiobook/chapters")
files.download("audiobook.zip")
```

## Tips

- Use Colab Pro for longer runtimes and better GPUs
- Save output to Google Drive to persist across sessions:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  !epub-audiobook convert book.epub --output /content/drive/MyDrive/audiobook
  ```
- Float16 is used by default for faster inference on CUDA
- T4 GPU handles most books; A100 recommended for very long books
