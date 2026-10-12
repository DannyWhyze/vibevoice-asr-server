# Utilities Documentation: `split_audio.py`

## Overview
`split_audio.py` is a preprocessing utility designed to partition large audio recordings (e.g., podcasts, conference calls, long interviews) into manageable, fixed-duration chunks (default: 10 minutes) before passing them to the transcription pipeline.

---

## Purpose in the Project

The upstream model `microsoft/VibeVoice-ASR-HF` faces context limitations during long-form transcription:
* The text decoder (`Qwen2`) operates within a hard context ceiling (32k tokens).
* Because the model outputs verbose JSON (`Speaker`, `Start`, `End`, `Content` per sentence), long audio files (> 20–30 minutes) regularly trigger `Unterminated string` errors or enter decoder degeneration loops.
* `split_audio.py` ensures that all audio inputs remain well below this threshold, producing safe, predictable transcription chunks.

---

## Technical Features

1. **Streaming Processing via `soundfile.blocks`:**
   - The file is not loaded entirely into system RAM.
   - It reads and writes memory-efficient sample blocks chunk-by-chunk using `sf.blocks(input_file, blocksize=chunk_samples)`.
2. **Automatic Metadata Inspection:**
   - Detects input sample rate (`samplerate`) and total duration automatically using `sf.info()`.
   - Computes exact sample boundaries (`chunk_samples = int(chunk_minutes * 60 * sr)`).
3. **Structured Export:**
   - Automatically generates an output directory named `<audio_basename>_chunks/`.
   - Exports sequential uncompressed `.wav` files named `<audio_basename>_part001.wav`, `<audio_basename>_part002.wav`, etc.

---

## Requirements

Ensure `soundfile` is installed in your virtual environment:

```powershell
pip install soundfile
# or via uv:
uv pip install soundfile
```

*Note: For MP3 decoding, ensure `libsndfile` or `ffmpeg` is available on the host system.*

---

## Usage

```powershell
# Split an audio file into default 10-minute chunks
python utils/split_audio.py "path/to/recording.mp3"

# Specify custom chunk duration (e.g., 5-minute chunks)
python utils/split_audio.py "path/to/recording.mp3" --minutes 5.0
```

### Arguments

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_file` | `str` (Positional) | *Required* | Path to the input audio file (`.mp3`, `.wav`, etc.) |
| `--minutes` | `float` (Optional) | `10.0` | Target chunk length in minutes |
