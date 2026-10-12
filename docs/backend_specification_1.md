# Backend Specification 1: VibeVoice ASR Server & Testing Baseline

> Living specification of backend architecture, modules, REST API endpoints, and verification suite (Rules 4, 12 & 39).

## 1. System Architecture Overview

The system provides local, high-performance transcription using Microsoft's `microsoft/VibeVoice-ASR-HF` model. It consists of:
- **FastAPI HTTP Service:** `legacy_scripts/server_failed.py` exposes `/transcribe` with BitsAndBytes 8-bit quantization and FlashAttention-2.
- **Local CLI Runner:** `legacy_scripts/test_local.py` provides scriptable CLI execution (`--audio`, `--minutes`, `--output`) with Windows file-descriptor safety and fallback handling.
- **Audio Preprocessing Utilities:** `utils/split_audio.py` for chunking large audio files; `utils/download_model.py` for model acquisition and license bookkeeping.
- **Verification Baseline:** Deterministic CPU-only unit test suite in `tests/test_audio_processing.py`.

## 2. Dependencies & Runtime Environment

Managed via `uv` with Python `>=3.10`.
Key production dependencies:
- `fastapi`, `uvicorn[standard]`, `python-multipart`
- `torch`, `torchaudio` (CUDA 12.8 wheel via `tool.uv.sources`)
- `transformers`, `accelerate`, `bitsandbytes`
- `librosa`, `soundfile`
- `pytest` (test baseline)

## 3. Algorithmic Fixes & Verification

### 3.1 3,200-Sample Attention Mask Padding
When raw audio length is an exact multiple of 3,200 (hop length), convolutional downsampling produces $N + 1$ feature frames while naive integer division reserves only $N$ mask frames, causing an `IndexError` during feature indexing.
- **Workaround:** If `padding_mask.shape[-1] % 3200 == 0`, append 3,200 ones along dimension `-1`.
- **Unit Test:** `tests/test_audio_processing.py::test_padding_mask_fix_applied_when_multiple_of_3200`.

### 3.2 Streaming Offset Accumulation
Long audio is processed in sequential chunks. Segments must offset their `Start` and `End` timestamps by the cumulative `time_offset`.
- **Unit Test:** `tests/test_audio_processing.py::test_timestamp_offset_math`.

### 3.3 Structured JSON Decode Fallback
When autoregressive generation exceeds token boundaries or outputs malformed JSON, `processor.decode(return_format="parsed")` fails with `ValueError` / `JSONDecodeError`.
- **Workaround:** Wrapped in `try-except` falling back to `return_format="text"` with an envelope `[{"Start": 0.0, "End": duration, "Speaker": "Unknown", "Content": text}]`.
- **Unit Test:** `tests/test_audio_processing.py::test_decode_chunk_with_fallback_on_json_error`.

## 4. Local CLI Runner Usage

```bash
uv run python legacy_scripts/test_local.py --audio "path/to/audio.wav" --minutes 10 --output "transcript.json"
```
Options:
- `--audio`, `-a`: Path to audio file (required). Validated before model initialization.
- `--minutes`, `-m`: Chunk duration in minutes (default: 10).
## 5. FastAPI Service & Concurrency Model

### 5.1 Route: `POST /transcribe`
Accepts multipart audio upload and runs local ASR inference.

**Security Constraints (Rules 9 & 21):**
- Rejects filenames containing `..` or path traversal with HTTP 400.
- Allowed extensions: `.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`.
- Enforces 500 MB max file size limit (HTTP 413).
- Writes temporary uploads using randomized UUIDs inside `temp_audio/`, validated against `.is_relative_to()`.

**Concurrency & Hardware Constraints (Rules 24 & 25):**
- Single-GPU execution serialized via `asyncio.Lock()` to prevent concurrent VRAM exhaustion.
- Heavy PyTorch inference offloaded from the asyncio event loop using `asyncio.to_thread(_run_inference, ...)`.
- Wrapped in `torch.inference_mode()`.

**Response Schema (`TranscribeResponse`):**
```json
{
  "filename": "sample.wav",
  "status": "success",
  "is_fallback": false,
  "warning": null,
  "transcription": [
    {
      "Start": 0.0,
      "End": 4.5,
      "Speaker": "1",
      "Content": "Hello world"
    }
  ]
}
```


