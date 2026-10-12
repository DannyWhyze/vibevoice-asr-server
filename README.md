# VibeVoice ASR server

FastAPI server for `microsoft/VibeVoice-ASR-HF`.

## 1. Local development

### Build image
Build the container image:
```bash
podman build -t vibevoice_asr_server .
```

### Start server
Run the container with a bind mount for hot-reloading:
```bash
podman run --name vibevoice-asr-server -d -p 8000:8000 --device nvidia.com/gpu=all -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" -v "./legacy_scripts/server_failed.py:/app/server.py" vibevoice_asr_server
```
Restart the container after changing `server.py` on the host:
```bash
podman restart vibevoice-asr-server
```

### Monitor logs
Model weights load into VRAM on startup. Watch progress with:
```bash
podman logs -f vibevoice-asr-server
```

### Run tests
Run the CPU unit test baseline:
```bash
uv run pytest tests/
```

### API docs
Swagger docs run at:
**http://localhost:8000/docs**

Upload audio directly to `/transcribe` to test.

---

## 2. Production deployment

To run this in production, set up:

1. **Auto-restart.** Pass `--restart unless-stopped` to recover from host reboots or crashes.
2. **Reverse proxy.** Put Nginx or Traefik in front for TLS. Do not expose port 8000 directly.
3. **Auth.** Check API tokens in FastAPI to block unmetered GPU use.
4. **Queue.** Queue jobs sequentially (Celery, RQ). Concurrent runs will OOM a 16 GB card.

---

## 3. Hugging Face cache

The server sets `MODEL_NAME = "microsoft/VibeVoice-ASR-HF"` and uses the hub cache:

1. **Portability.** The container pulls weights on first start in a new environment without hardcoded host paths.
2. **Version control.** Weights cache by commit hash. Upstream updates only download changed layers.
3. **Symlinks.** Shared tensors across revisions reuse disk blocks.

---

## 4. Utility scripts

Helper scripts located in `utils/` for audio preprocessing and weight acquisition:

### Split audio (`split_audio.py`)
Cuts large recordings into fixed-duration WAV blocks (default: 10 minutes) to prevent autoregressive token limit crashes:
```powershell
python utils/split_audio.py "path/to/audio.mp3" --minutes 10
```
Sequential uncompressed `.wav` blocks are written to `<name>_chunks/`.

### Download model weights (`download_model.py`)
Downloads model snapshots from Hugging Face and records commit sha and license metadata to `download_info.json`:
```powershell
python utils/download_model.py microsoft/VibeVoice-ASR-HF
```


---

## 5. Known limits and bugs

The upstream model struggles on long audio out of the box. Below is what works and what broke during testing.

### What works
* Speaker diarization works. The model labels speakers and outputs timestamps.
* FlashAttention-2 runs inside the container and cuts prefill VRAM enough to avoid startup OOMs.

### What fails

1. **The 60-minute claim hits a token wall.**
   Microsoft advertises 60-minute audio support. The acoustic encoder accepts that length, but the text decoder (Qwen2) caps out at 32,768 tokens. Because the output JSON is wordy (repeating `Speaker`, `Start`, and `End` for each sentence), a 37-minute podcast produces over 80,000 characters. Generation cuts off mid-string, `processor.decode` crashes with an unclosed JSON error (`Unterminated string starting at...`), and all inference work is lost.

   *Workaround:* `server.py` catches the error and returns raw text. For clean JSON, split files into 10-to-15-minute pieces first.

2. **Transformers off-by-one mask bug.**
   Passing audio to `apply_transcription_request` triggers an `IndexError` inside `get_audio_features`. The convolution layer outputs $X + 1$ frames, but the `padding_mask` calculation uses a naive `ceil` formula that only reserves $X$ frames. The slice crashes.

   *Workaround:* Pad `padding_mask` by 3,200 samples (one frame) whenever the audio length is a multiple of the stride.

3. **VRAM and CPU spillover.**
   Loading the 1.5B model in `bfloat16` takes more than 16 GB of VRAM. PyTorch pushes overflow layers into system RAM, dropping throughput to ~1 token/sec.

   *Workaround:* Load in 8-bit (`BitsAndBytesConfig(load_in_8bit=True)`). This cuts weight size in half and keeps execution entirely on the GPU.

4. **Slow inference.**
   Without an engine like `vLLM`, `model.generate()` is bound by PyTorch memory bandwidth. Even with FlashAttention-2, generation remains slow.

---

## 6. License

This project is licensed under the [MIT License](LICENSE). Copyright (c) 2026 Danny Whyze.

### Model weights license

The model weights for `microsoft/VibeVoice-ASR-HF` are published under the **MIT License** on Hugging Face. When downloading weights via `utils/download_model.py`, license metadata and commit revision hashes are verified and tracked in `download_info.json` alongside the weights.

