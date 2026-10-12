# Legacy scripts

This folder archives early prototypes, diagnostic runners, and bug documentation for `microsoft/VibeVoice-ASR-HF`.

---

## 1. `server_failed.py` (FastAPI server)

### Overview
`server_failed.py` was an initial attempt to expose `microsoft/VibeVoice-ASR-HF` as an HTTP endpoint (`POST /transcribe`) for transcription, speaker labels, and timestamps.

### Implementation
* **8-bit quantization.** Uses `BitsAndBytesConfig(load_in_8bit=True)` to cut weight size in half. Unquantized `bfloat16` spills past 16 GB VRAM on cards like the RTX 5070 Ti, pushing layers to system RAM and dropping speed to ~1 token/sec.
* **Attention.** Uses `flash_attention_2` on the Qwen2 text decoder and SDPA on the audio encoders.
* **Cleanup.** Runs `gc.collect()` and empties the CUDA cache on shutdown.
* **Audio handling.** Saves incoming uploads to `temp_audio/` before processing.
* **Padding mask fix.** Appends 3,200 ones (`torch.ones`) to `padding_mask` whenever the sample count divides cleanly by 3,200. Checks `mask.shape[-1]` instead of `mask.sum()` to avoid `bfloat16` precision drift.
* **JSON decode fallback.** Catches JSON syntax errors from `processor.decode` when the model cuts off mid-sentence, returning raw unparsed text instead of crashing.

### Why it failed
1. **The 60-minute claim hits a token wall.** The acoustic encoder can read 60 minutes, but the Qwen2 text decoder stops at 32,768 tokens. Verbose JSON (`Speaker`, `Start`, `End` per line) on a 37-minute podcast produces over 80,000 characters. The model runs out of tokens mid-string and crashes `processor.decode`.
2. **Slow inference in pure PyTorch.** Generating text in Python without an optimized engine like vLLM or `VibeASR.cpp` is memory-bandwidth bound. Large files take hours.
3. **No job queue.** Long generation runs block the FastAPI thread. Concurrent uploads trigger instant CUDA OOM.

---

## 2. `test_local.py` (local CLI runner)

### Overview
`test_local.py` runs `microsoft/VibeVoice-ASR-HF` directly on the local GPU without Docker or a web server. It cuts audio into 10-minute blocks to avoid context limits and stitches timestamps back together.

### Implementation
* **Cache isolation.** Supports configuring `HF_HOME` before imports so model downloads do not fill the primary drive.
* **8-bit loading.** Fits the model within 16 GB VRAM using `BitsAndBytesConfig(load_in_8bit=True)`.
* **Interactive prompt.** Reads the audio path from `input()` and strips quotes for easy drag-and-drop on Windows.
* **Streaming chunks.** Uses `soundfile.blocks` to slice audio into 10-minute WAV files without buffering the whole recording in RAM.
* **Padding mask fix.** Appends 3,200 ones to `padding_mask` if sample count divides by 3,200.
* **Timestamp tracking.** Adds a running `time_offset` to each chunk's `Start` and `End` values, printing live diarization and a combined final transcript:
  ```python
  segment["Start"] = segment.get("Start", 0.0) + time_offset
  segment["End"] = segment.get("End", 0.0) + time_offset
  ```

### Why it moved to legacy
1. **Manual input.** The interactive `input()` prompt cannot run unattended in background scripts or containers.
2. **Vanilla PyTorch speed.** Running sequential chunks through `model.generate()` still suffers from high decoding latency.
3. **Prototype code.** The chunking and timestamp math works, but belongs in a clean library rather than a loose test script.

---

## 3. Known bugs

### Bug A: Transformers off-by-one mask bug (`IndexError`)
* **File:** `transformers/models/vibevoice_asr/modeling_vibevoice_asr.py` inside `get_audio_features()`.
* **Traceback:**
  ```text
  File ".../transformers/models/vibevoice_asr/modeling_vibevoice_asr.py", line 388, in get_audio_features
    combined_features = combined_features[padding_mask]
  IndexError: The shape of the mask [1, 4500] at index 1 does not match the shape of the indexed tensor [1, 4501, 3584] at index 1
  ```
* **Cause:** When sample count divides by 3,200 (such as 14,400,000 samples for 10 minutes at 24 kHz), the naive `ceil` calculation gives 4,500 mask frames, while the convolution padding produces 4,501 frames. Slicing with the shorter mask crashes.
* **Fix in scripts:** Append 3,200 ones (one frame) whenever `mask.shape[-1] % 3200 == 0`:
  ```python
  if "padding_mask" in inputs:
      mask = inputs["padding_mask"]
      if mask.shape[-1] % 3200 == 0:
          extra_padding = torch.ones((mask.shape[0], 3200), dtype=mask.dtype, device=mask.device)
          inputs["padding_mask"] = torch.cat([mask, extra_padding], dim=-1)
  ```
  *Precision note:* Check `mask.shape[-1]`, not `mask.sum()`. Summing millions of ones in `bfloat16` introduces large floating-point rounding errors.

### Bug B: Unhandled JSON parsing in `test_local.py`
Line 77 calls `processor.decode(generated_ids, return_format="parsed")[0]` without a `try-except` guard. If a chunk hits token limits or loops, `json.loads` crashes with `Unterminated string` and discards every completed chunk.

### Bug C: Speaker drift across chunks
Each 10-minute chunk runs in a fresh `model.generate()` call. The model has no memory of past voices. Speaker 1 in the first block often becomes Speaker 2 or Speaker 3 in the next.

### Bug D: Audio clipping at chunk boundaries
`soundfile.blocks` cuts strictly by sample count without voice activity detection (VAD). Slicing mid-syllable causes words to get cut in half and produces hallucinations at the edges.

---

## 4. `test_vllm.py` (vLLM experiment)

### Overview
`test_vllm.py` tested whether vLLM could serve `microsoft/VibeVoice-ASR-HF` to speed up generation with PagedAttention and optimized CUDA kernels.

### Setup and intent
* Initialized `LLM(model="microsoft/VibeVoice-ASR-HF", max_model_len=32768, gpu_memory_utilization=0.90)`.
* Set greedy decoding (`temperature=0.0`, `max_tokens=8192`).
* Passed audio via `multi_modal_data={"audio": args.audio}`.

### Why it failed
1. **Unsupported architecture.** vLLM lacks an internal model runner for `VibeVoiceAsrForConditionalGeneration`. The script crashes on load:
   ```text
   ValueError: Model architectures ['VibeVoiceAsrForConditionalGeneration'] are not supported for now to be run by vLLM.
   ```
2. **Transformers code is not vLLM code.** Passing `trust_remote_code=True` does not help because the checkpoint only provides Hugging Face Transformers definitions, not vLLM tensor-parallel graph runners.
3. **Result.** Using vLLM would require writing a custom C++/CUDA backend implementation for VibeVoice inside vLLM.
