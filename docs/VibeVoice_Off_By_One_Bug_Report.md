# Bug report: IndexError in VibeVoiceAsrModel from off-by-one mask calculation

## System info
* Transformers version: 5.13.0 (and affects current `main`)
* Model: `microsoft/VibeVoice-ASR-HF`
* Execution: Local PyTorch on GPU

## Problem
Audio inputs whose sample count divides cleanly by `acoustic_tokenizer_chunk_size` (64,000) and `hop_length` (3,200) crash during `generate()` with an `IndexError`.

The `acoustic_tokenizer_encoder` produces a convolutional feature tensor of `N + 1` frames, but the code calculates `padding_mask` length as `N` frames. The resulting shape mismatch breaks indexing.

## Traceback
```text
Traceback (most recent call last):
  ...
  File ".../transformers/models/vibevoice_asr/modeling_vibevoice_asr.py", line 388, in get_audio_features
    combined_features = combined_features[padding_mask]
                        ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^
IndexError: The shape of the mask [1, 4500] at index 1 does not match the shape of the indexed tensor [1, 4501, 3584] at index 1
```

## Root cause
The calculation at line 382 of `transformers/models/vibevoice_asr/modeling_vibevoice_asr.py` uses naive division:

```python
num_audio_tokens = torch.ceil(
    padding_mask.sum(dim=-1) / self.config.acoustic_tokenizer_encoder_config.hop_length
).to(torch.int64)
```

For a 10-minute chunk at 24 kHz:
1. The raw audio has 14,400,000 samples.
2. `hop_length` is 3,200.
3. `14,400,000 / 3,200 = 4,500`. `torch.ceil(4500)` returns 4,500.
4. The mask shape is `[1, 4500]`.
5. The convolution uses sliding windows with boundary padding and returns 4,501 frames.
6. Line 388 tries to index the `[1, 4501, 3584]` tensor with the `[1, 4500]` mask and throws `IndexError`.

## Reproduction
Pass an audio array of 14,400,000 samples (or any multiple of 64,000 divisible by 3,200) into `AutoProcessor` and run `model.generate()`.

## Workaround
Pad `inputs["padding_mask"]` with 3,200 ones before calling `model.generate()`. That pushes `torch.ceil` to 4,501 so the dimensions match:

```python
inputs = processor.apply_transcription_request(audio=audio_path)

if "padding_mask" in inputs:
    mask = inputs["padding_mask"]
    # Append 3200 samples (1 frame) to fix the off-by-one division
    extra_padding = torch.ones((mask.shape[0], 3200), dtype=mask.dtype, device=mask.device)
    inputs["padding_mask"] = torch.cat([mask, extra_padding], dim=-1)

output_ids = model.generate(**inputs)
```

## Expected behavior
`get_audio_features` should compute `num_audio_tokens` from the convolution output shape instead of dividing samples by `hop_length`.
