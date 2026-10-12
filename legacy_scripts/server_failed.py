import asyncio
from contextlib import asynccontextmanager
import gc
import os
from pathlib import Path
from typing import List, Optional
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
import torch
import uvicorn

MODEL_NAME = "microsoft/VibeVoice-ASR-HF"
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
MAX_UPLOAD_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB

processor = None
model = None
gpu_lock: Optional[asyncio.Lock] = None


class TranscriptionSegment(BaseModel):
    Start: float
    End: float
    Speaker: str
    Content: str


class TranscribeResponse(BaseModel):
    filename: str
    status: str
    is_fallback: bool = False
    warning: Optional[str] = None
    transcription: List[TranscriptionSegment]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, model, gpu_lock
    gpu_lock = asyncio.Lock()
    print(f"Loading {MODEL_NAME} model into GPU memory (bfloat16 for maximum speed)...")

    from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration, BitsAndBytesConfig

    # 8-bit quantization saves VRAM and prevents CPU offloading on 16GB cards.
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
        MODEL_NAME, 
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        attn_implementation={
            "language_model": "flash_attention_2", 
            "acoustic_tokenizer_encoder": "sdpa",
            "semantic_tokenizer_encoder": "sdpa"
        }
    )
    print(f"Model successfully loaded! Server ready on: {model.device}")
    yield
    print("Shutting down: Clearing GPU memory...")
    del model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="VibeVoice ASR Server", 
    description="Transcribe 60 min audio files with Speaker & Timestamps",
    lifespan=lifespan
)


def _run_inference(audio_path: str, prompt: Optional[str] = None) -> tuple[list[dict], bool]:
    with torch.inference_mode():
        inputs = processor.apply_transcription_request(
            audio=audio_path,
            prompt=prompt
        ).to(model.device, model.dtype)

        if "padding_mask" in inputs:
            mask = inputs["padding_mask"]
            # Only pad if it's a perfect multiple of 3200.
            # We use mask.shape[-1] instead of mask.sum() because mask is in bfloat16, 
            # and summing 14 million ones in bfloat16 causes massive floating point precision errors!
            if mask.shape[-1] % 3200 == 0:
                extra_padding = torch.ones((mask.shape[0], 3200), dtype=mask.dtype, device=mask.device)
                inputs["padding_mask"] = torch.cat([mask, extra_padding], dim=-1)

        output_ids = model.generate(
            **inputs,
            acoustic_tokenizer_chunk_size=64000
        )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]

        try:
            return processor.decode(generated_ids, return_format="parsed")[0], False
        except Exception as exc:
            print(f"JSON Parsing failed: {exc}. Falling back to raw text.")
            raw_text = processor.decode(generated_ids, return_format="text")[0]
            fallback_segments = [{
                "Start": 0.0,
                "End": 0.0,
                "Speaker": "Unknown",
                "Content": raw_text.strip()
            }]
            return fallback_segments, True


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(file: UploadFile = File(...), prompt: Optional[str] = None):
    raw_filename = file.filename or "audio.wav"
    if ".." in raw_filename:
        raise HTTPException(status_code=400, detail="Path traversal detected in filename.")

    safe_filename = Path(raw_filename).name
    suffix = Path(safe_filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format: {suffix}")

    temp_dir = Path("temp_audio").resolve()
    temp_dir.mkdir(exist_ok=True)

    unique_name = f"{uuid.uuid4().hex}_{safe_filename}"
    temp_file_path = (temp_dir / unique_name).resolve()

    if not temp_file_path.is_relative_to(temp_dir):
        raise HTTPException(status_code=400, detail="Invalid file path detected.")

    total_bytes = 0
    with open(temp_file_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_SIZE_BYTES:
                buffer.close()
                if temp_file_path.exists():
                    os.remove(temp_file_path)
                raise HTTPException(status_code=413, detail="File too large (exceeds 500 MB limit).")
            buffer.write(chunk)

    global gpu_lock
    if gpu_lock is None:
        gpu_lock = asyncio.Lock()

    try:
        async with gpu_lock:
            segments, is_fallback = await asyncio.to_thread(_run_inference, str(temp_file_path), prompt)

        warning_msg = (
            "Model token limit reached (Unterminated JSON string). Returning raw text."
            if is_fallback
            else None
        )
        return TranscribeResponse(
            filename=safe_filename,
            status="success",
            is_fallback=is_fallback,
            warning=warning_msg,
            transcription=segments
        )
    finally:
        if temp_file_path.exists():
            try:
                os.remove(temp_file_path)
            except OSError:
                pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
