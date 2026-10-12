import argparse
import json
import os
import tempfile
import torch
import soundfile as sf

# Optional custom cache directory (defaults to system cache if HF_HOME is unset)
# os.environ.setdefault("HF_HOME", "path/to/models")

MODEL_NAME = "microsoft/VibeVoice-ASR-HF"


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Local CLI runner for VibeVoice ASR")
    parser.add_argument("--audio", "-a", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--minutes", "-m", type=int, default=10, help="Chunk duration in minutes (default: 10)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Optional output JSON file path")
    return parser.parse_args(args)


def decode_chunk_with_fallback(processor, generated_ids, block_duration: float, chunk_index: int = 1) -> list:
    """Decode generated tokens into structured transcription with text fallback."""
    try:
        return processor.decode(generated_ids, return_format="parsed")[0]
    except Exception as exc:
        print(f"Warning: JSON parsing failed for chunk {chunk_index} ({exc}). Falling back to text.")
        raw_text = processor.decode(generated_ids, return_format="text")[0]
        return [{
            "Start": 0.0,
            "End": block_duration,
            "Speaker": "Unknown",
            "Content": raw_text.strip()
        }]


def main():
    args = parse_args()
    audio_path = os.path.abspath(args.audio.strip('"\''))
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at '{audio_path}'")
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print("Loading model with 8-bit quantization to fit in 16GB VRAM...")
    from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration, BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
        MODEL_NAME, 
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config
    )
    
    print(f"\nProcessing: {audio_path}")
    
    CHUNK_MINUTES = args.minutes
    info = sf.info(audio_path)
    sr = info.samplerate
    chunk_samples = int(CHUNK_MINUTES * 60 * sr)
    
    full_transcription = []
    time_offset = 0.0
    
    total_blocks = (info.frames // chunk_samples) + 1
    for i, block in enumerate(sf.blocks(audio_path, blocksize=chunk_samples)):
        print(f"\n--- Processing Chunk {i+1}/{total_blocks} (Offset: {time_offset:.2f}s) ---")
        
        # Save chunk to temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_name = tmp.name
        tmp.close()  # Critical for Windows: close handle so sf.write can open it safely
        
        try:
            sf.write(tmp_name, block, sr)
            inputs = processor.apply_transcription_request(
                audio=tmp_name,
                prompt=None
            ).to(model.device, model.dtype)
            
            # The Transformers off-by-one bug occurs because convolution generates 4501 frames 
            # while the mask calculates 4500 frames. 
            # To fix this SAFELY without deleting the mask (which prevents hallucinations on real padding),
            # we pad the attention mask by 3200 samples (exactly 1 frame).
            if "padding_mask" in inputs:
                mask = inputs["padding_mask"]
                if mask.shape[-1] % 3200 == 0:
                    extra_padding = torch.ones((mask.shape[0], 3200), dtype=mask.dtype, device=mask.device)
                    inputs["padding_mask"] = torch.cat([mask, extra_padding], dim=-1)
            
            print(f"Token shape for chunk {i+1}: {inputs['input_ids'].shape}")
            print(f"Generating transcription for chunk {i+1}...")
            
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    acoustic_tokenizer_chunk_size=64000
                )
            generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
            
            chunk_transcription = decode_chunk_with_fallback(
                processor=processor,
                generated_ids=generated_ids,
                block_duration=len(block) / sr,
                chunk_index=i + 1
            )
            
            # Adjust timestamps for this chunk
            for segment in chunk_transcription:
                segment["Start"] = segment.get("Start", 0.0) + time_offset
                segment["End"] = segment.get("End", 0.0) + time_offset
                full_transcription.append(segment)
                
            # Print live results for the chunk
            for segment in chunk_transcription:
                print(f"[{segment['Start']:.2f} - {segment['End']:.2f}] Speaker {segment.get('Speaker', 'Unknown')}: {segment.get('Content', '')}")
                
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
            
        time_offset += (len(block) / sr)
        
    print("\n\n--- Final Combined Result ---")
    for segment in full_transcription:
        print(f"[{segment.get('Start', 0.0):.2f} - {segment.get('End', 0.0):.2f}] Speaker {segment.get('Speaker', 'Unknown')}: {segment.get('Content', '')}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(full_transcription, f, indent=2, ensure_ascii=False)
        print(f"\nSaved full transcription to {args.output}")


if __name__ == "__main__":
    main()
