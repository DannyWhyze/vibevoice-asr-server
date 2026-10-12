"""Archived Diagnostic Prototype: vLLM High-Throughput Inference Experiment.

STATUS: NON-FUNCTIONAL (ARCHIVED)
REASON: vLLM does not support the 'VibeVoiceAsrForConditionalGeneration' architecture.
For technical details and post-mortem analysis, refer to legacy_scripts/README.md section 4.
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Test VibeVoice with vLLM (Archived Experiment)")
    parser.add_argument("--audio", type=str, required=False, help="Path to the audio file")
    parser.add_argument(
        "--force-experimental",
        action="store_true",
        help="Attempt to run vLLM despite known incompatibility"
    )
    args = parser.parse_args()

    if not args.force_experimental:
        print(
            "ERROR: 'microsoft/VibeVoice-ASR-HF' is currently incompatible with vLLM.\n"
            "Architecture 'VibeVoiceAsrForConditionalGeneration' lacks a vLLM model runner.\n"
            "See legacy_scripts/README.md for full post-mortem details.\n"
            "To force execution anyway, pass --force-experimental.",
            file=sys.stderr
        )
        sys.exit(1)

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("ERROR: vLLM is not installed in the active environment.", file=sys.stderr)
        sys.exit(1)

    if not args.audio:
        print("ERROR: --audio is required when running with --force-experimental.", file=sys.stderr)
        sys.exit(1)

    # 1. Initialize vLLM (This replaces the entire transformers model loading)
    # vLLM automatically uses PagedAttention and handles all CUDA optimizations.
    print("Loading VibeVoice via vLLM...")
    llm = LLM(
        model="microsoft/VibeVoice-ASR-HF",
        trust_remote_code=True,
        max_model_len=32768,  # Reserve enough context for long audio
        gpu_memory_utilization=0.90  # Use 90% of your 16GB VRAM for the KV Cache
    )

    # 2. Define the prompt and generation parameters
    prompt = "<|audio|> Please transcribe the audio into text."

    # vLLM uses SamplingParams instead of model.generate(**kwargs)
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy decoding for accurate transcription
        max_tokens=8192,  # Generate up to 8192 words
    )

    print(f"Processing audio: {args.audio}")

    # 3. Generate using vLLM
    # Note: For multimodal inputs in vLLM (like audio), we pass a dictionary 
    # to 'multi_modal_data' containing the audio file or tensor.
    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {"audio": args.audio},
        },
        sampling_params=sampling_params
    )

    # 4. Print the result
    for output in outputs:
        generated_text = output.outputs[0].text
        print("\n--- Transcription Result ---")
        print(generated_text)
        print("----------------------------\n")


if __name__ == "__main__":
    main()
