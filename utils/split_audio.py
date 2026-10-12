import soundfile as sf
import os
import argparse
import sys

def split_audio(input_file, chunk_minutes=10):
    try:
        info = sf.info(input_file)
    except Exception as e:
        print(f"Error reading file '{input_file}': {e}")
        print("Note: If this is an MP3 file and it fails, you may need to install ffmpeg or use WAV files.")
        return

    sr = info.samplerate
    chunk_samples = int(chunk_minutes * 60 * sr)
    
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    out_dir = f"{base_name}_chunks"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Splitting '{input_file}' into {chunk_minutes}-minute chunks...")
    print(f"Total duration: {info.frames / sr / 60:.2f} minutes")
    
    for i, block in enumerate(sf.blocks(input_file, blocksize=chunk_samples)):
        out_file = os.path.join(out_dir, f"{base_name}_part{i+1:03d}.wav")
        sf.write(out_file, block, sr)
        print(f"Saved: {out_file} (Duration: {len(block)/sr:.2f} seconds)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split an audio file into chunks of N minutes.")
    parser.add_argument("input_file", help="Path to the audio file (e.g. podcast.mp3 or podcast.wav)")
    parser.add_argument("--minutes", type=float, default=10.0, help="Chunk length in minutes (default: 10.0)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: File '{args.input_file}' not found.")
        sys.exit(1)
        
    split_audio(args.input_file, args.minutes)
