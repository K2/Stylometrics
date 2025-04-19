"""
ApiNotes.md (File-level) – generate_wavs_from_json.py

Role:
    Script to batch-generate WAV files from all text fields in JSON datasets (e.g., warmup_data/).
    Uses the canonical synthesize_audio routine from tts_tool.py.
    Designed for reproducibility and integration with downstream acoustic analysis.

Happy-Path:
    1. For each .json in warmup_data/, load entries.
    2. For each entry, synthesize WAV for each text field (original, paraphrased).
    3. Save WAVs to audio/ with unique filenames.
    4. Optionally, write a manifest mapping text to WAV path.
"""

import os
import json
from tts_tool import synthesize_audio

WARMUP_DIR = "warmup_data"
AUDIO_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

def generate_wavs():
    for fname in os.listdir(WARMUP_DIR):
        if not fname.endswith(".json"):
            continue
        base = os.path.splitext(fname)[0]
        with open(os.path.join(WARMUP_DIR, fname), "r") as f:
            data = json.load(f)
        for idx, entry in enumerate(data):
            for field in ("original", "paraphrased"):
                text = entry.get(field)
                if isinstance(text, list):
                    text = text[0]
                if not text:
                    continue
                wav_path = os.path.join(AUDIO_DIR, f"{base}_{idx}_{field}.wav")
                synthesize_audio(text, wav_path=wav_path)
                print(f"Generated: {wav_path}")

if __name__ == "__main__":
    generate_wavs()