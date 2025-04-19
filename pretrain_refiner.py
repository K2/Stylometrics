"""
ApiNotes.md (File-level) – pretrain_refiner.py

Role:
    Provides routines for generating and refining a high-quality pretrain dataset,
    using OpenSMILE to review and filter/refine candidate entries.

Design Goals:
    - Enable robust, reproducible generation of pretrain data for stylometric models.
    - Use OpenSMILE to ensure only high-quality, phonetically diverse, and well-characterized entries are included.
    - Callable from the main application via --generate-pretrain.

Architectural Constraints:
    - All routines must be stateless and callable from main or helper modules.
    - All dependencies must be explicit.
    - All interface changes must be reflected in ApiNotes and acceptance tests.

Happy-Path:
    1. Load candidate pretrain data (from a file or corpus).
    2. For each entry, synthesize audio (if needed) and extract OpenSMILE features.
    3. Filter or refine entries based on OpenSMILE metrics (e.g., F0 stdev, MFCCs, energy).
    4. Save the refined dataset to output_path.
    5. Log all actions and statistics.

ASCII Diagram:
    +-------------------+
    |  Load Candidates  |
    +-------------------+
              |
              v
    +-------------------+
    |  Synthesize Audio |
    +-------------------+
              |
              v
    +-------------------+
    |  OpenSMILE Review |
    +-------------------+
              |
              v
    +-------------------+
    |  Filter/Refine    |
    +-------------------+
              |
              v
    +-------------------+
    |  Save/Log Output  |
    +-------------------+
"""

import json
import os
from acoustic_features import AcousticFeatureExtractor
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


def generate_high_quality_pretrain(opensmile_path, opensmile_config, output_path, log_file=None, tts_cmd=None, audio_outdir="pretrain_audio/"):
    """
    ApiNotes: Generates a high-quality pretrain dataset, reviewed by OpenSMILE.
    """
    # Load candidate pretrain data (replace with your actual source)
    candidate_file = "pretrain_candidates.json"
    assert os.path.exists(candidate_file), "Candidate pretrain data file must exist"
    with open(candidate_file, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    if log_file:
        log_file.write("Starting high-quality pretrain dataset generation...\n")

    extractor = AcousticFeatureExtractor(
        opensmile_path=opensmile_path,
        opensmile_config=opensmile_config
    )

    refined = []
    for idx, entry in enumerate(candidates):
        text = entry["text"]
        # Synthesize audio using TTS
        wav_path = os.path.join(audio_outdir, f"pretrain_tmp_{idx}.wav")
        
        if tts_cmd:
            try:
                import subprocess
                import shlex
                
                # Format the TTS command with text and output path
                formatted_cmd = tts_cmd.format(text=shlex.quote(text), out=shlex.quote(wav_path))
                
                # Execute the TTS command
                result = subprocess.run(
                    formatted_cmd,
                    shell=True,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    if log_file:
                        log_file.write(f"[ERROR] TTS command failed for entry {idx}: {result.stderr}\n")
                    continue
                
                if not os.path.exists(wav_path):
                    if log_file:
                        log_file.write(f"[ERROR] TTS did not generate WAV file for entry {idx}\n")
                    continue
                    
                if log_file:
                    log_file.write(f"[INFO] Generated audio for entry {idx}\n")
            except Exception as e:
                if log_file:
                    log_file.write(f"[ERROR] Exception during TTS for entry {idx}: {str(e)}\n")
                continue
        else:
            # No TTS command provided
            if log_file:
                log_file.write(f"[WARN] No TTS command provided for entry {idx}, skipping audio generation\n")
            continue  # Skip this entry if we can't generate audio

        # Extract OpenSMILE features
        features = extractor.extract_opensmile(wav_path, output_csv=f"pretrain_features_{idx}.csv")
        
        # Parse features from CSV if features is a string (likely a path to the CSV file)
        if isinstance(features, str):
            feature_dict = {}
            # Assuming the CSV has headers and the format is compatible with standard CSV parsing
            import csv
            with open(features, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                # Take the first row of data (assuming single-instance extraction)
                for row in reader:
                    feature_dict = {k: float(v) if v.replace('.', '', 1).isdigit() else v 
                                   for k, v in row.items()}
                    break
            features = feature_dict
                
        # Now features should be a dictionary
        try:
            f0_std_value = float(features.get("F0smaStdd", 0))
            rms_energy_value = float(features.get("pcm_RMSenergy_sma", 0))
            if f0_std_value > 10 and rms_energy_value > 0.1:
                entry["opensmile_features"] = features
                refined.append(entry)
                if log_file:
                    log_file.write(f"[INFO] Accepted entry {idx}: {text}\n")
            else:
                if log_file:
                    log_file.write(f"[WARN] Rejected entry {idx}: {text}\n")
        except (ValueError, TypeError):
            # Handle case where conversion to float fails
            if log_file:
                log_file.write(f"[ERROR] Failed to process entry {idx} due to invalid feature values: {text}\n")

    # Save refined dataset
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(refined, f, indent=2)
    if log_file:
        log_file.write(f"[INFO] Saved {len(refined)} refined pretrain entries to {output_path}\n")

# ApiNotes: This implementation is imperative, modular, and justified by file-level and project-level ApiNotes.
#           Adjust candidate loading, synthesis, and filtering as needed for your project.