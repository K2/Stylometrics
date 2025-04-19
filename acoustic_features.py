# Writing a Python module that leverages openSMILE and Praat (via Parselmouth)

"""acoustic_features.py

Module for extracting acoustic features using openSMILE and Praat (via Parselmouth).
Requirements:
    - openSMILE (SMILExtract binary) installed and accessible.
    - openSMILE configuration file (e.g., eGeMAPS.conf or ComParE.conf).
    - praat_parselmouth (`pip install praat-parselmouth`).

Functions:
    - extract_opensmile: extract features via openSMILE.
    - extract_praat: extract pitch, formants, intensity, jitter, shimmer, HNR via Praat.
    - extract_all: convenience function to do both extractions."""

import os
import subprocess
import parselmouth

class AcousticFeatureExtractor:
    def __init__(self, opensmile_path: str, opensmile_config: str):
        
        """
        Initialize the extractor.

        :param opensmile_path: Path to the SMILExtract binary.
        :param opensmile_config: Path to the openSMILE config file (e.g., eGeMAPS.conf).
        """
        self.opensmile_path = opensmile_path
        self.opensmile_config = opensmile_config

    def extract_opensmile(self, wav_path: str, output_csv: str):
        """
        Extract features using openSMILE.

        :param wav_path: Input WAV file.
        :param output_csv: Path to the output CSV file.
        """
        cmd = [
            self.opensmile_path,
            '-C', self.opensmile_config,
            '-I', wav_path,
            '-O', output_csv
        ]
        subprocess.run(cmd, check=True)
        return output_csv
    

    """
    ApiNotes.md (File-level) – acoustic_features.py

    Role:
        Provides routines for extracting acoustic features (openSMILE, Praat) for stylometric analysis.
        Used by main orchestrator and dataset_helpers for feature extraction and prompt engineering.

    Design Goals:
        - Enable robust, reproducible extraction of acoustic features from audio files.
        - Support both openSMILE and Praat (via parselmouth) for feature diversity.
        - All routines must be stateless and callable from main or helper modules.

    Architectural Constraints:
        - All dependencies must be explicit and imported at the top of the file.
        - All error handling must be imperative and documented.
        - All interface changes must be reflected in ApiNotes and acceptance tests.

    Happy-Path:
        1. Given a WAV file, extract openSMILE and Praat features.
        2. Return features as dicts or write to disk as needed.

    ASCII Diagram:
        +-------------------+
        | acoustic_features |
        +-------------------+
            |   |   |
            v   v   v
        [openSMILE][Praat][summary]
    """

    import parselmouth  # ApiNotes: Explicit import required for all Praat routines

    def extract_praat(self, wav_path: str) -> dict:
        """
        Extracts Praat features using parselmouth.
        Returns a dictionary of features.
        """
        # ApiNotes: All dependencies must be imported at the top of the file.
        assert wav_path and isinstance(wav_path, str), "wav_path must be a non-empty string"
        snd = parselmouth.Sound(wav_path)
        pitch_obj = snd.to_pitch()
        # Only consider voiced frames for mean F0
        frequencies = pitch_obj.selected_array['frequency']
        voiced_frequencies = frequencies[frequencies > 0]
        mean_f0 = float(voiced_frequencies.mean()) if voiced_frequencies.size > 0 else 0.0
        stdev_f0 = float(voiced_frequencies.std()) if voiced_frequencies.size > 0 else 0.0

        # Jitter and shimmer extraction (Praat docs: 6 arguments for shimmer, 5 for jitter)
        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 300)
        # Jitter (local): start_time, end_time, min_period, max_period, max_factor
        jitter_local = parselmouth.praat.call(
            point_process, "Get jitter (local)",
            0, 0, 0.0001, 0.02, 1.3
        )
        # Shimmer (local): start_time, end_time, min_period, max_period, max_factor, min_amplitude
        shimmer_local = parselmouth.praat.call(
            [snd, point_process], "Get shimmer (local)",
            0, 0, 0.0001, 0.02, 1.3, 0.0001
        )

        # Harmonicity (HNR)
        harmonicity_obj = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = parselmouth.praat.call(harmonicity_obj, "Get mean", 0, 0)

        return {
            "mean_f0": mean_f0,
            "sd_f0": stdev_f0,
            "jitter_local": jitter_local,
            "shimmer_local": shimmer_local,
            "hnr": hnr,
        }

    def extract_all(self, wav_path: str, output_dir: str):
        """
        Run both openSMILE and Praat extractions.

        :param wav_path: Input WAV file.
        :param output_dir: Directory where to save results.
        :return: Tuple of (opensmile_csv_path, praat_features_dict).
        """
        base = os.path.splitext(os.path.basename(wav_path))[0]
        os.makedirs(output_dir, exist_ok=True)
        opensmile_csv = os.path.join(output_dir, f"{base}_opensmile.csv")
        praat_features = os.path.join(output_dir, f"{base}_praat.json")

        # openSMILE extraction
        self.extract_opensmile(wav_path, opensmile_csv)

        # Praat extraction
        features = self.extract_praat(wav_path)

        # Save Praat features as JSON
        import json
        with open(praat_features, "w") as f:
            json.dump(features, f, indent=2)

        return opensmile_csv, praat_features

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract acoustic features via openSMILE and Praat.")
    parser.add_argument("wav", help="Input WAV file")
    parser.add_argument("--opensmile_path", default="SMILExtract", help="Path to SMILExtract binary")
    parser.add_argument("--config", default="eGeMAPS.conf", help="openSMILE config file")
    parser.add_argument("--output_dir", default="features", help="Directory to store output")
    args = parser.parse_args()

    extractor = AcousticFeatureExtractor(args.opensmile_path, args.config)
    smile_csv, praat_json = extractor.extract_all(args.wav, args.output_dir)
    print(f"openSMILE output: {smile_csv}")
    print(f"Praat features: {praat_json}")
