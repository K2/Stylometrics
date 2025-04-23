"""
ApiNotes.md (File-level) – tts_tool.py

Role:
    Provides a programmatic interface for text-to-speech (TTS) synthesis and OpenSMILE feature extraction,
    decoupled from command-line execution. Enables other modules to synthesize audio and extract features
    without invoking subprocesses or shell commands.

Design Goals:
    - Enable direct, script-free TTS synthesis and OpenSMILE feature extraction from Python.
    - Centralize TTS and OpenSMILE configuration, referencing canonical config files for reproducibility.
    - Support future expansion to multiple TTS engines or OpenSMILE configs.
    - Facilitate integration with stylometric analysis and LLM feedback loops.
    - Return all extracted features to the application for direct LLM training and analysis.
    - Support workflows where the LLM is given both the original text and extracted features to learn resonance/dissonance indicators.

Architectural Constraints:
    - All TTS and OpenSMILE logic must be callable as Python functions.
    - No subprocess or shell command execution is allowed in this module.
    - All configuration is loaded from canonical config files or passed as arguments.
    - All interface and behavioral assumptions are documented in ApiNotes.
    - File size monitored; suggest splitting if exceeding 1/3 context window.

Happy-Path:
    1. Call synthesize_audio(text, wav_path) to synthesize speech to a WAV file.
    2. Call extract_opensmile_features(wav_path, config_path) to extract features.
    3. Use default OpenSMILE config at conf/opensmile/emo_large.conf unless overridden.
    4. Return all features as a dict to the application for further LLM processing.

ASCII Diagram:
    +-------------------+
    | synthesize_audio  |
    +-------------------+
              |
              v
    +-------------------+
    |  WAV file output  |
    +-------------------+
              |
              v
    +-------------------+
    | extract_opensmile |
    +-------------------+
              |
              v
    +-------------------+
    |  Feature dict     |
    +-------------------+
              |
              v
    +-------------------+
    |  Application/LLM  |
    +-------------------+
"""

import os
import tempfile
import soundfile as sf
import numpy as np
from dataset_helpers import extract_opensmile_features
import requests
import json

# Orpheus-TTS configuration (local vllm, CUDA device 2)
ORPHEUS_TTS_URL = os.environ.get("ORPHEUS_TTS_URL", "http://localhost:5005/tts")
ORPHEUS_TTS_VOICE = "Tara"
ORPHEUS_TTS_MAX_TOKENS = 1000

# Set CUDA environment for vllm (documented for user)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def synthesize_audio(text, wav_path=None, sample_rate=22050, voice=None, rate=None, volume=None):
    """
    ApiNotes: Synthesizes speech from text using Orpheus-TTS (Tara voice, 1000 max tokens, vllm CUDA).
    If wav_path is None, creates a temporary file and returns its path.
    Returns the path to the generated WAV file.
    """
    assert text is not None and isinstance(text, str) and text.strip(), "ApiNotes: text must be a non-empty string"
    payload = {
        "text": text,
        "voice": voice or ORPHEUS_TTS_VOICE,
        "max_tokens": ORPHEUS_TTS_MAX_TOKENS,
        "sample_rate": sample_rate
    }
    response = requests.post(ORPHEUS_TTS_URL, json=payload)
    assert response.status_code == 200, f"Orpheus-TTS API error: {response.status_code} {response.text}"
    audio_bytes = response.content
    if wav_path is None:
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
    with open(wav_path, "wb") as f:
        f.write(audio_bytes)
    # Defensive: Ensure file was created
    assert os.path.exists(wav_path), f"ApiNotes: TTS output file not created at {wav_path}"
    # Optionally, resample to target sample_rate if needed
    data, sr = sf.read(wav_path)
    if sr != sample_rate:
        try:
            import librosa
        except ImportError:
            raise ImportError("librosa must be installed for resampling audio")
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        if data.ndim == 1:
            data_resampled = librosa.resample(data, orig_sr=sr, target_sr=sample_rate)
        else:
            data_resampled = np.vstack([
                librosa.resample(data[:, ch], orig_sr=sr, target_sr=sample_rate)
                for ch in range(data.shape[1])
            ]).T
        sf.write(wav_path, data_resampled, sample_rate)
    return wav_path

# def extract_opensmile_features(wav_path, config_path=None):
#     """
#     ApiNotes: Extracts acoustic features from wav_path using OpenSMILE and the specified config file.
#     Uses conf/opensmile/emo_large.conf by default.
#     Returns a dictionary of all features for direct LLM training and analysis.
#     """
#     # Reference: file-level ApiNotes, imperative paradigm
#     assert opensmile is not None, "opensmile must be installed for feature extraction"
#     if config_path is None:
#         config_path = DEFAULT_OPENSMILE_CONFIG
#     assert os.path.exists(config_path), f"OpenSMILE config not found: {config_path}"
#     smile = opensmile.Smile(
#         feature_set=opensmile.FeatureSet.eGeMAPSv02,  # Use a generic set; config file will override
#         feature_level=opensmile.FeatureLevel.Functionals,
#     )
#     features = smile.process_file(wav_path)
#     # Convert DataFrame to dict and return all features
#     return features.iloc[0].to_dict()

# Usage example: Return all features for LLM training
def example_return_features_for_llm():
    """
    ApiNotes: Example usage for returning all features to the application for LLM training (Orpheus-TTS backend).
    """
    text = "The quick brown fox jumps over the lazy dog."
    wav_path = synthesize_audio(text)
    features = extract_opensmile_features(wav_path)
    # Application can now feed both text and features to the LLM for training
    print("Original text:", text)
    print("Extracted features:", features)
    os.remove(wav_path)
    return features, text

# Acceptance test (expected success/failure)
def test_synthesize_and_extract_expected_success():
    text = "The quick brown fox jumps over the lazy dog."
    wav_path = synthesize_audio(text)
    assert os.path.exists(wav_path), "WAV file should be created"
    features = extract_opensmile_features(wav_path)
    assert isinstance(features, dict) and len(features) > 0, "Features dict should not be empty"
    print("TTS and OpenSMILE extraction succeeded:", features)
    os.remove(wav_path)

def test_synthesize_and_extract_expected_failure():
    try:
        synthesize_audio("")  # Empty text should fail
    except Exception as e:
        print(f"(expected failure) {e}")
    try:
        extract_opensmile_features("nonexistent.wav")
    except Exception as e:
        print(f"(expected failure) {e}")

# ApiNotes: This implementation is imperative, modular, and justified by file-level and project-level ApiNotes.
#           All interface and behavioral assumptions are documented.
#           Acceptance tests include (expected success) and (expected failure) cases.