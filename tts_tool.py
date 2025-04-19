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
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    import opensmile
except ImportError:
    opensmile = None

# Canonical OpenSMILE config path (see directory-level ApiNotes for config management)
DEFAULT_OPENSMILE_CONFIG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "conf", "opensmile", "emo_large.conf")
)
def synthesize_audio(text, wav_path=None, sample_rate=22050, voice=None, rate=None, volume=None):
    """
    ApiNotes: Synthesizes speech from text and writes to wav_path.
    If wav_path is None, creates a temporary file and returns its path.
    Uses pyttsx3 for TTS (offline, cross-platform).
    Optional parameters:
      - sample_rate: Target sample rate for output WAV file (default 22050 Hz).
      - voice: Optional voice id or name to use.
      - rate: Optional speech rate (words per minute).
      - volume: Optional volume (0.0 to 1.0).
    Returns the path to the generated WAV file.
    """
    # Reference: file-level ApiNotes, imperative paradigm
    assert pyttsx3 is not None, "pyttsx3 must be installed for TTS synthesis"
    assert text is not None and isinstance(text, str) and text.strip(), "ApiNotes: text must be a non-empty string"
    engine = pyttsx3.init()
    # Set voice if provided
    if voice is not None:
        voices = engine.getProperty('voices')
        matched = False
        for v in voices:
            if voice in (v.id, v.name):
                engine.setProperty('voice', v.id)
                matched = True
                break
        assert matched, f"ApiNotes: requested voice '{voice}' not found in available voices"
    # Set rate if provided
    if rate is not None:
        engine.setProperty('rate', rate)
    # Set volume if provided
    if volume is not None:
        engine.setProperty('volume', volume)
    # Use a temporary file if wav_path is not provided
    if wav_path is None:
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
    # Synthesize to file
    engine.save_to_file(text, wav_path)
    engine.runAndWait()
    # Defensive: Ensure file was created
    assert os.path.exists(wav_path), f"ApiNotes: TTS output file not created at {wav_path}"
    # Optionally, resample to target sample_rate if needed
    data, sr = sf.read(wav_path)
    if sr != sample_rate:
        try:
            import librosa
        except ImportError:
            raise ImportError("librosa must be installed for resampling audio")
        # Defensive: librosa expects float32, ensure correct dtype
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        # Handle mono/stereo
        if data.ndim == 1:
            data_resampled = librosa.resample(data, orig_sr=sr, target_sr=sample_rate)
        else:
            # Resample each channel separately
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
    ApiNotes: Example usage for returning all features to the application for LLM training.
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
        synthesize_audio("Test", wav_path="/invalid/path/to/file.wav")
    except Exception as e:
        print(f"(expected failure) {e}")
    try:
        extract_opensmile_features("nonexistent.wav")
    except Exception as e:
        print(f"(expected failure) {e}")

# ApiNotes: This implementation is imperative, modular, and justified by file-level and project-level ApiNotes.
#           All interface and behavioral assumptions are documented.
#           Acceptance tests include (expected success) and (expected failure) cases.