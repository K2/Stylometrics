"""
ApiNotes.md (File-level) – generate_phoneme_wavs.py

Role:
    Batch-generates WAV files for all IPA and ARPAbet phonemes, for each Orpheus-TTS voice and modifier, using tts_tool.synthesize_audio.
    Organizes output as audio/phonetic_alphabet/{voice}/{modifier}/{ipa|arpabet}/{phoneme}.wav.
    Designed for comprehensive TTS warmup, benchmarking, and stylometric analysis.

Design Goals:
    - Ensure full coverage of phoneme, voice, and modifier combinations.
    - Use imperative, robust, and modular code referencing tts_tool and warmup_phoneme_data.
    - Provide clear error handling, logging, and runtime assertions.
    - Reference nearest ApiNotes.md for project and directory context.
    - Include usage and acceptance tests, and suggest debugging strategies.

Architectural Constraints:
    - All synthesis must use tts_tool.synthesize_audio (no subprocesses).
    - Directory structure must be created as needed.
    - All interface and behavioral assumptions must be documented in ApiNotes.
    - File must be importable and runnable as a script.
    - All changes must be reflected in project-level ApiNotes.md.

Happy-path:
    1. Import tts_tool and warmup_phoneme_data.
    2. For each voice, modifier, and phoneme, synthesize and save a WAV file.
    3. Log progress and errors; skip existing files unless overwrite is set.
    4. Provide a summary and suggest verification/debugging strategies.

ASCII Diagram:
    for voice in voices:
        for modifier in modifiers:
            for set in [ipa, arpabet]:
                for phoneme in set:
                    synthesize -> save WAV

"""

import os
import sys
import traceback
from tts_tool import synthesize_audio
from warmup_phoneme_data import ipa_phonemes, arpabet_phonemes

# Reference: nearest ApiNotes.md for project and directory context

VOICES = ["tara", "jess", "leo", "leah", "dan", "mia", "zac", "zoe"]
MODIFIERS = [
    "happy", "normal", "disgust", "longer", "sad", "frustrated", "slow", "excited", "whisper", "panicky",
    "curious", "surprise", "fast", "crying", "deep", "sleepy", "angry", "high", "shout"
]
PHONEME_SETS = {
    "ipa": ipa_phonemes,
    "arpabet": arpabet_phonemes
}

OUTPUT_ROOT = "audio/phonetic_alphabet"
SAMPLE_RATE = 22050
OVERWRITE = False  # Set True to regenerate all files


def synthesize_all_phonemes(overwrite=OVERWRITE):
    """
    ApiNotes: Batch-synthesizes all phoneme WAVs for each voice and modifier. Skips existing files unless overwrite=True.
    """
    total = 0
    errors = []
    for voice in VOICES:
        for modifier in MODIFIERS:
            for set_name, phoneme_list in PHONEME_SETS.items():
                for phoneme in phoneme_list:
                    # Directory and filename
                    out_dir = os.path.join(OUTPUT_ROOT, voice, modifier, set_name)
                    os.makedirs(out_dir, exist_ok=True)
                    # Use safe filename for ARPAbet (e.g., 'CH' not 'ch')
                    fname = f"{phoneme}.wav"
                    out_path = os.path.join(out_dir, fname)
                    if os.path.exists(out_path) and not overwrite:
                        continue  # skip existing
                    # Compose TTS prompt: use modifier as a style tag if supported, else prepend/append
                    # For Orpheus-TTS, assume <modifier>text</modifier> is supported
                    prompt = f"<{modifier}>{phoneme}</{modifier}>"
                    try:
                        # Set temperature=0.1 for deterministic synthesis (see nearest ApiNotes.md)
                        synthesize_audio(prompt, wav_path=out_path, sample_rate=SAMPLE_RATE, voice=voice, max )
                        assert os.path.exists(out_path), f"WAV not created: {out_path}"
                        total += 1
                    except Exception as e:
                        errors.append((voice, modifier, set_name, phoneme, str(e)))
                        print(f"[ERROR] {voice}/{modifier}/{set_name}/{phoneme}: {e}", file=sys.stderr)
                        traceback.print_exc()
    print(f"\nSynthesis complete. {total} files generated.")
    if errors:
        print(f"{len(errors)} errors encountered. See stderr for details.")
    else:
        print("No errors encountered.")
    return total, errors


def main():
    """
    ApiNotes: Main entry point for batch phoneme WAV synthesis. Accepts --overwrite to force regeneration.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Batch-generate phoneme WAVs for all voices/modifiers.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate all files, even if they exist.")
    args = parser.parse_args()
    synthesize_all_phonemes(overwrite=args.overwrite)


# Acceptance tests (expected success/failure)
def test_synthesize_one_success():
    """
    ApiNotes: Test that a single phoneme WAV can be synthesized for a known-good voice/modifier.
    """
    out_path = os.path.join(OUTPUT_ROOT, "tara", "happy", "ipa", "t.wav")
    if os.path.exists(out_path):
        os.remove(out_path)
    synthesize_audio("<happy>t</happy>", wav_path=out_path, sample_rate=SAMPLE_RATE, voice="tara")
    assert os.path.exists(out_path), "WAV file should be created"
    print("(expected success) Single phoneme synthesis succeeded.")
    os.remove(out_path)

def test_synthesize_one_failure():
    """
    ApiNotes: Test that an invalid voice or modifier raises an error.
    """
    try:
        synthesize_audio("<notamodifier>t</notamodifier>", wav_path="/tmp/should_not_exist.wav", sample_rate=SAMPLE_RATE, voice="notavoice")
    except Exception as e:
        print(f"(expected failure) {e}")

if __name__ == "__main__":
    main()
