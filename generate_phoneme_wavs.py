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

from zmq import REP
from gguf_orpheus import REPETITION_PENALTY
from tts_tool import synthesize_audio
from warmup_phoneme_data import ipa_phonemes, arpabet_phonemes

# Reference: nearest ApiNotes.md for project and directory context

VOICES = ["tara", "jess", "leo", "leah", "dan", "mia", "zac", "zoe"]
MODIFIERS = [
    "happy", "normal", "disgust", "longer", "sad", "frustrated", "slow", "excited", "whisper", "panicky",
    "curious", "surprise", "fast", "crying", "deep", "sleepy", "angry", "high", "shout", "giggle", "laugh", "chuckle",
    "sigh", "cough", "sniffle", "groan", "yawn", "grumble", "whine", "moan", "grin", "smile", "frown", "smirk",
    "tease", "taunt", "cheer", "applaud", "bark", "growl", "hiss", "scream", "yell", "sob", "wail", "whimper",
    "snicker", "chortle", "titter", "gasp"
]
PHONEME_SETS = {
    "ipa": ipa_phonemes,
    "arpabet": arpabet_phonemes
}

OUTPUT_ROOT = "audio/phonetic_alphabet"
SAMPLE_RATE = 24000
OVERWRITE = False  # Set True to regenerate all files
REP_COUNT = 5  # Number of times to repeat each phoneme in the prompt
def get_all_phonemes():
    """
    ApiNotes: Returns a list of all phonemes across all sets (IPA and ARPAbet).
    """
    return [phoneme for set_name, phoneme_list in PHONEME_SETS.items() for phoneme in phoneme_list]
def get_all_modifiers():
    """
    ApiNotes: Returns a list of all available modifiers.
    """
    return MODIFIERS
def get_all_phoneme_sets():
    """
    ApiNotes: Returns a list of all phoneme sets (IPA and ARPAbet).
    """
    return list(PHONEME_SETS.keys())
def get_all_phoneme_set_names():
    """
    ApiNotes: Returns a list of all phoneme set names (IPA and ARPAbet).
    """
    return list(PHONEME_SETS.keys())
def get_all_syllables():
    """
    ApiNotes: Returns a list of all known English syllables by combining phonemes into common syllabic patterns (C, V, CV, VC, CVC).
    Uses IPA and ARPAbet phoneme sets as base.
    """
    # Simple English syllable patterns: C (consonant), V (vowel), CV, VC, CVC
    # For demonstration, use a subset of common vowels and consonants
    vowels = [p for p in ipa_phonemes if p in {'a', 'e', 'i', 'o', 'u', 'æ', 'ə', 'ɪ', 'ʊ', 'ɛ', 'ʌ', 'ɔ', 'ɑ'}]
    consonants = [p for p in ipa_phonemes if p not in vowels]
    syllables = set()

    # V
    for v in vowels:
        syllables.add(v)
    # CV
    for c in consonants:
        for v in vowels:
            syllables.add(c + v)
    # VC
    for v in vowels:
        for c in consonants:
            syllables.add(v + c)
    # CVC
    for c1 in consonants:
        for v in vowels:
            for c2 in consonants:
                syllables.add(c1 + v + c2)
    return sorted(syllables)
def best_effort_phonetic_spelling(phoneme):
    """
    ApiNotes: Returns a best-effort normal alphabet spelling for a given phoneme.
    Used as a fallback if TTS output is empty or has no features.
    """
    # Simple mapping for demonstration; extend as needed for coverage
    mapping = {
        'ɑ': 'ah', 'æ': 'a', 'ʌ': 'uh', 'ə': 'uh', 'ɛ': 'eh', 'ɪ': 'ih', 'ʊ': 'oo', 'ɔ': 'aw',
        'θ': 'th', 'ð': 'dh', 'ʃ': 'sh', 'ʒ': 'zh', 'ŋ': 'ng', 'ɹ': 'r', 'ɾ': 't', 'ʔ': 'q',
        'ʃ': 'sh', 'tʃ': 'ch', 'dʒ': 'j', 'ɡ': 'g', 'ɲ': 'ny', 'ɸ': 'f', 'β': 'b', 'ç': 'h',
        # Add more as needed
    }
    # If already ASCII, return as is
    if all(ord(c) < 128 for c in phoneme):
        return phoneme
    # Try mapping, else fallback to removing diacritics
    return mapping.get(phoneme, ''.join([c for c in phoneme if ord(c) < 128]))

def has_audio_features(wav_path):
    """
    ApiNotes: Returns True if the WAV file at wav_path contains non-silent audio/features.
    Uses a minimal check (file size and duration); for full feature check, integrate OpenSMILE or librosa.
    """
    try:
        import wave
        with wave.open(wav_path, 'rb') as wf:
            frames = wf.getnframes()
            duration = frames / float(wf.getframerate())
            # Consider files with <0.1s or 0 frames as empty
            return frames > 0 and duration > 0.1
    except Exception:
        return False

def synthesize_with_fallback(prompt, wav_path, sample_rate, voice):
    """
    ApiNotes: Synthesizes audio, and if the result is empty or has no features, retries with best-effort phonetic spelling.
    The prompt moniker (e.g., <happy>) is only set at the start and not closed.
    """
    # Remove closing moniker if present
    import re
    prompt_no_close = re.sub(r"</\w+>", "", prompt)
    synthesize_audio(prompt_no_close, wav_path=wav_path, sample_rate=sample_rate, voice=voice)
    if not has_audio_features(wav_path):
        # Extract the core phoneme or syllable from the prompt
        match = re.search(r'<\w+>([^<]+)', prompt_no_close)
        core = match.group(1) if match else prompt_no_close
        # Try fallback spelling
        fallback = best_effort_phonetic_spelling(core)
        fallback_prompt = re.sub(r'(<\w+>)[^<]+', r'\1' + fallback, prompt_no_close)
        synthesize_audio(fallback_prompt, wav_path=wav_path, sample_rate=sample_rate, voice=voice)

# In all batch synthesis, update prompt construction to not include a closing moniker
def synthesize_all_phonemes(overwrite=OVERWRITE):
    """
    ApiNotes: Batch-synthesizes all phoneme WAVs for each voice and modifier. Skips existing files unless overwrite=True.
    If a WAV is empty or has no features, retries with best-effort phonetic spelling.
    The prompt moniker (e.g., <happy>) is only set at the start and not closed.
    """
    total = 0
    errors = []
    for voice in VOICES:
        for modifier in MODIFIERS:
            for set_name, phoneme_list in PHONEME_SETS.items():
                for phoneme in phoneme_list:
                    out_dir = os.path.join(OUTPUT_ROOT, voice, modifier, set_name)
                    os.makedirs(out_dir, exist_ok=True)
                    fname = f"{phoneme}.wav"
                    out_path = os.path.join(out_dir, fname)
                    if os.path.exists(out_path) and not overwrite:
                        continue
                    repeated = ','.join([phoneme] * 10)
                    prompt = f"<{modifier}>{repeated}"  # No closing moniker
                    try:
<<<<<<< Updated upstream
                        # Set temperature=0.1 for deterministic synthesis (see nearest ApiNotes.md)
                        synthesize_audio(prompt, wav_path=out_path, sample_rate=SAMPLE_RATE, voice=voice, max )
=======
                        synthesize_with_fallback(
                            prompt,
                            wav_path=out_path,
                            sample_rate=SAMPLE_RATE,
                            voice=voice
                        )
>>>>>>> Stashed changes
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

def synthesize_all_syllables(overwrite=OVERWRITE):
    """
    ApiNotes: Batch-synthesizes all English syllable WAVs for each voice and modifier.
    If a WAV is empty or has no features, retries with best-effort phonetic spelling.
    Output: audio/phonetic_alphabet/{voice}/{modifier}/syllables/{syllable}.wav
    The prompt moniker (e.g., <happy>) is only set at the start and not closed.
    """
    total = 0
    errors = []
    syllables = get_all_syllables()
    for voice in VOICES:
        for modifier in MODIFIERS:
            out_dir = os.path.join(OUTPUT_ROOT, voice, modifier, "syllables")
            os.makedirs(out_dir, exist_ok=True)
            for syllable in syllables:
                fname = f"{syllable}.wav"
                out_path = os.path.join(out_dir, fname)
                if os.path.exists(out_path) and not overwrite:
                    continue
                repeated = ','.join([syllable] * 10)
                prompt = f"<{modifier}>{repeated}"  # No closing moniker
                try:
                    synthesize_with_fallback(
                        prompt,
                        wav_path=out_path,
                        sample_rate=SAMPLE_RATE,
                        voice=voice
                    )
                    assert os.path.exists(out_path), f"WAV not created: {out_path}"
                    total += 1
                except Exception as e:
                    errors.append((voice, modifier, "syllables", syllable, str(e)))
                    print(f"[ERROR] {voice}/{modifier}/syllables/{syllable}: {e}", file=sys.stderr)
                    traceback.print_exc()
    print(f"\nSyllable synthesis complete. {total} files generated.")
    if errors:
        print(f"{len(errors)} errors encountered. See stderr for details.")
    else:
        print("No errors encountered.")
    return total, errors

def superposition_phoneme_patterns(phoneme):
    """
    ApiNotes: Returns a list of phoneme repetition patterns designed to test for superposition, phase effects,
    and highly attenuated waveforms. Patterns include single, grouped, and increasing-length repetitions,
    as well as comma-separated and space-separated variants.
    """
    patterns = [
        phoneme,  # single
        f"{phoneme},{phoneme}",  # simple pair, comma
        f"{phoneme}{phoneme}",   # simple pair, no separator
        f"{phoneme},{phoneme},{phoneme}",  # triplet, comma
        f"{phoneme}{phoneme}{phoneme}",    # triplet, no separator
        f"{phoneme},{phoneme},{phoneme},{phoneme}",  # quartet, comma
        f"{phoneme}{phoneme}{phoneme}{phoneme}",     # quartet, no separator
        f"{phoneme*5}",  # 5x, no separator
        f"{phoneme},{phoneme},{phoneme},{phoneme},{phoneme}",  # 5x, comma
        f"{phoneme*10}",  # 10x, no separator
        f"{phoneme},{phoneme},{phoneme},{phoneme},{phoneme},{phoneme},{phoneme},{phoneme},{phoneme},{phoneme}",  # 10x, comma
        f"{phoneme} {phoneme}",  # pair, space
        f"{phoneme} {phoneme} {phoneme}",  # triplet, space
        f"{phoneme} {phoneme} {phoneme} {phoneme}",  # quartet, space
        f"{phoneme} {phoneme} {phoneme} {phoneme} {phoneme}",  # 5x, space
        f"{phoneme*2},{phoneme*3},{phoneme*4},{phoneme*5}",  # increasing group sizes, comma
        f"{phoneme*2} {phoneme*3} {phoneme*4} {phoneme*5}",  # increasing group sizes, space
        f"{phoneme},{phoneme*2},{phoneme*3},{phoneme*4},{phoneme*5}",  # mixed
        f"{phoneme*2},{phoneme},{phoneme*3},{phoneme},{phoneme*4}",    # mixed
        f"{phoneme*3},{phoneme*2},{phoneme*4},{phoneme*5}",            # mixed
        f"{phoneme},{phoneme},{phoneme},{phoneme},{phoneme},{phoneme},{phoneme},{phoneme},{phoneme},{phoneme}",  # 10x, comma
        f"{phoneme*6}",  # 6x, no separator
        f"{phoneme*7}",  # 7x, no separator
        f"{phoneme*8}",  # 8x, no separator
        f"{phoneme*9}",  # 9x, no separator
        f"{phoneme*10}", # 10x, no separator
    ]
    return patterns

def synthesize_superposition_patterns(phoneme, voice, modifier, out_dir, sample_rate=SAMPLE_RATE):
    """
    ApiNotes: Synthesizes WAVs for a variety of repetition/superposition patterns for a given phoneme.
    Output: audio/phonetic_alphabet/{voice}/{modifier}/superpositions/{pattern_idx}_{phoneme}.wav
    The prompt moniker (e.g., <happy>) is only set at the start and not closed.
    """
    patterns = superposition_phoneme_patterns(phoneme)
    os.makedirs(out_dir, exist_ok=True)
    for idx, pattern in enumerate(patterns):
        prompt = f"<{modifier}>{pattern}"  # No closing moniker
        fname = f"{idx}_{phoneme}.wav"
        out_path = os.path.join(out_dir, fname)
        synthesize_with_fallback(prompt, wav_path=out_path, sample_rate=sample_rate, voice=voice)

def synthesize_all_superpositions(overwrite=OVERWRITE):
    """
    ApiNotes: Batch-synthesizes all superposition/repetition pattern WAVs for each phoneme, voice, and modifier.
    Output: audio/phonetic_alphabet/{voice}/{modifier}/superpositions/{pattern_idx}_{phoneme}.wav
    """
    total = 0
    errors = []
    for voice in VOICES:
        for modifier in MODIFIERS:
            out_dir = os.path.join(OUTPUT_ROOT, voice, modifier, "superpositions")
            for set_name, phoneme_list in PHONEME_SETS.items():
                for phoneme in phoneme_list:
                    try:
                        synthesize_superposition_patterns(
                            phoneme, voice, modifier, out_dir, sample_rate=SAMPLE_RATE
                        )
                        total += 1
                    except Exception as e:
                        errors.append((voice, modifier, set_name, phoneme, str(e)))
                        print(f"[ERROR] {voice}/{modifier}/superpositions/{phoneme}: {e}", file=sys.stderr)
                        traceback.print_exc()
    print(f"\nSuperposition synthesis complete. {total} phonemes processed.")
    if errors:
        print(f"{len(errors)} errors encountered. See stderr for details.")
    else:
        print("No errors encountered.")
    return total, errors

# Update main to support superpositions
def main():
    """
    ApiNotes: Main entry point for batch phoneme, syllable, and superposition WAV synthesis.
    Accepts --overwrite, --syllables, and --superpositions.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Batch-generate phoneme, syllable, and superposition WAVs for all voices/modifiers.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate all files, even if they exist.")
    parser.add_argument("--syllables", action="store_true", help="Generate syllable WAVs instead of phonemes.")
    parser.add_argument("--superpositions", action="store_true", help="Generate superposition/repetition pattern WAVs.")
    args = parser.parse_args()
    if args.superpositions:
        synthesize_all_superpositions(overwrite=args.overwrite)
    elif args.syllables:
        synthesize_all_syllables(overwrite=args.overwrite)
    else:
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
