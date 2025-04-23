"""
ApiNotes.md (File-level) – warmup_phoneme_data.py

Role:
    Provides comprehensive warmup data for phonetic TTS synthesis, including IPA and ARPAbet phoneme lists.
    Intended for use in batch generation scripts for phoneme WAVs, supporting all voices and modifiers.

Design Goals:
    - Centralize phoneme definitions for both IPA and ARPAbet.
    - Enable easy extension or modification of phoneme sets.
    - Facilitate reproducible, comprehensive TTS warmup and benchmarking.
    - Reference nearest ApiNotes.md for project-level and directory-level context.

Architectural Constraints:
    - Data must be structured for direct use in imperative batch scripts.
    - All phoneme lists must be exhaustive for general American English (can be extended for other languages).
    - File must be importable as a module.
    - All changes must be reflected in project-level ApiNotes.md.

Happy-path:
    1. Import this file in a batch synthesis script.
    2. Iterate over ipa_phonemes and arpabet_phonemes for TTS synthesis.
    3. Use the symbol/code as the text prompt for isolated phoneme synthesis.

"""

# IPA phonemes (General American English, not exhaustive for all languages)
ipa_phonemes = [
    "p", "b", "t", "d", "k", "g", "m", "n", "ŋ", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h", "tʃ", "dʒ",
    "l", "r", "j", "w", "i", "ɪ", "e", "ɛ", "æ", "ɑ", "ɒ", "ɔ", "ʊ", "u", "ʌ", "ə", "ɜ", "aɪ", "aʊ", "ɔɪ", "eɪ", "oʊ"
]

# ARPAbet phonemes (General American English)
arpabet_phonemes = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY",
    "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"
]

# Example usage:
#   from warmup_phoneme_data import ipa_phonemes, arpabet_phonemes
#   for symbol in ipa_phonemes:
#       synthesize_audio(symbol, ...)
#   for code in arpabet_phonemes:
#       synthesize_audio(code, ...)
