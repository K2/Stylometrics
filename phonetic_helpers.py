"""
ApiNotes.md (File-level) – phonetic_helpers.py

Role:
    Provides stateless phonetic feature extraction and scoring helpers for the Stylometric Phonetic Encoder project.
    Used by dataset_helpers.py and other modules for ARPABET phoneme extraction, vectorization, similarity, and syllable counting.

Design Goals:
    - Decouple phonetic/statistical routines from dataset and orchestration logic.
    - Enable reuse and testing of phonetic helpers.
    - Support future expansion (e.g., new phoneme sets, scoring metrics).

Architectural Constraints:
    - All functions must be stateless and not reference UI or main-loop state.
    - Imports from main file are not allowed; all dependencies must be explicit.
    - All code must be justifiable against this ApiNotes and referenced directory/project-level ApiNotes.

Happy-Path:
    1. Extract phonemes from text using CMUdict.
    2. Vectorize phoneme lists for similarity.
    3. Compute cosine similarity.
    4. Count syllables.
    5. Compute eight-word harmonic score.

ASCII Diagram:
    +-------------------+
    | phonetic_helpers  |
    +-------------------+
        |   |   |   |   |
        v   v   v   v   v
    [phonemes][vector][cosine][syllables][eightword]
"""

import numpy as np
import sys
from nltk.corpus import cmudict

try:
    cmu_dict = cmudict.dict()
    phoneme_set = sorted({p for v in cmu_dict.values() for p in v[0]})
except Exception as e:
    print(f"[ERROR] Could not load CMUdict: {e}", file=sys.stderr)
    cmu_dict = {}
    phoneme_set = []

def text_to_phonemes(txt: str):
    try:
        words = txt.lower().split()
        phs = []
        for w in words:
            if w in cmu_dict:
                phs.extend(cmu_dict[w][0])
        return phs
    except Exception as e:
        print(f"[ERROR] Failed to extract phonemes: {e}", file=sys.stderr)
        return []

def phoneme_vec(phs):
    v = np.zeros(len(phoneme_set))
    for p in phs:
        try:
            v[phoneme_set.index(p)] += 1
        except Exception:
            pass
    return v

def cosine(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def text_syllables(txt: str) -> int:
    try:
        words = txt.lower().split()
        count = 0
        for w in words:
            if w in cmu_dict:
                count += sum(1 for p in cmu_dict[w][0] if p[-1].isdigit())
        return count
    except Exception as e:
        print(f"[ERROR] Failed to count syllables: {e}", file=sys.stderr)
        return 0

def _get_word_phonemes(w: str):
    return cmu_dict[w][0] if w in cmu_dict else []

def eightword_harmonic_score(sentence: str, stride: int = 1) -> float:
    words = sentence.lower().split()
    scores, width = [], len(phoneme_set)
    for i in range(0, max(1, len(words) - 7), stride):
        phs = [_p for w in words[i:i + 8] for _p in _get_word_phonemes(w)]
        if not phs:
            continue
        vec = np.zeros(width)
        for p in phs:
            try:
                vec[phoneme_set.index(p)] += 1
            except Exception:
                pass
        s = np.sum(vec)
        if s == 0:
            continue
        simpson = np.sum((vec / s) ** 2)
        scores.append(simpson)
    return float(np.mean(scores)) if scores else 0.0