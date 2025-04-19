"""
ApiNotes.md (File-level) – dataset_helpers.py

Role:
    Centralizes all dataset, corpus, and phonetic feature extraction logic for the Stylometric Phonetic Encoder project.
    Provides a stable, testable API for persistence, corpus/pretrain loading, phonetic vectorization, and performance analysis.
    Now includes routines for extracting and summarizing openSMILE acoustic features for use in LLM prompt engineering.

Design Goals:
    - Decouple data and feature logic from UI and imperative orchestration.
    - Enable reuse and testing of phonetic/statistical routines.
    - Support future expansion (e.g., new feature types, corpus formats, LLM prompt aids).
    - Allow in-process extraction and summarization of openSMILE features for prompt injection.

Architectural Constraints:
    - All functions must be stateless and not reference UI or main-loop state.
    - Imports from main file are not allowed; all dependencies must be explicit.
    - All code must be justifiable against this ApiNotes and referenced directory/project-level ApiNotes.
    - openSMILE must be available and callable from Python (via subprocess or a wrapper).
    - Logging must use the unified logging_utils module for all log output.

Happy-Path:
    1. Load or save datasets (NPZ with metadata).
    2. Load corpus or pretrain data (JSON).
    3. Extract phonemes, vectors, and compute similarities.
    4. Analyze past performance for adaptive prompting.
    5. Extract and summarize openSMILE features for example-based LLM prompting.
    6. All logic is callable from the main application or tests.

ASCII Diagram:
    +-------------------+
    |  dataset_helpers  |
    +-------------------+
        |   |   |   |   |
        v   v   v   v   v
    [load/save][corpus][phoneme][analysis][acoustic_examples]
"""

import os
import json
import numpy as np
import subprocess
from typing import Any, Dict, List, Tuple, Optional
import sys
from nltk.corpus import cmudict

# Unified logging import
from logging_utils import log_message
from phonetic_helpers import text_to_phonemes, phoneme_vec, cosine, eightword_harmonic_score


def load_warmup_data(
    warmup_dir: str,
    eightword_mode: bool = False,
    log_file: Optional[Any] = None
) -> Tuple[List[List[float]], List[int]]:
    """
    ApiNotes: Loads all .json files from the specified warmup directory.
    For each entry, computes phonetic similarity (and optionally eightword score) between 'original' and 'paraphrased'.
    Returns (features, labels) lists suitable for pretrain/bootstrapping.
    """
    features: List[List[float]] = []
    labels: List[int] = []
    processed = 0
    skipped = 0

    if not os.path.isdir(warmup_dir):
        msg = f"[WARN] Warmup data directory not found: {warmup_dir}"
        print(msg)
        if log_file:
            log_message("WARN", msg, log_file)
        return [], []

    for fname in os.listdir(warmup_dir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(warmup_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            for idx, entry in enumerate(data):
                original_text = entry.get("original", [""])[0]
                paraphrased_text = entry.get("paraphrased", [""])[0]
                label = entry.get("label")
                if not original_text or not paraphrased_text or label is None:
                    skipped += 1
                    if log_file:
                        log_message("SKIP", f"Missing text or label in {fname} entry {idx}", log_file)
                    continue
                try:
                    original_phonemes = text_to_phonemes(original_text)
                    paraphrased_phonemes = text_to_phonemes(paraphrased_text)
                    if not original_phonemes or not paraphrased_phonemes:
                        skipped += 1
                        if log_file:
                            log_message("SKIP", f"No phonemes extracted for {fname} entry {idx}", log_file)
                        continue
                    original_vec = phoneme_vec(original_phonemes)
                    paraphrased_vec = phoneme_vec(paraphrased_phonemes)
                    sim_cos = cosine(original_vec, paraphrased_vec)
                    final_sim = sim_cos
                    if eightword_mode:
                        ew_orig = eightword_harmonic_score(original_text)
                        ew_para = eightword_harmonic_score(paraphrased_text)
                        ew_sim = 1.0 - abs(ew_orig - ew_para)
                        final_sim = (sim_cos + ew_sim) / 2.0
                    features.append([final_sim])
                    labels.append(label)
                    processed += 1
                    if log_file:
                        log_message("PROCESSED", f"{fname} entry {idx}: Similarity={final_sim:.3f}, label={label}", log_file)
                except Exception as e:
                    skipped += 1
                    if log_file:
                        log_message("ERROR", f"Processing {fname} entry {idx}: {e}", log_file)
        except Exception as e:
            if log_file:
                log_message("ERROR", f"Failed to load {fpath}: {e}", log_file)
            continue

    msg = f"[INFO] Loaded {processed} warmup entries, skipped {skipped}"
    print(msg)
    if log_file:
        log_message("SUMMARY", msg, log_file)
    return features, labels

# -----------------------------------------------------------------------------
# 📦 Dataset persistence
# -----------------------------------------------------------------------------

def load_dataset(path: str) -> Tuple[List[List[float]], List[int], Dict[str, Any]]:
    """
    ApiNotes: Loads dataset arrays X, y, and metadata from a .npz file.
    Returns (X, y, metadata) where metadata is always a dict.
    """
    metadata: Dict[str, Any] = {}
    if os.path.isfile(path):
        try:
            d = np.load(path, allow_pickle=True)
            X = d['X'].tolist() if 'X' in d else []
            y = d['y'].tolist() if 'y' in d else []
            # --- FIX: Always look for 'meta' and 'metadata' ---
            meta = None
            if 'metadata' in d:
                meta = d['metadata']
            elif 'meta' in d:
                meta = d['meta']
            if meta is not None:
                if isinstance(meta, np.ndarray):
                    NumpyData = meta.item() if meta.size == 1 else {}
                    if isinstance(NumpyData, dict):
                        metadata = NumpyData
                    else:
                        metadata = {}
                elif isinstance(meta, dict):
                    metadata = meta
            return X, y, metadata
        except Exception as e:
            print(f"[ERROR] Could not load dataset {path}: {e}")
    return [], [], metadata

def save_dataset(filename, X, y, meta=None):
    """
    ApiNotes: Saves dataset arrays X, y, and optional meta to a .npz file.
    Ensures all arrays are homogeneous or uses dtype=object for variable-length/inhomogeneous data.
    Raises AssertionError with context if shapes are not as expected.
    """
    def is_inhomogeneous(arr):
        first_shape = np.shape(arr[0])
        return any(np.shape(a) != first_shape for a in arr)

    if len(X) > 0 and is_inhomogeneous(X):
        X_arr = np.array(X, dtype=object)
    else:
        X_arr = np.array(X)
    if len(y) > 0 and is_inhomogeneous(y):
        y_arr = np.array(y, dtype=object)
    else:
        y_arr = np.array(y)
    if meta is not None:
        # Save as both 'meta' and 'metadata' for backward compatibility
        meta_arr = np.array(meta, dtype=object)
        np.savez(filename, X=X_arr, y=y_arr, meta=meta_arr, metadata=meta_arr)
    else:
        np.savez(filename, X=X_arr, y=y_arr)

    assert X_arr.dtype == object or len(set([np.shape(x) for x in X])) == 1, \
        "ApiNotes: X must be homogeneous or saved as dtype=object"
    assert y_arr.dtype == object or len(set([np.shape(yy) for yy in y])) == 1, \
        "ApiNotes: y must be homogeneous or saved as dtype=object"

# -----------------------------------------------------------------------------
# 🗃️ Corpus and pretrain data loading
# -----------------------------------------------------------------------------

def load_corpus(paths: List[str]) -> List[Dict[str, Any]]:
    corpus: List[Dict[str, Any]] = []
    for p in paths:
        try:
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
                corpus.extend(data.get('corpus', []))
        except Exception as e:
            print(f"[ERROR] Failed to load corpus file {p}: {e}")
    return corpus

# -----------------------------------------------------------------------------
# 🎤 WAV and openSMILE feature extraction from JSON dataset
# -----------------------------------------------------------------------------

def generate_wavs_and_features_from_json(
    json_path: str,
    audio_dir: str,
    config_path: str = "conf/opensmile/emo_large.conf",
    opensmile_bin: str = "SMILExtract",
    log_file: Optional[Any] = None,
    tts_func=None
) -> list:
    """
    ApiNotes: For each entry in a JSON dataset, synthesize WAVs for each text field (original, paraphrased),
    then extract openSMILE features from the generated WAVs. Returns a list of dicts with text, wav path, and features.
    """
    assert tts_func is not None, "tts_func (e.g., synthesize_audio) must be provided"
    assert os.path.isfile(json_path), f"JSON file not found: {json_path}"
    os.makedirs(audio_dir, exist_ok=True)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = []
    for idx, entry in enumerate(data):
        for field in ("original", "paraphrased"):
            text = entry.get(field)
            if isinstance(text, list):
                text = text[0]
            if not text:
                if log_file:
                    log_message("SKIP", f"No text for {field} in entry {idx}", log_file)
                continue
            wav_path = os.path.join(audio_dir, f"{os.path.splitext(os.path.basename(json_path))[0]}_{idx}_{field}.wav")
            try:
                tts_func(text, wav_path=wav_path)
                features = extract_opensmile_features(wav_path, config_path, opensmile_bin)
                results.append({
                    "text": text,
                    "wav_path": wav_path,
                    "features": features,
                    "field": field,
                    "entry_idx": idx
                })
                if log_file:
                    log_message("SUCCESS", f"WAV+features for {field} entry {idx} at {wav_path}", log_file)
            except Exception as e:
                if log_file:
                    log_message("ERROR", f"Failed to process {field} entry {idx}: {e}", log_file)
    return results

# -----------------------------------------------------------------------------
# 🧬 Phonetic feature extraction and scoring
# -----------------------------------------------------------------------------

from phonetic_helpers import (
    text_to_phonemes,
    phoneme_vec,
    cosine,
    text_syllables,
    eightword_harmonic_score,
)

# -----------------------------------------------------------------------------
# 📊 Performance analysis
# -----------------------------------------------------------------------------

def analyze_past_performance(data_file: str, log_file: Optional[Any] = None) -> Dict[str, Any]:
    try:
        X, y, metadata = load_dataset(data_file)
        if not X or not y:
            if log_file:
                log_message("INFO", "No past performance data available for analysis", log_file)
            return {"available": False}
        consecutive_similar_counter = metadata.get('consecutive_similar', 0)
        similarities = [x[0] for x in X]
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(similarities, bins)
        bin_stats = {}
        problematic_ranges = []
        for i in range(1, len(bins)):
            bin_mask = (bin_indices == i)
            bin_y = np.array(y)[bin_mask]
            bin_sim = np.array(similarities)[bin_mask]
            if len(bin_y) > 0:
                correct_h = sum((bin_sim >= 0.5) & (bin_y == 1))
                correct_d = sum((bin_sim < 0.5) & (bin_y == 0))
                total_h = sum(bin_y == 1)
                total_d = sum(bin_y == 0)
                h_accuracy = correct_h / total_h if total_h > 0 else 0
                d_accuracy = correct_d / total_d if total_d > 0 else 0
                overall_accuracy = (correct_h + correct_d) / len(bin_y)
                range_key = f"{bins[i-1]:.1f}-{bins[i]:.1f}"
                bin_stats[range_key] = {
                    "h_accuracy": h_accuracy,
                    "d_accuracy": d_accuracy,
                    "overall_accuracy": overall_accuracy,
                    "samples": len(bin_y),
                    "h_samples": total_h,
                    "d_samples": total_d
                }
                if overall_accuracy < 0.7 and len(bin_y) >= 5:
                    problematic_ranges.append({
                        "range": range_key,
                        "accuracy": overall_accuracy,
                        "samples": len(bin_y),
                        "h_accuracy": h_accuracy,
                        "d_accuracy": d_accuracy
                    })
        h_mean = np.mean([s for s, label in zip(similarities, y) if label == 1])
        d_mean = np.mean([s for s, label in zip(similarities, y) if label == 0])
        separation = h_mean - d_mean
        recommendations = []
        if separation < 0.15:
            recommendations.append("increase_contrast")
            consecutive_similar_counter += 1
        else:
            consecutive_similar_counter = 0
        if h_mean < 0.6:
            recommendations.append("strengthen_harmonic")
        if d_mean > 0.4:
            recommendations.append("strengthen_dissonant")
        if problematic_ranges:
            if any(p["h_accuracy"] < 0.6 for p in problematic_ranges):
                recommendations.append("focus_harmonic_clarity")
            if any(p["d_accuracy"] < 0.6 for p in problematic_ranges):
                recommendations.append("focus_dissonant_distinctness")
        if consecutive_similar_counter > 2:
            recommendations.append("increase_contrast")
        if consecutive_similar_counter > 4:
            recommendations.append("severe_contrast_penalty")
        metadata['consecutive_similar'] = consecutive_similar_counter
        if log_file:
            log_message("PERF", f"Total samples analyzed: {len(y)}", log_file)
            log_message("PERF", f"Harmonic mean: {h_mean:.3f}, Dissonant mean: {d_mean:.3f}", log_file)
            log_message("PERF", f"Separation: {separation:.3f}", log_file)
            log_message("PERF", f"Consecutive similar outputs: {consecutive_similar_counter}", log_file)
            log_message("PERF", f"Problematic ranges: {len(problematic_ranges)}", log_file)
            for pr in problematic_ranges:
                log_message("PERF", f"{pr['range']}: acc={pr['accuracy']:.2f}, samples={pr['samples']}", log_file)
            log_message("PERF", f"Recommendations: {recommendations}", log_file)
        return {
            "available": True,
            "samples": len(y),
            "h_mean": h_mean,
            "d_mean": d_mean,
            "separation": separation,
            "consecutive_similar": consecutive_similar_counter,
            "bin_stats": bin_stats,
            "problematic_ranges": problematic_ranges,
            "recommendations": recommendations,
            "metadata": metadata
        }
    except Exception as e:
        error_msg = f"[ERROR] Failed to analyze past performance: {e}"
        print(error_msg)
        if log_file:
            log_message("ERROR", error_msg, log_file)
        return {"available": False, "error": str(e)}

# -----------------------------------------------------------------------------
# 🎤 openSMILE feature extraction (standalone)
# -----------------------------------------------------------------------------

def extract_opensmile_features(wav_path: str, config_path: str = "conf/opensmile/emo_large.conf", opensmile_bin: str = "SMILExtract") -> Dict[str, float]:
    """
    Extracts openSMILE features from a WAV file using the specified config.
    Returns a dictionary of feature names to values.
    Assumes openSMILE is installed and available in PATH or via opensmile_bin.
    """
    assert os.path.isfile(wav_path), "WAV file must exist for feature extraction"
    assert os.path.isfile(config_path), "openSMILE config file must exist"
    out_csv = wav_path + ".smile.csv"
    cmd = [
        opensmile_bin,
        "-C", config_path,
        "-I", wav_path,
        "-O", out_csv
    ]
    subprocess.run(cmd, check=True)
    with open(out_csv, "r", encoding="utf-8") as f:
        lines = f.readlines()
    header = lines[0].strip().split(";")
    data = lines[1].strip().split(";")
    features = {k: float(v) for k, v in zip(header, data) if v.replace('.', '', 1).replace('-', '', 1).isdigit()}
    return features

def summarize_acoustic_examples(
    harmonic_wavs: List[str],
    dissonant_wavs: List[str],
    config_path: str,
    opensmile_bin: str = "conf/SMILExtract",
    feature_keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    harmonic_feats = []
    dissonant_feats = []
    feats = []
    for wav in harmonic_wavs:
        feats = extract_opensmile_features(wav, config_path, opensmile_bin)
        harmonic_feats.append(feats)
    for wav in dissonant_wavs:
        feats = extract_opensmile_features(wav, config_path, opensmile_bin)
        dissonant_feats.append(feats)
    if not feature_keys and harmonic_feats and len(harmonic_feats) > 0:
        feature_keys = list(harmonic_feats[0].keys())
    def mean_feats(feat_list, feature_keys):
        return {k: float(np.mean([f[k] for f in feat_list if k in f])) for k in feature_keys}
    harmonic_summary = mean_feats(harmonic_feats, feature_keys)
    dissonant_summary = mean_feats(dissonant_feats, feature_keys)
    return {
        "feats": feats,
        "harmonic": harmonic_summary,
        "dissonant": dissonant_summary,
        "feature_keys": feature_keys
    }

def format_acoustic_examples_for_prompt(
    harmonic_text: str,
    dissonant_text: str,
    harmonic_feats: Dict[str, float],
    dissonant_feats: Dict[str, float],
    feature_keys: Optional[List[str]] = None
) -> str:
    if not feature_keys:
        feature_keys = list(harmonic_feats.keys())
    harmonic_lines = [f"{k}: {harmonic_feats[k]:.2f}" for k in feature_keys if k in harmonic_feats]
    dissonant_lines = [f"{k}: {dissonant_feats[k]:.2f}" for k in feature_keys if k in dissonant_feats]
    return (
        "Here are examples and their acoustic features:\n\n"
        f"Harmonic Example:\nText: \"{harmonic_text}\"\nFeatures: " + ", ".join(harmonic_lines) + "\n\n"
        f"Dissonant Example:\nText: \"{dissonant_text}\"\nFeatures: " + ", ".join(dissonant_lines) + "\n\n"
        "Guidance:\n"
        "- Harmonic outputs should use repeated, smooth sounds and produce low pitch variation (e.g., low F0 stdev).\n"
        "- Dissonant outputs should use clashing, abrupt sounds and produce high pitch variation (e.g., high F0 stdev).\n"
        "- Aim for a clear, measurable difference in these features.\n"
    )

# ApiNotes: See file-level and directory-level ApiNotes.md for rationale and usage.