"""
ApiNotes.md (File-level) – stylometric_phonetic_encoder_ollama.py

Role:
    Main orchestrator for phonetic resonance analysis, dataset generation, and evaluation.
    Implements imperative workflow, TUI, and LLM integration for stylistic text variation and pair generation.
    Now integrates direct access to extracted OpenSMILE features for LLM-based pattern discovery and supports bootstrapping
    from LLM-compiled feature associations (see llm_feature_bootstrap.py).

Design Goals:
    - Provide reproducible, extensible research platform for stylometric phonetic analysis.
    - Support incremental dataset persistence, adaptive prompting, and live evaluation.
    - Modularize feature extraction, data loading, and analysis logic for maintainability and testability.
    - Centralize all LLM (Ollama) chat invocations in call_llm/chat_llm for maintainability and reduced line count.
    - Expose all extracted phonetic features to the encoder and LLM for direct pattern analysis and bootstrapping.
    - Enable the LLM to receive both original text and extracted features, and query for patterns indicating resonance or dissonance.
    - Integrate with LLM feature bootstrapping (llm_feature_bootstrap.py) for persistent, reusable feature associations.

Architectural Constraints:
    - All non-UI, stateless, or reusable logic is moved to helper modules.
    - UI logic and imperative orchestration remain in this file.
    - All functions and data structures are documented and justified via ApiNotes.
    - File size monitored; refactor or split if exceeding 1/3 context window.
    - All feature extraction and LLM query logic must be observable and testable.
    - Must reference file, directory, and project-level ApiNotes.md for all design and interface decisions.

Happy-Path:
    1. Parse CLI arguments and initialize logging, random seeds, and UI.
    2. Load or reset persistent dataset and metadata.
    3. Optionally ingest pretrain data and analyze past performance.
    4. Load and randomize corpus entries.
    5. For each entry and variation:
        - Generate stylistic variations and harmonic/dissonant pairs via LLM (using chat_llm).
        - Extract phonetic features and compute similarity.
        - Expose all features to the phonetic encoder and LLM.
        - Query the LLM for patterns in the features that indicate resonance or dissonance.
        - If OpenSMILE analysis indicates low quality, provide feedback and retry LLM generation.
        - Update statistics, UI, and logs.
    6. Save updated dataset and metadata.
    7. Optionally train and evaluate a classifier.
    8. Present results and finalize logs.

ASCII Diagram:
    +-------------------+
    |  CLI & Logging    |
    +-------------------+
              |
              v
    +-------------------+
    |  Dataset Load/Save|
    +-------------------+
              |
              v
    +-------------------+
    |  Corpus/Pretrain  |
    +-------------------+
              |
              v
    +-------------------+
    |  Main Loop:       |
    |  - LLM Variation  |
    |  - Pair Gen       |
    |  - Feature Extract|
    |  - Expose Features|
    |  - LLM Pattern Q  |
    |  - Bootstrapping  |
    |  - Feedback/Retry |
    |  - Stats/Log/UI   |
    +-------------------+
              |
              v
    +-------------------+
    |  Save & Evaluate  |
    +-------------------+
"""

import argparse
from collections import defaultdict
import json
import os
from pprint import pp
import re
import sys
import datetime
import random
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Optional, TextIO, Generator, Union

import numpy as np
import nltk
from typing import Any, Optional
from nltk.corpus import cmudict
from rich import print_json
from rich.progress import track
from rich.table import Table, box
from rich.panel import Panel
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.style import Style
from rich.markdown import Markdown
from rich.syntax import Syntax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import ollama
from logging_utils import log_message
from pretrain_refiner import generate_high_quality_pretrain
from tts_tool import synthesize_audio
from bootstrap import run_llm_feature_bootstrap, prompt_llm_for_feature_patterns

from acoustic_features import AcousticFeatureExtractor
from dataset_helpers import (
    extract_opensmile_features, load_dataset, save_dataset, load_corpus, analyze_past_performance,
    text_to_phonemes, phoneme_vec, cosine, text_syllables, eightword_harmonic_score,
    summarize_acoustic_examples, format_acoustic_examples_for_prompt, load_warmup_data
)
# default config path for tsfm_adapter
TSFM_CONFIG_PATH = "conf/opensmile/emo_large.conf"

console = Console()
def load_pretrain_data_from_csv(
    audio_dir: str,
    label_map: dict,
    feature_keys: list = [],
    log_file: str = "",
    no_dataframe: bool = False  # option to use manual parsing instead of DataFrame adapter
) -> Tuple[List[List[float]], List[int]]:
    """
    ApiNotes: Loads pretrain data from OpenSMILE ARFF/CSV files in audio_dir.
    Handles ARFF-style headers, finds @data, then parses the CSV data row.
    Uses filename markers to assign labels (e.g., *_harmonic.wav.smile.csv, *_paraphrased.wav.smile.csv).
    Returns (features, labels) lists for training.
    """
    features = []
    labels = []
    processed = 0
    skipped = 0

    for fname in os.listdir(audio_dir):
        if not fname.endswith(".wav.smile.csv"):
            continue
        csv_path = os.path.join(audio_dir, fname)
        # Determine label for this file
        label_assigned = None
        for key, val in label_map.items():
            if key in fname:
                label_assigned = val
                break
        assert label_assigned is not None, f"Unrecognized label for file: {fname}"
        # Manual parsing branch
        if no_dataframe:
            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                # Parse ARFF header and locate @data
                data_start = None
                header_keys = []
                for idx, line in enumerate(lines):
                    if line.strip().lower() == "@data":
                        data_start = idx
                        break
                    if line.startswith("@attribute"):
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            header_keys.append(parts[1])
                if data_start is None or data_start + 1 >= len(lines):
                    skipped += 1
                    log_message("ERROR", f"[ERROR] No @data section found in {fname}", log_file)
                    continue
                # Read first data row
                data_line = lines[data_start + 1].strip()
                data_values = [v.strip() for v in data_line.split(',')]
                use_keys = feature_keys if feature_keys else header_keys
                key_to_idx = {k: i for i, k in enumerate(header_keys)}
                # Extract requested features
                feats = []
                missing = []
                for k in use_keys:
                    idx = key_to_idx.get(k)
                    if idx is not None and idx < len(data_values):
                        try:
                            feats.append(float(data_values[idx]))
                        except Exception:
                            feats.append(float('nan'))
                    else:
                        missing.append(k)
                if missing and log_file:
                    log_message("WARN", f"[WARN] {fname} missing features: {missing}", log_file)
                features.append(feats)
                labels.append(label_assigned)
                processed += 1
            except Exception as e:
                skipped += 1
                log_message("ERROR", f"[ERROR] Failed manual parse {fname}: {e}", log_file)
            continue
        # DataFrame adapter branch
        try:
            df = arff_csv_to_timeseries(csv_path, TSFM_CONFIG_PATH)
            feat_df = df.drop(columns=["time"])
            for row in feat_df.itertuples(index=False, name=None):
                features.append(list(row))
                labels.append(label_assigned)
            processed += len(feat_df)
        except Exception as e:
            skipped += 1
            log_message("ERROR", f"[ERROR] Failed DataFrame parse {fname}: {e}", log_file)
            continue

    msg = f"[INFO] Loaded {processed} pretrain entries from CSV, skipped {skipped}"
    print(msg)
    log_message("INFO", msg, log_file)
    return features, labels

def read_opensmile_csv(csv_path: str, feature_keys: list = []) -> dict:
    """
    Reads an OpenSMILE ARFF/CSV file and returns a dict of feature_name: value.
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # Find header
    header_keys = []
    data_start = None
    for idx, line in enumerate(lines):
        if line.strip().lower() == "@data":
            data_start = idx
            break
        if line.startswith("@attribute"):
            parts = line.strip().split()
            if len(parts) >= 3:
                header_keys.append(parts[1])
    if data_start is None or data_start + 1 >= len(lines):
        raise ValueError("No @data section found in CSV")
    data_line = lines[data_start + 1].strip()
    data_values = [v.strip() for v in data_line.split(",")]
    if not feature_keys:
        feature_keys = header_keys
    key_to_idx = {k: i for i, k in enumerate(header_keys)}
    features = {}
    for k in feature_keys:
        idx = key_to_idx.get(k)
        if idx is not None and idx < len(data_values):
            try:
                features[k] = float(data_values[idx])
            except Exception:
                features[k] = float('nan')
    return features
# def scan_csvs_for_phonetic_patterns(
#     audio_dir: str,
#     label_map: dict,
#     log_file: str = "",
#     max_files: int = 100,
#     feature_keys: list = []
#     ):
#     """
#     ApiNotes: Scans all OpenSMILE CSVs in audio_dir, aggregates all values for each feature across files,
#     and sends the full table to the LLM for analysis. The LLM can then observe how each feature varies
#     across different contexts (spoken words) and labels.
#     """
#     # Reference: file-level ApiNotes.md for imperative, observable, and testable design.
#     from bootstrap import prompt_llm_for_feature_patterns

#     # Step 1: Gather all feature vectors and metadata
#     all_feature_rows = []
#     all_labels = []
#     all_filenames = []
#     processed = 0
#     skipped = 0
#     all_header_keys = set()

#     for fname in os.listdir(audio_dir):
#         if not fname.endswith(".wav.smile.csv"):
#             continue
#         # Determine label from filename marker
#         label = None
#         for marker, lab in label_map.items():
#             if marker in fname:
#                 label = lab
#                 break
#         if label is None:
#             skipped += 1
#             log_message("INFO", f"[SKIP] Could not determine label for {fname}", log_file)
#             continue
#         csv_path = os.path.join(audio_dir, fname)
#         try:
#             # Parse ARFF-style OpenSMILE CSV
#             with open(csv_path, "r", encoding="utf-8") as f:
#                 lines = f.readlines()
#             data_start = None
#             header_keys = []
#             for idx, line in enumerate(lines):
#                 if line.strip().lower() == "@data":
#                     data_start = idx
#                     break
#                 if line.startswith("@attribute"):
#                     parts = line.strip().split()
#                     if len(parts) >= 3:
#                         header_keys.append(parts[1])
#             if data_start is None or data_start + 1 >= len(lines):
#                 skipped += 1
#                 log_message("ERROR", f"[ERROR] No @data section found in {fname}", log_file)
#                 continue
#             data_line = lines[data_start + 1].strip()
#             data_values = [v.strip() for v in data_line.split(",")]
#             all_header_keys.update(header_keys)
#             # If feature_keys is empty, use all
#             use_keys = feature_keys if feature_keys else header_keys
#             key_to_idx = {k: i for i, k in enumerate(header_keys)}
#             row = {}
#             for k in use_keys:
#                 idx = key_to_idx.get(k)
#                 if idx is not None and idx < len(data_values):
#                     try:
#                         row[k] = float(data_values[idx])
#                     except Exception:
#                         row[k] = float('nan')
#                 else:
#                     row[k] = None
#             row['label'] = label
#             row['filename'] = fname
#             all_feature_rows.append(row)
#             all_labels.append(label)
#             all_filenames.append(fname)
#             processed += 1
#             if processed >= max_files:
#                 break
#         except Exception as e:
#             skipped += 1
#             log_message("ERROR", f"[ERROR] Failed to process {fname}: {e}", log_file)
#             continue

#     # Step 2: Build a table (list of dicts) with all feature values, labels, and filenames
#     #         This allows the LLM to see how each feature varies across samples and labels.
#     if not all_feature_rows:
#         log_message("ERROR", "[ERROR] No feature rows collected for LLM analysis.", log_file)
#         print("[ERROR] No feature rows collected for LLM analysis.")
#         return {}

#     # Step 3: Send the full table to the LLM for pattern analysis
#     #         Prompt the LLM to identify which features most strongly differentiate the labels.
#     prompt = (
#         "You are given a table of OpenSMILE acoustic features extracted from speech samples. "
#         "Each row corresponds to a sample, with its feature values, label (harmonic=1, dissonant=0), and filename. "
#         "Analyze the table and identify which features most strongly distinguish harmonic from dissonant samples. "
#         "Summarize your reasoning and list the most important features. "
#         "Respond with a JSON object: {\"important_features\": [...], \"reasoning\": \"...\"}\n\n"
#         "Feature Table (JSON):\n"
#         f"{json.dumps(all_feature_rows, indent=2)}"
#     )
#     # ApiNotes: Use call_llm directly for full context window support.
#     llm_response = call_llm(prompt, max_tokens=2048)
#     log_message("INFO", f"[LLM_ANALYSIS] Table analysis: {llm_response}", log_file)

#     # Step 4: Parse and summarize LLM response
#     parsed_response = None
#     if isinstance(llm_response, str):
#         try:
#             parsed_response = json.loads(llm_response)
#         except Exception as e:
#             log_message("ERROR", f"Failed to parse LLM response as JSON: {e}", log_file)
#     summary = {
#         "total_processed": processed,
#         "total_skipped": skipped,
#         "llm_response": parsed_response if parsed_response else llm_response,
#         "feature_keys": list(all_header_keys),
#         "results": all_feature_rows
#     }
#     print(f"[INFO] LLM feature scan complete. See log for details.")
#     log_message("INFO", f"[SUMMARY] LLM feature scan complete. Processed: {processed}, Skipped: {skipped}", log_file)
#     return summary

def build_prompt(entry, resonance):
    """
    ApiNotes: Constructs an LLM prompt for a given entry and resonance type.
    - entry: dict containing the original text and any metadata.
    - resonance: 'harmonic' or 'dissonant'.
    Returns a string prompt for the LLM.
    """
    # Reference: See imperative prompt engineering guidance in file-level ApiNotes.
    base_text = entry.get("text", "")
    if resonance == "harmonic":
        instructions = (
            "Generate a harmonic stylistic variation of the following text. "
            "Use repeated, smooth sounds, alliteration, assonance, and consonance. "
            "Aim for low pitch variation and a musical, flowing quality."
        )
    else:
        instructions = (
            "Generate a dissonant stylistic variation of the following text. "
            "Use clashing, abrupt, or harsh sound patterns. Avoid repetition, alliteration, or musical qualities. "
            "Aim for high pitch variation, abrupt phoneme transitions, and unpredictable rhythm."
        )
    prompt = f"{instructions}\nText: \"{base_text}\""
    return prompt

# ApiNotes: ctx_length is set from global configuration for model context window management.

ctx_length = 131072  # Fallback default, update as needed

# Global Ollama model reference and context window
ollama_model_ref = None
ollama_model_name = "hf.co/ibm-granite/granite-3.3-8b-instruct-GGUF:Q8_0"
num_ctx = 131072  # Default; will be set at load time

def load_ollama_model(model=ollama_model_name, num_ctx_val=131072):
    """
    ApiNotes: Loads the Ollama model and sets num_ctx for the session.
    This must be called at startup before any LLM calls.
    """
    global ollama_model_ref, num_ctx, ollama_model_name
    ollama_model_name = model
    num_ctx = num_ctx_val
    # No explicit model object in ollama-py, but we store config for future calls
    ollama_model_ref = {
        "model": model,
        "num_ctx": num_ctx_val
    }

def call_llm(
    prompt,
    ctx=None,
    system_prompt=None,
    temperature=0.1,
    max_tokens=131072,
    stream=False,
    log_file=None,
    response_idx=None,
    frequency_penalty=None,
    seed=None,
    top_p=None,
    top_k=None
) -> Union[str, Generator[str, None, None]]:
    """
    ApiNotes: Centralizes all Ollama LLM chat invocations.
    - prompt: User prompt string.
    - ctx: Optional context object (for conversation continuity).
    - system_prompt: Optional system prompt for LLM context.
    - temperature: Sampling temperature.
    - max_tokens: Maximum tokens to generate.
    - stream: If True, yields tokens as they arrive.
    - log_file: Optional file-like object for logging responses.
    - response_idx: Optional integer for enumerating responses in logs.
    - frequency_penalty, seed, top_p, top_k: Optional generation tuning parameters.
    Returns the generated text (or yields tokens if stream=True).
    """
    assert ollama_model_ref is not None, "ApiNotes: Ollama model must be loaded with load_ollama_model() before calling LLM"
    model = ollama_model_ref["model"]

    # Build message list for Ollama chat
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if ctx:
        # Truncate context to fit within num_ctx if needed
        ctx_window = ollama_model_ref["num_ctx"]
        total_tokens = sum(len(m.get("content", "")) for m in ctx)
        if total_tokens > ctx_window:
            truncated_ctx = []
            running_tokens = 0
            for m in reversed(ctx):
                running_tokens += len(m.get("content", ""))
                if running_tokens > ctx_window:
                    break
                truncated_ctx.insert(0, m)
            messages.extend(truncated_ctx)
        else:
            messages.extend(ctx)
    messages.append({"role": "user", "content": prompt})

    # Build options dict for runtime tuning
    options = {
        "temperature": temperature,
        "num_predict": max_tokens,
        "num_ctx": ollama_model_ref["num_ctx"]
    }
    if frequency_penalty is not None:
        options["frequency_penalty"] = frequency_penalty
    if seed is not None:
        options["seed"] = seed
    if top_p is not None:
        options["top_p"] = top_p
    if top_k is not None:
        options["top_k"] = top_k

    try:
        if stream:
            response_stream = ollama.chat(
                model=model,
                messages=messages,
                stream=True,
                options=options
            )
            for part in response_stream:
                msg = part['message']
                done = part.get('done', False)
                if not done:
                    if log_file:
                        tag = f"RESPONSE {response_idx}" if response_idx is not None else "RESPONSE"
                        log_file.write(f"[{tag}] {msg['content']}\n")
                    yield msg['content']
                else:
                    if log_file:
                        tag = f"NEXT RESPONSE {response_idx}" if response_idx is not None else "NEXT RESPONSE"
                        log_file.write(f"[{tag}] [DONE]\n")
                    break
        else:
            responses: ollama.ChatResponse = ollama.chat(
                model=model,
                messages=messages,
                stream=False,
                options=options
            )
            rv = []
            for r  in responses:
                content: str = responses['message']['content']
                if log_file:
                    tag = f"RESPONSE {response_idx}" if response_idx is not None else "RESPONSE"
                    log_file.write(f"[{tag}] {content}\n")
                rv.append(content)
            return "\n".join(rv)
    except Exception as e:
        if log_file:
            tag = f"NEXT RESPONSE {response_idx}" if response_idx is not None else "NEXT RESPONSE"
            log_file.write(f"[{tag}] LLM call failed: {e}\n")
        raise

def chat_llm(
    prompt,
    ctx=None,
    system_prompt=None,
    temperature=0.7,
    max_tokens=512,
    stream=False,
    log_file=None,
    response_idx=None,
    frequency_penalty=None,
    seed=None,
    top_p=None,
    top_k=None
):
    """
    ApiNotes: See call_llm for full documentation. This is an alias for call_llm for clarity in imperative code.
    """
    return call_llm(
        prompt=prompt,
        ctx=ctx,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
        log_file=log_file,
        response_idx=response_idx,
        frequency_penalty=frequency_penalty,
        seed=seed,
        top_p=top_p,
        top_k=top_k
    )


# ApiNotes: These stubs are imperative, modular, and justified by file-level and project-level ApiNotes.
#           When integrating, replace the NotImplementedError with actual LLM and TTS calls.
#           Update all ApiNotes and acceptance tests when interface or behavior changes.

def log_panel(log: list) -> Panel:
    """
    ApiNotes: Render log as a full-width panel, using rich styles with Style objects for maximum compatibility.
    Style markers are not shown; instead, lines are styled by keyword or marker.
    """
    colored_lines = []
    for line in log:
        # Use Style objects for explicit styling
        if "ERROR" in line or "[red]" in line:
            colored_lines.append(Text(line, style=Style(color="red")))
        elif "WARN" in line or "[yellow]" in line:
            colored_lines.append(Text(line, style=Style(color="yellow")))
        elif "SUCCESS" in line or "[green]" in line:
            colored_lines.append(Text(line, style=Style(color="green")))
        elif "INFO" in line or "[cyan]" in line:
            colored_lines.append(Text(line, style=Style(color="cyan", italic=True)))
        elif "blue" in line or "[blue]" in line:
            colored_lines.append(Text(line, style=Style(color="blue", italic=True)))
        else:
            colored_lines.append(Text(line))
    # Join with newlines and display as a single Text object
    log_text = Text.assemble(*[l.append("\n") or l for l in colored_lines])
    return Panel(
        log_text,
        title=Text("Log", style=Style(color="cyan", italic=True)),
        border_style="green",
        width=console.width
    )

# ApiNotes: This approach uses Style objects for all log coloring, avoids literal markup display,
#           and demonstrates use of italic and color per user instruction.
#           Adjust or expand keyword detection as needed for your log format.

# -----------------------------------------------------------------------------
# 📦  Setup – download CMUdict if necessary
# -----------------------------------------------------------------------------

def _ensure_cmudict():
    try:
        nltk.data.find("corpora/cmudict.zip")
    except LookupError:
        nltk.download("cmudict", quiet=True)

_ensure_cmudict()
cmu_dict: Dict[str, List[List[str]]] = cmudict.dict()  # type: ignore
phoneme_set: List[str] = sorted({p for v in cmu_dict.values() for p in v[0]})


# -----------------------------------------------------------------------------
# 🗃️  Carrier (optional hyphenation) – unchanged
# -----------------------------------------------------------------------------

def create_hyphenation_carrier():
    pairs = {
        'state of the art': 'state-of-the-art',
        'long term': 'long-term',
        'short term': 'short-term',
        'well being': 'well-being',
        'cutting edge': 'cutting-edge',
        'user friendly': 'user-friendly',
        'cost effective': 'cost-effective',
    }
    rx = re.compile(r"\b(" + "|".join(map(re.escape, pairs.keys() | pairs.values())) + r")\b", re.I)
    rev = {v: k for k, v in pairs.items()}

    def find_sites(t: str):
        return [(m.start(), m.group()) for m in rx.finditer(t)]

    def estimate(t: str):
        return len(find_sites(t))

    def apply(t: str, bits: List[bool]):
        if not bits:
            return t, 0
        chars = list(t)
        used = 0
        for (idx, word), bit in zip(reversed(find_sites(t)), bits[::-1]):
            target = pairs.get(word.lower(), rev.get(word.lower(), word))
            want_hyph = '-' in target
            have_hyph = '-' in word
            if want_hyph == have_hyph:
                used += 1
                continue
            repl = target if bit else rev.get(word.lower(), word.replace('-', ' '))
            chars[idx: idx + len(word)] = list(repl)
            used += 1
        return ''.join(chars), used

    return {'estimate': estimate, 'apply': apply}

# -----------------------------------------------------------------------------
# 💬  LLM helpers
# -----------------------------------------------------------------------------

def generate_with_feedback(prompt, resonance, max_attempts, opensmile_analyze_fn, quality_thresholds, log_file=None, temperature=0.7, top_p=0.95, top_k=40):
    """
    ApiNotes: Attempts LLM generation up to max_attempts, using OpenSMILE analysis to grade output.
    If generation is low-quality, provides feedback and retries.
    Returns the best generation and its analysis.
    """
    analysis = 0
    candidate = call_llm(prompt, temperature=temperature, top_p=top_p, top_k=top_k)
    #wav_path = synthesize_audio(candidate)
    results = []
    for attempt in range(1, max_attempts + 1):
        # Generate candidate
        gen_candidate = call_llm(prompt, temperature=temperature, top_p=top_p, top_k=top_k)
        # Synthesize audio and analyze with OpenSMILE
        wav_path = synthesize_audio(candidate)
        features = extract_opensmile_features(wav_path, config_path="conf/opensmile/emo_large.conf", opensmile_bin="conf/SMILExtract")
        results.append({
            "text": gen_candidate,
            "wav_path": wav_path,
            "features": features,
            })
        # Grade quality
        f0_std_value = float(features.get("F0smaStdd", 0))
        if resonance == "dissonant":
            is_good = f0_std_value >= quality_thresholds["dissonant"]
        else:
            is_good = f0_std_value >= quality_thresholds["harmonic"]
        if is_good:
            if log_file:
                log_file.write(f"[INFO] Generation accepted on attempt {attempt}: {candidate}\n")
            return candidate, analysis
        # Feedback for retry
        feedback = (
            f"Your previous {resonance} output did not meet quality standards "
            f"(F0 stdev={f0_std_value:.2f}, expected >= {quality_thresholds[resonance]}). "
            "Please try again, making the output more "
            + ("abrupt and unpredictable." if resonance == "dissonant" else "smooth and harmonious.")
        )
        prompt = f"{prompt}\n\nFEEDBACK: {feedback}"
        if log_file:
            log_file.write(f"[WARN] Generation rejected on attempt {attempt}: {candidate}\n")
    # If all attempts fail, return last candidate
    if log_file:
        log_file.write(f"[ERROR] No high-quality {resonance} generation after {max_attempts} attempts.\n")
    return candidate, analysis


def generate_variations_ollama(
    base_text,
    n_variations=3,
    ctx=8192,
    system_prompt=None,
    temperature=0.8,
    max_tokens=8192,
    log_file=None,
    frequency_penalty=None,
    seed=None,
    top_p=None,
    top_k=None
):
    """
    ApiNotes: Generates stylistic variations of base_text using the LLM via chat_llm.
    Returns a list of generated variations.
    """
    # Reference file-level ApiNotes for imperative design and logging requirements.
    variations = []
    for i in range(n_variations):
        prompt = (
            f"Generate a stylistic variation of the following text. "
            f"Be creative, but preserve the core meaning. "
            f"Text: \"{base_text}\""
        )
        try:
            variation = chat_llm(
                prompt=prompt,
                ctx=ctx,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                log_file=log_file,
                response_idx=i + 1,
                frequency_penalty=frequency_penalty,
                seed=seed,
                top_p=top_p,
                top_k=top_k
            )
            assert variation is not None and isinstance(variation, str), \
                "ApiNotes: LLM must return a non-empty string for each variation"
            variations.append(variation.strip())
        except Exception as e:
            # ApiNotes: Log and continue on failure, but enforce at least one successful variation.
            if log_file:
                log_file.write(f"[ERROR] Variation {i+1} failed: {e}\n")
            continue
    assert len(variations) > 0, "ApiNotes: At least one variation must be generated"
    return variations


def enhance_prompt_with_analysis(base_prompt: str, resonance: str, analysis: Dict[str, Any], 
                                acoustic_examples: Optional[Dict[str, Any]] = None,
                                harmonic_text: Optional[str] = None,
                                dissonant_text: Optional[str] = None) -> str:
    """
    Enhance prompts based on performance analysis and, if needed, inject openSMILE-based acoustic examples.
    ApiNotes: This function is imperative and may call stateless helpers for formatting.
    """
    if not analysis.get("available", False):
        return base_prompt
    
    recommendations = analysis.get("recommendations", [])
    
    enhanced_prompt = base_prompt
    
    # Add specific guidance based on analysis
    if resonance == "harmonic":
        if "strengthen_harmonic" in recommendations:
            enhanced_prompt += "\n\nNOTE: Focus on creating STRONGER phonetic patterns. Choose words with similar sounds, repeated consonants or vowels, and consistent rhythm. Emphasize alliteration, assonance, and consonance."
        
        if "focus_harmonic_clarity" in recommendations:
            enhanced_prompt += "\n\nIMPORTANT: Your previous harmonic outputs have been inconsistent. Focus on creating clear, recognizable sound patterns that enhance the text's flow and musicality."
    
    elif resonance == "dissonant":
        if "strengthen_dissonant" in recommendations:
            enhanced_prompt += "\n\nNOTE: Focus on creating STRONGER phonetic disruption. Mix contrasting sounds, break rhythm patterns, and use words with clashing phonemes. Avoid alliteration or repeated sound patterns."
        
        if "focus_dissonant_distinctness" in recommendations:
            enhanced_prompt += "\n\nIMPORTANT: Your previous dissonant outputs have been too similar to harmonic ones. Create more jarring phonetic contrasts by mixing harsh and soft sounds in unexpected ways."
    
    if "increase_contrast" in recommendations:
        enhanced_prompt += "\n\nCRITICAL: The distinction between harmonic and dissonant outputs has been insufficient. Create a CLEAR phonetic difference between the two styles."
        # ApiNotes: If acoustic_examples are provided, inject them into the prompt for LLM guidance.
        if acoustic_examples and harmonic_text and dissonant_text:
            prompt_block = format_acoustic_examples_for_prompt(
                harmonic_text=harmonic_text,
                dissonant_text=dissonant_text,
                harmonic_feats=acoustic_examples["harmonic"],
                dissonant_feats=acoustic_examples["dissonant"],
                feature_keys=acoustic_examples["feature_keys"]
            )
            enhanced_prompt += "\n\n" + prompt_block
    
    # Log the enhancement
    print(f"[INFO] Prompt for {resonance} enhanced based on {len(recommendations)} recommendations")
    
    return enhanced_prompt


def generate_pair_ollama(
    base_text,
    pair_type="harmonic",
    ctx=None,
    system_prompt=None,
    temperature=0.7,
    max_tokens=512,
    log_file=None,
    frequency_penalty=None,
    seed=None,
    top_p=None,
    top_k=None
):
    """
    ApiNotes: Generates a stylistic pair (harmonic or dissonant) for base_text using the LLM via chat_llm.
    Returns the generated pair as a string.
    """
    # Reference file-level ApiNotes for imperative design and logging requirements.
    if pair_type == "harmonic":
        prompt = (
            f"Generate a harmonic stylistic variation of the following text. "
            f"Use repeated, smooth sounds, alliteration, assonance, and consonance. "
            f"Aim for low pitch variation and a musical, flowing quality.\n"
            f"Text: \"{base_text}\""
        )
    else:
        prompt = (
            f"Generate a dissonant stylistic variation of the following text. "
            f"Use clashing, abrupt, or harsh sound patterns. Avoid repetition, alliteration, or musical qualities. "
            f"Aim for high pitch variation, abrupt phoneme transitions, and unpredictable rhythm.\n"
            f"Text: \"{base_text}\""
        )
    try:
        pair = chat_llm(
            prompt=prompt,
            ctx=ctx,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            log_file=log_file,
            response_idx=1,
            frequency_penalty=frequency_penalty,
            seed=seed,
            top_p=top_p,
            top_k=top_k
        )
        assert pair is not None and isinstance(pair, str), \
            "ApiNotes: LLM must return a non-empty string for the generated pair"
        return pair.strip()
    except Exception as e:
        # ApiNotes: Log and propagate error for debugging and acceptance test coverage.
        if log_file:
            log_file.write(f"[ERROR] Pair generation failed: {e}\n")
        raise


dissonant_base_str = """dissonant, Generate a dissonant stylistic variation of the following text. 
- Use clashing, abrupt, or harsh sound patterns.
- Avoid repetition, alliteration, assonance, or any musical/flowing qualities.
- Use words with abrupt consonant clusters, irregular rhythm, and unpredictable phoneme transitions.
- Aim for high pitch variation (F0 stdev), low similarity to the original, and maximize abruptness in sound transitions.
- Example of dissonant output: "Cracked glass shrieks, abrupt and harsh."
- Do NOT use repeated sounds, rhyme, or smooth transitions.
Text: "{original_text}""}).

Score on a scale of 1-10:
1. How well does this variation match its intended phonetic goal ({expected_type})?
2. How different is it phonetically from the original?

Provide scores only, separated by commas."""

def evaluate_pair_quality(
    original_text,
    generated_text,
    opensmile_analyze_fn,
    similarity_fn,
    quality_thresholds,
    log_file=None
):
    """
    ApiNotes: Evaluates the quality of a generated stylistic pair using OpenSMILE and similarity metrics.
    - original_text: The original input string.
    - generated_text: The generated stylistic variation.
    - opensmile_analyze_fn: Callable that takes a WAV path and returns OpenSMILE metrics.
    - similarity_fn: Callable that takes (original_text, generated_text) and returns a similarity score.
    - quality_thresholds: Dict with keys for relevant OpenSMILE metrics and similarity (e.g., {'F0smaStdd': 10, 'similarity': 0.7}).
    - log_file: Optional file-like object for logging.
    Returns a dict with keys: {'opensmile': ..., 'similarity': ..., 'quality': bool, 'reasons': [...]}
    """
    # Synthesize audio for generated_text (assume synthesize_audio returns wav_path)
    wav_path = synthesize_audio(generated_text)
    opensmile_metrics = opensmile_analyze_fn(wav_path)
    similarity = similarity_fn(original_text, generated_text)

    # Evaluate against thresholds
    reasons = []
    quality = True

    # Check OpenSMILE metrics
    for metric, threshold in quality_thresholds.items():
        if metric == "similarity":
            continue
        value = opensmile_metrics.get(metric, None)
        assert value is not None, f"ApiNotes: OpenSMILE metric '{metric}' missing from analysis"
        if value < threshold:
            quality = False
            reasons.append(f"{metric}={value} < threshold {threshold}")

    # Check similarity (for dissonant, want low similarity; for harmonic, want high)
    sim_threshold = quality_thresholds.get("similarity", None)
    if sim_threshold is not None:
        if similarity < sim_threshold:
            quality = False
            reasons.append(f"similarity={similarity} < threshold {sim_threshold}")

    # Logging
    if log_file:
        log_file.write(f"[EVAL] OpenSMILE: {opensmile_metrics}, Similarity: {similarity}, Quality: {quality}, Reasons: {reasons}\n")

    return {
        "opensmile": opensmile_metrics,
        "similarity": similarity,
        "quality": quality,
        "reasons": reasons
    }
# -----------------------------------------------------------------------------
# 🖥️  Rich UI helpers
# -----------------------------------------------------------------------------

def get_stat_style(key, value):
    """
    ApiNotes: Returns a style for a stat value based on warning/critical thresholds.
    Adjust thresholds as needed for your domain.
    """
    # Example thresholds (customize as needed)
    warning_style = "yellow"
    critical_style = "red on yellow"
    ok_style = "green"

    # Define thresholds for specific metrics
    if key == "Prediction Accuracy":
        try:
            percent = float(str(value).replace("%", "0.00"))
            if percent < 60:
                return critical_style
            elif percent < 80:
                return warning_style
            else:
                return ok_style
        except Exception:
            return ""
    if key == "Low Harmonic Quality" or key == "Low Dissonant Quality":
        if int(value) > 2:
            return warning_style
    if key == "Low Differentiation Count":
        if int(value) > 1:
            return critical_style
    # Add more rules as needed
    return ""

def make_summary_table(stats: dict) -> Panel:
    """
    ApiNotes: Returns a Panel containing a table of all relevant session statistics.
    All metrics tracked in stats are included, with color-coded values for warnings/critical.
    """
    table = Table(show_header=True, header_style="magenta")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    for k, v in stats.items():
        style = get_stat_style(k, v)
        if style:
            table.add_row(str(k), Text(str(v), style=style))
        else:
            table.add_row(str(k), str(v))
    return Panel(table, title="[yellow]Session Statistics[/]", border_style="yellow")

def make_progress_panel(current_idx: int, total: int, entry_id: str, stats: dict) -> Panel:
    """
    ApiNotes: Returns a Panel showing progress and current entry, updated per-entry.
    """
    table = Table(show_header=False, box=None)
    table.add_row("Entry", f"{current_idx+1} / {total}")
    table.add_row("Entry ID", entry_id)
    table.add_row("Correct", str(stats.get("Correct", 0)))
    table.add_row("Incorrect", str(stats.get("Incorrect", 0)))
    table.add_row("Accuracy", str(stats.get("Prediction Accuracy", "0.00")))
    return Panel(table, title="[green]Progress[/]", border_style="green")

def make_table() -> Table:
    t = Table(
        show_lines=True, 
        box=box.ROUNDED, 
        title="Phonetic Similarity Analysis",
        title_style="cyan",
        caption="H=Harmonic, D=Dissonant, Pred=Predicted, Exp=Expected",
        caption_style="dim"
    )
    t.add_column("#", justify='right', style="cyan", no_wrap=True)
    t.add_column("Sim", justify='center', style="magenta", no_wrap=True)
    t.add_column("Exp", justify='center', style="green", no_wrap=True)
    t.add_column("Pred", justify='center', no_wrap=True)
    t.add_column("Entry", justify='left', style="yellow", no_wrap=True)
    t.add_column("Type", justify='center', style="blue", no_wrap=True)
    t.add_column("Status", justify='center', no_wrap=True)
    
    # Add a placeholder row to prevent empty table rendering errors
    t.add_row("0", "0.000", "?", "?", "Initializing...", "?", "?")
    
    return t

def make_hist(sims: List[float], labs: List[int], bins: int = 10):
    if not sims or len(sims) < 2:
        return Panel("Not enough data for histogram", title="[cyan]Similarity Distribution[/]", border_style="green")
    
    try:
        s0 = [s for s, l in zip(sims, labs) if l == 0]
        s1 = [s for s, l in zip(sims, labs) if l == 1]
        h0, edges = np.histogram(s0, bins=bins, range=(0, 1))
        h1, _ = np.histogram(s1, bins=bins, range=(0, 1))
        
        if len(edges) <= 1:
            return Panel("Cannot create histogram bins", title="[cyan]Similarity Distribution[/]", border_style="green")
            
        width = edges[1] - edges[0]
        lines = []
        
        # Calculate stats for subtitle
        h_mean = np.mean(s1) if s1 else 0
        d_mean = np.mean(s0) if s0 else 0
        separation = h_mean - d_mean
        
        for h0_count, h1_count, edge in zip(h0, h1, edges[:-1]):
            bin_label = f"{edge:.2f}-{edge + width:.2f}: "
            # Use block characters with different colors
            h0_bar = "[blue]" + "█" * int(h0_count) + "[/]"
            h1_bar = "[magenta]" + "█" * int(h1_count) + "[/]"
            # Add counts
            counts = f"  ([blue]{int(h0_count)}D[/]/[magenta]{int(h1_count)}H[/])"
            lines.append(bin_label + h0_bar + h1_bar + counts)
        
        stats_text = f"\nStats: H mean={h_mean:.3f} | D mean={d_mean:.3f} | Separation={separation:.3f}"
        histogram_content = "\n".join(lines) + stats_text
        
        return Panel(
            histogram_content, 
            title="[cyan]Similarity Distribution[/]",
            subtitle="[blue]Dissonant[/] vs [magenta]Harmonic[/]",
            border_style="green"
        )
    except Exception as e:
        return Panel(f"Error creating histogram: {e}", title="[cyan]Similarity Distribution[/]", border_style="red")

# -----------------------------------------------------------------------------
# 📚  Corpus loader
# -----------------------------------------------------------------------------

def load_corpus(paths: List[str]):
    corpus: List[Dict[str, Any]] = []
    for p in paths:
        try:
            with open(p) as f:
                corpus.extend(json.load(f).get('corpus', []))
        except Exception as e:
            print(f'[ERROR] Failed to load corpus file {p}: {e}', file=sys.stderr)
    return corpus


def load_pretrain_data(path: str, eightword_mode: bool, log_file: Optional[TextIO] = None) -> Tuple[List[List[float]], List[int]]:
    """Load pretrained data from a JSON file and randomize it."""
    if not os.path.isfile(path):
        print(f"[WARN] Pretrain data file not found: {path}", file=sys.stderr)
        return [], []
    
    features = []
    labels = []
    processed = 0
    skipped = 0
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if log_file:
            log_file.write(f"\n=== LOADING PRETRAIN DATA ===\n")
            log_file.write(f"SOURCE: {path}\n")
            log_file.write(f"ENTRIES: {len(data)}\n")
        
        # Randomize the data order
        random.shuffle(data)
        
        for entry in data:
            original_text = entry.get("original", [""])[0]
            paraphrased_text = entry.get("paraphrased", [""])[0]
            label = entry.get("label")
            
            # Skip invalid entries
            if not original_text or not paraphrased_text or label is None:
                skipped += 1
                continue
            
            try:
                # Calculate phonetic similarity
                original_phonemes = text_to_phonemes(original_text)
                paraphrased_phonemes = text_to_phonemes(paraphrased_text)
                
                if not original_phonemes or not paraphrased_phonemes:
                    if log_file:
                        log_file.write(f"SKIPPED: No phonemes extracted for entry\n")
                    skipped += 1
                    continue
                
                original_vec = phoneme_vec(original_phonemes)
                paraphrased_vec = phoneme_vec(paraphrased_phonemes)
                
                sim_cos = cosine(original_vec, paraphrased_vec)
                final_sim = sim_cos
                
                # Apply eightword score if enabled
                if eightword_mode:
                    ew_orig = eightword_harmonic_score(original_text)
                    ew_para = eightword_harmonic_score(paraphrased_text)
                    ew_sim = 1.0 - abs(ew_orig - ew_para)
                    final_sim = (sim_cos + ew_sim) / 2.0
                
                features.append([final_sim])
                labels.append(label)
                processed += 1
                
                if log_file:
                    log_file.write(f"PROCESSED: Entry with similarity {final_sim:.3f}, label {label}\n")
                    
            except Exception as e:
                if log_file:
                    log_file.write(f"ERROR: Processing entry: {e}\n")
                skipped += 1
        
        print(f"[INFO] Loaded {processed} pretrain entries, skipped {skipped}")
        if log_file:
            log_file.write(f"SUMMARY: Processed {processed}, Skipped {skipped}\n")
            
    except Exception as e:
        error_msg = f"[ERROR] Failed to process pretrain data {path}: {e}"
        print(error_msg, file=sys.stderr)
        if log_file:
            log_file.write(f"{error_msg}\n")
    
    return features, labels

def analyze_csv_with_llm(csv_path, original_text, label=None):
    features = read_opensmile_csv(csv_path)
    llm_response = prompt_llm_for_feature_patterns(original_text, features, label)
    print("LLM Analysis:", llm_response)
    return llm_response


# -----------------------------------------------------------------------------
# 🧵 Thread-safe UI Update Queue
# -----------------------------------------------------------------------------

class UIUpdateQueue:
    """
    ApiNotes: Thread-safe queue for UI update callables.
    Used by update_thread to batch and apply UI changes in the TUI.
    """

    def __init__(self):
        self._queue = []
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._stop = False

    def add_update(self, fn):
        with self._lock:
            self._queue.append(fn)
            self._event.set()

    def get_updates(self):
        with self._lock:
            updates = self._queue[:]
            self._queue.clear()
            self._event.clear()
        return updates

    def wait_for_update(self, timeout=0.1):
        self._event.wait(timeout)

    def stop(self):
        self._stop = True
        self._event.set()

    def should_stop(self):
        return self._stop

    def process_updates(self, live):
        """Process all pending updates in the queue"""
        updates_to_process = []
        
        # Get all updates under lock
        with self._lock:
            updates_to_process = self._queue.copy()
            self._queue = []
        
        # Process updates
        if updates_to_process:
            try:
                for update_func in updates_to_process:
                    update_func()
                
                # Only refresh once after all updates
                live.refresh()
            except Exception as e:
                print(f"[ERROR] UI update error: {e}", file=sys.stderr)

    
def update_thread(ui_queue, live):
        """
        ApiNotes: Background thread that applies UI updates from the UIUpdateQueue.
        Runs until ui_queue.should_stop() is True.
        """
        while not ui_queue.should_stop():
            ui_queue.wait_for_update(timeout=0.1)
            updates = ui_queue.get_updates()
            for fn in updates:
                try:
                    fn()
                except Exception as e:
                    # ApiNotes: Log but do not crash on UI update error
                    print(f"[UIUpdateThread] Error in UI update: {e}")
            # Sleep briefly to avoid busy-waiting
            time.sleep(0.05)

#------------------------------------------
# 🚀  Main
# -----------------------------------------------------------------------------
def make_livestats_panel(live_stats: dict) -> Panel:
        table = Table(show_header=True, header_style="blue")
        table.add_column("Live Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")
        for k, v in live_stats.items():
            table.add_row(str(k), str(v))
        return Panel(table, title="[blue]Live Data Stats[/]", border_style="blue")

def make_historical_panel(historical_stats: dict) -> Panel:
    table = Table(show_header=True, header_style="magenta")
    table.add_column("Historical Metric", style="magenta", no_wrap=True)
    table.add_column("Value", style="white")
    for k, v in historical_stats.items():
        table.add_row(str(k), str(v))
    return Panel(table, title="[magenta]Historical Data Stats[/]", border_style="magenta")

def collect_livetime_statistics(data_file: str, log_file=None) -> dict:
    """
    ApiNotes: Collects statistics from the current data-file set since last --data-reset.
    Uses analyze_past_performance from dataset_helpers to summarize the dataset.
    Returns a dictionary of live statistics for display in the TUI.
    """
    # For now, this is just a direct call to analyze_past_performance.
    # If you want to filter for only "since last reset", you must track reset time in metadata.
    stats = analyze_past_performance(data_file, log_file=log_file)
    return stats

def collect_historical_statistics(data_file: str, log_file=None) -> dict:
    """
    ApiNotes: Collects historical statistics from the data-file set.
    For now, this is identical to collect_livetime_statistics, but can be extended to aggregate across resets.
    """
    # In a future version, you could load multiple data files or use metadata to distinguish historical from live.
    stats = analyze_past_performance(data_file, log_file=log_file)
    return stats

def calculate_llm_params(aggression, base_temp=0.7, base_top_p=0.95, base_top_k=40):
    """
    ApiNotes: Calculates LLM generation parameters based on aggression level.
    - aggression: Float from 0.0 to 1.0 controlling overall creativity
    - base_temp/top_p/top_k: Default values to use if no CLI overrides provided
    
    Low aggression (0.0) = deterministic, focused outputs
    High aggression (1.0) = creative, diverse outputs
    
    Returns tuple of (temperature, top_p, top_k)
    """
    # Scale temperature: low aggression = lower temp, high aggression = higher temp
    temperature = base_temp * (0.5 + aggression)  # Range from 0.5*base to 1.5*base
    
    # Calculate top_p: low aggression = lower top_p (more focused), high aggression = higher (more diverse)
    top_p = min(0.99, base_top_p * (0.8 + 0.4 * aggression))  # Range from 0.8*base to 1.2*base, max 0.99
    
    # Calculate top_k: low aggression = lower top_k (fewer options), high aggression = higher (more options)
    top_k = max(1, int(base_top_k * (0.6 + 0.8 * aggression)))  # Range from 0.6*base to 1.4*base, min 1
    
    return temperature, top_p, top_k

def _parse_opensmile_csv_for_mp(args):
    """
    ApiNotes: Helper function for multiprocessing that parses a single OpenSMILE CSV file.
    - args: Tuple of (csv_path, label_map, feature_keys) 
    Returns: Tuple of (filename, features_dict, label) or None if parsing fails
    
    This function is designed to be called by concurrent.futures.ProcessPoolExecutor
    for parallel CSV processing, improving performance on multi-core systems.
    """
    csv_path, label_map, feature_keys = args
    fname = os.path.basename(csv_path)
    
    # Determine label from filename marker
    # label = None
    # for marker, lab in label_map.items():
#             if marker in fname:
#                 label = lab
#                 break
    
#     if label is None:
#         return None  # Skip if no label can be determined
    
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Parse ARFF header and find @data section
        data_start = None
        header_keys = []
        for idx, line in enumerate(lines):
            if line.strip().lower() == "@data":
                data_start = idx + 2
                break
            if line.startswith("@attribute"):
                parts = line.strip().split()
                if len(parts) >= 3:
                    header_keys.append(parts[1])
        
        if data_start is None or data_start >= len(lines):
            return None  # Invalid CSV format
        
        # Parse data values
        data_line = lines[data_start].strip()
        data_values = [v.strip() for v in data_line.split(",")]
        
        # Extract feature values
        feature_dict = {}
        use_keys = feature_keys if feature_keys else header_keys
        key_to_idx = {k: i for i, k in enumerate(header_keys)}
        
        for k in use_keys:
            idx = key_to_idx.get(k)
            if idx is not None and idx < len(data_values):
                try:
                    feature_dict[k] = float(data_values[idx])
                except ValueError:
                    feature_dict[k] = float('nan')
            else:
                feature_dict[k] = None
        
        return (fname, feature_dict, "")
    
    except Exception:
        return None  # Return None on any parsing error




def scan_csvs_for_phonetic_patterns(
    audio_dir: str,
    label_map: dict,
    log_file: str = "",
    max_files: int = 200,
    feature_keys: list = [],
    num_threads: int = 4
    ):
    """
    ApiNotes: Scans all OpenSMILE CSVs in audio_dir, analyzing features in parallel across multiple threads.
    For each feature, extracts its value from all files, then analyzes that single feature with the LLM.
    This vertical analysis approach provides focused insights on how each acoustic feature 
    varies across different speech samples.
    
    Performance optimization: 
    1. Processes features in parallel using ThreadPoolExecutor (num_threads)
    2. Distributes LLM inference across multiple GPUs (if available in Ollama setup)
    3. Creates a focused column-based analysis examining patterns in each individual acoustic measure
    """
    # Reference: file-level ApiNotes.md for imperative, observable, and testable design.
    from rich.markdown import Markdown
    from rich.progress import track
    import json
    import os
    import threading
    import concurrent.futures
    from logging_utils import log_message
    
    # Initialize the LLM model with large context window
    load_ollama_model("hf.co/ibm-granite/granite-3.3-8b-instruct-GGUF:Q8_0", num_ctx_val=131072)
    
    # Gather all CSV files to process
    csv_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) 
                if f.endswith(".wav.smile.csv")]
    
    if max_files:
        csv_files = csv_files[:max_files]
    
    log_message("INFO", f"[START] Processing {len(csv_files)} CSV files using {num_threads} parallel threads", log_file)
    
    
    # First, determine all available feature keys from the first file
    all_feature_keys = []
    label_dict = {}
    
    # Parse one file to get header keys if feature_keys not provided
    if not feature_keys and csv_files:
        try:
            with open(csv_files[0], "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            # Find header keys in the first file
            for line in lines:
                if line.strip() == "":
                    continue
                if line.strip().lower() == "@data":
                    break
                if line.startswith("@attribute"):
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        all_feature_keys.append(parts[1])
            
            if not all_feature_keys:
                log_message("ERROR", f"[ERROR] Could not extract feature keys from {csv_files[0]}", log_file)
                # Instead of returning early, use a reasonable default
                log_message("WARN", f"[WARN] Using default feature key 'F0' to continue processing", log_file)
                all_feature_keys = ["F0"]  # Default to a common acoustic feature as fallback
            else:
                # Skip first and last feature (typically name or class features) as requested
                if len(all_feature_keys) > 2:
                    all_feature_keys = all_feature_keys[1:-1]
                    log_message("INFO", f"[INFO] Skipping first and last features, analyzing {len(all_feature_keys)} features", log_file)
                else:
                    log_message("WARN", f"[WARN] Not enough features to skip first and last, using all available features", log_file)
            
        except Exception as e:
            log_message("ERROR", f"[ERROR] Failed to analyze first file: {e}", log_file)
            # Instead of returning early, use default feature keys
            log_message("WARN", f"[WARN] Using default feature keys to continue processing", log_file)
            all_feature_keys = ["F0", "jitter", "shimmer"]  # Default to common acoustic features
    else:
        # Use provided feature keys
        all_feature_keys = feature_keys
    
    # Function to process a single feature across all files and analyze with LLM
    def process_feature(feature_name, feature_idx, all_feature_keys_len):
        """Process a single feature across all files and analyze with LLM"""
        # Collect this feature's values across all files
        feature_values = {}
        skipped_local = 0
        data_values = []
        for csv_path in csv_files:
            fname = os.path.basename(csv_path)
            try:
                # Extract just this one feature from the file
                with open(csv_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                
                data_start = None
                header_keys = []
                for idx, line in enumerate(lines):
                    if line.strip().lower() == "@data":
                        data_start = idx + 2
                        break
                    if line.startswith("@attribute"):
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            header_keys.append(parts[1])
                if data_start is None or data_start >= len(lines):
                    # Log the issue and continue to next file
                    log_message("WARN", f"[WARN] Invalid CSV format in {fname}, skipping", log_file)
                    skipped_local += 1
                    continue
                
                # Get index of our feature in this file's header
                try:
                    feature_pos = header_keys.index(feature_name)
                except ValueError:
                    # Feature not found in this file, continue to next
                    continue
                
                # Extract just this one value
                data_line = lines[data_start].strip()
                data_values = [v.strip() for v in data_line.split(",")][1:-1]
                
                #if feature_pos < len(data_values):
                #    try:
                #        feature_values[fname] = float(data_values[feature_pos])
                #    except ValueError:
                #        feature_values[fname] = float('nan')
                #else:
                #    feature_values[fname] = None
                
            except Exception as e:
                skipped_local += 1
        
            # If no values were collected, return None
            #if not feature_values:
            #    log_message("WARN", f"[WARN] No values found for feature {feature_name}", log_file)
            #    return feature_name, None, skipped_local
            
            # Complete vertical analysis of this feature using LLM
            prompt = (
                f"You are given the values of a SINGLE OpenSMILE acoustic feature '{feature_name}' "
                f"across multiple speech samples. Each sample is identified by filename. "
                f"You are also given a label map indicating whether each file is harmonic (1) or dissonant (0).\n"
                f"Analyze the values for '{feature_name}' and reply with:\n"
                "- The variance of the feature across all files\n"
                "- The direction of change (does it tend to be higher for harmonic or dissonant?)\n"
                "- Any additional qualifiers or patterns you observe (e.g., outliers, bimodality, etc)\n"
                "Respond with a JSON object of the form:\n"
                "{\n"
                "  \"variance\": float,\n"
                "  \"direction\": \"higher for harmonic\" | \"higher for dissonant\" | \"no clear direction\",\n"
                "  \"qualifiers\": \"...\"\n"
                "}\n\n"
                f"Feature Values for '{feature_name}' (JSON):\n{data_values}\n"
                #f"Label Map (JSON):\n{json.dumps(label_dict, indent=2)}"
            )
            
            # Progress tracking (thread-safe print)
            print(f"[INFO] Thread analyzing feature {feature_idx+1}/{all_feature_keys_len}: {feature_name}")
            
            # Call LLM and collect full response
            try:
                parts = list(call_llm(prompt, max_tokens=131072, stream=False))
                response = "".join(parts)
                log_message("INFO", f"[LLM] {feature_name}: {response}", log_file)
                return [(feature_name, response, skipped_local)]
            except Exception as e:
                log_message("ERROR", f"LLM failed for {feature_name}: {e}", log_file)
                return [(feature_name, f"ERROR: {e}", skipped_local)]
            
    # Process features in parallel using ThreadPoolExecutor
    processed = 0
    skipped = 0
    feature_results = []
    
    print(f"[INFO] Starting parallel analysis with {num_threads} threads for {len(all_feature_keys)} features")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all feature processing tasks to the executor
        future_to_feature = {
            executor.submit(
                process_feature, 
                feature_name, 
                feature_idx, 
                len(all_feature_keys)
            ): feature_name 
            for feature_idx, feature_name in enumerate(all_feature_keys)
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_feature):
            feature_name = future_to_feature[future]
            try:
                if future is not None:
                    results = future.result()
                    feature_results.append(results)
                    processed += 1
            except Exception as e:
                skipped += 1

    # Create a summary dictionary with all the results and statistics
    summary = {
        "total_processed": processed,
        "total_skipped": skipped,
        "feature_keys": all_feature_keys,
        "results": feature_results
    }
    print(f"[INFO] Parallel feature scan complete. Processed {processed} features across {len(csv_files)} files.")
    if log_file:
        log_message("INFO", f"[SUMMARY] Parallel feature scan complete. Processed: {processed} features using {num_threads} threads", log_file)
    # Print accessible summary information
    print(f"Analyzed {len(feature_results)} features in parallel")
    print(f"Processed a total of {len(csv_files)} CSV files")
    return summary

def main():
    # Set random seeds for reproducibility but still random behavior
    random_seed = int(datetime.datetime.now().timestamp())
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    parser = argparse.ArgumentParser(description="Stylometric Phonetic Encoder (Ollama)")
    parser.add_argument('--llm-bootstrap', action='store_true', help='Run LLM feature bootstrap training loop')
    parser.add_argument('--llm-model', type=str, default="phi4", help='Ollama model name')
    parser.add_argument('--llm-temperature', type=float, default=0.3, help='LLM temperature')
    parser.add_argument('--llm-max-tokens', type=int, default=512, help='LLM max tokens')
    parser.add_argument('--llm-num-ctx', type=int, default=8192, help='LLM context window')
    parser.add_argument('--llm-frequency-penalty', type=float, default=1.2, help='LLM frequency penalty')
    parser.add_argument('--llm-seed', type=int, default=None, help='LLM random seed')
    parser.add_argument('--llm-top-p', type=float, default=None, help='LLM top_p')
    parser.add_argument('--llm-top-k', type=int, default=None, help='LLM top_k')
    parser.add_argument('--llm-log-file', type=str, default='./bootstrap.log', help='Log file for LLM bootstrap')
    parser.add_argument('--model', required=True)
    parser.add_argument('--n', type=int, default=2)
    parser.add_argument('--corpus-limit', type=int, default=5)
    parser.add_argument('--data-sets', nargs='+', default=['data/set2.json', 'data/set1.json'])
    parser.add_argument('--pretrain-data')
    parser.add_argument('--mode', choices=['both', 'train', 'test'], default='both')
    parser.add_argument('--split', type=float, default=0.8)
    parser.add_argument('--aggression', type=float, default=0.5)
    parser.add_argument('--visible-rows', type=int, default=25)
    parser.add_argument('--eightword-mode', action='store_true')
    parser.add_argument('--use-hyphenation-carrier', action='store_true')
    parser.add_argument('--data-file', default='./phonetic_featset.npz')
    parser.add_argument('--reset-data', action='store_true')
    parser.add_argument('--log-file', default='./phonetic_analysis.log')
    parser.add_argument('--refresh-rate', type=float, default=0.5, 
                   help="UI refresh rate in seconds (higher values are more stable)")
    parser.add_argument('--ctx-length', type=int, 
                   help="Context length for Ollama model (e.g., 8192, 16384, 32768)")
    parser.add_argument('--tts-cmd',   help="Shell command template for TTS, use {text} and {out} placeholders")
    parser.add_argument('--opensmile-path',  default='SMILExtract', help="Path to openSMILE SMILExtract binary")
    parser.add_argument('--opensmile-config', default='eGeMAPS.conf', help="openSMILE config file")
    parser.add_argument('--audio-outdir',     default='audio/', help="Where to save intermediate WAVs")
    # Add to argument parser in main()
    parser.add_argument('--evaluate-frequency', type=int, default=0, 
                help="Frequency of LLM-based evaluations (0 to disable, N to evaluate every N pairs)")
    parser.add_argument('--generate-pretrain', action='store_true', help='Generate/refine a high-quality pretrain dataset using OpenSMILE review')
    parser.add_argument('--scan-csv-features', action='store_true', help='Scan OpenSMILE CSVs for phonetic feature patterns using LLM')
    args = parser.parse_args()

    # ------------------------------------------------------------------
    if args.scan_csv_features:
        # Scan CSVs for phonetic patterns
        if not args.audio_outdir or not os.path.isdir(args.audio_outdir):
            print("[ERROR] --audio-outdir must be a valid directory for scanning CSVs.")
            return
        
        # Example label map, customize as needed
        label_map = {
            "harmonic": "harmonic",
            "dissonant": "dissonant"
        }
        
        scan_csvs_for_phonetic_patterns(
            audio_dir=args.audio_outdir,
            label_map=label_map,
            log_file=args.log_file,
            max_files=200,
            feature_keys=[],
            num_threads=args.n
        )
        return
    
    future_keys = defaultdict(dict)
    future_results = defaultdict(dict)
    results = defaultdict()
    feature_results = []
    processed = 0

    if args.llm_bootstrap:
        run_llm_feature_bootstrap(
            model=args.llm_model,
            temperature=args.llm_temperature,
            max_tokens=args.llm_max_tokens,
            num_ctx=args.llm_num_ctx,
            frequency_penalty=args.llm_frequency_penalty,
            seed=args.llm_seed,
            top_p=args.llm_top_p,
            top_k=args.llm_top_k,
            log_file_path=args.llm_log_file if args.llm_log_file else ""
        )
        return  # Early exit after bootstrap

    extractor = AcousticFeatureExtractor(
        opensmile_path   = args.opensmile_path,
        opensmile_config = args.opensmile_config
    )
    os.makedirs(args.audio_outdir, exist_ok=True)
    
    log: List[str] = []
    
    # --- Reset log file every launch ---
    log_file = open(args.log_file, 'w', encoding='utf-8')  # 'w' mode ensures reset
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(f"=== PHONETIC ANALYSIS LOG ===\n")
    log_file.write(f"Started: {timestamp}\n")
    log_file.write(f"Model: {args.model}\n")
    log_file.write(f"Parameters: n={args.n}, corpus_limit={args.corpus_limit}, aggression={args.aggression}\n")
    log_file.write(f"Eightword mode: {args.eightword_mode}\n")
    log_file.write(f"Hyphenation carrier: {args.use_hyphenation_carrier}\n")
    log_file.write(f"Random seed: {random_seed}\n")
    
    # ------------------------------------------------------------------
    # Dataset persistence
    # ------------------------------------------------------------------
    X_persist, y_persist, metadata = ([], [], {}) if args.reset_data else load_dataset(args.data_file)
    log.append(f'Persisted samples: [cyan]{len(y_persist)}[/]')
    log_file.write(f"Loaded {len(y_persist)} persisted samples\n")

    # -----------------------------------------------------------------------------
    # 🔌  Bootstrap integration
    # -----------------------------------------------------------------------------

    # Replace the pretrain data loading section
    label_map = {
        "_harmonic": 1,
        "_dissonant": 0,
        "_original": 1,      # Treat 'original' as harmonic
        "_paraphrased": 0    # Treat 'paraphrased' as dissonant
    }
    pretrain_audio_dir = os.path.join("warmup_data", "wav_files")
    pretrain_features, pretrain_labels = load_pretrain_data_from_csv(
        pretrain_audio_dir,
        label_map,
        feature_keys=[],  # or your desired features
        log_file=log_file.name
    )

    if not pretrain_features or not pretrain_labels:
        # Try loading from warmup_data if specified pretrain file is empty
        log.append('[yellow]Pretrain file empty, attempting to load from warmup_data[/]')
        log_file.write("Pretrain file empty, loading from warmup_data\n")
        warmup_features, warmup_labels = load_warmup_data("warmup_data", args.eightword_mode, log_file)
        
        if warmup_features and warmup_labels:
            log.append(f'Loaded [cyan]{len(warmup_labels)}[/] entries from warmup_data')
            pretrain_features, pretrain_labels = warmup_features, warmup_labels
        
        # If still no data or explicitly requested to generate
        if (not pretrain_features or not pretrain_labels or args.generate_pretrain):
            log.append('[yellow]Generating high-quality pretrain dataset[/]')
            generate_high_quality_pretrain(
                opensmile_path=args.opensmile_path, 
                opensmile_config=args.opensmile_config,
                output_path=args.pretrain_data,
                log_file=log_file,
                tts_cmd=args.tts_cmd,
                audio_outdir=args.audio_outdir
            )
    pretrain_features, pretrain_labels = load_pretrain_data(args.pretrain_data, args.eightword_mode, log_file)
    
    log.append(f'Pretrain samples: [cyan]{len(pretrain_labels)}[/]')
    X_persist.extend(pretrain_features)
    y_persist.extend(pretrain_labels)
    log.append(f'Combined dataset: [green]{len(y_persist)}[/] samples')
    log_file.write(f"Added {len(pretrain_labels)} pretrain samples\n")

    # ------------------------------------------------------------------
    # Analyze past performance to inform prompting
    # ------------------------------------------------------------------
    performance_analysis = analyze_pformance_analysis = analyze_past_performance(args.data_file, log_file)
    if performance_analysis.get("available", False):
        log.append(f"[blue]Analyzing[/] past performance ({performance_analysis['samples']} samples)")
        if "recommendations" in performance_analysis:
            log.append(f"Found [yellow]{len(performance_analysis['recommendations'])}[/] prompt improvement recommendations")
        
        # Log to file
        log_file.write(f"\n=== USING PERFORMANCE ANALYSIS ===\n")
        log_file.write(f"Data points: {performance_analysis.get('samples', 0)}\n")
        log_file.write(f"Harmonic mean: {performance_analysis.get('h_mean', 0):.3f}\n")
        log_file.write(f"Dissonant mean: {performance_analysis.get('d_mean', 0):.3f}\n")
        log_file.write(f"Recommendations: {performance_analysis.get('recommendations', [])}\n")
    else:
        log.append("[dim]No past performance data available for analysis[/]")

    # ------------------------------------------------------------------
    # Corpus
    # ------------------------------------------------------------------
    corpus = load_corpus(args.data_sets)[: args.corpus_limit]
    # Randomize corpus order
    random.shuffle(corpus)
    log.append(f'Corpus entries: [green]{len(corpus)}[/] (limit {args.corpus_limit})')
    log_file.write(f"Loaded {len(corpus)} corpus entries\n")

    # ------------------------------------------------------------------
    # Statistics tracking
    # ------------------------------------------------------------------
    stats = {
        "Total Pairs": 0,
        "Correct": 0,
        "Incorrect": 0,
        "Harmonic": 0,
        "Dissonant": 0,
        "Harmonic Correct": 0,
        "Dissonant Correct": 0,
        "Harmonic Incorrect": 0,
        "Dissonant Incorrect": 0,
        "Eval Count": 0,
        "Eval Harmonic": 0,
        "Eval Dissonant": 0,
        "Eval Score Sum": 0.0,
        "Eval Score Count": 0,
        "Low Harmonic Quality": 0,
        "Low Dissonant Quality": 0,
        "Low Differentiation Count": 0,
        "Avg H Similarity": "0.00",
        "Avg D Similarity": "0.00",
        "Prediction Accuracy": "0.00",
    }
    
    # After loading or updating the dataset, collect live and historical statistics
    data_file = args.data_file  # or wherever your current data file path is stored
    live_stats = collect_livetime_statistics(data_file, log_file=log_file)
    historical_stats = collect_historical_statistics(data_file, log_file=log_file)  # You must implement this or use analyze_past_performance with a different filter

    # Layout: live stats panel above, historical stats panel below
    stats_column = Layout(name='stats_column')
    stats_column.split_column(
        Layout(name='livestats', ratio=1),
        Layout(name='historical', ratio=1)
    )

    layout = Layout()
    layout.split_column(
        Layout(name='main', ratio=3),
        Layout(name='log', ratio=1)
    )
    layout['main'].split_row(
        Layout(name='progress', ratio=2),
        stats_column,
        Layout(name='hist', ratio=1)
    )

    # Initial panel updates
    stats_column['livestats'].update(make_livestats_panel(live_stats))
    stats_column['historical'].update(make_historical_panel(historical_stats))

    # ...layout setup...
    layout = Layout()
    layout.split_column(
        Layout(name='main', ratio=3),
        Layout(name='log', ratio=1)
    )
    layout['main'].split_row(
        Layout(name='progress', ratio=2),
        Layout(name='summary', ratio=1),
        Layout(name='hist', ratio=1)
    )
    layout['progress'].update(make_progress_panel(0, len(corpus), "0.00", stats))
    layout['summary'].update(make_summary_table(stats))
    layout['hist'].update(Panel("Waiting for data...", title="[cyan]Similarity Distribution[/]", border_style="green"))

    # Set up UI update queue
    ui_queue = UIUpdateQueue()
    
    feats = []
    sims = []
    labels = []
    row_idx = 0
    all_row_data = []
    h_similarities = []
    d_similarities = []
    
    # Process corpus entries with simpler UI updates
    with Live(layout, console=console, auto_refresh=False, refresh_per_second=10.0) as live:
        not_enough_variations_warned = False  # ApiNotes: Only warn once if not enough variations are generated
        update_thread_obj = threading.Thread(target=update_thread, args=(ui_queue, live), daemon=True)
        update_thread_obj.start()

        for entry_idx, entry in enumerate(track(corpus, description='Processing entries')):
            entry_id = entry.get('id', f"entry_{entry_idx}")
            
            # Use the correct field for the base text (ApiNotes: see corpus schema)
            base_text = entry.get('content', '')  # Use 'content' as the base text for LLM variation

            # ...existing code...
            variations = generate_variations_ollama(
                base_text,
                n_variations=args.n,
                ctx=args.ctx_length,
                temperature=args.llm_temperature,
                top_p=args.llm_top_p,
                top_k=args.llm_top_k
            )
            # ApiNotes: Only warn once per session if not enough variations are generated, then always proceed.
            if len(variations) < args.n and not not_enough_variations_warned:
                warning_msg = (
                    f"[WARN] Not enough stylistic variations generated for at least one entry "
                    f"(requested {args.n}, got {len(variations)}). Proceeding with available generations."
                )
                print(warning_msg)
                if log_file:
                    log_file.write(warning_msg + "\n")
                not_enough_variations_warned = True

            # Proceed with whatever variations are available...
            for var_idx, variation in enumerate(variations):
                log.append(f"Var {var_idx+1}: [dim]{variation}...[/]")
                
                # Queue UI log update
                ui_queue.add_update(lambda l=log.copy(): layout['log'].update(log_panel(l)))
                
                log_file.write(f"\n--- Processing Variation {var_idx+1}/{len(variations)} ---\n")
                log_file.write(f"VARIATION: {variation}\n")
                
                # Randomize order of harmonic/dissonant generation to get more variety
                resonance_pairs = [('harmonic', 1), ('dissonant', 0)]
                random.shuffle(resonance_pairs)
                for resonance in ["harmonic", "dissonant"]:
                    prompt = build_prompt(entry, resonance)
                    candidate, analysis = generate_with_feedback(
                        prompt, resonance, max_attempts=args.n, opensmile_analyze_fn=extractor.extract_opensmile, 
                        quality_thresholds={"harmonic": 8.0, "dissonant": 12.0},  # Different thresholds by type
                        log_file=log_file,
                        temperature=args.llm_temperature,
                        top_p=args.llm_top_p,
                        top_k=args.llm_top_k
                    )
                for resonance, lab in resonance_pairs:
                    # Generate pair
                    txt = generate_pair_ollama(variation, pair_type=resonance, 
                                              temperature=args.llm_temperature,
                                              max_tokens=args.llm_max_tokens,
                                              #seed=args.
                                              top_p=args.llm_top_p,
                                              log_file=log_file)
                    
                    # --- begin acoustic‐feature integration ---
                    # ApiNotes: All stateful lists (like feats) must be managed explicitly and in lockstep with the main loop.
                    #           See file-level ApiNotes.md for imperative state management.
                    # Always append a new feature vector for this pair/variation BEFORE any extension.
                    feats.append([])  # ApiNotes: This ensures feats[-1] is always valid for extension.

                    if args.tts_cmd:
                        wav_name = f"e{entry_idx}_v{var_idx}_{resonance}.wav"
                        wav_path = os.path.join(args.audio_outdir, wav_name)
                        os.system(args.tts_cmd.format(text=txt.replace('"','\\"'), out=wav_path))
                        praat_features = extractor.extract_praat(wav_path)
                        tonal_amp     = praat_features['sd_f0']
                        tonal_collapse = 1.0 - tonal_amp / max(praat_features['mean_f0'], 1e-3)
                        # feats[-1] is guaranteed to exist here
                        feats[-1].extend([tonal_amp, tonal_collapse])
                        log_file.write(f"AUDIO_FEATURES: sdF0={tonal_amp:.3f}, collapse={tonal_collapse:.3f}\n")
                    # --- end acoustic‐feature integration ---

                    
                    # Extract phonemes and calculate similarity
                    try:
                        variation_phonemes = text_to_phonemes(variation)
                        txt_phonemes = text_to_phonemes(txt)
                        
                        if not variation_phonemes or not txt_phonemes:
                            log.append(f"[red]Error:[/] No phonemes extracted for {'variant' if not variation_phonemes else 'generated text'}")
                            log_file.write(f"ERROR: No phonemes extracted\n")
                            
                            # Queue UI log update
                            ui_queue.add_update(lambda l=log.copy(): layout['log'].update(log_panel(l)))
                            continue

                        sim_cos = cosine(phoneme_vec(variation_phonemes), phoneme_vec(txt_phonemes))
                        final_sim = sim_cos

                        # Apply eightword score if enabled
                        if args.eightword_mode:
                            ew_v = eightword_harmonic_score(variation)
                            ew_t = eightword_harmonic_score(txt)
                            ew_sim = 1.0 - abs(ew_v - ew_t)
                            final_sim = (sim_cos + ew_sim) / 2.0

                        # --- Logging for threshold rejection ---
                        # Example: If you have a minimum similarity threshold for acceptance
                        min_sim_threshold = 0.2  # Example threshold, adjust as needed
                        max_sim_threshold = 0.95 # Example for dissonant, adjust as needed

                        # For harmonic, throw out if too low; for dissonant, throw out if too high
                        if (lab == 1 and final_sim < min_sim_threshold) or (lab == 0 and final_sim > max_sim_threshold):
                            reason = (
                                f"Rejected by threshold: "
                                f"{'harmonic' if lab == 1 else 'dissonant'} "
                                f"similarity={final_sim:.3f} "
                                f"(min={min_sim_threshold} max={max_sim_threshold})"
                            )
                            log.append(f"[yellow]{reason}[/]")
                            log_file.write(f"[REJECTED] {reason}\n")
                            log_file.write(f"[REJECTED] Variation: {variation}\n")
                            log_file.write(f"[REJECTED] Generated: {txt}\n")
                            # Optionally, log phonemes for debugging
                            log_file.write(f"[REJECTED] Variation phonemes: {variation_phonemes}\n")
                            log_file.write(f"[REJECTED] Generated phonemes: {txt_phonemes}\n")
                            # UI update
                            ui_queue.add_update(lambda l=log.copy(): layout['log'].update(log_panel(l)))
                            continue

                        # Store similarity for stats
                        if lab == 1:
                            h_similarities.append(final_sim)
                        else:
                            d_similarities.append(final_sim)
                            
                        # Update data structures
                        sims.append(final_sim)
                        labels.append(lab)
                        feats.append([final_sim])
                        
                        # Predict label based on similarity threshold
                        pred = 1 if final_sim >= 0.5 else 0
                        is_correct = pred == lab
                        
                        # Update stats
                        stats["Total Pairs"] += 1
                        if is_correct:
                            stats["Correct"] += 1
                        else:
                            stats["Incorrect"] += 1

                        # HARMONIC/DISSONANT
                        if lab == 1:
                            stats["Harmonic"] += 1
                            if is_correct:
                                stats["Harmonic Correct"] += 1
                            else:
                                stats["Harmonic Incorrect"] += 1
                        else:
                            stats["Dissonant"] += 1
                            if is_correct:
                                stats["Dissonant Correct"] += 1
                            else:
                                stats["Dissonant Incorrect"] += 1
                            
                        # Create row data for table
                        row_data = (
                            f"{row_idx}",
                            f"{final_sim:.3f}",
                            f"{lab}",
                            f"[{'green' if is_correct else 'red'}]{pred}[/]",
                            f"{entry.get('id', 'entry_'+str(entry_idx))}-{var_idx}",
                            f"{'H' if lab == 1 else 'D'}",
                            f"[{'green' if is_correct else 'red'}]{'✓' if is_correct else '✗'}[/]"
                        )
                        all_row_data.append(row_data)
                        row_idx += 1
                        
                        # Example: Evaluation logic
                        stats["Eval Count"] += 1
                        if lab == 1:
                            stats["Eval Harmonic"] += 1
                        else:
                            stats["Eval Dissonant"] += 1

                        stats["Eval Score Sum"] += final_sim
                        stats["Eval Score Count"] += 1

                        # Example: Low quality detection (adjust threshold as needed)
                        if lab == 1 and final_sim < 0.5:
                            stats["Low Harmonic Quality"] += 1
                        if lab == 0 and final_sim > 0.5:
                            stats["Low Dissonant Quality"] += 1

                        # Example: Low differentiation (difference between harmonic and dissonant similarity)
                        if abs(final_sim - (sum(h_similarities)/len(h_similarities) if h_similarities else 0)) < 0.1:
                            stats["Low Differentiation Count"] += 1

                        # Example: Update average similarity metrics
                        if h_similarities:
                            stats["Avg H Similarity"] = f"{sum(h_similarities) / len(h_similarities):.3f}"
                        if d_similarities:
                            stats["Avg D Similarity"] = f"{sum(d_similarities) / len(d_similarities):.3f}"

                        # Example: Update prediction accuracy
                        if stats["Total Pairs"] > 0:
                            stats["Prediction Accuracy"] = f"{(stats['Correct'] / stats['Total Pairs']) * 100:.1f}%"
                        
                        # Queue UI statistics updates
                        ui_queue.add_update(
                            lambda s=stats.copy(): layout['summary'].update(make_summary_table(s))
                        )
                        
                        # Only update histogram when we have enough data (more stable)
                        if len(sims) >= 4:
                            ui_queue.add_update(
                                lambda s=sims.copy(), l=labels.copy(): layout['hist'].update(make_hist(s, l))
                            )
                        
                        log_file.write(f"PREDICTION: {pred} (Expected: {lab}), Correct: {is_correct}\n")
                        
                    except Exception as e:
                        error_msg = f"[ERROR] Processing pair failed: {e}"
                        log.append(f"[red]Error:[/] {e}")
                        print(error_msg, file=sys.stderr)
                        log_file.write(f"{error_msg}\n")
                        
                        # Queue UI log update
                        ui_queue.add_update(lambda l=log.copy(): layout['log'].update(log_panel(l)))

                    # After calculating similarity in main loop
                    do_evaluation = args.evaluate_frequency > 0 and (stats["Total Pairs"] % args.evaluate_frequency) == 0

                    if do_evaluation:
                        evaluation = evaluate_pair_quality(variation, txt, resonance, args.model, log_file)
                        if 'error' not in evaluation:
                            quality_msg = f"Quality: {evaluation['quality']}/10, Diff: {evaluation['differentiation']}/10"
                            log.append(f"[{resonance}] {quality_msg}")
                            
                            # Track low quality texts for adaptation
                            if resonance == "harmonic" and evaluation['quality'] < 5:
                                stats["Low Harmonic Quality"] = stats.get("Low Harmonic Quality", 0) + 1
                            elif resonance == "dissonant" and evaluation['quality'] < 5:
                                stats["Low Dissonant Quality"] = stats.get("Low Dissonant Quality", 0) + 1
                            
                            # Track overall differentiation for adaptation
                            if evaluation['differentiation'] < 5:
                                stats["Low Differentiation Count"] = stats.get("Low Differentiation Count", 0) + 1
                            
                            # Update stats display
                            ui_queue.add_update(
                                lambda s=stats.copy(): layout['summary'].update(make_summary_table(s))
                            )
                            
                            # Add evaluation info to log file
                            log_file.write(f"EVALUATION: Quality={evaluation['quality']}/10, Differentiation={evaluation['differentiation']}/10\n")
                        else:
                            log.append(f"[{resonance}] Evaluation failed: {evaluation.get('error', 'Unknown error')}")
                            log_file.write(f"EVALUATION ERROR: {evaluation.get('error', 'Unknown error')}\n")
                        
                        # Queue UI log update
                        ui_queue.add_update(lambda l=log.copy(): layout['log'].update(log_panel(l)))
        
        # Stop UI update thread
        ui_queue.stop()
        update_thread_obj.join(timeout=1.0)
        
        # Display final table with all results
        final_table = make_table()
        for row_data in all_row_data:
            final_table.add_row(*row_data)
        
        layout['main'].update(final_table)
        live.refresh()

    # ------------------------------------------------------------------
    # Merge + save dataset
    # ------------------------------------------------------------------
    X_all, y_all = X_persist + feats, y_persist + labels
    save_dataset(args.data_file, X_all, y_all)
    log_file.write(f"\n=== DATASET SAVED ===\n")
    log_file.write(f"Path: {args.data_file}\n")
    log_file.write(f"Total samples: {len(y_all)}\n")

    # ------------------------------------------------------------------
    # Optional training / evaluation
    # ------------------------------------------------------------------
    log_file.write(f"\n=== MODEL TRAINING/EVALUATION ===\n")
    if args.mode in {'both', 'train'} and len(y_all) > 4:
        if args.mode == 'both':
            # Use stratified split to maintain class distribution
            X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=1 - args.split, stratify=y_all, random_state=random_seed)
            log_file.write(f"Split: {len(X_tr)} train, {len(X_te)} test samples\n")
        else:
            X_tr, y_tr = X_all, y_all
            X_te, y_te = None, None
            log_file.write(f"Using all {len(X_tr)} samples for training\n")

        log_file.write(f"Training LogisticRegression classifier\n")
        clf = LogisticRegression(solver='liblinear').fit(X_tr, y_tr)
        
        if X_te is not None and y_te is not None and len(X_te) > 0:
            log_message("INFO", "Hold-out Evaluation:", log_file)
            log_file.write(f"\n--- Test Set Evaluation ---\n")
            y_pred = clf.predict(X_te)
            
            # Use zero_division=1 to avoid warnings when a class has no samples
            report = classification_report(y_te, y_pred, zero_division=1)
            print(report)
            accuracy = accuracy_score(y_te, y_pred)
            log_message("INFO", f"Accuracy: {accuracy:.3f}", log)
            log_file.write(f"Classification Report:\n{report}\n")
            log_file.write(f"Accuracy: {accuracy:.3f}\n")
        else:
            console.print('\n[cyan]Training Set Evaluation:[/]')
            log_file.write(f"\n--- Training Set Evaluation ---\n")
            y_pred = clf.predict(X_tr)
            report = classification_report(y_tr, y_pred, zero_division=1)
            print(report)
            accuracy = accuracy_score(y_tr, y_pred)
            console.print(f'Accuracy: [green]{accuracy:.3f}[/]')
            log_file.write(f"Classification Report:\n{report}\n")
            log_file.write(f"Accuracy: {accuracy:.3f}\n")
    else:
        console.print('[yellow]Not enough data to train / evaluate.[/]')
        log_file.write("Not enough data to train/evaluate classifier\n")

    # Replace the current save_dataset call
    metadata = performance_analysis.get('metadata', {})
    # Add quality evaluation stats
    if "Low Harmonic Quality" in stats or "Low Dissonant Quality" in stats or "Low Differentiation Count" in stats:
        metadata["quality_stats"] = {
            "low_harmonic": stats.get("Low Harmonic Quality", 0),
            "low_dissonant": stats.get("Low Dissonant Quality", 0),
            "low_differentiation": stats.get("Low Differentiation Count", 0)
        }

    X_all, y_all = X_persist + feats, y_persist + labels
    save_dataset(args.data_file, X_all, y_all, metadata)

    # Close log file
    log_file.write(f"\n=== SESSION COMPLETE ===\n")
    log_file.write(f"Ended: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.close()
    
    # Print final results
    console.print(f"\n[green]Processing complete![/]")
    console.print(f"  Total pairs processed: [cyan]{stats['Total Pairs']}[/]")
    console.print(f"  Prediction accuracy: [magenta]{stats['Prediction Accuracy']}[/]")
    console.print(f"  Dataset saved to: [yellow]{args.data_file}[/]")
    console.print(f"  Detailed log saved to: [yellow]{args.log_file}[/]")
    
main()
