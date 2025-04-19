"""
ApiNotes.md (File-level) – llm_feature_bootstrap.py

Role:
    Implements an imperative training loop that feeds all .json warmup training data (containing original text and extracted OpenSMILE features)
    to the LLM, prompting it to compile and summarize the features associated with resonance/dissonance. The LLM's output is persisted to a new
    bootstrap file, containing all information needed to initialize future LLM-based phonetic analysis.

Design Goals:
    - Provide a reproducible, extensible training loop for LLM-based feature pattern discovery.
    - Aggregate and persist LLM-identified feature associations for future bootstrapping.
    - Ensure all data, prompts, and LLM responses are logged and traceable.
    - Support incremental extension as new training data or features are added.
    - Maintain imperative, step-by-step logic per project and file-level ApiNotes.
    - Avoid circular imports by using the Ollama API directly with explicit arguments for all supported variables.
    - Use centralized logging utilities for all log output.

Architectural Constraints:
    - All training data must be loaded from .json files in a canonical directory.
    - LLM calls must use the Ollama API directly in this file to avoid circular import.
    - All LLM prompts, responses, and summary outputs must be persisted for reproducibility.
    - File size monitored; suggest splitting if exceeding 1/3 context window.
    - All interface and behavioral assumptions are documented in ApiNotes.

Happy-Path:
    1. Discover and load all .json warmup training data files.
    2. For each entry, extract original text and features.
    3. Prompt the LLM to analyze and summarize feature associations.
    4. Aggregate LLM responses and persist to a new bootstrap file.
    5. Log all steps and outputs for traceability.

ASCII Diagram:
    +-------------------+
    |  Load .json data  |
    +-------------------+
              |
              v
    +-------------------+
    |  For each entry:  |
    |  - Prompt LLM     |
    |  - Collect resp   |
    +-------------------+
              |
              v
    +-------------------+
    |  Aggregate & Save |
    +-------------------+
"""

import os
import json
from datetime import datetime
from typing import Union

from tts_tool import synthesize_audio

try:
    import ollama
except ImportError:
    ollama = None

from dataset_helpers import generate_wavs_and_features_from_json
from logging_utils import log_message, LogContext

# Canonical directory for warmup training data
TRAINING_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "warmup_data"))
BOOTSTRAP_OUTPUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "llm_feature_bootstrap.jsonl"))
LOG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "llm_feature_bootstrap.log"))

def load_training_data(data_dir):
    """
    ApiNotes: Loads all .json files from the specified directory.
    Returns a list of dicts with keys: 'original', 'features', and optionally 'label'.
    """


    
    entries = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".json"):
            fpath = os.path.join(data_dir, fname)
            entry = generate_wavs_and_features_from_json(fpath, 
                                                         audio_dir=os.path.join(data_dir, "wav_files"), tts_func=synthesize_audio)
            entries.append(entry)   
           
    assert len(entries) > 0, "ApiNotes: No training data found in directory"
    return entries

def ollama_chat(
    prompt,
    model="phi4",
    temperature=0.3,
    max_tokens=512,
    num_ctx=8192,
    frequency_penalty=None,
    seed=None,
    top_p=None,
    top_k=None,
    log_file=None,
    response_idx=None
):
    """
    ApiNotes: Calls the Ollama API with all supported arguments for LLM inference.
    Returns the response string.
    """
    assert ollama is not None, "ollama must be installed for LLM inference"
    messages = [{"role": "user", "content": prompt}]
    options = {
        "temperature": temperature,
        "num_predict": max_tokens,
        "num_ctx": num_ctx
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
        response = ollama.chat(
            model=model,
            messages=messages,
            stream=False,
            options=options
        )
        content = response['message']['content']
        if log_file:
            tag = f"RESPONSE {response_idx}" if response_idx is not None else "RESPONSE"
            log_message(tag, content, log_file)
        return content
    except Exception as e:
        if log_file:
            tag = f"NEXT RESPONSE {response_idx}" if response_idx is not None else "NEXT RESPONSE"
            log_message(tag, f"LLM call failed: {e}", log_file)
        raise

def prompt_llm_for_feature_patterns(original, features, label=None):
    """
    ApiNotes: Prepares a prompt for the LLM to analyze features and associate them with resonance/dissonance.
    Returns the LLM's response string.
    """
    prompt = (
        "You are an expert in phonetic resonance and dissonance analysis. "
        "Given the following original text and its extracted phonetic features (from OpenSMILE), "
        "identify which features are most indicative of resonance or dissonance. "
        "If a label is provided, use it to guide your analysis. "
        "Summarize your reasoning and list the most important features.\n\n"
        f"Original text:\n{original}\n\n"
        f"Extracted features (JSON):\n{json.dumps(features, indent=2)}\n"
    )
    if label:
        prompt += f"\nLabel: {label}\n"
    prompt += "\nRespond with a JSON object: {\"important_features\": [...], \"reasoning\": \"...\"}\n"
    return prompt

def run_llm_feature_bootstrap(
    model="phi4",
    temperature=0.3,
    max_tokens=512,
    num_ctx=8192,
    frequency_penalty=None,
    seed=None,
    top_p=None,
    top_k=None,
    log_file_path=LOG_FILE
    ) -> Union[str, dict]:
    """
    ApiNotes: Main training loop for LLM feature pattern discovery and bootstrap file creation.
    Uses the Ollama API directly with all supported arguments to avoid circular import.
    Accepts all LLM generation arguments from the caller (e.g., CLI).
    """
    # Step 1: Load all training data
    entries = load_training_data(TRAINING_DATA_DIR)
    results = []
    with LogContext(log_file_path) as logf, open(BOOTSTRAP_OUTPUT, "w") as outf:
        for idx, entry in enumerate(entries):
            original = entry["original"]
            features = entry["features"]
            label = entry.get("label")
            prompt = prompt_llm_for_feature_patterns(original, features, label)
            # Step 2: Query LLM
            try:
                response = ollama_chat(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    num_ctx=num_ctx,
                    frequency_penalty=frequency_penalty,
                    seed=seed,
                    top_p=top_p,
                    top_k=top_k,
                    log_file=logf,
                    response_idx=idx + 1
                )
                # Defensive: Try to parse LLM response as JSON
                try:
                    parsed = json.loads(response)
                except Exception as e:
                    log_message("ERROR", f"Failed to parse LLM response as JSON for entry {idx+1}: {e}\nResponse: {response}", logf)
                    parsed = {"important_features": [], "reasoning": response}
                # Step 3: Aggregate and persist
                result = {
                    "original": original,
                    "features": features,
                    "label": label,
                    "llm_response": parsed,
                    "timestamp": datetime.utcnow().isoformat()
                }
                results.append(result)
                outf.write(json.dumps(result) + "\n")
                log_message("INFO", f"Entry {idx+1} processed and saved.", logf)
            except Exception as e:
                log_message("ERROR", f"LLM call failed for entry {idx+1}: {e}", logf)
                continue
    # Step 4: Summary
    print(f"LLM feature bootstrap completed. {len(results)} entries processed.")
    print(f"Results saved to: {BOOTSTRAP_OUTPUT}")
    
    # Return results to satisfy the return type
    return {"status": "success", "entries_processed": len(results), "output_path": BOOTSTRAP_OUTPUT}

# Acceptance test (expected failure)
def test_llm_feature_bootstrap_expected_failure():
    # Should fail if no data directory or no .json files
    try:
        load_training_data("/tmp/nonexistent_dir")
    except AssertionError as e:
        print(f"(expected failure) {e}")

# ApiNotes: This implementation is imperative, modular, and justified by file-level and project-level ApiNotes.
#           All interface and behavioral assumptions are documented.
#           Acceptance test includes (expected failure) case.