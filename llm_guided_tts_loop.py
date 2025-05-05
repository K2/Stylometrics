"""
ApiNotes.md (File-level) – llm_guided_tts_loop.py

Role:
    Provides an LLM-guided loop for predictive planning, generation, and feature extraction of TTS audio.
    The LLM acts as an agent: it proposes phoneme/syllable/superposition patterns, predicts likely synthesis outcomes,
    requests generation, and analyzes extracted features to iteratively refine its strategy.
    Designed for closed-loop, self-improving TTS benchmarking, stylometric probing, and dataset enrichment.

Design Goals:
    - Enable an LLM to autonomously guide TTS synthesis and feature extraction.
    - Use modular, composable functions for planning, generation, and analysis.
    - Avoid code duplication by leveraging tts_tool, gguf_orpheus, generate_phoneme_wavs, and acoustic_features modules.
    - Document all interface and behavioral assumptions in ApiNotes.md.
    - Provide hooks for human-in-the-loop or fully autonomous operation.
    - Log all LLM prompts, responses, and actions for reproducibility and future LLM uptake.

Architectural Constraints:
    - No subprocesses for TTS or feature extraction; use Python APIs only.
    - All plans, actions, and results are logged for traceability.
    - File must be importable and runnable as a script.
    - All changes must be reflected in project-level ApiNotes.md and this file's ApiNotes.md.

Happy-path:
    1. LLM proposes a batch of phoneme/syllable/pattern prompts to synthesize.
    2. System generates audio using gguf_orpheus.generate_speech_from_api.
    3. Features are extracted using multiple OpenSMILE configs (prosody, cvssink, etc.) for comparison.
    4. LLM receives results, updates its plan, and repeats until convergence or stop condition.
    5. All actions, prompts, and results are logged for review.

"""

from multiprocessing import Array
import os
import json
import string
from typing import Any
import uuid
import ollama  # Official Python Ollama client
from gguf_orpheus import generate_speech_from_api, AVAILABLE_VOICES
from generate_phoneme_wavs import (
    best_effort_phonetic_spelling,
    has_audio_features,
    superposition_phoneme_patterns,
    get_all_phonemes,
    get_all_syllables,
    VOICES,
    MODIFIERS,
    OUTPUT_ROOT,
    SAMPLE_RATE,
)
from dataset_helpers import extract_opensmile_features

OLLAMA_MODEL = "granite3.3:latest"

# --- OpenSMILE config discovery and feature extraction ---

def get_prosody_configs(prosody_dir="conf/prosody"):
    """
    ApiNotes: Returns a list of OpenSMILE config file paths under the given prosody directory.
    """
    configs = []
    if os.path.isdir(prosody_dir):
        for fname in os.listdir(prosody_dir):
            if fname.endswith(".conf"):
                configs.append(os.path.join(prosody_dir, fname))
    return configs

def extract_features_with_configs(wav_path, configs):
    """
    ApiNotes: Extracts features from wav_path using each OpenSMILE config in configs.
    Returns a dict: {config_name: features}
    """
    results = {}
    for config_path in configs:
        config_name = os.path.basename(config_path)
        try:
            features = extract_opensmile_features(wav_path, config_path=config_path)
            results[config_name] = features
        except Exception as e:
            results[config_name] = {"error": str(e)}
    return results

def ollama_llm(prompt, system=None):
    if system is None:
        system = (
            "You are an expert in TTS waveform analysis and synthesis planning. "
            "You have access to the following tool: tts_loop_tool(prompts: list[dict], voice: str, modifier: str, ...) -> list[dict]. "
            "Call this tool when you want to synthesize and analyze a batch of prompts. "
            "Note: You do not always need to use expressive/emotive modifiers (such as <happy>, <sad>, etc). "
            "A baseline with default settings (no modifier or 'normal') is often preferred and most capable, "
            "unless a specific expressive effect is required for the experiment. "
            "Default, non-emotive generations are typically the most robust for waveform and superposition analysis. "
            "For all synthesis and analysis, only use the voices 'leo' and 'tara'."
        )
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
    )
    return response["message"]["content"]

# --- LLM Agent Interface (stub for real LLM integration) ---

def llm_propose_batch(context):
    """
    Uses Ollama to propose a batch of prompts for TTS synthesis.
    Ensures each token sequence is at least 7 tokens long.
    Prompts are guided by ApiNotes.md to cover phonetic, stenographic, waveform, distance, amplification, harmony, rhyming, and song-like generation.
    """
    context_summary = json.dumps(context["history"][-1] if context["history"] else {}, indent=2)
    prompt = (
        "You are guiding a TTS system for stylometric, phonetic, and waveform analysis. "
        "Your goal is to propose a batch of TTS synthesis prompts that comprehensively probe the following aspects:\n"
        "- Phonetic coverage: Use a variety of IPA and ARPAbet phonemes, including edge cases and ambiguous sounds.\n"
        "- Stenographic/Minimal pairs: Include prompts that test minimal differences (e.g., 'p' vs 'b', 's' vs 'z').\n"
        "- Waveform detection: Use repeated tokens and superpositions (e.g., 'kkkkkkk', 'sssssss', 'ppppppp') to test for presence and attenuation.\n"
        "- Distance/contrast: Include prompts that alternate or juxtapose distant phonemes (e.g., 'k,s,k,s,k,s', 'a,e,i,o,u').\n"
        "- Amplification/harmony: Propose patterns that may cause constructive or destructive interference (e.g., 'k,k,kk,kkk', 'ss,ss,ssss').\n"
        "- Rhyming and song-like: Attempt prompts that rhyme or have a song-like structure (e.g., 'bee,see,tree,free', 'la,la,la,la,la,la,la', 'do,re,mi,fa,so,la,ti', 'fa,fa,fa,fa,fa,fa,fa').\n"
        "- Use both 'leo' and 'tara' voices, and do not use any modifier tag (such as <normal> or <happy>), unless expressivity is required for the experiment.\n"
        "- Each prompt should be at least 7 tokens long (e.g., 'kkkkkkk').\n"
        "- Respond with a JSON list of dicts: {prompt, voice, modifier, pattern_type}.\n"
        f"Context:\n{context_summary}\n"
        "Available voices: leo, tara\n"
        "Available modifiers: " + ", ".join(MODIFIERS) + "\n"
        "Example: ["
        "{\"prompt\": \"kkkkkkk\", \"voice\": \"leo\", \"modifier\": \"\", \"pattern_type\": \"waveform\"}, "
        "{\"prompt\": \"p,b,p,b,p,b\", \"voice\": \"tara\", \"modifier\": \"\", \"pattern_type\": \"stenographic\"}, "
        "{\"prompt\": \"k,s,k,s,k,s\", \"voice\": \"leo\", \"modifier\": \"\", \"pattern_type\": \"distance\"}, "
        "{\"prompt\": \"ss,ss,ssss\", \"voice\": \"tara\", \"modifier\": \"\", \"pattern_type\": \"amplification\"}, "
        "{\"prompt\": \"bee,see,tree,free\", \"voice\": \"leo\", \"modifier\": \"\", \"pattern_type\": \"rhyme\"}, "
        "{\"prompt\": \"la,la,la,la,la,la,la\", \"voice\": \"tara\", \"modifier\": \"\", \"pattern_type\": \"song\"}, "
        "{\"prompt\": \"do,re,mi,fa,so,la,ti\", \"voice\": \"leo\", \"modifier\": \"\", \"pattern_type\": \"song\"}, "
        "{\"prompt\": \"fa,fa,fa,fa,fa,fa,fa\", \"voice\": \"tara\", \"modifier\": \"\", \"pattern_type\": \"song\"}, "
        "{\"prompt\": \"mi,mi,mi,mi,mi,mi,mi\", \"voice\": \"leo\", \"modifier\": \"\", \"pattern_type\": \"song\"}"
        "]"
    )
    try:
        llm_response = ollama_llm(prompt)
        batch = json.loads(llm_response)
        # Validate batch
        if isinstance(batch, list) and all(isinstance(x, dict) for x in batch):
            return batch
    except Exception as e:
        print(f"[LLM ERROR] {e}\nFalling back to default batch.")
    # Fallback: default batch with diverse, ApiNotes-guided prompts (no <normal>)
    batch = [
        {"prompt": "kkkkkkk", "voice": "leo", "modifier": "", "pattern_type": "waveform"},
        {"prompt": "sssssss", "voice": "tara", "modifier": "", "pattern_type": "waveform"},
        {"prompt": "p,b,p,b,p,b", "voice": "leo", "modifier": "", "pattern_type": "stenographic"},
        {"prompt": "k,s,k,s,k,s", "voice": "tara", "modifier": "", "pattern_type": "distance"},
        {"prompt": "ss,ss,ssss", "voice": "leo", "modifier": "", "pattern_type": "amplification"},
        {"prompt": "bee,see,tree,free", "voice": "tara", "modifier": "", "pattern_type": "rhyme"},
        {"prompt": "la,la,la,la,la,la,la", "voice": "leo", "modifier": "", "pattern_type": "song"},
        {"prompt": "do,re,mi,fa,so,la,ti", "voice": "tara", "modifier": "", "pattern_type": "song"},
        {"prompt": "fa,fa,fa,fa,fa,fa,fa", "voice": "leo", "modifier": "", "pattern_type": "song"},
        {"prompt": "mi,mi,mi,mi,mi,mi,mi", "voice": "tara", "modifier": "", "pattern_type": "song"},
    ]
    return batch

def llm_analyze_results(batch, features):
    """
    Uses Ollama to analyze the extracted features and propose the next plan.
    If the LLM decides to stop, it must print a thorough analysis discussing its insights and reasoning for stopping.
    """
    llm_response = None
    plan = None
    prompt = (
        "Given the following batch and extracted features, analyze the results. "
        "Suggest whether to continue, and if so, propose the next batch of prompts. "
        "If you believe waveform detection or superposition control is achieved, signal stop. "
        "If you decide to stop, provide a thorough analysis discussing your insights, reasoning, and what was learned from the experiment. "
        "Respond with a JSON dict: {continue: bool, next_batch: list, analysis: str}.\n"
        f"Batch:\n{json.dumps(batch, indent=2)}\n"
        f"Features:\n{json.dumps(features, indent=2)}"
    )
    try:
        llm_response: Any = ollama_llm(prompt)
        plan = json.loads(llm_response)
        if isinstance(plan, dict) and "continue" in plan:
            if not plan.get("continue", False):
                print("\n[LLM FINAL ANALYSIS]\n" + plan.get("analysis", "(No analysis provided by LLM)"))
            return plan
    except Exception as e:
        print(f"[LLM ERROR] {e}\nFalling back to stop. \nllm_response: {llm_response}\nPlan: {plan}")
    return {"continue": False, "next_batch": [], "analysis": "LLM failed to provide analysis due to an error."}

# --- Main LLM-guided loop ---

def llm_guided_tts_loop(log_dir="llm_guided_logs", prosody_dir="conf/prosody"):
    """
    ApiNotes: Main loop for LLM-guided TTS planning, generation, and feature extraction.
    Uses gguf_orpheus.generate_speech_from_api for all TTS synthesis.
    Extracts features using all OpenSMILE configs in the prosody folder (and csvsink if available).
    """
    os.makedirs(log_dir, exist_ok=True)
    context = {"history": []}
    loop_id = str(uuid.uuid4())
    step = 0
    prosody_configs = get_prosody_configs(prosody_dir)
    if not prosody_configs:
        print(f"[WARN] No OpenSMILE prosody configs found in {prosody_dir}.")
    while True:
        # 1. LLM proposes batch
        batch = llm_propose_batch(context)
        if not batch:
            print("LLM proposed no prompts. Exiting.")
            break
        print(f"[LLM] Proposed {len(batch)} prompts for step {step}.")
        results = []
        for item in batch:
            prompt = item["prompt"]
            voice = item["voice"]
            modifier = item["modifier"]
            pattern_type = item["pattern_type"]
            fname = f"{pattern_type}_{modifier}_{voice}_{uuid.uuid4().hex[:8]}.wav"
            wav_path = os.path.join(log_dir, fname)
            # Use gguf_orpheus for TTS generation
            generate_speech_from_api(prompt=prompt, voice=voice, output_file=wav_path)
            if not has_audio_features(wav_path):
                # Fallback with best-effort spelling
                core = prompt.split(">")[1] if ">" in prompt else prompt
                fallback = best_effort_phonetic_spelling(core)
                fallback_prompt = f"<{modifier}>{fallback}"
                generate_speech_from_api(prompt=fallback_prompt, voice=voice, output_file=wav_path)
            # Extract features using all discovered prosody configs and csvsink
            features_by_config = extract_features_with_configs(wav_path, prosody_configs)
            results.append({
                "prompt": prompt,
                "voice": voice,
                "modifier": modifier,
                "pattern_type": pattern_type,
                "wav_path": wav_path,
                "features_by_config": features_by_config,
            })
        # Log results for this step
        log_path = os.path.join(log_dir, f"step_{step}_{loop_id}.json")
        with open(log_path, "w") as f:
            json.dump(results, f, indent=2)
        # 2. LLM analyzes results and proposes next batch or stop
        plan = llm_analyze_results(batch, results)
        context["history"].append({"batch": batch, "results": results, "plan": plan})
        if not plan.get("continue", False):
            print("LLM signaled stop. Exiting loop.")
            break
        step += 1

def tts_loop_tool(
    prompts,
    voice=VOICES[0],
    modifier=MODIFIERS[0],
    log_dir="llm_guided_logs",
    prosody_dir="conf/prosody"
):
    """
    ApiNotes: Exposes the TTS loop as a callable tool for the LLM or other agents.
    Accepts a list of prompt dicts: [{"prompt": str, "voice": str, "modifier": str, "pattern_type": str}, ...]
    Synthesizes audio and extracts features for each prompt using all OpenSMILE prosody configs.
    Returns a list of result dicts with prompt, wav_path, and features_by_config.
    """
    os.makedirs(log_dir, exist_ok=True)
    prosody_configs = get_prosody_configs(prosody_dir)
    results: list[dict] = []  # FIXED: was list[str], should be list[dict]
    for item in prompts:
        prompt = item["prompt"]
        v = item.get("voice", voice)
        m = item.get("modifier", modifier)
        pattern_type = item.get("pattern_type", "custom")
        fname: str    = f"{pattern_type}_{m}_{v}_{uuid.uuid4().hex[:8]}.wav"
        wav_path: str = os.path.join(log_dir, fname)
        generate_speech_from_api(prompt=prompt, voice=v, output_file=wav_path)
        if not has_audio_features(wav_path):
            core = prompt.split(">")[1] if ">" in prompt else prompt
            fallback = best_effort_phonetic_spelling(core)
            fallback_prompt = f"<{m}>{fallback}"
            generate_speech_from_api(prompt=fallback_prompt, voice=v, output_file=wav_path)
        features_by_config = extract_features_with_configs(wav_path, prosody_configs)
        results.append({
            "prompt": prompt,
            "voice": v,
            "modifier": m,
            "pattern_type": pattern_type,
            "wav_path": wav_path,
            "features_by_config": features_by_config,
        })
    return results

# Register tools for external access (e.g., via Langchain)
TOOLS = {
    "tts_loop_tool": {
        "function": tts_loop_tool,
        "description": (
            "Synthesizes audio and extracts features for a list of prompts. "
            "Input: list of dicts [{prompt, voice, modifier, pattern_type}], "
            "returns: list of dicts with prompt, wav_path, features_by_config."
        )
    }
}

if __name__ == "__main__":
    # Default: run the LLM-guided TTS loop
    llm_guided_tts_loop()

# filepath: /home/files/git/Stylometrics/llm_guided_tts_loop.py.ApiNotes.md
"""
ApiNotes.md (Sidecar) – llm_guided_tts_loop.py

- This module enables an LLM (via Ollama and phi4-reasoning:plus) to autonomously plan, generate, and analyze TTS audio in a closed loop.
- All batch planning, synthesis, and feature extraction are logged for reproducibility.
- The LLM agent interface is now real: llm_propose_batch and llm_analyze_results use the official Python Ollama client.
- TTS synthesis is performed via gguf_orpheus.generate_speech_from_api.
- Feature extraction is performed using all OpenSMILE configs in the prosody folder, with csvsink if available.
- The LLM is instructed to focus on waveform detection and superposition control, with at least 7 tokens per prompt.
- The tts_loop_tool function exposes the TTS loop as a callable tool for the LLM or other agents, enabling flexible, programmatic synthesis and feature extraction.
- Extend this module for human-in-the-loop, curriculum learning, or active learning scenarios.
- See project-level ApiNotes.md for integration and extension guidance.
"""

# Example: agent detects tool call in LLM output and invokes tts_loop_tool
llm_output = ollama_llm("I want to synthesize these prompts: ...")
if "tts_loop_tool" in llm_output:
    # Parse arguments from llm_output (e.g., using json.loads or regex)
    try:
        # FIXED: Parse llm_output as JSON before checking type
        parsed = None
        try:
            parsed = json.loads(llm_output)
        except Exception as e:
            print(f"[TOOL ERROR] Failed to parse LLM output as JSON: {e}")
            parsed = None
        if isinstance(parsed, list):
            prompts = parsed
        elif isinstance(parsed, dict):
            prompts = [parsed]
        else:
            print("[TOOL ERROR] LLM output did not contain valid prompt list or dict.")
            prompts = []
        if prompts:
            results: list[dict] = tts_loop_tool(prompts)
    except Exception as e:
        print(f"[TOOL ERROR] {e}")
    # Pass results back to LLM or use in your workflow

# Example usage:
# prompts = [
#     {"prompt": "<normal>kkkkkkk", "voice": "leo", "modifier": "normal", "pattern_type": "phoneme"}
# ]
# results = tts_loop_tool(prompts)