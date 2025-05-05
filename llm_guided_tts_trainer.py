"""
LLM-Guided TTS Trainer/Warmup Script

- This script "warms up" or trains the LLM-guided TTS loop using CAPT and concept datasets.
- It first tests the LLM's initial (zero-shot) ability to generate prompts for phonetic, stress, intonation, and concept principles.
- Then, it iteratively exposes the LLM to CAPT and concept questions, letting it practice generating prompts and analyzing results, updating its context/history.
- At the end, it saves the LLM's "trained" state/context for future use or fine-tuning.
- See llm_guided_tts_trainer.py.ApiNotes.md for detailed context and extension guidance.
"""

import json
import os
from llm_guided_tts_loop import llm_propose_batch, llm_analyze_results

DATASETS = [
    "data/capt_test.json",
    "data/concept_test.json",
    "data/capt_dev.json",
    "data/concept_dev.json"
]

def load_questions(paths, max_questions=20):
    questions = []
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            data = json.load(f)
        # Accept both dicts with string keys and dicts with non-string keys
        for k in sorted(data.keys(), key=lambda x: str(x)):
            q = data[k]
            # Only include questions with a 'question' field
            if "question" in q:
                questions.append(q)
            if len(questions) >= max_questions:
                break
        if len(questions) >= max_questions:
            break
    return questions

def llm_tts_trainer_warmup(save_path="llm_trained_context.json"):
    """
    1. Test LLM's initial state (zero-shot) on CAPT and concept data.
    2. Iteratively train/warm up the LLM by letting it practice on all datasets.
    3. Save the trained context for future use or fine-tuning.
    """
    questions = load_questions(DATASETS, max_questions=40)
    context = {"history": []}

    print("=== LLM Zero-Shot (Initial State) ===")
    for i, q in enumerate(questions):
        print(f"\n[Q{i}] {q['question']}")
        batch = llm_propose_batch({"history": [{"question": q["question"], "options": {k: v for k, v in q.items() if k in "ABCD"}, "answer": q.get("answer", ""), "type": q.get("type", "")}]})
        print(f"[LLM Zero-Shot Prompts] {batch}")

    print("\n=== LLM Warmup/Training Loop ===")
    for i, q in enumerate(questions):
        # Add question to context/history
        context["history"].append({
            "question": q["question"],
            "options": {k: v for k, v in q.items() if k in "ABCD"},
            "answer": q.get("answer", ""),
            "type": q.get("type", "")
        })
        batch = llm_propose_batch(context)
        print(f"\n[Q{i}] {q['question']}")
        print(f"[LLM Warmup Prompts] {batch}")
        # Simulate feature extraction (dummy features for now)
        fake_features = [{"prompt": item["prompt"], "features": {"dummy": 1}} for item in batch]
        plan = llm_analyze_results(batch, fake_features)
        print(f"[LLM Warmup Analysis] {plan.get('analysis', '')}")
        # Optionally, update context/history with results/plan for curriculum learning

    # Save the trained context for future use or fine-tuning
    with open(save_path, "w") as f:
        json.dump(context, f, indent=2)
    print(f"\n[INFO] LLM trained context saved to {save_path}")

if __name__ == "__main__":
    llm_tts_trainer_warmup()

# filepath: /home/files/git/Stylometrics/llm_guided_tts_trainer.py.ApiNotes.md
"""
ApiNotes.md – llm_guided_tts_trainer.py

- This script is a trainer/warmup for the LLM-guided TTS loop, using CAPT and concept datasets.
- It first tests the LLM's zero-shot (untrained) ability to generate prompts for phonetic, stress, intonation, and concept principles.
- Then, it iteratively exposes the LLM to all questions, letting it practice generating prompts and analyzing results, updating its context/history.
- At the end, it saves the LLM's "trained" state/context for future use or fine-tuning.
- This process helps the LLM acquire the skill of guiding TTS synthesis for language learning and stylometric analysis.
- Extend this script for curriculum learning, active learning, or to include additional datasets.
- No code clones or duplicate logic: all prompt generation and analysis uses llm_guided_tts_loop APIs.
- See project-level ApiNotes.md for integration and extension guidance.
"""