# Rich Debugger for Phonetic Resonance - API Notes

## Overview

This Python script (`stylometric_rich_debugger.py`) provides an interactive command-line interface using the `rich` library to visualize and debug the process of generating phonetically harmonic and dissonant text pairs using Ollama. It processes a corpus of text entries, generates variations based on metadata, creates harmonic/dissonant pairs for each variation, calculates phonetic similarity, trains a simple classifier, and displays results in real-time.

## Design Goals

*   Provide a visual tool for inspecting the phonetic similarity scores of generated text pairs.
*   Allow users to observe the effectiveness of Ollama prompts in creating phonetically distinct (harmonic vs. dissonant) text based on metadata.
*   Generate multiple variations (`N`) for each base text entry to increase data points.
*   Use metadata associated with corpus entries to guide the LLM in generating variations and pairs that maintain the original style.
*   Train a basic classifier (Logistic Regression) to assess the separability of harmonic and dissonant pairs based on phonetic cosine similarity.
*   Utilize the `rich` library for a dynamic and informative terminal UI (Table, Histogram, Logs).

## Architectural Constraints

*   **Paradigm:** Imperative. The script follows a clear sequence: load data, setup UI, loop through entries/variations, generate pairs, calculate features, update UI, train classifier, report results.
*   **Dependencies:** Requires Python 3.x, `ollama`, `nltk` (with `cmudict`), `scikit-learn`, `numpy`, and `rich`.
*   **External Services:** Relies on a running Ollama instance and the availability of the specified model.
*   **Input Data:** Expects JSON dataset files (`data/set*.json`) containing a list of corpus entries, each with `content` and optional `metadata` (genre, subgenre, keywords, etc.). See `set1.json` for format.
*   **Phonetic Analysis:** Uses `nltk.corpus.cmudict` for phoneme conversion. Limited to words present in the dictionary.
*   **Similarity Metric:** Uses cosine similarity between phoneme frequency vectors.

## Core Flow (Happy Path)

1.  **Parse Arguments:** Get Ollama model, N variations, corpus limit, dataset paths.
2.  **Load Data:** Read JSON files specified in `--data-sets` using `load_datasets`. Handle file not found or invalid JSON.
3.  **Initialize UI:** Create `rich` Console, Layout, Table, and initial Panels for histogram and logs. Start `Live` display.
4.  **Process Corpus:** Loop through each `entry` in the (limited) corpus.
    a.  **Generate Variations:** Call `generate_variations_ollama` to get `N` text variations based on `entry`'s content and metadata.
    b.  **Process Variations:** Loop through each `variation_text`.
        i.  **Generate Pairs:** Call `generate_pair_ollama` twice (once for 'harmonic', once for 'dissonant'), passing the `variation_text` and `entry`'s metadata.
        ii. **Calculate Phonetics:** Convert `variation_text`, `harmonic_text`, and `dissonant_text` to phoneme lists using `text_to_phonemes`. Handle potential errors if words are not in `cmudict`.
        iii. **Vectorize:** Convert phoneme lists to frequency vectors using `phoneme_vector`.
        iv. **Calculate Similarity:** Compute cosine similarity between the variation vector and the harmonic/dissonant vectors using `cosine_similarity`.
        v.  **Store Data:** Append the variation-generator pairs, labels (1/0), similarity scores, and features (similarity score) to respective lists (`pairs`, `labels`, `similarities`, `all_features`).
        vi. **Update UI:** Add rows to the `table` showing the score, expected label, simple prediction, variation ID, and resonance type. Update the `histogram` panel using `create_histogram_panel`. Update the `log` panel using `create_log_panel`.
5.  **Train Classifier:** After the loop, check if sufficient data was collected. If yes, call `train_classifier` with `all_features` and `labels`.
6.  **Report Results:** Predict labels using the trained classifier and print a `classification_report`. Handle potential errors during training/prediction.

## Key Functions & Assumptions

*   **`generate_variations_ollama`:** Assumes Ollama can generate meaningful variations based on the prompt and metadata, separated by `---VARIATION---`. Falls back to returning originals if generation fails or returns too few.
*   **`generate_pair_ollama`:** Assumes Ollama can generate smoother or disrupted text based on the prompt and metadata. Falls back to the input text if generation fails.
*   **`text_to_phonemes`:** Assumes `cmudict` is available. Ignores words not found in the dictionary.
*   **`train_classifier`:** Assumes `LogisticRegression` can handle the single-feature input. `zero_division=1` is used in `classification_report` for robustness.

## Debugging & Verification

*   **Logs:** The log panel provides real-time updates on processing steps, generated similarities, and potential errors during generation or calculation.
*   **Table:** Shows individual similarity scores, expected vs. predicted labels (based on a simple 0.5 threshold for display), allowing quick identification of misclassifications or unexpected scores.
*   **Histogram:** Visualizes the distribution of similarity scores for harmonic (1) and dissonant (0) pairs, indicating the separability of the two classes based on the phonetic feature.
*   **Assertions:** Basic assertions are included to check for empty content or failed phoneme extraction, though more could be added.
*   **Error Handling:** `try...except` blocks are used around Ollama calls and phonetic calculations to prevent crashes and log issues. Fallbacks (using original text) are implemented for generation failures.

## Future Improvements

*   More sophisticated feature engineering beyond simple cosine similarity.
*   Option to use different classifiers.
*   Save/load generated data and trained models.
*   More robust error handling and reporting for Ollama API issues.
*   Allow configuration of the prediction threshold used in the table display.
*   Integrate CRC hashing or other methods to track code duplication if this script's logic is reused elsewhere.
```// filepath: /home/files/git/Stylometrics/stylometric_rich_debugger.ApiNotes.md

Finally, update the project-level `ApiNotes.md`.

````markdown
// filepath: /home/files/git/Stylometrics/ApiNotes.md
// ... existing content ...

## Architecture

*   **Carriers (`stylometric_carrier.genai.mts`, `quote_style_carrier.mts`, etc.):** Implement specific methods for encoding/decoding bits into text features. `stylometric_carrier.genai.mts` provides a suite of techniques based on linguistic patterns.
*   **Encoders (`safety_*.mts`):** Orchestrate the use of carriers to hide data.
    *   `safety_embedded_word.genai.mts`: Zero-width character encoding.
    *   `safety_stylometric_encoder.genai.mts`: Basic stylometric encoding (adverb presence).
    *   `safety_structural_encoder.genai.mts`: Structural encoding (paragraph manipulation).
    *   `safety_enhanced_integration.genai.mts`: Multi-layer integration.
*   **Phonetic Encoding & Debugging (`stylometric_phonetic_encoder_ollama.py`, `stylometric_rich_debugger.py`):** External Python scripts using Ollama for phonetic resonance-based encoding/analysis and interactive debugging.
*   **Analysis & Detection (`carrier_matrix.mts`, `stylometric_detection.genai.mts`, `stylometry_features.genai.mts`):** Tools for analyzing text capacity and detecting stylometric features.
*   **Fingerprinting (`stylometric_fingerprinter.mts`):** Embedding unique identifiers using stylometric carriers.
*   **Fusion (`stylometric_fusion.genai.mts`):** Advanced detection using ML models (requires external embedding model).
*   **Utilities (`stylometric_toolkit.mts`):** Helper functions.
*   **Demos (`demo_*.mts`, `safety_*_demo.genai.mts`):** Example usage scripts.

## Modules

*   [carrier_matrix.ApiNotes.md](./carrier_matrix.ApiNotes.md)
*   [stylometric_carrier.ApiNotes.md](./stylometric_carrier.ApiNotes.md)
*   [stylometric_fingerprinter.ApiNotes.md](./stylometric_fingerprinter.ApiNotes.md)
*   [stylometric_fusion.ApiNotes.md](./stylometric_fusion.ApiNotes.md)
*   [safety_enhanced_integration.ApiNotes.md](./safety_enhanced_integration.ApiNotes.md)
*   [stylometric_phonetic_encoder_ollama.ApiNotes.md](./stylometric_phonetic_encoder_ollama.ApiNotes.md)
*   [stylometric_rich_debugger.ApiNotes.md](./stylometric_rich_debugger.ApiNotes.md) <!-- Added link -->
*   *(Add links as other module/file ApiNotes are created)*

## Core Principles
// ... existing content ...

## Dependencies

*   Node.js (Latest LTS)
*   TypeScript
*   GenAIScript Tooling (`genaiscript` CLI or VS Code Extension)
*   External Python scripts (for specific feature extraction/prediction)
    *   Requires Python 3.x
    *   Python Dependencies (for `stylometric_phonetic_encoder_ollama.py`, `stylometric_rich_debugger.py`): `ollama`, `nltk`, `scikit-learn`, `numpy`, `matplotlib`, `rich` (Install via `pip install ollama nltk scikit-learn numpy matplotlib rich`)
    *   NLTK Data: Requires `cmudict` (`python -m nltk.downloader cmudict`)
*   External Embedding Model (for `stylometric_fusion.genai.mts`)
*   Ollama (for Python scripts) - Requires a running Ollama instance and specified model.
*   Testing Framework (e.g., `vitest`)

## Setup & Execution

*   Install Node dependencies: `npm install`
*   Install Python dependencies (if using Python scripts): `pip install ollama nltk scikit-learn numpy matplotlib rich`
*   Download NLTK data: `python -m nltk.downloader cmudict`
*   Compile TypeScript: `npm run build` (assuming a `tsconfig.json` and build script exist)
*   Run demos (GenAIScript): Use the GenAIScript CLI (`genaiscript run <script_id>`) or VS Code extension. See [demo_runner.ApiNotes.md](./demo_runner.ApiNotes.md).
*   Run Python scripts: `python <script_name.py> --model <ollama_model_name> [options]`
    *   Example Debugger: `python stylometric_rich_debugger.py --model llama3 --n 2 --corpus-limit 3`
*   Run tests: `npm test`

// ... existing content ...