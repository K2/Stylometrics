# Project-Level ApiNotes for Stylometrics

**Version:** 0.1.0 (Initial Generation: 2025-04-15)

**Project Goal:** To develop a robust framework for embedding persistent, verifiable information within digital text content using multi-layered steganographic techniques based on stylometry, structure, and zero-width characters. This framework aims to support Digital Rights Management (DRM), Data Loss Prevention (DLP), and content provenance tracking, even when content undergoes significant transformation.

**Core Architectural Principles:**

1.  **Layered Security:** Employ multiple encoding techniques (zero-width, stylometric, structural) with varying trade-offs in capacity, resilience, and detectability.
2.  **Modularity:** Encapsulate specific encoding/decoding techniques and analysis tools into distinct modules (`.genai.mts` scripts).
3.  **Verification:** Integrate cryptographic signing and hashing to ensure the authenticity and integrity of embedded metadata and the content itself.
4.  **Resilience:** Utilize erasure coding to allow recovery of embedded data even if parts of the carrier text are corrupted or modified.
5.  **Configuration:** Allow flexible combination and configuration of encoding layers and techniques.
6.  **Detection Awareness:** Incorporate stylometric analysis both for embedding (to minimize detection) and for detecting potential manipulation or AI generation.

**Key Modules & Interactions:**

*   **`safety_embedded_word.genai.mts`:** High-capacity zero-width encoding with cryptographic signing. (Foundation Layer)
*   **`stylometric_carrier.genai.mts`:** Implements individual stylometric feature manipulation techniques (e.g., sentence length, punctuation). Acts as a toolkit. [paradigm:functional] for analysis, [paradigm:imperative] for text manipulation.
*   **`safety_stylometric_encoder.genai.mts`:** Uses `stylometric_carrier` to embed data via linguistic style changes. (Resilience Layer 1)
*   **`safety_structural_encoder.genai.mts`:** Embeds data via high-level document structure changes (POV, tense, paragraph patterns). (Resilience Layer 2 - Highest)
*   **`stylometric_fusion.genai.mts`:** Orchestrates multiple stylometric carriers from `stylometric_carrier` for increased capacity/resilience. Also includes a TF.js model for *detecting* AI text using fused features. [paradigm:imperative] for TF.js.
*   **`stylometric_detection.genai.mts`:** Extracts stylometric features and implements detection algorithms (AI classification, change point detection). [paradigm:imperative] for analysis loops.
*   **Integration/Demo Scripts (`safety_*_demo.genai.mts`, `safety_enhanced_integration.genai.mts`):** Combine layers to demonstrate end-to-end workflows.

**LLM Regeneration Context:**

*   This project uses GenAIScript (`.genai.mts`) for orchestration and core logic.
*   TypeScript is the primary language.
*   Dependencies include `compromise` for NLP and `@tensorflow/tfjs` for ML.
*   Focus on implementing the specific encoding/decoding logic described in each module's documentation and comments.
*   Ensure cryptographic signing and erasure coding are integrated where specified.
*   Maintain clear separation between encoding layers and analysis tools.

**Constraints:**

*   Implementations should be functional but may not be perfectly optimized for naturalness or computational efficiency initially.
*   Synonym substitution requires a basic internal list or reliance on `compromise`.
*   Advanced NLP tasks (voice change, complex comma rules) will use simplified heuristics.
*   TensorFlow.js models are basic feedforward networks as described. Training requires sample data (provided in demos or assumed).