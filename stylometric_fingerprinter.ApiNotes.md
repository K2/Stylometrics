# Stylometric Fingerprinter Module (`stylometric_fingerprinter.genai.mts`)

## Module Overview

This GenAIScript module performs a multi-step process to embed metadata (like a cryptographic fingerprint) into text while attempting to mask the AI's authorship. It leverages an LLM for both stylistic analysis and text generation, followed by a conceptual stylometric embedding phase.

1.  **Stylometric Analysis**: Analyzes an input text (`originalText`) to determine its stylistic profile across phraseology, punctuation, and linguistic diversity dimensions.
2.  **Mimetic Generation**: Instructs the LLM to generate *new* text content that closely matches the analyzed stylistic profile, aiming for statistical indistinguishability from the original author's style.
3.  **Stylometric Embedding**: Takes the AI-generated text and embeds provided `metadataToEmbed` (after erasure coding) using multiple conceptual stylometric carrier techniques (phraseology, punctuation, linguistic diversity).

## Design Philosophy

-   **Mimicry for Plausible Deniability**: By generating text that statistically resembles a target style, the script aims to make it harder to identify the generated text as AI-authored using common stylometric detectors.
-   **Multi-Carrier Embedding**: Distributes the embedded data across different linguistic features (sentence structure, punctuation choices, word choices) to increase resilience against detection and partial data loss. This follows the principles outlined in `stylometric_carrier.ApiNotes.md` and `stylometric_fusion.ApiNotes.md`.
-   **Functional Transformation**: The script treats the process as a transformation from (original text + metadata) to (stylistically similar text with embedded metadata).
-   **LLM-Driven Core**: Relies heavily on the LLM's ability to understand and replicate complex stylistic patterns.
-   **Post-Generation Embedding**: Separates the text generation (LLM task) from the data embedding (algorithmic/rule-based task, handled in `defOutputProcessor`).

## Inputs

-   `originalText` (string): The text whose style serves as the target profile.
-   `metadataToEmbed` (object): A JSON object containing data to embed (e.g., `{ "fingerprintHash": "...", "timestamp": "..." }`).
-   `targetLength` (integer, optional): Approximate desired word count for the generated text (default: 500).

## Outputs

-   A string containing the newly generated text with the metadata conceptually embedded via stylometric modifications.
-   Annotations (warnings/errors) if generation or embedding encounters issues (e.g., insufficient capacity).

## Core Steps (Internal)

1.  **Prompt Construction**: Builds a detailed prompt instructing the LLM to perform Phase 1 (Analysis) and Phase 2 (Generation).
2.  **LLM Execution**: Runs the prompt using `runPrompt`.
3.  **Output Processing (`defOutputProcessor`)**:
    -   Retrieves the generated text.
    -   Applies conceptual erasure coding to the metadata string.
    -   Converts the coded metadata to a binary sequence.
    -   **Conceptually** applies stylometric modifications to the generated text to embed the binary sequence, using carriers related to:
        -   Phraseology (e.g., sentence length patterns)
        -   Punctuation (e.g., quote style, optional commas)
        -   Linguistic Diversity (e.g., synonym substitution, TTR modulation)
    -   Returns the final modified text.

## Limitations and Considerations

-   **LLM Capability**: The success heavily depends on the LLM's ability to accurately analyze and mimic nuanced stylistic features. This is a challenging task.
-   **Embedding Complexity**: The actual implementation of stylometric carriers (`StylometricCarrier` or similar) is complex and not fully implemented within this script (it's simulated). Real-world embedding requires careful algorithms to modify text subtly without corrupting meaning or introducing obvious artifacts.
-   **Capacity vs. Detectability**: Embedding data inevitably alters the text. There's a trade-off between the amount of data embedded (capacity) and the subtlety of the changes (detectability/naturalness). The conceptual embedding here doesn't precisely model this trade-off.
-   **Erasure Coding**: Relies on an external or helper function for erasure coding.
-   **Verification**: This script only handles encoding. A corresponding decoding script would be needed to extract the embedded metadata.
-   **Functional Purity**: While the script structure is functional, the reliance on an LLM and the complexity of stateful text modification during embedding mean true functional purity is difficult.

## Usage Example (Conceptual `gps` call)

```javascript
const result = await runPrompt("stylometric_fingerprinter", {
  vars: {
    originalText: "The quick brown fox jumps over the lazy dog. It was a bright, sunny day.",
    metadataToEmbed: {
      fingerprintHash: "a1b2c3d4e5f6...",
      timestamp: "2024-01-15T10:00:00Z",
      sourceId: "doc-123"
    },
    targetLength: 150
  }
})

console.log(result.text) // Output: Newly generated text mimicking the style, with embedded data.
```

## Stylometric Carriers

### Overview

The stylometric carriers are methods used to embed binary data into text features. These carriers are implemented in the `StylometricCarrier` class and provide diverse, testable methods for capacity estimation, application, and extraction logic.

### Available Carriers

1. **Synonym Choice Carrier**: Encodes bits by choosing synonyms based on payload.
2. **Punctuation Style Carrier**: Encodes bits using the presence or absence of the Oxford comma.
3. **Whitespace Style Carrier**: Encodes bits using single/double spaces after periods.
4. **Voice Style Carrier**: Encodes bits by subtly shifting narrative voice (requires advanced NLP, not fully implemented).
5. **Description Detail Carrier**: Encodes bits by adding/removing descriptive words (requires advanced NLP, not fully implemented).
6. **Counterpoint Phrase Carrier**: Encodes bits by inserting/removing counterpoint phrases.

### Implementation Notes

- **Synonym Choice Carrier**: Uses a basic thesaurus structure to encode bits by substituting synonyms.
- **Punctuation Style Carrier**: Modifies text based on the presence or absence of the Oxford comma in lists.
- **Whitespace Style Carrier**: Encodes bits using single or double spaces after periods.
- **Voice Style Carrier**: Requires advanced NLP for passive/active voice detection and transformation.
- **Description Detail Carrier**: Requires NLP for adjective/adverb identification and modification.
- **Counterpoint Phrase Carrier**: Inserts or removes counterpoint phrases in sentences.

### Limitations

- Some carriers require advanced NLP capabilities and are not fully implemented.
- Extraction logic for certain carriers is unreliable and requires further development.

### Example Usage

```javascript
const carrier = new StylometricCarrier({ carriers: ['synonym-choice', 'punctuation-style'] });
const text = "The quick brown fox jumps over the lazy dog. It was a happy day.";
const payload = Buffer.from("test");
const { modifiedText } = await carrier.encodePayload(text, payload);
const extractedPayload = await carrier.extractPayload(modifiedText);
console.log(extractedPayload); // Output: Original payload if extraction is successful.
```