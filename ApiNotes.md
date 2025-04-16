# Project: Stylometrics - ApiNotes

## Overview

This project implements various steganographic techniques focusing on stylometry (linguistic patterns), zero-width characters, and structural text modifications. The goal is to embed data within text robustly against transformations like paraphrasing and translation.

## Architecture

*   **Carriers (`stylometric_carrier.genai.mts`, `quote_style_carrier.mts`, etc.):** Implement specific methods for encoding/decoding bits into text features.
*   **Encoders (`safety_*.mts`):** Orchestrate the use of carriers to hide data.
    *   `safety_embedded_word.genai.mts`: Zero-width character encoding.
    *   `safety_stylometric_encoder.genai.mts`: Basic stylometric encoding (adverb presence).
    *   `safety_structural_encoder.genai.mts`: Structural encoding (paragraph manipulation).
    *   `safety_enhanced_integration.genai.mts`: Multi-layer integration.
*   **Analysis & Detection (`carrier_matrix.mts`, `stylometric_detection.genai.mts`, `stylometry_features.genai.mts`):** Tools for analyzing text capacity and detecting stylometric features.
*   **Fingerprinting (`stylometric_fingerprinter.mts`):** Embedding unique identifiers using stylometric carriers.
*   **Fusion (`stylometric_fusion.genai.mts`):** Advanced detection using ML models (requires external embedding model).
*   **Utilities (`stylometric_toolkit.mts`):** Helper functions.
*   **Demos (`demo_*.mts`, `safety_*_demo.genai.mts`):** Example usage scripts.

## Modules

*   [capacity_matrix/ApiNotes.md](./capacity_matrix/ApiNotes.md)
*   [stylometric_carrier.ApiNotes.md](./stylometric_carrier.ApiNotes.md)
*   [stylometric_fingerprinter.ApiNotes.md](./stylometric_fingerprinter.ApiNotes.md)
*   [stylometric_fusion.ApiNotes.md](./stylometric_fusion.ApiNotes.md)
*   [safety_enhanced_integration.ApiNotes.md](./safety_enhanced_integration.ApiNotes.md)
*   *(Add links as other module/file ApiNotes are created)*

## Core Principles

*   **Modularity:** Carriers and encoders should be distinct.
*   **Robustness:** Encoding should survive common text transformations where possible.
*   **Testability:** All encoding/decoding pairs must be rigorously tested.
*   **Documentation:** ApiNotes must be maintained at all levels.
*   **Paradigm:** Primarily imperative for text manipulation, functional for data transformation.

## Dependencies

*   Node.js (Latest LTS)
*   TypeScript
*   External Python scripts (for specific feature extraction/prediction)
*   External Embedding Model (for `stylometric_fusion.genai.mts`)
*   Testing Framework (e.g., `vitest`)

## Setup & Execution

*   Install dependencies: `npm install`
*   Compile TypeScript: `npm run build` (assuming a `tsconfig.json` and build script exist)
*   Run demos: `node dist/demo_runner.genai.js` (adjust path as needed)
*   Run tests: `npm test`