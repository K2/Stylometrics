# Project: Stylometrics - Resilient Content Fingerprinting

## Overview

This project implements advanced steganographic techniques for embedding robust, verifiable fingerprints and metadata within text content. It focuses on resilience against common transformations (copy-paste, reformatting, AI paraphrasing) and leverages cryptographic signing and erasure coding.

## Core Modules & Design

1.  **GenAIScripts (`*.genai.mts`)**: High-level scripts orchestrating encoding/decoding flows, often using LLMs for tasks like style analysis or generation.
    *   `safety_*.genai.mts`: Implement specific encoding layers (Zero-Width, Stylometric, Structural) and integrations.
    *   `stylometric_*.genai.mts`: Focus on stylometric analysis, detection, and carrier/fusion logic.
    *   `*_demo.genai.mts`: Provide demonstrations and visualizations.
    *   **`stylometric_fingerprinter.genai.mts`**: Analyzes style, generates mimicking text, and embeds data using stylometry.

2.  **Core Logic (`src/`)**: TypeScript modules providing the underlying mechanisms.
    *   **`src/types/`**: Defines core interfaces (`Carrier`, `DocumentSegment`, `EncodingMetadata`, `AnalysisResult`, etc.).
        *   `ApiNotes.md`: Describes the purpose of shared type definitions.
    *   **`src/utils/`**: Utility functions and classes.
        *   `ApiNotes.md`: Describes utility components.
        *   `StyleFeatureExtractor.ts`: (Placeholder) Extracts linguistic features.
        *   `BitUtils.ts`: Handles bit/byte conversions.
    *   **`src/carriers/`**: Concrete implementations of the `Carrier` interface.
        *   `ApiNotes.md`: Describes the role of carrier implementations.
        *   `BaseCarrierImpl.ts`: Base class with common utilities.
        *   `PunctuationCarrier.ts`, `WhitespaceCarrier.ts`, `SynonymCarrier.ts`: (Placeholders) Specific embedding techniques.
    *   **`src/matrix/`**: Implements the structure-aware, resilient encoding system.
        *   `ApiNotes.md`: Describes the matrix module's design.
        *   `CarrierMatrix.ts`: Central orchestrator for analysis, planning, encoding, and decoding using multiple carriers and erasure coding.
        *   `ErrorCorrection.ts`: `ReedSolomon` class (Placeholder) for erasure coding.

## Key Concepts

*   **Layered Steganography**: Combining Zero-Width (high capacity, fragile), Stylometric (medium capacity, moderate resilience), and Structural (low capacity, high resilience) techniques.
*   **Cryptographic Fingerprinting**: Using digital signatures (e.g., ECDSA) to verify origin and integrity.
*   **Erasure Coding**: Using techniques like Reed-Solomon to allow data recovery even if parts of the embedded information are lost or corrupted.
*   **Carrier Matrix**: A system for analyzing document structure and carrier capacities to optimally distribute erasure-coded data across multiple steganographic techniques for maximum resilience and capacity.
*   **Stylometric Analysis**: Quantifying linguistic style for detection, mimicry, and weighting embedding techniques.

## Design Philosophy

*   **Resilience First**: Prioritize the ability of embedded data to survive transformations.
*   **Modularity**: Separate concerns into distinct carriers, utilities, and orchestration logic.
*   **Configurability**: Allow tuning of parameters like redundancy levels and carrier weights.
*   **Metadata-Driven Recovery**: Embed necessary metadata during encoding to enable robust decoding.
*   **Plausible Deniability**: Aim for embedding techniques that minimize statistical detectability.

## Getting Started / Usage

*   Use GenAIScripts (e.g., `safety_encoder_demo.genai.mts`, `stylometric_fingerprinter.genai.mts`) for high-level encoding/decoding tasks.
*   The `CarrierMatrix` class provides the core engine for advanced, resilient encoding. Instantiate it, analyze a document, and then use `encodePayload` and `decodePayload`.

## Future Work / Placeholders

*   Implement functional `StyleFeatureExtractor` using NLP libraries.
*   Develop robust, non-placeholder `Carrier` implementations for various techniques.
*   Implement a functional `ReedSolomon` class or integrate a library.
*   Refine document segmentation logic.
*   Add comprehensive test suites.
