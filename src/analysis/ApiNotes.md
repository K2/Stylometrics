# Text Analysis Module ApiNotes

## Overview
This module provides stylometric analysis capabilities for the Stylometrics project. It focuses on extracting and analyzing textual features that can be used for author profiling, content classification, and supporting the counterfactual embedding system.

## Key Components

### TextAnalyzer
The primary class for analyzing text and generating author profiles. This analyzer extracts stylometric features including:

- Sentence structure patterns
- Vocabulary richness and distribution
- Grammatical constructions
- Error rates and patterns
- Syntactic preferences

## Design Goals
1. Maintain high accuracy in author profiling
2. Support embedding capacity calculations
3. Provide statistical baselines for determining "natural variation" ranges
4. Enable verification of embedded content vs. author style

## Module Relationships
- Provides data to `CounterfactualEmbedder` for determining embedding capacity
- Works with `AuthorProfile` models to store and retrieve stylometric data
- May leverage the `stylometric_detection` module for feature extraction

## Constraints
- Analysis should be deterministic for the same input text
- Feature extraction must be statistically significant
- Error budgets should be calculated conservatively to preserve text naturalness

## Usage Example
```typescript
const analyzer = new TextAnalyzer();
const profile = analyzer.generateAuthorProfile(text);
// Use profile for embedding or verification
```
