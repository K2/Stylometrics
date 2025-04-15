# Stylometric Fusion Model

## Module Overview
This module implements a fusion architecture that combines stylometric features with language model embeddings to improve AI-generated text detection. It is based on the approach described by Kumarage et al. (2023) where both feature types provide complementary signals for detection.

## Design Philosophy
The fusion architecture leverages the strengths of both feature types:
1. Stylometric features capture writing style, consistency, and structural patterns
2. Language model embeddings encode semantic content and linguistic knowledge
3. Combined, they provide more robust detection, especially for short texts

## Core Components

### Feature Extraction Pipeline
- Extracts stylometric features from raw text
- Interfaces with language models to obtain embeddings
- Handles data normalization and preparation

### Neural Fusion Architecture
- Reduce Network: Creates unified representation from combined features
- Classification Network: Determines text authorship from fused representation
- Training Pipeline: Optimizes model parameters for detection

## Data Flow

1. **Input Processing**:
   - Raw text processing and normalization
   - Stylometric feature extraction
   - Language model embedding generation

2. **Feature Fusion**:
   - Feature standardization and alignment
   - Concatenation of stylometric features and language model embeddings
   - Dimensionality reduction via neural network

3. **Classification Process**:
   - Process fused representation through classification network
   - Generate confidence scores and binary classification
   - Apply threshold for final prediction

## Technical Details

### Input Format:
- Text of arbitrary length (optimized for short texts like tweets)
- Pre-trained language model for embedding extraction
- Configuration parameters for feature selection

### Architecture Diagram:
```
                                  ┌─────────────────┐
                                  │                 │
                                  │ Classification  │
                                  │    Network      │
                                  │                 │
                                  └────────┬────────┘
                                           │
                                           │
                                  ┌────────▼────────┐
                                  │                 │
                                  │ Reduce Network  │
                                  │                 │
                                  └────────┬────────┘
                                           │
                           ┌───────────────┴──────────────┐
                           │                              │
               ┌───────────▼────────────┐   ┌─────────────▼────────────┐
               │                        │   │                          │
               │  Stylometric Features  │   │  Language Model Embeddings│
               │                        │   │                          │
               └────────────────────────┘   └──────────────────────────┘
```

## Integration Points

- **Language Models**: Compatible with any LM that provides sentence embeddings
- **Stylometric Analysis**: Uses features from the StyleFeatureExtractor
- **Inference Pipeline**: Easily integrates with existing text classification systems

## Usage Notes

- Requires a language model for embedding generation
- Performs best when fine-tuned on domain-specific data
- Can be used for both binary classification and confidence scoring
- Supports batch processing for efficiency

## Limitations and Constraints

- Performance dependent on language model quality
- Training requires paired examples of human and AI-generated text
- May have biases from underlying language models
- Requires more computational resources than stylometric analysis alone

## References

- Kumarage, T., Garland, J., Bhattacharjee, A., Trapeznikov, K., Ruston, S., & Liu, H. (2023). Stylometric Detection of AI-Generated Text in Twitter Timelines.
- Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
- Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach.