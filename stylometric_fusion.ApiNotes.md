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

# API Notes: Stylometric Fusion Model (`stylometric_fusion.genai.mts`)

**Version:** 1.0
**Date:** 2025-04-15

## Overview

This module implements a stylometric fusion model for detecting AI-generated text, based on the architecture described by Kumarage et al. (2023). It combines traditional stylometric features with language model (LM) embeddings using TensorFlow.js (TF.js).

## Design Goals

*   Implement the two-stage neural network architecture (ReduceNetwork and ClassificationNetwork) using TF.js.
*   Provide a `StyleFusionModel` class encapsulating feature extraction, embedding generation, normalization, and prediction.
*   Include a `trainFusionModel` function for training the network weights on labeled data.
*   Ensure proper tensor management and disposal using `tf.tidy`.
*   Allow for configurable network dimensions and training parameters.

## Architecture & Constraints

*   **Dependencies:**
    *   `@tensorflow/tfjs`: Core library for neural network operations.
    *   `stylometric_detection.genai.mts`: Provides `StyleFeatureExtractor`.
    *   An `EmbeddingModel` implementation (e.g., `MockEmbeddingModel` provided for demo).
*   **Input:** Raw text strings.
*   **Output:** Prediction (`isAiGenerated`, `probability`), extracted features.
*   **Normalization:** Requires calculation of feature mean and standard deviation from a representative training dataset via `setNormalizationParams`. Prediction without normalization is possible but discouraged (warning issued).
*   **Training:** The `trainFusionModel` function handles data preprocessing (feature/embedding extraction, normalization), model compilation, and training loop execution.
*   **Paradigm:** Primarily imperative due to TF.js API, with functional elements for data transformation.

## Key Components

*   **`ReduceNetwork`:** A TF.js `Sequential` model that takes concatenated stylometric features and LM embeddings and reduces their dimensionality.
*   **`ClassificationNetwork`:** A TF.js `Sequential` model that takes the output of the `ReduceNetwork` and performs binary classification (Human vs. AI) using softmax.
*   **`StyleFusionModel`:**
    *   Orchestrates the prediction process.
    *   Manages feature extraction (`StyleFeatureExtractor`), embedding generation (`EmbeddingModel`), normalization parameters, and the two network components.
    *   `predict()`: The main method for getting a prediction for a single text input.
    *   `setNormalizationParams()`: Method to load pre-calculated normalization statistics.
    *   `featureMapToTensor()`: Converts extracted features into a normalized tensor.
*   **`trainFusionModel`:** Standalone function to train the weights of the `ReduceNetwork` and `ClassificationNetwork` within a `StyleFusionModel` instance.

## Happy Path (Prediction)

1.  Instantiate `StyleFusionModel` with an `EmbeddingModel`.
2.  (Optional but recommended) Call `setNormalizationParams` with mean/stdDev calculated from training data.
3.  Call `model.predict("some text")`.
4.  Inside `predict`:
    *   Extract stylometric features using `StyleFeatureExtractor`.
    *   Convert features to a tensor, applying normalization if available (`featureMapToTensor`).
    *   Generate LM embedding using `EmbeddingModel`.
    *   Convert embedding to a tensor.
    *   Concatenate feature and embedding tensors.
    *   Pass concatenated tensor through `ReduceNetwork`.
    *   Pass the result through `ClassificationNetwork`.
    *   Extract the probability of the "AI" class (index 1) from the softmax output.
    *   Return `{ isAiGenerated, probability, features }`.

## Happy Path (Training)

1.  Instantiate `StyleFusionModel` with an `EmbeddingModel`.
2.  Prepare training data: `texts` (string array) and `labels` (number array, 0=human, 1=AI).
3.  Call `trainFusionModel(model, texts, labels, options)`.
4.  Inside `trainFusionModel`:
    *   Pre-extract all features and embeddings for the training set.
    *   Calculate mean and stdDev for features across the training set.
    *   Call `model.setNormalizationParams()` with calculated values.
    *   Normalize all extracted features.
    *   Concatenate normalized features and embeddings.
    *   Create a combined `Sequential` model (Reduce -> Classify).
    *   Compile the combined model (optimizer, loss, metrics).
    *   Call `combinedModel.fit()` with the prepared tensors and labels.
    *   Return training history.

## Regeneration Notes

*   The core logic follows the described architecture.
*   Network layer configurations (units, activation, dropout) can be adjusted.
*   Normalization is crucial for good performance.
*   Error handling and tensor disposal are important (primarily handled by `tf.tidy`).