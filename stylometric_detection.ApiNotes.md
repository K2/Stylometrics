# Stylometric Detection Module

## Module Overview
This module implements stylometric detection techniques for identifying AI-generated text based on research by Kumarage et al. (2023). Unlike steganographic approaches, it analyzes natural language patterns to determine text authorship and detect potential transitions between human and AI authors.

## Design Philosophy
The stylometric detection approach prioritizes three key capabilities:
1. Binary classification of text as human or AI-generated
2. Change point detection in text timelines to identify author transitions
3. Fusion of stylometric features with language model embeddings for enhanced accuracy

## Core Components

### Feature Extraction Engine
- Extracts three categories of stylometric features: Phraseology, Punctuation, and Linguistic Diversity
- Implements Moving Average Type-Token Ratio (MATTR) for lexical richness measurement
- Calculates readability metrics and structural patterns

### Fusion Architecture
- Combines stylometric features with language model embeddings
- Employs a reduce network to create joint representations
- Applies classification network for final detection

### Change Point Detection
- Implements StyloCPA (Stylometric Change Point Agreement) methodology
- Analyzes feature time series for statistically significant change points
- Measures agreement across multiple features to identify author transitions

## Data Flow

1. **Feature Extraction Process**:
   - Analyze text structure and extract quantitative stylometric features
   - Process text through three feature categories (Phraseology, Punctuation, Linguistic Diversity)
   - Normalize and prepare features for analysis

2. **Classification Process**:
   - Extract stylometric features
   - Optionally combine with language model embeddings
   - Apply classification algorithm to determine authorship

3. **Change Point Detection Process**:
   - Extract feature time series from text timeline
   - Apply PELT algorithm to identify change points in each feature
   - Measure agreement between detected change points
   - Report change points with sufficient agreement as author transitions

## Technical Details

### Feature Categories:
- **Phraseology Features**:
  - Word count, sentence count, paragraph count
  - Mean/std of words per sentence and paragraph
  - Sentence structure complexity metrics

- **Punctuation Features**:
  - Total punctuation count
  - Frequency of specific punctuation marks
  - Punctuation density and patterns

- **Linguistic Diversity Features**:
  - Moving-Average Type-Token Ratio (MATTR)
  - Readability scores
  - Vocabulary richness metrics

### Classification Architecture:
```
[Stylometric Features] --> [Feature Processing] --\
                                                   +--> [Reduce Network] --> [Classification Network] --> [Prediction]
[Language Model Embedding] --> [Embedding] -------/
```

### Change Point Detection Format:
```
[Timeline] --> [Feature Extraction] --> [Time Series Analysis] --> [Change Point Detection] --> [Agreement Measurement] --> [Author Transition Points]
```

## Usage Notes

- Detection is most effective with sufficient text (multiple sentences)
- Performance improves when combining stylometric features with language model embeddings
- Change point detection requires a time series of texts from the same presumed author
- Best applied to informal content like social media posts where AI generation is common

## Limitations and Constraints

- Performance varies based on text length and domain
- Requires sufficient text for reliable feature extraction
- May need retraining for different languages or specialized domains
- Should be used as part of a broader detection strategy rather than in isolation

## References

- Kumarage, T., Garland, J., Bhattacharjee, A., Trapeznikov, K., Ruston, S., & Liu, H. (2023). Stylometric Detection of AI-Generated Text in Twitter Timelines.
- Killick, R., et al. (2012). Optimal detection of changepoints with a linear computational cost. Journal of the American Statistical Association.
- Covington, M.A., McFall, J.D. (2010). Cutting the gordian knot: The moving-average type–token ratio (mattr). Journal of quantitative linguistics.