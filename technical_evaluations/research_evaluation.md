# Technical Evaluation: Stylometric Detection Techniques

## Overview

This document evaluates the research paper "Stylometric Detection of AI-Generated Text in Twitter Timelines" for techniques that could enhance our steganographic system. The paper demonstrates how stylometric features can be used to detect AI-generated content and identify authorship change points, which provides valuable insights for both strengthening our encoding methods and potentially developing countermeasures against detection.

## Key Findings from Research

1. **Effectiveness of Stylometric Features** - The research demonstrates that stylometric features significantly improve AI-generated text detection, especially when combined with language model embeddings. These features remain effective even with limited semantic information.

2. **Three Categories of Stylometric Features**:
   - **Phraseology** - Features quantifying how authors organize words and phrases (word count, sentence count, paragraph count, mean/standard deviation metrics)
   - **Punctuation** - Features measuring punctuation usage (total count, frequency of specific punctuation marks)
   - **Linguistic Diversity** - Features analyzing lexical richness and readability

3. **Change Point Detection** - The Pruned Exact Linear Time (PELT) algorithm effectively identifies points in a text stream where authorship changes from human to AI.

4. **Feature Importance** - Punctuation and phraseology features proved more important than linguistic diversity features, especially for shorter text samples.

5. **Generator Size Impact** - Larger language models produce content that is slightly harder to detect, though the difference is not dramatic.

## Applicable Techniques for Our Steganographic System

### 1. Enhanced Plausible Deniability via Stylometric Matching

The paper's stylometric analysis framework could be inverted to create a "stylometric normalization" component in our encoding system. This would:

- Analyze the stylometric signature of the original text
- Ensure all modified text maintains the same signature after encoding
- Adapt encoding techniques based on which features are least likely to trigger detection

**Implementation Approach:**
```typescript
function stylometricNormalization(
  originalText: string, 
  encodedText: string, 
  targetFeatures: StyleFeature[] = ['punctuation', 'phraseology']
): string {
  const originalMetrics = extractStylometricFeatures(originalText);
  let normalizedText = encodedText;
  
  // Iterative refinement to match original metrics
  while (!areStyleMetricsWithinThreshold(
    extractStylometricFeatures(normalizedText),
    originalMetrics,
    DETECTION_EVASION_THRESHOLD
  )) {
    normalizedText = adjustTextFeatures(normalizedText, originalMetrics, targetFeatures);
  }
  
  return normalizedText;
}
```

### 2. Adversarial Stylometric Techniques

We could incorporate adversarial stylometric techniques to make our encoded content more resistant to detection:

- Deliberately balance punctuation usage to match human norms
- Maintain consistent phraseology metrics before and after encoding
- Target the specific features identified as most important in the research (punctuation patterns and phraseology)

**Implementation Approach:**
```typescript
export interface StylemetricNormalizationOptions {
  prioritizeFeatures: StyleFeature[];
  matchReadability: boolean;
  preservePunctuation: boolean;
  balancePhraseology: boolean;
}

function applyAdversarialStylometry(
  content: string, 
  metadata: any, 
  options: StylemetricNormalizationOptions
): string {
  // First, apply standard encoding
  let encodedContent = hideDataStructurally(content, metadata);
  
  // Then apply adversarial normalization
  if (options.preservePunctuation) {
    encodedContent = normalizePunctuation(content, encodedContent);
  }
  
  if (options.balancePhraseology) {
    encodedContent = balancePhraseology(content, encodedContent);
  }
  
  if (options.matchReadability) {
    encodedContent = matchReadabilityScores(content, encodedContent);
  }
  
  return encodedContent;
}
```

### 3. Change Point Detection for Self-Verification

Implementing change point detection for our own verification system would:

- Allow the sender to check if their encoded content has noticeable stylometric "jumps"
- Provide a quality control mechanism to ensure encodings remain undetectable
- Help calibrate the aggressiveness of encoding based on detection risk

**Implementation Approach:**
```typescript
function verifyStylometricContinuity(
  encodedText: string, 
  segmentSize: number = 100
): VerificationResult {
  const segments = splitIntoSegments(encodedText, segmentSize);
  const stylometricFeatureMatrix = segments.map(extractStylometricFeatures);
  
  // Use PELT algorithm to detect change points
  const changePoints = detectChangePoints(stylometricFeatureMatrix);
  
  return {
    isConsistent: changePoints.length === 0,
    changePointLocations: changePoints,
    riskScore: calculateRiskScore(changePoints, stylometricFeatureMatrix)
  };
}
```

### 4. Feature-Specific Encoding Strategies

Based on the paper's finding that different feature categories have different detection importance, we could implement feature-specific encoding strategies:

- Prioritize maintaining consistent punctuation patterns (high detection importance)
- Allow more flexibility in linguistic diversity features (lower detection importance)
- Implement different encoding strategies based on content length

**Implementation Approach:**
```typescript
function selectEncodingStrategy(content: string): EncodingStrategy {
  const contentLength = content.length;
  
  if (contentLength < 500) {
    // For short content, focus heavily on preserving punctuation patterns
    return {
      enabledLayers: ['zero-width', 'stylometric'],
      stylometricPriorities: ['punctuation', 'phraseology'],
      structuralEnabled: false
    };
  } else if (contentLength < 2000) {
    // For medium content, balanced approach
    return {
      enabledLayers: ['zero-width', 'stylometric', 'structural'],
      stylometricPriorities: ['punctuation', 'phraseology', 'linguistic'],
      structuralEnabled: true,
      structuralStrength: 'moderate'
    };
  } else {
    // For long content, can use more structural encoding
    return {
      enabledLayers: ['zero-width', 'stylometric', 'structural'],
      stylometricPriorities: ['phraseology', 'linguistic', 'punctuation'],
      structuralEnabled: true,
      structuralStrength: 'aggressive'
    };
  }
}
```

### 5. Integrated Stylometric Analysis Components

We could add components to systematically analyze and track stylometric features:

- **StyleFeatureExtractor**: Extract all relevant stylometric features from text
- **StyleNormalizer**: Adjust encoded text to match original stylometric signature
- **StyleDetectionRiskCalculator**: Assess how likely the encoding is to be detected

These could be integrated into the `safety_enhanced_integration.genai.mts` module to provide advanced evasion capabilities.

## Risk Assessment

The paper also provides valuable information on detection risks:

1. **Generator Size Impact**: Larger language models (like GPT-3) produce content that is harder to detect, suggesting we can potentially use these models to help generate steganographically encoded content that's more resistant to detection

2. **Resilience to Topic Changes**: The paper found stylometric detection worked across different topics, suggesting our encodings should focus on deeper structural patterns rather than topic-specific features

3. **Linguistic Feature Limitations**: For shorter texts, linguistic diversity features were less important, suggesting that for short encoded content, we should prioritize maintaining consistent punctuation and phraseology patterns

## Recommended Implementation Path

1. **Phase 1**: Implement the StyleFeatureExtractor to analyze all three categories of stylometric features
2. **Phase 2**: Develop the StyleNormalizer to ensure encoded content maintains the original stylometric signature
3. **Phase 3**: Create the change point self-verification system to check encoding quality
4. **Phase 4**: Implement adaptive encoding strategies based on content length and stylometric detection risk

By incorporating these techniques derived from the research paper, our steganographic system can become significantly more resistant to stylometric detection methods while maintaining its primary function of preserving metadata through content transformations.

## References

- Kumarage, T., et al.: Stylometric Detection of AI-Generated Text in Twitter Timelines
- Killick, R., et al.: Optimal detection of changepoints with a linear computational cost
- Gómez-Adorno, H., et al.: Stylometry-based approach for detecting writing style changes in literary texts