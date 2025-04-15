# Capacity Matrix Module: ApiNotes

## Module Overview
The Capacity Matrix module provides advanced analysis and optimization of steganographic carrying capacity across different stylometric dimensions. It uses weighted matrices to represent the embedding potential of text, allowing for adaptive encoding strategies that respect the natural characteristics of the original content.

## Design Philosophy
This module extends the core stylometric carrier approach with sophisticated mathematical models to:
1. Maximize carrying capacity while minimizing textual changes
2. Distribute payloads optimally across different carrier techniques
3. Create structured redundancy for resilient information recovery
4. Adapt to document structure and versioning opportunities

## Core Components

### Capacity Matrix Analysis
- Multi-dimensional analysis of text segments
- Weighted evaluation of carrying capacity across techniques
- Optimization algorithms for payload distribution
- Contextual adaptation to text characteristics

### Content-Aware Embedding
- Genre and style-specific carrier weighting
- Document structure exploitation (chapters, sections)
- Metadata utilization for additional capacity
- Version-based capacity analysis for evolving documents

### Erasure Coding & Redundancy
- Error correction through distributed encoding
- Parity systems for partial recovery
- Slice-based recovery from document fragments
- Redundancy optimization based on recovery requirements

## Interfaces

### Matrix Analysis API
```typescript
analyzeCapacityMatrix(
  text: string, 
  options?: CapacityAnalysisOptions
): CapacityMatrix;

optimizeEmbedding(
  matrix: CapacityMatrix,
  payload: Uint8Array,
  options?: OptimizationOptions
): EmbeddingPlan;
```

### Content Structure API
```typescript
analyzeDocumentStructure(
  text: string
): DocumentStructure;

identifyEmbeddingOpportunities(
  structure: DocumentStructure
): OpportunityMap;
```

### Erasure Coding API
```typescript
createRedundantEncoding(
  payload: Uint8Array, 
  redundancyLevel: number
): RedundantPayload;

recoverFromFragments(
  fragments: PayloadFragment[], 
  requiredFragments: number
): Uint8Array | null;
```

## Mathematical Models

### Carrying Capacity Calculation
The weighted capacity matrix M represents the embedding potential across text segments and carrier techniques:

M[i,j] = w_j * c(s_i, t_j)

Where:
- s_i is text segment i
- t_j is carrier technique j
- c(s,t) is the base capacity function for technique t on segment s
- w_j is the weight assigned to technique j (based on detectability, robustness, etc.)

### Optimization Problem
Given payload P of size |P| bits, find optimal distribution D[i,j] that minimizes:

min sum_{i,j} (D[i,j] * d(s_i, t_j))

Subject to:
- sum_{i,j} D[i,j] >= |P| (sufficient capacity)
- D[i,j] <= M[i,j] for all i,j (capacity constraints)
- Additional constraints for natural text preservation

Where d(s,t) is a distortion function measuring impact of technique t on segment s.

### Erasure Coding Model
For a payload divided into n fragments with k required for recovery:

- Original payload: P
- Encoded fragments: F = [f_1, f_2, ..., f_n]
- Recovery property: any k fragments from F can reconstruct P
- Optimal k for balancing redundancy and recoverability

## Implementation Strategies

### Text Segmentation Approaches
- Natural boundaries (paragraphs, sentences)
- Equal-sized chunks for uniform distribution
- Feature-based segmentation for optimal carrier selection
- Hierarchical segmentation for multi-level embedding

### Carrier Selection Heuristics
- Detectability-based prioritization
- Text-specific carrier affinity
- Capacity-maximizing greedy algorithms
- Constraint-based optimization

### Document Structure Exploitation
- Special treatment for headings, footnotes, quotes
- Utilizing front/back matter in books
- Version comparison for change-tolerant encoding
- Metadata embedding in document properties

## Integration with Core System

This module extends the base Stylometric Carrier system by:
1. Providing enhanced analysis prior to embedding
2. Suggesting optimal carrier technique allocation
3. Adding resilience through redundancy encoding
4. Exploiting document structure for capacity gains

## Usage Patterns

1. **Analysis Phase**: Generate capacity matrix for text
2. **Planning Phase**: Optimize embedding across carriers
3. **Encoding Phase**: Apply the embedding plan to text
4. **Verification Phase**: Validate recoverability and naturality

## References

- Fridrich, J. (2009). Steganography in Digital Media: Principles, Algorithms, and Applications.
- Reed, I.S. & Solomon, G. (1960). Polynomial Codes over Certain Finite Fields.
- Kumarage, T., et al. (2023). Stylometric Detection of AI-Generated Text.
- Bennett, K. (2004). Linguistic Steganography: Survey, Analysis, and Robustness Concerns for Hiding Information in Text.