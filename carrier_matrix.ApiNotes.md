# Carrier Matrix and Resilient Encoding System

## Module Overview
This module extends the stylometric carrier framework with advanced concepts:
1. A weighted capacity matrix for optimal carrier allocation
2. Document structure-aware encoding that leverages chapter/section patterns
3. Parity/erasure coding techniques for resilient payload recovery
4. Versioning-aware strategies that exploit natural document revisions

## Design Philosophy

The carrier matrix approach recognizes that different document types and structures offer variable carrying capacities across different stylometric dimensions. By modeling these capacities explicitly and applying weighted distribution algorithms, we can:

1. Maximize total carrying capacity while minimizing detectability
2. Adapt encoding strategies to document-specific characteristics
3. Distribute payload across multiple dimensions with optimal redundancy
4. Leverage natural document structures for enhanced steganographic hiding

## Core Components

### Carrier Capacity Matrix
- Represents carrying capacity across multiple dimensions and document segments
- Each cell contains capacity metrics, detectability ratings, and robustness scores
- Weighted by document characteristics and structural elements
- Dynamically adjusted based on document analysis

### Structure-Aware Encoding
- Leverages document organization (chapters, headings, metadata)
- Distributes payload across structural elements based on capacity profile
- Uses predictable patterns in document structure as encoding anchors
- Exploits structural redundancies for enhanced robustness

### Parity and Erasure Coding
- Implements Reed-Solomon-inspired coding across stylometric dimensions
- Allows partial recovery from incomplete document fragments
- Creates cross-carrier parity blocks for error detection and correction
- Enables flexible recovery strategies with minimal information

### Version-Differential Encoding
- Exploits natural document versioning patterns
- Embeds payloads in version-to-version differences
- Uses expected revision patterns (date updates, edition changes) as carriers
- Maintains plausible deniability through natural evolution of documents

## Algorithmic Approach

The system follows a multi-phase approach:

1. **Analysis Phase**: Construct capacity matrix by analyzing document structure and stylometric features
2. **Planning Phase**: Apply weighted allocation algorithm to distribute payload optimally
3. **Encoding Phase**: Implement parity/erasure coding and distribute across carriers
4. **Verification Phase**: Validate recoverability with simulated partial information

## Mathematical Model

### Capacity Matrix Formulation
The capacity matrix **C** is defined as:

```
C[i,j] = (capacity, detectability, robustness)
```

Where:
- **i** represents document segments (chapters, sections, metadata blocks)
- **j** represents carrier techniques (punctuation, lexical richness, etc.)

### Weighted Allocation Function
For a payload **P** of size |P|, the allocation function **A** minimizes:

```
min f(A) = α·D(A) + β·(1-R(A)) + γ·(1-U(A))
```

Where:
- **D(A)** is the overall detectability score
- **R(A)** is the recoverability under partial information
- **U(A)** is the capacity utilization ratio
- **α**, **β**, and **γ** are tunable weights

## Design Constraints

1. **Adaptation to Source**: Encoding must adapt to document's natural stylometric profile
2. **Structural Awareness**: Must leverage document structure without distorting it
3. **Payload Distribution**: Distribute payload and parity data optimally across carriers
4. **Partial Recoverability**: Support recovery from partial document segments
5. **Version Coherence**: Maintain coherence across document versions and editions

## Integration with Project

This module enhances the existing stylometric carrier framework by:

1. Adding structured analysis of document-specific carrying capacity
2. Providing optimal allocation strategies across carriers
3. Implementing resilient encoding with parity/erasure coding
4. Supporting partial recovery from document fragments

## References

1. Reed, I. S., & Solomon, G. (1960). Polynomial Codes Over Certain Finite Fields.
2. Fontaine, C., & Galand, F. (2007). A Survey of Homomorphic Encryption for Nonspecialists.
3. Wayner, P. (2009). Disappearing Cryptography: Information Hiding: Steganography & Watermarking.

# Carrier Matrix Design Notes

## Overview
The carrier matrix system provides optimal payload distribution and resilient encoding across document structures using multiple steganographic carriers.

## Design Goals
- Maximize encoding capacity while maintaining plausible deniability
- Ensure resilience through parity/erasure coding
- Preserve document structure and natural language properties
- Support multiple carrier techniques with different characteristics

## Architecture
The system uses a weighted capacity matrix approach:
```
Document Segments × Carrier Techniques → Capacity Metrics
```

### Components
- Document Segmentation: Hierarchical structure analysis
- Capacity Analysis: Per-segment carrier evaluation
- Optimal Distribution: Weight-based payload allocation
- Resilient Encoding: Parity data generation and recovery

## Constraints
- Detectability threshold: 0.3 (30%) maximum per carrier
- Redundancy level: 20% minimum for parity data
- Preserve document structure by default
- Support version-aware encoding for content evolution

## Integration Requirements
- Requires carrier implementations with analyzeCapacity() and encode()/extract() methods
- Works with any carrier that provides detectability/robustness metrics
- Must maintain backwards compatibility with existing carriers

## Testing & Verification
- Test with varying document structures and payload sizes
- Verify successful recovery with up to 20% content modification
- Measure detectability against baseline stylometric analysis

## Usage Notes
- Initialize with appropriate redundancyLevel for use case
- Monitor detectability metrics when adding new carriers
- Use version-aware mode when document evolution expected

# Carrier Matrix Module API Notes

## Overview

This module implements the `CarrierMatrix` class, responsible for managing stylometric data embedding across multiple segments of a document using various carrier techniques. It aims to distribute a payload optimally based on carrier capacity, detectability, robustness, and document structure, while adding resilience through erasure coding.

## Design Goals

*   **Structure-Aware:** Segment documents logically (chapters, sections) to apply carriers appropriately.
*   **Weighted Distribution:** Allocate payload bits based on a calculated weight for each segment-carrier pair, considering capacity, detectability, robustness, naturalness, and user preferences.
*   **Resilience:** Integrate erasure coding (e.g., Reed-Solomon) to allow payload recovery even if some segments are modified or carriers fail.
*   **Extensibility:** Easily add new carrier techniques by implementing the `Carrier` interface.
*   **Configurability:** Allow users to tune encoding parameters like redundancy level and detectability thresholds.
*   **Metadata Driven Recovery:** Generate and utilize metadata during encoding/decoding to map erasure-coded chunks to specific segment/carrier locations, enabling robust recovery.

## Key Components

*   `CarrierMatrix`: Main class orchestrating analysis, encoding, and decoding.
*   `DocumentSegment`: Interface representing a logical part of the document.
*   `CapacityCell`: Stores metrics and weights for a segment-carrier pair.
*   `Carrier` Interface (from `src/types/CarrierTypes.ts`): Defines the contract for carrier implementations (`analyzeCapacity`, `encode`, `extract`).
*   `ReedSolomon` (from `src/matrix/ErrorCorrection.ts`): Provides erasure coding capabilities.
*   `StyleFeatureExtractor` (from `stylometric_detection.genai.mjs`): Used for analyzing text features to inform weighting.
*   `EncodingMetadata`: Interface defining the structure (`totalChunks`, `chunkMap`) required to reconstruct the payload during decoding.
*   `ChunkLocation`: Interface detailing where a specific part of a chunk is encoded (`chunkIndex`, `segmentId`, `carrierKey`, `bitOffset`, `bitLength`).
*   `EncodeResult` (from `CarrierTypes.ts`): Interface returned by `carrier.encode`, includes `modifiedContent` and `bitsEncoded`.
*   `EncodePayloadResult`: Interface returned by `CarrierMatrix.encodePayload`, includes `encodedSegments` and `metadata`.

## Core Flow

1.  **Initialization:** Create `CarrierMatrix` instance, initialize carriers.
2.  **Analysis (`analyzeDocument`):**
    *   Segment the input document (`segmentDocument`).
    *   For each segment and carrier, call `carrier.analyzeCapacity`.
    *   Build the `capacityMatrix`.
    *   Analyze text features (`analyzeTextFeatures` using `StyleFeatureExtractor`).
    *   Calculate weights for each cell (`calculateWeights`).
    *   Return `AnalysisResult`.
3.  **Encoding (`encodePayload`):**
    *   Calculate required capacity based on payload size and redundancy level.
    *   Apply erasure coding (`applyErasureCoding` using `ReedSolomon.encode`) to get data and parity chunks. Determine `totalChunks`.
    *   Distribute encoded chunks (`distributePayload`):
        *   Iterate through each chunk.
        *   For each chunk, find suitable segment/carrier slots based on weights and available capacity.
        *   Call `carrier.encode` for parts of the chunk.
        *   Use the `bitsEncoded` from `EncodeResult` to track progress.
        *   Record the exact location (`segmentId`, `carrierKey`), chunk index, bit offset within the chunk, and bit length for each successfully encoded part in a `chunkMap`.
        *   Update the modified segment content map (`encodedSegments`).
    *   Assemble the `EncodingMetadata` (containing `totalChunks` and `chunkMap`).
    *   Return `EncodePayloadResult` { `encodedSegments`, `metadata` }.
4.  **Decoding (`decodePayload`):**
    *   Accept `encodedSegments` and `metadata` as input.
    *   Initialize an `orderedChunks` array of size `metadata.totalChunks` with `null` values.
    *   Use a cache (`extractionCache`) to store results of `carrier.extract` calls per segment/carrier.
    *   Iterate from `chunkIndex = 0` to `metadata.totalChunks - 1`.
    *   For each `chunkIndex`, find all its parts in `metadata.chunkMap`.
    *   For each part, retrieve/perform extraction using the cached `carrier.extract` for the specified `segmentId` and `carrierKey`.
    *   If extraction is successful, slice the required bits (`bitOffset`, `bitLength`) from the extracted data.
    *   Assemble the bits for the current chunk from all its parts.
    *   If all parts are successfully extracted and assembled, convert the bits to bytes (`bitsToBytes`) and place the `Uint8Array` in `orderedChunks` at the correct `chunkIndex`. If any part fails, the chunk remains `null`.
    *   Recover the original payload using erasure code correction (`recoverPayloadFromChunks` using `ReedSolomon.decode` on `orderedChunks`).
    *   Return the recovered payload.

## Constraints & Assumptions

*   Assumes `ReedSolomon` and `StyleFeatureExtractor` classes/modules are available and provide the expected interfaces.
*   Segmentation logic (`segmentDocument`) is currently basic and may need refinement for complex documents.
*   Carrier implementations must accurately report their capacity and adhere to the `Carrier` interface.
*   Payload size must be within the calculated weighted capacity after accounting for redundancy.
*   The `Carrier.encode` method must accurately report the number of bits it successfully encoded in the `EncodeResult`.
*   The `EncodingMetadata` generated by `encodePayload` must be passed unmodified to `decodePayload` for successful recovery.
*   The `ReedSolomon.decode` implementation must be able to handle `null` entries in the input chunk array, representing missing chunks.
*   Assumes `bytesToBits` and `bitsToBytes` utility functions exist and work correctly.