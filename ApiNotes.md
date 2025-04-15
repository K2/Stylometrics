# Stylometrics Project: ApiNotes

## Project Overview
Stylometrics is a TypeScript library that implements advanced text analysis techniques for both AI-generated text detection and steganographic information embedding. The project leverages statistical and linguistic patterns in text to provide dual-purpose functionality: security analysis and covert communication.

## Architectural Principles

### Dual-Purpose Design
The core insight driving this project is that the same stylometric features that distinguish human from AI writing can be deliberately manipulated to carry hidden information. This creates a unique steganographic approach that maintains natural text properties while embedding data.

### Layered Architecture
1. **Feature Extraction Layer**: Analyzes raw text to extract stylometric features
2. **Detection Layer**: Classifies text and identifies authorship changes
3. **Carrier Layer**: Manipulates stylometric features to embed information
4. **Integration Layer**: Unifies detection and steganography capabilities

### Modularity & Extension
The system is designed with clear separation of concerns, allowing for:
- Independent development of new carrier techniques
- Extension of feature extraction capabilities
- Integration with external language models and embedding systems
- Application to various text analysis use cases

### Steganographic Design Principles
1. **Distributed Payload**: Information is spread across multiple stylometric dimensions
2. **Naturalness Preservation**: Modifications stay within human writing parameters
3. **Detection Resistance**: Changes are calibrated to avoid triggering AI detection
4. **Adaptive Capacity**: Capacity analysis adapts to specific text characteristics

## Core Components

### Stylometric Detection (.genai.mts)
- Feature extraction from text (lexical richness, readability, etc.)
- Binary classification of text as human/AI-generated
- Timeline analysis for authorship change detection

### Stylometric Carrier (.genai.mts)
- Analysis of text carrying capacity
- Multiple carrier techniques to embed information
- Verification of text naturalness after modification
- Safe modification ranges to preserve text authenticity

### Specialized Carriers (.mts)
- Individual implementations of specific carrier techniques
- Optimized for particular text features (punctuation, quotation, etc.)
- Graduated detectability ratings for different techniques

### Unified Toolkit (.mts)
- Integration of detection and carrier capabilities
- Comprehensive text analysis for dual purposes
- Common utilities for bit/byte/text conversion

## Data Flow

```
Text Input → Feature Extraction → |→ AI Detection → Classification
                                 |
                                 |→ Carrying Capacity Analysis → Carrier Selection
                                            ↓
                        Payload → Encoding → Text Modification → Output
                                            ↓
                                    Extract → Recovered Payload
```

## Extension Points

### New Carrier Techniques
- Additional stylometric dimensions can be exploited
- Novel carrier implementations can target specific text types
- Detectability vs. capacity tradeoffs can be explored

### Advanced Feature Extraction
- Deep learning models for feature extraction
- Language-specific stylometric features
- Genre and domain-specific feature sets

### Integration Capabilities
- API for external applications
- CLI tools for batch processing
- Web interface for interactive analysis

## Implementation Constraints

### Language & Runtime
- TypeScript with ES Modules (.mts extension)
- Compatible with both browser and Node.js environments
- Minimal external dependencies for core functionality

### Performance Considerations
- Feature extraction should be efficient for large texts
- Memory usage should scale reasonably with text size
- Encoding/decoding operations should be optimizable

### Security Aspects
- Steganographic techniques should resist statistical analysis
- Detection should work against evolving AI models
- Payload extraction should require appropriate keys/parameters

## Future Directions
- Weighted matrix capacity analysis for adaptive embedding
- Higher-order feature manipulation techniques
- Parity/erasure coding for resilient information recovery
- Metadata-based carrier techniques for structured documents
- Multi-modal steganography across text properties