# Counterfactual Stylometric Embedding

## Design Philosophy

This module exploits the "error space" in natural language to embed information through deliberate 
grammatical anomalies that fall within the statistical norms of human writing. The system leverages 
the fact that certain types of grammatical anomalies are difficult for LLMs to detect and correct
due to alignment constraints, safety guardrails, and the ambiguity inherent in natural language.

## High-Entropy Encoding Zones

We've identified several grammatical and structural patterns that create high-entropy zones
where information can be efficiently encoded:

1. **Subject-Verb Agreement Violations**: Embedding bits by toggling between singular/plural agreement
2. **Tense Inconsistencies**: Encoding information through unexpected tense shifts
3. **Possessive Marker Anomalies**: Using variations in apostrophe placement to carry data
4. **Comma Splicing**: Joining independent clauses with commas to embed information
5. **Character Repetition Patterns**: Using variable letter repetitions (e.g., "bananaaaa" vs "bananaaaaaa")
6. **Temporal Discontinuities**: Creating narrative time shifts without transition markers
7. **Recursive/Self-Referential Patterns**: Constructing potentially confusing self-references

## Architectural Constraints

- **Statistical Camouflage**: All embeddings must stay within the natural error budget of the text
- **Author-Mimicry**: Pattern distribution should mimic the author's natural error patterns
- **Readability Preservation**: Embeddings should minimize impact on human readability
- **LLM Sanitization Resistance**: Patterns should be resistant to LLM-based detection and correction

## Implementation Notes

- The `CounterfactualPatternRegistry` maintains a catalog of embedding patterns with their properties
- `CounterfactualEmbedder` handles the actual encoding and decoding of information
- `LLMVulnerabilityProber` tests different patterns against various LLMs to identify optimal techniques
- Embedding capacity is calculated based on text length, genre, and estimated error budget

## Primary Use Cases

- **Stylometric Watermarking**: Embedding attribution information in text
- **Deniable Communication**: Creating text with embedded messages that appear natural
- **LLM Detection Evasion**: Generating text that appears human-authored
- **Adversarial Examples**: Testing and improving LLM robustness

## Anti-Patterns

- Overloading text with too many anomalies, exceeding statistical norms
- Using patterns with high detection rates by current LLMs
- Embedding in ways that significantly impact readability
- Relying on a single pattern type rather than distributing across multiple types