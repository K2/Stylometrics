# Stylometric Carrier Module

## Module Overview
This module inverts the stylometric detection techniques identified by Kumarage et al. (2023) to create information carriers within text. Rather than detecting AI-generated content, we exploit the same features to embed and extract steganographic payloads within text while maintaining natural readability and avoiding detection.

## Design Philosophy
The stylometric carrier approach leverages several observations:

1. Stylometric features that differentiate human from AI text can be subtly manipulated to carry information
2. These features offer multiple dimensions of carrying capacity that can be adjusted independently
3. By inverting detection techniques, we can create steganographic channels that remain invisible to common detection tools
4. The carrying capacity is distributed across multiple textual properties rather than concentrated in a single dimension

## Core Components

### Carrier Capacity Analysis
- Evaluates text for potential carrying capacity across stylometric dimensions
- Identifies which features can be safely manipulated without triggering detection
- Calculates bits-per-feature and total embedding capacity

### Stylometric Encoding Engine
- Embeds information by modifying phraseology patterns
- Uses punctuation frequency adjustments as data carriers
- Manipulates lexical diversity within natural bounds
- Adjusts readability scores strategically to encode bits

### Steganographic Recovery
- Extracts embedded information from carrier text
- Reconstructs the payload from multiple stylometric dimensions
- Provides error checking and correction for reliable recovery

## Manipulation Techniques

### Phraseology Carriers
- **Sentence Length Patterns**: Encode bits by varying sentence length sequences
- **Paragraph Structure**: Use paragraph length and structure to carry information
- **Word Position Modulation**: Strategic placement of specific parts of speech

### Punctuation Carriers
- **Punctuation Frequency**: Alter frequency of specific punctuation marks
- **Quote Style Alternation**: Switch between single/double quotes or different dash styles
- **Optional Comma Placement**: Use or omit stylistically optional commas

### Linguistic Diversity Carriers
- **Synonym Substitution**: Replace words with synonyms of varying frequency
- **Lexical Richness Modulation**: Adjust type-token ratio in specific text windows
- **Function Word Distribution**: Modify distribution patterns of common function words

### Readability Carriers
- **Syllable Count Adjustment**: Modify word choices to affect syllable counts
- **Sentence Complexity Variation**: Adjust syntax complexity in specific patterns
- **Passive/Active Voice Switching**: Alternate between voice styles to encode bits

## Carrying Capacity Analysis

| Feature Category | Manipulable Features | Estimated Bits Per 1000 Words |
|------------------|----------------------|------------------------------|
| Phraseology      | 5-7                  | 10-15                        |
| Punctuation      | 8-12                 | 15-25                        |
| Linguistic       | 4-6                  | 8-16                         |
| Readability      | 3-5                  | 5-12                         |
| **Total**        | **20-30**            | **38-68**                    |

## Design Constraints

1. **Naturalness Preservation**: All modifications must maintain natural reading flow and avoid detection
2. **Robustness**: Embedded information should survive minor text edits and formatting changes
3. **Density vs. Detectability**: Balance embedding density against risk of detection
4. **Language Specificity**: Techniques need to be adapted for different languages
5. **Error Tolerance**: Implement error correction for payload recovery

## Integration with Project

This module serves as a complementary approach to existing steganographic techniques in the project:

1. While other modules focus on direct text manipulation, this approaches steganography through statistical properties
2. The stylometric carriers provide a more distributed approach that's harder to detect
3. These techniques can be combined with existing methods for higher capacity or redundancy

## Usage Patterns

1. **Analysis Phase**: Analyze text to calculate potential carrying capacity
2. **Encoding Phase**: Embed payload by distributing across selected carriers
3. **Verification Phase**: Confirm readability and naturalness of modified text
4. **Extraction Phase**: Recover embedded information from carrier text

## References

- Kumarage, T., Garland, J., Bhattacharjee, A., Trapeznikov, K., Ruston, S., & Liu, H. (2023). Stylometric Detection of AI-Generated Text in Twitter Timelines.
- Bennett, K. (2004). Linguistic steganography: Survey, analysis, and robustness concerns for hiding information in text.
- Taskiran, C.M., et al. (2006). Attacks on lexical natural language steganography systems.