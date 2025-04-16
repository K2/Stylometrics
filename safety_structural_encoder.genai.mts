/**
 * Structural Steganography Module (Experimental)
 *
 * WARNING: This module implements highly experimental techniques for encoding data
 * in high-level linguistic structures. These methods are complex, may significantly
 * alter the meaning or naturalness of the text, and are likely fragile to further
 * editing or transformation. Use with extreme caution and expect potential failures
 * in encoding or extraction.
 *
 * Flow:
 * Content → Split → Multi-level structural encoding → Combine → Output
 */

import crypto from 'crypto'; // Not used currently, but kept
import assert from 'assert';

/**
 * Configuration for the structural encoding process
 */
interface StructuralEncodingOptions {
  minContentLength: number;
  maxEncodableBits: number;
  preserveExistingStructure: boolean; // Note: Currently has limited effect
  encodingStrength: 'subtle' | 'moderate' | 'aggressive';
}

// --- Interfaces and Enums (Keep as is) ---
interface NarrativePatterns { firstPerson: string[]; thirdPerson: string[]; dialogue: string[]; descriptive: string[]; }
const NARRATIVE_PATTERNS: NarrativePatterns = {
  firstPerson: [
    "I remember when",
    "As I considered",
    "In my experience",
    "Looking back, I realized",
    "From my perspective"
  ],
  thirdPerson: [
    "They observed that",
    "From their viewpoint",
    "It became clear to them",
    "After consideration, she decided",
    "He recognized the pattern"
  ],
  dialogue: [
    '"I see your point," she said. "Tell me more about that approach."',
    '"Consider the implications," he suggested. "What happens when we apply this further?"',
    '"That\'s interesting. And what about the alternative?" they asked.',
    '"This reveals something important. We should examine it carefully," noted the researcher.'
  ],
  descriptive: [
    "The structure stood thirty feet tall, metal beams intersecting at precise angles.",
    "Connections between concepts formed gradually, like a photograph developing.",
    "The room contained three chairs, a table, and a cabinet with seven drawers.",
    "Ideas floated through the discussion, nebulous at first, then crystallizing."
  ]
};
enum ParagraphPattern { ShortLongAlternating = 0, ConsistentMedium = 1, DecreasingLength = 0, IncreasingLength = 1 }
enum RhetoricalPattern { EndingQuestion = 0, EndingEmphasis = 1, Metaphorical = 0, Direct = 1, ClaimThenExample = 0, ExampleThenClaim = 1 }
interface ContentUnitAnalysis { wordCount: number; sentenceCount: number; averageSentenceLength: number; complexity: number; pov: 'first' | 'second' | 'third' | 'mixed'; tense: 'past' | 'present' | 'future' | 'mixed'; rhetorical: RhetoricalPattern[]; dialectic: 'argumentative' | 'descriptive' | 'narrative' | 'mixed'; }
// --- End Interfaces and Enums ---

/**
 * Default options for structural encoding
 */
const DEFAULT_OPTIONS: StructuralEncodingOptions = {
  minContentLength: 1000, // Requires substantial text
  maxEncodableBits: 64,   // Limited capacity
  preserveExistingStructure: true, // Attempt, but transformations are intrusive
  encodingStrength: 'moderate'
};

// Simple sentence splitter
const splitSentences = (text: string): string[] => {
    // Improved regex to handle more sentence endings and avoid splitting URLs etc.
    // It's still basic and won't handle all edge cases perfectly.
    return text.match(/([^\.!\?]+[\.!\?])(?=\s+[A-Z]|$)/g) || [text];
};

// Simple word splitter
const splitWords = (text: string): string[] => {
    return text.match(/\b\w+\b/g) || []; // Use regex to get only words
};

/**
 * Analyzes a section of text to determine its structural characteristics (Simplified)
 * @param text Content to analyze
 * @returns Analysis of content structure
 */
const analyzeContentUnit = (text: string): ContentUnitAnalysis => {
  const sentences = splitSentences(text);
  const words = splitWords(text);
  const wordCount = words.length;
  const sentenceCount = sentences.length;

  // --- POV Detection (Simplified) ---
  const firstPersonIndicators = text.match(/\b(I|me|my|mine|myself)\b/gi) || [];
  const secondPersonIndicators = text.match(/\b(you|your|yours|yourself)\b/gi) || [];
  const thirdPersonIndicators = text.match(/\b(he|she|it|they|him|her|them|his|hers|their|theirs)\b/i) || []; // Case insensitive for third person
  let pov: ContentUnitAnalysis['pov'] = 'mixed';
  const povCounts = [firstPersonIndicators.length, secondPersonIndicators.length, thirdPersonIndicators.length];
  const totalPov = povCounts.reduce((a, b) => a + b, 0);
  if (totalPov > 5) { // Require a minimum number of indicators
      if (firstPersonIndicators.length / totalPov > 0.7) pov = 'first';
      else if (secondPersonIndicators.length / totalPov > 0.7) pov = 'second';
      else if (thirdPersonIndicators.length / totalPov > 0.7) pov = 'third';
  }
  // --- End POV ---

  // --- Tense Detection (Simplified) ---
  const pastTenseIndicators = text.match(/\b(\w+ed|was|were|had|went|saw|did)\b/gi) || [];
  const presentTenseIndicators = text.match(/\b(is|are|am|being|has|have|go|see|do|does)\b/gi) || []; // Added more verbs
  const futureTenseIndicators = text.match(/\b(will|shall|going to)\b/gi) || [];
  let tense: ContentUnitAnalysis['tense'] = 'mixed';
   const tenseCounts = [pastTenseIndicators.length, presentTenseIndicators.length, futureTenseIndicators.length];
   const totalTense = tenseCounts.reduce((a, b) => a + b, 0);
   if (totalTense > 5) { // Require minimum indicators
       if (pastTenseIndicators.length / totalTense > 0.7) tense = 'past';
       else if (presentTenseIndicators.length / totalTense > 0.7) tense = 'present';
       else if (futureTenseIndicators.length / totalTense > 0.7) tense = 'future';
   }
  // --- End Tense ---

  // --- Rhetorical Pattern Detection (Simplified) ---
  const rhetorical: RhetoricalPattern[] = [];
  const lastSentence = sentences[sentenceCount - 1]?.trim() || '';
  if (lastSentence.endsWith('?')) rhetorical.push(RhetoricalPattern.EndingQuestion);
  else if (lastSentence.endsWith('!') || lastSentence.match(/\b(must|indeed|clearly|crucial|essential|vital|paramount)\b/i)) rhetorical.push(RhetoricalPattern.EndingEmphasis); // Expanded emphasis check

  const metaphorIndicators = text.match(/\b(like|as if|as though|resembles|similar to|metaphor for|akin to)\b/gi) || [];
  if (metaphorIndicators.length > 1) rhetorical.push(RhetoricalPattern.Metaphorical); // Require more than one indicator
  else rhetorical.push(RhetoricalPattern.Direct); // Assume direct if no clear metaphor indicators

  // Simplified claim/example order check (very basic)
  const firstSentence = sentences[0]?.trim() || '';
  const exampleKeywords = /\b(example|instance|consider|illustrate|case in point)\b/i;
  if (exampleKeywords.test(firstSentence) && !exampleKeywords.test(lastSentence)) rhetorical.push(RhetoricalPattern.ExampleThenClaim);
  else if (!exampleKeywords.test(firstSentence) && exampleKeywords.test(lastSentence)) rhetorical.push(RhetoricalPattern.ClaimThenExample);
  else rhetorical.push(RhetoricalPattern.ClaimThenExample); // Default assumption if unclear
  // --- End Rhetorical ---

  // --- Dialectic Detection (Simplified) ---
  const argumentativeIndicators = text.match(/\b(because|therefore|thus|hence|so|consequently|however|but|although|yet|since|as a result)\b/gi) || [];
  const descriptiveIndicators = text.match(/\b(appears|looks|seems|features|contains|consists|is|are|has|have|includes|measures|weighs)\b/gi) || [];
  const narrativeIndicators = text.match(/\b(then|next|after|before|finally|eventually|when|while|suddenly|later|meanwhile)\b/gi) || [];
  let dialectic: ContentUnitAnalysis['dialectic'] = 'mixed';
   const dialecticCounts = [argumentativeIndicators.length, descriptiveIndicators.length, narrativeIndicators.length];
   const totalDialectic = dialecticCounts.reduce((a, b) => a + b, 0);
   if (totalDialectic > 5) { // Require minimum indicators
       if (argumentativeIndicators.length / totalDialectic > 0.6) dialectic = 'argumentative';
       else if (descriptiveIndicators.length / totalDialectic > 0.6) dialectic = 'descriptive';
       else if (narrativeIndicators.length / totalDialectic > 0.6) dialectic = 'narrative';
   }
  // --- End Dialectic ---

  // --- Complexity (Simple Heuristic) ---
  const uniqueWords = new Set(words.map(w => w.toLowerCase()));
  const lexicalDiversity = wordCount > 0 ? uniqueWords.size / wordCount : 0;
  const avgSentenceLength = sentenceCount > 0 ? wordCount / sentenceCount : 0;
  // Normalize complexity score (e.g., to 0-1 range, very roughly)
  const complexity = Math.min(1, (avgSentenceLength / 30) * lexicalDiversity * 2); // Adjusted formula
  // --- End Complexity ---

  return {
    wordCount, sentenceCount, averageSentenceLength: avgSentenceLength,
    complexity, pov, tense, rhetorical, dialectic
  };
};

/**
 * Splits content into workable structural units (paragraphs)
 * @param content Full text content
 * @returns Array of content units (paragraphs)
 */
const splitContentIntoUnits = (content: string): string[] => {
  // Split by one or more empty lines to get paragraphs, trim, and filter empty
  return content.split(/\n\s*\n+/).map(p => p.trim()).filter(p => p.length > 0);
};

/**
 * Hides data by modifying paragraph structure (splitting/merging).
 * Bit = 1: Try to merge current paragraph with next (if both are short).
 * Bit = 0: Try to split current paragraph (if long enough).
 * @param originalText The text content.
 * @param bitsToEncode Array of boolean bits to encode.
 * @param strength 'subtle' (few changes), 'moderate', 'aggressive' (more changes). Not fully implemented yet, acts like 'moderate'.
 * @param preserveExistingStructure Attempt to avoid major structural changes (e.g., headings). Basic implementation.
 * @returns Modified text with embedded bits and count of bits encoded.
 */
function encodeStructuralBits(
    originalText: string,
    bitsToEncode: boolean[],
    strength: 'subtle' | 'moderate' | 'aggressive' = 'moderate',
    preserveExistingStructure: boolean = true
): { modifiedText: string; bitsEncoded: number } {
    assert(originalText != null, '[encodeStructuralBits] Input text cannot be null.');
    assert(bitsToEncode != null, '[encodeStructuralBits] Input bits cannot be null.');

    const paragraphs = splitContentIntoUnits(originalText); // Use refined splitter
    if (paragraphs.length < 2) {
        console.warn("[encodeStructuralBits] Insufficient paragraphs (< 2) for structural encoding.");
        return { modifiedText: originalText, bitsEncoded: 0 };
    }

    let modifiedParagraphs: string[] = [];
    let bitIndex = 0;
    let i = 0;

    // Adjust thresholds based on strength (example)
    const strengthMultiplier = strength === 'subtle' ? 1.5 : (strength === 'aggressive' ? 0.7 : 1.0);
    const MIN_PARA_LEN_SPLIT = 150 * strengthMultiplier; // Higher threshold for subtle, lower for aggressive
    const MAX_PARA_LEN_MERGE = 100 / strengthMultiplier; // Lower threshold for subtle, higher for aggressive
    const HEADING_REGEX = /^(#+|\*+|\-+|\d+\.\s)/; // Basic check for headings or list items

    while (i < paragraphs.length && bitIndex < bitsToEncode.length) {
        const currentPara = paragraphs[i]; // Already trimmed by splitContentIntoUnits
        const nextPara = (i + 1 < paragraphs.length) ? paragraphs[i + 1] : null;

        // Skip headings or list items if preserving structure
        if (preserveExistingStructure && HEADING_REGEX.test(currentPara)) {
             modifiedParagraphs.push(currentPara);
             i++;
             continue;
        }

        const bit = bitsToEncode[bitIndex];
        let appliedChange = false;

        // Try merging (Bit = 1)
        if (bit && nextPara && currentPara.length < MAX_PARA_LEN_MERGE && nextPara.length < MAX_PARA_LEN_MERGE && (!preserveExistingStructure || !HEADING_REGEX.test(nextPara))) {
            // Merge with a single space, assuming paragraphs end without punctuation needing removal
            modifiedParagraphs.push(`${currentPara} ${nextPara}`);
            i += 2; // Skip next paragraph as it's merged
            bitIndex++;
            appliedChange = true;
        // Try splitting (Bit = 0)
        } else if (!bit && currentPara.length >= MIN_PARA_LEN_SPLIT) {
            const sentences = splitSentences(currentPara); // Use refined splitter
            if (sentences && sentences.length > 1) {
                // Find a split point roughly in the middle, preferring after a sentence end
                let splitPoint = Math.floor(sentences.length / 2);
                // Ensure split point is not 0
                splitPoint = Math.max(1, splitPoint);

                let para1 = sentences.slice(0, splitPoint).join(' ').trim();
                let para2 = sentences.slice(splitPoint).join(' ').trim();

                // Avoid creating very short paragraphs if possible
                if (para1.length > 20 && para2.length > 20) {
                    modifiedParagraphs.push(para1);
                    modifiedParagraphs.push(para2);
                    i++; // Move to next original paragraph index
                    bitIndex++;
                    appliedChange = true;
                } else {
                     // Split would create tiny paragraphs, skip modification
                     modifiedParagraphs.push(currentPara);
                     i++;
                }
            } else {
                 // Not enough sentences to split
                 modifiedParagraphs.push(currentPara);
                 i++;
            }
        } else {
            // Condition not met for merge/split, or paragraph skipped
            modifiedParagraphs.push(currentPara);
            i++;
        }
    }

    // Add remaining paragraphs that weren't processed
    while (i < paragraphs.length) {
        modifiedParagraphs.push(paragraphs[i]); // Already trimmed
        i++;
    }

    // Reconstruct text with double line breaks
    const modifiedText = modifiedParagraphs.join('\n\n');
    return { modifiedText, bitsEncoded: bitIndex };
}

/**
 * Extracts bits based on paragraph structure.
 * Assumes merging short paragraphs = 1, splitting long paragraphs = 0.
 * Highly sensitive to reformatting and ambiguity. Requires original text for reliable extraction.
 * @param text Text possibly containing structurally encoded data.
 * @param originalText The original text *before* encoding. Required for reliable extraction.
 * @returns Array of boolean bits or null if extraction seems invalid or original text is missing.
 */
function extractStructuralBits(text: string, originalText?: string): boolean[] | null {
    assert(text != null, '[extractStructuralBits] Input text cannot be null.');

    if (!originalText) {
        console.warn("[extractStructuralBits] Original text is required for reliable structural bit extraction. Returning empty array (unreliable).");
        // Attempting extraction without original is highly unreliable and likely incorrect.
        // This part would need a complex heuristic comparing paragraph lengths/counts
        // to guess if splits/merges happened, which is beyond the scope of this example.
        return []; // Return empty as a placeholder for unreliable extraction
    }

    const currentParagraphs = splitContentIntoUnits(text);
    const originalParagraphs = splitContentIntoUnits(originalText);

    const bits: boolean[] = [];
    let origIdx = 0;
    let modIdx = 0;

    const HEADING_REGEX = /^(#+|\*+|\-+|\d+\.\s)/;

    // This logic attempts to reconstruct the encoding decisions by comparing original and modified paragraphs.
    // It's still fragile and assumes the encoding process was the primary change.
    while (origIdx < originalParagraphs.length && modIdx < currentParagraphs.length) {
        const origPara = originalParagraphs[origIdx];
        const modPara = currentParagraphs[modIdx];

        // Skip headings if they match
        if (HEADING_REGEX.test(origPara) && origPara === modPara) {
            origIdx++;
            modIdx++;
            continue;
        }

        // Check for potential merge (orig[i] + orig[i+1] == mod[j])
        if (origIdx + 1 < originalParagraphs.length) {
            const nextOrigPara = originalParagraphs[origIdx + 1];
            // Approximate check for merge (allow for space difference)
            if (`${origPara} ${nextOrigPara}` === modPara) {
                bits.push(true); // Merge implies bit 1
                origIdx += 2;
                modIdx += 1;
                continue;
            }
        }

        // Check for potential split (orig[i] == mod[j] + mod[j+1])
        if (modIdx + 1 < currentParagraphs.length) {
            const nextModPara = currentParagraphs[modIdx + 1];
             // Approximate check for split
            if (`${modPara} ${nextModPara}` === origPara) {
                bits.push(false); // Split implies bit 0
                origIdx += 1;
                modIdx += 2;
                continue;
            }
        }

        // If no merge or split detected, assume no bit encoded here, advance both
        // This might happen if a paragraph was too long/short for modification
        // or if other edits occurred.
        if (origPara === modPara) {
             origIdx++;
             modIdx++;
        } else {
            // Paragraphs differ but don't match split/merge pattern.
            // This indicates either an unhandled encoding case or external modification.
            console.warn(`[extractStructuralBits] Unmatched paragraphs at origIdx ${origIdx}, modIdx ${modIdx}. Extraction may be inaccurate.`);
            // Attempt to resync - simple strategy: advance both
            origIdx++;
            modIdx++;
        }
    }

    if (origIdx < originalParagraphs.length || modIdx < currentParagraphs.length) {
         console.warn("[extractStructuralBits] Mismatch in paragraph counts after comparison. Extraction might be incomplete or inaccurate.");
    }

    return bits;
}


/**
 * Hides data within the structural elements of the text.
 * Adds a marker to indicate structural encoding was used.
 * WARNING: This method can significantly alter text structure and is fragile.
 */
export function hideDataStructurally(
    originalText: string,
    data: string,
    options?: Partial<StructuralEncodingOptions> // Allow partial options
): string {
    assert(originalText != null, 'Original text must not be null');
    assert(data != null, 'Data to hide must not be null');

    const mergedOptions: StructuralEncodingOptions = { ...DEFAULT_OPTIONS, ...options };

    // Check min length requirement
    if (originalText.length < mergedOptions.minContentLength) {
        console.warn(`[hideDataStructurally] Content length (${originalText.length}) is less than minimum required (${mergedOptions.minContentLength}). Skipping structural encoding.`);
        return originalText; // Return original text if too short
    }

    const buffer = Buffer.from(data, 'utf-8');
    const bits: boolean[] = [];
    const dataLengthInBits = buffer.length * 8;

    // Limit data length based on maxEncodableBits
    const maxDataBits = mergedOptions.maxEncodableBits - 16; // Reserve 16 bits for length
    if (dataLengthInBits > maxDataBits) {
        console.warn(`[hideDataStructurally] Data size (${dataLengthInBits} bits) exceeds maxEncodableBits (${maxDataBits}). Truncating data.`);
        // Adjust buffer length to fit
        const maxBytes = Math.floor(maxDataBits / 8);
        buffer.slice(0, maxBytes); // This doesn't modify buffer in place, need reassignment
        const truncatedBuffer = buffer.slice(0, maxBytes);
        const truncatedLengthInBits = truncatedBuffer.length * 8;

        const lengthBuffer = Buffer.alloc(2);
        lengthBuffer.writeUInt16BE(truncatedLengthInBits, 0);
        for (let i = 0; i < lengthBuffer.length; i++) {
            for (let j = 7; j >= 0; j--) bits.push(((lengthBuffer[i] >> j) & 1) === 1);
        }
        for (let i = 0; i < truncatedBuffer.length; i++) {
            for (let j = 7; j >= 0; j--) bits.push(((truncatedBuffer[i] >> j) & 1) === 1);
        }

    } else {
        // Encode length (16 bits)
        const lengthBuffer = Buffer.alloc(2);
        lengthBuffer.writeUInt16BE(dataLengthInBits, 0);
        for (let i = 0; i < lengthBuffer.length; i++) {
            for (let j = 7; j >= 0; j--) bits.push(((lengthBuffer[i] >> j) & 1) === 1);
        }
        // Encode data
        for (let i = 0; i < buffer.length; i++) {
            for (let j = 7; j >= 0; j--) bits.push(((buffer[i] >> j) & 1) === 1);
        }
    }


    const { modifiedText, bitsEncoded } = encodeStructuralBits(
        originalText,
        bits,
        mergedOptions.encodingStrength,
        mergedOptions.preserveExistingStructure
    );

    if (bitsEncoded < bits.length) {
        console.warn(`[hideDataStructurally] Structural encoding incomplete. Only ${bitsEncoded} of ${bits.length} bits encoded.`);
        // Decide if partial encoding is acceptable or should throw/return original
        // For now, proceed with partially encoded text but add a warning marker
         const marker = "\n\n[Structure Modified for Encoding - INCOMPLETE]\n\n";
         return marker + modifiedText + marker;
    }

    const marker = "\n\n[Structure Modified for Encoding]\n\n";
    return marker + modifiedText + marker;
}

/**
 * Extracts data hidden using structural encoding.
 * Relies on finding markers and the highly unreliable extractStructuralBits.
 * WARNING: Extraction is likely to fail if the text was reformatted or original text is unavailable.
 * @param text Text to extract from.
 * @param originalText The original text *before* encoding. Required for reliable extraction.
 */
export function extractHiddenStructuralData(text: string, originalText?: string): string | null {
    assert(text != null, 'Text to extract from must not be null');
    const marker = "[Structure Modified for Encoding"; // Allow incomplete marker too
    const startMarkerIndex = text.indexOf(marker);
    const endMarkerIndex = text.lastIndexOf(marker); // Find the last occurrence

    if (startMarkerIndex === -1) {
        // console.log("[extractHiddenStructuralData] Encoding marker not found.");
        return null; // No marker found
    }

    // Find the end of the start marker line
    const markerEndLineIndex = text.indexOf('\n', startMarkerIndex);
    if (markerEndLineIndex === -1) {
         console.warn("[extractHiddenStructuralData] Malformed start marker.");
         return null;
    }

    // Find the start of the end marker line
    let contentEndIndex = endMarkerIndex;
    if (endMarkerIndex !== -1) {
        const endMarkerLineStart = text.lastIndexOf('\n', endMarkerIndex);
        contentEndIndex = (endMarkerLineStart !== -1) ? endMarkerLineStart : endMarkerIndex; // Use start of line if found
    } else {
        // If no end marker, assume content goes to end of text (less reliable)
        contentEndIndex = text.length;
        console.warn("[extractHiddenStructuralData] End marker not found. Attempting extraction on content after start marker.");
    }


    const contentToDecode = text.substring(markerEndLineIndex + 1, contentEndIndex).trim();

    // Pass original text if available
    const extractedBits = extractStructuralBits(contentToDecode, originalText);
    if (!extractedBits) {
         console.warn("[extractHiddenStructuralData] Failed to extract structural bits (likely due to ambiguity or missing original text).");
         return null;
    }

    if (extractedBits.length < 16) {
        console.warn(`[extractHiddenStructuralData] Not enough bits (${extractedBits.length}) for length prefix.`);
        return null;
    }

    // Decode length (MSB first)
    let lengthValue = 0;
    for (let i = 0; i < 16; i++) {
        lengthValue = (lengthValue << 1) | (extractedBits[i] ? 1 : 0);
    }

    const expectedTotalBits = 16 + lengthValue;
    if (extractedBits.length < expectedTotalBits) {
        console.warn(`[extractHiddenStructuralData] Incomplete data: Expected ${expectedTotalBits} bits (16 + ${lengthValue}), got ${extractedBits.length}. Attempting partial decode.`);
         // Adjust lengthValue to the number of available data bits
         lengthValue = extractedBits.length - 16;
         if (lengthValue <= 0) return null; // Not enough bits even for partial data
    } else if (extractedBits.length > expectedTotalBits) {
         console.warn(`[extractHiddenStructuralData] Extra bits detected: Expected ${expectedTotalBits}, got ${extractedBits.length}. Using expected length.`);
         // Truncate extractedBits to expected length
         extractedBits.splice(expectedTotalBits);
    }


    const dataBits = extractedBits.slice(16); // Get all bits after length prefix up to the determined end
    const numBytes = Math.ceil(lengthValue / 8);
    if (numBytes === 0) return ""; // Handle case where length is 0

    const buffer = Buffer.alloc(numBytes);
    for (let i = 0; i < numBytes; i++) {
        let byteValue = 0;
        for (let j = 0; j < 8; j++) {
            const bitIndex = i * 8 + j;
            if (bitIndex < dataBits.length) { // Check against actual dataBits length
                byteValue = (byteValue << 1) | (dataBits[bitIndex] ? 1 : 0);
            } else {
                // If lengthValue indicated more bits than available in dataBits (due to truncation warning)
                // pad with zeros.
                byteValue <<= 1;
            }
        }
        buffer[i] = byteValue;
    }

    try {
        return buffer.toString('utf-8');
    } catch (e) {
        console.error("[extractHiddenStructuralData] Error converting buffer to UTF-8:", e);
        return null; // Return null if buffer is invalid UTF-8
    }
}