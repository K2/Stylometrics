/**
 * Stylometric Steganography Module (Simplified Version)
 *
 * This module provides basic linguistic steganography capabilities by encoding data into
 * natural language text patterns without using invisible characters. It manipulates
 * sentence structure/word patterns to embed data while maintaining readability.
 *
 * NOTE: This is a simplified implementation. For more advanced and diverse techniques,
 * refer to `stylometric_carrier.genai.mts`. This version uses a basic
 * adverb presence/absence mechanism.
 *
 * Flow:
 * Input text → Split into units → Encode bits by pattern transforms → Combine → Output
 */

import crypto from 'crypto'; // Not used in current simplified version, but kept for potential future use
import assert from 'assert';

/**
 * Function to convert data to bits, potentially adding length prefix
 * @param data String data to convert
 * @returns Array of boolean bits
 */
function dataToBitsWithLength(data: string): boolean[] {
    const buffer = Buffer.from(data, 'utf-8');
    const bits: boolean[] = [];
    const dataLengthInBits = buffer.length * 8;

    // Encode length first (e.g., using 16 bits = 2 bytes)
    const lengthBuffer = Buffer.alloc(2);
    lengthBuffer.writeUInt16BE(dataLengthInBits, 0); // Max length ~65k bits

    for (let i = 0; i < lengthBuffer.length; i++) {
        const byte = lengthBuffer[i];
        for (let j = 7; j >= 0; j--) { // MSB first for length
            bits.push(((byte >> j) & 1) === 1);
        }
    }

    // Encode actual data
    for (let i = 0; i < buffer.length; i++) {
        const byte = buffer[i];
        for (let j = 7; j >= 0; j--) { // Keep MSB first for consistency or LSB if required by carrier
            bits.push(((byte >> j) & 1) === 1);
        }
    }
    return bits;
}

/**
 * Function to convert bits back to data, reading length first
 * @param bits Array of boolean bits
 * @returns Decoded string or null if decoding fails
 */
function bitsToDataWithLength(bits: boolean[]): string | null {
    if (bits.length < 16) { // Need at least 16 bits for the length prefix
        console.warn("Not enough bits to decode length prefix.");
        return null;
    }

    // Decode length (16 bits)
    let lengthValue = 0;
    for (let i = 0; i < 16; i++) {
        lengthValue = (lengthValue << 1) | (bits[i] ? 1 : 0);
    }

    const expectedTotalBits = 16 + lengthValue;
    if (bits.length < expectedTotalBits) {
        console.warn(`Incomplete data: Expected ${expectedTotalBits} bits (16 + ${lengthValue}), got ${bits.length}.`);
        lengthValue = bits.length - 16; // Adjust length to what's available
        if (lengthValue <= 0) return null;
    }

    const dataBits = bits.slice(16, expectedTotalBits);
    const numBytes = Math.ceil(lengthValue / 8);
    const buffer = Buffer.alloc(numBytes);

    for (let i = 0; i < numBytes; i++) {
        let byteValue = 0;
        for (let j = 0; j < 8; j++) {
            const bitIndex = i * 8 + j;
            if (bitIndex < dataBits.length) {
                byteValue = (byteValue << 1) | (dataBits[bitIndex] ? 1 : 0);
            } else {
                byteValue <<= 1; // Pad with 0 if dataBits is not multiple of 8
            }
        }
        buffer[i] = byteValue;
    }

    return buffer.toString('utf-8');
}

/**
 * Encodes a bit string by manipulating sentence structures (presence/absence of a marker adverb).
 * @param text Original text content to embed data within
 * @param bitsToEncode Binary string of bits to encode
 * @returns Modified text with embedded bits and count of bits encoded
 */
const encodeStylometricBits = (text: string, bitsToEncode: boolean[]): { modifiedText: string, bitsEncoded: number } => {
    const sentences = text.match(/[^.!?]+[.!?]+/g) || [];
    if (sentences.length === 0) {
        console.warn("[encodeStylometricBits] No sentences found in text.");
        return { modifiedText: text, bitsEncoded: 0 };
    }

    const capacity = Math.max(0, sentences.length - 1);
    if (capacity < bitsToEncode.length) {
        console.warn(`[encodeStylometricBits] Text capacity (${capacity} bits) may be insufficient for payload (${bitsToEncode.length} bits).`);
    }

    let encodedText = '';
    let bitIndex = 0;
    const markerAdverb = ' actually';
    const markerRegex = /\bactually\b/i;

    const startIndex = 1;
    encodedText += sentences[0] || '';

    for (let i = startIndex; i < sentences.length; i++) {
        let currentSentence = sentences[i].trim();
        if (currentSentence.split(' ').length < 4 || bitIndex >= bitsToEncode.length) {
            encodedText += (i > 0 ? ' ' : '') + currentSentence;
            continue;
        }

        const bit = bitsToEncode[bitIndex];
        let modifiedCurrent = currentSentence;
        let appliedModification = false;

        const hasMarker = markerRegex.test(currentSentence);

        if (bit) {
            if (!hasMarker) {
                const words = currentSentence.split(' ');
                const insertPos = Math.min(2, words.length - 1);
                words.splice(insertPos, 0, markerAdverb.trim());
                modifiedCurrent = words.join(' ');
                appliedModification = true;
            } else {
                appliedModification = true;
            }
        } else {
            if (hasMarker) {
                modifiedCurrent = currentSentence.replace(markerRegex, '').replace(/\s{2,}/g, ' ');
                appliedModification = true;
            } else {
                appliedModification = true;
            }
        }

        if (appliedModification) {
            encodedText += (i > 0 ? ' ' : '') + modifiedCurrent;
            bitIndex++;
        } else {
            encodedText += (i > 0 ? ' ' : '') + currentSentence;
            console.warn(`[encodeStylometricBits] Could not apply bit ${bit} to sentence ${i}. Skipping bit.`);
        }
    }

    for (let i = startIndex + bitIndex; i < sentences.length; i++) {
        if (sentences[i]) {
            encodedText += (i > 0 ? ' ' : '') + sentences[i].trim();
        }
    }

    if (bitIndex < bitsToEncode.length) {
        console.warn(`[encodeStylometricBits] Encoding incomplete. Only ${bitIndex} of ${bitsToEncode.length} bits were encoded.`);
    }

    return { modifiedText: encodedText.trim(), bitsEncoded: bitIndex };
};

/**
 * Encodes data using paragraph structure and punctuation patterns
 * @param originalText Text to encode within
 * @param data String data to hide
 * @returns Text with stylometrically encoded data
 */
export function hideDataStylometrically(originalText: string, data: string): string {
    assert(originalText != null, 'Original text must not be null');
    assert(data != null, 'Data to hide must not be null');
    const bitsToEncode = dataToBitsWithLength(data);
    const { modifiedText, bitsEncoded } = encodeStylometricBits(originalText, bitsToEncode);
    if (bitsEncoded < bitsToEncode.length) {
        console.warn(`Stylometric encoding incomplete: Only ${bitsEncoded} of ${bitsToEncode.length} bits could be encoded.`);
    }
    const markerText = "Please note the following information carefully. ";
    return markerText + modifiedText;
}

/**
 * Extracts bits from text based on stylometric patterns (marker adverb presence/absence)
 * @param text Text with encoded data
 * @returns Extracted binary string or null if not found or marker missing
 */
const extractStylometricBits = (text: string): boolean[] | null => {
    const sentences = text.match(/[^.!?]+[.!?]+/g) || [];
    if (sentences.length < 1) {
        console.log("[extractStylometricBits] No sentences found.");
        return null;
    }

    const extractedBits: boolean[] = [];
    const markerRegex = /\bactually\b/i;

    const startIndex = 1;

    for (let i = startIndex; i < sentences.length; i++) {
        const sentence = sentences[i].trim();
        if (sentence.split(' ').length < 4) {
            continue;
        }

        const hasMarker = markerRegex.test(sentence);
        extractedBits.push(hasMarker);
    }

    return extractedBits;
};

/**
 * Extracts hidden data from text with stylometric encoding
 * @param text Text to analyze
 * @returns Extracted string or null if not found or decoding fails
 */
export function extractHiddenStylometricData(text: string): string | null {
    assert(text != null, 'Text to extract from must not be null');
    const markerText = "Please note the following information carefully.";
    const markerIndex = text.indexOf(markerText);
    if (markerIndex === -1) {
        return null;
    }
    const contentToDecode = text.substring(markerIndex + markerText.length).trimStart();

    const extractedBits = extractStylometricBits(contentToDecode);
    if (!extractedBits || extractedBits.length === 0) {
        return null;
    }

    return bitsToDataWithLength(extractedBits);
}

/**
 * Demonstrates the stylometric encoding/decoding process
 * @param originalText The text content to use for the demonstration
 * @param dataToHide The data to hide within the text
 */
export const demonstrateStylometricEncoding = (originalText: string, dataToHide: string): void => {
    console.log("=== STYLOMETRIC ENCODER DEMO (Simplified) ===");
    console.log("Original text:");
    console.log(originalText);
    console.log("\nData to hide:", dataToHide);
    console.log("\n=== ENCODING DATA ===\n");

    const encodedText = hideDataStylometrically(originalText, dataToHide);
    console.log("Text with hidden data (stylometrically encoded):");
    console.log(encodedText);
    console.log("\n=== DECODING DATA ===\n");

    const extractedData = extractHiddenStylometricData(encodedText);
    console.log("Extracted data:", extractedData);
    console.log("Original data: ", dataToHide);

    if (extractedData === null) {
        console.log("Match: ✗ (Extraction failed)");
    } else {
        const extractedRelevant = extractedData.substring(0, dataToHide.length);
        console.log(`Match (up to original length): ${extractedRelevant === dataToHide ? '✓' : '✗'}`);
        if (extractedRelevant !== dataToHide) {
            console.log(`  Expected: "${dataToHide}"`);
            console.log(`  Got:      "${extractedRelevant}"`);
        }
        if (extractedData.length > dataToHide.length) {
            console.log(`  (Extracted data has extra characters: "${extractedData.substring(dataToHide.length)}")`);
        }
    }
    console.log("==============================================");
};

/**
 * Adds unit tests for the simplified stylometric encoder/decoder.
 * Suggestion: Use vitest or jest. Add to a master test suite.
 * Example test structure:
 * describe('Safety Stylometric Encoder (Simplified)', () => {
 *
 *   describe('Length Prefixing', () => {
 *      it('should correctly encode data to bits with length (expected success)', () => {
 *          const data = "Hi"; // 2 bytes = 16 bits
 *          const bits = dataToBitsWithLength(data);
 *          // Expected: 16 bits for length (value 16) + 16 bits for data
 *          expect(bits.length).toBe(32);
 *          // Check length prefix (16 = 0x0010)
 *          const lengthBits = bits.slice(0, 16);
 *          let lengthValue = 0;
 *          for(let i=0; i<16; i++) lengthValue = (lengthValue << 1) | (lengthBits[i]?1:0);
 *          expect(lengthValue).toBe(16);
 *      });
 *
 *      it('should correctly decode bits with length back to data (expected success)', () => {
 *          const data = "Test Data";
 *          const bits = dataToBitsWithLength(data);
 *          const decoded = bitsToDataWithLength(bits);
 *          expect(decoded).toEqual(data);
 *      });
 *
 *       it('should handle incomplete data during decoding (expected success)', () => {
 *          const data = "Test Data";
 *          const bits = dataToBitsWithLength(data);
 *          const truncatedBits = bits.slice(0, bits.length - 5); // Remove last 5 bits
 *          const decoded = bitsToDataWithLength(truncatedBits);
 *          expect(decoded).not.toEqual(data); // Should be truncated
 *          expect(data.startsWith(decoded!)).toBe(true); // Should match the beginning
 *      });
 *
 *       it('should return null if bits are too short for length (expected success)', () => {
 *          const bits = [true, false, true, false]; // Only 4 bits
 *          const decoded = bitsToDataWithLength(bits);
 *          expect(decoded).toBeNull();
 *      });
 *   });
 *
 *   describe('Adverb Carrier Logic', () => {
 *      const text = "This is the first sentence. This sentence actually has the marker. This one does not. This one actually does too.";
 *      const expectedBits = [true, false, true]; // Based on sentences 2, 3, 4
 *
 *      it('should extract bits based on adverb presence (expected success)', () => {
 *          const bits = extractStylometricBits(text);
 *          expect(bits).toEqual(expectedBits);
 *      });
 *
 *      it('should encode bits by adding/removing adverb (expected success)', () => {
 *          const original = "Sentence one. Sentence two needs a marker. Sentence three has one actually and needs it removed. Sentence four needs one.";
 *          const bitsToEncode = [true, false, true];
 *          const { modifiedText, bitsEncoded } = encodeStylometricBits(original, bitsToEncode);
 *
 *          expect(bitsEncoded).toBe(3);
 *          // Check specific sentences for changes (this is approximate)
 *          expect(modifiedText).toContain("Sentence two actually needs"); // Added
 *          expect(modifiedText).not.toContain("three has one actually"); // Removed
 *          expect(modifiedText).toContain("Sentence four actually needs"); // Added
 *
 *          // Verify extraction from modified text
 *          const extracted = extractStylometricBits(modifiedText);
 *          expect(extracted).toEqual(bitsToEncode);
 *      });
 *
 *       it('should handle insufficient sentences for encoding (expected success)', () => {
 *          const original = "One sentence only.";
 *          const bitsToEncode = [true, false];
 *          const { modifiedText, bitsEncoded } = encodeStylometricBits(original, bitsToEncode);
 *          expect(bitsEncoded).toBe(0);
 *          expect(modifiedText).toEqual(original);
 *      });
 *   });
 *
 *   describe('End-to-End', () => {
 *      it('should hide and extract data successfully (expected success)', () => {
 *          const original = "This is a longer piece of text. It contains several sentences. Some of them might actually be modified. Others will remain the same. We need enough capacity for the test data. Let's add one more sentence.";
 *          const data = "Secret!";
 *          const encoded = hideDataStylometrically(original, data);
 *          const extracted = extractHiddenStylometricData(encoded);
 *          expect(extracted).toEqual(data);
 *      });
 *
 *       it('should return null if marker is missing (expected success)', () => {
 *          const textWithoutMarker = "This text has no marker.";
 *          const extracted = extractHiddenStylometricData(textWithoutMarker);
 *          expect(extracted).toBeNull();
 *      });
 *   });
 * });
 */