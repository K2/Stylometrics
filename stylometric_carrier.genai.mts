/**
 * Stylometric Carrier - Information Embedding Using Stylometric Features
 *
 * This module inverts the stylometric detection techniques identified by Kumarage et al. (2023)
 * to create steganographic carriers within text. It exploits stylometric features that typically
 * differentiate human from AI text to instead carry embedded information.
 *
 * Design Goals: Implement functional apply/extract methods for various stylometric techniques.
 * Constraints: Relies on 'compromise' for NLP. Naturalness is secondary to functionality for now. See stylometric_carrier.ApiNotes.md.
 * Paradigm: Functional analysis, Imperative text manipulation.
 *
 * Flow:
 * 1. Analyze text to determine carrying capacity (`analyzeCarryingCapacity` calls `estimate` for each technique).
 * 2. Process payload to distribute across available carriers (`encodePayload` selects and calls `apply`).
 * 3. Apply modifications to embed payload in text (`apply` methods perform changes).
 * 4. Extract payload by reversing the analysis (`extractPayload` calls `extract`).
 *
 * Happy path: analyzeCarryingCapacity -> encodePayload -> extractPayload
 */

// @ts-ignore - compromise types might not be perfectly aligned
import * as nlp from 'compromise';
// @ts-ignore - compromise plugin types might not be perfectly aligned
import compromiseSentences from 'compromise/plugins/sentences';
// @ts-ignore - compromise plugin types might not be perfectly aligned
import compromiseNumbers from 'compromise/plugins/numbers';
// Apply plugins
nlp.plugin(compromiseSentences);
nlp.plugin(compromiseNumbers);

// --- Interfaces (keep as is) ---
export interface CarrierAnalysis {
    totalCapacityBits: number;
    carrierDistribution: {
        phraseology: number;
        punctuation: number;
        linguistic: number;
        readability: number;
    };
    safeModificationRanges: FeatureMap; // Define FeatureMap if not imported
    recommendedMaxPayloadBytes: number;
}

// Define FeatureMap if not importing from detection
export interface FeatureMap {
    [key: string]: number;
}

export interface EncodingOptions {
    usePhraseologyCarriers?: boolean;
    usePunctuationCarriers?: boolean;
    useLinguisticCarriers?: boolean;
    useReadabilityCarriers?: boolean;
    errorCorrection?: boolean; // Note: Error correction happens *before* calling encodePayload
    preserveReadingLevel?: boolean;
    preserveLinguisticFingerprint?: boolean;
    preserveKeyFeatures?: string[];
    maxDetectionRisk?: number; // 0-1 scale
}

export interface CarrierTechnique {
    id: string;
    name: string;
    category: 'phraseology' | 'punctuation' | 'linguistic' | 'readability';
    bitsPerThousandWords: number; // Approximate guide
    apply: (text: string, bits: boolean[]) => { modifiedText: string; bitsEncoded: number };
    extract: (text: string) => boolean[];
    estimate: (text: string) => number; // Estimate capacity in bits
    detectability: number; // 0-1 scale, lower is better
}
// --- End Interfaces ---

// Basic synonym map for substitution carrier
const synonymMap: Record<string, { lowFreq: string, highFreq: string }> = {
    'use': { lowFreq: 'utilize', highFreq: 'use' },
    'help': { lowFreq: 'assist', highFreq: 'help' },
    'show': { lowFreq: 'demonstrate', highFreq: 'show' },
    'start': { lowFreq: 'commence', highFreq: 'start' },
    'end': { lowFreq: 'terminate', highFreq: 'end' },
    'big': { lowFreq: 'substantial', highFreq: 'big' },
    'small': { lowFreq: 'minuscule', highFreq: 'small' },
    'fast': { lowFreq: 'rapid', highFreq: 'fast' },
    'important': { lowFreq: 'crucial', highFreq: 'important' },
    'good': { lowFreq: 'beneficial', highFreq: 'good' },
};

/**
 * StylometricCarrier enables embedding information in text using stylometric features.
 * Design Goals: Provide various techniques (carriers) to encode/decode boolean data
 *               into text by subtly altering stylistic features like punctuation,
 *               sentence structure, word choice, etc. Each carrier should estimate
 *               its capacity and apply/extract bits.
 * Architectural Constraints: Uses 'compromise' NLP library for text analysis.
 *                          Carrier implementations should be self-contained functions.
 *                          Focus on reversibility and minimizing semantic distortion.
 * Happy Path:
 * 1. Instantiate StylometricCarrier.
 * 2. Analyze text capacity using analyzeCarryingCapacity().
 * 3. Encode data using encodeData().
 * 4. Transmit/store modified text.
 * 5. Decode data using decodeData().
 * ApiNotes: ./stylometric_carrier.ApiNotes.md (Assumed)
 */
export class StylometricCarrier {
    private carriers: CarrierTechnique[];

    /**
     * Initialize the stylometric carrier
     */
    constructor() {
        this.carriers = this.initializeCarriers();
    }

    /**
     * Initialize all available carrier techniques
     */
    private initializeCarriers(): CarrierTechnique[] {
        // Reference: ./stylometric_carrier.ApiNotes.md#Initialization
        return [
            this.createSentenceLengthCarrier(),
            this.createParagraphStructureCarrier(),
            this.createPunctuationFrequencyCarrier(),
            this.createQuoteStyleCarrier(),
            this.createOptionalCommaCarrier(),
            this.createSynonymSubstitutionCarrier(),
            this.createFunctionWordCarrier(),
            this.createVoiceStyleCarrier(), // Note: Experimental
            this.createRhymingSynonymCarrier(), // New
            this.createDescriptionDetailCarrier(), // New
            this.createCounterpointPhraseCarrier(), // New
        ];
    }

    /**
     * Analyze carrying capacity of text for embedding information
     *
     * @param text Text to analyze
     * @returns Analysis of carrying capacity
     */
    analyzeCarryingCapacity(text: string): CarrierAnalysis {
        const wordCount = (text.match(/\b\w+\b/g) || []).length;

        // Calculate baseline capacity for each carrier
        const carrierEstimates = this.carriers.map(carrier => ({
            id: carrier.id,
            category: carrier.category,
            bitCapacity: carrier.estimate(text),
            detectability: carrier.detectability
        }));

        // Group by category and sum capacities
        const phraseologyBits = this.sumCarrierBits(carrierEstimates, 'phraseology');
        const punctuationBits = this.sumCarrierBits(carrierEstimates, 'punctuation');
        const linguisticBits = this.sumCarrierBits(carrierEstimates, 'linguistic');
        const readabilityBits = this.sumCarrierBits(carrierEstimates, 'readability');

        // Total capacity
        const totalBits = phraseologyBits + punctuationBits + linguisticBits + readabilityBits;

        // Calculate safe modification ranges (Placeholder - requires feature extractor and thresholds)
        const safeModificationRanges = this.calculateSafeModificationRanges();

        // Max payload assuming ~25% overhead for robustness/naturalness/metadata
        const recommendedMaxPayloadBytes = Math.floor((totalBits * 0.75) / 8);

        return {
            totalCapacityBits: totalBits,
            carrierDistribution: {
                phraseology: phraseologyBits,
                punctuation: punctuationBits,
                linguistic: linguisticBits,
                readability: readabilityBits
            },
            safeModificationRanges,
            recommendedMaxPayloadBytes: Math.max(0, recommendedMaxPayloadBytes) // Ensure non-negative
        };
    }

    /**
     * Sum bits capacity by carrier category
     */
    private sumCarrierBits(
        estimates: Array<{ id: string; category: string; bitCapacity: number; detectability: number }>,
        category: string
    ): number {
        return estimates
            .filter(e => e.category === category)
            .reduce((sum, e) => sum + e.bitCapacity, 0);
    }

    /**
     * Calculate safe ranges for modifying features without triggering detection
     * Placeholder: Requires actual feature extraction and defined thresholds.
     */
    private calculateSafeModificationRanges(): FeatureMap {
        console.warn("StylometricCarrier: calculateSafeModificationRanges is a placeholder.");
        return {};
    }

    /**
     * Encode payload into text using stylometric features as carriers
     *
     * @param text Original text to use as carrier
     * @param payload Binary data to encode (assumed to be already erasure-coded if needed)
     * @param options Encoding options
     * @returns Modified text with embedded payload
     */
    encodePayload(text: string, payload: Uint8Array, options: EncodingOptions = {}): string {
        const analysis = this.analyzeCarryingCapacity(text);
        const payloadBits = payload.length * 8;

        const activeCarriers = this.getActiveCarriers(options);
        const estimatedUsableBits = activeCarriers.reduce((sum, c) => sum + c.estimate(text), 0);

        // Adjust warning threshold, allow slightly over estimation
        if (payloadBits > estimatedUsableBits * 1.2) {
            console.warn(`Payload size (${payloadBits} bits) significantly exceeds estimated capacity (${estimatedUsableBits} bits). Encoding likely to fail or be incomplete.`);
        } else if (payloadBits > estimatedUsableBits) {
            console.warn(`Payload size (${payloadBits} bits) exceeds estimated capacity (${estimatedUsableBits} bits). Encoding may be incomplete.`);
        }

        if (payloadBits === 0) {
            console.warn("Payload is empty, returning original text.");
            return text;
        }
        if (activeCarriers.length === 0) {
            console.error("No active carriers selected or available based on options. Cannot encode.");
            return text; // Return original text if no carriers
        }

        const bits = this.bytesToBits(payload);
        let modifiedText = text;
        let bitsRemaining = [...bits];
        let totalBitsEncoded = 0;

        // Sort carriers by detectability (lower first)
        activeCarriers.sort((a, b) => a.detectability - b.detectability);

        for (const carrier of activeCarriers) {
            if (bitsRemaining.length === 0) break;

            try {
                // Pass only the bits that still need encoding
                const result = carrier.apply(modifiedText, bitsRemaining);
                if (result.modifiedText && typeof result.bitsEncoded === 'number' && result.bitsEncoded >= 0) {
                    modifiedText = result.modifiedText;
                    // Ensure we don't process more bits than were actually encoded by the carrier OR remaining
                    const encodedCount = Math.min(result.bitsEncoded, bitsRemaining.length);
                    if (encodedCount > 0) {
                        bitsRemaining = bitsRemaining.slice(encodedCount);
                        totalBitsEncoded += encodedCount;
                        console.log(`Carrier ${carrier.id} encoded ${encodedCount} bits. ${bitsRemaining.length} bits remaining.`);
                    } else if (result.modifiedText !== text) {
                        console.warn(`Carrier ${carrier.id} modified text but reported 0 bits encoded.`);
                    }
                } else {
                    console.warn(`Carrier ${carrier.id} returned invalid result (modifiedText: ${!!result.modifiedText}, bitsEncoded: ${result.bitsEncoded}). Skipping.`);
                }
            } catch (error) {
                if (error instanceof Error) {
                    console.warn(`Error applying carrier ${carrier.id}: ${error.message}. Skipping.`);
                } else {
                    console.warn(`Unknown error applying carrier ${carrier.id}. Skipping.`);
                }
            }
        }

        if (bitsRemaining.length > 0) {
            console.warn(`Could not encode entire payload: ${bitsRemaining.length} bits remain out of ${payloadBits}. Only ${totalBitsEncoded} bits were encoded.`);
        } else {
            console.log(`Successfully encoded ${totalBitsEncoded} bits.`);
        }

        return modifiedText;
    }

    /**
     * Extract payload from text that was embedded using stylometric carriers
     *
     * @param text Text with embedded payload
     * @param options Extraction options (should match encoding options)
     * @returns Extracted payload (may be incomplete if encoding failed or text was altered)
     */
    extractPayload(text: string, options: EncodingOptions = {}): Uint8Array {
        const activeCarriers = this.getActiveCarriers(options);
        // Ensure extraction order matches encoding order (by detectability)
        activeCarriers.sort((a, b) => a.detectability - b.detectability);

        const extractedBits: boolean[] = [];
        // let expectedBits = -1; // This seems unused

        for (const carrier of activeCarriers) {
            try {
                const bits = carrier.extract(text);
                if (Array.isArray(bits)) {
                    // Assume each carrier extracts only the bits it encoded
                    extractedBits.push(...bits);
                    console.log(`Carrier ${carrier.id} extracted ${bits.length} bits.`);
                } else {
                    console.warn(`Carrier ${carrier.id} returned invalid extraction result (not an array). Skipping.`);
                }
            } catch (error) {
                if (error instanceof Error) {
                    console.warn(`Error extracting from carrier ${carrier.id}: ${error.message}. Skipping.`);
                } else {
                    console.warn(`Unknown error extracting from carrier ${carrier.id}. Skipping.`);
                }
            }
        }

        console.log(`Total extracted bits: ${extractedBits.length}`);
        const payload = this.bitsToBytes(extractedBits);

        return payload;
    }

    /**
     * Get active carriers based on options
     */
    private getActiveCarriers(options: EncodingOptions): CarrierTechnique[] {
        const {
            usePhraseologyCarriers = true,
            usePunctuationCarriers = true,
            useLinguisticCarriers = true,
            useReadabilityCarriers = true,
            maxDetectionRisk = 0.6 // Default max risk
        } = options;

        return this.carriers.filter(carrier => {
            if (carrier.detectability > maxDetectionRisk) {
                console.log(`Carrier ${carrier.id} skipped (detectability ${carrier.detectability} > max ${maxDetectionRisk})`);
                return false;
            }
            switch (carrier.category) {
                case 'phraseology': return usePhraseologyCarriers;
                case 'punctuation': return usePunctuationCarriers;
                case 'linguistic': return useLinguisticCarriers;
                case 'readability': return useReadabilityCarriers;
                default:
                    console.warn(`Carrier ${carrier.id} has unknown category: ${carrier.category}`);
                    return false;
            }
        });
    }

    /**
     * Convert bytes to bits array (Least Significant Bit first in array)
     */
    private bytesToBits(bytes: Uint8Array): boolean[] {
        const bits: boolean[] = [];
        bytes.forEach(byte => {
            for (let j = 0; j < 8; j++) {
                // LSB first: ((byte >> j) & 1) === 1
                // MSB first: ((byte >> (7 - j)) & 1) === 1
                bits.push(((byte >> j) & 1) === 1); // Keep LSB first as originally implemented
            }
        });
        return bits;
    }

    /**
     * Convert bits array to bytes (Assumes LSB first in array)
     */
    private bitsToBytes(bits: boolean[]): Uint8Array {
        const byteCount = Math.ceil(bits.length / 8);
        const bytes = new Uint8Array(byteCount);
        for (let i = 0; i < byteCount; i++) {
            let byte = 0;
            for (let j = 0; j < 8; j++) {
                const bitIndex = i * 8 + j;
                if (bitIndex < bits.length && bits[bitIndex]) {
                    // LSB first: byte |= (1 << j);
                    // MSB first: byte |= (1 << (7 - j));
                    byte |= (1 << j); // Keep LSB first
                }
            }
            bytes[i] = byte;
        }
        return bytes;
    }

    /*** CARRIER TECHNIQUE IMPLEMENTATIONS ***/

    /**
     * Sentence length pattern carrier (Example Implementation)
     * Encodes bits by adding/removing low-impact adverbs like 'truly', 'really'.
     * This is a simplified example and may not be robust or natural.
     */
    private createSentenceLengthCarrier(): CarrierTechnique {
        const id = 'sentence_length_adverb'; // More specific ID
        const bitsPerThousandWords = 5; // Rough estimate
        const detectability = 0.4; // Moderate detectability

        return {
            id, name: 'Sentence Adverb Marker', category: 'phraseology',
            bitsPerThousandWords, detectability,

            estimate: (text: string) => {
                try {
                    // Estimate capacity based on number of sentences where modification is possible
                    const doc = nlp(text);
                    const sentences = doc.sentences().out('array');
                    // Capacity: roughly one bit per sentence (can add or potentially remove marker)
                    return Math.max(0, sentences.length - 1); // Avoid modifying first sentence maybe?
                } catch (e) {
                    console.warn(`[${id}] Estimate error: ${e instanceof Error ? e.message : e}`);
                    return 0;
                }
            },

            apply: (text: string, bits: boolean[]) => {
                let modifiedText = text;
                let bitsEncoded = 0;
                try {
                    const doc = nlp(modifiedText);
                    const sentences = doc.sentences();
                    const sentenceTexts = sentences.out('array');
                    const numSentences = sentenceTexts.length;
                    let currentBitIndex = 0;

                    if (numSentences < 1 || bits.length === 0) {
                        return { modifiedText, bitsEncoded: 0 };
                    }

                    const modifiedSentences = [...sentenceTexts];
                    const markerAdverb = ' truly'; // Adverb used as marker (space included)
                    const markerRegex = /\s+(truly|really|actually|just)\b/i; // Regex to find markers

                    // Iterate through sentences (start from 0 or 1?) to apply bits
                    for (let i = 0; i < numSentences && currentBitIndex < bits.length; i++) {
                        const currentSentence = modifiedSentences[i];
                        const bit = bits[currentBitIndex];
                        let modifiedCurrent = currentSentence;
                        let appliedModification = false;

                        if (bit) { // Encode 1: Ensure marker is present
                            if (!markerRegex.test(currentSentence)) {
                                // Add marker after the first word (simple approach)
                                modifiedCurrent = currentSentence.replace(/^(\s*\w+)/, `$1${markerAdverb}`);
                                if (modifiedCurrent !== currentSentence) {
                                    appliedModification = true;
                                }
                            } else {
                                // Marker already present, bit '1' is represented. No change needed.
                                // We should still count this as encoding the bit.
                                appliedModification = true; // Represents the bit '1' state
                            }
                        } else { // Encode 0: Ensure marker is absent
                            const match = currentSentence.match(markerRegex);
                            if (match) {
                                // Remove the first found marker
                                modifiedCurrent = currentSentence.replace(markerRegex, '');
                                if (modifiedCurrent !== currentSentence) {
                                    appliedModification = true;
                                }
                            } else {
                                // Marker already absent, bit '0' is represented. No change needed.
                                appliedModification = true; // Represents the bit '0' state
                            }
                        }

                        // If a conceptual modification occurred (state matches bit), count bit and update text
                        if (appliedModification) {
                            modifiedSentences[i] = modifiedCurrent; // Update sentence text even if no change needed
                            bitsEncoded++;
                            currentBitIndex++;
                        } else {
                            // Could not apply the bit (e.g., adding marker failed?)
                            // Skip this sentence for encoding? Or log warning?
                            console.warn(`[${id}] Could not apply bit ${bit} to sentence ${i}. Skipping.`);
                        }
                    }
                    // Reconstruct the text from modified sentences
                    // This simple join might lose original spacing between sentences.
                    // A more robust approach would use sentence start/end indices.
                    modifiedText = modifiedSentences.join(' '); // Simple join

                } catch (error) {
                    console.warn(`[${id}] Apply error: ${error instanceof Error ? error.message : error}`);
                    // Return potentially partially modified text and bits encoded so far
                    return { modifiedText, bitsEncoded };
                }
                return { modifiedText, bitsEncoded };
            },

            extract: (text: string) => {
                const extractedBits: boolean[] = [];
                try {
                    const doc = nlp(text);
                    const sentences = doc.sentences().out('array');
                    const numSentences = sentences.length;
                    const markerRegex = /\s+(truly|really|actually|just)\b/i; // Use same regex as apply

                    if (numSentences < 1) return [];

                    // Extract bit from each sentence based on marker presence
                    for (let i = 0; i < numSentences; i++) {
                        const currentSentence = sentences[i];
                        const hasMarker = markerRegex.test(currentSentence);
                        extractedBits.push(hasMarker); // true (1) if marker present, false (0) otherwise
                    }
                } catch (error) {
                    console.warn(`[${id}] Extract error: ${error instanceof Error ? error.message : error}`);
                }
                // Return all extracted bits; decoder needs to know how many were originally encoded.
                return extractedBits;
            }
        };
    }

    /**
     * Rhyming Synonym Carrier
     * Encodes bits by swapping between a word and a rhyming synonym.
     * Note: Relies on a predefined map and simple rhyme checking. Highly experimental.
     * Reference: ./stylometric_carrier.ApiNotes.md#RhymingSynonym
     * [paradigm:imperative]
     */
    private createRhymingSynonymCarrier(): CarrierTechnique {
        const id = 'rhyming_synonym';
        const bitsPerThousandWords = 3; // Low capacity, depends heavily on map
        const detectability = 0.7; // Potentially high detectability if rhymes are forced/unnatural

        // Simple map of words to potential rhyming synonyms.
        // Bit 1 = use rhyme, Bit 0 = use original/non-rhyme
        // This map needs to be carefully curated for naturalness.
        const rhymeSwapMap: Record<string, { original: string, rhyme: string }> = {
            'fast': { original: 'fast', rhyme: 'vast' }, // Example pair
            'quick': { original: 'quick', rhyme: 'slick' }, // Example pair
            'big': { original: 'big', rhyme: 'dig' }, // May change meaning significantly
            'large': { original: 'large', rhyme: 'charge' }, // May change meaning
            'happy': { original: 'happy', rhyme: 'sappy' }, // Changes connotation
            'sad': { original: 'sad', rhyme: 'mad' } // Changes connotation
        };
        // Create reverse lookup for extraction
        const reverseRhymeMap: Record<string, { original: string, isRhyme: boolean }> = {};
        for (const key in rhymeSwapMap) {
            const pair = rhymeSwapMap[key];
            reverseRhymeMap[pair.original] = { original: pair.original, isRhyme: false };
            reverseRhymeMap[pair.rhyme] = { original: pair.original, isRhyme: true };
        }

        return {
            id, name: 'Rhyming Synonym Substitution', category: 'linguistic',
            bitsPerThousandWords, detectability,

            estimate: (text: string) => {
                try {
                    const doc = nlp(text);
                    let count = 0;
                    const potentialWords = Object.keys(reverseRhymeMap);
                    doc.terms().forEach(term => {
                        const word = term.text('reduced');
                        if (potentialWords.includes(word)) {
                            count++;
                        }
                    });
                    return count;
                } catch (e) {
                    console.warn(`[${id}] Estimate error: ${e instanceof Error ? e.message : e}`);
                    return 0;
                }
            },

            apply: (text: string, bits: boolean[]) => {
                let modifiedText = text;
                let bitsEncoded = 0;
                let currentBitIndex = 0;
                const wordBoundaries: { start: number, end: number, original: string, replacement?: string }[] = [];

                try {
                    const doc = nlp(text);
                    doc.terms().forEach(term => {
                        const word = term.text('reduced');
                        const originalCase = term.text();

                        if (reverseRhymeMap[word] && currentBitIndex < bits.length) {
                            const mapping = reverseRhymeMap[word];
                            const baseOriginal = mapping.original;
                            const currentIsRhyme = mapping.isRhyme;
                            const targetRhyme = rhymeSwapMap[baseOriginal]?.rhyme;
                            const targetOriginal = rhymeSwapMap[baseOriginal]?.original;

                            // Ensure we have both original and rhyme defined for the base word
                            if (!targetRhyme || !targetOriginal) {
                                console.warn(`[${id}] Incomplete rhyme map for base word derived from '${word}'. Skipping.`);
                                return; // Skip if map is inconsistent
                            }

                            const bit = bits[currentBitIndex];
                            let replacement = originalCase; // Default: no change
                            let applied = false;

                            if (bit) { // Encode 1: Use rhyme
                                if (!currentIsRhyme) {
                                    replacement = targetRhyme;
                                    applied = true;
                                } else {
                                    applied = true; // Already the rhyme
                                }
                            } else { // Encode 0: Use original/non-rhyme
                                if (currentIsRhyme) {
                                    replacement = targetOriginal;
                                    applied = true;
                                } else {
                                    applied = true; // Already the original
                                }
                            }

                            if (applied) {
                                // Preserve case roughly (simple capitalization)
                                if (originalCase === originalCase.toUpperCase()) {
                                    replacement = replacement.toUpperCase();
                                } else if (originalCase[0] === originalCase[0].toUpperCase()) {
                                    replacement = replacement.charAt(0).toUpperCase() + replacement.slice(1);
                                }

                                const pointer = term.pointer?.[0];
                                if (pointer) {
                                    wordBoundaries.push({ start: pointer[0], end: pointer[1], original: originalCase, replacement: replacement });
                                }
                                bitsEncoded++;
                                currentBitIndex++;
                            }
                        }
                    });

                    // Apply changes in reverse order
                    wordBoundaries.sort((a, b) => b.start - a.start);
                    for (const change of wordBoundaries) {
                        if (change.replacement && change.original !== change.replacement) {
                            modifiedText = modifiedText.substring(0, change.start) + change.replacement + modifiedText.substring(change.end);
                        }
                    }
                } catch (error) {
                    console.warn(`[${id}] Apply error: ${error instanceof Error ? error.message : error}`);
                }
                return { modifiedText, bitsEncoded };
            },

            extract: (text: string) => {
                const extractedBits: boolean[] = [];
                try {
                    const doc = nlp(text);
                    doc.terms().forEach(term => {
                        const word = term.text('reduced');
                        if (reverseRhymeMap[word]) {
                            extractedBits.push(reverseRhymeMap[word].isRhyme);
                        }
                    });
                } catch (error) {
                    console.warn(`[${id}] Extract error: ${error instanceof Error ? error.message : error}`);
                }
                return extractedBits;
            }
        };
    }

    /**
     * Description Detail Carrier
     * Encodes bits by adding/removing specific descriptive adjectives or adverbs.
     * Aims to subtly increase/decrease descriptive density.
     * Reference: ./stylometric_carrier.ApiNotes.md#DescriptionDetail
     * [paradigm:imperative]
     */
    private createDescriptionDetailCarrier(): CarrierTechnique {
        const id = 'description_detail';
        const bitsPerThousandWords = 4; // Moderate capacity potential
        const detectability = 0.4; // Moderate, depends on naturalness of additions

        // Map of nouns/verbs to potential optional descriptors (adjective/adverb)
        // Bit 1 = Add descriptor, Bit 0 = Ensure descriptor is absent
        const detailMap: Record<string, { word: string, descriptor: string, position: 'before' | 'after' }> = {
            'car': { word: 'car', descriptor: 'shiny', position: 'before' }, // e.g., "shiny car"
            'ran': { word: 'ran', descriptor: 'quickly', position: 'after' }, // e.g., "ran quickly"
            'house': { word: 'house', descriptor: 'old', position: 'before' }, // e.g., "old house"
            'walked': { word: 'walked', descriptor: 'slowly', position: 'after' }, // e.g., "walked slowly"
            'sky': { word: 'sky', descriptor: 'blue', position: 'before' }, // e.g., "blue sky" - careful with common pairs
        };
        const allDescriptors = Object.values(detailMap).map(d => d.descriptor);
        const descriptorRegex = new RegExp(`\\b(${allDescriptors.join('|')})\\b`, 'gi');

        return {
            id, name: 'Description Detail Level', category: 'phraseology',
            bitsPerThousandWords, detectability,

            estimate: (text: string) => {
                try {
                    const doc = nlp(text);
                    let count = 0;
                    const targetWords = Object.keys(detailMap);
                    doc.terms().forEach(term => {
                        const word = term.text('reduced');
                        if (targetWords.includes(word)) {
                            count++;
                        }
                        if (allDescriptors.includes(word)) {
                            count++;
                        }
                    });
                    return Math.max(0, Math.floor(count / 2));
                } catch (e) {
                    console.warn(`[${id}] Estimate error: ${e instanceof Error ? e.message : e}`);
                    return 0;
                }
            },

            apply: (text: string, bits: boolean[]) => {
                let modifiedText = text;
                let bitsEncoded = 0;
                let currentBitIndex = 0;
                const edits: { start: number, end: number, text: string }[] = [];

                try {
                    const doc = nlp(text);
                    doc.terms().forEach((term, i) => {
                        const word = term.text('reduced');
                        const originalTermText = term.text();

                        if (detailMap[word] && currentBitIndex < bits.length) {
                            const detail = detailMap[word];
                            const descriptor = detail.descriptor;
                            const position = detail.position;
                            const bit = bits[currentBitIndex];
                            let applied = false;
                            let existingDescriptorTerm: any = null;

                            if (position === 'before') {
                                const prevTerm = doc.terms().get(i - 1);
                                if (prevTerm && prevTerm.text('reduced') === descriptor) {
                                    existingDescriptorTerm = prevTerm;
                                }
                            } else {
                                const nextTerm = doc.terms().get(i + 1);
                                if (nextTerm && nextTerm.text('reduced') === descriptor) {
                                    existingDescriptorTerm = nextTerm;
                                }
                            }

                            const pointer = term.pointer?.[0];
                            if (!pointer) return;

                            if (bit) {
                                if (!existingDescriptorTerm) {
                                    if (position === 'before') {
                                        edits.push({ start: pointer[0], end: pointer[0], text: `${descriptor} ` });
                                    } else {
                                        edits.push({ start: pointer[1], end: pointer[1], text: ` ${descriptor}` });
                                    }
                                    applied = true;
                                } else {
                                    applied = true;
                                }
                            } else {
                                if (existingDescriptorTerm) {
                                    const descPointer = existingDescriptorTerm.pointer?.[0];
                                    if (descPointer) {
                                        let start = descPointer[0];
                                        let end = descPointer[1];
                                        if (position === 'before' && text[end] === ' ') end++;
                                        if (position === 'after' && text[start - 1] === ' ') start--;
                                        edits.push({ start: start, end: end, text: '' });
                                        applied = true;
                                    }
                                } else {
                                    applied = true;
                                }
                            }

                            if (applied) {
                                bitsEncoded++;
                                currentBitIndex++;
                            }
                        }
                    });

                    edits.sort((a, b) => b.start - a.start);
                    for (const edit of edits) {
                        modifiedText = modifiedText.substring(0, edit.start) + edit.text + modifiedText.substring(edit.end);
                    }

                } catch (error) {
                    console.warn(`[${id}] Apply error: ${error instanceof Error ? error.message : error}`);
                }
                return { modifiedText, bitsEncoded };
            },

            extract: (text: string) => {
                const extractedBits: boolean[] = [];
                try {
                    const doc = nlp(text);
                    doc.terms().forEach((term, i) => {
                        const word = term.text('reduced');
                        if (detailMap[word]) {
                            const detail = detailMap[word];
                            const descriptor = detail.descriptor;
                            const position = detail.position;
                            let hasDescriptor = false;

                            if (position === 'before') {
                                const prevTerm = doc.terms().get(i - 1);
                                if (prevTerm && prevTerm.text('reduced') === descriptor) {
                                    hasDescriptor = true;
                                }
                            } else {
                                const nextTerm = doc.terms().get(i + 1);
                                if (nextTerm && nextTerm.text('reduced') === descriptor) {
                                    hasDescriptor = true;
                                }
                            }
                            extractedBits.push(hasDescriptor);
                        }
                    });
                } catch (error) {
                    console.warn(`[${id}] Extract error: ${error instanceof Error ? error.message : error}`);
                }
                return extractedBits;
            }
        };
    }

    /**
     * Counterpoint Phrase Carrier
     * Encodes bits by adding/removing short counterpoint phrases (e.g., "however", "on the other hand").
     * Reference: ./stylometric_carrier.ApiNotes.md#CounterpointPhrase
     * [paradigm:imperative]
     */
    private createCounterpointPhraseCarrier(): CarrierTechnique {
        const id = 'counterpoint_phrase';
        const bitsPerThousandWords = 2; // Low capacity
        const detectability = 0.5; // Moderate, depends on context

        // Phrases to add/remove. Bit 1 = Add, Bit 0 = Remove.
        // Position: 'start' of sentence, 'before_conjunction' (like but/yet)
        const counterpointPhrases = [
            { phrase: 'However, ', position: 'start' },
            { phrase: 'On the other hand, ', position: 'start' },
            { phrase: ', though, ', position: 'middle' }
        ];
        const allPhrasesText = counterpointPhrases.map(p => p.phrase.trim().replace(/,/g, ''));
        const phraseRegex = new RegExp(`\\b(${allPhrasesText.join('|')})\\b`, 'gi');

        return {
            id, name: 'Counterpoint Phrase Presence', category: 'phraseology',
            bitsPerThousandWords, detectability,

            estimate: (text: string) => {
                try {
                    const doc = nlp(text);
                    const sentences = doc.sentences().length;
                    const conjunctions = doc.match('#Conjunction').length;
                    let existing = 0;
                    let match;
                    while ((match = phraseRegex.exec(text)) !== null) {
                        existing++;
                    }
                    const estimate = Math.floor(sentences / 2) + conjunctions + existing;
                    return Math.max(0, estimate);
                } catch (e) {
                    console.warn(`[${id}] Estimate error: ${e instanceof Error ? e.message : e}`);
                    return 0;
                }
            },

            apply: (text: string, bits: boolean[]) => {
                let modifiedText = text;
                let bitsEncoded = 0;
                let currentBitIndex = 0;
                const edits: { start: number, end: number, text: string }[] = [];

                try {
                    const doc = nlp(modifiedText);
                    const sentences = doc.sentences();

                    sentences.forEach((sentence, sentenceIndex) => {
                        if (currentBitIndex >= bits.length) return;

                        const sentenceText = sentence.text();
                        const phraseInfo = counterpointPhrases[sentenceIndex % counterpointPhrases.length];
                        const phrase = phraseInfo.phrase;
                        const phraseTrimmed = phrase.trim().replace(/,/g, '');
                        const bit = bits[currentBitIndex];
                        let applied = false;

                        const hasPhrase = sentenceText.toLowerCase().includes(phraseTrimmed.toLowerCase());
                        const pointer = sentence.pointer?.[0];
                        if (!pointer) return;

                        if (bit) {
                            if (!hasPhrase) {
                                if (phraseInfo.position === 'start') {
                                    edits.push({ start: pointer[0], end: pointer[0], text: phrase });
                                    applied = true;
                                } else if (phraseInfo.position === 'middle') {
                                    const commaIndex = sentenceText.indexOf(',');
                                    const insertPos = commaIndex !== -1 ? commaIndex : Math.floor(sentenceText.length / 2);
                                    const actualInsertPos = pointer[0] + insertPos;
                                    edits.push({ start: actualInsertPos, end: actualInsertPos, text: phrase });
                                    applied = true;
                                }
                            } else {
                                applied = true;
                            }
                        } else {
                            if (hasPhrase) {
                                const regex = new RegExp(phrase.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&'), 'gi');
                                const tempText = sentenceText.replace(regex, '');
                                edits.push({ start: pointer[0], end: pointer[1], text: tempText });
                                applied = true;
                            } else {
                                applied = true;
                            }
                        }

                        if (applied) {
                            bitsEncoded++;
                            currentBitIndex++;
                        }
                    });

                    edits.sort((a, b) => b.start - a.start);
                    for (const edit of edits) {
                        if (edit.start >= 0 && edit.end <= modifiedText.length && edit.start <= edit.end) {
                            modifiedText = modifiedText.substring(0, edit.start) + edit.text + modifiedText.substring(edit.end);
                        } else {
                            console.warn(`[${id}] Invalid edit indices, skipping:`, edit);
                        }
                    }

                } catch (error) {
                    console.warn(`[${id}] Apply error: ${error instanceof Error ? error.message : error}`);
                }
                return { modifiedText, bitsEncoded };
            },

            extract: (text: string) => {
                const extractedBits: boolean[] = [];
                try {
                    const doc = nlp(text);
                    const sentences = doc.sentences();

                    sentences.forEach((sentence, sentenceIndex) => {
                        const sentenceText = sentence.text();
                        const phraseInfo = counterpointPhrases[sentenceIndex % counterpointPhrases.length];
                        const phraseTrimmed = phraseInfo.phrase.trim().replace(/,/g, '');

                        const hasPhrase = sentenceText.toLowerCase().includes(phraseTrimmed.toLowerCase());
                        extractedBits.push(hasPhrase);
                    });
                } catch (error) {
                    console.warn(`[${id}] Extract error: ${error instanceof Error ? error.message : error}`);
                }
                return extractedBits;
            }
        };
    }
}

/**
 * Demonstrate the stylometric carrier capabilities
 *
 * @param text Sample text to use as carrier
 * @param payload Sample payload to embed
 */
export function demonstrateStylometricCarrier(
    text: string = "This is a sample text for demonstrating stylometric carriers. It contains several sentences, uses common words like 'use' and 'help', and includes punctuation like commas, and quotes \"like this\". We can try to embed data.",
    payload: string = "secret"
): void {
    console.log("=== STYLOMETRIC CARRIER DEMO ===");

    const carrier = new StylometricCarrier();

    console.log("\n1. ANALYZING CARRYING CAPACITY:");
    console.log("-------------------------------");
    let analysis: CarrierAnalysis;
    try {
        analysis = carrier.analyzeCarryingCapacity(text);
        console.log(`Total estimated capacity: ${analysis.totalCapacityBits} bits (~${Math.floor(analysis.totalCapacityBits / 8)} bytes)`);
        console.log("Carrier distribution (estimated bits):");
        console.log(`- Phraseology: ${analysis.carrierDistribution.phraseology}`);
        console.log(`- Punctuation: ${analysis.carrierDistribution.punctuation}`);
        console.log(`- Linguistic: ${analysis.carrierDistribution.linguistic}`);
        console.log(`- Readability: ${analysis.carrierDistribution.readability}`);
        console.log(`Recommended max payload: ${analysis.recommendedMaxPayloadBytes} bytes`);
    } catch (e) {
        console.error("Error during analysis:", e.message);
        analysis = {
            totalCapacityBits: 0, carrierDistribution: { phraseology: 0, punctuation: 0, linguistic: 0, readability: 0 },
            safeModificationRanges: {}, recommendedMaxPayloadBytes: 0
        };
    }

    const encoder = new TextEncoder();
    const payloadBytes = encoder.encode(payload);
    console.log(`\nPayload to encode: "${payload}" (${payloadBytes.length} bytes)`);

    if (payloadBytes.length > analysis.recommendedMaxPayloadBytes && analysis.recommendedMaxPayloadBytes > 0) {
        console.warn(`Payload size (${payloadBytes.length} bytes) exceeds recommended capacity (${analysis.recommendedMaxPayloadBytes} bytes). Encoding might be lossy or fail.`);
    } else if (analysis.totalCapacityBits === 0) {
        console.error("Estimated capacity is zero. Cannot encode payload.");
        console.log("\n=== DEMO COMPLETE (Encoding Skipped) ===");
        return;
    }

    try {
        const options: EncodingOptions = {
            usePhraseologyCarriers: true,
            usePunctuationCarriers: true,
            useLinguisticCarriers: true,
            useReadabilityCarriers: false,
            maxDetectionRisk: 0.7
        };

        console.log("\n2. ENCODING PAYLOAD:");
        console.log("--------------------");
        console.log("Original text:\n", text);

        const modifiedText = carrier.encodePayload(text, payloadBytes, options);
        console.log("\nModified text:\n", modifiedText);
        if (modifiedText === text) {
            console.warn("Encoding did not modify the text. Payload might be too large or no suitable carriers found/applied.");
        }

        console.log("\n3. EXTRACTING PAYLOAD:");
        console.log("----------------------");
        const extractedBytes = carrier.extractPayload(modifiedText, options);

        const decoder = new TextDecoder("utf-8", { fatal: false, ignoreBOM: true });
        const relevantExtractedBytes = extractedBytes.slice(0, payloadBytes.length);
        const extractedPayload = decoder.decode(relevantExtractedBytes).replace(/\u0000/g, '');

        console.log(`Extracted payload (first ${payloadBytes.length} bytes): "${extractedPayload}"`);

        const originalBits = carrier.bytesToBits(payloadBytes);
        const extractedBits = carrier.bytesToBits(extractedBytes);
        let correctBits = 0;
        for (let i = 0; i < originalBits.length && i < extractedBits.length; ++i) {
            if (originalBits[i] === extractedBits[i]) {
                correctBits++;
            }
        }
        const accuracy = originalBits.length > 0 ? (correctBits / originalBits.length * 100).toFixed(1) : "N/A";
        console.log(`Bit-level accuracy (approx): ${accuracy}% (${correctBits}/${originalBits.length} correct bits)`);
        console.log(`Extraction matches original payload: ${extractedPayload === payload ? 'YES' : 'NO'`);

    } catch (error) {
        console.error(`\nError during encoding/extraction demo: ${error.message}`);
        console.error(error.stack);
    }

    console.log("\n=== DEMO COMPLETE ===");
    console.log("Note: Accuracy depends heavily on the text, payload size, and the heuristic nature of some carriers.");
}