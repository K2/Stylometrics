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
import nlp from 'compromise';
// @ts-ignore - compromise plugin types might not be perfectly aligned
import compromiseSentences from 'compromise-sentences';
// @ts-ignore - compromise plugin types might not be perfectly aligned
import compromiseNumbers from 'compromise-numbers';
// Apply plugins safely
if (typeof (nlp as any).plugin === 'function') {
    (nlp as any).plugin(compromiseSentences);
    (nlp as any).plugin(compromiseNumbers);
} else {
    console.warn("compromise.plugin is not available. NLP features might be limited.");
}

// --- Interfaces ---
// Using types from central file now
import type { CarrierTechnique, FeatureMap } from './types/CarrierTypes.ts'; // Corrected import path with extension and type-only
import { StyleFeatureExtractor } from './stylometric_detection.genai.mts'; // Import for safe modification ranges

export interface CarrierAnalysis {
    totalCapacityBits: number;
    carrierDistribution: {
        phraseology: number;
        punctuation: number;
        linguistic: number;
        readability: number;
    };
    safeModificationRanges: FeatureMap;
    recommendedMaxPayloadBytes: number;
}

// Define FeatureMap if not importing from detection - Now imported from CarrierTypes

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
    // Add more pairs for better capacity
    'make': { lowFreq: 'construct', highFreq: 'make' },
    'get': { lowFreq: 'obtain', highFreq: 'get' },
    'see': { lowFreq: 'observe', highFreq: 'see' },
    'think': { lowFreq: 'consider', highFreq: 'think' },
};

// Function word map for swapping (Example pairs)
// TODO: Implement actual swapping in createFunctionWordCarrier
const functionWordMap: Record<string, { alt1: string, alt2: string }> = {
    'about': { alt1: 'regarding', alt2: 'concerning' },
    'after': { alt1: 'following', alt2: 'subsequent to' },
    'before': { alt1: 'prior to', alt2: 'ahead of' },
    'but': { alt1: 'however', alt2: 'although' }, // Be careful with semantic changes
    'because': { alt1: 'since', alt2: 'as' },
    'if': { alt1: 'provided that', alt2: 'assuming' },
    'like': { alt1: 'similar to', alt2: 'such as' }, // Context dependent
    'so': { alt1: 'therefore', alt2: 'consequently' },
    'though': { alt1: 'despite', alt2: 'even if' },
    'until': { alt1: 'till', alt2: 'up to' },
    'when': { alt1: 'while', alt2: 'at the time that' },
    'while': { alt1: 'whereas', alt2: 'during the time that' },
};

// Rhyming map (Example pairs - very basic)
// TODO: Implement actual swapping in createRhymingSynonymCarrier
const rhymeMap: Record<string, { rhyme1: string, rhyme2: string }> = {
    'bright': { rhyme1: 'light', rhyme2: 'slight' },
    'day': { rhyme1: 'way', rhyme2: 'say' },
    'night': { rhyme1: 'sight', rhyme2: 'might' },
    'ground': { rhyme1: 'found', rhyme2: 'sound' },
    'care': { rhyme1: 'share', rhyme2: 'rare' },
    'mind': { rhyme1: 'find', rhyme2: 'kind' },
    'heart': { rhyme1: 'start', rhyme2: 'part' },
    'play': { rhyme1: 'stay', rhyme2: 'gray' },
    'time': { rhyme1: 'rhyme', rhyme2: 'climb' },
    'year': { rhyme1: 'dear', rhyme2: 'clear' },
};

// Description detail map (Example modifiers)
// TODO: Implement actual swapping/addition in createDescriptionDetailCarrier
const descriptionMap: Record<string, { mod1: string, mod2: string }> = {
    'sky': { mod1: 'vast', mod2: 'blue' },
    'building': { mod1: 'tall', mod2: 'old' },
    'car': { mod1: 'fast', mod2: 'red' },
    'person': { mod1: 'kind', mod2: 'tall' },
    'dog': { mod1: 'loyal', mod2: 'small' },
    'cat': { mod1: 'sly', mod2: 'soft' },
    'tree': { mod1: 'green', mod2: 'large' },
    'city': { mod1: 'busy', mod2: 'bright' },
    'road': { mod1: 'long', mod2: 'dark' },
    'water': { mod1: 'clear', mod2: 'deep' },
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
 * 3. Encode data using encodePayload().
 * 4. Transmit/store modified text.
 * 5. Decode data using decodePayload().
 * ApiNotes: ./stylometric_carrier.ApiNotes.md (Assumed)
 */
export class StylometricCarrier {
    private carriers: CarrierTechnique[];
    private featureExtractor: StyleFeatureExtractor; // Added for safe ranges

    /**
     * Initialize the stylometric carrier
     */
    constructor() {
        this.carriers = this.initializeCarriers();
        this.featureExtractor = new StyleFeatureExtractor(); // Initialize feature extractor
    }

    /**
     * Get all available carrier techniques
     * @returns Array of carrier techniques
     */
    getCarriers(): CarrierTechnique[] {
        return [...this.carriers]; // Return a copy
    }

    /**
     * Get available carriers for external use or testing
     */
    getAvailableCarriers(): CarrierTechnique[] {
        return [...this.carriers]; // Return a copy
    }

    /**
     * Initialize all available carrier techniques
     */
    private initializeCarriers(): CarrierTechnique[] {
        const initializedCarriers: CarrierTechnique[] = [];
        try {
            initializedCarriers.push(this.createSentenceLengthCarrier());
            initializedCarriers.push(this.createParagraphStructureCarrier());
            initializedCarriers.push(this.createPunctuationFrequencyCarrier());
            initializedCarriers.push(this.createOptionalCommaCarrier());
            initializedCarriers.push(this.createSynonymSubstitutionCarrier());
            initializedCarriers.push(this.createFunctionWordCarrier());
            initializedCarriers.push(this.createVoiceStyleCarrier());
            initializedCarriers.push(this.createRhymingSynonymCarrier());
            initializedCarriers.push(this.createDescriptionDetailCarrier());
            // Note: QuoteStyleCarrier is separate and registered by CarrierMatrix by default
        } catch (e) {
            console.error(`Error during carrier initialization: ${(e as Error).message}`);
        }
        return initializedCarriers;
    }

    /**
     * Analyze carrying capacity of text for embedding information
     *
     * @param text Text to analyze
     * @returns Analysis of carrying capacity
     */
    analyzeCarryingCapacity(text: string): CarrierAnalysis {
        const wordCount = (text.match(/\b\w+\b/g) || []).length;
        if (wordCount === 0) {
            return {
                totalCapacityBits: 0,
                carrierDistribution: { phraseology: 0, punctuation: 0, linguistic: 0, readability: 0 },
                safeModificationRanges: {},
                recommendedMaxPayloadBytes: 0
            };
        }

        const carrierEstimates = this.carriers.map(carrier => {
            let bitCapacity = 0;
            try {
                // Use getCapacity if available, otherwise estimate
                bitCapacity = typeof carrier.getCapacity === 'function'
                    ? carrier.getCapacity(text)
                    : carrier.estimate(text);
            } catch (e) {
                console.warn(`Error estimating capacity for carrier ${carrier.id}: ${(e as Error).message}`);
            }
            return {
                id: carrier.id,
                category: carrier.category,
                bitCapacity: Math.max(0, bitCapacity),
                detectability: carrier.getDetectability ? carrier.getDetectability() : 0.5 // Use getter, provide default
            };
        });

        const phraseologyBits = this.sumCarrierBits(carrierEstimates, 'phraseology');
        const punctuationBits = this.sumCarrierBits(carrierEstimates, 'punctuation');
        const linguisticBits = this.sumCarrierBits(carrierEstimates, 'linguistic');
        const readabilityBits = this.sumCarrierBits(carrierEstimates, 'readability');

        const totalBits = phraseologyBits + punctuationBits + linguisticBits + readabilityBits;

        const safeModificationRanges = this.calculateSafeModificationRanges(text);

        // Recommend slightly less than total capacity to account for estimation errors/interactions
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
            recommendedMaxPayloadBytes: Math.max(0, recommendedMaxPayloadBytes)
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
     * Implementation: Requires actual feature extraction and defined thresholds.
     * This version provides a basic structure.
     * @param text The input text to analyze.
     * @returns A map of feature names to their safe modification range (e.g., +/- value).
     */
    private calculateSafeModificationRanges(text: string): FeatureMap {
        // TODO: Define actual thresholds based on detection model sensitivity.
        // These are just illustrative placeholders.
        const thresholds = {
            mean_words_per_sentence: 0.1, // Allow 10% variation
            stdev_words_per_sentence: 0.15,
            mean_words_per_paragraph: 0.15,
            stdev_words_per_paragraph: 0.2,
            mean_sentences_per_paragraph: 0.1,
            stdev_sentences_per_paragraph: 0.15,
            total_punctuation_freq: 0.1, // Relative frequency
            comma_period_ratio: 0.2,
            ttr: 0.05, // Absolute difference
            mattr: 0.05, // Absolute difference
            mean_word_length: 0.05, // Absolute difference in chars
            stdev_word_length: 0.1,
            mean_syllables_per_word: 0.1, // Absolute difference in syllables
            stdev_syllables_per_word: 0.15,
            flesch_reading_ease: 5.0, // Absolute difference in score
        };

        const safeRanges: FeatureMap = {};
        try {
            const currentFeatures = this.featureExtractor.extractAllFeatures(text);

            for (const feature in thresholds) {
                if (currentFeatures.hasOwnProperty(feature)) {
                    const currentValue = currentFeatures[feature];
                    const threshold = thresholds[feature as keyof typeof thresholds];

                    // Calculate delta based on threshold type (relative or absolute)
                    let delta: number;
                    if (feature.includes('freq') || feature.includes('ratio') || feature.includes('per_')) {
                        // Relative threshold for frequencies and ratios
                        delta = Math.abs(currentValue * threshold);
                    } else {
                        // Absolute threshold for counts, lengths, scores, TTR/MATTR
                        delta = threshold;
                    }

                    // Store the calculated safe delta
                    safeRanges[`safeDelta_${feature}`] = delta;
                    // Optionally store the current value as well
                    safeRanges[feature] = currentValue;
                }
            }
        } catch (e) {
            console.error(`Error calculating features for safe ranges: ${(e as Error).message}`);
        }

        return safeRanges;
    }

    /**
     * Encode payload into text using stylometric features as carriers
     *
     * @param text Original text to use as carrier
     * @param payload Binary data to encode
     * @param options Encoding options
     * @returns Modified text with embedded payload
     */
    encodePayload(text: string, payload: Uint8Array, options: EncodingOptions = {}): string {
        let processedPayload = payload;
        // --- Error Correction Pre-processing ---
        if (options.errorCorrection) {
            console.warn("Error correction pre-processing requested but not implemented.");
            // TODO: Implement payload processing using an error correction library (e.g., Reed-Solomon)
            // Example: processedPayload = applyErrorCorrectionEncoding(payload);
            // This would likely increase the size of `processedPayload`.
        }
        // --- End Error Correction Pre-processing ---

        const payloadBits = processedPayload.length * 8;
        const activeCarriers = this.getActiveCarriers(options);
        const estimatedUsableBits = activeCarriers.reduce((sum, c) => {
            try {
                 // Use getCapacity if available, otherwise estimate
                 const capacity = typeof c.getCapacity === 'function'
                    ? c.getCapacity(text)
                    : c.estimate(text);
                 return sum + Math.max(0, capacity);
            } catch (e) {
                console.warn(`Error estimating capacity for carrier ${c.id} during encoding: ${(e as Error).message}`);
                return sum;
            }
        }, 0);

        if (payloadBits > estimatedUsableBits * 1.2) { // Increased threshold for warning
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
            return text;
        }

        const bits = this.bytesToBits(processedPayload);
        let modifiedText = text;
        let bitsRemaining = [...bits];
        let totalBitsEncoded = 0;

        // Sort carriers by detectability (lowest first) to prioritize less detectable changes
        activeCarriers.sort((a, b) => (a.getDetectability ? a.getDetectability() : 0.5) - (b.getDetectability ? b.getDetectability() : 0.5));

        for (const carrier of activeCarriers) {
            if (bitsRemaining.length === 0) break;

            try {
                // Pass all remaining bits and let the carrier consume what it can.
                // Use encode if available, otherwise apply
                const applyFn = typeof carrier.encode === 'function' ? carrier.encode : carrier.apply;
                const result = applyFn(modifiedText, bitsRemaining);

                // Validate result structure
                if (result && typeof result.modifiedText === 'string' && typeof result.bitsEncoded === 'number' && result.bitsEncoded >= 0) {
                    // How many bits did the carrier *actually* use from the input `bitsRemaining`?
                    const consumedBits = Math.min(result.bitsEncoded, bitsRemaining.length);

                    if (consumedBits > 0) {
                        modifiedText = result.modifiedText;
                        bitsRemaining = bitsRemaining.slice(consumedBits);
                        totalBitsEncoded += consumedBits;
                        console.log(`Carrier ${carrier.id} encoded ${consumedBits} bits. ${bitsRemaining.length} bits remaining.`);
                    } else if (result.modifiedText !== modifiedText && result.bitsEncoded === 0) {
                        // If text changed but no bits reported encoded, update text but log warning
                        modifiedText = result.modifiedText;
                        console.warn(`Carrier ${carrier.id} modified text but reported 0 bits encoded.`);
                    } else if (result.bitsEncoded < 0) {
                         console.warn(`Carrier ${carrier.id} reported negative bits encoded (${result.bitsEncoded}). Skipping.`);
                    }
                    // If bitsEncoded > 0 but consumedBits is 0 (because bitsRemaining was empty), do nothing.
                } else {
                    console.warn(`Carrier ${carrier.id} returned invalid result structure or negative bitsEncoded. Skipping.`);
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
        // IMPORTANT: Extraction order must match encoding order (lowest detectability first)
        activeCarriers.sort((a, b) => (a.getDetectability ? a.getDetectability() : 0.5) - (b.getDetectability ? b.getDetectability() : 0.5));

        const extractedBits: boolean[] = [];
        let currentTextState = text; // Text state doesn't change during extraction

        for (const carrier of activeCarriers) {
            try {
                const bits = carrier.extract(currentTextState);
                if (Array.isArray(bits)) {
                    // Assume bits extracted by each carrier are sequential and concatenated
                    extractedBits.push(...bits);
                    console.log(`Carrier ${carrier.id} extracted ${bits.length} potential bits.`);
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
        let finalPayload = this.bitsToBytes(extractedBits);

        // --- Error Correction Post-processing ---
        if (options.errorCorrection) {
            console.warn("Error correction post-processing requested but not implemented.");
            // TODO: Implement payload decoding using an error correction library
            // Example: finalPayload = applyErrorCorrectionDecoding(finalPayload);
            // This might return the original payload size or null/error if decoding fails.
        }
        // --- End Error Correction Post-processing ---

        return finalPayload;
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
            const detectability = carrier.getDetectability ? carrier.getDetectability() : 0.5; // Use getter, provide default
            if (detectability > maxDetectionRisk) {
                // console.log(`Carrier ${carrier.id} skipped (detectability ${detectability} > max ${maxDetectionRisk})`);
                return false;
            }
            switch (carrier.category) {
                case 'phraseology': return usePhraseologyCarriers;
                case 'punctuation': return usePunctuationCarriers;
                case 'linguistic': return useLinguisticCarriers;
                case 'readability': return useReadabilityCarriers;
                default:
                    const categoryString = typeof carrier.category === 'string' ? carrier.category : '[unknown type]';
                    console.warn(`Carrier ${carrier.id} has unknown or unsupported category: ${categoryString}`);
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
                // Push LSB first for each byte
                bits.push(((byte >> j) & 1) === 1);
            }
        });
        return bits;
    }

    /**
     * Convert bits array to bytes (Assumes LSB first in array)
     * Handles cases where bit length is not a multiple of 8 by padding the last byte with zeros.
     */
    private bitsToBytes(bits: boolean[]): Uint8Array {
        if (!bits || bits.length === 0) {
            return new Uint8Array(0);
        }
        const byteCount = Math.ceil(bits.length / 8);
        const bytes = new Uint8Array(byteCount);
        for (let i = 0; i < byteCount; i++) {
            let byte = 0;
            for (let j = 0; j < 8; j++) {
                const bitIndex = i * 8 + j;
                if (bitIndex < bits.length && bits[bitIndex]) {
                    // Set the j-th bit if the corresponding boolean is true (LSB mapping)
                    byte |= (1 << j);
                }
            }
            bytes[i] = byte;
        }
        return bytes;
    }

    /*** CARRIER TECHNIQUE IMPLEMENTATIONS ***/
    // TODO: Review and refine all carrier implementations for correctness, robustness, and naturalness.
    // TODO: Implement actual logic for function_word, voice_style, rhyming_synonym, description_detail carriers.

    private createSentenceLengthCarrier(): CarrierTechnique {
        const id = 'sentence_length_adverb';
        const name = 'Sentence Adverb Marker';
        const category = 'phraseology';
        const bitsPerThousandWords = 5; // Low capacity
        const detectability = 0.4;
        const markerAdverb = 'truly'; // Example marker, could be configurable
        const markerRegex = new RegExp(`\\b${markerAdverb}\\b`, 'i');
        const MIN_SENTENCE_WORDS = 6; // Avoid modifying very short sentences

        const estimate = (text: string): number => {
            try {
                const doc = (nlp as any)(text);
                 if (!doc || typeof doc.sentences !== 'function') return 0;
                const sentences = doc.sentences().out('array');
                let potentialSites = 0;
                sentences.forEach((s: string) => {
                    // Count sentences long enough to be modified
                    if (s.split(/\s+/).length >= MIN_SENTENCE_WORDS) {
                        potentialSites++;
                    }
                });
                return Math.max(0, potentialSites);
            } catch (e) {
                console.error(`Error estimating for ${id}: ${(e as Error).message}`);
                return 0;
            }
        };

        const apply = (text: string, bits: boolean[]): { modifiedText: string; bitsEncoded: number } => {
            let modifiedText = text;
            let bitsEncoded = 0;
            try {
                const doc = (nlp as any)(text);
                 if (!doc || typeof doc.sentences !== 'function') return { modifiedText: text, bitsEncoded: 0 };
                const sentences = doc.sentences();
                let currentBitIndex = 0;
                const sentenceTexts: string[] = [];
                let lastEnd = 0;

                sentences.forEach((sentence: any) => {
                     // Preserve original whitespace/structure between sentences
                     const originalSentenceWithTrailing = text.substring(lastEnd, sentence.pointer[0][1]);
                     const trailingSpace = originalSentenceWithTrailing.substring(sentence.text().length);
                     lastEnd = sentence.pointer[0][1];

                    if (currentBitIndex >= bits.length) {
                        sentenceTexts.push(sentence.text() + trailingSpace);
                        return;
                    }

                    let currentSentenceText = sentence.text();
                    const words = currentSentenceText.split(/\s+/); // Split by whitespace

                    // Check if this sentence is a potential site
                    if (words.length < MIN_SENTENCE_WORDS) {
                        sentenceTexts.push(currentSentenceText + trailingSpace);
                        return; // Skip short sentences, do not consume bit
                    }

                    // --- Site identified, consume bit ---
                    const bit = bits[currentBitIndex];
                    const hasMarker = markerRegex.test(currentSentenceText);
                    let madeChange = false;

                    if (bit && !hasMarker) {
                        // Insert marker adverb (e.g., after first verb or near beginning)
                        // Simple insertion for now: after the first word if possible
                        const insertPos = 1; // Insert after the first word
                        if (words.length > insertPos) {
                             // Preserve original case of the word following the insertion? - Less critical for adverbs
                             words.splice(insertPos, 0, markerAdverb);
                             currentSentenceText = words.join(' ');
                             madeChange = true;
                        } else {
                            // Cannot insert, maybe append? Or just skip? Skip for now.
                            console.warn(`Cannot insert adverb in short sentence: "${currentSentenceText}"`);
                        }
                    } else if (!bit && hasMarker) {
                        // Remove marker adverb
                        currentSentenceText = currentSentenceText.replace(markerRegex, '').replace(/\s{2,}/g, ' ').trim();
                        madeChange = true;
                    }

                    // Always count the bit if it was a potential site
                    bitsEncoded++;
                    currentBitIndex++;
                    sentenceTexts.push(currentSentenceText + trailingSpace);
                });
                 // Add any remaining text after the last sentence
                 if (lastEnd < text.length) {
                     sentenceTexts.push(text.substring(lastEnd));
                 }

                modifiedText = sentenceTexts.join('');

            } catch (e) {
                console.error(`Error applying ${id}: ${(e as Error).message}`);
                // Return 0 bits encoded on error, but potentially modified text if error occurred mid-way
                return { modifiedText: modifiedText, bitsEncoded: 0 };
            }
            // Ensure bitsEncoded doesn't exceed the number of bits provided
            bitsEncoded = Math.min(bitsEncoded, bits.length);
            return { modifiedText, bitsEncoded };
        };

        const extract = (text: string): boolean[] => {
            const extractedBits: boolean[] = [];
            try {
                const doc = (nlp as any)(text);
                 if (!doc || typeof doc.sentences !== 'function') return [];
                doc.sentences().forEach((sentence: any) => {
                    const currentSentenceText = sentence.text();
                    // Only extract from sentences that were potential sites during encoding
                    if (currentSentenceText.split(/\s+/).length >= MIN_SENTENCE_WORDS) {
                        const hasMarker = markerRegex.test(currentSentenceText);
                        extractedBits.push(hasMarker);
                    }
                });
            } catch (e) {
                console.error(`Error extracting ${id}: ${(e as Error).message}`);
                return [];
            }
            return extractedBits;
        };

        // Return the CarrierTechnique object
        return {
            id,
            name,
            category,
            bitsPerThousandWords,
            detectability,
            estimate,
            apply,
            extract,
            encode: apply, // Alias
            getDetectability: () => detectability,
            getCapacity: estimate, // Alias
            getNaturalness: () => 0.7, // Adverbs can sometimes feel forced
            getRobustness: () => 0.6, // Simple word changes might survive some edits
            // --- Default/Placeholder implementations for any missing required fields ---
            analyze: (text: string) => ({ capacity: estimate(text) }), // Basic analyze implementation
            estimateCapacity: estimate, // Alias
        };
    }


    private createSynonymSubstitutionCarrier(): CarrierTechnique {
        const id = 'synonym_substitution';
        const name = 'Synonym Substitution';
        const category = 'linguistic';
        const bitsPerThousandWords = 10; // Depends heavily on map size and text
        const detectability = 0.35; // Can be detected by frequency analysis

        const estimate = (text: string): number => {
            let count = 0;
            try {
                // Use regex to find whole words matching keys or values in the map
                const allWords = Object.keys(synonymMap).flatMap(k => [synonymMap[k].lowFreq, synonymMap[k].highFreq]);
                const uniqueWords = [...new Set(allWords)];
                if (uniqueWords.length === 0) return 0;

                // Create a regex to find any of the target words
                const regex = new RegExp(`\\b(${uniqueWords.join('|')})\\b`, 'gi');
                const matches = text.match(regex);
                count = matches ? matches.length : 0;

            } catch (e) {
                console.error(`Error estimating for ${id}: ${(e as Error).message}`);
                return 0;
            }
            return count;
        };

        const apply = (text: string, bits: boolean[]): { modifiedText: string; bitsEncoded: number } => {
            let bitsEncoded = 0;
            let currentBitIndex = 0;
            // Process text segment by segment to preserve non-word characters and whitespace
            // Regex includes word boundaries, words, and non-words/whitespace
            const segments = text.split(/(\b\w+'?\w*\b|[^\w\s]+|\s+)/g).filter(s => s); // Improved word regex, keep separators

            const modifiedSegments = segments.map(segment => {
                if (currentBitIndex >= bits.length) return segment;

                // Check if the segment is a word
                if (!/\b\w+'?\w*\b/.test(segment)) return segment;

                const lowerWord = segment.toLowerCase();
                let mappingInfo: { original: string, lowFreq: string, highFreq: string } | null = null;
                let currentForm: 'low' | 'high' | null = null;

                // Find if this word (or its alternative) is in our map
                if (synonymMap[lowerWord]) {
                    // Word is a key (high frequency) in the map
                    mappingInfo = { original: lowerWord, ...synonymMap[lowerWord] };
                    currentForm = 'high';
                } else {
                    // Check if word is a value (low or high frequency) in the map
                    for (const original in synonymMap) {
                        if (synonymMap[original].lowFreq === lowerWord) {
                            mappingInfo = { original: original, ...synonymMap[original] };
                            currentForm = 'low';
                            break;
                        }
                         if (synonymMap[original].highFreq === lowerWord) {
                             // This case means the word is a high-freq value but not a key.
                             // This implies the key itself might be low-freq, which is unusual for the setup.
                             // However, handle it for completeness.
                             mappingInfo = { original: original, ...synonymMap[original] };
                             currentForm = 'high';
                             break;
                         }
                    }
                }


                if (mappingInfo && currentForm) {
                    // --- Site identified, consume bit ---
                    const targetBit = bits[currentBitIndex]; // true for lowFreq, false for highFreq
                    const targetForm = targetBit ? 'low' : 'high';

                    let replacement = segment; // Default to original segment
                    let madeChange = false;

                    if (targetForm !== currentForm) {
                         replacement = targetBit ? mappingInfo.lowFreq : mappingInfo.highFreq;
                         madeChange = true;

                         // Attempt to preserve original capitalization
                         if (segment.length > 0 && segment[0] === segment[0].toUpperCase()) {
                             if (segment.length === 1 || segment.slice(1) === segment.slice(1).toLowerCase()) {
                                 // Title case (e.g., "Use" -> "Utilize")
                                 replacement = replacement.charAt(0).toUpperCase() + replacement.slice(1);
                             } else if (segment === segment.toUpperCase()) {
                                 // ALL CAPS (e.g., "USE" -> "UTILIZE")
                                 replacement = replacement.toUpperCase();
                             }
                             // else: Mixed case (e.g., "useThis") - leave replacement as lowercase (safer default)
                         }
                    }

                    // Consume the bit for this potential site
                    bitsEncoded++;
                    currentBitIndex++;
                    return replacement; // Return the (potentially modified) segment
                }
                // If the word is not in our map, return the original segment
                return segment;
            });
             // Ensure bitsEncoded doesn't exceed the number of bits provided
             bitsEncoded = Math.min(bitsEncoded, bits.length);

            return { modifiedText: modifiedSegments.join(''), bitsEncoded };
        };

        const extract = (text: string): boolean[] => {
            const extractedBits: boolean[] = [];
            try {
                const segments = text.split(/(\b\w+'?\w*\b|[^\w\s]+|\s+)/g).filter(s => s);

                segments.forEach(segment => {
                    if (!/\b\w+'?\w*\b/.test(segment)) return;

                    const lowerWord = segment.toLowerCase();
                    // Check if the word matches a low or high frequency form in the map
                    for (const original in synonymMap) {
                        const mapping = synonymMap[original];
                        if (lowerWord === mapping.lowFreq) {
                            extractedBits.push(true); // lowFreq represents bit 1 (true)
                            return; // Move to next segment once a match is found
                        }
                        if (lowerWord === mapping.highFreq) {
                            extractedBits.push(false); // highFreq represents bit 0 (false)
                            return; // Move to next segment
                        }
                    }
                });
            } catch (e) {
                console.error(`Error extracting ${id}: ${(e as Error).message}`);
                return [];
            }
            return extractedBits;
        };

        return {
            id,
            name,
            category,
            bitsPerThousandWords,
            detectability,
            estimate,
            apply,
            extract,
            encode: apply,
            getDetectability: () => detectability,
            getCapacity: estimate,
            getNaturalness: () => 0.8, // Generally good if synonyms fit context
            getRobustness: () => 0.7, // Robust to minor edits unless the specific word is changed
            // --- Default/Placeholder implementations ---
            analyze: (text: string) => ({ capacity: estimate(text) }),
            estimateCapacity: estimate,
        };
    }

    private createParagraphStructureCarrier(): CarrierTechnique {
        const id = 'paragraph_structure';
        const name = 'Paragraph Structure Modifier';
        const category = 'phraseology';
        const bitsPerThousandWords = 2; // Very low capacity
        const detectability = 0.5; // Can be noticeable if structure changes drastically
        const MIN_PARA_WORDS = 15; // Minimum words for a paragraph to be considered for merging/splitting

        const estimate = (text: string): number => {
            try {
                // Find pairs of adjacent paragraphs that are both long enough
                const paragraphs = text.split(/\n\s*\n/).filter(p => p.trim().length > 0);
                let potentialSites = 0;
                for (let i = 0; i < paragraphs.length - 1; i++) {
                    if (paragraphs[i].split(/\s+/).length >= MIN_PARA_WORDS &&
                        paragraphs[i + 1].split(/\s+/).length >= MIN_PARA_WORDS) {
                        potentialSites++;
                    }
                }
                // Could also estimate potential splits for long paragraphs, but merging is simpler
                return potentialSites;
            } catch (e) {
                console.error(`Error estimating for ${id}: ${(e as Error).message}`);
                return 0;
            }
        };

        const apply = (text: string, bits: boolean[]): { modifiedText: string; bitsEncoded: number } => {
            if (bits.length === 0) return { modifiedText: text, bitsEncoded: 0 };

            const paragraphs = text.split(/(\n\s*\n)/); // Keep separators
            const resultSegments: string[] = [];
            let bitsEncoded = 0;
            let currentBitIndex = 0;
            let i = 0;

            while (i < paragraphs.length) {
                const currentPara = paragraphs[i]?.trim() || '';
                const separator = paragraphs[i+1] || ''; // Separator after currentPara
                const nextPara = paragraphs[i+2]?.trim() || ''; // Next actual paragraph content

                // Check if current and next paragraphs form a potential site
                const isSite = currentPara.split(/\s+/).length >= MIN_PARA_WORDS &&
                               nextPara.split(/\s+/).length >= MIN_PARA_WORDS &&
                               currentBitIndex < bits.length;

                if (isSite) {
                    // --- Site identified, consume bit ---
                    const bit = bits[currentBitIndex];
                    if (bit) { // Merge paragraphs (bit 1)
                        // Append current paragraph, replace separator with single space, append next paragraph
                        resultSegments.push(currentPara + " " + nextPara);
                        // Skip currentPara, separator, nextPara, nextSeparator (if exists)
                        i += (paragraphs[i+3] !== undefined ? 4 : 3);
                    } else { // Keep paragraphs separate (bit 0)
                        resultSegments.push(currentPara + separator);
                        // Skip only currentPara and separator
                        i += 2;
                    }
                    bitsEncoded++;
                    currentBitIndex++;
                } else {
                    // Not a site, just append the current paragraph and its separator (if they exist)
                    if (paragraphs[i] !== undefined) resultSegments.push(paragraphs[i]);
                    if (paragraphs[i+1] !== undefined) resultSegments.push(paragraphs[i+1]);
                     // Move past currentPara and separator
                    i += 2;
                }
            }
             // Ensure bitsEncoded doesn't exceed the number of bits provided
             bitsEncoded = Math.min(bitsEncoded, bits.length);

            return {
                modifiedText: resultSegments.join(""),
                bitsEncoded
            };
        };

        const extract = (text: string): boolean[] => {
            // Extraction is inherently unreliable as it requires guessing the original structure.
            // This carrier is likely NOT suitable for robust extraction without markers.
            console.error(`Extraction logic for ${id} is unreliable and not implemented. Returning empty array.`);
            return [];
        };

        return {
            id,
            name,
            category,
            bitsPerThousandWords,
            detectability,
            estimate,
            apply,
            extract,
            encode: apply,
            getDetectability: () => detectability,
            getCapacity: estimate,
            getNaturalness: () => 0.6, // Merging/splitting can affect flow
            getRobustness: () => 0.7, // Robust unless paragraphs are heavily edited/reformatted
            // --- Default/Placeholder implementations ---
            analyze: (text: string) => ({ capacity: estimate(text) }),
            estimateCapacity: estimate,
        };
    }

    private createPunctuationFrequencyCarrier(): CarrierTechnique {
        // Re-implemented as Optional Semicolon vs Comma/Period.
        const id = 'semicolon_vs_comma';
        const name = 'Semicolon vs Comma/Period';
        const category = 'punctuation';
        const bitsPerThousandWords = 8;
        const detectability = 0.3;

        // Find places where two related independent clauses are joined by a comma + conjunction (and, but, or, so, for, nor, yet)
        // or could be joined by a semicolon.
        const findPotentialSites = (text: string): { index: number, type: 'comma' | 'semicolon', conjunction?: string, length: number }[] => {
            const sites: { index: number, type: 'comma' | 'semicolon', conjunction?: string, length: number }[] = [];
            try {
                const doc = nlp(text);
                 if (!doc || typeof doc.sentences !== 'function') return [];
                doc.sentences().forEach((sentence: any) => {
                    const sentenceText = sentence.text();
                    const sentenceStartOffset = sentence.pointer?.[0]?.[0] ?? 0; // Get sentence start index in full text

                    // Look for comma + conjunction joining potential clauses
                    // Regex: comma, optional whitespace, conjunction, required whitespace
                    const commaConjRegex = /,\s*(and|but|or|so|for|nor|yet)\s+/gi;
                    let match;
                    while ((match = commaConjRegex.exec(sentenceText)) !== null) {
                        const indexInSentence = match.index;
                        const matchText = match[0];
                        const conjunction = match[1];
                        // Basic check: are parts before and after reasonably long?
                        const before = sentenceText.substring(0, indexInSentence);
                        const after = sentenceText.substring(indexInSentence + matchText.length);
                        if (before.split(/\s+/).length > 3 && after.split(/\s+/).length > 3) {
                            // Index points to the comma
                            sites.push({ index: sentenceStartOffset + indexInSentence, type: 'comma', conjunction: conjunction, length: matchText.length });
                        }
                    }

                    // Look for existing semicolons joining clauses
                    const semicolonRegex = /;\s*/g; // Semicolon followed by optional whitespace
                     while ((match = semicolonRegex.exec(sentenceText)) !== null) {
                         const indexInSentence = match.index;
                         const matchText = match[0];
                         const before = sentenceText.substring(0, indexInSentence);
                         const after = sentenceText.substring(indexInSentence + matchText.length);
                         if (before.split(/\s+/).length > 3 && after.split(/\s+/).length > 3) {
                             // Index points to the semicolon
                             sites.push({ index: sentenceStartOffset + indexInSentence, type: 'semicolon', length: matchText.length });
                         }
                     }
                });
            } catch(e) {
                 console.error(`Error finding sites for ${id}: ${(e as Error).message}`);
            }
            // Sort sites by index
            sites.sort((a, b) => a.index - b.index);
            return sites;
        };


        const estimate = (text: string): number => {
            try {
                return findPotentialSites(text).length;
            } catch (e) {
                console.error(`Error estimating for ${id}: ${(e as Error).message}`);
                return 0;
            }
        };

        const apply = (text: string, bits: boolean[]): { modifiedText: string; bitsEncoded: number } => {
            if (bits.length === 0) return { modifiedText: text, bitsEncoded: 0 };

            const sites = findPotentialSites(text);
            let modifiedText = text;
            let bitsEncoded = 0;
            let currentBitIndex = 0;
            let indexAdjustment = 0;

            // Apply changes from end to start to avoid index issues
            for (let i = sites.length - 1; i >= 0; i--) {
                if (currentBitIndex >= bits.length) break;

                const site = sites[i];
                // --- Site identified, consume bit ---
                const bit = bits[currentBitIndex]; // true for semicolon, false for comma/period
                const targetType = bit ? 'semicolon' : 'comma'; // 'comma' here means comma/period replacement
                let madeChange = false;
                let replacement = "";
                let originalLength = site.length;

                const adjustedIndex = site.index + indexAdjustment;

                if (targetType === 'semicolon' && site.type === 'comma') {
                    // Replace ", conjunction " with "; "
                    replacement = "; "; // Ensure space after semicolon
                    madeChange = true;
                } else if (targetType === 'comma' && site.type === 'semicolon') {
                    // Replace "; " with ". " (simplest, grammatically safe)
                    // Using period is generally safer than trying to infer a conjunction.
                    replacement = ". "; // Ensure space after period
                    madeChange = true;
                }

                if (madeChange) {
                     modifiedText = modifiedText.substring(0, adjustedIndex) + replacement + modifiedText.substring(adjustedIndex + originalLength);
                     indexAdjustment += (replacement.length - originalLength);
                }

                // Consume bit for this site
                bitsEncoded++;
                currentBitIndex++;
            }
             // Ensure bitsEncoded doesn't exceed the number of bits provided
             bitsEncoded = Math.min(bitsEncoded, bits.length);

            return { modifiedText, bitsEncoded };
        };

        const extract = (text: string): boolean[] => {
            const extractedBits: boolean[] = [];
            const sites = findPotentialSites(text); // Find sites in the potentially modified text

            sites.forEach(site => {
                // If the site uses a semicolon, bit is true (1)
                // If the site uses a comma (presumably with conjunction) or period (replacement), bit is false (0)
                extractedBits.push(site.type === 'semicolon');
            });

            return extractedBits;
        };

        return {
            id,
            name,
            category,
            bitsPerThousandWords,
            detectability,
            estimate,
            apply,
            extract,
            encode: apply,
            getDetectability: () => detectability,
            getCapacity: estimate,
            getNaturalness: () => 0.7, // Semicolon use varies; replacing with period is safe but might change rhythm
            getRobustness: () => 0.6, // Fairly robust unless punctuation is heavily edited
            // --- Default/Placeholder implementations ---
            analyze: (text: string) => ({ capacity: estimate(text) }),
            estimateCapacity: estimate,
        };
    }

    private createOptionalCommaCarrier(): CarrierTechnique {
        // Uses commas after introductory elements (e.g., "However, ...") or before conjunctions in lists.
        const id = 'optional_comma';
        const name = 'Optional Comma Modifier';
        const category = 'punctuation';
        const bitsPerThousandWords = 12; // Moderate capacity
        const detectability = 0.2; // Very low, often stylistic choice

        // Find potential sites: introductory words/phrases, items in lists before 'and'/'or'
        const findPotentialSites = (text: string): { index: number, type: 'intro' | 'list', hasComma: boolean, length: number }[] => {
             const sites: { index: number, type: 'intro' | 'list', hasComma: boolean, length: number }[] = [];
             try {
                 // Introductory words/phrases (simplified regex)
                 // Matches word, optional comma, optional space
                 const introRegex = /\b(However|Moreover|Therefore|Consequently|Nevertheless|Furthermore|Meanwhile|Instead|Thus|Finally|First|Second|Next|Then|Also)\b(,?)\s*/gi;
                 let match;
                 while ((match = introRegex.exec(text)) !== null) {
                     const introWord = match[1];
                     const comma = match[2];
                     const hasComma = comma === ',';
                     // Index points *after* the intro word, where comma would be/is
                     // Length is 1 if comma exists, 0 otherwise (for replacement logic)
                     sites.push({ index: match.index + introWord.length, type: 'intro', hasComma, length: hasComma ? 1 : 0 });
                 }

                 // Commas before 'and'/'or' in lists of 3+ items (Oxford comma)
                 // Regex: item, comma, space, (optional item, comma, space)+, conjunction, space, last_item
                 // This is complex and prone to errors. Let's try a simpler regex targeting the comma before the final conjunction.
                 // Matches: comma, optional space, (and|or), required space
                 const oxfordRegex = /,\s*(and|or)\s+/gi;
                 while ((match = oxfordRegex.exec(text)) !== null) {
                     // Check if this looks like a list ending
                     const index = match.index;
                     const textBefore = text.substring(0, index);
                     // Heuristic: Check if there's another comma relatively close before this one
                     const lastCommaBefore = textBefore.lastIndexOf(',');
                     if (lastCommaBefore !== -1 && (index - lastCommaBefore) < 50) { // Arbitrary proximity check
                         // Index points to the comma
                         // Length is 1 (just the comma)
                         sites.push({ index: index, type: 'list', hasComma: true, length: 1 });
                     }
                 }
                 // Also need to find potential sites *without* the Oxford comma
                 // Matches: word boundary, non-comma/space char, space, (and|or), space
                 const noOxfordRegex = /(\b\S)\s+(and|or)\s+/gi;
                  while ((match = noOxfordRegex.exec(text)) !== null) {
                      // Check if this looks like a list ending without an Oxford comma
                      const indexBeforeConj = match.index + match[1].length; // Index after the word preceding conjunction
                      const textBefore = text.substring(0, indexBeforeConj);
                      const lastCommaBefore = textBefore.lastIndexOf(',');
                      if (lastCommaBefore !== -1 && (indexBeforeConj - lastCommaBefore) < 50) {
                          // Index points to the space *before* the conjunction where comma *could* be added
                          // Length is 0 (no comma exists)
                          sites.push({ index: indexBeforeConj, type: 'list', hasComma: false, length: 0 });
                      }
                  }


             } catch (e) {
                 console.error(`Error finding sites for ${id}: ${(e as Error).message}`);
             }
             sites.sort((a, b) => a.index - b.index);
             // Deduplicate sites that might overlap or be very close
             const uniqueSites: { index: number, type: 'intro' | 'list', hasComma: boolean, length: number }[] = [];
             const seenIndices = new Set<number>();
             for(const site of sites) {
                 // Avoid adding sites too close to each other (e.g., intro word followed by list)
                 let tooClose = false;
                 for (const seenIndex of seenIndices) {
                     if (Math.abs(site.index - seenIndex) < 2) {
                         tooClose = true;
                         break;
                     }
                 }
                 if (!seenIndices.has(site.index) && !tooClose) {
                     uniqueSites.push(site);
                     seenIndices.add(site.index);
                 }
             }
             return uniqueSites;
        };

        const estimate = (text: string): number => {
            try {
                return findPotentialSites(text).length;
            } catch (e) {
                console.error(`Error estimating for ${id}: ${(e as Error).message}`);
                return 0;
            }
        };

        const apply = (text: string, bits: boolean[]): { modifiedText: string; bitsEncoded: number } => {
            if (bits.length === 0) return { modifiedText: text, bitsEncoded: 0 };

            const sites = findPotentialSites(text);
            let modifiedText = text;
            let bitsEncoded = 0;
            let currentBitIndex = 0;
            let indexAdjustment = 0;

            // Apply changes from end to start
            for (let i = sites.length - 1; i >= 0; i--) {
                if (currentBitIndex >= bits.length) break;

                const site = sites[i];
                // --- Site identified, consume bit ---
                const bit = bits[currentBitIndex]; // true for comma, false for no comma
                const targetHasComma = bit;
                let madeChange = false;
                let replacement = "";
                let originalLength = site.length; // Length of the comma (1) or 0 if no comma

                const adjustedIndex = site.index + indexAdjustment;

                if (targetHasComma && !site.hasComma) {
                    // Add comma (and potentially a space if needed)
                    let charAfter = modifiedText[adjustedIndex];
                    // If adding before space (intro case) or before conjunction (list case), just add comma.
                    // If adding where no space exists (unlikely for intro), add ", ".
                    replacement = (charAfter === ' ' || site.type === 'list') ? "," : ", ";
                    originalLength = 0; // We are inserting
                    madeChange = true;
                } else if (!targetHasComma && site.hasComma) {
                    // Remove comma (and potentially adjacent space)
                    replacement = ""; // Remove the comma
                    originalLength = 1; // Length of the comma
                    // Check if there's a space immediately after the comma and remove it too
                    if (modifiedText[adjustedIndex + 1] === ' ') {
                        originalLength = 2; // Remove comma and space
                    }
                    madeChange = true;
                }

                if (madeChange) {
                     modifiedText = modifiedText.substring(0, adjustedIndex) + replacement + modifiedText.substring(adjustedIndex + originalLength);
                     indexAdjustment += (replacement.length - originalLength);
                }

                // Consume bit for this site
                bitsEncoded++;
                currentBitIndex++;
            }
             // Ensure bitsEncoded doesn't exceed the number of bits provided
             bitsEncoded = Math.min(bitsEncoded, bits.length);

            return { modifiedText, bitsEncoded };
        };

        const extract = (text: string): boolean[] => {
            const extractedBits: boolean[] = [];
            const sites = findPotentialSites(text); // Find sites in the potentially modified text

            sites.forEach(site => {
                extractedBits.push(site.hasComma); // Bit is true if comma exists, false otherwise
            });

            return extractedBits;
        };

        return {
            id,
            name,
            category,
            bitsPerThousandWords,
            detectability,
            estimate,
            apply,
            extract,
            encode: apply,
            getDetectability: () => detectability,
            getCapacity: estimate,
            getNaturalness: () => 0.9, // Usually acceptable style variation
            getRobustness: () => 0.6, // Robust unless formatting tools enforce specific comma rules
            // --- Default/Placeholder implementations ---
            analyze: (text: string) => ({ capacity: estimate(text) }),
            estimateCapacity: estimate,
        };
    }

    private createFunctionWordCarrier(): CarrierTechnique {
        const id = 'function_word';
        const name = 'Function Word Swapper';
        const category = 'linguistic';
        const bitsPerThousandWords = 15; // Potentially high capacity
        const detectability = 0.35; // Can alter naturalness if choices are poor

        const estimate = (text: string): number => {
            try {
                let count = 0;
                // Consider only words that have alternatives defined in the map
                const targetWords = Object.keys(functionWordMap).flatMap(k => [k, functionWordMap[k].alt1, functionWordMap[k].alt2]);
                const uniqueWords = [...new Set(targetWords)];
                 if (uniqueWords.length === 0) return 0;
                const regex = new RegExp(`\\b(${uniqueWords.join('|')})\\b`, 'gi');
                const matches = text.match(regex);
                count = matches ? matches.length : 0;
                return Math.max(0, count);
            } catch (e) {
                console.error(`Error estimating for ${id}: ${(e as Error).message}`);
                return 0;
            }
        };

        const apply = (text: string, bits: boolean[]): { modifiedText: string; bitsEncoded: number } => {
            console.warn(`Carrier ${id} apply() uses a simple map and may affect naturalness.`);
            if (bits.length === 0) return { modifiedText: text, bitsEncoded: 0 };

            let modifiedText = text;
            let bitsEncoded = 0;
            let currentBitIndex = 0;
            let indexAdjustment = 0;

            const targetWords = Object.keys(functionWordMap).flatMap(k => [k, functionWordMap[k].alt1, functionWordMap[k].alt2]);
            const uniqueWords = [...new Set(targetWords)];
             if (uniqueWords.length === 0) return { modifiedText: text, bitsEncoded: 0 };
            const regex = new RegExp(`\\b(${uniqueWords.join('|')})\\b`, 'gi');

            const matches: { word: string, index: number, length: number }[] = [];
            let match;
            while ((match = regex.exec(text)) !== null) {
                matches.push({ word: match[0].toLowerCase(), index: match.index, length: match[0].length }); // Store lowercase word and original length
            }

            // Apply changes from end to start
            for (let i = matches.length - 1; i >= 0; i--) {
                 if (currentBitIndex >= bits.length) break;
                 const { word, index, length } = matches[i];
                 const originalSegment = text.substring(index, index + length); // Get original case and length

                 let mappingInfo: { original: string, alt1: string, alt2: string } | null = null;
                 let currentForm: 'original' | 'alt1' | 'alt2' | null = null;

                 // Find the mapping based on the matched word (lowercase)
                 if (functionWordMap[word]) {
                     mappingInfo = { original: word, ...functionWordMap[word] };
                     currentForm = 'original';
                 } else {
                     for (const originalKey in functionWordMap) {
                         if (functionWordMap[originalKey].alt1 === word) {
                             mappingInfo = { original: originalKey, ...functionWordMap[originalKey] };
                             currentForm = 'alt1';
                             break;
                         }
                         if (functionWordMap[originalKey].alt2 === word) {
                             mappingInfo = { original: originalKey, ...functionWordMap[originalKey] };
                             currentForm = 'alt2';
                             break;
                         }
                     }
                 }

                 if (mappingInfo && currentForm) {
                     // --- Site identified, consume bit ---
                     const bit = bits[currentBitIndex];
                     // Mapping: bit 0 -> alt1, bit 1 -> alt2
                     // We need a consistent mapping. Let's use:
                     // If current is original: bit 0 -> alt1, bit 1 -> alt2
                     // If current is alt1: bit 0 -> original, bit 1 -> alt2
                     // If current is alt2: bit 0 -> original, bit 1 -> alt1
                     // This allows cycling through options, but extraction needs to know the *original* state.
                     // Simpler binary mapping: bit 0 -> alt1, bit 1 -> alt2 (assuming original is not used for encoding state)
                     const targetForm = bit ? 'alt2' : 'alt1';
                     let replacement = originalSegment;
                     let madeChange = false;

                     if (targetForm !== currentForm) {
                         replacement = bit ? mappingInfo.alt2 : mappingInfo.alt1;
                         // Basic check to avoid replacing with empty string if map is incomplete
                         if (replacement && replacement.length > 0) {
                             madeChange = true;
                             // Preserve case (simple title/all caps)
                             if (originalSegment[0] === originalSegment[0].toUpperCase()) {
                                 if (originalSegment.length === 1 || originalSegment.slice(1) === originalSegment.slice(1).toLowerCase()) {
                                     replacement = replacement.charAt(0).toUpperCase() + replacement.slice(1);
                                 } else if (originalSegment === originalSegment.toUpperCase()) {
                                     replacement = replacement.toUpperCase();
                                 }
                             }
                         } else {
                             console.warn(`Skipping replacement for ${word} as target form ${targetForm} is empty in map.`);
                         }
                     }

                     if (madeChange) {
                          modifiedText = modifiedText.substring(0, index + indexAdjustment) + replacement + modifiedText.substring(index + indexAdjustment + length);
                          indexAdjustment += (replacement.length - length);
                     }
                     bitsEncoded++;
                     currentBitIndex++;
                 }
            }
             // Ensure bitsEncoded doesn't exceed the number of bits provided
             bitsEncoded = Math.min(bitsEncoded, bits.length);

            return { modifiedText, bitsEncoded };
        };

        const extract = (text: string): boolean[] => {
            console.warn(`Carrier ${id} extract() assumes bit 0 maps to alt1, bit 1 to alt2.`);
            const extractedBits: boolean[] = [];

            // Find only alt1 or alt2 words
            const targetWords = Object.keys(functionWordMap).flatMap(k => [functionWordMap[k].alt1, functionWordMap[k].alt2]).filter(w => w && w.length > 0);
             const uniqueWords = [...new Set(targetWords)];
             if (uniqueWords.length === 0) return [];
            const regex = new RegExp(`\\b(${uniqueWords.join('|')})\\b`, 'gi');

            const matches: { word: string, index: number }[] = [];
            let match;
            while ((match = regex.exec(text)) !== null) {
                matches.push({ word: match[0].toLowerCase(), index: match.index });
            }
            matches.sort((a,b) => a.index - b.index); // Process in order found

            for (const { word } of matches) {
                 for (const originalKey in functionWordMap) {
                     if (functionWordMap[originalKey].alt1 === word) {
                         extractedBits.push(false); // alt1 represents bit 0
                         break; // Found the word's mapping, move to next match
                     }
                     if (functionWordMap[originalKey].alt2 === word) {
                         extractedBits.push(true); // alt2 represents bit 1
                         break; // Found the word's mapping, move to next match
                     }
                 }
            }

            return extractedBits;
        };

        return {
            id,
            name,
            category,
            bitsPerThousandWords,
            detectability,
            estimate,
            apply,
            extract,
            encode: apply,
            getDetectability: () => detectability,
            getCapacity: estimate,
            getNaturalness: () => 0.7, // Risk of unnatural phrasing
            getRobustness: () => 0.7, // Fairly robust
            // --- Default/Placeholder implementations ---
            analyze: (text: string) => ({ capacity: estimate(text) }),
            estimateCapacity: estimate,
        };
    }

    private createVoiceStyleCarrier(): CarrierTechnique {
        const id = 'voice_style';
        const name = 'Voice Style Transformer';
        const category = 'linguistic';
        const bitsPerThousandWords = 3; // Low capacity
        const detectability = 0.6; // Passive voice can be detectable

        const estimate = (text: string): number => {
            // Estimate based on sentences that *could* be transformed
            try {
                let siteCount = 0;
                const doc = nlp(text);
                 if (!doc || typeof doc.sentences !== 'function') return 0;
                doc.sentences().forEach((sentence: any) => {
                    // Check if sentence is likely active and transitive (has object)
                    // OR if it's passive and could potentially be made active (has 'by' agent)
                    const verbs = sentence.verbs();
                    if (verbs.length === 1) { // Focus on simple sentences for reliability
                         const verb = verbs.first();
                         const isPassive = verb.has('#Passive');
                         const hasObject = sentence.match('#Verb+ #Noun+').found; // Basic object check
                         const hasAgent = sentence.match('by #Noun+').found; // Basic agent check

                         if (!isPassive && hasObject) {
                             // Active sentence that could potentially become passive
                             siteCount++;
                         } else if (isPassive && hasAgent) {
                             // Passive sentence that could potentially become active
                             siteCount++;
                         }
                    }
                });
                return Math.floor(siteCount);
            } catch (e) {
                console.error(`Error estimating for ${id}: ${(e as Error).message}`);
                return 0;
            }
        };

        const apply = (text: string, bits: boolean[]): { modifiedText: string; bitsEncoded: number } => {
            console.warn(`Carrier ${id} apply() relies on compromise NLP transformation, which may not always be perfect or preserve meaning.`);
            if (bits.length === 0) return { modifiedText: text, bitsEncoded: 0 };

            let modifiedText = text;
            let bitsEncoded = 0;
            let currentBitIndex = 0;

            try {
                 const doc = nlp(text);
                  if (!doc || typeof doc.sentences !== 'function') return { modifiedText: text, bitsEncoded: 0 };
                 const sentences = doc.sentences().fullSentences(); // Get sentence objects
                 const modifications: { index: number, original: string, modified: string }[] = [];

                 sentences.forEach((sentence: any) => {
                     if (currentBitIndex >= bits.length) return;

                     const sentenceText = sentence.text();
                     let transformedSentence: string | null = null;
                     let madeChange = false;
                     const bit = bits[currentBitIndex]; // true for passive, false for active

                     // Check if compromise supports reliable transformation and if sentence is a potential site
                     let isPotentialSite = false;
                     const verbs = sentence.verbs();
                     if (verbs.length === 1) {
                         const isCurrentlyPassive = verbs.isPassive().found;
                         const hasObject = sentence.match('#Verb+ #Noun+').found;
                         const hasAgent = sentence.match('by #Noun+').found;
                         if ((!isCurrentlyPassive && hasObject) || (isCurrentlyPassive && hasAgent)) {
                             isPotentialSite = true;
                         }
                     }

                     if (!isPotentialSite) {
                         return; // Not a site we can reliably transform, skip without consuming bit
                     }

                     // --- Site identified, consume bit ---
                     const isCurrentlyPassive = verbs.isPassive().found;

                     if (typeof sentence.toPassive === 'function' && typeof sentence.toActive === 'function') {
                         if (bit && !isCurrentlyPassive) { // Target: Passive, Current: Active
                             const passiveAttempt = sentence.toPassive();
                             // Check if transformation actually changed the text and seems valid
                             if (passiveAttempt.text() !== sentenceText && passiveAttempt.text().length > 0) {
                                 transformedSentence = passiveAttempt.text();
                                 madeChange = true;
                             }
                         } else if (!bit && isCurrentlyPassive) { // Target: Active, Current: Passive
                             const activeAttempt = sentence.toActive();
                             if (activeAttempt.text() !== sentenceText && activeAttempt.text().length > 0) {
                                 transformedSentence = activeAttempt.text();
                                 madeChange = true;
                             }
                         }
                     } else {
                          console.warn(`compromise .toPassive()/.toActive() not available or reliable for carrier ${id}.`);
                          return; // Skip transformation if methods aren't available, do not consume bit
                     }

                     // If a transformation occurred OR the current state already matches the target bit, consume the bit.
                     if (madeChange || (bit && isCurrentlyPassive) || (!bit && !isCurrentlyPassive)) {
                          if (madeChange && transformedSentence) {
                              modifications.push({
                                  index: sentence.pointer[0][0], // Start index of sentence
                                  original: sentenceText,
                                  modified: transformedSentence
                              });
                          }
                          bitsEncoded++;
                          currentBitIndex++;
                     } else {
                         // Transformation failed or wasn't possible, but it was a potential site.
                         // Consume the bit anyway to maintain sequence, but log a warning.
                         console.warn(`Carrier ${id}: Failed to transform sentence or no change needed, but consuming bit.`);
                         bitsEncoded++;
                         currentBitIndex++;
                     }
                 });

                 // Apply modifications from end to start
                 modifications.sort((a, b) => b.index - a.index);
                 let workingText = text;
                 for (const mod of modifications) {
                     // Use substring replacement based on index to be safer than string.replace
                     workingText = workingText.substring(0, mod.index) + mod.modified + workingText.substring(mod.index + mod.original.length);
                 }
                 modifiedText = workingText;

            } catch (e) {
                 console.error(`Error applying ${id}: ${(e as Error).message}`);
                 return { modifiedText: text, bitsEncoded: 0 }; // Return original on error
            }
             // Ensure bitsEncoded doesn't exceed the number of bits provided
             bitsEncoded = Math.min(bitsEncoded, bits.length);

            return { modifiedText, bitsEncoded };
        };

        const extract = (text: string): boolean[] => {
            console.warn(`Carrier ${id} extract() relies on NLP detection.`);
            const extractedBits: boolean[] = [];

            try {
                 const doc = nlp(text);
                  if (!doc || typeof doc.sentences !== 'function') return [];
                 doc.sentences().forEach((sentence: any) => {
                     // Check if compromise supports reliable detection and if it was a potential site
                     if (typeof sentence.verbs === 'function' && typeof sentence.verbs().isPassive === 'function') {
                         const verbs = sentence.verbs();
                         let isPotentialSite = false;
                         if (verbs.length === 1) {
                             const isCurrentlyPassive = verbs.isPassive().found;
                             const hasObject = sentence.match('#Verb+ #Noun+').found;
                             const hasAgent = sentence.match('by #Noun+').found;
                             if ((!isCurrentlyPassive && hasObject) || (isCurrentlyPassive && hasAgent)) {
                                 isPotentialSite = true;
                             }
                         }

                         if (isPotentialSite) {
                              // Extract the state (passive = true, active = false)
                              const isPassive = verbs.isPassive().found;
                              extractedBits.push(isPassive);
                         }
                     } else {
                          console.warn(`compromise .isPassive() not available or reliable for carrier ${id} extraction.`);
                     }
                 });
            } catch (e) {
                console.error(`Error extracting from ${id}: ${(e as Error).message}`);
                return [];
            }

            return extractedBits;
        };

        return {
            id,
            name,
            category,
            bitsPerThousandWords,
            detectability,
            estimate,
            apply,
            extract,
            encode: apply,
            getDetectability: () => detectability,
            getCapacity: estimate,
            getNaturalness: () => 0.5, // Can sound unnatural or overly formal/informal
            getRobustness: () => 0.8, // Structure change is relatively robust
            // --- Default/Placeholder implementations ---
            analyze: (text: string) => ({ capacity: estimate(text) }),
            estimateCapacity: estimate,
        };
    }

    private createRhymingSynonymCarrier(): CarrierTechnique {
        const id = 'rhyming_synonym';
        const name = 'Rhyming Synonym Carrier';
        const category = 'linguistic';
        const bitsPerThousandWords = 4; // Low capacity, depends on map
        const detectability = 0.55; // Can be very noticeable if rhymes are forced

        const estimate = (text: string): number => {
            try {
                let count = 0;
                const targetWords = Object.keys(rhymeMap).flatMap(k => [k, rhymeMap[k].rhyme1, rhymeMap[k].rhyme2]);
                const uniqueWords = [...new Set(targetWords)];
                 if (uniqueWords.length === 0) return 0;
                const regex = new RegExp(`\\b(${uniqueWords.join('|')})\\b`, 'gi');
                const matches = text.match(regex);
                count = matches ? matches.length : 0;
                return count;
            } catch (e) {
                console.error(`Error estimating for ${id}: ${(e as Error).message}`);
                return 0;
            }
        };

        const apply = (text: string, bits: boolean[]): { modifiedText: string; bitsEncoded: number } => {
            console.warn(`Carrier ${id} apply() uses a simple rhyme map and may sound unnatural.`);
            if (bits.length === 0) return { modifiedText: text, bitsEncoded: 0 };

            let modifiedText = text;
            let bitsEncoded = 0;
            let currentBitIndex = 0;
            let indexAdjustment = 0;

             const targetWords = Object.keys(rhymeMap).flatMap(k => [k, rhymeMap[k].rhyme1, rhymeMap[k].rhyme2]);
             const uniqueWords = [...new Set(targetWords)];
              if (uniqueWords.length === 0) return { modifiedText: text, bitsEncoded: 0 };
             const regex = new RegExp(`\\b(${uniqueWords.join('|')})\\b`, 'gi');

             const matches: { word: string, index: number, length: number }[] = [];
             let match;
             while ((match = regex.exec(text)) !== null) {
                 matches.push({ word: match[0].toLowerCase(), index: match.index, length: match[0].length });
             }

             // Apply changes from end to start
             for (let i = matches.length - 1; i >= 0; i--) {
                  if (currentBitIndex >= bits.length) break;
                  const { word, index, length } = matches[i];
                  const originalSegment = text.substring(index, index + length);

                  let mappingInfo: { original: string, rhyme1: string, rhyme2: string } | null = null;
                  let currentForm: 'original' | 'rhyme1' | 'rhyme2' | null = null;

                  // Find mapping based on lowercase word
                  if (rhymeMap[word]) {
                      mappingInfo = { original: word, ...rhymeMap[word] };
                      currentForm = 'original';
                  } else {
                      for (const originalKey in rhymeMap) {
                          if (rhymeMap[originalKey].rhyme1 === word) {
                              mappingInfo = { original: originalKey, ...rhymeMap[originalKey] };
                              currentForm = 'rhyme1';
                              break;
                          }
                          if (rhymeMap[originalKey].rhyme2 === word) {
                              mappingInfo = { original: originalKey, ...rhymeMap[originalKey] };
                              currentForm = 'rhyme2';
                              break;
                          }
                      }
                  }

                  if (mappingInfo && currentForm) {
                      // --- Site identified, consume bit ---
                      const bit = bits[currentBitIndex];
                      // Mapping: bit 0 -> rhyme1, bit 1 -> rhyme2
                      const targetForm = bit ? 'rhyme2' : 'rhyme1';
                      let replacement = originalSegment;
                      let madeChange = false;

                      if (targetForm !== currentForm) {
                          replacement = bit ? mappingInfo.rhyme2 : mappingInfo.rhyme1;
                          if (replacement && replacement.length > 0) {
                              madeChange = true;
                              // Preserve case
                              if (originalSegment[0] === originalSegment[0].toUpperCase()) {
                                  if (originalSegment.length === 1 || originalSegment.slice(1) === originalSegment.slice(1).toLowerCase()) {
                                      replacement = replacement.charAt(0).toUpperCase() + replacement.slice(1);
                                  } else if (originalSegment === originalSegment.toUpperCase()) {
                                      replacement = replacement.toUpperCase();
                                  }
                              }
                          } else {
                               console.warn(`Skipping replacement for ${word} as target form ${targetForm} is empty in map.`);
                          }
                      }

                      if (madeChange) {
                           modifiedText = modifiedText.substring(0, index + indexAdjustment) + replacement + modifiedText.substring(index + indexAdjustment + length);
                           indexAdjustment += (replacement.length - length);
                      }
                      bitsEncoded++;
                      currentBitIndex++;
                  }
             }
              // Ensure bitsEncoded doesn't exceed the number of bits provided
              bitsEncoded = Math.min(bitsEncoded, bits.length);


            return { modifiedText, bitsEncoded };
        };

        const extract = (text: string): boolean[] => {
            console.warn(`Carrier ${id} extract() assumes bit 0 maps to rhyme1, bit 1 to rhyme2.`);
            const extractedBits: boolean[] = [];

             // Find only rhyme1 or rhyme2 words
             const targetWords = Object.keys(rhymeMap).flatMap(k => [rhymeMap[k].rhyme1, rhymeMap[k].rhyme2]).filter(w => w && w.length > 0);
              const uniqueWords = [...new Set(targetWords)];
              if (uniqueWords.length === 0) return [];
             const regex = new RegExp(`\\b(${uniqueWords.join('|')})\\b`, 'gi');

             const matches: { word: string, index: number }[] = [];
             let match;
             while ((match = regex.exec(text)) !== null) {
                 matches.push({ word: match[0].toLowerCase(), index: match.index });
             }
             matches.sort((a,b) => a.index - b.index);

             for (const { word } of matches) {
                  for (const originalKey in rhymeMap) {
                      if (rhymeMap[originalKey].rhyme1 === word) {
                          extractedBits.push(false); // rhyme1 represents bit 0
                          break;
                      }
                      if (rhymeMap[originalKey].rhyme2 === word) {
                          extractedBits.push(true); // rhyme2 represents bit 1
                          break;
                      }
                  }
             }

            return extractedBits;
        };

        return {
            id,
            name,
            category,
            bitsPerThousandWords,
            detectability,
            estimate,
            apply,
            extract,
            encode: apply,
            getDetectability: () => detectability,
            getCapacity: estimate,
            getNaturalness: () => 0.6, // High risk of sounding unnatural
            getRobustness: () => 0.7, // Fairly robust
            // --- Default/Placeholder implementations ---
            analyze: (text: string) => ({ capacity: estimate(text) }),
            estimateCapacity: estimate,
        };
    }

    private createDescriptionDetailCarrier(): CarrierTechnique {
        // Swaps between two predefined modifiers (mod1/mod2) for specific nouns.
        const id = 'description_detail';
        const name = 'Description Detail Carrier';
        const category = 'readability'; // Affects readability/style
        const bitsPerThousandWords = 3; // Low capacity
        const detectability = 0.5; // Adding/changing adjectives can be noticeable

        const estimate = (text: string): number => {
            // Estimate based on occurrences of nouns *preceded by* one of their potential modifiers.
            try {
                let count = 0;
                const targetNouns = Object.keys(descriptionMap);
                 if (targetNouns.length === 0) return 0;
                 const allModifiers = Object.values(descriptionMap).flatMap(m => [m.mod1, m.mod2]).filter(m => m && m.length > 0);
                 const uniqueModifiers = [...new Set(allModifiers)];
                 if (uniqueModifiers.length === 0) return 0;

                 // Regex: modifier, space, noun
                 const regex = new RegExp(`\\b(${uniqueModifiers.join('|')})\\s+(${targetNouns.join('|')})\\b`, 'gi');
                 const matches = text.match(regex);
                 // Filter matches to ensure the modifier actually belongs to the noun in our map
                 if (matches) {
                     matches.forEach(matchText => {
                         const parts = matchText.split(/\s+/);
                         if (parts.length === 2) {
                             const mod = parts[0].toLowerCase();
                             const noun = parts[1].toLowerCase();
                             if (descriptionMap[noun] && (descriptionMap[noun].mod1 === mod || descriptionMap[noun].mod2 === mod)) {
                                 count++;
                             }
                         }
                     });
                 }
                return count;
            } catch (e) {
                console.error(`Error estimating for ${id}: ${(e as Error).message}`);
                return 0;
            }
        };

        const apply = (text: string, bits: boolean[]): { modifiedText: string; bitsEncoded: number } => {
            console.warn(`Carrier ${id} apply() swaps predefined adjectives and may affect naturalness.`);
            if (bits.length === 0) return { modifiedText: text, bitsEncoded: 0 };

            let modifiedText = text;
            let bitsEncoded = 0;
            let currentBitIndex = 0;
            let indexAdjustment = 0;

             const targetNouns = Object.keys(descriptionMap);
              if (targetNouns.length === 0) return { modifiedText: text, bitsEncoded: 0 };
             const allModifiers = Object.values(descriptionMap).flatMap(m => [m.mod1, m.mod2]).filter(m => m && m.length > 0);
             const uniqueModifiers = [...new Set(allModifiers)];
              if (uniqueModifiers.length === 0) return { modifiedText: text, bitsEncoded: 0 };

             // Regex to find modifier + noun pairs from our map
             const regex = new RegExp(`\\b(${uniqueModifiers.join('|')})\\s+(${targetNouns.join('|')})\\b`, 'gi');

             const matches: { noun: string, modifier: string, index: number, length: number }[] = [];
             let match;
             while ((match = regex.exec(text)) !== null) {
                 const matchedMod = match[1].toLowerCase();
                 const matchedNoun = match[2].toLowerCase();
                 // Verify this pair exists in the map before adding
                 if (descriptionMap[matchedNoun] && (descriptionMap[matchedNoun].mod1 === matchedMod || descriptionMap[matchedNoun].mod2 === matchedMod)) {
                      matches.push({
                          noun: matchedNoun,
                          modifier: matchedMod,
                          index: match.index,
                          length: match[0].length
                      });
                 }
             }


             // Apply changes from end to start
             for (let i = matches.length - 1; i >= 0; i--) {
                  if (currentBitIndex >= bits.length) break;
                  const { noun, modifier, index, length } = matches[i];
                  const originalSegment = text.substring(index, index + length); // e.g., "vast sky"

                  const mapping = descriptionMap[noun]; // Already verified this exists
                  if (!mapping) continue;

                  // --- Site identified, consume bit ---
                  const bit = bits[currentBitIndex];
                  // Mapping: bit 0 -> mod1, bit 1 -> mod2
                  const targetModifier = bit ? mapping.mod2 : mapping.mod1;
                  let currentModifier = modifier; // Lowercase modifier found
                  let replacement = originalSegment;
                  let madeChange = false;

                  if (currentModifier !== targetModifier) {
                      // Find the noun part in the original segment to preserve its case
                      const nounIndexInSegment = originalSegment.toLowerCase().indexOf(noun);
                      const nounSegment = originalSegment.substring(nounIndexInSegment);
                      // Get the original modifier segment to preserve its case
                      const modSegment = originalSegment.substring(0, nounIndexInSegment).trimEnd();

                      let newModifierSegment = targetModifier;
                      // Preserve case of original modifier
                       if (modSegment[0] === modSegment[0].toUpperCase()) {
                           if (modSegment.length === 1 || modSegment.slice(1) === modSegment.slice(1).toLowerCase()) {
                               newModifierSegment = newModifierSegment.charAt(0).toUpperCase() + newModifierSegment.slice(1);
                           } else if (modSegment === modSegment.toUpperCase()) {
                               newModifierSegment = newModifierSegment.toUpperCase();
                           }
                       }

                      replacement = newModifierSegment + " " + nounSegment;
                      madeChange = true;
                  }

                  if (madeChange) {
                       modifiedText = modifiedText.substring(0, index + indexAdjustment) + replacement + modifiedText.substring(index + indexAdjustment + length);
                       indexAdjustment += (replacement.length - length);
                  }
                  bitsEncoded++;
                  currentBitIndex++;
             }
              // Ensure bitsEncoded doesn't exceed the number of bits provided
              bitsEncoded = Math.min(bitsEncoded, bits.length);


            return { modifiedText, bitsEncoded };
        };

        const extract = (text: string): boolean[] => {
            console.warn(`Carrier ${id} extract() assumes bit 0 maps to mod1, bit 1 to mod2.`);
            const extractedBits: boolean[] = [];

             const targetNouns = Object.keys(descriptionMap);
              if (targetNouns.length === 0) return [];
             const allModifiers = Object.values(descriptionMap).flatMap(m => [m.mod1, m.mod2]).filter(m => m && m.length > 0);
             const uniqueModifiers = [...new Set(allModifiers)];
             if (uniqueModifiers.length === 0) return [];

             // Regex: modifier, space, noun
             const regex = new RegExp(`\\b(${uniqueModifiers.join('|')})\\s+(${targetNouns.join('|')})\\b`, 'gi');

             const matches: { noun: string, modifier: string, index: number }[] = [];
             let match;
             while ((match = regex.exec(text)) !== null) {
                  const matchedMod = match[1].toLowerCase();
                  const matchedNoun = match[2].toLowerCase();
                  // Verify pair exists in map
                  if (descriptionMap[matchedNoun] && (descriptionMap[matchedNoun].mod1 === matchedMod || descriptionMap[matchedNoun].mod2 === matchedMod)) {
                      matches.push({
                          noun: matchedNoun,
                          modifier: matchedMod,
                          index: match.index
                      });
                  }
             }
             matches.sort((a,b) => a.index - b.index);

             for (const { noun, modifier } of matches) {
                  const mapping = descriptionMap[noun]; // Already verified
                  if (mapping) {
                      if (modifier === mapping.mod1) {
                          extractedBits.push(false); // mod1 represents bit 0
                      } else if (modifier === mapping.mod2) {
                          extractedBits.push(true); // mod2 represents bit 1
                      }
                  }
             }

            return extractedBits;
        };

        return {
            id,
            name,
            category,
            bitsPerThousandWords,
            detectability,
            estimate,
            apply,
            extract,
            encode: apply,
            getDetectability: () => detectability,
            getCapacity: estimate,
            getNaturalness: () => 0.6, // Adding/swapping adjectives can feel forced
            getRobustness: () => 0.7, // Fairly robust
            // --- Default/Placeholder implementations ---
            analyze: (text: string) => ({ capacity: estimate(text) }),
            estimateCapacity: estimate,
        };
    }
}