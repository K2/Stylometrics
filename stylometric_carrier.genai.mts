/**
 * Stylometric Carrier - Information Embedding Using Stylometric Features
 * 
 * This module inverts the stylometric detection techniques identified by Kumarage et al. (2023)
 * to create steganographic carriers within text. It exploits stylometric features that typically
 * differentiate human from AI text to instead carry embedded information.
 * 
 * Flow:
 * 1. Analyze text to determine carrying capacity
 * 2. Process payload to distribute across available carriers 
 * 3. Apply modifications to embed payload in text
 * 4. Verify modified text remains natural and detection-resistant
 * 
 * Happy path: analyzeCarryingCapacity → encodePayload → extractPayload
 */

import { StyleFeatureExtractor, type FeatureMap } from './stylometric_detection.genai.mjs';
import * as nlp from 'compromise';

/**
 * Interface for carrier analysis results
 */
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

/**
 * Interface for encoding options
 */
export interface EncodingOptions {
    usePhraseologyCarriers?: boolean;
    usePunctuationCarriers?: boolean;
    useLinguisticCarriers?: boolean;
    useReadabilityCarriers?: boolean;
    errorCorrection?: boolean;
    preserveReadingLevel?: boolean;
    preserveLinguisticFingerprint?: boolean;
    preserveKeyFeatures?: string[];
    maxDetectionRisk?: number; // 0-1 scale
}

/**
 * Interface for an individual carrier technique
 */
export interface CarrierTechnique {
    id: string;
    name: string;
    category: 'phraseology' | 'punctuation' | 'linguistic' | 'readability';
    bitsPerThousandWords: number;
    apply: (text: string, bits: boolean[]) => { modifiedText: string; bitsEncoded: number };
    extract: (text: string) => boolean[];
    estimate: (text: string) => number;
    detectability: number; // 0-1 scale
}

/**
 * StylometricCarrier enables embedding information in text using stylometric features
 * identified in the Kumarage et al. (2023) research.
 */
export class StylometricCarrier {
    private featureExtractor: StyleFeatureExtractor;
    private carriers: CarrierTechnique[];
    
    /**
     * Initialize the stylometric carrier
     */
    constructor() {
        this.featureExtractor = new StyleFeatureExtractor();
        this.carriers = this.initializeCarriers();
    }
    
    /**
     * Initialize all available carrier techniques
     */
    private initializeCarriers(): CarrierTechnique[] {
        return [
            this.createSentenceLengthCarrier(),
            this.createParagraphStructureCarrier(),
            this.createPunctuationFrequencyCarrier(),
            this.createQuoteStyleCarrier(),
            this.createOptionalCommaCarrier(),
            this.createSynonymSubstitutionCarrier(),
            this.createLexicalRichnessCarrier(),
            this.createFunctionWordCarrier(),
            this.createSyllableCountCarrier(),
            this.createVoiceStyleCarrier()
        ];
    }
    
    /**
     * Analyze carrying capacity of text for embedding information
     * 
     * @param text Text to analyze
     * @returns Analysis of carrying capacity
     */
    analyzeCarryingCapacity(text: string): CarrierAnalysis {
        const features = this.featureExtractor.extractAllFeatures(text);
        const wordCount = features.word_count || 0;
        
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
        
        // Calculate safe modification ranges
        const safeModificationRanges = this.calculateSafeModificationRanges(features);
        
        // Max payload with some error correction overhead
        const maxPayloadBytes = Math.floor((totalBits * 0.75) / 8);
        
        return {
            totalCapacityBits: totalBits,
            carrierDistribution: {
                phraseology: phraseologyBits,
                punctuation: punctuationBits,
                linguistic: linguisticBits,
                readability: readabilityBits
            },
            safeModificationRanges,
            recommendedMaxPayloadBytes: maxPayloadBytes
        };
    }
    
    /**
     * Sum bits capacity by carrier category
     */
    private sumCarrierBits(
        estimates: Array<{id: string; category: string; bitCapacity: number; detectability: number}>,
        category: string
    ): number {
        return estimates
            .filter(e => e.category === category)
            .reduce((sum, e) => sum + e.bitCapacity, 0);
    }
    
    /**
     * Calculate safe ranges for modifying features without triggering detection
     */
    private calculateSafeModificationRanges(features: FeatureMap): FeatureMap {
        const ranges: FeatureMap = {};
        
        // For each feature, determine a safe min/max range for modifications
        // These values are based on the research paper's identified thresholds
        
        // Lexical richness (MATTR) range
        const lexicalRichness = features.lexical_richness || 0.7;
        ranges.lexical_richness_min = Math.max(0.5, lexicalRichness - 0.08);
        ranges.lexical_richness_max = Math.min(0.9, lexicalRichness + 0.08);
        
        // Readability score range
        const readability = features.readability || 60;
        ranges.readability_min = Math.max(30, readability - 10);
        ranges.readability_max = Math.min(90, readability + 10);
        
        // Words per sentence range
        const meanWordsPerSentence = features.mean_words_per_sentence || 15;
        ranges.mean_words_per_sentence_min = Math.max(8, meanWordsPerSentence - 4);
        ranges.mean_words_per_sentence_max = Math.min(25, meanWordsPerSentence + 4);
        
        // Sentence variance range
        const stdevWordsPerSentence = features.stdev_words_per_sentence || 5;
        ranges.stdev_words_per_sentence_min = Math.max(2, stdevWordsPerSentence - 2);
        ranges.stdev_words_per_sentence_max = Math.min(12, stdevWordsPerSentence + 2);
        
        return ranges;
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
        // Analyze the carrying capacity
        const analysis = this.analyzeCarryingCapacity(text);
        
        // Check if payload will fit
        const payloadBits = payload.length * 8;
        if (payloadBits > analysis.totalCapacityBits) {
            throw new Error(`Payload too large: ${payload.length} bytes exceeds capacity of ${Math.floor(analysis.totalCapacityBits / 8)} bytes`);
        }
        
        // Convert payload to bit array
        const bits = this.bytesToBits(payload);
        
        // Clone original text as starting point
        let modifiedText = text;
        let bitsRemaining = [...bits]; // Clone the bits array
        
        // Apply carriers according to options and priority
        const activeCarriers = this.getActiveCarriers(options);
        
        // Sort carriers by detectability (least detectable first)
        activeCarriers.sort((a, b) => a.detectability - b.detectability);
        
        // Apply each carrier until all bits are encoded
        for (const carrier of activeCarriers) {
            if (bitsRemaining.length === 0) break;
            
            // Apply this carrier technique
            const result = carrier.apply(modifiedText, bitsRemaining);
            modifiedText = result.modifiedText;
            
            // Remove encoded bits from remaining
            bitsRemaining = bitsRemaining.slice(result.bitsEncoded);
        }
        
        // If bits remain, we couldn't encode everything
        if (bitsRemaining.length > 0) {
            throw new Error(`Could not encode entire payload: ${bitsRemaining.length} bits remain`);
        }
        
        // Verify the text still looks natural
        this.verifyNaturalText(text, modifiedText, options);
        
        return modifiedText;
    }
    
    /**
     * Extract payload from text that was embedded using stylometric carriers
     * 
     * @param text Text with embedded payload
     * @param options Extraction options (should match encoding options)
     * @returns Extracted payload
     */
    extractPayload(text: string, options: EncodingOptions = {}): Uint8Array {
        // Get active carriers (must match encoding options)
        const activeCarriers = this.getActiveCarriers(options);
        
        // Sort carriers in the same order as encoding
        activeCarriers.sort((a, b) => a.detectability - b.detectability);
        
        // Extract bits from each carrier
        const extractedBits: boolean[] = [];
        for (const carrier of activeCarriers) {
            const bits = carrier.extract(text);
            extractedBits.push(...bits);
        }
        
        // Convert bits back to bytes
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
            maxDetectionRisk = 0.5
        } = options;
        
        return this.carriers.filter(carrier => {
            // Filter by detection risk
            if (carrier.detectability > maxDetectionRisk) {
                return false;
            }
            
            // Filter by category
            switch (carrier.category) {
                case 'phraseology':
                    return usePhraseologyCarriers;
                case 'punctuation':
                    return usePunctuationCarriers;
                case 'linguistic':
                    return useLinguisticCarriers;
                case 'readability':
                    return useReadabilityCarriers;
                default:
                    return false;
            }
        });
    }
    
    /**
     * Verify the modified text still looks natural
     */
    private verifyNaturalText(original: string, modified: string, options: EncodingOptions): void {
        // Extract features from both texts
        const originalFeatures = this.featureExtractor.extractAllFeatures(original);
        const modifiedFeatures = this.featureExtractor.extractAllFeatures(modified);
        
        // If preserveReadingLevel is enabled, check readability score
        if (options.preserveReadingLevel) {
            const readabilityDiff = Math.abs(
                (modifiedFeatures.readability || 0) - 
                (originalFeatures.readability || 0)
            );
            
            if (readabilityDiff > 10) {
                throw new Error(`Modified text readability differs too much: ${readabilityDiff} points`);
            }
        }
        
        // If preserveLinguisticFingerprint is enabled, check key linguistic features
        if (options.preserveLinguisticFingerprint) {
            const lexicalDiff = Math.abs(
                (modifiedFeatures.lexical_richness || 0) - 
                (originalFeatures.lexical_richness || 0)
            );
            
            if (lexicalDiff > 0.1) {
                throw new Error(`Modified text linguistic fingerprint differs too much: ${lexicalDiff} MATTR difference`);
            }
        }
        
        // Check preserved key features
        if (options.preserveKeyFeatures) {
            for (const feature of options.preserveKeyFeatures) {
                if (feature in originalFeatures && feature in modifiedFeatures) {
                    const diff = Math.abs(originalFeatures[feature] - modifiedFeatures[feature]);
                    const threshold = originalFeatures[feature] * 0.15; // 15% difference tolerance
                    
                    if (diff > threshold) {
                        throw new Error(`Modified text key feature '${feature}' differs too much: ${diff}`);
                    }
                }
            }
        }
    }
    
    /**
     * Convert bytes to bits array
     */
    private bytesToBits(bytes: Uint8Array): boolean[] {
        const bits: boolean[] = [];
        for (let i = 0; i < bytes.length; i++) {
            const byte = bytes[i];
            for (let j = 7; j >= 0; j--) {
                bits.push(((byte >> j) & 1) === 1);
            }
        }
        return bits;
    }
    
    /**
     * Convert bits array to bytes
     */
    private bitsToBytes(bits: boolean[]): Uint8Array {
        // Ensure the bit count is a multiple of 8
        const padding = (8 - (bits.length % 8)) % 8;
        const paddedBits = [...bits];
        for (let i = 0; i < padding; i++) {
            paddedBits.push(false);
        }
        
        // Convert bits to bytes
        const bytes = new Uint8Array(paddedBits.length / 8);
        for (let i = 0; i < paddedBits.length; i += 8) {
            let byte = 0;
            for (let j = 0; j < 8; j++) {
                if (paddedBits[i + j]) {
                    byte |= 1 << (7 - j);
                }
            }
            bytes[i / 8] = byte;
        }
        
        return bytes;
    }
    
    /*** CARRIER TECHNIQUE IMPLEMENTATIONS ***/
    
    /**
     * Sentence length pattern carrier
     * Encodes bits by varying sentence length within natural ranges
     */
    private createSentenceLengthCarrier(): CarrierTechnique {
        return {
            id: 'sentence_length',
            name: 'Sentence Length Pattern',
            category: 'phraseology',
            bitsPerThousandWords: 8,
            detectability: 0.3,
            
            estimate: (text: string) => {
                const wordCount = text.split(/\s+/).length;
                return Math.floor((wordCount / 1000) * this.bitsPerThousandWords);
            },
            
            apply: (text: string, bits: boolean[]) => {
                // Implementation would modify sentence lengths to encode bits
                // This is a simplified placeholder for the concept
                const doc = nlp(text);
                const sentences = doc.sentences().out('array');
                
                // Determine how many bits we can encode
                const maxBits = Math.min(bits.length, Math.floor(sentences.length / 2));
                
                // For each bit, modify adjacent sentence pairs
                let modifiedText = text;
                for (let i = 0; i < maxBits; i++) {
                    // This is where actual sentence length modifications would happen
                    // For each bit, we'd adjust the relative lengths of adjacent sentences
                }
                
                return { modifiedText, bitsEncoded: maxBits };
            },
            
            extract: (text: string) => {
                // Extract bits from sentence length patterns
                const extractedBits: boolean[] = [];
                // Implementation would analyze sentence lengths to extract encoded bits
                return extractedBits;
            }
        };
    }
    
    /**
     * Paragraph structure carrier
     * Encodes bits in paragraph lengths and structures
     */
    private createParagraphStructureCarrier(): CarrierTechnique {
        return {
            id: 'paragraph_structure',
            name: 'Paragraph Structure',
            category: 'phraseology',
            bitsPerThousandWords: 5,
            detectability: 0.35,
            
            estimate: (text: string) => {
                const wordCount = text.split(/\s+/).length;
                const paragraphs = text.split(/\n\s*\n/).length;
                
                // More accurate estimation based on paragraph count
                const estimatedBits = Math.floor(Math.min(
                    paragraphs / 3, // Approximately 1 bit per 3 paragraphs
                    (wordCount / 1000) * this.bitsPerThousandWords
                ));
                
                return estimatedBits;
            },
            
            apply: (text: string, bits: boolean[]) => {
                // Implementation would modify paragraph structures to encode bits
                // This is a simplified placeholder
                const paragraphs = text.split(/\n\s*\n/);
                const maxBits = Math.min(bits.length, Math.floor(paragraphs.length / 3));
                
                return { modifiedText: text, bitsEncoded: maxBits };
            },
            
            extract: (text: string) => {
                // Extract bits from paragraph structures
                return [];
            }
        };
    }
    
    /**
     * Punctuation frequency carrier
     * Encodes bits by adjusting frequencies of specific punctuation marks
     */
    private createPunctuationFrequencyCarrier(): CarrierTechnique {
        return {
            id: 'punctuation_frequency',
            name: 'Punctuation Frequency',
            category: 'punctuation',
            bitsPerThousandWords: 12,
            detectability: 0.25,
            
            estimate: (text: string) => {
                const wordCount = text.split(/\s+/).length;
                // Count potential punctuation modification points
                const punctCount = (text.match(/[,;:!?]/g) || []).length;
                
                // Estimate based on available punctuation marks
                return Math.min(
                    Math.floor(punctCount / 8), // Approx 1 bit per 8 punctuation marks
                    Math.floor((wordCount / 1000) * this.bitsPerThousandWords)
                );
            },
            
            apply: (text: string, bits: boolean[]) => {
                // Find modifiable punctuation positions
                const punctPos: number[] = [];
                const punctRe = /[,;:!?]/g;
                let match;
                
                while ((match = punctRe.exec(text)) !== null) {
                    punctPos.push(match.index);
                }
                
                // Determine how many bits we can encode
                const maxBits = Math.min(bits.length, Math.floor(punctPos.length / 8));
                
                // For each encodable bit, modify punctuation
                // (Actual implementation would make specific changes)
                
                return { modifiedText: text, bitsEncoded: maxBits };
            },
            
            extract: (text: string) => {
                // Extract bits from punctuation frequency
                return [];
            }
        };
    }
    
    /**
     * Quote style carrier
     * Encodes bits by switching between quote styles
     */
    private createQuoteStyleCarrier(): CarrierTechnique {
        return {
            id: 'quote_style',
            name: 'Quote Style Alternation',
            category: 'punctuation',
            bitsPerThousandWords: 4,
            detectability: 0.2,
            
            estimate: (text: string) => {
                const quoteCount = (text.match(/["']/g) || []).length;
                return Math.floor(quoteCount / 4); // Approx 1 bit per 4 quotes
            },
            
            apply: (text: string, bits: boolean[]) => {
                // Implementation would switch between quote styles
                const quotePositions = [];
                const quoteRe = /["']/g;
                let match;
                
                while ((match = quoteRe.exec(text)) !== null) {
                    quotePositions.push({
                        pos: match.index,
                        char: match[0]
                    });
                }
                
                // Group quotes into pairs
                const quotePairs = [];
                for (let i = 0; i < quotePositions.length; i += 2) {
                    if (i + 1 < quotePositions.length) {
                        quotePairs.push([quotePositions[i], quotePositions[i + 1]]);
                    }
                }
                
                // Determine how many bits we can encode
                const maxBits = Math.min(bits.length, quotePairs.length);
                
                // Here we would modify the quotes based on bits
                
                return { modifiedText: text, bitsEncoded: maxBits };
            },
            
            extract: (text: string) => {
                // Extract bits from quote style patterns
                return [];
            }
        };
    }
    
    /**
     * Optional comma carrier
     * Encodes bits by using or omitting stylistically optional commas
     */
    private createOptionalCommaCarrier(): CarrierTechnique {
        return {
            id: 'optional_comma',
            name: 'Optional Comma Placement',
            category: 'punctuation',
            bitsPerThousandWords: 10,
            detectability: 0.2,
            
            estimate: (text: string) => {
                const doc = nlp(text);
                const sentences = doc.sentences().out('array');
                
                // Estimate optional comma positions (simplified)
                let optionalCommaPositions = 0;
                for (const sentence of sentences) {
                    // Count potential optional comma positions
                    const clauses = sentence.split(/\b(and|or|but|however|therefore|moreover)\b/);
                    if (clauses.length > 2) {
                        optionalCommaPositions += Math.floor(clauses.length / 2);
                    }
                }
                
                return Math.floor(optionalCommaPositions / 2); // ~1 bit per 2 positions
            },
            
            apply: (text: string, bits: boolean[]) => {
                // Implementation would add/remove optional commas
                // This is complex and requires linguistic analysis
                // Simplified placeholder
                return { modifiedText: text, bitsEncoded: 0 };
            },
            
            extract: (text: string) => {
                // Extract bits from optional comma patterns
                return [];
            }
        };
    }
    
    /**
     * Synonym substitution carrier
     * Encodes bits by replacing words with synonyms of varying frequency
     */
    private createSynonymSubstitutionCarrier(): CarrierTechnique {
        return {
            id: 'synonym_substitution',
            name: 'Synonym Substitution',
            category: 'linguistic',
            bitsPerThousandWords: 10,
            detectability: 0.4,
            
            estimate: (text: string) => {
                const wordCount = text.split(/\s+/).length;
                // Conservatively estimate substitutable words (adjectives, adverbs, some verbs)
                // ~10-15% of words in typical text
                const substitutableWords = Math.floor(wordCount * 0.12);
                return Math.min(
                    Math.floor(substitutableWords / 2), // 1 bit per 2 suitable words
                    Math.floor((wordCount / 1000) * this.bitsPerThousandWords)
                );
            },
            
            apply: (text: string, bits: boolean[]) => {
                // Implementation would replace words with synonyms
                // Requires a synonym dictionary and POS tagging
                
                return { modifiedText: text, bitsEncoded: 0 };
            },
            
            extract: (text: string) => {
                // Would need original text or synonym dictionary to extract
                return [];
            }
        };
    }
    
    /**
     * Lexical richness carrier
     * Encodes bits by adjusting the type-token ratio in specific windows
     */
    private createLexicalRichnessCarrier(): CarrierTechnique {
        return {
            id: 'lexical_richness',
            name: 'Lexical Richness Modulation',
            category: 'linguistic',
            bitsPerThousandWords: 3,
            detectability: 0.45,
            
            estimate: (text: string) => {
                const wordCount = text.split(/\s+/).length;
                // MATTR windows provide few carrier opportunities
                return Math.floor(wordCount / 300); // ~1 bit per 300 words
            },
            
            apply: (text: string, bits: boolean[]) => {
                // Implementation would adjust word variety in specific windows
                // This is complex and would need careful word substitution
                
                const estimatedBits = this.estimate(text);
                const bitsToEncode = Math.min(bits.length, estimatedBits);
                
                return { modifiedText: text, bitsEncoded: bitsToEncode };
            },
            
            extract: (text: string) => {
                // Extract bits from lexical richness patterns
                return [];
            }
        };
    }
    
    /**
     * Function word carrier
     * Encodes bits by modifying distribution patterns of common function words
     */
    private createFunctionWordCarrier(): CarrierTechnique {
        return {
            id: 'function_word',
            name: 'Function Word Distribution',
            category: 'linguistic',
            bitsPerThousandWords: 6,
            detectability: 0.3,
            
            estimate: (text: string) => {
                // Function words are ~40% of typical text
                const wordCount = text.split(/\s+/).length;
                const functionWordCount = Math.floor(wordCount * 0.4);
                
                return Math.min(
                    Math.floor(functionWordCount / 12), // ~1 bit per 12 function words
                    Math.floor((wordCount / 1000) * this.bitsPerThousandWords)
                );
            },
            
            apply: (text: string, bits: boolean[]) => {
                // Implementation would modify function word patterns
                // (e.g., "that" vs omission, "which" vs "that", etc.)
                
                return { modifiedText: text, bitsEncoded: 0 };
            },
            
            extract: (text: string) => {
                // Extract bits from function word patterns
                return [];
            }
        };
    }
    
    /**
     * Syllable count carrier
     * Encodes bits by adjusting word choices to affect syllable counts
     */
    private createSyllableCountCarrier(): CarrierTechnique {
        return {
            id: 'syllable_count',
            name: 'Syllable Count Adjustment',
            category: 'readability',
            bitsPerThousandWords: 4,
            detectability: 0.35,
            
            estimate: (text: string) => {
                const sentenceCount = text.split(/[.!?]+\s/).length;
                return Math.floor(sentenceCount / 5); // ~1 bit per 5 sentences
            },
            
            apply: (text: string, bits: boolean[]) => {
                // Implementation would adjust word choices to affect syllable counts
                // This would require a syllable counter and word alternatives
                
                return { modifiedText: text, bitsEncoded: 0 };
            },
            
            extract: (text: string) => {
                // Extract bits from syllable patterns
                return [];
            }
        };
    }
    
    /**
     * Voice style carrier
     * Encodes bits by alternating between passive and active voice
     */
    private createVoiceStyleCarrier(): CarrierTechnique {
        return {
            id: 'voice_style',
            name: 'Passive/Active Voice Switching',
            category: 'readability',
            bitsPerThousandWords: 3,
            detectability: 0.5,
            
            estimate: (text: string) => {
                const doc = nlp(text);
                // This is a very rough estimate since voice detection is complex
                return Math.floor(doc.sentences().length() / 10); // ~1 bit per 10 sentences
            },
            
            apply: (text: string, bits: boolean[]) => {
                // Implementation would convert between active/passive voice
                // This is complex and requires deep language understanding
                
                return { modifiedText: text, bitsEncoded: 0 };
            },
            
            extract: (text: string) => {
                // Extract bits from voice patterns
                return [];
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
    text: string,
    payload: string
): void {
    console.log("=== STYLOMETRIC CARRIER DEMO ===");
    
    // Create carrier
    const carrier = new StylometricCarrier();
    
    // Analyze carrying capacity
    console.log("\n1. ANALYZING CARRYING CAPACITY:");
    console.log("-------------------------------");
    const analysis = carrier.analyzeCarryingCapacity(text);
    console.log(`Total capacity: ${analysis.totalCapacityBits} bits (${Math.floor(analysis.totalCapacityBits / 8)} bytes)`);
    console.log("Carrier distribution:");
    console.log(`- Phraseology: ${analysis.carrierDistribution.phraseology} bits`);
    console.log(`- Punctuation: ${analysis.carrierDistribution.punctuation} bits`);
    console.log(`- Linguistic: ${analysis.carrierDistribution.linguistic} bits`);
    console.log(`- Readability: ${analysis.carrierDistribution.readability} bits`);
    console.log(`Recommended max payload: ${analysis.recommendedMaxPayloadBytes} bytes`);
    
    // Create sample payload
    const encoder = new TextEncoder();
    const payloadBytes = encoder.encode(payload);
    
    console.log(`\nPayload size: ${payloadBytes.length} bytes`);
    
    // Encode payload (with error handling)
    try {
        // Define encoding options
        const options: EncodingOptions = {
            usePhraseologyCarriers: true,
            usePunctuationCarriers: true,
            useLinguisticCarriers: true,
            useReadabilityCarriers: false, // Disable readability carriers for demo
            errorCorrection: true,
            preserveReadingLevel: true
        };
        
        console.log("\n2. ENCODING PAYLOAD:");
        console.log("--------------------");
        console.log("Original text (first 100 chars):");
        console.log(text.substring(0, 100) + "...");
        
        // Encode the payload
        const modifiedText = carrier.encodePayload(text, payloadBytes, options);
        
        console.log("\nModified text (first 100 chars):");
        console.log(modifiedText.substring(0, 100) + "...");
        
        // Extract the payload
        console.log("\n3. EXTRACTING PAYLOAD:");
        console.log("----------------------");
        const extractedBytes = carrier.extractPayload(modifiedText, options);
        
        // Decode the extracted payload
        const decoder = new TextDecoder();
        const extractedPayload = decoder.decode(extractedBytes);
        
        console.log(`Extracted payload: "${extractedPayload}"`);
        console.log(`Extraction successful: ${extractedPayload === payload ? 'YES' : 'NO'}`);
        
        // Show stylometric features before and after
        console.log("\n4. STYLOMETRIC FEATURES COMPARISON:");
        console.log("----------------------------------");
        const originalFeatures = carrier.analyzeCarryingCapacity(text);
        const modifiedFeatures = carrier.analyzeCarryingCapacity(modifiedText);
        
        console.log(`Original capacity: ${originalFeatures.totalCapacityBits} bits`);
        console.log(`Modified capacity: ${modifiedFeatures.totalCapacityBits} bits`);
    } catch (error) {
        console.error(`Error: ${error.message}`);
    }
    
    console.log("\n=== DEMO COMPLETE ===");
}