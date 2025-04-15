/**
 * Quote Style Carrier Implementation
 *
 * This module provides a specific implementation of the Quote Style carrier technique
 * identified in the stylometric carrier framework. It encodes information by
 * alternating between different quotation styles (single/double quotes, curly quotes, etc.)
 *
 * This technique has a low detectability (0.2/1.0) as identified by analyzing the
 * Kumarage et al. research on stylometric features.
 *
 * Flow:
 * 1. Identify quotation pairs in text
 * 2. For each bit to encode, modify a quotation pair's style
 * 3. Ensure modifications remain natural and consistent within context
 */

// Assuming FeatureMap might be needed for other carriers, keep import for now
import { FeatureMap } from './stylometric_detection.genai.mjs';
// Import necessary types from CarrierTypes
import { type Carrier, type CarrierMetrics, type EncodeResult, type CarrierConfiguration } from './src/types/CarrierTypes.js'; // Assuming CarrierTypes.js is the compiled output or adjust path/extension

/**
 * Interface for a quote position in text
 */
interface QuotePosition {
    index: number;
    char: string;
}

/**
 * Interface for a matched quotation pair
 */
interface QuotePair {
    opening: QuotePosition;
    closing: QuotePosition;
    content: string;
    modified?: boolean; // Internal state for encoding logic
    assignedBit?: boolean; // Store the bit assigned during encoding
    originalOpenChar?: string; // Store original style if needed for consistency checks
    originalCloseChar?: string;
}

/**
 * Quote style encoding options specific to this carrier
 */
export interface QuoteStyleOptions extends CarrierConfiguration { // Inherit base config
    preferDoubleQuotes?: boolean;   // Whether to prefer double quotes as the primary style (maps to bit 1)
    useSmartQuotes?: boolean;       // Whether to use curly/typographic quotes
    strictConsistency?: boolean;    // Whether to enforce strict consistency in style per context (e.g., paragraph)
}

/**
 * QuoteStyleCarrier provides steganographic encoding using quotation mark styles.
 * Implements the Carrier interface.
 */
export class QuoteStyleCarrier implements Carrier {
    readonly id: string = 'quoteStyle';
    readonly name: string = 'Quotation Mark Style Alternation';

    // Standard quote characters
    private straightSingleQuote = "'";
    private straightDoubleQuote = "\"";
    // Note: Using standard ASCII for curly quotes for broader compatibility,
    // replace with actual Unicode if needed: ‘ ’ “ ”
    private curlySingleOpenQuote = "'";
    private curlySingleCloseQuote = "'";
    private curlyDoubleOpenQuote = "\"";
    private curlyDoubleCloseQuote = "\"";

    // Default configuration
    private configuration: Required<QuoteStyleOptions>;

    /**
     * Initialize the quote style carrier
     * @param config Configuration options
     */
    constructor(config: QuoteStyleOptions = {}) {
        this.configuration = {
            preferDoubleQuotes: config.preferDoubleQuotes ?? true,
            useSmartQuotes: config.useSmartQuotes ?? false, // Default to false for simplicity
            strictConsistency: config.strictConsistency ?? false, // Default to false
            // Inherited defaults (adjust as needed for this carrier)
            maxDetectability: config.maxDetectability ?? 0.3,
            minRobustness: config.minRobustness ?? 0.2, // Quotes are easily changed
            prioritizeCapacity: config.prioritizeCapacity ?? false,
            ...config // Allow overriding inherited defaults
        };
    }

    /**
     * Get the current configuration.
     */
    getConfiguration(): Record<string, any> {
        return { ...this.configuration };
    }

    /**
     * Set configuration parameters.
     */
    configure(config: Record<string, any>): void {
        this.configuration = { ...this.configuration, ...config };
    }

    /**
     * Analyze text to determine quote carrier capacity and metrics.
     * @param content Text to analyze.
     * @returns Promise resolving to CarrierMetrics.
     */
    async analyzeCapacity(content: string): Promise<CarrierMetrics> {
        const pairs = this.findQuotePairs(content);
        const capacity = pairs.length; // Each pair can potentially encode 1 bit

        // Estimate metrics (these are subjective and context-dependent)
        const detectability = 0.2; // Relatively low if styles are mixed naturally
        const robustness = 0.2;    // Low, easily altered by formatting/editing
        const naturalness = 0.6;   // Can be natural if consistent, less so if mixed randomly

        return {
            capacity: capacity,
            detectability: detectability,
            robustness: robustness,
            naturalness: naturalness
        };
    }

    /**
     * Encode bits into text using quote styles.
     * @param content Original text content.
     * @param bits Bits to encode.
     * @returns Promise resolving to EncodeResult.
     */
    async encode(content: string, bits: boolean[]): Promise<EncodeResult> {
        const pairs = this.findQuotePairs(content);
        const maxBits = Math.min(bits.length, pairs.length);

        if (maxBits === 0) {
            return { modifiedContent: content, bitsEncoded: 0 };
        }

        let modifiedContent = content;
        let bitsEncoded = 0;
        const modifications: { index: number; newChar: string }[] = [];

        // Assign bits to pairs and determine necessary changes
        for (let i = 0; i < maxBits; i++) {
            const pair = pairs[i];
            const bit = bits[i];
            pair.assignedBit = bit; // Store assigned bit for potential consistency logic

            const targetStyle = this.getQuoteStyleForBit(bit);

            // Check if opening quote needs changing
            if (pair.opening.char !== targetStyle.openChar) {
                modifications.push({ index: pair.opening.index, newChar: targetStyle.openChar });
                pair.modified = true;
            }
            // Check if closing quote needs changing
            if (pair.closing.char !== targetStyle.closeChar) {
                modifications.push({ index: pair.closing.index, newChar: targetStyle.closeChar });
                pair.modified = true;
            }

            if (pair.modified) {
                bitsEncoded++; // Count bit as encoded if a modification is planned
            }
        }

        // Apply modifications in reverse index order to avoid shifting subsequent indices
        modifications.sort((a, b) => b.index - a.index);

        const contentChars = modifiedContent.split('');
        for (const mod of modifications) {
            if (mod.index >= 0 && mod.index < contentChars.length) {
                contentChars[mod.index] = mod.newChar;
            } else {
                console.warn(`QuoteStyleCarrier: Invalid modification index ${mod.index}`);
            }
        }
        modifiedContent = contentChars.join('');

        return {
            modifiedContent: modifiedContent,
            bitsEncoded: bitsEncoded // Return the number of bits we attempted to encode by modification
        };
    }

    /**
     * Extract bits from text that was encoded with quote styles.
     * @param content Encoded text content.
     * @returns Promise resolving to the extracted bits or null.
     */
    async extract(content: string): Promise<boolean[] | null> {
        const pairs = this.findQuotePairs(content);
        if (pairs.length === 0) {
            return []; // Return empty array if no pairs found
        }

        const extractedBits: boolean[] = [];
        for (const pair of pairs) {
            const bit = this.getBitFromQuoteStyle(pair.opening.char, pair.closing.char);
            extractedBits.push(bit);
        }

        return extractedBits;
    }

    /**
     * Find matching quotation pairs in text.
     * Improved logic to handle nested quotes better.
     */
    private findQuotePairs(text: string): QuotePair[] {
        const pairs: QuotePair[] = [];
        const stack: QuotePosition[] = [];
        const quoteCharsRegex = /['"“”‘’]/g; // Include curly quotes if used
        let match;

        while ((match = quoteCharsRegex.exec(text)) !== null) {
            const index = match.index;
            const char = match[0];
            const currentPos: QuotePosition = { index, char };

            const top = stack.length > 0 ? stack[stack.length - 1] : null;

            if (top && this.isMatchingPair(top.char, char)) {
                // Found a closing quote matching the top of the stack
                const openingPos = stack.pop()!;
                pairs.push({
                    opening: openingPos,
                    closing: currentPos,
                    content: text.substring(openingPos.index + 1, currentPos.index),
                    originalOpenChar: openingPos.char, // Store original chars
                    originalCloseChar: currentPos.char,
                });
            } else if (this.isOpeningQuote(char)) {
                // Found an opening quote, push onto stack
                stack.push(currentPos);
            }
        }

        // Sort pairs by their appearance order (opening index)
        pairs.sort((a, b) => a.opening.index - b.opening.index);
        return pairs;
    }

    /**
     * Get appropriate quote style for encoded bit based on configuration.
     */
    private getQuoteStyleForBit(bit: boolean): { openChar: string; closeChar: string } {
        const useSmart = this.configuration.useSmartQuotes;
        const preferDouble = this.configuration.preferDoubleQuotes;

        // Determine target style: double or single
        const useDouble = (bit === preferDouble); // If bit matches preference, use double

        if (useDouble) {
            return useSmart
                ? { openChar: this.curlyDoubleOpenQuote, closeChar: this.curlyDoubleCloseQuote }
                : { openChar: this.straightDoubleQuote, closeChar: this.straightDoubleQuote };
        } else {
            return useSmart
                ? { openChar: this.curlySingleOpenQuote, closeChar: this.curlySingleCloseQuote }
                : { openChar: this.straightSingleQuote, closeChar: this.straightSingleQuote };
        }
    }

    /**
     * Determine bit value from quote style based on configuration.
     */
    private getBitFromQuoteStyle(openChar: string, closeChar: string): boolean {
        const preferDouble = this.configuration.preferDoubleQuotes;

        const isDouble =
            (openChar === this.straightDoubleQuote && closeChar === this.straightDoubleQuote) ||
            (openChar === this.curlyDoubleOpenQuote && closeChar === this.curlyDoubleCloseQuote);

        // If preferDouble=true, double quotes mean bit=1, single mean bit=0
        // If preferDouble=false, double quotes mean bit=0, single mean bit=1
        return isDouble === preferDouble;
    }

    /** Check if character is an opening quote */
    private isOpeningQuote(char: string): boolean {
        return char === this.straightDoubleQuote ||
               char === this.straightSingleQuote ||
               char === this.curlyDoubleOpenQuote ||
               char === this.curlySingleOpenQuote;
    }

    /** Check if character is a closing quote */
    private isClosingQuote(char: string): boolean {
        return char === this.straightDoubleQuote ||
               char === this.straightSingleQuote ||
               char === this.curlyDoubleCloseQuote ||
               char === this.curlySingleCloseQuote;
    }

    /** Check if opening and closing quotes form a matching pair type */
    private isMatchingPair(openChar: string, closeChar: string): boolean {
        return (
            (openChar === this.straightSingleQuote && closeChar === this.straightSingleQuote) ||
            (openChar === this.straightDoubleQuote && closeChar === this.straightDoubleQuote) ||
            (openChar === this.curlySingleOpenQuote && closeChar === this.curlySingleCloseQuote) ||
            (openChar === this.curlyDoubleOpenQuote && closeChar === this.curlyDoubleCloseQuote)
        );
    }
}

/**
 * Demonstrate the quote style carrier (Updated)
 *
 * @param text Sample text containing quotations
 * @param bits Bits to encode
 */
export async function demonstrateQuoteStyleCarrier(text: string, bits: boolean[]): Promise<void> {
    console.log("=== QUOTE STYLE CARRIER DEMO ===");

    // Create carrier instance
    const carrier = new QuoteStyleCarrier({ useSmartQuotes: false, preferDoubleQuotes: true });

    try {
        // Analyze capacity
        const metrics = await carrier.analyzeCapacity(text);
        console.log(`Carrier: ${carrier.name}`);
        console.log(`Estimated capacity: ${metrics.capacity} bits`);
        console.log(`Estimated metrics: Detectability=${metrics.detectability.toFixed(2)}, Robustness=${metrics.robustness.toFixed(2)}, Naturalness=${metrics.naturalness.toFixed(2)}`);

        // Encode bits
        console.log(`\nEncoding ${bits.length} bits: [${bits.map(b => b ? 1 : 0).join('')}]`);
        console.log(`Original text (first 100 chars):\n${text.substring(0, 100)}...`);

        const encodeResult = await carrier.encode(text, bits);

        console.log(`\nModified text (first 100 chars):\n${encodeResult.modifiedContent.substring(0, 100)}...`);
        console.log(`Bits encoded reported: ${encodeResult.bitsEncoded}`);

        if (encodeResult.bitsEncoded < bits.length) {
            console.warn(`Warning: Not all bits may have been encoded (${encodeResult.bitsEncoded}/${bits.length})`);
        }
        if (encodeResult.modifiedContent === text && encodeResult.bitsEncoded > 0) {
            console.warn("Warning: Carrier reported bits encoded but text was not modified.");
        }
        if (encodeResult.modifiedContent !== text && encodeResult.bitsEncoded === 0) {
            console.warn("Warning: Carrier modified text but reported 0 bits encoded.");
        }

        // Extract bits
        const extractedBits = await carrier.extract(encodeResult.modifiedContent);

        if (extractedBits) {
            console.log(`\nExtracted bits (${extractedBits.length}): [${extractedBits.map(b => b ? 1 : 0).join('')}]`);

            // Verify correctness up to the number of bits reportedly encoded
            const originalBitsToCompare = bits.slice(0, encodeResult.bitsEncoded);
            const extractedBitsToCompare = extractedBits.slice(0, encodeResult.bitsEncoded);

            let match = originalBitsToCompare.length === extractedBitsToCompare.length;
            if (match) {
                for (let i = 0; i < originalBitsToCompare.length; i++) {
                    if (originalBitsToCompare[i] !== extractedBitsToCompare[i]) {
                        match = false;
                        break;
                    }
                }
            }
            console.log(`Extraction matches encoded portion: ${match ? 'YES' : 'NO'}`);
            if (!match) {
                console.log("Original bits: ", originalBitsToCompare.map(b => b ? 1 : 0).join(''));
                console.log("Extracted bits:", extractedBitsToCompare.map(b => b ? 1 : 0).join(''));
            }
        } else {
            console.error("\nExtraction failed, returned null.");
        }

    } catch (error) {
        console.error("\n--- DEMO FAILED ---");
        if (error instanceof Error) {
            console.error(`Error during demonstration: ${error.message}`);
            if (error.stack) {
                console.error(error.stack);
            }
        } else {
            console.error("An unexpected error occurred during demonstration:", error);
        }
    }

    console.log("\n=== DEMO COMPLETE ===");
}

// Example Usage:
// const sampleText = "He said, \"This is 'important'.\" Then she asked, 'Really?'";
// const sampleBits = [true, false, true, false]; // 1 0 1 0
// demonstrateQuoteStyleCarrier(sampleText, sampleBits);