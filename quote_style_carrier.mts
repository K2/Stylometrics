/**
 * Quote Style Carrier
 * 
 * Embeds information by altering the style of quotation marks (e.g., " vs “”).
 * Design Goals: Implement a reversible carrier using quote styles.
 * Constraints: Relies on consistent quote usage for extraction. May be sensitive to auto-formatting.
 * Paradigm: Imperative text manipulation.
 * Happy Path: Analyze text -> Identify quotes -> Apply style change based on bit -> Extract bit based on style.
 * ApiNotes: ./ApiNotes.md (File Level), ../capacity_matrix/ApiNotes.md (Directory Level)
 */
import type { CarrierTechnique } from './types/CarrierTypes.ts'; // Corrected import path extension

// Define the different quote styles
const STRAIGHT_DOUBLE = '"';
const CURLY_DOUBLE_OPEN = '“';
const CURLY_DOUBLE_CLOSE = '”';

// ApiNotes: ./ApiNotes.md
// Define specific implementation details for the quote style carrier.
// Capacity estimation should count replaceable quote pairs.
// Apply should replace quotes based on bits.
// Extract should identify quote style to determine bits.

export class QuoteStyleCarrier implements CarrierTechnique {
    analyze: any;
    estimateCapacity(text: string): number {
        throw new Error('Method not implemented.');
    }
    id = 'quote_style';
    name = 'Quote Style Modifier';
    category: CarrierTechnique['category'] = 'punctuation';
    bitsPerThousandWords = 15; // Estimate, depends heavily on quote frequency
    detectability = 0.15; // Low, often normalized by editors

    estimate(text: string): number {
        // Count occurrences of straight double quotes that can be replaced
        // A more robust implementation would parse sentences/dialogue
        // Also consider existing curly quotes as potential sites if the bit requires straight quotes
        const quoteMatches = text.match(new RegExp(`(${STRAIGHT_DOUBLE}|${CURLY_DOUBLE_OPEN})`, 'g'));
        // Each pair of quotes can potentially encode one bit
        return quoteMatches ? Math.floor(quoteMatches.length / 2) : 0;
    }

    apply(text: string, bits: boolean[]): { modifiedText: string; bitsEncoded: number } {
        // ApiNotes: ./quote_style_carrier.ApiNotes.md#Behavior
        // Refined logic to handle existing quotes and ensure bits are counted even if no change occurs.
        const potentialSites: { openIndex: number, closeIndex: number, currentStyle: 'straight' | 'curly' }[] = [];
        let inQuote = false;
        let lastQuoteIndex = -1;
        let currentStyle: 'straight' | 'curly' | null = null;

        // 1. Find all potential quote pairs (indices and current style).
        for (let i = 0; i < text.length; i++) {
            const char = text[i];
            if (!inQuote) {
                if (char === STRAIGHT_DOUBLE) {
                    lastQuoteIndex = i;
                    inQuote = true;
                    currentStyle = 'straight';
                } else if (char === CURLY_DOUBLE_OPEN) {
                    lastQuoteIndex = i;
                    inQuote = true;
                    currentStyle = 'curly';
                }
            } else {
                if (currentStyle === 'straight' && char === STRAIGHT_DOUBLE) {
                    potentialSites.push({ openIndex: lastQuoteIndex, closeIndex: i, currentStyle: 'straight' });
                    inQuote = false;
                    currentStyle = null;
                } else if (currentStyle === 'curly' && char === CURLY_DOUBLE_CLOSE) {
                    potentialSites.push({ openIndex: lastQuoteIndex, closeIndex: i, currentStyle: 'curly' });
                    inQuote = false;
                    currentStyle = null;
                } else if (char === STRAIGHT_DOUBLE || char === CURLY_DOUBLE_OPEN) {
                    // Mismatched or nested quote - reset and re-evaluate
                    inQuote = false;
                    currentStyle = null;
                    i--; // Re-process current character as potential start
                }
            }
        }

        let modifiedText = text;
        let bitsEncoded = 0;
        let currentBitIndex = 0;
        let indexAdjustment = 0; // Track index shifts due to replacements

        // 2. Iterate through pairs and bits, applying changes from end to start to avoid index issues.
        for (let siteIndex = potentialSites.length - 1; siteIndex >= 0; siteIndex--) {
            if (currentBitIndex >= bits.length) break; // No more bits to encode

            const site = potentialSites[siteIndex];
            const targetBit = bits[currentBitIndex]; // Read bit for this site
            const targetStyle = targetBit ? 'curly' : 'straight';

            let madeChange = false;
            let openQuote = '';
            let closeQuote = '';

            if (targetStyle === 'curly' && site.currentStyle === 'straight') {
                // 3. If bit requires curly and quotes are straight -> replace.
                openQuote = CURLY_DOUBLE_OPEN;
                closeQuote = CURLY_DOUBLE_CLOSE;
                madeChange = true;
            } else if (targetStyle === 'straight' && site.currentStyle === 'curly') {
                // 4. If bit requires straight and quotes are curly -> replace.
                openQuote = STRAIGHT_DOUBLE;
                closeQuote = STRAIGHT_DOUBLE;
                madeChange = true;
            }

            // 5. Consume bit even if no change needed.
            bitsEncoded++;
            currentBitIndex++; // Always advance bit index for each potential site processed

            if (madeChange) {
                const adjustedOpenIndex = site.openIndex + indexAdjustment;
                const adjustedCloseIndex = site.closeIndex + indexAdjustment;

                const beforeOpen = modifiedText.substring(0, adjustedOpenIndex);
                const between = modifiedText.substring(adjustedOpenIndex + (site.currentStyle === 'curly' ? CURLY_DOUBLE_OPEN.length : STRAIGHT_DOUBLE.length), adjustedCloseIndex);
                const afterClose = modifiedText.substring(adjustedCloseIndex + (site.currentStyle === 'curly' ? CURLY_DOUBLE_CLOSE.length : STRAIGHT_DOUBLE.length));

                modifiedText = beforeOpen + openQuote + between + closeQuote + afterClose;

                // Calculate the change in length caused by this replacement
                const oldLength = (site.currentStyle === 'curly' ? CURLY_DOUBLE_OPEN.length + CURLY_DOUBLE_CLOSE.length : STRAIGHT_DOUBLE.length + STRAIGHT_DOUBLE.length);
                const newLength = openQuote.length + closeQuote.length;
                indexAdjustment += (newLength - oldLength);
            }
        }
        // Reverse the bits encoded count as we iterated backwards over bits
        // No, currentBitIndex tracks how many bits we *attempted* to encode from the start.
        // bitsEncoded should reflect how many sites we processed.

        // The number of bits successfully encoded is the number of sites we processed up to the available bits.
        bitsEncoded = Math.min(potentialSites.length, bits.length);


        return { modifiedText, bitsEncoded };
    }

    extract(text: string): boolean[] {
        const extractedBits: boolean[] = [];
        let inQuote = false;
        let potentialBit: boolean | null = null;

        for (let i = 0; i < text.length; i++) {
            const char = text[i];
            if (!inQuote) {
                if (char === CURLY_DOUBLE_OPEN) {
                    potentialBit = true; // Assume bit is true if curly quotes start
                    inQuote = true;
                } else if (char === STRAIGHT_DOUBLE) {
                    potentialBit = false; // Assume bit is false if straight quotes start
                    inQuote = true;
                }
            } else {
                // Check for closing quote matching the opening style
                if (potentialBit === true && char === CURLY_DOUBLE_CLOSE) {
                    extractedBits.push(true);
                    inQuote = false;
                    potentialBit = null;
                } else if (potentialBit === false && char === STRAIGHT_DOUBLE) {
                    extractedBits.push(false);
                    inQuote = false;
                    potentialBit = null;
                } else if (char === CURLY_DOUBLE_OPEN || char === STRAIGHT_DOUBLE || char === CURLY_DOUBLE_CLOSE /* Handle unexpected closer */) {
                    // Mismatched or nested quote? Ignore this pair for simplicity.
                    // Or potentially an unclosed quote followed by a new one.
                    inQuote = false; // Reset state
                    potentialBit = null;
                    // Re-evaluate current char as a potential opener
                    i--;
                }
                // Consider edge case: end of text while still inQuote? Discard potential bit.
            }
        }
        return extractedBits;
    }

    // Implement CarrierTechnique methods
    getDetectability(): number { return this.detectability; }
    getCapacity(text: string): number { return this.estimate(text); }
    getNaturalness(): number { return 0.8; } // Curly quotes often preferred
    encode(text: string, bits: boolean[]): { modifiedText: string; bitsEncoded: number } { return this.apply(text, bits); }
    getRobustness(): number { return 0.4; } // Easily normalized by editors/formatters
}

// Add ApiNotes.md for this file
// ```markdown
// // filepath: /home/files/git/Stylometrics/quote_style_carrier.ApiNotes.md
// # Quote Style Carrier - ApiNotes
//
// ## Design
// Encodes data by switching between straight (`"`) and curly (`“”`) double quotation marks.
// - Bit `1` (true) is represented by curly quotes (`“”`).
// - Bit `0` (false) is represented by straight quotes (`"`).
//
// ## Behavior
// - `estimate`: Counts pairs of straight or curly quotes as potential sites.
// - `apply`: Finds all quote pairs (straight or curly). Iterates through potential sites and input bits *backwards* through the text to simplify index management during replacement. Replaces quote style if it doesn't match the target bit. Correctly reports `bitsEncoded` based on the minimum of available sites and input bits.
// - `extract`: Iterates through text, identifying quote pairs and determining the bit based on whether they are straight or curly. Basic handling for mismatched/nested quotes.
//
// ## Rationale
// Quote style is a subtle feature often overlooked but easily manipulated. Curly quotes are common in published text, straight quotes in code/plain text.
//
// ## Constraints
// - Assumes standard English quote pairing.
// - Sensitive to text normalization tools that enforce a single quote style.
// - Current implementation might still misinterpret complex quoting scenarios (deeply nested, mixed single/double).
// - Does not handle single quotes.
// ```