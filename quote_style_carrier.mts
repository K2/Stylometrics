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

import { FeatureMap } from './stylometric_detection.genai.mjs';

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
    modified?: boolean;
}

/**
 * Quote style encoding options
 */
export interface QuoteStyleOptions {
    preferDoubleQuotes?: boolean;   // Whether to prefer double quotes as the primary style
    useSmartQuotes?: boolean;       // Whether to use curly/typographic quotes
    strictConsistency?: boolean;    // Whether to enforce strict consistency in style per context
    balanceQuotationStyles?: boolean; // Whether to maintain balanced distribution of styles
}

/**
 * QuoteStyleCarrier provides steganographic encoding using quotation mark styles
 */
export class QuoteStyleCarrier {
    // Standard quote characters
    private straightSingleQuote = "'";
    private straightDoubleQuote = "\"";
    private curlySingleOpenQuote = "'";
    private curlySingleCloseQuote = "'";
    private curlyDoubleOpenQuote = """;
    private curlyDoubleCloseQuote = """;
    
    // Options
    private options: QuoteStyleOptions;
    
    /**
     * Initialize the quote style carrier
     * @param options Encoding options
     */
    constructor(options: QuoteStyleOptions = {}) {
        this.options = {
            preferDoubleQuotes: options.preferDoubleQuotes ?? true,
            useSmartQuotes: options.useSmartQuotes ?? false,
            strictConsistency: options.strictConsistency ?? true,
            balanceQuotationStyles: options.balanceQuotationStyles ?? false
        };
    }
    
    /**
     * Analyze text to determine quote carrier capacity
     * @param text Text to analyze
     * @returns Analysis of carrying capacity
     */
    analyzeCapacity(text: string): {
        quotePairs: number;
        capacityBits: number;
        quotationStyles: FeatureMap;
    } {
        const pairs = this.findQuotePairs(text);
        
        // Count different quote styles
        const quotationStyles: FeatureMap = {};
        quotationStyles.singleStraightQuotes = 0;
        quotationStyles.doubleStraightQuotes = 0;
        quotationStyles.singleCurlyQuotes = 0;
        quotationStyles.doubleCurlyQuotes = 0;
        
        for (const pair of pairs) {
            // Identify quote style
            if (pair.opening.char === this.straightSingleQuote) {
                quotationStyles.singleStraightQuotes++;
            } else if (pair.opening.char === this.straightDoubleQuote) {
                quotationStyles.doubleStraightQuotes++;
            } else if (pair.opening.char === this.curlySingleOpenQuote) {
                quotationStyles.singleCurlyQuotes++;
            } else if (pair.opening.char === this.curlyDoubleOpenQuote) {
                quotationStyles.doubleCurlyQuotes++;
            }
        }
        
        // Calculate capacity (conservative estimate: 1 bit per 2 pairs)
        // In practice, we can encode 1 bit per pair, but we use a conservative
        // estimate to account for consistency constraints
        const capacityBits = Math.floor(pairs.length / 2);
        
        return {
            quotePairs: pairs.length,
            capacityBits,
            quotationStyles
        };
    }
    
    /**
     * Encode bits into text using quote styles
     * @param text Original text
     * @param bits Bits to encode
     * @returns Modified text with encoded bits
     */
    encode(text: string, bits: boolean[]): {
        modifiedText: string;
        bitsEncoded: number;
        quotePairsModified: number;
    } {
        // Find quotation pairs
        const pairs = this.findQuotePairs(text);
        
        // Determine how many bits we can encode
        const maxBits = Math.min(bits.length, pairs.length);
        
        // If no bits to encode or no pairs, return original text
        if (maxBits === 0 || pairs.length === 0) {
            return { 
                modifiedText: text, 
                bitsEncoded: 0, 
                quotePairsModified: 0 
            };
        }
        
        // Group pairs by context to maintain consistency
        const pairsByContext = this.groupPairsByContext(pairs, text);
        
        // Track which pairs are modified
        let quotePairsModified = 0;
        
        // For each encodable bit, assign a quotation style
        for (let i = 0; i < maxBits; i++) {
            const pair = pairs[i];
            const bit = bits[i];
            
            // Get the appropriate quote style for this bit
            const quoteStyle = this.getQuoteStyleForBit(
                bit, 
                pair.opening.char, 
                pair.closing.char,
                pairsByContext
            );
            
            // Apply the style to the pair
            if (quoteStyle.openChar !== pair.opening.char || 
                quoteStyle.closeChar !== pair.closing.char) {
                pair.modified = true;
                quotePairsModified++;
            }
        }
        
        // Apply modifications to text (starting from the end to preserve indices)
        let modifiedText = text;
        pairs.sort((a, b) => b.closing.index - a.closing.index); // Sort in reverse order
        
        for (const pair of pairs) {
            // Skip unmodified pairs
            if (!pair.modified) continue;
            
            // Get the appropriate quote style for this pair
            const context = this.getPairContext(pair, text);
            const consistentStyle = this.getConsistentStyle(pair, pairsByContext.get(context) || [], this.options);
            
            // Replace closing quote
            modifiedText = 
                modifiedText.substring(0, pair.closing.index) + 
                consistentStyle.closeChar + 
                modifiedText.substring(pair.closing.index + 1);
                
            // Replace opening quote
            modifiedText = 
                modifiedText.substring(0, pair.opening.index) + 
                consistentStyle.openChar + 
                modifiedText.substring(pair.opening.index + 1);
        }
        
        return {
            modifiedText,
            bitsEncoded: maxBits,
            quotePairsModified
        };
    }
    
    /**
     * Extract bits from text that was encoded with quote styles
     * @param text Encoded text
     * @param expectedBits Number of bits to extract (if known)
     * @returns Extracted bits
     */
    extract(text: string, expectedBits?: number): boolean[] {
        // Find quotation pairs
        const pairs = this.findQuotePairs(text);
        
        // Determine how many bits to extract
        const numBits = expectedBits ?? pairs.length;
        const maxBits = Math.min(numBits, pairs.length);
        
        // Extract bits based on quote styles
        const extractedBits: boolean[] = [];
        
        for (let i = 0; i < maxBits; i++) {
            const pair = pairs[i];
            
            // Determine bit based on quote style
            const bit = this.getBitFromQuoteStyle(pair.opening.char, pair.closing.char);
            extractedBits.push(bit);
        }
        
        return extractedBits;
    }
    
    /**
     * Find matching quotation pairs in text
     */
    private findQuotePairs(text: string): QuotePair[] {
        const quoteChars = [
            this.straightSingleQuote, 
            this.straightDoubleQuote,
            this.curlySingleOpenQuote,
            this.curlySingleCloseQuote, 
            this.curlyDoubleOpenQuote,
            this.curlyDoubleCloseQuote
        ];
        
        // Find all quote characters
        const quotePositions: QuotePosition[] = [];
        
        for (const char of quoteChars) {
            let pos = text.indexOf(char);
            while (pos !== -1) {
                quotePositions.push({ index: pos, char });
                pos = text.indexOf(char, pos + 1);
            }
        }
        
        // Sort positions by index
        quotePositions.sort((a, b) => a.index - b.index);
        
        // Match quotes into pairs
        const pairs: QuotePair[] = [];
        const openingQuotes: QuotePosition[] = [];
        
        for (const pos of quotePositions) {
            // Handle different quote types
            if (this.isOpeningQuote(pos.char)) {
                // Opening quote
                openingQuotes.push(pos);
            } else if (this.isClosingQuote(pos.char) && openingQuotes.length > 0) {
                // Find matching opening quote (last one of compatible type)
                let matchingIndex = -1;
                for (let i = openingQuotes.length - 1; i >= 0; i--) {
                    if (this.isMatchingPair(openingQuotes[i].char, pos.char)) {
                        matchingIndex = i;
                        break;
                    }
                }
                
                if (matchingIndex >= 0) {
                    const opening = openingQuotes[matchingIndex];
                    
                    // Create a quote pair
                    pairs.push({
                        opening,
                        closing: pos,
                        content: text.substring(opening.index + 1, pos.index)
                    });
                    
                    // Remove the opening quote from the stack
                    openingQuotes.splice(matchingIndex, 1);
                }
            }
        }
        
        return pairs;
    }
    
    /**
     * Group pairs by contextual surroundings to maintain consistency
     */
    private groupPairsByContext(pairs: QuotePair[], text: string): Map<string, QuotePair[]> {
        const contextMap = new Map<string, QuotePair[]>();
        
        for (const pair of pairs) {
            const context = this.getPairContext(pair, text);
            
            if (!contextMap.has(context)) {
                contextMap.set(context, []);
            }
            
            contextMap.get(context)!.push(pair);
        }
        
        return contextMap;
    }
    
    /**
     * Get the context surrounding a quote pair (used for consistency)
     */
    private getPairContext(pair: QuotePair, text: string): string {
        // Get words before and after the quote as context
        const contextStart = Math.max(0, pair.opening.index - 15);
        const contextEnd = Math.min(text.length, pair.closing.index + 15);
        
        const beforeText = text.substring(contextStart, pair.opening.index).trim();
        const afterText = text.substring(pair.closing.index + 1, contextEnd).trim();
        
        // Use simplified context identifier
        const lastWordBefore = beforeText.split(/\s+/).pop() || "";
        const firstWordAfter = afterText.split(/\s+/)[0] || "";
        
        return `${lastWordBefore}|${firstWordAfter}`;
    }
    
    /**
     * Get appropriate quote style for encoded bit
     */
    private getQuoteStyleForBit(
        bit: boolean,
        currentOpenChar: string,
        currentCloseChar: string,
        contextGroups: Map<string, QuotePair[]>
    ): { openChar: string; closeChar: string } {
        // Determine quote styles based on bit value
        // bit = 0: single quotes (straight or curly based on options)
        // bit = 1: double quotes (straight or curly based on options)
        
        const preferDouble = this.options.preferDoubleQuotes;
        const useSmartQuotes = this.options.useSmartQuotes;
        
        let openChar: string, closeChar: string;
        
        if (bit === preferDouble) {
            // Use double quotes for matching bit
            if (useSmartQuotes) {
                openChar = this.curlyDoubleOpenQuote;
                closeChar = this.curlyDoubleCloseQuote;
            } else {
                openChar = this.straightDoubleQuote;
                closeChar = this.straightDoubleQuote;
            }
        } else {
            // Use single quotes for non-matching bit
            if (useSmartQuotes) {
                openChar = this.curlySingleOpenQuote;
                closeChar = this.curlySingleCloseQuote;
            } else {
                openChar = this.straightSingleQuote;
                closeChar = this.straightSingleQuote;
            }
        }
        
        return { openChar, closeChar };
    }
    
    /**
     * Get consistent quote style for a pair based on context
     */
    private getConsistentStyle(
        pair: QuotePair,
        contextPairs: QuotePair[],
        options: QuoteStyleOptions
    ): { openChar: string; closeChar: string } {
        // For strict consistency mode, all quotes in the same context
        // should use the same style
        if (options.strictConsistency && contextPairs.length > 1) {
            // Find another modified pair in the same context to match style
            const otherModifiedPair = contextPairs.find(
                p => p !== pair && p.modified
            );
            
            if (otherModifiedPair) {
                return {
                    openChar: otherModifiedPair.opening.char,
                    closeChar: otherModifiedPair.closing.char
                };
            }
        }
        
        // Otherwise use the style assigned to this pair (or original style if not modified)
        return {
            openChar: pair.opening.char,
            closeChar: pair.closing.char
        };
    }
    
    /**
     * Check if character is an opening quote
     */
    private isOpeningQuote(char: string): boolean {
        return char === this.straightSingleQuote || 
               char === this.straightDoubleQuote ||
               char === this.curlySingleOpenQuote || 
               char === this.curlyDoubleOpenQuote;
    }
    
    /**
     * Check if character is a closing quote
     */
    private isClosingQuote(char: string): boolean {
        return char === this.straightSingleQuote || 
               char === this.straightDoubleQuote ||
               char === this.curlySingleCloseQuote || 
               char === this.curlyDoubleCloseQuote;
    }
    
    /**
     * Check if opening and closing quotes form a matching pair
     */
    private isMatchingPair(openChar: string, closeChar: string): boolean {
        if (openChar === this.straightSingleQuote && closeChar === this.straightSingleQuote) return true;
        if (openChar === this.straightDoubleQuote && closeChar === this.straightDoubleQuote) return true;
        if (openChar === this.curlySingleOpenQuote && closeChar === this.curlySingleCloseQuote) return true;
        if (openChar === this.curlyDoubleOpenQuote && closeChar === this.curlyDoubleCloseQuote) return true;
        return false;
    }
    
    /**
     * Determine bit value from quote style
     */
    private getBitFromQuoteStyle(openChar: string, closeChar: string): boolean {
        const preferDouble = this.options.preferDoubleQuotes;
        
        // Check if it's a double quote style
        const isDoubleQuote = 
            (openChar === this.straightDoubleQuote && closeChar === this.straightDoubleQuote) ||
            (openChar === this.curlyDoubleOpenQuote && closeChar === this.curlyDoubleCloseQuote);
            
        // If preferDouble=true, double quotes encode 1, single quotes encode 0
        // If preferDouble=false, double quotes encode 0, single quotes encode 1
        return isDoubleQuote === preferDouble;
    }
}

/**
 * Demonstrate the quote style carrier
 * 
 * @param text Sample text containing quotations
 * @param bits Bits to encode
 */
export function demonstrateQuoteStyleCarrier(text: string, bits: boolean[]): void {
    console.log("=== QUOTE STYLE CARRIER DEMO ===");
    
    // Create carrier
    const carrier = new QuoteStyleCarrier({ useSmartQuotes: true });
    
    // Analyze capacity
    const analysis = carrier.analyzeCapacity(text);
    console.log(`Quote pairs found: ${analysis.quotePairs}`);
    console.log(`Estimated capacity: ${analysis.capacityBits} bits`);
    console.log("Quote styles found:");
    console.log(`- Single straight quotes: ${analysis.quotationStyles.singleStraightQuotes}`);
    console.log(`- Double straight quotes: ${analysis.quotationStyles.doubleStraightQuotes}`);
    console.log(`- Single curly quotes: ${analysis.quotationStyles.singleCurlyQuotes}`);
    console.log(`- Double curly quotes: ${analysis.quotationStyles.doubleCurlyQuotes}`);
    
    // Encode bits
    console.log(`\nEncoding ${bits.length} bits: ${bits.join(', ')}`);
    const { modifiedText, bitsEncoded, quotePairsModified } = carrier.encode(text, bits);
    
    console.log(`\nOriginal text (first 100 chars):`);
    console.log(text.substring(0, 100) + "...");
    console.log(`\nModified text (first 100 chars):`);
    console.log(modifiedText.substring(0, 100) + "...");
    console.log(`\nBits encoded: ${bitsEncoded}`);
    console.log(`Quote pairs modified: ${quotePairsModified}`);
    
    // Extract bits
    const extractedBits = carrier.extract(modifiedText, bits.length);
    console.log(`\nExtracted bits: ${extractedBits.join(', ')}`);
    
    // Verify correctness
    const correct = JSON.stringify(extractedBits) === JSON.stringify(bits.slice(0, bitsEncoded));
    console.log(`Extraction successful: ${correct ? 'YES' : 'NO'}`);
    
    console.log("\n=== DEMO COMPLETE ===");
}