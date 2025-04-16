/**
 * Quote Style Carrier
 * 
 * This carrier encodes information by alternating between single and double quotes
 * to represent binary data. The choice between quote styles is stylistically neutral
 * in most contexts, making this technique relatively stealthy.
 * 
 * Flow:
 * Text → Identify quotes → Map to bits → Embed by replacing quote style → Output text
 * 
 * The happy path involves finding suitable quote marks in the text, encoding bits by
 * choosing appropriate quote styles, and ensuring the result maintains readability.
 */

import { Carrier } from '../stylometric_carrier.genai';

/**
 * Implements steganographic carrier that uses quote style variations 
 * (single vs double quotes) to embed information
 */
export class QuoteStyleCarrier implements Carrier {
    readonly id = "quote-style-carrier";
    readonly name = "Quote Style Carrier";
    readonly category = "punctuation";
    readonly detectability = 0.3; // Relatively low detectability
    readonly bitsPerThousandWords = 15; // Conservative estimate
    readonly resilience = 0.8; // High resilience to edits

    /**
     * Analyzes text to determine capacity for embedding using quote style variations
     * @param text The input text to analyze
     * @returns The number of bits that can be embedded
     */
    analyzeCapacity(text: string): number {
        // Count potential quote marks that can be modified
        const quotes = this.findQuotes(text);
        // Each quote pair can encode 1 bit
        return quotes.length;
    }

    /**
     * Embeds data by altering quote styles in the text
     * @param text The carrier text
     * @param bits Binary data to embed
     * @returns Modified text with embedded data
     */
    embed(text: string, bits: boolean[]): string {
        const quotes = this.findQuotes(text);
        let result = text;
        let bitIndex = 0;

        // Only process as many quotes as we have bits (or quotes)
        const quoteCount = Math.min(quotes.length, bits.length);
        
        for (let i = 0; i < quoteCount; i++) {
            if (bitIndex >= bits.length) break;
            
            const quote = quotes[i];
            // 0 = single quotes, 1 = double quotes
            const targetStyle = bits[bitIndex] ? '"' : "'";
            const replacementQuote = {
                start: targetStyle,
                end: targetStyle
            };

            // Replace end quote first (to avoid position shifts)
            result = 
                result.substring(0, quote.endPos) + 
                replacementQuote.end + 
                result.substring(quote.endPos + 1);
                
            // Replace start quote
            result = 
                result.substring(0, quote.startPos) + 
                replacementQuote.start + 
                result.substring(quote.startPos + 1);
                
            bitIndex++;
        }

        return result;
    }

    /**
     * Extracts embedded data from the quote styles
     * @param text The text with embedded data
     * @returns Extracted bits
     */
    extract(text: string): boolean[] {
        const quotes = this.findQuotes(text);
        const bits: boolean[] = [];
        
        for (const quote of quotes) {
            // Double quotes represent 1, single quotes represent 0
            bits.push(quote.style === '"');
        }
        
        return bits;
    }

    /**
     * Finds quote pairs in the text
     * @private
     * @param text The input text
     * @returns Array of quote information objects
     */
    private findQuotes(text: string): Array<{startPos: number, endPos: number, style: string}> {
        const quotes: Array<{startPos: number, endPos: number, style: string}> = [];
        let inQuote = false;
        let quoteStyle = '';
        let startPos = -1;
        
        for (let i = 0; i < text.length; i++) {
            const char = text[i];
            
            // Check for quote characters
            if ((char === '"' || char === "'") && 
                (i === 0 || text[i-1] !== '\\')) { // Not escaped
                
                if (!inQuote) {
                    // Start of quote
                    inQuote = true;
                    quoteStyle = char;
                    startPos = i;
                } else if (char === quoteStyle) {
                    // End of quote with matching style
                    inQuote = false;
                    quotes.push({
                        startPos: startPos,
                        endPos: i,
                        style: quoteStyle
                    });
                }
            }
        }
        
        return quotes;
    }
}
