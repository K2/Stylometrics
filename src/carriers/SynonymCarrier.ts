/**
 * @module src/carriers/SynonymCarrier
 * @description (Placeholder) Implements the Carrier interface for embedding data
 * by choosing between synonymous words.
 * See: src/carriers/ApiNotes.md, src/types/CarrierTypes.ts
 *
 * NOTE: This requires a thesaurus and NLP context analysis for natural substitution.
 * This is a basic placeholder.
 */
import { Carrier, CarrierMetrics, EncodeResult, CarrierConfiguration } from "../types/CarrierTypes";
import { DocumentSegment } from "../types/DocumentTypes";
import { BaseCarrierImpl } from "./BaseCarrierImpl";

// Placeholder thesaurus
const SYNONYM_PAIRS: { [key: string]: string[] } = {
    "use": ["utilize"],
    "large": ["big"],
    "small": ["little"],
    "fast": ["quick"],
    "begin": ["start"],
    "end": ["finish"],
    "help": ["assist"],
    "show": ["demonstrate"],
};

export class SynonymCarrier extends BaseCarrierImpl implements Carrier {
    readonly key = "synonym";
    readonly description = "Embeds data by choosing between synonymous words.";

    constructor(private config: CarrierConfiguration = {}) {
        super();
    }

    async analyzeCapacity(segment: DocumentSegment): Promise<CarrierMetrics> {
        // Placeholder: Count occurrences of words in our mini-thesaurus.
        const text = segment.content?.toLowerCase() || "";
        let potentialBits = 0;
        for (const word in SYNONYM_PAIRS) {
            potentialBits += (text.match(new RegExp(`\\b${word}\\b`, 'g')) || []).length;
            SYNONYM_PAIRS[word].forEach(syn => {
                 potentialBits += (text.match(new RegExp(`\\b${syn}\\b`, 'g')) || []).length;
            });
        }
        // Assume we can use roughly 1/5th of potential locations
        const capacity = Math.floor(potentialBits / 5);
        return {
            capacityBits: capacity,
            detectability: 0.4, // Can be detectable if choices are unnatural
            robustness: 0.5,    // Moderately robust to simple edits
        };
    }

    async encode(segment: DocumentSegment, bits: boolean[]): Promise<EncodeResult> {
        // Placeholder: Simulate encoding by appending bits visually.
        // A real implementation would find words and swap them based on bits.
        let modifiedContent = segment.content || "";
        let bitsEncoded = 0;
        const capacity = (await this.analyzeCapacity(segment)).capacityBits;
        const bitsToEncodeCount = Math.min(bits.length, capacity);

        if (bitsToEncodeCount > 0) {
            const bitString = bits.slice(0, bitsToEncodeCount).map(b => b ? '1' : '0').join('');
            modifiedContent += `[SYN:${bitString}]`; // Visual marker
            bitsEncoded = bitsToEncodeCount;
            console.log(`[SynonymCarrier] Placeholder encoded ${bitsEncoded} bits into segment ${segment.id}`);
        }

        return {
            modifiedContent: modifiedContent,
            bitsEncoded: bitsEncoded,
        };
    }

    async extract(segment: DocumentSegment): Promise<boolean[]> {
        // Placeholder: Extract bits from the visual marker.
        const text = segment.content || "";
        const matches = text.matchAll(/\[SYN:([01]+)\]/g);
        let allBits: boolean[] = [];
        for (const match of matches) {
            const bitString = match[1];
            allBits = allBits.concat(bitString.split('').map(b => b === '1'));
        }
         console.log(`[SynonymCarrier] Placeholder extracted ${allBits.length} bits from segment ${segment.id}`);
        return allBits;
    }
}