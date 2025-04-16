/**
 * Capacity Matrix Analyzer
 * 
 * This module implements advanced carrying capacity analysis using a weighted matrix approach.
 * It analyzes text across multiple dimensions and carrier techniques to optimize steganographic
 * embedding while preserving text naturalness and minimizing detection risk.
 * 
 * Flow:
 * Text → Segmentation → Technique Evaluation → Matrix Construction → Optimization → Embedding Plan
 * 
 * The happy path involves analyzing text structure, evaluating carrier techniques for each segment,
 * constructing a weighted capacity matrix, and solving the optimization problem to distribute
 * payload bits optimally across available carriers.
 */

import { 
    StyleFeatureExtractor, 
    
    type FeatureMap 
} from '../stylometric_detection.genai.mjs';
import { 
    StylometricCarrier,
} from '../stylometric_carrier.genai.mjs'; // Removed Carrier import
import { QuoteStyleCarrier } from './quote_style_carrier.mts'; // Add back the .mts extension

/**
 * Interface defining a carrier for steganographic embedding
 */
export interface Carrier {
    id: string;                           // Unique identifier for the carrier
    analyzeCapacity(text: string): number; // Method to analyze carrying capacity in bits
}

// Define weights as constants or load from config/ApiNotes
const WEIGHT_CHAPTER = 1.0;
const WEIGHT_SECTION = 0.9;
const WEIGHT_METADATA = 1.2;
const WEIGHT_FORWARD = 0.8;
const WEIGHT_NOTES = 0.7;
// Reference: carrier_matrix.ApiNotes.md#WeightingFactors

// ApiNotes: ./ApiNotes.md (Assumed for directory)

/**
 * Interface for capacity matrix options
 */
export interface CapacityAnalysisOptions {
    // Segmentation options
    segmentationStrategy?: 'natural' | 'equal' | 'feature-based' | 'hierarchical';
    minSegmentSize?: number;  // Minimum segment size in words
    maxSegmentCount?: number; // Maximum number of segments
    
    // Weighting options
    prioritizeNaturalness?: boolean;   // Prioritize techniques that preserve natural text
    prioritizeCapacity?: boolean;      // Prioritize techniques with higher capacity
    prioritizeResilience?: boolean;    // Prioritize techniques with higher resilience
    detectabilityThreshold?: number;   // Maximum allowed detectability (0-1)
    
    // Analysis detail level
    includeFeatureDetails?: boolean;   // Include detailed feature analysis in result
    includeSegmentDetails?: boolean;   // Include detailed segment analysis in result
}

/**
 * Interface for optimization options
 */
export interface OptimizationOptions {
    // Distribution goals
    minimizeChanges?: boolean;       // Minimize the total number of text changes
    balanceChanges?: boolean;        // Distribute changes evenly across the text
    preferFrontLoading?: boolean;    // Prefer embedding more data at the beginning
    preferBackLoading?: boolean;     // Prefer embedding more data at the end
    prioritizeNaturalness?: boolean; // Added based on usage in carrier_matrix.mts
    prioritizeResilience?: boolean;  // Added based on usage in carrier_matrix.mts

    // Constraints
    preserveReadingLevel?: boolean;  // Maintain original reading level
    preserveKeyPhrases?: string[];   // List of phrases to leave unchanged
    preserveSegments?: number[];     // List of segment indices to leave unchanged
    
    // Redundancy
    redundancyFactor?: number;       // Factor for redundant encoding (1.0 = no redundancy)
    redundancyStrategy?: 'parity' | 'replication' | 'erasure'; // Strategy for redundancy
}

/**
 * Interface representing a text segment for capacity analysis
 */
export interface TextSegment {
    text: string;               // The segment text
    index: number;              // Position in the original text
    startChar: number;          // Starting character position
    endChar: number;            // Ending character position
    wordCount: number;          // Number of words
    features: FeatureMap;       // Stylometric features of the segment
    boundaries: {               // Boundary type information
        startsNewParagraph: boolean;
        startsNewSection: boolean;
        endsWithPunctuation: boolean;
        containsDialog: boolean;
    };
}

/**
 * Interface for a capacity matrix
 */
export interface CapacityMatrix {
    segments: TextSegment[];                         // Text segments
    techniques: string[];                            // Carrier technique IDs
    capacities: number[][];                          // Raw capacity values [segment][technique]
    weights: number[];                               // Technique weights
    weightedCapacities: number[][];                  // Weighted capacity values [segment][technique]
    totalCapacity: number;                           // Total bits capacity
    recommendedCapacity: number;                     // Recommended usable capacity
    techniqueMetadata: Record<string, CarrierMetadata>; // Details about each technique
}

/**
 * Interface for carrier technique metadata
 */
export interface CarrierMetadata {
    id: string;
    name: string;
    category: string;
    detectability: number;
    bitsPerThousandWords: number;
    resilience: number; // 0-1 scale of how resilient the technique is to text changes
}

/**
 * Interface for embedding plan
 */
export interface EmbeddingPlan {
    segmentPlans: SegmentEmbeddingPlan[];
    totalBits: number;
    utilization: number; // 0-1 ratio of used capacity
    techniqueDistribution: Record<string, number>; // Technique ID to bit count mapping
    expectedTextChanges: number; // Estimated number of text modifications
    redundantEncoding: boolean;
    redundancyDetails?: {
        originalSize: number;
        encodedSize: number;
        requiredFragments: number;
        totalFragments: number;
    };
}

/**
 * Interface for segment-level embedding plan
 */
export interface SegmentEmbeddingPlan {
    segmentIndex: number;
    segmentType: TextStructureType; // Added based on usage in carrier_matrix.mts
    techniques: Array<{
        techniqueId: string;
        bitCount: number;
        bitOffset: number;
    }>;
    totalBits: number;
}

/**
 * Interface for document structure analysis
 */
export interface DocumentStructure {
    sections: Array<{
        title?: string;
        level: number;
        startChar: number;
        endChar: number;
        paragraphs: number[];
    }>;
    paragraphs: Array<{
        startChar: number;
        endChar: number;
        sentences: number[];
        isList?: boolean;
        isQuote?: boolean;
    }>;
    sentences: Array<{
        startChar: number;
        endChar: number;
        text: string;
    }>;
    metadata: {
        title?: string;
        author?: string;
        date?: string;
        genre?: string;
        wordCount: number;
        sentenceCount: number;
        paragraphCount: number;
        sectionCount: number;
    };
}

/**
 * Type definition for text structure types
 */
export type TextStructureType = 'chapter' | 'section' | 'metadata' | 'forward' | 'notes';

/**
 * Interface for a text structure segment
 */
export interface TextStructure {
    type: TextStructureType;
    content: string;
    startLine: number;
    endLine: number;
}

/**
 * Interface for capacity data point used in the carrier matrix
 */
interface CapacityDataPoint {
    segmentIndex: number;
    techniqueId: string;
    rawCapacity: number;
    weightedCapacity: number;
    segmentType: TextStructureType;
}

/**
 * Analyzes the carrying capacity of a given text using available carriers.
 * Design Goals: Provide a simple interface to get capacity estimates for a text.
 * Architectural Constraints: Uses StylometricCarrier and CarrierMatrix.
 * Happy Path: Text input -> Instantiate Carrier/Matrix -> Analyze -> Return capacity map.
 * ApiNotes: ./capacity_analyzer.ApiNotes.md
 * [paradigm:imperative]
 * @param text The input text string.
 * @returns A promise resolving to a record mapping carrier IDs to their estimated capacity, or null on error.
 */
export async function analyzeTextCapacity(text: string): Promise<Record<string, number> | null> {
    // assert typeof text === 'string' : 'Input text must be a string.';
    // Reference: ./capacity_analyzer.ApiNotes.md#AnalysisLogic
    try {
        // Assuming StylometricCarrier constructor doesn't throw for basic setup
        const carrierInstance = new StylometricCarrier();
        
        // Try to get carriers using the method that actually returns an array
        // The constructor can handle undefined, so we'll use that as fallback
        let allCarriers;
        try {
            // Try getAvailableCarriers first, as mentioned in the constructor comments
            if (typeof carrierInstance.getAvailableCarriers === 'function') {
                allCarriers = carrierInstance.getAvailableCarriers();
            } 
            // Fall back to getCarriers if it returns an array
            else if (typeof carrierInstance.getCarriers === 'function') {
                const carriers = carrierInstance.getCarriers();
                // Only use the result if it's an array
                if (Array.isArray(carriers)) {
                    allCarriers = carriers;
                }
            }
        } catch (error) {
            console.warn(`Failed to get carriers: ${error.message}`);
            // Continue with undefined carriers, letting constructor use defaults
        }
        
        // Use the locally defined CarrierMatrix class with the correct constructor signature
        // If allCarriers is undefined or not an array, the constructor will use defaults
        const matrix = new CarrierMatrix(text, allCarriers);

        // Get capacity from our local implementation
        const capacity = matrix.analyzeCapacity(text);

        // assert typeof capacity === 'object' && capacity !== null : 'Capacity analysis should return an object.';
        return capacity;
    } catch (error) {
        console.error(`[CapacityAnalyzer] Failed to analyze text capacity: ${error.message}`);
        // Reference: ./capacity_analyzer.ApiNotes.md#ErrorHandling
        return null; // Indicate failure to the caller
    }
}

/**
 * CarrierMatrix class for capacity analysis
 */
export class CarrierMatrix {
    private carriers: Carrier[] = [];
    private structure: TextStructure[] = [];
    private matrix: CapacityDataPoint[] = [];

    // Implements the core logic described in carrier_matrix.ApiNotes.md.
    constructor(text: string, registeredCarriers?: Carrier[]) {
        //assert(text != null, 'Input text cannot be null for CarrierMatrix initialization.');
        // Reference: carrier_matrix.ApiNotes.md#Initialization
        this.structure = this.extractStructure(text);

        // Register provided carriers or default set
        if (registeredCarriers && registeredCarriers.length > 0) {
            this.carriers = registeredCarriers;
        } else {
            // Default registration - replace/extend as needed
            // Reference: carrier_matrix.ApiNotes.md#CarrierRegistration
            this.carriers.push(new QuoteStyleCarrier());
            // Assuming StylometricCarrier provides access to its internal carriers or is iterable
            const stylometricHandler = new StylometricCarrier();
            
            // Create adapters for any CarrierTechnique objects to match the Carrier interface
            const techniques = stylometricHandler.getAvailableCarriers();
            const adaptedCarriers = techniques.map(technique => ({
                id: technique.id, // Assuming CarrierTechnique has an id property
                analyzeCapacity: (text: string): number => {
                    // Map to an equivalent method if available, or provide implementation
                    if (typeof technique.estimateCapacity === 'function') {
                        return technique.estimateCapacity(text);
                    } else if (typeof technique.analyze === 'function') {
                        return technique.analyze(text).capacity || 0;
                    }
                    // Fallback capacity (could log a warning here)
                    return 0;
                }
            } as Carrier));
            
            this.carriers.push(...adaptedCarriers);
        }
        //assert(this.carriers.length > 0, 'CarrierMatrix must have at least one registered carrier.');

        this.analyzeCapacity(text);
    }

    // Method to explicitly register carriers after construction if needed
    registerCarrier(carrier: Carrier): void {
        // Reference: carrier_matrix.ApiNotes.md#CarrierRegistration
        if (!this.carriers.some(c => c.id === carrier.id)) {
            this.carriers.push(carrier);
            // Optionally re-analyze capacity: this.analyzeCapacity(this.getFullText());
        }
    }

    // Add other carriers here as they are implemented and implement the Carrier interface
    // Example: this.carriers.push(new SynonymChoiceCarrier());
    //          this.carriers.push(new PunctuationStyleCarrier());
    // Ensure carriers are registered before analyzeCapacity is called if not done in constructor.

    private extractStructure(text: string): TextStructure[] {
        // Reference: carrier_matrix.ApiNotes.md#StructuralAnalysis
        // Current implementation is basic (Markdown headings).
        // TODO: Enhance this based on ApiNotes if more complex structure is needed (e.g., paragraphs, lists)
        const lines = text.split('\n');
        const structure: TextStructure[] = [];
        let currentType: TextStructureType = 'section'; // Default type
        let currentContent = '';
        let currentStartLine = 0;

        // Basic metadata/forward/notes detection (example)
        if (text.toLowerCase().includes("forward") || text.toLowerCase().includes("preface")) {
            structure.push({ type: 'forward', content: text.substring(0, Math.min(text.length, 500)), startLine: 0, endLine: text.substring(0, Math.min(text.length, 500)).split('\n').length - 1 });
            // Adjust start line/content for the rest if needed
        }
        if (text.toLowerCase().includes("notes") || text.toLowerCase().includes("references")) {
            const notesIndex = Math.max(text.toLowerCase().lastIndexOf("notes"), text.toLowerCase().lastIndexOf("references"));
            const notesContent = text.substring(notesIndex);
            structure.push({ type: 'notes', content: notesContent, startLine: text.substring(0, notesIndex).split('\n').length, endLine: lines.length - 1 });
            // Adjust end line/content for the main part if needed
        }

        lines.forEach((line, index) => {
            let newType: TextStructureType | null = null;
            if (line.startsWith('# ')) {
                newType = 'chapter';
            } else if (line.startsWith('## ')) {
                newType = 'section';
            }
            // Add more rules based on ApiNotes (e.g., detecting metadata blocks)

            if (newType && currentContent.trim()) {
                // End previous block
                structure.push({ type: currentType, content: currentContent.trimEnd(), startLine: currentStartLine, endLine: index - 1 });
                currentContent = ''; // Reset content
                currentStartLine = index;
                currentType = newType;
            }
            // Skip adding the heading line itself to the content, but include other lines
            if (!newType) {
                currentContent += line + '\n';
            } else if (!currentContent) {
                // Handle case where a heading is the very first line
                currentStartLine = index;
                currentType = newType;
            }
        });

        // Add the last block
        if (currentContent.trim()) {
            structure.push({ type: currentType, content: currentContent.trimEnd(), startLine: currentStartLine, endLine: lines.length - 1 });
        }

        // Add a default 'metadata' block if none detected (placeholder)
        if (!structure.some(s => s.type === 'metadata')) {
            structure.unshift({ type: 'metadata', content: '', startLine: 0, endLine: 0 });
        }

        return structure.length > 0 ? structure : [{ type: 'section', content: text, startLine: 0, endLine: lines.length - 1 }]; // Fallback
    }

    private applyWeight(type: TextStructureType, weight: number): number {
        // Reference: carrier_matrix.ApiNotes.md#WeightingLogic
        switch (type) {
            case 'chapter': weight *= WEIGHT_CHAPTER; break;
            case 'section': weight *= WEIGHT_SECTION; break;
            case 'metadata': weight *= WEIGHT_METADATA; break;
            case 'forward': weight *= WEIGHT_FORWARD; break;
            case 'notes': weight *= WEIGHT_NOTES; break;
            default: break; // No change for unknown types
        }
        return weight;
    }

    /**
     * Analyzes the carrying capacity of the text for all registered carriers
     * @param text The text to analyze
     * @returns An object mapping carrier IDs to their capacity in bits
     */
    analyzeCapacity(text: string): Record<string, number> {
        const result: Record<string, number> = {};
        
        // Iterate through registered carriers and compute capacity
        for (const carrier of this.carriers) {
            try {
                // Use the carrier's analyzeCapacity method to get raw capacity
                const capacity = carrier.analyzeCapacity(text);
                result[carrier.id] = capacity;
            } catch (error) {
                console.warn(`Failed to analyze capacity for carrier ${carrier.id}: ${error.message}`);
                result[carrier.id] = 0; // Default to zero capacity on error
            }
        }
        
        return result;
    }
}