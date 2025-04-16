/**
 * Carrier Matrix
 * 
 * Represents the text structure and potential embedding capacity across different
 * carrier techniques and text segments. Used for optimizing payload distribution.
 * 
 * Design Goals: Model text structure, calculate weighted capacity, support optimization queries.
 * Architectural Constraints: Relies on registered CarrierTechnique implementations. Structure analysis is currently basic.
 * Paradigm: Imperative data structure management, Functional analysis for capacity calculation.
 * Happy Path: Instantiate with text -> Register carriers -> Analyze capacity -> Provide matrix data for optimization.
 * ApiNotes: ./carrier_matrix.ApiNotes.md
 */

import type { 
    CarrierAnalysis, // Assuming this is defined elsewhere or needed for future use
    EncodingOptions, // Assuming this is defined elsewhere or needed for future use
} from './stylometric_carrier.genai.mts'; // Use .mts extension, import types
import { StylometricCarrier } from './stylometric_carrier.genai.mts'; // Import class
import { QuoteStyleCarrier } from './quote_style_carrier.mts'; // Correct path
import type { CarrierTechnique, FeatureMap } from './types/CarrierTypes.ts'; // Import type from central types file, ensure FeatureMap is imported here
import type { 
    CapacityAnalysisOptions, 
    OptimizationOptions, 
    TextSegment, 
    CapacityMatrix as ICapacityMatrix, // Use interface alias
    CarrierMetadata, 
    EmbeddingPlan, 
    SegmentEmbeddingPlan,
    DocumentStructure,
    TextStructureType, // Import TextStructureType
    TextStructure // Import TextStructure interface
} from './capacity_matrix/capacity_analyzer.mts'; // Import types from analyzer

// Define weights (consider moving to a config or ApiNotes reference)
const WEIGHT_CHAPTER = 1.0;
const WEIGHT_SECTION = 0.9;
const WEIGHT_METADATA = 1.2; // Higher weight for metadata? Maybe less detectable?
const WEIGHT_FORWARD = 0.8;
const WEIGHT_NOTES = 0.7;
// Reference: ./carrier_matrix.ApiNotes.md#WeightingFactors

// Define CapacityDataPoint locally as it's specific to this matrix's internal representation
interface CapacityDataPoint {
    segmentIndex: number;
    techniqueId: string;
    rawCapacity: number;
    weightedCapacity: number;
    segmentType: TextStructureType;
}


export class CarrierMatrix {
    private text: string;
    private structure: TextStructure[] = [];
    private carriers: CarrierTechnique[] = [];
    private matrix: CapacityDataPoint[] = []; // Use the defined interface

    /**
     * Implements the core logic described in carrier_matrix.ApiNotes.md.
     * @param text The input text.
     * @param registeredCarriers Optional array of pre-registered carriers.
     */
    constructor(text: string, registeredCarriers?: CarrierTechnique[]) {
        if (text == null) throw new Error('Input text cannot be null for CarrierMatrix initialization.');
        // Reference: ./carrier_matrix.ApiNotes.md#Initialization
        this.text = text;
        this.structure = this.extractStructure(text);

        // Register provided carriers or default set
        const initialCarriers: CarrierTechnique[] = [];
        if (registeredCarriers && Array.isArray(registeredCarriers) && registeredCarriers.length > 0) {
            initialCarriers.push(...registeredCarriers); // Use spread to create a new array
        } else {
            console.warn("No carriers provided or invalid format, using default set.");
            // Default registration - replace/extend as needed
            // Reference: ./carrier_matrix.ApiNotes.md#CarrierRegistration
            initialCarriers.push(new QuoteStyleCarrier());
            // Assuming StylometricCarrier provides access to its internal carriers
            try {
                const stylometricHandler = new StylometricCarrier();
                // Use the implemented method getAvailableCarriers
                const availableCarriers = stylometricHandler.getAvailableCarriers(); 
                if (Array.isArray(availableCarriers)) {
                    initialCarriers.push(...availableCarriers);
                } else {
                     console.error("StylometricCarrier.getAvailableCarriers() did not return an array.");
                }
            } catch (e) {
                 console.error(`Failed to instantiate or get carriers from StylometricCarrier: ${(e as Error).message}`);
            }
        }
        
        // Ensure no duplicate carriers by ID
        const carrierMap = new Map<string, CarrierTechnique>();
        initialCarriers.forEach(carrier => {
            if (!carrierMap.has(carrier.id)) {
                carrierMap.set(carrier.id, carrier);
            } else {
                console.warn(`Duplicate carrier ID found during initialization: ${carrier.id}. Keeping the first instance.`);
            }
        });
        this.carriers = Array.from(carrierMap.values());


        if (this.carriers.length === 0) {
             console.error('CarrierMatrix initialized with zero registered carriers. Capacity analysis will yield nothing.');
             // Consider throwing an error if no carriers is an invalid state:
             // throw new Error('CarrierMatrix must have at least one registered carrier.');
        }

        this.analyzeCapacity(); // Analyze capacity upon initialization
    }

    /**
     * Method to explicitly register carriers after construction if needed.
     * @param carrier The carrier technique to register.
     */
    registerCarrier(carrier: CarrierTechnique): void {
        // Reference: ./carrier_matrix.ApiNotes.md#CarrierRegistration
        if (!this.carriers.some(c => c.id === carrier.id)) {
            this.carriers.push(carrier);
            this.analyzeCapacity(); // Re-analyze capacity after adding a carrier
        } else {
            console.warn(`Carrier with ID ${carrier.id} is already registered.`);
        }
    }

    /**
     * Extracts the structure of the text into segments.
     * @param text The input text.
     * @returns An array of TextStructure segments.
     */
    private extractStructure(text: string): TextStructure[] {
        // Reference: carrier_matrix.ApiNotes.md#StructuralAnalysis
        // Basic implementation using Markdown headings. Enhance as needed.
        const lines = text.split('\n');
        const structure: TextStructure[] = [];
        let currentType: TextStructureType = 'section'; // Default type
        let currentContent = '';
        let currentStartLine = 0;

        // Simple detection for potential metadata/forward/notes (can be improved)
        const lowerText = text.toLowerCase();
        let mainContentStartLine = 0;
        let mainContentEndLine = lines.length - 1;

        // Example: Look for common metadata patterns (e.g., frontmatter)
        if (lines[0] === '---') {
            const endFrontmatter = lines.indexOf('---', 1);
            if (endFrontmatter !== -1) {
                structure.push({
                    type: 'metadata',
                    content: lines.slice(1, endFrontmatter).join('\n'),
                    startLine: 1,
                    endLine: endFrontmatter -1
                });
                mainContentStartLine = endFrontmatter + 1;
                currentStartLine = mainContentStartLine; // Adjust start for subsequent content
            }
        }

        // Example: Look for "Forward" or "Preface" near the beginning
        // This needs more robust boundary detection. Placeholder logic.
        const forwardIndex = lowerText.indexOf("forward\n");
        const prefaceIndex = lowerText.indexOf("preface\n");
        const introIndex = (forwardIndex !== -1 && prefaceIndex !== -1) ? Math.min(forwardIndex, prefaceIndex) : Math.max(forwardIndex, prefaceIndex);
        if (introIndex !== -1 && introIndex < text.length * 0.1) { // If found early
            const introStartLine = text.substring(0, introIndex).split('\n').length;
            // Find end boundary (e.g., first #, ##, or significant blank space)
            let introEndLine = mainContentEndLine;
            const firstChapterMark = text.indexOf('\n# ', introIndex);
            const firstSectionMark = text.indexOf('\n## ', introIndex);
            const firstMajorBreak = text.indexOf('\n\n\n', introIndex); // Example break
            let endMarkIndex = text.length;
            if (firstChapterMark !== -1) endMarkIndex = Math.min(endMarkIndex, firstChapterMark);
            if (firstSectionMark !== -1) endMarkIndex = Math.min(endMarkIndex, firstSectionMark);
            if (firstMajorBreak !== -1) endMarkIndex = Math.min(endMarkIndex, firstMajorBreak);

            if (endMarkIndex < text.length) {
                introEndLine = text.substring(0, endMarkIndex).split('\n').length - 1;
                if (introStartLine <= introEndLine && introStartLine >= mainContentStartLine) {
                    structure.push({
                        type: 'forward',
                        content: text.substring(introIndex, endMarkIndex).trim(),
                        startLine: introStartLine,
                        endLine: introEndLine
                    });
                    // Adjust main content boundaries if forward is detected after metadata
                    if (introStartLine >= mainContentStartLine) {
                         mainContentStartLine = introEndLine + 1;
                         currentStartLine = mainContentStartLine;
                    }
                }
            }
        }

        // Example: Look for "Notes" or "References" near the end
        const notesIndex = Math.max(lowerText.lastIndexOf("\nnotes\n"), lowerText.lastIndexOf("\nreferences\n"));
        if (notesIndex !== -1 && notesIndex > text.length * 0.8) { // If found late
            const notesStartLine = text.substring(0, notesIndex).split('\n').length;
             // Ensure notes section doesn't overlap with potential forward/preface added earlier
            const existingEndLine = structure.length > 0 ? structure[structure.length - 1].endLine : -1;
            if (notesStartLine > existingEndLine && notesStartLine <= mainContentEndLine) {
                 structure.push({
                     type: 'notes',
                     content: text.substring(notesIndex).trim(),
                     startLine: notesStartLine,
                     endLine: lines.length - 1
                 });
                 mainContentEndLine = notesStartLine - 1; // Adjust end line for main content
            }
        }


        // Process main content lines
        for (let index = mainContentStartLine; index <= mainContentEndLine; index++) {
            const line = lines[index];
            if (line === undefined) continue; // Should not happen, but safety check

            let newType: TextStructureType | null = null;
            if (line.startsWith('# ')) {
                newType = 'chapter';
            } else if (line.startsWith('## ')) {
                newType = 'section';
            }
            // Add more rules based on ApiNotes (e.g., detecting lists, paragraphs)

            if (newType && currentContent.trim()) {
                // End previous block if it wasn't just empty lines
                structure.push({ type: currentType, content: currentContent.trimEnd(), startLine: currentStartLine, endLine: index - 1 });
                currentContent = line + '\n'; // Start new content with the heading line
                currentStartLine = index;
                currentType = newType;
            } else if (newType) {
                 // Handle case where a heading follows another heading immediately or is first
                 if (currentContent.trim()) { // If there was content before (even just the previous heading)
                     structure.push({ type: currentType, content: currentContent.trimEnd(), startLine: currentStartLine, endLine: index - 1 });
                 }
                 currentContent = line + '\n'; // Start new content with the heading line
                 currentStartLine = index;
                 currentType = newType;
            } else {
                 // Add line to current content
                 currentContent += line + '\n';
            }
        }

        // Add the last block of the main content
        if (currentContent.trim()) {
            structure.push({ type: currentType, content: currentContent.trimEnd(), startLine: currentStartLine, endLine: mainContentEndLine });
        }

        // Ensure at least one segment exists if structure is still empty
        if (structure.length === 0 && text.length > 0) {
            console.warn("No structure detected, treating entire text as one section.");
            structure.push({ type: 'section', content: text, startLine: 0, endLine: lines.length - 1 });
        }

        // Add default metadata if none found and structure is not empty
        if (structure.length > 0 && !structure.some(s => s.type === 'metadata')) {
             structure.unshift({ type: 'metadata', content: '', startLine: 0, endLine: -1 }); // Empty metadata
        }


        return structure;
    }

    /**
     * Applies a weight multiplier based on the segment type.
     * @param type The type of the text structure segment.
     * @param baseWeight The initial weight (usually 1.0).
     * @returns The adjusted weight.
     */
    private applyWeight(type: TextStructureType, baseWeight: number = 1.0): number {
        // Reference: carrier_matrix.ApiNotes.md#WeightingLogic
        let multiplier = 1.0;
        switch (type) {
            case 'chapter': multiplier = WEIGHT_CHAPTER; break;
            case 'section': multiplier = WEIGHT_SECTION; break;
            case 'metadata': multiplier = WEIGHT_METADATA; break;
            case 'forward': multiplier = WEIGHT_FORWARD; break;
            case 'notes': multiplier = WEIGHT_NOTES; break;
            // default: keep multiplier 1.0
        }
        // assert multiplier >= 0 : 'Weight multiplier must be non-negative.';
        return baseWeight * multiplier;
    }

    /**
     * Analyzes the carrying capacity for each segment and carrier.
     * Populates the internal `matrix`.
     */
    analyzeCapacity(): void {
        // Reference: carrier_matrix.ApiNotes.md#CapacityCalculation
        this.matrix = []; // Clear previous results
        if (this.carriers.length === 0) {
             console.warn("No carriers registered, cannot analyze capacity.");
             return;
        }

        this.structure.forEach((segment, segmentIndex) => {
            // assert segment.content != null : `Segment ${segmentIndex} content is null.`;
            if (!segment.content && segment.type !== 'metadata') { // Allow empty metadata, skip others
                 // console.warn(`Skipping empty segment ${segmentIndex} of type ${segment.type}.`);
                 return;
            }

            const segmentWeight = this.applyWeight(segment.type); // Get weight based on segment type

            this.carriers.forEach(carrier => {
                let rawCapacity = 0;
                try {
                    // Estimate capacity for the specific segment content
                    // Handle potentially empty metadata content gracefully
                    rawCapacity = (segment.content) ? carrier.estimate(segment.content) : 0;
                    // assert rawCapacity >= 0 : `Carrier ${carrier.id} estimated negative capacity (${rawCapacity}) for segment ${segmentIndex}.`;
                    rawCapacity = Math.max(0, rawCapacity); // Ensure non-negative
                } catch (e) {
                    console.warn(`Error estimating capacity for carrier ${carrier.id} on segment ${segmentIndex} (type: ${segment.type}): ${(e as Error).message}`);
                    rawCapacity = 0; // Default to 0 on error
                }

                // Apply segment type weight and potentially carrier-specific weights (e.g., detectability)
                // Simple weighting: capacity * segmentWeight * (1 - detectability)
                // More complex weighting could involve naturalness, robustness etc. from CarrierTechnique
                const detectability = carrier.getDetectability ? carrier.getDetectability() : 0.5; // Default detectability if not provided
                const weightedCapacity = rawCapacity * segmentWeight * (1.0 - detectability);

                this.matrix.push({
                    segmentIndex,
                    techniqueId: carrier.id,
                    rawCapacity,
                    weightedCapacity: Math.max(0, weightedCapacity), // Ensure non-negative
                    segmentType: segment.type
                });
            });
        });
         // console.log("Capacity matrix calculated:", this.matrix); // Debug log
    }

    /**
     * Retrieves the calculated capacity matrix data.
     * @returns An array of CapacityDataPoint objects.
     */
    getCapacityMatrix(): CapacityDataPoint[] {
        // assert this.matrix != null : 'Capacity matrix has not been calculated yet.';
        return [...this.matrix]; // Return a copy
    }

    /**
     * Calculates the total raw capacity across all segments and carriers.
     * @returns Total capacity in bits.
     */
    getTotalRawCapacity(): number {
        return this.matrix.reduce((sum, point) => sum + point.rawCapacity, 0);
    }

    /**
     * Calculates the total weighted capacity across all segments and carriers.
     * @returns Total weighted capacity (unitless, for optimization).
     */
    getTotalWeightedCapacity(): number {
        return this.matrix.reduce((sum, point) => sum + point.weightedCapacity, 0);
    }

    /**
     * Generates an embedding plan based on the desired payload size and optimization options.
     * Placeholder: Requires an optimization algorithm (e.g., greedy, linear programming).
     * @param payloadSizeBits The desired payload size in bits.
     * @param options Optimization options.
     * @returns An EmbeddingPlan or null if payload doesn't fit.
     */
    generateEmbeddingPlan(payloadSizeBits: number, options: OptimizationOptions = {}): EmbeddingPlan | null {
        // Reference: carrier_matrix.ApiNotes.md#Optimization
        console.warn("CarrierMatrix.generateEmbeddingPlan uses a basic greedy algorithm. Needs refinement for complex optimization.");
        // assert payloadSizeBits >= 0 : 'Payload size must be non-negative.';
        if (payloadSizeBits === 0) {
            return { // Return an empty plan for zero payload
                 segmentPlans: [],
                 totalBits: 0,
                 utilization: 0,
                 techniqueDistribution: {},
                 expectedTextChanges: 0,
                 redundantEncoding: false
            };
        }

        // --- Placeholder Greedy Algorithm ---
        // 1. Sort matrix points by weightedCapacity descending (or other criteria based on options)
        // 2. Iterate through sorted points, allocating bits until payloadSizeBits is reached.

        const sortedPoints = [...this.matrix]
            .filter(p => p.rawCapacity > 0) // Only consider points with actual capacity
            .sort((a, b) => {
                // Prioritize based on options (example: minimize changes -> higher capacity first)
                if (options.minimizeChanges) {
                    // Maybe prioritize techniques with lower detectability first?
                    // Or techniques known to make fewer changes per bit? Needs more metadata.
                    // For now, stick to raw capacity as a proxy for fewer changes needed overall.
                    return b.rawCapacity - a.rawCapacity; // Higher raw capacity first
                }
                 if (options.prioritizeNaturalness) {
                     // Requires naturalness score on CarrierTechnique
                     // const natA = this.carriers.find(c => c.id === a.techniqueId)?.getNaturalness() || 0;
                     // const natB = this.carriers.find(c => c.id === b.techniqueId)?.getNaturalness() || 0;
                     // return natB - natA; // Higher naturalness first
                 }
                 if (options.prioritizeResilience) {
                     // Requires robustness score on CarrierTechnique
                     // const robA = this.carriers.find(c => c.id === a.techniqueId)?.getRobustness() || 0;
                     // const robB = this.carriers.find(c => c.id === b.techniqueId)?.getRobustness() || 0;
                     // return robB - robA; // Higher robustness first
                 }
                // Default: highest weighted capacity first (balances capacity, segment weight, detectability)
                return b.weightedCapacity - a.weightedCapacity;
            });

        let bitsToAllocate = payloadSizeBits;
        const segmentPlansMap: Map<number, SegmentEmbeddingPlan> = new Map();
        const techniqueDistribution: Record<string, number> = {};

        for (const point of sortedPoints) {
            if (bitsToAllocate <= 0) break;

            const bitsForThisPoint = Math.min(bitsToAllocate, point.rawCapacity);

            if (bitsForThisPoint > 0) {
                let segmentPlan = segmentPlansMap.get(point.segmentIndex);
                if (!segmentPlan) {
                    const segmentType = this.structure[point.segmentIndex]?.type || 'section'; // Use 'section' as default instead of 'unknown'
                    segmentPlan = { segmentIndex: point.segmentIndex, segmentType: segmentType, techniques: [], totalBits: 0 };
                    segmentPlansMap.set(point.segmentIndex, segmentPlan);
                }

                // Calculate bit offset within this technique for this segment
                const currentTechniqueBits = segmentPlan.techniques
                    .filter(t => t.techniqueId === point.techniqueId)
                    .reduce((sum, t) => sum + t.bitCount, 0);

                segmentPlan.techniques.push({
                    techniqueId: point.techniqueId,
                    bitCount: bitsForThisPoint,
                    bitOffset: currentTechniqueBits // Offset for applying bits correctly
                });
                segmentPlan.totalBits += bitsForThisPoint;

                techniqueDistribution[point.techniqueId] = (techniqueDistribution[point.techniqueId] || 0) + bitsForThisPoint;
                bitsToAllocate -= bitsForThisPoint;
            }
        }

        if (bitsToAllocate > 0) {
            const allocatedBits = payloadSizeBits - bitsToAllocate;
            console.warn(`Could not allocate all requested bits (${payloadSizeBits}). Only ${allocatedBits} bits could be planned. ${bitsToAllocate} bits remaining.`);
            // Depending on requirements, either return null or a partial plan.
            // For now, return null if full payload doesn't fit.
             return null;
        }

        const totalBitsEncoded = payloadSizeBits; // Since we allocated exactly this amount
        const totalRawCapacity = this.getTotalRawCapacity();

        const plan: EmbeddingPlan = {
            segmentPlans: Array.from(segmentPlansMap.values()),
            totalBits: totalBitsEncoded,
            utilization: totalRawCapacity > 0 ? totalBitsEncoded / totalRawCapacity : 0,
            techniqueDistribution,
            expectedTextChanges: totalBitsEncoded, // Simplistic estimate: 1 change per bit
            redundantEncoding: (options.redundancyFactor || 1.0) > 1.0,
            // Add redundancy details if implemented
        };

        // assert plan.totalBits === payloadSizeBits : 'Generated plan does not match requested payload size.';
        return plan;
    }

    /**
     * Gets the full text used to initialize the matrix.
     * @returns The original text string.
     */
    getFullText(): string {
        return this.text;
    }

    /**
     * Gets the extracted text structure.
     * @returns An array of TextStructure segments.
     */
    getTextStructure(): TextStructure[] {
        return [...this.structure]; // Return a copy
    }
}

// Add ApiNotes.md for this file
// ```markdown
// // filepath: /home/files/git/Stylometrics/carrier_matrix.ApiNotes.md
// # Carrier Matrix - ApiNotes
//
// ## Design
// The `CarrierMatrix` class models the steganographic capacity of a text document. It segments the text based on structure (chapters, sections, metadata, etc.) and evaluates the capacity of various registered `CarrierTechnique` implementations within each segment. It applies weighting factors based on segment type and carrier properties (like detectability) to create a matrix of weighted capacities.
//
// ## Behavior
// 1.  **Initialization**: Takes text and optional carriers. Extracts text structure (`extractStructure`) and registers default or provided carriers (handling duplicates). Calls `analyzeCapacity`.
// 2.  **`extractStructure`**: Divides text into `TextStructure` segments (e.g., based on Markdown headings, frontmatter, potential forward/notes sections). This implementation is improved but can be further enhanced for complex documents.
// 3.  **`registerCarrier`**: Adds a new carrier technique if the ID is unique and re-analyzes capacity.
// 4.  **`applyWeight`**: Calculates a weight multiplier based on the `TextStructureType`.
// 5.  **`analyzeCapacity`**: Iterates through segments and registered carriers. For each pair, it calls the carrier's `estimate` method, applies weighting (`applyWeight` and carrier detectability), and stores the raw and weighted capacity in the internal `matrix` (an array of `CapacityDataPoint`). Handles empty segments and estimation errors.
// 6.  **`getCapacityMatrix`**: Returns a copy of the calculated matrix data.
// 7.  **`getTotalRawCapacity` / `getTotalWeightedCapacity`**: Sums capacities from the matrix.
// 8.  **`generateEmbeddingPlan`**: Takes desired payload size and optimization options. Uses the matrix to determine how to distribute bits across segments and techniques. Currently uses a basic greedy approach sorting by weighted capacity (or raw capacity if `minimizeChanges` is true). Returns `null` if the full payload cannot be allocated. Returns an empty plan for a zero payload. Needs enhancement for more sophisticated optimization (e.g., linear programming, considering naturalness/resilience).
//
// ## Weighting Factors (`#WeightingFactors`)
// Constants define multipliers for different text structure types:
// - `WEIGHT_CHAPTER`: 1.0
// - `WEIGHT_SECTION`: 0.9
// - `WEIGHT_METADATA`: 1.2 (Hypothesis: less scrutinized?)
// - `WEIGHT_FORWARD`: 0.8
// - `WEIGHT_NOTES`: 0.7 (Hypothesis: less critical content?)
// These weights influence the `weightedCapacity` used in optimization. The detectability of the carrier also negatively impacts the weighted capacity (`weighted = raw * segmentWeight * (1 - detectability)`).
//
// ## Optimization (`#Optimization`)
// The `generateEmbeddingPlan` method aims to select the best `CapacityDataPoint` entries from the matrix to fulfill the payload requirement, considering `OptimizationOptions`. The current greedy approach sorts points and fills capacity. Future work could involve linear programming or other solvers for constraints like `preserveReadingLevel`, `balanceChanges`, `prioritizeNaturalness`, or `prioritizeResilience`.
//
// ## Constraints
// - Relies heavily on the accuracy and consistency of `CarrierTechnique.estimate()` and `getDetectability()` methods.
// - `extractStructure` implementation is rule-based; complex or non-standard documents might need more sophisticated parsing (e.g., NLP-based).
// - Optimization logic is currently basic (greedy).
// ```