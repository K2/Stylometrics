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
    FeatureMap 
} from '../stylometric_detection.genai.mjs';
import { 
    StylometricCarrier,
    CarrierTechnique
} from '../stylometric_carrier.genai.mjs';

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
 * The CapacityMatrixAnalyzer class provides advanced analysis of steganographic carrying capacity
 * across different text segments and carrier techniques, enabling optimized embedding strategies.
 */
export class CapacityMatrixAnalyzer {
    private featureExtractor: StyleFeatureExtractor;
    private carrier: StylometricCarrier;
    
    /**
     * Initialize the capacity matrix analyzer
     */
    constructor() {
        this.featureExtractor = new StyleFeatureExtractor();
        this.carrier = new StylometricCarrier();
    }
    
    /**
     * Analyze text to create a weighted capacity matrix
     * 
     * @param text Text to analyze
     * @param options Analysis options
     * @returns Capacity matrix
     */
    analyzeCapacityMatrix(text: string, options: CapacityAnalysisOptions = {}): CapacityMatrix {
        // Set default options
        const opts: Required<CapacityAnalysisOptions> = {
            segmentationStrategy: options.segmentationStrategy || 'natural',
            minSegmentSize: options.minSegmentSize || 50,
            maxSegmentCount: options.maxSegmentCount || 100,
            prioritizeNaturalness: options.prioritizeNaturalness ?? true,
            prioritizeCapacity: options.prioritizeCapacity ?? false,
            prioritizeResilience: options.prioritizeResilience ?? false,
            detectabilityThreshold: options.detectabilityThreshold ?? 0.5,
            includeFeatureDetails: options.includeFeatureDetails ?? false,
            includeSegmentDetails: options.includeSegmentDetails ?? true
        };
        
        // Segment the text
        const segments = this.segmentText(text, opts);
        
        // Get available carrier techniques
        const techniques = this.getCarrierTechniques(opts.detectabilityThreshold);
        
        // Calculate capacity for each segment and technique
        const capacities: number[][] = [];
        for (const segment of segments) {
            const segmentCapacities: number[] = [];
            
            for (const technique of techniques) {
                // Calculate raw capacity for this segment and technique
                const capacity = this.calculateCapacity(segment, technique);
                segmentCapacities.push(capacity);
            }
            
            capacities.push(segmentCapacities);
        }
        
        // Calculate weights for techniques
        const weights = this.calculateTechniqueWeights(techniques, opts);
        
        // Calculate weighted capacities
        const weightedCapacities: number[][] = [];
        for (let i = 0; i < capacities.length; i++) {
            const segmentWeightedCapacities: number[] = [];
            
            for (let j = 0; j < capacities[i].length; j++) {
                segmentWeightedCapacities.push(capacities[i][j] * weights[j]);
            }
            
            weightedCapacities.push(segmentWeightedCapacities);
        }
        
        // Calculate total capacity
        let totalCapacity = 0;
        for (const segmentCapacities of weightedCapacities) {
            totalCapacity += segmentCapacities.reduce((sum, val) => sum + val, 0);
        }
        
        // Create technique metadata
        const techniqueMetadata: Record<string, CarrierMetadata> = {};
        for (let i = 0; i < techniques.length; i++) {
            techniqueMetadata[techniques[i].id] = {
                id: techniques[i].id,
                name: techniques[i].name,
                category: techniques[i].category,
                detectability: techniques[i].detectability,
                bitsPerThousandWords: techniques[i].bitsPerThousandWords,
                resilience: 1 - techniques[i].detectability, // Inverse of detectability as approximation
            };
        }
        
        // Calculate recommended capacity (80% of total as a conservative estimate)
        const recommendedCapacity = Math.floor(totalCapacity * 0.8);
        
        return {
            segments,
            techniques: techniques.map(t => t.id),
            capacities,
            weights,
            weightedCapacities,
            totalCapacity,
            recommendedCapacity,
            techniqueMetadata
        };
    }
    
    /**
     * Create an optimal embedding plan for a payload based on capacity matrix
     * 
     * @param matrix Capacity matrix
     * @param payload Payload to embed
     * @param options Optimization options
     * @returns Embedding plan
     */
    optimizeEmbedding(
        matrix: CapacityMatrix,
        payload: Uint8Array,
        options: OptimizationOptions = {}
    ): EmbeddingPlan {
        // Calculate required bits (including redundancy)
        const payloadBits = payload.length * 8;
        const redundancyFactor = options.redundancyFactor ?? 1.0;
        const totalRequiredBits = Math.ceil(payloadBits * redundancyFactor);
        
        // Check if the matrix has enough capacity
        if (totalRequiredBits > matrix.totalCapacity) {
            throw new Error(
                `Payload too large: Requires ${totalRequiredBits} bits, ` +
                `but capacity is ${matrix.totalCapacity} bits`
            );
        }
        
        // Create distribution based on optimization strategy
        let distribution: number[][];
        
        if (options.minimizeChanges) {
            distribution = this.createMinimalChangeDistribution(
                matrix, totalRequiredBits, options
            );
        } else if (options.balanceChanges) {
            distribution = this.createBalancedDistribution(
                matrix, totalRequiredBits, options
            );
        } else if (options.preferFrontLoading) {
            distribution = this.createFrontLoadedDistribution(
                matrix, totalRequiredBits, options
            );
        } else if (options.preferBackLoading) {
            distribution = this.createBackLoadedDistribution(
                matrix, totalRequiredBits, options
            );
        } else {
            // Default: create optimal distribution
            distribution = this.createOptimalDistribution(
                matrix, totalRequiredBits, options
            );
        }
        
        // Calculate technique distribution
        const techniqueDistribution: Record<string, number> = {};
        for (let j = 0; j < matrix.techniques.length; j++) {
            let techTotal = 0;
            for (let i = 0; i < matrix.segments.length; i++) {
                techTotal += distribution[i][j];
            }
            techniqueDistribution[matrix.techniques[j]] = techTotal;
        }
        
        // Create segment-level plans
        const segmentPlans: SegmentEmbeddingPlan[] = [];
        let bitOffset = 0;
        let totalExpectedChanges = 0;
        
        for (let i = 0; i < matrix.segments.length; i++) {
            const techniques: Array<{
                techniqueId: string;
                bitCount: number;
                bitOffset: number;
            }> = [];
            
            let segmentBits = 0;
            
            for (let j = 0; j < matrix.techniques.length; j++) {
                const bitCount = distribution[i][j];
                if (bitCount > 0) {
                    techniques.push({
                        techniqueId: matrix.techniques[j],
                        bitCount,
                        bitOffset: bitOffset
                    });
                    
                    bitOffset += bitCount;
                    segmentBits += bitCount;
                    
                    // Estimate text changes based on technique (rough estimate: 1 change per 2-3 bits)
                    const technique = matrix.techniqueMetadata[matrix.techniques[j]];
                    totalExpectedChanges += Math.ceil(bitCount / 
                                                     (2 + technique.resilience * 2));
                }
            }
            
            if (segmentBits > 0) {
                segmentPlans.push({
                    segmentIndex: i,
                    techniques,
                    totalBits: segmentBits
                });
            }
        }
        
        // Calculate redundancy details if applicable
        const redundancyDetails = redundancyFactor > 1.0 ? {
            originalSize: payloadBits,
            encodedSize: totalRequiredBits,
            requiredFragments: Math.ceil(payloadBits / (totalRequiredBits / 
                                       (options.redundancyStrategy === 'erasure' ? 3 : 2))),
            totalFragments: options.redundancyStrategy === 'erasure' ? 
                          Math.ceil(matrix.segments.length / 3) : 
                          segmentPlans.length
        } : undefined;
        
        return {
            segmentPlans,
            totalBits: totalRequiredBits,
            utilization: totalRequiredBits / matrix.totalCapacity,
            techniqueDistribution,
            expectedTextChanges: totalExpectedChanges,
            redundantEncoding: redundancyFactor > 1.0,
            redundancyDetails
        };
    }
    
    /**
     * Analyze document structure to identify embedding opportunities
     * 
     * @param text Document text
     * @returns Document structure analysis
     */
    analyzeDocumentStructure(text: string): DocumentStructure {
        // This is a simplified implementation of document structure analysis
        const sections: DocumentStructure['sections'] = [];
        const paragraphs: DocumentStructure['paragraphs'] = [];
        const sentences: DocumentStructure['sentences'] = [];
        
        // Basic detection of paragraphs (separated by double newlines)
        const paraRegex = /(.+?)(\n\s*\n|$)/gs;
        let paraMatch;
        let paraIndex = 0;
        let paraStart = 0;
        
        while ((paraMatch = paraRegex.exec(text)) !== null) {
            const paraText = paraMatch[1];
            const paraEnd = paraStart + paraText.length;
            
            // Detect if this paragraph is a heading (simple heuristic)
            const isHeading = paraText.length < 100 && 
                            !paraText.includes('.') && 
                            paraText.trim().match(/^[A-Z0-9]/) !== null;
            
            // Split paragraph into sentences
            const sentenceRegex = /(.+?[.!?]+)(?:\s|$)/g;
            let sentMatch;
            let sentStart = paraStart;
            const sentIndices: number[] = [];
            
            while (sentMatch = sentenceRegex.exec(paraText)) {
                const sentText = sentMatch[1];
                const sentEnd = sentStart + sentText.length;
                
                sentences.push({
                    startChar: sentStart,
                    endChar: sentEnd,
                    text: sentText
                });
                
                sentIndices.push(sentences.length - 1);
                sentStart = sentEnd + 1;
            }
            
            // Determine if paragraph is a quote block (simplified heuristic)
            const isQuote = paraText.startsWith('>') || 
                          paraText.includes('\n>') || 
                          (paraText.startsWith('"') && paraText.endsWith('"'));
            
            paragraphs.push({
                startChar: paraStart,
                endChar: paraEnd,
                sentences: sentIndices,
                isQuote: isQuote,
                isList: paraText.match(/^\s*[-*•]\s/) !== null
            });
            
            // If this is a heading, start a new section
            if (isHeading) {
                sections.push({
                    title: paraText.trim(),
                    level: paraText.startsWith('#') ? 
                          (paraText.match(/^#+/) || ['#'])[0].length : 
                          (paraText.length < 30 ? 1 : 2),
                    startChar: paraStart,
                    endChar: text.length, // Will be updated when we find the next section
                    paragraphs: []
                });
                
                // Update the end position of the previous section if it exists
                if (sections.length > 1) {
                    sections[sections.length - 2].endChar = paraStart;
                }
            }
            
            // Add paragraph to current section
            if (sections.length > 0) {
                sections[sections.length - 1].paragraphs.push(paraIndex);
            }
            
            paraStart = paraEnd + paraMatch[2].length;
            paraIndex++;
        }
        
        // Set end of last section
        if (sections.length > 0) {
            sections[sections.length - 1].endChar = text.length;
        }
        
        // If no sections were detected, create a default one
        if (sections.length === 0) {
            sections.push({
                level: 0,
                startChar: 0,
                endChar: text.length,
                paragraphs: Array.from({ length: paragraphs.length }, (_, i) => i)
            });
        }
        
        return {
            sections,
            paragraphs,
            sentences,
            metadata: {
                wordCount: text.split(/\s+/).length,
                sentenceCount: sentences.length,
                paragraphCount: paragraphs.length,
                sectionCount: sections.length
            }
        };
    }
    
    /**
     * Identify embedding opportunities in document structure
     * 
     * @param structure Document structure analysis
     * @returns Map of embedding opportunities
     */
    identifyEmbeddingOpportunities(structure: DocumentStructure): Record<string, number> {
        const opportunities: Record<string, number> = {};
        
        // Calculate opportunities based on document structure
        
        // 1. Quotes offer good opportunities for style-based embedding
        let quoteBits = 0;
        for (const para of structure.paragraphs) {
            if (para.isQuote) {
                // Estimate capacity based on quote length
                const quoteLength = para.endChar - para.startChar;
                quoteBits += Math.floor(quoteLength / 20); // ~1 bit per 20 chars in quotes
            }
        }
        opportunities['quote_style'] = quoteBits;
        
        // 2. Lists are good for punctuation and formatting variations
        let listBits = 0;
        for (const para of structure.paragraphs) {
            if (para.isList) {
                // List items can encode information in formatting, punctuation, etc.
                listBits += para.sentences.length * 2; // ~2 bits per list item
            }
        }
        opportunities['list_format'] = listBits;
        
        // 3. Section headings offer metadata opportunities
        opportunities['heading_style'] = structure.sections.length * 4; // ~4 bits per heading
        
        // 4. Paragraph structure
        opportunities['paragraph_structure'] = 
            Math.floor(structure.paragraphs.length / 3) * 5; // ~5 bits per 3 paragraphs
        
        // 5. Sentence-level opportunities (across all text)
        opportunities['sentence_structure'] = 
            Math.floor(structure.sentences.length / 5) * 8; // ~8 bits per 5 sentences
            
        // 6. Document-level metadata
        opportunities['document_metadata'] = 40; // Fixed capacity for metadata
        
        return opportunities;
    }
    
    /**
     * Segment text for capacity analysis
     * 
     * @param text Text to segment
     * @param options Segmentation options
     * @returns Array of text segments
     */
    private segmentText(text: string, options: Required<CapacityAnalysisOptions>): TextSegment[] {
        const segments: TextSegment[] = [];
        
        if (options.segmentationStrategy === 'natural') {
            // Natural segmentation by paragraphs
            const structure = this.analyzeDocumentStructure(text);
            
            for (let i = 0; i < structure.paragraphs.length; i++) {
                const para = structure.paragraphs[i];
                const paraText = text.substring(para.startChar, para.endChar);
                
                // Skip very short paragraphs
                if (paraText.split(/\s+/).length < options.minSegmentSize && i < structure.paragraphs.length - 1) {
                    continue;
                }
                
                // Detect boundaries
                const startsNewSection = structure.sections.some(
                    sec => sec.paragraphs[0] === i
                );
                
                // Extract features
                const features = this.featureExtractor.extractAllFeatures(paraText);
                
                segments.push({
                    text: paraText,
                    index: segments.length,
                    startChar: para.startChar,
                    endChar: para.endChar,
                    wordCount: paraText.split(/\s+/).length,
                    features,
                    boundaries: {
                        startsNewParagraph: true,
                        startsNewSection,
                        endsWithPunctuation: /[.!?]$/.test(paraText.trim()),
                        containsDialog: paraText.includes('"') || paraText.includes("'")
                    }
                });
                
                // Check if we've reached the maximum segment count
                if (segments.length >= options.maxSegmentCount) {
                    break;
                }
            }
        } else if (options.segmentationStrategy === 'equal') {
            // Equal-sized segmentation
            const words = text.split(/\s+/);
            const segmentSize = Math.max(options.minSegmentSize, 
                                      Math.ceil(words.length / options.maxSegmentCount));
            
            for (let i = 0; i < words.length; i += segmentSize) {
                const segmentWords = words.slice(i, i + segmentSize);
                const segmentText = segmentWords.join(' ');
                
                // Find the actual character positions in the original text
                const startIdx = text.indexOf(segmentWords[0], 
                                           i === 0 ? 0 : segments[segments.length - 1].endChar);
                const endIdx = startIdx + segmentText.length;
                
                // Extract features
                const features = this.featureExtractor.extractAllFeatures(segmentText);
                
                segments.push({
                    text: segmentText,
                    index: segments.length,
                    startChar: startIdx,
                    endChar: endIdx,
                    wordCount: segmentWords.length,
                    features,
                    boundaries: {
                        startsNewParagraph: text.substring(Math.max(0, startIdx - 2), startIdx).includes('\n\n'),
                        startsNewSection: false, // Cannot determine without full analysis
                        endsWithPunctuation: /[.!?]$/.test(segmentText.trim()),
                        containsDialog: segmentText.includes('"') || segmentText.includes("'")
                    }
                });
                
                // Check if we've reached the maximum segment count
                if (segments.length >= options.maxSegmentCount) {
                    break;
                }
            }
        } else if (options.segmentationStrategy === 'feature-based') {
            // Feature-based segmentation
            // This would require a more sophisticated approach analyzing text features
            // For now, we'll use a simplified version that looks for natural boundaries
            
            const structure = this.analyzeDocumentStructure(text);
            
            // Use sentences as the basic unit
            let currentSegmentText = '';
            let currentSegmentStart = 0;
            let currentWordCount = 0;
            
            for (let i = 0; i < structure.sentences.length; i++) {
                const sentence = structure.sentences[i];
                const sentenceText = text.substring(sentence.startChar, sentence.endChar);
                const wordCount = sentenceText.split(/\s+/).length;
                
                // If adding this sentence would exceed the target size, create a segment
                if (currentWordCount > 0 && 
                    currentWordCount + wordCount >= options.minSegmentSize) {
                    
                    // Extract features
                    const features = this.featureExtractor.extractAllFeatures(currentSegmentText);
                    
                    segments.push({
                        text: currentSegmentText,
                        index: segments.length,
                        startChar: currentSegmentStart,
                        endChar: sentence.startChar,
                        wordCount: currentWordCount,
                        features,
                        boundaries: {
                            startsNewParagraph: this.isStartOfParagraph(currentSegmentStart, structure),
                            startsNewSection: this.isStartOfSection(currentSegmentStart, structure),
                            endsWithPunctuation: true, // Sentences always end with punctuation
                            containsDialog: currentSegmentText.includes('"') || currentSegmentText.includes("'")
                        }
                    });
                    
                    // Start a new segment
                    currentSegmentText = sentenceText;
                    currentSegmentStart = sentence.startChar;
                    currentWordCount = wordCount;
                } else {
                    // Add this sentence to the current segment
                    if (currentSegmentText.length === 0) {
                        currentSegmentStart = sentence.startChar;
                    }
                    currentSegmentText += (currentSegmentText ? ' ' : '') + sentenceText;
                    currentWordCount += wordCount;
                }
                
                // Check if we've reached the maximum segment count
                if (segments.length >= options.maxSegmentCount) {
                    break;
                }
            }
            
            // Add the last segment if not empty
            if (currentSegmentText.length > 0) {
                const features = this.featureExtractor.extractAllFeatures(currentSegmentText);
                
                segments.push({
                    text: currentSegmentText,
                    index: segments.length,
                    startChar: currentSegmentStart,
                    endChar: text.length,
                    wordCount: currentWordCount,
                    features,
                    boundaries: {
                        startsNewParagraph: this.isStartOfParagraph(currentSegmentStart, structure),
                        startsNewSection: this.isStartOfSection(currentSegmentStart, structure),
                        endsWithPunctuation: /[.!?]$/.test(currentSegmentText.trim()),
                        containsDialog: currentSegmentText.includes('"') || currentSegmentText.includes("'")
                    }
                });
            }
        } else if (options.segmentationStrategy === 'hierarchical') {
            // Hierarchical segmentation by sections then paragraphs
            const structure = this.analyzeDocumentStructure(text);
            
            for (const section of structure.sections) {
                // Segment large sections further by paragraphs
                if (section.paragraphs.length > 3) {
                    for (const paraIdx of section.paragraphs) {
                        const para = structure.paragraphs[paraIdx];
                        const paraText = text.substring(para.startChar, para.endChar);
                        
                        // Skip very short paragraphs unless it's a heading
                        const wordCount = paraText.split(/\s+/).length;
                        if (wordCount < options.minSegmentSize && 
                            !this.isStartOfSection(para.startChar, structure)) {
                            continue;
                        }
                        
                        // Extract features
                        const features = this.featureExtractor.extractAllFeatures(paraText);
                        
                        segments.push({
                            text: paraText,
                            index: segments.length,
                            startChar: para.startChar,
                            endChar: para.endChar,
                            wordCount,
                            features,
                            boundaries: {
                                startsNewParagraph: true,
                                startsNewSection: this.isStartOfSection(para.startChar, structure),
                                endsWithPunctuation: /[.!?]$/.test(paraText.trim()),
                                containsDialog: paraText.includes('"') || paraText.includes("'")
                            }
                        });
                        
                        // Check if we've reached the maximum segment count
                        if (segments.length >= options.maxSegmentCount) {
                            break;
                        }
                    }
                } else {
                    // For small sections, use the entire section as a segment
                    const sectionText = text.substring(section.startChar, section.endChar);
                    
                    // Extract features
                    const features = this.featureExtractor.extractAllFeatures(sectionText);
                    
                    segments.push({
                        text: sectionText,
                        index: segments.length,
                        startChar: section.startChar,
                        endChar: section.endChar,
                        wordCount: sectionText.split(/\s+/).length,
                        features,
                        boundaries: {
                            startsNewParagraph: true,
                            startsNewSection: true,
                            endsWithPunctuation: /[.!?]$/.test(sectionText.trim()),
                            containsDialog: sectionText.includes('"') || sectionText.includes("'")
                        }
                    });
                    
                    // Check if we've reached the maximum segment count
                    if (segments.length >= options.maxSegmentCount) {
                        break;
                    }
                }
            }
        }
        
        return segments;
    }
    
    /**
     * Get carrier techniques with detectability below threshold
     * 
     * @param detectabilityThreshold Maximum detectability threshold (0-1)
     * @returns Array of carrier techniques
     */
    private getCarrierTechniques(detectabilityThreshold: number): CarrierTechnique[] {
        // In a real implementation, this would access the carrier techniques
        // from the StylometricCarrier class. For now, we'll define some example techniques.
        
        // Get carrier techniques from the StylometricCarrier
        const allTechniques: CarrierTechnique[] = [
            {
            id: 'sentence_length',
            name: 'Sentence Length Pattern',
            category: 'phraseology',
            bitsPerThousandWords: 8,
            detectability: 0.3,
            apply: (text: string, bits: boolean[]): { modifiedText: string; bitsEncoded: number } => {
                // Real implementation would alter sentence lengths to encode bits
            },
            {
                id: 'paragraph_structure',
                name: 'Paragraph Structure',
                category: 'phraseology' as 'phraseology' | 'punctuation' | 'linguistic' | 'readability',
                bitsPerThousandWords: 5,
                detectability: 0.35,
                apply: () => ({ modifiedText: '', bitsEncoded: 0 }),
                extract: () => [],
                estimate: () => 0
            },
            {
                id: 'punctuation_frequency',
                name: 'Punctuation Frequency',
                category: 'punctuation' as 'phraseology' | 'punctuation' | 'linguistic' | 'readability',
                bitsPerThousandWords: 12,
                detectability: 0.25,
                apply: () => ({ modifiedText: '', bitsEncoded: 0 }),
                extract: () => [],
                estimate: () => 0
            },
            {
                id: 'quote_style',
                name: 'Quote Style Alternation',
                category: 'punctuation' as 'phraseology' | 'punctuation' | 'linguistic' | 'readability',
                bitsPerThousandWords: 4,
                detectability: 0.2,
                apply: () => ({ modifiedText: '', bitsEncoded: 0 }),
                extract: () => [],
                estimate: () => 0
            },
            {
                id: 'optional_comma',
                name: 'Optional Comma Placement',
                category: 'punctuation' as 'phraseology' | 'punctuation' | 'linguistic' | 'readability',
                bitsPerThousandWords: 10,
                detectability: 0.2,
                apply: () => ({ modifiedText: '', bitsEncoded: 0 }),
                extract: () => [],
                estimate: () => 0
            },
            {
                id: 'synonym_substitution',
                name: 'Synonym Substitution',
                category: 'linguistic' as 'phraseology' | 'punctuation' | 'linguistic' | 'readability',
                bitsPerThousandWords: 10,
                detectability: 0.4,
                apply: () => ({ modifiedText: '', bitsEncoded: 0 }),
                extract: () => [],
                estimate: () => 0
            }
        ].filter(technique => technique.detectability <= detectabilityThreshold);
    }
    
    /**
     * Calculate capacity for a segment and technique
     * 
     * @param segment Text segment
     * @param technique Carrier technique
     * @returns Capacity in bits
     */
    private calculateCapacity(segment: TextSegment, technique: CarrierTechnique): number {
        // Calculate capacity based on technique and segment characteristics
        const baseCapacity = Math.floor((segment.wordCount / 1000) * technique.bitsPerThousandWords);
        
        // Adjust capacity based on segment features
        let adjustedCapacity = baseCapacity;
        
        switch (technique.id) {
            case 'sentence_length':
                // More effective with varied sentence lengths
                if (segment.features.stdev_words_per_sentence > 5) {
                    adjustedCapacity *= 1.2;
                }
                break;
                
            case 'paragraph_structure':
                // More effective at paragraph boundaries
                if (segment.boundaries.startsNewParagraph) {
                    adjustedCapacity *= 1.5;
                }
                break;
                
            case 'punctuation_frequency':
                // More effective with high punctuation density
                const punctuationCount = (segment.text.match(/[,;:!?]/g) || []).length;
                const punctuationDensity = punctuationCount / segment.wordCount;
                if (punctuationDensity > 0.15) {
                    adjustedCapacity *= 1.3;
                }
                break;
                
            case 'quote_style':
                // Only effective with dialog
                if (!segment.boundaries.containsDialog) {
                    adjustedCapacity = 0;
                } else {
                    const quoteCount = (segment.text.match(/["']/g) || []).length;
                    adjustedCapacity = Math.floor(quoteCount / 4); // ~1 bit per 4 quote characters
                }
                break;
                
            case 'optional_comma':
                // More effective with complex sentences
                if (segment.features.mean_words_per_sentence > 15) {
                    adjustedCapacity *= 1.2;
                }
                break;
                
            case 'synonym_substitution':
                // More effective with rich vocabulary
                if (segment.features.lexical_richness > 0.7) {
                    adjustedCapacity *= 1.3;
                }
                break;
        }
        
        return Math.max(0, Math.floor(adjustedCapacity));
    }
    
    /**
     * Calculate weights for carrier techniques based on options
     * 
     * @param techniques Carrier techniques
     * @param options Analysis options
     * @returns Array of weights
     */
    private calculateTechniqueWeights(
        techniques: CarrierTechnique[],
        options: Required<CapacityAnalysisOptions>
    ): number[] {
        const weights: number[] = [];
        
        for (const technique of techniques) {
            // Base weight is 1.0
            let weight = 1.0;
            
            // Adjust based on prioritization options
            if (options.prioritizeNaturalness) {
                // Lower detectability means more natural text preservation
                weight *= (1.1 - technique.detectability);
            }
            
            if (options.prioritizeCapacity) {
                // Higher bits per thousand words means more capacity
                weight *= (technique.bitsPerThousandWords / 10);
            }
            
            if (options.prioritizeResilience) {
                // Lower detectability techniques are often more resilient
                weight *= (1.1 - technique.detectability);
            }
            
            weights.push(weight);
        }
        
        // Normalize weights so they sum to the number of techniques
        const weightSum = weights.reduce((sum, w) => sum + w, 0);
        const normalizedWeights = weights.map(
            w => w * (techniques.length / weightSum)
        );
        
        return normalizedWeights;
    }
    
    /**
     * Create optimal distribution that minimizes text distortion
     * 
     * @param matrix Capacity matrix
     * @param requiredBits Total bits required
     * @param options Optimization options
     * @returns Distribution matrix
     */
    private createOptimalDistribution(
        matrix: CapacityMatrix,
        requiredBits: number,
        options: OptimizationOptions
    ): number[][] {
        const distribution: number[][] = Array(matrix.segments.length)
            .fill(null)
            .map(() => Array(matrix.techniques.length).fill(0));
        
        // Create a flattened list of all segment-technique combinations
        type Cell = {
            segmentIdx: number;
            techniqueIdx: number;
            capacity: number;
            distortion: number;  // Lower is better
            efficiency: number;  // Higher is better (bits per distortion unit)
        };
        
        const cells: Cell[] = [];
        
        for (let i = 0; i < matrix.segments.length; i++) {
            const segment = matrix.segments[i];
            
            // Skip preserved segments
            if (options.preserveSegments && options.preserveSegments.includes(i)) {
                continue;
            }
            
            for (let j = 0; j < matrix.techniques.length; j++) {
                const techniqueId = matrix.techniques[j];
                const technique = matrix.techniqueMetadata[techniqueId];
                const capacity = matrix.weightedCapacities[i][j];
                
                // Skip if no capacity
                if (capacity <= 0) {
                    continue;
                }
                
                // Calculate distortion based on technique properties and segment features
                // Lower distortion is better for preserving text naturalness
                let distortion = technique.detectability;
                
                // Adjust for segment-specific factors
                if (segment.boundaries.startsNewParagraph && 
                    techniqueId === 'paragraph_structure') {
                    distortion *= 0.8; // Reduce distortion at natural boundaries
                }
                
                if (segment.boundaries.containsDialog && 
                    techniqueId === 'quote_style') {
                    distortion *= 0.7; // Quote style changes are less noticeable in dialog
                }
                
                // Calculate efficiency (bits per distortion unit)
                const efficiency = capacity / distortion;
                
                cells.push({
                    segmentIdx: i,
                    techniqueIdx: j,
                    capacity,
                    distortion,
                    efficiency
                });
            }
        }
        
        // Sort by efficiency (highest first)
        cells.sort((a, b) => b.efficiency - a.efficiency);
        
        // Greedy allocation
        let remainingBits = requiredBits;
        
        for (const cell of cells) {
            if (remainingBits <= 0) break;
            
            const { segmentIdx, techniqueIdx, capacity } = cell;
            
            // Allocate up to capacity or remaining bits
            const bits = Math.min(capacity, remainingBits);
            distribution[segmentIdx][techniqueIdx] = bits;
            remainingBits -= bits;
        }
        
        // Check if we've allocated all required bits
        if (remainingBits > 0) {
            throw new Error(
                `Could not allocate all bits: ${remainingBits} bits remaining`
            );
        }
        
        return distribution;
    }
    
    /**
     * Create minimal change distribution that concentrates changes
     * 
     * @param matrix Capacity matrix
     * @param requiredBits Total bits required
     * @param options Optimization options
     * @returns Distribution matrix
     */
    private createMinimalChangeDistribution(
        matrix: CapacityMatrix,
        requiredBits: number,
        options: OptimizationOptions
    ): number[][] {
        const distribution: number[][] = Array(matrix.segments.length)
            .fill(null)
            .map(() => Array(matrix.techniques.length).fill(0));
        
        // Group by segment
        const segmentCapacities: Array<{
            segmentIdx: number;
            totalCapacity: number;
            techniques: Array<{
                techniqueIdx: number;
                capacity: number;
            }>;
        }> = [];
        
        for (let i = 0; i < matrix.segments.length; i++) {
            // Skip preserved segments
            if (options.preserveSegments && options.preserveSegments.includes(i)) {
                continue;
            }
            
            let segmentTotal = 0;
            const techniques: Array<{techniqueIdx: number; capacity: number}> = [];
            
            for (let j = 0; j < matrix.techniques.length; j++) {
                const capacity = matrix.weightedCapacities[i][j];
                if (capacity > 0) {
                    techniques.push({ techniqueIdx: j, capacity });
                    segmentTotal += capacity;
                }
            }
            
            if (segmentTotal > 0) {
                segmentCapacities.push({
                    segmentIdx: i,
                    totalCapacity: segmentTotal,
                    techniques
                });
            }
        }
        
        // Sort segments by total capacity (highest first)
        segmentCapacities.sort((a, b) => b.totalCapacity - a.totalCapacity);
        
        // Greedy allocation by segment
        let remainingBits = requiredBits;
        
        for (const segment of segmentCapacities) {
            if (remainingBits <= 0) break;
            
            // Sort techniques for this segment by capacity (highest first)
            segment.techniques.sort((a, b) => b.capacity - a.capacity);
            
            // Allocate to techniques within this segment
            for (const tech of segment.techniques) {
                if (remainingBits <= 0) break;
                
                const bits = Math.min(tech.capacity, remainingBits);
                distribution[segment.segmentIdx][tech.techniqueIdx] = bits;
                remainingBits -= bits;
            }
        }
        
        // Check if we've allocated all required bits
        if (remainingBits > 0) {
            throw new Error(
                `Could not allocate all bits: ${remainingBits} bits remaining`
            );
        }
        
        return distribution;
    }
    
    /**
     * Create balanced distribution that spreads changes evenly
     * 
     * @param matrix Capacity matrix
     * @param requiredBits Total bits required
     * @param options Optimization options
     * @returns Distribution matrix
     */
    private createBalancedDistribution(
        matrix: CapacityMatrix,
        requiredBits: number,
        options: OptimizationOptions
    ): number[][] {
        const distribution: number[][] = Array(matrix.segments.length)
            .fill(null)
            .map(() => Array(matrix.techniques.length).fill(0));
        
        // Calculate total capacity and distribution ratio
        let totalCapacity = 0;
        for (let i = 0; i < matrix.segments.length; i++) {
            // Skip preserved segments
            if (options.preserveSegments && options.preserveSegments.includes(i)) {
                continue;
            }
            
            for (let j = 0; j < matrix.techniques.length; j++) {
                totalCapacity += matrix.weightedCapacities[i][j];
            }
        }
        
        const ratio = requiredBits / totalCapacity;
        
        // Allocate proportionally
        let allocatedBits = 0;
        
        for (let i = 0; i < matrix.segments.length; i++) {
            // Skip preserved segments
            if (options.preserveSegments && options.preserveSegments.includes(i)) {
                continue;
            }
            
            for (let j = 0; j < matrix.techniques.length; j++) {
                const capacity = matrix.weightedCapacities[i][j];
                if (capacity > 0) {
                    // Proportional allocation with floor to ensure integer
                    let bits = Math.floor(capacity * ratio);
                    
                    // Ensure we don't exceed required bits
                    if (allocatedBits + bits > requiredBits) {
                        bits = requiredBits - allocatedBits;
                    }
                    
                    distribution[i][j] = bits;
                    allocatedBits += bits;
                    
                    // Stop if we've allocated all bits
                    if (allocatedBits >= requiredBits) {
                        break;
                    }
                }
            }
            
            if (allocatedBits >= requiredBits) {
                break;
            }
        }
        
        // If we haven't allocated all bits due to flooring, add the remainder
        if (allocatedBits < requiredBits) {
            let remainingBits = requiredBits - allocatedBits;
            
            // Find cells with non-zero capacity and allocate remaining bits
            for (let i = 0; i < matrix.segments.length && remainingBits > 0; i++) {
                // Skip preserved segments
                if (options.preserveSegments && options.preserveSegments.includes(i)) {
                    continue;
                }
                
                for (let j = 0; j < matrix.techniques.length && remainingBits > 0; j++) {
                    const capacity = matrix.weightedCapacities[i][j];
                    if (capacity > distribution[i][j]) {
                        distribution[i][j]++;
                        remainingBits--;
                    }
                }
            }
        }
        
        return distribution;
    }
    
    /**
     * Create front-loaded distribution that prioritizes earlier segments
     * 
     * @param matrix Capacity matrix
     * @param requiredBits Total bits required
     * @param options Optimization options
     * @returns Distribution matrix
     */
    private createFrontLoadedDistribution(
        matrix: CapacityMatrix,
        requiredBits: number, 
        options: OptimizationOptions
    ): number[][] {
        const distribution: number[][] = Array(matrix.segments.length)
            .fill(null)
            .map(() => Array(matrix.techniques.length).fill(0));
        
        let remainingBits = requiredBits;
        
        // Iterate through segments from first to last
        for (let i = 0; i < matrix.segments.length && remainingBits > 0; i++) {
            // Skip preserved segments
            if (options.preserveSegments && options.preserveSegments.includes(i)) {
                continue;
            }
            
            // Sort techniques by capacity for this segment
            const techniques = matrix.techniques
                .map((id, idx) => ({ id, idx, capacity: matrix.weightedCapacities[i][idx] }))
                .filter(t => t.capacity > 0)
                .sort((a, b) => b.capacity - a.capacity);
            
            // Allocate to techniques within this segment
            for (const tech of techniques) {
                if (remainingBits <= 0) break;
                
                const bits = Math.min(tech.capacity, remainingBits);
                distribution[i][tech.idx] = bits;
                remainingBits -= bits;
            }
        }
        
        // Check if we've allocated all required bits
        if (remainingBits > 0) {
            throw new Error(
                `Could not allocate all bits: ${remainingBits} bits remaining`
            );
        }
        
        return distribution;
    }
    
    /**
     * Create back-loaded distribution that prioritizes later segments
     * 
     * @param matrix Capacity matrix
     * @param requiredBits Total bits required
     * @param options Optimization options
     * @returns Distribution matrix
     */
    private createBackLoadedDistribution(
        matrix: CapacityMatrix,
        requiredBits: number,
        options: OptimizationOptions
    ): number[][] {
        const distribution: number[][] = Array(matrix.segments.length)
            .fill(null)
            .map(() => Array(matrix.techniques.length).fill(0));
        
        let remainingBits = requiredBits;
        
        // Iterate through segments from last to first
        for (let i = matrix.segments.length - 1; i >= 0 && remainingBits > 0; i--) {
            // Skip preserved segments
            if (options.preserveSegments && options.preserveSegments.includes(i)) {
                continue;
            }
            
            // Sort techniques by capacity for this segment
            const techniques = matrix.techniques
                .map((id, idx) => ({ id, idx, capacity: matrix.weightedCapacities[i][idx] }))
                .filter(t => t.capacity > 0)
                .sort((a, b) => b.capacity - a.capacity);
            
            // Allocate to techniques within this segment
            for (const tech of techniques) {
                if (remainingBits <= 0) break;
                
                const bits = Math.min(tech.capacity, remainingBits);
                distribution[i][tech.idx] = bits;
                remainingBits -= bits;
            }
        }
        
        // Check if we've allocated all required bits
        if (remainingBits > 0) {
            throw new Error(
                `Could not allocate all bits: ${remainingBits} bits remaining`
            );
        }
        
        return distribution;
    }
    
    /**
     * Check if a position is the start of a paragraph
     */
    private isStartOfParagraph(position: number, structure: DocumentStructure): boolean {
        return structure.paragraphs.some(p => p.startChar === position);
    }
    
    /**
     * Check if a position is the start of a section
     */
    private isStartOfSection(position: number, structure: DocumentStructure): boolean {
        return structure.sections.some(s => s.startChar === position);
    }
}

/**
 * Demonstrate capacity matrix analysis
 * 
 * @param text Sample text to analyze
 * @param payload Sample payload to embed
 */
export function demonstrateCapacityMatrix(text: string, payload: Uint8Array): void {
    console.log("=== CAPACITY MATRIX ANALYSIS DEMO ===");
    
    // Create matrix analyzer
    const analyzer = new CapacityMatrixAnalyzer();
    
    // 1. Analyze text structure
    console.log("\n1. ANALYZING DOCUMENT STRUCTURE:");
    console.log("--------------------------------");
    const structure = analyzer.analyzeDocumentStructure(text);
    console.log(`Document has ${structure.metadata.sectionCount} sections, ${structure.metadata.paragraphCount} paragraphs, ${structure.metadata.sentenceCount} sentences`);
    console.log(`Total word count: ${structure.metadata.wordCount}`);
    
    // 2. Identify embedding opportunities
    console.log("\n2. IDENTIFYING EMBEDDING OPPORTUNITIES:");
    console.log("-------------------------------------");
    const opportunities = analyzer.identifyEmbeddingOpportunities(structure);
    console.log("Embedding opportunities by technique:");
    for (const [technique, bits] of Object.entries(opportunities)) {
        console.log(`- ${technique}: ${bits} bits`);
    }
    
    // 3. Generate capacity matrix
    console.log("\n3. GENERATING CAPACITY MATRIX:");
    console.log("-----------------------------");
    const matrix = analyzer.analyzeCapacityMatrix(text, {
        segmentationStrategy: 'natural',
        prioritizeNaturalness: true
    });
    console.log(`Text divided into ${matrix.segments.length} segments`);
    console.log(`Available techniques: ${matrix.techniques.join(', ')}`);
    console.log(`Total capacity: ${matrix.totalCapacity} bits (${Math.floor(matrix.totalCapacity / 8)} bytes)`);
    console.log(`Recommended capacity: ${matrix.recommendedCapacity} bits (${Math.floor(matrix.recommendedCapacity / 8)} bytes)`);
    
    // 4. Generate optimal embedding plan
    console.log("\n4. GENERATING EMBEDDING PLAN:");
    console.log("----------------------------");
    try {
        // Configure optimization options
        const options: OptimizationOptions = {
            minimizeChanges: true,
            preserveReadingLevel: true,
            redundancyFactor: 1.2, // 20% redundancy
            redundancyStrategy: 'erasure'
        };
        
        const plan = analyzer.optimizeEmbedding(matrix, payload, options);
        
        console.log(`Payload size: ${payload.length} bytes (${payload.length * 8} bits)`);
        console.log(`Planned embedding: ${plan.totalBits} bits (including redundancy)`);
        console.log(`Capacity utilization: ${(plan.utilization * 100).toFixed(1)}%`);
        console.log(`Expected text changes: ~${plan.expectedTextChanges}`);
        console.log("\nTechnique distribution:");
        for (const [technique, bits] of Object.entries(plan.techniqueDistribution).sort((a, b) => b[1] - a[1])) {
            if (bits > 0) {
                console.log(`- ${technique}: ${bits} bits (${(bits / plan.totalBits * 100).toFixed(1)}%)`);
            }
        }
        
        if (plan.redundantEncoding && plan.redundancyDetails) {
            console.log("\nRedundancy encoding details:");
            console.log(`Original payload: ${plan.redundancyDetails.originalSize} bits`);
            console.log(`Encoded size: ${plan.redundancyDetails.encodedSize} bits`);
            console.log(`Recovery requires ${plan.redundancyDetails.requiredFragments} out of ${plan.redundancyDetails.totalFragments} fragments`);
        }
        
        console.log(`\nSegment plans: ${plan.segmentPlans.length} segments modified`);
    } catch (error) {
        console.error(`Error in embedding plan: ${error.message}`);
    }
    
    console.log("\n=== DEMO COMPLETE ===");
}