/**
 * Carrier Matrix Implementation
 *
 * This module implements the weighted capacity matrix system for stylometric steganography,
 * providing document structure-aware encoding with parity/erasure coding for resilience.
 *
 * Flow:
 * 1. Document Analysis -> Capacity Matrix Construction & Weighting
 * 2. Erasure Coding of Payload
 * 3. Weighted Distribution of Coded Payload Chunks via Carriers
 * 4. Extraction & Erasure Code Correction
 *
 * See carrier_matrix.ApiNotes.md for detailed design rationale and constraints.
 * [paradigm:imperative]
 */

type FeatureMap = Record<string, number>; // Placeholder if StyleFeatureExtractor is not available

import { QuoteStyleCarrier } from './quote_style_carrier.mjs';
import { ReedSolomon } from './src/matrix/ErrorCorrection.js'; // Adjust path if needed
import { type Carrier, type CarrierMatrixOptions, type EncodeResult } from './src/types/CarrierTypes.js'; // Added EncodeResult
import { bytesToBits, bitsToBytes } from './src/utils/bitUtils.js'; // Assuming bit helpers exist

/**
 * Results from document analysis containing capacity and segment information.
 * Aligned with carrier_matrix.ApiNotes.md.
 */
interface AnalysisResult {
    totalCapacityBits: number; // Renamed for clarity
    segmentCount: number;
    segments: DocumentSegment[]; // Include segments for reference
    capacityMatrix: Map<string, Map<string, CapacityCell>>; // Include matrix for reference
}

/**
 * Represents a segment of a document (chapter, section, etc.)
 */
interface DocumentSegment {
    id: string;
    content: string;
    type: 'chapter' | 'section' | 'metadata' | 'forward' | 'notes';
    level: number;
    parent?: string;
}

/**
 * Capacity metrics for a specific carrier technique
 */
interface CarrierMetrics {
    capacityBits: number;
    detectability: number;  // 0-1 scale
    robustness: number;    // 0-1 scale
    naturalness: number;   // 0-1 scale
}

/**
 * Matrix cell containing capacity information for a segment-carrier pair
 */
interface CapacityCell {
    segment: DocumentSegment;
    carrier: string;
    metrics: CarrierMetrics;
    weight: number;
}

/**
 * Options for encoding configuration.
 * See carrier_matrix.ApiNotes.md.
 */
export interface EncodingOptions extends CarrierMatrixOptions { // Inherit from base options if applicable
    redundancyLevel?: number;  // 0-1 scale for parity data ratio
    preserveStructure?: boolean;
    maxDetectability?: number;
    preferredCarriers?: string[];
    versionAware?: boolean;
    rsBlockSize?: number; // Example: data block size for RS coding
}

/**
 * Describes the location and length of a specific chunk's encoded bits.
 * Used within EncodingMetadata.
 */
interface ChunkLocation {
    chunkIndex: number;     // Index of the chunk (data or parity)
    segmentId: string;      // ID of the segment where encoded
    carrierKey: string;     // Key of the carrier used
    bitOffset: number;      // Starting bit position within the carrier's extracted bits for this segment
    bitLength: number;      // Number of bits belonging to this chunk at this location
}

/**
 * Metadata required for decoding, returned by encodePayload.
 * Aligned with carrier_matrix.ApiNotes.md requirements for recovery.
 */
interface EncodingMetadata {
    totalChunks: number;    // Total number of chunks (data + parity)
    chunkMap: ChunkLocation[]; // Array mapping chunk parts to their locations
    // Optional: Add payload hash, version info, etc. later if needed
}

/**
 * Result object returned by encodePayload.
 */
interface EncodePayloadResult {
    encodedSegments: Map<string, string>;
    metadata: EncodingMetadata;
}

/**
 * CarrierMatrix provides optimal payload distribution and resilient encoding.
 * Implements the core logic described in carrier_matrix.ApiNotes.md.
 */
export class CarrierMatrix {
    private capacityMatrix: Map<string, Map<string, CapacityCell>>;
    private segments: DocumentSegment[];
    private carriers: Map<string, Carrier>; // Use Carrier interface type
    private options: Required<EncodingOptions>; // Use Required for definite options
    private rsCodec: ReedSolomon;

    constructor(options: EncodingOptions = {}) {
        this.capacityMatrix = new Map();
        this.segments = [];
        this.carriers = new Map();
        // Provide defaults for all options
        this.options = {
            redundancyLevel: options.redundancyLevel ?? 0.2,
            preserveStructure: options.preserveStructure ?? true,
            maxDetectability: options.maxDetectability ?? 0.3,
            preferredCarriers: options.preferredCarriers ?? [],
            versionAware: options.versionAware ?? false,
            rsBlockSize: options.rsBlockSize ?? 200, // Default RS data block size
        };

        // Assertions based on design
        if (this.options.redundancyLevel < 0 || this.options.redundancyLevel > 1) {
            throw new Error("Redundancy level must be between 0 and 1.");
        }
        if (this.options.maxDetectability < 0 || this.options.maxDetectability > 1) {
            throw new Error("Max detectability must be between 0 and 1.");
        }

        // Initialize Reed-Solomon codec
        this.rsCodec = new ReedSolomon(); // Assuming default constructor or pass options

        // Initialize known carriers
        this.initializeCarriers();
    }

    /**
     * Initialize available carrier techniques. Extensible point.
     */
    private initializeCarriers(): void {
        // Add quote style carrier
        this.carriers.set('quoteStyle', new QuoteStyleCarrier());
    }

    /**
     * Analyze document to extract segments, build capacity matrix, and calculate weights.
     * @param document Full document content as a string.
     * @returns Promise resolving to AnalysisResult containing matrix and stats.
     */
    public async analyzeDocument(document: string): Promise<AnalysisResult> {
        this.segments = this.segmentDocument(document);
        if (this.segments.length === 0) {
            console.warn("Document segmentation yielded no segments.");
        }

        await this.buildCapacityMatrix();
        this.calculateWeights();

        const totalCapacity = this.calculateTotalCapacity();
        console.log(`Document analysis complete. Segments: ${this.segments.length}, Total Weighted Capacity: ${totalCapacity.toFixed(0)} bits`);

        return {
            totalCapacityBits: totalCapacity,
            segmentCount: this.segments.length,
            segments: this.segments,
            capacityMatrix: this.capacityMatrix,
        };
    }

    /**
     * Segment document into logical parts.
     * Current implementation is basic (Markdown headings).
     * @param document Full document content.
     * @returns Array of document segments.
     */
    private segmentDocument(document: string): DocumentSegment[] {
        const segments: DocumentSegment[] = [];
        const chapters = document.split(/Chapter \d+|CHAPTER \d+/);

        chapters.forEach((chapterContent, idx) => {
            const trimmedContent = chapterContent.trim();
            if (trimmedContent.length === 0) return;

            if (idx === 0) {
                segments.push({ id: `frontmatter`, content: trimmedContent, type: 'forward', level: 0 });
            } else {
                const chapterId = `chapter-${idx}`;
                segments.push({ id: chapterId, content: trimmedContent, type: 'chapter', level: 1 });

                const lines = trimmedContent.split('\n');
                let currentSectionContent = '';
                let sectionIdx = 0;
                let inSection = false;
                for (const line of lines) {
                    const sectionMatch = line.match(/^(#{2,4})\s+(.*)/);
                    if (sectionMatch) {
                        if (inSection && currentSectionContent.trim().length > 0) {
                            segments.push({
                                id: `${chapterId}-section-${sectionIdx}`,
                                content: currentSectionContent.trim(),
                                type: 'section',
                                level: sectionMatch[1].length,
                                parent: chapterId
                            });
                        }
                        sectionIdx++;
                        currentSectionContent = line + '\n';
                        inSection = true;
                    } else if (inSection) {
                        currentSectionContent += line + '\n';
                    }
                }
                if (inSection && currentSectionContent.trim().length > 0) {
                    segments.push({
                        id: `${chapterId}-section-${sectionIdx}`,
                        content: currentSectionContent.trim(),
                        type: 'section',
                        level: 2,
                        parent: chapterId
                    });
                }
            }
        });
        return segments;
    }

    /**
     * Build capacity matrix by analyzing each segment with each carrier.
     */
    private async buildCapacityMatrix(): Promise<void> {
        this.capacityMatrix = new Map();
        if (this.carriers.size === 0) {
            console.warn("No carriers initialized. Capacity matrix will be empty.");
            return;
        }

        for (const segment of this.segments) {
            if (!segment.content || segment.content.trim().length === 0) {
                console.warn(`Segment ${segment.id} has empty content, skipping capacity analysis.`);
                continue;
            }

            const segmentMap = new Map<string, CapacityCell>();
            this.capacityMatrix.set(segment.id, segmentMap);

            for (const [carrierKey, carrier] of this.carriers.entries()) {
                try {
                    const metrics = await carrier.analyzeCapacity(segment.content);
                    segmentMap.set(carrierKey, {
                        segment,
                        carrier: carrierKey,
                        metrics,
                        weight: 1.0
                    });
                } catch (error) {
                    console.error(`Error analyzing capacity for segment ${segment.id} with carrier ${carrierKey}:`, error);
                }
            }
        }
    }

    /**
     * Calculate optimal weights for each segment-carrier pair.
     */
    private calculateWeights(): void {
        for (const [segmentId, carrierMap] of this.capacityMatrix.entries()) {
            for (const [carrierKey, cell] of carrierMap.entries()) {
                let weight = 1.0;

                if (cell.metrics.detectability > this.options.maxDetectability) {
                    weight *= Math.pow(1 - (cell.metrics.detectability - this.options.maxDetectability) / (1 - this.options.maxDetectability), 2);
                }

                weight *= (0.5 + cell.metrics.naturalness * 0.5);
                weight *= (0.7 + cell.metrics.robustness * 0.3);

                switch (cell.segment.type) {
                    case 'metadata': weight *= this.options.versionAware ? 1.5 : 0.5; break;
                    case 'forward': weight *= 0.8; break;
                    case 'chapter': weight *= 1.2; break;
                    case 'section': weight *= 1.1; break;
                    case 'notes': weight *= 0.7; break;
                }

                if (this.options.preferredCarriers.includes(carrierKey)) {
                    weight *= 1.3;
                }

                cell.weight = Math.max(0.001, weight);
            }
        }
    }

    /**
     * Encode payload data with redundancy and distribute across carriers.
     * Generates metadata required for decoding. See ApiNotes.
     * @param payload Data to encode as Uint8Array.
     * @returns Promise resolving to EncodePayloadResult containing modified segments and metadata.
     */
    public async encodePayload(payload: Uint8Array): Promise<EncodePayloadResult> {
        const totalCapacity = this.calculateTotalCapacity();
        const payloadBits = payload.length * 8;

        const estimatedParityBits = payloadBits * this.options.redundancyLevel / (1 - this.options.redundancyLevel);
        const requiredCapacity = payloadBits + estimatedParityBits;

        console.log(`Payload: ${payload.length} bytes (${payloadBits} bits). Required capacity (est.): ${requiredCapacity.toFixed(0)} bits. Available: ${totalCapacity.toFixed(0)} bits.`);

        if (requiredCapacity > totalCapacity) {
            throw new Error(`Payload too large: Requires ~${requiredCapacity.toFixed(0)} bits, but only ${totalCapacity.toFixed(0)} weighted bits available.`);
        }
        if (payload.length === 0) {
            console.warn("Payload is empty. Returning unmodified segments and empty metadata.");
            return {
                encodedSegments: new Map(this.segments.map(s => [s.id, s.content])),
                metadata: { totalChunks: 0, chunkMap: [] }
            };
        }

        const { dataChunks, parityChunks } = this.applyErasureCoding(payload);
        const allChunks = [...dataChunks, ...parityChunks];
        const totalChunks = allChunks.length;
        const totalEncodedBits = allChunks.reduce((sum, chunk) => sum + (chunk?.length ?? 0) * 8, 0);
        console.log(`Applied Reed-Solomon: ${dataChunks.length} data chunks, ${parityChunks.length} parity chunks. Total chunks: ${totalChunks}. Total encoded size: ${totalEncodedBits} bits.`);

        if (totalEncodedBits > totalCapacity) {
            throw new Error(`Payload with actual erasure coding overhead too large: Requires ${totalEncodedBits} bits, but only ${totalCapacity.toFixed(0)} weighted bits available.`);
        }

        const { encodedSegments, chunkMap } = await this.distributePayload(allChunks);

        const metadata: EncodingMetadata = {
            totalChunks,
            chunkMap,
        };

        return { encodedSegments, metadata };
    }

    /** Calculate total weighted capacity across all carriers and segments. */
    private calculateTotalCapacity(): number {
        let total = 0;
        for (const carrierMap of this.capacityMatrix.values()) {
            for (const cell of carrierMap.values()) {
                total += cell.metrics.capacityBits * cell.weight;
            }
        }
        return total;
    }

    /**
     * Apply Reed-Solomon erasure coding to the payload.
     * @param payload Original data.
     * @returns Object containing data and parity chunks.
     */
    private applyErasureCoding(payload: Uint8Array): { dataChunks: Uint8Array[], parityChunks: Uint8Array[] } {
        const dataSize = payload.length;
        const dataChunksCount = Math.ceil(dataSize / this.options.rsBlockSize);
        const parityChunksCount = Math.ceil(dataChunksCount * this.options.redundancyLevel);

        if (dataChunksCount === 0) {
            return { dataChunks: [], parityChunks: [] };
        }
        if (parityChunksCount === 0 && this.options.redundancyLevel > 0) {
            console.warn("Calculated 0 parity chunks despite redundancy level > 0. Ensure data size and block size are appropriate.");
        }

        const dataChunks: Uint8Array[] = [];
        for (let i = 0; i < dataSize; i += this.options.rsBlockSize) {
            const end = Math.min(i + this.options.rsBlockSize, dataSize);
            dataChunks.push(payload.slice(i, end));
        }

        try {
            const parityChunks = this.rsCodec.encode(dataChunks, parityChunksCount);
            if (parityChunks.length !== parityChunksCount) {
                console.warn(`ReedSolomon encode returned ${parityChunks.length} parity chunks, expected ${parityChunksCount}.`);
            }
            return { dataChunks, parityChunks };
        } catch (error) {
            console.error("Reed-Solomon encoding failed:", error);
            throw new Error("Failed to apply erasure coding.");
        }
    }

    /**
     * Distribute payload chunks across carriers based on capacity weights.
     * Iterates through chunks and finds suitable segment/carrier slots.
     * Generates the chunk mapping metadata required for decoding.
     * @param allChunks Array of data and parity chunks (Uint8Array).
     * @returns Object containing the map of modified segments and the chunk location map.
     */
    private async distributePayload(allChunks: Uint8Array[]): Promise<{ encodedSegments: Map<string, string>, chunkMap: ChunkLocation[] }> {
        const availableSlots: (CapacityCell & { availableCapacityBits: number })[] = [];
        for (const [segmentId, carrierMap] of this.capacityMatrix.entries()) {
            for (const [carrierKey, cell] of carrierMap.entries()) {
                const initialCapacity = cell.metrics.capacityBits;
                if (initialCapacity > 0 && cell.weight > 0) {
                    availableSlots.push({
                        ...cell,
                        availableCapacityBits: initialCapacity
                    });
                }
            }
        }

        availableSlots.sort((a, b) => b.weight - a.weight);

        const segmentModifications = new Map<string, string>();
        const currentSegmentContent = new Map<string, string>();
        this.segments.forEach(s => currentSegmentContent.set(s.id, s.content));

        const chunkMap: ChunkLocation[] = [];
        let totalBitsEncodedOverall = 0;

        for (let chunkIndex = 0; chunkIndex < allChunks.length; chunkIndex++) {
            const chunk = allChunks[chunkIndex];
            console.assert(chunk instanceof Uint8Array, `Chunk at index ${chunkIndex} is not a Uint8Array.`);

            const chunkBits = bytesToBits(chunk);
            let bitsEncodedForThisChunk = 0;
            let currentBitOffsetInChunk = 0;

            console.log(`Distributing Chunk ${chunkIndex} (${chunkBits.length} bits)...`);

            while (bitsEncodedForThisChunk < chunkBits.length) {
                let encodedInThisPass = false;

                for (const slot of availableSlots) {
                    if (slot.availableCapacityBits <= 0) continue;

                    const segmentId = slot.segment.id;
                    const carrierKey = slot.carrier;
                    const carrier = this.carriers.get(carrierKey);

                    if (!carrier) {
                        console.warn(`Carrier ${carrierKey} not found during distribution for chunk ${chunkIndex}.`);
                        continue;
                    }

                    const contentToEncodeIn = segmentModifications.get(segmentId) ?? currentSegmentContent.get(segmentId)!;
                    console.assert(contentToEncodeIn !== undefined, `Content for segment ${segmentId} not found.`);

                    const remainingBitsInChunk = chunkBits.length - bitsEncodedForThisChunk;
                    const bitsToAttempt = Math.min(remainingBitsInChunk, slot.availableCapacityBits);

                    if (bitsToAttempt <= 0) continue;

                    const bitSliceToEncode = chunkBits.slice(bitsEncodedForThisChunk, bitsEncodedForThisChunk + bitsToAttempt);

                    try {
                        const encodeResult: EncodeResult = await carrier.encode(contentToEncodeIn, bitSliceToEncode);
                        const actualBitsEncodedInCall = encodeResult.bitsEncoded;

                        if (actualBitsEncodedInCall > 0) {
                            encodedInThisPass = true;

                            chunkMap.push({
                                chunkIndex,
                                segmentId,
                                carrierKey,
                                bitOffset: currentBitOffsetInChunk,
                                bitLength: actualBitsEncodedInCall
                            });

                            segmentModifications.set(segmentId, encodeResult.modifiedContent);

                            bitsEncodedForThisChunk += actualBitsEncodedInCall;
                            currentBitOffsetInChunk += actualBitsEncodedInCall;
                            totalBitsEncodedOverall += actualBitsEncodedInCall;
                            slot.availableCapacityBits -= actualBitsEncodedInCall;

                            console.log(`  Encoded ${actualBitsEncodedInCall} bits of chunk ${chunkIndex} into ${segmentId}/${carrierKey}. Chunk progress: ${bitsEncodedForThisChunk}/${chunkBits.length}. Slot capacity left: ${slot.availableCapacityBits}`);

                            if (bitsEncodedForThisChunk >= chunkBits.length) {
                                break;
                            }
                        } else if (encodeResult.modifiedContent !== contentToEncodeIn) {
                            console.warn(`Carrier ${carrierKey} modified segment ${segmentId} but reported 0 bits encoded.`);
                            segmentModifications.set(segmentId, encodeResult.modifiedContent);
                        } else {
                            slot.availableCapacityBits = Math.max(0, slot.availableCapacityBits - 1);
                        }
                    } catch (error) {
                        console.error(`Error encoding ${bitsToAttempt} bits of chunk ${chunkIndex} into segment ${segmentId} using carrier ${carrierKey}:`, error);
                        slot.availableCapacityBits = 0;
                    }
                }

                if (!encodedInThisPass && bitsEncodedForThisChunk < chunkBits.length) {
                    const remaining = chunkBits.length - bitsEncodedForThisChunk;
                    console.error(`Could not find suitable slot for remaining ${remaining} bits of chunk ${chunkIndex}. Distribution failed.`);
                    throw new Error(`Failed to distribute chunk ${chunkIndex}. Capacity might be overestimated or carriers ineffective.`);
                }
            }
            console.log(`Finished distributing Chunk ${chunkIndex}.`);
        }

        console.log(`Payload distribution complete. Total bits encoded: ${totalBitsEncodedOverall}`);

        return { encodedSegments: segmentModifications, chunkMap };
    }

    /**
     * Decode payload from encoded document segments using carriers, metadata, and erasure coding.
     * @param encodedSegments Map of segment IDs to potentially modified content.
     * @param metadata The EncodingMetadata generated during the encode process.
     * @returns Promise resolving to the original decoded payload data, or null if recovery fails.
     */
    public async decodePayload(
        encodedSegments: Map<string, string>,
        metadata: EncodingMetadata
    ): Promise<Uint8Array | null> {
        if (!metadata || !metadata.chunkMap || typeof metadata.totalChunks !== 'number') {
            console.error("Decoding failed: Invalid or missing metadata provided.");
            throw new Error("Invalid or missing metadata for decoding.");
        }

        const { totalChunks, chunkMap } = metadata;
        if (totalChunks === 0) {
            console.log("Metadata indicates 0 chunks, returning empty payload.");
            return new Uint8Array(0);
        }

        const extractionCache = new Map<string, Map<string, Promise<boolean[] | null>>>();

        const getExtractedBits = async (segmentId: string, carrierKey: string, modifiedContent: string | undefined): Promise<boolean[] | null> => {
            if (!modifiedContent) return null;

            if (!extractionCache.has(segmentId)) {
                extractionCache.set(segmentId, new Map());
            }
            const segmentCache = extractionCache.get(segmentId)!;

            if (!segmentCache.has(carrierKey)) {
                const carrier = this.carriers.get(carrierKey);
                if (!carrier) {
                    console.warn(`Decoder: Carrier ${carrierKey} not found.`);
                    segmentCache.set(carrierKey, Promise.resolve(null));
                } else {
                    const extractionPromise = carrier.extract(modifiedContent)
                        .catch(err => {
                            console.error(`Decoder: Error extracting from ${segmentId}/${carrierKey}:`, err);
                            return null;
                        });
                    segmentCache.set(carrierKey, extractionPromise);
                }
            }
            return segmentCache.get(carrierKey)!;
        };

        const orderedChunks: (Uint8Array | null)[] = new Array(totalChunks).fill(null);
        let successfullyAssembledChunks = 0;

        console.log(`Attempting to decode payload: ${totalChunks} chunks expected.`);

        for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
            const parts = chunkMap.filter(p => p.chunkIndex === chunkIndex);
            parts.sort((a, b) => a.bitOffset - b.bitOffset);

            if (parts.length === 0) {
                console.warn(`Chunk ${chunkIndex}: No mapping found in metadata. Marking as missing.`);
                continue;
            }

            let assembledChunkBits: boolean[] = [];
            let expectedTotalBits = 0;
            let failedExtraction = false;

            for (const part of parts) {
                const { segmentId, carrierKey, bitOffset, bitLength } = part;
                expectedTotalBits += bitLength;

                const modifiedContent = encodedSegments.get(segmentId);
                const extractedBits = await getExtractedBits(segmentId, carrierKey, modifiedContent);

                if (extractedBits && extractedBits.length >= bitOffset + bitLength) {
                    const partBits = extractedBits.slice(bitOffset, bitOffset + bitLength);
                    console.assert(partBits.length === bitLength, `Chunk ${chunkIndex}, Part ${segmentId}/${carrierKey}: Expected ${bitLength} bits, got ${partBits.length}`);
                    assembledChunkBits.push(...partBits);
                } else {
                    console.warn(`Chunk ${chunkIndex}: Failed to extract part from ${segmentId}/${carrierKey} (offset ${bitOffset}, length ${bitLength}). Extracted: ${extractedBits ? extractedBits.length + ' bits' : 'null'}.`);
                    failedExtraction = true;
                    break;
                }
            }

            if (!failedExtraction) {
                if (assembledChunkBits.length !== expectedTotalBits) {
                    console.warn(`Chunk ${chunkIndex}: Assembled bits length (${assembledChunkBits.length}) does not match sum of part lengths (${expectedTotalBits}).`);
                }

                try {
                    const chunkBytes = bitsToBytes(assembledChunkBits);
                    orderedChunks[chunkIndex] = chunkBytes;
                    successfullyAssembledChunks++;
                    console.log(`Chunk ${chunkIndex}: Successfully assembled ${chunkBytes.length} bytes.`);
                } catch (error) {
                    console.error(`Chunk ${chunkIndex}: Error converting assembled bits to bytes:`, error);
                    orderedChunks[chunkIndex] = null;
                }
            } else {
                console.log(`Chunk ${chunkIndex}: Assembly failed due to missing parts.`);
            }
        }

        console.log(`Finished chunk assembly. ${successfullyAssembledChunks}/${totalChunks} chunks successfully assembled.`);

        return this.recoverPayloadFromChunks(orderedChunks);
    }

    /**
     * Recover original payload from potentially incomplete/corrupted chunks using Reed-Solomon.
     * @param chunks Array of recovered data chunks (potentially including parity, possibly with nulls for missing).
     * @returns Original payload data, or null if recovery fails.
     */
    private recoverPayloadFromChunks(chunks: (Uint8Array | null)[]): Uint8Array | null {
        if (chunks.every(c => c === null)) {
            console.error("Recovery failed: All chunks are missing.");
            return null;
        }

        try {
            console.log(`Attempting Reed-Solomon recovery with ${chunks.filter(c => c !== null).length} available chunks.`);
            const recoveredData = this.rsCodec.decode(chunks);

            if (recoveredData) {
                console.log(`Reed-Solomon recovery successful. Decoded payload size: ${recoveredData.length} bytes.`);
                return recoveredData;
            } else {
                console.error("Reed-Solomon recovery failed: Decoder returned null.");
                return null;
            }
        } catch (error) {
            console.error("Reed-Solomon decoding failed:", error);
            return null;
        }
    }
}

/**
 * Demonstrate the carrier matrix capabilities (Updated for Metadata).
 * Note: Requires `content` and `payload` to be defined elsewhere.
 * Includes basic test cases for success and simulated failure.
 */
export async function demonstrateCarrierMatrix(
    content: string,
    payload: Uint8Array,
    simulateLoss: boolean = false // Add option to simulate segment loss
): Promise<boolean> { // Return true on success, false on failure
    console.log(`\n=== CARRIER MATRIX DEMO (${simulateLoss ? 'Simulating Loss' : 'Normal'}) ===`);
    console.log(`Input Payload Size: ${payload.length} bytes`);
    let success = false;

    // Create matrix instance with options
    const matrix = new CarrierMatrix({
        redundancyLevel: 0.3, // Example: 30% redundancy
        maxDetectability: 0.4,
        preferredCarriers: ['quoteStyle'],
        rsBlockSize: 128, // Example block size
    });

    try {
        // 1. Analyze document
        console.log("\n1. ANALYZING DOCUMENT...");
        const analysisResult = await matrix.analyzeDocument(content);
        console.log(`   Segments found: ${analysisResult.segmentCount}`);
        console.log(`   Total weighted capacity: ${analysisResult.totalCapacityBits.toFixed(0)} bits`);
        // Assertion: Ensure capacity is reported
        console.assert(analysisResult.totalCapacityBits >= 0, "Analysis reported negative capacity.");


        // 2. Encode payload
        console.log("\n2. ENCODING PAYLOAD...");
        // encodePayload now returns an object with encodedSegments and metadata
        const { encodedSegments, metadata } = await matrix.encodePayload(payload);
        console.log(`   Encoded content generated for ${encodedSegments.size} segments.`);
        console.log(`   Generated Metadata: ${metadata.totalChunks} total chunks.`);
        // Assertion: Ensure metadata looks reasonable
        console.assert(metadata.totalChunks >= 0, "Encoding metadata reported negative chunks.");
        console.assert(metadata.chunkMap != null, "Encoding metadata missing chunkMap.");


        // --- Simulate Transmission/Modification (Optional) ---
        let segmentsToDecode = encodedSegments;
        if (simulateLoss && encodedSegments.size > 1) {
            // Create a copy to modify
            segmentsToDecode = new Map(encodedSegments);
            const segmentIds = Array.from(segmentsToDecode.keys());
            // Remove roughly 10-20% of segments carrying data, ensuring at least one remains
            const numToRemove = Math.max(1, Math.min(Math.floor(segmentIds.length * 0.2), segmentIds.length - 1));
            for (let i = 0; i < numToRemove; i++) {
                 // Remove a segment (e.g., the second one found)
                 const lostSegmentId = segmentIds[i+1]; // Avoid removing the first segment maybe?
                 if(lostSegmentId) {
                     segmentsToDecode.delete(lostSegmentId);
                     console.log(`   SIMULATING LOSS of segment: ${lostSegmentId}`);
                 }
            }
             console.log(`   Segments remaining for decoding: ${segmentsToDecode.size}`);
        } else if (simulateLoss) {
             console.log("   Skipping simulation: Not enough segments to simulate loss.");
        }


        // 3. Decode payload
        console.log("\n3. DECODING PAYLOAD...");
        // Pass the metadata obtained during encoding
        const decodedPayload = await matrix.decodePayload(segmentsToDecode, metadata);


        // 4. Verify result
        console.log("\n4. VERIFICATION...");
        if (decodedPayload) {
            console.log(`   Decoded Payload Size: ${decodedPayload.length} bytes`);
            // Compare original payload with decoded payload
            let match = payload.length === decodedPayload.length;
            if (match) {
                for (let i = 0; i < payload.length; i++) {
                    if (payload[i] !== decodedPayload[i]) {
                        match = false;
                        break;
                    }
                }
            }
            console.log(`   Payload matches original: ${match ? 'YES' : 'NO'`);
            if (!match) {
                console.error("   Verification failed: Decoded payload does not match original.");
                // Optionally log parts of the arrays for comparison
                // console.log("Original:", payload.slice(0, 20));
                // console.log("Decoded: ", decodedPayload.slice(0, 20));
                success = false; // Explicitly mark as failure
            } else {
                 // Verification successful
                 success = true;
                 if (simulateLoss) {
                     console.log("   (Expected Success with simulated loss due to RS coding)");
                 } else {
                     console.log("   (Expected Success)");
                 }
            }
        } else {
            console.error("   Verification failed: Decoding returned null.");
            if (simulateLoss) {
                 console.log("   (Potential expected failure if loss exceeded RS tolerance)");
                 // Depending on RS parameters and loss amount, this might be expected.
                 // For the demo, we'll treat null return as failure.
                 success = false;
            } else {
                 console.log("   (Expected Success, but failed)");
                 success = false;
            }
        }

    } catch (error) {
        console.error("\n--- DEMO FAILED ---");
        console.error(`Error during demonstration: ${error.message}`);
        if (error.stack) {
            console.error(error.stack);
        }
        success = false; // Mark as failure on error
    }

    console.log(`\n=== DEMO COMPLETE (Success: ${success}) ===`);
    return success; // Return status
}

// Example Usage (Add to a main script or test runner)
/*
import fs from 'fs/promises'; // Use ESM import

async function runDemo() {
    try {
        const sampleDocument = await fs.readFile('path/to/your/document.txt', 'utf-8');
        const samplePayload = new TextEncoder().encode("This is the secret message that needs to be encoded using the carrier matrix system with Reed-Solomon for resilience.");

        console.log("--- Running Normal Demo ---");
        await demonstrateCarrierMatrix(sampleDocument, samplePayload, false);

        console.log("\n--- Running Demo with Simulated Loss ---");
        await demonstrateCarrierMatrix(sampleDocument, samplePayload, true);

    } catch (err) {
        console.error("Error running demo:", err);
    }
}

runDemo();
*/