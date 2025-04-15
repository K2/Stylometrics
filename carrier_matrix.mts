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
// Define base options interface if not imported
interface CarrierMatrixOptions {
    // Base options common to matrix operations, if any
}
import { type Carrier, type EncodeResult } from './src/types/CarrierTypes.js'; // Removed CarrierMatrixOptions import
import { bytesToBits, bitsToBytes } from './src/utils/BitUtils.js'; // Corrected import path and extension

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
 * Matches the definition in CarrierTypes.ts
 */
interface CarrierMetrics {
    capacity: number; // Changed from capacityBits to match CarrierTypes.ts
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
    metrics: CarrierMetrics; // Uses the updated CarrierMetrics interface
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
        this.rsCodec = new ReedSolomon(); // Assuming default constructor or pass options if needed

        // Initialize known carriers
        this.initializeCarriers();
    }

    /**
     * Initialize available carrier techniques. Extensible point.
     */
    private initializeCarriers(): void {
        // Add quote style carrier
        // Ensure QuoteStyleCarrier implements the Carrier interface fully
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
        // Basic Markdown chapter/section splitting
        const chapters = document.split(/^#\s+(.*)$/m); // Split by H1

        let currentChapterId: string | null = null;
        let chapterIndex = 0;
        let sectionIndex = 0;

        for (let i = 0; i < chapters.length; i++) {
            const content = chapters[i].trim();
            if (i % 2 === 1) { // Chapter title
                chapterIndex++;
                currentChapterId = `chapter-${chapterIndex}`;
                segments.push({ id: currentChapterId, content: `# ${content}`, type: 'chapter', level: 1 });
                sectionIndex = 0; // Reset section index for new chapter
            } else if (content.length > 0) { // Content between chapters or before first chapter
                if (currentChapterId) {
                    // Split content by sections (H2, H3, H4)
                    const sectionParts = content.split(/^(#{2,4})\s+(.*)$/m);
                    let currentSectionContent = '';
                    let currentSectionLevel = 2; // Default level if no sections found within chapter content

                    for (let j = 0; j < sectionParts.length; j++) {
                        const sectionContent = sectionParts[j].trim();
                        if (j % 3 === 1) { // Section heading level (e.g., ##)
                            currentSectionLevel = sectionContent.length;
                        } else if (j % 3 === 2) { // Section title
                            if (currentSectionContent.trim().length > 0) {
                                // Add previous section/content before starting new one
                                segments.push({
                                    id: `${currentChapterId}-section-${sectionIndex}`,
                                    content: currentSectionContent.trim(),
                                    type: 'section',
                                    level: currentSectionLevel, // Use level captured before title
                                    parent: currentChapterId
                                });
                            }
                            sectionIndex++;
                            currentSectionContent = `${'#'.repeat(currentSectionLevel)} ${sectionContent}\n`; // Start new section content with heading
                        } else if (sectionContent.length > 0) { // Content within section or before first section
                            currentSectionContent += sectionContent + '\n';
                        }
                    }
                    // Add the last section/content part
                    if (currentSectionContent.trim().length > 0) {
                         segments.push({
                            id: `${currentChapterId}-section-${sectionIndex}`,
                            content: currentSectionContent.trim(),
                            type: 'section',
                            level: currentSectionLevel, // Use last known level
                            parent: currentChapterId
                        });
                    }

                } else {
                    // Content before the first chapter (e.g., frontmatter)
                    segments.push({ id: `frontmatter`, content: content, type: 'forward', level: 0 });
                }
            }
        }

        // Fallback if no chapters were found
        if (segments.length === 0 && document.trim().length > 0) {
             segments.push({ id: `main`, content: document.trim(), type: 'section', level: 1 });
        }

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
                    // Use the CarrierMetrics from CarrierTypes.ts which has 'capacity'
                    const metrics = await carrier.analyzeCapacity(segment.content);
                    segmentMap.set(carrierKey, {
                        segment,
                        carrier: carrierKey,
                        metrics: { // Ensure all properties are present
                            capacity: metrics.capacity,
                            detectability: metrics.detectability,
                            robustness: metrics.robustness,
                            naturalness: metrics.naturalness
                        },
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

                // Penalize high detectability
                if (cell.metrics.detectability > this.options.maxDetectability) {
                    // Sharper penalty as detectability exceeds max
                    weight *= Math.pow(1 - (cell.metrics.detectability - this.options.maxDetectability) / (1 - this.options.maxDetectability), 3);
                }

                // Reward naturalness and robustness
                weight *= (0.4 + cell.metrics.naturalness * 0.6); // Increased weight for naturalness
                weight *= (0.6 + cell.metrics.robustness * 0.4); // Increased weight for robustness

                // Adjust weight based on segment type
                switch (cell.segment.type) {
                    case 'metadata': weight *= this.options.versionAware ? 1.5 : 0.5; break;
                    case 'forward': weight *= 0.8; break;
                    case 'chapter': weight *= 1.2; break;
                    case 'section': weight *= 1.1; break;
                    case 'notes': weight *= 0.7; break;
                }

                // Boost preferred carriers
                if (this.options.preferredCarriers.includes(carrierKey)) {
                    weight *= 1.5; // Increased boost
                }

                // Ensure weight is positive but can be very small
                cell.weight = Math.max(0.0001, weight);
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

        // Calculate required capacity more accurately based on RS code parameters
        const dataChunksCount = Math.ceil(payload.length / this.options.rsBlockSize);
        const parityChunksCount = Math.ceil(dataChunksCount * this.options.redundancyLevel); // Simplified, actual RS might differ slightly
        const totalChunksEstimate = dataChunksCount + parityChunksCount;
        // Estimate total bits needed: payload bits + parity bits (assuming parity chunks are roughly same size as data chunks)
        const requiredCapacity = payloadBits + (parityChunksCount * this.options.rsBlockSize * 8);


        console.log(`Payload: ${payload.length} bytes (${payloadBits} bits). Required capacity (est.): ${requiredCapacity.toFixed(0)} bits. Available: ${totalCapacity.toFixed(0)} bits.`);

        if (requiredCapacity > totalCapacity) {
            // Provide more detailed error message
            const shortfall = requiredCapacity - totalCapacity;
            throw new Error(`Payload too large: Requires ~${requiredCapacity.toFixed(0)} bits (including estimated redundancy), but only ${totalCapacity.toFixed(0)} weighted bits available. Short by ~${shortfall.toFixed(0)} bits.`);
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
        // Calculate actual total bits after encoding
        const totalEncodedBits = allChunks.reduce((sum, chunk) => sum + (chunk?.length ?? 0) * 8, 0);
        console.log(`Applied Reed-Solomon: ${dataChunks.length} data chunks, ${parityChunks.length} parity chunks. Total chunks: ${totalChunks}. Total encoded size: ${totalEncodedBits} bits.`);

        if (totalEncodedBits > totalCapacity) {
             const shortfall = totalEncodedBits - totalCapacity;
            throw new Error(`Payload with actual erasure coding overhead too large: Requires ${totalEncodedBits} bits, but only ${totalCapacity.toFixed(0)} weighted bits available. Short by ${shortfall.toFixed(0)} bits.`);
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
                // Use 'capacity' from the metrics
                total += cell.metrics.capacity * cell.weight;
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
        // Ensure rsBlockSize is positive
        if (this.options.rsBlockSize <= 0) {
            throw new Error("rsBlockSize must be positive.");
        }
        const dataChunksCount = Math.ceil(dataSize / this.options.rsBlockSize);
        // Ensure redundancyLevel is valid for calculation
        if (this.options.redundancyLevel < 0 || this.options.redundancyLevel >= 1) {
             throw new Error("Redundancy level must be between 0 (inclusive) and 1 (exclusive) for parity calculation.");
        }
        // Calculate parity chunks needed based on the ratio of parity to *data* chunks
        const parityChunksCount = Math.ceil(dataChunksCount * this.options.redundancyLevel);


        if (dataChunksCount === 0 && dataSize > 0) {
             throw new Error("Calculated 0 data chunks for non-empty payload. Check rsBlockSize.");
        }
         if (dataChunksCount === 0) {
            return { dataChunks: [], parityChunks: [] };
        }
        if (parityChunksCount === 0 && this.options.redundancyLevel > 0) {
            console.warn("Calculated 0 parity chunks despite redundancy level > 0. Ensure data size and block size are appropriate, or redundancy level is high enough.");
        }

        const dataChunks: Uint8Array[] = [];
        for (let i = 0; i < dataSize; i += this.options.rsBlockSize) {
            const end = Math.min(i + this.options.rsBlockSize, dataSize);
            dataChunks.push(payload.slice(i, end));
        }

        // Pad the last data chunk if necessary (some RS implementations require equal chunk sizes)
        // This depends on the specific ReedSolomon library implementation.
        // Assuming the library handles padding or variable sizes. If not, padding is needed here.
        // const lastChunk = dataChunks[dataChunks.length - 1];
        // if (lastChunk.length < this.options.rsBlockSize) {
        //     const paddedChunk = new Uint8Array(this.options.rsBlockSize).fill(0);
        //     paddedChunk.set(lastChunk);
        //     dataChunks[dataChunks.length - 1] = paddedChunk;
        // }


        try {
            // Pass the expected number of data and parity chunks if the library requires it
            // Assuming encode(data, numParity) signature
            const parityChunks = this.rsCodec.encode(dataChunks, parityChunksCount);
            if (parityChunks.length !== parityChunksCount) {
                console.warn(`ReedSolomon encode returned ${parityChunks.length} parity chunks, expected ${parityChunksCount}.`);
                // Depending on the library, this might be acceptable or an error.
            }
            return { dataChunks, parityChunks };
        } catch (error) {
            console.error("Reed-Solomon encoding failed:", error);
            throw new Error(`Failed to apply erasure coding: ${error.message}`);
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
                // Use 'capacity' from metrics
                const initialCapacity = cell.metrics.capacity;
                if (initialCapacity > 0 && cell.weight > 0) {
                    availableSlots.push({
                        ...cell,
                        // Use weighted capacity for distribution decisions? Or raw capacity?
                        // Using raw capacity here, weighted capacity used for sorting.
                        availableCapacityBits: initialCapacity
                    });
                }
            }
        }

        // Sort by weighted capacity (weight * capacity) for better prioritization
        availableSlots.sort((a, b) => (b.weight * b.metrics.capacity) - (a.weight * a.metrics.capacity));


        const segmentModifications = new Map<string, string>();
        const currentSegmentContent = new Map<string, string>();
        this.segments.forEach(s => currentSegmentContent.set(s.id, s.content));

        const chunkMap: ChunkLocation[] = [];
        let totalBitsEncodedOverall = 0;

        for (let chunkIndex = 0; chunkIndex < allChunks.length; chunkIndex++) {
            const chunk = allChunks[chunkIndex];
            // Replace console.assert with if/throw
            if (!(chunk instanceof Uint8Array)) {
                 throw new Error(`Chunk at index ${chunkIndex} is not a Uint8Array.`);
            }


            const chunkBits = bytesToBits(chunk); // Use imported function
            let bitsEncodedForThisChunk = 0;
            let currentBitOffsetInChunk = 0; // Tracks position within the *chunk's* bits

            console.log(`Distributing Chunk ${chunkIndex} (${chunkBits.length} bits)...`);

            while (bitsEncodedForThisChunk < chunkBits.length) {
                let encodedInThisPass = false;
                let bestSlotIndex = -1;
                let maxEncodableBitsInSlot = 0;

                // Find the best slot in the current pass (highest weight first)
                for (let i = 0; i < availableSlots.length; i++) {
                    const slot = availableSlots[i];
                     if (slot.availableCapacityBits <= 0) continue;

                     // Simple selection: use the first available slot with capacity
                     bestSlotIndex = i;
                     maxEncodableBitsInSlot = slot.availableCapacityBits;
                     break; // Take the highest weighted available slot
                }


                if (bestSlotIndex === -1) {
                     // No slots with available capacity found
                     const remaining = chunkBits.length - bitsEncodedForThisChunk;
                     console.error(`Could not find any suitable slot for remaining ${remaining} bits of chunk ${chunkIndex}. Distribution failed.`);
                     throw new Error(`Failed to distribute chunk ${chunkIndex}. No available capacity in any slot.`);
                }


                const slot = availableSlots[bestSlotIndex];
                const segmentId = slot.segment.id;
                const carrierKey = slot.carrier;
                const carrier = this.carriers.get(carrierKey);

                if (!carrier) {
                    console.warn(`Carrier ${carrierKey} not found during distribution for chunk ${chunkIndex}. Skipping slot.`);
                    slot.availableCapacityBits = 0; // Mark slot as unusable
                    continue; // Try next iteration to find another slot
                }

                const contentToEncodeIn = segmentModifications.get(segmentId) ?? currentSegmentContent.get(segmentId)!;
                 // Replace console.assert
                 if (contentToEncodeIn === undefined) {
                     throw new Error(`Content for segment ${segmentId} not found.`);
                 }


                const remainingBitsInChunk = chunkBits.length - bitsEncodedForThisChunk;
                // Attempt to encode as many bits as possible, up to slot capacity or remaining chunk bits
                const bitsToAttempt = Math.min(remainingBitsInChunk, slot.availableCapacityBits);


                if (bitsToAttempt <= 0) {
                     slot.availableCapacityBits = 0; // Mark slot as exhausted for safety
                     continue; // Should not happen if bestSlotIndex was found, but check anyway
                }


                // Get the specific slice of bits from the current chunk to encode
                const bitSliceToEncode = chunkBits.slice(bitsEncodedForThisChunk, bitsEncodedForThisChunk + bitsToAttempt);


                try {
                    const encodeResult: EncodeResult = await carrier.encode(contentToEncodeIn, bitSliceToEncode);
                    const actualBitsEncodedInCall = encodeResult.bitsEncoded;

                    if (actualBitsEncodedInCall > 0) {
                        encodedInThisPass = true; // Mark that progress was made

                        // Record the location of the successfully encoded part of the chunk
                        chunkMap.push({
                            chunkIndex,
                            segmentId,
                            carrierKey,
                            // The offset within the *chunk's* data that these bits represent
                            bitOffset: bitsEncodedForThisChunk,
                            bitLength: actualBitsEncodedInCall
                        });

                        // Update the segment content with the modifications
                        segmentModifications.set(segmentId, encodeResult.modifiedContent);

                        // Update progress counters
                        bitsEncodedForThisChunk += actualBitsEncodedInCall;
                        // currentBitOffsetInChunk is implicitly tracked by bitsEncodedForThisChunk
                        totalBitsEncodedOverall += actualBitsEncodedInCall;
                        slot.availableCapacityBits -= actualBitsEncodedInCall; // Reduce slot capacity

                        console.log(`  Encoded ${actualBitsEncodedInCall} bits of chunk ${chunkIndex} into ${segmentId}/${carrierKey}. Chunk progress: ${bitsEncodedForThisChunk}/${chunkBits.length}. Slot capacity left: ${slot.availableCapacityBits.toFixed(0)}`);

                        // If the current chunk is fully encoded, break the inner loop to move to the next chunk
                        if (bitsEncodedForThisChunk >= chunkBits.length) {
                            break; // Exit the while loop for this chunk
                        }
                        // Continue the while loop to encode the rest of the current chunk in the next best slot

                    } else {
                         // Carrier reported 0 bits encoded.
                         if (encodeResult.modifiedContent !== contentToEncodeIn) {
                            // Content was modified but no bits reported? Suspicious. Update content anyway.
                            console.warn(`Carrier ${carrierKey} modified segment ${segmentId} but reported 0 bits encoded.`);
                            segmentModifications.set(segmentId, encodeResult.modifiedContent);
                         }
                         // Assume this slot is ineffective for now, reduce capacity slightly to avoid infinite loops
                         slot.availableCapacityBits = Math.max(0, slot.availableCapacityBits - 1);
                         console.log(`  Carrier ${carrierKey} in ${segmentId} reported 0 bits encoded for ${bitsToAttempt} attempted bits. Reducing slot capacity slightly.`);
                    }
                } catch (error) {
                    console.error(`Error encoding ${bitsToAttempt} bits of chunk ${chunkIndex} into segment ${segmentId} using carrier ${carrierKey}:`, error);
                    // Mark slot as unusable after error
                    slot.availableCapacityBits = 0;
                }
            } // End of while loop for encoding a single chunk

            // After exiting the while loop, check if the chunk was fully encoded
            if (bitsEncodedForThisChunk < chunkBits.length) {
                 const remaining = chunkBits.length - bitsEncodedForThisChunk;
                 console.error(`Failed to fully distribute chunk ${chunkIndex}. ${remaining} bits remaining. Insufficient effective capacity or carrier errors.`);
                 throw new Error(`Failed to distribute chunk ${chunkIndex}. Capacity might be overestimated or carriers ineffective.`);
            }


            console.log(`Finished distributing Chunk ${chunkIndex}.`);
        } // End of for loop iterating through all chunks

        console.log(`Payload distribution complete. Total bits encoded: ${totalBitsEncodedOverall}`);

        // Return only the segments that were actually modified
        const finalEncodedSegments = new Map<string, string>();
        for (const [id, content] of segmentModifications.entries()) {
            finalEncodedSegments.set(id, content);
        }
        // Include unmodified segments? The caller might expect the full set.
        // Let's return only modified ones for now. The decoder needs the map passed to it anyway.


        return { encodedSegments: finalEncodedSegments, chunkMap };
    }


    /**
     * Decode payload from encoded document segments using carriers, metadata, and erasure coding.
     * @param encodedSegments Map of segment IDs to potentially modified content.
     * @param metadata The EncodingMetadata generated during the encode process.
     * @returns Promise resolving to the original decoded payload data, or null if recovery fails.
     */
    public async decodePayload(
        encodedSegments: Map<string, string>, // Should contain content for segments listed in metadata
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
        if (chunkMap.length === 0 && totalChunks > 0) {
             console.error("Decoding failed: Metadata has non-zero totalChunks but empty chunkMap.");
             throw new Error("Invalid metadata: chunkMap is empty.");
        }


        // Cache for extracted bits from each segment/carrier pair to avoid redundant extraction
        const extractionCache = new Map<string, Map<string, Promise<boolean[] | null>>>();

        // Helper function to get extracted bits, using cache
        const getExtractedBits = async (segmentId: string, carrierKey: string): Promise<boolean[] | null> => {
            const modifiedContent = encodedSegments.get(segmentId);
            if (!modifiedContent) {
                 console.warn(`Decoder: Segment ${segmentId} not found in provided encodedSegments map.`);
                 return null; // Segment content missing
            }


            if (!extractionCache.has(segmentId)) {
                extractionCache.set(segmentId, new Map());
            }
            const segmentCache = extractionCache.get(segmentId)!;

            if (!segmentCache.has(carrierKey)) {
                const carrier = this.carriers.get(carrierKey);
                if (!carrier) {
                    console.warn(`Decoder: Carrier ${carrierKey} not found.`);
                    // Store null promise in cache to avoid retrying
                    segmentCache.set(carrierKey, Promise.resolve(null));
                } else {
                    // Start extraction and store the promise in the cache
                    const extractionPromise = carrier.extract(modifiedContent)
                        .then(bits => {
                            if (!bits) {
                                console.warn(`Decoder: Extraction from ${segmentId}/${carrierKey} returned null or empty.`);
                                return null;
                            }
                            return bits;
                        })
                        .catch(err => {
                            console.error(`Decoder: Error extracting from ${segmentId}/${carrierKey}:`, err);
                            return null; // Return null on error
                        });
                    segmentCache.set(carrierKey, extractionPromise);
                }
            }
            // Return the promise from the cache
            return segmentCache.get(carrierKey)!;
        };

        // Array to hold the assembled bits for each chunk before converting to bytes
        const assembledChunkBitsArray: (boolean[] | null)[] = new Array(totalChunks).fill(null);
        let successfullyAssembledChunks = 0;

        console.log(`Attempting to decode payload: ${totalChunks} chunks expected.`);

        // Group chunk parts by chunk index for easier processing
        const chunkPartsByIndex = new Map<number, ChunkLocation[]>();
        for (const part of chunkMap) {
            if (!chunkPartsByIndex.has(part.chunkIndex)) {
                chunkPartsByIndex.set(part.chunkIndex, []);
            }
            chunkPartsByIndex.get(part.chunkIndex)!.push(part);
        }

        // Process each chunk
        for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
            const parts = chunkPartsByIndex.get(chunkIndex);

            if (!parts || parts.length === 0) {
                console.warn(`Chunk ${chunkIndex}: No mapping found in metadata. Marking as missing.`);
                assembledChunkBitsArray[chunkIndex] = null; // Mark as missing
                continue;
            }

            // Sort parts by their bit offset within the chunk to assemble in correct order
            parts.sort((a, b) => a.bitOffset - b.bitOffset);

            let currentAssembledBits: boolean[] = [];
            let expectedNextBitOffset = 0;
            let failedExtraction = false;
            let totalExpectedBitsForChunk = 0;


            for (const part of parts) {
                const { segmentId, carrierKey, bitOffset, bitLength } = part;

                 // Check for gaps or overlaps in bit offsets
                 if (bitOffset !== expectedNextBitOffset) {
                     console.warn(`Chunk ${chunkIndex}: Unexpected bit offset gap or overlap. Expected ${expectedNextBitOffset}, got ${bitOffset} from ${segmentId}/${carrierKey}.`);
                     // Depending on strategy, might try to continue or fail the chunk
                     failedExtraction = true;
                     break;
                 }


                const extractedBits = await getExtractedBits(segmentId, carrierKey);

                if (extractedBits === null) {
                    // Extraction failed for this part
                    console.warn(`Chunk ${chunkIndex}: Failed to extract any bits from ${segmentId}/${carrierKey}.`);
                    failedExtraction = true;
                    break;
                }

                // Determine the start index within the *extracted* bits for this part.
                // This requires the carrier's `extract` method to return *all* bits it finds,
                // and the metadata's bitOffset/bitLength refer to positions within the *original* chunk data,
                // *not* positions within the carrier's extracted stream.
                // Let's refine the metadata or carrier interface if needed.
                // ---
                // Assumption Revisit: Let's assume `carrier.extract` returns *all* bits it encoded in that segment.
                // The `chunkMap` needs to store where *within the carrier's extracted bits* the chunk part lies.
                // This requires changing `encode` to return this info, or changing `extract`.
                // ---
                // Alternative Assumption: `bitOffset` and `bitLength` in `ChunkLocation` refer to the position
                // within the *original chunk data*. The `distributePayload` correctly sliced the chunk data.
                // The `decodePayload` needs to reassemble these slices.
                // Let's stick to this assumption for now.

                // We need to know the *total* number of bits expected for this chunk to preallocate or validate.
                // Let's calculate it from the parts list.
                if (part === parts[parts.length - 1]) { // Calculate on the last part
                    totalExpectedBitsForChunk = bitOffset + bitLength;
                }


                // We need the specific bits corresponding to this part.
                // How does the carrier know *which* bits to return if it just gets the whole segment?
                // The current `Carrier` interface `extract(content)` doesn't support this.
                // ---
                // Redesign Needed:
                // Option A: `extract` needs context (like expected length or offset).
                // Option B: `encode` needs to return precise location info within the *carrier's* potential bitstream.
                // Option C: `extract` returns *all* bits, and we assume the `ChunkLocation` refers to indices within *that* returned array.
                // ---
                // Let's try Option C for now, assuming `extract` gives all bits from that carrier in that segment.
                // And assume `ChunkLocation.bitOffset` refers to the index in the *extractedBits* array.
                // This implies `distributePayload` needs to track this offset during encoding.
                // ---
                // Re-Revisiting `distributePayload`: It *doesn't* track offset within the carrier stream. It tracks offset within the *chunk*.
                // This means Option C is incompatible with the current `distributePayload`.
                // ---
                // Let's revert to the original logic: `bitOffset` and `bitLength` refer to the chunk data.
                // We extract *all* bits from the carrier/segment. We need to know *where* in that stream
                // our desired `bitLength` bits start. This information is missing.
                // ---
                // Simplification: Assume each call to `carrier.encode(content, bitSlice)` only embeds *those* bits,
                // and `carrier.extract(modifiedContent)` returns *exactly* those bits in order.
                // This is a strong assumption about carrier behavior.

                // Let's proceed with the simplification: `extract` returns the bits for this part.
                // This requires `extract` to somehow know which part it's extracting, or for `encode` to have placed it predictably.
                // ---
                // Final Attempt with Current Structure: Assume `extract` returns *all* bits for that carrier in the segment.
                // Assume `ChunkLocation.bitOffset` is the starting index *within those extracted bits*.
                // Assume `ChunkLocation.bitLength` is the number of bits *at that location*.
                // This requires `distributePayload` to calculate and store this `bitOffset`. Let's modify `distributePayload` later if needed.

                if (extractedBits.length < bitOffset + bitLength) {
                    console.warn(`Chunk ${chunkIndex}: Extracted bits from ${segmentId}/${carrierKey} too short. Expected at least ${bitOffset + bitLength}, got ${extractedBits.length}.`);
                    failedExtraction = true;
                    break;
                }

                const partBits = extractedBits.slice(bitOffset, bitOffset + bitLength);
                // Replace console.assert
                if (partBits.length !== bitLength) {
                     // This should not happen if the slice parameters are correct and length check passed
                     console.error(`Chunk ${chunkIndex}, Part ${segmentId}/${carrierKey}: Slice error. Expected ${bitLength} bits, got ${partBits.length}`);
                     failedExtraction = true;
                     break;
                }

                currentAssembledBits.push(...partBits);
                expectedNextBitOffset = bitOffset + bitLength; // Update expected offset for the next part

            } // End loop over parts of a chunk

            // After processing all parts for the chunk
            if (!failedExtraction) {
                 // Calculate total expected bits from the last part's offset and length
                 const lastPart = parts[parts.length - 1];
                 totalExpectedBitsForChunk = lastPart.bitOffset + lastPart.bitLength;


                if (currentAssembledBits.length !== totalExpectedBitsForChunk) {
                    console.warn(`Chunk ${chunkIndex}: Assembled bits length (${currentAssembledBits.length}) does not match expected total length (${totalExpectedBitsForChunk}) based on metadata. Chunk may be corrupt.`);
                    // Decide whether to proceed with potentially corrupt chunk or mark as null
                    // For RS, providing a corrupt chunk might be worse than null. Mark as null.
                    assembledChunkBitsArray[chunkIndex] = null;
                } else {
                    assembledChunkBitsArray[chunkIndex] = currentAssembledBits;
                    successfullyAssembledChunks++;
                    console.log(`Chunk ${chunkIndex}: Successfully assembled ${currentAssembledBits.length} bits.`);
                }
            } else {
                console.log(`Chunk ${chunkIndex}: Assembly failed due to missing or inconsistent parts.`);
                assembledChunkBitsArray[chunkIndex] = null; // Mark as missing/failed
            }
        } // End loop over all chunks

        console.log(`Finished chunk assembly. ${successfullyAssembledChunks}/${totalChunks} chunks successfully assembled.`);

        // Convert assembled bits to bytes for each chunk
        const orderedChunks: (Uint8Array | null)[] = assembledChunkBitsArray.map((bits, index) => {
            if (bits) {
                try {
                    return bitsToBytes(bits); // Use imported function
                } catch (error) {
                    console.error(`Chunk ${index}: Error converting assembled bits to bytes:`, error);
                    return null;
                }
            }
            return null;
        });


        return this.recoverPayloadFromChunks(orderedChunks);
    }

    /**
     * Recover original payload from potentially incomplete/corrupted chunks using Reed-Solomon.
     * @param chunks Array of recovered data chunks (potentially including parity, possibly with nulls for missing).
     * @returns Original payload data, or null if recovery fails.
     */
    private recoverPayloadFromChunks(chunks: (Uint8Array | null)[]): Uint8Array | null {
        const availableChunkCount = chunks.filter(c => c !== null).length;
        if (availableChunkCount === 0) {
            console.error("Recovery failed: All chunks are missing or failed assembly.");
            return null;
        }

        try {
            console.log(`Attempting Reed-Solomon recovery with ${availableChunkCount}/${chunks.length} available chunks.`);
            // Assuming the decode function can handle nulls in the input array
            const recoveredData = this.rsCodec.decode(chunks);

            if (recoveredData) {
                console.log(`Reed-Solomon recovery successful. Decoded payload size: ${recoveredData.length} bytes.`);
                return recoveredData;
            } else {
                // Decoder returning null usually means insufficient chunks or uncorrectable errors
                console.error("Reed-Solomon recovery failed: Not enough data or too many errors.");
                return null;
            }
        } catch (error) {
            console.error("Reed-Solomon decoding threw an error:", error);
            return null;
        }
        // Ensure return path exists - though try/catch should cover it.
        // return null; // Should be unreachable if try/catch works
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
        if (!(analysisResult.totalCapacityBits >= 0)) {
             throw new Error("Assertion failed: Analysis reported negative capacity.");
        }


        // 2. Encode payload
        console.log("\n2. ENCODING PAYLOAD...");
        // encodePayload now returns an object with encodedSegments and metadata
        const { encodedSegments, metadata } = await matrix.encodePayload(payload);
        console.log(`   Encoded content generated for ${encodedSegments.size} segments.`);
        console.log(`   Generated Metadata: ${metadata.totalChunks} total chunks.`);
        // Assertion: Ensure metadata looks reasonable
        if (!(metadata.totalChunks >= 0)) {
             throw new Error("Assertion failed: Encoding metadata reported negative chunks.");
        }
        if (!(metadata.chunkMap != null)) {
             throw new Error("Assertion failed: Encoding metadata missing chunkMap.");
        }


        // --- Simulate Transmission/Modification (Optional) ---
        let segmentsToDecode = new Map(encodedSegments); // Start with a copy
        const originalSegmentIds = new Set(matrix.segments.map(s => s.id)); // Get all original segment IDs

        if (simulateLoss && encodedSegments.size > 1) {
            const segmentIdsCarryingData = new Set(metadata.chunkMap.map(p => p.segmentId));
            const deletableSegments = Array.from(segmentIdsCarryingData).filter(id => encodedSegments.has(id));

            if (deletableSegments.length > 1) {
                // Remove roughly 10-20% of segments carrying data, ensuring at least one remains
                const numToRemove = Math.max(1, Math.min(Math.floor(deletableSegments.length * 0.2), deletableSegments.length - 1));
                console.log(`   SIMULATING LOSS: Attempting to remove ${numToRemove} segment(s) carrying data...`);
                for (let i = 0; i < numToRemove; i++) {
                     // Remove a segment (e.g., the second one found carrying data)
                     const lostSegmentId = deletableSegments[i + 1]; // Avoid removing the first one maybe?
                     if(lostSegmentId && segmentsToDecode.has(lostSegmentId)) {
                         segmentsToDecode.delete(lostSegmentId);
                         console.log(`   -> Removed segment: ${lostSegmentId}`);
                     } else if (lostSegmentId) {
                         console.log(`   -> Segment ${lostSegmentId} already removed or not in encoded set.`);
                     }
                }
                 console.log(`   Segments remaining for decoding: ${segmentsToDecode.size}`);
            } else {
                 console.log("   Skipping loss simulation: Not enough distinct segments carrying data to remove.");
            }
        } else if (simulateLoss) {
             console.log("   Skipping loss simulation: Not enough encoded segments to simulate loss.");
        }

        // Add back any original segments that weren't modified/encoded into,
        // as the decoder might need the full document structure context,
        // or just to pass the complete (partially modified) document.
        for (const seg of matrix.segments) {
            if (!segmentsToDecode.has(seg.id)) {
                segmentsToDecode.set(seg.id, seg.content);
            }
        }


        // 3. Decode payload
        console.log("\n3. DECODING PAYLOAD...");
        // Pass the potentially reduced map of segments and the metadata
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
                        console.error(`   Mismatch at byte ${i}: Original=${payload[i]}, Decoded=${decodedPayload[i]}`);
                        break;
                    }
                }
            }
            console.log(`   Payload matches original: ${match ? 'YES' : 'NO'}`);
            if (!match) {
                console.error("   Verification failed: Decoded payload does not match original.");
                // Optionally log parts of the arrays for comparison
                console.log("   Original (first 20 bytes):", payload.slice(0, 20));
                console.log("   Decoded (first 20 bytes): ", decodedPayload.slice(0, 20));
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
        // Ensure error is an Error object
        if (error instanceof Error) {
            console.error(`Error during demonstration: ${error.message}`);
            if (error.stack) {
                console.error(error.stack);
            }
        } else {
             console.error("An unexpected error occurred during demonstration:", error);
        }
        success = false; // Mark as failure on error
    } // Fixed missing closing brace for catch

    console.log(`\n=== DEMO COMPLETE (Success: ${success}) ===`); // Fixed syntax
    return success; // Return status
}

// Example Usage (Add to a main script or test runner)
/* // Fixed comment termination
import fs from 'fs/promises'; // Use ESM import

async function runDemo() {
    try {
        // Use a placeholder document for testing if file read fails
        let sampleDocument = "Chapter 1\n\nThis is the first chapter.\n\n## Section 1.1\n\nSome content here with \"quotes\".\n\n## Section 1.2\n\nMore content with 'other quotes'.\n\nChapter 2\n\nSecond chapter content, \"double quotes\" again.";
        try {
             sampleDocument = await fs.readFile('path/to/your/document.txt', 'utf-8');
        } catch (readErr) {
             console.warn("Could not read document file, using placeholder content for demo.");
        }

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
*/ // Fixed comment termination