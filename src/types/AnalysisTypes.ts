/**
 * @module src/types/AnalysisTypes
 * @description Defines interfaces related to the analysis phase of the CarrierMatrix,
 * including capacity representation and weighting.
 */

import { CarrierMetrics } from "./CarrierTypes";
import { DocumentSegment } from "./DocumentTypes";

/**
 * Represents the calculated capacity and metrics for a specific carrier within a specific document segment.
 */
export interface CapacityCell extends CarrierMetrics {
    /** Calculated weight indicating the suitability of this cell for encoding (higher is better). */
    weight: number;
    /** Remaining bit capacity after allocation. */
    remainingCapacity: number;
    /** Unique identifier for the segment. */
    segmentId: string;
    /** Key identifying the carrier. */
    carrierKey: string;
}

/**
 * Represents the structure holding the capacity analysis for all carriers across all segments.
 * The matrix maps segment IDs to another map, which maps carrier keys to their CapacityCell.
 */
export type CapacityMatrix = Map<string, Map<string, CapacityCell>>;

/**
 * Represents the result of the document analysis phase.
 */
export interface AnalysisResult {
    /** The segmented document structure. */
    segments: DocumentSegment[];
    /** The calculated capacity matrix. */
    capacityMatrix: CapacityMatrix;
    /** Total estimated weighted capacity available across the document. */
    totalWeightedCapacity: number;
}

/**
 * Represents a plan for embedding a chunk or part of a chunk.
 */
export interface EmbeddingPlanItem {
    segmentId: string;
    carrierKey: string;
    bitsToEncode: number; // How many bits to attempt encoding in this slot
}

/**
 * Represents the overall plan for distributing the payload across segments and carriers.
 * Maps chunk indices to an array of plan items detailing where parts of that chunk should go.
 */
export type EmbeddingPlan = Map<number, EmbeddingPlanItem[]>;
