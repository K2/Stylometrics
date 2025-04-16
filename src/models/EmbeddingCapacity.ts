/**
 * EmbeddingCapacity Interface
 * 
 * Represents the capacity of a text to embed information using various
 * counterfactual patterns. Used to track and manage the information hiding
 * potential across different pattern types.
 * 
 * [paradigm:functional]
 */

import { CounterfactualPatternType } from '../embedding/CounterfactualPatterns';

/**
 * Defines the information hiding capacity of text across different pattern types
 */
export interface EmbeddingCapacity {
  /**
   * Total number of bits that can be embedded in the analyzed text
   */
  totalBits: number;
  
  /**
   * Map of pattern types to their specific bit capacities within the text
   */
  patternCapacity: Map<CounterfactualPatternType, number>;
  
  /**
   * Optional metadata about the capacity calculation
   */
  metadata?: {
    textLength?: number;
    confidenceScore?: number;
    detectionRisk?: number;
  };
}
