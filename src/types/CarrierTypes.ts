/**
 * CarrierTypes.ts
 * 
 * This file defines the core interfaces and types for carrier implementations
 * in the stylometric steganography system. These types establish the contract
 * that all carriers must follow to participate in the matrix-based encoding.
 * 
 * Design Goals:
 * - Provide a consistent interface for all carrier implementations
 * - Support capacity analysis for optimal payload distribution
 * - Enable detectability and robustness metrics for decision making
 * - Allow carriers to maintain their own state and configuration
 */

import type { DocumentSegment } from './DocumentTypes';

/**
 * Metrics describing a carrier's capacity and characteristics
 * for a specific segment of content
 */
export interface CarrierMetrics {
  /** Number of bits this carrier can encode in the given context */
  capacity: number;
  
  /** Likelihood of detection (0-1 scale, lower is better) */
  detectability: number;
  
  /** Resistance to modification/corruption (0-1 scale, higher is better) */
  robustness: number;
  
  /** How natural the encoding appears in the context (0-1 scale, higher is better) */
  naturalness: number;
}

/**
 * Result from an encode operation.
 */
export interface EncodeResult {
  /** The modified content after encoding. */
  modifiedContent: string;
  /** The actual number of bits successfully encoded. May be less than requested. */
  bitsEncoded: number;
}

/**
 * Interface for stylometric carrier techniques.
 * See relevant ApiNotes for specific carrier implementations.
 */
export interface Carrier {
  /** Unique identifier for this carrier type */
  readonly id: string;
  
  /** Human-readable name for this carrier */
  readonly name: string;
  
  /** Analyze content to determine potential capacity and metrics. */
  analyzeCapacity(content: string): Promise<CarrierMetrics>;

  /**
   * Encode data bits into the content.
   * @param content The original content segment.
   * @param bits The data bits to encode.
   * @returns A promise resolving to an object containing the modified content
   *          and the number of bits actually encoded.
   */
  encode(content: string, bits: boolean[]): Promise<EncodeResult>;

  /**
   * Extract data bits from potentially modified content.
   * @param content The content potentially containing encoded data.
   * @returns A promise resolving to the extracted bits, or null/empty if none found.
   */
  extract(content: string): Promise<boolean[] | null>;

  /** Get the configuration object for this carrier */
  getConfiguration(): Record<string, any>;
  
  /** Set configuration parameters for this carrier */
  configure(config: Record<string, any>): void;
}

/**
 * Configuration options for carrier behavior
 */
export interface CarrierConfiguration {
  /** Maximum acceptable detectability (0-1) */
  maxDetectability?: number;
  
  /** Minimum required robustness (0-1) */
  minRobustness?: number;
  
  /** Prefer capacity over stealth if true */
  prioritizeCapacity?: boolean;
  
  /** Carrier-specific options */
  [key: string]: any;
}

/**
 * Abstract base class for carriers to extend
 */
abstract class BaseCarrier implements Carrier {
  readonly id: string;
  readonly name: string;
  protected configuration: CarrierConfiguration;
  
  constructor(id: string, name: string, defaultConfig: CarrierConfiguration = {}) {
    this.id = id;
    this.name = name;
    this.configuration = {
      maxDetectability: 0.3,
      minRobustness: 0.5,
      prioritizeCapacity: false,
      ...defaultConfig
    };
  }
  
  abstract analyzeCapacity(content: string): Promise<CarrierMetrics>;
  abstract encode(content: string, bits: boolean[]): Promise<EncodeResult>;
  abstract extract(content: string): Promise<boolean[] | null>;
  
  getConfiguration(): Record<string, any> {
    return { ...this.configuration };
  }
  
  configure(config: Record<string, any>): void {
    this.configuration = {
      ...this.configuration,
      ...config
    };
  }
  
  /**
   * Utility method to convert boolean bits to a byte array
   */
  protected bitsToBytes(bits: boolean[]): Uint8Array {
    const bytes = new Uint8Array(Math.floor(bits.length / 8));
    for (let i = 0; i < bytes.length; i++) {
      let byte = 0;
      for (let j = 0; j < 8; j++) {
        if (bits[i * 8 + j]) {
          byte |= 1 << (7 - j);
        }
      }
      bytes[i] = byte;
    }
    return bytes;
  }
  
  /**
   * Utility method to convert a byte array to boolean bits
   */
  protected bytesToBits(bytes: Uint8Array): boolean[] {
    const bits: boolean[] = [];
    for (let i = 0; i < bytes.length; i++) {
      for (let j = 7; j >= 0; j--) {
        bits.push((bytes[i] & (1 << j)) !== 0);
      }
    }
    return bits;
  }
}

// Export the class for CommonJS compatibility with verbatimModuleSyntax
export type { BaseCarrier };
module.exports.BaseCarrier = BaseCarrier;