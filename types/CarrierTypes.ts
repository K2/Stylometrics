/**
 * CarrierTypes.ts
 * 
 * Defines common types and interfaces used across different carrier technique implementations.
 * This promotes consistency and reusability.
 * 
 * Design Goals: Centralize type definitions for carriers.
 * Architectural Constraints: Keep types simple and focused on carrier needs.
 * Happy Path: Carriers import necessary types from this file.
 * ApiNotes: ./ApiNotes.md (Project Level)
 */

// Define FeatureMap here if it's a shared concept, or import from detection module if defined there
export interface FeatureMap {
    [key: string]: number;
}

/**
 * Core type definitions for the Stylometrics carrier system
 * 
 * These interfaces define the contract for all carrier implementations
 * in the stylometric steganography framework.
 */

/**
 * Base configuration options available to all carriers
 */
export interface CarrierConfiguration {
    maxDetectability?: number;      // Maximum acceptable detectability level (0-1)
    minRobustness?: number;         // Minimum acceptable robustness level (0-1)
    prioritizeCapacity?: boolean;   // Whether to prioritize capacity over naturalness
    [key: string]: any;             // Allow for carrier-specific configuration options
}

/**
 * Performance metrics for stylometric carriers
 */
export interface CarrierMetrics {
    capacity: number;       // Number of bits that can be encoded
    detectability: number;  // How easily detected (0-1, lower is better)
    robustness: number;     // Resistance to modification (0-1, higher is better)
    naturalness: number;    // How natural the encoded text appears (0-1, higher is better)
}

/**
 * Result of an encoding operation
 */
export interface EncodeResult {
    modifiedContent: string; // The text with encoded bits
    bitsEncoded: number;     // Number of bits successfully encoded
}

/**
 * Interface for all stylometric carriers (High-level abstraction)
 */
export interface Carrier {
    readonly id: string;     // Unique identifier for this carrier
    readonly name: string;   // Human-readable name
    
    /** Get current configuration */
    getConfiguration(): Record<string, any>;
    
    /** Configure the carrier */
    configure(config: Record<string, any>): void;
    
    /** Analyze capacity and metrics for a given text */
    analyzeCapacity(content: string): Promise<CarrierMetrics>;
    
    /** Encode bits into text */
    encode(content: string, bits: boolean[]): Promise<EncodeResult>;
    
    /** Extract bits from encoded text */
    extract(content: string): Promise<boolean[] | null>;
}

/**
 * Interface for specific carrier techniques (Low-level implementation)
 */
export interface CarrierTechnique {
    analyze: any;
    estimateCapacity(text: string): number;
    id: string;
    name: string;
    category: 'phraseology' | 'punctuation' | 'linguistic' | 'readability';
    bitsPerThousandWords: number; // Approximate guide
    detectability: number; // 0-1 scale, lower is better
    apply: (text: string, bits: boolean[]) => { modifiedText: string; bitsEncoded: number };
    extract: (text: string) => boolean[];
    estimate: (text: string) => number; // Estimate capacity in bits
    getDetectability(): number;
    getCapacity(text: string): number;
    getNaturalness(): number;
    encode(text: string, bits: boolean[]): { modifiedText: string; bitsEncoded: number };
    getRobustness(): number;
}

// Add other shared types if needed, e.g., for specific analysis results
