/**
 * Stylometric Toolkit - Unified Detection and Steganography
 *
 * This module integrates both the stylometric detection and carrier techniques into a unified toolkit.
 * It leverages the same stylometric features for both:
 * 1. Detecting if text is AI-generated (based on Kumarage et al. research)
 * 2. Embedding hidden information in text using stylometric carriers
 *
 * Flow:
 * - For detection: text → feature extraction → classification/timeline analysis
 * - For steganography: text + payload → feature analysis → carrier selection → embedding
 *
 * The module demonstrates how the same stylometric features that differentiate human from AI
 * writing can be leveraged as information carriers in steganographic applications.
 */

import {
    StyleFeatureExtractor,
    type FeatureMap
} from './stylometric_detection.genai.mjs';
import { StyleChangePointDetector } from './stylometric_change_detection.genai.mjs';
import { FusionModel as StyleClassifier } from './stylometric_fusion.genai.mjs';
import { StylometricCarrier } from './stylometric_carrier.genai.mjs';
import type { CarrierAnalysis, EncodingOptions } from './stylometric_carrier.genai.mjs';

/**
 * Interface for integrated text analysis results
 */
export interface TextAnalysisResult {
    features: FeatureMap;
    aiDetection: {
        isAiGenerated: boolean;
        probability: number;
    };
    steganographicCapacity: CarrierAnalysis;
}

/**
 * Interface for integrated timeline analysis results
 */
export interface TimelineAnalysisResult {
    changeDetected: boolean;
    changePoint: number;
    capacityDistribution: number[]; // Capacity in bits per text
}

/**
 * StylometricToolkit integrates detection and steganographic capabilities
 */
export class StylometricToolkit {
    private featureExtractor: StyleFeatureExtractor;
    private classifier: StyleClassifier;
    private changePointDetector: StyleChangePointDetector;
    private carrier: StylometricCarrier;

    /**
     * Initialize the stylometric toolkit
     */
    constructor() {
        this.featureExtractor = new StyleFeatureExtractor();
        this.classifier = new StyleClassifier();
        this.changePointDetector = new StyleChangePointDetector();
        this.carrier = new StylometricCarrier();
    }

    /**
     * Perform comprehensive analysis of text for both detection and steganographic capacity
     *
     * @param text Text to analyze
     * @returns Analysis results
     */
    analyzeText(text: string): TextAnalysisResult {
        // Extract features
        const features = this.featureExtractor.extractAllFeatures(text);

        // Perform AI detection
        this.classifier.classify(text);
        const isAiGenerated = this.classifier.getIsAIGenerated();
        const probability = this.classifier.getProbability();

        // Analyze carrying capacity
        const capacityAnalysis = this.carrier.analyzeCarryingCapacity(text);

        return {
            features,
            aiDetection: {
                isAiGenerated,
                probability
            },
            steganographicCapacity: capacityAnalysis
        };
    }

    /**
     * Analyze timeline of texts for change points and steganographic capacity
     *
     * @param timeline Array of text samples in chronological order
     * @returns Timeline analysis results
     */
    analyzeTimeline(timeline: string[]): TimelineAnalysisResult {
        // Perform change point detection
        const changeResult = this.changePointDetector.detectAuthorChange(timeline);

        // Analyze carrying capacity for each text
        const capacities = timeline.map(text => {
            const analysis = this.carrier.analyzeCarryingCapacity(text);
            return analysis.totalCapacityBits;
        });

        return {
            changeDetected: changeResult.changeDetected,
            changePoint: changeResult.changePoint,
            capacityDistribution: capacities
        };
    }

    /**
     * Embed payload in text using stylometric features as carriers
     *
     * @param text Text to use as carrier
     * @param payload Payload to embed
     * @param options Encoding options
     * @returns Modified text with embedded payload
     */
    embedPayload(
        text: string,
        payload: Uint8Array,
        options: EncodingOptions = {}
    ): string {
        // Use the carrier to embed the payload
        return this.carrier.encodePayload(text, payload, options);
    }

    /**
     * Extract payload from text using stylometric features
     *
     * @param text Text with embedded payload
     * @param options Encoding options (must match those used for embedding)
     * @returns Extracted payload
     */
    extractPayload(text: string, options: EncodingOptions = {}): Uint8Array {
        // Use the carrier to extract the payload
        return this.carrier.extractPayload(text, options);
    }

    /**
     * Convert text payload to bits
     *
     * @param payload Text to convert
     * @returns Binary representation as boolean array
     */
    textToBits(payload: string): boolean[] {
        const encoder = new TextEncoder();
        const bytes = encoder.encode(payload);

        return this.bytesToBits(bytes);
    }

    /**
     * Convert bits back to text
     *
     * @param bits Binary representation as boolean array
     * @returns Decoded text
     */
    bitsToText(bits: boolean[]): string {
        const bytes = this.bitsToBytes(bits);
        const decoder = new TextDecoder();

        return decoder.decode(bytes);
    }

    /**
     * Convert bytes to boolean array of bits
     */
    private bytesToBits(bytes: Uint8Array): boolean[] {
        const bits: boolean[] = [];
        for (let i = 0; i < bytes.length; i++) {
            const byte = bytes[i];
            for (let j = 7; j >= 0; j--) {
                bits.push(((byte >> j) & 1) === 1);
            }
        }
        return bits;
    }

    /**
     * Convert boolean array of bits to bytes
     */
    private bitsToBytes(bits: boolean[]): Uint8Array {
        // Ensure the bit count is a multiple of 8
        const padding = (8 - (bits.length % 8)) % 8;
        const paddedBits = [...bits];
        for (let i = 0; i < padding; i++) {
            paddedBits.push(false);
        }

        // Convert bits to bytes
        const bytes = new Uint8Array(paddedBits.length / 8);
        for (let i = 0; i += 8) {
            let byte = 0;
            for (let j = 0; j < 8; j++) {
                if (paddedBits[i + j]) {
                    byte |= 1 << (7 - j);
                }
            }
            bytes[i / 8] = byte;
        }

        return bytes;
    }
}

/**
 * Demonstrate the unified toolkit capabilities
 *
 * @param humanText Sample human-written text
 * @param aiText Sample AI-generated text
 * @param payloadText Text payload to embed
 */
export function demonstrateToolkit(
    humanText: string,
    aiText: string,
    payloadText: string
): void {
    console.log("=== STYLOMETRIC TOOLKIT DEMO ===");

    // Create toolkit
    const toolkit = new StylometricToolkit();

    // 1. Analyze human text
    console.log("\n1. ANALYZING HUMAN TEXT:");
    console.log("------------------------");
    const humanAnalysis = toolkit.analyzeText(humanText);
    console.log(`AI detection result: ${humanAnalysis.aiDetection.isAiGenerated ? 'AI-GENERATED' : 'HUMAN-WRITTEN'}`);
    console.log(`AI probability: ${(humanAnalysis.aiDetection.probability * 100).toFixed(2)}%`);
    console.log(`Steganographic capacity: ${humanAnalysis.steganographicCapacity.totalCapacityBits} bits`);
    console.log(`Recommended max payload: ${humanAnalysis.steganographicCapacity.recommendedMaxPayloadBytes} bytes`);
    console.log("Key features:");
    console.log(`- Lexical richness: ${humanAnalysis.features.lexical_richness.toFixed(3)}`);
    console.log(`- Readability: ${humanAnalysis.features.readability.toFixed(1)}`);

    // 2. Analyze AI text
    console.log("\n2. ANALYZING AI TEXT:");
    console.log("--------------------");
    const aiAnalysis = toolkit.analyzeText(aiText);
    console.log(`AI detection result: ${aiAnalysis.aiDetection.isAiGenerated ? 'AI-GENERATED' : 'HUMAN-WRITTEN'}`);
    console.log(`AI probability: ${(aiAnalysis.aiDetection.probability * 100).toFixed(2)}%`);
    console.log(`Steganographic capacity: ${aiAnalysis.steganographicCapacity.totalCapacityBits} bits`);
    console.log(`Recommended max payload: ${aiAnalysis.steganographicCapacity.recommendedMaxPayloadBytes} bytes`);
    console.log("Key features:");
    console.log(`- Lexical richness: ${aiAnalysis.features.lexical_richness.toFixed(3)}`);
    console.log(`- Readability: ${aiAnalysis.features.readability.toFixed(1)}`);

    // 3. Perform steganography using human text
    console.log("\n3. PERFORMING STEGANOGRAPHY:");
    console.log("---------------------------");

    // Convert payload text to binary
    const encoder = new TextEncoder();
    const payloadBytes = encoder.encode(payloadText);

    console.log(`Payload: "${payloadText}"`);
    console.log(`Payload size: ${payloadBytes.length} bytes`);

    try {
        // Embed payload into human text
        const modifiedText = toolkit.embedPayload(humanText, payloadBytes);

        console.log("\nOriginal text (first 100 chars):");
        console.log(humanText.substring(0, 100) + "...");

        console.log("\nModified text (first 100 chars):");
        console.log(modifiedText.substring(0, 100) + "...");

        // Extract payload
        const extractedBytes = toolkit.extractPayload(modifiedText);
        const decoder = new TextDecoder();
        const extractedText = decoder.decode(extractedBytes);

        console.log(`\nExtracted payload: "${extractedText}"`);
        console.log(`Recovery successful: ${extractedText === payloadText ? 'YES' : 'NO'}`);

        // Analyze modified text for AI detection
        const modifiedAnalysis = toolkit.analyzeText(modifiedText);
        console.log("\nSteganalysis of modified text:");
        console.log(`AI detection result: ${modifiedAnalysis.aiDetection.isAiGenerated ? 'AI-GENERATED' : 'HUMAN-WRITTEN'}`);
        console.log(`AI probability: ${(modifiedAnalysis.aiDetection.probability * 100).toFixed(2)}%`);

        // Compare with original analysis
        const probDiff = Math.abs(modifiedAnalysis.aiDetection.probability - humanAnalysis.aiDetection.probability) * 100;
        console.log(`Change in AI probability: ${probDiff.toFixed(2)}%`);
    } catch (error) {
        console.error(`Error in steganography demonstration: ${error.message}`);
    }

    console.log("\n=== DEMO COMPLETE ===");
}