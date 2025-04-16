/**
 * StyleChangePointDetector - Authorship Change Detection
 * 
 * This module specializes in detecting points where writing style changes within
 * a sequence of texts, indicating potential authorship changes.
 * 
 * Design Goals: Identify style shift boundaries in text sequences.
 * Architectural Constraints: Uses StyleFeatureExtractor for feature analysis.
 */

import { StyleFeatureExtractor } from './stylometric_detection.genai.mjs';

export interface ChangeDetectionResult {
    changeDetected: boolean;
    changePoint: number; // Index where change is detected
    confidenceScore: number; // 0-1 confidence in the detection
}

export class StyleChangePointDetector {
    private featureExtractor: StyleFeatureExtractor;

    constructor() {
        this.featureExtractor = new StyleFeatureExtractor();
    }

    /**
     * Detect if and where authorship changes in a timeline of text samples
     * 
     * @param timeline Array of text samples in chronological order
     * @returns Detection results with change point information
     */
    detectAuthorChange(timeline: string[]): ChangeDetectionResult {
        if (!timeline || timeline.length < 2) {
            return {
                changeDetected: false,
                changePoint: -1,
                confidenceScore: 0
            };
        }

        // Extract features for each text sample
        const featureTimeline = timeline.map(text => 
            this.featureExtractor.extractAllFeatures(text)
        );
        
        // Find the point with maximum feature divergence
        let maxChangePoint = -1;
        let maxDivergence = -1;
        
        for (let i = 1; i < timeline.length; i++) {
            const divergence = this.calculateFeatureDivergence(
                featureTimeline.slice(0, i),
                featureTimeline.slice(i)
            );
            
            if (divergence > maxDivergence) {
                maxDivergence = divergence;
                maxChangePoint = i;
            }
        }
        
        // Determine if the divergence indicates a significant change
        const threshold = 0.5; // Arbitrary threshold for demonstration
        const changeDetected = maxDivergence > threshold;
        
        return {
            changeDetected,
            changePoint: changeDetected ? maxChangePoint : -1,
            confidenceScore: Math.min(maxDivergence, 1.0)
        };
    }
    
    /**
     * Calculate divergence between two sets of feature vectors
     */
    private calculateFeatureDivergence(
        featuresSetA: any[],
        featuresSetB: any[]
    ): number {
        // Simple implementation for demonstration
        // In a real implementation, this would use statistical measures
        
        // Calculate average feature vectors for each set
        const avgA = this.calculateAverageFeatures(featuresSetA);
        const avgB = this.calculateAverageFeatures(featuresSetB);
        
        // Calculate Euclidean distance between averages as a simple divergence measure
        return this.featureDistance(avgA, avgB);
    }
    
    /**
     * Calculate average feature vector from a set of feature vectors
     */
    private calculateAverageFeatures(featuresSet: any[]): Record<string, number> {
        const result: Record<string, number> = {};
        const numericFeatures: Record<string, number[]> = {};
        
        // Collect all numeric features
        for (const features of featuresSet) {
            for (const [key, value] of Object.entries(features)) {
                if (typeof value === 'number') {
                    if (!numericFeatures[key]) {
                        numericFeatures[key] = [];
                    }
                    numericFeatures[key].push(value);
                }
            }
        }
        
        // Calculate averages
        for (const [key, values] of Object.entries(numericFeatures)) {
            result[key] = values.reduce((sum, val) => sum + val, 0) / values.length;
        }
        
        return result;
    }
    
    /**
     * Calculate distance between feature vectors
     */
    private featureDistance(featuresA: Record<string, number>, featuresB: Record<string, number>): number {
        let sumSquaredDiff = 0;
        let numFeatures = 0;
        
        // Calculate squared differences for all common features
        for (const key of Object.keys(featuresA)) {
            if (key in featuresB) {
                const diff = featuresA[key] - featuresB[key];
                sumSquaredDiff += diff * diff;
                numFeatures++;
            }
        }
        
        // Return normalized distance (or 0 if no common features)
        return numFeatures > 0 ? Math.sqrt(sumSquaredDiff / numFeatures) : 0;
    }
}
