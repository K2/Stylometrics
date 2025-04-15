/**
 * Stylometric Feature Extraction for AI Text Detection
 * 
 * This module implements stylometric feature extraction techniques based on:
 * "Stylometric Detection of AI-Generated Text in Twitter Timelines"
 * by Kumarage et al. (2023)
 * 
 * Flow:
 * Text -> Feature Extraction (Phraseology, Punctuation, Linguistic) -> Feature Vector
 * 
 * The extracted features can be used for:
 * 1. Binary classification of text authorship (human vs AI)
 * 2. Change point detection in text timelines
 * 3. Fusion with language model embeddings for enhanced detection
 */

/**
 * Interface representing a dictionary of feature values
 */
export interface FeatureMap {
    [key: string]: number;
}

/**
 * Text statistics utilities for feature extraction
 */
class TextStatistics {
    /**
     * Split text into paragraphs
     */
    static getParagraphs(text: string): string[] {
        return text.split(/\n\s*\n/).filter(p => p.trim().length > 0);
    }
    
    /**
     * Split text into sentences (simple approach)
     */
    static getSentences(text: string): string[] {
        return text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    }
    
    /**
     * Get word count for text
     */
    static getWords(text: string): string[] {
        return text.split(/\s+/).filter(w => w.trim().length > 0);
    }
    
    /**
     * Calculate mean of an array of numbers
     */
    static mean(values: number[]): number {
        if (values.length === 0) return 0;
        return values.reduce((sum, val) => sum + val, 0) / values.length;
    }
    
    /**
     * Calculate standard deviation of an array of numbers
     */
    static standardDeviation(values: number[]): number {
        if (values.length <= 1) return 0;
        const avg = TextStatistics.mean(values);
        const squareDiffs = values.map(value => Math.pow(value - avg, 2));
        const avgSquareDiff = TextStatistics.mean(squareDiffs);
        return Math.sqrt(avgSquareDiff);
    }
    
    /**
     * Count syllables in a word (approximate algorithm)
     * Based on approach described in Kincaid et al. (1975)
     */
    static countSyllables(word: string): number {
        word = word.toLowerCase().trim();
        
        // Edge cases
        if (word.length <= 3) return 1;
        
        // Remove punctuation
        word = word.replace(/[^\w]/g, '');
        
        // Count vowel groups
        const vowels = "aeiouy";
        let count = 0;
        let prevIsVowel = false;
        
        for (let i = 0; i < word.length; i++) {
            const isVowel = vowels.indexOf(word[i]) !== -1;
            if (isVowel && !prevIsVowel) {
                count++;
            }
            prevIsVowel = isVowel;
        }
        
        // Adjust for common patterns
        if (word.endsWith('e')) count--;
        if (word.endsWith('le') && word.length > 2 && vowels.indexOf(word.charAt(word.length - 3)) === -1) count++;
        if (count === 0) count = 1;
        
        return count;
    }
}

/**
 * StyleFeatureExtractor extracts stylometric features from text
 * as described in Kumarage et al. (2023)
 */
export class StyleFeatureExtractor {
    private windowSize: number;
    private specialPunct: string[];
    
    /**
     * Initialize the stylometric feature extractor
     * 
     * @param windowSize Size of the window for Moving Average Type-Token Ratio (MATTR)
     *                   as described in Covington & McFall (2010)
     */
    constructor(windowSize: number = 50) {
        this.windowSize = windowSize;
        this.specialPunct = ['!', "'", ',', ':', ';', '?', '"', '-', '–', '@', '#'];
    }
    
    /**
     * Extract all stylometric features from the given text
     * 
     * @param text Input text to analyze
     * @returns Dictionary containing all extracted stylometric features
     */
    extractAllFeatures(text: string): FeatureMap {
        const features: FeatureMap = {};
        
        // Extract each feature category
        const phraseology = this.extractPhraseologyFeatures(text);
        const punctuation = this.extractPunctuationFeatures(text);
        const linguistic = this.extractLinguisticFeatures(text);
        
        // Combine all features
        return {
            ...phraseology,
            ...punctuation,
            ...linguistic
        };
    }
    
    /**
     * Extract phraseology features that quantify how the author organizes words and phrases
     * 
     * @param text Input text to analyze
     * @returns Dictionary of phraseology features
     */
    extractPhraseologyFeatures(text: string): FeatureMap {
        const features: FeatureMap = {};
        
        // Split into paragraphs, sentences, and words
        const paragraphs = TextStatistics.getParagraphs(text);
        const sentences = TextStatistics.getSentences(text);
        const words = TextStatistics.getWords(text);
        
        // Calculate words per sentence
        const wordsPerSentence = sentences.map(s => 
            TextStatistics.getWords(s).length
        );
        
        // Calculate words per paragraph
        const wordsPerParagraph = paragraphs.map(p => 
            TextStatistics.getWords(p).length
        );
        
        // Calculate sentences per paragraph
        const sentencesPerParagraph = paragraphs.map(p => 
            TextStatistics.getSentences(p).length
        );
        
        // Store features
        features.word_count = words.length;
        features.sentence_count = sentences.length;
        features.paragraph_count = paragraphs.length;
        features.mean_words_per_sentence = TextStatistics.mean(wordsPerSentence);
        features.stdev_words_per_sentence = TextStatistics.standardDeviation(wordsPerSentence);
        features.mean_words_per_paragraph = TextStatistics.mean(wordsPerParagraph);
        features.stdev_words_per_paragraph = TextStatistics.standardDeviation(wordsPerParagraph);
        features.mean_sentences_per_paragraph = TextStatistics.mean(sentencesPerParagraph);
        features.stdev_sentences_per_paragraph = TextStatistics.standardDeviation(sentencesPerParagraph);
        
        return features;
    }
    
    /**
     * Extract punctuation features that quantify how the author uses punctuation
     * 
     * @param text Input text to analyze
     * @returns Dictionary of punctuation features
     */
    extractPunctuationFeatures(text: string): FeatureMap {
        const features: FeatureMap = {};
        
        // Count all punctuation
        const allPunct = text.match(/[^\w\s]/g) || [];
        features.total_punct_count = allPunct.length;
        
        // Count specific punctuation marks
        for (const punct of this.specialPunct) {
            // Count occurrences using split
            const count = text.split(punct).length - 1;
            features[`punct_${punct}`] = count;
        }
        
        return features;
    }
    
    /**
     * Extract linguistic diversity features
     * 
     * @param text Input text to analyze
     * @returns Dictionary of linguistic features
     */
    extractLinguisticFeatures(text: string): FeatureMap {
        const features: FeatureMap = {};
        
        // Calculate lexical richness (Moving-Average Type-Token Ratio)
        features.lexical_richness = this.calculateMATTR(text);
        
        // Calculate readability (Flesch Reading Ease)
        features.readability = this.calculateFleschReadingEase(text);
        
        return features;
    }
    
    /**
     * Calculate the Moving-Average Type-Token Ratio (MATTR) as described in
     * Covington & McFall (2010)
     * 
     * @param text Input text to analyze
     * @returns MATTR score
     */
    private calculateMATTR(text: string): number {
        const words = text.toLowerCase().split(/\s+/).filter(w => w.trim().length > 0);
        
        // If we don't have enough words for the window, use the standard TTR
        if (words.length <= this.windowSize) {
            if (!words.length) return 0;
            const uniqueWords = new Set(words);
            return uniqueWords.size / words.length;
        }
        
        // Calculate the average TTR over sliding windows
        const ttrs: number[] = [];
        for (let i = 0; i <= words.length - this.windowSize; i++) {
            const window = words.slice(i, i + this.windowSize);
            const uniqueWords = new Set(window);
            const ttr = uniqueWords.size / this.windowSize;
            ttrs.push(ttr);
        }
        
        return TextStatistics.mean(ttrs);
    }
    
    /**
     * Calculate the Flesch Reading Ease score as described in Kincaid et al. (1975)
     * 
     * @param text Input text to analyze
     * @returns Flesch Reading Ease score (0-100 scale)
     */
    private calculateFleschReadingEase(text: string): number {
        // Count sentences
        const sentences = TextStatistics.getSentences(text);
        const sentenceCount = sentences.length;
        
        // Count words
        const words = TextStatistics.getWords(text);
        const wordCount = words.length;
        
        // Count syllables
        const syllableCount = words.reduce((count, word) => 
            count + TextStatistics.countSyllables(word), 0);
        
        // Avoid division by zero
        if (sentenceCount === 0 || wordCount === 0) {
            return 0;
        }
        
        // Calculate Flesch Reading Ease score
        const score = 206.835 - 1.015 * (wordCount / sentenceCount) - 84.6 * (syllableCount / wordCount);
        
        // Clamp to 0-100 range
        return Math.max(0, Math.min(100, score));
    }
}

/**
 * Interface for timeline features analysis
 */
export interface TimelineFeatures {
    features: FeatureMap[];
    positions: number[];
}

/**
 * Implementation of Stylometric Change Point Agreement (StyloCPA) detector
 * based on Kumarage et al. (2023)
 */
export class StyloCPADetector {
    private featureExtractor: StyleFeatureExtractor;
    private agreementThreshold: number;
    
    /**
     * Initialize the StyloCPA detector
     * 
     * @param agreementThreshold Percentage of features that must agree on a change point
     *                          (γ in the paper, default 0.15)
     */
    constructor(agreementThreshold: number = 0.15) {
        this.featureExtractor = new StyleFeatureExtractor();
        this.agreementThreshold = agreementThreshold;
    }
    
    /**
     * Extract features for each text in a timeline
     * 
     * @param timeline List of text samples in chronological order
     * @returns Matrix of feature values and positions
     */
    extractTimelineFeatures(timeline: string[]): TimelineFeatures {
        const features: FeatureMap[] = [];
        const positions: number[] = [];
        
        // Extract features for each text
        for (let i = 0; i < timeline.length; i++) {
            const text = timeline[i];
            const textFeatures = this.featureExtractor.extractAllFeatures(text);
            features.push(textFeatures);
            positions.push(i);
        }
        
        return { features, positions };
    }
    
    /**
     * Detect if and where an author change occurs in a timeline of texts
     * using simplified change point detection
     * 
     * @param timeline List of text samples in chronological order
     * @returns Object with detection result and change point index
     */
    detectAuthorChange(timeline: string[]): { 
        changeDetected: boolean; 
        changePoint: number;
    } {
        // For a full implementation, this would use a time series change point detection
        // algorithm like PELT. This is a simplified version that looks for significant
        // shifts in feature values.
        
        if (timeline.length < 3) {
            return { changeDetected: false, changePoint: -1 };
        }
        
        // Extract features for the timeline
        const { features } = this.extractTimelineFeatures(timeline);
        
        // Compute feature differences between consecutive texts
        const featureNames = Object.keys(features[0]);
        const changeScores: number[] = new Array(timeline.length - 1).fill(0);
        
        for (let i = 0; i < timeline.length - 1; i++) {
            let totalDiff = 0;
            let totalFeatures = 0;
            
            for (const feature of featureNames) {
                const current = features[i][feature];
                const next = features[i+1][feature];
                
                // Skip features with zero values
                if (current === 0 && next === 0) continue;
                
                // Calculate normalized difference
                const diff = Math.abs(next - current);
                const avgValue = (Math.abs(current) + Math.abs(next)) / 2;
                
                if (avgValue > 0) {
                    const normalizedDiff = diff / avgValue;
                    totalDiff += normalizedDiff;
                    totalFeatures++;
                }
            }
            
            // Average difference across features
            changeScores[i] = totalFeatures > 0 ? totalDiff / totalFeatures : 0;
        }
        
        // Find the point with the maximum change score
        let maxChangeIdx = 0;
        let maxChangeScore = changeScores[0];
        
        for (let i = 1; i < changeScores.length; i++) {
            if (changeScores[i] > maxChangeScore) {
                maxChangeScore = changeScores[i];
                maxChangeIdx = i;
            }
        }
        
        // Determine if the change is significant (exceeds threshold)
        // A more sophisticated approach would use a statistical test
        const changeDetected = maxChangeScore > this.agreementThreshold;
        
        return {
            changeDetected,
            changePoint: changeDetected ? maxChangeIdx : -1
        };
    }
}

/**
 * Simple classifier for detecting if text is AI-generated
 * based on stylometric features
 */
export class StyleClassifier {
    private featureExtractor: StyleFeatureExtractor;
    private thresholds: FeatureMap;
    
    /**
     * Initialize the classifier
     */
    constructor() {
        this.featureExtractor = new StyleFeatureExtractor();
        
        // These thresholds would typically be learned from training data
        // These are placeholder values for demonstration
        this.thresholds = {
            lexical_richness: 0.72,  // Higher in human text
            readability: 65,         // Usually lower in AI text
            mean_words_per_sentence: 18, // AI often uses longer sentences
            stdev_words_per_sentence: 5  // Human text has more variance
        };
    }
    
    /**
     * Predict if text is AI-generated using stylometric features
     * 
     * @param text Text to analyze
     * @returns Probability that the text is AI-generated (0-1)
     */
    predict(text: string): number {
        const features = this.featureExtractor.extractAllFeatures(text);
        let aiScore = 0.5; // Starting with neutral score
        
        // Adjust based on lexical richness (higher is more human-like)
        if (features.lexical_richness < this.thresholds.lexical_richness) {
            aiScore += 0.1;
        } else {
            aiScore -= 0.1;
        }
        
        // Adjust based on readability (extremely high is suspicious)
        if (features.readability > this.thresholds.readability) {
            aiScore += 0.1;
        } else {
            aiScore -= 0.05;
        }
        
        // Adjust based on sentence length (AI often uses more uniform sentence length)
        if (features.mean_words_per_sentence > this.thresholds.mean_words_per_sentence) {
            aiScore += 0.05;
        }
        
        if (features.stdev_words_per_sentence < this.thresholds.stdev_words_per_sentence) {
            aiScore += 0.15; // Low variance is a strong indicator of AI
        } else {
            aiScore -= 0.1;
        }
        
        // Clamp to 0-1 range
        return Math.max(0, Math.min(1, aiScore));
    }
    
    /**
     * Classify text as either human or AI-written
     * 
     * @param text Text to classify
     * @param threshold Decision threshold (default 0.6)
     * @returns Classification result
     */
    classify(text: string, threshold: number = 0.6): { 
        isAiGenerated: boolean; 
        probability: number;
        features: FeatureMap;
    } {
        const probability = this.predict(text);
        const features = this.featureExtractor.extractAllFeatures(text);
        
        return {
            isAiGenerated: probability >= threshold,
            probability,
            features
        };
    }
}

/**
 * Demonstrate the stylometric detection capabilities
 * 
 * @param humanText Example of known human-written text
 * @param aiText Example of known AI-generated text
 * @param timeline Optional timeline of texts to analyze for author change
 */
export function demonstrateStylometricDetection(
    humanText: string, 
    aiText: string, 
    timeline?: string[]
): void {
    console.log("=== STYLOMETRIC DETECTION DEMO ===");
    
    // Create classifier and detector
    const classifier = new StyleClassifier();
    const changeDetector = new StyloCPADetector();
    
    // Classify human text
    console.log("\n1. ANALYZING HUMAN TEXT:");
    console.log("------------------------");
    const humanResult = classifier.classify(humanText);
    console.log(`Classification: ${humanResult.isAiGenerated ? 'AI-GENERATED' : 'HUMAN-WRITTEN'}`);
    console.log(`Probability of being AI-generated: ${(humanResult.probability * 100).toFixed(2)}%`);
    console.log("Key features:");
    console.log(`- Lexical richness: ${humanResult.features.lexical_richness.toFixed(3)}`);
    console.log(`- Readability: ${humanResult.features.readability.toFixed(1)}`);
    console.log(`- Words per sentence: ${humanResult.features.mean_words_per_sentence.toFixed(1)}`);
    
    // Classify AI text
    console.log("\n2. ANALYZING AI TEXT:");
    console.log("--------------------");
    const aiResult = classifier.classify(aiText);
    console.log(`Classification: ${aiResult.isAiGenerated ? 'AI-GENERATED' : 'HUMAN-WRITTEN'}`);
    console.log(`Probability of being AI-generated: ${(aiResult.probability * 100).toFixed(2)}%`);
    console.log("Key features:");
    console.log(`- Lexical richness: ${aiResult.features.lexical_richness.toFixed(3)}`);
    console.log(`- Readability: ${aiResult.features.readability.toFixed(1)}`);
    console.log(`- Words per sentence: ${aiResult.features.mean_words_per_sentence.toFixed(1)}`);
    
    // Timeline analysis
    if (timeline && timeline.length >= 3) {
        console.log("\n3. TIMELINE ANALYSIS:");
        console.log("-------------------");
        const changeResult = changeDetector.detectAuthorChange(timeline);
        
        if (changeResult.changeDetected) {
            console.log(`Author change detected at position ${changeResult.changePoint}`);
            console.log(`Text at position ${changeResult.changePoint}: "${timeline[changeResult.changePoint].substring(0, 50)}..."`);
            console.log(`Next text: "${timeline[changeResult.changePoint + 1].substring(0, 50)}..."`);
        } else {
            console.log("No author change detected in the timeline");
        }
    }
    
    console.log("\n=== DEMO COMPLETE ===");
}