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
        // Improved regex to handle various newline combinations and trim results
        return text.split(/(?:\r?\n){2,}/).map(p => p.trim()).filter(p => p.length > 0);
    }

    /**
     * Split text into sentences (improved approach using lookarounds)
     */
    static getSentences(text: string): string[] {
        // Use lookarounds to keep delimiters, split, then filter empty strings
        // Handles basic cases like Mr. Mrs. Dr. but might need more refinement for edge cases
        const sentences = text
            .replace(/([.!?])\s*(?=[A-Z"'])/g, "$1|") // Add marker after sentence end before capital letter/quote
            .split("|")
            .map(s => s.trim())
            .filter(s => s.length > 0);
        // Fallback for texts without clear sentence-ending punctuation followed by caps
        if (sentences.length <= 1 && text.length > 0) {
             return text.split(/[.!?]+/).map(s => s.trim()).filter(s => s.length > 0);
        }
        return sentences;
    }

    /**
     * Get word count for text, handling punctuation attached to words
     */
    static getWords(text: string): string[] {
        // Match sequences of word characters
        return (text.match(/\b[\w']+\b/g) || []).filter(w => w.length > 0);
    }

    /**
     * Get all characters count
     */
    static getCharacterCount(text: string): number {
        return text.length;
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
     * Note: This is a heuristic and may not be accurate for all English words.
     */
    static countSyllables(word: string): number {
        word = word.toLowerCase().trim();

        // Remove non-alphabetic characters from the end (like punctuation)
        word = word.replace(/[^a-z]+$/, '');
        // Remove non-alphabetic characters from the start
        word = word.replace(/^[^a-z]+/, '');

        // Edge cases for very short words or empty strings after cleaning
        if (word.length === 0) return 0;
        if (word.length <= 3) return 1;

        // Handle 'ia' diphthong, common 'le' ending
        if (word.endsWith('ia') && word.length > 4) word = word.substring(0, word.length - 2) + 'a'; // Treat 'ia' as one vowel sound in this heuristic
        if (word.endsWith('le') && word.length > 2 && !/[aeiouy]/.test(word.charAt(word.length - 3))) {
             // If 'le' ending preceded by consonant, add a syllable count later
             word = word.substring(0, word.length - 2);
        }

        // Remove silent 'e' at the end, but not if it's the only vowel
        if (word.endsWith('e') && word.length > 1 && /[aeiouy]/.test(word.substring(0, word.length - 1))) {
            word = word.substring(0, word.length - 1);
        }

        // Count vowel groups (sequences of vowels)
        const vowelGroups = word.match(/[aeiouy]+/g);
        let count = vowelGroups ? vowelGroups.length : 0;

        // Adjust for 'le' ending if removed earlier
        if (word.endsWith('le') && word.length > 2 && !/[aeiouy]/.test(word.charAt(word.length - 3))) {
             count++;
        }

        // Ensure at least one syllable for any word
        if (count === 0 && word.length > 0) count = 1;

        return count;
    }

    /**
     * Calculate Type-Token Ratio (TTR)
     */
    static calculateTTR(words: string[]): number {
        if (words.length === 0) return 0;
        const uniqueWords = new Set(words.map(w => w.toLowerCase()));
        return uniqueWords.size / words.length;
    }

     /**
      * Calculate Moving Average Type-Token Ratio (MATTR)
      * @param words List of words
      * @param windowSize Size of the moving window
      */
     static calculateMATTR(words: string[], windowSize: number): number {
         if (words.length === 0 || windowSize <= 0) return 0;
         if (words.length <= windowSize) return TextStatistics.calculateTTR(words);

         const ttrValues: number[] = [];
         for (let i = 0; i <= words.length - windowSize; i++) {
             const windowWords = words.slice(i, i + windowSize);
             ttrValues.push(TextStatistics.calculateTTR(windowWords));
         }
         return TextStatistics.mean(ttrValues);
     }
}

/**
 * StyleFeatureExtractor extracts stylometric features from text
 * as described in Kumarage et al. (2023)
 */
export class StyleFeatureExtractor {
    extract(text: string): import("@tensorflow/tfjs-core").Tensor1D | PromiseLike<import("@tensorflow/tfjs-core").Tensor1D> {
        throw new Error('Method not implemented.');
    }
    private windowSize: number;
    // Expanded list based on common usage and potential AI differences
    private punctuationMarks: string[];

    /**
     * Initialize the stylometric feature extractor
     *
     * @param windowSize Size of the window for Moving Average Type-Token Ratio (MATTR)
     *                   as described in Covington & McFall (2010)
     */
    constructor(windowSize: number = 50) {
        this.windowSize = windowSize;
        // Define the set of punctuation marks to count
        this.punctuationMarks = [
            ',', '.', ';', ':', '!', '?', // Standard sentence punctuation
            '"', "'", '`', // Quotes
            '(', ')', '[', ']', '{', '}', // Brackets/Parentheses
            '-', '–', '—', // Hyphens/Dashes
            '/', '\\', // Slashes
            '@', '#', '$', '%', '&', '*', // Symbols often used online
            '+', '=', '<', '>' // Other symbols
        ];
    }

    /**
     * Extract all stylometric features from the given text
     *
     * @param text Input text to analyze
     * @returns Dictionary containing all extracted stylometric features
     */
    extractAllFeatures(text: string): FeatureMap {
        const features: FeatureMap = {};

        // Basic counts needed by multiple categories
        const paragraphs = TextStatistics.getParagraphs(text);
        const sentences = TextStatistics.getSentences(text);
        const words = TextStatistics.getWords(text);
        const charCount = TextStatistics.getCharacterCount(text); // Total characters

        // Extract each feature category
        const phraseology = this.extractPhraseologyFeatures(text, paragraphs, sentences, words);
        const punctuation = this.extractPunctuationFeatures(text, charCount);
        const linguistic = this.extractLinguisticFeatures(text, words, sentences, charCount);

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
     * @param paragraphs Pre-calculated paragraphs
     * @param sentences Pre-calculated sentences
     * @param words Pre-calculated words
     * @returns Dictionary of phraseology features
     */
    extractPhraseologyFeatures(
        text: string, // Keep text param if needed for future features
        paragraphs: string[],
        sentences: string[],
        words: string[]
    ): FeatureMap {
        const features: FeatureMap = {};

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
     * @param charCount Total character count
     * @returns Dictionary of punctuation features
     */
    extractPunctuationFeatures(text: string, charCount: number): FeatureMap {
        const features: FeatureMap = {};
        let totalPunctuation = 0;

        // Count each specific punctuation mark
        this.punctuationMarks.forEach(punct => {
            const regex = new RegExp(`\\${punct}`, 'g'); // Escape the punctuation for regex
            const count = (text.match(regex) || []).length;
            features[`punct_${punct}_freq`] = charCount > 0 ? count / charCount : 0;
            totalPunctuation += count;
        });

        // Total punctuation frequency
        features.total_punctuation_freq = charCount > 0 ? totalPunctuation / charCount : 0;

        // Ratio of specific punctuation types (example: commas vs periods)
        const commaCount = (text.match(/,/g) || []).length;
        const periodCount = (text.match(/\./g) || []).length;
        features.comma_period_ratio = periodCount > 0 ? commaCount / periodCount : (commaCount > 0 ? Infinity : 0);

        // Add more complex punctuation features if needed (e.g., frequency of quotes, brackets)

        return features;
    }

    /**
     * Extract linguistic features related to vocabulary and complexity
     *
     * @param text Input text to analyze
     * @param words Pre-calculated words
     * @param sentences Pre-calculated sentences
     * @param charCount Total character count
     * @returns Dictionary of linguistic features
     */
    extractLinguisticFeatures(
        text: string, // Keep text param if needed
        words: string[],
        sentences: string[],
        charCount: number
    ): FeatureMap {
        const features: FeatureMap = {};
        const wordCount = words.length;

        // 1. Lexical Diversity
        features.ttr = TextStatistics.calculateTTR(words); // Type-Token Ratio
        features.mattr = TextStatistics.calculateMATTR(words, this.windowSize); // Moving Average TTR

        // 2. Word Length Statistics
        const wordLengths = words.map(w => w.length);
        features.mean_word_length = TextStatistics.mean(wordLengths);
        features.stdev_word_length = TextStatistics.standardDeviation(wordLengths);

        // 3. Syllable Statistics (using approximate syllable counter)
        const syllablesPerWord = words.map(w => TextStatistics.countSyllables(w));
        features.mean_syllables_per_word = TextStatistics.mean(syllablesPerWord);
        features.stdev_syllables_per_word = TextStatistics.standardDeviation(syllablesPerWord);

        // 4. Character Frequency (relative to total characters)
        features.char_freq = charCount > 0 ? wordCount / charCount : 0; // Ratio of word chars to total chars (approx)

        // 5. Readability Scores (Example: Flesch Reading Ease - requires syllable counts)
        // Flesch Reading Ease = 206.835 - 1.015 * (total words / total sentences) - 84.6 * (total syllables / total words)
        const avgWordsPerSentence = features.mean_words_per_sentence || TextStatistics.mean(sentences.map(s => TextStatistics.getWords(s).length));
        const avgSyllablesPerWord = features.mean_syllables_per_word || TextStatistics.mean(words.map(w => TextStatistics.countSyllables(w)));

        if (avgWordsPerSentence > 0 && avgSyllablesPerWord > 0) {
            features.flesch_reading_ease = 206.835 - 1.015 * avgWordsPerSentence - 84.6 * avgSyllablesPerWord;
        } else {
            features.flesch_reading_ease = 0; // Assign default if calculation isn't possible
        }

        // Add more features: function word frequency, POS tag frequencies, etc. if needed

        return features;
    }
}

// Example Usage (Optional)
/*
const extractor = new StyleFeatureExtractor();
const sampleText = `
This is the first paragraph. It has two sentences.

This is the second paragraph. It's a bit longer and uses some different words, like "utilize" and "demonstrate". It also has three sentences! Is that right? Yes.
`;
const allFeatures = extractor.extractAllFeatures(sampleText);
console.log(allFeatures);
*/