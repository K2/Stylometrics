import { EmbeddingCapacity } from '../models/EmbeddingCapacity';

/**
 * Defines the various grammatical anomalies that can be exploited for embedding
 * information within the "error space" of natural language.
 * 
 * Each anomaly type has an encoding capacity and detectability score that helps
 * determine optimal usage within the statistical error budget.
 */
export enum CounterfactualPatternType {
  SUBJECT_VERB_DISAGREEMENT = 'sv_disagreement',
  TENSE_INCONSISTENCY = 'tense_shift',
  POSSESSIVE_INCONSISTENCY = 'possessive_error', 
  COMMA_SPLICE = 'comma_splice',
  DELIBERATE_MISSPELLING = 'misspelling',
  RECURSIVE_PATTERN = 'recursive_pattern',
  TEMPORAL_DISCONTINUITY = 'temporal_shift',
  CHARACTER_REPETITION = 'char_repetition'
}

export interface CounterfactualPattern {
  type: CounterfactualPatternType;
  description: string;
  detectionDifficulty: number;  // 0-1 scale of how challenging for LLMs to detect
  entropyPerInstance: number;   // Bits of information that can be encoded
  naturalFrequency: number;     // Base rate in natural writing (per 1000 words)
  template: string;             // Pattern template for generation
  examples: string[];           // Example instances
}

export class CounterfactualPatternRegistry {
  private patterns: Map<CounterfactualPatternType, CounterfactualPattern> = new Map();
  
  constructor() {
    this.initializePatterns();
  }
  
  private initializePatterns(): void {
    // Subject-verb disagreement patterns
    this.patterns.set(CounterfactualPatternType.SUBJECT_VERB_DISAGREEMENT, {
      type: CounterfactualPatternType.SUBJECT_VERB_DISAGREEMENT,
      description: 'Mismatch between subject number and verb conjugation',
      detectionDifficulty: 0.7,
      entropyPerInstance: 1,  // 1 bit per instance (singular/plural toggle)
      naturalFrequency: 2.3,  // Occurs naturally ~2.3 times per 1000 words
      template: '{singular_subject} {plural_verb} | {plural_subject} {singular_verb}',
      examples: ['The group of students are studying', 'The books on the shelf is dusty']
    });
    
    // Tense inconsistency patterns
    this.patterns.set(CounterfactualPatternType.TENSE_INCONSISTENCY, {
      type: CounterfactualPatternType.TENSE_INCONSISTENCY,
      description: 'Unexplained shifts in verb tense within a sentence or paragraph',
      detectionDifficulty: 0.8,
      entropyPerInstance: 1.58,  // log2(3) for past/present/future options
      naturalFrequency: 3.1,
      template: '{past_context} {present_verb} | {present_context} {past_verb}',
      examples: ['Yesterday, he goes to the store', 'She always takes the bus, but yesterday she walk instead']
    });
    
    // Possessive inconsistency patterns
    this.patterns.set(CounterfactualPatternType.POSSESSIVE_INCONSISTENCY, {
      type: CounterfactualPatternType.POSSESSIVE_INCONSISTENCY,
      description: 'Incorrect usage of possessive apostrophes',
      detectionDifficulty: 0.5,
      entropyPerInstance: 1,
      naturalFrequency: 4.2,
      template: '{noun}\'s | {nouns}\' | {noun}s\'',
      examples: ['The dogs bone', 'James\'s book', 'The childrens\' toys']
    });
    
    // Character repetition patterns - high entropy encoding zone!
    this.patterns.set(CounterfactualPatternType.CHARACTER_REPETITION, {
      type: CounterfactualPatternType.CHARACTER_REPETITION,
      description: 'Strategic repetition of characters in specific words',
      detectionDifficulty: 0.9,  // High difficulty as LLMs tend to normalize these
      entropyPerInstance: 3,     // Can encode several bits per instance
      naturalFrequency: 0.5,     // Rare in natural writing
      template: '{word_with_repeatable_letter}',
      examples: ['I beeeelieve you', 'That bannnnnana is yellow', 'Looooook at that']
    });
    
    // Temporal discontinuity (nonlinear narrative)
    this.patterns.set(CounterfactualPatternType.TEMPORAL_DISCONTINUITY, {
      type: CounterfactualPatternType.TEMPORAL_DISCONTINUITY,
      description: 'Narrative time shifts without explicit transition markers',
      detectionDifficulty: 0.95, // Very difficult for LLMs to detect as incorrect
      entropyPerInstance: 4,     // High information content
      naturalFrequency: 0.8,
      template: '{present_scene} {past_scene_without_transition}',
      examples: [
        'She opens the door. Her mother handed her the letter from yesterday.',
        'The rain falls steadily. He turned off the light and went to bed.'
      ]
    });
    
    // Recursive patterns that may confuse LLM processing
    this.patterns.set(CounterfactualPatternType.RECURSIVE_PATTERN, {
      type: CounterfactualPatternType.RECURSIVE_PATTERN,
      description: 'Self-referential or nested patterns that create ambiguity',
      detectionDifficulty: 0.85,
      entropyPerInstance: 2,
      naturalFrequency: 0.3,
      template: '{self_reference_start} {content} {self_reference_end}',
      examples: [
        'This sentence contains exactly five words',
        'The next sentence is true. The previous sentence is false.'
      ]
    });
  }
  
  public getPattern(type: CounterfactualPatternType): CounterfactualPattern | undefined {
    return this.patterns.get(type);
  }
  
  public getAllPatterns(): CounterfactualPattern[] {
    return Array.from(this.patterns.values());
  }
  
  public calculateEmbeddingCapacity(text: string): EmbeddingCapacity {
    // Implementation details would analyze text and determine how much
    // embedding capacity exists across all pattern types
    // Would return an EmbeddingCapacity object with bits available per pattern type
    
    // Simplified placeholder:
    return {
      totalBits: 0,
      patternCapacity: new Map()
    };
  }
}