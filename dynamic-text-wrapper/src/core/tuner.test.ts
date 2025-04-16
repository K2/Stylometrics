// filepath: /dynamic-text-wrapper/dynamic-text-wrapper/src/core/tuner.test.ts
/**
 * Unit Tests for Threshold Tuning Implementation
 * This file contains tests for the tuning logic that adjusts the dynamic text wrapping behavior
 * across various text structures such as word pairs, sentence fragments, sentences, paragraphs,
 * sections, chapters, and volumes.
 * 
 * The tests ensure that the tuning logic behaves as expected under different configurations and
 * thresholds, validating the integration with the wrapping logic.
 */

import { tuneThreshold } from './tuner'; // Import the tuning function
import { assert } from 'chai'; // Import assertion library

describe('Threshold Tuner', () => {
    it('should correctly tune threshold for word pairs', () => {
        const result = tuneThreshold('word pair', { type: 'wordPair', threshold: 5 });
        assert.isTrue(result.success, 'Threshold tuning for word pairs failed');
        assert.equal(result.newThreshold, 5, 'Expected threshold did not match');
    });

    it('should correctly tune threshold for sentence fragments', () => {
        const result = tuneThreshold('This is a fragment.', { type: 'sentenceFragment', threshold: 10 });
        assert.isTrue(result.success, 'Threshold tuning for sentence fragments failed');
        assert.equal(result.newThreshold, 10, 'Expected threshold did not match');
    });

    it('should correctly tune threshold for full sentences', () => {
        const result = tuneThreshold('This is a complete sentence.', { type: 'sentence', threshold: 15 });
        assert.isTrue(result.success, 'Threshold tuning for sentences failed');
        assert.equal(result.newThreshold, 15, 'Expected threshold did not match');
    });

    it('should correctly tune threshold for paragraphs', () => {
        const result = tuneThreshold('This is a paragraph that contains multiple sentences.', { type: 'paragraph', threshold: 20 });
        assert.isTrue(result.success, 'Threshold tuning for paragraphs failed');
        assert.equal(result.newThreshold, 20, 'Expected threshold did not match');
    });

    it('should correctly tune threshold for sections', () => {
        const result = tuneThreshold('This is a section with several paragraphs.', { type: 'section', threshold: 25 });
        assert.isTrue(result.success, 'Threshold tuning for sections failed');
        assert.equal(result.newThreshold, 25, 'Expected threshold did not match');
    });

    it('should correctly tune threshold for chapters', () => {
        const result = tuneThreshold('This is a chapter containing multiple sections.', { type: 'chapter', threshold: 30 });
        assert.isTrue(result.success, 'Threshold tuning for chapters failed');
        assert.equal(result.newThreshold, 30, 'Expected threshold did not match');
    });

    it('should correctly tune threshold for volumes', () => {
        const result = tuneThreshold('This is a volume with several chapters.', { type: 'volume', threshold: 35 });
        assert.isTrue(result.success, 'Threshold tuning for volumes failed');
        assert.equal(result.newThreshold, 35, 'Expected threshold did not match');
    });
});