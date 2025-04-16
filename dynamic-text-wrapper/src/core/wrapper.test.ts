// filepath: /dynamic-text-wrapper/dynamic-text-wrapper/src/core/wrapper.test.ts
/**
 * Unit tests for the dynamic text wrapping implementation.
 * This file ensures that the wrapping logic behaves as expected under various conditions,
 * including different text structures and threshold settings.
 * 
 * Design Goals:
 * - Validate the wrapping functionality for various text inputs.
 * - Ensure that non-perceptible white-space handling is accurate.
 * - Test dynamic length adjustments for end-of-line and word wrapping.
 * 
 * Happy Path:
 * 1. Import necessary modules and dependencies.
 * 2. Define test cases for different text structures (e.g., sentences, paragraphs).
 * 3. Execute wrapping functions with expected inputs.
 * 4. Assert that the output matches the expected wrapped text.
 * 
 * Example Test Structure:
 * - Test for single sentence wrapping.
 * - Test for paragraph wrapping with varying lengths.
 * - Test for edge cases (e.g., empty strings, long words).
 */

import { wrapText } from '../core/wrapper'; // Import the wrapping function
import { expect } from 'chai'; // Import assertion library

describe('Dynamic Text Wrapper', () => {
    it('should wrap a single sentence correctly', () => {
        const input = "This is a test sentence that should wrap correctly.";
        const expectedOutput = "This is a test sentence that should\nwrap correctly."; // Expected wrapped output
        const result = wrapText(input, { maxWidth: 40 }); // Call the wrap function with a max width
        expect(result).to.equal(expectedOutput); // Assert the result matches expected output
    });

    it('should handle empty strings', () => {
        const input = "";
        const expectedOutput = ""; // Expected output for empty input
        const result = wrapText(input, { maxWidth: 40 });
        expect(result).to.equal(expectedOutput);
    });

    it('should wrap long words correctly', () => {
        const input = "Supercalifragilisticexpialidocious is a long word.";
        const expectedOutput = "Supercalifragilisticexpialidocious\nis a long word."; // Expected wrapped output
        const result = wrapText(input, { maxWidth: 30 });
        expect(result).to.equal(expectedOutput);
    });

    // Additional tests can be added here for other structures and edge cases
});