import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { 
    StylometricFingerprinter, 
    generateFingerprintFromFeatures, 
    __setFeatureExtractor, 
    __restoreFeatureExtractor 
} from './stylometric_fingerprinter.mts';
import { StylometricCarrier } from './stylometric_carrier.genai.mts';

describe('StylometricFingerprinter', () => {
    let fingerprinter: StylometricFingerprinter;
    let mockCarrier: StylometricCarrier;
    const mockFeatures = { feature1: 0.5, feature2: 1.2, another: 0.8888 };
    let expectedFingerprint: string;

    beforeEach(async () => {
        vi.mock('./stylometry_features.genai.mts', async (importOriginal) => {
            const original = await importOriginal() as any;
            return {
                ...original,
                extractStylometricFeatures: vi.fn().mockResolvedValue(mockFeatures)
            };
        });
        __setFeatureExtractor(vi.fn().mockResolvedValue(mockFeatures));

        mockCarrier = new StylometricCarrier();
        mockCarrier.encodePayload = vi.fn().mockImplementation(async (text, payload, options) => {
            const fingerprint = Buffer.from(payload).toString('utf-8');
            return `${text}<!--fp:${fingerprint}-->`;
        });
        mockCarrier.extractPayload = vi.fn().mockImplementation(async (text, options) => {
            const match = text.match(/<!--fp:(.*?)-->/);
            if (match && match[1]) {
                return Buffer.from(match[1], 'utf-8');
            }
            return Buffer.from('');
        });
        mockCarrier.getAvailableCarriers = vi.fn().mockReturnValue([
            { id: 'mock_punct', name: 'Mock Punctuation', category: 'punctuation', detectability: 0.2, estimate: () => 100, apply: vi.fn(), extract: vi.fn() },
            { id: 'mock_ling', name: 'Mock Linguistic', category: 'linguistic', detectability: 0.3, estimate: () => 100, apply: vi.fn(), extract: vi.fn() }
        ]);

        fingerprinter = new StylometricFingerprinter({ fingerprintLength: 16 });
        (fingerprinter as any).carrier = mockCarrier;

        expectedFingerprint = await generateFingerprintFromFeatures(mockFeatures, 16);
    });

    afterEach(() => {
        vi.restoreAllMocks();
        __restoreFeatureExtractor();
    });

    // All the test cases from the original file
    it('should generate a fingerprint from features (expected success)', async () => {
        const features = { rate: 0.1, count: 15 };
        const fp = await generateFingerprintFromFeatures(features, 16);
        expect(fp).toHaveLength(16);
        expect(fp).toMatch(/^[a-f0-9]{16}$/);

        const fp2 = await generateFingerprintFromFeatures(features, 16);
        expect(fp).toEqual(fp2);

        const fp_diff_len = await generateFingerprintFromFeatures(features, 8);
        expect(fp_diff_len).toHaveLength(8);
        expect(fp.startsWith(fp_diff_len)).toBe(true);
    });

    // ...rest of the test cases...
});
