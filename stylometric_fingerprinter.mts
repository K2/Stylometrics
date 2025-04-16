import assert from 'assert';
import crypto from 'crypto';
import { StylometricCarrier, type EncodingOptions as CarrierEncodingOptions } from './stylometric_carrier.genai.mts';
import { extractStylometricFeatures as actualExtractStylometricFeatures } from './stylometry_features.genai.mts';

type FeatureExtractor = (text: string) => Promise<Record<string, number>>;

let extractStylometricFeatures: FeatureExtractor = actualExtractStylometricFeatures;

export function __setFeatureExtractor(extractor: FeatureExtractor) {
    extractStylometricFeatures = extractor;
}

export function __restoreFeatureExtractor() {
    extractStylometricFeatures = actualExtractStylometricFeatures;
}

export interface FingerprinterOptions {
    carrierOptions?: CarrierEncodingOptions;
    fingerprintCarriers?: string[];
    fingerprintLength?: number;
}

export async function generateFingerprintFromFeatures(features: Record<string, number>, length: number = 16): Promise<string> {
    assert(features != null, '[generateFingerprintFromFeatures] Features cannot be null.');
    if (Object.keys(features).length === 0) {
        console.warn("[generateFingerprintFromFeatures] No features provided, returning empty fingerprint.");
        return "";
    }
    const featureItems = Object.entries(features)
        .map(([key, value]) => `${key}:${value.toFixed(3)}`)
        .sort();
    const featureString = featureItems.join('|');
    const hash = crypto.createHash('sha256').update(featureString).digest('hex');
    return hash.substring(0, Math.max(4, length));
}

async function applyFingerprint(text: string, fingerprint: string, carrier: StylometricCarrier, carrierOptions: CarrierEncodingOptions): Promise<string> {
    assert(text != null, '[applyFingerprint] Text cannot be null.');
    assert(fingerprint != null, '[applyFingerprint] Fingerprint cannot be null.');
    assert(carrier != null, '[applyFingerprint] Carrier cannot be null.');
    console.log(`Applying fingerprint "${fingerprint}"...`);

    if (fingerprint.length === 0) {
        console.warn("[applyFingerprint] Fingerprint is empty, returning original text.");
        return text;
    }

    const payload = Buffer.from(fingerprint, 'utf-8');
    const payloadBits = payload.length * 8;

    const modifiedText = await carrier.encodePayload(text, payload, carrierOptions);

    const extractedPayload = await carrier.extractPayload(modifiedText, carrierOptions);
    const extractedFingerprint = Buffer.from(extractedPayload).toString('utf-8');

    if (!extractedFingerprint.startsWith(fingerprint)) {
        console.warn(`[applyFingerprint] Verification failed! Fingerprint encoding likely incomplete or corrupted. Expected "${fingerprint}", extracted "${extractedFingerprint}".`);
        return text;
    } else if (extractedFingerprint.length < fingerprint.length) {
        console.warn(`[applyFingerprint] Verification warning: Extracted fingerprint is shorter than original. Expected ${fingerprint.length} chars, got ${extractedFingerprint.length}.`);
    } else {
        console.log(`[applyFingerprint] Successfully encoded and verified ${payloadBits} bits.`);
    }

    return modifiedText;
}

export class StylometricFingerprinter {
    private carrier: StylometricCarrier;
    private options: Required<FingerprinterOptions>;
    private carrierEncodingOptions: CarrierEncodingOptions;

    constructor(options: FingerprinterOptions = {}) {
        this.carrier = new StylometricCarrier();

        const defaultOptions: Required<FingerprinterOptions> = {
            carrierOptions: {
                usePhraseologyCarriers: false,
                usePunctuationCarriers: true,
                useLinguisticCarriers: true,
                useReadabilityCarriers: false,
                maxDetectionRisk: 0.5,
            },
            fingerprintCarriers: [],
            fingerprintLength: 16,
        };

        this.options = { ...defaultOptions, ...options };
        if (options.carrierOptions) {
            this.options.carrierOptions = { ...defaultOptions.carrierOptions, ...options.carrierOptions };
        }

        this.carrierEncodingOptions = { ...this.options.carrierOptions };
        if (this.options.fingerprintCarriers && this.options.fingerprintCarriers.length > 0) {
            const allCarriers = this.carrier.getAvailableCarriers();
            this.carrierEncodingOptions.usePhraseologyCarriers = allCarriers.some(c => c.category === 'phraseology' && this.options.fingerprintCarriers.includes(c.id));
            this.carrierEncodingOptions.usePunctuationCarriers = allCarriers.some(c => c.category === 'punctuation' && this.options.fingerprintCarriers.includes(c.id));
            this.carrierEncodingOptions.useLinguisticCarriers = allCarriers.some(c => c.category === 'linguistic' && this.options.fingerprintCarriers.includes(c.id));
            this.carrierEncodingOptions.useReadabilityCarriers = allCarriers.some(c => c.category === 'readability' && this.options.fingerprintCarriers.includes(c.id));
            console.warn("StylometricFingerprinter: Filtering by specific fingerprintCarriers IDs is not fully supported by the current StylometricCarrier. Using category flags based on specified IDs.");
        }

        const activeCarriers = this.carrier.getAvailableCarriers().filter(carrier => {
            if (carrier.category === 'phraseology' && !this.carrierEncodingOptions.usePhraseologyCarriers) return false;
            if (carrier.category === 'punctuation' && !this.carrierEncodingOptions.usePunctuationCarriers) return false;
            if (carrier.category === 'linguistic' && !this.carrierEncodingOptions.useLinguisticCarriers) return false;
            if (carrier.category === 'readability' && !this.carrierEncodingOptions.useReadabilityCarriers) return false;
            if (carrier.detectability > this.carrierEncodingOptions.maxDetectionRisk) return false;
            return true;
        });
        console.log(`StylometricFingerprinter initialized. Fingerprint length: ${this.options.fingerprintLength}. Active carriers for fingerprinting: ${activeCarriers.map(c => c.id).join(', ') || 'None'}`);
    }

    async extractFeatures(text: string): Promise<Record<string, number>> {
        assert(text != null, '[extractFeatures] Input text cannot be null.');
        if (!text.trim()) {
            console.warn("[extractFeatures] Called with empty or whitespace-only text.");
            return {};
        }
        try {
            console.log("[extractFeatures] Calling feature extractor...");
            const features = await extractStylometricFeatures(text);
            assert(features != null, '[extractFeatures] Feature extraction returned null/undefined.');
            console.log(`[extractFeatures] Extracted ${Object.keys(features).length} features.`);
            return features;
        } catch (error: any) {
            console.error("[extractFeatures] Error during feature extraction:", error.message || error);
            return {};
        }
    }

    async addFingerprint(originalText: string, fingerprintData?: string): Promise<string> {
        assert(originalText != null, '[addFingerprint] Original text must not be null.');

        let fingerprintToEmbed: string;
        if (fingerprintData) {
            fingerprintToEmbed = fingerprintData.substring(0, this.options.fingerprintLength);
            if (fingerprintData.length !== this.options.fingerprintLength) {
                console.warn(`[addFingerprint] Provided fingerprint length (${fingerprintData.length}) differs from configured length (${this.options.fingerprintLength}). Using truncated/padded fingerprint: ${fingerprintToEmbed}`);
            } else {
                console.log(`[addFingerprint] Using provided fingerprint data: ${fingerprintToEmbed}`);
            }
        } else {
            console.log("[addFingerprint] Extracting features to generate fingerprint...");
            const features = await this.extractFeatures(originalText);
            if (Object.keys(features).length === 0) {
                console.error("[addFingerprint] Failed to extract features. Cannot generate or apply fingerprint.");
                return originalText;
            }
            fingerprintToEmbed = await generateFingerprintFromFeatures(features, this.options.fingerprintLength);
            console.log(`[addFingerprint] Generated fingerprint from features: ${fingerprintToEmbed}`);
        }

        assert(fingerprintToEmbed != null, '[addFingerprint] Fingerprint to embed is null.');
        if (fingerprintToEmbed.length === 0) {
            console.warn("[addFingerprint] Generated or provided fingerprint is empty. Skipping embedding.");
            return originalText;
        }

        return applyFingerprint(originalText, fingerprintToEmbed, this.carrier, this.carrierEncodingOptions);
    }

    async extractFingerprint(modifiedText: string): Promise<string | null> {
        assert(modifiedText != null, '[extractFingerprint] Modified text must not be null.');

        const activeCarriers = this.carrier.getAvailableCarriers().filter(carrier => {
            if (carrier.category === 'phraseology' && !this.carrierEncodingOptions.usePhraseologyCarriers) return false;
            if (carrier.category === 'punctuation' && !this.carrierEncodingOptions.usePunctuationCarriers) return false;
            if (carrier.category === 'linguistic' && !this.carrierEncodingOptions.useLinguisticCarriers) return false;
            if (carrier.category === 'readability' && !this.carrierEncodingOptions.useReadabilityCarriers) return false;
            if (carrier.detectability > this.carrierEncodingOptions.maxDetectionRisk) return false;
            return true;
        });

        console.log(`[extractFingerprint] Attempting to extract fingerprint using ${activeCarriers.map(c => c.id).join(', ') || 'None'} carriers...`);

        const extractedPayload = await this.carrier.extractPayload(modifiedText, this.carrierEncodingOptions);

        if (extractedPayload && extractedPayload.length > 0) {
            const potentialFingerprint = Buffer.from(extractedPayload).toString('utf-8');
            const fingerprint = potentialFingerprint.substring(0, this.options.fingerprintLength);

            const fingerprintRegex = new RegExp(`^[a-f0-9]{${this.options.fingerprintLength}}$`);
            if (fingerprintRegex.test(fingerprint)) {
                console.log(`[extractFingerprint] Extracted potential fingerprint: ${fingerprint}`);
                return fingerprint;
            } else {
                console.warn(`[extractFingerprint] Extracted payload start "${fingerprint}" does not match expected format/length (${this.options.fingerprintLength}). Discarding.`);
                console.warn(`   (Full extracted payload: "${potentialFingerprint.substring(0, 50)}...")`);
                return null;
            }
        } else {
            console.log("[extractFingerprint] No fingerprint payload found.");
            return null;
        }
    }

    async verifyFingerprint(text: string): Promise<{ match: boolean; extracted?: string | null; generated?: string }> {
        assert(text != null, '[verifyFingerprint] Input text cannot be null.');
        console.log("[verifyFingerprint] Verifying fingerprint...");
        const extracted = await this.extractFingerprint(text);

        console.log("[verifyFingerprint] Re-extracting features to generate current fingerprint...");
        const features = await this.extractFeatures(text);
        if (Object.keys(features).length === 0) {
            console.error("[verifyFingerprint] Failed to extract features from current text. Cannot verify.");
            return { match: false, extracted: extracted ?? null };
        }
        const generated = await generateFingerprintFromFeatures(features, this.options.fingerprintLength);
        console.log(`[verifyFingerprint] Generated fingerprint from current features: ${generated}`);

        const match = extracted === generated && extracted !== null;
        console.log(`[verifyFingerprint] Verification result: ${match ? 'Match' : 'No Match'}`);
        return { match, extracted: extracted ?? null, generated };
    }
}