import { ErrorCorrectionAdapter } from '../utils/ErrorCorrectionAdapter';

/**
 * Wrapper to apply error correction to any stylometric features
 */
export class ErrorCorrectedStylometric {
  private adapter: ErrorCorrectionAdapter;
  
  constructor(options?: { redundancyLevel?: number, useGrayCoding?: boolean }) {
    this.adapter = new ErrorCorrectionAdapter({
      redundancyLevel: options?.redundancyLevel ?? 0.2,
      useGrayCoding: options?.useGrayCoding ?? true
    });
  }
  
  /**
   * Applies error correction to n-gram frequency features
   */
  public encodeNGramFeatures(features: number[]): number[] {
    return this.adapter.encodeFeatureVector(features);
  }
  
  /**
   * Recovers n-gram frequency features with error correction
   */
  public decodeNGramFeatures(encodedFeatures: number[], originalLength: number): number[] {
    return this.adapter.decodeNumericArray(encodedFeatures, originalLength);
  }
  
  /**
   * Encodes text embeddings with error correction
   */
  public encodeEmbedding(embedding: Float32Array | number[]): number[] {
    const array = Array.from(embedding);
    // Scale and quantize to integers for error correction
    const scaledIntegers = array.map(x => Math.round(x * 1000));
    return this.adapter.encodeFeatureVector(scaledIntegers);
  }
  
  /**
   * Decodes text embeddings with error correction
   */
  public decodeEmbedding(encodedEmbedding: number[], originalLength: number): Float32Array {
    const decodedIntegers = this.adapter.decodeNumericArray(encodedEmbedding, originalLength);
    // Convert back to floating point
    return new Float32Array(decodedIntegers.map(x => x / 1000));
  }
  
  /**
   * Applies error correction to counterfactual embeddings
   */
  public applyToCounterfactualEmbedding(data: Buffer): Buffer {
    return this.adapter.encodeBuffer(data);
  }
  
  /**
   * Recovers counterfactual embeddings with error correction
   */
  public recoverFromCounterfactualEmbedding(encodedData: Buffer, originalLength: number): Buffer {
    return this.adapter.decodeBuffer(encodedData, originalLength);
  }
}