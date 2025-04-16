import { ErrorCorrection, ErrorCorrectionOptions } from './ErrorCorrection';

/**
 * ErrorCorrectionAdapter - Provides convenient methods for applying error correction
 * to various data types used in stylometric analysis
 */
export class ErrorCorrectionAdapter {
  private errorCorrection: ErrorCorrection;
  
  constructor(options?: Partial<ErrorCorrectionOptions>) {
    this.errorCorrection = new ErrorCorrection(options);
  }
  
  /**
   * Encodes a buffer with error correction
   */
  public encodeBuffer(buffer: Buffer): Buffer {
    const bits = this.bufferToBits(buffer);
    const encodedBits = this.errorCorrection.encode(bits);
    return this.bitsToBuffer(encodedBits);
  }
  
  /**
   * Decodes a buffer with error correction
   */
  public decodeBuffer(buffer: Buffer, originalLength: number): Buffer {
    const bits = this.bufferToBits(buffer);
    const decodedBits = this.errorCorrection.decode(bits, originalLength * 8);
    return this.bitsToBuffer(decodedBits);
  }
  
  /**
   * Encodes a numeric array with error correction
   */
  public encodeNumericArray(values: number[], bitDepth: number = 8): number[] {
    const bits = this.numericArrayToBits(values, bitDepth);
    const encodedBits = this.errorCorrection.encode(bits);
    return this.bitsToNumericArray(encodedBits, bitDepth);
  }
  
  /**
   * Decodes a numeric array with error correction
   */
  public decodeNumericArray(values: number[], originalLength: number, bitDepth: number = 8): number[] {
    const bits = this.numericArrayToBits(values, bitDepth);
    const decodedBits = this.errorCorrection.decode(bits, originalLength * bitDepth);
    return this.bitsToNumericArray(decodedBits, bitDepth);
  }
  
  /**
   * Encodes a feature vector with error correction
   */
  public encodeFeatureVector(vector: number[]): number[] {
    // Determine appropriate bit depth based on value range
    const maxValue = Math.max(...vector.map(Math.abs));
    const bitDepth = Math.max(8, Math.ceil(Math.log2(maxValue + 1)) + 1);
    
    return this.encodeNumericArray(vector, bitDepth);
  }
  
  /**
   * Converts buffer to bit array
   */
  private bufferToBits(buffer: Buffer): boolean[] {
    const bits: boolean[] = [];
    for (let i = 0; i < buffer.length; i++) {
      for (let j = 0; j < 8; j++) {
        bits.push((buffer[i] & (1 << j)) !== 0);
      }
    }
    return bits;
  }
  
  /**
   * Converts bit array to buffer
   */
  private bitsToBuffer(bits: boolean[]): Buffer {
    const bufferLength = Math.ceil(bits.length / 8);
    const buffer = Buffer.alloc(bufferLength);
    for (let i = 0; i < bits.length; i++) {
      if (bits[i]) {
        buffer[Math.floor(i / 8)] |= (1 << (i % 8));
      }
    }
    return buffer;
  }
  
  /**
   * Converts numeric array to bit array
   */
  private numericArrayToBits(values: number[], bitDepth: number): boolean[] {
    const bits: boolean[] = [];
    for (const value of values) {
      // Handle negative numbers with two's complement
      const unsignedValue = value < 0 ? ((1 << bitDepth) + value) : value;
      
      for (let j = 0; j < bitDepth; j++) {
        bits.push((unsignedValue & (1 << j)) !== 0);
      }
    }
    return bits;
  }
  
  /**
   * Converts bit array to numeric array
   */
  private bitsToNumericArray(bits: boolean[], bitDepth: number): number[] {
    const values: number[] = [];
    
    for (let i = 0; i < bits.length; i += bitDepth) {
      let value = 0;
      const remaining = Math.min(bitDepth, bits.length - i);
      
      for (let j = 0; j < remaining; j++) {
        if (bits[i + j]) {
          value |= (1 << j);
        }
      }
      
      // Handle two's complement negative numbers
      if (value & (1 << (bitDepth - 1))) {
        value = value - (1 << bitDepth);
      }
      
      values.push(value);
    }
    
    return values;
  }
}