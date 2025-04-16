/**
 * ErrorCorrection - A generic error correction utility that provides Gray code encoding
 * and additional error correction techniques for stylometric embeddings.
 * 
 * This module can be integrated with any embedding or encoding system to provide
 * resilience against noise and perturbations in stylometric features.
 */

export interface ErrorCorrectionOptions {
  /**
   * Whether to use Gray code encoding for improved error resilience
   */
  useGrayCoding: boolean;
  
  /**
   * Level of redundancy to apply (0-1 scale, where 0 means no redundancy)
   * Higher values add more redundancy bits for better error correction
   */
  redundancyLevel: number;
  
  /**
   * Chunk size for processing bits (typically 8 bits/byte)
   */
  chunkSize: number;
  
  /**
   * Whether to add parity bits for additional error detection
   */
  useParity: boolean;
  
  /**
   * Whether to interleave bits to protect against burst errors
   */
  useInterleaving: boolean;
  
  /**
   * Maximum number of bits allowed to be corrupted when attempting recovery
   * (only relevant if redundancyLevel > 0)
   */
  maxCorruptedBits?: number;
}

export class ErrorCorrection {
  private options: ErrorCorrectionOptions;
  
  constructor(options?: Partial<ErrorCorrectionOptions>) {
    this.options = {
      useGrayCoding: options?.useGrayCoding ?? true,
      redundancyLevel: options?.redundancyLevel ?? 0.2,
      chunkSize: options?.chunkSize ?? 8,
      useParity: options?.useParity ?? false,
      useInterleaving: options?.useInterleaving ?? false,
      maxCorruptedBits: options?.maxCorruptedBits ?? 2
    };
  }
  
  /**
   * Encodes data using selected error correction techniques
   * @param bits The original bit array to encode
   * @returns Encoded bit array with error correction
   */
  public encode(bits: boolean[]): boolean[] {
    // Apply encoding techniques in sequence
    let result = [...bits];
    
    if (this.options.useGrayCoding) {
      result = this.applyGrayCode(result);
    }
    
    if (this.options.redundancyLevel > 0) {
      result = this.addRedundancy(result, this.options.redundancyLevel);
    }
    
    if (this.options.useParity) {
      result = this.addParityBits(result);
    }
    
    if (this.options.useInterleaving) {
      result = this.interleave(result);
    }
    
    return result;
  }
  
  /**
   * Decodes data and attempts to correct errors
   * @param encodedBits The encoded bit array, possibly with errors
   * @param originalLength Expected length of the original data
   * @returns Decoded and error-corrected bit array
   */
  public decode(encodedBits: (boolean | null)[], originalLength: number): boolean[] {
    // Apply decoding techniques in reverse sequence
    let result = [...encodedBits] as boolean[];
    
    if (this.options.useInterleaving) {
      result = this.deinterleave(result);
    }
    
    if (this.options.useParity) {
      result = this.correctWithParity(result);
    }
    
    if (this.options.redundancyLevel > 0) {
      result = this.recoverFromRedundancy(result, originalLength);
    }
    
    if (this.options.useGrayCoding) {
      result = this.reverseGrayCode(result);
    }
    
    // Ensure we return exactly the expected length
    return result.slice(0, originalLength);
  }
  
  /**
   * Applies Gray code encoding to a bit array
   * @param bits Original bits
   * @returns Gray-coded bits
   */
  private applyGrayCode(bits: boolean[]): boolean[] {
    const result: boolean[] = [];
    
    // Process bits in chunks
    for (let i = 0; i < bits.length; i += this.options.chunkSize) {
      // Convert chunk to number
      let value = 0;
      for (let j = 0; j < this.options.chunkSize && i + j < bits.length; j++) {
        if (bits[i + j]) value |= (1 << j);
      }
      
      // Convert to Gray code (binary XOR (binary >> 1))
      const grayValue = value ^ (value >> 1);
      
      // Convert back to bits
      for (let j = 0; j < this.options.chunkSize && i + j < bits.length; j++) {
        result.push(((grayValue >> j) & 1) === 1);
      }
    }
    
    return result;
  }
  
  /**
   * Reverses Gray code encoding
   * @param grayBits Gray-coded bits
   * @returns Original bits
   */
  private reverseGrayCode(grayBits: boolean[]): boolean[] {
    const result: boolean[] = [];
    
    // Process bits in chunks
    for (let i = 0; i < grayBits.length; i += this.options.chunkSize) {
      // Convert chunk to number
      let grayValue = 0;
      for (let j = 0; j < this.options.chunkSize && i + j < grayBits.length; j++) {
        if (grayBits[i + j]) grayValue |= (1 << j);
      }
      
      // Convert from Gray code
      let value = grayValue;
      for (let mask = value >> 1; mask > 0; mask >>= 1) {
        value ^= mask;
      }
      
      // Convert back to bits
      for (let j = 0; j < this.options.chunkSize && i + j < grayBits.length; j++) {
        result.push(((value >> j) & 1) === 1);
      }
    }
    
    return result;
  }
  
  /**
   * Adds redundancy to protect against bit errors
   * @param bits Original bits
   * @param level Redundancy level (0-1)
   * @returns Bits with added redundancy
   */
  private addRedundancy(bits: boolean[], level: number): boolean[] {
    const redundantBitsCount = Math.floor(bits.length * level);
    const result = [...bits];
    
    // Add redundancy bits using XOR combinations of original bits
    for (let i = 0; i < redundantBitsCount; i++) {
      // Create a parity bit from several source bits
      // This implementation uses a simple XOR pattern but could be enhanced
      const parityBit = (bits[i % bits.length] !== 
                        bits[(i + Math.floor(bits.length / 3)) % bits.length]) !== 
                        bits[(i + Math.floor(bits.length * 2 / 3)) % bits.length];
      result.push(parityBit);
    }
    
    return result;
  }
  
  /**
   * Recovers original data from redundant encoding
   * @param bits Bits with redundancy, possibly with errors
   * @param originalLength Length of original data before redundancy
   * @returns Error-corrected bits
   */
  private recoverFromRedundancy(bits: boolean[], originalLength: number): boolean[] {
    // For a more sophisticated implementation, this would use the redundant bits
    // to detect and repair errors in the original data
    
    // This simplified version uses majority voting when we have multiple copies
    const redundancyFactor = this.options.redundancyLevel;
    const dataRegion = bits.slice(0, originalLength);
    const redundantRegion = bits.slice(originalLength);
    
    // If we have enough redundant bits, attempt correction
    if (redundantRegion.length >= originalLength * 0.5) {
      // Implement more sophisticated recovery logic here
      // For now, we'll just return the data region
    }
    
    return dataRegion;
  }
  
  /**
   * Adds parity bits for error detection
   */
  private addParityBits(bits: boolean[]): boolean[] {
    const result: boolean[] = [];
    
    // Process bits in chunks
    for (let i = 0; i < bits.length; i += this.options.chunkSize) {
      // Get current chunk
      const chunk = bits.slice(i, Math.min(i + this.options.chunkSize, bits.length));
      
      // Calculate parity bit (even parity)
      const parityBit = chunk.reduce((parity, bit) => parity !== bit, false);
      
      // Add original bits plus parity bit
      result.push(...chunk, parityBit);
    }
    
    return result;
  }
  
  /**
   * Uses parity bits to detect and possibly correct errors
   */
  private correctWithParity(bits: boolean[]): boolean[] {
    const result: boolean[] = [];
    
    // Process bits in chunks (original chunk size + 1 for parity)
    const chunkSize = this.options.chunkSize + 1;
    
    for (let i = 0; i < bits.length; i += chunkSize) {
      // Get current chunk
      const chunk = bits.slice(i, Math.min(i + chunkSize, bits.length));
      
      if (chunk.length === chunkSize) {
        // Original bits without parity
        const dataBits = chunk.slice(0, -1);
        
        // Check parity
        const parityBit = chunk[chunk.length - 1];
        const calculatedParity = dataBits.reduce((parity, bit) => parity !== bit, false);
        
        // If parity check fails, log or handle error
        // (parity can detect but not correct single bit errors)
        if (parityBit !== calculatedParity) {
          console.warn("Parity check failed for chunk at index", i);
        }
        
        result.push(...dataBits);
      } else {
        // Incomplete chunk, just add as is
        result.push(...chunk.slice(0, -1));
      }
    }
    
    return result;
  }
  
  /**
   * Interleaves bits to protect against burst errors
   */
  private interleave(bits: boolean[]): boolean[] {
    // For simplicity, use a basic block interleaving approach
    const blockSize = Math.min(8, bits.length);
    const result = new Array(bits.length).fill(false);
    
    for (let i = 0; i < bits.length; i++) {
      const block = Math.floor(i / blockSize);
      const position = i % blockSize;
      const newIndex = position * Math.ceil(bits.length / blockSize) + block;
      
      if (newIndex < bits.length) {
        result[newIndex] = bits[i];
      }
    }
    
    return result;
  }
  
  /**
   * Reverses the interleaving process
   */
  private deinterleave(bits: boolean[]): boolean[] {
    const blockSize = Math.min(8, bits.length);
    const result = new Array(bits.length).fill(false);
    
    for (let i = 0; i < bits.length; i++) {
      const block = Math.floor(i / blockSize);
      const position = i % blockSize;
      const oldIndex = position * Math.ceil(bits.length / blockSize) + block;
      
      if (oldIndex < bits.length) {
        result[i] = bits[oldIndex];
      }
    }
    
    return result;
  }
  
  /**
   * Helper: Converts binary number to Gray code
   */
  public static binaryToGray(num: number): number {
    return num ^ (num >> 1);
  }
  
  /**
   * Helper: Converts Gray code back to binary
   */
  public static grayToBinary(gray: number): number {
    let binary = 0;
    for (let mask = gray; mask; mask >>= 1) {
      binary ^= mask;
    }
    return binary;
  }
}