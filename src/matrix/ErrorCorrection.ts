/**
 * ErrorCorrection.ts
 * 
 * This file implements Reed-Solomon error correction for the carrier matrix system.
 * Reed-Solomon is particularly well-suited for steganographic applications as it
 * provides robust protection against both random and burst errors.
 * 
 * Design Goals:
 * - Provide customizable redundancy levels for different risk profiles
 * - Support incremental recovery when partial data is available
 * - Enable reconstruction of data from incomplete segment sets
 * - Efficiently encode/decode with minimal computational overhead
 * 
 * Happy-path flow:
 * 1. Original data → split into shards
 * 2. Generate parity shards based on redundancy level
 * 3. Distribute all shards across carriers
 * 4. On extraction, validate and reconstruct if needed
 */

/**
 * Configuration for error correction
 */
export interface ErrorCorrectionConfig {
  /** Number of data shards (pieces of original data) */
  dataShards: number;
  
  /** Number of parity shards (for error correction) */
  parityShards: number;
  
  /** Shard size in bytes (default: 32) */
  shardSize?: number;
}

/**
 * Implementation of Reed-Solomon error correction
 */
export class ReedSolomon {
  private readonly dataShards: number;
  private readonly parityShards: number;
  private readonly totalShards: number;
  private readonly shardSize: number;
  private readonly gfLog: Uint8Array;
  private readonly gfExp: Uint8Array;

  /**
   * Create a new Reed-Solomon error correction instance
   */
  constructor(config: ErrorCorrectionConfig) {
    this.dataShards = config.dataShards;
    this.parityShards = config.parityShards;
    this.totalShards = this.dataShards + this.parityShards;
    this.shardSize = config.shardSize || 32;
    
    // Initialize Galois Field tables for GF(2^8)
    this.gfLog = new Uint8Array(256);
    this.gfExp = new Uint8Array(256);
    this.initGaloisTables();
    
    // Validate configuration
    if (this.dataShards <= 0 || this.parityShards <= 0) {
      throw new Error('Invalid Reed-Solomon configuration: shards must be positive');
    }
    
    if (this.totalShards > 255) {
      throw new Error('Invalid Reed-Solomon configuration: total shards must be <= 255');
    }
  }

  /**
   * Encode data with parity information
   * @param data Original data to encode
   * @returns Data + parity shards
   */
  encode(data: Uint8Array): Uint8Array[] {
    // Prepare data shards
    const dataShards = this.splitIntoShards(data);
    const shards = [...dataShards];
    
    // Create empty parity shards
    for (let i = 0; i < this.parityShards; i++) {
      shards.push(new Uint8Array(this.shardSize));
    }
    
    // Generate parity data
    this.generateParityShards(shards);
    
    return shards;
  }

  /**
   * Attempt to reconstruct original data from potentially corrupted shards
   * @param shards Array of data+parity shards, with null for missing shards
   * @param shardPresent Boolean array indicating which shards are present
   * @returns Reconstructed original data or null if reconstruction failed
   */
  decode(shards: (Uint8Array | null)[], shardPresent: boolean[]): Uint8Array | null {
    // Count available shards
    const availableShards = shardPresent.filter(p => p).length;
    
    // Cannot reconstruct if we have fewer shards than data shards
    if (availableShards < this.dataShards) {
      return null;
    }
    
    // If all data shards are present, no reconstruction needed
    if (shardPresent.slice(0, this.dataShards).every(p => p)) {
      return this.joinShards(shards.slice(0, this.dataShards) as Uint8Array[]);
    }
    
    // Reconstruct missing shards
    const reconstructed = this.reconstructShards([...shards], [...shardPresent]);
    if (!reconstructed) {
      return null;
    }
    
    // Combine data shards to get original data
    return this.joinShards(reconstructed.slice(0, this.dataShards));
  }

  /**
   * Split data into fixed-size shards
   */
  private splitIntoShards(data: Uint8Array): Uint8Array[] {
    const shards: Uint8Array[] = [];
    const paddedSize = Math.ceil(data.length / this.dataShards / this.shardSize) * this.shardSize;
    const paddedData = new Uint8Array(this.dataShards * paddedSize);
    
    // Copy original data and pad with zeros
    paddedData.set(data);
    
    // Split into equal shards
    for (let i = 0; i < this.dataShards; i++) {
      const start = i * paddedSize;
      const shard = paddedData.slice(start, start + paddedSize);
      shards.push(shard);
    }
    
    return shards;
  }

  /**
   * Join data shards back into original data
   */
  private joinShards(dataShards: Uint8Array[]): Uint8Array {
    // Calculate total data length
    const shardSize = dataShards[0].length;
    const result = new Uint8Array(this.dataShards * shardSize);
    
    // Copy all data shards to result buffer
    for (let i = 0; i < dataShards.length; i++) {
      result.set(dataShards[i], i * shardSize);
    }
    
    return result;
  }

  /**
   * Generate parity shards using Reed-Solomon algorithm
   * Modifies the shards array in place
   */
  private generateParityShards(shards: Uint8Array[]): void {
    // Generate encoding matrix
    const matrix = this.buildEncodingMatrix();
    
    // For each byte in the shard
    for (let byteIndex = 0; byteIndex < this.shardSize; byteIndex++) {
      // For each parity row
      for (let outputRow = 0; outputRow < this.parityShards; outputRow++) {
        let value = 0;
        // For each data shard
        for (let inputCol = 0; inputCol < this.dataShards; inputCol++) {
          value ^= this.gfMultiply(
            matrix[this.dataShards + outputRow][inputCol],
            shards[inputCol][byteIndex]
          );
        }
        shards[this.dataShards + outputRow][byteIndex] = value;
      }
    }
  }

  /**
   * Attempt to reconstruct missing shards
   * @returns Array of reconstructed shards or null if insufficient data
   */
  private reconstructShards(shards: (Uint8Array | null)[], present: boolean[]): Uint8Array[] | null {
    // Count present shards
    const presentCount = present.filter(p => p).length;
    if (presentCount < this.dataShards) {
      return null; // Not enough shards to reconstruct
    }
    
    // Create decoder matrix
    const decoderMatrix = this.buildDecoderMatrix(present);
    if (!decoderMatrix) {
      return null; // Matrix inversion failed
    }
    
    // Initialize result shards
    const result: Uint8Array[] = [];
    for (let i = 0; i < this.totalShards; i++) {
      result.push(new Uint8Array(this.shardSize));
    }
    
    // Copy available shards
    for (let i = 0; i < this.totalShards; i++) {
      if (present[i] && shards[i]) {
        result[i] = shards[i]!;
      }
    }
    
    // Reconstruct missing shards
    this.decodeMissingShards(result, decoderMatrix, present);
    
    return result;
  }

  /**
   * Reconstruct missing shards using the decoder matrix
   */
  private decodeMissingShards(shards: Uint8Array[], decoderMatrix: number[][], present: boolean[]): void {
    // For each missing shard
    for (let outputRow = 0; outputRow < this.dataShards; outputRow++) {
      if (!present[outputRow]) {
        // For each byte in the shard
        for (let byteIndex = 0; byteIndex < this.shardSize; byteIndex++) {
          let value = 0;
          // Use matrix to recalculate missing values
          for (let inputCol = 0; inputCol < this.dataShards; inputCol++) {
            const inputRow = this.presentToRow(present, inputCol);
            value ^= this.gfMultiply(
              decoderMatrix[outputRow][inputCol],
              shards[inputRow][byteIndex]
            );
          }
          shards[outputRow][byteIndex] = value;
        }
      }
    }
  }

  /**
   * Map from present shard index to row index
   */
  private presentToRow(present: boolean[], index: number): number {
    let count = 0;
    for (let i = 0; i < this.totalShards; i++) {
      if (present[i]) {
        if (count === index) {
          return i;
        }
        count++;
      }
    }
    throw new Error('Invalid index');
  }

  /**
   * Build Vandermonde encoding matrix
   */
  private buildEncodingMatrix(): number[][] {
    // Create matrix
    const matrix: number[][] = [];
    for (let i = 0; i < this.totalShards; i++) {
      matrix[i] = new Array(this.dataShards).fill(0);
    }
    
    // Identity matrix for data shards
    for (let i = 0; i < this.dataShards; i++) {
      matrix[i][i] = 1;
    }
    
    // Vandermonde matrix for parity shards
    for (let i = 0; i < this.parityShards; i++) {
      for (let j = 0; j < this.dataShards; j++) {
        matrix[this.dataShards + i][j] = this.gfExp[((i + 1) * j) % 255];
      }
    }
    
    return matrix;
  }

  /**
   * Build decoder matrix from available shards
   * @returns Decoder matrix or null if matrix inversion failed
   */
  private buildDecoderMatrix(present: boolean[]): number[][] | null {
    // Extract submatrix of encoding matrix for available shards
    const encodingMatrix = this.buildEncodingMatrix();
    const subMatrix: number[][] = [];
    
    // Identify available shards
    let subMatrixRow = 0;
    for (let i = 0; i < this.totalShards; i++) {
      if (present[i]) {
        subMatrix[subMatrixRow] = encodingMatrix[i].slice();
        subMatrixRow++;
        
        if (subMatrixRow >= this.dataShards) {
          break; // We have enough shards
        }
      }
    }
    
    // Invert the matrix
    return this.invertMatrix(subMatrix);
  }

  /**
   * Invert a matrix in the Galois Field
   * @returns Inverted matrix or null if not invertible
   */
  private invertMatrix(matrix: number[][]): number[][] | null {
    const size = matrix.length;
    const inverse: number[][] = [];
    
    // Initialize inverse as identity matrix
    for (let i = 0; i < size; i++) {
      inverse[i] = new Array(size).fill(0);
      inverse[i][i] = 1;
    }
    
    // Copy matrix to avoid modifying original
    const working: number[][] = [];
    for (let i = 0; i < size; i++) {
      working[i] = matrix[i].slice();
    }
    
    // Gaussian elimination
    // Forward elimination
    for (let i = 0; i < size; i++) {
      // Find pivot
      let pivotRow = i;
      for (let j = i + 1; j < size; j++) {
        if (working[j][i] > working[pivotRow][i]) {
          pivotRow = j;
        }
      }
      
      if (working[pivotRow][i] === 0) {
        return null; // Matrix is not invertible
      }
      
      // Swap rows if needed
      if (pivotRow !== i) {
        [working[i], working[pivotRow]] = [working[pivotRow], working[i]];
        [inverse[i], inverse[pivotRow]] = [inverse[pivotRow], inverse[i]];
      }
      
      // Scale the pivot row
      const pivot = working[i][i];
      const pivotInverse = this.gfInverse(pivot);
      
      for (let j = 0; j < size; j++) {
        working[i][j] = this.gfMultiply(working[i][j], pivotInverse);
        inverse[i][j] = this.gfMultiply(inverse[i][j], pivotInverse);
      }
      
      // Eliminate other rows
      for (let j = 0; j < size; j++) {
        if (j !== i) {
          const factor = working[j][i];
          for (let k = 0; k < size; k++) {
            working[j][k] ^= this.gfMultiply(factor, working[i][k]);
            inverse[j][k] ^= this.gfMultiply(factor, inverse[i][k]);
          }
        }
      }
    }
    
    return inverse;
  }

  /**
   * Initialize Galois Field tables for GF(2^8)
   */
  private initGaloisTables(): void {
    let x = 1;
    for (let i = 0; i < 255; i++) {
      this.gfExp[i] = x;
      this.gfLog[x] = i;
      
      // x = x * 2 (multiply by primitive element)
      x = (x << 1) ^ (x & 0x80 ? 0x1D : 0); // 0x1D is the irreducible polynomial
    }
    
    // Special case for log(0) = 0
    this.gfExp[255] = this.gfExp[0];
    this.gfLog[0] = 0;
  }

  /**
   * Multiply two numbers in the Galois Field
   */
  private gfMultiply(a: number, b: number): number {
    if (a === 0 || b === 0) return 0;
    return this.gfExp[(this.gfLog[a] + this.gfLog[b]) % 255];
  }

  /**
   * Calculate the inverse of a number in the Galois Field
   */
  private gfInverse(x: number): number {
    if (x === 0) throw new Error('Cannot invert zero');
    return this.gfExp[(255 - this.gfLog[x]) % 255];
  }
}