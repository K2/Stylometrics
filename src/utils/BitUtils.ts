/**
 * @module src/utils/BitUtils
 * @description Provides utility functions for converting between byte arrays (Uint8Array)
 * and boolean arrays (representing bits) or bit strings.
 */

/**
 * Converts a Uint8Array to an array of booleans representing bits (MSB first).
 * @param bytes - The input byte array.
 * @returns An array of booleans (true for 1, false for 0).
 */
export function bytesToBits(bytes: Uint8Array): boolean[] {
    const bits: boolean[] = [];
    bytes.forEach(byte => {
        for (let i = 7; i >= 0; i--) {
            bits.push(((byte >> i) & 1) === 1);
        }
    });
    return bits;
}

/**
 * Converts an array of booleans (representing bits) to a Uint8Array.
 * Pads with false (0) if the bit array length is not a multiple of 8.
 * @param bits - The input array of booleans (true for 1, false for 0).
 * @returns A Uint8Array containing the corresponding bytes.
 */
export function bitsToBytes(bits: boolean[]): Uint8Array {
    const byteLength = Math.ceil(bits.length / 8);
    const bytes = new Uint8Array(byteLength);
    for (let i = 0; i < byteLength; i++) {
        let byte = 0;
        for (let j = 0; j < 8; j++) {
            const bitIndex = i * 8 + j;
            if (bitIndex < bits.length && bits[bitIndex]) {
                byte |= (1 << (7 - j));
            }
        }
        bytes[i] = byte;
    }
    return bytes;
}

/**
 * Converts a Uint8Array to a binary string ('0' and '1').
 * @param bytes - The input byte array.
 * @returns A string of '0's and '1's.
 */
export function bytesToBitString(bytes: Uint8Array): string {
    let bitString = "";
    bytes.forEach(byte => {
        bitString += byte.toString(2).padStart(8, '0');
    });
    return bitString;
}

/**
 * Converts a binary string ('0' and '1') to a Uint8Array.
 * Pads with '0' if the string length is not a multiple of 8.
 * @param bitString - The input string of '0's and '1's.
 * @returns A Uint8Array.
 */
export function bitStringToBytes(bitString: string): Uint8Array {
    const byteLength = Math.ceil(bitString.length / 8);
    const bytes = new Uint8Array(byteLength);
    const paddedString = bitString.padEnd(byteLength * 8, '0'); // Pad end

    for (let i = 0; i < byteLength; i++) {
        const byteString = paddedString.substring(i * 8, (i + 1) * 8);
        bytes[i] = parseInt(byteString, 2);
    }
    return bytes;
}