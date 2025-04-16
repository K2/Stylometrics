// ApiNotes: This example demonstrates the practical application of steganographic
// signatures for tracking generated content while maintaining plausible deniability.
// The flow is: generate content -> embed metadata -> sign -> distribute -> verify

import crypto from 'crypto';
import fs from 'fs';

// Define key types for our system
interface KeyPair {
  publicKey: string;
  privateKey: string;
  keyId: string; // Identifier for this key pair
}

// Key management system - in production this would use secure storage
export class KeyManager { // Export class for use in other modules
  private keyPairs: Map<string, KeyPair> = new Map();

  generateNewKeyPair(name: string): KeyPair {
    const { publicKey, privateKey } = crypto.generateKeyPairSync('rsa', {
      // Increased key size for better security
      modulusLength: 2048, // WAS 512
      publicKeyEncoding: { type: 'spki', format: 'pem' },
      privateKeyEncoding: { type: 'pkcs8', format: 'pem' }
    });

    const keyId = crypto.createHash('sha256').update(publicKey).digest('hex').substring(0, 8);
    const keyPair = { publicKey, privateKey, keyId };
    this.keyPairs.set(name, keyPair);
    return keyPair;
  }

  getKeyPair(name: string): KeyPair | undefined {
    return this.keyPairs.get(name);
  }

  findKeyByPublicKey(pubKey: string): { name: string, keyPair: KeyPair } | null {
    for (const [name, keyPair] of this.keyPairs.entries()) {
      if (keyPair.publicKey === pubKey) {
        return { name, keyPair };
      }
    }
    return null;
  }

  getAllPublicKeys(): Map<string, string> {
    const publicKeys = new Map<string, string>();
    for (const [name, keyPair] of this.keyPairs.entries()) {
      publicKeys.set(name, keyPair.publicKey);
    }
    return publicKeys;
  }

  // Added method to sign content with a specific identity's key
  signWithKey(name: string, content: string): string | null {
    const keyPair = this.keyPairs.get(name);
    if (!keyPair) return null;

    const signer = crypto.createSign('SHA256');
    signer.update(content);
    return signer.sign(keyPair.privateKey, 'base64');
  }

  // Added method to verify content against all known keys
  verifySignature(content: string, signature: string): string | null {
    for (const [name, keyPair] of this.keyPairs.entries()) {
      const verifier = crypto.createVerify('SHA256');
      verifier.update(content);
      if (verifier.verify(keyPair.publicKey, signature, 'base64')) {
        return name;
      }
    }
    return null;
  }
}

// Zero-width characters for steganographic encoding
const ZWSP = '\u200B'; // Zero-width space (bit 0)
const ZWNJ = '\u200C'; // Zero-width non-joiner (bit 1)
const ZWJ = '\u200D';  // Zero-width joiner (marker)

/**
 * Hides data within text using zero-width characters
 * @param text The original text to embed data within
 * @param dataToHide The string data to hide within the text
 * @returns Text with hidden data embedded using zero-width characters
 */
export const hideDataInText = (text: string, dataToHide: string): string => {
    // Convert data to binary string
    const binaryData = Array.from(dataToHide)
        .map(char => char.charCodeAt(0).toString(2).padStart(8, '0'))
        .join('');

    // Create preamble (3 ZWJ characters as start marker)
    const preamble = ZWJ.repeat(3);

    // Encode length as 16-bit binary using ZWSP/ZWNJ
    const lengthBinary = binaryData.length.toString(2).padStart(16, '0');
    // Convert length binary to zero-width characters
    const encodedLength = lengthBinary
        .split('')
        .map(bit => bit === '0' ? ZWSP : ZWNJ)
        .join('');

    // Encode actual data
    const encodedData = binaryData
        .split('')
        .map(bit => bit === '0' ? ZWSP : ZWNJ)
        .join('');

    // Create epilogue (3 ZWJ characters as end marker)
    const epilogue = ZWJ.repeat(3);

    // Insert at a pseudo-random but consistent position based on text content
    // Using modulo ensures consistency for the same text length, simple approach.
    const position = text.length > 20 ? (text.charCodeAt(5) % (text.length - 10)) + 5 : Math.min(text.length, 5);

    return text.slice(0, position) +
           preamble + encodedLength + encodedData + epilogue +
           text.slice(position);
};

/**
 * Extracts hidden data from text containing zero-width character encoding
 * @param text Text that may contain hidden data
 * @returns The extracted hidden string data, or null if none found
 */
export const extractHiddenData = (text: string): string | null => {
    const preamble = ZWJ.repeat(3);
    const epilogue = ZWJ.repeat(3);

    const preambleIndex = text.indexOf(preamble);
    if (preambleIndex === -1) return null;

    // Search for epilogue *after* the preamble
    const epilogueIndex = text.indexOf(epilogue, preambleIndex + preamble.length);
    if (epilogueIndex === -1) return null;

    const encodedContent = text.substring(preambleIndex + preamble.length, epilogueIndex);

    // Check if encoded content has at least 16 characters for length
    if (encodedContent.length < 16) return null;

    // Extract length (first 16 zero-width chars)
    const lengthBits = encodedContent.slice(0, 16)
        .split('')
        .map(char => {
            if (char === ZWSP) return '0';
            if (char === ZWNJ) return '1';
            return ''; // Invalid character
        })
        .join('');

    // Ensure lengthBits is exactly 16 bits
    if (lengthBits.length !== 16 || lengthBits.includes('')) {
        console.error("Invalid characters found in length encoding.");
        return null;
    }
    const dataLength = parseInt(lengthBits, 2);

    // Check if the remaining content length matches the expected data length
    if (encodedContent.length < 16 + dataLength) {
         console.error(`Encoded content length mismatch. Expected ${16 + dataLength}, got ${encodedContent.length}`);
         // Allow extraction even if slightly truncated, but log warning
         // return null;
    }

    // Extract data bits (up to the expected length)
    const dataBits = encodedContent.slice(16, 16 + dataLength)
        .split('')
        .map(char => {
            if (char === ZWSP) return '0';
            if (char === ZWNJ) return '1';
            return ''; // Invalid character
        })
        .join('');

    // Ensure dataBits length matches expected dataLength, handling potential truncation
     if (dataBits.length !== dataLength) {
        console.warn(`Data bits length mismatch. Expected ${dataLength}, got ${dataBits.length}. Possible truncation.`);
        if (dataBits.includes('')) {
             console.error("Invalid characters found in data encoding.");
             return null;
        }
     }


    // Convert binary to string
    let result = '';
    try {
        for (let i = 0; i < dataBits.length; i += 8) {
            const byte = dataBits.substr(i, 8);
            // Ensure we have a full byte, except possibly the last one if dataLength wasn't multiple of 8
            if (byte.length === 8 || (i + byte.length === dataBits.length && byte.length > 0)) {
                 // Pad the last byte if necessary
                 const paddedByte = byte.padEnd(8, '0');
                 result += String.fromCharCode(parseInt(paddedByte, 2));
            } else if (byte.length > 0) {
                 console.warn(`Partial byte encountered at end of data bits: ${byte}`);
            }
        }
    } catch (e) {
        console.error("Error converting bits to string:", e);
        return null; // Return null if conversion fails
    }

    // If the original data length was not a multiple of 8, the last character might be incorrect due to padding.
    // This is a limitation of simple byte conversion. More complex schemes could handle this.
    // For simplicity, we return the result as is.

    return result;
};

/**
 * Creates a cryptographic signature for a message using a private key
 * @param message The message to sign
 * @param privateKey The RSA private key in PEM format
 * @returns Base64-encoded signature
 */
const createSignature = (message: string, privateKey: string): string => {
    const signer = crypto.createSign('SHA256');
    signer.update(message);
    return signer.sign(privateKey, 'base64');
};

/**
 * Verifies a signature against a message using a public key
 * @param message The message to verify
 * @param signature The Base64-encoded signature
 * @param publicKey The RSA public key in PEM format
 * @returns Boolean indicating whether signature is valid
 */
const verifySignature = (message: string, signature: string, publicKey: string): boolean => {
    const verifier = crypto.createVerify('SHA256');
    verifier.update(message);
    return verifier.verify(publicKey, signature, 'base64');
};

/**
 * Demonstrates the steganographic signature system
 */
async function runDemo(): Promise<void> {
  console.log("STEGANOGRAPHIC SIGNATURE DEMONSTRATION (Zero-Width)");
  console.log("===================================================\n");

  // Initialize key manager and create multiple identities
  const keyManager = new KeyManager();
  const alice = keyManager.generateNewKeyPair('alice');
  const bob = keyManager.generateNewKeyPair('bob');
  const charlie = keyManager.generateNewKeyPair('charlie');

  console.log("Generated key pairs for Alice, Bob, and Charlie");
  console.log(`Alice's key ID: ${alice.keyId}`);
  console.log(`Bob's key ID: ${bob.keyId}`);
  console.log(`Charlie's key ID: ${charlie.keyId}\n`);

  // 1. Original content generation
  const originalContent = `
This is a sample document that represents a long-form content piece that would
be generated by an AI system. It might contain information that needs to be
traced back to its source for accountability or tracking purposes. The content
itself appears normal to humans, with no visible signs of embedded tracking data.

The document could be quite lengthy, spanning multiple paragraphs and covering
various topics. This would be representative of outputs in contexts like content
creation, educational materials, or technical documentation where tracking the
origin of AI-generated content is important.
  `;

  console.log("STEP 1: ORIGINAL CONTENT");
  console.log(originalContent);
  console.log("\n" + "-".repeat(80) + "\n");

  // 2. Generate metadata to embed
  const timestamp = new Date().toISOString();
  const requestId = crypto.randomBytes(4).toString('hex');
  const metadataObject = { // Keep as object for clarity
    creator: "alice",
    keyId: alice.keyId,
    timestamp,
    requestId,
    contentHash: crypto.createHash('sha256').update(originalContent).digest('hex').substring(0, 16)
  };
  const metadata = JSON.stringify(metadataObject); // Stringify for embedding

  console.log("STEP 2: METADATA TO EMBED");
  console.log(metadataObject); // Log the object
  console.log("\n" + "-".repeat(80) + "\n");

  // 3. Embed steganographic data
  const contentWithSteg = hideDataInText(originalContent, metadata);

  console.log("STEP 3: CONTENT WITH EMBEDDED STEGANOGRAPHIC DATA");
  console.log("(Note: The embedded data uses zero-width characters and is not visible in console output)");
  console.log("Length difference:", contentWithSteg.length - originalContent.length, "zero-width characters added.");
  // console.log(contentWithSteg); // Output would look identical to original
  console.log("\n" + "-".repeat(80) + "\n");

  // 4. Sign the content with Alice's private key
  const signature = createSignature(contentWithSteg, alice.privateKey);

  console.log("STEP 4: GENERATED SIGNATURE");
  console.log(signature);
  console.log("\n" + "-".repeat(80) + "\n");

  // 5. Package for distribution (in a real system, this might be stored in a database or sent to a user)
  const distributedPackage = {
    content: contentWithSteg,
    signature
  };

  console.log("STEP 5: FINAL PACKAGE FOR DISTRIBUTION");
  console.log("Content length:", distributedPackage.content.length);
  console.log("Signature length:", distributedPackage.signature.length);
  console.log("\n" + "-".repeat(80) + "\n");

  // LATER: Someone receives this content and wants to verify its origin

  console.log("VERIFICATION PROCESS");
  console.log("===================\n");

  // 6. Extract the hidden data
  const extractedData = extractHiddenData(distributedPackage.content);

  console.log("STEP 6: EXTRACTED RAW METADATA STRING");
  console.log(extractedData);
  console.log("\n" + "-".repeat(80) + "\n");

  // 7. Parse the metadata
  if (extractedData) {
    let parsedMetadata: any = null;
    try {
        parsedMetadata = JSON.parse(extractedData);
        console.log("STEP 7: PARSED METADATA");
        console.log(parsedMetadata);
    } catch (e) {
        console.error("STEP 7: FAILED TO PARSE EXTRACTED METADATA AS JSON");
        console.error(e);
    }
    console.log("\n" + "-".repeat(80) + "\n");

    // 8. Verify the signature against all known public keys
    console.log("STEP 8: SIGNATURE VERIFICATION");
    let foundSigner: string | null = null; // Store the verified signer name

    for (const [name, keyPair] of keyManager.keyPairs.entries()) { // Use keyPairs map directly
      const isValid = verifySignature(
        distributedPackage.content,
        distributedPackage.signature,
        keyPair.publicKey // Use publicKey from keyPair
      );

      console.log(`Checking signature against ${name}'s public key: ${isValid ? "VALID ✓" : "invalid ✗"}`);

      if (isValid) {
        foundSigner = name; // Store the name if valid
        // No need to break, check all keys to detect potential conflicts (though unlikely)
      }
    }

    if (foundSigner) {
        console.log(`\nContent signature verified. Signed by: ${foundSigner}`);
        if (parsedMetadata) {
            console.log(`Claimed creator in metadata: ${parsedMetadata.creator}`);
            console.log(`Key ID in metadata: ${parsedMetadata.keyId}`);

            // Verify if the claimed identity matches the actual signer
            if (foundSigner === parsedMetadata.creator) {
              console.log("✓ Identity verification successful: Claimed creator matches actual signer");
              // Optional: Verify keyId matches the signer's keyId
              const signerKeyPair = keyManager.getKeyPair(foundSigner);
              if (signerKeyPair && signerKeyPair.keyId === parsedMetadata.keyId) {
                  console.log("✓ Key ID verification successful: Metadata key ID matches signer's key ID");
              } else {
                  console.warn("⚠ Key ID mismatch: Metadata key ID does not match signer's key ID!");
              }
            } else {
              console.warn("⚠ Identity mismatch: Claimed creator does not match actual signer!");
            }
        } else {
            console.warn("Cannot verify identity details because metadata parsing failed.");
        }
    } else {
      console.log("\nSignature verification failed: Could not verify signature with any known public key.");
    }
  } else {
    console.log("No hidden data found in the content!");
  }
}

// Run the demonstration if executed directly
if (require.main === module) {
    runDemo().catch(console.error);
}

