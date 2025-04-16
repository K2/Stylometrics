/**
 * Steganographic Encoding Demo (Multilayer: Zero-Width + Stylometric)
 *
 * This module demonstrates the combined use of zero-width and simplified stylometric
 * steganography techniques to provide redundant encoding.
 *
 * Flow: Original Content → Encode Zero-Width → Encode Stylometric → Sign → Distribute
 */

import crypto from 'crypto';
// Import the canonical KeyManager
import { KeyManager } from './safety_embedded_word.genai.mts'; // Updated import
import { hideDataInText, extractHiddenData } from './safety_embedded_word.genai.mts';
import { hideDataStylometrically, extractHiddenStylometricData } from './safety_stylometric_encoder.genai.mts';

/**
 * Result of encoded content package
 */
export interface EncodedPackage { // Export interface
  content: string;
  signature: string;
}

/**
 * Result of decoded content
 */
export interface DecodedResult { // Export interface
  metadata: any;
  signerName: string | null;
  extractionSource?: 'stylometric' | 'zero-width' | 'none'; // Added source info
}

/**
 * Encode content with multiple layers of steganography (Zero-Width then Stylometric)
 * @param originalContent Original text content
 * @param metadata Metadata to embed
 * @param keyManager Key manager for signing
 * @param signerName Identity to sign with
 * @returns Encoded package or null if encoding fails
 */
export const encodeMultilayerContent = (
  originalContent: string,
  metadata: any,
  keyManager: KeyManager,
  signerName: string
): EncodedPackage | null => {
  try {
    // Convert metadata to JSON string
    const metadataStr = JSON.stringify(metadata);

    // Layer 1: Hide metadata using zero-width characters
    const contentWithZeroWidth = hideDataInText(originalContent, metadataStr);
    console.log(`Applied zero-width encoding. Length: ${contentWithZeroWidth.length}`);


    // Layer 2: Apply stylometric encoding (simplified version) with the same metadata
    // This provides redundancy in case one encoding is stripped
    const contentWithBothEncodings = hideDataStylometrically(contentWithZeroWidth, metadataStr);
    console.log(`Applied stylometric encoding. Length: ${contentWithBothEncodings.length}`);


    // Sign the final content
    const signature = keyManager.signWithKey(signerName, contentWithBothEncodings);
    if (!signature) {
      console.error(`Failed to sign content: key for ${signerName} not found`);
      return null;
    }

    return {
      content: contentWithBothEncodings,
      signature
    };
  } catch (error) {
    console.error("Error encoding multilayer content:", error);
    return null;
  }
};

/**
 * Decode and verify content with multiple steganographic layers
 * Attempts Stylometric first, then Zero-Width as fallback.
 * @param encodedContent Content to decode
 * @param signature Signature to verify
 * @param keyManager Key manager for verification
 * @returns Decoded result or null if decoding fails
 */
export const decodeMultilayerContent = (
  encodedContent: string,
  signature: string,
  keyManager: KeyManager
): DecodedResult | null => {
  try {
    // Verify signature first
    const signerName = keyManager.verifySignature(encodedContent, signature);
    if (!signerName) {
      // Don't log error here, let the caller decide. Return indication of failure.
      console.warn("Could not verify content signature with any known key.");
      return { metadata: null, signerName: null, extractionSource: 'none' };
    }
    console.log(`Signature verified. Signed by: ${signerName}`);


    // Try both extraction methods
    let extractedMetadata = null;
    let extractionSource: DecodedResult['extractionSource'] = 'none';

    // Try stylometric extraction first (as it might be more robust to some changes)
    const stylometricData = extractHiddenStylometricData(encodedContent);
    if (stylometricData) {
      try {
        extractedMetadata = JSON.parse(stylometricData);
        extractionSource = 'stylometric';
        console.log("✓ Extracted metadata using stylometric method");
      } catch (e) {
        console.warn("⚠ Stylometric data found but was not valid JSON. Falling back...");
        console.warn("  Stylometric raw data:", stylometricData.substring(0, 100) + "..."); // Log snippet
      }
    } else {
         console.log("No valid stylometric data found.");
    }

    // If stylometric failed or data was invalid JSON, try zero-width extraction
    if (!extractedMetadata) {
      console.log("Attempting zero-width extraction...");
      const zeroWidthData = extractHiddenData(encodedContent);
      if (zeroWidthData) {
        try {
          extractedMetadata = JSON.parse(zeroWidthData);
          extractionSource = 'zero-width';
          console.log("✓ Extracted metadata using zero-width method");
        } catch (e) {
          console.warn("⚠ Zero-width data found but was not valid JSON.");
           console.warn("  Zero-width raw data:", zeroWidthData.substring(0, 100) + "..."); // Log snippet
        }
      } else {
           console.log("No valid zero-width data found.");
      }
    }

    if (!extractedMetadata) {
      console.error("Could not extract metadata using any available method.");
      // Return signer name even if metadata extraction fails, as signature was valid
      return { metadata: null, signerName, extractionSource: 'none' };
    }

    return {
      metadata: extractedMetadata,
      signerName,
      extractionSource
    };
  } catch (error) {
    console.error("Error decoding multilayer content:", error);
    return null; // Indicate failure
  }
};

/**
 * Demo function to show multilayer encoding and decoding
 */
export const runMultilayerDemo = (): void => {
  console.log("\nMULTILAYER STEGANOGRAPHIC DEMO (Zero-Width + Stylometric)");
  console.log("=========================================================\n");

  // Setup key manager and identities
  const keyManager = new KeyManager();
  const alice = keyManager.generateNewKeyPair("alice"); // Get keypair info
  keyManager.generateNewKeyPair("bob");
  console.log(`Generated keys for alice (ID: ${alice.keyId}) and bob.`);


  // Original content
  const originalContent = `
This is an example of a document that contains important information.
It spans multiple paragraphs to demonstrate how the stylometric encoding works.

The information in this document appears normal to human readers while containing
hidden metadata that can be extracted programmatically.

This approach provides plausible deniability while enabling content tracking and
verification through cryptographic signatures. It needs enough sentences for the stylometric part.
Let's add another sentence here just in case. And one more for good measure.
  `;

  // Metadata to embed
  const metadata = {
    creator: "alice",
    keyId: alice.keyId, // Include keyId in metadata
    timestamp: new Date().toISOString(),
    contentId: crypto.randomBytes(4).toString('hex'),
    classification: "demonstration",
    version: "1.0"
  };

  console.log("Original content length:", originalContent.length);
  console.log("Metadata to hide:", metadata);

  // Encode content
  console.log("\nENCODING CONTENT");
  console.log("================");
  const encodedPackage = encodeMultilayerContent(originalContent, metadata, keyManager, "alice");
  if (!encodedPackage) {
    console.error("FATAL: Failed to encode content");
    return;
  }

  console.log("\nEncoded content length:", encodedPackage.content.length);
  console.log("Signature:", encodedPackage.signature.substring(0, 60) + "..."); // Show partial signature

  // Decode and verify
  console.log("\nDECODING CONTENT");
  console.log("================");

  const decodedResult = decodeMultilayerContent(encodedPackage.content, encodedPackage.signature, keyManager);
  if (!decodedResult) {
    console.error("FATAL: Failed to decode content");
    return;
  }

  console.log("\n--- Verification Results ---");
  console.log("Verified signer:", decodedResult.signerName || "!! VERIFICATION FAILED !!");
  console.log("Extraction source:", decodedResult.extractionSource);
  console.log("Extracted metadata:", decodedResult.metadata);

  // Validate metadata integrity
  console.log("\n--- Metadata Integrity Check ---");
  if (decodedResult.metadata && decodedResult.signerName === "alice") {
    let matches = true;
    for (const [key, value] of Object.entries(metadata)) {
      if (decodedResult.metadata[key] !== value) {
        matches = false;
        console.log(`  ✗ ${key}: MISMATCH (expected ${value}, got ${decodedResult.metadata[key]})`);
      } else {
         console.log(`  ✓ ${key}: Match`);
      }
    }
    console.log(matches ? "\n✓ All metadata values match the original data" : "\n✗ Metadata mismatch detected!");
  } else if (decodedResult.signerName !== "alice") {
      console.log("✗ Metadata check skipped: Signer verification failed or metadata missing.");
  } else {
       console.log("✗ Metadata check skipped: Metadata extraction failed.");
  }
   console.log("=========================================================");
};

// Auto-run the demonstration if this file is executed directly
if (require.main === module) {
  runMultilayerDemo();
}