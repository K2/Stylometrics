/**
 * @module examples/matrix_usage
 * @description Demonstrates basic usage of the CarrierMatrix for encoding and decoding.
 * NOTE: This uses placeholder components, so the actual embedding/extraction is simulated.
 */

import { CarrierMatrix } from '../src/matrix/CarrierMatrix';
import { Document } from '../src/types/DocumentTypes';
import { AnalysisResult } from '../src/types/AnalysisTypes';
import { EncodePayloadResult } from '../src/types/EncodingMetadata';

async function runMatrixDemo() {
    console.log("--- CarrierMatrix Demo Start ---");

    // 1. Initialize CarrierMatrix (using default placeholder carriers)
    const matrix = new CarrierMatrix({
        redundancyLevel: 0.3, // 30% redundancy
        dataShards: 5,       // Example: 5 data shards
    });

    // 2. Define Input Document and Payload
    const inputDocument: Document = {
        id: "doc1",
        content: `This is the first paragraph.\n\nIt contains some text, punctuation, and structure. We want to embed data here.\n\nThis is the second paragraph; it might offer different capacities for various carriers. Let's see how the matrix handles it. Use large words, utilize small words.`,
        metadata: { title: "Demo Document" },
    };

    const originalPayloadString = "This is the secret message we want to embed resiliently!";
    const originalPayload = new TextEncoder().encode(originalPayloadString);

    try {
        // 3. Analyze Document
        console.log("\n--- Analyzing Document ---");
        const analysisResult: AnalysisResult = await matrix.analyzeDocument(inputDocument);
        // console.log("Analysis Result:", analysisResult); // Can be verbose

        // 4. Encode Payload
        console.log("\n--- Encoding Payload ---");
        const encodeResult: EncodePayloadResult = await matrix.encodePayload(originalPayload, analysisResult);
        console.log("Encoding Metadata:", JSON.stringify(encodeResult.metadata, null, 2));
        console.log("Encoded Segments (showing differences):");
        analysisResult.segments.forEach(seg => {
            const originalContent = seg.content || "";
            const encodedContent = encodeResult.encodedSegments.get(seg.id) || "";
            if (originalContent !== encodedContent) {
                console.log(`  Segment ${seg.id}:`);
                // Basic diff visualization
                console.log(`    Original: "${originalContent.substring(0, 50)}..."`);
                console.log(`    Encoded:  "${encodedContent.substring(0, 80)}..."`); // Show more to see markers
            } else {
                 console.log(`  Segment ${seg.id}: (No changes detected)`);
            }
        });


        // --- Simulation of Transmission/Modification ---
        // For demo, we pass the encoded segments directly. In reality, these might be
        // combined, stored, transmitted, and potentially modified before decoding.
        const receivedSegments = encodeResult.encodedSegments;
        const receivedMetadata = encodeResult.metadata;

        // Optional: Simulate data loss by modifying/removing parts of receivedSegments
        // or altering receivedMetadata (though altering metadata usually breaks decoding)
        // Example: Simulate corruption of one segment's carrier data
        /*
        const segmentToCorrupt = "seg_1";
        const carrierToCorrupt = "punctuation";
        if (receivedSegments.has(segmentToCorrupt)) {
            let content = receivedSegments.get(segmentToCorrupt)!;
            content = content.replace(/\[PUNCT:[01]+\]/g, "[PUNCT:CORRUPTED]"); // Corrupt marker
            receivedSegments.set(segmentToCorrupt, content);
            console.log(`\n--- Simulated Corruption in ${segmentToCorrupt} for ${carrierToCorrupt} ---`);
        }
        */


        // 5. Decode Payload
        console.log("\n--- Decoding Payload ---");
        const recoveredPayload: Uint8Array = await matrix.decodePayload(receivedSegments, receivedMetadata);
        const recoveredPayloadString = new TextDecoder().decode(recoveredPayload);

        console.log("\n--- Results ---");
        console.log("Original Payload:", originalPayloadString);
        console.log("Recovered Payload:", recoveredPayloadString);

        // 6. Verification
        if (originalPayloadString === recoveredPayloadString) {
            console.log("\nSUCCESS: Payload recovered successfully!");
        } else {
            console.error("\nFAILURE: Recovered payload does not match original!");
            // Note: Failure is expected if corruption simulation is enabled and redundancy is insufficient.
        }

    } catch (error) {
        console.error("\n--- DEMO FAILED ---");
        console.error(error);
    } finally {
        console.log("\n--- CarrierMatrix Demo End ---");
    }
}

runMatrixDemo();
