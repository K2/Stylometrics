/// <reference path="genaiscript.d.ts" />

/**
 * GenAIScript wrapper for stylometry.py feature extraction.
 * Design Goals: Interface with Python script, handle I/O via files/JSON.
 * Architectural Constraints: Uses host.exec, assumes Python3 and dependencies installed.
 * Happy Path: Input text -> Temp file -> Run python -> Parse JSON stdout -> Return features.
 * ApiNotes: ./stylometry_features.ApiNotes.md
 */

// Add type declaration if @types/compromise doesn't resolve the issue
declare module 'compromise' {
    export function plugin(compromisePlugin: any): any;
}

script({
    title: "Python Stylometry Feature Extractor",
    description: "Extracts stylometric features using the stylometry.py script.",
    group: "Python Integration",
    parameters: {
        inputText: {
            type: "string",
            description: "The text to analyze.",
        },
    },
})

// [paradigm:imperative]

// Reference: ./stylometry_features.ApiNotes.md#Core Flow
const { inputText } = env.vars

// Validate input
if (!inputText || typeof inputText !== "string") {
    console.error("Error: inputText parameter is missing or not a string.")
    throw new Error("Missing or invalid inputText parameter.")
}

// 1. Write input text to a temporary file
// Reference: ./stylometry_features.ApiNotes.md#Input Handling
const tempFile = `stylometry_input_${Date.now()}.txt`
await workspace.writeText(tempFile, inputText)
env.output.log(`Input text written to ${tempFile}`)

let resultJson: any = undefined
let executionError: SerializedError | undefined = undefined

try {
    // 2. Execute the Python script
    // Reference: ./stylometry_features.ApiNotes.md#Execution
    env.output.log(`Executing: python3 stylometry.py ${tempFile}`)
    // Ensure stylometry.py is in the same directory or adjust path
    const res = await host.exec("python3", [`${path.dirname(env.dir)}/stylometry.py`, tempFile], {
        label: "stylometry.py execution",
        ignoreError: true, // Handle error manually based on exitCode and stderr
    })

    // 3. Process the result
    // Reference: ./stylometry_features.ApiNotes.md#Output Parsing
    if (res.exitCode === 0 && res.stdout) {
        try {
            resultJson = JSON.parse(res.stdout)
            env.output.log("Python script executed successfully.")
            env.output.detailsFenced("Extracted Features (JSON)", resultJson)
        } catch (e) {
            console.error("Error parsing JSON output from Python script.")
            console.error("stdout:", res.stdout)
            executionError = {
                name: "JSONParseError",
                message: `Failed to parse stdout from stylometry.py: ${e.message}`,
            }
            if (res.stderr) {
                env.output.detailsFenced("Python stderr (JSON parse error)", res.stderr, "text")
            }
        }
    } else {
        console.error(`Python script execution failed with exit code ${res.exitCode}.`)
        executionError = {
            name: "PythonExecutionError",
            message: `stylometry.py failed with exit code ${res.exitCode}.`,
            stack: res.stderr || "No stderr output.",
        }
        if (res.stderr) {
            env.output.detailsFenced("Python stderr (execution error)", res.stderr, "text")
        } else {
             env.output.warn("Python script failed but produced no stderr.")
        }
         if (res.stdout) {
            env.output.detailsFenced("Python stdout (on error)", res.stdout, "text")
        }
    }
} catch (e) {
    // Catch errors related to host.exec itself (e.g., command not found)
    console.error(`Error executing host.exec: ${e.message}`)
    executionError = {
        name: "HostExecError",
        message: `Failed to start python3: ${e.message}`,
        stack: e.stack
    }
} finally {
    // Clean up the temporary file (optional, depends on workspace cleanup policy)
    // await workspace.deleteFile(tempFile);
    // env.output.log(`Temporary file ${tempFile} deleted.`);
}

// 4. Output results
// Reference: ./stylometry_features.ApiNotes.md#Return
if (executionError) {
    // Throwing ensures the error is propagated in runPrompt results
    throw new Error(executionError.message, { cause: executionError })
}

// Implicitly returns the last expression's value (the parsed JSON)
// For clarity, explicitly define the output structure if needed:
// return { features: resultJson }
// However, runPrompt captures stdout, fences, etc., so just logging is often sufficient.
// The result.json will be populated if resultJson is the last evaluated expression.
resultJson

/**
 * Module: Stylometry Features Extractor
 * Role: Extracts stylometric features from text for analysis and fingerprinting
 * Design Goals: Provide comprehensive feature extraction for text stylometry
 * Architectural Constraints: Should be performant with medium-to-large texts
 * Happy-path: Text input -> Feature extraction -> Feature vector output
 */

import * as nlp from 'compromise';
import assert from 'assert';

/**
 * Extracts stylometric features from text
 * @param text Input text to analyze
 * @returns Promise resolving to a record of feature names to numerical values
 */
export async function extractStylometricFeatures(text: string): Promise<Record<string, number>> {
  assert(text != null, '[extractStylometricFeatures] Input text cannot be null');
  if (!text.trim()) {
    console.warn("[extractStylometricFeatures] Called with empty text");
    return {};
  }
  
  const doc = nlp(text);
  const wordCount = doc.wordCount();
  
  if (wordCount === 0) {
    console.warn("[extractStylometricFeatures] Text has no recognizable words");
    return {};
  }
  
  // Basic lexical features
  const sentences = doc.sentences().json();
  const sentenceCount = sentences.length;
  const avgSentenceLength = sentenceCount > 0 ? wordCount / sentenceCount : 0;
  
  // Parts of speech
  const nouns = doc.nouns().json();
  const verbs = doc.verbs().json();
  const adjectives = doc.adjectives().json();
  const adverbs = doc.adverbs().json();
  
  // Punctuation features
  const commaCount = (text.match(/,/g) || []).length;
  const periodCount = (text.match(/\./g) || []).length;
  const exclamationCount = (text.match(/!/g) || []).length;
  const questionCount = (text.match(/\?/g) || []).length;
  const semicolonCount = (text.match(/;/g) || []).length;
  
  // Readability metrics approximation
  const charCount = text.length;
  const avgWordLength = wordCount > 0 ? charCount / wordCount : 0;
  
  return {
    wordCount,
    sentenceCount,
    avgSentenceLength,
    nounRatio: wordCount > 0 ? nouns.length / wordCount : 0,
    verbRatio: wordCount > 0 ? verbs.length / wordCount : 0,
    adjectiveRatio: wordCount > 0 ? adjectives.length / wordCount : 0,
    adverbRatio: wordCount > 0 ? adverbs.length / wordCount : 0,
    commaRatio: wordCount > 0 ? commaCount / wordCount : 0,
    periodRatio: wordCount > 0 ? periodCount / wordCount : 0,
    exclamationRatio: wordCount > 0 ? exclamationCount / wordCount : 0,
    questionRatio: wordCount > 0 ? questionCount / wordCount : 0,
    semicolonRatio: wordCount > 0 ? semicolonCount / wordCount : 0,
    avgWordLength,
    charToWordRatio: avgWordLength,
  };
}