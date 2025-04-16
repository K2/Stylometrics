/// <reference path="genaiscript.d.ts" />

/**
 * Demo Runner Script
 * Design Goals: Execute all runnable .genai.mts scripts in the workspace to demonstrate functionality and perform basic integration checks.
 * Architectural Constraints: Relies on workspace.findFiles to discover scripts. Uses runPrompt for execution. Assumes scripts handle their own errors gracefully to some extent.
 * Happy Path:
 * 1. Find all `.genai.mts` files (excluding self and system prompts).
 * 2. Iterate through each found script.
 * 3. Determine appropriate sample input based on script name convention.
 * 4. Execute the script using runPrompt.
 * 5. Log success or failure.
 * ApiNotes: ./demo_runner.ApiNotes.md
 */
script({
    title: "Demo Runner",
    description: "Runs all demoable GenAIScripts in the workspace.",
    group: "Utility",
    system: [], // No system prompts needed for the runner itself
})

// [paradigm:imperative]

// Reference: ./demo_runner.ApiNotes.md#Script Discovery
// Find all genai scripts, excluding system prompts and this runner
const scriptsToRun = await workspace.findFiles("**/*.genai.mts", {
    ignore: [
        "**/system.*.genai.mts",
        "**/demo_runner.genai.mts", // Exclude self
        "**/node_modules/**",
    ],
})

if (!scriptsToRun.length) {
    env.output.log("No scripts found to run.")
//    return // Exit if no scripts
}

env.output.log(`Found ${scriptsToRun.length} scripts to run:`)
scriptsToRun.forEach((f) => env.output.item(f.filename))

// Sample data (adjust as needed)
const sampleText = "This is a piece of text. It has multiple sentences. Stylometry can analyze it."
const sampleTimeline = [
    "This is the first text in a timeline.",
    "This is the second text, written by the same author.",
    "However, this third text seems stylistically different.",
    "And the fourth one confirms the change in authorship.",
]
const samplePayload = { some: "data" } // Keep for other scripts if needed

// Iterate and run each script
// Reference: ./demo_runner.ApiNotes.md#Execution Loop
for (const scriptFile of scriptsToRun) {
    const scriptId = path.basename(scriptFile.filename, ".genai.mts")
    env.output.log(`\n--- Running: ${scriptId} ---`)

    let options: PromptGeneratorOptions = { vars: {} }
    let run = true

    try {
        // Determine input based on script ID (add new cases)
        // Reference: ./demo_runner.ApiNotes.md#Input Determination
        if (scriptId.includes("encoder") || scriptId.includes("carrier") || scriptId === "stylometric_fusion") {
            options.vars = { inputText: sampleText, payload: samplePayload }
            env.output.log(`  Providing inputText and payload...`)
        } else if (scriptId === "stylometric_detection") {
            options.vars = { inputText: sampleText }
            env.output.log(`  Providing inputText...`)
        } else if (scriptId === "stylometry_features") { // New Python Wrapper
            options.vars = { inputText: sampleText }
            env.output.log(`  Providing inputText for Python wrapper...`)
        } else if (scriptId === "change_point_detector") { // New Python Wrapper
            options.vars = { timelineData: sampleTimeline }
            env.output.log(`  Providing timelineData for Python wrapper...`)
        } else if (scriptId === "fusion_model_predictor") { // New Python Wrapper
            options.vars = { inputText: sampleText }
            // Optionally add a default model path if you have one for testing
            // options.vars.modelPath = "/path/to/your/test_model.pth";
            env.output.log(`  Providing inputText for Python wrapper...`)
        } else if (scriptId === "safety_guard") {
             options.vars = { inputText: sampleText }
             env.output.log(`  Providing inputText...`)
        } else {
            env.output.log(`  No specific input provided.`)
        }

        // Execute the script
        // Reference: ./demo_runner.ApiNotes.md#Execution Attempt
        const result = await runPrompt(scriptId, options)

        // Log success
        // Reference: ./demo_runner.ApiNotes.md#Logging
        env.output.log(`  ✅ Success: ${scriptId}`)
        // Display result structure (text is often less useful for JSON results)
        env.output.detailsFenced("RunPrompt Result", {
             text: result.text?.slice(0, 200) + (result.text?.length > 200 ? "..." : ""), // Show snippet of text
             json: result.json, // Show JSON if present
             error: result.error,
             annotations: result.annotations,
             finishReason: result.finishReason,
             usage: result.usage
        })

    } catch (e) {
        // Log failure
        // Reference: ./demo_runner.ApiNotes.md#Logging
        console.error(`  ❌ Error running ${scriptId}: ${e.message}`)
        env.output.detailsFenced("Execution Error", e, "error")
    }
}

env.output.log("\n--- Demo Runner Finished ---")