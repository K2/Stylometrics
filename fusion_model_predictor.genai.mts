/// <reference path="genaiscript.d.ts" />

/**
 * GenAIScript wrapper for fusion_model.py prediction.
 * Design Goals: Interface with Python script for AI text prediction.
 * Architectural Constraints: Uses host.exec, assumes Python3, torch, etc. installed. Prediction only.
 * Happy Path: Input text -> Temp file -> Run python -> Parse JSON [probs, preds] -> Return {probabilities, predictions}.
 * ApiNotes: ./fusion_model_predictor.ApiNotes.md
 */
script({
    title: "Python Fusion Model Predictor",
    description: "Predicts if text is AI-generated using fusion_model.py.",
    group: "Python Integration",
    parameters: {
        inputText: {
            type: "string",
            description: "The text to analyze.",
        },
        modelPath: {
            type: "string",
            description: "Optional path to the pre-trained fusion model file.",
            optional: true,
        },
    },
})

// [paradigm:imperative]

// Reference: ./fusion_model_predictor.ApiNotes.md#Core Flow
const { inputText, modelPath } = env.vars

// Validate input
if (!inputText || typeof inputText !== "string") {
    console.error("Error: inputText parameter is missing or not a string.")
    throw new Error("Missing or invalid inputText parameter.")
}

// 1. Write input text to a temporary file
// Reference: ./fusion_model_predictor.ApiNotes.md#Temp File
const tempFile = `fusion_input_${Date.now()}.txt`
await workspace.writeText(tempFile, inputText)
env.output.log(`Input text written to ${tempFile}`)

let resultJson: { probabilities: number[]; predictions: number[] } | undefined = undefined
let executionError: SerializedError | undefined = undefined

try {
    // 2. Prepare arguments and execute the Python script
    // Reference: ./fusion_model_predictor.ApiNotes.md#Execution
    const pythonScriptPath = `${path.dirname(env.dir)}/fusion_model.py`
    const args = [pythonScriptPath, tempFile]
    if (modelPath && typeof modelPath === 'string') {
        args.push("--model", modelPath) // Assuming argparse uses --model
    }

    env.output.log(`Executing: python3 ${args.join(" ")}`)
    const res = await host.exec("python3", args, {
        label: "fusion_model.py prediction",
        ignoreError: true, // Handle error manually
        // Increase timeout if model loading/prediction is slow
        timeout: 120000, // 120 seconds
    })

    // 3. Process the result
    // Reference: ./fusion_model_predictor.ApiNotes.md#Output Handling
    if (res.exitCode === 0 && res.stdout) {
        try {
            const parsedOutput = JSON.parse(res.stdout)
            // assert Array.isArray(parsedOutput) && parsedOutput.length === 2 : 'Python script should output JSON [probs_array, preds_array]';
            if (Array.isArray(parsedOutput) && parsedOutput.length === 2 && Array.isArray(parsedOutput[0]) && Array.isArray(parsedOutput[1])) {
                 resultJson = { probabilities: parsedOutput[0], predictions: parsedOutput[1] }
                 env.output.log("Python script executed successfully.")
                 env.output.detailsFenced("Prediction Result", resultJson)
            } else {
                 throw new Error("Output format is not [array, array]")
            }
        } catch (e) {
            console.error("Error parsing JSON output from Python script.")
            console.error("stdout:", res.stdout)
            executionError = {
                name: "JSONParseError",
                message: `Failed to parse stdout from fusion_model.py: ${e.message}`,
            }
             if (res.stderr) {
                env.output.detailsFenced("Python stderr (JSON parse error)", res.stderr, "text")
            }
        }
    } else {
        console.error(`Python script execution failed with exit code ${res.exitCode}.`)
        executionError = {
            name: "PythonExecutionError",
            message: `fusion_model.py failed with exit code ${res.exitCode}.`,
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
    console.error(`Error executing host.exec: ${e.message}`)
    executionError = {
        name: "HostExecError",
        message: `Failed to start python3: ${e.message}`,
        stack: e.stack
    }
} finally {
    // Clean up temp file
    // await workspace.deleteFile(tempFile);
    // env.output.log(`Temporary file ${tempFile} deleted.`);
}

// 4. Output results
// Reference: ./fusion_model_predictor.ApiNotes.md#Return
if (executionError) {
    throw new Error(executionError.message, { cause: executionError })
}

// Define the final output structure explicitly as a JSON string
def("result", JSON.stringify(resultJson || { probabilities: [], predictions: [] }))

// Return the defined result
resultJson