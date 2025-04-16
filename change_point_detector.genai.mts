/// <reference path="genaiscript.d.ts" />

/**
 * GenAIScript wrapper for change_point_detection.py.
 * Design Goals: Interface with Python script for change point detection.
 * Architectural Constraints: Uses host.exec, assumes Python3 and dependencies. No plotting.
 * Happy Path: Input timeline -> Temp file -> Run python -> Parse JSON [bool, int] -> Return {detected, changePoint}.
 * ApiNotes: ./change_point_detector.ApiNotes.md
 */
script({
    title: "Python Change Point Detector",
    description: "Detects author change points in a text timeline using change_point_detection.py.",
    group: "Python Integration",
    parameters: {
        timelineData: {
            type: ["string", "array"],
            description: "An array of text strings in chronological order, or a single string with texts separated by newlines.",
        },
        agreementThreshold: {
            type: "number",
            description: "Optional agreement threshold (gamma) for detection (default 0.15).",
            default: 0.15,
        },
    },
})

// [paradigm:imperative]

// Reference: ./change_point_detector.ApiNotes.md#Core Flow
const { timelineData, agreementThreshold } = env.vars

// Validate and format input
// Reference: ./change_point_detector.ApiNotes.md#Format Input
let timelineText: string
if (Array.isArray(timelineData)) {
    timelineText = timelineData.join("\n")
} else if (typeof timelineData === "string") {
    timelineText = timelineData
} else {
    console.error("Error: timelineData parameter is missing or not a string/array.")
    throw new Error("Missing or invalid timelineData parameter.")
}

if (timelineText.trim() === "") {
     console.warn("Warning: timelineData is empty.")
     // Return early indicating no change possible in empty timeline
     def("result", { detected: false, changePoint: -1 })
     // Need to return explicitly if not relying on last expression
     // return { detected: false, changePoint: -1 }
}


// 1. Write timeline to a temporary file
// Reference: ./change_point_detector.ApiNotes.md#Temp File
const tempFile = `timeline_input_${Date.now()}.txt`
await workspace.writeText(tempFile, timelineText)
env.output.log(`Timeline data written to ${tempFile}`)

let resultJson: { detected: boolean; changePoint: number } | undefined = undefined
let executionError: SerializedError | undefined = undefined

try {
    // 2. Prepare arguments and execute the Python script
    // Reference: ./change_point_detector.ApiNotes.md#Execution
    const pythonScriptPath = `${path.dirname(env.dir)}/change_point_detection.py`
    const args = [pythonScriptPath, tempFile]
    if (typeof agreementThreshold === 'number' && agreementThreshold !== 0.15) {
        args.push("--threshold", agreementThreshold.toString())
    }

    env.output.log(`Executing: python3 ${args.join(" ")}`)
    const res = await host.exec("python3", args, {
        label: "change_point_detection.py execution",
        ignoreError: true, // Handle error manually
    })

    // 3. Process the result
    // Reference: ./change_point_detector.ApiNotes.md#Output Handling
    if (res.exitCode === 0 && res.stdout) {
        try {
            const parsedOutput = JSON.parse(res.stdout)
            // assert Array.isArray(parsedOutput) && parsedOutput.length === 2 : 'Python script should output JSON [bool, int]';
            if (Array.isArray(parsedOutput) && parsedOutput.length === 2 && typeof parsedOutput[0] === 'boolean' && typeof parsedOutput[1] === 'number') {
                 resultJson = { detected: parsedOutput[0], changePoint: parsedOutput[1] }
                 env.output.log("Python script executed successfully.")
                 env.output.detailsFenced("Detection Result", resultJson)
            } else {
                 throw new Error("Output format is not [boolean, number]")
            }
        } catch (e) {
            console.error("Error parsing JSON output from Python script.")
            console.error("stdout:", res.stdout)
            executionError = {
                name: "JSONParseError",
                message: `Failed to parse stdout from change_point_detection.py: ${e.message}`,
            }
             if (res.stderr) {
                env.output.detailsFenced("Python stderr (JSON parse error)", res.stderr, "text")
            }
        }
    } else {
        console.error(`Python script execution failed with exit code ${res.exitCode}.`)
        executionError = {
            name: "PythonExecutionError",
            message: `change_point_detection.py failed with exit code ${res.exitCode}.`,
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
    // env.output.log(`Temporary file ${tempFile} deleted.`);`);
}

// 4. Output results
// Reference: ./change_point_detector.ApiNotes.md#Return
if (executionError) {
    throw new Error(executionError.message, { cause: executionError })
}

// Define the final output structure explicitly
def("result", resultJson || { detected: false, changePoint: -1 })

// Return the defined result
resultJson