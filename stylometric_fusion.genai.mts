// filepath: /home/files/git/Stylometrics/stylometric_fusion.genai.mts
/**
 * Module: Stylometric Fusion Model
 * Role: Combines stylometric features and semantic embeddings for robust authorship/style detection.
 * Design Goals: Create a TF.js model that fuses different feature types for prediction.
 * Architectural Constraints: Requires a concrete EmbeddingModel implementation (external dependency). Relies on StyleFeatureExtractor. See ./stylometric_fusion.ApiNotes.md.
 * Happy-path: Instantiate with feature extractor and embedding model -> train (optional) -> predict -> get style prediction.
 */
import * as tf from '@tensorflow/tfjs-node'; // Use tfjs-node for backend flexibility
import assert from 'assert';
// Assuming StyleFeatureExtractor is correctly implemented elsewhere
// We need a mockable way to import this for testing
import { StyleFeatureExtractor as ActualStyleFeatureExtractor } from './stylometric_detection.genai.mts';

// Define a type for the feature extractor class for easier mocking
type StyleFeatureExtractorType = typeof ActualStyleFeatureExtractor;

// Use a variable that can be reassigned in tests
let StyleFeatureExtractor: StyleFeatureExtractorType = ActualStyleFeatureExtractor;

// Allow tests to replace the implementation
export function __setStyleFeatureExtractor(extractor: StyleFeatureExtractorType) {
    StyleFeatureExtractor = extractor;
}
export function __restoreStyleFeatureExtractor() {
    StyleFeatureExtractor = ActualStyleFeatureExtractor;
}

// --- Interfaces ---
// Reference: ./stylometric_fusion.ApiNotes.md#Interfaces

/**
 * Interface for any model capable of generating text embeddings.
 * This needs to be implemented externally (e.g., using @tensorflow-models/universal-sentence-encoder,
 * or calling an API like OpenAI Embeddings, or loading a local HuggingFace model via transformers.js).
 */
export interface EmbeddingModel {
    /**
     * Generates an embedding for the given text.
     * @param text The input text.
     * @returns A TensorFlow Tensor1D representing the embedding.
     */
    embed(text: string): Promise<tf.Tensor1D>;

    /**
     * Returns the dimensionality of the embeddings produced by this model.
     */
    getEmbeddingDimension(): number;

    /**
     * Releases any resources held by the model.
     */
    dispose(): void;
}

export interface FusionModelConfig {
    learningRate?: number;
    epochs?: number;
    batchSize?: number;
    validationSplit?: number; // Added for training config
    earlyStoppingPatience?: number; // Added for training config
    // Add other relevant hyperparameters
}

// --- Concrete Implementation Required ---
// The user MUST provide a class that implements EmbeddingModel.
// Example structure using a hypothetical library:
/*
import { SomeEmbeddingLibrary } from 'some-embedding-library'; // Hypothetical

export class MyEmbeddingModel implements EmbeddingModel {
    private model: SomeEmbeddingLibrary.Model; // Replace with actual model type
    private dimension: number;
    private isDisposed: boolean = false;

    constructor(modelPathOrUrl: string) {
        // Load the actual model (this is specific to the chosen library)
        console.log(`[MyEmbeddingModel] Loading embedding model from ${modelPathOrUrl}...`);
        // this.model = await SomeEmbeddingLibrary.load(modelPathOrUrl); // Example async loading
        // this.dimension = this.model.getOutputShape()[1]; // Example: Get dimension
        this.dimension = 512; // Placeholder - SET ACTUAL DIMENSION
        assert(this.dimension > 0, "[MyEmbeddingModel] Embedding dimension must be positive.");
        console.log(`[MyEmbeddingModel] Model loaded. Dimension: ${this.dimension}`);
    }

    async embed(text: string): Promise<tf.Tensor1D> {
        assert(!this.isDisposed, '[MyEmbeddingModel] Model has been disposed.');
        assert(text != null, '[MyEmbeddingModel] Input text cannot be null.');
        // Use the loaded model to generate embeddings
        // const embeddingsArray = await this.model.embed([text]); // Example API call
        // const tensor = tf.tidy(() => tf.tensor1d(embeddingsArray[0]));
        // return tensor;

        // Placeholder implementation - REPLACE WITH ACTUAL EMBEDDING LOGIC
        console.warn("[MyEmbeddingModel] embed() using placeholder implementation!");
        return tf.tidy(() => tf.randomNormal([this.dimension]));
    }

    getEmbeddingDimension(): number {
        return this.dimension;
    }

    dispose(): void {
        if (!this.isDisposed) {
            console.log("[MyEmbeddingModel] Disposing embedding model...");
            // Add actual disposal logic if needed (e.g., this.model.dispose())
            // if (this.model && typeof this.model.dispose === 'function') {
            //     this.model.dispose();
            // }
            this.isDisposed = true;
        }
    }
}
*/

// --- Fusion Model ---
// [paradigm:functional] - TF.js operations
// [paradigm:imperative] - Model training loop
export class FusionModel {
    classify(text: string) {
        throw new Error('Method not implemented.');
    }
    private featureExtractor: InstanceType<StyleFeatureExtractorType>; // Use instance type
    private embeddingModel: EmbeddingModel; // Use the interface
    private config: Required<FusionModelConfig>;
    private model: tf.Sequential | null = null;
    private featureDimension: number = 0; // Will be set after first feature extraction
    private embeddingDimension: number = 0;
    private isDisposed: boolean = false;

    // Reference: ./stylometric_fusion.ApiNotes.md#Initialization
    constructor(
        featureExtractor: InstanceType<StyleFeatureExtractorType>, // Inject the concrete implementation instance
        embeddingModel: EmbeddingModel, // Inject the concrete implementation instance
        config: FusionModelConfig = {}
    ) {
        assert(featureExtractor != null, '[FusionModel] StyleFeatureExtractor cannot be null.');
        assert(embeddingModel != null, '[FusionModel] EmbeddingModel cannot be null.');

        this.featureExtractor = featureExtractor;
        this.embeddingModel = embeddingModel;
        this.embeddingDimension = this.embeddingModel.getEmbeddingDimension();
        assert(this.embeddingDimension > 0, '[FusionModel] Embedding dimension must be positive.');

        // Define default config including new training parameters
        const defaultConfig: Required<FusionModelConfig> = {
            learningRate: 0.001,
            epochs: 10,
            batchSize: 32,
            validationSplit: 0.2,
            earlyStoppingPatience: 3,
        };

        this.config = { ...defaultConfig, ...config };

        console.log(`[FusionModel] Initialized with Embedding Dim: ${this.embeddingDimension}, Config:`, this.config);
    }

    // [paradigm:functional]
    private async preprocessInput(text: string): Promise<tf.Tensor1D | null> {
        assert(!this.isDisposed, '[FusionModel] Model has been disposed.');
        // Reference: ./stylometric_fusion.ApiNotes.md#Preprocessing
        assert(text != null, '[FusionModel.preprocessInput] Input text cannot be null.');
        let featuresTensor: tf.Tensor1D | null = null;
        let embeddingTensor: tf.Tensor1D | null = null;
        try {
            // Assume featureExtractor.extract returns Tensor1D
            featuresTensor = await this.featureExtractor.extract(text);
            embeddingTensor = await this.embeddingModel.embed(text);

            if (!featuresTensor || !embeddingTensor) {
                throw new Error("Feature or embedding tensor generation failed.");
            }

            if (!this.featureDimension) {
                this.featureDimension = featuresTensor.shape[0];
                assert(this.featureDimension > 0, '[FusionModel] Feature dimension must be positive.');
                console.log(`[FusionModel] Determined Feature Dimension: ${this.featureDimension}`);
            } else {
                assert(featuresTensor.shape[0] === this.featureDimension, `[FusionModel] Feature dimension mismatch. Expected ${this.featureDimension}, got ${featuresTensor.shape[0]}`);
            }
            assert(embeddingTensor.shape[0] === this.embeddingDimension, `[FusionModel] Embedding dimension mismatch. Expected ${this.embeddingDimension}, got ${embeddingTensor.shape[0]}`);

            // Concatenate features and embeddings within tidy to manage memory
            const combined = tf.tidy(() => {
                // Ensure tensors are valid before concat
                if (!featuresTensor || !embeddingTensor) {
                    throw new Error("Invalid tensors for concatenation.");
                }
                return tf.concat([featuresTensor!, embeddingTensor!]);
            });

            // Dispose intermediate tensors outside tidy if they were created outside
            featuresTensor.dispose();
            embeddingTensor.dispose();

            return combined;
        } catch (error: any) {
            console.error("[FusionModel.preprocessInput] Error during preprocessing:", error.message || error);
            // Ensure disposal even on error
            if (featuresTensor) featuresTensor.dispose();
            if (embeddingTensor) embeddingTensor.dispose();
            return null;
        }
    }

    // [paradigm:imperative]
    private buildModel(inputDim: number): tf.Sequential {
        assert(!this.isDisposed, '[FusionModel] Model has been disposed.');
        // Reference: ./stylometric_fusion.ApiNotes.md#ModelArchitecture
        assert(inputDim > 0, '[FusionModel.buildModel] Input dimension must be positive.');
        console.log(`[FusionModel] Building model with Input Dimension: ${inputDim}`);
        const model = tf.sequential();
        // Simple dense layers for fusion - adjust architecture as needed
        model.add(tf.layers.dense({ inputShape: [inputDim], units: 64, activation: 'relu', name: 'dense_1' }));
        model.add(tf.layers.dropout({ rate: 0.3, name: 'dropout_1' }));
        model.add(tf.layers.dense({ units: 32, activation: 'relu', name: 'dense_2' }));
        // Output layer - adjust units based on number of classes (e.g., authors, styles)
        // Assuming binary classification for simplicity
        model.add(tf.layers.dense({ units: 1, activation: 'sigmoid', name: 'output' }));

        model.compile({
            optimizer: tf.train.adam(this.config.learningRate),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy'],
        });
        console.log("[FusionModel] Model built and compiled.");
        model.summary();
        return model;
    }

    // [paradigm:imperative]
    async train(texts: string[], labels: number[]): Promise<tf.History> {
        assert(!this.isDisposed, '[FusionModel] Model has been disposed.');
        // Reference: ./stylometric_fusion.ApiNotes.md#Training
        assert(texts && texts.length > 0, '[FusionModel.train] Input texts array cannot be null or empty.');
        assert(labels && labels.length > 0, '[FusionModel.train] Input labels array cannot be null or empty.');
        assert(labels.length === texts.length, '[FusionModel.train] Labels array must match texts array length.');
        console.log(`[FusionModel] Starting training with ${texts.length} samples...`);

        // Preprocess all texts and filter out failures
        const processedData = await Promise.all(texts.map(async (text, i) => {
            const tensor = await this.preprocessInput(text);
            if (tensor) {
                return { tensor, label: labels[i] };
            } else {
                console.warn(`[FusionModel.train] Skipping sample at index ${i} due to preprocessing error.`);
                return null;
            }
        }));

        const validData = processedData.filter((item): item is { tensor: tf.Tensor1D; label: number } => item !== null);

        if (validData.length === 0) {
            throw new Error("[FusionModel.train] No samples could be processed successfully.");
        }

        // Ensure model is built
        const inputDim = validData[0].tensor.shape[0];
        if (!this.model) {
            this.model = this.buildModel(inputDim);
        } else {
            // Verify input dimension matches existing model
            const currentInputShape = (this.model.layers[0].batchInputShape as number[])[1];
            assert(currentInputShape === inputDim, `[FusionModel.train] Input dimension mismatch. Model expects ${currentInputShape}, data has ${inputDim}`);
        }

        // Stack tensors and create labels tensor
        const xs = tf.stack(validData.map(item => item.tensor));
        const ys = tf.tensor1d(validData.map(item => item.label), 'int32'); // Assuming integer labels for classification

        console.log(`[FusionModel] Training data shape: xs=${xs.shape}, ys=${ys.shape}`);

        const history = await this.model.fit(xs, ys, {
            epochs: this.config.epochs,
            batchSize: this.config.batchSize,
            validationSplit: this.config.validationSplit,
            callbacks: tf.callbacks.earlyStopping({ patience: this.config.earlyStoppingPatience }),
            shuffle: true,
        });

        console.log("[FusionModel] Training finished.");
        // Dispose intermediate tensors
        xs.dispose();
        ys.dispose();
        validData.forEach(item => item.tensor.dispose());

        return history;
    }

    // [paradigm:functional]
    async predict(text: string): Promise<number | null> {
        assert(!this.isDisposed, '[FusionModel] Model has been disposed.');
        // Reference: ./stylometric_fusion.ApiNotes.md#Prediction
        assert(text != null, '[FusionModel.predict] Input text cannot be null.');
        if (!this.model) {
            console.error("[FusionModel.predict] Model not trained or loaded. Cannot predict.");
            // Optionally load a pre-trained model here if available
            return null;
        }

        console.log("[FusionModel] Preprocessing text for prediction...");
        const inputTensor = await this.preprocessInput(text);
        if (!inputTensor) {
            console.error("[FusionModel.predict] Failed to preprocess input text.");
            return null;
        }

        const inputBatch = tf.tidy(() => inputTensor.expandDims(0)); // Add batch dimension
        console.log(`[FusionModel] Predicting with input shape: ${inputBatch.shape}`);
        const output = this.model.predict(inputBatch) as tf.Tensor;
        const prediction = await output.data();

        // Dispose tensors
        inputTensor.dispose();
        inputBatch.dispose();
        output.dispose();

        // Assuming binary classification with sigmoid output
        const result = prediction[0];
        console.log(`[FusionModel] Prediction output: ${result}`);
        return result; // Return the raw prediction score (e.g., probability)
    }

    dispose(): void {
        console.log("[FusionModel] Disposing fusion model...");
        if (!this.isDisposed) {
            if (this.model) {
                this.model.dispose();
                this.model = null;
            }
            // Dispose feature extractor and embedding model if they own resources
            if (this.featureExtractor && typeof (this.featureExtractor as any).dispose === 'function') {
                (this.featureExtractor as any).dispose();
            }
            if (this.embeddingModel) {
                this.embeddingModel.dispose(); // Crucial for the injected model
            }
            this.isDisposed = true;
        }
    }
}

// --- Main/Demo Function ---
async function main() {
    console.log("--- Fusion Model Demo ---");

    // --- USER ACTION REQUIRED ---
    // 1. Implement the `EmbeddingModel` interface (e.g., `MyEmbeddingModel` above).
    // 2. Choose and install the necessary embedding library (e.g., @tensorflow-models/universal-sentence-encoder, @xenova/transformers, or an API client).
    // 3. Provide the path or configuration needed to load your chosen embedding model.

    // Example Instantiation (Requires MyEmbeddingModel implementation and model path)
    let embeddingModel: EmbeddingModel;
    try {
        // embeddingModel = new MyEmbeddingModel('path/to/your/embedding/model'); // Replace with actual path/config
        // --- TEMPORARY FALLBACK ---
        console.warn("DEMO: Using placeholder EmbeddingModel. Implement and instantiate a real EmbeddingModel.");
        embeddingModel = { // Placeholder implementation for demo to run
            embed: async (text: string) => tf.tidy(() => tf.randomNormal([512]).as1D()),
            getEmbeddingDimension: () => 512,
            dispose: () => {}
        };
        // --- END TEMPORARY FALLBACK ---
    } catch (error: any) {
        console.error("Failed to initialize EmbeddingModel:", error.message || error);
        console.error("Please ensure you have implemented the EmbeddingModel interface and provided correct model details.");
        return; // Exit demo if embedding model fails
    }

    // Assuming StyleFeatureExtractor is available and works
    const featureExtractor = new StyleFeatureExtractor(/* config if needed */);

    const fusionModel = new FusionModel(featureExtractor, embeddingModel);

    // Example Training Data (Replace with actual data)
    const trainTexts = ["This is example text one.", "Another piece of text here.", "Style A example.", "Style B text."];
    const trainLabels = [0, 0, 0, 1]; // Example binary labels

    try {
        console.log("\n--- Training ---");
        await fusionModel.train(trainTexts, trainLabels);
        console.log("Training complete.");

        console.log("\n--- Prediction ---");
        const testText = "A new text to classify for style B.";
        const prediction = await fusionModel.predict(testText);

        if (prediction !== null) {
            console.log(`Prediction score for "${testText}": ${prediction.toFixed(4)}`);
            console.log(`Predicted class: ${prediction > 0.5 ? 1 : 0}`); // Threshold at 0.5 for binary
        } else {
            console.log("Prediction failed.");
        }

    } catch (error: any) {
        console.error("An error occurred during demo:", error.message || error);
    } finally {
        // Clean up models
        fusionModel.dispose();
        // featureExtractor might need disposal if it holds resources
        // embeddingModel is disposed via fusionModel.dispose()
        console.log("Demo finished.");
    }
}

// Run the demo if executed directly (optional)
// main().catch(console.error);

/**
 * Adds unit tests for the FusionModel.
 * Suggestion: Use vitest or jest. Add to a master test suite.
 * Requires mocking StyleFeatureExtractor and providing a mock/stub EmbeddingModel for testing purposes.
 * Example test structure:
 * describe('FusionModel', () => {
 *   let mockFeatureExtractor: StyleFeatureExtractor;
 *   let mockEmbeddingModel: EmbeddingModel;
 *   let fusionModel: FusionModel;
 *
 *   beforeEach(() => {
 *     // Mock dependencies
 *     mockFeatureExtractor = {
 *       extract: vi.fn().mockResolvedValue(tf.tidy(() => tf.randomNormal([10]))), // 10 features
 *       // dispose: vi.fn() // if needed
 *     } as unknown as StyleFeatureExtractor;
 *
 *     mockEmbeddingModel = {
 *       embed: vi.fn().mockResolvedValue(tf.tidy(() => tf.randomNormal([50]))), // 50 embedding dims
 *       getEmbeddingDimension: vi.fn().mockReturnValue(50),
 *       dispose: vi.fn()
 *     };
 *
 *     fusionModel = new FusionModel(mockFeatureExtractor, mockEmbeddingModel);
 *   });
 *
 *   afterEach(() => {
 *      fusionModel.dispose(); // Ensure model resources are cleaned up
 *      tf.disposeVariables(); // Clean up TF variables
 *      vi.restoreAllMocks();
 *   });
 *
 *   it('should build the model on first train call (expected success)', async () => {
 *     const buildSpy = vi.spyOn(fusionModel as any, 'buildModel');
 *     await fusionModel.train(['text1'], [0]);
 *     expect(buildSpy).toHaveBeenCalled();
 *     expect((fusionModel as any).model).not.toBeNull();
 *     const inputShape = ((fusionModel as any).model.layers[0].inputShape as number[]);
 *     expect(inputShape[1]).toBe(10 + 50); // features + embeddings
 *   });
 *
 *   it('should preprocess input correctly (expected success)', async () => {
 *      const text = "sample text";
 *      const tensor = await (fusionModel as any).preprocessInput(text);
 *      expect(tensor).toBeInstanceOf(tf.Tensor);
 *      expect(tensor.shape).toEqual([60]); // 10 features + 50 embedding dims
 *      expect(mockFeatureExtractor.extract).toHaveBeenCalledWith(text);
 *      expect(mockEmbeddingModel.embed).toHaveBeenCalledWith(text);
 *      tensor.dispose();
 *   });
 *
 *   it('should run prediction after training (expected success)', async () => {
 *      await fusionModel.train(['text1', 'text2'], [0, 1]); // Train first
 *      const predictSpy = vi.spyOn((fusionModel as any).model, 'predict');
 *      const result = await fusionModel.predict('new text');
 *      expect(predictSpy).toHaveBeenCalled();
 *      expect(typeof result).toBe('number');
 *   });
 *
 *    it('should return null prediction if model not trained (expected success)', async () => {
 *      const result = await fusionModel.predict('new text');
 *      expect(result).toBeNull();
 *   });
 *
 *    it('should handle preprocessing errors during training (expected success)', async () => {
 *       (mockEmbeddingModel.embed as Mock).mockRejectedValueOnce(new Error("Embedding failed"));
 *       // Training should still complete with the valid sample
 *       await expect(fusionModel.train(['fail text', 'good text'], [0, 1])).resolves.toBeInstanceOf(tf.History);
 *       // Ensure model was built with data from 'good text'
 *       expect((fusionModel as any).model).not.toBeNull();
 *    });
 *
 *    it('should throw error if all training samples fail preprocessing (expected failure)', async () => {
 *       (mockFeatureExtractor.extract as Mock).mockRejectedValue(new Error("Feature failed"));
 *       await expect(fusionModel.train(['fail1', 'fail2'], [0, 1])).rejects.toThrow(/No samples could be processed/);
 *    });
 * });
 */