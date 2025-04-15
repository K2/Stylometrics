// filepath: /home/files/git/Stylometrics/stylometric_fusion.genai.mts
/**
 * Stylometric Fusion Model for AI-Generated Text Detection
 *
 * Implements the fusion network architecture from Kumarage et al. (2023) using TF.js.
 * Combines stylometric features and LM embeddings for detection.
 *
 * Design Goals: Implement TF.js networks, prediction, and training logic. Improve normalization.
 * Constraints: Requires TF.js, StyleFeatureExtractor, EmbeddingModel. See stylometric_fusion.ApiNotes.md.
 * Paradigm: Imperative (TF.js), Functional (Data transformation).
 *
 * Flow:
 * 1. `StyleFusionModel` constructor: Initializes networks, gets feature/embedding dimensions.
 * 2. `predict`: Extracts features, gets embedding, runs through ReduceNetwork -> ClassificationNetwork -> outputs probability.
 * 3. `trainFusionModel`: Prepares data, compiles models, runs training loop using `fitDataset`.
 */

import * as tf from '@tensorflow/tfjs';
// Ensure tfjs backend is initialized
// import '@tensorflow/tfjs-node'; // or '@tensorflow/tfjs-backend-wasm' etc. depending on environment

// Assuming StyleFeatureExtractor is in the detection module (adjust path if needed)
import { StyleFeatureExtractor, type FeatureMap } from './stylometric_detection.genai.mts'; // Use .mts if that's the actual file

/**
 * Interface for embedding model that can generate text embeddings
 */
export interface EmbeddingModel {
    generateEmbedding(text: string): Promise<Float32Array>;
    getEmbeddingDimension(): number;
}

/**
 * Reduce Network component
 */
export class ReduceNetwork {
    model: tf.Sequential; // Make model public for access in training
    readonly styloFeatureDim: number;
    readonly lmEmbedDim: number;
    readonly outputDim: number;

    constructor(
        styloFeatureDim: number,
        lmEmbedDim: number,
        hiddenDim: number = 128,
        outputDim: number = 64
    ) {
        // assert styloFeatureDim > 0 : 'Stylo feature dimension must be positive';
        // assert lmEmbedDim > 0 : 'LM embedding dimension must be positive';
        this.styloFeatureDim = styloFeatureDim;
        this.lmEmbedDim = lmEmbedDim;
        this.outputDim = outputDim;
        this.model = this.buildNetwork(hiddenDim);
    }

    private buildNetwork(hiddenDim: number): tf.Sequential {
        const model = tf.sequential({ name: 'ReduceNetwork' });
        const inputDim = this.styloFeatureDim + this.lmEmbedDim;
        // assert inputDim > 0 : 'Total input dimension must be positive';

        model.add(tf.layers.dense({
            units: hiddenDim, activation: 'relu', inputShape: [inputDim]
        }));
        model.add(tf.layers.dropout({ rate: 0.3 })); // Increased dropout slightly
        model.add(tf.layers.dense({ units: this.outputDim, activation: 'relu' })); // Added ReLU based on common patterns
        return model;
    }

    predict(styloFeatures: tf.Tensor, lmEmbedding: tf.Tensor): tf.Tensor {
        // assert styloFeatures.shape[0] === lmEmbedding.shape[0] : 'Batch sizes must match for features and embeddings';
        // assert styloFeatures.shape[1] === this.styloFeatureDim : 'Feature tensor has incorrect dimension';
        // assert lmEmbedding.shape[1] === this.lmEmbedDim : 'Embedding tensor has incorrect dimension';
        const combined = tf.concat([styloFeatures, lmEmbedding], 1);
        const result = this.model.predict(combined) as tf.Tensor;
        // combined.dispose(); // Dispose intermediate tensor - handled by tf.tidy in caller
        // assert result.shape[1] === this.outputDim : 'ReduceNetwork output has incorrect dimension';
        return result;
    }

    getModel(): tf.Sequential { return this.model; }
}

/**
 * Classification Network component
 */
export class ClassificationNetwork {
    model: tf.Sequential; // Make model public for access in training
    readonly inputDim: number;

    constructor(inputDim: number, hiddenDim: number = 32) {
        // assert inputDim > 0 : 'Input dimension must be positive';
        this.inputDim = inputDim;
        this.model = this.buildNetwork(hiddenDim);
    }

    private buildNetwork(hiddenDim: number): tf.Sequential {
        const model = tf.sequential({ name: 'ClassificationNetwork' });
        model.add(tf.layers.dense({
            units: hiddenDim, activation: 'relu', inputShape: [this.inputDim]
        }));
        model.add(tf.layers.dropout({ rate: 0.3 })); // Increased dropout slightly
        // Output layer: 2 units (Human, AI) with softmax for probabilities
        model.add(tf.layers.dense({ units: 2, activation: 'softmax' }));
        return model;
    }

    predict(x: tf.Tensor): tf.Tensor {
        // assert x.shape[1] === this.inputDim : 'ClassificationNetwork input has incorrect dimension';
        const result = this.model.predict(x) as tf.Tensor;
        // assert result.shape[1] === 2 : 'ClassificationNetwork output should have 2 units';
        return result;
    }

    getModel(): tf.Sequential { return this.model; }
}

/**
 * Interface for normalization parameters (mean and standard deviation)
 */
interface NormalizationParams {
    mean: tf.Tensor1D;
    stdDev: tf.Tensor1D;
}

/**
 * Complete StyleFusionModel
 */
export class StyleFusionModel {
    featureExtractor: StyleFeatureExtractor; // Make public for access in training
    embeddingModel: EmbeddingModel; // Make public for access in training
    reduceNetwork: ReduceNetwork;
    classificationNetwork: ClassificationNetwork;
    styloFeatureNames: string[] = [];
    private normalizationParams: NormalizationParams | null = null; // Store normalization params

    constructor(
        embeddingModel: EmbeddingModel,
        reduceHiddenDim: number = 128,
        reduceOutputDim: number = 64,
        classifyHiddenDim: number = 32
    ) {
        this.featureExtractor = new StyleFeatureExtractor();
        this.embeddingModel = embeddingModel;

        // Determine feature dimension from an example
        const exampleFeatures = this.featureExtractor.extractAllFeatures("Example text.");
        this.styloFeatureNames = Object.keys(exampleFeatures).sort(); // Sort for consistent order
        const styloDim = this.styloFeatureNames.length;
        const lmEmbedDim = embeddingModel.getEmbeddingDimension();
        // assert styloDim > 0 : 'Could not determine stylometric feature dimension';
        // assert lmEmbedDim > 0 : 'Could not determine embedding dimension';


        this.reduceNetwork = new ReduceNetwork(styloDim, lmEmbedDim, reduceHiddenDim, reduceOutputDim);
        this.classificationNetwork = new ClassificationNetwork(reduceOutputDim, classifyHiddenDim);
    }

    /**
     * Set normalization parameters (mean and standard deviation for each feature).
     * These should be calculated from the training dataset.
     */
    setNormalizationParams(params: NormalizationParams): void {
        // assert params.mean.shape[0] === this.styloFeatureNames.length : 'Normalization mean dimension mismatch';
        // assert params.stdDev.shape[0] === this.styloFeatureNames.length : 'Normalization stdDev dimension mismatch';
        // Dispose previous params if they exist
        this.normalizationParams?.mean.dispose();
        this.normalizationParams?.stdDev.dispose();
        this.normalizationParams = {
             mean: params.mean.clone(), // Clone to avoid external modification
             stdDev: params.stdDev.clone()
        };
    }

    /**
     * Convert a feature map to a normalized tensor using stored parameters.
     */
    featureMapToTensor(features: FeatureMap): tf.Tensor2D {
        // assert this.styloFeatureNames.length > 0 : 'Feature names not initialized';
        const featureValues = this.styloFeatureNames.map(name => features[name] || 0);
        const tensor = tf.tensor1d(featureValues);

        if (this.normalizationParams) {
            // Apply standardization: (value - mean) / stdDev
            // Add small epsilon to stdDev to avoid division by zero
            const normalized = tensor.sub(this.normalizationParams.mean)
                                   .div(this.normalizationParams.stdDev.add(tf.scalar(1e-6)));
            tensor.dispose(); // Dispose original tensor
            // Reshape to 2D tensor (batch size 1)
            return normalized.reshape([1, this.styloFeatureNames.length]);
        } else {
            // Warn if normalization params not set, return unnormalized
            console.warn("Normalization parameters not set. Returning unnormalized features.");
            // Reshape to 2D tensor (batch size 1)
            return tensor.reshape([1, this.styloFeatureNames.length]);
        }
    }


    /**
     * Predict whether text is AI-generated
     */
    async predict(text: string): Promise<{
        isAiGenerated: boolean;
        probability: number; // Probability of being AI-generated (class 1)
        features: FeatureMap;
    }> {
        // assert text && text.length > 0 : 'Input text cannot be empty';
        if (!text) {
             throw new Error("Input text cannot be empty for prediction.");
        }

        // Keep track of tensors created outside tidy for potential manual disposal on error
        let styloFeaturesTensor: tf.Tensor2D | null = null;
        let lmEmbeddingTensor: tf.Tensor2D | null = null;

        try {
            // Use tf.tidy to automatically dispose intermediate tensors like reducedTensor, logitsTensor
            return await tf.tidy(async () => {
                const features = this.featureExtractor.extractAllFeatures(text);
                styloFeaturesTensor = this.featureMapToTensor(features); // Managed outside tidy if normalization fails

                const embedding = await this.embeddingModel.generateEmbedding(text);
                // assert embedding && embedding.length === this.embeddingModel.getEmbeddingDimension() : 'Invalid embedding received';
                lmEmbeddingTensor = tf.tensor2d([Array.from(embedding)]); // Managed outside tidy

                // --- Start of added logic ---
                // assert styloFeaturesTensor.shape[0] === 1 : 'Feature tensor should have batch size 1';
                // assert lmEmbeddingTensor.shape[0] === 1 : 'Embedding tensor should have batch size 1';

                const reducedTensor = this.reduceNetwork.predict(styloFeaturesTensor, lmEmbeddingTensor);
                const logitsTensor = this.classificationNetwork.predict(reducedTensor);
                // --- End of added logic ---

                // Get probabilities [prob_human, prob_ai]
                const probabilities = await logitsTensor.array() as number[][];
                // assert probabilities && probabilities.length === 1 && probabilities[0].length === 2 : 'Logits tensor has unexpected shape';
                const aiProbability = probabilities[0][1];
                // assert aiProbability >= 0 && aiProbability <= 1 : `Invalid AI probability calculated: ${aiProbability}`;


                return {
                    isAiGenerated: aiProbability >= 0.5,
                    probability: aiProbability,
                    features // Return original (unnormalized) features for inspection
                };
            });
        } catch (error) {
            console.error("Error in fusion model prediction:", error);
            // Manually dispose tensors created outside tidy if an error occurred before tidy completed
            // Note: Tensors created *inside* tidy are usually handled, but being explicit can help in complex cases.
            styloFeaturesTensor?.dispose();
            lmEmbeddingTensor?.dispose();
            // reducedTensor and logitsTensor are handled by tf.tidy if created successfully
            throw error; // Re-throw the error
        } finally {
             // Ensure tensors created outside tidy are disposed even if predict returns normally
             // (featureMapToTensor might return an unnormalized tensor outside tidy)
             if (styloFeaturesTensor && !styloFeaturesTensor.isDisposed) {
                 styloFeaturesTensor.dispose();
             }
             if (lmEmbeddingTensor && !lmEmbeddingTensor.isDisposed) {
                 lmEmbeddingTensor.dispose();
             }
        }
    }

    /**
     * Get the TensorFlow models (useful for training)
     */
    getModels(): {
        reduceNetwork: tf.Sequential;
        classificationNetwork: tf.Sequential;
    } {
        return {
            reduceNetwork: this.reduceNetwork.getModel(),
            classificationNetwork: this.classificationNetwork.getModel()
        };
    }

     /**
     * Dispose of the model's tensors (normalization params)
     */
    dispose(): void {
        this.normalizationParams?.mean.dispose();
        this.normalizationParams?.stdDev.dispose();
        this.normalizationParams = null;
        // Note: The models themselves (weights) are managed by TF.js
        // If models were loaded from files, tf.loadLayersModel returns a tf.LayersModel
        // which might have a dispose method, but Sequential models created directly
        // don't typically need manual disposal of the model structure itself.
        // Weights are tensors managed internally by TF.js layers.
        console.log("StyleFusionModel disposed normalization parameters.");
    }
}

/**
 * Simple mock embedding model implementation
 */
export class MockEmbeddingModel implements EmbeddingModel {
    private dimension: number;
    constructor(dimension: number = 768) { this.dimension = dimension; }
    async generateEmbedding(text: string): Promise<Float32Array> {
        // Simple deterministic embedding based on text length and char codes
        const embedding = new Float32Array(this.dimension);
        // Ensure text is not empty to avoid NaN/errors in charCodeAt
        const safeText = text || " ";
        const seed = safeText.length + safeText.charCodeAt(0 % safeText.length) + safeText.charCodeAt(Math.min(5, safeText.length - 1) % safeText.length);
        for (let i = 0; i < this.dimension; i++) {
            embedding[i] = (Math.sin(seed + i * 0.1) + 1) / 2; // Value between 0 and 1
        }
        return embedding;
    }
    getEmbeddingDimension(): number { return this.dimension; }
}

/**
 * Train the fusion model on labeled data
 */
export async function trainFusionModel(
    model: StyleFusionModel,
    texts: string[],
    labels: number[], // 0 for human, 1 for AI
    options: {
        epochs?: number,
        batchSize?: number,
        learningRate?: number,
        validationSplit?: number
    } = {}
): Promise<tf.History> {
    const {
        epochs = 5,
        batchSize = 16,
        learningRate = 0.001,
        validationSplit = 0.2
    } = options;

    // --- Start of added/completed logic ---
    // assert texts.length === labels.length : `Number of texts (${texts.length}) and labels (${labels.length}) must match`;
    // assert texts.length > 0 : 'Training data cannot be empty';
    // assert validationSplit >= 0 && validationSplit < 1 : `Validation split must be between 0 and 1, got ${validationSplit}`;
    if (texts.length !== labels.length) {
        throw new Error(`Number of texts (${texts.length}) and labels (${labels.length}) must match`);
    }
    if (texts.length === 0) {
        throw new Error('Training data cannot be empty');
    }
     if (validationSplit < 0 || validationSplit >= 1) {
        throw new Error(`Validation split must be between 0 and 1, got ${validationSplit}`);
    }


    console.log(`Starting training with ${texts.length} examples...`);

    // 1. Pre-extract features and embeddings (and calculate normalization params)
    console.log("Preprocessing data (extracting features and embeddings)...");
    const allFeatures: number[][] = [];
    const allEmbeddings: Float32Array[] = [];

    // Use Promise.all for potentially faster embedding generation if model supports concurrency
    await Promise.all(texts.map(async (text) => {
        const features = model.featureExtractor.extractAllFeatures(text);
        // Ensure consistent feature order using sorted names from the model
        const featureValues = model.styloFeatureNames.map(name => {
            const value = features[name];
            // Handle potential NaN or Infinity values from feature extraction
            if (value === undefined || value === null || !isFinite(value)) {
                 console.warn(`Invalid feature value for '${name}' in text: "${text.substring(0, 50)}...". Using 0.`);
                 return 0;
            }
            return value;
        });
        allFeatures.push(featureValues); // Order might be mixed due to async, need to align later if order matters before tensor creation (it doesn't here as we process text by text)

        // In a real scenario, embeddings might be pre-computed or fetched in batches
        const embedding = await model.embeddingModel.generateEmbedding(text);
        allEmbeddings.push(embedding);
    }));

    // Ensure features and embeddings align with original texts/labels order if Promise.all reordered
    // (Re-iterate to build tensors in the correct order)
    const orderedFeatures: number[][] = [];
    const orderedEmbeddings: Float32Array[] = [];
    for (let i = 0; i < texts.length; i++) {
        const text = texts[i];
        const features = model.featureExtractor.extractAllFeatures(text);
        const featureValues = model.styloFeatureNames.map(name => {
             const value = features[name];
             return (value === undefined || value === null || !isFinite(value)) ? 0 : value;
        });
        orderedFeatures.push(featureValues);
        // Find the corresponding embedding (assuming generateEmbedding is deterministic or we cached results)
        // This is inefficient; better to store results indexed by text or index in the Promise.all map.
        // For simplicity here, we re-generate, assuming the mock is fast and deterministic.
        orderedEmbeddings.push(await model.embeddingModel.generateEmbedding(text));
    }


    const featuresTensor = tf.tensor2d(orderedFeatures);
    const embeddingsTensor = tf.tensor2d(orderedEmbeddings.map(e => Array.from(e)));
    // One-hot encode labels: [1, 0] for human (0), [0, 1] for AI (1)
    const labelsTensor = tf.tidy(() => tf.oneHot(tf.tensor1d(labels, 'int32'), 2));
    // assert featuresTensor.shape[0] === texts.length : 'Features tensor batch size mismatch';
    // assert embeddingsTensor.shape[0] === texts.length : 'Embeddings tensor batch size mismatch';
    // assert labelsTensor.shape[0] === texts.length : 'Labels tensor batch size mismatch';


    // 2. Calculate and set normalization parameters based on training data
    console.log("Calculating normalization parameters...");
    let mean: tf.Tensor | null = null;
    let variance: tf.Tensor | null = null;
    let stdDev: tf.Tensor | null = null;
    let normalizedFeaturesTensor: tf.Tensor | null = null;
    let combinedInputTensor: tf.Tensor | null = null;

    try {
        ({ mean, variance } = tf.moments(featuresTensor, 0));
        stdDev = tf.sqrt(variance);
        // Handle features with zero variance (replace stdDev with epsilon to avoid NaN)
        const epsilon = 1e-6;
        const safeStdDev = tf.where(tf.greater(stdDev, epsilon), stdDev, tf.fill(stdDev.shape, epsilon));

        model.setNormalizationParams({ mean: mean as tf.Tensor1D, stdDev: safeStdDev as tf.Tensor1D });
        console.log("Normalization parameters set.");

        // 3. Normalize features using the calculated parameters
        normalizedFeaturesTensor = tf.tidy(() =>
             featuresTensor.sub(mean!).div(safeStdDev) // Use safeStdDev
        );
        // assert normalizedFeaturesTensor.shape[0] === texts.length : 'Normalized features tensor batch size mismatch';


        // 4. Combine features and embeddings for the ReduceNetwork input
        combinedInputTensor = tf.concat([normalizedFeaturesTensor, embeddingsTensor], 1);
        // assert combinedInputTensor.shape[0] === texts.length : 'Combined input tensor batch size mismatch';
        // assert combinedInputTensor.shape[1] === model.reduceNetwork.styloFeatureDim + model.reduceNetwork.lmEmbedDim : 'Combined input tensor dimension mismatch';


        // 5. Define the full model for training (Reduce -> Classify)
        // We use the existing network instances from the model
        const fullModel = tf.sequential({ name: 'FullFusionModel' });
        fullModel.add(model.reduceNetwork.getModel());
        fullModel.add(model.classificationNetwork.getModel());

        // 6. Compile the full model
        fullModel.compile({
            optimizer: tf.train.adam(learningRate),
            loss: 'categoricalCrossentropy', // Use categoricalCrossentropy for one-hot labels
            metrics: ['accuracy']
        });
        console.log("Model compiled.");
        fullModel.summary(); // Log model structure


        // 7. Train the model
        console.log(`Training for ${epochs} epochs with batch size ${batchSize}...`);
        const history = await fullModel.fit(combinedInputTensor, labelsTensor, {
            epochs: epochs,
            batchSize: batchSize,
            validationSplit: validationSplit,
            shuffle: true, // Shuffle data each epoch
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    // Log training progress
                    const logMsg = `Epoch ${epoch + 1}/${epochs} - loss: ${logs?.loss?.toFixed(4) ?? 'N/A'} - acc: ${logs?.acc?.toFixed(4) ?? 'N/A'}` +
                                   (validationSplit > 0 ? ` - val_loss: ${logs?.val_loss?.toFixed(4) ?? 'N/A'} - val_acc: ${logs?.val_acc?.toFixed(4) ?? 'N/A'}` : '');
                    console.log(logMsg);
                }
                // Consider adding tf.callbacks.earlyStopping() for more robust training
            }
        });
        console.log("Training complete.");

        return history;

    } finally {
        // 8. Clean up tensors
        featuresTensor.dispose();
        embeddingsTensor.dispose();
        labelsTensor.dispose();
        mean?.dispose();
        variance?.dispose();
        stdDev?.dispose(); // Dispose original stdDev
        // safeStdDev is derived from stdDev, should be disposed if stdDev is, or managed by tidy if created inside
        normalizedFeaturesTensor?.dispose();
        combinedInputTensor?.dispose();
        console.log("Training tensors disposed.");
        // Note: model weights are retained in the model object passed by reference.
        // The compiled fullModel itself doesn't need explicit disposal here as it uses the layers from the original model.
    }
    // --- End of added/completed logic ---
}


/**
 * Demonstrate the fusion model capabilities
 */
export async function demonstrateFusionModel(
    humanText: string = "This is a short text written by a human. It has some variation.",
    aiText: string = "This text segment was meticulously generated by an advanced large language model, exhibiting consistent syntactical structures and lexical patterns often associated with artificial intelligence.",
    train: boolean = true // Option to include training in the demo
): Promise<void> {
    console.log("=== FUSION MODEL DEMO ===");

    let model: StyleFusionModel | null = null;
    let meanTensor: tf.Tensor | null = null; // Keep track for disposal
    let stdDevTensor: tf.Tensor | null = null; // Keep track for disposal

    try {
        const embeddingModel = new MockEmbeddingModel(768); // Use mock for demo
        model = new StyleFusionModel(embeddingModel);

        if (train) {
            console.log("\n0. TRAINING MODEL (DEMO):");
            console.log("-------------------------");
            // Create minimal training data for demo
            const trainTexts = [
                humanText,
                aiText,
                "Another human example, quite simple really.",
                "A further AI sample demonstrating verbosity and complex sentence construction.",
                "Humans write like this sometimes.",
                "Generated content often follows predictable formats."
                ];
            const trainLabels = [0, 1, 0, 1, 0, 1]; // 0=human, 1=AI
            try {
                 // Train for a few epochs for demonstration
                 await trainFusionModel(model, trainTexts, trainLabels, {
                     epochs: 5, // Increased epochs slightly for demo
                     batchSize: 2,
                     learningRate: 0.002,
                     validationSplit: 0 // No validation split for this tiny dataset
                    });
                 console.log("Demo training finished.");
            } catch (trainError) {
                 console.error("Error during demo training:", trainError);
                 console.log("Proceeding with untrained model for prediction (using dummy normalization).");
                 // Reset normalization if training failed midway or didn't happen
                 model.dispose(); // Dispose potentially invalid normalization params
                 // Need to set dummy normalization params if not training or if training failed
                 const styloDim = model.styloFeatureNames.length;
                 meanTensor = tf.zeros([styloDim]);
                 stdDevTensor = tf.ones([styloDim]);
                 model.setNormalizationParams({
                     mean: meanTensor as tf.Tensor1D,
                     stdDev: stdDevTensor as tf.Tensor1D
                 });
            }

        } else {
            console.log("\nSkipping training phase. Using dummy normalization.");
            // Need to set dummy normalization params if not training
            const styloDim = model.styloFeatureNames.length;
            meanTensor = tf.zeros([styloDim]);
            stdDevTensor = tf.ones([styloDim]);
            model.setNormalizationParams({
                mean: meanTensor as tf.Tensor1D,
                stdDev: stdDevTensor as tf.Tensor1D
            });
        }

        console.log("\n1. PREDICTING HUMAN TEXT:");
        console.log("-------------------------");
        const humanPrediction = await model.predict(humanText);
        console.log(`Human text prediction: ${humanPrediction.isAiGenerated ? "AI-generated" : "Human"} (probability: ${humanPrediction.probability.toFixed(4)})`);
        // console.log("Features:", humanPrediction.features); // Optional: Log features

        console.log("\n2. PREDICTING AI TEXT:");
        console.log("----------------------");
        const aiPrediction = await model.predict(aiText);
        console.log(`AI text prediction: ${aiPrediction.isAiGenerated ? "AI-generated" : "Human"} (probability: ${aiPrediction.probability.toFixed(4)})`);
        // console.log("Features:", aiPrediction.features); // Optional: Log features

        console.log("\n3. PREDICTING EMPTY TEXT (EXPECT ERROR):");
        console.log("---------------------------------------");
        try {
            await model.predict("");
        } catch (error: any) {
            console.log(`Caught expected error: ${error.message}`);
        }


    } catch (error) {
        console.error("Error during fusion model demonstration:", error);
    } finally {
        model?.dispose(); // Dispose model resources (normalization params)
        // Dispose dummy tensors if they were created
        meanTensor?.dispose();
        stdDevTensor?.dispose();
        console.log("Demo finished.");
        // Check for memory leaks (optional, requires tfjs-node usually)
        // console.log("TF Memory:", tf.memory());
    }
}

// Example of how to run the demo (e.g., in a main script or test runner)
/*
 if (require.main === module) {
     demonstrateFusionModel().catch(console.error);
 }
 */