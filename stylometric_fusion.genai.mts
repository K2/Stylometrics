/**
 * Stylometric Fusion Model for AI-Generated Text Detection
 * 
 * This module implements the fusion network architecture described in:
 * "Stylometric Detection of AI-Generated Text in Twitter Timelines"
 * by Kumarage et al. (2023)
 * 
 * Flow:
 * 1. Extract stylometric features from text using StyleFeatureExtractor
 * 2. Extract language model embedding from text using provided LM
 * 3. Combine features and embedding in Reduce Network
 * 4. Process reduced representation through Classification Network
 * 5. Output binary classification and confidence score
 */

import * as tf from '@tensorflow/tfjs';
import { StyleFeatureExtractor, type FeatureMap } from './stylometric_detection.genai.mjs';

/**
 * Interface for embedding model that can generate text embeddings
 */
export interface EmbeddingModel {
    /**
     * Generate embeddings for the given text
     * @param text Text to embed
     * @returns Float32Array embedding
     */
    generateEmbedding(text: string): Promise<Float32Array>;

    /**
     * Get the embedding dimension
     * @returns Embedding dimension
     */
    getEmbeddingDimension(): number;
}

/**
 * Reduce Network component that combines stylometric features and language model embeddings
 */
export class ReduceNetwork {
    private model: tf.Sequential;
    private styloFeatureDim: number;
    private lmEmbedDim: number;
    private outputDim: number;

    /**
     * Initialize the reduce network
     * @param styloFeatureDim Dimension of the stylometric feature vector
     * @param lmEmbedDim Dimension of the language model embedding
     * @param hiddenDim Dimension of the hidden layer
     * @param outputDim Dimension of the output layer
     */
    constructor(
        styloFeatureDim: number,
        lmEmbedDim: number,
        hiddenDim: number = 128,
        outputDim: number = 64
    ) {
        this.styloFeatureDim = styloFeatureDim;
        this.lmEmbedDim = lmEmbedDim;
        this.outputDim = outputDim;
        this.model = this.buildNetwork(hiddenDim);
    }

    /**
     * Build the reduce network architecture
     * @param hiddenDim Dimension of hidden layers
     * @returns TensorFlow sequential model
     */
    private buildNetwork(hiddenDim: number): tf.Sequential {
        const model = tf.sequential();
        
        // Input layer dimensions (stylometric features + LM embedding)
        const inputDim = this.styloFeatureDim + this.lmEmbedDim;
        
        // First dense layer with ReLU activation
        model.add(tf.layers.dense({
            units: hiddenDim,
            activation: 'relu',
            inputShape: [inputDim]
        }));
        
        // Dropout for regularization
        model.add(tf.layers.dropout({ rate: 0.2 }));
        
        // Output layer (linear activation)
        model.add(tf.layers.dense({
            units: this.outputDim
        }));
        
        return model;
    }

    /**
     * Forward pass through the reduce network
     * @param styloFeatures Stylometric features tensor
     * @param lmEmbedding Language model embedding tensor
     * @returns Reduced joint representation
     */
    predict(styloFeatures: tf.Tensor, lmEmbedding: tf.Tensor): tf.Tensor {
        // Concatenate features
        const combined = tf.concat([styloFeatures, lmEmbedding], 1);
        
        // Forward pass
        return this.model.predict(combined) as tf.Tensor;
    }

    /**
     * Get the TensorFlow model
     * @returns Sequential model
     */
    getModel(): tf.Sequential {
        return this.model;
    }
}

/**
 * Classification Network component that outputs the final binary classification
 */
export class ClassificationNetwork {
    private model: tf.Sequential;
    private inputDim: number;

    /**
     * Initialize the classification network
     * @param inputDim Dimension of the input (from reduce network)
     * @param hiddenDim Dimension of the hidden layer
     */
    constructor(inputDim: number, hiddenDim: number = 32) {
        this.inputDim = inputDim;
        this.model = this.buildNetwork(hiddenDim);
    }

    /**
     * Build the classification network architecture
     * @param hiddenDim Dimension of hidden layer
     * @returns TensorFlow sequential model
     */
    private buildNetwork(hiddenDim: number): tf.Sequential {
        const model = tf.sequential();
        
        // First dense layer with ReLU activation
        model.add(tf.layers.dense({
            units: hiddenDim,
            activation: 'relu',
            inputShape: [this.inputDim]
        }));
        
        // Dropout for regularization
        model.add(tf.layers.dropout({ rate: 0.2 }));
        
        // Output layer with sigmoid activation for binary classification
        model.add(tf.layers.dense({
            units: 2,
            activation: 'softmax'
        }));
        
        return model;
    }

    /**
     * Forward pass through the classification network
     * @param x Input tensor from reduce network
     * @returns Classification probabilities
     */
    predict(x: tf.Tensor): tf.Tensor {
        return this.model.predict(x) as tf.Tensor;
    }

    /**
     * Get the TensorFlow model
     * @returns Sequential model
     */
    getModel(): tf.Sequential {
        return this.model;
    }
}

/**
 * Complete StyleFusionModel that combines stylometric features with language model embeddings
 */
export class StyleFusionModel {
    private featureExtractor: StyleFeatureExtractor;
    private embeddingModel: EmbeddingModel;
    private reduceNetwork: ReduceNetwork;
    private classificationNetwork: ClassificationNetwork;
    private styloFeatureNames: string[] = [];

    /**
     * Initialize the complete fusion model
     * @param embeddingModel Pre-trained language model for embeddings
     * @param reduceHiddenDim Hidden dimension for the reduce network
     * @param reduceOutputDim Output dimension for the reduce network
     * @param classifyHiddenDim Hidden dimension for the classification network
     */
    constructor(
        embeddingModel: EmbeddingModel,
        reduceHiddenDim: number = 128,
        reduceOutputDim: number = 64,
        classifyHiddenDim: number = 32
    ) {
        this.featureExtractor = new StyleFeatureExtractor();
        this.embeddingModel = embeddingModel;
        
        // Extract example features to determine dimension
        const exampleFeatures = this.featureExtractor.extractAllFeatures("Example text to determine feature dimensions.");
        this.styloFeatureNames = Object.keys(exampleFeatures);
        const styloDim = this.styloFeatureNames.length;
        const lmEmbedDim = embeddingModel.getEmbeddingDimension();
        
        // Initialize networks
        this.reduceNetwork = new ReduceNetwork(
            styloDim,
            lmEmbedDim,
            reduceHiddenDim,
            reduceOutputDim
        );
        
        this.classificationNetwork = new ClassificationNetwork(
            reduceOutputDim,
            classifyHiddenDim
        );
    }

    /**
     * Convert a feature map to a normalized tensor
     * @param features Feature map
     * @returns Tensor of normalized features
     */
    private featureMapToTensor(features: FeatureMap): tf.Tensor2D {
        // Ensure consistent feature ordering by using styloFeatureNames
        const featureValues = this.styloFeatureNames.map(name => features[name] || 0);
        
        // Simple min-max normalization (in production, would use pre-computed stats)
        // This is a placeholder for proper normalization
        const normalized = featureValues.map(value => {
            if (value === 0) return 0;
            // Simple normalization to roughly 0-1 range
            return value > 100 ? value / 1000 : value / 100;
        });
        
        // Convert to 2D tensor (batch size 1)
        return tf.tensor2d([normalized]);
    }

    /**
     * Extract stylometric features and convert to tensor
     * @param text Input text
     * @returns Tensor of stylometric features
     */
    private extractStylometricFeatures(text: string): tf.Tensor2D {
        const features = this.featureExtractor.extractAllFeatures(text);
        return this.featureMapToTensor(features);
    }

    /**
     * Predict whether text is AI-generated
     * @param text Input text
     * @returns Promise resolving to prediction object
     */
    async predict(text: string): Promise<{
        isAiGenerated: boolean;
        probability: number;
        features: FeatureMap;
    }> {
        try {
            // Extract stylometric features
            const features = this.featureExtractor.extractAllFeatures(text);
            const styloFeatures = this.featureMapToTensor(features);
            
            // Get language model embedding
            const embedding = await this.embeddingModel.generateEmbedding(text);
            const lmEmbedding = tf.tensor2d([Array.from(embedding)]);
            
            // Forward pass through reduce network
            const reduced = this.reduceNetwork.predict(styloFeatures, lmEmbedding);
            
            // Forward pass through classification network
            const logits = this.classificationNetwork.predict(reduced);
            
            // Get probability of AI-generated (class 1)
            const probabilities = await logits.array() as number[][];
            const aiProbability = probabilities[0][1];
            
            // Clean up tensors
            styloFeatures.dispose();
            lmEmbedding.dispose();
            reduced.dispose();
            logits.dispose();
            
            return {
                isAiGenerated: aiProbability >= 0.5,
                probability: aiProbability,
                features
            };
        } catch (error) {
            console.error("Error in fusion model prediction:", error);
            throw error;
        }
    }

    /**
     * Get the TensorFlow models
     * @returns Object containing reduce and classification networks
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
}

/**
 * Simple embedding model implementation using random embeddings
 * In practice, you would use a real language model like BERT, RoBERTa, etc.
 */
export class MockEmbeddingModel implements EmbeddingModel {
    private dimension: number;
    
    /**
     * Initialize the mock embedding model
     * @param dimension Dimension of the embeddings
     */
    constructor(dimension: number = 768) {
        this.dimension = dimension;
    }
    
    /**
     * Generate a random embedding
     * @param text Input text (unused in mock)
     * @returns Promise resolving to random embedding
     */
    async generateEmbedding(text: string): Promise<Float32Array> {
        // Create a deterministic but simple "embedding" based on text features
        const embedding = new Float32Array(this.dimension);
        
        // Fill with pseudo-random values derived from text
        const seed = Array.from(text).reduce((acc, char) => acc + char.charCodeAt(0), 0);
        
        for (let i = 0; i < this.dimension; i++) {
            // Simple deterministic function based on text and position
            embedding[i] = Math.sin(seed * (i + 1) * 0.01) * 0.5 + 0.5;
        }
        
        return embedding;
    }
    
    /**
     * Get the embedding dimension
     * @returns Embedding dimension
     */
    getEmbeddingDimension(): number {
        return this.dimension;
    }
}

/**
 * Train the fusion model on labeled data
 * @param model StyleFusionModel to train
 * @param texts Array of training texts
 * @param labels Array of training labels (1 for AI-generated, 0 for human)
 * @param epochs Number of training epochs
 * @param batchSize Batch size for training
 */
export async function trainFusionModel(
    model: StyleFusionModel,
    texts: string[],
    labels: number[],
    epochs: number = 5,
    batchSize: number = 16
): Promise<void> {
    const models = model.getModels();
    
    // Compile models
    models.reduceNetwork.compile({
        optimizer: tf.train.adam(0.0001),
        loss: 'meanSquaredError'
    });
    
    models.classificationNetwork.compile({
        optimizer: tf.train.adam(0.0001),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });
    
    // In a real implementation, we would:
    // 1. Pre-extract all features and embeddings
    // 2. Create proper TensorFlow datasets
    // 3. Implement a full training loop with validation
    // 4. Save model checkpoints
    
    console.log(`Training would start here with ${texts.length} examples over ${epochs} epochs`);
    console.log("This is a placeholder for actual training implementation");
}

/**
 * Demonstrate the fusion model capabilities
 * @param humanText Example of known human-written text
 * @param aiText Example of known AI-generated text
 */
export async function demonstrateFusionModel(
    humanText: string,
    aiText: string
): Promise<void> {
    console.log("=== FUSION MODEL DEMO ===");
    
    // Create mock embedding model (in practice, use a real language model)
    const embeddingModel = new MockEmbeddingModel(768);
    
    // Create fusion model
    const fusionModel = new StyleFusionModel(embeddingModel);
    
    // Classify human text
    console.log("\n1. ANALYZING HUMAN TEXT:");
    console.log("------------------------");
    try {
        const humanResult = await fusionModel.predict(humanText);
        console.log(`Classification: ${humanResult.isAiGenerated ? 'AI-GENERATED' : 'HUMAN-WRITTEN'}`);
        console.log(`Probability of being AI-generated: ${(humanResult.probability * 100).toFixed(2)}%`);
        console.log("Key features:");
        console.log(`- Lexical richness: ${humanResult.features.lexical_richness.toFixed(3)}`);
        console.log(`- Readability: ${humanResult.features.readability.toFixed(1)}`);
        console.log(`- Words per sentence: ${humanResult.features.mean_words_per_sentence.toFixed(1)}`);
    } catch (error) {
        console.error("Error analyzing human text:", error);
    }
    
    // Classify AI text
    console.log("\n2. ANALYZING AI TEXT:");
    console.log("--------------------");
    try {
        const aiResult = await fusionModel.predict(aiText);
        console.log(`Classification: ${aiResult.isAiGenerated ? 'AI-GENERATED' : 'HUMAN-WRITTEN'}`);
        console.log(`Probability of being AI-generated: ${(aiResult.probability * 100).toFixed(2)}%`);
        console.log("Key features:");
        console.log(`- Lexical richness: ${aiResult.features.lexical_richness.toFixed(3)}`);
        console.log(`- Readability: ${aiResult.features.readability.toFixed(1)}`);
        console.log(`- Words per sentence: ${aiResult.features.mean_words_per_sentence.toFixed(1)}`);
    } catch (error) {
        console.error("Error analyzing AI text:", error);
    }
    
    console.log("\n=== DEMO COMPLETE ===");
}