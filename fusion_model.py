"""
Fusion Network for AI-Generated Text Detection

This module implements the fusion network architecture described in:
"Stylometric Detection of AI-Generated Text in Twitter Timelines"
by Kumarage et al. (2023)

The fusion network combines language model embeddings with stylometric features
for improved detection of AI-generated text, especially for short texts like tweets.

References:
- Kumarage, T., Garland, J., Bhattacharjee, A., Trapeznikov, K., Ruston, S., & Liu, H. (2023).
  Stylometric Detection of AI-Generated Text in Twitter Timelines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
from stylometry import StyleFeatureExtractor

class ReduceNetwork(nn.Module):
    """
    Reduce network component of the fusion architecture.
    
    This network takes the concatenation of stylometric features and language model
    embeddings and reduces them to a joint representation.
    """
    
    def __init__(self, stylo_dim: int, lm_embed_dim: int, hidden_dim: int, output_dim: int):
        """
        Initialize the reduce network
        
        Args:
            stylo_dim: Dimension of the stylometric feature vector
            lm_embed_dim: Dimension of the language model embedding
            hidden_dim: Dimension of the hidden layer
            output_dim: Dimension of the output layer
        """
        super().__init__()
        
        # Input is concatenation of stylometric features and LM embedding
        input_dim = stylo_dim + lm_embed_dim
        
        # Two fully-connected layers as specified in the paper
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, stylo_features: torch.Tensor, lm_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the reduce network
        
        Args:
            stylo_features: Stylometric features tensor
            lm_embedding: Language model embedding tensor
            
        Returns:
            Reduced joint representation
        """
        # Concatenate features
        combined = torch.cat([stylo_features, lm_embedding], dim=1)
        
        # First layer with ReLU activation
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        
        # Second layer
        x = self.fc2(x)
        
        return x


class ClassificationNetwork(nn.Module):
    """
    Classification network component of the fusion architecture.
    
    This network takes the reduced representation and outputs the final classification probability.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize the classification network
        
        Args:
            input_dim: Dimension of the input (from reduce network)
            hidden_dim: Dimension of the hidden layer
        """
        super().__init__()
        
        # Two fully-connected layers as specified in the paper
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)  # Binary classification: human vs. AI
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the classification network
        
        Args:
            x: Input tensor from reduce network
            
        Returns:
            Classification logits
        """
        # First layer with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Output layer (logits)
        x = self.fc2(x)
        
        return x


class StyleFusionModel(nn.Module):
    """
    Complete fusion model for AI-generated text detection.
    
    This model combines stylometric features with language model embeddings
    through a reduce network and classification network as described in 
    Kumarage et al. (2023).
    """
    
    def __init__(
        self, 
        lm_model: nn.Module,
        stylo_dim: int,
        lm_embed_dim: int,
        reduce_hidden_dim: int = 256,
        reduce_output_dim: int = 128,
        classify_hidden_dim: int = 64
    ):
        """
        Initialize the complete fusion model
        
        Args:
            lm_model: Pre-trained language model (e.g., RoBERTa)
            stylo_dim: Dimension of the stylometric feature vector
            lm_embed_dim: Dimension of the language model embedding
            reduce_hidden_dim: Hidden dimension for the reduce network
            reduce_output_dim: Output dimension for the reduce network
            classify_hidden_dim: Hidden dimension for the classification network
        """
        super().__init__()
        
        # Language model for text encoding
        self.lm_model = lm_model
        
        # Stylometric feature extractor
        self.stylometric_extractor = StyleFeatureExtractor()
        
        # Reduce network
        self.reduce_network = ReduceNetwork(
            stylo_dim=stylo_dim,
            lm_embed_dim=lm_embed_dim,
            hidden_dim=reduce_hidden_dim,
            output_dim=reduce_output_dim
        )
        
        # Classification network
        self.classification_network = ClassificationNetwork(
            input_dim=reduce_output_dim,
            hidden_dim=classify_hidden_dim
        )
        
    def extract_lm_embedding(self, texts: List[str]) -> torch.Tensor:
        """
        Extract language model embeddings for the input texts
        
        Args:
            texts: List of input texts
            
        Returns:
            Tensor of language model embeddings
        """
        # This implementation depends on the specific language model being used
        # Typically, we would tokenize the texts and pass them through the model
        
        # For RoBERTa, we would extract the CLS token embedding as discussed in the paper
        with torch.no_grad():
            outputs = self.lm_model(texts, output_hidden_states=True)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, embed_dim]
            
        return cls_embeddings
    
    def extract_stylometric_features(self, texts: List[str]) -> torch.Tensor:
        """
        Extract stylometric features for the input texts
        
        Args:
            texts: List of input texts
            
        Returns:
            Tensor of stylometric features
        """
        features = []
        
        for text in texts:
            # Extract features using the stylometric feature extractor
            text_features = self.stylometric_extractor.extract_all_features(text)
            
            # Convert dict to list of values in a consistent order
            feature_values = list(text_features.values())
            features.append(feature_values)
            
        return torch.tensor(features, dtype=torch.float)
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Forward pass for the fusion model
        
        Args:
            texts: List of input texts
            
        Returns:
            Classification logits
        """
        # Extract language model embeddings
        lm_embeddings = self.extract_lm_embedding(texts)
        
        # Extract stylometric features
        stylo_features = self.extract_stylometric_features(texts)
        
        # Pass through reduce network
        reduced = self.reduce_network(stylo_features, lm_embeddings)
        
        # Pass through classification network
        logits = self.classification_network(reduced)
        
        return logits
    
    def predict(self, texts: Union[str, List[str]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict whether texts are AI-generated
        
        Args:
            texts: Input text or list of texts
            
        Returns:
            Tuple of (probabilities, predictions)
            - probabilities: Probability that each text is AI-generated
            - predictions: Binary prediction (1 = AI-generated, 0 = human)
        """
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
        
        # Set model to evaluation mode
        self.eval()
        
        # Get logits
        with torch.no_grad():
            logits = self.forward(texts)
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)
            
            # Get AI-generated probabilities (class 1)
            ai_probs = probs[:, 1].numpy()
            
            # Binary predictions (threshold at 0.5)
            preds = (ai_probs > 0.5).astype(int)
            
        return ai_probs, preds


def train_fusion_model(
    model: StyleFusionModel,
    train_texts: List[str],
    train_labels: List[int],
    val_texts: Optional[List[str]] = None,
    val_labels: Optional[List[int]] = None,
    batch_size: int = 32,
    epochs: int = 5,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
):
    """
    Train the fusion model
    
    Args:
        model: StyleFusionModel instance
        train_texts: Training texts
        train_labels: Training labels (1 = AI-generated, 0 = human)
        val_texts: Validation texts (optional)
        val_labels: Validation labels (optional)
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        
    Returns:
        Trained model
    """
    # Set model to training mode
    model.train()
    
    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        # Process in batches
        for i in range(0, len(train_texts), batch_size):
            batch_texts = train_texts[i:i+batch_size]
            batch_labels = torch.tensor(train_labels[i:i+batch_size], dtype=torch.long)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(batch_texts)
            
            # Compute loss
            loss = criterion(logits, batch_labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)
        
        # Print epoch statistics
        train_loss = total_loss / (len(train_texts) // batch_size)
        train_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Acc: {train_acc:.4f}")
        
        # Validation
        if val_texts and val_labels:
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for i in range(0, len(val_texts), batch_size):
                    batch_texts = val_texts[i:i+batch_size]
                    batch_labels = torch.tensor(val_labels[i:i+batch_size], dtype=torch.long)
                    
                    # Forward pass
                    logits = model(batch_texts)
                    
                    # Compute loss
                    loss = criterion(logits, batch_labels)
                    val_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(logits, 1)
                    val_correct += (predicted == batch_labels).sum().item()
                    val_total += batch_labels.size(0)
            
            # Print validation statistics
            val_loss = val_loss / (len(val_texts) // batch_size)
            val_acc = val_correct / val_total
            print(f"Validation - Loss: {val_loss:.4f} - Acc: {val_acc:.4f}")
            
            # Set model back to training mode
            model.train()
    
    return model