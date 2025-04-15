"""
Stylometric Feature Extraction for AI Text Detection

This module implements stylometric feature extraction techniques based on:
"Stylometric Detection of AI-Generated Text in Twitter Timelines"
by Kumarage et al. (2023)

The module extracts three categories of stylometric features:
1. Phraseology - features which quantify how the author organizes words and phrases
2. Punctuation - features to quantify how the author utilizes different punctuation
3. Linguistic Diversity - features to quantify how the author uses different words

References:
- Kumarage, T., Garland, J., Bhattacharjee, A., Trapeznikov, K., Ruston, S., & Liu, H. (2023).
  Stylometric Detection of AI-Generated Text in Twitter Timelines.
- Covington, M.A., McFall, J.D. (2010). Cutting the gordian knot: The moving-average 
  type–token ratio (mattr). Journal of quantitative linguistics.
- Kincaid, J.P., et al. (1975). Derivation of new readability formulas.
"""

import re
import numpy as np
import pandas as pd
import statistics
from collections import Counter
from typing import Dict, List, Tuple, Union, Optional

class StyleFeatureExtractor:
    """
    Extract stylometric features from text as described in Kumarage et al. (2023)
    """
    
    def __init__(self, window_size: int = 50):
        """
        Initialize the stylometric feature extractor
        
        Args:
            window_size: Size of the window for Moving Average Type-Token Ratio (MATTR)
                         as described in Covington & McFall (2010)
        """
        self.window_size = window_size
        self.special_punct = ['!', "'", ',', ':', ';', '?', '"', '-', '–', '@', '#']
    
    def extract_all_features(self, text: str) -> Dict[str, float]:
        """
        Extract all stylometric features from the given text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing all extracted stylometric features
        """
        features = {}
        
        # Extract each feature category
        phraseology = self.extract_phraseology_features(text)
        punctuation = self.extract_punctuation_features(text)
        linguistic = self.extract_linguistic_features(text)
        
        # Combine all features
        features.update(phraseology)
        features.update(punctuation)
        features.update(linguistic)
        
        return features
    
    def extract_phraseology_features(self, text: str) -> Dict[str, float]:
        """
        Extract phraseology features that quantify how the author organizes words and phrases
        
        Features:
        - word count
        - sentence count
        - paragraph count
        - mean and stdev of word count per sentence
        - mean and stdev of word count per paragraph
        - mean and stdev of sentence count per paragraph
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary of phraseology features
        """
        # Split into paragraphs (split by double newlines)
        paragraphs = re.split(r'\n\s*\n', text.strip())
        paragraphs = [p for p in paragraphs if p.strip()]
        
        # Split into sentences (simple split by punctuation that ends sentences)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]
        
        # Get words (tokens split by whitespace)
        words = text.split()
        
        # Calculate words per sentence
        words_per_sentence = [len(s.split()) for s in sentences if s.strip()]
        
        # Calculate words per paragraph
        words_per_paragraph = [len(p.split()) for p in paragraphs if p.strip()]
        
        # Calculate sentences per paragraph
        sentences_per_paragraph = []
        for p in paragraphs:
            if p.strip():
                sent_count = len(re.split(r'[.!?]+', p))
                sentences_per_paragraph.append(sent_count)
        
        features = {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'paragraph_count': len(paragraphs),
            'mean_words_per_sentence': np.mean(words_per_sentence) if words_per_sentence else 0,
            'stdev_words_per_sentence': np.std(words_per_sentence) if len(words_per_sentence) > 1 else 0,
            'mean_words_per_paragraph': np.mean(words_per_paragraph) if words_per_paragraph else 0,
            'stdev_words_per_paragraph': np.std(words_per_paragraph) if len(words_per_paragraph) > 1 else 0,
            'mean_sentences_per_paragraph': np.mean(sentences_per_paragraph) if sentences_per_paragraph else 0,
            'stdev_sentences_per_paragraph': np.std(sentences_per_paragraph) if len(sentences_per_paragraph) > 1 else 0
        }
        
        return features
    
    def extract_punctuation_features(self, text: str) -> Dict[str, float]:
        """
        Extract punctuation features that quantify how the author uses punctuation
        
        Features:
        - total punctuation count
        - counts of special punctuation marks (!, ', ,, :, ;, ?, ", -, –, @, #)
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary of punctuation features
        """
        features = {}
        
        # Count all punctuation
        all_punct = re.findall(r'[^\w\s]', text)
        features['total_punct_count'] = len(all_punct)
        
        # Count specific punctuation marks
        for punct in self.special_punct:
            count = text.count(punct)
            features[f'punct_{punct}'] = count
            
        return features
    
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """
        Extract linguistic diversity features
        
        Features:
        - lexical richness (MATTR)
        - readability (Flesch Reading Ease)
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary of linguistic features
        """
        features = {}
        
        # Calculate lexical richness (Moving-Average Type-Token Ratio)
        features['lexical_richness'] = self._calculate_mattr(text)
        
        # Calculate readability (Flesch Reading Ease)
        features['readability'] = self._calculate_flesch_reading_ease(text)
        
        return features
    
    def _calculate_mattr(self, text: str) -> float:
        """
        Calculate the Moving-Average Type-Token Ratio (MATTR) as described in
        Covington & McFall (2010)
        
        Args:
            text: Input text to analyze
            
        Returns:
            MATTR score
        """
        words = text.lower().split()
        
        # If we don't have enough words for the window, use the standard TTR
        if len(words) <= self.window_size:
            if not words:
                return 0
            return len(set(words)) / len(words)
        
        # Calculate the average TTR over sliding windows
        ttrs = []
        for i in range(len(words) - self.window_size + 1):
            window = words[i:i+self.window_size]
            ttr = len(set(window)) / self.window_size
            ttrs.append(ttr)
            
        return np.mean(ttrs)
    
    def _calculate_flesch_reading_ease(self, text: str) -> float:
        """
        Calculate the Flesch Reading Ease score as described in Kincaid et al. (1975)
        
        Args:
            text: Input text to analyze
            
        Returns:
            Flesch Reading Ease score (0-100 scale)
        """
        # Count sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]
        sentence_count = len(sentences)
        
        # Count words
        words = text.split()
        word_count = len(words)
        
        # Count syllables (simple approximation)
        syllable_count = 0
        for word in words:
            word = word.lower()
            if len(word) <= 3:
                syllable_count += 1
                continue
                
            # Count vowel groups as syllables
            vowels = "aeiouy"
            prev_is_vowel = False
            count = 0
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_is_vowel:
                    count += 1
                prev_is_vowel = is_vowel
                
            # Adjust for common patterns
            if word.endswith('e'):
                count -= 1
            if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
                count += 1
            if count == 0:
                count = 1
                
            syllable_count += count
        
        # Avoid division by zero
        if sentence_count == 0 or word_count == 0:
            return 0
        
        # Calculate Flesch Reading Ease score
        score = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count)
        
        # Clamp to 0-100 range
        return max(0, min(100, score))


class StyloCPADetector:
    """
    Implementation of Stylometric Change Point Agreement (StyloCPA) detector
    based on Kumarage et al. (2023)
    
    This detector identifies if and when an author change occurs in a text timeline
    by measuring agreement between change points in different stylometric features.
    """
    
    def __init__(self, agreement_threshold: float = 0.15):
        """
        Initialize the StyloCPA detector
        
        Args:
            agreement_threshold: Percentage of features that must agree on a change point
                                 (γ in the paper, default 0.15)
        """
        self.feature_extractor = StyleFeatureExtractor()
        self.agreement_threshold = agreement_threshold
        
        try:
            # Attempt to import ruptures for change point detection
            import ruptures as rpt
            self.rpt = rpt
        except ImportError:
            raise ImportError(
                "Package 'ruptures' is required for change point detection. "
                "Install with: pip install ruptures"
            )
    
    def detect_author_change(self, timeline: List[str]) -> Tuple[bool, int]:
        """
        Detect if and where an author change occurs in a timeline of texts
        
        Args:
            timeline: List of text samples in chronological order
            
        Returns:
            Tuple of (change_detected, change_point_index)
        """
        # Extract features for each text in the timeline
        feature_matrix = []
        for text in timeline:
            features = self.feature_extractor.extract_all_features(text)
            feature_matrix.append(list(features.values()))
        
        # Convert to numpy array and transpose to get K time series of length N
        feature_matrix = np.array(feature_matrix).T
        
        # Apply change point detection to each feature time series
        change_points = []
        for feature_series in feature_matrix:
            # Use PELT algorithm as described in Killick et al. (2012)
            algo = self.rpt.Pelt(model="l2").fit(feature_series.reshape(-1, 1))
            result = algo.predict(pen=np.log(len(feature_series)))
            
            # The PELT algorithm returns indices that come after the change point
            # We filter out the last point which is just the series length
            points = [pt - 1 for pt in result if pt < len(feature_series)]
            
            # Add the detected change points to our list
            change_points.extend(points)
        
        # If no change points were detected
        if not change_points:
            return (False, -1)
        
        # Count occurrences of each change point
        point_counts = Counter(change_points)
        
        # Find the most agreed-upon change point
        most_common_point, count = point_counts.most_common(1)[0]
        
        # Calculate agreement percentage
        agreement = count / len(feature_matrix)
        
        # Check if agreement exceeds threshold
        if agreement >= self.agreement_threshold:
            return (True, most_common_point)
        else:
            return (False, -1)


class FusionModel:
    """
    Implementation of the stylometry fusion model from Kumarage et al. (2023)
    
    This model combines stylometric features with language model embeddings
    for improved AI-generated text detection.
    """
    
    def __init__(self, lm_model=None):
        """
        Initialize the fusion model
        
        Args:
            lm_model: Pre-trained language model to use for embeddings
                     (should have a method to extract CLS token embeddings)
        """
        self.feature_extractor = StyleFeatureExtractor()
        self.lm_model = lm_model
        
        # This would be implemented with PyTorch or TensorFlow in practice
        # Here we just define the architecture conceptually
        self.reduce_network_size = 2  # Number of fully connected layers for reduction
        self.classification_network_size = 2  # Number of fully connected layers for classification
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """
        Extract both stylometric features and language model embeddings
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary of extracted features
        """
        # Extract stylometric features
        stylo_features = self.feature_extractor.extract_all_features(text)
        
        # In a real implementation, this would extract the LM embedding
        # and combine it with stylometric features
        if self.lm_model is not None:
            # This is a placeholder for LM embedding extraction
            # lm_embedding = self.lm_model.extract_cls_embedding(text)
            pass
        
        return stylo_features
    
    def predict(self, text: str) -> float:
        """
        Predict whether text is AI-generated using the fusion model
        
        Args:
            text: Input text to analyze
            
        Returns:
            Probability that the text is AI-generated
        """
        # This is a conceptual implementation
        # In practice, this would use the trained neural networks
        features = self.extract_features(text)
        
        # Placeholder for actual prediction logic
        # In a real implementation, this would use the reduce and classification networks
        # reduced_features = self.reduce_network(features)
        # prediction = self.classification_network(reduced_features)
        
        # For now, just return a placeholder value
        return 0.5