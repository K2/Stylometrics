"""
Change Point Detection for Author Changes in Text Timelines

This module implements the Stylometric Change Point Agreement (StyloCPA) methodology
described in "Stylometric Detection of AI-Generated Text in Twitter Timelines" 
by Kumarage et al. (2023).

The StyloCPA methodology identifies if and when an author change occurs in a text timeline
by detecting change points in stylometric feature time series and measuring agreement
between these change points.

References:
- Kumarage, T., Garland, J., Bhattacharjee, A., Trapeznikov, K., Ruston, S., & Liu, H. (2023).
  Stylometric Detection of AI-Generated Text in Twitter Timelines.
- Killick, R., et al. (2012). Optimal detection of changepoints with a linear computational cost.
  Journal of the American Statistical Association.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, List, Tuple, Union, Optional
from stylometry import StyleFeatureExtractor

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False


class TimelineAnalyzer:
    """
    Analyzes a timeline of texts to extract stylometric features and visualize them.
    """
    
    def __init__(self):
        """Initialize the timeline analyzer"""
        self.feature_extractor = StyleFeatureExtractor()
        
    def extract_timeline_features(self, timeline: List[str]) -> pd.DataFrame:
        """
        Extract stylometric features for each text in a timeline
        
        Args:
            timeline: List of texts in chronological order
            
        Returns:
            DataFrame containing stylometric features for each text
        """
        features_list = []
        
        for i, text in enumerate(timeline):
            # Extract features for this text
            features = self.feature_extractor.extract_all_features(text)
            
            # Add position index
            features['position'] = i
            
            # Add to list
            features_list.append(features)
            
        # Convert to DataFrame
        df = pd.DataFrame(features_list)
        
        return df
    
    def visualize_features(self, feature_df: pd.DataFrame, selected_features: Optional[List[str]] = None):
        """
        Visualize stylometric features over time
        
        Args:
            feature_df: DataFrame containing extracted features
            selected_features: List of features to visualize (if None, uses a default set)
        """
        if selected_features is None:
            # Default set of features to visualize
            selected_features = [
                'mean_words_per_sentence',
                'total_punct_count',
                'lexical_richness',
                'readability'
            ]
        
        # Create figure with subplots
        fig, axes = plt.subplots(len(selected_features), 1, figsize=(10, 3*len(selected_features)))
        
        # If only one feature, axes is not an array
        if len(selected_features) == 1:
            axes = [axes]
        
        # Plot each feature
        for i, feature in enumerate(selected_features):
            if feature in feature_df.columns:
                axes[i].plot(feature_df['position'], feature_df[feature])
                axes[i].set_title(feature)
                axes[i].grid(True)
            else:
                axes[i].text(0.5, 0.5, f"Feature '{feature}' not found", ha='center')
        
        plt.tight_layout()
        plt.show()


class StyloCPADetector:
    """
    Implementation of the Stylometric Change Point Agreement (StyloCPA) detector.
    
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
        
        if not RUPTURES_AVAILABLE:
            print("Warning: 'ruptures' package not found. Install with: pip install ruptures")
    
    def detect_author_change(
        self, 
        timeline: List[str],
        return_feature_matrix: bool = False
    ) -> Union[Tuple[bool, int], Tuple[bool, int, np.ndarray, List[int]]]:
        """
        Detect if and where an author change occurs in a timeline of texts
        
        Args:
            timeline: List of text samples in chronological order
            return_feature_matrix: If True, also return the feature matrix and
                                  detected change points for each feature
            
        Returns:
            Tuple of (change_detected, change_point_index) or
            Tuple of (change_detected, change_point_index, feature_matrix, feature_change_points)
        """
        if not RUPTURES_AVAILABLE:
            raise RuntimeError(
                "Package 'ruptures' is required for change point detection. "
                "Install with: pip install ruptures"
            )
            
        # Extract features for each text in the timeline
        feature_matrix = []
        feature_names = []
        first_features = None
        
        for text in timeline:
            # Extract features for this text
            features = self.feature_extractor.extract_all_features(text)
            
            # Store feature names on first iteration
            if first_features is None:
                first_features = features
                feature_names = list(features.keys())
            
            # Add values to feature matrix
            feature_matrix.append([features[name] for name in feature_names])
        
        # Convert to numpy array
        feature_matrix = np.array(feature_matrix)
        
        # Transpose to get K time series of length N
        # Each row is now a feature's values across all texts
        feature_matrix_t = feature_matrix.T
        
        # Apply change point detection to each feature time series
        change_points = []
        feature_change_points = []
        
        for feature_series in feature_matrix_t:
            # Normalize the feature series
            mean = np.mean(feature_series)
            std = np.std(feature_series)
            normalized_series = feature_series
            if std > 0:  # Avoid division by zero
                normalized_series = (feature_series - mean) / std
                
            # Use PELT algorithm as described in Killick et al. (2012)
            algo = rpt.Pelt(model="l2").fit(normalized_series.reshape(-1, 1))
            result = algo.predict(pen=np.log(len(normalized_series)))
            
            # The PELT algorithm returns indices that come after the change point
            # We filter out the last point which is just the series length
            points = [pt - 1 for pt in result if pt < len(normalized_series)]
            
            # Store the change points for this feature
            feature_change_points.append(points)
            
            # Add the detected change points to our overall list
            change_points.extend(points)
        
        # If no change points were detected
        if not change_points:
            if return_feature_matrix:
                return (False, -1, feature_matrix, feature_change_points)
            return (False, -1)
        
        # Count occurrences of each change point
        point_counts = Counter(change_points)
        
        # Find the most agreed-upon change point
        most_common_point, count = point_counts.most_common(1)[0]
        
        # Calculate agreement percentage
        agreement = count / len(feature_matrix_t)
        
        # Check if agreement exceeds threshold
        change_detected = agreement >= self.agreement_threshold
        change_point = most_common_point if change_detected else -1
        
        if return_feature_matrix:
            return (change_detected, change_point, feature_matrix, feature_change_points)
        
        return (change_detected, change_point)
    
    def analyze_timeline(
        self, 
        timeline: List[str],
        title: str = "Timeline Analysis",
        true_change_point: Optional[int] = None
    ):
        """
        Analyze a timeline, detect change points, and visualize results
        
        Args:
            timeline: List of text samples in chronological order
            title: Title for the visualization
            true_change_point: True change point index for comparison (if known)
        """
        if not RUPTURES_AVAILABLE:
            raise RuntimeError("Package 'ruptures' is required for analysis.")
        
        # Detect change points and get feature matrix
        change_detected, change_point, feature_matrix, feature_change_points = \
            self.detect_author_change(timeline, return_feature_matrix=True)
        
        # Get feature names
        features = {}
        first_text = timeline[0] if timeline else ""
        if first_text:
            features = self.feature_extractor.extract_all_features(first_text)
        feature_names = list(features.keys())
        
        # Transpose to get features as columns
        feature_matrix_df = pd.DataFrame(feature_matrix, columns=feature_names)
        feature_matrix_df['position'] = range(len(timeline))
        
        # Select features to visualize (top features by importance from the paper)
        top_features = [
            'mean_words_per_sentence',
            'total_punct_count',
            'punct_,',
            'punct_.',
            'lexical_richness',
            'readability'
        ]
        
        # Only use features that exist in our data
        viz_features = [f for f in top_features if f in feature_matrix_df.columns]
        if not viz_features:
            viz_features = feature_names[:min(5, len(feature_names))]
        
        # Create figure with subplots
        fig, axes = plt.subplots(len(viz_features), 1, figsize=(12, 3*len(viz_features)))
        fig.suptitle(title, fontsize=16)
        
        # If only one feature, axes is not an array
        if len(viz_features) == 1:
            axes = [axes]
        
        # Plot each feature and its change points
        for i, feature in enumerate(viz_features):
            ax = axes[i]
            
            # Plot the feature values
            ax.plot(feature_matrix_df['position'], feature_matrix_df[feature], 'b-', label=feature)
            
            # Mark the detected change point if exists
            if change_detected and change_point >= 0:
                ax.axvline(x=change_point, color='r', linestyle='--', alpha=0.7, 
                          label=f'Detected Change Point ({change_point})')
            
            # Mark the true change point if provided
            if true_change_point is not None:
                ax.axvline(x=true_change_point, color='g', linestyle='-', alpha=0.7,
                          label=f'True Change Point ({true_change_point})')
            
            ax.set_title(feature)
            ax.grid(True)
            ax.legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for main title
        
        # Show overall result
        if change_detected:
            print(f"Author change detected at position {change_point}")
        else:
            print("No author change detected")
        
        plt.show()
        
        return change_detected, change_point


def detect_timeline_author_change(timeline: List[str], agreement_threshold: float = 0.15) -> Tuple[bool, int]:
    """
    Utility function to detect if and when an author change occurs in a timeline
    
    Args:
        timeline: List of text samples in chronological order
        agreement_threshold: Agreement threshold for change point detection
        
    Returns:
        Tuple of (change_detected, change_point_index)
    """
    detector = StyloCPADetector(agreement_threshold=agreement_threshold)
    return detector.detect_author_change(timeline)