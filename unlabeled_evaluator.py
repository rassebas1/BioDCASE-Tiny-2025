import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pydantic import BaseModel, field_validator
import yaml
from pathlib import Path
import json

# Configuration classes
class DataPreprocessing(BaseModel):
    audio_slice_duration_ms: int
    sample_rate: int


class FeatureExtraction(BaseModel):
    window_len: int
    window_stride: int
    window_scaling_bits: int
    mel_n_channels: int
    mel_low_hz: int
    mel_high_hz: int
    mel_post_scaling_bits: int

    @field_validator('mel_high_hz')
    @classmethod
    def validate_mel_high_hz(cls, v, values):
        if v <= values.data['mel_low_hz']:
            raise ValueError(f'mel_high_hz must be strictly greater than mel_low_hz')
        return v


class ModelTraining(BaseModel):
    class EarlyStopping(BaseModel):
        patience: int
    seed: int
    n_epochs: int
    shuffle_buff_n: int
    batch_size: int
    early_stopping: EarlyStopping


class EmbeddedCodeGeneration(BaseModel):
    serial_device: str


class Config(BaseModel):
    data_preprocessing: DataPreprocessing
    feature_extraction: FeatureExtraction
    model_training: ModelTraining
    embedded_code_generation: EmbeddedCodeGeneration


def load_config(config_path: Path) -> Config:
    with config_path.open("r") as file:
        yaml_data = yaml.safe_load(file)
    return Config(**yaml_data)


class AudioModelEvaluator:
    """
    Class for evaluating audio models on unlabeled data using model-based techniques
    Focuses on model performance evaluation rather than data analysis
    """
    
    def __init__(self, model, config: Config):
        self.model = model
        self.config = config
        self.batch_size = config.model_training.batch_size
        
    def predict_with_confidence(self, data, batch_size=None):
        """
        Get predictions with confidence scores for classification models
        Uses batch_size from config if not specified
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        predictions = self.model.predict(data, batch_size=batch_size)
        
        # For classification: confidence = max probability
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            confidence_scores = np.max(predictions, axis=1)
            predicted_classes = np.argmax(predictions, axis=1)
            return predicted_classes, confidence_scores, predictions
        else:
            # For regression or binary classification
            return predictions, None, predictions
    
    def confidence_based_evaluation(self, data, confidence_threshold=0.8):
        """
        Evaluate based on prediction confidence
        """
        _, confidence_scores, raw_predictions = self.predict_with_confidence(data)
        
        if confidence_scores is not None:
            high_confidence_mask = confidence_scores >= confidence_threshold
            high_confidence_ratio = np.mean(high_confidence_mask)
            avg_confidence = np.mean(confidence_scores)
            
            return {
                'high_confidence_ratio': high_confidence_ratio,
                'average_confidence': avg_confidence,
                'confidence_scores': confidence_scores,
                'high_confidence_predictions': raw_predictions[high_confidence_mask]
            }
        else:
            return {'message': 'Confidence evaluation not applicable for this model type'}
    
    def entropy_based_evaluation(self, data):
        """
        Evaluate using prediction entropy (uncertainty measure)
        """
        _, _, raw_predictions = self.predict_with_confidence(data)
        
        if len(raw_predictions.shape) > 1 and raw_predictions.shape[1] > 1:
            # Calculate entropy: -sum(p * log(p))
            epsilon = 1e-12  # Avoid log(0)
            entropy = -np.sum(raw_predictions * np.log(raw_predictions + epsilon), axis=1)
            
            return {
                'average_entropy': np.mean(entropy),
                'entropy_scores': entropy,
                'low_uncertainty_ratio': np.mean(entropy < np.median(entropy))
            }
        else:
            return {'message': 'Entropy evaluation not applicable for this model type'}
    
    def feature_space_evaluation(self, data, layer_name=None):
        """
        Evaluate using feature representations from intermediate layers
        """
        if layer_name:
            # Extract features from specific layer
            feature_extractor = tf.keras.Model(
                inputs=self.model.input,
                outputs=self.model.get_layer(layer_name).output
            )
            features = feature_extractor.predict(data, batch_size=self.batch_size)
        else:
            # Use the last hidden layer before output
            if len(self.model.layers) > 1:
                feature_extractor = tf.keras.Model(
                    inputs=self.model.input,
                    outputs=self.model.layers[-2].output
                )
                features = feature_extractor.predict(data, batch_size=self.batch_size)
            else:
                features = data
        
        # Flatten features if needed
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        
        # Clustering-based evaluation
        n_clusters = min(10, len(features) // 10)  # Reasonable number of clusters
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.config.model_training.seed)
            cluster_labels = kmeans.fit_predict(features)
            silhouette_avg = silhouette_score(features, cluster_labels)
            
            return {
                'silhouette_score': silhouette_avg,
                'n_clusters': n_clusters,
                'cluster_labels': cluster_labels,
                'features': features
            }
        else:
            return {'message': 'Not enough samples for clustering evaluation'}
    
    def prediction_consistency_evaluation(self, data, n_runs=5, dropout_rate=0.1):
        """
        Evaluate prediction consistency across multiple runs (Monte Carlo Dropout)
        """
        if not any('dropout' in layer.name.lower() for layer in self.model.layers):
            print("Warning: No dropout layers found. Adding temporary dropout for evaluation.")
            # Create a temporary model with dropout for MC evaluation
            temp_model = self._add_dropout_to_model(dropout_rate)
        else:
            temp_model = self.model
        
        predictions_list = []
        for _ in range(n_runs):
            # Enable training mode to activate dropout
            preds = temp_model(data, training=True)
            predictions_list.append(preds.numpy())
        
        predictions_array = np.array(predictions_list)
        
        # Calculate prediction variance
        pred_variance = np.var(predictions_array, axis=0)
        avg_variance = np.mean(pred_variance)
        
        # Calculate prediction mean
        pred_mean = np.mean(predictions_array, axis=0)
        
        return {
            'prediction_variance': avg_variance,
            'prediction_std': np.sqrt(avg_variance),
            'mean_predictions': pred_mean,
            'all_predictions': predictions_array
        }
    
    def _add_dropout_to_model(self, dropout_rate):
        """Helper method to add dropout layers for MC evaluation"""
        # This is a simplified version - you might need to adapt based on your model architecture
        inputs = self.model.input
        x = inputs
        
        for layer in self.model.layers[1:-1]:  # Skip input and output layers
            x = layer(x)
            if 'dense' in layer.name.lower():
                x = tf.keras.layers.Dropout(dropout_rate)(x)
        
        # Add final layer
        outputs = self.model.layers[-1](x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def comprehensive_model_evaluation(self, data, save_plots=False, plots_save_dir=None):
        """
        Run all model evaluation methods and return comprehensive results
        """
        results = {}
        
        # Confidence-based evaluation
        print("Running confidence-based evaluation...")
        results['confidence'] = self.confidence_based_evaluation(data)
        
        # Entropy-based evaluation
        print("Running entropy-based evaluation...")
        results['entropy'] = self.entropy_based_evaluation(data)
        
        # Feature space evaluation
        print("Running feature space evaluation...")
        results['feature_space'] = self.feature_space_evaluation(data)
        
        # Consistency evaluation
        print("Running prediction consistency evaluation...")
        results['consistency'] = self.prediction_consistency_evaluation(data)
        
        # Generate summary report
        results['summary'] = self._generate_summary_report(results)
        results['config_info'] = self._get_config_summary()
        
        if save_plots:
            self._save_evaluation_plots(results, data, plots_save_dir)
        
        return results
    
    def _get_config_summary(self):
        """Generate a summary of the configuration used"""
        return {
            'batch_size': self.config.model_training.batch_size,
            'random_seed': self.config.model_training.seed,
            'mel_channels': self.config.feature_extraction.mel_n_channels
        }
    
    def _generate_summary_report(self, results):
        """Generate a summary report of model evaluation"""
        summary = []
        
        # Add config information
        summary.append("=== Model Configuration Summary ===")
        config = results['config_info']
        summary.append(f"Batch size: {config['batch_size']}")
        summary.append(f"Random seed: {config['random_seed']}")
        summary.append(f"Mel channels: {config['mel_channels']}")
        
        summary.append("\n=== Model Performance Summary ===")
        
        if 'high_confidence_ratio' in results.get('confidence', {}):
            ratio = results['confidence']['high_confidence_ratio']
            summary.append(f"High confidence predictions: {ratio:.2%}")
        
        if 'average_entropy' in results.get('entropy', {}):
            entropy = results['entropy']['average_entropy']
            summary.append(f"Average prediction entropy: {entropy:.3f}")
        
        if 'silhouette_score' in results.get('feature_space', {}):
            silhouette = results['feature_space']['silhouette_score']
            summary.append(f"Feature clustering quality: {silhouette:.3f}")
        
        if 'prediction_std' in results.get('consistency', {}):
            std = results['consistency']['prediction_std']
            summary.append(f"Prediction consistency (std): {std:.3f}")
        
        return summary
    
    def _save_evaluation_plots(self, results, data, save_dir=None):
        """Save visualization plots of the model evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Confidence distribution
        if 'confidence_scores' in results.get('confidence', {}):
            axes[0, 0].hist(results['confidence']['confidence_scores'], bins=30, alpha=0.7)
            axes[0, 0].set_title('Model Prediction Confidence Distribution')
            axes[0, 0].set_xlabel('Confidence Score')
            axes[0, 0].set_ylabel('Frequency')
        
        # Plot 2: Entropy distribution
        if 'entropy_scores' in results.get('entropy', {}):
            axes[0, 1].hist(results['entropy']['entropy_scores'], bins=30, alpha=0.7)
            axes[0, 1].set_title('Model Prediction Entropy Distribution')
            axes[0, 1].set_xlabel('Entropy')
            axes[0, 1].set_ylabel('Frequency')
        
        # Plot 3: Feature space clusters (2D projection)
        if 'features' in results.get('feature_space', {}):
            features = results['feature_space']['features']
            labels = results['feature_space']['cluster_labels']
            
            # Use PCA for 2D projection if features are high-dimensional
            if features.shape[1] > 2:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                features_2d = pca.fit_transform(features)
            else:
                features_2d = features
            
            scatter = axes[1, 0].scatter(features_2d[:, 0], features_2d[:, 1], 
                                       c=labels, cmap='viridis', alpha=0.6)
            axes[1, 0].set_title('Model Feature Space Clusters')
            axes[1, 0].set_xlabel('Component 1')
            axes[1, 0].set_ylabel('Component 2')
            plt.colorbar(scatter, ax=axes[1, 0])
        
        # Plot 4: Prediction variance
        if 'prediction_variance' in results.get('consistency', {}):
            variance = results['consistency']['prediction_variance']
            if len(variance.shape) > 1:
                variance = np.mean(variance, axis=1)
            axes[1, 1].plot(variance)
            axes[1, 1].set_title('Model Prediction Variance per Sample')
            axes[1, 1].set_xlabel('Sample Index')
            axes[1, 1].set_ylabel('Variance')
        
        plt.tight_layout()
        
        # Save to specified directory or current directory
        if save_dir:
            save_path = Path(save_dir) / 'model_evaluation_plots.png'
            save_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            save_path = 'model_evaluation_plots.png'
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def load_evaluation_data_from_parquet(parquet_path: Path, features_shape_json_path: Path = None):
    """
    Load evaluation data from parquet file and reshape if needed
    
    Args:
        parquet_path: Path to the parquet file containing features
        features_shape_json_path: Optional path to JSON file containing original shape info
    
    Returns:
        numpy array with evaluation data
    """
    print(f"Loading evaluation data from: {parquet_path}")
    
    # Load parquet file
    df = pd.read_parquet(parquet_path)
    
    # Convert to numpy array
    if 'features' in df.columns:
        # If there's a 'features' column, use that
        features = df['features'].values
        # Convert list of arrays to single array if needed
        if isinstance(features[0], (list, np.ndarray)):
            features = np.array([np.array(f) for f in features])
    else:
        # Use all numeric columns as features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        features = df[numeric_cols].values
    
    # Load and apply original shape if available
    if features_shape_json_path and features_shape_json_path.exists():
        print(f"Loading shape information from: {features_shape_json_path}")
        with open(features_shape_json_path, 'r') as f:
            shape_info = json.load(f)
        
        if 'shape' in shape_info:
            target_shape = shape_info['shape']
            print(f"Reshaping data from {features.shape} to {target_shape}")
            # Reshape keeping the batch dimension
            features = features.reshape(-1, *target_shape[1:])
    
    print(f"Loaded evaluation data with shape: {features.shape}")
    return features


# Path constants - define these at the module level or import them
def get_pipeline_paths(base_path: Path = None):
    """Get standard pipeline paths"""
    if base_path is None:
        base_path = Path(__file__).parent
    
    DATA_DIR = base_path / 'data'
    FEATURES_DIR = DATA_DIR / "03_features"
    MODELS_DIR = DATA_DIR / '04_models'
    REPORTING_DIR = DATA_DIR / '05_reporting'
    
    return {
        'FEATURES_PRQ_PATH': FEATURES_DIR / "features.parquet",
        'FEATURES_SHAPE_JSON_PATH': FEATURES_DIR / "features_shape.json",
        'KERAS_MODEL_PATH': MODELS_DIR / 'model.keras',
        'REPORTING_DIR': REPORTING_DIR
    }


def evaluate_model_from_pipeline_paths(config: Config, base_path: Path = None):
    """
    Evaluate model using your pipeline's standard path structure
    Gets all paths automatically from the standard directory structure
    
    Args:
        config: Config object with pipeline configuration
        base_path: Optional base path of your project (defaults to current script's parent)
    
    Returns:
        Dictionary with model evaluation results
    """
    print("="*60)
    print("LOADING PIPELINE COMPONENTS FOR MODEL EVALUATION")
    print("="*60)
    
    # Get standard paths
    paths = get_pipeline_paths(base_path)
    keras_model_path = paths['KERAS_MODEL_PATH']
    evaluation_data_path = paths['FEATURES_PRQ_PATH']
    features_shape_json_path = paths['FEATURES_SHAPE_JSON_PATH']
    plots_save_dir = paths['REPORTING_DIR']
    
    print(f"Using paths:")
    print(f"  Model: {keras_model_path}")
    print(f"  Evaluation data: {evaluation_data_path}")
    print(f"  Features shape: {features_shape_json_path}")
    print(f"  Plots output: {plots_save_dir}")
    
    # Load model
    print(f"\nLoading Keras model from: {keras_model_path}")
    model = tf.keras.models.load_model(keras_model_path)
    print(f"Model loaded successfully. Architecture summary:")
    model.summary()
    
    # Load evaluation data
    evaluation_data = load_evaluation_data_from_parquet(
        evaluation_data_path, 
        features_shape_json_path
    )
    
    # Set random seeds for reproducibility
    tf.random.set_seed(config.model_training.seed)
    np.random.seed(config.model_training.seed)
    
    # Initialize evaluator
    evaluator = AudioModelEvaluator(model, config)
    
    # Run comprehensive evaluation
    print("\n" + "="*60)
    print("STARTING MODEL EVALUATION")
    print("="*60)
    
    results = evaluator.comprehensive_model_evaluation(
        evaluation_data, 
        save_plots=True,
        plots_save_dir=plots_save_dir
    )
    
    # Add path information to results
    results['paths'] = {
        'model_path': str(keras_model_path),
        'evaluation_data_path': str(evaluation_data_path),
        'features_shape_path': str(features_shape_json_path),
        'plots_save_dir': str(plots_save_dir)
    }
    
    # Print summary
    print("\n" + "="*60)
    print("AUDIO MODEL EVALUATION RESULTS")
    print("="*60)
    for item in results['summary']:
        print(item)
    
    return results


# Convenience function using your exact path structure
def evaluate_model_with_standard_paths(base_path: Path = None):
    """
    Evaluate model using the standard pipeline directory structure
    Assumes the script is run from the project root or base_path is provided
    
    Args:
        base_path: Base path of your project (defaults to current script's parent)
    
    Returns:
        Dictionary with model evaluation results
    """
    if base_path is None:
        base_path = Path(__file__).parent
    
    # Define paths using your structure
    PIPELINE_CONFIG_FILE = base_path / "pipeline_config.yaml"
    DATA_DIR = base_path / 'data'
    FEATURES_DIR = DATA_DIR / "03_features"
    FEATURES_PRQ_PATH = FEATURES_DIR / "features.parquet"
    FEATURES_SHAPE_JSON_PATH = FEATURES_DIR / "features_shape.json"
    MODELS_DIR = DATA_DIR / '04_models'
    KERAS_MODEL_PATH = MODELS_DIR / 'model.keras'
    REPORTING_DIR = DATA_DIR / '05_reporting'
    
    return evaluate_model_from_pipeline_paths(
        pipeline_config_path=PIPELINE_CONFIG_FILE,
        keras_model_path=KERAS_MODEL_PATH,
        evaluation_data_path=FEATURES_PRQ_PATH,
        features_shape_json_path=FEATURES_SHAPE_JSON_PATH,
        save_plots=True,
        plots_save_dir=REPORTING_DIR
    )


# Example usage with your path structure
"""
# Method 1: Using standard paths (recommended)
results = evaluate_model_with_standard_paths()

# Method 2: Simple usage with just config object (NEW - RECOMMENDED)
config = load_config(Path("pipeline_config.yaml"))
results = evaluate_model_from_pipeline_paths(config)

# Method 3: With custom base path
config = load_config(Path("pipeline_config.yaml"))
results = evaluate_model_from_pipeline_paths(config, base_path=Path("your_project_root"))

# Access specific evaluation results
print(f"Model confidence: {results['confidence']['average_confidence']:.3f}")
print(f"Feature clustering quality: {results['feature_space']['silhouette_score']:.3f}")
"""