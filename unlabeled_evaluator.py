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
                'high_confidence_predictions': raw_predictions[high_confidence_mask],
                'raw_predictions': raw_predictions  # Added for saving
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
                'low_uncertainty_ratio': np.mean(entropy < np.median(entropy)),
                'raw_predictions': raw_predictions  # Added for saving
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
    
    def _get_detailed_model_metadata(self):
        """
        Extract comprehensive metadata about the model
        """
        metadata = {
            'model_name': self.model.name,
            'model_input_shape': self.model.input_shape,
            'model_output_shape': self.model.output_shape,
            'total_params': self.model.count_params(),
            'trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]),
            'non_trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights]),
            'optimizer': str(self.model.optimizer.__class__.__name__) if self.model.optimizer else None,
            'loss_function': str(self.model.loss) if hasattr(self.model, 'loss') else None,
            'metrics': [str(m) for m in self.model.metrics] if hasattr(self.model, 'metrics') else [],
            'layers': []
        }
        
        # Add detailed layer information
        for i, layer in enumerate(self.model.layers):
            layer_info = {
                'index': i,
                'name': layer.name,
                'type': layer.__class__.__name__,
                'param_count': layer.count_params(),
                'trainable': layer.trainable
            }
            
            # Add layer-specific parameters
            if hasattr(layer, 'activation') and layer.activation:
                layer_info['activation'] = str(layer.activation.__name__)
            if hasattr(layer, 'units'):
                layer_info['units'] = layer.units
            if hasattr(layer, 'filters'):
                layer_info['filters'] = layer.filters
            if hasattr(layer, 'kernel_size'):
                layer_info['kernel_size'] = layer.kernel_size
            if hasattr(layer, 'strides'):
                layer_info['strides'] = layer.strides
            if hasattr(layer, 'rate') and hasattr(layer, 'dropout'):  # Dropout layer
                layer_info['dropout_rate'] = layer.rate
                
            metadata['layers'].append(layer_info)
        
        return metadata

    def save_model_outputs(self, results, save_dir=None):
        """
        Save model outputs from evaluation results to files
        
        Args:
            results: Dictionary containing evaluation results
            save_dir: Directory to save outputs (defaults to reporting directory)
        """
        if save_dir is None:
            save_dir = Path('.')
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Get comprehensive model metadata
        model_metadata = self._get_detailed_model_metadata()
        
        # Save confidence-based predictions
        if 'confidence' in results and 'raw_predictions' in results['confidence']:
            confidence_data = {
                'raw_predictions': results['confidence']['raw_predictions'],
                'confidence_scores': results['confidence'].get('confidence_scores'),
                'high_confidence_predictions': results['confidence'].get('high_confidence_predictions'),
                'model_metadata': model_metadata,
                'evaluation_config': results.get('config', {})
            }
            np.savez(save_dir / 'confidence_predictions.npz', **confidence_data)
            print(f"Saved confidence predictions to: {save_dir / 'confidence_predictions.npz'}")
        
        # Save entropy-based predictions
        if 'entropy' in results and 'raw_predictions' in results['entropy']:
            entropy_data = {
                'raw_predictions': results['entropy']['raw_predictions'],
                'entropy_scores': results['entropy'].get('entropy_scores'),
                'model_metadata': model_metadata,
                'evaluation_config': results.get('config', {})
            }
            np.savez(save_dir / 'entropy_predictions.npz', **entropy_data)
            print(f"Saved entropy predictions to: {save_dir / 'entropy_predictions.npz'}")
        
        # Save feature space outputs
        if 'feature_space' in results and 'features' in results['feature_space']:
            feature_data = {
                'features': results['feature_space']['features'],
                'cluster_labels': results['feature_space'].get('cluster_labels'),
                'model_metadata': model_metadata,
                'evaluation_config': results.get('config', {})
            }
            np.savez(save_dir / 'feature_space_outputs.npz', **feature_data)
            print(f"Saved feature space outputs to: {save_dir / 'feature_space_outputs.npz'}")
        
        # Save consistency predictions
        if 'consistency' in results and 'all_predictions' in results['consistency']:
            consistency_data = {
                'all_predictions': results['consistency']['all_predictions'],
                'mean_predictions': results['consistency'].get('mean_predictions'),
                'prediction_variance': results['consistency'].get('prediction_variance'),
                'model_metadata': model_metadata,
                'evaluation_config': results.get('config', {})
            }
            np.savez(save_dir / 'consistency_predictions.npz', **consistency_data)
            print(f"Saved consistency predictions to: {save_dir / 'consistency_predictions.npz'}")
        
        # Save a summary of all outputs with comprehensive metadata
        summary_data = {
            'model_metadata': model_metadata,
            'evaluation_config': results.get('config', {}),
            'evaluation_timestamp': str(pd.Timestamp.now()),
            'evaluation_summary': results.get('summary', [])
        }
        
        for eval_type in ['confidence', 'entropy', 'consistency']:
            if eval_type in results and 'raw_predictions' in results[eval_type]:
                summary_data[f'{eval_type}_predictions'] = results[eval_type]['raw_predictions']
            elif eval_type == 'consistency' and 'mean_predictions' in results[eval_type]:
                summary_data[f'{eval_type}_predictions'] = results[eval_type]['mean_predictions']
        
        if any(key.endswith('_predictions') for key in summary_data.keys()):
            np.savez(save_dir / 'all_model_outputs.npz', **summary_data)
            print(f"Saved summary of all outputs to: {save_dir / 'all_model_outputs.npz'}")
        
        # Save model metadata as separate JSON file for easy reading
        import json
        with open(save_dir / 'model_metadata.json', 'w') as f:
            json.dump(model_metadata, f, indent=2, default=str)
        print(f"Saved model metadata to: {save_dir / 'model_metadata.json'}")
        
        # Save evaluation configuration as separate JSON file
        eval_config = {
            'evaluation_config': results.get('config', {}),
            'evaluation_timestamp': str(pd.Timestamp.now()),
            'paths': results.get('paths', {}),
            'summary': results.get('summary', [])
        }
        with open(save_dir / 'evaluation_config.json', 'w') as f:
            json.dump(eval_config, f, indent=2, default=str)
        print(f"Saved evaluation config to: {save_dir / 'evaluation_config.json'}")
    
    def comprehensive_model_evaluation(self, data, save_plots=False, plots_save_dir=None, save_outputs=True):
        """
        Run all model evaluation methods and return comprehensive results
        Added save_outputs parameter to control output saving
        """
        results = {}
        results['config'] = self._get_config_summary()
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
        
        # Save model outputs if requested
        if save_outputs:
            self.save_model_outputs(results, plots_save_dir)
        
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
        config = results['config']
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
        
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        if 'confidence_scores' in results.get('confidence', {}):
            ax1.hist(results['confidence']['confidence_scores'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title('Model Prediction Confidence Distribution', fontweight='bold', fontsize=12)
            ax1.set_xlabel('Confidence Score')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)

        # Plot 2: Entropy distribution (top-right)
        ax2 = plt.subplot2grid((2, 2), (0, 1))
        if 'entropy_scores' in results.get('entropy', {}):
            ax2.hist(results['entropy']['entropy_scores'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
            ax2.set_title('Model Prediction Entropy Distribution', fontweight='bold', fontsize=12)
            ax2.set_xlabel('Entropy')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)

        # Plot 3: Feature space clusters (bottom row, spanning both columns)
        ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
        if 'features' in results.get('feature_space', {}):
            features = results['feature_space']['features']
            labels = results['feature_space']['cluster_labels']
            
            # Use PCA for 2D projection if features are high-dimensional
            if features.shape[1] > 2:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                features_2d = pca.fit_transform(features)
                xlabel = f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)'
                ylabel = f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)'
            else:
                features_2d = features
                xlabel = 'Component 1'
                ylabel = 'Component 2'
            
            scatter = ax3.scatter(features_2d[:, 0], features_2d[:, 1], 
                                c=labels, cmap='viridis', alpha=0.7, s=50)
            ax3.set_title('Model Feature Space Clusters', fontweight='bold', fontsize=12)
            ax3.set_xlabel(xlabel)
            ax3.set_ylabel(ylabel)
            ax3.grid(True, alpha=0.3)
            
            # Add colorbar with better positioning
            cbar = plt.colorbar(scatter, ax=ax3, shrink=0.8)
            cbar.set_label('Cluster Labels', rotation=270, labelpad=20)

               
        plt.tight_layout(pad=3.0)
        
        # Save to specified directory or current directory
        if save_dir:
            save_path = Path(save_dir) / 'model_evaluation_plots.png'
            save_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            save_path = 'model_evaluation_plots.png'
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def load_evaluation_data_from_parquet_fixed(parquet_path: Path, features_shape_json_path: Path = None, expected_shape=None):
    """
    Load evaluation data from parquet file and reshape properly for model input
    
    Args:
        parquet_path: Path to the parquet file containing features
        features_shape_json_path: Optional path to JSON file containing original shape info
        expected_shape: Expected input shape for the model (e.g., (55, 40, 1))
    
    Returns:
        numpy array with evaluation data properly shaped for model
    """
    print(f"Loading evaluation data from: {parquet_path}")
    
    # Load parquet file
    df = pd.read_parquet(parquet_path)
    
    # Convert to numpy array
    if 'features' in df.columns:
        features = df['features'].values
        # Convert list of arrays to single array if needed
        if isinstance(features[0], (list, np.ndarray)):
            features = np.array([np.array(f) for f in features])
    else:
        # Use all numeric columns as features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        features = df[numeric_cols].values
    
    print(f"Raw data shape: {features.shape}")
    
    # Method 1: Use shape from JSON file
    if features_shape_json_path and features_shape_json_path.exists():
        print(f"Loading shape information from: {features_shape_json_path}")
        with open(features_shape_json_path, 'r') as f:
            shape_info = json.load(f)
        
        if 'shape' in shape_info:
            target_shape = shape_info['shape']
            print(f"Target shape from JSON: {target_shape}")
            
            # Ensure we have the right number of elements
            expected_elements = np.prod(target_shape[1:])  # Skip batch dimension
            actual_elements = features.shape[1] if len(features.shape) > 1 else len(features)
            
            if expected_elements == actual_elements:
                features = features.reshape(-1, *target_shape[1:])
                print(f"Successfully reshaped to: {features.shape}")
                return features
            else:
                print(f"Warning: Shape mismatch. Expected {expected_elements} elements, got {actual_elements}")
    
    # Method 2: Use provided expected_shape
    if expected_shape is not None:
        expected_elements = np.prod(expected_shape)
        actual_elements = features.shape[1] if len(features.shape) > 1 else len(features)
        
        print(f"Trying to reshape to expected shape: {expected_shape}")
        print(f"Expected elements: {expected_elements}, Actual elements: {actual_elements}")
        
        if expected_elements == actual_elements:
            features = features.reshape(-1, *expected_shape)
            print(f"Successfully reshaped to: {features.shape}")
            return features
    
    # Method 3: Auto-detect common audio feature shapes
    if len(features.shape) == 2:
        n_samples, n_features = features.shape
        
        # Common mel-spectrogram shapes for audio
        possible_shapes = [
            (55, 40, 1),    # Your model's expected shape
            (64, 64, 1),    # Square mel-spectrogram
            (128, 128, 1),  # Larger square
            (80, 80, 1),    # Another common size
            (n_features, 1, 1), # Treat as 1D signal
        ]
        
        for shape in possible_shapes:
            if np.prod(shape) == n_features:
                print(f"Auto-detected shape: {shape}")
                features = features.reshape(-1, *shape)
                print(f"Reshaped data to: {features.shape}")
                return features
    
    # Method 4: If it's 2200 features, likely it's 55x40 flattened
    if features.shape[1] == 2200:
        # 55 * 40 = 2200, so this is likely your mel-spectrogram flattened
        features = features.reshape(-1, 55, 40, 1)
        print(f"Detected flattened mel-spectrogram, reshaped to: {features.shape}")
        return features
    
    print(f"Could not determine proper shape. Returning original: {features.shape}")
    return features

def debug_model_input_requirements(model):
    """
    Debug function to understand what input shape the model expects
    """
    print("=== MODEL INPUT REQUIREMENTS ===")
    print(f"Model name: {model.name}")
    print(f"Expected input shape: {model.input_shape}")
    print(f"Input spec: {model.input_spec}")
    
    # Print first few layers to understand the architecture
    print("\nFirst few layers:")
    for i, layer in enumerate(model.layers[:5]):
        print(f"  Layer {i}: {layer.name} - {type(layer).__name__}")
        if hasattr(layer, 'input_shape'):
            print(f"    Input shape: {layer.input_shape}")
        if hasattr(layer, 'output_shape'):
            print(f"    Output shape: {layer.output_shape}")
    
    return model.input_shape


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
        'FEATURES_PRQ_PATH': FEATURES_DIR / "Evaluation_Set" / "features.parquet",
        'FEATURES_SHAPE_JSON_PATH': FEATURES_DIR / "Evaluation_Set" / "features_shape.json",
        'KERAS_MODEL_PATH': MODELS_DIR / 'model.keras',
        'REPORTING_DIR': REPORTING_DIR
    }


def evaluate_model_from_pipeline_paths(config: Config, base_path: Path = None):
    """
    Fixed version of the evaluation function with proper data reshaping
    """
    print("="*60)
    print("LOADING PIPELINE COMPONENTS FOR MODEL EVALUATION (FIXED)")
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
    print(f"Model loaded successfully.")
    
    # Debug model input requirements
    expected_input_shape = debug_model_input_requirements(model)
    expected_shape = expected_input_shape[1:]  # Remove batch dimension
    
    # Load evaluation data with proper reshaping
    evaluation_data = load_evaluation_data_from_parquet_fixed(
        evaluation_data_path, 
        features_shape_json_path,
        expected_shape=expected_shape
    )
    
    # Verify the shapes match
    print(f"\nShape verification:")
    print(f"  Model expects: {expected_input_shape}")
    print(f"  Data shape: {evaluation_data.shape}")
    
    if evaluation_data.shape[1:] != expected_input_shape[1:]:
        print("ERROR: Shape mismatch still exists!")
        print("Please check your data preprocessing pipeline.")
        return None
    
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