"""
Model inference and explainability using SHAP.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import argparse
import warnings
warnings.filterwarnings('ignore')

from src.data_preprocessing import load_and_preprocess, create_sequences
from src.model_lstm import build_lstm_model
from src.model_transformer import build_transformer_model
from src.utils import load_model, calculate_metrics

def load_trained_model(model_path, model_type='lstm', input_shape=None):
    """Load a trained model based on type."""
    if model_type == 'lstm':
        model = build_lstm_model(input_shape=input_shape)
    elif model_type == 'transformer':
        model = build_transformer_model(input_shape=input_shape)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_weights(model_path)
    return model

def generate_shap_explanations(model, X_sample, feature_names, model_type='lstm'):
    """
    Generate SHAP explanations for model predictions.
    
    Args:
        model: Trained model
        X_sample: Input samples (n_samples, lookback, n_features)
        feature_names: List of feature names
        model_type: Type of model ('lstm' or 'transformer')
    
    Returns:
        shap_values: SHAP values
    """
    
    # Create a background dataset (can be subsampled)
    background = X_sample[np.random.choice(X_sample.shape[0], 50, replace=False)]
    
    if model_type == 'lstm':
        # Use DeepExplainer for LSTM
        explainer = shap.DeepExplainer(model, background)
    else:
        # Use KernelExplainer for Transformer (more stable)
        def model_predict(X):
            return model.predict(X).flatten()
        
        explainer = shap.KernelExplainer(
            model_predict, 
            background.reshape(background.shape[0], -1)
        )
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_sample)
    
    return shap_values, explainer

def plot_shap_summary(shap_values, X_sample, feature_names, save_path=None):
    """Plot SHAP summary plot."""
    plt.figure(figsize=(12, 8))
    
    # For time series, we might want to reshape for better visualization
    shap_sum = np.abs(shap_values).mean(axis=(0, 1))  # Average over samples and time
    
    # Create bar plot of feature importance
    plt.subplot(2, 2, 1)
    plt.barh(feature_names, shap_sum)
    plt.xlabel('Mean |SHAP value|')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    
    # SHAP summary plot (beeswarm)
    plt.subplot(2, 2, 2)
    shap.summary_plot(
        shap_values.reshape(-1, shap_values.shape[-1]), 
        X_sample.reshape(-1, X_sample.shape[-1]),
        feature_names=feature_names,
        show=False,
        max_display=10
    )
    
    # Time-step specific importance (heatmap)
    plt.subplot(2, 2, 3)
    shap_time_avg = np.abs(shap_values).mean(axis=0)  # Average over samples
    sns.heatmap(shap_time_avg.T, 
                yticklabels=feature_names,
                xticklabels=range(shap_time_avg.shape[0]),
                cmap='viridis')
    plt.xlabel('Time Step (lookback)')
    plt.ylabel('Feature')
    plt.title('SHAP Importance Over Time')
    
    # Force plot for a single sample
    plt.subplot(2, 2, 4)
    sample_idx = 0
    shap.force_plot(
        explainer.expected_value, 
        shap_values[sample_idx].mean(axis=0), 
        X_sample[sample_idx].mean(axis=0),
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def explain_predictions(model_path, data_path, model_type='lstm', 
                       lookback=24, n_samples=100):
    """Main function to explain model predictions."""
    
    print(f"Loading model: {model_path}")
    
    # Load and preprocess data
    df = load_and_preprocess(data_path)
    X, y, scaler = create_sequences(df, lookback=lookback)
    
    # Use a subset for explanation
    X_sample = X[:n_samples]
    
    # Load model
    model = load_trained_model(
        model_path, 
        model_type=model_type,
        input_shape=(X.shape[1], X.shape[2])
    )
    
    # Get feature names
    feature_names = df.columns.tolist()
    
    print("Generating SHAP explanations...")
    shap_values, explainer = generate_shap_explanations(
        model, X_sample, feature_names, model_type
    )
    
    # Plot explanations
    plot_shap_summary(
        shap_values, 
        X_sample, 
        feature_names,
        save_path=f'results/shap_{model_type}.png'
    )
    
    # Save SHAP values for further analysis
    np.save(f'results/shap_values_{model_type}.npy', shap_values)
    
    print(f"SHAP analysis completed! Results saved to results/shap_{model_type}.png")
    
    return shap_values

def batch_predict(model_path, data_path, model_type='lstm', 
                 lookback=24, output_path='results/predictions.csv'):
    """Generate batch predictions on new data."""
    
    print("Generating batch predictions...")
    
    # Load and preprocess data
    df = load_and_preprocess(data_path)
    X, y, scaler = create_sequences(df, lookback=lookback)
    
    # Load model
    model = load_trained_model(
        model_path, 
        model_type=model_type,
        input_shape=(X.shape[1], X.shape[2])
    )
    
    # Generate predictions
    predictions = model.predict(X)
    predictions_original = scaler.inverse_transform_y(predictions)
    y_original = scaler.inverse_transform_y(y)
    
    # Calculate metrics
    metrics = calculate_metrics(y_original, predictions_original)
    
    # Save predictions
    results_df = pd.DataFrame({
        'actual': y_original.flatten(),
        'predicted': predictions_original.flatten(),
        'error': (y_original.flatten() - predictions_original.flatten())
    })
    results_df.to_csv(output_path, index=False)
    
    print(f"Predictions saved to {output_path}")
    print(f"Metrics: {metrics}")
    
    return results_df, metrics

def main():
    parser = argparse.ArgumentParser(description='Model inference and explainability')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--model_type', type=str, default='lstm',
                       choices=['lstm', 'transformer'],
                       help='Type of model')
    parser.add_argument('--data', type=str, default='data/raw/air_quality.csv',
                       help='Path to data')
    parser.add_argument('--predict', action='store_true',
                       help='Generate predictions')
    parser.add_argument('--explain', action='store_true',
                       help='Generate SHAP explanations')
    parser.add_argument('--lookback', type=int, default=24,
                       help='Lookback window')
    
    args = parser.parse_args()
    
    if args.predict:
        batch_predict(
            args.model,
            args.data,
            model_type=args.model_type,
            lookback=args.lookback
        )
    
    if args.explain:
        explain_predictions(
            args.model,
            args.data,
            model_type=args.model_type,
            lookback=args.lookback
        )
    
    if not args.predict and not args.explain:
        print("Please specify --predict or --explain flag")

if __name__ == '__main__':
    main()
