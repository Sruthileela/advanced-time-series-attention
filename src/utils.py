import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    # Directional accuracy
    y_true_dir = np.diff(y_true.flatten()) > 0
    y_pred_dir = np.diff(y_pred.flatten()) > 0
    directional_acc = np.mean(y_true_dir == y_pred_dir)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'Directional_Accuracy': directional_acc
    }

def plot_predictions(y_true, y_pred, title='Predictions vs Actual', save_path=None):
    """Plot predictions vs actual values."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', alpha=0.7)
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.fill_between(range(len(y_true)), y_true.flatten(), y_pred.flatten(), 
                     alpha=0.2, color='gray')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_model(model, path):
    """Save model weights."""
    model.save_weights(path)
    print(f"Model saved to {path}")

def load_model(path, model_builder, **kwargs):
    """Load model from weights."""
    model = model_builder(**kwargs)
    model.load_weights(path)
    return model
