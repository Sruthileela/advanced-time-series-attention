---

### 3. **train.py** (COMPLETE REWRITE)
```python
"""
Main training script for time series forecasting models.
Supports LSTM, LSTM-Attention, and Transformer architectures.
"""

import argparse
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import optuna
import warnings
warnings.filterwarnings('ignore')

# Custom imports
from src.data_preprocessing import load_and_preprocess, create_sequences
from src.model_lstm import build_lstm_model, build_lstm_attention_model
from src.model_transformer import build_transformer_model
from src.utils import calculate_metrics, plot_predictions, save_model

def parse_args():
    parser = argparse.ArgumentParser(description='Train time series forecasting models')
    parser.add_argument('--model', type=str, default='lstm', 
                       choices=['lstm', 'lstm_attention', 'transformer'],
                       help='Model architecture to train')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lookback', type=int, default=24,
                       help='Lookback window for sequences')
    parser.add_argument('--forecast_horizon', type=int, default=1,
                       help='Forecast horizon')
    parser.add_argument('--tune', action='store_true',
                       help='Enable hyperparameter tuning with Optuna')
    parser.add_argument('--trials', type=int, default=20,
                       help='Number of Optuna trials if tuning is enabled')
    parser.add_argument('--data_path', type=str, default='data/raw/air_quality.csv',
                       help='Path to dataset')
    return parser.parse_args()

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def objective(trial, X_train, y_train, X_val, y_val, model_type):
    """Optuna objective function for hyperparameter tuning"""
    
    if model_type == 'lstm':
        params = {
            'lstm_units': trial.suggest_categorical('lstm_units', [32, 64, 128, 256]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64])
        }
        model = build_lstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            lstm_units=params['lstm_units'],
            dropout_rate=params['dropout_rate']
        )
    
    elif model_type == 'transformer':
        params = {
            'd_model': trial.suggest_categorical('d_model', [32, 64, 128]),
            'num_heads': trial.suggest_int('num_heads', 2, 8),
            'ff_dim': trial.suggest_categorical('ff_dim', [32, 64, 128]),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
        }
        model = build_transformer_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            d_model=params['d_model'],
            num_heads=params['num_heads'],
            ff_dim=params['ff_dim']
        )
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,  # Fixed for tuning
        batch_size=params.get('batch_size', 32),
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Return validation loss
    return history.history['val_loss'][-1]

def main():
    args = parse_args()
    config = load_config()
    
    print(f"Training {args.model.upper()} model...")
    print(f"Configuration: {args.__dict__}")
    
    # 1. Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess(args.data_path)
    
    # 2. Create sequences
    X, y, scaler = create_sequences(
        df, 
        lookback=args.lookback,
        forecast_horizon=args.forecast_horizon
    )
    
    # 3. Train-validation split (temporal)
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {X_train.shape}, Validation samples: {X_val.shape}")
    
    # 4. Hyperparameter tuning or direct training
    if args.tune:
        print(f"Starting hyperparameter tuning with {args.trials} trials...")
        
        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_val, y_val, args.model),
            n_trials=args.trials
        )
        
        print(f"Best trial: {study.best_trial.params}")
        best_params = study.best_trial.params
        
        # Build model with best parameters
        if args.model == 'lstm':
            model = build_lstm_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                lstm_units=best_params['lstm_units'],
                dropout_rate=best_params['dropout_rate']
            )
            optimizer = tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
        
        elif args.model == 'transformer':
            model = build_transformer_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                d_model=best_params['d_model'],
                num_heads=best_params['num_heads'],
                ff_dim=best_params['ff_dim']
            )
            optimizer = tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
    
    else:
        # Use default parameters from config
        if args.model == 'lstm':
            model = build_lstm_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                lstm_units=config['lstm']['units'],
                dropout_rate=config['lstm']['dropout']
            )
        elif args.model == 'lstm_attention':
            model = build_lstm_attention_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                lstm_units=config['lstm_attention']['units']
            )
        elif args.model == 'transformer':
            model = build_transformer_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                d_model=config['transformer']['d_model'],
                num_heads=config['transformer']['num_heads'],
                ff_dim=config['transformer']['ff_dim']
            )
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate'])
    
    # 5. Compile and train model
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f'models/{args.model}_best.h5',
            save_best_only=True,
            monitor='val_loss'
        )
    ]
    
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # 6. Evaluate model
    print("Evaluating model...")
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    # Inverse transform predictions
    train_pred_original = scaler.inverse_transform_y(train_pred)
    val_pred_original = scaler.inverse_transform_y(val_pred)
    y_train_original = scaler.inverse_transform_y(y_train)
    y_val_original = scaler.inverse_transform_y(y_val)
    
    # Calculate metrics
    metrics_train = calculate_metrics(y_train_original, train_pred_original)
    metrics_val = calculate_metrics(y_val_original, val_pred_original)
    
    print(f"\nTraining Metrics: {metrics_train}")
    print(f"Validation Metrics: {metrics_val}")
    
    # 7. Save results
    save_model(model, f'models/{args.model}_final.h5')
    
    # Plot predictions
    plot_predictions(
        y_val_original[:100], 
        val_pred_original[:100],
        title=f'{args.model.upper()} Predictions',
        save_path=f'results/{args.model}_predictions.png'
    )
    
    # Save metrics to results.txt
    with open('results.txt', 'a') as f:
        f.write(f"\n{args.model.upper()} Results:\n")
        f.write(f"Training: {metrics_train}\n")
        f.write(f"Validation: {metrics_val}\n")
    
    print(f"\nTraining completed! Model saved to models/{args.model}_final.h5")

if __name__ == '__main__':
    main()
