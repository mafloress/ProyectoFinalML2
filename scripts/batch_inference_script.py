import pandas as pd
import numpy as np
import pickle
import os

# Define file paths relative to the script's location in the Docker container
MODEL_PATH = '/app/trained_model.pkl'
PROCESSED_DATA_PATH = '/app/processed_lending_club_data.csv'
OUTPUT_PREDICTIONS_PATH = '/app/batch_predictions.csv' # Output will also be in /app

def load_model(path):
    """Loads a pickled model from the specified path."""
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from: {path}")
        print(f"Model type: {type(model)}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {path}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_data(path):
    """Loads processed data from a CSV file."""
    try:
        df = pd.read_csv(path)
        print(f"Data loaded successfully from: {path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def prepare_batch_data(df_full, batch_size=1000):
    """Prepares a batch of data for inference."""
    if df_full is None:
        return None

    X_full = df_full.copy()
    if 'is_default' in X_full.columns:
        X_full = X_full.drop('is_default', axis=1)
        print("Target column 'is_default' dropped to create feature set.")
    
    if len(X_full) >= batch_size:
        X_batch = X_full.head(batch_size).copy()
        print(f"Batch data created with the first {batch_size} rows. Shape: {X_batch.shape}")
    else:
        X_batch = X_full.copy()
        print(f"Full dataset has less than {batch_size} rows. Using all rows for batch. Shape: {X_batch.shape}")
    
    return X_batch

def generate_predictions(model, X_batch):
    """Generates class predictions and probabilities."""
    if model is None or X_batch is None or X_batch.empty:
        print("Cannot generate predictions: model not loaded or batch data is empty.")
        return None, None
    
    try:
        print("Generating predictions...")
        # Ensure columns are in the same order as during training if model stores feature names
        if hasattr(model, 'feature_name_') and list(X_batch.columns) != list(model.feature_name_):
            print("Warning: Batch data columns are not in the same order as model's training data. Reordering...")
            X_batch = X_batch[model.feature_name_]
        elif hasattr(model, 'n_features_in_') and X_batch.shape[1] != model.n_features_in_:
             print(f"Error: Model expects {model.n_features_in_} features, but batch data has {X_batch.shape[1]} features.")
             return None, None


        predictions = model.predict(X_batch)
        probabilities = model.predict_proba(X_batch)[:, 1] # Probability of class '1' (default)
        print("Predictions generated successfully.")
        return predictions, probabilities
    except Exception as e:
        print(f"Error during prediction generation: {e}")
        return None, None

def save_predictions(X_batch_index, predictions, probabilities, path):
    """Saves predictions to a CSV file."""
    if predictions is None or probabilities is None:
        print("No predictions to save.")
        return

    df_predictions = pd.DataFrame({
        'record_id': X_batch_index,
        'prediction_default': predictions,
        'probability_default': probabilities
    })
    
    try:
        df_predictions.to_csv(path, index=False)
        print(f"Predictions saved successfully to: {path}")
        print("First 5 predictions saved:")
        print(df_predictions.head().to_string())
    except Exception as e:
        print(f"Error saving predictions: {e}")

def main():
    """Main function to run the batch inference pipeline."""
    print("Starting batch inference process...")

    # 1. Load Model
    model = load_model(MODEL_PATH)
    if model is None:
        print("Exiting due to model loading failure.")
        return

    # 2. Load Data
    df_full_processed = load_data(PROCESSED_DATA_PATH)
    if df_full_processed is None:
        print("Exiting due to data loading failure.")
        return

    # 3. Prepare Batch Data
    # Using a smaller batch size for quick testing if needed, e.g., 100
    # For the task, it's 1000.
    X_batch = prepare_batch_data(df_full_processed, batch_size=1000) 
    if X_batch is None or X_batch.empty:
        print("Exiting due to batch data preparation failure.")
        return

    # 4. Generate Predictions
    predictions, probabilities = generate_predictions(model, X_batch)
    if predictions is None:
        print("Exiting due to prediction generation failure.")
        return
        
    # 5. Save Predictions
    save_predictions(X_batch.index, predictions, probabilities, OUTPUT_PREDICTIONS_PATH)
    
    print("Batch inference process completed.")

if __name__ == '__main__':
    # Ensure necessary libraries for the model are implicitly available
    # (e.g. lightgbm if the model is LGBMClassifier)
    # These should be in requirements.txt for the Docker image.
    try:
        import lightgbm 
    except ImportError:
        pass # If not LightGBM, it might be another model type

    try:
        from sklearn.ensemble import RandomForestClassifier # Example if RF was a fallback
    except ImportError:
        pass
        
    main()
