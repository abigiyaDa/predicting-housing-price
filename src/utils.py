import joblib
import os

def save_model(model, filename):
    """Save the trained model to a file."""
    os.makedirs('models', exist_ok=True)
    filepath = os.path.join('models', filename)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filename):
    """Load a trained model from a file."""
    filepath = os.path.join('models', filename)
    if os.path.exists(filepath):
        return joblib.load(filepath)
    else:
        raise FileNotFoundError(f"No model found at {filepath}")

def save_preprocessor(preprocessor, filename):
    """Save the preprocessor object."""
    os.makedirs('models', exist_ok=True)
    filepath = os.path.join('models', filename)
    joblib.dump(preprocessor, filepath)
    print(f"Preprocessor saved to {filepath}")
