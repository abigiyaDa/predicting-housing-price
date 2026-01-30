import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import joblib

def load_data(file_path):
    """Load the dataset from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def preprocess_data(df):
    """
    Perform full preprocessing: cleaning, imprinting, encoding, and scaling.
    """
    # 1. Drop duplicates
    df = df.drop_duplicates()
    
    # 2. Separate features and target
    target = 'SalePrice'
    X = df.drop(columns=[target, 'Order', 'PID']) # Drop identifiers
    y = df[target]
    
    # 3. Identify numerical and categorical features
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numerical features: {len(numerical_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    
    # 4. Create preprocessing pipelines
    # For numerical: Impute with median, then scale
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # For categorical: Impute with mode (most_frequent), then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine both using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return X, y, preprocessor

def main():
    # File paths
    data_path = 'data/ames_housing.csv'
    
    # Load and preprocess
    df = load_data(data_path)
    X, y, preprocessor = preprocess_data(df)
    
    # 5. Split data: 70% train, 15% validation, 15% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1765, random_state=42) # 0.15 / 0.85 = ~0.1765
    
    print(f"Train set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # 6. Fit preprocessor on training data only to avoid data leakage
    preprocessor.fit(X_train)
    
    # Save the preprocessor for later use in the app/inference
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    print("Preprocessor saved to models/preprocessor.joblib")
    
    # Transform data
    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save processed data for training and evaluation
    os.makedirs('data/processed', exist_ok=True)
    np.save('data/processed/X_train.npy', X_train_processed)
    np.save('data/processed/X_val.npy', X_val_processed)
    np.save('data/processed/X_test.npy', X_test_processed)
    np.save('data/processed/y_train.npy', y_train.values)
    np.save('data/processed/y_val.npy', y_val.values)
    np.save('data/processed/y_test.npy', y_test.values)
    
    print("Processed data saved to data/processed/")

if __name__ == "__main__":
    main()
