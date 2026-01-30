import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import joblib
import os
from utils import save_model

def load_processed_data():
    """Load the processed training and validation data."""
    X_train = np.load('data/processed/X_train.npy')
    X_val = np.load('data/processed/X_val.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_val = np.load('data/processed/y_val.npy')
    return X_train, X_val, y_train, y_val

def train_linear_regression(X_train, y_train):
    """Train a Linear Regression model."""
    print("Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # Optional cross-validation
    scores = cross_val_score(lr, X_train, y_train, cv=5, scoring='r2')
    print(f"Linear Regression CV R2: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    return lr

def train_random_forest(X_train, y_train):
    """Train a Random Forest Regressor."""
    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Optional cross-validation
    scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2')
    print(f"Random Forest CV R2: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    return rf

def main():
    # 1. Load data
    X_train, X_val, y_train, y_val = load_processed_data()
    
    # 2. Train models
    lr_model = train_linear_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    
    # 3. Save models
    save_model(lr_model, 'linear_regression.joblib')
    save_model(rf_model, 'random_forest.joblib')
    
    print("All models trained and saved successfully.")

if __name__ == "__main__":
    main()
