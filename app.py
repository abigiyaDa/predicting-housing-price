import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page title
st.set_page_config(page_title="Ames Housing Price Predictor", layout="centered")

# --- Helper Functions ---
@st.cache_resource
def load_resources():
    # Get the directory where app.py is located
    base_path = os.path.dirname(__file__)

    # Construct absolute paths relative to app.py
    model_path = os.path.join(base_path, 'models', 'random_forest.joblib')
    preproc_path = os.path.join(base_path, 'models', 'preprocessor.joblib')
    data_path = os.path.join(base_path, 'data', 'ames_housing.csv')

    # Check if files exist to provide better error messages
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file missing on server: {model_path}")
    if not os.path.exists(preproc_path):
        raise FileNotFoundError(f"Preprocessor file missing on server: {preproc_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file missing on server: {data_path}")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preproc_path)
    # Load a sample row to use as a template for features
    df_sample = pd.read_csv(data_path).drop(columns=['SalePrice', 'Order', 'PID']).iloc[0:1]
    return model, preprocessor, df_sample

# Load model, preprocessor, and template
try:
    model, preprocessor, df_template = load_resources()
except Exception as e:
    st.error(f"Error loading models or data. Please ensure you've run the training script. Error: {e}")
    st.stop()

# --- App UI ---
st.title("🏡 Ames Housing Price Predictor")
st.write("""
This app predicts the **Sale Price** of a house in Ames, Iowa, based on its characteristics.
Using a Random Forest Regressor model trained on the Ames Housing Dataset.
""")

st.subheader("Enter House Details")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 6)
    gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", 500, 10000, 1500)
    year_built = st.number_input("Year Built", 1870, 2010, 1970)
    total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", 0, 6000, 1000)

with col2:
    neighborhood = st.selectbox("Neighborhood", [
        'Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr', 'Crawfor',
        'Edwards', 'Gilbert', 'Greens', 'GrnHill', 'IDOTRR', 'Landmrk', 'MeadowV',
        'Mitchel', 'NAmes', 'NPkVill', 'NWAmes', 'NoRidge', 'NridgHt', 'OldTown',
        'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker'
    ], index=15) # Default to NAmes
    garage_cars = st.slider("Garage Capacity (Cars)", 0, 5, 2)
    full_bath = st.slider("Full Bathrooms", 0, 4, 2)
    fireplaces = st.slider("Number of Fireplaces", 0, 4, 1)

# --- Prediction Logic ---
if st.button("Predict Price"):
    # 1. Prepare input dataframe
    # Start with our template
    input_df = df_template.copy()

    # 2. Update with user inputs
    input_df['Overall Qual'] = overall_qual
    input_df['Gr Liv Area'] = gr_liv_area
    input_df['Year Built'] = year_built
    input_df['Total Bsmt SF'] = total_bsmt_sf
    input_df['Neighborhood'] = neighborhood
    input_df['Garage Cars'] = garage_cars
    input_df['Full Bath'] = full_bath
    input_df['Fireplaces'] = fireplaces

  
