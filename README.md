# Predicting Housing Prices using the Ames Housing Dataset

## 1. Problem Definition & Motivation
This project aims to predict the sale prices of residential properties in Ames, Iowa, using various feature descriptors. This is a classic **Supervised Learning Regression** problem. 

**Motivation:** Accurate house price prediction is beneficial for:
* **Homeowners:** To understand the market value of their property.
* **Buyers:** To make informed decisions and avoid overpaying.
* **Real Estate Agents:** To provide better guidance to clients.
* **Investors:** To identify undervalued properties.

## 2. Dataset Description
* **Dataset:** Ames Housing Dataset
* **Source:** [Kaggle](https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset)
* **File Location:** [data/ames_housing.csv](data/ames_housing.csv)
* **Target Variable:** `SalePrice`
* **Features:** Includes 80+ descriptors of residential homes, such as living area size, quality ratings, neighborhood, and basement details.

## 3. Data Preprocessing
The preprocessing steps are implemented in [src/data_preprocessing.py](src/data_preprocessing.py):
* **Handling Missing Values:**
    * Numerical features were imputed with the **median**.
    * Categorical features were imputed with the **mode** (most frequent value).
* **Feature Cleaning:** Dropped identifier columns like `Order` and `PID`.
* **Encoding:** Categorical features were encoded using **One-Hot Encoding**.
* **Scaling:** All features were scaled using **StandardScaler** to ensure uniform contribution to the model.
* **Splitting:** Data was split into:
    * **Training Set:** 70%
    * **Validation Set:** 15%
    * **Test Set:** 15%

## 4. Exploratory Data Analysis (EDA)
A detailed exploratory analysis can be found in [notebooks/eda.ipynb](notebooks/eda.ipynb). Highlights include:
* Analysis of the distribution of `SalePrice`.
* Correlation heatmap between numerical features and price.
* Visualizing key drivers of price such as `Overall Qual` and `Gr Liv Area`.

## 5. Model Selection
Two regression models were chosen:
1. **Linear Regression (Baseline):** Chosen for its simplicity and interpretability. It serves as a good benchmark for more complex models.
2. **Random Forest Regressor:** Chosen because it can capture non-linear relationships and complex interactions between features.

## 6. Model Training
The training logic is in [src/train.py](src/train.py). Models were trained with 5-fold cross-validation, ensuring the results are robust and not overfitted to a specific split.

## 7. Model Evaluation
Evaluated in [src/evaluate.py](src/evaluate.py) using the following metrics:
* **RMSE (Root Mean Squared Error):** Measures the average error in prediction (lower is better).
* **R² Score:** Measures how much variance in price is explained by the model (higher is better).

**Test Results:**
| Model | RMSE | R² Score |
|-------|------|----------|
| Linear Regression | 29239.10 | 0.8887 |
| Random Forest | 29064.27 | 0.8900 |

## 8. Results & Analysis
The **Random Forest** model performed slightly better than the Linear Regression model, capturing about 89% of the variation in housing prices. The results suggest that features like overall house quality and living area size are powerful predictors of sale price.

## 9. Deployment
A simple interactive web app built with **Streamlit** allows users to input house details and get an instant price prediction.
* **Launch:** `streamlit run [app.py](app.py)`

## 10. How to Run the Project
1. **Clone the repository.**
2. **Install dependencies:**
   ```bash
   pip install -r [requirements.txt](requirements.txt)
   ```
3. **Run Preprocessing:**
   ```bash
   python [src/data_preprocessing.py](src/data_preprocessing.py)
   ```
4. **Train Models:**
   ```bash
   python [src/train.py](src/train.py)
   ```
5. **Evaluate Results:**
   ```bash
   python [src/evaluate.py](src/evaluate.py)
   ```
6. **Start Streamlit App:**
   ```bash
   streamlit run [app.py](app.py)
   ```

## 11. Code Quality & Reproducibility
* **Modular Code:** Logic is separated into preprocessing, training, and evaluation scripts.
* **Reproducibility:** A fixed random seed (`42`) is used throughout the project to ensure results can be replicated exactly.
* **Documentation:** All functions are documented with docstrings and comments.

---
*Developed for the Final Course Project - January 2026*
