# House Price Prediction

The project focuses on predicting residential house prices using machine learning regression techniques.

## Overview
The goal of this project is to provide a data-driven model for estimating property values based on historical transaction data from the Kaggle "House Prices: Advanced Regression Techniques" dataset. The pipeline emphasizes robust methodology, particularly preventing data leakage by strictly separating train, validation, and test sets before any data cleaning or feature engineering.

## Models Evaluated
1. **Linear Regression (Baseline):** Trained on 26 stable numerical features after removing exact linear combinations and highly collinear pairs to ensure numerical stability.
2. **Ridge Regression:** Trained on all 263 one-hot encoded features, using L2 regularization to handle high dimensionality.
3. **Lasso Regression:** Trained on all 263 features, using L1 regularization for both prediction and inherent feature selection.

## Key Results
The regularized models successfully leveraged categorical features (like neighborhood and exterior material) to outperform the numerical-only baseline without overfitting.

- **Baseline Linear Regression:** $R^2 = 0.801$ (RMSE: \$40,158)
- **Ridge Regression ($\alpha=10$):** $R^2 = 0.860$ (RMSE: \$33,714)
- **Lasso Regression ($\alpha=100$):** $R^2 = 0.865$ (RMSE: \$33,050)

Overall quality and above-ground living area were identified as the strongest predictors of sale price.

## Files
- `House_Prices_Prediction.ipynb`: The complete Jupyter Notebook containing data preparation, exploratory data analysis (EDA), model training, hyperparameter tuning, and evaluation.
- `house_prices.csv`: The dataset used for the analysis.

## Getting Started
To run the notebook locally:
1. Ensure you have Python 3 installed.
2. Install the required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter kaggle
   ```
3. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook House_Prices_Prediction.ipynb
   ```
