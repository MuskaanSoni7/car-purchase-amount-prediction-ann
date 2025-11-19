# car-purchase-amount-prediction-ann
car-purchase-amount-prediction-ann/
│── README.md
│── data/
│     └── car_purchasing.csv
│── notebooks/
│     └── Car_Purchase_Prediction.ipynb
│── src/
│     ├── preprocess.py
│     ├── linear_regression_model.py
│     ├── ann_model.py
│── results/
│     ├── EDA_plots.png
│     ├── model_comparison.csv
│── requirements.txt

##Project Overview

This project aims to predict the car purchase amount a customer is likely to spend using demographic and financial attributes.
The pipeline includes:

- Data cleaning
- Outlier removal (IQR method)
- Exploratory Data Analysis (EDA)
- Feature engineering
- Building regression models
- Building an Artificial Neural Network (ANN) using TensorFlow
- Comparing model performances

##Dataset Description

The dataset contains 500 rows and 9 features, obtained from Kaggle.
The target variable is:
-car purchase amount

The features include:
- Customer name
- Customer email
- Country
- Gender (1 = Male, 0 = Female)
- Age
- Annual Salary
- Credit Card Debt
- Net Worth
To focus on numerical ML modeling, irrelevant character-based columns were removed.

##Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Statsmodels
- Scikit-learn
- TensorFlow / Keras

##Models Implemented

1️⃣ Linear Regression (OLS)

Built using Statsmodels
Analyzed coefficients, p-values, and R²
Checked multicollinearity with VIF
Baseline model for comparison

2️⃣ Ridge Regression

Applied L2 regularization
Handled potential overfitting
Improved model stability

3️⃣ Artificial Neural Network (ANN)

A 2-layer deep learning model built using TensorFlow/Keras:
Layer 1: Dense(6 neurons, relu)
Layer 2: Dense(6 neurons, relu)
Output: Dense(1 neuron)
Loss: MSE
Optimizer: Adam
Epochs: 100
Batch Size: 32

 ##Model Performance

Model	Metric	Score
Linear Regression	MAPE	~X%
Ridge Regression	MAPE	~X%
ANN (TensorFlow)	MAPE	0.87%

🔹 ANN achieved the best performance, capturing non-linear patterns more effectively than OLS or Ridge.

## Key Insights

- Net worth and age were strong predictors of purchase amount
- Outlier removal significantly improved model accuracy
- ANN captured deeper relationships that linear models couldn’t
- Minimal overfitting due to effective scaling and architecture
