# car-purchase-amount-prediction-ann
car-purchase-amount-prediction/
│── data/
│     └── car_purchasing.csv
│── notebooks/
│     └── Car_Purchase_Prediction.ipynb
│── src/
│     ├── preprocess.py
│     └── linear_regression_model.py
│── results/
│     └── model_metrics.txt    
│── requirements.txt
│── README.md


##Project Overview

This is a pure Linear Regression project, focusing on:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature selection
- Building a Linear Regression model
- Evaluating performance (MAPE, R²)
This project demonstrates strong fundamentals in statistics, ML workflow, and predictive modelling.

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


##Model Implemented
Linear Regression

- The dataset has numeric features
- Relationship between features & target is nearly linear
- Very easy to interpret
- Works well for baseline regression problems

 ##Model Performance

Model	Metric	Score-
MAPE:	~13%
R² Score: ~0.617

##Results Summary

- Linear Regression performed well on the dataset
- Model captured the relationship between income/net worth and spending
- Outlier removal significantly improved model accuracy
- The model is simple, interpretable, and effective
