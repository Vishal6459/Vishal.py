# Titanic Survival Analysis & Classification

## Overview

This project analyzes the Titanic passenger dataset and builds machine learning models to predict passenger survival.

The notebook covers descriptive statistics, missing-value analysis, exploratory data analysis (EDA), preprocessing, model training, evaluation, and model comparison.

## Files

- `Titanic_Survival_Analysis.ipynb` — Jupyter Notebook containing the complete analysis and classification workflow.
- `Titanic-Dataset(1).csv` — Titanic dataset used for the analysis.

## Dataset

The dataset contains **891 passenger records** and **12 columns**, including:

- PassengerId
- Survived
- Pclass
- Name
- Sex
- Age
- SibSp
- Parch
- Ticket
- Fare
- Cabin
- Embarked

The target variable is **Survived**.

## Analysis Covered

### Exploratory Data Analysis

The notebook examines:

- Survival distribution
- Survival by gender
- Survival by passenger class
- Age distribution
- Fare distribution
- Correlation between numerical variables
- Missing values
- Frequency and percentage distributions

### Data Preprocessing

The notebook includes:

- 80:20 train-test split
- Median imputation for missing numerical values
- Most-frequent imputation for missing categorical values
- One-hot encoding for categorical predictors
- Standardization of numerical predictors
- Fitting preprocessing steps on training data and transforming test data

### Machine Learning Models

Two classification models are implemented and evaluated:

1. **Gaussian Naïve Bayes**
2. **K-Nearest Neighbors (KNN)** with `k = 3`

The notebook includes model evaluation, comparison tables, confusion matrices, classification reports, and metric-based interpretation.

## Tools & Libraries

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## How to Run

1. Open `Titanic_Survival_Analysis.ipynb` in Jupyter Notebook, JupyterLab, Google Colab, or VS Code.
2. Keep `Titanic-Dataset(1).csv` in the same folder as the notebook.
3. Run the notebook cells from top to bottom.

## Project Structure

```text
titanic/
├── README.md
├── Titanic_Survival_Analysis.ipynb
└── Titanic-Dataset(1).csv
```