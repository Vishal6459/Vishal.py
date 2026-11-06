# H1N1 Vaccine Prediction 💉📊

A Machine Learning project focused on predicting whether individuals received the H1N1 flu vaccine based on their background, opinions, and health behaviors. This project demonstrates end-to-end data processing, exploratory data analysis, and the implementation of multiple classification models.

## 🚀 Project Overview & Features

* **Exploratory Data Analysis (EDA):** Comprehensive visualization of data distributions and feature relationships using Seaborn (Heatmaps, Countplots, Boxplots, Violinplots, and Jointplots).
* **Data Preprocessing:** Handled missing values, treated outliers using the Interquartile Range (IQR) method, and performed one-hot encoding on categorical variables (e.g., race, sex, income level).
* **Predictive Modeling:** Built and evaluated multiple classification models to determine the most effective algorithm for predicting vaccination status.
* **Model Evaluation:** Utilized Accuracy Scores, Classification Reports, and Confusion Matrices to assess model performance.

## 💻 Tech Stack

* **Language:** Python 3
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (scikit-learn)

## 🧠 Models Evaluated

1. **Logistic Regression:** A baseline linear classification model.
2. **Decision Tree Classifier:** A non-linear model capturing complex decision boundaries.
3. **Bagging Classifier:** An ensemble learning method (using Decision Trees as base estimators) to reduce variance and prevent overfitting.

## 📋 Prerequisites

To run this notebook locally, ensure you have Python installed along with the following libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
