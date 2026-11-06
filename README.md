# Chronic Kidney Disease (CKD) Decision Support System

A machine-learning mini-project that develops and deploys a **Chronic Kidney Disease (CKD) screening and decision-support application**. The project is organized into two machine-learning phases and a Streamlit deployment application.

The workflow starts with dataset exploration and preprocessing, proceeds to training and evaluating six machine-learning models, and finishes with a web application that accepts individual patient inputs or batch CSV files.

> **Medical disclaimer:** This is an academic screening prototype, not a medical diagnostic device. Its output is a model-generated estimate and must not be treated as a confirmed diagnosis or used as the sole basis for treatment. Clinical assessment and confirmatory testing by qualified healthcare professionals are essential.

## Project structure

```text
CKD-Decision-Support-System/
│
├── app.py
├── ckd_best_model.joblib
├── requirements.txt
├── README.md
├── .gitignore
│
└── notebooks/
    ├── Phase_1.ipynb
    └── Phase_2.ipynb
```

| File | Description |
|---|---|
| `app.py` | Streamlit application for individual and batch CKD screening |
| `ckd_best_model.joblib` | Saved trained model pipeline, including preprocessing and the selected classifier |
| `requirements.txt` | Python dependencies required to run the project |
| `notebooks/Phase_1.ipynb` | Dataset exploration and preprocessing |
| `notebooks/Phase_2.ipynb` | Model training, evaluation and comparison |
| `.gitignore` | Excludes local environments, IDE files and sensitive configuration |

## Dataset overview

The project uses a tabular CKD dataset containing demographic, lifestyle, medical-history, clinical and laboratory-related attributes.

- **Target column:** `Diagnosis`
- **Target encoding:** `1` represents CKD and `0` represents non-CKD.
- **Predictors:** 51 input features are used by the application.
- Identifier fields such as `PatientID` and `DoctorInCharge` are not used as model predictors.

The notebooks contain the detailed dataset analysis and preparation workflow. Refer to the notebooks for the exact operations and experimental results.

## Phase 1 — Data preprocessing

**Notebook:** `notebooks/Phase_1.ipynb`

The first phase focuses on preparing the raw dataset for machine-learning experiments. It covers the data preparation and preprocessing workflow, including:

- Loading and inspecting the dataset.
- Understanding the dataset structure, feature types and target variable.
- Exploring data quality and missing values.
- Separating numerical, categorical and binary features for appropriate preprocessing.
- Preparing the feature matrix and target variable for model development.
- Building preprocessing transformations for the different feature types.

The preprocessing workflow uses a `ColumnTransformer` with:
- `StandardScaler` for numerical features.
- `OneHotEncoder` for categorical features, configured to handle previously unseen categories.
- Passthrough handling for binary features.

The fitted preprocessing steps are incorporated into the model pipeline used by the application, so new inputs receive the same transformations as the training data.

## Phase 2 — Model development and evaluation

**Notebook:** `notebooks/Phase_2.ipynb`

The second phase focuses on developing, training and comparing six machine-learning models. The models are grouped into four individual (non-ensemble) learning algorithms and two ensemble learning algorithms.

### Individual machine-learning models

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Multi-Layer Perceptron (MLP)
4. Gaussian Naive Bayes

### Ensemble machine-learning models

5. Random Forest
6. Gradient Boosting

The models are evaluated and compared using classification performance metrics. The comparison considers metrics such as accuracy, precision, recall, specificity, F1-score and ROC-AUC, as implemented in the notebook.

The project also considers the class distribution and the effect of the decision threshold when interpreting predictions. The selected model and its fitted preprocessing pipeline are saved as `ckd_best_model.joblib` for use in the Streamlit application.

**Note:** The exact experimental setup, metric values, plots and model-selection rationale should be read from the Phase 2 notebook. The README intentionally does not reproduce numerical results that may change if the notebook is rerun.

## Streamlit application

The deployed application provides two ways to submit patient data.

### 1. Individual patient screening

Users can enter clinical, biochemical, demographic and lifestyle-related parameters through the Streamlit form. The application passes the input through the saved pipeline and displays the estimated CKD probability and threshold-based screening result.

### 2. Batch CSV screening

Users can upload a CSV file containing multiple patient records. The application generates a prediction and estimated probability for each row.

The batch input must contain the 51 predictor columns expected by the trained model. The current application removes `PatientID`, `DoctorInCharge` and `Diagnosis` columns if they are present before prediction.

### Adjustable decision threshold

A probability threshold can be adjusted in the sidebar. The application classifies a record as CKD-positive when its estimated probability is greater than or equal to the selected threshold. Changing the threshold changes the classification decision; it does not retrain the model.

## Run the project locally

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd CKD-Decision-Support-System
```

### 2. Create a virtual environment (recommended)

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

1. Push the project to a GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
3. Select **Create app**.
4. Choose the repository and deployment branch.
5. Set the main file path to `app.py`.
6. Deploy and wait for dependency installation.
7. Open the generated app URL and test both individual and batch prediction workflows.

Keep `app.py`, `requirements.txt` and `ckd_best_model.joblib` in the repository root unless the paths in the application are changed.

## Reproducibility and model compatibility

The saved `joblib` model should be loaded with compatible versions of Python and its machine-learning dependencies, especially `scikit-learn`. Before deployment, check the versions used in the training notebook and pin the matching versions in `requirements.txt`. If the environment differs, test the saved model in the deployment environment before publishing the app.

## Repository and data privacy

- Do not commit passwords, API keys, tokens or `.env` files.
- Do not publish identifiable or confidential patient information.
- Check the dataset's license and sharing conditions before including the raw dataset in a public repository.
- Include the model artifact only if its source data and project terms permit redistribution.
- Use synthetic or properly de-identified examples for public demonstrations.

## Technologies used

- Python
- Pandas and NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib and Seaborn (for notebook visualizations)
- Jupyter Notebook

## Academic project note

This repository contains the two notebooks documenting the machine-learning workflow and the Streamlit application demonstrating the selected trained pipeline. It is intended for academic learning, experimentation and demonstration.
